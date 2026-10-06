use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, ValueEnum};

use desloppify::agent;
use desloppify::model::Provider;
use desloppify::rate_limit::{self, AdaptiveConfig, RateLimiter, SystemClock, TokenBucketConfig};
use desloppify::review::{Call, plan, review};
use desloppify::rule::{self, Rule};
use desloppify::skills;

const SECONDS_PER_MINUTE: f64 = 60.0;
/// The adaptive limiter never cuts below one call a minute.
const FLOOR_RPM: f64 = 1.0;
/// Calls per minute the adaptive limiter gains per minute unthrottled: a
/// halved 50 rpm is back in two and a half minutes.
const GROWTH_RPM_PER_MINUTE: f64 = 10.0;
/// Providers count requests per minute, so a minute after a cut every reply
/// to a call made at the old rate is in.
const COOLDOWN: Duration = Duration::from_secs(60);

#[derive(Clone, Copy, ValueEnum)]
enum Limiter {
    /// Find the provider's limit: slow down on 429s, speed up without them.
    Adaptive,
    /// A fixed budget: `--burst` calls, one more every `--refill-ms`.
    TokenBucket,
}

/// Review code against the rules in `rules/`.
#[derive(Parser)]
struct Args {
    /// Files or directories to review.
    #[arg(required = true)]
    paths: Vec<PathBuf>,
    #[arg(long, value_enum, default_value_t = Provider::Anthropic)]
    provider: Provider,
    #[arg(long, value_enum, default_value_t = Limiter::Adaptive)]
    limiter: Limiter,
    /// Adaptive: calls per minute to start at.
    #[arg(long, default_value_t = 50.0)]
    rpm: f64,
    /// Adaptive: calls per minute never to exceed.
    #[arg(long, default_value_t = 1000.0)]
    max_rpm: f64,
    /// Token bucket: calls that may go out back to back.
    #[arg(long, default_value_t = 10)]
    burst: u64,
    /// Token bucket: milliseconds to earn back one call.
    #[arg(long, default_value_t = 1200)]
    refill_ms: u64,
    /// Give up on a call rather than wait longer than this for it.
    #[arg(long, default_value_t = 300)]
    max_wait_secs: u64,
    /// Most calls in flight at once, retries included.
    #[arg(long, default_value = "8")]
    jobs: NonZeroUsize,
    /// Count the calls each rule would make, and make none.
    #[arg(long)]
    dry_run: bool,
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/rules"))]
    rules: PathBuf,
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/skills"))]
    skills: PathBuf,
}

#[tokio::main]
async fn main() -> Result<ExitCode> {
    let args = Args::parse();
    let skills = skills::load(&args.skills)?;
    let rules = Arc::new(rule::load_dir(&args.rules, &skills)?);

    let mut files = Vec::new();
    for path in &args.paths {
        collect(path, &mut files)?;
    }
    let plan = plan(&rules, &files)?;

    if args.dry_run {
        print_plan(&rules, &plan);
        return Ok(ExitCode::SUCCESS);
    }

    let agent = Arc::new(agent::from_env(args.provider, limiter(&args), args.jobs)?);
    let report = review(agent, rules, plan).await?;
    for f in &report.findings {
        println!(
            "{}:{}: [{}] {}",
            f.path.display(),
            f.line,
            f.rule,
            f.message
        );
    }
    for failure in &report.failures {
        eprintln!("error: {failure:#}");
    }
    let clean = report.findings.is_empty() && report.failures.is_empty();
    Ok(if clean {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

/// Calls per rule, with each rule's level, and the total.
fn print_plan(rules: &[Rule], plan: &[Call]) {
    let mut counts = vec![0_u64; rules.len()];
    for call in plan {
        counts[call.rule] += 1;
    }
    for (rule, count) in rules.iter().zip(&counts) {
        println!("{count:>8}  level {}  {}", u64::from(rule.level), rule.id);
    }
    println!("{:>8}  total", plan.len());
}

fn limiter(args: &Args) -> Box<dyn RateLimiter> {
    let max_wait = Duration::from_secs(args.max_wait_secs);
    match args.limiter {
        Limiter::TokenBucket => Box::new(rate_limit::token_bucket(
            TokenBucketConfig {
                capacity: args.burst,
                refill: Duration::from_millis(args.refill_ms),
                max_wait,
            },
            SystemClock,
        )),
        Limiter::Adaptive => Box::new(rate_limit::adaptive(
            AdaptiveConfig {
                floor: FLOOR_RPM / SECONDS_PER_MINUTE,
                start: args.rpm / SECONDS_PER_MINUTE,
                ceiling: args.max_rpm / SECONDS_PER_MINUTE,
                increase: GROWTH_RPM_PER_MINUTE / (SECONDS_PER_MINUTE * SECONDS_PER_MINUTE),
                cooldown: COOLDOWN,
                max_wait,
            },
            agent::classify,
            SystemClock,
        )),
    }
}

/// Every file under `path`, skipping hidden entries and build output
/// (`target`, and variants like `target.noindex`).
fn collect(path: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if path.is_file() {
        files.push(path.to_owned());
        return Ok(());
    }
    for entry in std::fs::read_dir(path)? {
        let entry = entry?.path();
        let name = entry
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if name.starts_with('.') || name.starts_with("target") {
            continue;
        }
        collect(&entry, files)?;
    }
    Ok(())
}
