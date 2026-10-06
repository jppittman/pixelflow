use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, ValueEnum};

use std::collections::BTreeMap;

use desloppify::agent::{self, Ask, Usage};
use desloppify::model::Provider;
use desloppify::rate_limit::{self, AdaptiveConfig, RateLimiter, SystemClock, TokenBucketConfig};
use desloppify::review::{Call, Report, plan, review, synthesize};
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

/// Who answers the calls.
#[derive(Clone, Copy, ValueEnum)]
enum Backend {
    /// Anthropic's API, from `ANTHROPIC_API_KEY`.
    Anthropic,
    /// Gemini's API, from `GEMINI_API_KEY`.
    Gemini,
    /// The Claude Code CLI, `claude -p`, on the account it is logged in to.
    ClaudeCode,
    /// No model: price the review without making a call.
    DryRun,
}

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
    #[arg(long, value_enum, default_value_t = Backend::Anthropic)]
    backend: Backend,
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
    /// Print each finding as `path:line: [rule] message` instead of the
    /// lead reviewer's synthesized review.
    #[arg(long)]
    findings_only: bool,
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

    match args.backend {
        Backend::Anthropic => run(api(Provider::Anthropic, &args)?, &args, rules, plan).await,
        Backend::Gemini => run(api(Provider::Gemini, &args)?, &args, rules, plan).await,
        Backend::ClaudeCode => run(agent::claude_code(args.jobs), &args, rules, plan).await,
        Backend::DryRun => run(agent::dry_run(), &args, rules, plan).await,
    }
}

fn api(provider: Provider, args: &Args) -> Result<impl Ask> {
    agent::from_env(provider, limiter(args), args.jobs)
}

async fn run<A: Ask>(
    agent: A,
    args: &Args,
    rules: Arc<Vec<Rule>>,
    plan: Vec<Call>,
) -> Result<ExitCode> {
    let agent = Arc::new(agent);
    let report = review(agent.clone(), rules.clone(), plan).await?;
    for failure in &report.failures {
        eprintln!("error: {failure:#}");
    }
    let lead = if args.findings_only {
        None
    } else {
        synthesize(&*agent, &rules, &report).await?
    };
    match &lead {
        Some(review) => println!("{}", review.text),
        None => print_findings(&report),
    }
    print_usage(&rules, &report, lead.map(|l| l.usage));
    let clean = report.findings.is_empty() && report.failures.is_empty();
    Ok(if clean {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

/// Calls and tokens per rule, per level, and the lead's, to stderr.
fn print_usage(rules: &[Rule], report: &Report, lead: Option<Usage>) {
    let row = |usage: &Usage, level: &str, name: &str| {
        eprintln!(
            "{:>8}  {:>12}  {:>10}  {level:>5}  {name}",
            usage.calls, usage.input, usage.output
        );
    };
    eprintln!(
        "{:>8}  {:>12}  {:>10}  level  rule",
        "calls", "in tokens", "out tokens"
    );
    let mut per_level: BTreeMap<u64, Usage> = BTreeMap::new();
    for rule in rules {
        let Some(usage) = report.usage.get(&rule.id) else {
            continue;
        };
        let level = u64::from(rule.level);
        row(usage, &level.to_string(), &rule.id);
        *per_level.entry(level).or_default() += *usage;
    }
    for (level, usage) in &per_level {
        row(usage, &level.to_string(), "(level total)");
    }
    if let Some(lead) = lead {
        row(&lead, "4", "(lead review)");
    }
}

fn print_findings(report: &Report) {
    for f in &report.findings {
        println!(
            "{}:{}: [{}] {}",
            f.path.display(),
            f.line,
            f.rule,
            f.message
        );
    }
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
