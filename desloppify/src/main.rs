use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;

use desloppify::agent::Agent;
use desloppify::model::Provider;
use desloppify::rate_limit::TokenBucket;
use desloppify::review::review;
use desloppify::rule::Rule;
use desloppify::skills::Skills;

/// Review code against the rules in `rules/`.
#[derive(Parser)]
struct Args {
    /// Files or directories to review.
    #[arg(required = true)]
    paths: Vec<PathBuf>,
    #[arg(long, value_enum, default_value_t = Provider::Anthropic)]
    provider: Provider,
    /// Retries that may happen back to back.
    #[arg(long, default_value_t = 10)]
    retry_burst: u64,
    /// Seconds to earn back one retry.
    #[arg(long, default_value_t = 6)]
    retry_refill_secs: u64,
    /// Give up on a call rather than wait longer than this for a retry.
    #[arg(long, default_value_t = 120)]
    retry_max_wait_secs: u64,
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/rules"))]
    rules: PathBuf,
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/skills"))]
    skills: PathBuf,
}

#[tokio::main]
async fn main() -> Result<ExitCode> {
    let args = Args::parse();
    let skills = Skills::load(&args.skills)?;
    let rules = Arc::new(Rule::load_dir(&args.rules, &skills)?);
    let limiter = TokenBucket::new(
        args.retry_burst,
        Duration::from_secs(args.retry_refill_secs),
        Duration::from_secs(args.retry_max_wait_secs),
    );
    let agent = Arc::new(Agent::from_env(args.provider, Box::new(limiter))?);

    let mut files = Vec::new();
    for path in &args.paths {
        collect(path, &mut files)?;
    }

    let report = review(agent, rules, &files).await?;
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

/// Every file under `path`, skipping hidden entries and build output.
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
        if name.starts_with('.') || name == "target" {
            continue;
        }
        collect(&entry, files)?;
    }
    Ok(())
}
