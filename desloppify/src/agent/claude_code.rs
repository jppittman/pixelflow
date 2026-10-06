//! [`Ask`](super::Ask) through the Claude Code CLI.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::Stdio;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::sync::Semaphore;

use super::{Answer, Ask, Question, Usage};
use crate::model::{ModelLevel, Provider};

const CLI: &str = "claude";

pub(super) struct ClaudeCode {
    in_flight: Semaphore,
    /// An empty directory to run in, so no project's `CLAUDE.md` or settings
    /// ride along with the prompt.
    workdir: PathBuf,
}

/// The CLI's `--output-format json` result, the fields read from it.
#[derive(Deserialize)]
struct Outcome {
    is_error: bool,
    result: String,
    usage: CliUsage,
}

#[derive(Deserialize)]
struct CliUsage {
    input_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
    output_tokens: u64,
}

impl ClaudeCode {
    pub(super) fn new(jobs: NonZeroUsize) -> Self {
        Self {
            in_flight: Semaphore::new(jobs.get()),
            workdir: std::env::temp_dir().join("desloppify-claude-code"),
        }
    }
}

impl Ask for ClaudeCode {
    async fn ask(&self, question: &Question<'_>) -> Result<Answer> {
        let _permit = self.in_flight.acquire().await?;
        std::fs::create_dir_all(&self.workdir)
            .with_context(|| format!("creating {}", self.workdir.display()))?;
        let mut command = Command::new(CLI);
        command
            .current_dir(&self.workdir)
            .args(["-p", "--output-format", "json", "--model"])
            .arg(Provider::Anthropic.model(question.level))
            .arg("--system-prompt")
            .arg(question.system)
            .args([
                "--tools",
                "",
                "--strict-mcp-config",
                "--setting-sources",
                "",
            ])
            .arg("--no-session-persistence");
        think(&mut command, question.level);
        if let Some(schema) = question.schema {
            command.arg("--json-schema").arg(schema.to_string());
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("running `{CLI}`; is Claude Code installed?"))?;
        let mut stdin = child.stdin.take().context("no stdin to the CLI")?;
        stdin.write_all(question.prompt.as_bytes()).await?;
        drop(stdin);
        let output = child.wait_with_output().await?;

        let outcome: Outcome = serde_json::from_slice(&output.stdout).with_context(|| {
            format!(
                "`{CLI}` exited {} without a JSON result: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            )
        })?;
        if outcome.is_error {
            bail!("`{CLI}` failed: {}", outcome.result);
        }
        let usage = outcome.usage;
        Ok(Answer {
            text: outcome.result,
            usage: Usage {
                calls: 1,
                input: usage.input_tokens
                    + usage.cache_creation_input_tokens
                    + usage.cache_read_input_tokens,
                output: usage.output_tokens,
            },
        })
    }
}

/// How hard the CLI thinks at each level. The cheap levels answer narrow
/// questions and don't think at all: their thinking was most of their output
/// tokens, and output costs several times input. The levels asked to judge
/// design think, harder at the top.
fn think(command: &mut Command, level: ModelLevel) {
    match level {
        ModelLevel::Lite | ModelLevel::Fast => command.env("MAX_THINKING_TOKENS", "0"),
        ModelLevel::Strong => command.args(["--effort", "medium"]),
        ModelLevel::Frontier => command.args(["--effort", "high"]),
    };
}
