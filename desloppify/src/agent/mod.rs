//! Asking a model a question.
//!
//! The contract is [`Ask`]. Three backends implement it:
//!
//! - [`from_env`]: a provider's API (Anthropic, Gemini) through rig, paced by
//!   a rate limiter. [`classify`] reads its failures for an
//!   [`adaptive`](crate::rate_limit::adaptive) limiter.
//! - [`claude_code`]: the Claude Code CLI, one `claude -p` per call, on
//!   whatever account the CLI is logged in to.
//! - [`dry_run`]: no model at all — every call is answered `[]` and reports
//!   the input tokens it would have sent. Running a review through it is how
//!   a review is priced, through the same pipeline a real one takes.

mod claude_code;
mod dry_run;
mod rig;

use std::error::Error;
use std::num::NonZeroUsize;
use std::ops::AddAssign;

use anyhow::Result;

use crate::model::{ModelLevel, Provider};
use crate::rate_limit::{RateLimiter, Signal};

/// A model that answers one prompt at a time, from many tasks at once.
///
/// # Contract
///
/// - `ask(question)` sends `question.system` as the system prompt and
///   `question.prompt` as the user turn to the model at `question.level`, and
///   returns the text of its reply, unaltered, with the tokens the call used.
/// - With `question.schema` set, the reply text is a JSON document that
///   satisfies the schema: the backend has the provider enforce it, not the
///   caller parse around it.
/// - `Err` means no reply will come: the provider refused the call in a way a
///   retry cannot fix (bad credentials, a malformed request), or the rate
///   limiter gave up on retrying. An implementation retries what is worth
///   retrying before it returns.
/// - Safe to call concurrently; an implementation bounds its own concurrency
///   and pacing, so a caller may issue every call at once.
pub trait Ask: Send + Sync + 'static {
    fn ask(&self, question: &Question<'_>) -> impl Future<Output = Result<Answer>> + Send;
}

/// One prompt to one model.
#[derive(Debug, Clone, Copy)]
pub struct Question<'a> {
    pub level: ModelLevel,
    pub system: &'a str,
    pub prompt: &'a str,
    /// A JSON Schema the reply must satisfy; `None` for free text.
    pub schema: Option<&'a serde_json::Value>,
}

/// A model's reply and what it cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer {
    pub text: String,
    pub usage: Usage,
}

/// Tokens billed, summed over some number of calls.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub calls: u64,
    /// Every input token, cached or not.
    pub input: u64,
    /// Every output token, reasoning included.
    pub output: u64,
}

impl AddAssign for Usage {
    fn add_assign(&mut self, other: Self) {
        self.calls += other.calls;
        self.input += other.input;
        self.output += other.output;
    }
}

/// An [`Ask`] that calls `provider`, credentialed from its usual environment
/// variable (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`). Every call and retry is
/// paced by `limiter`, with at most `jobs` calls in flight — retries included,
/// so a pace the limiter sets reaches the next call rather than the end of a
/// queue booked at the old one.
pub fn from_env(
    provider: Provider,
    limiter: Box<dyn RateLimiter>,
    jobs: NonZeroUsize,
) -> Result<impl Ask> {
    rig::RigAgent::from_env(provider, limiter, jobs)
}

/// An [`Ask`] that runs `claude -p` for each call, with at most `jobs` at
/// once. Each runs with no tools, no MCP servers, no settings and no
/// session, from an empty directory, with `preamble` replacing Claude Code's
/// own system prompt; the model is the Anthropic model for the call's level.
/// Usage is what the CLI reports, which includes its own per-call overhead.
#[must_use]
pub fn claude_code(jobs: NonZeroUsize) -> impl Ask {
    claude_code::ClaudeCode::new(jobs)
}

/// An [`Ask`] that calls nothing: every answer is `[]` — no findings — and
/// its usage is one call whose input is the preamble and prompt at four
/// characters a token, and whose output is zero. An estimate for pricing a
/// review, not a bill: tokenizers differ, and replies are not counted.
#[must_use]
pub fn dry_run() -> impl Ask {
    dry_run::DryRun
}

/// What a provider's failure says about the call rate: a 429 is
/// [`Signal::Throttled`], anything else — an outage, a timeout, an error that
/// isn't the provider's — is [`Signal::Failed`]. Either carries the seconds
/// form of `Retry-After` when the provider sent one; the date form is ignored.
#[must_use]
pub fn classify(error: &(dyn Error + 'static)) -> Signal {
    rig::classify(error)
}
