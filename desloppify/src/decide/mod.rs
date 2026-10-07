//! System One: closed-set decisions about one piece of code, many questions
//! at once, each answered with a calibrated confidence.
//!
//! The contract is [`Decide`]. A review asks it first, with every rule's
//! question about a unit in one request; an answer it is confident in settles
//! that rule for the unit, and the rest go on to a model through
//! [`Ask`](crate::agent::Ask). Three implementations:
//!
//! - [`jev`] / [`jev_from_env`]: TypeSafe AI's Jev, a model that returns
//!   typed decisions with calibrated probabilities instead of text.
//! - [`dry_run`]: no model — every question is answered with its first label
//!   at full confidence, and priced by its input.
//! - [`none`]: answers nothing, so every question goes to the model.

mod dry_run;
mod jev;
mod none;

use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use anyhow::Result;

use crate::agent::Usage;

/// Answers closed-set questions about one state.
///
/// # Contract
///
/// - `decide(state, questions)` evaluates every question against `state`
///   and returns, for each question it answers, the label it chose — one of
///   that question's labels — and its confidence, from 0 to 1.
/// - A question it does not answer is absent from the result; the caller
///   treats it as unanswered, not as any label.
/// - `Err` means no question was answered, and nothing was billed.
/// - Safe to call concurrently; an implementation bounds its own concurrency.
pub trait Decide: Send + Sync + 'static {
    fn decide(
        &self,
        state: &str,
        questions: &[Choice<'_>],
    ) -> impl Future<Output = Result<Decisions>> + Send;
}

/// One closed-set question.
#[derive(Debug, Clone, Copy)]
pub struct Choice<'a> {
    /// The question's name, unique within a request; answers are keyed by it.
    pub name: &'a str,
    /// What to decide.
    pub instructions: &'a str,
    /// The labels to choose from, each with what it means. Never empty.
    pub labels: &'a [(&'a str, &'a str)],
}

/// One answer.
#[derive(Debug, Clone, PartialEq)]
pub struct Decided {
    pub label: String,
    pub confidence: f64,
}

/// A request's answers, by question name, and what it cost.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Decisions {
    pub answers: BTreeMap<String, Decided>,
    pub usage: Usage,
}

/// Where and how to reach Jev.
#[derive(Debug, Clone)]
pub struct JevConfig {
    /// Without the trailing `/v1/systemone`, e.g. `https://api.typesafe.ai`.
    pub base_url: String,
    pub api_key: String,
    /// A model name or alias, e.g. `jev-latest`.
    pub model: String,
    /// Most requests in flight at once.
    pub jobs: NonZeroUsize,
}

/// Jev, at `config`. Each [`Decide::decide`] is one `POST /v1/systemone`,
/// retried on 408, 429 and 5xx after the server's `Retry-After` (or a short
/// backoff), at most twice.
#[must_use]
pub fn jev(config: JevConfig) -> impl Decide {
    jev::Jev::new(config)
}

/// Jev configured the way TypeSafe's own SDK is: `TYPESAFE_API_KEY`,
/// `TYPESAFE_BASE_URL` (default `https://api.typesafe.ai`) and
/// `TYPESAFE_DEFAULT_MODEL` (default `jev-latest`).
///
/// # Errors
///
/// `TYPESAFE_API_KEY` is unset or empty.
pub fn jev_from_env(jobs: NonZeroUsize) -> Result<impl Decide> {
    jev::from_env(jobs).map(jev::Jev::new)
}

/// Decides without a model: every question gets its first label at
/// confidence 1, and the request is priced as one call whose input is the
/// state and questions at four characters a token.
#[must_use]
pub fn dry_run() -> impl Decide {
    dry_run::DryRun
}

/// Answers nothing and costs nothing, so every question goes to the model.
#[must_use]
pub fn none() -> impl Decide {
    none::Nobody
}
