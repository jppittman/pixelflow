//! Asking a model a question.
//!
//! The contract is [`Ask`]; [`from_env`] builds the implementation that calls
//! a real provider. [`classify`] reads that provider's failures for a
//! [`rate_limit::adaptive`](crate::rate_limit::adaptive) limiter.

mod rig;

use std::error::Error;
use std::num::NonZeroUsize;

use anyhow::Result;

use crate::model::{ModelLevel, Provider};
use crate::rate_limit::{RateLimiter, Signal};

/// A model that answers one prompt at a time, from many tasks at once.
///
/// # Contract
///
/// - `ask(level, preamble, prompt)` sends `preamble` as the system prompt and
///   `prompt` as the user turn to the model at `level`, and returns the text
///   of its reply, unaltered.
/// - `Err` means no reply will come: the provider refused the call in a way a
///   retry cannot fix (bad credentials, a malformed request), or the rate
///   limiter gave up on retrying. An implementation retries what is worth
///   retrying before it returns.
/// - Safe to call concurrently; an implementation bounds its own concurrency
///   and pacing, so a caller may issue every call at once.
pub trait Ask: Send + Sync + 'static {
    fn ask(
        &self,
        level: ModelLevel,
        preamble: &str,
        prompt: &str,
    ) -> impl Future<Output = Result<String>> + Send;
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

/// What a provider's failure says about the call rate: a 429 is
/// [`Signal::Throttled`], anything else — an outage, a timeout, an error that
/// isn't the provider's — is [`Signal::Failed`]. Either carries the seconds
/// form of `Retry-After` when the provider sent one; the date form is ignored.
#[must_use]
pub fn classify(error: &(dyn Error + 'static)) -> Signal {
    rig::classify(error)
}
