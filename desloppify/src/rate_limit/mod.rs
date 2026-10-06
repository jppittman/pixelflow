//! Pacing calls to a provider: before each call, how long to wait.

mod adaptive;
mod token_bucket;

use std::error::Error;
use std::time::Duration;

pub use adaptive::{Adaptive, AdaptiveConfig, Classify, Signal};
pub use token_bucket::TokenBucket;

pub type BoxError = Box<dyn Error + Send + Sync + 'static>;

pub trait RateLimiter: Send + Sync {
    /// How long to wait before the next call. `last` is `None` before a first
    /// attempt and the failure being retried otherwise. `Err` refuses the
    /// call, handing back `last` (or an error of its own if there was none).
    fn wait(&self, last: Option<BoxError>) -> Result<Duration, BoxError>;
}

/// A first attempt refused because its wait would exceed the limiter's
/// `max_wait`; a refused retry hands back its own failure instead.
#[derive(Debug, thiserror::Error)]
#[error("rate limit: no slot within {0:?}")]
pub struct Exhausted(Duration);
