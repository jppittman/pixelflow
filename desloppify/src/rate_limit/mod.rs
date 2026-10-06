//! Pacing calls to a provider: before each call, how long to wait.
//!
//! The contract is [`RateLimiter`]; [`token_bucket`] and [`adaptive`] build
//! the two implementations. Time is an input — every limiter reads it from
//! the [`Clock`] it was built with — so a limiter's behavior is a function of
//! the calls it is asked about and the instants they are asked at.

mod adaptive;
mod token_bucket;

use std::error::Error;
use std::time::{Duration, Instant};

pub type BoxError = Box<dyn Error + Send + Sync + 'static>;

/// Decides how long each call to a provider waits before it goes out.
///
/// # Contract
///
/// - Called before every call: with `None` before a first attempt, with
///   `Some(failure)` before retrying `failure`. Only failures worth retrying
///   are passed; a caller gives up on the others without asking.
/// - `Ok(wait)`: the call may go out once `wait` has passed, measured from the
///   clock's `now` at the time of asking. Every `Ok` books a slot: two calls
///   asked about at the same instant are never both told to go at once unless
///   the limiter's budget allows both.
/// - `Err(error)`: the call must not be made. `error` is the `last` failure
///   handed in, or [`Exhausted`] when there was none.
/// - Safe to call from many tasks at once; slots are booked in the order the
///   calls are made.
pub trait RateLimiter: Send + Sync {
    fn wait(&self, last: Option<BoxError>) -> Result<Duration, BoxError>;
}

/// Where a limiter reads the time.
pub trait Clock: Send + Sync + 'static {
    /// The current instant; never earlier than an instant it returned before.
    fn now(&self) -> Instant;
}

/// The process's monotonic clock.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// A first attempt refused because its wait would exceed the limiter's
/// `max_wait`; a refused retry hands back its own failure instead.
#[derive(Debug, thiserror::Error)]
#[error("rate limit: no slot within {0:?}")]
pub struct Exhausted(pub Duration);

/// What a failed call says about the rate. Telling one from the other means
/// reading a provider's errors, which is the caller's business: see
/// [`Classify`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// The provider refused the call for its rate (a 429).
    Throttled { retry_after: Option<Duration> },
    /// Any other retryable failure: an outage or a timeout, which says
    /// nothing about how fast it is safe to call.
    Failed { retry_after: Option<Duration> },
}

/// Reads a failure the limiter was handed as a [`Signal`].
pub type Classify = fn(&(dyn Error + 'static)) -> Signal;

/// A fixed budget.
#[derive(Debug, Clone, Copy)]
pub struct TokenBucketConfig {
    /// Calls that may go out back to back from a full bucket.
    pub capacity: u64,
    /// The time to earn back one call.
    pub refill: Duration,
    /// Refuse a call rather than ask it to wait longer than this.
    pub max_wait: Duration,
}

/// A discovered budget. Rates are in calls per second.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveConfig {
    /// The rate is never cut below this.
    pub floor: f64,
    /// The rate the first call goes out at; held within `floor..=ceiling`.
    pub start: f64,
    /// The rate never grows past this.
    pub ceiling: f64,
    /// Calls per second gained per second of calling at the full rate
    /// without a throttle.
    pub increase: f64,
    /// After a cut, how long further throttles count as the same one. It
    /// should cover a call's latency, so the replies of calls already in
    /// flight at the old rate land inside it.
    pub cooldown: Duration,
    /// Refuse a call rather than ask it to wait longer than this.
    pub max_wait: Duration,
}

/// A token bucket, starting full.
///
/// Every call takes a token. The bucket holds at most `capacity` and gains one
/// every `refill`; a call with no token waits for the next one — calls queue,
/// each behind the last — unless that wait exceeds `max_wait`. It does not
/// look at the failure: a retry is a call like any other.
pub fn token_bucket(config: TokenBucketConfig, clock: impl Clock) -> impl RateLimiter {
    token_bucket::TokenBucket::new(config, clock)
}

/// An AIMD limiter that searches for the rate the provider will bear, as TCP
/// congestion control does.
///
/// - **Pacing.** Calls are booked `1 / rate` apart, each after the last; an
///   idle limiter lets one call through at once. There is no burst.
/// - **Decrease.** A [`Signal::Throttled`] halves the rate, at most once per
///   `cooldown`: 429s from calls already in flight are the same event. The
///   rate never drops below `floor`.
/// - **Increase.** While calls are queued (a slot is booked ahead) and no
///   cooldown is running, the rate grows by `increase` per second, up to
///   `ceiling`. Idle time earns nothing.
/// - **Retry-After** is a floor on the wait, and calls booked after it queue
///   behind it.
/// - **Other failures** ([`Signal::Failed`]) are paced at the current rate and
///   change nothing.
///
/// # Panics
///
/// If `config.floor` is not positive or `config.ceiling` is below it.
pub fn adaptive(config: AdaptiveConfig, classify: Classify, clock: impl Clock) -> impl RateLimiter {
    adaptive::Adaptive::new(config, classify, clock)
}
