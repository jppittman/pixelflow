//! A fixed budget: a token bucket.

use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use super::{BoxError, Clock, Exhausted, RateLimiter, TokenBucketConfig};

/// Every call takes a token. The bucket holds at most `capacity` and gains
/// one every `refill`; a call with no token waits for the next one, unless
/// that wait exceeds `max_wait`, in which case it is refused.
///
/// It does not look at the error: a retry is a call like any other.
pub(super) struct TokenBucket<C> {
    config: TokenBucketConfig,
    clock: C,
    state: Mutex<Bucket>,
}

struct Bucket {
    /// Negative while calls are queued waiting for tokens.
    tokens: f64,
    updated: Instant,
}

impl<C: Clock> TokenBucket<C> {
    pub(super) fn new(config: TokenBucketConfig, clock: C) -> Self {
        let state = Mutex::new(Bucket {
            tokens: config.capacity as f64,
            updated: clock.now(),
        });
        Self {
            config,
            clock,
            state,
        }
    }
}

impl<C: Clock> RateLimiter for TokenBucket<C> {
    fn wait(&self, last: Option<BoxError>) -> Result<Duration, BoxError> {
        // The bucket is plain numbers, valid after any panic mid-update.
        let mut bucket = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let now = self.clock.now();
        let TokenBucketConfig {
            capacity,
            refill,
            max_wait,
        } = self.config;
        let earned = now.duration_since(bucket.updated).as_secs_f64() / refill.as_secs_f64();
        bucket.tokens = (bucket.tokens + earned).min(capacity as f64);
        bucket.updated = now;

        let shortfall = (1.0 - bucket.tokens).max(0.0);
        let wait = refill.mul_f64(shortfall);
        if wait > max_wait {
            return Err(last.unwrap_or_else(|| Box::new(Exhausted(max_wait))));
        }
        bucket.tokens -= 1.0;
        Ok(wait)
    }
}
