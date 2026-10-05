//! Retry budgets: after a failed call, whether to retry it and when.

use std::error::Error;
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

pub trait RateLimiter: Send + Sync {
    /// A call just failed with `error`. How long to wait before retrying it,
    /// or `None` to give up and report the error.
    fn on_error(&self, error: &(dyn Error + Send + Sync + 'static)) -> Option<Duration>;
}

/// Every retry takes a token. The bucket holds at most `capacity` and gains
/// one every `refill`; a retry with no token waits for the next one, unless
/// that wait exceeds `max_wait`, in which case it gives up.
///
/// It does not look at the error: a failure is a failure.
pub struct TokenBucket {
    capacity: u64,
    refill: Duration,
    max_wait: Duration,
    state: Mutex<Bucket>,
}

struct Bucket {
    /// Negative while retries are queued waiting for tokens.
    tokens: f64,
    updated: Instant,
}

impl TokenBucket {
    /// A full bucket.
    #[must_use]
    pub fn new(capacity: u64, refill: Duration, max_wait: Duration) -> Self {
        let state = Mutex::new(Bucket {
            tokens: capacity as f64,
            updated: Instant::now(),
        });
        Self {
            capacity,
            refill,
            max_wait,
            state,
        }
    }
}

impl RateLimiter for TokenBucket {
    fn on_error(&self, _error: &(dyn Error + Send + Sync + 'static)) -> Option<Duration> {
        // The bucket is plain numbers, valid after any panic mid-update.
        let mut bucket = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let now = Instant::now();
        let earned = now.duration_since(bucket.updated).as_secs_f64() / self.refill.as_secs_f64();
        bucket.tokens = (bucket.tokens + earned).min(self.capacity as f64);
        bucket.updated = now;

        let shortfall = (1.0 - bucket.tokens).max(0.0);
        let wait = self.refill.mul_f64(shortfall);
        if wait > self.max_wait {
            return None;
        }
        bucket.tokens -= 1.0;
        Some(wait)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOUR: Duration = Duration::from_secs(3600);

    fn failure() -> std::io::Error {
        std::io::Error::other("429")
    }

    #[test]
    fn full_bucket_retries_at_once_until_empty() {
        let bucket = TokenBucket::new(2, HOUR, Duration::ZERO);
        assert_eq!(bucket.on_error(&failure()), Some(Duration::ZERO));
        assert_eq!(bucket.on_error(&failure()), Some(Duration::ZERO));
        assert_eq!(bucket.on_error(&failure()), None);
    }

    #[test]
    fn empty_bucket_waits_for_the_next_token_and_queues_behind_it() {
        let bucket = TokenBucket::new(0, HOUR, 3 * HOUR);
        let first = bucket.on_error(&failure()).unwrap();
        let second = bucket.on_error(&failure()).unwrap();
        assert!(first <= HOUR && first > HOUR - Duration::from_secs(1));
        assert!(second > first + HOUR - Duration::from_secs(1));
    }

    #[test]
    fn retry_past_max_wait_gives_up_without_taking_a_token() {
        let bucket = TokenBucket::new(0, HOUR, Duration::from_secs(1));
        assert_eq!(bucket.on_error(&failure()), None);
        assert_eq!(bucket.on_error(&failure()), None);
    }
}
