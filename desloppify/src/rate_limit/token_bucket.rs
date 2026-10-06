//! A fixed budget: a token bucket.

use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use super::{BoxError, Exhausted, RateLimiter};

/// Every call takes a token. The bucket holds at most `capacity` and gains
/// one every `refill`; a call with no token waits for the next one, unless
/// that wait exceeds `max_wait`, in which case it is refused.
///
/// It does not look at the error: a retry is a call like any other.
pub struct TokenBucket {
    capacity: u64,
    refill: Duration,
    max_wait: Duration,
    state: Mutex<Bucket>,
}

struct Bucket {
    /// Negative while calls are queued waiting for tokens.
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
    fn wait(&self, last: Option<BoxError>) -> Result<Duration, BoxError> {
        // The bucket is plain numbers, valid after any panic mid-update.
        let mut bucket = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let now = Instant::now();
        let earned = now.duration_since(bucket.updated).as_secs_f64() / self.refill.as_secs_f64();
        bucket.tokens = (bucket.tokens + earned).min(self.capacity as f64);
        bucket.updated = now;

        let shortfall = (1.0 - bucket.tokens).max(0.0);
        let wait = self.refill.mul_f64(shortfall);
        if wait > self.max_wait {
            return Err(last.unwrap_or_else(|| Box::new(Exhausted(self.max_wait))));
        }
        bucket.tokens -= 1.0;
        Ok(wait)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOUR: Duration = Duration::from_secs(3600);

    fn failure() -> Option<BoxError> {
        Some(Box::new(std::io::Error::other("429")))
    }

    #[test]
    fn full_bucket_lets_calls_through_until_empty() {
        let bucket = TokenBucket::new(2, HOUR, Duration::ZERO);
        assert_eq!(bucket.wait(None).unwrap(), Duration::ZERO);
        assert_eq!(bucket.wait(failure()).unwrap(), Duration::ZERO);
        assert!(bucket.wait(None).is_err());
    }

    #[test]
    fn empty_bucket_waits_for_the_next_token_and_queues_behind_it() {
        let bucket = TokenBucket::new(0, HOUR, 3 * HOUR);
        let first = bucket.wait(None).unwrap();
        let second = bucket.wait(None).unwrap();
        assert!(first <= HOUR && first > HOUR - Duration::from_secs(1));
        assert!(second > first + HOUR - Duration::from_secs(1));
    }

    #[test]
    fn refusal_hands_back_the_error_it_was_given() {
        let bucket = TokenBucket::new(0, HOUR, Duration::from_secs(1));
        assert_eq!(bucket.wait(failure()).unwrap_err().to_string(), "429");
        assert!(bucket.wait(None).unwrap_err().is::<Exhausted>());
    }
}
