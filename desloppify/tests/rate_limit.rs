//! The two limiters, driven through `RateLimiter::wait` with a manual clock.
//!
//! A limiter's rate is observed the only way a caller can: two calls asked
//! about at the same instant are booked one slot apart, so the gap between
//! their waits is `1 / rate`.

use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use desloppify::rate_limit::{
    self, AdaptiveConfig, BoxError, Clock, Exhausted, RateLimiter, Signal, TokenBucketConfig,
};

const SECOND: Duration = Duration::from_secs(1);
const HOUR: Duration = Duration::from_secs(3600);
/// How far a computed wait may stray from the exact one: `f64` seconds
/// round-trip through `Duration`'s nanoseconds.
const SLACK: Duration = Duration::from_micros(1);

/// A clock that moves only when told to.
#[derive(Clone)]
struct Manual(Arc<Mutex<Instant>>);

impl Manual {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(Instant::now())))
    }

    fn advance(&self, by: Duration) {
        *self.0.lock().unwrap() += by;
    }
}

impl Clock for Manual {
    fn now(&self) -> Instant {
        *self.0.lock().unwrap()
    }
}

/// A provider failure: an HTTP status and an optional Retry-After.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct Status(u16, Option<Duration>);

fn classify(error: &(dyn Error + 'static)) -> Signal {
    let Some(Status(status, retry_after)) = error.downcast_ref::<Status>() else {
        return Signal::Failed { retry_after: None };
    };
    let retry_after = *retry_after;
    match status {
        429 => Signal::Throttled { retry_after },
        _ => Signal::Failed { retry_after },
    }
}

fn throttled() -> Option<BoxError> {
    Some(Box::new(Status(429, None)))
}

fn outage() -> Option<BoxError> {
    Some(Box::new(Status(503, None)))
}

fn close(actual: Duration, expected: Duration) -> bool {
    actual.abs_diff(expected) <= SLACK
}

/// The gap between two fresh calls asked about now: `1 / rate`.
fn spacing(limiter: &impl RateLimiter) -> Duration {
    let first = limiter.wait(None).unwrap();
    let second = limiter.wait(None).unwrap();
    second - first
}

// ── token bucket ────────────────────────────────────────────────────────────

fn bucket(capacity: u64, max_wait: Duration) -> (impl RateLimiter, Manual) {
    let clock = Manual::new();
    let config = TokenBucketConfig {
        capacity,
        refill: HOUR,
        max_wait,
    };
    (rate_limit::token_bucket(config, clock.clone()), clock)
}

#[test]
fn a_full_bucket_lets_calls_through_at_once_until_it_is_empty() {
    let (bucket, _) = bucket(2, Duration::ZERO);
    assert_eq!(bucket.wait(None).unwrap(), Duration::ZERO);
    assert_eq!(bucket.wait(throttled()).unwrap(), Duration::ZERO);
    assert!(bucket.wait(None).unwrap_err().is::<Exhausted>());
}

#[test]
fn an_empty_bucket_queues_each_call_one_refill_behind_the_last() {
    let (bucket, _) = bucket(0, 3 * HOUR);
    assert_eq!(bucket.wait(None).unwrap(), HOUR);
    assert_eq!(bucket.wait(None).unwrap(), 2 * HOUR);
}

#[test]
fn a_bucket_earns_back_a_call_per_refill() {
    let (bucket, clock) = bucket(1, Duration::ZERO);
    assert_eq!(bucket.wait(None).unwrap(), Duration::ZERO);
    clock.advance(HOUR);
    assert_eq!(bucket.wait(None).unwrap(), Duration::ZERO);
}

#[test]
fn a_bucket_refusing_a_retry_hands_back_its_failure() {
    let (bucket, _) = bucket(0, SECOND);
    assert!(bucket.wait(throttled()).unwrap_err().is::<Status>());
}

// ── adaptive ────────────────────────────────────────────────────────────────

/// Floor 0.5/s, start 2/s, ceiling 4/s; growth 0.1/s per second.
fn config() -> AdaptiveConfig {
    AdaptiveConfig {
        floor: 0.5,
        start: 2.0,
        ceiling: 4.0,
        increase: 0.1,
        cooldown: 10 * SECOND,
        max_wait: HOUR,
    }
}

fn adaptive(config: AdaptiveConfig) -> (impl RateLimiter, Manual) {
    let clock = Manual::new();
    (rate_limit::adaptive(config, classify, clock.clone()), clock)
}

fn without_growth() -> AdaptiveConfig {
    AdaptiveConfig {
        increase: 0.0,
        ..config()
    }
}

#[test]
fn calls_are_paced_at_the_starting_rate() {
    let (limiter, _) = adaptive(without_growth());
    for i in 0..4 {
        let wait = limiter.wait(None).unwrap();
        assert!(close(wait, SECOND / 2 * i), "call {i} waited {wait:?}");
    }
}

#[test]
fn a_throttle_halves_the_rate_for_calls_booked_after_it() {
    let (limiter, _) = adaptive(without_growth());
    limiter.wait(None).unwrap();
    // The retry takes the slot booked at the old rate…
    assert!(close(limiter.wait(throttled()).unwrap(), SECOND / 2));
    // …and the calls after it are spaced at the new one.
    assert!(close(spacing(&limiter), SECOND));
}

#[test]
fn throttles_within_a_cooldown_cut_the_rate_once() {
    let (limiter, clock) = adaptive(without_growth());
    for _ in 0..5 {
        limiter.wait(throttled()).unwrap();
        clock.advance(SECOND);
    }
    assert!(close(spacing(&limiter), SECOND));
    clock.advance(config().cooldown);
    limiter.wait(throttled()).unwrap();
    assert!(close(spacing(&limiter), 2 * SECOND));
}

#[test]
fn the_rate_grows_while_calls_queue_and_stops_at_the_ceiling() {
    let (limiter, clock) = adaptive(config());
    let start = clock.now();
    while clock.now() < start + 10 * SECOND {
        clock.advance(limiter.wait(None).unwrap());
    }
    // Ten seconds busy at 0.1/s per second: about 3/s.
    let gap = spacing(&limiter).as_secs_f64();
    assert!((gap - 1.0 / 3.0).abs() < 0.01, "spacing {gap}");
    while clock.now() < start + HOUR {
        clock.advance(limiter.wait(None).unwrap());
    }
    assert!(close(spacing(&limiter), SECOND / 4));
}

#[test]
fn idle_time_does_not_grow_the_rate() {
    let (limiter, clock) = adaptive(config());
    limiter.wait(None).unwrap();
    clock.advance(HOUR);
    limiter.wait(None).unwrap();
    // Only the half second the first call's slot was booked ahead counts.
    assert!(close(
        spacing(&limiter),
        Duration::from_secs_f64(1.0 / 2.05)
    ));
}

#[test]
fn cuts_stop_at_the_floor() {
    let (limiter, clock) = adaptive(without_growth());
    for _ in 0..10 {
        limiter.wait(throttled()).unwrap();
        clock.advance(HOUR);
    }
    assert!(close(spacing(&limiter), 2 * SECOND));
}

#[test]
fn retry_after_is_a_floor_on_the_wait_and_later_calls_queue_behind_it() {
    let (limiter, _) = adaptive(without_growth());
    let told = 30 * SECOND;
    let error: BoxError = Box::new(Status(429, Some(told)));
    assert!(close(limiter.wait(Some(error)).unwrap(), told));
    assert!(close(limiter.wait(None).unwrap(), told + SECOND));
}

#[test]
fn an_outage_is_paced_without_cutting_the_rate() {
    let (limiter, _) = adaptive(without_growth());
    limiter.wait(None).unwrap();
    assert!(close(limiter.wait(outage()).unwrap(), SECOND / 2));
    assert!(close(spacing(&limiter), SECOND / 2));
}

#[test]
fn a_refused_retry_hands_back_its_failure() {
    let (limiter, _) = adaptive(AdaptiveConfig {
        max_wait: SECOND,
        ..config()
    });
    let error: BoxError = Box::new(Status(429, Some(HOUR)));
    assert!(limiter.wait(Some(error)).unwrap_err().is::<Status>());
}

#[test]
fn a_refused_first_attempt_is_exhausted() {
    let (limiter, _) = adaptive(AdaptiveConfig {
        max_wait: Duration::ZERO,
        ..config()
    });
    limiter.wait(None).unwrap();
    assert!(limiter.wait(None).unwrap_err().is::<Exhausted>());
}
