//! A discovered budget: AIMD on the call rate.

use std::error::Error;
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use super::{BoxError, Exhausted, RateLimiter};

/// The share of the rate kept after a throttle. Halving is TCP Reno's choice:
/// growth is slow and additive, so a throttle means the rate has only just
/// passed the limit, and half of it is safely under. AIMD with any factor
/// below one converges; a gentler one recovers faster but takes more 429s to
/// get under a limit that dropped.
const DECREASE: f64 = 0.5;

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

impl Signal {
    fn retry_after(self) -> Option<Duration> {
        match self {
            Self::Throttled { retry_after } | Self::Failed { retry_after } => retry_after,
        }
    }
}

/// Reads a failure the limiter was handed as a [`Signal`].
pub type Classify = fn(&(dyn Error + 'static)) -> Signal;

/// Rates are in calls per second.
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

/// Paces calls `1 / rate` apart and searches for the rate the provider will
/// bear, as TCP congestion control does: additive increase while it is not
/// throttled, multiplicative decrease when it is.
///
/// - **Pacing.** Each call is booked the slot after the last one booked, so
///   concurrent callers queue behind each other. There is no burst: an idle
///   limiter lets one call through at once, then spaces the rest.
/// - **Decrease.** A [`Signal::Throttled`] halves the rate, but at most once
///   per `cooldown`: the calls in flight when the first 429 came back went out
///   at the old rate, and their 429s are the same congestion event.
/// - **Increase.** `wait` is only ever told about failures, so "not
///   throttled" is the passage of time without a throttle. The rate grows by
///   `increase` per second, but only for time the pacing was binding — a slot
///   was booked ahead — and outside a cooldown. Time spent idle proves nothing
///   about a faster rate, so it earns nothing (RFC 7661's rule for an
///   application-limited TCP sender), and growth during a cooldown would be
///   counted before the replies to the old rate are in. Counting `wait(None)`
///   calls instead would credit a queue of fresh calls the moment it forms,
///   before any of them has gone out.
/// - **Retry-After** is a floor on the wait, and every call booked after it
///   queues behind it: the provider said when it will take calls again.
/// - **Other failures** ([`Signal::Failed`]) are paced and nothing more. An
///   outage is not a rate problem, and cutting for it would discard a rate
///   that took minutes to find.
pub struct Adaptive {
    config: AdaptiveConfig,
    classify: Classify,
    state: Mutex<Pace>,
}

struct Pace {
    rate: f64,
    /// When the slot after the last one booked opens; pacing binds until then.
    next: Instant,
    /// When growth was last credited.
    credited: Instant,
    /// The end of the current cooldown: throttles before it don't cut, and
    /// time before it doesn't grow the rate.
    calm_from: Instant,
}

impl Adaptive {
    /// A limiter at `config.start`, reading failures with `classify`.
    ///
    /// # Panics
    ///
    /// If `config.floor` is not positive or `config.ceiling` is below it.
    #[must_use]
    pub fn new(config: AdaptiveConfig, classify: Classify) -> Self {
        assert!(config.floor > 0.0, "adaptive rate floor must be positive");
        assert!(
            config.ceiling >= config.floor,
            "adaptive rate ceiling below its floor"
        );
        let now = Instant::now();
        let state = Mutex::new(Pace {
            rate: config.start.clamp(config.floor, config.ceiling),
            next: now,
            credited: now,
            calm_from: now,
        });
        Self {
            config,
            classify,
            state,
        }
    }

    fn wait_at(&self, last: Option<BoxError>, now: Instant) -> Result<Duration, BoxError> {
        // The pace is plain numbers and instants, valid after any panic.
        let mut pace = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        pace.grow(now, &self.config);

        let signal = last.as_deref().map(|error| (self.classify)(error));
        if let Some(Signal::Throttled { .. }) = signal {
            pace.cut(now, &self.config);
        }
        let not_before = now + signal.and_then(Signal::retry_after).unwrap_or_default();
        let slot = pace.next.max(not_before);
        let wait = slot - now;
        if wait > self.config.max_wait {
            return Err(last.unwrap_or_else(|| Box::new(Exhausted(self.config.max_wait))));
        }
        pace.next = slot + Duration::from_secs_f64(pace.rate.recip());
        Ok(wait)
    }
}

impl Pace {
    /// Credit growth for the time since the last credit that pacing was
    /// binding and no cooldown was running.
    fn grow(&mut self, now: Instant, config: &AdaptiveConfig) {
        let from = self.credited.max(self.calm_from);
        let busy = now.min(self.next).saturating_duration_since(from);
        self.credited = now;
        self.rate = (self.rate + config.increase * busy.as_secs_f64()).min(config.ceiling);
    }

    fn cut(&mut self, now: Instant, config: &AdaptiveConfig) {
        if now < self.calm_from {
            return;
        }
        self.rate = (self.rate * DECREASE).max(config.floor);
        self.calm_from = now + config.cooldown;
    }
}

impl RateLimiter for Adaptive {
    fn wait(&self, last: Option<BoxError>) -> Result<Duration, BoxError> {
        self.wait_at(last, Instant::now())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECOND: Duration = Duration::from_secs(1);
    const HOUR: Duration = Duration::from_secs(3600);
    /// How far a computed wait may stray from the exact one: `f64` seconds
    /// round-trip through `Duration`'s nanoseconds.
    const SLACK: Duration = Duration::from_micros(1);

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

    fn fresh(config: AdaptiveConfig) -> (Adaptive, Instant) {
        let limiter = Adaptive::new(config, classify);
        let now = limiter.state.lock().unwrap().next;
        (limiter, now)
    }

    fn rate(limiter: &Adaptive) -> f64 {
        limiter.state.lock().unwrap().rate
    }

    fn close(actual: Duration, expected: Duration) -> bool {
        actual.abs_diff(expected) <= SLACK
    }

    #[test]
    fn steady_state_paces_calls_at_the_current_rate() {
        let (limiter, t0) = fresh(AdaptiveConfig {
            increase: 0.0,
            ..config()
        });
        let waits: Vec<_> = (0..4).map(|_| limiter.wait_at(None, t0).unwrap()).collect();
        let half = SECOND / 2;
        for (i, wait) in (0u32..).zip(waits) {
            assert!(close(wait, half * i), "call {i} waited {wait:?}");
        }
    }

    #[test]
    fn a_throttle_halves_the_rate() {
        let (limiter, t0) = fresh(config());
        limiter.wait_at(None, t0).unwrap();
        let retry = limiter.wait_at(throttled(), t0).unwrap();
        assert_eq!(rate(&limiter), 1.0);
        // The retry takes the slot booked at the old rate; the next is
        // spaced at the new one.
        assert!(close(retry, SECOND / 2));
        let next = limiter.wait_at(None, t0).unwrap();
        assert!(close(next, SECOND / 2 + SECOND));
    }

    #[test]
    fn throttles_within_a_cooldown_cut_once() {
        let (limiter, t0) = fresh(config());
        for i in 0..5 {
            let wait = limiter.wait_at(throttled(), t0 + SECOND * i);
            assert!(wait.is_ok());
        }
        assert_eq!(rate(&limiter), 1.0);
        let after = t0 + config().cooldown;
        assert!(limiter.wait_at(throttled(), after).is_ok());
        assert!(rate(&limiter) < 1.0);
    }

    #[test]
    fn the_rate_grows_while_busy_up_to_the_ceiling() {
        let (limiter, t0) = fresh(config());
        // Ten seconds of calls booked back to back: 0.1/s per second.
        let mut now = t0;
        while now < t0 + 10 * SECOND {
            now += limiter.wait_at(None, now).unwrap();
        }
        let grown = rate(&limiter);
        assert!(grown > 2.9 && grown < 3.1, "rate {grown}");
        while now < t0 + HOUR {
            now += limiter.wait_at(None, now).unwrap();
        }
        assert_eq!(rate(&limiter), config().ceiling);
    }

    #[test]
    fn idle_time_does_not_grow_the_rate() {
        let (limiter, t0) = fresh(config());
        limiter.wait_at(None, t0).unwrap();
        limiter.wait_at(None, t0 + HOUR).unwrap();
        // Only the half second the first call's slot was binding counts.
        let grown = rate(&limiter);
        assert!((grown - 2.05).abs() < 1e-9, "rate {grown}");
    }

    #[test]
    fn cuts_stop_at_the_floor() {
        let (limiter, t0) = fresh(AdaptiveConfig {
            increase: 0.0,
            ..config()
        });
        for i in 0..10 {
            assert!(limiter.wait_at(throttled(), t0 + HOUR * i).is_ok());
        }
        assert_eq!(rate(&limiter), config().floor);
    }

    #[test]
    fn retry_after_is_honoured_and_queues_later_calls() {
        let (limiter, t0) = fresh(config());
        let told = 30 * SECOND;
        let error: BoxError = Box::new(Status(429, Some(told)));
        assert!(close(limiter.wait_at(Some(error), t0).unwrap(), told));
        let next = limiter.wait_at(None, t0).unwrap();
        assert!(close(next, told + SECOND));
    }

    #[test]
    fn an_outage_is_paced_but_not_cut() {
        let (limiter, t0) = fresh(AdaptiveConfig {
            increase: 0.0,
            ..config()
        });
        limiter.wait_at(None, t0).unwrap();
        let retry = limiter.wait_at(outage(), t0).unwrap();
        assert_eq!(rate(&limiter), config().start);
        assert!(close(retry, SECOND / 2));
    }

    #[test]
    fn refusal_past_max_wait_hands_back_the_error() {
        let (limiter, t0) = fresh(AdaptiveConfig {
            max_wait: SECOND,
            ..config()
        });
        let error: BoxError = Box::new(Status(429, Some(HOUR)));
        let refused = limiter.wait_at(Some(error), t0).unwrap_err();
        assert!(refused.is::<Status>());
        let (limiter, t0) = fresh(AdaptiveConfig {
            max_wait: Duration::ZERO,
            ..config()
        });
        limiter.wait_at(None, t0).unwrap();
        assert!(limiter.wait_at(None, t0).unwrap_err().is::<Exhausted>());
    }
}
