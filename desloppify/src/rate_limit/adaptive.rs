//! A discovered budget: AIMD on the call rate.

use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use super::{AdaptiveConfig, BoxError, Classify, Clock, Exhausted, RateLimiter, Signal};

/// The share of the rate kept after a throttle. Halving is TCP Reno's choice:
/// growth is slow and additive, so a throttle means the rate has only just
/// passed the limit, and half of it is safely under. AIMD with any factor
/// below one converges; a gentler one recovers faster but takes more 429s to
/// get under a limit that dropped.
const DECREASE: f64 = 0.5;

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
pub(super) struct Adaptive<C> {
    config: AdaptiveConfig,
    classify: Classify,
    clock: C,
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

impl<C: Clock> Adaptive<C> {
    pub(super) fn new(config: AdaptiveConfig, classify: Classify, clock: C) -> Self {
        assert!(config.floor > 0.0, "adaptive rate floor must be positive");
        assert!(
            config.ceiling >= config.floor,
            "adaptive rate ceiling below its floor"
        );
        let now = clock.now();
        let state = Mutex::new(Pace {
            rate: config.start.clamp(config.floor, config.ceiling),
            next: now,
            credited: now,
            calm_from: now,
        });
        Self {
            config,
            classify,
            clock,
            state,
        }
    }
}

impl<C: Clock> RateLimiter for Adaptive<C> {
    fn wait(&self, last: Option<BoxError>) -> Result<Duration, BoxError> {
        let now = self.clock.now();
        // The pace is plain numbers and instants, valid after any panic.
        let mut pace = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        pace.grow(now, &self.config);

        let signal = last.as_deref().map(|error| (self.classify)(error));
        if let Some(Signal::Throttled { .. }) = signal {
            pace.cut(now, &self.config);
        }
        let not_before = now + signal.and_then(retry_after).unwrap_or_default();
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

fn retry_after(signal: Signal) -> Option<Duration> {
    match signal {
        Signal::Throttled { retry_after } | Signal::Failed { retry_after } => retry_after,
    }
}
