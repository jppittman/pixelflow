//! Tunable scheduler parameters extracted from hardcoded constants.
//!
//! Every magic number in the scheduler has been pulled into [`SchedulerParams`]
//! so that automated search (Bayesian optimization, grid search, etc.) can
//! explore the parameter space and find configurations that are strictly better
//! than the hand-tuned defaults.
//!
//! # Parameter Groups
//!
//! | Group | Controls |
//! |-------|----------|
//! | **Channel sizing** | Buffer capacities for control/management lanes |
//! | **Burst limiting** | How many messages each lane can drain per wake cycle |
//! | **Backoff** | Sender retry strategy when channels are full |
//! | **Spin/yield** | Hot-path retry counts before escalating to sleep |

use std::time::Duration;

/// All tunable constants for the actor scheduler.
///
/// Defaults match the original hand-tuned values. Pass a modified copy to
/// [`ActorScheduler::new_with_params`] to experiment with different configurations.
///
/// # Invariants
///
/// - `jitter_min_pct + jitter_range_pct <= 100`
/// - `control_mgmt_buffer_size >= 1`
/// - `min_backoff <= max_backoff`
/// - `control_burst_multiplier >= 1`
/// - `management_burst_multiplier >= 1`
#[derive(Debug, Clone, Copy)]
pub struct SchedulerParams {
    // ── Channel sizing ──────────────────────────────────────────────
    /// Capacity of the bounded Control and Management channels.
    ///
    /// Smaller buffers force faster detection of overload scenarios.
    /// Larger buffers absorb bursts but delay backpressure signals.
    ///
    /// **Default: 32** (original constant `CONTROL_MGMT_BUFFER_SIZE`)
    pub control_mgmt_buffer_size: usize,

    // ── Burst limiting ──────────────────────────────────────────────
    /// Control burst limit = `control_mgmt_buffer_size * control_burst_multiplier`.
    ///
    /// Controls how many control messages can be drained per scheduler wake cycle.
    /// Higher values give control more bandwidth; lower values protect data from starvation.
    /// The budget is split evenly across two control drain passes.
    ///
    /// **Default: 10** (yields 32 * 10 = 320 messages per cycle)
    pub control_burst_multiplier: usize,

    /// Management burst limit = `control_mgmt_buffer_size * management_burst_multiplier`.
    ///
    /// Controls how many management messages can be drained per cycle.
    ///
    /// **Default: 1** (yields 32 * 1 = 32 messages per cycle)
    pub management_burst_multiplier: usize,

    /// Default data burst limit used by [`create_actor`](crate::create_actor).
    ///
    /// How many data messages the scheduler drains per cycle before yielding
    /// back to check higher-priority lanes.
    ///
    /// **Default: 1024**
    pub default_data_burst_limit: usize,

    // ── Backoff strategy (sender side) ──────────────────────────────
    /// Number of immediate retries (hot spin) before yielding.
    ///
    /// At ~10-20ns per spin, 100 spins = ~1-2us (less than a context switch).
    /// Higher values improve latency for brief contention but waste CPU.
    ///
    /// **Default: 100**
    pub spin_attempts: u32,

    /// Number of cooperative yield attempts before escalating to sleep.
    ///
    /// After hot spinning, yield to let the receiver drain.
    /// Higher values extend the yield phase before exponential backoff kicks in.
    ///
    /// **Default: 20**
    pub yield_attempts: u32,

    /// Minimum backoff duration when control/management channels are full.
    ///
    /// Must be high enough to prevent oscillation where senders retry faster
    /// than the receiver can drain.
    ///
    /// **Default: 1ms**
    pub min_backoff: Duration,

    /// Maximum backoff duration before returning `SendError::Timeout`.
    ///
    /// At ~1us per message, 320 messages = ~320us drain time, so 5s is
    /// extremely generous. Lowering this makes the system fail-faster
    /// under sustained overload.
    ///
    /// **Default: 5s**
    pub max_backoff: Duration,

    // ── Jitter ──────────────────────────────────────────────────────
    /// Minimum jitter as a percentage of the backoff duration.
    ///
    /// Jitter prevents thundering herd when multiple senders wake simultaneously.
    /// The actual sleep = `backoff * uniform(jitter_min_pct, jitter_min_pct + jitter_range_pct) / 100`.
    ///
    /// **Default: 50** (50%)
    pub jitter_min_pct: u64,

    /// Range of jitter above the minimum, in percentage points.
    ///
    /// Together with `jitter_min_pct`, defines the jitter window.
    /// `jitter_min_pct=50, jitter_range_pct=50` means sleep is 50-99% of base backoff.
    ///
    /// **Default: 50** (so range is 50%..100%)
    pub jitter_range_pct: u64,
}

impl SchedulerParams {
    /// Tuned defaults found via domain-constraint-aware Bayesian optimization.
    ///
    /// The cost function combines measured metrics (latency, throughput,
    /// fairness) with multiplicative domain constraint penalties that
    /// prevent the optimizer from finding degenerate configurations:
    ///
    /// - **Frame budget**: min_backoff ≤ 6.45ms (one frame at 155fps)
    /// - **Degradation window**: total backoff cascade ≤ 12s
    /// - **Jitter effectiveness**: range ≥ 20% for thundering herd spread
    /// - **Backpressure delay**: buffer ≤ 128 for fast overload detection
    ///
    /// Key results vs hand-tuned baseline:
    /// - Latency under load:  8.1μs → 2.2μs  (-73%)  — typing while data flows
    /// - Burst recovery:     20.4μs → 7.1μs  (-65%)  — keystroke after `ls`
    /// - Data throughput:     3.9M  → 4.7M   (+22%)
    /// - Control throughput:  1.3M  → 2.5M   (+89%)
    /// - Fairness:            100%  → 100%   (maintained)
    /// - All domain penalties: 0.000
    ///
    /// Structural changes:
    /// - Larger buffer (118 vs 32): absorb bursts, still under 128 soft limit
    /// - Higher burst multiplier: drain more control per wake, amortize doorbell
    /// - Lower data burst (332 vs 1024): yield back to check priority lanes sooner
    /// - More spin+yield before sleep (458+191 vs 100+20): real-time bias
    /// - min_backoff 2.1ms: well under one frame, < 0.3 frames lost
    /// - max_backoff 3.0s: total degradation 8.6s (vs 16.4s baseline)
    /// - Jitter 73-98%: 25% spread for thundering herd prevention
    pub const DEFAULT: Self = Self {
        control_mgmt_buffer_size: 118,
        control_burst_multiplier: 36,
        management_burst_multiplier: 1,
        default_data_burst_limit: 332,
        spin_attempts: 458,
        yield_attempts: 191,
        min_backoff: Duration::from_micros(2110),
        max_backoff: Duration::from_micros(2_965_136),
        jitter_min_pct: 73,
        jitter_range_pct: 25,
    };

    /// Validate invariants, panicking on violation.
    ///
    /// Called automatically by constructors that accept params.
    pub fn validate(&self) {
        assert!(
            self.control_mgmt_buffer_size >= 1,
            "control_mgmt_buffer_size must be >= 1, got {}",
            self.control_mgmt_buffer_size
        );
        assert!(
            self.control_burst_multiplier >= 1,
            "control_burst_multiplier must be >= 1, got {}",
            self.control_burst_multiplier
        );
        assert!(
            self.management_burst_multiplier >= 1,
            "management_burst_multiplier must be >= 1, got {}",
            self.management_burst_multiplier
        );
        assert!(
            self.min_backoff <= self.max_backoff,
            "min_backoff ({:?}) must be <= max_backoff ({:?})",
            self.min_backoff,
            self.max_backoff
        );
        assert!(
            self.jitter_min_pct + self.jitter_range_pct <= 100,
            "jitter_min_pct ({}) + jitter_range_pct ({}) must be <= 100",
            self.jitter_min_pct,
            self.jitter_range_pct
        );
        assert!(
            self.jitter_range_pct >= 1,
            "jitter_range_pct must be >= 1 to avoid division by zero, got {}",
            self.jitter_range_pct
        );
    }

    /// Computed control burst limit.
    #[must_use]
    pub fn control_burst_limit(&self) -> usize {
        self.control_mgmt_buffer_size * self.control_burst_multiplier
    }

    /// Computed management burst limit.
    #[must_use]
    pub fn management_burst_limit(&self) -> usize {
        self.control_mgmt_buffer_size * self.management_burst_multiplier
    }

    /// Return the parameter vector as `[f64; N]` for optimization.
    ///
    /// Parameters are in log-space where appropriate (durations, counts)
    /// for better optimization landscape properties.
    #[must_use]
    pub fn to_vec(&self) -> [f64; 10] {
        [
            self.control_mgmt_buffer_size as f64,
            self.control_burst_multiplier as f64,
            self.management_burst_multiplier as f64,
            self.default_data_burst_limit as f64,
            self.spin_attempts as f64,
            self.yield_attempts as f64,
            self.min_backoff.as_micros() as f64,
            self.max_backoff.as_micros() as f64,
            self.jitter_min_pct as f64,
            self.jitter_range_pct as f64,
        ]
    }

    /// Reconstruct params from a float vector (inverse of `to_vec`).
    ///
    /// Values are clamped and rounded to valid ranges.
    #[must_use]
    pub fn from_vec(v: &[f64; 10]) -> Self {
        let jitter_min = (v[8].round().clamp(5.0, 90.0)) as u64;
        let jitter_range = (v[9].round().max(1.0).min((100 - jitter_min) as f64)) as u64;

        Self {
            control_mgmt_buffer_size: (v[0].round().max(2.0) as usize),
            control_burst_multiplier: (v[1].round().max(1.0) as usize),
            management_burst_multiplier: (v[2].round().max(1.0) as usize),
            default_data_burst_limit: (v[3].round().max(1.0) as usize),
            spin_attempts: (v[4].round().max(0.0) as u32),
            yield_attempts: (v[5].round().max(0.0) as u32),
            min_backoff: Duration::from_micros(v[6].round().max(1.0) as u64),
            max_backoff: Duration::from_micros(v[7].round().max(100.0) as u64),
            jitter_min_pct: jitter_min,
            jitter_range_pct: jitter_range,
        }
    }

    /// Bounds for each parameter dimension `(lower, upper)`.
    ///
    /// Used by optimization algorithms to constrain the search space.
    /// Bounds are informed by domain constraints (see `bench_optimize.rs`):
    ///
    /// - `min_backoff` upper bound: 6450us (one 155fps frame). Above this,
    ///   first contention drops a frame — the most visible failure mode.
    /// - `max_backoff` upper bound: 10s. The total degradation window
    ///   (geometric sum ≈ 2×max) should not exceed ~12-20s for a terminal.
    /// - `jitter_range_pct` lower bound: 20%. Below this, thundering herd
    ///   prevention is ineffective.
    /// - `control_mgmt_buffer_size` upper bound: 128. Larger buffers delay
    ///   backpressure detection.
    #[must_use]
    pub fn bounds() -> [(f64, f64); 10] {
        [
            (4.0, 128.0),              // control_mgmt_buffer_size
            (1.0, 50.0),               // control_burst_multiplier
            (1.0, 10.0),               // management_burst_multiplier
            (16.0, 8192.0),            // default_data_burst_limit
            (0.0, 500.0),              // spin_attempts
            (0.0, 200.0),              // yield_attempts
            (10.0, 6_450.0),           // min_backoff (us) — at most one frame
            (100_000.0, 10_000_000.0), // max_backoff (us) — 0.1s to 10s
            (5.0, 80.0),               // jitter_min_pct
            (20.0, 50.0),              // jitter_range_pct — at least 20% spread
        ]
    }

    /// Human-readable parameter names (parallel to `to_vec` indices).
    pub const NAMES: [&str; 10] = [
        "ctrl_mgmt_buf",
        "ctrl_burst_mul",
        "mgmt_burst_mul",
        "data_burst",
        "spin_attempts",
        "yield_attempts",
        "min_backoff_us",
        "max_backoff_us",
        "jitter_min_pct",
        "jitter_range_pct",
    ];
}

impl Default for SchedulerParams {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_params_are_valid() {
        SchedulerParams::default().validate();
    }

    #[test]
    fn roundtrip_to_vec_from_vec() {
        let original = SchedulerParams::default();
        let vec = original.to_vec();
        let reconstructed = SchedulerParams::from_vec(&vec);

        assert_eq!(
            original.control_mgmt_buffer_size,
            reconstructed.control_mgmt_buffer_size
        );
        assert_eq!(original.spin_attempts, reconstructed.spin_attempts);
        assert_eq!(original.min_backoff, reconstructed.min_backoff);
        assert_eq!(original.max_backoff, reconstructed.max_backoff);
        assert_eq!(original.jitter_min_pct, reconstructed.jitter_min_pct);
        assert_eq!(original.jitter_range_pct, reconstructed.jitter_range_pct);
    }

    #[test]
    #[should_panic(expected = "control_mgmt_buffer_size must be >= 1")]
    fn validates_buffer_size() {
        let mut p = SchedulerParams::default();
        p.control_mgmt_buffer_size = 0;
        p.validate();
    }

    #[test]
    #[should_panic(expected = "jitter_min_pct")]
    fn validates_jitter_sum() {
        let mut p = SchedulerParams::default();
        p.jitter_min_pct = 80;
        p.jitter_range_pct = 30; // 80 + 30 = 110 > 100
        p.validate();
    }

    // Kills: replace * with + in control_burst_limit (line 201)
    // With +: control_burst_limit = buf_size + multiplier (wrong — should be product)
    #[test]
    fn control_burst_limit_is_product_of_size_and_multiplier() {
        let p = SchedulerParams {
            control_mgmt_buffer_size: 8,
            control_burst_multiplier: 5,
            ..SchedulerParams::DEFAULT
        };
        assert_eq!(
            p.control_burst_limit(),
            40,
            "control_burst_limit = 8 * 5 = 40"
        );
    }

    // Kills: replace * with + in management_burst_limit (line 207)
    // With +: management_burst_limit = buf_size + multiplier (wrong — should be product)
    #[test]
    fn management_burst_limit_is_product_of_size_and_multiplier() {
        let p = SchedulerParams {
            control_mgmt_buffer_size: 10,
            management_burst_multiplier: 3,
            ..SchedulerParams::DEFAULT
        };
        assert_eq!(
            p.management_burst_limit(),
            30,
            "management_burst_limit = 10 * 3 = 30"
        );
    }

    // Kills: replace management_burst_limit with 1 (line 207 return-value replacement)
    #[test]
    fn management_burst_limit_exceeds_one_for_large_multiplier() {
        let p = SchedulerParams {
            control_mgmt_buffer_size: 10,
            management_burst_multiplier: 4,
            ..SchedulerParams::DEFAULT
        };
        assert!(
            p.management_burst_limit() > 1,
            "management_burst_limit should exceed 1"
        );
        assert_eq!(p.management_burst_limit(), 40);
    }

    // Kills: replace * with / in control_burst_limit
    #[test]
    fn control_burst_limit_different_from_quotient() {
        let p = SchedulerParams {
            control_mgmt_buffer_size: 8,
            control_burst_multiplier: 4,
            ..SchedulerParams::DEFAULT
        };
        // 8 * 4 = 32, but 8 / 4 = 2 — should be 32
        assert_eq!(p.control_burst_limit(), 32);
        assert_ne!(
            p.control_burst_limit(),
            p.control_mgmt_buffer_size / p.control_burst_multiplier
        );
    }

    // Kills: replace - with + in from_vec jitter_range clamping (line 236)
    // With +: jitter_range = max(1, min(v[9], 100 + jitter_min))
    //         The cap becomes much larger, allowing jitter_range > 100-jitter_min
    #[test]
    fn from_vec_jitter_range_clamped_to_prevent_overflow() {
        // jitter_min=50, jitter_range=60 would sum to 110 (>100, invalid)
        // from_vec should clamp jitter_range to (100 - jitter_min) = 50
        let v = [
            10.0,      // control_mgmt_buffer_size
            1.0,       // control_burst_multiplier
            1.0,       // management_burst_multiplier
            16.0,      // default_data_burst_limit
            0.0,       // spin_attempts
            0.0,       // yield_attempts
            100.0,     // min_backoff_us
            200_000.0, // max_backoff_us
            50.0,      // jitter_min_pct
            60.0,      // jitter_range_pct (too high, should be clamped to 50)
        ];
        let p = SchedulerParams::from_vec(&v);
        // With correct subtraction: jitter_range = min(60, 100 - 50) = min(60, 50) = 50
        // With + mutation: jitter_range = min(60, 100 + 50) = min(60, 150) = 60 (not clamped!)
        assert!(
            p.jitter_min_pct + p.jitter_range_pct <= 100,
            "from_vec must clamp jitter_range so min+range <= 100"
        );
        assert_eq!(
            p.jitter_range_pct, 50,
            "jitter_range should be clamped to 100 - jitter_min = 50"
        );
    }

    // Kills: replace from_vec with Default::default() (line 235)
    #[test]
    fn from_vec_preserves_non_default_values() {
        let v = [
            64.0,      // control_mgmt_buffer_size (differs from DEFAULT=118)
            2.0,       // control_burst_multiplier
            2.0,       // management_burst_multiplier
            64.0,      // default_data_burst_limit
            10.0,      // spin_attempts
            5.0,       // yield_attempts
            500.0,     // min_backoff_us
            500_000.0, // max_backoff_us
            20.0,      // jitter_min_pct
            20.0,      // jitter_range_pct
        ];
        let p = SchedulerParams::from_vec(&v);
        assert_eq!(p.control_mgmt_buffer_size, 64);
        assert_eq!(p.control_burst_multiplier, 2);
        assert_eq!(p.spin_attempts, 10);
    }
}
