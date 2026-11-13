//! Label minting: turning a [`BenchResult`] into a training target, and
//! recording the conditions the targets were minted under.
//!
//! [`normalized_label_ns`] applies the drift correction the sentinel
//! measured — to the raw reading, before the opening-clock call overhead
//! comes off, because those two numbers are denominated in different
//! clocks (see that function). `BenchSession` records `calibration_ns /
//! local_sentinel_ns` on every [`BenchResult`], but recording a correction
//! is not applying one: feeding `adjusted_ns` to `log_ns` leaves timing drift
//! correlated with *collection order*, and collection order correlates
//! with expression structure (generation emits size/family bands
//! sequentially). Drift then stops being noise and becomes learnable
//! spurious signal. Every label-minting path must go through this
//! function, and a [`BenchResult`] with no sentinel (the sessionless
//! wrappers) is a hard error there — an unprotected measurement silently
//! accepted as a label is exactly the failure the sentinel exists to make
//! impossible.
//!
//! The weights sidecar (the objective/identity record written next to an
//! extraction-head checkpoint) left with the extraction-head
//! program on 2026-09-01; see
//! docs/compiler/2026-09-01-schedule-cost-model-denotation.md (which cites the
//! closed workshop paper, PR #1072, branch `claude/workshop-writeup`) and
//! git history for it.
//!
//! Both functions below are thin `f64` wrappers around
//! [`BenchResult::corrected_session_ns`] / [`crate::jit_bench::CostLabel`]
//! (docs/archive/plans/2026-08-17-cost-model-domain.md, J3/J4): the ordering they
//! document is enforced there by the [`crate::jit_bench::LocalNs`] /
//! [`crate::jit_bench::SessionNs`] newtypes, not merely described here in
//! prose. Callers that track collection order should mint a
//! [`crate::jit_bench::CostLabel`] directly instead of going through these —
//! they exist for callers that only need the scalar value.

use serde::{Deserialize, Serialize};

use crate::jit_bench::BenchResult;

// ── Sentinel normalization ──────────────────────────────────────────────────

/// The drift-corrected nanosecond value of a measurement, in the run's
/// *opening* clock.
///
/// # Order of operations
///
/// Normalize the **raw** reading, *then* subtract the call overhead:
///
/// ```text
/// label = ns * (calibration_ns / local_sentinel_ns) - call_overhead_ns
/// ```
///
/// The two operands live on different clocks and only one order puts them on
/// the same one. `call_overhead_ns` is the identity-kernel cost measured once
/// at session open, so it is denominated in the *opening* clock;
/// `BenchResult::ns` is denominated in the clock that was running when the
/// kernel was timed. The sentinel factor converts the second into the first,
/// which makes it a conversion on `ns` alone — `adjusted_ns` is a mixture of
/// the two clocks and scaling it converts neither correctly.
///
/// Scaling `adjusted_ns` instead (the original form) leaves a residue. With a
/// slowdown factor `d`, the raw reading is `d * (kernel + overhead)` and the
/// factor is `1/d`, so
///
/// ```text
/// (d * (kernel + overhead) - overhead) / d  =  kernel + overhead * (1 - 1/d)
/// ```
///
/// — a term that depends on `d`, i.e. on *when in the run* the expression was
/// benchmarked. That is precisely the order-dependent contamination the
/// sentinel exists to remove, and at the accepted sub-50% drift band with a
/// ~4ns call overhead it is >1ns, which is a large fraction of the smallest
/// kernels — the bulk of the corpus. This function no longer has an
/// opportunity to get that order wrong: it delegates to
/// [`BenchResult::corrected_session_ns`], where the raw reading is a
/// [`crate::jit_bench::LocalNs`] with no `Sub` impl, so overhead cannot be
/// taken off before normalization runs.
///
/// The alternative, remeasuring the identity kernel next to every label and
/// normalizing that too, is rejected: it buys the same number at the price of
/// a benchmark per label plus that benchmark's own noise, when the sentinel
/// factor already carries exactly the clock-speed information needed to move
/// the one operand that is on the wrong clock.
///
/// The sign of the result is not clamped: a value `<= 0` is a documented
/// measurement condition for kernels cheaper than the identity call overhead.
///
/// # Panics
///
/// Panics when `bench.sentinel` is `None`. That means the measurement came
/// from a sessionless wrapper which runs no sentinel at all, so there is no
/// drift information to correct with and no drift protection behind the
/// number. Minting a label from it would silently mix an unprotected
/// measurement into a normalized set — the correction would be skipped, not
/// unnecessary. `context` names the call site in the panic.
#[must_use]
pub fn normalized_label_ns(bench: &BenchResult, context: &str) -> f64 {
    bench.corrected_session_ns(context).0.get()
}

/// The sentinel normalization factor a measurement carries, for persisting
/// alongside the label so training can down-weight heavily-corrected samples.
///
/// # Panics
///
/// Same contract as [`normalized_label_ns`]: a missing sentinel is a hard
/// error in label-minting paths.
#[must_use]
pub fn label_normalization(bench: &BenchResult, context: &str) -> f64 {
    bench.corrected_session_ns(context).1.get()
}

/// Summary statistics of the per-label normalization factors a run applied.
///
/// A run whose factors are all ~1.0 saw no drift; a run whose factors span a
/// wide range drifted a lot and its labels lean hard on the correction. Both
/// are reportable facts.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct NormalizationStats {
    pub count: usize,
    pub min: f64,
    pub median: f64,
    pub max: f64,
}

impl NormalizationStats {
    /// Summarize the factors applied across a run.
    ///
    /// # Panics
    ///
    /// Panics on an empty slice (a run that minted no labels has no stats to
    /// report and must not fabricate any) or on a non-finite factor.
    #[must_use]
    pub fn summarize(factors: &[f64]) -> Self {
        assert!(
            !factors.is_empty(),
            "NormalizationStats::summarize: no normalization factors — a run that minted zero \
             labels has no drift summary"
        );
        let mut sorted = factors.to_vec();
        sorted.sort_by(|a, b| {
            a.partial_cmp(b)
                .unwrap_or_else(|| panic!("non-finite normalization factor: {a} vs {b}"))
        });
        Self {
            count: sorted.len(),
            min: sorted[0],
            median: sorted[sorted.len() / 2],
            max: sorted[sorted.len() - 1],
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit_bench::{BenchMode, SentinelContext};

    /// A `BenchResult` shaped only for the fields these tests exercise, with
    /// no call overhead (`adjusted_ns == ns`).
    fn bench(adjusted_ns: f64, sentinel: Option<SentinelContext>) -> BenchResult {
        bench_raw(adjusted_ns, 0.0, sentinel)
    }

    /// A `BenchResult` as a `BenchSession` builds one: `ns` is the raw reading
    /// on whatever clock was running, `call_overhead_ns` is the identity
    /// measurement taken at session open, and `adjusted_ns` is their
    /// difference — the cross-clock mixture that must not be scaled.
    fn bench_raw(ns: f64, call_overhead_ns: f64, sentinel: Option<SentinelContext>) -> BenchResult {
        BenchResult {
            ns,
            adjusted_ns: ns - call_overhead_ns,
            call_overhead_ns,
            iqr_ns: 0.0,
            mode: BenchMode::Latency,
            repeat_batches: 1,
            output: [0.0; 4],
            outputs: Vec::new(),
            sentinel,
        }
    }

    /// The measurement a machine running `drift`x slower than the opening
    /// clock produces for a kernel whose true cost is `kernel_ns`, given an
    /// identity call overhead of `overhead_ns` measured at session open.
    ///
    /// The whole clock inflates, so the raw reading covers kernel *and*
    /// overhead: `drift * (kernel + overhead)`.
    fn drifted(kernel_ns: f64, overhead_ns: f64, drift: f64) -> BenchResult {
        const CALIBRATION_NS: f64 = 8.0;
        bench_raw(
            (kernel_ns + overhead_ns) * drift,
            overhead_ns,
            Some(SentinelContext {
                calibration_ns: CALIBRATION_NS,
                local_sentinel_ns: CALIBRATION_NS * drift,
            }),
        )
    }

    /// The property the whole normalization design exists for: with a nonzero
    /// call overhead in play, the minted label is the true kernel time and
    /// does not depend on the drift factor at all.
    ///
    /// `d = 1.0` is the identity case; the rest sit inside the accepted
    /// ±50% band (`SENTINEL_REGIME_CHANGE_FRAC`), on both sides of it.
    #[test]
    fn label_is_the_true_kernel_time_independent_of_drift() {
        // A ~4ns call overhead against a small kernel is the regime where the
        // order-of-operations residue was largest, and small kernels are most
        // of the corpus.
        const OVERHEAD_NS: f64 = 4.0;
        for kernel_ns in [0.5f64, 3.0, 120.0] {
            for drift in [1.0f64, 0.62, 0.85, 1.2, 1.49] {
                let label = normalized_label_ns(&drifted(kernel_ns, OVERHEAD_NS, drift), "test");
                assert!(
                    (label - kernel_ns).abs() < 1e-9,
                    "kernel {kernel_ns}ns measured under {drift}x drift minted {label}ns — the \
                     label must be the drift-free kernel time, not a function of when in the \
                     run it was benchmarked"
                );
            }
        }
    }

    /// The bias this ordering was chosen to remove, pinned as arithmetic so
    /// the swap cannot silently come back: scaling `adjusted_ns` leaves
    /// `overhead * (1 - 1/d)`, which is >1ns for a 4ns overhead well inside
    /// the accepted drift band.
    #[test]
    fn scaling_the_adjusted_value_would_leave_a_drift_dependent_residue() {
        const OVERHEAD_NS: f64 = 4.0;
        const KERNEL_NS: f64 = 3.0;
        let drift = 1.45f64;
        let b = drifted(KERNEL_NS, OVERHEAD_NS, drift);
        let factor = b.sentinel.expect("sentinel").normalization().get();

        let wrong = b.adjusted_ns * factor;
        let residue = wrong - KERNEL_NS;
        assert!(
            (residue - OVERHEAD_NS * (1.0 - 1.0 / drift)).abs() < 1e-9,
            "the old form's error must be exactly overhead*(1 - 1/d), got {residue}"
        );
        assert!(
            residue > 1.0,
            "the residue must be big enough to matter against a {KERNEL_NS}ns kernel, got \
             {residue}ns"
        );
        assert!((normalized_label_ns(&b, "test") - KERNEL_NS).abs() < 1e-9);
    }

    /// Same expression, two collection positions, nonzero overhead: the two
    /// labels must agree. Order invariance is what the correction buys, and
    /// the previous ordering broke it precisely when overhead was nonzero.
    #[test]
    fn labels_agree_across_collection_positions_with_overhead() {
        const OVERHEAD_NS: f64 = 4.0;
        let early = normalized_label_ns(&drifted(2.0, OVERHEAD_NS, 1.0), "test");
        let late = normalized_label_ns(&drifted(2.0, OVERHEAD_NS, 1.4), "test");
        assert!((early - late).abs() < 1e-9, "{early} vs {late}");
    }

    #[test]
    fn normalization_rescales_into_the_opening_clock() {
        // The machine slowed to 1.25x its opening speed: a 125ns reading is
        // a 100ns expression.
        let b = bench(
            125.0,
            Some(SentinelContext {
                calibration_ns: 10.0,
                local_sentinel_ns: 12.5,
            }),
        );
        assert!((normalized_label_ns(&b, "test") - 100.0).abs() < 1e-9);
    }

    #[test]
    fn normalization_preserves_nonpositive_adjusted() {
        // A kernel whose ILP hides inside the identity call overhead: the raw
        // reading is a physical 16ns on a 2x-slowed clock (8ns in the opening
        // clock) against a 10ns opening overhead. The documented `<= 0`
        // condition must survive as `<= 0` rather than being clamped, flipped,
        // or turned into a different negative number by the ordering.
        let b = bench_raw(
            16.0,
            10.0,
            Some(SentinelContext {
                calibration_ns: 10.0,
                local_sentinel_ns: 20.0,
            }),
        );
        assert!((normalized_label_ns(&b, "test") - (-2.0)).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "no sentinel context")]
    fn missing_sentinel_is_a_loud_error_not_a_skipped_correction() {
        let _refused = normalized_label_ns(&bench(42.0, None), "unit test");
    }

    /// The point of item 6: with a drift profile applied to the measurements,
    /// the *normalized* targets must not depend on which order the
    /// expressions were benchmarked in.
    #[test]
    fn normalized_targets_are_order_invariant_under_drift() {
        // Three expressions with distinct true costs.
        let truth = [10.0f64, 40.0, 90.0];
        const OVERHEAD_NS: f64 = 4.0;
        // A monotone slowdown over the run: the machine gets 1.0x, 1.3x,
        // 1.45x slower at successive collection positions.
        let drift = [1.0f64, 1.3, 1.45];

        let measure_in_order = |order: [usize; 3]| -> Vec<(usize, f64)> {
            order
                .iter()
                .enumerate()
                .map(|(position, &expr)| {
                    let b = drifted(truth[expr], OVERHEAD_NS, drift[position]);
                    (expr, normalized_label_ns(&b, "test"))
                })
                .collect()
        };

        let forward = measure_in_order([0, 1, 2]);
        let reversed = measure_in_order([2, 1, 0]);
        let interleaved = measure_in_order([1, 2, 0]);

        for orderings in [&reversed, &interleaved] {
            for &(expr, value) in orderings.iter() {
                let (_, base) = forward
                    .iter()
                    .find(|(e, _)| *e == expr)
                    .expect("same expressions in every ordering");
                assert!(
                    (value - base).abs() < 1e-9,
                    "expression {expr} minted {value} in one order and {base} in another — \
                     drift is still correlated with collection order"
                );
                assert!((value - truth[expr]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn unnormalized_targets_are_order_dependent() {
        // The control: without the correction, the same expression measured
        // early and late disagrees. This is what item 6 is fixing, so it is
        // pinned rather than assumed.
        let early = drifted(10.0, 4.0, 1.0);
        let late = drifted(10.0, 4.0, 1.45);
        assert!((early.adjusted_ns - late.adjusted_ns).abs() > 1.0);
        assert!(
            (normalized_label_ns(&early, "t") - normalized_label_ns(&late, "t")).abs() < 1e-9,
            "normalization must reconcile them"
        );
    }

    #[test]
    fn normalization_stats_summarize() {
        let s = NormalizationStats::summarize(&[1.0, 0.5, 2.0]);
        assert_eq!(s.count, 3);
        assert!((s.min - 0.5).abs() < 1e-12);
        assert!((s.median - 1.0).abs() < 1e-12);
        assert!((s.max - 2.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "no normalization factors")]
    fn normalization_stats_refuse_empty() {
        let _refused = NormalizationStats::summarize(&[]);
    }
}
