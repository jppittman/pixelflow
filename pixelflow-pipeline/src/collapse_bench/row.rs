//! One measurement: a kernel, at a shape, on a tier, built from a git ref.
//!
//! Rows are the whole interface between the bench and the analysis, so they
//! carry everything either side needs and nothing that changes between two
//! runs of the same build — no timestamps, no paths, no host name. Two runs
//! of one binary differ only in the numbers that were measured, which is what
//! makes a re-run a diff.

use serde::{Deserialize, Serialize};

/// Bumped whenever a field's meaning changes, so a stale file is rejected
/// rather than silently mixed into an analysis.
pub const SCHEMA: &str = "collapse-cost-v1";

/// Emitted traffic in one scope of the collapse nest.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopeRow {
    pub bytes: u32,
    pub instructions: u32,
    pub loads_transient: u32,
    pub loads_kept: u32,
    pub remats: u32,
    pub stores: u32,
}

impl ScopeRow {
    /// Loads plus stores: the quantity the rejected allocator policies
    /// optimized.
    #[must_use]
    pub const fn memory_ops(&self) -> u32 {
        self.loads_transient + self.loads_kept + self.stores
    }
}

/// Everything a predictor is allowed to read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticFeatures {
    /// Bytes of the whole emitted function, scaffold included.
    pub bytes_total: u32,
    pub frame: ScopeRow,
    pub row: ScopeRow,
    pub body: ScopeRow,
    /// The collapse scaffold's own traffic. Identical under every allocation
    /// of a kernel, so it is kept out of the derived dynamic figures below and
    /// recorded here instead.
    pub scaffold: ScopeRow,
    /// Frame slots holding a value with an address (`FrameLayout::slots`).
    pub spill_slots: u32,
    /// Stack bytes the frame occupies.
    pub frame_bytes: u32,
    /// Values LICM lifted out of the body.
    pub hoisted: u32,
    /// Of those, the ones holding a register across the loops inside.
    pub carried: u32,
    /// Registers the allocator could hand out.
    pub pool: u32,
    /// Bytes one spilled register occupies — 16 for NEON, 32 for AVX2, 64 for AVX-512.
    pub vector_bytes: u32,
    /// Σ over scopes of (memory ops in scope × executions per call).
    pub dyn_memory_ops: u64,
    /// The same weighting applied to instruction counts.
    pub dyn_instructions: u64,
    /// The same weighting applied to code bytes.
    pub dyn_bytes: u64,
}

/// The timing half of a row.
///
/// Two estimators are recorded because they do not agree on how reproducible
/// they are, and which one to trust is a measurement rather than a taste:
/// across three passes of this corpus the per-sample **minimum** reproduced to
/// 1.2% (median kernel) against the median's 1.5%, and the drift-corrected
/// median to 2.0% — i.e. the correction made the numbers *less* reproducible
/// than the numbers it corrected. So [`Self::ns_median`] is the raw local
/// clock and the correction rides alongside as data.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    /// Median ns per `call_collapse`, on the clock that was running. The
    /// headline figure.
    pub ns_median: f64,
    /// Fastest sample — the same measurement with the scheduler's
    /// interruptions taken off one side.
    pub ns_min: f64,
    /// Interquartile range of the per-call samples: this row's own spread.
    pub ns_iqr: f64,
    /// `ns_median` re-expressed in the session's opening clock via the
    /// sentinel. Recorded, not used: see the type's doc.
    pub ns_median_drift_corrected: f64,
    /// `calibration / local_sentinel` at the moment this row was taken.
    pub drift: f64,
    /// The session's opening sentinel reading. The sentinel is the same tiny
    /// kernel on every build, so a difference here between two runs is the
    /// machine, not the allocator — which is what makes cross-run comparison
    /// auditable rather than assumed.
    pub sentinel_calibration_ns: f64,
    /// Bytes of the sentinel's own emitted code. If this differs between two
    /// refs, the sentinel is not the same anchor on both and the calibration
    /// above cannot be compared.
    pub sentinel_bytes: u32,
    pub samples: u32,
    /// Calls the timer bracketed per sample, after the dispersion autoscale.
    pub calls_per_sample: u64,
}

/// One (kernel, tier, ref, pass) measurement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Row {
    pub schema: String,
    /// Human name of the allocation variant this build came from.
    pub git_ref: String,
    /// The commit it was built at.
    pub git_sha: String,
    /// `release` or `bench`.
    pub profile: String,
    /// `sse2`, `avx2`, `avx512` or `neon`.
    pub tier: String,
    /// Which sweep of the corpus this row came from; two passes of one build
    /// are this harness's A/A floor.
    pub pass: u32,
    pub kernel: String,
    pub family: String,
    pub extent: [u32; 2],
    pub lanes: u32,
    pub rows: u64,
    pub groups: u64,
    pub measured: Measurement,
    pub statics: StaticFeatures,
}

/// Which timing estimator an analysis reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stat {
    /// [`Measurement::ns_median`].
    Median,
    /// [`Measurement::ns_min`].
    Min,
    /// [`Measurement::ns_median_drift_corrected`].
    DriftCorrected,
}

impl Stat {
    /// Parse the `--stat` flag.
    ///
    /// # Errors
    /// The name, if it is not one of the three.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "median" => Ok(Self::Median),
            "min" => Ok(Self::Min),
            "drift" => Ok(Self::DriftCorrected),
            other => Err(format!(
                "unknown statistic {other:?}: expected median, min or drift"
            )),
        }
    }

    #[must_use]
    pub fn of(self, m: &Measurement) -> f64 {
        match self {
            Self::Median => m.ns_median,
            Self::Min => m.ns_min,
            Self::DriftCorrected => m.ns_median_drift_corrected,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Median => "median",
            Self::Min => "min",
            Self::DriftCorrected => "drift-corrected median",
        }
    }
}
