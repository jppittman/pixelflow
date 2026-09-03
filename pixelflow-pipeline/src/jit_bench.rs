//! Shared JIT benchmarking infrastructure.
//!
//! Shared helpers for timing arena-native JIT kernels, plus [`BenchSession`],
//! which owns measurement integrity for label-minting runs: QoS pinning,
//! DVFS burn-in, a drift sentinel, identity-kernel call-overhead subtraction,
//! and per-expression plausibility floors. Findings referenced as H#/M#/L#
//! come from `docs/results/2026-08-05-bench-harness-integrity-audit.md`.

use std::fmt;

use pixelflow_codegen::emit::compile;
use pixelflow_codegen::emit::executable::ExecutableCode;
use pixelflow_codegen::error::CompileError;
use pixelflow_ir::{ExprArena, ExprId, OpKind};

/// Number of timed samples per expression. Take the median.
const TIMED_RUNS: usize = 20;

/// Warmup passes over the input buffer before timed runs (each pass is
/// [`INPUT_TUPLES`] evals). Warms icache, branch predictor, and
/// microarchitectural state so timed samples measure hot performance — not
/// cold-start overhead. DVFS ramp-up is NOT this constant's job: that is the
/// session-level burn-in ([`BURN_IN_NS`], audit L5).
const WARMUP_PASSES: usize = 8;

/// Number of distinct coordinate tuples cycled through per timed batch
/// (audit H3: a single fixed input hides all data-dependent cost and narrows
/// the equivalence check to one point). Must match the index list in
/// `with_all_inputs!`.
pub const INPUT_TUPLES: usize = 64;

/// The timed loop must accumulate at least this many nanoseconds per sample
/// (audit H5): one `mach_absolute_time` tick is 41.67ns on Apple Silicon
/// (timebase 125/3, see `nanos_now`), so a sample below ~100 ticks (~4.2µs)
/// carries double-digit-percent quantization error. `repeat_batches`
/// auto-scales until the median sample clears this floor.
const MIN_TIMED_SAMPLE_NS: u64 = 4_167;

/// Upper bound for the `repeat_batches` autoscale. At the plausibility floor
/// of 0.05ns/eval, `65_536 × 64` evals is ~210µs — far above
/// [`MIN_TIMED_SAMPLE_NS`] — so a median still below the tick floor at this
/// cap means the timer or the emitted code is broken, and the harness panics.
const MAX_REPEAT_BATCHES: usize = 65_536;

/// Session burn-in duration (audit L5): run real work this long before
/// calibrating anything, so DVFS has ramped to steady state. Warmup loops
/// (~1µs) are far too short for frequency ramp; this is not.
const BURN_IN_NS: u64 = 10_000_000;

/// Re-measure the sentinel kernel after this many benchmarked expressions
/// (audit H4).
const SENTINEL_INTERVAL: usize = 64;

/// Sentinel drift beyond which the session aborts: a **regime-change
/// detector, not a noise gate** (audit H4, revised per the 2026-08-08 plan
/// update). The drift policy is three-tiered, and only the last tier aborts:
///
/// - **Contention noise** (daemons, interrupts) is zero-mean and is absorbed
///   the same way more runs absorb it: corpus volume plus randomized
///   benchmark order decorrelate it from expression structure, so it widens
///   error bars without biasing the model. No action.
/// - **Slow one-signed drift** (thermal throttling, background load ramping)
///   is *measured, not fatal*: every batch records the local sentinel next
///   to the run's opening calibration ([`SentinelContext`]), so post-hoc
///   consumers apply `calibration_ns / local_sentinel_ns` as a running
///   clock-speed correction instead of discarding the run.
/// - **A step-function regime change** — the scheduler parking the run on an
///   E-core is the canonical case — is a 2–3× shift that genuinely poisons
///   every label after it and cannot be corrected by a scalar factor
///   calibrated on P-core behavior. That is what this threshold detects:
///   ±50% is far beyond anything contention or thermals produce (macOS's own
///   post-build daemons measured 11–19%) and far below the 2–3× E-core step,
///   so it fires exactly on the class that must abort.
const SENTINEL_REGIME_CHANGE_FRAC: f64 = 0.50;

/// Conservative per-op lower bound on plausible cost (audit M4). An M-series
/// core peaks at ~4 SIMD FP ops/cycle ≈ 0.078ns/op at 3.2GHz, so 0.05ns/op
/// sits safely below any real execution while still catching a JIT bug that
/// emits an early `ret`, a mis-scaled timebase, or an unroll miscount.
const MIN_NS_PER_OP: f64 = 0.05;

/// Maximum plausible single-eval time: 1 second.
/// Anything above this is a timing artifact (e.g., u64 underflow in
/// `nanos_now() - start`, OS scheduling jitter, or JIT codegen bug).
const MAX_PLAUSIBLE_NS: f64 = 1_000_000_000.0;

/// Errors from JIT benchmarking.
#[derive(Debug)]
pub enum BenchError {
    /// JIT compilation failed.
    CompileFailed(CompileError),
    /// Architecture not supported for JIT.
    UnsupportedArch,
    /// Measurement was invalid (NaN, negative, or absurdly large).
    InvalidMeasurement(f64),
}

impl fmt::Display for BenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BenchError::CompileFailed(msg) => write!(f, "compile failed: {}", msg),
            BenchError::UnsupportedArch => {
                write!(f, "unsupported architecture for JIT benchmarking")
            }
            BenchError::InvalidMeasurement(v) => write!(f, "invalid measurement: {}ns", v),
        }
    }
}

// Platform-specific high-resolution timing.

#[cfg(target_os = "macos")]
#[repr(C)]
struct MachTimebaseInfo {
    numer: u32,
    denom: u32,
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn mach_absolute_time() -> u64;
    fn mach_timebase_info(info: *mut MachTimebaseInfo) -> libc::c_int;
}

#[cfg(target_os = "macos")]
fn nanos_now() -> u64 {
    // mach_absolute_time() ticks are NOT nanoseconds on native Apple Silicon:
    // the timebase is 125/3 (one tick = 41.67ns; 1:1 only holds on Intel Macs
    // and under Rosetta). Convert via mach_timebase_info, queried once.
    // Verified empirically 2026-07-20: a 100ms sleep measured 2.47M raw ticks.
    static TIMEBASE: std::sync::OnceLock<(u32, u32)> = std::sync::OnceLock::new();
    let (numer, denom) = *TIMEBASE.get_or_init(|| {
        let mut info = MachTimebaseInfo { numer: 0, denom: 0 };
        let rc = unsafe { mach_timebase_info(&mut info) };
        assert_eq!(rc, 0, "mach_timebase_info failed with {}", rc);
        assert_ne!(info.denom, 0, "mach_timebase_info returned denom=0");
        (info.numer, info.denom)
    });
    let ticks = unsafe { mach_absolute_time() };
    // u128 intermediate: ticks * numer overflows u64 after ~50 days of uptime
    // at timebase 125/3.
    ((ticks as u128 * numer as u128) / denom as u128) as u64
}

#[cfg(target_os = "linux")]
fn nanos_now() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC_RAW, &mut ts);
    }
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn nanos_now() -> u64 {
    use std::time::Instant;
    static EPOCH: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let epoch = EPOCH.get_or_init(Instant::now);
    epoch.elapsed().as_nanos() as u64
}

// QoS pinning (audit H4). libc 0.2.183 does not re-export the pthread/qos.h
// bindings (they live in its private `new` module), so declare the one symbol
// we need directly; the ABI is a u32 enum + c_int priority offset.

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn pthread_set_qos_class_self_np(class: libc::c_uint, priority: libc::c_int) -> libc::c_int;
}

#[cfg(target_os = "macos")]
const QOS_CLASS_USER_INTERACTIVE: libc::c_uint = 0x21;

/// Pin the calling thread to the highest QoS class so macOS schedules it on a
/// P-core and resists mid-run E-core migration (audit H4). Panics on failure:
/// an unpinned label-minting run is a harness bug, not a degraded mode.
fn pin_qos() {
    #[cfg(target_os = "macos")]
    {
        let rc = unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) };
        assert_eq!(
            rc, 0,
            "pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE) failed with rc={} — \
             refusing to mint labels without P-core QoS pinning (audit H4)",
            rc
        );
    }
    #[cfg(not(target_os = "macos"))]
    eprintln!(
        "jit_bench: QoS pinning is macOS-only; running at default scheduling class \
         (labels are exposed to core-migration drift — rely on the sentinel)"
    );
}

/// Validate a raw median timing, rejecting garbage values.
///
/// Rejects: NaN, negative (impossible for u64→f64 but defensive),
/// and absurdly large values (>1s, indicating u64 underflow from clock
/// non-monotonicity or OS scheduling artifacts). Zero is valid for
/// constant-folded expressions.
fn validate_median(median: f64) -> Result<f64, BenchError> {
    if median.is_nan() || median.is_infinite() {
        return Err(BenchError::InvalidMeasurement(median));
    }
    if median < 0.0 {
        return Err(BenchError::InvalidMeasurement(median));
    }
    if median > MAX_PLAUSIBLE_NS {
        return Err(BenchError::InvalidMeasurement(median));
    }
    Ok(median)
}

/// Interquartile range of a sample set (audit M5). Sorts a copy; does not
/// require pre-sorted input.
fn iqr(samples: &[f64]) -> f64 {
    assert!(
        samples.len() >= 4,
        "iqr needs at least 4 samples, got {}",
        samples.len()
    );
    let mut sorted = samples.to_vec();
    sorted.sort_unstable_by(|a, b| {
        a.partial_cmp(b)
            .unwrap_or_else(|| panic!("iqr: non-comparable samples {a} vs {b}"))
    });
    sorted[(sorted.len() * 3) / 4] - sorted[sorted.len() / 4]
}

/// Whether a sentinel measurement has drifted beyond the tolerated fraction
/// of the session's opening calibration (audit H4). At the session's
/// [`SENTINEL_REGIME_CHANGE_FRAC`] this is the E-core-placement detector;
/// sub-threshold drift is recorded and post-hoc corrected, never fatal (see
/// the constant's doc for the three-tier policy). Pure so the drift policy is
/// testable without timing.
fn sentinel_drift_exceeded(calibration_ns: f64, measured_ns: f64, max_drift_frac: f64) -> bool {
    assert!(
        calibration_ns > 0.0,
        "sentinel calibration must be positive, got {calibration_ns}ns"
    );
    ((measured_ns - calibration_ns) / calibration_ns).abs() > max_drift_frac
}

/// Conservative lower bound on plausible per-eval cost for an expression with
/// `op_count` compute ops (audit M4).
///
/// [`MIN_NS_PER_OP`] prices ops the machine actually *issues*, but `op_count`
/// counts nodes in the arena the caller hands in — and those two stopped being
/// the same number when the backend learned to fuse (#1076, MulAdd shapes). A
/// `Mul` feeding an `Add` is two arena nodes and one `FMLA`, so the source
/// count is an UPPER bound on issued ops where the floor needs a LOWER one.
/// Using the upper bound as if it were the lower bound is unsound in the
/// false-positive direction: it rejects correct measurements of fusable
/// kernels. `measure_latency_prior`'s 47-node throughput probe is one — it
/// computes the right answer (checked against `eval_scalar`) and measures
/// 1.82ns against a 2.35ns floor.
///
/// Fusion retires at most one node per surviving node, so half the source
/// count is the sound lower bound on issued ops. Halving costs the guard
/// nothing it was built for: an early `ret`, a mis-scaled timebase, or an
/// unroll miscount lands at or near zero, orders of magnitude below either
/// floor.
fn plausibility_floor_ns(op_count: usize) -> f64 {
    op_count.div_ceil(2) as f64 * MIN_NS_PER_OP
}

/// Panic if a RAW per-eval measurement is below the dependency-chain floor
/// (audit M4). Sub-floor raw numbers are harness bugs — an early `ret`, a
/// mis-scaled timebase, an unroll miscount — never data.
///
/// The floor bounds the raw measurement, never the overhead-adjusted one:
/// 0.05ns/op is a valid lower bound on what executing the ops can take (below
/// the ~0.078ns/op 4-SIMD-ops/cycle machine limit), but the *marginal* cost
/// over the identity kernel is legitimately ~0 for small kernels in
/// throughput mode — their FP work hides in the ports left idle by
/// ILP-overlapped call/ret overhead — so `adjusted_ns <= 0` is a recorded
/// measurement condition (both `ns` and `adjusted_ns` are persisted on every
/// [`BenchResult`]), not a failure.
///
/// `op_count == 0` (a bare Var/Const leaf) has a 0ns floor, so ~0 raw is
/// legitimate there by construction.
fn assert_plausible(raw_ns: f64, op_count: usize) {
    let floor = plausibility_floor_ns(op_count);
    if raw_ns < floor {
        panic!(
            "jit_bench plausibility failure (harness bug, not data): raw {raw_ns:.4}ns for a \
             {op_count}-op expression is below the {floor:.4}ns floor ({MIN_NS_PER_OP}ns/op)"
        );
    }
}

/// Count compute ops reachable from `root` (unique DAG nodes whose kind is
/// not a Var/Const/Buffer leaf). This is the op count *after* whatever
/// folding produced the arena the caller hands us, which is what identifies
/// legitimately-instant constant-folded expressions for the plausibility
/// floor (audit M4).
fn op_count(arena: &ExprArena, root: ExprId) -> usize {
    let mut visited = vec![false; arena.len()];
    let mut stack = vec![root];
    let mut count = 0usize;
    while let Some(id) = stack.pop() {
        let idx = id.0 as usize;
        if visited[idx] {
            continue;
        }
        visited[idx] = true;
        match arena.kind(id) {
            OpKind::Var | OpKind::Const | OpKind::Buffer => {}
            _ => count += 1,
        }
        stack.extend(arena.children(id));
    }
    count
}

/// Whether any `Var` node is reachable from `root` — i.e. whether the kernel
/// has a data path from its inputs at all.
///
/// Not used to weaken the plausibility floor, and deliberately so: a var-free
/// arena is only *foldable in principle*. Neither emitter folds
/// (`compile` lowers Dwrt/Reduce/Gather/transcendentals and
/// then schedules every remaining node on both aarch64 and x86-64 — there is
/// no constant-propagation pass), so a var-free expression still executes its
/// ops and the floor still describes real work. Exempting it would silently
/// switch off the audit-M4 harness-bug detector for exactly the shapes where
/// an early `ret` is easiest to emit. See
/// `var_free_arenas_still_execute_their_ops`, which fails loudly if a folding
/// pass ever lands — at which point the fix is to derive the floor from the
/// LOWERED schedule the emitter actually built, not to blanket-exempt a class
/// of expressions from checking.
#[cfg(test)]
fn has_reachable_var(arena: &ExprArena, root: ExprId) -> bool {
    let mut visited = vec![false; arena.len()];
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        let idx = id.0 as usize;
        if visited[idx] {
            continue;
        }
        visited[idx] = true;
        if arena.kind(id) == OpKind::Var {
            return true;
        }
        stack.extend(arena.children(id));
    }
    false
}

/// The fixed input-coordinate buffer (audit H3/H1): 64 deterministic tuples
/// spanning signs and magnitudes (1e-4..1e4, ±0.0), cycled across evals so
/// data-dependent cost (denormals, select paths) is visible and the recorded
/// outputs cover 64 points instead of one.
///
/// Tuple 0 is the legacy single test point `(0.5, 0.7, 1.3, -0.2)`, so
/// [`BenchResult::output`] keeps its historical meaning.
fn input_tuples() -> &'static [[f32; 4]; INPUT_TUPLES] {
    static TUPLES: std::sync::OnceLock<[[f32; 4]; INPUT_TUPLES]> = std::sync::OnceLock::new();
    TUPLES.get_or_init(|| {
        let mut t = [[0.0f32; 4]; INPUT_TUPLES];
        t[0] = [0.5, 0.7, 1.3, -0.2];
        t[1] = [0.0, -0.0, 1.0, -1.0];
        t[2] = [1e-4, -1e-4, 1e4, -1e4];
        t[3] = [1.0, 2.0, 0.5, 0.25];
        // Remaining tuples: xorshift64, fixed seed — sign × mantissa [1,2) ×
        // 10^{-4..=4}. Deterministic so every measurement (and every
        // equivalence comparison) sees the identical buffer.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for tuple in t.iter_mut().skip(4) {
            for lane in tuple.iter_mut() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let sign = if state & 1 == 0 { 1.0f32 } else { -1.0f32 };
                let mantissa = 1.0 + ((state >> 8) & 0xFF) as f32 / 256.0;
                let exponent = ((state >> 16) % 9) as i32 - 4;
                *lane = sign * mantissa * 10.0f32.powi(exponent);
            }
        }
        t
    })
}

/// Which dependency structure the timed loop imposes (audit H3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchMode {
    /// Independent back-to-back calls: the CPU overlaps them, measuring
    /// throughput under perfect ILP. This is the historical behavior and the
    /// default for un-migrated callers.
    Throughput,
    /// Each eval's output is fed back into EVERY input lane of the next eval
    /// through `black_box`, serializing the dependency chain through
    /// whichever variables the expression actually reads: measures
    /// single-evaluation latency. Feeding all four lanes rather than only x
    /// keeps the chain intact for expressions that never read Var(0) — with
    /// a single fed lane their chain silently breaks and the "latency" label
    /// is an ILP-overlapped throughput number (the exact audit-H3
    /// under-billing this mode exists to close). The varied-input buffer
    /// drives only [`BenchMode::Throughput`] and the recorded outputs; in
    /// this mode the operand values are the orbit of the kernel's own
    /// iteration. An expression that reads NO variable has no data path from
    /// inputs to output, so no chaining scheme can serialize it: its chained
    /// latency legitimately equals its throughput.
    Latency,
}

/// Result of benchmarking: timing, dispersion, and outputs for correctness checks.
pub struct BenchResult {
    /// Raw median ns per evaluation (call overhead included).
    pub ns: f64,
    /// Median ns per evaluation minus the session's identity-kernel call
    /// overhead for this mode (audit M1). Equal to `ns` for sessionless
    /// wrappers, which measure no overhead. Can legitimately be <= 0 for
    /// small kernels in [`BenchMode::Throughput`]: their op work hides in the
    /// execution ports left idle by ILP-overlapped call/ret overhead, so the
    /// marginal cost over the identity kernel is ~0 ± noise. The plausibility
    /// floor (audit M4) is asserted on the raw `ns`, which the machine limit
    /// genuinely bounds.
    pub adjusted_ns: f64,
    /// Per-eval call overhead that was subtracted (0.0 for sessionless wrappers).
    pub call_overhead_ns: f64,
    /// Interquartile range of the per-eval ns across the timed samples
    /// (audit M5) — persist it so training can weight or reject jittery labels.
    pub iqr_ns: f64,
    /// Dependency structure the timed loop imposed.
    pub mode: BenchMode,
    /// Batches per timed sample after tick-floor autoscaling (audit H5); each
    /// batch is [`INPUT_TUPLES`] evals.
    pub repeat_batches: usize,
    /// All 4 SIMD lanes at the legacy test point (0.5, 0.7, 1.3, -0.2)
    /// (input tuple 0; lanes are identical because inputs are splatted).
    pub output: [f32; 4],
    /// First 4 SIMD lanes of the kernel output at every input tuple, in
    /// buffer order (audit H1): [`INPUT_TUPLES`] entries, so downstream
    /// correctness checks can cover the full input buffer instead of one
    /// point. Always measured with independent evals regardless of `mode`.
    pub outputs: Vec<[f32; 4]>,
    /// The sentinel context this label was minted under (audit H4): the most
    /// recent sentinel measurement plus the run's opening calibration, so
    /// drift is observable per label and post-hoc correctable via
    /// [`SentinelContext::normalization`]. `None` only for the sessionless
    /// wrappers ([`benchmark_jit_arena`]), which run no sentinel at all —
    /// label-minting consumers should treat `None` as "unprotected
    /// measurement", not as "no drift".
    pub sentinel: Option<SentinelContext>,
}

// =============================================================================
// Clock newtypes (docs/plans/2026-08-17-cost-model-domain.md, J4)
// =============================================================================
//
// A nanosecond reading is meaningless without saying WHICH clock it is
// denominated in: the local clock running when a particular sample was
// taken (drifts over a session — thermal, contention, scheduler placement),
// or the session's OPENING clock, the one canonical unit every label and
// every call-overhead figure is compared in. Before these types, both were
// the same `f64`, and the only thing stopping "subtract overhead, then
// normalize" (WRONG — leaves a drift-dependent residue, see
// `training::mint::normalized_label_ns`) from replacing "normalize, then
// subtract overhead" (RIGHT) was a doc comment. Review caught the swap
// twice. `LocalNs` has no `Sub` impl at all, so the raw reading cannot have
// overhead taken off it before [`LocalNs::normalize`] — the sole conversion
// to [`SessionNs`] — runs. The wrong order is now a compile error, not a
// convention.

/// A duration on the LOCAL clock: the clock that was actually running when
/// this particular sample was taken. Not comparable to another `LocalNs`
/// from a different point in the same session without normalizing — that is
/// the whole reason [`SentinelContext`] exists.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct LocalNs(f64);

impl LocalNs {
    /// # Panics
    /// Panics on a non-finite reading — a raw timer sample is never NaN or
    /// infinite short of a harness bug, and propagating one silently would
    /// poison every downstream label.
    #[must_use]
    pub fn new(ns: f64) -> Self {
        assert!(ns.is_finite(), "LocalNs::new: non-finite reading {ns}");
        Self(ns)
    }

    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }

    /// The ONLY way to leave the local clock: re-express this reading in the
    /// session's opening clock via the drift factor the sentinel measured.
    /// Nothing else produces a [`SessionNs`] from a raw reading, so "apply
    /// drift" cannot be skipped or reordered by a future caller.
    #[must_use]
    pub fn normalize(self, drift: DriftFactor) -> SessionNs {
        SessionNs(self.0 * drift.0)
    }
}

/// A duration in the SESSION's opening clock — the canonical, comparable
/// unit. Every [`CostLabel::value`] and every call-overhead figure lives
/// here. Produced either by [`LocalNs::normalize`] (a drift-corrected
/// reading) or [`SessionNs::new`] (a value already known to be on this
/// clock — e.g. the identity-kernel call overhead, itself measured once at
/// session open where the local and opening clocks coincide by
/// construction).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct SessionNs(f64);

impl SessionNs {
    /// # Panics
    /// Panics on a non-finite value, for the same reason as [`LocalNs::new`].
    #[must_use]
    pub fn new(ns: f64) -> Self {
        assert!(ns.is_finite(), "SessionNs::new: non-finite value {ns}");
        Self(ns)
    }

    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}

impl std::ops::Sub for SessionNs {
    type Output = SessionNs;
    /// Call overhead is a `SessionNs`, so it can only be subtracted from a
    /// `SessionNs` — `LocalNs` has no `Sub` impl at all. That asymmetry is
    /// what makes "normalize, then subtract overhead" the only expression
    /// that type-checks (J4).
    fn sub(self, overhead: SessionNs) -> SessionNs {
        SessionNs(self.0 - overhead.0)
    }
}

/// The clock-speed correction factor `calibration_ns / local_sentinel_ns`
/// (see [`SentinelContext::normalization`]) — the only producer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DriftFactor(f64);

impl DriftFactor {
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}

/// How many expressions a run had already produced labels for when this one
/// was minted — collection order, the axis timing drift correlates with
/// (generation emits size/family bands sequentially, so an uncorrected label
/// set turns residual drift into learnable spurious signal; see
/// `training::mint`). Persisted alongside [`CostLabel`] so a downstream
/// consumer can verify residual drift isn't confounded with structure,
/// rather than assuming the benchmark-order shuffle removed the
/// correlation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct BenchPosition(pub usize);

/// The sentinel state a [`BenchResult`] was measured under: the most recent
/// sentinel reading (`local_sentinel_ns`) and the session's opening
/// calibration (`calibration_ns`).
///
/// Persist both next to every minted label. Drift between them is *data*,
/// not an error: contention noise averages out over corpus volume and
/// randomized benchmark order, and slow one-signed drift (thermal, background
/// load) becomes a measured correction through [`Self::normalization`]. Only
/// a step-function regime change aborts the session — see
/// [`SENTINEL_REGIME_CHANGE_FRAC`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SentinelContext {
    /// Median throughput ns/eval of the most recent sentinel measurement
    /// taken at or before this label (the opening calibration until the
    /// first [`SENTINEL_INTERVAL`] boundary).
    pub local_sentinel_ns: f64,
    /// The session's opening sentinel calibration (ns/eval).
    pub calibration_ns: f64,
}

impl SentinelContext {
    /// The running clock-speed correction `calibration_ns /
    /// local_sentinel_ns` — the intended use of the sentinel record.
    ///
    /// The sentinel is a fixed kernel, so if it now takes 1.25× its opening
    /// time, the machine is running ~1.25× slower and every label minted
    /// nearby is inflated by the same factor. Multiplying a label's ns by
    /// this factor re-expresses it in the run's opening clock — slow drift
    /// becomes a correction instead of a discarded run. The factor is `< 1`
    /// when the machine slowed down after calibration (inflated labels scale
    /// back down), `> 1` when it sped up, and exactly `1.0` for no drift.
    ///
    /// # Panics
    ///
    /// Panics if either reading is non-positive — a sentinel that measured
    /// nothing cannot correct anything, and silently returning `inf`/`NaN`
    /// would poison every corrected label downstream.
    #[must_use]
    pub fn normalization(&self) -> DriftFactor {
        assert!(
            self.calibration_ns > 0.0 && self.local_sentinel_ns > 0.0,
            "SentinelContext::normalization: non-positive sentinel reading \
             (calibration {}ns, local {}ns)",
            self.calibration_ns,
            self.local_sentinel_ns
        );
        DriftFactor(self.calibration_ns / self.local_sentinel_ns)
    }
}

impl BenchResult {
    /// Check that two results compute the same function at the legacy test
    /// point (all lanes within epsilon). Returns Err with the max divergence
    /// if any lane differs.
    ///
    /// Non-finite lanes follow `Tolerance::Numeric`'s rule: identical bit
    /// patterns and NaN-vs-NaN agree; any other non-finite pairing is
    /// divergence (`Err(f32::INFINITY)`), never a pass. The old
    /// `diff > max_diff` comparison was NaN-blind — `|finite − NaN|` is NaN,
    /// `NaN > 0.0` is false, so a NaN-producing rewrite sailed through as
    /// max_diff 0.0 and minted a fake label (the audit-M2 silent-censoring
    /// shape, inverted).
    ///
    /// This is deliberately still the historical one-point check; widening it
    /// to the full [`BenchResult::outputs`] buffer (audit H1) belongs to the
    /// interpreter-reference work, not this harness layer.
    pub fn check_equivalence(&self, other: &BenchResult, epsilon: f32) -> Result<(), f32> {
        let mut max_diff: f32 = 0.0;
        for i in 0..4 {
            let (a, b) = (self.output[i], other.output[i]);
            if a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) {
                continue;
            }
            if !a.is_finite() || !b.is_finite() {
                return Err(f32::INFINITY);
            }
            let diff = (a - b).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        if max_diff > epsilon {
            Err(max_diff)
        } else {
            Ok(())
        }
    }

    /// The measurement's value in the session's opening clock: the sentinel's
    /// drift applied to the raw reading, THEN call overhead subtracted — the
    /// only order [`LocalNs`]/[`SessionNs`] type-check (J4). Also returns the
    /// drift factor that was applied, for callers that persist it.
    ///
    /// This is the one place that arithmetic happens; [`CostLabel::mint`] and
    /// `training::mint::normalized_label_ns`/`label_normalization` both call
    /// it rather than restating the formula.
    ///
    /// # Panics
    ///
    /// Panics if `self.sentinel` is `None` — a sessionless measurement (no
    /// `BenchSession`) carries no drift information, so it cannot be
    /// corrected into a value comparable with session-minted ones. `context`
    /// names the call site in the panic message.
    pub fn corrected_session_ns(&self, context: &str) -> (SessionNs, DriftFactor) {
        let calibration = self.sentinel.unwrap_or_else(|| {
            panic!(
                "{context}: label minted from a BenchResult with no sentinel context — the \
                 measurement came from a sessionless wrapper (no QoS pin, no drift sentinel, no \
                 overhead subtraction), so its drift is neither measured nor correctable. Mint \
                 labels through a BenchSession"
            )
        });
        let drift = calibration.normalization();
        let overhead = SessionNs::new(self.call_overhead_ns);
        let value = LocalNs::new(self.ns).normalize(drift) - overhead;
        (value, drift)
    }
}

/// A measurement that has left the clock it was taken on
/// (docs/plans/2026-08-17-cost-model-domain.md, J3): a value in
/// [`SessionNs`], plus the context that makes the value meaningful — which
/// [`BenchMode`] it was measured under, how much drift correction it took,
/// where in the run it was minted, and the sentinel calibration it was
/// corrected against.
///
/// These five pieces used to travel separately — `ns`/`adjusted_ns`/
/// `call_overhead_ns` as bare fields on [`BenchResult`], `mode` beside them,
/// [`SentinelContext`] behind an `Option` — and nothing stopped a raw,
/// unnormalized `f64` from reaching a training target. [`CostLabel::mint`] is
/// now the only way to produce one, and [`CostLabel::target_log_ns`] is the
/// only bridge from a label to the `f32` a model is trained on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostLabel {
    value: SessionNs,
    mode: BenchMode,
    drift: DriftFactor,
    position: BenchPosition,
    calibration: SentinelContext,
}

impl CostLabel {
    /// Mint a label from a session-produced [`BenchResult`] at collection
    /// `position`. `context` names the call site for the panic message.
    ///
    /// # Panics
    ///
    /// See [`BenchResult::corrected_session_ns`].
    #[must_use]
    pub fn mint(bench: &BenchResult, position: BenchPosition, context: &str) -> Self {
        let (value, drift) = bench.corrected_session_ns(context);
        let calibration = bench
            .sentinel
            .expect("corrected_session_ns already panicked above on None");
        Self {
            value,
            mode: bench.mode,
            drift,
            position,
            calibration,
        }
    }

    #[must_use]
    pub fn value(&self) -> SessionNs {
        self.value
    }

    #[must_use]
    pub fn mode(&self) -> BenchMode {
        self.mode
    }

    #[must_use]
    pub fn drift(&self) -> DriftFactor {
        self.drift
    }

    #[must_use]
    pub fn position(&self) -> BenchPosition {
        self.position
    }

    #[must_use]
    pub fn calibration(&self) -> SentinelContext {
        self.calibration
    }

    /// The training target: `log_ns` of the normalized value. The only way
    /// to turn a `CostLabel` into the `f32` a model is trained on.
    #[must_use]
    pub fn target_log_ns(&self) -> f32 {
        log_ns(self.value)
    }
}

/// Raw timing before overhead adjustment and plausibility checks.
struct RawMeasurement {
    ns: f64,
    iqr_ns: f64,
    repeat_batches: usize,
    mode: BenchMode,
    output: [f32; 4],
    outputs: Vec<[f32; 4]>,
}

const LANES: usize = pixelflow_codegen::JIT_VECTOR_BYTES / 4;

/// Core timed loop: warmup, output capture over the full input buffer,
/// tick-floor autoscaling of `repeat_batches` (audit H5), and TIMED_RUNS
/// samples reduced to median + IQR (audit M5).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn measure_exec_code(
    exec_code: &ExecutableCode,
    start_batches: usize,
    mode: BenchMode,
) -> Result<RawMeasurement, BenchError> {
    use pixelflow_codegen::emit::executable::{Point4, TileSlice};

    const INPUT_VECTORS: usize = INPUT_TUPLES / LANES;
    let mut input_points = [Point4::new(
        [0.0f32; LANES],
        [0.0f32; LANES],
        [0.0f32; LANES],
        [0.0f32; LANES],
    ); INPUT_VECTORS];
    let tuples = input_tuples();
    for (v_idx, pt) in input_points.iter_mut().enumerate() {
        let mut xs = [0.0f32; LANES];
        let mut ys = [0.0f32; LANES];
        let mut zs = [0.0f32; LANES];
        let mut ws = [0.0f32; LANES];
        for lane in 0..LANES {
            let t = &tuples[v_idx * LANES + lane];
            xs[lane] = t[0];
            ys[lane] = t[1];
            zs[lane] = t[2];
            ws[lane] = t[3];
        }
        *pt = Point4::new(xs, ys, zs, ws);
    }

    let mut out_buf = [0.0f32; LANES];
    for _ in 0..WARMUP_PASSES {
        for &p in &input_points {
            unsafe {
                exec_code.call_collapse(
                    core::ptr::null(),
                    TileSlice::single(out_buf.as_mut_ptr()),
                    p,
                );
            }
        }
    }

    // Independent (non-chained) evals over the whole buffer, so the
    // recorded outputs are mode-independent pure function values.
    let mut outputs = Vec::with_capacity(INPUT_TUPLES);
    for &p in &input_points {
        let mut out = [0.0f32; LANES];
        unsafe {
            exec_code.call_collapse(core::ptr::null(), TileSlice::single(out.as_mut_ptr()), p);
        }
        for &val in &out {
            outputs.push([val; 4]);
        }
    }
    let output = outputs[0];

    let mut repeat_batches = start_batches.max(1);
    loop {
        let mut totals = [0u64; TIMED_RUNS];
        for t in &mut totals {
            match mode {
                BenchMode::Throughput => {
                    let start = nanos_now();
                    for _ in 0..repeat_batches {
                        for &p in &input_points {
                            unsafe {
                                exec_code.call_collapse(
                                    core::ptr::null(),
                                    TileSlice::single(out_buf.as_mut_ptr()),
                                    p,
                                );
                            }
                        }
                    }
                    *t = nanos_now() - start;
                }
                BenchMode::Latency => {
                    let mut prev = [0.0f32; LANES];
                    unsafe {
                        exec_code.call_collapse(
                            core::ptr::null(),
                            TileSlice::single(prev.as_mut_ptr()),
                            input_points[0],
                        );
                    }
                    let start = nanos_now();
                    for _ in 0..repeat_batches {
                        for _ in 0..INPUT_VECTORS {
                            let p = Point4::new(prev, prev, prev, prev);
                            std::hint::black_box(&p);
                            unsafe {
                                exec_code.call_collapse(
                                    core::ptr::null(),
                                    TileSlice::single(prev.as_mut_ptr()),
                                    p,
                                );
                            }
                        }
                    }
                    std::hint::black_box(prev);
                    *t = nanos_now() - start;
                }
            }
        }
        totals.sort_unstable();
        let median_total = totals[TIMED_RUNS / 2];

        if median_total < MIN_TIMED_SAMPLE_NS {
            assert!(
                repeat_batches < MAX_REPEAT_BATCHES,
                "jit_bench harness bug: median timed sample {}ns is below the {}ns tick \
                     floor even at repeat_batches={} ({} evals/sample) — the timer or the \
                     emitted code is broken (audit H5)",
                median_total,
                MIN_TIMED_SAMPLE_NS,
                repeat_batches,
                INPUT_TUPLES * repeat_batches,
            );
            repeat_batches = (repeat_batches * 2).min(MAX_REPEAT_BATCHES);
            continue;
        }

        let evals = (INPUT_TUPLES * repeat_batches) as f64;
        // totals is sorted and the per-eval transform is monotone, so the
        // median index is unchanged (audit M5: keep the median selection).
        let per_eval: Vec<f64> = totals.iter().map(|&t| t as f64 / evals).collect();
        let median = per_eval[TIMED_RUNS / 2];
        let dispersion = iqr(&per_eval);
        return validate_median(median).map(|ns| RawMeasurement {
            ns,
            iqr_ns: dispersion,
            repeat_batches,
            mode,
            output,
            outputs,
        });
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn measure_exec_code(
    exec_code: &ExecutableCode,
    start_batches: usize,
    mode: BenchMode,
) -> Result<RawMeasurement, BenchError> {
    let _ = (exec_code, start_batches, mode);
    Err(BenchError::UnsupportedArch)
}

/// Assemble a [`BenchResult`] from a raw measurement, applying the
/// identity-kernel overhead adjustment (audit M1) and the plausibility floor
/// on the raw measurement (audit M4 — panics on sub-floor raw values, never
/// clamps; a non-positive adjusted value is recorded, not rejected).
/// `sentinel` is the session's current drift context, `None` on the
/// sessionless paths that run no sentinel.
fn finalize(
    raw: RawMeasurement,
    ops: usize,
    call_overhead_ns: f64,
    sentinel: Option<SentinelContext>,
) -> BenchResult {
    assert_plausible(raw.ns, ops);
    let adjusted_ns = raw.ns - call_overhead_ns;
    BenchResult {
        ns: raw.ns,
        adjusted_ns,
        call_overhead_ns,
        iqr_ns: raw.iqr_ns,
        mode: raw.mode,
        repeat_batches: raw.repeat_batches,
        output: raw.output,
        outputs: raw.outputs,
        sentinel,
    }
}

/// JIT-compile and benchmark an arena expression. No `Expr` conversion.
///
/// Sessionless compatibility wrapper: [`BenchMode::Throughput`] (the
/// historical dependency structure), no QoS pinning, no sentinel, and no
/// overhead subtraction (`adjusted_ns == ns`). Label-minting callers should
/// migrate to [`BenchSession`].
pub fn benchmark_jit_arena(arena: &ExprArena, root: ExprId) -> Result<BenchResult, BenchError> {
    benchmark_jit_arena_repeated(arena, root, 1)
}

/// Sessionless benchmark with an explicit starting `repeat_batches`.
///
/// `repeat_batches` is now only the autoscale *starting point*: the core loop
/// raises it until the median timed sample clears the timer-tick floor
/// (audit H5), so passing 1 is always safe.
pub fn benchmark_jit_arena_repeated(
    arena: &ExprArena,
    root: ExprId,
    repeat_batches: usize,
) -> Result<BenchResult, BenchError> {
    let result = compile(arena, root).map_err(BenchError::CompileFailed)?;
    let raw = measure_exec_code(&result.code, repeat_batches, BenchMode::Throughput)?;
    Ok(finalize(raw, op_count(arena, root), 0.0, None))
}

// =============================================================================
// BenchSession — measurement integrity for label-minting runs
// =============================================================================

/// One sentinel measurement, recorded so callers can persist the session's
/// drift history alongside minted labels (audit H4).
#[derive(Clone, Copy, Debug)]
pub struct SentinelSample {
    /// How many expressions the session had benchmarked when this sample was
    /// taken (0 for the opening calibration).
    pub expr_index: usize,
    /// Median throughput ns/eval of the sentinel kernel.
    pub ns: f64,
}

/// The fixed moderate-size reference kernel: ~40 compute ops of mixed
/// mul/add/sqrt over all four coordinates. Big enough that its cost tracks
/// real kernel throughput, small enough to re-measure cheaply every
/// [`SENTINEL_INTERVAL`] expressions.
fn sentinel_arena() -> (ExprArena, ExprId) {
    let mut arena = ExprArena::new();
    let x = arena.push_var(0);
    let y = arena.push_var(1);
    let z = arena.push_var(2);
    let w = arena.push_var(3);
    let vars = [x, y, z, w];
    let mut acc = x;
    // 8 rounds × 5 ops = 40 ops (+ 8 consts + 4 vars = 52 nodes).
    // sqrt argument is acc² + positive constant, so it never goes negative.
    let round_consts = [1.25f32, 0.75, 2.5, 0.5, 3.0, 1.5, 0.25, 2.0];
    for (i, &c) in round_consts.iter().enumerate() {
        let k = arena.push_const(c);
        let v = vars[i % 4];
        let scaled = arena.push_binary(OpKind::Mul, acc, v);
        let squared = arena.push_binary(OpKind::Mul, acc, acc);
        let shifted = arena.push_binary(OpKind::Add, squared, k);
        let rooted = arena.push_unary(OpKind::Sqrt, shifted);
        acc = arena.push_binary(OpKind::Add, scaled, rooted);
    }
    (arena, acc)
}

/// The identity kernel (`|x, _, _, _| x`): its measured per-eval time is pure
/// call overhead — fn-pointer call/ret plus register setup (audit M1). Under
/// [`BenchMode::Latency`] its chained measurement serializes the same
/// all-lanes-fed call/ret + register-move path every candidate's chained
/// measurement pays, so the per-mode subtraction stays coherent.
fn identity_arena() -> (ExprArena, ExprId) {
    let mut arena = ExprArena::new();
    let root = arena.push_var(0);
    (arena, root)
}

/// Owns measurement integrity for a benchmarking run (audit H4/M1/L5):
/// QoS-pins the thread at creation, burns in DVFS, measures identity-kernel
/// call overhead per mode, calibrates a fixed sentinel kernel, and re-measures
/// the sentinel every [`SENTINEL_INTERVAL`] benchmarked expressions.
///
/// The sentinel is a **calibration signal, not a tripwire**: each reading is
/// recorded ([`Self::sentinel_samples`]) and stamped onto every
/// [`BenchResult`] as a [`SentinelContext`], so drift is observable per label
/// and post-hoc correctable via [`SentinelContext::normalization`]. The
/// session aborts only on a ≥[`SENTINEL_REGIME_CHANGE_FRAC`] step — the
/// E-core-placement class of shift that no scalar correction can repair; see
/// that constant's doc for the full three-tier drift policy.
pub struct BenchSession {
    sentinel_code: ExecutableCode,
    calibration_ns: f64,
    sentinel_samples: Vec<SentinelSample>,
    overhead_throughput_ns: f64,
    overhead_latency_ns: f64,
    exprs_benchmarked: usize,
    exprs_at_last_sentinel: usize,
}

impl BenchSession {
    /// Create a session: pin QoS, burn in, measure call overhead, calibrate
    /// the sentinel.
    ///
    /// # Panics
    ///
    /// Panics if QoS pinning fails (macOS), or if the sentinel/identity
    /// kernels fail to compile or measure — a session that cannot establish
    /// its integrity machinery must not mint labels.
    #[must_use]
    pub fn new() -> Self {
        pin_qos();

        let (sentinel_arena, sentinel_root) = sentinel_arena();
        let sentinel_code = compile(&sentinel_arena, sentinel_root)
            .unwrap_or_else(|e| panic!("BenchSession: sentinel kernel failed to compile: {e}"))
            .code;

        // Burn-in (audit L5): run real work for BURN_IN_NS so DVFS reaches
        // steady state before anything is calibrated.
        let burn_start = nanos_now();
        while nanos_now().wrapping_sub(burn_start) < BURN_IN_NS {
            if let Err(e) = measure_exec_code(&sentinel_code, 1, BenchMode::Throughput) {
                panic!("BenchSession: burn-in measurement failed: {e}");
            }
        }

        // Identity-kernel call overhead per mode (audit M1). Throughput and
        // latency overheads differ (overlapped vs serialized call/ret), so
        // each mode subtracts its own.
        let (identity_arena, identity_root) = identity_arena();
        let identity_code = compile(&identity_arena, identity_root)
            .unwrap_or_else(|e| panic!("BenchSession: identity kernel failed to compile: {e}"))
            .code;
        let overhead_throughput_ns = measure_exec_code(&identity_code, 1, BenchMode::Throughput)
            .unwrap_or_else(|e| {
                panic!("BenchSession: identity overhead (throughput) measurement failed: {e}")
            })
            .ns;
        let overhead_latency_ns = measure_exec_code(&identity_code, 1, BenchMode::Latency)
            .unwrap_or_else(|e| {
                panic!("BenchSession: identity overhead (latency) measurement failed: {e}")
            })
            .ns;

        // Opening sentinel calibration (audit H4).
        let calibration_ns = measure_exec_code(&sentinel_code, 1, BenchMode::Throughput)
            .unwrap_or_else(|e| panic!("BenchSession: sentinel calibration failed: {e}"))
            .ns;

        Self {
            sentinel_code,
            calibration_ns,
            sentinel_samples: vec![SentinelSample {
                expr_index: 0,
                ns: calibration_ns,
            }],
            overhead_throughput_ns,
            overhead_latency_ns,
            exprs_benchmarked: 0,
            exprs_at_last_sentinel: 0,
        }
    }

    /// Per-eval identity-kernel call overhead for `mode` (audit M1).
    #[must_use]
    pub fn call_overhead_ns(&self, mode: BenchMode) -> f64 {
        match mode {
            BenchMode::Throughput => self.overhead_throughput_ns,
            BenchMode::Latency => self.overhead_latency_ns,
        }
    }

    /// The session's opening sentinel calibration (ns/eval).
    #[must_use]
    pub fn calibration_ns(&self) -> f64 {
        self.calibration_ns
    }

    /// Every sentinel measurement taken so far (opening calibration first),
    /// for persisting alongside minted labels.
    #[must_use]
    pub fn sentinel_samples(&self) -> &[SentinelSample] {
        &self.sentinel_samples
    }

    /// The drift context the *next* label would be minted under: the most
    /// recent sentinel reading paired with the opening calibration. The same
    /// value is stamped onto every [`BenchResult`] this session produces.
    #[must_use]
    pub fn sentinel_context(&self) -> SentinelContext {
        let last = self
            .sentinel_samples
            .last()
            .expect("BenchSession always records the opening calibration");
        SentinelContext {
            local_sentinel_ns: last.ns,
            calibration_ns: self.calibration_ns,
        }
    }

    /// JIT-compile and benchmark an arena expression under `mode`, with
    /// sentinel drift checking, overhead adjustment, and the plausibility
    /// floor.
    ///
    /// # Panics
    ///
    /// Panics on a sentinel regime change — drift beyond
    /// ±[`SENTINEL_REGIME_CHANGE_FRAC`] of calibration (audit H4, the E-core
    /// detector; smaller drift is recorded on the result's
    /// [`SentinelContext`], not fatal) — and on measurements below the
    /// per-expression plausibility floor (audit M4).
    pub fn benchmark_arena(
        &mut self,
        arena: &ExprArena,
        root: ExprId,
        mode: BenchMode,
    ) -> Result<BenchResult, BenchError> {
        // Sentinel BEFORE compiling, as this entry point has always done: the
        // drift check runs its own calibration kernel, so doing it after
        // compilation would leave different cache and branch-predictor state
        // immediately ahead of the timed loop. Delegating wholesale to
        // `benchmark_compiled` would silently reorder that.
        self.check_sentinel_if_due();
        let compiled = compile(arena, root).map_err(BenchError::CompileFailed)?;
        self.measure_gated(&compiled.code, arena, root, mode)
    }

    /// Benchmark a PRE-COMPILED code object under `mode`, with the same
    /// sentinel drift checking, overhead adjustment, and plausibility floor
    /// as [`BenchSession::benchmark_arena`].
    ///
    /// This is the "time the exact artifact that passed the check" entry
    /// point: callers that compiled and correctness-gated an
    /// [`ExecutableCode`] in a preparation phase hand that same object here,
    /// so no compilation (with its allocation and icache pollution) happens
    /// inside the timed phase, and the timed code is bit-identical to the
    /// checked code. `arena`/`root` must be the expression `code` was
    /// compiled from — they parameterize the plausibility floor only.
    ///
    /// # Panics
    ///
    /// Panics on a sentinel regime change — drift beyond
    /// ±[`SENTINEL_REGIME_CHANGE_FRAC`] of calibration (audit H4, the E-core
    /// detector; smaller drift is recorded on the result's
    /// [`SentinelContext`], not fatal) — and on measurements below the
    /// per-expression plausibility floor (audit M4).
    pub fn benchmark_compiled(
        &mut self,
        code: &ExecutableCode,
        arena: &ExprArena,
        root: ExprId,
        mode: BenchMode,
    ) -> Result<BenchResult, BenchError> {
        self.check_sentinel_if_due();
        self.measure_gated(code, arena, root, mode)
    }

    /// Time `code` and finalize the result. The shared tail of
    /// [`BenchSession::benchmark_arena`] and
    /// [`BenchSession::benchmark_compiled`], which differ only in whether they
    /// compile first — and therefore in where the sentinel check falls
    /// relative to compilation, which is why this is factored out instead of
    /// one delegating to the other.
    fn measure_gated(
        &mut self,
        code: &ExecutableCode,
        arena: &ExprArena,
        root: ExprId,
        mode: BenchMode,
    ) -> Result<BenchResult, BenchError> {
        let raw = measure_exec_code(code, 1, mode)?;
        self.exprs_benchmarked += 1;
        Ok(finalize(
            raw,
            op_count(arena, root),
            self.call_overhead_ns(mode),
            Some(self.sentinel_context()),
        ))
    }

    /// Convenience: compile once, measure both modes back-to-back. Counts as
    /// one benchmarked expression for sentinel cadence.
    pub fn benchmark_arena_both(
        &mut self,
        arena: &ExprArena,
        root: ExprId,
    ) -> Result<(BenchResult, BenchResult), BenchError> {
        self.check_sentinel_if_due();
        let compiled = compile(arena, root).map_err(BenchError::CompileFailed)?;
        let ops = op_count(arena, root);
        let throughput = measure_exec_code(&compiled.code, 1, BenchMode::Throughput)?;
        let latency = measure_exec_code(&compiled.code, 1, BenchMode::Latency)?;
        self.exprs_benchmarked += 1;
        let sentinel = Some(self.sentinel_context());
        Ok((
            finalize(throughput, ops, self.overhead_throughput_ns, sentinel),
            finalize(latency, ops, self.overhead_latency_ns, sentinel),
        ))
    }

    fn check_sentinel_if_due(&mut self) {
        if self.exprs_benchmarked - self.exprs_at_last_sentinel < SENTINEL_INTERVAL {
            return;
        }
        self.exprs_at_last_sentinel = self.exprs_benchmarked;
        let measured_ns = measure_exec_code(&self.sentinel_code, 1, BenchMode::Throughput)
            .unwrap_or_else(|e| {
                panic!(
                    "BenchSession: sentinel re-measurement failed at expression index {}: {e}",
                    self.exprs_benchmarked
                )
            })
            .ns;
        // Record first, judge second: the reading is data (it rides on every
        // subsequent BenchResult as a SentinelContext for post-hoc clock
        // correction) even when it is also grounds for aborting. WHY the
        // threshold is 50% and not tighter: contention noise is absorbed by
        // corpus volume and randomized benchmark order; slow one-signed drift
        // becomes a measured correction via SentinelContext::normalization;
        // only a step-function regime change — E-core placement, a 2–3×
        // shift — poisons everything after it in a way no scalar factor
        // repairs, and that is the one class worth killing the run over.
        self.sentinel_samples.push(SentinelSample {
            expr_index: self.exprs_benchmarked,
            ns: measured_ns,
        });
        if sentinel_drift_exceeded(
            self.calibration_ns,
            measured_ns,
            SENTINEL_REGIME_CHANGE_FRAC,
        ) {
            panic!(
                "jit_bench sentinel regime change (audit H4): calibration {:.3}ns, now {:.3}ns \
                 ({:+.1}% > ±{:.0}%) at expression index {} — a shift this large is E-core \
                 placement (or an equivalent scheduling regime change), not correctable drift; \
                 labels since the last good sentinel are poisoned",
                self.calibration_ns,
                measured_ns,
                (measured_ns / self.calibration_ns - 1.0) * 100.0,
                SENTINEL_REGIME_CHANGE_FRAC * 100.0,
                self.exprs_benchmarked
            );
        }
    }
}

impl Default for BenchSession {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Compile-cost benchmarking (kernel-unification Phase 0, gate G0)
// =============================================================================

/// Timed compile samples per measurement. A single compile costs microseconds,
/// far above the ~1ns timer resolution, so one compile per sample is enough;
/// the median over many samples rejects OS scheduling jitter.
const COMPILE_TIMED_RUNS: usize = 101;

/// Warmup compiles before timing. Warms the allocator, icache, and (on the
/// fresh path) the kernel's VM-map fast paths so the timed samples measure
/// steady-state cost, not cold-start page faults.
const COMPILE_WARMUP_ITERS: usize = 16;

/// Number of structurally distinct kernels [`benchmark_compile_cached_miss`]
/// consumes: warmup plus timed samples, every one of which must be a genuine
/// cache miss.
pub const COMPILE_MISS_KERNELS: usize = COMPILE_WARMUP_ITERS + COMPILE_TIMED_RUNS;

/// Result of a compile-cost measurement.
pub struct CompileCostResult {
    /// Median ns for one complete compile.
    pub ns: f64,
    /// Emitted machine-code size in bytes.
    pub code_bytes: usize,
}

/// Median wall-clock cost of one *production* cache-miss compile through
/// [`pixelflow_codegen::jit_cache::compile`]. On a miss that entry
/// point runs canonical-key construction, the cache lock,
/// `optimize_runtime_arena` (e-graph saturation + extraction, itself keyed by
/// the same structure and therefore also a miss for a distinct kernel),
/// `compile`, and cache insertion. This is what one distinct kernel
/// actually costs at runtime; [`benchmark_compile_fresh`] times only the emit
/// step and exists for attribution.
///
/// `kernels` must contain exactly [`COMPILE_MISS_KERNELS`] canonically
/// distinct entries (e.g. each embedding a different const leaf that survives
/// optimization) — built by the caller beforehand so arena construction stays
/// outside the timed window. Distinctness is not trusted: the entry-count
/// delta of the global cache is asserted afterwards, so a stream that
/// accidentally repeats a kernel fails loudly instead of silently timing
/// cache hits.
///
/// Both global caches (the JIT cache and the optimizer's) are unbounded and
/// retain every entry compiled here. That is fine for a one-shot bench
/// process; do not call this in a long-lived process expecting the memory
/// back.
pub fn benchmark_compile_cached_miss(kernels: Vec<(ExprArena, ExprId)>) -> Result<f64, BenchError> {
    use pixelflow_codegen::jit_cache;

    assert_eq!(
        kernels.len(),
        COMPILE_MISS_KERNELS,
        "benchmark_compile_cached_miss needs exactly COMPILE_MISS_KERNELS ({}) kernels, got {}",
        COMPILE_MISS_KERNELS,
        kernels.len()
    );

    let entries_before = jit_cache::entry_count();
    let mut kernels = kernels.into_iter();

    for (arena, root) in kernels.by_ref().take(COMPILE_WARMUP_ITERS) {
        let compiled = jit_cache::compile(&arena, root, pixelflow_ir::LatticeShape::POINT)
            .map_err(BenchError::CompileFailed)?;
        std::hint::black_box(&compiled);
    }

    let mut times = [0u64; COMPILE_TIMED_RUNS];
    for t in &mut times {
        let (arena, root) = kernels.next().expect("stream length asserted above");
        let start = nanos_now();
        let compiled = jit_cache::compile(&arena, root, pixelflow_ir::LatticeShape::POINT)
            .map_err(BenchError::CompileFailed)?;
        std::hint::black_box(&compiled);
        *t = nanos_now() - start;
        // The cache retains its own Arc, so dropping ours (and the source
        // arena) here is deallocation bookkeeping outside the timed window.
    }

    let interned = jit_cache::entry_count() - entries_before;
    assert_eq!(
        interned, COMPILE_MISS_KERNELS,
        "benchmark_compile_cached_miss: only {} of {} calls missed the cache — the kernel \
         stream was not canonically distinct, so the timings measured hits and are wrong",
        interned, COMPILE_MISS_KERNELS
    );

    times.sort_unstable();
    validate_median(times[COMPILE_TIMED_RUNS / 2] as f64)
}

/// Median wall-clock cost of one `compile` call, fresh-allocation
/// path: every compile mmaps a new executable region and munmaps it on drop.
/// Both syscalls are inside the timed window.
///
/// This is the **emit step only** — no optimizer, no canonical key, no cache
/// lock or insertion. A real cache miss through `jit_cache::compile_cached`
/// pays all of those on top; measure that with
/// [`benchmark_compile_cached_miss`]. Keep this series for attributing how
/// much of the miss cost is codegen proper.
pub fn benchmark_compile_fresh(
    arena: &ExprArena,
    root: ExprId,
) -> Result<CompileCostResult, BenchError> {
    for _ in 0..COMPILE_WARMUP_ITERS {
        let result = compile(arena, root).map_err(BenchError::CompileFailed)?;
        std::hint::black_box(result.code.as_bytes().first());
    }

    let mut times = [0u64; COMPILE_TIMED_RUNS];
    let mut code_bytes = 0usize;
    for t in &mut times {
        let start = nanos_now();
        let result = compile(arena, root).map_err(BenchError::CompileFailed)?;
        std::hint::black_box(result.code.as_bytes().first());
        code_bytes = result.code.len();
        drop(result); // munmap inside the timed window
        *t = nanos_now() - start;
    }

    times.sort_unstable();
    let median = times[COMPILE_TIMED_RUNS / 2] as f64;
    validate_median(median).map(|ns| CompileCostResult { ns, code_bytes })
}

/// Convert a session-clock nanosecond value to log-nanoseconds (floored at
/// 1e-3ns, capped at 1s).
///
/// Takes [`SessionNs`], not a bare `f64` (J3/J4): a raw, un-drift-corrected
/// reading must go through [`LocalNs::normalize`] (directly, or via
/// [`CostLabel::mint`] / [`BenchResult::corrected_session_ns`]) before it can
/// become a training target — that ordering is exactly the bug class this
/// module's clock newtypes exist to make a type error. Relocated from the
/// deleted `training::gen_es` (the ES-guided corpus-growth optimizer it lived
/// in was RL-adjacent scaffolding removed per
/// docs/plans/2026-07-07-guided-saturation-redesign.md).
///
/// # Panics
///
/// Panics if `ns` is NaN.
#[must_use]
pub fn log_ns(ns: SessionNs) -> f32 {
    let raw = ns.get();
    assert!(!raw.is_nan(), "log_ns called with NaN");
    // NaN already rejected above, so clamp's total order is well-defined here.
    let clamped = raw.clamp(1e-3, 1e9);
    libm::logf(clamped as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::ExprArena;

    #[test]
    fn verify_log_ns() {
        // log(1.0) = 0.0
        let v = log_ns(SessionNs::new(1.0));
        assert!(
            libm::fabsf(v) < 0.001,
            "log_ns(1.0) should be ~0.0, got {}",
            v
        );

        // Values below 1e-3 should be floored to 1e-3.
        let v_low = log_ns(SessionNs::new(0.0001));
        let v_floor = log_ns(SessionNs::new(1e-3));
        assert!(
            libm::fabsf(v_low - v_floor) < 0.001,
            "log_ns(0.0001) should equal log_ns(1e-3), got {} vs {}",
            v_low,
            v_floor
        );

        // log(e) ≈ 1.0
        let v_e = log_ns(SessionNs::new(core::f64::consts::E));
        assert!(
            libm::fabsf(v_e - 1.0) < 0.01,
            "log_ns(e) should be ~1.0, got {}",
            v_e
        );
    }

    #[test]
    fn constant_expr_benchmarks_successfully() {
        let mut arena = ExprArena::new();
        let root = arena.push_const(core::f32::consts::PI);
        let result = benchmark_jit_arena(&arena, root)
            .expect("constant expression must JIT-compile and benchmark");
        assert_eq!(result.outputs.len(), INPUT_TUPLES);
        for lanes in &result.outputs {
            for lane in lanes {
                assert!(
                    (*lane - core::f32::consts::PI).abs() < 1e-5,
                    "expected all lanes ~PI at every input tuple, got {}",
                    lane
                );
            }
        }
        assert!(
            result.ns >= 0.0,
            "timing must be non-negative, got {}",
            result.ns
        );
    }

    #[test]
    fn autoscale_reaches_tick_floor() {
        // A 1-op kernel at repeat_batches=1 is 64 evals ≈ hundreds of ns —
        // far below the ~4.2µs tick floor — so autoscale must engage.
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let root = arena.push_binary(OpKind::Add, x, y);
        let result = benchmark_jit_arena(&arena, root).expect("tiny kernel must benchmark");
        assert!(
            result.repeat_batches > 1,
            "autoscale should have raised repeat_batches above 1 for a 1-op kernel, got {}",
            result.repeat_batches
        );
        let median_sample_ns = result.ns * (INPUT_TUPLES * result.repeat_batches) as f64;
        assert!(
            median_sample_ns >= MIN_TIMED_SAMPLE_NS as f64,
            "median timed sample {}ns is below the {}ns tick floor",
            median_sample_ns,
            MIN_TIMED_SAMPLE_NS
        );
    }

    #[test]
    fn iqr_should_be_order_independent_and_zero_for_a_flat_sample() {
        // 0..20: q25 = sorted[5] = 5, q75 = sorted[15] = 15.
        let samples: Vec<f64> = (0..20).map(f64::from).collect();
        assert_eq!(iqr(&samples), 10.0);

        // Order-independent: iqr sorts internally.
        let mut shuffled = samples.clone();
        shuffled.reverse();
        shuffled.swap(3, 17);
        assert_eq!(iqr(&shuffled), 10.0);

        // Zero dispersion.
        let flat = [7.5f64; 20];
        assert_eq!(iqr(&flat), 0.0);
    }

    #[test]
    #[should_panic(expected = "plausibility failure")]
    fn plausibility_floor_fires_below_floor() {
        // 10 source ops → at most 5 issued after fusion → floor 0.25ns;
        // 0.1ns raw is a harness bug.
        assert_plausible(0.1, 10);
    }

    #[test]
    fn plausibility_floor_allows_full_fma_fusion() {
        // The floor must not reject a kernel whose every Mul fused into the
        // Add above it: 2N source nodes can issue as N `FMLA`s, so the floor
        // has to sit at or below N * MIN_NS_PER_OP. Without the halving this
        // is 2N * MIN_NS_PER_OP and correct measurements panic — the exact
        // false positive `measure_latency_prior` hit on current main.
        let source_ops = 48;
        let fully_fused_ns = (source_ops / 2) as f64 * MIN_NS_PER_OP;
        assert_plausible(fully_fused_ns, source_ops);
    }

    #[test]
    #[should_panic(expected = "plausibility failure")]
    fn plausibility_floor_fires_on_zero_raw() {
        // 1 op → floor 0.05ns; a raw 0 for real ops is an early-`ret` shape.
        assert_plausible(0.0, 1);
    }

    #[test]
    fn plausibility_floor_spares_constant_folded() {
        // op_count == 0: folded to a leaf — ~0 raw is legitimate.
        assert_plausible(0.0, 0);
    }

    /// A var-free arena with `pairs * 2` ops: nothing in it depends on an
    /// input, so a constant-folding compiler could emit one `ret`.
    fn var_free_multi_op_arena(pairs: usize) -> (ExprArena, ExprId) {
        let mut arena = ExprArena::new();
        let mut acc = arena.push_const(1.0);
        for i in 0..pairs {
            let c = arena.push_const(1.0 + i as f32);
            acc = arena.push_binary(OpKind::Add, acc, c);
            let d = arena.push_const(1.0 + i as f32 * 0.5);
            acc = arena.push_binary(OpKind::Mul, acc, d);
        }
        (arena, acc)
    }

    #[test]
    fn var_free_arenas_keep_their_full_op_count() {
        // The floor is computed from the arena the caller hands us, var-free
        // or not — see `has_reachable_var` for why no exemption exists.
        let (arena, root) = var_free_multi_op_arena(3);
        assert_eq!(op_count(&arena, root), 6);
        assert!(!has_reachable_var(&arena, root));

        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let c = arena.push_const(2.0);
        let root = arena.push_binary(OpKind::Mul, x, c);
        assert!(has_reachable_var(&arena, root));
        assert_eq!(op_count(&arena, root), 1);
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn var_free_arenas_still_execute_their_ops() {
        // The premise behind exempting var-free arenas from the plausibility
        // floor is that the JIT folds them to a constant `ret`. It does not:
        // `compile` has no constant-propagation pass, so a 300-op var-free
        // kernel really runs 300 ops and sits far above the 15ns floor
        // (measured ~185ns on M-series).
        //
        // If this test ever fails, a folding pass has landed — and the fix is
        // to compute the floor from the LOWERED schedule the emitter built,
        // NOT to exempt var-free expressions, which would switch off the
        // audit-M4 early-`ret` detector for precisely the easiest shape to
        // emit an early `ret` for.
        let (arena, root) = var_free_multi_op_arena(150);
        let ops = op_count(&arena, root);
        assert_eq!(ops, 300);
        let result =
            benchmark_jit_arena(&arena, root).expect("var-free multi-op expression must benchmark");
        assert!(
            result.ns >= plausibility_floor_ns(ops),
            "a var-free {ops}-op kernel measured {:.3}ns, below the {:.3}ns floor — the JIT \
             now folds constants, so the floor must be derived from the lowered schedule",
            result.ns,
            plausibility_floor_ns(ops)
        );
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn benchmark_compiled_times_the_given_code_object() {
        // Fix 3 substrate: a pre-compiled ExecutableCode can be timed
        // directly, with no compile inside the session call, and yields the
        // same outputs as the compile-inside path.
        let (arena, root) = sentinel_arena();
        let compiled = compile(&arena, root).expect("sentinel kernel compiles");
        let mut session = BenchSession::new();
        let via_code = session
            .benchmark_compiled(&compiled.code, &arena, root, BenchMode::Throughput)
            .expect("precompiled object must benchmark");
        let via_arena = session
            .benchmark_arena(&arena, root, BenchMode::Throughput)
            .expect("arena path must benchmark");
        via_code
            .check_equivalence(&via_arena, 0.0)
            .expect("same expression through both entry points: outputs must match exactly");
        assert_eq!(via_code.outputs.len(), INPUT_TUPLES);
    }

    #[test]
    fn plausibility_floor_passes_above_floor() {
        assert_plausible(1.0, 10);
    }

    #[test]
    fn nonpositive_adjusted_is_recorded_not_fatal() {
        // Small kernel in throughput mode: raw clears the floor but the
        // identity-kernel overhead exceeds it (op work hidden under
        // ILP-overlapped call overhead) — adjusted <= 0 must be recorded,
        // never panicked on (the floor bounds RAW time only, audit M4).
        let raw = RawMeasurement {
            ns: 0.6,
            iqr_ns: 0.01,
            repeat_batches: 128,
            mode: BenchMode::Throughput,
            output: [1.0; 4],
            outputs: vec![[1.0; 4]],
        };
        let result = finalize(raw, 5, 0.65, None);
        assert!(result.adjusted_ns < 0.0);
        assert_eq!(result.ns, 0.6);
        assert!(result.sentinel.is_none(), "no session, no sentinel context");
    }

    #[test]
    fn check_equivalence_rejects_nan_vs_finite() {
        let make = |output: [f32; 4]| BenchResult {
            ns: 1.0,
            adjusted_ns: 1.0,
            call_overhead_ns: 0.0,
            iqr_ns: 0.0,
            mode: BenchMode::Throughput,
            repeat_batches: 1,
            output,
            outputs: vec![output],
            sentinel: None,
        };
        let finite = make([1.0, 2.0, 3.0, 4.0]);
        let nan = make([1.0, f32::NAN, 3.0, 4.0]);
        let inf = make([1.0, 2.0, f32::INFINITY, 4.0]);

        // NaN or inf against a finite lane is divergence, never max_diff 0.0.
        assert_eq!(finite.check_equivalence(&nan, 1e-3), Err(f32::INFINITY));
        assert_eq!(finite.check_equivalence(&inf, 1e-3), Err(f32::INFINITY));
        // Both-NaN lanes agree (domain agreement, Tolerance::Numeric's rule).
        assert!(nan.check_equivalence(&nan, 0.0).is_ok());
        // Finite lanes still use the epsilon path.
        let close = make([1.0, 2.0, 3.0, 4.0 + 5e-4]);
        assert!(finite.check_equivalence(&close, 1e-3).is_ok());
        assert!(finite.check_equivalence(&close, 1e-5).is_err());
    }

    #[test]
    fn sentinel_regime_change_detection() {
        let frac = SENTINEL_REGIME_CHANGE_FRAC;
        // The observed benign band (macOS post-build daemons measured 11–19%
        // drift) must NOT abort: it is recorded and post-hoc corrected.
        assert!(!sentinel_drift_exceeded(100.0, 111.0, frac));
        assert!(!sentinel_drift_exceeded(100.0, 119.0, frac));
        assert!(!sentinel_drift_exceeded(100.0, 85.0, frac));
        // At the boundary: 150ns is exactly +50%, still not *beyond* it.
        assert!(!sentinel_drift_exceeded(100.0, 150.0, frac));
        // Beyond ±50%: the E-core regime-change class, both directions. A
        // P→E migration is a 2–3× step, comfortably past this.
        assert!(sentinel_drift_exceeded(100.0, 151.0, frac));
        assert!(sentinel_drift_exceeded(100.0, 250.0, frac));
        assert!(sentinel_drift_exceeded(100.0, 49.0, frac));
    }

    #[test]
    #[should_panic(expected = "sentinel calibration must be positive")]
    fn sentinel_drift_rejects_bad_calibration() {
        let _ = sentinel_drift_exceeded(0.0, 100.0, SENTINEL_REGIME_CHANGE_FRAC);
    }

    #[test]
    fn sentinel_normalization_is_a_clock_correction() {
        // Machine slowed to 1.25x its opening speed: the sentinel reads 125ns
        // against a 100ns calibration, so every nearby label is inflated by
        // 1.25x and the correction factor is 100/125 = 0.8.
        let slowed = SentinelContext {
            local_sentinel_ns: 125.0,
            calibration_ns: 100.0,
        };
        assert!((slowed.normalization().get() - 0.8).abs() < 1e-12);
        // An 80ns label minted under that sentinel corrects to 64ns in the
        // run's opening clock.
        assert!((80.0 * slowed.normalization().get() - 64.0).abs() < 1e-12);

        // Machine sped up: factor > 1 scales deflated labels back up.
        let sped_up = SentinelContext {
            local_sentinel_ns: 80.0,
            calibration_ns: 100.0,
        };
        assert!((sped_up.normalization().get() - 1.25).abs() < 1e-12);

        // No drift: identity.
        let steady = SentinelContext {
            local_sentinel_ns: 100.0,
            calibration_ns: 100.0,
        };
        assert!((steady.normalization().get() - 1.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "non-positive sentinel reading")]
    fn sentinel_normalization_rejects_zero_reading() {
        let _ = SentinelContext {
            local_sentinel_ns: 0.0,
            calibration_ns: 100.0,
        }
        .normalization();
    }

    // ── Clock newtypes + CostLabel (J3/J4) ──────────────────────────────────

    #[test]
    fn local_ns_normalize_is_the_only_path_to_session_ns() {
        // 1.25x slowdown: a 125ns local reading is a 100ns session-clock
        // value, matching SentinelContext::normalization's own worked
        // example.
        let drift = SentinelContext {
            local_sentinel_ns: 125.0,
            calibration_ns: 100.0,
        }
        .normalization();
        let value = LocalNs::new(125.0).normalize(drift);
        assert!((value.get() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn session_ns_sub_is_the_only_way_to_remove_overhead() {
        let a = SessionNs::new(10.0);
        let b = SessionNs::new(3.0);
        assert!(((a - b).get() - 7.0).abs() < 1e-12);
        // LocalNs deliberately has no `Sub` impl — there is no expression
        // `LocalNs::new(10.0) - SessionNs::new(3.0)` that compiles, which is
        // the point: overhead cannot be subtracted before the reading is
        // normalized onto the session clock.
    }

    #[test]
    fn cost_label_mint_matches_the_hand_derived_ordering() {
        // Same drifted-measurement shape as `training::mint`'s tests: a
        // nonzero overhead so the two possible orderings of
        // normalize-then-subtract disagree, and CostLabel::mint must land on
        // the correct one.
        const OVERHEAD_NS: f64 = 4.0;
        const KERNEL_NS: f64 = 3.0;
        const DRIFT: f64 = 1.45;
        const CALIBRATION_NS: f64 = 8.0;
        let bench = BenchResult {
            ns: (KERNEL_NS + OVERHEAD_NS) * DRIFT,
            adjusted_ns: (KERNEL_NS + OVERHEAD_NS) * DRIFT - OVERHEAD_NS,
            call_overhead_ns: OVERHEAD_NS,
            iqr_ns: 0.0,
            mode: BenchMode::Latency,
            repeat_batches: 1,
            output: [0.0; 4],
            outputs: Vec::new(),
            sentinel: Some(SentinelContext {
                calibration_ns: CALIBRATION_NS,
                local_sentinel_ns: CALIBRATION_NS * DRIFT,
            }),
        };
        let label = CostLabel::mint(&bench, BenchPosition(7), "test");
        assert!((label.value().get() - KERNEL_NS).abs() < 1e-9);
        assert_eq!(label.mode(), BenchMode::Latency);
        assert_eq!(label.position(), BenchPosition(7));
        assert_eq!(label.calibration(), bench.sentinel.unwrap());
        assert!((label.drift().get() - 1.0 / DRIFT).abs() < 1e-9);
        assert_eq!(label.target_log_ns(), log_ns(SessionNs::new(KERNEL_NS)));
    }

    #[test]
    #[should_panic(expected = "no sentinel context")]
    fn cost_label_mint_refuses_an_unprotected_measurement() {
        let bench = BenchResult {
            ns: 42.0,
            adjusted_ns: 42.0,
            call_overhead_ns: 0.0,
            iqr_ns: 0.0,
            mode: BenchMode::Throughput,
            repeat_batches: 1,
            output: [0.0; 4],
            outputs: Vec::new(),
            sentinel: None,
        };
        let _ = CostLabel::mint(&bench, BenchPosition(0), "unit test");
    }

    #[test]
    fn op_count_counts_compute_ops_once() {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let c = arena.push_const(2.0);
        let mul = arena.push_binary(OpKind::Mul, x, c);
        // DAG sharing: `mul` referenced twice, counted once.
        let root = arena.push_binary(OpKind::Add, mul, mul);
        assert_eq!(op_count(&arena, root), 2);
        assert_eq!(op_count(&arena, x), 0);
        assert_eq!(op_count(&arena, c), 0);
    }

    #[test]
    fn sentinel_arena_is_moderate_size() {
        let (arena, root) = sentinel_arena();
        // arena.len() is the unique-node count (the arena holds only the
        // sentinel's nodes); node_count_subtree would multiply out the DAG.
        let nodes = arena.len();
        assert!(
            (30..=60).contains(&nodes),
            "sentinel kernel should be 30-60 nodes, got {}",
            nodes
        );
        assert_eq!(op_count(&arena, root), 40);
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn latency_mode_chains_x_independent_expressions() {
        // Audit-H3 regression guard: an expression that never reads Var(0)
        // must still be chain-serialized (prev feeds EVERY lane). Under the
        // old x-lane-only feeding, y+z measured in the throughput regime
        // (raw ~0.8ns, far below the serialized identity overhead ~4ns) and
        // the overhead subtraction drove adjusted_ns negative.
        let mut arena = ExprArena::new();
        let y = arena.push_var(1);
        let z = arena.push_var(2);
        let root = arena.push_binary(OpKind::Add, y, z);
        let mut session = BenchSession::new();
        let bench = session
            .benchmark_arena(&arena, root, BenchMode::Latency)
            .expect("y+z must benchmark in latency mode");
        let overhead = session.call_overhead_ns(BenchMode::Latency);
        assert!(
            bench.ns > overhead * 0.5,
            "x-independent expression measured {:.3}ns raw latency against a {:.3}ns serialized \
             identity overhead — the dependency chain is broken (audit H3)",
            bench.ns,
            overhead
        );
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn session_benchmarks_both_modes() {
        let mut session = BenchSession::new();
        assert!(session.calibration_ns() > 0.0);
        assert!(session.call_overhead_ns(BenchMode::Throughput) > 0.0);
        assert!(session.call_overhead_ns(BenchMode::Latency) > 0.0);
        assert_eq!(session.sentinel_samples().len(), 1);

        let (arena, root) = sentinel_arena();
        let (throughput, latency) = session
            .benchmark_arena_both(&arena, root)
            .expect("sentinel-shaped kernel must benchmark in both modes");

        assert_eq!(throughput.mode, BenchMode::Throughput);
        assert_eq!(latency.mode, BenchMode::Latency);
        assert_eq!(throughput.outputs.len(), INPUT_TUPLES);
        assert_eq!(latency.outputs.len(), INPUT_TUPLES);
        for (result, mode) in [
            (&throughput, BenchMode::Throughput),
            (&latency, BenchMode::Latency),
        ] {
            let expected_overhead = session.call_overhead_ns(mode);
            assert_eq!(result.call_overhead_ns, expected_overhead);
            assert_eq!(result.adjusted_ns, result.ns - expected_overhead);
            assert!(result.iqr_ns >= 0.0);
            // Session results carry the drift context; before the first
            // SENTINEL_INTERVAL boundary the local reading IS the opening
            // calibration, so the clock correction is exactly 1.
            let ctx = result
                .sentinel
                .expect("session-minted results carry a SentinelContext");
            assert_eq!(ctx.calibration_ns, session.calibration_ns());
            assert_eq!(ctx.local_sentinel_ns, session.calibration_ns());
            assert!((ctx.normalization().get() - 1.0).abs() < 1e-12);
        }
        // Outputs are mode-independent pure function values.
        throughput
            .check_equivalence(&latency, 0.0)
            .expect("same kernel, same inputs: outputs must match exactly");
    }
}
