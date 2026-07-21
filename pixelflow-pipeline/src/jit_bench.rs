//! Shared JIT benchmarking infrastructure.
//!
//! Shared helpers for timing arena-native JIT kernels.

use std::fmt;

use pixelflow_ir::backend::emit::compile_arena_dag;
use pixelflow_ir::{ExprArena, ExprId};

/// Number of timed samples per expression. Take the median.
/// 20 samples × 100 inner iters = 2000 evals. Enough to resolve
/// sub-nanosecond differences on small expressions.
const TIMED_RUNS: usize = 20;

/// Warmup iterations before timed runs. Warms icache, branch predictor,
/// and microarchitectural state so every benchmark_jit() call measures
/// hot performance — not cold-start overhead.
const WARMUP_ITERS: usize = 64;

/// Inner iterations per timed sample. Must be large enough that the total
/// time exceeds clock resolution (~1ns on Apple Silicon). The inner loop
/// is manually unrolled 10x to eliminate loop counter overhead — a constant
/// ~0.3ns/iter bias that corrupts the additive cost model the NNUE is
/// learning (small expressions appear proportionally more expensive).
const INNER_ITERS: usize = 100;

/// Fully unrolled inner loop — 100 evals, zero loop counter overhead.
/// At INNER_ITERS=100 this means exactly 1 outer iteration with no loop
/// branch at all. Eliminates the ~0.3ns/iter constant bias that corrupts
/// the additive cost model for small expressions.
macro_rules! eval100 {
    ($func:expr, $x:expr, $y:expr, $z:expr, $w:expr) => {{
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
        std::hint::black_box($func($x, $y, $z, $w));
    }};
}

/// Maximum plausible single-eval time: 1 second.
/// Anything above this is a timing artifact (e.g., u64 underflow in
/// `nanos_now() - start`, OS scheduling jitter, or JIT codegen bug).
const MAX_PLAUSIBLE_NS: f64 = 1_000_000_000.0;

/// Errors from JIT benchmarking.
#[derive(Debug)]
pub enum BenchError {
    /// JIT compilation failed.
    CompileFailed(&'static str),
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

/// Result of benchmarking: timing and the full SIMD output for correctness checks.
pub struct BenchResult {
    /// Median ns per evaluation.
    pub ns: f64,
    /// All 4 SIMD lanes at the test point (0.5, 0.7, 1.3, -0.2).
    pub output: [f32; 4],
}

impl BenchResult {
    /// Check that two results compute the same function (all lanes within epsilon).
    /// Returns Err with the max divergence if any lane differs.
    pub fn check_equivalence(&self, other: &BenchResult, epsilon: f32) -> Result<(), f32> {
        let mut max_diff: f32 = 0.0;
        for i in 0..4 {
            let diff = (self.output[i] - other.output[i]).abs();
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
}

fn benchmark_exec_code(
    exec_code: pixelflow_ir::backend::emit::executable::ExecutableCode,
    repeat_batches: usize,
) -> Result<BenchResult, BenchError> {
    let repeat_batches = repeat_batches.max(1);

    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        unsafe {
            use pixelflow_ir::backend::emit::executable::KernelFn;
            let func: KernelFn = exec_code.as_fn();

            let x = vdupq_n_f32(0.5);
            let y = vdupq_n_f32(0.7);
            let z = vdupq_n_f32(1.3);
            let w = vdupq_n_f32(-0.2);

            for _ in 0..WARMUP_ITERS {
                std::hint::black_box(func(x, y, z, w));
            }

            let out = func(x, y, z, w);
            let mut output = [0.0f32; 4];
            vst1q_f32(output.as_mut_ptr(), out);

            let mut times = [0u64; TIMED_RUNS];
            for t in &mut times {
                let start = nanos_now();
                for _ in 0..repeat_batches {
                    eval100!(func, x, y, z, w);
                }
                *t = nanos_now() - start;
            }

            times.sort_unstable();
            let median_total = times[TIMED_RUNS / 2];
            let ns = median_total as f64 / (INNER_ITERS * repeat_batches) as f64;

            // `return` is required: this aarch64 block is syntactically followed
            // by the cfg'd-out x86_64/fallback blocks, so it is not a tail expr.
            #[allow(clippy::needless_return)]
            return validate_median(ns).map(|ns| BenchResult { ns, output });
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::*;
        unsafe {
            use pixelflow_ir::backend::emit::executable::KernelFn;
            let func: KernelFn = exec_code.as_fn();

            // Inputs at the KernelFn's SIMD width: __m512 under +avx512f, else
            // __m128 (SSE2). eval100! is width-agnostic (just calls func).
            #[cfg(target_feature = "avx512f")]
            let (x, y, z, w) = (
                _mm512_set1_ps(0.5),
                _mm512_set1_ps(0.7),
                _mm512_set1_ps(1.3),
                _mm512_set1_ps(-0.2),
            );
            #[cfg(not(target_feature = "avx512f"))]
            let (x, y, z, w) = (
                _mm_set1_ps(0.5),
                _mm_set1_ps(0.7),
                _mm_set1_ps(1.3),
                _mm_set1_ps(-0.2),
            );

            for _ in 0..WARMUP_ITERS {
                std::hint::black_box(func(x, y, z, w));
            }

            let out = func(x, y, z, w);
            // BenchResult.output keeps the first 4 lanes regardless of width.
            let mut output = [0.0f32; 4];
            #[cfg(target_feature = "avx512f")]
            {
                let mut wide = [0.0f32; 16];
                _mm512_storeu_ps(wide.as_mut_ptr(), out);
                output.copy_from_slice(&wide[..4]);
            }
            #[cfg(not(target_feature = "avx512f"))]
            _mm_storeu_ps(output.as_mut_ptr(), out);

            let mut times = [0u64; TIMED_RUNS];
            for t in &mut times {
                let start = nanos_now();
                for _ in 0..repeat_batches {
                    eval100!(func, x, y, z, w);
                }
                *t = nanos_now() - start;
            }

            times.sort_unstable();
            let median_total = times[TIMED_RUNS / 2];
            let ns = median_total as f64 / (INNER_ITERS * repeat_batches) as f64;

            validate_median(ns).map(|ns| BenchResult { ns, output })
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = exec_code;
        Err(BenchError::UnsupportedArch)
    }
}

/// JIT-compile and benchmark an arena expression. No `Expr` conversion.
pub fn benchmark_jit_arena(arena: &ExprArena, root: ExprId) -> Result<BenchResult, BenchError> {
    benchmark_jit_arena_repeated(arena, root, 1)
}

/// Validation-only heavier benchmark path.
///
/// `repeat_batches=100` means each timed sample performs 10_000 evals
/// (100 batches × 100 unrolled inner iterations), which is enough to lift
/// tiny held-out shader expressions above timer noise without slowing the
/// regular training path.
pub fn benchmark_jit_arena_repeated(
    arena: &ExprArena,
    root: ExprId,
    repeat_batches: usize,
) -> Result<BenchResult, BenchError> {
    let result = compile_arena_dag(arena, root).map_err(BenchError::CompileFailed)?;
    benchmark_exec_code(result.code, repeat_batches)
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

/// Code-buffer capacity for the reused-buffer path. 256KB is generous for
/// expressions up to ~2000 nodes (`CompileWorkspace` docs say 64KB covers
/// ~500 nodes).
#[cfg(target_arch = "aarch64")]
const REUSED_CODE_CAPACITY: usize = 256 * 1024;

/// Result of a compile-cost measurement.
pub struct CompileCostResult {
    /// Median ns for one complete compile.
    pub ns: f64,
    /// Emitted machine-code size in bytes.
    pub code_bytes: usize,
}

/// Median wall-clock cost of one `compile_arena_dag` call, fresh-allocation
/// path: every compile mmaps a new executable region and munmaps it on drop.
/// Both syscalls are inside the timed window — this is the full per-compile
/// lifecycle a naive per-kernel JIT would pay.
pub fn benchmark_compile_fresh(
    arena: &ExprArena,
    root: ExprId,
) -> Result<CompileCostResult, BenchError> {
    for _ in 0..COMPILE_WARMUP_ITERS {
        let result = compile_arena_dag(arena, root).map_err(BenchError::CompileFailed)?;
        std::hint::black_box(result.code.as_bytes().first());
    }

    let mut times = [0u64; COMPILE_TIMED_RUNS];
    let mut code_bytes = 0usize;
    for t in &mut times {
        let start = nanos_now();
        let result = compile_arena_dag(arena, root).map_err(BenchError::CompileFailed)?;
        std::hint::black_box(result.code.as_bytes().first());
        code_bytes = result.code.len();
        drop(result); // munmap inside the timed window
        *t = nanos_now() - start;
    }

    times.sort_unstable();
    let median = times[COMPILE_TIMED_RUNS / 2] as f64;
    validate_median(median).map(|ns| CompileCostResult { ns, code_bytes })
}

/// Median wall-clock cost of one compile into a reused
/// [`CompileWorkspace`](pixelflow_ir::backend::emit::CompileWorkspace):
/// the executable region is mmap'd once up front, and each compile pays only
/// `pthread_jit_write_protect_np` toggles + icache invalidation (Apple
/// Silicon) instead of mmap/munmap. This is the amortized cost gate G0 cares
/// about. aarch64 only — `CompileWorkspace` has no x86-64 counterpart.
///
/// Returns median ns per compile. Note the workspace path skips the lowering
/// passes (`expand_reduce`/`expand_gather`/`expand_transcendentals`), so the
/// arena must contain only directly-emittable ops for the comparison against
/// [`benchmark_compile_fresh`] to be apples-to-apples.
#[cfg(target_arch = "aarch64")]
pub fn benchmark_compile_reused(arena: &ExprArena, root: ExprId) -> Result<f64, BenchError> {
    use pixelflow_ir::backend::emit::CompileWorkspace;

    let mut ws = CompileWorkspace::new(REUSED_CODE_CAPACITY).map_err(BenchError::CompileFailed)?;

    // SAFETY: the returned KernelFn is never called, and is discarded before
    // the next compile_arena call overwrites the buffer.
    for _ in 0..COMPILE_WARMUP_ITERS {
        let func = unsafe { ws.compile_arena(arena, root) }.map_err(BenchError::CompileFailed)?;
        std::hint::black_box(func);
    }

    let mut times = [0u64; COMPILE_TIMED_RUNS];
    for t in &mut times {
        let start = nanos_now();
        // SAFETY: as above — pointer discarded, never called.
        let func = unsafe { ws.compile_arena(arena, root) }.map_err(BenchError::CompileFailed)?;
        std::hint::black_box(func);
        *t = nanos_now() - start;
    }

    times.sort_unstable();
    let median = times[COMPILE_TIMED_RUNS / 2] as f64;
    validate_median(median)
}

/// Convert nanoseconds to log-nanoseconds (floored at 1e-3ns, capped at 1s).
///
/// Relocated from the deleted `training::gen_es` (the ES-guided corpus-growth
/// optimizer it lived in was RL-adjacent scaffolding removed per
/// docs/plans/2026-07-07-guided-saturation-redesign.md); this conversion
/// itself is just a unit change on a [`BenchResult::ns`] measurement, used by
/// the surviving supervised extraction-head training path.
///
/// # Panics
///
/// Panics if `ns` is NaN.
#[must_use]
pub fn log_ns(ns: f64) -> f32 {
    assert!(!ns.is_nan(), "log_ns called with NaN");
    // NaN already rejected above, so clamp's total order is well-defined here.
    let clamped = ns.clamp(1e-3, 1e9);
    libm::logf(clamped as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::ExprArena;

    #[test]
    fn verify_log_ns() {
        // log(1.0) = 0.0
        let v = log_ns(1.0);
        assert!(
            libm::fabsf(v) < 0.001,
            "log_ns(1.0) should be ~0.0, got {}",
            v
        );

        // Values below 1e-3 should be floored to 1e-3.
        let v_low = log_ns(0.0001);
        let v_floor = log_ns(1e-3);
        assert!(
            libm::fabsf(v_low - v_floor) < 0.001,
            "log_ns(0.0001) should equal log_ns(1e-3), got {} vs {}",
            v_low,
            v_floor
        );

        // log(e) ≈ 1.0
        let v_e = log_ns(core::f64::consts::E);
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
        for lane in &result.output {
            assert!(
                (*lane - core::f32::consts::PI).abs() < 1e-5,
                "expected all lanes ~PI, got {}",
                lane
            );
        }
        assert!(
            result.ns >= 0.0,
            "timing must be non-negative, got {}",
            result.ns
        );
    }
}
