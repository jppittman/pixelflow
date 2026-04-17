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
unsafe extern "C" {
    fn mach_absolute_time() -> u64;
}

#[cfg(target_os = "macos")]
fn nanos_now() -> u64 {
    // On Apple Silicon, mach_absolute_time() ticks == nanoseconds (timebase 1:1).
    unsafe { mach_absolute_time() }
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

/// JIT-compile and benchmark one expression. Returns ns/eval (median of TIMED_RUNS).
///
/// Rejects:
/// - NaN or negative timings
/// - Absurdly large timings (>1s, indicating measurement failure)
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

            return validate_median(ns).map(|ns| BenchResult { ns, output });
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::*;
        unsafe {
            use pixelflow_ir::backend::emit::executable::KernelFn;
            let func: KernelFn = exec_code.as_fn();

            let x = _mm_set1_ps(0.5);
            let y = _mm_set1_ps(0.7);
            let z = _mm_set1_ps(1.3);
            let w = _mm_set1_ps(-0.2);

            for _ in 0..WARMUP_ITERS {
                std::hint::black_box(func(x, y, z, w));
            }

            let out = func(x, y, z, w);
            let mut output = [0.0f32; 4];
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

            return validate_median(ns).map(|ns| BenchResult { ns, output });
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

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::ExprArena;

    #[test]
    fn constant_expr_benchmarks_successfully() {
        let mut arena = ExprArena::new();
        let root = arena.push_const(3.14);
        let result = benchmark_jit_arena(&arena, root)
            .expect("constant expression must JIT-compile and benchmark");
        for lane in &result.output {
            assert!(
                (*lane - 3.14).abs() < 1e-5,
                "expected all lanes ~3.14, got {}",
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
