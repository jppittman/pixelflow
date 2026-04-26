//! # Expression Benchmarking with Core Pinning
//!
//! Measures real execution costs of expressions by:
//! 1. Evaluating expressions using scalar f32 operations
//! 2. Pinning the benchmark thread to a specific CPU core
//! 3. Taking multiple samples and reporting median
//!
//! This provides ground-truth costs for training the neural network.
//! While we use scalar evaluation (not SIMD), relative costs correlate
//! well with SIMD execution since all operations scale similarly.

use crate::nnue::{Expr, OpType};

pub struct EvalContext {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::time::Instant;

/// Result of benchmarking an expression.
#[derive(Clone, Debug)]
pub struct BenchmarkResult {
    /// Median execution time in nanoseconds per evaluation.
    pub median_ns: u64,
    /// Minimum execution time observed.
    pub min_ns: u64,
    /// Maximum execution time observed.
    pub max_ns: u64,
    /// Number of iterations run.
    pub iterations: usize,
}

/// Configuration for benchmarking.
#[derive(Clone, Debug)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations (not measured).
    pub warmup_iterations: usize,
    /// Number of measured iterations.
    pub measure_iterations: usize,
    /// Number of evaluations per iteration (for amortizing timing overhead).
    pub evals_per_iteration: usize,
    /// CPU core to pin to (None = don't pin).
    pub pin_to_core: Option<usize>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 100,
            measure_iterations: 1000,
            evals_per_iteration: 100,
            pin_to_core: Some(0),
        }
    }
}

/// Evaluate expression using scalar f32 arithmetic.
///
/// This is the core evaluation function used for benchmarking.
/// It uses standard f32 operations which correlate well with SIMD
/// performance for relative cost ranking.
pub fn eval_expr_scalar(expr: &Expr, ctx: &EvalContext) -> f32 {
    match expr {
        Expr::Var(0) => ctx.x,
        Expr::Var(1) => ctx.y,
        Expr::Var(2) => ctx.z,
        Expr::Var(3) => ctx.w,
        Expr::Var(_) => 0.0,
        Expr::Const(c) => *c,
        Expr::Binary(op, lhs, rhs) => {
            let l = eval_expr_scalar(lhs, ctx);
            let r = eval_expr_scalar(rhs, ctx);
            match op {
                OpType::Add => l + r,
                OpType::Sub => l - r,
                OpType::Mul => l * r,
                OpType::Div => l / r,
                OpType::Min => l.min(r),
                OpType::Max => l.max(r),
                _ => l,
            }
        }
        Expr::Unary(op, arg) => {
            let a = eval_expr_scalar(arg, ctx);
            match op {
                OpType::Neg => -a,
                OpType::Sqrt => libm::sqrtf(a),
                OpType::Rsqrt => 1.0 / libm::sqrtf(a),
                OpType::Abs => libm::fabsf(a),
                _ => a,
            }
        }
        Expr::Ternary(op, a, b, c) => {
            let va = eval_expr_scalar(a, ctx);
            let vb = eval_expr_scalar(b, ctx);
            let vc = eval_expr_scalar(c, ctx);
            match op {
                OpType::MulAdd => libm::fmaf(va, vb, vc),
                OpType::MulRsqrt => va / libm::sqrtf(vb),
                _ => va,
            }
        }
    }
}

/// Pin current thread to a specific CPU core.
///
/// Returns Ok(()) on success, Err with message on failure.
#[cfg(all(feature = "std", target_os = "linux"))]
pub fn pin_to_core(core_id: usize) -> Result<(), &'static str> {
    use std::mem;

    unsafe {
        let mut set: libc::cpu_set_t = mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(core_id, &mut set);

        let result = libc::sched_setaffinity(0, mem::size_of::<libc::cpu_set_t>(), &set);
        if result == 0 {
            Ok(())
        } else {
            Err("sched_setaffinity failed")
        }
    }
}

#[cfg(all(feature = "std", target_os = "macos"))]
pub fn pin_to_core(_core_id: usize) -> Result<(), &'static str> {
    // macOS doesn't support thread affinity directly
    Ok(())
}

#[cfg(all(feature = "std", not(any(target_os = "linux", target_os = "macos"))))]
pub fn pin_to_core(_core_id: usize) -> Result<(), &'static str> {
    Ok(())
}

/// Benchmark an expression's execution time.
///
/// Returns detailed timing statistics.
#[cfg(feature = "std")]
pub fn benchmark_expr(expr: &Expr, config: &BenchmarkConfig) -> BenchmarkResult {
    // Pin to core if requested
    if let Some(core) = config.pin_to_core {
        let _ = pin_to_core(core);
    }

    // Test coordinates
    let x = 0.5f32;
    let y = 1.5f32;
    let z = 2.5f32;
    let w = 1.0f32;

    // Warmup
    for _ in 0..config.warmup_iterations {
        for _ in 0..config.evals_per_iteration {
            let ctx = EvalContext { x, y, z, w };
            let _ = std::hint::black_box(eval_expr_scalar(
                std::hint::black_box(expr),
                &ctx,
            ));
        }
    }

    // Measure
    let mut times_ns = Vec::with_capacity(config.measure_iterations);

    for _ in 0..config.measure_iterations {
        let start = Instant::now();
        for _ in 0..config.evals_per_iteration {
            let ctx = EvalContext { x, y, z, w };
            let _ = std::hint::black_box(eval_expr_scalar(
                std::hint::black_box(expr),
                &ctx,
            ));
        }
        let elapsed = start.elapsed();
        let ns_per_eval = elapsed.as_nanos() as u64 / config.evals_per_iteration as u64;
        times_ns.push(ns_per_eval);
    }

    // Sort for percentiles
    times_ns.sort_unstable();

    let median_ns = times_ns[times_ns.len() / 2];
    let min_ns = times_ns[0];
    let max_ns = times_ns[times_ns.len() - 1];

    BenchmarkResult {
        median_ns,
        min_ns,
        max_ns,
        iterations: config.measure_iterations,
    }
}

/// Benchmark multiple expressions in parallel across cores.
///
/// Each expression is benchmarked on its own pinned core for isolation.
/// Returns results in the same order as input expressions.
#[cfg(feature = "std")]
pub fn benchmark_parallel(
    exprs: &[Expr],
    config: &BenchmarkConfig,
    num_threads: usize,
) -> Vec<BenchmarkResult> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    let num_threads = num_threads.min(exprs.len()).max(1);
    if num_threads <= 1 || exprs.len() <= 1 {
        return exprs.iter().map(|e| benchmark_expr(e, config)).collect();
    }

    // Pre-allocate results with mutex protection for each slot
    let results: Vec<Mutex<Option<BenchmarkResult>>> = (0..exprs.len())
        .map(|_| Mutex::new(None))
        .collect();

    let next_idx = AtomicUsize::new(0);

    std::thread::scope(|s| {
        for thread_id in 0..num_threads {
            let exprs = exprs;
            let config = config;
            let next_idx = &next_idx;
            let results = &results;

            s.spawn(move || {
                // Each thread pins to its own core
                let mut thread_config = config.clone();
                thread_config.pin_to_core = Some(thread_id);

                loop {
                    let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                    if idx >= exprs.len() {
                        break;
                    }

                    let result = benchmark_expr(&exprs[idx], &thread_config);

                    // Each index is processed by exactly one thread
                    *results[idx].lock().unwrap() = Some(result);
                }
            });
        }
    });

    // Extract results - all slots should be filled
    results
        .into_iter()
        .map(|m| m.into_inner().unwrap().expect("unfilled benchmark slot"))
        .collect()
}

/// Quick benchmark with default settings.
#[cfg(feature = "std")]
pub fn quick_benchmark(expr: &Expr) -> u64 {
    let config = BenchmarkConfig {
        warmup_iterations: 10,
        measure_iterations: 100,
        evals_per_iteration: 100,
        pin_to_core: Some(0),
    };
    benchmark_expr(expr, &config).median_ns
}

/// Estimate cost based on node count and operation weights.
///
/// This provides a fast approximation without actual benchmarking,
/// useful for generating synthetic training data.
pub fn estimate_cost(expr: &Expr) -> usize {
    match expr {
        Expr::Var(_) => 0,  // Free (register access)
        Expr::Const(_) => 0, // Free (immediate or register)
        Expr::Binary(op, lhs, rhs) => {
            let base = match op {
                OpType::Add | OpType::Sub => 1,
                OpType::Mul => 1,
                OpType::Div => 10, // Division is expensive
                OpType::Min | OpType::Max => 1,
                _ => 1,
            };
            base + estimate_cost(lhs) + estimate_cost(rhs)
        }
        Expr::Unary(op, arg) => {
            let base = match op {
                OpType::Neg | OpType::Abs => 1,
                OpType::Sqrt => 5,
                OpType::Rsqrt => 3, // Fast approximation
                _ => 1,
            };
            base + estimate_cost(arg)
        }
        Expr::Ternary(op, a, b, c) => {
            let base = match op {
                OpType::MulAdd => 1, // FMA is single instruction
                OpType::MulRsqrt => 4,
                _ => 2,
            };
            base + estimate_cost(a) + estimate_cost(b) + estimate_cost(c)
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn test_eval_expr_scalar_simple() {
        // Test: x + y
        let expr = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        );

        let ctx = EvalContext { x: 3.0, y: 4.0, z: 0.0, w: 0.0 };
        let result = eval_expr_scalar(&expr, &ctx);
        assert!((result - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_eval_expr_scalar_nested() {
        // Test: (x * y) + (z - w)
        let expr = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Binary(
                OpType::Mul,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
            )),
            Box::new(Expr::Binary(
                OpType::Sub,
                Box::new(Expr::Var(2)),
                Box::new(Expr::Var(3)),
            )),
        );

        // (3 * 4) + (5 - 2) = 12 + 3 = 15
        let ctx = EvalContext { x: 3.0, y: 4.0, z: 5.0, w: 2.0 };
        let result = eval_expr_scalar(&expr, &ctx);
        assert!((result - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_benchmark_runs() {
        let expr = Expr::Binary(
            OpType::Mul,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        );

        let config = BenchmarkConfig {
            warmup_iterations: 1,
            measure_iterations: 10,
            evals_per_iteration: 10,
            pin_to_core: None,
        };

        let result = benchmark_expr(&expr, &config);
        assert!(result.min_ns <= result.median_ns);
        assert!(result.median_ns <= result.max_ns);
    }

    #[test]
    fn test_estimate_cost() {
        // Simple add
        let simple = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        );

        // Division (expensive)
        let with_div = Expr::Binary(
            OpType::Div,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        );

        assert!(estimate_cost(&with_div) > estimate_cost(&simple));
    }
}
