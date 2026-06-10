//! Comprehensive benchmarks for pixelflow-core
//!
//! Tests SIMD operations, manifold evaluation, and composition performance.

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use pixelflow_core::{
    FastMathGuard, Field, ManifoldCompat, ManifoldExt, PARALLELISM, X, Y, Z, combinators::Fix,
    jet::Jet2,
};

// ============================================================================
// SIMD Field Benchmarks
// ============================================================================

fn bench_field_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_creation");

    group.bench_function("from_f32_splat", |b| {
        b.iter(|| black_box(Field::from(core::f32::consts::PI)))
    });

    group.bench_function("sequential", |b| {
        b.iter(|| black_box(Field::sequential(0.5)))
    });

    group.finish();
}

fn bench_field_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_arithmetic");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let a = Field::sequential(1.0);
    let b = Field::sequential(2.0);

    // Note: Field ops return AST types, so we call .constant() to evaluate
    group.bench_function("add", |bencher| {
        bencher.iter(|| black_box((black_box(a) + black_box(b)).constant()))
    });

    group.bench_function("sub", |bencher| {
        bencher.iter(|| black_box((black_box(a) - black_box(b)).constant()))
    });

    group.bench_function("mul", |bencher| {
        bencher.iter(|| black_box((black_box(a) * black_box(b)).constant()))
    });

    group.bench_function("div", |bencher| {
        bencher.iter(|| black_box((black_box(a) / black_box(b)).constant()))
    });

    group.bench_function("chained_mad", |bencher| {
        // Multiply-add chain: a * b + c → MulAdd (fused to FMA)
        let c = Field::from(0.5);
        bencher.iter(|| black_box((black_box(a) * black_box(b) + black_box(c)).constant()))
    });

    group.finish();
}

fn bench_field_math(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_math");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let a = Field::sequential(1.0);
    let b = Field::sequential(2.0);

    group.bench_function("sqrt", |bencher| {
        bencher.iter(|| {
            let val = black_box(a);
            black_box(val.sqrt())
        })
    });

    group.bench_function("abs", |bencher| {
        let neg = Field::from(-3.5);
        bencher.iter(|| {
            let val = black_box(neg);
            black_box(val.abs())
        })
    });

    group.bench_function("min", |bencher| {
        bencher.iter(|| black_box(black_box(a).min(black_box(b))))
    });

    group.bench_function("max", |bencher| {
        bencher.iter(|| black_box(black_box(a).max(black_box(b))))
    });

    group.finish();
}

fn bench_field_transcendental(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_transcendental");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    // Test across different ranges to measure average performance
    let small = Field::sequential(0.1); // [0.1, 0.1+PARALLELISM*step]
    let mid = Field::sequential(1.0); // [1.0, 1.0+PARALLELISM*step]
    let large = Field::sequential(10.0); // [10.0, 10.0+PARALLELISM*step]

    group.bench_function("log2_small", |bencher| {
        bencher.iter(|| {
            let val = black_box(small);
            black_box(val.log2())
        })
    });

    group.bench_function("log2_mid", |bencher| {
        bencher.iter(|| {
            let val = black_box(mid);
            black_box(val.log2())
        })
    });

    group.bench_function("log2_large", |bencher| {
        bencher.iter(|| {
            let val = black_box(large);
            black_box(val.log2())
        })
    });

    group.bench_function("exp2_small", |bencher| {
        bencher.iter(|| {
            let val = black_box(small);
            black_box(val.exp2())
        })
    });

    group.bench_function("exp2_mid", |bencher| {
        bencher.iter(|| {
            let val = black_box(mid);
            black_box(val.exp2())
        })
    });

    group.bench_function("exp2_large", |bencher| {
        bencher.iter(|| {
            let val = black_box(large);
            black_box(val.exp2())
        })
    });

    // Roundtrip to measure compounded cost
    group.bench_function("log2_exp2_roundtrip", |bencher| {
        bencher.iter(|| {
            let val = black_box(mid);
            black_box(val.log2().exp2())
        })
    });

    group.finish();
}

fn bench_field_comparisons(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_comparisons");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Field::sequential(0.0);
    let y = Field::from(5.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    // Use manifold comparisons which return Field when evaluated
    group.bench_function("lt_manifold", |bencher| {
        let m = X.lt(2.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), y, z, w)))
    });

    group.bench_function("le_manifold", |bencher| {
        let m = X.le(2.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), y, z, w)))
    });

    group.bench_function("gt_manifold", |bencher| {
        let m = X.gt(2.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), y, z, w)))
    });

    group.bench_function("ge_manifold", |bencher| {
        let m = X.ge(2.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), y, z, w)))
    });

    group.finish();
}

fn bench_field_select(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_select");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Field::sequential(0.0);
    let y = Field::from(5.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    // Both compute the comparison inside the loop for fair comparison
    group.bench_function("select_with_gt_ast", |bencher| {
        // Select with Gt<X, f32> AST condition - goes through FieldCondition
        let m = X.gt(2.0f32).select(1.0f32, 0.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), y, z, w)))
    });

    group.bench_function("select_with_field_condition", |bencher| {
        // Select with pre-computed Field mask - UNFAIR: mask computed once!
        let mask = x.gt(Field::from(2.0)); // computed once outside loop
        let m = mask.select(Field::from(1.0), Field::from(0.0));
        bencher.iter(|| black_box(m.eval_raw(black_box(x), y, z, w)))
    });

    group.bench_function("select_gt_recompute_each_iter", |bencher| {
        // Compute mask inside loop - fair comparison with AST path
        let if_true = Field::from(1.0);
        let if_false = Field::from(0.0);
        bencher.iter(|| {
            let mask = black_box(x).gt(Field::from(2.0));
            let m = mask.select(if_true, if_false);
            black_box(m.eval_raw(x, y, z, w))
        })
    });

    group.finish();
}

fn bench_field_bitwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_bitwise");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Field::sequential(0.0);
    let y = Field::from(5.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    group.bench_function("and_manifold", |bencher| {
        // x > 0 AND x < 3
        let m = X.gt(0.0f32) & X.lt(3.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), y, z, w)))
    });

    group.bench_function("or_manifold", |bencher| {
        // x < 1 OR x > 2
        let m = X.lt(1.0f32) | X.gt(2.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), y, z, w)))
    });

    group.finish();
}

// ============================================================================
// Manifold Evaluation Benchmarks
// ============================================================================

fn bench_manifold_constants(c: &mut Criterion) {
    let mut group = c.benchmark_group("manifold_constants");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Field::sequential(0.0);
    let y = Field::from(5.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    group.bench_function("f32_constant", |bencher| {
        let m = 42.0f32;
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("X_variable", |bencher| {
        bencher.iter(|| black_box(X.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("Y_variable", |bencher| {
        bencher.iter(|| black_box(Y.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.finish();
}

fn bench_manifold_simple(c: &mut Criterion) {
    let mut group = c.benchmark_group("manifold_simple");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Field::sequential(0.0);
    let y = Field::from(5.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    group.bench_function("X_plus_Y", |bencher| {
        let m = X + Y;
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("X_mul_Y", |bencher| {
        let m = X * Y;
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("X_squared", |bencher| {
        let m = X * X;
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    // FMA benchmark: X * Y + Z goes through MulAdd combinator
    group.bench_function("fma_X_mul_Y_plus_Z", |bencher| {
        let m = X * Y + Z; // This is MulAdd<X, Y, Z> - uses vfmadd instruction
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("distance_squared", |bencher| {
        // x² + y² - this is MulAdd<X, X, MulAdd<Y, Y, ...>> due to chaining
        let m = X * X + Y * Y;
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("distance_from_origin", |bencher| {
        // √(x² + y²)
        let m = (X * X + Y * Y).sqrt();
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.finish();
}

fn bench_manifold_circle(c: &mut Criterion) {
    let mut group = c.benchmark_group("manifold_circle");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Field::sequential(0.0);
    let y = Field::from(5.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    group.bench_function("unit_circle_sdf", |bencher| {
        // Signed distance: √(x² + y²) - 1
        let m = (X * X + Y * Y).sqrt() - 1.0f32;
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("circle_inside_test", |bencher| {
        // x² + y² < 1
        let m = (X * X + Y * Y).lt(1.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.finish();
}

fn bench_manifold_select(c: &mut Criterion) {
    let mut group = c.benchmark_group("manifold_select");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Field::sequential(0.0);
    let y = Field::from(5.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    group.bench_function("simple_select", |bencher| {
        // if x < 2 then 1.0 else 0.0
        let m = X.lt(2.0f32).select(1.0f32, 0.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("circle_select", |bencher| {
        // if inside circle then 1.0 else 0.0
        let m = (X * X + Y * Y).lt(100.0f32).select(1.0f32, 0.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("nested_select", |bencher| {
        // Nested: if x < 2 then (if y < 3 then 1 else 0.5) else 0
        let inner = Y.lt(3.0f32).select(1.0f32, 0.5f32);
        let m = X.lt(2.0f32).select(inner, 0.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.finish();
}

fn bench_manifold_complex(c: &mut Criterion) {
    let mut group = c.benchmark_group("manifold_complex");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Field::sequential(0.0);
    let y = Field::from(5.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    group.bench_function("polynomial_degree3", |bencher| {
        // x³ + 2x² - 5x + 3
        let m = X * X * X + X * X * 2.0f32 - X * 5.0f32 + 3.0f32;
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("bilinear_interp", |bencher| {
        // Bilinear-like computation using manifold operations
        // x*y for corner blending pattern
        let m = X * Y * 3.0f32 + X * 0.5f32 + Y * 0.25f32;
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("min_max_chain", |bencher| {
        let m = X.max(Y).min(10.0f32).max(0.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.finish();
}

// ============================================================================
// Jet2 Auto-Differentiation Benchmarks
// ============================================================================

fn bench_jet2_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("jet2_creation");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let val = Field::sequential(1.0);

    group.bench_function("x_seeded", |bencher| {
        bencher.iter(|| black_box(Jet2::x(black_box(val))))
    });

    group.bench_function("y_seeded", |bencher| {
        bencher.iter(|| black_box(Jet2::y(black_box(val))))
    });

    group.bench_function("constant", |bencher| {
        bencher.iter(|| black_box(Jet2::constant(black_box(val))))
    });

    group.finish();
}

fn bench_jet2_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("jet2_arithmetic");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Jet2::x(Field::sequential(1.0));
    let y = Jet2::y(Field::from(2.0));

    group.bench_function("add", |bencher| {
        bencher.iter(|| black_box(black_box(x) + black_box(y)))
    });

    group.bench_function("sub", |bencher| {
        bencher.iter(|| black_box(black_box(x) - black_box(y)))
    });

    group.bench_function("mul", |bencher| {
        bencher.iter(|| black_box(black_box(x) * black_box(y)))
    });

    group.bench_function("div", |bencher| {
        bencher.iter(|| black_box(black_box(x) / black_box(y)))
    });

    group.finish();
}

fn bench_jet2_math(c: &mut Criterion) {
    let mut group = c.benchmark_group("jet2_math");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Jet2::x(Field::sequential(1.0));
    let y = Jet2::y(Field::from(2.0));

    group.bench_function("sqrt", |bencher| {
        bencher.iter(|| black_box(black_box(x).sqrt()))
    });

    group.bench_function("abs", |bencher| {
        let neg = Jet2::x(Field::from(-3.5));
        bencher.iter(|| black_box(black_box(neg).abs()))
    });

    group.bench_function("min", |bencher| {
        bencher.iter(|| black_box(black_box(x).min(black_box(y))))
    });

    group.bench_function("max", |bencher| {
        bencher.iter(|| black_box(black_box(x).max(black_box(y))))
    });

    group.finish();
}

fn bench_jet2_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("jet2_gradient");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    group.bench_function("circle_sdf_gradient", |bencher| {
        // Gradient of √(x² + y²) - r
        // ∂f/∂x = x / √(x² + y²)
        // ∂f/∂y = y / √(x² + y²)
        bencher.iter(|| {
            let x = Jet2::x(Field::sequential(3.0));
            let y = Jet2::y(Field::from(4.0));
            let dist = (x * x + y * y).sqrt();
            black_box(dist)
        })
    });

    group.bench_function("polynomial_gradient", |bencher| {
        // Gradient of x³ + xy²
        // ∂f/∂x = 3x² + y²
        // ∂f/∂y = 2xy
        bencher.iter(|| {
            let x = Jet2::x(Field::sequential(2.0));
            let y = Jet2::y(Field::from(3.0));
            let result = x * x * x + x * y * y;
            black_box(result)
        })
    });

    group.finish();
}

// ============================================================================
// Fix Combinator Benchmarks
// ============================================================================

fn bench_fix_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("fix_iteration");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Field::sequential(0.0);
    let y = Field::from(0.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    group.bench_function("converge_fast_all_lanes", |bencher| {
        // All lanes converge immediately: w >= 10 (seed = 100)
        use pixelflow_core::W;
        let fix = Fix {
            seed: 100.0f32,      // All lanes start at 100
            step: W + 1.0f32,    // Increment (never used if done)
            done: W.ge(10.0f32), // All lanes done immediately
        };
        bencher.iter(|| black_box(fix.eval_raw(black_box(x), y, z, w)))
    });

    group.bench_function("converge_10_iterations", |bencher| {
        use pixelflow_core::W;
        // Each iteration: w += 1, done when w >= 10
        let fix = Fix {
            seed: 0.0f32,
            step: W + 1.0f32,
            done: W.ge(10.0f32),
        };
        bencher.iter(|| black_box(fix.eval_raw(black_box(x), y, z, w)))
    });

    group.bench_function("converge_variable_lanes", |bencher| {
        use pixelflow_core::W;
        // Different lanes converge at different times based on x
        // seed = x, done when w >= 5
        let fix = Fix {
            seed: X, // Lanes start at [0, 1, 2, 3, ...]
            step: W + 1.0f32,
            done: W.ge(5.0f32), // Each lane needs 5-x iterations
        };
        bencher.iter(|| black_box(fix.eval_raw(black_box(x), y, z, w)))
    });

    group.finish();
}

// ============================================================================
// Evaluation Throughput Benchmark
// ============================================================================

fn bench_evaluation_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluation_throughput");

    // Different sizes to measure scaling
    for &size in &[64, 256, 1024] {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(format!("circle_sdf_{}px", size), |bencher| {
            let m = (X * X + Y * Y).sqrt() - 50.0f32;
            let z = Field::from(0.0);
            let w = Field::from(0.0);

            bencher.iter(|| {
                let mut total = Field::from(0.0);
                let rows = size / PARALLELISM;
                for row in 0..rows {
                    let y = Field::from(row as f32);
                    let x = Field::sequential(0.0);
                    total = (total + m.eval_raw(x, y, z, w)).constant();
                }
                black_box(total)
            })
        });
    }

    group.finish();
}

// ============================================================================
// FastMath Guard Benchmarks
// ============================================================================

fn bench_fastmath_guard(c: &mut Criterion) {
    let mut group = c.benchmark_group("fastmath_denormals");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    // Create denormal values (very small numbers near zero)
    // These are 2^-120, well into denormal range (denormals: 2^-126 to 2^-149)
    let denormal = 1.0e-36f32; // ~2^-120
    let denormal_field = Field::from(denormal);

    let y = Field::from(5.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    // Benchmark 1: Denormal multiplication without guard (very slow)
    group.bench_function("denormal_mul_no_guard", |bencher| {
        bencher.iter(|| {
            // Multiply denormals - triggers slow path on CPU
            let result = black_box(denormal_field) * black_box(denormal_field);
            black_box(result)
        })
    });

    // Benchmark 2: Denormal multiplication with guard (fast, treated as zero)
    group.bench_function("denormal_mul_with_guard", |bencher| {
        bencher.iter(|| {
            let _guard = unsafe { FastMathGuard::new() };
            let result = black_box(denormal_field) * black_box(denormal_field);
            black_box(result)
        })
    });

    // Benchmark 3: Denormal-producing division without guard
    group.bench_function("denormal_div_no_guard", |bencher| {
        let large = Field::from(1.0e30f32);
        let tiny = Field::from(1.0e-30f32);
        bencher.iter(|| {
            // tiny / large produces denormal
            let result = black_box(tiny) / black_box(large);
            black_box(result)
        })
    });

    // Benchmark 4: Denormal-producing division with guard
    group.bench_function("denormal_div_with_guard", |bencher| {
        let large = Field::from(1.0e30f32);
        let tiny = Field::from(1.0e-30f32);
        bencher.iter(|| {
            let _guard = unsafe { FastMathGuard::new() };
            let result = black_box(tiny) / black_box(large);
            black_box(result)
        })
    });

    // Benchmark 5: Complex manifold with denormals (realistic rendering scenario)
    group.bench_function("manifold_denormal_heavy_no_guard", |bencher| {
        // Multiplication that approaches denormals
        // tiny * tiny produces denormal results
        let tiny = 1.0e-20f32;
        let m = X * tiny * Y * tiny;
        let far_x = Field::from(1.0);
        bencher.iter(|| black_box(m.eval_raw(black_box(far_x), y, z, w)))
    });

    // Benchmark 6: Same manifold with guard
    group.bench_function("manifold_denormal_heavy_with_guard", |bencher| {
        let tiny = 1.0e-20f32;
        let m = X * tiny * Y * tiny;
        let far_x = Field::from(1.0);
        bencher.iter(|| {
            let _guard = unsafe { FastMathGuard::new() };
            black_box(m.eval_raw(black_box(far_x), y, z, w))
        })
    });

    // Benchmark 7: Normal values without guard (baseline - should be same speed)
    group.bench_function("normal_mul_no_guard", |bencher| {
        let normal = Field::from(1.5f32);
        bencher.iter(|| {
            let result = black_box(normal) * black_box(normal);
            black_box(result)
        })
    });

    // Benchmark 8: Normal values with guard (should be same as baseline)
    group.bench_function("normal_mul_with_guard", |bencher| {
        let normal = Field::from(1.5f32);
        bencher.iter(|| {
            let _guard = unsafe { FastMathGuard::new() };
            let result = black_box(normal) * black_box(normal);
            black_box(result)
        })
    });

    // Benchmark 9: Iterative denormal accumulation (extreme case)
    group.bench_function("denormal_accumulation_no_guard", |bencher| {
        let tiny_step = Field::from(1.0e-40f32);
        bencher.iter(|| {
            let mut acc = Field::from(0.0);
            for _ in 0..100 {
                // Each iteration adds denormal value
                acc = (acc + black_box(tiny_step)).constant();
            }
            black_box(acc)
        })
    });

    // Benchmark 10: Same accumulation with guard
    group.bench_function("denormal_accumulation_with_guard", |bencher| {
        let tiny_step = Field::from(1.0e-40f32);
        bencher.iter(|| {
            let _guard = unsafe { FastMathGuard::new() };
            let mut acc = Field::from(0.0);
            for _ in 0..100 {
                acc = (acc + black_box(tiny_step)).constant();
            }
            black_box(acc)
        })
    });

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    field_benches,
    bench_field_creation,
    bench_field_arithmetic,
    bench_field_math,
    bench_field_transcendental,
    bench_field_comparisons,
    bench_field_select,
    bench_field_bitwise,
);

criterion_group!(
    manifold_benches,
    bench_manifold_constants,
    bench_manifold_simple,
    bench_manifold_circle,
    bench_manifold_select,
    bench_manifold_complex,
);

criterion_group!(
    jet2_benches,
    bench_jet2_creation,
    bench_jet2_arithmetic,
    bench_jet2_math,
    bench_jet2_gradient,
);

criterion_group!(fix_benches, bench_fix_iteration,);

criterion_group!(throughput_benches, bench_evaluation_throughput,);

criterion_group!(fastmath_benches, bench_fastmath_guard,);

criterion_main!(
    field_benches,
    manifold_benches,
    jet2_benches,
    fix_benches,
    throughput_benches,
    fastmath_benches
);
