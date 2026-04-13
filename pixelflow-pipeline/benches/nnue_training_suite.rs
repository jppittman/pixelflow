//! NNUE Training Data Collection Suite
//!
//! This benchmark generates training data for NNUE by running diverse kernels
//! and collecting (Expr features, actual SIMD runtime) pairs.
//!
//! Run with: cargo bench -p pixelflow-pipeline --bench nnue_training_suite
//!
//! The results can be processed to create NNUE training data.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use pixelflow_compiler::kernel_raw;
use pixelflow_core::{Field, Manifold, ManifoldExt, W, X, Y, Z};

// ============================================================================
// Benchmark Categories
// ============================================================================

fn bench_basic_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_arithmetic");
    group.sample_size(200);

    let xf = Field::sequential(1.0);
    let yf = Field::from(2.0);
    let zf = Field::from(3.0);
    let wf = Field::from(0.5);

    // Simple operations
    {
        let m = X + Y;
        group.bench_function("add_xy", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    {
        let m = X * Y;
        group.bench_function("mul_xy", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    {
        let m = X - Y;
        group.bench_function("sub_xy", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    {
        let m = X / Y;
        group.bench_function("div_xy", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }

    // Two operations
    {
        let m = (X + Y) * Z;
        group.bench_function("add_mul", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }
    {
        let m = X * Y + Z;
        group.bench_function("mul_add", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }

    // Three operations - linear chain
    {
        let m = X + Y + Z;
        group.bench_function("chain3_add", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }
    {
        let m = X * Y * Z;
        group.bench_function("chain3_mul", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }

    // Four operations
    {
        let m = X + Y + Z + W;
        group.bench_function("chain4_add", |b| {
            b.iter(|| {
                black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf))))
            })
        });
    }
    {
        let m = X * Y * Z * W;
        group.bench_function("chain4_mul", |b| {
            b.iter(|| {
                black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf))))
            })
        });
    }

    // Wide trees (parallel friendly)
    {
        let m = (X + Y) + (Z + W);
        group.bench_function("wide2_add", |b| {
            b.iter(|| {
                black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf))))
            })
        });
    }
    {
        let m = X * Y + Z * W;
        group.bench_function("wide2_mul", |b| {
            b.iter(|| {
                black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf))))
            })
        });
    }
    {
        let m = (X + Y) * (Z + W);
        group.bench_function("wide2_mix", |b| {
            b.iter(|| {
                black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf))))
            })
        });
    }

    group.finish();
}

fn bench_expensive_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("expensive_ops");
    group.sample_size(200);

    let xf = Field::sequential(1.0);
    let yf = Field::from(2.0);
    let zf = Field::from(3.0);
    let wf = Field::from(0.5);

    // Single expensive ops
    {
        let m = X.sqrt();
        group.bench_function("sqrt_x", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }
    {
        let m = (X + Y).sqrt();
        group.bench_function("sqrt_xy", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }

    // Multiple sqrts - parallel
    {
        let m = X.sqrt() + Y.sqrt();
        group.bench_function("sqrt2_wide", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    {
        let m = X.sqrt() + Y.sqrt() + Z.sqrt();
        group.bench_function("sqrt3_wide", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }
    {
        let m = X.sqrt() + Y.sqrt() + Z.sqrt() + W.sqrt();
        group.bench_function("sqrt4_wide", |b| {
            b.iter(|| {
                black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf))))
            })
        });
    }

    // Multiple sqrts - sequential
    {
        let m = (X.sqrt() + 1.0f32).sqrt();
        group.bench_function("sqrt2_deep", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }
    {
        let m = ((X.sqrt() + 1.0f32).sqrt() + 1.0f32).sqrt();
        group.bench_function("sqrt3_deep", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }

    // Multiple divs - parallel
    {
        let m = X / Y + Z / W;
        group.bench_function("div2_wide", |b| {
            b.iter(|| {
                black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf))))
            })
        });
    }

    // Multiple divs - sequential
    {
        let m = X / Y / Z;
        group.bench_function("div2_deep", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }
    {
        let m = X / Y / Z / W;
        group.bench_function("div3_deep", |b| {
            b.iter(|| {
                black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf))))
            })
        });
    }

    // Mixed expensive
    {
        let m = X.sqrt() + Y / Z;
        group.bench_function("sqrt_div_wide", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }
    {
        let m = X.sqrt() / Y.sqrt();
        group.bench_function("sqrt_div_deep", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }

    group.finish();
}

fn bench_distance_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_functions");
    group.sample_size(200);

    let xf = Field::sequential(1.0);
    let yf = Field::from(2.0);
    let zf = Field::from(3.0);
    let wf = Field::from(0.5);

    // 2D distance
    {
        let m = (X * X + Y * Y).sqrt();
        group.bench_function("dist2d", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }

    // 3D distance
    {
        let m = (X * X + Y * Y + Z * Z).sqrt();
        group.bench_function("dist3d", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }

    // 4D distance
    {
        let m = (X * X + Y * Y + Z * Z + W * W).sqrt();
        group.bench_function("dist4d", |b| {
            b.iter(|| {
                black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf))))
            })
        });
    }

    // Distance squared (no sqrt)
    {
        let m = X * X + Y * Y;
        group.bench_function("dist2d_sq", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    {
        let m = X * X + Y * Y + Z * Z;
        group.bench_function("dist3d_sq", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }

    // Circle SDF: sqrt(x² + y²) - r
    {
        let r = 1.0f32;
        let m = (X * X + Y * Y).sqrt() - r;
        group.bench_function("circle_sdf", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }

    // Sphere SDF
    {
        let r = 1.0f32;
        let m = (X * X + Y * Y + Z * Z).sqrt() - r;
        group.bench_function("sphere_sdf", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }

    // Box SDF (approximate)
    {
        let m = X.abs().max(Y.abs());
        group.bench_function("box2d_sdf", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }

    // Normalize
    {
        let m = X / (X * X + Y * Y).sqrt();
        group.bench_function("normalize_x", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }

    group.finish();
}

fn bench_polynomials(c: &mut Criterion) {
    let mut group = c.benchmark_group("polynomials");
    group.sample_size(200);

    let xf = Field::sequential(1.0);
    let yf = Field::from(2.0);
    let zf = Field::from(3.0);
    let wf = Field::from(0.5);

    // Linear: ax + b
    {
        let m = X * 2.0f32 + 1.0f32;
        group.bench_function("linear", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }

    // Quadratic: ax² + bx + c
    {
        let m = X * X * 2.0f32 + X * 3.0f32 + 1.0f32;
        group.bench_function("quadratic", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }

    // Cubic: x³
    {
        let m = X * X * X;
        group.bench_function("cubic", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }

    // Quartic: x⁴
    {
        let m = X * X * X * X;
        group.bench_function("quartic", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }

    // 2-var quadratic: x² + y²
    {
        let m = X * X + Y * Y;
        group.bench_function("quad2v", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }

    // 2-var cubic: x³ + y³
    {
        let m = X * X * X + Y * Y * Y;
        group.bench_function("cubic2v", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }

    // Cross terms
    {
        let m = X * Y;
        group.bench_function("cross_xy", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    {
        let m = X * Y + Y * Z;
        group.bench_function("cross_xyz", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }

    // Full quadratic 2D: ax² + bxy + cy²
    {
        let m = X * X * 2.0f32 + X * Y * 3.0f32 + Y * Y * 4.0f32;
        group.bench_function("full_quad2d", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }

    group.finish();
}

fn bench_depth_vs_width(c: &mut Criterion) {
    let mut group = c.benchmark_group("depth_vs_width");
    group.sample_size(200);

    let xf = Field::sequential(1.0);
    let yf = Field::from(2.0);
    let zf = Field::from(3.0);
    let wf = Field::from(0.5);

    // Depth 2, width 4
    {
        let m = (X + Y) + (Z + W);
        group.bench_function("d2w4", |b| {
            b.iter(|| {
                black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf))))
            })
        });
    }

    // Depth 3, width 2
    {
        let m = ((X + Y) + Z) + W;
        group.bench_function("d3w2_left", |b| {
            b.iter(|| {
                black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf))))
            })
        });
    }
    {
        let m = X + (Y + (Z + W));
        group.bench_function("d3w2_right", |b| {
            b.iter(|| {
                black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf))))
            })
        });
    }

    // Depth 4, width 1 (fully sequential)
    {
        let m = (((X + 1.0f32) + 2.0f32) + 3.0f32) + 4.0f32;
        group.bench_function("d4w1", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }

    // Wide with expensive ops
    {
        let m = (X.sqrt() + Y.sqrt()) + (Z.sqrt() + W.sqrt());
        group.bench_function("wide_sqrt4", |b| {
            b.iter(|| {
                black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf))))
            })
        });
    }

    // Deep with expensive ops
    {
        let m = ((X.sqrt() + 1.0f32).sqrt() + 1.0f32).sqrt();
        group.bench_function("deep_sqrt3", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }

    // Wide with divs
    {
        let m = X / Y + Z / W;
        group.bench_function("wide_div2", |b| {
            b.iter(|| {
                black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf))))
            })
        });
    }

    // Deep with divs
    {
        let m = ((X / Y) / Z) / W;
        group.bench_function("deep_div3", |b| {
            b.iter(|| {
                black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf))))
            })
        });
    }

    group.finish();
}

fn bench_minmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("minmax");
    group.sample_size(200);

    let xf = Field::sequential(1.0);
    let yf = Field::from(2.0);
    let zf = Field::from(3.0);
    let wf = Field::from(0.5);

    // Single min/max
    {
        let m = X.min(Y);
        group.bench_function("min_xy", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    {
        let m = X.max(Y);
        group.bench_function("max_xy", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }

    // Clamp: max(lo, min(hi, x))
    {
        let m = X.min(1.0f32).max(0.0f32);
        group.bench_function("clamp", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }

    // Abs via max: max(x, -x)
    {
        let neg_x = X * -1.0f32;
        let m = X.max(neg_x);
        group.bench_function("abs_via_max", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }

    // SDF union: min(sdf1, sdf2)
    {
        let c1 = (X * X + Y * Y).sqrt() - 1.0f32;
        let c2 = ((X - 3.0f32) * (X - 3.0f32) + Y * Y).sqrt() - 1.0f32;
        let m = c1.min(c2);
        group.bench_function("sdf_union", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }

    // SDF intersection: max(sdf1, sdf2)
    {
        let c1 = (X * X + Y * Y).sqrt() - 2.0f32;
        let box_sdf = X.abs() - 1.0f32;
        let m = c1.max(box_sdf);
        group.bench_function("sdf_intersect", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }

    group.finish();
}

fn bench_kernel_raw_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_raw");
    group.sample_size(200);

    let xf = Field::sequential(1.0);
    let yf = Field::from(2.0);
    let zf = Field::from(3.0);
    let wf = Field::from(0.5);

    // Compare kernel_raw! generated code vs manual manifold composition
    // This helps validate that the compiler generates efficient code

    // Simple add - manual
    {
        let m = X + Y;
        group.bench_function("add_manual", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    // Simple add - kernel_raw
    {
        let k = kernel_raw!(|| X + Y);
        let m = k();
        group.bench_function("add_kernel_raw", |b| {
            b.iter(|| {
                black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf))))
            })
        });
    }

    // Distance 2D - manual
    {
        let m = (X * X + Y * Y).sqrt();
        group.bench_function("dist2d_manual", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    // Distance 2D - kernel_raw
    {
        let k = kernel_raw!(|| (X * X + Y * Y).sqrt());
        let m = k();
        group.bench_function("dist2d_kernel_raw", |b| {
            b.iter(|| {
                black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf))))
            })
        });
    }

    // Complex expression - manual
    {
        let m = ((X * X + Y * Y).sqrt() - 1.0f32)
            .min(((X - 3.0f32) * (X - 3.0f32) + Y * Y).sqrt() - 1.0f32);
        group.bench_function("sdf_union_manual", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }
    // Complex expression - kernel_raw
    {
        let k = kernel_raw!(
            || ((X * X + Y * Y).sqrt() - 1.0).min(((X - 3.0) * (X - 3.0) + Y * Y).sqrt() - 1.0)
        );
        let m = k();
        group.bench_function("sdf_union_kernel_raw", |b| {
            b.iter(|| {
                black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf))))
            })
        });
    }

    // FMA candidate - manual
    {
        let m = X * Y + Z;
        group.bench_function("fma_manual", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), wf))))
        });
    }
    // FMA candidate - kernel_raw (should fuse to mul_add)
    {
        let k = kernel_raw!(|| X * Y + Z);
        let m = k();
        group.bench_function("fma_kernel_raw", |b| {
            b.iter(|| {
                black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf))))
            })
        });
    }

    group.finish();
}

fn bench_transcendental(c: &mut Criterion) {
    let mut group = c.benchmark_group("transcendental");
    group.sample_size(200);

    let xf = Field::sequential(0.5); // Keep values reasonable for trig
    let yf = Field::from(0.3);
    let zf = Field::from(0.7);
    let wf = Field::from(0.1);

    // Sin/Cos
    {
        let m = X.sin();
        group.bench_function("sin", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }
    {
        let m = X.cos();
        group.bench_function("cos", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }
    {
        let m = X.sin() + X.cos();
        group.bench_function("sin_cos", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }

    // Exp/Log
    {
        let m = X.exp();
        group.bench_function("exp", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }
    {
        let m = X.ln();
        group.bench_function("ln", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), yf, zf, wf))))
        });
    }

    // Atan2
    {
        let m = Y.atan2(X);
        group.bench_function("atan2", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }

    // Pow
    {
        let m = X.pow(Y);
        group.bench_function("pow", |b| {
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), zf, wf))))
        });
    }

    group.finish();
}

criterion_group!(
    name = nnue_training;
    config = Criterion::default().sample_size(200);
    targets =
        bench_basic_arithmetic,
        bench_expensive_ops,
        bench_distance_functions,
        bench_polynomials,
        bench_depth_vs_width,
        bench_minmax,
        bench_kernel_raw_comparison,
        bench_transcendental,
);

criterion_main!(nnue_training);
