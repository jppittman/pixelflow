//! Benchmarks for the kernel! macro expansion pipeline
//!
//! These benchmarks measure the compile-time performance of the macro,
//! not runtime performance. They help identify bottlenecks in the compiler frontend.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use pixelflow_core::{Field, Manifold, ManifoldCompat, X, Y};
use pixelflow_compiler::kernel;

// ============================================================================
// Macro Expansion Benchmarks (Compile-Time, measured via codegen)
// ============================================================================

fn bench_simple_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_construction");

    group.bench_function("zero_params", |b| {
        b.iter(|| {
            let k = kernel!(|| X + Y);
            black_box(k)
        })
    });

    group.bench_function("one_param", |b| {
        b.iter(|| {
            let k = kernel!(|a: f32| X + a);
            black_box(k)
        })
    });

    group.bench_function("two_params", |b| {
        b.iter(|| {
            let k = kernel!(|cx: f32, cy: f32| (X - cx) * (X - cx) + (Y - cy) * (Y - cy));
            black_box(k)
        })
    });

    group.bench_function("with_block", |b| {
        b.iter(|| {
            let k = kernel!(|cx: f32, cy: f32| {
                let dx = X - cx;
                let dy = Y - cy;
                dx * dx + dy * dy
            });
            black_box(k)
        })
    });

    group.bench_function("complex_expression", |b| {
        b.iter(|| {
            let k = kernel!(|cx: f32, cy: f32| {
                let dx = X - cx;
                let dy = Y - cy;
                let d2 = dx * dx + dy * dy;
                let d = d2.sqrt();
                d - 1.0
            });
            black_box(k)
        })
    });

    group.finish();
}

// ============================================================================
// Kernel Evaluation Benchmarks (Runtime)
// ============================================================================

fn bench_kernel_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_evaluation");

    let x = Field::from(3.0);
    let y = Field::from(4.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    group.bench_function("zero_params_eval", |b| {
        let k = kernel!(|| X + Y);
        b.iter(|| black_box(k().eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("one_param_eval", |b| {
        let k = kernel!(|a: f32| X + a);
        let instance = k(5.0);
        b.iter(|| black_box(instance.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("two_params_eval", |b| {
        let k = kernel!(|cx: f32, cy: f32| (X - cx) * (X - cx) + (Y - cy) * (Y - cy));
        let instance = k(1.0, 2.0);
        b.iter(|| black_box(instance.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("circle_sdf_eval", |b| {
        let k = kernel!(|cx: f32, cy: f32| {
            let dx = X - cx;
            let dy = Y - cy;
            (dx * dx + dy * dy).sqrt() - 1.0
        });
        let instance = k(1.0, 2.0);
        b.iter(|| black_box(instance.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.finish();
}

// ============================================================================
// Comparison: Macro vs Manual Construction
// ============================================================================

fn bench_macro_vs_manual(c: &mut Criterion) {
    let mut group = c.benchmark_group("macro_vs_manual");

    let x = Field::from(3.0);
    let y = Field::from(4.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    // Macro version
    group.bench_function("macro_circle", |b| {
        let k = kernel!(|cx: f32, cy: f32, r: f32| {
            let dx = X - cx;
            let dy = Y - cy;
            (dx * dx + dy * dy).sqrt() - r
        });
        let instance = k(1.0, 2.0, 1.5);
        b.iter(|| black_box(instance.eval_raw(black_box(x), black_box(y), z, w)))
    });

    // Manual version
    group.bench_function("manual_circle", |b| {
        use pixelflow_core::ManifoldExt;

        let cx = 1.0f32;
        let cy = 2.0f32;
        let r = 1.5f32;
        let dx = X - cx;
        let dy = Y - cy;
        let manual = (dx.clone() * dx + dy.clone() * dy).sqrt() - r;

        b.iter(|| black_box(manual.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.finish();
}

// ============================================================================
// Type Complexity Benchmarks
// ============================================================================

fn bench_type_depth(c: &mut Criterion) {
    let mut group = c.benchmark_group("type_depth");

    let x = Field::from(3.0);
    let y = Field::from(4.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    group.bench_function("depth_1", |b| {
        let m = X + Y;
        b.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("depth_2", |b| {
        let m = (X + Y) * (X - Y);
        b.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("depth_3", |b| {
        let m = ((X + Y) * (X - Y)) + ((X * X) - (Y * Y));
        b.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("depth_4", |bench| {
        let a = X + Y;
        let b = X - Y;
        let c = X * X;
        let d = Y * Y;
        let m = (a * b + c) - (c * d + a);
        bench.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.finish();
}

criterion_group!(
    macro_benches,
    bench_simple_kernels,
    bench_kernel_evaluation,
    bench_macro_vs_manual,
    bench_type_depth,
);

criterion_main!(macro_benches);
