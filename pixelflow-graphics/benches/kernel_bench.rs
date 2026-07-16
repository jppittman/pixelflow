//! Benchmark comparing manual manifold construction vs kernel! macro.
//!
//! Tests the effectiveness of the e-graph optimizer on complex algebraic expressions.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use pixelflow_core::{Field, ManifoldCompat, ManifoldExt, PARALLELISM, X, Y};
use pixelflow_compiler::kernel;

/// Complex polynomial: f(x, y) = x³ + 2x²y + 3xy² + y³
/// Manual construction (no automatic fusion anymore)
#[inline(never)]
pub fn manual_poly(x: Field, y: Field) -> Field {
    let expr = X * X * X + X * X * Y * 2.0f32 + X * Y * Y * 3.0f32 + Y * Y * Y;
    expr.eval_raw(x, y, Field::from(0.0), Field::from(0.0))
}

/// Same polynomial using kernel! (optimized by e-graph)
#[inline(never)]
pub fn kernel_poly(x: Field, y: Field) -> Field {
    let k = kernel!(|| {
        X * X * X + X * X * Y * 2.0 + X * Y * Y * 3.0 + Y * Y * Y
    });
    k().eval_raw(x, y, Field::from(0.0), Field::from(0.0))
}

fn bench_polynomial_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("polynomial_optimization");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Field::sequential(1.0);
    let y = Field::from(2.0);

    group.bench_function("manual_unfused", |b| {
        b.iter(|| black_box(manual_poly(black_box(x), black_box(y))))
    });

    group.bench_function("kernel_optimized", |b| {
        b.iter(|| black_box(kernel_poly(black_box(x), black_box(y))))
    });

    group.finish();
}

criterion_group!(benches, bench_polynomial_optimization);
criterion_main!(benches);
