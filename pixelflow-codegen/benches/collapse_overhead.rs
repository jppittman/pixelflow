//! Isolate Rust-to-JIT call overhead from expression cost.
//!
//! Both cases execute the *same compiled kernel* over the same lattice and
//! differ only in call granularity: the baseline crosses the Rust↔JIT boundary
//! once per SIMD group from a Rust loop, the collapse case once for the whole
//! frame. One kernel timed two ways is what makes the delta attributable to
//! the boundary; compiling the baseline separately would fold codegen
//! differences into the same figure.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use pixelflow_codegen::JIT_VECTOR_BYTES;
use pixelflow_codegen::emit::{CompileResult, compile_dag};
use pixelflow_ir::OpKind;
use pixelflow_ir::{ExprBuilder, ExprGraph};

const LANES: usize = JIT_VECTOR_BYTES / core::mem::size_of::<f32>();
const GROUPS: usize = 240;
const ROWS: usize = 64;

fn graph() -> ExprGraph {
    let mut builder = ExprBuilder::new();
    let x = builder.var(0);
    let y = builder.var(1);
    let scale = builder.constant(0.013);
    let bias = builder.constant(1.75);
    let xs = builder.binary(OpKind::Mul, x, scale);
    let ys = builder.binary(OpKind::Mul, y, scale);
    let xy = builder.binary(OpKind::Mul, xs, ys);
    // Was `Z * Z`, on an axis a lattice no longer has — and the ABI passed
    // zero in that lane, so it was a multiply whose result never mattered.
    // Squaring `ys` keeps the node count and the op mix, which is all this
    // expression owes a call-overhead measurement: the two cases run the
    // *same* compiled kernel and differ only in call granularity.
    let ys2 = builder.binary(OpKind::Mul, ys, ys);
    let sum = builder.binary(OpKind::Add, xy, ys2);
    let root = builder.binary(OpKind::Add, sum, bias);
    builder.finish_one(root)
}

fn bench_collapse_overhead(c: &mut Criterion) {
    let graph = graph();
    let collapse =
        compile_dag(graph.root(), graph.environment()).expect("collapse compile must succeed");
    let mut out = vec![0.0f32; GROUPS * LANES * ROWS];
    let seq: Vec<f32> = (0..LANES).map(|lane| lane as f32 + 0.5).collect();

    // Warm executable pages and branch predictors before Criterion samples.
    per_group_frame(&collapse, &mut out, &seq, GROUPS, ROWS);
    collapse_frame(&collapse, &mut out, &seq, GROUPS, ROWS);

    let mut group = c.benchmark_group("jit_collapse_call_overhead");
    group.throughput(Throughput::Elements((GROUPS * LANES * ROWS) as u64));
    group.bench_function(BenchmarkId::new("rust_per_group_loop", LANES), |b| {
        b.iter(|| {
            per_group_frame(
                black_box(&collapse),
                black_box(&mut out),
                &seq,
                GROUPS,
                ROWS,
            )
        });
    });
    group.bench_function(BenchmarkId::new("one_2d_collapse_call", LANES), |b| {
        b.iter(|| {
            collapse_frame(
                black_box(&collapse),
                black_box(&mut out),
                &seq,
                GROUPS,
                ROWS,
            )
        });
    });
    group.finish();
}

/// One boundary crossing per SIMD group, driven from a Rust loop.
fn per_group_frame(
    result: &CompileResult,
    out: &mut [f32],
    _seq: &[f32],
    groups: usize,
    rows: usize,
) {
    for row in 0..rows {
        let y = row as f32 + 0.5;
        let row_out = &mut out[row * groups * LANES..][..groups * LANES];
        for g in 0..groups {
            let x0 = (g * LANES) as f32 + 0.5;
            let mut x0_vec = [0.0f32; LANES];
            for (i, lane) in x0_vec.iter_mut().enumerate() {
                *lane = x0 + i as f32;
            }
            let chunk = &mut row_out[g * LANES..][..LANES];
            unsafe {
                result.code.call_collapse(
                    core::ptr::null(),
                    pixelflow_codegen::TileSlice::single(chunk.as_mut_ptr()),
                    pixelflow_codegen::Point4::new(x0_vec, [y; LANES], [0.0; LANES], [0.0; LANES]),
                );
            }
        }
    }
}

/// One boundary crossing for the whole frame.
fn collapse_frame(
    result: &CompileResult,
    out: &mut [f32],
    _seq: &[f32],
    groups: usize,
    rows: usize,
) {
    let mut x0_vec = [0.0f32; LANES];
    for (i, lane) in x0_vec.iter_mut().enumerate() {
        *lane = 0.5 + i as f32;
    }
    unsafe {
        result.code.call_collapse(
            core::ptr::null(),
            pixelflow_codegen::TileSlice::contiguous(out.as_mut_ptr(), groups, rows),
            pixelflow_codegen::Point4::new(x0_vec, [0.5; LANES], [0.0; LANES], [0.0; LANES]),
        );
    }
}

criterion_group!(benches, bench_collapse_overhead);
criterion_main!(benches);
