//! Time one collapse call over a plane.
//!
//! The old per-batch ABI crossed the Rust↔JIT boundary once per SIMD group,
//! so this used to isolate that crossing's cost by timing one compiled
//! kernel two ways — a Rust loop calling it once per group, and one call for
//! the whole frame — and attributing the delta to the boundary. The collapse
//! ABI does not have two granularities to compare any more: the lattice's
//! row, column and lane folds are wrapped around the kernel and compiled
//! into the loop nest itself (docs/plans/2026-09-16-collapse-is-a-fold.md),
//! so a compiled kernel's `call` always fills its whole shape in exactly one
//! crossing — there is no narrower call to time it against.
//!
//! What is left, and what this measures, is that one operation: one `call`
//! filling a plane of a production-representative size.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use pixelflow_codegen::emit::compile;
use pixelflow_ir::arena::ExprArena;
use pixelflow_ir::{LatticeShape, OpKind};

/// A 256×64 plane — the height a full-frame collapse measured before, and a
/// width that divides every vector width the host might select (16, 32 or
/// 64 bytes), so this shape's story does not change with `jit_vector_bytes()`.
const WIDTH: usize = 256;
const HEIGHT: usize = 64;

fn arena() -> (ExprArena, pixelflow_ir::arena::ExprId) {
    let mut arena = ExprArena::new();
    let x = arena.push_var(0);
    let y = arena.push_var(1);
    let scale = arena.push_const(0.013);
    let bias = arena.push_const(1.75);
    let xs = arena.push_binary(OpKind::Mul, x, scale);
    let ys = arena.push_binary(OpKind::Mul, y, scale);
    let xy = arena.push_binary(OpKind::Mul, xs, ys);
    // Was `Z * Z`, on an axis a lattice no longer has — and the ABI passed
    // zero in that lane, so it was a multiply whose result never mattered.
    // Squaring `ys` keeps the node count and the op mix, which is all this
    // expression owes a call-timing measurement.
    let ys2 = arena.push_binary(OpKind::Mul, ys, ys);
    let sum = arena.push_binary(OpKind::Add, xy, ys2);
    let root = arena.push_binary(OpKind::Add, sum, bias);
    (arena, root)
}

fn bench_collapse_call(c: &mut Criterion) {
    let (arena, root) = arena();
    let shape = LatticeShape::new([WIDTH as u32, HEIGHT as u32]);
    let collapse = compile(&arena, root, shape).expect("collapse compile must succeed");
    let mut out = vec![0.0f32; WIDTH * HEIGHT];
    let origin = [0.5f32, 0.5];
    // This kernel declares no buffer and no uniform, so the context is the
    // origin block alone at the slot after the (empty) buffer table and the
    // (absent) uniform block.
    let ctx: [*const f32; 2] = [core::ptr::null(), origin.as_ptr()];

    // Warm executable pages and branch predictors before Criterion samples.
    // SAFETY: `out` holds exactly `WIDTH * HEIGHT` elements laid out at
    // pitch `WIDTH`, and `ctx` is live for the call.
    unsafe {
        collapse.code.call(ctx.as_ptr(), out.as_mut_ptr(), WIDTH);
    }

    let mut group = c.benchmark_group("jit_collapse_call");
    group.throughput(Throughput::Elements((WIDTH * HEIGHT) as u64));
    group.bench_function("one_call_fills_the_plane", |b| {
        b.iter(|| {
            // SAFETY: see the warm-up call above; nothing here changes the
            // shape or the buffers this context describes.
            unsafe {
                black_box(&collapse).code.call(
                    black_box(ctx.as_ptr()),
                    black_box(out.as_mut_ptr()),
                    WIDTH,
                );
            }
        });
    });
    group.finish();
}

criterion_group!(benches, bench_collapse_call);
criterion_main!(benches);
