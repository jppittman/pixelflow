//! Execute the collapse ABI once, the way `pixelflow-core` does.
//!
//! `collapse_overhead` drives `CompiledKernel::call` directly — its own
//! context pointer, its own plane — and **no CI job runs it**: `cargo
//! nextest` does not execute benchmark targets, and the bench declares
//! `harness = false`, so a `#[test]` inside it would not run either. Clippy
//! compiles it and nothing more.
//!
//! Two bugs shipped green through that hole in a single change: a bench
//! expression left reading the retired Z axis, and a fix for it that read a
//! uniform through the null context the bench passes, which segfaults. Both
//! are execution failures in code that type-checks perfectly.
//!
//! So this is the same ABI usage as a test: one call filling one plane. It
//! is not a measurement and takes no timings — it exists so that "the
//! benchmark still runs" is something a normal `cargo test` can answer.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_codegen::emit::compile;
use pixelflow_codegen::jit_vector_bytes;
use pixelflow_ir::arena::ExprArena;
use pixelflow_ir::{LatticeShape, OpKind};

/// Lanes in one batch at the tier this host selected.
fn lanes() -> usize {
    jit_vector_bytes() / core::mem::size_of::<f32>()
}
/// Three full batches and a remainder of every width below a batch.
fn width() -> usize {
    3 * lanes() + lanes() - 1
}
const ROWS: usize = 2;
/// Wider than the extent: the rows are not contiguous, so a store that
/// stepped by the width rather than the pitch would land in the wrong row.
fn pitch() -> usize {
    width() + 5
}
const BIAS: f32 = 1.75;
const ORIGIN: [f32; 2] = [0.5, 0.5];

/// `X * Y + BIAS` — reads both axes, so every lane and row must differ, and
/// a loop that failed to step X or Y would land on the wrong answer rather
/// than merely a slow one.
fn kernel() -> (ExprArena, pixelflow_ir::arena::ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let xy = a.push_binary(OpKind::Mul, x, y);
    let bias = a.push_const(BIAS);
    let root = a.push_binary(OpKind::Add, xy, bias);
    (a, root)
}

#[test]
fn one_call_fills_the_plane() {
    let (width, pitch) = (width(), pitch());
    let (arena, root) = kernel();
    let shape = LatticeShape::new([width as u32, ROWS as u32]);
    let code = compile(&arena, root, shape).expect("the smoke kernel must compile");

    const UNWRITTEN: f32 = -1000.0;
    let mut out = vec![UNWRITTEN; ROWS * pitch];
    // This kernel declares neither a buffer nor a uniform, so the context is
    // the origin block alone at the slot after the (empty) buffer table and
    // the (absent) uniform block.
    let origin = ORIGIN;
    let ctx: [*const f32; 2] = [core::ptr::null(), origin.as_ptr()];
    unsafe {
        code.code.call(ctx.as_ptr(), out.as_mut_ptr(), pitch);
    }

    for row in 0..ROWS {
        let y = ORIGIN[1] + row as f32;
        for col in 0..pitch {
            let got = out[row * pitch + col];
            if col < width {
                let want = (ORIGIN[0] + col as f32) * y + BIAS;
                assert!(
                    (got - want).abs() <= 1e-4,
                    "row {row} col {col}: got {got}, want {want}"
                );
            } else {
                assert_eq!(
                    got, UNWRITTEN,
                    "row {row} col {col}: past the width was written"
                );
            }
        }
    }
}
