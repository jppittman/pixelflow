//! A fold over nothing is its monoid's identity, and costs nothing.
//!
//! A lattice narrower than one SIMD batch has no full batch to run: the
//! column fold's main range is empty, and only the remainder fold has terms.
//! The empty fold is the identity whatever its body says, so its body is not
//! compiled: the kernel at width 1 has two scopes fewer (the main column fold
//! and the sum inside it) than the same kernel at a width that has a main
//! batch, and still computes the same samples. The scope count is what fails
//! when the change is reverted; the empty `SEQ` fold's path is pinned by
//! GOLDEN's first row.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_codegen::emit::compile;
use pixelflow_ir::fold::{Binder, Fold, Monoid};
use pixelflow_ir::{ExprArena, ExprId, LatticeShape, OpKind};

const ROWS: u32 = 2;
const ORIGIN: [f32; 2] = [3.0, 7.0];
/// More than one batch on every tier (4, 8 and 16 lanes would leave
/// remainders of 1, 5 and 5), so the column fold has a main range.
const WIDE: usize = 37;
const PIECES: u32 = 5;
/// What the main column fold is: its own loop, and the sum's loop inside it.
const EMPTY_MAIN_SCOPES: usize = 2;

/// `Σ_{i<5} (x/2 + y/4 + i/8)`: a surviving fold whose body varies with the
/// column, so the lattice's column fold is not hoisted out from under it.
fn kernel() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let binder = Binder::from_slot(0).expect("slot 0 exists");
    let [x, y, i] = [0, 1, binder.var()].map(|v| a.push_var(v));
    let [half, quarter, eighth] = [0.5, 0.25, 0.125].map(|v| a.push_const(v));
    let sx = a.push_binary(OpKind::Mul, x, half);
    let sy = a.push_binary(OpKind::Mul, y, quarter);
    let si = a.push_binary(OpKind::Mul, i, eighth);
    let column = a.push_binary(OpKind::Add, sx, sy);
    let body = a.push_binary(OpKind::Add, column, si);
    let root = a.push_reduce(Fold::new(Monoid::SUM, binder, 0..PIECES), body);
    (a, root)
}

/// The samples of a `width`-wide, two-row lattice, and the number of scopes it
/// compiled.
fn run(width: usize) -> (Vec<f32>, usize) {
    let (arena, root) = kernel();
    let shape = LatticeShape::new([width as u32, ROWS]);
    let compiled = compile(&arena, root, shape).expect("the kernel compiles");
    let mut out = vec![f32::NAN; ROWS as usize * width];
    let uniforms: [f32; 0] = [];
    let ctx = [uniforms.as_ptr(), ORIGIN.as_ptr()];
    // SAFETY: the kernel declares no buffer and no uniform, so the uniform
    // block is unread; the origin block is `ORIGIN`, and `out` holds
    // `ROWS * width` samples at pitch `width`.
    unsafe {
        compiled.code.call(ctx.as_ptr(), out.as_mut_ptr(), width);
    }
    (out, compiled.traffic.scopes.len())
}

#[test]
fn a_lattice_with_no_full_batch_compiles_no_loop_for_it() {
    let (narrow, narrow_scopes) = run(1);
    let (wide, wide_scopes) = run(WIDE);
    assert_eq!(
        narrow_scopes + EMPTY_MAIN_SCOPES,
        wide_scopes,
        "the empty main column fold, or the sum inside it, still has a scope"
    );
    for row in 0..ROWS as usize {
        assert_eq!(
            narrow[row].to_bits(),
            wide[row * WIDE].to_bits(),
            "row {row}: the first sample differs between widths"
        );
        assert!(narrow[row].is_finite());
    }
}

#[test]
#[should_panic(expected = "degenerate extent")]
fn a_lattice_with_no_column_is_refused() {
    let (arena, root) = kernel();
    drop(compile(&arena, root, LatticeShape::new([0, ROWS])));
}
