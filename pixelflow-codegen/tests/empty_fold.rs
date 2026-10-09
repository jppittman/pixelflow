//! A fold over nothing is its monoid's identity, and costs nothing.
//!
//! A lattice narrower than one SIMD batch has no full batch to run: `pack`
//! builds no main column fold, and only the remainder fold has terms. The
//! kernel at width 1 has two scopes fewer (the main column fold and the sum
//! inside it) than the same kernel at a width that has a main batch, and
//! still computes the same samples. A fold the kernel itself writes over
//! nothing is not compiled either, and lowers to its identity;
//! `a_fold_the_kernel_writes_over_nothing_is_its_identity` fails when `arena_to_schedule`'s arm for it is reverted.

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
const MAIN_SCOPES: usize = 2;

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

/// `Σ_{i<5} (x/2 + y/4 + i/8)` in closed form. Every term is a multiple of
/// `1/8` and every partial sum is small, so `f32` sums them exactly in any
/// order.
fn exact(x: f32, y: f32) -> f32 {
    let i_sum: f32 = (0..PIECES).map(|i| i as f32 / 8.0).sum();
    PIECES as f32 * (x / 2.0 + y / 4.0) + i_sum
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
        narrow_scopes + MAIN_SCOPES,
        wide_scopes,
        "the narrow lattice still has a main column fold, or a sum inside one"
    );
    for (width, samples) in [(1, &narrow), (WIDE, &wide)] {
        for row in 0..ROWS as usize {
            for col in 0..width {
                let (x, y) = (ORIGIN[0] + col as f32, ORIGIN[1] + row as f32);
                assert_eq!(
                    samples[row * width + col],
                    exact(x, y),
                    "width {width}, row {row}, col {col}"
                );
            }
        }
    }
}

/// `(x + Σ_{i<0} body) · (y + Π_{i<0} body)`: two folds over nothing, whose
/// bodies are never compiled, and which read as `0` and `1`.
#[test]
fn a_fold_the_kernel_writes_over_nothing_is_its_identity() {
    let mut a = ExprArena::new();
    let binder = Binder::from_slot(0).expect("slot 0 exists");
    let [x, y, i] = [0, 1, binder.var()].map(|v| a.push_var(v));
    let body = a.push_binary(OpKind::Mul, i, x);
    let none = |monoid| Fold::new(monoid, binder, 0..0);
    let sum = a.push_reduce(none(Monoid::SUM), body);
    let product = a.push_reduce(none(Monoid::PRODUCT), body);
    let left = a.push_binary(OpKind::Add, x, sum);
    let right = a.push_binary(OpKind::Add, y, product);
    let root = a.push_binary(OpKind::Mul, left, right);

    let width = 3;
    let compiled =
        compile(&a, root, LatticeShape::new([width as u32, ROWS])).expect("the kernel compiles");
    let mut out = vec![f32::NAN; ROWS as usize * width];
    let uniforms: [f32; 0] = [];
    let ctx = [uniforms.as_ptr(), ORIGIN.as_ptr()];
    // SAFETY: as in `run`.
    unsafe {
        compiled.code.call(ctx.as_ptr(), out.as_mut_ptr(), width);
    }
    for row in 0..ROWS as usize {
        for col in 0..width {
            let (x, y) = (ORIGIN[0] + col as f32, ORIGIN[1] + row as f32);
            assert_eq!(
                out[row * width + col],
                x * (y + 1.0),
                "row {row}, col {col}"
            );
        }
    }
}

#[test]
#[should_panic(expected = "degenerate extent")]
fn a_lattice_with_no_column_is_refused() {
    let (arena, root) = kernel();
    drop(compile(&arena, root, LatticeShape::new([0, ROWS])));
}
