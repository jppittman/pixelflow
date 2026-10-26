//! A fold over nothing is its monoid's identity.
//!
//! A lattice narrower than one SIMD batch has no full batch to run, and
//! still computes the same samples. A fold the kernel itself writes over
//! nothing is not compiled either, and lowers to its identity;
//! `a_fold_the_kernel_writes_over_nothing_is_its_identity` fails when
//! `arena_to_schedule`'s arm for it is reverted.
//!
//! The identity is also whether a derivative may pass through a read the fold
//! addresses: that reads the arena's variance, in which a fold over nothing is
//! `CONST` (`compute_arena_variance`).
//! `the_derivative_of_a_read_addressed_by_a_fold_over_nothing_is_zero` fails
//! when `compute_arena_variance`'s empty-fold arm
//! (`pixelflow-ir/src/variance.rs`) is reverted.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_codegen::CompileError;
use pixelflow_codegen::emit::compile;
use pixelflow_ir::arena::{BufferDecl, BufferIdentity};
use pixelflow_ir::fold::{Binder, Bound, Fold, Monoid};
use pixelflow_ir::{ExprArena, ExprId, LatticeShape, OpKind};

const ROWS: u32 = 2;
const ORIGIN: [f32; 2] = [3.0, 7.0];
/// More than one batch on every tier (4, 8 and 16 lanes would leave
/// remainders of 1, 5 and 5), so the column fold has a main range.
const WIDE: usize = 37;
const PIECES: u64 = 5;

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
    let root = a.push_reduce(
        Fold::new(Monoid::SUM, binder),
        Bound::Lit(0),
        Bound::Lit(PIECES),
        body,
    );
    (a, root)
}

/// `Σ_{i<5} (x/2 + y/4 + i/8)` in closed form. Every term is a multiple of
/// `1/8` and every partial sum is small, so `f32` sums them exactly in any
/// order.
fn exact(x: f32, y: f32) -> f32 {
    let i_sum: f32 = (0..PIECES).map(|i| i as f32 / 8.0).sum();
    PIECES as f32 * (x / 2.0 + y / 4.0) + i_sum
}

/// The samples of a `width`-wide, two-row lattice.
fn run(width: usize) -> Vec<f32> {
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
    out
}

#[test]
fn a_lattice_with_no_full_batch_computes_its_samples() {
    let (narrow, wide) = (run(1), run(WIDE));
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
    let none = |a: &mut ExprArena, monoid| {
        a.push_reduce(
            Fold::new(monoid, binder),
            Bound::Lit(0),
            Bound::Lit(0),
            body,
        )
    };
    let sum = none(&mut a, Monoid::SUM);
    let product = none(&mut a, Monoid::PRODUCT);
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

/// `buffer[address, y]`, the address built in the arena it is read from.
fn read_at(address: impl FnOnce(&mut ExprArena) -> ExprId) -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let buffer = a.declare_buffer(BufferDecl {
        id: BufferIdentity::mint(),
        width: WIDE as u32,
        height: ROWS,
    });
    let y = a.push_var(1);
    let address = address(&mut a);
    let root = a.push_gather(buffer, address, y);
    (a, root)
}

/// `∂/∂x` of a read addressed by `Π_{i<trips}(i·x)`: a table read whose
/// address names the differentiated axis through a fold.
fn dx_of_a_read_addressed_by_a_fold(trips: u64) -> (ExprArena, ExprId) {
    let (mut a, read) = read_at(|a| {
        let binder = Binder::from_slot(0).expect("slot 0 exists");
        let [x, i] = [0, binder.var()].map(|v| a.push_var(v));
        let body = a.push_binary(OpKind::Mul, i, x);
        a.push_reduce(
            Fold::new(Monoid::PRODUCT, binder),
            Bound::Lit(0),
            Bound::Lit(trips),
            body,
        )
    });
    let axis = a.push_const(0.0);
    let root = a.push_binary(OpKind::Dwrt, read, axis);
    (a, root)
}

/// A read does not vary with a coordinate; only its address does
/// (`passes::lower_dwrt`'s tabulation rule), so the derivative of a read is
/// `0` exactly where the address is constant along the axis. An address that
/// is a fold over nothing is the monoid's identity, a constant, whatever its
/// never-evaluated body reads: the derivative compiles, and is `0` at every
/// sample. Read off the body it would name `x`, and the compile would be
/// refused as a derivative through a table read.
#[test]
fn the_derivative_of_a_read_addressed_by_a_fold_over_nothing_is_zero() {
    let (a, root) = dx_of_a_read_addressed_by_a_fold(0);
    let table: Vec<f32> = (0..WIDE * ROWS as usize).map(|i| 1.0 + i as f32).collect();
    let compiled = compile(&a, root, LatticeShape::new([WIDE as u32, ROWS]))
        .expect("the derivative of a constant address compiles");
    let mut out = vec![f32::NAN; ROWS as usize * WIDE];
    let uniforms: [f32; 0] = [];
    let ctx = [table.as_ptr(), uniforms.as_ptr(), ORIGIN.as_ptr()];
    // SAFETY: the kernel declares one buffer (`table`, the context's first
    // slot) and no uniform; the origin block is `ORIGIN`, and `out` holds
    // `ROWS * WIDE` samples at pitch `WIDE`.
    unsafe {
        compiled.code.call(ctx.as_ptr(), out.as_mut_ptr(), WIDE);
    }
    for (sample, got) in out.iter().enumerate() {
        assert_eq!(*got, 0.0, "sample {sample}");
    }
}

/// The other side of the same rule: a fold over something still names its
/// body's axis, and the derivative through that address is refused.
#[test]
fn the_derivative_of_a_read_addressed_by_a_fold_over_something_is_refused() {
    let (a, root) = dx_of_a_read_addressed_by_a_fold(PIECES);
    let refused = compile(&a, root, LatticeShape::new([WIDE as u32, ROWS])).err();
    assert_eq!(
        refused,
        Some(CompileError::Legalize(
            "lower_dwrt: cannot differentiate a bound-memory read"
        ))
    );
}

#[test]
#[should_panic(expected = "degenerate extent")]
fn a_lattice_with_no_column_is_refused() {
    let (arena, root) = kernel();
    drop(compile(&arena, root, LatticeShape::new([0, ROWS])));
}
