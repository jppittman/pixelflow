//! A value an `If` arm and a sibling fold outside the `If` both read is still
//! computed for the fold on a batch that skips the arm.
//!
//! `v = X·Y − Y` is read by the true arm `a = (v² − v)·v` and by the fold
//! `Σ_{i<4} (v + i)`, which sits outside the `If`. The false arm is the
//! constant `3`, so on a batch whose lanes all take it the arm's block is
//! skipped. A compiler that let `v` belong to the arm would leave the fold
//! reading a `v` nothing computed on that batch. The lattice spans batches
//! where every lane takes `a`, batches where every lane takes the constant,
//! and a batch that is mixed, so the guards run both ways and the select
//! blends.
//!
//! `ownership::reads` is the edge that makes the fold a reader of `v` at the
//! scope, not only the arm; dropping it fails this test.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_codegen::emit::compile;
use pixelflow_codegen::jit_vector_bytes;
use pixelflow_ir::fold::{Binder, Bound, Fold, Monoid};
use pixelflow_ir::{ExprArena, ExprId, LatticeShape, OpKind};

const ROWS: u32 = 2;
const WIDTH: usize = 64;
const ORIGIN: [f32; 2] = [0.1, 0.3];
/// The true arm is taken where `X < THRESHOLD`: columns 0 through 20.
const THRESHOLD: f32 = 21.0;
const TRIPS: u64 = 4;
/// Each `f32` op rounds once, half an ulp of its result. The chain's error is
/// a few of those, all at the largest magnitude it reaches, so this many ulps
/// of that magnitude is the documented tolerance; the chain has no
/// singularity, so every sample is well conditioned.
const ULPS: f32 = 16.0;

/// `if X < 21 { (v² − v)·v } else { 3 } + Σ_{i<4} (v + i)`, `v = X·Y − Y`.
fn kernel() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let binder = Binder::from_slot(0).expect("slot 0 exists");
    let [x, y, i] = [0, 1, binder.var()].map(|v| a.push_var(v));
    let [threshold, three] = [THRESHOLD, 3.0].map(|v| a.push_const(v));

    let xy = a.push_binary(OpKind::Mul, x, y);
    let v = a.push_binary(OpKind::Sub, xy, y);

    let vv = a.push_binary(OpKind::Mul, v, v);
    let vv_less_v = a.push_binary(OpKind::Sub, vv, v);
    let cubic = a.push_binary(OpKind::Mul, vv_less_v, v);
    let mask = a.push_binary(OpKind::Lt, x, threshold);
    let branch = a.push_ternary(OpKind::If, mask, cubic, three);

    let term = a.push_binary(OpKind::Add, v, i);
    let fold = a.push_reduce(
        Fold::new(Monoid::SUM, binder),
        Bound::Lit(0),
        Bound::Lit(TRIPS),
        term,
    );

    let root = a.push_binary(OpKind::Add, branch, fold);
    (a, root)
}

/// The kernel at `(x, y)` in closed form, and its tolerance. The magnitude is
/// that of `v`'s operands and the cube the chain raises them to, plus one so
/// the bound stays positive where every term is near zero.
fn oracle(x: f32, y: f32) -> (f32, f32) {
    let v = x * y - y;
    let cubic = (v * v - v) * v;
    let fold = TRIPS as f32 * v + 6.0;
    let branch = if x < THRESHOLD { cubic } else { 3.0 };
    let reach = (x * y).abs() + y.abs() + 1.0;
    (branch + fold, ULPS * f32::EPSILON * reach.powi(3))
}

#[test]
fn a_fold_reads_a_value_a_skipped_arm_would_have_computed() {
    let lanes = jit_vector_bytes() / core::mem::size_of::<f32>();
    let takes_true_arm = |col: usize| (ORIGIN[0] + col as f32) < THRESHOLD;
    let batches: Vec<usize> = (0..WIDTH)
        .step_by(lanes)
        .map(|start| {
            (start..start + lanes)
                .filter(|&col| takes_true_arm(col))
                .count()
        })
        .collect();
    assert!(
        batches.contains(&lanes)
            && batches.contains(&0)
            && batches.iter().any(|&n| 0 < n && n < lanes),
        "the lattice must hold an all-arm batch, an all-constant batch and a mixed one: {batches:?}"
    );

    let (arena, root) = kernel();
    let compiled = compile(&arena, root, LatticeShape::new([WIDTH as u32, ROWS]))
        .expect("the kernel compiles");
    let mut out = vec![f32::NAN; ROWS as usize * WIDTH];
    let uniforms: [f32; 0] = [];
    let ctx = [uniforms.as_ptr(), ORIGIN.as_ptr()];
    // SAFETY: the kernel declares no buffer and no uniform, so the uniform
    // block is unread; the origin block is `ORIGIN`, and `out` holds
    // `ROWS * WIDTH` samples at pitch `WIDTH`.
    unsafe {
        compiled.code.call(ctx.as_ptr(), out.as_mut_ptr(), WIDTH);
    }

    for row in 0..ROWS as usize {
        for col in 0..WIDTH {
            let (x, y) = (ORIGIN[0] + col as f32, ORIGIN[1] + row as f32);
            let (want, tolerance) = oracle(x, y);
            let got = out[row * WIDTH + col];
            assert!(
                (got - want).abs() <= tolerance,
                "row {row}, col {col}: {got} against {want} (tolerance {tolerance})"
            );
        }
    }
}
