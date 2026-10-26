//! A fold's result that the kernel reads again long after the loop, with a wall
//! of values live in between, is the number it computes.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_codegen::emit::{CompileResult, compile};
use pixelflow_ir::fold::{Binder, Bound, Fold, Monoid};
use pixelflow_ir::{ExprArena, ExprId, LatticeShape, OpKind};

/// The one sample of a kernel compiled at one point, at `(x, y)`.
fn eval_point(result: &CompileResult, x: f32, y: f32) -> f32 {
    let mut out = [0.0f32; 1];
    let origin = [x, y];
    let no_block: [f32; 0] = [];
    // SAFETY: `ctx[0]` is the uniform block, which these kernels declare
    // nothing for, `ctx[1]` is the origin block, and `out` holds the one
    // sample a single-point lattice writes.
    let ctx: [*const f32; 2] = [no_block.as_ptr(), origin.as_ptr()];
    unsafe {
        result.code.call(ctx.as_ptr(), out.as_mut_ptr(), 1);
    }
    out[0]
}

/// A fold's result that is read again long after the loop is given up for a
/// wall of values in between, and still read at its value.
///
/// `s = Σ_{i<T} (x + i)` starts a chain of `WALL` terms, all live at once
/// because they are summed in both orders, and `s` is added to the total.
#[test]
fn a_fold_result_given_up_after_its_loop_is_read_at_its_value() {
    /// More values live at once than any tier's vector file holds.
    const WALL: u64 = 48;
    const TRIPS: u64 = 7;

    let mut a = ExprArena::new();
    let (x, y) = (a.push_var(0), a.push_var(1));
    let binder = Binder::from_slot(0).expect("slot 0 exists");
    let i = a.push_var(binder.var());
    let along = a.push_binary(OpKind::Add, x, i);
    let sum = a.push_reduce(
        Fold::new(Monoid::SUM, binder),
        Bound::Lit(0),
        Bound::Lit(TRIPS),
        along,
    );

    let mut term = sum;
    let mut terms = Vec::new();
    for _ in 0..WALL {
        term = a.push_ternary(OpKind::MulAdd, term, x, y);
        terms.push(term);
    }
    let total = |a: &mut ExprArena, terms: &[ExprId]| {
        terms[1..]
            .iter()
            .fold(terms[0], |acc, &t| a.push_binary(OpKind::Add, acc, t))
    };
    let up = total(&mut a, &terms);
    terms.reverse();
    let down = total(&mut a, &terms);
    let wall = a.push_binary(OpKind::Add, up, down);
    let root = a.push_binary(OpKind::Add, wall, sum);
    let result = compile(&a, root, LatticeShape::POINT).expect("the kernel compiles");

    let (px, py) = (0.25f32, 0.5f32);
    let fold: f32 = (0..TRIPS).fold(0.0, |acc, i| acc + (px + i as f32));
    let mut term = fold;
    let mut values = Vec::new();
    for _ in 0..WALL {
        term = term.mul_add(px, py);
        values.push(term);
    }
    let total = |values: &[f32]| values[1..].iter().fold(values[0], |acc, &t| acc + t);
    let up = total(&values);
    values.reverse();
    let want = up + total(&values) + fold;
    let got = eval_point(&result, px, py);
    assert!(
        (got - want).abs() <= want.abs() * 1e-5,
        "{got} against {want}"
    );
}
