//! JIT-vs-interpreter equivalence under register pressure.
//!
//! The fused font kernels (a whole glyph as one arena) put far more values in
//! flight than the allocator has registers, so every *spill* path in
//! `resolve_operands` becomes load-bearing: reload-into-scratch for unary and
//! binary ops, the Select mask/branch choreography, decomposed MulAdd/Clamp,
//! and the beyond-red-zone stack frame. These tests force each of those paths
//! deliberately — the arena's append-only order IS the schedule, so pushing
//! values early and consuming them late pins them live across the middle —
//! and assert the JIT agrees with the IR interpreter exactly.
//!
//! Regression context: glyph-scale kernels ('O' at 32px) were the first to hit
//! `Select` with both branches spilled; anything the interpreter and JIT
//! disagree on here is a miscompile of the kind that painted whole glyph
//! regions solid.

#![cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]

use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId};
use pixelflow_ir::backend::emit::compile_arena_dag;
use pixelflow_ir::binding::BindingTable;
use pixelflow_ir::eval_scalar;
use pixelflow_ir::jit_manifold::JitManifold;

fn jit_eval(jit: &JitManifold, x: f32, y: f32, z: f32, w: f32) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        _mm_cvtss_f32(jit.call(
            _mm_set1_ps(x),
            _mm_set1_ps(y),
            _mm_set1_ps(z),
            _mm_set1_ps(w),
        ))
    }
}

/// Compile and compare against the interpreter over a coordinate grid.
/// Returns the spill count so scenarios can assert they really exercised
/// the spill paths (a pressure test that doesn't spill proves nothing).
fn assert_jit_matches_interp(arena: &ExprArena, root: ExprId, label: &str) -> u32 {
    let result = compile_arena_dag(arena, root)
        .unwrap_or_else(|e| panic!("{label}: JIT compile failed: {e}"));
    let spills = result.spill_count;
    let jit = JitManifold::new(result.code);
    let coords = [-2.5f32, -1.0, -0.3, 0.0, 0.4, 1.0, 1.7, 3.0];
    for &x in &coords {
        for &y in &coords {
            let want = eval_scalar(arena, root, &[x, y, 0.1, 0.9], &BindingTable::empty());
            let got = jit_eval(&jit, x, y, 0.1, 0.9);
            assert!(
                (want.is_nan() && got.is_nan()) || want == got,
                "{label}: JIT {got} != interp {want} at ({x}, {y}) [spills={spills}]"
            );
        }
    }
    spills
}

/// A "wide" balanced expression tree: needs ~depth registers transiently, so a
/// few of these in flight exceed the 6-register x86 budget. Leaves cycle
/// through coordinates and small constants; ops stay NaN-free (add/sub/mul by
/// small constants).
fn tree(a: &mut ExprArena, depth: usize, salt: u32) -> ExprId {
    if depth == 0 {
        return match salt % 6 {
            0 => a.push_var(0),
            1 => a.push_var(1),
            2 => a.push_var(2),
            3 => a.push_var(3),
            4 => a.push_const(0.5 + (salt % 5) as f32 * 0.25),
            _ => {
                let x = a.push_var(0);
                let c = a.push_const(0.125 + (salt % 3) as f32 * 0.375);
                a.push_binary(OpKind::Mul, x, c)
            }
        };
    }
    let l = tree(a, depth - 1, salt.wrapping_mul(2).wrapping_add(1));
    let r = tree(a, depth - 1, salt.wrapping_mul(2).wrapping_add(2));
    let op = match salt % 3 {
        0 => OpKind::Add,
        1 => OpKind::Sub,
        _ => OpKind::Add, // adds dominate: keeps magnitudes tame
    };
    a.push_binary(op, l, r)
}

/// Left-fold `Add` over already-pushed values.
fn fold_add(a: &mut ExprArena, vals: &[ExprId]) -> ExprId {
    let (&first, rest) = vals.split_first().expect("nonempty");
    rest.iter()
        .fold(first, |acc, &v| a.push_binary(OpKind::Add, acc, v))
}

/// Select whose mask AND both branches are pushed long before the select node,
/// with a wall of filler trees pinning them live in between: the both-branches-
/// spilled case ('O' glyph shape), plus the spilled-mask case.
#[test]
fn select_operands_spilled_across_pressure() {
    let mut a = ExprArena::new();

    // Operands first (they must survive the wall).
    let if_true = tree(&mut a, 3, 11);
    let if_false = tree(&mut a, 3, 23);
    let ml = tree(&mut a, 2, 37);
    let mr = tree(&mut a, 2, 41);
    let mask = a.push_binary(OpKind::Lt, ml, mr);

    // The wall: 10 filler trees, all live until the final fold.
    let fillers: Vec<ExprId> = (0..10).map(|i| tree(&mut a, 2, 100 + i * 7)).collect();

    // The select fires only now — mask/if_true/if_false have been live across
    // the whole wall and must have been spilled.
    let sel = a.push_ternary(OpKind::Select, mask, if_true, if_false);

    let mut all = vec![sel];
    all.extend(fillers);
    let root = fold_add(&mut a, &all);

    let spills = assert_jit_matches_interp(&a, root, "select_operands_spilled");
    assert!(spills > 0, "scenario failed to create register pressure");
}

/// A sum of glyph-shaped terms: each term is `select(lt, contrib, 0)` with wide
/// operand trees — the exact shape `Kernel::sum` builds for a glyph's segment
/// contributions, including consts as if_false branches (rematerialized
/// reloads, not stack reloads).
#[test]
fn glyph_shaped_sum_of_selects() {
    let mut a = ExprArena::new();

    let mut terms = Vec::new();
    for i in 0..8u32 {
        let ml = tree(&mut a, 3, 300 + i * 13);
        let mr = tree(&mut a, 3, 301 + i * 13);
        let mask = a.push_binary(OpKind::Lt, ml, mr);
        let contrib = tree(&mut a, 3, 302 + i * 13);
        let zero = a.push_const(0.0);
        terms.push(a.push_ternary(OpKind::Select, mask, contrib, zero));
    }
    let sum = fold_add(&mut a, &terms);
    // The winding rule on top, like a real glyph.
    let abs = a.push_unary(OpKind::Abs, sum);
    let one = a.push_const(1.0);
    let root = a.push_binary(OpKind::Min, abs, one);

    let spills = assert_jit_matches_interp(&a, root, "glyph_shaped_sum");
    assert!(spills > 0, "scenario failed to create register pressure");
}

/// Nested selects under pressure: a select whose branches are themselves
/// selects whose operands crossed the wall.
#[test]
fn nested_selects_spilled() {
    let mut a = ExprArena::new();

    let t1 = tree(&mut a, 3, 51);
    let f1 = tree(&mut a, 3, 53);
    let t2 = tree(&mut a, 3, 57);
    let f2 = tree(&mut a, 3, 59);
    let m1l = tree(&mut a, 2, 61);
    let m1r = tree(&mut a, 2, 67);
    let m2l = tree(&mut a, 2, 71);
    let m2r = tree(&mut a, 2, 73);
    let outer_ml = tree(&mut a, 2, 79);
    let outer_mr = tree(&mut a, 2, 83);

    let fillers: Vec<ExprId> = (0..8).map(|i| tree(&mut a, 2, 400 + i * 11)).collect();

    let m1 = a.push_binary(OpKind::Lt, m1l, m1r);
    let m2 = a.push_binary(OpKind::Ge, m2l, m2r);
    let inner1 = a.push_ternary(OpKind::Select, m1, t1, f1);
    let inner2 = a.push_ternary(OpKind::Select, m2, t2, f2);
    let outer_m = a.push_binary(OpKind::Lt, outer_ml, outer_mr);
    let sel = a.push_ternary(OpKind::Select, outer_m, inner1, inner2);

    let mut all = vec![sel];
    all.extend(fillers);
    let root = fold_add(&mut a, &all);

    let spills = assert_jit_matches_interp(&a, root, "nested_selects");
    assert!(spills > 0, "scenario failed to create register pressure");
}

/// Decomposed ternaries (MulAdd, Clamp) with spilled operands.
///
/// Pressure note: an arena containing `Clamp` is REBUILT by
/// `lowering::expand_clamp` in DFS order, which dissolves the "operands
/// early, consumers late" liveness trick the other scenarios use. Pressure
/// here must survive any schedule order, so the operands are trees DEEPER
/// than the register budget (Sethi-Ullman number > 6 spills regardless of
/// order).
#[test]
fn muladd_and_clamp_spilled() {
    let mut a = ExprArena::new();

    let ma_a = tree(&mut a, 7, 91);
    let ma_b = tree(&mut a, 7, 93);
    let ma_c = tree(&mut a, 3, 97);
    let cl_v = tree(&mut a, 7, 101);
    let cl_lo = tree(&mut a, 2, 103);
    let cl_hi = tree(&mut a, 2, 107);

    let fillers: Vec<ExprId> = (0..4).map(|i| tree(&mut a, 2, 500 + i * 19)).collect();

    let ma = a.push_ternary(OpKind::MulAdd, ma_a, ma_b, ma_c);
    // Order lo <= hi is not guaranteed by the trees; clamp semantics still must
    // match the interpreter exactly, whatever they are.
    let cl = a.push_ternary(OpKind::Clamp, cl_v, cl_lo, cl_hi);

    let mut all = vec![ma, cl];
    all.extend(fillers);
    let root = fold_add(&mut a, &all);

    let spills = assert_jit_matches_interp(&a, root, "muladd_clamp");
    assert!(spills > 0, "scenario failed to create register pressure");
}

/// Enough simultaneously-live values to overflow the 128-byte red zone
/// (more than 8 spill slots), forcing the allocated-frame prologue path.
#[test]
fn frame_mode_beyond_red_zone() {
    let mut a = ExprArena::new();

    // 24 moderate trees, all pinned live until the single final fold.
    let vals: Vec<ExprId> = (0..24).map(|i| tree(&mut a, 2, 700 + i * 23)).collect();
    let root = fold_add(&mut a, &vals);

    let result = compile_arena_dag(&a, root).expect("frame-mode compile failed");
    assert!(
        result.spill_bytes > 128,
        "scenario stayed inside the red zone (spill_bytes={}), not testing frame mode",
        result.spill_bytes
    );
    let jit = JitManifold::new(result.code);
    for &(x, y) in &[(0.3f32, -1.2f32), (2.0, 0.7), (-0.9, 3.1)] {
        let want = eval_scalar(&a, root, &[x, y, 0.1, 0.9], &BindingTable::empty());
        let got = jit_eval(&jit, x, y, 0.1, 0.9);
        assert!(
            want == got,
            "frame_mode: JIT {got} != interp {want} at ({x}, {y})"
        );
    }
}
