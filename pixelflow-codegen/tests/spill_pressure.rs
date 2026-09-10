//! JIT-vs-interpreter equivalence under register pressure.
//!
//! The fused font kernels (a whole glyph as one graph) put far more values in
//! flight than the allocator has registers, so every *spill* path in
//! `resolve_operands` becomes load-bearing: reload-into-scratch for unary and
//! binary ops, the Select mask/branch choreography, decomposed MulAdd/Clamp,
//! and the beyond-red-zone stack frame. These tests force each of those paths
//! deliberately — the graph's append-only order IS the schedule, so pushing
//! values early and consuming them late pins them live across the middle —
//! and assert the JIT agrees with the IR interpreter exactly.
//!
//! Regression context: glyph-scale kernels ('O' at 32px) were the first to hit
//! `Select` with both branches spilled; anything the interpreter and JIT
//! disagree on here is a miscompile of the kind that painted whole glyph
//! regions solid.

#![cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]

use pixelflow_codegen::emit::compile;
use pixelflow_codegen::emit::executable::{Point4, TileSlice};
use pixelflow_codegen::{CompiledKernel, JIT_VECTOR_BYTES};
use pixelflow_ir::OpKind;
use pixelflow_ir::Term;
use pixelflow_ir::binding::BindingTable;
use pixelflow_ir::eval_scalar;
use pixelflow_ir::expr::{ExprBuilder, ExprRef};

/// Lanes in one emitted batch.
const LANES: usize = JIT_VECTOR_BYTES / core::mem::size_of::<f32>();

/// One point of a compiled kernel: a single-batch collapse call, lane 0 read
/// back.
///
/// This test owns its loop. `call_collapse` is the collapse driver's entry and
/// the only one there is; pixelflow-codegen cannot reach `Lattice::bake` (core
/// depends on this crate, not the other way round), so a test that wants one
/// number spells the batch itself rather than the crate growing a point API for
/// it.
fn eval_point(jit: &CompiledKernel, x: f32, y: f32, z: f32, w: f32) -> f32 {
    let mut out = [0.0f32; LANES];
    // SAFETY: `out` holds exactly one whole batch, and every kernel in this file
    // declares no buffers, so the null context is never read.
    unsafe {
        jit.call_collapse(
            core::ptr::null(),
            TileSlice::single(out.as_mut_ptr()),
            Point4::new([x; LANES], [y; LANES], [z; LANES], [w; LANES]),
        );
    }
    out[0]
}

/// How many registers this build's backend actually allocates.
///
/// The scenarios below are about what happens *past* the pool, so every one of
/// them has to be sized against it. Writing that size in as a literal is how
/// this file's pressure quietly evaporated once: `muladd_and_clamp_spilled`
/// said "Sethi-Ullman number > 6", which stopped being more than the pool the
/// moment the pool grew, and a scenario that no longer spills asserts nothing.
fn pool_size() -> usize {
    let mut a = ExprBuilder::new();
    let x = a.push_var(0);
    let root = a.push_binary(OpKind::Add, x, x);
    let built = a.finish(&[root]);
    compile(Term::new(built.0.entry(), &built.1))
        .expect("trivial compile")
        .max_regs as usize
}

/// Compile, assert the scenario actually spilled, and compare against the
/// interpreter over a coordinate grid.
///
/// The spill assertion lives **here**, not in each scenario. Every test in
/// this file is a claim about what happens past the pool, so one that no
/// longer reaches the pool has stopped testing its subject while still
/// reporting green — and that is not hypothetical: `muladd_and_clamp_spilled`
/// spent an unknown stretch asserting nothing after the pool grew past the
/// literal 6 written into it. A scenario cannot opt out of the check by
/// forgetting to write it, which is the difference between a convention and
/// an invariant.
///
/// It returns nothing on purpose. Handing back a count that callers were
/// trusted to test was the shape that let the omission happen.
fn assert_spills_and_matches_interp(t: Term<'_>, label: &str) {
    let result = compile(t).unwrap_or_else(|e| panic!("{label}: JIT compile failed: {e}"));
    let spills = result.spill_count;
    assert!(
        spills > 0,
        "{label}: compiled without spilling (pool is {} registers), so this \
         scenario no longer exercises the spill paths it exists to test — \
         size its pressure from `pool_size()` rather than a literal",
        result.max_regs
    );
    let jit = CompiledKernel::new(result.code, pixelflow_ir::LatticeShape::POINT);
    let coords = [-2.5f32, -1.0, -0.3, 0.0, 0.4, 1.0, 1.7, 3.0];
    for &x in &coords {
        for &y in &coords {
            let want = eval_scalar(t, &[x, y], &BindingTable::empty());
            let got = eval_point(&jit, x, y, 0.1, 0.9);
            assert!(
                (want.is_nan() && got.is_nan()) || floats_agree(want, got),
                "{label}: JIT {got} != interp {want} at ({x}, {y}) [spills={spills}]"
            );
        }
    }
}

/// Bit-exact under every build this file was originally tested with (no
/// hardware FMA target feature => `fp-contract=fast` has nothing to contract
/// into, so both tiers round identically). Under `+avx2,+fma`, `eval_scalar`'s
/// scalar `a*b+c` gets contracted by LLVM into one rounding (a real `fma`
/// instruction); a `DecomposedMulAdd` (register pressure forced `a`/`b` apart
/// from `c` — see `muladd_and_clamp_spilled`, which deliberately builds
/// operands deeper than the register budget) is architecturally a two-step
/// mul-then-add and rounds twice, regardless of backend. That is a genuine
/// last-bit IEEE-754 rounding difference, not a miscompile, so this tolerates
/// a few ULPs of f32 relative error instead of weakening to a coarse epsilon
/// that would also hide a real one.
fn floats_agree(want: f32, got: f32) -> bool {
    if want == got {
        return true;
    }
    let ulp = f32::EPSILON * want.abs().max(f32::MIN_POSITIVE);
    (want - got).abs() <= 4.0 * ulp
}

/// A "wide" balanced expression tree: needs ~depth registers transiently, so a
/// few of these in flight exceed the 6-register x86 budget. Leaves cycle
/// through coordinates and small constants; ops stay NaN-free (add/sub/mul by
/// small constants).
fn tree(a: &mut ExprBuilder, depth: usize, salt: u32) -> ExprRef {
    if depth == 0 {
        return match salt % 6 {
            0 => a.push_var(0),
            1 => a.push_var(1),
            // A lattice has two axes, so the leaves past them are distinct
            // *values* of the two rather than distinct coordinates.
            2 => {
                let y = a.push_var(1);
                let c = a.push_const(0.75 + (salt % 4) as f32 * 0.5);
                a.push_binary(OpKind::Mul, y, c)
            }
            3 => {
                let y = a.push_var(1);
                let c = a.push_const(0.25 + (salt % 7) as f32 * 0.125);
                a.push_binary(OpKind::Add, y, c)
            }
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
fn fold_add(a: &mut ExprBuilder, vals: &[ExprRef]) -> ExprRef {
    let (&first, rest) = vals.split_first().expect("nonempty");
    rest.iter()
        .fold(first, |acc, &v| a.push_binary(OpKind::Add, acc, v))
}

/// Select whose mask AND both branches are pushed long before the select node,
/// with a wall of filler trees pinning them live in between: the both-branches-
/// spilled case ('O' glyph shape), plus the spilled-mask case.
#[test]
fn select_operands_spilled_across_pressure() {
    let mut a = ExprBuilder::new();

    // Operands first (they must survive the wall).
    let if_true = tree(&mut a, 3, 11);
    let if_false = tree(&mut a, 3, 23);
    let ml = tree(&mut a, 2, 37);
    let mr = tree(&mut a, 2, 41);
    let mask = a.push_binary(OpKind::Lt, ml, mr);

    // The wall: 10 filler trees, all live until the final fold.
    let fillers: Vec<ExprRef> = (0..10).map(|i| tree(&mut a, 2, 100 + i * 7)).collect();

    // The select fires only now — mask/if_true/if_false have been live across
    // the whole wall and must have been spilled.
    let sel = a.push_ternary(OpKind::Select, mask, if_true, if_false);

    let mut all = vec![sel];
    all.extend(fillers);
    let root = fold_add(&mut a, &all);
    let built = a.finish(&[root]);
    let t = Term::new(built.0.entry(), &built.1);

    assert_spills_and_matches_interp(t, "select_operands_spilled");
}

/// A sum of glyph-shaped terms: each term is `select(lt, contrib, 0)` with wide
/// operand trees — the exact shape `Kernel::sum` builds for a glyph's segment
/// contributions, including consts as if_false branches (rematerialized
/// reloads, not stack reloads).
#[test]
fn glyph_shaped_sum_of_selects() {
    let mut a = ExprBuilder::new();

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
    let built = a.finish(&[root]);
    let t = Term::new(built.0.entry(), &built.1);

    assert_spills_and_matches_interp(t, "glyph_shaped_sum");
}

/// Nested selects under pressure: a select whose branches are themselves
/// selects whose operands crossed the wall.
#[test]
fn nested_selects_spilled() {
    let mut a = ExprBuilder::new();

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

    let fillers: Vec<ExprRef> = (0..8).map(|i| tree(&mut a, 2, 400 + i * 11)).collect();

    let m1 = a.push_binary(OpKind::Lt, m1l, m1r);
    let m2 = a.push_binary(OpKind::Ge, m2l, m2r);
    let inner1 = a.push_ternary(OpKind::Select, m1, t1, f1);
    let inner2 = a.push_ternary(OpKind::Select, m2, t2, f2);
    let outer_m = a.push_binary(OpKind::Lt, outer_ml, outer_mr);
    let sel = a.push_ternary(OpKind::Select, outer_m, inner1, inner2);

    let mut all = vec![sel];
    all.extend(fillers);
    let root = fold_add(&mut a, &all);
    let built = a.finish(&[root]);
    let t = Term::new(built.0.entry(), &built.1);

    assert_spills_and_matches_interp(t, "nested_selects");
}

/// Decomposed `MulAdd` plus a min/max clamp chain, with spilled operands.
///
/// Pressure note: the fillers are pushed before the `MulAdd` and consumed after
/// it, so they hold the pool across it and its operands have to go to memory —
/// which is what makes the `MulAdd` decompose. There are `pool + 2` of them so
/// that stays true at whatever width this build allocates; the operand trees
/// are deep as well, but depth alone stopped being enough once the pool grew
/// past their Sethi-Ullman number.
#[test]
fn muladd_and_clamp_spilled() {
    let mut a = ExprBuilder::new();

    let ma_a = tree(&mut a, 7, 91);
    let ma_b = tree(&mut a, 7, 93);
    let ma_c = tree(&mut a, 3, 97);
    let cl_v = tree(&mut a, 7, 101);
    let cl_lo = tree(&mut a, 2, 103);
    let cl_hi = tree(&mut a, 2, 107);

    let fillers: Vec<ExprRef> = (0..pool_size() as u32 + 2)
        .map(|i| tree(&mut a, 2, 500 + i * 19))
        .collect();

    let ma = a.push_ternary(OpKind::MulAdd, ma_a, ma_b, ma_c);
    // Order lo <= hi is not guaranteed by the trees; the composition must
    // match the interpreter exactly either way.
    let cl_floored = a.push_binary(OpKind::Max, cl_v, cl_lo);
    let cl = a.push_binary(OpKind::Min, cl_floored, cl_hi);

    let mut all = vec![ma, cl];
    all.extend(fillers);
    let root = fold_add(&mut a, &all);
    let built = a.finish(&[root]);
    let t = Term::new(built.0.entry(), &built.1);

    assert_spills_and_matches_interp(t, "muladd_clamp");
}

/// Enough simultaneously-live values to overflow the 128-byte red zone
/// (more than 8 spill slots), forcing the allocated-frame prologue path.
#[test]
fn frame_mode_beyond_red_zone() {
    let mut a = ExprBuilder::new();

    // 24 moderate trees, all pinned live until the single final fold.
    let vals: Vec<ExprRef> = (0..24).map(|i| tree(&mut a, 2, 700 + i * 23)).collect();
    let root = fold_add(&mut a, &vals);
    let built = a.finish(&[root]);
    let t = Term::new(built.0.entry(), &built.1);

    let result = compile(t).expect("frame-mode compile failed");
    assert!(
        result.spill_bytes > 128,
        "scenario stayed inside the red zone (spill_bytes={}), not testing frame mode",
        result.spill_bytes
    );
    let jit = CompiledKernel::new(result.code, pixelflow_ir::LatticeShape::POINT);
    for &(x, y) in &[(0.3f32, -1.2f32), (2.0, 0.7), (-0.9, 3.1)] {
        let want = eval_scalar(t, &[x, y], &BindingTable::empty());
        let got = eval_point(&jit, x, y, 0.1, 0.9);
        assert!(
            want == got,
            "frame_mode: JIT {got} != interp {want} at ({x}, {y})"
        );
    }
}
