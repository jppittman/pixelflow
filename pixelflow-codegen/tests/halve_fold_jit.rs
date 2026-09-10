//! `HalveFold` (run through the production fold vocabulary) must reach the
//! same value as peeling a fold one term at a time — checked end to end
//! through saturation and the JIT, for both an even and an odd trip count.
//!
//! There is no interpreter left in this tree (CLAUDE.md), so "same value" is
//! checked the way CLAUDE.md asks for: compile both decompositions to real
//! machine code and compare what they compute. The "peeling" side is not
//! `egraph::fold_rules::PeelFold` run alone — in the production rule set
//! `PeelFold` declines whenever a fold can still be halved (see its doc: two
//! rules independently unrolling the same fold would defeat the point of
//! halving), so it is no longer a rule that *can* unroll an even fold by
//! itself for a test to isolate. The reference here is instead a completely
//! independent, hand-built left-leaning chain — `((f(0)⊕f(1))⊕f(2))⊕…` — the
//! shape peeling one term at a time has always produced, built without
//! touching `Fold`, `Reduce`, or the e-graph at all.
//!
//! Every [`Monoid`](pixelflow_ir::Monoid) this crate has (`SUM`, `PRODUCT`,
//! `MIN`, `MAX`, the two mask quantifiers) is commutative, so this cannot
//! exercise the property that actually distinguishes the stride-2
//! decomposition `Fold::halve` implements from the rejected
//! "halve-and-offset" alternative (interleaving terms, which needs
//! commutativity that stride-2 does not rely on): a numeric total cannot
//! tell "terms combined out of order" apart from "terms combined in order"
//! when the combiner does not care about order either way. What *is*
//! checked, and is the real risk in the code, is that `Fold::halve` and
//! `HalveFold`'s bookkeeping visit every index exactly once and produce the
//! sum the terms actually have — a duplicated or dropped index would show up
//! as a wrong total even though the monoid is commutative.
//!
//! The applications claim (O(log n), not O(n)) is measured separately, at
//! the glyph scale that motivated it — see
//! [`halving_a_glyph_scale_fold_costs_far_fewer_than_n_applications`].

use pixelflow_codegen::JIT_VECTOR_BYTES;
use pixelflow_codegen::emit::compile;
use pixelflow_codegen::emit::executable::{Point4, TileSlice};
use pixelflow_ir::{ExprArena, ExprId, ExprNode, Kernel, OpKind};
use pixelflow_search::egraph::{
    CostModel, EGraph, SaturationConfig, Vocabulary, extract, fold_rules, insert,
};

const LANES: usize = JIT_VECTOR_BYTES / core::mem::size_of::<f32>();

/// Whether a `Reduce` survives *reachable from `root`* — not whether one
/// merely sits somewhere in `arena.nodes_raw()`. `passes::expand_reduce`
/// clones and lowers rather than pruning, so its output still *holds* the
/// pre-lowering `Reduce` the arena arrived with; nothing reaches it any
/// longer, and scanning every node instead of walking from `root` would
/// refuse a perfectly legal arena (see `ExprArena::retired_axis`'s own doc
/// for the same pitfall, already paid for once).
fn has_fold(arena: &ExprArena, root: ExprId) -> bool {
    let mut seen = vec![false; arena.nodes_raw().len()];
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if core::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        if matches!(arena.node(id), ExprNode::Reduce { .. }) {
            return true;
        }
        stack.extend(arena.children(id));
    }
    false
}

/// Compile a lattice-invariant kernel (no X/Y dependence) and read the value
/// it computes.
fn eval_point(arena: &ExprArena, root: ExprId) -> f32 {
    let jit = compile(arena, root).expect("JIT compile");
    let p = Point4::new(
        [0.0f32; LANES],
        [0.0f32; LANES],
        [0.0f32; LANES],
        [0.0f32; LANES],
    );
    let mut out = [0.0f32; LANES];
    // SAFETY: no buffers/uniforms (ctx unused by a pure arithmetic kernel),
    // `out` holds one full vector batch, and `[f32; LANES]` is exactly
    // `JIT_VECTOR_BYTES` wide.
    unsafe {
        jit.code
            .call_collapse(core::ptr::null(), TileSlice::single(out.as_mut_ptr()), p);
    }
    out[0]
}

/// `Σ_{k=0}^{n-1} k` as a kernel: `Kernel::sum_over` binds the index and
/// hands it back as the body, so — before optimization — this is exactly `n`
/// terms of the reduction's own bound variable.
fn sum_kernel(n: u32) -> Kernel {
    Kernel::sum_over(n, |i| i.clone())
}

/// `((0+1)+2)+…+(n-1)` as a plain arena — no `Fold`, no `Reduce`, no e-graph.
/// The independent reference: the left-leaning chain peeling one term at a
/// time has always produced, built by literally peeling one term at a time
/// in Rust rather than by asking anything under test to do it.
fn hand_unrolled_sum(n: u32) -> (ExprArena, ExprId) {
    assert!(n >= 1, "test kernels here are always non-empty");
    let mut arena = ExprArena::new();
    let mut acc = arena.push_const(0.0);
    for k in 1..n {
        let term = arena.push_const(k as f32);
        acc = arena.push_binary(OpKind::Add, acc, term);
    }
    (arena, acc)
}

/// Saturate `(arena, root)` with the production fold vocabulary
/// (`egraph::fold_rules::fold_rules`: `HalveFold`, `PeelFold` as its
/// odd-remainder epilogue, `EmptyFold`), run to exhaustion, extract, and —
/// exactly as the real pipeline does (`pixelflow_search::runtime`'s
/// `Saturate` is always followed by `LowerDwrt, ExpandReduce`) — legalize
/// whatever `Reduce` extraction still preferred to leave in place.
///
/// A survivor here is expected, not a bug: a fold this short can extract
/// cheaper *as* a trivial `Reduce` than as the `Op` chain expanding it would
/// cost one more node than — `Kernel::sum_over`'s body is bare `Var`, priced
/// at 0, so nothing here ever spends more to unroll the last term than to
/// leave it folded. `passes::expand_reduce` is exactly the pass that turns
/// "extraction may leave this folded" into "codegen never sees a `Reduce`",
/// which is why production always runs it last regardless of what survived.
fn unroll_by_halving(
    arena: &ExprArena,
    root: ExprId,
    iterations: usize,
) -> (ExprArena, ExprId, u64) {
    let mut eg = EGraph::with_rules(fold_rules());
    let class = insert(arena, root, &mut eg, Vocabulary::Runtime).expect("a fold inserts");
    SaturationConfig::compatibility(iterations).run(&mut eg);
    let (extracted, extracted_root, _cost) = extract(&eg, class, &CostModel::latency_prior());
    let (out, out_root) = pixelflow_ir::passes::expand_reduce_owned(&extracted, extracted_root);
    assert!(
        !has_fold(&out, out_root),
        "the legalizer must leave no Reduce behind, whatever extraction chose"
    );
    (out, out_root, eg.application_count())
}

/// The load-bearing comparison: halving `Σ_{k<n} k` to exhaustion reaches the
/// same value as `n` terms peeled one at a time and chained by hand, for `n`
/// small enough to also JIT-execute the hand-built reference directly.
///
/// Applications are *not* asserted here — at an `n` this small, saturation's
/// own per-round bookkeeping (re-matching every fold each round until it
/// quiesces) is a bigger term than the O(log n) this feature is about; see
/// [`halving_a_glyph_scale_fold_costs_far_fewer_than_n_applications`] for the
/// applications claim, measured where it actually dominates.
fn assert_halving_matches_peeling(n: u32) {
    let want: f32 = (0..n).sum::<u32>() as f32;

    let kernel = sum_kernel(n);
    let (arena, root) = kernel.parts();
    // Halving needs `O(log n)` rounds; 64 is generous headroom for every `n`
    // this test uses.
    let (halve_arena, halve_root, _halve_apps) = unroll_by_halving(arena, root, 64);
    let halve_value = eval_point(&halve_arena, halve_root);

    let (ref_arena, ref_root) = hand_unrolled_sum(n);
    let ref_value = eval_point(&ref_arena, ref_root);

    assert_eq!(
        ref_value, want,
        "the hand-unrolled reference for 0..{n} must itself sum to the closed form"
    );
    assert_eq!(
        halve_value, want,
        "halving {n} terms of 0..{n} must sum to the same closed form the reference reaches"
    );
    assert_eq!(
        ref_value, halve_value,
        "n={n}: the hand-peeled reference and halve-to-exhaustion must compute the identical value"
    );
}

#[test]
fn halving_matches_peeling_even_trip_count() {
    // A power of two: every level halves cleanly, so this is pure halving
    // with no odd remainder ever arising — `HalveFold` end to end with no
    // help from `PeelFold` beyond the final single-term epilogue.
    assert_halving_matches_peeling(64);
}

#[test]
fn halving_matches_peeling_odd_trip_count() {
    // 37 is prime: the recursion hits an odd remainder at more than one
    // level (37 -> peel -> 36 -> halve -> 18 -> halve -> 9 -> peel -> 8 ->
    // halve -> 4 -> halve -> 2 -> halve -> 1), exercising `PeelFold` as the
    // epilogue repeatedly, not just once at the top.
    assert_halving_matches_peeling(37);
}

/// The performance claim itself, at the scale that motivated it
/// (CLAUDE.md's "A 34,993-node glyph fold therefore burns ~n applications").
/// No JIT here — a 34,993-term reference chain would make this test about
/// codegen throughput, not about the application count, which is measured
/// directly and is the only thing this checks. Value correctness at this
/// shape is what the two tests above already cover, at a scale a JIT and a
/// hand-built reference can still check quickly.
#[test]
fn halving_a_glyph_scale_fold_costs_far_fewer_than_n_applications() {
    let n = 34_993;
    let kernel = sum_kernel(n);
    let (arena, root) = kernel.parts();

    let mut eg = EGraph::with_rules(fold_rules());
    let class = insert(arena, root, &mut eg, Vocabulary::Runtime).expect("a fold inserts");
    // Peeling one term at a time would need on the order of `n` rounds;
    // halving needs `O(log n)` — comfortable headroom either way, and this
    // test is about the applications actually spent, not the round cap.
    SaturationConfig::compatibility(64).run(&mut eg);
    let (extracted, extracted_root, _cost) = extract(&eg, class, &CostModel::latency_prior());
    let (out, out_root) = pixelflow_ir::passes::expand_reduce_owned(&extracted, extracted_root);

    assert!(
        !has_fold(&out, out_root),
        "the legalizer must leave no Reduce behind, whatever extraction chose"
    );
    let apps = eg.application_count();
    // Measured ~89 applications for n=34,993 (versus ~n for one-term-at-a-time
    // peeling) — this asserts three orders of magnitude below `n`, generous
    // headroom around that measurement rather than a tight pin on it.
    assert!(
        apps < 1_000,
        "n={n}: {apps} applications is not O(log n) — a 34,993-term fold should cost \
         hundreds of applications, not thousands"
    );
}
