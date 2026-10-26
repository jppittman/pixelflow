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
//! **What the halving side hands the JIT.** Under the production latency
//! prior a loop and its unrolling cost the same, so extraction keeps the
//! first node the class held — the `Reduce` — and the chain the rules built
//! is never the term anyone compiles (docs/BACKLOG.md, E6). A test that
//! extracted under that prior and compiled the result would compile a loop
//! and say nothing about `HalveFold`. So the extraction here prices a loop
//! out ([`LoopsCostMore`]): the chain is priced for, not asserted, and the
//! value the JIT computes is the value halving to exhaustion built, or the
//! loop's, which is the same value.
//!
//! Every [`Monoid`](pixelflow_ir::Monoid) this crate has (`SUM`, `PRODUCT`,
//! `MIN`, `MAX`, the two mask quantifiers) is commutative, so this cannot
//! exercise the property that actually distinguishes the stride-2
//! decomposition `KnownRange::halve` implements from the rejected
//! "halve-and-offset" alternative (interleaving terms, which needs
//! commutativity that stride-2 does not rely on): a numeric total cannot
//! tell "terms combined out of order" apart from "terms combined in order"
//! when the combiner does not care about order either way. What *is*
//! checked, and is the real risk in the code, is that `KnownRange::halve` and
//! `HalveFold`'s bookkeeping visit every index exactly once and produce the
//! sum the terms actually have — a duplicated or dropped index would show up
//! as a wrong total even though the monoid is commutative.

use pixelflow_codegen::emit::compile;
use pixelflow_ir::{ExprArena, ExprId, Kernel, LatticeShape, OpKind};
use pixelflow_search::egraph::{
    CostFunction, CostModel, ENode, RuleSet, Saturation, SaturationConfig, Vocabulary, extract,
    fold_rules, insert,
};

/// What a loop costs [`LoopsCostMore`] beyond the latency prior: more than any
/// chain these tests build, so a class that holds both a `Reduce` and a chain
/// always extracts the chain.
const LOOP_PENALTY: usize = 1 << 24;

/// The latency prior, with every `Reduce` priced out.
struct LoopsCostMore(CostModel);

impl CostFunction for LoopsCostMore {
    fn node_cost(&self, node: &ENode, parent: Option<OpKind>) -> usize {
        let own = self.0.node_cost(node, parent);
        match node {
            ENode::Reduce { .. } => own.saturating_add(LOOP_PENALTY),
            _ => own,
        }
    }
}

/// Compile a lattice-invariant kernel (no X/Y dependence) at a single-point
/// lattice and read the value it computes.
fn eval_point(arena: &ExprArena, root: ExprId) -> f32 {
    let jit = compile(arena, root, LatticeShape::POINT).expect("JIT compile");
    let mut out = [0.0f32; 1];
    let origin = [0.0f32, 0.0f32];
    // SAFETY: this arena declares no buffers and no uniform (a pure
    // arithmetic kernel), so `ctx[0]` — the uniform-block slot — is unread;
    // `ctx[1]` is the origin block, and `out` holds the one sample a
    // single-point lattice writes.
    let ctx: [*const f32; 2] = [core::ptr::null(), origin.as_ptr()];
    unsafe {
        jit.code.call(ctx.as_ptr(), out.as_mut_ptr(), 1);
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
/// odd-remainder epilogue, `EmptyFold`), run to exhaustion, and extract the
/// cheapest term with loops priced out — the chain halving built.
fn unroll_by_halving(arena: &ExprArena, root: ExprId, iterations: usize) -> (ExprArena, ExprId) {
    let mut eg = Saturation::over(RuleSet::new(fold_rules()));
    let class = insert(arena, root, &mut eg, Vocabulary::Runtime).expect("a fold inserts");
    SaturationConfig::compatibility(iterations).run(&mut eg);
    let (extracted, extracted_root, _cost) =
        extract(&eg, class, &LoopsCostMore(CostModel::latency_prior()));
    (extracted, extracted_root)
}

/// The load-bearing comparison: halving `Σ_{k<n} k` to exhaustion reaches the
/// same value as `n` terms peeled one at a time and chained by hand, for `n`
/// small enough to also JIT-execute the hand-built reference directly.
fn assert_halving_matches_peeling(n: u32) {
    let want: f32 = (0..n).sum::<u32>() as f32;

    let kernel = sum_kernel(n);
    let (arena, root) = kernel.parts();
    // Halving needs `O(log n)` rounds; 64 is generous headroom for every `n`
    // this test uses.
    let (halve_arena, halve_root) = unroll_by_halving(arena, root, 64);
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
