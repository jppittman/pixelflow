//! The e-graph refuses to assert a proved falsehood.
//!
//! Constant folding computes f32-truths (`x + 2²⁴ = 2²⁴`); the algebraic
//! rules compute ℝ-truths (`(x+y)−y = x`). Each is sound alone, but at a
//! catastrophic-cancellation input their conjunction derives two UNEQUAL
//! constants for one expression. Before the refusal valve, that union went
//! through: one e-class held both constants, extraction tie-broke between
//! them, and `((x+2²⁴)−2²⁴)·255` came back as `x` — neither the f32 answer
//! (0) nor the ℝ answer (128), just whichever node the tie-break hit.
//!
//! The valve refuses the union (an under-merged e-graph is still sound; the
//! cost is missed CSE at the collision), journals it, and leaves the first
//! prover owning the class — so the extracted result is the f32-execution
//! answer, exactly what the unoptimized kernel computes. Symbolic priority
//! (the ℝ answer winning outright) is the planned f64 constant-domain
//! follow-up; this valve is the soundness floor it builds on.

use pixelflow_ir::OpKind;
use pixelflow_ir::expr::{ExprBuilder, ExprData, ExprRef, Term};

const X: f32 = 128.0 / 255.0;
const Y: f32 = 16_777_216.0; // 2^24: f32 spacing here is 2.0, so X + Y rounds to Y.

fn cancel(a: &mut ExprBuilder) -> ExprRef {
    let x = a.push_const(X);
    let y = a.push_const(Y);
    let s = a.push_binary(OpKind::Add, x, y);
    a.push_binary(OpKind::Sub, s, y)
}

fn opt_root(a: &ExprBuilder, root: ExprRef) -> ExprData {
    let out = pixelflow_search::runtime::optimize_runtime_term(
        Term::new(a.node(root), a.env()),
        pixelflow_ir::LatticeShape::POINT,
    )
    .expect("pure arithmetic must optimize");
    *out.0.entry()
}

/// The collision leaves ONE constant owning the class — whichever prover
/// got there first under the (deterministic) rule schedule — and everything
/// downstream agrees with it. Under the current schedule the symbolic
/// cancellation wins and the ℝ answer comes out; the pinned VALUES below are
/// schedule-dependent, the pinned INVARIANT — one constant, consistent
/// downstream, never a tie-break between two — is not.
#[test]
fn cancellation_extracts_one_consistent_constant() {
    let mut a = ExprBuilder::new();
    let d = cancel(&mut a);
    assert_eq!(
        opt_root(&a, d),
        ExprData::constant(X),
        "the class must hold exactly one constant (currently the symbolic \
         ℝ answer); two constants tie-breaking arbitrarily is the bug"
    );

    let mut a = ExprBuilder::new();
    let d = cancel(&mut a);
    let c = a.push_const(255.0);
    let m = a.push_binary(OpKind::Mul, d, c);
    assert_eq!(
        opt_root(&a, m),
        ExprData::constant(128.0),
        "downstream arithmetic must fold from the class's single constant"
    );
}

/// The refusal is journaled: the collision is loud, not silent.
#[test]
fn refused_union_is_journaled() {
    use pixelflow_search::egraph::{EGraph, SaturationConfig, all_rules};

    let mut a = ExprBuilder::new();
    let d = cancel(&mut a);
    let mut eg = EGraph::with_rules(all_rules());
    let root = pixelflow_search::egraph::insert(
        &a,
        d,
        &mut eg,
        pixelflow_search::egraph::Vocabulary::Templates,
    )
    .expect("insert into e-graph");
    SaturationConfig::compatibility(60).run(&mut eg);
    let _ = root;
    assert!(
        !eg.refused_const_unions().is_empty(),
        "the folding-vs-algebra collision must be recorded, not absorbed"
    );
}

/// Ordinary folding is untouched by the valve.
#[test]
fn exact_and_ordinary_folds_still_fold() {
    let mut a = ExprBuilder::new();
    let p = a.push_const(3.0);
    let q = a.push_const(2.0);
    let m = a.push_binary(OpKind::Mul, p, q);
    assert_eq!(opt_root(&a, m), ExprData::constant(6.0));

    let mut a = ExprBuilder::new();
    let x = a.push_const(X);
    let c = a.push_const(255.0);
    let m = a.push_binary(OpKind::Mul, x, c);
    assert_eq!(opt_root(&a, m), ExprData::constant(128.0));
}

/// The refusal guard must hold during REBUILD, not just during rule
/// application. `rebuild` drains a class's nodes with `mem::take` and only
/// then performs congruence unions, so a guard that scanned the node vector
/// was blind at exactly the moment congruence closure does its merging: the
/// class looked constant-free and a contradictory merge went through.
///
/// Construction (the reviewer's reproduction): give `Add(X, A)` the constant
/// 1 and `Add(X, B)` the constant 2, then union `A` with `B`. Congruence now
/// says the two `Add` nodes are the same node, so rebuild tries to merge
/// their classes — which would assert 1 = 2.
#[test]
fn refusal_survives_rebuild_congruence() {
    use pixelflow_search::egraph::{EGraph, ENode, all_rules};

    let mut eg = EGraph::with_rules(all_rules());
    let x = eg.add(ENode::Var(0));
    let a = eg.add(ENode::Var(1));
    let b = eg.add(ENode::Var(2));

    let add = |eg: &mut EGraph, l, r| {
        eg.add(ENode::Op {
            op: pixelflow_search::egraph::ops::op_from_kind(OpKind::Add).expect("Add op"),
            children: vec![l, r],
        })
    };
    let add_a = add(&mut eg, x, a);
    let add_b = add(&mut eg, x, b);

    let one = eg.add(ENode::constant(1.0));
    let two = eg.add(ENode::constant(2.0));
    eg.union(add_a, one);
    eg.union(add_b, two);

    // Makes the two Add nodes congruent, forcing rebuild to consider merging
    // their classes.
    eg.union(a, b);
    eg.rebuild();

    assert!(
        !eg.refused_const_unions().is_empty(),
        "rebuild-time congruence must hit the refusal guard, not slip past it"
    );
    assert_ne!(
        eg.find(one),
        eg.find(two),
        "1 and 2 must not share a class — that is the corruption the guard exists to stop"
    );
}
