//! The live e-class counter is an O(1) stand-in for `class_ids().count()`.
//! It is only worth having if it cannot drift, so every test here holds it
//! against the O(n) definition it replaces, at every stable point.
//!
//! "Stable point" is load-bearing: `EGraph::rebuild_budgeted` drains a
//! class's nodes with `mem::take` and only then performs congruence unions,
//! so a canonical class is legitimately empty *inside* that window. The
//! counter is checked either side of it, never within.

use std::time::Duration;

use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId};
use pixelflow_search::egraph::{
    Budget, ClassCeiling, ClassCeilings, EGraph, ENode, HARD_CLASS_LIMIT, Optimizer,
    SaturationStop, all_rules,
};

const GENEROUS: Duration = Duration::from_secs(60);

/// Enough shared algebra (commutativity, distribution, FMA shapes) that
/// saturation mints and merges thousands of classes.
fn busy_expression() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let z = a.push_var(2);
    let sum = a.push_binary(OpKind::Add, x, y);
    let diff = a.push_binary(OpKind::Sub, x, y);
    let sq = a.push_binary(OpKind::Mul, sum, sum);
    let lhs = a.push_binary(OpKind::Mul, sq, diff);
    let xy = a.push_binary(OpKind::Mul, x, y);
    let yx = a.push_binary(OpKind::Mul, y, x);
    let twice = a.push_binary(OpKind::Add, xy, yx);
    let rhs = a.push_binary(OpKind::Mul, twice, sum);
    let scaled = a.push_binary(OpKind::Mul, rhs, z);
    let root = a.push_binary(OpKind::Add, lhs, scaled);
    (a, root)
}

fn assert_no_drift(eg: &EGraph, when: &str) {
    assert_eq!(
        eg.live_class_count(),
        eg.class_ids().count(),
        "live class counter drifted from class_ids() {when}"
    );
}

#[test]
fn counter_tracks_class_ids_through_add_union_and_rebuild() {
    let mut eg = EGraph::with_rules(all_rules());
    assert_no_drift(&eg, "on an empty graph");

    // `add`: every fresh class is live; a re-`add` of the same node is not
    // a new class at all.
    let mut ids = Vec::new();
    for v in 0..8u8 {
        ids.push(eg.add(ENode::Var(v)));
        assert_no_drift(&eg, "after add");
    }
    let repeat = eg.add(ENode::Var(0));
    assert_eq!(repeat, ids[0], "add must hash-cons");
    assert_no_drift(&eg, "after a hash-consed add");

    // `union`: each merge takes exactly one class out of the live set, and
    // a redundant merge takes none.
    for pair in ids.chunks(2) {
        if let [a, b] = pair {
            let before = eg.live_class_count();
            eg.union(*a, *b);
            assert_eq!(eg.live_class_count(), before - 1, "one merge, one class");
            assert_no_drift(&eg, "after union");
            eg.union(*a, *b);
            assert_no_drift(&eg, "after a redundant union");
        }
    }

    // `rebuild`: the drain/refill window closes with the counter intact.
    eg.rebuild();
    assert_no_drift(&eg, "after rebuild");
}

#[test]
fn counter_survives_a_full_production_saturation() {
    let (arena, root) = busy_expression();
    let mut optimizer = Optimizer::production();
    let mut eg = optimizer.egraph();
    let root_class = eg.add_arena(&arena, root);
    assert_no_drift(&eg, "after add_arena");

    let out = optimizer.run(&mut eg, root_class, 64);
    assert_no_drift(&eg, "after a production saturation");
    assert!(
        eg.live_class_count() <= eg.num_classes(),
        "live can never exceed allocated"
    );
    // The measured defect, pinned: a production run leaves strictly more
    // allocated slots than live classes, so budgeting the search against
    // `num_classes()` spends it on classes that are not there.
    assert!(
        eg.num_classes() > eg.live_class_count(),
        "this corpus expression merges, so allocated must exceed live: {} vs {}",
        eg.num_classes(),
        eg.live_class_count()
    );
    let _ = out.to_arena(&eg, root_class);
}

#[test]
fn counter_survives_interleaved_rule_batches_and_partial_rebuilds() {
    let (arena, root) = busy_expression();
    let mut eg = EGraph::with_rules(all_rules());
    eg.add_arena(&arena, root);
    let ceilings = ClassCeilings::live_budget(400);
    for _ in 0..12 {
        for rule_idx in 0..eg.num_rules() {
            eg.apply_rule_at_index(rule_idx, ceilings);
            eg.rebuild_budgeted(3);
            assert_no_drift(&eg, "between partial rebuilds");
        }
        eg.rebuild();
        assert_no_drift(&eg, "after a full rebuild");
    }
}

/// The live budget is what stops the run, and the stop reason says so.
#[test]
fn live_budget_stops_the_run_and_names_itself() {
    let (arena, root) = busy_expression();
    let mut eg = EGraph::with_rules(all_rules());
    eg.add_arena(&arena, root);
    let cap = eg.live_class_count() + 2;
    let stats = eg.saturate_with_limits(100, cap, GENEROUS);
    assert_eq!(
        stats.stop,
        SaturationStop::ClassCap(ClassCeiling::Live),
        "{stats:?}"
    );
    assert!(
        eg.live_class_count() <= cap,
        "the live budget must bound the live count: {} > {cap}",
        eg.live_class_count()
    );
    assert_no_drift(&eg, "after a live-capped run");
}

/// The allocated ceiling is the memory guard, and it names itself too — set
/// it below the live budget and it is the one that fires.
#[test]
fn allocated_ceiling_stops_the_run_and_names_itself() {
    let (arena, root) = busy_expression();
    let mut optimizer = Optimizer::production().budget(Budget::Explicit {
        iterations: 100,
        classes: HARD_CLASS_LIMIT,
        allocated_classes: 40,
        applications: None,
    });
    let mut eg = optimizer.egraph();
    let root_class = eg.add_arena(&arena, root);
    let out = optimizer.run(&mut eg, root_class, 64);
    assert_eq!(
        out.stats.stop,
        SaturationStop::ClassCap(ClassCeiling::Allocated),
        "{:?}",
        out.stats
    );
    assert_no_drift(&eg, "after an allocated-capped run");
}

/// Extraction still works on a graph the live budget stopped: the counter is
/// a budget input, not a structural change, so nothing downstream moves.
#[test]
fn a_live_capped_graph_still_extracts() {
    let (arena, root) = busy_expression();
    let mut optimizer = Optimizer::production().budget(Budget::Explicit {
        iterations: 100,
        classes: 60,
        allocated_classes: HARD_CLASS_LIMIT,
        applications: None,
    });
    let mut eg = optimizer.egraph();
    let root_class = eg.add_arena(&arena, root);
    let out = optimizer.run(&mut eg, root_class, 64);
    let (extracted, _extracted_root) = out.to_arena(&eg, root_class);
    assert!(
        !extracted.nodes_raw().is_empty(),
        "a capped run must still extract a term"
    );
    assert_no_drift(&eg, "after extraction");
}
