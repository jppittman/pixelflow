//! Growth telemetry must not move saturation.
//!
//! Mirrors `optimizer_laws.rs`'s G4
//! (`observation_is_optional_and_does_not_move_the_budget`) for the same
//! reason: CLAUDE.md's budget-determinism guarantee ("the same kernel
//! compiles to the same code") has to hold whether or not a diagnostic
//! feature happens to be compiled in and switched on — an instrument is
//! optional only if attaching it cannot change what the instrumented thing
//! does. `docs/plans/2026-08-31-guide-design-revision.md` §4.1's growth
//! telemetry reads counters (`EGraph::next_enode_id`, `EGraph::classes.len()`)
//! around the same `apply_action` call every path already makes and writes
//! only into its own accumulator, so it should be inert by construction —
//! this file is that claim, checked rather than assumed.
//!
//! Whole file gated on `saturation-telemetry`: with the feature off,
//! `EGraph::enable_growth_telemetry` does not exist, and there is nothing
//! here to test — `cargo test -p pixelflow-search` (which does not request
//! the feature) simply compiles this file to an empty test binary.

#![cfg(feature = "saturation-telemetry")]

use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId};
use pixelflow_search::egraph::{Budget, EClassId, EGraph, Optimized, Optimizer};

/// A structural rendering of the extracted DAG, for equality comparisons —
/// same construction as `optimizer_laws.rs::arena_shape`: the arena is
/// append-only and extraction emits children before parents, so the node
/// vector plus the root is already canonical for a given configuration.
fn arena_shape(arena: &ExprArena, root: ExprId) -> String {
    format!("{root:?}|{:?}", arena.nodes_raw())
}

/// A mid-sized expression with sharing and several rule families in reach —
/// enough rewrite activity that growth telemetry's hook actually fires many
/// times over, not just zero or one.
fn fixture() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let one = a.push_const(1.0);
    let two = a.push_const(2.0);
    let xx = a.push_binary(OpKind::Mul, x, x);
    let yy = a.push_binary(OpKind::Mul, y, y);
    let sum = a.push_binary(OpKind::Add, xx, yy);
    let scaled = a.push_binary(OpKind::Mul, sum, two);
    // `sum` is shared: once scaled, once inside the offset.
    let offset = a.push_binary(OpKind::Add, sum, one);
    let prod = a.push_binary(OpKind::Mul, scaled, offset);
    let s = a.push_unary(OpKind::Sqrt, prod);
    let root = a.push_binary(OpKind::Sub, prod, s);
    (a, root)
}

/// Insert `root` into a fresh e-graph carrying `optimizer`'s rule set,
/// optionally switching on growth telemetry first, then run.
fn insert_and_run(
    mut optimizer: Optimizer,
    arena: &ExprArena,
    root: ExprId,
    enable_growth: bool,
) -> (EGraph, EClassId, Optimized) {
    let mut eg = optimizer.egraph();
    if enable_growth {
        eg.enable_growth_telemetry();
    }
    let root_class = pixelflow_search::egraph::insert(
        arena,
        root,
        &mut eg,
        pixelflow_search::egraph::Vocabulary::Templates,
    )
    .expect("insert into e-graph");
    let optimized = optimizer.run(&mut eg, root_class, arena.len());
    (eg, root_class, optimized)
}

/// **Enabling growth telemetry must not move the budget or the extracted
/// term.** Same law as G4 pins for `Observer`, for the same reason.
#[test]
fn growth_telemetry_does_not_move_saturation() {
    let (arena, root) = fixture();

    let (eg_off, root_off, out_off) = insert_and_run(Optimizer::production(), &arena, root, false);
    let (eg_on, root_on, out_on) = insert_and_run(Optimizer::production(), &arena, root, true);

    assert_eq!(
        out_off.stats.stop, out_on.stats.stop,
        "stop reason must be unaffected by growth telemetry"
    );
    assert_eq!(
        out_off.stats.iterations, out_on.stats.iterations,
        "rounds completed must be unaffected"
    );
    assert_eq!(
        out_off.stats.applications, out_on.stats.applications,
        "application count -- the saturation budget's own denominator -- must be unaffected"
    );
    assert_eq!(
        out_off.stats.unions, out_on.stats.unions,
        "union count must be unaffected"
    );
    assert_eq!(
        out_off.stats.classes, out_on.stats.classes,
        "e-class count must be unaffected"
    );
    assert_eq!(
        out_off.cost.tree, out_on.cost.tree,
        "extraction cost must be unaffected"
    );

    let (arena_off, root_id_off) = out_off.to_arena(&eg_off, root_off);
    let (arena_on, root_id_on) = out_on.to_arena(&eg_on, root_on);
    assert_eq!(
        arena_shape(&arena_off, root_id_off),
        arena_shape(&arena_on, root_id_on),
        "enabling growth telemetry must not change what is extracted"
    );

    // And it actually measured something, so the above isn't a vacuous pass
    // (e.g. a hook that silently never fires would pass every assertion
    // above too).
    let growth = eg_on.growth_telemetry().expect("telemetry was enabled");
    let total_applications: u64 = growth.by_rule().map(|(_, g)| g.applications()).sum::<u64>()
        + growth.unlabeled().applications();
    assert_eq!(
        total_applications, out_on.stats.applications,
        "growth telemetry must record exactly the applications the run counted"
    );
    assert!(
        out_on.stats.applications > 0,
        "the fixture must actually drive rewrite activity for this test to mean anything"
    );
}

/// The same check under a tight, explicit application budget — the boundary
/// condition where a run stops mid-scan (`ScanStop`/`SaturationStop`
/// variants other than quiescence) is exactly where a measurement hook
/// reading counters at the wrong moment would be most likely to disagree
/// with the unobserved run.
#[test]
fn growth_telemetry_does_not_move_saturation_under_a_tight_budget() {
    let (arena, root) = fixture();
    for budget in [1u64, 3, 10, 50, 500] {
        let (eg_off, root_off, out_off) = insert_and_run(
            Optimizer::production().budget(Budget::Applications(budget)),
            &arena,
            root,
            false,
        );
        let (eg_on, root_on, out_on) = insert_and_run(
            Optimizer::production().budget(Budget::Applications(budget)),
            &arena,
            root,
            true,
        );
        assert_eq!(
            out_off.stats.applications, out_on.stats.applications,
            "budget {budget}: application count must be unaffected"
        );
        assert_eq!(
            out_off.stats.stop, out_on.stats.stop,
            "budget {budget}: stop reason must be unaffected"
        );
        let (arena_off, root_id_off) = out_off.to_arena(&eg_off, root_off);
        let (arena_on, root_id_on) = out_on.to_arena(&eg_on, root_on);
        assert_eq!(
            arena_shape(&arena_off, root_id_off),
            arena_shape(&arena_on, root_id_on),
            "budget {budget}: enabling growth telemetry must not change what is extracted"
        );
    }
}
