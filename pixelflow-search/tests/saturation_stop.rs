//! `SaturationStop` is read off the loop that stops, never inferred from the
//! counts. Pinned here because the class-cap case is invisible to the counts:
//! `apply_rule` truncates its own scan to keep `classes.len()` at or under
//! the cap, so a capped run and a quiesced run can end with the same
//! `iterations` and `classes_after` — and the same zero-union final sweep.

use std::time::Duration;

use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId};
use pixelflow_search::egraph::{EGraph, SaturationStop, all_rules, saturate_with_full_budget};

const GENEROUS: Duration = Duration::from_secs(60);

/// `(x + y)² · (x − y) + (x · y + y · x) · (x + y)` — enough shared algebra
/// (commutativity, distribution, FMA shapes) that saturation wants far more
/// e-classes than the handful it starts with.
fn busy_expression() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let sum = a.push_binary(OpKind::Add, x, y);
    let diff = a.push_binary(OpKind::Sub, x, y);
    let sq = a.push_binary(OpKind::Mul, sum, sum);
    let lhs = a.push_binary(OpKind::Mul, sq, diff);
    let xy = a.push_binary(OpKind::Mul, x, y);
    let yx = a.push_binary(OpKind::Mul, y, x);
    let twice = a.push_binary(OpKind::Add, xy, yx);
    let rhs = a.push_binary(OpKind::Mul, twice, sum);
    let root = a.push_binary(OpKind::Add, lhs, rhs);
    (a, root)
}

fn egraph_of(arena: &ExprArena, root: ExprId) -> EGraph {
    let mut eg = EGraph::with_rules(all_rules());
    eg.add_arena(arena, root);
    eg
}

#[test]
fn class_cap_reports_class_cap_not_quiesced() {
    let (arena, root) = busy_expression();
    let mut eg = egraph_of(&arena, root);
    let cap = eg.num_classes() + 2;
    let result = saturate_with_full_budget(&mut eg, 100, cap, GENEROUS);
    assert_eq!(result.stop, SaturationStop::ClassCap, "{result:?}");
    assert!(
        result.classes_after <= cap,
        "budget scan must hold the cap: {result:?}"
    );
}

/// `x + y`: commutativity has one thing to say and then nothing applies.
fn small_expression() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let root = a.push_binary(OpKind::Add, x, y);
    (a, root)
}

#[test]
fn full_sweep_with_zero_unions_is_quiesced() {
    let (arena, root) = small_expression();
    let mut eg = egraph_of(&arena, root);
    let result = saturate_with_full_budget(&mut eg, 100, 10_000, GENEROUS);
    assert_eq!(result.stop, SaturationStop::Quiesced, "{result:?}");
    assert!(result.iterations < 100, "{result:?}");
}

/// The busy expression under production's largest class budget: it is the
/// cap, not quiescence, that ends the run — the case the pre-`truncated`
/// loop reported as converged.
#[test]
fn busy_expression_under_production_cap_is_class_capped() {
    let (arena, root) = busy_expression();
    let mut eg = egraph_of(&arena, root);
    let result = saturate_with_full_budget(&mut eg, 100, 10_000, GENEROUS);
    assert_eq!(result.stop, SaturationStop::ClassCap, "{result:?}");
}

#[test]
fn zero_iterations_is_iteration_ceiling() {
    let (arena, root) = busy_expression();
    let mut eg = egraph_of(&arena, root);
    let result = saturate_with_full_budget(&mut eg, 0, 10_000, GENEROUS);
    assert_eq!(result.stop, SaturationStop::IterationCeiling, "{result:?}");
    assert_eq!(result.iterations, 0);
}

#[test]
fn expired_deadline_is_timeout() {
    let (arena, root) = busy_expression();
    let mut eg = egraph_of(&arena, root);
    let result = saturate_with_full_budget(&mut eg, 100, 10_000, Duration::ZERO);
    assert_eq!(result.stop, SaturationStop::Timeout, "{result:?}");
}
