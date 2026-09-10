//! `SaturationStop` is read off the loop that stops, never inferred from the
//! counts. Pinned here because the class-cap case is invisible to the counts:
//! `apply_rule` truncates its own scan to keep `classes.len()` at or under
//! the cap, so a capped run and a quiesced run can end with the same
//! `iterations` and `classes_after` — and the same zero-union final sweep.

use std::time::Duration;

use pixelflow_ir::OpKind;
use pixelflow_ir::Rooted;
use pixelflow_ir::expr::{Environment, ExprBuilder, ExprData, Term};

/// A graph plus the declarations its leaves index.
type Graph = (Rooted<ExprData>, Environment);

fn term(g: &Graph) -> Term<'_> {
    Term::new(g.0.entry(), &g.1)
}
use pixelflow_search::egraph::{
    EClassId, EGraph, ENode, Rewrite, RewriteAction, SaturationStop, all_rules,
    saturate_with_full_budget,
};

const GENEROUS: Duration = Duration::from_secs(60);

/// `(x + y)² · (x − y) + (x · y + y · x) · (x + y)` — enough shared algebra
/// (commutativity, distribution, FMA shapes) that saturation wants far more
/// e-classes than the handful it starts with.
fn busy_expression() -> Graph {
    let mut a = ExprBuilder::new();
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
    a.finish(&[root])
}

fn egraph_of(g: &Graph) -> EGraph {
    let mut eg = EGraph::with_rules(all_rules());
    pixelflow_search::egraph::insert_term(
        term(g),
        &mut eg,
        pixelflow_search::egraph::Vocabulary::Templates,
    )
    .expect("insert into e-graph");
    eg
}

#[test]
fn class_cap_reports_class_cap_not_quiesced() {
    let g = busy_expression();
    let mut eg = egraph_of(&g);
    let cap = eg.num_classes() + 2;
    let result = saturate_with_full_budget(&mut eg, 100, cap, GENEROUS);
    assert_eq!(result.stop, SaturationStop::ClassCap, "{result:?}");
    assert!(
        result.classes_after <= cap,
        "budget scan must hold the cap: {result:?}"
    );
}

/// `x + y`: commutativity has one thing to say and then nothing applies.
fn small_expression() -> Graph {
    let mut a = ExprBuilder::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let root = a.push_binary(OpKind::Add, x, y);
    a.finish(&[root])
}

#[test]
fn full_sweep_with_zero_unions_is_quiesced() {
    let g = small_expression();
    let mut eg = egraph_of(&g);
    let result = saturate_with_full_budget(&mut eg, 100, 10_000, GENEROUS);
    assert_eq!(result.stop, SaturationStop::Quiesced, "{result:?}");
    assert!(result.iterations < 100, "{result:?}");
}

/// The busy expression under production's largest class budget: it is the
/// cap, not quiescence, that ends the run — the case the pre-`truncated`
/// loop reported as converged.
#[test]
fn busy_expression_under_production_cap_is_class_capped() {
    let g = busy_expression();
    let mut eg = egraph_of(&g);
    let result = saturate_with_full_budget(&mut eg, 100, 10_000, GENEROUS);
    assert_eq!(result.stop, SaturationStop::ClassCap, "{result:?}");
}

#[test]
fn zero_iterations_is_iteration_ceiling() {
    let g = busy_expression();
    let mut eg = egraph_of(&g);
    let result = saturate_with_full_budget(&mut eg, 0, 10_000, GENEROUS);
    assert_eq!(result.stop, SaturationStop::IterationCeiling, "{result:?}");
    assert_eq!(result.iterations, 0);
}

#[test]
fn expired_deadline_is_timeout() {
    let g = busy_expression();
    let mut eg = egraph_of(&g);
    let result = saturate_with_full_budget(&mut eg, 100, 10_000, Duration::ZERO);
    assert_eq!(result.stop, SaturationStop::Timeout, "{result:?}");
}

/// A truncated sweep is that budget's stop reason even when it also committed
/// unions — the two are independent facts. The reviewer's reproduction: the
/// busy expression with `max_iters = 1` and a cap of `initial_classes + 2`
/// truncates on the class budget *after* producing unions, and the pre-fix
/// loop only consulted `truncated` on the `unions == 0` path, so the run fell
/// through to the loop's default `IterationCeiling`.
#[test]
fn productive_but_class_capped_final_sweep_is_class_cap() {
    let g = busy_expression();
    let mut eg = egraph_of(&g);
    let cap = eg.num_classes() + 2;
    let result = saturate_with_full_budget(&mut eg, 1, cap, GENEROUS);
    assert!(
        result.total_unions > 0,
        "repro needs a sweep that both truncated and committed unions: {result:?}"
    );
    assert_eq!(result.stop, SaturationStop::ClassCap, "{result:?}");
    assert!(!result.saturated, "{result:?}");
}

/// A rule whose `apply` is slow: the deadline elapses *inside* it, reaching
/// none of the checks in the e-class walk.
struct SleepyRule;

impl Rewrite for SleepyRule {
    fn name(&self) -> &str {
        "sleepy"
    }

    fn apply(&self, _egraph: &EGraph, _id: EClassId, _node: &ENode) -> Option<RewriteAction> {
        std::thread::sleep(Duration::from_millis(20));
        None
    }
}

/// The wall-clock ceiling is hard, so a sweep that blew it cannot be reported
/// as a fixed point. The reviewer's reproduction: one e-class, one rewrite
/// whose `apply` sleeps 20 ms, a 1 ms timeout — the pre-fix scan polled the
/// deadline only at e-class boundaries, so it never fired and the run came
/// back `Quiesced` / `saturated`.
#[test]
fn deadline_elapsing_inside_a_rule_apply_is_timeout() {
    let mut a = ExprBuilder::new();
    let root = a.push_var(0);
    let mut eg = EGraph::with_rules(vec![Box::new(SleepyRule)]);
    pixelflow_search::egraph::insert(
        &a,
        root,
        &mut eg,
        pixelflow_search::egraph::Vocabulary::Templates,
    )
    .expect("insert into e-graph");
    assert_eq!(eg.num_classes(), 1, "repro wants a single e-class");

    let result = saturate_with_full_budget(&mut eg, 100, 10_000, Duration::from_millis(1));
    assert_eq!(result.stop, SaturationStop::Timeout, "{result:?}");
    assert!(!result.saturated, "{result:?}");
}
