//! Anytime (quality-at-budget) curve sampling for e-graph saturation.
//!
//! ONE definition of the guided-saturation program's anytime metric
//! (docs/plans/2026-08-31-guide-design-revision.md §0/§5, and the Phase 3
//! pre-registration docs/plans/2026-09-01-phase3-registration.md): best
//! extraction cost under a static, deterministic cost model, plotted against
//! work performed, where work is denominated in **rule applications** — never
//! wall-clock, never sweeps. Both the scoping harness
//! (`pixelflow-search/examples/oracle_filtered_budget_curves.rs`) and the
//! pipeline baseline/experiment binaries import this module instead of
//! restating the loop, so the pre-registered baseline and any later guided
//! run cannot drift apart in what "cost at budget B" means (a copy is a
//! future divergence — this codebase has paid for that before).
//!
//! # Why applications, and why this grid
//!
//! The 2026-08-30 scoping round's budget-curve run was null because its
//! checkpoint grid (fractions of a per-tier nominal *sweep* budget) started
//! past the point where 97.8% of expressions had already quiesced or hit
//! their class cap. Applications are the right x-axis because they are what
//! the per-expression work distribution is actually measured in (median ~195,
//! heavy-tailed to hundreds of thousands, per
//! docs/results/2026-08-30-guide-headroom.md), and a geometric grid resolves
//! both the median regime and the tail with a bounded number of samples.
//!
//! # Budget semantics (binding)
//!
//! - The x-axis counts **recorded applications** exactly as provenance
//!   records them — including idempotent re-fires (91% of the total). That
//!   is deliberate: a Guide pays for scoring/scanning a candidate whether or
//!   not it commits, so the honest work axis counts them.
//! - Checkpoint targets are crossed at **rule-sweep granularity**
//!   ([`crate::egraph::EGraph::saturate_until_applications`]): the sample at
//!   target `B` is taken at the first between-sweeps point where the
//!   cumulative count is `>= B`, and [`AnytimeCheckpoint::app_actual`]
//!   records the exact count. Analysis plots against `app_actual`.
//! - The class cap is **environment, not work**: it is the production tier's
//!   memory-protection cap, fixed for the whole curve, identical across any
//!   curves being compared. Under application denomination there is no
//!   per-checkpoint class-cap scaling problem (PR #1067's finding against
//!   the old fraction grid): the x-axis measures the work actually done, so
//!   a checkpoint cannot silently reflect more work than its label claims.
//! - Wall-clock appears ONLY as a per-curve safety ceiling that fails loud
//!   ([`run_anytime_curve`] panics on [`SaturationStop::Timeout`]); it never
//!   gates which samples count.

use std::time::{Duration, Instant};

use pixelflow_ir::{ExprArena, ExprId};

use super::cost::CostModel;
use super::extract::{ExtractedDAG, extract_dag};
use super::graph::{EGraph, SaturationStop};
use super::rewrite::Rewrite;

/// Geometric application-count checkpoint grid: resolves the median-~195
/// regime (25/50/100/200/400) and the heavy tail (…/204800), with the
/// run's own end state ([`AnytimeCurve::ended_at_apps`]) as the final,
/// implicit "cap" point.
pub const APP_CHECKPOINT_GRID: &[usize] = &[
    25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800,
];

/// One sampled point on an anytime curve.
#[derive(Clone, Copy, Debug)]
pub struct AnytimeCheckpoint {
    /// The grid target this sample was taken for.
    pub app_target: usize,
    /// Cumulative recorded applications at the sample — the real x-value
    /// (may overshoot `app_target` by up to one rule sweep).
    pub app_actual: usize,
    /// Cumulative completed sweeps at the sample (diagnostic only).
    pub sweeps: usize,
    /// E-class count at the sample.
    pub classes: usize,
    /// E-node count at the sample.
    pub nodes: usize,
    /// Best extraction cost at the sample (static cost model units).
    pub cost: usize,
    /// Stop reason of the saturation call that produced this sample;
    /// [`SaturationStop::ApplicationBudget`] means the curve was still live.
    pub stop: SaturationStop,
    /// `true` when the run had already ended (quiesced / class cap /
    /// iteration ceiling) at or before an EARLIER checkpoint and this row is
    /// filled from the final state rather than newly sampled.
    pub clamped: bool,
}

/// A full anytime curve for one (expression, rule set) pair.
#[derive(Clone, Debug)]
pub struct AnytimeCurve {
    /// One entry per grid target, in grid order (clamped rows included, so
    /// every curve has the same shape — analysis never has to ragged-join).
    pub checkpoints: Vec<AnytimeCheckpoint>,
    /// How the run ended. [`SaturationStop::ApplicationBudget`] here means
    /// the grid was exhausted while saturation was still finding unions —
    /// the curve is truncated at the last grid target, which is then the
    /// "cap" point.
    pub ended: SaturationStop,
    /// Cumulative applications when the run ended.
    pub ended_at_apps: usize,
}

/// Curve plus the final e-graph/extraction (for hindsight labeling by
/// callers that need it, e.g. the oracle-filtered harness).
pub struct AnytimeCurveOutput {
    pub curve: AnytimeCurve,
    pub egraph: EGraph,
    pub extraction: ExtractedDAG,
}

/// Run one incremental saturation of `(arena, root)` under `rules`, sampling
/// extraction cost at each application-count target in `grid`.
///
/// `max_classes` is the fixed environment cap (production tier value),
/// `max_sweeps` a generous safety ceiling on total sweeps, and
/// `safety_timeout` a per-curve wall-clock ceiling that PANICS if exceeded
/// (offline measurement must fail loud, never silently truncate — the
/// 2026-09-01 review round caught exactly that failure mode twice).
pub fn run_anytime_curve(
    arena: &ExprArena,
    root: ExprId,
    rules: Vec<Box<dyn Rewrite>>,
    grid: &[usize],
    max_classes: usize,
    max_sweeps: usize,
    safety_timeout: Duration,
    costs: &CostModel,
) -> AnytimeCurveOutput {
    assert!(!grid.is_empty(), "anytime: empty checkpoint grid");
    assert!(
        grid.windows(2).all(|w| w[0] < w[1]),
        "anytime: checkpoint grid must be strictly increasing, got {grid:?}"
    );

    let mut egraph = EGraph::with_rules(rules);
    let root_class = egraph.add_arena(arena, root);
    let deadline = Instant::now() + safety_timeout;

    let mut checkpoints: Vec<AnytimeCheckpoint> = Vec::with_capacity(grid.len());
    let mut sweeps_total = 0usize;
    let mut ended: Option<SaturationStop> = None;
    let mut last_live: Option<AnytimeCheckpoint> = None;

    for &target in grid {
        if let (Some(_), Some(prev)) = (ended, last_live) {
            // Run already ended at an earlier checkpoint: fill from the
            // final state instead of re-sampling (cost is frozen).
            checkpoints.push(AnytimeCheckpoint {
                app_target: target,
                clamped: true,
                ..prev
            });
            continue;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(
            !remaining.is_zero(),
            "anytime: curve exceeded the {safety_timeout:?} safety ceiling before target \
             {target} — fail loud rather than report a partial curve as data"
        );
        let sweeps_left = max_sweeps.checked_sub(sweeps_total).unwrap_or_else(|| {
            panic!("anytime: sweep accounting underflow (sweeps_total={sweeps_total})")
        });
        let stats = egraph.saturate_until_applications(target, sweeps_left, max_classes, remaining);
        assert!(
            stats.stop != SaturationStop::Timeout,
            "anytime: saturation hit the wall-clock safety ceiling at target {target} — \
             offline measurement must fail loud, never silently truncate"
        );
        sweeps_total += stats.iterations;
        let extraction = extract_dag(&egraph, root_class, costs);
        let cp = AnytimeCheckpoint {
            app_target: target,
            app_actual: stats.applications,
            sweeps: sweeps_total,
            classes: egraph.num_classes(),
            nodes: egraph.node_count(),
            cost: extraction.total_cost,
            stop: stats.stop,
            clamped: false,
        };
        checkpoints.push(cp);
        last_live = Some(cp);
        if stats.stop != SaturationStop::ApplicationBudget {
            ended = Some(stats.stop);
        }
    }

    let last = last_live.expect("grid is non-empty, so at least one live checkpoint exists");
    let extraction = extract_dag(&egraph, root_class, costs);
    AnytimeCurveOutput {
        curve: AnytimeCurve {
            checkpoints,
            ended: ended.unwrap_or(SaturationStop::ApplicationBudget),
            ended_at_apps: last.app_actual,
        },
        egraph,
        extraction,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::all_rules;
    use pixelflow_ir::OpKind;

    fn small_expr() -> (ExprArena, ExprId) {
        // (x + y) * (x + y) + 2 * (x + y) — the `redundant` shader shape.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let s = a.push_binary(OpKind::Add, x, y);
        let s2 = a.push_binary(OpKind::Mul, s, s);
        let two = a.push_const(2.0);
        let ts = a.push_binary(OpKind::Mul, two, s);
        let out = a.push_binary(OpKind::Add, s2, ts);
        (a, out)
    }

    #[test]
    fn curve_has_one_row_per_grid_target_and_monotone_cost() {
        let (arena, root) = small_expr();
        let grid = [25, 50, 100, 200, 400];
        let out = run_anytime_curve(
            &arena,
            root,
            all_rules(),
            &grid,
            2000,
            10_000,
            Duration::from_secs(60),
            &CostModel::latency_prior(),
        );
        assert_eq!(out.curve.checkpoints.len(), grid.len());
        for (cp, &target) in out.curve.checkpoints.iter().zip(grid.iter()) {
            assert_eq!(cp.app_target, target);
            assert!(
                cp.clamped
                    || cp.app_actual >= target
                    || cp.stop != SaturationStop::ApplicationBudget,
                "live checkpoint below target without the run having ended: {cp:?}"
            );
        }
        // Anytime cost is non-increasing along the curve (append-only growth
        // only ever adds extraction options).
        for w in out.curve.checkpoints.windows(2) {
            assert!(
                w[1].cost <= w[0].cost,
                "extraction cost increased along the anytime curve: {w:?}"
            );
        }
    }

    #[test]
    fn clamped_rows_freeze_the_final_state() {
        let (arena, root) = small_expr();
        // Absurdly large later targets force clamping after quiescence.
        let grid = [25, 1_000_000, 2_000_000];
        let out = run_anytime_curve(
            &arena,
            root,
            all_rules(),
            &grid,
            2000,
            10_000,
            Duration::from_secs(60),
            &CostModel::latency_prior(),
        );
        assert_eq!(out.curve.ended, SaturationStop::Quiesced);
        let cps = &out.curve.checkpoints;
        assert!(cps.iter().any(|c| c.clamped), "expected clamped tail rows");
        let final_cost = cps.last().unwrap().cost;
        for c in cps.iter().filter(|c| c.clamped) {
            assert_eq!(c.cost, final_cost);
            assert_eq!(c.app_actual, out.curve.ended_at_apps);
        }
    }
}
