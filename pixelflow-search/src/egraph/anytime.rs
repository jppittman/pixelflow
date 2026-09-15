//! Anytime (quality-at-budget) curve sampling for e-graph saturation.
//!
//! ONE definition of the guided-saturation program's anytime metric
//! (docs/plans/2026-08-31-guide-design-revision.md §0/§5, and the Phase 3
//! pre-registration docs/plans/2026-09-01-phase3-registration.md): best
//! extraction cost under a static, deterministic cost model, plotted against
//! work performed, where work is denominated in **rule applications** — never
//! wall-clock, never sweeps. Every harness imports this module instead of
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
//! - The x-axis counts **applications** exactly as
//!   [`EGraph::application_count`] counts them — one per action commit,
//!   including idempotent re-fires (91% of the total). That is deliberate: a
//!   Guide pays for scoring and scanning a candidate whether or not it
//!   commits, so the honest work axis counts them. It is also the budget's
//!   own denominator, and unconditional — never gated on whether anyone is
//!   recording provenance.
//! - The per-checkpoint budget is a **delta, not an absolute target**.
//!   [`Budget::Applications`] resolves to `current + n`, and
//!   [`EGraph::application_count`] is cumulative, so a stepper passes the
//!   *gap* to the next grid point. Reading it the other way silently makes
//!   the x-axis a prefix sum of itself.
//! - Checkpoint targets are crossed **mid-scan**, at the moment an
//!   application is about to commit ([`ScanStop::ApplicationBudget`](super::graph::ScanStop::ApplicationBudget)),
//!   so `app_actual == app_target` exactly. **This is an instrument change
//!   from Round 1**, where the target was crossed at rule-sweep granularity
//!   and [`AnytimeCheckpoint::app_actual`] recorded an overshoot of up to one
//!   sweep. It is a strictly better instrument — the overshoot was an
//!   artifact — but it means a re-run does not reproduce the registered
//!   curves even with nothing else changed, and `app_actual` can no longer
//!   explain a discrepancy against a Round-1 artifact. It stays in the schema
//!   because a curve that always equals its target is still worth asserting.
//! - The reported cost is [`ChoiceCost::dag`], what the emitted kernel pays.
//!   **Also an instrument change**: every Round-1 number was measured on the
//!   TREE cost, which prices a shared subterm once per use — 1.4e7 against
//!   716 on `shader:julia_set` (#1117).
//! - The class cap is **environment, not work**: the production tier's
//!   memory-protection cap, fixed for the whole curve and identical across
//!   any curves being compared.
//! - Wall clock appears ONLY as [`Optimizer::hard_ceiling`], which panics.
//!   It never gates which samples count, and no stop reason can name it —
//!   the budgeted loops take no clock at all.

use alloc::vec::Vec;

use pixelflow_ir::{ExprArena, ExprId};

use super::extract::ChoiceCost;
use super::graph::{EGraph, SaturationStop};
use super::node::EClassId;
use super::optimizer::{Budget, Limits, Optimizer};

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
    /// Cumulative applications at the sample — the real x-value. Equal to
    /// `app_target` on a live row now that the budget binds mid-scan; see
    /// the module doc's instrument-change note.
    pub app_actual: usize,
    /// Cumulative completed sweeps at the sample (diagnostic only).
    pub sweeps: usize,
    /// Cumulative rule-match ATTEMPTS at the sample
    /// ([`EGraph::total_evals`]) — "raw matches enumerated", the
    /// denominator Round 2's Guide-overhead-flatness precondition needs
    /// (docs/plans/2026-09-01-phase3-round2-registration-v2.md §7.1).
    /// Diagnostic: nothing in the curve's y-axis reads it.
    pub evals_actual: usize,
    /// E-class count at the sample.
    pub classes: usize,
    /// E-node count at the sample.
    pub nodes: usize,
    /// Extraction cost at the sample, in both shapes. Plot
    /// [`ChoiceCost::dag`] — [`ChoiceCost::tree`] is the DP's own objective,
    /// kept so a Round-1 comparison can name which number it is looking at.
    pub cost: ChoiceCost,
    /// Stop reason of the saturation call that produced this sample;
    /// [`SaturationStop::ApplicationBudget`] means the curve was still live.
    pub stop: SaturationStop,
    /// `true` when the run had already ended (quiesced / class cap /
    /// iteration ceiling) at or before an EARLIER checkpoint and this row is
    /// filled from the final state rather than newly sampled.
    pub clamped: bool,
}

/// A full anytime curve for one (expression, configuration) pair.
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

/// Curve plus the final e-graph and root, for hindsight labeling by callers
/// that need it.
pub struct AnytimeCurveOutput {
    /// The curve.
    pub curve: AnytimeCurve,
    /// The e-graph as the last checkpoint left it.
    pub egraph: EGraph,
    /// The root e-class in [`Self::egraph`].
    pub root: EClassId,
    /// The final extraction's choices and cost.
    pub extraction: super::optimizer::Optimized,
}

/// Number of DISTINCT arena nodes reachable from `root`.
///
/// Not `ExprArena::node_count_subtree`, which is documented to count a shared
/// subtree once per reference (tree size, not DAG size) and so would not
/// equal `nodes_raw().len()` for any expression with sharing.
fn reachable_node_count(arena: &ExprArena, root: ExprId) -> usize {
    let mut seen = alloc::vec![false; arena.nodes_raw().len()];
    let mut stack = alloc::vec![root];
    let mut count = 0usize;
    while let Some(id) = stack.pop() {
        let slot = &mut seen[id.0 as usize];
        if *slot {
            continue;
        }
        *slot = true;
        count += 1;
        stack.extend(arena.children(id));
    }
    count
}

/// Run one incremental saturation of `(arena, root)` under `optimizer`,
/// sampling extraction cost at each application-count target in `grid`.
///
/// The guided and unguided arms differ **only** in whether `optimizer`
/// carries a [`guide`](Optimizer::guide) — there is no second curve
/// function and no stepper trait, because after #1108 the thing that varies
/// is a field. Everything else about the curve (grid, sampling, clamping,
/// extraction, the regret reference convention) is shared, which was the
/// whole point of having one definition.
///
/// The curve's *environment* — the class cap and the total sweep ceiling —
/// comes from `optimizer`'s own budget resolved at this expression's node
/// count, so a registration pins it by naming a
/// [`Budget::Explicit`]. Only the application dimension is overridden, once
/// per checkpoint, with the **gap** to the next grid point.
///
/// # Panics
///
/// - If `grid` is empty or not strictly increasing.
/// - If `arena` holds nodes unreachable from `root` (see below).
/// - If `optimizer`'s wall-clock ceiling is exceeded — offline measurement
///   fails loud, never silently truncates.
pub fn run_anytime_curve(
    optimizer: &mut Optimizer,
    arena: &ExprArena,
    root: ExprId,
    grid: &[usize],
) -> AnytimeCurveOutput {
    assert!(!grid.is_empty(), "anytime: empty checkpoint grid");
    assert!(
        grid.windows(2).all(|w| w[0] < w[1]),
        "anytime: checkpoint grid must be strictly increasing, got {grid:?}"
    );

    // An append-only `ExprArena` that has been rewritten in place carries
    // abandoned nodes, and saturating them would spend this curve's
    // application budget rewriting an expression nobody asked about — making
    // the curve a function of the arena's allocation history rather than of
    // the expression. This used to be a precondition callers had to meet by
    // compacting first, asserted here, because `EGraph::add_arena` inserted
    // every node of `arena` rather than the subtree reachable from `root`.
    // `egraph::insert` is reachable-only, so the property now holds by
    // construction and the budget is keyed on exactly what was inserted.
    let node_count = reachable_node_count(arena, root);

    let mut egraph = optimizer.egraph();
    let root_class = crate::egraph::insert(
        arena,
        root,
        &mut egraph,
        crate::egraph::Vocabulary::Templates,
    )
    .expect("anytime: arena must be e-graph representable");
    // Sized after insertion and before any rewrite, as `Optimizer::run`
    // sizes it: the class count here is the hash-consed input.
    let input = super::saturate::InputSize {
        nodes: node_count,
        classes: egraph.num_classes(),
    };
    let env = optimizer.limits_for(input);

    let mut checkpoints: Vec<AnytimeCheckpoint> = Vec::with_capacity(grid.len());
    let mut sweeps_total = 0usize;
    let mut ended: Option<SaturationStop> = None;
    let mut last_live: Option<AnytimeCheckpoint> = None;
    let mut final_extraction = None;

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
        let sweeps_left = env.iterations.checked_sub(sweeps_total).unwrap_or_else(|| {
            panic!("anytime: sweep accounting underflow (sweeps_total={sweeps_total})")
        });
        // THE DELTA. `Budget::Applications(n)` is `current + n`; passing
        // `target` here would make each checkpoint cost the whole prefix
        // again and the x-axis a prefix sum of itself.
        let already = egraph.application_count();
        let delta = (target as u64).saturating_sub(already);
        let out = optimizer.run_bounded(
            &mut egraph,
            root_class,
            Limits {
                iterations: sweeps_left,
                classes: env.classes,
                applications: Some(delta),
            },
            input,
        );
        assert_ne!(
            out.stats.stop,
            SaturationStop::Timeout,
            "anytime: a budgeted saturation reported Timeout, which it cannot do — the \
             budgeted loops take no clock (target {target})"
        );
        sweeps_total += out.stats.iterations;
        let cp = AnytimeCheckpoint {
            app_target: target,
            app_actual: out.stats.applications as usize,
            sweeps: sweeps_total,
            evals_actual: egraph.total_evals(),
            classes: egraph.num_classes(),
            nodes: egraph.node_count(),
            cost: out.cost,
            stop: out.stats.stop,
            clamped: false,
        };
        checkpoints.push(cp);
        last_live = Some(cp);
        if out.stats.stop != SaturationStop::ApplicationBudget {
            ended = Some(out.stats.stop);
        }
        final_extraction = Some(out);
    }

    let last = last_live.expect("grid is non-empty, so at least one live checkpoint exists");
    let extraction =
        final_extraction.expect("a live checkpoint always produces an extraction to keep");
    AnytimeCurveOutput {
        curve: AnytimeCurve {
            checkpoints,
            ended: ended.unwrap_or(SaturationStop::ApplicationBudget),
            ended_at_apps: last.app_actual,
        },
        egraph,
        root: root_class,
        extraction,
    }
}

/// The registered Phase 3 measurement environment: the classical tier's
/// class cap, a generous total sweep ceiling, and no application cap of its
/// own (the curve supplies one per checkpoint).
///
/// A named constructor rather than a comment on every harness, because it is
/// a **registered constant** — `docs/plans/2026-09-01-phase3-registration.md`
/// — and a harness that spells its own is a harness that can drift from the
/// registration silently.
#[must_use]
pub fn registered_curve_budget() -> Budget {
    Budget::Explicit {
        iterations: 10_000,
        classes: 2_000,
        applications: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn curve_optimizer() -> Optimizer {
        Optimizer::production()
            .budget(registered_curve_budget())
            .no_ceiling()
    }

    #[test]
    fn curve_has_one_row_per_grid_target_and_monotone_cost() {
        let (arena, root) = small_expr();
        let grid = [25, 50, 100, 200, 400];
        let mut opt = curve_optimizer();
        let out = run_anytime_curve(&mut opt, &arena, root, &grid);
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
        // Pinned on THIS fixture only — not a general law. `extract_dag` is a
        // heuristic DAG extraction, so anytime cost can rise before it falls
        // (measured on 2026-09-01: classical median regret at B=100 exceeds
        // B=25), which is why the registered regret reference is the
        // empirical best over all checkpoints, never the final state.
        for w in out.curve.checkpoints.windows(2) {
            assert!(
                w[1].cost.dag <= w[0].cost.dag,
                "extraction cost increased along the anytime curve: {w:?}"
            );
        }
    }

    /// The budget is a delta, so a live checkpoint lands EXACTLY on its
    /// target — the instrument change from Round 1's sweep granularity. A
    /// stepper that passed the absolute target instead would land at the
    /// prefix sum and this would read far above the grid.
    #[test]
    fn live_checkpoints_land_exactly_on_their_target() {
        let (arena, root) = small_expr();
        let grid = [5, 10, 20];
        let mut opt = curve_optimizer();
        let out = run_anytime_curve(&mut opt, &arena, root, &grid);
        for cp in &out.curve.checkpoints {
            if cp.clamped || cp.stop != SaturationStop::ApplicationBudget {
                continue;
            }
            assert_eq!(
                cp.app_actual, cp.app_target,
                "the application cap binds mid-scan, so a live sample lands on its target: \
                 {cp:?}"
            );
        }
    }

    #[test]
    fn clamped_rows_freeze_the_final_state() {
        let (arena, root) = small_expr();
        // Absurdly large later targets force clamping after quiescence.
        let grid = [25, 1_000_000, 2_000_000];
        let mut opt = curve_optimizer();
        let out = run_anytime_curve(&mut opt, &arena, root, &grid);
        // Any terminal reason will do — this fixture reaches the registered
        // 2 000-class cap before it quiesces, and which of the two fires is
        // a property of the rule set, not of clamping. What clamping owes is
        // that once the run has ended, later grid rows are frozen copies.
        assert_ne!(
            out.curve.ended,
            SaturationStop::ApplicationBudget,
            "a 2 000 000-application grid must outlast this expression"
        );
        let cps = &out.curve.checkpoints;
        assert!(cps.iter().any(|c| c.clamped), "expected clamped tail rows");
        let final_cost = cps.last().expect("non-empty grid").cost.dag;
        for c in cps.iter().filter(|c| c.clamped) {
            assert_eq!(c.cost.dag, final_cost);
            assert_eq!(c.app_actual, out.curve.ended_at_apps);
        }
    }

    /// A guided curve is the same function with a field set — no second
    /// entry point, and the episode's dedup set carried across checkpoints
    /// because the `Optimizer` outlives them.
    #[test]
    fn a_guided_curve_is_the_same_curve_with_a_guide_attached() {
        use crate::nnue::factored::OpEmbeddings;
        use crate::nnue::guide::Guide;
        use alloc::boxed::Box;

        let (arena, root) = small_expr();
        let grid = [25, 50, 100];
        let mut opt = curve_optimizer().guide(Some(Box::new(Guide::new_random(
            OpEmbeddings::new_random(5),
            9,
        ))));
        let out = run_anytime_curve(&mut opt, &arena, root, &grid);
        assert_eq!(out.curve.checkpoints.len(), grid.len());
        assert!(out.curve.checkpoints.iter().all(|c| c.cost.dag > 0));
    }
}
