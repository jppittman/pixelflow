//! Budget-limited saturation with instrumentation.
//!
//! This module provides depth-limited saturation for training data generation.
//! The key insight from Stockfish-style training: we want NNUE to predict
//! "what's achievable within budget", not the theoretical optimum.
//!
//! # Usage
//!
//! ```ignore
//! let mut eg = EGraph::new();
//! let root = insert_tree(&mut eg, &expr_tree);
//! let result = saturate_with_budget(&mut eg, 100);
//!
//! // result contains stats about what happened during saturation
//! println!("Unions: {}, Saturated: {}", result.total_unions, result.saturated);
//! ```

use rustc_hash::FxHashMap as HashMap;

use super::graph::{EGraph, SaturationStop};
use super::node::EClassId;
use super::rules::RuleId;

/// Result of a budget-limited saturation run.
///
/// This captures everything needed for training data generation:
/// - How much work was done (iterations, unions)
/// - Whether saturation completed or was cut off
/// - E-graph size before and after
#[derive(Clone, Debug)]
pub struct SaturationResult {
    /// Number of iterations completed.
    pub iterations: usize,

    /// Total unions performed across all iterations.
    pub total_unions: usize,

    /// Whether saturation completed (no more changes) before budget
    /// exhausted. Exactly `stop == SaturationStop::Quiesced` — the same
    /// decision the loop made, never a second opinion derived from the
    /// counters (`iterations < max_iterations` is true of a class-cap and a
    /// timeout break too).
    pub saturated: bool,

    /// Number of e-classes before saturation.
    pub classes_before: usize,

    /// Number of e-classes after saturation.
    pub classes_after: usize,

    /// Rule match counts by rule name.
    /// Rule match counts, keyed by stable rule identity.
    ///
    /// Was keyed by `Rewrite::name()`, which is a *family* name: all four
    /// `Commutative` instances answered to `"commutative"` and landed in one
    /// bucket, so every per-rule number derived from this map was wrong by
    /// aggregation. [`RuleId`](crate::egraph::RuleId) is per instance.
    pub rule_matches: HashMap<RuleId, usize>,

    /// The rewrite budget that was used.
    pub budget: usize,

    /// The rest of the budget triple this run was given (`budget` is its
    /// `max_iterations`) — recorded on the result so an observer of the run
    /// (the `saturation-telemetry` feature) reads the limits that actually
    /// applied instead of re-deriving them from `node_count`.
    pub max_classes: usize,
    pub hard_timeout: std::time::Duration,

    /// Which condition ended the run — read off `EGraph::saturate_with_limits`'s
    /// own stopping decision, not inferred from the counts above (those can
    /// tie: a class-cap or timeout break can leave `iterations <
    /// max_iterations` exactly like a quiesced run).
    pub stop: SaturationStop,
}

impl SaturationResult {
    /// Calculate the improvement ratio (how much the e-graph grew).
    pub fn growth_ratio(&self) -> f64 {
        if self.classes_before == 0 {
            1.0
        } else {
            self.classes_after as f64 / self.classes_before as f64
        }
    }

    /// Whether the budget was exhausted (saturation was cut off).
    pub fn budget_exhausted(&self) -> bool {
        !self.saturated && self.iterations >= self.budget
    }
}

/// Run saturation with a budget limit, returning detailed statistics.
///
/// This is the teacher for Stockfish-style training: it runs full saturation
/// (up to the budget) and records what cost was achievable.
///
/// # Arguments
///
/// * `egraph` - The e-graph to saturate (mutated in place)
/// * `max_iterations` - Maximum number of saturation iterations (rewrite budget)
///
/// # Returns
///
/// A `SaturationResult` containing statistics about the saturation run.
///
/// # Example
///
/// ```ignore
/// let mut eg = EGraph::new();
/// let root = eg.add(ENode::Var(0));
/// let result = saturate_with_budget(&mut eg, 100);
/// assert!(result.saturated || result.iterations <= 100);
/// ```
pub fn saturate_with_budget(egraph: &mut EGraph, max_iterations: usize) -> SaturationResult {
    saturate_with_full_budget(
        egraph,
        max_iterations,
        10_000,
        std::time::Duration::from_secs(5),
    )
}

/// Run saturation with budget, class, and time limits.
///
/// Unlike `saturate_with_budget`, this gives full control over safety limits.
/// The e-graph stops growing when ANY limit is reached.
pub fn saturate_with_full_budget(
    egraph: &mut EGraph,
    max_iterations: usize,
    max_classes: usize,
    timeout: std::time::Duration,
) -> SaturationResult {
    let classes_before = egraph.classes.len();
    egraph.match_counts.clear();

    // One call drives the entire multi-round run — the loop that decides
    // when to stop (timeout, class limit, or convergence) lives exactly
    // once, in `EGraph::saturate_with_limits`
    // (docs/plans/2026-08-17-cost-model-domain.md, J11). This module
    // previously re-decided the same stopping conditions in a second,
    // hand-rolled outer loop that drove the e-graph one round at a time —
    // the duplicate-loop drift the domain model doc calls out by name.
    let stats = egraph.saturate_with_limits(max_iterations, max_classes, timeout);

    let saturated = stats.stop == SaturationStop::Quiesced;
    let classes_after = egraph.classes.len();
    let rule_matches = egraph.match_counts.clone();

    SaturationResult {
        iterations: stats.iterations,
        total_unions: stats.total_unions,
        saturated,
        classes_before,
        classes_after,
        rule_matches,
        budget: max_iterations,
        max_classes,
        hard_timeout: timeout,
        stop: stats.stop,
    }
}

// ============================================================================
// Saturation budget presets — the compile-time policy shared by every caller
// ============================================================================

/// Safety limits for one saturation run, chess-clock style: several
/// independent limits, any one tripping ends the run. This is the policy
/// knob; [`saturate_with_full_budget`] is the mechanism both the AOT macro
/// tier (`pixelflow-compiler`) and the runtime tier
/// ([`crate::runtime::optimize_runtime_arena`]) drive with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaturationConfig {
    /// Rewrite-round budget (each round applies every rule once).
    pub max_iterations: usize,
    /// E-class count budget — caps pathological e-graph blowup.
    pub max_classes: usize,
    /// Rule-application budget — deterministic replacement for wall clock
    /// as a *stopping condition*. See
    /// `docs/plans/2026-09-01-production-budget-determinism.md` for the
    /// calibration (40 applications per e-class of the tier's own class
    /// cap, 1.9x the highest applications-per-class ratio ever observed).
    pub max_applications: u64,
    /// Wall-clock **assertion**, not a budget dimension: exceeding it is a
    /// bug in the budget (or the ceiling), so [`Optimizer::hard_ceiling`]
    /// panics rather than silently truncating and reporting success — which
    /// is what the old `hard_timeout` did when passed into
    /// [`EGraph::saturate_with_limits`]. Named `safety_ceiling` rather than
    /// `hard_timeout` to make that distinction unmissable at every call
    /// site.
    ///
    /// [`Optimizer::hard_ceiling`]: super::optimizer::Optimizer::hard_ceiling
    pub safety_ceiling: std::time::Duration,
}

/// The application budget per e-class of a tier's class cap — 1.9× the
/// highest applications-per-class ratio ever observed at a tier's own cap
/// (docs/plans/2026-09-01-production-budget-determinism.md, "Decision: the
/// application budgets").
pub const APPLICATIONS_PER_CLASS: u64 = 40;

/// The safety ceiling per application of a tier's budget: a floor
/// throughput of 1 application/ms, rounded up by half — 14× below the
/// slowest rate ever measured — so the ceiling is "this budget could not
/// take this long on any machine we would build on" (the same plan,
/// "Decision: the wall-clock safety ceiling"; 20,000 / 80,000 / 200,000
/// applications give the 30 s / 120 s / 300 s it tabulates).
const SAFETY_CEILING_PER_APPLICATION: std::time::Duration = std::time::Duration::from_micros(1_500);

/// How many e-classes the classical cap grants per e-class the input
/// **inserts** to — the hash-consed size the e-graph actually holds, which
/// is what the first round of rewrites multiplies. Calibrated on the
/// 2026-09-08 class-cap sweep (`docs/results/2026-09-08-class-cap-sweep.md`):
/// the production rule set's first sweep grows every DejaVu glyph to
/// between 6 and 7 times its inserted size — at 6 per inserted class the
/// same 44 of 95 glyphs stop on the cap inside round 1 as at the flat
/// 5,000, at 7 none do, and 7 and 8 extract identical terms (Σ `dag_cost`
/// −16.7% against the flat cap on both). 8 rather than the measured
/// threshold's 7 buys a held-out font one seventh of headroom over the cliff
/// for a third more compile time on the raised glyphs.
pub const CLASSICAL_CLASSES_PER_INSERTED_CLASS: usize = 8;

/// The classical cap's floor — the flat cap every classical kernel had
/// before 2026-09-08, kept for every kernel whose input does not ask for
/// more (the shaders, the cell grid, the scene kernels).
pub const CLASSICAL_CLASS_FLOOR: usize = 5_000;

/// The classical cap's ceiling — **pinned at the floor, which switches the
/// input-sized cap off.**
///
/// The rule is calibrated and measured (8 per inserted class, ceiling 50,000:
/// `docs/results/2026-09-08-class-cap-sweep.md`, −16.7% Σ `dag_cost` on the
/// 44 DEJaVu glyphs it raises, none dearer) and it is blocked by a rendering
/// defect it exposes: raising `'8'` off the floor changes which fusion the
/// extractor picks for its quadratic solver, `disc >= 0` at the waist's
/// tangency lands on the other side of exact zero, and a half-covered smear
/// appears along the waist at 13–21 px where FreeType has no ink
/// (`pixelflow-graphics/tests/freetype_oracle.rs`, the optimized arm — the
/// `'8'` defect the raw arm's oracle found in 2026-09, back by the route its
/// own comment predicted). The knife edge is the glyph kernel's
/// (`quad_tangency_winding.rs`), not the optimizer's, and it has to be fixed
/// there first. When it is, this constant becomes 50,000 — a 50,000-class
/// e-graph peaks near 25 MB of live heap on the sweep's glyphs and the
/// extraction pass's reach sets stay under 2 MB there — and nothing else
/// changes.
pub const CLASSICAL_CLASS_CEILING: usize = CLASSICAL_CLASS_FLOOR;

/// What [`CLASSICAL_CLASS_CEILING`] becomes when the `'8'` tangency is fixed:
/// the ceiling the sweep calibrated the rule under.
pub const CLASSICAL_CLASS_CEILING_CALIBRATED: usize = 50_000;

// The saturation loop clamps every cap to `HARD_CLASS_LIMIT`; a ceiling above
// it would be a cap that silently never applies.
const _: () = assert!(CLASSICAL_CLASS_CEILING_CALIBRATED <= super::graph::HARD_CLASS_LIMIT);
const _: () = assert!(CLASSICAL_CLASS_FLOOR <= CLASSICAL_CLASS_CEILING);
const _: () = assert!(CLASSICAL_CLASS_CEILING <= CLASSICAL_CLASS_CEILING_CALIBRATED);

impl SaturationConfig {
    /// A tier from its two free dimensions: the application budget and the
    /// safety ceiling are derived, so the plan's ratios hold at every cap
    /// rather than at three hand-copied points.
    const fn tier(max_iterations: usize, max_classes: usize) -> Self {
        let max_applications = max_classes as u64 * APPLICATIONS_PER_CLASS;
        Self {
            max_iterations,
            max_classes,
            max_applications,
            safety_ceiling: SAFETY_CEILING_PER_APPLICATION
                .checked_mul(max_applications as u32)
                .expect("a safety ceiling fits a Duration"),
        }
    }

    /// Trivial expressions (≤10 nodes): minimal budget.
    pub fn blitz() -> Self {
        Self::tier(20, 500)
    }

    /// Normal complexity (11-50 nodes): balanced.
    pub fn rapid() -> Self {
        Self::tier(50, 2_000)
    }

    /// Complex expressions (51+ nodes) at the classical **floor**: the flat
    /// 5,000-class cap. Production sizes classical from the input through
    /// [`Self::classical_for`]; this is what that resolves to for any input
    /// of at most 500 inserted classes, and what every offline caller that
    /// names "the classical budget" without an input gets.
    pub fn classical() -> Self {
        Self::tier(100, CLASSICAL_CLASS_FLOOR)
    }

    /// Complex expressions, with the class cap sized by what the input
    /// inserts to: [`CLASSICAL_CLASSES_PER_INSERTED_CLASS`] per inserted
    /// class, never below [`Self::classical`]'s floor nor above
    /// [`CLASSICAL_CLASS_CEILING`]. The application budget and the safety
    /// ceiling scale with it. **Inert while the ceiling is pinned at the
    /// floor** — see [`CLASSICAL_CLASS_CEILING`] for what blocks it.
    ///
    /// Why the inserted class count and not the node count the tier is
    /// keyed on: a `Kernel` built by composition re-expands its shared
    /// subtrees, so the arena is 2.3 nodes per class on a glyph and 840 on
    /// the chrome scene (a 390,815-node tree that hash-conses to 465).
    /// Sizing by nodes would hand the chrome scene the ceiling, and at the
    /// ceiling its extraction is 42% dearer (the sweep's chrome rows); by
    /// inserted classes it keeps the floor it had.
    pub fn classical_for(inserted_classes: usize) -> Self {
        let classes = inserted_classes
            .saturating_mul(CLASSICAL_CLASSES_PER_INSERTED_CLASS)
            .clamp(CLASSICAL_CLASS_FLOOR, CLASSICAL_CLASS_CEILING);
        Self::tier(100, classes)
    }

    /// The pre-2026-09 fixed budget — 10,000 e-classes and 500 ms, no
    /// application cap — with the caller's own round count.
    ///
    /// **Not a production preset.** Production sizes its budget from the
    /// expression via [`config_for_node_count`], reached through
    /// [`Optimizer::run`](super::optimizer::Optimizer::run).
    /// This one exists for the non-production call sites that must reproduce
    /// results from before that policy landed — unit tests, the hindsight
    /// labeler, and offline measurement harnesses — so the budget is named
    /// once here instead of the same two magic numbers being re-spelled at
    /// every such site, where they would drift apart the moment one of them
    /// was revised. `max_applications` is `u64::MAX`: an application budget
    /// was never one of the three dimensions this regime bounded, and giving
    /// it a finite value here would silently change what these sites
    /// reproduce.
    ///
    /// The round count stays a parameter because it always was one: the
    /// budget these sites inherited fixed the class cap and the deadline but
    /// let each caller choose how many rewrite rounds it wanted. A site that
    /// needs to vary one of the other two should say so at the call site,
    /// with a reason:
    ///
    /// ```ignore
    /// SaturationConfig {
    ///     safety_ceiling: SAFETY_CEILING, // offline: measuring caps, not the machine
    ///     ..SaturationConfig::compatibility(100)
    /// }
    /// ```
    pub fn compatibility(max_iterations: usize) -> Self {
        Self {
            max_iterations,
            max_classes: 10_000,
            max_applications: u64::MAX,
            safety_ceiling: std::time::Duration::from_millis(500),
        }
    }

    /// Run one saturation of `egraph` under this budget.
    ///
    /// Goes through [`EGraph::saturate_with_limits`], which knows only
    /// three limits (rounds, classes, a real wall-clock deadline) — the
    /// same regime [`SaturationConfig::compatibility`] exists to reproduce.
    /// `max_applications` is not threaded through here: no caller of this
    /// method wants an application-budgeted, clock-truncated run at once,
    /// and production reaches the application budget through
    /// [`Optimizer::run`](super::optimizer::Optimizer::run) /
    /// [`EGraph::saturate_budgeted`] instead, which never looks at a clock.
    pub fn run(&self, egraph: &mut EGraph) -> super::graph::SaturationStats {
        egraph.saturate_with_limits(self.max_iterations, self.max_classes, self.safety_ceiling)
    }
}

/// The two sizes production's budget is keyed on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputSize {
    /// A rough node count — the tier key. AST node count for the macro
    /// tier, reachable-arena-node count for the runtime tier: a proxy, and
    /// both serve equally well as "how big is this expression".
    pub nodes: usize,
    /// E-classes the input inserted to before any rewrite — the hash-consed
    /// size, which sizes the classical class cap
    /// ([`SaturationConfig::classical_for`]).
    pub classes: usize,
}

/// Pick a [`SaturationConfig`] from the input's sizes.
///
/// | Nodes | Config | Rationale |
/// |-------|--------|-----------|
/// | 0-10 | blitz | Trivial expressions need minimal optimization |
/// | 11-50 | rapid | Normal complexity, balanced approach |
/// | 51+ | classical, sized by `input.classes` | Complex expressions need thorough search, and a cap the input itself does not fill |
pub fn config_for_input(input: InputSize) -> SaturationConfig {
    tier_for_input(input).1
}

/// The tier's **floor** preset for a rough node count — classical at its
/// flat 5,000-class floor. Production goes through [`config_for_input`],
/// which sizes classical from the inserted class count; this is what an
/// offline caller that has only a node count gets, and what every tier
/// resolves to for an input of at most 500 inserted classes.
pub fn config_for_node_count(node_count: usize) -> SaturationConfig {
    config_for_input(InputSize {
        nodes: node_count,
        classes: 0,
    })
}

/// The one place the tier thresholds live: a tier's **name** and its
/// **budget** come out of the same match, so a diagnostic that names a tier
/// can never disagree with the budget it is describing.
///
/// [`config_for_input`] is the budget half. The name half exists for
/// [`Optimizer::run`](super::optimizer::Optimizer::run)'s safety-ceiling
/// panic, which has to say *which* tier's ceiling was exceeded — and which,
/// spelled as its own `match` on the node count, would be a second copy of
/// this table, free to drift out of step with it.
pub(crate) fn tier_for_input(input: InputSize) -> (&'static str, SaturationConfig) {
    match input.nodes {
        0..=10 => ("blitz", SaturationConfig::blitz()),
        11..=50 => ("rapid", SaturationConfig::rapid()),
        _ => ("classical", SaturationConfig::classical_for(input.classes)),
    }
}

/// Configuration for multi-budget training data generation.
///
/// Generate training data at multiple budget levels for curriculum learning.
#[derive(Clone, Debug)]
pub struct MultiBudgetConfig {
    /// Budget levels to generate data at (e.g., [50, 100, 200, 500]).
    pub budgets: Vec<usize>,

    /// Number of samples to generate at each budget level.
    pub samples_per_budget: usize,
}

impl Default for MultiBudgetConfig {
    fn default() -> Self {
        Self {
            budgets: vec![50, 100, 200, 500],
            samples_per_budget: 2500,
        }
    }
}

/// Extract the best achievable cost within budget.
///
/// This is the ground truth label for training: given an expression,
/// what's the lowest cost we can achieve with `budget` rewrite iterations?
pub fn achievable_cost_within_budget(
    egraph: &mut EGraph,
    root: EClassId,
    budget: usize,
    costs: &super::cost::CostModel,
) -> (usize, SaturationResult) {
    // Run budget-limited saturation
    let result = saturate_with_budget(egraph, budget);

    // Extract best cost
    let (_arena, _arena_root, cost) = egraph.extract_best(root, costs);

    (cost, result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::{CostModel, ENode, Rewrite, ops};
    use crate::math::algebra::{
        AddNeg, Annihilator, Cancellation, Canonicalize, Commutative, Identity,
        InverseAnnihilation, Involution, MulRecip,
    };

    /// Create an e-graph with standard algebraic rules for testing.
    fn egraph_with_rules() -> EGraph {
        let rules: Vec<Box<dyn Rewrite>> = vec![
            Canonicalize::<AddNeg>::new(),
            Involution::<AddNeg>::new(),
            Cancellation::<AddNeg>::new(),
            InverseAnnihilation::<AddNeg>::new(),
            Canonicalize::<MulRecip>::new(),
            Involution::<MulRecip>::new(),
            Cancellation::<MulRecip>::new(),
            InverseAnnihilation::<MulRecip>::new(),
            Commutative::new(&ops::Add),
            Commutative::new(&ops::Mul),
            Identity::new(&ops::Add),
            Identity::new(&ops::Mul),
            Annihilator::new(&ops::Mul),
        ];
        EGraph::with_rules(rules)
    }

    #[test]
    fn saturate_with_budget_simple() {
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let zero = eg.add(ENode::constant(0.0));
        let _sum = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, zero],
        });

        let result = saturate_with_budget(&mut eg, 10);

        // Should saturate quickly for simple expression
        assert!(result.iterations <= 10);
        assert!(result.classes_after >= result.classes_before);
    }

    #[test]
    fn saturate_with_budget_exhausted() {
        let mut eg = egraph_with_rules();
        // Create a moderately complex expression
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let mul = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![x, y],
        });
        let add = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![mul, x],
        });
        let _sub = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![add, y],
        });

        // Very small budget - may not saturate
        let result = saturate_with_budget(&mut eg, 1);

        assert_eq!(result.budget, 1);
        assert!(result.iterations <= 1);
    }

    #[test]
    fn achievable_cost() {
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let zero = eg.add(ENode::constant(0.0));
        let sum = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, zero],
        });

        let costs = CostModel::new();
        let (cost, result) = achievable_cost_within_budget(&mut eg, sum, 10, &costs);

        // x + 0 should simplify to x (cost 0)
        assert_eq!(cost, 0);
        assert!(result.saturated);
    }

    #[test]
    fn saturation_result_growth_ratio() {
        let result = SaturationResult {
            iterations: 5,
            total_unions: 10,
            saturated: true,
            classes_before: 10,
            classes_after: 15,
            rule_matches: HashMap::default(),
            budget: 100,
            max_classes: 10_000,
            hard_timeout: std::time::Duration::from_millis(500),
            stop: SaturationStop::Quiesced,
        };

        assert!((result.growth_ratio() - 1.5).abs() < 0.01);
    }
}
