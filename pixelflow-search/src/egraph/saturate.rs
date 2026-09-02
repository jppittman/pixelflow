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

use std::collections::HashMap;

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
#[derive(Clone, Copy, Debug)]
pub struct SaturationConfig {
    /// Rewrite-round budget (each round applies every rule once).
    pub max_iterations: usize,
    /// Wall-clock budget for the whole run.
    pub hard_timeout: std::time::Duration,
    /// LIVE e-class budget — canonical, non-empty classes: the population
    /// [`EGraph::class_ids`] enumerates, not the allocated slot count.
    ///
    /// There is deliberately no allocated-class field beside it: the
    /// allocated ceiling is the memory guard and every shipping tier holds
    /// it at [`HARD_CLASS_LIMIT`](super::graph::HARD_CLASS_LIMIT), so it is
    /// a constant here rather than a knob. A caller that needs to move it
    /// says so with [`Budget::Explicit`](super::optimizer::Budget::Explicit),
    /// which names both ceilings.
    pub max_classes: usize,
}

impl SaturationConfig {
    /// Trivial expressions (≤10 nodes): minimal budget.
    pub fn blitz() -> Self {
        Self {
            max_iterations: 20,
            hard_timeout: std::time::Duration::from_millis(10),
            max_classes: 500,
        }
    }

    /// Normal complexity (11-50 nodes): balanced.
    pub fn rapid() -> Self {
        Self {
            max_iterations: 50,
            hard_timeout: std::time::Duration::from_millis(50),
            max_classes: 2000,
        }
    }

    /// Complex expressions (51+ nodes): thorough search.
    pub fn classical() -> Self {
        Self {
            max_iterations: 100,
            hard_timeout: std::time::Duration::from_millis(200),
            max_classes: 5000,
        }
    }

    /// The pre-2026-09 fixed budget — 10,000 e-classes and 500 ms — with the
    /// caller's own round count.
    ///
    /// **Not a production preset.** Production sizes its budget from the
    /// expression via [`config_for_node_count`], reached through
    /// [`Optimizer::run`](super::optimizer::Optimizer::run).
    /// This one exists for the non-production call sites that must reproduce
    /// results from before that policy landed — unit tests, the hindsight
    /// labeler, and offline measurement harnesses — so the budget is named
    /// once here instead of the same two magic numbers being re-spelled at
    /// every such site, where they would drift apart the moment one of them
    /// was revised.
    ///
    /// The round count stays a parameter because it always was one: the
    /// budget these sites inherited fixed the class cap and the deadline but
    /// let each caller choose how many rewrite rounds it wanted. A site that
    /// needs to vary one of the other two should say so at the call site,
    /// with a reason:
    ///
    /// ```ignore
    /// SaturationConfig {
    ///     hard_timeout: SAFETY_CEILING, // offline: measuring caps, not the machine
    ///     ..SaturationConfig::compatibility(100)
    /// }
    /// ```
    pub fn compatibility(max_iterations: usize) -> Self {
        Self {
            max_iterations,
            hard_timeout: std::time::Duration::from_millis(500),
            max_classes: 10_000,
        }
    }

    /// Run one saturation of `egraph` under this budget.
    ///
    /// The three fields are exactly [`EGraph::saturate_with_limits`]'s three
    /// arguments, so this spares every caller from unpacking them — and from
    /// re-deriving the order.
    pub fn run(&self, egraph: &mut EGraph) -> super::graph::SaturationStats {
        egraph.saturate_with_limits(self.max_iterations, self.max_classes, self.hard_timeout)
    }
}

/// Pick a [`SaturationConfig`] preset from a rough expression-size measure.
///
/// | Nodes | Config | Rationale |
/// |-------|--------|-----------|
/// | 0-10 | blitz | Trivial expressions need minimal optimization |
/// | 11-50 | rapid | Normal complexity, balanced approach |
/// | 51+ | classical | Complex expressions need thorough search |
///
/// `node_count` is a proxy, not a precise measure — AST node count for the
/// macro tier, reachable-arena-node count for the runtime tier both serve
/// equally well as "how big is this expression".
pub fn config_for_node_count(node_count: usize) -> SaturationConfig {
    match node_count {
        0..=10 => SaturationConfig::blitz(),
        11..=50 => SaturationConfig::rapid(),
        _ => SaturationConfig::classical(),
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
            rule_matches: HashMap::new(),
            budget: 100,
            max_classes: 10_000,
            hard_timeout: std::time::Duration::from_millis(500),
            stop: SaturationStop::Quiesced,
        };

        assert!((result.growth_ratio() - 1.5).abs() < 0.01);
    }
}
