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

use super::graph::EGraph;
use super::node::EClassId;

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

    /// Whether saturation completed (no more changes) before budget exhausted.
    pub saturated: bool,

    /// Number of e-classes before saturation.
    pub classes_before: usize,

    /// Number of e-classes after saturation.
    pub classes_after: usize,

    /// Rule match counts by rule name.
    pub rule_matches: HashMap<String, usize>,

    /// The rewrite budget that was used.
    pub budget: usize,
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

    let start = std::time::Instant::now();
    let mut iterations = 0;
    let mut total_unions = 0;

    for _ in 0..max_iterations {
        // Global time limit
        if start.elapsed() >= timeout {
            break;
        }
        // Global class limit
        if egraph.classes.len() > max_classes {
            break;
        }

        iterations += 1;
        let remaining = timeout.saturating_sub(start.elapsed());
        let unions = apply_rules_counted_with_limits(egraph, max_classes, remaining);
        total_unions += unions;

        if unions == 0 {
            break;
        }
    }

    let saturated = iterations < max_iterations || total_unions == 0;
    let classes_after = egraph.classes.len();
    let rule_matches = egraph.match_counts.clone();

    SaturationResult {
        iterations,
        total_unions,
        saturated,
        classes_before,
        classes_after,
        rule_matches,
        budget: max_iterations,
    }
}

/// Apply all rules once and count new classes, with safety limits.
fn apply_rules_counted_with_limits(
    egraph: &mut EGraph,
    max_classes: usize,
    timeout: std::time::Duration,
) -> usize {
    let classes_before = egraph.classes.len();
    egraph.saturate_with_limits(1, max_classes, timeout);
    let classes_after = egraph.classes.len();
    classes_after.saturating_sub(classes_before)
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
        };

        assert!((result.growth_ratio() - 1.5).abs() < 0.01);
    }
}
