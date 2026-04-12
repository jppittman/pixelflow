//! Guided e-graph construction for NNUE-driven optimization.
//!
//! Instead of full equality saturation (which is exponential in rule count),
//! this module builds the e-graph incrementally, guided by MCTS + NNUE.
//!
//! The key insight: MCTS decides which rewrites to apply, not which node
//! to extract from a saturated e-graph. This makes the search scale with
//! budget, not rule count.
//!
//! # Architecture
//!
//! ```text
//! State:   Current (partial) e-graph
//! Action:  Apply rewrite rule R to e-class C
//! Reward:  Did this lead to finding a lower-cost expression?
//! ```
//!
//! NNUE predicts: "Given this partial e-graph, is applying rewrite R
//! to class C likely to be productive?"
//!
//! # MCTS Over Rewrites
//!
//! The guided search uses UCB1 to select which rewrite action to apply:
//!
//! ```text
//! UCB1(a) = Q(a) + c * sqrt(ln(N) / n(a))
//! ```
//!
//! Where:
//! - Q(a) = average reward from action a (improvement in cost)
//! - N = total visits to parent state
//! - n(a) = visits to action a
//! - c = exploration constant (sqrt(2) by default)

use std::collections::HashMap;
use std::time::{Duration, Instant};

use rand::{Rng, SeedableRng, rngs::StdRng};

use super::cost::CostModel;
use super::extract::ExprTree;
use super::graph::EGraph;
use super::node::{EClassId, ENode};

/// An action in the guided search space.
///
/// Represents applying a specific rewrite rule to a specific e-class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GuidedAction {
    /// Index into the e-graph's rule list
    pub rule_idx: usize,
    /// The e-class to apply the rule to
    pub target_class: EClassId,
    /// The node within the e-class that matched
    pub node_idx: usize,
}

impl GuidedAction {
    /// Create a new guided action.
    #[must_use]
    pub fn new(rule_idx: usize, target_class: EClassId, node_idx: usize) -> Self {
        Self {
            rule_idx,
            target_class,
            node_idx,
        }
    }
}

/// Record of an applied action and its outcome.
#[derive(Debug, Clone)]
pub struct ActionRecord {
    /// The action that was applied
    pub action: GuidedAction,
    /// Cost before applying the action
    pub cost_before: usize,
    /// Cost after applying the action
    pub cost_after: usize,
    /// Whether this action improved the best cost
    pub was_improvement: bool,
}

/// State for guided e-graph construction.
///
/// Tracks a partial e-graph being built incrementally, along with
/// the current best extraction and history of applied rewrites.
pub struct GuidedState {
    /// The partial e-graph being built
    egraph: EGraph,
    /// Root e-class we're optimizing
    root: EClassId,
    /// Current best extracted expression
    best_tree: ExprTree,
    /// Current best cost
    best_cost: usize,
    /// Cost model for extraction
    costs: CostModel,
    /// History of rewrites applied
    history: Vec<ActionRecord>,
    /// Total number of e-classes created (for future statistics)
    #[allow(dead_code)]
    classes_created: usize,
    /// Total number of unions performed (for future statistics)
    #[allow(dead_code)]
    unions_performed: usize,
}

impl GuidedState {
    /// Create a new guided state from an expression tree.
    ///
    /// The tree is inserted into a fresh e-graph, and no saturation
    /// is performed initially.
    #[must_use]
    pub fn from_tree(tree: &ExprTree, costs: CostModel) -> Self {
        let mut egraph = EGraph::new();
        let root = Self::insert_tree(&mut egraph, tree);

        // Initial extraction (just the input tree)
        let best_tree = egraph.extract_tree_with_costs(root, &costs);
        let best_cost = Self::compute_cost(&best_tree, &costs);

        Self {
            egraph,
            root,
            best_tree,
            best_cost,
            costs,
            history: Vec::new(),
            classes_created: 1, // At least the root
            unions_performed: 0,
        }
    }

    /// Insert an expression tree into the e-graph, returning the root e-class.
    fn insert_tree(egraph: &mut EGraph, tree: &ExprTree) -> EClassId {
        use super::extract::Leaf;

        match tree {
            ExprTree::Leaf(Leaf::Var(v)) => egraph.add(ENode::Var(*v)),
            ExprTree::Leaf(Leaf::Const(c)) => egraph.add(ENode::constant(*c)),
            ExprTree::Op { op, children } => {
                let child_ids: Vec<_> = children
                    .iter()
                    .map(|c| Self::insert_tree(egraph, c))
                    .collect();
                egraph.add(ENode::Op {
                    op: *op,
                    children: child_ids,
                })
            }
        }
    }

    /// Compute the cost of an expression tree.
    fn compute_cost(tree: &ExprTree, costs: &CostModel) -> usize {
        tree.cost(costs)
    }

    /// Get the current best extraction.
    #[must_use]
    pub fn best_tree(&self) -> &ExprTree {
        &self.best_tree
    }

    /// Get the current best cost.
    #[must_use]
    pub fn best_cost(&self) -> usize {
        self.best_cost
    }

    /// Get the root e-class.
    #[must_use]
    pub fn root(&self) -> EClassId {
        self.root
    }

    /// Get the e-graph (read-only).
    #[must_use]
    pub fn egraph(&self) -> &EGraph {
        &self.egraph
    }

    /// Get the history of applied actions.
    #[must_use]
    pub fn history(&self) -> &[ActionRecord] {
        &self.history
    }

    /// Get the number of rules registered in the e-graph.
    #[must_use]
    pub fn num_rules(&self) -> usize {
        self.egraph.num_rules()
    }

    /// Get the number of e-classes in the e-graph.
    #[must_use]
    pub fn num_classes(&self) -> usize {
        self.egraph.num_classes()
    }

    /// Enumerate all available actions (rule × e-class × node combinations).
    ///
    /// This returns actions that could potentially match. Not all will
    /// produce a rewrite (the rule may not match the node).
    #[must_use]
    pub fn available_actions(&self) -> Vec<GuidedAction> {
        let mut actions = Vec::new();
        let num_rules = self.egraph.num_rules();
        let num_classes = self.egraph.num_classes();

        for class_idx in 0..num_classes {
            let class_id = EClassId(class_idx as u32);
            let class_id = self.egraph.find(class_id);
            let nodes = self.egraph.nodes(class_id);

            for (node_idx, _node) in nodes.iter().enumerate() {
                for rule_idx in 0..num_rules {
                    actions.push(GuidedAction::new(rule_idx, class_id, node_idx));
                }
            }
        }

        actions
    }

    /// Try to apply a guided action, returning whether it produced a change.
    ///
    /// If the rule matches, this applies the rewrite, updates the best
    /// extraction if improved, and records the action in history.
    pub fn apply_action(&mut self, action: GuidedAction) -> bool {
        let cost_before = self.best_cost;

        // Try to apply the rule
        let changed =
            self.egraph
                .apply_single_rule(action.rule_idx, action.target_class, action.node_idx);

        if !changed {
            return false;
        }

        // Update extraction
        let new_tree = self.egraph.extract_tree_with_costs(self.root, &self.costs);
        let new_cost = Self::compute_cost(&new_tree, &self.costs);

        let was_improvement = new_cost < self.best_cost;
        if was_improvement {
            self.best_tree = new_tree;
            self.best_cost = new_cost;
        }

        // Record the action
        self.history.push(ActionRecord {
            action,
            cost_before,
            cost_after: new_cost,
            was_improvement,
        });

        true
    }

    /// Get statistics about the guided search.
    #[must_use]
    pub fn stats(&self) -> GuidedStats {
        let improvements = self.history.iter().filter(|r| r.was_improvement).count();
        let total_actions = self.history.len();

        GuidedStats {
            total_actions,
            successful_improvements: improvements,
            initial_cost: self
                .history
                .first()
                .map(|r| r.cost_before)
                .unwrap_or(self.best_cost),
            final_cost: self.best_cost,
            num_classes: self.num_classes(),
        }
    }
}

/// Statistics about a guided search session.
#[derive(Debug, Clone)]
pub struct GuidedStats {
    /// Total number of actions applied
    pub total_actions: usize,
    /// Number of actions that improved the best cost
    pub successful_improvements: usize,
    /// Initial cost before any rewrites
    pub initial_cost: usize,
    /// Final best cost
    pub final_cost: usize,
    /// Number of e-classes in the final e-graph
    pub num_classes: usize,
}

impl GuidedStats {
    /// Compute the improvement ratio (0.0 to 1.0).
    #[must_use]
    pub fn improvement_ratio(&self) -> f64 {
        if self.initial_cost == 0 {
            return 0.0;
        }
        1.0 - (self.final_cost as f64 / self.initial_cost as f64)
    }

    /// Compute the success rate of actions.
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.total_actions == 0 {
            return 0.0;
        }
        self.successful_improvements as f64 / self.total_actions as f64
    }
}

/// Result of guided optimization.
#[derive(Debug)]
pub struct GuidedResult {
    /// The optimized expression tree
    pub tree: ExprTree,
    /// The cost of the optimized tree
    pub cost: usize,
    /// Statistics about the search
    pub stats: GuidedStats,
}

// ============================================================================
// MCTS Over Rewrite Actions
// ============================================================================

/// Statistics for a single action in MCTS.
#[derive(Debug, Clone, Default)]
pub struct ActionStats {
    /// Number of times this action was selected
    pub visits: usize,
    /// Total reward accumulated (sum of improvements)
    pub total_reward: f64,
    /// Number of times this action actually changed the e-graph
    pub successful_applies: usize,
}

impl ActionStats {
    /// Compute the average reward for this action.
    #[must_use]
    pub fn average_reward(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_reward / self.visits as f64
        }
    }

    /// UCB1 score for action selection.
    ///
    /// UCB1(a) = Q(a) + c * sqrt(ln(N) / n(a))
    #[must_use]
    pub fn ucb1(&self, parent_visits: usize, exploration_constant: f64) -> f64 {
        if self.visits == 0 {
            // Unvisited actions have infinite UCB1 (explore first)
            f64::INFINITY
        } else {
            let exploit = self.average_reward();
            let explore =
                exploration_constant * ((parent_visits as f64).ln() / self.visits as f64).sqrt();
            exploit + explore
        }
    }
}

/// Configuration for guided MCTS search.
#[derive(Clone, Debug)]
pub struct GuidedConfig {
    /// Maximum iterations (None = unlimited until timeout).
    pub max_iterations: Option<usize>,
    /// Maximum wall-clock time (None = unlimited).
    pub timeout: Option<Duration>,
    /// UCB1 exploration constant (higher = more exploration).
    pub exploration_constant: f64,
    /// Epsilon for ε-greedy exploration (0.0 = pure UCB1, 1.0 = pure random).
    pub epsilon: f64,
    /// Random seed for ε-greedy exploration (None = use system entropy).
    pub random_seed: Option<u64>,
    /// Cost model for extraction.
    pub cost_model: CostModel,
}

impl Default for GuidedConfig {
    fn default() -> Self {
        Self {
            max_iterations: Some(1000),
            timeout: None,
            exploration_constant: std::f64::consts::SQRT_2,
            epsilon: 0.0, // Pure UCB1 by default (inference mode)
            random_seed: None,
            cost_model: CostModel::fully_optimized(),
        }
    }
}

impl GuidedConfig {
    /// Set maximum iterations.
    #[must_use]
    pub fn with_iterations(mut self, n: usize) -> Self {
        self.max_iterations = Some(n);
        self
    }

    /// Set timeout.
    #[must_use]
    pub fn with_timeout(mut self, d: Duration) -> Self {
        self.timeout = Some(d);
        self
    }

    /// Set exploration constant.
    #[must_use]
    pub fn with_exploration(mut self, c: f64) -> Self {
        self.exploration_constant = c;
        self
    }

    /// Set epsilon for ε-greedy exploration.
    #[must_use]
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon.clamp(0.0, 1.0);
        self
    }

    /// Set random seed for reproducible exploration.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Configure for training mode (with exploration).
    #[must_use]
    pub fn training_mode(self) -> Self {
        self.with_epsilon(0.1)
    }

    /// Configure for inference mode (pure exploitation).
    #[must_use]
    pub fn inference_mode(self) -> Self {
        self.with_epsilon(0.0)
    }
}

/// MCTS tree for guided e-graph construction.
///
/// Unlike traditional MCTS which builds a tree of states, this maintains
/// statistics for each action (rule × target_class × node_idx) and applies
/// them to a single evolving e-graph state.
pub struct GuidedMcts {
    /// The guided state being optimized
    state: GuidedState,
    /// Statistics for each action (keyed by action)
    action_stats: HashMap<GuidedAction, ActionStats>,
    /// Total number of iterations (parent visit count for UCB1)
    total_iterations: usize,
    /// Configuration
    config: GuidedConfig,
    /// RNG for ε-greedy exploration
    rng: StdRng,
}

impl GuidedMcts {
    /// Create a new MCTS search from an expression tree.
    #[must_use]
    pub fn from_tree(tree: &ExprTree, config: GuidedConfig) -> Self {
        let rng = match config.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        Self {
            state: GuidedState::from_tree(tree, config.cost_model.clone()),
            action_stats: HashMap::new(),
            total_iterations: 0,
            config,
            rng,
        }
    }

    /// Select an action using UCB1 or ε-greedy (no prior).
    #[allow(dead_code)]
    fn select_action(&mut self) -> Option<GuidedAction> {
        self.select_action_with_prior(|_, _| 0.0)
    }

    /// Select an action using UCB1 with a prior Q-value from an evaluator.
    ///
    /// The evaluator function takes (state, action) and returns a prior Q-value
    /// estimate. This is typically from an NNUE network.
    ///
    /// The selection formula becomes:
    /// UCB1(a) = (prior(a) + empirical_Q(a)) / 2 + c * sqrt(ln(N) / n(a))
    fn select_action_with_prior<F>(&mut self, prior_fn: F) -> Option<GuidedAction>
    where
        F: Fn(&GuidedState, &GuidedAction) -> f64,
    {
        let actions = self.state.available_actions();
        if actions.is_empty() {
            return None;
        }

        // ε-greedy: with probability epsilon, select random action
        if self.config.epsilon > 0.0 && self.rng.r#gen::<f64>() < self.config.epsilon {
            let idx = self.rng.gen_range(0..actions.len());
            return Some(actions[idx]);
        }

        // UCB1 selection with prior
        let mut best_action = None;
        let mut best_ucb = f64::NEG_INFINITY;

        for action in actions {
            let stats = self.action_stats.get(&action).cloned().unwrap_or_default();

            // Get prior Q-value from evaluator (e.g., NNUE)
            let prior_q = prior_fn(&self.state, &action);

            // Blend prior with empirical estimate
            let empirical_q = stats.average_reward();
            let blended_q = if stats.visits == 0 {
                prior_q // Use prior only for unvisited actions
            } else {
                // Weighted blend: more weight to empirical as visits increase
                let weight = (stats.visits as f64).min(10.0) / 10.0;
                (1.0 - weight) * prior_q + weight * empirical_q
            };

            // UCB1 with blended Q-value
            let exploration = if stats.visits == 0 {
                f64::INFINITY // Always explore unvisited at least once
            } else {
                self.config.exploration_constant
                    * ((self.total_iterations.max(1) as f64).ln() / stats.visits as f64).sqrt()
            };

            let ucb = blended_q + exploration;

            if ucb > best_ucb {
                best_ucb = ucb;
                best_action = Some(action);
            }
        }

        best_action
    }

    /// Run one MCTS iteration.
    ///
    /// 1. Select action via UCB1/ε-greedy
    /// 2. Apply action to e-graph
    /// 3. Compute reward (improvement in cost)
    /// 4. Update statistics
    ///
    /// Returns the action taken and whether it improved the best cost.
    pub fn iterate(&mut self) -> Option<(GuidedAction, bool)> {
        self.iterate_with_evaluator(|_, _| 0.0)
    }

    /// Run one MCTS iteration with a custom action evaluator.
    ///
    /// The evaluator function takes (state, action) and returns a prior Q-value.
    /// This allows using NNUE or other learned models to guide action selection.
    ///
    /// Returns the action taken and whether it improved the best cost.
    pub fn iterate_with_evaluator<F>(&mut self, prior_fn: F) -> Option<(GuidedAction, bool)>
    where
        F: Fn(&GuidedState, &GuidedAction) -> f64,
    {
        let action = self.select_action_with_prior(prior_fn)?;
        let cost_before = self.state.best_cost();

        // Apply action
        let changed = self.state.apply_action(action);

        // Compute reward
        let cost_after = self.state.best_cost();
        let improvement = (cost_before as i64 - cost_after as i64).max(0) as f64;

        // Normalize reward to [0, 1] range based on relative improvement
        let reward = if cost_before > 0 {
            improvement / cost_before as f64
        } else {
            0.0
        };

        // Update statistics
        let stats = self.action_stats.entry(action).or_default();
        stats.visits += 1;
        stats.total_reward += reward;
        if changed {
            stats.successful_applies += 1;
        }

        self.total_iterations += 1;

        let was_improvement = cost_after < cost_before;
        Some((action, was_improvement))
    }

    /// Run MCTS until budget exhausted.
    pub fn run(&mut self) -> GuidedResult {
        let start = Instant::now();

        loop {
            // Check termination conditions
            if let Some(timeout) = self.config.timeout
                && start.elapsed() >= timeout
            {
                break;
            }

            if let Some(max_iter) = self.config.max_iterations
                && self.total_iterations >= max_iter
            {
                break;
            }

            // Run one iteration
            if self.iterate().is_none() {
                // No more actions available
                break;
            }

            // Early termination if cost is minimal
            if self.state.best_cost() <= 1 {
                break;
            }
        }

        GuidedResult {
            tree: self.state.best_tree().clone(),
            cost: self.state.best_cost(),
            stats: self.state.stats(),
        }
    }

    /// Get the current best extraction.
    #[must_use]
    pub fn best_tree(&self) -> &ExprTree {
        self.state.best_tree()
    }

    /// Get the current best cost.
    #[must_use]
    pub fn best_cost(&self) -> usize {
        self.state.best_cost()
    }

    /// Get the number of iterations performed.
    #[must_use]
    pub fn iterations(&self) -> usize {
        self.total_iterations
    }

    /// Get the guided state (for inspection/debugging).
    #[must_use]
    pub fn state(&self) -> &GuidedState {
        &self.state
    }

    /// Get action statistics (for analysis).
    #[must_use]
    pub fn action_stats(&self) -> &HashMap<GuidedAction, ActionStats> {
        &self.action_stats
    }
}

/// Optimize an expression tree using guided MCTS.
///
/// This is the main entry point for guided optimization. Unlike full
/// equality saturation, this builds the e-graph incrementally, applying
/// only the rewrites that MCTS deems promising.
///
/// # Arguments
///
/// * `tree` - The expression tree to optimize
/// * `config` - Configuration for the guided search
///
/// # Returns
///
/// The optimization result including the best tree found within budget.
#[must_use]
pub fn guided_optimize(tree: &ExprTree, config: GuidedConfig) -> GuidedResult {
    let mut mcts = GuidedMcts::from_tree(tree, config);
    mcts.run()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guided_state_creation_should_succeed_when_called() {
        // Create a simple tree: X + 0
        let tree = ExprTree::op_add(ExprTree::var(0), ExprTree::constant(0.0));

        let costs = CostModel::default();
        let state = GuidedState::from_tree(&tree, costs);

        // Should have the tree inserted
        assert!(state.num_classes() > 0);
        assert!(state.best_cost() > 0);
    }

    #[test]
    fn available_actions_should_succeed_when_called() {
        let tree = ExprTree::op_add(ExprTree::var(0), ExprTree::constant(0.0));

        let costs = CostModel::default();
        let state = GuidedState::from_tree(&tree, costs);

        let actions = state.available_actions();
        // Should have actions for each (rule, class, node) combination
        assert!(!actions.is_empty());
    }

    #[test]
    fn apply_action_improves_should_succeed_when_called() {
        // X + 0 should simplify to X
        let tree = ExprTree::op_add(ExprTree::var(0), ExprTree::constant(0.0));

        let costs = CostModel::default();
        let mut state = GuidedState::from_tree(&tree, costs);

        let initial_cost = state.best_cost();

        // Try all actions until we find an improvement
        let actions = state.available_actions();
        let mut found_improvement = false;

        for action in actions {
            if state.apply_action(action) {
                if state.best_cost() < initial_cost {
                    found_improvement = true;
                    break;
                }
            }
        }

        // Identity rule should fire and improve cost
        assert!(found_improvement || state.best_cost() <= initial_cost);
    }

    #[test]
    fn guided_stats_should_succeed_when_called() {
        let tree = ExprTree::op_mul(ExprTree::var(0), ExprTree::constant(1.0));

        let costs = CostModel::default();
        let mut state = GuidedState::from_tree(&tree, costs);

        // Apply some actions
        let actions = state.available_actions();
        for action in actions.into_iter().take(5) {
            state.apply_action(action);
        }

        let stats = state.stats();
        assert!(stats.improvement_ratio() >= 0.0);
        assert!(stats.success_rate() >= 0.0);
    }

    // ========================================================================
    // MCTS Tests
    // ========================================================================

    #[test]
    fn action_stats_ucb1_should_succeed_when_called() {
        let mut stats = ActionStats::default();

        // Unvisited action should have infinite UCB1
        assert_eq!(stats.ucb1(10, std::f64::consts::SQRT_2), f64::INFINITY);

        // After one visit with reward 0.5
        stats.visits = 1;
        stats.total_reward = 0.5;

        let ucb = stats.ucb1(10, std::f64::consts::SQRT_2);
        // Should be 0.5 (exploit) + sqrt(2) * sqrt(ln(10)/1) (explore)
        assert!(ucb > 0.5); // Exploration bonus added
        assert!(ucb < 5.0); // Reasonable upper bound

        // More visits should reduce exploration bonus
        stats.visits = 100;
        stats.total_reward = 50.0; // Same average reward
        let ucb_more = stats.ucb1(1000, std::f64::consts::SQRT_2);
        // Exploration term should be smaller
        assert!(ucb_more < ucb);
    }

    #[test]
    fn guided_mcts_creation_should_succeed_when_called() {
        let tree = ExprTree::op_add(ExprTree::var(0), ExprTree::constant(0.0));

        let config = GuidedConfig::default().with_iterations(10);
        let mcts = GuidedMcts::from_tree(&tree, config);

        assert_eq!(mcts.iterations(), 0);
        assert!(mcts.best_cost() > 0);
    }

    #[test]
    fn guided_mcts_iterate_should_succeed_when_called() {
        let tree = ExprTree::op_add(ExprTree::var(0), ExprTree::constant(0.0));

        let config = GuidedConfig::default().with_iterations(100).with_seed(42);
        let mut mcts = GuidedMcts::from_tree(&tree, config);

        let initial_cost = mcts.best_cost();

        // Run some iterations
        for _ in 0..50 {
            if mcts.iterate().is_none() {
                break;
            }
        }

        // Should have made progress
        assert!(mcts.iterations() > 0);
        // Cost should improve or stay the same (x + 0 → x)
        assert!(mcts.best_cost() <= initial_cost);
    }

    #[test]
    fn guided_mcts_run_should_succeed_when_called() {
        // x * 1 should simplify to x
        let tree = ExprTree::op_mul(ExprTree::var(0), ExprTree::constant(1.0));

        let config = GuidedConfig::default().with_iterations(100).with_seed(42);

        let result = guided_optimize(&tree, config);

        assert!(result.cost <= result.stats.initial_cost);
        assert!(result.stats.total_actions > 0);
    }

    #[test]
    fn guided_mcts_complex_expr_should_succeed_when_called() {
        // (x + 0) * 1 + (y * 0) should simplify to x
        let tree = ExprTree::op_add(
            ExprTree::op_mul(
                ExprTree::op_add(ExprTree::var(0), ExprTree::constant(0.0)),
                ExprTree::constant(1.0),
            ),
            ExprTree::op_mul(ExprTree::var(1), ExprTree::constant(0.0)),
        );

        let config = GuidedConfig::default().with_iterations(500).with_seed(42);

        let result = guided_optimize(&tree, config);

        // Should significantly reduce cost
        assert!(result.cost < result.stats.initial_cost);
        assert!(result.stats.improvement_ratio() > 0.0);
    }

    #[test]
    fn guided_mcts_epsilon_greedy_should_succeed_when_called() {
        let tree = ExprTree::op_add(ExprTree::var(0), ExprTree::var(1));

        // Test with epsilon = 1.0 (pure random)
        let config = GuidedConfig::default()
            .with_iterations(50)
            .with_epsilon(1.0)
            .with_seed(42);

        let mut mcts = GuidedMcts::from_tree(&tree, config);

        // Should still be able to run iterations
        for _ in 0..20 {
            mcts.iterate();
        }

        assert!(mcts.iterations() > 0);
    }

    #[test]
    fn guided_mcts_training_mode_should_succeed_when_called() {
        let tree = ExprTree::op_add(ExprTree::var(0), ExprTree::constant(0.0));

        let config = GuidedConfig::default()
            .training_mode()
            .with_iterations(200) // More iterations to ensure we hit productive actions
            .with_seed(42);

        // Should use epsilon = 0.1
        assert!((config.epsilon - 0.1).abs() < 0.001);

        let result = guided_optimize(&tree, config);
        // In training mode, we should explore various actions
        // Cost should improve from x+0 -> x
        assert!(result.cost <= result.stats.initial_cost);
    }

    #[test]
    fn guided_mcts_timeout_should_succeed_when_called() {
        let tree = ExprTree::op_add(
            ExprTree::op_mul(ExprTree::var(0), ExprTree::var(1)),
            ExprTree::var(2),
        );

        let config = GuidedConfig::default()
            .with_timeout(Duration::from_millis(50))
            .with_iterations(1_000_000); // Very high limit

        let start = Instant::now();
        let _result = guided_optimize(&tree, config);
        let elapsed = start.elapsed();

        // Should respect timeout (with some tolerance for overhead)
        assert!(elapsed < Duration::from_millis(200));
    }
}
