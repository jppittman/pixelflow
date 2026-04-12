//! Best-First Search for E-Graph Optimization
//!
//! This is just Dijkstra/A* with NNUE as the heuristic.
//!
//! ## The Graph Search Mental Model
//!
//! | ML/MCTS Name | Graph/Dijkstra Name |
//! |--------------|---------------------|
//! | State | Node in search graph |
//! | Rewrite | Edge |
//! | NNUE | Heuristic (estimated distance to optimal) |
//! | Objective | Minimize cost (runtime of kernel) |
//!
//! ## Why This is Simple
//!
//! 1. **No Policy Head**: Just score the result of each move
//! 2. **No Rollouts**: NNUE already knows that `a*(b+c)` beats `a*b+a*c`
//! 3. **Easy Debugging**: Print the path, see why NNUE said cost was X

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::hash::{Hash, Hasher};

use super::{CostModel, EClassId, EGraph, ExprTree};

// ============================================================================
// Search State Context (passed to evaluator)
// ============================================================================

/// Search state context passed to the NNUE evaluator.
///
/// This provides all the information needed for feature extraction:
/// - The expression tree itself (for structural features)
/// - Search progress metrics (for temporal features)
///
/// The evaluator can use these to learn patterns like:
/// - "Early in search, explore more aggressively"
/// - "When frontier is large, prefer simpler trees"
/// - "When improvement has stalled, try different rewrites"
#[derive(Clone, Debug)]
pub struct BestFirstContext<'a> {
    /// The expression tree being evaluated.
    pub tree: &'a ExprTree,
    /// Cost of this tree (ground truth from CostModel).
    pub tree_cost: usize,
    /// Depth in the search tree (0 = root).
    pub depth: usize,
    /// Current best cost found anywhere in search.
    pub best_cost: usize,
    /// Number of expansions performed so far.
    pub expansions: usize,
    /// Number of states in the frontier (open set).
    pub frontier_size: usize,
    /// Number of visited states (closed set).
    pub visited_count: usize,
    /// Initial cost before any optimization.
    pub initial_cost: usize,
}

impl<'a> BestFirstContext<'a> {
    /// Improvement ratio: how much better is current best vs initial?
    /// Returns 0.0 (no improvement) to 1.0 (perfect optimization).
    #[must_use]
    pub fn improvement_ratio(&self) -> f64 {
        if self.initial_cost == 0 {
            0.0
        } else {
            1.0 - (self.best_cost as f64 / self.initial_cost as f64)
        }
    }

    /// Search progress: fraction of max_expansions used.
    #[must_use]
    pub fn search_progress(&self, max_expansions: usize) -> f64 {
        self.expansions as f64 / max_expansions.max(1) as f64
    }

    /// Frontier pressure: how crowded is the open set?
    #[must_use]
    pub fn frontier_pressure(&self, max_queue_size: usize) -> f64 {
        self.frontier_size as f64 / max_queue_size.max(1) as f64
    }
}

/// A search state: an e-graph at some point in the search.
#[derive(Clone)]
struct SearchState {
    /// The e-graph state (cloneable via Arc-shared rules)
    egraph: EGraph,
    /// Root e-class being optimized
    root: EClassId,
    /// Current best extracted tree
    best_tree: ExprTree,
    /// Current best cost (ground truth from CostModel)
    best_cost: usize,
    /// NNUE-predicted potential (lower = more promising)
    priority: i64,
    /// Depth in search tree (for tie-breaking)
    depth: usize,
    /// Hash for cycle detection
    state_hash: u64,
}

impl SearchState {
    /// Create initial state from an expression tree.
    fn new(tree: &ExprTree, costs: &CostModel) -> Self {
        let mut egraph = EGraph::new();
        let root = egraph.add_expr(tree);
        let (best_tree, best_cost) = egraph.extract_best(root, costs);
        let state_hash = Self::compute_hash(&egraph, root);

        Self {
            egraph,
            root,
            best_tree,
            best_cost,
            priority: best_cost as i64,
            depth: 0,
            state_hash,
        }
    }

    /// Expand this state by applying all rules once.
    fn expand(&self, costs: &CostModel) -> Option<SearchState> {
        let mut child_egraph = self.egraph.clone();
        let changes = child_egraph.apply_rules_once();

        if changes == 0 {
            return None; // No new states reachable
        }

        let (tree, cost) = child_egraph.extract_best(self.root, costs);
        let state_hash = Self::compute_hash(&child_egraph, self.root);

        Some(SearchState {
            egraph: child_egraph,
            root: self.root,
            best_tree: tree,
            best_cost: cost,
            priority: cost as i64, // Will be overwritten by evaluator
            depth: self.depth + 1,
            state_hash,
        })
    }

    /// Compute a hash for cycle detection.
    ///
    /// Uses the number of classes and nodes as a rough fingerprint.
    /// More sophisticated hashing could use the actual graph structure.
    fn compute_hash(egraph: &EGraph, root: EClassId) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        egraph.num_classes().hash(&mut hasher);
        egraph.node_count().hash(&mut hasher);
        root.hash(&mut hasher);
        hasher.finish()
    }
}

impl PartialEq for SearchState {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.depth == other.depth
    }
}

impl Eq for SearchState {}

impl PartialOrd for SearchState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchState {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower priority = better, so reverse the comparison
        // Tie-break by depth (prefer shallower for faster solutions)
        other
            .priority
            .cmp(&self.priority)
            .then_with(|| other.depth.cmp(&self.depth))
    }
}

/// Configuration for best-first search.
#[derive(Clone, Debug)]
pub struct BestFirstConfig {
    /// Maximum states to expand before stopping.
    pub max_expansions: usize,
    /// Maximum queue size (prune worst if exceeded).
    pub max_queue_size: usize,
    /// Epsilon for ε-greedy exploration (0.0 = pure greedy, 1.0 = random).
    pub epsilon: f64,
    /// Node count threshold below which we just saturate.
    pub saturation_threshold: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for BestFirstConfig {
    fn default() -> Self {
        Self {
            max_expansions: 1000,
            max_queue_size: 10000,
            epsilon: 0.1,
            saturation_threshold: 100,
            seed: 42,
        }
    }
}

impl BestFirstConfig {
    /// Set epsilon for exploration.
    #[must_use]
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set maximum expansions.
    #[must_use]
    pub fn with_max_expansions(mut self, max: usize) -> Self {
        self.max_expansions = max;
        self
    }

    /// Set saturation threshold.
    #[must_use]
    pub fn with_saturation_threshold(mut self, threshold: usize) -> Self {
        self.saturation_threshold = threshold;
        self
    }

    /// Training mode: higher exploration.
    #[must_use]
    pub fn training_mode(mut self) -> Self {
        self.epsilon = 0.2;
        self
    }

    /// Inference mode: pure greedy.
    #[must_use]
    pub fn inference_mode(mut self) -> Self {
        self.epsilon = 0.0;
        self
    }
}

/// Result of best-first search.
#[derive(Clone, Debug)]
pub struct BestFirstResult {
    /// The best expression tree found.
    pub best_tree: ExprTree,
    /// Cost of the best tree.
    pub best_cost: usize,
    /// Number of states expanded.
    pub expansions: usize,
    /// Whether saturation was used (for small kernels).
    pub used_saturation: bool,
    /// Initial cost before optimization.
    pub initial_cost: usize,
}

/// Simple LCG random number generator.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn gen_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 33) as f64 / (1u64 << 31) as f64
    }

    fn gen_range(&mut self, max: usize) -> usize {
        (self.gen_f64() * max as f64) as usize
    }
}

/// Best-first search for e-graph optimization.
///
/// This is Dijkstra/A* where NNUE provides the heuristic.
/// Much simpler than MCTS - just a priority queue.
pub struct BestFirstPlanner {
    /// Open set: states to explore, ordered by priority (lower = better)
    open_set: BinaryHeap<SearchState>,
    /// Closed set: visited state hashes (cycle detection)
    visited: HashSet<u64>,
    /// Best state found so far
    best_found: SearchState,
    /// Configuration
    config: BestFirstConfig,
    /// Cost model for extraction
    costs: CostModel,
    /// RNG for epsilon-greedy
    rng: Rng,
    /// Number of expansions performed
    expansions: usize,
    /// Initial cost (for computing improvement)
    initial_cost: usize,
}

impl BestFirstPlanner {
    /// Create a new planner from an expression tree.
    #[must_use]
    pub fn from_tree(tree: &ExprTree, config: BestFirstConfig) -> Self {
        let costs = CostModel::default();
        let initial = SearchState::new(tree, &costs);
        let initial_cost = initial.best_cost;
        let seed = config.seed;

        let mut open_set = BinaryHeap::new();
        let mut visited = HashSet::new();

        visited.insert(initial.state_hash);
        open_set.push(initial.clone());

        Self {
            open_set,
            visited,
            best_found: initial,
            config,
            costs,
            rng: Rng::new(seed),
            expansions: 0,
            initial_cost,
        }
    }

    /// Check if the kernel is small enough to just saturate.
    #[must_use]
    pub fn should_saturate(&self) -> bool {
        self.best_found.egraph.node_count() < self.config.saturation_threshold
    }

    /// Run search with a heuristic evaluator function.
    ///
    /// The evaluator receives a `BestFirstContext` containing:
    /// - The expression tree to evaluate
    /// - Search progress metrics (depth, expansions, frontier size, etc.)
    ///
    /// Returns a priority score (lower = more promising).
    /// This is where NNUE integration happens.
    pub fn run<F>(&mut self, mut evaluator: F) -> BestFirstResult
    where
        F: FnMut(BestFirstContext<'_>) -> i64,
    {
        // Small kernel? Just saturate.
        if self.should_saturate() {
            return self.run_saturation();
        }

        // Best-first search with ε-greedy exploration
        while self.expansions < self.config.max_expansions && !self.open_set.is_empty() {
            // ε-greedy: occasionally pick random instead of best
            let current = if self.rng.gen_f64() < self.config.epsilon && self.open_set.len() > 1 {
                self.pick_random()
            } else {
                self.open_set.pop().unwrap()
            };

            // Expand: apply one round of rewrites
            if let Some(mut neighbor) = current.expand(&self.costs) {
                // Skip if already visited (cycle detection)
                if self.visited.contains(&neighbor.state_hash) {
                    continue;
                }

                // Build context for evaluator
                let context = BestFirstContext {
                    tree: &neighbor.best_tree,
                    tree_cost: neighbor.best_cost,
                    depth: neighbor.depth,
                    best_cost: self.best_found.best_cost,
                    expansions: self.expansions,
                    frontier_size: self.open_set.len(),
                    visited_count: self.visited.len(),
                    initial_cost: self.initial_cost,
                };

                // Score with NNUE heuristic
                neighbor.priority = evaluator(context);

                // Update global best if improved
                if neighbor.best_cost < self.best_found.best_cost {
                    self.best_found = neighbor.clone();
                }

                // Add to frontier
                self.visited.insert(neighbor.state_hash);
                self.open_set.push(neighbor);
            }

            self.expansions += 1;

            // Prune queue if too large
            if self.open_set.len() > self.config.max_queue_size {
                self.prune_queue();
            }
        }

        BestFirstResult {
            best_tree: self.best_found.best_tree.clone(),
            best_cost: self.best_found.best_cost,
            expansions: self.expansions,
            used_saturation: false,
            initial_cost: self.initial_cost,
        }
    }

    /// Run with default evaluator (just uses CostModel).
    pub fn run_default(&mut self) -> BestFirstResult {
        self.run(|ctx| {
            // Simple heuristic: tree depth + node count
            (ctx.tree.depth() * 10 + ctx.tree.node_count()) as i64
        })
    }

    /// Saturate the e-graph completely (for small kernels).
    fn run_saturation(&mut self) -> BestFirstResult {
        self.best_found.egraph.saturate();
        let (tree, cost) = self
            .best_found
            .egraph
            .extract_best(self.best_found.root, &self.costs);

        BestFirstResult {
            best_tree: tree,
            best_cost: cost,
            expansions: 0,
            used_saturation: true,
            initial_cost: self.initial_cost,
        }
    }

    /// Pick a random state from the queue.
    fn pick_random(&mut self) -> SearchState {
        let mut all: Vec<_> = std::iter::from_fn(|| self.open_set.pop()).collect();
        let idx = self.rng.gen_range(all.len());
        let picked = all.remove(idx);
        for s in all {
            self.open_set.push(s);
        }
        picked
    }

    /// Prune the queue to max size by removing worst states.
    fn prune_queue(&mut self) {
        let mut all: Vec<_> = std::iter::from_fn(|| self.open_set.pop()).collect();
        all.truncate(self.config.max_queue_size / 2);
        for s in all {
            self.open_set.push(s);
        }
    }

    /// Get current best cost.
    #[must_use]
    pub fn best_cost(&self) -> usize {
        self.best_found.best_cost
    }

    /// Get current best tree.
    #[must_use]
    pub fn best_tree(&self) -> &ExprTree {
        &self.best_found.best_tree
    }

    /// Get number of expansions performed.
    #[must_use]
    pub fn expansions(&self) -> usize {
        self.expansions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::{Leaf, ops};

    #[test]
    fn small_kernel_saturates_should_succeed_when_called() {
        // x + 0 should simplify to x
        let tree = ExprTree::Op {
            op: &ops::Add,
            children: vec![
                ExprTree::Leaf(Leaf::Var(0)),
                ExprTree::Leaf(Leaf::Const(0.0)),
            ],
        };

        let config = BestFirstConfig::default().with_saturation_threshold(100);

        let mut planner = BestFirstPlanner::from_tree(&tree, config);
        let result = planner.run_default();

        assert!(result.used_saturation);
        // Should simplify to just Var(0)
        assert!(matches!(result.best_tree, ExprTree::Leaf(Leaf::Var(0))));
    }

    #[test]
    fn epsilon_greedy_should_succeed_when_called() {
        let tree = ExprTree::Op {
            op: &ops::Mul,
            children: vec![
                ExprTree::Leaf(Leaf::Var(0)),
                ExprTree::Leaf(Leaf::Const(1.0)),
            ],
        };

        // High epsilon = lots of exploration
        let config = BestFirstConfig::default()
            .with_epsilon(0.5)
            .with_max_expansions(10);

        let mut planner = BestFirstPlanner::from_tree(&tree, config);
        let result = planner.run_default();

        // Should still find the optimum
        assert!(result.best_cost <= 2); // Var(0) has cost ~1
    }

    #[test]
    fn egraph_is_clone_should_succeed_when_called() {
        // Verify EGraph can be cloned (needed for search branching)
        let mut eg = EGraph::new();
        let x = eg.add_expr(&ExprTree::Leaf(Leaf::Var(0)));
        let _cloned = eg.clone();
        assert!(eg.num_classes() > 0);
        let _ = x; // silence warning
    }
}
