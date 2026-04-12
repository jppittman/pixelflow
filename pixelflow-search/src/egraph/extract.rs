//! Extraction: Pull out a concrete expression tree from an e-graph.
//!
//! An e-graph compresses many equivalent expressions. Extraction picks
//! the "best" one according to a cost model.

use alloc::vec::Vec;
use super::cost::{CostFunction, CostModel};
use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::ops::Op;

/// A concrete expression tree extracted from an e-graph.
///
/// Unlike ENode (which uses EClassId children), this has direct child ownership.
/// This is the output of extraction - a single concrete expression from the e-class.
#[derive(Clone, Debug)]
pub enum ExprTree {
    /// Leaf nodes
    Leaf(Leaf),
    /// Operation with child subtrees
    Op {
        op: &'static dyn Op,
        children: Vec<ExprTree>,
    },
}

/// Leaf nodes in an expression tree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Leaf {
    /// Variable with index (0=X, 1=Y, 2=Z, 3=W, etc.)
    Var(u8),
    /// Constant value
    Const(f32),
}

impl ExprTree {
    /// Create a variable.
    pub fn var(idx: u8) -> Self {
        Self::Leaf(Leaf::Var(idx))
    }

    /// Create a constant.
    pub fn constant(val: f32) -> Self {
        Self::Leaf(Leaf::Const(val))
    }

    /// Count total nodes in the tree.
    ///
    /// Uses iterative traversal to avoid stack overflow on deep trees.
    pub fn node_count(&self) -> usize {
        let mut count = 0usize;
        let mut stack: Vec<&ExprTree> = vec![self];
        while let Some(node) = stack.pop() {
            count += 1;
            if let Self::Op { children, .. } = node {
                for child in children {
                    stack.push(child);
                }
            }
        }
        count
    }

    /// Compute depth of the tree.
    ///
    /// Uses iterative traversal to avoid stack overflow on deep trees.
    pub fn depth(&self) -> usize {
        // Stack stores (node, depth_at_node)
        let mut stack: Vec<(&ExprTree, usize)> = vec![(self, 1)];
        let mut max_depth = 0usize;
        while let Some((node, d)) = stack.pop() {
            match node {
                Self::Leaf(_) => {
                    if d > max_depth {
                        max_depth = d;
                    }
                }
                Self::Op { children, .. } => {
                    for child in children {
                        stack.push((child, d + 1));
                    }
                }
            }
        }
        max_depth
    }

    /// Check if this expression has valid types.
    ///
    /// Returns `false` if comparison results are used in invalid contexts
    /// (e.g., as operands to arithmetic operations).
    ///
    /// Type rules:
    /// - Comparison ops (lt, le, gt, ge, eq, ne) return boolean
    /// - Select's first arg must be boolean, returns numeric
    /// - All other ops require numeric operands and return numeric
    /// - Leaves (Var, Const) are numeric
    pub fn is_type_valid(&self) -> bool {
        self.check_type().is_some()
    }

    /// Returns Some(is_boolean) if valid, None if invalid.
    ///
    /// Uses iterative traversal to avoid stack overflow on deep trees.
    fn check_type(&self) -> Option<bool> {
        enum Task<'a> {
            Visit(&'a ExprTree),
            Complete { name: &'static str, arity: usize },
        }

        let mut stack: Vec<Task<'_>> = vec![Task::Visit(self)];
        // result_stack holds Option<bool>: Some(true)=boolean, Some(false)=numeric, None=invalid
        let mut result_stack: Vec<Option<bool>> = Vec::new();

        while let Some(task) = stack.pop() {
            match task {
                Task::Visit(node) => match node {
                    Self::Leaf(_) => result_stack.push(Some(false)), // Leaves are numeric

                    Self::Op { op, children } => {
                        let name = op.name();
                        stack.push(Task::Complete { name, arity: children.len() });
                        for child in children.iter().rev() {
                            stack.push(Task::Visit(child));
                        }
                    }
                },
                Task::Complete { name, arity } => {
                    let start = result_stack.len().saturating_sub(arity);
                    let child_types: Vec<Option<bool>> = result_stack.drain(start..).collect();

                    // Comparison ops: require numeric children, return boolean
                    if matches!(name, "lt" | "le" | "gt" | "ge" | "eq" | "ne") {
                        let valid = child_types.iter().all(|t| matches!(t, Some(false)));
                        if valid {
                            result_stack.push(Some(true)); // Return boolean
                        } else {
                            result_stack.push(None); // Invalid
                        }
                        continue;
                    }

                    // Select: first child MUST be boolean, others numeric
                    if name == "select" && arity == 3 {
                        let cond_ok = matches!(child_types[0], Some(true));
                        let branches_ok = child_types[1..].iter().all(|t| matches!(t, Some(false)));
                        if cond_ok && branches_ok {
                            result_stack.push(Some(false)); // Return numeric
                        } else {
                            result_stack.push(None); // Invalid
                        }
                        continue;
                    }

                    // All other ops: require numeric children, return numeric
                    let valid = child_types.iter().all(|t| matches!(t, Some(false)));
                    if valid {
                        result_stack.push(Some(false)); // Return numeric
                    } else {
                        result_stack.push(None); // Invalid
                    }
                }
            }
        }

        result_stack.pop().unwrap_or_else(|| panic!("check_type: empty result stack"))
    }

    // Constructor helpers for common operations
    pub fn add(a: Self, b: Self) -> Self {
        Self::Op {
            op: &super::ops::Add,
            children: alloc::vec![a, b],
        }
    }

    pub fn sub(a: Self, b: Self) -> Self {
        Self::Op {
            op: &super::ops::Sub,
            children: alloc::vec![a, b],
        }
    }

    pub fn mul(a: Self, b: Self) -> Self {
        Self::Op {
            op: &super::ops::Mul,
            children: alloc::vec![a, b],
        }
    }

    pub fn div(a: Self, b: Self) -> Self {
        Self::Op {
            op: &super::ops::Div,
            children: alloc::vec![a, b],
        }
    }

    pub fn neg(a: Self) -> Self {
        Self::Op {
            op: &super::ops::Neg,
            children: alloc::vec![a],
        }
    }

    pub fn sqrt(a: Self) -> Self {
        Self::Op {
            op: &super::ops::Sqrt,
            children: alloc::vec![a],
        }
    }

    pub fn abs(a: Self) -> Self {
        Self::Op {
            op: &super::ops::Abs,
            children: alloc::vec![a],
        }
    }

    pub fn min(a: Self, b: Self) -> Self {
        Self::Op {
            op: &super::ops::Min,
            children: alloc::vec![a, b],
        }
    }

    pub fn max(a: Self, b: Self) -> Self {
        Self::Op {
            op: &super::ops::Max,
            children: alloc::vec![a, b],
        }
    }

    pub fn mul_add(a: Self, b: Self, c: Self) -> Self {
        Self::Op {
            op: &super::ops::MulAdd,
            children: alloc::vec![a, b, c],
        }
    }

    /// Compute the cost of this expression tree using the given cost model.
    ///
    /// Uses iterative traversal to avoid stack overflow on deep trees.
    pub fn cost(&self, costs: &CostModel) -> usize {
        let mut total = 0usize;
        let mut stack: Vec<&ExprTree> = vec![self];
        while let Some(node) = stack.pop() {
            match node {
                Self::Leaf(_) => {} // Variables and constants are free
                Self::Op { op, children } => {
                    total += costs.cost(op.kind());
                    for child in children {
                        stack.push(child);
                    }
                }
            }
        }
        total
    }

}

use crate::nnue::{ExprNnue, EdgeAccumulator};

/// Incremental 3-pass neural extractor.
///
/// Instead of recomputing the full `EdgeAccumulator` from scratch for each
/// candidate, performs an O(Δ) swap: remove old subtree edges, add new.
/// This makes each candidate evaluation O(subtree_size) instead of O(whole_tree).
///
/// ## Algorithm
///
/// - **Pass 1 (Bootstrap)**: extract shallowest tree (minimum AST node count per
///   e-class). This is fast and gives a reasonable starting point.
/// - **Passes 2-3 (Refine)**: for each active e-class, try alternative nodes
///   using O(Δ) accumulator updates. Accept if strictly lower cost.
///   Repeat until fixpoint or `MAX_PASSES` (3) reached.
pub struct IncrementalExtractor<'a> {
    nnue: &'a ExprNnue,
    top_k: usize,
}

impl<'a> IncrementalExtractor<'a> {
    /// Create a new incremental extractor.
    ///
    /// `top_k` bounds how many alternative nodes per e-class are evaluated
    /// during refinement passes (default 8 is a good trade-off).
    pub fn new(nnue: &'a ExprNnue, top_k: usize) -> Self {
        Self { nnue, top_k }
    }

    /// Run the 3-pass extraction and return the best tree with its neural cost.
    pub fn extract(&self, egraph: &EGraph, root_class: EClassId) -> (ExprTree, f32) {
        const MAX_PASSES: usize = 3;

        // Pass 1: Bootstrap with shallowest tree (minimum node count)
        let mut choices = self.extract_shallowest(egraph, root_class);

        // Build initial tree and its full-tree accumulator
        let initial_tree = super::nnue_adapter::build_tree_with_choices(
            egraph, root_class, &choices,
        );
        let initial_expr = super::nnue_adapter::expr_tree_to_nnue(&initial_tree);
        // Use dedup variant: extracted trees may contain CSE (shared subtrees).
        let mut current_acc = EdgeAccumulator::from_expr_dedup(
            &initial_expr, &self.nnue.embeddings,
        );
        let mut current_cost = self.nnue.predict_log_cost_with_features(&current_acc);

        // Passes 2..MAX_PASSES: Refine via O(Δ) swaps
        for _pass in 1..MAX_PASSES {
            let active = self.get_active_classes(egraph, root_class, &choices);
            let mut improved = false;

            for &class in &active {
                let canonical = egraph.find(class);
                let nodes = egraph.nodes(canonical);
                if nodes.len() <= 1 {
                    continue; // No alternatives
                }

                let current_node_idx = choices[canonical.0 as usize].unwrap_or(0);

                // Build old subtree for this e-class
                let old_subtree = super::nnue_adapter::build_subtree_with_choices(
                    egraph, canonical, current_node_idx, &choices,
                );
                let old_subtree_expr = super::nnue_adapter::expr_tree_to_nnue(
                    &old_subtree,
                );

                let candidates_to_try = nodes.len().min(self.top_k);

                for node_idx in 0..candidates_to_try {
                    if node_idx == current_node_idx {
                        continue;
                    }

                    // Build candidate subtree (same descendants, different root node)
                    let cand_subtree = super::nnue_adapter::build_subtree_with_choices(
                        egraph, canonical, node_idx, &choices,
                    );
                    let cand_expr = super::nnue_adapter::expr_tree_to_nnue(
                        &cand_subtree,
                    );

                    // O(Δ) accumulator swap: remove old, add new.
                    // Candidate subtrees are freshly built slices — use dedup to
                    // guard against any shared subtrees they may contain.
                    let mut test_acc = current_acc.clone();
                    test_acc.remove_expr_edges(
                        &old_subtree_expr, &self.nnue.embeddings,
                    );
                    let cand_acc = EdgeAccumulator::from_expr_dedup(
                        &cand_expr, &self.nnue.embeddings,
                    );
                    test_acc.merge(&cand_acc);

                    let test_cost = self.nnue.predict_log_cost_with_features(&test_acc);

                    if test_cost < current_cost {
                        choices[canonical.0 as usize] = Some(node_idx);
                        current_acc = test_acc;
                        current_cost = test_cost;
                        improved = true;
                        break; // First-improvement: move to next e-class
                    }
                }
            }

            if !improved {
                break; // Fixpoint
            }
        }

        // Build final tree from refined choices
        let tree = super::nnue_adapter::build_tree_with_choices(
            egraph, root_class, &choices,
        );
        (tree, current_cost)
    }

    /// Pass 1: Bottom-up DP choosing the node with fewest total AST nodes.
    fn extract_shallowest(
        &self,
        egraph: &EGraph,
        root: EClassId,
    ) -> Vec<Option<usize>> {
        use alloc::collections::BTreeSet;

        const CYCLE_COUNT: usize = 1_000_000;

        let num_classes = egraph.num_classes();
        let mut best_count: Vec<Option<usize>> = alloc::vec![None; num_classes];
        let mut best_node: Vec<Option<usize>> = alloc::vec![None; num_classes];

        let mut stack: Vec<(EClassId, bool)> = vec![(root, false)];
        let mut on_stack: BTreeSet<u32> = BTreeSet::new();

        while let Some((class, children_done)) = stack.pop() {
            let canonical = egraph.find(class);

            if best_count[canonical.0 as usize].is_some() {
                continue;
            }

            if !children_done {
                if !on_stack.insert(canonical.0) {
                    continue;
                }
                stack.push((canonical, true));

                for node in egraph.nodes(canonical) {
                    if let ENode::Op { children, .. } = node {
                        for &child in children {
                            let child_can = egraph.find(child);
                            if best_count[child_can.0 as usize].is_none() {
                                stack.push((child, false));
                            }
                        }
                    }
                }
            } else {
                on_stack.remove(&canonical.0);

                let nodes = egraph.nodes(canonical);
                let mut min_count = usize::MAX;
                let mut min_idx = 0;

                for (idx, node) in nodes.iter().enumerate() {
                    let count = match node {
                        ENode::Var(_) | ENode::Const(_) => 1,
                        ENode::Op { children, .. } => {
                            if children.iter().any(|&c| egraph.find(c) == canonical) {
                                CYCLE_COUNT
                            } else {
                                let child_sum: usize = children
                                    .iter()
                                    .map(|&c| {
                                        let cc = egraph.find(c);
                                        best_count[cc.0 as usize]
                                            .unwrap_or(CYCLE_COUNT)
                                    })
                                    .sum();
                                1usize.saturating_add(child_sum)
                            }
                        }
                    };

                    if count < min_count {
                        min_count = count;
                        min_idx = idx;
                    }
                }

                best_count[canonical.0 as usize] = Some(min_count);
                best_node[canonical.0 as usize] = Some(min_idx);
            }
        }

        best_node
    }

    /// Walk the current best tree and collect active (reachable) e-class IDs.
    fn get_active_classes(
        &self,
        egraph: &EGraph,
        root: EClassId,
        choices: &[Option<usize>],
    ) -> Vec<EClassId> {
        use alloc::collections::BTreeSet;

        let mut active = Vec::new();
        let mut visited = BTreeSet::new();
        let mut stack = vec![root];

        while let Some(class) = stack.pop() {
            let canonical = egraph.find(class);
            if !visited.insert(canonical.0) {
                continue;
            }

            active.push(canonical);

            let node_idx = choices[canonical.0 as usize].unwrap_or(0);
            let nodes = egraph.nodes(canonical);
            if node_idx < nodes.len() {
                if let ENode::Op { children, .. } = &nodes[node_idx] {
                    for &child in children {
                        stack.push(child);
                    }
                }
            }
        }

        active
    }
}

/// Extract the minimum-cost expression tree from an e-class.
///
/// Uses dynamic programming: cost(class) = min over all nodes in class.
///
/// # Type Parameter
///
/// The cost function can be any type implementing `CostFunction`:
/// - `CostModel` for hardcoded costs
/// - Neural cost models (e.g., `ExprNnue` via adapter)
/// - Custom domain-specific cost functions
pub fn extract<C: CostFunction>(egraph: &EGraph, root: EClassId, costs: &C) -> (ExprTree, usize) {
    use alloc::collections::BTreeSet;

    // Cap for cycle/self-referential costs - high but not astronomical
    const CYCLE_COST: usize = 1_000_000;

    let num_classes = egraph.num_classes();
    let mut best_cost: Vec<Option<usize>> = alloc::vec![None; num_classes];
    let mut best_node: Vec<Option<usize>> = alloc::vec![None; num_classes];

    // Phase 1: Iterative bottom-up cost computation using topological order
    // We use a work stack to avoid recursion
    let mut stack: Vec<(EClassId, bool)> = vec![(root, false)]; // (class, children_processed)
    let mut on_stack: BTreeSet<u32> = BTreeSet::new();

    while let Some((class, children_done)) = stack.pop() {
        let canonical = egraph.find(class);

        // Already computed
        if best_cost[canonical.0 as usize].is_some() {
            continue;
        }

        if !children_done {
            // First visit: push self back (to process after children), then push children
            if !on_stack.insert(canonical.0) {
                // Cycle detected - don't cache, parent will handle with high cost
                continue;
            }

            stack.push((canonical, true)); // Come back after children

            // Push all children that need processing
            for node in egraph.nodes(canonical) {
                if let ENode::Op { children, .. } = node {
                    for &child in children {
                        let child_canonical = egraph.find(child);
                        if best_cost[child_canonical.0 as usize].is_none() {
                            stack.push((child, false));
                        }
                    }
                }
            }
        } else {
            // Second visit: all children are computed, now compute this class
            on_stack.remove(&canonical.0);

            let nodes = egraph.nodes(canonical);
            let mut min_cost = usize::MAX;
            let mut min_idx = 0;

            for (idx, node) in nodes.iter().enumerate() {
                let this_node_cost = match node {
                    ENode::Var(_) | ENode::Const(_) => costs.node_cost(node, None),
                    ENode::Op { children, .. } => {
                        // Check for self-referential children
                        if children.iter().any(|&c| egraph.find(c) == canonical) {
                            CYCLE_COST
                        } else {
                            let op_cost = costs.node_cost(node, None);
                            let children_cost: usize = children
                                .iter()
                                .map(|&child| {
                                    let c = egraph.find(child);
                                    best_cost[c.0 as usize].unwrap_or(CYCLE_COST)
                                })
                                .sum();
                            op_cost.saturating_add(children_cost)
                        }
                    }
                };

                if this_node_cost < min_cost {
                    min_cost = this_node_cost;
                    min_idx = idx;
                }
            }

            best_cost[canonical.0 as usize] = Some(min_cost);
            best_node[canonical.0 as usize] = Some(min_idx);
        }
    }

    let total_cost = best_cost[egraph.find(root).0 as usize].unwrap_or(usize::MAX);

    // Phase 2: Iterative top-down tree construction
    // Use a stack of (class, partially_built_tree_slot)
    enum BuildTask {
        Visit(EClassId),
        Complete { canonical: u32, op: &'static dyn super::ops::Op, num_children: usize },
    }

    let mut build_stack: Vec<BuildTask> = vec![BuildTask::Visit(root)];
    let mut result_stack: Vec<ExprTree> = Vec::new();
    let mut building: BTreeSet<u32> = BTreeSet::new();

    while let Some(task) = build_stack.pop() {
        match task {
            BuildTask::Visit(class) => {
                let canonical = egraph.find(class);

                // Cycle detection
                if !building.insert(canonical.0) {
                    result_stack.push(ExprTree::Leaf(Leaf::Const(0.0)));
                    continue;
                }

                let node_idx = best_node[canonical.0 as usize].unwrap_or(0);
                let node = &egraph.nodes(canonical)[node_idx];

                match node {
                    ENode::Var(idx) => {
                        building.remove(&canonical.0);
                        result_stack.push(ExprTree::Leaf(Leaf::Var(*idx)));
                    }
                    ENode::Const(bits) => {
                        building.remove(&canonical.0);
                        result_stack.push(ExprTree::Leaf(Leaf::Const(f32::from_bits(*bits))));
                    }
                    ENode::Op { op, children } => {
                        // Push completion task, then visit children in reverse order
                        build_stack.push(BuildTask::Complete {
                            canonical: canonical.0,
                            op: *op,
                            num_children: children.len(),
                        });
                        for &child in children.iter().rev() {
                            build_stack.push(BuildTask::Visit(child));
                        }
                    }
                }
            }
            BuildTask::Complete { canonical, op, num_children } => {
                building.remove(&canonical);
                // Pop children from result stack (they're in correct order now)
                let start = result_stack.len().saturating_sub(num_children);
                let child_trees: Vec<ExprTree> = result_stack.drain(start..).collect();
                result_stack.push(ExprTree::Op { op, children: child_trees });
            }
        }
    }

    let tree = result_stack.pop().unwrap_or_else(|| ExprTree::Leaf(Leaf::Const(0.0)));
    (tree, total_cost)
}

// ============================================================================
// DAG-Aware Extraction
// ============================================================================

/// Result of DAG-aware extraction with sharing information.
///
/// Unlike regular extraction which produces a tree, this tracks:
/// - Which e-classes are used multiple times (candidates for let-binding)
/// - The topological order for emission (dependencies first)
/// - The best node choice per e-class
///
/// # Example
///
/// For `sin(X) * sin(X) + sin(X)`:
/// - E-class containing `sin(X)` is used 3 times
/// - DAG extraction identifies this for let-binding
/// - Codegen emits: `let __0 = X.sin().eval(__p); (__0 * __0 + __0).eval(__p)`
#[derive(Clone, Debug)]
pub struct ExtractedDAG {
    /// The root e-class of the expression.
    pub root: EClassId,

    /// E-classes used more than once: (class_id, use_count).
    /// These are candidates for let-binding in codegen.
    pub shared: Vec<(EClassId, usize)>,

    /// Topological order for emission (dependencies before dependents).
    /// Shared e-classes appear before e-classes that use them.
    pub schedule: Vec<EClassId>,

    /// Best node choice per e-class (indexed by canonical e-class ID).
    pub choices: Vec<Option<usize>>,

    /// Total cost of the extracted expression.
    pub total_cost: usize,
}

impl ExtractedDAG {
    /// Check if an e-class is shared (used more than once).
    pub fn is_shared(&self, class: EClassId) -> bool {
        self.shared.iter().any(|(id, _)| *id == class)
    }

    /// Get the use count for an e-class.
    pub fn use_count(&self, class: EClassId) -> usize {
        self.shared.iter()
            .find(|(id, _)| *id == class)
            .map(|(_, count)| *count)
            .unwrap_or(1)
    }

    /// Get the index of the best node for an e-class.
    pub fn best_node_idx(&self, class: EClassId) -> Option<usize> {
        self.choices.get(class.0 as usize).and_then(|o| *o)
    }
}

/// Extract a DAG with sharing information from an e-class.
///
/// This is the DAG-aware version of `extract()`. It returns structural
/// information about sharing that codegen can use to emit let-bindings.
///
/// # Arguments
///
/// * `egraph` - The e-graph to extract from
/// * `root` - The root e-class
/// * `costs` - The cost function for choosing best nodes
///
/// # Returns
///
/// An `ExtractedDAG` containing:
/// - Best node per e-class
/// - Shared e-classes (for let-binding)
/// - Topological order for emission
pub fn extract_dag<C: CostFunction>(egraph: &EGraph, root: EClassId, costs: &C) -> ExtractedDAG {
    use alloc::collections::BTreeSet;

    const CYCLE_COST: usize = 1_000_000;

    let num_classes = egraph.num_classes();
    let mut best_cost: Vec<Option<usize>> = alloc::vec![None; num_classes];
    let mut best_node: Vec<Option<usize>> = alloc::vec![None; num_classes];

    // Phase 1: Compute best node per e-class (same as regular extraction)
    let mut stack: Vec<(EClassId, bool)> = vec![(root, false)];
    let mut on_stack: BTreeSet<u32> = BTreeSet::new();

    while let Some((class, children_done)) = stack.pop() {
        let canonical = egraph.find(class);

        if best_cost[canonical.0 as usize].is_some() {
            continue;
        }

        if !children_done {
            if !on_stack.insert(canonical.0) {
                continue;
            }

            stack.push((canonical, true));

            for node in egraph.nodes(canonical) {
                if let ENode::Op { children, .. } = node {
                    for &child in children {
                        let child_canonical = egraph.find(child);
                        if best_cost[child_canonical.0 as usize].is_none() {
                            stack.push((child, false));
                        }
                    }
                }
            }
        } else {
            on_stack.remove(&canonical.0);

            let nodes = egraph.nodes(canonical);
            let mut min_cost = usize::MAX;
            let mut min_idx = 0;

            for (idx, node) in nodes.iter().enumerate() {
                let this_node_cost = match node {
                    ENode::Var(_) | ENode::Const(_) => costs.node_cost(node, None),
                    ENode::Op { children, .. } => {
                        if children.iter().any(|&c| egraph.find(c) == canonical) {
                            CYCLE_COST
                        } else {
                            let op_cost = costs.node_cost(node, None);
                            let children_cost: usize = children
                                .iter()
                                .map(|&child| {
                                    let c = egraph.find(child);
                                    best_cost[c.0 as usize].unwrap_or(CYCLE_COST)
                                })
                                .sum();
                            op_cost.saturating_add(children_cost)
                        }
                    }
                };

                if this_node_cost < min_cost {
                    min_cost = this_node_cost;
                    min_idx = idx;
                }
            }

            best_cost[canonical.0 as usize] = Some(min_cost);
            best_node[canonical.0 as usize] = Some(min_idx);
        }
    }

    let total_cost = best_cost[egraph.find(root).0 as usize].unwrap_or(usize::MAX);

    // Phase 2: Count references to each e-class in the extracted DAG
    let mut ref_counts: Vec<usize> = alloc::vec![0; num_classes];
    count_refs_recursive(egraph, root, &best_node, &mut ref_counts);

    // Phase 3: Identify shared e-classes (count > 1)
    let shared: Vec<(EClassId, usize)> = ref_counts.iter()
        .enumerate()
        .filter(|(_, count)| **count > 1)
        .map(|(idx, count)| (EClassId(idx as u32), *count))
        .collect();

    // Phase 4: Topological sort for emission order
    let schedule = toposort_dag(egraph, root, &best_node, &shared);

    ExtractedDAG {
        root: egraph.find(root),
        shared,
        schedule,
        choices: best_node,
        total_cost,
    }
}

/// Count references to each e-class in the extracted expression.
///
/// Uses iterative traversal with explicit stack to avoid thread stack overflow.
fn count_refs_recursive(
    egraph: &EGraph,
    class: EClassId,
    best_node: &[Option<usize>],
    ref_counts: &mut [usize],
) {
    let mut stack: Vec<EClassId> = alloc::vec![class];

    while let Some(cls) = stack.pop() {
        let canonical = egraph.find(cls);
        ref_counts[canonical.0 as usize] += 1;

        // Only recurse on first visit to count true structural refs
        if ref_counts[canonical.0 as usize] == 1 {
            if let Some(node_idx) = best_node[canonical.0 as usize] {
                let node = &egraph.nodes(canonical)[node_idx];
                if let ENode::Op { children, .. } = node {
                    for &child in children {
                        stack.push(child);
                    }
                }
            }
        }
    }
}

/// Topological sort of e-classes for emission order.
///
/// Returns e-classes in order such that dependencies come before dependents.
/// Shared e-classes are prioritized to appear early.
///
/// Uses iterative post-order traversal to avoid thread stack overflow.
fn toposort_dag(
    egraph: &EGraph,
    root: EClassId,
    best_node: &[Option<usize>],
    shared: &[(EClassId, usize)],
) -> Vec<EClassId> {
    use alloc::collections::BTreeSet;

    let shared_set: BTreeSet<u32> = shared.iter().map(|(id, _)| id.0).collect();
    let mut visited: BTreeSet<u32> = BTreeSet::new();
    let mut result = Vec::new();

    // Iterative post-order: (class, children_pushed)
    let mut stack: Vec<(EClassId, bool)> = alloc::vec![(root, false)];

    while let Some((class, children_done)) = stack.pop() {
        let canonical = egraph.find(class);

        if visited.contains(&canonical.0) {
            continue;
        }

        if !children_done {
            stack.push((canonical, true));

            if let Some(node_idx) = best_node.get(canonical.0 as usize).and_then(|o| *o) {
                let node = &egraph.nodes(canonical)[node_idx];
                if let ENode::Op { children, .. } = node {
                    for &child in children {
                        let child_can = egraph.find(child);
                        if !visited.contains(&child_can.0) {
                            stack.push((child, false));
                        }
                    }
                }
            }
        } else {
            visited.insert(canonical.0);

            if shared_set.contains(&canonical.0) {
                result.push(canonical);
            }
        }
    }

    // Add root if not already included
    let root_canonical = egraph.find(root);
    if !result.iter().any(|id| *id == root_canonical) {
        result.push(root_canonical);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expr_tree_node_count_should_succeed_when_called() {
        let x = ExprTree::var(0);
        assert_eq!(x.node_count(), 1);

        let sum = ExprTree::add(ExprTree::var(0), ExprTree::var(1));
        assert_eq!(sum.node_count(), 3); // Add + X + Y
    }

    #[test]
    fn expr_tree_depth_should_succeed_when_called() {
        let x = ExprTree::var(0);
        assert_eq!(x.depth(), 1);

        let sum = ExprTree::add(ExprTree::var(0), ExprTree::var(1));
        assert_eq!(sum.depth(), 2);

        // (X + Y) * Z
        let nested = ExprTree::mul(sum, ExprTree::var(2));
        assert_eq!(nested.depth(), 3);
    }

    #[test]
    fn extract_simple_should_succeed_when_called() {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));

        let costs = CostModel::default();
        let (tree, cost) = extract(&egraph, x, &costs);

        assert!(matches!(tree, ExprTree::Leaf(Leaf::Var(0))));
        assert_eq!(cost, 0); // Leaf nodes (Var/Const) have cost 0
    }

    #[test]
    fn extract_with_ops_should_succeed_when_called() {
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let sum = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![x, y],
        });

        let costs = CostModel::default();
        let (tree, _cost) = extract(&egraph, sum, &costs);

        assert!(matches!(tree, ExprTree::Op { .. }));
        assert_eq!(tree.node_count(), 3); // Add + X + Y
    }

    // ========================================================================
    // DAG Extraction Tests
    // ========================================================================

    #[test]
    fn extract_dag_simple_should_succeed_when_called() {
        // X + Y: no sharing
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let sum = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![x, y],
        });

        let costs = CostModel::default();
        let dag = extract_dag(&egraph, sum, &costs);

        assert!(dag.shared.is_empty(), "X + Y should have no shared subexprs");
        assert_eq!(dag.root, egraph.find(sum));
    }

    #[test]
    fn extract_dag_shared_subexpr_should_succeed_when_called() {
        // X * X: X is used twice
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let x_squared = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![x, x], // X used twice!
        });

        let costs = CostModel::default();
        let dag = extract_dag(&egraph, x_squared, &costs);

        // X should be marked as shared (used 2 times)
        assert!(!dag.shared.is_empty(), "X * X should have X as shared");
        assert!(dag.is_shared(x), "X should be shared");
        assert_eq!(dag.use_count(x), 2);
    }

    #[test]
    fn extract_dag_triple_use_should_succeed_when_called() {
        // sin(X) * sin(X) + sin(X): sin(X) used 3 times
        // We simulate this structure without actual sin
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        // Simulate sin(X) as sqrt(X) for test purposes
        let sin_x = egraph.add(ENode::Op {
            op: &super::super::ops::Sqrt,
            children: alloc::vec![x],
        });
        let sin_x_squared = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![sin_x, sin_x],
        });
        let result = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![sin_x_squared, sin_x],
        });

        let costs = CostModel::default();
        let dag = extract_dag(&egraph, result, &costs);

        // sin_x should be shared (used 3 times: twice in Mul, once in Add)
        assert!(dag.is_shared(sin_x), "sqrt(X) should be shared (used 3 times)");
        assert_eq!(dag.use_count(sin_x), 3);

        // Schedule should have sin_x before the operations that use it
        let sin_x_idx = dag.schedule.iter().position(|&id| id == egraph.find(sin_x));
        assert!(sin_x_idx.is_some(), "sin_x should be in schedule");
    }

    #[test]
    fn extract_dag_nested_sharing_should_succeed_when_called() {
        // (X + Y) * (X + Y): (X + Y) is shared
        let mut egraph = EGraph::new();
        let x = egraph.add(ENode::Var(0));
        let y = egraph.add(ENode::Var(1));
        let sum = egraph.add(ENode::Op {
            op: &super::super::ops::Add,
            children: alloc::vec![x, y],
        });
        let product = egraph.add(ENode::Op {
            op: &super::super::ops::Mul,
            children: alloc::vec![sum, sum], // sum used twice
        });

        let costs = CostModel::default();
        let dag = extract_dag(&egraph, product, &costs);

        // (X + Y) should be shared
        assert!(dag.is_shared(sum), "(X + Y) should be shared");
        assert_eq!(dag.use_count(sum), 2);
    }
}
