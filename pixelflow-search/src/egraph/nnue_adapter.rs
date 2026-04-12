//! Adapter to convert between e-graph expressions and NNUE expression trees.
//!
//! Bridges the gap between `pixelflow_search::egraph` types and `pixelflow_search::nnue` types
//! for feature extraction and training data generation.
//!
//! ## Dual-Head NNUE Integration
//!
//! This module provides adapters for using [`ExprNnue`] with the e-graph:
//!
//! - [`expr_tree_to_nnue`]: Converts `ExprTree` to NNUE `Expr` for prediction
//! - [`extract_neural`]: Bottom-up DP extraction using full NNUE forward pass

use crate::egraph::{EClassId, EGraph, ENode, ops};
use crate::egraph::extract::{ExprTree, Leaf};
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use crate::nnue::{Expr, OpKind, ExprNnue};

/// Convert e-graph `Op` to `OpKind`.
///
/// Uses the `Op::kind()` method which delegates to the canonical `OpKind`.
#[inline]
pub fn op_to_nnue(op: &dyn crate::egraph::ops::Op) -> OpKind {
    op.kind()
}

/// Convert `OpKind` to e-graph `Op` reference.
///
/// Delegates to `ops::op_from_kind`.
#[inline]
pub fn nnue_to_op(kind: OpKind) -> Option<&'static dyn crate::egraph::ops::Op> {
    ops::op_from_kind(kind)
}

/// Extract a concrete `Expr` from an e-class.
/// Picks the first representative (a real implementation would use cost-based extraction).
///
/// Uses iterative traversal with explicit stack to avoid thread stack overflow
/// on deep expression trees from e-graph extraction.
pub fn eclass_to_expr(egraph: &EGraph, class: EClassId) -> Expr {
    enum Task {
        Visit(EClassId),
        Complete { op_kind: crate::nnue::OpKind, arity: usize },
    }

    let mut stack: Vec<Task> = vec![Task::Visit(class)];
    let mut result_stack: Vec<Expr> = Vec::new();

    while let Some(task) = stack.pop() {
        match task {
            Task::Visit(cls) => {
                let node = &egraph.nodes(cls)[0];
                match node {
                    ENode::Var(idx) => result_stack.push(Expr::Var(*idx)),
                    ENode::Const(bits) => result_stack.push(Expr::Const(f32::from_bits(*bits))),
                    ENode::Op { op, children } => {
                        let op_kind = op_to_nnue(*op);
                        let arity = children.len();
                        if arity == 0 || arity > 3 {
                            panic!("Unsupported arity {} for op {}", arity, op.name());
                        }
                        stack.push(Task::Complete { op_kind, arity });
                        for &child in children.iter().rev() {
                            stack.push(Task::Visit(child));
                        }
                    }
                }
            }
            Task::Complete { op_kind, arity } => {
                let start = result_stack.len().saturating_sub(arity);
                let mut children: Vec<Expr> = result_stack.drain(start..).collect();
                let expr = match arity {
                    1 => Expr::Unary(op_kind, Box::new(children.remove(0))),
                    2 => {
                        let b = children.remove(1);
                        let a = children.remove(0);
                        Expr::Binary(op_kind, Box::new(a), Box::new(b))
                    }
                    3 => {
                        let c = children.remove(2);
                        let b = children.remove(1);
                        let a = children.remove(0);
                        Expr::Ternary(op_kind, Box::new(a), Box::new(b), Box::new(c))
                    }
                    _ => unreachable!("arity checked above"),
                };
                result_stack.push(expr);
            }
        }
    }

    result_stack.pop().unwrap_or_else(|| panic!("eclass_to_expr: empty result stack"))
}

/// Convert an `ExprTree` to a NNUE `Expr` for feature extraction.
///
/// Uses iterative traversal with explicit stack to avoid thread stack overflow.
pub fn expr_tree_to_nnue(tree: &ExprTree) -> Expr {
    enum Task<'a> {
        Visit(&'a ExprTree),
        Complete { op_kind: crate::nnue::OpKind, arity: usize },
    }

    let mut stack: Vec<Task<'_>> = vec![Task::Visit(tree)];
    let mut result_stack: Vec<Expr> = Vec::new();

    while let Some(task) = stack.pop() {
        match task {
            Task::Visit(node) => match node {
                ExprTree::Leaf(Leaf::Var(i)) => result_stack.push(Expr::Var(*i)),
                ExprTree::Leaf(Leaf::Const(c)) => result_stack.push(Expr::Const(*c)),
                ExprTree::Op { op, children } => {
                    let op_kind = op_to_nnue(*op);
                    match children.len() {
                        0 => result_stack.push(Expr::Const(0.0)),
                        1 | 2 | 3 => {
                            stack.push(Task::Complete { op_kind, arity: children.len() });
                            for child in children.iter().rev() {
                                stack.push(Task::Visit(child));
                            }
                        }
                        _ => {
                            // Variadic: only use first element (matches original behavior)
                            stack.push(Task::Visit(&children[0]));
                        }
                    }
                }
            },
            Task::Complete { op_kind, arity } => {
                let start = result_stack.len().saturating_sub(arity);
                let mut children: Vec<Expr> = result_stack.drain(start..).collect();
                let expr = match arity {
                    1 => Expr::Unary(op_kind, Box::new(children.remove(0))),
                    2 => {
                        let b = children.remove(1);
                        let a = children.remove(0);
                        Expr::Binary(op_kind, Box::new(a), Box::new(b))
                    }
                    3 => {
                        let c = children.remove(2);
                        let b = children.remove(1);
                        let a = children.remove(0);
                        Expr::Ternary(op_kind, Box::new(a), Box::new(b), Box::new(c))
                    }
                    _ => unreachable!("arity checked above"),
                };
                result_stack.push(expr);
            }
        }
    }

    result_stack.pop().unwrap_or_else(|| panic!("expr_tree_to_nnue: empty result stack"))
}

/// Predict full expression cost using the value head.
///
/// This is for cases where you have an `ExprTree` and want the
/// neural network's full prediction (not just per-node costs).
pub fn predict_tree_cost(tree: &ExprTree, nnue: &ExprNnue) -> f32 {
    let expr = expr_tree_to_nnue(tree);
    nnue.predict_cost(&expr)
}

// ============================================================================
// Neural Extraction (uses full predict_log_cost, not per-node summation)
// ============================================================================

/// Extract the minimum-cost expression using incremental NNUE evaluation.
///
/// Delegates to [`IncrementalExtractor`](super::extract::IncrementalExtractor)
/// which uses a 3-pass strategy:
///
/// 1. **Bootstrap**: extract shallowest tree (minimum AST node count)
/// 2. **Refine** (×2): for each active e-class, try alternative nodes
///    using O(Δ) accumulator updates. Accept if strictly lower cost.
///
/// This captures non-additive structural features (ILP, critical path)
/// that per-node cost summation misses.
pub fn extract_neural(egraph: &EGraph, root: EClassId, nnue: &ExprNnue) -> (ExprTree, f32) {
    let extractor = super::extract::IncrementalExtractor::new(nnue, 8);
    extractor.extract(egraph, root)
}

/// Build a subtree using a specific node choice for the root, and best choices for descendants.
/// Uses iterative approach to avoid stack overflow.
///
/// Cycles in the e-graph (from class merging during saturation) are normal.
/// When a cycle is detected, we truncate with `Leaf::Const(0.0)` — the neural
/// network gets a finite approximation of the infinite unrolling.
pub(super) fn build_subtree_with_choices(
    egraph: &EGraph,
    class: EClassId,
    node_idx: usize,
    best_node: &[Option<usize>],
) -> ExprTree {
    use alloc::collections::BTreeSet;

    enum BuildTask {
        Visit { class: EClassId, node_idx: usize },
        Complete { canonical: u32, op: &'static dyn crate::egraph::ops::Op, num_children: usize },
    }

    let mut stack: Vec<BuildTask> = vec![BuildTask::Visit { class, node_idx }];
    let mut result_stack: Vec<ExprTree> = Vec::new();
    let mut visiting: BTreeSet<u32> = BTreeSet::new();

    while let Some(task) = stack.pop() {
        match task {
            BuildTask::Visit { class, node_idx } => {
                let canonical = egraph.find(class);

                // Cycle: truncate with a constant leaf (same as standard extract)
                if !visiting.insert(canonical.0) {
                    result_stack.push(ExprTree::Leaf(Leaf::Const(0.0)));
                    continue;
                }

                let node = &egraph.nodes(canonical)[node_idx];

                match node {
                    ENode::Var(idx) => {
                        visiting.remove(&canonical.0);
                        result_stack.push(ExprTree::Leaf(Leaf::Var(*idx)));
                    }
                    ENode::Const(bits) => {
                        visiting.remove(&canonical.0);
                        result_stack.push(ExprTree::Leaf(Leaf::Const(f32::from_bits(*bits))));
                    }
                    ENode::Op { op, children } => {
                        stack.push(BuildTask::Complete { canonical: canonical.0, op: *op, num_children: children.len() });
                        for &child in children.iter().rev() {
                            let child_canonical = egraph.find(child);
                            // If child was resolved as all-cyclic, pick node 0 as fallback
                            let child_node_idx = best_node[child_canonical.0 as usize].unwrap_or(0);
                            stack.push(BuildTask::Visit { class: child, node_idx: child_node_idx });
                        }
                    }
                }
            }
            BuildTask::Complete { canonical, op, num_children } => {
                visiting.remove(&canonical);
                let start = result_stack.len().saturating_sub(num_children);
                let child_trees: Vec<ExprTree> = result_stack.drain(start..).collect();
                result_stack.push(ExprTree::Op { op, children: child_trees });
            }
        }
    }

    result_stack.pop().unwrap_or_else(|| ExprTree::Leaf(Leaf::Const(0.0)))
}

/// Build the final tree using best node choices for all e-classes.
///
/// Cycles in the e-graph are truncated with `Leaf::Const(0.0)`, matching
/// the behavior of the standard additive `extract()`.
pub(super) fn build_tree_with_choices(
    egraph: &EGraph,
    root: EClassId,
    best_node: &[Option<usize>],
) -> ExprTree {
    let root_canonical = egraph.find(root);
    // Every class gets a best_node (fallback to 0), so unwrap_or(0) is safe
    let root_node_idx = best_node[root_canonical.0 as usize].unwrap_or(0);
    build_subtree_with_choices(egraph, root, root_node_idx, best_node)
}

// ============================================================================
// Beam Search Extraction
// ============================================================================

/// Extract using beam search with full tree evaluation at each step.
///
/// Unlike bottom-up DP which assumes optimal substructure (broken for neural costs),
/// beam search evaluates full trees from root at each decision point, keeping
/// the top-k candidates.
///
/// ## Algorithm
///
/// 1. Start with root e-class, try all node choices → k candidates
/// 2. For each candidate, find next unassigned e-class (BFS from root)
/// 3. Expand: try all node choices for that e-class
/// 4. Evaluate full trees (using choice 0 for still-unassigned classes)
/// 5. Keep top-k by neural cost
/// 6. Repeat until all reachable e-classes are assigned
///
/// ## Parameters
///
/// - `beam_width`: Number of candidates to keep at each step (k)
pub fn extract_beam(
    egraph: &EGraph,
    root: EClassId,
    nnue: &ExprNnue,
    beam_width: usize,
) -> (ExprTree, f32) {
    let num_classes = egraph.num_classes();

    // A candidate is a partial assignment of e-class → node index
    // None means "not yet decided, use default (0)"
    type Choices = Vec<Option<usize>>;

    // Initialize beam with all choices for root e-class
    let root_canonical = egraph.find(root);
    let root_nodes = egraph.nodes(root_canonical);

    let mut beam: Vec<(Choices, f32)> = Vec::with_capacity(beam_width);

    for node_idx in 0..root_nodes.len() {
        let mut choices: Choices = vec![None; num_classes];
        choices[root_canonical.0 as usize] = Some(node_idx);

        // Evaluate full tree using NNUE value head (legacy path that judge training uses)
        if let Some(tree) = build_tree_with_partial_choices(egraph, root, &choices) {
            let expr = expr_tree_to_nnue(&tree);
            // predict_log_cost uses value_w/value_b which judge training updates
            let cost = nnue.predict_log_cost(&expr);
            beam.push((choices, cost));
        }
    }

    // Sort and truncate to beam width (lower log-cost = better)
    beam.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
    beam.truncate(beam_width);

    // If beam is empty, the e-graph has issues - panic with details
    if beam.is_empty() {
        panic!(
            "extract_beam: no valid trees from root {:?}. E-graph has {} classes, root has {} nodes",
            root, num_classes, root_nodes.len()
        );
    }

    // Iteratively expand until no more unassigned e-classes
    loop {
        // Find the next unassigned e-class reachable from root (any candidate)
        let next_class = find_next_unassigned(egraph, root, &beam[0].0);

        let next_class = match next_class {
            Some(c) => c,
            None => break, // All reachable classes assigned
        };

        let next_canonical = egraph.find(next_class);
        let next_nodes = egraph.nodes(next_canonical);

        // Expand beam: for each candidate, try all choices for next_class
        let mut new_beam: Vec<(Choices, f32)> = Vec::new();

        for (choices, _old_cost) in &beam {
            for node_idx in 0..next_nodes.len() {
                let mut new_choices = choices.clone();
                new_choices[next_canonical.0 as usize] = Some(node_idx);

                // Evaluate full tree using NNUE value head (legacy path)
                if let Some(tree) = build_tree_with_partial_choices(egraph, root, &new_choices) {
                    let expr = expr_tree_to_nnue(&tree);
                    let cost = nnue.predict_log_cost(&expr);
                    new_beam.push((new_choices, cost));
                }
            }
        }

        if new_beam.is_empty() {
            break; // No valid expansions (shouldn't happen)
        }

        // Sort and truncate
        new_beam.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
        new_beam.truncate(beam_width);
        beam = new_beam;
    }

    // Best candidate is beam[0]
    let (best_choices, best_cost) = &beam[0];
    let tree = build_tree_with_partial_choices(egraph, root, best_choices)
        .unwrap_or_else(|| ExprTree::Leaf(Leaf::Const(0.0)));

    (tree, *best_cost)
}

/// Find the next unassigned e-class reachable from root in BFS order.
fn find_next_unassigned(
    egraph: &EGraph,
    root: EClassId,
    choices: &[Option<usize>],
) -> Option<EClassId> {
    use alloc::collections::VecDeque;
    use alloc::collections::BTreeSet;

    let mut visited: BTreeSet<u32> = BTreeSet::new();
    let mut queue: VecDeque<EClassId> = VecDeque::new();

    queue.push_back(root);

    while let Some(class) = queue.pop_front() {
        let canonical = egraph.find(class);

        if !visited.insert(canonical.0) {
            continue;
        }

        // If this class is unassigned, return it
        if choices[canonical.0 as usize].is_none() {
            return Some(canonical);
        }

        // Otherwise, add its children to the queue
        let node_idx = choices[canonical.0 as usize].unwrap_or(0);
        let node = &egraph.nodes(canonical)[node_idx];

        if let ENode::Op { children, .. } = node {
            for &child in children {
                queue.push_back(child);
            }
        }
    }

    None
}

/// Build a tree using partial choices (None → use node 0).
fn build_tree_with_partial_choices(
    egraph: &EGraph,
    root: EClassId,
    choices: &[Option<usize>],
) -> Option<ExprTree> {
    use alloc::collections::BTreeSet;

    enum BuildTask {
        Visit { class: EClassId },
        Complete { op: &'static dyn crate::egraph::ops::Op, num_children: usize },
        PopPath { class_id: u32 },  // Remove from path when done with subtree
    }

    let mut stack: Vec<BuildTask> = vec![BuildTask::Visit { class: root }];
    let mut result_stack: Vec<ExprTree> = Vec::new();
    let mut on_path: BTreeSet<u32> = BTreeSet::new();  // Current path, not all visited

    while let Some(task) = stack.pop() {
        match task {
            BuildTask::Visit { class } => {
                let canonical = egraph.find(class);

                // Cycle detection: are we already on the current path?
                if on_path.contains(&canonical.0) {
                    return None;  // Cycle in this specific tree
                }

                // Use choice if assigned, else default to 0
                let node_idx = choices[canonical.0 as usize].unwrap_or(0);
                let nodes = egraph.nodes(canonical);

                if node_idx >= nodes.len() {
                    return None; // Invalid choice
                }

                let node = &nodes[node_idx];

                match node {
                    ENode::Var(idx) => {
                        result_stack.push(ExprTree::Leaf(Leaf::Var(*idx)));
                    }
                    ENode::Const(bits) => {
                        result_stack.push(ExprTree::Leaf(Leaf::Const(f32::from_bits(*bits))));
                    }
                    ENode::Op { op, children } => {
                        // Add to path before visiting children
                        on_path.insert(canonical.0);

                        // Pop from path after all children are done
                        stack.push(BuildTask::PopPath { class_id: canonical.0 });
                        stack.push(BuildTask::Complete { op: *op, num_children: children.len() });

                        for &child in children.iter().rev() {
                            stack.push(BuildTask::Visit { class: child });
                        }
                    }
                }
            }
            BuildTask::Complete { op, num_children } => {
                let start = result_stack.len().saturating_sub(num_children);
                let child_trees: Vec<ExprTree> = result_stack.drain(start..).collect();
                result_stack.push(ExprTree::Op { op, children: child_trees });
            }
            BuildTask::PopPath { class_id } => {
                on_path.remove(&class_id);
            }
        }
    }

    result_stack.pop()
}

// ============================================================================
// Expression Conversion
// ============================================================================

/// Insert an `Expr` into the e-graph, returning the root e-class.
///
/// Uses iterative traversal with explicit stack to avoid thread stack overflow.
pub fn expr_to_egraph(expr: &Expr, egraph: &mut EGraph) -> EClassId {
    enum Task<'a> {
        Visit(&'a Expr),
        CompleteOp { kind: OpKind, arity: usize },
        CompleteTuple { arity: usize },
    }

    let mut stack: Vec<Task<'_>> = vec![Task::Visit(expr)];
    let mut result_stack: Vec<EClassId> = Vec::new();

    while let Some(task) = stack.pop() {
        match task {
            Task::Visit(node) => match node {
                Expr::Var(idx) => result_stack.push(egraph.add(ENode::Var(*idx))),
                Expr::Const(val) => result_stack.push(egraph.add(ENode::Const(val.to_bits()))),
                Expr::Param(i) => panic!("Expr::Param({i}) reached NNUE cost model — call substitute_params before use"),
                Expr::Unary(kind, a) => {
                    stack.push(Task::CompleteOp { kind: *kind, arity: 1 });
                    stack.push(Task::Visit(a));
                }
                Expr::Binary(kind, a, b) => {
                    stack.push(Task::CompleteOp { kind: *kind, arity: 2 });
                    stack.push(Task::Visit(b));
                    stack.push(Task::Visit(a));
                }
                Expr::Ternary(kind, a, b, c) => {
                    stack.push(Task::CompleteOp { kind: *kind, arity: 3 });
                    stack.push(Task::Visit(c));
                    stack.push(Task::Visit(b));
                    stack.push(Task::Visit(a));
                }
                Expr::Nary(kind, children) => {
                    match kind {
                        OpKind::Tuple => {
                            if children.is_empty() {
                                panic!("expr_to_egraph: empty Tuple not supported");
                            }
                            stack.push(Task::CompleteTuple { arity: children.len() });
                            for child in children.iter().rev() {
                                stack.push(Task::Visit(child));
                            }
                        }
                        _ => panic!("Unsupported n-ary op type: {:?}", kind),
                    }
                }
            },
            Task::CompleteOp { kind, arity } => {
                let start = result_stack.len().saturating_sub(arity);
                let child_classes: Vec<EClassId> = result_stack.drain(start..).collect();
                let op_ref = nnue_to_op(kind)
                    .unwrap_or_else(|| panic!("Unsupported op: {:?}", kind));
                result_stack.push(egraph.add(ENode::Op {
                    op: op_ref,
                    children: child_classes,
                }));
            }
            Task::CompleteTuple { arity } => {
                // Tuple: flatten to first element (tuples aren't fully supported in e-graph yet)
                let start = result_stack.len().saturating_sub(arity);
                let child_classes: Vec<EClassId> = result_stack.drain(start..).collect();
                result_stack.push(child_classes[0]);
            }
        }
    }

    result_stack.pop().unwrap_or_else(|| panic!("expr_to_egraph: empty result stack"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expr_equals(a: &Expr, b: &Expr) -> bool {
        match (a, b) {
            (Expr::Var(i), Expr::Var(j)) => i == j,
            (Expr::Const(x), Expr::Const(y)) => (x - y).abs() < 1e-6,
            (Expr::Unary(op1, a1), Expr::Unary(op2, a2)) => op1 == op2 && expr_equals(a1, a2),
            (Expr::Binary(op1, a1, b1), Expr::Binary(op2, a2, b2)) => {
                op1 == op2 && expr_equals(a1, a2) && expr_equals(b1, b2)
            }
            (Expr::Ternary(op1, a1, b1, c1), Expr::Ternary(op2, a2, b2, c2)) => {
                op1 == op2 && expr_equals(a1, a2) && expr_equals(b1, b2) && expr_equals(c1, c2)
            }
            (Expr::Nary(op1, c1), Expr::Nary(op2, c2)) => {
                op1 == op2 && c1.len() == c2.len() &&
                c1.iter().zip(c2.iter()).all(|(x, y)| expr_equals(x, y))
            }
            _ => false,
        }
    }

    #[test]
    fn op_to_nnue_roundtrip_should_succeed_when_called() {
        let ops_to_test: &[&dyn crate::egraph::ops::Op] = &[
            &ops::Add,
            &ops::Sub,
            &ops::Mul,
            &ops::Div,
            &ops::Neg,
            &ops::Min,
            &ops::Max,
            &ops::Sqrt,
            &ops::Rsqrt,
            &ops::Abs,
            &ops::MulAdd,
        ];
        for op in ops_to_test {
            let nnue_op = op_to_nnue(*op);
            let back = nnue_to_op(nnue_op);
            assert!(back.is_some(), "Roundtrip failed for {}", op.name());
            assert_eq!(back.expect("Expected value but got None/Err").name(), op.name(), "Roundtrip failed for {}", op.name());
        }
    }

    #[test]
    fn eclass_to_expr_leaf_should_succeed_when_called() {
        let mut egraph = EGraph::new();
        let var_class = egraph.add(ENode::Var(0));
        let expr = eclass_to_expr(&egraph, var_class);
        assert!(matches!(expr, Expr::Var(0)));
    }

    #[test]
    fn roundtrip_simple_should_succeed_when_called() {
        let expr = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        );
        let mut egraph = EGraph::new();
        let class = expr_to_egraph(&expr, &mut egraph);
        let recovered = eclass_to_expr(&egraph, class);
        assert!(
            expr_equals(&expr, &recovered),
            "Roundtrip failed for simple binary expression"
        );
    }

    #[test]
    fn roundtrip_nested_should_succeed_when_called() {
        // (x * 2.0) + y
        let expr = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Binary(
                OpKind::Mul,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(2.0)),
            )),
            Box::new(Expr::Var(1)),
        );
        let mut egraph = EGraph::new();
        let class = expr_to_egraph(&expr, &mut egraph);
        let recovered = eclass_to_expr(&egraph, class);
        assert!(
            expr_equals(&expr, &recovered),
            "Roundtrip failed for nested expression"
        );
    }

    #[test]
    fn predict_tree_cost_should_succeed_when_called() {
        let nnue = ExprNnue::new_with_latency_prior(42);

        // Simple tree: x + y
        let tree = ExprTree::Op {
            op: &ops::Add,
            children: vec![
                ExprTree::Leaf(Leaf::Var(0)),
                ExprTree::Leaf(Leaf::Var(1)),
            ],
        };

        let cost = predict_tree_cost(&tree, &nnue);
        assert!(cost.is_finite(), "Cost should be finite");
        assert!(cost > 0.0, "Cost should be positive");
    }

}
