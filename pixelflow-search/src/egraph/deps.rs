//! Dependency / Variance Analysis for E-Graphs
//!
//! Tracks which coordinate variables each e-class depends on via a 4-bit
//! bitset (`Variance`). This enables:
//! - **Loop-invariant hoisting**: X-invariant expressions computed once per scanline
//! - **Frame-uniform hoisting**: Z/W-only expressions computed once per frame
//! - **Variance-aware extraction**: NNUE can see variance when picking cheapest form
//!
//! # Variance Bitset
//!
//! Each e-class gets a `Variance` annotation: `{X?, Y?, Z?, W?}`.
//! - `Var(0)` → `{X}`, `Var(1)` → `{Y}`, `Var(2)` → `{Z}`, `Var(3)` → `{W}`
//! - `BinaryOp(a, b)` → `variance(a).union(variance(b))`
//! - Across e-nodes in the same class: `meet` (pick lowest-deps representative)

use std::collections::{HashMap, VecDeque};

use pixelflow_ir::Variance;

use super::graph::EGraph;
use super::node::{EClassId, ENode};

/// Legacy type alias for backward compatibility.
/// New code should use `Variance` directly.
pub type Deps = Variance;

/// Convert a variable index to its variance (single-bit set).
///
/// 0→X, 1→Y, 2→Z, 3→W. Unknown indices are conservatively ALL.
#[inline]
#[must_use]
pub fn var_variance(v: u8) -> Variance {
    if v < 4 {
        Variance::from_var(v)
    } else {
        Variance::ALL // Unknown vars are conservatively all-varying
    }
}

/// Legacy alias.
#[inline]
#[must_use]
pub fn var_deps(v: u8) -> Variance {
    var_variance(v)
}

/// Dependency analysis results for an e-graph.
pub struct DepsAnalysis {
    /// Map from e-class to its variance (coordinate dependencies).
    /// Uses canonical (root) e-class IDs.
    deps: HashMap<EClassId, Variance>,
}

impl DepsAnalysis {
    /// Analyze variance using BFS from leaves.
    ///
    /// For each e-class, we compute variance for each e-node independently
    /// (unioning children variance within a node), then take the MEET across
    /// all e-nodes in the class. This reflects the best available implementation:
    /// if any representation has fewer deps, the class can be computed with those.
    ///
    /// - WITHIN a single e-node: children variance is UNIONED — an add
    ///   depending on X and Y depends on both.
    /// - ACROSS e-nodes in the same e-class: variance is MEET'd — if one
    ///   representation has fewer deps, the class can use that representation.
    ///
    /// This is O(n) since deps only flow upward through the DAG.
    #[must_use]
    pub fn analyze(egraph: &EGraph) -> Self {
        let mut deps: HashMap<EClassId, Variance> = HashMap::new();
        let mut in_degree: HashMap<EClassId, usize> = HashMap::new();
        let mut dependents: HashMap<EClassId, Vec<EClassId>> = HashMap::new();

        // Collect canonical class IDs (deduplicate after union-find)
        let mut seen_classes = std::collections::HashSet::new();

        // Build reverse graph and compute in-degrees
        for idx in 0..egraph.classes.len() {
            let id = EClassId(idx as u32);
            let id = egraph.find(id);

            // Skip duplicate canonical IDs
            if !seen_classes.insert(id) {
                continue;
            }

            // Collect the union of all children across all e-nodes.
            // This determines when the class is "ready" for evaluation.
            let mut children_set = std::collections::HashSet::new();
            // Track whether any node is a leaf (has known deps immediately)
            let mut has_leaf = false;

            for node in egraph.nodes(id) {
                let (node_deps, children) = Self::leaf_deps_and_children(node);
                if node_deps.is_some() {
                    has_leaf = true;
                }
                for child in children {
                    let child = egraph.find(child);
                    if child != id {
                        children_set.insert(child);
                    }
                }
            }

            // Store in-degree (number of unique children we must wait for)
            let deg = children_set.len();
            in_degree.insert(id, deg);

            // Build reverse edges
            for child in children_set {
                dependents.entry(child).or_default().push(id);
            }

            // If all children are resolved (in_degree 0), compute deps now.
            // For pure leaf classes this is immediate; for classes with a mix
            // of leaf and non-leaf nodes, leaf nodes still seed the computation.
            if deg == 0 {
                let class_deps = Self::compute_class_deps(egraph, id, &deps);
                deps.insert(id, class_deps);
            } else if has_leaf {
                // Pre-seed with leaf minimum so BFS propagation can refine.
                // This will be recomputed properly when in_degree reaches 0.
                // (Not strictly necessary, but documents intent.)
            }
        }

        // BFS from leaves (in_degree == 0)
        let mut queue: VecDeque<EClassId> = in_degree
            .iter()
            .filter(|&(_, deg)| *deg == 0)
            .map(|(&id, _)| id)
            .collect();

        while let Some(id) = queue.pop_front() {
            // Propagate to dependents
            if let Some(parents) = dependents.get(&id) {
                for &parent in parents {
                    // Decrement in-degree
                    if let Some(deg) = in_degree.get_mut(&parent) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            // All children resolved — compute this class's deps
                            // by examining each e-node independently and taking min.
                            let class_deps = Self::compute_class_deps(egraph, parent, &deps);
                            deps.insert(parent, class_deps);
                            queue.push_back(parent);
                        }
                    }
                }
            }
        }

        Self { deps }
    }

    /// Compute variance for an e-class by examining each e-node independently.
    ///
    /// For each e-node, union its children's variance. Then take the meet across
    /// all e-nodes in the class — the class can be realized via whichever
    /// representation has the fewest dependencies.
    fn compute_class_deps(
        egraph: &EGraph,
        id: EClassId,
        resolved: &HashMap<EClassId, Variance>,
    ) -> Variance {
        let mut class_deps = Variance::ALL; // Start pessimistic, take meet

        for node in egraph.nodes(id) {
            let node_v = Self::node_deps(egraph, id, node, resolved);
            class_deps = class_deps.meet(node_v);
        }

        class_deps
    }

    /// Compute variance for a single e-node by unioning its children's variance.
    fn node_deps(
        egraph: &EGraph,
        self_id: EClassId,
        node: &ENode,
        resolved: &HashMap<EClassId, Variance>,
    ) -> Variance {
        match node {
            ENode::Const(_) => Variance::CONST,
            ENode::Var(v) => var_variance(*v),
            ENode::Op { children, .. } => {
                let mut v = Variance::CONST; // Identity for union
                for &child in children {
                    let child = egraph.find(child);
                    if child == self_id {
                        // Self-referential — skip to avoid infinite loop.
                        // Conservative: treat as all-varying.
                        return Variance::ALL;
                    }
                    let child_v = resolved.get(&child).copied().unwrap_or(Variance::ALL);
                    v = v.union(child_v);
                }
                v
            }
        }
    }

    /// Get leaf variance (if leaf) and children for a node.
    fn leaf_deps_and_children(node: &ENode) -> (Option<Variance>, Vec<EClassId>) {
        match node {
            ENode::Const(_) => (Some(Variance::CONST), vec![]),
            ENode::Var(v) => (Some(var_variance(*v)), vec![]),
            ENode::Op { children, .. } => (None, children.clone()),
        }
    }

    /// Get the variance for an e-class.
    #[must_use]
    pub fn get(&self, egraph: &EGraph, id: EClassId) -> Variance {
        let id = egraph.find(id);
        self.deps.get(&id).copied().unwrap_or(Variance::ALL)
    }

    /// Find all X-invariant subexpressions that should be hoisted
    /// out of the pixel loop.
    ///
    /// Returns e-class IDs of expressions that:
    /// 1. Are X-invariant (don't depend on X)
    /// 2. Are non-const (worth computing at runtime)
    /// 3. Are used by X-dependent expressions (worth hoisting)
    #[must_use]
    pub fn find_hoistable(&self, egraph: &EGraph, root: EClassId) -> Vec<EClassId> {
        let mut hoistable = Vec::new();
        let mut visited = std::collections::HashSet::new();

        self.find_hoistable_recursive(egraph, root, &mut hoistable, &mut visited);

        hoistable
    }

    fn find_hoistable_recursive(
        &self,
        egraph: &EGraph,
        id: EClassId,
        hoistable: &mut Vec<EClassId>,
        visited: &mut std::collections::HashSet<EClassId>,
    ) {
        let id = egraph.find(id);
        if !visited.insert(id) {
            return;
        }

        let my_v = self.get(egraph, id);

        // Visit children
        for node in egraph.nodes(id) {
            for child_id in node.children() {
                let child_v = self.get(egraph, child_id);

                // If I depend on X and child doesn't (and child isn't const), child is hoistable
                if my_v.depends_on_x() && child_v.is_x_invariant() && !child_v.is_const() {
                    hoistable.push(egraph.find(child_id));
                }

                // Continue recursion
                self.find_hoistable_recursive(egraph, child_id, hoistable, visited);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::ops;
    use super::*;

    #[test]
    fn variance_union_should_succeed_when_invoked() {
        assert_eq!(Variance::CONST.union(Variance::CONST), Variance::CONST);
        assert_eq!(Variance::CONST.union(Variance::W), Variance::W);
        assert_eq!(Variance::CONST.union(Variance::X), Variance::X);
        assert_eq!(Variance::W.union(Variance::W), Variance::W);
        assert_eq!(
            Variance::W.union(Variance::X),
            Variance::X.union(Variance::W)
        );
        assert_eq!(Variance::X.union(Variance::X), Variance::X);
    }

    #[test]
    fn var_variance_should_succeed_when_invoked() {
        assert_eq!(var_variance(0), Variance::X);
        assert_eq!(var_variance(1), Variance::Y);
        assert_eq!(var_variance(2), Variance::Z);
        assert_eq!(var_variance(3), Variance::W);
    }

    #[test]
    fn const_analysis_should_succeed_when_invoked() {
        let mut eg = EGraph::new();
        let c = eg.add(ENode::constant(42.0));

        let analysis = DepsAnalysis::analyze(&eg);
        assert_eq!(analysis.get(&eg, c), Variance::CONST);
    }

    #[test]
    fn w_only_analysis_should_succeed_when_invoked() {
        let mut eg = EGraph::new();
        let w = eg.add(ENode::Var(3)); // W
        let two = eg.add(ENode::constant(2.0));
        let w_times_2 = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![w, two],
        });
        let sin_w = eg.add(ENode::Op {
            op: &ops::Sin,
            children: vec![w_times_2],
        });

        let analysis = DepsAnalysis::analyze(&eg);
        assert_eq!(analysis.get(&eg, w), Variance::W);
        assert_eq!(analysis.get(&eg, w_times_2), Variance::W);
        assert_eq!(analysis.get(&eg, sin_w), Variance::W);
        // All are frame-uniform (no spatial deps)
        assert!(analysis.get(&eg, sin_w).is_frame_uniform());
    }

    #[test]
    fn xy_analysis_should_succeed_when_invoked() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(0)); // X
        let y = eg.add(ENode::Var(1)); // Y
        let sum = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, y],
        });

        let analysis = DepsAnalysis::analyze(&eg);
        assert_eq!(analysis.get(&eg, x), Variance::X);
        assert_eq!(analysis.get(&eg, y), Variance::Y);
        assert_eq!(analysis.get(&eg, sum), Variance::X.union(Variance::Y));
        assert!(analysis.get(&eg, sum).depends_on_x());
        assert!(analysis.get(&eg, sum).depends_on_y());
        assert!(!analysis.get(&eg, sum).depends_on_z());
    }

    #[test]
    fn mixed_deps_hoistable_should_succeed_when_invoked() {
        // sin(W * 2.0) + X:
        //   sin(W * 2.0) has variance {W} (X-invariant, hoistable)
        //   X has variance {X}
        //   result has variance {X, W}
        let mut eg = EGraph::new();
        let w = eg.add(ENode::Var(3));
        let two = eg.add(ENode::constant(2.0));
        let w2 = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![w, two],
        });
        let sin_w2 = eg.add(ENode::Op {
            op: &ops::Sin,
            children: vec![w2],
        });

        let x = eg.add(ENode::Var(0));
        let result = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![sin_w2, x],
        });

        let analysis = DepsAnalysis::analyze(&eg);
        assert_eq!(analysis.get(&eg, sin_w2), Variance::W);
        assert!(analysis.get(&eg, sin_w2).is_x_invariant());
        assert_eq!(analysis.get(&eg, result), Variance::X.union(Variance::W));

        // sin(W * 2) should be hoistable (X-invariant child of X-dependent parent)
        let hoistable = analysis.find_hoistable(&eg, result);
        assert!(hoistable.contains(&eg.find(sin_w2)));
    }

    #[test]
    fn y_only_is_hoistable_should_succeed_when_invoked() {
        // sin(Y) + X: sin(Y) has variance {Y}, is X-invariant, should be hoistable
        let mut eg = EGraph::new();
        let y = eg.add(ENode::Var(1));
        let sin_y = eg.add(ENode::Op {
            op: &ops::Sin,
            children: vec![y],
        });
        let x = eg.add(ENode::Var(0));
        let result = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![sin_y, x],
        });

        let analysis = DepsAnalysis::analyze(&eg);
        assert_eq!(analysis.get(&eg, sin_y), Variance::Y);
        assert!(analysis.get(&eg, sin_y).is_x_invariant());

        let hoistable = analysis.find_hoistable(&eg, result);
        assert!(
            hoistable.contains(&eg.find(sin_y)),
            "sin(Y) should be hoistable out of the X loop"
        );
    }

    /// Regression test: after discovering X - X = 0, the e-class containing
    /// both Sub(X, X) and Const(0) should have Const variance.
    ///
    /// The analysis must take the MEET across e-nodes in a class:
    /// - Sub(X, X) has variance = union({X}, {X}) = {X}
    /// - Const(0)  has variance = {}
    /// - class variance = meet({X}, {}) = {} (CONST)
    #[test]
    fn meet_across_enodes_x_minus_x_is_const_should_succeed_when_invoked() {
        let mut eg = EGraph::new();

        let x = eg.add(ENode::Var(0));
        let sub_xx = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![x, x],
        });

        // Before rewrite: Sub(X, X) depends on X
        let before = DepsAnalysis::analyze(&eg);
        assert_eq!(before.get(&eg, sub_xx), Variance::X);

        // Simulate X - X => 0
        let zero = eg.add(ENode::constant(0.0));
        eg.union(sub_xx, zero);
        eg.rebuild();

        // After: meet({X}, {}) = CONST
        let after = DepsAnalysis::analyze(&eg);
        assert_eq!(
            after.get(&eg, eg.find(sub_xx)),
            Variance::CONST,
            "After X - X = 0 rewrite, e-class should have CONST variance"
        );
    }

    /// Meet of {X,Y} and {W} should give {W} (fewer deps).
    #[test]
    fn meet_xy_and_w_gives_w_should_succeed_when_invoked() {
        let mut eg = EGraph::new();

        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let xy = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, y],
        });

        let w = eg.add(ENode::Var(3));

        eg.union(xy, w);
        eg.rebuild();

        let analysis = DepsAnalysis::analyze(&eg);
        let merged = eg.find(xy);
        assert_eq!(
            analysis.get(&eg, merged),
            Variance::W,
            "Meet of {{X,Y}} and {{W}} should be {{W}} (fewer deps)"
        );
    }

    /// Parent benefits from meet-reduced child.
    #[test]
    fn meet_propagates_to_parents_should_succeed_when_invoked() {
        let mut eg = EGraph::new();

        let x = eg.add(ENode::Var(0));
        let sub_xx = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![x, x],
        });
        let sin_sub = eg.add(ENode::Op {
            op: &ops::Sin,
            children: vec![sub_xx],
        });

        let before = DepsAnalysis::analyze(&eg);
        assert_eq!(before.get(&eg, sin_sub), Variance::X);

        let zero = eg.add(ENode::constant(0.0));
        eg.union(sub_xx, zero);
        eg.rebuild();

        let after = DepsAnalysis::analyze(&eg);
        assert_eq!(
            after.get(&eg, eg.find(sin_sub)),
            Variance::CONST,
            "sin(X - X) should become CONST after X - X = 0 rewrite"
        );
    }
}
