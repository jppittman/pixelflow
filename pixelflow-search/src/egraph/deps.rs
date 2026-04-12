//! Dependency Analysis for Uniform Hoisting
//!
//! Tracks which variables each e-class depends on, enabling:
//! - **Uniform hoisting**: Expressions depending only on W (time) can be
//!   computed once per frame instead of per-pixel.
//! - **Staged extraction**: Extract two programs - one for frame-time uniforms,
//!   one for per-pixel varying computation.
//!
//! # Dependency Lattice
//!
//! ```text
//!         Varying (X, Y, Z)
//!            /      \
//!       Uniform (W)  |
//!            \      /
//!           Const
//! ```
//!
//! Dependencies propagate upward: `Uniform + Varying = Varying`.

use std::collections::{HashMap, VecDeque};

use super::graph::EGraph;
use super::node::{EClassId, ENode};

/// Dependency classification for an expression.
///
/// Forms a lattice where `join(a, b)` gives the least upper bound:
/// - `join(Const, x) = x`
/// - `join(Uniform, Varying) = Varying`
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Deps {
    /// No variable dependencies - a literal constant.
    /// Can be evaluated at compile time.
    Const,

    /// Depends only on W (time/frame index).
    /// Uniform across all pixels - compute once per frame.
    Uniform,

    /// Depends on X, Y, or Z (spatial coordinates).
    /// Must be computed per-pixel.
    Varying,
}

impl Deps {
    /// Join two dependencies (least upper bound in the lattice).
    #[inline]
    #[must_use]
    pub fn join(self, other: Self) -> Self {
        // Ord is derived with Const < Uniform < Varying, so max works
        self.max(other)
    }

    /// Is this expression uniform (can be hoisted to frame-time)?
    #[inline]
    #[must_use]
    pub fn is_uniform(&self) -> bool {
        matches!(self, Deps::Const | Deps::Uniform)
    }

    /// Is this expression varying (must be computed per-pixel)?
    #[inline]
    #[must_use]
    pub fn is_varying(&self) -> bool {
        matches!(self, Deps::Varying)
    }
}

/// Variable indices in the coordinate system.
/// X=0, Y=1, Z=2, W=3 (matching pixelflow-core convention).
pub mod var {
    pub const X: u8 = 0;
    pub const Y: u8 = 1;
    pub const Z: u8 = 2;
    pub const W: u8 = 3;
}

/// Compute the dependency of a variable.
#[inline]
pub fn var_deps(v: u8) -> Deps {
    match v {
        var::X | var::Y | var::Z => Deps::Varying,
        var::W => Deps::Uniform,
        _ => Deps::Varying, // Unknown vars are conservatively varying
    }
}

/// Dependency analysis results for an e-graph.
pub struct DepsAnalysis {
    /// Map from e-class to its dependency.
    /// Uses canonical (root) e-class IDs.
    deps: HashMap<EClassId, Deps>,
}

impl DepsAnalysis {
    /// Analyze dependencies using BFS from leaves.
    ///
    /// This is O(n) since deps only flow upward through the DAG.
    #[must_use]
    pub fn analyze(egraph: &EGraph) -> Self {
        let mut deps: HashMap<EClassId, Deps> = HashMap::new();
        let mut in_degree: HashMap<EClassId, usize> = HashMap::new();
        let mut dependents: HashMap<EClassId, Vec<EClassId>> = HashMap::new();

        // Build reverse graph and compute in-degrees
        for idx in 0..egraph.classes.len() {
            let id = EClassId(idx as u32);
            let id = egraph.find(id);

            // Get minimum deps from any node in this e-class
            let mut min_deps = Deps::Varying;
            let mut children_set = std::collections::HashSet::new();

            for node in egraph.nodes(id) {
                let (node_deps, children) = Self::leaf_deps_and_children(node);
                if let Some(d) = node_deps {
                    // It's a leaf - we know its deps immediately
                    min_deps = min_deps.min(d);
                }
                for child in children {
                    let child = egraph.find(child);
                    if child != id {
                        children_set.insert(child);
                    }
                }
            }

            // Store in-degree (number of children we depend on)
            let deg = children_set.len();
            in_degree.insert(id, deg);

            // Build reverse edges
            for child in children_set {
                dependents.entry(child).or_default().push(id);
            }

            // If it's a leaf (in_degree 0), we already know deps
            if deg == 0 {
                deps.insert(id, min_deps);
            }
        }

        // BFS from leaves (in_degree == 0)
        let mut queue: VecDeque<EClassId> = in_degree
            .iter()
            .filter(|&(_, deg)| *deg == 0)
            .map(|(&id, _)| id)
            .collect();

        while let Some(id) = queue.pop_front() {
            let my_deps = deps.get(&id).copied().unwrap_or(Deps::Varying);

            // Propagate to dependents
            if let Some(parents) = dependents.get(&id) {
                for &parent in parents {
                    // Update parent's deps (join with this child's deps)
                    let parent_deps = deps.entry(parent).or_insert(Deps::Const);
                    *parent_deps = parent_deps.join(my_deps);

                    // Decrement in-degree
                    if let Some(deg) = in_degree.get_mut(&parent) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            queue.push_back(parent);
                        }
                    }
                }
            }
        }

        Self { deps }
    }

    /// Get leaf deps (if leaf) and children for a node.
    fn leaf_deps_and_children(node: &ENode) -> (Option<Deps>, Vec<EClassId>) {
        match node {
            ENode::Const(_) => (Some(Deps::Const), vec![]),
            ENode::Var(v) => (Some(var_deps(*v)), vec![]),
            ENode::Op { children, .. } => (None, children.clone()),
        }
    }

    /// Get the dependency for an e-class.
    #[must_use]
    pub fn get(&self, egraph: &EGraph, id: EClassId) -> Deps {
        let id = egraph.find(id);
        self.deps.get(&id).copied().unwrap_or(Deps::Varying)
    }

    /// Find all uniform subexpressions that should be hoisted.
    ///
    /// Returns e-class IDs of expressions that:
    /// 1. Are uniform (depend only on W or constants)
    /// 2. Are used by varying expressions (worth hoisting)
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

        let my_deps = self.get(egraph, id);

        // Visit children
        for node in egraph.nodes(id) {
            for child_id in node.children() {
                let child_deps = self.get(egraph, child_id);

                // If I'm varying and child is uniform (but not const), child is hoistable
                if my_deps.is_varying() && child_deps.is_uniform() && child_deps != Deps::Const {
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
    fn deps_lattice_should_succeed_when_called() {
        assert_eq!(Deps::Const.join(Deps::Const), Deps::Const);
        assert_eq!(Deps::Const.join(Deps::Uniform), Deps::Uniform);
        assert_eq!(Deps::Const.join(Deps::Varying), Deps::Varying);
        assert_eq!(Deps::Uniform.join(Deps::Uniform), Deps::Uniform);
        assert_eq!(Deps::Uniform.join(Deps::Varying), Deps::Varying);
        assert_eq!(Deps::Varying.join(Deps::Varying), Deps::Varying);
    }

    #[test]
    fn var_deps_should_succeed_when_called() {
        assert_eq!(var_deps(var::X), Deps::Varying);
        assert_eq!(var_deps(var::Y), Deps::Varying);
        assert_eq!(var_deps(var::Z), Deps::Varying);
        assert_eq!(var_deps(var::W), Deps::Uniform);
    }

    #[test]
    fn const_analysis_should_succeed_when_called() {
        let mut eg = EGraph::new();
        let c = eg.add(ENode::constant(42.0));

        let analysis = DepsAnalysis::analyze(&eg);
        assert_eq!(analysis.get(&eg, c), Deps::Const);
    }

    #[test]
    fn uniform_analysis_should_succeed_when_called() {
        let mut eg = EGraph::new();
        let w = eg.add(ENode::Var(var::W));
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
        assert_eq!(analysis.get(&eg, w), Deps::Uniform);
        assert_eq!(analysis.get(&eg, w_times_2), Deps::Uniform);
        assert_eq!(analysis.get(&eg, sin_w), Deps::Uniform);
    }

    #[test]
    fn varying_analysis_should_succeed_when_called() {
        let mut eg = EGraph::new();
        let x = eg.add(ENode::Var(var::X));
        let y = eg.add(ENode::Var(var::Y));
        let sum = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, y],
        });

        let analysis = DepsAnalysis::analyze(&eg);
        assert_eq!(analysis.get(&eg, x), Deps::Varying);
        assert_eq!(analysis.get(&eg, y), Deps::Varying);
        assert_eq!(analysis.get(&eg, sum), Deps::Varying);
    }

    #[test]
    fn mixed_deps_should_succeed_when_called() {
        // (W * 2.0).sin() + X should be Varying
        // but (W * 2.0).sin() should be Uniform (hoistable)
        let mut eg = EGraph::new();
        let w = eg.add(ENode::Var(var::W));
        let two = eg.add(ENode::constant(2.0));
        let w2 = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![w, two],
        });
        let sin_w2 = eg.add(ENode::Op {
            op: &ops::Sin,
            children: vec![w2],
        });

        let x = eg.add(ENode::Var(var::X));
        let result = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![sin_w2, x],
        });

        let analysis = DepsAnalysis::analyze(&eg);
        assert_eq!(analysis.get(&eg, sin_w2), Deps::Uniform);
        assert_eq!(analysis.get(&eg, result), Deps::Varying);

        // sin(W * 2) should be hoistable
        let hoistable = analysis.find_hoistable(&eg, result);
        assert!(hoistable.contains(&eg.find(sin_w2)));
    }
}
