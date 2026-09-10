//! The feature-quotient structural key of an arena expression.
//!
//! One key per expression DAG, shared by every stage that must agree on
//! "the model has already seen this" (docs/plans/2026-08-17-cost-model-domain.md,
//! J8 / P1(d)): `gen_bench_corpus`'s cross-tier dedup ledger today, and any
//! trainer's holdout fence tomorrow. Keeping the key in one place is the
//! point: two divergent notions of "the same expression" would re-open
//! exactly the leak this module exists to close.
//!
//! # Why "feature-quotient" and not "structural"
//!
//! `pixelflow_search`'s extraction head never sees a literal. Every feature
//! path funnels through [`OpKind`] identity —
//! `ArenaCostDag::resolve`/`child_kind` compute it via
//! [`ExprArena::kind`], which maps `Var(_)` to `OpKind::Var` and
//! `Const(_)`/`Param(_)` to `OpKind::Const` regardless of the index or
//! value carried — so `X * 2.0` and `X * 3.0` are the *identical* input
//! vector to the model. A holdout fence keyed on the literal-carrying
//! structural hash (`Const` compared by bit pattern) would let `X * 3.0`
//! train while `X * 2.0` sits in DEV: two points the model cannot tell
//! apart, one on each side of the holdout. That is leakage by the plan's
//! own definition — "structure the model has effectively seen appearing on
//! the certifying side" — even though the literal bytes never repeat.
//!
//! [`FenceKey::of`] is the ONE constructor, and it computes op identity by
//! calling [`ExprArena::kind`] — the exact function the featurizer calls —
//! rather than re-deriving a parallel notion of "same op". Collapsing
//! *more* than the featurizer distinguishes (this key also drops which
//! coordinate a `Var` names, whereas the model's variance-fraction feature
//! can sometimes tell `X` from `Z`) is safe: a coarser fence only ever
//! drops an extra TRAIN candidate that was not actually a leak
//! (over-fencing), never misses one that was (under-fencing) — see the
//! plan's "at least as coarse as the feature equivalence" requirement.

use std::collections::{HashMap, HashSet};

use pixelflow_ir::{ExprArena, ExprData, ExprId, Node, OpKind};

/// One node of a [`FenceKey`]: the node's [`OpKind`] identity plus the
/// key-local ids of its children, in deterministic post-order. No literal
/// payload — see the module docs for why.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct QuotientNode {
    op: OpKind,
    children: Box<[u32]>,
}

/// The feature-quotient key of the DAG reachable from `root`: what the
/// expression looks like to the extraction head, not what it literally is.
/// Two arenas holding the same op-identity/topology produce equal keys
/// regardless of node numbering, dead nodes, or literal values (`Const`
/// bit pattern, `Var`/`Param` index).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FenceKey(Vec<QuotientNode>);

impl FenceKey {
    /// The ONE constructor. Walks the DAG reachable from `root`, calling
    /// [`ExprArena::kind`] for op identity — the same call the
    /// extraction-head featurizer makes — so this key is a canonicalization
    /// of the featurizer's own view, not a re-implementation of it.
    #[must_use]
    pub fn of(arena: &ExprArena, root: ExprId) -> Self {
        enum Task {
            Visit(ExprId),
            Emit(ExprId),
        }

        let mut work = vec![Task::Visit(root)];
        let mut visited = HashSet::new();
        let mut nodes = Vec::new();
        let mut ids = HashMap::<ExprId, u32>::new();

        while let Some(task) = work.pop() {
            match task {
                Task::Visit(id) => {
                    if !visited.insert(id) {
                        continue;
                    }
                    work.push(Task::Emit(id));
                    let children: Vec<ExprId> = arena.children(id).collect();
                    for child in children.into_iter().rev() {
                        work.push(Task::Visit(child));
                    }
                }
                Task::Emit(id) => {
                    let children: Box<[u32]> = arena.children(id).map(|c| ids[&c]).collect();
                    let node = QuotientNode {
                        op: arena.kind(id),
                        children,
                    };
                    let key_id = nodes.len() as u32;
                    nodes.push(node);
                    ids.insert(id, key_id);
                }
            }
        }

        FenceKey(nodes)
    }

    /// Build the same feature-quotient key from a rooted DAG node.
    ///
    /// The traversal follows the DAG's child-before-parent iteration order;
    /// no storage index or arena layout is observed.  This is the preferred
    /// entry point for corpus and holdout consumers.  `Ref` remains a hard
    /// error for the same reason as [`Self::of`]: references have no local
    /// operation identity and must be expanded before structural analysis.
    #[must_use]
    pub fn of_dag(root: Node<'_, ExprData>) -> Self {
        let reachable: HashSet<Node<'_, ExprData>> = root.descendants().collect();
        let mut ids = HashMap::<Node<'_, ExprData>, u32>::new();
        let mut nodes = Vec::new();
        for node in root.dag().iter() {
            if !reachable.contains(&node) {
                continue;
            }
            let op = match *node {
                ExprData::Var(_) => OpKind::Var,
                ExprData::Const(_) | ExprData::Param(_) => OpKind::Const,
                ExprData::Buffer(_) => OpKind::Buffer,
                ExprData::Uniform(_) => OpKind::Uniform,
                ExprData::Ref(key) => panic!(
                    "FenceKey::of_dag: {key:?} is a reference to a kernel, not an operation; \
                     run passes::expand_refs before asking for a structural key"
                ),
                ExprData::Op(op) => op,
                ExprData::Reduce(_) => OpKind::Reduce,
            };
            let children: Box<[u32]> = node
                .children()
                .map(|child| {
                    *ids.get(&child)
                        .expect("FenceKey::of_dag: child must precede parent")
                })
                .collect();
            let key_id = nodes.len() as u32;
            nodes.push(QuotientNode { op, children });
            ids.insert(node, key_id);
        }
        assert!(
            ids.contains_key(&root),
            "FenceKey::of_dag: root missing from reachable DAG"
        );
        FenceKey(nodes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scaled_var(k: f32) -> (ExprArena, ExprId) {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let c = arena.push_const(k);
        let root = arena.push_binary(OpKind::Mul, x, c);
        (arena, root)
    }

    #[test]
    fn feature_quotient_collapses_literals() {
        // The exact review case: X * 2.0 and X * 3.0 are the same input to
        // the extraction head (OpKind::Const carries no value), so they must
        // be the same FenceKey.
        let (a, ra) = scaled_var(2.0);
        let (b, rb) = scaled_var(3.0);
        assert_eq!(FenceKey::of(&a, ra), FenceKey::of(&b, rb));
    }

    #[test]
    fn feature_quotient_distinguishes_different_topology() {
        let mut a = ExprArena::new();
        let ax = a.push_var(0);
        let ac = a.push_const(2.0);
        let mul = a.push_binary(OpKind::Mul, ax, ac);
        let add = a.push_binary(OpKind::Add, ax, ac);
        assert_ne!(FenceKey::of(&a, mul), FenceKey::of(&a, add));
    }

    #[test]
    fn feature_quotient_collapses_var_index() {
        // Var(0) and Var(1) both featurize as plain `OpKind::Var` — the
        // model's embedding table has no per-coordinate row.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let mut b = ExprArena::new();
        let y = b.push_var(1);
        assert_eq!(FenceKey::of(&a, x), FenceKey::of(&b, y));
    }

    #[test]
    fn identical_structures_in_different_arenas_share_a_key() {
        let mut a = ExprArena::new();
        let ax = a.push_var(0);
        let ac = a.push_const(2.0);
        let aroot = a.push_binary(OpKind::Mul, ax, ac);

        let mut b = ExprArena::new();
        let _dead = b.push_const(99.0);
        let bx = b.push_var(0);
        let bc = b.push_const(2.0);
        let broot = b.push_binary(OpKind::Mul, bx, bc);

        assert_eq!(FenceKey::of(&a, aroot), FenceKey::of(&b, broot));
    }

    #[test]
    fn key_is_stable_under_dag_sharing() {
        // `s + s` where `s = sqrt(X)`: the shared child must contribute one
        // key-local id, not two — mirroring the featurizer's reload-edge
        // policy for shared subexpressions.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let s = a.push_unary(OpKind::Sqrt, x);
        let root = a.push_binary(OpKind::Add, s, s);
        let key = FenceKey::of(&a, root);
        // Root has two children referencing the same key-local id.
        assert_eq!(
            key.0.last().unwrap().children[0],
            key.0.last().unwrap().children[1]
        );
    }
}
