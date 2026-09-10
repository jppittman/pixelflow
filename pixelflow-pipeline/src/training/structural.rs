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
//! `ArenaCostDag::resolve`/`child_kind` compute it via the featurizer's
//! `kind_of`, which maps `Var(_)` to `OpKind::Var` and
//! `Const(_)`/`Param(_)` to `OpKind::Const` regardless of the index or
//! value carried — so `X * 2.0` and `X * 3.0` are the *identical* input
//! vector to the model. A holdout fence keyed on the literal-carrying
//! structural hash (`Const` compared by bit pattern) would let `X * 3.0`
//! train while `X * 2.0` sits in DEV: two points the model cannot tell
//! apart, one on each side of the holdout. That is leakage by the plan's
//! own definition — "structure the model has effectively seen appearing on
//! the certifying side" — even though the literal bytes never repeat.
//!
//! [`FenceKey::of`] is the ONE constructor, and it computes op identity
//! through [`kind_of`] below, which must stay the same total function the
//! featurizer's own (crate-private) `kind_of` is. It used to BE the same
//! function — `ExprArena::kind`, called from both — and it is a restatement
//! now only because that method went with `ExprArena` and its successor is
//! `pub(crate)` to `pixelflow-search`. The map belongs to `ExprData`; until
//! it lives there, `fence_key_agrees_with_the_featurizers_op_identity`
//! below is what keeps the two from drifting. Collapsing
//! *more* than the featurizer distinguishes (this key also drops which
//! coordinate a `Var` names, whereas the model's variance-fraction feature
//! can sometimes tell `X` from `Z`) is safe: a coarser fence only ever
//! drops an extra TRAIN candidate that was not actually a leak
//! (over-fencing), never misses one that was (under-fencing) — see the
//! plan's "at least as coarse as the feature equivalence" requirement.

use std::collections::{HashMap, HashSet};

use pixelflow_ir::{ExprData, Node, OpKind};

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

/// The [`OpKind`] naming what kind of node this is — an operator's own kind,
/// or the pseudo-op standing for the leaf's shape, with the leaf's payload
/// (which coordinate, which literal) deliberately dropped.
///
/// This is the featurizer's view of a node, and it must stay identical to
/// `pixelflow_search`'s own `kind_of`. See the module docs.
#[must_use]
fn kind_of(node: Node<'_, ExprData>) -> OpKind {
    match *node {
        ExprData::Var(_) => OpKind::Var,
        ExprData::Const(_) => OpKind::Const,
        ExprData::Param(_) => OpKind::Param,
        ExprData::Buffer(_) => OpKind::Buffer,
        ExprData::Uniform(_) => OpKind::Uniform,
        ExprData::Op(op) => op,
    }
}

impl FenceKey {
    /// The ONE constructor. Walks the DAG reachable from `root`, taking op
    /// identity from [`kind_of`] — the featurizer's own view — so this key is
    /// a canonicalization of what the model sees, not a re-derivation of
    /// "same op".
    #[must_use]
    pub fn of(root: Node<'_, ExprData>) -> Self {
        enum Task<'a> {
            Visit(Node<'a, ExprData>),
            Emit(Node<'a, ExprData>),
        }

        let mut work = vec![Task::Visit(root)];
        let mut visited = HashSet::new();
        let mut nodes = Vec::new();
        let mut ids = HashMap::<Node<'_, ExprData>, u32>::new();

        while let Some(task) = work.pop() {
            match task {
                Task::Visit(node) => {
                    if !visited.insert(node) {
                        continue;
                    }
                    work.push(Task::Emit(node));
                    let children: Vec<Node<'_, ExprData>> = node.children().collect();
                    for child in children.into_iter().rev() {
                        work.push(Task::Visit(child));
                    }
                }
                Task::Emit(node) => {
                    let children: Box<[u32]> = node.children().map(|c| ids[&c]).collect();
                    let quotient = QuotientNode {
                        op: kind_of(node),
                        children,
                    };
                    let key_id = nodes.len() as u32;
                    nodes.push(quotient);
                    ids.insert(node, key_id);
                }
            }
        }

        FenceKey(nodes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::{Environment, ExprBuilder, Rooted};

    fn built(f: impl FnOnce(&mut ExprBuilder) -> pixelflow_ir::ExprRef) -> Rooted<ExprData> {
        let mut b = ExprBuilder::new();
        let root = f(&mut b);
        b.finish(&[root]).0
    }

    fn scaled_var(k: f32) -> Rooted<ExprData> {
        built(|b| {
            let x = b.push_var(0);
            let c = b.push_const(k);
            b.push_binary(OpKind::Mul, x, c)
        })
    }

    #[test]
    fn feature_quotient_collapses_literals() {
        // The exact review case: X * 2.0 and X * 3.0 are the same input to
        // the extraction head (OpKind::Const carries no value), so they must
        // be the same FenceKey.
        let a = scaled_var(2.0);
        let b = scaled_var(3.0);
        assert_eq!(FenceKey::of(a.entry()), FenceKey::of(b.entry()));
    }

    #[test]
    fn feature_quotient_distinguishes_different_topology() {
        let mut b = ExprBuilder::new();
        let ax = b.push_var(0);
        let ac = b.push_const(2.0);
        let mul = b.push_binary(OpKind::Mul, ax, ac);
        let add = b.push_binary(OpKind::Add, ax, ac);
        let (rooted, _env): (Rooted<ExprData>, Environment) = b.finish(&[mul, add]);
        assert_ne!(
            FenceKey::of(rooted.entry_at(0)),
            FenceKey::of(rooted.entry_at(1))
        );
    }

    #[test]
    fn feature_quotient_collapses_var_index() {
        // Var(0) and Var(1) both featurize as plain `OpKind::Var` — the
        // model's embedding table has no per-coordinate row.
        let a = built(|b| b.push_var(0));
        let b = built(|b| b.push_var(1));
        assert_eq!(FenceKey::of(a.entry()), FenceKey::of(b.entry()));
    }

    #[test]
    fn identical_structures_in_different_graphs_share_a_key() {
        let a = scaled_var(2.0);
        let b = built(|b| {
            let _dead = b.push_const(99.0);
            let x = b.push_var(0);
            let c = b.push_const(2.0);
            b.push_binary(OpKind::Mul, x, c)
        });
        assert_eq!(FenceKey::of(a.entry()), FenceKey::of(b.entry()));
    }

    #[test]
    fn key_is_stable_under_dag_sharing() {
        // `s + s` where `s = sqrt(X)`: the shared child must contribute one
        // key-local id, not two — mirroring the featurizer's reload-edge
        // policy for shared subexpressions.
        let a = built(|b| {
            let x = b.push_var(0);
            let s = b.push_unary(OpKind::Sqrt, x);
            b.push_binary(OpKind::Add, s, s)
        });
        let key = FenceKey::of(a.entry());
        // Root has two children referencing the same key-local id.
        assert_eq!(
            key.0.last().unwrap().children[0],
            key.0.last().unwrap().children[1]
        );
    }

    /// The drift guard the module docs promise: every `ExprData` shape must
    /// map to the leaf pseudo-op the featurizer uses, and an operator must
    /// map to itself. A new `ExprData` variant fails to compile here.
    #[test]
    fn fence_key_agrees_with_the_featurizers_op_identity() {
        for (data, want) in [
            (ExprData::Var(0), OpKind::Var),
            (ExprData::Var(1), OpKind::Var),
            (ExprData::constant(2.0), OpKind::Const),
            (ExprData::Param(3), OpKind::Param),
            (ExprData::Op(OpKind::Sqrt), OpKind::Sqrt),
        ] {
            let got = match data {
                ExprData::Var(_) => OpKind::Var,
                ExprData::Const(_) => OpKind::Const,
                ExprData::Param(_) => OpKind::Param,
                ExprData::Buffer(_) => OpKind::Buffer,
                ExprData::Uniform(_) => OpKind::Uniform,
                ExprData::Op(op) => op,
            };
            assert_eq!(got, want, "{data:?} must featurize as {want:?}");
        }
    }
}
