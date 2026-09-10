//! [`ExprBuilder`] as an [`Ir`], and the generic "materialise a term
//! elsewhere" walk over one.
//!
//! An expression graph is a DAG, so `embed` expresses sharing by returning the
//! same [`ExprRef`] a caller already holds — the caller reuses it and nothing
//! more is needed. What `embed` does own is declaration: a `Shape::Buffer`
//! carries a whole [`BufferDecl`](crate::decl::BufferDecl) (an e-class
//! outlives any one graph), and turning that back into a local slot means
//! declaring one slot per distinct identity. That dedup is what "declare a
//! buffer here" means for an environment, and it lives here rather than in
//! whatever happens to be walking the graph.

use alloc::vec::Vec;

use crate::dag::Node;
use crate::expr::{ExprBuilder, ExprData, ExprRef, Term};
use crate::term::{Children, Ir, Shape};

impl Ir for ExprBuilder {
    type Ref = ExprRef;

    fn project(&self, r: ExprRef) -> Shape<'_, ExprRef> {
        let node = self.node(r);
        match *node {
            ExprData::Var(i) => Shape::Var(i),
            ExprData::Const(bits) => Shape::Const(f32::from_bits(bits)),
            ExprData::Param(i) => Shape::Param(i),
            ExprData::Buffer(b) => Shape::Buffer(self.env().buffer(b)),
            ExprData::Uniform(u) => Shape::Uniform(self.env().uniform(u)),
            // `Many` at every arity, not the inline forms at one to three: the
            // builder holds its children contiguously, which is exactly the
            // case `Children::Many` is for. The inline variants exist for
            // representations that store children *in* the node and so have
            // no slice to lend; this one does.
            ExprData::Op(op) => Shape::Op(op, Children::Many(self.child_refs(r))),
        }
    }

    fn embed(&mut self, shape: Shape<'_, ExprRef>) -> ExprRef {
        match shape {
            Shape::Var(i) => self.push_var(i),
            Shape::Const(v) => self.push_const(v),
            Shape::Param(i) => self.push_param(i),
            Shape::Buffer(decl) => {
                let slot = self.declare_buffer(decl);
                self.push_buffer(slot)
            }
            Shape::Uniform(decl) => {
                let slot = self.declare_uniform(decl);
                self.push_uniform(slot)
            }
            Shape::Op(op, children) => {
                assert!(
                    !children.is_empty(),
                    "ExprBuilder::embed: {op:?} with no children — an operator \
                     node of arity zero is not a term this language has"
                );
                let kids: Vec<ExprRef> = children.iter().collect();
                self.push_nary(op, &kids)
            }
        }
    }
}

/// Rebuild the subgraph reachable from `term`'s root into `out`, returning the
/// reference it maps to there.
///
/// The generic shape of "materialise a term elsewhere": iterative (expression
/// depth is unbounded in principle — `Dwrt` chain-rule expansion, deep
/// composition — so this must not blow the Rust stack) and memoized per node,
/// so DAG sharing survives the copy instead of being expanded into a tree.
pub fn rebuild_into<O: Ir>(term: Term<'_>, out: &mut O) -> O::Ref {
    let root = term.root();
    let mut memo = term.dag().side_table(None);
    let mut work: Vec<(Node<'_, ExprData>, bool)> = alloc::vec![(root, false)];

    while let Some((node, children_done)) = work.pop() {
        if memo[node].is_some() {
            continue;
        }
        if !children_done {
            let pending: Vec<Node<'_, ExprData>> =
                node.children().filter(|c| memo[*c].is_none()).collect();
            if !pending.is_empty() {
                work.push((node, true));
                work.extend(pending.into_iter().map(|k| (k, false)));
                continue;
            }
        }
        let built = match *node {
            ExprData::Var(i) => out.embed(Shape::Var(i)),
            ExprData::Const(bits) => out.embed(Shape::Const(f32::from_bits(bits))),
            ExprData::Param(i) => out.embed(Shape::Param(i)),
            ExprData::Buffer(b) => out.embed(Shape::Buffer(term.env().buffer(b))),
            ExprData::Uniform(u) => out.embed(Shape::Uniform(term.env().uniform(u))),
            ExprData::Op(op) => {
                let mapped: Vec<O::Ref> = node
                    .children()
                    .map(|k| memo[k].expect("rebuild_into: child built before parent"))
                    .collect();
                out.embed(Shape::Op(op, Children::Many(&mapped)))
            }
        };
        memo[node] = Some(built);
    }

    memo[root].expect("rebuild_into: root not built")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OpKind;
    use crate::Uniform;
    use crate::expr::display;
    use alloc::format;

    #[test]
    fn rebuild_into_preserves_sharing_and_redeclares_by_identity() {
        let decl = Uniform::new(2.0).decl();
        let mut b = ExprBuilder::new();
        let slot = b.declare_uniform(decl);
        let u = b.push_uniform(slot);
        let x = b.push_var(0);
        let s = b.push_binary(OpKind::Mul, x, u);
        let root = b.push_binary(OpKind::Add, s, s);
        let (rooted, env) = b.finish(&[root]);

        let mut out = ExprBuilder::new();
        let copied = rebuild_into(Term::new(rooted.entry(), &env), &mut out);
        let (copy, copy_env) = out.finish(&[copied]);

        assert_eq!(copy_env.uniforms, [decl], "one identity, one slot");
        assert_eq!(
            copy.entry().node_count(),
            rooted.entry().node_count(),
            "the shared product is copied once"
        );
        assert_eq!(
            format!("{}", display(copy.entry())),
            format!("{}", display(rooted.entry()))
        );
    }

    #[test]
    fn project_reads_back_every_arity() {
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let c = b.push_const(1.5);
        let neg = b.push_unary(OpKind::Neg, x);
        let add = b.push_binary(OpKind::Add, neg, c);
        let sel = b.push_ternary(OpKind::Select, add, x, c);
        let tup = b.push_nary(OpKind::Tuple, &[x, c, neg, add, sel]);

        assert_eq!(b.project(x), Shape::Var(0));
        assert_eq!(b.project(c), Shape::Const(1.5));
        for (r, op, kids) in [
            (neg, OpKind::Neg, alloc::vec![x]),
            (add, OpKind::Add, alloc::vec![neg, c]),
            (sel, OpKind::Select, alloc::vec![add, x, c]),
            (tup, OpKind::Tuple, alloc::vec![x, c, neg, add, sel]),
        ] {
            assert_eq!(b.project(r), Shape::Op(op, Children::Many(&kids)));
        }
    }

    /// The `Ir` round trip: project a node, embed the projection, and the copy
    /// is the same node — which is what makes an optimizer an endomorphism
    /// rather than a conversion with two failure modes.
    #[test]
    fn embedding_a_projection_reproduces_the_node() {
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let c = b.push_const(2.5);
        let add = b.push_binary(OpKind::Add, x, c);

        let mut out = ExprBuilder::new();
        let x2 = out.embed(b.project(x));
        let c2 = out.embed(b.project(c));
        let add2 = out.embed(Shape::Op(OpKind::Add, Children::Two(x2, c2)));
        let (copy, _) = out.finish(&[add2]);
        let (orig, _) = b.finish(&[add]);
        assert!(copy.entry().subtree_eq(orig.entry()));
    }
}
