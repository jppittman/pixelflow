//! [`ExprArena`] as an [`Ir`].
//!
//! The arena is a DAG, so `embed` expresses sharing by returning the same
//! [`ExprId`] a caller already holds — the caller reuses it and nothing more
//! is needed. What `embed` does own is buffer declaration: a `Shape::Buffer`
//! carries a whole [`BufferDecl`] (an e-class outlives any one arena), and
//! turning that back into an arena-local slot means declaring one slot per
//! distinct [`BufferIdentity`]. That dedup is what "declare a buffer here"
//! means for an arena, and it lives here rather than in whatever happens to be
//! walking the graph.

use alloc::vec::Vec;

use crate::arena::{BufferDecl, BufferId, ExprArena, ExprId, ExprNode};
use crate::term::{Children, Ir, Shape};

impl Ir for ExprArena {
    type Ref = ExprId;

    fn project(&self, r: ExprId) -> Shape<'_, ExprId> {
        match *self.node(r) {
            ExprNode::Var(i) => Shape::Var(i),
            ExprNode::Const(v) => Shape::Const(v),
            ExprNode::Param(i) => Shape::Param(i),
            ExprNode::Buffer(b) => Shape::Buffer(*self.buffer_decl(b)),
            ExprNode::Uniform(u) => Shape::Uniform(*self.uniform_decl(u)),
            ExprNode::Ref(k) => Shape::Ref(k),
            ExprNode::Unary(op, a) => Shape::Op(op, Children::One(a)),
            ExprNode::Binary(op, a, b) => Shape::Op(op, Children::Two(a, b)),
            ExprNode::Ternary(op, a, b, c) => Shape::Op(op, Children::Three(a, b, c)),
            ExprNode::Nary(op, start, len) => {
                Shape::Op(op, Children::Many(self.nary_children_slice(start, len)))
            }
            ExprNode::Reduce { fold, body } => Shape::Reduce { fold, body },
        }
    }

    fn embed(&mut self, shape: Shape<'_, ExprId>) -> ExprId {
        match shape {
            Shape::Var(i) => self.push_var(i),
            Shape::Const(v) => self.push_const(v),
            Shape::Param(i) => self.push_param(i),
            Shape::Buffer(decl) => {
                let slot = self.buffer_slot_for(decl);
                self.push_buffer(slot)
            }
            Shape::Uniform(decl) => {
                let slot = self.uniform_slot_for(decl);
                self.push_uniform(slot)
            }
            Shape::Ref(key) => self.push_ref(key),
            Shape::Op(op, children) => match children {
                Children::Zero => {
                    panic!("ExprArena::embed: {op:?} with no children — no arena node has arity 0")
                }
                Children::One(a) => self.push_unary(op, a),
                Children::Two(a, b) => self.push_binary(op, a, b),
                Children::Three(a, b, c) => self.push_ternary(op, a, b, c),
                Children::Many(s) => match *s {
                    [a] => self.push_unary(op, a),
                    [a, b] => self.push_binary(op, a, b),
                    [a, b, c] => self.push_ternary(op, a, b, c),
                    _ => self.push_nary(op, s),
                },
            },
            Shape::Reduce { fold, body } => self.push_reduce(fold, body),
        }
    }
}

impl ExprArena {
    /// The slot naming `decl`'s memory in this arena, declaring one if this is
    /// the first time that identity has been seen.
    ///
    /// Two declarations of one identity with different extents is a corrupt
    /// graph — one memory described two ways — so it is an assertion rather
    /// than a silent alias onto whichever arrived first.
    fn buffer_slot_for(&mut self, decl: BufferDecl) -> BufferId {
        let existing = self
            .buffers()
            .iter()
            .position(|d| d.id == decl.id)
            .map(|i| BufferId(i as u16));
        match existing {
            Some(slot) => {
                let held = self.buffer_decl(slot);
                assert_eq!(
                    *held, decl,
                    "ExprArena::embed: buffer identity {:?} declared with two different \
                     extents ({}x{} and {}x{}) — one memory cannot have two shapes",
                    decl.id, held.width, held.height, decl.width, decl.height,
                );
                slot
            }
            None => self.declare_buffer(decl),
        }
    }

    /// Rebuild the subgraph reachable from `root` into `out`, returning the
    /// reference `root` maps to there.
    ///
    /// The generic shape of "materialise a term elsewhere": iterative (arena
    /// depth is unbounded in principle — `Dwrt` chain-rule expansion, deep
    /// composition — so this must not blow the Rust stack) and memoized on
    /// `Self::Ref`, so DAG sharing survives the copy instead of being expanded
    /// into a tree.
    #[must_use]
    pub fn rebuild_into<O: Ir>(&self, root: ExprId, out: &mut O) -> O::Ref {
        let mut memo: Vec<Option<O::Ref>> = alloc::vec![None; self.len()];
        let mut work: Vec<(ExprId, bool)> = alloc::vec![(root, false)];

        while let Some((id, children_done)) = work.pop() {
            if memo[id.0 as usize].is_some() {
                continue;
            }
            let shape = self.project(id);
            if !children_done {
                let pending: Vec<ExprId> = match shape {
                    Shape::Op(_, kids) => kids
                        .iter()
                        .filter(|k| memo[k.0 as usize].is_none())
                        .collect(),
                    Shape::Reduce { body, .. } if memo[body.0 as usize].is_none() => {
                        alloc::vec![body]
                    }
                    _ => Vec::new(),
                };
                if !pending.is_empty() {
                    work.push((id, true));
                    work.extend(pending.into_iter().map(|k| (k, false)));
                    continue;
                }
            }
            let built = match shape {
                Shape::Var(i) => out.embed(Shape::Var(i)),
                Shape::Const(v) => out.embed(Shape::Const(v)),
                Shape::Param(i) => out.embed(Shape::Param(i)),
                Shape::Buffer(d) => out.embed(Shape::Buffer(d)),
                Shape::Uniform(d) => out.embed(Shape::Uniform(d)),
                Shape::Ref(k) => out.embed(Shape::Ref(k)),
                Shape::Op(op, kids) => {
                    let mapped: Vec<O::Ref> = kids
                        .iter()
                        .map(|k| {
                            memo[k.0 as usize].expect("rebuild_into: child built before parent")
                        })
                        .collect();
                    out.embed(Shape::Op(op, Children::Many(&mapped)))
                }
                Shape::Reduce { fold, body } => {
                    let body = memo[body.0 as usize].expect("rebuild_into: body before fold");
                    out.embed(Shape::Reduce { fold, body })
                }
            };
            memo[id.0 as usize] = Some(built);
        }

        memo[root.0 as usize].expect("rebuild_into: root not built")
    }
}
