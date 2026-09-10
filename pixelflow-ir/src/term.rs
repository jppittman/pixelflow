//! The term language the e-graph speaks.
//!
//! An optimizer is an endomorphism on the IR. Writing one requires a type for
//! "the IR" — and until this module there was none, only two hand-rolled
//! arena↔e-graph conversions with different failure modes (see
//! docs/plans/2026-09-04-ir-as-a-trait.md §2). [`Ir`] is that type.
//!
//! Two operations, mutually inverse up to the equalities saturation adds:
//!
//! - [`Ir::project`] destructures one node, children already resolved to
//!   references — the unfold.
//! - [`Ir::embed`] builds one node from already-built children — the fold.
//!
//! [`Shape`] is the signature both are phrased in: the pattern functor of
//! [`ExprNode`](crate::arena::ExprNode), with a reference parameter in place
//! of `ExprId` and a whole [`BufferDecl`] in place of `BufferId`. The latter
//! is not an accident of convenience: an e-class outlives any one arena, so a
//! buffer leaf must carry the identity that answers "the same memory?" across
//! a merge rather than a slot index that only means something to the arena it
//! came from.
//!
//! The trait earns its keep with a single implementor. It names the API the
//! e-graph is entitled to use against a term language, which is the API tests
//! should be calling; the two conversions it replaces disagreed about
//! panicking, about which ops were representable, and about whether
//! unreachable nodes were inserted, precisely because no such boundary was
//! written down.

use crate::arena::{BufferDecl, UniformDecl};
use crate::fold::Fold;
use crate::key::KernelKey;
use crate::kind::OpKind;

/// The children of one node, borrowed where they are contiguous.
///
/// Mirrors [`ExprChildren`](crate::arena::ExprChildren): a node with inline
/// children (unary through ternary) has no slice to lend, so projecting one
/// must not be forced to allocate or to borrow a temporary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Children<'a, R> {
    /// A leaf.
    Zero,
    /// One child.
    One(R),
    /// Two children, in operand order.
    Two(R, R),
    /// Three children, in operand order.
    Three(R, R, R),
    /// Four or more, or any count already held contiguously.
    Many(&'a [R]),
}

impl<R: Copy> Children<'_, R> {
    /// How many children this node has.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Zero => 0,
            Self::One(_) => 1,
            Self::Two(..) => 2,
            Self::Three(..) => 3,
            Self::Many(s) => s.len(),
        }
    }

    /// Whether this node is a leaf.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The `i`th child, or `None` past the end.
    #[must_use]
    pub fn get(&self, i: usize) -> Option<R> {
        match (self, i) {
            (Self::One(a), 0) | (Self::Two(a, _), 0) | (Self::Three(a, _, _), 0) => Some(*a),
            (Self::Two(_, b), 1) | (Self::Three(_, b, _), 1) => Some(*b),
            (Self::Three(_, _, c), 2) => Some(*c),
            (Self::Many(s), i) => s.get(i).copied(),
            _ => None,
        }
    }

    /// The children in operand order.
    pub fn iter(&self) -> impl Iterator<Item = R> + '_ {
        (0..self.len()).filter_map(move |i| self.get(i))
    }
}

/// One node of a term language, with children resolved to `R`.
///
/// The signature functor of [`ExprNode`](crate::arena::ExprNode). Every
/// variant an e-graph can hold appears here and nothing else does: no spans,
/// no source syntax, no cost. A term language carrying more than this is free
/// to keep it — [`Ir::embed`] hands the extra back in whatever the
/// implementor's own representation is — but the e-graph never sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape<'a, R> {
    /// Coordinate or bound variable by index.
    Var(u8),
    /// Literal.
    Const(f32),
    /// Macro-parameter slot. Valid only before kernel compilation; an e-graph
    /// insert declines one rather than inventing a value for it.
    Param(u8),
    /// Bound-memory leaf, carrying the identity that survives a merge.
    Buffer(BufferDecl),
    /// Per-call scalar leaf, carrying its identity and default for the same
    /// reason a buffer carries its declaration.
    Uniform(UniformDecl),
    /// A kernel named by content — a leaf here, because its body is in the
    /// [`KernelStore`](crate::store::KernelStore) and not in this term. An
    /// e-graph insert declines one, for the reason it declines a `Param`:
    /// there is no value to reason about until something resolves it.
    ///
    /// (The `Ref` in [`Ir::Ref`] is unrelated — that is how a representation
    /// *names* one of its own nodes; this is a name for a whole kernel.)
    Ref(KernelKey),
    /// An operation over `children`.
    Op(OpKind, Children<'a, R>),
    /// A bounded fold: `⊕_{k ∈ fold.range()} body[fold.binder() := k]`.
    ///
    /// Deliberately *not* a [`Shape::Op`]. A fold's algebra, binder and range
    /// are metadata, not operands: handing them to a walker as children is
    /// what let the combiner's `Const` share an e-class with any literal of
    /// the same value, and put a trip count within reach of every arithmetic
    /// rule in the set. The one child is the body, and the binder is bound in
    /// it — this is the only shape in the language that binds anything.
    Reduce { fold: Fold, body: R },
}

/// A term language the e-graph can destructure and rebuild.
///
/// Implementors are *term representations*, not optimizers: the trait says how
/// to read one node and how to write one node, and says nothing about which
/// nodes are worth writing. Cost, evaluation and codegen are deliberately
/// absent — they are where tiers genuinely differ, and requiring them here
/// would exclude implementors that have no business answering them.
pub trait Ir {
    /// How this representation names a node.
    ///
    /// `Copy + Ord` because callers memoize on it — a DAG-shared term must be
    /// walked once per node, not once per reference — and the memo has to be
    /// an `alloc`-only map, since the crates that walk terms build without
    /// `std`. A reference is an index or a handle in every representation this
    /// models, so ordering one is free.
    type Ref: Copy + Ord;

    /// Destructure the node `r` names.
    fn project(&self, r: Self::Ref) -> Shape<'_, Self::Ref>;

    /// Build a node from already-built children, returning its reference.
    ///
    /// Reusing a returned `Ref` is how sharing is expressed. An implementor
    /// whose representation is a DAG returns the same reference and is done;
    /// one whose representation is a tree decides here what reuse means for
    /// it (a let-binding, say). That choice belongs to the representation, not
    /// to the caller walking it.
    fn embed(&mut self, shape: Shape<'_, Self::Ref>) -> Self::Ref;
}
