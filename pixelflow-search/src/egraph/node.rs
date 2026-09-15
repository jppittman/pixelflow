//! Core e-graph data structures: EClassId and ENode.

use super::ops::Op;
use alloc::vec::Vec;
use pixelflow_ir::arena::{BufferDecl, UniformDecl};
use pixelflow_ir::fold::Fold;

/// Identifier for an equivalence class in the e-graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EClassId(pub(crate) u32);

impl EClassId {
    /// Get the index of this e-class ID.
    ///
    /// This is useful for using EClassId as a key in external data structures.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// An expression node in the e-graph.
///
/// Children point to e-classes (not other nodes), enabling compact
/// representation of equivalent expressions.
#[derive(Clone, Debug)]
pub enum ENode {
    /// Variable with index (0=X, 1=Y, 2=Z, 3=W, etc.)
    Var(u8),
    /// Constant value (stored as f32 bits)
    Const(u32),
    /// Bound-memory leaf: a buffer declaration, read through a `Gather` node.
    ///
    /// Carries the full [`BufferDecl`] rather than an arena-local `BufferId`
    /// because e-classes outlive any one arena: `BufferIdentity` is
    /// process-unique, so two `Buffer` leaves are the same e-class iff they
    /// name the same memory with the same extents — which is exactly the
    /// hash-consing CSE this leaf exists for. Extraction redeclares the decl
    /// into the output arena.
    Buffer(BufferDecl),
    /// Per-call scalar leaf: a uniform declaration, hash-consed by identity
    /// for the same reason as [`ENode::Buffer`]. No rule matches it as a
    /// `Const`, so it is never folded; its gain in the e-graph is CSE of the
    /// arithmetic that depends on it alone.
    Uniform(UniformDecl),
    /// Unbound scalar slot of a `kernel!` builder, hash-consed by index.
    ///
    /// The same opaque leaf as [`ENode::Uniform`], one tier earlier: no rule
    /// matches it, nothing folds it, its derivative is zero, and its gain in
    /// the e-graph is CSE of the arithmetic depending on it alone. A builder
    /// substitutes it away before anything is compiled, so only the macro
    /// tier's [`Vocabulary::Templates`](super::ops::Vocabulary) may hold one.
    ///
    /// Before this leaf existed the e-graph refused `Param` outright, and
    /// each macro-side caller smuggled one past as something else — as
    /// `Var(16 + i)` in the compiler's arena bridge, as an opaque synthetic
    /// identifier in its AST optimizer. Two encodings of one idea, next to
    /// two worked examples of it.
    Param(u8),
    /// Operation with children
    Op {
        op: &'static dyn Op,
        children: Vec<EClassId>,
    },
    /// A bounded fold: `⊕_{k ∈ fold.range()} body[fold.binder() := k]`.
    ///
    /// The one node in the graph that *binds*, and the reason it can be in
    /// the graph at all: its algebra, binder and range live in the node's
    /// identity rather than in three `Const` children. As children they were
    /// e-classes like any other — the `Const(4.0)` naming the `Add` combiner
    /// was the same class as any literal `4.0` in the kernel, and every
    /// arithmetic rule in the set could reach a trip count — so `Reduce` was
    /// simply lowered before insertion and the e-graph never saw a fold.
    ///
    /// Hash-consing therefore does what it should: two folds are one node iff
    /// they fold the same body, under the same algebra, over the same range.
    Reduce { fold: Fold, body: EClassId },
}

impl ENode {
    /// Create a constant node.
    pub fn constant(val: f32) -> Self {
        ENode::Const(val.to_bits())
    }

    /// Get the constant value if this is a Const node.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            ENode::Const(bits) => Some(f32::from_bits(*bits)),
            _ => None,
        }
    }

    /// Check if this node is a specific constant value.
    pub fn is_const(&self, val: f32) -> bool {
        self.as_f32() == Some(val)
    }

    /// Get the operation if this is an Op node.
    pub fn op(&self) -> Option<&'static dyn Op> {
        match self {
            ENode::Op { op, .. } => Some(*op),
            _ => None,
        }
    }

    /// The fold this node performs, if it is one.
    pub fn fold(&self) -> Option<Fold> {
        match self {
            ENode::Reduce { fold, .. } => Some(*fold),
            _ => None,
        }
    }

    /// Get children of this node.
    ///
    /// Allocates and clones — see [`Self::children_slice`] for the
    /// zero-allocation borrow, which every call site on a saturation/
    /// extraction hot path should prefer. This owned form stays for callers
    /// (training/eval tooling outside this crate) that want a `Vec` to hold
    /// past the node's borrow.
    pub fn children(&self) -> Vec<EClassId> {
        self.children_slice().to_vec()
    }

    /// Borrow this node's children with no allocation.
    pub fn children_slice(&self) -> &[EClassId] {
        match self {
            ENode::Var(_)
            | ENode::Const(_)
            | ENode::Buffer(_)
            | ENode::Uniform(_)
            | ENode::Param(_) => &[],
            ENode::Op { children, .. } => children,
            ENode::Reduce { body, .. } => core::slice::from_ref(body),
        }
    }

    /// Borrow this node's children mutably — what canonicalization needs, and
    /// the only reason a caller should want it.
    pub fn children_slice_mut(&mut self) -> &mut [EClassId] {
        match self {
            ENode::Var(_)
            | ENode::Const(_)
            | ENode::Buffer(_)
            | ENode::Uniform(_)
            | ENode::Param(_) => &mut [],
            ENode::Op { children, .. } => children,
            ENode::Reduce { body, .. } => core::slice::from_mut(body),
        }
    }

    /// Get binary operands if this is a binary operation.
    pub fn binary_operands(&self) -> Option<(EClassId, EClassId)> {
        match self {
            ENode::Op { children, .. } if children.len() == 2 => Some((children[0], children[1])),
            _ => None,
        }
    }
}

// Implement PartialEq and Eq manually since we can't derive for dyn Op.
// We use OpKind for comparison since ZST pointer addresses are unreliable.
impl PartialEq for ENode {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ENode::Var(a), ENode::Var(b)) => a == b,
            (ENode::Const(a), ENode::Const(b)) => a == b,
            // Full-decl equality: same identity AND same extents. Identity
            // alone would hash-cons two decls that disagree on extents, and
            // extraction would then redeclare only one of them — a silent
            // extent corruption. Splice already asserts decls agree, so in a
            // well-formed graph this is equivalent to identity equality.
            (ENode::Buffer(a), ENode::Buffer(b)) => a == b,
            // Identity and default, bitwise (`UniformDecl`'s own equality).
            (ENode::Uniform(a), ENode::Uniform(b)) => a == b,
            (ENode::Param(a), ENode::Param(b)) => a == b,
            (
                ENode::Op {
                    op: op1,
                    children: c1,
                },
                ENode::Op {
                    op: op2,
                    children: c2,
                },
            ) => {
                // Compare by OpKind - ZST pointer addresses are unreliable
                op1.kind() == op2.kind() && c1 == c2
            }
            (ENode::Reduce { fold: f1, body: b1 }, ENode::Reduce { fold: f2, body: b2 }) => {
                f1 == f2 && b1 == b2
            }
            _ => false,
        }
    }
}

impl Eq for ENode {}

// Implement Hash manually using OpKind for ops.
impl core::hash::Hash for ENode {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        match self {
            ENode::Var(idx) => {
                0u8.hash(state);
                idx.hash(state);
            }
            ENode::Const(bits) => {
                1u8.hash(state);
                bits.hash(state);
            }
            ENode::Buffer(decl) => {
                3u8.hash(state);
                decl.hash(state);
            }
            ENode::Uniform(decl) => {
                4u8.hash(state);
                decl.hash(state);
            }
            ENode::Param(i) => {
                5u8.hash(state);
                i.hash(state);
            }
            ENode::Op { op, children } => {
                2u8.hash(state);
                // Hash by OpKind - ZST pointer addresses are unreliable
                op.kind().hash(state);
                children.hash(state);
            }
            ENode::Reduce { fold, body } => {
                6u8.hash(state);
                fold.hash(state);
                body.hash(state);
            }
        }
    }
}
