//! Core e-graph data structures: EClassId and ENode.

use super::ops::Op;
use alloc::vec::Vec;

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
    /// Operation with children
    Op {
        op: &'static dyn Op,
        children: Vec<EClassId>,
    },
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

    /// Get children of this node.
    pub fn children(&self) -> Vec<EClassId> {
        match self {
            ENode::Var(_) | ENode::Const(_) => vec![],
            ENode::Op { children, .. } => children.clone(),
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
            ENode::Op { op, children } => {
                2u8.hash(state);
                // Hash by OpKind - ZST pointer addresses are unreliable
                op.kind().hash(state);
                children.hash(state);
            }
        }
    }
}
