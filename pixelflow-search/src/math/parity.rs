//! Parity trait for even/odd function behavior under negation.
//!
//! Functions can be classified by their parity:
//! - **Odd**: f(-x) = -f(x) (sin, tan, sinh, atan)
//! - **Even**: f(-x) = f(x) (cos, abs, cosh)
//!
//! From one Parity trait impl, `ParityNegation` derives one rule:
//! odd functions rewrite Op(neg(x)) → neg(Op(x)), even ones Op(neg(x)) → Op(x).

use std::marker::PhantomData;

use crate::arena_pat;
use crate::egraph::{EClassId, EGraph, ENode, Rewrite, RewriteAction, RootShapes};
use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId};

// ============================================================================
// Parity Trait
// ============================================================================

/// The parity of a function under negation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParityKind {
    /// f(-x) = -f(x)
    Odd,
    /// f(-x) = f(x)
    Even,
}

/// A function with definite parity under negation.
///
/// Implementing this trait for a type automatically derives:
/// - For Odd: `Op(neg(x)) → neg(Op(x))`
/// - For Even: `Op(neg(x)) → Op(x)`
pub trait Parity: Send + Sync {
    /// The operation this parity applies to.
    fn op() -> OpKind;
    /// Whether the function is odd or even.
    fn parity() -> ParityKind;
}

// ============================================================================
// Parity Declarations
// ============================================================================

/// Sin is odd: sin(-x) = -sin(x)
pub struct SinParity;
impl Parity for SinParity {
    fn op() -> OpKind {
        OpKind::Sin
    }
    fn parity() -> ParityKind {
        ParityKind::Odd
    }
}

/// Cos is even: cos(-x) = cos(x)
pub struct CosParity;
impl Parity for CosParity {
    fn op() -> OpKind {
        OpKind::Cos
    }
    fn parity() -> ParityKind {
        ParityKind::Even
    }
}

/// Tan is odd: tan(-x) = -tan(x)
pub struct TanParity;
impl Parity for TanParity {
    fn op() -> OpKind {
        OpKind::Tan
    }
    fn parity() -> ParityKind {
        ParityKind::Odd
    }
}

/// Asin is odd: asin(-x) = -asin(x)
pub struct AsinParity;
impl Parity for AsinParity {
    fn op() -> OpKind {
        OpKind::Asin
    }
    fn parity() -> ParityKind {
        ParityKind::Odd
    }
}

/// Atan is odd: atan(-x) = -atan(x)
pub struct AtanParity;
impl Parity for AtanParity {
    fn op() -> OpKind {
        OpKind::Atan
    }
    fn parity() -> ParityKind {
        ParityKind::Odd
    }
}

/// Abs is even: abs(-x) = abs(x)
pub struct AbsParity;
impl Parity for AbsParity {
    fn op() -> OpKind {
        OpKind::Abs
    }
    fn parity() -> ParityKind {
        ParityKind::Even
    }
}

// ============================================================================
// Derived Rules
// ============================================================================

/// Unified parity rule that handles both odd and even functions.
///
/// For odd functions: Op(neg(x)) creates neg(Op(x))
/// For even functions: Op(neg(x)) creates Op(x)
pub struct ParityNegation<T: Parity>(PhantomData<T>);

impl<T: Parity> ParityNegation<T> {
    #[must_use]
    pub fn new() -> Box<Self> {
        Box::new(Self(PhantomData))
    }
}

impl<T: Parity> Default for ParityNegation<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T: Parity> Rewrite for ParityNegation<T> {
    fn name(&self) -> &str {
        match T::parity() {
            ParityKind::Odd => "odd-negation",
            ParityKind::Even => "even-negation",
        }
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::kind(T::op())
    }

    fn specialization(&self) -> Option<pixelflow_ir::OpKind> {
        Some(T::op())
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: Op(neg(x))
        let &ENode::Op { op, children } = node else {
            return None;
        };
        if op != T::op() {
            return None;
        }
        if children.len() != 1 {
            return None;
        }

        let arg = children[0];

        // Check if argument is neg(x)
        for arg_node in egraph.nodes(arg) {
            if let &ENode::Op {
                op: arg_op,
                children: arg_children,
            } = arg_node
                && arg_op == OpKind::Neg
                && arg_children.len() == 1
            {
                let x = arg_children[0];

                match T::parity() {
                    ParityKind::Even => {
                        // Op(neg(x)) → Op(x)
                        return Some(RewriteAction::Create(ENode::apply(T::op(), &[x])));
                    }
                    ParityKind::Odd => {
                        // Op(neg(x)) → neg(Op(x))
                        // `add` finds or creates Op(x) when it builds neg(Op(x)).
                        return Some(RewriteAction::OddParity {
                            func: T::op(),
                            inner: x,
                        });
                    }
                }
            }
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, un T::op(), (un OpKind::Neg, (var 0))))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        match T::parity() {
            // Neg(Op(V0))
            ParityKind::Odd => Some(arena_pat!(__a, un OpKind::Neg, (un T::op(), (var 0)))),
            // Op(V0)
            ParityKind::Even => Some(arena_pat!(__a, un T::op(), (var 0))),
        }
    }
}

// ============================================================================
// Rule Collection
// ============================================================================

/// All parity-based rules for trig and other functions.
#[must_use]
pub fn parity_rules() -> Vec<Box<dyn Rewrite>> {
    vec![
        // Odd functions: Op(neg(x)) → neg(Op(x))
        ParityNegation::<SinParity>::new(),
        ParityNegation::<TanParity>::new(),
        ParityNegation::<AsinParity>::new(),
        ParityNegation::<AtanParity>::new(),
        // Even functions: Op(neg(x)) → Op(x)
        ParityNegation::<CosParity>::new(),
        ParityNegation::<AbsParity>::new(),
    ]
}
