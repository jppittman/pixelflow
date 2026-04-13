//! Exponential and logarithmic identities via categorical traits.
//!
//! Two key algebraic structures:
//!
//! ## FunctionInverse - Inverse function pairs
//! f(f⁻¹(x)) = x and f⁻¹(f(x)) = x
//!
//! Examples:
//! - exp/ln: exp(ln(x)) = x, ln(exp(x)) = x
//! - exp2/log2: 2^(log2(x)) = x, log2(2^x) = x
//! - sqrt/square (partial): sqrt(x²) = |x|, (√x)² = x (for x ≥ 0)
//!
//! ## Homomorphism - Structure-preserving maps
//! f(a ⊕ b) = f(a) ⊗ f(b)
//!
//! Examples:
//! - exp: exp(a + b) = exp(a) * exp(b) (additive → multiplicative)
//! - ln: ln(a * b) = ln(a) + ln(b) (multiplicative → additive)
//!
//! Deep connection: Via Euler's identity e^(ix) = cos(x) + i·sin(x),
//! the exp Homomorphism IS the trig angle addition rules in the complex plane.

use std::marker::PhantomData;
use std::sync::Arc;

use crate::egraph::{EClassId, EGraph, ENode, Op, Rewrite, RewriteAction, ops};
use crate::egraph::Pattern as Expr;
use pixelflow_ir::OpKind;

fn b(e: Expr) -> Arc<Expr> {
    Arc::new(e)
}

// ============================================================================
// FunctionInverse Trait
// ============================================================================

/// A pair of mutually inverse functions.
///
/// Implementing this trait derives two rules:
/// - f(f⁻¹(x)) → x
/// - f⁻¹(f(x)) → x
pub trait FunctionInverse: Send + Sync {
    /// The forward function (e.g., exp).
    fn forward() -> &'static dyn Op;
    /// The backward/inverse function (e.g., ln).
    fn backward() -> &'static dyn Op;
}

// ============================================================================
// FunctionInverse Declarations
// ============================================================================

/// exp and ln are inverses: exp(ln(x)) = x, ln(exp(x)) = x
pub struct ExpLn;

impl FunctionInverse for ExpLn {
    fn forward() -> &'static dyn Op {
        &ops::Exp
    }
    fn backward() -> &'static dyn Op {
        &ops::Ln
    }
}

/// exp2 and log2 are inverses: 2^(log2(x)) = x, log2(2^x) = x
pub struct Exp2Log2;

impl FunctionInverse for Exp2Log2 {
    fn forward() -> &'static dyn Op {
        &ops::Exp2
    }
    fn backward() -> &'static dyn Op {
        &ops::Log2
    }
}

/// sqrt and square (mul self) are partial inverses.
/// Note: sqrt(x²) = |x|, not x (domain restriction).
/// We only implement (√x)² = x for now.
pub struct SqrtSquare;

impl FunctionInverse for SqrtSquare {
    fn forward() -> &'static dyn Op {
        &ops::Sqrt
    }
    // Square isn't a unary op, so we handle this specially
    fn backward() -> &'static dyn Op {
        &ops::Mul
    } // Placeholder
}

// ============================================================================
// FunctionInverse Derived Rules
// ============================================================================

/// Rule: f(f⁻¹(x)) → x (forward cancels backward)
///
/// Example: exp(ln(x)) → x
pub struct ForwardBackward<T: FunctionInverse>(PhantomData<T>);

impl<T: FunctionInverse> ForwardBackward<T> {
    pub fn new() -> Box<Self> {
        Box::new(Self(PhantomData))
    }
}

impl<T: FunctionInverse> Default for ForwardBackward<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T: FunctionInverse> Rewrite for ForwardBackward<T> {
    fn name(&self) -> &str {
        match T::forward().kind() {
            OpKind::Exp => "exp-ln-cancel",
            OpKind::Exp2 => "exp2-log2-cancel",
            _ => "forward-backward-cancel",
        }
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: forward(backward(x))
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != T::forward().kind() {
            return None;
        }
        if children.len() != 1 {
            return None;
        }

        let arg = children[0];

        // Check if argument is backward(x)
        for arg_node in egraph.nodes(arg) {
            if let ENode::Op {
                op: arg_op,
                children: arg_children,
            } = arg_node
            {
                if arg_op.kind() == T::backward().kind() && arg_children.len() == 1 {
                    let x = arg_children[0];
                    // forward(backward(x)) → x
                    return Some(RewriteAction::Union(x));
                }
            }
        }
        None
    }

    fn lhs_template(&self) -> Option<Expr> {
        // Forward(Backward(V0))
        Some(Expr::Unary(
            T::forward().kind(),
            b(Expr::Unary(T::backward().kind(), b(Expr::Var(0)))),
        ))
    }

    fn rhs_template(&self) -> Option<Expr> {
        // V0
        Some(Expr::Var(0))
    }
}

/// Rule: f⁻¹(f(x)) → x (backward cancels forward)
///
/// Example: ln(exp(x)) → x
pub struct BackwardForward<T: FunctionInverse>(PhantomData<T>);

impl<T: FunctionInverse> BackwardForward<T> {
    pub fn new() -> Box<Self> {
        Box::new(Self(PhantomData))
    }
}

impl<T: FunctionInverse> Default for BackwardForward<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T: FunctionInverse> Rewrite for BackwardForward<T> {
    fn name(&self) -> &str {
        match T::backward().kind() {
            OpKind::Ln => "ln-exp-cancel",
            OpKind::Log2 => "log2-exp2-cancel",
            _ => "backward-forward-cancel",
        }
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: backward(forward(x))
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != T::backward().kind() {
            return None;
        }
        if children.len() != 1 {
            return None;
        }

        let arg = children[0];

        // Check if argument is forward(x)
        for arg_node in egraph.nodes(arg) {
            if let ENode::Op {
                op: arg_op,
                children: arg_children,
            } = arg_node
            {
                if arg_op.kind() == T::forward().kind() && arg_children.len() == 1 {
                    let x = arg_children[0];
                    // backward(forward(x)) → x
                    return Some(RewriteAction::Union(x));
                }
            }
        }
        None
    }

    fn lhs_template(&self) -> Option<Expr> {
        // Backward(Forward(V0))
        Some(Expr::Unary(
            T::backward().kind(),
            b(Expr::Unary(T::forward().kind(), b(Expr::Var(0)))),
        ))
    }

    fn rhs_template(&self) -> Option<Expr> {
        // V0
        Some(Expr::Var(0))
    }
}

// ============================================================================
// Homomorphism Trait
// ============================================================================

/// A structure-preserving map: f(a ⊕ b) = f(a) ⊗ f(b)
///
/// Implementing this trait derives the rule:
/// f(source_op(a, b)) → target_op(f(a), f(b))
pub trait Homomorphism: Send + Sync {
    /// The function (e.g., exp, ln).
    fn func() -> &'static dyn Op;
    /// The source operation (e.g., Add for exp).
    fn source_op() -> &'static dyn Op;
    /// The target operation (e.g., Mul for exp).
    fn target_op() -> &'static dyn Op;
}

// ============================================================================
// Homomorphism Declarations
// ============================================================================

/// exp is a homomorphism from (R, +) to (R⁺, ×)
/// exp(a + b) = exp(a) * exp(b)
pub struct ExpHomomorphism;

impl Homomorphism for ExpHomomorphism {
    fn func() -> &'static dyn Op {
        &ops::Exp
    }
    fn source_op() -> &'static dyn Op {
        &ops::Add
    }
    fn target_op() -> &'static dyn Op {
        &ops::Mul
    }
}

/// ln is a homomorphism from (R⁺, ×) to (R, +)
/// ln(a * b) = ln(a) + ln(b)
pub struct LnHomomorphism;

impl Homomorphism for LnHomomorphism {
    fn func() -> &'static dyn Op {
        &ops::Ln
    }
    fn source_op() -> &'static dyn Op {
        &ops::Mul
    }
    fn target_op() -> &'static dyn Op {
        &ops::Add
    }
}

// ============================================================================
// Homomorphism Derived Rules
// ============================================================================

/// Rule: f(a ⊕ b) → f(a) ⊗ f(b)
///
/// Example: exp(a + b) → exp(a) * exp(b)
pub struct HomomorphismRule<T: Homomorphism>(PhantomData<T>);

impl<T: Homomorphism> HomomorphismRule<T> {
    pub fn new() -> Box<Self> {
        Box::new(Self(PhantomData))
    }
}

impl<T: Homomorphism> Default for HomomorphismRule<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T: Homomorphism> Rewrite for HomomorphismRule<T> {
    fn name(&self) -> &str {
        match T::func().kind() {
            OpKind::Exp => "exp-homomorphism",
            OpKind::Ln => "ln-homomorphism",
            _ => "homomorphism",
        }
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: func(source_op(a, b))
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != T::func().kind() {
            return None;
        }
        if children.len() != 1 {
            return None;
        }

        let arg = children[0];

        // Check if argument is source_op(a, b)
        for arg_node in egraph.nodes(arg) {
            if let ENode::Op {
                op: arg_op,
                children: arg_children,
            } = arg_node
            {
                if arg_op.kind() == T::source_op().kind() && arg_children.len() == 2 {
                    let a = arg_children[0];
                    let b = arg_children[1];
                    // func(source_op(a, b)) → target_op(func(a), func(b))
                    return Some(RewriteAction::Homomorphism {
                        func: T::func(),
                        target_op: T::target_op(),
                        a,
                        b,
                    });
                }
            }
        }
        None
    }

    fn lhs_template(&self) -> Option<Expr> {
        // Func(SourceOp(V0, V1))
        Some(Expr::Unary(
            T::func().kind(),
            b(Expr::Binary(
                T::source_op().kind(),
                b(Expr::Var(0)),
                b(Expr::Var(1)),
            )),
        ))
    }

    fn rhs_template(&self) -> Option<Expr> {
        // TargetOp(Func(V0), Func(V1))
        let fk = T::func().kind();
        Some(Expr::Binary(
            T::target_op().kind(),
            b(Expr::Unary(fk, b(Expr::Var(0)))),
            b(Expr::Unary(fk, b(Expr::Var(1)))),
        ))
    }
}

// ============================================================================
// Power Rules (Special Cases)
// ============================================================================

/// Rule: x^a * x^b → x^(a+b)
///
/// This is the inverse of the exponential homomorphism rule.
/// Combines powers with the same base.
pub struct PowerCombine;

impl PowerCombine {
    pub fn new() -> Box<Self> {
        Box::new(Self)
    }
}

impl Rewrite for PowerCombine {
    fn name(&self) -> &str {
        "power-combine"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: Mul(Pow(x, a), Pow(y, b)) where x == y
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Mul {
            return None;
        }
        if children.len() != 2 {
            return None;
        }

        let left = children[0];
        let right = children[1];

        // Extract Pow(base, exp) from both sides
        let (base_l, exp_l) = self.extract_pow(egraph, left)?;
        let (base_r, exp_r) = self.extract_pow(egraph, right)?;

        // Check if bases are the same
        if egraph.find(base_l) == egraph.find(base_r) {
            // x^a * x^b → x^(a+b)
            return Some(RewriteAction::PowerCombine {
                base: base_l,
                exp_a: exp_l,
                exp_b: exp_r,
            });
        }

        None
    }

    fn lhs_template(&self) -> Option<Expr> {
        // Mul(Pow(V0, V1), Pow(V0, V2))
        Some(Expr::Binary(
            OpKind::Mul,
            b(Expr::Binary(OpKind::Pow, b(Expr::Var(0)), b(Expr::Var(1)))),
            b(Expr::Binary(OpKind::Pow, b(Expr::Var(0)), b(Expr::Var(2)))),
        ))
    }

    fn rhs_template(&self) -> Option<Expr> {
        // Pow(V0, Add(V1, V2))
        Some(Expr::Binary(
            OpKind::Pow,
            b(Expr::Var(0)),
            b(Expr::Binary(OpKind::Add, b(Expr::Var(1)), b(Expr::Var(2)))),
        ))
    }
}

impl PowerCombine {
    fn extract_pow(&self, egraph: &EGraph, class: EClassId) -> Option<(EClassId, EClassId)> {
        for node in egraph.nodes(class) {
            if let ENode::Op { op, children } = node {
                if op.kind() == OpKind::Pow && children.len() == 2 {
                    return Some((children[0], children[1]));
                }
            }
        }
        None
    }
}

// ============================================================================
// Rule Collection
// ============================================================================

/// All exponential/logarithmic rules.
///
/// Inf/NaN inputs are UB in this language — these rules are sound
/// over the well-defined domain.
pub fn exp_rules() -> Vec<Box<dyn Rewrite>> {
    vec![
        // Inverse cancellation
        ForwardBackward::<ExpLn>::new(),
        BackwardForward::<ExpLn>::new(),
        ForwardBackward::<Exp2Log2>::new(),
        BackwardForward::<Exp2Log2>::new(),
        // Homomorphisms
        HomomorphismRule::<ExpHomomorphism>::new(),
        HomomorphismRule::<LnHomomorphism>::new(),
        // Power rules
        PowerCombine::new(),
    ]
}
