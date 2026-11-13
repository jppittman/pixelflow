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

use crate::arena_pat;
use crate::egraph::{EClassId, EGraph, ENode, Rewrite, RewriteAction, RootShapes};
use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId};

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
    fn forward() -> OpKind;
    /// The backward/inverse function (e.g., ln).
    fn backward() -> OpKind;
}

// ============================================================================
// FunctionInverse Declarations
// ============================================================================

/// exp and ln are inverses: exp(ln(x)) = x, ln(exp(x)) = x
pub struct ExpLn;

impl FunctionInverse for ExpLn {
    fn forward() -> OpKind {
        OpKind::Exp
    }
    fn backward() -> OpKind {
        OpKind::Ln
    }
}

/// exp2 and log2 are inverses: 2^(log2(x)) = x, log2(2^x) = x
pub struct Exp2Log2;

impl FunctionInverse for Exp2Log2 {
    fn forward() -> OpKind {
        OpKind::Exp2
    }
    fn backward() -> OpKind {
        OpKind::Log2
    }
}

// ============================================================================
// FunctionInverse Derived Rules
// ============================================================================

/// Rule: f(f⁻¹(x)) → x (forward cancels backward)
///
/// Example: exp(ln(x)) → x
pub struct ForwardBackward<T: FunctionInverse>(PhantomData<T>);

impl<T: FunctionInverse> ForwardBackward<T> {
    #[must_use]
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
        match T::forward() {
            OpKind::Exp => "exp-ln-cancel",
            OpKind::Exp2 => "exp2-log2-cancel",
            _ => "forward-backward-cancel",
        }
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::kind(T::forward())
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: forward(backward(x))
        let &ENode::Op { op, children } = node else {
            return None;
        };
        if op != T::forward() {
            return None;
        }
        if children.len() != 1 {
            return None;
        }

        let arg = children[0];

        // Check if argument is backward(x)
        for arg_node in egraph.nodes(arg) {
            if let &ENode::Op {
                op: arg_op,
                children: arg_children,
            } = arg_node
                && arg_op == T::backward()
                && arg_children.len() == 1
            {
                let x = arg_children[0];
                // forward(backward(x)) → x
                return Some(RewriteAction::Union(x));
            }
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, un T::forward(), (un T::backward(), (var 0))))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, var 0))
    }
}

/// Rule: f⁻¹(f(x)) → x (backward cancels forward)
///
/// Example: ln(exp(x)) → x
pub struct BackwardForward<T: FunctionInverse>(PhantomData<T>);

impl<T: FunctionInverse> BackwardForward<T> {
    #[must_use]
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
        match T::backward() {
            OpKind::Ln => "ln-exp-cancel",
            OpKind::Log2 => "log2-exp2-cancel",
            _ => "backward-forward-cancel",
        }
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::kind(T::backward())
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: backward(forward(x))
        let &ENode::Op { op, children } = node else {
            return None;
        };
        if op != T::backward() {
            return None;
        }
        if children.len() != 1 {
            return None;
        }

        let arg = children[0];

        // Check if argument is forward(x)
        for arg_node in egraph.nodes(arg) {
            if let &ENode::Op {
                op: arg_op,
                children: arg_children,
            } = arg_node
                && arg_op == T::forward()
                && arg_children.len() == 1
            {
                let x = arg_children[0];
                // backward(forward(x)) → x
                return Some(RewriteAction::Union(x));
            }
        }
        None
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, un T::backward(), (un T::forward(), (var 0))))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, var 0))
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
    fn func() -> OpKind;
    /// The source operation (e.g., Add for exp).
    fn source_op() -> OpKind;
    /// The target operation (e.g., Mul for exp).
    fn target_op() -> OpKind;
}

// ============================================================================
// Homomorphism Declarations
// ============================================================================

/// exp is a homomorphism from (R, +) to (R⁺, ×)
/// exp(a + b) = exp(a) * exp(b)
pub struct ExpHomomorphism;

impl Homomorphism for ExpHomomorphism {
    fn func() -> OpKind {
        OpKind::Exp
    }
    fn source_op() -> OpKind {
        OpKind::Add
    }
    fn target_op() -> OpKind {
        OpKind::Mul
    }
}

/// ln is a homomorphism from (R⁺, ×) to (R, +)
/// ln(a * b) = ln(a) + ln(b)
pub struct LnHomomorphism;

impl Homomorphism for LnHomomorphism {
    fn func() -> OpKind {
        OpKind::Ln
    }
    fn source_op() -> OpKind {
        OpKind::Mul
    }
    fn target_op() -> OpKind {
        OpKind::Add
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
    #[must_use]
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
        match T::func() {
            OpKind::Exp => "exp-homomorphism",
            OpKind::Ln => "ln-homomorphism",
            _ => "homomorphism",
        }
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::kind(T::func())
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: func(source_op(a, b))
        let &ENode::Op { op, children } = node else {
            return None;
        };
        if op != T::func() {
            return None;
        }
        if children.len() != 1 {
            return None;
        }

        let arg = children[0];

        // Check if argument is source_op(a, b)
        for arg_node in egraph.nodes(arg) {
            if let &ENode::Op {
                op: arg_op,
                children: arg_children,
            } = arg_node
                && arg_op == T::source_op()
                && arg_children.len() == 2
            {
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
        None
    }

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, un T::func(), (bin T::source_op(), (var 0), (var 1))))
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        // TargetOp(Func(V0), Func(V1))
        let fk = T::func();
        Some(arena_pat!(__a, bin T::target_op(), (un fk, (var 0)), (un fk, (var 1))))
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
    #[must_use]
    pub fn new() -> Box<Self> {
        Box::new(Self)
    }
}

impl Rewrite for PowerCombine {
    fn name(&self) -> &str {
        "power-combine"
    }

    fn root_shapes(&self) -> RootShapes {
        RootShapes::kind(OpKind::Mul)
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: Mul(Pow(x, a), Pow(y, b)) where x == y
        let &ENode::Op { op, children } = node else {
            return None;
        };
        if op != OpKind::Mul {
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

    fn lhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(
            arena_pat!(__a, bin OpKind::Mul, (bin OpKind::Pow, (var 0), (var 1)), (bin OpKind::Pow, (var 0), (var 2))),
        )
    }

    fn rhs_template(&self, __a: &mut ExprArena) -> Option<ExprId> {
        Some(arena_pat!(__a, bin OpKind::Pow, (var 0), (bin OpKind::Add, (var 1), (var 2))))
    }
}

impl PowerCombine {
    fn extract_pow(&self, egraph: &EGraph, class: EClassId) -> Option<(EClassId, EClassId)> {
        for node in egraph.nodes(class) {
            if let &ENode::Op { op, children } = node
                && op == OpKind::Pow
                && children.len() == 2
            {
                return Some((children[0], children[1]));
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
#[must_use]
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
