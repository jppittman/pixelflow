//! Rewrite rule infrastructure.

use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::ops::Op;
use pixelflow_ir::arena::{ExprArena, ExprId};

/// Build a rewrite-rule template directly into an [`ExprArena`], returning the
/// root [`ExprId`]. The DSL mirrors the structural pattern: `var N` / `cst V` /
/// `par N` leaves, and `un OP, (..)` / `bin OP, (..), (..)` / `tern OP, (..),
/// (..), (..)` nodes. Metavariables are encoded as `var N` (Var(0) = A, etc.).
#[macro_export]
macro_rules! arena_pat {
    ($a:expr, var $i:expr) => { $a.push_var($i) };
    ($a:expr, cst $v:expr) => { $a.push_const($v) };
    ($a:expr, par $i:expr) => { $a.push_param($i) };
    ($a:expr, un $op:expr, ($($c:tt)+)) => {{
        let __c = $crate::arena_pat!($a, $($c)+);
        $a.push_unary($op, __c)
    }};
    ($a:expr, bin $op:expr, ($($l:tt)+), ($($r:tt)+)) => {{
        let __l = $crate::arena_pat!($a, $($l)+);
        let __r = $crate::arena_pat!($a, $($r)+);
        $a.push_binary($op, __l, __r)
    }};
    ($a:expr, tern $op:expr, ($($x:tt)+), ($($y:tt)+), ($($z:tt)+)) => {{
        let __x = $crate::arena_pat!($a, $($x)+);
        let __y = $crate::arena_pat!($a, $($y)+);
        let __z = $crate::arena_pat!($a, $($z)+);
        $a.push_ternary($op, __x, __y, __z)
    }};
}

/// Actions that a rewrite rule can produce.
#[derive(Debug, Clone)]
pub enum RewriteAction {
    /// Union this e-class with another
    Union(EClassId),
    /// Create a new e-node and union with it
    Create(ENode),
    /// Distribute: A * (B + C) -> A*B + A*C
    Distribute {
        outer: &'static dyn Op,
        inner: &'static dyn Op,
        a: EClassId,
        b: EClassId,
        c: EClassId,
    },
    /// Factor: A*B + A*C -> A * (B + C)
    Factor {
        outer: &'static dyn Op,
        inner: &'static dyn Op,
        common: EClassId,
        unique_l: EClassId,
        unique_r: EClassId,
    },
    /// Canonicalize: Sub(a,b) -> Add(a, Neg(b))
    Canonicalize {
        target: &'static dyn Op,
        inverse: &'static dyn Op,
        a: EClassId,
        b: EClassId,
    },
    /// Associate: (a op b) op c -> a op (b op c)
    Associate {
        op: &'static dyn Op,
        a: EClassId,
        b: EClassId,
        c: EClassId,
    },
    /// ReverseAssociate: a op (b op c) -> (a op b) op c
    ReverseAssociate {
        op: &'static dyn Op,
        a: EClassId,
        b: EClassId,
        c: EClassId,
    },
    /// OddParity: Op(neg(x)) -> neg(Op(x))
    /// Creates Op(inner), then wraps in Neg.
    OddParity {
        func: &'static dyn Op,
        inner: EClassId,
    },
    /// AngleAddition: sin(a+b) -> sin(a)cos(b) + cos(a)sin(b)
    /// or cos(a+b) -> cos(a)cos(b) - sin(a)sin(b)
    AngleAddition {
        term1_op1: &'static dyn Op,
        term1_op2: &'static dyn Op,
        term2_op1: &'static dyn Op,
        term2_op2: &'static dyn Op,
        term2_sign: crate::math::trig::Sign,
        a: EClassId,
        b: EClassId,
    },
    /// Homomorphism: f(a ⊕ b) -> f(a) ⊗ f(b)
    /// e.g., exp(a + b) -> exp(a) * exp(b)
    Homomorphism {
        func: &'static dyn Op,
        target_op: &'static dyn Op,
        a: EClassId,
        b: EClassId,
    },
    /// PowerCombine: x^a * x^b -> x^(a+b)
    PowerCombine {
        base: EClassId,
        exp_a: EClassId,
        exp_b: EClassId,
    },
    /// ReverseAngleAddition: sin(a)cos(b) + cos(a)sin(b) -> sin(a + b)
    /// (The inverse of angle addition, enables double angle discovery)
    ReverseAngleAddition {
        trig_op: &'static dyn Op,
        a: EClassId,
        b: EClassId,
    },
    /// HalfAngleProduct: sin(x) * cos(x) -> sin(x + x) / 2
    /// Derived from sin(2x) = 2*sin(x)*cos(x)
    HalfAngleProduct { x: EClassId },
    /// Doubling: a + a -> 2 * a
    Doubling { a: EClassId },
    /// Halving: 2 * a -> a + a (reverse of doubling)
    Halving { a: EClassId },
    /// PowerRecurrence: pow(x, n) -> x * pow(x, n-1) for integer n >= 3
    PowerRecurrence { base: EClassId, exponent: i32 },
    /// LogPower: log(pow(x, n)) -> n * log(x)
    LogPower {
        log_op: &'static dyn Op,
        base: EClassId,
        exponent: EClassId,
    },
    /// ExpandSquare: (a+b)² -> a² + 2ab + b²
    ExpandSquare { a: EClassId, b: EClassId },
    /// DiffOfSquares: a² - b² -> (a+b)(a-b)
    DiffOfSquares { a: EClassId, b: EClassId },
}

/// A rewrite rule that can be applied to e-graph nodes.
///
/// Requires `Send + Sync` so rules can be shared across worker threads
/// during parallel trajectory generation.
pub trait Rewrite: Send + Sync {
    /// Human-readable name for debugging.
    fn name(&self) -> &str;

    /// Try to apply this rule to a node in an e-class.
    /// Returns `Some(action)` if the rule matches.
    fn apply(&self, egraph: &EGraph, id: EClassId, node: &ENode) -> Option<RewriteAction>;

    /// Whether this rule is destructive: the matched LHS node should be
    /// removed from the e-class after the action is applied.
    ///
    /// Only safe for rules that provably simplify: the RHS is strictly
    /// cheaper than the LHS (involution, identity, annihilator, constant-fold).
    /// Destructive rules reduce e-graph size, preventing node accumulation
    /// that slows future rule matching.
    ///
    /// Default: false (non-destructive, standard equality saturation).
    fn is_destructive(&self) -> bool {
        false
    }

    /// LHS template expression (what this rule matches).
    ///
    /// Uses metavariables: `Expr::Var(0)` = A, `Expr::Var(1)` = B, etc.
    /// These describe the structural pattern that triggers the rule.
    ///
    /// Example: Distribute rule (`A * (B + C) → A*B + A*C`) would return:
    /// ```ignore
    /// Expr::Binary(Mul, Var(0), Expr::Binary(Add, Var(1), Var(2)))
    /// ```
    ///
    /// Returns `None` if the rule doesn't have a defined template.
    /// Rules can opt-in by overriding this method.
    fn lhs_template(&self, _arena: &mut ExprArena) -> Option<ExprId> {
        None
    }

    /// RHS template expression (what this rule produces).
    ///
    /// Uses the same metavariables as `lhs_template()`.
    ///
    /// Example: Distribute rule would return:
    /// ```ignore
    /// Expr::Binary(Add,
    ///     Expr::Binary(Mul, Var(0), Var(1)),
    ///     Expr::Binary(Mul, Var(0), Var(2)))
    /// ```
    ///
    /// Returns `None` if the rule doesn't have a defined template.
    fn rhs_template(&self, _arena: &mut ExprArena) -> Option<ExprId> {
        None
    }
}
