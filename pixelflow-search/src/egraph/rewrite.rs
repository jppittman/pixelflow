//! Rewrite rule infrastructure.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;

use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::ops::Op;
use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId};

/// Structural rewrite template expression.
#[derive(Clone, Debug, PartialEq)]
pub enum Pattern {
    Var(u8),
    Const(f32),
    Param(u8),
    Unary(OpKind, Arc<Pattern>),
    Binary(OpKind, Arc<Pattern>, Arc<Pattern>),
    Ternary(OpKind, Arc<Pattern>, Arc<Pattern>, Arc<Pattern>),
    Nary(OpKind, Vec<Pattern>),
}

impl Pattern {
    #[must_use]
    pub fn kind(&self) -> OpKind {
        match self {
            Self::Var(_) => OpKind::Var,
            Self::Const(_) | Self::Param(_) => OpKind::Const,
            Self::Unary(op, _) => *op,
            Self::Binary(op, _, _) => *op,
            Self::Ternary(op, _, _, _) => *op,
            Self::Nary(op, _) => *op,
        }
    }

    #[must_use]
    pub fn op_type(&self) -> OpKind {
        self.kind()
    }

    #[must_use]
    pub fn node_count(&self) -> usize {
        match self {
            Self::Var(_) | Self::Const(_) | Self::Param(_) => 1,
            Self::Unary(_, a) => 1 + a.node_count(),
            Self::Binary(_, a, b) => 1 + a.node_count() + b.node_count(),
            Self::Ternary(_, a, b, c) => 1 + a.node_count() + b.node_count() + c.node_count(),
            Self::Nary(_, children) => 1 + children.iter().map(Self::node_count).sum::<usize>(),
        }
    }

    #[must_use]
    pub fn depth(&self) -> usize {
        match self {
            Self::Var(_) | Self::Const(_) | Self::Param(_) => 1,
            Self::Unary(_, a) => 1 + a.depth(),
            Self::Binary(_, a, b) => 1 + a.depth().max(b.depth()),
            Self::Ternary(_, a, b, c) => 1 + a.depth().max(b.depth()).max(c.depth()),
            Self::Nary(_, children) => 1 + children.iter().map(Self::depth).max().unwrap_or(0),
        }
    }

    pub fn push_into(&self, arena: &mut ExprArena) -> ExprId {
        match self {
            Self::Var(i) => arena.push_var(*i),
            Self::Const(v) => arena.push_const(*v),
            Self::Param(i) => arena.push_param(*i),
            Self::Unary(op, a) => {
                let a = a.push_into(arena);
                arena.push_unary(*op, a)
            }
            Self::Binary(op, a, b) => {
                let a = a.push_into(arena);
                let b = b.push_into(arena);
                arena.push_binary(*op, a, b)
            }
            Self::Ternary(op, a, b, c) => {
                let a = a.push_into(arena);
                let b = b.push_into(arena);
                let c = c.push_into(arena);
                arena.push_ternary(*op, a, b, c)
            }
            Self::Nary(op, children) => {
                let mut child_ids = Vec::with_capacity(children.len());
                for child in children {
                    child_ids.push(child.push_into(arena));
                }
                arena.push_nary(*op, &child_ids)
            }
        }
    }
}

impl fmt::Display for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Var(0) => write!(f, "X"),
            Self::Var(1) => write!(f, "Y"),
            Self::Var(2) => write!(f, "Z"),
            Self::Var(3) => write!(f, "W"),
            Self::Var(i) => write!(f, "v{i}"),
            Self::Const(v) => write!(f, "{v}"),
            Self::Param(i) => write!(f, "p{i}"),
            Self::Unary(op, a) => write!(f, "({} {})", op.name(), a),
            Self::Binary(op, a, b) => write!(f, "({} {} {})", op.name(), a, b),
            Self::Ternary(op, a, b, c) => write!(f, "({} {} {} {})", op.name(), a, b, c),
            Self::Nary(op, children) => {
                write!(f, "({}", op.name())?;
                for child in children {
                    write!(f, " {}", child)?;
                }
                write!(f, ")")
            }
        }
    }
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
    fn lhs_template(&self) -> Option<Pattern> {
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
    fn rhs_template(&self) -> Option<Pattern> {
        None
    }
}
