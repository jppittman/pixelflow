//! Power strength reduction and tower rules.
//!
//! The add/mul/pow tower: each level is repeated application of the level below.
//!
//! ```text
//! pow(x, n) = x * pow(x, n-1)    [pow = repeated mul]
//! n * x     = x + (n-1)*x        [mul = repeated add, via doubling/halving]
//! ```
//!
//! Log rules fall out as inverses of the tower:
//!
//! ```text
//! log(a * b) = log(a) + log(b)   [log distributes mul → add]
//! log(a^n)   = n * log(a)         [log of repeated = repeated of log]
//! ```
//!
//! This module provides:
//! - Special-value strength reduction via [`PowSpecialValue`]: pow(x,0)→1,
//!   pow(x,1)→x, pow(x,2)→mul(x,x), pow(x,0.5)→sqrt(x), pow(x,-1)→recip(x),
//!   pow(x,-0.5)→rsqrt(x)
//! - Power recurrence: pow(x,n) ↔ x * pow(x,n-1) for integer n≥2
//! - Log-power: ln(pow(x,n)) → n * ln(x), log2(pow(x,n)) → n * log2(x)
//! - Square expansion: pow(a+b, 2) → a²+2ab+b²
//! - Difference of squares: a²-b² → (a+b)(a-b)
//!
//! Self-division (x/x→1) and self-subtraction (x-x→0) were removed because
//! they are derivable via existing InversePair + InverseAnnihilation chains:
//! - x/x → Canonicalize(MulRecip) → x * recip(x) → InverseAnnihilation → 1
//! - x-x → Canonicalize(AddNeg) → x + neg(x) → InverseAnnihilation → 0

use std::sync::Arc;

use crate::egraph::Pattern as Expr;
use crate::egraph::{EClassId, EGraph, ENode, Op, Rewrite, RewriteAction, ops};
use pixelflow_ir::OpKind;

fn b(e: Expr) -> Arc<Expr> {
    Arc::new(e)
}

const EPSILON: f32 = 1e-6;

fn const_eq(val: f32, target: f32) -> bool {
    (val - target).abs() < EPSILON
}

/// Extract the constant value from an e-class, if any node is a constant.
fn eclass_const(egraph: &EGraph, id: EClassId) -> Option<f32> {
    for node in egraph.nodes(id) {
        if let Some(val) = node.as_f32() {
            return Some(val);
        }
    }
    None
}

// ============================================================================
// Power special values — parameterized
// ============================================================================

/// What a pow(x, special_value) rewrites to.
enum PowResult {
    /// pow(x, 0) → constant (e.g. 1.0)
    Constant(f32),
    /// pow(x, 1) → x (union with base)
    Identity,
    /// pow(x, exp) → unary_op(x) (e.g. sqrt, recip, rsqrt)
    UnaryOp(&'static dyn Op),
    /// pow(x, 2) → mul(x, x)
    SelfMul,
}

/// Parameterized rule for pow(x, special_value) → result.
///
/// Consolidates PowerZero, PowerIdentity, PowerExpandSquare, PowerSqrt,
/// PowerRecip, and PowerRsqrt into one struct with six instances.
struct PowSpecialValue {
    target_exponent: f32,
    name: &'static str,
    result: PowResult,
}

impl Rewrite for PowSpecialValue {
    fn name(&self) -> &str {
        self.name
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Pow {
            return None;
        }
        if children.len() != 2 {
            return None;
        }

        let exp_val = eclass_const(egraph, children[1])?;
        if !const_eq(exp_val, self.target_exponent) {
            return None;
        }

        let base = children[0];
        match &self.result {
            PowResult::Constant(c) => Some(RewriteAction::Create(ENode::constant(*c))),
            PowResult::Identity => Some(RewriteAction::Union(base)),
            PowResult::UnaryOp(op) => Some(RewriteAction::Create(ENode::Op {
                op: *op,
                children: vec![base],
            })),
            PowResult::SelfMul => Some(RewriteAction::Create(ENode::Op {
                op: &ops::Mul,
                children: vec![base, base],
            })),
        }
    }

    fn lhs_template(&self) -> Option<Expr> {
        Some(Expr::Binary(
            OpKind::Pow,
            b(Expr::Var(0)),
            b(Expr::Const(self.target_exponent)),
        ))
    }

    fn rhs_template(&self) -> Option<Expr> {
        match &self.result {
            PowResult::Constant(c) => Some(Expr::Const(*c)),
            PowResult::Identity => Some(Expr::Var(0)),
            PowResult::UnaryOp(op) => Some(Expr::Unary(op.kind(), b(Expr::Var(0)))),
            PowResult::SelfMul => Some(Expr::Binary(OpKind::Mul, b(Expr::Var(0)), b(Expr::Var(0)))),
        }
    }
}

/// Create the 6 parameterized pow-special-value rule instances.
fn pow_special_value_rules() -> Vec<Box<dyn Rewrite>> {
    vec![
        // pow(x, 0) → 1
        Box::new(PowSpecialValue {
            target_exponent: 0.0,
            name: "power-zero",
            result: PowResult::Constant(1.0),
        }),
        // pow(x, 1) → x
        Box::new(PowSpecialValue {
            target_exponent: 1.0,
            name: "power-identity",
            result: PowResult::Identity,
        }),
        // pow(x, 2) → mul(x, x)
        Box::new(PowSpecialValue {
            target_exponent: 2.0,
            name: "power-expand-2",
            result: PowResult::SelfMul,
        }),
        // pow(x, 0.5) → sqrt(x)
        Box::new(PowSpecialValue {
            target_exponent: 0.5,
            name: "power-sqrt",
            result: PowResult::UnaryOp(&ops::Sqrt),
        }),
        // pow(x, -1) → recip(x)
        Box::new(PowSpecialValue {
            target_exponent: -1.0,
            name: "power-recip",
            result: PowResult::UnaryOp(&ops::Recip),
        }),
        // pow(x, -0.5) → rsqrt(x)
        Box::new(PowSpecialValue {
            target_exponent: -0.5,
            name: "power-rsqrt",
            result: PowResult::UnaryOp(&ops::Rsqrt),
        }),
    ]
}

// ============================================================================
// Power recurrence (the tower rule)
// ============================================================================

/// pow(x, n) → x * pow(x, n-1) for integer n ≥ 2
///
/// This is the core tower rule: exponentiation is repeated multiplication.
/// Combined with PowSpecialValue (base case pow(x,2) → x*x and pow(x,1) → x),
/// this enables full strength reduction of any integer power into a chain of
/// multiplies.
///
/// The e-graph explores both representations (compact pow vs expanded mul chain)
/// and the cost extractor picks the cheaper one.
pub struct PowerRecurrence;

impl PowerRecurrence {
    pub fn new() -> Box<Self> {
        Box::new(Self)
    }
}

impl Rewrite for PowerRecurrence {
    fn name(&self) -> &str {
        "power-recurrence"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Pow {
            return None;
        }
        if children.len() != 2 {
            return None;
        }

        let n = eclass_const(egraph, children[1])?;
        // Only for integer n ≥ 3 (n=2 handled by PowSpecialValue, n=1 likewise)
        if n < 2.5 || (n - n.round()).abs() > EPSILON {
            return None;
        }
        let n_int = n.round() as i32;
        if n_int > 8 {
            return None;
        } // Don't explode large powers

        let x = children[0];
        Some(RewriteAction::PowerRecurrence {
            base: x,
            exponent: n_int,
        })
    }

    fn lhs_template(&self) -> Option<Expr> {
        // pow(V0, N) — generic pattern, N is any integer
        Some(Expr::Binary(OpKind::Pow, b(Expr::Var(0)), b(Expr::Var(1))))
    }

    fn rhs_template(&self) -> Option<Expr> {
        // V0 * pow(V0, N-1) — can't express N-1 in template, so None
        None
    }
}

// ============================================================================
// Log-power rules
// ============================================================================

/// ln(pow(x, n)) → n * ln(x)
///
/// Log of a power becomes multiplication: the log distributes exponentiation
/// into multiplication, completing the tower.
pub struct LogPower {
    log_op: &'static dyn Op,
}

impl LogPower {
    pub fn ln() -> Box<Self> {
        Box::new(Self { log_op: &ops::Ln })
    }
    pub fn log2() -> Box<Self> {
        Box::new(Self { log_op: &ops::Log2 })
    }
}

impl Rewrite for LogPower {
    fn name(&self) -> &str {
        match self.log_op.kind() {
            OpKind::Ln => "log-power",
            OpKind::Log2 => "log2-power",
            _ => "logN-power",
        }
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: log(pow(x, n))
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != self.log_op.kind() {
            return None;
        }
        if children.len() != 1 {
            return None;
        }

        let arg = children[0];

        for arg_node in egraph.nodes(arg) {
            if let ENode::Op {
                op: arg_op,
                children: arg_children,
            } = arg_node
            {
                if arg_op.kind() == OpKind::Pow && arg_children.len() == 2 {
                    let x = arg_children[0];
                    let n = arg_children[1];
                    // log(pow(x, n)) → n * log(x)
                    return Some(RewriteAction::LogPower {
                        log_op: self.log_op,
                        base: x,
                        exponent: n,
                    });
                }
            }
        }
        None
    }

    fn lhs_template(&self) -> Option<Expr> {
        let lk = self.log_op.kind();
        Some(Expr::Unary(
            lk,
            b(Expr::Binary(OpKind::Pow, b(Expr::Var(0)), b(Expr::Var(1)))),
        ))
    }

    fn rhs_template(&self) -> Option<Expr> {
        let lk = self.log_op.kind();
        Some(Expr::Binary(
            OpKind::Mul,
            b(Expr::Var(1)),
            b(Expr::Unary(lk, b(Expr::Var(0)))),
        ))
    }
}

// ============================================================================
// Expand square: (a+b)² → a² + 2ab + b²
// ============================================================================

/// pow(a+b, 2) → a² + 2ab + b²
///
/// Combined with Factor, this enables the e-graph to discover
/// a²+2ab+b² = (a+b)² — because both forms end up in the same e-class,
/// and the cost extractor picks the cheaper representation.
pub struct ExpandSquare;

impl ExpandSquare {
    pub fn new() -> Box<Self> {
        Box::new(Self)
    }
}

impl Rewrite for ExpandSquare {
    fn name(&self) -> &str {
        "expand-square"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: Pow(sum, 2) where sum = Add(a, b)
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Pow {
            return None;
        }
        if children.len() != 2 {
            return None;
        }

        let exp_val = eclass_const(egraph, children[1])?;
        if !const_eq(exp_val, 2.0) {
            return None;
        }

        let sum_id = children[0];
        for sum_node in egraph.nodes(sum_id) {
            if let ENode::Op {
                op: sum_op,
                children: sum_children,
            } = sum_node
            {
                if sum_op.kind() == OpKind::Add && sum_children.len() == 2 {
                    let a = sum_children[0];
                    let b = sum_children[1];
                    return Some(RewriteAction::ExpandSquare { a, b });
                }
            }
        }
        None
    }

    fn lhs_template(&self) -> Option<Expr> {
        Some(Expr::Binary(
            OpKind::Pow,
            b(Expr::Binary(OpKind::Add, b(Expr::Var(0)), b(Expr::Var(1)))),
            b(Expr::Const(2.0)),
        ))
    }

    fn rhs_template(&self) -> Option<Expr> {
        // a² + 2ab + b²
        let a2 = Expr::Binary(OpKind::Mul, b(Expr::Var(0)), b(Expr::Var(0)));
        let b2 = Expr::Binary(OpKind::Mul, b(Expr::Var(1)), b(Expr::Var(1)));
        let ab = Expr::Binary(OpKind::Mul, b(Expr::Var(0)), b(Expr::Var(1)));
        let two_ab = Expr::Binary(OpKind::Mul, b(Expr::Const(2.0)), b(ab));
        let sum = Expr::Binary(
            OpKind::Add,
            b(a2),
            b(Expr::Binary(OpKind::Add, b(two_ab), b(b2))),
        );
        Some(sum)
    }
}

// ============================================================================
// Difference of squares: a² - b² → (a+b)(a-b)
// ============================================================================

/// a² - b² → (a+b)(a-b)
///
/// Matches Sub(Mul(a,a), Mul(b,b)) or after canonicalization
/// Add(Mul(a,a), Neg(Mul(b,b))).
pub struct DiffOfSquares;

impl DiffOfSquares {
    pub fn new() -> Box<Self> {
        Box::new(Self)
    }
}

impl Rewrite for DiffOfSquares {
    fn name(&self) -> &str {
        "diff-of-squares"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: Sub(X, Y) where X contains Mul(a,a) and Y contains Mul(b,b)
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Sub {
            return None;
        }
        if children.len() != 2 {
            return None;
        }

        let a = self.extract_self_mul(egraph, children[0])?;
        let b = self.extract_self_mul(egraph, children[1])?;

        Some(RewriteAction::DiffOfSquares { a, b })
    }

    fn lhs_template(&self) -> Option<Expr> {
        // Sub(Mul(V0, V0), Mul(V1, V1))
        Some(Expr::Binary(
            OpKind::Sub,
            b(Expr::Binary(OpKind::Mul, b(Expr::Var(0)), b(Expr::Var(0)))),
            b(Expr::Binary(OpKind::Mul, b(Expr::Var(1)), b(Expr::Var(1)))),
        ))
    }

    fn rhs_template(&self) -> Option<Expr> {
        // Mul(Add(V0, V1), Sub(V0, V1))
        Some(Expr::Binary(
            OpKind::Mul,
            b(Expr::Binary(OpKind::Add, b(Expr::Var(0)), b(Expr::Var(1)))),
            b(Expr::Binary(OpKind::Sub, b(Expr::Var(0)), b(Expr::Var(1)))),
        ))
    }
}

impl DiffOfSquares {
    /// Check if an e-class contains Mul(x, x) for some x, and return x.
    fn extract_self_mul(&self, egraph: &EGraph, id: EClassId) -> Option<EClassId> {
        for node in egraph.nodes(id) {
            if let ENode::Op { op, children } = node {
                if op.kind() == OpKind::Mul
                    && children.len() == 2
                    && egraph.find(children[0]) == egraph.find(children[1])
                {
                    return Some(children[0]);
                }
            }
        }
        None
    }
}

// ============================================================================
// Rule collection
// ============================================================================

/// All power and algebraic strength reduction rules (11 rules).
///
/// - 6 PowSpecialValue instances (power-zero, power-identity, power-expand-2,
///   power-sqrt, power-recip, power-rsqrt)
/// - 1 PowerRecurrence (tower rule)
/// - 2 LogPower (ln, log2)
/// - 1 ExpandSquare
/// - 1 DiffOfSquares
pub fn power_rules() -> Vec<Box<dyn Rewrite>> {
    let mut rules = pow_special_value_rules();
    // Tower rule
    rules.push(PowerRecurrence::new());
    // Log-power (2 rules)
    rules.push(LogPower::ln());
    rules.push(LogPower::log2());
    // Algebraic identities (2 rules)
    rules.push(ExpandSquare::new());
    rules.push(DiffOfSquares::new());
    rules
}
