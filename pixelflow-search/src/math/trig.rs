//! Trigonometric identities derived from angle addition formulas.
//!
//! The core insight: most trig identities derive from just two formulas:
//! - sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
//! - cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
//!
//! Combined with canonicalization (a - b → a + neg(b)) and the Parity trait
//! (sin is odd, cos is even), we get:
//! - sin(a - b) via canonicalization then angle addition
//! - sin(2a) = sin(a + a) when e-graph sees a + a
//! - cos(2a) variants from angle addition + Pythagorean
//!
//! The Pythagorean identity sin²(x) + cos²(x) = 1 is explicit (not derivable).
//!
//! Deep connection: These rules ARE the exponential Homomorphism rules via
//! Euler's identity e^(ix) = cos(x) + i·sin(x). The angle addition formulas
//! are just the real/imaginary parts of e^(i(a+b)) = e^(ia) · e^(ib).

use std::marker::PhantomData;
use std::sync::Arc;

use crate::egraph::Pattern as Expr;
use crate::egraph::{EClassId, EGraph, ENode, Op, Rewrite, RewriteAction, ops};
use pixelflow_ir::OpKind;

fn b(e: Expr) -> Arc<Expr> {
    Arc::new(e)
}

// ============================================================================
// AngleAddition Trait
// ============================================================================

/// The sign of the second term in an angle addition formula.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sign {
    Plus,
    Minus,
}

/// Expansion coefficients for angle addition formulas.
///
/// f(a + b) = term1_op1(a) * term1_op2(b) + sign * term2_op1(a) * term2_op2(b)
#[derive(Debug, Clone, Copy)]
pub struct AngleExpansion {
    /// First term: op1(a) * op2(b)
    pub term1_op1: &'static dyn Op,
    pub term1_op2: &'static dyn Op,
    /// Second term: sign * op1(a) * op2(b)
    pub term2_op1: &'static dyn Op,
    pub term2_op2: &'static dyn Op,
    pub term2_sign: Sign,
}

/// Angle addition formulas for trig functions.
///
/// Implementing this trait derives the rule:
/// f(a + b) → term1_op1(a)*term1_op2(b) + sign*term2_op1(a)*term2_op2(b)
pub trait AngleAddition: Send + Sync {
    /// The trig operation (sin or cos).
    fn op() -> &'static dyn Op;
    /// The expansion formula.
    fn expansion() -> AngleExpansion;
}

// ============================================================================
// AngleAddition Declarations
// ============================================================================

/// sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
pub struct SinAngleAddition;

impl AngleAddition for SinAngleAddition {
    fn op() -> &'static dyn Op {
        &ops::Sin
    }
    fn expansion() -> AngleExpansion {
        AngleExpansion {
            term1_op1: &ops::Sin,
            term1_op2: &ops::Cos,
            term2_op1: &ops::Cos,
            term2_op2: &ops::Sin,
            term2_sign: Sign::Plus,
        }
    }
}

/// cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
pub struct CosAngleAddition;

impl AngleAddition for CosAngleAddition {
    fn op() -> &'static dyn Op {
        &ops::Cos
    }
    fn expansion() -> AngleExpansion {
        AngleExpansion {
            term1_op1: &ops::Cos,
            term1_op2: &ops::Cos,
            term2_op1: &ops::Sin,
            term2_op2: &ops::Sin,
            term2_sign: Sign::Minus,
        }
    }
}

// ============================================================================
// Derived Rules
// ============================================================================

/// Rule for angle addition: Op(a + b) → expansion
///
/// Matches: Sin(Add(a, b)) or Cos(Add(a, b))
/// Creates: term1_op1(a)*term1_op2(b) +/- term2_op1(a)*term2_op2(b)
pub struct AngleAdditionRule<T: AngleAddition>(PhantomData<T>);

impl<T: AngleAddition> AngleAdditionRule<T> {
    pub fn new() -> Box<Self> {
        Box::new(Self(PhantomData))
    }
}

impl<T: AngleAddition> Default for AngleAdditionRule<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T: AngleAddition> Rewrite for AngleAdditionRule<T> {
    fn name(&self) -> &str {
        // Use a static string based on the op kind
        match T::op().kind() {
            OpKind::Sin => "sin-angle-addition",
            OpKind::Cos => "cos-angle-addition",
            _ => "angle-addition",
        }
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: Op(Add(a, b))
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != T::op().kind() {
            return None;
        }
        if children.len() != 1 {
            return None;
        }

        let arg = children[0];

        // Check if argument is Add(a, b)
        for arg_node in egraph.nodes(arg) {
            if let ENode::Op {
                op: arg_op,
                children: arg_children,
            } = arg_node
            {
                if arg_op.kind() == OpKind::Add && arg_children.len() == 2 {
                    let a = arg_children[0];
                    let b = arg_children[1];
                    let exp = T::expansion();

                    // Create the expansion:
                    // term1_op1(a)*term1_op2(b) + sign*term2_op1(a)*term2_op2(b)
                    return Some(RewriteAction::AngleAddition {
                        term1_op1: exp.term1_op1,
                        term1_op2: exp.term1_op2,
                        term2_op1: exp.term2_op1,
                        term2_op2: exp.term2_op2,
                        term2_sign: exp.term2_sign,
                        a,
                        b,
                    });
                }
            }
        }
        None
    }

    fn lhs_template(&self) -> Option<Expr> {
        // Op(Add(V0, V1))
        Some(Expr::Unary(
            T::op().kind(),
            b(Expr::Binary(OpKind::Add, b(Expr::Var(0)), b(Expr::Var(1)))),
        ))
    }

    fn rhs_template(&self) -> Option<Expr> {
        let exp = T::expansion();
        // term1: Mul(term1_op1(V0), term1_op2(V1))
        let term1 = Expr::Binary(
            OpKind::Mul,
            b(Expr::Unary(exp.term1_op1.kind(), b(Expr::Var(0)))),
            b(Expr::Unary(exp.term1_op2.kind(), b(Expr::Var(1)))),
        );
        // term2: Mul(term2_op1(V0), term2_op2(V1))
        let term2 = Expr::Binary(
            OpKind::Mul,
            b(Expr::Unary(exp.term2_op1.kind(), b(Expr::Var(0)))),
            b(Expr::Unary(exp.term2_op2.kind(), b(Expr::Var(1)))),
        );
        // Combine: Plus -> Add(term1, term2), Minus -> Sub(term1, term2)
        let combiner = match exp.term2_sign {
            Sign::Plus => OpKind::Add,
            Sign::Minus => OpKind::Sub,
        };
        Some(Expr::Binary(combiner, b(term1), b(term2)))
    }
}

// ============================================================================
// Pythagorean Identity
// ============================================================================

/// Pythagorean identity: sin²(x) + cos²(x) → 1
///
/// This is an explicit rule that cannot be derived from angle addition.
/// Matches: Add(Mul(Sin(x), Sin(x)), Mul(Cos(y), Cos(y))) where x == y
pub struct Pythagorean;

impl Pythagorean {
    pub fn new() -> Box<Self> {
        Box::new(Self)
    }
}

impl Rewrite for Pythagorean {
    fn name(&self) -> &str {
        "pythagorean"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: Add(A, B) where A = sin²(x), B = cos²(y), x == y
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Add {
            return None;
        }
        if children.len() != 2 {
            return None;
        }

        let left = children[0];
        let right = children[1];

        // Try to find sin²(x) + cos²(x) pattern
        if let Some(_x) = self.match_pythagorean_pair(egraph, left, right) {
            // sin²(x) + cos²(x) → 1
            return Some(RewriteAction::Create(ENode::Const(1.0_f32.to_bits())));
        }

        // Also try cos²(x) + sin²(x) (commuted)
        if let Some(_x) = self.match_pythagorean_pair(egraph, right, left) {
            return Some(RewriteAction::Create(ENode::Const(1.0_f32.to_bits())));
        }

        None
    }

    fn lhs_template(&self) -> Option<Expr> {
        // Add(Mul(Sin(V0), Sin(V0)), Mul(Cos(V0), Cos(V0)))
        let sin_v0 = Expr::Unary(OpKind::Sin, b(Expr::Var(0)));
        let cos_v0 = Expr::Unary(OpKind::Cos, b(Expr::Var(0)));
        Some(Expr::Binary(
            OpKind::Add,
            b(Expr::Binary(OpKind::Mul, b(sin_v0.clone()), b(sin_v0))),
            b(Expr::Binary(OpKind::Mul, b(cos_v0.clone()), b(cos_v0))),
        ))
    }

    fn rhs_template(&self) -> Option<Expr> {
        // Const(1.0)
        Some(Expr::Const(1.0))
    }
}

impl Pythagorean {
    /// Check if left = sin²(x) and right = cos²(x) for same x.
    /// Returns Some(x) if pattern matches.
    fn match_pythagorean_pair(
        &self,
        egraph: &EGraph,
        left: EClassId,
        right: EClassId,
    ) -> Option<EClassId> {
        // Check left for sin²(x) pattern: Mul(Sin(x), Sin(x)) or Pow(Sin(x), 2)
        let sin_x = self.extract_squared_trig(egraph, left, OpKind::Sin)?;

        // Check right for cos²(x) pattern
        let cos_x = self.extract_squared_trig(egraph, right, OpKind::Cos)?;

        // Check if x is the same (same e-class)
        if egraph.find(sin_x) == egraph.find(cos_x) {
            Some(sin_x)
        } else {
            None
        }
    }

    /// Extract x from trig(x)² pattern (either Mul(trig(x), trig(x)) or Pow(trig(x), 2))
    fn extract_squared_trig(
        &self,
        egraph: &EGraph,
        class: EClassId,
        trig_op: OpKind,
    ) -> Option<EClassId> {
        for node in egraph.nodes(class) {
            if let ENode::Op { op, children } = node {
                // Pattern 1: Mul(trig(x), trig(x))
                if op.kind() == OpKind::Mul && children.len() == 2 {
                    let a = children[0];
                    let b = children[1];
                    if egraph.find(a) == egraph.find(b) {
                        // Both children are the same, check if it's trig(x)
                        if let Some(x) = self.extract_trig_arg(egraph, a, trig_op) {
                            return Some(x);
                        }
                    }
                }

                // Pattern 2: Pow(trig(x), 2) - if we have Pow op
                // For now, just handle the Mul pattern
            }
        }
        None
    }

    /// Extract x from trig(x)
    fn extract_trig_arg(
        &self,
        egraph: &EGraph,
        class: EClassId,
        trig_op: OpKind,
    ) -> Option<EClassId> {
        for node in egraph.nodes(class) {
            if let ENode::Op { op, children } = node {
                if op.kind() == trig_op && children.len() == 1 {
                    return Some(children[0]);
                }
            }
        }
        None
    }
}

// ============================================================================
// Reverse Angle Addition (Product-to-Sum)
// ============================================================================

/// Reverse angle addition: sin(a)cos(b) + cos(a)sin(b) → sin(a + b)
///
/// This is the inverse of angle addition. Combined with the forward rule,
/// it allows the e-graph to discover double angle identities:
///   - sin(x)*cos(x) appears in sin(x)cos(x) + cos(x)sin(x) = 2*sin(x)*cos(x)
///   - reverse rule recognizes this as sin(x + x) = sin(2x)
///   - therefore sin(x)*cos(x) = sin(2x)/2
pub struct ReverseAngleAddition;

impl ReverseAngleAddition {
    pub fn new() -> Box<Self> {
        Box::new(Self)
    }
}

impl Rewrite for ReverseAngleAddition {
    fn name(&self) -> &str {
        "reverse-angle-addition"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: Add(Mul(sin(a), cos(b)), Mul(cos(a), sin(b)))
        // → sin(a + b)
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Add {
            return None;
        }
        if children.len() != 2 {
            return None;
        }

        let left = children[0];
        let right = children[1];

        // Try to match sin(a)cos(b) + cos(a)sin(b) pattern
        if let Some((a, b)) = self.match_sin_angle_sum(egraph, left, right) {
            // Create sin(a + b)
            return Some(RewriteAction::ReverseAngleAddition {
                trig_op: &ops::Sin,
                a,
                b,
            });
        }

        // Try to match cos(a)cos(b) - sin(a)sin(b) pattern for cos(a+b)
        // (Note: This would need Sub matching, which is canonicalized to Add(_, Neg(_)))
        // For now, just handle the sin case

        None
    }

    fn lhs_template(&self) -> Option<Expr> {
        // Add(Mul(Sin(V0), Cos(V1)), Mul(Cos(V0), Sin(V1)))
        Some(Expr::Binary(
            OpKind::Add,
            b(Expr::Binary(
                OpKind::Mul,
                b(Expr::Unary(OpKind::Sin, b(Expr::Var(0)))),
                b(Expr::Unary(OpKind::Cos, b(Expr::Var(1)))),
            )),
            b(Expr::Binary(
                OpKind::Mul,
                b(Expr::Unary(OpKind::Cos, b(Expr::Var(0)))),
                b(Expr::Unary(OpKind::Sin, b(Expr::Var(1)))),
            )),
        ))
    }

    fn rhs_template(&self) -> Option<Expr> {
        // Sin(Add(V0, V1))
        Some(Expr::Unary(
            OpKind::Sin,
            b(Expr::Binary(OpKind::Add, b(Expr::Var(0)), b(Expr::Var(1)))),
        ))
    }
}

impl ReverseAngleAddition {
    /// Match pattern: Mul(sin(a), cos(b)) in one class, Mul(cos(a), sin(b)) in other
    fn match_sin_angle_sum(
        &self,
        egraph: &EGraph,
        left: EClassId,
        right: EClassId,
    ) -> Option<(EClassId, EClassId)> {
        // Check left = sin(a) * cos(b)
        let (sin_arg_l, cos_arg_l) = self.extract_sin_cos_mul(egraph, left)?;
        // Check right = cos(a) * sin(b)
        let (cos_arg_r, sin_arg_r) = self.extract_cos_sin_mul(egraph, right)?;

        // Verify a and b match across terms
        let a = egraph.find(sin_arg_l);
        let b = egraph.find(cos_arg_l);

        if egraph.find(cos_arg_r) == a && egraph.find(sin_arg_r) == b {
            return Some((a, b));
        }

        None
    }

    /// Extract (sin_arg, cos_arg) from Mul(sin(x), cos(y)) or Mul(cos(y), sin(x))
    fn extract_sin_cos_mul(
        &self,
        egraph: &EGraph,
        class: EClassId,
    ) -> Option<(EClassId, EClassId)> {
        for node in egraph.nodes(class) {
            if let ENode::Op { op, children } = node {
                if op.kind() != OpKind::Mul || children.len() != 2 {
                    continue;
                }
                let l = children[0];
                let r = children[1];

                // Try sin(a) * cos(b)
                if let (Some(sin_a), Some(cos_b)) = (
                    self.extract_trig_arg(egraph, l, OpKind::Sin),
                    self.extract_trig_arg(egraph, r, OpKind::Cos),
                ) {
                    return Some((sin_a, cos_b));
                }

                // Try cos(b) * sin(a)
                if let (Some(cos_b), Some(sin_a)) = (
                    self.extract_trig_arg(egraph, l, OpKind::Cos),
                    self.extract_trig_arg(egraph, r, OpKind::Sin),
                ) {
                    return Some((sin_a, cos_b));
                }
            }
        }
        None
    }

    /// Extract (cos_arg, sin_arg) from Mul(cos(x), sin(y)) or Mul(sin(y), cos(x))
    fn extract_cos_sin_mul(
        &self,
        egraph: &EGraph,
        class: EClassId,
    ) -> Option<(EClassId, EClassId)> {
        for node in egraph.nodes(class) {
            if let ENode::Op { op, children } = node {
                if op.kind() != OpKind::Mul || children.len() != 2 {
                    continue;
                }
                let l = children[0];
                let r = children[1];

                // Try cos(a) * sin(b)
                if let (Some(cos_a), Some(sin_b)) = (
                    self.extract_trig_arg(egraph, l, OpKind::Cos),
                    self.extract_trig_arg(egraph, r, OpKind::Sin),
                ) {
                    return Some((cos_a, sin_b));
                }

                // Try sin(b) * cos(a)
                if let (Some(sin_b), Some(cos_a)) = (
                    self.extract_trig_arg(egraph, l, OpKind::Sin),
                    self.extract_trig_arg(egraph, r, OpKind::Cos),
                ) {
                    return Some((cos_a, sin_b));
                }
            }
        }
        None
    }

    fn extract_trig_arg(
        &self,
        egraph: &EGraph,
        class: EClassId,
        trig_op: OpKind,
    ) -> Option<EClassId> {
        for node in egraph.nodes(class) {
            if let ENode::Op { op, children } = node {
                if op.kind() == trig_op && children.len() == 1 {
                    return Some(children[0]);
                }
            }
        }
        None
    }
}

// ============================================================================
// Half Angle Product Identity
// ============================================================================

/// Half angle product: sin(x)*cos(x) → sin(x + x)/2 = sin(2x)/2
///
/// This rule directly handles the common case where we have sin(x)*cos(x)
/// with the same argument. It complements the reverse angle addition rule
/// which only fires on the full sum pattern.
///
/// Derivation:
///   sin(2x) = sin(x + x) = sin(x)cos(x) + cos(x)sin(x) = 2*sin(x)*cos(x)
///   Therefore: sin(x)*cos(x) = sin(2x)/2
pub struct HalfAngleProduct;

impl HalfAngleProduct {
    pub fn new() -> Box<Self> {
        Box::new(Self)
    }
}

impl Rewrite for HalfAngleProduct {
    fn name(&self) -> &str {
        "half-angle-product"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: Mul(sin(x), cos(y)) or Mul(cos(x), sin(y)) where x == y
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

        // Try sin(x) * cos(x) pattern
        if let Some(x) = self.match_sin_cos_same_arg(egraph, left, right) {
            // sin(x) * cos(x) → sin(x + x) / 2
            return Some(RewriteAction::HalfAngleProduct { x });
        }

        // Also try cos(x) * sin(x) (commuted)
        if let Some(x) = self.match_sin_cos_same_arg(egraph, right, left) {
            return Some(RewriteAction::HalfAngleProduct { x });
        }

        None
    }

    fn lhs_template(&self) -> Option<Expr> {
        // Mul(Sin(V0), Cos(V0))
        Some(Expr::Binary(
            OpKind::Mul,
            b(Expr::Unary(OpKind::Sin, b(Expr::Var(0)))),
            b(Expr::Unary(OpKind::Cos, b(Expr::Var(0)))),
        ))
    }

    fn rhs_template(&self) -> Option<Expr> {
        // Div(Sin(Add(V0, V0)), Const(2.0))
        // Matches what the e-graph action actually produces
        Some(Expr::Binary(
            OpKind::Div,
            b(Expr::Unary(
                OpKind::Sin,
                b(Expr::Binary(OpKind::Add, b(Expr::Var(0)), b(Expr::Var(0)))),
            )),
            b(Expr::Const(2.0)),
        ))
    }
}

impl HalfAngleProduct {
    /// Check if we have sin(x) and cos(x) with same argument
    fn match_sin_cos_same_arg(
        &self,
        egraph: &EGraph,
        sin_class: EClassId,
        cos_class: EClassId,
    ) -> Option<EClassId> {
        let sin_arg = self.extract_trig_arg(egraph, sin_class, OpKind::Sin)?;
        let cos_arg = self.extract_trig_arg(egraph, cos_class, OpKind::Cos)?;

        if egraph.find(sin_arg) == egraph.find(cos_arg) {
            Some(sin_arg)
        } else {
            None
        }
    }

    fn extract_trig_arg(
        &self,
        egraph: &EGraph,
        class: EClassId,
        trig_op: OpKind,
    ) -> Option<EClassId> {
        for node in egraph.nodes(class) {
            if let ENode::Op { op, children } = node {
                if op.kind() == trig_op && children.len() == 1 {
                    return Some(children[0]);
                }
            }
        }
        None
    }
}

// ============================================================================
// Rule Collection
// ============================================================================

/// All trigonometric rules.
///
/// The rules work together to discover double angle identities:
///   - Forward angle addition: sin(x + x) → sin(x)cos(x) + cos(x)sin(x)
///   - Reverse angle addition: sin(a)cos(b) + cos(a)sin(b) → sin(a + b)
///   - Half angle product: sin(x)*cos(x) → sin(x + x)/2 (when a == b)
///
/// This means sin(x)*cos(x) and sin(2x)/2 end up in the same equivalence class
/// WITHOUT needing an explicit "double angle" rule.
pub fn trig_rules() -> Vec<Box<dyn Rewrite>> {
    vec![
        // Angle addition formulas (forward direction)
        AngleAdditionRule::<SinAngleAddition>::new(),
        AngleAdditionRule::<CosAngleAddition>::new(),
        // Reverse angle addition (sum-to-product)
        ReverseAngleAddition::new(),
        // Half angle product (single product case)
        HalfAngleProduct::new(),
        // Pythagorean identity
        Pythagorean::new(),
    ]
}
