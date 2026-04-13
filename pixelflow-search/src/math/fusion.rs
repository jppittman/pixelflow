//! CPU instruction fusion rewrite rules.
//!
//! These rules encode knowledge about CPU instruction sets, not mathematical
//! identities. They transform patterns into fused instructions that execute
//! as a single operation on modern hardware.
//!
//! ## FMA (Fused Multiply-Add)
//! `a * b + c` → `muladd(a, b, c)` — single instruction on AVX2/NEON
//!
//! ## Reciprocal Square Root (rsqrt)
//! `1 / sqrt(x)` → `rsqrt(x)` — fast approximate on x86 (rsqrtps/vrsqrtps)
//!

use std::sync::Arc;

use crate::egraph::{EClassId, EGraph, ENode, Rewrite, RewriteAction, ops};
use crate::egraph::Pattern as Expr;
use pixelflow_ir::OpKind;

fn b(e: Expr) -> Arc<Expr> {
    Arc::new(e)
}

// ============================================================================
// FMA Fusion
// ============================================================================

/// Fused Multiply-Add: a * b + c → muladd(a, b, c)
///
/// Modern CPUs (AVX2, ARM NEON) have single-instruction FMA.
/// This reduces latency and improves numerical precision (one rounding).
pub struct FmaFusion;

impl FmaFusion {
    pub fn new() -> Box<Self> {
        Box::new(Self)
    }
}

impl Default for FmaFusion {
    fn default() -> Self {
        Self
    }
}

impl Rewrite for FmaFusion {
    fn name(&self) -> &str {
        "fma-fusion"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: Add(Mul(a, b), c)
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

        // Try left = Mul(a, b), right = c
        if let Some((a, b)) = extract_mul(egraph, left) {
            return Some(RewriteAction::Create(ENode::Op {
                op: &ops::MulAdd,
                children: vec![a, b, right],
            }));
        }

        // Try left = c, right = Mul(a, b)
        if let Some((a, b)) = extract_mul(egraph, right) {
            return Some(RewriteAction::Create(ENode::Op {
                op: &ops::MulAdd,
                children: vec![a, b, left],
            }));
        }

        None
    }

    fn lhs_template(&self) -> Option<Expr> {
        // Add(Mul(V0, V1), V2)
        Some(Expr::Binary(
            OpKind::Add,
            b(Expr::Binary(OpKind::Mul, b(Expr::Var(0)), b(Expr::Var(1)))),
            b(Expr::Var(2)),
        ))
    }

    fn rhs_template(&self) -> Option<Expr> {
        // MulAdd(V0, V1, V2)
        Some(Expr::Ternary(
            OpKind::MulAdd,
            b(Expr::Var(0)),
            b(Expr::Var(1)),
            b(Expr::Var(2)),
        ))
    }
}

// ============================================================================
// Reciprocal Square Root
// ============================================================================

/// Reciprocal square root: 1 / sqrt(x) → rsqrt(x)
///
/// Common pattern in vector normalization: v / |v| = v * rsqrt(dot(v,v))
/// CPUs have fast approximate rsqrt (rsqrtps/vrsqrtps on x86).
pub struct RecipSqrt;

impl RecipSqrt {
    pub fn new() -> Box<Self> {
        Box::new(Self)
    }
}

impl Default for RecipSqrt {
    fn default() -> Self {
        Self
    }
}

impl Rewrite for RecipSqrt {
    fn name(&self) -> &str {
        "recip-sqrt"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: Div(1, Sqrt(x)) or Recip(Sqrt(x))
        let ENode::Op { op, children } = node else {
            return None;
        };

        match op.kind() {
            OpKind::Div if children.len() == 2 => {
                // Check if numerator is 1
                let num = children[0];
                let denom = children[1];

                if !is_one(egraph, num) {
                    return None;
                }

                // Check if denominator is sqrt(x)
                if let Some(x) = extract_sqrt(egraph, denom) {
                    return Some(RewriteAction::Create(ENode::Op {
                        op: &ops::Rsqrt,
                        children: vec![x],
                    }));
                }
            }
            OpKind::Recip if children.len() == 1 => {
                // Check if argument is sqrt(x)
                if let Some(x) = extract_sqrt(egraph, children[0]) {
                    return Some(RewriteAction::Create(ENode::Op {
                        op: &ops::Rsqrt,
                        children: vec![x],
                    }));
                }
            }
            _ => {}
        }

        None
    }

    fn lhs_template(&self) -> Option<Expr> {
        // Recip(Sqrt(V0))
        Some(Expr::Unary(
            OpKind::Recip,
            b(Expr::Unary(OpKind::Sqrt, b(Expr::Var(0)))),
        ))
    }

    fn rhs_template(&self) -> Option<Expr> {
        // Rsqrt(V0)
        Some(Expr::Unary(OpKind::Rsqrt, b(Expr::Var(0))))
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn extract_mul(egraph: &EGraph, class: EClassId) -> Option<(EClassId, EClassId)> {
    for node in egraph.nodes(class) {
        if let ENode::Op { op, children } = node {
            if op.kind() == OpKind::Mul && children.len() == 2 {
                return Some((children[0], children[1]));
            }
        }
    }
    None
}

fn is_one(egraph: &EGraph, class: EClassId) -> bool {
    for node in egraph.nodes(class) {
        if let ENode::Const(bits) = node {
            let v = f32::from_bits(*bits);
            if (v - 1.0).abs() < 1e-10 {
                return true;
            }
        }
    }
    false
}

fn extract_sqrt(egraph: &EGraph, class: EClassId) -> Option<EClassId> {
    for node in egraph.nodes(class) {
        if let ENode::Op { op, children } = node {
            if op.kind() == OpKind::Sqrt && children.len() == 1 {
                return Some(children[0]);
            }
        }
    }
    None
}

// ============================================================================
// Rule Collection
// ============================================================================

/// All CPU instruction fusion rules.
///
/// These are performance optimization rules, not mathematical identities:
/// - FMA: `a * b + c` → `muladd(a, b, c)` (FmaFusion)
/// - Rsqrt: `1/sqrt(x)` → `rsqrt(x)` (RecipSqrt)
pub fn fusion_rules() -> Vec<Box<dyn Rewrite>> {
    vec![FmaFusion::new(), RecipSqrt::new()]
}
