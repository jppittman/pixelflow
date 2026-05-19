//! CPU instruction fusion rules for training data generation.
//!
//! These rules encode knowledge about CPU instruction sets and performance.
//! They live here (in the pipeline) because:
//! 1. They're used for training the Judge neural network
//! 2. They teach the Judge about instruction selection trade-offs
//! 3. Mathematical identities remain in pixelflow_search::math
//!
//! ## Rules
//!
//! - **FMA Fusion**: `a * b + c` → `muladd(a, b, c)`
//! - **Reciprocal Square Root**: `1 / sqrt(x)` → `rsqrt(x)`

use pixelflow_ir::OpKind;
use pixelflow_search::egraph::{EGraph, EClassId, ENode, ops};
use pixelflow_search::egraph::rewrite::{Rewrite, RewriteAction};

// ============================================================================
// FMA Fusion
// ============================================================================

/// Fused Multiply-Add: a * b + c → muladd(a, b, c)
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
    fn name(&self) -> &str { "fma-fusion" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else { return None };
        if op.kind() != OpKind::Add { return None; }
        if children.len() != 2 { return None; }

        let left = children[0];
        let right = children[1];

        if let Some((a, b)) = extract_mul(egraph, left) {
            return Some(RewriteAction::Create(ENode::Op {
                op: &ops::MulAdd,
                children: vec![a, b, right],
            }));
        }

        if let Some((a, b)) = extract_mul(egraph, right) {
            return Some(RewriteAction::Create(ENode::Op {
                op: &ops::MulAdd,
                children: vec![a, b, left],
            }));
        }

        None
    }
}

// ============================================================================
// Reciprocal Square Root
// ============================================================================

/// Reciprocal square root: 1 / sqrt(x) → rsqrt(x)
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
    fn name(&self) -> &str { "recip-sqrt" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Op { op, children } = node else { return None };

        match op.kind() {
            OpKind::Div if children.len() == 2 => {
                let num = children[0];
                let denom = children[1];

                if !is_one(egraph, num) {
                    return None;
                }

                if let Some(x) = extract_sqrt(egraph, denom) {
                    return Some(RewriteAction::Create(ENode::Op {
                        op: &ops::Rsqrt,
                        children: vec![x],
                    }));
                }
            }
            OpKind::Recip if children.len() == 1 => {
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

/// CPU instruction fusion rules for training.
pub fn fusion_rules() -> Vec<Box<dyn Rewrite>> {
    vec![
        FmaFusion::new(),
        RecipSqrt::new(),
    ]
}
