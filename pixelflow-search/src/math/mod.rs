//! Mathematical rewrite rules organized by algebraic structure.
//!
//! This module provides categorical, trait-based rule derivation. Instead of
//! enumerating identities, we declare algebraic properties and derive rules.
//!
//! ## Module Organization
//!
//! - [`algebra`]: Core algebraic structures (InversePair, Commutative, Identity, etc.)
//! - [`parity`]: Even/odd function symmetry (sin is odd, cos is even)
//! - [`trig`]: Trigonometric identities (angle addition, Pythagorean)
//! - [`exp`]: Exponential/logarithmic identities (inverse pairs, homomorphisms)
//!
//! - [`fusion`]: CPU instruction fusion (FMA, rsqrt)
//!
//! ## Math vs Fusion
//!
//! Mathematical rules are algebraic identities (true on all hardware).
//! Fusion rules encode CPU instruction knowledge (FMA, rsqrt) that is
//! architecture-aware. Both categories are rewrite rules and live here.
//!
//! ## Design Philosophy
//!
//! Rules are derived from algebraic properties, not enumerated:
//!
//! ```text
//! // One trait declaration...
//! impl InversePair for AddNeg {
//!     fn base() -> &'static dyn Op { &ops::Add }
//!     fn inverse() -> &'static dyn Op { &ops::Neg }
//!     fn derived() -> &'static dyn Op { &ops::Sub }
//!     fn identity() -> f32 { 0.0 }
//! }
//!
//! // ...yields four rules:
//! // - Canonicalize: a - b → a + neg(b)
//! // - Involution: neg(neg(x)) → x
//! // - Cancellation: (x + a) - a → x
//! // - InverseAnnihilation: x + neg(x) → 0
//! ```
//!
//! ## Categorical Structure
//!
//! The traits reflect mathematical categories:
//!
//! - **InversePair**: Group structure (operation + inverse + identity)
//! - **Parity**: Z₂ action (negation symmetry)
//! - **AngleAddition**: Lie group structure (angle as group element)
//! - **FunctionInverse**: Bijection (forward/backward maps)
//! - **Homomorphism**: Structure-preserving maps between algebraic structures
//!
//! The deep insight: Many identities are the same identity in different
//! presentations. For example, the exp Homomorphism (exp(a+b) = exp(a)*exp(b))
//! IS the trig angle addition rule via Euler's identity.

pub mod algebra;
pub mod exp;
pub mod fusion;
pub mod inflate;
pub mod parity;
pub mod power;
pub mod trig;

/// Round 2, mode (iii) — genuinely new rewrite rules, harness-only.
/// See [`round2_rules::experimental_rules`]. Never referenced by
/// [`all_rules`] or [`all_math_rules`].
pub mod round2_rules;

#[cfg(test)]
#[cfg(test)]
mod pict_rewrite_tests; // PICT-style pairwise testing of the rewrite rules (POC)

use crate::egraph::rewrite::Rewrite;

// Re-export key types for convenience
pub use algebra::{
    AddNeg, Annihilator, Associative, Commutative, Identity, InversePair, MulRecip,
    ReverseAssociative, algebra_rules, basic_algebra_rules, inverse_pair_rules,
};
pub use exp::{
    Exp2Log2, ExpHomomorphism, ExpLn, FunctionInverse, Homomorphism, LnHomomorphism, exp_rules,
};
pub use fusion::{FmaFusion, RecipSqrt, fusion_rules};
pub use parity::{
    AbsParity, AsinParity, AtanParity, CosParity, Parity, ParityKind, SinParity, TanParity,
    parity_rules,
};
pub use power::power_rules;
pub use trig::{
    AngleAddition, AngleExpansion, CosAngleAddition, Sign, SinAngleAddition, trig_rules,
};

/// All mathematical rewrite rules.
///
/// This is the primary entry point for getting all math rules. Categories:
/// - Algebra (30 rules): 8 InversePair (AddNeg/MulRecip × 4 each) + 22 basic
///   (constant fold, commutative×4, identity×2, annihilator, idempotent×2,
///    distributive, factor, doubling, halving, associative×4, reverse-associative×4)
/// - Parity (6 rules): sin, cos, tan, asin, atan, abs negation symmetry
/// - Trig (5 rules): angle addition×2, reverse angle addition, half angle, Pythagorean
/// - Exp (7 rules): function inverse cancellation×4, homomorphisms×2, power combine
/// - Power (11 rules): special values×6, recurrence, log-power×2, expand-square,
///   diff-of-squares
///
/// Total: 59 math rules
///
/// For the full set including fusion (FMA, rsqrt) and differentiation rules,
/// use [`all_rules`] which returns 62 rules.
pub fn all_math_rules() -> Vec<Box<dyn Rewrite>> {
    let mut rules = Vec::new();
    rules.extend(algebra_rules());
    rules.extend(parity_rules());
    rules.extend(trig_rules());
    rules.extend(exp_rules());
    rules.extend(power_rules());
    rules
}

/// All rewrite rules: math (59) + fusion (2) + differentiation (1) = 62 total.
///
/// This is the complete rule set for optimization. Use this for training
/// and production optimization where all rules should be available. The
/// differentiation rule is inert unless the expression contains a `Dwrt`
/// node, so it costs nothing for derivative-free kernels.
pub fn all_rules() -> Vec<Box<dyn Rewrite>> {
    let mut rules = all_math_rules();
    rules.extend(fusion_rules());
    rules.extend(crate::egraph::derivative::derivative_rules());
    rules
}

/// Core arithmetic rules only (fast, always applicable).
///
/// Use this for quick optimization passes where trig/exp rules
/// aren't needed.
pub fn core_rules() -> Vec<Box<dyn Rewrite>> {
    algebra_rules()
}

/// Transcendental function rules (trig, exp, log).
///
/// Use this when optimizing expressions with transcendental functions.
pub fn transcendental_rules() -> Vec<Box<dyn Rewrite>> {
    let mut rules = Vec::new();
    rules.extend(parity_rules());
    rules.extend(trig_rules());
    rules.extend(exp_rules());
    rules
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena_pat;
    use crate::egraph::{EClassId, EGraph, ENode, saturate_with_budget};
    use pixelflow_ir::OpKind;
    use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};

    /// Insert an arena subtree into an e-graph, returning its e-class.
    fn expr_to_egraph(arena: &ExprArena, id: ExprId, egraph: &mut EGraph) -> EClassId {
        match *arena.node(id) {
            ExprNode::Var(idx) => egraph.add(ENode::Var(idx)),
            ExprNode::Const(val) => egraph.add(ENode::Const(val.to_bits())),
            ExprNode::Param(i) => panic!("Param({i}) reached math tests"),
            ExprNode::Buffer(b) => panic!("Buffer({}) reached math tests", b.0),
            ExprNode::Ref(k) => panic!("Ref({k:?}) reached math tests"),
            ExprNode::Reduce { .. } => panic!("a bounded fold reached math tests"),
            ExprNode::Uniform(u) => egraph.add(ENode::Uniform(*arena.uniform_decl(u))),
            ExprNode::Unary(kind, a) => {
                let ca = expr_to_egraph(arena, a, egraph);
                let op = crate::egraph::ops::op_from_kind(kind)
                    .unwrap_or_else(|| panic!("unsupported op in math test: {kind:?}"));
                egraph.add(ENode::Op {
                    op,
                    children: vec![ca],
                })
            }
            ExprNode::Binary(kind, a, b) => {
                let ca = expr_to_egraph(arena, a, egraph);
                let cb = expr_to_egraph(arena, b, egraph);
                let op = crate::egraph::ops::op_from_kind(kind)
                    .unwrap_or_else(|| panic!("unsupported op in math test: {kind:?}"));
                egraph.add(ENode::Op {
                    op,
                    children: vec![ca, cb],
                })
            }
            ExprNode::Ternary(kind, a, b, c) => {
                let ca = expr_to_egraph(arena, a, egraph);
                let cb = expr_to_egraph(arena, b, egraph);
                let cc = expr_to_egraph(arena, c, egraph);
                let op = crate::egraph::ops::op_from_kind(kind)
                    .unwrap_or_else(|| panic!("unsupported op in math test: {kind:?}"));
                egraph.add(ENode::Op {
                    op,
                    children: vec![ca, cb, cc],
                })
            }
            ExprNode::Nary(kind, _) => panic!("unsupported n-ary op in math test: {kind:?}"),
        }
    }

    /// Materialise the cheapest representative of an e-class into `arena`.
    fn eclass_to_arena(egraph: &EGraph, class: EClassId, arena: &mut ExprArena) -> ExprId {
        // Snapshot the chosen node so the egraph borrow is released before we
        // recurse (which borrows egraph again) and push into the arena.
        let node = egraph.nodes(class)[0].clone();
        match node {
            ENode::Var(idx) => arena.push_var(idx),
            ENode::Const(bits) => arena.push_const(f32::from_bits(bits)),
            ENode::Buffer(decl) => panic!("Buffer({decl:?}) reached math tests"),
            ENode::Param(i) => panic!("Param({i}) reached math tests"),
            ENode::Reduce { .. } => panic!("a bounded fold reached math tests"),
            ENode::Uniform(decl) => {
                let slot = arena.declare_uniform(decl);
                arena.push_uniform(slot)
            }
            ENode::Op { op, children } => {
                let kind = op.kind();
                let child_ids: Vec<ExprId> = children
                    .iter()
                    .map(|&c| eclass_to_arena(egraph, c, arena))
                    .collect();
                match child_ids.len() {
                    1 => arena.push_unary(kind, child_ids[0]),
                    2 => arena.push_binary(kind, child_ids[0], child_ids[1]),
                    3 => arena.push_ternary(kind, child_ids[0], child_ids[1], child_ids[2]),
                    n => panic!("unsupported arity in math test: {n}"),
                }
            }
        }
    }

    /// A third free scalar, as the kernel argument it would be: a lattice
    /// has two axes, so `var 2` is not a coordinate. Never folded, so the
    /// tree shape the rules act on is the one the test wrote.
    fn arg(a: &mut ExprArena, default: f32) -> ExprId {
        let slot = a.declare_uniform(pixelflow_ir::Uniform::new(default).decl());
        a.push_uniform(slot)
    }

    /// Standard test points including edge cases.
    fn standard_test_points() -> Vec<[f32; 2]> {
        vec![
            [0.5, 0.7],
            [0.0, 0.0],
            [1.0, 1.0],
            [-1.0, -1.0],
            [100.0, 100.0],
            [-100.0, -100.0],
            [0.001, 0.001],
            [3.14159, 1.5708],
            [-0.5, 0.3],
        ]
    }

    #[test]
    fn associativity_templates() {
        // Verify all associativity rules have valid lhs/rhs templates and that
        // Associative LHS == ReverseAssociative RHS (and vice versa) structurally.
        let assoc = Associative::new(&crate::egraph::ops::Add);
        let rev = ReverseAssociative::new(&crate::egraph::ops::Add);

        let mut a = ExprArena::new();
        let assoc_lhs = assoc
            .lhs_template(&mut a)
            .expect("Associative Add missing lhs_template");
        let assoc_rhs = assoc
            .rhs_template(&mut a)
            .expect("Associative Add missing rhs_template");
        let rev_lhs = rev
            .lhs_template(&mut a)
            .expect("ReverseAssociative Add missing lhs_template");
        let rev_rhs = rev
            .rhs_template(&mut a)
            .expect("ReverseAssociative Add missing rhs_template");

        assert!(
            a.subtree_eq(assoc_lhs, &a, rev_rhs),
            "Associative LHS should equal ReverseAssociative RHS (same structural pattern)"
        );
        assert!(
            a.subtree_eq(assoc_rhs, &a, rev_lhs),
            "Associative RHS should equal ReverseAssociative LHS (same structural pattern)"
        );
    }

    #[test]
    fn all_rules_count() {
        // Verify we have the expected number of rules after removal.
        let rules = all_rules();
        assert_eq!(
            rules.len(),
            62,
            "Expected 62 rules (59 math + 2 fusion + 1 differentiation), got {}",
            rules.len()
        );
    }
}
