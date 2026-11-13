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
//!     fn base() -> OpKind { OpKind::Add }
//!     fn inverse() -> OpKind { OpKind::Neg }
//!     fn derived() -> OpKind { OpKind::Sub }
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
///   distributive, factor, doubling, halving, associative×4, reverse-associative×4)
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
#[must_use]
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
#[must_use]
pub fn all_rules() -> Vec<Box<dyn Rewrite>> {
    let mut rules = all_math_rules();
    rules.extend(fusion_rules());
    rules.extend(crate::egraph::derivative::derivative_rules());
    rules
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::OpKind;
    use pixelflow_ir::arena::ExprArena;

    #[test]
    fn associativity_templates() {
        // Verify all associativity rules have valid lhs/rhs templates and that
        // Associative LHS == ReverseAssociative RHS (and vice versa) structurally.
        let assoc = Associative::new(OpKind::Add);
        let rev = ReverseAssociative::new(OpKind::Add);

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
