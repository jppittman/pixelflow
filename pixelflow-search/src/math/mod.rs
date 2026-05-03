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
pub mod parity;
pub mod power;
pub mod trig;

use crate::egraph::rewrite::Rewrite;

// Re-export key types for convenience
pub use algebra::{
    InversePair, AddNeg, MulRecip,
    Commutative, Identity, Annihilator, Associative, ReverseAssociative,
    algebra_rules, inverse_pair_rules, basic_algebra_rules,
};
pub use parity::{
    Parity, ParityKind,
    SinParity, CosParity, TanParity, AsinParity, AtanParity, AbsParity,
    parity_rules,
};
pub use trig::{
    AngleAddition, Sign, AngleExpansion,
    SinAngleAddition, CosAngleAddition,
    trig_rules,
};
pub use exp::{
    FunctionInverse, Homomorphism,
    ExpLn, Exp2Log2, ExpHomomorphism, LnHomomorphism,
    exp_rules,
};
pub use power::{
    power_rules,
};
pub use fusion::{
    FmaFusion, RecipSqrt,
    fusion_rules,
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
/// For the full set including fusion rules (FMA, rsqrt),
/// use [`all_rules`] which returns 61 rules.
pub fn all_math_rules() -> Vec<Box<dyn Rewrite>> {
    let mut rules = Vec::new();
    rules.extend(algebra_rules());
    rules.extend(parity_rules());
    rules.extend(trig_rules());
    rules.extend(exp_rules());
    rules.extend(power_rules());
    rules
}

/// All rewrite rules: math (59) + fusion (2) = 61 total.
///
/// This is the complete rule set for optimization. Use this for training
/// and production optimization where all rules should be available.
pub fn all_rules() -> Vec<Box<dyn Rewrite>> {
    let mut rules = all_math_rules();
    rules.extend(fusion_rules());
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
    use pixelflow_ir::{Expr, OpKind};
    use crate::egraph::{EGraph, expr_to_egraph, eclass_to_expr, saturate_with_budget, CostModel};

    fn b(e: Expr) -> Box<Expr> { Box::new(e) }

    /// Evaluate an IR Expr at given variable values.
    fn eval_expr(expr: &Expr, vars: &[f32; 4]) -> f32 {
        match expr {
            Expr::Var(i) => vars[*i as usize],
            Expr::Const(c) => *c,
            Expr::Param(_) => panic!("Param in eval_expr"),
            Expr::Unary(op, a) => {
                let a = eval_expr(a, vars);
                op.eval_unary(a).unwrap_or_else(|| panic!("eval_unary failed for {:?}", op))
            }
            Expr::Binary(op, a, b) => {
                let a = eval_expr(a, vars);
                let b = eval_expr(b, vars);
                op.eval_binary(a, b).unwrap_or_else(|| panic!("eval_binary failed for {:?}", op))
            }
            Expr::Ternary(op, a, b, c) => {
                let a = eval_expr(a, vars);
                let b = eval_expr(b, vars);
                let c = eval_expr(c, vars);
                op.eval_ternary(a, b, c).unwrap_or_else(|| panic!("eval_ternary failed for {:?}", op))
            }
            Expr::Nary(_, _) => panic!("Nary in eval_expr"),
        }
    }

    /// Run an expression through the egraph optimizer and check that the
    /// optimized result produces the same output at all test points.
    fn check_optimization_preserves_semantics(expr: &Expr, test_points: &[[f32; 4]], epsilon: f32) {
        let mut eg = EGraph::new();
        let root = expr_to_egraph(expr, &mut eg);
        let _result = saturate_with_budget(&mut eg, 200);

        // Extract optimized expression
        let optimized = eclass_to_expr(&eg, root);

        for point in test_points {
            let original = eval_expr(expr, point);
            let opt = eval_expr(&optimized, point);

            // Both NaN => OK. Both inf with same sign => OK.
            if original.is_nan() && opt.is_nan() { continue; }
            if original.is_infinite() && opt.is_infinite() && original.signum() == opt.signum() { continue; }

            let diff = (original - opt).abs();
            // Use relative error for large values
            let threshold = if original.abs() > 1.0 {
                epsilon * original.abs()
            } else {
                epsilon
            };
            assert!(
                diff <= threshold,
                "Optimization changed semantics!\n\
                 Expression: {expr}\n\
                 Optimized:  {optimized}\n\
                 Point: {point:?}\n\
                 Original: {original}\n\
                 Optimized: {opt}\n\
                 Diff: {diff} > threshold {threshold}"
            );
        }
    }

    /// Standard test points including edge cases.
    fn standard_test_points() -> Vec<[f32; 4]> {
        vec![
            [0.5, 0.7, 1.3, -0.2],      // Normal values
            [0.0, 0.0, 0.0, 0.0],        // Zeros
            [1.0, 1.0, 1.0, 1.0],        // Ones
            [-1.0, -1.0, -1.0, -1.0],    // Negative ones
            [100.0, 100.0, 100.0, 100.0], // Large values (exp overflow territory)
            [-100.0, -100.0, 0.01, 0.01], // Mixed large negative / small positive
            [0.001, 0.001, 0.001, 0.001], // Very small
            [3.14159, 1.5708, 0.7854, 2.3562], // Pi-related (trig)
            [-0.5, 0.3, -0.8, 0.1],      // Mixed sign small
        ]
    }





    #[test]
    fn test_associativity_left_to_right() {
        // (v0 + v1) + v2 should produce v0 + (v1 + v2) in the e-graph
        let v0 = Expr::Var(0);
        let v1 = Expr::Var(1);
        let v2 = Expr::Var(2);

        // Build (v0 + v1) + v2
        let left = Expr::Binary(OpKind::Add, b(v0.clone()), b(v1.clone()));
        let expr = Expr::Binary(OpKind::Add, b(left), b(v2.clone()));

        let mut eg = EGraph::with_rules(all_rules());
        let root = expr_to_egraph(&expr, &mut eg);
        // Budget 5 is sufficient — associativity fires on the first iteration.
        // Higher budgets cause combinatorial explosion with commutativity.
        let _result = saturate_with_budget(&mut eg, 5);

        // Verify semantic equivalence
        let optimized = eclass_to_expr(&eg, root);
        let pts = standard_test_points();
        for point in &pts {
            let original = eval_expr(&expr, point);
            let opt = eval_expr(&optimized, point);
            if original.is_nan() && opt.is_nan() { continue; }
            let diff = (original - opt).abs();
            let threshold = if original.abs() > 1.0 { 1e-5 * original.abs() } else { 1e-5 };
            assert!(diff <= threshold,
                "Associativity L->R changed semantics at {point:?}: {original} vs {opt}");
        }

        // Verify the e-graph contains the right-associated form by checking
        // that the root class has grown (new nodes added by associativity)
        let root = eg.find(root);
        let node_count = eg.nodes(root).len();
        assert!(node_count > 1,
            "Expected associativity to add alternative tree shapes, \
             but root class has only {node_count} node(s)");
    }

    #[test]
    fn test_associativity_right_to_left() {
        // v0 + (v1 + v2) should produce (v0 + v1) + v2 in the e-graph
        let v0 = Expr::Var(0);
        let v1 = Expr::Var(1);
        let v2 = Expr::Var(2);

        // Build v0 + (v1 + v2)
        let right = Expr::Binary(OpKind::Add, b(v1.clone()), b(v2.clone()));
        let expr = Expr::Binary(OpKind::Add, b(v0.clone()), b(right));

        let mut eg = EGraph::with_rules(all_rules());
        let root = expr_to_egraph(&expr, &mut eg);
        let _result = saturate_with_budget(&mut eg, 5);

        // Verify semantic equivalence
        let optimized = eclass_to_expr(&eg, root);
        let pts = standard_test_points();
        for point in &pts {
            let original = eval_expr(&expr, point);
            let opt = eval_expr(&optimized, point);
            if original.is_nan() && opt.is_nan() { continue; }
            let diff = (original - opt).abs();
            let threshold = if original.abs() > 1.0 { 1e-5 * original.abs() } else { 1e-5 };
            assert!(diff <= threshold,
                "Associativity R->L changed semantics at {point:?}: {original} vs {opt}");
        }

        // Verify the e-graph grew from reverse associativity
        let root = eg.find(root);
        let node_count = eg.nodes(root).len();
        assert!(node_count > 1,
            "Expected reverse associativity to add alternative tree shapes, \
             but root class has only {node_count} node(s)");
    }





    #[test]
    fn test_associativity_templates() {
        // Verify all associativity rules have valid lhs/rhs templates
        let assoc_add = Associative::new(&crate::egraph::ops::Add);
        assert!(assoc_add.lhs_template().is_some(), "Associative Add missing lhs_template");
        assert!(assoc_add.rhs_template().is_some(), "Associative Add missing rhs_template");

        let rev_assoc_add = ReverseAssociative::new(&crate::egraph::ops::Add);
        assert!(rev_assoc_add.lhs_template().is_some(), "ReverseAssociative Add missing lhs_template");
        assert!(rev_assoc_add.rhs_template().is_some(), "ReverseAssociative Add missing rhs_template");

        // Verify template structure: LHS of Associative should be RHS of ReverseAssociative
        let assoc_lhs = assoc_add.lhs_template().unwrap();
        let rev_rhs = rev_assoc_add.rhs_template().unwrap();
        assert_eq!(format!("{}", assoc_lhs), format!("{}", rev_rhs),
            "Associative LHS should equal ReverseAssociative RHS (same structural pattern)");

        let assoc_rhs = assoc_add.rhs_template().unwrap();
        let rev_lhs = rev_assoc_add.lhs_template().unwrap();
        assert_eq!(format!("{}", assoc_rhs), format!("{}", rev_lhs),
            "Associative RHS should equal ReverseAssociative LHS (same structural pattern)");
    }

    #[test]
    fn test_all_rules_count() {
        // Verify we have the expected number of rules after removal.
        let rules = all_rules();
        assert_eq!(rules.len(), 61,
            "Expected 61 rules (59 math + 2 fusion), got {}", rules.len());
    }
}
