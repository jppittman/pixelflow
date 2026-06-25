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
            ExprNode::Nary(kind, _, _) => panic!("unsupported n-ary op in math test: {kind:?}"),
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

    /// Evaluate an arena subtree at the given variable values.
    fn eval_arena(arena: &ExprArena, id: ExprId, vars: &[f32; 4]) -> f32 {
        match *arena.node(id) {
            ExprNode::Var(i) => vars[i as usize],
            ExprNode::Const(c) => c,
            ExprNode::Param(_) => panic!("Param in eval_arena"),
            ExprNode::Unary(op, a) => {
                let a = eval_arena(arena, a, vars);
                op.eval_unary(a)
                    .unwrap_or_else(|| panic!("eval_unary failed for {op:?}"))
            }
            ExprNode::Binary(op, a, b) => {
                let a = eval_arena(arena, a, vars);
                let b = eval_arena(arena, b, vars);
                op.eval_binary(a, b)
                    .unwrap_or_else(|| panic!("eval_binary failed for {op:?}"))
            }
            ExprNode::Ternary(op, a, b, c) => {
                let a = eval_arena(arena, a, vars);
                let b = eval_arena(arena, b, vars);
                let c = eval_arena(arena, c, vars);
                op.eval_ternary(a, b, c)
                    .unwrap_or_else(|| panic!("eval_ternary failed for {op:?}"))
            }
            ExprNode::Nary(..) => panic!("Nary in eval_arena"),
        }
    }

    /// Run an expression through the egraph optimizer and check that the
    /// optimized result produces the same output at all test points.
    fn check_optimization_preserves_semantics(
        arena: &ExprArena,
        root: ExprId,
        test_points: &[[f32; 4]],
        epsilon: f32,
    ) {
        let mut eg = EGraph::new();
        let root_class = expr_to_egraph(arena, root, &mut eg);
        let _result = saturate_with_budget(&mut eg, 200);

        let mut opt_arena = ExprArena::new();
        let opt_root = eclass_to_arena(&eg, root_class, &mut opt_arena);

        for point in test_points {
            let original = eval_arena(arena, root, point);
            let opt = eval_arena(&opt_arena, opt_root, point);

            if original.is_nan() && opt.is_nan() {
                continue;
            }
            if original.is_infinite() && opt.is_infinite() && original.signum() == opt.signum() {
                continue;
            }

            let diff = (original - opt).abs();
            let threshold = if original.abs() > 1.0 {
                epsilon * original.abs()
            } else {
                epsilon
            };
            assert!(
                diff <= threshold,
                "Optimization changed semantics!\n\
                 Expression: {}\n\
                 Optimized:  {}\n\
                 Point: {point:?}\n\
                 Original: {original}\n\
                 Optimized: {opt}\n\
                 Diff: {diff} > threshold {threshold}",
                arena.display(root),
                opt_arena.display(opt_root),
            );
        }
    }

    /// Standard test points including edge cases.
    fn standard_test_points() -> Vec<[f32; 4]> {
        vec![
            [0.5, 0.7, 1.3, -0.2],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0],
            [100.0, 100.0, 100.0, 100.0],
            [-100.0, -100.0, 0.01, 0.01],
            [0.001, 0.001, 0.001, 0.001],
            [3.14159, 1.5708, 0.7854, 2.3562],
            [-0.5, 0.3, -0.8, 0.1],
        ]
    }

    /// Assert two e-graph root classes are semantically equal at all points,
    /// and that associativity added alternative tree shapes to the root class.
    fn check_assoc(arena: &ExprArena, root: ExprId) {
        let mut eg = EGraph::with_rules(all_rules());
        let root_class = expr_to_egraph(arena, root, &mut eg);
        // Budget 5: associativity fires on the first iteration. Higher budgets
        // cause combinatorial explosion with commutativity.
        let _result = saturate_with_budget(&mut eg, 5);

        let mut opt_arena = ExprArena::new();
        let opt_root = eclass_to_arena(&eg, root_class, &mut opt_arena);
        for point in &standard_test_points() {
            let original = eval_arena(arena, root, point);
            let opt = eval_arena(&opt_arena, opt_root, point);
            if original.is_nan() && opt.is_nan() {
                continue;
            }
            let diff = (original - opt).abs();
            let threshold = if original.abs() > 1.0 {
                1e-5 * original.abs()
            } else {
                1e-5
            };
            assert!(
                diff <= threshold,
                "associativity changed semantics at {point:?}: {original} vs {opt}"
            );
        }

        let canon = eg.find(root_class);
        let node_count = eg.nodes(canon).len();
        assert!(
            node_count > 1,
            "expected associativity to add alternative tree shapes, but root class has {node_count} node(s)"
        );
    }

    #[test]
    fn verify_algebraic_rules_preserve_semantics() {
        let pts = standard_test_points();
        let mut a = ExprArena::new();

        // a - b (canonicalize: sub -> add+neg)
        let e = arena_pat!(&mut a, bin OpKind::Sub, (var 0), (var 1));
        check_optimization_preserves_semantics(&a, e, &pts, 1e-5);

        // a / b (canonicalize: div -> mul+recip)
        let e = arena_pat!(&mut a, bin OpKind::Div, (var 0), (var 1));
        check_optimization_preserves_semantics(&a, e, &pts, 1e-4);

        // neg(neg(x)) (involution)
        let e = arena_pat!(&mut a, un OpKind::Neg, (un OpKind::Neg, (var 0)));
        check_optimization_preserves_semantics(&a, e, &pts, 1e-6);

        // (x + y) - y (cancellation)
        let e = arena_pat!(&mut a, bin OpKind::Sub, (bin OpKind::Add, (var 0), (var 1)), (var 1));
        check_optimization_preserves_semantics(&a, e, &pts, 1e-4);

        // x * 0 (annihilator)
        let e = arena_pat!(&mut a, bin OpKind::Mul, (var 0), (cst 0.0));
        check_optimization_preserves_semantics(&a, e, &pts, 1e-6);

        // x + 0 (identity)
        let e = arena_pat!(&mut a, bin OpKind::Add, (var 0), (cst 0.0));
        check_optimization_preserves_semantics(&a, e, &pts, 1e-6);

        // x * 1 (identity)
        let e = arena_pat!(&mut a, bin OpKind::Mul, (var 0), (cst 1.0));
        check_optimization_preserves_semantics(&a, e, &pts, 1e-6);
    }

    #[test]
    fn verify_trig_rules_preserve_semantics() {
        let pts = standard_test_points();
        let mut a = ExprArena::new();

        // sin(x + y) (angle addition)
        let e = arena_pat!(&mut a, un OpKind::Sin, (bin OpKind::Add, (var 0), (var 1)));
        check_optimization_preserves_semantics(&a, e, &pts, 1e-4);

        // cos(x + y) (angle addition)
        let e = arena_pat!(&mut a, un OpKind::Cos, (bin OpKind::Add, (var 0), (var 1)));
        check_optimization_preserves_semantics(&a, e, &pts, 1e-4);

        // sin(neg(x)) (parity: odd)
        let e = arena_pat!(&mut a, un OpKind::Sin, (un OpKind::Neg, (var 0)));
        check_optimization_preserves_semantics(&a, e, &pts, 1e-5);

        // cos(neg(x)) (parity: even)
        let e = arena_pat!(&mut a, un OpKind::Cos, (un OpKind::Neg, (var 0)));
        check_optimization_preserves_semantics(&a, e, &pts, 1e-5);
    }

    #[test]
    fn verify_associativity_left_to_right() {
        // (v0 + v1) + v2 should produce v0 + (v1 + v2) in the e-graph
        let mut a = ExprArena::new();
        let e = arena_pat!(&mut a, bin OpKind::Add, (bin OpKind::Add, (var 0), (var 1)), (var 2));
        check_assoc(&a, e);
    }

    #[test]
    fn verify_associativity_right_to_left() {
        // v0 + (v1 + v2) should produce (v0 + v1) + v2 in the e-graph
        let mut a = ExprArena::new();
        let e = arena_pat!(&mut a, bin OpKind::Add, (var 0), (bin OpKind::Add, (var 1), (var 2)));
        check_assoc(&a, e);
    }

    #[test]
    fn verify_associativity_mul() {
        // (v0 * v1) * v2 should produce v0 * (v1 * v2) and vice versa
        let mut a = ExprArena::new();
        let e = arena_pat!(&mut a, bin OpKind::Mul, (bin OpKind::Mul, (var 0), (var 1)), (var 2));
        check_optimization_preserves_semantics(&a, e, &standard_test_points(), 1e-4);
    }

    #[test]
    fn verify_associativity_min_max() {
        let pts = standard_test_points();
        let mut a = ExprArena::new();

        // min(min(v0, v1), v2) should produce min(v0, min(v1, v2))
        let e = arena_pat!(&mut a, bin OpKind::Min, (bin OpKind::Min, (var 0), (var 1)), (var 2));
        check_optimization_preserves_semantics(&a, e, &pts, 1e-6);

        // max(max(v0, v1), v2) should produce max(v0, max(v1, v2))
        let e = arena_pat!(&mut a, bin OpKind::Max, (bin OpKind::Max, (var 0), (var 1)), (var 2));
        check_optimization_preserves_semantics(&a, e, &pts, 1e-6);
    }

    #[test]
    fn verify_associativity_templates() {
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
    fn verify_all_rules_count() {
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
