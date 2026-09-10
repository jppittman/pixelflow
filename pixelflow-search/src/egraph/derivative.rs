//! Symbolic differentiation as e-graph rewrites.
//!
//! Autodiff is a single operator, `Dwrt(expr, var)` ([`OpKind::Dwrt`]). The
//! author writes `D(expr, var)` and never learns the mechanism: the chain rule
//! lives here, as one rewrite rule that expands a `Dwrt` node one step toward
//! the leaves. Equality saturation runs it to fixpoint, and the residual
//! arithmetic is then optimised by the ordinary algebra/fusion rules in the
//! same e-graph — there is no ordering problem, symbolic differentiation and
//! FMA fusion saturate together.
//!
//! A `Dwrt` that survives saturation (an operator with no differentiation
//! rule, or a budget miss) is left in the graph with a prohibitive cost so the
//! extractor never prefers it. The fallback tier is the runtime `lower_dwrt`
//! pass in pixelflow-ir — the same algebra applied directly to the arena —
//! which errors loudly on genuinely non-differentiable ops.
//!
//! The actual derivative construction lives in `EGraph::build_derivative`,
//! reached through [`RewriteAction::Differentiate`]; this rule only recognises
//! a `Dwrt` node, reads the variable index from its constant operand, and picks
//! a representative of the differentiand to hand off.

use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::rewrite::{Rewrite, RewriteAction};
use alloc::boxed::Box;
use alloc::vec::Vec;
use pixelflow_ir::kind::OpKind;

/// The chain rule: expand `Dwrt(expr, var)` one differentiation step.
pub struct ChainRule;

impl ChainRule {
    #[must_use]
    pub fn new() -> Box<Self> {
        Box::new(Self)
    }
}

impl Rewrite for ChainRule {
    fn name(&self) -> &str {
        "differentiate"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match `Dwrt(expr, var)`.
        let ENode::Op { op, children } = node else {
            return None;
        };
        if op.kind() != OpKind::Dwrt || children.len() != 2 {
            return None;
        }
        let expr_class = children[0];
        let var_class = children[1];

        // The differentiation variable is encoded as a constant operand.
        let var = egraph.nodes(var_class).iter().find_map(ENode::as_f32)? as u8;

        // Differentiate a representative of the differentiand. All nodes in the
        // class are equal, so any non-`Dwrt` representative gives the same
        // derivative; skipping `Dwrt` nodes avoids differentiating a pending
        // derivative back into itself.
        let inner = egraph
            .nodes(expr_class)
            .iter()
            .find(|n| !is_dwrt(n))
            .cloned()?;

        Some(RewriteAction::Differentiate { inner, var })
    }
}

fn is_dwrt(node: &ENode) -> bool {
    matches!(node, ENode::Op { op, .. } if op.kind() == OpKind::Dwrt)
}

/// The differentiation rule set: just the chain rule. It only matches `Dwrt`
/// nodes, so it is inert for kernels that contain no derivatives.
#[must_use]
pub fn derivative_rules() -> Vec<Box<dyn Rewrite>> {
    let mut rules: Vec<Box<dyn Rewrite>> = Vec::new();
    rules.push(ChainRule::new());
    rules
}

#[cfg(test)]
mod tests {
    use super::super::CostModel;
    use super::super::extract::extract;
    use super::super::saturate::SaturationConfig;
    use super::*;
    use crate::arena_pat;
    use pixelflow_ir::Rooted;
    use pixelflow_ir::expr::{Environment, ExprBuilder, ExprData, ExprRef, Term};

    /// Evaluate an expression via the reference interpreter.
    ///
    /// Delegates to `pixelflow_ir::eval_scalar` rather than walking the graph
    /// here: that is the language's semantics (it lowers transcendentals to the
    /// expansion the compiler emits), and a private walker would be a second
    /// definition free to drift from it.
    fn eval(out: &(Rooted<ExprData>, Environment), vars: &[f32; 2]) -> f32 {
        pixelflow_ir::eval_scalar(
            Term::new(out.0.entry(), &out.1),
            vars,
            &pixelflow_ir::binding::BindingTable::empty(),
        )
    }

    /// Saturate `D(differentiand, var)` with the derivative rules, extract the
    /// cheapest representative, and assert it is `Dwrt`-free.
    ///
    /// Takes the builder by `&mut` and inserts straight out of it — an
    /// `ExprBuilder` is itself an `Ir`, so there is no finished graph needed in
    /// between, which is also how the caller keeps building afterwards.
    fn differentiate(
        a: &mut ExprBuilder,
        differentiand: ExprRef,
        var: u8,
    ) -> (Rooted<ExprData>, Environment) {
        let v = a.push_const(f32::from(var));
        let root = a.push_binary(OpKind::Dwrt, differentiand, v);

        // Isolate the differentiation rules. They expand `Dwrt` to fixpoint at
        // the leaves; correctness of the residual arithmetic is checked by
        // `eval`, so no algebraic cleanup is needed. (Running the full rule set
        // here only invites e-graph explosion on `x²+y²`, which is a saturation
        // budgeting concern orthogonal to autodiff.)
        let mut eg = EGraph::with_rules(derivative_rules());
        let root_class =
            crate::egraph::insert(&*a, root, &mut eg, crate::egraph::Vocabulary::Templates)
                .expect("insert into e-graph");
        SaturationConfig::compatibility(60).run(&mut eg);

        let (out, env, _cost) = extract(&eg, root_class, &CostModel::default());
        assert!(
            !contains_dwrt(&out),
            "extracted derivative still contains Dwrt: {}",
            pixelflow_ir::display(out.entry()),
        );
        (out, env)
    }

    fn contains_dwrt(rooted: &Rooted<ExprData>) -> bool {
        rooted.iter().any(|n| n.op() == Some(OpKind::Dwrt))
    }

    fn assert_close(got: f32, want: f32, pt: &[f32; 2]) {
        let tol = 1e-3 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tol,
            "at {pt:?}: got {got}, want {want} (tol {tol})"
        );
    }

    #[test]
    fn d_var_is_one_or_zero() {
        let mut a = ExprBuilder::new();
        let x = a.push_var(0);
        let out = differentiate(&mut a, x, 0);
        let pts = [[3.0, 5.0], [-2.0, 7.0]];
        for p in &pts {
            assert_close(eval(&out, p), 1.0, p); // dx/dx = 1
        }

        let mut a = ExprBuilder::new();
        let y = a.push_var(1);
        let out = differentiate(&mut a, y, 0);
        for p in &pts {
            assert_close(eval(&out, p), 0.0, p); // dy/dx = 0
        }
    }

    #[test]
    fn d_product_obeys_product_rule() {
        // d/dx (x * x) = 2x.
        let mut a = ExprBuilder::new();
        let e = arena_pat!(&mut a, bin OpKind::Mul, (var 0), (var 0));
        let out = differentiate(&mut a, e, 0);
        for p in &[[1.5, 0.0], [-3.0, 0.0], [4.2, 0.0]] {
            assert_close(eval(&out, p), 2.0 * p[0], p);
        }
    }

    #[test]
    fn d_sqrt_sum_of_squares() {
        // The north-star case: d/dx sqrt(x^2 + y^2) = x / sqrt(x^2 + y^2).
        let mut a = ExprBuilder::new();
        let e = arena_pat!(
            &mut a,
            un OpKind::Sqrt,
            (bin OpKind::Add,
                (bin OpKind::Mul, (var 0), (var 0)),
                (bin OpKind::Mul, (var 1), (var 1)))
        );
        let out = differentiate(&mut a, e, 0);

        let pts: [[f32; 2]; 4] = [[3.0, 4.0], [1.0, 1.0], [-2.0, 5.0], [0.5, 0.25]];
        for p in &pts {
            let want = p[0] / (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert_close(eval(&out, p), want, p);
        }
    }

    #[test]
    fn d_sin_is_cos() {
        // d/dx sin(x) = cos(x), where `cos` means the language's cos — built
        // as an expression and evaluated the same way, not `f32::cos`.
        // The rule is exact; the polynomial `cos` is expanded from
        // `sin(x + π/2)` and carries its own approximation error, which
        // comparing against libm would charge to the chain rule.
        let mut a = ExprBuilder::new();
        let e = arena_pat!(&mut a, un OpKind::Sin, (var 0));
        let out = differentiate(&mut a, e, 0);

        let mut b = ExprBuilder::new();
        let cos = arena_pat!(&mut b, un OpKind::Cos, (var 0));
        let expected = b.finish(&[cos]);

        for p in &[[0.0, 0.0], [0.7, 0.0], [-1.2, 0.0]] {
            assert_close(eval(&out, p), eval(&expected, p), p);
        }
    }
}

#[cfg(test)]
mod piecewise_tests {
    use super::super::CostModel;
    use super::super::extract::extract;
    use super::super::saturate::SaturationConfig;
    use super::*;
    use crate::arena_pat;
    use pixelflow_ir::Rooted;
    use pixelflow_ir::expr::{Environment, ExprBuilder, ExprData, ExprRef, Term};

    // Reuse the sibling module's helpers via a local copy of the entry point:
    // saturate D(e, var) with the derivative rules only, extract, assert
    // Dwrt-free. (The helpers in `tests` are `#[cfg(test)]`-private to it.)
    fn differentiate(
        a: &mut ExprBuilder,
        differentiand: ExprRef,
        var: u8,
    ) -> (Rooted<ExprData>, Environment) {
        let v = a.push_const(f32::from(var));
        let root = a.push_binary(OpKind::Dwrt, differentiand, v);
        let mut eg = EGraph::with_rules(derivative_rules());
        let root_class =
            crate::egraph::insert(&*a, root, &mut eg, crate::egraph::Vocabulary::Templates)
                .expect("insert into e-graph");
        SaturationConfig::compatibility(60).run(&mut eg);
        let (out, env, _cost) = extract(&eg, root_class, &CostModel::default());
        assert!(
            !out.iter().any(|n| n.op() == Some(OpKind::Dwrt)),
            "extracted derivative still contains Dwrt: {}",
            pixelflow_ir::display(out.entry()),
        );
        (out, env)
    }

    fn eval(out: &(Rooted<ExprData>, Environment), vars: &[f32; 2]) -> f32 {
        pixelflow_ir::eval_scalar(
            Term::new(out.0.entry(), &out.1),
            vars,
            &pixelflow_ir::binding::BindingTable::empty(),
        )
    }

    fn assert_close(got: f32, want: f32, pt: &[f32; 2]) {
        let tol = 1e-3 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tol,
            "at {pt:?}: got {got}, want {want} (tol {tol})"
        );
    }

    #[test]
    fn d_min_picks_branch_derivative() {
        // d/dx min(x·2, y·3): 2 where x·2 < y·3, else 0.
        let mut a = ExprBuilder::new();
        let e = arena_pat!(&mut a, bin OpKind::Min,
            (bin OpKind::Mul, (var 0), (cst 2.0)),
            (bin OpKind::Mul, (var 1), (cst 3.0)));
        let out = differentiate(&mut a, e, 0);
        assert_close(eval(&out, &[1.0, 5.0]), 2.0, &[1.0, 5.0]);
        assert_close(eval(&out, &[9.0, 1.0]), 0.0, &[9.0, 1.0]);
    }

    #[test]
    fn d_select_blends_branch_derivatives() {
        // d/dx select(y > 0, x·x, x·5): 2x above the axis, 5 below.
        let mut a = ExprBuilder::new();
        let e = arena_pat!(&mut a, tern OpKind::Select,
            (bin OpKind::Gt, (var 1), (cst 0.0)),
            (bin OpKind::Mul, (var 0), (var 0)),
            (bin OpKind::Mul, (var 0), (cst 5.0)));
        let out = differentiate(&mut a, e, 0);
        assert_close(eval(&out, &[3.0, 1.0]), 6.0, &[3.0, 1.0]);
        assert_close(eval(&out, &[3.0, -1.0]), 5.0, &[3.0, -1.0]);
    }

    #[test]
    fn d_clamp_saturates() {
        // d/dx clamp(x·x, 0, 10): 2x inside, 0 saturated. `clamp` is library,
        // so this is its min/max composition and the derivative falls out of
        // the min/max rules — there is no clamp-specific rule to exercise.
        let mut a = ExprBuilder::new();
        let e = arena_pat!(&mut a, bin OpKind::Min,
            (bin OpKind::Max,
                (bin OpKind::Mul, (var 0), (var 0)),
                (cst 0.0)),
            (cst 10.0));
        let out = differentiate(&mut a, e, 0);
        assert_close(eval(&out, &[2.0, 0.0]), 4.0, &[2.0, 0.0]);
        assert_close(eval(&out, &[5.0, 0.0]), 0.0, &[5.0, 0.0]);
    }

    #[test]
    fn d_comparison_is_zero() {
        let mut a = ExprBuilder::new();
        let e = arena_pat!(&mut a, bin OpKind::Lt, (var 0), (var 1));
        let out = differentiate(&mut a, e, 0);
        assert_close(eval(&out, &[3.0, 5.0]), 0.0, &[3.0, 5.0]);
    }
}
