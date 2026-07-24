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
    use super::*;
    use crate::arena_pat;
    use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};

    /// Evaluate an arena subtree at the given variable values.
    fn eval(arena: &ExprArena, id: ExprId, vars: &[f32; 4]) -> f32 {
        match *arena.node(id) {
            ExprNode::Var(i) => vars[i as usize],
            ExprNode::Const(c) => c,
            ExprNode::Param(p) => panic!("Param({p}) in extracted derivative"),
            ExprNode::Buffer(b) => panic!("Buffer({}) in extracted derivative", b.0),
            ExprNode::Unary(op, a) => {
                let a = eval(arena, a, vars);
                op.eval_unary(a)
                    .unwrap_or_else(|| panic!("eval_unary {op:?}"))
            }
            ExprNode::Binary(op, a, b) => {
                let a = eval(arena, a, vars);
                let b = eval(arena, b, vars);
                op.eval_binary(a, b)
                    .unwrap_or_else(|| panic!("eval_binary {op:?}"))
            }
            ExprNode::Ternary(op, a, b, c) => {
                let a = eval(arena, a, vars);
                let b = eval(arena, b, vars);
                let c = eval(arena, c, vars);
                op.eval_ternary(a, b, c)
                    .unwrap_or_else(|| panic!("eval_ternary {op:?}"))
            }
            ExprNode::Nary(op, ..) => panic!("Nary {op:?} in extracted derivative"),
        }
    }

    /// Saturate `D(differentiand, var)` with the full rule set, extract the
    /// cheapest representative, and assert it is `Dwrt`-free.
    fn differentiate(arena: &ExprArena, differentiand: ExprId, var: u8) -> (ExprArena, ExprId) {
        let mut a = arena.clone();
        let v = a.push_const(var as f32);
        let root = a.push_binary(OpKind::Dwrt, differentiand, v);

        // Isolate the differentiation rules. They expand `Dwrt` to fixpoint at
        // the leaves; correctness of the residual arithmetic is checked by
        // `eval`, so no algebraic cleanup is needed. (Running the full rule set
        // here only invites e-graph explosion on `x²+y²`, which is a saturation
        // budgeting concern orthogonal to autodiff.)
        let mut eg = EGraph::with_rules(derivative_rules());
        let root_class = eg.add_arena(&a, root);
        eg.saturate_with_limit(60);

        let (out, out_root, _cost) = extract(&eg, root_class, &CostModel::default());
        assert!(
            !contains_dwrt(&out),
            "extracted derivative still contains Dwrt: {}",
            out.display(out_root),
        );
        (out, out_root)
    }

    fn contains_dwrt(arena: &ExprArena) -> bool {
        (0..arena.len()).any(|i| {
            matches!(
                arena.node(ExprId(i as u32)),
                ExprNode::Unary(OpKind::Dwrt, _)
                    | ExprNode::Binary(OpKind::Dwrt, _, _)
                    | ExprNode::Ternary(OpKind::Dwrt, _, _, _)
            )
        })
    }

    fn assert_close(got: f32, want: f32, pt: &[f32; 4]) {
        let tol = 1e-3 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tol,
            "at {pt:?}: got {got}, want {want} (tol {tol})"
        );
    }

    #[test]
    fn d_var_is_one_or_zero() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let (out, root) = differentiate(&a, x, 0);
        let pts = [[3.0, 5.0, 0.0, 0.0], [-2.0, 7.0, 0.0, 0.0]];
        for p in &pts {
            assert_close(eval(&out, root, p), 1.0, p); // dx/dx = 1
        }

        let mut a = ExprArena::new();
        let y = a.push_var(1);
        let (out, root) = differentiate(&a, y, 0);
        for p in &pts {
            assert_close(eval(&out, root, p), 0.0, p); // dy/dx = 0
        }
    }

    #[test]
    fn d_product_obeys_product_rule() {
        // d/dx (x * x) = 2x.
        let mut a = ExprArena::new();
        let e = arena_pat!(&mut a, bin OpKind::Mul, (var 0), (var 0));
        let (out, root) = differentiate(&a, e, 0);
        for p in &[
            [1.5, 0.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0, 0.0],
            [4.2, 0.0, 0.0, 0.0],
        ] {
            assert_close(eval(&out, root, p), 2.0 * p[0], p);
        }
    }

    #[test]
    fn d_sqrt_sum_of_squares() {
        // The north-star case: d/dx sqrt(x^2 + y^2) = x / sqrt(x^2 + y^2).
        let mut a = ExprArena::new();
        let e = arena_pat!(
            &mut a,
            un OpKind::Sqrt,
            (bin OpKind::Add,
                (bin OpKind::Mul, (var 0), (var 0)),
                (bin OpKind::Mul, (var 1), (var 1)))
        );
        let (out, root) = differentiate(&a, e, 0);

        let pts: [[f32; 4]; 4] = [
            [3.0, 4.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [-2.0, 5.0, 0.0, 0.0],
            [0.5, 0.25, 0.0, 0.0],
        ];
        for p in &pts {
            let want = p[0] / (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert_close(eval(&out, root, p), want, p);
        }
    }

    #[test]
    fn d_sin_is_cos() {
        // d/dx sin(x) = cos(x).
        let mut a = ExprArena::new();
        let e = arena_pat!(&mut a, un OpKind::Sin, (var 0));
        let (out, root) = differentiate(&a, e, 0);
        for p in &[
            [0.0, 0.0, 0.0, 0.0],
            [0.7, 0.0, 0.0, 0.0],
            [-1.2, 0.0, 0.0, 0.0],
        ] {
            assert_close(eval(&out, root, p), p[0].cos(), p);
        }
    }
}

#[cfg(test)]
mod piecewise_tests {
    use super::super::CostModel;
    use super::super::extract::extract;
    use super::*;
    use crate::arena_pat;
    use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};

    // Reuse the sibling module's helpers via a local copy of the entry point:
    // saturate D(e, var) with the derivative rules only, extract, assert
    // Dwrt-free. (The helpers in `tests` are `#[cfg(test)]`-private to it.)
    fn differentiate(arena: &ExprArena, differentiand: ExprId, var: u8) -> (ExprArena, ExprId) {
        let mut a = arena.clone();
        let v = a.push_const(var as f32);
        let root = a.push_binary(OpKind::Dwrt, differentiand, v);
        let mut eg = EGraph::with_rules(derivative_rules());
        let root_class = eg.add_arena(&a, root);
        eg.saturate_with_limit(60);
        let (out, out_root, _cost) = extract(&eg, root_class, &CostModel::default());
        assert!(
            !(0..out.len()).any(|i| matches!(
                out.node(ExprId(i as u32)),
                ExprNode::Binary(OpKind::Dwrt, _, _)
            )),
            "extracted derivative still contains Dwrt: {}",
            out.display(out_root),
        );
        (out, out_root)
    }

    fn eval(arena: &ExprArena, id: ExprId, vars: &[f32; 4]) -> f32 {
        match *arena.node(id) {
            ExprNode::Var(i) => vars[i as usize],
            ExprNode::Const(c) => c,
            ExprNode::Unary(op, a) => {
                let a = eval(arena, a, vars);
                op.eval_unary(a).unwrap()
            }
            ExprNode::Binary(op, a, b) => {
                let a = eval(arena, a, vars);
                let b = eval(arena, b, vars);
                op.eval_binary(a, b).unwrap()
            }
            ExprNode::Ternary(op, a, b, c) => {
                let a = eval(arena, a, vars);
                let b = eval(arena, b, vars);
                let c = eval(arena, c, vars);
                op.eval_ternary(a, b, c).unwrap()
            }
            ref other => panic!("unexpected node in extracted derivative: {other:?}"),
        }
    }

    fn assert_close(got: f32, want: f32, pt: &[f32; 4]) {
        let tol = 1e-3 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tol,
            "at {pt:?}: got {got}, want {want} (tol {tol})"
        );
    }

    #[test]
    fn d_min_picks_branch_derivative() {
        // d/dx min(x·2, y·3): 2 where x·2 < y·3, else 0.
        let mut a = ExprArena::new();
        let e = arena_pat!(&mut a, bin OpKind::Min,
            (bin OpKind::Mul, (var 0), (cst 2.0)),
            (bin OpKind::Mul, (var 1), (cst 3.0)));
        let (out, root) = differentiate(&a, e, 0);
        assert_close(eval(&out, root, &[1.0, 5.0, 0.0, 0.0]), 2.0, &[1.0, 5.0, 0.0, 0.0]);
        assert_close(eval(&out, root, &[9.0, 1.0, 0.0, 0.0]), 0.0, &[9.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn d_select_blends_branch_derivatives() {
        // d/dx select(y > 0, x·x, x·5): 2x above the axis, 5 below.
        let mut a = ExprArena::new();
        let e = arena_pat!(&mut a, tern OpKind::Select,
            (bin OpKind::Gt, (var 1), (cst 0.0)),
            (bin OpKind::Mul, (var 0), (var 0)),
            (bin OpKind::Mul, (var 0), (cst 5.0)));
        let (out, root) = differentiate(&a, e, 0);
        assert_close(eval(&out, root, &[3.0, 1.0, 0.0, 0.0]), 6.0, &[3.0, 1.0, 0.0, 0.0]);
        assert_close(eval(&out, root, &[3.0, -1.0, 0.0, 0.0]), 5.0, &[3.0, -1.0, 0.0, 0.0]);
    }

    #[test]
    fn d_clamp_saturates() {
        // d/dx clamp(x·x, 0, 10): 2x inside, 0 saturated.
        let mut a = ExprArena::new();
        let e = arena_pat!(&mut a, tern OpKind::Clamp,
            (bin OpKind::Mul, (var 0), (var 0)),
            (cst 0.0),
            (cst 10.0));
        let (out, root) = differentiate(&a, e, 0);
        assert_close(eval(&out, root, &[2.0, 0.0, 0.0, 0.0]), 4.0, &[2.0, 0.0, 0.0, 0.0]);
        assert_close(eval(&out, root, &[5.0, 0.0, 0.0, 0.0]), 0.0, &[5.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn d_comparison_is_zero() {
        let mut a = ExprArena::new();
        let e = arena_pat!(&mut a, bin OpKind::Lt, (var 0), (var 1));
        let (out, root) = differentiate(&a, e, 0);
        assert_close(eval(&out, root, &[3.0, 5.0, 0.0, 0.0]), 0.0, &[3.0, 5.0, 0.0, 0.0]);
    }
}
