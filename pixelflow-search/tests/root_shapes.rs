//! A rule's [`RootShapes`] are a claim about its `apply`, and the saturation
//! scan spends the claim: a node outside the set is never handed to the rule.
//! A set that is too small therefore loses rewrites without a trace, and
//! nothing downstream can see it — the extracted kernel is merely a little
//! worse. This is the check that sees it: on a graph rich enough to wake
//! every family, `apply` is asked about every node, and a `Some` on a node
//! the set excludes is the failure.
//!
//! The graph is saturated by the production scan, which is already filtered,
//! so the nodes it holds are the ones the filtered scan can produce — and the
//! check runs after every iteration, so it sees the graph at each stage the
//! scan would.

use std::collections::BTreeSet;
use std::time::Duration;

use pixelflow_ir::arena::{BufferDecl, BufferIdentity, ExprArena, ExprId};
use pixelflow_ir::{Binder, Bound, Fold, Monoid, OpKind};
use pixelflow_search::egraph::{
    Rewrite, RuleIdx, RuleSet, Saturation, Vocabulary, all_rules, derivative_rules, fold_rules,
    insert, saturate_with_full_budget,
};
use pixelflow_search::math::round2_rules::experimental_rules;

const GENEROUS: Duration = Duration::from_secs(120);
const CLASS_CAP: usize = 6_000;
const ITERATIONS: usize = 5;

/// Every rule the workspace can put in a graph: the 62, the fold rules, the
/// chain rule, and the harness-only experimental set.
fn every_rule() -> Vec<Box<dyn Rewrite>> {
    let mut rules = all_rules();
    rules.extend(fold_rules());
    rules.extend(derivative_rules());
    rules.extend(experimental_rules());
    rules
}

/// Terms shaped like the left-hand side of each rule family, so that the
/// graph holds a match for as many of them as one graph can.
fn un(a: &mut ExprArena, op: OpKind, c: ExprId) -> ExprId {
    a.push_unary(op, c)
}

fn bin(a: &mut ExprArena, op: OpKind, l: ExprId, r: ExprId) -> ExprId {
    a.push_binary(op, l, r)
}

fn terms(a: &mut ExprArena) -> Vec<ExprId> {
    let x = a.push_var(0);
    let y = a.push_var(1);
    let z = a.push_var(2);
    let two = a.push_const(2.0);
    let three = a.push_const(3.0);
    let zero = a.push_const(0.0);
    let one = a.push_const(1.0);
    let half = a.push_const(0.5);
    let minus_one = a.push_const(-1.0);
    let minus_half = a.push_const(-0.5);

    let sum = bin(a, OpKind::Add, x, y);
    let diff = bin(a, OpKind::Sub, x, y);
    let sq = bin(a, OpKind::Mul, sum, sum);
    let xy = bin(a, OpKind::Mul, x, y);
    let yx = bin(a, OpKind::Mul, y, x);
    let xz = bin(a, OpKind::Mul, x, z);
    let mut roots = Vec::new();

    // Algebra: distribution, factoring, commutation, association.
    let lhs = bin(a, OpKind::Mul, sq, diff);
    let twice = bin(a, OpKind::Add, xy, yx);
    let rhs = bin(a, OpKind::Mul, twice, sum);
    roots.push(bin(a, OpKind::Add, lhs, rhs));
    roots.push(bin(a, OpKind::Add, xy, xz));
    let nested = bin(a, OpKind::Add, sum, z);
    roots.push(bin(a, OpKind::Add, x, nested));

    // Identities, annihilators, idempotence, constants, doubling, halving.
    roots.push(bin(a, OpKind::Add, x, zero));
    roots.push(bin(a, OpKind::Mul, x, one));
    roots.push(bin(a, OpKind::Mul, x, zero));
    roots.push(bin(a, OpKind::Add, two, three));
    roots.push(bin(a, OpKind::Min, x, x));
    roots.push(bin(a, OpKind::Max, y, y));
    roots.push(bin(a, OpKind::Add, z, z));
    roots.push(bin(a, OpKind::Mul, two, z));
    roots.push(bin(a, OpKind::Mul, z, two));

    // Inverse pairs.
    let neg_y = un(a, OpKind::Neg, y);
    roots.push(bin(a, OpKind::Add, x, neg_y));
    roots.push(un(a, OpKind::Neg, neg_y));
    let recip_y = un(a, OpKind::Recip, y);
    roots.push(bin(a, OpKind::Mul, y, recip_y));
    roots.push(un(a, OpKind::Recip, recip_y));
    roots.push(bin(a, OpKind::Div, x, y));
    let plus_y = bin(a, OpKind::Add, x, y);
    roots.push(bin(a, OpKind::Sub, plus_y, y));
    let times_y = bin(a, OpKind::Mul, x, y);
    roots.push(bin(a, OpKind::Div, times_y, y));
    let sqrt_x = un(a, OpKind::Sqrt, x);
    roots.push(bin(a, OpKind::Div, one, sqrt_x));
    let sqrt_z0 = un(a, OpKind::Sqrt, z);
    roots.push(un(a, OpKind::Recip, sqrt_z0));

    // Trigonometry.
    roots.push(un(a, OpKind::Sin, sum));
    roots.push(un(a, OpKind::Cos, sum));
    let sin_x = un(a, OpKind::Sin, x);
    let cos_x = un(a, OpKind::Cos, x);
    let sin_y = un(a, OpKind::Sin, y);
    let cos_y = un(a, OpKind::Cos, y);
    let sin_sq = bin(a, OpKind::Mul, sin_x, sin_x);
    let cos_sq = bin(a, OpKind::Mul, cos_x, cos_x);
    roots.push(bin(a, OpKind::Add, sin_sq, cos_sq));
    roots.push(bin(a, OpKind::Mul, sin_x, cos_x));
    let sin_cos = bin(a, OpKind::Mul, sin_x, cos_y);
    let cos_sin = bin(a, OpKind::Mul, cos_x, sin_y);
    roots.push(bin(a, OpKind::Add, sin_cos, cos_sin));
    let recip_cos = un(a, OpKind::Recip, cos_x);
    roots.push(bin(a, OpKind::Mul, sin_x, recip_cos));
    roots.push(un(a, OpKind::Tan, y));

    // Parity.
    let neg_x = un(a, OpKind::Neg, x);
    for op in [
        OpKind::Sin,
        OpKind::Cos,
        OpKind::Tan,
        OpKind::Asin,
        OpKind::Atan,
        OpKind::Abs,
    ] {
        roots.push(un(a, op, neg_x));
    }

    // Exponentials, logarithms and powers.
    let ln_x = un(a, OpKind::Ln, x);
    roots.push(un(a, OpKind::Exp, ln_x));
    let exp_x = un(a, OpKind::Exp, x);
    roots.push(un(a, OpKind::Ln, exp_x));
    let log2_y = un(a, OpKind::Log2, y);
    roots.push(un(a, OpKind::Exp2, log2_y));
    let exp2_y = un(a, OpKind::Exp2, y);
    roots.push(un(a, OpKind::Log2, exp2_y));
    roots.push(un(a, OpKind::Exp, sum));
    roots.push(un(a, OpKind::Ln, xy));
    roots.push(un(a, OpKind::Log10, z));
    for exponent in [zero, one, two, half, minus_one, minus_half, three] {
        roots.push(bin(a, OpKind::Pow, x, exponent));
    }
    roots.push(bin(a, OpKind::Pow, sum, two));
    let pow_xy = bin(a, OpKind::Pow, x, y);
    roots.push(un(a, OpKind::Ln, pow_xy));
    roots.push(un(a, OpKind::Log2, pow_xy));
    let pow_xz = bin(a, OpKind::Pow, x, z);
    roots.push(bin(a, OpKind::Mul, pow_xy, pow_xz));
    let xx = bin(a, OpKind::Mul, x, x);
    let yy = bin(a, OpKind::Mul, y, y);
    roots.push(bin(a, OpKind::Sub, xx, yy));

    // Fusion and the harness-only shapes.
    roots.push(bin(a, OpKind::Add, xy, z));
    roots.push(a.push_ternary(OpKind::MulAdd, x, y, z));
    roots.push(a.push_ternary(OpKind::MulAdd, x, one, z));
    roots.push(a.push_ternary(OpKind::MulAdd, x, y, zero));
    let lt = bin(a, OpKind::Lt, x, y);
    roots.push(a.push_ternary(OpKind::If, lt, x, y));
    roots.push(a.push_ternary(OpKind::If, lt, y, x));
    roots.push(a.push_ternary(OpKind::If, lt, z, z));
    let neg_y2 = un(a, OpKind::Neg, y);
    let neg_z = un(a, OpKind::Neg, z);
    roots.push(a.push_ternary(OpKind::If, lt, neg_y2, neg_z));
    let abs_y = un(a, OpKind::Abs, y);
    let abs_z = un(a, OpKind::Abs, z);
    roots.push(a.push_ternary(OpKind::If, lt, abs_y, abs_z));
    let sqrt_y = un(a, OpKind::Sqrt, y);
    let sqrt_z = un(a, OpKind::Sqrt, z);
    roots.push(a.push_ternary(OpKind::If, lt, sqrt_y, sqrt_z));
    roots.push(bin(a, OpKind::Mul, sqrt_y, sqrt_z));
    let rsqrt_y = un(a, OpKind::Rsqrt, y);
    roots.push(bin(a, OpKind::Mul, rsqrt_y, rsqrt_y));
    roots.push(bin(a, OpKind::Mul, y, rsqrt_y));
    let recip_z = un(a, OpKind::Recip, z);
    roots.push(bin(a, OpKind::Mul, recip_y, recip_z));
    roots.push(un(a, OpKind::Recip, xy));
    roots.push(un(a, OpKind::Neg, sum));
    roots.push(un(a, OpKind::Neg, xy));
    roots.push(bin(a, OpKind::Div, x, three));
    let max_y = bin(a, OpKind::Max, x, y);
    roots.push(bin(a, OpKind::Min, x, max_y));
    let min_y = bin(a, OpKind::Min, x, y);
    roots.push(bin(a, OpKind::Max, x, min_y));
    roots.push(bin(a, OpKind::Add, min_y, z));
    roots.push(bin(a, OpKind::Mul, two, min_y));
    let max_yz = bin(a, OpKind::Max, y, z);
    roots.push(bin(a, OpKind::Min, x, max_yz));
    let neg_x2 = un(a, OpKind::Neg, x);
    roots.push(bin(a, OpKind::Max, x, neg_x2));
    roots.push(un(a, OpKind::Abs, z));
    roots.push(un(a, OpKind::Exp, z));
    roots.push(un(a, OpKind::Ln, z));

    // A derivative.
    let var = a.push_const(0.0);
    roots.push(bin(a, OpKind::Dwrt, sq, var));

    // Folds: a table-reading sum, an empty fold, a factorable one, and a
    // triangular one, whose inner bound is the outer index until a peel turns
    // it into a literal (`KnownBound`'s match).
    let buffer = a.declare_buffer(BufferDecl {
        id: BufferIdentity::mint(),
        width: 8,
        height: 1,
    });
    let binder = Binder::from_slot(0).expect("a live binder");
    let i = a.push_var(binder.var());
    let row = a.push_const(0.0);
    let read = a.push_gather(buffer, i, row);
    let weighted = bin(a, OpKind::Mul, read, x);
    roots.push(a.push_reduce(
        Fold::new(Monoid::SUM, binder),
        Bound::Lit(0),
        Bound::Lit(4),
        weighted,
    ));
    let constant_body = a.push_const(1.0);
    roots.push(a.push_reduce(
        Fold::new(Monoid::SUM, binder),
        Bound::Lit(0),
        Bound::Lit(0),
        constant_body,
    ));
    let outer = Binder::from_slot(1).expect("a live binder");
    let outer_index = a.push_var(outer.var());
    let triangle_body = bin(a, OpKind::Mul, x, i);
    let triangle = a.push_reduce(
        Fold::new(Monoid::SUM, binder),
        Bound::Lit(0),
        Bound::Value(outer_index),
        triangle_body,
    );
    roots.push(a.push_reduce(
        Fold::new(Monoid::SUM, outer),
        Bound::Lit(0),
        Bound::Lit(3),
        triangle,
    ));
    roots
}

#[test]
fn no_rule_matches_a_node_outside_its_root_shapes() {
    let mut arena = ExprArena::new();
    let roots = terms(&mut arena);

    let mut eg = Saturation::over(RuleSet::new(every_rule()));
    for root in roots {
        insert(&arena, root, &mut eg, Vocabulary::Runtime).expect("every term inserts");
    }

    let mut fired: BTreeSet<RuleIdx> = BTreeSet::new();
    for iteration in 0..ITERATIONS {
        saturate_with_full_budget(&mut eg, 1, CLASS_CAP, GENEROUS);
        let mut nodes_checked = 0usize;
        for class in eg.class_ids().collect::<Vec<_>>() {
            for node in eg.nodes(class) {
                for idx in eg.rules().indices() {
                    let rule = eg.rules().rule(idx);
                    nodes_checked += 1;
                    if rule.apply(&eg, class, node).is_none() {
                        continue;
                    }
                    fired.insert(idx);
                    assert!(
                        rule.root_shapes().admits(node),
                        "iteration {iteration}: rule {idx} ({}, {:?}) matched {node:?}, \
                         which its root_shapes() {:?} exclude — the scan would never offer it",
                        rule.name(),
                        rule.specialization(),
                        rule.root_shapes(),
                    );
                }
            }
        }
        assert!(nodes_checked > 0);
    }

    // The check above is vacuous for a rule that never fires, so name them.
    let silent: Vec<String> = eg
        .rules()
        .indices()
        .filter(|idx| !fired.contains(idx))
        .map(|idx| {
            let rule = eg.rules().rule(idx);
            format!("{idx}:{}({:?})", rule.name(), rule.specialization())
        })
        .collect();
    assert!(
        silent.is_empty(),
        "these rules never matched anything in the graph, so their root shapes are \
         unchecked here: {silent:?}"
    );
}
