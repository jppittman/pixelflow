// src/math/pict_rewrite_tests.rs

//! Proof-of-concept #3: PICT-style pairwise (combinatorial) testing for the
//! e-graph rewrite rules.
//!
//! The existing `algebraic_rules_preserve_semantics` test hand-picks a handful
//! of expression templates and checks that optimization preserves semantics.
//! That is exactly the situation PICT is built for: the space of expression
//! *shapes* — (outer op) × (unary wrapper) × (left subtree) × (right subtree) ×
//! (constant) — is far too large to enumerate (4 × 5 × 7 × 7 × 5 = 4900 here),
//! yet most rewrite bugs are triggered by the interaction of at most two of
//! those choices. A pairwise covering array exercises every 2-way interaction
//! in a few dozen cases.
//!
//! ## Oracle: algebraic equivalence, not bit-exact FP
//!
//! The rewrite rules are explicitly allowed to reassociate, distribute, and
//! fuse (`a*b + c → fma(a,b,c)`), all of which change floating-point rounding.
//! So the oracle compares the optimized expression against the original with a
//! generous *relative* tolerance and ignores any sample point where either
//! side is non-finite (a singularity such as `recip(0)`, where "algebra
//! allows" the two forms to differ). Only a genuine change in the mathematical
//! value — a divergence between two finite results — is a failure.
//!
//! The generator is the same dependency-free pairwise routine used by the
//! other two POCs, duplicated per the workspace's minimal-dependency policy.

use crate::egraph::{EGraph, ENode, saturate_with_budget};
use crate::math::all_rules;
use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};

// ============================================================================
// Pairwise covering-array generator (see the ANSI/SGR POC for the annotated
// version; this is a verbatim copy).
// ============================================================================

fn pairwise(level_counts: &[usize]) -> Vec<Vec<usize>> {
    use std::collections::BTreeSet;

    let n = level_counts.len();
    assert!(n >= 2);
    assert!(level_counts.iter().all(|&c| c >= 1));

    let mut uncovered: BTreeSet<(usize, usize, usize, usize)> = BTreeSet::new();
    for i in 0..n {
        for j in (i + 1)..n {
            for a in 0..level_counts[i] {
                for b in 0..level_counts[j] {
                    uncovered.insert((i, a, j, b));
                }
            }
        }
    }

    let pair_key = |i: usize, a: usize, j: usize, b: usize| {
        if i < j { (i, a, j, b) } else { (j, b, i, a) }
    };

    let mut rows: Vec<Vec<usize>> = Vec::new();
    while let Some(&(si, sa, sj, sb)) = uncovered.iter().next() {
        let mut row: Vec<Option<usize>> = vec![None; n];
        row[si] = Some(sa);
        row[sj] = Some(sb);

        for f in 0..n {
            if row[f].is_some() {
                continue;
            }
            let mut best_level = 0;
            let mut best_gain = usize::MAX;
            for level in 0..level_counts[f] {
                let mut gain = 0;
                for (g, assigned) in row.iter().enumerate() {
                    let Some(av) = *assigned else { continue };
                    let (i, a, j, b) = pair_key(f, level, g, av);
                    if uncovered.contains(&(i, a, j, b)) {
                        gain += 1;
                    }
                }
                if best_gain == usize::MAX || gain > best_gain {
                    best_gain = gain;
                    best_level = level;
                }
            }
            row[f] = Some(best_level);
        }

        let row: Vec<usize> = row.into_iter().map(|v| v.unwrap()).collect();
        for i in 0..n {
            for j in (i + 1)..n {
                uncovered.remove(&(i, row[i], j, row[j]));
            }
        }
        rows.push(row);
    }
    rows
}

// ============================================================================
// E-graph plumbing (mirrors the harness in `math::tests`).
// ============================================================================

fn expr_to_egraph(arena: &ExprArena, id: ExprId, egraph: &mut EGraph) -> crate::egraph::EClassId {
    match *arena.node(id) {
        ExprNode::Var(idx) => egraph.add(ENode::Var(idx)),
        ExprNode::Const(val) => egraph.add(ENode::Const(val.to_bits())),
        ExprNode::Unary(kind, a) => {
            let ca = expr_to_egraph(arena, a, egraph);
            let op = crate::egraph::ops::op_from_kind(kind).expect("op");
            egraph.add(ENode::Op {
                op,
                children: vec![ca],
            })
        }
        ExprNode::Binary(kind, a, b) => {
            let ca = expr_to_egraph(arena, a, egraph);
            let cb = expr_to_egraph(arena, b, egraph);
            let op = crate::egraph::ops::op_from_kind(kind).expect("op");
            egraph.add(ENode::Op {
                op,
                children: vec![ca, cb],
            })
        }
        ExprNode::Ternary(kind, a, b, c) => {
            let ca = expr_to_egraph(arena, a, egraph);
            let cb = expr_to_egraph(arena, b, egraph);
            let cc = expr_to_egraph(arena, c, egraph);
            let op = crate::egraph::ops::op_from_kind(kind).expect("op");
            egraph.add(ENode::Op {
                op,
                children: vec![ca, cb, cc],
            })
        }
        ExprNode::Param(_) | ExprNode::Buffer(_) | ExprNode::Nary(..) => {
            panic!("unsupported node in rewrite POC")
        }
    }
}

fn eval_arena(arena: &ExprArena, id: ExprId, vars: &[f32; 4]) -> f32 {
    match *arena.node(id) {
        ExprNode::Var(i) => vars[i as usize],
        ExprNode::Const(c) => c,
        ExprNode::Unary(op, a) => op.eval_unary(eval_arena(arena, a, vars)).expect("unary"),
        ExprNode::Binary(op, a, b) => op
            .eval_binary(eval_arena(arena, a, vars), eval_arena(arena, b, vars))
            .expect("binary"),
        ExprNode::Ternary(op, a, b, c) => op
            .eval_ternary(
                eval_arena(arena, a, vars),
                eval_arena(arena, b, vars),
                eval_arena(arena, c, vars),
            )
            .expect("ternary"),
        _ => panic!("unsupported node in eval"),
    }
}

// ============================================================================
// The combinatorial model of expression shapes.
// ============================================================================

const OUTER_OPS: [OpKind; 4] = [OpKind::Add, OpKind::Sub, OpKind::Mul, OpKind::Div];

/// Unary wrapper applied to the whole expression. `None` means "no wrapper".
/// Includes transcendentals to engage the parity/trig/exp rewrite rules.
const UNARY_WRAPPERS: [Option<OpKind>; 8] = [
    None,
    Some(OpKind::Neg),
    Some(OpKind::Abs),
    Some(OpKind::Sqrt),
    Some(OpKind::Recip),
    Some(OpKind::Sin),
    Some(OpKind::Cos),
    Some(OpKind::Exp),
];

const CONSTS: [f32; 5] = [0.0, 1.0, 2.0, -1.0, 0.5];

const SHAPE_COUNT: usize = 10;

/// Build one operand subtree. `konst` is used by the constant-bearing shapes.
fn build_shape(a: &mut ExprArena, shape: usize, konst: f32) -> ExprId {
    match shape {
        0 => a.push_var(0),
        1 => a.push_var(1),
        2 => {
            let (l, r) = (a.push_var(0), a.push_var(1));
            a.push_binary(OpKind::Add, l, r)
        }
        3 => {
            let (l, r) = (a.push_var(0), a.push_var(2));
            a.push_binary(OpKind::Mul, l, r)
        }
        4 => {
            let v = a.push_var(0);
            a.push_unary(OpKind::Neg, v)
        }
        5 => {
            let (l, r) = (a.push_var(0), a.push_const(konst));
            a.push_binary(OpKind::Sub, l, r)
        }
        6 => {
            let (l, r) = (a.push_var(1), a.push_const(konst));
            a.push_binary(OpKind::Mul, l, r)
        }
        // Shapes that hand the parity/trig rules their exact rewrite patterns.
        7 => {
            let v = a.push_var(0);
            a.push_unary(OpKind::Sin, v)
        }
        8 => {
            // sin(neg(x)) — parity rule should rewrite to neg(sin(x)).
            let v = a.push_var(0);
            let n = a.push_unary(OpKind::Neg, v);
            a.push_unary(OpKind::Sin, n)
        }
        9 => {
            // cos(neg(x)) — parity rule should rewrite to cos(x).
            let v = a.push_var(1);
            let n = a.push_unary(OpKind::Neg, v);
            a.push_unary(OpKind::Cos, n)
        }
        _ => unreachable!(),
    }
}

fn build_expr(
    a: &mut ExprArena,
    outer: OpKind,
    wrapper: Option<OpKind>,
    left: usize,
    right: usize,
    konst: f32,
) -> ExprId {
    let l = build_shape(a, left, konst);
    let r = build_shape(a, right, konst);
    let inner = a.push_binary(outer, l, r);
    match wrapper {
        None => inner,
        Some(op) => a.push_unary(op, inner),
    }
}

/// A deterministic spread of sample points (no RNG — reproducible failures).
fn test_points() -> Vec<[f32; 4]> {
    let base = [-2.0f32, -0.75, -0.1, 0.1, 0.5, 1.0, 1.5, 3.0];
    let mut pts = Vec::new();
    for (i, &x) in base.iter().enumerate() {
        let y = base[(i + 1) % base.len()];
        let z = base[(i + 3) % base.len()];
        let w = base[(i + 5) % base.len()];
        pts.push([x, y, z, w]);
    }
    // A few explicit edge points.
    pts.push([0.0, 1.0, -1.0, 2.0]);
    pts.push([1.0, 1.0, 1.0, 1.0]);
    pts.push([-1.0, -1.0, -1.0, -1.0]);
    pts
}

/// Relative tolerance: rewrites may reassociate/fuse, so only a genuine change
/// in mathematical value (well beyond FP rounding) counts as a failure.
const REL_EPS: f32 = 1e-3;

/// Adversarial cost models. Extracting the same saturated e-graph under each
/// pulls out a *different* rewritten form (the fma-favoring model extracts the
/// fused version, the neg/recip-favoring model extracts the canonicalized
/// `a + neg(b)` / `a * recip(b)` forms, and so on), so a single saturation
/// exercises many more rewrite rules than one extraction would.
fn adversarial_cost_models() -> Vec<(&'static str, crate::egraph::CostModel)> {
    use crate::egraph::CostModel;
    let cheap = 1;
    let dear = 1000;

    let mut fma = CostModel::new();
    fma.set_cost(OpKind::MulAdd, cheap);

    let mut canonical = CostModel::new();
    canonical.set_cost(OpKind::Neg, cheap);
    canonical.set_cost(OpKind::Recip, cheap);
    canonical.set_cost(OpKind::Sub, dear);
    canonical.set_cost(OpKind::Div, dear);

    let mut cheap_add = CostModel::new();
    cheap_add.set_cost(OpKind::Add, cheap);
    cheap_add.set_cost(OpKind::Mul, dear);

    let mut cheap_mul = CostModel::new();
    cheap_mul.set_cost(OpKind::Mul, cheap);
    cheap_mul.set_cost(OpKind::Add, dear);

    vec![
        ("default", CostModel::new()),
        ("fma", fma),
        ("canonical", canonical),
        ("cheap_add", cheap_add),
        ("cheap_mul", cheap_mul),
    ]
}

#[test]
fn pict_rewrite_rules_preserve_semantics() {
    let level_counts = [
        OUTER_OPS.len(),
        UNARY_WRAPPERS.len(),
        SHAPE_COUNT,
        SHAPE_COUNT,
        CONSTS.len(),
    ];
    let rows = pairwise(&level_counts);
    let exhaustive: usize = level_counts.iter().product();
    let points = test_points();

    assert!(!all_rules().is_empty(), "expected a non-empty rule set");
    let mut failures: Vec<String> = Vec::new();
    let mut changed = 0usize; // how many cases the optimizer actually rewrote

    for row in &rows {
        let outer = OUTER_OPS[row[0]];
        let wrapper = UNARY_WRAPPERS[row[1]];
        let (left, right) = (row[2], row[3]);
        let konst = CONSTS[row[4]];

        let mut arena = ExprArena::new();
        let root = build_expr(&mut arena, outer, wrapper, left, right, konst);
        let orig_str = arena.display(root).to_string();

        // Optimize the way the compiler does: build e-graph, saturate once,
        // then extract under several cost models — each pulls out a different
        // rewritten form from the same saturated graph.
        let mut eg = EGraph::with_rules(all_rules());
        let root_class = expr_to_egraph(&arena, root, &mut eg);
        let _ = saturate_with_budget(&mut eg, 25);
        let canon_root = eg.find(root_class);

        for (model_name, costs) in adversarial_cost_models() {
            let (opt_arena, opt_root, _cost) =
                crate::egraph::extract::extract(&eg, canon_root, &costs);
            let opt_str = opt_arena.display(opt_root).to_string();
            if opt_str != orig_str {
                changed += 1;
            }

            for point in &points {
                let original = eval_arena(&arena, root, point);
                let optimized = eval_arena(&opt_arena, opt_root, point);

                // Singularities and overflow: "algebra allows" the two forms to
                // differ here, so this point does not constrain correctness.
                if !original.is_finite() || !optimized.is_finite() {
                    continue;
                }

                let diff = (original - optimized).abs();
                let denom = original.abs().max(1.0);
                if diff / denom > REL_EPS {
                    failures.push(format!(
                        "  [{model_name}] {orig_str}\n    rewrote to {opt_str}\n    at {point:?}: {original} vs {optimized} (rel diff {:.3e})",
                        diff / denom,
                    ));
                    break;
                }
            }
        }
    }

    eprintln!(
        "PICT rewrite sweep: {} pairwise cases (vs {exhaustive} exhaustive), \
         extracted under {} cost models; {} extractions differed from input",
        rows.len(),
        adversarial_cost_models().len(),
        changed,
    );
    assert!(
        failures.is_empty(),
        "PICT pairwise rewrite testing found {} algebraically-unsound rewrite(s):\n{}",
        failures.len(),
        failures.join("\n"),
    );
}
