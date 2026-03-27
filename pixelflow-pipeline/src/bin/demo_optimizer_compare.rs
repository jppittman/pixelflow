#![allow(dead_code, unused_variables, unused_imports, clippy::all)]
//! Optimizer Comparison Demo
//!
//! Compares three optimization strategies on real shader expressions:
//! 1. Unoptimized - raw expression, no e-graph
//! 2. Old optimizer - extract_tree_with_costs + static CostModel
//! 3. New optimizer - extract_beam + trained Judge (ExprNnue)
//!
//! For each, shows:
//! - Predicted cost (from respective cost model)
//! - The extracted expression form
//! - Node count / depth metrics

use pixelflow_search::egraph::{EGraph, ExprTree, Leaf, ops, extract_beam, extract_neural, predict_tree_cost};
use pixelflow_search::math::all_math_rules;
use pixelflow_search::nnue::ExprNnue;
use std::path::Path;

/// Path to trained Judge weights
const JUDGE_WEIGHTS: &str = "pixelflow-pipeline/data/judge.bin";

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("         OPTIMIZER COMPARISON: Old vs New (Neural)");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Load the trained Judge
    let judge = ExprNnue::load(Path::new(JUDGE_WEIGHTS))
        .unwrap_or_else(|e| panic!("Failed to load Judge from {}: {}", JUDGE_WEIGHTS, e));
    println!("Loaded trained Judge from {}\n", JUDGE_WEIGHTS);

    // Test expressions from real shaders
    compare_expression("Simple: x + y", simple_add(), &judge);
    compare_expression("FMA candidate: a*b + c", fma_candidate(), &judge);
    compare_expression("Exp-Log identity: exp(ln(x) + ln(y))", exp_log_identity(), &judge);
    compare_expression("Trig identity: sin(x)*cos(x) + cos(x)*sin(x)", trig_double_angle(), &judge);
    compare_expression("Psychedelic swirl (simplified)", psychedelic_swirl(), &judge);
    compare_expression("Radial falloff: exp(-4 * r²)", radial_falloff(), &judge);
    compare_expression("Soft clamp: x / (|x| + 1)", soft_clamp(), &judge);
    compare_expression("Sphere discriminant: d² - (c² - r²)", sphere_discriminant(), &judge);

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                       COMPARISON COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");
}

fn compare_expression(name: &str, expr: ExprTree, judge: &ExprNnue) {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  {}", name);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // === 1. UNOPTIMIZED ===
    let unopt_cost = predict_tree_cost(&expr, judge);
    println!("  [1] UNOPTIMIZED (raw expression)");
    println!("      Nodes: {}, Depth: {}", expr.node_count(), expr.depth());
    println!("      Neural cost: {:.2} ns", unopt_cost);
    println!("      Form: {}\n", format_expr(&expr));

    // === 2. OLD OPTIMIZER (CostModel + extract_tree_with_costs) ===
    let mut egraph_old = EGraph::with_rules(all_math_rules());
    let root_old = egraph_old.add_expr(&expr);

    // Saturate
    for _ in 0..20 {
        if egraph_old.apply_rules_once() == 0 { break; }
    }

    let (old_tree, old_log_cost) = extract_neural(&egraph_old, root_old, judge);
    let old_neural_cost = predict_tree_cost(&old_tree, judge);

    println!("  [2] NEURAL-DP (extract_neural — bottom-up NNUE)");
    println!("      E-graph: {} classes, {} nodes", egraph_old.num_classes(), egraph_old.node_count());
    println!("      Nodes: {}, Depth: {}", old_tree.node_count(), old_tree.depth());
    println!("      Neural cost:   {:.2} ns (Judge evaluation)", old_neural_cost);
    println!("      Form: {}\n", format_expr(&old_tree));

    // === 3. NEW OPTIMIZER (Judge + beam search) ===
    let mut egraph_new = EGraph::with_rules(all_math_rules());
    let root_new = egraph_new.add_expr(&expr);

    // Saturate
    for _ in 0..20 {
        if egraph_new.apply_rules_once() == 0 { break; }
    }

    let beam_width = 16;
    let (new_tree, new_log_cost) = extract_beam(&egraph_new, root_new, judge, beam_width);
    let new_cost = libm::expf(new_log_cost);

    println!("  [3] NEW OPTIMIZER (Judge + beam search, k={})", beam_width);
    println!("      E-graph: {} classes, {} nodes", egraph_new.num_classes(), egraph_new.node_count());
    println!("      Nodes: {}, Depth: {}", new_tree.node_count(), new_tree.depth());
    println!("      Neural cost:   {:.2} ns (log: {:.3})", new_cost, new_log_cost);
    println!("      Form: {}\n", format_expr(&new_tree));

    // === SUMMARY ===
    let old_vs_unopt = ((old_neural_cost - unopt_cost) / unopt_cost * 100.0) as i32;
    let new_vs_unopt = ((new_cost - unopt_cost) / unopt_cost * 100.0) as i32;
    let new_vs_old = ((new_cost - old_neural_cost) / old_neural_cost * 100.0) as i32;

    println!("  SUMMARY:");
    println!("      Old vs Unopt: {:+}% neural cost", old_vs_unopt);
    println!("      New vs Unopt: {:+}% neural cost", new_vs_unopt);
    println!("      New vs Old:   {:+}% neural cost", new_vs_old);

    if new_cost < old_neural_cost {
        println!("      → Neural beam search found BETTER solution");
    } else if (new_cost - old_neural_cost).abs() < 0.1 {
        println!("      → Solutions are equivalent");
    } else {
        println!("      → Old optimizer found better solution (surprising!)");
    }
    println!();
}

// ============================================================================
// TEST EXPRESSIONS
// ============================================================================

fn simple_add() -> ExprTree {
    // x + y
    ExprTree::add(ExprTree::var(0), ExprTree::var(1))
}

fn fma_candidate() -> ExprTree {
    // a * b + c
    ExprTree::add(
        ExprTree::mul(ExprTree::var(0), ExprTree::var(1)),
        ExprTree::var(2),
    )
}

fn exp_log_identity() -> ExprTree {
    // exp(ln(x) + ln(y)) -> x * y
    ExprTree::Op {
        op: &ops::Exp,
        children: vec![ExprTree::add(
            ExprTree::Op { op: &ops::Ln, children: vec![ExprTree::var(0)] },
            ExprTree::Op { op: &ops::Ln, children: vec![ExprTree::var(1)] },
        )],
    }
}

fn trig_double_angle() -> ExprTree {
    // sin(x)*cos(x) + cos(x)*sin(x) -> sin(2x)
    let sin_x = ExprTree::Op { op: &ops::Sin, children: vec![ExprTree::var(0)] };
    let cos_x = ExprTree::Op { op: &ops::Cos, children: vec![ExprTree::var(0)] };
    ExprTree::add(
        ExprTree::mul(sin_x.clone(), cos_x.clone()),
        ExprTree::mul(cos_x, sin_x),
    )
}

fn psychedelic_swirl() -> ExprTree {
    // Simplified version of the psychedelic shader's swirl computation:
    // ((vx + phase).sin() + 1.0) * ((vx + phase) - (vy + phase * 0.7)).abs() * 0.2
    //
    // Let's use: sin(x + t) * |x - y| * 0.2
    let x = ExprTree::var(0);
    let y = ExprTree::var(1);
    let t = ExprTree::var(2);

    let x_plus_t = ExprTree::add(x.clone(), t);
    let sin_xpt = ExprTree::Op { op: &ops::Sin, children: vec![x_plus_t] };

    let x_minus_y = ExprTree::Op {
        op: &ops::Sub,
        children: vec![x, y]
    };
    let abs_diff = ExprTree::Op { op: &ops::Abs, children: vec![x_minus_y] };

    let product = ExprTree::mul(sin_xpt, abs_diff);
    ExprTree::mul(product, ExprTree::Leaf(Leaf::Const(0.2)))
}

fn radial_falloff() -> ExprTree {
    // exp(-4 * (x*x + y*y)) - radial gaussian
    let x = ExprTree::var(0);
    let y = ExprTree::var(1);

    let x_sq = ExprTree::mul(x.clone(), x);
    let y_sq = ExprTree::mul(y.clone(), y);
    let r_sq = ExprTree::add(x_sq, y_sq);

    let neg_four = ExprTree::Leaf(Leaf::Const(-4.0));
    let exponent = ExprTree::mul(neg_four, r_sq);

    ExprTree::Op { op: &ops::Exp, children: vec![exponent] }
}

fn soft_clamp() -> ExprTree {
    // x / (|x| + 1) - soft clamp to [-1, 1]
    let x = ExprTree::var(0);
    let abs_x = ExprTree::Op { op: &ops::Abs, children: vec![x.clone()] };
    let denom = ExprTree::add(abs_x, ExprTree::Leaf(Leaf::Const(1.0)));
    ExprTree::Op { op: &ops::Div, children: vec![x, denom] }
}

fn sphere_discriminant() -> ExprTree {
    // d_dot_c² - (c_sq - r_sq)
    // This is the ray-sphere intersection discriminant
    let d_dot_c = ExprTree::var(0);
    let c_sq = ExprTree::var(1);
    let r_sq = ExprTree::var(2);

    let d_sq = ExprTree::mul(d_dot_c.clone(), d_dot_c);
    let inner = ExprTree::Op { op: &ops::Sub, children: vec![c_sq, r_sq] };

    ExprTree::Op { op: &ops::Sub, children: vec![d_sq, inner] }
}

// ============================================================================
// FORMATTING
// ============================================================================

fn format_expr(expr: &ExprTree) -> String {
    match expr {
        ExprTree::Leaf(Leaf::Var(i)) => {
            let names = ["x", "y", "z", "w", "a", "b", "c", "d", "t"];
            names.get(*i as usize).unwrap_or(&"?").to_string()
        }
        ExprTree::Leaf(Leaf::Const(v)) => {
            if *v == 1.0 { "1".to_string() }
            else if *v == 0.0 { "0".to_string() }
            else if *v == -1.0 { "-1".to_string() }
            else { format!("{:.2}", v) }
        }
        ExprTree::Op { op, children } => {
            use pixelflow_ir::OpKind;
            let name = match op.kind() {
                OpKind::Add => "+",
                OpKind::Sub => "-",
                OpKind::Mul => "*",
                OpKind::Div => "/",
                OpKind::Neg => "neg",
                OpKind::Recip => "recip",
                OpKind::Sin => "sin",
                OpKind::Cos => "cos",
                OpKind::Exp => "exp",
                OpKind::Ln => "ln",
                OpKind::Sqrt => "sqrt",
                OpKind::Abs => "abs",
                OpKind::MulAdd => "fma",
                _ => "op",
            };

            if children.len() == 1 {
                format!("{}({})", name, format_expr(&children[0]))
            } else if children.len() == 2 && ["+", "-", "*", "/"].contains(&name) {
                format!("({} {} {})", format_expr(&children[0]), name, format_expr(&children[1]))
            } else if children.len() == 3 && name == "fma" {
                format!("fma({}, {}, {})", format_expr(&children[0]), format_expr(&children[1]), format_expr(&children[2]))
            } else {
                let args: Vec<_> = children.iter().map(format_expr).collect();
                format!("{}({})", name, args.join(", "))
            }
        }
    }
}
