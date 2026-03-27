#![allow(dead_code, unused_variables, unused_imports, clippy::all)]
//! Demo: E-graph discovers non-obvious equivalent forms.
//!
//! Shows how equality saturation explores the space of equivalent expressions
//! and finds representations that humans wouldn't think of.

use pixelflow_search::egraph::{CostModel, EGraph, ExprTree, Leaf, all_rules, ops};
use pixelflow_search::math::all_math_rules;

fn main() {
    println!("=== E-Graph Discovery Demo ===");
    println!("Watch the e-graph find non-obvious equivalent forms\n");

    let costs = CostModel::default();

    // Demo 1: Chain of operations that simplifies unexpectedly
    demo_chain_simplification(&costs);

    // Demo 2: Double angle via rule composition
    demo_hidden_trig_identity(&costs);

    // Demo 3: Pythagorean + double angle
    demo_pythagorean_simplification(&costs);

    // Demo 4: Nested inverses
    demo_nested_inverses(&costs);

    // Demo 5: Distributivity discovers factoring
    demo_factor_discovery(&costs);

    // Demo 6: Complex expression with multiple optimization paths
    demo_multi_path(&costs);

    println!("\n=== Discovery Demo Complete ===");
}

fn demo_chain_simplification(costs: &CostModel) {
    println!("━━━ Demo 1: Chain Simplification ━━━");
    println!("Input: exp(ln(x) + ln(y))");
    println!("Human sees: two logs, an add, an exp");
    println!("E-graph discovers: exp(ln(x*y)) → x*y via homomorphism + inverse!\n");

    // Build: exp(ln(x) + ln(y))
    let ln_x = ExprTree::Op {
        op: &ops::Ln,
        children: vec![ExprTree::var(0)],
    };
    let ln_y = ExprTree::Op {
        op: &ops::Ln,
        children: vec![ExprTree::var(1)],
    };
    let sum = ExprTree::Op {
        op: &ops::Add,
        children: vec![ln_x, ln_y],
    };
    let expr = ExprTree::Op {
        op: &ops::Exp,
        children: vec![sum],
    };

    run_discovery(&expr, costs, 15);
}

fn demo_hidden_trig_identity(costs: &CostModel) {
    println!("\n━━━ Demo 2: Double Angle via Rule Composition ━━━");
    println!("Input: sin(x)*cos(x) + cos(x)*sin(x)");
    println!("Human sees: two products of trig functions");
    println!("E-graph discovers:");
    println!("  1. Commutativity: sin(x)*cos(x) = cos(x)*sin(x)");
    println!("  2. Reverse angle addition: sin(a)cos(b) + cos(a)sin(b) = sin(a+b)");
    println!("  3. With a=b=x: this = sin(x+x) = sin(2x)");
    println!("  4. So sin(x)*cos(x) = sin(2x)/2!\n");

    // Build: sin(x)*cos(x) + cos(x)*sin(x)
    // This should simplify to sin(2x) via reverse angle addition when a=b
    let sin_x = ExprTree::Op { op: &ops::Sin, children: vec![ExprTree::var(0)] };
    let cos_x = ExprTree::Op { op: &ops::Cos, children: vec![ExprTree::var(0)] };

    // sin(x)*cos(x)
    let term1 = ExprTree::Op { op: &ops::Mul, children: vec![sin_x.clone(), cos_x.clone()] };
    // cos(x)*sin(x)
    let term2 = ExprTree::Op { op: &ops::Mul, children: vec![cos_x, sin_x] };
    // sin(x)*cos(x) + cos(x)*sin(x) = 2*sin(x)*cos(x)
    let expr = ExprTree::Op { op: &ops::Add, children: vec![term1, term2] };

    run_discovery(&expr, costs, 20);
}

fn demo_pythagorean_simplification(costs: &CostModel) {
    println!("\n━━━ Demo 3: Pythagorean + Double Angle ━━━");
    println!("Input: sin(x)*sin(x) + cos(x)*cos(x) + sin(y)*cos(y)");
    println!("Human sees: mess of trig");
    println!("E-graph discovers: 1 + sin(2y)/2 via Pythagorean + double angle!\n");

    // Build: sin²(x) + cos²(x) + sin(y)*cos(y)
    let sin_x = ExprTree::Op { op: &ops::Sin, children: vec![ExprTree::var(0)] };
    let cos_x = ExprTree::Op { op: &ops::Cos, children: vec![ExprTree::var(0)] };
    let sin_y = ExprTree::Op { op: &ops::Sin, children: vec![ExprTree::var(1)] };
    let cos_y = ExprTree::Op { op: &ops::Cos, children: vec![ExprTree::var(1)] };

    let sin_sq = ExprTree::Op { op: &ops::Mul, children: vec![sin_x.clone(), sin_x] };
    let cos_sq = ExprTree::Op { op: &ops::Mul, children: vec![cos_x.clone(), cos_x] };
    let sin_cos_y = ExprTree::Op { op: &ops::Mul, children: vec![sin_y, cos_y] };

    let pythagorean = ExprTree::Op { op: &ops::Add, children: vec![sin_sq, cos_sq] };
    let expr = ExprTree::Op { op: &ops::Add, children: vec![pythagorean, sin_cos_y] };

    run_discovery(&expr, costs, 20);
}

fn demo_nested_inverses(costs: &CostModel) {
    println!("\n━━━ Demo 4: Nested Inverses ━━━");
    println!("Input: 1 / (1 / (1 / x))");
    println!("Human might miss: triple reciprocal = single reciprocal\n");

    // Build: recip(recip(recip(x)))
    let r1 = ExprTree::Op { op: &ops::Recip, children: vec![ExprTree::var(0)] };
    let r2 = ExprTree::Op { op: &ops::Recip, children: vec![r1] };
    let expr = ExprTree::Op { op: &ops::Recip, children: vec![r2] };

    run_discovery(&expr, costs, 10);
}

fn demo_factor_discovery(costs: &CostModel) {
    println!("\n━━━ Demo 5: Factoring Discovery ━━━");
    println!("Input: a*x + a*y + a*z");
    println!("E-graph discovers: a*(x + y + z) via distributivity!\n");

    // Build: a*x + a*y + a*z
    let a = ExprTree::var(0);
    let x = ExprTree::var(1);
    let y = ExprTree::var(2);
    let z = ExprTree::var(3);

    let ax = ExprTree::Op { op: &ops::Mul, children: vec![a.clone(), x] };
    let ay = ExprTree::Op { op: &ops::Mul, children: vec![a.clone(), y] };
    let az = ExprTree::Op { op: &ops::Mul, children: vec![a.clone(), z] };

    let sum1 = ExprTree::Op { op: &ops::Add, children: vec![ax, ay] };
    let expr = ExprTree::Op { op: &ops::Add, children: vec![sum1, az] };

    run_discovery(&expr, costs, 15);
}

fn demo_multi_path(costs: &CostModel) {
    println!("\n━━━ Demo 6: Multiple Optimization Paths ━━━");
    println!("Input: (a - b) + b + neg(neg(c))");
    println!("Multiple rewrites combine:");
    println!("  - Canonicalize: a - b → a + neg(b)");
    println!("  - Cancel: (a + neg(b)) + b → a");
    println!("  - Involution: neg(neg(c)) → c");
    println!("Result: a + c\n");

    // Build: (a - b) + b + neg(neg(c))
    let a = ExprTree::var(0);
    let b = ExprTree::var(1);
    let c = ExprTree::var(2);

    let a_minus_b = ExprTree::Op { op: &ops::Sub, children: vec![a, b.clone()] };
    let neg_c = ExprTree::Op { op: &ops::Neg, children: vec![c] };
    let neg_neg_c = ExprTree::Op { op: &ops::Neg, children: vec![neg_c] };

    let sum1 = ExprTree::Op { op: &ops::Add, children: vec![a_minus_b, b] };
    let expr = ExprTree::Op { op: &ops::Add, children: vec![sum1, neg_neg_c] };

    run_discovery(&expr, costs, 20);
}

fn run_discovery(expr: &ExprTree, costs: &CostModel, max_iters: usize) {
    let cost_before = tree_cost(expr, costs);
    println!("Initial cost: {}", cost_before);
    println!("Initial size: {} nodes", tree_size(expr));

    let rules = all_math_rules();
    let mut egraph = EGraph::with_rules(rules);
    let root = egraph.add_expr(expr);

    println!("\nSaturation progress:");
    let mut prev_classes = 0;
    let mut prev_nodes = 0;

    for i in 1..=max_iters {
        let changes = egraph.apply_rules_once();

        let classes = egraph.num_classes();
        let nodes = egraph.node_count();

        if classes != prev_classes || nodes != prev_nodes {
            println!("  Iter {:2}: {} e-classes, {} nodes (+{} new)",
                     i, classes, nodes, changes);
            prev_classes = classes;
            prev_nodes = nodes;
        }

        if changes == 0 {
            println!("  Saturated at iteration {}", i);
            break;
        }
    }

    let (best_expr, cost_after) = egraph.extract_best(root, costs);

    println!("\nBest expression found:");
    println!("  Cost: {} → {} (saved {})", cost_before, cost_after,
             cost_before.saturating_sub(cost_after));
    println!("  Size: {} → {} nodes", tree_size(expr), tree_size(&best_expr));
    println!("  Form: {}", format_expr(&best_expr));

    // Show what the e-graph contains
    println!("\nE-graph explored {} equivalent forms in {} e-classes",
             egraph.node_count(), egraph.num_classes());
}

fn tree_cost(expr: &ExprTree, costs: &CostModel) -> usize {
    match expr {
        ExprTree::Leaf(_) => 0,
        ExprTree::Op { op, children } => {
            let child_cost: usize = children.iter().map(|c| tree_cost(c, costs)).sum();
            child_cost + costs.cost(op.kind())
        }
    }
}

fn tree_size(expr: &ExprTree) -> usize {
    match expr {
        ExprTree::Leaf(_) => 1,
        ExprTree::Op { children, .. } => {
            1 + children.iter().map(tree_size).sum::<usize>()
        }
    }
}

fn format_expr(expr: &ExprTree) -> String {
    match expr {
        ExprTree::Leaf(Leaf::Var(i)) => {
            let names = ["x", "y", "z", "w", "a", "b", "c", "d"];
            names.get(*i as usize).unwrap_or(&"?").to_string()
        }
        ExprTree::Leaf(Leaf::Const(v)) => {
            if *v == 1.0 { "1".to_string() }
            else if *v == 0.0 { "0".to_string() }
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
                _ => "op",
            };

            if children.len() == 1 {
                format!("{}({})", name, format_expr(&children[0]))
            } else if children.len() == 2 && ["+", "-", "*", "/"].contains(&name) {
                format!("({} {} {})", format_expr(&children[0]), name, format_expr(&children[1]))
            } else {
                let args: Vec<_> = children.iter().map(format_expr).collect();
                format!("{}({})", name, args.join(", "))
            }
        }
    }
}
