#![allow(dead_code, unused_variables, unused_imports, clippy::all)]
//! Demo: Algebraic rewrite rules on real graphics patterns.
//!
//! Shows how the new categorical trait-based rules transform common
//! graphics computations from pixelflow-graphics.

use pixelflow_search::egraph::{CostModel, EGraph, ExprTree, Leaf, all_rules, ops};

fn main() {
    println!("=== PixelFlow Algebraic Rewrite Demo ===\n");

    let rules = all_rules();
    println!("Total mathematical rules: {}\n", rules.len());

    let costs = CostModel::default();

    // Demo 1: Parity rules - sin(-x) → -sin(x)
    demo_parity(&costs);

    // Demo 2: Even function - cos(-x) → cos(x)
    demo_even_parity(&costs);

    // Demo 3: Double negation - neg(neg(x)) → x
    demo_double_negation(&costs);

    // Demo 4: exp/ln inverse - exp(ln(x)) → x
    demo_exp_ln_inverse(&costs);

    // Demo 5: ln homomorphism - ln(a * b) → ln(a) + ln(b)
    demo_ln_homomorphism(&costs);

    // Demo 6: Pythagorean identity - sin²(x) + cos²(x) → 1
    demo_pythagorean(&costs);

    // Demo 7: Angle addition - sin(a + b)
    demo_angle_addition(&costs);

    // Demo 8: Canonicalization - a - b → a + neg(b)
    demo_canonicalize(&costs);

    // Demo 9: Real graphics pattern: vector normalization
    demo_vector_normalize(&costs);

    println!("\n=== Demo Complete ===");
}

fn demo_parity(costs: &CostModel) {
    println!("--- Demo 1: Parity (Odd Function) ---");
    println!("Input:  sin(neg(x))");
    println!("Rule:   sin is odd → sin(-x) = -sin(x)");

    // Build: sin(neg(x))
    let expr = ExprTree::Op {
        op: &ops::Sin,
        children: vec![
            ExprTree::Op {
                op: &ops::Neg,
                children: vec![ExprTree::var(0)],
            }
        ],
    };

    let (before, after) = optimize_and_compare(&expr, costs);
    println!("Before: cost = {}", before);
    println!("After:  cost = {}", after);
    println!("Saved:  {} ops\n", before.saturating_sub(after));
}

fn demo_even_parity(costs: &CostModel) {
    println!("--- Demo 2: Parity (Even Function) ---");
    println!("Input:  cos(neg(x))");
    println!("Rule:   cos is even → cos(-x) = cos(x)");

    // Build: cos(neg(x))
    let expr = ExprTree::Op {
        op: &ops::Cos,
        children: vec![
            ExprTree::Op {
                op: &ops::Neg,
                children: vec![ExprTree::var(0)],
            }
        ],
    };

    let (before, after) = optimize_and_compare(&expr, costs);
    println!("Before: cost = {}", before);
    println!("After:  cost = {}", after);
    println!("Saved:  {} ops (negation removed!)\n", before.saturating_sub(after));
}

fn demo_double_negation(costs: &CostModel) {
    println!("--- Demo 3: Involution ---");
    println!("Input:  neg(neg(x))");
    println!("Rule:   neg(neg(x)) → x");

    // Build: neg(neg(x))
    let expr = ExprTree::Op {
        op: &ops::Neg,
        children: vec![
            ExprTree::Op {
                op: &ops::Neg,
                children: vec![ExprTree::var(0)],
            }
        ],
    };

    let (before, after) = optimize_and_compare(&expr, costs);
    println!("Before: cost = {}", before);
    println!("After:  cost = {}", after);
    println!("Saved:  {} ops\n", before.saturating_sub(after));
}

fn demo_exp_ln_inverse(costs: &CostModel) {
    println!("--- Demo 4: Function Inverse ---");
    println!("Input:  exp(ln(x))");
    println!("Rule:   exp(ln(x)) → x");

    // Build: exp(ln(x))
    let expr = ExprTree::Op {
        op: &ops::Exp,
        children: vec![
            ExprTree::Op {
                op: &ops::Ln,
                children: vec![ExprTree::var(0)],
            }
        ],
    };

    let (before, after) = optimize_and_compare(&expr, costs);
    println!("Before: cost = {}", before);
    println!("After:  cost = {}", after);
    println!("Saved:  {} ops\n", before.saturating_sub(after));
}

fn demo_ln_homomorphism(costs: &CostModel) {
    println!("--- Demo 5: Homomorphism ---");
    println!("Input:  ln(a * b)");
    println!("Rule:   ln(a * b) → ln(a) + ln(b)");

    // Build: ln(x * y)
    let expr = ExprTree::Op {
        op: &ops::Ln,
        children: vec![
            ExprTree::Op {
                op: &ops::Mul,
                children: vec![ExprTree::var(0), ExprTree::var(1)],
            }
        ],
    };

    let rules = all_rules();
    let mut egraph = EGraph::with_rules(rules);
    let root = egraph.add_expr(&expr);

    // Run a few iterations
    for _ in 0..5 {
        egraph.apply_rules_once();
    }

    let (_, cost_after) = egraph.extract_best(root, costs);
    let cost_before = simple_cost(&expr, costs);

    println!("Before: cost = {}", cost_before);
    println!("After:  cost = {} (expands to ln(a) + ln(b))", cost_after);
    println!("Note:   Expansion may increase cost but enables further opts\n");
}

fn demo_pythagorean(costs: &CostModel) {
    println!("--- Demo 6: Pythagorean Identity ---");
    println!("Input:  sin(x)² + cos(x)²");
    println!("Rule:   sin²(x) + cos²(x) → 1");

    // Build: sin(x)*sin(x) + cos(x)*cos(x)
    let sin_x = ExprTree::Op {
        op: &ops::Sin,
        children: vec![ExprTree::var(0)],
    };
    let cos_x = ExprTree::Op {
        op: &ops::Cos,
        children: vec![ExprTree::var(0)],
    };
    let sin_sq = ExprTree::Op {
        op: &ops::Mul,
        children: vec![sin_x.clone(), sin_x],
    };
    let cos_sq = ExprTree::Op {
        op: &ops::Mul,
        children: vec![cos_x.clone(), cos_x],
    };
    let expr = ExprTree::Op {
        op: &ops::Add,
        children: vec![sin_sq, cos_sq],
    };

    let (before, after) = optimize_and_compare(&expr, costs);
    println!("Before: cost = {} (2 trig + 2 mul + 1 add)", before);
    println!("After:  cost = {} (constant 1!)", after);
    println!("Saved:  {} ops\n", before.saturating_sub(after));
}

fn demo_angle_addition(costs: &CostModel) {
    println!("--- Demo 7: Angle Addition ---");
    println!("Input:  sin(a + b)");
    println!("Rule:   sin(a + b) → sin(a)cos(b) + cos(a)sin(b)");

    // Build: sin(x + y)
    let expr = ExprTree::Op {
        op: &ops::Sin,
        children: vec![
            ExprTree::Op {
                op: &ops::Add,
                children: vec![ExprTree::var(0), ExprTree::var(1)],
            }
        ],
    };

    let rules = all_rules();
    let mut egraph = EGraph::with_rules(rules);
    let root = egraph.add_expr(&expr);

    // Run iterations
    for _ in 0..5 {
        egraph.apply_rules_once();
    }

    let (_, cost_after) = egraph.extract_best(root, costs);
    let cost_before = simple_cost(&expr, costs);

    println!("Before: cost = {}", cost_before);
    println!("After:  cost = {} (now has sin(a)cos(b) + cos(a)sin(b) available)", cost_after);
    println!("Note:   E-graph contains both forms for downstream use\n");
}

fn demo_canonicalize(costs: &CostModel) {
    println!("--- Demo 8: Canonicalization ---");
    println!("Input:  a - b");
    println!("Rule:   a - b → a + neg(b)");

    // Build: x - y
    let expr = ExprTree::Op {
        op: &ops::Sub,
        children: vec![ExprTree::var(0), ExprTree::var(1)],
    };

    let rules = all_rules();
    let mut egraph = EGraph::with_rules(rules);
    let root = egraph.add_expr(&expr);
    egraph.apply_rules_once();

    println!("E-graph now contains both 'a - b' and 'a + neg(b)'");
    println!("This enables further optimizations like (a - b) + b → a\n");
}

fn demo_vector_normalize(costs: &CostModel) {
    println!("--- Demo 9: Vector Normalization (Real Graphics Pattern) ---");
    println!("Input:  x / sqrt(x² + y² + z²)");
    println!("This appears in scene3d.rs for normal vector computation\n");

    // Build: x / sqrt(x*x + y*y + z*z)
    let x = ExprTree::var(0);
    let y = ExprTree::var(1);
    let z = ExprTree::var(2);

    let x_sq = ExprTree::Op {
        op: &ops::Mul,
        children: vec![x.clone(), x.clone()],
    };
    let y_sq = ExprTree::Op {
        op: &ops::Mul,
        children: vec![y.clone(), y.clone()],
    };
    let z_sq = ExprTree::Op {
        op: &ops::Mul,
        children: vec![z.clone(), z.clone()],
    };

    let sum_xy = ExprTree::Op {
        op: &ops::Add,
        children: vec![x_sq, y_sq],
    };
    let sum_xyz = ExprTree::Op {
        op: &ops::Add,
        children: vec![sum_xy, z_sq],
    };

    let magnitude = ExprTree::Op {
        op: &ops::Sqrt,
        children: vec![sum_xyz],
    };

    let normalized_x = ExprTree::Op {
        op: &ops::Div,
        children: vec![x, magnitude],
    };

    let (before, after) = optimize_and_compare(&normalized_x, costs);
    println!("Before: cost = {} (3 mul + 2 add + 1 sqrt + 1 div)", before);
    println!("After:  cost = {}", after);
    println!("Note:   With FMA fusion (in compiler), this optimizes further\n");
}

fn optimize_and_compare(expr: &ExprTree, costs: &CostModel) -> (usize, usize) {
    let cost_before = simple_cost(expr, costs);

    let rules = all_rules();
    let mut egraph = EGraph::with_rules(rules);
    let root = egraph.add_expr(expr);

    // Run saturation
    for _ in 0..10 {
        let changes = egraph.apply_rules_once();
        if changes == 0 {
            break;
        }
    }

    let (_, cost_after) = egraph.extract_best(root, costs);
    (cost_before, cost_after)
}

fn simple_cost(expr: &ExprTree, costs: &CostModel) -> usize {
    match expr {
        ExprTree::Leaf(Leaf::Var(_)) | ExprTree::Leaf(Leaf::Const(_)) => 0,
        ExprTree::Op { op, children } => {
            let child_cost: usize = children.iter().map(|c| simple_cost(c, costs)).sum();
            child_cost + costs.cost(op.kind())
        }
    }
}
