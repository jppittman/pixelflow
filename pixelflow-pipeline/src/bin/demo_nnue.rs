#![allow(dead_code, unused_variables, unused_imports, clippy::all)]
//! Demo: NNUE-based cost extraction.
//!
//! Shows how the ExprNnue (Judge) works with e-graph extraction.
//! Uses `extract_beam` which calls `predict_log_cost()` on full trees,
//! capturing ILP and structural information via dual accumulator.

use pixelflow_search::egraph::{EGraph, ExprTree, Leaf, ops};
use pixelflow_search::egraph::{extract_beam, predict_tree_cost, expr_tree_to_nnue, CostModel};
use pixelflow_search::nnue::ExprNnue;
use pixelflow_search::math::all_math_rules;
use std::path::Path;

/// Default beam width for neural extraction
const BEAM_WIDTH: usize = 16;

/// Path to trained Judge weights
const JUDGE_WEIGHTS: &str = "pixelflow-pipeline/data/judge.bin";

fn main() {
    println!("=== NNUE Neural Extraction Demo ===");
    println!("Using TRAINED Judge with beam search extraction\n");

    // Load the trained Judge - fail loudly if missing
    let nnue = ExprNnue::load(Path::new(JUDGE_WEIGHTS))
        .unwrap_or_else(|e| panic!("Failed to load trained Judge from {}: {}", JUDGE_WEIGHTS, e));
    println!("Loaded trained Judge from {}", JUDGE_WEIGHTS);

    println!("ExprNnue Parameters: {} (~{} KB)",
             ExprNnue::param_count(),
             ExprNnue::memory_bytes() / 1024);

    // Demo 1: Show dual accumulator captures ILP
    demo_ilp_awareness(&nnue);

    // Demo 2: Expression with optimization potential
    demo_optimization(&nnue);

    // Demo 3: Compare neural extraction vs additive extraction
    demo_extraction_comparison(&nnue);

    // Demo 4: Trig identity with neural extraction
    demo_trig_neural(&nnue);

    println!("\n=== NNUE Demo Complete ===");
}

fn demo_ilp_awareness(nnue: &ExprNnue) {
    println!("━━━ Demo 1: ILP Awareness ━━━");
    println!("The Judge sees tree geometry via dual accumulator (flat + depth-encoded).\n");

    // Two expressions with same ops but different structure:
    // Sequential: ((x * y) * z) - one critical path of depth 2
    // Parallel: (x * y) + (z * w) - two parallel muls, then add

    let sequential = ExprTree::mul(
        ExprTree::mul(ExprTree::var(0), ExprTree::var(1)),
        ExprTree::var(2),
    );

    let parallel = ExprTree::add(
        ExprTree::mul(ExprTree::var(0), ExprTree::var(1)),
        ExprTree::mul(ExprTree::var(2), ExprTree::var(3)),
    );

    let seq_expr = expr_tree_to_nnue(&sequential);
    let par_expr = expr_tree_to_nnue(&parallel);

    let seq_cost = nnue.predict_log_cost(&seq_expr);
    let par_cost = nnue.predict_log_cost(&par_expr);

    println!("  Sequential: (x * y) * z");
    println!("    Depth: {}, Nodes: {}", sequential.depth(), sequential.node_count());
    println!("    Log-cost: {:.3} → {:.2} ns\n", seq_cost, libm::expf(seq_cost));

    println!("  Parallel: (x * y) + (z * w)");
    println!("    Depth: {}, Nodes: {}", parallel.depth(), parallel.node_count());
    println!("    Log-cost: {:.3} → {:.2} ns\n", par_cost, libm::expf(par_cost));

    // Show accumulator stats
    use pixelflow_search::nnue::EdgeAccumulator;
    let seq_acc = EdgeAccumulator::from_expr(&seq_expr, &nnue.embeddings);
    let par_acc = EdgeAccumulator::from_expr(&par_expr, &nnue.embeddings);

    println!("  Accumulator Stats:");
    println!("    {:20} {:>10} {:>10}", "", "Sequential", "Parallel");
    println!("    {:20} {:>10} {:>10}", "Edge Count", seq_acc.edge_count, par_acc.edge_count);
    println!("    {:20} {:>10} {:>10}", "Node Count", seq_acc.node_count, par_acc.node_count);
    println!("    (Depth-encoded half captures tree geometry via sinusoidal Hadamard PE)");
    println!();
}

fn demo_optimization(nnue: &ExprNnue) {
    println!("━━━ Demo 2: Beam Search Optimization ━━━");
    println!("Input: exp(ln(x) + ln(y))");
    println!("Using extract_beam() with beam_width={}.\n", BEAM_WIDTH);

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

    // Initial NNUE cost
    let initial_cost = predict_tree_cost(&expr, nnue);
    println!("Initial neural cost: {:.2} ns", initial_cost);

    // Run e-graph saturation
    let rules = all_math_rules();
    let mut egraph = EGraph::with_rules(rules);
    let root = egraph.add_expr(&expr);

    for _ in 0..15 {
        let changes = egraph.apply_rules_once();
        if changes == 0 { break; }
    }

    println!("E-graph: {} classes, {} nodes", egraph.num_classes(), egraph.node_count());

    // Extract with beam search
    let (best_expr, log_cost) = extract_beam(&egraph, root, nnue, BEAM_WIDTH);
    let final_cost = libm::expf(log_cost);

    println!("Final neural cost: {:.2} ns (log: {:.3})", final_cost, log_cost);
    println!("Form: {}\n", format_expr(&best_expr));
}

fn demo_extraction_comparison(nnue: &ExprNnue) {
    println!("━━━ Demo 3: Beam Search vs Additive Extraction ━━━");
    println!("Comparing extract_beam() vs standard extract().\n");

    // Build: (a*b + c*d) * (e*f + g*h) - lots of ILP potential
    let ab = ExprTree::mul(ExprTree::var(0), ExprTree::var(1));
    let cd = ExprTree::mul(ExprTree::var(2), ExprTree::var(3));
    let ef = ExprTree::mul(ExprTree::var(4), ExprTree::var(5));
    let gh = ExprTree::mul(ExprTree::var(6), ExprTree::var(7));
    let left = ExprTree::add(ab, cd);
    let right = ExprTree::add(ef, gh);
    let expr = ExprTree::mul(left, right);

    let rules = all_math_rules();
    let mut egraph = EGraph::with_rules(rules);
    let root = egraph.add_expr(&expr);

    for _ in 0..10 {
        let changes = egraph.apply_rules_once();
        if changes == 0 { break; }
    }

    println!("E-graph: {} classes, {} nodes\n", egraph.num_classes(), egraph.node_count());

    // Additive extraction (standard)
    let additive_costs = CostModel::default();
    let (additive_tree, additive_cost) = egraph.extract_best(root, &additive_costs);
    let additive_neural = predict_tree_cost(&additive_tree, nnue);

    println!("  Additive extraction (CostModel):");
    println!("    Additive cost: {}", additive_cost);
    println!("    Neural eval:   {:.2} ns", additive_neural);
    println!("    Form: {}\n", format_expr(&additive_tree));

    // Beam search extraction
    let (beam_tree, beam_log_cost) = extract_beam(&egraph, root, nnue, BEAM_WIDTH);
    let beam_cost = libm::expf(beam_log_cost);

    println!("  Beam search extraction (k={}):", BEAM_WIDTH);
    println!("    Neural cost:   {:.2} ns (log: {:.3})", beam_cost, beam_log_cost);
    println!("    Form: {}\n", format_expr(&beam_tree));
}

fn demo_trig_neural(nnue: &ExprNnue) {
    println!("━━━ Demo 4: Trig Identity with Beam Search ━━━");
    println!("Input: sin(x)*cos(x) + cos(x)*sin(x)");
    println!("Beam search (k={}) evaluates full trees.\n", BEAM_WIDTH);

    // Build: sin(x)*cos(x) + cos(x)*sin(x)
    let sin_x = ExprTree::Op { op: &ops::Sin, children: vec![ExprTree::var(0)] };
    let cos_x = ExprTree::Op { op: &ops::Cos, children: vec![ExprTree::var(0)] };
    let term1 = ExprTree::mul(sin_x.clone(), cos_x.clone());
    let term2 = ExprTree::mul(cos_x, sin_x);
    let expr = ExprTree::add(term1, term2);

    let initial_cost = predict_tree_cost(&expr, nnue);
    println!("Initial neural cost: {:.2} ns", initial_cost);

    // Run e-graph saturation
    let rules = all_math_rules();
    let mut egraph = EGraph::with_rules(rules);
    let root = egraph.add_expr(&expr);

    let mut prev_classes = 0;
    for i in 1..=20 {
        let changes = egraph.apply_rules_once();
        if egraph.num_classes() != prev_classes {
            println!("  Iter {:2}: {} classes, {} nodes (+{} new)",
                     i, egraph.num_classes(), egraph.node_count(), changes);
            prev_classes = egraph.num_classes();
        }
        if changes == 0 {
            println!("  Saturated at iteration {}", i);
            break;
        }
    }

    // Extract with beam search
    let (best_expr, log_cost) = extract_beam(&egraph, root, nnue, BEAM_WIDTH);
    let final_cost = libm::expf(log_cost);

    println!("\nBeam search selected (k={}):", BEAM_WIDTH);
    println!("  Cost: {:.2} ns → {:.2} ns", initial_cost, final_cost);
    println!("  Log-cost: {:.3}", log_cost);
    println!("  Form: {}", format_expr(&best_expr));

    // Show the accumulator stats of the result
    let result_expr = expr_tree_to_nnue(&best_expr);
    let result_acc = pixelflow_search::nnue::EdgeAccumulator::from_expr(&result_expr, &nnue.embeddings);
    println!("\n  Result accumulator stats:");
    println!("    Edge Count: {}", result_acc.edge_count);
    println!("    Node Count: {}", result_acc.node_count);
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
