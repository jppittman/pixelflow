#![allow(dead_code, unused_variables, unused_imports, clippy::all)]
//! E2E test for NNUE-guided optimization.
//!
//! **All costs are NNUE value head predictions** - no hardcoded lookup tables.

use pixelflow_ir::{Expr, OpKind};
use pixelflow_search::egraph::{NnueOptimizer, OptimizeConfig};
use std::time::Instant;

fn main() {
    println!("NNUE-Guided Optimization E2E Test");
    println!("==================================");
    println!("All costs are NNUE value head predictions (log-ns).\n");

    // Try to load the mask model (trained for filtering + value prediction)
    // mask_reinforce.bin was trained with REINFORCE algorithm
    let model_path = "pixelflow-pipeline/data/mask_reinforce.bin";
    let optimizer = match NnueOptimizer::load(model_path) {
        Ok(opt) => {
            println!("✓ Loaded mask model from {}", model_path);
            opt
        }
        Err(e) => {
            // Fallback to judge if mask not found
            println!("! Could not load mask model: {}. Trying judge...", e);
            let judge_path = "pixelflow-pipeline/data/judge.bin";
            match NnueOptimizer::load(judge_path) {
                Ok(opt) => {
                    println!("✓ Loaded judge model from {}", judge_path);
                    opt
                }
                Err(e2) => {
                    println!("! Could not load judge: {}. Using random model.", e2);
                    NnueOptimizer::random(42)
                }
            }
        }
    };

    // Test cases
    let test_cases: Vec<(&str, Expr)> = vec![
        // Simple identity: x + 0 -> x
        ("x + 0", Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(0.0)),
        )),

        // Zero multiplication: x * 0 -> 0
        ("x * 0", Expr::Binary(
            OpKind::Mul,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(0.0)),
        )),

        // Identity multiplication: x * 1 -> x
        ("x * 1", Expr::Binary(
            OpKind::Mul,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(1.0)),
        )),

        // Double identity: (x + 0) * 1 -> x
        ("(x + 0) * 1", Expr::Binary(
            OpKind::Mul,
            Box::new(Expr::Binary(
                OpKind::Add,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(0.0)),
            )),
            Box::new(Expr::Const(1.0)),
        )),

        // FMA candidate: a * b + c
        ("a * b + c", Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Binary(
                OpKind::Mul,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
            )),
            Box::new(Expr::Var(2)),
        )),

        // More complex: (x + 0) * (y + 0) + 0
        ("(x + 0) * (y + 0) + 0", Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Binary(
                OpKind::Mul,
                Box::new(Expr::Binary(
                    OpKind::Add,
                    Box::new(Expr::Var(0)),
                    Box::new(Expr::Const(0.0)),
                )),
                Box::new(Expr::Binary(
                    OpKind::Add,
                    Box::new(Expr::Var(1)),
                    Box::new(Expr::Const(0.0)),
                )),
            )),
            Box::new(Expr::Const(0.0)),
        )),

        // Pythagorean: sqrt(x*x + y*y)
        ("sqrt(x*x + y*y)", Expr::Unary(
            OpKind::Sqrt,
            Box::new(Expr::Binary(
                OpKind::Add,
                Box::new(Expr::Binary(
                    OpKind::Mul,
                    Box::new(Expr::Var(0)),
                    Box::new(Expr::Var(0)),
                )),
                Box::new(Expr::Binary(
                    OpKind::Mul,
                    Box::new(Expr::Var(1)),
                    Box::new(Expr::Var(1)),
                )),
            )),
        )),
    ];

    // Try different thresholds
    println!("Testing different thresholds...\n");

    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9] {
        let config = OptimizeConfig {
            threshold,
            ..OptimizeConfig::default()
        };

        // Test on a complex expression
        let expr = Expr::Binary(
            OpKind::Mul,
            Box::new(Expr::Binary(
                OpKind::Add,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(0.0)),
            )),
            Box::new(Expr::Binary(
                OpKind::Mul,
                Box::new(Expr::Var(1)),
                Box::new(Expr::Const(1.0)),
            )),
        );

        let result = optimizer.optimize(&expr, config);
        println!("threshold={:.1}: cost {:.3} -> {:.3}, skip={:.0}%, epochs={}",
            threshold, result.initial_cost, result.final_cost,
            result.skip_rate() * 100.0, result.epochs_used);
    }

    println!();

    let config = OptimizeConfig {
        threshold: 0.1,  // Lower threshold to let more through
        ..OptimizeConfig::default()
    };

    println!("Config: max_epochs={}, max_classes={}, threshold={}, beam_width={}\n",
        config.max_epochs, config.max_classes, config.threshold, config.beam_width);

    let mut total_time = std::time::Duration::ZERO;
    let mut total_improvement = 0.0;

    for (name, expr) in &test_cases {
        let start = Instant::now();
        let result = optimizer.optimize(expr, config.clone());
        let elapsed = start.elapsed();
        total_time += elapsed;

        let improvement = if result.initial_cost > 0.0 {
            (1.0 - result.cost_ratio()) * 100.0
        } else {
            0.0
        };
        total_improvement += improvement;

        println!("{:30} cost: {:7.3} -> {:7.3} ({:+5.1}%) skip: {:4.0}% epochs: {:2} time: {:?}",
            name,
            result.initial_cost,
            result.final_cost,
            improvement,
            result.skip_rate() * 100.0,
            result.epochs_used,
            elapsed,
        );
    }

    println!("\n---");
    println!("Total time: {:?}", total_time);
    println!("Avg improvement: {:.1}%", total_improvement / test_cases.len() as f32);

    // Comparison with baseline (uniform guide)
    println!("\n\nComparison: NNUE-guided vs Uniform");
    println!("===================================\n");

    // Create a simple baseline by running with threshold=0 (accept everything)
    let baseline_config = OptimizeConfig {
        threshold: 0.0,  // Accept all rules
        ..config.clone()
    };

    for (name, expr) in &test_cases[..3] {
        // NNUE-guided
        let start = Instant::now();
        let nnue_result = optimizer.optimize(expr, config.clone());
        let nnue_time = start.elapsed();

        // Baseline (uniform)
        let start = Instant::now();
        let baseline_result = optimizer.optimize(expr, baseline_config.clone());
        let baseline_time = start.elapsed();

        println!("{}:", name);
        println!("  NNUE:     cost {:.3} -> {:.3}, pairs tried: {:4}, time: {:?}",
            nnue_result.initial_cost, nnue_result.final_cost,
            nnue_result.pairs_tried, nnue_time);
        println!("  Baseline: cost {:.3} -> {:.3}, pairs tried: {:4}, time: {:?}",
            baseline_result.initial_cost, baseline_result.final_cost,
            baseline_result.pairs_tried, baseline_time);
        println!("  Speedup:  {:.1}x fewer pairs\n",
            baseline_result.pairs_tried as f32 / nnue_result.pairs_tried.max(1) as f32);
    }
}
