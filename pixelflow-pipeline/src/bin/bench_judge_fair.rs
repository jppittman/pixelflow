#![allow(dead_code, unused_variables, unused_imports, clippy::all)]
//! Fair benchmark: Judge predictions vs actual JIT kernel execution.
//!
//! Uses the same `compile_dag` path as training data collection, so predictions
//! and measurements are on the same execution footing.

use pixelflow_search::egraph::{ExprTree, Leaf, ops, expr_tree_to_nnue};
use pixelflow_search::nnue::ExprNnue;
use pixelflow_ir::backend::emit::compile_dag;
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::time::Instant;

const JUDGE_WEIGHTS: &str = "pixelflow-pipeline/data/judge.bin";
const JUDGE_META: &str = "pixelflow-pipeline/data/judge.meta.json";
const WARMUP_ITERS: usize = 100_000;
const TIMED_ITERS: usize = 10_000_000;

#[derive(Debug, Deserialize)]
struct ModelMeta {
    target_mean: f32,
    target_std: f32,
}

fn predict_cost(tree: &ExprTree, nnue: &ExprNnue, meta: &ModelMeta) -> f32 {
    let expr = expr_tree_to_nnue(tree);
    let normalized = nnue.predict_log_cost(&expr);
    let log_cost = normalized * meta.target_std + meta.target_mean;
    log_cost.exp()
}

fn bench_jit(tree: &ExprTree) -> f64 {
    let expr = expr_tree_to_nnue(tree);
    let result = compile_dag(&expr)
        .unwrap_or_else(|e| panic!("JIT failed: {e}"));

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        use pixelflow_ir::backend::emit::executable::KernelFn;
        let func: KernelFn = result.code.as_fn();
        let x = vdupq_n_f32(0.5);
        let y = vdupq_n_f32(0.7);
        let z = vdupq_n_f32(1.3);
        let w = vdupq_n_f32(-0.2);
        for _ in 0..WARMUP_ITERS { let _ = func(x, y, z, w); }
        let start = Instant::now();
        for _ in 0..TIMED_ITERS { let _ = func(x, y, z, w); }
        start.elapsed().as_nanos() as f64 / TIMED_ITERS as f64
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        use pixelflow_ir::backend::emit::executable::KernelFn;
        let func: KernelFn = result.code.as_fn();
        let x = _mm_set1_ps(0.5);
        let y = _mm_set1_ps(0.7);
        let z = _mm_set1_ps(1.3);
        let w = _mm_set1_ps(-0.2);
        for _ in 0..WARMUP_ITERS { let _ = func(x, y, z, w); }
        let start = Instant::now();
        for _ in 0..TIMED_ITERS { let _ = func(x, y, z, w); }
        start.elapsed().as_nanos() as f64 / TIMED_ITERS as f64
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    panic!("unsupported architecture")
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  FAIR BENCHMARK: Judge vs JIT Reality");
    println!("  ExprTree → compile_dag → benchmark (same path as training)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let judge = ExprNnue::load(Path::new(JUDGE_WEIGHTS))
        .unwrap_or_else(|e| panic!("Failed to load Judge: {e}"));
    let meta: ModelMeta = serde_json::from_str(
        &fs::read_to_string(JUDGE_META)
            .unwrap_or_else(|e| panic!("Failed to read {JUDGE_META}: {e}")),
    )
    .unwrap_or_else(|e| panic!("Failed to parse judge metadata: {e}"));

    println!("Model: mean={:.3}, std={:.3}", meta.target_mean, meta.target_std);
    println!("Iters: {} warmup + {} timed\n", WARMUP_ITERS, TIMED_ITERS);

    let mut correct = 0;
    let mut total = 0;

    // === DIVISION FAMILY ===
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    DIVISION FAMILY                            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
    correct += cmp(&judge, &meta, "X / Y",
        ExprTree::Op { op: &ops::Div, children: vec![ExprTree::var(0), ExprTree::var(1)] },
        "X * recip(Y)",
        ExprTree::mul(ExprTree::var(0), ExprTree::Op { op: &ops::Recip, children: vec![ExprTree::var(1)] }),
    ); total += 1;

    let denom = ExprTree::add(ExprTree::var(1), ExprTree::Leaf(Leaf::Const(1.0)));
    correct += cmp(&judge, &meta, "X / (Y+1)",
        ExprTree::Op { op: &ops::Div, children: vec![ExprTree::var(0), denom.clone()] },
        "X * recip(Y+1)",
        ExprTree::mul(ExprTree::var(0), ExprTree::Op { op: &ops::Recip, children: vec![denom] }),
    ); total += 1;

    correct += cmp(&judge, &meta, "X / 2",
        ExprTree::Op { op: &ops::Div, children: vec![ExprTree::var(0), ExprTree::Leaf(Leaf::Const(2.0))] },
        "X * 0.5",
        ExprTree::mul(ExprTree::var(0), ExprTree::Leaf(Leaf::Const(0.5))),
    ); total += 1;

    // === NEGATION FAMILY ===
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    NEGATION FAMILY                            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
    correct += cmp(&judge, &meta, "X - Y",
        ExprTree::Op { op: &ops::Sub, children: vec![ExprTree::var(0), ExprTree::var(1)] },
        "X + neg(Y)",
        ExprTree::add(ExprTree::var(0), ExprTree::Op { op: &ops::Neg, children: vec![ExprTree::var(1)] }),
    ); total += 1;

    correct += cmp(&judge, &meta, "neg(neg(X))",
        ExprTree::Op { op: &ops::Neg, children: vec![
            ExprTree::Op { op: &ops::Neg, children: vec![ExprTree::var(0)] },
        ]},
        "X",
        ExprTree::var(0),
    ); total += 1;

    // === SQRT FAMILY ===
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                      SQRT FAMILY                              ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
    correct += cmp(&judge, &meta, "1 / sqrt(X)",
        ExprTree::Op { op: &ops::Div, children: vec![
            ExprTree::Leaf(Leaf::Const(1.0)),
            ExprTree::Op { op: &ops::Sqrt, children: vec![ExprTree::var(0)] },
        ]},
        "rsqrt(X)",
        ExprTree::Op { op: &ops::Rsqrt, children: vec![ExprTree::var(0)] },
    ); total += 1;

    // === COMPOUND ===
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                  COMPOUND EXPRESSIONS                         ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
    let abs_x = ExprTree::Op { op: &ops::Abs, children: vec![ExprTree::var(0)] };
    let denom2 = ExprTree::add(abs_x, ExprTree::Leaf(Leaf::Const(1.0)));
    correct += cmp(&judge, &meta, "X / (|X|+1)",
        ExprTree::Op { op: &ops::Div, children: vec![ExprTree::var(0), denom2.clone()] },
        "X * recip(|X|+1)",
        ExprTree::mul(ExprTree::var(0), ExprTree::Op { op: &ops::Recip, children: vec![denom2] }),
    ); total += 1;

    let dx = ExprTree::Op { op: &ops::Sub, children: vec![ExprTree::var(0), ExprTree::Leaf(Leaf::Const(0.5))] };
    let dy = ExprTree::Op { op: &ops::Sub, children: vec![ExprTree::var(1), ExprTree::Leaf(Leaf::Const(0.5))] };
    let dx_neg = ExprTree::add(ExprTree::var(0), ExprTree::Op { op: &ops::Neg, children: vec![ExprTree::Leaf(Leaf::Const(0.5))] });
    let dy_neg = ExprTree::add(ExprTree::var(1), ExprTree::Op { op: &ops::Neg, children: vec![ExprTree::Leaf(Leaf::Const(0.5))] });
    correct += cmp(&judge, &meta, "(X-0.5)²+(Y-0.5)²",
        ExprTree::add(ExprTree::mul(dx.clone(), dx), ExprTree::mul(dy.clone(), dy)),
        "(X+neg(0.5))²+...",
        ExprTree::add(ExprTree::mul(dx_neg.clone(), dx_neg), ExprTree::mul(dy_neg.clone(), dy_neg)),
    ); total += 1;

    // === SUMMARY ===
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SUMMARY: Judge got {}/{} correct ({:.0}%)",
        correct, total, 100.0 * correct as f64 / total as f64);
    println!("═══════════════════════════════════════════════════════════════");
}

fn cmp(judge: &ExprNnue, meta: &ModelMeta, name_a: &str, tree_a: ExprTree, name_b: &str, tree_b: ExprTree) -> usize {
    println!("━━━ {} vs {} ━━━", name_a, name_b);
    let pred_a = predict_cost(&tree_a, judge, meta);
    let pred_b = predict_cost(&tree_b, judge, meta);
    let actual_a = bench_jit(&tree_a);
    let actual_b = bench_jit(&tree_b);

    let ratio_a = actual_a / pred_a as f64;
    let ratio_b = actual_b / pred_b as f64;
    println!("  {:<25} pred: {:>5.2} ns  actual: {:>5.2} ns  ratio: {:.2}x", name_a, pred_a, actual_a, ratio_a);
    println!("  {:<25} pred: {:>5.2} ns  actual: {:>5.2} ns  ratio: {:.2}x", name_b, pred_b, actual_b, ratio_b);

    let pred_winner = if pred_a < pred_b { "A" } else { "B" };
    let actual_winner = if actual_a < actual_b { "A" } else { "B" };
    let diff_pct = ((actual_a - actual_b).abs() / actual_a.min(actual_b) * 100.0) as i32;

    if diff_pct < 5 {
        println!("  → TIE (within 5%): {:.2} vs {:.2} ns\n", actual_a, actual_b);
        1
    } else {
        let correct = pred_winner == actual_winner;
        println!("  → Judge: {}  Reality: {}  {}\n",
            pred_winner, actual_winner,
            if correct { "✓ CORRECT" } else { "✗ WRONG" });
        if correct { 1 } else { 0 }
    }
}
