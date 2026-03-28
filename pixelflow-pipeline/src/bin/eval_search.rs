#![allow(warnings)]
//! Evaluate GuidedSearch vs egg baseline.
//!
//! This binary compares the Guide-filtered search against egg-style saturation
//! to measure:
//! 1. Whether Guide achieves similar final costs
//! 2. Whether Guide uses fewer epochs (efficiency)
//! 3. Whether Guide correctly prunes non-matching rules
//!
//! # Usage
//!
//! ```bash
//! cargo run -p pixelflow-pipeline --bin eval_search --release
//! ```
//!
//! # Output
//!
//! Comparison metrics between guided and baseline search.

use pixelflow_search::egraph::{Leaf, ops};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use pixelflow_search::egraph::{
    EGraph, ExprTree, GuidedSearch, Rewrite,
    extract_neural, predict_tree_cost, all_rules,
};
use pixelflow_search::nnue::{BwdGenConfig, BwdGenerator, ExprNnue, Expr, OpKind, RuleTemplates};

/// Evaluate GuidedSearch vs baseline.
#[derive(Parser, Debug)]
#[command(name = "eval_search")]
#[command(about = "Evaluate GuidedSearch vs egg baseline")]
struct Args {
    /// Number of expressions to evaluate
    #[arg(short, long, default_value_t = 50)]
    count: usize,

    /// Random seed
    #[arg(short, long, default_value_t = 123)]
    seed: u64,

    /// Maximum epochs for search
    #[arg(long, default_value_t = 50)]
    max_epochs: usize,

    /// Path to Judge NNUE weights (also provides mask head for guided search)
    #[arg(long, default_value = "pixelflow-pipeline/data/judge.bin")]
    judge_model: String,

    /// Path to mask weights (optional; if absent, bootstraps random mask from judge)
    #[arg(long)]
    mask_weights: Option<String>,

    /// Mask probability threshold for guided search
    #[arg(long, default_value_t = 0.4)]
    threshold: f32,

    /// Max e-graph classes for guided search budget
    #[arg(long, default_value_t = 200)]
    max_classes: usize,

    /// Include hard shader expressions (psychedelic channel, normalize, etc.)
    #[arg(long)]
    hard: bool,
}

/// Results from one search run.
#[derive(Debug)]
struct SearchResult {
    initial_cost: f32,
    final_cost: f32,
    epochs_used: usize,
    rules_applied: usize,
    duration_us: u128,
    node_count: usize,
}

fn main() {
    let args = Args::parse();

    let workspace_root = find_workspace_root();

    // Load Judge model (provides both value head and mask head)
    let judge_path = workspace_root.join(&args.judge_model);
    let judge = ExprNnue::load(&judge_path)
        .unwrap_or_else(|e| panic!("Failed to load Judge from {}: {}", judge_path.display(), e));
    println!("Loaded Judge model from: {}", judge_path.display());

    // Build mask-equipped model: either load trained mask weights or bootstrap random
    let mask_model = match &args.mask_weights {
        Some(mask_path) => {
            let full_path = workspace_root.join(mask_path);
            let m = ExprNnue::load(&full_path).unwrap_or_else(|e| {
                panic!("Failed to load mask weights from {}: {}", full_path.display(), e)
            });
            println!("Loaded mask weights from: {}", full_path.display());
            m
        }
        None => {
            let m = judge.with_randomized_mask_weights(42);
            println!("Bootstrapped random mask weights from judge (seed=42)");
            m
        }
    };

    // Build rule templates for mask scoring
    let rules = all_rules();
    let templates = build_rule_templates(&rules);
    println!("Built LHS/RHS templates for {} rules", templates.len());

    println!("\nEvaluation Configuration:");
    println!("  Expressions: {}", args.count);
    println!("  Max epochs: {}", args.max_epochs);
    println!("  Mask threshold: {}", args.threshold);
    println!("  Max classes: {}", args.max_classes);
    println!("  Seed: {}", args.seed);

    // Build named test expressions
    let mut named_exprs: Vec<(String, ExprTree)> = Vec::new();

    // Random expressions
    let config = BwdGenConfig {
        max_depth: 8,
        leaf_prob: 0.2,
        num_vars: 4,
        fused_op_prob: 0.3,
        max_junkify_passes: 2,
        junkify_prob: 0.5,
        max_junkified_nodes: 200,
    };
    let mut expr_gen = BwdGenerator::new(args.seed, config, templates.clone());
    for i in 0..args.count {
        let pair = expr_gen.generate();
        named_exprs.push((format!("random_{:03}", i), expr_to_tree(&pair.unoptimized)));
    }

    // Hard shader expressions (saturation-infeasible)
    if args.hard {
        for (name, tree) in hard_shader_exprs() {
            named_exprs.push((name.to_string(), tree));
        }
    }

    let mut baseline_results = Vec::new();
    let mut guided_results = Vec::new();
    let mut names = Vec::new();

    println!("\n=== Running Evaluation ({} expressions) ===\n", named_exprs.len());

    for (i, (name, expr_tree)) in named_exprs.iter().enumerate() {
        let baseline = run_baseline(expr_tree, &judge, args.max_epochs);

        let guided = run_guided_mask(
            expr_tree, &judge, &mask_model, &templates,
            args.max_epochs, args.threshold, args.max_classes,
        );
        guided_results.push(guided);

        baseline_results.push(baseline);
        names.push(name.clone());

        if (i + 1) % 10 == 0 || name.starts_with("shader_") {
            let b = &baseline_results[i];
            let g = &guided_results[i];
            let pct = if b.initial_cost > 0.0 {
                (1.0 - b.final_cost / b.initial_cost) * 100.0
            } else { 0.0 };
            let gpct = if g.initial_cost > 0.0 {
                (1.0 - g.final_cost / g.initial_cost) * 100.0
            } else { 0.0 };
            print!("  {:30} nodes={:4} cost {:.3} → {:.3} ({:+.1}%)",
                name, b.node_count, b.initial_cost, b.final_cost, -pct);
            print!("  | mask: {:.1} ({:+.1}%) in {}ep", g.final_cost, -gpct, g.epochs_used);
            println!();
        }
    }

    // Aggregate statistics
    println!("\n=== Aggregate Results ===\n");

    print_stats("Baseline (egg-style saturation)", &baseline_results);
    println!();
    print_stats("Mask-Guided Search", &guided_results);

    println!("\n=== Comparison ===\n");
    compare_results(&baseline_results, &guided_results);

    // Per-expression improvement table for hard exprs
    if args.hard {
        println!("\n=== Hard Shader Expressions (Detail) ===\n");
        println!("  {:30} {:>8} {:>8} {:>8} {:>8} {:>6} {:>6}",
            "Name", "Initial", "Baseline", "Mask", "Improv%", "BEp", "MEp");
        println!("  {}", "-".repeat(86));
        for (i, name) in names.iter().enumerate() {
            if !name.starts_with("shader_") { continue; }
            let b = &baseline_results[i];
            let g = &guided_results[i];
            let improv = if b.initial_cost > 0.0 {
                (1.0 - g.final_cost / b.initial_cost) * 100.0
            } else { 0.0 };
            println!("  {:30} {:8.1} {:8.1} {:8.1} {:7.1}% {:6} {:6}",
                name, b.initial_cost, b.final_cost, g.final_cost,
                improv, b.epochs_used, g.epochs_used);
        }
    }
}

/// Run baseline egg-style search (apply all rules every epoch).
fn run_baseline(
    expr_tree: &ExprTree,
    judge: &ExprNnue,
    max_epochs: usize,
) -> SearchResult {
    let start = Instant::now();

    // Judge prediction on the ORIGINAL expression (before optimization)
    let initial_cost = predict_tree_cost(expr_tree, judge);

    let rules = all_rules();
    let num_rules = rules.len();
    let mut egraph = EGraph::with_rules(rules);
    let root = egraph.add_expr(expr_tree);

    let mut epochs_used = 0;
    let mut rules_applied = 0;

    for _ in 0..max_epochs {
        let changes = egraph.apply_rules_once();
        epochs_used += 1;
        rules_applied += num_rules;

        if changes == 0 {
            break; // Saturated
        }
    }

    let (extracted, _log_cost) = extract_neural(&egraph, root, judge);
    let final_cost = predict_tree_cost(&extracted, judge);
    let duration_us = start.elapsed().as_micros();

    SearchResult {
        initial_cost,
        final_cost,
        epochs_used,
        rules_applied,
        duration_us,
        node_count: extracted.node_count(),
    }
}

/// Run mask-guided search using ExprNnue's bilinear mask head.
fn run_guided_mask(
    expr_tree: &ExprTree,
    judge: &ExprNnue,
    mask_model: &ExprNnue,
    templates: &RuleTemplates,
    max_epochs: usize,
    threshold: f32,
    max_classes: usize,
) -> SearchResult {
    let start = Instant::now();
    let initial_cost = predict_tree_cost(expr_tree, judge);

    let rules = all_rules();
    let mut egraph = EGraph::with_rules(rules);
    let root = egraph.add_expr(expr_tree);
    let mut search = GuidedSearch::new(egraph, root, max_epochs);

    let result = search.run_dual_mask_with_templates(
        mask_model,
        templates,
        |t| t.node_count() as i64,
        judge,
        threshold,
        max_classes,
    );

    let final_cost = predict_tree_cost(&result.best_tree, judge);
    let duration_us = start.elapsed().as_micros();

    SearchResult {
        initial_cost,
        final_cost,
        epochs_used: result.epochs_used,
        rules_applied: result.pairs_tried,
        duration_us,
        node_count: result.best_tree.node_count(),
    }
}

/// Build rule templates from rule definitions.
///
/// Each rule provides LHS/RHS expression templates via the Rewrite trait.
fn build_rule_templates(rules: &[Box<dyn Rewrite>]) -> RuleTemplates {
    let mut templates = RuleTemplates::with_capacity(rules.len());

    for (idx, rule) in rules.iter().enumerate() {
        if let (Some(lhs), Some(rhs)) = (rule.lhs_template(), rule.rhs_template()) {
            templates.set(idx, lhs, rhs);
        }
    }

    templates
}

/// Print statistics for a set of results.
fn print_stats(name: &str, results: &[SearchResult]) {
    if results.is_empty() {
        println!("{}: No results", name);
        return;
    }

    let n = results.len() as f64;

    let avg_cost: f64 = results.iter().map(|r| r.final_cost as f64).sum::<f64>() / n;
    let min_cost: f32 = results.iter().map(|r| r.final_cost).fold(f32::MAX, f32::min);
    let max_cost: f32 = results.iter().map(|r| r.final_cost).fold(f32::MIN, f32::max);
    let avg_epochs: f64 = results.iter().map(|r| r.epochs_used as f64).sum::<f64>() / n;
    let avg_rules: f64 = results.iter().map(|r| r.rules_applied as f64).sum::<f64>() / n;
    let avg_time: f64 = results.iter().map(|r| r.duration_us as f64).sum::<f64>() / n;

    println!("{}:", name);
    println!("  Avg final cost:    {:.3} (log-ns)", avg_cost);
    println!("  Cost range:        [{:.3}, {:.3}]", min_cost, max_cost);
    println!("  Avg epochs used:   {:.1}", avg_epochs);
    println!("  Avg rules applied: {:.1}", avg_rules);
    println!("  Avg time (µs):     {:.1}", avg_time);
}

/// Compare baseline and guided results.
fn compare_results(baseline: &[SearchResult], guided: &[SearchResult]) {
    if baseline.len() != guided.len() || baseline.is_empty() {
        println!("Cannot compare: different number of results");
        return;
    }

    let n = baseline.len() as f64;

    // Cost comparison
    let mut same_cost = 0;
    let mut guided_better = 0;
    let mut baseline_better = 0;
    let mut cost_diff_sum: f64 = 0.0;

    for (b, g) in baseline.iter().zip(guided.iter()) {
        let diff = g.final_cost - b.final_cost;
        if diff.abs() < 1e-6 {
            same_cost += 1;
        } else if g.final_cost < b.final_cost {
            guided_better += 1;
        } else {
            baseline_better += 1;
        }
        cost_diff_sum += diff as f64;
    }

    println!("Cost comparison:");
    println!("  Same cost:       {} ({:.1}%)", same_cost, same_cost as f64 / n * 100.0);
    println!("  Guided better:   {} ({:.1}%)", guided_better, guided_better as f64 / n * 100.0);
    println!("  Baseline better: {} ({:.1}%)", baseline_better, baseline_better as f64 / n * 100.0);
    println!("  Avg cost diff:   {:.2} (positive = guided worse)", cost_diff_sum / n);

    // Efficiency comparison
    let baseline_epochs: f64 = baseline.iter().map(|r| r.epochs_used as f64).sum::<f64>();
    let guided_epochs: f64 = guided.iter().map(|r| r.epochs_used as f64).sum::<f64>();
    let baseline_rules: f64 = baseline.iter().map(|r| r.rules_applied as f64).sum::<f64>();
    let guided_rules: f64 = guided.iter().map(|r| r.rules_applied as f64).sum::<f64>();
    let baseline_time: f64 = baseline.iter().map(|r| r.duration_us as f64).sum::<f64>();
    let guided_time: f64 = guided.iter().map(|r| r.duration_us as f64).sum::<f64>();

    println!("\nEfficiency comparison:");
    println!("  Epochs ratio:    {:.2}x (guided/baseline)", guided_epochs / baseline_epochs);
    println!("  Rules ratio:     {:.2}x (guided/baseline)", guided_rules / baseline_rules);
    println!("  Time ratio:      {:.2}x (guided/baseline)", guided_time / baseline_time);

    // Summary
    println!("\n=== Summary ===");

    let no_regression = baseline_better as f64 / n < 0.10; // < 10% regression
    let efficiency_gain = guided_rules / baseline_rules < 0.90; // > 10% fewer rules

    if no_regression && efficiency_gain {
        println!("✓ PASS: Guide achieves similar costs with fewer rule applications");
    } else if no_regression {
        println!("~ PARTIAL: Guide achieves similar costs but no efficiency gain");
    } else {
        println!("✗ FAIL: Guide causes cost regression in > 10% of cases");
    }
}

/// Hard shader expressions where saturation is infeasible.
///
/// These are real expressions from the psychedelic shader and similar
/// production kernels. They're big enough that apply-all-rules saturation
/// explodes the e-graph, so the only way to optimize them is with a
/// guide that selectively applies rules.
fn hard_shader_exprs() -> Vec<(&'static str, ExprTree)> {
    let x = || ExprTree::var(0);
    let y = || ExprTree::var(1);
    let t = || ExprTree::var(2);
    let c = |v: f32| ExprTree::Leaf(Leaf::Const(v));

    vec![
        // Radial field: |x² + y² - 0.7|
        ("shader_radial", ExprTree::Op {
            op: &ops::Abs,
            children: vec![ExprTree::Op {
                op: &ops::Sub,
                children: vec![
                    ExprTree::add(ExprTree::mul(x(), x()), ExprTree::mul(y(), y())),
                    c(0.7),
                ],
            }],
        }),

        // Swirl: sin(x + t) * |x - y| * 0.2
        ("shader_swirl", ExprTree::mul(
            ExprTree::mul(
                ExprTree::Op { op: &ops::Sin, children: vec![ExprTree::add(x(), t())] },
                ExprTree::Op { op: &ops::Abs, children: vec![
                    ExprTree::Op { op: &ops::Sub, children: vec![x(), y()] },
                ] },
            ),
            c(0.2),
        )),

        // Soft clamp: exp(y) / (|exp(y)| + 1)
        ("shader_soft_clamp", {
            let exp_y = ExprTree::Op { op: &ops::Exp, children: vec![y()] };
            let abs_exp = ExprTree::Op { op: &ops::Abs, children: vec![exp_y.clone()] };
            ExprTree::Op {
                op: &ops::Div,
                children: vec![exp_y, ExprTree::add(abs_exp, c(1.0))],
            }
        }),

        // Full red channel: ((exp(y) * exp(-4*r²) / swirl) / (|...| + 1) + 1) * 0.5
        ("shader_red_channel", {
            let x_sq = ExprTree::mul(x(), x());
            let y_sq = ExprTree::mul(y(), y());
            let r_sq = ExprTree::add(x_sq, y_sq);
            let radial = ExprTree::Op {
                op: &ops::Exp,
                children: vec![ExprTree::mul(c(-4.0), r_sq)],
            };
            let y_factor = ExprTree::Op { op: &ops::Exp, children: vec![y()] };
            let sin_x = ExprTree::Op { op: &ops::Sin, children: vec![x()] };
            let swirl = ExprTree::add(ExprTree::mul(sin_x, c(0.2)), c(0.001));
            let numerator = ExprTree::mul(y_factor, radial);
            let raw = ExprTree::Op {
                op: &ops::Div, children: vec![numerator, swirl],
            };
            let abs_raw = ExprTree::Op { op: &ops::Abs, children: vec![raw.clone()] };
            let soft = ExprTree::Op {
                op: &ops::Div,
                children: vec![raw, ExprTree::add(abs_raw, c(1.0))],
            };
            ExprTree::mul(ExprTree::add(soft, c(1.0)), c(0.5))
        }),

        // 2D normalize: x / sqrt(x² + y²)
        ("shader_normalize", ExprTree::Op {
            op: &ops::Div,
            children: vec![
                x(),
                ExprTree::Op {
                    op: &ops::Sqrt,
                    children: vec![ExprTree::add(
                        ExprTree::mul(x(), x()),
                        ExprTree::mul(y(), y()),
                    )],
                },
            ],
        }),

        // SDF circle with smooth edge: smoothstep-like
        // 3t² - 2t³ where t = clamp((dist - r) / edge, 0, 1)
        // Simplified: 3*d*d - 2*d*d*d where d = sqrt(x²+y²) - 0.5
        ("shader_smooth_sdf", {
            let d = ExprTree::Op {
                op: &ops::Sub,
                children: vec![
                    ExprTree::Op {
                        op: &ops::Sqrt,
                        children: vec![ExprTree::add(
                            ExprTree::mul(x(), x()),
                            ExprTree::mul(y(), y()),
                        )],
                    },
                    c(0.5),
                ],
            };
            let d2 = ExprTree::mul(d.clone(), d.clone());
            let d3 = ExprTree::mul(d2.clone(), d.clone());
            ExprTree::Op {
                op: &ops::Sub,
                children: vec![
                    ExprTree::mul(c(3.0), d2),
                    ExprTree::mul(c(2.0), d3),
                ],
            }
        }),

        // Gaussian blur kernel weight: exp(-(dx² + dy²) / (2σ²))
        // with dx = x - cx, dy = y - cy, σ = 0.3
        ("shader_gaussian", {
            let dx = ExprTree::Op { op: &ops::Sub, children: vec![x(), c(0.5)] };
            let dy = ExprTree::Op { op: &ops::Sub, children: vec![y(), c(0.5)] };
            let dist_sq = ExprTree::add(
                ExprTree::mul(dx.clone(), dx),
                ExprTree::mul(dy.clone(), dy),
            );
            let neg_scaled = ExprTree::Op {
                op: &ops::Div,
                children: vec![
                    ExprTree::Op { op: &ops::Neg, children: vec![dist_sq] },
                    ExprTree::mul(c(2.0), ExprTree::mul(c(0.3), c(0.3))),
                ],
            };
            ExprTree::Op { op: &ops::Exp, children: vec![neg_scaled] }
        }),
    ]
}

/// Convert pixelflow Expr to ExprTree.
fn expr_to_tree(expr: &Expr) -> ExprTree {
    match expr {
        Expr::Var(v) => ExprTree::var(*v),
        Expr::Const(c) => ExprTree::constant(*c),
        Expr::Param(i) => panic!("Expr::Param({i}) in expr_to_tree — call substitute_params first"),
        Expr::Unary(op, a) => {
            let child = expr_to_tree(a);
            let op_ref = op_kind_to_static(*op);
            ExprTree::Op {
                op: op_ref,
                children: vec![child],
            }
        }
        Expr::Binary(op, a, b) => {
            let left = expr_to_tree(a);
            let right = expr_to_tree(b);
            let op_ref = op_kind_to_static(*op);
            ExprTree::Op {
                op: op_ref,
                children: vec![left, right],
            }
        }
        Expr::Ternary(op, a, b, c) => {
            let c1 = expr_to_tree(a);
            let c2 = expr_to_tree(b);
            let c3 = expr_to_tree(c);
            let op_ref = op_kind_to_static(*op);
            ExprTree::Op {
                op: op_ref,
                children: vec![c1, c2, c3],
            }
        }
        Expr::Nary(op, children) => {
            let child_trees: Vec<_> = children.iter().map(|c| expr_to_tree(c)).collect();
            let op_ref = op_kind_to_static(*op);
            ExprTree::Op {
                op: op_ref,
                children: child_trees,
            }
        }
    }
}

/// Convert OpKind to static Op reference.
fn op_kind_to_static(kind: OpKind) -> &'static dyn pixelflow_search::egraph::ops::Op {
    use pixelflow_search::egraph::ops;

    match kind {
        OpKind::Add => &ops::Add,
        OpKind::Sub => &ops::Sub,
        OpKind::Mul => &ops::Mul,
        OpKind::Div => &ops::Div,
        OpKind::Neg => &ops::Neg,
        OpKind::Recip => &ops::Recip,
        OpKind::Sqrt => &ops::Sqrt,
        OpKind::Rsqrt => &ops::Rsqrt,
        OpKind::Abs => &ops::Abs,
        OpKind::Min => &ops::Min,
        OpKind::Max => &ops::Max,
        OpKind::MulAdd => &ops::MulAdd,
        OpKind::Floor => &ops::Floor,
        OpKind::Ceil => &ops::Ceil,
        OpKind::Round => &ops::Round,
        OpKind::Fract => &ops::Fract,
        OpKind::Sin => &ops::Sin,
        OpKind::Cos => &ops::Cos,
        OpKind::Tan => &ops::Tan,
        OpKind::Asin => &ops::Asin,
        OpKind::Acos => &ops::Acos,
        OpKind::Atan => &ops::Atan,
        OpKind::Atan2 => &ops::Atan2,
        OpKind::Exp => &ops::Exp,
        OpKind::Exp2 => &ops::Exp2,
        OpKind::Ln => &ops::Ln,
        OpKind::Log2 => &ops::Log2,
        OpKind::Log10 => &ops::Log10,
        OpKind::Pow => &ops::Pow,
        OpKind::Hypot => &ops::Hypot,
        OpKind::Lt => &ops::Lt,
        OpKind::Le => &ops::Le,
        OpKind::Gt => &ops::Gt,
        OpKind::Ge => &ops::Ge,
        OpKind::Eq => &ops::Eq,
        OpKind::Ne => &ops::Ne,
        OpKind::Select => &ops::Select,
        OpKind::Clamp => &ops::Clamp,
        OpKind::Tuple => &ops::Tuple,
        OpKind::Var | OpKind::Const => panic!("Var/Const should not need op conversion"),
    }
}

/// Find workspace root by looking for Cargo.toml with [workspace].
fn find_workspace_root() -> PathBuf {
    let mut current = std::env::current_dir().expect("Failed to get current directory");
    loop {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.exists() {
            let contents = fs::read_to_string(&cargo_toml).unwrap_or_default();
            if contents.contains("[workspace]") {
                return current;
            }
        }
        if !current.pop() {
            return std::env::current_dir().expect("Failed to get current directory");
        }
    }
}
