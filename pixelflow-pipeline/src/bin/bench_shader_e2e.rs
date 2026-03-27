#![allow(dead_code, unused_variables, unused_imports, clippy::all)]
//! End-to-end benchmark: HCE vs Judge vs Guided on real shader expressions
//!
//! Two-step workflow:
//!   Step 1: cargo run --release -p pixelflow-pipeline --bin bench_shader_e2e
//!           → runs 3-lane extraction, prints search stats, appends to bench_corpus.jsonl
//!
//!   Step 2: cargo run --release -p pixelflow-pipeline --bin bench_jit_corpus --features training
//!           → JIT-compiles corpus expressions, benchmarks real SIMD execution
//!
//! Three lanes:
//!   Neural-DP: Full saturation (all rules) → Neural DP extract (full NNUE forward pass)
//!   Judge:     (shares Neural-DP's saturated e-graph) → Beam extract (ExprNnue)
//!   Guided:    Guide-filtered saturation (new e-graph) → Beam extract (ExprNnue)

use pixelflow_search::egraph::{EGraph, EClassId, ExprTree, Leaf, ops, extract_beam, extract_neural, codegen};
use pixelflow_search::math::all_math_rules;
use pixelflow_search::nnue::ExprNnue;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

const JUDGE_WEIGHTS: &str = "pixelflow-pipeline/data/judge.bin";
const MAX_EPOCHS: usize = 30;
const GUIDED_NODE_BUDGET: usize = 10_000;
/// Node budget for full (unguided) saturation — prevents e-graph explosion.
const FULL_SAT_NODE_BUDGET: usize = 2_000;
/// Max e-graph nodes for beam search; above this fall back to neural DP extraction.
/// Beam search is O(beam × nodes_per_class × classes) — intractable above ~200 nodes.
const BEAM_NODE_LIMIT: usize = 200;
const BEAM_WIDTH: usize = 32;
const BENCH_OUTPUT: &str = "pixelflow-pipeline/data/bench_corpus.jsonl";

/// Search statistics for one lane (no kernel timing — that's step 2).
struct LaneResult {
    search_ms: f64,
    egraph_nodes: usize,
    epochs: usize,
    extracted_nodes: usize,
    kernel_body: String,
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  STEP 1: Extract optimized kernels (HCE vs Judge vs Guided)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let judge = ExprNnue::load(Path::new(JUDGE_WEIGHTS))
        .unwrap_or_else(|e| panic!("Failed to load Judge from {}: {}", JUDGE_WEIGHTS, e));
    println!("  Judge loaded from {}", JUDGE_WEIGHTS);

    let expressions: Vec<(&str, &str, ExprTree)> = vec![
        ("radial",    "Radial Field: |x² + y² - 0.7|",               build_radial_field()),
        ("softclamp", "Soft Clamp: x / (|x| + 1)",                   build_soft_clamp()),
        ("distsq",    "Distance²: (x-0.5)² + (y-0.5)²",             build_distance_sq()),
        ("expdecay",  "Exp Decay: exp(-4*(x²+y²))",                  build_exp_decay()),
        ("normalize", "Normalize: x / sqrt(x²+y²)",                  build_normalize()),
        ("channel",   "Channel: exp(y) * exp(-r²) / (|...| + 1)",    build_channel()),
        ("psychred",  "Psychedelic Red (40+ nodes)",                  build_psychedelic_red()),
    ];

    // Collect extracted kernels for benchmark generation:
    // (name, ExprTree) pairs — original + 3 lanes per expression
    let mut bench_variants: Vec<(String, ExprTree)> = Vec::new();

    let mut total_hce_search = 0.0;
    let mut total_guided_search = 0.0;
    let mut total_hce_nodes = 0usize;
    let mut total_guided_nodes = 0usize;

    for (slug, label, expr) in &expressions {
        println!("━━━ {} ━━━\n", label);

        // Add original (unoptimized) expression as baseline
        bench_variants.push((format!("{}_original", slug), expr.clone()));

        let (hce_res, judge_res, guided_res, hce_tree, judge_tree, guided_tree) =
            run_extraction(expr, &judge);

        // Collect extracted trees
        bench_variants.push((format!("{}_nnue_dp", slug), hce_tree));
        bench_variants.push((format!("{}_judge", slug), judge_tree));
        bench_variants.push((format!("{}_guided", slug), guided_tree));

        // Print search stats
        println!(
            "  {:18} {:>14} {:>14} {:>14}",
            "", "NNUE-DP", "Judge", "Guided"
        );
        println!(
            "  {:18} {:>12.1} ms {:>12} {:>12.1} ms",
            "Search:", hce_res.search_ms, "(shared)", guided_res.search_ms
        );
        println!(
            "  {:18} {:>10} nodes {:>12} {:>10} nodes",
            "E-graph:", hce_res.egraph_nodes, "(shared)", guided_res.egraph_nodes
        );
        println!(
            "  {:18} {:>14} {:>12} {:>14}",
            "Epochs:", hce_res.epochs, "(shared)", guided_res.epochs
        );
        println!(
            "  {:18} {:>10} nodes {:>8} nodes {:>8} nodes",
            "Extracted:",
            hce_res.extracted_nodes,
            judge_res.extracted_nodes,
            guided_res.extracted_nodes
        );

        // Show the actual extracted expressions
        println!("\n  NNUE-DP: {}", truncate(&hce_res.kernel_body, 72));
        println!("  Judge:   {}", truncate(&judge_res.kernel_body, 72));
        println!("  Guided:  {}", truncate(&guided_res.kernel_body, 72));

        // Speedup summary
        let search_speedup = if guided_res.search_ms > 0.0 {
            hce_res.search_ms / guided_res.search_ms
        } else {
            f64::INFINITY
        };
        println!("\n  Guide: {:.1}x search speedup, {} vs {} e-graph nodes\n",
            search_speedup, guided_res.egraph_nodes, hce_res.egraph_nodes);

        total_hce_search += hce_res.search_ms;
        total_guided_search += guided_res.search_ms;
        total_hce_nodes += hce_res.egraph_nodes;
        total_guided_nodes += guided_res.egraph_nodes;
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SEARCH SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    let avg_speedup = if total_guided_search > 0.0 {
        total_hce_search / total_guided_search
    } else {
        f64::INFINITY
    };
    let avg_reduction = if total_hce_nodes > 0 {
        100.0 * (1.0 - total_guided_nodes as f64 / total_hce_nodes as f64)
    } else {
        0.0
    };
    println!("  Avg search speedup:     {:.1}x", avg_speedup);
    println!("  Avg e-graph reduction:  {:.0}% fewer nodes", avg_reduction);
    println!("  Extracted {} kernel variants ({} expressions × 4 lanes)",
        bench_variants.len(), expressions.len());

    // Generate corpus JSONL (appended — preserves previous entries).
    let corpus_jsonl = codegen::generate_corpus_jsonl(&bench_variants);
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(BENCH_OUTPUT)
        .unwrap_or_else(|e| panic!("Failed to open {}: {}", BENCH_OUTPUT, e));
    f.write_all(corpus_jsonl.as_bytes())
        .unwrap_or_else(|e| panic!("Failed to write {}: {}", BENCH_OUTPUT, e));

    println!("\n  Appended {} entries → {}", bench_variants.len(), BENCH_OUTPUT);
    println!("\n  STEP 2: cargo run --release -p pixelflow-pipeline --bin bench_jit_corpus --features training");
    println!("═══════════════════════════════════════════════════════════════");
}

/// Extract using NNUE: beam search for small e-graphs, neural DP for large ones.
/// Beam search gives higher quality (evaluates full trees from root) but is O(exponential).
/// Neural DP builds full subtrees per e-class with `predict_log_cost()` — polynomial, scales fine.
fn judge_extract(egraph: &EGraph, root: EClassId, judge: &ExprNnue) -> (ExprTree, f32) {
    let nodes = egraph.node_count();
    if nodes <= BEAM_NODE_LIMIT {
        extract_beam(egraph, root, judge, BEAM_WIDTH)
    } else {
        extract_neural(egraph, root, judge)
    }
}

/// Run 3-lane extraction, return stats + extracted trees.
fn run_extraction(
    expr: &ExprTree,
    judge: &ExprNnue,
) -> (LaneResult, LaneResult, LaneResult, ExprTree, ExprTree, ExprTree) {
    // ── Lane 1 + 2: Full saturation (shared by HCE and Judge) ──────────
    let mut egraph = EGraph::with_rules(all_math_rules());
    let root = egraph.add_expr(expr);

    let sat_start = Instant::now();
    let num_rules = egraph.num_rules();
    let mut sat_epochs = 0;
    'sat: for epoch in 0..MAX_EPOCHS {
        let mut epoch_changes = 0;
        for rule_idx in 0..num_rules {
            if egraph.node_count() > FULL_SAT_NODE_BUDGET {
                sat_epochs = epoch;
                break 'sat;
            }
            epoch_changes += egraph.apply_rule_at_index(rule_idx).changes;
        }
        sat_epochs = epoch + 1;
        if epoch_changes == 0 {
            break;
        }
    }
    let sat_ms = sat_start.elapsed().as_secs_f64() * 1000.0;
    let sat_nodes = egraph.node_count();

    // Lane 1: Neural-DP — bottom-up DP extract with full NNUE forward pass
    let (hce_tree, _) = extract_neural(&egraph, root, judge);
    let hce_body = codegen::expr_tree_to_kernel_body(&hce_tree);
    let hce_result = LaneResult {
        search_ms: sat_ms,
        egraph_nodes: sat_nodes,
        epochs: sat_epochs,
        extracted_nodes: hce_tree.node_count(),
        kernel_body: hce_body,
    };

    // Lane 2: Judge — neural extract from same saturated e-graph
    let (judge_tree, _) = judge_extract(&egraph, root, judge);
    let judge_body = codegen::expr_tree_to_kernel_body(&judge_tree);
    let judge_result = LaneResult {
        search_ms: sat_ms,
        egraph_nodes: sat_nodes,
        epochs: sat_epochs,
        extracted_nodes: judge_tree.node_count(),
        kernel_body: judge_body,
    };

    // ── Lane 3: Budget-limited saturation → beam extract ────────────────
    let (guided_egraph, guided_root, guided_epochs, guided_ms) =
        run_budget_limited_saturation(expr, MAX_EPOCHS, GUIDED_NODE_BUDGET);
    let guided_nodes = guided_egraph.node_count();

    let (guided_tree, _) = judge_extract(&guided_egraph, guided_root, judge);
    let guided_body = codegen::expr_tree_to_kernel_body(&guided_tree);
    let guided_result = LaneResult {
        search_ms: guided_ms,
        egraph_nodes: guided_nodes,
        epochs: guided_epochs,
        extracted_nodes: guided_tree.node_count(),
        kernel_body: guided_body,
    };

    (hce_result, judge_result, guided_result, hce_tree, judge_tree, guided_tree)
}

/// Run budget-limited e-graph saturation (uniform guide — all rules applied).
///
/// This is a node-budget-limited baseline: applies all rules every epoch,
/// stopping when the node budget is exhausted or saturation is reached.
fn run_budget_limited_saturation(
    expr: &ExprTree,
    max_epochs: usize,
    node_budget: usize,
) -> (EGraph, EClassId, usize, f64) {
    let start = Instant::now();

    let mut egraph = EGraph::with_rules(all_math_rules());
    let root = egraph.add_expr(expr);

    let num_rules = egraph.num_rules();
    let mut epochs_used = 0;

    for epoch in 0..max_epochs {
        if egraph.node_count() > node_budget {
            break;
        }

        let mut total_changes = 0;
        for rule_idx in 0..num_rules {
            if egraph.node_count() > node_budget {
                break;
            }
            total_changes += egraph.apply_rule_at_index(rule_idx).changes;
        }

        epochs_used = epoch + 1;
        if total_changes == 0 {
            break;
        }
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    (egraph, root, epochs_used, elapsed_ms)
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}

// ============================================================================
// Expression builders
// ============================================================================

fn build_radial_field() -> ExprTree {
    let x = ExprTree::var(0);
    let y = ExprTree::var(1);
    let x_sq = ExprTree::mul(x.clone(), x);
    let y_sq = ExprTree::mul(y.clone(), y);
    let r_sq = ExprTree::add(x_sq, y_sq);
    let shifted = ExprTree::Op { op: &ops::Sub, children: vec![r_sq, ExprTree::Leaf(Leaf::Const(0.7))] };
    ExprTree::Op { op: &ops::Abs, children: vec![shifted] }
}

fn build_soft_clamp() -> ExprTree {
    let x = ExprTree::var(0);
    let abs_x = ExprTree::Op { op: &ops::Abs, children: vec![x.clone()] };
    let denom = ExprTree::add(abs_x, ExprTree::Leaf(Leaf::Const(1.0)));
    ExprTree::Op { op: &ops::Div, children: vec![x, denom] }
}

fn build_distance_sq() -> ExprTree {
    let x = ExprTree::var(0);
    let y = ExprTree::var(1);
    let half = ExprTree::Leaf(Leaf::Const(0.5));
    let dx = ExprTree::Op { op: &ops::Sub, children: vec![x.clone(), half.clone()] };
    let dy = ExprTree::Op { op: &ops::Sub, children: vec![y, half] };
    ExprTree::add(ExprTree::mul(dx.clone(), dx), ExprTree::mul(dy.clone(), dy))
}

fn build_exp_decay() -> ExprTree {
    let x = ExprTree::var(0);
    let y = ExprTree::var(1);
    let x_sq = ExprTree::mul(x.clone(), x);
    let y_sq = ExprTree::mul(y.clone(), y);
    let r_sq = ExprTree::add(x_sq, y_sq);
    let neg_4_r_sq = ExprTree::mul(ExprTree::Leaf(Leaf::Const(-4.0)), r_sq);
    ExprTree::Op { op: &ops::Exp, children: vec![neg_4_r_sq] }
}

fn build_normalize() -> ExprTree {
    let x = ExprTree::var(0);
    let y = ExprTree::var(1);
    let x_sq = ExprTree::mul(x.clone(), x.clone());
    let y_sq = ExprTree::mul(y.clone(), y);
    let len_sq = ExprTree::add(x_sq, y_sq);
    let len = ExprTree::Op { op: &ops::Sqrt, children: vec![len_sq] };
    ExprTree::Op { op: &ops::Div, children: vec![x, len] }
}

fn build_channel() -> ExprTree {
    let x = ExprTree::var(0);
    let y = ExprTree::var(1);
    let x_sq = ExprTree::mul(x.clone(), x);
    let y_sq = ExprTree::mul(y.clone(), y.clone());
    let r_sq = ExprTree::add(x_sq, y_sq);
    let neg_4_r_sq = ExprTree::mul(ExprTree::Leaf(Leaf::Const(-4.0)), r_sq);
    let radial = ExprTree::Op { op: &ops::Exp, children: vec![neg_4_r_sq] };
    let ey = ExprTree::Op { op: &ops::Exp, children: vec![y] };
    let raw = ExprTree::mul(ey, radial);
    let abs_raw = ExprTree::Op { op: &ops::Abs, children: vec![raw.clone()] };
    let denom = ExprTree::add(abs_raw, ExprTree::Leaf(Leaf::Const(1.0)));
    ExprTree::Op { op: &ops::Div, children: vec![raw, denom] }
}

/// Red channel from the psychedelic shader.
/// Uses x=var(0), y=var(1), time=var(2).
/// ~40 internal ops — first realistic-sized expression in the benchmark.
fn build_psychedelic_red() -> ExprTree {
    let x = ExprTree::var(0);
    let y = ExprTree::var(1);
    let time = ExprTree::var(2);
    let c = |v: f32| ExprTree::Leaf(Leaf::Const(v));

    // Radial field
    let r_sq = ExprTree::add(
        ExprTree::mul(x.clone(), x.clone()),
        ExprTree::mul(y.clone(), y.clone()),
    );
    let radial = ExprTree::Op {
        op: &ops::Abs,
        children: vec![ExprTree::Op {
            op: &ops::Sub,
            children: vec![r_sq, c(0.7)],
        }],
    };

    // Swirl scale
    let swirl_scale = ExprTree::mul(
        ExprTree::Op {
            op: &ops::Sub,
            children: vec![c(1.0), radial.clone()],
        },
        c(5.0),
    );
    let vx = ExprTree::mul(x.clone(), swirl_scale.clone());
    let vy = ExprTree::mul(y.clone(), swirl_scale);

    // Time-based values
    let phase = ExprTree::mul(time.clone(), c(0.5));
    let sin_w03 = ExprTree::Op {
        op: &ops::Sin,
        children: vec![ExprTree::mul(time.clone(), c(0.3))],
    };
    let sin_w20 = ExprTree::Op {
        op: &ops::Sin,
        children: vec![ExprTree::mul(time, c(2.0))],
    };

    // Swirl computation
    let vx_plus_phase = ExprTree::add(vx, phase.clone());
    let sin_vxp = ExprTree::Op {
        op: &ops::Sin,
        children: vec![vx_plus_phase.clone()],
    };
    let sin_vxp_plus_1 = ExprTree::add(sin_vxp, c(1.0));
    let vy_plus_phase07 = ExprTree::add(vy, ExprTree::mul(phase, c(0.7)));
    let diff = ExprTree::Op {
        op: &ops::Sub,
        children: vec![vx_plus_phase, vy_plus_phase07],
    };
    let abs_diff = ExprTree::Op {
        op: &ops::Abs,
        children: vec![diff],
    };
    let swirl = ExprTree::add(
        ExprTree::mul(ExprTree::mul(sin_vxp_plus_1, abs_diff), c(0.2)),
        c(0.001),
    );

    // Radial falloff with pulsing
    let pulse = ExprTree::add(c(1.0), ExprTree::mul(sin_w20, c(0.1)));
    let radial_factor = ExprTree::Op {
        op: &ops::Exp,
        children: vec![ExprTree::mul(ExprTree::mul(radial, c(-4.0)), pulse)],
    };

    // Red channel
    let y_factor_r = ExprTree::Op {
        op: &ops::Exp,
        children: vec![ExprTree::add(y, ExprTree::mul(sin_w03, c(0.2)))],
    };
    let raw_r = ExprTree::Op {
        op: &ops::Div,
        children: vec![ExprTree::mul(y_factor_r, radial_factor), swirl],
    };
    let soft_r = ExprTree::Op {
        op: &ops::Div,
        children: vec![
            raw_r.clone(),
            ExprTree::add(
                ExprTree::Op { op: &ops::Abs, children: vec![raw_r] },
                c(1.0),
            ),
        ],
    };

    ExprTree::mul(ExprTree::add(soft_r, c(1.0)), c(0.5))
}
