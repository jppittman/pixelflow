//! # Self-Play Trajectory Generator
//!
//! Runs hill-climbing trajectories with the current mask weights (self-play,
//! no perturbation) and records per-step payload for the Critic.
//!
//! This is the "GENERATE" phase of the unified training loop. Unlike
//! `collect_guide_data` which uses random Gaussian perturbation of mask
//! weights, this module uses the model weights as-is (true self-play).
//!
//! ## Key differences from `collect_guide_data`
//!
//! - No perturbation -- uses model weights directly
//! - Records `accumulator_state` + `rule_embedding` per step (for Critic)
//! - JIT benchmarks final expression for ground-truth terminal cost
//! - Returns [`Trajectory`] struct (defined in `unified.rs`) not `TrajectorySample`

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use pixelflow_search::egraph::{
    all_rules, expr_tree_to_nnue, extract_neural, predict_tree_cost, EGraph, ENode, Rewrite,
};
use pixelflow_search::nnue::{
    BwdGenConfig, BwdGenerator, EdgeAccumulator, Expr, ExprNnue, GraphAccumulator, OpKind,
    RuleTemplates, EMBED_DIM, GRAPH_INPUT_DIM,
};
use pixelflow_search::nnue::factored::MAX_ARITY;
use pixelflow_search::nnue::factored::INPUT_DIM;

use crate::jit_bench::benchmark_jit;
use crate::training::factored::parse_kernel_code;
use crate::training::unified::{Trajectory, TrajectoryAdvantages, TrajectoryStep};

// ============================================================================
// Helpers
// ============================================================================

/// Sigmoid activation for mask score -> probability conversion.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Convert an [`EdgeAccumulator`] to a flat `INPUT_DIM`-element `Vec<f32>`.
///
/// Layout: `[values (4*K=128 floats), edge_count, node_count]` = 130 floats total.
pub fn acc_to_vec(acc: &EdgeAccumulator) -> Vec<f32> {
    let mut v = Vec::with_capacity(INPUT_DIM);
    v.extend_from_slice(&acc.values);
    v.push(acc.edge_count as f32);
    v.push(acc.node_count as f32);
    v
}

/// Convert a [`GraphAccumulator`] to a flat `GRAPH_INPUT_DIM`-element `Vec<f32>`.
///
/// Layout: `[values (3*K=96 floats), edge_count, node_count]` = 98 floats total.
pub fn gacc_to_vec(gacc: &GraphAccumulator) -> Vec<f32> {
    let mut v = Vec::with_capacity(GRAPH_INPUT_DIM);
    v.extend_from_slice(&gacc.values);
    v.push(gacc.edge_count as f32);
    v.push(gacc.node_count as f32);
    v
}

/// Build a [`GraphAccumulator`] from the current e-graph state.
///
/// O(V+E) traversal. Walks all canonical e-classes, resolves child ops
/// via union-find, and accumulates VSA-encoded edges. Rebuilt from scratch
/// each epoch because union-find merges change canonical representatives,
/// making incremental updates unsound.
fn build_graph_acc(egraph: &EGraph, emb: &pixelflow_search::nnue::OpEmbeddings) -> GraphAccumulator {
    let mut gacc = GraphAccumulator::new();
    for class_id in egraph.class_ids() {
        for node in egraph.nodes(class_id) {
            match node {
                ENode::Var(_) | ENode::Const(_) => gacc.add_leaf(),
                ENode::Op { op, children } => {
                    let child_ops: Vec<OpKind> = children
                        .iter()
                        .map(|c| egraph.canonical_op(*c))
                        .collect();
                    gacc.add_op_node(emb, op.kind(), &child_ops);
                }
            }
        }
    }
    gacc
}

/// Build [`RuleTemplates`] from rule definitions.
///
/// Each rule provides LHS/RHS expression templates via the [`Rewrite`] trait.
pub fn build_rule_templates(rules: &[Box<dyn Rewrite>]) -> RuleTemplates {
    let mut templates = RuleTemplates::with_capacity(rules.len());

    for (idx, rule) in rules.iter().enumerate() {
        if let (Some(lhs), Some(rhs)) = (rule.lhs_template(), rule.rhs_template()) {
            templates.set(idx, lhs, rhs);
        }
        // Rules without templates get zero embedding (handled by RuleTemplates)
    }

    templates
}

/// Convert an NNUE [`Expr`] to an [`ExprTree`] for e-graph insertion.
///
/// Uses iterative traversal with explicit stack to avoid thread stack overflow
/// on deeply nested expression trees.
fn expr_to_tree(
    expr: &Expr,
) -> pixelflow_search::egraph::ExprTree {
    use pixelflow_search::egraph::ExprTree;

    enum Task<'a> {
        Visit(&'a Expr),
        Complete {
            op_ref: &'static dyn pixelflow_search::egraph::ops::Op,
            arity: usize,
        },
    }

    let mut stack: Vec<Task<'_>> = vec![Task::Visit(expr)];
    let mut result_stack: Vec<ExprTree> = Vec::new();

    while let Some(task) = stack.pop() {
        match task {
            Task::Visit(node) => match node {
                Expr::Var(v) => result_stack.push(ExprTree::var(*v)),
                Expr::Const(c) => result_stack.push(ExprTree::constant(*c)),
                Expr::Param(i) => panic!("Expr::Param({i}) in expr_to_tree -- call substitute_params first"),
                Expr::Unary(op, a) => {
                    let op_ref = op_kind_to_static(*op);
                    stack.push(Task::Complete { op_ref, arity: 1 });
                    stack.push(Task::Visit(a));
                }
                Expr::Binary(op, a, b) => {
                    let op_ref = op_kind_to_static(*op);
                    stack.push(Task::Complete { op_ref, arity: 2 });
                    stack.push(Task::Visit(b));
                    stack.push(Task::Visit(a));
                }
                Expr::Ternary(op, a, b, c) => {
                    let op_ref = op_kind_to_static(*op);
                    stack.push(Task::Complete { op_ref, arity: 3 });
                    stack.push(Task::Visit(c));
                    stack.push(Task::Visit(b));
                    stack.push(Task::Visit(a));
                }
                Expr::Nary(op, children) => {
                    let op_ref = op_kind_to_static(*op);
                    stack.push(Task::Complete { op_ref, arity: children.len() });
                    for child in children.iter().rev() {
                        stack.push(Task::Visit(child));
                    }
                }
            },
            Task::Complete { op_ref, arity } => {
                let start = result_stack.len().saturating_sub(arity);
                let child_trees: Vec<ExprTree> = result_stack.drain(start..).collect();
                result_stack.push(ExprTree::Op {
                    op: op_ref,
                    children: child_trees,
                });
            }
        }
    }

    result_stack.pop().unwrap_or_else(|| panic!("expr_to_tree: empty result stack"))
}

/// Convert [`OpKind`] to a static [`Op`] reference for e-graph construction.
fn op_kind_to_static(
    kind: pixelflow_search::nnue::OpKind,
) -> &'static dyn pixelflow_search::egraph::ops::Op {
    use pixelflow_search::egraph::ops;
    use pixelflow_search::nnue::OpKind;

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

// ============================================================================
// Edge collection for embedding gradients
// ============================================================================

/// Collect edges from an expression tree for embedding gradient flow.
///
/// Mirrors the traversal in `EdgeAccumulator::from_expr_dedup` but only records
/// `(parent_op_index, child_op_index, effective_depth)` tuples. Uses dedup via
/// structural hashing to match what the accumulator actually sees.
pub fn collect_edges_dedup(expr: &Expr) -> Vec<(u8, u8, u16)> {
    use std::collections::BTreeSet;

    let mut edges = Vec::new();
    let mut seen = BTreeSet::<u64>::new();
    let mut stack: Vec<(&Expr, u32)> = vec![(expr, 0)];

    while let Some((node, d)) = stack.pop() {
        let h = pixelflow_search::nnue::factored::structural_hash(node);
        if !seen.insert(h) {
            continue;
        }

        let parent_op = node.op_type();

        match node {
            Expr::Var(_) | Expr::Const(_) => {}
            Expr::Param(i) => panic!(
                "Expr::Param({i}) in collect_edges_dedup — call substitute_params first"
            ),
            Expr::Unary(_, child) => {
                let eff_depth = d * MAX_ARITY as u32;
                edges.push((parent_op.index() as u8, child.op_type().index() as u8, eff_depth as u16));
                stack.push((child, d + 1));
            }
            Expr::Binary(_, left, right) => {
                edges.push((parent_op.index() as u8, left.op_type().index() as u8, (d * MAX_ARITY as u32) as u16));
                edges.push((parent_op.index() as u8, right.op_type().index() as u8, (d * MAX_ARITY as u32 + 1) as u16));
                stack.push((right, d + 1));
                stack.push((left, d + 1));
            }
            Expr::Ternary(_, a, b, c) => {
                edges.push((parent_op.index() as u8, a.op_type().index() as u8, (d * MAX_ARITY as u32) as u16));
                edges.push((parent_op.index() as u8, b.op_type().index() as u8, (d * MAX_ARITY as u32 + 1) as u16));
                edges.push((parent_op.index() as u8, c.op_type().index() as u8, (d * MAX_ARITY as u32 + 2) as u16));
                stack.push((c, d + 1));
                stack.push((b, d + 1));
                stack.push((a, d + 1));
            }
            Expr::Nary(_, children) => {
                for (idx, child) in children.iter().enumerate() {
                    let eff_depth = d * MAX_ARITY as u32 + (idx.min(MAX_ARITY - 1)) as u32;
                    edges.push((parent_op.index() as u8, child.op_type().index() as u8, eff_depth as u16));
                }
                for child in children.iter().rev() {
                    stack.push((child, d + 1));
                }
            }
        }
    }

    edges
}

// ============================================================================
// Core trajectory runner
// ============================================================================

/// Run a single self-play trajectory through e-graph hill-climbing.
///
/// Records per-step Critic payload: accumulator state, rule embedding,
/// action probability, and match status.
///
/// Returns `None` if the final expression cannot be JIT-benchmarked.
///
/// # Algorithm
///
/// 1. Insert seed expression into an e-graph with all rewrite rules.
/// 2. For each epoch (up to `max_epochs`):
///    a. Extract current best expression via NNUE-guided neural extraction.
///    b. Build an [`EdgeAccumulator`] and score all rules in one forward pass.
///    c. For each approved rule (sigmoid(score) > threshold), record a
///       [`TrajectoryStep`] BEFORE applying, then apply and update `matched`.
///    d. Update best cost if improved.
/// 3. JIT-benchmark the final expression for ground-truth terminal cost.
pub fn run_self_play_trajectory(
    seed_expr: &Expr,
    seed_name: &str,
    model: &ExprNnue,
    rule_embeds: &[[f32; EMBED_DIM]],
    rules: &[Box<dyn Rewrite>],
    threshold: f32,
    max_epochs: usize,
    trajectory_id: String,
) -> Option<Trajectory> {
    // Hard wall-clock deadline per trajectory: safety net against runaway extraction.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);

    // 1. Convert seed expression to e-graph
    let expr_tree = expr_to_tree(seed_expr);
    let mut egraph = EGraph::with_rules(all_rules());
    let root = egraph.add_expr(&expr_tree);

    // 2. Score the initial expression
    let (initial_tree, _initial_extract_cost) = extract_neural(&egraph, root, model);
    let initial_cost = predict_tree_cost(&initial_tree, model);

    // 2b. JIT benchmark the initial expression for ground-truth initial cost
    let initial_expr_nnue = expr_tree_to_nnue(&initial_tree);
    if initial_expr_nnue.has_degenerate() {
        eprintln!("Skipping degenerate seed expression in {trajectory_id} (seed={seed_name})");
        return None;
    }
    let initial_bench = match benchmark_jit(&initial_expr_nnue) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "JIT bench failed for initial expr in trajectory {trajectory_id} (seed={seed_name}): {e}"
            );
            return None;
        }
    };
    let initial_cost_ns = initial_bench.ns;

    // 3. Track best
    let mut best_cost = initial_cost;
    let mut steps: Vec<TrajectoryStep> = Vec::new();

    let num_rules = rules.len();

    // Randomize resource constraints per trajectory for training variety.
    let traj_hash: usize = trajectory_id.bytes().fold(0usize, |h, b| h.wrapping_mul(31).wrapping_add(b as usize));
    let node_budget = 500 + (traj_hash % 4501);          // [500, 5000]
    let epoch_budget = 5 + (traj_hash.wrapping_mul(7) % 26); // [5, 30]
    let effective_epochs = max_epochs.min(epoch_budget);

    // 4. Epoch loop
    //
    // KEY DESIGN: Score rules from the e-graph state (GraphAccumulator), not
    // from a single extracted expression. After rules fire and create new
    // e-nodes/equivalences, rebuild the GraphAccumulator and re-score to
    // approve newly-passing rules. The EdgeAccumulator is still built from
    // the initial expression for the value head (TrajectoryStep.accumulator_state).
    //
    // No per-epoch extraction or JIT. The only JIT benchmark is the final one.
    // The transformer critic assigns per-step credit from the single terminal reward.

    // Build EdgeAccumulator from initial expression (for value head training)
    let acc = EdgeAccumulator::from_expr_dedup(&initial_expr_nnue, &model.embeddings);
    let acc_vec = acc_to_vec(&acc);
    let edges = collect_edges_dedup(&initial_expr_nnue);
    let hidden = model.forward_shared(&acc);
    let expr_embed = model.compute_expr_embed(&hidden);
    let expr_embed_vec: Vec<f32> = expr_embed.to_vec();

    // Build GraphAccumulator from initial e-graph state (for mask scoring)
    let mut gacc = build_graph_acc(&egraph, &model.embeddings);
    let mut gacc_vec = gacc_to_vec(&gacc);

    // Score all rules via graph pathway
    let scores = model.mask_score_all_rules_graph(&gacc, rule_embeds);

    // Determine which rules are approved
    let mut approved_rules: Vec<(usize, f32)> = Vec::new();
    for r in 0..num_rules {
        let score = if r < scores.len() { scores[r] } else { 0.0 };
        let prob = sigmoid(score);
        if prob > threshold {
            approved_rules.push((r, prob));
        }
    }

    // Record one step per approved rule (decisions made, not yet applied)
    for &(r, prob) in &approved_rules {
        steps.push(TrajectoryStep {
            accumulator_state: acc_vec.clone(),
            expression_embedding: expr_embed_vec.clone(),
            rule_embedding: rule_embeds.get(r).map_or_else(
                || vec![0.0; EMBED_DIM],
                |e| e.to_vec(),
            ),
            budget_remaining: (node_budget as i32) - (egraph.node_count() as i32),
            epochs_remaining: effective_epochs as i32,
            action_probability: prob,
            matched: false, // Updated during saturation
            jit_cost_ns: f64::NAN, // Backfilled with final cost
            edges: edges.clone(),
            graph_accumulator_state: gacc_vec.clone(),
        });
    }

    // Saturate: apply approved rules, rebuild graph accumulator, re-score
    for epoch in 0..effective_epochs {
        if std::time::Instant::now() > deadline {
            panic!(
                "Trajectory {trajectory_id} hit 30s wall-clock deadline at epoch {epoch} \
                 (node_budget={node_budget}, egraph_nodes={}, epoch_budget={epoch_budget})",
                egraph.node_count()
            );
        }

        if egraph.node_count() > node_budget {
            break;
        }

        let mut any_changed = false;
        for (step_idx, &(r, _)) in approved_rules.iter().enumerate() {
            let result = egraph.apply_rule_at_index(r);
            if result.changes > 0 {
                steps[step_idx].matched = true;
                any_changed = true;
            }
        }

        // Fixed point: no rule produced new unions this epoch
        if !any_changed {
            break;
        }

        // Rebuild graph accumulator from post-union-find state
        gacc = build_graph_acc(&egraph, &model.embeddings);
        gacc_vec = gacc_to_vec(&gacc);

        // Re-score with accurate graph state
        let new_scores = model.mask_score_all_rules_graph(&gacc, rule_embeds);

        // Approve newly-passing rules, record new steps with updated gacc
        let epochs_remaining = (effective_epochs as i32) - (epoch as i32) - 1;
        for r in 0..num_rules {
            let score = if r < new_scores.len() { new_scores[r] } else { 0.0 };
            let prob = sigmoid(score);
            if prob > threshold && !approved_rules.iter().any(|(idx, _)| *idx == r) {
                approved_rules.push((r, prob));
                steps.push(TrajectoryStep {
                    accumulator_state: acc_vec.clone(),
                    expression_embedding: expr_embed_vec.clone(),
                    rule_embedding: rule_embeds.get(r).map_or_else(
                        || vec![0.0; EMBED_DIM],
                        |e| e.to_vec(),
                    ),
                    budget_remaining: (node_budget as i32) - (egraph.node_count() as i32),
                    epochs_remaining,
                    action_probability: prob,
                    matched: false,
                    jit_cost_ns: f64::NAN,
                    edges: edges.clone(),
                    graph_accumulator_state: gacc_vec.clone(),
                });
            }
        }
    }

    // 5. JIT benchmark the final expression for terminal cost
    let (final_tree, _) = extract_neural(&egraph, root, model);
    let final_expr = expr_tree_to_nnue(&final_tree);
    if final_expr.has_degenerate() {
        eprintln!("Rewrite produced degenerate expression in {trajectory_id} (seed={seed_name})");
        return None;
    }
    let final_bench = match benchmark_jit(&final_expr) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "JIT bench failed for trajectory {trajectory_id} (seed={seed_name}): {e}"
            );
            return None;
        }
    };
    let final_cost_ns = final_bench.ns;

    // Backfill all steps with the final cost. The transformer critic will
    // assign per-step advantages from this single terminal reward.
    for step in &mut steps {
        step.jit_cost_ns = final_cost_ns;
    }

    // Correctness check: rewrites must preserve semantics.
    // Compare all SIMD lanes at the test point.
    const EQUIV_EPSILON: f32 = 1e-3;
    if let Err(max_diff) = initial_bench.check_equivalence(&final_bench, EQUIV_EPSILON) {
        eprintln!(
            "REWRITE BUG: trajectory {trajectory_id} (seed={seed_name})\n\
             \x20 inputs: x=0.5 y=0.7 z=1.3 w=-0.2\n\
             \x20 initial output: {:?}\n\
             \x20 final   output: {:?}\n\
             \x20 max_diff={max_diff:.6} epsilon={EQUIV_EPSILON}\n\
             \x20 initial expr: {initial_expr_nnue}\n\
             \x20 final   expr: {final_expr}\n\
             \x20 steps: {} ({} matched)",
            initial_bench.output, final_bench.output,
            steps.len(), steps.iter().filter(|s| s.matched).count(),
        );
        return None;
    }

    // 6. Log the rewrite for visibility
    let speedup = if final_cost_ns > 0.0 { initial_cost_ns / final_cost_ns } else { 0.0 };
    let matched = steps.iter().filter(|s| s.matched).count();
    eprintln!(
        "[REWRITE] {seed_name}: {speedup:.2}x ({initial_cost_ns:.1}ns -> {final_cost_ns:.1}ns) \
         [{matched}/{} steps]\n\
         \x20 before: {initial_expr_nnue}\n\
         \x20 after:  {final_expr}",
        steps.len(),
    );

    // 7. Return trajectory
    Some(Trajectory {
        trajectory_id,
        seed_expr: seed_name.to_string(),
        steps,
        initial_cost_ns,
        final_cost_ns,
        initial_cost: Some(initial_cost),
        final_cost: Some(best_cost),
    })
}

// ============================================================================
// Batch generator
// ============================================================================

/// Stack size for worker threads: 16MB. Default 8MB is usually sufficient since
/// we use iterative traversals, but 16MB provides safety margin for deep
/// expression trees during e-graph extraction.
const WORKER_STACK_SIZE: usize = 16 * 1024 * 1024;

/// Determine effective worker count from an optional hint.
///
/// Returns `workers.unwrap_or(available_parallelism)`, clamped to `[1, 256]`.
/// Falls back to 1 if `available_parallelism` cannot be determined.
fn effective_workers(workers: Option<usize>) -> usize {
    let n = workers.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    n.clamp(1, 256)
}

/// Generate `count` trajectories from random seeds using [`BwdGenerator`].
///
/// Each seed expression is generated randomly, then run through self-play
/// hill-climbing. Trajectories that fail JIT benchmarking are silently
/// dropped (logged to stderr).
pub fn generate_trajectory_batch(
    model: &ExprNnue,
    templates: &RuleTemplates,
    rules: &[Box<dyn Rewrite>],
    count: usize,
    seed: u64,
    threshold: f32,
    max_epochs: usize,
    config: &BwdGenConfig,
) -> Vec<Trajectory> {
    generate_trajectory_batch_parallel(
        model, templates, rules, count, seed, threshold, max_epochs, config, None,
    )
}

/// Generate `count` trajectories in parallel across `workers` threads.
///
/// Expression seeds are pre-generated sequentially (BwdGenerator is stateful),
/// then trajectory self-play runs in parallel. Each worker thread gets its own
/// stack and runs independent trajectories sharing the model read-only.
///
/// Pass `workers = None` to auto-detect from `available_parallelism`.
/// Pass `workers = Some(1)` for sequential execution (useful for debugging).
pub fn generate_trajectory_batch_parallel(
    model: &ExprNnue,
    templates: &RuleTemplates,
    rules: &[Box<dyn Rewrite>],
    count: usize,
    seed: u64,
    threshold: f32,
    max_epochs: usize,
    config: &BwdGenConfig,
    workers: Option<usize>,
) -> Vec<Trajectory> {
    let num_workers = effective_workers(workers);
    let rule_embeds = model.encode_all_rules_from_templates(templates);

    // Pre-generate all expression pairs sequentially. BwdGenerator maintains
    // internal PRNG state so it cannot be safely split across threads.
    let mut generator = BwdGenerator::new(seed, config.clone(), templates.clone());
    let work_items: Vec<_> = (0..count)
        .map(|i| {
            let pair = generator.generate();
            let traj_id = format!("seed_{seed}_t{i}");
            let name = format!("expr_{i}");
            (pair.unoptimized, name, traj_id)
        })
        .collect();

    if num_workers <= 1 || count <= 1 {
        // Sequential fast path: no thread overhead.
        let mut trajectories = Vec::new();
        for (expr, name, traj_id) in &work_items {
            if let Some(traj) = run_self_play_trajectory(
                expr, name, model, &rule_embeds, rules, threshold, max_epochs,
                traj_id.clone(),
            ) {
                trajectories.push(traj);
            }
        }
        eprintln!("Generated {}/{count} trajectories (sequential)", trajectories.len());
        return trajectories;
    }

    // Parallel: partition work items across workers, spawn scoped threads.
    let num_rules = rules.len();
    let chunk_size = (count + num_workers - 1) / num_workers;
    let chunks: Vec<&[(Expr, String, String)]> = work_items.chunks(chunk_size).collect();
    let actual_workers = chunks.len();

    eprintln!(
        "[PARALLEL] Spawning {actual_workers} workers for {count} trajectories ({chunk_size} each)"
    );

    let mut all_trajectories: Vec<Trajectory> = Vec::new();

    std::thread::scope(|scope| {
        let handles: Vec<_> = chunks
            .into_iter()
            .enumerate()
            .map(|(worker_id, chunk)| {
                // Each worker borrows model, rule_embeds, and creates its own rules.
                let model_ref = &*model;
                let rule_embeds_ref = &rule_embeds;

                std::thread::Builder::new()
                    .name(format!("traj-worker-{worker_id}"))
                    .stack_size(WORKER_STACK_SIZE)
                    .spawn_scoped(scope, move || {
                        // Each worker creates its own rules (avoids sharing Box<dyn Rewrite>
                        // through the hot loop — rules are used by EGraph internally).
                        let worker_rules: Vec<Box<dyn Rewrite>> = all_rules();
                        assert_eq!(
                            worker_rules.len(), num_rules,
                            "Worker {worker_id}: rule count mismatch ({} vs {num_rules})",
                            worker_rules.len()
                        );

                        let mut results = Vec::with_capacity(chunk.len());
                        for (expr, name, traj_id) in chunk {
                            if let Some(traj) = run_self_play_trajectory(
                                expr,
                                name,
                                model_ref,
                                rule_embeds_ref,
                                &worker_rules,
                                threshold,
                                max_epochs,
                                traj_id.clone(),
                            ) {
                                results.push(traj);
                            }
                        }
                        results
                    })
                    .unwrap_or_else(|e| {
                        panic!("Failed to spawn trajectory worker {worker_id}: {e}")
                    })
            })
            .collect();

        for handle in handles {
            let worker_results = handle.join().unwrap_or_else(|e| {
                panic!(
                    "Trajectory worker panicked: {}",
                    e.downcast_ref::<String>()
                        .map(|s| s.as_str())
                        .or_else(|| e.downcast_ref::<&str>().copied())
                        .unwrap_or("unknown panic")
                )
            });
            all_trajectories.extend(worker_results);
        }
    });

    eprintln!(
        "Generated {}/{count} trajectories ({actual_workers} workers)",
        all_trajectories.len()
    );
    all_trajectories
}

// ============================================================================
// JSONL I/O helpers
// ============================================================================

/// Write trajectories to a JSONL file (one JSON object per line).
///
/// Panics on I/O or serialization errors (fail fast, fail loudly).
pub fn write_trajectories_jsonl(trajectories: &[Trajectory], path: &Path) {
    let file = std::fs::File::create(path)
        .unwrap_or_else(|e| panic!("Failed to create {}: {e}", path.display()));
    let mut writer = BufWriter::new(file);
    for traj in trajectories {
        let json = serde_json::to_string(traj)
            .unwrap_or_else(|e| panic!("Failed to serialize trajectory: {e}"));
        writeln!(writer, "{json}")
            .unwrap_or_else(|e| panic!("Failed to write to {}: {e}", path.display()));
    }
    writer
        .flush()
        .unwrap_or_else(|e| panic!("Failed to flush {}: {e}", path.display()));
}

/// Read trajectories from a JSONL file.
///
/// Each line is a JSON-serialized [`Trajectory`].
/// Empty lines are skipped. Panics on I/O or parse errors (fail fast, fail loudly).
pub fn load_trajectories_jsonl(path: &Path) -> Vec<Trajectory> {
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));
    let reader = std::io::BufReader::new(file);
    let mut trajectories = Vec::new();
    for (line_num, line) in std::io::BufRead::lines(reader).enumerate() {
        let line = line.unwrap_or_else(|e| panic!("Failed to read line {} of {}: {e}", line_num + 1, path.display()));
        if line.trim().is_empty() { continue; }
        let traj: Trajectory = serde_json::from_str(&line)
            .unwrap_or_else(|e| panic!("Failed to parse trajectory at {}:{}: {e}", path.display(), line_num + 1));
        trajectories.push(traj);
    }
    trajectories
}

/// Read per-trajectory advantage scores from a JSONL file.
///
/// Each line is a JSON-serialized [`TrajectoryAdvantages`].
/// Empty lines are skipped. Panics on I/O or parse errors (fail fast, fail loudly).
pub fn read_advantages_jsonl(path: &Path) -> Vec<TrajectoryAdvantages> {
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));
    let reader = BufReader::new(file);
    reader
        .lines()
        .enumerate()
        .filter_map(|(line_num, line)| {
            let line = line
                .unwrap_or_else(|e| panic!("Read error at line {line_num}: {e}"));
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return None;
            }
            Some(
                serde_json::from_str(trimmed)
                    .unwrap_or_else(|e| panic!("JSON parse error at line {line_num}: {e}")),
            )
        })
        .collect()
}

// ============================================================================
// Corpus loading and trajectory generation
// ============================================================================

/// Load expressions from bench_corpus.jsonl, parse via `parse_kernel_code`.
///
/// Returns up to `max_count` `(name, Expr)` pairs, sampled uniformly via
/// LCG shuffle.
///
/// # Panics
///
/// Panics if zero expressions parse successfully from the file.
pub fn load_corpus_exprs(path: &Path, max_count: usize, seed: u64) -> Vec<(String, Expr)> {
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("Failed to open corpus file {}: {e}", path.display()));
    let reader = BufReader::new(file);

    let mut parsed: Vec<(String, Expr)> = Vec::new();
    let mut total_lines = 0u64;
    let mut parse_failures = 0u64;

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result
            .unwrap_or_else(|e| panic!("Read error at line {line_num} in {}: {e}", path.display()));
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        total_lines += 1;

        // Parse JSON to extract "name" and "expression" fields
        let json: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                eprintln!(
                    "WARNING: JSON parse error at line {line_num} in {}: {e}",
                    path.display()
                );
                parse_failures += 1;
                continue;
            }
        };

        let name = match json.get("name").and_then(|v| v.as_str()) {
            Some(n) => n.to_string(),
            None => {
                eprintln!(
                    "WARNING: missing or non-string 'name' at line {line_num} in {}",
                    path.display()
                );
                parse_failures += 1;
                continue;
            }
        };

        let expression = match json.get("expression").and_then(|v| v.as_str()) {
            Some(e) => e,
            None => {
                eprintln!(
                    "WARNING: missing or non-string 'expression' at line {line_num} in {}",
                    path.display()
                );
                parse_failures += 1;
                continue;
            }
        };

        match parse_kernel_code(expression) {
            Some(expr) => parsed.push((name, expr)),
            None => {
                eprintln!(
                    "WARNING: parse_kernel_code failed for '{}' (line {line_num}, name='{name}') in {}",
                    expression,
                    path.display()
                );
                parse_failures += 1;
            }
        }
    }

    assert!(
        !parsed.is_empty(),
        "Zero expressions parsed successfully from {} ({total_lines} lines, {parse_failures} failures)",
        path.display()
    );

    // LCG-based Fisher-Yates shuffle
    let mut state = seed;
    let len = parsed.len();
    for i in (1..len).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        parsed.swap(i, j);
    }

    // Truncate to max_count
    parsed.truncate(max_count);

    eprintln!(
        "Loaded {} corpus expressions from {} ({total_lines} lines, {parse_failures} parse failures)",
        parsed.len(),
        path.display()
    );

    parsed
}

/// Run self-play trajectories on corpus expressions.
///
/// Samples `count` expressions from `corpus` with replacement using LCG,
/// then runs each through [`run_self_play_trajectory`].
pub fn generate_corpus_trajectories(
    model: &ExprNnue,
    templates: &RuleTemplates,
    rules: &[Box<dyn Rewrite>],
    corpus: &[(String, Expr)],
    count: usize,
    seed: u64,
    threshold: f32,
    max_epochs: usize,
) -> Vec<Trajectory> {
    generate_corpus_trajectories_parallel(
        model, templates, rules, corpus, count, seed, threshold, max_epochs, None,
    )
}

/// Run self-play trajectories on corpus expressions in parallel.
///
/// Samples `count` expressions from `corpus` with replacement using LCG
/// (deterministic regardless of worker count), then distributes trajectory
/// self-play across `workers` threads.
///
/// Pass `workers = None` to auto-detect from `available_parallelism`.
pub fn generate_corpus_trajectories_parallel(
    model: &ExprNnue,
    templates: &RuleTemplates,
    rules: &[Box<dyn Rewrite>],
    corpus: &[(String, Expr)],
    count: usize,
    seed: u64,
    threshold: f32,
    max_epochs: usize,
    workers: Option<usize>,
) -> Vec<Trajectory> {
    assert!(!corpus.is_empty(), "corpus must be non-empty");

    let num_workers = effective_workers(workers);
    let rule_embeds = model.encode_all_rules_from_templates(templates);
    let corpus_len = corpus.len();

    // Pre-compute corpus indices deterministically using LCG.
    let mut state = seed;
    let work_items: Vec<_> = (0..count)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let idx = (state >> 33) as usize % corpus_len;
            let traj_id = format!("corpus_{seed}_t{i}");
            (idx, traj_id)
        })
        .collect();

    if num_workers <= 1 || count <= 1 {
        // Sequential fast path.
        let mut trajectories = Vec::new();
        for (idx, traj_id) in &work_items {
            let (ref name, ref expr) = corpus[*idx];
            match run_self_play_trajectory(
                expr, name, model, &rule_embeds, rules, threshold, max_epochs,
                traj_id.clone(),
            ) {
                Some(traj) => trajectories.push(traj),
                None => {
                    eprintln!(
                        "[corpus] no trajectory produced for '{name}' (JIT error or no rules matched)"
                    );
                }
            }
        }
        eprintln!(
            "Generated {}/{count} corpus trajectories (sequential)",
            trajectories.len()
        );
        return trajectories;
    }

    // Parallel: partition work items across workers.
    let num_rules = rules.len();
    let chunk_size = (count + num_workers - 1) / num_workers;
    let chunks: Vec<&[(usize, String)]> = work_items.chunks(chunk_size).collect();
    let actual_workers = chunks.len();

    eprintln!(
        "[PARALLEL] Spawning {actual_workers} workers for {count} corpus trajectories"
    );

    let mut all_trajectories: Vec<Trajectory> = Vec::new();

    std::thread::scope(|scope| {
        let handles: Vec<_> = chunks
            .into_iter()
            .enumerate()
            .map(|(worker_id, chunk)| {
                let model_ref = &*model;
                let rule_embeds_ref = &rule_embeds;
                let corpus_ref = corpus;

                std::thread::Builder::new()
                    .name(format!("corpus-worker-{worker_id}"))
                    .stack_size(WORKER_STACK_SIZE)
                    .spawn_scoped(scope, move || {
                        let worker_rules: Vec<Box<dyn Rewrite>> = all_rules();
                        assert_eq!(
                            worker_rules.len(), num_rules,
                            "Worker {worker_id}: rule count mismatch ({} vs {num_rules})",
                            worker_rules.len()
                        );

                        let mut results = Vec::with_capacity(chunk.len());
                        for (idx, traj_id) in chunk {
                            let (ref name, ref expr) = corpus_ref[*idx];
                            match run_self_play_trajectory(
                                expr,
                                name,
                                model_ref,
                                rule_embeds_ref,
                                &worker_rules,
                                threshold,
                                max_epochs,
                                traj_id.clone(),
                            ) {
                                Some(traj) => results.push(traj),
                                None => {
                                    eprintln!(
                                        "[corpus] no trajectory produced for '{name}' \
                                         (JIT error or no rules matched)"
                                    );
                                }
                            }
                        }
                        results
                    })
                    .unwrap_or_else(|e| {
                        panic!("Failed to spawn corpus worker {worker_id}: {e}")
                    })
            })
            .collect();

        for handle in handles {
            let worker_results = handle.join().unwrap_or_else(|e| {
                panic!(
                    "Corpus worker panicked: {}",
                    e.downcast_ref::<String>()
                        .map(|s| s.as_str())
                        .or_else(|| e.downcast_ref::<&str>().copied())
                        .unwrap_or("unknown panic")
                )
            });
            all_trajectories.extend(worker_results);
        }
    });

    eprintln!(
        "Generated {}/{count} corpus trajectories ({actual_workers} workers)",
        all_trajectories.len()
    );
    all_trajectories
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acc_to_vec_correct_length() {
        let acc = EdgeAccumulator::default();
        let v = acc_to_vec(&acc);
        assert_eq!(
            v.len(),
            INPUT_DIM,
            "Expected {INPUT_DIM} floats, got {}",
            v.len()
        );
    }

    #[test]
    fn acc_to_vec_preserves_values() {
        let mut acc = EdgeAccumulator::default();
        acc.values[0] = 1.5;
        acc.values[127] = -3.0;
        acc.edge_count = 7;
        acc.node_count = 12;

        let v = acc_to_vec(&acc);
        assert!((v[0] - 1.5).abs() < 1e-9, "values[0] mismatch");
        assert!((v[127] - (-3.0)).abs() < 1e-9, "values[127] mismatch");
        assert!((v[128] - 7.0).abs() < 1e-9, "edge_count mismatch");
        assert!((v[129] - 12.0).abs() < 1e-9, "node_count mismatch");
    }

    #[test]
    fn trajectory_jsonl_round_trip() {
        let traj = Trajectory {
            trajectory_id: "test".into(),
            seed_expr: "X".into(),
            steps: vec![TrajectoryStep {
                accumulator_state: vec![0.0; 130],
                expression_embedding: vec![0.0; 32],
                rule_embedding: vec![0.0; 32],
                budget_remaining: 1000,
                epochs_remaining: 10,
                action_probability: 0.5,
                matched: true,
                jit_cost_ns: 5.0,
                edges: vec![(2, 3, 0)],
                graph_accumulator_state: vec![0.0; GRAPH_INPUT_DIM],
            }],
            initial_cost_ns: 5.0,
            final_cost_ns: 1.0,
            initial_cost: Some(5.0),
            final_cost: Some(1.0),
        };

        let tmp = std::env::temp_dir().join("test_self_play_traj.jsonl");
        write_trajectories_jsonl(&[traj], &tmp);

        // Read back and verify
        let contents = std::fs::read_to_string(&tmp)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", tmp.display()));
        let back: Trajectory = serde_json::from_str(contents.trim())
            .unwrap_or_else(|e| panic!("Failed to parse trajectory JSONL: {e}"));
        assert_eq!(back.trajectory_id, "test");
        assert_eq!(back.steps.len(), 1);
        assert!(back.steps[0].matched);
        assert!((back.final_cost_ns - 1.0).abs() < 1e-6);
        assert!((back.initial_cost.expect("Expected value but got None/Err") - 5.0).abs() < 1e-6);

        std::fs::remove_file(&tmp)
            .unwrap_or_else(|e| panic!("Failed to remove temp file {}: {e}", tmp.display()));
    }

    #[test]
    fn advantages_jsonl_round_trip() {
        let adv = TrajectoryAdvantages {
            trajectory_idx: 0,
            advantages: vec![0.1, -0.2, 0.3],
        };

        let tmp = std::env::temp_dir().join("test_self_play_adv.jsonl");
        {
            let file = std::fs::File::create(&tmp)
                .unwrap_or_else(|e| panic!("Failed to create {}: {e}", tmp.display()));
            let mut w = BufWriter::new(file);
            writeln!(
                w,
                "{}",
                serde_json::to_string(&adv)
                    .unwrap_or_else(|e| panic!("Failed to serialize advantages: {e}"))
            )
            .unwrap_or_else(|e| panic!("Failed to write: {e}"));
            w.flush()
                .unwrap_or_else(|e| panic!("Failed to flush: {e}"));
        }

        let read_back = read_advantages_jsonl(&tmp);
        assert_eq!(read_back.len(), 1);
        assert_eq!(read_back[0].advantages.len(), 3);
        assert!((read_back[0].advantages[0] - 0.1).abs() < 1e-6);
        assert!((read_back[0].advantages[1] - (-0.2)).abs() < 1e-6);
        assert!((read_back[0].advantages[2] - 0.3).abs() < 1e-6);

        std::fs::remove_file(&tmp)
            .unwrap_or_else(|e| panic!("Failed to remove temp file {}: {e}", tmp.display()));
    }

    #[test]
    fn advantages_jsonl_skips_empty_lines() {
        let tmp = std::env::temp_dir().join("test_self_play_adv_empty.jsonl");
        {
            let file = std::fs::File::create(&tmp)
                .unwrap_or_else(|e| panic!("Failed to create {}: {e}", tmp.display()));
            let mut w = BufWriter::new(file);
            // Write with some empty lines interspersed
            writeln!(w).expect("Expected value but got None/Err");
            writeln!(
                w,
                "{}",
                serde_json::to_string(&TrajectoryAdvantages {
                    trajectory_idx: 0,
                    advantages: vec![1.0],
                })
                .expect("Expected value but got None/Err")
            )
            .expect("Expected value but got None/Err");
            writeln!(w).expect("Expected value but got None/Err");
            writeln!(w, "   ").expect("Expected value but got None/Err");
            writeln!(
                w,
                "{}",
                serde_json::to_string(&TrajectoryAdvantages {
                    trajectory_idx: 1,
                    advantages: vec![2.0, 3.0],
                })
                .expect("Expected value but got None/Err")
            )
            .expect("Expected value but got None/Err");
            w.flush().expect("Expected value but got None/Err");
        }

        let read_back = read_advantages_jsonl(&tmp);
        assert_eq!(read_back.len(), 2, "Should have 2 records, skipping empty lines");
        assert_eq!(read_back[0].trajectory_idx, 0);
        assert_eq!(read_back[1].trajectory_idx, 1);

        std::fs::remove_file(&tmp)
            .unwrap_or_else(|e| panic!("Failed to remove temp file {}: {e}", tmp.display()));
    }

    #[test]
    fn sigmoid_basic_values() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6, "sigmoid(0) should be 0.5");
        assert!(sigmoid(10.0) > 0.999, "sigmoid(10) should be ~1.0");
        assert!(sigmoid(-10.0) < 0.001, "sigmoid(-10) should be ~0.0");
    }

    #[test]
    fn gacc_to_vec_correct_length() {
        let gacc = GraphAccumulator::new();
        let v = gacc_to_vec(&gacc);
        assert_eq!(
            v.len(),
            GRAPH_INPUT_DIM,
            "Expected {GRAPH_INPUT_DIM} floats, got {}",
            v.len()
        );
    }

    #[test]
    fn gacc_to_vec_preserves_values() {
        let mut gacc = GraphAccumulator::new();
        gacc.values[0] = 2.5;
        gacc.values[95] = -1.0;
        gacc.edge_count = 5;
        gacc.node_count = 8;

        let v = gacc_to_vec(&gacc);
        assert!((v[0] - 2.5).abs() < 1e-9, "values[0] mismatch");
        assert!((v[95] - (-1.0)).abs() < 1e-9, "values[95] mismatch");
        assert!((v[96] - 5.0).abs() < 1e-9, "edge_count mismatch");
        assert!((v[97] - 8.0).abs() < 1e-9, "node_count mismatch");
    }

    #[test]
    fn build_graph_acc_empty_egraph() {
        use pixelflow_search::nnue::OpEmbeddings;

        let egraph = EGraph::new();
        let emb = OpEmbeddings::default();
        let gacc = build_graph_acc(&egraph, &emb);
        assert_eq!(gacc.edge_count, 0, "empty e-graph should have 0 edges");
        assert_eq!(gacc.node_count, 0, "empty e-graph should have 0 nodes");
    }

    #[test]
    fn build_graph_acc_single_var() {
        use pixelflow_search::egraph::{EGraph, ENode};
        use pixelflow_search::nnue::OpEmbeddings;

        let mut egraph = EGraph::new();
        egraph.add(ENode::Var(0));
        let emb = OpEmbeddings::default();
        let gacc = build_graph_acc(&egraph, &emb);
        assert_eq!(gacc.node_count, 1, "single var should have 1 node");
        assert_eq!(gacc.edge_count, 0, "single var should have 0 edges");
    }
}
