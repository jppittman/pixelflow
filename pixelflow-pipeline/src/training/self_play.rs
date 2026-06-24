//! # Self-Play Trajectory Generator
//!
//! Runs hill-climbing trajectories with the current mask weights (self-play,
//! no perturbation) and records per-step payload for the Critic.
//!
//! This is the "GENERATE" phase of the unified training loop.
//!
//! It runs the current saturation policy directly, records per-step payload for
//! the Critic, benchmarks the final extracted expression for terminal cost, and
//! returns [`Trajectory`] values for the shared replay/update path.

use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use pixelflow_ir::{ExprArena, ExprId, ExprNode};
use pixelflow_search::egraph::{
    EGraph, ENode, Rewrite, all_rules, compute_ref_counts, extract_neural_to_arena,
};
use pixelflow_search::nnue::factored::INPUT_DIM;
use pixelflow_search::nnue::factored::MAX_ARITY;
use pixelflow_search::nnue::{GRAPH_ACC_DIM, BwdGenConfig, BwdGenerator, EMBED_DIM, EdgeAccumulator, ExprNnue,
    GRAPH_INPUT_DIM, GraphAccumulator, OpKind, RuleTemplates,
};

use crate::jit_bench::benchmark_jit_arena;
use crate::training::corpus;
use crate::training::factored::arena_to_kernel_code;
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
/// Layout: `[values (4*K=128 floats), edge_count, node_count, node_budget, epoch_budget]` = 132 floats total.
pub fn acc_to_vec(acc: &EdgeAccumulator) -> Vec<f32> {
    let mut v = Vec::with_capacity(INPUT_DIM);
    v.extend_from_slice(&acc.values);
    v.push(acc.edge_count as f32);
    v.push(acc.node_count as f32);
    v.push(acc.node_budget as f32);
    v.push(acc.epoch_budget as f32);
    v
}

/// Convert a [`GraphAccumulator`] to a flat `GRAPH_INPUT_DIM`-element `Vec<f32>`.
///
/// Layout: `[values (4*K=128 floats), edge_count, node_count, node_budget, epoch_budget]` = 132 floats total.
pub fn gacc_to_vec(gacc: &GraphAccumulator) -> Vec<f32> {
    let mut v = Vec::with_capacity(GRAPH_INPUT_DIM);
    v.extend_from_slice(&gacc.values);
    v.push(gacc.edge_count as f32);
    v.push(gacc.node_count as f32);
    v.push(gacc.node_budget as f32);
    v.push(gacc.epoch_budget as f32);
    v
}

/// Build a [`GraphAccumulator`] from the current e-graph state.
///
/// O(V+E) traversal. Walks all canonical e-classes, resolves child ops
/// via union-find, and accumulates VSA-encoded edges. Rebuilt from scratch
/// each epoch because union-find merges change canonical representatives,
/// making incremental updates unsound.
fn build_graph_acc(
    egraph: &EGraph,
    emb: &pixelflow_search::nnue::OpEmbeddings,
) -> GraphAccumulator {
    use std::collections::HashMap;

    let mut gacc = GraphAccumulator::new();

    // Pass 1: collect 1-hop edges and build a parent map (child_class → parent_op).
    // For 2-hop: if child class C has parent op P, and P's class has parent op GP,
    // then (GP, P, child_op) is a 2-hop path.
    //
    // parent_ops maps: canonical class id → Vec of OpKind that appear as parents of that class.
    let mut parent_ops: HashMap<usize, Vec<OpKind>> = HashMap::new();

    for class_id in egraph.class_ids() {
        for node in egraph.nodes(class_id) {
            match node {
                ENode::Var(_) | ENode::Const(_) => gacc.add_leaf(),
                ENode::Op { op, children } => {
                    let op_kind = op.kind();
                    let child_ops: Vec<OpKind> =
                        children.iter().map(|c| egraph.canonical_op(*c)).collect();
                    gacc.add_op_node(emb, op_kind, &child_ops);

                    // Record this op as a parent of each child class
                    for c in children {
                        parent_ops.entry(c.index()).or_default().push(op_kind);
                    }
                }
            }
        }
    }

    // Pass 2: 2-hop edges. For each (parent → child) edge, look up grandparents
    // of the parent's class and emit (grandparent, parent, child) triples.
    for class_id in egraph.class_ids() {
        // What ops are parents of this class?
        let gp_ops = match parent_ops.get(&class_id.index()) {
            Some(ops) => ops.clone(),
            None => continue, // root class, no grandparent paths
        };

        for node in egraph.nodes(class_id) {
            if let ENode::Op { op, children } = node {
                let parent_op = op.kind();
                for child in children {
                    let child_op = egraph.canonical_op(*child);
                    // Each grandparent of this class creates a 2-hop path
                    for &gp_op in &gp_ops {
                        gacc.add_2hop_edge(emb, gp_op, parent_op, child_op);
                    }
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
        templates.build(idx, rule.as_ref());
    }

    templates
}

// ============================================================================
// Edge collection for embedding gradients
// ============================================================================

/// Collect edges from an arena expression for embedding gradient flow.
pub fn collect_edges_dedup_arena(arena: &ExprArena, root: ExprId) -> Vec<(u8, u8, u16)> {
    use std::collections::BTreeSet;

    let mut edges = Vec::new();
    let mut seen = BTreeSet::<ExprId>::new();
    let mut stack: Vec<(ExprId, u32)> = vec![(root, 0)];

    while let Some((id, d)) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }

        match arena.node(id) {
            ExprNode::Var(_) | ExprNode::Const(_) => {}
            ExprNode::Param(i) => {
                panic!("ExprNode::Param({i}) reached edge collection — substitute params first")
            }
            ExprNode::Unary(op, child) => {
                edges.push((
                    op.index() as u8,
                    arena.kind(*child).index() as u8,
                    (d * MAX_ARITY as u32) as u16,
                ));
                stack.push((*child, d + 1));
            }
            ExprNode::Binary(op, left, right) => {
                edges.push((
                    op.index() as u8,
                    arena.kind(*left).index() as u8,
                    (d * MAX_ARITY as u32) as u16,
                ));
                edges.push((
                    op.index() as u8,
                    arena.kind(*right).index() as u8,
                    (d * MAX_ARITY as u32 + 1) as u16,
                ));
                stack.push((*right, d + 1));
                stack.push((*left, d + 1));
            }
            ExprNode::Ternary(op, a, b, c) => {
                edges.push((
                    op.index() as u8,
                    arena.kind(*a).index() as u8,
                    (d * MAX_ARITY as u32) as u16,
                ));
                edges.push((
                    op.index() as u8,
                    arena.kind(*b).index() as u8,
                    (d * MAX_ARITY as u32 + 1) as u16,
                ));
                edges.push((
                    op.index() as u8,
                    arena.kind(*c).index() as u8,
                    (d * MAX_ARITY as u32 + 2) as u16,
                ));
                stack.push((*c, d + 1));
                stack.push((*b, d + 1));
                stack.push((*a, d + 1));
            }
            ExprNode::Nary(op, _, _) => {
                for (idx, child) in arena.children(id).enumerate() {
                    let eff_depth = d * MAX_ARITY as u32 + (idx.min(MAX_ARITY - 1)) as u32;
                    edges.push((
                        op.index() as u8,
                        arena.kind(child).index() as u8,
                        eff_depth as u16,
                    ));
                }
                for child in arena.children(id) {
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
    seed_arena: &ExprArena,
    seed_root: ExprId,
    seed_name: &str,
    model: &ExprNnue,
    rule_embeds: &[[f32; EMBED_DIM]],
    rules: &[Box<dyn Rewrite>],
    threshold: f32,
    max_epochs: usize,
    trajectory_id: String,
) -> Option<Trajectory> {
    // Hard wall-clock deadline per trajectory: safety net against runaway extraction.
    // 5s is enough for any reasonable expression — pathological ones get skipped.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let traj_start = std::time::Instant::now();

    // 1. Convert seed expression to e-graph
    let mut egraph = EGraph::with_rules(all_rules());
    let root = egraph.add_arena(seed_arena, seed_root);
    let t_egraph_build = traj_start.elapsed();

    // 2. Score the initial expression
    let (initial_arena, initial_root, initial_cost) = extract_neural_to_arena(&egraph, root, model);

    // 2b. JIT benchmark the initial expression for ground-truth initial cost
    if initial_arena.has_degenerate(initial_root) {
        eprintln!("Skipping degenerate seed expression in {trajectory_id} (seed={seed_name})");
        return None;
    }
    let t_before_initial_jit = traj_start.elapsed();
    let initial_bench = match benchmark_jit_arena(&initial_arena, initial_root) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "JIT bench failed for initial expr in trajectory {trajectory_id} (seed={seed_name}): {e}"
            );
            return None;
        }
    };
    let t_after_initial_jit = traj_start.elapsed();

    // Domain check: if any SIMD lane is NaN or Inf at the test point, the
    // expression is undefined there. Rewrite equivalence checks would always
    // fire false positives (REWRITE BUG) because IEEE 754 NaN/Inf behavior
    // diverges across mathematically-equivalent forms. Skip early.
    if initial_bench.output.iter().any(|x| !x.is_finite()) {
        eprintln!("[SKIP] {seed_name}: initial output contains NaN/Inf, skipping");
        return None;
    }

    let initial_cost_ns = initial_bench.ns;

    // 3. Track best
    let best_cost = initial_cost;
    let mut steps: Vec<TrajectoryStep> = Vec::new();

    let num_rules = rules.len();

    // Randomize resource constraints per trajectory, scaled by expression size.
    // Larger expressions need more budget to explore meaningful rewrites.
    // Budget = base + multiplier * node_count, with random multiplier.
    let traj_hash: usize = trajectory_id
        .bytes()
        .fold(0usize, |h, b| h.wrapping_mul(31).wrapping_add(b as usize));
    let initial_nodes = initial_arena.node_count_subtree(initial_root);
    // multiplier in [3, 10] — small exprs get 50-200, large (100-node) get 300-1000
    let budget_mult = 3 + (traj_hash % 8); // [3, 10]
    let node_budget = (50 + budget_mult * initial_nodes).min(2000); // floor 50, cap 2000
    let epoch_budget = 10 + (traj_hash.wrapping_mul(7) % 51); // [10, 60]
    let effective_epochs = max_epochs.min(epoch_budget);

    // 4. Epoch loop
    //
    // KEY DESIGN: Score rules from the e-graph state (GraphAccumulator), not
    // from a single extracted expression. After rules fire and create new
    // e-nodes/equivalences, rebuild the GraphAccumulator and re-score to
    // approve newly-passing rules. The EdgeAccumulator is still built from
    // the initial expression for the extraction head (TrajectoryStep.accumulator_state).
    //
    // No per-epoch extraction or JIT. The only JIT benchmark is the final one.
    // The transformer critic assigns per-step credit from the single terminal reward.

    let mut acc =
        EdgeAccumulator::from_arena_dedup(&initial_arena, initial_root, &model.embeddings);
    acc.node_budget = node_budget as u32;
    acc.epoch_budget = epoch_budget as u32;
    let acc_vec = acc_to_vec(&acc);
    let edges = collect_edges_dedup_arena(&initial_arena, initial_root);
    let hidden = model.forward_shared(&acc);
    let expr_embed = model.compute_expr_embed(&hidden);
    let expr_embed_vec: Vec<f32> = expr_embed.to_vec();

    // Build GraphAccumulator from initial e-graph state (for mask scoring)
    let mut gacc = build_graph_acc(&egraph, &model.embeddings);
    gacc.node_budget = node_budget as u32;
    gacc.epoch_budget = epoch_budget as u32;
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
            rule_embedding: rule_embeds
                .get(r)
                .map_or_else(|| vec![0.0; EMBED_DIM], |e| e.to_vec()),
            budget_remaining: (node_budget as i32) - (egraph.node_count() as i32),
            epochs_remaining: effective_epochs as i32,
            action_probability: prob,
            matched: false,        // Updated during saturation
            jit_cost_ns: f64::NAN, // Backfilled with final cost
            edges: edges.clone(),
            graph_accumulator_state: gacc_vec.clone(),
        });
    }

    // Saturate: apply approved rules, rebuild graph accumulator, re-score
    let t_before_saturate = traj_start.elapsed();
    for epoch in 0..effective_epochs {
        if std::time::Instant::now() > deadline {
            panic!(
                "Trajectory {trajectory_id} hit 5s wall-clock deadline at epoch {epoch} \
                 (node_budget={node_budget}, egraph_nodes={}, epoch_budget={epoch_budget}). \
                 This means resource budgets are not preventing runaway computation.",
                egraph.node_count()
            );
        }

        if egraph.node_count() > node_budget {
            break;
        }

        // Apply all approved rules in a single batch — one rebuild per epoch.
        let any_changed = {
            let mut batch = egraph.batch();
            let mut changed = false;
            for (step_idx, &(r, _)) in approved_rules.iter().enumerate() {
                if batch.node_count() > node_budget {
                    break;
                }
                let result = batch.apply_rule(r, node_budget, Some(deadline));
                if result.changes > 0 {
                    steps[step_idx].matched = true;
                    changed = true;
                }
            }
            changed
            // rebuild happens here on drop
        };

        if !any_changed {
            break;
        }

        // Rebuild graph accumulator from post-union-find state
        gacc = build_graph_acc(&egraph, &model.embeddings);
        gacc.node_budget = node_budget as u32;
        gacc.epoch_budget = epoch_budget as u32;
        gacc_vec = gacc_to_vec(&gacc);

        // Re-score with accurate graph state
        let new_scores = model.mask_score_all_rules_graph(&gacc, rule_embeds);

        // Approve newly-passing rules, record new steps with updated gacc
        let epochs_remaining = (effective_epochs as i32) - (epoch as i32) - 1;
        for r in 0..num_rules {
            let score = if r < new_scores.len() {
                new_scores[r]
            } else {
                0.0
            };
            let prob = sigmoid(score);
            if prob > threshold && !approved_rules.iter().any(|(idx, _)| *idx == r) {
                approved_rules.push((r, prob));
                steps.push(TrajectoryStep {
                    accumulator_state: acc_vec.clone(),
                    expression_embedding: expr_embed_vec.clone(),
                    rule_embedding: rule_embeds
                        .get(r)
                        .map_or_else(|| vec![0.0; EMBED_DIM], |e| e.to_vec()),
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
    let t_after_saturate = traj_start.elapsed();
    let post_sat_nodes = egraph.node_count();
    if post_sat_nodes > 5000 {
        eprintln!(
            "Trajectory {trajectory_id} egraph too large ({post_sat_nodes} nodes, budget={node_budget}), skipping"
        );
        return None;
    }
    if std::time::Instant::now() > deadline {
        eprintln!(
            "Trajectory {trajectory_id} hit deadline before extraction (seed={seed_name}, nodes={post_sat_nodes})"
        );
        return None;
    }
    let extractor = pixelflow_search::egraph::IncrementalExtractor::new(model, 8);
    let (_final_cost, final_choices) = extractor.extract_choices_only(&egraph, root);
    let final_ref_counts = compute_ref_counts(&egraph, root, &final_choices);
    let (final_arena, final_arena_root) =
        pixelflow_search::egraph::choices_to_arena(&egraph, root, &final_choices);

    // Degenerate check on the arena (NaN/Inf constants, recip/div-by-zero).
    if final_arena.has_degenerate(final_arena_root) {
        eprintln!("Rewrite produced degenerate expression in {trajectory_id} (seed={seed_name})");
        return None;
    }
    let final_bench = match benchmark_jit_arena(&final_arena, final_arena_root) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("JIT bench failed for trajectory {trajectory_id} (seed={seed_name}): {e}");
            return None;
        }
    };
    let final_cost_ns = final_bench.ns;
    let t_total = traj_start.elapsed();

    // Log slow trajectories with phase breakdown
    if t_total.as_millis() > 100 {
        eprintln!(
            "[SLOW_TRAJ] {trajectory_id} (seed={seed_name}) total={:.1}ms: \
             egraph_build={:.1}ms extract+score={:.1}ms initial_jit={:.1}ms \
             setup={:.1}ms saturate={:.1}ms final={:.1}ms nodes={post_sat_nodes}",
            t_total.as_secs_f64() * 1000.0,
            t_egraph_build.as_secs_f64() * 1000.0,
            (t_before_initial_jit - t_egraph_build).as_secs_f64() * 1000.0,
            (t_after_initial_jit - t_before_initial_jit).as_secs_f64() * 1000.0,
            (t_before_saturate - t_after_initial_jit).as_secs_f64() * 1000.0,
            (t_after_saturate - t_before_saturate).as_secs_f64() * 1000.0,
            (t_total - t_after_saturate).as_secs_f64() * 1000.0,
        );
    }

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
             \x20 initial expr: {}\n\
             \x20 final   expr: {}\n\
             \x20 steps: {} ({} matched)",
            initial_bench.output,
            final_bench.output,
            arena_to_kernel_code(&initial_arena, initial_root),
            arena_to_kernel_code(&final_arena, final_arena_root),
            steps.len(),
            steps.iter().filter(|s| s.matched).count(),
        );
        return None;
    }

    // 6. Build final EdgeAccumulator for extraction head training.
    // The extraction head needs: "this expression's structure → this expression's cost."
    // We provide two paired data points per trajectory:
    //   (initial_acc, initial_cost_ns) and (final_acc, final_cost_ns).
    // Build final accumulator using DAG-aware path: shared subexpressions get
    // (ref_count - 1) var_ref edges instead of duplicated subtree edges.
    // This matches what the extraction head evaluates during hill climbing.
    let final_acc = EdgeAccumulator::from_dag_choices(
        &egraph,
        root,
        &final_choices,
        &final_ref_counts,
        &model.embeddings,
    );
    let final_acc_vec = acc_to_vec(&final_acc);
    let final_edges = collect_edges_dedup_arena(&final_arena, final_arena_root);

    // 7. Log the rewrite for visibility
    let speedup = if final_cost_ns > 0.0 {
        initial_cost_ns / final_cost_ns
    } else {
        0.0
    };
    let matched = steps.iter().filter(|s| s.matched).count();
    eprintln!(
        "[REWRITE] {seed_name}: {speedup:.2}x ({initial_cost_ns:.1}ns -> {final_cost_ns:.1}ns) \
         [{matched}/{} steps]",
        steps.len(),
    );

    // 8. Return trajectory
    Some(Trajectory {
        trajectory_id,
        steps,
        initial_cost_ns,
        final_cost_ns,
        initial_cost: Some(initial_cost),
        final_cost: Some(best_cost),
        initial_nodes,
        node_budget,
        initial_accumulator_state: acc_vec.clone(),
        initial_edges: edges.clone(),
        final_accumulator_state: final_acc_vec,
        final_edges,
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
            let pair = generator.generate_arena();
            let traj_id = format!("seed_{seed}_t{i}");
            let name = format!("expr_{i}");
            (pair.arena, pair.unoptimized, name, traj_id)
        })
        .collect();

    if num_workers <= 1 || count <= 1 {
        // Sequential fast path: no thread overhead.
        let mut trajectories = Vec::new();
        for (arena, root, name, traj_id) in &work_items {
            if let Some(traj) = run_self_play_trajectory(
                arena,
                *root,
                name,
                model,
                &rule_embeds,
                rules,
                threshold,
                max_epochs,
                traj_id.clone(),
            ) {
                trajectories.push(traj);
            }
        }
        eprintln!(
            "Generated {}/{count} trajectories (sequential)",
            trajectories.len()
        );
        return trajectories;
    }

    // Parallel: partition work items across workers, spawn scoped threads.
    let num_rules = rules.len();
    let chunk_size = count.div_ceil(num_workers);
    let chunks: Vec<&[(ExprArena, ExprId, String, String)]> =
        work_items.chunks(chunk_size).collect();
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
                let model_ref = model;
                let rule_embeds_ref = &rule_embeds;

                std::thread::Builder::new()
                    .name(format!("traj-worker-{worker_id}"))
                    .stack_size(WORKER_STACK_SIZE)
                    .spawn_scoped(scope, move || {
                        // Each worker creates its own rules (avoids sharing Box<dyn Rewrite>
                        // through the hot loop — rules are used by EGraph internally).
                        let worker_rules: Vec<Box<dyn Rewrite>> = all_rules();
                        assert_eq!(
                            worker_rules.len(),
                            num_rules,
                            "Worker {worker_id}: rule count mismatch ({} vs {num_rules})",
                            worker_rules.len()
                        );

                        let mut results = Vec::with_capacity(chunk.len());
                        for (arena, root, name, traj_id) in chunk {
                            if let Some(traj) = run_self_play_trajectory(
                                arena,
                                *root,
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
// Trajectory I/O helpers
// ============================================================================

const TRAJ_MAGIC: &[u8; 8] = b"PFTJ0001";
const ADV_MAGIC: &[u8; 8] = b"PFAD0001";

fn write_u8<W: Write>(w: &mut W, v: u8) {
    w.write_all(&[v]).expect("write u8 failed");
}

fn write_u32<W: Write>(w: &mut W, v: u32) {
    w.write_all(&v.to_le_bytes()).expect("write u32 failed");
}

fn write_u64<W: Write>(w: &mut W, v: u64) {
    w.write_all(&v.to_le_bytes()).expect("write u64 failed");
}

fn write_i32<W: Write>(w: &mut W, v: i32) {
    w.write_all(&v.to_le_bytes()).expect("write i32 failed");
}

fn write_f32<W: Write>(w: &mut W, v: f32) {
    w.write_all(&v.to_le_bytes()).expect("write f32 failed");
}

fn write_f64<W: Write>(w: &mut W, v: f64) {
    w.write_all(&v.to_le_bytes()).expect("write f64 failed");
}

fn write_string<W: Write>(w: &mut W, s: &str) {
    write_u32(w, s.len() as u32);
    w.write_all(s.as_bytes()).expect("write string failed");
}

fn write_f32_vec<W: Write>(w: &mut W, values: &[f32]) {
    write_u32(w, values.len() as u32);
    for &v in values {
        write_f32(w, v);
    }
}

fn write_edges<W: Write>(w: &mut W, edges: &[(u8, u8, u16)]) {
    write_u32(w, edges.len() as u32);
    for &(parent, child, depth) in edges {
        write_u8(w, parent);
        write_u8(w, child);
        w.write_all(&depth.to_le_bytes())
            .expect("write edge depth failed");
    }
}

fn read_exact<const N: usize, R: std::io::Read>(r: &mut R) -> [u8; N] {
    let mut buf = [0u8; N];
    r.read_exact(&mut buf).expect("binary read failed");
    buf
}

fn read_u8<R: std::io::Read>(r: &mut R) -> u8 {
    read_exact::<1, _>(r)[0]
}

fn read_u32<R: std::io::Read>(r: &mut R) -> u32 {
    u32::from_le_bytes(read_exact(r))
}

fn read_u64<R: std::io::Read>(r: &mut R) -> u64 {
    u64::from_le_bytes(read_exact(r))
}

fn read_i32<R: std::io::Read>(r: &mut R) -> i32 {
    i32::from_le_bytes(read_exact(r))
}

fn read_f32<R: std::io::Read>(r: &mut R) -> f32 {
    f32::from_le_bytes(read_exact(r))
}

fn read_f64<R: std::io::Read>(r: &mut R) -> f64 {
    f64::from_le_bytes(read_exact(r))
}

fn read_string<R: std::io::Read>(r: &mut R) -> String {
    let len = read_u32(r) as usize;
    let mut bytes = vec![0u8; len];
    r.read_exact(&mut bytes).expect("read string failed");
    String::from_utf8(bytes).expect("binary trajectory id is not UTF-8")
}

fn read_f32_vec<R: std::io::Read>(r: &mut R) -> Vec<f32> {
    let len = read_u32(r) as usize;
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(read_f32(r));
    }
    values
}

fn read_edges<R: std::io::Read>(r: &mut R) -> Vec<(u8, u8, u16)> {
    let len = read_u32(r) as usize;
    let mut edges = Vec::with_capacity(len);
    for _ in 0..len {
        let parent = read_u8(r);
        let child = read_u8(r);
        let depth = u16::from_le_bytes(read_exact(r));
        edges.push((parent, child, depth));
    }
    edges
}

/// Write trajectories to the compact binary training format.
///
/// This is the live Rust↔critic IPC format.
pub fn write_trajectories_binary(trajectories: &[Trajectory], path: &Path) {
    let file = std::fs::File::create(path)
        .unwrap_or_else(|e| panic!("Failed to create {}: {e}", path.display()));
    let mut writer = BufWriter::new(file);
    writer
        .write_all(TRAJ_MAGIC)
        .unwrap_or_else(|e| panic!("Failed to write {}: {e}", path.display()));
    write_u32(&mut writer, trajectories.len() as u32);
    for traj in trajectories {
        write_string(&mut writer, &traj.trajectory_id);
        write_u32(&mut writer, traj.steps.len() as u32);
        write_f64(&mut writer, traj.initial_cost_ns);
        write_f64(&mut writer, traj.final_cost_ns);
        match traj.initial_cost {
            Some(v) => {
                write_u8(&mut writer, 1);
                write_f32(&mut writer, v);
            }
            None => write_u8(&mut writer, 0),
        }
        match traj.final_cost {
            Some(v) => {
                write_u8(&mut writer, 1);
                write_f32(&mut writer, v);
            }
            None => write_u8(&mut writer, 0),
        }
        write_u64(&mut writer, traj.initial_nodes as u64);
        write_u64(&mut writer, traj.node_budget as u64);
        write_f32_vec(&mut writer, &traj.initial_accumulator_state);
        write_edges(&mut writer, &traj.initial_edges);
        write_f32_vec(&mut writer, &traj.final_accumulator_state);
        write_edges(&mut writer, &traj.final_edges);

        for step in &traj.steps {
            write_f32_vec(&mut writer, &step.accumulator_state);
            write_f32_vec(&mut writer, &step.expression_embedding);
            write_f32_vec(&mut writer, &step.rule_embedding);
            write_i32(&mut writer, step.budget_remaining);
            write_i32(&mut writer, step.epochs_remaining);
            write_f32(&mut writer, step.action_probability);
            write_u8(&mut writer, u8::from(step.matched));
            write_f64(&mut writer, step.jit_cost_ns);
            write_edges(&mut writer, &step.edges);
            write_f32_vec(&mut writer, &step.graph_accumulator_state);
        }
    }
    writer
        .flush()
        .unwrap_or_else(|e| panic!("Failed to flush {}: {e}", path.display()));
}

/// Read trajectories from the compact binary training format.
pub fn load_trajectories_binary(path: &Path) -> Vec<Trajectory> {
    let mut reader = BufReader::new(
        std::fs::File::open(path)
            .unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display())),
    );
    let magic = read_exact::<8, _>(&mut reader);
    assert_eq!(
        &magic,
        TRAJ_MAGIC,
        "Invalid trajectory binary magic in {}",
        path.display()
    );
    let count = read_u32(&mut reader) as usize;
    let mut trajectories = Vec::with_capacity(count);
    for _ in 0..count {
        let trajectory_id = read_string(&mut reader);
        let step_count = read_u32(&mut reader) as usize;
        let initial_cost_ns = read_f64(&mut reader);
        let final_cost_ns = read_f64(&mut reader);
        let initial_cost = if read_u8(&mut reader) != 0 {
            Some(read_f32(&mut reader))
        } else {
            None
        };
        let final_cost = if read_u8(&mut reader) != 0 {
            Some(read_f32(&mut reader))
        } else {
            None
        };
        let initial_nodes = read_u64(&mut reader) as usize;
        let node_budget = read_u64(&mut reader) as usize;
        let initial_accumulator_state = read_f32_vec(&mut reader);
        let initial_edges = read_edges(&mut reader);
        let final_accumulator_state = read_f32_vec(&mut reader);
        let final_edges = read_edges(&mut reader);

        let mut steps = Vec::with_capacity(step_count);
        for _ in 0..step_count {
            steps.push(TrajectoryStep {
                accumulator_state: read_f32_vec(&mut reader),
                expression_embedding: read_f32_vec(&mut reader),
                rule_embedding: read_f32_vec(&mut reader),
                budget_remaining: read_i32(&mut reader),
                epochs_remaining: read_i32(&mut reader),
                action_probability: read_f32(&mut reader),
                matched: read_u8(&mut reader) != 0,
                jit_cost_ns: read_f64(&mut reader),
                edges: read_edges(&mut reader),
                graph_accumulator_state: read_f32_vec(&mut reader),
            });
        }
        trajectories.push(Trajectory {
            trajectory_id,
            steps,
            initial_cost_ns,
            final_cost_ns,
            initial_cost,
            final_cost,
            initial_nodes,
            node_budget,
            initial_accumulator_state,
            initial_edges,
            final_accumulator_state,
            final_edges,
        });
    }
    trajectories
}

/// Read per-trajectory advantage scores from the compact binary format.
pub fn read_advantages_binary(path: &Path) -> Vec<TrajectoryAdvantages> {
    let mut reader = BufReader::new(
        std::fs::File::open(path)
            .unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display())),
    );
    let magic = read_exact::<8, _>(&mut reader);
    assert_eq!(
        &magic,
        ADV_MAGIC,
        "Invalid advantage binary magic in {}",
        path.display()
    );
    let count = read_u32(&mut reader) as usize;
    let mut advantages = Vec::with_capacity(count);
    for _ in 0..count {
        let trajectory_idx = read_u64(&mut reader) as usize;
        let values = read_f32_vec(&mut reader);
        advantages.push(TrajectoryAdvantages {
            trajectory_idx,
            advantages: values,
        });
    }
    advantages
}

// ============================================================================
// Corpus loading and trajectory generation
// ============================================================================

/// Load expressions from binary corpus (`bench_corpus.bin`).
///
/// Returns up to `max_count` `(name, Expr)` pairs, sampled uniformly via
/// LCG shuffle. Expressions with >1000 nodes are filtered out.
///
/// # Panics
///
/// Panics if zero expressions load successfully from the file.
pub fn load_corpus_exprs(
    path: &Path,
    max_count: usize,
    seed: u64,
) -> Vec<(String, ExprArena, ExprId)> {
    let raw = corpus::read_corpus(path)
        .unwrap_or_else(|e| panic!("Failed to read binary corpus {}: {e}", path.display()));
    let total_entries = raw.len();

    let mut parsed: Vec<(String, ExprArena, ExprId)> = Vec::new();
    let mut skipped_large = 0u64;

    for (name, arena, root) in raw {
        if arena.len() > 1000 {
            skipped_large += 1;
            continue;
        }
        parsed.push((name, arena, root));
    }

    assert!(
        !parsed.is_empty(),
        "Zero expressions loaded from {} ({} entries, {skipped_large} oversized)",
        path.display(),
        total_entries
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
        "Loaded {} corpus expressions from {} ({} entries, {skipped_large} oversized)",
        parsed.len(),
        path.display(),
        total_entries
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
    corpus: &[(String, ExprArena, ExprId)],
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
    corpus: &[(String, ExprArena, ExprId)],
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
            let (ref name, ref arena, root) = corpus[*idx];
            match run_self_play_trajectory(
                arena,
                root,
                name,
                model,
                &rule_embeds,
                rules,
                threshold,
                max_epochs,
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
    let chunk_size = count.div_ceil(num_workers);
    let chunks: Vec<&[(usize, String)]> = work_items.chunks(chunk_size).collect();
    let actual_workers = chunks.len();

    eprintln!("[PARALLEL] Spawning {actual_workers} workers for {count} corpus trajectories");

    let mut all_trajectories: Vec<Trajectory> = Vec::new();

    std::thread::scope(|scope| {
        let handles: Vec<_> = chunks
            .into_iter()
            .enumerate()
            .map(|(worker_id, chunk)| {
                let model_ref = model;
                let rule_embeds_ref = &rule_embeds;
                let corpus_ref = corpus;

                std::thread::Builder::new()
                    .name(format!("corpus-worker-{worker_id}"))
                    .stack_size(WORKER_STACK_SIZE)
                    .spawn_scoped(scope, move || {
                        let worker_rules: Vec<Box<dyn Rewrite>> = all_rules();
                        assert_eq!(
                            worker_rules.len(),
                            num_rules,
                            "Worker {worker_id}: rule count mismatch ({} vs {num_rules})",
                            worker_rules.len()
                        );

                        let mut results = Vec::with_capacity(chunk.len());
                        for (idx, traj_id) in chunk {
                            let (ref name, ref arena, root) = corpus_ref[*idx];
                            match run_self_play_trajectory(
                                arena,
                                root,
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
                    .unwrap_or_else(|e| panic!("Failed to spawn corpus worker {worker_id}: {e}"))
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
    fn trajectory_binary_round_trip() {
        let traj = Trajectory {
            trajectory_id: "test".into(),
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
            initial_nodes: 1,
            node_budget: 100,
            initial_accumulator_state: vec![],
            initial_edges: vec![],
            final_accumulator_state: vec![],
            final_edges: vec![],
        };

        let tmp = std::env::temp_dir().join("test_self_play_traj.pftraj");
        write_trajectories_binary(&[traj], &tmp);

        let read_back = load_trajectories_binary(&tmp);
        assert_eq!(read_back.len(), 1);
        let back = &read_back[0];
        assert_eq!(back.trajectory_id, "test");
        assert_eq!(back.steps.len(), 1);
        assert!(back.steps[0].matched);
        assert!((back.final_cost_ns - 1.0).abs() < 1e-6);
        assert!((back.initial_cost.unwrap() - 5.0).abs() < 1e-6);

        std::fs::remove_file(&tmp)
            .unwrap_or_else(|e| panic!("Failed to remove temp file {}: {e}", tmp.display()));
    }

    #[test]
    fn advantages_binary_round_trip() {
        let tmp = std::env::temp_dir().join("test_self_play_adv.pfadv");
        {
            let file = std::fs::File::create(&tmp)
                .unwrap_or_else(|e| panic!("Failed to create {}: {e}", tmp.display()));
            let mut w = BufWriter::new(file);
            w.write_all(ADV_MAGIC).unwrap();
            write_u32(&mut w, 1);
            write_u64(&mut w, 0);
            write_f32_vec(&mut w, &[0.1, -0.2, 0.3]);
            w.flush().unwrap_or_else(|e| panic!("Failed to flush: {e}"));
        }

        let read_back = read_advantages_binary(&tmp);
        assert_eq!(read_back.len(), 1);
        assert_eq!(read_back[0].advantages.len(), 3);
        assert!((read_back[0].advantages[0] - 0.1).abs() < 1e-6);
        assert!((read_back[0].advantages[1] - (-0.2)).abs() < 1e-6);
        assert!((read_back[0].advantages[2] - 0.3).abs() < 1e-6);

        std::fs::remove_file(&tmp)
            .unwrap_or_else(|e| panic!("Failed to remove temp file {}: {e}", tmp.display()));
    }

    #[test]
    fn sigmoid_basic_values() {
        assert!(
            (sigmoid(0.0) - 0.5).abs() < 1e-6,
            "sigmoid(0) should be 0.5"
        );
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
        // Scalar features are appended after the `GRAPH_ACC_DIM`-long values block.
        assert!((v[GRAPH_ACC_DIM] - 5.0).abs() < 1e-9, "edge_count mismatch");
        assert!(
            (v[GRAPH_ACC_DIM + 1] - 8.0).abs() < 1e-9,
            "node_count mismatch"
        );
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
