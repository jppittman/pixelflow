//! # Episode Generation
//!
//! Runs budget-bounded e-graph saturation episodes and JIT-benchmarks the
//! initial and final (post-saturation) forms of a seed expression.
//!
//! This is the surviving core of what was `self_play.rs`: the RL policy
//! (mask-head rule scoring, threshold-gated rule approval, per-step
//! Trajectory/PFTJ export for the transformer critic) has been deleted per
//! docs/plans/2026-07-07-guided-saturation-redesign.md — that estimator was
//! methodologically unsound and its policy was never consumed by the
//! compiler. What remains — expression building, budget-bounded saturation,
//! JIT benchmarking of initial/final forms, semantic-equivalence checking —
//! is deliberately kept simple and synchronous: **the seed of the future
//! episode collector, not the collector itself** (mirroring
//! `pixelflow_search::egraph::labeler::run_episode`, which plays the same
//! role one layer down — at the e-graph/provenance level, without JIT
//! measurement).

use std::path::Path;

use pixelflow_ir::{ExprArena, ExprId};
use pixelflow_search::egraph::saturate::saturate_with_full_budget;
use pixelflow_search::egraph::{
    EGraph, IncrementalExtractor, Rewrite, all_rules, compute_ref_counts, extract_neural_to_arena,
};
use pixelflow_search::nnue::ExprNnue;
use pixelflow_search::nnue::RuleTemplates;

use crate::jit_bench::benchmark_jit_arena;
use crate::training::corpus;
use crate::training::factored::arena_to_kernel_code;

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
// Episode outcome
// ============================================================================

/// Outcome of one budget-bounded saturation episode: the seed expression's
/// JIT-benchmarked cost before and after saturation.
///
/// No policy, no critic, no per-step export payload — just the measured
/// facts an episode collector needs: what the e-graph achieved within
/// budget, ground-truth JIT cost at both ends, and a proof the rewrite
/// preserved semantics.
pub struct Episode {
    /// Human-readable identifier for this episode (propagated from the caller).
    pub episode_id: String,
    /// Name of the seed expression (for logging).
    pub seed_name: String,
    /// The seed expression, unmodified.
    pub initial_arena: ExprArena,
    pub initial_root: ExprId,
    /// JIT-benchmarked cost of the seed expression (nanoseconds).
    pub initial_cost_ns: f64,
    /// The expression extracted after budget-bounded saturation.
    pub final_arena: ExprArena,
    pub final_root: ExprId,
    /// JIT-benchmarked cost of the extracted expression (nanoseconds).
    pub final_cost_ns: f64,
    /// The e-class budget saturation was run under.
    pub node_budget: usize,
    /// The iteration budget saturation was run under.
    pub epoch_budget: usize,
}

/// Run a single budget-bounded saturation episode.
///
/// Returns `None` if the seed or extracted expression cannot be JIT-benchmarked,
/// is degenerate (NaN/Inf constants), or if the rewrite fails the
/// semantic-equivalence check.
///
/// # Algorithm
///
/// 1. Insert the seed expression into an e-graph with all rewrite rules.
/// 2. JIT-benchmark the seed for ground-truth initial cost.
/// 3. Saturate within a size/iteration budget (randomized per episode, scaled
///    by expression size — see [`saturate_with_full_budget`]). No rule
///    filtering: every rule fires whenever it matches, same as production
///    `saturate()`, just budget-capped for training-corpus diversity.
/// 4. Extract the best expression the budget-bounded e-graph achieved.
/// 5. JIT-benchmark the extraction for ground-truth final cost.
/// 6. Verify the rewrite preserved semantics (SIMD-lane equivalence at a
///    fixed test point).
pub fn run_episode(
    seed_arena: &ExprArena,
    seed_root: ExprId,
    seed_name: &str,
    model: &ExprNnue,
    max_epochs: usize,
    episode_id: String,
) -> Option<Episode> {
    // Hard wall-clock deadline per episode: safety net against runaway extraction.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let episode_start = std::time::Instant::now();

    // 1. Convert seed expression to e-graph
    let mut egraph = EGraph::with_rules(all_rules());
    let root = egraph.add_arena(seed_arena, seed_root);

    // 2. Extract + JIT-benchmark the initial expression
    let (initial_arena, initial_root, _initial_cost) =
        extract_neural_to_arena(&egraph, root, model);

    if initial_arena.has_degenerate(initial_root) {
        eprintln!("Skipping degenerate seed expression in {episode_id} (seed={seed_name})");
        return None;
    }
    let initial_bench = match benchmark_jit_arena(&initial_arena, initial_root) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "JIT bench failed for initial expr in episode {episode_id} (seed={seed_name}): {e}"
            );
            return None;
        }
    };

    // Domain check: if any SIMD lane is NaN or Inf at the test point, the
    // expression is undefined there. Rewrite equivalence checks would always
    // fire false positives (REWRITE BUG) because IEEE 754 NaN/Inf behavior
    // diverges across mathematically-equivalent forms. Skip early.
    if initial_bench.output.iter().any(|x| !x.is_finite()) {
        eprintln!("[SKIP] {seed_name}: initial output contains NaN/Inf, skipping");
        return None;
    }

    let initial_cost_ns = initial_bench.ns;

    // 3. Randomize resource constraints per episode, scaled by expression size.
    // Larger expressions need more budget to explore meaningful rewrites.
    // Budget = base + multiplier * node_count, with random multiplier.
    let episode_hash: usize = episode_id
        .bytes()
        .fold(0usize, |h, b| h.wrapping_mul(31).wrapping_add(b as usize));
    let initial_nodes = initial_arena.node_count_subtree(initial_root);
    // multiplier in [3, 10]: small exprs get 50-200, large (100-node) get 300-1000
    let budget_mult = 3 + (episode_hash % 8);
    let node_budget = (50 + budget_mult * initial_nodes).min(2000); // floor 50, cap 2000
    let epoch_budget = 10 + (episode_hash.wrapping_mul(7) % 51); // [10, 60]
    let effective_epochs = max_epochs.min(epoch_budget);

    // 4. Saturate within budget. No rule filtering — every rule fires
    // whenever it matches (same rewrite semantics as production `saturate()`),
    // just capped by node/iteration budget so episodes stay bounded and diverse.
    let remaining = deadline.saturating_duration_since(std::time::Instant::now());
    let sat_result =
        saturate_with_full_budget(&mut egraph, effective_epochs, node_budget, remaining);

    // 5. Extract + JIT-benchmark the final expression
    let post_sat_nodes = egraph.node_count();
    if post_sat_nodes > 5000 {
        eprintln!(
            "Episode {episode_id} egraph too large ({post_sat_nodes} nodes, budget={node_budget}), skipping"
        );
        return None;
    }
    if std::time::Instant::now() > deadline {
        eprintln!(
            "Episode {episode_id} hit deadline before extraction (seed={seed_name}, nodes={post_sat_nodes})"
        );
        return None;
    }
    let extractor = IncrementalExtractor::new(model, 8);
    let (_final_cost, final_choices) = extractor.extract_choices_only(&egraph, root);
    let _final_ref_counts = compute_ref_counts(&egraph, root, &final_choices);
    let (final_arena, final_arena_root) =
        pixelflow_search::egraph::choices_to_arena(&egraph, root, &final_choices);

    if final_arena.has_degenerate(final_arena_root) {
        eprintln!("Rewrite produced degenerate expression in {episode_id} (seed={seed_name})");
        return None;
    }
    let final_bench = match benchmark_jit_arena(&final_arena, final_arena_root) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("JIT bench failed for episode {episode_id} (seed={seed_name}): {e}");
            return None;
        }
    };
    let final_cost_ns = final_bench.ns;

    if episode_start.elapsed().as_millis() > 100 {
        eprintln!(
            "[SLOW_EPISODE] {episode_id} (seed={seed_name}) total={:.1}ms: \
             iterations={} unions={} saturated={} nodes={post_sat_nodes}",
            episode_start.elapsed().as_secs_f64() * 1000.0,
            sat_result.iterations,
            sat_result.total_unions,
            sat_result.saturated,
        );
    }

    // 6. Correctness check: rewrites must preserve semantics.
    const EQUIV_EPSILON: f32 = 1e-3;
    if let Err(max_diff) = initial_bench.check_equivalence(&final_bench, EQUIV_EPSILON) {
        eprintln!(
            "REWRITE BUG: episode {episode_id} (seed={seed_name})\n\
             \x20 inputs: x=0.5 y=0.7 z=1.3 w=-0.2\n\
             \x20 initial output: {:?}\n\
             \x20 final   output: {:?}\n\
             \x20 max_diff={max_diff:.6} epsilon={EQUIV_EPSILON}\n\
             \x20 initial expr: {}\n\
             \x20 final   expr: {}",
            initial_bench.output,
            final_bench.output,
            arena_to_kernel_code(&initial_arena, initial_root),
            arena_to_kernel_code(&final_arena, final_arena_root),
        );
        return None;
    }

    let speedup = if final_cost_ns > 0.0 {
        initial_cost_ns / final_cost_ns
    } else {
        0.0
    };
    eprintln!("[REWRITE] {seed_name}: {speedup:.2}x ({initial_cost_ns:.1}ns -> {final_cost_ns:.1}ns)");

    Some(Episode {
        episode_id,
        seed_name: seed_name.to_string(),
        initial_arena,
        initial_root,
        initial_cost_ns,
        final_arena,
        final_root: final_arena_root,
        final_cost_ns,
        node_budget,
        epoch_budget,
    })
}

// ============================================================================
// Corpus loading
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::OpKind;

    /// End-to-end smoke test: a tiny seed expression survives a full episode
    /// (extract -> JIT bench -> saturate under budget -> extract -> JIT bench
    /// -> equivalence check) without panicking, and returns sane costs.
    #[test]
    fn run_episode_smoke() {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let one = arena.push_const(1.0);
        let mul = arena.push_binary(OpKind::Mul, x, one); // x * 1.0 -> simplifies to x
        let root = arena.push_binary(OpKind::Add, mul, x); // (x * 1.0) + x

        let model = ExprNnue::new();
        let episode = run_episode(&arena, root, "smoke_seed", &model, 10, "smoke_0".to_string())
            .expect("episode should complete for a small, well-defined seed expression");

        assert!(
            episode.initial_cost_ns.is_finite() && episode.initial_cost_ns >= 0.0,
            "initial_cost_ns should be a finite non-negative measurement, got {}",
            episode.initial_cost_ns
        );
        assert!(
            episode.final_cost_ns.is_finite() && episode.final_cost_ns >= 0.0,
            "final_cost_ns should be a finite non-negative measurement, got {}",
            episode.final_cost_ns
        );
        assert!(episode.node_budget >= 50, "node_budget should respect the floor");
        assert!(episode.epoch_budget >= 10, "epoch_budget should respect the floor");
    }

    #[test]
    fn build_rule_templates_covers_all_rules() {
        let rules = all_rules();
        let templates = build_rule_templates(&rules);
        assert_eq!(
            templates.len(),
            rules.len(),
            "every rule should get a template entry"
        );
    }
}
