#![allow(warnings)]
//! # Unified Self-Play Training Binary
//!
//! Orchestrates the outer training loop:
//!
//! ```text
//! GENERATE → EXPORT → CRITIQUE → UPDATE → CHECKPOINT
//! ```
//!
//! This is the single binary that replaces `train_online`, `collect_guide_data`,
//! `train_mask_reinforce`, and `gen_mask_data`. It runs an AlphaZero-style
//! training loop:
//!
//! 1. **GENERATE**: Self-play trajectories via e-graph hill-climbing with the
//!    current mask head.
//! 2. **EXPORT**: Write trajectory JSONL for the Python Critic.
//! 3. **CRITIQUE**: Python Causal Transformer Critic assigns per-step temporal
//!    credit (advantages A_t = R_T - V_t).
//! 4. **UPDATE**: Joint policy + value gradient update through the shared
//!    ExprNnue backbone using REINFORCE + MSE.
//! 5. **CHECKPOINT**: Save model weights and log metrics.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin train_unified -- \
//!   --rounds 30 --trajectories-per-round 50
//! ```

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;
use pixelflow_search::egraph::{all_rules, Rewrite};
use pixelflow_search::nnue::factored::{
    EdgeAccumulator, ExprNnue, GraphAccumulator, EMBED_DIM, GRAPH_ACC_DIM, GRAPH_INPUT_DIM,
    INPUT_DIM, K,
};

use pixelflow_pipeline::training::gen_es::{log_ns, GenEs, GenEsConfig};
use pixelflow_pipeline::training::self_play::{
    build_rule_templates, generate_corpus_trajectories_parallel,
    generate_trajectory_batch_parallel, load_corpus_exprs, load_trajectories_jsonl,
    read_advantages_jsonl, write_trajectories_jsonl,
};
use pixelflow_pipeline::training::unified::{Trajectory, TrajectoryAdvantages, TrajectoryStep};
use pixelflow_pipeline::training::unified_backward::{
    apply_unified_sgd, backward_policy, backward_value,
    backward_through_accumulator, compute_d_acc_input_value,
    forward_cached, UnifiedGradients,
};

// ============================================================================
// Memory observability
// ============================================================================

/// Get current process RSS in megabytes via `ps`. Works on macOS and Linux.
/// Returns 0.0 if the measurement fails (no silent crash).
fn rss_mb() -> f64 {
    let pid = std::process::id();
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|kb| kb as f64 / 1024.0)
        .unwrap_or(0.0)
}

// ============================================================================
// CLI
// ============================================================================

/// Unified self-play training: joint value + policy through shared backbone.
///
/// Outer loop: GENERATE -> EXPORT -> CRITIQUE -> UPDATE -> CHECKPOINT
#[derive(Parser, Debug)]
#[command(name = "train_unified")]
#[command(about = "Unified self-play training with temporal credit assignment")]
struct Args {
    /// Number of outer training rounds.
    #[arg(long, default_value_t = 30)]
    rounds: usize,

    /// Trajectories per round.
    #[arg(long, default_value_t = 50)]
    trajectories_per_round: usize,

    /// Max hill-climbing steps per trajectory.
    #[arg(long, default_value_t = 50)]
    max_steps: usize,

    /// Mask threshold (sigmoid > threshold to apply rule).
    #[arg(long, default_value_t = 0.3)]
    threshold: f32,

    /// Learning rate for SGD.
    #[arg(long, default_value_t = 8.17e-3)]
    lr: f32,

    /// Momentum for SGD.
    #[arg(long, default_value_t = 0.891)]
    momentum: f32,

    /// Weight decay.
    #[arg(long, default_value_t = 5.34e-6)]
    weight_decay: f32,

    /// Value loss coefficient.
    #[arg(long, default_value_t = 1.175)]
    value_coeff: f32,

    /// Gradient clipping threshold (global L2 norm).
    #[arg(long, default_value_t = 8.74)]
    grad_clip: f32,

    /// Entropy bonus coefficient (prevents policy collapse).
    #[arg(long, default_value_t = 0.098)]
    entropy_coeff: f32,

    /// Minimum entropy coefficient (floor for annealing — never goes below this).
    #[arg(long, default_value_t = 0.02)]
    entropy_floor: f32,

    /// Miss penalty: scales down advantage for steps where the rule didn't match.
    /// Lower values encourage exploration (0.0 = no penalty for misses, 1.0 = full penalty).
    #[arg(long, default_value_t = 0.1)]
    miss_penalty: f32,

    /// Path to model weights (loaded at start, saved at end).
    #[arg(long, default_value = "pixelflow-pipeline/data/judge.bin")]
    model: PathBuf,

    /// Output directory for checkpoints, trajectories, advantages.
    #[arg(long, default_value = "pixelflow-pipeline/data/unified")]
    output_dir: PathBuf,

    /// Path to Python critic script.
    #[arg(long, default_value = "pixelflow-pipeline/scripts/critic.py")]
    critic_script: PathBuf,

    /// Critic checkpoint path (reused across rounds).
    #[arg(long, default_value = "pixelflow-pipeline/data/unified/critic.pt")]
    critic_checkpoint: PathBuf,

    /// Critic training epochs per round.
    #[arg(long, default_value_t = 80)]
    critic_epochs: usize,

    /// Critic learning rate (forwarded to critic.py --lr).
    #[arg(long, default_value_t = 1.66e-4)]
    critic_lr: f64,

    /// Critic dropout (forwarded to critic.py --dropout).
    #[arg(long, default_value_t = 0.124)]
    critic_dropout: f64,

    /// Random seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    // ── ES-guided generation ─────────────────────────────────────
    /// ES population size (perturbation candidates per round).
    #[arg(long, default_value_t = 10)]
    es_population: usize,

    /// ES noise standard deviation.
    #[arg(long, default_value_t = 0.1)]
    es_sigma: f32,

    /// ES learning rate.
    #[arg(long, default_value_t = 0.05)]
    es_alpha: f32,

    /// Fraction of trajectories per round sampled from corpus (0.0–1.0).
    #[arg(long, default_value_t = 0.3)]
    corpus_fraction: f32,

    /// Path to bench_corpus.jsonl for corpus-based trajectories.
    #[arg(long, default_value = "pixelflow-pipeline/data/bench_corpus.jsonl")]
    corpus_path: PathBuf,

    /// Max corpus expressions to hold in memory.
    #[arg(long, default_value_t = 2000)]
    corpus_max: usize,

    // ── Replay buffer ───────────────────────────────────────────
    /// Replay buffer capacity (max stored steps across rounds).
    #[arg(long, default_value_t = 200000)]
    replay_capacity: usize,

    /// Re-label entire replay buffer every N rounds (0 = never).
    #[arg(long, default_value_t = 5)]
    relabel_interval: usize,

    /// Mini-batch size for SGD updates from replay buffer.
    #[arg(long, default_value_t = 2048)]
    mini_batch_size: usize,

    /// Number of gradient update steps per round.
    #[arg(long, default_value_t = 19)]
    updates_per_round: usize,

    /// Offline mode: skip self-play, load existing trajectory JSONL from output_dir,
    /// train critic once, then do pure SGD rounds on the replay buffer.
    /// Ideal for Optuna hyperparameter sweeps.
    #[arg(long, default_value_t = false)]
    offline: bool,

    /// Directory containing existing trajectory JSONL files to load in offline mode.
    /// Defaults to --output-dir if not set.
    #[arg(long)]
    trajectory_dir: Option<PathBuf>,

    // ── Parallelism ────────────────────────────────────────────
    /// Number of worker threads for parallel trajectory generation.
    /// Defaults to the number of available CPU cores.
    /// Set to 1 for sequential execution (useful for debugging).
    #[arg(long)]
    workers: Option<usize>,
}

// ============================================================================
// Helpers
// ============================================================================

/// Reconstruct [`EdgeAccumulator`] from a [`TrajectoryStep`]'s accumulator_state.
///
/// Layout: `[values (4*K=128 floats), edge_count, node_count]` = 130 floats total.
fn acc_from_step(step: &TrajectoryStep) -> EdgeAccumulator {
    assert_eq!(
        step.accumulator_state.len(),
        INPUT_DIM,
        "Expected {} accumulator values, got {}",
        INPUT_DIM,
        step.accumulator_state.len()
    );

    let mut acc = EdgeAccumulator::default();
    acc.values.copy_from_slice(&step.accumulator_state[..4 * K]);
    acc.edge_count = step.accumulator_state[4 * K] as u32;
    acc.node_count = step.accumulator_state[4 * K + 1] as u32;
    acc
}

/// Reconstruct [`EdgeAccumulator`] from a [`ReplayStep`].
fn acc_from_replay(step: &ReplayStep) -> EdgeAccumulator {
    assert_eq!(step.acc.len(), INPUT_DIM,
        "Expected {} accumulator values in replay step, got {}", INPUT_DIM, step.acc.len());
    let mut acc = EdgeAccumulator::default();
    acc.values.copy_from_slice(&step.acc[..4 * K]);
    acc.edge_count = step.acc[4 * K] as u32;
    acc.node_count = step.acc[4 * K + 1] as u32;
    acc
}

/// Extract rule embedding from a [`ReplayStep`].
fn embed_from_replay(step: &ReplayStep) -> [f32; EMBED_DIM] {
    assert_eq!(step.rule_embed.len(), EMBED_DIM,
        "Expected {} rule embedding dims in replay step, got {}", EMBED_DIM, step.rule_embed.len());
    let mut embed = [0.0f32; EMBED_DIM];
    embed.copy_from_slice(&step.rule_embed);
    embed
}

/// Reconstruct [`GraphAccumulator`] from a [`ReplayStep`].
fn gacc_from_replay(step: &ReplayStep) -> GraphAccumulator {
    let mut gacc = GraphAccumulator::new();
    if step.graph_acc.len() == GRAPH_INPUT_DIM {
        gacc.values.copy_from_slice(&step.graph_acc[..GRAPH_ACC_DIM]);
        gacc.edge_count = step.graph_acc[GRAPH_ACC_DIM] as u32;
        gacc.node_count = step.graph_acc[GRAPH_ACC_DIM + 1] as u32;
    }
    // If graph_acc is empty (old replay data), return zeroed accumulator.
    // The graph backbone will produce a zero-centered embedding, which is fine
    // for backward compat — the mask gradient will be small but non-zero.
    gacc
}

// ============================================================================
// Replay Buffer
// ============================================================================

struct ReplayStep {
    acc: Vec<f32>,           // 130 floats (INPUT_DIM)
    graph_acc: Vec<f32>,     // 98 floats (GRAPH_INPUT_DIM) — VSA graph state
    expr_embed: Vec<f32>,    // 32 floats (EMBED_DIM) — expr_proj output at decision time
    rule_embed: Vec<f32>,    // 32 floats (EMBED_DIM)
    edges: Vec<(u8, u8, u16)>, // Edge list for embedding gradient flow
    matched: bool,
    jit_cost_ns: f64,
    advantage: f32,
}

struct ReplayBuffer {
    steps: Vec<ReplayStep>,
    max_steps: usize,
}

impl ReplayBuffer {
    fn new(capacity: usize) -> Self {
        Self { steps: Vec::new(), max_steps: capacity }
    }

    /// Flatten trajectory steps + advantages into ReplaySteps and append.
    fn push_round(
        &mut self,
        trajectories: &[Trajectory],
        advantages: &[TrajectoryAdvantages],
    ) {
        for (traj, adv) in trajectories.iter().zip(advantages.iter()) {
            assert_eq!(traj.steps.len(), adv.advantages.len(),
                "Step/advantage count mismatch in trajectory {}: {} vs {}",
                traj.trajectory_id, traj.steps.len(), adv.advantages.len());
            for (step, &advantage) in traj.steps.iter().zip(adv.advantages.iter()) {
                self.steps.push(ReplayStep {
                    acc: step.accumulator_state.clone(),
                    graph_acc: step.graph_accumulator_state.clone(),
                    expr_embed: step.expression_embedding.clone(),
                    rule_embed: step.rule_embedding.clone(),
                    edges: step.edges.clone(),
                    matched: step.matched,
                    jit_cost_ns: step.jit_cost_ns,
                    advantage,
                });
            }
        }
        self.prune();
    }

    /// FIFO evict oldest steps when over capacity.
    fn prune(&mut self) {
        if self.steps.len() > self.max_steps {
            let excess = self.steps.len() - self.max_steps;
            self.steps.drain(..excess);
        }
    }

    fn len(&self) -> usize {
        self.steps.len()
    }

    /// Sample batch_size random indices using LCG PRNG.
    fn sample_batch(&self, batch_size: usize, seed: u64) -> Vec<usize> {
        let n = self.steps.len();
        assert!(n > 0, "Cannot sample from empty replay buffer");
        let batch_size = batch_size.min(n);
        let mut indices = Vec::with_capacity(batch_size);
        let mut state = seed;
        for _ in 0..batch_size {
            // LCG: state = state * 6364136223846793005 + 1442695040888963407
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            indices.push((state >> 33) as usize % n);
        }
        indices
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    // E-graph expressions can get deep; default 8MB stack isn't enough.
    let builder = std::thread::Builder::new().stack_size(64 * 1024 * 1024);
    let handler = builder
        .spawn(real_main)
        .expect("failed to spawn main thread with larger stack");
    handler.join().expect("main thread panicked");
}

fn real_main() {
    let args = Args::parse();

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)
        .unwrap_or_else(|e| panic!("Failed to create output dir {:?}: {e}", args.output_dir));

    // Load model or initialize fresh
    let mut model = if args.model.exists() {
        let m = ExprNnue::load(&args.model)
            .unwrap_or_else(|e| panic!("Failed to load model from {:?}: {e}", args.model));
        eprintln!("Loaded model from {:?}", args.model);
        m
    } else {
        eprintln!("No model at {:?}, initializing fresh with seed {}", args.model, args.seed);
        ExprNnue::new_with_latency_prior(args.seed)
    };

    // ── OFFLINE MODE ─────────────────────────────────────────────
    if args.offline {
        run_offline(&args, &mut model);
        return;
    }

    // Report parallelism
    let effective_workers = args.workers.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    eprintln!("Workers: {effective_workers} ({})",
        if args.workers.is_some() { "explicit" } else { "auto-detected" });

    // Build rules + templates
    let rules: Vec<Box<dyn Rewrite>> = all_rules();
    let templates = build_rule_templates(&rules);

    // Initialize momentum buffer
    let mut momentum_buf = UnifiedGradients::zero();

    // Initialize ES optimizer
    let mut gen_es = GenEs::new(GenEsConfig {
        sigma: args.es_sigma,
        alpha: args.es_alpha,
        population: args.es_population,
        samples_per_candidate: 8,
        seed: args.seed.wrapping_add(0xE5),
    }, templates.clone());

    // Load corpus expressions
    let corpus_exprs = if args.corpus_fraction > 0.0 {
        load_corpus_exprs(&args.corpus_path, args.corpus_max, args.seed)
    } else {
        eprintln!("Corpus fraction is 0.0, skipping corpus loading");
        Vec::new()
    };

    // Initialize replay buffer
    let mut replay_buffer = ReplayBuffer::new(args.replay_capacity);

    // Metrics log
    let metrics_path = args.output_dir.join("metrics.jsonl");
    let mut metrics_file = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&metrics_path)
            .unwrap_or_else(|e| panic!("Failed to open metrics file {:?}: {e}", metrics_path)),
    );

    for round in 0..args.rounds {
        let round_start = std::time::Instant::now();
        let round_seed = args.seed.wrapping_add(round as u64 * 1000);

        eprintln!("\n{}", "=".repeat(60));
        eprintln!("Round {}/{}", round + 1, args.rounds);
        eprintln!("{}", "=".repeat(60));
        eprintln!("[MEMORY] round {} start: {:.0}MB", round, rss_mb());

        // ── PHASE 1: GENERATE ──────────────────────────────────────────
        // ES step: adapt generator toward judge's blind spots
        let gen_config = gen_es.step(&model);
        let es_fitness = gen_es.last_fitness();
        eprintln!(
            "[ES] depth={} leaf={:.2} vars={} fused={:.2} junkify={}/{:.2} fitness={:.4}",
            gen_config.max_depth, gen_config.leaf_prob, gen_config.num_vars,
            gen_config.fused_op_prob, gen_config.max_junkify_passes, gen_config.junkify_prob,
            es_fitness,
        );

        // Split trajectories between ES-generated and corpus-sampled
        let corpus_count = (args.trajectories_per_round as f32 * args.corpus_fraction) as usize;
        let es_count = args.trajectories_per_round - corpus_count;

        eprintln!(
            "[GENERATE] Running {} ES + {} corpus trajectories...",
            es_count, corpus_count,
        );
        let gen_start = std::time::Instant::now();

        let mut trajectories = generate_trajectory_batch_parallel(
            &model,
            &templates,
            &rules,
            es_count,
            round_seed,
            args.threshold,
            args.max_steps,
            &gen_config,
            args.workers,
        );

        if corpus_count > 0 && !corpus_exprs.is_empty() {
            trajectories.extend(generate_corpus_trajectories_parallel(
                &model,
                &templates,
                &rules,
                &corpus_exprs,
                corpus_count,
                round_seed.wrapping_add(0xC0),
                args.threshold,
                args.max_steps,
                args.workers,
            ));
        }
        let gen_elapsed = gen_start.elapsed();

        // Filter out zero-step trajectories (expressions where no rules matched)
        let pre_filter = trajectories.len();
        let mut trajectories: Vec<_> = trajectories
            .into_iter()
            .filter(|t| !t.steps.is_empty())
            .collect();
        let empty_count = pre_filter - trajectories.len();

        if trajectories.is_empty() {
            eprintln!("[GENERATE] No valid trajectories generated, skipping round");
            continue;
        }

        // Replace NaN/non-finite jit_cost_ns with a penalty cost instead of dropping trajectories
        let max_cost_ns = trajectories.iter()
            .flat_map(|t| t.steps.iter())
            .filter_map(|s| if s.jit_cost_ns.is_finite() { Some(s.jit_cost_ns) } else { None })
            .fold(0.0f64, f64::max);
        let penalty_cost_ns = (max_cost_ns * 2.0).max(100.0);

        let mut nan_replaced = 0usize;
        for traj in &mut trajectories {
            for step in &mut traj.steps {
                if !step.jit_cost_ns.is_finite() || step.jit_cost_ns < 0.0 {
                    step.jit_cost_ns = penalty_cost_ns;
                    nan_replaced += 1;
                }
            }
        }
        if nan_replaced > 0 {
            eprintln!("[GENERATE] Replaced {nan_replaced} NaN jit_cost_ns with penalty={penalty_cost_ns:.1}ns");
        }

        if empty_count > 0 {
            eprintln!(
                "[GENERATE] Filtered {empty_count} empty out of {pre_filter} trajectories",
            );
        }

        eprintln!("[MEMORY] after generate: {:.0}MB", rss_mb());

        let total_steps: usize = trajectories.iter().map(|t| t.steps.len()).sum();
        eprintln!(
            "[GENERATE] {} trajectories, {} total steps in {:.1}s",
            trajectories.len(),
            total_steps,
            gen_elapsed.as_secs_f64()
        );

        // ── PHASE 2: EXPORT ────────────────────────────────────────────
        let traj_path = args.output_dir.join(format!("trajectories_r{round}.jsonl"));
        write_trajectories_jsonl(&trajectories, &traj_path);
        eprintln!("[EXPORT] Wrote {}", traj_path.display());

        // ── PHASE 3: CRITIQUE ──────────────────────────────────────────
        let adv_path = args.output_dir.join(format!("advantages_r{round}.jsonl"));
        eprintln!("[CRITIQUE] Running Python critic...");
        let critic_start = std::time::Instant::now();

        let critic_script_str = args
            .critic_script
            .to_str()
            .unwrap_or_else(|| panic!("Invalid UTF-8 in critic script path: {:?}", args.critic_script));
        let traj_path_str = traj_path
            .to_str()
            .unwrap_or_else(|| panic!("Invalid UTF-8 in trajectory path: {:?}", traj_path));
        let adv_path_str = adv_path
            .to_str()
            .unwrap_or_else(|| panic!("Invalid UTF-8 in advantages path: {:?}", adv_path));
        let critic_ckpt_str = args
            .critic_checkpoint
            .to_str()
            .unwrap_or_else(|| panic!("Invalid UTF-8 in critic checkpoint path: {:?}", args.critic_checkpoint));

        let status = std::process::Command::new("uv")
            .args(["run", critic_script_str])
            .arg("train")
            .args(["--input", traj_path_str])
            .args(["--output", adv_path_str])
            .args(["--checkpoint", critic_ckpt_str])
            .args(["--epochs", &args.critic_epochs.to_string()])
            .args(["--lr", &args.critic_lr.to_string()])
            .args(["--dropout", &args.critic_dropout.to_string()])
            .status()
            .unwrap_or_else(|e| panic!("Failed to run critic (uv run {:?} train): {e}", args.critic_script));

        if !status.success() {
            panic!(
                "Critic failed with exit code: {} (script: {:?})",
                status, args.critic_script
            );
        }
        let critic_elapsed = critic_start.elapsed();
        eprintln!("[CRITIQUE] Done in {:.1}s", critic_elapsed.as_secs_f64());

        // ── PHASE 4: UPDATE ────────────────────────────────────────────
        eprintln!("[UPDATE] Pushing {} steps into replay buffer...", total_steps);
        let advantages = read_advantages_jsonl(&adv_path);

        if advantages.len() != trajectories.len() {
            panic!(
                "Trajectory/advantage count mismatch: {} trajectories vs {} advantage records",
                trajectories.len(),
                advantages.len()
            );
        }

        // Anneal entropy bonus toward 0 as training progresses
        let annealed_entropy = (args.entropy_coeff * (1.0 - round as f32 / args.rounds as f32)).max(args.entropy_floor);

        replay_buffer.push_round(&trajectories, &advantages);

        // ── RELABEL: periodically re-critique entire replay buffer ───
        if args.relabel_interval > 0 && round > 0 && round % args.relabel_interval == 0 {
            eprintln!("[RELABEL] Re-critiquing all trajectories at round {round}...");
            let relabel_start = std::time::Instant::now();

            // Concat all trajectory JSONL files into one temp file
            let combined_path = args.output_dir.join("_relabel_trajectories.jsonl");
            {
                let mut combined = std::fs::File::create(&combined_path)
                    .unwrap_or_else(|e| panic!("Failed to create relabel file: {e}"));
                for r in 0..=round {
                    let rpath = args.output_dir.join(format!("trajectories_r{r}.jsonl"));
                    if rpath.exists() {
                        let content = std::fs::read(&rpath)
                            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", rpath.display()));
                        std::io::Write::write_all(&mut combined, &content)
                            .unwrap_or_else(|e| panic!("Failed to write relabel file: {e}"));
                    }
                }
            }

            // Run critic on combined trajectories
            let relabel_adv_path = args.output_dir.join("_relabel_advantages.jsonl");
            let status = std::process::Command::new("uv")
                .args(["run", critic_script_str])
                .arg("train")
                .args(["--input", combined_path.to_str().unwrap()])
                .args(["--output", relabel_adv_path.to_str().unwrap()])
                .args(["--checkpoint", critic_ckpt_str])
                .args(["--epochs", &args.critic_epochs.to_string()])
                .args(["--lr", &args.critic_lr.to_string()])
                .args(["--dropout", &args.critic_dropout.to_string()])
                .status()
                .unwrap_or_else(|e| panic!("Failed to run relabel critic: {e}"));

            if !status.success() {
                panic!("Relabel critic failed with exit code: {status}");
            }

            // Reload all trajectories and rebuild replay buffer
            let all_trajs = load_trajectories_jsonl(&combined_path);
            let all_advs = read_advantages_jsonl(&relabel_adv_path);
            assert_eq!(all_trajs.len(), all_advs.len(),
                "Relabel trajectory/advantage mismatch: {} vs {}", all_trajs.len(), all_advs.len());

            replay_buffer.steps.clear();
            replay_buffer.push_round(&all_trajs, &all_advs);

            // Cleanup temp files
            let _ = std::fs::remove_file(&combined_path);
            let _ = std::fs::remove_file(&relabel_adv_path);

            eprintln!(
                "[RELABEL] Rebuilt replay buffer with {} steps from {} trajectories in {:.1}s",
                replay_buffer.len(), all_trajs.len(), relabel_start.elapsed().as_secs_f64()
            );
        }

        eprintln!("[UPDATE] Buffer size: {} steps", replay_buffer.len());
        eprintln!("[MEMORY] after buffer push: {:.0}MB ({} steps)", rss_mb(), replay_buffer.len());

        // Multiple mini-batch gradient steps from replay buffer
        let mut round_grad_norm = 0.0f32;
        let mut total_policy_steps = 0usize;
        let mut total_value_steps = 0usize;
        let nan_cost_count = 0usize; // Always zero — NaN costs replaced with penalty upstream

        for update_idx in 0..args.updates_per_round {
            let batch_indices = replay_buffer.sample_batch(
                args.mini_batch_size,
                round_seed.wrapping_add(update_idx as u64 * 7919),
            );
            let mut grads = UnifiedGradients::zero();
            let mut batch_policy = 0usize;
            let mut batch_value = 0usize;

            for &idx in &batch_indices {
                let step = &replay_buffer.steps[idx];

                // NaN costs replaced with penalty upstream — none should reach replay buffer.
                assert!(
                    step.jit_cost_ns.is_finite() && step.jit_cost_ns >= 0.0,
                    "NaN/negative jit_cost_ns={} in replay buffer at index {idx} — \
                     trajectory filtering is broken",
                    step.jit_cost_ns,
                );

                let acc = acc_from_replay(step);
                let gacc = gacc_from_replay(step);
                let rule_embed = embed_from_replay(step);
                let cache = forward_cached(&model, &acc, &gacc, &rule_embed);

                backward_policy(
                    &model, &cache, &rule_embed,
                    step.matched, step.advantage, annealed_entropy, args.miss_penalty, &mut grads,
                );
                batch_policy += 1;

                let target = log_ns(step.jit_cost_ns);
                backward_value(&model, &cache, target, args.value_coeff, &mut grads);
                batch_value += 1;

                // Embedding gradients: value loss flows through expr backbone → acc_input.
                // Policy loss flows through graph backbone → graph_input (handled by
                // backward_policy → graph_w1). Only value path needs acc_input gradient.
                if !step.edges.is_empty() {
                    let d_acc_v = compute_d_acc_input_value(
                        &model, &cache, target, args.value_coeff,
                    );
                    let node_count = acc.node_count;
                    backward_through_accumulator(
                        &d_acc_v, &step.edges, node_count, &mut grads,
                    );
                }
            }

            let batch_size = batch_policy.max(1) as f32;
            grads.scale(1.0 / batch_size);
            let grad_norm = grads.norm();
            round_grad_norm += grad_norm;

            apply_unified_sgd(
                &mut model, &grads, &mut momentum_buf,
                args.lr, args.momentum, args.weight_decay, args.grad_clip,
            );

            eprintln!(
                "[UPDATE] step {}/{}: {} policy, {} value, grad_norm={:.6}",
                update_idx + 1, args.updates_per_round,
                batch_policy, batch_value, grad_norm,
            );

            total_policy_steps += batch_policy;
            total_value_steps += batch_value;
        }

        let avg_grad_norm = round_grad_norm / args.updates_per_round as f32;

        // NaN costs are replaced with penalty upstream, so none should reach here.
        // Assert this invariant holds rather than silently tolerating bad data.
        assert_eq!(nan_cost_count, 0, "NaN cost leaked into replay buffer");

        eprintln!("[MEMORY] after update: {:.0}MB", rss_mb());

        eprintln!(
            "[UPDATE] {} updates x {} batch = {} effective steps, \
             entropy_coeff={:.4}, avg_grad_norm={:.6}",
            args.updates_per_round, args.mini_batch_size,
            args.updates_per_round * args.mini_batch_size,
            annealed_entropy, avg_grad_norm,
        );

        // ── PHASE 5: CHECKPOINT ────────────────────────────────────────
        let ckpt_path = args.output_dir.join(format!("model_r{round}.bin"));
        model
            .save(&ckpt_path)
            .unwrap_or_else(|e| panic!("Failed to save checkpoint to {:?}: {e}", ckpt_path));

        // ── Compute metrics ──────────────────────────────────────────
        let round_elapsed = round_start.elapsed();
        let avg_steps = total_steps as f64 / trajectories.len() as f64;

        // Speedup: median of per-trajectory initial_ns / final_ns.
        // Median is robust to constant-collapse outliers (legit zero-mul simplifications
        // that hit the clock floor).
        const MAX_REASONABLE_NS: f64 = 1_000_000_000.0;
        let valid_jit_trajs: Vec<_> = trajectories
            .iter()
            .filter(|t| t.initial_cost_ns.is_finite() && t.initial_cost_ns > 0.0
                     && t.initial_cost_ns < MAX_REASONABLE_NS
                     && t.final_cost_ns.is_finite() && t.final_cost_ns > 0.0
                     && t.final_cost_ns < MAX_REASONABLE_NS)
            .collect();
        let mut speedups: Vec<f64> = valid_jit_trajs
            .iter()
            .map(|t| t.initial_cost_ns / t.final_cost_ns)
            .collect();
        speedups.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let speedup_median = if speedups.is_empty() {
            f64::NAN
        } else {
            let mid = speedups.len() / 2;
            if speedups.len() % 2 == 0 {
                (speedups[mid - 1] + speedups[mid]) / 2.0
            } else {
                speedups[mid]
            }
        };
        // Log outliers so we can see the constant-collapse simplifications
        let collapse_count = speedups.iter().filter(|&&s| s > 100.0).count();
        if collapse_count > 0 {
            eprintln!(
                "[METRICS] {collapse_count}/{} trajectories with >100x speedup \
                 (likely constant-fold), median={speedup_median:.2}x, max={:.0}x",
                speedups.len(),
                speedups.last().copied().unwrap_or(0.0),
            );
        }

        let avg_initial_ns: f64 = if valid_jit_trajs.is_empty() {
            f64::NAN
        } else {
            valid_jit_trajs.iter().map(|t| t.initial_cost_ns).sum::<f64>()
                / valid_jit_trajs.len() as f64
        };
        let avg_final_ns: f64 = if valid_jit_trajs.is_empty() {
            f64::NAN
        } else {
            valid_jit_trajs.iter().map(|t| t.final_cost_ns).sum::<f64>()
                / valid_jit_trajs.len() as f64
        };

        // Judge MAE: mean |predict_log_cost - log_ns(jit_cost)| across all steps
        let mut judge_error_sum = 0.0f64;
        let mut judge_error_count = 0u64;
        for traj in &trajectories {
            for step in &traj.steps {
                if step.jit_cost_ns.is_finite() && step.jit_cost_ns >= 0.5 {
                    let acc = acc_from_step(step);
                    let predicted = model.predict_log_cost_with_features(&acc);
                    let actual = log_ns(step.jit_cost_ns);
                    judge_error_sum += libm::fabs((predicted - actual) as f64);
                    judge_error_count += 1;
                }
            }
        }
        let judge_mae = if judge_error_count == 0 {
            f64::NAN
        } else {
            judge_error_sum / judge_error_count as f64
        };

        // Log metrics as JSONL
        let metrics = serde_json::json!({
            "round": round,
            "trajectories": trajectories.len(),
            "total_steps": total_steps,
            "avg_steps": avg_steps,
            "speedup_median": speedup_median,
            "avg_initial_ns": avg_initial_ns,
            "avg_final_ns": avg_final_ns,
            "judge_mae": judge_mae,
            "es_depth": gen_config.max_depth,
            "es_leaf_prob": gen_config.leaf_prob,
            "es_fitness": es_fitness,
            "policy_steps": total_policy_steps,
            "value_steps": total_value_steps,
            "nan_cost_count": nan_cost_count,
            "entropy_coeff": annealed_entropy,
            "grad_norm": avg_grad_norm,
            "buffer_size": replay_buffer.len(),
            "updates_this_round": args.updates_per_round,
            "effective_batch": args.updates_per_round * args.mini_batch_size,
            "workers": effective_workers,
            "gen_elapsed_s": gen_elapsed.as_secs_f64(),
            "rss_mb": rss_mb(),
            "elapsed_s": round_elapsed.as_secs_f64(),
        });
        writeln!(
            metrics_file,
            "{}",
            serde_json::to_string(&metrics)
                .unwrap_or_else(|e| panic!("Failed to serialize metrics: {e}"))
        )
        .unwrap_or_else(|e| panic!("Failed to write metrics: {e}"));
        metrics_file
            .flush()
            .unwrap_or_else(|e| panic!("Failed to flush metrics file: {e}"));

        eprintln!("[CHECKPOINT] Saved to {}", ckpt_path.display());
        eprintln!(
            "[METRICS] speedup={speedup_median:.3}x init={avg_initial_ns:.1}ns final={avg_final_ns:.1}ns \
             judge_mae={judge_mae:.3} es_fit={es_fitness:.3} steps={avg_steps:.1} \
             grad={avg_grad_norm:.2} buf={} time={:.1}s",
            replay_buffer.len(),
            round_elapsed.as_secs_f64()
        );
    }

    // Save final model back to original path
    model
        .save(&args.model)
        .unwrap_or_else(|e| panic!("Failed to save final model to {:?}: {e}", args.model));
    eprintln!("\nTraining complete. Final model saved to {:?}", args.model);
    eprintln!("Metrics log: {:?}", metrics_path);
}

// ============================================================================
// Offline Mode
// ============================================================================

/// Pure SGD on existing trajectory data. No self-play, no JIT benchmarking.
///
/// Pipeline:
/// 1. Load all trajectory JSONL from trajectory_dir (or output_dir)
/// 2. Run critic once to produce advantages
/// 3. Build replay buffer
/// 4. Do N rounds of pure SGD
/// 5. Save model + per-round metrics
fn run_offline(args: &Args, model: &mut ExprNnue) {
    let traj_dir = args.trajectory_dir.as_ref().unwrap_or(&args.output_dir);

    eprintln!("[OFFLINE] Loading trajectories from {:?}...", traj_dir);

    // Collect all trajectory files
    let mut traj_files: Vec<PathBuf> = std::fs::read_dir(traj_dir)
        .unwrap_or_else(|e| panic!("Failed to read trajectory dir {:?}: {e}", traj_dir))
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("trajectories_r") && name.ends_with(".jsonl") {
                Some(entry.path())
            } else {
                None
            }
        })
        .collect();
    traj_files.sort();

    if traj_files.is_empty() {
        panic!("[OFFLINE] No trajectory files found in {:?}", traj_dir);
    }

    // Concatenate all trajectory files
    let combined_path = args.output_dir.join("_offline_trajectories.jsonl");
    {
        let mut combined = std::fs::File::create(&combined_path)
            .unwrap_or_else(|e| panic!("Failed to create combined file: {e}"));
        for f in &traj_files {
            let content = std::fs::read(f)
                .unwrap_or_else(|e| panic!("Failed to read {}: {e}", f.display()));
            std::io::Write::write_all(&mut combined, &content)
                .unwrap_or_else(|e| panic!("Failed to write combined file: {e}"));
        }
    }

    let all_trajs = load_trajectories_jsonl(&combined_path);
    let total_steps: usize = all_trajs.iter().map(|t| t.steps.len()).sum();
    eprintln!(
        "[OFFLINE] Loaded {} trajectories ({} steps) from {} files",
        all_trajs.len(), total_steps, traj_files.len()
    );

    // Run critic to produce advantages
    let adv_path = args.output_dir.join("_offline_advantages.jsonl");
    let critic_script_str = args.critic_script.to_str()
        .unwrap_or_else(|| panic!("Invalid UTF-8 in critic script path"));
    let critic_ckpt_str = args.critic_checkpoint.to_str()
        .unwrap_or_else(|| panic!("Invalid UTF-8 in critic checkpoint path"));

    eprintln!("[OFFLINE] Training critic on {} trajectories...", all_trajs.len());
    let critic_start = std::time::Instant::now();
    let status = std::process::Command::new("uv")
        .args(["run", critic_script_str])
        .arg("train")
        .args(["--input", combined_path.to_str().unwrap()])
        .args(["--output", adv_path.to_str().unwrap()])
        .args(["--checkpoint", critic_ckpt_str])
        .args(["--epochs", &args.critic_epochs.to_string()])
        .args(["--lr", &args.critic_lr.to_string()])
        .args(["--dropout", &args.critic_dropout.to_string()])
        .status()
        .unwrap_or_else(|e| panic!("Failed to run critic: {e}"));

    if !status.success() {
        panic!("[OFFLINE] Critic failed with exit code: {status}");
    }
    eprintln!("[OFFLINE] Critic done in {:.1}s", critic_start.elapsed().as_secs_f64());

    let all_advs = read_advantages_jsonl(&adv_path);
    assert_eq!(
        all_trajs.len(), all_advs.len(),
        "Trajectory/advantage count mismatch: {} vs {}", all_trajs.len(), all_advs.len()
    );

    // Build replay buffer
    let mut replay_buffer = ReplayBuffer::new(args.replay_capacity);
    replay_buffer.push_round(&all_trajs, &all_advs);
    eprintln!("[OFFLINE] Replay buffer: {} steps", replay_buffer.len());

    // Cleanup temp files
    let _ = std::fs::remove_file(&combined_path);
    let _ = std::fs::remove_file(&adv_path);

    // Metrics log
    let metrics_path = args.output_dir.join("metrics.jsonl");
    let mut metrics_file = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&metrics_path)
            .unwrap_or_else(|e| panic!("Failed to open metrics file: {e}")),
    );

    // Initialize momentum buffer
    let mut momentum_buf = UnifiedGradients::zero();

    // Pure SGD rounds
    for round in 0..args.rounds {
        let round_start = std::time::Instant::now();
        let round_seed = args.seed.wrapping_add(round as u64 * 1000);
        let annealed_entropy = (args.entropy_coeff * (1.0 - round as f32 / args.rounds as f32)).max(args.entropy_floor);

        let mut round_grad_norm = 0.0f32;
        let mut total_policy_steps = 0usize;
        let mut total_value_steps = 0usize;

        for update_idx in 0..args.updates_per_round {
            let batch_indices = replay_buffer.sample_batch(
                args.mini_batch_size,
                round_seed.wrapping_add(update_idx as u64 * 7919),
            );
            let mut grads = UnifiedGradients::zero();
            let mut batch_policy = 0usize;
            let mut batch_value = 0usize;

            for &idx in &batch_indices {
                let step = &replay_buffer.steps[idx];

                assert!(
                    step.jit_cost_ns.is_finite() && step.jit_cost_ns >= 0.0,
                    "NaN/negative jit_cost_ns={} in replay buffer at index {idx}",
                    step.jit_cost_ns,
                );

                let acc = acc_from_replay(step);
                let gacc = gacc_from_replay(step);
                let rule_embed = embed_from_replay(step);
                let cache = forward_cached(model, &acc, &gacc, &rule_embed);

                backward_policy(
                    model, &cache, &rule_embed,
                    step.matched, step.advantage, annealed_entropy, args.miss_penalty, &mut grads,
                );
                batch_policy += 1;

                let target = log_ns(step.jit_cost_ns);
                backward_value(model, &cache, target, args.value_coeff, &mut grads);
                batch_value += 1;

                // Embedding gradients: value loss flows through expr backbone → acc_input.
                // Policy loss flows through graph backbone → graph_input (handled by
                // backward_policy → graph_w1). Only value path needs acc_input gradient.
                if !step.edges.is_empty() {
                    let d_acc_v = compute_d_acc_input_value(
                        model, &cache, target, args.value_coeff,
                    );
                    backward_through_accumulator(
                        &d_acc_v, &step.edges, acc.node_count, &mut grads,
                    );
                }
            }

            let batch_size = batch_policy.max(1) as f32;
            grads.scale(1.0 / batch_size);
            let grad_norm = grads.norm();
            round_grad_norm += grad_norm;

            apply_unified_sgd(
                model, &grads, &mut momentum_buf,
                args.lr, args.momentum, args.weight_decay, args.grad_clip,
            );

            total_policy_steps += batch_policy;
            total_value_steps += batch_value;
        }

        let avg_grad_norm = round_grad_norm / args.updates_per_round as f32;

        // Compute value loss on a sample for metrics
        let eval_indices = replay_buffer.sample_batch(
            args.mini_batch_size.min(replay_buffer.len()),
            round_seed.wrapping_add(0xE7A1),
        );
        let mut value_loss_sum = 0.0f64;
        let mut policy_loss_sum = 0.0f64;
        for &idx in &eval_indices {
            let step = &replay_buffer.steps[idx];
            let acc = acc_from_replay(step);
            let gacc = gacc_from_replay(step);
            let rule_embed = embed_from_replay(step);
            let cache = forward_cached(model, &acc, &gacc, &rule_embed);

            let target = log_ns(step.jit_cost_ns);
            let value_err = cache.value_pred - target;
            value_loss_sum += (value_err * value_err) as f64;

            // Policy loss: -log(prob) if matched, -log(1-prob) if not
            let p = (cache.prob as f64).clamp(1e-7, 1.0 - 1e-7);
            policy_loss_sum += -libm::log(p);
        }
        let avg_value_loss = value_loss_sum / eval_indices.len().max(1) as f64;
        let avg_policy_loss = policy_loss_sum / eval_indices.len().max(1) as f64;

        // Checkpoint
        let ckpt_path = args.output_dir.join(format!("model_r{round}.bin"));
        model.save(&ckpt_path)
            .unwrap_or_else(|e| panic!("Failed to save checkpoint: {e}"));

        let round_elapsed = round_start.elapsed();

        let metrics = serde_json::json!({
            "round": round,
            "mode": "offline",
            "policy_steps": total_policy_steps,
            "value_steps": total_value_steps,
            "avg_value_loss": avg_value_loss,
            "avg_policy_loss": avg_policy_loss,
            "entropy_coeff": annealed_entropy,
            "grad_norm": avg_grad_norm,
            "buffer_size": replay_buffer.len(),
            "updates_this_round": args.updates_per_round,
            "effective_batch": args.updates_per_round * args.mini_batch_size,
            "rss_mb": rss_mb(),
            "elapsed_s": round_elapsed.as_secs_f64(),
        });
        writeln!(metrics_file, "{}", serde_json::to_string(&metrics)
            .unwrap_or_else(|e| panic!("Failed to serialize metrics: {e}")))
            .unwrap_or_else(|e| panic!("Failed to write metrics: {e}"));
        metrics_file.flush()
            .unwrap_or_else(|e| panic!("Failed to flush metrics: {e}"));

        eprintln!(
            "[OFFLINE] round {}/{}: val_loss={:.4} pol_loss={:.4} grad={:.4} time={:.2}s",
            round + 1, args.rounds, avg_value_loss, avg_policy_loss,
            avg_grad_norm, round_elapsed.as_secs_f64()
        );
    }

    let final_path = args.output_dir.join("final_model.bin");
    model.save(&final_path)
        .unwrap_or_else(|e| panic!("Failed to save final model: {e}"));
    eprintln!("\n[OFFLINE] Complete. Final model saved to {:?}", final_path);
    eprintln!("[OFFLINE] Metrics: {:?}", metrics_path);
}
