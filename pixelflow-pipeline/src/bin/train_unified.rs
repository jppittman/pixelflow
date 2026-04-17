//! # Unified Self-Play Training Binary
//!
//! Orchestrates the outer training loop:
//!
//! ```text
//! GENERATE → EXPORT → CRITIQUE → UPDATE → CHECKPOINT
//! ```
//!
//! This is the single binary that replaced several older fragmented training
//! binaries. It runs an AlphaZero-style
//! training loop:
//!
//! 1. **GENERATE**: Self-play trajectories via e-graph hill-climbing with the
//!    current saturation head.
//! 2. **EXPORT**: Write binary trajectory batches for the Python Critic.
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
//!
//! The binary auto-starts a persistent critic server (`critic_server.py`) if
//! one is not already running at the `--critic-url` (default `http://localhost:8765`).
//! The server is killed on exit if we started it.

use std::io::{BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};

/// Buffered stderr logging. Replaces `eprintln!` to avoid per-line syscalls.
macro_rules! logln {
    ($($arg:tt)*) => {{
        use std::io::Write as _;
        let stderr = std::io::stderr();
        let mut lock = stderr.lock();
        let _ = writeln!(lock, $($arg)*);
    }};
}

use clap::Parser;
use pixelflow_pipeline::jit_bench::benchmark_jit_arena_repeated;
use pixelflow_pipeline::training::factored::parse_kernel_code_arena;
use pixelflow_search::egraph::{EGraph, Rewrite, all_rules, extract_neural_to_arena};
use pixelflow_search::math::all_math_rules;
use pixelflow_search::nnue::factored::{
    EMBED_DIM, EdgeAccumulator, ExprNnue, GRAPH_ACC_DIM, GRAPH_INPUT_DIM, GraphAccumulator,
    INPUT_DIM, K, RuleTemplates,
};

use pixelflow_pipeline::training::gen_es::log_ns;
use pixelflow_pipeline::training::self_play::{
    build_rule_templates, generate_corpus_trajectories_parallel,
    generate_trajectory_batch_parallel, load_corpus_exprs, load_trajectories_binary,
    read_advantages_binary, write_trajectories_binary,
};
use pixelflow_pipeline::training::unified::{Trajectory, TrajectoryAdvantages, TrajectoryStep};
use pixelflow_pipeline::training::unified_backward::{
    UnifiedGradients, apply_unified_sgd, backward_policy, backward_through_accumulator,
    backward_value, compute_d_acc_input_value, forward_cached,
};
use pixelflow_search::nnue::BwdGenConfig;

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
// Critic dispatch — HTTP server only
// ============================================================================

/// Call the persistent Python critic server to train on `traj_path` and write
/// advantages to `adv_path`.
///
/// Sends a single HTTP POST to `{critic_url}/train` with a JSON body. The
/// server keeps the model, optimizer, and rolling replay buffer warm across
/// rounds, eliminating ~1.8s Python startup overhead per round.
///
/// Fail-fast: any error panics with a clear message.

fn run_critic(
    critic_url: &str,
    traj_path_str: &str,
    adv_path_str: &str,
    epochs: usize,
    lr: f64,
    dropout: f64,
    mini_batch_size: usize,
) {
    let url = format!("{critic_url}/train");
    let body = serde_json::json!({
        "traj_path":      traj_path_str,
        "output_path":    adv_path_str,
        "epochs":         epochs,
        "lr":             lr,
        "dropout":        dropout,
        "mini_batch_size": mini_batch_size,
    });
    let resp = ureq::post(&url)
        .set("Content-Type", "application/json")
        .send_string(&body.to_string())
        .unwrap_or_else(|e| panic!("[CRITIQUE] HTTP POST to {url} failed: {e}"));
    if resp.status() != 200 {
        let status = resp.status();
        let text = resp.into_string().unwrap_or_default();
        panic!("[CRITIQUE] critic server returned HTTP {status}: {text}");
    }
    let result: serde_json::Value = resp
        .into_json()
        .unwrap_or_else(|e| panic!("[CRITIQUE] Failed to parse critic server response: {e}"));
    logln!(
        "[CRITIQUE] Server: loss={:.6}, steps={}",
        result["loss"].as_f64().unwrap_or(f64::NAN),
        result["steps"].as_u64().unwrap_or(0),
    );
}

/// Predict advantages only, without mutating critic training state.
///
/// This is the correct endpoint for replay-buffer relabeling: it re-grades
/// existing trajectories without appending duplicate history to the critic's
/// internal buffer.
fn predict_critic(critic_url: &str, traj_path_str: &str, adv_path_str: &str) {
    let url = format!("{critic_url}/predict");
    let body = serde_json::json!({
        "traj_path": traj_path_str,
        "output_path": adv_path_str,
    });
    let resp = ureq::post(&url)
        .set("Content-Type", "application/json")
        .timeout(std::time::Duration::from_secs(300))
        .send_string(&body.to_string())
        .unwrap_or_else(|e| panic!("[PREDICT] HTTP POST to {url} failed: {e}"));
    if resp.status() != 200 {
        let status = resp.status();
        let text = resp.into_string().unwrap_or_default();
        panic!("[PREDICT] critic server returned HTTP {status}: {text}");
    }
    let result: serde_json::Value = resp
        .into_json()
        .unwrap_or_else(|e| panic!("[PREDICT] Failed to parse response: {e}"));
    let trajs = result["trajectories"].as_u64().unwrap_or(0);
    let steps = result["steps"].as_u64().unwrap_or(0);
    logln!("[PREDICT] {trajs} trajs, {steps} steps");
}

/// Predict advantages AND do one incremental backprop step in a single call (~150ms total).
/// Uses the /step endpoint — forward pass + one gradient update, faster than separate
/// predict + retrain cycles.
fn step_critic(critic_url: &str, traj_path_str: &str, adv_path_str: &str) {
    let url = format!("{critic_url}/step");
    let body = serde_json::json!({
        "traj_path":   traj_path_str,
        "output_path": adv_path_str,
    });
    let resp = ureq::post(&url)
        .set("Content-Type", "application/json")
        .timeout(std::time::Duration::from_secs(300))
        .send_string(&body.to_string())
        .unwrap_or_else(|e| panic!("[STEP] HTTP POST to {url} failed: {e}"));
    if resp.status() != 200 {
        let status = resp.status();
        let text = resp.into_string().unwrap_or_default();
        panic!("[STEP] critic server returned HTTP {status}: {text}");
    }
    let result: serde_json::Value = resp
        .into_json()
        .unwrap_or_else(|e| panic!("[STEP] Failed to parse response: {e}"));
    let trajs = result["trajectories"].as_u64().unwrap_or(0);
    let steps = result["steps"].as_u64().unwrap_or(0);
    let loss = result["train_loss"].as_f64().unwrap_or(f64::NAN);
    logln!("[STEP] {trajs} trajs, {steps} steps, loss={loss:.6}");
}

/// Periodically retrain the critic from buffered history, then let the server
/// blend from the old teacher toward the new one over subsequent inference
/// calls. This is intentionally separate from replay relabeling.
fn retrain_critic(
    critic_url: &str,
    epochs: usize,
    lr: f64,
    dropout: f64,
    mini_batch_size: usize,
    max_trajectories: usize,
) {
    let url = format!("{critic_url}/retrain");
    let body = serde_json::json!({
        "epochs": epochs,
        "lr": lr,
        "dropout": dropout,
        "mini_batch_size": mini_batch_size,
        "max_trajectories": max_trajectories,
    });
    let resp = ureq::post(&url)
        .set("Content-Type", "application/json")
        .timeout(std::time::Duration::from_secs(3600))
        .send_string(&body.to_string())
        .unwrap_or_else(|e| panic!("[RETRAIN] HTTP POST to {url} failed: {e}"));
    if resp.status() != 200 {
        let status = resp.status();
        let text = resp.into_string().unwrap_or_default();
        panic!("[RETRAIN] critic server returned HTTP {status}: {text}");
    }
    let result: serde_json::Value = resp
        .into_json()
        .unwrap_or_else(|e| panic!("[RETRAIN] Failed to parse response: {e}"));
    logln!(
        "[RETRAIN] Server: loss={:.6}, steps={}",
        result["loss"].as_f64().unwrap_or(f64::NAN),
        result["steps"].as_u64().unwrap_or(0),
    );
}

// ============================================================================
// Critic server lifecycle
// ============================================================================

/// Ensure the critic server is running at `critic_url`. If not reachable,
/// spawn it as a background process and wait for it to become healthy.
///
/// Returns `Some(Child)` if we spawned the server (caller must kill on exit),
/// or `None` if it was already running.
fn critic_port_from_url(critic_url: &str) -> u16 {
    let without_scheme = critic_url.split("://").nth(1).unwrap_or(critic_url);
    let host_port = without_scheme.split('/').next().unwrap_or(without_scheme);
    let (_, port_str) = host_port.rsplit_once(':').unwrap_or_else(|| {
        panic!(
            "[CRITIC] critic_url must include an explicit port, got `{critic_url}`"
        )
    });
    port_str.parse::<u16>().unwrap_or_else(|e| {
        panic!("[CRITIC] Failed to parse port from critic_url `{critic_url}`: {e}")
    })
}

fn ensure_critic_server(
    critic_url: &str,
    critic_checkpoint: &str,
    critic_lr: f64,
    critic_dropout: f64,
) -> Option<std::process::Child> {
    let health_url = format!("{critic_url}/health");
    let port = critic_port_from_url(critic_url);

    // Check if server is already running
    if ureq::get(&health_url).call().is_ok() {
        logln!("[CRITIC] Server already running at {critic_url}");
        return None;
    }

    logln!("[CRITIC] Server not reachable at {critic_url}, starting...");

    let script_path = "pixelflow-pipeline/scripts/critic_server.py";
    let mut cmd = std::process::Command::new("uv");
    cmd.args(["run", script_path]);
    cmd.args(["--port", &port.to_string()]);
    cmd.args(["--checkpoint", critic_checkpoint]);
    cmd.args(["--lr", &critic_lr.to_string()]);
    cmd.args(["--dropout", &critic_dropout.to_string()]);
    // Drop server logs instead of piping them into an unread buffer, which can
    // eventually block the child process.
    cmd.stdout(std::process::Stdio::null());
    cmd.stderr(std::process::Stdio::null());

    let mut child = cmd.spawn().unwrap_or_else(|e| {
        panic!(
            "[CRITIC] Failed to spawn critic server (`uv run {script_path}`): {e}\n\
             Make sure `uv` is installed and {script_path} exists."
        )
    });
    logln!(
        "[CRITIC] Spawned critic server (pid={}), waiting for /health...",
        child.id()
    );

    // Cold Python + torch startup can take a while on some machines, especially
    // the first time `uv run` resolves/imports the stack.
    let startup_timeout = std::time::Duration::from_secs(60);
    let deadline = std::time::Instant::now() + startup_timeout;
    let poll_interval = std::time::Duration::from_millis(250);
    loop {
        if let Some(status) = child
            .try_wait()
            .unwrap_or_else(|e| panic!("[CRITIC] Failed to poll critic server process: {e}"))
        {
            panic!(
                "[CRITIC] Critic server exited before becoming healthy: {status}\n\
                 URL: {health_url}\n\
                 Check that `uv run {script_path}` starts correctly."
            );
        }
        if std::time::Instant::now() >= deadline {
            panic!(
                "[CRITIC] Critic server failed to become healthy within {:.0} seconds.\n\
                 URL: {health_url}\n\
                 Check that `uv run {script_path}` starts correctly.",
                startup_timeout.as_secs_f64()
            );
        }
        std::thread::sleep(poll_interval);
        if ureq::get(&health_url).call().is_ok() {
            logln!("[CRITIC] Server is healthy at {critic_url}");
            return Some(child);
        }
    }
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

    /// Gradient clipping threshold (per-group L2 norm).
    #[arg(long, default_value_t = 1.0)]
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

    /// Optional path to model weights to load at start.
    ///
    /// When omitted, training initializes a fresh model from latency-prior
    /// embeddings instead of implicitly loading an old checkpoint.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Path to save the final model. Defaults to --output-dir/final_model.bin.
    #[arg(long)]
    final_model: Option<PathBuf>,

    /// Output directory for checkpoints, trajectories, advantages.
    #[arg(long, default_value = "pixelflow-pipeline/data/unified")]
    output_dir: PathBuf,

    /// Critic checkpoint path (reused across rounds).
    #[arg(long, default_value = "pixelflow-pipeline/data/unified/critic.pt")]
    critic_checkpoint: PathBuf,

    /// Critic training epochs per round.
    #[arg(long, default_value_t = 20)]
    critic_epochs: usize,

    /// Trajectories per mini-batch in critic training. Prevents memorization
    /// of terminal rewards on small per-round batches.
    #[arg(long, default_value_t = 32)]
    critic_mini_batch_size: usize,

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

    /// Path to bench_corpus.bin for corpus-based trajectories.
    #[arg(long, default_value = "pixelflow-pipeline/data/bench_corpus.bin")]
    corpus_path: PathBuf,

    /// Max corpus expressions to hold in memory.
    #[arg(long, default_value_t = 2000)]
    corpus_max: usize,

    // ── Replay buffer ───────────────────────────────────────────
    /// Replay buffer capacity (max stored steps across rounds).
    #[arg(long, default_value_t = 200000)]
    replay_capacity: usize,

    /// Re-label entire replay buffer every N rounds via critic inference only
    /// (0 = never).
    #[arg(long, default_value_t = 5)]
    relabel_interval: usize,

    /// Retrain the critic from its buffered history every N rounds (0 = never).
    /// This is separate from replay relabeling: the server retrains once, then
    /// gradually blends from the old teacher to the new one on subsequent
    /// `/predict` / `/step` calls.
    #[arg(long, default_value_t = 0)]
    critic_retrain_interval: usize,

    /// Maximum number of buffered trajectories to use in each critic retrain
    /// (0 = all buffered trajectories).
    #[arg(long, default_value_t = 2000)]
    critic_retrain_max_trajectories: usize,

    /// Mini-batch size for SGD updates from replay buffer.
    #[arg(long, default_value_t = 2048)]
    mini_batch_size: usize,

    /// Number of gradient update steps per round.
    #[arg(long, default_value_t = 19)]
    updates_per_round: usize,

    /// Offline mode: skip self-play, load existing trajectory batches from output_dir,
    /// train critic once, then do pure SGD rounds on the replay buffer.
    /// Ideal for Optuna hyperparameter sweeps.
    #[arg(long, default_value_t = false)]
    offline: bool,

    /// Directory containing existing trajectory batch files to load in offline mode.
    /// Defaults to --output-dir if not set.
    #[arg(long)]
    trajectory_dir: Option<PathBuf>,

    /// Maximum number of trajectory files to load in offline mode (0 = all).
    /// Use a small value (e.g. 50) in Optuna sweeps to keep critic training fast —
    /// the most-recent files are loaded first (highest round numbers).
    #[arg(long, default_value_t = 0)]
    max_trajectory_files: usize,

    // ── Parallelism ────────────────────────────────────────────
    /// Number of worker threads for parallel trajectory generation.
    /// Defaults to the number of available CPU cores.
    /// Set to 1 for sequential execution (useful for debugging).
    #[arg(long)]
    workers: Option<usize>,

    // ── Persistent critic server ────────────────────────────────
    /// Base URL of the critic_server.py instance. Each CRITIQUE phase uses the
    /// server's batch file endpoints (`/step`, `/predict`, optionally
    /// `/retrain`). The server is auto-started if not
    /// already running.
    #[arg(long, default_value = "http://localhost:8765")]
    critic_url: String,

    /// Skip the Python critic entirely (use uniform advantages = 1.0).
    /// Useful for profiling the Rust self-play and SGD paths in isolation.
    #[arg(long, default_value_t = false)]
    skip_critic: bool,

    /// Run as a unix-socket server for Optuna trials.
    /// Loads corpus once, then accepts trial configs as JSON.
    /// Each trial: fresh model init → run N rounds → return metrics.
    #[arg(long)]
    server: Option<PathBuf>,
}

// ============================================================================
// Helpers
// ============================================================================

/// Reconstruct [`EdgeAccumulator`] from a [`TrajectoryStep`]'s accumulator_state.
///
/// Layout: `[values (4*K=128 floats), edge_count, node_count, node_budget, epoch_budget]` = 132 floats total.
/// Reconstruct [`EdgeAccumulator`] from a raw float vector (trajectory-level data).
fn acc_from_vec(v: &[f32]) -> EdgeAccumulator {
    if v.len() < INPUT_DIM {
        return EdgeAccumulator::default();
    }
    let mut acc = EdgeAccumulator::default();
    acc.values.copy_from_slice(&v[..4 * K]);
    acc.edge_count = v[4 * K] as u32;
    acc.node_count = v[4 * K + 1] as u32;
    acc.node_budget = v[4 * K + 2] as u32;
    acc.epoch_budget = v[4 * K + 3] as u32;
    acc
}

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
    acc.node_budget = step.accumulator_state[4 * K + 2] as u32;
    acc.epoch_budget = step.accumulator_state[4 * K + 3] as u32;
    acc
}

fn final_model_path(args: &Args) -> PathBuf {
    args.final_model
        .clone()
        .unwrap_or_else(|| args.output_dir.join("final_model.bin"))
}

/// Reconstruct [`EdgeAccumulator`] from a [`ReplayStep`], with VSA values normalized to unit L2.
///
/// EdgeAccumulator VSA values grow in scale over training rounds for the same
/// reason as rule_embed and graph_acc (they encode op embeddings). Normalizing
/// keeps value-head backbone gradients bounded regardless of replay step age.
fn acc_from_replay(step: &ReplayStep) -> EdgeAccumulator {
    assert_eq!(
        step.acc.len(),
        INPUT_DIM,
        "Expected {} accumulator values in replay step, got {}",
        INPUT_DIM,
        step.acc.len()
    );
    let l2: f32 = step.acc[..4 * K]
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
        .max(1e-8);
    let mut acc = EdgeAccumulator::default();
    for i in 0..4 * K {
        acc.values[i] = step.acc[i] / l2;
    }
    acc.edge_count = step.acc[4 * K] as u32;
    acc.node_count = step.acc[4 * K + 1] as u32;
    acc.node_budget = step.acc[4 * K + 2] as u32;
    acc.epoch_budget = step.acc[4 * K + 3] as u32;
    acc
}

/// Extract rule embedding from a [`ReplayStep`], normalized to unit L2.
///
/// Op embeddings grow in scale over many training rounds (L2 can reach 100+
/// by round 299). Normalizing at load time ensures both the forward pass
/// (score computation) and the backward pass (gradient chain rule) use the
/// same unit-norm vector, keeping the policy gradient magnitude bounded.
fn embed_from_replay(step: &ReplayStep) -> [f32; EMBED_DIM] {
    assert_eq!(
        step.rule_embed.len(),
        EMBED_DIM,
        "Expected {} rule embedding dims in replay step, got {}",
        EMBED_DIM,
        step.rule_embed.len()
    );
    let l2: f32 = step
        .rule_embed
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
        .max(1e-8);
    let mut embed = [0.0f32; EMBED_DIM];
    for i in 0..EMBED_DIM {
        embed[i] = step.rule_embed[i] / l2;
    }
    embed
}

/// Reconstruct [`GraphAccumulator`] from a [`ReplayStep`].
fn gacc_from_replay(step: &ReplayStep) -> GraphAccumulator {
    let mut gacc = GraphAccumulator::new();
    if step.graph_acc.len() == GRAPH_INPUT_DIM {
        // The VSA values stored in replay were computed with op embeddings from
        // the self-play model, which can drift to large L2 norms over many rounds
        // (same root cause as rule_embed scale inflation). Normalize to unit L2
        // so the graph_input fed into graph_w1 has bounded magnitude regardless
        // of when the trajectory was collected.
        let l2: f32 = step.graph_acc[..GRAPH_ACC_DIM]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(1e-8);
        for i in 0..GRAPH_ACC_DIM {
            gacc.values[i] = step.graph_acc[i] / l2;
        }
        gacc.edge_count = step.graph_acc[GRAPH_ACC_DIM] as u32;
        gacc.node_count = step.graph_acc[GRAPH_ACC_DIM + 1] as u32;
        gacc.node_budget = step.graph_acc[GRAPH_ACC_DIM + 2] as u32;
        gacc.epoch_budget = step.graph_acc[GRAPH_ACC_DIM + 3] as u32;
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
    acc: Vec<f32>,             // 132 floats (INPUT_DIM)
    graph_acc: Vec<f32>,       // GRAPH_INPUT_DIM floats — VSA graph state
    expr_embed: Vec<f32>,      // 32 floats (EMBED_DIM) — expr_proj output at decision time
    rule_embed: Vec<f32>,      // 32 floats (EMBED_DIM)
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
        Self {
            steps: Vec::new(),
            max_steps: capacity,
        }
    }

    /// Flatten trajectory steps + advantages into ReplaySteps and append.
    fn push_round(&mut self, trajectories: &[Trajectory], advantages: &[TrajectoryAdvantages]) {
        for (traj, adv) in trajectories.iter().zip(advantages.iter()) {
            assert_eq!(
                traj.steps.len(),
                adv.advantages.len(),
                "Step/advantage count mismatch in trajectory {}: {} vs {}",
                traj.trajectory_id,
                traj.steps.len(),
                adv.advantages.len()
            );
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
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            indices.push((state >> 33) as usize % n);
        }
        indices
    }
}

// ============================================================================
// Advantage normalization
// ============================================================================

/// Normalize a batch of advantages to zero mean and unit variance.
///
/// Standard REINFORCE stabilization technique: without this, the policy
/// gradient magnitude is proportional to the critic's output scale. As
/// the critic's scale drifts over many rounds, unnormalized advantages
/// cause the policy/graph weights to grow unboundedly even with clipping.
///
/// Returns a Vec of the same length with mean ≈ 0, std ≈ 1.
/// Falls back to centering-only when std < 1e-6 (degenerate constant batch).
fn normalize_advantages(raw: &[f32]) -> Vec<f32> {
    let n = raw.len();
    assert!(!raw.is_empty(), "normalize_advantages: empty batch");
    let mean = raw.iter().sum::<f32>() / n as f32;
    let variance = raw.iter().map(|&a| (a - mean) * (a - mean)).sum::<f32>() / n as f32;
    let std = variance.sqrt();
    if std < 1e-6 {
        // All advantages identical — center only, no scaling.
        raw.iter().map(|&a| a - mean).collect()
    } else {
        raw.iter().map(|&a| (a - mean) / std).collect()
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
    #[cfg(feature = "profiling")]
    let _guard = {
        let guard = pprof::ProfilerGuardBuilder::default()
            .frequency(997)
            .blocklist(&["libc", "libgcc", "pthread", "vDSP"])
            .build()
            .expect("failed to start pprof profiler");
        logln!("[PPROF] Profiler started (997 Hz). Flamegraph on exit.");
        guard
    };

    let args = Args::parse();

    // Server mode: load corpus once, accept trial configs over unix socket.
    if let Some(ref socket_path) = args.server {
        run_server(&args, socket_path);
        return;
    }

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)
        .unwrap_or_else(|e| panic!("Failed to create output dir {:?}: {e}", args.output_dir));

    // Ensure critic server is running (auto-start if needed)
    let critic_ckpt_str = args.critic_checkpoint.to_str().unwrap_or_else(|| {
        panic!(
            "Invalid UTF-8 in critic checkpoint path: {:?}",
            args.critic_checkpoint
        )
    });
    let mut critic_child = if args.skip_critic {
        logln!("[CRITIC] Skipped (--skip-critic).");
        None
    } else {
        ensure_critic_server(
            &args.critic_url,
            critic_ckpt_str,
            args.critic_lr,
            args.critic_dropout,
        )
    };

    // Load model or initialize fresh
    let mut model = if let Some(model_path) = &args.model {
        if model_path.exists() {
            let m = ExprNnue::load(model_path)
                .unwrap_or_else(|e| panic!("Failed to load model from {:?}: {e}", model_path));
            logln!("Loaded model from {:?}", model_path);
            m
        } else {
            logln!(
                "No model at {:?}, initializing fresh with seed {}",
                model_path,
                args.seed
            );
            ExprNnue::new_with_latency_prior(args.seed)
        }
    } else {
        logln!(
            "No initial model provided, initializing fresh with seed {}",
            args.seed
        );
        ExprNnue::new_with_latency_prior(args.seed)
    };

    // ── OFFLINE MODE ─────────────────────────────────────────────
    if args.offline {
        run_offline(&args, &mut model);
        // Kill critic server if we started it
        if let Some(ref mut child) = critic_child {
            logln!("[CRITIC] Shutting down server (pid={})...", child.id());
            let _ = child.kill();
            let _ = child.wait();
            logln!("[CRITIC] Server stopped.");
        }
        return;
    }

    // Report parallelism
    let effective_workers = args.workers.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    logln!(
        "Workers: {effective_workers} ({})",
        if args.workers.is_some() {
            "explicit"
        } else {
            "auto-detected"
        }
    );

    // Build rules + templates
    let rules: Vec<Box<dyn Rewrite>> = all_rules();
    let templates = build_rule_templates(&rules);

    // Initialize momentum buffer
    let mut momentum_buf = UnifiedGradients::zero();

    // Load corpus expressions
    let corpus_exprs = if args.corpus_fraction > 0.0 {
        load_corpus_exprs(&args.corpus_path, args.corpus_max, args.seed)
    } else {
        logln!("Corpus fraction is 0.0, skipping corpus loading");
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

        {
            let mut w = BufWriter::new(std::io::stderr().lock());
            let _ = writeln!(w, "\n{}", "=".repeat(60));
            let _ = writeln!(w, "Round {}/{}", round + 1, args.rounds);
            let _ = writeln!(w, "{}", "=".repeat(60));
            let _ = writeln!(w, "[MEMORY] round {} start: {:.0}MB", round, rss_mb());
        }

        // ── PHASE 1: GENERATE ──────────────────────────────────────────
        // Vary generator config per round via simple LCG — no ES overhead.
        // Covers the search space that ES was tuning, but in ~0ms instead of ~45s.
        let mut rs = round_seed;
        let mut rf = || -> f32 {
            rs = rs.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rs >> 33) as f32 / (1u64 << 31) as f32
        };
        let gen_config = BwdGenConfig {
            max_depth: 5 + (rf() * 6.0) as usize, // 5-10
            leaf_prob: 0.10 + rf() * 0.20,        // 0.10-0.30
            num_vars: 4,
            fused_op_prob: 0.05 + rf() * 0.15, // 0.05-0.20
            max_junkify_passes: 2 + (rf() * 4.0) as usize, // 2-5
            junkify_prob: 0.5 + rf() * 0.3,    // 0.5-0.8
            max_junkified_nodes: 300 + (rf() * 400.0) as usize, // 300-700
        };

        // Split trajectories between random-generated and corpus-sampled
        let corpus_count = (args.trajectories_per_round as f32 * args.corpus_fraction) as usize;
        let es_count = args.trajectories_per_round - corpus_count;

        logln!(
            "[GENERATE] Running {} ES + {} corpus trajectories...",
            es_count,
            corpus_count,
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
            logln!("[GENERATE] No valid trajectories generated, skipping round");
            continue;
        }

        // Replace NaN/non-finite jit_cost_ns with a penalty cost instead of dropping trajectories
        let max_cost_ns = trajectories
            .iter()
            .flat_map(|t| t.steps.iter())
            .filter_map(|s| {
                if s.jit_cost_ns.is_finite() {
                    Some(s.jit_cost_ns)
                } else {
                    None
                }
            })
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
            logln!(
                "[GENERATE] Replaced {nan_replaced} NaN jit_cost_ns with penalty={penalty_cost_ns:.1}ns"
            );
        }

        if empty_count > 0 {
            logln!("[GENERATE] Filtered {empty_count} empty out of {pre_filter} trajectories",);
        }

        logln!("[MEMORY] after generate: {:.0}MB", rss_mb());

        let total_steps: usize = trajectories.iter().map(|t| t.steps.len()).sum();
        logln!(
            "[GENERATE] {} trajectories, {} total steps in {:.1}s",
            trajectories.len(),
            total_steps,
            gen_elapsed.as_secs_f64()
        );

        // ── PHASE 2: EXPORT ────────────────────────────────────────────
        let traj_path = args
            .output_dir
            .join(format!("trajectories_r{round}.pftraj"));
        write_trajectories_binary(&trajectories, &traj_path);
        logln!("[EXPORT] Wrote {}", traj_path.display());

        // ── PHASE 3: CRITIQUE ──────────────────────────────────────────
        // Inference-only on most rounds (sub-second). Retrain critic every 200
        // rounds on all accumulated trajectory data.
        let adv_path = args.output_dir.join(format!("advantages_r{round}.pfadv"));
        // Remove stale advantages file before critic writes — prevents accumulation
        // across retries or interrupted runs hitting the same round number.
        let _ = std::fs::remove_file(&adv_path);

        let traj_path_str = traj_path
            .to_str()
            .unwrap_or_else(|| panic!("Invalid UTF-8 in trajectory path: {:?}", traj_path));
        let adv_path_str = adv_path
            .to_str()
            .unwrap_or_else(|| panic!("Invalid UTF-8 in advantages path: {:?}", adv_path));

        let advantages = if args.skip_critic {
            logln!(
                "[CRITIQUE] Skipped. Uniform advantages for {} trajectories.",
                trajectories.len()
            );
            trajectories
                .iter()
                .enumerate()
                .map(|(i, traj)| TrajectoryAdvantages {
                    trajectory_idx: i,
                    advantages: vec![1.0f32; traj.steps.len()],
                })
                .collect::<Vec<_>>()
        } else {
            let critic_start = std::time::Instant::now();
            step_critic(&args.critic_url, traj_path_str, adv_path_str);
            let critic_elapsed = critic_start.elapsed();
            logln!("[STEP] Done in {:.1}s", critic_elapsed.as_secs_f64());
            read_advantages_binary(&adv_path)
        };

        // ── PHASE 4: UPDATE ────────────────────────────────────────────
        logln!(
            "[UPDATE] Pushing {} steps into replay buffer...",
            total_steps
        );

        if advantages.len() != trajectories.len() {
            panic!(
                "Trajectory/advantage count mismatch: {} trajectories vs {} advantage records",
                trajectories.len(),
                advantages.len()
            );
        }

        // Anneal entropy bonus toward 0 as training progresses
        let annealed_entropy = (args.entropy_coeff * (1.0 - round as f32 / args.rounds as f32))
            .max(args.entropy_floor);

        replay_buffer.push_round(&trajectories, &advantages);

        // ── CRITIC RETRAIN: periodically refresh the teacher from buffered history ───
        if args.critic_retrain_interval > 0
            && round > 0
            && round % args.critic_retrain_interval == 0
        {
            logln!("[RETRAIN] Refreshing critic from buffered history at round {round}...");
            let retrain_start = std::time::Instant::now();
            retrain_critic(
                &args.critic_url,
                args.critic_epochs,
                args.critic_lr,
                args.critic_dropout,
                args.critic_mini_batch_size,
                args.critic_retrain_max_trajectories,
            );
            logln!(
                "[RETRAIN] Done in {:.1}s",
                retrain_start.elapsed().as_secs_f64()
            );
        }

        // ── RELABEL: periodically re-critique entire replay buffer ───
        if args.relabel_interval > 0 && round > 0 && round % args.relabel_interval == 0 {
            logln!("[RELABEL] Re-critiquing all trajectories at round {round}...");
            let relabel_start = std::time::Instant::now();

            // Materialize one temporary binary batch for critic inference.
            let combined_path = args.output_dir.join("_relabel_trajectories.pftraj");
            let mut all_trajs = Vec::new();
            for r in 0..=round {
                let rpath = args.output_dir.join(format!("trajectories_r{r}.pftraj"));
                if rpath.exists() {
                    all_trajs.extend(load_trajectories_binary(&rpath));
                }
            }
            write_trajectories_binary(&all_trajs, &combined_path);

            // Re-grade the combined replay without mutating critic buffer state.
            let relabel_adv_path = args.output_dir.join("_relabel_advantages.pfadv");
            predict_critic(
                &args.critic_url,
                combined_path
                    .to_str()
                    .unwrap_or_else(|| panic!("Invalid UTF-8 in combined relabel path")),
                relabel_adv_path
                    .to_str()
                    .unwrap_or_else(|| panic!("Invalid UTF-8 in relabel adv path")),
            );

            let all_advs = read_advantages_binary(&relabel_adv_path);
            assert_eq!(
                all_trajs.len(),
                all_advs.len(),
                "Relabel trajectory/advantage mismatch: {} vs {}",
                all_trajs.len(),
                all_advs.len()
            );

            replay_buffer.steps.clear();
            replay_buffer.push_round(&all_trajs, &all_advs);

            // Cleanup temp files
            let _ = std::fs::remove_file(&combined_path);
            let _ = std::fs::remove_file(&relabel_adv_path);

            logln!(
                "[RELABEL] Rebuilt replay buffer with {} steps from {} trajectories in {:.1}s",
                replay_buffer.len(),
                all_trajs.len(),
                relabel_start.elapsed().as_secs_f64()
            );
        }

        // Lock stderr once for the entire UPDATE phase — avoids per-line lock/unlock
        // overhead when updates_per_round is large.
        let mut buf_err = BufWriter::new(std::io::stderr().lock());
        let _ = writeln!(
            buf_err,
            "[UPDATE] Buffer size: {} steps",
            replay_buffer.len()
        );
        let _ = writeln!(
            buf_err,
            "[MEMORY] after buffer push: {:.0}MB ({} steps)",
            rss_mb(),
            replay_buffer.len()
        );
        let _ = buf_err.flush();
        drop(buf_err);

        // Multiple mini-batch gradient steps from replay buffer
        let mut round_grad_norm = 0.0f32;
        let mut round_clipped_grad_norm = 0.0f32;
        let mut total_policy_steps = 0usize;
        let mut total_value_steps = 0usize;
        let nan_cost_count = 0usize; // Always zero — NaN costs replaced with penalty upstream

        // Accumulate per-step log lines, flush once after the loop
        let mut update_log = String::with_capacity(256 * args.updates_per_round);

        for update_idx in 0..args.updates_per_round {
            let batch_indices = replay_buffer.sample_batch(
                args.mini_batch_size,
                round_seed.wrapping_add(update_idx as u64 * 7919),
            );
            let raw_advantages: Vec<f32> = batch_indices
                .iter()
                .map(|&i| replay_buffer.steps[i].advantage)
                .collect();
            let norm_advantages = normalize_advantages(&raw_advantages);

            let mut grads = UnifiedGradients::zero();
            let mut batch_policy = 0usize;
            let mut batch_value = 0usize;

            for (pos, &idx) in batch_indices.iter().enumerate() {
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
                    &model,
                    &cache,
                    &rule_embed,
                    step.matched,
                    norm_advantages[pos],
                    annealed_entropy,
                    args.miss_penalty,
                    &mut grads,
                );
                batch_policy += 1;

                // NOTE: extraction head (value) is trained separately below on
                // trajectory-level paired data (initial_acc→initial_cost,
                // final_acc→final_cost). NOT here — per-step accumulator_state
                // is the initial expression, but jit_cost_ns is the final cost.
                // Training here would teach "initial structure → final cost"
                // which is wrong for extraction.
                batch_value += 1;

                // Extraction head embedding gradients are computed in the
                // trajectory-level extraction training loop below, not here.
            }

            let batch_size = batch_policy.max(1) as f32;
            grads.scale(1.0 / batch_size);
            let clip_stats = grads.clip_stats(args.grad_clip);
            let grad_norm = clip_stats.raw_norm;
            round_grad_norm += grad_norm;
            round_clipped_grad_norm += clip_stats.clipped_norm;

            apply_unified_sgd(
                &mut model,
                &grads,
                &mut momentum_buf,
                args.lr,
                args.momentum,
                args.weight_decay,
                args.grad_clip,
            );

            use std::fmt::Write as FmtWrite;
            let _ = writeln!(
                update_log,
                "[UPDATE] step {}/{}: {} policy, {} value, grad_raw={:.3} grad_clip={:.3} \
                 (bb={:.3} val={:.3} pol={:.3} graph={:.3} trunk={:.3} emb={:.3})",
                update_idx + 1,
                args.updates_per_round,
                batch_policy,
                batch_value,
                grad_norm,
                clip_stats.clipped_norm,
                clip_stats.backbone_norm,
                clip_stats.value_norm,
                clip_stats.policy_norm,
                clip_stats.graph_norm,
                clip_stats.trunk_norm,
                clip_stats.embeddings_norm,
            );

            total_policy_steps += batch_policy;
            total_value_steps += batch_value;
        }

        // Flush all update logs in one write
        {
            let mut w = BufWriter::new(std::io::stderr().lock());
            let _ = w.write_all(update_log.as_bytes());
        }

        let avg_grad_norm = round_grad_norm / args.updates_per_round as f32;
        let avg_clipped_grad_norm = round_clipped_grad_norm / args.updates_per_round as f32;

        // NaN costs are replaced with penalty upstream, so none should reach here.
        // Assert this invariant holds rather than silently tolerating bad data.
        assert_eq!(nan_cost_count, 0, "NaN cost leaked into replay buffer");

        // ── Extraction head training (trajectory-level) ─────────────
        // Train on paired data: (accumulator, jit_cost) where both describe
        // the SAME expression. Two pairs per trajectory: initial and final.
        // This teaches "expression structure → execution cost" correctly.
        {
            let mut ext_grads = UnifiedGradients::zero();
            let mut ext_count = 0usize;
            for traj in &trajectories {
                // Initial expression pair
                if !traj.initial_accumulator_state.is_empty()
                    && traj.initial_cost_ns.is_finite()
                    && traj.initial_cost_ns > 0.0
                {
                    let acc = acc_from_vec(&traj.initial_accumulator_state);
                    let gacc = GraphAccumulator::new(); // dummy — extraction head ignores it
                    let rule_embed = [0.0f32; EMBED_DIM]; // dummy
                    let cache = forward_cached(&model, &acc, &gacc, &rule_embed);
                    let target = log_ns(traj.initial_cost_ns);
                    backward_value(&model, &cache, target, args.value_coeff, &mut ext_grads);

                    if !traj.initial_edges.is_empty() {
                        let d_acc =
                            compute_d_acc_input_value(&model, &cache, target, args.value_coeff);
                        backward_through_accumulator(
                            &d_acc,
                            &traj.initial_edges,
                            acc.node_count,
                            &mut ext_grads,
                        );
                    }
                    ext_count += 1;
                }

                // Final expression pair
                if !traj.final_accumulator_state.is_empty()
                    && traj.final_cost_ns.is_finite()
                    && traj.final_cost_ns > 0.0
                {
                    let acc = acc_from_vec(&traj.final_accumulator_state);
                    let gacc = GraphAccumulator::new();
                    let rule_embed = [0.0f32; EMBED_DIM];
                    let cache = forward_cached(&model, &acc, &gacc, &rule_embed);
                    let target = log_ns(traj.final_cost_ns);
                    backward_value(&model, &cache, target, args.value_coeff, &mut ext_grads);

                    if !traj.final_edges.is_empty() {
                        let d_acc =
                            compute_d_acc_input_value(&model, &cache, target, args.value_coeff);
                        backward_through_accumulator(
                            &d_acc,
                            &traj.final_edges,
                            acc.node_count,
                            &mut ext_grads,
                        );
                    }
                    ext_count += 1;
                }
            }

            if ext_count > 0 {
                ext_grads.scale(1.0 / ext_count as f32);
                let ext_clip_stats = ext_grads.clip_stats(args.grad_clip);
                apply_unified_sgd(
                    &mut model,
                    &ext_grads,
                    &mut momentum_buf,
                    args.lr,
                    args.momentum,
                    args.weight_decay,
                    args.grad_clip,
                );
                logln!(
                    "[EXTRACTION] {ext_count} paired samples, grad_raw={:.3} grad_clip={:.3}",
                    ext_clip_stats.raw_norm,
                    ext_clip_stats.clipped_norm
                );
            }
        }

        {
            let mut w = BufWriter::new(std::io::stderr().lock());
            let _ = writeln!(w, "[MEMORY] after update: {:.0}MB", rss_mb());
            let _ = writeln!(
                w,
                "[UPDATE] {} updates x {} batch = {} effective steps, \
                 entropy_coeff={:.4}, avg_grad_raw={:.6} avg_grad_clip={:.6}",
                args.updates_per_round,
                args.mini_batch_size,
                args.updates_per_round * args.mini_batch_size,
                annealed_entropy,
                avg_grad_norm,
                avg_clipped_grad_norm,
            );
        }

        // ── PHASE 5: CHECKPOINT (every 10 rounds + final) ─────────────
        let is_last = round + 1 == args.rounds;
        if round % 10 == 0 || is_last {
            let ckpt_path = args.output_dir.join(format!("model_r{round}.bin"));
            model
                .save(&ckpt_path)
                .unwrap_or_else(|e| panic!("Failed to save checkpoint to {:?}: {e}", ckpt_path));
            logln!("[CHECKPOINT] Saved to {}", ckpt_path.display());
        }

        // ── Compute metrics ──────────────────────────────────────────
        let round_elapsed = round_start.elapsed();
        let avg_steps = total_steps as f64 / trajectories.len() as f64;

        // Speedup: median of per-trajectory initial_ns / final_ns.
        // Median is robust to constant-collapse outliers (legit zero-mul simplifications
        // that hit the clock floor).
        const MAX_REASONABLE_NS: f64 = 1_000_000_000.0;
        let valid_jit_trajs: Vec<_> = trajectories
            .iter()
            .filter(|t| {
                t.initial_cost_ns.is_finite()
                    && t.initial_cost_ns > 0.0
                    && t.initial_cost_ns < MAX_REASONABLE_NS
                    && t.final_cost_ns.is_finite()
                    && t.final_cost_ns > 0.0
                    && t.final_cost_ns < MAX_REASONABLE_NS
            })
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
            let mut w = BufWriter::new(std::io::stderr().lock());
            let _ = writeln!(
                w,
                "[METRICS] {collapse_count}/{} trajectories with >100x speedup \
                 (likely constant-fold), median={speedup_median:.2}x, max={:.0}x",
                speedups.len(),
                speedups.last().copied().unwrap_or(0.0),
            );
        }

        let avg_initial_ns: f64 = if valid_jit_trajs.is_empty() {
            f64::NAN
        } else {
            valid_jit_trajs
                .iter()
                .map(|t| t.initial_cost_ns)
                .sum::<f64>()
                / valid_jit_trajs.len() as f64
        };
        let avg_final_ns: f64 = if valid_jit_trajs.is_empty() {
            f64::NAN
        } else {
            valid_jit_trajs.iter().map(|t| t.final_cost_ns).sum::<f64>()
                / valid_jit_trajs.len() as f64
        };

        // Extraction head MAE: mean |predict_log_cost - log_ns(jit_cost)| across all steps
        let mut extraction_error_sum = 0.0f64;
        let mut extraction_error_count = 0u64;
        for traj in &trajectories {
            for step in &traj.steps {
                if step.jit_cost_ns.is_finite() && step.jit_cost_ns >= 0.5 {
                    let acc = acc_from_step(step);
                    let predicted = model.predict_log_cost_with_features(&acc);
                    let actual = log_ns(step.jit_cost_ns);
                    extraction_error_sum += libm::fabs((predicted - actual) as f64);
                    extraction_error_count += 1;
                }
            }
        }
        let extraction_mae = if extraction_error_count == 0 {
            f64::NAN
        } else {
            extraction_error_sum / extraction_error_count as f64
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
            "extraction_mae": extraction_mae,
            "es_depth": gen_config.max_depth,
            "es_leaf_prob": gen_config.leaf_prob,
            "gen_depth": gen_config.max_depth,
            "policy_steps": total_policy_steps,
            "value_steps": total_value_steps,
            "nan_cost_count": nan_cost_count,
            "entropy_coeff": annealed_entropy,
            "grad_norm": avg_grad_norm,
            "grad_norm_raw": avg_grad_norm,
            "grad_norm_clipped": avg_clipped_grad_norm,
            "grad_clip_threshold": args.grad_clip,
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

        // Timestamp for log correlation
        // Local time from `date +%T` — always correct regardless of DST.
        let hms = std::process::Command::new("date")
            .arg("+%T")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|_| "??:??:??".into());
        let hms = &hms;

        {
            let mut w = BufWriter::new(std::io::stderr().lock());
            let _ = writeln!(
                w,
                "[{hms}] [METRICS] speedup={speedup_median:.3}x init={avg_initial_ns:.1}ns final={avg_final_ns:.1}ns \
                 extraction_mae={extraction_mae:.3} depth={} steps={avg_steps:.1} \
                 grad_raw={avg_grad_norm:.2} grad_clip={avg_clipped_grad_norm:.2} buf={} time={:.1}s",
                gen_config.max_depth,
                replay_buffer.len(),
                round_elapsed.as_secs_f64()
            );
        }
    }

    let final_path = final_model_path(&args);
    model
        .save(&final_path)
        .unwrap_or_else(|e| panic!("Failed to save final model to {:?}: {e}", final_path));
    logln!("\nTraining complete. Final model saved to {:?}", final_path);
    logln!("Metrics log: {:?}", metrics_path);

    // Kill critic server if we started it
    if let Some(ref mut child) = critic_child {
        logln!("[CRITIC] Shutting down server (pid={})...", child.id());
        let _ = child.kill();
        let _ = child.wait();
        logln!("[CRITIC] Server stopped.");
    }

    #[cfg(feature = "profiling")]
    {
        use std::fs::File;
        if let Ok(report) = _guard.report().build() {
            let fg_path = "/tmp/train_unified_flamegraph.svg";
            let file =
                File::create(fg_path).unwrap_or_else(|e| panic!("Failed to create {fg_path}: {e}"));
            report
                .flamegraph(file)
                .unwrap_or_else(|e| panic!("Failed to write flamegraph: {e}"));
            logln!("[PPROF] Flamegraph written to {fg_path}");
        }
    }
}

// ============================================================================
// Offline Mode
// ============================================================================

/// Pure SGD on existing trajectory data. No self-play, no JIT benchmarking.
///
/// Pipeline:
/// 1. Load all binary trajectory batches from trajectory_dir (or output_dir)
/// 2. Run critic once to produce advantages
/// 3. Build replay buffer
/// 4. Do N rounds of pure SGD
/// 5. Save model + per-round metrics
fn run_offline(args: &Args, model: &mut ExprNnue) {
    let traj_dir = args.trajectory_dir.as_ref().unwrap_or(&args.output_dir);

    logln!("[OFFLINE] Loading trajectories from {:?}...", traj_dir);

    // Collect all trajectory files
    let mut traj_files: Vec<PathBuf> = std::fs::read_dir(traj_dir)
        .unwrap_or_else(|e| panic!("Failed to read trajectory dir {:?}: {e}", traj_dir))
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("trajectories_r") && name.ends_with(".pftraj") {
                Some(entry.path())
            } else {
                None
            }
        })
        .collect();
    // Sort ascending so the most-recent files (highest round number) are at the end.
    traj_files.sort();

    if traj_files.is_empty() {
        panic!("[OFFLINE] No trajectory files found in {:?}", traj_dir);
    }

    // Optionally limit to the N most-recent files (highest round numbers).
    // Useful for Optuna sweeps where critic training must be fast.
    if args.max_trajectory_files > 0 && traj_files.len() > args.max_trajectory_files {
        let drop = traj_files.len() - args.max_trajectory_files;
        traj_files.drain(..drop);
        logln!(
            "[OFFLINE] Limited to {} most-recent trajectory files (--max-trajectory-files)",
            args.max_trajectory_files
        );
    }

    let mut all_trajs = Vec::new();
    for f in &traj_files {
        all_trajs.extend(load_trajectories_binary(f));
    }
    let combined_path = args.output_dir.join("_offline_trajectories.pftraj");
    write_trajectories_binary(&all_trajs, &combined_path);
    let total_steps: usize = all_trajs.iter().map(|t| t.steps.len()).sum();
    logln!(
        "[OFFLINE] Loaded {} trajectories ({} steps) from {} files",
        all_trajs.len(),
        total_steps,
        traj_files.len()
    );

    // Run critic to produce advantages
    let adv_path = args.output_dir.join("_offline_advantages.pfadv");

    logln!(
        "[OFFLINE] Training critic on {} trajectories...",
        all_trajs.len()
    );
    let critic_start = std::time::Instant::now();
    run_critic(
        &args.critic_url,
        combined_path
            .to_str()
            .unwrap_or_else(|| panic!("Invalid UTF-8 in combined path")),
        adv_path
            .to_str()
            .unwrap_or_else(|| panic!("Invalid UTF-8 in adv path")),
        args.critic_epochs,
        args.critic_lr,
        args.critic_dropout,
        args.critic_mini_batch_size,
    );
    logln!(
        "[OFFLINE] Critic done in {:.1}s",
        critic_start.elapsed().as_secs_f64()
    );

    let all_advs = read_advantages_binary(&adv_path);
    assert_eq!(
        all_trajs.len(),
        all_advs.len(),
        "Trajectory/advantage count mismatch: {} vs {}",
        all_trajs.len(),
        all_advs.len()
    );

    // Build replay buffer
    let mut replay_buffer = ReplayBuffer::new(args.replay_capacity);
    replay_buffer.push_round(&all_trajs, &all_advs);
    logln!("[OFFLINE] Replay buffer: {} steps", replay_buffer.len());

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
        let annealed_entropy = (args.entropy_coeff * (1.0 - round as f32 / args.rounds as f32))
            .max(args.entropy_floor);

        let mut round_grad_norm = 0.0f32;
        let mut round_clipped_grad_norm = 0.0f32;
        let mut total_policy_steps = 0usize;
        let mut total_value_steps = 0usize;

        for update_idx in 0..args.updates_per_round {
            let batch_indices = replay_buffer.sample_batch(
                args.mini_batch_size,
                round_seed.wrapping_add(update_idx as u64 * 7919),
            );
            let raw_advantages: Vec<f32> = batch_indices
                .iter()
                .map(|&i| replay_buffer.steps[i].advantage)
                .collect();
            let norm_advantages = normalize_advantages(&raw_advantages);

            let mut grads = UnifiedGradients::zero();
            let mut batch_policy = 0usize;
            let mut batch_value = 0usize;

            for (pos, &idx) in batch_indices.iter().enumerate() {
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
                    model,
                    &cache,
                    &rule_embed,
                    step.matched,
                    norm_advantages[pos],
                    annealed_entropy,
                    args.miss_penalty,
                    &mut grads,
                );
                batch_policy += 1;

                let target = log_ns(step.jit_cost_ns);
                backward_value(model, &cache, target, args.value_coeff, &mut grads);
                batch_value += 1;

                // Embedding gradients: value loss flows through expr backbone → acc_input.
                // Policy loss flows through graph backbone → graph_input (handled by
                // backward_policy → graph_w1). Only value path needs acc_input gradient.
                if !step.edges.is_empty() {
                    let d_acc_v =
                        compute_d_acc_input_value(model, &cache, target, args.value_coeff);
                    backward_through_accumulator(&d_acc_v, &step.edges, acc.node_count, &mut grads);
                }
            }

            let batch_size = batch_policy.max(1) as f32;
            grads.scale(1.0 / batch_size);
            let clip_stats = grads.clip_stats(args.grad_clip);
            let grad_norm = clip_stats.raw_norm;
            round_grad_norm += grad_norm;
            round_clipped_grad_norm += clip_stats.clipped_norm;

            // Per-group norms on first update of first two rounds for diagnostics.
            if round < 2 && update_idx == 0 {
                logln!(
                    "[DIAG] round={} bb={:.3} val={:.3} pol={:.3} graph={:.3} trunk={:.3} emb={:.3} raw={:.3} clipped={:.3}",
                    round + 1,
                    clip_stats.backbone_norm,
                    clip_stats.value_norm,
                    clip_stats.policy_norm,
                    clip_stats.graph_norm,
                    clip_stats.trunk_norm,
                    clip_stats.embeddings_norm,
                    grad_norm,
                    clip_stats.clipped_norm,
                );
            }

            apply_unified_sgd(
                model,
                &grads,
                &mut momentum_buf,
                args.lr,
                args.momentum,
                args.weight_decay,
                args.grad_clip,
            );

            total_policy_steps += batch_policy;
            total_value_steps += batch_value;
        }

        let avg_grad_norm = round_grad_norm / args.updates_per_round as f32;
        let avg_clipped_grad_norm = round_clipped_grad_norm / args.updates_per_round as f32;

        // Compute extraction/saturation losses on a replay sample for metrics
        let eval_indices = replay_buffer.sample_batch(
            args.mini_batch_size.min(replay_buffer.len()),
            round_seed.wrapping_add(0xE7A1),
        );
        let mut extraction_loss_sum = 0.0f64;
        let mut saturation_loss_sum = 0.0f64;
        for &idx in &eval_indices {
            let step = &replay_buffer.steps[idx];
            let acc = acc_from_replay(step);
            let gacc = gacc_from_replay(step);
            let rule_embed = embed_from_replay(step);
            let cache = forward_cached(model, &acc, &gacc, &rule_embed);

            let target = log_ns(step.jit_cost_ns);
            let extraction_err = cache.value_pred - target;
            extraction_loss_sum += (extraction_err * extraction_err) as f64;

            // Saturation loss proxy: negative log-probability for the sampled matched rewrite.
            let p = (cache.prob as f64).clamp(1e-7, 1.0 - 1e-7);
            saturation_loss_sum += -libm::log(p);
        }
        let avg_extraction_loss = extraction_loss_sum / eval_indices.len().max(1) as f64;
        let avg_saturation_loss = saturation_loss_sum / eval_indices.len().max(1) as f64;

        // Checkpoint
        let ckpt_path = args.output_dir.join(format!("model_r{round}.bin"));
        model
            .save(&ckpt_path)
            .unwrap_or_else(|e| panic!("Failed to save checkpoint: {e}"));

        let round_elapsed = round_start.elapsed();

        let metrics = serde_json::json!({
            "round": round,
            "mode": "offline",
            "policy_steps": total_policy_steps,
            "value_steps": total_value_steps,
            "avg_extraction_loss": avg_extraction_loss,
            "avg_saturation_loss": avg_saturation_loss,
            "entropy_coeff": annealed_entropy,
            "grad_norm": avg_grad_norm,
            "grad_norm_raw": avg_grad_norm,
            "grad_norm_clipped": avg_clipped_grad_norm,
            "grad_clip_threshold": args.grad_clip,
            "buffer_size": replay_buffer.len(),
            "updates_this_round": args.updates_per_round,
            "effective_batch": args.updates_per_round * args.mini_batch_size,
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
            .unwrap_or_else(|e| panic!("Failed to flush metrics: {e}"));

        logln!(
            "[OFFLINE] round {}/{}: extraction_loss={:.4} saturation_loss={:.4} grad_raw={:.4} grad_clip={:.4} time={:.2}s",
            round + 1,
            args.rounds,
            avg_extraction_loss,
            avg_saturation_loss,
            avg_grad_norm,
            avg_clipped_grad_norm,
            round_elapsed.as_secs_f64()
        );
    }

    let final_path = final_model_path(args);
    model
        .save(&final_path)
        .unwrap_or_else(|e| panic!("Failed to save final model: {e}"));
    logln!(
        "\n[OFFLINE] Complete. Final model saved to {:?}",
        final_path
    );
    logln!("[OFFLINE] Metrics: {:?}", metrics_path);
}

// ============================================================================
// Server Mode — Optuna Hyperparameter Sweeps
// ============================================================================

/// Trial hyperparameters received as JSON from the Optuna driver.
///
/// Every field is optional with sensible defaults so the Optuna study can
/// sweep any subset. The defaults here match the CLI defaults in [`Args`].
#[derive(serde::Deserialize, Debug)]
struct TrialConfig {
    #[serde(default = "default_rounds")]
    rounds: usize,
    #[serde(default = "default_trajectories_per_round")]
    trajectories_per_round: usize,
    #[serde(default = "default_max_steps")]
    max_steps: usize,
    #[serde(default = "default_lr")]
    lr: f32,
    #[serde(default = "default_momentum")]
    momentum: f32,
    #[serde(default = "default_weight_decay")]
    weight_decay: f32,
    #[serde(default = "default_grad_clip")]
    grad_clip: f32,
    #[serde(default = "default_entropy_coeff")]
    entropy_coeff: f32,
    #[serde(default = "default_entropy_floor")]
    entropy_floor: f32,
    #[serde(default = "default_value_coeff")]
    value_coeff: f32,
    #[serde(default = "default_miss_penalty")]
    miss_penalty: f32,
    #[serde(default = "default_threshold")]
    threshold: f32,
    #[serde(default = "default_mini_batch_size")]
    mini_batch_size: usize,
    #[serde(default = "default_updates_per_round")]
    updates_per_round: usize,
    #[serde(default = "default_corpus_fraction")]
    corpus_fraction: f32,
    #[serde(default = "default_seed")]
    seed: u64,
    #[serde(default = "default_replay_capacity")]
    replay_capacity: usize,
    #[serde(default = "default_offline")]
    offline: bool,
    #[serde(default)]
    trajectory_dir: Option<PathBuf>,
    #[serde(default = "default_max_trajectory_files")]
    max_trajectory_files: usize,
    #[serde(default)]
    critic_epochs: Option<usize>,
    #[serde(default)]
    critic_lr: Option<f64>,
    #[serde(default)]
    critic_dropout: Option<f64>,
    #[serde(default)]
    critic_mini_batch_size: Option<usize>,
}

fn default_rounds() -> usize {
    15
}
fn default_trajectories_per_round() -> usize {
    30
}
fn default_max_steps() -> usize {
    50
}
fn default_lr() -> f32 {
    8.17e-3
}
fn default_momentum() -> f32 {
    0.891
}
fn default_weight_decay() -> f32 {
    5.34e-6
}
fn default_grad_clip() -> f32 {
    1.0
}
fn default_entropy_coeff() -> f32 {
    0.098
}
fn default_entropy_floor() -> f32 {
    0.02
}
fn default_value_coeff() -> f32 {
    1.175
}
fn default_miss_penalty() -> f32 {
    0.1
}
fn default_threshold() -> f32 {
    0.3
}
fn default_mini_batch_size() -> usize {
    2048
}
fn default_updates_per_round() -> usize {
    19
}
fn default_corpus_fraction() -> f32 {
    0.3
}
fn default_seed() -> u64 {
    42
}
fn default_replay_capacity() -> usize {
    200000
}
fn default_offline() -> bool {
    false
}
fn default_max_trajectory_files() -> usize {
    0
}

// ============================================================================
// Validation on held-out real shader expressions
// ============================================================================

/// Build the held-out validation expressions in arena form.
///
/// These are real ShaderToy-style expressions that the training corpus must
/// not contain. Keep this set fixed so Optuna trials compare on the same
/// out-of-sample workload instead of chasing synthetic easy cases.
fn build_validation_exprs() -> Vec<(&'static str, pixelflow_ir::ExprArena, pixelflow_ir::ExprId)> {
    let sources = [
        ("val_radial", "(((X * X) + (Y * Y)) - 0.7).abs()"),
        ("val_softclamp", "(X / (X.abs() + 1.0))"),
        (
            "val_distsq",
            "(((X - 0.5) * (X - 0.5)) + ((Y - 0.5) * (Y - 0.5)))",
        ),
        ("val_expdecay", "((-4.0 * ((X * X) + (Y * Y))).exp())"),
        ("val_normalize", "(X / ((X * X) + (Y * Y)).sqrt())"),
        (
            "val_channel",
            "(((Y.exp()) * ((-4.0 * ((X * X) + (Y * Y))).exp())) / ((((Y.exp()) * ((-4.0 * ((X * X) + (Y * Y))).exp())).abs()) + 1.0))",
        ),
        (
            "val_psychred",
            "((((((((((X * ((1.0 - ((((X * X) + (Y * Y)) - 0.7).abs())) * 5.0)) + (Z * 0.5)).sin()) + 1.0) * (((((X * ((1.0 - ((((X * X) + (Y * Y)) - 0.7).abs())) * 5.0)) + (Z * 0.5)) - ((Y * ((1.0 - ((((X * X) + (Y * Y)) - 0.7).abs())) * 5.0)) + ((Z * 0.5) * 0.7))).abs())) * 0.2) + 0.001)) / ((((Y + ((Z * 0.3).sin() * 0.2)).exp()) * (((((((X * X) + (Y * Y)) - 0.7).abs()) * -4.0) * (1.0 + ((Z * 2.0).sin() * 0.1))).exp())) / (((((((((X * ((1.0 - ((((X * X) + (Y * Y)) - 0.7).abs())) * 5.0)) + (Z * 0.5)).sin()) + 1.0) * (((((X * ((1.0 - ((((X * X) + (Y * Y)) - 0.7).abs())) * 5.0)) + (Z * 0.5)) - ((Y * ((1.0 - ((((X * X) + (Y * Y)) - 0.7).abs())) * 5.0)) + ((Z * 0.5) * 0.7))).abs())) * 0.2) + 0.001)).abs()) + 1.0)) + 1.0) * 0.5))",
        ),
        (
            "val_shadertoy_warp_x",
            "((X) + (((X - 0.5) * 1.7777777) / ((((X - 0.5) * 1.7777777) * ((X - 0.5) * 1.7777777) + ((Y - 0.5) * (Y - 0.5))).sqrt())) * (((Z + 0.07).sin()) + 1.0) * (((((((X - 0.5) * 1.7777777) * ((X - 0.5) * 1.7777777) + ((Y - 0.5) * (Y - 0.5))).sqrt()) * 9.0) - (Z + 0.07) - (Z + 0.07)).sin()).abs())",
        ),
        (
            "val_shadertoy_fuji_sun",
            "(((((0.3) - (((X * X) + (Y * Y)).sqrt())) / (0.3 - 0.29)).clamp(0.0, 1.0)) * ((3.0 * ((((Y) + (Z * 0.2 * 1.02)) * 100.0).sin()) + (((Y) * 14.0) + 1.0).clamp((-(6.0)), 6.0)).clamp(0.0, 1.0))).clamp(0.0, 1.0) + ((0.7 - (((X * X) + (Y * Y)).sqrt())) / 0.7).clamp(0.0, 1.0) * 0.6",
        ),
        (
            "val_shadertoy_smooth_union",
            "((Y + ((X - Y) * (0.5 + (0.5 * ((Y - X) / Z))).clamp(0.0, 1.0))) - (Z * (0.5 + (0.5 * ((Y - X) / Z))).clamp(0.0, 1.0) * (1.0 - (0.5 + (0.5 * ((Y - X) / Z))).clamp(0.0, 1.0))))",
        ),
    ];

    sources
        .into_iter()
        .map(|(name, src)| {
            let (arena, root) = parse_kernel_code_arena(src)
                .unwrap_or_else(|| panic!("validation parse failed for {name}: {src}"));
            (name, arena, root)
        })
        .collect()
}

/// Validation node budget for e-graph saturation — matches bench_shader_e2e.
const VAL_SAT_NODE_BUDGET: usize = 2_000;
/// Validation saturation epochs — matches bench_shader_e2e.
const VAL_SAT_MAX_EPOCHS: usize = 30;
/// Repeat the microbenchmark enough times that held-out validation timings are
/// not dominated by timer quantization.
const VAL_BENCH_REPEAT_BATCHES: usize = 100;

/// Evaluate the current model on held-out validation shader expressions.
///
/// For each expression:
///   1. Saturate e-graph with a small node budget (same as bench_shader_e2e)
///   2. Extract best expression with the trained model via the arena extractor
///   3. JIT-benchmark original and extracted
///   4. Compute speedup = original_ns / extracted_ns
///
/// Returns the median speedup across all expressions where benchmarking
/// succeeds. Returns 1.0 if no expressions could be benchmarked (fail-safe).
fn validate_on_shaders(model: &ExprNnue) -> f64 {
    let validation_exprs = build_validation_exprs();
    let mut speedups: Vec<f64> = Vec::with_capacity(validation_exprs.len());

    for (name, arena, arena_root) in &validation_exprs {
        // Benchmark the original expression
        let original_ns =
            match benchmark_jit_arena_repeated(arena, *arena_root, VAL_BENCH_REPEAT_BATCHES) {
                Ok(r) => r.ns,
                Err(e) => {
                    logln!("[VAL] {name}: original JIT failed: {e}");
                    continue;
                }
            };

        // Saturate with a small budget (fast, prevents e-graph explosion)
        let mut egraph = EGraph::with_rules(all_math_rules());
        let root = egraph.add_arena(arena, *arena_root);

        let num_rules = egraph.num_rules();
        for epoch in 0..VAL_SAT_MAX_EPOCHS {
            if egraph.node_count() > VAL_SAT_NODE_BUDGET {
                break;
            }
            let mut changes = 0;
            for rule_idx in 0..num_rules {
                if egraph.node_count() > VAL_SAT_NODE_BUDGET {
                    break;
                }
                changes += egraph.apply_rule_at_index(rule_idx, 10_000).changes;
            }
            if changes == 0 {
                let _ = epoch; // suppress unused warning
                break;
            }
        }

        // Extract best expression using the trained model
        let (extracted_arena, extracted_root, _cost) =
            extract_neural_to_arena(&egraph, root, model);

        // Benchmark the extracted expression
        let extracted_ns = match benchmark_jit_arena_repeated(
            &extracted_arena,
            extracted_root,
            VAL_BENCH_REPEAT_BATCHES,
        ) {
            Ok(r) => r.ns,
            Err(e) => {
                logln!("[VAL] {name}: extracted JIT failed: {e}");
                continue;
            }
        };

        // Guard against degenerate timings
        if !original_ns.is_finite()
            || original_ns <= 0.0
            || !extracted_ns.is_finite()
            || extracted_ns <= 0.0
        {
            logln!(
                "[VAL] {name}: degenerate timing original={original_ns:.3}ns extracted={extracted_ns:.3}ns"
            );
            continue;
        }

        let speedup = original_ns / extracted_ns;
        logln!("[VAL] {name}: {original_ns:.3}ns -> {extracted_ns:.3}ns = {speedup:.3}x");
        speedups.push(speedup);
    }

    if speedups.is_empty() {
        logln!("[VAL] WARNING: no validation expressions benchmarked successfully, returning 1.0");
        return 1.0;
    }

    speedups.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let mid = speedups.len() / 2;
    let median = if speedups.len() % 2 == 0 {
        (speedups[mid - 1] + speedups[mid]) / 2.0
    } else {
        speedups[mid]
    };

    logln!(
        "[VAL] Median validation speedup: {median:.3}x ({} expressions)",
        speedups.len()
    );
    median
}

/// Run one training trial with the given config, reusing pre-loaded corpus/rules.
///
/// Returns a JSON value with `{"metrics": [...], "best_score": float}`.
/// The score is the best (lowest) average value loss observed across rounds.
fn run_trial(
    trial_id: usize,
    config: &TrialConfig,
    rules: &[Box<dyn Rewrite>],
    templates: &RuleTemplates,
    corpus_exprs: &[(String, pixelflow_ir::ExprArena, pixelflow_ir::ExprId)],
    critic_url: &str,
    output_dir: &Path,
    workers: Option<usize>,
    default_critic_epochs: usize,
    default_critic_lr: f64,
    default_critic_dropout: f64,
    default_critic_mini_batch_size: usize,
) -> serde_json::Value {
    let trial_dir = output_dir.join(format!("trial_{trial_id}"));
    std::fs::create_dir_all(&trial_dir)
        .unwrap_or_else(|e| panic!("[SERVER] Failed to create trial dir {:?}: {e}", trial_dir));

    // Fresh model
    let mut model = ExprNnue::new_with_latency_prior(config.seed);

    // Fresh momentum + replay
    let mut momentum_buf = UnifiedGradients::zero();
    let mut replay_buffer = ReplayBuffer::new(config.replay_capacity);

    let effective_workers = workers.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });

    // Critic checkpoint path for this trial (cleaned up at the end)
    let critic_ckpt = trial_dir.join("critic.pt");

    let mut round_metrics = Vec::with_capacity(config.rounds);
    let mut best_score = f64::INFINITY;
    let critic_epochs = config.critic_epochs.unwrap_or(default_critic_epochs);
    let critic_lr = config.critic_lr.unwrap_or(default_critic_lr);
    let critic_dropout = config.critic_dropout.unwrap_or(default_critic_dropout);
    let critic_mini_batch_size = config
        .critic_mini_batch_size
        .unwrap_or(default_critic_mini_batch_size);

    if config.offline {
        let traj_dir = config.trajectory_dir.as_deref().unwrap_or(output_dir);
        logln!(
            "[SERVER] Trial {trial_id}: offline replay from {:?} (max_files={})",
            traj_dir,
            config.max_trajectory_files,
        );

        let mut traj_files: Vec<PathBuf> = std::fs::read_dir(traj_dir)
            .unwrap_or_else(|e| {
                panic!("[SERVER] Failed to read trajectory dir {:?}: {e}", traj_dir)
            })
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("trajectories_r") && name.ends_with(".pftraj") {
                    Some(entry.path())
                } else {
                    None
                }
            })
            .collect();
        traj_files.sort();

        if traj_files.is_empty() {
            panic!("[SERVER] No trajectory files found in {:?}", traj_dir);
        }

        if config.max_trajectory_files > 0 && traj_files.len() > config.max_trajectory_files {
            let drop = traj_files.len() - config.max_trajectory_files;
            traj_files.drain(..drop);
        }

        let mut all_trajs = Vec::new();
        for path in &traj_files {
            all_trajs.extend(load_trajectories_binary(path));
        }

        let combined_path = trial_dir.join("_offline_trajectories.pftraj");
        write_trajectories_binary(&all_trajs, &combined_path);
        let adv_path = trial_dir.join("_offline_advantages.pfadv");

        run_critic(
            critic_url,
            combined_path
                .to_str()
                .unwrap_or_else(|| panic!("[SERVER] Invalid UTF-8 in trajectory path")),
            adv_path
                .to_str()
                .unwrap_or_else(|| panic!("[SERVER] Invalid UTF-8 in advantage path")),
            critic_epochs,
            critic_lr,
            critic_dropout,
            critic_mini_batch_size,
        );

        let all_advs = read_advantages_binary(&adv_path);
        assert_eq!(
            all_trajs.len(),
            all_advs.len(),
            "[SERVER] Trial {trial_id}: trajectory/advantage count mismatch: {} vs {}",
            all_trajs.len(),
            all_advs.len()
        );

        replay_buffer.push_round(&all_trajs, &all_advs);
        let _ = std::fs::remove_file(&combined_path);
        let _ = std::fs::remove_file(&adv_path);
        logln!(
            "[SERVER] Trial {trial_id}: offline replay buffer contains {} steps from {} trajectories",
            replay_buffer.len(),
            all_trajs.len(),
        );
    }

    for round in 0..config.rounds {
        let round_start = std::time::Instant::now();
        let round_seed = config.seed.wrapping_add(round as u64 * 1000);

        logln!(
            "[SERVER] Trial {trial_id} round {}/{}: starting...",
            round + 1,
            config.rounds,
        );

        let mut speedup_median = f64::NAN;
        let mut total_steps = 0usize;
        let mut gen_elapsed_s = 0.0f64;
        let mut trajectory_count = 0usize;

        if !config.offline {
            // ── PHASE 1: GENERATE ─────────────────────────────────────────
            let mut rs = round_seed;
            let mut rf = || -> f32 {
                rs = rs.wrapping_mul(6364136223846793005).wrapping_add(1);
                (rs >> 33) as f32 / (1u64 << 31) as f32
            };
            let gen_config = BwdGenConfig {
                max_depth: 5 + (rf() * 6.0) as usize,
                leaf_prob: 0.10 + rf() * 0.20,
                num_vars: 4,
                fused_op_prob: 0.05 + rf() * 0.15,
                max_junkify_passes: 2 + (rf() * 4.0) as usize,
                junkify_prob: 0.5 + rf() * 0.3,
                max_junkified_nodes: 300 + (rf() * 400.0) as usize,
            };

            let corpus_count =
                (config.trajectories_per_round as f32 * config.corpus_fraction) as usize;
            let es_count = config.trajectories_per_round - corpus_count;

            let gen_start = std::time::Instant::now();

            let mut trajectories = generate_trajectory_batch_parallel(
                &model,
                templates,
                rules,
                es_count,
                round_seed,
                config.threshold,
                config.max_steps,
                &gen_config,
                Some(effective_workers),
            );

            if corpus_count > 0 && !corpus_exprs.is_empty() {
                trajectories.extend(generate_corpus_trajectories_parallel(
                    &model,
                    templates,
                    rules,
                    corpus_exprs,
                    corpus_count,
                    round_seed.wrapping_add(0xC0),
                    config.threshold,
                    config.max_steps,
                    Some(effective_workers),
                ));
            }
            gen_elapsed_s = gen_start.elapsed().as_secs_f64();

            // Filter empty trajectories
            let mut trajectories: Vec<_> = trajectories
                .into_iter()
                .filter(|t| !t.steps.is_empty())
                .collect();

            if trajectories.is_empty() {
                logln!(
                    "[SERVER] Trial {trial_id} round {}: no valid trajectories, skipping",
                    round + 1
                );
                round_metrics.push(serde_json::json!({
                    "round": round,
                    "skipped": true,
                    "mode": "online",
                }));
                continue;
            }

            // Replace NaN/non-finite jit_cost_ns with penalty
            let max_cost_ns = trajectories
                .iter()
                .flat_map(|t| t.steps.iter())
                .filter_map(|s| {
                    if s.jit_cost_ns.is_finite() {
                        Some(s.jit_cost_ns)
                    } else {
                        None
                    }
                })
                .fold(0.0f64, f64::max);
            let penalty_cost_ns = (max_cost_ns * 2.0).max(100.0);

            for traj in &mut trajectories {
                for step in &mut traj.steps {
                    if !step.jit_cost_ns.is_finite() || step.jit_cost_ns < 0.0 {
                        step.jit_cost_ns = penalty_cost_ns;
                    }
                }
            }

            total_steps = trajectories.iter().map(|t| t.steps.len()).sum();
            trajectory_count = trajectories.len();

            // ── PHASE 2: EXPORT ───────────────────────────────────────────
            let traj_path = trial_dir.join(format!("trajectories_r{round}.pftraj"));
            write_trajectories_binary(&trajectories, &traj_path);

            // ── PHASE 3: CRITIQUE ─────────────────────────────────────────
            let adv_path = trial_dir.join(format!("advantages_r{round}.pfadv"));
            let _ = std::fs::remove_file(&adv_path);

            let traj_path_str = traj_path
                .to_str()
                .unwrap_or_else(|| panic!("[SERVER] Invalid UTF-8 in traj path"));
            let adv_path_str = adv_path
                .to_str()
                .unwrap_or_else(|| panic!("[SERVER] Invalid UTF-8 in adv path"));

            step_critic(critic_url, traj_path_str, adv_path_str);

            // ── PHASE 4: UPDATE ──────────────────────────────────────────
            let advantages = read_advantages_binary(&adv_path);

            if advantages.len() != trajectories.len() {
                panic!(
                    "[SERVER] Trial {trial_id} round {}: trajectory/advantage count mismatch: {} vs {}",
                    round,
                    trajectories.len(),
                    advantages.len(),
                );
            }

            replay_buffer.push_round(&trajectories, &advantages);

            const MAX_REASONABLE_NS: f64 = 1_000_000_000.0;
            let mut speedups: Vec<f64> = trajectories
                .iter()
                .filter(|t| {
                    t.initial_cost_ns.is_finite()
                        && t.initial_cost_ns > 0.0
                        && t.initial_cost_ns < MAX_REASONABLE_NS
                        && t.final_cost_ns.is_finite()
                        && t.final_cost_ns > 0.0
                        && t.final_cost_ns < MAX_REASONABLE_NS
                })
                .map(|t| t.initial_cost_ns / t.final_cost_ns)
                .collect();
            speedups
                .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            speedup_median = if speedups.is_empty() {
                f64::NAN
            } else {
                let mid = speedups.len() / 2;
                if speedups.len() % 2 == 0 {
                    (speedups[mid - 1] + speedups[mid]) / 2.0
                } else {
                    speedups[mid]
                }
            };
        }

        let annealed_entropy = (config.entropy_coeff * (1.0 - round as f32 / config.rounds as f32))
            .max(config.entropy_floor);

        // Mini-batch gradient steps from replay buffer
        let mut round_grad_norm = 0.0f32;
        let mut round_clipped_grad_norm = 0.0f32;
        let mut total_policy_steps = 0usize;
        let mut total_value_steps = 0usize;

        for update_idx in 0..config.updates_per_round {
            let batch_indices = replay_buffer.sample_batch(
                config.mini_batch_size,
                round_seed.wrapping_add(update_idx as u64 * 7919),
            );
            let raw_advantages: Vec<f32> = batch_indices
                .iter()
                .map(|&i| replay_buffer.steps[i].advantage)
                .collect();
            let norm_advantages = normalize_advantages(&raw_advantages);

            let mut grads = UnifiedGradients::zero();
            let mut batch_policy = 0usize;
            let mut batch_value = 0usize;

            for (pos, &idx) in batch_indices.iter().enumerate() {
                let step = &replay_buffer.steps[idx];

                assert!(
                    step.jit_cost_ns.is_finite() && step.jit_cost_ns >= 0.0,
                    "[SERVER] NaN/negative jit_cost_ns={} at index {idx}",
                    step.jit_cost_ns,
                );

                let acc = acc_from_replay(step);
                let gacc = gacc_from_replay(step);
                let rule_embed = embed_from_replay(step);
                let cache = forward_cached(&model, &acc, &gacc, &rule_embed);

                backward_policy(
                    &model,
                    &cache,
                    &rule_embed,
                    step.matched,
                    norm_advantages[pos],
                    annealed_entropy,
                    config.miss_penalty,
                    &mut grads,
                );
                batch_policy += 1;

                let target = log_ns(step.jit_cost_ns);
                backward_value(&model, &cache, target, config.value_coeff, &mut grads);
                batch_value += 1;

                if !step.edges.is_empty() {
                    let d_acc_v =
                        compute_d_acc_input_value(&model, &cache, target, config.value_coeff);
                    backward_through_accumulator(&d_acc_v, &step.edges, acc.node_count, &mut grads);
                }
            }

            let batch_size = batch_policy.max(1) as f32;
            grads.scale(1.0 / batch_size);
            let clip_stats = grads.clip_stats(config.grad_clip);
            let grad_norm = clip_stats.raw_norm;
            round_grad_norm += grad_norm;
            round_clipped_grad_norm += clip_stats.clipped_norm;

            apply_unified_sgd(
                &mut model,
                &grads,
                &mut momentum_buf,
                config.lr,
                config.momentum,
                config.weight_decay,
                config.grad_clip,
            );

            total_policy_steps += batch_policy;
            total_value_steps += batch_value;
        }

        let avg_grad_norm = round_grad_norm / config.updates_per_round as f32;
        let avg_clipped_grad_norm = round_clipped_grad_norm / config.updates_per_round as f32;

        // ── Eval metrics ─────────────────────────────────────────────
        let eval_indices = replay_buffer.sample_batch(
            config.mini_batch_size.min(replay_buffer.len()),
            round_seed.wrapping_add(0xE7A1),
        );
        let mut extraction_loss_sum = 0.0f64;
        let mut saturation_loss_sum = 0.0f64;
        for &idx in &eval_indices {
            let step = &replay_buffer.steps[idx];
            let acc = acc_from_replay(step);
            let gacc = gacc_from_replay(step);
            let rule_embed = embed_from_replay(step);
            let cache = forward_cached(&model, &acc, &gacc, &rule_embed);

            let target = log_ns(step.jit_cost_ns);
            let extraction_err = cache.value_pred - target;
            extraction_loss_sum += (extraction_err * extraction_err) as f64;

            let p = (cache.prob as f64).clamp(1e-7, 1.0 - 1e-7);
            saturation_loss_sum += -libm::log(p);
        }
        let avg_extraction_loss = extraction_loss_sum / eval_indices.len().max(1) as f64;
        let avg_saturation_loss = saturation_loss_sum / eval_indices.len().max(1) as f64;

        // Track best score (lowest extraction loss)
        if avg_extraction_loss < best_score {
            best_score = avg_extraction_loss;
        }

        let round_elapsed = round_start.elapsed();

        let metrics = serde_json::json!({
            "round": round,
            "mode": if config.offline { "offline" } else { "online" },
            "trajectories": trajectory_count,
            "total_steps": total_steps,
            "speedup_median": speedup_median,
            "avg_extraction_loss": avg_extraction_loss,
            "avg_saturation_loss": avg_saturation_loss,
            "entropy_coeff": annealed_entropy,
            "grad_norm": avg_grad_norm,
            "grad_norm_raw": avg_grad_norm,
            "grad_norm_clipped": avg_clipped_grad_norm,
            "grad_clip_threshold": config.grad_clip,
            "buffer_size": replay_buffer.len(),
            "policy_steps": total_policy_steps,
            "value_steps": total_value_steps,
            "gen_elapsed_s": gen_elapsed_s,
            "rss_mb": rss_mb(),
            "elapsed_s": round_elapsed.as_secs_f64(),
        });
        round_metrics.push(metrics);

        // Local time from `date +%T` — always correct regardless of DST.
        let hms = std::process::Command::new("date")
            .arg("+%T")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|_| "??:??:??".into());
        let hms = &hms;

        logln!(
            "[{hms}] [SERVER] Trial {trial_id} round {}/{}: extraction_loss={:.4} saturation_loss={:.4} \
             speedup={:.2}x grad={:.4} time={:.1}s mode={}",
            round + 1,
            config.rounds,
            avg_extraction_loss,
            avg_saturation_loss,
            speedup_median,
            avg_grad_norm,
            round_elapsed.as_secs_f64(),
            if config.offline { "offline" } else { "online" },
        );
    }

    // Keep trajectory data — it's expensive to generate and can be reused
    // for offline training. Only clean up the critic checkpoint (it's saved
    // separately) and advantage files (regenerated by critic on demand).
    for round in 0..config.rounds {
        let _ = std::fs::remove_file(trial_dir.join(format!("advantages_r{round}.pfadv")));
    }
    let _ = std::fs::remove_file(&critic_ckpt);

    // ── Validation on held-out real shader expressions ────────────────
    // Score on expressions the training corpus does NOT contain.
    // This prevents the optimizer from gaming the metric on easy synthetics.
    let val_start = std::time::Instant::now();
    let val_speedup = validate_on_shaders(&model);
    logln!(
        "[SERVER] Trial {trial_id} validation: val_speedup={val_speedup:.3}x in {:.1}s",
        val_start.elapsed().as_secs_f64(),
    );

    serde_json::json!({
        "metrics": round_metrics,
        "best_score": best_score,
        "val_speedup": val_speedup,
    })
}

/// Unix socket server for Optuna hyperparameter sweeps.
///
/// Loads the expensive corpus + rules once, then accepts trial configs as
/// JSON lines over a Unix domain socket. Each connection = one trial.
/// Sequential processing (one trial at a time) since Optuna is single-threaded.
fn run_server(args: &Args, socket_path: &Path) {
    use std::os::unix::net::UnixListener;

    logln!("[SERVER] Loading rules + templates...");
    let rules: Vec<Box<dyn Rewrite>> = all_rules();
    let templates = build_rule_templates(&rules);
    logln!(
        "[SERVER] {} rules, {} templates loaded",
        rules.len(),
        templates.len()
    );

    logln!(
        "[SERVER] Loading corpus from {:?} (max={})...",
        args.corpus_path,
        args.corpus_max
    );
    let corpus_load_start = std::time::Instant::now();
    let corpus_exprs = load_corpus_exprs(&args.corpus_path, args.corpus_max, args.seed);
    logln!(
        "[SERVER] Loaded {} corpus expressions in {:.1}s",
        corpus_exprs.len(),
        corpus_load_start.elapsed().as_secs_f64(),
    );
    logln!("[SERVER] RSS after corpus load: {:.0}MB", rss_mb());

    // Ensure critic server is running
    let critic_ckpt_str = args.critic_checkpoint.to_str().unwrap_or_else(|| {
        panic!(
            "[SERVER] Invalid UTF-8 in critic checkpoint path: {:?}",
            args.critic_checkpoint
        )
    });
    let mut critic_child = ensure_critic_server(
        &args.critic_url,
        critic_ckpt_str,
        args.critic_lr,
        args.critic_dropout,
    );

    // Create output directory
    std::fs::create_dir_all(&args.output_dir).unwrap_or_else(|e| {
        panic!(
            "[SERVER] Failed to create output dir {:?}: {e}",
            args.output_dir
        )
    });

    // Remove stale socket file
    if socket_path.exists() {
        std::fs::remove_file(socket_path).unwrap_or_else(|e| {
            panic!(
                "[SERVER] Failed to remove stale socket {:?}: {e}",
                socket_path
            )
        });
    }

    let listener = UnixListener::bind(socket_path)
        .unwrap_or_else(|e| panic!("[SERVER] Failed to bind unix socket {:?}: {e}", socket_path));

    logln!("[SERVER] Listening on {}", socket_path.display());

    let mut trial_id = 0usize;

    for stream_result in listener.incoming() {
        let stream = match stream_result {
            Ok(s) => s,
            Err(e) => {
                logln!("[SERVER] Failed to accept connection: {e}");
                continue;
            }
        };

        trial_id += 1;

        // Read one JSON line from the stream
        let mut reader = std::io::BufReader::new(&stream);
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => {
                logln!("[SERVER] Trial {trial_id}: empty connection (EOF), skipping");
                continue;
            }
            Ok(_) => {}
            Err(e) => {
                logln!("[SERVER] Trial {trial_id}: failed to read request: {e}");
                let error_resp = serde_json::json!({"error": format!("read failed: {e}")});
                let mut writer = std::io::BufWriter::new(&stream);
                let _ = writeln!(writer, "{}", error_resp);
                let _ = writer.flush();
                continue;
            }
        }

        let config: TrialConfig = match serde_json::from_str(line.trim()) {
            Ok(c) => c,
            Err(e) => {
                logln!("[SERVER] Trial {trial_id}: invalid JSON config: {e}");
                let error_resp = serde_json::json!({"error": format!("invalid config JSON: {e}")});
                let mut writer = std::io::BufWriter::new(&stream);
                let _ = writeln!(writer, "{}", error_resp);
                let _ = writer.flush();
                continue;
            }
        };

        logln!(
            "[SERVER] Trial {trial_id}: rounds={} traj={} lr={:.2e} momentum={:.3} \
             wd={:.2e} ent={:.3} seed={} offline={} max_files={} critic_ep={} critic_lr={:.2e} critic_do={:.3} critic_bs={}",
            config.rounds,
            config.trajectories_per_round,
            config.lr,
            config.momentum,
            config.weight_decay,
            config.entropy_coeff,
            config.seed,
            config.offline,
            config.max_trajectory_files,
            config.critic_epochs.unwrap_or(args.critic_epochs),
            config.critic_lr.unwrap_or(args.critic_lr),
            config.critic_dropout.unwrap_or(args.critic_dropout),
            config
                .critic_mini_batch_size
                .unwrap_or(args.critic_mini_batch_size),
        );

        let trial_start = std::time::Instant::now();

        // Catch panics so one bad trial doesn't kill the server
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_trial(
                trial_id,
                &config,
                &rules,
                &templates,
                &corpus_exprs,
                &args.critic_url,
                &args.output_dir,
                args.workers,
                args.critic_epochs,
                args.critic_lr,
                args.critic_dropout,
                args.critic_mini_batch_size,
            )
        }));

        let response = match result {
            Ok(metrics_json) => {
                let elapsed = trial_start.elapsed();
                let best = metrics_json["best_score"].as_f64().unwrap_or(f64::NAN);
                logln!(
                    "[SERVER] Trial {trial_id} complete: best_score={:.6} in {:.1}s",
                    best,
                    elapsed.as_secs_f64(),
                );
                metrics_json
            }
            Err(panic_payload) => {
                let msg = if let Some(s) = panic_payload.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "unknown panic".to_string()
                };
                logln!("[SERVER] Trial {trial_id} PANICKED: {msg}");
                serde_json::json!({
                    "error": msg,
                    "metrics": [],
                    "best_score": f64::INFINITY,
                })
            }
        };

        // Write JSON response + newline
        let mut writer = std::io::BufWriter::new(&stream);
        match writeln!(
            writer,
            "{}",
            serde_json::to_string(&response)
                .unwrap_or_else(|e| panic!("[SERVER] Failed to serialize response: {e}"))
        ) {
            Ok(()) => {
                let _ = writer.flush();
            }
            Err(e) => {
                logln!("[SERVER] Trial {trial_id}: failed to write response: {e}");
            }
        }
    }

    // Cleanup
    if let Some(ref mut child) = critic_child {
        logln!(
            "[SERVER] Shutting down critic server (pid={})...",
            child.id()
        );
        let _ = child.kill();
        let _ = child.wait();
        logln!("[SERVER] Critic server stopped.");
    }

    let _ = std::fs::remove_file(socket_path);
    logln!("[SERVER] Shut down.");
}
