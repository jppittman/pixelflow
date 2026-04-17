//! # Unified Self-Play Trajectory Payload Structs
//!
//! These structs define the IPC boundary between Rust (Actor) and Python (Critic).
//!
//! - Rust writes binary [`Trajectory`] batches during self-play hill-climbing.
//! - Python reads trajectories, trains a Causal Transformer Critic, and writes
//!   binary advantage batches back.
//! - Rust reads advantages to apply policy gradient updates to the mask head.

/// A single epoch-granular step in a self-play trajectory.
///
/// One step per epoch: the mask is a multi-dimensional action (all rules decided
/// simultaneously), applied via clean batch, then one full rebuild.
#[derive(Debug, Clone)]
pub struct TrajectoryStep {
    /// E-graph VSA state at epoch start (GRAPH_INPUT_DIM floats).
    pub graph_accumulator_state: Vec<f32>,
    /// Full mask decision: (rule_idx, action_prob, approved) for EVERY rule.
    pub mask: Vec<(usize, f32, bool)>,
    /// Per-rule union counts for approved rules that were attempted: (rule_idx, unions).
    pub rule_outcomes: Vec<(usize, u32)>,
    /// Remaining node budget: node_budget - egraph.node_count(). Markov state.
    pub budget_remaining: i32,
    /// Remaining epoch budget: effective_epochs - current_epoch - 1. Markov state.
    pub epochs_remaining: i32,
    /// Terminal JIT cost backfilled at trajectory end (nanoseconds).
    pub jit_cost_ns: f64,
}

/// A complete self-play trajectory with terminal cost.
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// Unique identifier for this trajectory.
    pub trajectory_id: String,
    /// Sequence of epoch-granular steps taken during hill-climbing.
    pub steps: Vec<TrajectoryStep>,
    /// JIT-benchmarked initial cost before any rewrites (nanoseconds).
    pub initial_cost_ns: f64,
    /// JIT-benchmarked execution time of the final compiled AST (nanoseconds).
    pub final_cost_ns: f64,
    /// Judge-estimated initial cost before any rewrites.
    pub initial_cost: Option<f32>,
    /// Judge-estimated final cost after all rewrites.
    pub final_cost: Option<f32>,
    /// Node count of the initial expression (before rewrites).
    pub initial_nodes: usize,
    /// E-graph node budget used for this trajectory.
    pub node_budget: usize,
    /// EdgeAccumulator of the INITIAL expression (for extraction head training).
    /// Paired with `initial_cost_ns` as target: "this expression → this cost."
    pub initial_accumulator_state: Vec<f32>,
    /// Edge list of the initial expression (for embedding gradient flow).
    pub initial_edges: Vec<(u8, u8, u16)>,
    /// EdgeAccumulator of the FINAL extracted expression (for extraction head training).
    /// Paired with `final_cost_ns` as target: "this expression → this cost."
    pub final_accumulator_state: Vec<f32>,
    /// Edge list of the final expression (for embedding gradient flow).
    pub final_edges: Vec<(u8, u8, u16)>,
    /// Intermediate expression samples collected during saturation (p ≈ 0.1 per epoch).
    /// Each entry is (acc_state, edges, jit_cost_ns) for a mid-trajectory extraction.
    pub intermediate_pairs: Vec<(Vec<f32>, Vec<(u8, u8, u16)>, f64)>,
}

/// Per-trajectory advantage scores produced by the Python Critic.
#[derive(Debug, Clone)]
pub struct TrajectoryAdvantages {
    /// Index into the trajectory batch.
    pub trajectory_idx: usize,
    /// Per-step advantage A_t = R_T - V_t.
    pub advantages: Vec<f32>,
}
