//! # Unified Self-Play Trajectory Payload Structs
//!
//! These structs define the IPC boundary between Rust (Actor) and Python (Critic).
//!
//! - Rust writes binary [`Trajectory`] batches during self-play hill-climbing.
//! - Python reads trajectories, trains a Causal Transformer Critic, and writes
//!   binary advantage batches back.
//! - Rust reads advantages to apply policy gradient updates to the mask head.

/// A single step in a self-play trajectory.
#[derive(Debug, Clone)]
pub struct TrajectoryStep {
    /// EdgeAccumulator state at time t: 4*K dual sums + edge_count + node_count = 130 floats.
    pub accumulator_state: Vec<f32>,
    /// Expression embedding at decision time (EMBED_DIM floats).
    /// This is one side of the bilinear product: score = expr_embed^T @ M @ rule_embed.
    pub expression_embedding: Vec<f32>,
    /// Rule embedding for the selected rule (EMBED_DIM floats).
    pub rule_embedding: Vec<f32>,
    /// Remaining node budget: node_budget - egraph.node_count(). Markov state.
    pub budget_remaining: i32,
    /// Remaining epoch budget: epoch_budget - current_epoch. Markov state.
    pub epochs_remaining: i32,
    /// sigmoid(bilinear_score) — the Actor's confidence in this rule.
    pub action_probability: f32,
    /// Whether the rule actually matched and was applied.
    pub matched: bool,
    /// JIT-benchmarked cost of the expression at this epoch (nanoseconds).
    /// All steps within the same epoch share this value.
    pub jit_cost_ns: f64,
    /// Edge list for embedding gradient flow: (parent_op, child_op, depth).
    /// Compact representation of the expression structure at decision time.
    /// Each tuple is (parent OpKind index as u8, child OpKind index as u8, effective depth as u16).
    pub edges: Vec<(u8, u8, u16)>,
    /// Graph accumulator state (VSA encoding) at decision time (GRAPH_INPUT_DIM floats).
    /// Used by the graph backbone for mask scoring (separate from expression EdgeAccumulator).
    pub graph_accumulator_state: Vec<f32>,
}

/// A complete self-play trajectory with terminal cost.
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// Unique identifier for this trajectory.
    pub trajectory_id: String,
    /// Sequence of steps taken during hill-climbing.
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
}

/// Per-trajectory advantage scores produced by the Python Critic.
#[derive(Debug, Clone)]
pub struct TrajectoryAdvantages {
    /// Index into the trajectory batch.
    pub trajectory_idx: usize,
    /// Per-step advantage A_t = R_T - V_t.
    pub advantages: Vec<f32>,
}
