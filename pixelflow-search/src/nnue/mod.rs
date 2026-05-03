//! # NNUE for Instruction Selection
//!
//! An Efficiently Updatable Neural Network for compiler instruction selection,
//! inspired by Stockfish's NNUE approach to chess position evaluation.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![allow(dead_code)] // Prototype code
#![allow(clippy::only_used_in_recursion)]

extern crate alloc;

pub mod factored;
pub mod training;
pub mod window;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use libm::fabsf;

/// Re-export canonical IR types as the source of truth.
pub use pixelflow_ir::{Expr, OpKind};

/// Re-export ExprNnue for dual-head AlphaZero-style architecture.
pub use factored::ExprNnue;

/// Re-export key types from factored module.
pub use factored::{EdgeAccumulator, GraphAccumulator, OpEmbeddings};

/// Re-export InstructionWindow for sliding-window scheduling.
pub use window::InstructionWindow;

/// Re-export unified mask architecture constants and types.
pub use factored::{
    EMBED_DIM, MLP_HIDDEN, MASK_INPUT_DIM, RULE_FEATURE_DIM, MASK_MAX_RULES, RULE_CONCAT_DIM,
    GRAPH_ACC_DIM, GRAPH_INPUT_DIM,
    RuleFeatures, RuleTemplates,
};

/// Re-export training utilities.
pub use training::{Metrics, ResourceConfig, TrainingResult};

// Note: ExprGenConfig, ExprGenerator, BwdGenConfig, BwdGenerator, BwdTrainingPair
// are already public structs defined in this module - no re-export needed.

// ============================================================================
// HalfEP Features (Legacy - being phased out in favor of Factored)
// ============================================================================

/// Maximum depth we encode in features.
pub const MAX_DEPTH: usize = 8;

/// A HalfEP feature: (perspective_op, descendant_op, depth, child_path).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HalfEPFeature {
    /// The operation type we're evaluating from
    pub perspective_op: u8,
    /// The descendant operation type
    pub descendant_op: u8,
    /// Relative depth from perspective
    pub depth: u8,
    /// Child path encoding
    pub path: u8,
}

impl HalfEPFeature {
    /// Total number of possible features.
    pub const COUNT: usize = OpKind::COUNT * OpKind::COUNT * MAX_DEPTH * 256;

    /// Convert to a unique index for the feature vector.
    #[must_use]
    pub fn to_index(self) -> usize {
        let p = self.perspective_op as usize;
        let d = self.descendant_op as usize;
        let depth = self.depth as usize;
        let path = self.path as usize;

        ((p * OpKind::COUNT + d) * MAX_DEPTH + depth) * 256 + path
    }

    /// Create from a unique index.
    #[must_use]
    pub fn from_index(idx: usize) -> Self {
        let path = (idx % 256) as u8;
        let idx = idx / 256;
        let depth = (idx % MAX_DEPTH) as u8;
        let idx = idx / MAX_DEPTH;
        let descendant_op = (idx % OpKind::COUNT) as u8;
        let perspective_op = (idx / OpKind::COUNT) as u8;

        Self {
            perspective_op,
            descendant_op,
            depth,
            path,
        }
    }
}

/// Extract HalfEP features from an expression.
#[must_use]
pub fn extract_features(expr: &Expr) -> Vec<HalfEPFeature> {
    let mut features = Vec::new();
    extract_features_recursive(expr, &mut features, 0, 0);
    features
}

fn extract_features_recursive(
    expr: &Expr,
    features: &mut Vec<HalfEPFeature>,
    path: u8,
    depth: u8,
) {
    let root_op = expr.kind();

    // Add features for all descendants from this node's perspective
    add_descendant_features(expr, features, root_op as u8, 0, 0);

    // Recurse into children
    match expr {
        Expr::Var(_) | Expr::Const(_) => {}
        Expr::Param(i) => panic!("Expr::Param({}) reached NNUE cost model — call substitute_params before use", i),
        Expr::Unary(_, a) => {
            extract_features_recursive(a, features, path, depth.saturating_add(1));
        }
        Expr::Binary(_, a, b) => {
            extract_features_recursive(a, features, path << 1, depth.saturating_add(1));
            extract_features_recursive(b, features, (path << 1) | 1, depth.saturating_add(1));
        }
        Expr::Ternary(_, a, b, c) => {
            extract_features_recursive(a, features, path, depth.saturating_add(1));
            extract_features_recursive(b, features, path, depth.saturating_add(1));
            extract_features_recursive(c, features, path, depth.saturating_add(1));
        }
        Expr::Nary(_, children) => {
            for (i, child) in children.iter().enumerate() {
                let child_path = (path << 3) | (i as u8 & 0x7);
                extract_features_recursive(child, features, child_path, depth.saturating_add(1));
            }
        }
    }
}

fn add_descendant_features(
    expr: &Expr,
    features: &mut Vec<HalfEPFeature>,
    perspective_op: u8,
    depth: u8,
    path: u8,
) {
    if depth as usize >= MAX_DEPTH {
        return;
    }

    features.push(HalfEPFeature {
        perspective_op,
        descendant_op: expr.kind() as u8,
        depth,
        path,
    });

    match expr {
        Expr::Var(_) | Expr::Const(_) => {}
        Expr::Param(i) => panic!("Expr::Param({}) reached NNUE cost model — call substitute_params before use", i),
        Expr::Unary(_, a) => {
            add_descendant_features(a, features, perspective_op, depth + 1, path << 1);
        }
        Expr::Binary(_, a, b) => {
            add_descendant_features(a, features, perspective_op, depth + 1, path << 1);
            add_descendant_features(b, features, perspective_op, depth + 1, (path << 1) | 1);
        }
        Expr::Ternary(_, a, b, c) => {
            add_descendant_features(a, features, perspective_op, depth + 1, path << 2);
            add_descendant_features(b, features, perspective_op, depth + 1, (path << 2) | 1);
            add_descendant_features(c, features, perspective_op, depth + 1, (path << 2) | 2);
        }
        Expr::Nary(_, children) => {
            for (i, child) in children.iter().enumerate() {
                let child_path = (path << 3) | (i as u8 & 0x7);
                add_descendant_features(child, features, perspective_op, depth + 1, child_path);
            }
        }
    }
}

// ============================================================================
// Dense Features
// ============================================================================

/// Dense features extracted from an expression for ILP-aware evaluation.
#[derive(Clone, Debug, Default)]
pub struct DenseFeatures {
    /// Dense feature values (21 features).
    pub values: [i32; Self::COUNT],
}

impl DenseFeatures {
    /// Number of dense features
    pub const COUNT: usize = 20;

    // Feature indices for direct array access
    /// Addition count index
    pub const ADD: usize = 0;
    /// Subtraction count index
    pub const SUB: usize = 1;
    /// Multiplication count index
    pub const MUL: usize = 2;
    /// Division count index
    pub const DIV: usize = 3;
    /// Negation count index
    pub const NEG: usize = 4;
    /// Square root count index
    pub const SQRT: usize = 5;
    /// Reciprocal square root count index
    pub const RSQRT: usize = 6;
    /// Absolute value count index
    pub const ABS: usize = 7;
    /// Minimum count index
    pub const MIN: usize = 8;
    /// Maximum count index
    pub const MAX: usize = 9;
    /// Fused multiply-add count index
    pub const FMA: usize = 10;
    /// Total node count index
    pub const NODE_COUNT: usize = 11;
    /// Expression depth index
    pub const DEPTH: usize = 12;
    /// Variable reference count index
    pub const VAR_COUNT: usize = 13;
    /// Constant count index
    pub const CONST_COUNT: usize = 14;
    /// Has identity pattern (x*1, x+0) index
    pub const HAS_IDENTITY: usize = 15;
    /// Has self-cancel pattern (x-x, x/x) index
    pub const HAS_SELF_CANCEL: usize = 16;
    /// Has fusable pattern (a*b+c) index
    pub const HAS_FUSABLE: usize = 17;
    /// Critical path cost (ILP) index
    pub const CRITICAL_PATH: usize = 18;
    /// Maximum width (register pressure) index
    pub const MAX_WIDTH: usize = 19;

    /// Feature names for debugging
    pub const NAMES: [&'static str; Self::COUNT] = [
        "add", "sub", "mul", "div", "neg",
        "sqrt", "rsqrt", "abs", "min", "max",
        "fma",
        "nodes", "depth", "vars", "consts",
        "has_identity", "has_self_cancel", "has_fusable",
        "critical_path", "max_width",
    ];

    /// Get feature value by index
    #[inline]
    #[must_use]
    pub fn get(&self, i: usize) -> i32 {
        self.values.get(i).copied().unwrap_or(0)
    }

    /// Set feature value by index
    #[inline]
    pub fn set(&mut self, i: usize, v: i32) {
        if i < Self::COUNT {
            self.values[i] = v;
        }
    }
}

/// Extract dense features from an expression (ILP-aware).
#[must_use]
pub fn extract_dense_features(expr: &Expr) -> DenseFeatures {
    let mut features = DenseFeatures::default();
    let mut width_at_depth = Vec::new();
    let critical_path = extract_dense_recursive(expr, &mut features, 0, &mut width_at_depth);
    features.set(DenseFeatures::CRITICAL_PATH, critical_path);
    features.set(DenseFeatures::MAX_WIDTH, width_at_depth.iter().copied().max().unwrap_or(0));
    features
}

/// Recursive helper for dense feature extraction.
/// Returns the critical path cost of this subtree.
fn extract_dense_recursive(
    expr: &Expr,
    features: &mut DenseFeatures,
    depth: usize,
    width_at_depth: &mut Vec<i32>,
) -> i32 {
    features.values[DenseFeatures::NODE_COUNT] += 1;
    let current_depth = features.values[DenseFeatures::DEPTH];
    features.values[DenseFeatures::DEPTH] = current_depth.max(depth as i32 + 1);

    // Track width at each depth level
    if depth >= width_at_depth.len() {
        width_at_depth.resize(depth + 1, 0);
    }
    width_at_depth[depth] += 1;

    match expr {
        Expr::Var(_) => {
            features.values[DenseFeatures::VAR_COUNT] += 1;
            0 // No latency for variable access
        }
        Expr::Const(_) => {
            features.values[DenseFeatures::CONST_COUNT] += 1;
            0 // No latency for constant
        }
        Expr::Param(i) => panic!("Expr::Param({}) reached NNUE cost model — call substitute_params before use", i),
        Expr::Unary(op, a) => {
            let op_cost = match op {
                OpKind::Neg => { features.values[DenseFeatures::NEG] += 1; 1 }
                OpKind::Sqrt => { features.values[DenseFeatures::SQRT] += 1; 15 }
                OpKind::Rsqrt => { features.values[DenseFeatures::RSQRT] += 1; 5 }
                OpKind::Abs => { features.values[DenseFeatures::ABS] += 1; 1 }
                _ => 5,
            };
            let child_critical = extract_dense_recursive(a, features, depth + 1, width_at_depth);
            op_cost + child_critical
        }
        Expr::Binary(op, a, b) => {
            let op_cost = match op {
                OpKind::Add => {
                    features.values[DenseFeatures::ADD] += 1;
                    // Check for fusable: if 'a' is a Mul, this is a*b+c pattern
                    if matches!(a.as_ref(), Expr::Binary(OpKind::Mul, _, _)) {
                        features.values[DenseFeatures::HAS_FUSABLE] += 1;
                    }
                    // Check for identity: x + 0
                    if is_const_zero(b) || is_const_zero(a) {
                        features.values[DenseFeatures::HAS_IDENTITY] += 1;
                    }
                    4
                }
                OpKind::Sub => {
                    features.values[DenseFeatures::SUB] += 1;
                    // Check for self-cancel: x - x
                    if dense_exprs_equal(a, b) {
                        features.values[DenseFeatures::HAS_SELF_CANCEL] += 1;
                    }
                    4
                }
                OpKind::Mul => {
                    features.values[DenseFeatures::MUL] += 1;
                    // Check for identity: x * 1
                    if is_const_one(b) || is_const_one(a) {
                        features.values[DenseFeatures::HAS_IDENTITY] += 1;
                    }
                    // Check for fusable: x * rsqrt(y)
                    if matches!(b.as_ref(), Expr::Unary(OpKind::Rsqrt, _)) ||
                       matches!(a.as_ref(), Expr::Unary(OpKind::Rsqrt, _)) {
                        features.values[DenseFeatures::HAS_FUSABLE] += 1;
                    }
                    5
                }
                OpKind::Div => {
                    features.values[DenseFeatures::DIV] += 1;
                    // Check for self-cancel: x / x
                    if dense_exprs_equal(a, b) {
                        features.values[DenseFeatures::HAS_SELF_CANCEL] += 1;
                    }
                    15
                }
                OpKind::Min => { features.values[DenseFeatures::MIN] += 1; 4 }
                OpKind::Max => { features.values[DenseFeatures::MAX] += 1; 4 }
                _ => 5,
            };
            let crit_a = extract_dense_recursive(a, features, depth + 1, width_at_depth);
            let crit_b = extract_dense_recursive(b, features, depth + 1, width_at_depth);
            // Critical path = max of children (parallel execution) + this op
            op_cost + crit_a.max(crit_b)
        }
        Expr::Ternary(op, a, b, c) => {
            let op_cost = match op {
                OpKind::MulAdd => { features.values[DenseFeatures::FMA] += 1; 5 }
                _ => 10,
            };
            let crit_a = extract_dense_recursive(a, features, depth + 1, width_at_depth);
            let crit_b = extract_dense_recursive(b, features, depth + 1, width_at_depth);
            let crit_c = extract_dense_recursive(c, features, depth + 1, width_at_depth);
            // Critical path = max of all children + this op
            op_cost + crit_a.max(crit_b).max(crit_c)
        }
        Expr::Nary(op, children) => {
            // Tuple and similar n-ary ops are typically free (structural)
            let op_cost = match op {
                OpKind::Tuple => 0,
                _ => 5,
            };
            // Critical path = max of all children + this op
            let max_child_crit = children.iter()
                .map(|c| extract_dense_recursive(c, features, depth + 1, width_at_depth))
                .max()
                .unwrap_or(0);
            op_cost + max_child_crit
        }
    }
}

fn is_const_zero(expr: &Expr) -> bool {
    matches!(expr, Expr::Const(c) if fabsf(*c) < 1e-10)
}

fn is_const_one(expr: &Expr) -> bool {
    matches!(expr, Expr::Const(c) if fabsf(*c - 1.0) < 1e-10)
}

fn dense_exprs_equal(a: &Expr, b: &Expr) -> bool {
    match (a, b) {
        (Expr::Var(i), Expr::Var(j)) => i == j,
        (Expr::Const(x), Expr::Const(y)) => fabsf(x - y) < 1e-10,
        (Expr::Unary(op1, a1), Expr::Unary(op2, b1)) => {
            op1 == op2 && dense_exprs_equal(a1, b1)
        }
        (Expr::Binary(op1, a1, a2), Expr::Binary(op2, b1, b2)) => {
            op1 == op2 && dense_exprs_equal(a1, b1) && dense_exprs_equal(a2, b2)
        }
        (Expr::Ternary(op1, a1, a2, a3), Expr::Ternary(op2, b1, b2, b3)) => {
            op1 == op2 && dense_exprs_equal(a1, b1) && dense_exprs_equal(a2, b2) && dense_exprs_equal(a3, b3)
        }
        (Expr::Nary(op1, c1), Expr::Nary(op2, c2)) => {
            op1 == op2 && c1.len() == c2.len() &&
            c1.iter().zip(c2.iter()).all(|(a, b)| dense_exprs_equal(a, b))
        }
        // Different variants are never equal
        (Expr::Var(_), _) | (Expr::Const(_), _) | (Expr::Unary(_, _), _) |
        (Expr::Binary(_, _, _), _) | (Expr::Ternary(_, _, _, _), _) | (Expr::Nary(_, _), _) => false,
        (Expr::Param(i), _) | (_, Expr::Param(i)) => panic!("Expr::Param({}) reached NNUE cost model — call substitute_params before use", i),
    }
}


// ============================================================================
// NNUE Network Architecture
// ============================================================================

/// NNUE network configuration.
///
/// ## Hybrid Architecture
///
/// ```text
/// Sparse Input (HalfEP)     Dense Input (ILP Features)
///       │                           │
///       ▼                           ▼
/// ┌──────────┐               ┌──────────┐
/// │  W1      │               │ W_dense  │
/// │  (sparse)│               │ (dense)  │
/// │  → L1    │               │ → L_dense│
/// └────┬─────┘               └────┬─────┘
///      │                          │
///      └──────────┬───────────────┘
///                 │ (concat)
///                 ▼
///           ┌──────────┐
///           │   L2     │
///           │   L3     │
///           │   Output │
///           └──────────┘
/// ```
///
/// The dense branch allows NNUE to learn weights for ILP features
/// (critical_path, max_width) instead of hand-crafting them.
#[derive(Clone)]
pub struct NnueConfig {
    /// Size of the first hidden layer (sparse branch).
    pub l1_size: usize,
    /// Size of the dense feature layer.
    pub dense_size: usize,
    /// Size of the second hidden layer.
    pub l2_size: usize,
    /// Size of the third hidden layer.
    pub l3_size: usize,
}

impl Default for NnueConfig {
    fn default() -> Self {
        Self {
            l1_size: 256,
            dense_size: 32, // Smaller layer for 21 dense features
            l2_size: 32,
            l3_size: 32,
        }
    }
}

impl NnueConfig {
    /// Total size of combined layer (sparse L1 + dense branch)
    #[must_use]
    pub fn combined_size(&self) -> usize {
        self.l1_size + self.dense_size
    }
}

/// The NNUE network for expression cost prediction.
///
/// ## Hybrid Architecture
///
/// This is a hybrid network with two input branches:
///
/// 1. **Sparse branch**: HalfEP features → L1 (256)
///    - 401K sparse features for local structure
///    - Incrementally updatable accumulator
///
/// 2. **Dense branch**: DenseFeatures → L_dense (32)
///    - 21 dense features including ILP metrics
///    - Captures critical_path, max_width
///
/// The branches concatenate and flow through:
/// 3. L2 (32) - combined hidden
/// 4. L3 (32) - second hidden
/// 5. Output (1) - scalar cost prediction
#[derive(Clone)]
pub struct Nnue {
    /// Configuration.
    pub config: NnueConfig,

    // === Sparse Branch (HalfEP features) ===
    /// First layer weights: [feature_count, l1_size]
    /// Stored as column-major for efficient sparse updates.
    pub w1: Vec<i16>,
    /// First layer biases: [l1_size]
    pub b1: Vec<i32>,

    // === Dense Branch (ILP features) ===
    /// Dense layer weights: [DenseFeatures::COUNT, dense_size]
    pub w_dense: Vec<i16>,
    /// Dense layer biases: [dense_size]
    pub b_dense: Vec<i32>,

    // === Combined Layers ===
    /// Second layer weights: [l1_size + dense_size, l2_size]
    pub w2: Vec<i8>,
    /// Second layer biases: [l2_size]
    pub b2: Vec<i32>,
    /// Third layer weights: [l2_size, l3_size]
    pub w3: Vec<i8>,
    /// Third layer biases: [l3_size]
    pub b3: Vec<i32>,
    /// Output layer weights: [l3_size]
    pub w_out: Vec<i8>,
    /// Output layer bias
    pub b_out: i32,
}

impl Nnue {
    /// Create a new uninitialized NNUE network.
    #[must_use]
    pub fn new(config: NnueConfig) -> Self {
        let sparse_feature_count = HalfEPFeature::COUNT;
        let dense_feature_count = DenseFeatures::COUNT;
        let combined_size = config.combined_size();

        Self {
            // Sparse branch
            w1: alloc::vec![0i16; sparse_feature_count * config.l1_size],
            b1: alloc::vec![0i32; config.l1_size],

            // Dense branch (ILP features)
            w_dense: alloc::vec![0i16; dense_feature_count * config.dense_size],
            b_dense: alloc::vec![0i32; config.dense_size],

            // Combined layers
            w2: alloc::vec![0i8; combined_size * config.l2_size],
            b2: alloc::vec![0i32; config.l2_size],
            w3: alloc::vec![0i8; config.l2_size * config.l3_size],
            b3: alloc::vec![0i32; config.l3_size],
            w_out: alloc::vec![0i8; config.l3_size],
            b_out: 0,
            config,
        }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(NnueConfig::default())
    }

    /// Create a randomly initialized NNUE network.
    ///
    /// Uses He initialization scaled for quantized weights.
    /// This is essential - zero-initialized networks have zero gradients
    /// and never learn.
    #[must_use]
    pub fn new_random(config: NnueConfig, seed: u64) -> Self {
        let mut nnue = Self::new(config);
        nnue.randomize_weights(seed);
        nnue
    }

    /// Randomize weights for training.
    ///
    /// Uses a simple LCG for no_std compatibility.
    /// Weights are initialized to small random values that will
    /// produce non-zero activations after clipped ReLU.
    pub fn randomize_weights(&mut self, seed: u64) {
        let mut rng_state = seed.wrapping_add(1);

        // Inline LCG to avoid borrow issues
        macro_rules! next_u64 {
            () => {{
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1);
                rng_state
            }};
        }

        // W1 (sparse): Small values since many features may be active
        // NOTE: Must do modulo on u64 BEFORE casting to signed, or large u64
        // values wrap to negative i64 and give biased results!
        for w in &mut self.w1 {
            let r = next_u64!();
            *w = ((r % 129) as i16).wrapping_sub(64); // [-64, 64]
        }

        // B1: Small positive bias to start in active region of ReLU
        // clipped_relu(x) = (x >> 6).clamp(0, 127)
        // For output in [1, 100], need pre-activation in [64, 6400]
        // Start near the middle of this range
        for b in &mut self.b1 {
            let r = next_u64!();
            *b = (r % 1024) as i32 + 1024; // [1024, 2047]
        }

        // W_dense: Similar to W1
        for w in &mut self.w_dense {
            let r = next_u64!();
            *w = ((r % 257) as i16).wrapping_sub(128); // [-128, 128]
        }

        // B_dense: Small positive bias
        for b in &mut self.b_dense {
            let r = next_u64!();
            *b = (r % 1024) as i32 + 1024; // [1024, 2047]
        }

        // W2, W3: Larger values for smaller layers
        for w in &mut self.w2 {
            let r = next_u64!();
            *w = ((r % 65) as i8).wrapping_sub(32); // [-32, 32]
        }
        for b in &mut self.b2 {
            let r = next_u64!();
            *b = (r % 512) as i32 + 512; // [512, 1023]
        }

        for w in &mut self.w3 {
            let r = next_u64!();
            *w = ((r % 65) as i8).wrapping_sub(32); // [-32, 32]
        }
        for b in &mut self.b3 {
            let r = next_u64!();
            *b = (r % 512) as i32 + 512; // [512, 1023]
        }

        // Output layer
        for w in &mut self.w_out {
            let r = next_u64!();
            *w = ((r % 33) as i8).wrapping_sub(16); // [-16, 16]
        }
        // Output bias: start at 0, will learn the mean
        self.b_out = 0;
    }

    /// Evaluate dense features through the dense branch.
    ///
    /// Returns the dense layer activations (before ReLU).
    #[must_use]
    pub fn forward_dense(&self, features: &DenseFeatures) -> Vec<i32> {
        let dense_size = self.config.dense_size;
        let mut output = self.b_dense.clone();

        // Dense matrix multiply: output = W_dense * features + b_dense
        for (i, &feat_val) in features.values.iter().enumerate() {
            if feat_val == 0 {
                continue; // Skip zeros
            }
            let offset = i * dense_size;
            for (j, out_val) in output.iter_mut().enumerate().take(dense_size) {
                *out_val += (self.w_dense[offset + j] as i32) * feat_val;
            }
        }

        output
    }
}

/// Accumulator for incremental NNUE updates.
///
/// This is the key to NNUE efficiency: we maintain the output of the first
/// layer (the most expensive part) and incrementally update it when
/// features change.
///
/// The accumulator only handles the **sparse branch**. Dense features are
/// computed fresh each time (they're cheap - only 21 features).
#[derive(Clone)]
pub struct Accumulator {
    /// L1 activations from sparse branch (before clipped ReLU).
    pub values: Vec<i32>,
}

impl Accumulator {
    /// Create a new accumulator initialized with biases.
    #[must_use]
    pub fn new(nnue: &Nnue) -> Self {
        Self {
            values: nnue.b1.clone(),
        }
    }

    /// Reset to initial state (just biases).
    pub fn reset(&mut self, nnue: &Nnue) {
        self.values.copy_from_slice(&nnue.b1);
    }

    /// Add a feature to the accumulator.
    ///
    /// This is the incremental update: we only touch weights for this feature.
    #[inline]
    pub fn add_feature(&mut self, nnue: &Nnue, feature_idx: usize) {
        let l1_size = nnue.config.l1_size;
        let offset = feature_idx * l1_size;

        // Add the column of W1 corresponding to this feature
        for i in 0..l1_size {
            self.values[i] += nnue.w1[offset + i] as i32;
        }
    }

    /// Remove a feature from the accumulator.
    #[inline]
    pub fn remove_feature(&mut self, nnue: &Nnue, feature_idx: usize) {
        let l1_size = nnue.config.l1_size;
        let offset = feature_idx * l1_size;

        for i in 0..l1_size {
            self.values[i] -= nnue.w1[offset + i] as i32;
        }
    }

    /// Compute the full forward pass from the accumulator state (sparse only).
    ///
    /// Returns the predicted cost. Use `forward_hybrid` for full ILP-aware evaluation.
    #[must_use]
    pub fn forward(&self, nnue: &Nnue) -> i32 {
        // Create dummy dense features (all zeros)
        let dummy_dense = DenseFeatures::default();
        self.forward_hybrid(nnue, &dummy_dense)
    }

    /// Compute hybrid forward pass with both sparse accumulator and dense features.
    ///
    /// This is the main evaluation function for ILP-aware cost prediction.
    /// The dense features capture critical_path, max_width, etc.
    #[must_use]
    pub fn forward_hybrid(&self, nnue: &Nnue, dense_features: &DenseFeatures) -> i32 {
        let l1_size = nnue.config.l1_size;
        let dense_size = nnue.config.dense_size;
        let l2_size = nnue.config.l2_size;
        let l3_size = nnue.config.l3_size;

        // Compute dense branch output
        let dense_out = nnue.forward_dense(dense_features);

        // Concatenate sparse + dense for L2 input
        // Apply clipped ReLU to both branches
        let mut l2 = nnue.b2.clone();

        // Sparse branch contribution
        for (i, &val) in self.values.iter().enumerate().take(l1_size) {
            let a = (val >> 6).clamp(0, 127) as i8;
            for (j, l2_val) in l2.iter_mut().enumerate().take(l2_size) {
                *l2_val += (a as i32) * (nnue.w2[i * l2_size + j] as i32);
            }
        }

        // Dense branch contribution (offset by l1_size in W2)
        for (i, &val) in dense_out.iter().enumerate().take(dense_size) {
            let a = (val >> 6).clamp(0, 127) as i8;
            let w2_offset = (l1_size + i) * l2_size;
            for (j, l2_val) in l2.iter_mut().enumerate().take(l2_size) {
                *l2_val += (a as i32) * (nnue.w2[w2_offset + j] as i32);
            }
        }

        // L2 -> L3 with clipped ReLU
        let mut l3 = nnue.b3.clone();
        for (i, &val) in l2.iter().enumerate().take(l2_size) {
            let a = (val >> 6).clamp(0, 127) as i8;
            for (j, l3_val) in l3.iter_mut().enumerate().take(l3_size) {
                *l3_val += (a as i32) * (nnue.w3[i * l3_size + j] as i32);
            }
        }

        // L3 -> output
        let mut output = nnue.b_out;
        for (i, &val) in l3.iter().enumerate().take(l3_size) {
            let a = (val >> 6).clamp(0, 127) as i8;
            output += (a as i32) * (nnue.w_out[i] as i32);
        }

        output
    }

    /// Convenience method: evaluate an expression with both feature types.
    #[must_use]
    pub fn evaluate_expr(&self, nnue: &Nnue, expr: &Expr) -> i32 {
        let dense_features = extract_dense_features(expr);
        self.forward_hybrid(nnue, &dense_features)
    }
}

// ============================================================================
// Training Data Generation
// ============================================================================

/// A training sample: expression with its ground truth cost.
#[derive(Clone, Debug)]
pub struct TrainingSample {
    /// The expression (will be converted to features).
    pub expr: Expr,
    /// Ground truth cost in nanoseconds (from benchmarking).
    pub cost_ns: u64,
    /// Features extracted from the expression.
    pub features: Vec<HalfEPFeature>,
}

impl TrainingSample {
    /// Create a new training sample.
    #[must_use]
    pub fn new(features: Vec<HalfEPFeature>, cost_ns: u64) -> Self {
        Self {
            expr: Expr::Const(0.0), // Placeholder
            cost_ns,
            features,
        }
    }
}

/// A depth-limited training sample for Stockfish-style training.
///
/// This captures what cost is achievable within a rewrite budget,
/// enabling the NNUE to predict "what's possible in limited time"
/// rather than the theoretical optimum.
///
/// # Training Philosophy
///
/// Like Stockfish's NNUE, we train on what the search can actually
/// achieve rather than perfect evaluation. The NNUE learns:
/// - features(expr) -> achievable_cost(budget=N)
///
/// At inference time, MCTS uses this prediction to guide search.
#[derive(Clone, Debug)]
pub struct DepthLimitedSample {
    /// Original expression before optimization.
    pub expr: Expr,

    /// Features extracted from the expression.
    pub features: Vec<HalfEPFeature>,

    /// Initial cost before optimization (from CostModel).
    pub initial_cost: u32,

    /// Best cost achieved within the rewrite budget.
    pub achievable_cost: u32,

    /// The rewrite budget used for saturation.
    pub budget: u16,

    /// Whether saturation completed before budget exhausted.
    pub saturated: bool,
}

impl DepthLimitedSample {
    /// Create a new sample from an expression and saturation results.
    #[must_use]
    pub fn new(
        expr: Expr,
        initial_cost: u32,
        achievable_cost: u32,
        budget: u16,
        saturated: bool,
    ) -> Self {
        let features = extract_features(&expr);
        Self {
            expr,
            features,
            initial_cost,
            achievable_cost,
            budget,
            saturated,
        }
    }

    /// Calculate the cost improvement ratio.
    #[must_use]
    pub fn improvement_ratio(&self) -> f32 {
        if self.initial_cost == 0 {
            1.0
        } else {
            self.achievable_cost as f32 / self.initial_cost as f32
        }
    }

    /// Calculate absolute cost reduction.
    #[must_use]
    pub fn cost_reduction(&self) -> i32 {
        self.initial_cost as i32 - self.achievable_cost as i32
    }

    /// Serialize sample to binary format.
    ///
    /// Format:
    /// - initial_cost: u32
    /// - achievable_cost: u32
    /// - budget: u16
    /// - saturated: u8 (1 or 0)
    /// - feature_count: u16
    /// - features: [HalfEPFeature; feature_count]
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        bytes.extend_from_slice(&self.initial_cost.to_le_bytes());
        bytes.extend_from_slice(&self.achievable_cost.to_le_bytes());
        bytes.extend_from_slice(&self.budget.to_le_bytes());
        bytes.push(if self.saturated { 1 } else { 0 });

        let feature_count = self.features.len().min(u16::MAX as usize) as u16;
        bytes.extend_from_slice(&feature_count.to_le_bytes());

        for f in self.features.iter().take(feature_count as usize) {
            bytes.push(f.perspective_op);
            bytes.push(f.descendant_op);
            bytes.push(f.depth);
            bytes.push(f.path);
        }

        bytes
    }

    /// Deserialize sample from binary format.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Option<(Self, usize)> {
        if bytes.len() < 13 {
            return None;
        }

        let initial_cost = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let achievable_cost = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let budget = u16::from_le_bytes([bytes[8], bytes[9]]);
        let saturated = bytes[10] != 0;

        let feature_count = u16::from_le_bytes([bytes[11], bytes[12]]) as usize;

        let feature_bytes = 13 + feature_count * 4;
        if bytes.len() < feature_bytes {
            return None;
        }

        let mut features = Vec::with_capacity(feature_count);
        let mut offset = 13;
        for _ in 0..feature_count {
            features.push(HalfEPFeature {
                perspective_op: bytes[offset],
                descendant_op: bytes[offset + 1],
                depth: bytes[offset + 2],
                path: bytes[offset + 3],
            });
            offset += 4;
        }

        // We don't store the expression - just the features
        let sample = Self {
            expr: Expr::Const(0.0), // Placeholder
            features,
            initial_cost,
            achievable_cost,
            budget,
            saturated,
        };

        Some((sample, feature_bytes))
    }
}

// ============================================================================
// Binpack I/O (requires std)
// ============================================================================

/// Magic number for depth-limited binpack files.
pub const DEPTH_LIMITED_MAGIC: u32 = 0x444C4E55; // "DLNU" in ASCII

/// Version number for depth-limited binpack format.
pub const DEPTH_LIMITED_VERSION: u32 = 1;

/// Write a collection of samples to a binpack file.
///
/// Requires the `std` feature.
#[cfg(feature = "std")]
pub fn write_depth_limited_binpack(
    samples: &[DepthLimitedSample],
    path: &str,
) -> std::io::Result<()> {
    use std::io::Write;

    let mut file = std::fs::File::create(path)?;

    // Header
    file.write_all(&DEPTH_LIMITED_MAGIC.to_le_bytes())?;
    file.write_all(&DEPTH_LIMITED_VERSION.to_le_bytes())?;
    file.write_all(&(samples.len() as u32).to_le_bytes())?;

    // Samples
    for sample in samples {
        let bytes = sample.to_bytes();
        file.write_all(&(bytes.len() as u32).to_le_bytes())?;
        file.write_all(&bytes)?;
    }

    Ok(())
}

/// Read samples from a binpack file.
///
/// Requires the `std` feature.
#[cfg(feature = "std")]
pub fn read_depth_limited_binpack(path: &str) -> std::io::Result<Vec<DepthLimitedSample>> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut all_bytes = Vec::new();
    file.read_to_end(&mut all_bytes)?;

    if all_bytes.len() < 12 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "File too short for header",
        ));
    }

    let magic = u32::from_le_bytes([all_bytes[0], all_bytes[1], all_bytes[2], all_bytes[3]]);
    if magic != DEPTH_LIMITED_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid magic number",
        ));
    }

    let version = u32::from_le_bytes([all_bytes[4], all_bytes[5], all_bytes[6], all_bytes[7]]);
    if version != DEPTH_LIMITED_VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unsupported version: {}", version),
        ));
    }

    let count = u32::from_le_bytes([all_bytes[8], all_bytes[9], all_bytes[10], all_bytes[11]]) as usize;

    let mut samples = Vec::with_capacity(count);
    let mut offset = 12;

    for _ in 0..count {
        if offset + 4 > all_bytes.len() {
            break;
        }

        let sample_len = u32::from_le_bytes([
            all_bytes[offset],
            all_bytes[offset + 1],
            all_bytes[offset + 2],
            all_bytes[offset + 3],
        ]) as usize;
        offset += 4;

        if offset + sample_len > all_bytes.len() {
            break;
        }

        if let Some((sample, _)) = DepthLimitedSample::from_bytes(&all_bytes[offset..]) {
            samples.push(sample);
        }
        offset += sample_len;
    }

    Ok(samples)
}

/// Configuration for random expression generation.
#[derive(Clone, Debug)]
pub struct ExprGenConfig {
    /// Maximum depth of generated expressions.
    pub max_depth: usize,
    /// Probability of generating a leaf (var or const) vs operation.
    pub leaf_prob: f32,
    /// Number of variables available (0-3 for X,Y,Z,W).
    pub num_vars: usize,
    /// Whether to include fused operations.
    pub include_fused: bool,
}

impl Default for ExprGenConfig {
    fn default() -> Self {
        Self {
            max_depth: 6,
            leaf_prob: 0.3,
            num_vars: 4,
            include_fused: true,
        }
    }
}

/// Random expression generator for training data.
///
/// This is like Stockfish's position generator for self-play training data.
/// Op selection is driven by `OpKind::is_seed_op()` + `OpKind::arity()` so
/// adding a new variant to `OpKind` automatically includes it here.
pub struct ExprGenerator {
    /// Configuration.
    pub config: ExprGenConfig,
    /// Random state (simple LCG for no_std compatibility).
    state: u64,
    /// Cached seed ops, built once from OpKind.
    seed_ops: Vec<OpKind>,
}

impl ExprGenerator {
    /// Create a new generator with the given seed.
    #[must_use]
    pub fn new(seed: u64, config: ExprGenConfig) -> Self {
        let mut seed_ops = Vec::new();
        for i in 0..OpKind::COUNT {
            if let Some(op) = OpKind::from_index(i) {
                if op.is_seed_op() {
                    seed_ops.push(op);
                }
            }
        }
        assert!(!seed_ops.is_empty(), "No seed ops found in OpKind — is_seed_op() is broken");
        assert!(config.num_vars <= 4, "num_vars={} exceeds INPUT_REGS limit of 4", config.num_vars);
        Self { config, state: seed, seed_ops }
    }

    /// Generate a random f32 in [0, 1).
    fn rand_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.state >> 33) as f32 / (1u64 << 31) as f32
    }

    /// Generate a random usize in [0, max).
    fn rand_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        let val = (self.rand_f32() * max as f32) as usize;
        if val >= max { max - 1 } else { val }
    }

    /// Generate a random expression.
    #[must_use]
    pub fn generate(&mut self) -> Expr {
        self.generate_recursive(0)
    }

    fn generate_recursive(&mut self, depth: usize) -> Expr {
        // Force leaf at max depth or with probability leaf_prob
        if depth >= self.config.max_depth || self.rand_f32() < self.config.leaf_prob {
            if self.rand_f32() < 0.7 {
                // Variable
                Expr::Var(self.rand_usize(self.config.num_vars.min(4)) as u8)
            } else {
                // Constant (small values to avoid overflow)
                let val = self.rand_f32() * 10.0 - 5.0;
                Expr::Const(val)
            }
        } else {
            let n = self.seed_ops.len();
            let idx = self.rand_usize(n);
            let op = self.seed_ops[idx];
            match op.arity() {
                1 => Expr::Unary(op, Box::new(self.generate_recursive(depth + 1))),
                2 => Expr::Binary(
                    op,
                    Box::new(self.generate_recursive(depth + 1)),
                    Box::new(self.generate_recursive(depth + 1)),
                ),
                3 => Expr::Ternary(
                    op,
                    Box::new(self.generate_recursive(depth + 1)),
                    Box::new(self.generate_recursive(depth + 1)),
                    Box::new(self.generate_recursive(depth + 1)),
                ),
                _ => panic!("OpKind::{} has arity {} but is_seed_op() returned true", op.name(), op.arity()),
            }
        }
    }
}

// ============================================================================
// Rewrite Rules (as "Moves")
// ============================================================================

/// A rewrite rule that can be applied to expressions.
///
/// This is analogous to a chess move: it transforms one expression into
/// an equivalent one. The NNUE learns to evaluate which rewrites are good.
#[derive(Clone, Debug)]
pub enum RewriteRule {
    /// x + 0 → x
    AddZero,
    /// x * 1 → x
    MulOne,
    /// x * 0 → 0
    MulZero,
    /// x - x → 0
    SubSelf,
    /// x / x → 1
    DivSelf,
    /// --x → x
    DoubleNeg,
    /// x + x → 2 * x
    AddSelf,
    /// a * b + c → MulAdd(a, b, c)
    FuseToMulAdd,
    /// MulAdd(a, b, c) → a * b + c (unfuse)
    UnfuseMulAdd,
}

impl RewriteRule {
    /// All available rewrite rules.
    pub const ALL: &'static [RewriteRule] = &[
        RewriteRule::AddZero,
        RewriteRule::MulOne,
        RewriteRule::MulZero,
        RewriteRule::SubSelf,
        RewriteRule::DivSelf,
        RewriteRule::DoubleNeg,
        RewriteRule::AddSelf,
        RewriteRule::FuseToMulAdd,
        RewriteRule::UnfuseMulAdd,
    ];

    /// Try to apply this rule to an expression, returning the rewritten form.
    ///
    /// Returns None if the rule doesn't match.
    #[must_use]
    pub fn try_apply(&self, expr: &Expr) -> Option<Expr> {
        match self {
            RewriteRule::AddZero => match expr {
                Expr::Binary(OpKind::Add, a, b) => {
                    if matches!(b.as_ref(), Expr::Const(c) if *c == 0.0) {
                        Some(a.as_ref().clone())
                    } else if matches!(a.as_ref(), Expr::Const(c) if *c == 0.0) {
                        Some(b.as_ref().clone())
                    } else {
                        None
                    }
                }
                _ => None,
            },
            RewriteRule::MulOne => match expr {
                Expr::Binary(OpKind::Mul, a, b) => {
                    if matches!(b.as_ref(), Expr::Const(c) if *c == 1.0) {
                        Some(a.as_ref().clone())
                    } else if matches!(a.as_ref(), Expr::Const(c) if *c == 1.0) {
                        Some(b.as_ref().clone())
                    } else {
                        None
                    }
                }
                _ => None,
            },
            RewriteRule::MulZero => match expr {
                Expr::Binary(OpKind::Mul, _, b)
                    if matches!(b.as_ref(), Expr::Const(c) if *c == 0.0) =>
                {
                    Some(Expr::Const(0.0))
                }
                Expr::Binary(OpKind::Mul, a, _)
                    if matches!(a.as_ref(), Expr::Const(c) if *c == 0.0) =>
                {
                    Some(Expr::Const(0.0))
                }
                _ => None,
            },
            RewriteRule::SubSelf => match expr {
                Expr::Binary(OpKind::Sub, a, b) if exprs_equal(a, b) => Some(Expr::Const(0.0)),
                _ => None,
            },
            RewriteRule::DivSelf => match expr {
                Expr::Binary(OpKind::Div, a, b) if exprs_equal(a, b) => Some(Expr::Const(1.0)),
                _ => None,
            },
            RewriteRule::DoubleNeg => match expr {
                Expr::Unary(OpKind::Neg, inner) => match inner.as_ref() {
                    Expr::Unary(OpKind::Neg, x) => Some(x.as_ref().clone()),
                    _ => None,
                },
                _ => None,
            },
            RewriteRule::AddSelf => match expr {
                Expr::Binary(OpKind::Add, a, b) if exprs_equal(a, b) => Some(Expr::Binary(
                    OpKind::Mul,
                    Box::new(Expr::Const(2.0)),
                    a.clone(),
                )),
                _ => None,
            },
            RewriteRule::FuseToMulAdd => match expr {
                Expr::Binary(OpKind::Add, mul_expr, c) => match mul_expr.as_ref() {
                    Expr::Binary(OpKind::Mul, a, b) => {
                        Some(Expr::Ternary(OpKind::MulAdd, a.clone(), b.clone(), c.clone()))
                    }
                    _ => None,
                },
                _ => None,
            },
            RewriteRule::UnfuseMulAdd => match expr {
                Expr::Ternary(OpKind::MulAdd, a, b, c) => Some(Expr::Binary(
                    OpKind::Add,
                    Box::new(Expr::Binary(OpKind::Mul, a.clone(), b.clone())),
                    c.clone(),
                )),
                _ => None,
            },
        }
    }
}

/// Check if two expressions are structurally equal.
fn exprs_equal(a: &Expr, b: &Expr) -> bool {
    match (a, b) {
        (Expr::Var(i), Expr::Var(j)) => i == j,
        (Expr::Const(x), Expr::Const(y)) => fabsf(x - y) < 1e-10,
        (Expr::Unary(op1, a1), Expr::Unary(op2, a2)) => op1 == op2 && exprs_equal(a1, a2),
        (Expr::Binary(op1, a1, b1), Expr::Binary(op2, a2, b2)) => {
            op1 == op2 && exprs_equal(a1, a2) && exprs_equal(b1, b2)
        }
        (Expr::Ternary(op1, a1, b1, c1), Expr::Ternary(op2, a2, b2, c2)) => {
            op1 == op2 && exprs_equal(a1, a2) && exprs_equal(b1, b2) && exprs_equal(c1, c2)
        }
        (Expr::Nary(op1, c1), Expr::Nary(op2, c2)) => {
            op1 == op2 && c1.len() == c2.len() &&
            c1.iter().zip(c2.iter()).all(|(x, y)| exprs_equal(x, y))
        }
        // Different variants are never equal
        (Expr::Var(_), _) | (Expr::Const(_), _) | (Expr::Unary(_, _), _) |
        (Expr::Binary(_, _, _), _) | (Expr::Ternary(_, _, _, _), _) | (Expr::Nary(_, _), _) => false,
        (Expr::Param(i), _) | (_, Expr::Param(i)) => panic!("Expr::Param({}) reached NNUE cost model — call substitute_params before use", i),
    }
}

/// Find all applicable rewrites for an expression (at any position).
///
/// Returns (path_to_subexpr, rule, rewritten_expr) tuples.
#[must_use]
pub fn find_all_rewrites(expr: &Expr) -> Vec<(Vec<usize>, RewriteRule, Expr)> {
    let mut rewrites = Vec::new();
    find_rewrites_recursive(expr, &mut Vec::new(), &mut rewrites);
    rewrites
}

// ============================================================================
// Pattern Match + Substitute (for rule template rewriting)
// ============================================================================

/// Pattern-match `expr` against a `template` with `Var(n)` as metavariables.
///
/// Returns `None` if the pattern doesn't match.
/// Returns `Some(bindings)` mapping variable indices to matched sub-expressions.
///
/// # Panics
///
/// Panics if an `Expr::Param` is encountered in the template (templates should
/// only use `Var` as metavariables).
#[must_use]
pub fn pattern_match(expr: &Expr, template: &Expr) -> Option<BTreeMap<u8, Expr>> {
    let mut bindings = BTreeMap::new();
    if pattern_match_recursive(expr, template, &mut bindings) {
        Some(bindings)
    } else {
        None
    }
}

fn pattern_match_recursive(expr: &Expr, template: &Expr, bindings: &mut BTreeMap<u8, Expr>) -> bool {
    match template {
        // Var(n) is a metavariable -- bind or check consistency
        Expr::Var(n) => {
            if let Some(existing) = bindings.get(n) {
                *existing == *expr // PartialEq on Expr
            } else {
                bindings.insert(*n, expr.clone());
                true
            }
        }
        // Const must match exactly (within epsilon for floats)
        Expr::Const(c) => {
            matches!(expr, Expr::Const(e) if fabsf(e - c) < 1e-6)
        }
        // Structural match: kind must match, recurse children
        Expr::Unary(op_t, a_t) => {
            if let Expr::Unary(op_e, a_e) = expr {
                op_e == op_t && pattern_match_recursive(a_e, a_t, bindings)
            } else {
                false
            }
        }
        Expr::Binary(op_t, a_t, b_t) => {
            if let Expr::Binary(op_e, a_e, b_e) = expr {
                op_e == op_t
                    && pattern_match_recursive(a_e, a_t, bindings)
                    && pattern_match_recursive(b_e, b_t, bindings)
            } else {
                false
            }
        }
        Expr::Ternary(op_t, a_t, b_t, c_t) => {
            if let Expr::Ternary(op_e, a_e, b_e, c_e) = expr {
                op_e == op_t
                    && pattern_match_recursive(a_e, a_t, bindings)
                    && pattern_match_recursive(b_e, b_t, bindings)
                    && pattern_match_recursive(c_e, c_t, bindings)
            } else {
                false
            }
        }
        // Param -- not used in templates, but handle for completeness
        Expr::Param(i) => matches!(expr, Expr::Param(j) if i == j),
        Expr::Nary(op_t, children_t) => {
            if let Expr::Nary(op_e, children_e) = expr {
                op_e == op_t
                    && children_t.len() == children_e.len()
                    && children_t.iter().zip(children_e.iter()).all(|(t, e)| {
                        pattern_match_recursive(e, t, bindings)
                    })
            } else {
                false
            }
        }
    }
}

/// Substitute bindings into a template to produce a concrete expression.
///
/// Returns `None` if the template contains a `Var(n)` with no corresponding binding.
/// This can happen when the RHS template has variables not present in the LHS
/// (or vice versa), which means the rule is not applicable in this direction.
#[must_use]
pub fn substitute_template(template: &Expr, bindings: &BTreeMap<u8, Expr>) -> Option<Expr> {
    match template {
        Expr::Var(n) => bindings.get(n).cloned(),
        Expr::Const(c) => Some(Expr::Const(*c)),
        Expr::Param(i) => Some(Expr::Param(*i)),
        Expr::Unary(op, a) => {
            Some(Expr::Unary(*op, Box::new(substitute_template(a, bindings)?)))
        }
        Expr::Binary(op, a, b) => {
            let sa = substitute_template(a, bindings)?;
            let sb = substitute_template(b, bindings)?;
            Some(Expr::Binary(*op, Box::new(sa), Box::new(sb)))
        }
        Expr::Ternary(op, a, b, c) => {
            let sa = substitute_template(a, bindings)?;
            let sb = substitute_template(b, bindings)?;
            let sc = substitute_template(c, bindings)?;
            Some(Expr::Ternary(*op, Box::new(sa), Box::new(sb), Box::new(sc)))
        }
        Expr::Nary(op, children) => {
            let new_children: Option<Vec<_>> = children.iter()
                .map(|c| substitute_template(c, bindings))
                .collect();
            Some(Expr::Nary(*op, new_children?))
        }
    }
}

// ============================================================================
// Backward Generation (BWD) - Lample & Charton 2019
// ============================================================================

/// A training pair for backward generation.
///
/// Contains both the optimized form (target) and unoptimized form (input).
#[derive(Clone, Debug)]
pub struct BwdTrainingPair {
    /// The optimized expression (what we want the model to produce/recognize).
    pub optimized: Expr,
    /// The unoptimized expression (input to the model).
    pub unoptimized: Expr,
    /// Number of junkifying rewrites applied.
    pub rewrites_applied: usize,
}

/// Configuration for backward expression generation.
#[derive(Clone, Debug)]
pub struct BwdGenConfig {
    /// Maximum depth of generated optimized expressions.
    pub max_depth: usize,
    /// Probability of generating a leaf (var or const) vs operation.
    pub leaf_prob: f32,
    /// Number of variables available (0-3 for X,Y,Z,W).
    pub num_vars: usize,
    /// Probability of using a fused operation when generating.
    pub fused_op_prob: f32,
    /// Maximum number of junkifying rewrite passes to apply.
    pub max_junkify_passes: usize,
    /// Probability of applying a junkifying rewrite at each node.
    pub junkify_prob: f32,
    /// Maximum node count after junkification. Prevents exponential blowup
    /// from rules like Distributive that double subtree size per application.
    pub max_junkified_nodes: usize,
}

impl Default for BwdGenConfig {
    fn default() -> Self {
        Self {
            max_depth: 8,
            leaf_prob: 0.15,
            num_vars: 4,
            fused_op_prob: 0.1, // Low: mul_add is one op among many, not dominant
            max_junkify_passes: 4,
            junkify_prob: 0.7,
            max_junkified_nodes: 500,
        }
    }
}

/// Backward expression generator following Lample & Charton's approach.
///
/// Generates optimized expressions (with fused operations), then applies
/// junkifying rewrites (using all 41 rule templates in both directions)
/// to create equivalent but less efficient expressions.
///
/// This is the inverse of how we want the model to work:
/// - Generation: optimized → unoptimized (easy, deterministic)
/// - Inference: unoptimized → optimized (learned)
pub struct BwdGenerator {
    /// Configuration.
    pub config: BwdGenConfig,
    /// Random state.
    state: u64,
    /// Rule templates for pattern-matching junkification.
    templates: RuleTemplates,
}

impl BwdGenerator {
    /// Create a new backward generator with the given seed and rule templates.
    #[must_use]
    pub fn new(seed: u64, config: BwdGenConfig, templates: RuleTemplates) -> Self {
        assert!(config.num_vars <= 4, "num_vars={} exceeds INPUT_REGS limit of 4", config.num_vars);
        Self { config, state: seed, templates }
    }

    /// Generate a random f32 in [0, 1).
    fn rand_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.state >> 33) as f32 / (1u64 << 31) as f32
    }

    /// Generate a random usize in [0, max).
    fn rand_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        let val = (self.rand_f32() * max as f32) as usize;
        if val >= max { max - 1 } else { val }
    }

    /// Maximum retries when `generate_optimized` produces a variable-free
    /// expression. This is a safety net — with `min_depth >= 2` and 70%
    /// variable leaf probability, hitting this limit means the RNG is broken.
    const MAX_GENERATE_RETRIES: usize = 20;

    /// Generate a backward training pair.
    ///
    /// Returns (optimized, unoptimized) where unoptimized is derived from
    /// optimized via junkifying rewrites using all 41 rule templates.
    ///
    /// # Panics
    ///
    /// Panics if `MAX_GENERATE_RETRIES` consecutive attempts fail to produce
    /// an expression containing at least one variable. This indicates a
    /// configuration or RNG bug.
    #[must_use]
    pub fn generate(&mut self) -> BwdTrainingPair {
        // Retry loop: guarantee the optimized expression has variables.
        // Without variables, it constant-folds to the timing floor — useless.
        let optimized = {
            let mut attempts = 0;
            loop {
                let expr = self.generate_optimized(0);
                if expr.has_var() {
                    break expr;
                }
                attempts += 1;
                assert!(
                    attempts < Self::MAX_GENERATE_RETRIES,
                    "BwdGenerator failed to produce an expression with variables after {} attempts. \
                     Config: max_depth={}, leaf_prob={}, num_vars={}",
                    attempts, self.config.max_depth, self.config.leaf_prob, self.config.num_vars,
                );
            }
        };

        // Step 2: Apply junkifying rewrites to create unoptimized version
        let (unoptimized, rewrites_applied) = self.junkify_expression(&optimized);

        // Sanity: junkification must not eliminate all variables.
        assert!(
            unoptimized.has_var(),
            "BUG: junkification eliminated all variables from expression. \
             optimized nodes={}, unoptimized nodes={}, rewrites={}",
            optimized.node_count(), unoptimized.node_count(), rewrites_applied,
        );

        BwdTrainingPair {
            optimized,
            unoptimized,
            rewrites_applied,
        }
    }

    /// Minimum depth before leaf generation is allowed.
    /// Ensures expressions have at least some computational structure.
    const MIN_DEPTH: usize = 2;

    /// Generate an optimized expression tree (biased toward fused ops).
    ///
    /// At depths below `MIN_DEPTH`, always generates an operation (never a leaf).
    /// This prevents trivially simple expressions like bare variables or
    /// single-op-on-constant.
    fn generate_optimized(&mut self, depth: usize) -> Expr {
        // Force leaf at max depth
        if depth >= self.config.max_depth {
            return self.generate_leaf();
        }

        // Below MIN_DEPTH: always generate an operation, never a leaf.
        // Above MIN_DEPTH: probabilistic leaf.
        if depth >= Self::MIN_DEPTH && self.rand_f32() < self.config.leaf_prob {
            return self.generate_leaf();
        }

        // Decide: fused op or regular op
        if self.rand_f32() < self.config.fused_op_prob {
            // MulAdd: a * b + c
            Expr::Ternary(
                OpKind::MulAdd,
                Box::new(self.generate_optimized(depth + 1)),
                Box::new(self.generate_optimized(depth + 1)),
                Box::new(self.generate_optimized(depth + 1)),
            )
        } else {
            // Regular operation
            self.generate_regular_op(depth)
        }
    }

    /// Generate a leaf node (variable or constant).
    fn generate_leaf(&mut self) -> Expr {
        if self.rand_f32() < 0.7 {
            Expr::Var(self.rand_usize(self.config.num_vars.min(4)) as u8)
        } else {
            // Constants: small range to avoid numerical issues
            let val = self.rand_f32() * 4.0 - 2.0;
            Expr::Const(val)
        }
    }

    /// Generate a regular (non-fused) operation.
    ///
    /// Covers all op families that have rewrite rules:
    ///   - Arithmetic: Add, Sub, Mul, Div, Neg, Recip
    ///   - Comparison: Min, Max
    ///   - Radicals: Sqrt, Rsqrt, Abs
    ///   - Trig: Sin, Cos, Tan
    ///   - Exp/Log: Exp, Exp2, Ln, Log2
    ///   - Power: Pow, Hypot, Atan2
    fn generate_regular_op(&mut self, depth: usize) -> Expr {
        let choice = self.rand_usize(24);
        let mut child = || Box::new(self.generate_optimized(depth + 1));
        match choice {
            // Binary arithmetic (high weight — most rewrite rules target these)
            // Extra slots for Add/Mul so they appear more often, like real code.
            0 | 1 => Expr::Binary(OpKind::Add, child(), child()),
            2 => Expr::Binary(OpKind::Sub, child(), child()),
            3 | 4 => Expr::Binary(OpKind::Mul, child(), child()),
            5 => {
                // Div: guard denominator to be positive-nonzero
                let num = child();
                let denom = Self::guard_positive_nonzero(*child());
                Expr::Binary(OpKind::Div, num, Box::new(denom))
            }
            // Binary comparison / power
            6 => Expr::Binary(OpKind::Min, child(), child()),
            7 => Expr::Binary(OpKind::Max, child(), child()),
            8 => {
                // Pow: guard base to be positive-nonzero (negative base + fractional exp = NaN)
                let base = Self::guard_positive_nonzero(*child());
                Expr::Binary(OpKind::Pow, Box::new(base), child())
            }
            9 => Expr::Binary(OpKind::Hypot, child(), child()),
            10 => Expr::Binary(OpKind::Atan2, child(), child()),
            // Unary arithmetic
            11 => Expr::Unary(OpKind::Neg, child()),
            12 => Expr::Unary(OpKind::Recip, Box::new(Self::guard_positive_nonzero(*child()))),
            13 => Expr::Unary(OpKind::Abs, child()),
            14 => Expr::Unary(OpKind::Sqrt, Box::new(Self::guard_nonnegative(*child()))),
            15 => Expr::Unary(OpKind::Rsqrt, Box::new(Self::guard_positive_nonzero(*child()))),
            // Trig (parity + angle addition rules)
            16 => Expr::Unary(OpKind::Sin, child()),
            17 => Expr::Unary(OpKind::Cos, child()),
            18 => Expr::Unary(OpKind::Tan, child()),
            // Exp/Log (function inverse + homomorphism rules)
            19 => Expr::Unary(OpKind::Exp, child()),
            20 => Expr::Unary(OpKind::Exp2, child()),
            21 => Expr::Unary(OpKind::Ln, Box::new(Self::guard_positive_nonzero(*child()))),
            22 => Expr::Unary(OpKind::Log2, Box::new(Self::guard_positive_nonzero(*child()))),
            23 => Expr::Unary(OpKind::Log10, Box::new(Self::guard_positive_nonzero(*child()))),
            _ => unreachable!(),
        }
    }

    /// Wrap an expression in `abs(x) + 0.001` to guarantee positive nonzero input.
    /// Used for ops that are undefined at zero or negative values (recip, rsqrt, ln, log2, log10, div denominator).
    fn guard_positive_nonzero(expr: Expr) -> Expr {
        Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Unary(OpKind::Abs, Box::new(expr))),
            Box::new(Expr::Const(0.001)),
        )
    }

    /// Wrap an expression in `abs(x)` to guarantee non-negative input.
    /// Used for sqrt which is defined at zero but not negative values.
    fn guard_nonnegative(expr: Expr) -> Expr {
        Expr::Unary(OpKind::Abs, Box::new(expr))
    }

    /// Apply junkifying rewrites to make an expression MORE complex.
    ///
    /// Tries all rule templates in both directions (LHS->RHS and RHS->LHS),
    /// picks candidates that increase `node_count()`. Uses a shared node budget
    /// to prevent exponential blowup from size-doubling rules.
    fn junkify_expression(&mut self, expr: &Expr) -> (Expr, usize) {
        let mut result = expr.clone();
        let mut total_applied = 0;
        let budget = self.config.max_junkified_nodes;

        for _ in 0..self.config.max_junkify_passes {
            let current_size = result.node_count();
            if current_size >= budget {
                break;
            }
            let mut remaining = budget - current_size;
            let (new_result, applied) = self.junkify_pass(&result, &mut remaining);
            if applied == 0 {
                break;
            }
            result = new_result;
            total_applied += applied;
        }

        (result, total_applied)
    }

    /// Maximum recursion depth for junkification to prevent stack overflow.
    const MAX_JUNKIFY_DEPTH: usize = 32;

    /// Single pass of junkifying rewrites. `budget` tracks how many more
    /// nodes we're willing to add; decremented by each successful rewrite.
    fn junkify_pass(&mut self, expr: &Expr, budget: &mut usize) -> (Expr, usize) {
        let mut applied = 0;
        let result = self.junkify_recursive(expr, &mut applied, 0, budget);
        (result, applied)
    }

    /// Recursively try junkifying at each node, then recurse children.
    /// `budget` is shared across the entire pass — once exhausted, no more rewrites.
    fn junkify_recursive(&mut self, expr: &Expr, applied: &mut usize, depth: usize, budget: &mut usize) -> Expr {
        // Guard against stack overflow
        if depth >= Self::MAX_JUNKIFY_DEPTH {
            return expr.clone();
        }
        // Budget exhausted — stop expanding
        if *budget == 0 {
            return expr.clone();
        }
        // Probabilistically try to junkify at this node
        let expr = if self.rand_f32() < self.config.junkify_prob {
            self.try_junkify_node(expr, applied, budget)
        } else {
            expr.clone()
        };

        // Budget may have been exhausted by try_junkify_node
        if *budget == 0 {
            return expr;
        }

        // Then recurse into children
        match &expr {
            Expr::Var(_) | Expr::Const(_) | Expr::Param(_) => expr,
            Expr::Unary(op, a) => {
                let new_a = self.junkify_recursive(a, applied, depth + 1, budget);
                Expr::Unary(*op, Box::new(new_a))
            }
            Expr::Binary(op, a, b) => {
                let new_a = self.junkify_recursive(a, applied, depth + 1, budget);
                let new_b = self.junkify_recursive(b, applied, depth + 1, budget);
                Expr::Binary(*op, Box::new(new_a), Box::new(new_b))
            }
            Expr::Ternary(op, a, b, c) => {
                let new_a = self.junkify_recursive(a, applied, depth + 1, budget);
                let new_b = self.junkify_recursive(b, applied, depth + 1, budget);
                let new_c = self.junkify_recursive(c, applied, depth + 1, budget);
                Expr::Ternary(*op, Box::new(new_a), Box::new(new_b), Box::new(new_c))
            }
            Expr::Nary(op, children) => {
                let new_children: Vec<_> = children.iter()
                    .map(|c| self.junkify_recursive(c, applied, depth + 1, budget))
                    .collect();
                Expr::Nary(*op, new_children)
            }
        }
    }

    /// Try all rules in both directions at this node.
    /// Pick a random candidate that increases `node_count` within budget.
    ///
    /// Skips matching against bare `Var(_)` templates — these match anything
    /// and produce trivially-collapsible wrapping (e.g. `V0 → exp(ln(V0))`).
    fn try_junkify_node(&mut self, expr: &Expr, applied: &mut usize, budget: &mut usize) -> Expr {
        let original_cost = expr.node_count();
        let mut candidates: Vec<Expr> = Vec::new();

        for rule_idx in 0..self.templates.len() {
            if let (Some(lhs), Some(rhs)) = (self.templates.get_lhs(rule_idx), self.templates.get_rhs(rule_idx)) {
                // Skip bare Var templates — they match everything and produce
                // trivially-collapsible junk like exp(ln(x)), neg(neg(x)), x+0
                let lhs_is_bare_var = matches!(lhs, Expr::Var(_));
                let rhs_is_bare_var = matches!(rhs, Expr::Var(_));

                // Try LHS -> RHS direction (match LHS, produce RHS)
                if !lhs_is_bare_var {
                    if let Some(bindings) = pattern_match(expr, lhs) {
                        if let Some(result) = substitute_template(rhs, &bindings) {
                            let new_cost = result.node_count();
                            let growth = new_cost.saturating_sub(original_cost);
                            if new_cost > original_cost && growth <= *budget {
                                candidates.push(result);
                            }
                        }
                    }
                }
                // Try RHS -> LHS direction (match RHS, produce LHS)
                if !rhs_is_bare_var {
                    if let Some(bindings) = pattern_match(expr, rhs) {
                        if let Some(result) = substitute_template(lhs, &bindings) {
                            let new_cost = result.node_count();
                            let growth = new_cost.saturating_sub(original_cost);
                            if new_cost > original_cost && growth <= *budget {
                                candidates.push(result);
                            }
                        }
                    }
                }
            }
        }

        if candidates.is_empty() {
            return expr.clone();
        }

        // Pick a random candidate and deduct its growth from the budget
        let idx = self.rand_usize(candidates.len());
        let chosen = candidates.swap_remove(idx);
        let growth = chosen.node_count().saturating_sub(original_cost);
        *budget = budget.saturating_sub(growth);
        *applied += 1;
        chosen
    }
}

/// Count fused operations in an expression.
#[must_use]
pub fn count_fused_ops(expr: &Expr) -> usize {
    match expr {
        Expr::Var(_) | Expr::Const(_) => 0,
        Expr::Param(i) => panic!("Expr::Param({}) reached NNUE cost model — call substitute_params before use", i),
        Expr::Unary(_, a) => count_fused_ops(a),
        Expr::Binary(_, a, b) => {
            count_fused_ops(a) + count_fused_ops(b)
        }
        Expr::Ternary(op, a, b, c) => {
            let this = if *op == OpKind::MulAdd { 1 } else { 0 };
            this + count_fused_ops(a) + count_fused_ops(b) + count_fused_ops(c)
        }
        Expr::Nary(_, children) => {
            children.iter().map(count_fused_ops).sum()
        }
    }
}

fn find_rewrites_recursive(
    expr: &Expr,
    path: &mut Vec<usize>,
    rewrites: &mut Vec<(Vec<usize>, RewriteRule, Expr)>,
) {
    // Try all rules at this node
    for rule in RewriteRule::ALL {
        if let Some(rewritten) = rule.try_apply(expr) {
            rewrites.push((path.clone(), rule.clone(), rewritten));
        }
    }

    // Recurse into children
    match expr {
        Expr::Var(_) | Expr::Const(_) => {}
        Expr::Param(i) => panic!("Expr::Param({}) reached NNUE cost model — call substitute_params before use", i),
        Expr::Unary(_, a) => {
            path.push(0);
            find_rewrites_recursive(a, path, rewrites);
            path.pop();
        }
        Expr::Binary(_, a, b) => {
            path.push(0);
            find_rewrites_recursive(a, path, rewrites);
            path.pop();
            path.push(1);
            find_rewrites_recursive(b, path, rewrites);
            path.pop();
        }
        Expr::Ternary(_, a, b, c) => {
            path.push(0);
            find_rewrites_recursive(a, path, rewrites);
            path.pop();
            path.push(1);
            find_rewrites_recursive(b, path, rewrites);
            path.pop();
            path.push(2);
            find_rewrites_recursive(c, path, rewrites);
            path.pop();
        }
        Expr::Nary(_, children) => {
            for (i, child) in children.iter().enumerate() {
                path.push(i);
                find_rewrites_recursive(child, path, rewrites);
                path.pop();
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use libm::fabsf;

    #[test]
    fn test_op_type_roundtrip() {
        for i in 0..OpKind::COUNT {
            let op = OpKind::from_index(i).unwrap();
            assert_eq!(op.index(), i);
        }
    }

    #[test]
    fn test_feature_roundtrip() {
        let feature = HalfEPFeature {
            perspective_op: 5,
            descendant_op: 10,
            depth: 3,
            path: 0b10110,
        };
        let idx = feature.to_index();
        let recovered = HalfEPFeature::from_index(idx);
        assert_eq!(feature, recovered);
    }

    #[test]
    fn test_expr_generator() {
        let mut generator = ExprGenerator::new(42, ExprGenConfig::default());
        for _ in 0..10 {
            let expr = generator.generate();
            assert!(expr.depth() <= 7); // max_depth + 1 for leaf
            assert!(expr.node_count() > 0);
        }
    }

    #[test]
    fn test_feature_extraction() {
        let expr = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(1.0)),
        );
        let features = extract_features(&expr);
        assert!(!features.is_empty());
    }

    #[test]
    fn test_rewrite_add_zero() {
        let expr = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(0.0)),
        );
        let rewritten = RewriteRule::AddZero.try_apply(&expr);
        assert!(rewritten.is_some());
        assert!(matches!(rewritten.unwrap(), Expr::Var(0)));
    }

    #[test]
    fn test_rewrite_fuse_muladd() {
        // a * b + c
        let expr = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Binary(
                OpKind::Mul,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
            )),
            Box::new(Expr::Var(2)),
        );
        let rewritten = RewriteRule::FuseToMulAdd.try_apply(&expr);
        assert!(rewritten.is_some());
        assert!(matches!(
            rewritten.unwrap(),
            Expr::Ternary(OpKind::MulAdd, _, _, _)
        ));
    }

    #[test]
    fn test_find_all_rewrites() {
        // (x + 0) + y - should find AddZero at path [0]
        let expr = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Binary(
                OpKind::Add,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(0.0)),
            )),
            Box::new(Expr::Var(1)),
        );
        let rewrites = find_all_rewrites(&expr);
        assert!(!rewrites.is_empty());

        // Should find AddZero rewrite
        let add_zero_rewrites: Vec<_> = rewrites
            .iter()
            .filter(|(_, rule, _)| matches!(rule, RewriteRule::AddZero))
            .collect();
        assert!(!add_zero_rewrites.is_empty());
    }

    #[test]
    fn test_accumulator_add_remove() {
        let nnue = Nnue::with_defaults();
        let mut acc = Accumulator::new(&nnue);

        let feature_idx = 1000;

        // Add feature
        acc.add_feature(&nnue, feature_idx);

        // Remove same feature should return to original
        acc.remove_feature(&nnue, feature_idx);

        // Should be back to bias values
        for (i, &val) in acc.values.iter().enumerate() {
            assert_eq!(
                val, nnue.b1[i],
                "Accumulator should return to bias after add/remove"
            );
        }
    }

    // ========================================================================
    // Pattern Match + Substitute Tests
    // ========================================================================

    #[test]
    fn test_pattern_match_var() {
        // Var(0) matches anything
        let expr = Expr::Binary(OpKind::Add, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let template = Expr::Var(0);
        let bindings = pattern_match(&expr, &template).unwrap();
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[&0], expr);
    }

    #[test]
    fn test_pattern_match_structural() {
        // Match Add(V0, V1) against Add(X, Y)
        let expr = Expr::Binary(OpKind::Add, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let template = Expr::Binary(OpKind::Add, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let bindings = pattern_match(&expr, &template).unwrap();
        assert_eq!(bindings.len(), 2);
        assert_eq!(bindings[&0], Expr::Var(0));
        assert_eq!(bindings[&1], Expr::Var(1));
    }

    #[test]
    fn test_pattern_match_consistency() {
        // V0 appears twice -- must bind to same sub-expr
        let expr = Expr::Binary(OpKind::Add, Box::new(Expr::Var(0)), Box::new(Expr::Var(0)));
        let template = Expr::Binary(OpKind::Add, Box::new(Expr::Var(0)), Box::new(Expr::Var(0)));
        assert!(pattern_match(&expr, &template).is_some());

        // Different sub-exprs for same var -- must fail
        let expr2 = Expr::Binary(OpKind::Add, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        assert!(pattern_match(&expr2, &template).is_none());
    }

    #[test]
    fn test_pattern_match_const() {
        let expr = Expr::Const(1.0);
        let template = Expr::Const(1.0);
        assert!(pattern_match(&expr, &template).is_some());

        let template2 = Expr::Const(2.0);
        assert!(pattern_match(&expr, &template2).is_none());
    }

    #[test]
    fn test_substitute_template() {
        let mut bindings = BTreeMap::new();
        bindings.insert(0, Expr::Var(0)); // X
        bindings.insert(1, Expr::Var(1)); // Y

        // Template: Add(V0, V1) -> should produce Add(X, Y)
        let template = Expr::Binary(OpKind::Add, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let result = substitute_template(&template, &bindings)
            .expect("substitute_template returned None with complete bindings");
        assert_eq!(result, Expr::Binary(OpKind::Add, Box::new(Expr::Var(0)), Box::new(Expr::Var(1))));
    }

    #[test]
    fn test_substitute_template_unbound_var_returns_none() {
        let mut bindings = BTreeMap::new();
        bindings.insert(0, Expr::Var(0)); // Only V0 bound

        // Template uses V0 and V1 -- V1 is unbound, should return None
        let template = Expr::Binary(OpKind::Add, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let result = substitute_template(&template, &bindings);
        assert!(result.is_none(), "Expected None for unbound Var(1), got {:?}", result);
    }

    #[test]
    fn test_pattern_match_then_substitute_roundtrip() {
        // Match Add(X, Y) against Add(V0, V1), then substitute into Mul(V0, V1)
        let expr = Expr::Binary(OpKind::Add, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let lhs = Expr::Binary(OpKind::Add, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let rhs = Expr::Binary(OpKind::Mul, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));

        let bindings = pattern_match(&expr, &lhs).unwrap();
        let result = substitute_template(&rhs, &bindings)
            .expect("substitute_template returned None with complete bindings");
        assert_eq!(result, Expr::Binary(OpKind::Mul, Box::new(Expr::Var(0)), Box::new(Expr::Var(1))));
    }

    // ========================================================================
    // Backward Generation Tests
    // ========================================================================

    #[test]
    fn test_bwd_generator_produces_valid_pairs() {
        use crate::egraph::collect_rule_templates;
        let templates = collect_rule_templates();
        let config = BwdGenConfig::default();
        let mut generator = BwdGenerator::new(42, config, templates);

        for _ in 0..10 {
            let pair = generator.generate();

            // Both expressions should be valid
            assert!(pair.optimized.node_count() > 0);
            assert!(pair.unoptimized.node_count() > 0);

            // Unoptimized should generally be larger or equal
            // (junkifying increases or maintains size)
            assert!(pair.unoptimized.node_count() >= pair.optimized.node_count());
        }
    }

    #[test]
    fn test_bwd_generator_has_fused_ops() {
        use crate::egraph::collect_rule_templates;
        let templates = collect_rule_templates();
        let config = BwdGenConfig {
            fused_op_prob: 0.8, // High probability of fused ops
            max_depth: 4,
            ..Default::default()
        };
        let mut generator = BwdGenerator::new(12345, config, templates);

        let mut total_fused = 0;
        for _ in 0..20 {
            let pair = generator.generate();
            total_fused += count_fused_ops(&pair.optimized);
        }

        // With 80% fused op probability, we should see some fused ops
        assert!(
            total_fused > 0,
            "Expected some fused operations in generated expressions"
        );
    }

    #[test]
    fn test_bwd_generator_with_templates() {
        use crate::egraph::collect_rule_templates;
        let templates = collect_rule_templates();
        let config = BwdGenConfig::default();
        let mut generator = BwdGenerator::new(42, config, templates);

        // Generate 20 pairs and check they're non-trivial
        let mut total_rewrites = 0;
        for _ in 0..20 {
            let pair = generator.generate();
            assert!(pair.unoptimized.node_count() >= pair.optimized.node_count(),
                "unoptimized ({}) should have >= nodes than optimized ({})",
                pair.unoptimized.node_count(), pair.optimized.node_count());
            total_rewrites += pair.rewrites_applied;
        }
        // At least some rewrites should have been applied across 20 expressions
        assert!(total_rewrites > 0, "Expected at least one junkify rewrite across 20 expressions, got 0");
    }

    #[test]
    fn test_count_fused_ops() {
        // Expression with MulAdd
        let expr = Expr::Ternary(
            OpKind::MulAdd,
            Box::new(Expr::Binary(
                OpKind::Mul,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
            )),
            Box::new(Expr::Var(2)),
            Box::new(Expr::Var(3)),
        );

        assert_eq!(count_fused_ops(&expr), 1);
    }

    // ========================================================================
    // Dense Features and ILP Tests
    // ========================================================================

    #[test]
    fn test_dense_features_simple_add() {
        // x + y
        let expr = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        );
        let features = extract_dense_features(&expr);

        assert_eq!(features.values[DenseFeatures::ADD], 1);
        assert_eq!(features.values[DenseFeatures::VAR_COUNT], 2);
        assert_eq!(features.values[DenseFeatures::NODE_COUNT], 3);
        assert_eq!(features.values[DenseFeatures::DEPTH], 2);
    }

    #[test]
    fn test_dense_features_critical_path_wide_vs_deep() {
        // Wide expression: (a + b) + (c + d)
        // Critical path: 4 + 4 = 8 (two adds in sequence, but children parallel)
        let wide = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Binary(
                OpKind::Add,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
            )),
            Box::new(Expr::Binary(
                OpKind::Add,
                Box::new(Expr::Var(2)),
                Box::new(Expr::Var(3)),
            )),
        );

        // Deep expression: ((a + b) + c) + d
        // Critical path: 4 + 4 + 4 = 12 (three sequential adds)
        let deep = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Binary(
                OpKind::Add,
                Box::new(Expr::Binary(
                    OpKind::Add,
                    Box::new(Expr::Var(0)),
                    Box::new(Expr::Var(1)),
                )),
                Box::new(Expr::Var(2)),
            )),
            Box::new(Expr::Var(3)),
        );

        let wide_features = extract_dense_features(&wide);
        let deep_features = extract_dense_features(&deep);

        // Same total operation count
        assert_eq!(wide_features.values[DenseFeatures::ADD], 3);
        assert_eq!(deep_features.values[DenseFeatures::ADD], 3);

        // But different critical paths
        assert_eq!(wide_features.values[DenseFeatures::CRITICAL_PATH], 8);
        assert_eq!(deep_features.values[DenseFeatures::CRITICAL_PATH], 12);

        // Wide is better for ILP
        assert!(wide_features.values[DenseFeatures::CRITICAL_PATH] <
                deep_features.values[DenseFeatures::CRITICAL_PATH]);
    }

    #[test]
    fn test_dense_features_max_width() {
        // Wide expression: (a + b) + (c + d)
        // Max width = 4 (all vars at depth 2)
        let wide = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Binary(
                OpKind::Add,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
            )),
            Box::new(Expr::Binary(
                OpKind::Add,
                Box::new(Expr::Var(2)),
                Box::new(Expr::Var(3)),
            )),
        );

        let features = extract_dense_features(&wide);
        assert_eq!(features.values[DenseFeatures::MAX_WIDTH], 4);
    }

    #[test]
    fn test_dense_features_detect_identity() {
        // x * 1 - has identity
        let expr = Expr::Binary(
            OpKind::Mul,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(1.0)),
        );
        let features = extract_dense_features(&expr);
        assert_eq!(features.values[DenseFeatures::HAS_IDENTITY], 1);
    }

    #[test]
    fn test_dense_features_detect_fusable() {
        // (a * b) + c - fusable pattern
        let expr = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Binary(
                OpKind::Mul,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
            )),
            Box::new(Expr::Var(2)),
        );
        let features = extract_dense_features(&expr);
        assert_eq!(features.values[DenseFeatures::HAS_FUSABLE], 1);
    }



    #[test]
    fn test_hybrid_forward_with_features() {
        let nnue = Nnue::with_defaults();
        let mut acc = Accumulator::new(&nnue);

        // Add some sparse features
        acc.add_feature(&nnue, 100);
        acc.add_feature(&nnue, 200);

        // Create dense features
        let mut dense = DenseFeatures::default();
        dense.values[DenseFeatures::ADD] = 2;
        dense.values[DenseFeatures::CRITICAL_PATH] = 8;

        // Should produce different output than with zeros
        let output_with_features = acc.forward_hybrid(&nnue, &dense);
        let output_without = acc.forward(&nnue);

        // With a zero-initialized network, both should be 0
        // But the code paths are different, so this validates no panics
        assert_eq!(output_with_features, 0);
        assert_eq!(output_without, 0);
    }

    #[test]
    fn test_nnue_config_combined_size() {
        let config = NnueConfig::default();
        assert_eq!(config.combined_size(), 256 + 32);
    }
}
