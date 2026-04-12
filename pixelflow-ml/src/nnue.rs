//! # NNUE for Instruction Selection
//!
//! An Efficiently Updatable Neural Network for compiler instruction selection,
//! inspired by Stockfish's NNUE approach to chess position evaluation.
//!
//! ## The Core Insight
//!
//! Just as Stockfish uses NNUE to evaluate chess positions, we can use NNUE
//! to evaluate expression costs for instruction selection. The key parallel:
//!
//! | Chess (Stockfish)        | Compiler (PixelFlow)           |
//! |--------------------------|--------------------------------|
//! | Position                 | Expression AST / e-class       |
//! | Legal move               | Valid rewrite rule             |
//! | Evaluation (centipawns)  | Cost (cycles)                  |
//! | HalfKP features          | HalfEP features                |
//! | King position            | Root operation type            |
//! | Piece placement          | Subexpression structure        |
//!
//! ## HalfEP Feature Set
//!
//! Inspired by HalfKP (Half-King-Piece), we define HalfEP (Half-Expression-Position):
//!
//! ```text
//! Feature = (perspective_op, descendant_op, relative_depth, child_index)
//! ```
//!
//! Where:
//! - `perspective_op`: The operation we're evaluating "from" (like king position)
//! - `descendant_op`: An operation in the subtree
//! - `relative_depth`: How deep the descendant is from perspective
//! - `child_index`: Left (0) or right (1) branch
//!
//! This creates a sparse feature vector that captures the structure of expressions.
//!
//! ## Training Data Generation
//!
//! Like Stockfish's self-play, we generate training data by:
//!
//! 1. **Generate random expressions** (like random positions)
//! 2. **Enumerate valid rewrites** (like legal moves)
//! 3. **Benchmark each rewrite** (like deep search evaluation)
//! 4. **Record (features, best_rewrite, cost_delta)** tuples
//!
//! ## Incremental Updates
//!
//! The key to NNUE efficiency: most rewrites are local. When we apply a rewrite
//! rule to a subtree, only features involving that subtree change. We can:
//!
//! 1. Remove features for the old subtree
//! 2. Add features for the new subtree
//! 3. Incrementally update the accumulator
//!
//! This makes evaluation O(rewrite_size) instead of O(expression_size).

#![allow(dead_code)] // Prototype code

use alloc::boxed::Box;
use alloc::vec::Vec;
use libm::{fabsf, sqrtf};

// ============================================================================
// Operation Types - use pixelflow-ir's OpKind as the source of truth
// ============================================================================

/// Re-export OpKind from pixelflow-ir as our canonical operation type.
/// This ensures consistent operation indices across all pixelflow crates.
pub use pixelflow_ir::OpKind;

/// Backwards compatibility alias - prefer OpKind in new code.
pub type OpType = OpKind;

// ============================================================================
// Expression AST (for training data generation)
// ============================================================================

/// A concrete expression tree for training data generation.
///
/// Unlike the e-graph's `ENode` (which references e-classes), this is a
/// standalone AST that can be manipulated and benchmarked directly.
#[derive(Clone, Debug)]
pub enum Expr {
    /// Variable by index (0=X, 1=Y, 2=Z, 3=W)
    Var(u8),
    /// Floating-point constant
    Const(f32),
    /// Binary operation
    Binary(OpType, Box<Expr>, Box<Expr>),
    /// Unary operation
    Unary(OpType, Box<Expr>),
    /// Ternary operation (MulAdd)
    Ternary(OpType, Box<Expr>, Box<Expr>, Box<Expr>),
}

impl Expr {
    /// Get the operation type of this expression's root.
    #[must_use]
    pub fn op_type(&self) -> OpType {
        match self {
            Expr::Var(_) => OpType::Var,
            Expr::Const(_) => OpType::Const,
            Expr::Binary(op, _, _) => *op,
            Expr::Unary(op, _) => *op,
            Expr::Ternary(op, _, _, _) => *op,
        }
    }

    /// Compute the depth of this expression tree.
    #[must_use]
    pub fn depth(&self) -> usize {
        match self {
            Expr::Var(_) | Expr::Const(_) => 1,
            Expr::Unary(_, a) => 1 + a.depth(),
            Expr::Binary(_, a, b) => 1 + a.depth().max(b.depth()),
            Expr::Ternary(_, a, b, c) => 1 + a.depth().max(b.depth()).max(c.depth()),
        }
    }

    /// Count total nodes in the expression.
    #[must_use]
    pub fn node_count(&self) -> usize {
        match self {
            Expr::Var(_) | Expr::Const(_) => 1,
            Expr::Unary(_, a) => 1 + a.node_count(),
            Expr::Binary(_, a, b) => 1 + a.node_count() + b.node_count(),
            Expr::Ternary(_, a, b, c) => 1 + a.node_count() + b.node_count() + c.node_count(),
        }
    }

    /// Evaluate the expression with given variable values.
    #[must_use]
    pub fn eval(&self, vars: &[f32; 4]) -> f32 {
        match self {
            Expr::Var(i) => vars[*i as usize],
            Expr::Const(c) => *c,
            Expr::Unary(op, a) => {
                let a = a.eval(vars);
                match op {
                    OpType::Neg => -a,
                    OpType::Sqrt => sqrtf(a),
                    OpType::Rsqrt => 1.0 / sqrtf(a),
                    OpType::Abs => fabsf(a),
                    _ => unreachable!(),
                }
            }
            Expr::Binary(op, a, b) => {
                let a = a.eval(vars);
                let b = b.eval(vars);
                match op {
                    OpType::Add => a + b,
                    OpType::Sub => a - b,
                    OpType::Mul => a * b,
                    OpType::Div => a / b,
                    OpType::Min => {
                        if a < b {
                            a
                        } else {
                            b
                        }
                    }
                    OpType::Max => {
                        if a > b {
                            a
                        } else {
                            b
                        }
                    }
                    OpType::MulRsqrt => a / sqrtf(b),
                    _ => unreachable!(),
                }
            }
            Expr::Ternary(op, a, b, c) => {
                let a = a.eval(vars);
                let b = b.eval(vars);
                let c = c.eval(vars);
                match op {
                    OpType::MulAdd => a * b + c,
                    _ => unreachable!(),
                }
            }
        }
    }
}

// ============================================================================
// HalfEP Features
// ============================================================================

/// Maximum depth we encode in features.
pub const MAX_DEPTH: usize = 8;

/// Maximum child index (for position encoding).
pub const MAX_CHILD_INDEX: usize = 8;

/// A HalfEP feature: (perspective_op, descendant_op, depth, child_path).
///
/// This is analogous to HalfKP in chess:
/// - perspective_op is like the king position (which subtree we're "viewing from")
/// - descendant_op + depth + path is like the piece placement
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HalfEPFeature {
    /// The operation type we're evaluating from (0-13)
    pub perspective_op: u8,
    /// The descendant operation type (0-13)
    pub descendant_op: u8,
    /// Relative depth from perspective (0-7)
    pub depth: u8,
    /// Child path encoding (left=0, right=1 at each level, packed into bits)
    pub path: u8,
}

impl HalfEPFeature {
    /// Total number of possible features.
    ///
    /// 14 perspective ops × 14 descendant ops × 8 depths × 256 paths = 401,408
    /// This is much smaller than HalfKP's ~10M, which is good for our use case.
    pub const COUNT: usize = OpType::COUNT * OpType::COUNT * MAX_DEPTH * 256;

    /// Convert to a unique index for the feature vector.
    #[must_use]
    pub fn to_index(self) -> usize {
        let p = self.perspective_op as usize;
        let d = self.descendant_op as usize;
        let depth = self.depth as usize;
        let path = self.path as usize;

        ((p * OpType::COUNT + d) * MAX_DEPTH + depth) * 256 + path
    }

    /// Create from a unique index.
    #[must_use]
    pub fn from_index(idx: usize) -> Self {
        let path = (idx % 256) as u8;
        let idx = idx / 256;
        let depth = (idx % MAX_DEPTH) as u8;
        let idx = idx / MAX_DEPTH;
        let descendant_op = (idx % OpType::COUNT) as u8;
        let perspective_op = (idx / OpType::COUNT) as u8;

        Self {
            perspective_op,
            descendant_op,
            depth,
            path,
        }
    }
}

/// Extract HalfEP features from an expression.
///
/// For each node in the tree, we create features describing its descendants
/// from that node's perspective (like HalfKP creates features from each
/// king's perspective).
#[must_use]
pub fn extract_features(expr: &Expr) -> Vec<HalfEPFeature> {
    let mut features = Vec::new();
    extract_features_recursive(expr, &mut features, 0, 0);
    features
}

#[allow(clippy::only_used_in_recursion)]
fn extract_features_recursive(expr: &Expr, features: &mut Vec<HalfEPFeature>, path: u8, depth: u8) {
    let root_op = expr.op_type();

    // Add features for all descendants from this node's perspective
    add_descendant_features(expr, features, root_op.index() as u8, 0, 0);

    // Recurse into children with updated path
    match expr {
        Expr::Var(_) | Expr::Const(_) => {}
        Expr::Unary(_, a) => {
            extract_features_recursive(a, features, path, depth.saturating_add(1));
        }
        Expr::Binary(_, a, b) => {
            extract_features_recursive(a, features, path << 1, depth.saturating_add(1));
            extract_features_recursive(b, features, (path << 1) | 1, depth.saturating_add(1));
        }
        Expr::Ternary(_, a, b, c) => {
            // For ternary, we reuse the path for simplicity as they don't branch like binary
            // Alternatively, we could define a different path encoding for ternary
            extract_features_recursive(a, features, path, depth.saturating_add(1));
            extract_features_recursive(b, features, path, depth.saturating_add(1));
            extract_features_recursive(c, features, path, depth.saturating_add(1));
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

    // Add feature for this node
    features.push(HalfEPFeature {
        perspective_op,
        descendant_op: expr.op_type().index() as u8,
        depth,
        path,
    });

    // Recurse into children
    match expr {
        Expr::Var(_) | Expr::Const(_) => {}
        Expr::Unary(_, a) => {
            add_descendant_features(a, features, perspective_op, depth + 1, path << 1);
        }
        Expr::Binary(_, a, b) => {
            add_descendant_features(a, features, perspective_op, depth + 1, path << 1);
            add_descendant_features(b, features, perspective_op, depth + 1, (path << 1) | 1);
        }
        Expr::Ternary(_, a, b, c) => {
            // For ternary, use bits 0, 1, 2 for the three children
            // Need to shift by 2 bits to make room for 2 bits of index (0-3)
            add_descendant_features(a, features, perspective_op, depth + 1, path << 2);
            add_descendant_features(b, features, perspective_op, depth + 1, (path << 2) | 1);
            add_descendant_features(c, features, perspective_op, depth + 1, (path << 2) | 2);
        }
    }
}

// ============================================================================
// NNUE Network Architecture
// ============================================================================

/// NNUE network configuration.
///
/// Architecture: Sparse Input (HalfEP) -> L1 (256) -> L2 (32) -> L3 (32) -> Output (1)
///
/// This mirrors Stockfish's 256x2-32-32 but simplified:
/// - We don't need the x2 for "perspectives" (white/black) in the same way
/// - Instead, we could have separate accumulators for different subtree roots
#[derive(Clone)]
pub struct NnueConfig {
    /// Size of the first hidden layer.
    pub l1_size: usize,
    /// Size of the second hidden layer.
    pub l2_size: usize,
    /// Size of the third hidden layer.
    pub l3_size: usize,
}

impl Default for NnueConfig {
    fn default() -> Self {
        Self {
            l1_size: 256,
            l2_size: 32,
            l3_size: 32,
        }
    }
}

/// The NNUE network for expression cost prediction.
///
/// This is a 4-layer network:
/// 1. Sparse input -> L1 (HalfEP features to hidden)
/// 2. L1 -> L2 (hidden to hidden)
/// 3. L2 -> L3 (hidden to hidden)
/// 4. L3 -> output (hidden to scalar cost)
#[derive(Clone)]
pub struct Nnue {
    /// Configuration.
    pub config: NnueConfig,
    /// First layer weights: [feature_count, l1_size]
    /// Stored as column-major for efficient sparse updates.
    pub w1: Vec<i16>,
    /// First layer biases: [l1_size]
    pub b1: Vec<i32>,
    /// Second layer weights: [l1_size, l2_size]
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
        let feature_count = HalfEPFeature::COUNT;

        Self {
            w1: alloc::vec![0i16; feature_count * config.l1_size],
            b1: alloc::vec![0i32; config.l1_size],
            w2: alloc::vec![0i8; config.l1_size * config.l2_size],
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
}

/// Accumulator for incremental NNUE updates.
///
/// This is the key to NNUE efficiency: we maintain the output of the first
/// layer (the most expensive part) and incrementally update it when
/// features change.
#[derive(Clone)]
pub struct Accumulator {
    /// L1 activations (before clipped ReLU).
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

    /// Compute the full forward pass from the accumulator state.
    ///
    /// Returns the predicted cost in centipawns (will need to be scaled).
    #[must_use]
    pub fn forward(&self, nnue: &Nnue) -> i32 {
        let l1_size = nnue.config.l1_size;
        let l2_size = nnue.config.l2_size;
        let l3_size = nnue.config.l3_size;

        // L1 -> L2 with clipped ReLU
        let mut l2 = nnue.b2.clone();
        for i in 0..l1_size {
            // Clipped ReLU: clamp to [0, 127] then scale
            let a = (self.values[i] >> 6).clamp(0, 127) as i8;
            for j in 0..l2_size {
                l2[j] += (a as i32) * (nnue.w2[i * l2_size + j] as i32);
            }
        }

        // L2 -> L3 with clipped ReLU
        let mut l3 = nnue.b3.clone();
        for i in 0..l2_size {
            let a = (l2[i] >> 6).clamp(0, 127) as i8;
            for j in 0..l3_size {
                l3[j] += (a as i32) * (nnue.w3[i * l3_size + j] as i32);
            }
        }

        // L3 -> output
        let mut output = nnue.b_out;
        for i in 0..l3_size {
            let a = (l3[i] >> 6).clamp(0, 127) as i8;
            output += (a as i32) * (nnue.w_out[i] as i32);
        }

        output
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
pub struct ExprGenerator {
    /// Configuration.
    pub config: ExprGenConfig,
    /// Random state (simple LCG for no_std compatibility).
    state: u64,
}

impl ExprGenerator {
    /// Create a new generator with the given seed.
    #[must_use]
    pub fn new(seed: u64, config: ExprGenConfig) -> Self {
        Self {
            config,
            state: seed,
        }
    }

    /// Generate a random f32 in [0, 1).
    fn rand_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 33) as f32 / (1u64 << 31) as f32
    }

    /// Generate a random usize in [0, max).
    fn rand_usize(&mut self, max: usize) -> usize {
        (self.rand_f32() * max as f32) as usize
    }

    /// Generate a random expression.
    pub fn generate(&mut self) -> Expr {
        self.generate_recursive(0)
    }

    fn generate_recursive(&mut self, depth: usize) -> Expr {
        // Force leaf at max depth or with probability leaf_prob
        if depth >= self.config.max_depth || self.rand_f32() < self.config.leaf_prob {
            if self.rand_f32() < 0.7 {
                // Variable
                Expr::Var(self.rand_usize(self.config.num_vars) as u8)
            } else {
                // Constant (small values to avoid overflow)
                let val = self.rand_f32() * 10.0 - 5.0;
                Expr::Const(val)
            }
        } else {
            // Generate an operation
            let op_choice = if self.config.include_fused {
                self.rand_usize(12) // Include all ops
            } else {
                self.rand_usize(10) // Exclude fused ops
            };

            match op_choice {
                // Binary operations
                0 => Expr::Binary(
                    OpType::Add,
                    Box::new(self.generate_recursive(depth + 1)),
                    Box::new(self.generate_recursive(depth + 1)),
                ),
                1 => Expr::Binary(
                    OpType::Sub,
                    Box::new(self.generate_recursive(depth + 1)),
                    Box::new(self.generate_recursive(depth + 1)),
                ),
                2 => Expr::Binary(
                    OpType::Mul,
                    Box::new(self.generate_recursive(depth + 1)),
                    Box::new(self.generate_recursive(depth + 1)),
                ),
                3 => Expr::Binary(
                    OpType::Div,
                    Box::new(self.generate_recursive(depth + 1)),
                    Box::new(self.generate_recursive(depth + 1)),
                ),
                4 => Expr::Binary(
                    OpType::Min,
                    Box::new(self.generate_recursive(depth + 1)),
                    Box::new(self.generate_recursive(depth + 1)),
                ),
                5 => Expr::Binary(
                    OpType::Max,
                    Box::new(self.generate_recursive(depth + 1)),
                    Box::new(self.generate_recursive(depth + 1)),
                ),
                // Unary operations
                6 => Expr::Unary(OpType::Neg, Box::new(self.generate_recursive(depth + 1))),
                7 => Expr::Unary(OpType::Sqrt, Box::new(self.generate_recursive(depth + 1))),
                8 => Expr::Unary(OpType::Rsqrt, Box::new(self.generate_recursive(depth + 1))),
                9 => Expr::Unary(OpType::Abs, Box::new(self.generate_recursive(depth + 1))),
                // Fused operations
                10 => Expr::Ternary(
                    OpType::MulAdd,
                    Box::new(self.generate_recursive(depth + 1)),
                    Box::new(self.generate_recursive(depth + 1)),
                    Box::new(self.generate_recursive(depth + 1)),
                ),
                11 => Expr::Binary(
                    OpType::MulRsqrt,
                    Box::new(self.generate_recursive(depth + 1)),
                    Box::new(self.generate_recursive(depth + 1)),
                ),
                _ => unreachable!(),
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
    /// x * rsqrt(y) → MulRsqrt(x, y)
    FuseToMulRsqrt,
    /// x / sqrt(y) → x * rsqrt(y)
    DivSqrtToMulRsqrt,
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
        RewriteRule::FuseToMulRsqrt,
        RewriteRule::DivSqrtToMulRsqrt,
        RewriteRule::UnfuseMulAdd,
    ];

    /// Try to apply this rule to an expression, returning the rewritten form.
    ///
    /// Returns None if the rule doesn't match.
    #[must_use]
    pub fn try_apply(&self, expr: &Expr) -> Option<Expr> {
        match self {
            RewriteRule::AddZero => match expr {
                Expr::Binary(OpType::Add, a, b) => {
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
                Expr::Binary(OpType::Mul, a, b) => {
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
                Expr::Binary(OpType::Mul, _, b) if matches!(b.as_ref(), Expr::Const(c) if *c == 0.0) => {
                    Some(Expr::Const(0.0))
                }
                Expr::Binary(OpType::Mul, a, _) if matches!(a.as_ref(), Expr::Const(c) if *c == 0.0) => {
                    Some(Expr::Const(0.0))
                }
                _ => None,
            },
            RewriteRule::SubSelf => match expr {
                Expr::Binary(OpType::Sub, a, b) if exprs_equal(a, b) => Some(Expr::Const(0.0)),
                _ => None,
            },
            RewriteRule::DivSelf => match expr {
                Expr::Binary(OpType::Div, a, b) if exprs_equal(a, b) => Some(Expr::Const(1.0)),
                _ => None,
            },
            RewriteRule::DoubleNeg => match expr {
                Expr::Unary(OpType::Neg, inner) => match inner.as_ref() {
                    Expr::Unary(OpType::Neg, x) => Some(x.as_ref().clone()),
                    _ => None,
                },
                _ => None,
            },
            RewriteRule::AddSelf => match expr {
                Expr::Binary(OpType::Add, a, b) if exprs_equal(a, b) => Some(Expr::Binary(
                    OpType::Mul,
                    Box::new(Expr::Const(2.0)),
                    a.clone(),
                )),
                _ => None,
            },
            RewriteRule::FuseToMulAdd => match expr {
                Expr::Binary(OpType::Add, mul_expr, c) => match mul_expr.as_ref() {
                    Expr::Binary(OpType::Mul, a, b) => Some(Expr::Ternary(
                        OpType::MulAdd,
                        a.clone(),
                        b.clone(),
                        c.clone(),
                    )),
                    _ => None,
                },
                _ => None,
            },
            RewriteRule::FuseToMulRsqrt => match expr {
                Expr::Binary(OpType::Mul, x, rsqrt_expr) => match rsqrt_expr.as_ref() {
                    Expr::Unary(OpType::Rsqrt, y) => {
                        Some(Expr::Binary(OpType::MulRsqrt, x.clone(), y.clone()))
                    }
                    _ => None,
                },
                _ => None,
            },
            RewriteRule::DivSqrtToMulRsqrt => match expr {
                Expr::Binary(OpType::Div, x, sqrt_expr) => match sqrt_expr.as_ref() {
                    Expr::Unary(OpType::Sqrt, y) => Some(Expr::Binary(
                        OpType::Mul,
                        x.clone(),
                        Box::new(Expr::Unary(OpType::Rsqrt, y.clone())),
                    )),
                    _ => None,
                },
                _ => None,
            },
            RewriteRule::UnfuseMulAdd => match expr {
                Expr::Ternary(OpType::MulAdd, a, b, c) => Some(Expr::Binary(
                    OpType::Add,
                    Box::new(Expr::Binary(OpType::Mul, a.clone(), b.clone())),
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
        _ => false,
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
// Backward Generation (BWD) - Lample & Charton 2019
// ============================================================================

/// De-optimization rewrites for backward data generation.
///
/// These are the inverse of optimization rewrites. Given an optimized expression,
/// they produce an equivalent but less efficient form.
///
/// Based on: "Deep Learning for Symbolic Mathematics" (Lample & Charton, ICLR 2020)
/// The key insight: generate "answers" (optimized code), derive "questions"
/// (unoptimized code) via deterministic transformation.
#[derive(Clone, Debug)]
pub enum UnfuseRewrite {
    /// MulAdd(a, b, c) → a * b + c
    UnfuseMulAdd,
    /// MulRsqrt(x, y) → x * (1/sqrt(y))
    UnfuseMulRsqrt,
    /// MulRsqrt(x, y) → x / sqrt(y)
    MulRsqrtToDiv,
    /// x → x + 0 (add identity)
    AddIdentity,
    /// x → x * 1 (mul identity)
    MulIdentity,
    /// x → --x (double negation)
    DoubleNegate,
    /// 2 * x → x + x (strength reduction inverse)
    MulTwoToAddSelf,
}

impl UnfuseRewrite {
    /// All unfusing rewrites for backward generation.
    pub const ALL: &'static [UnfuseRewrite] = &[
        UnfuseRewrite::UnfuseMulAdd,
        UnfuseRewrite::UnfuseMulRsqrt,
        UnfuseRewrite::MulRsqrtToDiv,
        UnfuseRewrite::AddIdentity,
        UnfuseRewrite::MulIdentity,
        UnfuseRewrite::DoubleNegate,
        UnfuseRewrite::MulTwoToAddSelf,
    ];

    /// Apply this unfusing rewrite to an expression.
    ///
    /// Unlike optimization rewrites that may fail to match, these always succeed
    /// for the appropriate expression types.
    #[must_use]
    pub fn apply(&self, expr: &Expr) -> Option<Expr> {
        match self {
            UnfuseRewrite::UnfuseMulAdd => match expr {
                Expr::Ternary(OpType::MulAdd, a, b, c) => Some(Expr::Binary(
                    OpType::Add,
                    Box::new(Expr::Binary(OpType::Mul, a.clone(), b.clone())),
                    c.clone(),
                )),
                _ => None,
            },
            UnfuseRewrite::UnfuseMulRsqrt => match expr {
                Expr::Binary(OpType::MulRsqrt, x, y) => Some(Expr::Binary(
                    OpType::Mul,
                    x.clone(),
                    Box::new(Expr::Unary(OpType::Rsqrt, y.clone())),
                )),
                _ => None,
            },
            UnfuseRewrite::MulRsqrtToDiv => match expr {
                Expr::Binary(OpType::MulRsqrt, x, y) => Some(Expr::Binary(
                    OpType::Div,
                    x.clone(),
                    Box::new(Expr::Unary(OpType::Sqrt, y.clone())),
                )),
                _ => None,
            },
            UnfuseRewrite::AddIdentity => {
                // x → x + 0
                Some(Expr::Binary(
                    OpType::Add,
                    Box::new(expr.clone()),
                    Box::new(Expr::Const(0.0)),
                ))
            }
            UnfuseRewrite::MulIdentity => {
                // x → x * 1
                Some(Expr::Binary(
                    OpType::Mul,
                    Box::new(expr.clone()),
                    Box::new(Expr::Const(1.0)),
                ))
            }
            UnfuseRewrite::DoubleNegate => {
                // x → --x
                Some(Expr::Unary(
                    OpType::Neg,
                    Box::new(Expr::Unary(OpType::Neg, Box::new(expr.clone()))),
                ))
            }
            UnfuseRewrite::MulTwoToAddSelf => match expr {
                Expr::Binary(OpType::Mul, a, b) => {
                    // Check if either operand is 2.0
                    if matches!(a.as_ref(), Expr::Const(c) if (*c - 2.0).abs() < 1e-6) {
                        Some(Expr::Binary(OpType::Add, b.clone(), b.clone()))
                    } else if matches!(b.as_ref(), Expr::Const(c) if (*c - 2.0).abs() < 1e-6) {
                        Some(Expr::Binary(OpType::Add, a.clone(), a.clone()))
                    } else {
                        None
                    }
                }
                _ => None,
            },
        }
    }
}

/// A training pair for backward generation.
///
/// Contains both the optimized form (target) and unoptimized form (input).
#[derive(Clone, Debug)]
pub struct BwdTrainingPair {
    /// The optimized expression (what we want the model to produce/recognize).
    pub optimized: Expr,
    /// The unoptimized expression (input to the model).
    pub unoptimized: Expr,
    /// Which unfusing rewrites were applied.
    pub rewrites_applied: Vec<UnfuseRewrite>,
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
    /// Maximum number of unfusing rewrites to apply.
    pub max_unfuse_passes: usize,
    /// Probability of applying an unfusing rewrite at each opportunity.
    pub unfuse_prob: f32,
}

impl Default for BwdGenConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            leaf_prob: 0.25,
            num_vars: 4,
            fused_op_prob: 0.4, // Higher probability of fused ops
            max_unfuse_passes: 3,
            unfuse_prob: 0.7,
        }
    }
}

/// Backward expression generator following Lample & Charton's approach.
///
/// Generates optimized expressions (with fused operations), then applies
/// unfusing rewrites to create equivalent unoptimized expressions.
///
/// This is the inverse of how we want the model to work:
/// - Generation: optimized → unoptimized (easy, deterministic)
/// - Inference: unoptimized → optimized (learned)
pub struct BwdGenerator {
    /// Configuration.
    pub config: BwdGenConfig,
    /// Random state.
    state: u64,
}

impl BwdGenerator {
    /// Create a new backward generator with the given seed.
    #[must_use]
    pub fn new(seed: u64, config: BwdGenConfig) -> Self {
        Self {
            config,
            state: seed,
        }
    }

    /// Generate a random f32 in [0, 1).
    fn rand_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 33) as f32 / (1u64 << 31) as f32
    }

    /// Generate a random usize in [0, max).
    fn rand_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.rand_f32() * max as f32) as usize
    }

    /// Generate a backward training pair.
    ///
    /// Returns (optimized, unoptimized) where unoptimized is derived from
    /// optimized via unfusing rewrites.
    pub fn generate(&mut self) -> BwdTrainingPair {
        // Step 1: Generate an optimized expression (rich in fused ops)
        let optimized = self.generate_optimized(0);

        // Step 2: Apply unfusing rewrites to create unoptimized version
        let (unoptimized, rewrites) = self.unfuse_expression(&optimized);

        BwdTrainingPair {
            optimized,
            unoptimized,
            rewrites_applied: rewrites,
        }
    }

    /// Generate an optimized expression tree (biased toward fused ops).
    fn generate_optimized(&mut self, depth: usize) -> Expr {
        // Force leaf at max depth or with probability
        if depth >= self.config.max_depth || self.rand_f32() < self.config.leaf_prob {
            return self.generate_leaf();
        }

        // Decide: fused op or regular op
        if self.rand_f32() < self.config.fused_op_prob {
            // Generate a fused operation
            if self.rand_f32() < 0.6 {
                // MulAdd: a * b + c
                Expr::Ternary(
                    OpType::MulAdd,
                    Box::new(self.generate_optimized(depth + 1)),
                    Box::new(self.generate_optimized(depth + 1)),
                    Box::new(self.generate_optimized(depth + 1)),
                )
            } else {
                // MulRsqrt: x * rsqrt(y)
                Expr::Binary(
                    OpType::MulRsqrt,
                    Box::new(self.generate_optimized(depth + 1)),
                    Box::new(self.generate_optimized(depth + 1)),
                )
            }
        } else {
            // Regular operation
            self.generate_regular_op(depth)
        }
    }

    /// Generate a leaf node (variable or constant).
    fn generate_leaf(&mut self) -> Expr {
        if self.rand_f32() < 0.7 {
            Expr::Var(self.rand_usize(self.config.num_vars) as u8)
        } else {
            // Constants: small range to avoid numerical issues
            let val = self.rand_f32() * 4.0 - 2.0;
            Expr::Const(val)
        }
    }

    /// Generate a regular (non-fused) operation.
    fn generate_regular_op(&mut self, depth: usize) -> Expr {
        let choice = self.rand_usize(8);
        match choice {
            0 => Expr::Binary(
                OpType::Add,
                Box::new(self.generate_optimized(depth + 1)),
                Box::new(self.generate_optimized(depth + 1)),
            ),
            1 => Expr::Binary(
                OpType::Sub,
                Box::new(self.generate_optimized(depth + 1)),
                Box::new(self.generate_optimized(depth + 1)),
            ),
            2 => Expr::Binary(
                OpType::Mul,
                Box::new(self.generate_optimized(depth + 1)),
                Box::new(self.generate_optimized(depth + 1)),
            ),
            3 => Expr::Binary(
                OpType::Min,
                Box::new(self.generate_optimized(depth + 1)),
                Box::new(self.generate_optimized(depth + 1)),
            ),
            4 => Expr::Binary(
                OpType::Max,
                Box::new(self.generate_optimized(depth + 1)),
                Box::new(self.generate_optimized(depth + 1)),
            ),
            5 => Expr::Unary(OpType::Neg, Box::new(self.generate_optimized(depth + 1))),
            6 => Expr::Unary(OpType::Sqrt, Box::new(self.generate_optimized(depth + 1))),
            7 => Expr::Unary(OpType::Abs, Box::new(self.generate_optimized(depth + 1))),
            _ => unreachable!(),
        }
    }

    /// Apply unfusing rewrites to de-optimize an expression.
    ///
    /// Returns the unoptimized expression and the list of rewrites applied.
    fn unfuse_expression(&mut self, expr: &Expr) -> (Expr, Vec<UnfuseRewrite>) {
        let mut result = expr.clone();
        let mut applied = Vec::new();

        for _ in 0..self.config.max_unfuse_passes {
            let (new_result, new_applied) = self.unfuse_pass(&result);
            if new_applied.is_empty() {
                break; // No more unfusing opportunities
            }
            result = new_result;
            applied.extend(new_applied);
        }

        (result, applied)
    }

    /// Single pass of unfusing rewrites over the expression.
    fn unfuse_pass(&mut self, expr: &Expr) -> (Expr, Vec<UnfuseRewrite>) {
        let mut applied = Vec::new();
        let result = self.unfuse_recursive(expr, &mut applied);
        (result, applied)
    }

    /// Recursively apply unfusing rewrites.
    fn unfuse_recursive(&mut self, expr: &Expr, applied: &mut Vec<UnfuseRewrite>) -> Expr {
        // First, try to apply an unfusing rewrite at this node
        let expr = if self.rand_f32() < self.config.unfuse_prob {
            self.try_unfuse_node(expr, applied)
        } else {
            expr.clone()
        };

        // Then recurse into children
        match &expr {
            Expr::Var(_) | Expr::Const(_) => expr,
            Expr::Unary(op, a) => {
                let new_a = self.unfuse_recursive(a, applied);
                Expr::Unary(*op, Box::new(new_a))
            }
            Expr::Binary(op, a, b) => {
                let new_a = self.unfuse_recursive(a, applied);
                let new_b = self.unfuse_recursive(b, applied);
                Expr::Binary(*op, Box::new(new_a), Box::new(new_b))
            }
            Expr::Ternary(op, a, b, c) => {
                let new_a = self.unfuse_recursive(a, applied);
                let new_b = self.unfuse_recursive(b, applied);
                let new_c = self.unfuse_recursive(c, applied);
                Expr::Ternary(*op, Box::new(new_a), Box::new(new_b), Box::new(new_c))
            }
        }
    }

    /// Try to apply an unfusing rewrite at this node.
    fn try_unfuse_node(&mut self, expr: &Expr, applied: &mut Vec<UnfuseRewrite>) -> Expr {
        // Prioritize structural unfusing (MulAdd, MulRsqrt) over identity insertions
        let structural_rewrites = [
            UnfuseRewrite::UnfuseMulAdd,
            UnfuseRewrite::UnfuseMulRsqrt,
            UnfuseRewrite::MulRsqrtToDiv,
            UnfuseRewrite::MulTwoToAddSelf,
        ];

        // Try structural rewrites first
        for rewrite in &structural_rewrites {
            if let Some(result) = rewrite.apply(expr) {
                applied.push(rewrite.clone());
                return result;
            }
        }

        // Occasionally add identity operations (bloat the expression)
        if self.rand_f32() < 0.2 {
            let identity_rewrites = [
                UnfuseRewrite::AddIdentity,
                UnfuseRewrite::MulIdentity,
                UnfuseRewrite::DoubleNegate,
            ];
            let idx = self.rand_usize(identity_rewrites.len());
            let rewrite = &identity_rewrites[idx];
            if let Some(result) = rewrite.apply(expr) {
                applied.push(rewrite.clone());
                return result;
            }
        }

        expr.clone()
    }
}

/// Count fused operations in an expression.
#[must_use]
pub fn count_fused_ops(expr: &Expr) -> usize {
    match expr {
        Expr::Var(_) | Expr::Const(_) => 0,
        Expr::Unary(_, a) => count_fused_ops(a),
        Expr::Binary(op, a, b) => {
            let this = if *op == OpType::MulRsqrt { 1 } else { 0 };
            this + count_fused_ops(a) + count_fused_ops(b)
        }
        Expr::Ternary(op, a, b, c) => {
            let this = if *op == OpType::MulAdd { 1 } else { 0 };
            this + count_fused_ops(a) + count_fused_ops(b) + count_fused_ops(c)
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
    fn op_type_roundtrip_should_succeed_when_called() {
        for i in 0..OpType::COUNT {
            let op = OpType::from_index(i).expect("Expected value but got None/Err");
            assert_eq!(op.index(), i);
        }
    }

    #[test]
    fn feature_roundtrip_should_succeed_when_called() {
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
    fn expr_eval_should_succeed_when_called() {
        // x + y
        let expr = Expr::Binary(OpType::Add, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let result = expr.eval(&[3.0, 4.0, 0.0, 0.0]);
        assert!(fabsf(result - 7.0) < 1e-6);
    }

    #[test]
    fn expr_generator_should_succeed_when_called() {
        let mut generator = ExprGenerator::new(42, ExprGenConfig::default());
        for _ in 0..10 {
            let expr = generator.generate();
            assert!(expr.depth() <= 7); // max_depth + 1 for leaf
            assert!(expr.node_count() > 0);
        }
    }

    #[test]
    fn feature_extraction_should_succeed_when_called() {
        let expr = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(1.0)),
        );
        let features = extract_features(&expr);
        assert!(!features.is_empty());
    }

    #[test]
    fn rewrite_add_zero_should_succeed_when_called() {
        let expr = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(0.0)),
        );
        let rewritten = RewriteRule::AddZero.try_apply(&expr);
        assert!(rewritten.is_some());
        assert!(matches!(rewritten.expect("Expected value but got None/Err"), Expr::Var(0)));
    }

    #[test]
    fn rewrite_fuse_muladd_should_succeed_when_called() {
        // a * b + c
        let expr = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Binary(
                OpType::Mul,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
            )),
            Box::new(Expr::Var(2)),
        );
        let rewritten = RewriteRule::FuseToMulAdd.try_apply(&expr);
        assert!(rewritten.is_some());
        assert!(matches!(
            rewritten.expect("Expected value but got None/Err"),
            Expr::Ternary(OpType::MulAdd, _, _, _)
        ));
    }

    #[test]
    fn find_all_rewrites_should_succeed_when_called() {
        // (x + 0) + y - should find AddZero at path [0]
        let expr = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Binary(
                OpType::Add,
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
    fn accumulator_add_remove_should_succeed_when_called() {
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
    // Backward Generation Tests
    // ========================================================================

    #[test]
    fn unfuse_muladd_should_succeed_when_called() {
        // MulAdd(x, y, z) → x * y + z
        let expr = Expr::Ternary(
            OpType::MulAdd,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
            Box::new(Expr::Var(2)),
        );

        let result = UnfuseRewrite::UnfuseMulAdd.apply(&expr);
        assert!(result.is_some());

        let unoptimized = result.expect("Expected value but got None/Err");
        // Should be Add(Mul(x, y), z)
        assert!(matches!(unoptimized, Expr::Binary(OpType::Add, _, _)));

        // Verify semantic equivalence
        let vars = [2.0, 3.0, 4.0, 0.0];
        let orig_val = expr.eval(&vars);
        let unfused_val = unoptimized.eval(&vars);
        assert!((orig_val - unfused_val).abs() < 1e-6);
    }

    #[test]
    fn unfuse_mulrsqrt_should_succeed_when_called() {
        // MulRsqrt(x, y) → x * rsqrt(y)
        let expr = Expr::Binary(
            OpType::MulRsqrt,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        );

        let result = UnfuseRewrite::UnfuseMulRsqrt.apply(&expr);
        assert!(result.is_some());

        let unoptimized = result.expect("Expected value but got None/Err");
        // Should be Mul(x, Rsqrt(y))
        assert!(matches!(unoptimized, Expr::Binary(OpType::Mul, _, _)));

        // Verify semantic equivalence
        let vars = [6.0, 4.0, 0.0, 0.0]; // x=6, y=4 → 6/sqrt(4) = 3
        let orig_val = expr.eval(&vars);
        let unfused_val = unoptimized.eval(&vars);
        assert!((orig_val - unfused_val).abs() < 1e-6);
    }

    #[test]
    fn unfuse_add_identity_should_succeed_when_called() {
        let expr = Expr::Var(0);
        let result = UnfuseRewrite::AddIdentity.apply(&expr);
        assert!(result.is_some());

        let unoptimized = result.expect("Expected value but got None/Err");
        // Should be x + 0
        assert!(matches!(unoptimized, Expr::Binary(OpType::Add, _, _)));

        // Verify semantic equivalence
        let vars = [5.0, 0.0, 0.0, 0.0];
        let orig_val = expr.eval(&vars);
        let unfused_val = unoptimized.eval(&vars);
        assert!((orig_val - unfused_val).abs() < 1e-6);
    }

    #[test]
    fn bwd_generator_produces_valid_pairs_should_succeed_when_called() {
        let config = BwdGenConfig::default();
        let mut generator = BwdGenerator::new(42, config);

        for _ in 0..10 {
            let pair = generator.generate();

            // Both expressions should be valid
            assert!(pair.optimized.node_count() > 0);
            assert!(pair.unoptimized.node_count() > 0);

            // Unoptimized should generally be larger or equal
            // (unfusing increases or maintains size)
            assert!(pair.unoptimized.node_count() >= pair.optimized.node_count());

            // Verify semantic equivalence
            let vars = [1.5, 2.5, 3.5, 0.5];
            let opt_val = pair.optimized.eval(&vars);
            let unopt_val = pair.unoptimized.eval(&vars);

            // Allow for NaN (from sqrt of negative, etc)
            if !opt_val.is_nan() && !unopt_val.is_nan() {
                assert!(
                    (opt_val - unopt_val).abs() < 1e-4,
                    "Semantic equivalence failed: {} vs {}",
                    opt_val,
                    unopt_val
                );
            }
        }
    }

    #[test]
    fn bwd_generator_has_fused_ops_should_succeed_when_called() {
        let config = BwdGenConfig {
            fused_op_prob: 0.8, // High probability of fused ops
            max_depth: 4,
            ..Default::default()
        };
        let mut generator = BwdGenerator::new(12345, config);

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
    fn count_fused_ops_should_succeed_when_called() {
        // Expression with MulAdd and MulRsqrt
        let expr = Expr::Ternary(
            OpType::MulAdd,
            Box::new(Expr::Binary(
                OpType::MulRsqrt,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
            )),
            Box::new(Expr::Var(2)),
            Box::new(Expr::Var(3)),
        );

        assert_eq!(count_fused_ops(&expr), 2);
    }
}
