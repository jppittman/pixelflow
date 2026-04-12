//! # Evaluator Abstraction for Optimization Problems
//!
//! This module provides a clean abstraction for "Game-like" optimization problems,
//! decoupling the **Logic** (what moves are legal) from the **Physics** (what moves are good).
//!
//! ## The Chess Analogy
//!
//! Chess engines evolved through three eras:
//! 1. **Rule-Based**: Count material (queen=9, rook=5, etc.)
//! 2. **HCE (Hand-Crafted Evaluation)**: Linear sum of features with tuned weights
//! 3. **NNUE**: Learned evaluation from self-play
//!
//! We follow the same path for compiler optimization.
//!
//! ## The Abstraction
//!
//! Any optimization problem can be framed as:
//!
//! ```text
//! Position  →  Features  →  Evaluation
//!     ↓
//!   Moves
//!     ↓
//! New Position
//! ```
//!
//! The [`Domain`] trait captures this:
//! - **Position**: The thing we're evaluating (expression, schedule, etc.)
//! - **Features**: What we extract from the position (operation counts, patterns)
//! - **Move**: A transition to a new position (rewrite rule application)
//!
//! ## Example: Math Expressions
//!
//! ```ignore
//! // Position: x * 1 + 0
//! // Features: {mul_count: 1, add_count: 1, has_identity: true}
//! // Moves: [MulOne, AddZero, FuseToMulAdd]
//! //
//! // After AddZero: x * 1
//! // Features: {mul_count: 1, add_count: 0, has_identity: true}
//! //
//! // After MulOne: x
//! // Features: {mul_count: 0, add_count: 0, has_identity: false}
//! ```
//!
//! ## Design Principles
//!
//! 1. **Atoms before Molecules**: Features are the atoms. Evaluation combines them.
//! 2. **Physics is Configurable**: Same features, different weights = different engine
//! 3. **Incremental by Design**: Features designed for efficient delta updates

#![allow(dead_code)] // Evolving API

use alloc::vec::Vec;

// ============================================================================
// The Domain Trait: What You're Optimizing
// ============================================================================

/// A domain defines the "game" being played.
///
/// This is the core abstraction that decouples:
/// - **Logic**: What positions exist, what moves are legal
/// - **Physics**: What makes a position good or bad
///
/// # Type Parameters
///
/// The associated types define the "vocabulary" of your domain:
/// - `Position`: The state being evaluated (expression tree, IR graph, schedule)
/// - `Move`: A transformation between positions (rewrite rule, tile choice)
/// - `Features`: Extracted observations (counts, patterns, structure)
pub trait Domain {
    /// The state being evaluated.
    ///
    /// For expressions: an AST or e-class ID.
    /// For scheduling: a partial schedule.
    type Position;

    /// A transition between positions.
    ///
    /// For expressions: a rewrite rule + where to apply it.
    /// For scheduling: a tiling/ordering decision.
    type Move;

    /// Extracted features from a position.
    ///
    /// This is what the evaluator "sees". Design these for:
    /// 1. **Informative**: Capture what matters for cost
    /// 2. **Incremental**: Efficient delta updates when moves are made
    /// 3. **Bounded**: Fixed-size for efficient storage
    type Features: Clone;

    /// Extract features from a position.
    ///
    /// This is the "sensing" step - turning a complex position into
    /// a fixed-size feature vector that captures what matters.
    fn extract_features(pos: &Self::Position) -> Self::Features;

    /// Enumerate legal moves from a position.
    ///
    /// Like legal moves in chess - all valid transformations that
    /// maintain semantic equivalence.
    fn legal_moves(pos: &Self::Position) -> Vec<Self::Move>;

    /// Apply a move to get a new position.
    ///
    /// Returns `None` if the move is invalid for this position.
    fn apply_move(pos: &Self::Position, mv: &Self::Move) -> Option<Self::Position>;

    /// Delta-update features after a move (optional optimization).
    ///
    /// Default implementation re-extracts from scratch.
    /// Override this for O(move_size) instead of O(position_size).
    fn update_features(
        _old_features: &Self::Features,
        _old_pos: &Self::Position,
        _mv: &Self::Move,
        new_pos: &Self::Position,
    ) -> Self::Features {
        Self::extract_features(new_pos)
    }
}

// ============================================================================
// The Evaluator Trait: How You Score Positions
// ============================================================================

/// An evaluator scores positions in a domain.
///
/// This is the "physics" - given features, produce a score.
/// Lower is better (we're minimizing cost).
///
/// # Design
///
/// Evaluators are separate from domains because:
/// 1. Same domain, different evaluators (HCE vs NNUE)
/// 2. Evaluators can be trained/tuned independently
/// 3. Easy A/B testing of evaluation strategies
pub trait Evaluator<D: Domain> {
    /// Evaluate a position's features, returning a score.
    ///
    /// Lower scores are better (cost minimization).
    /// The scale is arbitrary but should be consistent.
    fn evaluate(&self, features: &D::Features) -> i32;

    /// Evaluate a position directly (convenience method).
    fn evaluate_position(&self, pos: &D::Position) -> i32 {
        let features = D::extract_features(pos);
        self.evaluate(&features)
    }
}

// ============================================================================
// Hand-Crafted Evaluator (HCE): The Linear Summer
// ============================================================================

/// A feature vector that can be evaluated by a linear HCE.
///
/// This trait allows the HCE to iterate over feature values
/// and compute a weighted sum.
pub trait LinearFeatures {
    /// Number of features in this vector.
    fn len(&self) -> usize;

    /// Get the value of feature at index `i`.
    ///
    /// Features are typically counts (0, 1, 2, ...) or
    /// boolean indicators (0 or 1).
    fn get(&self, i: usize) -> i32;

    /// Feature names for debugging/visualization.
    fn feature_names() -> &'static [&'static str];

    /// Check if the feature vector is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Hand-Crafted Evaluator: A simple linear combination of features.
///
/// ```text
/// eval = Σ(weight[i] * feature[i])
/// ```
///
/// This is effectively a 1-layer neural network with no activation.
/// It's the "apprentice" that we'll later upgrade to NNUE.
///
/// # Why Start Here?
///
/// 1. **Debuggable**: You can see exactly what each feature contributes
/// 2. **Tunable**: Adjust weights by hand to fix obvious problems
/// 3. **Fast**: Just a dot product, runs at memory bandwidth
/// 4. **Baseline**: Proves the system works before adding ML complexity
#[derive(Clone, Debug)]
pub struct HandCraftedEvaluator {
    /// Weights for each feature. `weights[i]` is the cost contribution
    /// of one unit of feature `i`.
    pub weights: Vec<i32>,
}

impl HandCraftedEvaluator {
    /// Create a new HCE with the given weights.
    #[must_use]
    pub fn new(weights: Vec<i32>) -> Self {
        Self { weights }
    }

    /// Create an HCE with all weights set to zero.
    #[must_use]
    pub fn zeros(num_features: usize) -> Self {
        Self {
            weights: alloc::vec![0; num_features],
        }
    }

    /// Set a specific weight.
    pub fn set_weight(&mut self, index: usize, weight: i32) {
        if index < self.weights.len() {
            self.weights[index] = weight;
        }
    }

    /// Get a specific weight.
    #[must_use]
    pub fn get_weight(&self, index: usize) -> i32 {
        self.weights.get(index).copied().unwrap_or(0)
    }

    /// Evaluate features using the linear combination.
    pub fn evaluate_linear<F: LinearFeatures>(&self, features: &F) -> i32 {
        let mut score = 0i32;
        for i in 0..features.len().min(self.weights.len()) {
            score = score.saturating_add(self.weights[i].saturating_mul(features.get(i)));
        }
        score
    }
}

// ============================================================================
// Expression Domain: Math Expressions as a "Game"
// ============================================================================

/// Features extracted from a math expression.
///
/// These are the "atoms" that the evaluator senses.
/// Designed to capture cost-relevant properties.
#[derive(Clone, Debug, Default)]
pub struct ExprFeatures {
    // === Operation Counts ===
    /// Number of additions
    pub add_count: i32,
    /// Number of subtractions
    pub sub_count: i32,
    /// Number of multiplications
    pub mul_count: i32,
    /// Number of divisions (expensive!)
    pub div_count: i32,
    /// Number of negations (cheap)
    pub neg_count: i32,
    /// Number of square roots (expensive!)
    pub sqrt_count: i32,
    /// Number of reciprocal square roots (cheaper than div+sqrt)
    pub rsqrt_count: i32,
    /// Number of absolute values (cheap)
    pub abs_count: i32,
    /// Number of min operations
    pub min_count: i32,
    /// Number of max operations
    pub max_count: i32,

    // === Fused Operation Counts ===
    /// Number of fused multiply-adds (cheaper than mul+add)
    pub fma_count: i32,
    /// Number of fused multiply-rsqrt (cheaper than mul+rsqrt)
    pub mul_rsqrt_count: i32,

    // === Structural Features ===
    /// Total node count (expression size)
    pub node_count: i32,
    /// Maximum depth of expression tree
    pub depth: i32,
    /// Number of variable references
    pub var_count: i32,
    /// Number of constants
    pub const_count: i32,

    // === Pattern Features ===
    /// Has identity operation (x*1, x+0, etc.) that could be simplified
    pub has_identity: i32,
    /// Has self-canceling pattern (x-x, x/x)
    pub has_self_cancel: i32,
    /// Has fusable pattern (a*b+c without FMA)
    pub has_fusable: i32,

    // === ILP Features (Non-linear insight captured as precomputed values) ===
    /// Critical path cost: longest dependency chain (not sum of all ops).
    /// This captures ILP - parallel ops don't add to critical path.
    pub critical_path: i32,
    /// Maximum width: max nodes at any depth level.
    /// Approximates register pressure (more parallel = more live values).
    pub max_width: i32,
}

impl ExprFeatures {
    /// Number of features in this struct.
    pub const COUNT: usize = 21;

    /// Feature names for debugging.
    pub const NAMES: [&'static str; Self::COUNT] = [
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "sqrt",
        "rsqrt",
        "abs",
        "min",
        "max",
        "fma",
        "mul_rsqrt",
        "nodes",
        "depth",
        "vars",
        "consts",
        "has_identity",
        "has_self_cancel",
        "has_fusable",
        "critical_path",
        "max_width",
    ];
}

impl LinearFeatures for ExprFeatures {
    fn len(&self) -> usize {
        Self::COUNT
    }

    fn get(&self, i: usize) -> i32 {
        match i {
            0 => self.add_count,
            1 => self.sub_count,
            2 => self.mul_count,
            3 => self.div_count,
            4 => self.neg_count,
            5 => self.sqrt_count,
            6 => self.rsqrt_count,
            7 => self.abs_count,
            8 => self.min_count,
            9 => self.max_count,
            10 => self.fma_count,
            11 => self.mul_rsqrt_count,
            12 => self.node_count,
            13 => self.depth,
            14 => self.var_count,
            15 => self.const_count,
            16 => self.has_identity,
            17 => self.has_self_cancel,
            18 => self.has_fusable,
            19 => self.critical_path,
            20 => self.max_width,
            _ => 0,
        }
    }

    fn feature_names() -> &'static [&'static str] {
        &Self::NAMES
    }
}

// ============================================================================
// Default Weights: "Gut Feeling" Initialization
// ============================================================================

/// Default weights based on approximate x86-64 cycle counts.
///
/// These are the "gut feeling" weights that we start with.
/// They're not perfect, but they're a reasonable starting point.
///
/// # Philosophy
///
/// - **Operations**: Cost ≈ latency in cycles
/// - **Fused ops**: Should be cheaper than unfused equivalents
/// - **Patterns**: Negative weights = "good to have" (will reduce cost)
/// - **Structure**: Slight bias toward smaller expressions
/// - **ILP features**: Critical path matters more than total ops
#[must_use]
pub fn default_expr_weights() -> HandCraftedEvaluator {
    let mut hce = HandCraftedEvaluator::zeros(ExprFeatures::COUNT);

    // Operation costs (approximate x86-64 cycles)
    // NOTE: These are now SECONDARY to critical_path for ILP-aware evaluation
    hce.set_weight(0, 4); // add: ~4 cycles
    hce.set_weight(1, 4); // sub: ~4 cycles
    hce.set_weight(2, 5); // mul: ~5 cycles
    hce.set_weight(3, 15); // div: ~15-20 cycles (expensive!)
    hce.set_weight(4, 1); // neg: ~1 cycle (just sign flip)
    hce.set_weight(5, 15); // sqrt: ~15-20 cycles
    hce.set_weight(6, 5); // rsqrt: ~5 cycles (fast approximation)
    hce.set_weight(7, 1); // abs: ~1 cycle (just clear sign bit)
    hce.set_weight(8, 4); // min: ~4 cycles
    hce.set_weight(9, 4); // max: ~4 cycles

    // Fused operation costs (should be cheaper than sum of parts)
    hce.set_weight(10, 5); // fma: ~5 cycles (same as mul alone!)
    hce.set_weight(11, 6); // mul_rsqrt: ~6 cycles

    // Structural features (mild preferences)
    hce.set_weight(12, 0); // node_count: neutral (covered by ops)
    hce.set_weight(13, 0); // depth: neutral (superseded by critical_path)
    hce.set_weight(14, 0); // var_count: free (just register refs)
    hce.set_weight(15, 0); // const_count: free (immediates)

    // Pattern features (opportunities for improvement)
    // Negative = "having this is good" because it indicates simplification potential
    hce.set_weight(16, 0); // has_identity: neutral (rewrite will fix)
    hce.set_weight(17, 0); // has_self_cancel: neutral
    hce.set_weight(18, -2); // has_fusable: slight bonus if FMA available

    // ILP features - THE KEY NON-LINEAR INSIGHT
    // critical_path captures actual execution time on superscalar CPUs
    hce.set_weight(19, 1); // critical_path: primary cost driver
    // max_width approximates register pressure (more parallel = more regs needed)
    hce.set_weight(20, 1); // max_width: penalty for wide expressions

    hce
}

/// Weights optimized for CPUs with FMA (Haswell+, Zen+).
///
/// These weights assume FMA is "free" (same cost as mul).
#[must_use]
pub fn fma_optimized_weights() -> HandCraftedEvaluator {
    let mut hce = default_expr_weights();

    // FMA is as cheap as mul on modern CPUs
    hce.set_weight(10, 5); // fma: ~5 cycles

    // Strong incentive to use FMA
    hce.set_weight(18, -4); // has_fusable: big bonus

    hce
}

// ============================================================================
// Feature Extraction from Expr (bridges nnue module)
// ============================================================================

use crate::nnue::{Expr, OpType};

/// Extract features from an expression.
///
/// This is the "sensing" step that converts a complex AST into
/// a fixed-size feature vector for the evaluator.
#[must_use]
pub fn extract_expr_features(expr: &Expr) -> ExprFeatures {
    let mut features = ExprFeatures::default();
    let mut width_at_depth = Vec::new();
    let critical_path = extract_features_recursive(expr, &mut features, 0, &mut width_at_depth);
    features.critical_path = critical_path;
    features.max_width = width_at_depth.iter().copied().max().unwrap_or(0);
    features
}

/// Returns the critical path cost of this subtree.
fn extract_features_recursive(
    expr: &Expr,
    features: &mut ExprFeatures,
    depth: usize,
    width_at_depth: &mut Vec<i32>,
) -> i32 {
    features.node_count += 1;
    features.depth = features.depth.max(depth as i32 + 1);

    // Track width at each depth level
    if depth >= width_at_depth.len() {
        width_at_depth.resize(depth + 1, 0);
    }
    width_at_depth[depth] += 1;

    match expr {
        Expr::Var(_) => {
            features.var_count += 1;
            0 // No latency for variable access
        }
        Expr::Const(c) => {
            features.const_count += 1;
            // Check for identity constants
            if (*c - 0.0).abs() < 1e-10 || (*c - 1.0).abs() < 1e-10 {
                // Might be part of identity pattern
            }
            0 // No latency for constant
        }
        Expr::Unary(op, a) => {
            let op_cost = match op {
                OpType::Neg => {
                    features.neg_count += 1;
                    1
                }
                OpType::Sqrt => {
                    features.sqrt_count += 1;
                    15
                }
                OpType::Rsqrt => {
                    features.rsqrt_count += 1;
                    5
                }
                OpType::Abs => {
                    features.abs_count += 1;
                    1
                }
                _ => 5, // Default for unknown ops
            };
            let child_critical = extract_features_recursive(a, features, depth + 1, width_at_depth);
            op_cost + child_critical
        }
        Expr::Binary(op, a, b) => {
            let op_cost = match op {
                OpType::Add => {
                    features.add_count += 1;
                    // Check for fusable: if 'a' is a Mul, this is a*b+c pattern
                    if matches!(a.as_ref(), Expr::Binary(OpType::Mul, _, _)) {
                        features.has_fusable += 1;
                    }
                    // Check for identity: x + 0
                    if is_zero(b) || is_zero(a) {
                        features.has_identity += 1;
                    }
                    4
                }
                OpType::Sub => {
                    features.sub_count += 1;
                    // Check for self-cancel: x - x
                    if exprs_structurally_equal(a, b) {
                        features.has_self_cancel += 1;
                    }
                    4
                }
                OpType::Mul => {
                    features.mul_count += 1;
                    // Check for identity: x * 1
                    if is_one(b) || is_one(a) {
                        features.has_identity += 1;
                    }
                    // Check for fusable: x * rsqrt(y)
                    if matches!(b.as_ref(), Expr::Unary(OpType::Rsqrt, _))
                        || matches!(a.as_ref(), Expr::Unary(OpType::Rsqrt, _))
                    {
                        features.has_fusable += 1;
                    }
                    5
                }
                OpType::Div => {
                    features.div_count += 1;
                    // Check for self-cancel: x / x
                    if exprs_structurally_equal(a, b) {
                        features.has_self_cancel += 1;
                    }
                    15
                }
                OpType::Min => {
                    features.min_count += 1;
                    4
                }
                OpType::Max => {
                    features.max_count += 1;
                    4
                }
                OpType::MulRsqrt => {
                    features.mul_rsqrt_count += 1;
                    6
                }
                _ => 5, // Default
            };
            let crit_a = extract_features_recursive(a, features, depth + 1, width_at_depth);
            let crit_b = extract_features_recursive(b, features, depth + 1, width_at_depth);
            // Critical path = max of children (parallel execution) + this op
            op_cost + crit_a.max(crit_b)
        }
        Expr::Ternary(op, a, b, c) => {
            let op_cost = match op {
                OpType::MulAdd => {
                    features.fma_count += 1;
                    5
                }
                _ => 10, // Default for unknown ternary
            };
            let crit_a = extract_features_recursive(a, features, depth + 1, width_at_depth);
            let crit_b = extract_features_recursive(b, features, depth + 1, width_at_depth);
            let crit_c = extract_features_recursive(c, features, depth + 1, width_at_depth);
            // Critical path = max of all children + this op
            op_cost + crit_a.max(crit_b).max(crit_c)
        }
    }
}

/// Check if an expression is the constant zero.
fn is_zero(expr: &Expr) -> bool {
    matches!(expr, Expr::Const(c) if (*c).abs() < 1e-10)
}

/// Check if an expression is the constant one.
fn is_one(expr: &Expr) -> bool {
    matches!(expr, Expr::Const(c) if (*c - 1.0).abs() < 1e-10)
}

/// Check if two expressions are structurally equal.
fn exprs_structurally_equal(a: &Expr, b: &Expr) -> bool {
    match (a, b) {
        (Expr::Var(i), Expr::Var(j)) => i == j,
        (Expr::Const(x), Expr::Const(y)) => (x - y).abs() < 1e-10,
        (Expr::Unary(op1, a1), Expr::Unary(op2, b1)) => {
            op1 == op2 && exprs_structurally_equal(a1.as_ref(), b1.as_ref())
        }
        (Expr::Binary(op1, a1, a2), Expr::Binary(op2, b1, b2)) => {
            op1 == op2
                && exprs_structurally_equal(a1.as_ref(), b1.as_ref())
                && exprs_structurally_equal(a2.as_ref(), b2.as_ref())
        }
        (Expr::Ternary(op1, a1, a2, a3), Expr::Ternary(op2, b1, b2, b3)) => {
            op1 == op2
                && exprs_structurally_equal(a1.as_ref(), b1.as_ref())
                && exprs_structurally_equal(a2.as_ref(), b2.as_ref())
                && exprs_structurally_equal(a3.as_ref(), b3.as_ref())
        }
        _ => false,
    }
}

// ============================================================================
// Domain Implementation for Expressions
// ============================================================================

use crate::nnue::{RewriteRule, find_all_rewrites};

/// The expression domain - math expressions as a "game".
pub struct ExprDomain;

/// A move in the expression game: a rewrite rule + path to apply it.
#[derive(Clone, Debug)]
pub struct ExprMove {
    /// Path to the subexpression (child indices).
    pub path: Vec<usize>,
    /// The rewrite rule to apply.
    pub rule: RewriteRule,
}

impl Domain for ExprDomain {
    type Position = Expr;
    type Move = ExprMove;
    type Features = ExprFeatures;

    fn extract_features(pos: &Self::Position) -> Self::Features {
        extract_expr_features(pos)
    }

    fn legal_moves(pos: &Self::Position) -> Vec<Self::Move> {
        find_all_rewrites(pos)
            .into_iter()
            .map(|(path, rule, _)| ExprMove { path, rule })
            .collect()
    }

    fn apply_move(pos: &Self::Position, mv: &Self::Move) -> Option<Self::Position> {
        // Navigate to the target subexpression and apply the rewrite
        apply_rewrite_at_path(pos, &mv.path, &mv.rule)
    }
}

/// Apply a rewrite at a specific path in the expression tree.
fn apply_rewrite_at_path(expr: &Expr, path: &[usize], rule: &RewriteRule) -> Option<Expr> {
    if path.is_empty() {
        // Apply rule at this node
        rule.try_apply(expr)
    } else {
        // Navigate deeper
        let idx = path[0];
        let rest = &path[1..];

        match expr {
            Expr::Var(_) | Expr::Const(_) => None, // Can't go deeper
            Expr::Unary(op, a) if idx == 0 => apply_rewrite_at_path(a, rest, rule)
                .map(|new_a| Expr::Unary(*op, alloc::boxed::Box::new(new_a))),
            Expr::Binary(op, a, b) => match idx {
                0 => apply_rewrite_at_path(a, rest, rule)
                    .map(|new_a| Expr::Binary(*op, alloc::boxed::Box::new(new_a), b.clone())),
                1 => apply_rewrite_at_path(b, rest, rule)
                    .map(|new_b| Expr::Binary(*op, a.clone(), alloc::boxed::Box::new(new_b))),
                _ => None,
            },
            Expr::Ternary(op, a, b, c) => match idx {
                0 => apply_rewrite_at_path(a, rest, rule).map(|new_a| {
                    Expr::Ternary(*op, alloc::boxed::Box::new(new_a), b.clone(), c.clone())
                }),
                1 => apply_rewrite_at_path(b, rest, rule).map(|new_b| {
                    Expr::Ternary(*op, a.clone(), alloc::boxed::Box::new(new_b), c.clone())
                }),
                2 => apply_rewrite_at_path(c, rest, rule).map(|new_c| {
                    Expr::Ternary(*op, a.clone(), b.clone(), alloc::boxed::Box::new(new_c))
                }),
                _ => None,
            },
            _ => None,
        }
    }
}

/// Implement the Evaluator trait for HCE + ExprDomain.
impl Evaluator<ExprDomain> for HandCraftedEvaluator {
    fn evaluate(&self, features: &ExprFeatures) -> i32 {
        self.evaluate_linear(features)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn expr_features_linear_should_succeed_when_called() {
        let mut features = ExprFeatures::default();
        features.add_count = 2;
        features.mul_count = 3;
        features.div_count = 1;

        let hce = default_expr_weights();
        let score = hce.evaluate_linear(&features);

        // 2*4 + 3*5 + 1*15 = 8 + 15 + 15 = 38
        assert_eq!(score, 38);
    }

    #[test]
    fn fma_incentive_should_succeed_when_called() {
        // Without FMA: mul + add = 5 + 4 = 9
        let mut without_fma = ExprFeatures::default();
        without_fma.mul_count = 1;
        without_fma.add_count = 1;

        // With FMA: single fused op = 5
        let mut with_fma = ExprFeatures::default();
        with_fma.fma_count = 1;

        let hce = fma_optimized_weights();

        let cost_without = hce.evaluate_linear(&without_fma);
        let cost_with = hce.evaluate_linear(&with_fma);

        // FMA should be cheaper
        assert!(
            cost_with < cost_without,
            "FMA ({}) should be cheaper than mul+add ({})",
            cost_with,
            cost_without
        );
    }

    #[test]
    fn div_is_expensive_should_succeed_when_called() {
        let mut one_div = ExprFeatures::default();
        one_div.div_count = 1;

        let mut three_muls = ExprFeatures::default();
        three_muls.mul_count = 3;

        let hce = default_expr_weights();

        let div_cost = hce.evaluate_linear(&one_div);
        let mul_cost = hce.evaluate_linear(&three_muls);

        // One div (15) should be as expensive as 3 muls (15)
        assert_eq!(div_cost, mul_cost);
    }

    #[test]
    fn feature_names_match_count_should_succeed_when_called() {
        assert_eq!(ExprFeatures::NAMES.len(), ExprFeatures::COUNT);
    }

    // =========================================================================
    // Feature Extraction Tests
    // =========================================================================

    #[test]
    fn extract_simple_add_should_succeed_when_called() {
        use alloc::boxed::Box;

        // x + y
        let expr = Expr::Binary(OpType::Add, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let features = extract_expr_features(&expr);

        assert_eq!(features.add_count, 1);
        assert_eq!(features.var_count, 2);
        assert_eq!(features.node_count, 3);
        assert_eq!(features.depth, 2);
    }

    #[test]
    fn extract_fma_pattern_should_succeed_when_called() {
        use alloc::boxed::Box;

        // (a * b) + c - fusable pattern
        let expr = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Binary(
                OpType::Mul,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
            )),
            Box::new(Expr::Var(2)),
        );
        let features = extract_expr_features(&expr);

        assert_eq!(features.add_count, 1);
        assert_eq!(features.mul_count, 1);
        assert_eq!(features.has_fusable, 1, "Should detect a*b+c as fusable");
    }

    #[test]
    fn extract_identity_mul_one_should_succeed_when_called() {
        use alloc::boxed::Box;

        // x * 1
        let expr = Expr::Binary(
            OpType::Mul,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(1.0)),
        );
        let features = extract_expr_features(&expr);

        assert_eq!(features.has_identity, 1, "Should detect x*1 as identity");
    }

    #[test]
    fn extract_self_cancel_should_succeed_when_called() {
        use alloc::boxed::Box;

        // x - x
        let expr = Expr::Binary(OpType::Sub, Box::new(Expr::Var(0)), Box::new(Expr::Var(0)));
        let features = extract_expr_features(&expr);

        assert_eq!(
            features.has_self_cancel, 1,
            "Should detect x-x as self-cancel"
        );
    }

    // =========================================================================
    // ILP Feature Tests
    // =========================================================================

    #[test]
    fn critical_path_wide_vs_deep_should_succeed_when_called() {
        use alloc::boxed::Box;

        // Wide expression: (a + b) + (c + d)
        // Critical path: 4 + 4 = 8 (two adds in sequence, but children parallel)
        let wide = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Binary(
                OpType::Add,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
            )),
            Box::new(Expr::Binary(
                OpType::Add,
                Box::new(Expr::Var(2)),
                Box::new(Expr::Var(3)),
            )),
        );

        // Deep expression: ((a + b) + c) + d
        // Critical path: 4 + 4 + 4 = 12 (three sequential adds)
        let deep = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Binary(
                OpType::Add,
                Box::new(Expr::Binary(
                    OpType::Add,
                    Box::new(Expr::Var(0)),
                    Box::new(Expr::Var(1)),
                )),
                Box::new(Expr::Var(2)),
            )),
            Box::new(Expr::Var(3)),
        );

        let wide_features = extract_expr_features(&wide);
        let deep_features = extract_expr_features(&deep);

        // Same total operation count
        assert_eq!(wide_features.add_count, 3, "Wide should have 3 adds");
        assert_eq!(deep_features.add_count, 3, "Deep should have 3 adds");

        // But different critical paths
        assert_eq!(
            wide_features.critical_path, 8,
            "Wide: two levels of adds = 8"
        );
        assert_eq!(
            deep_features.critical_path, 12,
            "Deep: three sequential adds = 12"
        );

        // Critical path prefers wide (more parallel)
        assert!(
            wide_features.critical_path < deep_features.critical_path,
            "Wide ({}) should have shorter critical path than deep ({})",
            wide_features.critical_path,
            deep_features.critical_path
        );
    }

    #[test]
    fn max_width_computation_should_succeed_when_called() {
        use alloc::boxed::Box;

        // Wide expression: (a + b) + (c + d)
        // Width at each depth:
        //   depth 0: 1 (root add)
        //   depth 1: 2 (two child adds)
        //   depth 2: 4 (four vars)
        let wide = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Binary(
                OpType::Add,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
            )),
            Box::new(Expr::Binary(
                OpType::Add,
                Box::new(Expr::Var(2)),
                Box::new(Expr::Var(3)),
            )),
        );

        // Deep expression: ((a + b) + c) + d
        // Width at each depth:
        //   depth 0: 1
        //   depth 1: 2 (add + var)
        //   depth 2: 2 (add + var)
        //   depth 3: 2 (two vars)
        let deep = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Binary(
                OpType::Add,
                Box::new(Expr::Binary(
                    OpType::Add,
                    Box::new(Expr::Var(0)),
                    Box::new(Expr::Var(1)),
                )),
                Box::new(Expr::Var(2)),
            )),
            Box::new(Expr::Var(3)),
        );

        let wide_features = extract_expr_features(&wide);
        let deep_features = extract_expr_features(&deep);

        // Wide has higher max_width (more parallel = more live values)
        assert_eq!(
            wide_features.max_width, 4,
            "Wide max_width should be 4 (all vars at depth 2)"
        );
        assert_eq!(deep_features.max_width, 2, "Deep max_width should be 2");

        assert!(
            wide_features.max_width > deep_features.max_width,
            "Wide ({}) should have higher max_width than deep ({})",
            wide_features.max_width,
            deep_features.max_width
        );
    }

    #[test]
    fn ilp_features_affect_cost_should_succeed_when_called() {
        use alloc::boxed::Box;

        // Wide expression: (a + b) + (c + d)
        let wide = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Binary(
                OpType::Add,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
            )),
            Box::new(Expr::Binary(
                OpType::Add,
                Box::new(Expr::Var(2)),
                Box::new(Expr::Var(3)),
            )),
        );

        // Deep expression: ((a + b) + c) + d
        let deep = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Binary(
                OpType::Add,
                Box::new(Expr::Binary(
                    OpType::Add,
                    Box::new(Expr::Var(0)),
                    Box::new(Expr::Var(1)),
                )),
                Box::new(Expr::Var(2)),
            )),
            Box::new(Expr::Var(3)),
        );

        let hce = default_expr_weights();
        let wide_cost = hce.evaluate_linear(&extract_expr_features(&wide));
        let deep_cost = hce.evaluate_linear(&extract_expr_features(&deep));

        // Wide has: 3 adds (12) + critical_path 8 + max_width 4 = 24
        // Deep has: 3 adds (12) + critical_path 12 + max_width 2 = 26
        // So even though wide has higher register pressure, its shorter critical path wins
        assert!(
            wide_cost < deep_cost,
            "Wide ({}) should be cheaper than deep ({}) due to shorter critical path",
            wide_cost,
            deep_cost
        );
    }

    // =========================================================================
    // Domain Tests
    // =========================================================================

    #[test]
    fn domain_legal_moves_should_succeed_when_called() {
        use alloc::boxed::Box;

        // x + 0 - should have AddZero move available
        let expr = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(0.0)),
        );

        let moves = ExprDomain::legal_moves(&expr);
        assert!(
            !moves.is_empty(),
            "Should have at least one legal move (AddZero)"
        );

        // Check that one move is AddZero
        let has_add_zero = moves.iter().any(|m| matches!(m.rule, RewriteRule::AddZero));
        assert!(has_add_zero, "Should have AddZero move");
    }

    #[test]
    fn domain_apply_move_should_succeed_when_called() {
        use alloc::boxed::Box;

        // x + 0
        let expr = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(0.0)),
        );

        let mv = ExprMove {
            path: vec![],
            rule: RewriteRule::AddZero,
        };

        let result = ExprDomain::apply_move(&expr, &mv);
        assert!(result.is_some(), "AddZero should apply");

        let simplified = result.expect("Expected value but got None/Err");
        assert!(
            matches!(simplified, Expr::Var(0)),
            "x + 0 should simplify to x"
        );
    }

    #[test]
    fn evaluator_prefers_simpler_should_succeed_when_called() {
        use alloc::boxed::Box;

        // Before: x * 1
        let before = Expr::Binary(
            OpType::Mul,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(1.0)),
        );

        // After: x
        let after = Expr::Var(0);

        let hce = default_expr_weights();

        let cost_before = hce.evaluate_position(&before);
        let cost_after = hce.evaluate_position(&after);

        assert!(
            cost_after < cost_before,
            "Simpler expression should have lower cost"
        );
    }
}
