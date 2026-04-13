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

use alloc::sync::Arc;
use alloc::vec::Vec;
use libm::{sqrtf, fabsf};

// Re-export OpKind as OpType for backward compatibility within this module.
// The canonical source of truth is `pixelflow_ir::OpKind`.
use pixelflow_ir::OpKind as OpType;

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
    pub fn depth(&self) -> usize {
        match self {
            Expr::Var(_) | Expr::Const(_) => 1,
            Expr::Unary(_, a) => 1 + a.depth(),
            Expr::Binary(_, a, b) => 1 + a.depth().max(b.depth()),
            Expr::Ternary(_, a, b, c) => 1 + a.depth().max(b.depth()).max(c.depth()),
        }
    }

    /// Count total nodes in the expression.
    pub fn node_count(&self) -> usize {
        match self {
            Expr::Var(_) | Expr::Const(_) => 1,
            Expr::Unary(_, a) => 1 + a.node_count(),
            Expr::Binary(_, a, b) => 1 + a.node_count() + b.node_count(),
            Expr::Ternary(_, a, b, c) => 1 + a.node_count() + b.node_count() + c.node_count(),
        }
    }

    /// Evaluate the expression with given variable values.
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
                    OpType::Min => if a < b { a } else { b },
                    OpType::Max => if a > b { a } else { b },
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
    pub fn to_index(self) -> usize {
        let p = self.perspective_op as usize;
        let d = self.descendant_op as usize;
        let depth = self.depth as usize;
        let path = self.path as usize;

        ((p * OpType::COUNT + d) * MAX_DEPTH + depth) * 256 + path
    }

    /// Create from a unique index.
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
            add_descendant_features(a, features, perspective_op, depth + 1, path << 2);
            add_descendant_features(b, features, perspective_op, depth + 1, (path << 2) | 1);
            add_descendant_features(c, features, perspective_op, depth + 1, (path << 2) | 2);
        }
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
                self.rand_usize(11) // Include all ops
            } else {
                self.rand_usize(10) // Exclude fused ops
            };

            match op_choice {
                // Binary operations
                0 => Expr::Binary(
                    OpType::Add,
                    Arc::new(self.generate_recursive(depth + 1)),
                    Arc::new(self.generate_recursive(depth + 1)),
                ),
                1 => Expr::Binary(
                    OpType::Sub,
                    Arc::new(self.generate_recursive(depth + 1)),
                    Arc::new(self.generate_recursive(depth + 1)),
                ),
                2 => Expr::Binary(
                    OpType::Mul,
                    Arc::new(self.generate_recursive(depth + 1)),
                    Arc::new(self.generate_recursive(depth + 1)),
                ),
                3 => Expr::Binary(
                    OpType::Div,
                    Arc::new(self.generate_recursive(depth + 1)),
                    Arc::new(self.generate_recursive(depth + 1)),
                ),
                4 => Expr::Binary(
                    OpType::Min,
                    Arc::new(self.generate_recursive(depth + 1)),
                    Arc::new(self.generate_recursive(depth + 1)),
                ),
                5 => Expr::Binary(
                    OpType::Max,
                    Arc::new(self.generate_recursive(depth + 1)),
                    Arc::new(self.generate_recursive(depth + 1)),
                ),
                // Unary operations
                6 => Expr::Unary(OpType::Neg, Arc::new(self.generate_recursive(depth + 1))),
                7 => Expr::Unary(OpType::Sqrt, Arc::new(self.generate_recursive(depth + 1))),
                8 => Expr::Unary(OpType::Rsqrt, Arc::new(self.generate_recursive(depth + 1))),
                9 => Expr::Unary(OpType::Abs, Arc::new(self.generate_recursive(depth + 1))),
                // Fused operations
                10 => Expr::Ternary(
                    OpType::MulAdd,
                    Arc::new(self.generate_recursive(depth + 1)),
                    Arc::new(self.generate_recursive(depth + 1)),
                    Arc::new(self.generate_recursive(depth + 1)),
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
                Expr::Binary(OpType::Sub, a, b) if exprs_equal(a, b) => {
                    Some(Expr::Const(0.0))
                }
                _ => None,
            },
            RewriteRule::DivSelf => match expr {
                Expr::Binary(OpType::Div, a, b) if exprs_equal(a, b) => {
                    Some(Expr::Const(1.0))
                }
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
                Expr::Binary(OpType::Add, a, b) if exprs_equal(a, b) => {
                    Some(Expr::Binary(
                        OpType::Mul,
                        Arc::new(Expr::Const(2.0)),
                        a.clone(),
                    ))
                }
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
            RewriteRule::UnfuseMulAdd => match expr {
                Expr::Ternary(OpType::MulAdd, a, b, c) => Some(Expr::Binary(
                    OpType::Add,
                    Arc::new(Expr::Binary(OpType::Mul, a.clone(), b.clone())),
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
pub fn find_all_rewrites(expr: &Expr) -> Vec<(Vec<usize>, RewriteRule, Expr)> {
    let mut rewrites = Vec::new();
    find_rewrites_recursive(expr, &mut Vec::new(), &mut rewrites);
    rewrites
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
    fn test_op_type_roundtrip() {
        for i in 0..OpType::COUNT {
            let op = OpType::from_index(i).unwrap();
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
    fn test_expr_eval() {
        // x + y
        let expr = Expr::Binary(
            OpType::Add,
            Arc::new(Expr::Var(0)),
            Arc::new(Expr::Var(1)),
        );
        let result = expr.eval(&[3.0, 4.0, 0.0, 0.0]);
        assert!(fabsf(result - 7.0) < 1e-6);
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
            OpType::Add,
            Arc::new(Expr::Var(0)),
            Arc::new(Expr::Const(1.0)),
        );
        let features = extract_features(&expr);
        assert!(!features.is_empty());
    }

    #[test]
    fn test_rewrite_add_zero() {
        let expr = Expr::Binary(
            OpType::Add,
            Arc::new(Expr::Var(0)),
            Arc::new(Expr::Const(0.0)),
        );
        let rewritten = RewriteRule::AddZero.try_apply(&expr);
        assert!(rewritten.is_some());
        assert!(matches!(rewritten.unwrap(), Expr::Var(0)));
    }

    #[test]
    fn test_rewrite_fuse_muladd() {
        // a * b + c
        let expr = Expr::Binary(
            OpType::Add,
            Arc::new(Expr::Binary(
                OpType::Mul,
                Arc::new(Expr::Var(0)),
                Arc::new(Expr::Var(1)),
            )),
            Arc::new(Expr::Var(2)),
        );
        let rewritten = RewriteRule::FuseToMulAdd.try_apply(&expr);
        assert!(rewritten.is_some());
        assert!(matches!(rewritten.unwrap(), Expr::Ternary(OpType::MulAdd, _, _, _)));
    }

    #[test]
    fn test_find_all_rewrites() {
        // (x + 0) + y - should find AddZero at path [0]
        let expr = Expr::Binary(
            OpType::Add,
            Arc::new(Expr::Binary(
                OpType::Add,
                Arc::new(Expr::Var(0)),
                Arc::new(Expr::Const(0.0)),
            )),
            Arc::new(Expr::Var(1)),
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
            assert_eq!(val, nnue.b1[i], "Accumulator should return to bias after add/remove");
        }
    }
}
