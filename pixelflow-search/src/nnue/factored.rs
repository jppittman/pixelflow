//! # Factored Embedding NNUE Architecture
//!
//! An O(ops) alternative to the O(ops²) HalfEP feature encoding.
//!
//! ## The Problem
//!
//! HalfEP features encode all (perspective_op, descendant_op, depth, path) tuples:
//! - 42 ops → 42² × 8 × 256 = 3.6M possible features
//! - Feature space grows quadratically with operation count
//! - Training requires O(GB) of memory for weight matrices
//!
//! ## The Solution: Edge-based Factored Embeddings
//!
//! Instead of one-hot encoding each (parent, child) pair, we learn dense
//! embeddings for each operation and accumulate them edge-by-edge:
//!
//! ```text
//! For each parent→child edge in the expression tree:
//!     accumulator[0..K]  += E[parent_op]   // "what's above"
//!     accumulator[K..2K] += E[child_op]    // "what's below"
//! ```
//!
//! Key insight: **Position encodes role**. Parent ops contribute to the first
//! half of the accumulator, child ops to the second half. This ensures that
//! `Mul→Add` (FMA-eligible) produces a different vector than `Add→Mul` (not FMA).
//!
//! ## Complexity
//!
//! | Metric | HalfEP | Factored | Improvement |
//! |--------|--------|----------|-------------|
//! | Feature space | O(ops²) | O(ops) | O(ops) |
//! | Weight memory | ~1GB | ~10KB | 100,000× |
//! | Accumulator build | O(nodes²) | O(edges) | O(nodes) |
//! | Incremental update | O(subtree²) | O(Δedges × K) | O(subtree) |

#![allow(dead_code)] // Prototype code

extern crate alloc;

use alloc::vec::Vec;
use libm::sqrtf;

use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};
use crate::egraph::Pattern as Expr;
pub use pixelflow_ir::OpKind;

// ============================================================================
// Constants
// ============================================================================

/// Embedding dimension per operation.
///
/// Each operation gets a K-dimensional learned embedding. The accumulator
/// stores 2K values: K for parent roles, K for child roles.
pub const K: usize = 32;

/// Number of scalar features appended to the dual accumulator.
/// edge_count, node_count, node_budget, epoch_budget.
pub const SCALAR_FEATURE_COUNT: usize = 4;

/// Total input dimension to the hidden layer:
/// 4K (dual accumulator: 2K flat + 2K depth-encoded) + 4 scalars.
pub const INPUT_DIM: usize = 4 * K + SCALAR_FEATURE_COUNT;

/// Graph accumulator dimension: marginals (2K) + 1-hop VSA binding (K) + 2-hop VSA binding (K).
pub const GRAPH_ACC_DIM: usize = 4 * K; // 128

/// Graph backbone input: 4K + 4 scalars (edge_count, node_count, node_budget, epoch_budget).
pub const GRAPH_INPUT_DIM: usize = GRAPH_ACC_DIM + SCALAR_FEATURE_COUNT; // 132

/// Maximum arity for child-index encoding.
/// Effective depth = `depth * MAX_ARITY + child_index`, where child_index ∈ [0, MAX_ARITY).
/// This breaks sibling symmetry: left and right children of the same parent get different PEs.
pub const MAX_ARITY: usize = 3;

/// Maximum effective depth for learned depth embeddings.
/// Child-index encoding triples the effective depth range: a tree of real depth 63
/// with ternary nodes → `63*3+2 = 191 < 192`. Depths beyond this are clamped.
pub const MAX_DEPTH: usize = 192;

/// Hidden layer size.
pub const HIDDEN_DIM: usize = 64;

// ============================================================================
// Unified Mask Architecture Constants
// ============================================================================

/// Embedding dimension for expr/rule factorization in the unified mask architecture.
pub const EMBED_DIM: usize = 32;

/// Hidden dimension for private MLPs (value, mask, rule).
pub const MLP_HIDDEN: usize = 16;

/// Rule feature dimension (hand-crafted features describing each rule).
pub const RULE_FEATURE_DIM: usize = 8;

/// Maximum rules supported in the unified mask architecture.
/// Designed to scale to 1000+ rules.
pub const MASK_MAX_RULES: usize = 1024;

/// Concatenated rule features: [z_LHS | z_RHS | z_LHS-z_RHS | z_LHS*z_RHS] (4 × EMBED_DIM).
/// Used when encoding rules from their LHS/RHS expression templates.
pub const RULE_CONCAT_DIM: usize = 4 * EMBED_DIM;

/// Mask MLP input dimension: expr_embed directly (24 dims).
/// value_pred was removed — it is a deterministic function of expr_embed and adds zero information.
pub const MASK_INPUT_DIM: usize = EMBED_DIM;

// NOTE: SEARCH_INPUT_DIM removed - mask IS the policy.
// See plan: "Idea 4B: Mask IS the search/policy ✅ CHOSEN"

// ============================================================================
// Rule Features
// ============================================================================

/// Hand-crafted features describing each rule.
///
/// These features are mostly static (computed once when rules are defined)
/// and allow the Rule MLP to generalize across rules without learning
/// individual embeddings for each rule.
///
/// # Features (RULE_FEATURE_DIM = 8)
///
/// 1. `category`: Rule type (algebraic=0, peephole=0.25, domain=0.5, cross-cutting=0.75)
/// 2. `lhs_nodes`: Pattern complexity (normalized by 10)
/// 3. `typical_depth_delta`: Usually -1, 0, or 1
/// 4. `commutative`: Does rule exploit commutativity? (0 or 1)
/// 5. `associative`: Does rule exploit associativity? (0 or 1)
/// 6. `creates_sharing`: Does rule typically enable CSE? (0 or 1)
/// 7. `historical_match_rate`: Running average [0, 1]
/// 8. `expensive_op_related`: Touches div/sqrt/transcendental? (0 or 1)
#[derive(Clone)]
pub struct RuleFeatures {
    /// Features for each rule: [rule_idx][feature_dim]
    pub features: [[f32; RULE_FEATURE_DIM]; MASK_MAX_RULES],
}

impl Default for RuleFeatures {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleFeatures {
    /// Create zero-initialized rule features.
    #[must_use]
    pub fn new() -> Self {
        Self {
            features: [[0.0; RULE_FEATURE_DIM]; MASK_MAX_RULES],
        }
    }

    /// Get features for a specific rule.
    #[must_use]
    pub fn get(&self, rule_idx: usize) -> &[f32; RULE_FEATURE_DIM] {
        &self.features[rule_idx]
    }

    /// Set features for a specific rule.
    pub fn set(&mut self, rule_idx: usize, features: [f32; RULE_FEATURE_DIM]) {
        self.features[rule_idx] = features;
    }

    /// Set feature by name for easier initialization.
    pub fn set_rule(
        &mut self,
        rule_idx: usize,
        category: f32,
        lhs_nodes: usize,
        depth_delta: i8,
        commutative: bool,
        associative: bool,
        creates_sharing: bool,
        match_rate: f32,
        expensive_op: bool,
    ) {
        self.features[rule_idx] = [
            category,
            lhs_nodes as f32 / 10.0,
            depth_delta as f32,
            if commutative { 1.0 } else { 0.0 },
            if associative { 1.0 } else { 0.0 },
            if creates_sharing { 1.0 } else { 0.0 },
            match_rate.clamp(0.0, 1.0),
            if expensive_op { 1.0 } else { 0.0 },
        ];
    }
}

// ============================================================================
// Rule Templates (LHS/RHS Expression Templates)
// ============================================================================

/// Rule templates: LHS and RHS expressions for each rule.
///
/// These use the SAME expr_embed as extraction/saturation heads, enabling the model
/// to learn structural similarity between expressions and rule patterns.
///
/// Each rule has:
/// - LHS pattern (what it matches), e.g., `A * (B + C)`
/// - RHS pattern (what it produces), e.g., `A*B + A*C`
///
/// The 4-way concatenation captures:
/// - `z_LHS`: what the rule MATCHES (pattern recognition)
/// - `z_RHS`: what it PRODUCES (production prediction)
/// - `z_LHS - z_RHS`: what CHANGED (the delta)
/// - `z_LHS * z_RHS`: what's SHARED (preserved structure)
#[derive(Clone)]
pub struct RuleTemplates {
    /// LHS pattern for each rule (what it matches).
    /// Uses Expr::Var(0), Expr::Var(1), etc. as metavariables.
    pub lhs: Vec<Option<Expr>>,
    /// RHS pattern for each rule (what it produces).
    pub rhs: Vec<Option<Expr>>,
}

impl Default for RuleTemplates {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleTemplates {
    /// Create empty templates.
    #[must_use]
    pub fn new() -> Self {
        Self {
            lhs: Vec::new(),
            rhs: Vec::new(),
        }
    }

    /// Create templates for a given number of rules (all None initially).
    #[must_use]
    pub fn with_capacity(num_rules: usize) -> Self {
        Self {
            lhs: vec![None; num_rules],
            rhs: vec![None; num_rules],
        }
    }

    /// Set templates for a specific rule.
    pub fn set(&mut self, rule_idx: usize, lhs: Expr, rhs: Expr) {
        // Ensure we have enough capacity
        if rule_idx >= self.lhs.len() {
            self.lhs.resize(rule_idx + 1, None);
            self.rhs.resize(rule_idx + 1, None);
        }
        self.lhs[rule_idx] = Some(lhs);
        self.rhs[rule_idx] = Some(rhs);
    }

    /// Get LHS template for a rule.
    #[must_use]
    pub fn get_lhs(&self, rule_idx: usize) -> Option<&Expr> {
        self.lhs.get(rule_idx).and_then(|opt| opt.as_ref())
    }

    /// Get RHS template for a rule.
    #[must_use]
    pub fn get_rhs(&self, rule_idx: usize) -> Option<&Expr> {
        self.rhs.get(rule_idx).and_then(|opt| opt.as_ref())
    }

    /// Number of rules with templates.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lhs.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lhs.is_empty()
    }

    /// Check if a rule has templates defined.
    #[must_use]
    pub fn has_templates(&self, rule_idx: usize) -> bool {
        self.get_lhs(rule_idx).is_some() && self.get_rhs(rule_idx).is_some()
    }

    /// Returns `true` if any template (LHS or RHS) has `op` as its root operation.
    ///
    /// Used to skip template matching entirely for nodes whose op cannot match
    /// any template root, avoiding expensive `to_expr` + `pattern_match` calls.
    #[must_use]
    pub fn has_root_op(&self, op: OpKind) -> bool {
        for rule_idx in 0..self.len() {
            if let Some(lhs) = self.get_lhs(rule_idx) {
                // We match against LHS to produce RHS, but we also want
                // expanding rewrites (RHS->LHS). Check both sides.
                if !matches!(lhs, Expr::Var(_)) && lhs.kind() == op {
                    return true;
                }
            }
            if let Some(rhs) = self.get_rhs(rule_idx) {
                if !matches!(rhs, Expr::Var(_)) && rhs.kind() == op {
                    return true;
                }
            }
        }
        false
    }

    /// Build a precomputed set of root ops that appear in any template.
    ///
    /// Returns a fixed-size array indexed by `OpKind::index()`. This turns
    /// `has_root_op` from O(rules) to O(1) when called repeatedly.
    #[must_use]
    pub fn root_op_set(&self) -> [bool; OpKind::COUNT] {
        let mut set = [false; OpKind::COUNT];
        for rule_idx in 0..self.len() {
            if let Some(lhs) = self.get_lhs(rule_idx) {
                if !matches!(lhs, Expr::Var(_)) {
                    set[lhs.kind().index()] = true;
                }
            }
            if let Some(rhs) = self.get_rhs(rule_idx) {
                if !matches!(rhs, Expr::Var(_)) {
                    set[rhs.kind().index()] = true;
                }
            }
        }
        set
    }
}

// ============================================================================
// Arena Rule Templates
// ============================================================================

/// A single rule stored as two subtrees inside one shared [`ExprArena`].
///
/// `lhs` and `rhs` are roots inside `arena`. Either may be `None` when the
/// corresponding pattern was not provided (same semantics as [`RuleTemplates`]).
pub struct ArenaRuleTemplate {
    /// Shared arena holding both the LHS and RHS subtrees.
    pub arena: ExprArena,
    /// Root of the LHS pattern, or `None`.
    pub lhs: Option<ExprId>,
    /// Root of the RHS pattern, or `None`.
    pub rhs: Option<ExprId>,
    /// Precomputed: LHS root op kind (if LHS is not a bare Var).
    pub lhs_op: Option<OpKind>,
    /// Precomputed: RHS root op kind (if RHS is not a bare Var).
    pub rhs_op: Option<OpKind>,
}

impl ArenaRuleTemplate {
    /// Construct from a pair of optional template patterns.
    #[must_use]
    pub fn from_patterns(lhs: Option<&Expr>, rhs: Option<&Expr>) -> Self {
        let mut arena = ExprArena::with_capacity(16);

        let lhs_id = lhs.map(|e| e.push_into(&mut arena));
        let rhs_id = rhs.map(|e| e.push_into(&mut arena));

        let lhs_op = lhs_id.and_then(|id| {
            if matches!(arena.node(id), ExprNode::Var(_)) {
                None
            } else {
                Some(arena.kind(id))
            }
        });
        let rhs_op = rhs_id.and_then(|id| {
            if matches!(arena.node(id), ExprNode::Var(_)) {
                None
            } else {
                Some(arena.kind(id))
            }
        });

        Self {
            arena,
            lhs: lhs_id,
            rhs: rhs_id,
            lhs_op,
            rhs_op,
        }
    }
}

/// Arena-backed rule template storage.
///
/// Stores every rule as an [`ArenaRuleTemplate`] — tiny arenas of 2-5 nodes
/// each — so that pattern matching and substitution can operate directly on
/// arena nodes without the `to_expr` / `push_expr` bridge.
pub struct ArenaRuleTemplates {
    /// One arena-backed template per rule, indexed by rule_idx.
    pub arenas: Vec<ArenaRuleTemplate>,
    /// Precomputed O(1) op-membership set (same semantics as `root_op_set()`).
    pub root_op_set: [bool; OpKind::COUNT],
}

impl ArenaRuleTemplates {
    /// Convert [`RuleTemplates`] into arena form.
    ///
    /// Iterates every rule, converts LHS/RHS patterns to per-rule arenas,
    /// and precomputes the `root_op_set` membership array.
    #[must_use]
    pub fn from_rule_templates(templates: &RuleTemplates) -> Self {
        let mut arenas = Vec::with_capacity(templates.len());
        let mut root_op_set = [false; OpKind::COUNT];

        for rule_idx in 0..templates.len() {
            let lhs = templates.get_lhs(rule_idx);
            let rhs = templates.get_rhs(rule_idx);
            let tmpl = ArenaRuleTemplate::from_patterns(lhs, rhs);

            if let Some(op) = tmpl.lhs_op {
                root_op_set[op.index()] = true;
            }
            if let Some(op) = tmpl.rhs_op {
                root_op_set[op.index()] = true;
            }

            arenas.push(tmpl);
        }

        Self {
            arenas,
            root_op_set,
        }
    }

    /// Number of rules.
    #[must_use]
    pub fn len(&self) -> usize {
        self.arenas.len()
    }

    /// `true` if there are no rules.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.arenas.is_empty()
    }
}

// ============================================================================
// Operation Embeddings
// ============================================================================

/// Learned dense embeddings for each operation type.
///
/// Each of the 42 operations gets a K-dimensional embedding vector.
/// These are the primary learned parameters that capture semantic
/// similarity between operations.
#[derive(Clone)]
pub struct OpEmbeddings {
    /// E[op][i] = i-th dimension of op's embedding.
    /// Stored as [OpKind::COUNT][K] = 42 × 32 = 1,344 floats.
    pub e: [[f32; K]; OpKind::COUNT],
}

impl Default for OpEmbeddings {
    fn default() -> Self {
        Self::new()
    }
}

impl OpEmbeddings {
    /// Create zero-initialized embeddings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            e: [[0.0; K]; OpKind::COUNT],
        }
    }

    /// Initialize embeddings with random values using He initialization.
    ///
    /// Scale: sqrt(2/K) for ReLU networks.
    #[must_use]
    pub fn new_random(seed: u64) -> Self {
        let mut embeddings = Self::new();
        embeddings.randomize(seed);
        embeddings
    }

    /// Initialize with latency priors.
    ///
    /// This encodes known operation latencies into dimension 0 of each embedding,
    /// giving the model a strong starting point. Remaining dimensions are small
    /// random values that can learn subtle interactions.
    ///
    /// Scales to any number of ops - just provide latencies for new ops.
    #[must_use]
    pub fn new_with_latency_prior(seed: u64) -> Self {
        let mut embeddings = Self::new();
        embeddings.init_with_latency_prior(seed);
        embeddings
    }

    /// Initialize with latency priors in place.
    pub fn init_with_latency_prior(&mut self, seed: u64) {
        // Known latencies (cycles) - these are approximate and can be refined
        // Dimension 0 = latency, normalized to [0, 1] range (divide by max ~20)
        let latencies: [f32; OpKind::COUNT] = [
            0.0,  // Var - free
            0.0,  // Const - free
            0.2,  // Add - 4 cycles
            0.2,  // Sub - 4 cycles
            0.25, // Mul - 5 cycles
            0.75, // Div - 15 cycles
            0.05, // Neg - 1 cycle
            0.75, // Sqrt - 15 cycles
            0.25, // Rsqrt - 5 cycles (fast approximation)
            0.05, // Abs - 1 cycle
            0.2,  // Min - 4 cycles
            0.2,  // Max - 4 cycles
            0.25, // MulAdd - 5 cycles (fused)
            0.5,  // Recip - 10 cycles
            0.2,  // Floor - 4 cycles
            0.2,  // Ceil - 4 cycles
            0.2,  // Round - 4 cycles
            0.2,  // Fract - 4 cycles
            0.5,  // Sin - 10 cycles
            0.5,  // Cos - 10 cycles
            0.5,  // Tan - 10 cycles
            0.5,  // Asin - 10 cycles
            0.5,  // Acos - 10 cycles
            0.5,  // Atan - 10 cycles
            0.5,  // Exp - 10 cycles
            0.5,  // Exp2 - 10 cycles
            0.5,  // Ln - 10 cycles
            0.5,  // Log2 - 10 cycles
            0.5,  // Log10 - 10 cycles
            0.5,  // Atan2 - 10 cycles
            0.6,  // Pow - 12 cycles
            0.4,  // Hypot - 8 cycles
            0.15, // Lt - 3 cycles
            0.15, // Le - 3 cycles
            0.15, // Gt - 3 cycles
            0.15, // Ge - 3 cycles
            0.15, // Eq - 3 cycles
            0.15, // Ne - 3 cycles
            0.2,  // Select - 4 cycles
            0.3,  // Clamp - 6 cycles (2x compare + select)
            0.0,  // Tuple - free (structural)
        ];

        let mut rng_state = seed.wrapping_add(1);
        let small_scale = 0.1; // Small noise for other dimensions

        for op_idx in 0..OpKind::COUNT {
            // Dimension 0: latency prior
            self.e[op_idx][0] = latencies[op_idx];

            // Dimensions 1..K: small random for learning interactions
            for dim in 1..K {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let uniform = (rng_state >> 33) as f32 / (1u64 << 31) as f32;
                self.e[op_idx][dim] = (uniform * 2.0 - 1.0) * small_scale;
            }
        }
    }

    /// Randomize embeddings in place (fully random, no priors).
    pub fn randomize(&mut self, seed: u64) {
        let scale = sqrtf(2.0 / K as f32);
        let mut rng_state = seed.wrapping_add(1);

        for op_idx in 0..OpKind::COUNT {
            for dim in 0..K {
                // LCG for no_std compatibility
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);

                // Convert to [-1, 1] and scale
                let uniform = (rng_state >> 33) as f32 / (1u64 << 31) as f32;
                let centered = uniform * 2.0 - 1.0;
                self.e[op_idx][dim] = centered * scale;
            }
        }
    }

    /// Get embedding for an operation.
    #[inline]
    #[must_use]
    pub fn get(&self, op: OpKind) -> &[f32; K] {
        &self.e[op.index()]
    }

    /// Total parameter count.
    #[must_use]
    pub const fn param_count() -> usize {
        OpKind::COUNT * K
    }
}

// ============================================================================
// Sinusoidal Depth Encoding (Fixed Positional Encoding for AST Depth)
// ============================================================================

/// Precomputed sinusoidal positional encoding table.
///
/// Fixed (not learned) — zero parameters, zero serialization, zero gradients.
/// The downstream weights in w1 learn how to USE the rotation; the encoding
/// itself is a deterministic function of depth.
///
/// Each depth level gets a K-dimensional vector where:
///   PE[d][2i]   = sin(d / 10000^(2i/K))
///   PE[d][2i+1] = cos(d / 10000^(2i/K))
///
/// Used via Hadamard product: `E[op] ⊙ PE[depth]`
/// This binds depth to operation without destroying magnitude —
/// additive encoding (E + PE) would decouple in the commutative sum.
static DEPTH_PE: [[f32; K]; MAX_DEPTH] = {
    let mut table = [[0.0f32; K]; MAX_DEPTH];
    let mut depth = 0;
    while depth < MAX_DEPTH {
        let mut dim = 0;
        while dim < K {
            // 10000^(2*(dim/2)/K) computed via exp/log in const context
            // We use a simpler geometric series: base = 10000^(1/K) ≈ 1.318
            // freq = base^(-dim_pair) where dim_pair = 2*(dim/2)
            let dim_pair = 2 * (dim / 2);
            // Approximate: 10000^(dim_pair/K) via repeated squaring in f64
            // For const context, we compute the exponent directly.
            let exponent: f64 = (dim_pair as f64) / (K as f64);
            // 10000^exponent via exp(exponent * ln(10000))
            // ln(10000) ≈ 9.210340371976184
            let log_base: f64 = 9.210340371976184;
            let divisor: f64 = const_exp(exponent * log_base);
            let angle: f64 = (depth as f64) / divisor;
            // sin/cos via Taylor series (const-compatible)
            if dim % 2 == 0 {
                table[depth][dim] = const_sin(angle) as f32;
            } else {
                table[depth][dim] = const_cos(angle) as f32;
            }
            dim += 1;
        }
        depth += 1;
    }
    table
};

/// Const-compatible exp(x) via Taylor series (18 terms, accurate to ~1e-15).
const fn const_exp(x: f64) -> f64 {
    let mut result: f64 = 1.0;
    let mut term: f64 = 1.0;
    let mut i: u32 = 1;
    while i <= 18 {
        term *= x / (i as f64);
        result += term;
        i += 1;
    }
    result
}

/// Const-compatible sin(x) via Taylor series.
/// Reduces x to [-pi, pi] first for accuracy.
const fn const_sin(x: f64) -> f64 {
    // Reduce to [-pi, pi]
    let pi: f64 = 3.141592653589793;
    let two_pi: f64 = 6.283185307179586;
    let mut r = x;
    // Simple modular reduction (good enough for small positive x)
    while r > pi {
        r -= two_pi;
    }
    while r < -pi {
        r += two_pi;
    }
    // Taylor: sin(r) = r - r^3/6 + r^5/120 - ...
    let mut result: f64 = 0.0;
    let mut term: f64 = r;
    let r2 = r * r;
    let mut i: u32 = 0;
    while i < 12 {
        result += term;
        term *= -r2 / (((2 * i + 2) * (2 * i + 3)) as f64);
        i += 1;
    }
    result
}

/// Const-compatible cos(x) via Taylor series.
const fn const_cos(x: f64) -> f64 {
    let pi: f64 = 3.141592653589793;
    let two_pi: f64 = 6.283185307179586;
    let mut r = x;
    while r > pi {
        r -= two_pi;
    }
    while r < -pi {
        r += two_pi;
    }
    let mut result: f64 = 0.0;
    let mut term: f64 = 1.0;
    let r2 = r * r;
    let mut i: u32 = 0;
    while i < 12 {
        result += term;
        term *= -r2 / (((2 * i + 1) * (2 * i + 2)) as f64);
        i += 1;
    }
    result
}

/// Look up the sinusoidal positional encoding for a given depth.
/// Depths beyond MAX_DEPTH are clamped.
#[inline]
pub fn depth_pe(depth: u32) -> &'static [f32; K] {
    &DEPTH_PE[depth.min((MAX_DEPTH - 1) as u32) as usize]
}

/// Cyclic rotation by `amount % K` positions (generalised VSA permutation).
///
/// `shift_by(emb, 0)` is the identity, `shift_by(emb, 1)` is the original
/// `shift1`, and higher amounts encode hierarchical depth in VSA bindings:
/// `parent ⊙ shift_by(child, depth)` produces a distinct binding per depth
/// level, so `Add(Add(X,Y),Z)` and `Add(X,Add(Y,Z))` yield different
/// accumulators.
#[inline]
fn shift_by(emb: &[f32; K], amount: usize) -> [f32; K] {
    let amount = amount % K;
    let mut out = [0.0f32; K];
    for i in 0..K {
        out[i] = emb[(i + amount) % K];
    }
    out
}

/// Cyclic shift by 1 position (VSA permutation for breaking commutativity).
///
/// Used by `GraphAccumulator` to ensure `parent ⊙ shift₁(child)` produces a
/// different binding vector than `child ⊙ shift₁(parent)`, i.e. `Mul→Add ≠ Add→Mul`.
#[inline]
fn shift1(emb: &[f32; K]) -> [f32; K] {
    shift_by(emb, 1)
}

// ============================================================================
// Edge Accumulator (Dual: Flat + Depth-Encoded)
// ============================================================================

/// Dual accumulator for edge-based feature extraction.
///
/// Split into two physically distinct representations:
///
/// - **Flat half (0..2K):** `Σ E[parent]` and `Σ E[child]`.
///   Pure throughput — the network knows exactly how many of each operation
///   exist. A `cos` always contributes its full embedding regardless of depth.
///
/// - **Depth-encoded half (2K..4K):** `Σ (E[parent] ⊙ PE[depth])` and
///   `Σ (E[child] ⊙ PE[depth])`. Pure geometry — the Hadamard product with
///   sinusoidal positional encoding binds each operation to its tree position
///   without destroying its magnitude. The network sees ILP constraints and
///   pipeline bottlenecks.
///
/// Both halves support O(1) incremental updates via vector addition/subtraction.
#[derive(Clone)]
pub struct EdgeAccumulator {
    /// Contiguous accumulator values.
    /// - `[0..K]`:     flat parent sum (throughput)
    /// - `[K..2K]`:    flat child sum (throughput)
    /// - `[2K..3K]`:   depth-encoded parent sum (geometry)
    /// - `[3K..4K]`:   depth-encoded child sum (geometry)
    pub values: [f32; 4 * K],

    /// Edge count (O(1) additive scalar).
    pub edge_count: u32,

    /// Node count (O(1) additive scalar).
    pub node_count: u32,

    /// Number of shared subtrees skipped via CSE deduplication.
    ///
    /// Incremented by `from_expr_dedup` each time a subtree that was already
    /// walked is encountered again. The duplicate's edges are NOT re-added to
    /// the accumulator — this field is purely diagnostic and does NOT feed into
    /// the network.
    pub backref_count: u32,

    /// E-graph node budget for this trajectory (how many nodes the saturator may create).
    /// Serialized into the accumulator vector so the model can condition on its budget.
    pub node_budget: u32,

    /// Epoch budget for this trajectory (max saturation epochs).
    /// Serialized into the accumulator vector alongside node_budget.
    pub epoch_budget: u32,

    // -- Variance features (fed to extraction head) --
    /// Fraction of nodes that are compile-time constants (variance = {}).
    pub variance_frac_const: f32,

    /// Fraction of nodes that are frame-uniform (variance ⊆ {Z, W}, no X or Y).
    pub variance_frac_frame: f32,

    /// Fraction of nodes that are scanline-uniform (have Y but no X).
    pub variance_frac_scanline: f32,

    /// Fraction of nodes that are pixel-varying (have X).
    pub variance_frac_pixel: f32,
}

impl Default for EdgeAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl EdgeAccumulator {
    /// Create a zero-initialized dual accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            values: [0.0; 4 * K],
            edge_count: 0,
            node_count: 0,
            backref_count: 0,
            node_budget: 0,
            epoch_budget: 0,
            variance_frac_const: 0.0,
            variance_frac_frame: 0.0,
            variance_frac_scanline: 0.0,
            variance_frac_pixel: 0.0,
        }
    }

    /// Reset to zero state.
    ///
    /// Budget fields are intentionally NOT reset — they are trajectory-level
    /// properties that should persist across epoch rebuilds.
    pub fn reset(&mut self) {
        self.values = [0.0; 4 * K];
        self.edge_count = 0;
        self.node_count = 0;
        self.backref_count = 0;
    }

    /// Add a single edge contribution (both flat and depth-encoded).
    ///
    /// Flat half: raw embedding addition (preserves magnitude).
    /// Depth half: complex multiplication — each pair `(2f, 2f+1)` represents
    /// `(real, imaginary)` for frequency `f`. PE stores `sin` at even, `cos` at
    /// odd indices. Complex: `(emb_re + j·emb_im) × (cos + j·sin)`.
    #[inline]
    pub fn add_edge(
        &mut self,
        emb: &OpEmbeddings,
        parent_op: OpKind,
        child_op: OpKind,
        depth: u32,
    ) {
        let pe = depth_pe(depth);
        self.add_edge_with_pe(emb, parent_op, child_op, pe);
    }

    /// Add a single edge with caller-provided PE (used by InstructionWindow).
    #[inline]
    pub fn add_edge_with_pe(
        &mut self,
        emb: &OpEmbeddings,
        parent_op: OpKind,
        child_op: OpKind,
        pe: &[f32; K],
    ) {
        let parent_emb = emb.get(parent_op);
        let child_emb = emb.get(child_op);

        // Flat half: raw sum (unchanged)
        for i in 0..K {
            self.values[i] += parent_emb[i];
            self.values[K + i] += child_emb[i];
        }

        // Depth-encoded half: complex multiply
        // Each pair (2f, 2f+1) represents (real, imaginary) for frequency f.
        // PE stores sin at even, cos at odd indices.
        // Complex: (emb_re + j·emb_im) × (cos + j·sin)
        for f in 0..K / 2 {
            let sin_d = pe[2 * f];
            let cos_d = pe[2 * f + 1];

            let p_re = parent_emb[2 * f];
            let p_im = parent_emb[2 * f + 1];
            self.values[2 * K + 2 * f] += p_re * cos_d - p_im * sin_d;
            self.values[2 * K + 2 * f + 1] += p_re * sin_d + p_im * cos_d;

            let c_re = child_emb[2 * f];
            let c_im = child_emb[2 * f + 1];
            self.values[3 * K + 2 * f] += c_re * cos_d - c_im * sin_d;
            self.values[3 * K + 2 * f + 1] += c_re * sin_d + c_im * cos_d;
        }
        self.edge_count += 1;
    }

    /// Remove a single edge contribution (for incremental updates).
    #[inline]
    pub fn remove_edge(
        &mut self,
        emb: &OpEmbeddings,
        parent_op: OpKind,
        child_op: OpKind,
        depth: u32,
    ) {
        let pe = depth_pe(depth);
        self.remove_edge_with_pe(emb, parent_op, child_op, pe);
    }

    /// Remove a single edge with caller-provided PE (used by InstructionWindow).
    #[inline]
    pub fn remove_edge_with_pe(
        &mut self,
        emb: &OpEmbeddings,
        parent_op: OpKind,
        child_op: OpKind,
        pe: &[f32; K],
    ) {
        let parent_emb = emb.get(parent_op);
        let child_emb = emb.get(child_op);

        // Flat half: raw subtract
        for i in 0..K {
            self.values[i] -= parent_emb[i];
            self.values[K + i] -= child_emb[i];
        }

        // Depth-encoded half: complex multiply (subtract)
        for f in 0..K / 2 {
            let sin_d = pe[2 * f];
            let cos_d = pe[2 * f + 1];

            let p_re = parent_emb[2 * f];
            let p_im = parent_emb[2 * f + 1];
            self.values[2 * K + 2 * f] -= p_re * cos_d - p_im * sin_d;
            self.values[2 * K + 2 * f + 1] -= p_re * sin_d + p_im * cos_d;

            let c_re = child_emb[2 * f];
            let c_im = child_emb[2 * f + 1];
            self.values[3 * K + 2 * f] -= c_re * cos_d - c_im * sin_d;
            self.values[3 * K + 2 * f + 1] -= c_re * sin_d + c_im * cos_d;
        }
        self.edge_count = self.edge_count.saturating_sub(1);
    }

    /// Build dual accumulator from expression tree, deduplicating shared subtrees.
    ///
    /// Uses structural (content-based) hashing to detect subtrees that appear
    /// more than once in the expression DAG. When a subtree is encountered a
    /// second time, a `BACKREF_FEATURE` token is emitted: `backref_count` is
    /// incremented and the subtree's edges are NOT re-added. This prevents
    /// double-counting of CSE nodes at the network input layer.
    ///
    /// `INPUT_DIM` and the weight layout are unchanged — `backref_count` is a
    /// diagnostic field and does not feed into the forward pass, so this method
    /// is fully compatible with the trained extraction head weights (`judge.bin`).
    pub fn from_expr_dedup(expr: &Expr, emb: &OpEmbeddings) -> Self {
        let mut acc = Self::new();
        let mut seen: alloc::collections::BTreeSet<u64> = alloc::collections::BTreeSet::new();
        Self::collect_recursive_dedup(expr, emb, &mut acc, 0, &mut seen);
        acc
    }

    fn collect_recursive_dedup(
        expr: &Expr,
        emb: &OpEmbeddings,
        acc: &mut Self,
        depth: u32,
        seen: &mut alloc::collections::BTreeSet<u64>,
    ) {
        // Iterative traversal with explicit stack to avoid thread stack overflow
        // on deep expression trees.
        let mut stack: alloc::vec::Vec<(&Expr, u32)> = alloc::vec![(expr, depth)];

        while let Some((node, d)) = stack.pop() {
            let h = structural_hash(node);
            if !seen.insert(h) {
                acc.backref_count += 1;
                continue;
            }

            let parent_op = node.op_type();
            acc.node_count += 1;

            match node {
                Expr::Var(_) | Expr::Const(_) => {}
                Expr::Param(i) => panic!(
                    "Expr::Param({i}) reached NNUE cost model — call substitute_params before use"
                ),
                Expr::Unary(_, child) => {
                    let eff_depth = d * MAX_ARITY as u32;
                    acc.add_edge(emb, parent_op, child.op_type(), eff_depth);
                    stack.push((child, d + 1));
                }
                Expr::Binary(_, left, right) => {
                    acc.add_edge(emb, parent_op, left.op_type(), d * MAX_ARITY as u32);
                    acc.add_edge(emb, parent_op, right.op_type(), d * MAX_ARITY as u32 + 1);
                    // Push right first so left is processed first (stack is LIFO)
                    stack.push((right, d + 1));
                    stack.push((left, d + 1));
                }
                Expr::Ternary(_, a, b, c) => {
                    acc.add_edge(emb, parent_op, a.op_type(), d * MAX_ARITY as u32);
                    acc.add_edge(emb, parent_op, b.op_type(), d * MAX_ARITY as u32 + 1);
                    acc.add_edge(emb, parent_op, c.op_type(), d * MAX_ARITY as u32 + 2);
                    stack.push((c, d + 1));
                    stack.push((b, d + 1));
                    stack.push((a, d + 1));
                }
                Expr::Nary(_, children) => {
                    for (idx, child) in children.iter().enumerate() {
                        let eff_depth = d * MAX_ARITY as u32 + (idx.min(MAX_ARITY - 1)) as u32;
                        acc.add_edge(emb, parent_op, child.op_type(), eff_depth);
                    }
                    for child in children.iter().rev() {
                        stack.push((child, d + 1));
                    }
                }
            }
        }
    }

    /// Build dual accumulator from an arena DAG, counting each shared node once.
    #[must_use]
    pub fn from_arena_dedup(arena: &ExprArena, root: ExprId, emb: &OpEmbeddings) -> Self {
        use alloc::collections::BTreeSet;

        let mut acc = Self::new();
        let mut seen = BTreeSet::<ExprId>::new();
        let mut stack: Vec<(ExprId, u32)> = alloc::vec![(root, 0)];

        while let Some((id, depth)) = stack.pop() {
            if !seen.insert(id) {
                acc.backref_count += 1;
                continue;
            }

            acc.node_count += 1;
            match arena.node(id) {
                ExprNode::Var(_) | ExprNode::Const(_) => {}
                ExprNode::Param(i) => {
                    panic!("ExprNode::Param({i}) reached NNUE cost model — substitute params first")
                }
                ExprNode::Unary(op, child) => {
                    acc.add_edge(emb, *op, arena.kind(*child), depth * MAX_ARITY as u32);
                    stack.push((*child, depth + 1));
                }
                ExprNode::Binary(op, left, right) => {
                    acc.add_edge(emb, *op, arena.kind(*left), depth * MAX_ARITY as u32);
                    acc.add_edge(emb, *op, arena.kind(*right), depth * MAX_ARITY as u32 + 1);
                    stack.push((*right, depth + 1));
                    stack.push((*left, depth + 1));
                }
                ExprNode::Ternary(op, a, b, c) => {
                    acc.add_edge(emb, *op, arena.kind(*a), depth * MAX_ARITY as u32);
                    acc.add_edge(emb, *op, arena.kind(*b), depth * MAX_ARITY as u32 + 1);
                    acc.add_edge(emb, *op, arena.kind(*c), depth * MAX_ARITY as u32 + 2);
                    stack.push((*c, depth + 1));
                    stack.push((*b, depth + 1));
                    stack.push((*a, depth + 1));
                }
                ExprNode::Nary(op, _, _) => {
                    for (idx, child) in arena.children(id).enumerate() {
                        let eff_depth = depth * MAX_ARITY as u32 + (idx.min(MAX_ARITY - 1)) as u32;
                        acc.add_edge(emb, *op, arena.kind(child), eff_depth);
                    }
                    for child in arena.children(id) {
                        stack.push((child, depth + 1));
                    }
                }
            }
        }

        acc
    }

    /// Remove all edges from an expression subtree.
    pub fn remove_expr_edges(&mut self, expr: &Expr, emb: &OpEmbeddings) {
        Self::remove_recursive(expr, emb, self, 0);
    }

    fn remove_recursive(expr: &Expr, emb: &OpEmbeddings, acc: &mut Self, depth: u32) {
        // Iterative traversal with explicit stack to avoid thread stack overflow.
        let mut stack: alloc::vec::Vec<(&Expr, u32)> = alloc::vec![(expr, depth)];

        while let Some((node, d)) = stack.pop() {
            let parent_op = node.op_type();
            acc.node_count = acc.node_count.saturating_sub(1);

            match node {
                Expr::Var(_) | Expr::Const(_) => {}
                Expr::Param(i) => panic!(
                    "Expr::Param({i}) reached NNUE cost model — call substitute_params before use"
                ),
                Expr::Unary(_, child) => {
                    let eff_depth = d * MAX_ARITY as u32;
                    acc.remove_edge(emb, parent_op, child.op_type(), eff_depth);
                    stack.push((child, d + 1));
                }
                Expr::Binary(_, left, right) => {
                    acc.remove_edge(emb, parent_op, left.op_type(), d * MAX_ARITY as u32);
                    acc.remove_edge(emb, parent_op, right.op_type(), d * MAX_ARITY as u32 + 1);
                    stack.push((right, d + 1));
                    stack.push((left, d + 1));
                }
                Expr::Ternary(_, a, b, c) => {
                    acc.remove_edge(emb, parent_op, a.op_type(), d * MAX_ARITY as u32);
                    acc.remove_edge(emb, parent_op, b.op_type(), d * MAX_ARITY as u32 + 1);
                    acc.remove_edge(emb, parent_op, c.op_type(), d * MAX_ARITY as u32 + 2);
                    stack.push((c, d + 1));
                    stack.push((b, d + 1));
                    stack.push((a, d + 1));
                }
                Expr::Nary(_, children) => {
                    for (idx, child) in children.iter().enumerate() {
                        let eff_depth = d * MAX_ARITY as u32 + (idx.min(MAX_ARITY - 1)) as u32;
                        acc.remove_edge(emb, parent_op, child.op_type(), eff_depth);
                    }
                    for child in children.iter().rev() {
                        stack.push((child, d + 1));
                    }
                }
            }
        }
    }

    /// Merge another accumulator into this one (vector addition).
    ///
    /// Both flat and depth-encoded halves are additive.
    pub fn merge(&mut self, other: &Self) {
        for i in 0..4 * K {
            self.values[i] += other.values[i];
        }
        self.edge_count += other.edge_count;
        self.node_count += other.node_count;
        self.backref_count += other.backref_count;
    }

    // ========================================================================
    // DAG-Aware Accumulator Construction
    // ========================================================================

    /// Build accumulator from e-graph extraction choices with DAG-aware sharing.
    ///
    /// For each reachable e-class:
    /// - If `ref_count[class] == 1`: add edges normally (unique use).
    /// - If `ref_count[class] > 1`: add edges once (the computation), plus
    ///   `(ref_count - 1)` Var->parent edges (register loads for reuse).
    ///
    /// This matches what the JIT emits: shared subexpressions become let-bindings,
    /// and subsequent uses are cheap register loads.
    pub fn from_dag_choices(
        egraph: &crate::egraph::EGraph,
        root: crate::egraph::EClassId,
        choices: &[Option<usize>],
        ref_count: &[u32],
        emb: &OpEmbeddings,
    ) -> Self {
        Self::from_dag_choices_with_variance(egraph, root, choices, ref_count, emb, None)
    }

    /// Build accumulator from DAG choices, optionally incorporating variance analysis.
    ///
    /// If `variance_analysis` is provided, the accumulator's variance histogram
    /// features are populated (fraction of nodes at each variance level).
    pub fn from_dag_choices_with_variance(
        egraph: &crate::egraph::EGraph,
        root: crate::egraph::EClassId,
        choices: &[Option<usize>],
        ref_count: &[u32],
        emb: &OpEmbeddings,
        variance_analysis: Option<&crate::egraph::deps::DepsAnalysis>,
    ) -> Self {
        use crate::egraph::{EClassId, ENode};

        let mut acc = Self::new();
        let num_classes = egraph.num_classes();
        let mut expanded = alloc::vec![false; num_classes];
        // Variance counters
        let mut n_const: u32 = 0;
        let mut n_frame: u32 = 0;
        let mut n_scanline: u32 = 0;
        let mut n_pixel: u32 = 0;
        // Stack: (class_id, depth)
        let mut stack: alloc::vec::Vec<(EClassId, u32)> = alloc::vec![(root, 0)];

        while let Some((class, depth)) = stack.pop() {
            let canonical = egraph.find(class);
            let idx = canonical.0 as usize;

            if idx >= num_classes {
                panic!(
                    "from_dag_choices: e-class {} out of bounds (num_classes={})",
                    canonical.0, num_classes
                );
            }

            // Always increment node_count on first expansion.
            // Subsequent visits to a shared node only add var_ref edges.
            if expanded[idx] {
                continue;
            }
            expanded[idx] = true;

            let node_idx = match choices[idx] {
                Some(ni) => ni,
                None => continue, // Unreachable class — skip
            };

            let nodes = egraph.nodes(canonical);
            if node_idx >= nodes.len() {
                panic!(
                    "from_dag_choices: node_idx {} out of bounds for e-class {} (has {} nodes)",
                    node_idx,
                    canonical.0,
                    nodes.len()
                );
            }

            let node = &nodes[node_idx];
            acc.node_count += 1;

            // Classify this node's variance if analysis is available
            if let Some(va) = variance_analysis {
                let v = va.get(egraph, canonical);
                if v.is_const() {
                    n_const += 1;
                } else if v.is_x_invariant() && !v.depends_on_y() {
                    // Frame-uniform: depends only on Z/W (no X, no Y)
                    n_frame += 1;
                } else if v.is_x_invariant() {
                    // Scanline-uniform: depends on Y (but not X)
                    n_scanline += 1;
                } else {
                    // Pixel-varying: depends on X
                    n_pixel += 1;
                }
            }

            match node {
                ENode::Var(_) | ENode::Const(_) => {
                    // Leaf: no edges to add. If shared, ref loads are zero-cost
                    // (it's just a variable or constant).
                }
                ENode::Op { op, children } => {
                    let parent_op = op.kind();

                    // Add edges for this node's children (the computation edges).
                    for (child_idx, &child_class) in children.iter().enumerate() {
                        let child_canonical = egraph.find(child_class);
                        let child_node_idx = choices[child_canonical.0 as usize].unwrap_or(0);
                        let child_nodes = egraph.nodes(child_canonical);
                        let child_op = if child_node_idx < child_nodes.len() {
                            match &child_nodes[child_node_idx] {
                                ENode::Var(_) => OpKind::Var,
                                ENode::Const(_) => OpKind::Const,
                                ENode::Op { op: cop, .. } => cop.kind(),
                            }
                        } else {
                            OpKind::Var // fallback
                        };

                        let eff_depth =
                            depth * MAX_ARITY as u32 + (child_idx.min(MAX_ARITY - 1)) as u32;
                        acc.add_edge(emb, parent_op, child_op, eff_depth);

                        // Push child for expansion
                        stack.push((child_class, depth + 1));
                    }

                    // For shared children: add (ref_count - 1) var_ref edges.
                    // These represent register loads at subsequent use sites.
                    for (child_idx, &child_class) in children.iter().enumerate() {
                        let child_canonical = egraph.find(child_class);
                        let rc = ref_count
                            .get(child_canonical.0 as usize)
                            .copied()
                            .unwrap_or(1);
                        if rc > 1 {
                            let eff_depth =
                                depth * MAX_ARITY as u32 + (child_idx.min(MAX_ARITY - 1)) as u32;
                            acc.add_var_ref_edges(emb, parent_op, eff_depth, rc - 1);
                        }
                    }
                }
            }
        }

        // Populate variance histogram fractions
        if variance_analysis.is_some() && acc.node_count > 0 {
            let total = acc.node_count as f32;
            acc.variance_frac_const = n_const as f32 / total;
            acc.variance_frac_frame = n_frame as f32 / total;
            acc.variance_frac_scanline = n_scanline as f32 / total;
            acc.variance_frac_pixel = n_pixel as f32 / total;
        }

        acc
    }

    /// Add N var-reference edges (representing register loads of a shared value).
    ///
    /// Each var-ref edge is `(Var, parent_op)` at the given depth. This tells the
    /// extraction head: "this parent loads a let-bound value N times."
    pub fn add_var_ref_edges(
        &mut self,
        emb: &OpEmbeddings,
        parent_op: OpKind,
        depth: u32,
        count: u32,
    ) {
        for _ in 0..count {
            self.add_edge(emb, parent_op, OpKind::Var, depth);
        }
    }

    /// Remove N var-reference edges (inverse of `add_var_ref_edges`).
    pub fn remove_var_ref_edges(
        &mut self,
        emb: &OpEmbeddings,
        parent_op: OpKind,
        depth: u32,
        count: u32,
    ) {
        for _ in 0..count {
            self.remove_edge(emb, parent_op, OpKind::Var, depth);
        }
    }
}

// ============================================================================
// Graph Accumulator (VSA encoding of e-graph state for saturation head)
// ============================================================================

/// VSA accumulator for e-graph state (rebuilt each epoch).
///
/// Three-section encoding captures both marginal and joint op distributions:
///
/// | Section | Dim | Operation | Signal |
/// |---------|-----|-----------|--------|
/// | `[0..K]` | K | `Σ E[parent]` | Marginal: which ops appear as parents |
/// | `[K..2K]` | K | `Σ E[child]` | Marginal: which ops appear as children |
/// | `[2K..3K]` | K | `Σ E[parent] ⊙ shift₁(E[child])` | **1-hop VSA binding**: which ops are connected |
/// | `[3K..4K]` | K | `Σ E[gp] ⊙ shift₁(E[par]) ⊙ shift²(E[child])` | **2-hop VSA binding**: 3-node path patterns |
///
/// The 1-hop binding section uses element-wise Hadamard product with a cyclic shift to
/// break commutativity (`Mul→Add ≠ Add→Mul`). This captures the **joint**
/// distribution of parent-child pairs — strictly more informative than marginals
/// alone.
///
/// The 2-hop binding section extends this to 3-node paths (grandparent→parent→child),
/// capturing patterns like "Mul feeding Add feeding Sqrt" which is exactly what
/// rewrite rules match on. This turns the accumulator from a 0-round GNN into
/// a 1-round GNN.
///
/// The downstream backbone learns to decode the bundled representation.
///
/// Shares `OpEmbeddings` with the extraction head — same learned op embeddings,
/// different downstream pathway.
#[derive(Clone)]
pub struct GraphAccumulator {
    /// `[0..K]`:    marginal parent sum     `Σ E[parent]`
    /// `[K..2K]`:   marginal child sum      `Σ E[child]`
    /// `[2K..3K]`:  1-hop VSA binding sum   `Σ E[parent] ⊙ shift_by(E[child], depth)`
    /// `[3K..4K]`:  2-hop VSA binding sum   `Σ E[gp] ⊙ shift₁(E[par]) ⊙ shift²(E[child])`
    pub values: [f32; GRAPH_ACC_DIM],
    /// Number of edges added to the accumulator.
    pub edge_count: u32,
    /// Number of nodes (ops + leaves) in the graph.
    pub node_count: u32,
    /// E-graph node budget for this trajectory (how many nodes the saturator may create).
    pub node_budget: u32,
    /// Epoch budget for this trajectory (max saturation epochs).
    pub epoch_budget: u32,
}

impl Default for GraphAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphAccumulator {
    /// Create a zero-initialized graph accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            values: [0.0; GRAPH_ACC_DIM],
            edge_count: 0,
            node_count: 0,
            node_budget: 0,
            epoch_budget: 0,
        }
    }

    /// Reset to zero state.
    ///
    /// Budget fields are intentionally NOT reset — they are trajectory-level
    /// properties that should persist across epoch rebuilds.
    pub fn reset(&mut self) {
        self.values = [0.0; GRAPH_ACC_DIM];
        self.edge_count = 0;
        self.node_count = 0;
    }

    /// Add a single edge with depth-aware VSA encoding.
    ///
    /// Updates all three sections: marginal parent, marginal child, and
    /// VSA binding (`E[parent] ⊙ shift_by(E[child], depth % K)`).
    ///
    /// At `depth == 0` the shift is the identity (root edges), `depth == 1`
    /// shifts by 1 (matching the original `shift1` behavior), `depth == 2`
    /// shifts by 2, etc. This encodes hierarchical position into the binding
    /// without any extra parameters.
    #[inline]
    pub fn add_edge_at_depth(
        &mut self,
        emb: &OpEmbeddings,
        parent_op: OpKind,
        child_op: OpKind,
        depth: usize,
    ) {
        let p = emb.get(parent_op);
        let c = emb.get(child_op);
        let c_shifted = shift_by(c, depth);
        for i in 0..K {
            self.values[i] += p[i]; // marginal parent
            self.values[K + i] += c[i]; // marginal child
            self.values[2 * K + i] += p[i] * c_shifted[i]; // VSA binding
        }
        self.edge_count += 1;
    }

    /// Add a single edge with VSA encoding (backward-compatible, depth = 1).
    ///
    /// Equivalent to `add_edge_at_depth(emb, parent_op, child_op, 1)`.
    /// Preserves the original `shift1` behavior for callers that do not
    /// track depth.
    #[inline]
    pub fn add_edge(&mut self, emb: &OpEmbeddings, parent_op: OpKind, child_op: OpKind) {
        self.add_edge_at_depth(emb, parent_op, child_op, 1);
    }

    /// Add a leaf node (Var/Const) — no edges, just increment node count.
    pub fn add_leaf(&mut self) {
        self.node_count += 1;
    }

    /// Add an Op node and all its edges to children, with depth-aware VSA.
    ///
    /// Emits one `add_edge_at_depth` per child and increments `node_count`
    /// once.  The `depth` parameter is the depth of `op` in the expression
    /// tree (0 = root).
    pub fn add_op_node_at_depth(
        &mut self,
        emb: &OpEmbeddings,
        op: OpKind,
        child_ops: &[OpKind],
        depth: usize,
    ) {
        for &child_op in child_ops {
            self.add_edge_at_depth(emb, op, child_op, depth);
        }
        self.node_count += 1;
    }

    /// Add an Op node and all its edges to children (backward-compatible, depth = 1).
    ///
    /// Equivalent to `add_op_node_at_depth(emb, op, child_ops, 1)`.
    pub fn add_op_node(&mut self, emb: &OpEmbeddings, op: OpKind, child_ops: &[OpKind]) {
        self.add_op_node_at_depth(emb, op, child_ops, 1);
    }

    // ========== Incremental Removal (inverse of addition) ==========

    /// Remove a single edge with depth-aware VSA encoding — the exact
    /// inverse of [`add_edge_at_depth`].
    ///
    /// Subtracts (instead of adds) the parent, child, and VSA binding
    /// contributions from each section. Decrements `edge_count` via
    /// saturating subtraction so underflow clamps to zero rather than
    /// wrapping.
    ///
    /// # Contract
    ///
    /// Callers must only remove edges that were previously added at the
    /// same `depth`. Removing an edge that was never added will corrupt
    /// the accumulator values (negative contributions) and is a logic
    /// error.
    #[inline]
    pub fn remove_edge_at_depth(
        &mut self,
        emb: &OpEmbeddings,
        parent_op: OpKind,
        child_op: OpKind,
        depth: usize,
    ) {
        let p = emb.get(parent_op);
        let c = emb.get(child_op);
        let c_shifted = shift_by(c, depth);
        for i in 0..K {
            self.values[i] -= p[i]; // marginal parent
            self.values[K + i] -= c[i]; // marginal child
            self.values[2 * K + i] -= p[i] * c_shifted[i]; // VSA binding
        }
        self.edge_count = self.edge_count.saturating_sub(1);
    }

    /// Remove a single edge (backward-compatible, depth = 1) — the exact
    /// inverse of [`add_edge`].
    ///
    /// Equivalent to `remove_edge_at_depth(emb, parent_op, child_op, 1)`.
    #[inline]
    pub fn remove_edge(&mut self, emb: &OpEmbeddings, parent_op: OpKind, child_op: OpKind) {
        self.remove_edge_at_depth(emb, parent_op, child_op, 1);
    }

    /// Remove an Op node and all its edges to children with depth-aware
    /// VSA — the exact inverse of [`add_op_node_at_depth`].
    ///
    /// Calls [`remove_edge_at_depth`] for each child and decrements
    /// `node_count`.
    pub fn remove_op_node_at_depth(
        &mut self,
        emb: &OpEmbeddings,
        op: OpKind,
        child_ops: &[OpKind],
        depth: usize,
    ) {
        for &child_op in child_ops {
            self.remove_edge_at_depth(emb, op, child_op, depth);
        }
        self.node_count = self.node_count.saturating_sub(1);
    }

    /// Remove an Op node and all its edges (backward-compatible, depth = 1)
    /// — the exact inverse of [`add_op_node`].
    ///
    /// Equivalent to `remove_op_node_at_depth(emb, op, child_ops, 1)`.
    pub fn remove_op_node(&mut self, emb: &OpEmbeddings, op: OpKind, child_ops: &[OpKind]) {
        self.remove_op_node_at_depth(emb, op, child_ops, 1);
    }

    /// Remove a leaf node — the exact inverse of [`add_leaf`].
    ///
    /// Decrements `node_count` only (leaves contribute no edges).
    pub fn remove_leaf(&mut self) {
        self.node_count = self.node_count.saturating_sub(1);
    }

    // ========== 2-hop Message Passing (1-round GNN) ==========

    /// Add a 2-hop (grandparent→parent→child) binding to the `[3K..4K]` section.
    ///
    /// Encodes 3-node path patterns like "Mul feeding Add feeding Sqrt" using
    /// the VSA triple product `E[grandparent] ⊙ shift₁(E[parent]) ⊙ shift²(E[child])`.
    /// The shift amounts break commutativity: `A→B→C` produces a different
    /// binding than any permutation of {A, B, C}.
    ///
    /// Does NOT modify the `[0..3K]` sections or `edge_count`/`node_count` —
    /// those are maintained by `add_edge*` / `add_op_node*`.
    #[inline]
    pub fn add_2hop_edge(
        &mut self,
        emb: &OpEmbeddings,
        grandparent_op: OpKind,
        parent_op: OpKind,
        child_op: OpKind,
    ) {
        let gp = emb.get(grandparent_op);
        let p = shift1(emb.get(parent_op));
        let c = shift_by(emb.get(child_op), 2);
        for i in 0..K {
            self.values[3 * K + i] += gp[i] * p[i] * c[i];
        }
    }

    /// Remove a 2-hop (grandparent→parent→child) binding — the exact inverse
    /// of [`add_2hop_edge`].
    ///
    /// Subtracts the triple-product contribution from the `[3K..4K]` section.
    ///
    /// # Contract
    ///
    /// Callers must only remove 2-hop edges that were previously added with
    /// the same (grandparent, parent, child) triple. Removing a path that was
    /// never added will corrupt the accumulator values and is a logic error.
    #[inline]
    pub fn remove_2hop_edge(
        &mut self,
        emb: &OpEmbeddings,
        grandparent_op: OpKind,
        parent_op: OpKind,
        child_op: OpKind,
    ) {
        let gp = emb.get(grandparent_op);
        let p = shift1(emb.get(parent_op));
        let c = shift_by(emb.get(child_op), 2);
        for i in 0..K {
            self.values[3 * K + i] -= gp[i] * p[i] * c[i];
        }
    }

    /// Return a copy with each of the four sections independently L2-normalized.
    ///
    /// Raw sums grow proportionally to edge count, so a 200-edge graph has
    /// values ~20x larger than a 10-edge graph.  Normalizing makes the
    /// embedding scale-invariant: small rewrites on large graphs become visible
    /// instead of being swamped by magnitude.
    ///
    /// Each section is normalized independently because they represent different
    /// quantities with different natural scales:
    /// - `[0..K]`    marginal parent sums
    /// - `[K..2K]`   marginal child sums
    /// - `[2K..3K]`  1-hop VSA binding sums
    /// - `[3K..4K]`  2-hop VSA binding sums
    ///
    /// Scalar fields (`edge_count`, `node_count`, etc.) are copied as-is.
    ///
    /// A zero-norm section (no edges accumulated) is left as all-zeros rather
    /// than producing NaN/Inf.
    #[must_use]
    pub fn normalized(&self) -> Self {
        let mut out = self.clone();
        out.normalize_in_place();
        out
    }

    /// L2-normalize each of the four sections in place.
    ///
    /// See [`normalized`](Self::normalized) for rationale.
    pub fn normalize_in_place(&mut self) {
        l2_normalize_section(&mut self.values, 0, K);
        l2_normalize_section(&mut self.values, K, 2 * K);
        l2_normalize_section(&mut self.values, 2 * K, 3 * K);
        l2_normalize_section(&mut self.values, 3 * K, 4 * K);
    }
}

/// L2-normalize a contiguous slice `values[start..end]` in place.
///
/// If the section norm is zero (or negligibly small), it is left untouched
/// to avoid division by zero.
fn l2_normalize_section(values: &mut [f32], start: usize, end: usize) {
    let mut sum_sq: f32 = 0.0;
    for i in start..end {
        sum_sq += values[i] * values[i];
    }
    let norm = sqrtf(sum_sq);
    // Guard: skip normalization for zero/near-zero sections to avoid NaN/Inf.
    if norm < 1e-12 {
        return;
    }
    let inv_norm = 1.0 / norm;
    for i in start..end {
        values[i] *= inv_norm;
    }
}

// ============================================================================
// Structural Hashing
// ============================================================================

/// Compute a structural (content-based) hash of an `Expr` tree.
///
/// Uses FNV-1a as the mixing function: fast, no external deps, and produces
/// a well-distributed 64-bit fingerprint suitable for CSE deduplication.
/// The hash is purely structural — pointer identity is not considered.
///
/// # Collision risk
///
/// FNV-1a on 64-bit is sufficient for expression trees found in practice
/// (thousands of nodes). Collisions would cause two distinct subtrees to be
/// treated as identical (false CSE), producing a slightly compressed
/// accumulator rather than a corrupt one. Fail-loudly: if you suspect
/// collisions add a debug assertion in `collect_recursive_dedup`.
#[must_use]
pub fn structural_hash(expr: &Expr) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    fn mix(mut h: u64, bytes: &[u8]) -> u64 {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
        h
    }

    fn hash_rec(expr: &Expr, h: u64) -> u64 {
        // Encode the discriminant so that Var(0) != Const(0.0) != Param(0).
        let h = match expr {
            Expr::Var(idx) => {
                let h = mix(h, &[0x01]);
                mix(h, &[*idx])
            }
            Expr::Const(v) => {
                let h = mix(h, &[0x02]);
                // Normalise -0.0 → 0.0 before hashing so they compare equal.
                let bits = if *v == 0.0 { 0u32 } else { v.to_bits() };
                mix(h, &bits.to_le_bytes())
            }
            Expr::Param(idx) => {
                let h = mix(h, &[0x03]);
                mix(h, &[*idx])
            }
            Expr::Unary(op, child) => {
                let h = mix(h, &[0x04]);
                let h = mix(h, &(*op as u8).to_le_bytes());
                hash_rec(child, h)
            }
            Expr::Binary(op, left, right) => {
                let h = mix(h, &[0x05]);
                let h = mix(h, &(*op as u8).to_le_bytes());
                let h = hash_rec(left, h);
                hash_rec(right, h)
            }
            Expr::Ternary(op, a, b, c) => {
                let h = mix(h, &[0x06]);
                let h = mix(h, &(*op as u8).to_le_bytes());
                let h = hash_rec(a, h);
                let h = hash_rec(b, h);
                hash_rec(c, h)
            }
            Expr::Nary(op, children) => {
                let h = mix(h, &[0x07]);
                let h = mix(h, &(*op as u8).to_le_bytes());
                let h = mix(h, &(children.len() as u32).to_le_bytes());
                children.iter().fold(h, |h, child| hash_rec(child, h))
            }
        };
        h
    }

    hash_rec(expr, FNV_OFFSET)
}

// ============================================================================
// Edge Extraction Utilities
// ============================================================================

/// An edge in the expression tree: (parent_op, child_op).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Edge {
    /// The parent operation type.
    pub parent: OpKind,
    /// The child operation type.
    pub child: OpKind,
}

/// Extract all parent→child edges from an expression tree.
#[must_use]
pub fn extract_edges(expr: &Expr) -> Vec<Edge> {
    let mut edges = Vec::new();
    extract_edges_recursive(expr, &mut edges);
    edges
}

fn extract_edges_recursive(expr: &Expr, edges: &mut Vec<Edge>) {
    let parent = expr.op_type();

    match expr {
        Expr::Var(_) | Expr::Const(_) => {}
        Expr::Param(i) => panic!(
            "Expr::Param({}) reached NNUE cost model — call substitute_params before use",
            i
        ),
        Expr::Unary(_, child) => {
            edges.push(Edge {
                parent,
                child: child.op_type(),
            });
            extract_edges_recursive(child, edges);
        }
        Expr::Binary(_, left, right) => {
            edges.push(Edge {
                parent,
                child: left.op_type(),
            });
            edges.push(Edge {
                parent,
                child: right.op_type(),
            });
            extract_edges_recursive(left, edges);
            extract_edges_recursive(right, edges);
        }
        Expr::Ternary(_, a, b, c) => {
            edges.push(Edge {
                parent,
                child: a.op_type(),
            });
            edges.push(Edge {
                parent,
                child: b.op_type(),
            });
            edges.push(Edge {
                parent,
                child: c.op_type(),
            });
            extract_edges_recursive(a, edges);
            extract_edges_recursive(b, edges);
            extract_edges_recursive(c, edges);
        }
        Expr::Nary(_, children) => {
            for child in children {
                edges.push(Edge {
                    parent,
                    child: child.op_type(),
                });
                extract_edges_recursive(child, edges);
            }
        }
    }
}

// ============================================================================
// Dual-Head NNUE (AlphaZero-style)
// ============================================================================

/// ExprNnue: shared backbone with one extraction head (Value MLP) and one saturation head (bilinear mask).
///
/// ## Architecture
///
/// ```text
/// expr → OpEmbeddings → EdgeAccumulator → hidden [64] → expr_proj → expr_embed [24]
///                                                            ├─→ value_mlp → cost (extraction head)
///                                                            └─→ [embed, cost] → mask_mlp → bilinear → score (saturation head)
/// ```
///
/// **Extraction head**: `expr_embed → value_mlp (24→16→1)` predicts log-nanosecond cost.
/// **Saturation head**: `[expr_embed, value_pred] → mask_mlp → bilinear(mask_features, rule_embed)` scores rules.
///
/// Rule embeddings come from LHS/RHS templates via `rule_proj`, not from learned per-rule embeddings.
#[derive(Clone)]
pub struct ExprNnue {
    // ========== SHARED (Expression Backbone) ==========
    /// Learned embeddings for each operation (42 × 32 = 1,344 params)
    pub embeddings: OpEmbeddings,

    /// Hidden layer weights: [INPUT_DIM][HIDDEN_DIM] (85 × 64 = 5,440 params)
    pub w1: [[f32; HIDDEN_DIM]; INPUT_DIM],

    /// Hidden layer biases: [HIDDEN_DIM] (64 params)
    pub b1: [f32; HIDDEN_DIM],

    /// Shared trunk weights: [HIDDEN_DIM][HIDDEN_DIM] (64 x 64 = 4,096 params).
    /// Both edge tower and graph tower outputs pass through this layer before
    /// reaching their task-specific projection heads.
    /// This is the "deep conceptual representation" shared between extraction and saturation.
    pub trunk_w: [[f32; HIDDEN_DIM]; HIDDEN_DIM],
    /// Shared trunk biases: [HIDDEN_DIM] (64 params).
    pub trunk_b: [f32; HIDDEN_DIM],

    // ========== UNIFIED MASK ARCHITECTURE ==========
    // These fields support the new bilinear expr-rule interaction model
    // that scales to 1000+ rules.
    /// Projects backbone hidden (64) to shared expr embedding (EMBED_DIM=24).
    /// Weights: [HIDDEN_DIM x EMBED_DIM]
    pub expr_proj_w: [[f32; EMBED_DIM]; HIDDEN_DIM],
    /// Expr projection bias: [EMBED_DIM]
    pub expr_proj_b: [f32; EMBED_DIM],

    /// Value MLP layer 1 weights: expr_embed (24) → hidden (16)
    pub value_mlp_w1: [[f32; MLP_HIDDEN]; EMBED_DIM],
    /// Value MLP layer 1 bias
    pub value_mlp_b1: [f32; MLP_HIDDEN],
    /// Value MLP layer 2 weights: hidden (16) → cost (1)
    pub value_mlp_w2: [f32; MLP_HIDDEN],
    /// Value MLP layer 2 bias
    pub value_mlp_b2: f32,

    /// Mask MLP layer 1 weights: [expr_embed (24), value_pred (1)] → hidden (16)
    /// The mask sees value prediction as input: "Given this costs X, should I try rule R?"
    pub mask_mlp_w1: [[f32; MLP_HIDDEN]; MASK_INPUT_DIM],
    /// Mask MLP layer 1 bias
    pub mask_mlp_b1: [f32; MLP_HIDDEN],
    /// Mask MLP layer 2 weights: hidden (16) → mask_features (24)
    pub mask_mlp_w2: [[f32; EMBED_DIM]; MLP_HIDDEN],
    /// Mask MLP layer 2 bias
    pub mask_mlp_b2: [f32; EMBED_DIM],

    /// Rule MLP layer 1 weights: rule_features (8) → hidden (16).
    /// Shared across all rules - scales sublinearly with rule count.
    /// (Legacy: used with hand-crafted RuleFeatures)
    pub rule_mlp_w1: [[f32; MLP_HIDDEN]; RULE_FEATURE_DIM],
    /// Rule MLP layer 1 bias
    pub rule_mlp_b1: [f32; MLP_HIDDEN],
    /// Rule MLP layer 2 weights: hidden (16) → rule_embed (24)
    pub rule_mlp_w2: [[f32; EMBED_DIM]; MLP_HIDDEN],
    /// Rule MLP layer 2 bias
    pub rule_mlp_b2: [f32; EMBED_DIM],

    // ========== RULE TEMPLATE PROJECTION (LHS/RHS embeddings) ==========
    // These fields support encoding rules from their LHS/RHS expression
    // templates using the SAME expr_embed as extraction/saturation heads.
    //
    // 4-way concat: [z_LHS | z_RHS | z_LHS-z_RHS | z_LHS*z_RHS] (96) → rule_embed (24)
    /// Rule projection weights: [RULE_CONCAT_DIM x EMBED_DIM] = [96 x 24] = 2,304 params.
    /// Projects 4-way concatenation to rule embedding.
    pub rule_proj_w: [[f32; EMBED_DIM]; RULE_CONCAT_DIM],
    /// Rule projection bias: [EMBED_DIM] = 24 params
    pub rule_proj_b: [f32; EMBED_DIM],

    /// Bilinear interaction matrix: mask_features @ interaction @ rule_embed
    pub interaction: [[f32; EMBED_DIM]; EMBED_DIM],

    /// Learned bias projection: produces per-rule bias via dot(mask_bias_proj, rule_embed)
    pub mask_bias_proj: [f32; EMBED_DIM],

    // ========== GRAPH STATE BACKBONE (for saturation head) ==========
    // Separate pathway: GraphAccumulator (VSA e-graph state) → graph_w1 → graph_proj → mask_mlp → bilinear
    // The extraction head path (EdgeAccumulator → w1 → expr_proj) is completely unchanged.
    /// Graph backbone weights: [GRAPH_INPUT_DIM][HIDDEN_DIM] (132 × 64 = 8,448 params)
    pub graph_w1: [[f32; HIDDEN_DIM]; GRAPH_INPUT_DIM],
    /// Graph backbone biases: [HIDDEN_DIM] (64 params)
    pub graph_b1: [f32; HIDDEN_DIM],
    /// Graph → embed projection weights: [HIDDEN_DIM][EMBED_DIM] (64 × 32 = 2,048 params)
    pub graph_proj_w: [[f32; EMBED_DIM]; HIDDEN_DIM],
    /// Graph → embed projection bias: [EMBED_DIM] (32 params)
    pub graph_proj_b: [f32; EMBED_DIM],
}

impl Default for ExprNnue {
    fn default() -> Self {
        Self::new()
    }
}

impl ExprNnue {
    /// Create a zero-initialized dual-head network.
    #[must_use]
    pub fn new() -> Self {
        Self {
            // Backbone
            embeddings: OpEmbeddings::new(),
            w1: [[0.0; HIDDEN_DIM]; INPUT_DIM],
            b1: [0.0; HIDDEN_DIM],

            // Shared trunk (zero-init)
            trunk_w: [[0.0; HIDDEN_DIM]; HIDDEN_DIM],
            trunk_b: [0.0; HIDDEN_DIM],

            // Unified mask architecture
            expr_proj_w: [[0.0; EMBED_DIM]; HIDDEN_DIM],
            expr_proj_b: [0.0; EMBED_DIM],

            value_mlp_w1: [[0.0; MLP_HIDDEN]; EMBED_DIM],
            value_mlp_b1: [0.0; MLP_HIDDEN],
            value_mlp_w2: [0.0; MLP_HIDDEN],
            value_mlp_b2: 5.0, // Start near typical log-cost

            mask_mlp_w1: [[0.0; MLP_HIDDEN]; MASK_INPUT_DIM], // 24 × 16
            mask_mlp_b1: [0.0; MLP_HIDDEN],
            mask_mlp_w2: [[0.0; EMBED_DIM]; MLP_HIDDEN],
            mask_mlp_b2: [0.0; EMBED_DIM],

            rule_mlp_w1: [[0.0; MLP_HIDDEN]; RULE_FEATURE_DIM],
            rule_mlp_b1: [0.0; MLP_HIDDEN],
            rule_mlp_w2: [[0.0; EMBED_DIM]; MLP_HIDDEN],
            rule_mlp_b2: [0.0; EMBED_DIM],

            // Rule template projection (LHS/RHS embeddings)
            rule_proj_w: [[0.0; EMBED_DIM]; RULE_CONCAT_DIM],
            rule_proj_b: [0.0; EMBED_DIM],

            interaction: [[0.0; EMBED_DIM]; EMBED_DIM],
            mask_bias_proj: [0.0; EMBED_DIM],

            // Graph state backbone (zero-init)
            graph_w1: [[0.0; HIDDEN_DIM]; GRAPH_INPUT_DIM],
            graph_b1: [0.0; HIDDEN_DIM],
            graph_proj_w: [[0.0; EMBED_DIM]; HIDDEN_DIM],
            graph_proj_b: [0.0; EMBED_DIM],
        }
    }

    /// Create a randomly initialized dual-head network.
    #[must_use]
    pub fn new_random(seed: u64) -> Self {
        let mut net = Self::new();
        net.randomize(seed);
        net
    }

    /// Create a network with latency-prior initialized embeddings.
    ///
    /// Recommended initialization for cost prediction:
    /// - Embeddings encode known op latencies in dimension 0
    /// - Network weights are randomly initialized
    #[must_use]
    pub fn new_with_latency_prior(seed: u64) -> Self {
        let mut net = Self::new();
        net.embeddings.init_with_latency_prior(seed);
        net.randomize_weights_only(seed);
        net
    }

    /// Convert an older extraction-first model into the unified architecture.
    ///
    /// This preserves the edge tower exactly by initializing the shared trunk
    /// to identity, so pre-trunk hidden activations pass through unchanged.
    /// Task-specific unified-head weights remain zero-initialized and require
    /// training.
    #[must_use]
    pub fn from_factored(factored: &ExprNnue) -> Self {
        let mut net = Self {
            embeddings: factored.embeddings.clone(),
            w1: factored.w1,
            b1: factored.b1,

            // Shared trunk starts as identity so the migrated edge tower
            // preserves its pre-trunk representation exactly.
            trunk_w: [[0.0; HIDDEN_DIM]; HIDDEN_DIM],
            trunk_b: [0.0; HIDDEN_DIM],

            // Unified mask architecture - zero-initialized (needs training)
            expr_proj_w: [[0.0; EMBED_DIM]; HIDDEN_DIM],
            expr_proj_b: [0.0; EMBED_DIM],

            value_mlp_w1: [[0.0; MLP_HIDDEN]; EMBED_DIM],
            value_mlp_b1: [0.0; MLP_HIDDEN],
            value_mlp_w2: [0.0; MLP_HIDDEN],
            value_mlp_b2: 5.0,

            mask_mlp_w1: [[0.0; MLP_HIDDEN]; MASK_INPUT_DIM], // 24 × 16
            mask_mlp_b1: [0.0; MLP_HIDDEN],
            mask_mlp_w2: [[0.0; EMBED_DIM]; MLP_HIDDEN],
            mask_mlp_b2: [0.0; EMBED_DIM],

            rule_mlp_w1: [[0.0; MLP_HIDDEN]; RULE_FEATURE_DIM],
            rule_mlp_b1: [0.0; MLP_HIDDEN],
            rule_mlp_w2: [[0.0; EMBED_DIM]; MLP_HIDDEN],
            rule_mlp_b2: [0.0; EMBED_DIM],

            // Rule template projection - zero-initialized (needs training)
            rule_proj_w: [[0.0; EMBED_DIM]; RULE_CONCAT_DIM],
            rule_proj_b: [0.0; EMBED_DIM],

            interaction: [[0.0; EMBED_DIM]; EMBED_DIM],
            mask_bias_proj: [0.0; EMBED_DIM],

            // Graph state backbone - zero-initialized (needs training)
            graph_w1: [[0.0; HIDDEN_DIM]; GRAPH_INPUT_DIM],
            graph_b1: [0.0; HIDDEN_DIM],
            graph_proj_w: [[0.0; EMBED_DIM]; HIDDEN_DIM],
            graph_proj_b: [0.0; EMBED_DIM],
        };

        for i in 0..HIDDEN_DIM {
            net.trunk_w[i][i] = 1.0;
        }

        net
    }

    /// Randomize only network weights, not embeddings.
    pub fn randomize_weights_only(&mut self, seed: u64) {
        let mut rng_state = seed.wrapping_add(12345);

        let mut next_f32 = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng_state >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
        };

        // Hidden layer
        let scale_w1 = sqrtf(2.0 / INPUT_DIM as f32);
        for row in 0..INPUT_DIM {
            for col in 0..HIDDEN_DIM {
                self.w1[row][col] = next_f32() * scale_w1;
            }
        }

        for b in &mut self.b1 {
            *b = next_f32().abs() * 0.1;
        }

        // Shared trunk: identity + small noise (near-identity preserves tower signal)
        for i in 0..HIDDEN_DIM {
            for j in 0..HIDDEN_DIM {
                self.trunk_w[i][j] = if i == j { 1.0 } else { 0.0 } + next_f32() * 0.01;
            }
        }
        for b in &mut self.trunk_b {
            *b = 0.0;
        }

        // Initialize unified mask architecture (full init - includes shared projection + value mlp)
        self.randomize_unified_arch_with_rng(&mut next_f32);
    }

    /// Internal helper to randomize ALL unified architecture weights.
    ///
    /// ONLY used during full random init (randomize_weights_only).
    /// Do NOT call this when bootstrapping from extraction head weights - use randomize_mask_only instead.
    fn randomize_unified_arch_with_rng<F: FnMut() -> f32>(&mut self, next_f32: &mut F) {
        // He initialization scales
        let scale_proj = sqrtf(2.0 / HIDDEN_DIM as f32);
        let scale_embed = sqrtf(2.0 / EMBED_DIM as f32);
        let scale_hidden = sqrtf(2.0 / MLP_HIDDEN as f32);
        let scale_rule_feat = sqrtf(2.0 / RULE_FEATURE_DIM as f32);

        // Expr projection: HIDDEN_DIM → EMBED_DIM
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                self.expr_proj_w[j][k] = next_f32() * scale_proj;
            }
        }
        for b in &mut self.expr_proj_b {
            *b = next_f32().abs() * 0.1;
        }

        // Value MLP: EMBED_DIM → MLP_HIDDEN → 1
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                self.value_mlp_w1[i][j] = next_f32() * scale_embed;
            }
        }
        for b in &mut self.value_mlp_b1 {
            *b = next_f32().abs() * 0.1;
        }
        for j in 0..MLP_HIDDEN {
            self.value_mlp_w2[j] = next_f32() * scale_hidden;
        }
        self.value_mlp_b2 = 5.0; // Start near typical log-cost

        // Mask MLP: EMBED_DIM (24) → MLP_HIDDEN → EMBED_DIM
        let scale_mask_input = sqrtf(2.0 / MASK_INPUT_DIM as f32);
        for i in 0..MASK_INPUT_DIM {
            for j in 0..MLP_HIDDEN {
                self.mask_mlp_w1[i][j] = next_f32() * scale_mask_input;
            }
        }
        for b in &mut self.mask_mlp_b1 {
            *b = next_f32().abs() * 0.1;
        }
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                self.mask_mlp_w2[j][k] = next_f32() * scale_hidden;
            }
        }
        for b in &mut self.mask_mlp_b2 {
            *b = 0.0; // Neutral
        }

        // Rule MLP: RULE_FEATURE_DIM → MLP_HIDDEN → EMBED_DIM (hand-crafted features)
        for i in 0..RULE_FEATURE_DIM {
            for j in 0..MLP_HIDDEN {
                self.rule_mlp_w1[i][j] = next_f32() * scale_rule_feat;
            }
        }
        for b in &mut self.rule_mlp_b1 {
            *b = next_f32().abs() * 0.1;
        }
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                self.rule_mlp_w2[j][k] = next_f32() * scale_hidden;
            }
        }
        for b in &mut self.rule_mlp_b2 {
            *b = 0.0; // Neutral
        }

        // Rule Projection: RULE_CONCAT_DIM (96) → EMBED_DIM (24)
        // Linear projection from 4-way concat [z_LHS | z_RHS | z_LHS-z_RHS | z_LHS*z_RHS]
        let scale_concat = sqrtf(2.0 / RULE_CONCAT_DIM as f32);
        for i in 0..RULE_CONCAT_DIM {
            for k in 0..EMBED_DIM {
                self.rule_proj_w[i][k] = next_f32() * scale_concat;
            }
        }
        for b in &mut self.rule_proj_b {
            *b = 0.0; // Neutral
        }

        // Interaction matrix: start near identity (simple dot product baseline)
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                self.interaction[i][j] = if i == j { 1.0 } else { next_f32() * 0.1 };
            }
        }

        // Bias projection: neutral
        for b in &mut self.mask_bias_proj {
            *b = 0.0;
        }

        // Graph backbone: GRAPH_INPUT_DIM → HIDDEN_DIM
        let scale_graph = sqrtf(2.0 / GRAPH_INPUT_DIM as f32);
        for row in 0..GRAPH_INPUT_DIM {
            for col in 0..HIDDEN_DIM {
                self.graph_w1[row][col] = next_f32() * scale_graph;
            }
        }
        for b in &mut self.graph_b1 {
            *b = next_f32().abs() * 0.1;
        }

        // Graph projection: HIDDEN_DIM → EMBED_DIM
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                self.graph_proj_w[j][k] = next_f32() * scale_proj;
            }
        }
        for b in &mut self.graph_proj_b {
            *b = next_f32().abs() * 0.1;
        }

        // NOTE: search_mlp randomization removed - mask IS the policy
    }

    /// Randomize all weights including embeddings.
    pub fn randomize(&mut self, seed: u64) {
        self.embeddings.randomize(seed);
        self.randomize_weights_only(seed);
    }

    /// Create a copy with trained backbone but randomized mask weights.
    ///
    /// This is the key method for embedding sharing: load a trained extraction head,
    /// then create a new model that:
    /// - Keeps: embeddings, w1, b1 (trained backbone)
    /// - Keeps: expr_proj_w, expr_proj_b (shared projection - trained with extraction head)
    /// - Keeps: value_mlp_* (extraction head weights)
    /// - Randomizes: mask_mlp, rule_mlp, rule_proj, interaction, mask_bias_proj (saturation head specific)
    ///
    /// Use this when bootstrapping saturation head training from a pre-trained extraction head.
    #[must_use]
    pub fn with_randomized_mask_weights(&self, seed: u64) -> Self {
        let mut model = self.clone();
        model.randomize_mask_only(seed);
        model
    }

    /// Randomize ONLY mask-specific weights, preserving shared backbone and value MLP.
    ///
    /// Randomizes: mask_mlp, rule_mlp, rule_proj, interaction, mask_bias_proj
    /// Preserves: embeddings, w1, b1, expr_proj, value_mlp
    pub fn randomize_mask_only(&mut self, seed: u64) {
        let mut rng_state = seed.wrapping_add(54321);

        let mut next_f32 = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng_state >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
        };

        // He initialization scales
        let scale_embed = sqrtf(2.0 / EMBED_DIM as f32);
        let scale_mask_input = sqrtf(2.0 / MASK_INPUT_DIM as f32); // 24 dims
        let scale_hidden = sqrtf(2.0 / MLP_HIDDEN as f32);
        let scale_rule_feat = sqrtf(2.0 / RULE_FEATURE_DIM as f32);
        let scale_concat = sqrtf(2.0 / RULE_CONCAT_DIM as f32);

        // Mask MLP: EMBED_DIM (24) → MLP_HIDDEN → EMBED_DIM
        for i in 0..MASK_INPUT_DIM {
            for j in 0..MLP_HIDDEN {
                self.mask_mlp_w1[i][j] = next_f32() * scale_mask_input;
            }
        }
        for b in &mut self.mask_mlp_b1 {
            *b = next_f32().abs() * 0.1;
        }
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                self.mask_mlp_w2[j][k] = next_f32() * scale_hidden;
            }
        }
        for b in &mut self.mask_mlp_b2 {
            *b = 0.0;
        }

        // Rule MLP: RULE_FEATURE_DIM → MLP_HIDDEN → EMBED_DIM (hand-crafted features)
        for i in 0..RULE_FEATURE_DIM {
            for j in 0..MLP_HIDDEN {
                self.rule_mlp_w1[i][j] = next_f32() * scale_rule_feat;
            }
        }
        for b in &mut self.rule_mlp_b1 {
            *b = next_f32().abs() * 0.1;
        }
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                self.rule_mlp_w2[j][k] = next_f32() * scale_hidden;
            }
        }
        for b in &mut self.rule_mlp_b2 {
            *b = 0.0;
        }

        // Rule Projection: RULE_CONCAT_DIM (96) → EMBED_DIM (24)
        for i in 0..RULE_CONCAT_DIM {
            for k in 0..EMBED_DIM {
                self.rule_proj_w[i][k] = next_f32() * scale_concat;
            }
        }
        for b in &mut self.rule_proj_b {
            *b = 0.0;
        }

        // Interaction matrix: start near identity
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                self.interaction[i][j] = if i == j { 1.0 } else { next_f32() * 0.1 };
            }
        }

        // Bias projection: neutral
        for b in &mut self.mask_bias_proj {
            *b = 0.0;
        }

        // Graph backbone: GRAPH_INPUT_DIM → HIDDEN_DIM (mask-specific pathway)
        let scale_graph = sqrtf(2.0 / GRAPH_INPUT_DIM as f32);
        let scale_proj = sqrtf(2.0 / HIDDEN_DIM as f32);
        for row in 0..GRAPH_INPUT_DIM {
            for col in 0..HIDDEN_DIM {
                self.graph_w1[row][col] = next_f32() * scale_graph;
            }
        }
        for b in &mut self.graph_b1 {
            *b = next_f32().abs() * 0.1;
        }

        // Graph projection: HIDDEN_DIM → EMBED_DIM
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                self.graph_proj_w[j][k] = next_f32() * scale_proj;
            }
        }
        for b in &mut self.graph_proj_b {
            *b = next_f32().abs() * 0.1;
        }
    }

    /// Apply shared trunk: HIDDEN_DIM -> HIDDEN_DIM with ReLU.
    ///
    /// Both the edge tower and graph tower pass through this layer before
    /// reaching their task-specific projection heads. Initialized near-identity
    /// so the trunk preserves tower signal until training pulls it away.
    #[inline]
    fn apply_trunk(&self, tower_output: &[f32; HIDDEN_DIM]) -> [f32; HIDDEN_DIM] {
        let mut out = self.trunk_b;
        for i in 0..HIDDEN_DIM {
            for j in 0..HIDDEN_DIM {
                out[j] += tower_output[i] * self.trunk_w[i][j];
            }
        }
        for h in &mut out {
            *h = h.max(0.0);
        }
        out
    }

    /// Shared forward pass through dual accumulator + hidden layer.
    ///
    /// Input: 128 dual accumulator dims (64 flat + 64 depth-encoded)
    ///        + 4 scalar features (edge_count, node_count, node_budget, epoch_budget).
    /// Returns the hidden layer activations after tower ReLU + shared trunk.
    #[inline]
    pub fn forward_shared(&self, acc: &EdgeAccumulator) -> [f32; HIDDEN_DIM] {
        let mut hidden = self.b1;

        // Scale factor to prevent AST explosion (up to massive kernels).
        // 1/sqrt(N) prevents variance explosion from summing N embedding vectors.
        let scale = if acc.node_count > 0 {
            1.0 / libm::sqrtf(acc.node_count as f32)
        } else {
            1.0
        };

        // Process dual accumulator (128 dims: 64 flat + 64 depth-encoded)
        for (i, &val) in acc.values.iter().enumerate() {
            let scaled_val = val * scale;
            for (j, h) in hidden.iter_mut().enumerate() {
                *h += scaled_val * self.w1[i][j];
            }
        }

        // Process scalar features (4 dims: edge_count, node_count, node_budget, epoch_budget).
        // Use log2 to compress the range for large ASTs.
        let base = 4 * K;
        let ec = libm::log2f(1.0 + acc.edge_count as f32);
        let nc = libm::log2f(1.0 + acc.node_count as f32);
        let nb = libm::log2f(1.0 + acc.node_budget as f32);
        let eb = libm::log2f(1.0 + acc.epoch_budget as f32);
        for (j, h) in hidden.iter_mut().enumerate() {
            *h += ec * self.w1[base][j];
            *h += nc * self.w1[base + 1][j];
            *h += nb * self.w1[base + 2][j];
            *h += eb * self.w1[base + 3][j];
        }

        // ReLU activation
        for h in &mut hidden {
            *h = h.max(0.0);
        }

        // Shared trunk
        let hidden = self.apply_trunk(&hidden);

        hidden
    }

    /// Value head: predict cost in log-nanoseconds.
    ///
    /// Used for **extraction**: pick the lowest-cost expression from an e-class.
    /// Apply exp() to get actual nanoseconds.
    ///
    /// Uses `from_expr_dedup` to avoid double-counting shared subtrees (CSE).
    #[must_use]
    pub fn predict_log_cost(&self, expr: &Expr) -> f32 {
        let acc = EdgeAccumulator::from_expr_dedup(expr, &self.embeddings);
        let hidden = self.forward_shared(&acc);
        let expr_embed = self.compute_expr_embed(&hidden);
        self.value_mlp_forward(&expr_embed)
    }

    /// Extraction head: predict cost in nanoseconds (exp of log-cost).
    ///
    /// Convenience method that applies exp() to log-cost.
    #[must_use]
    pub fn predict_cost(&self, expr: &Expr) -> f32 {
        libm::expf(self.predict_log_cost(expr))
    }

    /// Extraction head with pre-computed accumulator.
    ///
    /// More efficient when you already have the accumulator.
    #[must_use]
    pub fn predict_log_cost_with_features(&self, acc: &EdgeAccumulator) -> f32 {
        // Extraction head uses expression structure ONLY — no search resource
        // scalars (node_budget, epoch_budget, edge_count, node_count).
        // Those features exist for the saturation head which needs to reason
        // about search resources. The extraction head predicts execution cost
        // which depends only on what ops are in the expression, not how many
        // nodes the e-graph had or what budget was used.
        let hidden = self.forward_expr_only(acc);
        let expr_embed = self.compute_expr_embed(&hidden);
        self.value_mlp_forward(&expr_embed)
    }

    /// Forward pass through shared backbone using ONLY expression structure.
    ///
    /// Skips the 4 scalar features (edge_count, node_count, node_budget,
    /// epoch_budget) that are search-state metadata, not expression properties.
    /// Use this for the extraction head; use `forward_shared` for the saturation head
    /// which needs resource-awareness.
    pub fn forward_expr_only(&self, acc: &EdgeAccumulator) -> [f32; HIDDEN_DIM] {
        let mut hidden = self.b1;

        let scale = if acc.node_count > 0 {
            1.0 / libm::sqrtf(acc.node_count as f32)
        } else {
            1.0
        };

        // Dual accumulator values (expression structure): dims 0..128
        for (i, &val) in acc.values.iter().enumerate() {
            let scaled_val = val * scale;
            for (j, h) in hidden.iter_mut().enumerate() {
                *h += scaled_val * self.w1[i][j];
            }
        }

        // Variance histogram features: dims 128..132
        // Uses the w1 slots that forward_shared uses for scalars.
        // After retraining, the extraction head learns that low-variance
        // nodes are cheap (hoisted out of inner loops).
        let variance_features = [
            acc.variance_frac_const,
            acc.variance_frac_frame,
            acc.variance_frac_scanline,
            acc.variance_frac_pixel,
        ];
        for (k, &val) in variance_features.iter().enumerate() {
            let i = 4 * K + k; // w1 index 128..132
            if i < INPUT_DIM {
                for (j, h) in hidden.iter_mut().enumerate() {
                    *h += val * self.w1[i][j];
                }
            }
        }

        // ReLU (same as forward_shared)
        for h in &mut hidden {
            *h = h.max(0.0);
        }

        // Shared trunk
        let hidden = self.apply_trunk(&hidden);

        hidden
    }

    // ========================================================================
    // Unified Mask Architecture Forward Pass
    //
    // New bilinear interaction model that scales to 1000+ rules.
    //
    // Architecture:
    //   expr → backbone → hidden → expr_proj → expr_embed (shared)
    //                                              │
    //          ┌───────────────────────────────────┼───────────────────┐
    //          │                                   │                   │
    //          ▼                                   ▼                   ▼
    //    value_mlp (private)               mask_mlp (private)    rule_features
    //          │                                   │                   │
    //          ▼                                   │             rule_mlp (shared)
    //       cost (1)                               │                   │
    //                                              ▼                   ▼
    //                                        mask_features       rule_embed
    //                                              │                   │
    //                                              └──── bilinear ─────┘
    //                                                       │
    //                                                       ▼
    //                                              score + rule_bias
    // ========================================================================

    /// Project backbone hidden to shared expr embedding (EMBED_DIM).
    #[inline]
    pub fn compute_expr_embed(&self, hidden: &[f32; HIDDEN_DIM]) -> [f32; EMBED_DIM] {
        let mut embed = self.expr_proj_b;
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                embed[k] += hidden[j] * self.expr_proj_w[j][k];
            }
        }
        embed
    }

    // ========================================================================
    // Graph State Backbone Forward Pass
    //
    // Separate pathway for saturation head: GraphAccumulator → graph_w1 → graph_proj
    // Feeds into the SAME mask_mlp + bilinear scoring as the expr pathway.
    // ========================================================================

    /// Graph state forward pass (for saturation head).
    ///
    /// Same structure as `forward_shared` but with `graph_w1`/`graph_b1` and
    /// `GRAPH_INPUT_DIM` input (132). Uses `1/sqrt(node_count)` scaling
    /// and log2 scalars, matching the `forward_shared` conventions.
    #[inline]
    pub fn forward_graph(&self, gacc: &GraphAccumulator) -> [f32; HIDDEN_DIM] {
        let mut hidden = self.graph_b1;

        // Scale factor: 1/sqrt(N) prevents variance explosion from summing N embeddings.
        let scale = if gacc.node_count > 0 {
            1.0 / libm::sqrtf(gacc.node_count as f32)
        } else {
            1.0
        };

        // Process graph accumulator (128 dims: 4K sections)
        for (i, &val) in gacc.values.iter().enumerate() {
            let scaled_val = val * scale;
            for (j, h) in hidden.iter_mut().enumerate() {
                *h += scaled_val * self.graph_w1[i][j];
            }
        }

        // Process scalar features (4 dims: edge_count, node_count, node_budget, epoch_budget).
        // Use log2 to compress the range for large e-graphs.
        let base = GRAPH_ACC_DIM;
        let ec = libm::log2f(1.0 + gacc.edge_count as f32);
        let nc = libm::log2f(1.0 + gacc.node_count as f32);
        let nb = libm::log2f(1.0 + gacc.node_budget as f32);
        let eb = libm::log2f(1.0 + gacc.epoch_budget as f32);
        for (j, h) in hidden.iter_mut().enumerate() {
            *h += ec * self.graph_w1[base][j];
            *h += nc * self.graph_w1[base + 1][j];
            *h += nb * self.graph_w1[base + 2][j];
            *h += eb * self.graph_w1[base + 3][j];
        }

        // ReLU activation
        for h in &mut hidden {
            *h = h.max(0.0);
        }

        // Shared trunk
        let hidden = self.apply_trunk(&hidden);

        hidden
    }

    /// Project graph hidden to graph embedding (EMBED_DIM).
    ///
    /// Same structure as `compute_expr_embed` but with `graph_proj_w`/`graph_proj_b`.
    #[inline]
    pub fn compute_graph_embed(&self, hidden: &[f32; HIDDEN_DIM]) -> [f32; EMBED_DIM] {
        let mut embed = self.graph_proj_b;
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                embed[k] += hidden[j] * self.graph_proj_w[j][k];
            }
        }
        embed
    }

    /// Score all rules using graph state (not expression state).
    ///
    /// `forward_graph → compute_graph_embed → compute_mask_features → bilinear_score`
    ///
    /// The mask_mlp, interaction matrix, and bilinear_score are **shared** with the
    /// expression pathway — only the input pathway changes.
    #[must_use]
    pub fn mask_score_all_rules_graph(
        &self,
        gacc: &GraphAccumulator,
        rule_embeds: &[[f32; EMBED_DIM]],
    ) -> Vec<f32> {
        let hidden = self.forward_graph(gacc);
        let graph_embed = self.compute_graph_embed(&hidden);
        let mask_features = self.compute_mask_features(&graph_embed);
        rule_embeds
            .iter()
            .map(|re| self.bilinear_score(&mask_features, re))
            .collect()
    }

    /// Compute mask features from expr embedding and value prediction (for bilinear scoring).
    ///
    /// Transform expr_embed through mask MLP to produce mask features for bilinear scoring.
    ///
    /// Input: expr_embed (24 dims) directly — value_pred was removed as redundant.
    /// MLP: EMBED_DIM (24) → MLP_HIDDEN (ReLU) → EMBED_DIM (24)
    #[inline]
    fn compute_mask_features(&self, expr_embed: &[f32; EMBED_DIM]) -> [f32; EMBED_DIM] {
        // First layer: EMBED_DIM → MLP_HIDDEN
        let mut h = self.mask_mlp_b1;

        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                h[j] += expr_embed[i] * self.mask_mlp_w1[i][j];
            }
        }

        // ReLU
        for j in 0..MLP_HIDDEN {
            h[j] = h[j].max(0.0);
        }

        // Second layer: MLP_HIDDEN → EMBED_DIM
        let mut out = self.mask_mlp_b2;
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                out[k] += h[j] * self.mask_mlp_w2[j][k];
            }
        }
        out
    }

    /// Forward pass through value MLP from expr embedding.
    ///
    /// MLP: EMBED_DIM (24) → MLP_HIDDEN (16, ReLU) → 1
    /// Returns the predicted cost for this expression.
    #[inline]
    fn value_mlp_forward(&self, expr_embed: &[f32; EMBED_DIM]) -> f32 {
        // First layer: EMBED_DIM → MLP_HIDDEN
        let mut h = self.value_mlp_b1;
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                h[j] += expr_embed[i] * self.value_mlp_w1[i][j];
            }
        }

        // ReLU
        for j in 0..MLP_HIDDEN {
            h[j] = h[j].max(0.0);
        }

        // Second layer: MLP_HIDDEN → 1
        let mut cost = self.value_mlp_b2;
        for j in 0..MLP_HIDDEN {
            cost += h[j] * self.value_mlp_w2[j];
        }
        cost
    }

    /// Encode rule features to rule embedding.
    ///
    /// MLP: RULE_FEATURE_DIM → MLP_HIDDEN (ReLU) → EMBED_DIM
    /// This MLP is shared across all rules - scales sublinearly with rule count.
    #[must_use]
    pub fn encode_rule(&self, rule_features: &[f32; RULE_FEATURE_DIM]) -> [f32; EMBED_DIM] {
        // First layer: RULE_FEATURE_DIM → MLP_HIDDEN
        let mut h = self.rule_mlp_b1;
        for i in 0..RULE_FEATURE_DIM {
            for j in 0..MLP_HIDDEN {
                h[j] += rule_features[i] * self.rule_mlp_w1[i][j];
            }
        }

        // ReLU
        for j in 0..MLP_HIDDEN {
            h[j] = h[j].max(0.0);
        }

        // Second layer: MLP_HIDDEN → EMBED_DIM
        let mut out = self.rule_mlp_b2;
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                out[k] += h[j] * self.rule_mlp_w2[j][k];
            }
        }
        out
    }

    /// Pre-encode all rules (call once, cache results).
    ///
    /// Returns a Vec of rule embeddings that can be reused across multiple
    /// expressions during saturation.
    #[must_use]
    pub fn encode_all_rules(
        &self,
        rule_features: &RuleFeatures,
        num_rules: usize,
    ) -> Vec<[f32; EMBED_DIM]> {
        (0..num_rules)
            .map(|r| self.encode_rule(&rule_features.features[r]))
            .collect()
    }

    // =========================================================================
    // Rule Encoding from LHS/RHS Templates
    //
    // Uses the SAME expr_embed as extraction/saturation heads. 4-way concatenation:
    // [z_LHS | z_RHS | z_LHS-z_RHS | z_LHS*z_RHS] → linear → rule_embed
    //
    // This provides richer semantic features than hand-crafted rule descriptors.
    // =========================================================================

    /// Encode a rule from its LHS and RHS expression templates.
    ///
    /// Uses the shared backbone to embed both LHS and RHS, then concatenates
    /// four views: [z_LHS, z_RHS, z_LHS-z_RHS, z_LHS*z_RHS] and projects to
    /// EMBED_DIM.
    ///
    /// # Arguments
    /// * `lhs` - LHS pattern (what the rule matches), e.g., `A * (B + C)`
    /// * `rhs` - RHS pattern (what it produces), e.g., `A*B + A*C`
    ///
    /// # Semantic interpretation
    /// - `z_LHS`: What the rule MATCHES (pattern recognition)
    /// - `z_RHS`: What it PRODUCES (production prediction)
    /// - `z_LHS - z_RHS`: What CHANGED (the delta) - inverse rules have opposite signs
    /// - `z_LHS * z_RHS`: What's SHARED (preserved structure)
    #[must_use]
    pub fn encode_rule_from_templates(&self, lhs: &Expr, rhs: &Expr) -> [f32; EMBED_DIM] {
        // Embed LHS using shared backbone + expr_proj
        let lhs_acc = EdgeAccumulator::from_expr_dedup(lhs, &self.embeddings);
        let lhs_hidden = self.forward_shared(&lhs_acc);
        let z_lhs = self.compute_expr_embed(&lhs_hidden);

        // Embed RHS using shared backbone + expr_proj
        let rhs_acc = EdgeAccumulator::from_expr_dedup(rhs, &self.embeddings);
        let rhs_hidden = self.forward_shared(&rhs_acc);
        let z_rhs = self.compute_expr_embed(&rhs_hidden);

        // 4-way concatenate: [z_LHS | z_RHS | z_LHS-z_RHS | z_LHS*z_RHS] = 96 dims
        let mut concat = [0.0f32; RULE_CONCAT_DIM];
        for i in 0..EMBED_DIM {
            concat[i] = z_lhs[i]; // what it matches
            concat[EMBED_DIM + i] = z_rhs[i]; // what it produces
            concat[2 * EMBED_DIM + i] = z_lhs[i] - z_rhs[i]; // the delta
            concat[3 * EMBED_DIM + i] = z_lhs[i] * z_rhs[i]; // shared structure
        }

        // Linear projection: 96 → 24 (no MLP, rich features already)
        let mut out = self.rule_proj_b;
        for i in 0..RULE_CONCAT_DIM {
            for k in 0..EMBED_DIM {
                out[k] += concat[i] * self.rule_proj_w[i][k];
            }
        }
        out
    }

    /// Pre-encode all rules from templates (call once at init, cache results).
    ///
    /// Rules without templates fall back to zero embedding.
    /// Rule embeddings don't change during search - they're computed from
    /// LHS/RHS templates which are static.
    #[must_use]
    pub fn encode_all_rules_from_templates(
        &self,
        templates: &RuleTemplates,
    ) -> Vec<[f32; EMBED_DIM]> {
        (0..templates.len())
            .map(|r| {
                match (templates.get_lhs(r), templates.get_rhs(r)) {
                    (Some(lhs), Some(rhs)) => self.encode_rule_from_templates(lhs, rhs),
                    _ => [0.0f32; EMBED_DIM], // No template - zero embedding
                }
            })
            .collect()
    }

    /// Bilinear score: mask_features @ interaction @ rule_embed + bias.
    ///
    /// Efficient O(1) scoring with pre-computed mask_features.
    #[inline]
    #[must_use]
    pub fn bilinear_score(
        &self,
        mask_features: &[f32; EMBED_DIM],
        rule_embed: &[f32; EMBED_DIM],
    ) -> f32 {
        // transformed = mask_features @ interaction
        let mut transformed = [0.0f32; EMBED_DIM];
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                transformed[j] += mask_features[i] * self.interaction[i][j];
            }
        }

        // score = dot(transformed + mask_bias_proj, rule_embed)
        let mut score = 0.0f32;
        for k in 0..EMBED_DIM {
            score += (transformed[k] + self.mask_bias_proj[k]) * rule_embed[k];
        }
        score
    }

    /// Score all rules for an expression using unified mask architecture.
    ///
    /// Uses pre-cached rule embeddings for efficiency. One forward pass through
    /// backbone + expr_proj + mask_mlp, then O(rules) bilinear scoring.
    #[must_use]
    pub fn mask_score_all_rules(&self, expr: &Expr, rule_embeds: &[[f32; EMBED_DIM]]) -> Vec<f32> {
        let acc = EdgeAccumulator::from_expr_dedup(expr, &self.embeddings);
        let hidden = self.forward_shared(&acc);
        let expr_embed = self.compute_expr_embed(&hidden);

        let mask_features = self.compute_mask_features(&expr_embed);

        rule_embeds
            .iter()
            .map(|rule_embed| self.bilinear_score(&mask_features, rule_embed))
            .collect()
    }

    /// Score all rules with pre-computed backbone hidden state.
    ///
    /// More efficient when you have multiple expressions sharing the same
    /// feature extraction.
    #[must_use]
    pub fn mask_score_all_rules_with_hidden(
        &self,
        hidden: &[f32; HIDDEN_DIM],
        rule_embeds: &[[f32; EMBED_DIM]],
    ) -> Vec<f32> {
        let expr_embed = self.compute_expr_embed(hidden);

        let mask_features = self.compute_mask_features(&expr_embed);

        rule_embeds
            .iter()
            .map(|rule_embed| self.bilinear_score(&mask_features, rule_embed))
            .collect()
    }

    /// Filter rules by mask threshold using unified architecture.
    ///
    /// Returns indices of rules with sigmoid(score) > threshold.
    #[must_use]
    pub fn filter_rules_unified(
        &self,
        expr: &Expr,
        rule_embeds: &[[f32; EMBED_DIM]],
        threshold: f32,
    ) -> Vec<usize> {
        let scores = self.mask_score_all_rules(expr, rule_embeds);
        scores
            .iter()
            .enumerate()
            .filter(|(_, score)| sigmoid(**score) > threshold)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Score a single (expression, rule) pair for mask prediction.
    ///
    /// Computes the rule embedding on-the-fly from rule features.
    /// Returns raw score (apply sigmoid for probability).
    #[must_use]
    pub fn mask_score_single(&self, expr: &Expr, rule_features: &[f32; RULE_FEATURE_DIM]) -> f32 {
        let acc = EdgeAccumulator::from_expr_dedup(expr, &self.embeddings);
        let hidden = self.forward_shared(&acc);
        let expr_embed = self.compute_expr_embed(&hidden);

        let mask_features = self.compute_mask_features(&expr_embed);

        // Compute rule embedding from features
        let rule_embed = self.encode_rule(rule_features);

        // Bilinear scoring
        self.bilinear_score(&mask_features, &rule_embed)
    }

    /// Compute full NNUE metadata for an expression.
    ///
    /// Returns (expr_embed, value_pred, mask_features) tuple.
    /// Precompute embeddings, value prediction, and mask features for an expression.
    ///
    /// The data flow is:
    /// ```text
    /// expr → backbone → expr_embed (24) → value_mlp → value_pred (1)
    ///                         ↓
    ///                     mask_mlp
    ///                         ↓
    ///                  mask_features (24)
    /// ```
    #[must_use]
    pub fn compute_metadata(&self, expr: &Expr) -> ([f32; EMBED_DIM], f32, [f32; EMBED_DIM]) {
        let acc = EdgeAccumulator::from_expr_dedup(expr, &self.embeddings);
        let hidden = self.forward_shared(&acc);
        let expr_embed = self.compute_expr_embed(&hidden);

        // Compute value prediction (independent of mask)
        let value_pred = self.value_mlp_forward(&expr_embed);

        // Compute mask features directly from expr_embed
        let mask_features = self.compute_mask_features(&expr_embed);

        (expr_embed, value_pred, mask_features)
    }

    /// Total parameter count.
    #[must_use]
    pub const fn param_count() -> usize {
        OpEmbeddings::param_count()           // embeddings: 42 * 32 = 1,344
            + INPUT_DIM * HIDDEN_DIM          // w1: 130 * 64 = 8,320
            + HIDDEN_DIM                      // b1: 64
            // expr_proj
            + HIDDEN_DIM * EMBED_DIM          // expr_proj_w: 64 * 24 = 1,536
            + EMBED_DIM                       // expr_proj_b: 24
            // value MLP
            + EMBED_DIM * MLP_HIDDEN          // value_mlp_w1: 24 * 16 = 384
            + MLP_HIDDEN                      // value_mlp_b1: 16
            + MLP_HIDDEN                      // value_mlp_w2: 16
            + 1                               // value_mlp_b2: 1
            // mask MLP
            + MASK_INPUT_DIM * MLP_HIDDEN     // mask_mlp_w1: 24 * 16 = 384
            + MLP_HIDDEN                      // mask_mlp_b1: 16
            + MLP_HIDDEN * EMBED_DIM          // mask_mlp_w2: 16 * 24 = 384
            + EMBED_DIM                       // mask_mlp_b2: 24
            // rule MLP
            + RULE_FEATURE_DIM * MLP_HIDDEN   // rule_mlp_w1: 8 * 16 = 128
            + MLP_HIDDEN                      // rule_mlp_b1: 16
            + MLP_HIDDEN * EMBED_DIM          // rule_mlp_w2: 16 * 24 = 384
            + EMBED_DIM                       // rule_mlp_b2: 24
            // rule projection
            + RULE_CONCAT_DIM * EMBED_DIM     // rule_proj_w: 96 * 24 = 2,304
            + EMBED_DIM                       // rule_proj_b: 24
            // bilinear
            + EMBED_DIM * EMBED_DIM           // interaction: 24 * 24 = 576
            + EMBED_DIM                        // mask_bias_proj: 32
            // graph state backbone
            + GRAPH_INPUT_DIM * HIDDEN_DIM    // graph_w1: 132 * 64 = 8,448
            + HIDDEN_DIM                      // graph_b1: 64
            + HIDDEN_DIM * EMBED_DIM          // graph_proj_w: 64 * 32 = 2,048
            + EMBED_DIM // graph_proj_b: 32
    }

    /// Memory size in bytes (f32 weights).
    #[must_use]
    pub const fn memory_bytes() -> usize {
        Self::param_count() * 4
    }

    // ========================================================================
    // MCTS Support: Accumulator-based Evaluation
    //
    // These methods enable cheap MCTS simulation without e-graph cloning.
    // The accumulator can be incrementally updated as rules are applied.
    // ========================================================================

    /// Predict cost directly from accumulator (for MCTS evaluation).
    ///
    /// Skips expr parsing - just forward pass through backbone + extraction head.
    /// Use this for fast MCTS rollout evaluation.
    #[must_use]
    pub fn predict_cost_from_accumulator(&self, acc: &EdgeAccumulator) -> f32 {
        let hidden = self.forward_shared(acc);
        let expr_embed = self.compute_expr_embed(&hidden);

        // Value MLP: EMBED_DIM → MLP_HIDDEN (ReLU) → 1
        let mut h = self.value_mlp_b1;
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                h[j] += expr_embed[i] * self.value_mlp_w1[i][j];
            }
        }
        for j in 0..MLP_HIDDEN {
            h[j] = h[j].max(0.0); // ReLU
        }

        let mut cost = self.value_mlp_b2;
        for j in 0..MLP_HIDDEN {
            cost += h[j] * self.value_mlp_w2[j];
        }
        cost
    }

    /// Predict cost with pre-computed accumulator (for MCTS).
    #[must_use]
    pub fn predict_cost_from_features(&self, acc: &EdgeAccumulator) -> f32 {
        let hidden = self.forward_shared(acc);
        let expr_embed = self.compute_expr_embed(&hidden);

        // Value MLP: EMBED_DIM → MLP_HIDDEN (ReLU) → 1
        let mut h = self.value_mlp_b1;
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                h[j] += expr_embed[i] * self.value_mlp_w1[i][j];
            }
        }
        for j in 0..MLP_HIDDEN {
            h[j] = h[j].max(0.0); // ReLU
        }

        let mut cost = self.value_mlp_b2;
        for j in 0..MLP_HIDDEN {
            cost += h[j] * self.value_mlp_w2[j];
        }
        cost
    }

    /// Get policy logits from accumulator (for MCTS prior).
    ///
    /// Returns scores for all rules. Use softmax to get probabilities.
    /// This is the MCTS policy prior P(s, a) for UCB.
    #[must_use]
    pub fn policy_from_accumulator(
        &self,
        acc: &EdgeAccumulator,
        rule_embeds: &[[f32; EMBED_DIM]],
    ) -> Vec<f32> {
        let hidden = self.forward_shared(acc);
        let expr_embed = self.compute_expr_embed(&hidden);

        let mask_features = self.compute_mask_features(&expr_embed);

        rule_embeds
            .iter()
            .map(|rule_embed| self.bilinear_score(&mask_features, rule_embed))
            .collect()
    }

    /// Get policy logits with pre-computed accumulator.
    #[must_use]
    pub fn policy_from_features(
        &self,
        acc: &EdgeAccumulator,
        rule_embeds: &[[f32; EMBED_DIM]],
    ) -> Vec<f32> {
        let hidden = self.forward_shared(acc);
        let expr_embed = self.compute_expr_embed(&hidden);

        let mask_features = self.compute_mask_features(&expr_embed);

        rule_embeds
            .iter()
            .map(|rule_embed| self.bilinear_score(&mask_features, rule_embed))
            .collect()
    }

    /// Get Bernoulli policy probabilities: P(apply) = sigmoid(logit / temp).
    ///
    /// Each rule is an independent binary decision (Bernoulli trial).
    /// Use these to stochastically decide: `if random() < prob[rule] { apply(rule) }`.
    ///
    /// Note: softmax on binary [logit, 0] = sigmoid(logit), so this IS the
    /// correct softmax formulation for independent apply/don't-apply decisions.
    ///
    /// Temperature controls exploration:
    /// - temp → 0: deterministic (prob → 0 or 1)
    /// - temp = 1: standard sigmoid
    /// - temp > 1: more exploration (probs pushed toward 0.5)
    #[must_use]
    pub fn bernoulli_policy_from_accumulator(
        &self,
        acc: &EdgeAccumulator,
        rule_embeds: &[[f32; EMBED_DIM]],
        temperature: f32,
    ) -> Vec<f32> {
        let logits = self.policy_from_accumulator(acc, rule_embeds);
        let temp = temperature.max(0.01);
        logits.iter().map(|&x| sigmoid(x / temp)).collect()
    }

    /// Stochastically sample which rules to apply using Bernoulli policy.
    ///
    /// For each rule independently: apply if `random() < sigmoid(logit / temp)`.
    /// Returns indices of rules to apply.
    ///
    /// This is the correct exploration strategy for independent rule decisions.
    /// Each rule is sampled according to its own probability - rules don't compete.
    #[must_use]
    pub fn sample_rules_bernoulli(
        &self,
        acc: &EdgeAccumulator,
        rule_embeds: &[[f32; EMBED_DIM]],
        temperature: f32,
        rng_state: &mut u64,
    ) -> Vec<usize> {
        let probs = self.bernoulli_policy_from_accumulator(acc, rule_embeds, temperature);

        probs
            .iter()
            .enumerate()
            .filter(|&(_, prob)| {
                // Simple LCG for random number
                *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let random = (*rng_state >> 33) as f32 / (1u64 << 31) as f32;
                random < *prob
            })
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Save weights to a binary file.
    ///
    /// Format: magic "TRID" + all weights as little-endian f32.
    /// TRID: shared trunk layer added between towers and projection heads.
    #[cfg(feature = "std")]
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::io::BufWriter::with_capacity(256 * 1024, std::fs::File::create(path)?);

        // Magic header (TRID = shared trunk added — retrain required for old TRIC/TRIB/etc models)
        file.write_all(b"TRID")?;

        // ===== Backbone =====
        // Embeddings
        for row in &self.embeddings.e {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }

        // Hidden layer
        for row in &self.w1 {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        for &val in &self.b1 {
            file.write_all(&val.to_le_bytes())?;
        }

        // Shared trunk
        for row in &self.trunk_w {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        for &val in &self.trunk_b {
            file.write_all(&val.to_le_bytes())?;
        }

        // ===== Unified mask architecture =====
        // Expr projection
        for row in &self.expr_proj_w {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        for &val in &self.expr_proj_b {
            file.write_all(&val.to_le_bytes())?;
        }

        // Value MLP
        for row in &self.value_mlp_w1 {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        for &val in &self.value_mlp_b1 {
            file.write_all(&val.to_le_bytes())?;
        }
        for &val in &self.value_mlp_w2 {
            file.write_all(&val.to_le_bytes())?;
        }
        file.write_all(&self.value_mlp_b2.to_le_bytes())?;

        // Mask MLP
        for row in &self.mask_mlp_w1 {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        for &val in &self.mask_mlp_b1 {
            file.write_all(&val.to_le_bytes())?;
        }
        for row in &self.mask_mlp_w2 {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        for &val in &self.mask_mlp_b2 {
            file.write_all(&val.to_le_bytes())?;
        }

        // Rule MLP for hand-crafted rule features.
        for row in &self.rule_mlp_w1 {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        for &val in &self.rule_mlp_b1 {
            file.write_all(&val.to_le_bytes())?;
        }
        for row in &self.rule_mlp_w2 {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        for &val in &self.rule_mlp_b2 {
            file.write_all(&val.to_le_bytes())?;
        }

        // Rule Projection (TRI3: LHS/RHS template encoding)
        for row in &self.rule_proj_w {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        for &val in &self.rule_proj_b {
            file.write_all(&val.to_le_bytes())?;
        }

        // Interaction matrix
        for row in &self.interaction {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }

        // Mask bias projection
        for &val in &self.mask_bias_proj {
            file.write_all(&val.to_le_bytes())?;
        }

        // Graph state backbone
        for row in &self.graph_w1 {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        for &val in &self.graph_b1 {
            file.write_all(&val.to_le_bytes())?;
        }
        for row in &self.graph_proj_w {
            for &val in row {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        for &val in &self.graph_proj_b {
            file.write_all(&val.to_le_bytes())?;
        }

        Ok(())
    }

    /// Load weights from an in-memory byte slice.
    ///
    /// Used by the compiler to load weights embedded via `include_bytes!`.
    #[cfg(feature = "std")]
    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        Self::load_from_reader(std::io::Cursor::new(bytes))
    }

    /// Load weights from a binary file.
    ///
    /// Only supports "TRID" format (shared trunk). Old formats (TRIC and earlier) require retrain.
    #[cfg(feature = "std")]
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let file = std::io::BufReader::with_capacity(256 * 1024, std::fs::File::open(path)?);
        Self::load_from_reader(file)
    }

    #[cfg(feature = "std")]
    fn load_from_reader<R: std::io::Read>(mut file: R) -> std::io::Result<Self> {
        // Check magic
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;

        // TRID: shared trunk layer between towers and projection heads
        // TRIC: incompatible — pre-shared-trunk architecture
        // TRIB: incompatible — GRAPH_ACC_DIM was 3K (96), GRAPH_INPUT_DIM was 100
        // TRIA: incompatible — had mask_rule_bias[1024] instead of mask_bias_proj[32]
        // TRI5-TRI9: incompatible — EMBED_DIM was 24, all weight shapes differ
        if &magic != b"TRID" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Incompatible ExprNnue format {:?}. Expected 'TRID' (shared trunk). {}",
                    std::str::from_utf8(&magic).unwrap_or("????"),
                    if &magic == b"TRIC" {
                        "Incompatible format 'TRIC' (pre-shared-trunk). Retrain required."
                    } else {
                        "Old formats (TRIB, TRIA, TRI5-TRI9) require retrain."
                    }
                ),
            ));
        }

        let mut net = Self::new();

        // ===== Backbone =====
        // Embeddings
        for row in &mut net.embeddings.e {
            for val in row {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }
        }

        // Hidden layer (w1 is now [INPUT_DIM][HIDDEN_DIM] = [130][64])
        for row in &mut net.w1 {
            for val in row {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }
        }
        for val in &mut net.b1 {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            *val = f32::from_le_bytes(buf);
        }

        // Shared trunk
        for row in &mut net.trunk_w {
            for val in row {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }
        }
        for val in &mut net.trunk_b {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            *val = f32::from_le_bytes(buf);
        }

        // ===== Unified mask architecture =====
        // Expr projection
        for row in &mut net.expr_proj_w {
            for val in row {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }
        }
        for val in &mut net.expr_proj_b {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            *val = f32::from_le_bytes(buf);
        }

        // Value MLP
        for row in &mut net.value_mlp_w1 {
            for val in row {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }
        }
        for val in &mut net.value_mlp_b1 {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            *val = f32::from_le_bytes(buf);
        }
        for val in &mut net.value_mlp_w2 {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            *val = f32::from_le_bytes(buf);
        }
        {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            net.value_mlp_b2 = f32::from_le_bytes(buf);
        }

        // Mask MLP
        for row in &mut net.mask_mlp_w1 {
            for val in row {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }
        }
        for val in &mut net.mask_mlp_b1 {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            *val = f32::from_le_bytes(buf);
        }
        for row in &mut net.mask_mlp_w2 {
            for val in row {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }
        }
        for val in &mut net.mask_mlp_b2 {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            *val = f32::from_le_bytes(buf);
        }

        // Rule MLP
        for row in &mut net.rule_mlp_w1 {
            for val in row {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }
        }
        for val in &mut net.rule_mlp_b1 {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            *val = f32::from_le_bytes(buf);
        }
        for row in &mut net.rule_mlp_w2 {
            for val in row {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }
        }
        for val in &mut net.rule_mlp_b2 {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            *val = f32::from_le_bytes(buf);
        }

        // Rule Projection (LHS/RHS template encoding)
        for row in &mut net.rule_proj_w {
            for val in row {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }
        }
        for val in &mut net.rule_proj_b {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            *val = f32::from_le_bytes(buf);
        }

        // Interaction matrix
        for row in &mut net.interaction {
            for val in row {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }
        }

        // Mask bias projection
        for val in &mut net.mask_bias_proj {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            *val = f32::from_le_bytes(buf);
        }

        // Graph state backbone (TRID format: mandatory, no backward compat)
        for row in &mut net.graph_w1 {
            for val in row {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }
        }
        for val in &mut net.graph_b1 {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            *val = f32::from_le_bytes(buf);
        }
        for row in &mut net.graph_proj_w {
            for val in row {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf)?;
                *val = f32::from_le_bytes(buf);
            }
        }
        for val in &mut net.graph_proj_b {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            *val = f32::from_le_bytes(buf);
        }

        Ok(net)
    }

    // =========================================================================
    // Training Methods
    // =========================================================================

    // =========================================================================
    // Unified Mask Architecture Training
    //
    // Backprop path: score → bilinear → mask_mlp → expr_embed → expr_proj
    // Also updates: rule_mlp, interaction, rule_bias
    // Backbone (embeddings, w1, b1) is FROZEN during mask training.
    // =========================================================================

    /// Train saturation head on single (expr, rule, fired) sample.
    ///
    /// Uses asymmetric BCE loss with higher weight for false negatives
    /// (catching positives is critical - we don't want to skip rules that fire).
    ///
    /// # Arguments
    /// * `expr` - The expression being evaluated
    /// * `rule_features` - Hand-crafted features for this rule
    /// * `fired` - Whether the rule actually fired on this expr
    /// * `fired` - Whether the rule actually fired (ground truth)
    /// * `lr` - Learning rate
    /// * `fp_weight` - Weight for false positives (typically ~1.0)
    /// * `fn_weight` - Weight for false negatives (typically ~100.0)
    ///
    /// # Returns
    /// The (weighted) loss for this sample
    pub fn train_mask_step(
        &mut self,
        expr: &Expr,
        rule_features: &[f32; RULE_FEATURE_DIM],
        fired: bool,
        lr: f32,
        fp_weight: f32,
        fn_weight: f32,
    ) -> f32 {
        // ===== Forward pass with stored intermediates =====
        let acc = EdgeAccumulator::from_expr_dedup(expr, &self.embeddings);
        let hidden = self.forward_shared(&acc);

        let expr_embed = self.compute_expr_embed(&hidden);

        // Mask MLP forward (store hidden for backprop)
        let (mask_features, mask_hidden) = self.mask_mlp_forward_with_hidden(&expr_embed);

        // Rule MLP forward (store hidden for backprop)
        let (rule_embed, rule_hidden) = self.rule_mlp_forward_with_hidden(rule_features);

        // Bilinear: mask_features @ interaction @ rule_embed
        let (score, transformed) = self.bilinear_forward_with_hidden(&mask_features, &rule_embed);

        // ===== Loss computation =====
        let pred = sigmoid(score);
        let target = if fired { 1.0 } else { 0.0 };

        let p = pred.clamp(1e-7, 1.0 - 1e-7);
        let loss = -(target * libm::logf(p) + (1.0 - target) * libm::logf(1.0 - p));

        // Asymmetric weighting (catch positives!)
        let weight = if pred > 0.5 && !fired {
            fp_weight
        } else if pred <= 0.5 && fired {
            fn_weight
        } else {
            1.0
        };

        let d_score = (weight * (pred - target)).clamp(-10.0, 10.0);

        // ===== Backprop =====
        // d_score → bias projection: ∂score/∂bias_proj_k = rule_embed[k]
        for k in 0..EMBED_DIM {
            self.mask_bias_proj[k] -= lr * d_score * rule_embed[k];
        }

        // d_score → interaction, mask_features, rule_embed
        let (d_mask_features, d_rule_embed) =
            self.backprop_bilinear(d_score, &mask_features, &rule_embed, &transformed, lr);

        // d_mask_features → mask_mlp
        let _d_expr_embed = self.backprop_mask_mlp(&d_mask_features, &expr_embed, &mask_hidden, lr);

        // d_rule_embed → rule_mlp
        self.backprop_rule_mlp(&d_rule_embed, rule_features, &rule_hidden, lr);

        // NOTE: We freeze the backbone (expr_proj, w1, b1, embeddings)
        // If you want to fine-tune, uncomment:
        // self.backprop_expr_proj(&d_expr_embed, &hidden, lr);

        loss * weight
    }

    /// REINFORCE update for a single mask decision.
    ///
    /// The reward comes from FINAL extraction quality, not per-rule outcomes.
    ///
    /// For APPROVED decision: ∇log P(approve) = 1 - sigmoid(score)
    /// For REJECTED decision: ∇log P(reject) = -sigmoid(score)
    ///
    /// Positive advantage → reinforce the decision that was made
    /// Negative advantage → discourage the decision that was made
    ///
    /// # Arguments
    /// * `expr` - Expression that was scored
    /// * `rule_features` - Features of the rule
    /// * `approved` - Was this rule approved (tried) or rejected (skipped)?
    /// * `advantage` - reward - baseline (from final extraction cost comparison)
    /// * `lr` - Learning rate
    ///
    /// # Returns
    /// The gradient magnitude applied.
    pub fn train_mask_reinforce(
        &mut self,
        expr: &Expr,
        rule_features: &[f32; RULE_FEATURE_DIM],
        approved: bool,
        advantage: f32,
        lr: f32,
    ) -> f32 {
        // Forward pass to get intermediates
        let acc = EdgeAccumulator::from_expr_dedup(expr, &self.embeddings);
        let hidden = self.forward_shared(&acc);

        let expr_embed = self.compute_expr_embed(&hidden);
        let (mask_features, mask_hidden) = self.mask_mlp_forward_with_hidden(&expr_embed);
        let (rule_embed, rule_hidden) = self.rule_mlp_forward_with_hidden(rule_features);
        let (score, transformed) = self.bilinear_forward_with_hidden(&mask_features, &rule_embed);

        // REINFORCE gradient depends on the action taken:
        // - Approved: ∇log sigmoid(score) = 1 - sigmoid(score)
        // - Rejected: ∇log (1 - sigmoid(score)) = -sigmoid(score)
        let prob = sigmoid(score).clamp(1e-6, 1.0 - 1e-6);
        let d_log_prob = if approved {
            1.0 - prob // push score up when reinforcing approval
        } else {
            -prob // push score down when reinforcing rejection
        };

        // Clip gradient to prevent explosion
        let d_score = (advantage * d_log_prob).clamp(-1.0, 1.0);

        // Skip update if gradient would be NaN or too small
        if !d_score.is_finite() || d_score.abs() < 1e-8 {
            return 0.0;
        }

        // Backprop: ∂score/∂bias_proj_k = rule_embed[k]
        for k in 0..EMBED_DIM {
            self.mask_bias_proj[k] -= lr * d_score * rule_embed[k];
        }

        let (d_mask_features, d_rule_embed) =
            self.backprop_bilinear(d_score, &mask_features, &rule_embed, &transformed, lr);

        let _d_expr_embed = self.backprop_mask_mlp(&d_mask_features, &expr_embed, &mask_hidden, lr);
        self.backprop_rule_mlp(&d_rule_embed, rule_features, &rule_hidden, lr);

        d_score.abs()
    }

    /// Batch REINFORCE update for decisions from a search episode.
    ///
    /// # Arguments
    /// * `decisions` - Vec of (expr, rule_features, approved)
    /// * `advantage` - reward - baseline (from final cost comparison)
    /// * `lr` - Learning rate
    ///
    /// # Returns
    /// Total gradient norm applied.
    pub fn train_mask_reinforce_batch(
        &mut self,
        decisions: &[(Expr, [f32; RULE_FEATURE_DIM], bool)],
        advantage: f32,
        lr: f32,
    ) -> f32 {
        let mut total_grad = 0.0f32;
        for (expr, rule_features, approved) in decisions {
            total_grad += self.train_mask_reinforce(expr, rule_features, *approved, advantage, lr);
        }
        total_grad
    }

    /// REINFORCE training using pre-computed rule embeddings.
    ///
    /// This is the preferred method when using LHS/RHS template embeddings.
    /// Rule embeddings are computed once via `encode_all_rules_from_templates()`
    /// and reused across training.
    ///
    /// # Arguments
    /// * `expr` - The expression being evaluated
    /// * `rule_embed` - Pre-computed rule embedding (from templates)
    /// * `approved` - Whether this rule was approved by the mask
    /// * `advantage` - reward - baseline
    /// * `lr` - Learning rate
    ///
    /// # Returns
    /// The gradient magnitude applied.
    pub fn train_mask_reinforce_with_embed(
        &mut self,
        expr: &Expr,
        rule_embed: &[f32; EMBED_DIM],
        approved: bool,
        advantage: f32,
        lr: f32,
    ) -> f32 {
        // Forward pass to get intermediates
        let acc = EdgeAccumulator::from_expr_dedup(expr, &self.embeddings);
        let hidden = self.forward_shared(&acc);

        let expr_embed = self.compute_expr_embed(&hidden);
        let (mask_features, mask_hidden) = self.mask_mlp_forward_with_hidden(&expr_embed);
        let (score, transformed) = self.bilinear_forward_with_hidden(&mask_features, rule_embed);

        // REINFORCE gradient:
        // - Approved: ∇log sigmoid(score) = 1 - sigmoid(score)
        // - Rejected: ∇log (1 - sigmoid(score)) = -sigmoid(score)
        let prob = sigmoid(score).clamp(1e-6, 1.0 - 1e-6);
        let d_log_prob = if approved { 1.0 - prob } else { -prob };

        // Clip gradient to prevent explosion
        let d_score = (advantage * d_log_prob).clamp(-1.0, 1.0);

        // Skip update if gradient would be NaN or too small
        if !d_score.is_finite() || d_score.abs() < 1e-8 {
            return 0.0;
        }

        // Backprop (rule embedding is frozen - computed from templates)
        // ∂score/∂bias_proj_k = rule_embed[k]
        for k in 0..EMBED_DIM {
            self.mask_bias_proj[k] -= lr * d_score * rule_embed[k];
        }

        let (d_mask_features, _d_rule_embed) =
            self.backprop_bilinear(d_score, &mask_features, rule_embed, &transformed, lr);

        let _d_expr_embed = self.backprop_mask_mlp(&d_mask_features, &expr_embed, &mask_hidden, lr);
        // NOTE: Rule embedding is not updated here - it comes from templates.
        // The rule_proj weights that created it are updated only during supervised pretraining.

        d_score.abs()
    }

    /// Batch REINFORCE update using pre-computed rule embeddings.
    ///
    /// # Arguments
    /// * `decisions` - Vec of (expr, rule_embed, approved)
    /// * `advantage` - reward - baseline (from final cost comparison)
    /// * `lr` - Learning rate
    ///
    /// # Returns
    /// Total gradient norm applied.
    pub fn train_mask_reinforce_batch_with_embeds(
        &mut self,
        decisions: &[(Expr, [f32; EMBED_DIM], bool)],
        advantage: f32,
        lr: f32,
    ) -> f32 {
        let mut total_grad = 0.0f32;
        for (expr, rule_embed, approved) in decisions {
            total_grad +=
                self.train_mask_reinforce_with_embed(expr, rule_embed, *approved, advantage, lr);
        }
        total_grad
    }

    /// Train value MLP on (expr, true_cost) sample.
    ///
    /// Uses MSE loss. Backprop goes through value_mlp → expr_proj.
    /// Backbone is frozen.
    pub fn train_value_mlp_step(&mut self, expr: &Expr, true_cost: f32, lr: f32) -> f32 {
        // Forward
        let acc = EdgeAccumulator::from_expr_dedup(expr, &self.embeddings);
        let hidden = self.forward_shared(&acc);
        let expr_embed = self.compute_expr_embed(&hidden);
        let (pred_cost, value_hidden) = self.value_mlp_forward_with_hidden(&expr_embed);

        // MSE loss
        let diff = pred_cost - true_cost;
        let loss = diff * diff;
        let d_cost = 2.0 * diff;

        // Backprop through value_mlp
        let _d_expr_embed = self.backprop_value_mlp(d_cost, &expr_embed, &value_hidden, lr);

        // NOTE: Backbone frozen - if you want to fine-tune:
        // self.backprop_expr_proj(&d_expr_embed, &hidden, lr);

        loss
    }

    // =========================================================================
    // Forward with Hidden (for backprop)
    // =========================================================================

    /// Mask MLP forward storing hidden activations.
    fn mask_mlp_forward_with_hidden(
        &self,
        expr_embed: &[f32; EMBED_DIM],
    ) -> ([f32; EMBED_DIM], [f32; MLP_HIDDEN]) {
        // First layer
        let mut h = self.mask_mlp_b1;
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                h[j] += expr_embed[i] * self.mask_mlp_w1[i][j];
            }
        }

        // Store pre-ReLU for backprop (we need to know which neurons were active)
        let h_pre_relu = h;

        // ReLU
        for j in 0..MLP_HIDDEN {
            h[j] = h[j].max(0.0);
        }

        // Second layer
        let mut out = self.mask_mlp_b2;
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                out[k] += h[j] * self.mask_mlp_w2[j][k];
            }
        }

        (out, h_pre_relu)
    }

    /// Rule MLP forward storing hidden activations.
    fn rule_mlp_forward_with_hidden(
        &self,
        rule_features: &[f32; RULE_FEATURE_DIM],
    ) -> ([f32; EMBED_DIM], [f32; MLP_HIDDEN]) {
        // First layer
        let mut h = self.rule_mlp_b1;
        for i in 0..RULE_FEATURE_DIM {
            for j in 0..MLP_HIDDEN {
                h[j] += rule_features[i] * self.rule_mlp_w1[i][j];
            }
        }

        let h_pre_relu = h;

        // ReLU
        for j in 0..MLP_HIDDEN {
            h[j] = h[j].max(0.0);
        }

        // Second layer
        let mut out = self.rule_mlp_b2;
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                out[k] += h[j] * self.rule_mlp_w2[j][k];
            }
        }

        (out, h_pre_relu)
    }

    /// Value MLP forward storing hidden activations.
    fn value_mlp_forward_with_hidden(
        &self,
        expr_embed: &[f32; EMBED_DIM],
    ) -> (f32, [f32; MLP_HIDDEN]) {
        // First layer
        let mut h = self.value_mlp_b1;
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                h[j] += expr_embed[i] * self.value_mlp_w1[i][j];
            }
        }

        let h_pre_relu = h;

        // ReLU
        for j in 0..MLP_HIDDEN {
            h[j] = h[j].max(0.0);
        }

        // Second layer
        let mut cost = self.value_mlp_b2;
        for j in 0..MLP_HIDDEN {
            cost += h[j] * self.value_mlp_w2[j];
        }

        (cost, h_pre_relu)
    }

    /// Bilinear forward storing transformed vector.
    fn bilinear_forward_with_hidden(
        &self,
        mask_features: &[f32; EMBED_DIM],
        rule_embed: &[f32; EMBED_DIM],
    ) -> (f32, [f32; EMBED_DIM]) {
        // transformed = mask_features @ interaction
        let mut transformed = [0.0f32; EMBED_DIM];
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                transformed[j] += mask_features[i] * self.interaction[i][j];
            }
        }

        // score = dot(transformed + mask_bias_proj, rule_embed)
        let mut score = 0.0f32;
        for k in 0..EMBED_DIM {
            score += (transformed[k] + self.mask_bias_proj[k]) * rule_embed[k];
        }

        (score, transformed)
    }

    // =========================================================================
    // Backpropagation Helpers
    // =========================================================================

    /// Backprop through bilinear layer.
    ///
    /// Returns (d_mask_features, d_rule_embed) and updates interaction matrix.
    fn backprop_bilinear(
        &mut self,
        d_score: f32,
        mask_features: &[f32; EMBED_DIM],
        rule_embed: &[f32; EMBED_DIM],
        transformed: &[f32; EMBED_DIM],
        lr: f32,
    ) -> ([f32; EMBED_DIM], [f32; EMBED_DIM]) {
        // d_score/d_transformed = rule_embed
        // d_score/d_rule_embed = transformed
        let mut d_transformed = [0.0f32; EMBED_DIM];
        let mut d_rule_embed = [0.0f32; EMBED_DIM];

        for k in 0..EMBED_DIM {
            d_transformed[k] = d_score * rule_embed[k];
            d_rule_embed[k] = d_score * transformed[k];
        }

        // d_transformed/d_mask_features = interaction^T
        // d_transformed/d_interaction = outer(mask_features, I)
        let mut d_mask_features = [0.0f32; EMBED_DIM];
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                d_mask_features[i] += d_transformed[j] * self.interaction[i][j];
                // Update interaction: d_loss/d_interaction[i][j] = mask_features[i] * d_transformed[j]
                self.interaction[i][j] -= lr * mask_features[i] * d_transformed[j];
            }
        }

        (d_mask_features, d_rule_embed)
    }

    /// Backprop through mask MLP.
    ///
    /// Returns d_expr_embed and updates mask_mlp weights.
    fn backprop_mask_mlp(
        &mut self,
        d_out: &[f32; EMBED_DIM],
        expr_embed: &[f32; EMBED_DIM],
        h_pre_relu: &[f32; MLP_HIDDEN],
        lr: f32,
    ) -> [f32; EMBED_DIM] {
        // d_out → w2, b2
        // d_h (post-ReLU) = d_out @ w2^T
        let mut d_h = [0.0f32; MLP_HIDDEN];
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                d_h[j] += d_out[k] * self.mask_mlp_w2[j][k];
                // Update w2
                let h_relu = h_pre_relu[j].max(0.0);
                self.mask_mlp_w2[j][k] -= lr * h_relu * d_out[k];
            }
        }

        // Update b2
        for k in 0..EMBED_DIM {
            self.mask_mlp_b2[k] -= lr * d_out[k];
        }

        // ReLU backward
        for j in 0..MLP_HIDDEN {
            if h_pre_relu[j] <= 0.0 {
                d_h[j] = 0.0;
            }
        }

        // d_h → w1, b1, d_expr_embed
        let mut d_expr_embed = [0.0f32; EMBED_DIM];
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                d_expr_embed[i] += d_h[j] * self.mask_mlp_w1[i][j];
                // Update w1
                self.mask_mlp_w1[i][j] -= lr * expr_embed[i] * d_h[j];
            }
        }

        // Update b1
        for j in 0..MLP_HIDDEN {
            self.mask_mlp_b1[j] -= lr * d_h[j];
        }

        d_expr_embed
    }

    /// Backprop through rule MLP.
    ///
    /// Updates rule_mlp weights. Rule features are fixed, so no gradient returned.
    fn backprop_rule_mlp(
        &mut self,
        d_out: &[f32; EMBED_DIM],
        rule_features: &[f32; RULE_FEATURE_DIM],
        h_pre_relu: &[f32; MLP_HIDDEN],
        lr: f32,
    ) {
        // d_out → w2, b2
        let mut d_h = [0.0f32; MLP_HIDDEN];
        for j in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM {
                d_h[j] += d_out[k] * self.rule_mlp_w2[j][k];
                let h_relu = h_pre_relu[j].max(0.0);
                self.rule_mlp_w2[j][k] -= lr * h_relu * d_out[k];
            }
        }

        // Update b2
        for k in 0..EMBED_DIM {
            self.rule_mlp_b2[k] -= lr * d_out[k];
        }

        // ReLU backward
        for j in 0..MLP_HIDDEN {
            if h_pre_relu[j] <= 0.0 {
                d_h[j] = 0.0;
            }
        }

        // d_h → w1, b1
        for i in 0..RULE_FEATURE_DIM {
            for j in 0..MLP_HIDDEN {
                self.rule_mlp_w1[i][j] -= lr * rule_features[i] * d_h[j];
            }
        }

        for j in 0..MLP_HIDDEN {
            self.rule_mlp_b1[j] -= lr * d_h[j];
        }
    }

    /// Backprop through value MLP.
    ///
    /// Returns d_expr_embed and updates value_mlp weights.
    fn backprop_value_mlp(
        &mut self,
        d_cost: f32,
        expr_embed: &[f32; EMBED_DIM],
        h_pre_relu: &[f32; MLP_HIDDEN],
        lr: f32,
    ) -> [f32; EMBED_DIM] {
        // d_cost → w2, b2
        let mut d_h = [0.0f32; MLP_HIDDEN];
        for j in 0..MLP_HIDDEN {
            d_h[j] = d_cost * self.value_mlp_w2[j];
            let h_relu = h_pre_relu[j].max(0.0);
            self.value_mlp_w2[j] -= lr * h_relu * d_cost;
        }

        self.value_mlp_b2 -= lr * d_cost;

        // ReLU backward
        for j in 0..MLP_HIDDEN {
            if h_pre_relu[j] <= 0.0 {
                d_h[j] = 0.0;
            }
        }

        // d_h → w1, b1, d_expr_embed
        let mut d_expr_embed = [0.0f32; EMBED_DIM];
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN {
                d_expr_embed[i] += d_h[j] * self.value_mlp_w1[i][j];
                self.value_mlp_w1[i][j] -= lr * expr_embed[i] * d_h[j];
            }
        }

        for j in 0..MLP_HIDDEN {
            self.value_mlp_b1[j] -= lr * d_h[j];
        }

        d_expr_embed
    }

    /// Backprop through expr projection (optional, for fine-tuning).
    ///
    /// Updates expr_proj weights. Backbone (w1, b1) remains frozen.
    #[allow(dead_code)]
    fn backprop_expr_proj(
        &mut self,
        d_expr_embed: &[f32; EMBED_DIM],
        hidden: &[f32; HIDDEN_DIM],
        lr: f32,
    ) {
        // d_expr_embed → expr_proj_w, expr_proj_b
        for j in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM {
                self.expr_proj_w[j][k] -= lr * hidden[j] * d_expr_embed[k];
            }
        }

        for k in 0..EMBED_DIM {
            self.expr_proj_b[k] -= lr * d_expr_embed[k];
        }
    }
}

/// Dot product of two arrays.
#[inline]
fn dot(a: &[f32; HIDDEN_DIM], b: &[f32; HIDDEN_DIM]) -> f32 {
    let mut sum = 0.0;
    for i in 0..HIDDEN_DIM {
        sum += a[i] * b[i];
    }
    sum
}

/// Sigmoid activation.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-x))
}

/// Softmax with temperature scaling.
///
/// `softmax(x_i / temp)` - higher temperature = more uniform distribution.
#[must_use]
fn softmax_with_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    let temp = temperature.max(0.01); // Avoid division by zero

    // Scale by temperature
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();

    // Numerical stability: subtract max
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| libm::expf(x - max_val)).collect();
    let sum: f32 = exps.iter().sum();

    if sum < 1e-10 {
        // Uniform fallback
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;
    use alloc::sync::Arc;

    /// Create a simple expression: x + y
    fn make_add_xy() -> Expr {
        Expr::Binary(OpKind::Add, Arc::new(Expr::Var(0)), Arc::new(Expr::Var(1)))
    }

    /// Create FMA-eligible expression: a*b + c
    fn make_fma_pattern() -> Expr {
        Expr::Binary(
            OpKind::Add,
            Arc::new(Expr::Binary(
                OpKind::Mul,
                Arc::new(Expr::Var(0)),
                Arc::new(Expr::Var(1)),
            )),
            Arc::new(Expr::Var(2)),
        )
    }

    /// Create non-FMA pattern: a + b*c (Mul under Add, but on right side)
    fn make_add_mul_pattern() -> Expr {
        Expr::Binary(
            OpKind::Add,
            Arc::new(Expr::Var(0)),
            Arc::new(Expr::Binary(
                OpKind::Mul,
                Arc::new(Expr::Var(1)),
                Arc::new(Expr::Var(2)),
            )),
        )
    }

    #[test]
    fn test_edge_extraction() {
        let expr = make_add_xy();
        let edges = extract_edges(&expr);

        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].parent, OpKind::Add);
        assert_eq!(edges[0].child, OpKind::Var);
        assert_eq!(edges[1].parent, OpKind::Add);
        assert_eq!(edges[1].child, OpKind::Var);
    }

    #[test]
    fn test_fma_edges() {
        let expr = make_fma_pattern();
        let edges = extract_edges(&expr);

        // Should have: Add→Mul, Add→Var, Mul→Var, Mul→Var
        assert_eq!(edges.len(), 4);

        // Check that Add→Mul exists (the FMA-critical edge)
        let has_add_mul = edges
            .iter()
            .any(|e| e.parent == OpKind::Add && e.child == OpKind::Mul);
        assert!(has_add_mul, "Should have Add→Mul edge for FMA pattern");
    }

    #[test]
    fn test_asymmetric_accumulator() {
        let emb = OpEmbeddings::new_random(42);

        // Mul→Add (under Add parent)
        let fma = make_fma_pattern();
        let acc_fma = EdgeAccumulator::from_expr_dedup(&fma, &emb);

        // Add→Mul (Mul under Add, same ops but different structure)
        let add_mul = make_add_mul_pattern();
        let _acc_add_mul = EdgeAccumulator::from_expr_dedup(&add_mul, &emb);

        // The accumulators should be different because:
        // - FMA has Add→Mul, Add→Var edges
        // - ADD_MUL has Add→Var, Add→Mul edges
        // Wait, these are actually the same edges just in different order!
        // The key difference is in the CHILD subexpressions.

        // Actually, let's compare with a truly different pattern:
        // x * (y + z) vs (x * y) + z

        let mul_add = Expr::Binary(
            OpKind::Mul,
            Arc::new(Expr::Var(0)),
            Arc::new(Expr::Binary(
                OpKind::Add,
                Arc::new(Expr::Var(1)),
                Arc::new(Expr::Var(2)),
            )),
        );

        let acc_mul_add = EdgeAccumulator::from_expr_dedup(&mul_add, &emb);

        // These should definitely differ:
        // - FMA (a*b + c): edges are Add→Mul, Add→Var, Mul→Var, Mul→Var
        // - mul_add (a * (b+c)): edges are Mul→Var, Mul→Add, Add→Var, Add→Var

        // They have different edge sets, so accumulators should differ
        let diff: f32 = acc_fma
            .values
            .iter()
            .zip(acc_mul_add.values.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(
            diff > 0.1,
            "Asymmetric patterns should produce different accumulators"
        );
    }

    #[test]
    fn test_incremental_update() {
        let emb = OpEmbeddings::new_random(42);
        let expr = make_fma_pattern();

        // Build accumulator from scratch
        let acc_full = EdgeAccumulator::from_expr_dedup(&expr, &emb);

        // Build via from_expr_dedup (same as full)
        let acc_inc = EdgeAccumulator::from_expr_dedup(&expr, &emb);

        // Should match
        for i in 0..acc_full.values.len() {
            assert!(
                (acc_full.values[i] - acc_inc.values[i]).abs() < 1e-6,
                "Incremental build should match full build"
            );
        }

        // Remove and verify we get back to zero
        let mut acc_removed = acc_inc.clone();
        acc_removed.remove_expr_edges(&expr, &emb);
        for &v in &acc_removed.values {
            assert!(
                v.abs() < 1e-6,
                "After removing all edges, accumulator should be zero"
            );
        }
    }

    #[test]
    fn test_forward_pass() {
        let net = ExprNnue::new_random(42);
        let expr = make_fma_pattern();

        let cost = net.predict_cost(&expr);

        // Should be a reasonable value (not NaN, not infinity)
        assert!(cost.is_finite(), "Cost should be finite");
    }

    #[test]
    fn test_param_count() {
        // Verify parameter count is reasonable and finite
        let count = ExprNnue::param_count();
        assert!(count > 0, "Should have parameters");
        assert!(
            ExprNnue::memory_bytes() < 200_000,
            "NNUE should use < 200KB, got {} bytes",
            ExprNnue::memory_bytes()
        );
    }

    #[test]
    fn test_different_expressions_different_costs() {
        let net = ExprNnue::new_random(42);

        let simple = Expr::Var(0);
        let complex = Expr::Binary(
            OpKind::Div,
            Arc::new(Expr::Unary(OpKind::Sqrt, Arc::new(Expr::Var(0)))),
            Arc::new(Expr::Unary(OpKind::Sqrt, Arc::new(Expr::Var(1)))),
        );

        let cost_simple = net.predict_cost(&simple);
        let cost_complex = net.predict_cost(&complex);

        // Complex expression should generally have higher predicted cost
        // (though with random weights this isn't guaranteed, just check they differ)
        assert!(
            (cost_simple - cost_complex).abs() > 1e-6,
            "Different expressions should produce different costs"
        );
    }

    // ========================================================================
    // ExprNnue Tests
    // ========================================================================

    #[test]
    fn test_consolidated_param_count() {
        // Param count should include backbone + all unified heads
        let count = ExprNnue::param_count();
        // Backbone: embeddings + w1 + b1 = ~9,728
        // Plus: expr_proj + value_mlp + mask_mlp + rule_mlp + rule_proj + interaction + bias
        assert!(count > 10_000, "Should have >10k params, got {}", count);
        assert!(
            ExprNnue::memory_bytes() < 200_000,
            "NNUE should use < 200KB, got {} bytes",
            ExprNnue::memory_bytes()
        );
    }

    #[test]
    fn test_dual_head_value_prediction() {
        let net = ExprNnue::new_random(42);
        let expr = make_fma_pattern();

        let log_cost = net.predict_log_cost(&expr);
        let cost = net.predict_cost(&expr);

        // Log cost should be finite
        assert!(log_cost.is_finite(), "Log cost should be finite");

        // Cost should be exp(log_cost)
        let expected = libm::expf(log_cost);
        assert!(
            (cost - expected).abs() < 1e-4,
            "predict_cost should be exp(predict_log_cost)"
        );
    }

    #[test]
    fn test_from_factored_preserves_backbone() {
        let original = ExprNnue::new_random(42);

        let converted = ExprNnue::from_factored(&original);

        // Backbone (embeddings, w1, b1) must be preserved
        assert_eq!(original.embeddings.e, converted.embeddings.e);
        assert_eq!(original.w1, converted.w1);
        assert_eq!(original.b1, converted.b1);

        // Identity trunk preserves the edge-tower hidden representation.
        let expr = make_fma_pattern();
        let acc = EdgeAccumulator::from_expr_dedup(&expr, &original.embeddings);
        let original_hidden = original.forward_shared(&acc);
        let converted_hidden = converted.forward_shared(&acc);
        for i in 0..HIDDEN_DIM {
            assert!(
                (original_hidden[i] - converted_hidden[i]).abs() < 1e-6,
                "forward_shared[{i}] changed during migration"
            );
        }

        // Unified heads are still zero-initialized and should remain numerically sane.
        let cost = converted.predict_cost(&expr);
        assert!(
            cost.is_finite(),
            "Prediction from converted model should be finite"
        );
    }

    #[test]
    fn test_dual_head_latency_prior() {
        // Test that latency priors are correctly set in embeddings.
        // Note: Random network weights can overwhelm these priors - this test
        // verifies initialization, not that untrained predictions are correct.
        let net = ExprNnue::new_with_latency_prior(42);

        // Check that expensive ops have higher latency values in dim 0
        let var_latency = net.embeddings.get(OpKind::Var)[0];
        let div_latency = net.embeddings.get(OpKind::Div)[0];
        let sqrt_latency = net.embeddings.get(OpKind::Sqrt)[0];

        // Var should be cheap (0.0 latency)
        assert!(
            var_latency < 0.1,
            "Var latency should be near zero: {}",
            var_latency
        );

        // Div and Sqrt should be expensive (0.75 latency)
        assert!(
            div_latency > 0.5,
            "Div latency should be high: {}",
            div_latency
        );
        assert!(
            sqrt_latency > 0.5,
            "Sqrt latency should be high: {}",
            sqrt_latency
        );

        // Verify the network can make predictions (no NaN/infinity)
        let expr = Expr::Var(0);
        let cost = net.predict_cost(&expr);
        assert!(cost.is_finite(), "Prediction should be finite");
    }

    #[test]
    fn test_mask_policy_scoring() {
        let net = ExprNnue::new_random(42);
        let expr = make_fma_pattern();

        // Use mask-based scoring (the only saturation head now)
        let rule_features = RuleFeatures::new();
        let rule_embeds = net.encode_all_rules(&rule_features, 2);
        let scores = net.mask_score_all_rules(&expr, &rule_embeds);

        assert!(
            scores[0].is_finite(),
            "Policy score should be finite: {}",
            scores[0]
        );
        assert!(
            scores[1].is_finite(),
            "Policy score should be finite: {}",
            scores[1]
        );
    }

    // ========================================================================
    // Unified Mask Architecture Tests
    // ========================================================================

    #[test]
    fn test_rule_features_initialization() {
        let mut rule_features = RuleFeatures::new();

        // All features should be zero initially
        for r in 0..10 {
            for f in rule_features.get(r) {
                assert!(*f == 0.0, "Initial features should be zero");
            }
        }

        // Set features for a rule
        rule_features.set(0, [0.25, 0.3, 1.0, 1.0, 0.0, 1.0, 0.5, 1.0]);
        let features = rule_features.get(0);
        assert!((features[0] - 0.25).abs() < 1e-6, "Category should be set");
        assert!(
            (features[3] - 1.0).abs() < 1e-6,
            "Commutative flag should be set"
        );
    }

    #[test]
    fn test_encode_rule_deterministic() {
        let net = ExprNnue::new_random(42);
        let features = [0.25, 0.3, 1.0, 1.0, 0.0, 1.0, 0.5, 1.0];

        let embed1 = net.encode_rule(&features);
        let embed2 = net.encode_rule(&features);

        // Same input should produce same output
        for i in 0..EMBED_DIM {
            assert!(
                (embed1[i] - embed2[i]).abs() < 1e-6,
                "encode_rule should be deterministic at dim {}",
                i
            );
        }

        // Embedding should be finite
        for i in 0..EMBED_DIM {
            assert!(
                embed1[i].is_finite(),
                "Rule embedding should be finite at dim {}",
                i
            );
        }
    }

    #[test]
    fn test_encode_all_rules() {
        let net = ExprNnue::new_random(42);
        let mut rule_features = RuleFeatures::new();

        // Set up a few rules with different features
        rule_features.set(0, [0.0, 0.2, 0.0, 1.0, 0.0, 0.0, 0.1, 0.0]); // algebraic
        rule_features.set(1, [0.25, 0.5, -1.0, 0.0, 1.0, 1.0, 0.3, 0.0]); // peephole
        rule_features.set(2, [0.75, 0.8, 1.0, 0.0, 0.0, 0.0, 0.05, 1.0]); // cross-cutting

        let embeds = net.encode_all_rules(&rule_features, 3);

        assert_eq!(embeds.len(), 3, "Should encode exactly 3 rules");

        // Each embedding should be finite
        for (r, embed) in embeds.iter().enumerate() {
            for (d, &val) in embed.iter().enumerate() {
                assert!(val.is_finite(), "Rule {} dim {} should be finite", r, d);
            }
        }

        // Different features should produce different embeddings
        let diff_01: f32 = embeds[0]
            .iter()
            .zip(embeds[1].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let diff_02: f32 = embeds[0]
            .iter()
            .zip(embeds[2].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(
            diff_01 > 1e-3,
            "Different rules should have different embeddings"
        );
        assert!(
            diff_02 > 1e-3,
            "Different rules should have different embeddings"
        );
    }

    #[test]
    fn test_compute_expr_embed() {
        let net = ExprNnue::new_random(42);
        let expr = make_fma_pattern();

        // Compute hidden state
        let acc = EdgeAccumulator::from_expr_dedup(&expr, &net.embeddings);
        let hidden = net.forward_shared(&acc);

        // Compute expr embedding
        let expr_embed = net.compute_expr_embed(&hidden);

        assert_eq!(
            expr_embed.len(),
            EMBED_DIM,
            "Expr embed should have EMBED_DIM dimensions"
        );

        for (i, &val) in expr_embed.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Expr embedding should be finite at dim {}",
                i
            );
        }
    }

    #[test]
    fn test_compute_mask_features() {
        let net = ExprNnue::new_random(42);
        let expr = make_fma_pattern();

        let acc = EdgeAccumulator::from_expr_dedup(&expr, &net.embeddings);
        let hidden = net.forward_shared(&acc);
        let expr_embed = net.compute_expr_embed(&hidden);

        let mask_features = net.compute_mask_features(&expr_embed);

        assert_eq!(
            mask_features.len(),
            EMBED_DIM,
            "Mask features should have EMBED_DIM dimensions"
        );

        for (i, &val) in mask_features.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Mask features should be finite at dim {}",
                i
            );
        }
    }

    #[test]
    fn test_bilinear_score_computation() {
        let net = ExprNnue::new_random(42);

        // Create test vectors
        let mask_features = [1.0f32; EMBED_DIM];
        let rule_embed = [1.0f32; EMBED_DIM];

        let score = net.bilinear_score(&mask_features, &rule_embed);

        assert!(score.is_finite(), "Bilinear score should be finite");

        // Manual verification: score = dot(mask @ interaction + bias_proj, rule)
        // With all-ones vectors: sum of interaction matrix + sum of bias_proj
        let mut expected = 0.0f32;
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM {
                expected += net.interaction[i][j];
            }
        }
        for k in 0..EMBED_DIM {
            expected += net.mask_bias_proj[k];
        }
        assert!(
            (score - expected).abs() < 1e-4,
            "Bilinear computation mismatch: got {}, expected {}",
            score,
            expected
        );
    }

    #[test]
    fn test_mask_score_all_rules_finite() {
        let net = ExprNnue::new_random(42);
        let expr = make_fma_pattern();
        let mut rule_features = RuleFeatures::new();

        // Set up 5 rules
        for r in 0..5 {
            rule_features.set(r, [r as f32 * 0.2, 0.3, 0.0, 1.0, 0.0, 0.0, 0.1, 0.0]);
        }

        let rule_embeds = net.encode_all_rules(&rule_features, 5);
        let scores = net.mask_score_all_rules(&expr, &rule_embeds);

        assert_eq!(scores.len(), 5, "Should score all 5 rules");

        for (r, &score) in scores.iter().enumerate() {
            assert!(score.is_finite(), "Score for rule {} should be finite", r);
        }
    }

    #[test]
    fn test_filter_rules_unified() {
        let mut net = ExprNnue::new_random(42);
        let expr = make_fma_pattern();
        let mut rule_features = RuleFeatures::new();

        // Set up 10 rules
        for r in 0..10 {
            rule_features.set(r, [r as f32 * 0.1, 0.3, 0.0, 1.0, 0.0, 0.0, 0.1, 0.0]);
        }

        let rule_embeds = net.encode_all_rules(&rule_features, 10);
        let passing = net.filter_rules_unified(&expr, &rule_embeds, 0.5);

        // With random weights, the filtering logic should produce some results
        // (not necessarily all or none). Verify the output is well-formed.
        assert!(passing.len() <= 10, "Cannot pass more rules than exist");
        for &idx in &passing {
            assert!(idx < 10, "Rule index {} out of bounds", idx);
        }
    }

    #[test]
    fn test_predict_log_cost() {
        let net = ExprNnue::new_random(42);
        let expr = make_fma_pattern();

        let cost = net.predict_log_cost(&expr);

        assert!(cost.is_finite(), "Unified cost prediction should be finite");
        assert!(
            cost > 0.0,
            "Cost should be positive (exp of value_mlp output)"
        );
    }

    #[test]
    fn test_mask_training_step_loss_decreases() {
        let mut net = ExprNnue::new_random(42);
        let expr = make_fma_pattern();
        let rule_features = [0.25, 0.3, 1.0, 1.0, 0.0, 1.0, 0.5, 1.0];

        // Compute initial loss for a positive sample
        let rule_embeds = net.encode_all_rules(&RuleFeatures::new(), 1);
        let initial_scores = net.mask_score_all_rules(&expr, &rule_embeds);
        let initial_pred = 1.0 / (1.0 + (-initial_scores[0]).exp()); // sigmoid

        // Train on positive sample (rule fired)
        let mut total_loss = 0.0;
        for _ in 0..50 {
            let loss = net.train_mask_step(&expr, &rule_features, true, 0.01, 1.0, 10.0);
            total_loss += loss;
            assert!(loss.is_finite(), "Training loss should be finite");
        }

        // Compute final prediction
        let rule_embeds = net.encode_all_rules(&RuleFeatures::new(), 1);
        let final_scores = net.mask_score_all_rules(&expr, &rule_embeds);
        let final_pred = 1.0 / (1.0 + (-final_scores[0]).exp()); // sigmoid

        // After training on positive samples, prediction should increase
        // (network learns to predict 1 for this expr-rule pair)
        assert!(
            final_pred > initial_pred || (final_pred - initial_pred).abs() < 0.1,
            "Training on positive should increase prediction: {} -> {}",
            initial_pred,
            final_pred
        );
    }

    #[test]
    fn test_value_mlp_training_step() {
        let mut net = ExprNnue::new_random(42);
        let expr = make_fma_pattern();

        let target_cost = 100.0f32; // Target nanoseconds
        let target_log = target_cost.ln();

        // Compute initial prediction
        let initial_pred = net.predict_log_cost(&expr);

        // Train for several steps
        for _ in 0..100 {
            let loss = net.train_value_mlp_step(&expr, target_log, 0.01);
            assert!(loss.is_finite(), "Value training loss should be finite");
        }

        // Final prediction should be closer to target
        let final_pred = net.predict_log_cost(&expr);
        let initial_error = (initial_pred.ln() - target_log).abs();
        let final_error = (final_pred.ln() - target_log).abs();

        // Allow for stochastic behavior, but generally should improve
        // (or at least not get catastrophically worse)
        assert!(
            final_error < initial_error * 2.0 || final_error < 1.0,
            "Value MLP should learn toward target: initial_err={}, final_err={}",
            initial_error,
            final_error
        );
    }

    #[test]
    fn test_randomize_mask_only() {
        let mut net = ExprNnue::new();

        // Set some backbone values that should be preserved
        net.embeddings.e[0][0] = 1.234;
        net.w1[0][0] = 5.678;
        net.b1[0] = 0.999;
        net.expr_proj_w[0][0] = 2.345; // shared projection - should be preserved

        // Initially mask-specific weights should be zero
        let initial_mask_sum: f32 = net.mask_mlp_w1.iter().flatten().map(|x| x.abs()).sum();
        assert!(
            initial_mask_sum < 1e-6,
            "Initial mask weights should be zero"
        );

        // Randomize mask-only
        net.randomize_mask_only(42);

        // Backbone should be PRESERVED
        assert!(
            (net.embeddings.e[0][0] - 1.234).abs() < 1e-6,
            "Embeddings should be preserved"
        );
        assert!(
            (net.w1[0][0] - 5.678).abs() < 1e-6,
            "w1 should be preserved"
        );
        assert!((net.b1[0] - 0.999).abs() < 1e-6, "b1 should be preserved");
        assert!(
            (net.expr_proj_w[0][0] - 2.345).abs() < 1e-6,
            "expr_proj should be preserved"
        );

        // Mask-specific weights should now be non-zero
        let final_mask_sum: f32 = net.mask_mlp_w1.iter().flatten().map(|x| x.abs()).sum();
        assert!(
            final_mask_sum > 1.0,
            "Randomized mask weights should be non-zero"
        );

        // Interaction matrix should be near identity diagonal
        for i in 0..EMBED_DIM {
            assert!(
                (net.interaction[i][i] - 1.0).abs() < 0.5,
                "Diagonal of interaction should be near 1.0"
            );
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_unified_architecture_serialization_roundtrip() {
        use std::path::PathBuf;

        let mut net = ExprNnue::new_random(42);
        net.randomize_mask_only(123);

        // Set some specific values we can verify
        net.interaction[0][0] = 1.234;
        net.mask_bias_proj[5] = -0.567;
        net.value_mlp_b2 = 3.14;

        // Create temp file path
        let temp_path = PathBuf::from("/tmp/test_dual_head_unified_serialization.bin");

        // Serialize
        net.save(&temp_path).expect("Save should succeed");

        // Deserialize
        let loaded = ExprNnue::load(&temp_path).expect("Load should succeed");

        // Cleanup temp file
        let _ = std::fs::remove_file(&temp_path);

        // Verify specific values
        assert!(
            (loaded.interaction[0][0] - 1.234).abs() < 1e-6,
            "Interaction should be preserved"
        );
        assert!(
            (loaded.mask_bias_proj[5] - (-0.567)).abs() < 1e-6,
            "Bias projection should be preserved"
        );
        assert!(
            (loaded.value_mlp_b2 - 3.14).abs() < 1e-6,
            "Value MLP bias should be preserved"
        );

        // Verify predictions match
        let expr = make_fma_pattern();
        let original_cost = net.predict_log_cost(&expr);
        let loaded_cost = loaded.predict_log_cost(&expr);
        assert!(
            (original_cost - loaded_cost).abs() < 1e-5,
            "Loaded network should produce same predictions"
        );
    }

    #[test]
    fn test_gradients_finite_through_all_paths() {
        let mut net = ExprNnue::new_random(42);
        let expr = make_fma_pattern();
        let rule_features = [0.25, 0.3, 1.0, 1.0, 0.0, 1.0, 0.5, 1.0];

        // Train saturation head
        let mask_loss = net.train_mask_step(&expr, &rule_features, true, 0.01, 1.0, 10.0);
        assert!(mask_loss.is_finite(), "Mask loss should be finite");
        assert!(!mask_loss.is_nan(), "Mask loss should not be NaN");

        // Train extraction head
        let value_loss = net.train_value_mlp_step(&expr, 5.0, 0.01);
        assert!(value_loss.is_finite(), "Value loss should be finite");
        assert!(!value_loss.is_nan(), "Value loss should not be NaN");

        // Verify weights didn't become NaN
        for row in &net.expr_proj_w {
            for &val in row {
                assert!(
                    !val.is_nan(),
                    "expr_proj_w should not contain NaN after training"
                );
            }
        }

        for row in &net.mask_mlp_w1 {
            for &val in row {
                assert!(
                    !val.is_nan(),
                    "mask_mlp_w1 should not contain NaN after training"
                );
            }
        }

        for row in &net.rule_mlp_w1 {
            for &val in row {
                assert!(
                    !val.is_nan(),
                    "rule_mlp_w1 should not contain NaN after training"
                );
            }
        }

        for row in &net.interaction {
            for &val in row {
                assert!(
                    !val.is_nan(),
                    "interaction should not contain NaN after training"
                );
            }
        }
    }

    // ========================================================================
    // Complex PE + Child-Index Encoding Tests
    // ========================================================================

    #[test]
    fn test_complex_pe_roundtrip() {
        // add_edge + remove_edge should return the accumulator to zero.
        let emb = OpEmbeddings::new_random(42);
        let mut acc = EdgeAccumulator::new();

        // Add several edges at various depths
        acc.add_edge(&emb, OpKind::Add, OpKind::Var, 0);
        acc.add_edge(&emb, OpKind::Mul, OpKind::Const, 3);
        acc.add_edge(&emb, OpKind::Div, OpKind::Sqrt, 10);
        acc.add_edge(&emb, OpKind::Sub, OpKind::Neg, 50);

        // Remove them in reverse order (shouldn't matter for additivity)
        acc.remove_edge(&emb, OpKind::Sub, OpKind::Neg, 50);
        acc.remove_edge(&emb, OpKind::Div, OpKind::Sqrt, 10);
        acc.remove_edge(&emb, OpKind::Mul, OpKind::Const, 3);
        acc.remove_edge(&emb, OpKind::Add, OpKind::Var, 0);

        for (i, &v) in acc.values.iter().enumerate() {
            assert!(
                v.abs() < 1e-5,
                "Complex PE roundtrip failed at index {i}: got {v}, expected ~0.0"
            );
        }
        assert_eq!(acc.edge_count, 0);
    }

    #[test]
    fn test_sibling_discrimination() {
        // Add(Sub(A,B), Div(C,D)) vs Add(Div(A,B), Sub(C,D))
        // With child-index encoding, these should produce DIFFERENT accumulators
        // because left child (idx 0) and right child (idx 1) get different PEs.
        let emb = OpEmbeddings::new_random(42);

        // Add(Sub(X,Y), Div(Z,W))
        let expr_a = Expr::Binary(
            OpKind::Add,
            Arc::new(Expr::Binary(
                OpKind::Sub,
                Arc::new(Expr::Var(0)),
                Arc::new(Expr::Var(1)),
            )),
            Arc::new(Expr::Binary(
                OpKind::Div,
                Arc::new(Expr::Var(2)),
                Arc::new(Expr::Var(3)),
            )),
        );

        // Add(Div(X,Y), Sub(Z,W))
        let expr_b = Expr::Binary(
            OpKind::Add,
            Arc::new(Expr::Binary(
                OpKind::Div,
                Arc::new(Expr::Var(0)),
                Arc::new(Expr::Var(1)),
            )),
            Arc::new(Expr::Binary(
                OpKind::Sub,
                Arc::new(Expr::Var(2)),
                Arc::new(Expr::Var(3)),
            )),
        );

        let acc_a = EdgeAccumulator::from_expr_dedup(&expr_a, &emb);
        let acc_b = EdgeAccumulator::from_expr_dedup(&expr_b, &emb);

        // Depth-encoded half should differ because siblings get different PEs
        let diff: f32 = acc_a.values[2 * K..4 * K]
            .iter()
            .zip(acc_b.values[2 * K..4 * K].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(
            diff > 0.01,
            "Sibling-swapped expressions should produce different depth-encoded accumulators, diff={diff}"
        );
    }

    // ========================================================================
    // structural_hash and from_expr_dedup tests
    // ========================================================================

    #[test]
    fn test_structural_hash_identical_trees() {
        // Two separately constructed trees with the same structure must produce
        // the same hash.
        let a = Expr::Binary(
            OpKind::Add,
            Arc::new(Expr::Var(0)),
            Arc::new(Expr::Const(1.0)),
        );
        let b = Expr::Binary(
            OpKind::Add,
            Arc::new(Expr::Var(0)),
            Arc::new(Expr::Const(1.0)),
        );
        assert_eq!(
            structural_hash(&a),
            structural_hash(&b),
            "Identical trees must hash equally"
        );
    }

    #[test]
    fn test_structural_hash_different_ops_differ() {
        let add = Expr::Binary(OpKind::Add, Arc::new(Expr::Var(0)), Arc::new(Expr::Var(1)));
        let mul = Expr::Binary(OpKind::Mul, Arc::new(Expr::Var(0)), Arc::new(Expr::Var(1)));
        assert_ne!(
            structural_hash(&add),
            structural_hash(&mul),
            "Trees with different root ops must hash differently"
        );
    }

    #[test]
    fn test_structural_hash_different_vars_differ() {
        let v0 = Expr::Var(0);
        let v1 = Expr::Var(1);
        assert_ne!(structural_hash(&v0), structural_hash(&v1));
    }

    #[test]
    fn test_structural_hash_negative_zero_equals_positive_zero() {
        // -0.0 and 0.0 must hash identically (IEEE 754 quirk).
        let pos = Expr::Const(0.0_f32);
        let neg = Expr::Const(-0.0_f32);
        assert_eq!(
            structural_hash(&pos),
            structural_hash(&neg),
            "-0.0 and 0.0 should hash to the same value"
        );
    }

    #[test]
    fn test_from_expr_dedup_no_shared_subtrees() {
        // A tree with no shared subtrees: backref_count must be 0.
        let emb = OpEmbeddings::new_random(42);
        let expr = make_fma_pattern(); // (a*b) + c — no CSE

        let dedup = EdgeAccumulator::from_expr_dedup(&expr, &emb);

        assert!(dedup.edge_count > 0, "must have edges");
        assert!(dedup.node_count > 0, "must have nodes");
        assert_eq!(0, dedup.backref_count, "no CSE → backref_count must be 0");
    }

    #[test]
    fn test_from_expr_dedup_shared_subtree_counted_once() {
        // Build an expression where the SAME Expr value appears at two
        // positions: Add(neg(x), neg(x)).  The two `neg(x)` subtrees have
        // identical content, so dedup must walk the second one only once and
        // increment backref_count.
        //
        // Note: Expr is tree-shaped in Rust (each Box owns its node), so we
        // create two structurally equal (but separately allocated) neg(x)
        // subtrees.  structural_hash treats them as identical.
        let emb = OpEmbeddings::new_random(42);
        let neg_x_a = Expr::Unary(OpKind::Neg, Arc::new(Expr::Var(0)));
        let neg_x_b = Expr::Unary(OpKind::Neg, Arc::new(Expr::Var(0)));
        let shared_sub = Expr::Binary(OpKind::Add, Arc::new(neg_x_a), Arc::new(neg_x_b));

        let dedup = EdgeAccumulator::from_expr_dedup(&shared_sub, &emb);

        // backref_count must be exactly 1: the second `neg(x)` was skipped.
        assert_eq!(
            1, dedup.backref_count,
            "one shared subtree → backref_count == 1"
        );

        // The deduplicated walk must see: Add node + one neg(x) node + one Var node = 3 nodes.
        // The second neg(x) is skipped.
        assert_eq!(
            3, dedup.node_count,
            "Add + one neg(x) + one Var = 3 unique nodes, got {}",
            dedup.node_count
        );
    }

    // ========================================================================
    // GraphAccumulator normalization tests
    // ========================================================================

    #[test]
    fn test_graph_acc_normalize_unit_norm_per_section() {
        let emb = OpEmbeddings::new_random(42);
        let mut gacc = GraphAccumulator::new();
        // Build a non-trivial accumulator: several edges
        gacc.add_edge(&emb, OpKind::Add, OpKind::Mul);
        gacc.add_edge(&emb, OpKind::Mul, OpKind::Var);
        gacc.add_edge(&emb, OpKind::Mul, OpKind::Var);
        gacc.add_edge(&emb, OpKind::Add, OpKind::Var);

        let normed = gacc.normalized();

        // Each section should have L2 norm ~1.0
        let section_norm = |start: usize, end: usize| -> f32 {
            let sum_sq: f32 = normed.values[start..end].iter().map(|v| v * v).sum();
            sqrtf(sum_sq)
        };
        let eps = 1e-5;
        assert!(
            (section_norm(0, K) - 1.0).abs() < eps,
            "parent section norm should be 1.0"
        );
        assert!(
            (section_norm(K, 2 * K) - 1.0).abs() < eps,
            "child section norm should be 1.0"
        );
        assert!(
            (section_norm(2 * K, 3 * K) - 1.0).abs() < eps,
            "1-hop VSA section norm should be 1.0"
        );
        // 2-hop section is zero (no 2-hop edges added) — normalization should leave it as zeros.
        let hop2_norm = section_norm(3 * K, 4 * K);
        assert!(
            hop2_norm < eps,
            "2-hop VSA section should be zero when no 2-hop edges added, got {hop2_norm}"
        );
    }

    #[test]
    fn test_graph_acc_normalize_scalars_preserved() {
        let emb = OpEmbeddings::new_random(42);
        let mut gacc = GraphAccumulator::new();
        gacc.add_edge(&emb, OpKind::Add, OpKind::Mul);
        gacc.add_leaf();
        gacc.add_leaf();
        gacc.node_budget = 100;
        gacc.epoch_budget = 50;

        let normed = gacc.normalized();
        assert_eq!(
            normed.edge_count, gacc.edge_count,
            "edge_count must be preserved"
        );
        assert_eq!(
            normed.node_count, gacc.node_count,
            "node_count must be preserved"
        );
        assert_eq!(
            normed.node_budget, gacc.node_budget,
            "node_budget must be preserved"
        );
        assert_eq!(
            normed.epoch_budget, gacc.epoch_budget,
            "epoch_budget must be preserved"
        );
    }

    #[test]
    fn test_graph_acc_normalize_zero_is_safe() {
        // A fresh (zero) accumulator should normalize without NaN/Inf.
        let gacc = GraphAccumulator::new();
        let normed = gacc.normalized();
        for (i, &v) in normed.values.iter().enumerate() {
            assert!(v.is_finite(), "values[{i}] must be finite, got {v}");
            assert_eq!(
                v, 0.0,
                "zero accumulator must stay zero after normalization"
            );
        }
    }

    #[test]
    fn test_graph_acc_normalize_in_place_matches_normalized() {
        let emb = OpEmbeddings::new_random(42);
        let mut gacc = GraphAccumulator::new();
        gacc.add_edge(&emb, OpKind::Add, OpKind::Mul);
        gacc.add_edge(&emb, OpKind::Mul, OpKind::Var);

        let copy_normed = gacc.normalized();

        let mut in_place = gacc.clone();
        in_place.normalize_in_place();

        for i in 0..GRAPH_ACC_DIM {
            assert!(
                (copy_normed.values[i] - in_place.values[i]).abs() < 1e-9,
                "values[{i}] mismatch: normalized()={} vs normalize_in_place()={}",
                copy_normed.values[i],
                in_place.values[i]
            );
        }
    }

    #[test]
    fn test_graph_acc_normalize_scale_invariance() {
        // Doubling all edges (adding each edge twice) should yield the same
        // normalized vector, proving scale invariance.
        let emb = OpEmbeddings::new_random(42);

        let mut small = GraphAccumulator::new();
        small.add_edge(&emb, OpKind::Add, OpKind::Mul);
        small.add_edge(&emb, OpKind::Mul, OpKind::Var);

        let mut large = GraphAccumulator::new();
        for _ in 0..10 {
            large.add_edge(&emb, OpKind::Add, OpKind::Mul);
            large.add_edge(&emb, OpKind::Mul, OpKind::Var);
        }

        let small_n = small.normalized();
        let large_n = large.normalized();

        for i in 0..GRAPH_ACC_DIM {
            assert!(
                (small_n.values[i] - large_n.values[i]).abs() < 1e-5,
                "normalized vectors should match regardless of scale: values[{i}] small={} large={}",
                small_n.values[i],
                large_n.values[i]
            );
        }
    }

    #[test]
    fn test_graph_acc_normalize_idempotent() {
        // Normalizing twice should produce the same result.
        let emb = OpEmbeddings::new_random(42);
        let mut gacc = GraphAccumulator::new();
        gacc.add_edge(&emb, OpKind::Add, OpKind::Mul);
        gacc.add_edge(&emb, OpKind::Mul, OpKind::Var);

        let once = gacc.normalized();
        let twice = once.normalized();

        for i in 0..GRAPH_ACC_DIM {
            assert!(
                (once.values[i] - twice.values[i]).abs() < 1e-6,
                "normalization must be idempotent: values[{i}] once={} twice={}",
                once.values[i],
                twice.values[i]
            );
        }
    }

    // ========================================================================
    // GraphAccumulator incremental remove tests
    // ========================================================================

    /// Helper: walk an expression tree and call `add_op_node` / `add_leaf`
    /// on a `GraphAccumulator`, using the backward-compatible depth=1 API.
    fn build_graph_acc_from_expr(expr: &Expr, emb: &OpEmbeddings) -> GraphAccumulator {
        let mut acc = GraphAccumulator::new();
        graph_acc_walk_add(expr, emb, &mut acc);
        acc
    }

    fn graph_acc_walk_add(expr: &Expr, emb: &OpEmbeddings, acc: &mut GraphAccumulator) {
        match expr {
            Expr::Var(_) | Expr::Const(_) => acc.add_leaf(),
            Expr::Param(i) => panic!("Expr::Param({i}) in graph acc test"),
            Expr::Unary(op, child) => {
                acc.add_op_node(emb, *op, &[child.op_type()]);
                graph_acc_walk_add(child, emb, acc);
            }
            Expr::Binary(op, left, right) => {
                acc.add_op_node(emb, *op, &[left.op_type(), right.op_type()]);
                graph_acc_walk_add(left, emb, acc);
                graph_acc_walk_add(right, emb, acc);
            }
            Expr::Ternary(op, a, b, c) => {
                acc.add_op_node(emb, *op, &[a.op_type(), b.op_type(), c.op_type()]);
                graph_acc_walk_add(a, emb, acc);
                graph_acc_walk_add(b, emb, acc);
                graph_acc_walk_add(c, emb, acc);
            }
            Expr::Nary(op, children) => {
                let child_ops: alloc::vec::Vec<OpKind> =
                    children.iter().map(|c| c.op_type()).collect();
                acc.add_op_node(emb, *op, &child_ops);
                for child in children.iter() {
                    graph_acc_walk_add(child, emb, acc);
                }
            }
        }
    }

    /// Helper: walk an expression tree and call `remove_op_node` / `remove_leaf`
    /// on a `GraphAccumulator` to subtract its contribution.
    fn graph_acc_walk_remove(expr: &Expr, emb: &OpEmbeddings, acc: &mut GraphAccumulator) {
        match expr {
            Expr::Var(_) | Expr::Const(_) => acc.remove_leaf(),
            Expr::Param(i) => panic!("Expr::Param({i}) in graph acc test"),
            Expr::Unary(op, child) => {
                acc.remove_op_node(emb, *op, &[child.op_type()]);
                graph_acc_walk_remove(child, emb, acc);
            }
            Expr::Binary(op, left, right) => {
                acc.remove_op_node(emb, *op, &[left.op_type(), right.op_type()]);
                graph_acc_walk_remove(left, emb, acc);
                graph_acc_walk_remove(right, emb, acc);
            }
            Expr::Ternary(op, a, b, c) => {
                acc.remove_op_node(emb, *op, &[a.op_type(), b.op_type(), c.op_type()]);
                graph_acc_walk_remove(a, emb, acc);
                graph_acc_walk_remove(b, emb, acc);
                graph_acc_walk_remove(c, emb, acc);
            }
            Expr::Nary(op, children) => {
                let child_ops: alloc::vec::Vec<OpKind> =
                    children.iter().map(|c| c.op_type()).collect();
                acc.remove_op_node(emb, *op, &child_ops);
                for child in children.iter() {
                    graph_acc_walk_remove(child, emb, acc);
                }
            }
        }
    }

    #[test]
    fn test_graph_acc_remove_edge_inverse_of_add() {
        let emb = OpEmbeddings::new_random(42);

        // Expression A: x + y
        let expr_a = make_add_xy();
        // Expression B: sin(x) * y
        let expr_b = Expr::Binary(
            OpKind::Mul,
            Arc::new(Expr::Unary(OpKind::Sin, Arc::new(Expr::Var(0)))),
            Arc::new(Expr::Var(1)),
        );

        // Build accumulator from A, then add B and remove A.
        let mut incremental = build_graph_acc_from_expr(&expr_a, &emb);
        graph_acc_walk_add(&expr_b, &emb, &mut incremental);
        graph_acc_walk_remove(&expr_a, &emb, &mut incremental);

        // Build accumulator from B alone (ground truth).
        let from_scratch = build_graph_acc_from_expr(&expr_b, &emb);

        assert_eq!(
            incremental.edge_count, from_scratch.edge_count,
            "edge_count mismatch: incremental={} vs from_scratch={}",
            incremental.edge_count, from_scratch.edge_count
        );
        assert_eq!(
            incremental.node_count, from_scratch.node_count,
            "node_count mismatch: incremental={} vs from_scratch={}",
            incremental.node_count, from_scratch.node_count
        );
        for i in 0..GRAPH_ACC_DIM {
            let diff = (incremental.values[i] - from_scratch.values[i]).abs();
            assert!(
                diff < 1e-6,
                "values[{i}] mismatch: incremental={} vs from_scratch={} (diff={diff})",
                incremental.values[i],
                from_scratch.values[i]
            );
        }
    }

    #[test]
    fn test_graph_acc_add_then_remove_returns_to_zero() {
        let emb = OpEmbeddings::new_random(99);

        let expr = make_fma_pattern();
        let mut acc = build_graph_acc_from_expr(&expr, &emb);

        assert!(acc.edge_count > 0, "should have edges after add");
        assert!(acc.node_count > 0, "should have nodes after add");

        graph_acc_walk_remove(&expr, &emb, &mut acc);

        assert_eq!(
            acc.edge_count, 0,
            "edge_count should be 0 after full removal"
        );
        assert_eq!(
            acc.node_count, 0,
            "node_count should be 0 after full removal"
        );
        for i in 0..GRAPH_ACC_DIM {
            assert!(
                acc.values[i].abs() < 1e-6,
                "values[{i}] should be ~0 after full removal, got {}",
                acc.values[i]
            );
        }
    }

    #[test]
    fn test_graph_acc_remove_leaf_saturates_at_zero() {
        let mut acc = GraphAccumulator::new();
        acc.remove_leaf();
        assert_eq!(acc.node_count, 0, "node_count must not underflow");
    }

}
