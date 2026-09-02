//! # Op embeddings and the typed edge stream
//!
//! Two things live here, both reused by the saturation Guide
//! (`crate::nnue::guide`) by denotation rather than by history:
//!
//! - [`OpEmbeddings`]: one dense K-vector per [`OpKind`]. Dimension 0 can be
//!   seeded from the static latency table
//!   ([`OpEmbeddings::init_with_latency_prior`]), so an untrained op already
//!   carries a sane cost prior and the embedding shares a scale with the
//!   table the extractor actually uses.
//!
//! - The typed edge stream: a walk over an expression DAG — an [`ExprArena`]
//!   subtree, or an e-graph [`Extraction`](crate::egraph::extract::Extraction)
//!   — emits one [`CostEdge`] per parent→child slot, bound at a depth-encoded
//!   [`PeSlot`]. A second reference to a shared node is a register reload
//!   (`parent → Var`), exactly as the JIT's let-binding emitter sees it. An
//!   [`EdgeTrace`] is that stream recorded in emission order, and an
//!   [`EdgeSink`] is where a walk sends it — a parent-child edge is the
//!   up/down context primitive the Guide's candidate-context design folds.
//!
//! The extraction (value) head that used to sit on top of this — an
//! accumulator feature vector, a backbone + value MLP, its checkpoint format
//! — tied the static table on schedule-free kernels
//! (docs/paper/2026-08-egraph-nnue-parity.md) and its shape was deleted on
//! 2026-09-01; history is in VCS. What this module keeps is the denotation a
//! future residual reranker needs — the op vocabulary, the edge stream, and
//! per-node variance classification
//! (docs/plans/2026-09-01-schedule-cost-model-denotation.md).

extern crate alloc;

use alloc::vec::Vec;
use libm::{logf, sqrtf};

use crate::egraph::Rewrite;
use crate::egraph::cost::latency_prior_cycles;
use crate::egraph::extract::Extraction;
pub use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};
use pixelflow_ir::kind::OpMap;

// ============================================================================
// Constants
// ============================================================================

/// Embedding dimension per operation.
pub const K: usize = 32;

/// Number of log2-compressed search-resource scalars `nnue::guide`'s graph
/// tower appends to its accumulator (edge count, node count, node budget,
/// epoch budget).
pub const SCALAR_FEATURE_COUNT: usize = 4;

/// Maximum arity for child-index encoding.
/// Effective depth = `depth * MAX_ARITY + child_index`, where child_index ∈ [0, MAX_ARITY).
/// This breaks sibling symmetry: left and right children of the same parent get different PEs.
pub const MAX_ARITY: usize = 3;

/// Maximum effective depth for the sinusoidal depth table.
/// Child-index encoding triples the effective depth range: a tree of real depth 63
/// with ternary nodes → `63*3+2 = 191 < 192`. Depths beyond this are clamped.
pub const MAX_DEPTH: usize = 192;

/// Hidden layer size of `nnue::guide`'s graph tower and trunk.
pub const HIDDEN_DIM: usize = 64;

// ============================================================================
// Shared Embedding Constants
// ============================================================================

/// Embedding dimension of the rule/graph embedding space `nnue::guide`'s
/// saturation head scores in (mask MLP output, rule projection).
pub const EMBED_DIM: usize = 32;

/// Hidden dimension for private per-head MLPs.
pub const MLP_HIDDEN: usize = 16;

// ============================================================================
// Rule Templates (LHS/RHS Expression Templates)
// ============================================================================

/// Rule templates: LHS and RHS expressions for each rule.
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
/// Arena-backed rule templates: one [`ArenaRuleTemplate`] per rule index.
///
/// `None` slots are rules that define no structural template (or only one
/// side). Built from the [`Rewrite`] trait via [`RuleTemplates::build`], which
/// reads each rule's LHS/RHS directly into a per-rule [`ExprArena`].
#[derive(Clone, Default)]
pub struct RuleTemplates {
    /// One optional arena-backed template per rule, indexed by rule_idx.
    pub rules: Vec<Option<ArenaRuleTemplate>>,
}

impl RuleTemplates {
    /// Create empty templates.
    #[must_use]
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Create templates for a given number of rules (all `None` initially).
    #[must_use]
    pub fn with_capacity(num_rules: usize) -> Self {
        Self {
            rules: (0..num_rules).map(|_| None).collect(),
        }
    }

    /// Build and store the LHS/RHS template for `rule` at `rule_idx`, reading
    /// them directly from the [`Rewrite`] trait into a per-rule arena.
    ///
    /// Only stored when the rule defines BOTH sides (legacy semantics).
    pub fn build(&mut self, rule_idx: usize, rule: &dyn Rewrite) {
        if rule_idx >= self.rules.len() {
            self.rules.resize_with(rule_idx + 1, || None);
        }
        let tmpl = ArenaRuleTemplate::from_rule(rule);
        if tmpl.lhs.is_some() && tmpl.rhs.is_some() {
            self.rules[rule_idx] = Some(tmpl);
        }
    }

    /// Get the arena-backed template for a rule, if defined.
    #[must_use]
    pub fn get(&self, rule_idx: usize) -> Option<&ArenaRuleTemplate> {
        self.rules.get(rule_idx).and_then(|o| o.as_ref())
    }

    /// Number of rule slots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Check if a rule has a template defined.
    #[must_use]
    pub fn has_templates(&self, rule_idx: usize) -> bool {
        self.get(rule_idx).is_some()
    }

    /// Returns `true` if any template (LHS or RHS) has `op` as its root op.
    #[must_use]
    pub fn has_root_op(&self, op: OpKind) -> bool {
        self.rules
            .iter()
            .flatten()
            .any(|t| t.lhs_op == Some(op) || t.rhs_op == Some(op))
    }

    /// Build a precomputed O(1) set of root ops appearing in any template.
    #[must_use]
    pub fn root_op_set(&self) -> OpMap<bool> {
        let mut set = OpMap::splat(false);
        for t in self.rules.iter().flatten() {
            if let Some(op) = t.lhs_op {
                set[op] = true;
            }
            if let Some(op) = t.rhs_op {
                set[op] = true;
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
/// corresponding side was not provided by the rule.
#[derive(Clone)]
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
    /// Build the LHS/RHS templates of `rule` directly into a fresh arena.
    #[must_use]
    pub fn from_rule(rule: &dyn Rewrite) -> Self {
        let mut arena = ExprArena::with_capacity(16);
        let lhs = rule.lhs_template(&mut arena);
        let rhs = rule.rhs_template(&mut arena);

        let lhs_op = lhs.and_then(|id| {
            if matches!(arena.node(id), ExprNode::Var(_)) {
                None
            } else {
                Some(arena.kind(id))
            }
        });
        let rhs_op = rhs.and_then(|id| {
            if matches!(arena.node(id), ExprNode::Var(_)) {
                None
            } else {
                Some(arena.kind(id))
            }
        });

        Self {
            arena,
            lhs,
            rhs,
            lhs_op,
            rhs_op,
        }
    }
}

/// Arena-backed rule template storage for the mask head.
pub struct ArenaRuleTemplates {
    /// One arena-backed template per rule, indexed by rule_idx.
    pub arenas: Vec<ArenaRuleTemplate>,
    /// Precomputed O(1) op-membership set (same semantics as `root_op_set()`).
    pub root_op_set: OpMap<bool>,
}

impl ArenaRuleTemplates {
    /// Convert [`RuleTemplates`] into dense arena form (one entry per rule).
    #[must_use]
    pub fn from_rule_templates(templates: &RuleTemplates) -> Self {
        let mut arenas = Vec::with_capacity(templates.len());
        let mut root_op_set = OpMap::splat(false);

        for slot in &templates.rules {
            let tmpl = match slot {
                Some(t) => t.clone(),
                None => ArenaRuleTemplate {
                    arena: ExprArena::new(),
                    lhs: None,
                    rhs: None,
                    lhs_op: None,
                    rhs_op: None,
                },
            };
            if let Some(op) = tmpl.lhs_op {
                root_op_set[op] = true;
            }
            if let Some(op) = tmpl.rhs_op {
                root_op_set[op] = true;
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
/// Each of the [`OpMap::LEN`] operations gets a K-dimensional embedding
/// vector.
/// These are the primary learned parameters that capture semantic
/// similarity between operations.
#[derive(Clone)]
pub struct OpEmbeddings {
    /// E[op][i] = i-th dimension of op's embedding.
    /// Stored as one K-vector per op: `OpMap::LEN * K` = 50 × 32 = 1,600 floats.
    pub e: OpMap<[f32; K]>,
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
            e: OpMap::splat([0.0; K]),
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
    ///
    /// Dimension 0 = latency, squashed to `[0, 1]` from the shared
    /// [`latency_prior_cycles`] cycle table (source of truth, also used by
    /// `egraph::cost::CostModel::latency_prior`) via
    /// `ln(1+cycles) / ln(1+1000)`.
    ///
    /// The squash is logarithmic, not linear: the 2026-08-10 re-measurement
    /// of the cycle table spread real ops across 3..=196 cycles (Pow's
    /// lowered form is 196, not the old 12), so the previous linear `/20`
    /// clamp would pin every op from Rsqrt (21) to Pow (196) at 1.0 —
    /// indistinguishable from each other *and* from Dwrt's prohibitive 1000.
    /// The log curve keeps both ends discriminable: Add 0.23, Sqrt 0.40,
    /// Sin 0.62, Pow 0.77, Dwrt 1.0. Affects fresh initializations only;
    /// trained weight files are untouched.
    pub fn init_with_latency_prior(&mut self, seed: u64) {
        // Dwrt's deliberately-prohibitive 1000-cycle entry maps to exactly
        // 1.0; everything real lands strictly below it.
        const LATENCY_CEILING_CYCLES: f32 = 1000.0;

        let mut rng_state = seed.wrapping_add(1);
        let small_scale = 0.1; // Small noise for other dimensions

        let cycles_of = latency_prior_cycles();
        for op in OpKind::all() {
            // Dimension 0: latency prior, log-squashed from the shared cycle
            // table.
            let cycles = cycles_of[op] as f32;
            let squashed = logf(1.0 + cycles) / logf(1.0 + LATENCY_CEILING_CYCLES);
            self.e[op][0] = squashed.min(1.0);

            // Dimensions 1..K: small random for learning interactions
            for dim in 1..K {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let uniform = (rng_state >> 33) as f32 / (1u64 << 31) as f32;
                self.e[op][dim] = (uniform * 2.0 - 1.0) * small_scale;
            }
        }
    }

    /// Randomize embeddings in place (fully random, no priors).
    pub fn randomize(&mut self, seed: u64) {
        let scale = sqrtf(2.0 / K as f32);
        let mut rng_state = seed.wrapping_add(1);

        for op in OpKind::all() {
            for dim in 0..K {
                // LCG for no_std compatibility
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);

                // Convert to [-1, 1] and scale
                let uniform = (rng_state >> 33) as f32 / (1u64 << 31) as f32;
                let centered = uniform * 2.0 - 1.0;
                self.e[op][dim] = centered * scale;
            }
        }
    }

    /// Get embedding for an operation.
    #[inline]
    #[must_use]
    pub fn get(&self, op: OpKind) -> &[f32; K] {
        &self.e[op]
    }

    /// Total parameter count.
    #[must_use]
    pub const fn param_count() -> usize {
        OpMap::<[f32; K]>::LEN * K
    }
}

// ============================================================================
// Sinusoidal Depth Encoding (Fixed Positional Encoding for AST Depth)
// ============================================================================

/// Precomputed sinusoidal positional encoding table.
///
/// Fixed (not learned) — zero parameters, zero serialization, zero gradients.
/// Whatever consumes an edge stream learns how to USE the rotation; the
/// encoding itself is a deterministic function of depth.
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

// ============================================================================
// Typed Feature Edges (the walker's replayable output)
// ============================================================================

/// The row of the sinusoidal PE table a feature edge was bound at.
///
/// Constructed only by clamping an effective depth
/// (`tree_depth * MAX_ARITY + child_slot`) into the table, so a value of this
/// type IS a valid row index — [`PeSlot::pe`] cannot miss, and no consumer
/// has to re-validate a raw integer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PeSlot(u16);

impl PeSlot {
    /// Clamp an effective depth into the PE table.
    #[inline]
    #[must_use]
    pub fn from_effective_depth(effective_depth: u32) -> Self {
        Self(effective_depth.min((MAX_DEPTH - 1) as u32) as u16)
    }

    /// The PE table row this slot names.
    #[inline]
    #[must_use]
    pub fn pe(self) -> &'static [f32; K] {
        &DEPTH_PE[self.0 as usize]
    }
}

/// One edge of an expression DAG's feature stream: `parent → child`, bound
/// at PE row `pe`. A register reload (second and later references to a
/// shared node) is the edge `parent → OpKind::Var`, exactly as the JIT's
/// let-binding emitter sees it.
///
/// This is the walker's entire embedding-relevant output, so a recorded edge
/// stream is both sides of an embedding contract at once: a consumer folds it
/// forward against an [`OpEmbeddings`] (a context accumulator), and a trainer
/// differentiates the same fold backward to move the table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CostEdge {
    /// The referencing operation.
    pub parent: OpKind,
    /// The operation computed into this slot — [`OpKind::Var`] for a
    /// register reload of an already-emitted shared node.
    pub child: OpKind,
    /// The PE rotation this edge was bound under.
    pub pe: PeSlot,
}

/// The recorded stream of one walk over an expression DAG: its edges in
/// emission order, plus the number of distinct nodes the walk expanded.
///
/// Denotation: an `EdgeTrace` is an expression DAG *as an edge multiset with
/// order*. Two DAGs that emit the same trace are indistinguishable to every
/// embedding-based consumer, which is why the train-side
/// ([`EdgeTrace::from_arena_dag`]) and deploy-side
/// ([`EdgeTrace::from_extraction`]) adapters share one walker: there is no
/// second edge policy left to drift (pinned by
/// `edge_traces_from_arena_and_extraction_agree` in `egraph::extract`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EdgeTrace {
    edges: Vec<CostEdge>,
    node_count: u32,
}

impl EdgeTrace {
    /// Walk an arena subtree. Sharing is by `ExprId` — exactly the sharing
    /// the JIT's let-binding emitter sees when this arena is compiled, so the
    /// reload-edge policy describes the emitted object.
    ///
    /// # Panics
    ///
    /// Panics if the subtree contains `ExprNode::Param` — substitute
    /// parameters before walking.
    #[must_use]
    pub fn from_arena_dag(arena: &ExprArena, root: ExprId) -> Self {
        Self::walk(&ArenaCostDag { arena, root })
    }

    /// Walk the DAG an [`Extraction`] will materialise: its chosen node per
    /// reachable e-class, with every `Shl`/`Shr` count child pinned to its
    /// class's `Const` representative ([`Extraction::pinned_choices`]) — the
    /// same view `choices_to_arena` emits, so the trace describes the
    /// compiled DAG and not whichever node the extraction happened to record
    /// for a count class.
    #[must_use]
    pub fn from_extraction(extraction: &Extraction<'_>) -> Self {
        let pinned = extraction.pinned_choices();
        Self::walk(&ChoicesCostDag { extraction, pinned })
    }

    /// The recorded edges, in the walk's emission order.
    #[must_use]
    pub fn edges(&self) -> &[CostEdge] {
        &self.edges
    }

    /// Number of distinct nodes the walk expanded (each shared node counted
    /// once; its later references are reload edges, not nodes).
    #[must_use]
    pub fn node_count(&self) -> u32 {
        self.node_count
    }

    fn walk<D: CostDag>(dag: &D) -> Self {
        let mut edges = Vec::new();
        let node_count = walk_cost_dag(dag, &mut edges);
        Self { edges, node_count }
    }
}

/// Where a walk sends the edges it emits. `Vec<CostEdge>` records them (an
/// [`EdgeTrace`]); `()` discards them, for a consumer that folds each edge
/// on the spot and keeps only its own state. The walker emits through
/// exactly one call site, so no two consumers can see different streams.
trait EdgeSink {
    fn record(&mut self, edge: CostEdge);
}

impl EdgeSink for () {
    #[inline]
    fn record(&mut self, _: CostEdge) {}
}

impl EdgeSink for Vec<CostEdge> {
    #[inline]
    fn record(&mut self, edge: CostEdge) {
        self.push(edge);
    }
}

// ============================================================================
// The one walker
// ============================================================================

/// Emit the edge stream of `dag` into `sink`; returns the number of distinct
/// nodes expanded.
///
/// This is the ONLY function that turns an expression DAG into feature
/// edges, whether the DAG lives in an [`ExprArena`] or in an e-graph with
/// extraction choices. The 2026-08 round-0 audit found two walkers with
/// different edge policies (no reload edges on one side) biasing a deployed
/// model by −0.29 log-ns; one walker makes that divergence unrepresentable.
///
/// Edge policy (matches what the JIT emits):
/// - The first reference to a node is its computation edge
///   `(parent_op, child_op)` at the referencing slot's effective depth
///   (`depth * MAX_ARITY + child_slot`, so siblings bind differently).
/// - Every later reference is a register reload: a single
///   `(parent_op, Var)` edge — shared subexpressions become let-bindings,
///   so the DAG is not tree-bloated.
/// - Nodes with no recorded choice contribute nothing.
///
/// # Panics
///
/// Panics if a node id is out of bounds.
fn walk_cost_dag<D: CostDag, S: EdgeSink>(dag: &D, sink: &mut S) -> u32 {
    let bound = dag.id_bound();
    let mut expanded = alloc::vec![false; bound];
    let mut edge_emitted = alloc::vec![false; bound];
    let mut stack: Vec<(u32, u32)> = alloc::vec![(dag.root(), 0)];
    let mut children: Vec<u32> = Vec::new();
    let mut node_count: u32 = 0;

    while let Some((id, depth)) = stack.pop() {
        let idx = id as usize;
        assert!(
            idx < bound,
            "walk_cost_dag: node id {id} out of bounds (bound={bound})"
        );
        if expanded[idx] {
            continue;
        }
        expanded[idx] = true;

        children.clear();
        let Some(parent_op) = dag.resolve(id, &mut children) else {
            continue; // No recorded choice — contributes nothing.
        };
        node_count += 1;

        // One edge per child slot; leaves have no slots and add no edges.
        for (child_idx, &child) in children.iter().enumerate() {
            let Some(child_op) = dag.child_kind(child) else {
                continue;
            };
            let eff_depth = depth * MAX_ARITY as u32 + (child_idx.min(MAX_ARITY - 1)) as u32;
            let child_feature = if edge_emitted[child as usize] {
                // Shared reuse: a register reload, not a recomputation.
                OpKind::Var
            } else {
                edge_emitted[child as usize] = true;
                child_op
            };
            sink.record(CostEdge {
                parent: parent_op,
                child: child_feature,
                pe: PeSlot::from_effective_depth(eff_depth),
            });
            stack.push((child, depth + 1));
        }
    }

    node_count
}

// ============================================================================
// CostDag: the single view both adapters present to the walker
// ============================================================================

/// A DAG of chosen expression nodes, as the walker sees it.
///
/// Implemented by exactly two adapters — [`ArenaCostDag`] and
/// [`ChoicesCostDag`] — and consumed by exactly one walker,
/// [`walk_cost_dag`]. Ids must be canonical: structurally shared
/// subexpressions present the same id on every reference, because sharing
/// is what the reload-edge policy keys on.
trait CostDag {
    /// Exclusive upper bound on node ids.
    fn id_bound(&self) -> usize;

    /// Canonical id of the root node.
    fn root(&self) -> u32;

    /// Resolve the chosen representation of `id`: append its canonical child
    /// ids to `out` (nothing for leaves) and return its op kind. `None` when
    /// the node has no recorded choice.
    ///
    /// # Panics
    ///
    /// Panics when a recorded choice is malformed (e.g. out-of-bounds node
    /// index) — that is a broken invariant, not a tolerable state.
    fn resolve(&self, id: u32, out: &mut Vec<u32>) -> Option<OpKind>;

    /// Tolerant kind lookup for child edges: `None` when the child has no
    /// recorded (in-bounds) choice, in which case the edge is skipped rather
    /// than fabricated.
    fn child_kind(&self, id: u32) -> Option<OpKind>;
}

/// An [`ExprArena`] subtree as a [`CostDag`].
struct ArenaCostDag<'a> {
    arena: &'a ExprArena,
    root: ExprId,
}

impl CostDag for ArenaCostDag<'_> {
    fn id_bound(&self) -> usize {
        self.arena.len()
    }

    fn root(&self) -> u32 {
        self.root.0
    }

    fn resolve(&self, id: u32, out: &mut Vec<u32>) -> Option<OpKind> {
        let eid = ExprId(id);
        if let ExprNode::Param(i) = self.arena.node(eid) {
            panic!("ExprNode::Param({i}) reached the edge walker — substitute params first");
        }
        for child in self.arena.children(eid) {
            out.push(child.0);
        }
        Some(self.arena.kind(eid))
    }

    fn child_kind(&self, id: u32) -> Option<OpKind> {
        // `arena.kind` maps Param to Const; the child itself is expanded (and
        // `resolve` panics) right after, so a Param still fails loudly.
        Some(self.arena.kind(ExprId(id)))
    }
}

/// An [`Extraction`] (an e-graph plus a validated, well-founded choice
/// function) as a [`CostDag`].
struct ChoicesCostDag<'a> {
    extraction: &'a Extraction<'a>,
    /// [`Extraction::pinned_choices`] — the same `Shl`/`Shr` count
    /// substitution `choices_to_arena` applies, computed once so `resolve`
    /// and `child_kind` walk the DAG `choices_to_arena` will actually
    /// materialise. Using the raw (unpinned) `extraction.choice` here would
    /// let the walker descend into a count class's non-`Const` alternative,
    /// inflating the trace with nodes `choices_to_arena` never emits.
    pinned: Vec<Option<usize>>,
}

impl ChoicesCostDag<'_> {
    fn kind_of(node: &crate::egraph::ENode) -> OpKind {
        use crate::egraph::ENode;
        match node {
            ENode::Var(_) => OpKind::Var,
            ENode::Const(_) => OpKind::Const,
            ENode::Buffer(_) => OpKind::Buffer,
            ENode::Op { op, .. } => op.kind(),
        }
    }

    /// The pinned choice recorded for `class`'s canonical id, if any.
    fn pinned_choice(&self, class: crate::egraph::EClassId) -> Option<usize> {
        self.pinned.get(class.0 as usize).copied().flatten()
    }
}

impl CostDag for ChoicesCostDag<'_> {
    fn id_bound(&self) -> usize {
        self.extraction.egraph().num_classes()
    }

    fn root(&self) -> u32 {
        self.extraction.root().0
    }

    fn resolve(&self, id: u32, out: &mut Vec<u32>) -> Option<OpKind> {
        use crate::egraph::{EClassId, ENode};
        let egraph = self.extraction.egraph();
        let canonical = egraph.find(EClassId(id));
        let node_idx = self.pinned_choice(canonical)?;
        let nodes = egraph.nodes(canonical);
        let node = nodes.get(node_idx).unwrap_or_else(|| {
            panic!(
                "EdgeTrace::from_extraction: node_idx {} out of bounds for e-class {} (has {} nodes)",
                node_idx,
                canonical.0,
                nodes.len()
            )
        });
        if let ENode::Op { children, .. } = node {
            for &child in children {
                out.push(egraph.find(child).0);
            }
        }
        Some(Self::kind_of(node))
    }

    fn child_kind(&self, id: u32) -> Option<OpKind> {
        use crate::egraph::EClassId;
        let egraph = self.extraction.egraph();
        let canonical = egraph.find(EClassId(id));
        let node_idx = self.pinned_choice(canonical)?;
        egraph.nodes(canonical).get(node_idx).map(Self::kind_of)
    }
}

// ============================================================================
// Variance classification (denotation kept, accumulator shape deleted)
// ============================================================================
//
// The extraction-head program's value head — `ExprNnue`'s trunk and value
// MLP — was deleted after it tied the static table on schedule-free kernels
// (docs/paper/2026-08-egraph-nnue-parity.md). An earlier pass on this branch
// also restored the 4K per-level-sectioned edge accumulator that used to
// feed it (flat + depth-encoded, × parent/child, plus this variance
// histogram) under the name `LevelSectionedEdges`. Per JP's 2026-09-01
// ruling — "delete the shape, keep the denotation" — that accumulator was
// never actually level-indexed (variance entered only as four scalar
// fractions bolted onto a bag-of-edges sum), so restoring it was restoring
// the deleted SHAPE under a new name. It has been removed again; a
// genuinely level-indexed accumulator is specified for the future in
// docs/plans/2026-09-01-schedule-cost-model-denotation.md and is not built
// here. What survives is the denotation underneath it: per-node variance
// classification, kept as the free function below plus
// [`crate::egraph::extract::Extraction::chosen_variance`], which classifies
// the materialised (chosen) DAG rather than re-deriving variance from the
// e-graph (P1(c) of docs/plans/2026-08-17-cost-model-domain.md).

/// Classify every node of `arena` into const / frame-uniform / scanline-
/// uniform / pixel-varying, and return the fraction in each bucket. Shared
/// by any caller that classifies an [`ExprArena`] directly and by
/// [`crate::egraph::extract::Extraction::chosen_variance`], which
/// materialises the chosen DAG via `choices_to_arena` and classifies that —
/// one definition, imported, not restated.
pub(crate) fn variance_histogram(arena: &ExprArena) -> [f32; SCALAR_FEATURE_COUNT] {
    let variance = pixelflow_ir::variance::compute_arena_variance(arena);
    let total = variance.len() as f32;
    if total == 0.0 {
        return [0.0; SCALAR_FEATURE_COUNT];
    }

    let (mut n_const, mut n_frame, mut n_scanline, mut n_pixel) = (0u32, 0u32, 0u32, 0u32);
    for v in &variance {
        if v.is_const() {
            n_const += 1;
        } else if v.is_x_invariant() && !v.depends_on_y() {
            n_frame += 1;
        } else if v.is_x_invariant() {
            n_scanline += 1;
        } else {
            n_pixel += 1;
        }
    }
    [
        n_const as f32 / total,
        n_frame as f32 / total,
        n_scanline as f32 / total,
        n_pixel as f32 / total,
    ]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// An arena exercising every edge kind the walker emits: a shared
    /// subexpression (first reference = computation edge, second = register
    /// reload), a shared leaf, unary and binary ops.
    fn arena_with_sharing() -> (ExprArena, ExprId) {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let sq = arena.push_binary(OpKind::Mul, x, x);
        let s = arena.push_unary(OpKind::Sqrt, sq);
        // `s` referenced twice: Add(s, s) — the second reference must be
        // recorded as a reload edge (child = Var).
        let sum = arena.push_binary(OpKind::Add, s, s);
        let c = arena.push_const(2.0);
        let root = arena.push_binary(OpKind::Mul, sum, c);
        (arena, root)
    }

    #[test]
    fn trace_records_one_reload_edge_per_extra_reference_to_a_shared_node() {
        let (arena, root) = arena_with_sharing();
        let trace = EdgeTrace::from_arena_dag(&arena, root);

        // Six distinct nodes: x, x*x, sqrt, add, 2.0, root.
        assert_eq!(trace.node_count(), 6);
        // One edge per child slot of every expanded node: root(2) + add(2)
        // + sqrt(1) + mul(2) = 7.
        assert_eq!(trace.edges().len(), 7);

        // Add(s, s): the first reference computes sqrt, the second reloads.
        let add_children: Vec<OpKind> = trace
            .edges()
            .iter()
            .filter(|e| e.parent == OpKind::Add)
            .map(|e| e.child)
            .collect();
        assert_eq!(add_children, alloc::vec![OpKind::Sqrt, OpKind::Var]);
    }

    #[test]
    fn trace_binds_sibling_slots_to_distinct_pe_rows() {
        let (arena, root) = arena_with_sharing();
        let trace = EdgeTrace::from_arena_dag(&arena, root);

        // The root's two children sit at effective depths 0 and 1 — the
        // child-index term is what breaks left/right symmetry.
        let root_edges: Vec<PeSlot> = trace
            .edges()
            .iter()
            .filter(|e| e.parent == OpKind::Mul && e.child != OpKind::Var)
            .take(2)
            .map(|e| e.pe)
            .collect();
        assert_eq!(root_edges.len(), 2);
        assert_ne!(root_edges[0], root_edges[1]);
        assert_eq!(root_edges[0], PeSlot::from_effective_depth(0));
        assert_eq!(root_edges[1], PeSlot::from_effective_depth(1));
    }

    #[test]
    fn pe_slot_clamps_into_the_table() {
        let deep = PeSlot::from_effective_depth(u32::MAX);
        assert_eq!(deep, PeSlot::from_effective_depth((MAX_DEPTH - 1) as u32));
        assert_eq!(deep.pe(), &DEPTH_PE[MAX_DEPTH - 1]);
    }

    #[test]
    fn latency_prior_init_seeds_dimension_zero_from_the_static_table() {
        let emb = OpEmbeddings::new_with_latency_prior(3);
        let table = latency_prior_cycles();
        let add = emb.get(OpKind::Add)[0];
        let sqrt = emb.get(OpKind::Sqrt)[0];
        assert!(add.is_finite() && sqrt.is_finite());
        assert!(
            (table[OpKind::Sqrt] > table[OpKind::Add]) == (sqrt > add),
            "dimension 0 must order ops the way the latency table does: \
             add={add} (table {}), sqrt={sqrt} (table {})",
            table[OpKind::Add],
            table[OpKind::Sqrt]
        );
    }

    // ========================================================================
    // Variance classification (denotation kept, accumulator shape deleted —
    // see the module doc comment above `variance_histogram`). The
    // arena/extraction parity test lives in `egraph::extract` next to
    // `Extraction::chosen_variance`, mirroring
    // `arena_and_extraction_walks_record_the_same_edge_stream`.
    // ========================================================================

    #[test]
    fn variance_histogram_classifies_a_pure_constant_arena_as_all_const() {
        let mut arena = ExprArena::new();
        let root = arena.push_const(2.0);
        let hist = variance_histogram(&arena);
        assert_eq!(hist, [1.0, 0.0, 0.0, 0.0]);
        let _ = root; // arena root; histogram is over every node in `arena`.
    }

    #[test]
    fn variance_histogram_splits_pixel_and_scanline_nodes_correctly() {
        // Add(X, Y): X (var 0) is pixel-varying, Y (var 1) is scanline-
        // uniform, and Add inherits X's dependency so it is pixel-varying
        // too. 2 of 3 nodes pixel, 1 of 3 scanline.
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let _root = arena.push_binary(OpKind::Add, x, y);

        let hist = variance_histogram(&arena);
        assert_eq!(hist[0], 0.0, "no const nodes");
        assert_eq!(hist[1], 0.0, "no frame-uniform nodes (no Z/W)");
        assert!(
            (hist[2] - 1.0 / 3.0).abs() < 1e-6,
            "Y alone is scanline: {hist:?}"
        );
        assert!(
            (hist[3] - 2.0 / 3.0).abs() < 1e-6,
            "X and Add are pixel: {hist:?}"
        );
    }

    // ========================================================================
    // Latency-prior cold start (#1063): dim 0 must actually carry the signal
    // ========================================================================

    /// The extraction head's cold start seeded from
    /// [`OpEmbeddings::new_with_latency_prior`] rather than fully-random
    /// [`OpEmbeddings::new_random`] — the point being to hand the trainer a
    /// monotone latency signal in dimension 0 instead of noise. Its binary
    /// (`bootstrap_extraction_head`) was deleted with the extraction-head
    /// program in #1093; the property below is still worth pinning, because
    /// `new_with_latency_prior` remains and any future consumer wants the
    /// same guarantee.
    /// This pins that the signal is actually there: a future edit to the
    /// squash (or a seed that happens to overwrite dim 0 with the "small
    /// random" loop) would otherwise regress silently, since nothing else
    /// distinguishes prior-seeded from random init at the type level.
    #[test]
    fn latency_prior_init_dim0_correlates_with_log_cycles() {
        let emb = OpEmbeddings::new_with_latency_prior(42);
        let cycles_of = latency_prior_cycles();

        let xs: alloc::vec::Vec<f64> = OpKind::all()
            .map(|op| f64::from(logf(1.0 + cycles_of[op] as f32)))
            .collect();
        let ys: alloc::vec::Vec<f64> = OpKind::all().map(|op| f64::from(emb.e[op][0])).collect();
        assert_eq!(xs.len(), ys.len());

        let n = xs.len() as f64;
        let mean_x = xs.iter().sum::<f64>() / n;
        let mean_y = ys.iter().sum::<f64>() / n;
        let mut cov = 0.0f64;
        let mut var_x = 0.0f64;
        let mut var_y = 0.0f64;
        for (&x, &y) in xs.iter().zip(&ys) {
            let dx = x - mean_x;
            let dy = y - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
        assert!(
            var_x > 0.0 && var_y > 0.0,
            "degenerate input, correlation undefined"
        );
        let corr = cov / (var_x.sqrt() * var_y.sqrt());

        assert!(
            corr > 0.95,
            "dim0 of a latency-prior-seeded embedding must correlate with \
             log(1+cycles) across OpKind::all(); got r={corr:.4}"
        );
    }

    /// Random init (the OLD cold start, still used elsewhere e.g. saturation
    /// head / test fixtures) must NOT show this correlation — otherwise the
    /// test above would be vacuous, passing regardless of which
    /// initializer actually ran.
    #[test]
    fn random_init_dim0_does_not_correlate_with_log_cycles() {
        let emb = OpEmbeddings::new_random(42);
        let cycles_of = latency_prior_cycles();

        let xs: alloc::vec::Vec<f64> = OpKind::all()
            .map(|op| f64::from(logf(1.0 + cycles_of[op] as f32)))
            .collect();
        let ys: alloc::vec::Vec<f64> = OpKind::all().map(|op| f64::from(emb.e[op][0])).collect();

        let n = xs.len() as f64;
        let mean_x = xs.iter().sum::<f64>() / n;
        let mean_y = ys.iter().sum::<f64>() / n;
        let mut cov = 0.0f64;
        let mut var_x = 0.0f64;
        let mut var_y = 0.0f64;
        for (&x, &y) in xs.iter().zip(&ys) {
            let dx = x - mean_x;
            let dy = y - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
        let corr = cov / (var_x.sqrt() * var_y.sqrt());

        assert!(
            corr.abs() < 0.5,
            "fully-random init should show no meaningful correlation with \
             log(1+cycles); got r={corr:.4} — did the RNG seed happen to \
             collide with a structured pattern, or did new_random change?"
        );
    }
}
