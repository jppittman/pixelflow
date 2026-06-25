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
pub mod window;

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use libm::fabsf;

/// Re-export canonical IR types as the source of truth.
pub use pixelflow_ir::{ExprArena, ExprId, ExprNode, OpKind};

/// Re-export ExprNnue for dual-head AlphaZero-style architecture.
pub use factored::ExprNnue;

/// Re-export key types from factored module.
pub use factored::{EdgeAccumulator, GraphAccumulator, OpEmbeddings};

/// Re-export InstructionWindow for sliding-window scheduling.
pub use window::InstructionWindow;

/// Re-export unified mask architecture constants and types.
pub use factored::{
    ArenaRuleTemplates, EMBED_DIM, GRAPH_ACC_DIM, GRAPH_INPUT_DIM, MASK_INPUT_DIM, MASK_MAX_RULES,
    MLP_HIDDEN, RULE_CONCAT_DIM, RULE_FEATURE_DIM, RuleFeatures, RuleTemplates,
};

// Note: ExprGenConfig, ExprGenerator, BwdGenConfig, and BwdGenerator are already
// public structs defined in this module - no re-export needed.

// ============================================================================
// HalfEP Features (Legacy - being phased out in favor of Factored)
// ============================================================================

/// Maximum depth we encode in features.
pub const MAX_DEPTH: usize = 8;

// ============================================================================
// Dense Features
// ============================================================================

// ============================================================================
// NNUE Network Architecture
// ============================================================================

// ============================================================================
// Training Data Generation
// ============================================================================

// ============================================================================
// Binpack I/O (requires std)
// ============================================================================

/// Magic number for depth-limited binpack files.
pub const DEPTH_LIMITED_MAGIC: u32 = 0x444C4E55; // "DLNU" in ASCII

/// Version number for depth-limited binpack format.
pub const DEPTH_LIMITED_VERSION: u32 = 1;

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
            max_depth: 8,
            leaf_prob: 0.2,
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
    /// Op weights derived from ShaderToy corpus analysis.
    /// Real shaders are dominated by arithmetic (+, -, *, /), with moderate
    /// use of abs/sin/cos/clamp and rare use of exotic ops like atan2/rsqrt.
    /// Uniform weighting produces unrealistic expressions that the NNUE can't
    /// transfer to real workloads.
    fn shader_weight(op: OpKind) -> u32 {
        match op {
            // Arithmetic: ~70% of real shader ops
            OpKind::Mul => 50,
            OpKind::Add => 30,
            OpKind::Sub => 20,
            OpKind::Div => 10,
            OpKind::Neg => 10,
            // Common shader ops: ~20%
            OpKind::Abs => 12,
            OpKind::Sin => 8,
            OpKind::Cos => 8,
            OpKind::Clamp => 8,
            OpKind::Max => 6,
            OpKind::Min => 4,
            OpKind::Pow => 4,
            OpKind::Fract => 4,
            OpKind::Floor => 3,
            OpKind::Sqrt => 4,
            OpKind::Exp => 3,
            // Rare but valid: ~10%
            OpKind::Rsqrt => 2,
            OpKind::Recip => 2,
            OpKind::Ln => 2,
            OpKind::Log2 => 1,
            OpKind::Log10 => 1,
            OpKind::Exp2 => 1,
            OpKind::Tan => 1,
            OpKind::Atan => 1,
            OpKind::Atan2 => 1,
            OpKind::Asin => 1,
            OpKind::Acos => 1,
            OpKind::Hypot => 1,
            OpKind::Ceil => 1,
            OpKind::Round => 1,
            _ => 0,
        }
    }

    /// Create a new generator with the given seed.
    #[must_use]
    pub fn new(seed: u64, config: ExprGenConfig) -> Self {
        // Build weighted op table: each op appears proportional to its shader weight
        let mut seed_ops = Vec::new();
        for i in 0..OpKind::COUNT {
            if let Some(op) = OpKind::from_index(i) {
                if op.is_seed_op() {
                    let w = Self::shader_weight(op).max(1);
                    for _ in 0..w {
                        seed_ops.push(op);
                    }
                }
            }
        }
        assert!(
            !seed_ops.is_empty(),
            "No seed ops found in OpKind — is_seed_op() is broken"
        );
        assert!(
            config.num_vars <= 4,
            "num_vars={} exceeds INPUT_REGS limit of 4",
            config.num_vars
        );
        Self {
            config,
            state: seed,
            seed_ops,
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
        let val = (self.rand_f32() * max as f32) as usize;
        if val >= max { max - 1 } else { val }
    }
}

// ============================================================================
// Rewrite Rules (as "Moves")
// ============================================================================

// ============================================================================
// Pattern Match + Substitute (for rule template rewriting)
// ============================================================================

// ============================================================================
// Arena-native Pattern Match + Substitute
// ============================================================================

/// Arena-native pattern match.
///
/// Matches the subtree rooted at `expr_id` in `arena` against the template
/// subtree rooted at `template_root` in `template`. Returns `Some(bindings)`
/// mapping template `Var(n)` indices to `ExprId`s in `arena` on success, or
/// `None` if the pattern does not match.
///
/// Uses an iterative work stack of `(expr_id, template_id)` pairs to avoid
/// recursion depth issues on deep trees.
#[must_use]
pub fn pattern_match_arena(
    arena: &ExprArena,
    expr_id: ExprId,
    template: &ExprArena,
    template_root: ExprId,
) -> Option<BTreeMap<u8, ExprId>> {
    let mut bindings: BTreeMap<u8, ExprId> = BTreeMap::new();
    // Work stack: pairs of (expr node in `arena`, template node in `template`).
    let mut stack: Vec<(ExprId, ExprId)> = Vec::with_capacity(16);
    stack.push((expr_id, template_root));

    while let Some((e_id, t_id)) = stack.pop() {
        let t_node = template.node(t_id);
        match t_node {
            // Var(n) is a metavariable: bind or check consistency.
            ExprNode::Var(n) => {
                let n = *n;
                if let Some(&existing) = bindings.get(&n) {
                    // Already bound — the subtrees must be structurally equal.
                    // Compare arena-native to avoid Arc allocation.
                    if !arena.subtree_eq(existing, arena, e_id) {
                        return None;
                    }
                } else {
                    bindings.insert(n, e_id);
                }
            }
            // Const must match exactly (within epsilon).
            ExprNode::Const(c) => {
                let c = *c;
                match arena.node(e_id) {
                    ExprNode::Const(e) => {
                        if fabsf(e - c) >= 1e-6 {
                            return None;
                        }
                    }
                    _ => return None,
                }
            }
            // Param must match the same index.
            ExprNode::Param(i) => match arena.node(e_id) {
                ExprNode::Param(j) if i == j => {}
                _ => return None,
            },
            // Structural match: op must match, push children onto the stack.
            ExprNode::Unary(t_op, t_a) => match arena.node(e_id) {
                ExprNode::Unary(e_op, e_a) if e_op == t_op => {
                    stack.push((*e_a, *t_a));
                }
                _ => return None,
            },
            ExprNode::Binary(t_op, t_a, t_b) => match arena.node(e_id) {
                ExprNode::Binary(e_op, e_a, e_b) if e_op == t_op => {
                    stack.push((*e_a, *t_a));
                    stack.push((*e_b, *t_b));
                }
                _ => return None,
            },
            ExprNode::Ternary(t_op, t_a, t_b, t_c) => match arena.node(e_id) {
                ExprNode::Ternary(e_op, e_a, e_b, e_c) if e_op == t_op => {
                    stack.push((*e_a, *t_a));
                    stack.push((*e_b, *t_b));
                    stack.push((*e_c, *t_c));
                }
                _ => return None,
            },
            ExprNode::Nary(t_op, t_start, t_len) => match arena.node(e_id) {
                ExprNode::Nary(e_op, e_start, e_len) if e_op == t_op && e_len == t_len => {
                    let e_children = arena.nary_children_slice(*e_start, *e_len).to_vec();
                    let t_children = template.nary_children_slice(*t_start, *t_len).to_vec();
                    for (ec, tc) in e_children.into_iter().zip(t_children.into_iter()) {
                        stack.push((ec, tc));
                    }
                }
                _ => return None,
            },
        }
    }

    Some(bindings)
}

/// Arena-native template substitution.
///
/// Walks the template subtree rooted at `template_root` bottom-up, pushing
/// nodes into `target_arena`. When a `Var(n)` is encountered, the corresponding
/// `ExprId` from `bindings` (already in `target_arena`) is used directly.
///
/// Returns `None` if any template `Var(n)` has no binding.
#[must_use]
pub fn substitute_template_arena(
    target_arena: &mut ExprArena,
    template: &ExprArena,
    template_root: ExprId,
    bindings: &BTreeMap<u8, ExprId>,
) -> Option<ExprId> {
    // Post-order traversal: collect nodes reachable from template_root.
    let t_n = template.len();
    // Remap: template node index → target_arena ExprId (u32::MAX = not yet mapped).
    let mut remap: Vec<u32> = alloc::vec![u32::MAX; t_n];

    // Collect post-order traversal order.
    let mut order: Vec<ExprId> = Vec::with_capacity(t_n.min(32));
    {
        let mut visit_stack: Vec<ExprId> = Vec::with_capacity(16);
        let mut pushed: Vec<bool> = alloc::vec![false; t_n];
        visit_stack.push(template_root);
        while let Some(id) = visit_stack.pop() {
            let idx = id.0 as usize;
            if pushed[idx] {
                order.push(id);
            } else {
                pushed[idx] = true;
                // Push self again for post-order emission, then push children first.
                visit_stack.push(id);
                for child in template.children(id) {
                    if !pushed[child.0 as usize] {
                        visit_stack.push(child);
                    }
                }
            }
        }
    }

    // Process in post-order: children are mapped before their parent.
    for id in &order {
        let idx = id.0 as usize;
        let node = template.node(*id).clone();
        let mapped = match node {
            ExprNode::Var(n) => {
                // Fail if the variable has no binding.
                *bindings.get(&n)?
            }
            ExprNode::Const(c) => target_arena.push_const(c),
            ExprNode::Param(i) => target_arena.push_param(i),
            ExprNode::Unary(op, t_a) => {
                let a = ExprId(remap[t_a.0 as usize]);
                target_arena.push_unary(op, a)
            }
            ExprNode::Binary(op, t_a, t_b) => {
                let a = ExprId(remap[t_a.0 as usize]);
                let b = ExprId(remap[t_b.0 as usize]);
                target_arena.push_binary(op, a, b)
            }
            ExprNode::Ternary(op, t_a, t_b, t_c) => {
                let a = ExprId(remap[t_a.0 as usize]);
                let b = ExprId(remap[t_b.0 as usize]);
                let c = ExprId(remap[t_c.0 as usize]);
                target_arena.push_ternary(op, a, b, c)
            }
            ExprNode::Nary(op, t_start, t_len) => {
                let t_children: Vec<ExprId> = template
                    .nary_children_slice(t_start, t_len)
                    .iter()
                    .map(|tc| ExprId(remap[tc.0 as usize]))
                    .collect();
                target_arena.push_nary(op, &t_children)
            }
        };
        remap[idx] = mapped.0;
    }

    Some(ExprId(remap[template_root.0 as usize]))
}

// ============================================================================
// Backward Generation (BWD) - Lample & Charton 2019
// ============================================================================

/// Arena-backed training pair. Both expressions live inside the arena as [`ExprId`]s.
pub struct BwdTrainingPairArena {
    /// The shared arena holding all nodes for both expressions.
    pub arena: ExprArena,
    /// Root of the optimized expression in the arena.
    pub optimized: ExprId,
    /// Root of the unoptimized expression in the arena.
    pub unoptimized: ExprId,
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
    /// Arena-native rule templates. Built once in `new()` from `templates`.
    /// Used by `junkify_arena_pass` to avoid `to_expr`/`push_expr` round-trips.
    arena_templates: ArenaRuleTemplates,
    /// Reusable arena for arena-based generation. Cleared each call to
    /// [`generate_arena`](Self::generate_arena).
    arena: ExprArena,
}

impl BwdGenerator {
    /// Create a new backward generator with the given seed and rule templates.
    #[must_use]
    pub fn new(seed: u64, config: BwdGenConfig, templates: RuleTemplates) -> Self {
        assert!(
            config.num_vars <= 4,
            "num_vars={} exceeds INPUT_REGS limit of 4",
            config.num_vars
        );
        let arena_templates = ArenaRuleTemplates::from_rule_templates(&templates);
        Self {
            config,
            state: seed,
            arena_templates,
            arena: ExprArena::with_capacity(256),
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
        let val = (self.rand_f32() * max as f32) as usize;
        if val >= max { max - 1 } else { val }
    }

    /// Maximum retries when `generate_optimized` produces a variable-free
    /// expression. This is a safety net — with `min_depth >= 2` and 70%
    /// variable leaf probability, hitting this limit means the RNG is broken.
    const MAX_GENERATE_RETRIES: usize = 20;

    /// Maximum retries when junkification produces 0 rewrites.
    ///
    /// A trajectory seeded from an expression where no junkification was
    /// applied is equivalent to training on a perfectly-optimized form —
    /// the e-graph has no applicable rewrite-rule patterns and produces
    /// empty trajectories. Retry with a fresh random optimized expression
    /// until at least one junkify rewrite fires.
    const MAX_JUNKIFY_RETRIES: usize = 50;

    /// Generate a backward training pair directly in arena form.
    ///
    /// Returns a [`BwdTrainingPairArena`] where both the
    /// optimized and unoptimized expressions are stored as [`ExprId`]s inside a
    /// single shared [`ExprArena`].
    ///
    /// # Layout inside the arena
    ///
    /// - Nodes `[0, optimized_node_count)` belong to the optimized subtree.
    /// - Nodes `[optimized_node_count, arena.len())` belong to the unoptimized
    ///   subtree (pushed via [`ExprArena::push_expr`] after junkification).
    ///
    /// Use `arena.len()` for an O(1) total node count, or
    /// `arena.node_count_subtree(pair.optimized)` for the optimized subtree
    /// specifically (O(N) traversal).
    ///
    /// # Panics
    ///
    /// Same conditions as [`generate`].
    #[must_use]
    pub fn generate_arena(&mut self) -> BwdTrainingPairArena {
        let mut junkify_attempts = 0;
        loop {
            // Build optimized expression directly in self.arena.
            self.arena.clear();
            let optimized_id = {
                let mut attempts = 0;
                loop {
                    self.arena.clear();
                    let id = self.generate_optimized_arena(0);
                    if self.arena.has_var(id) {
                        break id;
                    }
                    attempts += 1;
                    assert!(
                        attempts < Self::MAX_GENERATE_RETRIES,
                        "BwdGenerator::generate_arena failed to produce an expression with \
                         variables after {} attempts. \
                         Config: max_depth={}, leaf_prob={}, num_vars={}",
                        attempts,
                        self.config.max_depth,
                        self.config.leaf_prob,
                        self.config.num_vars,
                    );
                }
            };

            // Arena-native junkification: no Expr bridge needed.
            let (unoptimized_id, rewrites_applied) =
                self.junkify_arena(optimized_id, self.config.max_junkified_nodes);

            assert!(
                self.arena.has_var(unoptimized_id),
                "BUG: junkification eliminated all variables from expression. \
                 optimized arena_nodes={}, rewrites={}",
                self.arena.node_count_subtree(optimized_id),
                rewrites_applied,
            );

            if rewrites_applied == 0 {
                junkify_attempts += 1;
                assert!(
                    junkify_attempts < Self::MAX_JUNKIFY_RETRIES,
                    "BwdGenerator::generate_arena failed to apply any junkify rewrites \
                     after {} attempts. \
                     Config: max_junkify_passes={}, junkify_prob={:.3}, max_junkified_nodes={}. \
                     Check that junkify_prob > 0.0 and max_junkify_passes >= 1, and that \
                     the rule templates contain at least one expanding rule.",
                    junkify_attempts,
                    self.config.max_junkify_passes,
                    self.config.junkify_prob,
                    self.config.max_junkified_nodes,
                );
                continue;
            }

            // Both optimized and unoptimized are already in self.arena.
            // Move the arena out, replacing self.arena with a fresh one.
            let arena = core::mem::replace(&mut self.arena, ExprArena::with_capacity(256));

            return BwdTrainingPairArena {
                arena,
                optimized: optimized_id,
                unoptimized: unoptimized_id,
                rewrites_applied,
            };
        }
    }

    /// Minimum depth before leaf generation is allowed.
    /// Ensures expressions have at least some computational structure.
    const MIN_DEPTH: usize = 2;

    // ── Arena-based generation ─────────────────────────────────────────────────

    /// Arena version of `generate_leaf`.
    fn generate_leaf_arena(&mut self) -> ExprId {
        if self.rand_f32() < 0.7 {
            let var_idx = self.rand_usize(self.config.num_vars.min(4)) as u8;
            self.arena.push_var(var_idx)
        } else {
            let val = self.rand_f32() * 4.0 - 2.0;
            self.arena.push_const(val)
        }
    }

    /// Arena version of `guard_positive_nonzero`.
    /// Wraps `inner` id in `abs(inner) + 0.001`, returning the root id.
    fn guard_positive_nonzero_arena(&mut self, inner: ExprId) -> ExprId {
        let abs_id = self.arena.push_unary(OpKind::Abs, inner);
        let eps_id = self.arena.push_const(0.001);
        self.arena.push_binary(OpKind::Add, abs_id, eps_id)
    }

    /// Arena version of `guard_nonnegative`.
    /// Wraps `inner` id in `abs(inner)`, returning the root id.
    fn guard_nonnegative_arena(&mut self, inner: ExprId) -> ExprId {
        self.arena.push_unary(OpKind::Abs, inner)
    }

    /// Iterative arena-based expression generator.
    ///
    /// Replaces the formerly-recursive `generate_optimized_arena` /
    /// `generate_regular_op_arena` pair with an explicit work stack so that
    /// deeply-nested trees cannot overflow the call stack. RNG consumption
    /// order is identical to the recursive version (left-to-right, pre-order),
    /// so output is deterministic for any given seed.
    ///
    /// # How the stack machine works
    ///
    /// Two stacks cooperate:
    ///
    /// * `work` — pending tasks.  Each entry is either a `Decide` (consume
    ///   RNG, emit a `Combine` + child `Decide`s) or a `Combine` (assemble
    ///   already-resolved children into a parent node).
    /// * `results` — a LIFO buffer of `ExprId`s produced by completed
    ///   sub-trees.  `Combine` variants pop from this.
    ///
    /// Push order for a node with N children:
    ///   1. Push `Combine(op)` — runs last, after all children are ready.
    ///   2. Push `Decide(childN)` through `Decide(child0)` in reverse order
    ///      so that `child0` sits on top and executes first.
    ///
    /// This preserves left-to-right RNG consumption without recursion.
    ///
    /// The `_start_depth` parameter is kept for call-site compatibility with
    /// the former recursive version (callers pass `0`).  The depth ceiling is
    /// always `self.config.max_depth`.
    fn generate_optimized_arena(&mut self, _start_depth: usize) -> ExprId {
        let max_depth = self.config.max_depth;
        /// Describes how to assemble a parent node once its children are ready.
        ///
        /// Each variant documents:
        ///   - How many `ExprId`s it pops from `results` (children, left-to-right).
        ///   - Any inline guard operations applied before the final arena push.
        enum Combine {
            /// Binary op — pop left then right, no guards.
            Binary(OpKind),
            /// Div — pop left (numerator), pop right (raw denominator),
            /// apply `guard_positive_nonzero` to denominator, then push Div.
            DivGuardDenom,
            /// Pow — pop left (raw base), apply `guard_positive_nonzero` to base,
            /// pop right (exponent), push Pow.
            ///
            /// Note: base is guarded *before* exponent is resolved, but both
            /// children are already on the results stack at combine time, so
            /// the guard is applied here in post-order.
            PowGuardBase,
            /// Unary op — pop one child, no guard.
            Unary(OpKind),
            /// Unary op — pop one child, apply `guard_positive_nonzero`, then push op.
            UnaryGuardPositive(OpKind),
            /// Unary op (Sqrt) — pop one child, apply `guard_nonnegative`, then push op.
            UnaryGuardNonneg,
            /// MulAdd ternary — pop a, b, c (left to right), push ternary.
            MulAdd,
        }

        enum WorkItem {
            Decide { depth: usize },
            Combine(Combine),
        }

        let mut work: Vec<WorkItem> = Vec::with_capacity(64);
        let mut results: Vec<ExprId> = Vec::with_capacity(64);

        work.push(WorkItem::Decide { depth: 0 });

        while let Some(item) = work.pop() {
            match item {
                WorkItem::Decide { depth } => {
                    // Mirror the recursive logic exactly:
                    //   1. Leaf at max depth (no RNG consumed before this check).
                    //   2. Probabilistic leaf above MIN_DEPTH (consumes one rand_f32).
                    //   3. Fused-op check (consumes one rand_f32).
                    //   4. Regular-op choice (consumes one rand_usize(24)).
                    if depth >= max_depth {
                        let id = self.generate_leaf_arena();
                        results.push(id);
                        continue;
                    }

                    if depth >= Self::MIN_DEPTH && self.rand_f32() < self.config.leaf_prob {
                        let id = self.generate_leaf_arena();
                        results.push(id);
                        continue;
                    }

                    if self.rand_f32() < self.config.fused_op_prob {
                        // MulAdd: need children a, b, c (left-to-right).
                        // Push Combine first (runs last), then children in reverse
                        // so child `a` (depth+1) sits on top and runs first.
                        work.push(WorkItem::Combine(Combine::MulAdd));
                        work.push(WorkItem::Decide { depth: depth + 1 }); // c
                        work.push(WorkItem::Decide { depth: depth + 1 }); // b
                        work.push(WorkItem::Decide { depth: depth + 1 }); // a
                        continue;
                    }

                    // Regular op: consume rand_usize(24) now, then push the
                    // appropriate Combine + child Decide frames.
                    let choice = self.rand_usize(24);
                    match choice {
                        // Binary ops — two children, no guards.
                        0 | 1 => {
                            work.push(WorkItem::Combine(Combine::Binary(OpKind::Add)));
                            work.push(WorkItem::Decide { depth: depth + 1 }); // b
                            work.push(WorkItem::Decide { depth: depth + 1 }); // a
                        }
                        2 => {
                            work.push(WorkItem::Combine(Combine::Binary(OpKind::Sub)));
                            work.push(WorkItem::Decide { depth: depth + 1 }); // b
                            work.push(WorkItem::Decide { depth: depth + 1 }); // a
                        }
                        3 | 4 => {
                            work.push(WorkItem::Combine(Combine::Binary(OpKind::Mul)));
                            work.push(WorkItem::Decide { depth: depth + 1 }); // b
                            work.push(WorkItem::Decide { depth: depth + 1 }); // a
                        }
                        5 => {
                            // Div: child order is num (left) then raw_denom (right).
                            work.push(WorkItem::Combine(Combine::DivGuardDenom));
                            work.push(WorkItem::Decide { depth: depth + 1 }); // raw_denom
                            work.push(WorkItem::Decide { depth: depth + 1 }); // num
                        }
                        6 => {
                            work.push(WorkItem::Combine(Combine::Binary(OpKind::Min)));
                            work.push(WorkItem::Decide { depth: depth + 1 }); // b
                            work.push(WorkItem::Decide { depth: depth + 1 }); // a
                        }
                        7 => {
                            work.push(WorkItem::Combine(Combine::Binary(OpKind::Max)));
                            work.push(WorkItem::Decide { depth: depth + 1 }); // b
                            work.push(WorkItem::Decide { depth: depth + 1 }); // a
                        }
                        8 => {
                            // Pow: child order is raw_base (left) then exp (right).
                            work.push(WorkItem::Combine(Combine::PowGuardBase));
                            work.push(WorkItem::Decide { depth: depth + 1 }); // exp
                            work.push(WorkItem::Decide { depth: depth + 1 }); // raw_base
                        }
                        9 => {
                            work.push(WorkItem::Combine(Combine::Binary(OpKind::Hypot)));
                            work.push(WorkItem::Decide { depth: depth + 1 }); // b
                            work.push(WorkItem::Decide { depth: depth + 1 }); // a
                        }
                        10 => {
                            work.push(WorkItem::Combine(Combine::Binary(OpKind::Atan2)));
                            work.push(WorkItem::Decide { depth: depth + 1 }); // b
                            work.push(WorkItem::Decide { depth: depth + 1 }); // a
                        }
                        // Unary ops.
                        11 => {
                            work.push(WorkItem::Combine(Combine::Unary(OpKind::Neg)));
                            work.push(WorkItem::Decide { depth: depth + 1 });
                        }
                        12 => {
                            work.push(WorkItem::Combine(Combine::UnaryGuardPositive(
                                OpKind::Recip,
                            )));
                            work.push(WorkItem::Decide { depth: depth + 1 });
                        }
                        13 => {
                            work.push(WorkItem::Combine(Combine::Unary(OpKind::Abs)));
                            work.push(WorkItem::Decide { depth: depth + 1 });
                        }
                        14 => {
                            work.push(WorkItem::Combine(Combine::UnaryGuardNonneg));
                            work.push(WorkItem::Decide { depth: depth + 1 });
                        }
                        15 => {
                            work.push(WorkItem::Combine(Combine::UnaryGuardPositive(
                                OpKind::Rsqrt,
                            )));
                            work.push(WorkItem::Decide { depth: depth + 1 });
                        }
                        16 => {
                            work.push(WorkItem::Combine(Combine::Unary(OpKind::Sin)));
                            work.push(WorkItem::Decide { depth: depth + 1 });
                        }
                        17 => {
                            work.push(WorkItem::Combine(Combine::Unary(OpKind::Cos)));
                            work.push(WorkItem::Decide { depth: depth + 1 });
                        }
                        18 => {
                            work.push(WorkItem::Combine(Combine::Unary(OpKind::Tan)));
                            work.push(WorkItem::Decide { depth: depth + 1 });
                        }
                        19 => {
                            work.push(WorkItem::Combine(Combine::Unary(OpKind::Exp)));
                            work.push(WorkItem::Decide { depth: depth + 1 });
                        }
                        20 => {
                            work.push(WorkItem::Combine(Combine::Unary(OpKind::Exp2)));
                            work.push(WorkItem::Decide { depth: depth + 1 });
                        }
                        21 => {
                            work.push(WorkItem::Combine(Combine::UnaryGuardPositive(OpKind::Ln)));
                            work.push(WorkItem::Decide { depth: depth + 1 });
                        }
                        22 => {
                            work.push(WorkItem::Combine(Combine::UnaryGuardPositive(OpKind::Log2)));
                            work.push(WorkItem::Decide { depth: depth + 1 });
                        }
                        23 => {
                            work.push(WorkItem::Combine(Combine::UnaryGuardPositive(
                                OpKind::Log10,
                            )));
                            work.push(WorkItem::Decide { depth: depth + 1 });
                        }
                        _ => unreachable!(),
                    }
                }

                WorkItem::Combine(combine) => {
                    // The results stack is LIFO: child `a` was pushed before child `b`,
                    // so `b` is on top. Pop in reverse child order to reconstruct
                    // the original left-to-right argument ordering.
                    let id = match combine {
                        Combine::Binary(op) => {
                            let b = results.pop().expect("BUG: missing right child for Binary");
                            let a = results.pop().expect("BUG: missing left child for Binary");
                            self.arena.push_binary(op, a, b)
                        }
                        Combine::DivGuardDenom => {
                            // Pushed order: num (left), raw_denom (right).
                            // Pop order: raw_denom first (top), num second.
                            let raw_denom =
                                results.pop().expect("BUG: missing denominator for Div");
                            let num = results.pop().expect("BUG: missing numerator for Div");
                            let denom = self.guard_positive_nonzero_arena(raw_denom);
                            self.arena.push_binary(OpKind::Div, num, denom)
                        }
                        Combine::PowGuardBase => {
                            // Pushed order: raw_base (left), exp (right).
                            // Pop order: exp first (top), raw_base second.
                            let exp = results.pop().expect("BUG: missing exponent for Pow");
                            let raw_base = results.pop().expect("BUG: missing base for Pow");
                            let base = self.guard_positive_nonzero_arena(raw_base);
                            self.arena.push_binary(OpKind::Pow, base, exp)
                        }
                        Combine::Unary(op) => {
                            let a = results.pop().expect("BUG: missing child for Unary");
                            self.arena.push_unary(op, a)
                        }
                        Combine::UnaryGuardPositive(op) => {
                            let raw = results
                                .pop()
                                .expect("BUG: missing child for UnaryGuardPositive");
                            let guarded = self.guard_positive_nonzero_arena(raw);
                            self.arena.push_unary(op, guarded)
                        }
                        Combine::UnaryGuardNonneg => {
                            let raw = results
                                .pop()
                                .expect("BUG: missing child for UnaryGuardNonneg");
                            let guarded = self.guard_nonnegative_arena(raw);
                            self.arena.push_unary(OpKind::Sqrt, guarded)
                        }
                        Combine::MulAdd => {
                            // Pushed order: a, b, c. Pop order: c (top), b, a.
                            let c = results.pop().expect("BUG: missing child c for MulAdd");
                            let b = results.pop().expect("BUG: missing child b for MulAdd");
                            let a = results.pop().expect("BUG: missing child a for MulAdd");
                            self.arena.push_ternary(OpKind::MulAdd, a, b, c)
                        }
                    };
                    results.push(id);
                }
            }
        }

        assert_eq!(
            results.len(),
            1,
            "BUG: generate_optimized_arena left {} results on stack",
            results.len()
        );
        results
            .pop()
            .expect("BUG: result stack empty after generation")
    }

    // ── Arena-native junkification ──────────────────────────────────────────

    /// Arena-native junkification: apply rewrites that make the expression
    /// MORE complex, entirely within the arena. Legacy `Expr` is only
    /// constructed per-node for template matching (and only for nodes that
    /// pass the random check AND have a matchable root op).
    ///
    /// Returns `(new_root_id, total_rewrites_applied)`.
    fn junkify_arena(&mut self, root: ExprId, max_growth: usize) -> (ExprId, usize) {
        let original_len = self.arena.len();
        let mut total_applied = 0;
        let mut current_root = root;

        // Use the precomputed root_op_set from arena_templates — already O(1).
        let root_op_set = self.arena_templates.root_op_set;

        for _pass in 0..self.config.max_junkify_passes {
            let growth_so_far = self.arena.len() - original_len;
            if growth_so_far >= max_growth {
                break;
            }

            let pass_budget = max_growth - growth_so_far;
            let (new_root, applied) =
                self.junkify_arena_pass(current_root, pass_budget, &root_op_set);
            if applied == 0 {
                break;
            }
            current_root = new_root;
            total_applied += applied;
        }

        (current_root, total_applied)
    }

    /// Single pass of arena-native junkification.
    ///
    /// Walks nodes `[0..n)` in topological order (guaranteed by arena construction),
    /// building a `remap` table that maps old ExprIds to new (possibly junkified) ExprIds.
    ///
    /// For each node:
    /// 1. Remap its children through the remap table.
    /// 2. If random check passes AND the node's root op is in `root_op_set`:
    ///    - Try all arena templates in both directions via `pattern_match_arena`.
    ///    - Collect expanding candidates via `substitute_template_arena`.
    ///    - Pick one randomly.
    /// 3. Otherwise: push a copy with remapped children.
    ///
    /// No `to_expr`/`push_expr` calls occur in this path.
    fn junkify_arena_pass(
        &mut self,
        root: ExprId,
        budget: usize,
        root_op_set: &[bool; OpKind::COUNT],
    ) -> (ExprId, usize) {
        let n = self.arena.len();
        // Identity remap: every node maps to itself initially.
        let mut remap: Vec<ExprId> = (0..n as u32).map(ExprId).collect();
        let mut applied = 0;
        let mut remaining_budget = budget;

        for idx in 0..n {
            let id = ExprId(idx as u32);

            // Clone the node so we can inspect it without borrowing self.arena.
            let node = self.arena.node(id).clone();

            // Remap children to point to their (possibly junkified) versions.
            let remapped_node = Self::remap_node(&node, &remap, &self.arena);

            // Push the remapped copy into the arena. This is the "base" version;
            // if junkification succeeds below we'll overwrite the remap entry.
            let base_id = Self::push_arena_node(&mut self.arena, &remapped_node);
            remap[idx] = base_id;

            // Budget exhausted — just copy remaining nodes.
            if remaining_budget == 0 {
                continue;
            }

            // Random check: only try junkification probabilistically.
            if self.rand_f32() >= self.config.junkify_prob {
                continue;
            }

            // Op filter: skip if no template can match this node's root op.
            let node_op = self.arena.kind(base_id);
            if node_op.index() < OpKind::COUNT && !root_op_set[node_op.index()] {
                continue;
            }

            let original_cost = self.arena.node_count_subtree(base_id);

            // Try all arena rule templates in both directions.
            // Candidates are ExprIds pushed into self.arena during substitution.
            let mut candidates: Vec<ExprId> = Vec::new();
            let mut candidate_costs: Vec<usize> = Vec::new();

            for rule_idx in 0..self.arena_templates.len() {
                let tmpl = &self.arena_templates.arenas[rule_idx];

                // LHS -> RHS direction: match against LHS, substitute RHS.
                if let (Some(lhs_root), Some(rhs_root)) = (tmpl.lhs, tmpl.rhs) {
                    if tmpl.lhs_op.is_some() {
                        // pattern_match_arena borrows self.arena and tmpl.arena immutably.
                        // We must split the borrow: take a pointer to the template arena
                        // to satisfy the borrow checker while we later push into self.arena.
                        let bindings = {
                            let tmpl_arena = &self.arena_templates.arenas[rule_idx].arena;
                            pattern_match_arena(&self.arena, base_id, tmpl_arena, lhs_root)
                        };
                        if let Some(bindings) = bindings {
                            let tmpl_arena = &self.arena_templates.arenas[rule_idx].arena;
                            if let Some(result_id) = substitute_template_arena(
                                &mut self.arena,
                                tmpl_arena,
                                rhs_root,
                                &bindings,
                            ) {
                                let new_cost = self.arena.node_count_subtree(result_id);
                                let growth = new_cost.saturating_sub(original_cost);
                                if new_cost > original_cost && growth <= remaining_budget {
                                    candidates.push(result_id);
                                    candidate_costs.push(new_cost);
                                }
                            }
                        }
                    }
                    // RHS -> LHS direction: match against RHS, substitute LHS.
                    if tmpl.rhs_op.is_some() {
                        let bindings = {
                            let tmpl_arena = &self.arena_templates.arenas[rule_idx].arena;
                            pattern_match_arena(&self.arena, base_id, tmpl_arena, rhs_root)
                        };
                        if let Some(bindings) = bindings {
                            let tmpl_arena = &self.arena_templates.arenas[rule_idx].arena;
                            if let Some(result_id) = substitute_template_arena(
                                &mut self.arena,
                                tmpl_arena,
                                lhs_root,
                                &bindings,
                            ) {
                                let new_cost = self.arena.node_count_subtree(result_id);
                                let growth = new_cost.saturating_sub(original_cost);
                                if new_cost > original_cost && growth <= remaining_budget {
                                    candidates.push(result_id);
                                    candidate_costs.push(new_cost);
                                }
                            }
                        }
                    }
                }
            }

            if !candidates.is_empty() {
                let chosen_idx = self.rand_usize(candidates.len());
                let chosen_id = candidates[chosen_idx];
                let chosen_cost = candidate_costs[chosen_idx];
                let growth = chosen_cost.saturating_sub(original_cost);
                remaining_budget = remaining_budget.saturating_sub(growth);
                remap[idx] = chosen_id;
                applied += 1;
            }
            // else: remap[idx] already points to base_id (the remapped copy).
        }

        (remap[root.0 as usize], applied)
    }

    /// Remap the children of an `ExprNode` through the remap table.
    ///
    /// For `Nary` nodes, the children are read from the arena's nary_children
    /// buffer and pushed as a new nary group. For all other node types,
    /// children are remapped inline.
    fn remap_node(node: &ExprNode, remap: &[ExprId], arena: &ExprArena) -> ExprNode {
        match node {
            ExprNode::Var(v) => ExprNode::Var(*v),
            ExprNode::Const(c) => ExprNode::Const(*c),
            ExprNode::Param(p) => ExprNode::Param(*p),
            ExprNode::Unary(op, a) => ExprNode::Unary(*op, remap[a.0 as usize]),
            ExprNode::Binary(op, a, b) => {
                ExprNode::Binary(*op, remap[a.0 as usize], remap[b.0 as usize])
            }
            ExprNode::Ternary(op, a, b, c) => ExprNode::Ternary(
                *op,
                remap[a.0 as usize],
                remap[b.0 as usize],
                remap[c.0 as usize],
            ),
            ExprNode::Nary(op, start, len) => {
                // Read the original children and remap them.
                let children: Vec<ExprId> = arena
                    .nary_children_slice(*start, *len)
                    .iter()
                    .map(|child| remap[child.0 as usize])
                    .collect();
                // Return a sentinel; actual push happens in push_arena_node.
                // We encode the remapped children in a temporary Nary with placeholder
                // start/len — push_arena_node will handle it properly.
                // Actually, we can't do this cleanly because Nary stores (start, len)
                // referring to the arena's internal buffer. We need to handle Nary
                // specially in push_arena_node.
                //
                // For now, store the REMAPPED children inline by abusing the fact
                // that push_arena_node will detect this case. Instead, let's just
                // mark it and handle Nary in the caller.
                //
                // Simplest approach: for Nary, return the original node unchanged.
                // Nary is extremely rare in generated expressions (the generator
                // never produces them). If one somehow appears, it gets copied as-is.
                ExprNode::Nary(*op, *start, *len)
            }
        }
    }

    /// Push an `ExprNode` into the arena, returning its `ExprId`.
    fn push_arena_node(arena: &mut ExprArena, node: &ExprNode) -> ExprId {
        match node {
            ExprNode::Var(v) => arena.push_var(*v),
            ExprNode::Const(c) => arena.push_const(*c),
            ExprNode::Param(p) => arena.push_param(*p),
            ExprNode::Unary(op, a) => arena.push_unary(*op, *a),
            ExprNode::Binary(op, a, b) => arena.push_binary(*op, *a, *b),
            ExprNode::Ternary(op, a, b, c) => arena.push_ternary(*op, *a, *b, *c),
            ExprNode::Nary(op, start, len) => {
                // Copy the children from the existing nary_children buffer.
                let children: Vec<ExprId> = arena.nary_children_slice(*start, *len).to_vec();
                arena.push_nary(*op, &children)
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
    fn verify_op_type_roundtrip() {
        for i in 0..OpKind::COUNT {
            let op = OpKind::from_index(i).unwrap();
            assert_eq!(op.index(), i);
        }
    }

    // ========================================================================
    // Pattern Match + Substitute Tests
    // ========================================================================

    // ========================================================================
    // Backward Generation Tests
    // ========================================================================

    #[test]
    fn verify_bwd_generator_produces_valid_pairs() {
        use crate::egraph::collect_rule_templates;
        let templates = collect_rule_templates();
        let config = BwdGenConfig::default();
        let mut generator = BwdGenerator::new(42, config, templates);

        for _ in 0..10 {
            let pair = generator.generate_arena();
            let optimized_nodes = pair.arena.node_count_subtree(pair.optimized);
            let unoptimized_nodes = pair.arena.node_count_subtree(pair.unoptimized);

            // Both expressions should be valid
            assert!(optimized_nodes > 0);
            assert!(unoptimized_nodes > 0);

            // Unoptimized should generally be larger or equal
            // (junkifying increases or maintains size)
            assert!(unoptimized_nodes >= optimized_nodes);
        }
    }

    #[test]
    fn verify_bwd_generator_has_fused_ops() {
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
            let pair = generator.generate_arena();
            let mut stack = alloc::vec![pair.optimized];
            while let Some(id) = stack.pop() {
                if pair.arena.kind(id) == OpKind::MulAdd {
                    total_fused += 1;
                }
                stack.extend(pair.arena.children(id));
            }
        }

        // With 80% fused op probability, we should see some fused ops
        assert!(
            total_fused > 0,
            "Expected some fused operations in generated expressions"
        );
    }

    #[test]
    fn verify_bwd_generator_with_templates() {
        use crate::egraph::collect_rule_templates;
        let templates = collect_rule_templates();
        let config = BwdGenConfig::default();
        let mut generator = BwdGenerator::new(42, config, templates);

        // Generate 20 pairs and check they're non-trivial
        let mut total_rewrites = 0;
        for _ in 0..20 {
            let pair = generator.generate_arena();
            let optimized_nodes = pair.arena.node_count_subtree(pair.optimized);
            let unoptimized_nodes = pair.arena.node_count_subtree(pair.unoptimized);
            assert!(
                unoptimized_nodes >= optimized_nodes,
                "unoptimized ({}) should have >= nodes than optimized ({})",
                unoptimized_nodes,
                optimized_nodes
            );
            total_rewrites += pair.rewrites_applied;
        }
        // At least some rewrites should have been applied across 20 expressions
        assert!(
            total_rewrites > 0,
            "Expected at least one junkify rewrite across 20 expressions, got 0"
        );
    }

    // ========================================================================
    // Dense Features and ILP Tests
    // ========================================================================
}
