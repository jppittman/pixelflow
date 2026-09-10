//! # Learned components of the e-graph optimizer
//!
//! Op embeddings and the typed edge stream (`factored`), the saturation Guide
//! (`guide`), and the backward expression generator (`BwdGenerator`) that
//! mints rewrite-pair corpora. The extraction (value) head this module was
//! named for — an NNUE cost model for e-graph extraction — tied the static
//! table on schedule-free kernels (workshop paper on branch
//! `claude/workshop-writeup`, PR #1072, closed without merging — not in this
//! tree; see the denotation doc below for the citations and numbers in-repo)
//! and its shape was deleted on 2026-09-01; the static latency prior is the
//! extraction policy, and the seam a future schedule-cost residual plugs into
//! is `egraph::extract::Reranker`
//! (docs/plans/2026-09-01-schedule-cost-model-denotation.md).

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![allow(clippy::only_used_in_recursion)]

extern crate alloc;

pub mod factored;
pub mod guide;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use libm::fabsf;
use pixelflow_ir::Node;
use pixelflow_ir::expr::{Environment, ExprBuilder, ExprData, ExprRef, Term};
use pixelflow_ir::kind::OpMap;
use pixelflow_ir::{Rooted, expr};

/// Re-export canonical IR types as the source of truth.
pub use pixelflow_ir::OpKind;

/// Re-export key types from factored module.
pub use factored::{CostEdge, EdgeTrace, OpEmbeddings, PeSlot};

/// Re-export shared embedding-space constants and rule-template types.
///
/// `RuleTemplates`/`ArenaRuleTemplates` feed `BwdGenerator`'s corpus
/// junkification below — unrelated to `nnue::guide`'s rule encoding.
pub use factored::{ArenaRuleTemplates, EMBED_DIM, MLP_HIDDEN, RuleTemplates};

// Note: BwdGenConfig and BwdGenerator are already public structs defined in
// this module - no re-export needed.

// ============================================================================
// Rewrite Rules (as "Moves")
// ============================================================================

// ============================================================================
// Pattern Match + Substitute (for rule template rewriting)
// ============================================================================

// ============================================================================
// Pattern Match + Substitute over an expression graph
// ============================================================================

/// Structural pattern match.
///
/// Matches the subtree `expr` names in `target` against the template subtree
/// rooted at `template`. Returns `Some(bindings)` mapping template `Var(n)`
/// indices to nodes in `target` on success, or `None` if the pattern does not
/// match.
///
/// Uses an iterative work stack of `(target node, template node)` pairs to
/// avoid recursion depth issues on deep trees.
#[must_use]
pub fn pattern_match(
    target: &ExprBuilder,
    expr: ExprRef,
    template: Node<'_, ExprData>,
) -> Option<BTreeMap<u8, ExprRef>> {
    let mut bindings: BTreeMap<u8, ExprRef> = BTreeMap::new();
    let mut stack: Vec<(ExprRef, Node<'_, ExprData>)> = Vec::with_capacity(16);
    stack.push((expr, template));

    while let Some((e, t)) = stack.pop() {
        match *t {
            // Var(n) is a metavariable: bind or check consistency.
            ExprData::Var(n) => match bindings.get(&n) {
                // Already bound — the subtrees must be structurally equal.
                Some(&existing) => {
                    if !target.node(existing).subtree_eq(target.node(e)) {
                        return None;
                    }
                }
                None => {
                    bindings.insert(n, e);
                }
            },
            // Const must match within epsilon.
            ExprData::Const(bits) => match target.node(e).as_f32() {
                Some(v) if fabsf(v - f32::from_bits(bits)) < 1e-6 => {}
                _ => return None,
            },
            // A leaf that names something must name the same thing.
            ExprData::Param(_) | ExprData::Buffer(_) | ExprData::Uniform(_) => {
                if *target.node(e) != *t {
                    return None;
                }
            }
            // Structural match: same op and arity, push children onto the stack.
            ExprData::Op(op) => {
                let node = target.node(e);
                if node.op() != Some(op) || node.child_count() != t.child_count() {
                    return None;
                }
                for (ec, tc) in target.child_refs(e).iter().zip(t.children()) {
                    stack.push((*ec, tc));
                }
            }
        }
    }

    Some(bindings)
}

/// Template substitution.
///
/// Walks the template subtree rooted at `template` bottom-up, pushing nodes
/// into `target`. When a `Var(n)` is encountered, the corresponding `ExprRef`
/// from `bindings` (already in `target`) is used directly.
///
/// Returns `None` if any template `Var(n)` has no binding.
///
/// # Panics
///
/// Panics on a `Buffer` or `Uniform` leaf: those name memory, which no rewrite
/// template rewrites.
#[must_use]
pub fn substitute_template(
    target: &mut ExprBuilder,
    template: Node<'_, ExprData>,
    bindings: &BTreeMap<u8, ExprRef>,
) -> Option<ExprRef> {
    let mut memo: BTreeMap<Node<'_, ExprData>, ExprRef> = BTreeMap::new();
    let mut stack: Vec<(Node<'_, ExprData>, bool)> = alloc::vec![(template, false)];

    while let Some((node, children_done)) = stack.pop() {
        if memo.contains_key(&node) {
            continue;
        }
        if !children_done {
            stack.push((node, true));
            for child in node.children() {
                if !memo.contains_key(&child) {
                    stack.push((child, false));
                }
            }
            continue;
        }
        let mapped = match *node {
            // Fail if the variable has no binding.
            ExprData::Var(n) => *bindings.get(&n)?,
            ExprData::Const(bits) => target.push_const(f32::from_bits(bits)),
            ExprData::Param(i) => target.push_param(i),
            ExprData::Buffer(b) => panic!(
                "Buffer({}) in a rewrite template — memory ops are not rewritable yet",
                b.0
            ),
            ExprData::Uniform(u) => panic!(
                "Uniform({}) in a rewrite template — uniforms are not rewritable",
                u.0
            ),
            ExprData::Op(op) => {
                let kids: Vec<ExprRef> = node
                    .children()
                    .map(|c| memo[&c])
                    .collect();
                target.push_nary(op, &kids)
            }
        };
        memo.insert(node, mapped);
    }

    Some(memo[&template])
}

/// Which way a rewrite template is being read: match its LHS and produce its
/// RHS, or the mirror. Junkification wants the expanding direction whichever
/// one that is, so it tries both.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Direction {
    /// Match the LHS, produce the RHS.
    Forward,
    /// Match the RHS, produce the LHS.
    Backward,
}

// ============================================================================
// Backward Generation (BWD) - Lample & Charton 2019
// ============================================================================

/// A training pair: both expressions in one graph, as its two entries.
///
/// Entry 0 is the optimized form, entry 1 the junkified one. Two entries of
/// one [`Rooted`] rather than two graphs, because the junkified form is built
/// *out of* the optimized one — they share nodes, and splitting them would
/// duplicate the shared subterms and lose the sharing that makes the pair a
/// pair.
pub struct BwdTrainingPair {
    /// The graph holding both expressions: `[optimized, unoptimized]`.
    pub rooted: Rooted<ExprData>,
    /// The declaration tables the leaves index. Empty — a generated
    /// expression names no memory — but carried so a [`Term`] can be formed.
    pub env: Environment,
    /// Number of junkifying rewrites applied.
    pub rewrites_applied: usize,
}

impl BwdTrainingPair {
    /// The optimized expression.
    #[must_use]
    pub fn optimized(&self) -> Term<'_> {
        Term::new(self.rooted.entry_at(0), &self.env)
    }

    /// The junkified (unoptimized) expression.
    #[must_use]
    pub fn unoptimized(&self) -> Term<'_> {
        Term::new(self.rooted.entry_at(1), &self.env)
    }
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
    /// Rule templates as expression graphs. Built once in `new()` from
    /// `templates`.
    arena_templates: ArenaRuleTemplates,
    /// The graph under construction. Replaced by a fresh one each call to
    /// [`generate`](Self::generate).
    arena: ExprBuilder,
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
            arena: ExprBuilder::new(),
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

    /// Generate a backward training pair.
    ///
    /// Returns a [`BwdTrainingPair`] whose two entries are the optimized and
    /// junkified forms of one expression, sharing the subterms junkification
    /// left alone.
    ///
    /// # Panics
    ///
    /// If generation cannot produce an expression containing a variable, or
    /// junkification cannot fire a single rewrite, within the retry limits —
    /// either means the configuration or the RNG is broken, not that the
    /// corpus is merely unlucky.
    #[must_use]
    pub fn generate(&mut self) -> BwdTrainingPair {
        let mut junkify_attempts = 0;
        loop {
            // Build optimized expression directly in self.arena.
            self.arena = ExprBuilder::new();
            let optimized_id = {
                let mut attempts = 0;
                loop {
                    self.arena = ExprBuilder::new();
                    let id = self.generate_optimized_arena(0);
                    if self.arena.node(id).has_var() {
                        break id;
                    }
                    attempts += 1;
                    assert!(
                        attempts < Self::MAX_GENERATE_RETRIES,
                        "BwdGenerator::generate failed to produce an expression with \
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
                self.arena.node(unoptimized_id).has_var(),
                "BUG: junkification eliminated all variables from expression. \
                 optimized nodes={}, rewrites={}",
                self.arena.node(optimized_id).node_count(),
                rewrites_applied,
            );

            if rewrites_applied == 0 {
                junkify_attempts += 1;
                assert!(
                    junkify_attempts < Self::MAX_JUNKIFY_RETRIES,
                    "BwdGenerator::generate failed to apply any junkify rewrites \
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

            // Both forms are already in self.arena. Move the builder out,
            // replacing it with a fresh one, and freeze it at both roots.
            let builder = core::mem::replace(&mut self.arena, ExprBuilder::new());
            let (rooted, env) = builder.finish(&[optimized_id, unoptimized_id]);

            return BwdTrainingPair {
                rooted,
                env,
                rewrites_applied,
            };
        }
    }

    /// Minimum depth before leaf generation is allowed.
    /// Ensures expressions have at least some computational structure.
    const MIN_DEPTH: usize = 2;

    // ── Arena-based generation ─────────────────────────────────────────────────

    /// Arena version of `generate_leaf`.
    fn generate_leaf_arena(&mut self) -> ExprRef {
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
    fn guard_positive_nonzero_arena(&mut self, inner: ExprRef) -> ExprRef {
        let abs_id = self.arena.push_unary(OpKind::Abs, inner);
        let eps_id = self.arena.push_const(0.001);
        self.arena.push_binary(OpKind::Add, abs_id, eps_id)
    }

    /// Arena version of `guard_nonnegative`.
    /// Wraps `inner` id in `abs(inner)`, returning the root id.
    fn guard_nonnegative_arena(&mut self, inner: ExprRef) -> ExprRef {
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
    /// * `results` — a LIFO buffer of `ExprRef`s produced by completed
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
    fn generate_optimized_arena(&mut self, _start_depth: usize) -> ExprRef {
        let max_depth = self.config.max_depth;
        /// Describes how to assemble a parent node once its children are ready.
        ///
        /// Each variant documents:
        ///   - How many `ExprRef`s it pops from `results` (children, left-to-right).
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
            /// Hypot — pop left then right, push `sqrt(a² + b²)`. `hypot` is
            /// library, not an op, so the corpus generates the composition it
            /// denotes; the sampled expressions are the same mathematics the
            /// generator always covered, now in primitives.
            Hypot,
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
        let mut results: Vec<ExprRef> = Vec::with_capacity(64);

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
                            work.push(WorkItem::Combine(Combine::Hypot));
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
                        Combine::Hypot => {
                            let b = results.pop().expect("BUG: missing right child for Hypot");
                            let a = results.pop().expect("BUG: missing left child for Hypot");
                            let aa = self.arena.push_binary(OpKind::Mul, a, a);
                            let bb = self.arena.push_binary(OpKind::Mul, b, b);
                            let sum = self.arena.push_binary(OpKind::Add, aa, bb);
                            self.arena.push_unary(OpKind::Sqrt, sum)
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

    /// Junkification: apply rewrites that make the expression MORE complex,
    /// entirely within the builder.
    ///
    /// Returns `(new_root_id, total_rewrites_applied)`.
    fn junkify_arena(&mut self, root: ExprRef, max_growth: usize) -> (ExprRef, usize) {
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

    /// Single pass of junkification.
    ///
    /// Walks the subgraph reachable from `root` bottom-up (children before
    /// parents), building a `remap` from each old node to its new (possibly
    /// junkified) copy.
    ///
    /// For each node:
    /// 1. Remap its children through the remap table.
    /// 2. If random check passes AND the node's root op is in `root_op_set`:
    ///    - Try all rule templates in both directions via [`pattern_match`].
    ///    - Collect expanding candidates via [`substitute_template`].
    ///    - Pick one randomly.
    /// 3. Otherwise: keep the copy with remapped children.
    ///
    /// Reachable-only, where the arena version walked every node the builder
    /// had ever pushed: a node the current root does not reach can only ever
    /// be rebuilt into another node nothing reaches, so the answer is the
    /// same and the garbage is not re-junkified.
    fn junkify_arena_pass(
        &mut self,
        root: ExprRef,
        budget: usize,
        root_op_set: &OpMap<bool>,
    ) -> (ExprRef, usize) {
        let mut remap: BTreeMap<ExprRef, ExprRef> = BTreeMap::new();
        let mut applied = 0;
        let mut remaining_budget = budget;

        // Post-order over the reachable subgraph, so every child is remapped
        // before its parent is rebuilt.
        let mut order: Vec<ExprRef> = Vec::new();
        {
            let mut seen: alloc::collections::BTreeSet<ExprRef> =
                alloc::collections::BTreeSet::new();
            let mut stack: Vec<(ExprRef, bool)> = alloc::vec![(root, false)];
            while let Some((r, expanded)) = stack.pop() {
                if expanded {
                    order.push(r);
                    continue;
                }
                if !seen.insert(r) {
                    continue;
                }
                stack.push((r, true));
                for &child in self.arena.child_refs(r) {
                    if !seen.contains(&child) {
                        stack.push((child, false));
                    }
                }
            }
        }

        for &r in &order {
            // Rebuild this node over its (possibly junkified) children. This
            // is the "base" version; if junkification succeeds below the
            // remap entry is overwritten.
            let data = *self.arena.node(r);
            let kids: Vec<ExprRef> = self
                .arena
                .child_refs(r)
                .iter()
                .map(|c| remap[c])
                .collect();
            let base_id = match data {
                ExprData::Var(v) => self.arena.push_var(v),
                ExprData::Const(bits) => self.arena.push_const(f32::from_bits(bits)),
                ExprData::Param(p) => self.arena.push_param(p),
                // The slot is already declared in this very builder, so
                // naming it again is all a copy needs.
                ExprData::Buffer(b) => self.arena.push_buffer(b),
                ExprData::Uniform(u) => self.arena.push_uniform(u),
                ExprData::Op(op) => {
                    assert!(!kids.is_empty(), "junkify: op with 0 children");
                    self.arena.push_nary(op, &kids)
                }
            };
            remap.insert(r, base_id);

            // Budget exhausted — just copy remaining nodes.
            if remaining_budget == 0 {
                continue;
            }

            // Random check: only try junkification probabilistically.
            if self.rand_f32() >= self.config.junkify_prob {
                continue;
            }

            // Op filter: skip if no template can match this node's root op.
            let node_op = factored::kind_of(self.arena.node(base_id));
            if !root_op_set[node_op] {
                continue;
            }

            let original_cost = self.arena.node(base_id).node_count();

            // Try every rule template in both directions. Candidates are
            // nodes pushed into the builder during substitution.
            let mut candidates: Vec<ExprRef> = Vec::new();
            let mut candidate_costs: Vec<usize> = Vec::new();

            for rule_idx in 0..self.arena_templates.len() {
                // Both directions need both sides: one to match, one to
                // produce.
                let (lhs_op, rhs_op) = {
                    let tmpl = &self.arena_templates.arenas[rule_idx];
                    if tmpl.lhs().is_none() || tmpl.rhs().is_none() {
                        continue;
                    }
                    (tmpl.lhs_op, tmpl.rhs_op)
                };

                // LHS -> RHS direction: match against LHS, substitute RHS.
                // RHS -> LHS direction: the mirror. Both are the same three
                // steps, so they are one loop rather than two copies.
                for direction in [Direction::Forward, Direction::Backward] {
                    let matchable = match direction {
                        Direction::Forward => lhs_op.is_some(),
                        Direction::Backward => rhs_op.is_some(),
                    };
                    if !matchable {
                        continue;
                    }
                    let tmpl = &self.arena_templates.arenas[rule_idx];
                    let (pattern, produce) = match direction {
                        Direction::Forward => (tmpl.lhs(), tmpl.rhs()),
                        Direction::Backward => (tmpl.rhs(), tmpl.lhs()),
                    };
                    let (Some(pattern), Some(produce)) = (pattern, produce) else {
                        continue;
                    };
                    let Some(bindings) = pattern_match(&self.arena, base_id, pattern.root())
                    else {
                        continue;
                    };
                    let Some(result_id) =
                        substitute_template(&mut self.arena, produce.root(), &bindings)
                    else {
                        continue;
                    };
                    let new_cost = self.arena.node(result_id).node_count();
                    let growth = new_cost.saturating_sub(original_cost);
                    if new_cost > original_cost && growth <= remaining_budget {
                        candidates.push(result_id);
                        candidate_costs.push(new_cost);
                    }
                }
            }

            if !candidates.is_empty() {
                let chosen_idx = self.rand_usize(candidates.len());
                let chosen_id = candidates[chosen_idx];
                let chosen_cost = candidate_costs[chosen_idx];
                let growth = chosen_cost.saturating_sub(original_cost);
                remaining_budget = remaining_budget.saturating_sub(growth);
                remap.insert(r, chosen_id);
                applied += 1;
            }
            // else: remap[r] already points to base_id (the remapped copy).
        }

        (remap[&root], applied)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Pattern Match + Substitute Tests
    // ========================================================================

    // ========================================================================
    // Backward Generation Tests
    // ========================================================================

    #[test]
    fn bwd_generator_produces_valid_pairs() {
        use crate::egraph::collect_rule_templates;
        let templates = collect_rule_templates();
        let config = BwdGenConfig::default();
        let mut generator = BwdGenerator::new(42, config, templates);

        for _ in 0..10 {
            let pair = generator.generate();
            let optimized_nodes = pair.optimized().root().node_count();
            let unoptimized_nodes = pair.unoptimized().root().node_count();

            // Both expressions should be valid
            assert!(optimized_nodes > 0);
            assert!(unoptimized_nodes > 0);

            // Unoptimized should generally be larger or equal
            // (junkifying increases or maintains size)
            assert!(unoptimized_nodes >= optimized_nodes);
        }
    }

    #[test]
    fn bwd_generator_has_fused_ops() {
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
            total_fused += pair
                .optimized()
                .root()
                .descendants()
                .filter(|n| n.op() == Some(OpKind::MulAdd))
                .count();
        }

        // With 80% fused op probability, we should see some fused ops
        assert!(
            total_fused > 0,
            "Expected some fused operations in generated expressions"
        );
    }

    #[test]
    fn bwd_generator_should_apply_rewrites_from_templates() {
        use crate::egraph::collect_rule_templates;
        let templates = collect_rule_templates();
        let config = BwdGenConfig::default();
        let mut generator = BwdGenerator::new(42, config, templates);

        // Generate 20 pairs and check they're non-trivial
        let mut total_rewrites = 0;
        for _ in 0..20 {
            let pair = generator.generate();
            let optimized_nodes = pair.optimized().root().node_count();
            let unoptimized_nodes = pair.unoptimized().root().node_count();
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
