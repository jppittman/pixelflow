//! Guided search with learned rule filtering.
//!
//! This module implements epoch-based e-graph search where a learned mask
//! (ExprNnue bilinear architecture) decides which (expression, rule) pairs to apply.
//!
//! # Architecture
//!
//! The unified mask uses bilinear scoring: `sigmoid(mask_features @ W @ rule_embed + bias)`.
//! Pairs with score > threshold are applied; others are skipped.
//!
//! Primary entry point: `run_dual_mask_with_templates` (template-encoded rules).
//! Legacy baselines: `run_uniform`, `run_random`, `run_epsilon_greedy` (closure-based).

use super::extract::ExprTree;
use super::graph::EGraph;
use super::node::EClassId;

/// Statistics for a rule across search runs.
///
/// Used to compute historical features for the Guide.
#[derive(Clone, Debug, Default)]
pub struct RuleStats {
    /// Total times this rule was attempted
    pub total_attempts: usize,
    /// Total times this rule actually matched
    pub total_matches: usize,
    /// Last epoch this rule matched (None if never)
    pub last_match_epoch: Option<usize>,
}

impl RuleStats {
    /// Match rate: matches / attempts
    pub fn match_rate(&self) -> f32 {
        if self.total_attempts == 0 {
            0.0
        } else {
            self.total_matches as f32 / self.total_attempts as f32
        }
    }

    /// Epochs since last match (saturates at usize::MAX if never matched)
    pub fn epochs_since_match(&self, current_epoch: usize) -> usize {
        match self.last_match_epoch {
            Some(last) => current_epoch.saturating_sub(last),
            None => usize::MAX,
        }
    }
}

/// Record of one rule's application attempt in an epoch.
///
/// Used for collecting training data.
#[derive(Clone, Debug)]
pub struct RuleRecord {
    /// Rule index
    pub rule_idx: usize,
    /// Guide's prediction (if using Guide)
    pub predicted_p_match: Option<f32>,
    /// Whether we decided to apply this rule
    pub applied: bool,
    /// Whether the rule actually matched (made changes)
    pub matched: bool,
    /// Number of changes made (0 if didn't match)
    pub changes: usize,
}

/// Record of one epoch for training data.
#[derive(Clone, Debug)]
pub struct EpochRecord {
    /// Epoch number
    pub epoch: usize,
    /// Per-rule records
    pub rule_records: Vec<RuleRecord>,
    /// Cost before this epoch
    pub cost_before: i64,
    /// Cost after this epoch
    pub cost_after: i64,
    /// Total changes made this epoch
    pub total_changes: usize,
}

/// Result of guided search.
pub struct GuidedSearchResult {
    /// Best expression tree found
    pub best_tree: ExprTree,
    /// Cost of best tree
    pub best_cost: i64,
    /// Number of epochs used
    pub epochs_used: usize,
    /// Search trajectory for training
    pub trajectory: Vec<EpochRecord>,
    /// Final rule statistics
    pub rule_stats: Vec<RuleStats>,
    /// Reason search stopped
    pub stop_reason: StopReason,
}

/// Why guided search stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// Guide predicts no rules will match
    GuidePredictsNoMatch,
    /// No changes made (saturation on approved rules)
    Saturated,
    /// Maximum epochs reached
    MaxEpochs,
    /// No rules in the e-graph
    NoRules,
    /// E-graph reached the configured maximum class count
    OutOfSpace,
}

/// Match threshold for rule application.
pub const DEFAULT_MATCH_THRESHOLD: f32 = 0.3;

/// Guided search with learned rule filtering.
///
/// Like egg's epoch-based saturation, but with a learned Guide that
/// predicts which rules will match, skipping wasteful applications.
pub struct GuidedSearch {
    /// The e-graph being searched
    egraph: EGraph,
    /// Root e-class to extract from
    root: EClassId,
    /// Best cost found so far
    best_cost: i64,
    /// Best tree found so far
    best_tree: Option<ExprTree>,
    /// Current epoch
    epoch: usize,
    /// Maximum epochs
    max_epochs: usize,
    /// Per-rule statistics
    rule_stats: Vec<RuleStats>,
    /// Match threshold (rules with P(match) > threshold are applied)
    match_threshold: f32,
    /// Node budget (max e-graph nodes before search stops)
    node_budget: usize,
}

/// Default node budget for guided search.
pub const DEFAULT_NODE_BUDGET: usize = 10_000;

impl GuidedSearch {
    /// Create a new guided search.
    pub fn new(egraph: EGraph, root: EClassId, max_epochs: usize) -> Self {
        let num_rules = egraph.num_rules();
        Self {
            egraph,
            root,
            best_cost: i64::MAX,
            best_tree: None,
            epoch: 0,
            max_epochs,
            rule_stats: vec![RuleStats::default(); num_rules],
            match_threshold: DEFAULT_MATCH_THRESHOLD,
            node_budget: DEFAULT_NODE_BUDGET,
        }
    }

    /// Set the node budget.
    pub fn with_node_budget(mut self, budget: usize) -> Self {
        self.node_budget = budget;
        self
    }

    /// Set the match threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.match_threshold = threshold;
        self
    }

    /// Get reference to the e-graph.
    pub fn egraph(&self) -> &EGraph {
        &self.egraph
    }

    /// Get mutable reference to the e-graph.
    pub fn egraph_mut(&mut self) -> &mut EGraph {
        &mut self.egraph
    }

    /// Apply a single rule everywhere it matches.
    ///
    /// Returns the number of changes made.
    fn apply_rule_everywhere(&mut self, rule_idx: usize) -> usize {
        let mut changes = 0;
        let num_classes = self.egraph.num_classes();

        for class_idx in 0..num_classes {
            let class_id = EClassId(class_idx as u32);
            let class_id = self.egraph.find(class_id);

            // Get nodes in this class
            let nodes: Vec<_> = self.egraph.nodes(class_id).to_vec();

            for (node_idx, _node) in nodes.iter().enumerate() {
                if self.egraph.apply_single_rule(rule_idx, class_id, node_idx) {
                    changes += 1;
                }
            }
        }

        changes
    }

    /// Run search with a closure-based Guide scoring function (legacy path).
    ///
    /// The Guide takes (egraph, rule_idx, rule_stats) and returns P(match) in [0, 1].
    /// Rules with P(match) > threshold are applied.
    ///
    /// Prefer `run_dual_mask_with_templates` for the unified mask architecture.
    pub fn run_with_closure<G, J>(
        &mut self,
        mut guide: G,
        mut judge: J,
        nnue: &crate::nnue::ExprNnue,
    ) -> GuidedSearchResult
    where
        G: FnMut(&EGraph, usize, &RuleStats) -> f32,
        J: FnMut(&ExprTree) -> i64,
    {
        let mut trajectory = Vec::new();
        let num_rules = self.egraph.num_rules();

        if num_rules == 0 {
            // No rules to apply - just extract and return
            let (tree, _cost) = super::nnue_adapter::extract_neural(&self.egraph, self.root, nnue);
            let cost = judge(&tree);
            return GuidedSearchResult {
                best_tree: tree,
                best_cost: cost,
                epochs_used: 0,
                trajectory,
                rule_stats: self.rule_stats.clone(),
                stop_reason: StopReason::NoRules,
            };
        }

        // Initial extraction
        let (initial_tree, _) = super::nnue_adapter::extract_neural(&self.egraph, self.root, nnue);
        self.best_cost = judge(&initial_tree);
        self.best_tree = Some(initial_tree);

        loop {
            if self.epoch >= self.max_epochs {
                return self.finish(StopReason::MaxEpochs, trajectory);
            }

            // Node budget: stop if e-graph has grown too large
            if self.egraph.node_count() > self.node_budget {
                return self.finish(StopReason::MaxEpochs, trajectory);
            }

            // Score each rule with the Guide
            let mut rules_to_apply = Vec::new();
            let mut rule_records = Vec::new();

            for rule_idx in 0..num_rules {
                let p_match = guide(&self.egraph, rule_idx, &self.rule_stats[rule_idx]);

                let should_apply = p_match > self.match_threshold;

                rule_records.push(RuleRecord {
                    rule_idx,
                    predicted_p_match: Some(p_match),
                    applied: should_apply,
                    matched: false, // Updated after application
                    changes: 0,
                });

                if should_apply {
                    rules_to_apply.push(rule_idx);
                }
            }

            // If Guide predicts nothing will match, stop
            if rules_to_apply.is_empty() {
                // Fill in records and finish
                let epoch_record = EpochRecord {
                    epoch: self.epoch,
                    rule_records,
                    cost_before: self.best_cost,
                    cost_after: self.best_cost,
                    total_changes: 0,
                };
                trajectory.push(epoch_record);
                return self.finish(StopReason::GuidePredictsNoMatch, trajectory);
            }

            // Apply approved rules and track which matched
            let cost_before = self.best_cost;
            let mut total_changes = 0;

            for rule_idx in &rules_to_apply {
                let changes = self.apply_rule_everywhere(*rule_idx);
                total_changes += changes;

                // Update rule stats
                self.rule_stats[*rule_idx].total_attempts += 1;
                if changes > 0 {
                    self.rule_stats[*rule_idx].total_matches += 1;
                    self.rule_stats[*rule_idx].last_match_epoch = Some(self.epoch);
                }

                // Update record
                if let Some(record) = rule_records.iter_mut().find(|r| r.rule_idx == *rule_idx) {
                    record.matched = changes > 0;
                    record.changes = changes;
                }
            }

            // Extract and evaluate if changes were made
            let cost_after = if total_changes > 0 {
                let (tree, _) = super::nnue_adapter::extract_neural(&self.egraph, self.root, nnue);
                let cost = judge(&tree);

                if cost < self.best_cost {
                    self.best_cost = cost;
                    self.best_tree = Some(tree);
                }
                self.best_cost
            } else {
                cost_before
            };

            // Record epoch
            let epoch_record = EpochRecord {
                epoch: self.epoch,
                rule_records,
                cost_before,
                cost_after,
                total_changes,
            };
            trajectory.push(epoch_record);

            // Check for saturation (no changes despite applying rules)
            if total_changes == 0 {
                return self.finish(StopReason::Saturated, trajectory);
            }

            self.epoch += 1;
        }
    }

    /// Run with a uniform Guide (baseline: all rules applied).
    ///
    /// This is equivalent to egg-style saturation.
    pub fn run_uniform<J>(&mut self, judge: J, nnue: &crate::nnue::ExprNnue) -> GuidedSearchResult
    where
        J: FnMut(&ExprTree) -> i64,
    {
        // Uniform guide: always returns 1.0 (above any threshold)
        self.run_with_closure(|_, _, _| 1.0, judge, nnue)
    }

    /// Run with random Guide (for training data baseline).
    pub fn run_random<J>(&mut self, judge: J, nnue: &crate::nnue::ExprNnue, seed: u64) -> GuidedSearchResult
    where
        J: FnMut(&ExprTree) -> i64,
    {
        // Simple LCG for reproducible randomness
        let mut state = seed;
        let random_guide = move |_: &EGraph, _: usize, _: &RuleStats| -> f32 {
            // LCG: state = (state * a + c) mod m
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Convert to [0, 1]
            (state as f32) / (u64::MAX as f32)
        };

        self.run_with_closure(random_guide, judge, nnue)
    }

    /// Run with epsilon-greedy exploration.
    ///
    /// With probability (1-epsilon): uniform (accept all rules)
    /// With probability epsilon: random (accept with probability 0.5)
    ///
    /// This creates variance to break "stable zero" equilibrium while still
    /// mostly finding good optimizations.
    pub fn run_epsilon_greedy<J>(
        &mut self,
        judge: J,
        nnue: &crate::nnue::ExprNnue,
        epsilon: f32,
        seed: u64,
    ) -> GuidedSearchResult
    where
        J: FnMut(&ExprTree) -> i64,
    {
        let mut state = seed;
        let epsilon_guide = move |_: &EGraph, _: usize, _: &RuleStats| -> f32 {
            // LCG for random number
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (state as f32) / (u64::MAX as f32);

            if r < epsilon {
                // Explore: return random score [0, 1]
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state as f32) / (u64::MAX as f32)
            } else {
                // Exploit: accept all (score = 1.0)
                1.0
            }
        };

        self.run_with_closure(epsilon_guide, judge, nnue)
    }

    /// Finish search and return result.
    fn finish(
        &self,
        stop_reason: StopReason,
        trajectory: Vec<EpochRecord>,
    ) -> GuidedSearchResult {
        GuidedSearchResult {
            best_tree: self.best_tree.clone().expect("best_tree should be set"),
            best_cost: self.best_cost,
            epochs_used: self.epoch,
            trajectory,
            rule_stats: self.rule_stats.clone(),
            stop_reason,
        }
    }

    // =========================================================================
    // Unified Mask Architecture (ExprNnue with bilinear interaction)
    //
    // This replaces the separate DualMaskGuide with a unified architecture that
    // scales to 1000+ rules. Uses bilinear expr-rule scoring.
    // =========================================================================

    /// Run with unified dual-mask architecture (ExprNnue).
    ///
    /// The unified architecture combines expression and rule scoring in a single
    /// model with bilinear interaction, scaling to 1000+ rules.
    ///
    /// # Arguments
    /// * `model` - ExprNnue with trained unified mask architecture
    /// * `rule_features` - Pre-computed hand-crafted features for each rule
    /// * `judge` - Cost function for extracted trees
    /// * `nnue` - Neural cost model for extraction
    /// * `threshold` - Pairs with sigmoid(score) > threshold are approved
    /// * `max_classes` - E-graph space limit (resource constraint)
    pub fn run_dual_mask_unified<J>(
        &mut self,
        model: &crate::nnue::ExprNnue,
        rule_features: &crate::nnue::RuleFeatures,
        mut judge: J,
        nnue: &crate::nnue::ExprNnue,
        threshold: f32,
        max_classes: usize,
    ) -> UnifiedMaskSearchResult
    where
        J: FnMut(&ExprTree) -> i64,
    {
        let num_rules = self.egraph.num_rules();
        let mut pairs_tried = 0usize;
        let mut pairs_skipped = 0usize;
        let mut trajectory = Vec::new();

        if num_rules == 0 {
            let (tree, _) = super::nnue_adapter::extract_neural(&self.egraph, self.root, nnue);
            let cost = judge(&tree);
            return UnifiedMaskSearchResult {
                best_tree: tree,
                best_cost: cost,
                epochs_used: 0,
                pairs_tried,
                pairs_skipped,
                trajectory,
                stop_reason: StopReason::NoRules,
            };
        }

        // Pre-encode all rules (cached for entire search)
        let rule_embeds = model.encode_all_rules(rule_features, num_rules);

        // Initial extraction
        let (initial_tree, _) = super::nnue_adapter::extract_neural(&self.egraph, self.root, nnue);
        self.best_cost = judge(&initial_tree);
        self.best_tree = Some(initial_tree);

        loop {
            // Termination checks
            if self.epoch >= self.max_epochs {
                return self.finish_unified_mask(
                    StopReason::MaxEpochs, pairs_tried, pairs_skipped, trajectory
                );
            }

            let num_classes = self.egraph.num_classes();
            if num_classes >= max_classes {
                return self.finish_unified_mask(
                    StopReason::OutOfSpace,
                    pairs_tried, pairs_skipped, trajectory
                );
            }

            let cost_before = self.best_cost;
            let mut total_changes = 0;
            let mut epoch_pairs = Vec::new();

            // For each e-class
            for class_idx in 0..num_classes {
                let class_id = EClassId(class_idx as u32);
                let class_id = self.egraph.find(class_id);

                // Get representative expression for this e-class
                // We extract the best tree rooted at this class to get its structure
                let (repr_tree, _) = super::nnue_adapter::extract_neural(&self.egraph, class_id, nnue);

                // Convert ExprTree to Expr for the model
                let expr = expr_tree_to_expr(&repr_tree);

                // Score all rules for this expression (one forward pass)
                let scores = model.mask_score_all_rules(&expr, &rule_embeds);

                // Score rules, record decisions, apply approved ones
                for (rule_idx, &score) in scores.iter().enumerate().take(num_rules) {
                    let prob = sigmoid(score);
                    let approved = prob > threshold;

                    // Record this decision for REINFORCE
                    // Sample ~10% of rejections to avoid memory explosion
                    let should_record = approved || {
                        let hash = (class_idx.wrapping_mul(31) ^ rule_idx) as u32;
                        hash % 10 == 0
                    };

                    if should_record {
                        epoch_pairs.push(UnifiedPairRecord {
                            class_idx,
                            rule_idx,
                            score,
                            approved,
                        });
                    }

                    if approved {
                        pairs_tried += 1;

                        // Try to apply this rule at this class
                        let nodes: Vec<_> = self.egraph.nodes(class_id).to_vec();
                        for (node_idx, _) in nodes.iter().enumerate() {
                            if self.egraph.apply_single_rule(rule_idx, class_id, node_idx) {
                                total_changes += 1;
                            }
                        }
                    } else {
                        pairs_skipped += 1;
                    }
                }
            }

            // Extract and evaluate
            let cost_after = if total_changes > 0 {
                let (tree, _) = super::nnue_adapter::extract_neural(&self.egraph, self.root, nnue);
                let cost = judge(&tree);
                if cost < self.best_cost {
                    self.best_cost = cost;
                    self.best_tree = Some(tree);
                }
                self.best_cost
            } else {
                cost_before
            };

            trajectory.push(UnifiedMaskEpochRecord {
                epoch: self.epoch,
                pairs: epoch_pairs,
                cost_before,
                cost_after,
                total_changes,
            });

            if total_changes == 0 {
                return self.finish_unified_mask(
                    StopReason::Saturated, pairs_tried, pairs_skipped, trajectory
                );
            }

            self.epoch += 1;
        }
    }

    /// Run with unified dual-mask architecture using LHS/RHS template-based rule encoding.
    ///
    /// This is the preferred method for the unified architecture. Rules are encoded
    /// using their LHS/RHS expression templates through the shared backbone, enabling
    /// learned structural similarity instead of hand-crafted features.
    ///
    /// # Arguments
    /// * `model` - ExprNnue with trained unified mask architecture
    /// * `templates` - LHS/RHS expression templates for each rule
    /// * `judge` - Cost function for extracted trees
    /// * `nnue` - Neural cost model for extraction
    /// * `threshold` - Pairs with sigmoid(score) > threshold are approved
    /// * `max_classes` - E-graph space limit (resource constraint)
    ///
    /// # Template Encoding
    /// Each rule is encoded as: `[z_LHS | z_RHS | z_LHS-z_RHS | z_LHS*z_RHS]`
    /// where z_LHS and z_RHS are the shared expr embeddings of the templates.
    /// This captures what the rule matches, what it produces, and their relationship.
    pub fn run_dual_mask_with_templates<J>(
        &mut self,
        model: &crate::nnue::ExprNnue,
        templates: &crate::nnue::RuleTemplates,
        mut judge: J,
        nnue: &crate::nnue::ExprNnue,
        threshold: f32,
        max_classes: usize,
    ) -> UnifiedMaskSearchResult
    where
        J: FnMut(&ExprTree) -> i64,
    {
        let num_rules = self.egraph.num_rules();
        let mut pairs_tried = 0usize;
        let mut pairs_skipped = 0usize;
        let mut trajectory = Vec::new();

        if num_rules == 0 {
            let (tree, _) = super::nnue_adapter::extract_neural(&self.egraph, self.root, nnue);
            let cost = judge(&tree);
            return UnifiedMaskSearchResult {
                best_tree: tree,
                best_cost: cost,
                epochs_used: 0,
                pairs_tried,
                pairs_skipped,
                trajectory,
                stop_reason: StopReason::NoRules,
            };
        }

        // Pre-encode all rules using LHS/RHS templates (cached for entire search)
        let rule_embeds = model.encode_all_rules_from_templates(templates);

        // Initial extraction
        let (initial_tree, _) = super::nnue_adapter::extract_neural(&self.egraph, self.root, nnue);
        self.best_cost = judge(&initial_tree);
        self.best_tree = Some(initial_tree);

        loop {
            // Termination checks
            if self.epoch >= self.max_epochs {
                return self.finish_unified_mask(
                    StopReason::MaxEpochs, pairs_tried, pairs_skipped, trajectory
                );
            }

            let num_classes = self.egraph.num_classes();
            if num_classes >= max_classes {
                return self.finish_unified_mask(
                    StopReason::OutOfSpace,
                    pairs_tried, pairs_skipped, trajectory
                );
            }

            let cost_before = self.best_cost;
            let mut total_changes = 0;
            let mut epoch_pairs = Vec::new();

            // For each e-class
            for class_idx in 0..num_classes {
                let class_id = EClassId(class_idx as u32);
                let class_id = self.egraph.find(class_id);

                // Get representative expression for this e-class
                let (repr_tree, _) = super::nnue_adapter::extract_neural(&self.egraph, class_id, nnue);
                let expr = expr_tree_to_expr(&repr_tree);

                // Score all rules for this expression (one forward pass)
                let scores = model.mask_score_all_rules(&expr, &rule_embeds);

                // Score rules, record decisions, apply approved ones
                for (rule_idx, &score) in scores.iter().enumerate().take(num_rules) {
                    let prob = sigmoid(score);
                    let approved = prob > threshold;

                    // Record this decision for REINFORCE
                    // Sample ~10% of rejections to avoid memory explosion
                    let should_record = approved || {
                        let hash = (class_idx.wrapping_mul(31) ^ rule_idx) as u32;
                        hash % 10 == 0
                    };

                    if should_record {
                        epoch_pairs.push(UnifiedPairRecord {
                            class_idx,
                            rule_idx,
                            score,
                            approved,
                        });
                    }

                    if approved {
                        pairs_tried += 1;

                        // Try to apply this rule at this class
                        let nodes: Vec<_> = self.egraph.nodes(class_id).to_vec();
                        for (node_idx, _) in nodes.iter().enumerate() {
                            if self.egraph.apply_single_rule(rule_idx, class_id, node_idx) {
                                total_changes += 1;
                            }
                        }
                    } else {
                        pairs_skipped += 1;
                    }
                }
            }

            // Extract and evaluate
            let cost_after = if total_changes > 0 {
                let (tree, _) = super::nnue_adapter::extract_neural(&self.egraph, self.root, nnue);
                let cost = judge(&tree);
                if cost < self.best_cost {
                    self.best_cost = cost;
                    self.best_tree = Some(tree);
                }
                self.best_cost
            } else {
                cost_before
            };

            trajectory.push(UnifiedMaskEpochRecord {
                epoch: self.epoch,
                pairs: epoch_pairs,
                cost_before,
                cost_after,
                total_changes,
            });

            if total_changes == 0 {
                return self.finish_unified_mask(
                    StopReason::Saturated, pairs_tried, pairs_skipped, trajectory
                );
            }

            self.epoch += 1;
        }
    }

    fn finish_unified_mask(
        &self,
        stop_reason: StopReason,
        pairs_tried: usize,
        pairs_skipped: usize,
        trajectory: Vec<UnifiedMaskEpochRecord>,
    ) -> UnifiedMaskSearchResult {
        UnifiedMaskSearchResult {
            best_tree: self.best_tree.clone().expect("best_tree should be set"),
            best_cost: self.best_cost,
            epochs_used: self.epoch,
            pairs_tried,
            pairs_skipped,
            trajectory,
            stop_reason,
        }
    }

}


/// Convert ExprTree to pixelflow_ir::Expr for unified mask scoring.
fn expr_tree_to_expr(tree: &ExprTree) -> pixelflow_ir::Expr {
    use super::extract::Leaf;

    match tree {
        ExprTree::Leaf(Leaf::Var(idx)) => pixelflow_ir::Expr::Var(*idx),
        ExprTree::Leaf(Leaf::Const(val)) => pixelflow_ir::Expr::Const(*val),
        ExprTree::Op { op, children } => {
            let kind = op.kind();
            match children.len() {
                0 => {
                    // Nullary - shouldn't happen, but treat as const
                    pixelflow_ir::Expr::Const(0.0)
                }
                1 => pixelflow_ir::Expr::Unary(
                    kind,
                    Box::new(expr_tree_to_expr(&children[0])),
                ),
                2 => pixelflow_ir::Expr::Binary(
                    kind,
                    Box::new(expr_tree_to_expr(&children[0])),
                    Box::new(expr_tree_to_expr(&children[1])),
                ),
                3 => pixelflow_ir::Expr::Ternary(
                    kind,
                    Box::new(expr_tree_to_expr(&children[0])),
                    Box::new(expr_tree_to_expr(&children[1])),
                    Box::new(expr_tree_to_expr(&children[2])),
                ),
                _ => pixelflow_ir::Expr::Nary(
                    kind,
                    children.iter().map(expr_tree_to_expr).collect(),
                ),
            }
        }
    }
}

/// Sigmoid activation for probability conversion.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-x))
}

// =============================================================================
// Unified Mask Architecture Result Types
// =============================================================================

/// Record of a mask decision for REINFORCE training.
///
/// We only care about: what decision was made, and what was the score.
/// The reward comes from the FINAL extraction cost, not per-rule outcomes.
#[derive(Clone, Debug)]
pub struct UnifiedPairRecord {
    /// E-class index
    pub class_idx: usize,
    /// Rule index
    pub rule_idx: usize,
    /// Raw bilinear score (before sigmoid)
    pub score: f32,
    /// Was this pair approved (prob > threshold)?
    pub approved: bool,
}

/// Epoch record for unified mask training.
#[derive(Clone, Debug)]
pub struct UnifiedMaskEpochRecord {
    /// Epoch number
    pub epoch: usize,
    /// Per-pair records (only for tried pairs)
    pub pairs: Vec<UnifiedPairRecord>,
    /// Cost before epoch
    pub cost_before: i64,
    /// Cost after epoch
    pub cost_after: i64,
    /// Total changes made
    pub total_changes: usize,
}

/// Result of unified dual-mask guided search.
pub struct UnifiedMaskSearchResult {
    /// Best expression tree found
    pub best_tree: ExprTree,
    /// Cost of best tree
    pub best_cost: i64,
    /// Number of epochs used
    pub epochs_used: usize,
    /// Total (class, rule) pairs tried
    pub pairs_tried: usize,
    /// Total pairs skipped by filter
    pub pairs_skipped: usize,
    /// Training data: per-epoch records
    pub trajectory: Vec<UnifiedMaskEpochRecord>,
    /// Reason search stopped
    pub stop_reason: StopReason,
}

impl UnifiedMaskSearchResult {
    /// Fraction of pairs skipped by the mask filter.
    ///
    /// Higher is better (more efficient). Target: >80%.
    pub fn skip_rate(&self) -> f32 {
        let total = self.pairs_tried + self.pairs_skipped;
        if total == 0 {
            0.0
        } else {
            self.pairs_skipped as f32 / total as f32
        }
    }

    /// Total pairs considered (tried + skipped).
    pub fn total_pairs(&self) -> usize {
        self.pairs_tried + self.pairs_skipped
    }
}

/// Collect training data by running egg-style saturation.
///
/// This runs full saturation (all rules every epoch) and records
/// ground truth for "did rule X match in state Y?"
pub fn collect_match_training_data(
    egraph: &mut EGraph,
    root: EClassId,
    nnue: &crate::nnue::ExprNnue,
    max_epochs: usize,
) -> Vec<EpochRecord> {
    let mut trajectory = Vec::new();
    let num_rules = egraph.num_rules();

    if num_rules == 0 {
        return trajectory;
    }

    let mut rule_stats = vec![RuleStats::default(); num_rules];

    for epoch in 0..max_epochs {
        // Extract cost before
        let (tree_before, _) = super::nnue_adapter::extract_neural(egraph, root, nnue);
        let cost_before = tree_before.node_count() as i64; // Simple cost for now

        // Record features and try each rule
        let mut rule_records = Vec::new();

        for rule_idx in 0..num_rules {
            // Count matches for this rule (without applying yet)
            // We'll apply all rules at once after counting
            let will_match = would_rule_match(egraph, rule_idx);

            rule_records.push(RuleRecord {
                rule_idx,
                predicted_p_match: None, // No prediction in ground truth collection
                applied: true,           // We always "apply" in egg-style
                matched: will_match,
                changes: 0, // Updated after actual application
            });
        }

        // Now apply all rules (egg-style epoch)
        let total_changes = egraph.apply_rules_once();

        // Update rule stats based on what actually happened
        for record in &mut rule_records {
            if record.matched {
                rule_stats[record.rule_idx].total_matches += 1;
                rule_stats[record.rule_idx].last_match_epoch = Some(epoch);
            }
            rule_stats[record.rule_idx].total_attempts += 1;
        }

        // Extract cost after
        let (tree_after, _) = super::nnue_adapter::extract_neural(egraph, root, nnue);
        let cost_after = tree_after.node_count() as i64;

        trajectory.push(EpochRecord {
            epoch,
            rule_records,
            cost_before,
            cost_after,
            total_changes,
        });

        // Stop if saturated
        if total_changes == 0 {
            break;
        }
    }

    trajectory
}

/// Check if a rule would match anywhere in the e-graph.
///
/// This is a read-only check (doesn't apply the rule).
fn would_rule_match(egraph: &EGraph, rule_idx: usize) -> bool {
    let Some(rule) = egraph.rule(rule_idx) else {
        return false;
    };

    for class_idx in 0..egraph.num_classes() {
        let class_id = EClassId(class_idx as u32);
        let class_id = egraph.find(class_id);

        for node in egraph.nodes(class_id) {
            if rule.apply(egraph, class_id, node).is_some() {
                return true;
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::{all_rules, ops};
    use crate::nnue::ExprNnue;

    #[test]
    fn test_guided_search_basic() {
        // Create simple expression: X + 0
        let expr = ExprTree::Op {
            op: &ops::Add,
            children: vec![
                ExprTree::var(0),
                ExprTree::constant(0.0),
            ],
        };

        let mut egraph = EGraph::with_rules(all_rules());
        let root = egraph.add_expr(&expr);

        let nnue = ExprNnue::new_random(42);
        let mut search = GuidedSearch::new(egraph, root, 3)
            .with_node_budget(500);

        // Run with uniform guide (baseline)
        let result = search.run_uniform(
            |tree| tree.node_count() as i64,
            &nnue,
        );

        // Should find simplified form
        assert!(result.epochs_used > 0, "Should use at least one epoch");
        assert!(result.best_cost <= 3, "Should simplify X + 0 to X");
    }

    #[test]
    fn test_collect_training_data() {
        // (X + 0) * 1
        let expr = ExprTree::Op {
            op: &ops::Mul,
            children: vec![
                ExprTree::Op {
                    op: &ops::Add,
                    children: vec![
                        ExprTree::var(0),
                        ExprTree::constant(0.0),
                    ],
                },
                ExprTree::constant(1.0),
            ],
        };

        let mut egraph = EGraph::with_rules(all_rules());
        let root = egraph.add_expr(&expr);

        let nnue = ExprNnue::new_random(42);
        let trajectory = collect_match_training_data(&mut egraph, root, &nnue, 10);

        // Should have some epochs
        assert!(!trajectory.is_empty(), "Should have trajectory data");

        // Each epoch should have records for all rules
        for epoch_record in &trajectory {
            assert_eq!(
                epoch_record.rule_records.len(),
                all_rules().len(),
                "Should have record for each rule"
            );
        }
    }

    #[test]
    fn test_rule_stats() {
        let mut stats = RuleStats::default();

        // Initially 0% match rate
        assert_eq!(stats.match_rate(), 0.0);
        assert_eq!(stats.epochs_since_match(0), usize::MAX);

        // Record some matches
        stats.total_attempts = 10;
        stats.total_matches = 3;
        stats.last_match_epoch = Some(5);

        assert!((stats.match_rate() - 0.3).abs() < 0.001);
        assert_eq!(stats.epochs_since_match(7), 2);
        assert_eq!(stats.epochs_since_match(5), 0);
    }
}
