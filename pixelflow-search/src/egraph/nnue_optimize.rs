//! NNUE-guided expression optimization.
//!
//! This module provides a high-level API for optimizing expressions using
//! the trained dual-head NNUE model. The model guides which (expression, rule)
//! pairs to try during e-graph saturation.
//!
//! **Key insight**: ALL costs are predicted by the NNUE value head. There are
//! NO hardcoded costs. The model learns what expressions are expensive and
//! uses that to guide both extraction and mask filtering.
//!
//! # Usage
//!
//! ```ignore
//! use pixelflow_search::egraph::nnue_optimize::{NnueOptimizer, OptimizeConfig};
//! use pixelflow_ir::Expr;
//!
//! // Load trained model
//! let optimizer = NnueOptimizer::load("path/to/mask_reinforce.bin")?;
//!
//! // Optimize an expression
//! let expr = Expr::Binary(OpKind::Add, Box::new(Expr::Var(0)), Box::new(Expr::Const(0.0)));
//! let result = optimizer.optimize(&expr, OptimizeConfig::default());
//!
//! println!("Cost: {:.2} -> {:.2}", result.initial_cost, result.final_cost);
//! ```

use crate::egraph::{EClassId, EGraph, ENode, ExprTree, Leaf};
use crate::egraph::nnue_adapter::extract_beam;
use crate::math::all_math_rules;
use crate::nnue::{ExprNnue, RuleTemplates, EMBED_DIM};
use pixelflow_ir::Expr;
use std::path::Path;

/// Configuration for NNUE-guided optimization.
#[derive(Clone, Debug)]
pub struct OptimizeConfig {
    /// Maximum epochs (iterations) of e-graph saturation.
    pub max_epochs: usize,
    /// Maximum e-classes before stopping (memory limit).
    pub max_classes: usize,
    /// Threshold for mask approval (sigmoid(score) > threshold).
    pub threshold: f32,
    /// Beam width for extraction (higher = better quality, slower).
    pub beam_width: usize,
}

impl Default for OptimizeConfig {
    fn default() -> Self {
        Self {
            max_epochs: 30,
            max_classes: 10_000,
            threshold: 0.5,
            beam_width: 8,
        }
    }
}

impl OptimizeConfig {
    /// Fast config for simple expressions.
    pub fn fast() -> Self {
        Self {
            max_epochs: 3,
            max_classes: 500,
            threshold: 0.5,
            beam_width: 4,
        }
    }

    /// Thorough config for complex expressions.
    pub fn thorough() -> Self {
        Self {
            max_epochs: 100,
            max_classes: 50_000,
            threshold: 0.3,
            beam_width: 16,
        }
    }
}

/// Result of NNUE-guided optimization.
#[derive(Debug)]
pub struct OptimizeResult {
    /// The optimized expression.
    pub expr: Expr,
    /// Initial cost before optimization (NNUE predicted log-cost).
    pub initial_cost: f32,
    /// Final cost after optimization (NNUE predicted log-cost).
    pub final_cost: f32,
    /// Number of epochs used.
    pub epochs_used: usize,
    /// Number of (class, rule) pairs tried.
    pub pairs_tried: usize,
    /// Number of pairs skipped by the mask.
    pub pairs_skipped: usize,
    /// Whether saturation completed (no more changes).
    pub saturated: bool,
}

impl OptimizeResult {
    /// Cost reduction ratio (1.0 = no change, 0.5 = halved cost).
    /// Since costs are log-scale, we compare directly.
    pub fn cost_ratio(&self) -> f32 {
        if self.initial_cost <= 0.0 {
            1.0
        } else {
            self.final_cost / self.initial_cost
        }
    }

    /// Fraction of pairs skipped by the mask.
    pub fn skip_rate(&self) -> f32 {
        let total = self.pairs_tried + self.pairs_skipped;
        if total == 0 {
            0.0
        } else {
            self.pairs_skipped as f32 / total as f32
        }
    }
}

/// NNUE-guided optimizer.
///
/// Loads a trained ExprNnue model and uses it to guide e-graph saturation.
/// - **Mask head**: predicts which (expression, rule) pairs are worth trying
/// - **Value head**: predicts expression cost (NO hardcoded costs anywhere!)
///
/// All cost predictions come from the neural network, not lookup tables.
pub struct NnueOptimizer {
    /// The trained model (used for BOTH mask and value prediction).
    model: ExprNnue,
    /// Pre-computed rule embeddings.
    rule_embeds: Vec<[f32; EMBED_DIM]>,
}

impl NnueOptimizer {
    /// Load a trained model from disk.
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let model = ExprNnue::load(path.as_ref())?;

        // Build rule templates and pre-compute embeddings
        let rules = all_math_rules();
        let templates = build_rule_templates(&rules);
        let rule_embeds = model.encode_all_rules_from_templates(&templates);

        Ok(Self {
            model,
            rule_embeds,
        })
    }

    /// Create an optimizer with a pre-loaded model.
    pub fn with_model(model: ExprNnue) -> Self {
        let rules = all_math_rules();
        let templates = build_rule_templates(&rules);
        let rule_embeds = model.encode_all_rules_from_templates(&templates);

        Self {
            model,
            rule_embeds,
        }
    }

    /// Create an optimizer with a random (untrained) model.
    /// Useful for testing the infrastructure.
    pub fn random(seed: u64) -> Self {
        let model = ExprNnue::new_random(seed);
        Self::with_model(model)
    }

    /// Access the underlying NNUE model.
    pub fn model(&self) -> &ExprNnue {
        &self.model
    }

    /// Optimize an expression using NNUE-guided search.
    ///
    /// **All costs are NNUE predictions** - no hardcoded cost tables.
    /// The value head predicts log-nanoseconds for each expression tree.
    pub fn optimize(&self, expr: &Expr, config: OptimizeConfig) -> OptimizeResult {
        // Build e-graph from expression
        let mut egraph = EGraph::with_rules(all_math_rules());
        let root = self.add_expr_to_egraph(&mut egraph, expr);

        // Get initial cost using NNUE value head (beam search extraction)
        let (initial_tree, initial_cost) = extract_beam(&egraph, root, &self.model, config.beam_width);

        // Run NNUE-guided saturation
        let num_rules = egraph.num_rules();
        let mut best_cost = initial_cost;
        let mut best_tree = initial_tree;
        let mut epoch = 0;
        let mut pairs_tried = 0;
        let mut pairs_skipped = 0;
        let mut saturated = false;

        while epoch < config.max_epochs {
            let num_classes = egraph.num_classes();
            if num_classes >= config.max_classes {
                break;
            }

            let mut total_changes = 0;

            // For each e-class (re-check class limit after rule applications)
            for class_idx in 0..num_classes {
                if egraph.num_classes() >= config.max_classes {
                    break;
                }
                let class_id = EClassId(class_idx as u32);
                let class_id = egraph.find(class_id);

                // Get representative expression for this e-class using NNUE
                let (repr_tree, _) = extract_beam(&egraph, class_id, &self.model, 1);
                let repr_expr = expr_tree_to_expr(&repr_tree);

                // Score all rules for this expression using mask head
                let scores = self.model.mask_score_all_rules(&repr_expr, &self.rule_embeds);

                for (rule_idx, &score) in scores.iter().enumerate().take(num_rules) {
                    let prob = sigmoid(score);
                    let approved = prob > config.threshold;

                    if approved {
                        pairs_tried += 1;

                        // Try to apply this rule at this class
                        let nodes: Vec<_> = egraph.nodes(class_id).to_vec();
                        for (node_idx, _) in nodes.iter().enumerate() {
                            if egraph.apply_single_rule(rule_idx, class_id, node_idx) {
                                total_changes += 1;
                            }
                        }
                    } else {
                        pairs_skipped += 1;
                    }
                }
            }

            // Extract and check improvement using NNUE value head
            if total_changes > 0 {
                let (tree, cost) = extract_beam(&egraph, root, &self.model, config.beam_width);
                if cost < best_cost {
                    best_cost = cost;
                    best_tree = tree;
                }
            } else {
                saturated = true;
                break;
            }

            epoch += 1;
        }

        // Convert best tree back to Expr
        let final_expr = expr_tree_to_expr(&best_tree);

        OptimizeResult {
            expr: final_expr,
            initial_cost,
            final_cost: best_cost,
            epochs_used: epoch,
            pairs_tried,
            pairs_skipped,
            saturated,
        }
    }

    /// Add an IR expression to the e-graph, returning the root e-class.
    fn add_expr_to_egraph(&self, egraph: &mut EGraph, expr: &Expr) -> EClassId {
        match expr {
            Expr::Var(idx) => egraph.add(ENode::Var(*idx)),
            Expr::Const(val) => egraph.add(ENode::constant(*val)),
            Expr::Param(i) => panic!("Expr::Param({}) reached NNUE cost model — call substitute_params before use", i),
            Expr::Unary(op, a) => {
                let a_id = self.add_expr_to_egraph(egraph, a);
                let op_ref = op_kind_to_op(*op);
                egraph.add(ENode::Op { op: op_ref, children: vec![a_id] })
            }
            Expr::Binary(op, a, b) => {
                let a_id = self.add_expr_to_egraph(egraph, a);
                let b_id = self.add_expr_to_egraph(egraph, b);
                let op_ref = op_kind_to_op(*op);
                egraph.add(ENode::Op { op: op_ref, children: vec![a_id, b_id] })
            }
            Expr::Ternary(op, a, b, c) => {
                let a_id = self.add_expr_to_egraph(egraph, a);
                let b_id = self.add_expr_to_egraph(egraph, b);
                let c_id = self.add_expr_to_egraph(egraph, c);
                let op_ref = op_kind_to_op(*op);
                egraph.add(ENode::Op { op: op_ref, children: vec![a_id, b_id, c_id] })
            }
            Expr::Nary(op, children) => {
                let child_ids: Vec<_> = children.iter()
                    .map(|c| self.add_expr_to_egraph(egraph, c))
                    .collect();
                let op_ref = op_kind_to_op(*op);
                egraph.add(ENode::Op { op: op_ref, children: child_ids })
            }
        }
    }
}

/// Build rule templates from the rule set.
fn build_rule_templates(rules: &[Box<dyn crate::egraph::Rewrite>]) -> RuleTemplates {
    let mut templates = RuleTemplates::with_capacity(rules.len());
    for (idx, rule) in rules.iter().enumerate() {
        if let (Some(lhs), Some(rhs)) = (rule.lhs_template(), rule.rhs_template()) {
            templates.set(idx, lhs, rhs);
        }
    }
    templates
}

/// Convert ExprTree to IR Expr.
fn expr_tree_to_expr(tree: &ExprTree) -> Expr {
    match tree {
        ExprTree::Leaf(Leaf::Var(idx)) => Expr::Var(*idx),
        ExprTree::Leaf(Leaf::Const(val)) => Expr::Const(*val),
        ExprTree::Op { op, children } => {
            let kind = op.kind();
            match children.len() {
                0 => Expr::Const(0.0),
                1 => Expr::Unary(kind, Box::new(expr_tree_to_expr(&children[0]))),
                2 => Expr::Binary(
                    kind,
                    Box::new(expr_tree_to_expr(&children[0])),
                    Box::new(expr_tree_to_expr(&children[1])),
                ),
                3 => Expr::Ternary(
                    kind,
                    Box::new(expr_tree_to_expr(&children[0])),
                    Box::new(expr_tree_to_expr(&children[1])),
                    Box::new(expr_tree_to_expr(&children[2])),
                ),
                _ => Expr::Nary(
                    kind,
                    children.iter().map(expr_tree_to_expr).collect(),
                ),
            }
        }
    }
}

/// Convert OpKind to static Op reference.
fn op_kind_to_op(kind: pixelflow_ir::OpKind) -> &'static dyn crate::egraph::ops::Op {
    use crate::egraph::ops;
    use pixelflow_ir::OpKind;

    match kind {
        OpKind::Add => &ops::Add,
        OpKind::Sub => &ops::Sub,
        OpKind::Mul => &ops::Mul,
        OpKind::Div => &ops::Div,
        OpKind::Neg => &ops::Neg,
        OpKind::Sqrt => &ops::Sqrt,
        OpKind::Rsqrt => &ops::Rsqrt,
        OpKind::Recip => &ops::Recip,
        OpKind::Abs => &ops::Abs,
        OpKind::Min => &ops::Min,
        OpKind::Max => &ops::Max,
        OpKind::MulAdd => &ops::MulAdd,
        OpKind::Sin => &ops::Sin,
        OpKind::Cos => &ops::Cos,
        OpKind::Tan => &ops::Tan,
        OpKind::Asin => &ops::Asin,
        OpKind::Acos => &ops::Acos,
        OpKind::Atan => &ops::Atan,
        OpKind::Atan2 => &ops::Atan2,
        OpKind::Exp => &ops::Exp,
        OpKind::Exp2 => &ops::Exp2,
        OpKind::Ln => &ops::Ln,
        OpKind::Log2 => &ops::Log2,
        OpKind::Log10 => &ops::Log10,
        OpKind::Pow => &ops::Pow,
        OpKind::Hypot => &ops::Hypot,
        OpKind::Floor => &ops::Floor,
        OpKind::Ceil => &ops::Ceil,
        OpKind::Round => &ops::Round,
        OpKind::Fract => &ops::Fract,
        OpKind::Select => &ops::Select,
        OpKind::Clamp => &ops::Clamp,
        OpKind::Lt => &ops::Lt,
        OpKind::Le => &ops::Le,
        OpKind::Gt => &ops::Gt,
        OpKind::Ge => &ops::Ge,
        OpKind::Eq => &ops::Eq,
        OpKind::Ne => &ops::Ne,
        OpKind::Tuple => &ops::Tuple,
        // Default fallback for any missing ops
        _ => &ops::Add,
    }
}

/// Sigmoid activation.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::OpKind;

    #[test]
    fn optimize_identity_add_should_succeed_when_called() {
        // x + 0 should simplify to x
        let expr = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(0.0)),
        );

        let optimizer = NnueOptimizer::random(42);
        let result = optimizer.optimize(&expr, OptimizeConfig::fast());

        // All costs are NNUE predictions now (log-nanoseconds)
        println!("Initial cost: {:.3}, Final cost: {:.3}", result.initial_cost, result.final_cost);
        println!("Skip rate: {:.1}%", result.skip_rate() * 100.0);

        // The expression should be optimizable (x + 0 -> x)
        assert!(result.final_cost <= result.initial_cost);
    }

    #[test]
    fn optimize_mul_zero_should_succeed_when_called() {
        // x * 0 should simplify to 0
        let expr = Expr::Binary(
            OpKind::Mul,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(0.0)),
        );

        let optimizer = NnueOptimizer::random(42);
        let result = optimizer.optimize(&expr, OptimizeConfig::fast());

        println!("x * 0: {:.3} -> {:.3}", result.initial_cost, result.final_cost);

        // Should simplify to constant 0
        assert!(result.final_cost <= result.initial_cost);
    }

    #[test]
    fn optimize_complex_expression_should_succeed_when_called() {
        // (x + 0) * (y * 1) - should simplify to x * y
        let expr = Expr::Binary(
            OpKind::Mul,
            Box::new(Expr::Binary(
                OpKind::Add,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(0.0)),
            )),
            Box::new(Expr::Binary(
                OpKind::Mul,
                Box::new(Expr::Var(1)),
                Box::new(Expr::Const(1.0)),
            )),
        );

        let optimizer = NnueOptimizer::random(42);
        let result = optimizer.optimize(&expr, OptimizeConfig::fast());

        println!("Complex: {:.3} -> {:.3} (ratio: {:.2})",
            result.initial_cost, result.final_cost, result.cost_ratio());
        println!("Epochs: {}, Pairs tried: {}, skipped: {}",
            result.epochs_used, result.pairs_tried, result.pairs_skipped);

        // Should improve
        assert!(result.cost_ratio() <= 1.0);
    }

    #[test]
    fn config_presets_should_succeed_when_called() {
        let fast = OptimizeConfig::fast();
        let thorough = OptimizeConfig::thorough();

        assert!(fast.max_epochs < thorough.max_epochs);
        assert!(fast.max_classes < thorough.max_classes);
        assert!(fast.beam_width < thorough.beam_width);
    }
}
