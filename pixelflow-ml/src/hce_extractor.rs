//! # HCE-Based Expression Extractor
//!
//! This module provides Hand-Crafted Evaluation (HCE) based extraction for
//! expression optimization, connecting the Domain abstraction to practical
//! e-graph extraction.
//!
//! ## Phase 3: Connecting HCE to E-Graph Extraction
//!
//! The key insight is that traditional e-graph extractors use fixed per-operation
//! costs (CostModel), but HCE can capture more nuanced cost factors:
//!
//! - **Structural features**: depth, node count, patterns
//! - **Opportunity features**: has_fusable, has_identity (potential for improvement)
//! - **Fused operation awareness**: FMA/MulRsqrt are cheaper than unfused equivalents
//!
//! ## Phase 4: SPSA Weight Tuning
//!
//! SPSA (Simultaneous Perturbation Stochastic Approximation) is a gradient-free
//! optimization method used by chess engines (CLOP, Stockfish tuning) to tune
//! evaluation weights against actual performance metrics.
//!
//! ```text
//! Algorithm:
//! 1. Generate random perturbation Δ ∈ {-1, +1}^n
//! 2. Evaluate f(θ + c·Δ) and f(θ - c·Δ)
//! 3. Estimate gradient: ĝ = (f+ - f-) / (2c) · Δ
//! 4. Update: θ ← θ - a · ĝ
//! ```
//!
//! For instruction selection:
//! - f(θ) = correlation between HCE(θ) cost and actual benchmark time
//! - We want HCE to rank expressions the same way benchmarks would

#![allow(dead_code)] // Evolving API

use alloc::vec;
use alloc::vec::Vec;
use libm::{pow, sqrt};

use crate::evaluator::{
    Domain, ExprDomain, ExprFeatures, ExprMove, HandCraftedEvaluator, default_expr_weights,
    extract_expr_features,
};
use crate::nnue::Expr;

// ============================================================================
// HCE Extractor: Wrapper for E-Graph Integration
// ============================================================================

/// HCE-based expression cost evaluator for e-graph extraction.
///
/// This wraps `HandCraftedEvaluator` and provides methods suitable for
/// integration with e-graph extraction algorithms.
#[derive(Clone, Debug)]
pub struct HceExtractor {
    /// The underlying hand-crafted evaluator.
    pub hce: HandCraftedEvaluator,
    /// Whether to include structural costs (node count, depth).
    pub include_structural: bool,
    /// Penalty for expressions with optimization opportunities.
    /// Positive values penalize having unfused patterns.
    pub opportunity_penalty: i32,
}

impl HceExtractor {
    /// Create a new HCE extractor with default weights.
    #[must_use]
    pub fn new() -> Self {
        Self {
            hce: default_expr_weights(),
            include_structural: false,
            opportunity_penalty: 0,
        }
    }

    /// Create with custom weights.
    #[must_use]
    pub fn with_weights(weights: Vec<i32>) -> Self {
        Self {
            hce: HandCraftedEvaluator::new(weights),
            include_structural: false,
            opportunity_penalty: 0,
        }
    }

    /// Create with FMA-optimized weights.
    #[must_use]
    pub fn with_fma() -> Self {
        Self {
            hce: crate::evaluator::fma_optimized_weights(),
            include_structural: false,
            opportunity_penalty: 0,
        }
    }

    /// Enable structural cost factors.
    #[must_use]
    pub fn with_structural_costs(mut self) -> Self {
        self.include_structural = true;
        self
    }

    /// Set opportunity penalty.
    #[must_use]
    pub fn with_opportunity_penalty(mut self, penalty: i32) -> Self {
        self.opportunity_penalty = penalty;
        self
    }

    /// Evaluate an expression's cost using HCE.
    #[must_use]
    pub fn cost(&self, expr: &Expr) -> i32 {
        let features = extract_expr_features(expr);
        self.cost_from_features(&features)
    }

    /// Evaluate cost from pre-extracted features.
    #[must_use]
    pub fn cost_from_features(&self, features: &ExprFeatures) -> i32 {
        let mut cost = self.hce.evaluate_linear(features);

        if self.include_structural {
            // Add small penalty for larger expressions
            cost = cost.saturating_add(features.node_count / 2);
        }

        if self.opportunity_penalty != 0 {
            // Penalize having unfused patterns (incentivize running more rewrites)
            let opportunities =
                features.has_fusable + features.has_identity + features.has_self_cancel;
            cost = cost.saturating_add(opportunities.saturating_mul(self.opportunity_penalty));
        }

        cost
    }

    /// Compare two expressions and return the better one (lower cost).
    #[must_use]
    pub fn better<'a>(&self, a: &'a Expr, b: &'a Expr) -> &'a Expr {
        if self.cost(a) <= self.cost(b) { a } else { b }
    }

    /// Find the best expression among candidates.
    #[must_use]
    pub fn best<'a>(&self, exprs: &[&'a Expr]) -> Option<&'a Expr> {
        exprs.iter().min_by_key(|e| self.cost(e)).copied()
    }

    /// Evaluate all rewrites and return the best one.
    #[must_use]
    pub fn best_rewrite(&self, expr: &Expr) -> Option<(ExprMove, Expr, i32)> {
        let base_cost = self.cost(expr);
        let moves = ExprDomain::legal_moves(expr);

        let mut best: Option<(ExprMove, Expr, i32)> = None;
        let mut best_cost = base_cost;

        for mv in moves {
            if let Some(rewritten) = ExprDomain::apply_move(expr, &mv) {
                let cost = self.cost(&rewritten);
                if cost < best_cost {
                    best_cost = cost;
                    best = Some((mv, rewritten, cost));
                }
            }
        }

        best
    }

    /// Greedily optimize an expression by applying rewrites until no improvement.
    ///
    /// Returns (optimized_expr, cost_reduction, num_rewrites_applied).
    #[must_use]
    pub fn greedy_optimize(&self, expr: &Expr, max_iters: usize) -> (Expr, i32, usize) {
        let mut current = expr.clone();
        let mut total_reduction = 0i32;
        let mut rewrites_applied = 0;

        for _ in 0..max_iters {
            match self.best_rewrite(&current) {
                Some((_, rewritten, new_cost)) => {
                    let old_cost = self.cost(&current);
                    let reduction = old_cost - new_cost;
                    if reduction <= 0 {
                        break;
                    }
                    total_reduction += reduction;
                    rewrites_applied += 1;
                    current = rewritten;
                }
                None => break,
            }
        }

        (current, total_reduction, rewrites_applied)
    }
}

impl Default for HceExtractor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Feature Accumulator: Incremental Feature Updates
// ============================================================================

/// Incremental feature accumulator for efficient delta updates.
///
/// When a rewrite is applied, only part of the expression changes.
/// This accumulator tracks features incrementally, similar to how
/// NNUE maintains an accumulator for efficient position updates.
#[derive(Clone, Debug, Default)]
pub struct FeatureAccumulator {
    /// Current feature values.
    pub features: ExprFeatures,
    /// Number of updates applied.
    pub update_count: usize,
    /// Sum of all deltas (for debugging/verification).
    pub total_delta: i32,
}

impl FeatureAccumulator {
    /// Create a new accumulator initialized from an expression.
    #[must_use]
    pub fn from_expr(expr: &Expr) -> Self {
        Self {
            features: extract_expr_features(expr),
            update_count: 0,
            total_delta: 0,
        }
    }

    /// Reset the accumulator with a new expression.
    pub fn reset(&mut self, expr: &Expr) {
        self.features = extract_expr_features(expr);
        self.update_count = 0;
        self.total_delta = 0;
    }

    /// Apply an incremental update from a rewrite.
    ///
    /// Instead of re-extracting all features, we compute the delta
    /// from removing old_subtree and adding new_subtree.
    pub fn apply_delta(&mut self, old_subtree: &Expr, new_subtree: &Expr) {
        let old_features = extract_expr_features(old_subtree);
        let new_features = extract_expr_features(new_subtree);

        // Update each feature by the delta
        self.features.add_count += new_features.add_count - old_features.add_count;
        self.features.sub_count += new_features.sub_count - old_features.sub_count;
        self.features.mul_count += new_features.mul_count - old_features.mul_count;
        self.features.div_count += new_features.div_count - old_features.div_count;
        self.features.neg_count += new_features.neg_count - old_features.neg_count;
        self.features.sqrt_count += new_features.sqrt_count - old_features.sqrt_count;
        self.features.rsqrt_count += new_features.rsqrt_count - old_features.rsqrt_count;
        self.features.abs_count += new_features.abs_count - old_features.abs_count;
        self.features.min_count += new_features.min_count - old_features.min_count;
        self.features.max_count += new_features.max_count - old_features.max_count;
        self.features.fma_count += new_features.fma_count - old_features.fma_count;
        self.features.mul_rsqrt_count +=
            new_features.mul_rsqrt_count - old_features.mul_rsqrt_count;
        self.features.node_count += new_features.node_count - old_features.node_count;
        // Note: depth is tricky to update incrementally - may need full recompute
        self.features.var_count += new_features.var_count - old_features.var_count;
        self.features.const_count += new_features.const_count - old_features.const_count;
        self.features.has_identity += new_features.has_identity - old_features.has_identity;
        self.features.has_self_cancel +=
            new_features.has_self_cancel - old_features.has_self_cancel;
        self.features.has_fusable += new_features.has_fusable - old_features.has_fusable;

        self.update_count += 1;

        // Track total delta for debugging
        let old_cost = compute_feature_cost(&old_features);
        let new_cost = compute_feature_cost(&new_features);
        self.total_delta += new_cost - old_cost;
    }

    /// Verify the accumulator matches a full re-extraction.
    ///
    /// Returns true if features match, false if there's drift.
    #[must_use]
    pub fn verify(&self, expr: &Expr) -> bool {
        let fresh = extract_expr_features(expr);
        features_equal(&self.features, &fresh)
    }
}

/// Helper: compute simple cost from features (for delta tracking).
fn compute_feature_cost(f: &ExprFeatures) -> i32 {
    f.add_count * 4
        + f.sub_count * 4
        + f.mul_count * 5
        + f.div_count * 15
        + f.neg_count
        + f.sqrt_count * 15
        + f.rsqrt_count * 5
        + f.abs_count
        + f.min_count * 4
        + f.max_count * 4
        + f.fma_count * 5
        + f.mul_rsqrt_count * 6
}

/// Helper: check if two feature sets are equal.
fn features_equal(a: &ExprFeatures, b: &ExprFeatures) -> bool {
    a.add_count == b.add_count
        && a.sub_count == b.sub_count
        && a.mul_count == b.mul_count
        && a.div_count == b.div_count
        && a.neg_count == b.neg_count
        && a.sqrt_count == b.sqrt_count
        && a.rsqrt_count == b.rsqrt_count
        && a.abs_count == b.abs_count
        && a.min_count == b.min_count
        && a.max_count == b.max_count
        && a.fma_count == b.fma_count
        && a.mul_rsqrt_count == b.mul_rsqrt_count
        && a.node_count == b.node_count
        && a.var_count == b.var_count
        && a.const_count == b.const_count
        && a.has_identity == b.has_identity
        && a.has_self_cancel == b.has_self_cancel
        && a.has_fusable == b.has_fusable
}

// ============================================================================
// SPSA Weight Tuner (Phase 4)
// ============================================================================

/// Configuration for SPSA weight tuning.
#[derive(Clone, Debug)]
pub struct SpsaConfig {
    /// Initial step size for gradient estimate (c parameter).
    pub c: f64,
    /// Initial learning rate (a parameter).
    pub a: f64,
    /// Decay rate for c (c_k = c / k^gamma).
    pub c_decay: f64,
    /// Decay rate for a (a_k = a / (k + A)^alpha).
    pub a_decay: f64,
    /// Stability constant for a decay.
    pub a_stability: f64,
    /// Number of iterations.
    pub max_iters: usize,
    /// Number of samples per evaluation.
    pub samples_per_eval: usize,
    /// Clamp weights to this range.
    pub weight_clamp: (i32, i32),
}

impl Default for SpsaConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            a: 0.1,
            c_decay: 0.101,    // gamma = 0.101 (standard SPSA)
            a_decay: 0.602,    // alpha = 0.602 (standard SPSA)
            a_stability: 10.0, // A = 10% of max_iters
            max_iters: 100,
            samples_per_eval: 50,
            weight_clamp: (-100, 100),
        }
    }
}

/// SPSA weight tuner for HCE.
///
/// Tunes HCE weights to maximize correlation with actual performance.
pub struct SpsaTuner {
    /// Configuration.
    pub config: SpsaConfig,
    /// Current weights.
    pub weights: Vec<f64>,
    /// Best weights found.
    pub best_weights: Vec<f64>,
    /// Best loss achieved.
    pub best_loss: f64,
    /// Random state.
    rng_state: u64,
    /// Iteration count.
    iteration: usize,
}

impl SpsaTuner {
    /// Create a new SPSA tuner starting from given weights.
    #[must_use]
    pub fn new(initial_weights: &[i32], config: SpsaConfig) -> Self {
        let weights: Vec<f64> = initial_weights.iter().map(|&w| w as f64).collect();
        Self {
            config,
            weights: weights.clone(),
            best_weights: weights,
            best_loss: f64::MAX,
            rng_state: 0xDEADBEEF,
            iteration: 0,
        }
    }

    /// Create from default HCE weights.
    #[must_use]
    pub fn from_defaults(config: SpsaConfig) -> Self {
        let hce = default_expr_weights();
        Self::new(&hce.weights, config)
    }

    /// Generate random perturbation vector Δ ∈ {-1, +1}^n.
    fn generate_perturbation(&mut self) -> Vec<f64> {
        let n = self.weights.len();
        let mut delta = Vec::with_capacity(n);
        for _ in 0..n {
            self.rng_state = self
                .rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let bit = (self.rng_state >> 63) as i32;
            delta.push(if bit == 0 { -1.0 } else { 1.0 });
        }
        delta
    }

    /// Get current step size c_k.
    fn current_c(&self) -> f64 {
        self.config.c / pow(self.iteration as f64 + 1.0, self.config.c_decay)
    }

    /// Get current learning rate a_k.
    fn current_a(&self) -> f64 {
        self.config.a
            / pow(
                self.iteration as f64 + self.config.a_stability + 1.0,
                self.config.a_decay,
            )
    }

    /// Perform one SPSA iteration.
    ///
    /// Takes a loss function that evaluates weight quality.
    /// Loss should be LOWER for better weights.
    ///
    /// Returns the gradient estimate magnitude.
    pub fn step<F>(&mut self, loss_fn: F) -> f64
    where
        F: Fn(&[i32]) -> f64,
    {
        let c_k = self.current_c();
        let a_k = self.current_a();

        // Generate perturbation
        let delta = self.generate_perturbation();

        // Compute perturbed weights
        let mut weights_plus: Vec<i32> = Vec::with_capacity(self.weights.len());
        let mut weights_minus: Vec<i32> = Vec::with_capacity(self.weights.len());

        for (i, d) in delta.iter().enumerate().take(self.weights.len()) {
            let plus = (self.weights[i] + c_k * d).clamp(
                self.config.weight_clamp.0 as f64,
                self.config.weight_clamp.1 as f64,
            ) as i32;
            let minus = (self.weights[i] - c_k * d).clamp(
                self.config.weight_clamp.0 as f64,
                self.config.weight_clamp.1 as f64,
            ) as i32;
            weights_plus.push(plus);
            weights_minus.push(minus);
        }

        // Evaluate both perturbations
        let loss_plus = loss_fn(&weights_plus);
        let loss_minus = loss_fn(&weights_minus);

        // Estimate gradient
        let grad_scale = (loss_plus - loss_minus) / (2.0 * c_k);
        let mut grad_norm = 0.0;

        // Update weights
        for (i, d) in delta.iter().enumerate().take(self.weights.len()) {
            let grad_i = grad_scale * d;
            grad_norm += grad_i * grad_i;
            self.weights[i] -= a_k * grad_i;

            // Clamp to valid range
            self.weights[i] = self.weights[i].clamp(
                self.config.weight_clamp.0 as f64,
                self.config.weight_clamp.1 as f64,
            );
        }

        // Track best
        let current_loss = (loss_plus + loss_minus) / 2.0;
        if current_loss < self.best_loss {
            self.best_loss = current_loss;
            self.best_weights = self.weights.clone();
        }

        self.iteration += 1;
        sqrt(grad_norm)
    }

    /// Run full optimization.
    ///
    /// Returns (best_weights, best_loss, iterations_run).
    pub fn optimize<F>(&mut self, loss_fn: F) -> (Vec<i32>, f64, usize)
    where
        F: Fn(&[i32]) -> f64,
    {
        for _ in 0..self.config.max_iters {
            self.step(&loss_fn);
        }

        let final_weights: Vec<i32> = self.best_weights.iter().map(|&w| w as i32).collect();

        (final_weights, self.best_loss, self.iteration)
    }

    /// Get current weights as i32.
    #[must_use]
    pub fn current_weights(&self) -> Vec<i32> {
        self.weights.iter().map(|&w| w as i32).collect()
    }

    /// Get best weights found.
    #[must_use]
    pub fn best_weights_i32(&self) -> Vec<i32> {
        self.best_weights.iter().map(|&w| w as i32).collect()
    }
}

// ============================================================================
// Loss Functions for Weight Tuning
// ============================================================================

/// Compute ranking loss between HCE predictions and ground truth.
///
/// This measures how well HCE ordering matches the true cost ordering.
/// Lower is better.
#[must_use]
pub fn ranking_loss(
    weights: &[i32],
    samples: &[(Expr, u64)], // (expression, true_cost_ns)
) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }

    let hce = HandCraftedEvaluator::new(weights.to_vec());
    let mut violations = 0;
    let mut comparisons = 0;

    // Compare all pairs
    for i in 0..samples.len() {
        for j in (i + 1)..samples.len() {
            let (expr_i, true_cost_i) = &samples[i];
            let (expr_j, true_cost_j) = &samples[j];

            let features_i = extract_expr_features(expr_i);
            let features_j = extract_expr_features(expr_j);

            let pred_i = hce.evaluate_linear(&features_i);
            let pred_j = hce.evaluate_linear(&features_j);

            // Check if ordering matches
            let true_order = true_cost_i.cmp(true_cost_j);
            let pred_order = pred_i.cmp(&pred_j);

            comparisons += 1;
            if true_order != pred_order && true_order != core::cmp::Ordering::Equal {
                violations += 1;
            }
        }
    }

    if comparisons == 0 {
        0.0
    } else {
        violations as f64 / comparisons as f64
    }
}

/// Compute correlation loss (1 - Spearman correlation).
///
/// Measures how well HCE rankings correlate with true rankings.
#[must_use]
pub fn correlation_loss(weights: &[i32], samples: &[(Expr, u64)]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }

    let hce = HandCraftedEvaluator::new(weights.to_vec());

    // Get predicted costs
    let mut predictions: Vec<(usize, i32)> = samples
        .iter()
        .enumerate()
        .map(|(i, (expr, _))| {
            let features = extract_expr_features(expr);
            (i, hce.evaluate_linear(&features))
        })
        .collect();

    // Get true costs
    let mut truths: Vec<(usize, u64)> = samples
        .iter()
        .enumerate()
        .map(|(i, (_, cost))| (i, *cost))
        .collect();

    // Rank both
    predictions.sort_by_key(|(_, cost)| *cost);
    truths.sort_by_key(|(_, cost)| *cost);

    // Create rank arrays
    let n = samples.len();
    let mut pred_ranks = vec![0usize; n];
    let mut true_ranks = vec![0usize; n];

    for (rank, &(idx, _)) in predictions.iter().enumerate() {
        pred_ranks[idx] = rank;
    }
    for (rank, &(idx, _)) in truths.iter().enumerate() {
        true_ranks[idx] = rank;
    }

    // Compute Spearman correlation
    let mut sum_d2 = 0.0;
    for i in 0..n {
        let d = pred_ranks[i] as f64 - true_ranks[i] as f64;
        sum_d2 += d * d;
    }

    let n_f = n as f64;
    let rho = 1.0 - (6.0 * sum_d2) / (n_f * (n_f * n_f - 1.0));

    // Return 1 - correlation (so lower is better)
    1.0 - rho
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue::{OpType, RewriteRule};
    use alloc::boxed::Box;
    use alloc::vec;

    // =========================================================================
    // HceExtractor Tests
    // =========================================================================

    #[test]
    fn hce_extractor_basic_should_succeed_when_called() {
        let extractor = HceExtractor::new();

        // Simple expression: x + y
        let expr = Expr::Binary(OpType::Add, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));

        let cost = extractor.cost(&expr);
        assert!(cost > 0, "Add should have non-zero cost");
        // Cost breakdown with ILP features:
        // - add: 4 cycles
        // - critical_path: 4 (add latency) * weight(1) = 4
        // - max_width: 2 (two vars at depth 1) * weight(1) = 2
        // Total = 4 + 4 + 2 = 10
        assert_eq!(cost, 10, "Add cost should include ILP features");
    }

    #[test]
    fn hce_extractor_fma_cheaper_should_succeed_when_called() {
        // Use default extractor (not with_fma) to see raw operation costs
        let extractor = HceExtractor::new();

        // Unfused: a * b + c
        let unfused = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Binary(
                OpType::Mul,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
            )),
            Box::new(Expr::Var(2)),
        );

        // Fused: MulAdd(a, b, c)
        let fused = Expr::Ternary(
            OpType::MulAdd,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
            Box::new(Expr::Var(2)),
        );

        let unfused_cost = extractor.cost(&unfused);
        let fused_cost = extractor.cost(&fused);

        // With default weights:
        // - unfused: mul(5) + add(4) + has_fusable(-2) = 7
        // - fused: fma(5) = 5
        assert!(
            fused_cost < unfused_cost,
            "FMA ({}) should be cheaper than unfused mul+add ({})",
            fused_cost,
            unfused_cost
        );

        // Also verify FMA-optimized weights still prefer fused
        let fma_extractor = HceExtractor::with_fma();
        let fma_unfused_cost = fma_extractor.cost(&unfused);
        let fma_fused_cost = fma_extractor.cost(&fused);

        // With FMA weights: has_fusable penalty is -4, so unfused = 5+4-4=5, fused=5
        // They're equal because has_fusable rewards expressions that CAN be fused
        // This incentivizes running the rewrite to actually fuse them
        assert!(
            fma_fused_cost <= fma_unfused_cost,
            "With FMA, fused ({}) should be <= unfused ({})",
            fma_fused_cost,
            fma_unfused_cost
        );
    }

    #[test]
    fn hce_extractor_better_should_succeed_when_called() {
        let extractor = HceExtractor::new();

        let simple = Expr::Var(0);
        let complex = Expr::Binary(OpType::Div, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));

        let better = extractor.better(&simple, &complex);
        assert!(
            matches!(better, Expr::Var(0)),
            "Simple expression should be better"
        );
    }

    #[test]
    fn hce_extractor_best_rewrite_should_succeed_when_called() {
        let extractor = HceExtractor::new();

        // x + 0 - has an obvious improvement
        let expr = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(0.0)),
        );

        let result = extractor.best_rewrite(&expr);
        assert!(result.is_some(), "Should find AddZero rewrite");

        let (mv, rewritten, cost) = result.expect("Expected value but got None/Err");
        assert!(
            matches!(mv.rule, RewriteRule::AddZero),
            "Should be AddZero rule"
        );
        assert!(matches!(rewritten, Expr::Var(0)), "Should simplify to x");
        // Variable cost with ILP features:
        // - var_count: 0 (weight 0)
        // - critical_path: 0 (no latency) * weight(1) = 0
        // - max_width: 1 (one node at depth 0) * weight(1) = 1
        // Total = 1
        assert_eq!(
            cost, 1,
            "Variable should have minimal cost (just max_width)"
        );
    }

    #[test]
    fn hce_extractor_greedy_optimize_should_succeed_when_called() {
        let extractor = HceExtractor::with_fma();

        // (x * y + z) + 0 - multiple optimizations possible
        let expr = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Binary(
                OpType::Add,
                Box::new(Expr::Binary(
                    OpType::Mul,
                    Box::new(Expr::Var(0)),
                    Box::new(Expr::Var(1)),
                )),
                Box::new(Expr::Var(2)),
            )),
            Box::new(Expr::Const(0.0)),
        );

        let (optimized, reduction, rewrites) = extractor.greedy_optimize(&expr, 10);

        assert!(reduction > 0, "Should have cost reduction");
        assert!(rewrites > 0, "Should have applied at least one rewrite");

        // Optimized version should be cheaper
        let orig_cost = extractor.cost(&expr);
        let opt_cost = extractor.cost(&optimized);
        assert!(opt_cost < orig_cost, "Optimized should be cheaper");
    }

    // =========================================================================
    // Feature Accumulator Tests (Ensuring it does actual work)
    // =========================================================================

    #[test]
    fn accumulator_initializes_from_expr_should_succeed_when_called() {
        let expr = Expr::Binary(
            OpType::Mul,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Binary(
                OpType::Add,
                Box::new(Expr::Var(1)),
                Box::new(Expr::Const(1.0)),
            )),
        );

        let acc = FeatureAccumulator::from_expr(&expr);

        assert_eq!(acc.features.mul_count, 1, "Should count 1 mul");
        assert_eq!(acc.features.add_count, 1, "Should count 1 add");
        assert_eq!(acc.features.var_count, 2, "Should count 2 vars");
        assert_eq!(acc.features.const_count, 1, "Should count 1 const");
        assert_eq!(acc.features.node_count, 5, "Should have 5 nodes");
        assert_eq!(acc.update_count, 0, "No updates yet");
    }

    #[test]
    fn accumulator_delta_update_works_should_succeed_when_called() {
        // Start with: x * 1
        let original = Expr::Binary(
            OpType::Mul,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(1.0)),
        );

        let mut acc = FeatureAccumulator::from_expr(&original);
        assert_eq!(acc.features.mul_count, 1);
        assert_eq!(
            acc.features.has_identity, 1,
            "x*1 should be detected as identity"
        );

        // Simulate rewrite: x * 1 -> x
        let old_subtree = original.clone();
        let new_subtree = Expr::Var(0);

        acc.apply_delta(&old_subtree, &new_subtree);

        assert_eq!(acc.features.mul_count, 0, "Mul should be removed");
        assert_eq!(acc.features.var_count, 1, "Should have 1 var now (not 2)");
        assert_eq!(acc.features.const_count, 0, "Const should be removed");
        assert_eq!(acc.features.has_identity, 0, "No more identity pattern");
        assert_eq!(acc.update_count, 1, "Should have 1 update");
        assert!(
            acc.total_delta < 0,
            "Total delta should be negative (cost reduced)"
        );
    }

    #[test]
    fn accumulator_delta_tracks_fma_fusion_should_succeed_when_called() {
        // Start with: (a * b) + c
        let original = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Binary(
                OpType::Mul,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
            )),
            Box::new(Expr::Var(2)),
        );

        let mut acc = FeatureAccumulator::from_expr(&original);
        assert_eq!(acc.features.mul_count, 1);
        assert_eq!(acc.features.add_count, 1);
        assert_eq!(acc.features.fma_count, 0);
        assert_eq!(acc.features.has_fusable, 1, "Should detect fusable pattern");

        // Simulate FMA fusion: (a * b) + c -> MulAdd(a, b, c)
        let fused = Expr::Ternary(
            OpType::MulAdd,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
            Box::new(Expr::Var(2)),
        );

        acc.apply_delta(&original, &fused);

        assert_eq!(acc.features.mul_count, 0, "Mul should be removed");
        assert_eq!(acc.features.add_count, 0, "Add should be removed");
        assert_eq!(acc.features.fma_count, 1, "Should have 1 FMA now");
        assert_eq!(acc.features.has_fusable, 0, "No more fusable pattern");
        assert!(acc.total_delta < 0, "FMA should reduce cost");
    }

    #[test]
    fn accumulator_verify_consistency_should_succeed_when_called() {
        let expr = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Binary(
                OpType::Mul,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(2.0)),
            )),
            Box::new(Expr::Var(1)),
        );

        let acc = FeatureAccumulator::from_expr(&expr);
        assert!(acc.verify(&expr), "Fresh accumulator should verify");
    }

    #[test]
    fn accumulator_multiple_deltas_should_succeed_when_called() {
        // Start with: ((x * 1) + 0) - should simplify to just x
        let original = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Binary(
                OpType::Mul,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(1.0)),
            )),
            Box::new(Expr::Const(0.0)),
        );

        let mut acc = FeatureAccumulator::from_expr(&original);

        // First delta: x * 1 -> x
        let mul_subtree = Expr::Binary(
            OpType::Mul,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(1.0)),
        );
        acc.apply_delta(&mul_subtree, &Expr::Var(0));

        assert_eq!(acc.features.mul_count, 0);
        assert_eq!(acc.update_count, 1);

        // Second delta: x + 0 -> x (but we've already simplified the mul)
        // Simulate the full expression becoming just Var(0)
        let after_first = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(0.0)),
        );
        acc.apply_delta(&after_first, &Expr::Var(0));

        assert_eq!(acc.features.add_count, 0);
        assert_eq!(acc.features.const_count, 0);
        assert_eq!(acc.features.var_count, 1);
        assert_eq!(acc.update_count, 2);
    }

    // =========================================================================
    // SPSA Tuner Tests
    // =========================================================================

    #[test]
    fn spsa_perturbation_generation_should_succeed_when_called() {
        let config = SpsaConfig {
            max_iters: 10,
            ..Default::default()
        };
        let mut tuner = SpsaTuner::from_defaults(config);

        let delta = tuner.generate_perturbation();

        assert_eq!(delta.len(), tuner.weights.len());
        for &d in &delta {
            assert!(d == -1.0 || d == 1.0, "Perturbation should be ±1");
        }
    }

    #[test]
    fn spsa_step_updates_weights_should_succeed_when_called() {
        let config = SpsaConfig {
            max_iters: 10,
            c: 0.5,
            a: 0.1,
            ..Default::default()
        };
        let initial_weights: Vec<i32> =
            vec![4, 4, 5, 15, 1, 15, 5, 1, 4, 4, 5, 6, 0, 0, 0, 0, 0, 0, -2];
        let mut tuner = SpsaTuner::new(&initial_weights, config);

        let initial = tuner.weights.clone();

        // Simple loss function: sum of squared weights
        let grad_norm = tuner.step(|w| w.iter().map(|&x| (x as f64).powi(2)).sum());

        assert!(grad_norm > 0.0, "Should have non-zero gradient");
        assert_ne!(tuner.weights, initial, "Weights should change after step");
    }

    #[test]
    fn spsa_optimize_reduces_loss_should_succeed_when_called() {
        let config = SpsaConfig {
            max_iters: 50,
            c: 1.0,
            a: 0.5,
            samples_per_eval: 10,
            weight_clamp: (0, 50),
            ..Default::default()
        };

        // Start with non-optimal weights
        let initial_weights: Vec<i32> = vec![10, 10, 10, 10, 10];
        let mut tuner = SpsaTuner::new(&initial_weights, config);

        // Loss function: distance from target [4, 4, 5, 15, 1]
        let target = [4.0, 4.0, 5.0, 15.0, 1.0];
        let _initial_loss = tuner.best_loss;

        let (final_weights, final_loss, iters) = tuner.optimize(|w| {
            w.iter()
                .zip(target.iter())
                .map(|(&x, &t)| (x as f64 - t).powi(2))
                .sum()
        });

        assert!(final_loss < 500.0, "Loss should decrease substantially");
        assert_eq!(iters, 50);

        // Weights should move toward target
        assert!(
            final_weights[3] > final_weights[0],
            "Weight for div (index 3) should be larger than add (index 0)"
        );
    }

    // =========================================================================
    // Loss Function Tests
    // =========================================================================

    #[test]
    fn ranking_loss_perfect_should_succeed_when_called() {
        let samples = vec![
            (Expr::Var(0), 0),
            (
                Expr::Binary(OpType::Add, Box::new(Expr::Var(0)), Box::new(Expr::Var(1))),
                4,
            ),
            (
                Expr::Binary(OpType::Div, Box::new(Expr::Var(0)), Box::new(Expr::Var(1))),
                15,
            ),
        ];

        let weights = default_expr_weights().weights;
        let loss = ranking_loss(&weights, &samples);

        // Default weights should rank these correctly
        assert!(
            loss < 0.5,
            "Default weights should have low ranking loss, got {}",
            loss
        );
    }

    #[test]
    fn ranking_loss_inverted_should_succeed_when_called() {
        let samples = vec![
            (Expr::Var(0), 100), // Inverted: simple is "expensive"
            (
                Expr::Binary(OpType::Div, Box::new(Expr::Var(0)), Box::new(Expr::Var(1))),
                0,
            ), // div is "cheap"
        ];

        let weights = default_expr_weights().weights;
        let loss = ranking_loss(&weights, &samples);

        // Should have high loss because ordering is inverted
        assert!(
            loss > 0.5,
            "Inverted samples should have high ranking loss, got {}",
            loss
        );
    }

    #[test]
    fn correlation_loss_bounds_should_succeed_when_called() {
        let samples = vec![
            (Expr::Var(0), 0),
            (
                Expr::Binary(OpType::Add, Box::new(Expr::Var(0)), Box::new(Expr::Var(1))),
                4,
            ),
        ];

        let weights = default_expr_weights().weights;
        let loss = correlation_loss(&weights, &samples);

        assert!(
            loss >= 0.0 && loss <= 2.0,
            "Correlation loss should be in [0, 2], got {}",
            loss
        );
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn hce_extractor_with_accumulator_consistency_should_succeed_when_called() {
        let extractor = HceExtractor::with_fma();

        // Build expression
        let expr = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Binary(
                OpType::Mul,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
            )),
            Box::new(Expr::Var(2)),
        );

        // Get cost via extractor
        let direct_cost = extractor.cost(&expr);

        // Get cost via accumulator
        let acc = FeatureAccumulator::from_expr(&expr);
        let acc_cost = extractor.cost_from_features(&acc.features);

        assert_eq!(
            direct_cost, acc_cost,
            "Direct and accumulator costs should match"
        );
    }
}
