//! # NNUE Trainer for Expression Cost Prediction
//!
//! Trains NNUE weights from benchmark data using gradient descent.
//!
//! ## Training Pipeline
//!
//! 1. Collect (Expr features, runtime_ns) pairs from benchmarks
//! 2. Convert to training samples with sparse HalfEP features
//! 3. Train via gradient descent to minimize MSE
//! 4. Export weights for use in e-graph cost function

extern crate alloc;

use crate::nnue::{Expr, HalfEPFeature, extract_features};
use alloc::vec::Vec;
use libm::logf;

// ============================================================================
// Training Sample
// ============================================================================

/// A training sample with features and target cost.
#[derive(Clone, Debug)]
pub struct NnueSample {
    /// Sparse feature indices (sorted, deduplicated).
    pub feature_indices: Vec<usize>,
    /// Target cost (in arbitrary units, usually nanoseconds).
    pub target_cost: f32,
    /// Log-transformed target for training.
    pub log_target: f32,
}

impl NnueSample {
    /// Create from an expression and measured cost.
    pub fn from_expr(expr: &Expr, cost_ns: f32) -> Self {
        let features = extract_features(expr);
        let mut indices: Vec<usize> = features.iter().map(|f| f.to_index()).collect();
        indices.sort_unstable();
        indices.dedup();

        let log_target = logf(cost_ns.max(0.01));

        Self {
            feature_indices: indices,
            target_cost: cost_ns,
            log_target,
        }
    }

    /// Create from pre-computed feature indices.
    pub fn from_features(feature_indices: Vec<usize>, cost_ns: f32) -> Self {
        let mut indices = feature_indices;
        indices.sort_unstable();
        indices.dedup();

        let log_target = logf(cost_ns.max(0.01));

        Self {
            feature_indices: indices,
            target_cost: cost_ns,
            log_target,
        }
    }
}

// ============================================================================
// Training Configuration
// ============================================================================

/// Configuration for NNUE training.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    /// Learning rate for gradient descent.
    pub learning_rate: f32,
    /// Number of training epochs.
    pub epochs: usize,
    /// Mini-batch size.
    pub batch_size: usize,
    /// L2 regularization strength.
    pub l2_lambda: f32,
    /// Whether to use log-transformed targets.
    pub use_log_transform: bool,
    /// Print progress every N epochs.
    pub print_every: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            epochs: 100,
            batch_size: 32,
            l2_lambda: 0.001,
            use_log_transform: true,
            print_every: 10,
        }
    }
}

// ============================================================================
// Simple Linear NNUE (actually learns!)
// ============================================================================

/// A simple linear model on sparse features.
///
/// This is the core of NNUE's first layer: a linear combination of
/// feature weights. For small datasets, this is more effective than
/// the full deep network.
pub struct LinearNnue {
    /// Weights for each HalfEP feature.
    pub weights: Vec<f32>,
    /// Bias term.
    pub bias: f32,
}

impl LinearNnue {
    /// Create a new linear NNUE with zero weights.
    pub fn new() -> Self {
        Self {
            weights: alloc::vec![0.0; HalfEPFeature::COUNT],
            bias: 0.0,
        }
    }

    /// Predict cost for a sample.
    pub fn predict(&self, sample: &NnueSample) -> f32 {
        let mut sum = self.bias;
        for &idx in &sample.feature_indices {
            if idx < self.weights.len() {
                sum += self.weights[idx];
            }
        }
        sum
    }
}

impl Default for LinearNnue {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// NNUE Trainer (Linear version that works)
// ============================================================================

/// Training state and methods for linear NNUE.
pub struct NnueTrainer {
    /// The linear model being trained.
    pub model: LinearNnue,
    /// Training configuration.
    pub config: TrainConfig,
    /// Training samples.
    samples: Vec<NnueSample>,
    /// Random state for shuffling.
    rng_state: u64,
}

impl NnueTrainer {
    /// Create a new trainer with default config.
    pub fn new() -> Self {
        Self {
            model: LinearNnue::new(),
            config: TrainConfig::default(),
            samples: Vec::new(),
            rng_state: 42,
        }
    }

    /// Create a new trainer with custom config.
    pub fn with_config(train_config: TrainConfig) -> Self {
        Self {
            model: LinearNnue::new(),
            config: train_config,
            samples: Vec::new(),
            rng_state: 42,
        }
    }

    /// Add a training sample.
    pub fn add_sample(&mut self, sample: NnueSample) {
        self.samples.push(sample);
    }

    /// Add multiple samples from (Expr, cost) pairs.
    pub fn add_expr_samples(&mut self, pairs: &[(&Expr, f32)]) {
        for (expr, cost) in pairs {
            self.samples.push(NnueSample::from_expr(expr, *cost));
        }
    }

    /// Simple LCG random number generator.
    fn rand_f32(&mut self) -> f32 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.rng_state >> 33) as f32 / (1u64 << 31) as f32
    }

    /// Shuffle samples for mini-batch training.
    fn shuffle_samples(&mut self) {
        let n = self.samples.len();
        for i in (1..n).rev() {
            let j = (self.rand_f32() * (i + 1) as f32) as usize;
            self.samples.swap(i, j);
        }
    }

    /// Get target value for a sample.
    fn get_target(&self, sample: &NnueSample) -> f32 {
        if self.config.use_log_transform {
            sample.log_target
        } else {
            sample.target_cost
        }
    }

    /// Train the model using SGD.
    ///
    /// Returns training history (loss per epoch).
    #[cfg(feature = "std")]
    pub fn train(&mut self) -> Vec<f32> {
        if self.samples.is_empty() {
            return Vec::new();
        }

        let mut history = Vec::with_capacity(self.config.epochs);

        // Initialize bias to mean of targets
        let mean_target: f32 = self.samples.iter().map(|s| self.get_target(s)).sum::<f32>()
            / self.samples.len() as f32;
        self.model.bias = mean_target;

        for epoch in 0..self.config.epochs {
            self.shuffle_samples();

            let mut epoch_loss = 0.0f32;
            let n_samples = self.samples.len();

            // SGD over all samples
            for i in 0..n_samples {
                let sample = &self.samples[i];
                let target = self.get_target(sample);
                let prediction = self.model.predict(sample);
                let error = prediction - target;

                epoch_loss += error * error;

                // Update weights for active features
                let lr = self.config.learning_rate;
                let l2 = self.config.l2_lambda;

                for &idx in &sample.feature_indices {
                    if idx < self.model.weights.len() {
                        // Gradient: d(MSE)/dw = 2 * error * 1 (since feature is 1 when active)
                        // Plus L2 regularization: + 2 * l2 * w
                        let grad = 2.0 * error + 2.0 * l2 * self.model.weights[idx];
                        self.model.weights[idx] -= lr * grad;
                    }
                }

                // Update bias
                self.model.bias -= lr * 2.0 * error;
            }

            epoch_loss /= n_samples as f32;
            history.push(epoch_loss);

            if self.config.print_every > 0 && (epoch + 1) % self.config.print_every == 0 {
                let corr = self.spearman_correlation(&self.samples.clone());
                eprintln!(
                    "Epoch {}/{}: loss = {:.4}, spearman = {:.4}",
                    epoch + 1,
                    self.config.epochs,
                    epoch_loss,
                    corr
                );
            }
        }

        history
    }

    /// Evaluate the model on a set of samples.
    ///
    /// Returns (predictions, targets) for correlation analysis.
    pub fn evaluate(&self, samples: &[NnueSample]) -> (Vec<f32>, Vec<f32>) {
        let mut predictions = Vec::with_capacity(samples.len());
        let mut targets = Vec::with_capacity(samples.len());

        for sample in samples {
            predictions.push(self.model.predict(sample));
            targets.push(sample.target_cost);
        }

        (predictions, targets)
    }

    /// Compute Spearman rank correlation between predictions and targets.
    pub fn spearman_correlation(&self, samples: &[NnueSample]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }

        let (predictions, targets) = self.evaluate(samples);

        // Compute ranks
        let pred_ranks = compute_ranks(&predictions);
        let target_ranks = compute_ranks(&targets);

        // Pearson correlation of ranks
        let n = samples.len() as f32;
        let mean_pred = pred_ranks.iter().sum::<f32>() / n;
        let mean_target = target_ranks.iter().sum::<f32>() / n;

        let mut cov = 0.0f32;
        let mut var_pred = 0.0f32;
        let mut var_target = 0.0f32;

        for i in 0..samples.len() {
            let dp = pred_ranks[i] - mean_pred;
            let dt = target_ranks[i] - mean_target;
            cov += dp * dt;
            var_pred += dp * dp;
            var_target += dt * dt;
        }

        if var_pred < 1e-10 || var_target < 1e-10 {
            return 0.0;
        }

        cov / (var_pred.sqrt() * var_target.sqrt())
    }
}

impl Default for NnueTrainer {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute ranks for a vector of values.
fn compute_ranks(values: &[f32]) -> Vec<f32> {
    let n = values.len();
    let mut indexed: Vec<(usize, f32)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();

    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

    let mut ranks = vec![0.0f32; n];
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        ranks[*idx] = rank as f32 + 1.0;
    }

    ranks
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue::OpKind;

    #[test]
    fn sample_creation_should_succeed_when_called() {
        let expr = Expr::Binary(
            OpKind::Add,
            alloc::boxed::Box::new(Expr::Var(0)),
            alloc::boxed::Box::new(Expr::Var(1)),
        );
        let sample = NnueSample::from_expr(&expr, 100.0);
        assert!(!sample.feature_indices.is_empty());
        assert!((sample.target_cost - 100.0).abs() < 0.001);
    }

    #[test]
    fn trainer_creation_should_succeed_when_called() {
        let trainer = NnueTrainer::new();
        assert_eq!(trainer.samples.len(), 0);
    }

    #[test]
    fn linear_nnue_predict_should_succeed_when_called() {
        let mut model = LinearNnue::new();
        model.bias = 1.0;
        model.weights[0] = 0.5;
        model.weights[1] = 0.3;

        let sample = NnueSample::from_features(vec![0, 1], 10.0);
        let pred = model.predict(&sample);
        // Should be 1.0 + 0.5 + 0.3 = 1.8
        assert!((pred - 1.8).abs() < 0.001);
    }

    #[test]
    fn compute_ranks_should_succeed_when_called() {
        let values = vec![3.0, 1.0, 4.0, 1.5, 2.0];
        let ranks = compute_ranks(&values);
        // Expected: 1.0→1, 1.5→2, 2.0→3, 3.0→4, 4.0→5
        // So: [4, 1, 5, 2, 3]
        assert!((ranks[0] - 4.0).abs() < 0.001); // 3.0 is 4th smallest
        assert!((ranks[1] - 1.0).abs() < 0.001); // 1.0 is 1st smallest
        assert!((ranks[2] - 5.0).abs() < 0.001); // 4.0 is 5th smallest
        assert!((ranks[3] - 2.0).abs() < 0.001); // 1.5 is 2nd smallest
        assert!((ranks[4] - 3.0).abs() < 0.001); // 2.0 is 3rd smallest
    }
}
