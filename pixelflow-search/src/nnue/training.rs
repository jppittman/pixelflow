//! Training utilities for NNUE-guided search.
//!
//! Self-Imitation Learning (SIL) with resource-asymmetric training:
//! - Oracle runs with abundant resources (high max_classes, many epochs)
//! - Guide runs with limited resources (low max_classes, few epochs)
//! - Guide learns to match oracle quality with fewer resources
//!
//! The key insight: saturation is the limit case. An oracle with unlimited
//! resources can explore freely. A resource-constrained guide must be selective.

/// Resource configuration for search.
#[derive(Clone, Debug)]
pub struct ResourceConfig {
    /// Maximum e-graph classes before stopping.
    pub max_classes: usize,
    /// Maximum epochs to run.
    pub max_epochs: usize,
    /// Filtering threshold (0.5 = balanced).
    pub threshold: f32,
    /// Exploration rate for epsilon-greedy.
    pub epsilon: f32,
}

impl ResourceConfig {
    /// Oracle config: abundant resources for near-saturation.
    #[must_use]
    pub fn oracle() -> Self {
        Self {
            max_classes: 500,
            max_epochs: 20,
            threshold: 0.3,  // permissive
            epsilon: 0.0,    // no exploration - oracle is the teacher
        }
    }

    /// Constrained config: limited resources, must be selective.
    #[must_use]
    pub fn constrained() -> Self {
        Self {
            max_classes: 50,
            max_epochs: 5,
            threshold: 0.5,  // balanced
            epsilon: 0.3,    // exploration during training
        }
    }

    /// Evaluation config: like constrained but no exploration.
    #[must_use]
    pub fn evaluation() -> Self {
        Self {
            max_classes: 50,
            max_epochs: 5,
            threshold: 0.5,
            epsilon: 0.0,  // no exploration for fair eval
        }
    }
}

/// Simple metrics for monitoring training.
#[derive(Clone, Debug, Default)]
pub struct Metrics {
    /// Total samples processed.
    pub total: usize,
    /// Correct predictions.
    pub correct: usize,
    /// False positives (predicted fired, but didn't).
    pub false_positives: usize,
    /// False negatives (predicted no-fire, but did fire).
    pub false_negatives: usize,
    /// Sum of losses.
    pub loss_sum: f32,
}

impl Metrics {
    /// Create new empty metrics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute accuracy (correct / total).
    #[must_use]
    pub fn accuracy(&self) -> f32 {
        if self.total == 0 { 1.0 } else { self.correct as f32 / self.total as f32 }
    }

    /// Compute average loss.
    #[must_use]
    pub fn avg_loss(&self) -> f32 {
        if self.total == 0 { 0.0 } else { self.loss_sum / self.total as f32 }
    }

    /// Record a prediction result.
    pub fn record(&mut self, predicted: bool, actual: bool, loss: f32) {
        self.total += 1;
        self.loss_sum += loss;
        if predicted == actual {
            self.correct += 1;
        } else if predicted && !actual {
            self.false_positives += 1;
        } else {
            self.false_negatives += 1;
        }
    }
}

/// Training result from resource-asymmetric training.
#[derive(Clone, Debug)]
pub struct TrainingResult {
    /// Oracle's final cost (target quality).
    pub oracle_cost: i64,
    /// Guided search's initial cost (before training).
    pub initial_guided_cost: i64,
    /// Guided search's final cost (after training).
    pub final_guided_cost: i64,
    /// Oracle's pairs tried.
    pub oracle_pairs: usize,
    /// Guided search's initial pairs tried.
    pub initial_guided_pairs: usize,
    /// Guided search's final pairs tried.
    pub final_guided_pairs: usize,
}

impl TrainingResult {
    /// Did the guide learn to match oracle quality?
    #[must_use]
    pub fn quality_achieved(&self) -> bool {
        self.final_guided_cost <= self.oracle_cost
    }

    /// Resource efficiency: oracle_pairs / guided_pairs.
    /// Higher = guide is more efficient.
    #[must_use]
    pub fn efficiency_ratio(&self) -> f32 {
        if self.final_guided_pairs == 0 {
            0.0
        } else {
            self.oracle_pairs as f32 / self.final_guided_pairs as f32
        }
    }
}

// ============================================================================
// RESOURCE-ASYMMETRIC TRAINING (Policy Gradient on Search Outcomes)
// ============================================================================
//
// The mask is NOT a recognizer ("did this rule fire?").
// It's a compute budget allocator ("should I spend compute on this?").
//
// The only thing that matters: did the search surface a good extraction?
//
// Training loop:
// 1. Oracle: run with abundant resources → oracle_cost
// 2. Student: run with mask + exploration → student_cost
// 3. Reward: 1 if student_cost <= oracle_cost, else 0
// 4. REINFORCE: update mask towards decisions that led to good outcomes
// ============================================================================

/// Single training episode result.
#[derive(Clone, Debug)]
pub struct EpisodeResult {
    /// Oracle's extraction cost (target).
    pub oracle_cost: i64,
    /// Student's extraction cost.
    pub student_cost: i64,
    /// Whether student matched or beat oracle.
    pub matched: bool,
    /// Number of (expr, rule) pairs the student tried.
    pub pairs_tried: usize,
    /// Number of pairs skipped by mask.
    pub pairs_skipped: usize,
    /// Policy gradient applied this episode.
    pub gradient_norm: f32,
}

impl EpisodeResult {
    /// Reward for this episode (1 if matched, 0 otherwise).
    #[must_use]
    pub fn reward(&self) -> f32 {
        if self.matched { 1.0 } else { 0.0 }
    }

    /// Ratio of oracle_cost to student_cost (higher = student did better).
    #[must_use]
    pub fn cost_ratio(&self) -> f32 {
        if self.student_cost == 0 {
            1.0
        } else {
            self.oracle_cost as f32 / self.student_cost as f32
        }
    }
}

/// Running statistics for baseline subtraction in REINFORCE.
#[derive(Clone, Debug)]
pub struct RewardBaseline {
    /// Exponential moving average of rewards.
    pub mean: f32,
    /// Decay factor for EMA.
    pub decay: f32,
    /// Number of episodes seen.
    pub count: usize,
}

impl Default for RewardBaseline {
    fn default() -> Self {
        Self {
            mean: 0.5, // Start neutral
            decay: 0.99,
            count: 0,
        }
    }
}

impl RewardBaseline {
    /// Create new baseline with specified decay.
    #[must_use]
    pub fn new(decay: f32) -> Self {
        Self { mean: 0.5, decay, count: 0 }
    }

    /// Update baseline with new reward, return advantage.
    pub fn update(&mut self, reward: f32) -> f32 {
        let advantage = reward - self.mean;
        self.mean = self.decay * self.mean + (1.0 - self.decay) * reward;
        self.count += 1;
        advantage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics() {
        let mut metrics = Metrics::new();

        // Record some results
        metrics.record(true, true, 0.1);   // correct
        metrics.record(false, false, 0.2); // correct
        metrics.record(true, false, 0.5);  // false positive
        metrics.record(false, true, 0.8);  // false negative

        assert_eq!(metrics.total, 4);
        assert_eq!(metrics.correct, 2);
        assert_eq!(metrics.false_positives, 1);
        assert_eq!(metrics.false_negatives, 1);
        assert!((metrics.accuracy() - 0.5).abs() < 1e-6);
        assert!((metrics.avg_loss() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_resource_configs() {
        let oracle = ResourceConfig::oracle();
        let constrained = ResourceConfig::constrained();
        let eval = ResourceConfig::evaluation();

        // Oracle should have more resources
        assert!(oracle.max_classes > constrained.max_classes);
        assert!(oracle.max_epochs > constrained.max_epochs);

        // Eval should match constrained but no exploration
        assert_eq!(eval.max_classes, constrained.max_classes);
        assert_eq!(eval.epsilon, 0.0);
    }

    #[test]
    fn test_training_result() {
        let result = TrainingResult {
            oracle_cost: 100,
            initial_guided_cost: 150,
            final_guided_cost: 95,
            oracle_pairs: 1000,
            initial_guided_pairs: 500,
            final_guided_pairs: 200,
        };

        assert!(result.quality_achieved());
        assert!((result.efficiency_ratio() - 5.0).abs() < 1e-6);
    }
}
