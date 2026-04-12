//! # Training for Factored Embedding NNUE
//!
//! This module provides training infrastructure for the factored NNUE architecture:
//!
//! - **SGD trainer** with momentum and learning rate scheduling
//! - **Backpropagation** through the network to embeddings
//! - **Data loading** from benchmark_cache.jsonl
//!
//! ## Training Data Format
//!
//! Expects JSONL with lines like:
//! ```json
//! {"expr": "Add(Mul(Var(0), Var(1)), Var(2))", "cost_ns": 1234}
//! ```
//!
//! ## Loss Function
//!
//! We train on log-cost (MSE loss):
//! ```text
//! L = (predicted_log_cost - actual_log_cost)²
//! ```
//!
//! This handles the wide dynamic range of costs (10ns to 10000ns).

use pixelflow_nnue::factored::{
    EdgeAccumulator, FactoredNnue, HIDDEN_DIM, INPUT_DIM, K, OpEmbeddings, StructuralFeatures,
};
use pixelflow_nnue::{Expr, OpKind};

// ============================================================================
// Training Sample
// ============================================================================

/// A training sample for factored NNUE.
#[derive(Clone)]
pub struct FactoredSample {
    /// The expression.
    pub expr: Expr,

    /// Ground truth cost in nanoseconds.
    pub cost_ns: f64,

    /// Precomputed edge accumulator.
    pub accumulator: EdgeAccumulator,

    /// Precomputed structural features.
    pub structural: StructuralFeatures,
}

impl FactoredSample {
    /// Create a new sample from expression and cost.
    pub fn new(expr: Expr, cost_ns: f64, embeddings: &OpEmbeddings) -> Self {
        let accumulator = EdgeAccumulator::from_expr(&expr, embeddings);
        let structural = StructuralFeatures::from_expr(&expr);
        Self {
            expr,
            cost_ns,
            accumulator,
            structural,
        }
    }

    /// Recompute the accumulator with updated embeddings.
    pub fn recompute_accumulator(&mut self, embeddings: &OpEmbeddings) {
        self.accumulator = EdgeAccumulator::from_expr(&self.expr, embeddings);
    }

    /// Target value for training (log of cost).
    #[inline]
    pub fn target(&self) -> f32 {
        (self.cost_ns as f32).ln()
    }
}

// ============================================================================
// Gradient Accumulators
// ============================================================================

/// Gradients for all network parameters.
#[derive(Clone)]
pub struct Gradients {
    /// Embedding gradients: d(loss)/d(embedding[op][k]).
    pub d_emb: [[f32; K]; OpKind::COUNT],

    /// W1 gradients.
    pub d_w1: [[f32; HIDDEN_DIM]; INPUT_DIM],

    /// B1 gradients.
    pub d_b1: [f32; HIDDEN_DIM],

    /// W2 gradients.
    pub d_w2: [f32; HIDDEN_DIM],

    /// B2 gradient.
    pub d_b2: f32,
}

impl Default for Gradients {
    fn default() -> Self {
        Self::new()
    }
}

impl Gradients {
    /// Create zero-initialized gradients.
    pub fn new() -> Self {
        Self {
            d_emb: [[0.0; K]; OpKind::COUNT],
            d_w1: [[0.0; HIDDEN_DIM]; INPUT_DIM],
            d_b1: [0.0; HIDDEN_DIM],
            d_w2: [0.0; HIDDEN_DIM],
            d_b2: 0.0,
        }
    }

    /// Reset all gradients to zero.
    pub fn zero(&mut self) {
        for row in &mut self.d_emb {
            row.fill(0.0);
        }
        for row in &mut self.d_w1 {
            row.fill(0.0);
        }
        self.d_b1.fill(0.0);
        self.d_w2.fill(0.0);
        self.d_b2 = 0.0;
    }

    /// Scale all gradients (for averaging over minibatch).
    pub fn scale(&mut self, factor: f32) {
        for row in &mut self.d_emb {
            for v in row.iter_mut() {
                *v *= factor;
            }
        }
        for row in &mut self.d_w1 {
            for v in row.iter_mut() {
                *v *= factor;
            }
        }
        for v in &mut self.d_b1 {
            *v *= factor;
        }
        for v in &mut self.d_w2 {
            *v *= factor;
        }
        self.d_b2 *= factor;
    }

    /// Accumulate gradients from another Gradients struct.
    pub fn accumulate(&mut self, other: &Gradients) {
        for (row, other_row) in self.d_emb.iter_mut().zip(other.d_emb.iter()) {
            for (v, &ov) in row.iter_mut().zip(other_row.iter()) {
                *v += ov;
            }
        }
        for (row, other_row) in self.d_w1.iter_mut().zip(other.d_w1.iter()) {
            for (v, &ov) in row.iter_mut().zip(other_row.iter()) {
                *v += ov;
            }
        }
        for (v, &ov) in self.d_b1.iter_mut().zip(other.d_b1.iter()) {
            *v += ov;
        }
        for (v, &ov) in self.d_w2.iter_mut().zip(other.d_w2.iter()) {
            *v += ov;
        }
        self.d_b2 += other.d_b2;
    }
}

// ============================================================================
// Forward/Backward Pass with Gradient Computation
// ============================================================================

/// Cached intermediate values from forward pass (needed for backward).
pub struct ForwardCache {
    /// Input to hidden layer (accumulator + structural).
    pub input: [f32; INPUT_DIM],

    /// Pre-activation hidden values.
    pub hidden_pre: [f32; HIDDEN_DIM],

    /// Post-ReLU hidden values.
    pub hidden: [f32; HIDDEN_DIM],

    /// Network output (predicted log-cost).
    pub output: f32,
}

impl ForwardCache {
    /// Run forward pass and cache intermediates.
    pub fn forward(
        net: &FactoredNnue,
        acc: &EdgeAccumulator,
        structural: &StructuralFeatures,
    ) -> Self {
        let mut input = [0.0f32; INPUT_DIM];

        // Copy accumulator values (first 2K dims)
        input[..2 * K].copy_from_slice(&acc.values);

        // Copy structural features (remaining dims)
        input[2 * K..].copy_from_slice(&structural.values);

        // Hidden layer: input @ W1 + b1
        let mut hidden_pre = net.b1;
        for (i, &inp) in input.iter().enumerate() {
            for (j, h) in hidden_pre.iter_mut().enumerate() {
                *h += inp * net.w1[i][j];
            }
        }

        // ReLU
        let mut hidden = hidden_pre;
        for h in &mut hidden {
            *h = h.max(0.0);
        }

        // Output layer
        let mut output = net.b2;
        for (&h, &w) in hidden.iter().zip(net.w2.iter()) {
            output += h * w;
        }

        Self {
            input,
            hidden_pre,
            hidden,
            output,
        }
    }
}

/// Compute gradients via backpropagation.
///
/// Returns the loss value (MSE on log-cost).
pub fn backward(
    net: &FactoredNnue,
    cache: &ForwardCache,
    target: f32,
    sample: &FactoredSample,
    grads: &mut Gradients,
) -> f32 {
    // MSE loss: L = (output - target)²
    let diff = cache.output - target;
    let loss = diff * diff;

    // d(loss)/d(output) = 2 * (output - target)
    let d_output = 2.0 * diff;

    // Gradient through output layer
    grads.d_b2 += d_output;
    let mut d_hidden = [0.0f32; HIDDEN_DIM];
    for (i, &h) in cache.hidden.iter().enumerate() {
        grads.d_w2[i] += d_output * h;
        d_hidden[i] = d_output * net.w2[i];
    }

    // Gradient through ReLU
    for (i, dh) in d_hidden.iter_mut().enumerate() {
        if cache.hidden_pre[i] <= 0.0 {
            *dh = 0.0; // ReLU gradient is 0 for negative pre-activation
        }
    }

    // Gradient through hidden layer
    for b in grads.d_b1.iter_mut().zip(d_hidden.iter()) {
        *b.0 += *b.1;
    }

    let mut d_input = [0.0f32; INPUT_DIM];
    for (i, inp) in cache.input.iter().enumerate() {
        for (j, &dh) in d_hidden.iter().enumerate() {
            grads.d_w1[i][j] += dh * inp;
            d_input[i] += dh * net.w1[i][j];
        }
    }

    // Gradient through accumulator to embeddings
    // The accumulator values come from: acc[0..K] = Σ E[parent], acc[K..2K] = Σ E[child]
    // So d(loss)/d(E[op]) = Σ d_input[i] for each edge where op appears

    // Extract edges from expression and propagate gradients
    propagate_embedding_gradients(&sample.expr, &d_input[..2 * K], &mut grads.d_emb);

    loss
}

/// Propagate gradients from accumulator to embeddings.
fn propagate_embedding_gradients(
    expr: &Expr,
    d_acc: &[f32],
    d_emb: &mut [[f32; K]; OpKind::COUNT],
) {
    let parent_op = expr.op_type();

    match expr {
        Expr::Var(_) | Expr::Const(_) => {}
        Expr::Unary(_, child) => {
            let child_op = child.op_type();

            // This edge contributes: acc[0..K] += E[parent], acc[K..2K] += E[child]
            // So gradients flow back:
            for k in 0..K {
                d_emb[parent_op.index()][k] += d_acc[k]; // parent contributes to first half
                d_emb[child_op.index()][k] += d_acc[K + k]; // child contributes to second half
            }

            propagate_embedding_gradients(child, d_acc, d_emb);
        }
        Expr::Binary(_, left, right) => {
            let left_op = left.op_type();
            let right_op = right.op_type();

            // Two edges from this parent
            for k in 0..K {
                d_emb[parent_op.index()][k] += d_acc[k] * 2.0; // parent appears in 2 edges
                d_emb[left_op.index()][k] += d_acc[K + k];
                d_emb[right_op.index()][k] += d_acc[K + k];
            }

            propagate_embedding_gradients(left, d_acc, d_emb);
            propagate_embedding_gradients(right, d_acc, d_emb);
        }
        Expr::Ternary(_, a, b, c) => {
            let a_op = a.op_type();
            let b_op = b.op_type();
            let c_op = c.op_type();

            // Three edges from this parent
            for k in 0..K {
                d_emb[parent_op.index()][k] += d_acc[k] * 3.0; // parent appears in 3 edges
                d_emb[a_op.index()][k] += d_acc[K + k];
                d_emb[b_op.index()][k] += d_acc[K + k];
                d_emb[c_op.index()][k] += d_acc[K + k];
            }

            propagate_embedding_gradients(a, d_acc, d_emb);
            propagate_embedding_gradients(b, d_acc, d_emb);
            propagate_embedding_gradients(c, d_acc, d_emb);
        }
    }
}

// ============================================================================
// SGD Trainer
// ============================================================================

/// Training configuration.
#[derive(Clone)]
pub struct TrainConfig {
    /// Learning rate.
    pub learning_rate: f32,

    /// Momentum coefficient.
    pub momentum: f32,

    /// Weight decay (L2 regularization).
    pub weight_decay: f32,

    /// Minibatch size.
    pub batch_size: usize,

    /// Number of epochs.
    pub epochs: usize,

    /// Learning rate decay factor per epoch.
    pub lr_decay: f32,

    /// Gradient clipping threshold.
    pub grad_clip: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: 0.9,
            weight_decay: 1e-5,
            batch_size: 32,
            epochs: 10,
            lr_decay: 0.95,
            grad_clip: 1.0,
        }
    }
}

/// Momentum buffers for SGD.
pub struct Momentum {
    /// Embedding momentum.
    pub v_emb: [[f32; K]; OpKind::COUNT],

    /// W1 momentum.
    pub v_w1: [[f32; HIDDEN_DIM]; INPUT_DIM],

    /// B1 momentum.
    pub v_b1: [f32; HIDDEN_DIM],

    /// W2 momentum.
    pub v_w2: [f32; HIDDEN_DIM],

    /// B2 momentum.
    pub v_b2: f32,
}

impl Default for Momentum {
    fn default() -> Self {
        Self::new()
    }
}

impl Momentum {
    /// Create zero-initialized momentum buffers.
    pub fn new() -> Self {
        Self {
            v_emb: [[0.0; K]; OpKind::COUNT],
            v_w1: [[0.0; HIDDEN_DIM]; INPUT_DIM],
            v_b1: [0.0; HIDDEN_DIM],
            v_w2: [0.0; HIDDEN_DIM],
            v_b2: 0.0,
        }
    }
}

/// SGD trainer for factored NNUE.
pub struct FactoredTrainer {
    /// The network being trained.
    pub net: FactoredNnue,

    /// Training configuration.
    pub config: TrainConfig,

    /// Momentum buffers.
    pub momentum: Momentum,

    /// Current learning rate (may decay).
    pub current_lr: f32,

    /// Training samples.
    pub samples: Vec<FactoredSample>,
}

impl FactoredTrainer {
    /// Create a new trainer with randomly initialized network.
    pub fn new(config: TrainConfig, seed: u64) -> Self {
        Self {
            net: FactoredNnue::new_random(seed),
            current_lr: config.learning_rate,
            config,
            momentum: Momentum::new(),
            samples: Vec::new(),
        }
    }

    /// Create a new trainer with latency-prior initialized embeddings.
    ///
    /// This is the recommended constructor for cost prediction:
    /// - Embeddings start with known op latencies
    /// - Model can immediately distinguish Add from Div from Sqrt
    /// - Remaining dimensions learn subtle interactions
    pub fn new_with_latency_prior(config: TrainConfig, seed: u64) -> Self {
        Self {
            net: FactoredNnue::new_with_latency_prior(seed),
            current_lr: config.learning_rate,
            config,
            momentum: Momentum::new(),
            samples: Vec::new(),
        }
    }

    /// Add a training sample.
    pub fn add_sample(&mut self, expr: Expr, cost_ns: f64) {
        let sample = FactoredSample::new(expr, cost_ns, &self.net.embeddings);
        self.samples.push(sample);
    }

    /// Recompute all accumulators (needed after embedding updates).
    pub fn recompute_accumulators(&mut self) {
        for sample in &mut self.samples {
            sample.recompute_accumulator(&self.net.embeddings);
        }
    }

    /// Train for one epoch.
    ///
    /// Returns the average loss over the epoch.
    pub fn train_epoch(&mut self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let mut batch_count = 0;
        let batch_size = self.config.batch_size;
        let num_samples = self.samples.len();

        // Simple sequential iteration (could shuffle for better training)
        for batch_start in (0..num_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(num_samples);
            let batch_loss = self.train_batch_range(batch_start, batch_end);
            total_loss += batch_loss * (batch_end - batch_start) as f32;
            batch_count += batch_end - batch_start;
        }

        // Decay learning rate
        self.current_lr *= self.config.lr_decay;

        // Recompute accumulators after embedding updates
        self.recompute_accumulators();

        total_loss / batch_count as f32
    }

    /// Train on a batch specified by index range.
    fn train_batch_range(&mut self, start: usize, end: usize) -> f32 {
        let mut grads = Gradients::new();
        let mut total_loss = 0.0;
        let batch_len = end - start;

        // Accumulate gradients over batch
        for i in start..end {
            let sample = &self.samples[i];
            let cache = ForwardCache::forward(&self.net, &sample.accumulator, &sample.structural);
            let target = sample.target();
            // Need to clone sample for backward since it borrows self
            let sample_clone = sample.clone();
            let loss = backward(&self.net, &cache, target, &sample_clone, &mut grads);
            total_loss += loss;
        }

        // Average gradients
        let scale = 1.0 / batch_len as f32;
        grads.scale(scale);

        // Clip gradients
        self.clip_gradients(&mut grads);

        // Apply gradients with momentum
        self.apply_gradients(&grads);

        total_loss / batch_len as f32
    }

    /// Clip gradients to prevent explosion.
    fn clip_gradients(&self, grads: &mut Gradients) {
        let clip = self.config.grad_clip;

        for row in &mut grads.d_emb {
            for v in row.iter_mut() {
                *v = v.clamp(-clip, clip);
            }
        }
        for row in &mut grads.d_w1 {
            for v in row.iter_mut() {
                *v = v.clamp(-clip, clip);
            }
        }
        for v in &mut grads.d_b1 {
            *v = v.clamp(-clip, clip);
        }
        for v in &mut grads.d_w2 {
            *v = v.clamp(-clip, clip);
        }
        grads.d_b2 = grads.d_b2.clamp(-clip, clip);
    }

    /// Apply gradients with momentum and weight decay.
    fn apply_gradients(&mut self, grads: &Gradients) {
        let lr = self.current_lr;
        let mom = self.config.momentum;
        let wd = self.config.weight_decay;

        // Update embeddings
        for (op_idx, (emb_row, grad_row)) in self
            .net
            .embeddings
            .e
            .iter_mut()
            .zip(grads.d_emb.iter())
            .enumerate()
        {
            for k in 0..K {
                // Momentum update
                self.momentum.v_emb[op_idx][k] = mom * self.momentum.v_emb[op_idx][k] + grad_row[k];

                // Weight decay + gradient step
                emb_row[k] -= lr * (self.momentum.v_emb[op_idx][k] + wd * emb_row[k]);
            }
        }

        // Update W1
        for i in 0..INPUT_DIM {
            for j in 0..HIDDEN_DIM {
                self.momentum.v_w1[i][j] = mom * self.momentum.v_w1[i][j] + grads.d_w1[i][j];
                self.net.w1[i][j] -= lr * (self.momentum.v_w1[i][j] + wd * self.net.w1[i][j]);
            }
        }

        // Update B1
        for i in 0..HIDDEN_DIM {
            self.momentum.v_b1[i] = mom * self.momentum.v_b1[i] + grads.d_b1[i];
            self.net.b1[i] -= lr * self.momentum.v_b1[i]; // No weight decay on biases
        }

        // Update W2
        for i in 0..HIDDEN_DIM {
            self.momentum.v_w2[i] = mom * self.momentum.v_w2[i] + grads.d_w2[i];
            self.net.w2[i] -= lr * (self.momentum.v_w2[i] + wd * self.net.w2[i]);
        }

        // Update B2
        self.momentum.v_b2 = mom * self.momentum.v_b2 + grads.d_b2;
        self.net.b2 -= lr * self.momentum.v_b2;
    }

    /// Compute evaluation metrics on current samples.
    pub fn evaluate(&self) -> TrainMetrics {
        if self.samples.is_empty() {
            return TrainMetrics::default();
        }

        let mut predictions = Vec::with_capacity(self.samples.len());
        let mut targets = Vec::with_capacity(self.samples.len());
        let mut total_loss = 0.0;

        for sample in &self.samples {
            let pred = self.net.forward(&sample.accumulator, &sample.structural);
            let target = sample.target();
            predictions.push(pred);
            targets.push(target);
            total_loss += (pred - target).powi(2);
        }

        let mse = total_loss / self.samples.len() as f32;
        let rmse = mse.sqrt();
        let spearman = compute_spearman(&predictions, &targets);

        TrainMetrics {
            mse,
            rmse,
            spearman,
        }
    }
}

/// Training metrics.
#[derive(Clone, Default)]
pub struct TrainMetrics {
    /// Mean squared error on log-costs.
    pub mse: f32,

    /// Root mean squared error.
    pub rmse: f32,

    /// Spearman rank correlation.
    pub spearman: f32,
}

/// Compute Spearman rank correlation coefficient.
fn compute_spearman(predictions: &[f32], targets: &[f32]) -> f32 {
    let n = predictions.len();
    if n < 2 {
        return 0.0;
    }

    // Get ranks for predictions
    let pred_ranks = compute_ranks(predictions);
    let target_ranks = compute_ranks(targets);

    // Compute correlation
    let mean_pred: f32 = pred_ranks.iter().sum::<f32>() / n as f32;
    let mean_target: f32 = target_ranks.iter().sum::<f32>() / n as f32;

    let mut cov = 0.0f32;
    let mut var_pred = 0.0f32;
    let mut var_target = 0.0f32;

    for i in 0..n {
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

/// Compute ranks for values.
fn compute_ranks(values: &[f32]) -> Vec<f32> {
    let n = values.len();
    let mut indexed: Vec<_> = values.iter().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    for (rank, (orig_idx, _)) in indexed.into_iter().enumerate() {
        ranks[orig_idx] = rank as f32 + 1.0;
    }
    ranks
}

// ============================================================================
// Expression Parsing (for loading training data)
// ============================================================================

/// Parse an expression from a string representation.
///
/// Format: `OpName(child1, child2, ...)` or `Var(n)` or `Const(value)`
pub fn parse_expr(s: &str) -> Option<Expr> {
    let s = s.trim();

    // Try parsing as Var
    if s.starts_with("Var(") && s.ends_with(')') {
        let inner = &s[4..s.len() - 1];
        let idx: u8 = inner.parse().ok()?;
        return Some(Expr::Var(idx));
    }

    // Try parsing as Const
    if s.starts_with("Const(") && s.ends_with(')') {
        let inner = &s[6..s.len() - 1];
        let val: f32 = inner.parse().ok()?;
        return Some(Expr::Const(val));
    }

    // Parse as operation
    let paren_pos = s.find('(')?;
    let op_name = &s[..paren_pos];
    let op = parse_op_kind(op_name)?;

    // Find matching closing paren
    let inner = &s[paren_pos + 1..s.len() - 1];
    let children = split_args(inner);

    match op.arity() {
        0 => None, // Should have been caught above
        1 => {
            if children.len() != 1 {
                return None;
            }
            let a = parse_expr(children[0])?;
            Some(Expr::Unary(op, Box::new(a)))
        }
        2 => {
            if children.len() != 2 {
                return None;
            }
            let a = parse_expr(children[0])?;
            let b = parse_expr(children[1])?;
            Some(Expr::Binary(op, Box::new(a), Box::new(b)))
        }
        3 => {
            if children.len() != 3 {
                return None;
            }
            let a = parse_expr(children[0])?;
            let b = parse_expr(children[1])?;
            let c = parse_expr(children[2])?;
            Some(Expr::Ternary(op, Box::new(a), Box::new(b), Box::new(c)))
        }
        _ => None,
    }
}

/// Parse operation name to OpKind.
fn parse_op_kind(name: &str) -> Option<OpKind> {
    match name.to_lowercase().as_str() {
        "add" => Some(OpKind::Add),
        "sub" => Some(OpKind::Sub),
        "mul" => Some(OpKind::Mul),
        "div" => Some(OpKind::Div),
        "neg" => Some(OpKind::Neg),
        "sqrt" => Some(OpKind::Sqrt),
        "rsqrt" => Some(OpKind::Rsqrt),
        "abs" => Some(OpKind::Abs),
        "min" => Some(OpKind::Min),
        "max" => Some(OpKind::Max),
        "muladd" | "mul_add" | "fma" => Some(OpKind::MulAdd),
        "mulrsqrt" | "mul_rsqrt" => Some(OpKind::MulRsqrt),
        "recip" => Some(OpKind::Recip),
        "floor" => Some(OpKind::Floor),
        "ceil" => Some(OpKind::Ceil),
        "round" => Some(OpKind::Round),
        "fract" => Some(OpKind::Fract),
        "sin" => Some(OpKind::Sin),
        "cos" => Some(OpKind::Cos),
        "tan" => Some(OpKind::Tan),
        "asin" => Some(OpKind::Asin),
        "acos" => Some(OpKind::Acos),
        "atan" => Some(OpKind::Atan),
        "atan2" => Some(OpKind::Atan2),
        "exp" => Some(OpKind::Exp),
        "exp2" => Some(OpKind::Exp2),
        "ln" => Some(OpKind::Ln),
        "log2" => Some(OpKind::Log2),
        "log10" => Some(OpKind::Log10),
        "pow" => Some(OpKind::Pow),
        "hypot" => Some(OpKind::Hypot),
        _ => None,
    }
}

/// Split comma-separated arguments, respecting nested parentheses.
fn split_args(s: &str) -> Vec<&str> {
    let mut args = Vec::new();
    let mut depth = 0;
    let mut start = 0;

    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                args.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }

    if start < s.len() {
        args.push(s[start..].trim());
    }

    args
}

// ============================================================================
// Kernel Code Parser (Functional Recursive Descent)
// ============================================================================
//
// Grammar:
//   expr     ::= additive
//   additive ::= multiplicative (('+' | '-') multiplicative)*
//   mult     ::= postfix (('*' | '/') postfix)*
//   postfix  ::= primary ('.' method)*
//   method   ::= IDENT '(' expr? ')'
//   primary  ::= '(' expr ')' | '-' postfix | VAR | NUM
//   VAR      ::= 'X' | 'Y' | 'Z' | 'W'
//   NUM      ::= float literal

/// Parser result: (parsed value, remaining input)
type ParseResult<'a, T> = Option<(T, &'a str)>;

/// Parse kernel code syntax like "(X + Y)" into Expr.
pub fn parse_kernel_code(s: &str) -> Option<Expr> {
    kc_expr(s.trim()).and_then(|(expr, rest)| rest.is_empty().then_some(expr))
}

/// Top-level: parse a complete expression
fn kc_expr(input: &str) -> ParseResult<Expr> {
    parse_additive(input.trim())
}

/// Parse additive: left-associative chain of +/-
fn parse_additive(input: &str) -> ParseResult<Expr> {
    let (mut acc, mut rest) = parse_multiplicative(input)?;

    while let Some((op, remaining)) = parse_additive_op(rest.trim_start()) {
        let (rhs, remaining) = parse_multiplicative(remaining.trim_start())?;
        acc = Expr::Binary(op, Box::new(acc), Box::new(rhs));
        rest = remaining;
    }

    Some((acc, rest))
}

fn parse_additive_op(input: &str) -> ParseResult<OpKind> {
    match input.chars().next()? {
        '+' => Some((OpKind::Add, &input[1..])),
        '-' => Some((OpKind::Sub, &input[1..])),
        _ => None,
    }
}

/// Parse multiplicative: left-associative chain of * /
fn parse_multiplicative(input: &str) -> ParseResult<Expr> {
    let (mut acc, mut rest) = parse_postfix(input)?;

    while let Some((op, remaining)) = parse_multiplicative_op(rest.trim_start()) {
        let (rhs, remaining) = parse_postfix(remaining.trim_start())?;
        acc = Expr::Binary(op, Box::new(acc), Box::new(rhs));
        rest = remaining;
    }

    Some((acc, rest))
}

fn parse_multiplicative_op(input: &str) -> ParseResult<OpKind> {
    match input.chars().next()? {
        '*' => Some((OpKind::Mul, &input[1..])),
        '/' => Some((OpKind::Div, &input[1..])),
        _ => None,
    }
}

/// Parse postfix: primary followed by method chains
fn parse_postfix(input: &str) -> ParseResult<Expr> {
    let (mut acc, mut rest) = parse_primary(input)?;

    while let Some((expr, remaining)) = parse_method_call(rest.trim_start(), acc.clone()) {
        acc = expr;
        rest = remaining;
    }

    Some((acc, rest))
}

/// Parse a method call: .method() or .method(arg)
fn parse_method_call<'a>(input: &'a str, base: Expr) -> ParseResult<'a, Expr> {
    let input = input.strip_prefix('.')?;
    let (method_name, rest) = parse_ident(input)?;
    let rest = rest.strip_prefix('(')?;

    // Check for unary method (empty args)
    if let Some(rest) = rest.trim_start().strip_prefix(')') {
        let op = match method_name {
            "sqrt" => OpKind::Sqrt,
            "rsqrt" => OpKind::Rsqrt,
            "abs" => OpKind::Abs,
            "floor" => OpKind::Floor,
            "ceil" => OpKind::Ceil,
            "round" => OpKind::Round,
            "fract" => OpKind::Fract,
            "sin" => OpKind::Sin,
            "cos" => OpKind::Cos,
            "tan" => OpKind::Tan,
            "exp" => OpKind::Exp,
            "exp2" => OpKind::Exp2,
            "ln" => OpKind::Ln,
            "log2" => OpKind::Log2,
            _ => return None,
        };
        return Some((Expr::Unary(op, Box::new(base)), rest));
    }

    // Binary method with argument
    let (arg, rest) = kc_expr(rest.trim_start())?;
    let rest = rest.trim_start().strip_prefix(')')?;

    let op = match method_name {
        "min" => OpKind::Min,
        "max" => OpKind::Max,
        "powf" => OpKind::Pow,
        "atan2" => OpKind::Atan2,
        "hypot" => OpKind::Hypot,
        _ => return None,
    };

    Some((Expr::Binary(op, Box::new(base), Box::new(arg)), rest))
}

/// Parse primary: parens, negation, variable, or number
fn parse_primary(input: &str) -> ParseResult<Expr> {
    let input = input.trim_start();

    // Parenthesized expression
    if let Some(rest) = input.strip_prefix('(') {
        let (expr, rest) = kc_expr(rest)?;
        let rest = rest.trim_start().strip_prefix(')')?;
        return Some((expr, rest));
    }

    // Unary negation
    if let Some(rest) = input.strip_prefix('-') {
        let (expr, rest) = parse_postfix(rest.trim_start())?;
        return Some((Expr::Unary(OpKind::Neg, Box::new(expr)), rest));
    }

    // Variable or number
    parse_variable(input).or_else(|| parse_number(input))
}

/// Parse a variable: X, Y, Z, W
fn parse_variable(input: &str) -> ParseResult<Expr> {
    let (c, rest) = input.split_at(1.min(input.len()));
    match c {
        "X" => Some((Expr::Var(0), rest)),
        "Y" => Some((Expr::Var(1), rest)),
        "Z" => Some((Expr::Var(2), rest)),
        "W" => Some((Expr::Var(3), rest)),
        _ => None,
    }
}

/// Parse a numeric literal
fn parse_number(input: &str) -> ParseResult<Expr> {
    let end = input
        .char_indices()
        .find(|(_, c)| !matches!(c, '0'..='9' | '.' | '-' | 'e' | 'E' | '+'))
        .map(|(i, _)| i)
        .unwrap_or(input.len());

    if end == 0 {
        return None;
    }

    let num_str = &input[..end];
    let val: f32 = num_str.parse().ok()?;
    Some((Expr::Const(val), &input[end..]))
}

/// Parse an identifier (method name)
fn parse_ident(input: &str) -> ParseResult<&str> {
    let end = input
        .char_indices()
        .find(|(_, c)| !c.is_ascii_alphabetic() && *c != '_')
        .map(|(i, _)| i)
        .unwrap_or(input.len());

    if end == 0 {
        return None;
    }

    Some((&input[..end], &input[end..]))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_expr_should_succeed_when_called() {
        let expr = parse_expr("Add(Var(0), Var(1))").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Binary(OpKind::Add, _, _)));

        let expr = parse_expr("Mul(Add(Var(0), Var(1)), Var(2))").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Binary(OpKind::Mul, _, _)));

        let expr = parse_expr("MulAdd(Var(0), Var(1), Var(2))").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Ternary(OpKind::MulAdd, _, _, _)));
    }

    #[test]
    fn forward_backward_should_succeed_when_called() {
        let net = FactoredNnue::new_random(42);
        let expr = parse_expr("Add(Mul(Var(0), Var(1)), Var(2))").expect("Expected value but got None/Err");
        let sample = FactoredSample::new(expr, 100.0, &net.embeddings);

        let cache = ForwardCache::forward(&net, &sample.accumulator, &sample.structural);
        assert!(cache.output.is_finite());

        let mut grads = Gradients::new();
        let loss = backward(&net, &cache, sample.target(), &sample, &mut grads);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);

        // Check that gradients are non-zero
        let total_grad: f32 = grads
            .d_emb
            .iter()
            .flat_map(|r| r.iter())
            .map(|&v| v.abs())
            .sum();
        assert!(total_grad > 0.0, "Should have non-zero embedding gradients");
    }

    #[test]
    fn training_reduces_loss_should_succeed_when_called() {
        let config = TrainConfig {
            learning_rate: 0.1,
            epochs: 5,
            batch_size: 4,
            ..Default::default()
        };

        let mut trainer = FactoredTrainer::new(config, 42);

        // Add some simple samples
        trainer.add_sample(parse_expr("Var(0)").expect("Expected value but got None/Err"), 10.0);
        trainer.add_sample(parse_expr("Add(Var(0), Var(1))").expect("Expected value but got None/Err"), 50.0);
        trainer.add_sample(parse_expr("Mul(Var(0), Var(1))").expect("Expected value but got None/Err"), 60.0);
        trainer.add_sample(parse_expr("Div(Var(0), Var(1))").expect("Expected value but got None/Err"), 200.0);

        let initial_metrics = trainer.evaluate();
        let mut final_loss = 0.0;

        for _ in 0..5 {
            final_loss = trainer.train_epoch();
        }

        let final_metrics = trainer.evaluate();

        // Training should reduce loss
        assert!(
            final_metrics.mse < initial_metrics.mse * 0.99 || final_loss < 1.0,
            "Training should reduce loss: initial={}, final={}",
            initial_metrics.mse,
            final_metrics.mse
        );
    }

    #[test]
    fn spearman_correlation_should_succeed_when_called() {
        // Perfect positive correlation
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let rho = compute_spearman(&a, &b);
        assert!(
            (rho - 1.0).abs() < 0.01,
            "Perfect correlation should be ~1.0"
        );

        // Perfect negative correlation
        let c = vec![50.0, 40.0, 30.0, 20.0, 10.0];
        let rho = compute_spearman(&a, &c);
        assert!(
            (rho - (-1.0)).abs() < 0.01,
            "Perfect negative correlation should be ~-1.0"
        );
    }

    // ========================================================================
    // Kernel Code Parser Tests
    // ========================================================================

    #[test]
    fn parse_kernel_code_variables_should_succeed_when_called() {
        assert!(matches!(parse_kernel_code("X"), Some(Expr::Var(0))));
        assert!(matches!(parse_kernel_code("Y"), Some(Expr::Var(1))));
        assert!(matches!(parse_kernel_code("Z"), Some(Expr::Var(2))));
        assert!(matches!(parse_kernel_code("W"), Some(Expr::Var(3))));
    }

    #[test]
    fn parse_kernel_code_constants_should_succeed_when_called() {
        assert!(matches!(parse_kernel_code("1.0"), Some(Expr::Const(v)) if (v - 1.0).abs() < 1e-6));
        assert!(
            matches!(parse_kernel_code("(4.595877)"), Some(Expr::Const(v)) if (v - 4.595877).abs() < 1e-5)
        );
        assert!(matches!(parse_kernel_code("0.0"), Some(Expr::Const(v)) if v.abs() < 1e-6));
    }

    #[test]
    fn parse_kernel_code_binary_ops_should_succeed_when_called() {
        // Basic addition
        let expr = parse_kernel_code("(X + Y)").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Binary(OpKind::Add, _, _)));

        // Basic subtraction
        let expr = parse_kernel_code("(X - Y)").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Binary(OpKind::Sub, _, _)));

        // Basic multiplication
        let expr = parse_kernel_code("(X * Y)").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Binary(OpKind::Mul, _, _)));

        // Basic division
        let expr = parse_kernel_code("(X / Y)").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Binary(OpKind::Div, _, _)));
    }

    #[test]
    fn parse_kernel_code_from_benchmark_cache_should_succeed_when_called() {
        // Real examples from benchmark_cache.jsonl
        let expr = parse_kernel_code("((4.595877) - Z)").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Binary(OpKind::Sub, _, _)));

        let expr = parse_kernel_code("((4.595877) + (-Z))").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Binary(OpKind::Add, _, _)));

        let expr = parse_kernel_code("((-Z) + (4.595877))").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Binary(OpKind::Add, _, _)));
    }

    #[test]
    fn parse_kernel_code_unary_ops_should_succeed_when_called() {
        // Negation
        let expr = parse_kernel_code("(-X)").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Unary(OpKind::Neg, _)));

        // Method calls on variables
        let expr = parse_kernel_code("(X).sqrt()").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Unary(OpKind::Sqrt, _)));

        let expr = parse_kernel_code("(X).abs()").expect("Expected value but got None/Err");
        assert!(matches!(expr, Expr::Unary(OpKind::Abs, _)));
    }

    #[test]
    fn parse_kernel_code_nested_should_succeed_when_called() {
        // Nested expression: (X + Y) * Z
        let expr = parse_kernel_code("((X + Y) * Z)").expect("Expected value but got None/Err");
        if let Expr::Binary(OpKind::Mul, left, right) = expr {
            assert!(matches!(*left, Expr::Binary(OpKind::Add, _, _)));
            assert!(matches!(*right, Expr::Var(2)));
        } else {
            panic!("Expected Binary(Mul, ...) got {:?}", expr);
        }
    }

    #[test]
    fn parse_kernel_code_method_chains_should_succeed_when_called() {
        // Method on parenthesized expression
        let expr = parse_kernel_code("(X).sqrt()");
        assert!(expr.is_some(), "Should parse (X).sqrt()");

        // min/max methods (binary)
        let expr = parse_kernel_code("(X).min(Y)");
        assert!(expr.is_some(), "Should parse (X).min(Y)");

        // Chained method calls
        let expr = parse_kernel_code("((X).abs()).abs()");
        assert!(expr.is_some(), "Should parse chained abs");

        // Complex real-world expression
        let expr = parse_kernel_code("(((X).rsqrt()).abs())");
        assert!(expr.is_some(), "Should parse rsqrt then abs");
    }

    #[test]
    fn parse_kernel_code_complex_expressions_should_succeed_when_called() {
        // Negative number in parens with rsqrt
        let expr = parse_kernel_code("((-0.724020)).rsqrt()");
        assert!(
            expr.is_some(),
            "Should parse rsqrt of negative const: {:?}",
            expr
        );

        // Deeply nested with multiple methods
        let expr = parse_kernel_code("((((X).rsqrt()).abs()).abs())");
        assert!(expr.is_some(), "Should parse deeply nested methods");

        // Complex min/max
        let expr = parse_kernel_code("(X).min(((-Z)).max(Y))");
        assert!(expr.is_some(), "Should parse nested min/max");

        // Real failing expression (simplified)
        let expr = parse_kernel_code("(((-3.551370)).rsqrt() * (1.0 / W))");
        assert!(expr.is_some(), "Should parse rsqrt multiplication");
    }

    #[test]
    fn parse_actual_failures_should_succeed_when_called() {
        // Exact expression from benchmark cache
        let expr = parse_kernel_code(
            "((((X).rsqrt()).abs()).abs()).min(((((-0.724020)).rsqrt() * (1.0 / (X).abs()))).min(W))",
        );
        assert!(expr.is_some(), "Should parse chained min: {:?}", expr);

        // Another actual failure
        let expr = parse_kernel_code(
            "((W * (((Y * X)).max(X)).rsqrt()) + (W * (-(((Y).abs() * Z) - (((0.296980) * Z) + (-W))))))",
        );
        assert!(expr.is_some(), "Should parse complex expression");
    }
}
