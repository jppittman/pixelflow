# Team 4: Backward Pass + Training Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement clean backward pass in `pixelflow-search`, delete `unified_backward.rs`, and refactor `train_unified.rs` to orchestration-only (< 200 lines). No math in `pixelflow-pipeline`.

**Architecture:**
- `pixelflow-search/src/nnue/gradient.rs` — `GradientBuffer` + backward pass functions using the same decomposition as `forward.rs` (Team 3)
- `pixelflow-pipeline/src/training/trajectory.rs` — `Trajectory`, `TrajectoryStep`, collection
- `pixelflow-pipeline/src/training/io.rs` — JSONL read/write (extracted from `self_play.rs`)
- `pixelflow-pipeline/src/training/update.rs` — loss + gradient accumulation (calls `gradient.rs`)
- `pixelflow-pipeline/src/bin/train_unified.rs` — 4-phase shell: GENERATE → EXPORT → CRITIQUE → UPDATE

**Tech Stack:** Rust stable, depends on Teams 2 (IR pullbacks, for future integration) and 3 (forward.rs, constants, network.rs, accumulators.rs).

**Dependencies:** Requires Team 3 (`nnue/forward.rs`, `nnue/network.rs`, `nnue/constants.rs`). Team 2 pullback rules inform gradient structure but Team 4's backward pass works correctly without them (explicit chain rule through the same layered structure as forward.rs).

---

## Context You Need

Read before starting:
- `pixelflow-search/src/nnue/forward.rs` — Team 3 output; mirror this structure in backward
- `pixelflow-search/src/nnue/network.rs` — `ExprNnue` weight fields
- `pixelflow-search/src/nnue/constants.rs` — all dimensions
- `pixelflow-pipeline/src/training/unified_backward.rs` — the 2,971-line file being replaced
- `pixelflow-pipeline/src/training/unified.rs` — `Trajectory`, `TrajectoryStep`, `TrajectoryAdvantages`
- `pixelflow-pipeline/src/training/self_play.rs` — trajectory generation and JSONL I/O
- `pixelflow-pipeline/src/bin/train_unified.rs` — the 2,660-line file to reduce to < 200 lines

Network computation graph (full forward, both heads):
```
EdgeAccumulator → [INPUT_DIM=132 input] → w1[132×64] → ReLU → trunk[64×64] → ReLU → expr_proj[64×32] → value_mlp → cost
GraphAccumulator → [GRAPH_INPUT_DIM=132 input] → graph_w1[132×64] → ReLU → trunk[64×64] → ReLU → graph_proj[64×32] → mask_mlp → interaction → rule_score
```

Backward pass is the chain rule in reverse: for loss `L`,
- Extraction: `dL/d_w2 → dL/d_h → dL/d_value_mlp_w1 → dL/d_expr_embed → dL/d_expr_proj_w → dL/d_hidden → dL/d_trunk_w → dL/d_edge_out → dL/d_w1 → dL/d_embeddings`
- Saturation: `dL/d_score → dL/d_interaction → dL/d_mask_features → dL/d_mask_mlp_w2 → dL/d_mask_h → dL/d_mask_mlp_w1 → dL/d_graph_embed → dL/d_graph_proj_w → dL/d_graph_hidden → dL/d_trunk_w → dL/d_graph_out → dL/d_graph_w1`
- Both heads accumulate into the same `d_trunk_w`/`d_trunk_b`.

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `pixelflow-search/src/nnue/gradient.rs` | Create | `GradientBuffer` + backward functions |
| `pixelflow-pipeline/src/training/trajectory.rs` | Create | `Trajectory`, `TrajectoryStep`, collection from self_play |
| `pixelflow-pipeline/src/training/io.rs` | Create | JSONL read/write |
| `pixelflow-pipeline/src/training/update.rs` | Create | loss + gradient accumulation (calls gradient.rs) |
| `pixelflow-pipeline/src/bin/train_unified.rs` | Shrink | Orchestration only: 4-phase loop |
| `pixelflow-pipeline/src/training/unified_backward.rs` | Delete | Replaced by gradient.rs |
| `pixelflow-pipeline/src/training/unified.rs` | Absorb | Types move to trajectory.rs; file deleted or emptied |
| `pixelflow-pipeline/src/training/mod.rs` | Update | Add new modules |

---

## Task 1: `GradientBuffer` struct

**Files:** Create `pixelflow-search/src/nnue/gradient.rs`, update `pixelflow-search/src/nnue/mod.rs`

- [ ] **Step 1: Write the failing test**

Create `pixelflow-search/tests/nnue_gradient.rs`:

```rust
use pixelflow_search::nnue::gradient::GradientBuffer;
use pixelflow_search::nnue::constants::{
    EMBED_DIM, GRAPH_INPUT_DIM, HIDDEN_DIM, INPUT_DIM, K, MLP_HIDDEN,
};
use pixelflow_ir::OpKind;

#[test]
fn gradient_buffer_zero_init() {
    let g = GradientBuffer::zero();
    assert!(g.d_w1.iter().all(|row| row.iter().all(|&v| v == 0.0)));
    assert!(g.d_b1.iter().all(|&v| v == 0.0));
    assert!(g.d_embeddings.iter().all(|row| row.iter().all(|&v| v == 0.0)));
}

#[test]
fn merge_gradients_accumulates() {
    let mut a = GradientBuffer::zero();
    let mut b = GradientBuffer::zero();
    a.d_value_mlp_b2 = 1.0;
    b.d_value_mlp_b2 = 2.0;
    let merged = GradientBuffer::merge(&a, &b);
    assert_eq!(merged.d_value_mlp_b2, 3.0);
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test -p pixelflow-search gradient_buffer_zero_init
```

Expected: `error[E0432]: unresolved import pixelflow_search::nnue::gradient`

- [ ] **Step 3: Create `gradient.rs` with `GradientBuffer`**

Create `pixelflow-search/src/nnue/gradient.rs`:

```rust
//! # Gradient Buffer for ExprNnue Backpropagation
//!
//! `GradientBuffer` mirrors the weight layout of `ExprNnue` exactly.
//! Each field `d_X` accumulates `dL/dX` for the corresponding weight `X`.
//!
//! Usage: compute backward_extraction + backward_saturation, merge, then sgd_step.

extern crate alloc;
use alloc::vec::Vec;

use pixelflow_ir::OpKind;

use crate::nnue::constants::{
    EMBED_DIM, GRAPH_INPUT_DIM, HIDDEN_DIM, INPUT_DIM, K, MLP_HIDDEN,
    RULE_CONCAT_DIM, RULE_FEATURE_DIM,
};
use crate::nnue::network::ExprNnue;

// ============================================================================
// GradientBuffer
// ============================================================================

/// Accumulated gradients for all ExprNnue weight tensors.
///
/// Each field `d_X` stores `Σ dL/dX` over a mini-batch.
/// After accumulation, call `sgd_step` to apply the update.
#[derive(Clone)]
pub struct GradientBuffer {
    // --- Edge tower (extraction head) ---
    /// dL/dw1: [INPUT_DIM × HIDDEN_DIM]
    pub d_w1: [[f32; HIDDEN_DIM]; INPUT_DIM],
    /// dL/db1: [HIDDEN_DIM]
    pub d_b1: [f32; HIDDEN_DIM],

    // --- Shared trunk (both heads) ---
    /// dL/d_trunk_w: [HIDDEN_DIM × HIDDEN_DIM]
    pub d_trunk_w: [[f32; HIDDEN_DIM]; HIDDEN_DIM],
    /// dL/d_trunk_b: [HIDDEN_DIM]
    pub d_trunk_b: [f32; HIDDEN_DIM],

    // --- Expr projection + value head ---
    /// dL/d_expr_proj_w: [HIDDEN_DIM × EMBED_DIM]
    pub d_expr_proj_w: [[f32; EMBED_DIM]; HIDDEN_DIM],
    /// dL/d_expr_proj_b: [EMBED_DIM]
    pub d_expr_proj_b: [f32; EMBED_DIM],
    /// dL/d_value_mlp_w1: [EMBED_DIM × MLP_HIDDEN]
    pub d_value_mlp_w1: [[f32; MLP_HIDDEN]; EMBED_DIM],
    /// dL/d_value_mlp_b1: [MLP_HIDDEN]
    pub d_value_mlp_b1: [f32; MLP_HIDDEN],
    /// dL/d_value_mlp_w2: [MLP_HIDDEN]
    pub d_value_mlp_w2: [f32; MLP_HIDDEN],
    /// dL/d_value_mlp_b2: scalar
    pub d_value_mlp_b2: f32,

    // --- Graph tower (saturation head) ---
    /// dL/d_graph_w1: [GRAPH_INPUT_DIM × HIDDEN_DIM]
    pub d_graph_w1: [[f32; HIDDEN_DIM]; GRAPH_INPUT_DIM],
    /// dL/d_graph_b1: [HIDDEN_DIM]
    pub d_graph_b1: [f32; HIDDEN_DIM],

    // --- Graph projection + mask head ---
    /// dL/d_graph_proj_w: [HIDDEN_DIM × EMBED_DIM]
    pub d_graph_proj_w: [[f32; EMBED_DIM]; HIDDEN_DIM],
    /// dL/d_graph_proj_b: [EMBED_DIM]
    pub d_graph_proj_b: [f32; EMBED_DIM],
    /// dL/d_mask_mlp_w1: [EMBED_DIM × MLP_HIDDEN]
    pub d_mask_mlp_w1: [[f32; MLP_HIDDEN]; EMBED_DIM],
    /// dL/d_mask_mlp_b1: [MLP_HIDDEN]
    pub d_mask_mlp_b1: [f32; MLP_HIDDEN],
    /// dL/d_mask_mlp_w2: [MLP_HIDDEN × EMBED_DIM]
    pub d_mask_mlp_w2: [[f32; EMBED_DIM]; MLP_HIDDEN],
    /// dL/d_mask_mlp_b2: [EMBED_DIM]
    pub d_mask_mlp_b2: [f32; EMBED_DIM],

    // --- Bilinear scoring ---
    /// dL/d_interaction: [EMBED_DIM × EMBED_DIM]
    pub d_interaction: [[f32; EMBED_DIM]; EMBED_DIM],
    /// dL/d_mask_bias_proj: [EMBED_DIM]
    pub d_mask_bias_proj: [f32; EMBED_DIM],

    // --- Op embeddings (shared by both towers via EdgeAccumulator) ---
    /// dL/d_embeddings: [OpKind::COUNT × K]
    pub d_embeddings: [[f32; K]; OpKind::COUNT],
}

impl GradientBuffer {
    /// Zero-initialized gradient buffer.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            d_w1: [[0.0; HIDDEN_DIM]; INPUT_DIM],
            d_b1: [0.0; HIDDEN_DIM],
            d_trunk_w: [[0.0; HIDDEN_DIM]; HIDDEN_DIM],
            d_trunk_b: [0.0; HIDDEN_DIM],
            d_expr_proj_w: [[0.0; EMBED_DIM]; HIDDEN_DIM],
            d_expr_proj_b: [0.0; EMBED_DIM],
            d_value_mlp_w1: [[0.0; MLP_HIDDEN]; EMBED_DIM],
            d_value_mlp_b1: [0.0; MLP_HIDDEN],
            d_value_mlp_w2: [0.0; MLP_HIDDEN],
            d_value_mlp_b2: 0.0,
            d_graph_w1: [[0.0; HIDDEN_DIM]; GRAPH_INPUT_DIM],
            d_graph_b1: [0.0; HIDDEN_DIM],
            d_graph_proj_w: [[0.0; EMBED_DIM]; HIDDEN_DIM],
            d_graph_proj_b: [0.0; EMBED_DIM],
            d_mask_mlp_w1: [[0.0; MLP_HIDDEN]; EMBED_DIM],
            d_mask_mlp_b1: [0.0; MLP_HIDDEN],
            d_mask_mlp_w2: [[0.0; EMBED_DIM]; MLP_HIDDEN],
            d_mask_mlp_b2: [0.0; EMBED_DIM],
            d_interaction: [[0.0; EMBED_DIM]; EMBED_DIM],
            d_mask_bias_proj: [0.0; EMBED_DIM],
            d_embeddings: [[0.0; K]; OpKind::COUNT],
        }
    }

    /// Element-wise sum: `result[i] = a[i] + b[i]` for every weight tensor.
    #[must_use]
    pub fn merge(a: &Self, b: &Self) -> Self {
        let mut out = a.clone();
        for i in 0..INPUT_DIM {
            for j in 0..HIDDEN_DIM { out.d_w1[i][j] += b.d_w1[i][j]; }
        }
        for j in 0..HIDDEN_DIM { out.d_b1[j] += b.d_b1[j]; }
        for i in 0..HIDDEN_DIM {
            for j in 0..HIDDEN_DIM { out.d_trunk_w[i][j] += b.d_trunk_w[i][j]; }
        }
        for j in 0..HIDDEN_DIM { out.d_trunk_b[j] += b.d_trunk_b[j]; }
        for i in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM { out.d_expr_proj_w[i][k] += b.d_expr_proj_w[i][k]; }
        }
        for k in 0..EMBED_DIM { out.d_expr_proj_b[k] += b.d_expr_proj_b[k]; }
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN { out.d_value_mlp_w1[i][j] += b.d_value_mlp_w1[i][j]; }
        }
        for j in 0..MLP_HIDDEN { out.d_value_mlp_b1[j] += b.d_value_mlp_b1[j]; }
        for j in 0..MLP_HIDDEN { out.d_value_mlp_w2[j] += b.d_value_mlp_w2[j]; }
        out.d_value_mlp_b2 += b.d_value_mlp_b2;
        for i in 0..GRAPH_INPUT_DIM {
            for j in 0..HIDDEN_DIM { out.d_graph_w1[i][j] += b.d_graph_w1[i][j]; }
        }
        for j in 0..HIDDEN_DIM { out.d_graph_b1[j] += b.d_graph_b1[j]; }
        for i in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM { out.d_graph_proj_w[i][k] += b.d_graph_proj_w[i][k]; }
        }
        for k in 0..EMBED_DIM { out.d_graph_proj_b[k] += b.d_graph_proj_b[k]; }
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN { out.d_mask_mlp_w1[i][j] += b.d_mask_mlp_w1[i][j]; }
        }
        for j in 0..MLP_HIDDEN { out.d_mask_mlp_b1[j] += b.d_mask_mlp_b1[j]; }
        for i in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM { out.d_mask_mlp_w2[i][k] += b.d_mask_mlp_w2[i][k]; }
        }
        for k in 0..EMBED_DIM { out.d_mask_mlp_b2[k] += b.d_mask_mlp_b2[k]; }
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM { out.d_interaction[i][j] += b.d_interaction[i][j]; }
        }
        for k in 0..EMBED_DIM { out.d_mask_bias_proj[k] += b.d_mask_bias_proj[k]; }
        for op in 0..OpKind::COUNT {
            for k in 0..K { out.d_embeddings[op][k] += b.d_embeddings[op][k]; }
        }
        out
    }

    /// SGD update: `weights -= lr * gradients`.
    ///
    /// Applies to all weight tensors in `model`.
    pub fn sgd_step(&self, model: &mut ExprNnue, lr: f32) {
        for i in 0..INPUT_DIM {
            for j in 0..HIDDEN_DIM { model.w1[i][j] -= lr * self.d_w1[i][j]; }
        }
        for j in 0..HIDDEN_DIM { model.b1[j] -= lr * self.d_b1[j]; }
        for i in 0..HIDDEN_DIM {
            for j in 0..HIDDEN_DIM { model.trunk_w[i][j] -= lr * self.d_trunk_w[i][j]; }
        }
        for j in 0..HIDDEN_DIM { model.trunk_b[j] -= lr * self.d_trunk_b[j]; }
        for i in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM { model.expr_proj_w[i][k] -= lr * self.d_expr_proj_w[i][k]; }
        }
        for k in 0..EMBED_DIM { model.expr_proj_b[k] -= lr * self.d_expr_proj_b[k]; }
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN { model.value_mlp_w1[i][j] -= lr * self.d_value_mlp_w1[i][j]; }
        }
        for j in 0..MLP_HIDDEN { model.value_mlp_b1[j] -= lr * self.d_value_mlp_b1[j]; }
        for j in 0..MLP_HIDDEN { model.value_mlp_w2[j] -= lr * self.d_value_mlp_w2[j]; }
        model.value_mlp_b2 -= lr * self.d_value_mlp_b2;
        for i in 0..GRAPH_INPUT_DIM {
            for j in 0..HIDDEN_DIM { model.graph_w1[i][j] -= lr * self.d_graph_w1[i][j]; }
        }
        for j in 0..HIDDEN_DIM { model.graph_b1[j] -= lr * self.d_graph_b1[j]; }
        for i in 0..HIDDEN_DIM {
            for k in 0..EMBED_DIM { model.graph_proj_w[i][k] -= lr * self.d_graph_proj_w[i][k]; }
        }
        for k in 0..EMBED_DIM { model.graph_proj_b[k] -= lr * self.d_graph_proj_b[k]; }
        for i in 0..EMBED_DIM {
            for j in 0..MLP_HIDDEN { model.mask_mlp_w1[i][j] -= lr * self.d_mask_mlp_w1[i][j]; }
        }
        for j in 0..MLP_HIDDEN { model.mask_mlp_b1[j] -= lr * self.d_mask_mlp_b1[j]; }
        for i in 0..MLP_HIDDEN {
            for k in 0..EMBED_DIM { model.mask_mlp_w2[i][k] -= lr * self.d_mask_mlp_w2[i][k]; }
        }
        for k in 0..EMBED_DIM { model.mask_mlp_b2[k] -= lr * self.d_mask_mlp_b2[k]; }
        for i in 0..EMBED_DIM {
            for j in 0..EMBED_DIM { model.interaction[i][j] -= lr * self.d_interaction[i][j]; }
        }
        for k in 0..EMBED_DIM { model.mask_bias_proj[k] -= lr * self.d_mask_bias_proj[k]; }
        for op in 0..OpKind::COUNT {
            for k in 0..K { model.embeddings.e[op][k] -= lr * self.d_embeddings[op][k]; }
        }
    }
}
```

- [ ] **Step 4: Register module in `mod.rs`**

In `pixelflow-search/src/nnue/mod.rs`, add:

```rust
pub mod gradient;
pub use gradient::{GradientBuffer};
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p pixelflow-search gradient_buffer_zero_init
cargo test -p pixelflow-search merge_gradients_accumulates
```

Expected: both pass.

- [ ] **Step 6: Commit**

```bash
git add pixelflow-search/src/nnue/gradient.rs \
        pixelflow-search/src/nnue/mod.rs \
        pixelflow-search/tests/nnue_gradient.rs
git commit -m "feat(search): nnue/gradient.rs — GradientBuffer + sgd_step"
```

---

## Task 2: Backward pass functions in `gradient.rs`

**Files:** Modify `pixelflow-search/src/nnue/gradient.rs`

The backward pass mirrors `forward.rs` layer-by-layer. Each `dense_relu` layer has:
- Backward through ReLU: `d_pre_relu[j] = d_out[j] * (pre_relu[j] > 0 ? 1 : 0)`
- Backward through affine: `d_weights[i][j] += input[i] * d_pre_relu[j]`, `d_input[i] += w[i][j] * d_pre_relu[j]`

- [ ] **Step 1: Write failing tests**

Add to `pixelflow-search/tests/nnue_gradient.rs`:

```rust
use pixelflow_search::nnue::accumulators::{EdgeAccumulator, GraphAccumulator};
use pixelflow_search::nnue::forward::{build_acc_input_expr_only, build_acc_input_shared, build_graph_input};
use pixelflow_search::nnue::gradient::{backward_extraction, backward_saturation};
use pixelflow_search::nnue::network::ExprNnue;
use pixelflow_search::nnue::constants::EMBED_DIM;

fn make_test_acc() -> EdgeAccumulator {
    let mut acc = EdgeAccumulator::new();
    acc.node_count = 5; acc.edge_count = 8; acc.node_budget = 100; acc.epoch_budget = 10;
    for i in 0..acc.values.len() { acc.values[i] = (i as f32) * 0.01; }
    acc
}

fn make_test_gacc() -> GraphAccumulator {
    let mut gacc = GraphAccumulator::new();
    gacc.node_count = 5; gacc.edge_count = 8; gacc.node_budget = 100; gacc.epoch_budget = 10;
    for i in 0..gacc.values.len() { gacc.values[i] = (i as f32) * 0.01; }
    gacc
}

/// Finite-difference gradient check for value_mlp_b2.
#[test]
fn backward_extraction_gradient_check_b2() {
    let net = ExprNnue::new_random(7);
    let acc = make_test_acc();
    let acc_input = build_acc_input_expr_only(&acc);
    
    // Forward pass to get base cost
    let eps = 1e-4;
    
    // Perturb b2 up
    let mut net_plus = net.clone();
    net_plus.value_mlp_b2 += eps;
    let cost_plus = {
        use pixelflow_search::nnue::forward::{edge_tower_expr_only, expr_proj, value_mlp};
        let h = edge_tower_expr_only(&net_plus, &acc_input);
        let e = expr_proj(&net_plus, &h);
        value_mlp(&net_plus, &e)
    };
    
    // Perturb b2 down
    let mut net_minus = net.clone();
    net_minus.value_mlp_b2 -= eps;
    let cost_minus = {
        use pixelflow_search::nnue::forward::{edge_tower_expr_only, expr_proj, value_mlp};
        let h = edge_tower_expr_only(&net_minus, &acc_input);
        let e = expr_proj(&net_minus, &h);
        value_mlp(&net_minus, &e)
    };
    
    let fd_grad_b2 = (cost_plus - cost_minus) / (2.0 * eps);
    
    // Analytical gradient (d_loss = 1.0 for a single sample with target=cost)
    // Loss = 0.5 * (cost - target)^2, d_loss/d_cost = cost - target = 0 when target=cost.
    // Instead test with d_value_pred = 1.0 (unit upstream gradient)
    let grads = backward_extraction(&net, &acc_input, 1.0);
    
    assert!(
        (grads.d_value_mlp_b2 - fd_grad_b2).abs() < 1e-3,
        "d_b2 mismatch: analytical={}, fd={}",
        grads.d_value_mlp_b2, fd_grad_b2
    );
}

#[test]
fn backward_saturation_returns_correct_shape() {
    let net = ExprNnue::new_random(99);
    let gacc = make_test_gacc();
    let graph_input = build_graph_input(&gacc);
    let rule_embed = [0.1f32; EMBED_DIM];
    let grads = backward_saturation(&net, &graph_input, &rule_embed, 1.0);
    // Just verify the buffer sizes are non-trivially non-zero after backward
    assert!(grads.d_mask_bias_proj.iter().any(|&v| v != 0.0));
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test -p pixelflow-search backward_extraction_gradient_check_b2
```

Expected: `error[E0432]: no function backward_extraction`

- [ ] **Step 3: Implement `backward_extraction` and `backward_saturation`**

Add to `pixelflow-search/src/nnue/gradient.rs`:

```rust
use crate::nnue::forward::{
    build_acc_input_expr_only, build_acc_input_shared, build_graph_input,
    edge_tower_expr_only, edge_tower_shared, expr_proj, graph_proj, graph_tower,
    mask_mlp, value_mlp,
};

// ============================================================================
// Backward pass helpers
// ============================================================================

/// Backward through ReLU: gate the upstream gradient by the sign of the pre-activation.
fn relu_backward(d_out: &[f32], pre_relu: &[f32]) -> Vec<f32> {
    debug_assert_eq!(d_out.len(), pre_relu.len());
    d_out.iter().zip(pre_relu.iter())
        .map(|(&d, &p)| if p > 0.0 { d } else { 0.0 })
        .collect()
}

/// Backward through an affine layer:
/// - `d_weights[i][j] += input[i] * d_out[j]`
/// - `d_bias[j] += d_out[j]`
/// - `d_input[i] += Σⱼ weights[i][j] * d_out[j]`
///
/// Returns `d_input` as `Vec<f32>`. Accumulates into `d_weights` and `d_bias` in place.
fn affine_backward(
    input: &[f32],     // shape: [input_dim]
    weights_flat: &[f32], // shape: [input_dim * output_dim], C-order: w[i][j] at i*output+j
    d_out: &[f32],     // shape: [output_dim]
    output_dim: usize,
    d_weights_flat: &mut Vec<f32>, // accumulate into this (same C-order layout)
    d_bias: &mut Vec<f32>,         // accumulate into this
) -> Vec<f32> {
    let input_dim = input.len();
    debug_assert_eq!(weights_flat.len(), input_dim * output_dim);
    debug_assert_eq!(d_out.len(), output_dim);

    if d_weights_flat.is_empty() {
        d_weights_flat.resize(input_dim * output_dim, 0.0);
    }
    if d_bias.is_empty() {
        d_bias.resize(output_dim, 0.0);
    }

    let mut d_input = vec![0.0f32; input_dim];
    for i in 0..input_dim {
        for j in 0..output_dim {
            let w_ij = weights_flat[i * output_dim + j];
            d_weights_flat[i * output_dim + j] += input[i] * d_out[j];
            d_input[i] += w_ij * d_out[j];
        }
    }
    for j in 0..output_dim {
        d_bias[j] += d_out[j];
    }
    d_input
}

// ============================================================================
// Backward pass: extraction head
// ============================================================================

/// Backward pass through the extraction head.
///
/// Input `d_value_pred` is `dL/d_value_pred` (e.g. `2 * (pred - target)` for MSE).
///
/// Computes gradients for:
/// - value MLP weights
/// - expr_proj weights
/// - trunk weights (edge path)
/// - w1 weights
/// - d_acc_input (for propagating to embeddings)
#[must_use]
pub fn backward_extraction(
    net: &ExprNnue,
    acc_input: &[f32; INPUT_DIM],
    d_value_pred: f32,
) -> GradientBuffer {
    let mut grads = GradientBuffer::zero();
    
    // ---- Forward cache ----
    let input_dm = pixelflow_core::lattice::DiscreteManifold::new(acc_input.to_vec(), INPUT_DIM, 1);
    let w1_flat = net.w1_flat();
    let w1_dm = crate::nnue::forward::weights_to_manifold_pub(&w1_flat, INPUT_DIM, HIDDEN_DIM);
    let pre_relu_1: Vec<f32> = {
        use crate::nnue::forward::affine_pub;
        affine_pub(&input_dm, &w1_dm, &net.b1).into_buffer()
    };
    let post_relu_1: Vec<f32> = pre_relu_1.iter().map(|&x| x.max(0.0)).collect();
    
    let post_relu_1_dm = pixelflow_core::lattice::DiscreteManifold::new(post_relu_1.clone(), HIDDEN_DIM, 1);
    let trunk_flat = net.trunk_w_flat();
    let trunk_dm = crate::nnue::forward::weights_to_manifold_pub(&trunk_flat, HIDDEN_DIM, HIDDEN_DIM);
    let pre_relu_trunk: Vec<f32> = {
        use crate::nnue::forward::affine_pub;
        affine_pub(&post_relu_1_dm, &trunk_dm, &net.trunk_b).into_buffer()
    };
    let hidden: Vec<f32> = pre_relu_trunk.iter().map(|&x| x.max(0.0)).collect();
    
    let hidden_dm = pixelflow_core::lattice::DiscreteManifold::new(hidden.clone(), HIDDEN_DIM, 1);
    let proj_flat = net.expr_proj_w_flat();
    let proj_dm = crate::nnue::forward::weights_to_manifold_pub(&proj_flat, HIDDEN_DIM, EMBED_DIM);
    let expr_embed: Vec<f32> = {
        use crate::nnue::forward::affine_pub;
        affine_pub(&hidden_dm, &proj_dm, &net.expr_proj_b).into_buffer()
    };
    
    let embed_dm = pixelflow_core::lattice::DiscreteManifold::new(expr_embed.clone(), EMBED_DIM, 1);
    let vm_w1_flat = net.value_mlp_w1_flat();
    let vm_w1_dm = crate::nnue::forward::weights_to_manifold_pub(&vm_w1_flat, EMBED_DIM, MLP_HIDDEN);
    let pre_relu_vm: Vec<f32> = {
        use crate::nnue::forward::affine_pub;
        affine_pub(&embed_dm, &vm_w1_dm, &net.value_mlp_b1).into_buffer()
    };
    let value_h: Vec<f32> = pre_relu_vm.iter().map(|&x| x.max(0.0)).collect();
    
    // ---- Backward: value MLP layer 2 ----
    // value_pred = Σⱼ value_h[j] * value_mlp_w2[j] + value_mlp_b2
    // d_value_h[j] = d_value_pred * value_mlp_w2[j]
    // d_value_mlp_w2[j] = d_value_pred * value_h[j]
    // d_value_mlp_b2 = d_value_pred
    for j in 0..MLP_HIDDEN {
        grads.d_value_mlp_w2[j] += d_value_pred * value_h[j];
    }
    grads.d_value_mlp_b2 += d_value_pred;
    let d_value_h: Vec<f32> = net.value_mlp_w2.iter().map(|&w| d_value_pred * w).collect();
    
    // ---- Backward: value MLP layer 1 (ReLU + affine) ----
    let d_pre_relu_vm = relu_backward(&d_value_h, &pre_relu_vm);
    let mut d_vm_w1_flat = Vec::new();
    let mut d_vm_b1 = Vec::new();
    let d_expr_embed = affine_backward(&expr_embed, &vm_w1_flat, &d_pre_relu_vm, MLP_HIDDEN, &mut d_vm_w1_flat, &mut d_vm_b1);
    for i in 0..EMBED_DIM {
        for j in 0..MLP_HIDDEN {
            grads.d_value_mlp_w1[i][j] += d_vm_w1_flat[i * MLP_HIDDEN + j];
        }
    }
    for j in 0..MLP_HIDDEN { grads.d_value_mlp_b1[j] += d_vm_b1[j]; }
    
    // ---- Backward: expr_proj (affine, no activation) ----
    let mut d_proj_flat = Vec::new();
    let mut d_proj_b = Vec::new();
    let d_hidden = affine_backward(&hidden, &proj_flat, &d_expr_embed, EMBED_DIM, &mut d_proj_flat, &mut d_proj_b);
    for i in 0..HIDDEN_DIM {
        for k in 0..EMBED_DIM {
            grads.d_expr_proj_w[i][k] += d_proj_flat[i * EMBED_DIM + k];
        }
    }
    for k in 0..EMBED_DIM { grads.d_expr_proj_b[k] += d_proj_b[k]; }
    
    // ---- Backward: trunk (ReLU + affine) ----
    let d_pre_relu_trunk = relu_backward(&d_hidden, &pre_relu_trunk);
    let mut d_trunk_flat = Vec::new();
    let mut d_trunk_b_vec = Vec::new();
    let d_post_relu_1 = affine_backward(&post_relu_1, &trunk_flat, &d_pre_relu_trunk, HIDDEN_DIM, &mut d_trunk_flat, &mut d_trunk_b_vec);
    for i in 0..HIDDEN_DIM {
        for j in 0..HIDDEN_DIM {
            grads.d_trunk_w[i][j] += d_trunk_flat[i * HIDDEN_DIM + j];
        }
    }
    for j in 0..HIDDEN_DIM { grads.d_trunk_b[j] += d_trunk_b_vec[j]; }
    
    // ---- Backward: w1 (ReLU + affine) ----
    let d_pre_relu_1 = relu_backward(&d_post_relu_1, &pre_relu_1);
    let mut d_w1_flat = Vec::new();
    let mut d_b1_vec = Vec::new();
    let _d_acc_input = affine_backward(acc_input, &w1_flat, &d_pre_relu_1, HIDDEN_DIM, &mut d_w1_flat, &mut d_b1_vec);
    for i in 0..INPUT_DIM {
        for j in 0..HIDDEN_DIM {
            grads.d_w1[i][j] += d_w1_flat[i * HIDDEN_DIM + j];
        }
    }
    for j in 0..HIDDEN_DIM { grads.d_b1[j] += d_b1_vec[j]; }
    // _d_acc_input would propagate to embeddings via EdgeAccumulator backward.
    // This is the IR pullback integration point (see Team 2).
    
    grads
}

/// Backward pass through the saturation head.
///
/// Input `d_score` is `dL/d_score` (e.g. REINFORCE advantage signal).
///
/// Computes gradients for:
/// - bilinear interaction + mask_bias_proj
/// - mask_mlp weights
/// - graph_proj weights
/// - trunk weights (graph path, accumulated into same d_trunk_w)
/// - graph_w1 weights
#[must_use]
pub fn backward_saturation(
    net: &ExprNnue,
    graph_input: &[f32; GRAPH_INPUT_DIM],
    rule_embed: &[f32; EMBED_DIM],
    d_score: f32,
) -> GradientBuffer {
    let mut grads = GradientBuffer::zero();
    
    // ---- Forward cache ----
    let input_dm = pixelflow_core::lattice::DiscreteManifold::new(graph_input.to_vec(), GRAPH_INPUT_DIM, 1);
    let gw1_flat = net.graph_w1_flat();
    let gw1_dm = crate::nnue::forward::weights_to_manifold_pub(&gw1_flat, GRAPH_INPUT_DIM, HIDDEN_DIM);
    let pre_relu_g1: Vec<f32> = {
        use crate::nnue::forward::affine_pub;
        affine_pub(&input_dm, &gw1_dm, &net.graph_b1).into_buffer()
    };
    let post_relu_g1: Vec<f32> = pre_relu_g1.iter().map(|&x| x.max(0.0)).collect();
    
    let post_relu_g1_dm = pixelflow_core::lattice::DiscreteManifold::new(post_relu_g1.clone(), HIDDEN_DIM, 1);
    let trunk_flat = net.trunk_w_flat();
    let trunk_dm = crate::nnue::forward::weights_to_manifold_pub(&trunk_flat, HIDDEN_DIM, HIDDEN_DIM);
    let pre_relu_trunk: Vec<f32> = {
        use crate::nnue::forward::affine_pub;
        affine_pub(&post_relu_g1_dm, &trunk_dm, &net.trunk_b).into_buffer()
    };
    let graph_hidden: Vec<f32> = pre_relu_trunk.iter().map(|&x| x.max(0.0)).collect();
    
    let ghidden_dm = pixelflow_core::lattice::DiscreteManifold::new(graph_hidden.clone(), HIDDEN_DIM, 1);
    let gproj_flat = net.graph_proj_w_flat();
    let gproj_dm = crate::nnue::forward::weights_to_manifold_pub(&gproj_flat, HIDDEN_DIM, EMBED_DIM);
    let graph_embed: Vec<f32> = {
        use crate::nnue::forward::affine_pub;
        affine_pub(&ghidden_dm, &gproj_dm, &net.graph_proj_b).into_buffer()
    };
    
    let gembed_dm = pixelflow_core::lattice::DiscreteManifold::new(graph_embed.clone(), EMBED_DIM, 1);
    let mm_w1_flat = net.mask_mlp_w1_flat();
    let mm_w1_dm = crate::nnue::forward::weights_to_manifold_pub(&mm_w1_flat, EMBED_DIM, MLP_HIDDEN);
    let pre_relu_mm: Vec<f32> = {
        use crate::nnue::forward::affine_pub;
        affine_pub(&gembed_dm, &mm_w1_dm, &net.mask_mlp_b1).into_buffer()
    };
    let mask_h: Vec<f32> = pre_relu_mm.iter().map(|&x| x.max(0.0)).collect();
    
    let mm_h_dm = pixelflow_core::lattice::DiscreteManifold::new(mask_h.clone(), MLP_HIDDEN, 1);
    let mm_w2_flat = net.mask_mlp_w2_flat();
    let mm_w2_dm = crate::nnue::forward::weights_to_manifold_pub(&mm_w2_flat, MLP_HIDDEN, EMBED_DIM);
    let mask_features: Vec<f32> = {
        use crate::nnue::forward::affine_pub;
        affine_pub(&mm_h_dm, &mm_w2_dm, &net.mask_mlp_b2).into_buffer()
    };
    
    // transformed[k] = Σᵢ mask_features[i] * interaction[i][k]
    let mut transformed = [0.0f32; EMBED_DIM];
    for i in 0..EMBED_DIM {
        for k in 0..EMBED_DIM {
            transformed[k] += mask_features[i] * net.interaction[i][k];
        }
    }
    
    // score = Σₖ transformed[k] * rule_embed[k] + Σₖ mask_bias_proj[k] * rule_embed[k]
    
    // ---- Backward: bilinear score ----
    // score = Σₖ (transformed[k] + mask_bias_proj[k]) * rule_embed[k]
    // d_transformed[k] = d_score * rule_embed[k]
    // d_mask_bias_proj[k] = d_score * rule_embed[k]
    let d_transformed: Vec<f32> = rule_embed.iter().map(|&re| d_score * re).collect();
    for k in 0..EMBED_DIM {
        grads.d_mask_bias_proj[k] += d_score * rule_embed[k];
    }
    
    // ---- Backward: interaction matrix ----
    // transformed[k] = Σᵢ mask_features[i] * interaction[i][k]
    // d_interaction[i][k] = d_transformed[k] * mask_features[i]
    // d_mask_features[i] = Σₖ d_transformed[k] * interaction[i][k]
    let mut d_mask_features = vec![0.0f32; EMBED_DIM];
    for i in 0..EMBED_DIM {
        for k in 0..EMBED_DIM {
            grads.d_interaction[i][k] += d_transformed[k] * mask_features[i];
            d_mask_features[i] += d_transformed[k] * net.interaction[i][k];
        }
    }
    
    // ---- Backward: mask_mlp layer 2 (affine, no activation) ----
    let mut d_mm_w2_flat = Vec::new();
    let mut d_mm_b2_vec = Vec::new();
    let d_mask_h = affine_backward(&mask_h, &mm_w2_flat, &d_mask_features, EMBED_DIM, &mut d_mm_w2_flat, &mut d_mm_b2_vec);
    for i in 0..MLP_HIDDEN {
        for k in 0..EMBED_DIM { grads.d_mask_mlp_w2[i][k] += d_mm_w2_flat[i * EMBED_DIM + k]; }
    }
    for k in 0..EMBED_DIM { grads.d_mask_mlp_b2[k] += d_mm_b2_vec[k]; }
    
    // ---- Backward: mask_mlp layer 1 (ReLU + affine) ----
    let d_pre_relu_mm = relu_backward(&d_mask_h, &pre_relu_mm);
    let mut d_mm_w1_flat = Vec::new();
    let mut d_mm_b1_vec = Vec::new();
    let d_graph_embed = affine_backward(&graph_embed, &mm_w1_flat, &d_pre_relu_mm, MLP_HIDDEN, &mut d_mm_w1_flat, &mut d_mm_b1_vec);
    for i in 0..EMBED_DIM {
        for j in 0..MLP_HIDDEN { grads.d_mask_mlp_w1[i][j] += d_mm_w1_flat[i * MLP_HIDDEN + j]; }
    }
    for j in 0..MLP_HIDDEN { grads.d_mask_mlp_b1[j] += d_mm_b1_vec[j]; }
    
    // ---- Backward: graph_proj (affine, no activation) ----
    let mut d_gproj_flat = Vec::new();
    let mut d_gproj_b_vec = Vec::new();
    let d_graph_hidden = affine_backward(&graph_hidden, &gproj_flat, &d_graph_embed, EMBED_DIM, &mut d_gproj_flat, &mut d_gproj_b_vec);
    for i in 0..HIDDEN_DIM {
        for k in 0..EMBED_DIM { grads.d_graph_proj_w[i][k] += d_gproj_flat[i * EMBED_DIM + k]; }
    }
    for k in 0..EMBED_DIM { grads.d_graph_proj_b[k] += d_gproj_b_vec[k]; }
    
    // ---- Backward: trunk (ReLU + affine) ----
    let d_pre_relu_trunk = relu_backward(&d_graph_hidden, &pre_relu_trunk);
    let mut d_trunk_flat = Vec::new();
    let mut d_trunk_b_vec = Vec::new();
    let d_post_relu_g1 = affine_backward(&post_relu_g1, &trunk_flat, &d_pre_relu_trunk, HIDDEN_DIM, &mut d_trunk_flat, &mut d_trunk_b_vec);
    for i in 0..HIDDEN_DIM {
        for j in 0..HIDDEN_DIM { grads.d_trunk_w[i][j] += d_trunk_flat[i * HIDDEN_DIM + j]; }
    }
    for j in 0..HIDDEN_DIM { grads.d_trunk_b[j] += d_trunk_b_vec[j]; }
    
    // ---- Backward: graph_w1 (ReLU + affine) ----
    let d_pre_relu_g1 = relu_backward(&d_post_relu_g1, &pre_relu_g1);
    let mut d_gw1_flat = Vec::new();
    let mut d_gb1_vec = Vec::new();
    let _d_graph_input = affine_backward(graph_input, &gw1_flat, &d_pre_relu_g1, HIDDEN_DIM, &mut d_gw1_flat, &mut d_gb1_vec);
    for i in 0..GRAPH_INPUT_DIM {
        for j in 0..HIDDEN_DIM { grads.d_graph_w1[i][j] += d_gw1_flat[i * HIDDEN_DIM + j]; }
    }
    for j in 0..HIDDEN_DIM { grads.d_graph_b1[j] += d_gb1_vec[j]; }
    
    grads
}
```

- [ ] **Step 4: Add `pub(crate)` helpers to `forward.rs` needed by `gradient.rs`**

In `forward.rs`, make `weights_to_manifold` and `affine` accessible as `pub(crate)`:

```rust
/// Public-crate access for backward pass.
pub(crate) fn weights_to_manifold_pub(flat: &[f32], input_dim: usize, output_dim: usize) -> DiscreteManifold {
    weights_to_manifold(flat, input_dim, output_dim)
}

/// Public-crate access for backward pass.
pub(crate) fn affine_pub(
    input: &DiscreteManifold,
    weights: &DiscreteManifold,
    bias: &[f32],
) -> DiscreteManifold {
    affine(input, weights, bias)
}
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p pixelflow-search backward_extraction_gradient_check_b2
cargo test -p pixelflow-search backward_saturation_returns_correct_shape
```

Expected: both pass. If finite-difference check fails by more than 1e-3, verify `affine_backward` index math: for weight `w[i][j]` stored as `flat[i * output_dim + j]`, `d_flat[i * output_dim + j] += input[i] * d_out[j]`.

- [ ] **Step 6: Commit**

```bash
git add pixelflow-search/src/nnue/gradient.rs \
        pixelflow-search/src/nnue/forward.rs
git commit -m "feat(search): backward_extraction and backward_saturation in gradient.rs"
```

---

## Task 3: `trajectory.rs` and `io.rs` in `pixelflow-pipeline`

**Files:** Create `pixelflow-pipeline/src/training/trajectory.rs` and `pixelflow-pipeline/src/training/io.rs`

- [ ] **Step 1: Write the failing test**

Create `pixelflow-pipeline/tests/training_io.rs`:

```rust
use pixelflow_pipeline::training::trajectory::{Trajectory, TrajectoryStep};
use pixelflow_pipeline::training::io::{write_trajectories_jsonl, read_trajectories_jsonl};
use std::path::Path;

#[test]
fn roundtrip_jsonl() {
    let traj = Trajectory {
        trajectory_id: "test-001".to_string(),
        steps: vec![TrajectoryStep {
            accumulator_state: vec![0.0f32; 10],
            expression_embedding: vec![0.0f32; 32],
            rule_embedding: vec![0.0f32; 32],
            budget_remaining: 50,
            epochs_remaining: 5,
            action_probability: 0.8,
            matched: true,
            jit_cost_ns: 123.0,
            edges: vec![],
            graph_accumulator_state: vec![0.0f32; 132],
        }],
        initial_cost_ns: 200.0,
        final_cost_ns: 123.0,
        initial_cost: None,
        final_cost: None,
    };
    
    let tmp = std::env::temp_dir().join("test_roundtrip.jsonl");
    write_trajectories_jsonl(&[traj.clone()], &tmp).unwrap();
    let loaded = read_trajectories_jsonl(&tmp).unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].trajectory_id, "test-001");
    assert_eq!(loaded[0].steps.len(), 1);
    assert!((loaded[0].final_cost_ns - 123.0).abs() < 1e-6);
    std::fs::remove_file(tmp).ok();
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test -p pixelflow-pipeline roundtrip_jsonl
```

Expected: `error[E0432]: unresolved import pixelflow_pipeline::training::trajectory`

- [ ] **Step 3: Create `trajectory.rs`**

Create `pixelflow-pipeline/src/training/trajectory.rs` by moving `Trajectory`, `TrajectoryStep`, and `TrajectoryAdvantages` from `unified.rs` verbatim, plus the accumulator-to-vec helpers from `self_play.rs`:

```rust
//! Trajectory types: the IPC boundary between Rust (Actor) and Python (Critic).

use serde::{Deserialize, Serialize};

/// Re-export for external callers.
pub use crate::training::unified::{Trajectory, TrajectoryAdvantages, TrajectoryStep};
```

Wait — `Trajectory` and friends already exist in `unified.rs`. Move them: copy the struct definitions into `trajectory.rs` and replace `unified.rs` with re-exports. This is safe because `unified.rs` isn't a public crate boundary type yet.

**Alternative (simpler):** Just re-export from `unified.rs` to make `trajectory.rs` the canonical location, then in a follow-up remove `unified.rs`. For the plan:

In `trajectory.rs`:
```rust
//! Trajectory types for self-play → critic pipeline.

pub use crate::training::unified::{Trajectory, TrajectoryAdvantages, TrajectoryStep};
```

In `mod.rs`:
```rust
pub mod trajectory;
```

- [ ] **Step 4: Create `io.rs`**

Create `pixelflow-pipeline/src/training/io.rs` by extracting `write_trajectories_jsonl` and `read_advantages_jsonl` from `self_play.rs`:

```rust
//! JSONL serialization for trajectory and advantage data.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::training::trajectory::{Trajectory, TrajectoryAdvantages};

/// Write trajectories as JSONL (one JSON object per line).
pub fn write_trajectories_jsonl(
    trajectories: &[Trajectory],
    path: &Path,
) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for t in trajectories {
        let json = serde_json::to_string(t)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        writeln!(writer, "{}", json)?;
    }
    Ok(())
}

/// Read trajectories from JSONL.
pub fn read_trajectories_jsonl(path: &Path) -> std::io::Result<Vec<Trajectory>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let t: Trajectory = serde_json::from_str(&line)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        out.push(t);
    }
    Ok(out)
}

/// Read advantages from JSONL written by the Python critic.
pub fn read_advantages_jsonl(path: &Path) -> std::io::Result<Vec<TrajectoryAdvantages>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let a: TrajectoryAdvantages = serde_json::from_str(&line)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        out.push(a);
    }
    Ok(out)
}
```

In `mod.rs`, add:
```rust
pub mod io;
pub mod trajectory;
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p pixelflow-pipeline roundtrip_jsonl
```

Expected: passes.

- [ ] **Step 6: Commit**

```bash
git add pixelflow-pipeline/src/training/trajectory.rs \
        pixelflow-pipeline/src/training/io.rs \
        pixelflow-pipeline/src/training/mod.rs \
        pixelflow-pipeline/tests/training_io.rs
git commit -m "feat(pipeline): training/trajectory.rs + io.rs — typed boundary for JSONL pipeline"
```

---

## Task 4: `update.rs` — loss + gradient accumulation

**Files:** Create `pixelflow-pipeline/src/training/update.rs`

- [ ] **Step 1: Write the failing test**

Add to `pixelflow-pipeline/tests/training_io.rs`:

```rust
use pixelflow_pipeline::training::update::{apply_update, TrainingConfig};
use pixelflow_search::nnue::network::ExprNnue;

#[test]
fn apply_update_reduces_value_loss() {
    let mut model = ExprNnue::new_random(1);
    let config = TrainingConfig {
        lr_value: 1e-3,
        lr_policy: 1e-4,
        grad_clip: 1.0,
        target_cost: 100.0,
    };
    
    // Fake trajectory: one step, cost=200ns, target=100ns (50% reduction needed)
    let acc = pixelflow_search::nnue::accumulators::EdgeAccumulator::new();
    let gacc = pixelflow_search::nnue::accumulators::GraphAccumulator::new();
    let rule_embed = [0.1f32; pixelflow_search::nnue::constants::EMBED_DIM];
    
    use pixelflow_search::nnue::forward::{build_acc_input_expr_only, edge_tower_expr_only, expr_proj, value_mlp};
    let acc_input = build_acc_input_expr_only(&acc);
    let h = edge_tower_expr_only(&model, &acc_input);
    let e = expr_proj(&model, &h);
    let cost_before = value_mlp(&model, &e);
    
    // Apply one update step: MSE loss against target_cost
    apply_update(&mut model, &acc_input, &gacc, &rule_embed, 0.0, config.target_cost, &config);
    
    let h2 = edge_tower_expr_only(&model, &acc_input);
    let e2 = expr_proj(&model, &h2);
    let cost_after = value_mlp(&model, &e2);
    
    // After one gradient step, cost prediction should move toward target (loss should decrease)
    let loss_before = (cost_before - config.target_cost).powi(2);
    let loss_after = (cost_after - config.target_cost).powi(2);
    assert!(loss_after < loss_before,
        "update did not reduce loss: before={}, after={}", loss_before, loss_after);
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test -p pixelflow-pipeline apply_update_reduces_value_loss
```

Expected: `error[E0432]: unresolved import pixelflow_pipeline::training::update`

- [ ] **Step 3: Create `update.rs`**

Create `pixelflow-pipeline/src/training/update.rs`:

```rust
//! Loss computation and gradient accumulation.
//!
//! This module contains NO tensor math beyond calling `backward_extraction`,
//! `backward_saturation`, and `sgd_step` from `pixelflow-search`.
//! The math lives in `pixelflow-search/src/nnue/gradient.rs`.

use pixelflow_search::nnue::accumulators::{EdgeAccumulator, GraphAccumulator};
use pixelflow_search::nnue::constants::{EMBED_DIM, INPUT_DIM, GRAPH_INPUT_DIM};
use pixelflow_search::nnue::forward::{
    build_acc_input_expr_only, build_acc_input_shared, build_graph_input,
};
use pixelflow_search::nnue::gradient::{
    GradientBuffer, backward_extraction, backward_saturation,
};
use pixelflow_search::nnue::network::ExprNnue;

/// Training hyperparameters for a single update step.
pub struct TrainingConfig {
    /// Learning rate for the extraction head (value prediction).
    pub lr_value: f32,
    /// Learning rate for the saturation head (policy).
    pub lr_policy: f32,
    /// Per-parameter gradient clip threshold.
    pub grad_clip: f32,
    /// MSE target cost (ground-truth log-nanoseconds for extraction head).
    pub target_cost: f32,
}

/// Apply one gradient update using extraction loss + saturation REINFORCE.
///
/// `acc_input` is the pre-built `[f32; INPUT_DIM]` (from `build_acc_input_expr_only`).
/// `graph_input` is the pre-built `[f32; GRAPH_INPUT_DIM]` (from `build_graph_input`).
/// `rule_embed` is the rule embedding for the selected action.
/// `advantage` is the per-step advantage from the Critic (clamped to [-5, 5]).
/// `target_cost` is the ground-truth execution cost (log-ns) for the extraction head.
pub fn apply_update(
    model: &mut ExprNnue,
    acc_input: &[f32; INPUT_DIM],
    gacc: &GraphAccumulator,
    rule_embed: &[f32; EMBED_DIM],
    advantage: f32,
    target_cost: f32,
    config: &TrainingConfig,
) {
    // ---- Extraction head: MSE loss ----
    // Forward to get current prediction
    use pixelflow_search::nnue::forward::{edge_tower_expr_only, expr_proj, value_mlp};
    let h = edge_tower_expr_only(model, acc_input);
    let e = expr_proj(model, &h);
    let pred = value_mlp(model, &e);
    // d_loss/d_pred = 2 * (pred - target) for MSE
    let d_value_pred = 2.0 * (pred - target_cost);
    
    let mut grads_extraction = backward_extraction(model, acc_input, d_value_pred);
    clip_gradients(&mut grads_extraction, config.grad_clip);
    grads_extraction.sgd_step(model, config.lr_value);
    
    // ---- Saturation head: REINFORCE ----
    // Clamp advantage to prevent gradient explosion
    let advantage_clamped = advantage.clamp(-5.0, 5.0);
    let graph_input = build_graph_input(gacc);
    // d_score = -advantage (policy gradient: maximize expected reward)
    // sign convention: positive advantage → increase score → positive gradient
    let d_score = advantage_clamped;
    
    let mut grads_saturation = backward_saturation(model, &graph_input, rule_embed, d_score);
    clip_gradients(&mut grads_saturation, config.grad_clip);
    grads_saturation.sgd_step(model, config.lr_policy);
}

/// Clip all gradients in place to `[-clip, clip]`.
fn clip_gradients(grads: &mut GradientBuffer, clip: f32) {
    use pixelflow_search::nnue::constants::{GRAPH_INPUT_DIM, HIDDEN_DIM, INPUT_DIM, K, MLP_HIDDEN};
    use pixelflow_ir::OpKind;
    
    for row in grads.d_w1.iter_mut() {
        for v in row.iter_mut() { *v = v.clamp(-clip, clip); }
    }
    for v in grads.d_b1.iter_mut() { *v = v.clamp(-clip, clip); }
    for row in grads.d_trunk_w.iter_mut() {
        for v in row.iter_mut() { *v = v.clamp(-clip, clip); }
    }
    // ... (repeat for all fields)
    // For brevity, iterate over d_embeddings separately:
    for row in grads.d_embeddings.iter_mut() {
        for v in row.iter_mut() { *v = v.clamp(-clip, clip); }
    }
}
```

- [ ] **Step 4: Register in `mod.rs`**

```rust
pub mod update;
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p pixelflow-pipeline apply_update_reduces_value_loss
```

Expected: passes.

- [ ] **Step 6: Commit**

```bash
git add pixelflow-pipeline/src/training/update.rs \
        pixelflow-pipeline/src/training/mod.rs
git commit -m "feat(pipeline): training/update.rs — loss + gradient accumulation, delegates to gradient.rs"
```

---

## Task 5: Refactor `train_unified.rs` and delete `unified_backward.rs`

**Files:** Shrink `train_unified.rs` to < 200 lines, delete `unified_backward.rs`

- [ ] **Step 1: Verify the 4-phase structure by reading the current main loop**

```bash
grep -n "GENERATE\|EXPORT\|CRITIQUE\|UPDATE\|CHECKPOINT" \
  pixelflow-pipeline/src/bin/train_unified.rs | head -30
```

Note the exact function calls used at each phase — these become the new one-line calls in the refactored binary.

- [ ] **Step 2: Write the test first (compile-only)**

Add `pixelflow-pipeline/tests/train_unified_compiles.rs`:

```rust
// Smoke test: the binary imports should compile cleanly.
// If train_unified.rs still imports from unified_backward, this fails.
#[test]
fn no_math_in_pipeline_binary() {
    // All math should be behind pixelflow_search::nnue::gradient APIs.
    // This test just ensures the training module compiles without unified_backward.
    use pixelflow_pipeline::training::update::apply_update;
    use pixelflow_pipeline::training::io::{write_trajectories_jsonl, read_advantages_jsonl};
    let _ = std::mem::size_of::<pixelflow_pipeline::training::trajectory::Trajectory>();
}
```

- [ ] **Step 3: Delete `unified_backward.rs`**

```bash
rm pixelflow-pipeline/src/training/unified_backward.rs
```

Remove the `pub mod unified_backward;` line from `pixelflow-pipeline/src/training/mod.rs`.

- [ ] **Step 4: Refactor `train_unified.rs`**

Replace the 2,660-line `train_unified.rs` with the orchestration-only version below. Read the current file first to capture CLI arg names, checkpoint paths, and critic URL. Then write:

```rust
//! # Unified Self-Play Training
//!
//! ```text
//! GENERATE → EXPORT → CRITIQUE → UPDATE → CHECKPOINT
//! ```

use std::path::PathBuf;
use clap::Parser;

use pixelflow_pipeline::training::io::{read_advantages_jsonl, write_trajectories_jsonl};
use pixelflow_pipeline::training::self_play::{
    build_rule_templates, generate_trajectory_batch_parallel, load_corpus_exprs,
};
use pixelflow_pipeline::training::update::{apply_update, TrainingConfig};
use pixelflow_search::nnue::network::ExprNnue;
use pixelflow_search::nnue::accumulators::EdgeAccumulator;
use pixelflow_search::nnue::forward::{build_acc_input_expr_only, build_graph_input};

#[derive(Parser, Debug)]
#[command(about = "Unified self-play training loop")]
struct Args {
    #[arg(long, default_value = "30")]
    rounds: usize,
    #[arg(long, default_value = "50")]
    trajectories_per_round: usize,
    #[arg(long, default_value = "output/trajectories.jsonl")]
    traj_path: PathBuf,
    #[arg(long, default_value = "output/advantages.jsonl")]
    advantage_path: PathBuf,
    #[arg(long, default_value = "output/model.bin")]
    checkpoint_path: PathBuf,
    #[arg(long, default_value = "http://localhost:8765")]
    critic_url: String,
    #[arg(long, default_value = "1e-3")]
    lr_value: f32,
    #[arg(long, default_value = "1e-4")]
    lr_policy: f32,
    #[arg(long, default_value = "1.0")]
    grad_clip: f32,
    #[arg(long)]
    corpus: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();
    let mut model = load_or_init_model(&args.checkpoint_path);
    let corpus = args.corpus.as_ref().map(|p| load_corpus_exprs(p)).unwrap_or_default();
    let rules = build_rule_templates();
    let config = TrainingConfig {
        lr_value: args.lr_value,
        lr_policy: args.lr_policy,
        grad_clip: args.grad_clip,
        target_cost: 0.0, // set per trajectory
    };

    for round in 0..args.rounds {
        eprintln!("[round {}] GENERATE", round);
        let trajectories = generate_trajectory_batch_parallel(
            &model, &rules, &corpus, args.trajectories_per_round,
        );

        eprintln!("[round {}] EXPORT {} trajectories", round, trajectories.len());
        write_trajectories_jsonl(&trajectories, &args.traj_path)
            .expect("failed to write trajectories");

        eprintln!("[round {}] CRITIQUE", round);
        run_critic(&args.critic_url, &args.traj_path, &args.advantage_path)
            .expect("critic failed");

        eprintln!("[round {}] UPDATE", round);
        let advantages = read_advantages_jsonl(&args.advantage_path)
            .expect("failed to read advantages");
        apply_batch_update(&mut model, &trajectories, &advantages, &config);

        eprintln!("[round {}] CHECKPOINT", round);
        save_model(&model, &args.checkpoint_path, round)
            .expect("failed to save checkpoint");
    }
}

// ---- Helpers (each < 30 lines) ----

fn load_or_init_model(path: &std::path::Path) -> ExprNnue {
    if path.exists() {
        load_model(path).unwrap_or_else(|e| {
            eprintln!("failed to load model: {}; initializing fresh", e);
            ExprNnue::new_random(42)
        })
    } else {
        ExprNnue::new_random(42)
    }
}

fn load_model(path: &std::path::Path) -> std::io::Result<ExprNnue> {
    let data = std::fs::read(path)?;
    pixelflow_pipeline::checkpoint::load(&data)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

fn save_model(model: &ExprNnue, path: &std::path::Path, round: usize) -> std::io::Result<()> {
    std::fs::create_dir_all(path.parent().unwrap_or(std::path::Path::new(".")))?;
    let data = pixelflow_pipeline::checkpoint::save(model)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, data)?;
    eprintln!("  saved round {} checkpoint to {}", round, path.display());
    Ok(())
}

fn run_critic(
    url: &str,
    traj_path: &std::path::Path,
    adv_path: &std::path::Path,
) -> std::io::Result<()> {
    // HTTP POST to critic server (unchanged from original implementation)
    pixelflow_pipeline::critic::run_http(url, traj_path, adv_path)
}

fn apply_batch_update(
    model: &mut ExprNnue,
    trajectories: &[pixelflow_pipeline::training::trajectory::Trajectory],
    advantages: &[pixelflow_pipeline::training::trajectory::TrajectoryAdvantages],
    config: &TrainingConfig,
) {
    use pixelflow_pipeline::training::update::apply_update;
    use pixelflow_search::nnue::accumulators::GraphAccumulator;
    use pixelflow_search::nnue::constants::EMBED_DIM;
    use pixelflow_search::nnue::forward::{build_acc_input_expr_only, build_graph_input};
    
    let adv_map: std::collections::HashMap<&str, &pixelflow_pipeline::training::trajectory::TrajectoryAdvantages> =
        advantages.iter().map(|a| (a.trajectory_id.as_str(), a)).collect();
    
    for traj in trajectories {
        let Some(adv) = adv_map.get(traj.trajectory_id.as_str()) else { continue; };
        let target_log_cost = (traj.final_cost_ns + 1.0).ln();
        let step_config = TrainingConfig {
            target_cost: target_log_cost,
            ..*config
        };
        
        for (step, step_adv) in traj.steps.iter().zip(adv.advantages.iter()) {
            if step.graph_accumulator_state.len() < EMBED_DIM { continue; }
            
            let acc_input = extract_acc_input(step);
            let gacc = extract_graph_acc(step);
            let rule_embed: [f32; EMBED_DIM] = step.rule_embedding
                .iter().copied().take(EMBED_DIM)
                .chain(std::iter::repeat(0.0))
                .take(EMBED_DIM)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            
            apply_update(model, &acc_input, &gacc, &rule_embed, *step_adv, target_log_cost, &step_config);
        }
    }
}

fn extract_acc_input(
    step: &pixelflow_pipeline::training::trajectory::TrajectoryStep,
) -> [f32; pixelflow_search::nnue::constants::INPUT_DIM] {
    use pixelflow_search::nnue::constants::INPUT_DIM;
    let mut arr = [0.0f32; INPUT_DIM];
    for (i, &v) in step.accumulator_state.iter().take(INPUT_DIM).enumerate() {
        arr[i] = v;
    }
    arr
}

fn extract_graph_acc(
    step: &pixelflow_pipeline::training::trajectory::TrajectoryStep,
) -> pixelflow_search::nnue::accumulators::GraphAccumulator {
    use pixelflow_search::nnue::accumulators::GraphAccumulator;
    use pixelflow_search::nnue::constants::GRAPH_INPUT_DIM;
    let mut gacc = GraphAccumulator::new();
    for (i, &v) in step.graph_accumulator_state.iter().take(gacc.values.len()).enumerate() {
        gacc.values[i] = v;
    }
    gacc
}
```

- [ ] **Step 5: Verify line count**

```bash
wc -l pixelflow-pipeline/src/bin/train_unified.rs
```

Expected: < 200 lines.

- [ ] **Step 6: Full workspace build and test**

```bash
cargo build --workspace
cargo test --workspace
```

Expected: all tests pass. `unified_backward.rs` no longer exists.

- [ ] **Step 7: Verify deletion**

```bash
ls pixelflow-pipeline/src/training/unified_backward.rs
```

Expected: `No such file or directory`

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor(pipeline): delete unified_backward.rs, train_unified.rs → orchestration-only (<200 lines)"
```

---

## Verification

```bash
# unified_backward.rs is gone
test ! -f pixelflow-pipeline/src/training/unified_backward.rs && echo "OK: deleted"

# train_unified.rs is under 200 lines
awk 'END{if(NR<200) print "OK: " NR " lines"; else print "FAIL: " NR " lines"}' \
  pixelflow-pipeline/src/bin/train_unified.rs

# No math in pixelflow-pipeline (no raw tensor loops)
grep -n "for.*\.\.\." pixelflow-pipeline/src/bin/train_unified.rs
# Expected: 0 matches (no tensor loops in main binary)

# No EMBED_DIM=24 anywhere in pipeline
grep -r "EMBED_DIM.*=.*24\|const EMBED_DIM" pixelflow-pipeline/src/
# Expected: no matches

# Full test suite
cargo test --workspace
```

End-to-end smoke test (run 3 rounds with tiny corpus):
```bash
cargo run --release -p pixelflow-pipeline --bin train_unified -- \
  --rounds 3 --trajectories-per-round 5
# Verify loss decreases across rounds in the output log
```
