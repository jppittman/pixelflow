# Team 3: Forward Pass Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `factored.rs` (5,418 lines) into focused modules and rewrite the NNUE forward pass as manifold compositions over `IndexLattice2D`, replacing nested loops with lattice collapse.

**Architecture:** Constants → Accumulators → Network → Forward pass expressed via `IndexLattice2D::collapse_axis`. The `ExprNnue` struct retains its raw weight arrays (backward-compatible); `forward.rs` wraps them as `DiscreteManifold` views and calls `collapse_axis(0, Add, &MatmulProduct)` for each matmul. Results are numerically identical to the existing `forward_shared`/`forward_graph` methods.

**Tech Stack:** Rust stable, `pixelflow-search`, `pixelflow-core` (requires Team 1: `IndexLattice2D`, `collapse_axis`, `DiscreteManifold::new`).

**Dependency:** Tasks 1–3 and 5 can start immediately. Task 4 (forward.rs) requires Team 1 to have shipped `IndexLattice2D` and `collapse_axis`.

---

## Context You Need

Read before starting:
- `pixelflow-search/src/nnue/factored.rs` — the file being split (5,418 lines)
- `pixelflow-search/src/nnue/mod.rs` — current re-exports; will be updated
- `pixelflow-pipeline/src/training/replay.rs` — has `const EMBED_DIM: usize = 24` (the bug to fix)
- `pixelflow-core/src/lattice.rs` — `IndexLattice2D`, `collapse_axis`, `DiscreteManifold` (Team 1 output)

Key constants in `factored.rs` (canonical, must be preserved exactly):
```
K = 32, INPUT_DIM = 132, GRAPH_INPUT_DIM = 132, HIDDEN_DIM = 64,
EMBED_DIM = 32 (NOT 24), MLP_HIDDEN = 16, RULE_FEATURE_DIM = 8,
MASK_MAX_RULES = 1024, RULE_CONCAT_DIM = 128, MASK_INPUT_DIM = 32,
SCALAR_FEATURE_COUNT = 4, GRAPH_ACC_DIM = 128, MAX_ARITY = 3, MAX_DEPTH = 192
```

Network data flow (two parallel towers sharing a trunk):
```
EdgeAccumulator[INPUT_DIM=132]  → w1[132×64]   → ReLU → trunk[64×64] → ReLU → expr_proj[64×32] → value_mlp → cost
GraphAccumulator[GRAPH_INPUT_DIM=132] → graph_w1[132×64] → ReLU → trunk[64×64] → ReLU → graph_proj[64×32] → mask_mlp → bilinear → rule_score
                                                              ^^^^^ SAME trunk weights
```

`DiscreteManifold` layout (row-major): `buffer[y * width + x]`. For weight matrix W with `eval(xi, yj) = W[i][j]` (i=input, j=output), set `buffer[j * input_dim + i] = W[i][j]` (transpose from C-order storage).

---

## File Structure

| File | Action | What it holds |
|------|--------|---------------|
| `pixelflow-search/src/nnue/constants.rs` | Create | All `const` definitions |
| `pixelflow-search/src/nnue/accumulators.rs` | Create | `EdgeAccumulator`, `GraphAccumulator`, `OpEmbeddings` + all methods |
| `pixelflow-search/src/nnue/network.rs` | Create | `ExprNnue`, `RuleFeatures`, `RuleTemplates`, `ArenaRuleTemplate`, `ArenaRuleTemplates` + flat weight accessors |
| `pixelflow-search/src/nnue/forward.rs` | Create | Forward pass as manifold compositions |
| `pixelflow-search/src/nnue/factored.rs` | Shrink | Keep only `ExprGenerator`, `BwdGenerator`; remove everything else |
| `pixelflow-search/src/nnue/mod.rs` | Update | Delete HalfEP section; add `pub mod` for new files; update re-exports |
| `pixelflow-pipeline/src/training/replay.rs` | Fix | Remove local `const EMBED_DIM: usize = 24;`, import from `pixelflow_search` |

---

## Task 1: Create `constants.rs`

**Files:** Create `pixelflow-search/src/nnue/constants.rs`, modify `pixelflow-search/src/nnue/mod.rs`

- [ ] **Step 1: Write the failing test**

Add to `pixelflow-search/tests/nnue_constants.rs` (create the file):

```rust
#[test]
fn embed_dim_is_32() {
    use pixelflow_search::nnue::EMBED_DIM;
    assert_eq!(EMBED_DIM, 32, "EMBED_DIM must be 32; 24 was the old wrong value");
}

#[test]
fn rule_concat_dim_is_four_embed() {
    use pixelflow_search::nnue::{EMBED_DIM, RULE_CONCAT_DIM};
    assert_eq!(RULE_CONCAT_DIM, 4 * EMBED_DIM);
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test -p pixelflow-search embed_dim_is_32
```

Expected: `error[E0432]: unresolved import` (constants.rs doesn't exist yet)

- [ ] **Step 3: Create `constants.rs`**

Create `pixelflow-search/src/nnue/constants.rs`:

```rust
//! # NNUE Architecture Constants
//!
//! Single source of truth for all dimension constants. Every other module
//! imports from here — never define these elsewhere.
//!
//! ## Network dimensions
//!
//! ```text
//! Input towers: INPUT_DIM=132 (edge) | GRAPH_INPUT_DIM=132 (graph)
//! First layer:  HIDDEN_DIM=64
//! Shared trunk: HIDDEN_DIM×HIDDEN_DIM
//! Projection:   EMBED_DIM=32
//! Value MLP:    EMBED_DIM → MLP_HIDDEN(16) → 1
//! Mask MLP:     EMBED_DIM → MLP_HIDDEN(16) → EMBED_DIM
//! ```

/// Embedding dimension per operation in the edge accumulator.
pub const K: usize = 32;

/// Number of scalar features appended to the dual accumulator.
/// edge_count, node_count, node_budget, epoch_budget.
pub const SCALAR_FEATURE_COUNT: usize = 4;

/// Total input dimension to the first hidden layer: 4K + 4 scalars = 132.
pub const INPUT_DIM: usize = 4 * K + SCALAR_FEATURE_COUNT;

/// Graph backbone accumulator dimension: 4K sections × K dims = 128.
pub const GRAPH_ACC_DIM: usize = 4 * K;

/// Graph backbone input: 4K + 4 scalars = 132.
pub const GRAPH_INPUT_DIM: usize = GRAPH_ACC_DIM + SCALAR_FEATURE_COUNT;

/// First hidden layer size (both towers).
pub const HIDDEN_DIM: usize = 64;

/// Shared expression/graph embedding dimension.
///
/// **32, not 24.** Historical code had 24 — all stale references must be updated.
pub const EMBED_DIM: usize = 32;

/// Hidden size of the private MLP heads (value, mask, rule).
pub const MLP_HIDDEN: usize = 16;

/// Rule hand-crafted feature dimension.
pub const RULE_FEATURE_DIM: usize = 8;

/// Maximum rules in the unified mask architecture.
pub const MASK_MAX_RULES: usize = 1024;

/// Rule projection input: 4 × EMBED_DIM = 128.
pub const RULE_CONCAT_DIM: usize = 4 * EMBED_DIM;

/// Mask MLP input dimension: expr_embed directly (EMBED_DIM = 32).
pub const MASK_INPUT_DIM: usize = EMBED_DIM;

/// Maximum child-index-encoded arity for depth embeddings.
pub const MAX_ARITY: usize = 3;

/// Maximum effective depth in the depth-position encoding.
pub const MAX_DEPTH: usize = 192;
```

- [ ] **Step 4: Add `pub mod constants` to `mod.rs`**

In `pixelflow-search/src/nnue/mod.rs`, add at the top (after existing module attributes):

```rust
pub mod constants;
pub use constants::{
    EMBED_DIM, GRAPH_ACC_DIM, GRAPH_INPUT_DIM, HIDDEN_DIM, INPUT_DIM, K,
    MASK_INPUT_DIM, MASK_MAX_RULES, MAX_ARITY, MAX_DEPTH, MLP_HIDDEN,
    RULE_CONCAT_DIM, RULE_FEATURE_DIM, SCALAR_FEATURE_COUNT,
};
```

- [ ] **Step 5: Update `factored.rs` — remove duplicate constant definitions**

In `factored.rs`, delete the entire `// ============================================================================\n// Constants\n// ============================================================================` block (lines 50–109) and replace with:

```rust
// Constants live in constants.rs — import them here.
use crate::nnue::constants::{
    EMBED_DIM, GRAPH_ACC_DIM, GRAPH_INPUT_DIM, HIDDEN_DIM, INPUT_DIM, K,
    MASK_INPUT_DIM, MASK_MAX_RULES, MAX_ARITY, MAX_DEPTH, MLP_HIDDEN,
    RULE_CONCAT_DIM, RULE_FEATURE_DIM, SCALAR_FEATURE_COUNT,
};
```

- [ ] **Step 6: Run tests**

```bash
cargo test -p pixelflow-search embed_dim_is_32
cargo test -p pixelflow-search rule_concat_dim
```

Expected: both pass.

- [ ] **Step 7: Commit**

```bash
git add pixelflow-search/src/nnue/constants.rs \
        pixelflow-search/src/nnue/mod.rs \
        pixelflow-search/src/nnue/factored.rs \
        pixelflow-search/tests/nnue_constants.rs
git commit -m "feat(search): extract nnue/constants.rs — single source of truth, EMBED_DIM=32"
```

---

## Task 2: Create `accumulators.rs`

**Files:** Create `pixelflow-search/src/nnue/accumulators.rs`, update `factored.rs` and `mod.rs`

- [ ] **Step 1: Write the failing test**

Add to `pixelflow-search/tests/nnue_accumulators.rs` (create the file):

```rust
use pixelflow_search::nnue::accumulators::EdgeAccumulator;

#[test]
fn edge_accumulator_default_is_zero() {
    let acc = EdgeAccumulator::new();
    assert!(acc.values.iter().all(|&v| v == 0.0));
    assert_eq!(acc.edge_count, 0);
    assert_eq!(acc.node_count, 0);
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test -p pixelflow-search edge_accumulator_default_is_zero
```

Expected: `error[E0432]: unresolved import pixelflow_search::nnue::accumulators`

- [ ] **Step 3: Create `accumulators.rs`**

Move the following blocks from `factored.rs` verbatim into a new file `pixelflow-search/src/nnue/accumulators.rs`:

- The `OpEmbeddings` struct and its `impl` block (starting at line 450)
- The `EdgeAccumulator` struct and its `impl` block (starting at line 755)
- The `GraphAccumulator` struct and its `impl` block (starting at line 1355)

Add at the top of `accumulators.rs`:

```rust
//! Edge and graph accumulators for NNUE input encoding.

extern crate alloc;

use alloc::vec::Vec;
use libm::sqrtf;

use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};
use crate::egraph::Pattern as Expr;
use pixelflow_ir::OpKind;

use crate::nnue::constants::{
    GRAPH_ACC_DIM, GRAPH_INPUT_DIM, K, MAX_ARITY, MAX_DEPTH, SCALAR_FEATURE_COUNT,
};
```

Remove the `use` imports from the top of each moved struct's block (they're already covered by the module-level imports above).

- [ ] **Step 4: Add `pub mod accumulators` to `mod.rs` and update re-exports**

In `mod.rs`, add:

```rust
pub mod accumulators;
pub use accumulators::{EdgeAccumulator, GraphAccumulator, OpEmbeddings};
```

- [ ] **Step 5: Replace moved code in `factored.rs` with imports**

In `factored.rs`, delete the moved blocks and add at the top:

```rust
use crate::nnue::accumulators::{EdgeAccumulator, GraphAccumulator, OpEmbeddings};
```

- [ ] **Step 6: Run tests**

```bash
cargo test -p pixelflow-search edge_accumulator_default_is_zero
cargo test -p pixelflow-search  # full suite
```

Expected: all existing tests still pass. New test passes.

- [ ] **Step 7: Commit**

```bash
git add pixelflow-search/src/nnue/accumulators.rs \
        pixelflow-search/src/nnue/mod.rs \
        pixelflow-search/src/nnue/factored.rs \
        pixelflow-search/tests/nnue_accumulators.rs
git commit -m "feat(search): extract nnue/accumulators.rs — EdgeAccumulator, GraphAccumulator, OpEmbeddings"
```

---

## Task 3: Create `network.rs`

**Files:** Create `pixelflow-search/src/nnue/network.rs`, update `factored.rs` and `mod.rs`

- [ ] **Step 1: Write the failing test**

Add to `pixelflow-search/tests/nnue_network.rs` (create the file):

```rust
use pixelflow_search::nnue::network::ExprNnue;
use pixelflow_search::nnue::constants::{EMBED_DIM, HIDDEN_DIM, INPUT_DIM};

#[test]
fn expr_nnue_zero_init() {
    let net = ExprNnue::new();
    assert!(net.w1.iter().all(|row| row.iter().all(|&v| v == 0.0)));
    assert!(net.b1.iter().all(|&v| v == 0.0));
}

#[test]
fn w1_flat_has_correct_len() {
    let net = ExprNnue::new();
    assert_eq!(net.w1_flat().len(), INPUT_DIM * HIDDEN_DIM);
}

#[test]
fn expr_proj_flat_has_correct_len() {
    let net = ExprNnue::new();
    assert_eq!(net.expr_proj_w_flat().len(), HIDDEN_DIM * EMBED_DIM);
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test -p pixelflow-search expr_nnue_zero_init
```

Expected: `error[E0432]: unresolved import pixelflow_search::nnue::network`

- [ ] **Step 3: Create `network.rs`**

Move the following blocks from `factored.rs` verbatim into a new file `pixelflow-search/src/nnue/network.rs`:

- `RuleFeatures` struct and impl (line ~133)
- `RuleTemplates` struct and impl (line ~208)
- `ArenaRuleTemplate` struct and impl (line ~337)
- `ArenaRuleTemplates` struct and impl (line ~384)
- `ExprNnue` struct and ALL its impl blocks (starting at line 1829)

Add at the top of `network.rs`:

```rust
//! ExprNnue: the dual-head NNUE network struct and rule template types.

extern crate alloc;

use alloc::vec::Vec;
use libm::sqrtf;

use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};
use pixelflow_ir::OpKind;
use crate::egraph::Pattern as Expr;

use crate::nnue::constants::{
    EMBED_DIM, GRAPH_ACC_DIM, GRAPH_INPUT_DIM, HIDDEN_DIM, INPUT_DIM, K,
    MASK_INPUT_DIM, MASK_MAX_RULES, MLP_HIDDEN, RULE_CONCAT_DIM, RULE_FEATURE_DIM,
};
use crate::nnue::accumulators::{EdgeAccumulator, GraphAccumulator, OpEmbeddings};
```

- [ ] **Step 4: Add flat weight accessors to `ExprNnue` in `network.rs`**

After the existing `impl ExprNnue` block in `network.rs`, add a new impl block with flat slice accessors used by `forward.rs`:

```rust
impl ExprNnue {
    /// `w1[i][j]` as flat C-order slice: `flat[i * HIDDEN_DIM + j] = w1[i][j]`.
    pub fn w1_flat(&self) -> Vec<f32> {
        self.w1.iter().flat_map(|row| row.iter().copied()).collect()
    }

    /// `trunk_w[i][j]` as flat C-order slice.
    pub fn trunk_w_flat(&self) -> Vec<f32> {
        self.trunk_w.iter().flat_map(|row| row.iter().copied()).collect()
    }

    /// `expr_proj_w[i][j]` as flat C-order slice: `flat[i * EMBED_DIM + j]`.
    pub fn expr_proj_w_flat(&self) -> Vec<f32> {
        self.expr_proj_w.iter().flat_map(|row| row.iter().copied()).collect()
    }

    /// `value_mlp_w1[i][j]` as flat C-order slice.
    pub fn value_mlp_w1_flat(&self) -> Vec<f32> {
        self.value_mlp_w1.iter().flat_map(|row| row.iter().copied()).collect()
    }

    /// `graph_w1[i][j]` as flat C-order slice.
    pub fn graph_w1_flat(&self) -> Vec<f32> {
        self.graph_w1.iter().flat_map(|row| row.iter().copied()).collect()
    }

    /// `graph_proj_w[i][j]` as flat C-order slice.
    pub fn graph_proj_w_flat(&self) -> Vec<f32> {
        self.graph_proj_w.iter().flat_map(|row| row.iter().copied()).collect()
    }

    /// `mask_mlp_w1[i][j]` as flat C-order slice.
    pub fn mask_mlp_w1_flat(&self) -> Vec<f32> {
        self.mask_mlp_w1.iter().flat_map(|row| row.iter().copied()).collect()
    }

    /// `mask_mlp_w2[i][j]` as flat C-order slice.
    pub fn mask_mlp_w2_flat(&self) -> Vec<f32> {
        self.mask_mlp_w2.iter().flat_map(|row| row.iter().copied()).collect()
    }
}
```

- [ ] **Step 5: Update `factored.rs` and `mod.rs`**

In `factored.rs`, delete the moved blocks and add imports:

```rust
use crate::nnue::network::{
    ArenaRuleTemplates, ExprNnue, RuleFeatures, RuleTemplates,
};
```

In `mod.rs`, add:

```rust
pub mod network;
pub use network::{
    ArenaRuleTemplates, ExprNnue, RuleFeatures, RuleTemplates,
};
```

- [ ] **Step 6: Run tests**

```bash
cargo test -p pixelflow-search expr_nnue_zero_init
cargo test -p pixelflow-search w1_flat_has_correct_len
cargo test -p pixelflow-search  # full suite
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add pixelflow-search/src/nnue/network.rs \
        pixelflow-search/src/nnue/mod.rs \
        pixelflow-search/src/nnue/factored.rs \
        pixelflow-search/tests/nnue_network.rs
git commit -m "feat(search): extract nnue/network.rs — ExprNnue struct + flat weight accessors"
```

---

## Task 4: Create `forward.rs` (depends on Team 1)

**Prerequisite:** `pixelflow-core` must have `IndexLattice2D`, `IndexLattice2D::collapse_axis`, and `DiscreteManifold::new` (Team 1 output).

**Files:** Create `pixelflow-search/src/nnue/forward.rs`, update `mod.rs`

- [ ] **Step 1: Write the golden test first**

Create `pixelflow-search/tests/nnue_forward_golden.rs`:

```rust
//! Golden test: new manifold forward pass must match the reference implementation.

use pixelflow_search::nnue::accumulators::{EdgeAccumulator, GraphAccumulator};
use pixelflow_search::nnue::network::ExprNnue;
use pixelflow_search::nnue::forward::{
    build_acc_input_shared, build_acc_input_expr_only, build_graph_input,
    edge_tower_shared, edge_tower_expr_only, graph_tower, expr_proj, graph_proj,
    value_mlp, mask_mlp,
};
use pixelflow_search::nnue::constants::{EMBED_DIM, HIDDEN_DIM};

fn make_test_acc() -> EdgeAccumulator {
    let mut acc = EdgeAccumulator::new();
    acc.node_count = 5;
    acc.edge_count = 8;
    acc.node_budget = 100;
    acc.epoch_budget = 10;
    for i in 0..acc.values.len() {
        acc.values[i] = (i as f32) * 0.01;
    }
    acc
}

fn make_test_gacc() -> GraphAccumulator {
    let mut gacc = GraphAccumulator::new();
    gacc.node_count = 5;
    gacc.edge_count = 8;
    gacc.node_budget = 100;
    gacc.epoch_budget = 10;
    for i in 0..gacc.values.len() {
        gacc.values[i] = (i as f32) * 0.01;
    }
    gacc
}

#[test]
fn edge_tower_shared_matches_reference() {
    let net = ExprNnue::new_random(42);
    let acc = make_test_acc();

    let ref_hidden = net.forward_shared(&acc);
    let acc_input = build_acc_input_shared(&acc);
    let new_hidden_dm = edge_tower_shared(&net, &acc_input);
    let new_hidden = new_hidden_dm.buffer();

    for j in 0..HIDDEN_DIM {
        assert!(
            (ref_hidden[j] - new_hidden[j]).abs() < 1e-4,
            "hidden[{}] mismatch: ref={}, new={}",
            j, ref_hidden[j], new_hidden[j]
        );
    }
}

#[test]
fn value_cost_matches_reference() {
    let net = ExprNnue::new_random(42);
    let acc = make_test_acc();

    let ref_hidden = net.forward_expr_only(&acc);
    let ref_embed = net.compute_expr_embed(&ref_hidden);
    let ref_cost = net.value_mlp_forward(&ref_embed);

    let acc_input = build_acc_input_expr_only(&acc);
    let hidden_dm = edge_tower_expr_only(&net, &acc_input);
    let embed_dm = expr_proj(&net, &hidden_dm);
    let new_cost = value_mlp(&net, &embed_dm);

    assert!(
        (ref_cost - new_cost).abs() < 1e-4,
        "value cost mismatch: ref={}, new={}",
        ref_cost, new_cost
    );
}

#[test]
fn graph_embed_matches_reference() {
    let net = ExprNnue::new_random(42);
    let gacc = make_test_gacc();

    let ref_hidden = net.forward_graph(&gacc);
    let ref_embed = net.compute_graph_embed(&ref_hidden);

    let graph_input = build_graph_input(&gacc);
    let hidden_dm = graph_tower(&net, &graph_input);
    let embed_dm = graph_proj(&net, &hidden_dm);

    for k in 0..EMBED_DIM {
        assert!(
            (ref_embed[k] - embed_dm.buffer()[k]).abs() < 1e-4,
            "graph_embed[{}] mismatch: ref={}, new={}",
            k, ref_embed[k], embed_dm.buffer()[k]
        );
    }
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test -p pixelflow-search nnue_forward_golden
```

Expected: `error[E0432]: unresolved import pixelflow_search::nnue::forward`

- [ ] **Step 3: Create `forward.rs` — manifold primitives**

Create `pixelflow-search/src/nnue/forward.rs`:

```rust
//! # NNUE Forward Pass as Manifold Compositions
//!
//! Each dense layer is a lattice collapse: `output[j] = bias[j] + Σᵢ W[i,j] * input[i]`
//! expressed as `IndexLattice2D(input_dim, output_dim).collapse_axis(0, Add, &MatmulProduct)`.
//!
//! This is the same pull-based composition used for graphics rendering.
//! A matmul IS a collapse; multi-layer forward IS multi-pass lattice collapse.

extern crate alloc;
use alloc::vec::Vec;

use pixelflow_core::{Field, ReduceOp};
use pixelflow_core::lattice::{DiscreteManifold, IndexLattice2D};
use pixelflow_core::Manifold;

use libm::{log2f, sqrtf};

use crate::nnue::constants::{
    EMBED_DIM, GRAPH_INPUT_DIM, HIDDEN_DIM, INPUT_DIM, K, MLP_HIDDEN,
    SCALAR_FEATURE_COUNT,
};
use crate::nnue::accumulators::{EdgeAccumulator, GraphAccumulator};
use crate::nnue::network::ExprNnue;

// ============================================================================
// Core manifold primitive
// ============================================================================

/// Manifold for a single dense-layer product: `W[i,j] * input[i]`.
///
/// Used inside `collapse_axis(0, Add, &DenseProduct)` to sum over input dim i.
struct DenseProduct<'a> {
    /// Weight matrix. `eval(xi, yj)` = W[input_i, output_j].
    /// Layout: `DiscreteManifold{width=input_dim, height=output_dim}`,
    /// `buffer[j * input_dim + i] = W[i][j]`.
    weights: &'a DiscreteManifold,
    /// Input vector. `eval(xi, 0)` = input[i].
    /// Layout: `DiscreteManifold{width=input_dim, height=1}`.
    input: &'a DiscreteManifold,
}

impl<'a> Manifold<(Field, Field, Field, Field)> for DenseProduct<'a> {
    type Output = Field;

    fn eval(&self, (xi, yj, _, _): (Field, Field, Field, Field)) -> Field {
        let zero = Field::from(0.0);
        self.weights.eval((xi, yj, zero, zero))
            * self.input.eval((xi, zero, zero, zero))
    }
}

// ============================================================================
// Helper: convert flat weight arrays to DiscreteManifold
// ============================================================================

/// Convert C-order flat weights `flat[i * output_dim + j] = W[i][j]`
/// into `DiscreteManifold{width=input_dim, height=output_dim}` where
/// `eval(xi=i, yj=j) = W[i][j]`, stored as `buffer[j * input_dim + i]`.
///
/// This transpose makes accessing all inputs for a fixed output index (j)
/// contiguous in memory, which is the access pattern inside `collapse_axis0`.
fn weights_to_manifold(flat: &[f32], input_dim: usize, output_dim: usize) -> DiscreteManifold {
    debug_assert_eq!(
        flat.len(), input_dim * output_dim,
        "weights_to_manifold: expected {} floats, got {}",
        input_dim * output_dim, flat.len()
    );
    let mut buf = vec![0.0f32; input_dim * output_dim];
    for i in 0..input_dim {
        for j in 0..output_dim {
            buf[j * input_dim + i] = flat[i * output_dim + j];
        }
    }
    DiscreteManifold::new(buf, input_dim, output_dim)
}

/// Wrap a 1D slice as `DiscreteManifold{width=len, height=1}`.
fn slice_to_dm(data: &[f32]) -> DiscreteManifold {
    DiscreteManifold::new(data.to_vec(), data.len(), 1)
}

// ============================================================================
// Affine + activation primitives
// ============================================================================

/// Affine transform: `output[j] = bias[j] + Σᵢ weights[i,j] * input[i]`.
///
/// Expressed as `IndexLattice2D(input_dim, output_dim).collapse_axis(0, Add, &DenseProduct)`.
///
/// Returns `DiscreteManifold{width=output_dim, height=1}`.
fn affine(
    input: &DiscreteManifold,
    weights: &DiscreteManifold,
    bias: &[f32],
) -> DiscreteManifold {
    let input_dim = weights.width();
    let output_dim = weights.height();
    debug_assert_eq!(input.width(), input_dim, "affine: input/weight dim mismatch");
    debug_assert_eq!(bias.len(), output_dim, "affine: bias/output dim mismatch");

    let lattice = IndexLattice2D::new(input_dim, output_dim);
    let product = DenseProduct { weights, input };
    let sums = lattice.collapse_axis(0, ReduceOp::Add, &product);

    let buf = (0..output_dim)
        .map(|j| bias[j] + sums.buffer()[j])
        .collect::<Vec<_>>();
    DiscreteManifold::new(buf, output_dim, 1)
}

/// Apply ReLU element-wise.
fn relu(dm: DiscreteManifold) -> DiscreteManifold {
    let width = dm.width();
    let buf = dm.into_buffer().into_iter().map(|x: f32| x.max(0.0)).collect();
    DiscreteManifold::new(buf, width, 1)
}

// ============================================================================
// Input preparation helpers
// ============================================================================

/// Build the `[INPUT_DIM]` network input for `forward_shared` (saturation head).
///
/// Uses log2-scaled search scalars (edge_count, node_count, node_budget, epoch_budget).
pub fn build_acc_input_shared(acc: &EdgeAccumulator) -> [f32; INPUT_DIM] {
    let mut input = [0.0f32; INPUT_DIM];
    let scale = if acc.node_count > 0 {
        1.0 / sqrtf(acc.node_count as f32)
    } else {
        1.0
    };
    for i in 0..4 * K {
        input[i] = acc.values[i] * scale;
    }
    let base = 4 * K;
    input[base]     = log2f(1.0 + acc.edge_count as f32);
    input[base + 1] = log2f(1.0 + acc.node_count as f32);
    input[base + 2] = log2f(1.0 + acc.node_budget as f32);
    input[base + 3] = log2f(1.0 + acc.epoch_budget as f32);
    input
}

/// Build the `[INPUT_DIM]` network input for `forward_expr_only` (extraction head).
///
/// Uses variance histogram features in place of search scalars.
pub fn build_acc_input_expr_only(acc: &EdgeAccumulator) -> [f32; INPUT_DIM] {
    let mut input = [0.0f32; INPUT_DIM];
    let scale = if acc.node_count > 0 {
        1.0 / sqrtf(acc.node_count as f32)
    } else {
        1.0
    };
    for i in 0..4 * K {
        input[i] = acc.values[i] * scale;
    }
    let base = 4 * K;
    input[base]     = acc.variance_frac_const;
    input[base + 1] = acc.variance_frac_frame;
    input[base + 2] = acc.variance_frac_scanline;
    input[base + 3] = acc.variance_frac_pixel;
    input
}

/// Build the `[GRAPH_INPUT_DIM]` network input for the graph backbone.
pub fn build_graph_input(gacc: &GraphAccumulator) -> [f32; GRAPH_INPUT_DIM] {
    let mut input = [0.0f32; GRAPH_INPUT_DIM];
    let scale = if gacc.node_count > 0 {
        1.0 / sqrtf(gacc.node_count as f32)
    } else {
        1.0
    };
    for i in 0..4 * K {
        input[i] = gacc.values[i] * scale;
    }
    let base = 4 * K;
    input[base]     = log2f(1.0 + gacc.edge_count as f32);
    input[base + 1] = log2f(1.0 + gacc.node_count as f32);
    input[base + 2] = log2f(1.0 + gacc.node_budget as f32);
    input[base + 3] = log2f(1.0 + gacc.epoch_budget as f32);
    input
}

// ============================================================================
// Forward pass functions
// ============================================================================

/// Edge tower (saturation path): `acc_input → w1 → ReLU → trunk → ReLU`.
///
/// Returns `DiscreteManifold{width=HIDDEN_DIM, height=1}`.
pub fn edge_tower_shared(net: &ExprNnue, acc_input: &[f32; INPUT_DIM]) -> DiscreteManifold {
    let input_dm = slice_to_dm(acc_input);
    let w1_dm = weights_to_manifold(&net.w1_flat(), INPUT_DIM, HIDDEN_DIM);
    let layer1 = relu(affine(&input_dm, &w1_dm, &net.b1));

    let trunk_dm = weights_to_manifold(&net.trunk_w_flat(), HIDDEN_DIM, HIDDEN_DIM);
    relu(affine(&layer1, &trunk_dm, &net.trunk_b))
}

/// Edge tower (extraction path): `acc_input → w1 → ReLU → trunk → ReLU`.
///
/// Uses `build_acc_input_expr_only` — variance histogram instead of search scalars.
pub fn edge_tower_expr_only(net: &ExprNnue, acc_input: &[f32; INPUT_DIM]) -> DiscreteManifold {
    // Same computation as edge_tower_shared; input differs.
    edge_tower_shared(net, acc_input)
}

/// Project edge hidden → expr_embed (no activation).
///
/// Returns `DiscreteManifold{width=EMBED_DIM, height=1}`.
pub fn expr_proj(net: &ExprNnue, hidden: &DiscreteManifold) -> DiscreteManifold {
    let w_dm = weights_to_manifold(&net.expr_proj_w_flat(), HIDDEN_DIM, EMBED_DIM);
    affine(hidden, &w_dm, &net.expr_proj_b)
}

/// Value MLP: `expr_embed → MLP_HIDDEN → ReLU → 1`.
///
/// Returns a scalar cost prediction.
pub fn value_mlp(net: &ExprNnue, expr_embed: &DiscreteManifold) -> f32 {
    let w1_dm = weights_to_manifold(&net.value_mlp_w1_flat(), EMBED_DIM, MLP_HIDDEN);
    let h = relu(affine(expr_embed, &w1_dm, &net.value_mlp_b1));
    h.buffer()
        .iter()
        .zip(net.value_mlp_w2.iter())
        .map(|(&h_j, &w_j)| h_j * w_j)
        .sum::<f32>()
        + net.value_mlp_b2
}

/// Graph tower: `graph_input → graph_w1 → ReLU → trunk → ReLU`.
///
/// Returns `DiscreteManifold{width=HIDDEN_DIM, height=1}`.
pub fn graph_tower(net: &ExprNnue, graph_input: &[f32; GRAPH_INPUT_DIM]) -> DiscreteManifold {
    let input_dm = slice_to_dm(graph_input);
    let w1_dm = weights_to_manifold(&net.graph_w1_flat(), GRAPH_INPUT_DIM, HIDDEN_DIM);
    let layer1 = relu(affine(&input_dm, &w1_dm, &net.graph_b1));

    let trunk_dm = weights_to_manifold(&net.trunk_w_flat(), HIDDEN_DIM, HIDDEN_DIM);
    relu(affine(&layer1, &trunk_dm, &net.trunk_b))
}

/// Project graph hidden → graph_embed (no activation).
///
/// Returns `DiscreteManifold{width=EMBED_DIM, height=1}`.
pub fn graph_proj(net: &ExprNnue, hidden: &DiscreteManifold) -> DiscreteManifold {
    let w_dm = weights_to_manifold(&net.graph_proj_w_flat(), HIDDEN_DIM, EMBED_DIM);
    affine(hidden, &w_dm, &net.graph_proj_b)
}

/// Mask MLP: `graph_embed → MLP_HIDDEN → ReLU → EMBED_DIM` (no activation on output).
///
/// Returns `DiscreteManifold{width=EMBED_DIM, height=1}`.
pub fn mask_mlp(net: &ExprNnue, graph_embed: &DiscreteManifold) -> DiscreteManifold {
    let w1_dm = weights_to_manifold(&net.mask_mlp_w1_flat(), EMBED_DIM, MLP_HIDDEN);
    let h = relu(affine(graph_embed, &w1_dm, &net.mask_mlp_b1));
    let w2_dm = weights_to_manifold(&net.mask_mlp_w2_flat(), MLP_HIDDEN, EMBED_DIM);
    affine(&h, &w2_dm, &net.mask_mlp_b2)
}
```

- [ ] **Step 4: Add `pub mod forward` to `mod.rs`**

```rust
pub mod forward;
pub use forward::{
    build_acc_input_expr_only, build_acc_input_shared, build_graph_input,
    edge_tower_expr_only, edge_tower_shared, expr_proj, graph_proj, graph_tower,
    mask_mlp, value_mlp,
};
```

- [ ] **Step 5: Run the golden tests**

```bash
cargo test -p pixelflow-search nnue_forward_golden
```

Expected:
```
test edge_tower_shared_matches_reference ... ok
test value_cost_matches_reference ... ok
test graph_embed_matches_reference ... ok
```

If any test fails with a value mismatch > 1e-4, check `weights_to_manifold` transpose logic: `buf[j * input_dim + i] = flat[i * output_dim + j]`.

- [ ] **Step 6: Full test suite**

```bash
cargo test -p pixelflow-search
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add pixelflow-search/src/nnue/forward.rs \
        pixelflow-search/src/nnue/mod.rs \
        pixelflow-search/tests/nnue_forward_golden.rs
git commit -m "feat(search): nnue/forward.rs — forward pass as manifold compositions over IndexLattice2D"
```

---

## Task 5: Cleanup — mod.rs, HalfEP deletion, EMBED_DIM fix

**Files:** `pixelflow-search/src/nnue/mod.rs`, `pixelflow-pipeline/src/training/replay.rs`, `docs/`

- [ ] **Step 1: Write the failing test for EMBED_DIM consistency**

Add to `pixelflow-search/tests/nnue_constants.rs` (append):

```rust
#[test]
fn pipeline_embed_dim_matches_search() {
    // This test lives in pixelflow-search but validates that pixelflow-pipeline
    // no longer has a stale constant. The numeric value is the contract.
    assert_eq!(pixelflow_search::nnue::EMBED_DIM, 32,
        "pixelflow-pipeline/src/training/replay.rs had 24 — ensure it now imports from here");
}
```

- [ ] **Step 2: Run to confirm (this should already pass from Task 1)**

```bash
cargo test -p pixelflow-search pipeline_embed_dim_matches_search
```

Expected: pass.

- [ ] **Step 3: Fix `replay.rs` in `pixelflow-pipeline`**

In `pixelflow-pipeline/src/training/replay.rs`:

Replace:
```rust
/// EMBED_DIM from ExprNnue.
const EMBED_DIM: usize = 24;
```

With:
```rust
use pixelflow_search::nnue::constants::EMBED_DIM;
```

- [ ] **Step 4: Build to confirm `replay.rs` compiles**

```bash
cargo build -p pixelflow-pipeline
```

Expected: compiles without errors. If `ReplayStep::rule_embed` field type `[f32; EMBED_DIM]` changes effective size, existing serialized data may be incompatible — but that's expected (constant was wrong before).

- [ ] **Step 5: Delete HalfEP section from `mod.rs`**

In `pixelflow-search/src/nnue/mod.rs`, delete everything from:
```rust
// ============================================================================
// HalfEP Features (Legacy - being phased out in favor of Factored)
// ============================================================================
```
...through the `extract_features_recursive` function and the `HalfEPNetwork` struct.

This removes approximately 350 lines of dead code. Run `grep -n HalfEP mod.rs` first to find the full extent.

- [ ] **Step 6: Fix stale "24 dims" doc comments in `factored.rs`**

Run:
```bash
grep -n "24 dims\|EMBED_DIM=24\|24)" pixelflow-search/src/nnue/factored.rs | head -20
```

For each match, update the comment to say "32 dims" or "EMBED_DIM=32". These are doc comments only — no code changes.

- [ ] **Step 7: Verify all files are under 500 lines**

```bash
wc -l pixelflow-search/src/nnue/*.rs
```

The spec requires no file over 500 lines. If `factored.rs` (ExprGenerator + BwdGenerator) is over 500 lines after the split, split it: `factored.rs` keeps `ExprGenerator`, move `BwdGenerator` to `bwd_generator.rs`.

- [ ] **Step 8: Full workspace test**

```bash
cargo test --workspace
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add pixelflow-search/src/nnue/mod.rs \
        pixelflow-pipeline/src/training/replay.rs \
        pixelflow-search/src/nnue/factored.rs
git commit -m "fix(search/pipeline): delete HalfEP legacy code, fix EMBED_DIM=24→32 in replay.rs"
```

---

## Verification

```bash
# All unit tests pass
cargo test -p pixelflow-search

# Golden test: forward pass outputs match reference on 10 random seeds
for seed in 1 2 3 4 5 6 7 8 9 10; do
  cargo test -p pixelflow-search nnue_forward_golden 2>&1 | grep -E "ok|FAILED"
done

# No file over 500 lines
wc -l pixelflow-search/src/nnue/*.rs | sort -n

# EMBED_DIM is 32 everywhere
grep -r "EMBED_DIM.*24\|24.*EMBED_DIM" pixelflow-search/ pixelflow-pipeline/
# Expected: no matches
```
