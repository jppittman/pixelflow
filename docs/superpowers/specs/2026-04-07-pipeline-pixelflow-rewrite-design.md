# Pipeline → PixelFlow Rewrite: Design Spec

**Date:** 2026-04-07  
**Status:** Approved  
**Approach:** Option B — Lattice + IR in parallel, forward pass third, backward + training loop fourth

---

## Context

The pixelflow-pipeline trains an NNUE cost model (ExprNnue) used by the e-graph optimizer to select rewrites. The pipeline's math is currently expressed as imperative Rust loops scattered across ~10,000 lines in three crates with no clear API boundaries, inconsistent constants (`EMBED_DIM` is 32 in pixelflow-search, 24 in pixelflow-pipeline), and the backward pass living in the wrong crate.

The rewrite expresses all math as pull-based pixelflow manifold compositions — the same model used for graphics. A neural network layer becomes a manifold over output indices; a matmul becomes `collapse_axis(Add)` over the input dimension; a training step becomes multi-pass lattice collapse. This is not a metaphor — the math is structurally identical.

**Autodiff direction:** Pullback rules live on IR nodes (`OpKind`), not on Rust combinator types. The JIT emits both forward and backward passes from the same `ExprArena`. This is the Conal Elliott dual-category approach implemented at the IR level. The combinator layer is transitional — don't architect around it.

---

## Five Sane Layers

```
Layer 0: pixelflow-ir
         ExprArena nodes + VJP pullback rules per OpKind.
         Stable across frontend changes.

Layer 1: pixelflow-core
         Lattice (demand side). Extended with IndexLattice for tensor indexing.
         Drives what gets evaluated and when.

Layer 2: pixelflow-search/nnue
         Accumulators, weights as DiscreteManifold, forward pass
         as manifold compositions over IndexLattice.

Layer 3: pixelflow-search/nnue
         Gradient buffer, backward pass via IR pullback traversal.
         Chain rule = reverse topological walk of ExprArena.

Layer 4: pixelflow-pipeline
         Orchestration only: generate → export → critique → update.
         No math. No tensor loops. Calls Layer 2/3 APIs, handles I/O.
```

Each layer depends only on layers below. `pixelflow-pipeline` never reaches into `nnue` internals.

---

## Network Ground Truth

The ExprNnue has two input towers feeding a shared trunk:

```
EdgeAccumulator[132]  → w1[132×64]     → ReLU → trunk[64×64] → ReLU → expr_proj[64×32] → value_mlp → cost
GraphAccumulator[132] → graph_w1[132×64] → ReLU → trunk[64×64] → ReLU → graph_proj[64×32] → mask_mlp → bilinear → rule_score
                                                    ^^^^^^ SAME WEIGHTS
```

Key constants (canonical, defined once in `pixelflow-search/src/nnue/constants.rs`):

| Constant | Value |
|----------|-------|
| `K` | 32 |
| `INPUT_DIM` | 132 (4K + 4 scalars) |
| `GRAPH_INPUT_DIM` | 132 (4K + 4 scalars) |
| `HIDDEN_DIM` | 64 |
| `EMBED_DIM` | 32 |
| `MLP_HIDDEN` | 16 |

---

## The Pull Reframe

A matmul is a `collapse_axis` over the input dimension:

```rust
// output[j] = bias[j] + Σᵢ W[i,j] * input[i]
// IndexLattice1D(INPUT) loops X over [0, INPUT). Y is fixed to output index j.
// W is a DiscreteManifold indexed by (input_i, output_j) = (X, Y).
// Collapse over X (axis 0) sums the products → scalar per j.
// Outer IndexLattice1D(OUTPUT) materializes all j at once.
let matmul_j = IndexLattice1D(INPUT).collapse_axis(0, Add, &(W.at(X, Y) * input.at(X)));
let output = IndexLattice1D(OUTPUT).collapse(&(bias.at(X) + matmul_j));
```

Multi-layer forward = multi-pass lattice collapse, same as multi-pass rendering.

Backward pass has the same structure:
- `d_input[i] = Σⱼ d_output[j] * W[i,j]` → `collapse_axis(1, Add, ...)`
- `d_W[i,j] = d_output[j] * input[i]` → 2D lattice, no reduction
- Both expressed as IR pullback rules on the matmul `OpKind`

---

## Team Workstreams

### Team 1 — Lattice Extensions (`pixelflow-core`)

**Goal:** Make the Lattice capable of expressing ML tensor operations.

**Deliverables:**
1. `IndexLattice1D(n)` — domain `[0, n)` mapped to X. No pixel-space semantics.
2. `IndexLattice2D(m, n)` — domain `[0,m) × [0,n)` mapped to (X, Y).
3. `fn collapse_axis(&self, axis: usize, op: ReduceOp, manifold: &M) -> DiscreteManifold` — reduces along one axis, keeps remaining dims. This is the matmul primitive.
4. Integer exact indexing in `DiscreteManifold` (currently floor+clamp nearest-neighbor). Add a `gather_exact` path for integer coordinates.

**Cleanup mandate:**
- Document the semantic distinction between `FrameLattice`/`ScanlineLattice` (pixel-space) and `IndexLattice` (tensor-space) in module doc comment.
- `lattice.rs` must stay under 500 lines. Split if needed.
- No public fields on concrete lattice types without documented invariants.

---

### Team 2 — IR Pullback Registry (`pixelflow-ir` + `pixelflow-compiler`)

**Goal:** Each `OpKind` knows its own VJP rule. The compiler can emit backward passes.

**Deliverables:**
1. `trait HasPullback` on `OpKind` (or a match-based registry):
   ```rust
   fn pullback(op: OpKind, output_grad: ExprId, inputs: &[ExprId], arena: &mut ExprArena) -> Vec<ExprId>
   ```
   Returns one `ExprId` per input — the gradient flowing back to that input.
2. Pullback rules for all current `OpKind` variants:
   - Arithmetic (`Add`, `Mul`, `Sub`, `Div`): standard chain rules
   - `Sqrt`, `Exp`, `Log`, `Log2`: standard derivatives
   - `Max(a, b)`: step function gating (`a > b ? d_out : 0`)
   - `ReLU` (max(0,x)): `x > 0 ? d_out : 0`
   - `MatMul` (once it exists as an `OpKind`): `d_input = Wᵀ @ d_out`, `d_W = d_out ⊗ input`
3. `fn emit_backward(arena: &ExprArena, loss_node: ExprId) -> ExprArena` in `pixelflow-compiler` — reverse topological traversal, accumulates gradients via pullback rules.

**Cleanup mandate:**
- Delete any remaining legacy `Expr` (Arc-based) shim code.
- `OpKind` variants must all have doc comments.
- Constants defined once. No magic numbers in pullback rules.
- `pixelflow-ir/src/` files each under 400 lines.

---

### Team 3 — Forward Pass Rewrite (`pixelflow-search`, depends on Team 1)

**Goal:** Rewrite `ExprNnue` forward pass as manifold compositions over `IndexLattice`.

**Deliverables:**
1. Split `factored.rs` (5,418 lines) into:
   - `constants.rs` — single source of truth for all dimension constants
   - `accumulators.rs` — `EdgeAccumulator`, `GraphAccumulator`, builder impls
   - `network.rs` — `ExprNnue` weight struct, weights as `DiscreteManifold`
   - `forward.rs` — forward pass expressed as manifold compositions
2. Forward pass for each layer as a clean manifold expression:
   - `edge_tower(acc_input) → DiscreteManifold[HIDDEN_DIM]`
   - `shared_trunk(tower_out) → DiscreteManifold[HIDDEN_DIM]`
   - `expr_proj(hidden) → DiscreteManifold[EMBED_DIM]`
   - `value_mlp(expr_embed) → f32`
   - `mask_mlp(graph_embed) → DiscreteManifold[EMBED_DIM]`
   - `bilinear_score(mask_features, rule_embed) → f32`
3. All outputs numerically identical to current `ExprNnue::forward_*` — verified by test.
4. Delete legacy `HalfEP` feature code from `nnue/mod.rs`.

**Cleanup mandate:**
- No file in `pixelflow-search/src/nnue/` over 500 lines after this work.
- `EMBED_DIM` is 32 everywhere. Fix all callsites where it was 24.
- All accumulator builder logic lives in `accumulators.rs`. None in pipeline.
- `nnue/mod.rs` becomes a thin re-export module only.

---

### Team 4 — Backward Pass + Training Loop (depends on Teams 2 + 3)

**Goal:** Backward pass via IR pullbacks; training loop is orchestration-only.

**Deliverables:**
1. `pixelflow-search/src/nnue/gradient.rs`:
   - `GradientBuffer` struct — mirrors `ExprNnue` weight shapes
   - `backward_extraction(cache, d_value_pred) → GradientBuffer`
   - `backward_saturation(cache, d_score, advantage) → GradientBuffer`
   - `merge_gradients(a: &GradientBuffer, b: &GradientBuffer) → GradientBuffer`
   - `sgd_step(model: &mut ExprNnue, grads: &GradientBuffer, lr: f32)`
2. Backward pass expressed using IR pullback traversal from Team 2 — no hand-written chain rule loops.
3. `pixelflow-pipeline/src/bin/train_unified.rs` refactored:
   - GENERATE: calls `trajectory::collect_batch(&model, &config)`
   - EXPORT: calls `io::write_trajectories_jsonl(&trajectories)`
   - CRITIQUE: calls `critic::run(&config)` (unchanged Python path)
   - UPDATE: calls `training::apply_update(&mut model, &trajectories, &advantages, &config)`
   - No tensor math in `main()` or the binary.
4. `pixelflow-pipeline/src/training/` modules:
   - `trajectory.rs` — `Trajectory`, `TrajectoryStep`, collection logic
   - `io.rs` — JSONL serialization/deserialization
   - `update.rs` — loss computation + gradient accumulation (calls Layer 3)

**Cleanup mandate:**
- `unified_backward.rs` does not exist after this work. Fully replaced.
- `train_unified.rs` under 200 lines. Orchestration only.
- No math in `pixelflow-pipeline` — it calls `pixelflow-search` APIs.
- Trajectory I/O separated from trajectory math. No `acc_to_vec` living in `self_play.rs`.

---

## Dependency Graph

```
Team 1 (Lattice Extensions)   ──────────────────────────┐
                                                          ▼
Team 2 (IR Pullbacks)    ──────────────────┐      Team 3 (Forward Pass)
                                           ▼             │
                                    Team 4 (Backward + Training Loop)
```

Teams 1 and 2 start simultaneously. Team 3 starts when Team 1 ships. Team 4 starts when Teams 2 and 3 ship.

---

## Verification

Each team verifies:

- **Team 1:** `cargo test -p pixelflow-core` passes. Add tests: `IndexLattice1D(4).collapse_axis(0, Add, dot_product_manifold)` matches hand-computed dot product.
- **Team 2:** `cargo test -p pixelflow-ir` passes. Add tests: `emit_backward` on a simple `y = W @ x + b` graph produces correct gradient nodes. Verify symbolically for `Add`, `Mul`, `ReLU`.
- **Team 3:** `cargo test -p pixelflow-search` passes. Golden test: forward pass outputs from new manifold path match old `ExprNnue::forward_*` outputs on 10 random inputs.
- **Team 4:** `cargo test --workspace` passes. End-to-end: run 3 rounds of `train_unified` with a small corpus, verify loss decreases. Verify `unified_backward.rs` is gone.
