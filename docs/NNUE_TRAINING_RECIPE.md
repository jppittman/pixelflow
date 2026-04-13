# NNUE Training Recipe

How to train the learned compiler optimizer from scratch. This document covers the full pipeline from data generation to deployment in the `kernel!` macro.

## Architecture Overview

Two neural networks cooperate:

- **ExprNnue** (Rust, CPU): Tiny NNUE with two heads
  - **Extraction head**: Predicts expression execution cost. Guides e-graph extraction to pick the cheapest equivalent form.
  - **Saturation head**: Decides which rewrite rules to apply during e-graph equality saturation.

- **Critic** (Python, PyTorch, GPU): Causal Decision Transformer that assigns per-step temporal credit for the saturation head's decisions.

The training loop is AlphaZero-inspired:
```
GENERATE (self-play) -> EXPORT (.pftraj) -> CRITIQUE (transformer) -> UPDATE (joint backprop)
```

## Prerequisites

```bash
# Rust stable toolchain
rustup default stable

# Python dependencies (via uv)
uv pip install torch fastapi uvicorn

# Build everything
cargo build --release -p pixelflow-pipeline --features training
```

## Step 1: Generate Training Corpus

The corpus contains expressions the optimizer will train on. Two sources:

### Synthetic (shader-weighted)
```bash
./target/release/gen_bench_corpus --target 50000 --max-nodes 1000 \
  --output pixelflow-pipeline/data/bench_corpus.bin
```

Op distribution is weighted to match real shader code (heavy on mul/add/sub, moderate sin/cos/abs/clamp, light on exotic ops).

### LLM-Generated Shader Expressions
Use Claude/Gemini to generate realistic shader math:
```bash
# See scripts/scrape_shadertoy.py for the extraction pipeline
# Or generate directly with LLM and validate:
cargo run --release -p pixelflow-pipeline --features training \
  --bin validate_corpus -- input.jsonl pixelflow-pipeline/data/bench_corpus.bin
```

The validator uses `parse_kernel_code_arena` with structural dedup — node counts reflect DAG size, not tree size.

## Step 2: Bootstrap the Extraction Head

Train the extraction head on JIT-benchmarked expressions (no self-play needed):

```bash
./target/release/bootstrap_extraction_head \
  --epochs 100 \
  --batch-size 128 \
  --seed 42 \
  --synthetic 20000 \
  --output pixelflow-pipeline/data/judge.bin
```

This generates random expressions, JIT-benchmarks them with `eval100!` unrolled timing, and trains the extraction head's value MLP on (EdgeAccumulator -> ln(cost_ns)) pairs.

Key: The extraction head uses `forward_expr_only()` which sees only expression structure (op types + edge depths), NOT search metadata (node counts, budgets). This prevents it from learning shortcuts like "more nodes = cheaper."

## Step 3: Start the Critic Server

```bash
uv run pixelflow-pipeline/scripts/critic_server.py \
  --checkpoint pixelflow-pipeline/data/unified/critic.pt \
  --port 8765
```

The critic server exposes:
- `POST /step` — predict advantages + one backprop step (main training endpoint)
- `POST /predict` — inference only
- `POST /retrain` — full retrain on accumulated buffer
- `GET /health` — status check

## Step 4: Online Training

```bash
./target/release/train_unified \
  --rounds 5000 \
  --trajectories-per-round 200 \
  --corpus-fraction 0.5 \
  --updates-per-round 4 \
  --mini-batch-size 1024 \
  --lr 0.003 \
  --momentum 0.002 \
  --weight-decay 0.0 \
  --grad-clip 5.0 \
  --threshold 0.4 \
  --miss-penalty 0.2 \
  --max-steps 61 \
  --value-coeff 0.22 \
  --entropy-coeff 0.034 \
  --relabel-interval 0 \
  --output-dir pixelflow-pipeline/data/unified
```

### What happens per round:
1. **GENERATE**: 200 trajectories via parallel self-play (12 workers)
   - Each trajectory: build e-graph, apply rules guided by saturation head, extract via extraction head, JIT benchmark initial + final expressions
   - Budget scales with expression size: `50 + mult * initial_nodes` (mult in [3,10])
2. **EXPORT**: Write `.pftraj` binary trajectory batches with both initial and final EdgeAccumulator states
3. **CRITIQUE**: `/step` endpoint does forward pass (produce advantages) + one backprop step on the current batch
4. **UPDATE**: 4 SGD steps from replay buffer (saturation head) + trajectory-level extraction head training on paired (accumulator, cost) data

### Hyperparameters (from Optuna sweep)
These were found via `scripts/optuna_unified.py` with 50 trials:
- **LR 0.003**: 30x higher than initial guess — the model learns fast
- **Momentum 0.002**: Basically zero — pure SGD works best
- **Grad clip 5.0**: Generous — don't clip too aggressively
- **Updates per round 4**: Few updates — don't overfit to each batch
- **Mini-batch 1024**: Large batches for stable gradients

## Step 5: Optuna Hyperparameter Sweep (Optional)

```bash
# Start server mode
cargo run --release -p pixelflow-pipeline --features training \
  --bin train_unified -- --server /tmp/train_unified.sock

# Run sweep (separate terminal)
uv run pixelflow-pipeline/scripts/optuna_unified.py \
  --n-trials 50 --max-rounds 15 --study-name my_sweep
```

Optuna scores trials by validation speedup on held-out real shader expressions (psychedelic, channel, normalize, etc.), not training speedup. This prevents the optimizer from gaming easy synthetic expressions.

## Step 6: Deploy Weights

```bash
# Copy best checkpoint to compiler
cp pixelflow-pipeline/data/unified/model_r<BEST>.bin \
   pixelflow-compiler/weights/expr_nnue.bin

# Rebuild — every kernel! now uses trained weights
cargo build --release
```

The weights are embedded via `include_bytes!` in `pixelflow-compiler/src/optimize.rs`. No runtime file loading.

## Key Design Decisions

### Extraction Head Training
The extraction head is trained on **correctly paired data**: (expression's own accumulator, expression's own JIT cost). NOT (initial accumulator, final cost) which would teach "predict optimization outcome" rather than "predict execution cost."

Training happens in a separate loop at the trajectory level, not in the per-step SGD loop.

### DAG-Aware Accumulator
`EdgeAccumulator::from_dag_choices()` counts shared subexpressions once (for computation) plus `(ref_count - 1)` Var-reference edges (for register loads). This matches what the JIT actually compiles — shared subexpressions become let-bindings.

`compute_ref_counts()` walks the extraction choices to find which e-classes are referenced multiple times.

### Benchmark Methodology
- `eval100!` macro: 100 function calls fully unrolled, zero loop counter overhead
- 20 timed samples, take median
- `mach_absolute_time()` on macOS (1ns resolution on Apple Silicon)
- `benchmark_jit` divides by INNER_ITERS to return per-eval nanoseconds

The loop counter bias (~0.3ns/iter) was corrupting the additive cost model — small expressions appeared proportionally more expensive. Full unrolling eliminates this.

### Critic Architecture
Causal Decision Transformer with 3 layers, 4 attention heads, 128-dim. The `/step` endpoint does predict + one backprop per round — keeping the critic current without expensive full retrains. The critic's forward pass produces advantages `A_t = R_T - V_t` where `R_T = -log(final_cost_ns)`.

### Expression Generator
Op weights match ShaderToy shader analysis: 36% mul, 18% add, 14% sub, 6% div, with moderate abs/sin/cos/clamp and rare exotic ops. Default depth 8, leaf probability 0.2.

## Results

- **Synthetic expressions**: 95%+ achieve >1x speedup, median 1.05x
- **Psychedelic shader (3-channel)**:
  - JIT: 0.420ns/eval
  - LLVM (random NNUE): 0.458ns/eval
  - LLVM (trained NNUE): 0.447ns/eval
  - Cross-channel CSE: 2.3x faster than 3 separate channels
- **Never anti-optimizes**: Extraction starts from original, only accepts NNUE-certified improvements

## File Map

| File | Purpose |
|------|---------|
| `pixelflow-pipeline/src/bin/train_unified.rs` | Main training orchestrator |
| `pixelflow-pipeline/src/training/self_play.rs` | Trajectory generation |
| `pixelflow-pipeline/src/training/unified_backward.rs` | Joint backprop |
| `pixelflow-pipeline/scripts/critic_server.py` | Decision Transformer critic |
| `pixelflow-pipeline/scripts/optuna_unified.py` | Hyperparameter sweep |
| `pixelflow-pipeline/src/bin/bootstrap_extraction_head.rs` | Initial value head training |
| `pixelflow-search/src/nnue/factored.rs` | ExprNnue model + EdgeAccumulator |
| `pixelflow-search/src/egraph/extract.rs` | DAG-aware incremental extraction |
| `pixelflow-compiler/src/optimize.rs` | kernel! macro integration |
| `pixelflow-compiler/weights/expr_nnue.bin` | Trained weights (shipped) |
