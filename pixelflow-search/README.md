# pixelflow-search

Arena-native e-graph optimization and model scoring for PixelFlow.

## Overview

This crate provides the live optimization stack:

1. **E-graph equality saturation** - find algebraically equivalent forms
2. **Arena-native extraction** - materialize the chosen result as `ExprArena`
3. **EgraphOptimizationModel** - score saturation and extraction without benchmarking

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training (offline)                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Generate seed expressions                                   │
│  2. E-graph saturation → find all equivalents                   │
│  3. Benchmark sampled variants (real SIMD costs)                │
│  4. Train the shared model:                                     │
│     - Extraction head: features(expr) → cost_ns                 │
│     - Saturation head: graph/rule state → rewrite score         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Inference (compile-time)                    │
├─────────────────────────────────────────────────────────────────┤
│  For new kernel K:                                              │
│  1. Insert K into e-graph                                       │
│  2. Saturate with algebraic rules                               │
│  3. Extract the best predicted arena form                       │
└─────────────────────────────────────────────────────────────────┘
```

## Usage

### Basic E-Graph Optimization

```rust
use pixelflow_ir::ExprArena;
use pixelflow_search::egraph::{ops, CostModel, EGraph, ENode};

let mut arena = ExprArena::new();
let x = arena.push_var(0);
let zero = arena.push_const(0.0);
let one = arena.push_const(1.0);
let add = arena.push_binary(pixelflow_ir::OpKind::Add, x, zero);
let root_expr = arena.push_binary(pixelflow_ir::OpKind::Mul, add, one);

let mut eg = EGraph::new();
let root = eg.add_arena(&arena, root_expr);
eg.saturate();

// Extract optimal arena form
let costs = CostModel::default();
let (optimized, optimized_root, cost) = eg.extract_best(root, &costs);
assert!(optimized.node_count_subtree(optimized_root) > 0);
assert!(cost <= 2);
```

## Modules

| Module | Purpose |
|--------|---------|
| `egraph` | E-graph implementation with equality saturation |
| `egraph::codegen` | DAG → `kernel!` macro code generation |
| `egraph::saturate` | Budget-limited saturation |
| `nnue` | Shared optimization model + accumulators |
| `nnue::factored` | O(ops) factored embedding architecture |
| `training` | Training infrastructure (feature-gated) |

## Features

- `std` (default): Enable standard library features
- `training`: Enable training utilities (data generation, backprop)
- `nnue`: Enable NNUE integration for best-first search

## Training Pipeline

```bash
# 1. Generate variants and benchmark code
cargo run -p pixelflow-compiler --example gen_egraph_variants --features training

# 2. Collect benchmark costs
cargo run -p pixelflow-compiler --example collect_benchmark_costs --features training

# 3. Train the Judge (value head)
cargo run -p pixelflow-compiler --example train_with_validation --features training

# 4. Collect search training data using the Judge
cargo run -p pixelflow-compiler --example collect_search_training --features training

# 5. Train the Guide (search head)
cargo run -p pixelflow-compiler --example train_search_head --features training
```

## Integration

This crate works with:

- **pixelflow-ir**: Shared IR types (Expr, OpKind)
- **pixelflow-compiler**: Compile-time kernel optimization

## References

- [egg: E-Graphs Good](https://egraphs-good.github.io/) - E-graph background
- [Stockfish NNUE](https://github.com/official-stockfish/Stockfish) - NNUE architecture inspiration
- [AlphaZero](https://arxiv.org/abs/1712.01815) - Dual-head value/policy network
