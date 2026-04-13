# NNUE Integration Status Analysis

## Executive Summary

**Status:** **ACTIVE CONSOLIDATION (2026)**

**The "Triple IR" Problem:** The project is migrating from three redundant IRs (`pixelflow-compiler`, `pixelflow-search`, `pixelflow-ml`) to a unified `pixelflow-ir` core.

**HCE Removal:** Hand-Crafted Evaluation (HCE) has been removed as a redundant and obsolete prototype. The system now focuses entirely on learned cost models ("The Judge") and neural search guidance ("The Guide").

## Current Infrastructure

### 1. The Judge: Learned Cost Model
**Crate:** `pixelflow-pipeline` (Training) / `pixelflow-search` (Inference)
- **Architecture:** `ExprNnue` (factored 128-dim embeddings).
- **Consolidation:** `BwdGenerator` has been unified in `pixelflow-search`. It now uses `RuleTemplates` derived from e-graph rewrites for data-driven "junkification" and training data generation.
- **Integration:** Successfully wired into `pixelflow-search::egraph::extract_neural` and `extract_beam`.

### 2. The Guide: Search Guidance
**Crate:** `pixelflow-pipeline` (Training) / `pixelflow-search` (Inference)
- **Status:** Integrated into `pixelflow-search::egraph::guided_search`.
- **Training:** Unified self-play loop in `pixelflow-pipeline/src/bin/train_unified.rs`.

## Simplification & Cruft Removal (Completed)

- **Scorched Earth on `pixelflow-ml`:** Removed the old NNUE prototype island, redundant AST, and manual `BwdGenerator`. `pixelflow-ml` is now focused strictly on graphics ML (harmonic attention, SH features).
- **HCE Decommissioned:** All references to "Hand-Crafted Evaluator" in benchmarks and extraction lanes have been replaced with "Neural-DP" (The Judge).
- **BwdGenerator Unified:** The search-based generator in `pixelflow-search` is now the sole implementation, supporting all e-graph rewrite rules via templates.

## Next Steps

1. **IR Unification (Critical):** Fully migrate `pixelflow-compiler` and `pixelflow-search` to use `pixelflow-ir`'s canonical `Expr` and `OpKind`.
2. **Cost Model Calibration:** Continue refining "The Judge" weights using real SIMD benchmark data from `bench_jit_corpus`.
3. **Training Flow Cleanup:** Delete superseded training binaries in `pixelflow-pipeline` once `train_unified` is fully validated.

The "Formula 1 car" (NNUE) is now out of the garage and being integrated as the primary engine for the compiler.
