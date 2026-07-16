# PixelFlow (core-term) Project Context

## Project Overview

**PixelFlow** is a research project demonstrating a novel paradigm for real-time graphics: **pull-based rendering** with **SIMD as algebra**. It achieves high performance (155 FPS at 1080p) on pure CPU without a GPU.

**`core-term`** is the primary consumer application: a high-performance, correct terminal emulator built entirely on the PixelFlow engine.

### Core Philosophy
1.  **Pull-based Rendering:** Pixels are sampled, not pushed. The system asks "what color is this pixel?", eliminating overdraw and complex rasterization state.
2.  **SIMD as Algebra:** The `Field` type wraps SIMD vectors (AVX-512, SSE2, NEON) transparently. Users write algebraic equations, and the compiler emits optimal vectorized assembly.
3.  **Manifold Abstraction:** Everything is a `Manifold`—a domain-generic function `Manifold<P> { fn eval(&self, p: P) -> Output }` from a coordinate domain `P` to a value (e.g. `(Field, Field)` for 2D, so a kernel never pays for dimensions it doesn't use). Composing manifolds creates complex scenes that are compiled into fused kernels. **Writing manifolds by hand is a legacy pattern** — the intended way to write PixelFlow code is the `kernel!` macro, which compiles expressions through an e-graph optimizer and codegen pipeline.
4.  **Zero Allocations:** The rendering loop is designed to have zero heap allocations per frame.

## Workspace Structure

The project is a Rust workspace with the following key members:

*   **`core-term`**: The terminal emulator application. (First consumer)
*   **`pixelflow-core`**: Pure algebra, `Field`, `Manifold` traits. `no_std`, SIMD backend implementations.
*   **`pixelflow-compiler`**: Proc-macro compiler for the `kernel!` macro (lexer, parser, sema, codegen).
*   **`pixelflow-ir`**: Shared IR (`ExprArena`, `OpKind`, backend execution traits).
*   **`pixelflow-search`**: E-graph optimization — rewrite rules, saturation, cost-model extraction.
*   **`pixelflow-pipeline`**: Cost-model tooling (JIT bench harness, corpus generation, extraction benchmarks).
*   **`pixelflow-graphics`**: Rendering logic, colors, fonts, rasterization.
*   **`pixelflow-ml`**: Graphics ML experiments (harmonic attention, spherical-harmonic feature maps).
*   **`pixelflow-runtime`**: Platform abstraction (Cocoa, X11, Web), input handling, render orchestration.
*   **`actor-scheduler`**: Lock-free, priority-based actor concurrency model (`Control > Management > Data` lanes).
*   **`actor-scheduler-macros`**: Procedural macros for the actor system.
*   **`xtask`**: Build automation (bundling macOS apps, etc.).

## Building and Running

### Prerequisites
*   **Rust Stable:** (See `rust-toolchain.toml`)
*   **macOS:** Native Cocoa support.
*   **Linux:** X11 development headers (`libx11-dev`, `libxft-dev`, etc.).

### Key Commands

*   **Build Release:** `cargo build --release`
*   **Run Terminal:** `cargo run --release -p core-term`
*   **Run macOS App:** `cargo xtask bundle-run` (Bundles and runs `CoreTerm.app`)
*   **Run Tests:** `cargo test --workspace`
*   **Benchmarks:** `cargo bench -p pixelflow-core`

### Build Profiles
*   **`dev`**: `opt-level = 1` (Required to prevent stack overflows from deep Manifold recursion).
*   **`release`**: `opt-level = 3`, `panic = "abort"`.
*   **`bench`**: `lto = true`, `codegen-units = 1`.
*   **`dist`**: inherits `release`, adds `lto = true`, `codegen-units = 1`, `strip = true`.

## Development Conventions

### Architectural Constraints
*   **No Terminal Logic in PixelFlow:** Keep `pixelflow-*` crates general-purpose. Terminal specific logic belongs in `core-term`.
*   **Pull, Don't Push:** Rendering logic must adhere to the pull-based paradigm.
*   **Types are Shaders:** Use the type system to build compute graphs.
*   **Platform Isolation:** Platform-specific code (macOS/Linux/Web) goes in `pixelflow-runtime`.

### Coding Style (See `docs/STYLE.md`)
*   **Comments:**
    *   **Public API (`///`):** Document **WHAT** and **HOW**.
    *   **Implementation (`//`):** Document **WHY**. Explain design rationale, not obvious logic.
    *   **No History:** Do not put changelogs or "old code" in comments.
*   **Structure:**
    *   Avoid deep nesting; use guard clauses.
    *   Prefer `match` over `else if`.
*   **Functions:**
    *   Keep argument count low (< 4). Group related args into structs.
    *   **No Boolean Args:** Use enums for clarity (e.g., `Persistence::Permanent` vs `true`).
*   **Magic Numbers:** Use named constants or enums.

### Git & Workflow
*   **Atomic Commits:** Focus on one logical change per commit.
*   **Commit Messages:** Explain *why* a change was made.
*   **Tests:** Public API changes require test updates.
