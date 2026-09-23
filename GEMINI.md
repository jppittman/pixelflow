# PixelFlow (core-term) Project Context

## Project Overview

**PixelFlow** is a research project exploring **pull-based rendering** with **SIMD as algebra** on pure CPU without a GPU.

**`core-term`** is the primary consumer application: a high-performance, correct terminal emulator built entirely on the PixelFlow engine.

### Core Philosophy
1.  **Pull-based Rendering:** Pixels are sampled, not pushed. The system asks "what color is this pixel?", eliminating overdraw and complex rasterization state.
2.  **SIMD as Algebra:** there is no vector type in the language or in `pixelflow-core`; the JIT's ISA tier (AVX2+FMA or AVX-512 on x86-64, NEON on aarch64) is decided at process startup by the CPU. Users write algebraic equations as `Kernel` values; the compiler owns the loop nest and emits vectorized assembly.
3.  **The Kernel/Lattice Abstraction:** A `Kernel` is an immutable handle to an `ExprArena` fragment — the language's one runtime value, JIT-first from the start. `Manifold::compile` specializes a `Kernel` at a lattice's shape, and `Lattice::collapse` is the one verb that produces numbers. There is no type-level combinator tier to write by hand any more — that tier (manifolds as zero-sized expression templates evaluated one SIMD batch at a time) was retired by [A Kernel with a Lattice](docs/plans/2026-09-06-kernel-with-a-lattice.md). The intended way to write PixelFlow code is the `kernel!` macro, which compiles expressions through an e-graph optimizer and codegen pipeline.
4.  **Zero Allocations:** The rendering loop is designed to have zero heap allocations per frame.

## Workspace Structure

The project is a Rust workspace with the following key members:

*   **`core-term`**: The terminal emulator application. (First consumer)
*   **`pixelflow-core`**: Lattices, the compiled `Manifold`, `collapse`, and the cell grid. `no_std`, and holds no vector code: the ISA tier is the JIT's.
*   **`pixelflow-compiler`**: Proc-macro compiler for the `kernel!` macro (lexer, parser, sema, codegen).
*   **`pixelflow-ir`**: Shared IR (`ExprArena`, `OpKind`, backend execution traits, the `Kernel` value/AST).
*   **`pixelflow-codegen`**: Per-ISA emitters (x86-64, aarch64), register allocation, executable memory, and the JIT compile cache — expression graphs to machine code.
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
*   **Run macOS App:** `cargo bundle-run` (Bundles and runs `CoreTerm.app`)
*   **Run Tests:** `cargo test --workspace`
*   **Benchmarks:** `cargo bench -p pixelflow-core`

### Build Profiles
*   **`dev`**: `opt-level = 0`, `panic = "abort"`. The former opt-level 1-2 workaround for deeply nested expression-template types is obsolete: the JIT-first `Kernel`/`ExprArena` architecture superseded that layer (see `docs/plans/2026-07-20-kernel-unification.md`).
*   **`release`**: `opt-level = 3`, `panic = "abort"`.
*   **`bench`**: `lto = true`, `codegen-units = 1`.
*   **`dist`**: inherits `release`, adds `lto = true`, `codegen-units = 1`, `strip = true`.

## Development Conventions

### Architectural Constraints
*   **No Terminal Logic in PixelFlow:** Keep `pixelflow-*` crates general-purpose. Terminal specific logic belongs in `core-term`.
*   **Pull, Don't Push:** Rendering logic must adhere to the pull-based paradigm.
*   **Types are Shaders:** Use the type system to build compute graphs.
*   **Platform Isolation:** Platform-specific code (macOS/Linux/Web) goes in `pixelflow-runtime`.

### Coding Style

[`docs/STYLE.md`](docs/STYLE.md) is the canonical style guide. Make all style
changes there so contributors and automated agents follow a single source of
truth.

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
