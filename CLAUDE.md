# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**core-term** is a GPU-free terminal emulator built on PixelFlow, a pull-based functional graphics engine using CPU SIMD. The project demonstrates that elegant algebraic abstractions can achieve 155 FPS at 1080p on pure CPU.
**pixelflow** is an eDSL built on rust isomorphic to the typed lambda calculus.
**pixelflow-graphics** is a graphic library built using the aforementioned eDSL.
**pixelflow-runtime** offers a platform agnostic runtime for applications using pixelflow rendering.
**actor-scheduler** offers a user space cooperative scheduler for actor model based libraries/applications

## Critical Constraints

- **NO TERMINAL LOGIC GOES IN PIXELFLOW.** PixelFlow is a general-purpose graphics library being extracted to its own crate/repo. Keep it terminal-agnostic.
- Exporting direct manipulation of fields from pixelflow-core is strictly forbidden. Construct compute kernels at load time and render them.
- **NO PUBLIC raw_mul, raw_select, raw_add ETC USAGE** NONE. ZERO. Do not perform raw operations on fields/jets without explicit direction. ALWAYS construct the AST, then use the nested contramap pattern to evaluate it.
- **SIMD is an implementation detail.** `Batch::splat` and `Field::splat` are `pub(crate)`. Do NOT expose them. Do not expose `SimdVec`s. Do not expose anything that hints at lanes. pixelflow-core is an algebra; writing it should look like Halide, not assembly.
- **Minimal public API** - Do NOT change visibility of internal APIs without explicit permission. Keep `pub(crate)` and private items encapsulated. Use Manifold composition instead of exposing internals.
- **Suckless Dependencies**: Keep dependencies to the bare, bare minimum. (no crossbeam)

### Philosophy

- **Pull-based rendering**: Pixels are sampled, not pushed. Nothing computes until coordinates arrive.
- **SIMD as algebra**: `Field` wraps SIMD vectors (AVX-512/NEON/SSE2) transparently. Users write equations, compiler emits assembly.
- **The Fixed Observer**: Camera is at origin. Movement is achieved by warping coordinate space.
- **Types are shaders**: Combinator trees monomorphize into fused kernels with no runtime dispatch.
    - Types are the AST
    - Fields/Jets are the IR
    - `variables.rs` is the symbol table
- **Zero allocations** - No per-frame heap allocation (ping-pong buffer strategy).
- **No copies of unknown sized types** - pixelflow language types are Copy iff they are provably zero sized.
- **Platform on main thread** - Especially macOS Cocoa (Apple requirement).

## Workspace Structure

Cargo workspace with 12 member crates:

| Crate | Purpose |
|-------|---------|
| `pixelflow-core` | SIMD algebra. `Field`, `Manifold`, coordinate variables, ops. Multi-backend (AVX-512/SSE2/NEON/scalar). Edition 2024. |
| `pixelflow-compiler` | Proc-macro compiler: `kernel!` macro, lexer, parser, sema, AST optimization, codegen. Edition 2024. |
| `pixelflow-ir` | Shared IR. `ExprArena` (sole IR), OpKind enum, backend execution traits, JIT manifold. |
| `pixelflow-graphics` | Font loading (TTF, SDF), colors (`Rgba8`, `Color`), rasterization, antialiasing, shapes. |
| `pixelflow-ml` | Neural networks for compiler optimization. NNUE training, HCE extraction, e-graph training. |
| `pixelflow-search` | E-graph optimization. Rewrite rules, saturation, cost extraction, NNUE-guided search. |
| `pixelflow-pipeline` | Training orchestrator. Self-play, unified backward pass, critic server, hyperparameter sweep. |
| `pixelflow-runtime` | Display drivers (macOS Cocoa, headless, Metal, Web WASM), input handling, vsync, render pool. |
| `actor-scheduler` | Priority channels with `troupe!` macro. Control > Management > Data lanes. |
| `actor-scheduler-macros` | Procedural macros for actor system. |
| `core-term` | Terminal application: PTY management, ANSI processing, terminal emulator, key translation. |
| `xtask` | Build tooling: macOS app bundling (`bundle-run`), codegen tasks. |

Agent context files for domain-specific knowledge live in `.claude/agents/`.

## Core Concepts

### The Manifold Abstraction

Everything is a `kernel` - the pixelflow-compiler uses this to generate profunctors from coordinates to values or a morphism on manifolds:
dimap is broken up into covariant `map` and contramap `at`
conditionals are performed using Select or postfix (ManifoldExt) `.select`

### Actor Model

Three-thread architecture for zero-latency input:

Priority lanes: **Control > Management > Data**

Control/Management prioritize latency over throughput.
Control creates backpressure by timing out senders who are too aggressive. If the timeout exceeds a threshold, an error is returned, likely causing a crash.

### Compiler Pipeline

```
Source → Lexer → Parser → Sema → Optimize → Codegen → Rust TokenStream
                   ↓           ↓
               Symbol Table  E-graph + NNUE
```

The compiler uses e-graphs (equality graphs) to find optimal instruction sequences:
1. **Build e-graph** from expression AST
2. **Saturate** by applying rewrite rules (associativity, FMA fusion, etc.)
3. **Extract** minimum-cost implementation using NNUE-guided search

NNUE cost model is inspired by Stockfish: HalfEP features with incremental updates, making evaluation O(rewrite_size).

### ExprArena

`ExprArena` is the sole IR representation everywhere. The old `Expr` (Arc-based tree) is deleted. All paths use arena-based expressions: e-graph extraction, NNUE features, compiler codegen, rewrite rule templates.

## Development Workflow

### Build Commands

```bash
cargo build                       # Auto-detects display driver
cargo build --release             # opt-level=3
cargo build --profile dist        # LTO, strip, codegen-units=1
cargo test --workspace            # All tests
cargo test -p pixelflow-core      # Single crate
cargo bench -p pixelflow-core     # Benchmarks
cargo run --release -p core-term  # Run terminal directly
cargo xtask bundle-run            # macOS bundled app
cargo xtask bundle-run --features profiling  # Flamegraph on exit
```

### Build Profiles

- **dev** - opt-level=1 because deep Manifold recursion causes stack overflow without inlining. panic=abort.
- **release** - opt-level=3, panic=abort
- **bench** - LTO, codegen-units=1
- **dist** - LTO, strip, codegen-units=1, panic=abort

### Workspace Lints

```toml
[workspace.lints.rust]
unused_must_use = "deny"  # Can't ignore Results with `let _ =`

[workspace.lints.clippy]
let_underscore_must_use = "deny"  # Catches `let _ = expr` on #[must_use]
must_use_candidate = "warn"       # Suggests adding #[must_use]
```

All errors must be explicitly handled. No silent failures.

### Toolchain

- **Rust stable** (configured in `rust-toolchain.toml`)
- SIMD backend auto-detected at compile time via `build.rs` and target features
- Platform features automatically selected based on OS

### SIMD Backend Selection

Priority: AVX-512 > SSE2 > NEON > Scalar fallback. Detection via `build.rs` CPU feature probing + `target_feature` flags. See `pixelflow-core/src/backend/`.

## Code Style

- **Clarity over comments** - Refactor unclear code rather than explaining it
- **Rustdoc (`///`)** for public API, **`//`** for WHY not what
- Guard clauses and early returns over deep nesting
- `match` over `else if` for enums
- Functions < 4 arguments (group into structs)
- No boolean arguments (use enums or separate functions)
- Named constants, no magic numbers

## Common Patterns

### Using the `kernel!` Macro

```rust
use pixelflow_compiler::kernel;
use pixelflow_core::{X, Y, Manifold, ManifoldExt};

let circle = kernel!(|cx: f32, cy: f32, r: f32| {
    let dx = X - cx;
    let dy = Y - cy;
    (dx * dx + dy * dy).sqrt() - r
});

let unit_circle = circle(0.0, 0.0, 1.0);
```

Use `kernel_raw!` to skip optimization (for benchmarking exact expression forms).

### Composing Manifolds

```rust
let warped = manifold.warp(|x, y, z, w| (x * 2.0, y * 2.0, z, w));
let selected = mask.select(if_true, if_false);
let circle = (X * X + Y * Y + Z * Z).sqrt();
```

### Actor Message Sending

```rust
handle.send(Message::Control(MyControlMsg))?;    // Highest priority
handle.send(Message::Management(MyMgmtMsg))?;    // Medium
handle.send(Message::Data(MyDataMsg))?;           // Lowest (backpressure)
```

## Platform Notes

### macOS
- Cocoa MUST run on main thread
- `cargo xtask bundle-run` creates `CoreTerm.app`
- PTY I/O: kqueue-based on dedicated thread

### Linux
- X11 via the `x11` crate (feature-gated with `display_x11`)
- Requires: `libx11-dev libxext-dev libxft-dev libfontconfig1-dev libfreetype6-dev libxkbcommon-dev`
- PTY I/O: epoll-based

## Debugging Pitfalls

- **SIMD mismatch between machines**: Check `build.rs` output, verify target features. `RUSTFLAGS="-C target-cpu=native"` to match CPU.
- **Unexpectedly slow**: May be falling back to scalar. Check build output.
- **Cocoa main thread panic**: Ensure `pixelflow_runtime::run()` called from `fn main()`, not a spawned thread.
- **Complex Manifold trait bounds**: Add explicit type annotations, break into named intermediates.
- **"method not found" on Manifold**: Import `use pixelflow_core::Manifold;` and extension traits.

## Performance

- **Target:** 155 FPS at 1080p, ~5ns per pixel
- **Hot paths:** `#[inline(always)]` on eval methods in Manifold implementations
- **Glyph caching:** Categorical morphisms ensure glyphs computed once (`fonts/combinators.rs`)
- **Antialiasing:** Automatic differentiation via `Jet2` dual numbers
- **Monomorphization:** Entire scene compiles to fused SIMD kernels

## Training Pipeline

Unified self-play training loop for the NNUE cost model:

```
GENERATE (Rust self-play) → EXPORT (.pftraj) → CRITIQUE (Python Transformer) → UPDATE (Rust joint backprop) → CHECKPOINT
```

Key entry point: `pixelflow-pipeline/src/bin/train_unified.rs` (supports `--server` mode for fast Optuna trials via unix socket).
