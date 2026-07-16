# PixelFlow — Pull-Based Functional Graphics on CPU SIMD

**A GPU-free graphics engine proving that elegant algebraic abstractions can achieve 155 FPS at 1080p on pure CPU.**

PixelFlow is a research project demonstrating a novel paradigm for real-time graphics: **pull-based rendering** with **SIMD as algebra**. Nothing computes until a coordinate arrives. Pixels are sampled, not pushed. The type system builds compute graphs. The compiler emits optimal vector assembly.

**`core-term`** is the first consumer application—a high-performance, correct terminal emulator built entirely on PixelFlow.

## Vision

PixelFlow answers three questions:

1. **What if we stopped pushing pixels and started pulling them?** In traditional rasterization, every primitive computes its contribution to every pixel. In PixelFlow, pixels ask "what color am I?" and only that computation happens.

2. **What if SIMD was algebra, not an optimization?** Instead of SIMD as a lower-level concern, PixelFlow treats SIMD vectors as the natural representation of continuous fields over coordinates.

3. **What if the type system compiled graphics?** Expressions written with the `kernel!` macro are compiled through an e-graph optimizer and codegen pipeline, monomorphizing into fused kernels with zero runtime dispatch.

## The Stack

```
┌─────────────────────────────────────────┐
│          core-term (App)                │  First consumer: Terminal emulator
├─────────────────────────────────────────┤
│ pixelflow-runtime (Platform)            │  Cocoa/X11/Web display drivers,
│ actor-scheduler (Concurrency)           │  input handling, render loop
├─────────────────────────────────────────┤
│ pixelflow-graphics (Materialization)    │  Colors, fonts, compositing,
│                                         │  rasterization to pixels
├─────────────────────────────────────────┤
│ pixelflow-compiler (Frontend)           │  kernel! macro: lexer, parser, sema,
│                                         │  codegen
│ pixelflow-search (Optimization)         │  E-graph saturation, rewrite rules,
│                                         │  cost-model extraction
│ pixelflow-ir (IR)                       │  ExprArena, OpKind, backend traits
├─────────────────────────────────────────┤
│ pixelflow-core (Algebra)                │  Field, Manifold, coordinates,
│                                         │  no_std, SIMD abstraction
└─────────────────────────────────────────┘
```

## Crates at a Glance

| Crate | Edition | Purpose |
|-------|---------|---------|
| `pixelflow-core` | 2024 | Pure algebra. `Field`, `Manifold`, coordinate variables. No I/O, no colors. |
| `pixelflow-compiler` | 2024 | Proc-macro compiler: `kernel!` macro, lexer, parser, sema, codegen. |
| `pixelflow-ir` | 2024 | Shared IR. `ExprArena`, `OpKind`, backend execution traits, JIT manifold. |
| `pixelflow-search` | 2024 | E-graph optimization: rewrite rules, saturation, cost-model extraction. |
| `pixelflow-pipeline` | 2024 | Cost-model tooling: JIT bench harness, corpus generation, extraction benchmarks. |
| `pixelflow-graphics` | 2021 | Colors (`Rgba8`), fonts, rasterization, antialiasing via automatic differentiation. |
| `pixelflow-ml` | 2024 | Graphics ML experiments (harmonic attention, spherical-harmonic feature maps). |
| `pixelflow-runtime` | 2021 | Display drivers (Cocoa/X11/Web), input handling, render orchestration. |
| `actor-scheduler` | 2024 | Priority message passing with `troupe!` macro for lock-free concurrent actors. |
| `actor-scheduler-macros` | 2024 | Procedural macros for the actor system. |
| `core-term` | 2021 | Terminal emulator. ANSI parsing, PTY management, state machine. The first PixelFlow consumer. |
| `xtask` | 2021 | Build tooling: macOS app bundling, codegen tasks. |

### Compiler Pipeline

Code written with the `kernel!` macro doesn't compile directly to assembly—it flows through an optimization pipeline:

```
Source → Lexer → Parser → Sema → Optimize (e-graph) → Codegen → Rust TokenStream
```

The optimizer builds an e-graph from the expression, saturates it with rewrite rules (associativity, FMA fusion, etc.), and extracts the minimum-cost implementation using a handwritten, latency-prior cost model (the default). A learned NNUE cost model exists as an opt-in experiment but hasn't beaten the handwritten model in benchmarks, so it isn't the default. See CLAUDE.md's "Cost-Model Training" section for details.

## The Manifold Abstraction

Everything in PixelFlow is a `Manifold`—a domain-generic function from coordinates to a value:

```rust
trait Manifold<P = (Field, Field, Field, Field)>: Send + Sync {
    type Output;
    fn eval(&self, p: P) -> Self::Output;
}
```

`P` is the domain the manifold operates on—`(Field, Field)` for a 2D kernel, `(Jet2, Jet2)` for 2D with automatic differentiation, and so on. A 2D kernel only pays for the coordinates it declares; there's no tax for unused dimensions.

Manifolds compose via operator overloading, and the type tree *is* the compute graph—when `eval` is called, the compiler monomorphizes and inlines the whole tree into a single fused kernel.

**Writing manifolds by hand is a legacy pattern.** The intended way to write PixelFlow code today is the `kernel!` macro, which compiles an expression through the e-graph optimizer and codegen pipeline instead of relying on hand-composed combinator types (see [Extending PixelFlow](#extending-pixelflow) below).

## Performance

- **Throughput:** 155 FPS at 1080p (~5 nanoseconds per pixel)
- **Backend:** Pure CPU, no GPU required. SIMD: AVX-512, SSE2, NEON
- **Memory:** Zero allocation per frame (ping-pong buffer strategy)
- **Compilation:** Entire scene monomorphizes into fused kernels
- **Latency:** <5ms input-to-render (actor model)

## Getting Started with PixelFlow

### Documentation

- **[CLAUDE.md](CLAUDE.md)** — Architectural constraints, workspace structure, and development guidelines
- **[docs/STYLE.md](docs/STYLE.md)** — Code style guide and design principles
- **[docs/designs/](docs/designs/)** — Design docs for the compiler, e-graph search, and actor scheduler
- **[docs/plans/](docs/plans/)** — In-flight design and migration plans

### Prerequisites

- **Rust:** Stable (see `rust-toolchain.toml`)
- **Platform dependencies:**
  - **macOS:** Native Cocoa support
  - **Linux:** X11 development headers
    ```bash
    sudo apt-get install libx11-dev libxext-dev libxft-dev libfontconfig1-dev libfreetype6-dev libxkbcommon-dev
    ```

### Building

Standard Rust build:
```bash
cargo build --release
```

Run tests:
```bash
cargo test
```

Run benchmarks:
```bash
cargo bench -p pixelflow-core
cargo bench -p pixelflow-graphics
```

### Running core-term

#### Standard
```bash
cargo run --release -p core-term
```

#### macOS (bundled app)
```bash
cargo xtask bundle-run
```
Builds and launches `CoreTerm.app` with native macOS integration.

#### With Profiling
```bash
cargo xtask bundle-run --features profiling
```
Writes flamegraph on exit.

## Architecture Overview

### Pull-Based Rendering

Traditional GPU pipeline: **push** every primitive to every pixel.

PixelFlow: **pull** each pixel samples what it needs.

```rust
// A pixel asks: "What color am I?"
// The manifold computes only what's necessary.
let color = manifold.eval((x, y));
```

This eliminates:
- Overdraw
- Primitive list parsing
- Conditional branching in the hot loop

### Actor Model for Zero-Latency Input

Three-thread architecture:

```
Main Thread (Display)          Orchestrator Thread          PTY I/O Thread
├─ Cocoa/X11 event loop       ├─ Terminal state machine   ├─ kqueue/epoll
├─ Platform events            ├─ ANSI parser              ├─ PTY read/write
└─ Render commands            └─ Scene generation         └─ I/O events
    (BackendEvent)                (Render command)            (IOEvent)
         ↓                              ↓                          ↓
    Three-lane priority channel (Control > Management > Data)
```

Input latency is decoupled from render latency.

### Crate Separation Philosophy

PixelFlow is extracted from core-term because:

1. **No terminal logic in PixelFlow.** Graphics library stays general-purpose.
2. **Gradual extraction:** Each crate is independently useful.
3. **Future applications:** PixelFlow can power other renderers (UI toolkits, games, simulations).

## Extending PixelFlow

### Writing a Kernel

The `kernel!` macro is the intended way to write PixelFlow code: it compiles your expression through the e-graph optimizer instead of relying on hand-composed combinator types.

```rust
use pixelflow_compiler::kernel;
use pixelflow_core::{Field, Manifold};

// A parameterized kernel: instantiating it with concrete params returns a manifold
let circle = kernel!(|cx: f32, cy: f32, r: f32| {
    let dx = X - cx;
    let dy = Y - cy;
    (dx * dx + dy * dy).sqrt() - r
});

let unit_circle = circle(0.0, 0.0, 1.0);

// Evaluate at a point
let p = (Field::from(1.5), Field::from(2.0), Field::from(0.0), Field::from(0.0));
let result = unit_circle.eval(p);
```

Hand-writing `Manifold` impls directly (composing `X`, `Y` with operators and combinators like `.at()` or `.select()`) still works and is used internally, but it's a legacy pattern being phased out in favor of `kernel!`.

## Contributing

See [CLAUDE.md](CLAUDE.md) for architectural constraints and development guidelines.

Key points:
- **Code style:** Follow Rust idioms. See [STYLE.md](docs/STYLE.md).
- **No magic in PixelFlow:** Keep the algebra pure and portable.
- **Tests:** Public API changes require test updates.

## Research Context

PixelFlow is inspired by:
- [Conal Elliott's denotational design](http://conal.net/papers/icfp97/)
- [Halide](https://halide-lang.org/) (pull-based, algebraic composition)
- [Elm](https://elm-lang.org/) and pure functional graphics
- [Seamless.js](https://github.com/scttnlsn/seamless) (algebraic surfaces)

The goal: prove that **pure algebra** scales to real-time graphics without GPU compromise.

## License

[MIT License](LICENSE.md)
