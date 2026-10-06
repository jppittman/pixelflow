# PixelFlow Core

`pixelflow-core` is the `no_std` substrate for PixelFlow: three objects, one verb, and no
other way to turn a program into numbers.

```text
Kernel ──Manifold::compile(extent)──▶ Manifold ──bind(&[(id, buf)])──▶ BoundManifold
       ──Lattice::collapse──▶ DiscreteManifold
```

- A **kernel** is the description — an `ExprArena` fragment with a root, carried by `Kernel`
  (defined in `pixelflow-ir`, re-exported here). It holds no code and no shape.
- A **manifold** is a kernel *compiled at a lattice's shape*: specialized on the extents it
  was compiled for, held behind the global compile cache. It has no `eval` and is not
  batch-shaped.
- A **lattice** is the domain: extents and an origin. The shape is data, not a type — a
  frame, a scanline, a point and an index range are one `Lattice` with different extents.
- **`collapse`** is the verb. It tabulates a bound manifold over a lattice into a
  `DiscreteManifold`, the buffer that *is* a manifold by the representable-functor law
  `index(collapse(f)) = f`.

The loop nest, the invariant hoisting and the register allocation live *inside* the emitted
code, so a collapse is one call per plane — not one per row, and not one per SIMD batch.

## Building a kernel

`Kernel` values compose directly, and `kernel!` (from `pixelflow-compiler`) is the same value
with closure syntax:

```rust
use pixelflow_compiler::kernel;
use pixelflow_core::{Kernel, Lattice};

let circle = kernel!(|cx: f32, cy: f32, radius: f32| {
    let dx = X - cx;
    let dy = Y - cy;
    (dx * dx + dy * dy).sqrt() - radius
});

let sdf: Kernel = circle(32.0, 32.0, 20.0);
let coverage = sdf
    .neg()
    .add(&Kernel::constant(0.5))
    .clamp(&Kernel::constant(0.0), &Kernel::constant(1.0));

let pixels = Lattice::frame(64, 64, 0.0).bake(&coverage);
assert_eq!(pixels.buffer().len(), 64 * 64);
```

`kernel!` parses and checks the DSL, runs e-graph optimization, and returns an uncompiled
arena fragment; with parameters it returns a builder closure that folds them in as constants.
`kernel_raw!` is the same thing without the e-graph, for benchmarking an exact expression
form. Nothing is compiled until a lattice asks for it.

The `Kernel` surface:

- arithmetic, transcendental functions, comparisons, masks, and `select`;
- coordinate substitution with `at`;
- symbolic derivatives (`dwrt`, `dx`, `dy`), resolved before backend emission;
- variadic sums and bounded monoid reductions (`sum_over`, `product_over`, `min_over`, and
  related operations);
- `Bits`, the integer/bitwise half, entered by `trunc_to_int`.

Bounded reductions have static extents. General unbounded recursion is intentionally outside
the language: **the language is a DAG.** A fixed-count iteration is unrolled at construction;
a trip count that must change is a recompile through the cache that already keys on shape.

## Lattices and materialization

A `Lattice` is a finite box over the four coordinate axes — the explicit boundary where a
pure function becomes stored samples.

```rust
use pixelflow_core::Lattice;

let frame = Lattice::frame(800, 600, 0.0);
let scanline = Lattice::scanline(800, 20.0, 0.0, 0.0);
let features = Lattice::index(128);
```

- `bake(&kernel)` compiles a kernel at this lattice's shape, binds nothing, and collapses it
  — the buffer-free case, which is most callers.
- `collapse(&bound)` tabulates a manifold whose declared buffers are already bound. A kernel
  that reads memory (a glyph atlas, a cached coverage plane) goes this way: `Manifold::bind`
  refuses a slot with nothing bound to it, by name.
- The resulting `DiscreteManifold` owns an `f32` buffer and reads back into the language as a
  gather (`DiscreteManifold::kernel`) or a 4-tap blend (`BilinearSampler`).

See [`KERNELS_AND_LATTICES.md`](../docs/designs/KERNELS_AND_LATTICES.md) and
[`2026-07-24-totality-and-the-cost-model.md`](../docs/designs/2026-07-24-totality-and-the-cost-model.md).

## What is not here

**No colour.** This crate knows fields, lattices and integer/bit ops — not RGBA, byte lanes or
pixel formats. A colour output is four channel kernels in `[0, 1]` packed by integer IR ops
that `pixelflow-graphics` composes.

**No SIMD at all.** This crate holds no vector type and no width. The batch a collapse
executes by is the JIT's, decided at process startup by the CPU
(`pixelflow_codegen::isa`); nothing here names a lane or a vector width, and a consumer
never constructs one.

**No expression templates, and no per-batch evaluation.** Manifolds were once zero-sized types
(`X * X + Y * Y`) monomorphized into a fused kernel and evaluated one SIMD batch at a time,
over `Field`, `Jet2` or `Jet3` domains. That tier is retired: the compiler beat it, and the
per-batch call boundary was most of why. Derivatives are symbolic (`Kernel::dx`), not jets.
See [`2026-09-06-kernel-with-a-lattice.md`](../docs/plans/2026-09-06-kernel-with-a-lattice.md).

## Constraints

- `pixelflow-core` is `no_std`; allocation-dependent facilities use `alloc` internally.
- x86-64 and aarch64 only. Rendering goes through the JIT, which has no other backends and no
  interpreter fallback.
- Consumers should not manipulate `ExprArena` directly. Compose `Kernel` values and compile at
  a materialization boundary.
- Do not silently fall back when a kernel cannot compile. `Lattice::bake` fails loudly.
- Do not assume a fixed SIMD lane count in consumer code.

## Test

```bash
cargo test -p pixelflow-core
```

## License

[Apache License 2.0](../LICENSE.md)
