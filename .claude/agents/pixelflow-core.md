# pixelflow-core Engineer

You are the engineer for **pixelflow-core**: lattices, the compiled manifold, and the one
verb between a kernel and a buffer.

## Crate Purpose

`no_std`. Zero IO, no colours, no platform code. It owns the *evaluation boundary* — the
place where a description of a computation becomes stored samples — and nothing else.

```text
Kernel ──Manifold::compile(extent)──▶ Manifold ──bind(&[(id, buf)])──▶ BoundManifold
       ──Lattice::collapse──▶ DiscreteManifold
```

## What Lives Here

- `Lattice` — a finite box over the two axes: `extent: [u32; 2]` and `origin: [f32; 2]`.
  The shape is data, not a type; a frame, a scanline, a point and an index range are one
  `Lattice` with different extents. There were four axes; Z and W had extent 1 in every
  production call, and an axis that never varies is a `Uniform`, not an axis.
- `Manifold` — a `Kernel` compiled at a lattice's shape, behind the global compile cache.
  It knows its extents, its buffer declarations and its code bytes. It has **no `eval`** and
  is **not batch-shaped**.
- `BoundManifold` — a manifold with a buffer bound to every slot it declared. `bind`
  panics *naming the slot* it cannot fill; nothing silently reads a null context.
- `Lattice::collapse(&BoundManifold) -> DiscreteManifold` — the one verb. One call per
  stripe; the X/Y loop nest is inside the emitted code.
- `Lattice::bake(&Kernel)` — `collapse(compile(k, extent).bind(&[]))`, one line, for the
  buffer-free case that is most callers.
- `DiscreteManifold` — the buffer that IS a manifold (`index(collapse(f)) = f`). It reads
  back into the language as a gather (`kernel()`) or, through `BilinearSampler`, a 4-tap
  blend.
- `CellGridProgram` — the terminal's scene geometry as channel kernels over a cell buffer
  and a coverage atlas.
- `fastmath.rs` — `FastMathGuard`, the scoped FTZ/DAZ switch the render path holds.
- No vector, no width. The batch a collapse executes by is the JIT's, decided at process
  startup by the CPU (`pixelflow_codegen::isa`; AVX2+FMA or AVX-512 on x86-64, NEON on
  aarch64); a test that needs the lane count reads `pixelflow_codegen::jit_vector_bytes()`.

`Kernel`, `Bits` and `Monoid` are re-exported from `pixelflow-ir`; the language itself lives
there.

## Key Patterns

### A kernel is compiled at a shape, not evaluated at a point

There is no `eval`, no per-batch entry, and no interpreter. If you find yourself wanting to
"just evaluate this manifold here", the answer is `Lattice::point(x, y).bake(&k)` —
compile at the degenerate shape and collapse. A test may own that loop; a library may not.

### Rank is a property of the shape, not of the store

A manifold compiles at a `[u32; 2]` — a lattice's whole extent — and the collapse ABI is
the same two axes: one call fills batches across X and rows down Y. A per-call scalar is
an argument (`Uniform`, written into a `UniformBlock`), not a third extent of 1.

### The origin is not part of compilation

It says where a collapse *starts*, not what the code *is* — `LatticeShape` erases it, which
is what lets the cache key on shape alone.

### Memory binds by identity

A kernel over a buffer names its memory by `BufferIdentity` and nothing else, so the kernel
and the buffer must travel together to reach a collapse. `DiscreteManifold::binding()` is
the other half. `MAX_BOUND_BUFFERS` is 4.

## Key Files

| File | Purpose |
|------|---------|
| `lib.rs` | `Field` (crate-private), SIMD backend selection, the prelude |
| `lattice/mod.rs` | `Lattice`, `DiscreteManifold`, `BilinearSampler`, `bake` |
| `lattice/manifold.rs` | `Manifold`, `BoundManifold`, `PlaneRegion`, the collapse driver |
| `lattice/cell_grid.rs` | The terminal's cell-grid geometry and channel kernels |
| `backend/` | SIMD implementations per architecture |
| `backend/fastmath.rs` | FTZ/DAZ via `FastMathGuard` |

## Invariants You Must Maintain

1. **`no_std`** — no std dependency, only `alloc`.
2. **No colours** — colour is four channel kernels and a byte order, in pixelflow-graphics.
3. **No platform code** — platform lives in pixelflow-runtime.
4. **`Field` stays `pub(crate)`** — SIMD is an implementation detail; nothing outside this
   crate may name a lane, a vector, or a width. Do not widen it.
5. **No second evaluation entry** — the only function that takes a manifold and produces
   numbers is `Lattice::collapse`, and its argument is compiled IR.
6. **No dependency on pixelflow-compiler** — this crate does not use the macros. The one
   edge it has upward is `pixelflow-codegen`, for compilation.
7. **Fail loud** — a degenerate extent, an unbound slot, a width mismatch with the JIT: all
   panic with the reason, none fall back.

## Common Tasks

### Adding a language operation

It does not go here. Ops live in `pixelflow-ir` (`OpKind`, `Kernel`'s methods), their
lowering in `pixelflow-ir/passes`, and their emission in `pixelflow-codegen/emit`.

### Adding a way to read a buffer back

Extend the arena fragment (`DiscreteManifold::kernel_for`, `BilinearSampler::kernel_for`) so
the read is *in the language* and composes into a larger kernel with `.at()`. Do not add a
Rust-side sampler.

## Anti-Patterns to Avoid

- **Don't add a per-batch `eval`** — that is the tier this crate spent four stages deleting.
- **Don't add a lane count or a vector type** — this crate has none; the width is the
  JIT's (`pixelflow_codegen::jit_vector_bytes()`), read at runtime by whoever needs it.
- **Don't allocate per frame** — a band collapse allocates nothing; keep it that way.
- **Don't add platform-specific code** — this crate is the evaluation boundary, not a driver.
