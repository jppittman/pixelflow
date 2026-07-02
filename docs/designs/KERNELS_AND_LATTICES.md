# Kernels and Lattices: Bound Memory in the Language

**Status:** Draft for review (June 13, 2026)

Builds on: [LATTICE_EVAL.md](LATTICE_EVAL.md) (lattice as representable functor),
[lattice-scheduling-types.md](lattice-scheduling-types.md) (variance as schedule),
[REDUCTIONS_AND_FOLDS.md](REDUCTIONS_AND_FOLDS.md) (reduction as variance shrinking),
[2025-02-21-kernel-jit-feature-parity-design.md](../plans/2025-02-21-kernel-jit-feature-parity-design.md)
(`Param` baking semantics).

## The Two Species

The language has exactly two kinds of value:

- **Kernel** — a pure morphism `ℝ⁴ → ℝ`. Total, memoryless, infinitely
  composable. The e-graph can rewrite it, the NNUE can cost it, the JIT can
  fuse it. This is Halide's *algorithm*.
- **Lattice** — a finite, bounded, discrete domain paired (after collapse)
  with materialized data. The lattice is the *only* place memory exists.
  Collapsing a kernel over a lattice produces a `DiscreteManifold`; indexing
  a `DiscreteManifold` is again a kernel. This is Halide's *schedule*.

The representable-functor law from LATTICE_EVAL.md is the contract:

```
index(collapse(k, domain)) = k        (up to discretization)
```

Memory never enters the language raw. It is always the image of a collapse,
and it is always read back through `index`. Purity is preserved because the
lattice is the only effect, and the effect is *staging* — choosing when a
subcomputation stops being symbolic and becomes data.

Fonts and ML both fall out of this single mechanism:

- A **glyph atlas** is a lattice: expensive winding-number kernels collapsed
  once at load time, sampled per-pixel forever after.
- A **weight matrix** is a lattice: a trained network's parameters collapsed
  (loaded) into an index-space domain, contracted against activations by
  reduction.

There is no font concept and no ML concept in the core language. There are
kernels, lattices, gathers, and reductions.

## What's Broken Today

The naive lattice in `pixelflow-core/src/lattice/` already implements
`collapse`/`collapse_with`/`index` correctly. The problem is that **memory is
invisible below the Rust trait layer**:

1. **The IR has no memory.** `ExprNode` leaves are `Var`/`Const`/`Param`;
   `OpKind` has no load. `DiscreteManifold::eval` and
   `combinators/texture.rs::Texture::eval` call `Field::gather` in
   interpreted Rust. Any kernel that samples a texture is opaque to the
   e-graph, to variance analysis, and to the JIT from that node down. The
   JIT-rendered frame (`fde6e93`) cannot sample a glyph atlas — so the
   terminal's actual hot path is excluded from the entire optimization
   pipeline.
2. **Reduction is interpreted.** `collapse_with` calls `manifold.eval` per
   SIMD batch through a function pointer. An ML inner product pays the
   Rust↔JIT boundary on every step, and nothing can be unrolled because no
   extent is visible to any code generator.
3. **Two redundant memory types.** `Texture` (combinators) and
   `DiscreteManifold` (lattice) duplicate the same floor/clamp/gather
   sampling. `CachedGlyph` wraps `Arc<Texture>`; the lattice docs promise the
   same role for `DiscreteManifold`.
4. **No binding story.** `JitManifold` compiles `(x,y,z,w) → field` with a
   fixed ABI. There is no way to give compiled code a buffer, let alone
   guarantee the buffer outlives the code.

## Design: Memory Enters the IR as Bound Buffers

Three additions to `pixelflow-ir`, all following precedents already in the
codebase.

### 1. Buffer leaves and the buffer table

A new leaf node and a side table on the arena — the memory analogue of the
symbol table:

```rust
// ExprNode gains one leaf variant (fits the 16-byte budget):
ExprNode::Buffer(BufferId)          // BufferId = u16 slot index

// ExprArena gains the declaration table:
pub struct BufferDecl {
    pub width: u32,    // static extent — REQUIRED, part of the IR
    pub height: u32,   // row-major, stride == width
}
buffers: Vec<BufferDecl>,
```

**Extents are static.** A buffer's *shape* is part of the expression, like a
`Const`, even when its *contents* are late-bound. This is the load-bearing
decision: static shape is what lets the emitter fold address arithmetic to
immediates, drop clamps when index ranges are provably in-bounds, and unroll
reduction loops (§3). Different shape ⇒ different kernel, exactly as
different `Param` values ⇒ different kernel.

Variance of a `Buffer` leaf is `CONST` — the buffer doesn't vary; a sample's
variance comes from its indices.

### 2. `OpKind::Gather` — reading a lattice

```rust
// Ternary, children uniform (good for e-graph hashing/CSE):
Ternary(OpKind::Gather, buffer_leaf, x_index, y_index)
```

Semantics: exactly `DiscreteManifold::eval` today — `floor`, clamp to
`[0, extent-1]`, row-major gather. One definition, shared by the interpreter
(`Field::gather`, already implemented in every backend), the JIT emitter
(AVX-512/AVX2 native gather; NEON/SSE2 four scalar loads — the same lowering
the core backends already use), and constant folding.

Algebraic properties:

- **Variance** = union of index variances. A gather with frame-uniform
  indices hoists out of the pixel loop like any other expression — glyph base
  offsets computed once per cell, not per pixel.
- **CSE**: same buffer leaf + same indices ⇒ same e-class. Purity holds
  because bound buffers are immutable during a collapse (§4).
- **Differentiation**: `Dwrt(Gather) = 0` w.r.t. coordinates (nearest-neighbor
  is piecewise constant). Antialiasing continues to come from collapsing
  *with* AA baked in, or from sampling smooth kernels directly; bilinear
  gather is future work (§Open Questions).

### 3. `OpKind::Reduce` — one op, combiner as a parameter

Reduction enters the IR as a **single** op with a **static trip count**. The
combiner is a *child*, not part of the opcode — one `Reduce` covers every
monoid, and because the combiner is a subexpression it can later generalize
from a monoid marker to an arbitrary fold function (the standard-library
direction). Encoded n-ary, following the `Dwrt` "op-params-as-`Const`-children"
precedent:

```rust
// Nary(Reduce, [Const(combiner), Const(reduce_var), Const(extent), body])
//   combiner:   OpKind index of the monoid (Add/Mul/Min/Max)
//   reduce_var: which variable the reduction binds (indices 4..8)
//   extent:     static trip count — the "bound" in bound memory
//   body:       expression that may reference Var(reduce_var)
```

Rejected: four `ReduceAdd/Mul/Min/Max` opcodes. Baking the combiner into the
opcode proliferates the IR and forecloses arbitrary combiners. Monoid metadata
lives on the base ops instead (`OpKind::monoid_identity()`, `is_monoid()`).

**Binding:** reduction variables take indices 4..=7. `Variance` is a `u8`
with only the low nibble used; the high nibble tracks reduce-var dependence
with no representation change. `Reduce` removes its variable's bit from the
body's variance — the IR-level statement of "reduction is variance-shrinking"
from REDUCTIONS_AND_FOLDS.md. Four reduction depths is enough: layers
materialize between collapses, so nesting stays shallow by construction.

**This is where loop unrolling comes from.** The emitter sees
`extent = 64`, a body cost from the NNUE, and chooses: full unroll below a
threshold, otherwise a loop unrolled by some factor. Because the extent is in
the IR, the choice is local, costed, and needs no runtime trip-count checks
or tails beyond the SIMD lane tail.

The decisive interaction is **unroll → const-fold through gather**. Once the
reduce variable is unrolled, `Gather(W, Const(i), j)` has a constant index
into a buffer of known shape: the address folds to an immediate. If the
buffer is additionally *frozen* (§4), the load folds to the weight value
itself — an NNUE layer compiles to a pure FMA chain with weights baked into
the instruction stream. That is the sentence "the JIT needs bound memory so
the kernels can be loop unrolled," made precise.

### 4. The binding model: `Param` semantics, extended to memory

`Expr::Param` established the model: *parameters are baked at build time;
different values = different kernel; no cache, caller decides lifetime.*
Buffers extend it. A `BufferId` is a slot; JIT compilation takes a binding
table:

```rust
pub enum Binding {
    /// Contents immutable for the life of the kernel. Address AND values
    /// may be folded to immediates. (Glyph atlas, trained weights, LUTs.)
    Frozen(Arc<DiscreteManifold>),
    /// Address-stable, contents may change BETWEEN collapses (never during).
    /// Address folds to an immediate; values never do. (Terminal cell grid,
    /// ping-pong buffers.)
    Pinned(Arc<DiscreteManifold>),
}

pub struct BindingTable { slots: Vec<Binding> }  // indexed by BufferId
```

- Binding **checks declared extents** against the actual buffer and fails
  loudly on mismatch (consistent with workspace `unused_must_use = "deny"` —
  no silent fallback).
- `JitManifold` **holds the `Arc`s**, so bound memory provably outlives the
  executable code. Send + Sync is preserved: frozen contents are immutable,
  pinned contents follow the existing ping-pong discipline (the renderer
  already never mutates a buffer mid-frame).
- Rebinding = recompiling. JIT compilation is microseconds; frames are
  milliseconds. For data that changes per frame (the cell grid) use `Pinned`
  and never recompile; recompile only when *shape* or *frozen contents*
  change (font size change, theme change) — exactly the events that already
  invalidate the glyph cache today.

`DiscreteManifold` changes to make this safe: the buffer becomes
`Arc<[f32]>` (cheap clone, stable address, immutable-after-collapse), with
`freeze()` as the explicit transition out of the mutable build phase.

### The Halide correspondence

| Halide | PixelFlow |
|---|---|
| `Func` (algorithm) | kernel (manifold expression) |
| `realize` | `Lattice::collapse` |
| `compute_at` / `store_at` | nested collapses — materialize at a chosen scope |
| `bound()` | static extents in `BufferDecl` / `Reduce` |
| `RDom` | `Reduce` node / `collapse_with` |
| `unroll` | automatic from static extents, NNUE-costed |
| `vectorize` | implicit (SIMD lanes are an implementation detail) |
| schedule language | none — the lattice tower IS the schedule |

The division of labor from lattice-scheduling-types.md is unchanged and now
complete: **variance decides what factors, the lattice decides where it
materializes, bound extents decide how the loop is emitted.** A user
"schedules" by choosing which subexpressions to collapse onto which lattices;
everything below that is the compiler's problem.

## Fonts on Lattices

Status quo: `Glyph` (pure winding-number kernel) → `CachedGlyph` rasterizes
via `execute()` into `Arc<Texture>` → `Texture::eval` gathers in interpreted
Rust → composed with colors *outside* the JIT.

Target:

1. `Texture` is absorbed by `DiscreteManifold`. One memory type, one
   sampling semantics, one `Gather` lowering. (`CachedGlyph` keeps its shape;
   only its payload type changes.)
2. The glyph atlas is a **frozen lattice**: glyph kernels collapsed over a
   `FrameLattice` at load/size-change time — the existing cache morphism,
   unchanged categorically, but now *visible to the compiler*.
3. The terminal frame becomes **one bound kernel**:

```
cell      = Gather(grid,  col(X), row(Y))      -- Pinned: emulator updates it
glyph_uv  = atlas coords from cell + fract(...)
coverage  = Gather(atlas, u, v)                -- Frozen: baked address
color     = coverage.select(fg, bg)
```

Variance does the scheduling automatically: `cell` and the glyph base offset
are uniform within a cell row's scanline span (hoisted), only the inner
`coverage` gather and select vary per pixel. Today none of this path can go
through the JIT; after this design, the entire frame is a single fused kernel
sampling two buffers.

## ML on Lattices

Status quo: `IndexLattice1D/2D` + `collapse_axis` express matmul as a lattice
collapse (per the team1 extensions plan), evaluated interpretively;
`pixelflow-ml` networks evaluate via ad-hoc Rust.

Target — a layer is a kernel over the output index `j`:

```
out(j) = act( Gather(bias, j, 0)
            + ReduceAdd(i, N, Gather(W, i, j) * Gather(input, i, 0)) )
```

collapsed over `Lattice::index(M)` to produce the next layer's
(pinned) input. With `N` static and weights frozen:

- small `N` → fully unrolled FMA chain, weights as immediates;
- large `N` → unrolled-by-lanes loop with immediate base addresses, no bounds
  checks, no Rust↔JIT boundary inside the product.

**Non-goal (v1):** training through `Gather`. Gradients w.r.t. buffer
*contents* require scatter (a write effect the language doesn't have).
Training stays on the existing interpreted/`Dwrt` path; this design covers
inference. Scatter-as-adjoint-of-gather is future work and should arrive, if
ever, as a lattice operation (collapse of a gradient kernel), not as a raw
store op.

## Changes by Crate

| Crate | Change |
|---|---|
| `pixelflow-ir` | `ExprNode::Buffer`, `BufferDecl` table on arena; `OpKind::Gather`, `OpKind::Reduce*` (+ arity/cost/name/emit metadata); variance: high-nibble reduce vars, gather = union of indices, reduce removes its var; emitter: gather lowering per backend, reduce lowering with unroll heuristics, address/value const-folding; `BindingTable`; `JitManifold` holds `Arc`s |
| `pixelflow-core` | `DiscreteManifold` buffer → `Arc<[f32]>` + `freeze()`; delete `combinators/texture.rs` in favor of `DiscreteManifold`; lattice `collapse` JIT fast path (the TODO at `lattice/mod.rs:13-15`) |
| `pixelflow-compiler` | `kernel!` grows lattice parameters: `kernel!(\|atlas: lattice<W, H>\| atlas(X * 2.0, Y))` → `Gather`; `sum(i, N, body)` → `Reduce`; sema rejects out-of-scope reduce vars |
| `pixelflow-search` | rewrite rules: gather CSE, hoisting uniform-index gathers, reduce linearity (`Σ(a·f + g) = a·Σf + Σg`), reduce-of-select; NNUE features for the new ops |
| `pixelflow-graphics` | fonts on `DiscreteManifold`; atlas as frozen lattice |
| `core-term` | frame as one bound kernel: grid `Pinned`, atlas `Frozen` |

## Milestones (independently landable)

1. **M1 — Memory in the IR.** `Buffer` leaf, `BufferDecl`, `Gather`,
   variance, interpreter execution via existing `Field::gather`. Tests:
   gather round-trips `DiscreteManifold::eval`.
2. **M2 — JIT binding.** `BindingTable`, `Arc` lifetimes, gather emission on
   all backends, address immediates, Frozen value folding. Tests: JIT vs
   interpreter equivalence on sampled kernels.
3. **M3 — Reduce + unrolling.** `Reduce*` ops, high-nibble variance, emitter
   unroll heuristics, `collapse_axis` lowering to the JIT. Tests: dot
   product/matmul vs interpreted lattice results; benchmark unrolled vs
   interpreted inner product.
4. **M4 — Fonts unification.** `Texture` → `DiscreteManifold`; atlas sampled
   in-kernel; terminal frame through the JIT end to end.
5. **M5 — ML inference.** NNUE forward pass as bound kernels; benchmark
   against the ad-hoc path.

## Open Questions

- ~~**Surface naming.**~~ **Resolved (Jun 13, 2026): no types in the name.**
  There is one `Lattice` struct — `extent: [u32; 4]`, `origin: [f32; 4]` —
  and the former `FrameLattice`/`ScanlineLattice`/`PointLattice`/
  `IndexLattice1D`/`IndexLattice2D` zoo is deleted in favor of constructors
  (`Lattice::frame/scanline/point/index/index2`). Shape is data, not a type:
  extents only need to be static at *JIT-compile* time (when the kernel is
  specialized), never at Rust-compile time, so there are no const generics
  and no dimension suffixes. The per-lane vs horizontal reduction split that
  used to be type-encoded is now method-encoded (`collapse_with` vs
  `collapse_scalar`). The kernel-side surface is likewise macro-level
  (`lattice` parameters in `kernel!`), not type-level. `DiscreteManifold`
  stays as the name for collapsed data.
- **Bilinear gather.** Nearest-neighbor has zero derivative; a `Gather2`
  (2×2 lerp) would be smooth and jet-friendly for texture-space AA. Defer
  until a concrete consumer appears — glyph AA is currently baked at collapse
  time.
- **Buffer dtype.** f32-only in v1. `Rgba8` frames stay on the existing
  discrete pipeline; channel-planar f32 lattices cover fonts/ML.
- **Unroll thresholds.** Initial heuristic (e.g., full unroll when
  `extent × body_ops ≤ ~256`), to be replaced by NNUE costing once M3
  generates training data — the same Judge/Guide loop as scalar extraction.
