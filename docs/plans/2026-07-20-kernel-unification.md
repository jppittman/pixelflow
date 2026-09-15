# One Kernel Language: Retiring the Combinator Emitter

**Date:** 2026-07-20
**Status:** Plan of record
**Supersedes:** the dual-backend framing of the 2025-02-21 kernel-jit-feature-parity plan (whose param-baking scope is done).

## Root axiom (2026-07-24): totality — see the design note

Everything in this plan descends from one axiom, written up in
[docs/designs/2026-07-24-totality-and-the-cost-model.md](../designs/2026-07-24-totality-and-the-cost-model.md):

**The kernel language is total. It is not Turing-complete. Every program's cost
is a closed-form function of static extents.** "Estimatable by the cost model"
and "not Turing-complete" are the same statement — the cost model can only be a
total function if there is no unbounded control flow.

The consequences that steer the remaining phases:

- **Discrete domains are coordinate domains.** Index domains (`(l,m)`,
  feature-index, sample-index) join continuous `X/Y/Z/W`. A tensor is a kernel
  over a product of discrete domains, never materialized — pull-sampled like an
  image. (Restores the old discrete fields, now as the typed boundary below.)
- **One new binder: `⊕_{i∈D}`** (monoid-parametrized reduction over a bounded
  domain). Contraction/matmul, softmax, SH projection, the AO integral are all
  this one binder. The Einstein-summation "shared bound index = contract" *is*
  the shape check — it falls out of scoping, not a type system.
- **Kernels stay untyped; fields are typed.** A field is a kernel pinned to a
  discrete domain and reified; types crystallize only there, and only as
  `(domain descriptor, element kind)` with ground extents. The continuous
  algebra never gets a type system.
- **Bounded iteration, never a general `fix`.** Ray-march, recurrence, depth,
  Newton are static unrolls (`iterate[N]`). No `while`/unbounded recursion, ever
  — it deletes the cost model.
- **The `Lattice` dissolves into the typed discrete field** (domain + reify),
  *not* into the iteration operator (those are orthogonal axes). `collapse`/
  `bake` become "reify a kernel on a discrete domain"; the tabulation machinery
  survives as reify's implementation, not as a distinct concept.
- **Type system and cost model are one artifact.** The type carries the extents;
  the cost model multiplies them (`cost(⊕_{i∈D} b) = |D|·cost(b)`). Cost is the
  program re-denoted into the cost semiring; type-check and cost-estimate are one
  pass.
- **Reverse-mode AD is the adjoint of the binder** (reduction's transpose is
  broadcast), so it stays total and needs no autodiff engine beside `Dwrt`.
- **Complex/quaternion/polar are library**, not primitives (bodies + warps over
  gradients), so they do not enter the op set.

This axiom does not change the JIT-first direction below — it names the law the
`Kernel` value, `bake`, and the coming domain/binder work must all obey.

## Course correction (2026-07-23): JIT-first, `Kernel` is the value

"Types are the AST" was JIT-*second*: encode the program in Rust's type system
(ZST combinator/expression templates), lean on rustc for typecheck and
monomorphization, treat the arena as a lowering target. It compiles slowly,
fights the borrow checker, and forces a `Lower`-per-generator bridge that leaks
compiler concepts into consumers (a graphics author should never see
`ExprArena`). **We are inverting to JIT-first:** the arena *is* the program.
The front end (`kernel!`) parses straight to it; our compiler does the
typecheck and monomorphization at codegen; Rust just hosts the compiler and
holds handles.

The consumer value is [`Kernel`](../../pixelflow-ir/src/kernel.rs): a handle
over an arena fragment with a fluent composition surface (`add`/`sqrt`/`min`,
`sum` (variadic), `select`, `at`, `dx`/`dy`) that hides the arena completely.
Graphics builds scene graphs as `Kernel` values — never combinator types,
never `Lower`, never IR. `Lattice::bake(&Kernel)` JIT-compiles once (cache) and
tabulates; `Dwrt` derivatives resolve during codegen (screen-space AA with no
jet domain). This subsumes `Lower` (the ZST→arena bridge is moot when there is
no ZST layer) and, at the end, the ZST combinator library itself. Landed so
far: the `Kernel` value + composition surface + `bake`, proven end to end.
The `2026-07-23-lower-realize-boundary.md` note is superseded by this.

## North star

We don't want two languages. There is one kernel language with one pipeline:

```
parse → sema → e-graph → ExprArena → backend emit
```

The proc-macro and the runtime compiler are the *same pipeline* run at different
times with different e-graph budgets:

- **AOT (`kernel!`):** parse/sema/e-graph run at macro-expansion time with a
  generous saturation budget; the macro emits tokens reconstructing the
  *pre-optimized arena*; machine code is emitted once at load time (kernels are
  constructed at load time — existing project law).
- **Runtime (fused roots, dynamic composition, param re-folding):** same
  pipeline with a bounded e-graph budget. The guided-saturation / rule-masking
  research (docs/plans/2026-07-07-guided-saturation-redesign.md) is the budget
  policy for this tier.

**The combinator emitter (`pixelflow-compiler/src/codegen/`, ZST Let/Var trees)
is the part that goes away.** It is not the long-term fallback. It keeps working
until the parity suite + font goldens prove arena coverage, then it is deleted.
`kernel!` and `kernel_jit!` converge into one macro surface.

## What replaces the combinator emitter's roles

| Role today | Replacement |
|---|---|
| Jet2/Jet3 domain evaluation (font AA, 3D normals) | `Dwrt` symbolic differentiation lowered in pixelflow-ir; derivatives are ordinary expressions compiled by the same backend. NOT the deleted emit-time jet mode (bd47aa7c) — that was an architectural trap. |
| Portable / reference semantics (SSE2, WASM, scalar — no JIT emitters) | The IR interpreter (`pixelflow-ir/src/eval.rs`). Same arena, same semantics, slow. WASM may later get its own arena emitter. |
| Composition (`.at()`, `Sum`, named structs, manifold params) | IR-carrying kernels: every kernel exposes its arena fragment; composition **inlines** that fragment into the host (contramap = Var substitution, Sum = chained Add). This plan and the code call the mechanism "splicing" (`ExprArena::splice`) — see the 2026-09-08 amendment: the name hid a policy decision. |
| Opaque Rust manifolds (DiscreteManifold, dynamic BSP, CachedGlyph) | Enter the IR as bound buffers / ctx-pointer calls (the existing Gather pattern), not as expressions. The `Manifold` trait survives as the composition substrate. |

Numerics are defined once: the JIT's near-libm behavior is canonical. The
combinator backend's inaccurate `sin` (0.5116 vs 0.8415 at x=1; see
pixelflow-compiler/tests/jit_parity.rs header) is a parity bug in the
to-be-retired backend, tracked until that backend dies.

### Amendment (2026-09-08): composition inlines, and inlining is the e-graph's call

This plan called composition "arena splicing." Splicing is the *mechanism*
(`ExprArena::splice`, a copy); the *operation* is *inlining* — composition copies
the callee's body into the caller. Naming it after the memcpy hid the fact that a
policy decision was being taken at all, and taken eagerly, at construction time,
by code that has no cost model to take it with.

That decision is now measured. See
[2026-09-06-egraph-at-production-scale.md](2026-09-06-egraph-at-production-scale.md)
§2: `at`, `select`, and the per-channel `Rgba` operations each re-inline the hit
point, so the tree the e-graph receives is **~40× its own DAG**. Hash-consing at
build folds it back, and that fold alone "lands within 90 classes of the cap" —
"almost the whole budget is spent representing the input", which is why the class
cap binds on iteration 2.

Inlining is a cost-driven choice — inline when the win beats the duplication,
outline when it does not — and an e-graph is precisely the machinery for holding
such a choice open until something can price it. So composition should build a
**reference** to a fragment, and inlining should be a rewrite the extractor
prices like any other. Today the choice is made before anything can price it,
and then the optimizer spends its budget undoing it.

What follows from that:

- Nothing downstream may assume a composed arena is already inlined.
- The ~40× blowup is an artifact of *construction*, not a property of the
  kernels. It is not evidence that these kernels are large.
- An un-inlined reference needs an IR node, and `ExprArena` has none today: the
  `op_table!` roster runs `Var … Uniform` with no call/reference form. Adding
  one is the first piece of that work, not the last.

This amendment is a correction to the vocabulary and to the design it concealed.
It does not block the arena-native macro-tier work, which is about *where* the
e-graph runs, not *what* composition builds.

## Phases

- **P0 — ground truth** *(in flight)*: JIT compile-cost benchmarks inform load-time compile
  budgets; golden coverage-lattice fixtures for the warm-ASCII font set are the
  visual regression guard for every backend switch.
- **P0.5 — e-graph type stability** *(in flight)*: extraction must be
  type-stable across the V/DX/DY projection boundary and must not share
  non-Copy manifold-param subexpressions. Prerequisite for fonts on `kernel!`;
  the defect class itself dies with the emitter, but fonts run on the emitter
  until parity lands.
- **P2 — `lower_dwrt` in pixelflow-ir** *(landed 2026-07-22)*:
  symbolic-differentiation lowering pass
  (peer of `expand_gather`/`expand_reduce`); ir_bridge maps V/DX/DY/DZ →
  `Dwrt` nodes (plus DXX/DXY/DYY as nested `Dwrt`); the codegen panic at a
  surviving `Dwrt` becomes an unreachable
  precondition. Errors loudly on unsupported ops. Acceptance: font-shaped
  coverage expressions JIT-compile and match combinator-over-Jet2 within
  tolerance — see `jit_font_coverage_matches_*` in
  pixelflow-compiler/tests/jit_parity.rs.
  **Tiering (2026-07-22):** the calculus runs in the optimizer first — the
  e-graph `ChainRule` (plus piecewise rules mirroring Jet2 semantics) expands
  and simplifies `Dwrt` at macro-expansion time in ir_bridge, with scalar
  params round-tripped as reserved `Var` indices. `lower_dwrt` is the fallback
  tier for residual `Dwrt` (budget miss / runtime-composed kernels) and errors
  loudly on genuinely non-differentiable ops. Jets are not a fallback: they
  would reintroduce the jet ABI this plan retires, and remain
  combinator-backend-only until P6.
- **P3 — routing scaffolding** *(landed 2026-07-23)*: `kernel!` routes
  eligible bodies through the arena backend behind
  `pixelflow-compiler/arena-backend`; the routing parity suite
  (tests/kernel_routing_parity.rs) runs both ways, and the full workspace is
  green with the feature on. Routing is gated by `is_transparent_routing_safe`:
  signature-eligible AND no derivative projections (their `Field`-domain
  semantics intentionally differ between backends — combinator `DX` over
  `Field` is the fonts' load-bearing "hard step" 0) AND the body references
  coordinates (constant kernels rely on combinator domain polymorphism, e.g.
  scalar-param expressions evaluated over `Jet3`). Emitted JIT paths go
  through a `#[doc(hidden)]` `pixelflow_core::__ir` re-export so consumers
  need no pixelflow-ir dependency. This is transition scaffolding, not an
  end-state decision — default flips when the parity suite + font goldens say
  so.
- **P4 — IR-carrying kernels + arena splicing** *(first slice landed
  2026-07-23)*: `ExprArena::splice` + `substitute_vars_with` (the generic
  contramap — manifold-slot substitution now, coordinate warps later);
  `HasIr { splice_into }` in pixelflow-ir; `kernel_jit!` wrappers carry their
  composed pre-lowering arena and implement `HasIr`; manifold-typed params
  are JIT-eligible — the builder splices each argument's fragment over
  reserved slot variables (`Var(8+k)`) and compiles ONE fused kernel, with
  derivative projections resolving post-splice in the runtime `lower_dwrt`
  tier (see tests/kernel_composition.rs: the AA-ramp-over-SDF font
  architecture as pure arena composition).
  **Tail landed 2026-07-23:** `.at(x, y, z, w)` on manifold params — each
  site gets its own slot (`Var(192+s)`) substituted with a fresh splice
  warped by the site's coordinate expressions; bare references share one
  fragment per param (`Var(128+k)`). Named `kernel!` structs get a
  conditional `HasIr` impl (`Mi: HasIr` bounds; `Field` domain/return only;
  skipped when the arena can't express the body): they stay combinator
  kernels for direct eval and never own JIT memory — a fused root absorbs
  them by splicing, which dissolves the "named structs need a construction
  API" problem. Known limitation: `.at()` through a manifold-bound local
  (`let t = tex; t.at(..)`) is rejected — the AST optimizer eliminates the
  binding while restoring the opaque call. `HasIr` for *anonymous*
  combinator results stays out (unnameable closure types); anonymous
  composition is arena-native via `kernel_jit!`/routing.
  See docs/designs/2026-07-23-jit-orthodoxy-survey.md for the architecture
  survey against V8/LuaJIT/HotSpot/Halide-class compilers.
- **P5 — JIT at the root**: glyph bake through the fused arena
  (`CachedGlyph::new`), then `Lattice::collapse` JIT fast path → frame as one
  bound kernel (M4 in docs/designs/KERNELS_AND_LATTICES.md). Interpreter/
  combinator path remains the fallback until goldens pass.
  **Boundary landed 2026-07-23** (docs/designs/2026-07-23-lower-realize-boundary.md):
  `Lower` (Option-returning, `LowerEnv` for De Bruijn `Let` scopes) is
  implemented per core generator beside each `Manifold` impl — coordinates,
  constants, all unary/binary/ternary ops, comparisons, mask `And`/`Or`,
  `Select`, `Let`/`Var`, `WithContext`/`CtxVar` (splat `Field` contexts baked
  back to consts via a uniformity check), derivative projections (`Dwrt`),
  and `At` (contramap = `substitute_vars_with`) — so **everything the macros
  build lowers by composition**. `Lattice::realize` is the consumer verb:
  lower → global compile cache → collapse via the fused kernel, silent
  generic-collapse fallback when any node declines. `HasIr` is absorbed and
  renamed away; macro wrappers/named structs implement `Lower`; x86 spill
  frames beyond the red zone unblock glyph-scale kernels; JIT wrappers
  compile lazily (spliced-only leaves never pay codegen). Remaining for P5
  proper: restructure `Sum`/`Geometry`/`Glyph` as combinator compositions
  (per the stated end goal — zero hand-written `Manifold` evals in graphics),
  switch `CachedGlyph::new` to `realize`, fused-vs-combinator golden.

  **Correction (2026-07-28):** the boundary described above did not survive
  the 2026-07-24 Kernel-value pivot. pixelflow-core today ships **no**
  per-generator `Lower` impls (the trait lives in pixelflow-ir; its
  implementors are `f32`, the emitter's named-struct bridge, and the
  `kernel_jit!` wrapper), and `Lattice::realize` does not exist — the
  consumer verb is `Lattice::bake(&Kernel)`. A bare `At`/`Select` combinator
  cannot splice into an arena. Plan P6 work from the code, not from the
  paragraph above.
- **P6 — deprecate and delete** the combinator emitter once P2–P5 cover the
  surface (parity suite + goldens green on arena-only).

  **P5 glyph→Kernel landed 2026-07-24:** `kernel_value!` produces a `Kernel`
  value (front end straight to arena, no combinator ZST, no eager JIT);
  `AnalyticalLine`/`AnalyticalQuad` gained `.kernel()`; `glyph_to_kernel` walks
  the parsed glyph into one coverage `Kernel` (`Sum`/`Affine`/`Bounded`/
  `Geometry` all dissolve into `sum`/`at`/`select`); `Font::glyph_kernel*`
  expose it. Goldens vs the combinator `CachedGlyph` (Jet2 AA): 'A', 'O', 'g',
  '8' at 32px all match (mean |Δcoverage| < 0.02, zero inside/outside flips).

  Getting quad glyphs green surfaced and fixed **two JIT soundness bugs** —
  both pre-existing, exposed by arena-composed (spliced) kernels whose schedule
  order differs from ir_bridge's AST order:
  1. **Select-guard mask-before-range hole:** the uniform-mask short-circuit
     guard emits its test at the skipped range's START, reading the mask
     register there — but nothing required the mask to be *computed* by then.
     Arena-composed kernels can schedule an arm before its mask; the guard then
     branched on an uninitialized register (region-shaped garbage in 'O').
     Fixed: an arm is only guarded when the mask's schedule index precedes the
     range (`analyze_select_guards`).
  2. **Unsorted Clamp decomposition:** every emitter inlined
     `max(lo, min(x, hi))`, but the reference semantics
     (`OpKind::eval_ternary`) SORT the bounds first — degenerate `lo > hi`
     ranges (which e-graph extraction and composed kernels produce) silently
     diverged from the interpreter. Fixed by `lowering::expand_clamp`: Clamp
     lowers to the bound-sorting min/max sequence as the last pass of every
     compile entry's lowering chain (per-batch x86/aarch64/avx512 + both
     scanline paths), and the per-ISA Clamp emissions are deleted — a Clamp
     reaching an emitter is now a loud error.
  Also landed: Select with BOTH branches spilled resolves via the two reload
  registers (mask reloads straight into `dst`), replacing the old hard error —
  and fixing its latent mask-reload clobber. The whole class is guarded by
  `pixelflow-ir/tests/spill_pressure.rs` (JIT-vs-interpreter equivalence under
  forced register pressure: both-spilled selects, glyph-shaped sums, nested
  selects, decomposed MulAdd/Clamp, beyond-red-zone frames).

  **Combinator glyph pipeline DELETED 2026-07-24.** There is no portable
  fallback and no interpreter tier (decision: kill it — an architecture
  without an arena backend fails to *build* the font path, loudly, rather
  than rendering through a slow secret code path). What changed:
  - `ttf.rs` compiles a glyph straight to a coverage `Kernel` during parse:
    `Glyph`/`Geometry`/`Sum`/`Affine`/`Line`/`Quad` and every `Manifold`/
    `Lower` impl are gone; `Font::glyph_kernel*` is the only glyph surface.
  - `ttf_curve_analytical.rs` keeps only the coefficient structs + their
    `kernel_value!`-built `.kernel()`; the per-domain `kernel!` stamps and
    `Lower` impls are deleted.
  - `text()` returns one fused `Kernel` (sum of advance-translated glyph
    kernels). `CachedText` composes baked *samplers* directly (a plain
    positioned-glyph struct — the one place composition legitimately stays
    in `Manifold` land, per the opaque-manifolds row above).
  - `CachedGlyph::new` (Jet2/`Antialiased` bake) deleted; `from_kernel` is
    the only bake. `render/aa.rs` (`Antialiased`) deleted — AA is intrinsic
    to the kernels (`Dwrt`), not a wrapper. Graphics' pixelflow-ir main
    dependency dropped (dev-dep only, for the interpreter-reference golden).
  - Goldens are now JIT-vs-interpreter on the same arena ('A','O','g','8');
    the AA behavioral guarantees (ramp width, scale invariance, tangent
    finiteness) are asserted on the kernel path.
  What remains of P6: delete the combinator *emitter* in pixelflow-compiler
  (and the ZST expression-template layer in core) once the remaining
  `kernel!` consumers migrate — fonts no longer depend on it.

  **P6 progress 2026-07-28.** The fonts' actual last emitter dependency was
  the glyph-cache read-back: `Bilinear` — a `kernel!` named struct over
  `DiscreteManifold` — was `CachedGlyph`'s stored sampler. Replaced by
  `pixelflow_core::BilinearSampler`: the 4-tap blend as one bound-memory
  arena (four `Gather`s plus weight arithmetic) compiled through the normal
  backend and bound to the buffer per call via the context-pointer ABI
  (`JitManifold::call_bound` — the first production use of bound buffers).
  `render/bilinear.rs` is deleted; fonts are now emitter-free in library
  code. Also landed: the last load-bearing `-> Jet2` combinator kernel
  (jit_parity.rs's coverage oracle) re-anchored on the IR interpreter, and
  1,053 lines of orphaned compiler sources deleted (`codegen/leveled.rs`,
  `codegen/binding.rs`, `fold.rs`, `lexer.rs`). Remaining emitter consumers
  in library code, ranked by unblocking order: `scene3d.rs`'s Jet3 surface
  structs (the long pole — `Dwrt` has no proven story for the 3D tangent
  frame), `Discrete`/color returns (`ColorSurface`, examples), manifold-param
  and named-struct forms in tests/examples, and `transform.rs`'s `Jet2`
  stamp. The ZST-layer deletion additionally requires migrating ~15
  downstream files that use `At`/`Select`/ops-as-types with no `kernel!`
  involvement (graphics color/shapes/subdiv, runtime coordinator, core-term
  surfaces, pixelflow-ml spherical harmonics).

  **P6 progress 2026-07-29: core-term's scene is a JIT cell grid.** The
  terminal frame was the largest remaining ZST consumer: a per-frame
  `SpatialBSP` of boxed `Select`/`At` combinator trees (one per cell) with
  dynamic dispatch per batch — boxing being the only way the static
  expression-template types could express a runtime-sized, resizable grid.
  Replaced by the tilemap primitive `pixelflow_core::CellGridProgram`
  (lattice/cell_grid.rs, the `BilinearSampler` precedent generalized): four
  channel kernels — cell-index arithmetic, per-cell data gathers, a 4-tap
  atlas read, the fg/bg blend, out-of-grid default background — over two
  bound buffers, ~50 arena nodes each *regardless of grid size*. The
  glyph-side half is `pixelflow_graphics::fonts::GlyphAtlas` (uniform padded
  tiles, bake-on-miss, growth bumps extents), keeping everything with a
  `char` in it out of core. core-term's per-frame work is now filling one
  flat `f32` buffer (10 floats per cell); a resize IS a recompile — four
  small kernels, microseconds, cached until the geometry moves — replacing
  the old dynamic-resize workaround outright. The dead stubbed
  `core-term/src/surface/` module (the pre-Field Select-tree builder that
  nothing called) is deleted with it. core-term's library code now touches
  the ZST layer only through `At`+`ColorCube` at the frame root (the
  runtime's `Manifold<Output = Discrete>` intake); a Kernel-native frame
  lane in pixelflow-runtime (bake channel planes via the collapse loop +
  LICM, pack to RGBA) is the natural next cut and would retire that too.

  **P6 progress 2026-07-29 (second): the frame lane is Kernel-native.**
  The rasterizer's input is now `pixelflow_graphics::render::scene::Scene`:
  `Surface` (an arbitrary color manifold, dense per-batch evaluation — the
  generic lane examples and the no-snapshot background still use) or
  `CellGrid` (a `CellGridFrame` rendered by ONE 2D-collapse call per channel
  per stripe — the internal pixel loop, both LICM prologue scopes active —
  then packed via `Pixel::from_rgba`). `CellGridProgram` compiles
  collapse-mode kernels only; the per-batch `CellGridChannel` manifolds are
  deleted with their `At`+`ColorCube` composition in core-term — the
  terminal's render path now contains no combinator evaluation at all.
  `Scene::CellGrid` renders in device-pixel space by contract (the
  coordinator's DPI contramap applies only to `Surface`; the cell grid
  carries its scale in its geometry, which makes atlas texels ≡ device
  pixels). `AppData::RenderSurfaceU32` — documented as u32-coordinate
  rendering but treated identically to `RenderSurface` by the engine, with
  zero senders — deleted per subtract-before-add.

## Beyond P6 — the language the totality axiom demands

P0–P6 unify the *pipeline*. These phases grow the *language* to cover the AO /
SH / transformer surface, each obeying the root axiom (static extents only).
Sequenced but not yet scheduled:

- **P7 — discrete domains.** Admit bounded discrete coordinate domains beside
  continuous `X/Y/Z/W` in the arena. A kernel over a product of discrete domains
  is a tensor; a kernel over one is a "vector". No storage change — pull-sampled.
  This is the substrate the binder and typed fields both stand on.
- **P8 — the reduction binder `⊕_{i∈D}`** *(landed 2026-07-24)*. The IR
  substrate turned out to already exist — `Nary(Reduce, [combiner, index,
  extent, body])` over `Var(4..8)`, reference semantics in `eval.rs`, and
  `expand_reduce` unrolling wired into every compile entry — with **no front
  door**: nothing outside pixelflow-ir's own tests constructed one. P8 was
  therefore a surface, not a mechanism: `Kernel::{sum_over, product_over,
  max_over, min_over}(extent, |i| body)`. The closure receives the bound index
  as a `Kernel`, so **Rust's scoping is the binder's scoping** — the shape check
  really is variable capture, not a type system, exactly as the axiom note
  predicted. Extent is a static count, so cost stays `|D| × cost(body)`.
  Hygiene was the one real hazard and the first draft had the bug: nested folds
  shared a construction-time placeholder index, so the inner rename captured the
  outer index (`Σ_i Σ_j f(i,j)` evaluated as `Σ_i Σ_j f(j,j)`). Fixed with a
  live-binder counter (`BinderScope`) handing each in-flight fold a unique
  placeholder. Also completed the scalar op surface the real programs need
  (trig/exp/log/pow/hypot/rounding — softmax needs `exp`).
  Tests: `pixelflow-ir/tests/reduction_binder.rs` builds contraction, softmax
  (normalization + monotonicity), an AO-style sample average with `at` warps
  inside the body, and three-deep nesting; a JIT tier checks the unrolled
  machine code against the interpreter's native fold.
- **P9 — typed discrete fields; the `Lattice` dissolves.** A field is a kernel
  pinned to a discrete domain and reified, typed `(domain, element kind)` with
  ground extents. `DiscreteManifold`/`collapse`/`bake` become "reify on a
  discrete domain"; the tabulation machinery survives as reify's implementation.
  Ping-pong = two stored fields swapped; memoization is a field attribute.
- **P10 — bounded iteration `iterate[N]`.** Primitive recursion over a static
  count (ray-march, recurrence, depth, Newton). Orthogonal to P9 — dynamics, not
  domains. Data-dependent behavior via static-max + early-out; cost = worst case.
- **P11 — cost model as re-denotation.** The arena walked into the cost semiring
  (`+` sequencing, `×` iteration over `|D|`), reusing the extents P9's field
  types already carry. Type-check and cost-estimate become one pass; this is the
  cost model the e-graph extraction and the shader budget both consume.
- **P12 — reverse-mode AD as the binder's adjoint.** Reduction's transpose is
  broadcast; backprop through `⊕_{i∈D}` is a symbolic arena transform, staying
  total. Unifies with `Dwrt` (see `ML_AUTODIFF_PIPELINE.md`).

Non-goals reaffirmed by the axiom: no general `fix`/`while`; no type system on
the continuous algebra; complex/quaternion/polar stay library (bodies + warps).

**Demoted:** native multi-domain stamping in `kernel!` (`Field | Jet2 -> Field`
syntax). It would invest in the dying emitter; the fonts' `macro_rules!`
stamping is acceptable transitional scaffolding. Revisit only if the transition
drags.

## Constraints and non-goals

- No per-leaf JIT-at-construction for the thousands of curve-kernel instances:
  compile cost is µs-scale (see P0 results), leaves are bake-time-only, and
  P4/P5 fuse at roots instead.
- A general jet-in/jet-out ABI is not needed: consumers seed screen-space at
  the root (`Antialiased`), so derivative kernels are single-output Field
  kernels containing `Dwrt`. Deferred indefinitely.
- Suckless: pixelflow-ir gets no pixelflow-search dependency; the runtime
  `lower_dwrt` pass is ir-local, while the e-graph `ChainRule` remains the
  compile-time optimizer of the same algebra.
