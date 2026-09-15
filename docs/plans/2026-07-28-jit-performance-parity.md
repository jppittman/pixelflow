> **Superseded in scope (2026-09-06).** Its "Surface" lane is explicitly superseded
> by [`2026-09-06-kernel-with-a-lattice.md`](2026-09-06-kernel-with-a-lattice.md),
> which states the end state; `pixelflow-core/src/jit.rs` and
> `pixelflow-runtime/examples/bench_psychedelic.rs` are deleted. The per-ISA
> scoping notes below (constant materialization, `vzeroupper`, select-vs-blend)
> were never a plan of record and several are overtaken by the S3/S3b schedule
> work — check `docs/results/` before acting on one.

# JIT Performance Parity: Scoping

**Date:** 2026-07-28
**Status:** Scoping (pre-implementation)
**Context:** feature parity (P6 of docs/plans/2026-07-20-kernel-unification.md)
is close; this doc scopes the path back to *performance* parity with the
monomorphized expression-template path the JIT replaced. Prior art:
docs/designs/2026-07-25-two-level-ir-and-backend-completeness.md (Parts 2b and
3 remain the correct deep-dive; some of its line references have drifted).

## The headline findings

1. **The loop-in-JIT kernel already exists and nobody calls it.**
   `compile_collapse_avx512` + `emit_avx512_collapse_loop`
   (pixelflow-ir/src/backend/emit/mod.rs) runs the full Stage-1 lowering
   pipeline, wraps `emit_dag_body` in an internal loop, sets the spill frame up
   once, stores straight to an output buffer, keeps `rdi` = ctx so gather
   works, and is tested against the interpreter (a 32-wide matmul in one
   call). It is AVX-512-only, 1-D (Y/Z/W zeroed), takes X from a
   caller-materialized `xs` array instead of generating it as an induction
   value, has no tail handling, and has zero production callers. The "`over`"
   work is generalization and wiring, not invention.

2. **Runtime-built `Kernel`s never see the optimizer — closed 2026-07-28 for
   the Reduce-free population.** `kernel!`/`kernel_jit!` run e-graph
   saturation (FMA fusion, CSE) at macro-expansion time, but
   `Lattice::bake(&Kernel)` and `jit_cache::compile_cached` called
   `compile_arena_dag`/`compile_collapse` directly — no saturation, no CSE, no
   FMA fusion. `pixelflow_search::runtime::optimize_runtime_arena` is the same
   pipeline (shared `SaturationConfig`/`ExtractionPolicy`, moved out of
   `pixelflow-compiler` so both tiers use one implementation) applied directly
   to a runtime arena, and `Lattice::bake` now calls it before compiling. It
   bails out (arena compiles unchanged) on `Buffer`/`Gather`/`Reduce` — memory
   ops and the `Kernel::over` binder aren't e-graph-modeled yet — which covers
   today's actual runtime population exactly: glyph coverage kernels (pure
   arithmetic + `Dwrt`, the highest-volume case) now get CSE/FMA fusion;
   `BilinearSampler` (`Gather`) and any future `Kernel::over` composition
   correctly skip until binder-aware rewriting exists (tracked, not silently
   wrong). Transcendental expansion's un-fused Chebyshev polys are a separate,
   still-open gap (item 3 below) — the e-graph runs *before* that lowering, so
   it can't fuse multiplies the lowering hasn't produced yet.

## Current call path and per-call overhead

One JIT call computes one SIMD batch (one pixel per lane); the Rust loop in
`Lattice::collapse` (pixelflow-core/src/lattice/mod.rs) iterates. Costs per
batch:

- Indirect call through a fn pointer re-loaded from
  `Arc<JitManifold> → ExecutableCode` every batch; opaque to the caller's
  scheduler.
- The `extern "C"` boundary makes every vector register caller-saved
  (SysV/AAPCS64), so all loop-carried SIMD state (`x_field`, `step`, Y/Z/W)
  spills and reloads around every call. This dominant cost is invisible in the
  JIT'd bytes.
- Loop-invariant recomputation: Y/Z/W-only subexpressions re-execute per
  batch. Variance analysis + hoisting exist (`variance.rs`) but feed nothing
  in production yet — the collapse loop's Y dimension (item 2 in "Remaining,
  in order" below) is where this substrate gets its first real consumer.
  (Historical note: this previously said the hoisting fed an
  `arena_to_hoisted_schedule`-driven aarch64 scanline path; that whole path
  was deleted 2026-07-28 along with the rest of the scanline tier — see the
  `over` plan section.)
- Constant re-materialization per call (AVX-512: stack round-trip per
  occurrence; aarch64: literal-pool re-anchor).
- `Lattice::collapse` additionally stores each batch to a scratch array and
  `copy_from_slice`s it into the row — an extra store+memcpy per batch that the
  collapse kernel's direct `emit_store` already avoids. (Trivially fixable
  independent of everything else.)

Where the loops live: `Lattice::collapse`/`fold_lanes`/`collapse_axis*`
(pixelflow-core/src/lattice/mod.rs) — production; the combinator rasterizer's
`execute_stripe` (pixelflow-graphics/src/render/rasterizer/mod.rs) is the
monomorphized loop we're chasing. Pathological sites calling
`Lattice::point(..).bake(..)` per single pixel exist in tests.

## The `over` plan (loop-in-JIT)

**Rule of construction (2026-07-28, project direction): one implementation per
job, replacement in place.** The new thing takes the existing good name; the
superseded thing is deleted in the same change, never left beside it as a
slow parallel path.

**Landed 2026-07-28:**

1. `IsaBackend::emit_collapse_loop` — each backend (SSE2/AVX2/AVX-512/NEON)
   wraps the shared `emit_dag_body` in its own ~60-line loop scaffold. X is an
   induction value (caller passes lane-sequential `x0`; the kernel steps it by
   the batch width), Y/Z/W are loop-invariant arguments, coordinate state
   lives in stack slots above the body's spill frame, the ctx register stays
   reserved for gather. `compile_collapse` replaced `compile_collapse_avx512`
   in place; the ABI is `CollapseKernelFn(ctx, out, groups, x0, y, z, w)`.
2. `Lattice::bake` tabulates through it — one call per row instead of one per
   batch, row tails through a one-batch scratch. The mode-tagged
   `jit_cache::compile_collapse_cached` shares compiles.
3. The superseded kernels are gone (−3,240 lines): both scanline compile
   entries and their hand-rolled loops, the x86 Sethi-Ullman tree-walk
   emitter, `ScanlineJitManifold`/`ScanlineKernelFn`, the aarch64
   v8-v15 hoisted-prologue machinery, `pixelflow-core/src/jit.rs`, and the
   two pipeline benches that measured the deleted path.
   `tests/collapse_loop.rs` pins collapse output bit-exact against the
   per-batch kernel (and the interpreter) across arithmetic, selects,
   transcendentals, gather, reduce, and forced spills.

**Landed 2026-07-28 (second wave): X-invariant LICM in the collapse kernel.**

The FreeType gap is algorithmic: FreeType solves each scanline's curve
intersections once (the active edge list), while dense evaluation re-derived
them per pixel batch. But post-Dwrt-lowering, a winding kernel's per-scanline
math — band masks (`Y >= y_min`), curve roots `f(Y)`, gradient factors
`f'(Y)` — is exactly the X-*invariant* sub-DAG, so the active-edge economics
is a compiler pass, not a rasterizer rewrite:

- `plan_collapse_hoist` (emit/mod.rs) partitions the collapse schedule by
  `Variance` (`schedule_variance` — the `variance.rs` substrate's first
  production consumer). A hoist root is an X-invariant, non-leaf value with an
  X-dependent consumer; the prologue schedule is the roots' operand closure,
  the body schedule treats roots as pre-spilled leaves.
- The prologue is emitted once per row-call by the same `emit_dag_body`
  driver (`HoistCtx::Prologue`), parking each root in a dedicated stack slot
  above the scaffold's coordinate slots; the body (`HoistCtx::Body`) skips
  the roots' defs and its consumers reload through the ordinary spill
  machinery — no new instruction forms, no per-backend emission changes, all
  four scaffolds just splice the prologue bytes and reserve the slots.
- Both emissions share one frame region (`max` of the two spill frames,
  pre-sized via the pure `linear_scan`/`FrameLayout` pair) with a 144-byte
  floor forcing x86's SSE2 backend out of red-zone mode so hoist-slot
  addressing agrees on both sides.
- Two deliberate scope cuts: select-guard short-circuits are disabled inside
  the prologue (a uniform mask would skip a hoist root's def, leaving its
  slot garbage for the whole row — and the prologue runs once, so guards buy
  nothing), and `Gather`s never hoist (hoisting would move a load out of any
  guard arm it sits in; arithmetic speculation is free, memory speculation is
  a policy decision deferred until something needs it).
- Hoisting reorders nothing within a value's own computation, so results are
  bit-exact against the per-batch kernel: pinned by four new cases in
  `tests/collapse_loop.rs` (hoisted sqrt/div chains, fully-invariant root
  degenerating to a store loop, a hoisted select whose slot must fill
  unguarded, prologue-side spill pressure) across all four ISAs, plus the
  glyph goldens.

**Remaining, in order:**

1. **`Reduce`'s loop lowering.** `expand_reduce` statically unrolls; a large
   extent needs a real loop. Per the rule above this is a *replacement* of
   the unroll, not a sibling keyed on extent — small trip counts may still
   unroll, but as a decision inside the one lowering, not as two lowerings.
   Semantic note: `over` is `⊕_i` (eliminates a dimension); the render loop
   is `for i {{ out[i] = f(i) }}` — the collapse scaffold is the second shape,
   a `Reduce` loop is the first, and they can share the backend loop
   primitives.
2. **The Y loop (2D collapse) — landed 2026-07-29.** `CollapseKernelFn` now
   accepts `rows` and `row_skip_bytes`; all four ISA scaffolds wrap the X loop
   in an outer loop that resets X, advances Y by 1.0, and skips a caller-owned
   scalar tail without overwriting it. `Lattice::bake` submits the full-width
   region of each Z/W plane in one call and keeps only the final partial SIMD
   batch per row on the scratch path. LICM is two-level: X/Y-invariant roots
   (Z/W-only or constant) run once per plane call, while Y-dependent/X-
   invariant roots run once per row. `tests/collapse_loop.rs` pins X reset, Y
   induction, row gaps, both hoist scopes, and interpreter parity; the existing
   collapse suite still covers selects, gathers, transcendentals, and spills.
3. **The e-graph gap — landed 2026-07-28 for the Reduce-free population.**
   `pixelflow-search/src/runtime.rs`'s `optimize_runtime_arena(arena, root)`
   inserts an arbitrary runtime `ExprArena` into a fresh `EGraph` (memoized by
   `ExprId` on top of the e-graph's own structural hash-consing, iterative —
   not recursive — matching `choices_to_arena`'s style since arena depth is
   unbounded in principle), saturates with the same size-based
   `SaturationConfig` preset the macro tier uses, extracts via the same
   `ExtractionPolicy` (static latency-prior by default,
   `PIXELFLOW_NNUE_WEIGHTS` opt-in — unchanged, still there, still reachable),
   and converts the winning choices back to an `ExprArena` via the
   already-existing `choices_to_arena`. Returns `None` — compile the original
   arena unchanged — the instant it meets a `Buffer`/`Gather`/`RawGather` (no
   rule reasons about memory) or `Nary`/`Reduce` (rewriting under a binder is
   unsound without binder-aware rules) anywhere reachable from root; `Param`
   too, defensively (a macro-only concept that should never reach a
   runtime-built `Kernel`). `Lattice::bake` calls it before
   `compile_collapse_cached`. Consolidated as one implementation, not two:
   `SaturationConfig`/`config_for_node_count` and the extraction-policy
   selector (`ExtractionPolicy`/`env_extraction_policy`, the
   `PIXELFLOW_NNUE_WEIGHTS` env-var opt-in) moved out of
   `pixelflow-compiler::optimize` into `pixelflow-search`, which already owned
   `EGraph`/`CostModel`/`ExprNnue`; the compiler's own hand-rolled saturation
   loop is deleted in favor of the already-existing (and more correct —
   properly time-slices sub-budgets per round) `saturate_with_full_budget`.
   `pixelflow-ir` still has no `pixelflow-search` dependency (the suckless
   constraint holds — the hook lives in `pixelflow-core`, which calls
   `pixelflow-search` directly now rather than only transitively through
   `pixelflow-compiler`). Tests in `pixelflow-search/src/runtime.rs`: FMA
   fusion and identity-collapse on synthetic runtime arenas, a
   shared-subexpression case, a `Dwrt`-bearing arena (cross-checked against
   `lower_dwrt_owned` — Dwrt is representable, since `derivative::ChainRule`
   already reduces it in the e-graph), and explicit bail-out tests for
   `Buffer`/`Gather` and `Reduce`. Font goldens (`kernel_glyph_golden.rs` et
   al.) are unaffected — optimization preserves semantics by construction —
   and now compile through the fused form.

   **Measured cost, and where it lands.** The optimization result is cached
   by structural arena shape (`Arc`-wrapped, so a hit is an atomic refcount
   bump, not a deep arena clone — matching how `jit_cache` hands back
   `Arc<JitManifold>`), so saturation runs once per distinct kernel shape,
   not once per bake. What remains on *every* call, hit or miss, is
   computing that structural key — an O(reachable-arena-size) walk, same
   shape as `jit_cache`'s own existing canonical-key cache. Direct
   measurement on real glyphs (`pixelflow-graphics/benches/font_rendering.rs`,
   `cargo bench -p pixelflow-graphics`): baking 'A' (1084 total arena nodes,
   277 reachable — most of the gap is construction garbage from splicing,
   not real content) went from ~16µs to ~20µs; the `cache_warmup_alphabet`
   bench (26 distinct glyphs, fresh `GlyphCache` per iteration — explicitly
   the one-time warm-up cost, per its own doc comment) went from ~4.7ms to
   ~5.5ms. The `cached_HELLO` bench — steady-state rendering through an
   already-warm `GlyphCache`, which never calls `bake` — is unchanged across
   every measurement, confirming the added cost lands *only* in the
   one-time per-glyph-bucket bake, never in the per-frame render path.
   `GlyphCache::get` bakes a given `(codepoint, size_bucket, density_bucket)`
   at most once for the cache's lifetime, so this reads as a few-to-tens-of-
   microseconds one-time tax per distinct glyph a real application ever
   renders — negligible against startup/warm-up, and it buys correctness-
   preserving CSE/FMA fusion that the loop-in-JIT and future NNUE-guided
   extraction work compounds with. Left deliberately unoptimized further
   (e.g. a pointer-identity fast path keyed on the `Kernel`'s own `Arc`
   would remove the structural walk on repeat calls to the *same* object,
   but needs the cache to hold a weak/strong reference to avoid an
   address-reuse hazard on an unbounded `'static` cache) — no measured
   caller pays this on a hot path, so per the "subtract before you add"
   rule there's nothing here to subtract yet.

### Making the e-graph actually fire on glyphs (landed 2026-07-28, same day)

The hookup above was necessary but not sufficient: on real glyph kernels the
tier was silently inert, twice over.

1. **`Dwrt` reached saturation with nothing to fold.** The runtime tier fed
   the raw arena to the e-graph and lowered `Dwrt` *afterwards*, so
   `ConstantFold` never saw the constants that derivative resolution creates.
   Fixed by running `lower_dwrt_owned` *first*, then saturating. On the
   analytical curve kernels this is where the wins are: a line's `d = X − f(Y)`
   has `DX(d) = 1`, so the whole gradient-magnitude `sqrt` folds away
   (29 → 22 nodes, sqrt count 1 → 0), and a quad's discriminant/`sqrt` chain
   is shared structurally between the value and gradient paths
   (185 → 79 nodes, −57%).

2. **The winding masks made the whole kernel bail.** The `in_y` band tests
   lower to `BitAnd`/`BitOr`, which `op_from_kind` deliberately excludes, so
   every real glyph arena hit the bail-out and compiled unoptimized. The fix
   is *tier-scoped*, and the scoping is load-bearing: the mask ops are
   registered only in `pixelflow-search/src/runtime.rs`
   (`runtime_op_from_kind` → local `MaskAnd`/`MaskOr` ZSTs), NOT in the global
   `egraph::ops::op_from_kind`. Registering them globally was tried and broke
   the fonts' density-dependent AA ramp: the AOT macro tier runs
   *pre-composition*, where resolving a leaf's `DX` to 1 is wrong the moment
   an enclosing `.at()` warp scales the coordinate. The runtime tier runs
   *post-composition* (bake time), where the same resolution is sound. Masks
   are bit patterns, not numbers (see CLAUDE.md's FP contract), so they ride
   through extraction opaquely — no rule reasons about their value.

Two spillover fixes the optimized arenas forced:

- **Select spill corner in the emitter.** Optimized glyph arenas under
  register pressure hit the previously-unsupported "Select with a spilled
  result and both branches spilled" case. `IsaBackend::select_extra_reload()`
  (v28 on aarch64, xmm13 on x86 — both outside RELOAD_REGS, gather clobbers,
  and `emit_select`'s own scratch) gives `resolve_operands` a third reload
  register instead of an `Err`.
- **Golden tolerance 1e-4 → 1e-3.** The optimized arena contains `MulAdd`
  (FMA fusion); its one-vs-two-roundings divergence is documented
  platform-specific, and winding sums amplify the last-bit difference to
  ~1e-4 between the JIT and the interpreter oracle on a no-FMA build. The
  goldens (`kernel_glyph_golden.rs`) now interpret the *optimized* arena
  (pinning the compiler), and a separate suite
  (`pixelflow-graphics/tests/kernel_glyph_optimize.rs`) pins optimization
  soundness — optimized-vs-raw on real glyphs within reassociation noise —
  plus structural guards: the line-gradient fold (sqrt count 0), the quad
  discriminant share (≤3 sqrts), and lowered-winding representability
  (every op the lowering produces must be `is_egraph_representable`, so a
  new op can't silently reintroduce the bail-out).

**Measured (SSE2 dev box, `cargo bench -p pixelflow-graphics`).** With the
tier actually firing, baking 'A' is ~14.1µs — below both the inert-e-graph
~20µs *and* the pre-hookup ~16µs baseline, so the optimization now more than
pays for its own structural-key walk. `cache_warmup_alphabet` (26 glyphs,
cold) recovered to ~5.2ms from the inert tier's ~5.5ms (pre-hookup: ~4.7ms —
the residual gap is the once-per-shape saturation, amortized away by the
cache on every subsequent bake of the same shape). Steady-state
`cached_HELLO` is unchanged (~74µs), confirming the cost lands only in the
one-time bake. Quad-heavy singles for the FreeType comparison: 'O' ~167µs,
'S' ~264µs.

## Measurement plan (do this first)

- `pixelflow-runtime/examples/bench_psychedelic.rs` is the JIT-vs-LLVM parity
  number (`kernel_raw!` vs `kernel!` vs `kernel_jit!`, ns/pixel).
- **Landed 2026-07-29:** `pixelflow-ir/benches/collapse_overhead.rs` replaces
  the ad-hoc example with a Criterion comparison of a Rust `KernelFn` loop
  nest against one 2D `CollapseKernelFn` call on the same arena. On the SSE2
  development host the deliberately cheap 61,440-pixel kernel is essentially
  tied (49.23µs Rust loop, 49.63µs collapse), an honest baseline showing that
  wins must come from LICM/expression cost rather than assuming the boundary
  alone dominates. `pixelflow-graphics/benches/font_rendering.rs` remains the
  production end-to-end measurement.
- To size finding 2 (missing optimization) separately: bake the same
  expression via `kernel_jit!` (e-graph'd at macro time) and via runtime
  `Kernel` composition (raw), and diff ns/pixel. If the gap is large, an
  optimization pass on the bake path (bounded-budget saturation, or at
  minimum a peephole FMA/CSE pass over the arena) may buy more than the loop.

## Other emit-path gaps (flagged, unfixed)

- AVX2 register budget cut from 6 to 4 workspace-wide to paper over
  `emit_gather_scalar`'s temp usage; the source names the fix (red-zone
  spill of the two temps).
- AVX-512 constant materialization: `mov [rsp-4], imm32` + `vbroadcastss` per
  occurrence, no pool, no dedup.
- `Select` compiles to short-circuit branches with a horizontal mask reduction
  per select; worth measuring against a straight blend inside a tight loop.
- No `vzeroupper` anywhere; SSE2/scanline paths use legacy SSE encodings — an
  AVX↔SSE transition hazard when the Rust caller is compiled with AVX. Flagged,
  unmeasured.
- Per-batch fn-pointer reload from `Arc` in `RealizedKernel::eval` (hoistable
  in the collapse-loop world; irrelevant once the loop is inside the JIT).
