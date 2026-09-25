# A kernel with a lattice is the only evaluation API

**Date:** 2026-09-06
**Status:** plan of record (project direction, stated 2026-09-06; shape
corrected the same day — see "The shape")
**Supersedes, in scope:** the "Surface" lane of
[2026-07-28-jit-performance-parity.md](2026-07-28-jit-performance-parity.md);
completes P6 of [2026-07-20-kernel-unification.md](2026-07-20-kernel-unification.md)
for evaluation and states its end state.

## The shape

Three objects, one verb, and the same vocabulary the language has always
had:

- A **kernel** is the description: an arena with a root, built with the
  combinator vocabulary — `X`, `Y`, arithmetic, `.at`, `.select`, `.sqrt`,
  `over`, `Dwrt`. `pixelflow_ir::Kernel` already carries it.
- A **manifold** is a kernel **compiled at a lattice shape**: the thing you
  can sample over a domain. It is compiled IR (today's `JitManifold`),
  specialized on the extents it was compiled for, held behind the global
  compile cache.
- A **lattice** is the domain: extents and origin.
- The one verb is **collapse**: tabulate a manifold over a lattice. The
  compiler owns everything inside it — the loop nest, invariant hoisting,
  the pack, register allocation — and the runtime calls it **one collapse
  call per stripe**, stripes across threads.

```
kernel  ──compile(shape)──▶  manifold  ──collapse(lattice)──▶  buffer
```

`Lattice::bake(&Kernel)` is `collapse(compile(k))` fused and can stop being
a separate name once `collapse` takes a compiled manifold.

**What this corrects.** The first draft of this plan kept the `Manifold`
*trait* — `eval(point) -> value`, one SIMD batch at a time — as "the
algebra's semantics" and deleted `Lattice::collapse` because it consumed
that trait. That had the word bound to the wrong level. The batch-shaped
trait is exactly what forces a Rust loop around the JIT and the per-batch
ABI cost; a compiled kernel cannot be a good implementor of it, and can be
a very good implementor of a manifold interface defined at the *lattice*
level. So: `collapse` is the right verb aimed at the wrong type, and it is
retargeted, not deleted. The trait, the ZST combinator library that
implements it, the `kernel!` LLVM tier that computes with it, and `Lower`
(which was to bridge them to IR and was implemented for two types) are the
**legacy tier**, on their way out because `Kernel` already has the same
vocabulary and the JIT already beats them.

**Per-batch evaluation is not an API.** The only way a consumer turns a
kernel into numbers is to compile it at a lattice's shape and collapse it
over that lattice.

## Why: the 2× is the boundary, and the compiler already wins without it

Measured 2026-09-06 on this host, the psychedelic shader (four `exp`, five
`sin`, six divides, three channels), 1920×1080, release, single thread,
against the monomorphized `kernel!` expression template:

| path | tier | templates | JIT | JIT vs templates |
|---|---|---:|---:|---|
| per-batch `eval`, one C call per SIMD batch | SSE2 | 22.8 ns/px | 25.0 | 1.10× slower |
| (`kernel_jit!` through the old `bench_psychedelic`) | AVX-512 | 3.39 | 6.71 | **1.98× slower** |
| one collapse call per plane (`Lattice::bake`) | SSE2 | 22.3 | 9.7 | **2.3× faster** |
| | AVX2 | 6.4 | 4.2 | 1.5× faster |
| | AVX-512 | 3.58 | 2.44 | 1.5× faster |

Same shader, same lattice, results agree to 5e-4. The per-batch path pays
the ABI — every vector register caller-saved, every invariant recomputed —
and that is the whole 2×. The collapse path beats the templates because it
hoists: three of the four `exp` and two of the five `sin` are
row-invariant, computed once per row where LLVM recomputes them per batch.
At 2.44 ns/px on AVX-512 the kernel spends ~107 cycles per 16-pixel batch,
of which six `vdivps zmm` are ~96: the compiler is on the division
roofline for this expression. The repo's FTZ/DAZ guard and LLVM's fast-math
flags move neither side.

Production is already on the collapse path: the terminal's cell grid bakes
one packed collapse kernel per stripe; glyphs bake through `Lattice::bake`.

## Inventory: every per-batch surface, and what it becomes

| surface | becomes |
|---|---|
| `kernel_jit!` → `__JitWrapper: Manifold` (calls the JIT per batch) | **landed (S1):** `kernel_jit!` returns a `Kernel`. |
| `JitManifold::eval_row` / `eval_at` / `eval_grid` | **landed (S1):** deleted; `call_collapse` is the collapse driver's one entry. |
| `Lattice::collapse_with` / `collapse_scalar` / `collapse_axis` / `ReduceOp` | **landed (S1):** deleted; a reduction is a binder inside the kernel (`Kernel::sum_over`). |
| `Lattice::collapse(&impl Manifold)` — the batch-shaped trait | **retargeted:** `Lattice::collapse(&Manifold)` takes a *compiled* manifold and is the one tabulate verb, one collapse call per stripe. `Lattice::bake(&Kernel)` = `collapse(compile(k))`. Its three remaining callers (`CachedGlyph` tests) move when a compiled manifold can bind buffers (below). |
| `Scene::Surface(Arc<dyn Manifold<Output = Discrete>>)`, `rasterize`, `render_parallel`, `render_work_stealing`, `execute_stripe`, `materialize_discrete*`, `Discrete::pack` in Rust | **S2:** a scene is a compiled manifold with four channels — four channel kernels compiled at the frame's shape with the pack inside (`×255`, clamp, `trunc_to_int`, `shl`, `bitor`, all IR), rendered one collapse per stripe with work stealing. The cell grid is an instance. The HiDPI contramap becomes `Kernel::at`. |
| `Manifold` trait (`eval(point)`), the ZST combinators, `kernel!`'s LLVM tier, `Lower` | **S4, the legacy tier:** retired once nothing consumes them. `Kernel` is the language; the compiled manifold is the object. |

**No colour in the language.** `pixelflow-core` and `pixelflow-ir` know
fields, lattices and integer/bit ops — not RGBA, byte lanes, or pixel
formats. A colour output is the colour-cube idea in the JIT: four channel
kernels in `[0, 1]`, packed at the frame boundary by integer IR ops the
*graphics* crate composes. The packed program, its shifts and the
`Pixel`-format mapping live in `pixelflow-graphics`; the cell grid keeps its
lattice geometry and channel kernels in core, not the pack.

**The language is a DAG.** No iteration binder. A fixed-count iteration is
unrolled at kernel construction into an ordinary DAG the e-graph can CSE
across (the same move `expand_reduce` makes for `over`); a trip count that
must change is a recompile through the cache that already keys on shape.
Nothing that cannot be written as a finite unrolled DAG is a kernel.

## Stages

**S1 — the JIT has no per-batch entry. Landed 2026-09-06 (#1172).**
`kernel_jit!` is a `Kernel`; the `JitManifold` evaluators and the lattice's
reduction family are gone; every caller bakes; `bench_psychedelic` times one
whole frame on both sides, then again under `FastMathGuard`. Net −630
lines; `pixelflow-codegen/src/emit/` untouched. Three findings:

1. **`Lower` is implemented for two types in the tree, `f32` and
   `Kernel`.** The lower/realize design's per-generator impls were never
   landed; nothing built from combinators lowers compositionally. Under the
   corrected shape this is not a gap to fill: there is nothing to lower
   when kernels are already IR, and `Lower` retires with the tier.
2. **`Lattice::bake` refuses an arena that binds memory** while the packed
   frame binds buffers at call time. A compiled manifold must be able to
   bind buffers for fields as well as packed pixels; that is S2's
   generalized program exposed for one channel, and `Lattice::bake` is its
   buffer-free instance. `CachedGlyph` (a `Gather` over bound coverage)
   moves then, and `collapse` is retargeted then.
3. **Outside core there is no other way to turn a `Manifold` into
   numbers** (`Field::store` is `pub(crate)`). Intended.

Bench (this host, whole frame, single thread, median of 5): SSE2 `kernel!`
31.9–35.0 vs bake 11.8–13.3 ns/px (2.6–2.9×); AVX-512 5.5–5.7 vs 4.0–4.3
(1.3–1.4×). The old ten-scanline baseline had flattered the templates ~20%.

**S2 — a scene is a compiled manifold with four channels.** The packed
program generalized from cell-grid geometry to any four channel kernels plus
shifts, living in graphics; `Scene::CellGrid` an instance; a bound-buffer
bake for fields exposed from the same program; runtime fixtures stop
constructing `Scene::Surface`; the stripe loop writes straight into the
frame (no staging, no per-row copy) and pulls small stripes work-stealing
from a pool sized to the cores; FTZ/DAZ on the workers. Gate: the
psychedelic shader through `Scene::render`, threads 1/4/12, both tiers —
first measurement of the recovered prototype: packed beats the surface
path 2.3–2.5× on SSE2 but scales only 1.8× on 4 cores at AVX-512 and loses
when oversubscribed, which is the staging copy and the fixed stripes, not
the compiler. Acceptance: packed never slower than surface at any thread
count; ≥ 3× on 4 cores on both tiers; no loss at 12 threads on 4 cores.

**S3 — scene3d as kernel constructors. Landed 2026-09-06.**
`pixelflow_graphics::scene3d` is plain functions over small structs —
`Ray::through_screen`, `Sphere::hit`, `Plane::hit`, `Hit::select`,
`checker`, `sky` — each producing `Kernel`s and a scene producing four of
them, compiled through S2's `PackedProgram` into `Scene::Packed`. No
`Manifold` impls, no `kernel!`, no jet domain, no `Discrete`. The jet tier
moves to `scene3d_surface` for S4 to delete, with two consumers left on it:
the S3 gate's "before" side and `subdivision_autodiff`, whose
`SubdivisionGeometry` evaluates a Catmull-Clark limit surface in `Jet3` and
has no kernel form. The six CI-named contracts render through the packed
lane; `mullet_vs_3channel_comparison` is replaced by
`four_channels_share_one_geometry`.

**The paragraph above this one was wrong about the code: scene3d is
analytic and there is no ray march.** Nothing in the module iterates ("No
iteration. Nesting is occlusion."): the sphere's `t` is the quadratic's
near root with an epsilon under the radical for grazing rays, the plane's
is a division, and reflection is a Householder step. So S3 needed no
unrolled march and no language feature; every geometry is a closed-form
kernel, and every derivative comes from `Kernel::dx()/dy()`, which
differentiates screen → direction → hit → reflection → hit symbolically.
A march, if a scene ever wants one, is `n` steps of an ordinary Rust `for`
at construction and still a DAG when it reaches the compiler.

Five findings, most of them about what the jet tier's derivatives were
actually doing:

1. **The hit mask was the discriminant, laundered through a derivative
   test.** `Surface` accepted a hit on `t > 0` plus `|∇t|² < 10⁸`; for a
   miss the sphere's `Field::sqrt` (`sqrt_fast`) selects 0, whose
   derivative rule `1/(2√u)` is then infinite, and that is what rejected
   the miss. Stated directly as `disc > 0` here, which also keeps the
   silhouette off `NaN > 0` — unordered and therefore *true* on x86, false
   on aarch64 (CLAUDE.md's `Gt` row). A mask read off `t` would have lit
   the whole frame on one target and nothing on the other.
2. **The jet tier's antialiasing filter was ~540 pixels wide.** Its jets
   were seeded *after* the pixel-to-screen remap, so `|∂P/∂screen|` was per
   *normalized* screen unit — half the frame height. That is why its
   distant floor showed whole flipped cells (coverage → 0 selects the
   neighbour's colour outright) while its near floor smeared checker edges
   over a hundred pixels. `Hit::footprint` is one pixel, which is what a
   footprint means, and the difference is 52% of the pixels in two goldens.
3. **The jet tier's reflection is off a non-unit normal.** It normalizes
   the tangent frame's cross product with `n_len_sq.sqrt().rsqrt()` — which
   is `|n|^-½`, not `|n|^-1` — so its "unit normal" has length `√|n|` and
   `D − 2(D·N)N` is a reflection only where the screen-to-surface map has
   unit area scale. That is very likely what the hand-tuned `2/|cos θ|`
   curvature factor was compensating for, and it is why the gate's mirror
   row disagrees over the whole sphere rather than at its edges. The packed
   lane reflects off the sphere's analytic normal;
   `a_reflection_off_a_sphere_is_a_unit_ray` pins `|R| = 1` wherever the
   sphere is hit, to 2.5% — loose only because `t = b − √(b² − c)` loses its
   leading digits at the grazing rim, which is worth a stable quadratic form
   some day.
4. **A checker filter must wash out to grey, not to its neighbour.**
   `coverage = clamp(d/f, 0, 1)` sends a pixel *on* an edge to the
   neighbour's colour and a surface whose footprint exceeds a cell to
   whichever cell its centre lands in, so cells swap across every boundary
   and a grazing floor flickers between whole cells. A box filter covers `½
   + d/f` of the cell it is in, and with that the two reflective goldens
   pass at every ISA level; with the old form they failed AVX2 by 1.10% of
   pixels and AVX-512 by 1.61%, over the golden helper's 1% platform-noise
   budget, at whole-cell flips.
5. **Time is a coordinate, not a rebuild.** `animated_sphere` rebuilt its
   scene per frame with `t` baked in as a constant, which under kernels is
   a ~200 ms JIT compile per frame. The sphere's centre is now
   `sin(W)·amplitude` and the frame binds its timestamp with
   `PackedFrame::on_slice` (`PlaneRegion` gained the `(z, w)` plane its
   band lies in, private and set by constructor). This is the
   uniform-shaped need of
   [2026-09-06-uniform-slot-identity.md](2026-09-06-uniform-slot-identity.md)
   met by a coordinate: it costs nothing because the scene already had a
   free axis, and it does not generalize to a second uniform.

**The gate** (`pixelflow-runtime/examples/bench_scene_chrome.rs`), chrome
sphere at 1920×1080, median of 9 frames after 3 warm-ups, on an otherwise
idle 4-core host (load ≈ 1.0 from the session itself):

| tier | threads | surface (ns/px) | packed (ns/px) | packed/surface |
|---|---:|---:|---:|---|
| SSE2 | 1 | 13.89 | 43.30 | 0.32× |
| SSE2 | 4 | 4.27 | 11.86 | 0.36× |
| SSE2 | 12 | 4.27 | 11.43 | 0.37× |
| AVX-512 | 1 | 6.08 | 15.61 | 0.39× |
| AVX-512 | 4 | 2.14 | 4.05 | 0.53× |
| AVX-512 | 12 | 2.40 | 4.08 | 0.59× |

Compile cost: 429,395 arena nodes over the four channels, built in 27 ms
and compiled in 211 ms (SSE2; 229 ms at AVX-512) to 8,881 bytes of code
(6,131 at AVX-512). One compile, not one per frame. No saturation
safety-ceiling panic.

Agreement, before either lane is timed. The checker has to come out of any
row that is meant to check something else, because the filter width is one
of the things that changed; so the geometry row is a **matte** sphere over
a matte floor under the sky, and it agrees within 2/255 on **99.48%** of
pixels (max channel delta 102, at the silhouette) — silhouette, horizon,
sky and pack are the same picture. The reflection gets its own checker-free
row, a sphere mirroring the sky alone, which differs on 2.83% of pixels
against a sphere covering 2.93% of the frame: the whole sphere, by up to
63/255, which is finding 4 below and not an antialiasing difference at all.
As shipped, the two lanes differ on 51.2% of pixels, which is finding 2 and
is the point.

**The acceptance criterion "packed never slower than surface" is not met,
and the reason is structural rather than a compiler defect.** Decomposed on
this host, SSE2, single thread:

- Floor and sky alone: surface 9.7, packed 18.6 ns/px. With a *constant*
  filter width the packed kernel is 10.3 — so everything except the
  derivatives is at parity, and two exact partials of the hit pipeline cost
  ~8 ns/px on a ~10 ns/px value computation. (This block first read that as
  "4× a forward-mode jet". It is not: the jet figure was inferred by
  subtraction, never measured, and the jet tier's derivatives were the
  degenerate ones this stage found — a ~540 px filter and a non-unit normal
  — which LLVM folds away. 0.8× the value cost for two partials is inside
  the forward-mode bound; see the S4b-2 block's open items.)
- The chrome scene doubles that: `select(hit, world(R), world(D))` computes
  both worlds in every lane, while the combinator tier's `Select` evaluates
  a branch **only when some lane in the batch needs it** — the sphere
  covers 2.9% of the frame, so the surface lane skips the reflected world
  in ~97% of batches. The packed kernel *has* the equivalent — the emitter
  guards an `If`'s arm-exclusive schedule range with a branch on the
  mask's uniformity (`pixelflow-codegen/src/emit/guards.rs`) — and it does
  not fire here, for a reason that is in the schedule analysis and not in
  the language: a value is guardable only if **every consumer lies inside
  that one select's arm**, and a colour is four selects sharing one mask.
  The reflected world's geometry (its `t`, hit point, checker cell,
  footprint) feeds all four channels' arms, so from any single select's
  view it is shared, and only the few per-channel scalar ops at the leaves
  are ever skipped. The old tier never had this problem because its colours
  flowed as one packed word through one `If`.

Two things that look like levers and are not, both measured: raising the
runtime saturation class cap (telemetry says the chrome kernel hits
`class_cap` after 2 iterations at 4,913 of 5,000 classes — hash-consing
alone folds 193k nodes into ~4.9k) from 5,000 to 60,000 costs 30× the
compile time and makes the kernel **15% slower** (43 → 51 ns/px); and
rewriting the scene as `world(select(m, R, D))` — identical per lane, since
a world is a function of its ray — buys 10% while taking the arena from
429k to 4.6M nodes and the compile from 0.2 s to 2.0 s, because the
derivative of a select is a select of both branches' derivatives.

Two things that are levers, neither measured yet, and a stage for them:

**S3b — a choice between colours is one select.** Two fixes, general and
local, and both should land:

1. *Language, typed:* `Bits::select(mask, a, b)`. `If` is already a
   bitwise blend on every backend, so a select over packed words is the
   same instruction with an honest type. A colour in graphics becomes a
   tree — four channel kernels at the leaves, a choice between colours at
   the nodes — that `packed_kernel` lowers by packing each leaf and
   selecting on the words. `Hit::select` then produces **one** select whose
   arms are whole packed colours, geometry included, and the existing
   per-select guard skips the reflected world exactly as the old tier did.
   The pack stays in graphics; core still knows no colour.
2. *Codegen, general:* the arm-exclusive ops **clustered contiguously** in
   the schedule, because a branch skips a range and exclusivity alone never
   earned one. (This was written as guard analysis over a *group* of
   selects sharing a mask. The instrument says otherwise, and grouping is
   not this stage's lever: once a colour is one select there is one select
   per mask, and what still refused the guard was the order. Grouping
   remains worth having for a kernel that selects the same condition
   several times without packing — a masked field, say — and is not needed
   here.)

The derivative cost is the other half and is measured separately: two
partials of a hit point cost ~0.8× the value computation on this host —
inside the forward-mode bound, so not a defect — and whether the remaining
constant factor is the `Sqrt` rule's fresh `Rsqrt` estimate chain or the
class cap starving CSE of the expanded derivative is a question for
`collapse_cost` (docs/plans/2026-09-01-schedule-cost-model-denotation.md §9),
not for a guess. Gate for S3b: the chrome scene, packed ≥ surface at every
thread count on both tiers — the acceptance S3 did not meet — and S4 does
not begin until it is.

### S3b landed — 2026-09-06

**The gate is met on both tiers, and the derivative never had to be
touched.** 1920×1080, median of 9 frames after 3 warm-ups, on the same
4-core host (load ≈ 0.7–1.5, shared with other sessions — the surface lane
moved ±7% between runs and is quoted beside the packed lane as the drift
reference):

| tier | threads | surface (ns/px) | packed before | packed after | packed/surface |
|---|---:|---:|---:|---:|---|
| SSE2 | 1 | 16.02 | 43.87 | **12.10** | 1.32× |
| SSE2 | 4 | 4.36 | 11.03 | **3.09** | 1.41× |
| SSE2 | 12 | 4.07 | 11.10 | **3.12** | 1.30× |
| AVX-512 | 1 | 5.99 | 16.03 | **4.91** | 1.22× |
| AVX-512 | 4 | 1.93 | 4.15 | **1.36** | 1.42× |
| AVX-512 | 12 | 2.40 | 4.05 | **1.37** | 1.75× |

Agreement is unchanged to the digit — matte 0.519% of pixels over 2/255,
mirror 2.821%, chrome 51.238% — which is the point: a select on packed
words is a lanewise blend of what the same lanes packed, so **no golden
moved**, and `selecting_packed_words_is_selecting_the_channels` pins that
under both byte orders. Compile: 416,420 arena nodes, 14 ms to build, 181
ms to compile, 9,760 bytes (SSE2).

**The instrument, which is what settled every question here.**
`PIXELFLOW_GUARD_TELEMETRY` makes the guard analysis print, per compiled
scope, the schedule's length and per select: where the mask lands, the
entries each arm owns exclusively, the entries a guard actually skips, and
the entries belonging to someone else that lie between an arm's first and
its last. On the chrome scene's body scope:

| | schedule | selects | arm-exclusive | guarded |
|---|---:|---:|---:|---:|
| S3, colour = four selects | 360 | — | ~0 usable | **0** |
| a colour is one select | 401 | 17 | 694 | 146 (all on the primary world) |
| + clustering, + closed exclusivity | 401 | 17 | 644 | **642** |

The middle row is the finding the language half alone produced, and it is
why this stage has two halves: the reflected world *was* 214 arm-exclusive
entries of 401 — exactly the subtree S3 wanted exclusive — and the guard
still did not fire, because 41 entries belonging to other expressions sat
between its first and its last and a branch skips a *range*. Arm order is
not the lever either: putting the reflected world in the arm the schedule
emits last (a complemented mask) left it with 108 intruders instead of 41.

**Three things in codegen, and the second two were found by the first.**

1. **Clustering.** `cluster_if_arms` stable-partitions the region
   between the mask and the select into shared, then true-exclusive, then
   false-exclusive — always a legal topological order, because a shared
   value can never depend on an arm-exclusive one — and *sinks* whatever
   the select does not read past it rather than hoisting it ahead of both
   arms, which shortens those live ranges instead of stretching them
   (worth 3 points of corpus clock on its own). It runs only on an arm the
   order refused, and its result is kept only if strictly more entries end
   up guarded with no select losing what it had.
2. **A latent miscompile in the exclusivity rule.** "Exclusive" meant every
   consumer *reaches* the arm, which admits a value whose consumer is
   shared with the world outside it; skipping the value leaves that
   consumer reading a register the branch never wrote. It never bit because
   contiguity happened to refuse those ranges — reordering made them legal
   and the cell-grid parity tests failed on the first run. Exclusivity is
   now a closure ("my consumers are skipped with me", to a fixed point),
   and `is_topological` asserts in debug builds that no partition puts a
   value behind its operand. That assert is what caught it.
3. **A guard must be able to pay for its branch.** Whether a mask is
   *coherent* — uniform per batch often enough for the branch to fire and
   to predict — is data, and no static analysis can know it. Its worst case
   is exact, so an arm whose latency-prior cost is under
   `MISPREDICT_PENALTY_CYCLES` (~16, an architectural figure from Agner
   Fog and ARM's optimization guides, not a knob) is never guarded. A bound
   rather than a threshold: the downside of any guard is capped by the
   upside it could deliver. It is the difference between a glyph's coverage
   mask (a handful of ops per arm, varying per lane) and a sphere's
   silhouette (214 entries, uniformly false in 97% of batches), and with it
   the psychedelic gate's kernel is **byte-identical** to before at AVX-512
   (4,352 bytes) — 32 bytes larger at SSE2, where one arm does clear the
   penalty.

**Pressure across the corpus** (`collapse_cost`, 208 kernels — 15
synthetic, 193 production glyph bakes — before = the language half alone,
after = everything):

| tier | bytes | static loads+stores+remats | trip-weighted mem ops | dyn instructions | spill slots | Σ drift-corrected clock |
|---|---:|---:|---:|---:|---:|---:|
| SSE2 | +9.99% | +10.15% | +40.80% | +0.00% | −0.22% | **−7.19%** |
| AVX-512 | +18.63% | +18.73% | +32.09% | +0.00% | −0.35% | **−8.27%** |

**`dyn_memory_ops` cannot see this change, and that is a finding about the
predictor rather than about the change.** It trip-weights every schedule
entry as if it executes, so a transformation whose entire purpose is to
*not execute* a range is invisible to it, while the slot traffic a guard
forces is counted in full. It reports +40% on a change whose clock is
−7%. Noted against its role in
[the schedule cost model](2026-09-01-schedule-cost-model-denotation.md),
where mask coherence is now named as the first concrete profile-dependent
term the learned residual is for.

Two caveats on the corpus clock, stated rather than smoothed: the per-kernel
ratios are not trustworthy below ~10% on this host (byte-identical machine
code measured 4× apart between two runs of the harness, with an A/A floor
inside each run of ~1%), and the glyph bakes that dominate the corpus are
once-per-glyph cached work, where the scene kernel runs every frame.

**The per-kernel tail, which is what the bound was written for.** Before it,
the tail was real and one-directional: at AVX-512 the worst kernels were the
sparsest glyphs — `glyph32 '.'` 3.58×, `'\''` 3.35×, `glyph16 '.'` 2.99×,
`'-'` 2.92×, `'|'` 2.81× — every one of them a shape whose arms are a handful
of ops, guarded because they were contiguous rather than because they were
worth skipping. After it, no tail survives re-measurement. Over the whole
corpus the worst AVX-512 ratio is 1.17× against an A/A spread whose p90 is
1.12×, and those five sit at 0.99–1.11× (0.78–0.91× at SSE2). Re-measured
properly — eight samples per build, the two builds alternated so drift
cancels — they are 0.90–1.01× at AVX-512 and 0.96–1.13× at SSE2. SSE2's
whole-corpus worst, `glyph32 'w'` at 1.91× and `'M'` at 1.65×, do not
reproduce either: 0.90× and 1.04× interleaved, both with *fewer* bytes and
fewer trip-weighted mem ops than before. The control is the number to read
alongside all of them: `invariant04_hot` is byte-identical across the two
builds (652 bytes, same trip-weighted traffic) and measures 1.00× at
AVX-512 and 1.22× at SSE2 — as far from 1.0 as the worst kernel that
actually changed. That is the caveat above restated as a control rather than
as an assertion.

It applies to the aggregate too, and the second run says how far. Repeating
the whole corpus from the rebased branch reproduces every deterministic
column *exactly* — bytes, static traffic, trip-weighted mem ops, dynamic
instructions and spill slots, to the digit, on both tiers — while Σ clock
moves to −20.7% (SSE2) and −2.7% (AVX-512) against the −7.19% and −8.27%
above. So the sign is the claim and the magnitude is not; a Σ over 208
kernels is steadier than any one of them and still not steady, and the
honest reading of the clock column is "it did not get slower".

**The bound refuses arms, not kernels** — worth stating because "the glyph
regression is fixed" invites the wrong picture. Those glyphs still carry
guards: `glyph32 '.'` guards 32 of its 38 arm-exclusive entries across three
selects, `'w'` 102 of 126 across nine. Every refused arm has zero intruders,
so cost alone refused it, and every refused arm is exactly six entries — the
coverage-mask select the bound was written for. What survives is 7, 10, 22
and 74 entries: with `Var` and `Const` free and `Sub`/`Mul`/`MulAdd` at 4–5
cycles, an arm passes 16 cycles by its fifth arithmetic op, so a surviving
guard is one that skips a segment distance and a refused one is a guard on a
sign test.

**What remains for S4 and after.** Guard grouping by mask, for kernels that
select one condition several times without packing. The derivative cost —
two partials at ~0.8× the value cost, with one known constant factor to
recover — which this stage did not need and did not touch. And mask coherence as a cost-model term: the
clustering decision is made statically today by bounding the downside,
which is sound and leaves the upside unclaimed.

**S4 — the legacy tier retires**, in two halves, because the rendering lane
and the language tier have different blast radii and different gates.

**S4a — the rendering lane.** `Scene::Surface`, `rasterize`,
`render_parallel`, `render_work_stealing`, `execute_stripe`,
`materialize_discrete*`, `Discrete::pack` and `Discrete` itself, the
`Manifold<Output = Discrete>` bound on rendering, and the jet-tier scene
modules (`scene3d_surface`, `subdivision`, `spatial_bsp`). Every consumer —
graphics, runtime, core-term, the gate examples, the goldens — moves to the
packed lane in the same change. Gate: **identity.** There is no performance
claim in a deletion, so what is checked is that nothing moved that should not
have: the two scene gates' emitted bytes and the 208-kernel `collapse_cost`
corpus's deterministic columns, on both tiers, before and after.

**S4b — the language tier.** The `Manifold` trait itself, the ZST combinator
library that implements it, `kernel!`'s LLVM backend and `Lower`.
`JitManifold` takes the name `Manifold`. `Lattice::collapse` takes the
compiled type and `bake` folds into it. The inventory is empty.

## Constraints

- **Subtract before you add.** Every row of the inventory is a deletion with
  a migration, not a new path beside an old one.
- **No batch-shaped evaluation entry.** After S4 the only function that
  takes a manifold and produces numbers is `Lattice::collapse`, and its
  argument is compiled IR.
- **The pack is IR.** No Rust-side `from_f32_scaled` on a render path.
- **One measurement per stage**, on the same shader, before and after.

### S4a landed — 2026-09-06

**Net −7,050 lines** (854 insertions, 7,904 deletions across 52 files), and
**nothing that should not have moved, moved.** A deletion has no performance
claim, so the gate is identity, recorded before the first deletion and again
after the last, on this host:

| artifact | SSE2 before → after | AVX-512 before → after |
|---|---|---|
| `bench_scene_psychedelic`'s packed program | 5,584 B `fnv1a=00f3a5ed124990bf` → **identical** | 4,352 B `fnv1a=2d6e37c15a633a65` → **identical** |
| `bench_scene_chrome`'s packed program | 9,760 B `fnv1a=ca0e1d4413e140c7` → **identical** | 7,645 B `fnv1a=5c099aea39b99996` → **identical** |
| its matte row | 1,837 B `fnv1a=c7c3b3c97a76c8ec` → **identical** | 1,507 B `fnv1a=4b3f45d2d7399d35` → **identical** |
| its mirror row | 2,728 B `fnv1a=dfe9e841cb476d7e` → **identical** | 1,765 B `fnv1a=e0413c637fcb5aa6` → **identical** |
| `collapse_cost` corpus fixture (208 kernels) | `md5=379282c4…` → **identical** | (the same fixture) |
| `collapse_cost` deterministic columns — bytes, static loads/stores/remats, trip-weighted mem ops, dynamic instructions, spill slots; every field but `measured` | `md5=55aadbd0…` → **identical** | `md5=bf7892a6…` → **identical** |

The chrome and psychedelic figures match the S3b landing block's 9,760 and
4,352 to the byte, which is the other half of the check: the *before* column
is not merely self-consistent, it is the number that stage recorded.

Also: **no golden was regenerated.** `e2e_render_gradient`,
`e2e_render_radial_gradient` and `e2e_render_circle` were rewritten as packed
scenes and pass against the pictures the per-batch rasterizer drew, at their
existing 2/255 and 1%-of-pixels tolerances, so the `.ppm` files are untouched
in the diff. Workspace `cargo test` wall clock: **227 s** from a cold target
on this 4-core host (133 test binaries, all green).

**Deleted.**

| what | lines | why it had no consumer |
|---|---:|---|
| `graphics/spatial_bsp.rs` | 2,132 | one `Manifold<Field4>` impl; nothing but `lib.rs` referenced it |
| `graphics/scene3d_surface.rs` | 1,160 | the jet tier S3 parked here; both consumers moved or went |
| `graphics/subdivision.rs` + `examples/subdivision_autodiff.rs` | 656 | `SubdivisionGeometry` is a Catmull-Clark limit surface in `Jet3` with no kernel form; the example was its only caller |
| `graphics/render/rasterizer/{mod,parallel}.rs` | 671 | `execute`, `execute_stripe`, `render_parallel`, `render_work_stealing`, `rasterize`, `Rasterize`, `Stripe`, `RenderOptions` |
| `graphics/transform.rs` | 284 | `Scale`/`Translate` via `At`; one e2e test, now precomposition in the language |
| `graphics/baked.rs` | 229 | the `Baked` combinator — a colour manifold cached to memory. Its successor is `PlaneProgram::bind`; **the brief's claim that `fonts/atlas.rs`, `fonts/cache.rs` and `subdiv/` consume it is wrong**, they only use the word "baked" in prose |
| `graphics/animation.rs` | 120 | `TimeShift`/`Oscillate`; superseded by S3's "time is a coordinate", and no consumer |
| `graphics/render/discrete.rs` | 117 | **`render/mod.rs` never declared it** — dead source, not compiled |
| `graphics/image.rs` | 95 | a `Vec<u8>` and a `render_mask(&impl Manifold)` whose body was a comment saying it needed a public evaluation API |
| `graphics/benches/work_stealing.rs`, `tests/raster_parallel.rs`, `tests/rasterizer_parallel_test.rs` | 421 | tested the deleted rasterizer; S2's `how_the_stripes_are_shared_out_does_not_change_pixels` is the surviving statement of "threads must not change pixels" |
| `runtime/tests/surface_scene_hidpi_warp.rs` | 101 | tested `bind`'s point→device contramap, which only ever applied to `Scene::Surface` |
| `runtime/src/render_pool.rs` | 5 | a one-line `pub use … rasterize` with no consumer |
| `core/examples/asm_check.rs` | 62 | called `Discrete::pack`, then wrote `0` into every pixel; printed "Rendered 0 non-zero pixels" and disassembled nothing |
| `Scene::Surface` and `From<Arc<dyn Manifold>>` | — | the variant and its conversion, in `graphics/render/scene.rs` |
| the `Manifold` impls in `graphics/render/color.rs` | 275 | `NamedColor`, `Color`, `ColorCube` and its `Rgba`/`Bgra`/`Platform` aliases, `Grayscale`, `color_manifold`. `Pixel`, `Rgba8`/`Bgra8`/`PlatformPixel`, and `Color`/`NamedColor` as **data** stay |
| `Discrete` in `core/src/lib.rs` | 266 | the alias, the `Field<u32>` inherent impl (`store`/`pack`/`select`), its `BitAnd`/`BitOr`/`Not`/`Computational`/`Selectable`, the `NativeU32Simd` aliases, and `materialize_discrete`/`_fields` |

**Migrated, and why.**

- **core-term** had one `Scene::Surface` left, the "nothing has ever been
  presented" background fill. It is now a **constant packed scene at the
  device-pixel frame size** the app already tracks from `Resized`, compiled
  once per size and cached in a `background` field, and `Skipped` when no
  window has been sized yet. `Skipped` because that is the only honest answer
  — there is no buffer to present into before the first window event, and the
  coordinator already takes `Skipped` on the synchronized-output path, so no
  contract had to change and no `Scene` variant was added. The device-pixel
  conversion the cell-grid path was doing inline became `device_frame()`,
  shared by both callers.
- **`render::rasterizer::{actor, messages}` were never the rasterizer.** The
  actor owns pause/resume, the thread count and the bootstrap handshake, and
  calls `Scene::render` — which since S2 is one collapse call per stripe. It
  is production (the runtime's whole render edge is built on it), so it moved
  to `render::renderer` and `Raster*` became `Render*`/`Renderer*`, down
  through the runtime's own wiring, which called its handle `rasterizer` for
  the same historical reason.
- **`Pixel::packed_shifts` is now the single home of byte order.**
  `RgbaColorCube::PACKED_SHIFTS` stated the same fact a second time, and
  `compile_platform_cell_grid`'s `debug_assert` existed only to hold the two
  statements together.
- **`bind` no longer contramaps for HiDPI.** That wrapper was for point-space
  surfaces; a packed scene is device-pixel space by construction, and
  wrapping one would mean recompiling its program every frame.
- **`psychedelic_shader` was rebuilding its scene every frame** with the
  timestamp baked in as a constant — S3's finding 5 for `animated_sphere`,
  unfixed here. It now puts time on `W` and collapses on a later plane.
- **`runtime/src/traits.rs`'s `Application<P>`** (one `render` method
  returning `Option<Box<dyn Manifold<Output = Discrete>>>`) had **no
  implementor** and was shadowed by `api::public::Application`, which is what
  the engine actually calls.
- **Tests keep their claims, on the packed lane.** `rendering_contract` and
  the render half of `pict_color_tests` still cross-check "what pixel is this
  colour", now the scalar `u32::from(Color)` against the **JIT** pack rather
  than against `materialize_discrete`. `jit_render` compares the two tiers as
  planes over one lattice instead of a rendered frame against a bake.
  `font_rendering`'s cached row times `Lattice::collapse` of the `CachedText`
  it always did, minus the colour pack. `bench_scene_chrome` and
  `bench_scene_psychedelic` lose their surface halves; their agreement rows
  established, once, that the two lanes draw the same picture, which is a
  landing block and not a per-run measurement.

**What the brief did not list, found in the tree.**

1. `Pack` in `core/src/combinators/pack.rs` is **not** `Discrete::pack` — it
   is a vector→scalar fold over a `Vector<Component = Field>`, part of the ZST
   combinator library. It stays for S4b.
2. `render/discrete.rs` was never in the module tree (above).
3. `animation.rs` and `image.rs` have **no consumers at all**; the brief
   expected `animation` to be used by core-term and the runtime API.
4. `shapes.rs` is **not** the terminal cursor's shape library and is not the
   rendering lane: it is `Field`-output ZST combinators with one test caller.
   The cursor is drawn by the cell grid. It stays for S4b.
5. The three `Lattice::collapse` callers in `fonts/cache.rs` **did not move**,
   and neither did a fourth found in `benches/font_rendering.rs`. Retargeting
   `collapse` onto a compiled manifold is S4b by the brief's own split, and
   `CachedGlyph`/`CachedText` are `Manifold` impls until then; moving the
   callers first would have meant writing the retarget here.

**The S4b inventory, as it now stands.**

| crate | `impl … Manifold … for` blocks | what they are |
|---|---:|---|
| `pixelflow-core` | 121 | the ZST combinator library — `ops/{unary,binary,compare,logic,ternary,derivative,reduce}`, `combinators/{at,select,map,fix,binding,computed,context,reduce,shift,spherical,texture,project,pack,block}`, `variables`, the jets, and `manifold.rs`'s constant and smart-pointer impls |
| `pixelflow-graphics` | 3 | `fonts/cache.rs`'s `CachedGlyph` and `CachedText` (a `Gather` over bound coverage — S1's finding, moves when `collapse` takes a compiled manifold), and `patch.rs`'s `BezierPatch` (no consumer) |
| `pixelflow-compiler` | 3 | what `kernel!` **emits** (`codegen/struct_emitter.rs`, `manifold_expr.rs`) — the LLVM tier itself |
| `pixelflow-runtime` | 0 | one mention, in a comment recording what S4a deleted |
| `core-term` | 0 | — |

Return-position `impl Manifold<Output = Field>` (not trait impls) survives in
`graphics/shapes.rs` (4) and `graphics/subdiv/mod.rs` (14); both are
`Field`-tier and go with the combinator library. `Lower` is still implemented
for exactly two types (S1's finding 1). Outside core there is still no way to
turn a `Manifold` into numbers except `Lattice::collapse` (S1's finding 3),
whose four callers are listed above.

**Gates run** on this 4-core host: `cargo fmt --all -- --check`;
`cargo clippy --workspace --all-targets -- -D warnings`;
`cargo test --workspace` (133 binaries, 227 s); every CI-named contract by
exact name, each run individually as CI runs it — "Check kernel tier parity"
(5 tests), "Check JIT/interpreter glyph parity" (4), "Check scene color,
reflection, and antialiasing contracts" (3), "Check ray/surface composition
contracts" (3); core, graphics and runtime tested at `+avx2,+fma` and
`+avx512f,+avx512dq` (47 binaries each); `cargo run -p xtask -- isa-matrix
--clippy --smoke` **PASS at all three levels**; `cargo test -p core-term`;
`cargo build --examples` for graphics and runtime; per-feature checks for
both crates; and `pixelflow-graphics` cross-built for
`aarch64-unknown-linux-gnu` and its whole suite run under
`qemu-aarch64-static` (17 binaries, 193 tests, all green — including the
goldens, whose tolerance budget exists for exactly this divergence).
**No CI job name changed**: every one of them already described what it
checks, and all six named contracts had been on the packed lane since S3.

One note on running that matrix here rather than in CI: a single ISA level's
`cargo test --workspace --no-run` plus `clippy --all-targets` at
`debuginfo = 2` did not fit this container's free disk, and `isa-matrix`
already wipes its target dir per level. `CARGO_PROFILE_DEV_DEBUG=0` was set
for that run — it changes artifact size, not what is compiled or executed.

### S4b-1 landed — 2026-09-06

**The name `Manifold` is the compiled object, and `collapse` takes it.** The
denotation, with the names as built:

```
Kernel ──Manifold::compile(extent)──▶ Manifold ──bind(&[(id, buf)])──▶ BoundManifold
       ──Lattice::collapse──▶ DiscreteManifold
```

- **`Kernel`** (pixelflow-ir) — the description. Unchanged.
- **`pixelflow_core::Manifold`** — a kernel compiled at a lattice's shape.
  Today's `PlaneProgram`, taking the name; `lattice::manifold` is its module.
  It knows its extents, its buffer declarations and its code bytes; it has no
  `eval` and is not batch-shaped.
- **`BoundManifold`** — `PlaneFrame`, renamed for what binding produces. A
  kernel that reads nothing binds the empty slice and is a bound manifold
  too; there is no second, buffer-free form.
- **`Lattice::collapse(&BoundManifold) -> DiscreteManifold`** — the one
  tabulate verb. `DiscreteManifold` keeps its name: it is the buffer that IS
  a manifold by the representable-functor law.
- **`pixelflow_codegen::CompiledKernel`** — `JitManifold` renamed to what it
  is, one kernel's emitted bytes at one shape. It was never re-exported from
  core (the brief expected it to be), and is not now.
- Graphics follows: `PackedManifold` and `CellGridPackedManifold`.
  `PackedFrame` keeps its name — in graphics' vocabulary a *frame* is exactly
  the bound form — and `render_packed` needed no change beyond its argument's
  type.

The per-batch `Manifold` **trait** could not be deleted here (121 impls in
core's combinator library, plus what `kernel!` emits), so it moved off the
crate root to **`pixelflow_core::combinator::Manifold`**, in `combinator.rs`.
No use site changed its spelling — only where the name is imported from. That
is what makes the trait's absence from the root a statement rather than an
accident, and it is what S4b-2 deletes.

**Rank was reconciled upward, and it cost nothing.** `PlaneProgram` compiled
at a 2D `[u32; 2]` while `Lattice::bake` compiled at the 4D `LatticeShape` of
the lattice's own extent. A manifold now compiles at `[u32; 4]` — a
`Lattice`'s whole extent, which is precisely what a `LatticeShape` already is
and what `bake` already passed, so the cache key and the emitted code are
unmoved (a frame is `[w, h, 1, 1]`). The **collapse ABI is untouched and
stays two-dimensional**: one call fills batches across X and rows down Y, a
band therefore lies in one `(z, w)` plane, and `Lattice::collapse` calls it
once per plane — which is the Rust loop `bake` already had. Rank is a
property of the shape a kernel is compiled for; two is a property of the
store. The lattice's **origin** is deliberately *not* part of compilation: it
says where a collapse starts, not what the code is (`LatticeShape` erases it
for the same reason).

What made one collapse loop serve both callers is that `PlaneRegion` now
carries **the coordinate of its first sample** instead of a row index plus a
`(z, w)` slice. That was the only difference between them: a pixel band
samples centers (`x + ½`, `y + ½`), a lattice samples `origin + index`.
`PlaneRegion::rows` builds the first, a crate-private `from_origin` the
second, and the loop below is one body. Its debug bound became
`rows <= extent[1]`, which is exactly what `CompiledKernel::call_collapse`
promises; the old `y0 + rows <= extent[1]` is not expressible once the origin
is general, and was never what the ABI checks.

**`bake` is one line and its refusal moved to where the rule already lived.**
`Lattice::bake(&Kernel)` is `collapse(compile(k, self.extent).bind(&[]))`.
S1's finding 2 — that `bake` refuses an arena binding memory — is now
structural: `bind(&[])` leaves any declared slot empty and
`Manifold::bind` panics **naming the slot** (`nothing bound to slot
BufferDecl { .. }`), which is the better message and one fewer statement of
the same rule. Nothing silently reads a null context. An empty domain still
bakes to an empty buffer without compiling, since a degenerate extent is not
a lattice for the JIT to specialize to.

**The glyph cache is a kernel producer.** `CachedGlyph::kernel()` is the
4-tap blend contramapped into texel space (`p·density − ½`) and masked to the
glyph's point-space extent — the same expression its `eval` built, now in the
language; `CachedText::kernel()` places each glyph with `Kernel::at` and sums
with `Kernel::sum`, which is what its `eval` loop was doing by hand. Both
`Manifold` impls are deleted, `binding()`/`bindings()` hand over the coverage
buffers by identity, and every test goes through
`Manifold::compile(..).bind(..)` + `Lattice::collapse`. **core-term consumes
neither type** — it draws text through `GlyphAtlas` and the cell grid, and
names `CachedGlyph`, `CachedText` and `GlyphCache` nowhere — so the migration
is contained to `pixelflow-graphics`. All four `kernel_glyph_golden`
contracts pass unchanged; no golden moved.

One honest limit this exposes: a `CachedText` run compiles to one kernel over
one slot **per distinct glyph**, and `MAX_BOUND_BUFFERS` is 4. Four distinct
characters is the ceiling for a run collapsed this way, which the tests and
the bench are within ("Hello", "HELLO"). Production text does not go through
this path — the atlas is one buffer for every glyph, which is why the cell
grid scales — and the ceiling is the compiled manifold's, not the cache's.

**The identity gate. Nothing that should not have moved, moved.** The *before*
column is not this stage reading itself — the emitted-byte figures are the ones
S4a's landing block recorded, re-measured here on the same host:

| artifact | SSE2 before → after | AVX-512 before → after |
|---|---|---|
| `bench_scene_psychedelic`'s packed program | 5,584 B `fnv1a=00f3a5ed124990bf` → **identical** | 4,352 B `fnv1a=2d6e37c15a633a65` → **identical** |
| `bench_scene_chrome`'s packed program | 9,760 B `fnv1a=ca0e1d4413e140c7` → **identical** | 7,645 B `fnv1a=5c099aea39b99996` → **identical** |
| `collapse_cost` corpus fixture (208 kernels: 15 synthetic, 193 glyph bakes) | `md5=379282c4…` → **identical** | (the same fixture) |
| `collapse_cost` deterministic columns — every field but `measured` | `md5=3f61ec4d…` — **new baseline** | `md5=470a3c27…` — **new baseline** |

The emitted-bytes instrument is the two examples' own scene constructors plus
an fnv1a over `Manifold::code_bytes()`, run once per ISA level. It is
deliberately **not** in the tree: there it would be a third copy of two scenes
that already exist as examples, and what it prints is a per-host,
per-ISA-level measurement for a landing block, not a contract a test could
assert.

The fixture row carries more than it looks: the corpus is captured by *this*
build, so an identical hash says the glyph bakes and the synthetic families
this stage compiles are the ones S4a compiled. The **columns**, though, could
not be compared — S4a's `md5=55aadbd0…` records a number without the recipe
that produced it, and none of six plausible reconstructions reproduces it. A
hash whose derivation is not written down is not a gate, so these two are a
new baseline **with the recipe beside them**:

```text
collapse_cost capture --out corpus/                        # the fixture
find corpus/ -type f | sort | xargs cat | md5sum
collapse_cost bench --corpus corpus/ --out rows.jsonl \
    --git-ref <ref> --passes 2                             # the columns
jq -c 'del(.measured, .git_ref, .git_sha, .profile)' rows.jsonl | md5sum
```

What stands in for the missing comparison is stronger than the hash would have
been: **every input to those columns is byte-identical to `origin/main`.** The
diff against it is empty over `pixelflow-ir/`, `pixelflow-codegen/src/emit/`,
`pixelflow-pipeline/src/`, and the two files `capture` bakes through
(`fonts/ttf.rs`, `fonts/atlas.rs`); `pixelflow-codegen`'s only change in this
stage is `JitManifold`'s rename, and `collapse_bench` never goes through it —
it calls `emit::compile` directly. Identical inputs through identical code
cannot produce different columns.

**Two things this stage owed and had not paid**, found when its own commits
were re-read against the brief:

- `bake`'s refusal of a buffer-declaring kernel moved into `Manifold::bind`,
  and **nothing pinned it** — the old assert's message (`"bake binds none"`)
  appears in no test at `origin/main` either, so the rule was one edit away
  from becoming a null base-pointer read through an entirely safe API.
  `baking_a_kernel_over_bound_memory_names_the_slot_it_cannot_fill`
  (`pixelflow-core/src/lattice/tests.rs`) is now the check.
- `pixelflow-graphics`'s crate root still re-exported the per-batch **trait**
  as `pixelflow_graphics::Manifold`, so the one name meant two different
  things in two crates — the exact ambiguity this stage exists to remove.
  Nothing imported it from there (`shapes.rs` and `subdiv/mod.rs` name the
  trait through `pixelflow_core::combinator`), so it is gone.

**Where the brief was wrong**, in the tree:

1. **There were five `Lattice::collapse` callers, not four.**
   `pixelflow-runtime/tests/jit_render.rs` tabulates three `kernel!`
   combinator planes through it. With `collapse` retargeted there is no
   library entry that tabulates a combinator at all, so that test now owns
   its loop over `Manifold::eval` — exactly as `kernel_routing_parity`
   already does, and for the reason stated there: a test owns its loop, and
   that is not an API.
2. **`JitManifold` was never re-exported from `pixelflow-core`.** The brief
   asked for that re-export to stop; there was none to stop. Core holds it in
   private fields only, and `__macro` re-exports just `pixelflow_ir`.
3. `PlaneProgram::extent()` and `PackedProgram::extent()` had no callers
   outside their own crates, so widening the first to rank 4 touched nothing.

**The S4b-2 inventory, as it now stands.**

| crate | `impl … Manifold … for` | `kernel!` | `kernel_raw!` | `kernel_value!` | `kernel_jit!` | `ManifoldExpr` |
|---|---:|---:|---:|---:|---:|---:|
| `pixelflow-core` | 113 in `src/`, 4 in `tests/` | 3 + 2 | — | — | — | 5 |
| `pixelflow-compiler` | 4 (emitted by `struct_emitter.rs`) | 10 | 2 | 3 | 5 | 6 |
| `pixelflow-graphics` | 0 | 5 | — | 1 | — | — |
| `pixelflow-runtime` | 0 | 2 | 1 | — | 2 | — |
| `pixelflow-search` | 0 | 1 | 1 | — | — | — |
| `pixelflow-pipeline` | 0 | 1 | 1 | — | — | — |
| `pixelflow-ir` | 0 | — | — | — | — | — |
| `core-term` | 0 | — | — | — | — | — |

**These counts are grep-reproducible, and the previous two were not** — S4a's
and this block's first draft disagreed by a factor of two on the same tree,
because one counted a looser pattern and the other counted prose. Impls are
`^\s*impl(<…>)?\s+.*\bManifold\b.*\bfor\b` less `ManifoldExpr` /
`ManifoldCompat` / `ManifoldExt`. The macro columns are *files containing an
invocation*, `(^|[^_a-zA-Z])<macro>!\s*[({[]` — which is why `pixelflow-ir`'s
row is empty (it cannot depend on the compiler; those were mentions in prose)
and core's reads `3 + 2` (three files in `src/`, two in `tests/`). The
`ManifoldExpr` column is files that name the trait at all.

The core impl count is 113 because it is the whole combinator library at once
— `ops/`, `combinators/`, `variables`, the jets, `combinator.rs`'s constant
and smart-pointer impls, `ext.rs`'s `BoxedManifold`, `lib.rs`'s `Field`
operator impls, and `lattice`'s `DiscreteManifold`/`BilinearSampler` — which
is what S4b-2 deletes in one move; the four in `tests/` are core's own
hand-written implementors of it. **Outside core and the compiler the count is
zero**: `pixelflow-graphics`'s `patch.rs` (`BezierPatch`, S4a's last non-core
impl) had no consumer at all and is deleted here rather than left for S4b-2
to find. Return-position `impl Manifold<Output = Field>` survives in
`graphics/shapes.rs` (4) and `graphics/subdiv/mod.rs` (14 occurrences, 8 of
them after a `->`), both `Field`-tier and both going with the library.
`Lower` is still implemented for exactly two types (`pixelflow-ir`'s `Kernel`
and `f32`).

**Gates run** on this 4-core host, every one green: `cargo fmt --all -- --check`;
`cargo clippy --workspace --all-targets -- -D warnings`; `cargo test --workspace`
(133 binaries, 2,634 tests); each CI-named contract by exact name — "Check
kernel tier parity" (5), "Check JIT/interpreter glyph parity" (4, no golden
moved), "Check scene color, reflection, and antialiasing contracts" (3), "Check
ray/surface composition contracts" (3); `pixelflow-core`, `-graphics` and
`-runtime` tested at `+avx2,+fma` and at `+avx512f,+avx512dq` (47 binaries
each); `cargo run -p xtask -- isa-matrix --clippy --smoke` **PASS at all three
levels** (sse2, avx2+fma, avx512f+dq);
`pixelflow-graphics` cross-built for `aarch64-unknown-linux-gnu` and run under
`qemu-aarch64-static` (17 binaries, 194 tests); `cargo test -p core-term`;
`cargo build --examples` for graphics and runtime; and
`scripts/check-cl-metadata.sh`. `CARGO_PROFILE_DEV_DEBUG=0` was set throughout,
as S4a noted for the same container: it changes artifact size, not what is
compiled or executed.

So S4b-2 is: delete `pixelflow_core::combinator`, the ZST combinator library,
`Lower`, and `kernel!`'s LLVM tier; consolidate `kernel!`/`kernel_raw!`/
`kernel_value!`/`kernel_jit!` into one `kernel!` returning a `Kernel`; and
move the ~24 files that invoke the macros for their combinator value onto it.

### S4b-2 landed — 2026-09-06. The inventory is empty.

**One `Manifold`, and it is the compiled object.** The per-batch trait, the
ZST combinator library that implemented it, `kernel!`'s LLVM backend that
computed with it, and `Lower` — which existed to bridge them to IR — are
deleted. `pixelflow_core::combinator` is gone, and with it the last thing in
the tree that turned an expression into numbers without going through a
lattice.

| surface | landed |
|---|---|
| `kernel_jit!` → `__JitWrapper: Manifold` | S1 (#1172) |
| `JitManifold::eval_row` / `eval_at` / `eval_grid` | S1 (#1172) |
| `Lattice::collapse_with` / `collapse_scalar` / `collapse_axis` / `ReduceOp` | S1 (#1172) |
| `Scene::Surface`, `rasterize`, `render_parallel`, `execute_stripe`, `Discrete::pack` | S2 (#1175), S3 (#1176), S3b (#1177), retired S4a (#1178) |
| `Lattice::collapse(&impl Manifold)` — retargeted onto the compiled type | S4b-1 (#1179) |
| the `Manifold` **trait**, the ZST combinators, `kernel!`'s LLVM tier, `Lower` | **S4b-2, this change** |

**The macros, as they now are.** Two, and the only difference between them is
whether the e-graph runs:

- **`kernel!`** — parse → sema → e-graph saturation and latency-prior
  extraction → arena. Zero params gives a `Kernel`; N params a builder closure
  `move |p0: f32, …| -> Kernel` that folds its arguments in as constants.
- **`kernel_raw!`** — the same, minus the e-graph, so the emitted arena has the
  shape that was written. For benchmarking an exact expression form, for corpus
  generation, and for seeing what the front end built before anything rewrote it.

`kernel_value!` and `kernel_jit!` were the same thing under other names and are
gone. Three pieces of syntax went with the backend that needed them:
**manifold-typed parameters** (`|sdf: kernel|` — a kernel composes `Kernel`
values, which is what `kernel_value!`'s own doc already said), **domain/return
annotations** (`|| Field -> Jet3` — there is one domain), and **named kernel
structs** (`kernel!(struct Circle = …)`). `Var(u8)`'s third magic range — the
manifold-param slot CLAUDE.md's "denote before you build" section names — is
gone with the first of those; two meanings remain.

**What `Field` is now.** One SIMD batch of `f32`, `pub(crate)`, with
`from(f32)`, `sequential(f32)` and a width. It is what the **collapse ABI is
denominated in** — the emitted code takes a batch of X coordinates and a
broadcast of each of Y, Z and W — and nothing else. Its arithmetic, its
comparisons, its selection and its transcendental approximations are the
compiler's now, so `algebra.rs`, `numeric.rs`, `mask.rs`, `storage.rs`,
`dual.rs` and `ops/trig.rs` went with them: those existed to make `Field`
polymorphic over `Jet2`/`Jet3`, and there are no jets. Nothing outside
pixelflow-core can name a lane, a vector or a width; `PARALLELISM` is the one
number that escapes, for a caller sizing a scratch buffer. That is CLAUDE.md's
"SIMD is an implementation detail" finished rather than asserted.

**Core no longer depends on pixelflow-compiler.** The edge existed for
`generate_binary_types!` (Peano indices for nested `Let` bindings, used by one
deleted file) and for the `kernel!` invocations in `ops/derivative.rs`,
`combinators/computed.rs` and `ext.rs`. Both dependency tables lose it, and the
`saturation-telemetry` feature now forwards only to pixelflow-codegen, which is
the crate that actually saturates on core's behalf.

**Two `Manifold` impls outside the combinator library, and both already had a
kernel form beside them.** `DiscreteManifold::eval` (a nearest-neighbour gather
with clamp) and `BilinearSampler::eval` (the 4-tap blend) sat next to
`DiscreteManifold::kernel_for` and `BilinearSampler::kernel_for`, which say the
same thing in the language. The impls go; every caller composes the kernel,
binds the buffer and collapses. **That made `bilinear()` free**: it used to
JIT-compile the blend at a point shape *for that `eval`*, and since S4b-1
nothing called it — a compile per baked glyph, for a call that no longer
existed.

**The identity gate. Nothing that should not have moved, moved.** The *before*
column is what S4a recorded and S4b-1 re-measured; this is the third
measurement of the same numbers on the same host, and the fixture and column
hashes are S4b-1's with its recipe:

| artifact | SSE2 before → after | AVX-512 before → after |
|---|---|---|
| `bench_scene_psychedelic`'s packed program | 5,584 B `fnv1a=00f3a5ed124990bf` → **identical** | 4,352 B `fnv1a=2d6e37c15a633a65` → **identical** |
| `bench_scene_chrome`'s packed program | 9,760 B `fnv1a=ca0e1d4413e140c7` → **identical** | 7,645 B `fnv1a=5c099aea39b99996` → **identical** |
| `collapse_cost` corpus fixture (208 kernels: 15 synthetic, 193 glyph bakes) | `md5=379282c4dc021f56…` → **identical** | (the same fixture) |
| `collapse_cost` deterministic columns — every field but `measured` | `md5=3f61ec4d383f83e4…` → **identical** | `md5=2a3d5c164a49679f…` — see below |

**The AVX-512 column hash S4b-1 recorded does not reproduce from the commit
that recorded it.** This build gives `md5=2a3d5c164a49679f…` where S4b-1's
block says `470a3c27…`, so rather than accept a mismatch or a match on faith,
the *before* side was measured directly: `origin/main` (be020d0c — S4b-1's own
merge) built and benched here, same recipe, back to back, gives
**`2a3d5c164a49679f…`, identical to this branch's**, and the same fixture hash.
So the columns did not move; what moved is confidence in the recorded number.

That is the second time this has happened — S4b-1 could not reproduce S4a's
`55aadbd0…` either — and the pattern now says something: a hash over the whole
JSONL is only a gate if its recipe is exact *and* the tier is one whose
capture is host-independent, and the AVX-512 tier's is evidently not (SSE2's
reproduced to the digit, three stages running). **The before/after that was
actually run here is stronger than the hash would have been**: two builds, one
host, one recipe, minutes apart, byte-identical columns. A cheaper and more
honest gate would compare the *rows* of two builds directly rather than a hash
of one against a number in prose, and that is worth building the next time this
is needed.

The fixture row carries more than it looks: the corpus is captured by *this*
build, through the same `Font`/`GlyphAtlas` production uses, so an identical
hash says the 193 glyph bakes this build compiles are the ones S4a compiled —
after the macro every one of them expands through changed its name and lost a
backend. `ttf_curve_analytical.rs`'s three `kernel_value!` bodies became
`kernel!` bodies and dropped their `-> Field` annotations, and the arenas are
byte-identical, which is the check that the two macros really were one thing.

**No golden moved**, and none was regenerated: all four `kernel_glyph_golden`
contracts pass against the pictures already in the tree.

**Net −28,548 lines** in this stage (129 files, 1,542 insertions, 30,090
deletions), counting `subdiv/coeffs.rs`'s 359,636 lines of generated
eigenstructure data separately — with it the raw figure is −388,184 over 130
files.

**The programme, S1 through S4b-2**, against `e6bc17b3` (the commit before S1):
224 files changed, 10,577 insertions, **41,816 deletions** — net **−31,239
lines** excluding that one generated table, or −390,875 with it. Four stages,
and every one of them a deletion with a migration rather than a new path beside
an old one.

**Where the brief was wrong**, in the tree:

1. **The macro consolidation and the core deletion cannot be separate commits.**
   The brief lists them as separate rows and they are not separable:
   `kernel!`'s emitter emits pixelflow-core's combinator types, and
   pixelflow-core's `combinators/binding.rs` invokes pixelflow-compiler's
   `generate_binary_types!`. The two crates hold each other up; either alone
   does not compile. Everything that *could* be split was, and landed first —
   search, pipeline, graphics, runtime, ml.
2. **`patch.rs` was already gone**, deleted in S4b-1 rather than left for this
   stage, and `mesh.rs` was not on the brief's list at all: an OBJ parser whose
   stated purpose is "the control cage for subdivision surfaces", with no caller
   left since S4a deleted `subdivision.rs`. It goes with `subdiv/`.
3. **`pixelflow-ml`'s SH module was portable, and the port found two empty
   tests.** `elu_feature_positive` bound its result to `_` and asserted nothing;
   `sh_feature_map_projects_direction` asserted only that nine of something came
   back. Both now check values against scalar truth. `ShCoeffs`, `SH_NORM` and
   `Sh2` moved to pixelflow-ml with them — scalar data, and that crate was their
   only consumer.
4. **`pixelflow-pipeline/benches/macro_bench.rs` was a byte-for-byte copy of
   `pixelflow-compiler/benches/macro_bench.rs`.** Both are deleted (what they
   timed was combinator construction and `eval_raw`, not the arena), so the
   duplication is moot — but a copy that had already drifted into two crates is
   worth recording.
5. **`pixelflow-runtime/tests/jit_render.rs` and
   `pixelflow-compiler/tests/kernel_routing_parity.rs` were the same test.**
   Both compared a combinator plane against a baked plane; with one tier both
   reduce to bake-against-truth, so `jit_render` is deleted rather than ported
   and the five renamed contracts are the surviving statement.

**The contracts, renamed for what they now assert.** `kernel_routing_parity`
becomes `bake_vs_scalar_truth`; the `template` column is gone and the five
properties are unchanged:

| was | is |
|---|---|
| `arithmetic_matches_truth_on_both_tiers` | `arithmetic_matches_scalar_truth` |
| `params_match_truth_on_both_tiers` | `params_match_scalar_truth` |
| `piecewise_matches_truth_on_both_tiers` | `piecewise_matches_scalar_truth` |
| `abs_recip_matches_truth_on_both_tiers` | `abs_recip_matches_scalar_truth` |
| `projections_differ_by_tier_and_that_is_the_contract` | `projections_are_the_symbolic_derivative` |

The fifth is the one that changed meaning rather than name: it used to assert
that the two tiers disagreed *on purpose* (over a `Field` domain the combinator
backend's `DX` was 0, which the fonts' "hard step" case relied on). With one
tier there is one answer and it is the calculus, so the test now checks
`DX(X*X) = 2x` and `DY(X*Y*Y) = 2xy` against the closed forms. Two CI job names
changed in the same commit — "Check kernel tier parity" → "Check baked kernels
against scalar truth", and the aarch64 cross-target job's "Type-check both
kernel tiers on aarch64" → "Type-check the kernel macro on aarch64" — because
each named a tier count that is now wrong.

**Gates run** on this 4-core host, every one green: `cargo fmt --all -- --check`;
`cargo clippy --workspace --all-targets -- -D warnings`; `cargo test --workspace`
(119 binaries, 2,362 tests); each CI-named contract by exact name, run
individually as CI runs it — "Check baked kernels against scalar truth" (5),
"Check JIT/interpreter glyph parity" (4, no golden moved), "Check scene color,
reflection, and antialiasing contracts" (3), "Check ray/surface composition
contracts" (3); `pixelflow-core`, `-compiler`, `-graphics` and `-runtime` tested
at `+avx2,+fma` (41 binaries, 541 tests) and at `+avx512f,+avx512dq` (41
binaries, 579 tests); `cargo run -p xtask -- isa-matrix --clippy --smoke`
**PASS at all three levels** (sse2, avx2+fma, avx512f+dq); `cargo test -p
core-term`; `cargo build --examples --benches --workspace`;
`scripts/check-feature-matrix.sh` ("every crate's every feature builds in
isolation"); `scripts/check-provenance-journal-scope.sh`; and
`scripts/check-cl-metadata.sh`. The `Cross-target ABI (aarch64)` job's check —
now `cargo check -p pixelflow-compiler --test bake_vs_scalar_truth --target
aarch64-unknown-linux-gnu` — passes against the renamed target.
`CARGO_PROFILE_DEV_DEBUG=0` was set throughout, as S4a and S4b-1 noted for the
same container: it changes artifact size, not what is compiled or executed.

**Two findings about the aarch64 setup, neither about this change.**
Cross-built for `aarch64-unknown-linux-gnu` and run under
`qemu-aarch64-static`:

1. `pixelflow-core` passes (7 binaries, 48 tests) — which is the ABI signal
   that matters, since `lattice` *is* the collapse ABI. `pixelflow-codegen`
   **SIGSEGVs under the test harness's default thread pool** and passes
   completely, 12 binaries and 190 tests, with `--test-threads=1`. That is
   qemu-user's handling of concurrent `mmap`/`mprotect` on executable pages,
   not a defect in the emitted code: `pixelflow-codegen`'s diff against
   `origin/main` in this change is *empty*. Any aarch64 job that runs it should
   pass the serial flag.
2. **`pixelflow-graphics` did not finish under qemu here, in either mode.** It
   stalls for tens of minutes inside `fonts::cache`'s point-sampling tests,
   and the reason is structural rather than a hang: those helpers sample a
   glyph by compiling a *fresh* kernel per point (a bound-memory arena is
   uncacheable, so `Manifold::compile` really does emit code each time), which
   is ~360 JIT compiles for one assertion. Native that is seconds; emulated it
   is not. The helpers date from S4b-1 and are unchanged here, so this is a
   property of the gate, not of the change — but it is the reason S4b-1's
   "17 binaries, 194 tests" did not reproduce, and a test that wants many
   samples should collapse **one** lattice rather than N points.

## What this plan leaves open

The inventory is empty and the plan is closed. Four things it named and did not
do, each a separate question rather than unfinished work here:

1. **Guard grouping by mask** (S3b). The emitter guards one select's
   arm-exclusive range; a kernel that selects the same condition several times
   *without* packing — a masked field, say — gets one guard per select where it
   could have one for the group. Not needed for a colour, which is one select
   since S3b, and measured to be not this stage's lever.
2. **The derivative cost.** Two exact partials of the chrome hit pipeline cost
   ~0.8× its value computation (S3: 18.6 vs 10.3 ns/px with a constant filter
   width). That is inside the forward-mode bound — a jet carrying two partials
   through the same ops does the same multiplies, and symbolic differentiation
   with hash-consing shares the same intermediates — so it is not a defect,
   and the "4× a forward-mode jet" earlier blocks recorded was a comparison
   against a number that was never measured: the jet tier's ~2 ns was
   inferred by subtraction, and its derivatives were the degenerate ones S3
   found (a ~540 px filter, a non-unit normal), which LLVM folds away. What
   is real is a constant factor: `lower_dwrt`'s `Sqrt` rule pushes a fresh
   `Rsqrt(u)` estimate chain rather than sharing anything with the value's
   `Sqrt(u)`, and the chrome kernel hits the runtime class cap after two
   iterations, so the e-graph barely rewrites the expansion. Both are
   `collapse_cost` measurements before they are changes.
3. **Mask coherence as a cost-model term.** Whether a mask is uniform per batch
   often enough for a branch to fire and to predict is *data*, and the guard
   decision is made statically today by bounding the downside
   (`MISPREDICT_PENALTY_CYCLES`). That is sound and leaves the upside
   unclaimed; it is named in
   [the schedule cost model](2026-09-01-schedule-cost-model-denotation.md) as
   the first concrete profile-dependent term the learned residual is for.
4. **`.claude/DOCS_TOC.md` drift wants a CI check.** Every stage of this plan
   deleted files that prose elsewhere still described, and every stage found
   them by grepping rather than by a job failing. This stage rewrote
   `pixelflow-core/README.md`, the root and graphics READMEs, CLAUDE.md and two
   agent briefs for exactly that reason — `.claude/agents/pixelflow-core.md`
   listed `Jet2H`, `PathJet`, `Discrete`, `chained.rs`'s "impl explosion" and
   `materialize_discrete`, none of which had existed for some time. A gap in CI
   is a check to write: a job that fails when a doc names a path or a type the
   tree does not have would have caught all of it, cheaply, every time.
