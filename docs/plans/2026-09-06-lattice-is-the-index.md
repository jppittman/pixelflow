# A lattice is the index; everything else is a contramap or a uniform

**Date:** 2026-09-06
**Status:** Landed. U0, L1, L2, L3 and L4 are merged on `main`; see §9 for
per-stage PRs, commits, and what the execution found that this plan's draft
did not anticipate. C1 (§5) did not land as written — it was superseded by
[2026-09-07-demand-is-a-dag-property.md](2026-09-07-demand-is-a-dag-property.md)
before merging; see that plan's amendment for what happened to the gate PR
(#1187) along the way. G1 (§4) was executed on 2026-09-08 as its own plan,
[2026-09-08-loop-blinn-glyph.md](2026-09-08-loop-blinn-glyph.md) (PR #1213);
§4's own §9.6 below records what that execution found.
**Author:** JP (direction), Claude (draft)
**Refines:** [2026-09-06-kernel-with-a-lattice.md](2026-09-06-kernel-with-a-lattice.md),
which defines a lattice as "extents and origin". This plan removes the
origin.
**Depends on:** [2026-09-06-uniform-slot-identity.md](2026-09-06-uniform-slot-identity.md)
for the leaf that two of the origin's jobs become.
**Decision it records (JP, 2026-09-06):** *"I don't think we should have
allowed lattices themselves to live in different coordinate spaces and
forced all of the translation to be contramaps plus uniforms."*

---

## The shape

CLAUDE.md states the law that makes a buffer *be* a manifold rather than
merely back one:

```
index(collapse(f)) = f
```

`Lattice` today is `{ extent: [u32; 4], origin: [f32; 4] }`. With an origin
`o`, `collapse(f)(i) = f(o + i)`, so `index ∘ collapse = f ∘ shift`, and the
law holds only if `index` also knows `o`. At that point the lattice is no
longer the representing object. It is an index carrying a coordinate frame,
and the frame is exactly the thing that should have been precomposed onto
`f`.

The correction is one removal:

- A **lattice** is an extent. Its points are the indices `[0, wₓ) × [0, w_y)`
  and nothing else. It is the representable functor's index, and the law
  holds without a side condition.
- A **contramap** is where a coordinate frame lives. Pixel-center sampling
  is `f ∘ (+½)`. A placed glyph is `f ∘ (−pos)`. A DPI scale is `f ∘ (·s)`.
  The kernel already has `Kernel::at`; the lattice stops competing with it.
- A **uniform** is where a per-call scalar lives. Time, cursor phase, light
  direction. The lattice's Z and W "axes" have extent 1 in every production
  call in the tree: an axis that never varies is not an axis.
- A **sub-lattice** is an index range: rows `y₀ .. y₀ + n` of a plane. This
  is the one thing that legitimately lives on the domain side, and
  `PlaneRegion::rows(width, y0: usize, rows)` already spells it as an index
  rather than a coordinate. The plan makes that the only spelling.

```
kernel  ──compile(extent)──▶  manifold  ──collapse(index range, uniforms)──▶  buffer
                                                       │
                                        coordinates enter here only through
                                        the kernel's own contramaps
```

**What the machine code already does.** `Lattice::bake` compiles on
`LatticeShape::new(self.extent)` alone and hands `origin` to the emitted
kernel at runtime as the `Point4` base coordinate. `PlaneRegion::on_slice(z, w)`
does the same for time. So the origin is *operationally* a uniform today.
What is wrong is the type: one `[f32; 4]` field conflates a coordinate frame,
two uniforms, and an index restriction, and breaks the law on paper while
the emitted code quietly respects it. This plan changes types and call
sites; it does not change what the JIT emits for a frame.

## Why now: the measurements that led here

All on this host (AVX-512, `target-cpu=native`), 2026-09-06, on
`0ad8a1c0` unless noted.

**The psychedelic shader, 1920×1080, whole frame.** `Lattice::bake` is 2.86×
faster than `kernel!` (1.84 vs 5.26 ns/px). Two of its three sines are
functions of `W` alone. LLVM, handed the fully inlined loop, does not hoist
them (hand-folding them to constants took the templates tier from 5.52 to
2.57 ns/px). The JIT hoists them because the lattice tells it `W` is
invariant. That is the whole win, and it is a win about *knowing which
values do not vary*, which is what a uniform is. Today that knowledge is
smuggled in as `origin[3]` on an axis of extent 1.

**Text as `Kernel::sum` is O(n) per pixel.** `text()` sums n translated
glyph kernels. A sum has no mask, so nothing in it can be guarded, and every
pixel evaluates every glyph:

| chars | sum, ns/px | balanced select tree, ns/px |
|---|---|---|
| 5 | 87.8 | 62.7 |
| 10 | 171.1 | 59.9 |
| 26 | 297.7 | 104.4 |

The tree is O(log n) and bit-identical. But it is the *range* encoding of an
extent: every pixel still visits the tree. The disjointness of the glyphs,
the one fact that makes text cheap, is nowhere in the program. The sum is
denotationally correct and structurally blind, which is exactly the failure
CLAUDE.md warns about: the invariant lived in the author's head. (The
`text_sizes/50` benchmark case also only renders 26 characters, because
`take(50)` on a 26-char string is 26; its curve is not the flattening it
appears to be.)

**A curved glyph is all row prologue.** `O` at 32 px: 33.1 ns/px on a 40-px
lattice, 2.0 ns/px on a 640-px one, a 16.5× ratio that is the batch-count
ratio. Solving the two widths: per-pixel body cost is ≈0, per-row cost is
≈1.3 µs. Every segment's root-finding is Y-only, so LICM hoists it to the
row prologue, and there it runs unconditionally for all 24 segments on
every row, because the hoist moved it out from under the `in_y` select that
would have skipped it. Guard telemetry on the row prologue reads
`selects=60 guarded=0`; the emitter's comment says "the prologues run once,
so a branch buys nothing there", which is true of the frame prologue and
false of the row prologue. That is a compiler finding, recorded in §5; it
is adjacent to this plan because a domain-side extent is what would make it
moot for a grid and not for a glyph.

**The terminal already does it right.** See §2.

## 1. Where the origin's four jobs go

| job today | example | becomes | mechanism |
|---|---|---|---|
| time / frame parameter | `Lattice::frame(w, h, z)`, `PlaneRegion::on_slice(z, w)`, shaders reading `W` | a **uniform** | uniform-slot-identity plan |
| pixel-center convention | font bench `origin: [0.5, 0.5, …]` | a **contramap** | `kernel.at(X + ½, Y + ½, …)` at the call site that owns the convention |
| point evaluation | `Lattice::point(x, y, z, w)` | a **contramap** over a 1×1 extent | `kernel.at(x, y, z, w)` then `collapse` at `[1, 1]`; or `eval_at`, which already takes a `Point4` |
| row band | `PlaneRegion::rows(width, y0, rows)` | an **index range** | unchanged in meaning; the type stops offering an `f32` alternative |

Z and W leave `Lattice` entirely. A kernel that reads `Z` or `W` today reads
a uniform tomorrow, and the variance analysis that already classifies
"invariant in X and Y" classifies it the same way with nothing to relearn.

## 2. The terminal is the worked example, and the BSP is the counterexample

`CellGrid` (`pixelflow-core/src/lattice/cell_grid.rs`) renders the terminal
as one program over the device-pixel frame at origin zero:

- `col = floor(x / cell_w)`, `lx = x − col·cell_w`: a contramap from pixel
  to (cell, offset), written as kernel arithmetic.
- `cells.at(col·10 + field, row)`: a contramap into the per-cell buffer.
  Which glyph is in which cell is **data**. A keystroke writes ten floats.
- `atlas.at(u0 + lx·density, v0 + ly·density)`: a contramap into atlas
  texel space through the bilinear sampler.

The retired `spatial_bsp` (deleted in `0ad8a1c0`) put which-glyph-where in
tree *structure*, with boxed leaves and per-batch descent:

| | `spatial_bsp` | `CellGrid` |
|---|---|---|
| programs | one per leaf, boxed | one, fused |
| which-glyph-where | tree structure | buffer data |
| routing | tree descent per batch, early-exit when lanes agree | `floor(x / cell_w)` |
| change one char | rebuild the tree | write ten floats |
| cost per pixel | O(log n) descent + leaf | O(1): two floors, four gathers, four taps |
| requires a grid | no | yes |

The BSP's generality is what killed fusion: arbitrary bounds with arbitrary
leaves means `dyn` leaves, and nothing optimizes across them. The grid
exploits the one fact the BSP refused to assume, that routing is arithmetic
rather than search. For a terminal, data beats structure. `CellGrid` is this
plan, written before the plan, by someone who understood the law.

Two places it still carries the old mistake at a smaller scale:

- **Geometry is baked as constants.** `cell_w`, `cols`, `density`, `frame_w`
  are `Kernel::constant`, so a resize recompiles. The module's own doc
  apologizes: the runtime renders "without any DPI contramap", and the scene
  "bakes its display scale in here". DPI is a contramap; cell geometry is a
  block of uniforms. With both, a resize is a parameter write. (Loop bounds
  stay static, per the uniforms plan; `frame_w × frame_h` is the extent,
  not a uniform.)
- **`in_grid.select(blended, default_bg)`.** The frame's extent, encoded in
  the range as a mask. The frame *is* the lattice; the border is the
  region outside the grid's sub-lattice, and it should be its own
  collapse over its own index range, not a per-pixel blend.

## 3. Extents on the domain

An extent is a contramap along an *inclusion* `ι : L' ↪ L`. That map is
partial on the frame, and a partial function in a total language has two
encodings:

- **Totalize it into the range.** `select(mask, value, 0)`. Every pixel
  evaluates; the mask says whether the answer counts. The guard
  (`cluster_select_arms`) recovers the domain encoding after the fact when
  the mask is coherent and the arm is expensive enough. This is what the
  font code does, and what the select-tree `text()` would do.
- **Restrict the index.** Don't ask outside `L'`. In a pull renderer the
  lattice does the asking, so this is a contramap on the *index* side: an
  index range. No mask, no guard, no profitability bound, no coherence
  question. The loop does not go there.

These agree on values where both are defined and differ on who visits what.
The index range already exists (`PlaneRegion`). What does not exist is a
lattice that is a **union of ranges**: `bake(⊔ Lᵢ, K)` distributing to one
loop nest per summand, and `bake(⊔ Lᵢ, Kᵢ)` with a kernel per summand, which
is a draw list. `CellGrid` is the uniform special case, where the union is a
grid and the kernel is shared. Text with a proportional font, sprites, and
any scene of placed things are the general case, and today they have no
lattice to pull from except the whole frame, which is why the only place
left to put the extent is the range.

**Overlap is what the select is for.** A union of *disjoint* ranges needs no
select. Pieces that genuinely overlap, where a pixel really must choose,
are where a select tree is the right encoding, and where the guard
machinery earns its keep. The two are not competing designs; they are the
disjoint and overlapping cases of one denotation, and a scene should be
able to say which it is.

**"One program" is preserved.** A sum of ranges is one program whose loop
nest is shaped by the summands: an outer loop over placed regions, an inner
loop over each region's pixels. That is a draw-call list inlined into a
single kernel, and it is the loop shape `CellGrid` already emits for the
uniform case.

## 4. Loop–Blinn is the shape of a glyph kernel over a domain extent

`ttf_curve_analytical.rs` is a scanline rasterizer written as a kernel:
per segment, solve for where the curve crosses `y = Y`, then compare `X` to
the crossing. It is inherently row-shaped, which is why all of its cost
hoists into the row prologue (§ "Why now"). It is correct without an extent,
because the winding number is global, and slow without one.

Loop–Blinn asks a per-pixel question instead. Assign `(u, v) = (0,0), (½,0),
(1,1)` to a quadratic's control points, interpolate across the control
triangle, and the curve is the zero set of `u² − v`. One affine map and one
`MulAdd` per pixel; nothing Y-only to hoist; the glyph interior is an
ordinary triangulation of the control polygon. Its antialiasing is
`f / ‖∇f‖`, which is the gradient-normalized ramp the font code already
computes, and `Dwrt` would derive `∇(u² − v)` through the affine map with
no new code.

The property that matters for this plan: **outside its control triangle,
`u² − v` is not slow, it is wrong.** It is the implicit of the infinite
parabola. Loop–Blinn is only meaningful *bounded*, so for it the extent is
not an optimization but the denotation. On the GPU that bound is the
rasterizer: the fragment shader never runs outside the triangle. In
pixelflow that bound is a domain-side extent, and the triangulation is the
BSP over which each pixel finds its one triangle. It is the first
formulation under which "a compiled glyph keeps up with an atlas read" is
plausible: a few guarded selects plus eight ops against a gather's latency.
It was not plausible at forty-eight Y-only ops per segment per row.

This section records the direction, not a commitment. TrueType is
quadratics only, which is the easy case; the triangulation and the
overlapping-control-triangle subdivision are real host-side geometry.

## 5. Compiler finding: exclusivity does not survive the partition

Recorded here because §3's domain-side extent is what makes it moot for
grids, and §4 is what makes it moot for glyphs; until both land, this is
where `O`'s 33 ns/px goes.

The emitter computes two facts about a kernel and never joins them:

- `partition_by_scope` takes `variance` and the binder mask, nothing else.
  It moves Y-only values into the row prologue with no memory of which
  select arm they were exclusive to.
- `analyze_select_guards` runs at emission on `allocation.body().schedule()`,
  both call sites. By then the Y-only values are gone from the body, so the
  arm it sees is the X-dependent remainder (the 9- and 23-op guards in the
  telemetry). The row prologue never gets `cluster_select_arms` at all, per
  a comment that reasons about the frame prologue and applies it to both.

The set the row prologue should compute *under a guard* is
**arm-exclusive ∧ Y-only**, guarded by a mask that is itself Y-only and
hoists alongside. Every input to that derivation is already computed. A
synthetic Y-only select with an expensive exclusive arm *does* get guarded in
the row prologue today (telemetry `guarded=37` of 45; timing ratio 3.2× vs
1.9× control), so the machinery works; it is the font kernel's arms that lose
exclusivity because the hoist strips it. `guards.rs` already says exclusivity
is "a property of the schedule, not of allocation"; the refinement is that it
is a property of the DAG, computed once before partitioning, and the scopes
inherit it.

Two honest limits: it is as tight as the mask the font wrote (for quads,
`disc ≥ 0` is a half-plane; a host-computed `y_min ≤ Y < y_max` is the tight
mask, and interval-bounding a quadratic is the font's job, not the
emitter's); and coherence remains a data property, ideal here only because a
row-uniform mask cannot diverge within a row.

## 6. Stages

Each stage is independently landable and leaves every benchmark's emitted
bytes unchanged unless it says otherwise.

**L1 — Z and W become uniforms.** Depends on the uniforms plan's leaf.
`Lattice::extent` becomes `[u32; 2]`; `Lattice::frame(w, h)` loses its `z`;
`PlaneRegion::on_slice` takes uniform values instead of coordinates. Kernels
reading `Kernel::z()` / `Kernel::w()` read a uniform slot. Identity gate: the
psychedelic and chrome packed programs' emitted bytes are unchanged (the
value already arrives at runtime; only its spelling moves).
**Landed as PR #1194, commit `919596e5`.** See §9.1.

**L2 — `origin` is removed.** `Lattice` is `{ extent }`. The pixel-center
`0.5` moves to a contramap at the two call sites that use it (`font_rendering`
bench, glyph cache). `Lattice::point` becomes `eval_at`. `Lattice::scanline`
becomes a row-range collapse. Identity gate: every golden unchanged.
**Landed as PR #1196, commit `bdc72e30`.** The identity gate as stated did
not hold — see §9.2 for the measured, bounded deviation and why the golden
test itself had to change.

**L3 — index ranges get a union.** `Lattice` (or a new `Domain`) is a sum of
index ranges; `collapse` distributes. Disjoint by construction, checked at
build. `text()` for a monospace font becomes a union of per-glyph ranges
with a kernel per range, and its per-pixel cost stops depending on string
length. This is the stage with a measurable target: `pixelflow_text_sizes/26`
from 4.12 ms toward the FreeType-shaped 55 µs, and the `text_sizes` bench
fixed to render the number of characters its label claims.
**Landed as PR #1184, commit `39f06552`** — before L1, off U0 directly.
`pixelflow_text_sizes/26` reached 676.87 µs (13.8×), short of the FreeType
target but flat in string length; see §9.3 for the `Union`'s documented
limit that L4 later ran into.

**L4 — `CellGrid` geometry becomes uniforms; DPI becomes a contramap.**
Resize is a parameter write. The `in_grid` select becomes a border range.
Depends on L1 and L3.
**Landed as PR #1197, commit `d0b504c2`.** "A resize is a parameter write"
holds only for the metric fields; see §9.4 for which resizes still
recompile and why.

**C1 — exclusivity survives the partition** (§5). Independent of L1–L4;
lands whenever. Gate: `O` at 40 px moves from 33 ns/px toward the 2 ns/px
its 640-px number already shows, with the font's tight `y_min/y_max` mask
supplied alongside.

**G1 — Loop–Blinn glyph kernel** (§4). Depends on L3 for the domain-side
triangle extent.
**Executed as its own plan on 2026-09-08**,
[2026-09-08-loop-blinn-glyph.md](2026-09-08-loop-blinn-glyph.md) (PR #1213).
Two of §4's proposals did not survive contact — see §9.6.

## 7. Constraints

- **Loop bounds stay static.** An extent is compiled against; a uniform can
  be a gather index but never an extent. Unchanged from the uniforms plan.
- **The law is the test.** `index(collapse(f)) = f` with no shift and no side
  condition, pinned by a test that fails if anyone reintroduces a coordinate
  onto the index.
- **No coordinate on the domain side.** A sub-lattice is a `usize` range. If
  a call site wants a coordinate, it wants a contramap, and the type says so
  by having nowhere else to put it.
- **The select stays.** For overlapping pieces it is the right encoding and
  the guard machinery is right to exist. This plan removes the cases where a
  select was standing in for a domain extent, not the select.

## 8. Non-goals

- Changing the emitted code for any frame-shaped bake. L1 and L2 are type
  changes over a runtime that already behaves this way.
- Proportional-font layout, kerning, shaping. L3's union is the primitive
  those would compose over; they are not this plan.
- Cubic Béziers in G1. TrueType is quadratic; CFF is later.
- Retiring the range-encoded extent. Overlap is real and the select is its
  encoding.

## 9. What happened

All five stages landed as squashed PRs directly on `main`, in this order
(dates 2026-09-06/07, one shared four-core host, AVX-512 with
`target-cpu=native` unless a PR states otherwise):

| stage | PR | commit | note |
|---|---|---|---|
| U0 (uniform leaf, dependency) | #1183 | `e0220d53` | not part of this plan's stage list; L1's prerequisite |
| L3 | #1184 | `39f06552` | landed *before* L1, directly off U0 |
| L1 | #1194 | `919596e5` | |
| L2 | #1196 | `bdc72e30` | |
| C1a (§5's gate, factored into its own plan) | #1187 | `4fe39607` | test-only; see the demand-is-a-dag-property amendment |
| L4 | #1197 | `d0b504c2` | current `main` HEAD |

C1 (§5, "exclusivity does not survive the partition") did not land in this
plan's form. JP's decision on 2026-09-07 was to build the general demand
predicate on the DAG rather than merge the per-select special case first;
that became its own plan,
[2026-09-07-demand-is-a-dag-property.md](2026-09-07-demand-is-a-dag-property.md),
whose amendment records what happened to the gate PR along the way. That
plan is in turn superseded by
[2026-09-08-one-conditional-three-lowerings.md](2026-09-08-one-conditional-three-lowerings.md),
which also refines §5 above: guards and index ranges are one thing because
they are two lowerings of one `Select`, and `Union`/`IndexRange` are what
the compiler should derive rather than what a caller writes.

G1 (§4) was not scheduled when this section was written; it was executed on
2026-09-08 (§9.6). On 2026-09-09 `Union` was **deleted** — which is that
refinement's conclusion arrived at from the other end. G1 was said to depend
on L3 because `u² − v` outside its control triangle is wrong rather than
slow, so the crescent needed a domain-side fence. It does not:
`{v ≥ u²} ∩ {v ≤ u}` is empty outside `u ∈ [0, 1]`, so two comparisons fence
it with no domain at all. With correctness no longer resting on it, `Union`
was a sum whose terms a caller declared disjoint — `place` panics on overlap,
so it could never say anything `+` could not — and it reached no screen.

The stage-by-stage findings below are corrections and additions to what the
draft above predicted — read them alongside the sections they amend, not as
a replacement for them.

### 9.1 L1: the axis guard is a runtime refusal, not a type, and it had to move

`Var(2)`/`Var(3)` (Z and W's old slots) are **refused, not unrepresentable**.
`ExprArena::push_var(2)` still compiles; nothing in the type system stops it.
What refuses it is a runtime assertion (`retired_axis`, `pixelflow-ir/src/arena.rs`)
checked at every boundary an arena crosses to become values —
`Kernel::from_parts`, `eval_scalar`, and `emit::compile` (`pixelflow-codegen/src/emit/mod.rs`).
`Var` could not become its own type here because it is *also* a reduction
binder's index and a rewrite metavariable, both dense from zero, and 25 rule
templates name `var 2`; giving the coordinate meaning its own type would be
a larger change than this stage.

The guard started at `jit_cache` and had to move: that call site is not the
only route to machine code (benchmark harnesses, corpus tools, and several
tests bypass it), and a live divergence — the scalar oracle for
`training::factored`'s fixtures substituting the retired axes' old values
while the JIT harness fed the emitter zeros in those lanes, three orders of
magnitude apart — was invisible to it. `emit::compile` is the one boundary
every route shares, so that is where the guard now sits.

Also worth recording because it is the opposite of what a reader would
guess: **retiring the axes made a stray reference to one worse, not
inert.** `Variance::from_var(2)` falls outside both `COORDS` (`0b11`) and
`BINDERS` (`0b1111_0000`), so `is_frame_uniform()` reads `true` for it and
LICM would hoist a value computed from a retired axis into the per-call
frame prologue — computed once, silently, from a lane that means nothing.
When Z was still `Variance::Z` inside `COORDS` this could not happen. That
is why the guard in `emit::compile` is unconditional rather than gated on
whether the retired axis "looks reachable": `substitute_vars_with` rewrites
the reachable graph but the arena is a bump list nothing prunes, so an arena
whose Z/W references were correctly substituted still holds the original
`Var(2)`/`Var(3)` nodes, unreachable and unemitted — a naive "scan every
node" guard flags that arena wrongly, which is why the check asks about
nodes reachable from the root, and only the root.

### 9.2 L2: the identity gate was not met, and the corrected claim is a bound

The plan's identity gate for L2 was "every golden unchanged." That is false
as measured. Building `origin/main` at `919596e5` (pre-L2) and the L2
branch in separate worktrees and diffing the production `GlyphAtlas`
coverage buffer byte-for-byte: **849 of 42,768 texels differ, max |delta|
3.612e-5** — deterministic (a same-branch rerun differs in 0 texels), and
below one 8-bit quantization step (1/255 ≈ 3.9e-3), so nothing visible
changes.

Cause: moving the pixel-center `½` from `PlaneRegion`'s runtime `origin`
(invisible to the optimizer) into the kernel arena, as `kernel.at(&(X +
0.5), ...)`, puts it where `optimize_runtime_arena`'s e-graph can
reassociate it. `(X + 0.5) − pen` at `X = i` is not the same float
expression as `X − pen` at `X = i + 0.5`. The existing same-form checks
(JIT vs. interpreter on one arena; `text()` vs. `text_union()` sharing one
`layout()`) each moved both sides together and so could not see it —
CLAUDE.md's "a same-form check cannot see a shared-definition bug; only an
external bound can," reproduced exactly.

The first fix attempted (an FNV-1a-64 hash of the raw `f32` coverage
buffer) was itself wrong and failed on both CI runners at a mismatched
constant, because it asserted bit-identity across ISA levels — something
CLAUDE.md already documents `Min`/`Max`, `Round` and friends do not give.
Measured on one host: up to 2,630 of 42,768 texels differ in raw bytes
between SSE2, AVX2+FMA and AVX-512F+DQ, worst delta ~2.1e-4. The gate that
shipped instead quantizes to the same 8-bit form the renderer actually
writes and compares through the crate's existing tolerance-plus-mismatch-
budget mechanism (`tests/common::assert_golden`), the same one every other
rendering golden in the crate uses. Record the measured bound here, not the
identity the draft assumed.

### 9.3 L3: the `Union`'s documented limit, found by L4

`Union`/`IndexRange` (`pixelflow-core`) binds no buffers, owns the single
output buffer it fills, and is `f32` only. L4 needed a border fill that
requires bound buffers, `u32` packing, and a collapse into memory the
*caller* owns (the packed frame) — none of which `Union` can express. L4's
PR states this as a finding rather than routing around it silently: "the
'range' here is `PlaneRegion` + `IndexRange::intersect` +
`IndexRange::paint_complement`, not a `Union` summand... fusing them is the
general case the plan leaves open." Nothing in this plan or #1197 closes
that gap; it is left open for whoever fuses the cell grid's border and its
metric-driven interior into one `Union`-shaped object.

Also worth recording: the identity gate for L3 (`the_union_agrees_with_the_
sum_to_within_the_compiler`) could not be made bit-exact either, for the
same schedule-is-a-function-of-the-arena-and-shape reason as L2 — see
#1184's own "the bit-identical gate: it cannot be met" section for the
bisection that isolated it to the compiler and not the union.

### 9.4 L4: "a resize is a parameter write" is conditional, not general

True only for the **metric** — `cell_w`, `cell_h`, `density`, `tile_w`,
`tile_h`, `scale` — split into `CellGridMetrics`/`CellGridSlots` as eight
`Uniform` slots. `CellGridShape` (`cols`, `rows`, `atlas_width`,
`atlas_height`, `frame_w`, `frame_h`) is still every field that is an
extent, and §7 of this plan already required extents to stay static:

| change | before L4 | after L4 |
|---|---|---|
| cell size / font size at fixed grid and frame (tilemap zoom) | recompile | parameter write |
| display scale (DPI) at a fixed device frame | recompile | parameter write |
| atlas sample density, tile content size | recompile | parameter write |
| default background / colour scheme | recompile | JIT-cache hit (no longer in the kernel at all — two backgrounds share one compiled program) |
| window resize (frame lattice extent moves) | recompile | **recompile** |
| grid gains/loses a column or row (cell buffer extent moves) | recompile | **recompile** |
| atlas grows (atlas buffer extent moves) | recompile | **recompile** |

So in `core-term` specifically, a window resize still compiles today,
because it moves both the frame lattice's extent and the cell buffer's.
What L4 buys is font-size change, a DPI transition at a fixed device
frame, and an atlas re-bake, none of which move an extent.

L4 also measured that the metric becoming a block of uniforms costs the
e-graph its ability to fold it: a uniform leaf is opaque to the folder, so
the packed kernel and a standalone channel kernel no longer necessarily
extract the same schedule for a shared subterm. Measured at 401×197 with a
fractional metric: 1 pixel of 78,997 moves by one step in one channel byte.
The four per-channel `f32` planes are bit-identical to the pre-uniform
tree — this is a schedule difference, not a value one — and the packed
parity oracle's tolerance was corrected from an unconditional "exact,
no epsilon" claim to one that states this bound and is exercised at both
the exact and the ±1 cases.

### 9.5 New checks this plan's execution added

- `pixelflow-graphics/tests/glyph_atlas_golden.rs` — the quantized,
  cross-ISA-tolerant atlas golden that replaced the bit-exact hash L2's
  first attempt used (§9.2). Directly run under `RUSTFLAGS` at SSE2,
  AVX2+FMA and AVX-512F+DQ as part of both L2's and L4's gates, because no
  smoke job covers `pixelflow-graphics` (§10.2).
- `cargo check --target aarch64-unknown-linux-gnu --all-targets` over the
  seven architecture-specific crates, added as a CI step during L1 after an
  aarch64-only file broke that no x86 job or the pre-existing (single-crate)
  aarch64 job could see (§10.4). Type-checks only — it does not execute — and
  every later stage (L2, L4) re-ran it and reported it as such.

### 9.6 G1: the extent was the denotation, but the triangulation was not

§4 got its central claim right and two of its proposals wrong, which is
worth separating because the claim is what made the stage worth doing.

**Right, and load-bearing.** "Outside its control triangle, `u² − v` is not
slow, it is wrong" — so the bound is the denotation, and L3's union is what
supplies it. Also right: nothing in the formulation is a function of `Y`
alone, so §5's row-prologue finding is moot for glyphs, and there is no
discriminant, so the `'8'` waist defect (§9 of the demand plan, open on
`main` through five refuted fixes) is not expressible. Both its pins are now
asserted empty rather than pinned.

**Wrong.** §4 says "the triangulation is the BSP over which each pixel finds
its one triangle" and "the glyph interior is an ordinary triangulation of
the control polygon". Neither was built, and §2 of this same plan is the
reason: the retired `spatial_bsp` put which-glyph-where in tree structure,
and that is what killed fusion. The interior is a winding constant per cell
and routing is `index / cell_size` — `CellGrid`'s own shape. A plan that
argues against structure in §2 should not have reached for a BSP in §4.

**Also wrong, and only found by measuring.** §4 says the antialiasing "is
`f / ‖∇f‖`, which is the gradient-normalized ramp the font code already
computes". It is what a GPU shader uses and it is **not sound as a distance
on the CPU**: it underestimates at high curvature, and an underestimate past
the ramp is a saturation failure rather than a rounding difference — which
is also what the cell pruning relies on. It survives only as one half of a
maximum against a chord-derived lower bound. See §7.2 of the G1 plan.

§4's cost estimate — "a few guarded selects plus eight ops against a
gather's latency" — is not yet tested: the execution recorded arena node
counts (33 per straight segment, 86 per curved one) and did not run the
benchmark comparison. That is open.

## 10. CI findings from executing this plan

The plan's stages kept discovering gates that read wider than they actually
checked. Recorded here as findings about the gate layout, not about any one
stage, per CLAUDE.md's "a gap in CI is a check to write, not a caveat to
attach" — collected in one place so a later stage does not have to
rediscover them from four different PR descriptions.

1. **`cargo nextest` does not execute benchmark targets.** A bench reading
   the (then still live) Z axis shipped green during L1, and so did the
   first attempted fix for it, which faulted on the null context the bench
   passes. Only found by running `cargo bench` by hand.
2. **`xtask isa-matrix --smoke` does not run `pixelflow-graphics`.** Its
   smoke set is codegen + ir + core + pipeline. Every pixel L2 and L4 moved
   lives in `pixelflow-graphics`, so the three green `PASS` lines this
   command prints carry no signal about either stage's own identity gate;
   both PRs say so explicitly and substitute a direct per-ISA-level run of
   the affected crate's tests instead.
3. **`isa-matrix --smoke` tolerates warnings; CI runs `isa-matrix --clippy
   --smoke`, which denies them.** The command a developer runs locally by
   habit is not the command CI runs. This bit L1 once, live: two
   `dead_code` warnings visible only at AVX2/AVX-512 passed the plain
   smoke command and would have failed CI's `--clippy` form.
4. **aarch64-gated tests are compiled by CI but executed only on the macOS
   runner.** L1's own `cargo check --target aarch64-unknown-linux-gnu` step
   (§9.5) type-checks that code on every runner but cannot execute it, so
   any aarch64-only test is effectively postsubmit for a developer on x86.
   This is what let a NEON-ABI test (`naked_scale.rs`) ship broken until a
   macOS runner caught it — the check exists to close the *type-checking*
   gap that bug came through, and it does not claim to close the execution
   gap.
5. **A feature-gated test file is invisible to a plain `cargo clippy
   --all-targets`.** `#1187`'s `freetype_oracle.rs` is gated on `--features
   freetype`; the plain local command a developer runs never compiles it.
   `--all-features` is required to see it at all, and CI already runs that
   form.
6. **Twice, a test asserted nothing and every job stayed green** — a
   `println!` where an `assert!` belonged (found during #1187's review of
   an earlier version of `optimized_glyph_matches_raw_within_reassociation_noise`),
   and `NaN >= x` reading `false` so a `>=`-style tolerance check silently
   accepted non-finite coverage (found in the same review, in both the
   golden comparison and the oracle). A mutation test during L4's review
   found a third instance of the same shape one level up: mutating `ceil(g
   − ½)` to `ceil(g)` — a real off-by-one whenever a grid's device extent is
   non-integral — left the entire `pixelflow-core`/`pixelflow-graphics`
   test suites green, `the_border_range_agrees_with_the_mask_it_replaced`
   included, because every fixture in the corpus happened to have an
   integral grid extent.
7. **A bit-exact `f32` golden is not stable across targets or ISA levels of
   the same machine.** L2's first attempted golden (an FNV-1a-64 hash of raw
   coverage bytes) shipped red on both CI runners at a mismatched constant
   because it asserted exactly the cross-ISA identity CLAUDE.md's
   floating-point section says the hardware does not give. Fixed by
   quantizing to the 8-bit form the renderer actually writes and comparing
   with the crate's existing tolerance mechanism, not by loosening a
   bit-exact check in place.
