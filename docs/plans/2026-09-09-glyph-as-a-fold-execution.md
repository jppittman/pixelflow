# Executing the glyph-as-a-fold rewrite

**Date:** 2026-09-09
**Status:** Plan. Staged, each stage independently landable and gated by the
same test.
**Author:** JP (direction), Claude (draft)
**Executes:**
[2026-09-09-a-glyph-is-a-circle.md](2026-09-09-a-glyph-is-a-circle.md)
**Replaces:** the per-piece arena construction in
`pixelflow-graphics/src/fonts/loop_blinn.rs`

---

## The gate, first

`pixelflow-graphics/tests/loop_blinn_winding.rs` is an independent `f64`
winding oracle sharing no code or constants with the kernel. It is green
today at 9/9 and it is the gate for every stage below. **No stage lands
without it.**

That test is why this rewrite is safe to do at all: the denotation is
unchanged, so a rewrite that passes the oracle computed the same function.
Every other pin (`freetype_oracle`, `text_union_identity`,
`kernel_glyph_optimize`) is a secondary signal and any change to one must be
argued, not renumbered.

## The two gating unknowns, answered

**U1 — does a buffer read compose with a reduce binder? Yes**, and for a
structural reason rather than a lucky one. `.at` splices its argument arenas
verbatim and rewrites only `Var(0)`/`Var(1)` in the receiver, so a binder
embedded in the row argument passes through untouched; `Kernel::over` then
renames its placeholder across the whole composed body, generically, wherever
it sits. `passes::legalize` runs `expand_reduce_owned` **before**
`expand_gather_owned` (`pixelflow-ir/src/passes.rs:67-68`), so the binder is
already `Const(k)` when the gather lowers and each row's address folds to an
immediate.

**It is untested.** No reduce body anywhere in the workspace reads a buffer.
The nearest thing, `reduction_binder.rs:263`, composes `.at()` with a binder
over `Kernel::x()`. That gap is S0's whole purpose.

**U2 — is the read a single tap? Yes.**
`DiscreteManifold::kernel_for(id, w, h)` (`pixelflow-core/src/lattice/mod.rs:505`)
is one `Gather` — nearest-neighbour, floor-then-clamp-to-edge, no
interpolation. At exact in-range integers the floor is a no-op and the clamp
never fires, so it is a true lookup. `cache.rs`'s four-tap bilinear is a
different entry point and is not on this path.

### Three constraints that came with the answers

- **`MAX_BOUND_BUFFERS = 4`** (`manifold.rs:58`) — a compiled `Manifold` may
  declare at most four buffer slots, asserted at compile. One table fits;
  a per-glyph table for a string does not, which forecloses one tempting
  shape for S3 before anyone tries it.
- **`width × height ≤ 2^24`** (`manifold.rs:422`) — gather indices are
  computed in `f32` and are exact only below that. A piece table is nowhere
  near it; worth knowing anyway, because it is a silent-wrong-answer limit
  rather than a panic.
- **`Gather` is e-graph representable** — `Vocabulary::resolve`
  (`pixelflow-search/src/egraph/ops.rs:347`) returns it in both vocabularies:
  *"opaque in both: representable as structure (hash-consing CSE), never
  nameable by a template."* So a table read does **not** make the optimizer
  bail, and the N unrolled copies of `winding` still collapse by CSE. This
  was the largest risk to the whole design.

### One correction to the denotation's claim

The denotation says "one body, not one per piece". That is true of the arena
as constructed, and it is where the 4.7M-node, seven-second blowup goes away.
It is **not** true after `ExpandReduce`: the unrolled arena is N copies, the
same size as today's hand-folded one. The wins are construction cost and
roughly 5× less source — not smaller emitted code.

## S0 — the table

**Deliverable:** a host-side row layout and a `Kernel` that reads it.

Row `i` is piece `i`. Columns are its coefficients. One row layout serves both
a line and a quadratic — a line is a quadratic whose sliver term is the
monoid identity — so there is no per-kind branch in the body, only data.

Provisional layout (widths to be confirmed once U1/U2 land):

| col | meaning | line | quad |
|---|---|---|---|
| 0–3 | chord `a.x, a.y, b.x, b.y` | the segment | the chord |
| 4 | `dx/dy` | crossing slope | crossing slope |
| 5 | `direction` (±1) | crossing sign | crossing sign |
| 6–11 | `(u, v)` contramap, six numbers | — | the sliver |
| 12 | sliver `sign` (±1) | **0** | ±1 |
| 13 | chord-bound deviation | 0 | curve's stray |

A line's sliver term is `0 · mask`, which is the sum's identity, so it costs
arithmetic but no correctness. Whether that is cheaper than a second body
under a second reduce is a **measurement**, deferred to S3.

**Gate:** a unit test that bakes a two-row table and reads both rows back
exactly, under a `sum_over`.

## S1 — the body

*Status.* Landed, in two steps. S1a: the winding as one `sum_over` reading
the table. S1b: the distance as one `min_over` reading the same table
(`PIECE_ROW_COLS` 13 → 22), its body naming the winding by reference —
reading it by value copied the `Reduce` per interior piece and made the
legalized arena quadratic
([composition-is-linking](2026-09-09-composition-is-linking.md) §7).
`prepare`, `Prepared`, `Included`, `min_of`, `Linear::kernel` and the
host-side pruning are gone. Two of the predictions below were wrong and the
code says so:

- **`may_be_interior` and `chord_winding` stay.** The delete list argued "a
  piece nothing else covers always separates" — true of the geometry, not
  of the *test*, which reads the crossing term and so compares `w₋ = w`
  against itself wherever the sample's row is outside the chord's Y band
  (every horizontal chord, and most rows of a near-horizontal one). Asked
  of every piece, it drops those pieces' ramps. It is a correctness gate,
  and it is now a table column (`COL_BOUNDARY_TEST`: `0.5`, or a threshold
  no winding reaches) rather than a host branch.
- **What saturation sees went *up*, not down**: ~500 → ~730 nodes per piece
  before the e-graph, because a table read is a `Gather` subtree where a
  constant was one leaf, and `in_pixels` builds its scale five times before
  CSE. The optimizer's output is a wash per piece (~102 → ~104 nodes) and
  the emitted arena +42 %, all of it address arithmetic on constant indices.

What it bought is the denotation, and it is large: the arena a glyph builds
is **5,990 nodes for any outline** — a 26-character string was 1.56 M —
and construction is flat at ~1 ms (was 196 ms). Through `Glyph::bake`,
warm, ns/px before → after: `O` 2380 → 415 at 16 px, 784 → 392 at 32 px,
582 → 990 at 64 px. The first two are the per-bake fixed cost collapsing
with the arena (the compile cache hashes it every bake); the last is the
per-term cost below, and it is the regression this section predicted.
`lower_dwrt` gained the rule it needed — a tabulation whose index does not
move with the variable is a constant — and a new pin,
`a_fold_may_name_another_fold`: two folds take the *same* binder slot when
the inner one is named, which is sound because expansion is inside-out.

**Deliverable:** `glyph()` as one `Kernel::over` body, per-glyph extent.

Per-glyph extent (`n` = that glyph's piece count), not font-wide. That gets
every deletion below with no padding waste, and leaves the padding question
to S3 where it can be measured instead of assumed.

Port, unchanged in meaning:

- `crossing_term` → reads cols 0–5 at the binder
- `sliver_term` → reads cols 6–12; `{v ≥ u²} ∩ {v ≤ u}`, no triangle
- `distance_to` → capsule bound vs implicit, cols 0–4 and 13
- `bounds` → `w₋` and `w₋ + dir` against zero, naming `winding` once

**Deletes at this stage:**

- `min_of` — `Kernel::min_over` is the fold
- `may_be_interior`, `chord_winding` — one body names `winding` once; the N
  unrolled copies are structurally identical and pixel-only, so CSE collapses
  them
- the f64/kernel-constant duplication in `Piece`

**Gate:** `loop_blinn_winding` 9/9. Plus `freetype_oracle`'s pins unchanged
(`KNOWN_ORPHAN_TEXELS = 0`, `TEXELS_WE_MISS_FAST = 3`).

**Expected regression, and it is not a defect.** The runtime pipeline is
`[LowerDwrt, ExpandReduce, Saturate]` (`pixelflow-search/src/runtime.rs:171`),
and `LowerDwrt` runs first *precisely because* differentiation manufactures
constants — its own comment names the case: "for a straight edge, a constant
`DY(d)` — making the whole gradient magnitude `√(DX²+DY²)` a compile-time
number." With `a`, `b` arriving from a gather, `DX(f) = a` is a load, so that
number is not manufactured and the fold does not happen. The derivative is
still *correct* — a gathered coefficient is constant in X and Y — it just is
not constant-folded.

The fix is ask B (hoist binder-only work out of the pixel loops), not a
revert. Measured after S1b: a regular polygon carries 4 `sqrt` per piece
where it carried 1 — the capsule's, which is per pixel, and three gradient
normalisations (`across`, `along`, and the implicit's — the last built for
a line too, where it folds to 0 at run time). Two of the three are
pixel-invariant after the unroll (`Variance::CONST` indices), which is
exactly what ask B hoists; splitting lines from curves into two folds
would remove the third and is S3's measurement to make.

### The tests this lands on

| test | fate |
|---|---|
| `loop_blinn_winding.rs` (9) | **the gate** — black-box against the oracle, must stay green unchanged |
| `font_antialiasing.rs` (6) | black-box ramp shape; expected to survive |
| `font_rasterization_regression.rs`, `kernel_glyph_golden.rs`, `render_glyph.rs`, `font_orientation_test.rs` | black-box; expected to survive |
| `kernel_glyph_optimize.rs::affine_edge_gradients_fold_to_constants` | **breaks by design.** Asserts `opt_dwrt == 0` and `opt_sqrt ≤ 5`, "one sqrt per edge". There is no longer one fragment per edge to count. Needs a new formulation, not a bumped number |
| `kernel_glyph_optimize.rs::lowered_glyph_ops_are_all_egraph_representable` | at risk on paper; `Gather` resolves in both vocabularies, so expected to pass. If it fails, that is a real finding |
| `freetype_oracle.rs` | pins **exact** texel counts (`KNOWN_ORPHAN_TEXELS = 0`, `TEXELS_WE_MISS_FAST = 3`). Coverage arithmetic changes shape here, so a shift is possible without a real regression. Any change must be argued against the oracle, never renumbered |
| `text_union_identity.rs` (deleted) | thresholds calibrated to today's scheduling noise (`SCHEDULING_NOISE = COVERAGE_STEP/4`). May need recalibration; recalibrating is allowed, loosening past a coverage step is not |
| `loop_blinn.rs`'s own `#[cfg(test)] mod tests` (2) | poke `Piece`/`Pieces` fields directly; rewritten with the representation |
| `production_glyph_arena_dump.rs` (`#[ignore]`) | panics on `Nary`. `ExpandReduce` runs before this sees anything, but confirm rather than assume |
| `golden/glyph_atlas_coverage.ppm` | regenerate if it moves; its own doc already anticipates this |

## S2 — the domain-side pruning, which is not what it looked like

**`cells` is not on any production path.** `core-term` renders through
`GlyphAtlas` → `Font::glyph_kernel_scaled` → `loop_blinn::glyph` — the
*single-kernel* form. `cells` reaches only `text_cells`/`text_union`, whose
sole non-test caller in the workspace is one Criterion bench
(`benches/font_rendering.rs:75`). Nothing on screen has ever gone through it.

That reframes the stage entirely. Deleting `cells` is not a performance
decision and there is no regression to fear; it costs a **demonstration**.
`cells` is what makes G1's dependency on L3 concrete — the worked example of
a domain-side extent — and `text_union_identity.rs` (deleted) is a real correctness
suite over it.

So S2 is not "blocked on ask A" as first written. The honest statement: the
pruning is speculative infrastructure that the denotation makes unnecessary,
and it should be kept until ask A lands *as documentation of the idea*, then
deleted along with the hand-proved theorems it encodes. What S1 must do is
leave that possible — pruning stays a wrapper around the body, never a fact
the body depends on.

## S3 — one program for the font

**Deliverable:** font-wide extent, table padded with monoid identities (`0`
for the sum, `+∞` for the min), so the compiled program is the same for every
glyph and a glyph is a `UniformBlock`-style table write.

This is a **trade, not a win**, and it must be measured: fewer compiles
against evaluation of rows that contribute nothing. The padding waste is
exactly what ask A removes, so S3 is worth much more after §A than before it.

Measure: distinct `jit_cache::entry_count()` deltas over a font bake, compile
wall clock, and collapse wall clock, both ways.

## Order and parallelism

```
U1, U2  ──▶  S0  ──▶  S1  ──▶  gate green  ──▶  land
                                    │
                                    ├──▶ S3 (measure, then decide)
                                    └──▶ S2 (blocked on ask A)
```

S0 and S1 are one thread of work — S1 cannot start until the table reads.
S2 and S3 are independent of each other and both wait on S1.

## What must not change

Carried forward from the denotation, and each one already cost something to
learn:

- **The winding is never approximated.** Splitting, padding, pruning: none
  may change an integer. Only the ramp is a tunable.
- **A distance is not negative.** The chord bound goes below zero on a
  curve's chord; unclamped it makes `½ − d` exceed one.
- **`|f|/‖∇f‖` alone is not a distance.** It underestimates at high
  curvature — a texel a full pixel outside `O`@7px read as an edge texel.
  Keep the capsule bound and the de Casteljau split that makes it tight.
- **Geometry stays on the host.** Affine maps are applied to control points
  before a kernel exists.
- **Pruning is exact or it is not pruning.** A piece dropped must be provably
  zero there, not merely small.
