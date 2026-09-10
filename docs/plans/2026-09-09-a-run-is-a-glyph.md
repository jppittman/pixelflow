# A run is a glyph, and glyphs form a monoid

**Date:** 2026-09-09
**Status:** Executed. §9 records what execution found that §1–§7 did not
anticipate — including one constraint that changed the shape of the change.
**Author:** JP (the direction and the method), Claude (draft)
**Supersedes:** `fonts/text.rs`'s merge argument — not its premise, which is
correct, but the conclusion it draws from it (§4).
**Related:** [a-glyph-is-a-circle](2026-09-09-a-glyph-is-a-circle.md),
[glyph-as-a-fold-execution](2026-09-09-glyph-as-a-fold-execution.md).

---

## 0. The method this document follows

Written down because it is reusable and because the last change skipped it.

1. **Grep the clusters.** Repeated argument groups and accreting name
   families. This is *demand* — evidence a thing is already load-bearing.
2. **Write the problem in plain English**, honestly, no jargon smuggled in.
3. **Read the parts of speech off.** Nouns → types. Verbs → functions (and a
   verb's subject is its *receiver*, which is where free functions leak).
   "or" → sum types. Acts on a, b, or c → ask what they have in common, and
   discriminate: *need the list to name it* → enum; *a property a future
   member could satisfy* → trait; **nothing actually varies** → neither, it
   is a receiver, and reaching for either is the mistake.
4. **Cross-reference 2 against 1.** English over-generates — every noun in a
   sentence looks like a candidate type. The draft says which ones are
   demanded. Build the intersection; **record the rejections**, or the next
   pass re-proposes them.
5. **Then look at the mathematical structure.** What is the functor, what is
   the morphism, what is the invariant. This is where the laws — and the
   tests that check them — come from.

Step 4 is why writing a draft first is not waste. You cannot get YAGNI from
a blank page.

## 1. The clusters

- `Support::shifted_x` — **zero callers.** Built to place a glyph's box along
  a run and never wired up. Anticipated demand, unmet.
- `text()` → `placed_outline()` → `Outline::append` — the merge. One function
  whose whole job is to destroy the per-character structure before the kernel
  is built.
- `glyph.kernel` read at ~11 sites (atlas, cache, five test files). The field
  is the *post-coverage* kernel, which is the one thing that must not compose.
- `Glyph { kernel, support }` — `text()` already returns a `Glyph`, so a run
  and a character already claim to be the same type. The representation does
  not honour it.

## 2. The sentence

> A text run is a sequence of characters, each placed at a pen position. Each
> character is a glyph: a set of closed contours with a bounding box. A
> pixel's coverage is decided by the winding of every contour in the run, and
> by the distance to the nearest boundary. **A character contributes nothing
> to a pixel outside its box** — no winding, because its contours are closed;
> no ramp, because the box is dilated by the ramp's reach.

## 3. Parts of speech

- **Nouns**: run, character, glyph, contour, box, coverage, **winding**,
  **distance**, pen position.
- **Verbs**: *places* (a glyph at a pen), *contributes* (a glyph to the two
  folds).
- **Sum**: none new. The "contributes nothing outside its box" is a
  conditional — a `Select`, which is dispatch, and both arms stay live.
- **a/b/c or letters**: winding combines under `+`, distance under `min`.
  Not "sum and min" — *monoid*. Answered as letters, and `Monoid` /
  `Kernel::over` already exists.

## 4. Cross-reference, with rejections

| candidate | demand in the draft | verdict |
|---|---|---|
| `Winding` | must compose *across* glyphs (`Σ`), carries the integrality invariant that licenses both `0.5` and `1.0` | **build** — demanded by this change |
| `Distance` | must compose across glyphs (`min`); `in_pixels(value, scale)` ×4 with value+gradient always travelling together | **build** — demanded by this change |
| `Glyph` as `{winding, distance, support}` | `shifted_x` dead, `text()` already returns `Glyph`, the merge exists only to fake composition | **build** — it is the monoid |
| `Coverage` | one function, one call site, clamp three lines from where the value is made | **reject, YAGNI** |
| `Row<C>` + `Column` | real (5 fns take `Coeff`; 22 loose `usize`) but *orthogonal* — about how one piece is read, not how many | **defer** to the retype |
| `Crossing` / `Sliver` as types | never travel apart; summed on the next line | **reject** — two constructors of `Winding` |

`Coverage` and the `Row` retype are both rejected *here* and for different
reasons: the first has no demand at all, the second has demand this change
does not exercise.

## 5. Structure

**`Glyph` is a monoid.** `over` combines componentwise: winding under `+`
(identity `0`), distance under `min` (identity `RAMP_REACH`), support under
union (identity `Support::EMPTY`). Associative and commutative because each
component is. `Glyph::EMPTY` is the unit, and `text("")` must equal it.

That gives a free law: **a run's coverage does not depend on character
order.** `over([a, b]) == over([b, a])`, `over([a, over([b, c])]) ==
over([a, b, c])`.

**`Glyph::at` is the morphism** — a contramap on coordinates, which must also
shift the support (this is what `shifted_x` was for). Its law is naturality:
`place(g, d)` sampled at `p + d` equals `g` sampled at `p`.

**Two invariants make the binning exact rather than approximate**, and both
are testable:

1. A closed contour's winding is **0** at every exterior point. So masking a
   glyph's winding to its box loses nothing. `Contour::new` refusing an open
   contour is what licenses this — closure is the precondition, not a
   convention.
2. `Support::around` dilates by exactly `RAMP_REACH`, and a piece further
   than `RAMP_REACH` contributes nothing to the ramp. So masking the distance
   to `RAMP_REACH` outside the box loses nothing either.

## 6. What changes

`text.rs`'s premise is right and its conclusion is wrong. The premise:
**coverage is not additive** — summing per-glyph coverages reaches 2 where
ink overlaps, and 2 is not a coverage. True, and it rules out combining
*coverages*. It does not rule out combining *windings*, which is what the
merge should have done:

```text
winding  = Σ_g   (in_box_g ? winding_g  : 0)            // +   , exact
distance = min_g (in_box_g ? distance_g : RAMP_REACH)   // min , exact
coverage = f(winding, distance)                          // once, at the end
```

Same non-zero rule over all contours the module doc says it wants — and it
bins for free.

- `Glyph` becomes `{ winding: Winding, distance: Distance, support: Support }`.
- `Glyph::kernel` stops being a field and becomes the method that applies
  `coverage` once. ~11 call sites gain `()`. Turning a `pub` field into a
  method tightens the surface, which is the direction the API rule points.
- `Glyph::over(&[Glyph])`, `Glyph::at`, `Glyph::EMPTY`.
- `text()` = `Glyph::over(layout(..).map(place))`. `placed_outline` is
  deleted; `Outline::append` loses its only caller and goes too if nothing
  else wants it.
- `Support::union`; `shifted_x` finally has a caller (or is replaced by a
  general `shifted`).

**Where the win comes from, stated honestly.** Structurally the DAG still
holds every glyph — `Select` is dispatch, both arms stay live. The saving is
entirely codegen's guard skipping an arm no lane selected. This is the best
case for it: a SIMD batch is adjacent pixels almost always inside one
character's box, so mask coherence is near-perfect, and a glyph's ~700 nodes
is far above `MISPREDICT_PENALTY_CYCLES`. It is a data-dependent win, not a
structural one, and the measurement decides whether it arrived.

## 7. Gates

- `font_rasterization_regression`'s `"HELLO"` golden is the differential
  test, free: **merging outlines and summing windings are the same
  function**, so the new `text()` must match the old pixel for pixel within
  float-reassociation tolerance.
- New: order independence (`over` commutes and associates), `text("")` is
  `EMPTY`, and naturality of `at` against the pen offset.
- Unchanged and must stay green: `loop_blinn_winding` (6), `freetype_oracle`
  both arms, `kernel_glyph_golden`, `glyph_atlas_golden`, `font_antialiasing`,
  `font_orientation_test`, `render_glyph`.
- Measure with `examples/text_kernel_cost.rs` (nodes per piece, unchanged
  structurally) **plus** a `Glyph::bake` ns/px sweep over run lengths, which
  is where the guard shows up or does not.

## 8. What this does not do

- Not a fill redesign in the GPU sense. Triangulation loses in a pull model:
  a triangle membership test is ~8 ops against a crossing term's ~5, over a
  comparable count, and the GPU's advantage is the rasterizer's *cull*, which
  is a scatter and has no pull analogue. This is family 3 — tile-binned
  analytic — with the glyph's box as the tile.
- Not the `Row`/`Column` retype (§4).
- Not the atlas. `glyph_kernel_scaled` bakes one character; there is nothing
  to bin. This changes `text()`, whose only callers today are benches, an
  example and two tests — but it is what would make an uncached text kernel a
  plausible screen path.

## 9. What execution found

**`MAX_BOUND_BUFFERS = 4`, immediately.** The first working version gave each
character its own `Glyph` — and therefore its own piece table — and a
five-character run stopped compiling: `Manifold::compile` binds at most four
buffer slots without allocating. §4 of
[composition-is-linking](2026-09-09-composition-is-linking.md) had named this
exact limit as the cost of naming memory a slot at a time; the denotation
above did not connect it, because it reasoned about the *folds* and forgot
that each fold's coefficients arrive through a *symbol*.

The fix is the honest reading rather than a bigger constant: **a piece table
is a tabulation of pieces, and which character a piece came from is a row
range, not a separate buffer.** So `loop_blinn::run(&[Outline])` concatenates
every outline's rows into one table and gives each outline's two folds a row
offset — one bound slot however long the run is, at the cost of one `add` per
fold. `glyph` is now `run` at a single outline, so there is one
implementation rather than two. `a_long_run_binds_one_buffer_slot` is the
regression guard, and it is not a tidiness test: without it a run of more
than four characters does not compile.

**Placement never needed a morphism.** §5 proposed `Glyph::at` with
`Support::shifted`, on the assumption that a character would be built at the
origin and moved. It is not: `layout` already translates each *outline*
host-side before the kernel exists, so every `Glyph` is born placed. So
`shifted_x` gained no caller and was deleted instead — the demand signal
that suggested it (§1) was real about the *concept* and wrong about the
*layer*.

**The sentinel fix came along.** `ALWAYS_A_BOUNDARY = f32::MAX` compared with
`lt` was slated for the deferred retype, but `boundary_distance` was being
rewritten anyway and the two are the same three lines. It is a mask now,
ORed into the test — "always bounds" is the absorbing element of `∨`, not a
threshold no winding reaches, and the latter's correctness was a numeric
argument that enough overlapping contours would falsify.

**One addition to the compiler**, and it earned itself by the §3 test:
`Kernel::fold(Monoid, &[Kernel])`. `Kernel::sum` existed; the distance fold
wanted the same thing under `min`. What those have in common is not "sum and
min" but *monoid*, and `Monoid` already existed — so there is one definition,
with `sum` its `Monoid::SUM` instance.

**Still open, and now better understood.** The per-pixel cost is unchanged
in the *structure* — every character is still in the DAG. Whether the guard
actually fires on the box masks is the measurement §7 asks for and it has
not been run. And at *piece* granularity the same idea needs the table's rows
sorted and a per-region extent, which is what `Union` was and what G1
deleted on a correctness argument without re-deciding the cost half.
