# A glyph is a formula

## Metadata
- **Author**: JP (direction), Claude (draft)
- **Status**: `Draft` — denotation proposed 2026-09-23. Nothing built.
- **Created**: 2026-09-23
- **Verified against**: `470e0e0e` (`claude/sse-deletion`: main with #1286,
  #1289, #1290, #1291, #1292), the tree
  [the FreeType comparison](../results/2026-09-23-freetype-comparison.md)
  was measured on.
- **Continues**: [loop-blinn-glyph](2026-09-08-loop-blinn-glyph.md) (the
  math, unchanged here), [glyph-as-a-fold-execution](2026-09-09-glyph-as-a-fold-execution.md)
  (the table and the two folds), [one-conditional-three-lowerings](2026-09-08-one-conditional-three-lowerings.md)
  (`Select` lowered as a jump), [demand-is-a-dag-property](2026-09-07-demand-is-a-dag-property.md)
  (§1–§2, control dependence as a DAG property), and
  [one-name-bound-later](2026-09-10-one-name-bound-later.md) (what is bound
  when).

**Decisions it records (JP, 2026-09-22/23):**

> *"We don't need union, we need bind."*
>
> *"Anytime you put some figure inside a box with a select, that's a natural
> bounded volume hierarchy. … Whether or not something's a bounded volume
> hierarchy is determined only by the price of selecting its parent."*
>
> *"We're not doing resolution lazily enough. If you let x be the set of
> integers from 1 to 5, and y be the sum of x, and z be the min of x, the
> fact that y and z reference the same x should make it very easy for the
> compiler to figure out that it only needs one loop."*
>
> *"Ideally what we want to do with the compiler is not blame the user. The
> font code is actually kind of ugly. We want as elegant a mathematical
> definition of how Loop–Blinn font rasterization works as we can write, and
> the compiler smart enough to take that and turn it into ridiculously fast
> code."*

---

## 1. The formula

A glyph is a finite set of **pieces** `P`. A piece `p` is a chord from `a`
to `b` with a direction `σ_p ∈ {−1, 0, +1}` (`0` for a horizontal chord),
and, when the outline curves there, a **bulge**: the affine map
`(u_p, v_p)(x, y)` sending its three control points to `(0,0)`, `(½,0)`,
`(1,1)`, and a sign `τ_p` for the control triangle's orientation.

For a sample `s = (x, y)`:

```text
crossing_p(s) = σ_p · [ y_min_p ≤ y < y_max_p ] · [ x < a_x + (y − a_y)·(dx/dy)_p ]
sliver_p(s)   = τ_p · [ u_p² ≤ v_p ≤ u_p ]                        (u_p, v_p affine in s)
t_p(s)        = crossing_p(s) + sliver_p(s)                        one piece's winding
w(s)          = Σ_p t_p(s)                                         the winding number, an integer
inside(s)     = w(s) ≠ 0                                           TrueType's non-zero rule

d_p(s)        = max( |u_p² − v_p| / ‖∇(u_p² − v_p)‖ ,  capsule_p(s) − δ_p )      pixels; a lower bound
boundary_p(s) = [ w(s) − t_p(s) = 0 ] ∨ [ w(s) − t_p(s) + σ_p = 0 ] ∨ β_p
d(s)          = min( R, min { d_p(s) : boundary_p(s) } )
coverage(s)   = inside(s) ? min(1, ½ + d(s)) : max(0, ½ − d(s))
```

`R` is the ramp's reach, `δ_p` the curve's deviation from its chord, `β_p`
whether the piece can never be interior to another contour. Every bracket is
a hard mask; the winding is exact and only `d` is soft. This is exactly what
`pixelflow-graphics/src/fonts/loop_blinn.rs` computes today, and the math is
not what this plan changes. [loop-blinn-glyph](2026-09-08-loop-blinn-glyph.md)
§1–§2 and §7 are its derivation and its defects, and they stand.

Two facts about the formula matter for everything below:

- `w` and `d` are two reductions over **one set**. `d`'s body reads `w`,
  the *finished* sum, so `d` cannot be computed in the same pass as `w`;
  but both range over the same `P`, and nothing in the formula says how
  many times to loop, in what order, or how many pieces there are.
- Every piece's terms are nonzero only on a **box**: `crossing_p` inside
  the chord's `y`-band and to its left, `sliver_p` inside the control
  triangle, `d_p` within `R + δ_p` of the chord. Outside a piece's box its
  `t_p` is exactly `0` and its `d_p` is at least `R`, which are the
  identities of `Σ` and `min` — so restricting either reduction to the
  pieces whose box contains `s` changes no bit of the answer.

## 2. What the code says instead, and why that is the defect

`loop_blinn.rs` is 1,400 lines, of which the formula is about sixty. The
rest is the author doing the compiler's job, and each part is load-bearing
today:

| in the code | what it is | whose job |
|---|---|---|
| a 22-column piece table, one row per piece | the set `P` as data | the author's, rightly |
| `sum_over(count, …)` and `min_over(count, …)`, each with its own binder | two loops, chosen at authoring time | the compiler's |
| `bucketed_trip_count`, `padding_row` | making unrelated glyphs share a program by rounding the loop's trip count up and filling with identities | the compiler's (a trip count is a shape) |
| `Kernel::by_ref` on the winding | sharing one reduction's result with another's body without copying the loop | the compiler's (that is what a DAG is) |
| `run`'s spans and offsets, `Glyph::over`'s masked sums and mins | one table for a run, one box per glyph, the pruning by hand | the compiler's, given the boxes |
| `Support`, `Support::around`, `contains()` | the box outside which the glyph is exactly zero | data — the author knows it, the compiler should use it |
| `text_cells` (deleted), `Union` (deleted) | the same pruning one level up, by hand | the compiler's |

None of it is wrong. All of it is *early*: the loop structure, the trip
counts, the sharing and the pruning are fixed when the kernel is written,
by someone who had to know how `extract_folds`, the JIT cache key and the
guard analysis behave. That is blaming the user. And it does not even
achieve what it is for: the measurement in
[the FreeType comparison](../results/2026-09-23-freetype-comparison.md) §3–§4
shows the per-glyph boxes prune nothing, so a 50-glyph run costs every
glyph's folds at every pixel (230× FreeType), and a single glyph spends
100–250 ns per ink pixel evaluating all 64 bucketed pieces twice (14–21×).

## 3. What the author should write

The formula of §1, over a **set that is a value**:

```text
let pieces = table(P)                        a set of pieces: data, no trip count, no bucket
let t      = pieces.map(p ↦ crossing(p) + sliver(p))
let w      = t.sum()
let d      = pieces.map(p ↦ boundary(p, w, t(p)) ? dist(p) : R).min() ∧ R
coverage(w, d)
```

and, because the author *does* know the geometry, the set's boxes:

```text
pieces = ⋃ subsets, each with a box;  a subset is again a set with boxes, or a leaf
```

That is the whole description. Two things the author says (the formula and
the boxes), and three things the author no longer says: how many loops,
their trip counts, and where to prune. The set's contents — which pieces,
which boxes — are per glyph and arrive at bind time; the program is per
shape.

## 4. What the compiler must do

Three capabilities, each a denotation the language nearly has.

### 4.1 A range is a value

Today a reduction is `Reduce { fold: Fold { monoid, binder, lo..hi, stride }, body }`:
the index set lives *inside* each reduction. Two reductions over the same
pieces are two nodes with two binders, and `extract_folds` makes two loops
because nothing in the DAG says they are one range.

Make the range a node — call it `Range(lo, hi, stride)` or let the `Binder`
leaf name an index set — and let `Reduce(monoid, range, body)` refer to it.
Then "y and z reference the same x" is structural sharing, which
hash-consing gives for free, and **fusion is a scheduling fact**: sibling
reductions over one range whose bodies do not depend on each other's results
are one loop scope with one accumulator each. The allocator already places
a fold's binder and accumulator as roots; two accumulators are two roots.
No product monoid, no tuple value. Where one body reads the other's result
(the boundary test reads `w`), the dependency sequences them: two passes
over one range, sharing the range's binder and bounds, and the second pass
is a leaf of the same box as the first.

This is where "resolution lazily enough" lands: the loop is decided at
schedule time from every reduction that reads the range, not at authoring
time per call to `sum_over`.

### 4.2 A box is a branch

`select(in_box, e, identity)` where `in_box` is uniform over a batch **is a
jump** — CLAUDE.md's "Select contains an if", and the guard machinery already
emits it. A box nested in a box nested in a box is then a bounded volume
hierarchy: a batch outside a node's box skips the node's whole subtree, and
the work per batch is the depth plus the pieces at the leaf it lands in —
logarithmic in the piece count, as JP said, and *no new mechanism*.

What stops it today is measured, not guessed
([results §4](../results/2026-09-23-freetype-comparison.md)): the guard
analysis refuses to let an arm own a **root** — a value some scope inside
reads — because "a guard skipping the arm would leave the value unwritten
for a loop that runs regardless". The winding's result is read by the
distance fold's body, so it is a root of the batch scope, and the glyph's
own box owns neither fold. The rule is right when the reading loop is
outside the arm and wrong when it is inside: **an arm may own a root when it
owns every scope that reads it.** With 4.1 the two passes are one range's
loops and the question dissolves; without it, this one-line change to
ownership is what lets a text run stop being linear in glyphs per pixel.

Two consequences the emitter already half has:

- A box test that depends on `Y` alone is invariant over the batch and over
  the column fold, so its jump belongs once per row, where placement
  already puts its *value*; the branch should land there too (lowering 2 in
  [one-conditional-three-lowerings](2026-09-08-one-conditional-three-lowerings.md)).
- The mispredict bound (`MISPREDICT_PENALTY_CYCLES`, 16 cycles) is not the
  blocker: a node's arm is a fold, priced at its trip count, far above it.
  Whether a box is worth a branch *is* decided only by its arm's price, as
  the decision says.

### 4.3 The set's contents and hierarchy are data

The program must stay one per font, not one per glyph — H1 in the backlog
is what made the terminal usable. So the tree's **shape** (depth, leaf
capacity) is part of the program and its **contents** (which pieces at each
leaf, each node's box) are the piece table, bound per glyph. The host builds
the tree the way it builds the piece rows today: a pass over the outline,
sorting pieces into boxes, padding leaves to the capacity with the identity
row. The per-leaf reductions read `row = leaf[node, i]` — a `Broadcast`,
since `node` is decided by the box tests, which are uniform over the batch.

This is "bind" in JP's sense: the active-piece list FreeType maintains per
scanline is data the host already has, bound at bind time, and read through
the same `Broadcast` a table read is today. Nothing is derived by the
compiler that the author knows; nothing is fixed by the author that the
compiler decides.

## 5. What is subtracted

From `loop_blinn.rs`: `bucketed_trip_count` and `padding_row` (a leaf's
capacity is the program's one bucket; the identity row stays as the leaf's
filler), `run`'s spans and offsets (the tree's root has children, which is
what a run is), `Kernel::by_ref` on the winding (4.1 shares it), `Glyph::over`'s
masked sums and mins (the tree), `Support`/`Support::around`/`contains()`
(the root box is a node like any other). From the compiler: the ownership
refusal of 4.2, once 4.1 makes it moot.

`Winding`/`Distance` and the piece row's columns stay: they are the formula.

## 6. Measure

The baseline is [the FreeType comparison](../results/2026-09-23-freetype-comparison.md):
`8`@32 collapse 153 µs (AVX-512), `A` 79 µs, a 50-glyph run 66 ms, FreeType
4.9–7.8 µs per glyph, the per-pixel budget 20–40 lane-instructions. What to
expect, in order:

| step | what changes | expected on `8`@32 |
|---|---|---|
| 4.2's ownership rule alone | a glyph's box skips both its folds | text runs linear, not quadratic; single glyph unchanged |
| a two-level tree (root + bands of 4 rows) as data | 128 trips per batch → ~16–32 | 153 µs → ~40–60 µs |
| depth log₂(pieces) | trips per batch ≈ depth + leaf | ~20–30 µs, within 3–4× of FreeType |
| 4.1 fusion | one range for both passes | the second pass shares the first's binder, bounds and reads; the win is code, then time |

Every step bit-exact against the goldens, `loop_blinn_winding` 9/9,
`freetype_oracle`'s pins unmoved: restricting a reduction to the pieces whose
box contains the sample is an identity by §1's second fact, so any changed
texel is a bug in the tree, not a rounding.

## 7. Open questions

- **The boundary test's dependency.** `d` needs the finished `w`. Two
  passes over the leaf's pieces is fine; whether the test can be restated
  to need only the *leaf's* winding (the pieces outside the leaf's box
  contribute a known constant to `w` at every sample inside it) would make
  the second pass local. Not needed for the win; noted.
- **What the range node is.** A new `Range` leaf, or `Fold`'s bounds pulled
  out of `Reduce` into a node both reductions point at. The second is the
  smaller change to `pixelflow-ir`; the first is the honest denotation.
  Decide at 4.1.
- **Leaf capacity and band height** are the tree's shape and so the
  program's key: a font-wide choice, like the bucket today. Measure two.

## 8. Not here

- A `Union` type, index ranges, or any compiler-derived region: the boxes
  are data the author has.
- Compile-time evaluation of masks (D1–D4): unrelated to fonts, stays where
  it is.
- Changing the coverage math, the ramp, or the winding rule.
