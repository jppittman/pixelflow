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
  pieces and the crescent, unchanged), [glyph-as-a-fold-execution](2026-09-09-glyph-as-a-fold-execution.md)
  (the table), [one-conditional-three-lowerings](2026-09-08-one-conditional-three-lowerings.md)
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
>
> *"If you want the value of one pixel to depend on the value of another,
> the way this is set up, you should use calculus."* And, on the winding
> number: *"Don't write the winding number. Write the integral."*

---

## 1. The formula

A glyph is a finite set of **pieces** `P`. A piece `p` is a chord from `a`
to `b` with a direction `σ_p ∈ {−1, 0, +1}` (`0` for a horizontal chord),
and, where the outline curves, a **bulge**: the affine map `(u_p, v_p)(x, y)`
sending its three control points to `(0,0)`, `(½,0)`, `(1,1)`, and a sign
`τ_p` for the control triangle's orientation.

The coverage of the pixel at `s` is the area of that pixel under ink, and
it is a sum over the pieces:

```text
C(s) = clamp( | Σ_p  σ_p · A_p(s) + τ_p · S_p(s) | , 0, 1 )
```

`A_p(s)` is the area of the pixel to the left of chord `p`, within the
chord's band. With the pixel `[x₀, x₀+1) × [y₀, y₀+1)` and the chord's line
`X_p(t) = a_x + (t − a_y)·k_p`, `k_p = Δx/Δy`:

```text
A_p(s) = ∫_{[y₀, y₀+1] ∩ [y_min_p, y_max_p]}  clamp( X_p(t) − x₀, 0, 1 ) dt
```

`clamp` of a linear function has an antiderivative, so this is closed form:

```text
G(u) = ½·clamp(u, 0, 1)² + max(u − 1, 0)

A_p(s) = (t₁ − t₀) · mean,   mean = (G(u₁) − G(u₀)) / (u₁ − u₀)   if u₁ ≠ u₀
                                    clamp(u₀, 0, 1)                 if u₁ = u₀   (a vertical chord)
```

with `t₀, t₁` the clipped band and `u_i = X_p(t_i) − x₀`. A dozen ops, one
`select`, no square root, no gradient. It is exact: the trapezoid FreeType
and font-rs accumulate per cell, evaluated at the pixel instead of pushed
into it. The half-open band the point test needed is gone, because a shared
vertex has measure zero.

`S_p(s) = ∫∫_pixel [u_p² ≤ v_p ≤ u_p]` is the area of the pixel inside the
crescent between the chord and the curve. Exact, it is a conic clipped to a
square: the parabola's crossings of the pixel's four edges are the roots of
four quadratics, and a polynomial integral between them. Area is continuous
in those roots, so two pieces disagreeing at a tangency cost a rounding of
area, never the half unit a sign test could lose. To first order it is the
ramp the code already has — `clamp(½ + d_p)` with `d_p` the sound signed
distance of [loop-blinn-glyph](2026-09-08-loop-blinn-glyph.md) §2, signed by
`τ_p` — now *added* rather than minimised. Exact or first order is an
accuracy budget, not a correctness question.

### What the sum does on its own

A pixel deep inside: every far chord contributes exactly `0` or `1` of its
band height, the terms telescope to an integer, `|Σ| = 1`. A pixel on an
edge: one piece is fractional, the rest are integers, the sum is the
coverage. A corner: two fractional terms add — the first-order corner every
one-ramp rasterizer makes. Two chords sharing a vertex: their bands are
complementary and integrate to exactly one across it, because they share the
coordinate. Overlapping contours reach `|Σ| = 2`, and the clamp folds them,
as FreeType's does.

### What is not in the formula

The winding number, and `inside`. The distance field, the minimum over
pieces, the boundary test and the second reduction it forced. The capsule
and the `hypot`. `Kernel::by_ref`. And the coupling argument in
`loop_blinn.rs`'s module doc — "a crossing's existence and its coverage
were one number, so a comparison landing on the wrong side of an edge moved
coverage by half a unit" — which was about a root solve at a tangency. There
is no root solve here; the objection needs re-examining under Loop–Blinn,
and `loop_blinn_winding` and `freetype_oracle` exist to do it.

`Dwrt` stays for exactly one thing: under `Kernel::at` the pixel's footprint
is the Jacobian, and that is what keeps `A_p` the area of one *screen*
pixel at any warp.

### The two structural facts

- Every piece's term is the sum's identity outside the piece's **box**:
  `A_p` outside the chord's band and to its right, `S_p` outside the control
  triangle. Restricting the sum to the pieces whose box contains the pixel
  changes no bit of the answer.
- The sum is **one reduction over one set**, and every term is a function
  of the pixel and one piece. There is no dependency between pieces and none
  between pixels.

## 2. What the code says instead, and why that is the defect

`loop_blinn.rs` is 1,400 lines. The rest of what it says beyond the formula
is the author doing the compiler's job, and each part is load-bearing today:

| in the code | what it is | whose job |
|---|---|---|
| a 22-column piece table, one row per piece | the set `P` as data | the author's, rightly |
| a winding `sum_over` and a distance `min_over`, each with its own binder | two loops, chosen at authoring time | the compiler's — and with §1, one |
| the boundary test, `by_ref` on the winding | one reduction's result inside another's body | gone with §1 |
| `bucketed_trip_count`, `padding_row` | making unrelated glyphs share a program by rounding the loop's trip count up and filling with identities | the compiler's (a trip count is a shape) |
| `run`'s spans and offsets, `Glyph::over`'s masked sums and mins | one table for a run, one box per glyph, the pruning by hand | the compiler's, given the boxes |
| `Support`, `Support::around`, `contains()` | the box outside which the glyph is exactly zero | data — the author knows it, the compiler should use it |
| `text_cells` (deleted), `Union` (deleted) | the same pruning one level up, by hand | the compiler's |

None of it is wrong. All of it is *early*: the loop structure, the trip
counts, the sharing and the pruning are fixed when the kernel is written,
by someone who had to know how `extract_folds`, the JIT cache key and the
guard analysis behave. That is blaming the user. And it does not achieve
what it is for: the measurement in
[the FreeType comparison](../results/2026-09-23-freetype-comparison.md) §3–§4
shows the per-glyph boxes prune nothing, so a 50-glyph run costs every
glyph's folds at every pixel (230× FreeType), and a single glyph spends
100–250 ns per ink pixel evaluating all 64 bucketed pieces twice (14–21×).

## 3. What the author should write

The formula of §1, over a **set that is a value**:

```text
let pieces = table(P)                                     data: no trip count, no bucket
coverage   = area( Σ_{p ∈ pieces} t_p )                    t_p the piece's indicator: crossing + sliver
```

`area` is the integral over the pixel's footprint. The author writes the
field — the sum of the pieces' indicators, which is the winding number as a
*function*, never evaluated — and asks for its area. And, because the author
*does* know the geometry, the set's boxes:

```text
pieces = ⋃ subsets, each with a box;  a subset is again a set with boxes, or a leaf
```

That is the whole description. Two things the author says (the field and
the boxes), and three things the author no longer says: how many loops,
their trip counts, and where to prune. The set's contents — which pieces,
which boxes — are per glyph and arrive at bind time; the program is per
shape.

## 4. What the compiler must do

Four capabilities, each a denotation the language nearly has.

### 4.1 `area` — the integral over the pixel

The sampling adjoint the lattice has been missing: `area(k)` is
`∫∫_footprint k`, the box-filtered sample. It is the antialiasing primitive
for everything, not only glyphs, and it is where the author stops doing
calculus by hand. The compiler lowers it by three rules:

- **Linearity.** `area(Σ) = Σ area`, `area(c·k) = c·area(k)`.
- **A clamped linear form integrates exactly**, through `G` above. That is
  `A_p` — the crossing indicator `[y ∈ band]·[x < X(y)]` is
  `∫ clamp(X(t) − x₀, 0, 1) dt` over the band, and the compiler can derive
  it from the indicator's structure (a product of half-planes in one
  variable each).
- **`area(step(f))` is `clamp(½ − f/‖∇f‖, 0, 1)` to first order**, through
  `Dwrt`, when nothing better is known. That is `S_p` at first order, and it
  is today's ramp, derived rather than written.

The footprint under `Kernel::at` is the Jacobian, and `Dwrt`'s chain rule
already carries it.

### 4.2 A range is a value

Today a reduction is `Reduce { fold: Fold { monoid, binder, lo..hi, stride }, body }`:
the index set lives *inside* each reduction. Two reductions over the same
pieces are two nodes with two binders, and `extract_folds` makes two loops
because nothing in the DAG says they are one range.

Make the range a node — `Range(lo, hi, stride)`, or the `Binder` leaf
naming an index set — and let `Reduce(monoid, range, body)` refer to it.
Then "y and z reference the same x" is structural sharing, which
hash-consing gives for free, and **fusion is a scheduling fact**: sibling
reductions over one range whose bodies do not depend on each other's results
are one loop scope with one accumulator each. The allocator already places
a fold's binder and accumulator as roots; two accumulators are two roots.

With §1 the glyph is one reduction and no longer needs this. It stays right
for the language — it is what "resolution lazily enough" means — and text
runs, which are many reductions over one table, are where it shows first.

### 4.3 A box is a branch

`select(in_box, e, identity)` where `in_box` is uniform over a batch **is a
jump** — CLAUDE.md's "Select contains an if", and the guard machinery already
emits it. A box nested in a box nested in a box is then a bounded volume
hierarchy: a batch outside a node's box skips the node's whole subtree, and
the work per batch is the depth plus the pieces at the leaf it lands in —
logarithmic in the piece count, as JP said, with no new mechanism.

What stops it today is measured, not guessed
([results §4](../results/2026-09-23-freetype-comparison.md)): the guard
analysis refuses to let an arm own a **root** — a value some scope inside
reads — because "a guard skipping the arm would leave the value unwritten
for a loop that runs regardless". The winding's result is read by the
distance fold's body, so it is a root of the batch scope, and the glyph's
own box owns neither fold. §1 removes that read; for every other kernel the
rule is still wrong when the reading loop is inside the arm: **an arm may
own a root when it owns every scope that reads it.** One-line, and it is
what lets a text run stop being linear in glyphs per pixel today.

Two consequences the emitter already half has:

- A box test that depends on `Y` alone is invariant over the batch and over
  the column fold, so its jump belongs once per row, where placement
  already puts its *value*; the branch should land there too (lowering 2 in
  [one-conditional-three-lowerings](2026-09-08-one-conditional-three-lowerings.md)).
- The mispredict bound (`MISPREDICT_PENALTY_CYCLES`, 16 cycles) is not the
  blocker: a node's arm is a fold, priced at its trip count, far above it.
  Whether a box is worth a branch *is* decided only by its arm's price, as
  the decision says.

### 4.4 The set's contents and hierarchy are data

The program must stay one per font, not one per glyph — H1 in the backlog
is what made the terminal usable. So the tree's **shape** (depth, leaf
capacity) is part of the program and its **contents** (which pieces at each
leaf, each node's box) are the piece table, bound per glyph. The host builds
the tree the way it builds the piece rows today: a pass over the outline,
sorting pieces into boxes, padding leaves to the capacity with the identity
row. The per-leaf reduction reads `row = leaf[node, i]` — a `Broadcast`,
since `node` is decided by the box tests, which are uniform over the batch.

This is "bind" in JP's sense: the active-piece list FreeType maintains per
scanline is data the host already has, bound at bind time, and read through
the same `Broadcast` a table read is today. Nothing is derived by the
compiler that the author knows; nothing is fixed by the author that the
compiler decides.

## 5. What is subtracted

From `loop_blinn.rs`: the distance fold, `boundary_distance`, `piece_distance`,
the capsule, `Distance` and its `hypot`, `Kernel::by_ref` on the winding,
`Winding::is_inside`, `coverage(winding, distance)`; `bucketed_trip_count`
and `padding_row` (a leaf's capacity is the program's one bucket; the
identity row stays as the leaf's filler); `run`'s spans and offsets (the
tree's root has children, which is what a run is); `Glyph::over`'s masked
sums and mins (the tree); `Support`/`Support::around`/`contains()` (the root
box is a node like any other). From the compiler: the ownership refusal of
4.3.

The piece row's chord and bulge columns stay: they are the formula.
`COL_DEVIATION` stays only while `S_p` is first order.

## 6. Measure

The baseline is [the FreeType comparison](../results/2026-09-23-freetype-comparison.md):
`8`@32 collapse 153 µs (AVX-512), `A` 79 µs, a 50-glyph run 66 ms, FreeType
4.9–7.8 µs per glyph, the per-pixel budget 20–40 lane-instructions. What to
expect, in order:

| step | what changes | expected on `8`@32 |
|---|---|---|
| §1 as written, flat (one fold, `A_p` exact, `S_p` first order) | 128 trips per batch → 64, and a dozen ops per trip instead of 49 + 116 | 153 µs → ~25–35 µs |
| 4.3's ownership rule alone | a glyph's box skips its fold | text runs linear, not quadratic |
| a two-level tree (root + bands of 4 rows) as data | 64 trips per batch → ~8–16 | → ~6–10 µs, at FreeType |
| depth log₂(pieces) | trips per batch ≈ depth + leaf | under FreeType; the budget is in reach |

Every step bit-exact against the goldens *after* the first, which changes
the antialiasing model and re-baselines them against the oracle:
`loop_blinn_winding` 9/9, `freetype_oracle`'s texel pins re-derived and
argued, never renumbered. Restricting the sum to the pieces whose box
contains the sample is an identity by §1's first structural fact, so from
the second step on any changed texel is a bug in the tree, not a rounding.

## 7. Open questions

- **`S_p` exact or first order.** First order is the code's ramp and costs
  nothing new; exact is four quadratics and a polynomial integral per
  curved piece, continuous in every input. Decide by the oracle's texel
  counts at 7 and 16 px, where the implicit's distance was measured unsound.
- **Overlapping contours.** `|Σ|` clamped is FreeType's approximation and
  seam-free in the interior. A pixel where two edges of two contours cross
  is off by the overlap of two fractions. Acceptable; say so in the module
  doc rather than in a second reduction.
- **What the range node is.** A new `Range` leaf, or `Fold`'s bounds pulled
  out of `Reduce` into a node both reductions point at. The second is the
  smaller change to `pixelflow-ir`; the first is the honest denotation.
  Decide at 4.2, which the glyph no longer waits on.
- **Leaf capacity and band height** are the tree's shape and so the
  program's key: a font-wide choice, like the bucket today. Measure two.

## 8. Not here

- A `Union` type, index ranges, or any compiler-derived region: the boxes
  are data the author has.
- Compile-time evaluation of masks (D1–D4): unrelated to fonts, stays where
  it is.
- Changing the pieces: chords and Loop–Blinn bulges, split at
  `MAX_DEVIATION`, as today.
