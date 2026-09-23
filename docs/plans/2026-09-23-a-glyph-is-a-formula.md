# A glyph is a formula

## Metadata
- **Author**: JP (direction), Claude (draft)
- **Status**: `Draft` — denotation proposed 2026-09-23; the measure's
  syntax and the definite-range rule added the same day. Nothing built.
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
>
> *"Your integrals are missing the wrt. I think this is worth syntax:
> `area[Dwrt(Z)](Z²)`."* And: *"`Dwrt[W, Z]` sane defaults are fine."*
>
> *"Integrals must be definite, over constant ranges."*
>
> *"Per pixel accounting sounds wrong. What we want is an e-graph that can
> factor an integral. This should be using demand. Same with the variable
> hoisting."*
>
> *"What does FreeType have that we don't? How can we not be better?
> Everything is 100% vectorized. We run multi-threaded on the ALU. We can
> do branch-free if we want (the integral will likely be) and we won't have
> memory reads. We're basically all in registers."*

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

`Dwrt` stays, and `area` joins it as the second operator kept through
`Kernel::at` and lowered after composition, so a warp reaches both
operands. The measure is the lattice's — `dX ∧ dY`, the screen pixel — and
`at` never substitutes it; it substitutes the integrand. Where a lowering
rule needs a derivative it takes `Dwrt`, and the chain rule under the warp
is what keeps `A_p` the area of one *screen* pixel at any warp (§4.1).

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
glyph's folds at every pixel (510× FreeType unhinted), and a single glyph
spends 100–250 ns per ink pixel evaluating all 64 bucketed pieces twice
(24–49× FreeType's whole `load_char`, 35–55× its raster alone).

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

### 4.1 `area` — the definite integral over the pixel

The sampling adjoint the lattice has been missing: `area(k)` is the
box-filtered sample, `∫∫_pixel k`. It is the antialiasing primitive for
everything, not only glyphs, and it is where the author stops doing
calculus by hand.

**Every integral is definite, over a constant range.** The range is the
pixel — `[x₀, x₀+1) × [y₀, y₀+1)` in lattice coordinates, the same cell at
every sample — and nothing else: no indefinite integral, no bound that
depends on data, no accumulation from one pixel into the next. The clipped
band `[t₀, t₁]` in `A_p` is not a bound the author writes. It is the
constant pixel range against an integrand that is zero outside the band,
`clamp(X_p(t) − x₀, 0, 1)·[t ∈ band_p]`, and the compiler narrows the range
by the indicator rule below, as an identity. A bound is a fact about the
lattice; the author only ever supplies integrands.

**The measure is a differential form, and it has syntax.** An integral
without its `wrt` is not a value:

```text
area[dX ∧ dY](k)     ∫∫_pixel k dx dy         coverage: the lattice's own measure
area[dZ](k)          ∫ k dz over the pixel     a line integral along one coordinate
area[dZ ∧ dW](k)     the form pulled back      Z, W any values: the Jacobian appears
area(k)              sugar: area[dX ∧ dY](k)   the sane default is the lattice's axes
Dwrt[Z](k)           ∂k/∂Z for any value Z     today's Dwrt(k, axis) is the case Z ∈ {X, Y}
```

`Z` and `W` are values, not axes, so the form is a node's operands like
any other: `Area { integrand, form: [ValueId; k] }`, `k ∈ {1, 2}`, and
`Dwrt` generalizes the same way, its axis becoming a value with `X`/`Y` as
its default. `area[dZ](Z²)` is then a thing the author can write, and it
means what it says: the integral of `Z²` along `Z` across one pixel, which
is `((Z+½)³ − (Z−½)³)/3` when `Z` is a lattice axis and, when `Z` is
`X·s + c`, that times `s` — the pull-back, by the chain rule `Dwrt`
already has. `area[dX ∧ dY]` under `Kernel::at` is *not* pulled back: the
measure is the screen pixel by definition, the integrand is what is
warped, and that is the difference between "the area of the warped shape"
and "the area of the pixel under the warped shape". Both are sayable; the
glyph wants the second.

`Area` is an operator of the language, and its algebra lives where the
language's algebra lives: in the e-graph, as rewrite rules, exactly as
`Dwrt` does (`pixelflow-search/src/egraph/derivative.rs`: one chain-rule
rewrite, a prohibitive price so the extractor never keeps a `Dwrt`, and
`LowerDwrt` for what the rules did not reach). Like `Dwrt` it is kept
through `Kernel::at` and reaches the e-graph after composition (CLAUDE.md,
"The macro tier does not resolve `Dwrt`"). The e-graph **factors the
integral**; `area` is not lowered by a pass that pattern-matches shapes.
The rules, each an identity of the integral:

- **Factoring.** `area[ω](f · g) = f · area[ω](g)` when `f` is invariant
  along `ω`'s coordinates, and `area[ω](c) = c · |range|`, which is `c`
  for the unit pixel along lattice axes. This is the rule that does the
  work: it pulls every factor with no `X` in it out of the `dX` integral,
  so what remains under the integral is exactly the part that varies
  along the axis being integrated. A glyph's per-piece coefficients are
  read once, not integrated, by this rule and no other mechanism.
- **Linearity.** `area[ω](Σ) = Σ area[ω]`. The glyph's sum over pieces
  passes straight through, so `area(Σ_p t_p) = Σ_p area(t_p)`, and the
  integral is per piece, inside the fold.
- **An indicator in one coordinate narrows the range.**
  `area[dZ]([lo ≤ Z ≤ hi]·k) = ∫_{[z₀, z₀+1] ∩ [lo, hi]} k dz`. This is the
  band `[t₀, t₁]`, derived, and it is the only place a data value touches a
  bound — as the intersection of a constant range with an interval, whose
  endpoints are two `clamp`s.
- **A half-plane over the clipped square is exact.**
  `area[dX ∧ dY]([a·X + b·Y + c ≥ 0])` is the trapezoid, closed form through
  `G` in §1. That is `A_p`, derived from the indicator's shape rather than
  written: the crossing term `[y ∈ band]·[x < X_p(y)]` is the previous rule
  along `Y` and this one along `X`.
- **A conic over the clipped square is exact.** `area[dX ∧ dY]([q ≥ 0])`
  for `q` of degree two: the parabola's crossings of the square's four edges
  are the roots of four quadratics, and the area between them is a
  polynomial integral (Green's theorem along the boundary). That is `S_p`
  exact, since `u_p, v_p` are affine in `(x, y)` and `u_p² − v_p` is
  exactly quadratic.
- **The Taylor rule.** `area[ω](step(f)) = area[ω]([Taylor_n(f) ≥ 0])`,
  the expansion about the pixel's centre through `Dwrt`, `n` an accuracy
  budget. `n = 1` is a half-plane and lands on the rule above — for a chord
  it is exact, because `f` was already affine; for the sliver it is today's
  ramp, `clamp(½ − f/‖∇f‖, 0, 1)`, derived rather than written. `n = 2` is
  a conic and lands on the rule above that — exact for the sliver, for the
  same reason. Which `n` is §7's open question, and it is an accuracy
  choice made in one place.
- **The midpoint fallback** is legalization, not a rule: `Area` is priced
  prohibitively like `Dwrt`, so extraction never keeps one, and a
  `LowerArea` pass beside `LowerDwrt` replaces a survivor by
  `k(centre)` — the point sample every kernel computes today. So `area` is
  total, and never worse than the status quo.

For the glyph, saturation does the calculus:

```text
area[dX ∧ dY]( Σ_p σ_p · [Y ∈ band_p] · [X < X_p(Y)] )
  = Σ_p σ_p · area[dY]( [Y ∈ band_p] · area[dX]([X < X_p(Y)]) )       linearity; factoring (no X in the rest)
  = Σ_p σ_p · ∫_{[y₀, y₀+1] ∩ band_p} clamp(X_p(t) − x₀, 0, 1) dt      half-plane along X; the indicator narrows dY
  = Σ_p σ_p · (t₁ − t₀) · mean_p                                       the antiderivative G
```

and what the extractor sees is a product of a `Y`-only factor,
`(t₁ − t₀)`, and one `X`-varying factor. For it to *choose* that form the
cost model must price a factor by where it is paid — a `Y`-only value
once per row, an `X`-varying one per batch. `Extraction::chosen_variance`
is the seam; the latency prior today prices a node the same wherever it
is placed, and that is the one thing factoring needs from the extractor.

**Where each factor is evaluated is demand, not a rule of `area`.** The
`X`-varying factor is a clamp — two selects — and its polynomial arm is
demanded under `0 < u < 1`, the chord's `X`-range within the band. Demand
is a property of the DAG
([demand-is-a-dag-property](2026-09-07-demand-is-a-dag-property.md)
§1–§2, carried into
[one-conditional-three-lowerings](2026-09-08-one-conditional-three-lowerings.md)):
a batch where that predicate is uniformly false skips the arm by the
existing lowering of a demand region as a jump, and a batch entirely to
the right of the chord observes only the constant arm. Nothing in `area`
knows about a batch, a lane or a row. Hoisting is the same property read
along an axis: a factor with no `X` in its variance is a row value,
placed once per row. Both are annotations of the DAG, read — not
accounting done per pixel, by the author or by a rule of the integral.

#### Build order and gate

For the session that builds `area`, in the order the pieces depend on
each other:

1. **The node.** `OpKind::Area` in `pixelflow-ir`, the integrand plus one
   or two form coordinates as operands (binary for `dZ`, ternary for
   `dZ ∧ dW`), with `Kernel::area()` as the `dX ∧ dY` sugar. The lattice
   axes are the coordinates that matter first; a general `Z` is §7's
   question and the glyph does not need it. `Area` survives the macro
   tier the way `Dwrt` does and reaches the runtime tier through
   `Kernel::at` with its integrand warped and its measure untouched
   (`pixelflow-compiler/tests/derivative_under_warp.rs` is the pattern).
2. **The rules**, beside `pixelflow-search/src/egraph/derivative.rs` and
   inert unless the arena holds an `Area`: factoring, linearity, the
   indicator narrowing the range, the half-plane and conic closed forms,
   the Taylor rule through `Dwrt`. `Area` takes `Dwrt`'s prohibitive price
   so extraction never keeps one.
3. **`LowerArea`** beside `LowerDwrt` in `pixelflow-ir`'s passes, in the
   runtime pipeline after saturation: a survivor becomes its midpoint
   sample.
4. **Pricing by placement**: the extractor weights a factor by where it is
   paid — once per row for a `Y`-only value, per batch otherwise — through
   `Extraction::chosen_variance`. Without it, factoring is cost-neutral
   and the extractor has no reason to choose the factored form.
5. **The gate.** A quadrature oracle: random integrands built from the
   rules' vocabulary (polynomials, clamps of linear forms, steps of
   quadratics) over the unit pixel, the compiled `area` against
   high-order numerical quadrature in scalar `f64`, tolerance by the
   rule that fired — the closed forms at `f32` rounding, the Taylor rule
   at its order's remainder, the midpoint fallback pinned as the point
   sample. Plus: no `Area` survives extraction (the `Dwrt survived
   extraction` assertion's twin), and the warp test above. The glyph
   rewrite (§6) starts only when this gate is green.

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
1.8–3.9 µs per glyph unhinted at the same size (its raster alone 1.5–4.3 µs,
the rest is loading and the TrueType hint VM), the per-pixel budget 20–40
lane-instructions. What to expect, in order:

| step | what changes | expected on `8`@32 |
|---|---|---|
| §1 as written, flat (one fold, `A_p` exact, `S_p` first order) | 128 trips per batch → 64, and a dozen ops per trip instead of 49 + 116 | 153 µs → ~25–35 µs |
| 4.3's ownership rule alone | a glyph's box skips its fold | text runs linear, not quadratic |
| a two-level tree (root + bands of 4 rows) as data | 64 trips per batch → ~8–16 | → ~6–10 µs, at FreeType's hinted total |
| depth log₂(pieces) | trips per batch ≈ depth + leaf | at FreeType's raster; the budget is in reach |

Every step bit-exact against the goldens *after* the first, which changes
the antialiasing model and re-baselines them against the oracle:
`loop_blinn_winding` 9/9, `freetype_oracle`'s texel pins re-derived and
argued, never renumbered. Restricting the sum to the pieces whose box
contains the sample is an identity by §1's first structural fact, so from
the second step on any changed texel is a bug in the tree, not a rounding.

### What FreeType has, and the estimate at the end

FreeType's rasterizer and this formula are one theorem applied two ways.
`A_p` *is* Green's theorem for one piece: the row-wise antiderivative of
the piece's crossing density, evaluated at the pixel. FreeType applies it
numerically — deposit each segment's derivative into the one or two cells
it crosses (perimeter work), then integrate along the row with a running
sum, one add per cell. We apply it analytically, per piece per pixel,
sixteen ops each. Both have an area term; the difference is its constant,
and the theorem is not what decides it.

What brings ours to FreeType's shape is not a rule of the integral and
not an accounting: it is the factoring of §4.1, and demand. Factored,
`A_p` is a `Y`-only band height times one `X`-varying factor whose
polynomial arm is demanded only across the chord's `X`-range. So the work
is what the DAG says it is: per row, per piece in the band, the band
height once; per batch, per piece whose `X`-range meets the batch, the
sixteen-op factor; per batch to the right of a piece, the constant arm;
and the descent. That is FreeType's shape — perimeter work for the
fractional part, near nothing for the interior — reached by the e-graph
factoring an integral and the scheduler reading demand, with no
dependency between pixels, and at 32 px a row is one or two batches
anyway. The instruction counts in the estimate below are a consequence
of that structure, stated for the measurement; they are not what the
design is organized around.

What the formula has is everything else. Per piece per batch the integral
form is ~16 vector ops with no branch, no root solve and no memory traffic
beyond the piece's coefficients (a broadcast from L1, with the `Y`-only
parts hoisted to the row); the descent is ~20 ops of uniform box tests
lowered as jumps; and the whole thing runs 16 lanes wide with no
accumulator, no cell buffer and no sweep. FreeType's inner loop is scalar
and serial through its cell list.

Two things FreeType does that are *not* in the comparison: its default path
runs the TrueType bytecode interpreter (1.5–1.8 µs per glyph of hinting,
not rasterizing; pixelflow does not hint), and it does no gamma — normal
mode writes linear coverage and leaves blending to the client, as pixelflow
does. The results file's breakdown separates them; the like-for-like row is
FreeType unhinted, and the like-for-like *work* is its raster alone.

Estimated per glyph, one core, AVX-512, the tree at `log₂(pieces)`:

| size | pixelflow, `A_p` exact + `S_p` first order + tree | FreeType |
|---|---|---|
| 8 px | ~0.2 µs | ~0.25 µs (raster alone, estimated) |
| 32 px | ~0.85 µs at two vector ops per cycle, ~2 µs at today's one | **1.8–3.9 µs unhinted, 3.5–5.5 µs hinted, measured** (`A`, `O`, `S`, whole `load_char(RENDER)`); the raster alone 1.5–4.3 µs |
| 128 px | ~6 µs | ~12 µs (estimated) |

Only the 32 px FreeType column is measured. The estimate's two rows are
the throughput question: today's body retires about one vector op per
cycle, and interleaving independent pieces to reach two is the scheduler's
business. The two caveats that could move it: the per-piece coefficient
reads (22 columns today, a dozen with §5) are L1 broadcasts, not
registers, and the number of pieces per leaf is the tree's shape (§7), so
a badly chosen leaf capacity puts the 32 px number at 2–3 µs rather than
1. So the honest claim is parity with FreeType's raster at one op per
cycle and a factor of two under it at two — not the order of magnitude
the lane count suggests, because the two have the same shape and the lanes
are spent on the fractional terms' sixteen ops, which FreeType's per-cell
deposit matches in scalar. The 24–49× of the baseline is the algorithm;
today not even the atlas gather (`cached_HELLO`, 16.7 µs for five glyphs
against FreeType's 8.4 µs) is under FreeType.

## 7. Open questions

- **`S_p` exact or first order** — the Taylor rule's `n`. First order is
  the code's ramp and costs nothing new; second order is exact for the
  sliver and costs four quadratics and a polynomial integral per curved
  piece, continuous in every input. Decide by the oracle's texel counts at
  7 and 16 px, where the implicit's distance was measured unsound.
- **The form's arity, and a general coordinate.** `k ∈ {1, 2}` covers a
  line integral and coverage; a third coordinate is a volume, which
  nothing here needs. `area[dZ ∧ dW](k)` for *any* values `Z, W` is
  well-defined on the pixel — `∫∫ k · |∂(Z,W)/∂(X,Y)| dX dY`, the Jacobian
  through `Dwrt` — but `area[dZ](k)` for a `Z` that is not a lattice axis
  needs a path across the pixel, and the syntax does not say which. Build
  the axis forms first; decide the general 1-form when something needs
  it, not before.
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
