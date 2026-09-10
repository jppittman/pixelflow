# A glyph is a winding number about a reference point

**Date:** 2026-09-08
**Status:** Landed as PR #1213 (draft at time of writing); see §7 for what the
execution found that this plan's §1–§4 did not anticipate, including two
measurements that changed the design after it was written.

**Superseded in part, 2026-09-09.** The title is stale: there is no reference
point. The formulation is a horizontal ray crossing per chord plus Loop–Blinn's
sliver, and the `w_poly(O)`/`shadow` decomposition in §"The shape" and §4's
half-open cones describe a design that was replaced during execution — §7.1's
adversarial verification was of *that* formulation and no longer bears on the
code. §3's domain-side extent is gone too: `loop_blinn::cells`, `text_union`
and `pixelflow-core`'s `Union` were removed, because the crescent fences
itself (`{v ≥ u²} ∩ {v ≤ u}` is empty outside `u ∈ [0,1]`), so the extent was
never load-bearing for correctness and was on no path to a screen. What
remains live in this document is §1, §2, §5, §6 and §7's findings.
**Author:** JP (direction), Claude (draft and execution)
**Executes:** §4 of
[2026-09-06-lattice-is-the-index.md](2026-09-06-lattice-is-the-index.md),
"Loop–Blinn is the shape of a glyph kernel over a domain extent", which
recorded the direction and stated it was not a commitment.
**Depends on:** L3 (#1184, `39f06552`) for the domain-side extent. §4's
own claim is that Loop–Blinn is *only meaningful bounded*, so the union of
index ranges is not an optimization here, it is the thing that makes the
formulation expressible.

---

## The shape

The scanline formulation asks, per segment: **where does this segment cross
the horizontal line `y = Y`?** Every part of that question except the final
comparison against `X` is a function of `Y` alone, so LICM hoists it into
the row prologue and runs it there for every segment on every row — which
is the 33 ns/px at 40 px the lattice plan measured, and the compiler finding
its §5 recorded. For a quadratic the question is also *ill-conditioned*: it
solves a quadratic in `Y` whose existence is decided by a discriminant, and
at a tangency two segments decide their own discriminants differently within
an ulp. That is the `'8'` waist defect (#1187: proved, five fixes refuted,
left open on `main`).

This plan asks a different question, per segment: **does the segment `O → P`
cross this piece of outline?** for a reference point `O` fixed per cell.

For any `O` off every chord line, the winding number decomposes exactly:

```
w(P) = w_poly(O) − Σᵢ σ(O, Aᵢ, Bᵢ)·[P ∈ shadowᵢ(O)] + Σⱼ σ(P0ⱼ, P1ⱼ, P2ⱼ)·[P ∈ Sⱼ]
```

- `i` ranges over the **chords**: every line, and every quadratic replaced
  by its straight chord `P0 → P2`. Those chords form a closed polygon, and
  `w_poly(O)` is its winding number at `O` — one host-side integer.
  `shadowᵢ(O)` is the region chord `i` shadows as seen from `O`: the cone
  spanned by the rays `O→Aᵢ`, `O→Bᵢ`, cut to the far side of the chord's
  line. A sample is in it exactly when the segment `O→P` crosses that chord.
- `j` ranges over the quadratics that curve. Loop and Blinn's
  `(u, v) = (0,0), (½,0), (1,1)` at `P0, P1, P2`, interpolated affinely,
  makes the curve the zero set of `f = u² − v` inside the control triangle;
  `Sⱼ` is the sliver between chord and curve. Replacing a curve by its chord
  adds a closed loop *chord → curve*, whose winding is `±1` on that sliver
  with the sign of the control triangle's orientation.

Every term is affine tests of `(X, Y)` plus one quadratic form. Nothing is a
function of `Y` alone; no ray is ever intersected with a curve.

**The scanline is this with `O` at infinity.** Send `O → (−∞, Y)`: the cone
becomes the horizontal band `y_min ≤ Y < y_max`, the far side of the chord
becomes `X > x_int(Y)`, and the formula becomes the leftward ray-crossing
count. The two formulations are one denotation at two reference points, and
everything the scanline paid — the Y-only hoist, the discriminant — is what
the point at infinity costs.

## 1. The winding is exact; the antialiasing is a separate distance

Every term above is a **hard** mask selecting a signed constant, so the sum
is the exact winding number: an integer. What is soft is one number, the
distance `d` from the sample to the nearest piece of outline:

```
coverage = inside ? min(1, ½ + d) : max(0, ½ − d)
```

This is the exact area of a pixel cut by a straight edge `d` away, and
within a pixel of a vertex it is the corner approximation every
one-ramp-per-edge rasterizer makes.

**Separating them is the point, not a detail.** The scanline made a
crossing's *existence* and its *coverage* the same number — a crossing
contributed a ramp — which is precisely why a decision landing on the wrong
side of a knife edge moved coverage by half a unit rather than by a
rounding. Here a mis-decided anything can only move a pixel by a rounding,
and there is no knife edge left to land on.

`Dwrt` is unchanged and still does exactly its old job: every distance is
divided by the gradient magnitude of its own edge function, `DX`/`DY`
resolved symbolically at bake, so the chain rule carries the scale through
every enclosing `Kernel::at` and the ramp is one *screen* pixel at any glyph
scale. For an affine edge function that whole normalisation folds to a
compile-time constant.

## 2. Where the distance comes from

Per piece, `d` is the larger of two estimates, because neither is usable
alone:

- The **capsule distance to the chord** — perpendicular within the segment's
  span, distance to the nearer endpoint beyond it. Exact for a line. For a
  curve it is a *lower bound* once the curve's deviation from its chord is
  subtracted, and only a bound.
- The **implicit's own distance** `|f|/‖∇f‖`, which is what a GPU Loop–Blinn
  shader antialiases with. Accurate near the arc, unsound away from it.

A lower bound never exceeds the truth, so the larger of the two is the
closer. The capsule and not the distance to the chord's infinite line: a
sample outside a convex corner is far from both segments but close to each
one's *line*, so a minimum over line distances would soften a pixel a corner
away — a dark halo on every serif.

## 3. The domain-side extent

`u² − v` outside its control triangle is not slow, it is **wrong** — it is
the implicit of the infinite parabola. On a GPU the bound is the rasterizer:
the fragment shader never runs outside the triangle. Here it is a
`Union` of `IndexRange`s (L3): the frame cut into rectangles, each carrying
only the chords and control triangles that reach it.

The pruning is **exact, not conservative**, and one line of convexity is
why: for `O ∈ C` and any `P ∈ C`, the segment `O→P` stays in `C`, so a chord
that does not meet `C` shadows nothing in it. Rectangles nothing reaches are
the constant coverage of their interior; those where that constant is 0 are
not placed at all.

**What §4's sketch proposed and this does not do.** §4 said "the
triangulation is the BSP over which each pixel finds its one triangle" and
"the glyph interior is an ordinary triangulation of the control polygon".
Neither is built. §2 of the same plan is the argument against both — the
retired `spatial_bsp` put which-glyph-where in tree structure and that is
what killed fusion; routing should be arithmetic, not search. So the
interior is a winding constant per cell and routing is `index / cell_size`,
which is the shape `CellGrid` already had.

## 4. Half-open cones

A sample exactly on a ray through a vertex must be counted by exactly one of
the two chords meeting there — and by *neither* where the outline turns back
and both cones lie on the same side. Each cone includes its clockwise
boundary ray and excludes its counter-clockwise one, **decided by the
chord's own sign**:

```
σ > 0:  orient(O, A, P) ≥ 0  ∧  orient(O, B, P) < 0
σ < 0:  orient(O, A, P) < 0  ∧  orient(O, B, P) ≥ 0
```

Both chords compute the shared ray's edge function from **one shared node**,
so the two decisions cannot disagree by a rounding.

The tempting simplification — always include the first ray, i.e.
`σ·orient(O,A,P) ≥ 0 ∧ σ·orient(O,B,P) < 0` — is correct at every
pass-through vertex and **wrong at every turn-back vertex**. See §7.1.

## 5. Geometry belongs on the host

`ttf.rs` parses to an `Outline` — contours of line and quadratic segments —
and every affine map (component placement, em scale, screen flip, pen
position) is applied to control points before a kernel exists. Quadratics
are closed under affine maps, so nothing is lost, and a kernel's constants
are its own frame's numbers.

This replaces the old arrangement, where segments became kernels in a
normalized unit square and every transform was a coordinate warp on the
finished kernel. That arrangement is also what forced L2's measured
deviation (§9.2 of the lattice plan): a `½` moved into the arena is a `½`
the e-graph may reassociate.

## 6. Constraints

- **The winding is never approximated.** Splitting a curve, dropping a
  nearly-flat one, choosing a reference point: none of these may change an
  integer. Only the ramp's distance is a tunable.

  **`FLAT_ENOUGH` violates this, and it is the case this line names.**
  `Piece::quad` replaces a quadratic straying less than `1/256` from its
  chord with the chord, which deletes the crescent where the two disagree
  about the winding — an integer, changed, by dropping a nearly-flat curve.
  It is sub-texel in the outline's own frame, but `Glyph::bound` takes the
  kernel to compile, so a contramap may magnify it: a 0.001-unit bulge at
  1000× is about a pixel wide and the kernel renders the chord. Found by
  review on 2026-09-09, unfixed.

  The exact fix is to emit a chord only where the sliver is *empty* rather
  than small — `cross(p0, p1, p2) == 0`, the control point exactly on the
  chord's line, where dropping it changes nothing. Whether that keeps the
  optimization is a measurement nobody has taken: control points are `f32`
  and the em scale is applied to them on the host, so triples that are
  exactly collinear in font units generally are not after scaling. If the
  answer is that almost nothing is exactly degenerate, then the cost of the
  invariant is one sliver and one implicit per quadratic in the font, and
  *that* number is the thing to decide on — not the `1/256` that currently
  buys it by breaking the rule.
- **A distance is not negative.** The chord bound goes below zero on the
  chord of a curve, and unclamped it makes the outside arm `½ − d` exceed
  one. Range is not on the table (CLAUDE.md).
- **Pruning is exact or it is not pruning.** A piece dropped from a cell
  must be provably zero there, not merely small.
- **No triangulation and no BSP.** Routing is arithmetic.

## 7. What happened

### 7.1 The half-open rule has a wrong twin, and it is the obvious one

An adversarial verification pass over the formulation, in exact rational
arithmetic (`fractions.Fraction`, ~600k chord/point checks, ~86k off-spoke
wedge-identity checks, ~24k samples placed *exactly* on spoke lines),
confirmed the decomposition and the rule in §4, and refuted the
σ-multiplied alternative: it is correct at all 2,836 pass-through vertex
instances and wrong at all 2,476 turn-back ones. Smallest counterexample
found: the CCW square `(0,0),(2,0),(2,2),(0,2)` with `O = (3,−1)` outside
it, vertex `B = (2,2)` turn-back as seen from `O`, and `P = (1,5)` on the
ray beyond `B` — true winding 0, the σ-multiplied rule gives −1.

Also worth recording, because the plan's own draft was loose about it: at a
turn-back vertex the two cones do not *partition* the spoke direction — it
is covered zero times or twice. What holds, and what the formula needs, is
that the **signed** sum there is zero.

### 7.2 The GPU's antialiasing distance is unsound on the CPU

`|f|/‖∇f‖` is what every Loop–Blinn shader uses, and taking it as the
distance is the obvious thing to do. Measured: it **underestimates** where
the curve is sharp — a texel a full pixel outside `O` at 7 px read as an
edge texel — and an underestimate past the ramp is not a rounding
difference. It is a saturation failure, and saturation is exactly what the
cell pruning relies on. The independent winding oracle caught it; no
same-form check could have, since both forms would have used it.

The fix is §2's bound, and it carries a requirement back into the geometry:
the bound has to be *tight*, so a segment straying more than half a pixel
from its chord is halved (de Casteljau, exact) until it is not. Measured on
a synthetic ring whose control points sat at its bounding square's corners
(deviation 5 px): before the split rule, every sample within 5 px of a chord
softened to exactly ½.

### 7.3 Flattening the curve for the distance field cost 3×

The first fix for §7.2 was to flatten each curve to a polyline and take the
minimum capsule distance over its pieces. It is correct and it is the wrong
trade: at 1/64-px flatness a 16-px curve becomes eight capsules, and a
curved glyph measured **267 optimized arena nodes per segment against 33
for a straight one**. The bounded implicit brings that to **86**. Recorded
because it is CLAUDE.md's "never hand-roll something worse" in a place where
the worse version looks like diligence.

### 7.4 Two defects the oracle found that review did not

- A soft chord edge applied *outside* its own cone: where the reference
  point sits near a chord's line the cone is thinner than the ramp, so a
  sample half a pixel past the edge fell outside the cone and picked up a
  partial value nothing cancelled. This is what drove the split in §1 —
  with the winding hard, no ramp lives inside a cone, and no reference point
  can bias one.
- A coverage of **1.17**, from the unclamped negative distance now forbidden
  by §6.

### 7.5 What the defect pins became

| pin | before | after |
|---|---|---|
| `freetype_oracle`: texels we ink where FreeType finds none (the `'8'` waist) | 4 | **0**, asserted empty |
| `freetype_oracle`: texels FreeType inks and we leave blank | 5 | **3** |
| `kernel_glyph_optimize`: optimized-vs-raw divergent texels | 29 | **0**, asserted empty |

The first and third are the same defect seen from two instruments, and it is
gone by construction rather than by a fix: there is no discriminant.

The second fell because a **horizontal edge is now antialiased**. The
scanline dropped horizontal segments as contributing no crossing, which is
correct for the winding and wrong for the coverage, and is why an
independent rasterizer found ink at texels this one left blank. The
remaining 3 are pinned, not asserted: they are not this plan's to explain.

### 7.6 The gate

`pixelflow-graphics/tests/loop_blinn_winding.rs`: an independent `f64`
winding oracle — horizontal ray casting, quadratics solved by their own
`y(t)`, sharing no code or constants with the kernel — compared against the
baked kernel at every sample clear of the antialiasing ramp, where coverage
is an exact integer. Synthetic outlines (holes, winding 2, bow ties, both
contour directions, concave bulges), every printable glyph of the test
font, and both encodings at several cell sizes including 1×1.

It is an *external* bound, in CLAUDE.md's sense, and §7.2 is why that
matters: the JIT-vs-interpreter golden and the optimized-vs-raw sweep both
compare this code to itself, and both would have agreed with the wrong
distance.

`quad_tangency_winding.rs` and `kernel_glyph_bake.rs` are deleted — both
were about the scanline's own numerics.

## 8. Non-goals, and what is left open

- **Kerning, shaping, proportional layout.** Unchanged: `text` models
  advances only (`text_union` is gone — see the status note).
- **An open contour is representable.** `Contour::segments` is public and
  re-exported, so a caller can build a contour whose segments neither
  connect nor close — one vertical line yields a filled half-plane inside
  the support rather than a rejection, because winding interprets whatever
  edges it is given as a closed boundary. The invariant lives in prose. It
  wants a private field and a constructor that closes or refuses the chain,
  which is CLAUDE.md's *extend the type* applied to the meaning "these
  edges close". Found by review on 2026-09-09, unfixed.
- **Cubic Béziers.** TrueType is quadratic; CFF is later. The decomposition
  in §"The shape" is not specific to quadratics — a cubic's implicit is a
  different `f` on a different hull — but nothing here builds one.
- **The cell size is argued, not measured.** `TEXT_CELL` is a constant
  chosen by reasoning about batch width and per-summand call overhead
  (L3 measured 1.46× at 15-px-wide cells). It should be a measurement.
- **The atlas still bakes the single-kernel form.** Whether `GlyphAtlas`
  should bake through `cells` is the same measurement question.
- **Wall clock.** This plan changed the shape of the per-pixel work and the
  arena node counts are recorded above, but the benchmark comparison
  against the scanline baseline (`single_char/O_quadratic` 57.1 µs,
  `S_complex` 99.0 µs, `text_sizes/union/26` 750 µs on the execution host)
  is not run. Until it is, the performance claim in §"The shape" is an
  argument about hoisting, not a result.
