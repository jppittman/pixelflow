# A glyph is a circle that cannot read its own sign

**Date:** 2026-09-09
**Status:** Denotation. No code. The implementation in
`pixelflow-graphics/src/fonts/loop_blinn.rs` deviates from this, and most of
the deviation is a misuse of the language rather than a limit of it.
**Author:** JP (direction), Claude (draft)
**Follows:** [2026-09-08-loop-blinn-glyph.md](2026-09-08-loop-blinn-glyph.md),
which is what was built. This is what it should have been.

---

## The whole thing

A circle is easy because it arrives as a polynomial whose **sign** is the
answer and whose **magnitude** is the distance:

```rust
let sdf = r.sub(&x.mul(&x).add(&y.mul(&y)).sqrt());
let coverage = half.add(&sdf).clamp(&zero, &one);
```

That is an antialiased circle, and it is the whole program. A glyph is the
same program. It differs in exactly one way, and the difference is worth
stating precisely because everything below follows from it:

> A glyph's boundary is many pieces, so no single polynomial has both
> properties. Its **magnitude** is still available — the distance to the
> nearest piece. Its **sign** is not, because the nearest piece does not know
> which side of the *outline* you are on. The winding number is the repair,
> and it is the only reason it appears.

So:

```
sign      = a fold over every piece      →  ℤ, exact
magnitude = a fold over every piece      →  pixels, soft
coverage  = clamp(½ + sign · magnitude, 0, 1)
```

Two folds and a clamp — and pixelflow already has the fold. `Kernel::over`
is the reduction binder; `sum_over` and `min_over` are one-liners on it. The
per-piece numbers are not an argument list. **They are a manifold**, read at
the binder, exactly the way `fonts/cache.rs` already reads a baked glyph.

```rust
pub fn glyph(pieces: &Kernel, n: u32) -> Kernel {
    // Row i of `pieces` is piece i; column k is its k-th coefficient.
    let c = |k: f32, i: &Kernel| pieces.at(&Kernel::constant(k), i);

    let winding = Kernel::sum_over(n, |i| crossing(&c, i));
    let sign = winding.abs().ge(&one).select(&one, &minus_one);

    let distance = Kernel::min_over(n, |i| {
        bounds(&winding, &c, i).select(&distance_to(&c, i), &far)
    });

    half.add(&sign.mul(&distance)).clamp(&zero, &one)
}
```

Ten lines, and **one body** — not one body per piece. The arena holds a
single `Reduce` node whose fourth child is that body (`arena.rs:550`);
unrolling is downstream, in `passes.rs`. A font is one program, because the
program never mentions any piece's numbers.

## The three pieces

A **piece** is a line, or a quadratic. Each contributes to both folds.

```
crossing(line a→b)  =  (a.y ≤ Y < b.y) ∧ (x_at(Y) > X)   →  ±1 else 0
crossing(quad)      =  crossing(chord)  +  sliver
sliver              =  (v ≤ u) ∧ (v ≥ u²)                →  ±1 else 0
```

`(u, v)` is an affine contramap of `(X, Y)` — six numbers per curve, computed
on the host — chosen so the arc lands on the one parabola `v = u²`. That is
Loop and Blinn's construction, and it does exactly one job: it makes a curved
edge as cheap to ask about as a straight one, by turning *where is the curve*
into *what is the sign of a formula*.

It needs no fence beyond `v ≤ u`, the chord's own half-plane. `u² ≤ v ≤ u`
holds nowhere but `u ∈ [0, 1]`, so the crescent bounds itself. A GPU spends a
rasterized triangle on this. Two comparisons do it here.

`bounds(piece)` — whether a drawn edge is a *boundary* rather than an edge
buried under other ink — is a question the sign fold already answers. With
`w₋` the winding less this piece's own term, the two sides of the piece are
`w₋` and `w₋ + dir`. They differ by one, so they are never both zero, and the
piece separates ink from no-ink exactly when one of them is.

Referring to `winding` inside the `min_over` body is safe: unrolled, the N
copies are structurally identical and depend only on the pixel, so CSE
collapses them to one. That is the ordinary case CSE exists for.

**One extent for the whole font.** `over` takes a static count, so a varying
piece count would mean a varying program. It needn't: pad the table to the
font's maximum with rows that are the monoid identity — `0` for the sum,
`+∞` for the min. A padded row is *data*, so the program is unchanged, and
padding costs only the evaluation of rows that contribute nothing. Which is
precisely what §A removes.

That is the entire denotation. No cell, no union, no index range, no bounding
box, no reference point, no host-side pruning, and no array.

## What the code says instead, and why

`loop_blinn.rs` builds one arena fragment per piece and folds them in Rust.
That is the source of nearly everything ugly in it, and it is not a limit of
the language — it is not using `Kernel::over`.

| in the code | why it is there | with `over` |
|---|---|---|
| `min_of`, a hand-rolled balanced tree | `acc.min(&d)` in a Rust loop copies a growing arena N times: 4.7M nodes and 7s to *construct* a 26-character string | `min_over` — already exists; one body, no copies |
| `may_be_interior`, `chord_winding` | asking `bounds` per piece splices `winding` per piece, so the host precomputes which pieces could possibly need it | `winding` appears once in one body; CSE does the rest |
| `Piece` carrying both f64 and kernel constants | the host needs numbers to prune with, the kernel needs them as constants | one table, one manifold, read at the binder |
| a fresh arena per glyph, per size | every number is a `Const` folded into the fragment | the numbers are data; the program is the same for the whole font |

I asked for a variadic `min` taking a slice. That was wrong twice over: the
binder is the right shape, and it already exists. `Kernel::sum(&[Kernel])`
does take a slice, and its own doc comment says the real fix is a hash-consed
arena, deferred deliberately to land with P7–P9's discrete domains. Nothing
here needs it.

## The two asks that survive

### A. Demand is a property of the DAG, not of the domain

`cells` cuts the frame into rectangles and gives each an `IndexRange` in a
`Union` carrying only the pieces that reach it; per-contour bounding boxes
drop whole characters. Both are *exact* — nothing is approximated — and both
are the host proving one of these and encoding the proof in the domain:

| the fact | what proves it |
|---|---|
| a crossing term is zero outside its chord's Y band | interval analysis on one node |
| a sliver term is zero outside its arc's box | interval analysis on one node |
| a *whole contour's* contribution is zero outside its box | its terms trace a closed loop, so the crossings cancel |

The first two are ordinary range propagation. The third is the interesting
one: it is not a property of any node but of a **fold whose terms trace a
cycle**. A ray crosses a closed contour an even number of times and the signs
cancel. No per-node analysis sees that; something has to know the summands
are a cycle.

In the denotation above this becomes a question about a `Reduce` over a
manifold — which range of the binder can contribute here — and it is what
makes the padding free. See
[2026-09-07-demand-is-a-dag-property.md](2026-09-07-demand-is-a-dag-property.md);
a glyph is its worked example, including its hard case.

Note what does **not** motivate this, because the earlier plan said it did:
the sliver needs no domain-side fence. `{v ≥ u²} ∩ {v ≤ u}` is bounded on its
own. The extent is a performance device throughout, never a correctness one.

### B. Hoist binder-only work out of the pixel loops

Coefficients read from a manifold are opaque to constant folding, so
`‖∇f‖ = √(a² + b²)` — the `Dwrt` normalisation that puts the ramp in *screen*
pixels through any enclosing warp — no longer folds to a compile-time
multiply. It looks like a `rsqrt` per piece per pixel.

It is not, and the reason is exact: `a` and `b` depend on the **binder only**.
`Dwrt` differentiates with respect to X and Y, a gathered coefficient is
constant in both, so the derivative is correct and the magnitude is invariant
across every pixel. It belongs in a prologue outside both pixel loops,
computed once per piece per collapse.

The scanline's failure was LICM hoisting *too eagerly* — Y-only work run for
every segment on every row. This is the same mechanism one level up, where it
is unambiguous: nothing in a collapse can change a binder-only value.

## What this buys, stated so it can be checked

- `min_of` deletes
- `may_be_interior` and `chord_winding` delete
- `cells`, `contour_bounds`, and the `Union` plumbing in `text.rs` delete (§A)
- `TEXT_CELL` — currently chosen by argument rather than measurement — stops
  existing to be chosen (§A)
- a font is one compiled program; a glyph is a table

Roughly 800 lines to roughly 150, no behaviour change, and most of the
deletion needs no compiler work at all.

## What is genuinely ours and stays

Not everything hard is the compiler's fault, and the difference matters.

- **The distance to a quadratic is a cubic root.** It is not computed. The
  code takes the larger of a capsule bound and the implicit's own
  `|f|/‖∇f‖`, and splits by de Casteljau until the bound is tight. A
  numerical trade, deliberately made.
- **The winding is never approximated.** Splitting, dropping, pruning: none
  may change an integer. Only the ramp is a tunable. This is the separation
  that made the `'8'` waist defect unexpressible.
- **Geometry belongs on the host.** Every affine map is applied to control
  points before a kernel exists. Quadratics are closed under affine maps, so
  nothing is lost, and a kernel's constants are its own frame's numbers.
