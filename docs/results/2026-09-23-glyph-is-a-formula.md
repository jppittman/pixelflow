# The glyph is a formula, 2026-09-23: exact area, closed, and faster

**Host:** one x86-64 machine, 4 cores of an Intel Xeon @ 2.10 GHz with
AVX-512 (F, DQ). The JIT's tier is the host's, AVX-512; the AVX2 figures
are the same binaries under `PIXELFLOW_ISA=avx2`.
**Tree:** `claude/integral-area-work-ojiej6` plus step 6 of
[an integral is a fold](../plans/2026-09-23-an-integral-is-a-fold.md) §8.
*Before* is `0df10f6` (the compound-transform fix, output-neutral for this
font); *after* is `594ab82` (the formula glyph, the regenerated golden and
the re-baselined pins). Both bench binaries carry the same `font_rendering`
source, `'8'` included.
**Font:** the crate's `DejaVuSansMono-Fallback.ttf`, printable ASCII.

The change ([a glyph is a formula](../plans/2026-09-23-a-glyph-is-a-formula.md)):
a glyph's coverage of a pixel was a ramp on the distance to the nearest
edge, `inside ? min(1, ½ + d) : max(0, ½ − d)`, from two folds — an exact
winding and a distance. It is now the area of the pixel under ink,
`min(|Σ_p σ_p·area(χ_p)|, 1)`, written as an integral over one fold with one
body and closed by the e-graph (`FactorFold`, `NarrowInterval`,
`ArcMoment`), every monotone arc exactly.

## 1. Against the exact area

`pixelflow-graphics/tests/glyph_exact_area.rs`: every printable glyph's
atlas tile against `min(|∫∫ w|, 1)` per texel in `f64`
(`pixelflow-graphics/tests/common/exact_area.rs`; the statistic and the
reference are described in
[the baseline](2026-09-23-glyph-exact-area-baseline.md)).
*Before* is that baseline's table (the SSE2 build, since deleted; the
AVX-512 build agreed with it to 1e-6). *After* is identical to every
printed digit on both tiers.

| size | | max `E_max` | mean `E_mean` | `Σ N₀.₁` | mean centroid shift (x, y) |
|---|---|---|---|---|---|
| 7 px | before | 0.357 (`t`) | 0.0763 | 405 (30.6%) | (−0.006, +0.013) px |
| | after | 0.00089 (`G`) | 0.000007 | 0 | (−0.00003, +0.000002) px |
| 16 px | before | 0.424 (`h`) | 0.0288 | 308 (7.4%) | (−0.005, −0.001) px |
| | after | 0.00095 (`9`) | 0.000003 | 0 | (+0.000002, −0.000003) px |
| 32 px | before | 0.427 (`)`) | 0.0130 | 370 (3.0%) | (−0.000, −0.003) px |
| | after | 0.00095 (`Z`) | 0.000005 | 0 | (−0.000003, +0.000008) px |

(94 inked glyphs and 1322, 4166, 12529 inked texels at each size; "mean
`E_mean`" averages the per-glyph means.) Every inked row got better on both
`E_max` and `E_mean`, and no glyph's centroid is off by more than 0.001 px.
What is left is the snap: coverage within `2⁻¹⁰ ≈ 0.00098` of 0 or 1 reads
0 or 1 (`COVERAGE_SNAP`), so a texel whose exact area is a sliver under
`2⁻¹⁰` reads 0. 105 of the 285 rows are exact to the sixth decimal.

Before → after for the glyphs the plan singles out (`0` is below
`5·10⁻⁷`):

| glyph | 7 px `E_max` / `E_mean` / `N₀.₁` | 16 px | 32 px |
|---|---|---|---|
| `A` (lines only) | 0.241 / 0.0710 / 5 → 0 / 0 / 0 | 0.183 / 0.0214 / 2 → 0.00020 / 0.000007 / 0 | 0.279 / 0.0143 / 8 → 0.00085 / 0.000015 / 0 |
| `8` | 0.167 / 0.0641 / 3 → 0 / 0 / 0 | 0.336 / 0.0368 / 4 → 0.00054 / 0.000008 / 0 | 0.136 / 0.0112 / 1 → 0.00083 / 0.000009 / 0 |
| `S` | 0.197 / 0.0854 / 6 → 0 / 0 / 0 | 0.127 / 0.0215 / 1 → 0.00079 / 0.000014 / 0 | 0.156 / 0.0102 / 1 → 0.00082 / 0.000007 / 0 |
| `O` | 0.140 / 0.0460 / 2 → 0 / 0 / 0 | 0.201 / 0.0195 / 1 → 0.00088 / 0.000014 / 0 | 0.052 / 0.0096 / 0 → 0.00021 / 0.000002 / 0 |

The ratchet (`todays_renderer_is_no_worse_than_its_baseline`) is tightened
to the after rows, with its platform slack kept at `10⁻³` — the width of
the snap, which is what one target can cross and another not.

## 2. The other external checks

- **`loop_blinn_winding`**, an independent winding-number oracle compared
  with `==` at every texel ≥ 0.75 px from the outline: 13/13 pass,
  unmodified — including two overlapping squares (their shared edges are no
  seam; wound oppositely, the overlap is a hole) and a bow tie. The step-5
  design predicted 0 wrong texels after the snap; it is 0.
- **`freetype_oracle`** (hinted FreeType at 16× supersampling): orphans stay
  0. Texels FreeType inks and we do not go from 3 to 2 over the full corpus
  and 3 to 1 over the presubmit subset. The three were the ramp's corner
  error (`{`@7 (2,1) read 0.448, `}`@7 (2,3) 0.452, `}`@7 (1,5) 0.485,
  against exact areas 0.494, 0.516, 0.501); the two left are `{`@7 (2,1)
  and `S`@7 (1,5), whose exact areas are 0.4936 and 0.4942 — which we read
  — and which hinted FreeType reads as 0.504 and 0.510. The ink ratio moves
  from 1.0020 to 0.9998 (full) and 1.0041 to 0.9986 (subset).
- **`glyph_atlas_golden`** is regenerated: 2824 of 42768 texels move, 2446
  by more than an 8-bit step, worst 102 steps; the atlas's ink falls 1.1%
  and its saturated texels go from 322 to 214 — the ramp saturated a texel
  whose centre was half a pixel inside the nearest edge, which at a corner
  or across a stem the ink does not cover. The AVX2 and AVX-512 atlases are
  now bit-identical (0 texels differ; the ramp's differed in up to 2630).

## 3. Closure

`pixelflow-graphics/tests/glyph_is_closed.rs`: for all 95 printable glyphs
at 7, 16 and 32 px and `text("HELLO", 16)`, at the atlas's shape,
`optimize_runtime_arena` is `Some`, `unclosed_integrals` is 0, and the
optimized term holds no `Recip`, `Rsqrt`, `Dwrt` or interval fold. A
glyph's arena is its bucketed trip count and nothing else, so those 286
kernels are seven saturations. With `saturation-telemetry` on:

| program | inserted | rounds | classes at stop | applications | stop | wall (debug) |
|---|---|---|---|---|---|---|
| a glyph, one of six buckets | 103 | 4–5 | 4708–5000 | 5177–6488 | class cap | 97–176 ms |
| `HELLO`, five folds in one graph | 358 | 4 | 5000 | 4053 | class cap | 167 ms |

Every one stops on the class cap, and every one is closed: the closing
phase (the integration family and `ConstantFold` to a fixpoint on the fresh
graph) runs before the full rule set, so the cap only limits how far the
algebra polishes a form that is already closed. One fold with one body is
one `ArcMoment` firing per span; `HELLO`'s five spans close in one graph,
well under the 38 separate arcs the step-5 review found the cap admits.

## 4. What a glyph costs

| | before | after |
|---|---|---|
| columns per row | 22 | 10 |
| pieces over printable ASCII (unpadded) | 1507 / 1633 / 1937 at 7 / 16 / 32 px | 1213 at every size |
| table rows (bucketed) | — | 1672 at every size |
| arena a glyph builds (reachable) | 165 | 100 |
| optimized arena (reachable) | 155 | 142 |
| sqrt in the optimized body | 4 (the capsule, three gradient norms) | 4 (the closed form's four roots) |
| div / recip in the optimized body | not counted | 4 / 0 |
| folds per glyph | 2 (winding, distance) | 1 |

The piece count no longer depends on the size: `MAX_DEVIATION`'s halving
is gone (the area of a curve is exact, however fat), and horizontal pieces
are dropped rather than carried as rows of direction 0. The unpadded
"before" counts are the step-5 critique's, which reproduced exactly.

## 5. Paired A/B, this host

`cargo bench -p pixelflow-graphics --bench font_rendering`, the bench
profile (LTO), the two binaries alternated round by round (before, after,
before, after, … at AVX-512; after, before, … at AVX2), Criterion with a
2 s warm-up and 5 s of measurement, each figure the median of the rounds'
point estimates (three rounds at AVX-512, two at AVX2) with the range.

| bench | AVX-512 before | AVX-512 after | | AVX2 before | AVX2 after | |
|---|---|---|---|---|---|---|
| single glyph `A`, 32 px, 40×45 | 86.3 µs (84.6–89.1) | 57.3 µs (57.1–58.4) | −34% | 97.8 µs (96.3–99.3) | 60.1 µs (59.1–61.0) | −39% |
| `O` | 129.4 µs (126.8–129.9) | 106.3 µs (106.1–113.5) | −18% | 165.5 µs (163.0–167.9) | 114.6 µs (113.1–116.0) | −31% |
| `S` | 252.4 µs (249.7–254.2) | 206.5 µs (205.2–210.3) | −18% | 327.7 µs (326.3–329.1) | 216.7 µs (213.2–220.1) | −34% |
| `8` | 249.9 µs (246.1–262.7) | 217.3 µs (207.8–219.0) | −13% | 325.5 µs (321.1–329.8) | 218.1 µs (216.8–219.4) | −33% |
| `uncached_HELLO`, 20 px, 100×30 | 28.09 ms (27.71–29.91) | 5.70 ms (5.70–5.91) | −80% | 27.93 ms (27.45–28.41) | 5.65 ms (5.63–5.67) | −80% |
| `cached_HELLO` (atlas gathers) | 8.3 µs (8.0–8.7) | 8.0 µs (8.0–9.0) | −3% | 11.8 µs (11.5–12.1) | 9.0 µs (9.0–9.1) | −23% |
| `cache_warmup_alphabet`, 26 bakes | 143.9 ms (142.4–144.4) | 34.3 ms (34.2–36.9) | −76% | 138.4 ms (137.1–139.7) | 33.7 ms (33.1–34.4) | −76% |

The single-glyph rows are one collapse each (the compile is cached): the
exact area is 13–34% cheaper than the ramp it replaces at AVX-512, and
31–39% at AVX2, where the two tiers now run it in about the same time.
`uncached_HELLO` and `cache_warmup_alphabet` also build their kernels and
look them up in the compile cache on every iteration; they fell 4–5×.
That breakdown was not measured. `cached_HELLO` gathers from baked tiles,
computes no geometry, and should not move; at AVX2 it did, by 23%, which
this run does not explain.

Against FreeType the gap is still an order of magnitude: the
[FreeType comparison](2026-09-23-freetype-comparison.md) measured 1.8–3.9 µs
per 32 px glyph unhinted (whole `load_char`), not paired with these runs.

## 6. The guards on `HELLO`

`PIXELFLOW_GUARD_TELEMETRY=1`, compiling `uncached_HELLO`'s kernel. Per
select, `(exclusive, guarded)` entries of its true arm:

| scope | before | after |
|---|---|---|
| batch | 138 entries, 12 selects, 10 guarded entries | 108 entries, 8 selects, 9 guarded entries |
| per-glyph box selects | winding boxes `(0, 0)` ×5 — the winding fold is read by two selects; distance boxes `(2, 2)` ×5 | `(2, 2)` ×4 and `(1, 1)`: each glyph's one fold is skipped for a batch outside its box |
| the run's box | `(119, 0)` | `(89, 0)`, 4 intruders |
| a fold's body | winding 71 entries, 0 guarded; distance 149, one select owning 65, guarded | 131 entries; the row cut owns 109, 0 guarded, 21 intruders (8 leaves) |

**Per-glyph boxes are now guarded**: a glyph is one fold, read by one
select, so exclusivity can own it — what D2 found blocked for the winding
fold. The per-piece row cut is not: its arm owns 109 of the body's 131
entries, but 21 entries the arm does not own sit in the middle of its run,
so one branch cannot span it. That cut is where a pixel stops paying for a
piece whose band misses its row — FreeType's perimeter shape — and it is
the next thing between this and FreeType's speed: the order-refusal in
`pixelflow-codegen/src/emit/guards.rs` (docs/BACKLOG.md, X1), or the
demand regions of D1. The run's box was not guarded before either.

## 7. Not measured, not done

- aarch64: no host. The ratchet's slack and the golden's budget are
  headroom for it, not measurements of it.
- The breakdown of `uncached_HELLO`'s 5× (construction, cache lookup,
  collapse).
- The box tree of a-glyph-is-a-formula §4.3–§4.4, and the row cut's guard
  above: a glyph still costs every piece's closed form at every batch inside
  its box, whether the piece's band reaches that row or not.
