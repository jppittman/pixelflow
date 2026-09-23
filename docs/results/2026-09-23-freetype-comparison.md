# FreeType comparison, 2026-09-23: the gap is the kernel's work per ink pixel

**Host:** one x86-64 core with AVX-512 (F, DQ, BW, CD, VL), FreeType 26.1.20.
**Tree:** `claude/sse-deletion` at `470e0e0e` — `main` (`250822e6`) plus
#1286 (constants are values), #1289 (broadcast load), #1290 (pointer class),
#1291 (ISA decided at startup), #1292 (SSE2 tier deleted). The AVX2 column
is the same binary under `PIXELFLOW_ISA=avx2`.
**Harness:** `cargo bench -p pixelflow-graphics --features freetype -- --warm-up-time 1 --measurement-time 3`
(`pixelflow-graphics/benches/font_rendering.rs`), release, quiet machine.
Criterion medians; the raw lines are in the appendix.

**Corrected the same day.** The bench had set FreeType's size in points at
96 dpi — 32 pt is 42.7 px, a 22×32 bitmap for `8` — against pixelflow's 32
px per em (16×23), and it ran FreeType's TrueType bytecode interpreter,
which pixelflow has no counterpart to. It now sets pixel sizes and reports
FreeType **hinted** and **unhinted**; the unhinted row is like-for-like. The
first numbers recorded here (FreeType 4.9/6.3/7.8 µs for `A`/`O`/`S`) were
the 42.7 px hinted glyphs, and understated the gap by 1.5–2.7×. Every
FreeType number below is from the fixed bench; pixelflow's are unchanged.

The question (JP, 2026-09-22): *"my dream was that we're so ALU that we can
remat font via computation at memory speed."* This is where that stands after
the H6 stack, and what is between here and there.

## 1. Single glyph, 32 px, over a 40×45 lattice

`glyph.bake(&kernel, Lattice::frame(40, 45))`, warm: the compile is cached,
so an iteration is one collapse. FreeType renders the glyph's own bitmap
(`load_char(RENDER)`, 32 px per em), which is roughly the ink box, not the
whole lattice.

| glyph | pixelflow AVX-512 | pixelflow AVX2 | FreeType unhinted | FreeType hinted | AVX-512 / unhinted |
|---|---|---|---|---|---|
| `A` (lines) | 89 µs | 88 µs | 1.8 µs | 3.5 µs | 49× |
| `O` (quadratics) | 87 µs | 116 µs | 3.6 µs | 4.3 µs | 24× |
| `S` (many pieces) | 166 µs | 227 µs | 3.9 µs | 5.5 µs | 42× |

### What FreeType's number contains

`load_char(RENDER)` is three things, and only one of them is rasterizing.
Timed separately (a scratch example, 4,000 iterations each, same host;
`NO_HINTING` alone for the load, `DEFAULT` for load plus hinting,
`FT_Glyph_To_Bitmap` on a copied outline for the raster alone):

| 32 px | load (`glyf` → outline) | TrueType hint VM | raster | whole `load_char(RENDER)` |
|---|---|---|---|---|
| `A` | 0.5 µs | 1.7 µs | 1.5 µs | 3.8 µs |
| `O` | 0.4 µs | 1.5 µs | 3.0 µs | 4.1 µs |
| `S` | 0.5 µs | 1.8 µs | 3.2 µs | 5.3 µs |
| `8` | 0.6 µs | 1.5 µs | 4.3 µs | 6.2 µs |

- **The hint VM** is FreeType's TrueType bytecode interpreter running the
  font's glyph programs (DejaVu Sans Mono carries `fpgm`, `prep` and `cvt`).
  pixelflow does not hint. It is 1.5–1.8 µs per glyph, flat in size, and it
  is why the hinted row is *not* the unhinted row plus the VM: a
  grid-fitted outline has fewer partially covered cells and rasterizes
  cheaper, so for `O` the hinted total is within 0.7 µs of the unhinted.
- **Gamma is not in this path.** `FT_Render_Glyph` in normal mode writes
  linear coverage; gamma-correct blending is the client's, and stem
  darkening (FreeType's substitute for it) is off by default. Neither side
  of this comparison does gamma.
- So the like-for-like number is pixelflow's **collapse** against FreeType's
  **raster alone**: `A` 79 µs against 1.5 µs, `8` 153 µs against 4.3 µs —
  **35–55×**.

## 2. Where a bake's time goes

The same bakes, split (`Manifold::compile` cached, `bind`, `Lattice::collapse`),
AVX-512:

| | compile (cached) | bind | collapse | whole bake |
|---|---|---|---|---|
| `A` 40×45 | 6.9 µs | 0.2 µs | 78.8 µs (43.8 ns/px) | 87.5 µs |
| `8` 40×45 | 7.6 µs | 0.2 µs | 152.6 µs (84.8 ns/px) | 165.0 µs |
| `HELLO` 200×45 | 26.1 µs | 0.2 µs | 1,168 µs (130 ns/px) | 1,159 µs |

So the per-call overhead is small — the canonical-key lookup of a ~6,000-node
arena is 7 µs — and the cost is the collapse. The extent sweep says where in
the collapse:

| glyph | 40×45 (1,800 px) | 80×90 (7,200 px) | 160×180 (28,800 px) | marginal, outside the ink |
|---|---|---|---|---|
| `A` | 88.7 µs | 95.2 µs | 126.9 µs | 1.4 ns/px |
| `O` | 89.7 µs | 93.7 µs | 118.0 µs | 1.0 ns/px |
| `S` | 164.2 µs | 215.7 µs | 194.5 µs | ~1.1 ns/px |
| `8` | 162.0 µs | 165.2 µs | 189.0 µs | 1.0 ns/px |

Pixels outside the glyph's box cost about a nanosecond: the outermost
support select is guarded, and a uniformly-false batch jumps over everything.
Nearly all of the 80–160 µs is spent inside the box, on a few hundred ink
pixels — on the order of **100–250 ns per ink pixel** depending on the piece
count. FreeType's raster over its own bitmap is **3–12 ns per pixel** (`A`
1.5 µs over 19×23, `8` 4.3 µs over 16×23).

AVX2 is between 1.0× and 1.4× slower than AVX-512 on the same glyph; the
lane count is not what decides this number.

## 3. Text runs, 16 px, one kernel per run

`text()` is one kernel over every glyph of the run (the `sum` encoding: one
piece table, a winding sum and a distance min per glyph, masked to each
glyph's box, all summed). Lattice `15·n × 24`. FreeType renders each glyph
of the run in turn at 16 px per em.

| glyphs | pixelflow AVX-512 | FreeType unhinted | FreeType hinted | AVX-512 / unhinted | pixelflow ns/px |
|---|---|---|---|---|---|
| 5 | 548 µs | 8.4 µs | 14.3 µs | 65× | 305 |
| 10 | 1.89 ms | 15.1 µs | 24.9 µs | 125× | 526 |
| 26 | 14.5 ms | 49.1 µs | 92.7 µs | 295× | 1,548 |
| 50 | 66.5 ms | 130 µs | 219 µs | 510× | 3,694 |

Per pixel the cost grows **linearly with the glyph count**: every batch
inside the run's box runs every glyph's folds. The per-glyph box selects do
not prune — see §4.

The cached path is the closest: `cached_HELLO` (five glyphs from the atlas,
gathers only, 20 px) is **16.7 µs** against FreeType's 8.4 µs unhinted and
14.3 µs hinted for five 16 px glyphs, with no atlas on FreeType's side. The
gather is not at parity either, and it is the path that does no geometry.

## 4. Why the per-glyph boxes do not prune

`PIXELFLOW_GUARD_TELEMETRY=1` on the `HELLO` kernel (batch scope, 138
entries, 12 selects): the select on the *run's* support box owns 88 entries
and is guarded; every per-glyph box select owns 1 or 2 entries and none of
its glyph's two folds, so no glyph's loop is ever skipped for a batch outside
its own box. Per select, `(exclusive, demand-exclusive, guarded)`:

```
per-glyph winding boxes:  (1, 3, 0)   ×5
per-glyph distance boxes: (2, 2, 0)   ×5
run box:                  (88, 87, 88)
```

The cause is in `analyze_select_guards`: a value read by a scope *inside* the
current one (a root) may not be owned by an arm — "a guard skipping the arm
would leave the value unwritten for a loop that runs regardless". The
winding fold's result is read by name from the distance fold's body
(`boundary_distance(c, &winding)`), which makes it a root of the batch scope.
The rule is right in general and wrong here: the loop that reads it sits
inside the same arm. An arm should be allowed to own a root when it owns
every scope that reads it. With that, each glyph's box would skip both of its
folds, and the text run would stop being linear in glyphs per pixel.

## 5. The budget

Memory speed for the output alone, one core: 2.5–5 gigapixels/s of `f32`
(0.2–0.4 ns/px). At 3 GHz and two 512-bit ops per cycle that is **20–40
lane-instructions per pixel**. The kernel today spends ~650 on `8` (two
folds of 64 trips, 49 and 116 ops per trip, over 16 lanes) — the measured
~230 ns per ink pixel says about one vector op per cycle on this body.

Where the 35× on `8` comes from, in instructions. The collapse is ~41
batches inside the box, each running 128 trips of the two folds: ~430,000
vector instructions, retired at about one per cycle, which is the 153 µs at
3 GHz. FreeType's raster is 4.3 µs — ~13,000 cycles, so ~30,000 scalar
instructions at the two to three per cycle a cell loop with independent
iterations gets. The instruction counts differ by ~14× and the throughput
by ~2.5×. The 16 lanes are spent on: two lanes for one at the box's edge
(a 16 px glyph on 16-lane batches), 64 fold slots for 34 pieces (the
bucket), and the rest on evaluating every piece at every lane of every
batch — while FreeType touches a cell only when a segment crosses it.
Nothing FreeType does is faster than a vector instruction; it issues
fifteen times fewer of them, each on a cell that needed the work.

What moves it, in order of size:

1. **The hierarchy.** A pixel row crosses a handful of `8`'s 34 pieces.
   Boxes nested around subsets of pieces, each a `select` the compiler
   lowers as a jump, make the work per batch logarithmic in the piece count
   rather than linear: `docs/plans/2026-09-23-a-glyph-is-a-formula.md`.
2. **The ownership rule above**, so a box select can own the loops inside it.
   Small, and it is what a text run needs to stop paying for every glyph at
   every pixel.
3. **The lane count.** The kernel is not lane-bound: AVX-512 buys 1.0–1.4×
   over AVX2 here.

The H6 stack (loops instead of unrolling, constants from a pool, broadcast
reads, a pointer class, one ISA per host) did not change this number's
order of magnitude and was not expected to: it removed instruction overhead
so that the algorithmic factor is now the whole gap and can be measured
rather than inferred.

## Appendix: criterion medians

AVX-512 (detected tier):

```
pixelflow_single_char/A_linear         89.412 µs
pixelflow_single_char/O_quadratic      87.353 µs
pixelflow_single_char/S_complex       166.21  µs
pixelflow_text_sizes/sum/5            548.02  µs
pixelflow_text_sizes/sum/10             1.8937 ms
pixelflow_text_sizes/sum/26            14.496  ms
pixelflow_text_sizes/sum/50            66.484  ms
pixelflow_caching/uncached_HELLO       31.485  ms
pixelflow_caching/cached_HELLO         16.716  µs
pixelflow_caching/cache_warmup_alphabet 165.57 ms
```

FreeType, fixed bench (pixel sizes; `hinted` = `RENDER`, `unhinted` =
`RENDER | NO_HINTING`):

```
freetype_single_char/A_linear/hinted      3.5467 µs
freetype_single_char/A_linear/unhinted    1.8230 µs
freetype_single_char/O_quadratic/hinted   4.2620 µs
freetype_single_char/O_quadratic/unhinted 3.5780 µs
freetype_single_char/S_complex/hinted     5.5120 µs
freetype_single_char/S_complex/unhinted   3.9070 µs
freetype_text/hinted/5                   14.275  µs
freetype_text/unhinted/5                  8.4200 µs
freetype_text/hinted/10                  24.947  µs
freetype_text/unhinted/10                15.141  µs
freetype_text/hinted/26                  92.650  µs
freetype_text/unhinted/26                49.114  µs
freetype_text/hinted/50                 218.98   µs
freetype_text/unhinted/50               130.37   µs
```

FreeType, the bench before the fix (32 pt and 16 pt at 96 dpi = 42.7 px and
21.3 px, hinted), as first recorded:

```
freetype_single_char/A_linear           4.8657 µs
freetype_single_char/O_quadratic        6.2629 µs
freetype_single_char/S_complex          7.8356 µs
freetype_text/5                        18.489  µs
freetype_text/10                       32.262  µs
freetype_text/26                      125.07   µs
freetype_text/50                      289.17   µs
```

`PIXELFLOW_ISA=avx2`, same binary:

```
pixelflow_single_char/A_linear         87.991 µs
pixelflow_single_char/O_quadratic     116.21  µs
pixelflow_single_char/S_complex       227.14  µs
pixelflow_text_sizes/sum/5            733.31  µs
pixelflow_text_sizes/sum/10             2.5480 ms
pixelflow_text_sizes/sum/26            21.332  ms
pixelflow_text_sizes/sum/50            96.552  ms
pixelflow_caching/uncached_HELLO       26.828  ms
pixelflow_caching/cached_HELLO         22.097  µs
pixelflow_caching/cache_warmup_alphabet 142.76 ms
freetype_single_char/A_linear           4.0304 µs
freetype_single_char/O_quadratic        5.3361 µs
freetype_single_char/S_complex          6.5335 µs
freetype_text/5                        15.301  µs
freetype_text/10                       27.262  µs
freetype_text/26                      106.93   µs
freetype_text/50                      240.11   µs
```

(FreeType's own numbers differ between the two runs by machine noise; it does
not read `PIXELFLOW_ISA`.)

The extent sweep and the compile/bind/collapse split were taken with two
scratch examples (`glyph_bake_scaling`, `glyph_bake_split`: 200 warm
iterations each, `Instant` around `glyph.bake`, `Manifold::compile`,
`manifold.bind(&[])` and `lattice.collapse`), not committed.
