# FreeType comparison, 2026-09-23: the gap is the kernel's work per ink pixel

**Host:** one x86-64 core with AVX-512 (F, DQ, BW, CD, VL), FreeType 26.1.20.
**Tree:** `claude/sse-deletion` at `470e0e0e` — `main` (`250822e6`) plus
#1286 (constants are values), #1289 (broadcast load), #1290 (pointer class),
#1291 (ISA decided at startup), #1292 (SSE2 tier deleted). The AVX2 column
is the same binary under `PIXELFLOW_ISA=avx2`.
**Harness:** `cargo bench -p pixelflow-graphics --features freetype -- --warm-up-time 1 --measurement-time 3`
(`pixelflow-graphics/benches/font_rendering.rs`), release, quiet machine.
Criterion medians; the raw lines are in the appendix.

The question (JP, 2026-09-22): *"my dream was that we're so ALU that we can
remat font via computation at memory speed."* This is where that stands after
the H6 stack, and what is between here and there.

## 1. Single glyph, 32 px, over a 40×45 lattice

`glyph.bake(&kernel, Lattice::frame(40, 45))`, warm: the compile is cached,
so an iteration is one collapse. FreeType renders the glyph's own bitmap
(`load_char(RENDER)`), which is roughly the ink box, not the whole lattice.

| glyph | pixelflow AVX-512 | pixelflow AVX2 | FreeType | AVX-512 / FreeType |
|---|---|---|---|---|
| `A` (lines) | 89 µs | 88 µs | 4.9 µs | 18× |
| `O` (quadratics) | 87 µs | 116 µs | 6.3 µs | 14× |
| `S` (many pieces) | 166 µs | 227 µs | 7.8 µs | 21× |

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
count. FreeType's 4.9 µs over the same box is about **8 ns per ink pixel**.

AVX2 is between 1.0× and 1.4× slower than AVX-512 on the same glyph; the
lane count is not what decides this number.

## 3. Text runs, 16 px, one kernel per run

`text()` is one kernel over every glyph of the run (the `sum` encoding: one
piece table, a winding sum and a distance min per glyph, masked to each
glyph's box, all summed). Lattice `15·n × 24`.

| glyphs | pixelflow AVX-512 | FreeType | ratio | pixelflow ns/px |
|---|---|---|---|---|
| 5 | 548 µs | 18.5 µs | 30× | 305 |
| 10 | 1.89 ms | 32.3 µs | 59× | 526 |
| 26 | 14.5 ms | 125 µs | 116× | 1,548 |
| 50 | 66.5 ms | 289 µs | 230× | 3,694 |

Per pixel the cost grows **linearly with the glyph count**: every batch
inside the run's box runs every glyph's folds. The per-glyph box selects do
not prune — see §4.

The cached path is the comparison that is already at parity: `cached_HELLO`
(five glyphs from the atlas, gathers only) is **16.7 µs** against FreeType's
18.5 µs for the same five glyphs, with no atlas on FreeType's side.

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
