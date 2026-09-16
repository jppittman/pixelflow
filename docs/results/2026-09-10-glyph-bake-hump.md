# The glyph-bake hump: compile is 99.7–99.9% of a bake, and `cluster_select_arms` is 73% of it

**Date:** 2026-09-10
**Verified against:** `3a4c4e3247c90bc1dd980e268f2edec505dd6fd1` ("perf(codegen): the guard
partition is one pass, not a search", 2026-09-10)
**Moved from:** `docs/BACKLOG.md`, "The hump". The H-row index (H1–H5) and each item's current
status stay in `docs/BACKLOG.md` — that table **is** the index. This document is the narrative
and measurements the index now points to; nothing below has been revised in the move.

---

**A glyph bake costs ~331 ms in release** — 31.5 s for a 95-glyph ASCII atlas,
measured at the sha above. `core-term` calls `atlas.warm(&font, ' '..='~')` at
startup (`core-term/src/terminal_app.rs`) and again on every font-size or
density change, so that is ~31 s to launch and ~31 s per resize. **The glyphs
are correct; the terminal is not usable.** Everything in this document is
about that number.

**H2 is answered (2026-09-10, release, this host): a glyph bake is 99.7–99.9%
compile.** Cold vs warm bake, using the fact that `jit_cache` keys compiles by
(canonical key, shape) so a second bake of the same glyph at the same shape
pays only collapse:

```
glyph   px    cold_ms    warm_ms compile_ms compile%
    A   16      172.5        0.2      172.3   99.9%
    O   16      561.8        0.8      561.1   99.9%
    8   16      758.0        0.9      757.1   99.9%
    m   16      531.8        0.5      531.4   99.9%
    A   32      163.0        0.4      162.6   99.7%
    O   32      672.1        1.7      670.4   99.7%
    8   32     2096.0        4.5     2091.4   99.8%
    m   32      639.4        1.6      637.8   99.8%
```

Two consequences, both of which move items in `docs/BACKLOG.md`'s hump table:

- **H1 alone is the fix.** 95 compiles → 1 takes ~31 s to ~2 s; the 95
  collapses that remain total well under a tenth of a second.
- **Compile is superlinear in piece count and depends on the shape.** `A` (11
  pieces) and `8` (34) at 32 px are 163 ms and 2,096 ms — 3.1× the pieces for
  12.9× the time — and `8` costs 758 ms at 16 px against 2,096 ms at 32 px on
  the same pieces.

**And compile is not saturation.** Measured the same day from the telemetry
feature's own `wall_clock_us`:

```
A@32: cold 164.4 ms   runtime saturation 32.2 ms   (~20%)
8@32: cold 2154.0 ms  runtime saturation 31.5 ms   (~1.5%)
```

Both saturations are *identical* — `nodes=2881, classes=5000, apps≈7054,
extracted cost 6573` — which is legalize-last working as designed: the fold
stays folded, so the e-graph sees the same program for an 11-piece glyph and
a 34-piece one. Saturation is a **constant ~32 ms**. `8`'s extra ~1,990 ms is
entirely downstream of it.

**Correction (same day).** So the earlier reading here — "E3/E4/E5 are on the
critical path, compile *is* saturation plus extraction plus emit" — was
**wrong about which term dominates**, and is corrected above.

**X1 is answered, and the hump is one function.** Splitting the compile:

```
glyph  optimize_ms  emit_ms  post_opt_nodes     (optimize = saturate+extract+legalize)
A@32          34.4    130.8           1,870
O@32          33.0    655.3           4,516
8@32          33.1  2,082.0           8,548
```

Optimization is flat ~33 ms — so **extraction is inside the constant too**,
which clears E3. Splitting the emitter, by schedule length `n`:

```
glyph       n  variance_ms  partition_ms  cluster_ms  regalloc_ms  emit_ms
A@32    6,055          0.0           0.4        96.1         19.8    133.0
O@32   17,521          0.1           1.8       493.8         93.0    672.8
8@32   34,993          0.1           1.9     1,539.7        285.1  2,088.4
```

**`guards::cluster_select_arms` is 73% of an entire glyph bake.** And what it
buys is constant — the same 282 bytes of extra code for every glyph, against
a search that grows superlinearly:

```
glyph   with cluster   without   code with   code without      Δ
A@32         131.8 ms   38.4 ms    126,745       126,463   282 B (0.22%)
O@32         658.8 ms  182.8 ms    367,846       367,564   282 B (0.08%)
8@32       2,101.6 ms  549.0 ms    735,238       734,956   282 B (0.04%)
```

With clustering bypassed the whole `pixelflow-graphics` suite is **171/171
green in 25.9 s**, `glyph_atlas_golden` included — so it is a pure
optimization, not load-bearing, and `glyph_atlas_coverage_is_unchanged`
alone goes **31.5 s → 9.8 s (3.2×)**.

**The missing bound is on the search, not on the guard.**
`MISPREDICT_PENALTY_CYCLES` bounds whether a guard *pays at runtime*, and
`guards.rs` argues carefully for it ("bounding the downside by the upside is
enough to keep the analysis honest without a tuned number anywhere"). Nothing
bounds the *search that looks for guards*. Its cost is compile-time and
superlinear in schedule length; its benefit is runtime and proportional to
trips × cycles saved. For a glyph baked once into an atlas at 32×32, a guard
can save at most microseconds — against 1.5 s of searching for it.

Both terms are already computable at the call site: schedule length is `n`,
and trips come from the `LatticeShape` the kernel is compiled for. So this is
a bound to *derive*, not a threshold to tune — which matters, because this
area has already produced three constants whose stated derivations did not
survive measurement (`EXTENT_SLOP`, `CLUSTER_ROUNDS_PER_SELECT`,
`DISC_BAND_ULPS`). **Do not fix this with a size cutoff.**

See **H5** in `docs/BACKLOG.md`'s index, and note that **D5 deletes the
question**: with a demand predicate, values of equal demand are contiguous
*by construction*, so there is nothing to cluster and no round count to
choose. That plan is no longer only an elegance argument — it is the largest
measured item in the tree.

S3's own doc calls itself "a trade, not a win — fewer compiles against
evaluation of rows that contribute nothing." For the **atlas** path that is
too pessimistic: a glyph bakes once into texels and is a gather forever after,
so the padding is a one-time bake cost, not per frame. Worth re-deciding when
H1 is picked up.
