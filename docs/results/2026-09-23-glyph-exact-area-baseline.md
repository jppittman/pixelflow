# Glyph coverage against the exact area, 2026-09-23: the baseline

**Tree:** `claude/integral-area-work-ojiej6` at `1b19ddd`, SSE2 baseline build
(the workspace's default `rustflags`), x86-64.
**Harness:** `pixelflow-graphics/tests/glyph_exact_area.rs`, with the
reference in `pixelflow-graphics/tests/common/exact_area.rs`.
**Font:** the crate's `DejaVuSansMono-Fallback.ttf`, every printable ASCII
glyph, at 7, 16 and 32 px (`GlyphAtlas` tiles of that many texels).

The question: how far is today's glyph coverage from the area of each pixel
the glyph actually covers? This is the number
[a glyph is a formula](../plans/2026-09-23-a-glyph-is-a-formula.md) sets out
to change, recorded before anything changes it.

## What is compared

**Ours** is the shipped path: `GlyphAtlas::new(size, 1.0, 128)`, warmed with
`' '..='~'`, each glyph's tile read back from the atlas buffer. That is
`Font::glyph_kernel_scaled(ch, size)` contramapped by `(X + ½, Y + ½)` and
baked, so texel `(i, j)` holds coverage sampled at its centre.

**Exact** is `min(|F|, 1)`, where `F = ∫∫ w` over the texel's square
`[i, i+1) × [j, j+1)` and `w` is the outline's non-zero winding number. It
is computed in `f64` from the font-unit outline, mapped to the screen frame by
the reference's own restatement of the font's metrics (ascent at `y = 0`,
descent at `y = size`), and accumulated per texel by Green's theorem:
lines are cut at every texel edge and integrated as trapezoids; quadratics
are halved until each piece sits inside one texel, where its `∫ (x − i) dy`
is integrated exactly as a polynomial, or is within 10⁻⁹ px of its chord.
`|F|` clamped to 1 is FreeType's coverage — exact except where contours
overlap inside one texel, which no ASCII glyph of this font does.

The reference is checked on its own before it judges anything: a rectangle
with fractional edges against interval arithmetic (to 1e-12), a parabolic
segment against Archimedes' two-thirds (to 1e-9 relative, at two scales),
a curved contour with a hole against point sampling at 128 × 128 per texel
(worst difference 5.4e-4), and orientation reversal negating every texel.

**Where the two agree on the pixel.** An integer-aligned square reads
exactly 0 or 1 on both sides (the renderer to within its ramp's gradient
floor, 5e-4). And the ink-weighted centroid of ours against exact, averaged
over the glyphs at a size, is within 0.013 px on both axes; `O`, `0`, `o`,
`H`, `I` are within 0.02 px at 32 px. A half-texel slip between the two
frames reads 0.5 there — injected on purpose, the mean shift read −0.59 and
both the centroid check and the ratchet failed. No glyph's exact ink falls
outside its tile at any of the three sizes.

## The statistic

Per `(glyph, size)`, with `Δ = ours − exact` over the tile:

- `E_max = max |Δ|`
- `E_mean = Σ |Δ| / #{exact > 0}`: the error per texel the exact glyph inks
  ("> 0" is above 10⁻⁸, the reference's resolution; the rises of a closed
  row cancel only to rounding, and leave up to 5e-15 where no contour is)
- `N₀.₁ = #{|Δ| > 0.1}`

`E_mean`'s denominator is the reference's alone, so no statistic can fall
when a texel gets worse. The union of both sides' ink would not do: a
renderer that laid faint ink around every glyph would enlarge it, and read
as better while every tile got worse. A NaN texel counts as an infinite
error; it must not vanish from a `max` and every comparison.

## Summary

| size | inked glyphs | inked texels | max `E_max` | mean of `E_mean` | texel-weighted `E_mean` | `Σ N₀.₁` | mean centroid shift (x, y) |
|---|---|---|---|---|---|---|---|
| 7 px | 94 | 1322 | 0.357 (`t`) | 0.0763 | 0.0741 | 405 (30.6%) | (−0.006, +0.013) px |
| 16 px | 94 | 4166 | 0.424 (`h`) | 0.0288 | 0.0270 | 308 (7.4%) | (−0.005, −0.001) px |
| 32 px | 94 | 12529 | 0.427 (`)`) | 0.0130 | 0.0127 | 370 (3.0%) | (−0.000, −0.003) px |

The glyphs the formula plan singles out:

| glyph | 7 px `E_max` / `E_mean` / `N₀.₁` | 16 px | 32 px |
|---|---|---|---|
| `A` (lines only) | 0.241 / 0.0710 / 5 | 0.183 / 0.0214 / 2 | 0.279 / 0.0143 / 8 |
| `8` | 0.167 / 0.0641 / 3 | 0.336 / 0.0368 / 4 | 0.136 / 0.0112 / 1 |
| `S` | 0.197 / 0.0854 / 6 | 0.127 / 0.0215 / 1 | 0.156 / 0.0102 / 1 |
| `O` | 0.140 / 0.0460 / 2 | 0.201 / 0.0195 / 1 | 0.052 / 0.0096 / 0 |

### Reading it

This is the antialiasing **model's** error, not arithmetic: repeating the
measurement on a `-C target-cpu=native` (AVX-512) build moves `E_max` or
`E_mean` in 20 of the 285 rows, each by 1e-6, and no `N₀.₁`. Coverage today is a ramp on one distance — the
nearest boundary's — so it is exact for a texel cut by one straight edge and
wrong wherever the area needs two: a corner, where two edges each cut the
texel, and a stem thinner than a texel, where both of its sides do. The
worst rows are those glyphs: `t` and `w` at 7 px, `h` and `n` at 16,
`)`, `w` and `g` at 32. The ramp
also lays down more ink than the area almost everywhere: the summed
`ours − exact` over a size's texels is +0.054 per texel at 7 px, +0.006 at
16 and +0.0025 at 32.

`E_mean` falls with size, as it must for an edge error — the texels an edge
crosses grow linearly with size and the texels compared quadratically — but
`E_max` does not: a corner is as wrong at 32 px as at 7.

## The gate

`todays_renderer_is_no_worse_than_its_baseline` is a **ratchet** over the
table below: it fails if any row's `E_max` or `E_mean` rises, or more of its
texels pass `0.1`, by more than `PLATFORM_NOISE` = 10⁻³ — five times the
2.1e-4 single-texel ISA spread `glyph_atlas_golden.rs` measured, a thousand
times the spread measured here. Rows may be beaten freely. The change that
replaces the model (glyph design §5(b), CL-E) re-pins the rows it moves, with
the acceptance criteria written there: aggregate `E_mean` at most half the
baseline at every size, `E_max` no worse for any `(glyph, size)`, `A` at
`E_max ≤ 0.03` with `N₀.₁ = 0`.

To see the measured rows (as the Rust literal the test holds):

```text
cargo test -p pixelflow-graphics --test glyph_exact_area todays_renderer -- --nocapture
```

## Per glyph

| glyph | 7 px E_max | E_mean | N₀.₁ | 16 px E_max | E_mean | N₀.₁ | 32 px E_max | E_mean | N₀.₁ |
|---|---|---|---|---|---|---|---|---|---|
| space | 0.000 | 0.0000 | 0 | 0.000 | 0.0000 | 0 | 0.000 | 0.0000 | 0 |
| `!` | 0.203 | 0.0410 | 1 | 0.180 | 0.0422 | 5 | 0.183 | 0.0080 | 2 |
| `"` | 0.218 | 0.1022 | 2 | 0.224 | 0.0278 | 3 | 0.221 | 0.0103 | 2 |
| `#` | 0.259 | 0.1256 | 11 | 0.289 | 0.0334 | 8 | 0.264 | 0.0164 | 11 |
| `$` | 0.243 | 0.0731 | 7 | 0.203 | 0.0371 | 8 | 0.263 | 0.0203 | 13 |
| `%` | 0.168 | 0.0924 | 9 | 0.190 | 0.0322 | 2 | 0.154 | 0.0156 | 2 |
| `&` | 0.305 | 0.1042 | 7 | 0.359 | 0.0363 | 4 | 0.286 | 0.0190 | 5 |
| `'` | 0.089 | 0.0476 | 0 | 0.232 | 0.0580 | 3 | 0.202 | 0.0145 | 1 |
| `(` | 0.203 | 0.0663 | 3 | 0.167 | 0.0186 | 2 | 0.148 | 0.0110 | 1 |
| `)` | 0.244 | 0.0672 | 2 | 0.282 | 0.0315 | 4 | 0.427 | 0.0183 | 3 |
| `*` | 0.250 | 0.0820 | 4 | 0.211 | 0.0691 | 10 | 0.280 | 0.0272 | 7 |
| `+` | 0.183 | 0.0774 | 4 | 0.069 | 0.0094 | 0 | 0.146 | 0.0058 | 3 |
| `,` | 0.210 | 0.0996 | 2 | 0.114 | 0.0264 | 2 | 0.169 | 0.0162 | 3 |
| `-` | 0.130 | 0.0569 | 1 | 0.217 | 0.0693 | 2 | 0.026 | 0.0022 | 0 |
| `.` | 0.236 | 0.1295 | 2 | 0.173 | 0.0473 | 1 | 0.211 | 0.0228 | 2 |
| `/` | 0.111 | 0.0434 | 1 | 0.102 | 0.0183 | 1 | 0.286 | 0.0134 | 1 |
| `0` | 0.175 | 0.0567 | 4 | 0.199 | 0.0224 | 1 | 0.055 | 0.0084 | 0 |
| `1` | 0.137 | 0.0503 | 4 | 0.196 | 0.0231 | 3 | 0.249 | 0.0110 | 6 |
| `2` | 0.235 | 0.0771 | 4 | 0.253 | 0.0281 | 2 | 0.128 | 0.0121 | 2 |
| `3` | 0.222 | 0.0786 | 6 | 0.268 | 0.0277 | 1 | 0.248 | 0.0126 | 2 |
| `4` | 0.238 | 0.1139 | 7 | 0.268 | 0.0424 | 8 | 0.285 | 0.0167 | 9 |
| `5` | 0.284 | 0.0987 | 6 | 0.226 | 0.0252 | 4 | 0.221 | 0.0098 | 3 |
| `6` | 0.202 | 0.0601 | 3 | 0.154 | 0.0219 | 2 | 0.314 | 0.0125 | 3 |
| `7` | 0.296 | 0.0914 | 4 | 0.184 | 0.0191 | 2 | 0.295 | 0.0132 | 4 |
| `8` | 0.167 | 0.0641 | 3 | 0.336 | 0.0368 | 4 | 0.136 | 0.0112 | 1 |
| `9` | 0.169 | 0.0612 | 3 | 0.188 | 0.0254 | 2 | 0.323 | 0.0110 | 1 |
| `:` | 0.236 | 0.1336 | 4 | 0.185 | 0.0503 | 3 | 0.211 | 0.0198 | 4 |
| `;` | 0.235 | 0.1165 | 4 | 0.185 | 0.0341 | 4 | 0.169 | 0.0164 | 5 |
| `<` | 0.170 | 0.0688 | 5 | 0.251 | 0.0362 | 4 | 0.166 | 0.0126 | 2 |
| `=` | 0.179 | 0.0674 | 3 | 0.210 | 0.0407 | 7 | 0.179 | 0.0084 | 3 |
| `>` | 0.277 | 0.0737 | 4 | 0.258 | 0.0367 | 5 | 0.142 | 0.0135 | 2 |
| `?` | 0.238 | 0.0664 | 4 | 0.197 | 0.0420 | 3 | 0.224 | 0.0161 | 2 |
| `@` | 0.213 | 0.0745 | 7 | 0.096 | 0.0218 | 0 | 0.147 | 0.0122 | 3 |
| `A` | 0.241 | 0.0710 | 5 | 0.183 | 0.0214 | 2 | 0.279 | 0.0143 | 8 |
| `B` | 0.217 | 0.0589 | 4 | 0.355 | 0.0283 | 2 | 0.295 | 0.0162 | 8 |
| `C` | 0.181 | 0.0569 | 4 | 0.220 | 0.0230 | 2 | 0.180 | 0.0121 | 2 |
| `D` | 0.234 | 0.0588 | 3 | 0.181 | 0.0122 | 1 | 0.129 | 0.0074 | 2 |
| `E` | 0.198 | 0.0612 | 4 | 0.192 | 0.0175 | 3 | 0.242 | 0.0086 | 7 |
| `F` | 0.198 | 0.0594 | 6 | 0.139 | 0.0128 | 3 | 0.076 | 0.0037 | 0 |
| `G` | 0.164 | 0.0625 | 4 | 0.195 | 0.0234 | 3 | 0.300 | 0.0138 | 6 |
| `H` | 0.235 | 0.0542 | 4 | 0.129 | 0.0078 | 3 | 0.228 | 0.0059 | 5 |
| `I` | 0.205 | 0.0549 | 3 | 0.172 | 0.0260 | 3 | 0.146 | 0.0090 | 6 |
| `J` | 0.245 | 0.1738 | 8 | 0.241 | 0.0285 | 4 | 0.324 | 0.0092 | 1 |
| `K` | 0.302 | 0.0894 | 6 | 0.149 | 0.0239 | 5 | 0.302 | 0.0170 | 7 |
| `L` | 0.173 | 0.0494 | 2 | 0.156 | 0.0210 | 4 | 0.204 | 0.0057 | 3 |
| `M` | 0.264 | 0.0952 | 7 | 0.229 | 0.0253 | 7 | 0.180 | 0.0110 | 10 |
| `N` | 0.294 | 0.0782 | 6 | 0.230 | 0.0165 | 3 | 0.320 | 0.0088 | 5 |
| `O` | 0.140 | 0.0460 | 2 | 0.201 | 0.0195 | 1 | 0.052 | 0.0096 | 0 |
| `P` | 0.230 | 0.0723 | 5 | 0.164 | 0.0286 | 5 | 0.295 | 0.0167 | 10 |
| `Q` | 0.180 | 0.0544 | 4 | 0.226 | 0.0231 | 2 | 0.098 | 0.0103 | 0 |
| `R` | 0.295 | 0.0735 | 7 | 0.278 | 0.0232 | 4 | 0.299 | 0.0141 | 6 |
| `S` | 0.197 | 0.0854 | 6 | 0.127 | 0.0215 | 1 | 0.156 | 0.0102 | 1 |
| `T` | 0.267 | 0.0812 | 4 | 0.132 | 0.0165 | 2 | 0.180 | 0.0061 | 3 |
| `U` | 0.136 | 0.0310 | 2 | 0.127 | 0.0131 | 2 | 0.228 | 0.0098 | 4 |
| `V` | 0.251 | 0.0669 | 4 | 0.146 | 0.0175 | 3 | 0.189 | 0.0106 | 5 |
| `W` | 0.222 | 0.1060 | 10 | 0.238 | 0.0229 | 9 | 0.374 | 0.0103 | 7 |
| `X` | 0.287 | 0.0749 | 5 | 0.197 | 0.0368 | 6 | 0.223 | 0.0196 | 7 |
| `Y` | 0.220 | 0.0642 | 4 | 0.142 | 0.0256 | 3 | 0.307 | 0.0159 | 4 |
| `Z` | 0.239 | 0.1026 | 6 | 0.261 | 0.0269 | 2 | 0.204 | 0.0144 | 6 |
| `[` | 0.179 | 0.1046 | 6 | 0.209 | 0.0207 | 2 | 0.167 | 0.0058 | 1 |
| `\` | 0.163 | 0.0483 | 2 | 0.132 | 0.0180 | 1 | 0.257 | 0.0138 | 1 |
| `]` | 0.139 | 0.0262 | 2 | 0.214 | 0.0194 | 3 | 0.174 | 0.0066 | 3 |
| `^` | 0.177 | 0.0818 | 3 | 0.323 | 0.0526 | 4 | 0.132 | 0.0204 | 2 |
| `_` | 0.089 | 0.0226 | 0 | 0.128 | 0.0142 | 1 | 0.034 | 0.0014 | 0 |
| `` ` `` | 0.132 | 0.0788 | 1 | 0.040 | 0.0184 | 0 | 0.216 | 0.0383 | 3 |
| `a` | 0.210 | 0.0829 | 5 | 0.182 | 0.0253 | 2 | 0.250 | 0.0129 | 4 |
| `b` | 0.181 | 0.0505 | 5 | 0.349 | 0.0414 | 5 | 0.198 | 0.0095 | 2 |
| `c` | 0.214 | 0.0639 | 3 | 0.124 | 0.0222 | 2 | 0.106 | 0.0122 | 1 |
| `d` | 0.241 | 0.0512 | 3 | 0.182 | 0.0185 | 2 | 0.394 | 0.0144 | 4 |
| `e` | 0.224 | 0.0761 | 5 | 0.166 | 0.0230 | 3 | 0.233 | 0.0117 | 3 |
| `f` | 0.287 | 0.1086 | 3 | 0.167 | 0.0331 | 5 | 0.220 | 0.0146 | 8 |
| `g` | 0.194 | 0.0714 | 6 | 0.166 | 0.0231 | 4 | 0.411 | 0.0167 | 6 |
| `h` | 0.191 | 0.0476 | 4 | 0.424 | 0.0320 | 5 | 0.185 | 0.0079 | 3 |
| `i` | 0.277 | 0.0892 | 6 | 0.147 | 0.0360 | 5 | 0.225 | 0.0205 | 12 |
| `j` | 0.246 | 0.0715 | 3 | 0.153 | 0.0225 | 2 | 0.246 | 0.0119 | 5 |
| `k` | 0.291 | 0.0793 | 7 | 0.211 | 0.0347 | 5 | 0.187 | 0.0168 | 7 |
| `l` | 0.190 | 0.0707 | 5 | 0.075 | 0.0128 | 0 | 0.218 | 0.0091 | 1 |
| `m` | 0.296 | 0.1131 | 9 | 0.177 | 0.0353 | 6 | 0.239 | 0.0173 | 9 |
| `n` | 0.191 | 0.0630 | 5 | 0.424 | 0.0330 | 4 | 0.185 | 0.0093 | 3 |
| `o` | 0.173 | 0.0550 | 3 | 0.069 | 0.0152 | 0 | 0.042 | 0.0084 | 0 |
| `p` | 0.173 | 0.0520 | 4 | 0.344 | 0.0407 | 5 | 0.217 | 0.0090 | 2 |
| `q` | 0.228 | 0.0604 | 4 | 0.098 | 0.0130 | 0 | 0.261 | 0.0116 | 5 |
| `r` | 0.266 | 0.1385 | 3 | 0.207 | 0.0530 | 7 | 0.302 | 0.0181 | 5 |
| `s` | 0.294 | 0.1218 | 7 | 0.228 | 0.0419 | 4 | 0.184 | 0.0127 | 2 |
| `t` | 0.357 | 0.1684 | 7 | 0.096 | 0.0180 | 0 | 0.144 | 0.0118 | 6 |
| `u` | 0.151 | 0.0420 | 3 | 0.176 | 0.0157 | 2 | 0.347 | 0.0119 | 4 |
| `v` | 0.245 | 0.0713 | 4 | 0.156 | 0.0255 | 4 | 0.224 | 0.0135 | 4 |
| `w` | 0.337 | 0.1252 | 8 | 0.289 | 0.0377 | 7 | 0.419 | 0.0148 | 7 |
| `x` | 0.273 | 0.1001 | 5 | 0.193 | 0.0399 | 3 | 0.341 | 0.0226 | 6 |
| `y` | 0.251 | 0.0854 | 5 | 0.314 | 0.0394 | 5 | 0.278 | 0.0194 | 7 |
| `z` | 0.281 | 0.0968 | 6 | 0.175 | 0.0319 | 5 | 0.267 | 0.0143 | 3 |
| `{` | 0.186 | 0.0570 | 2 | 0.269 | 0.0395 | 2 | 0.384 | 0.0180 | 4 |
| `\|` | 0.005 | 0.0006 | 0 | 0.181 | 0.0106 | 2 | 0.215 | 0.0034 | 1 |
| `}` | 0.203 | 0.0725 | 2 | 0.200 | 0.0412 | 4 | 0.230 | 0.0146 | 3 |
| `~` | 0.182 | 0.0842 | 3 | 0.353 | 0.0558 | 2 | 0.138 | 0.0180 | 2 |
