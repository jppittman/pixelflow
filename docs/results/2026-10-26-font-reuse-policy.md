# Font executable reuse policy: bucket floor, measured (JP-24)

**Date:** 2026-10-26. Follows [the A0 baseline](2026-10-20-production-glyph-baseline.md),
whose numbers this reads against (7 programs, 6 saturations at the class cap, cold warm ~100 ms).
**Host:** 4-vCPU KVM guest, tier as `isa::detect()` picks it; font DejaVuSansMono-Fallback
(Noto is an LFS pointer in this checkout). Probe: `examples/glyph_reuse_probe.rs`, one cold
process per row, three runs, printable ASCII, tile 16 (density 1) and tile 32 (density 2).
The trip-count policy was swapped through a temporary, uncommitted env override of
`bucketed_trip_count`.

## Results (ms, three runs each, density 1)

| policy | programs | cold warm | zoom (new tile extent) | re-warm (all cached) |
|---|---|---|---|---|
| exact count (specialize per glyph) | ~30+ | 230-243 | 117-142 | 48-51 |
| multiples of 8 | few | 81-95 | 59-66 | 48-61 |
| powers of two (before) | 7 | 87-100 | 62-68 | 46-51 |
| **powers of two, floor 64 (after)** | **1** | **61-67** | **55-62** | 51-57 |

Density 2 agrees (pow2 91-120 cold, floor-64 50-71; exact 227-263). Re-warm is flat across
policies, so the padding rows cost nothing measurable at bake: a padding row's term is a
literal `0.0` and the bake is under 1 ms per atlas.

## Choice

- **Specialize per glyph: refused.** It is 2.5x the cold cost and pays a program per glyph.
- **Floor the bucket at 64** (`MIN_TRIP_BUCKET`): every ASCII glyph of the monospace faces
  measured has at most 64 pieces, so a warm is one saturation and one emit instead of seven
  and cold falls ~35%. Glyphs above 64 keep their power-of-two bucket (a 262-piece glyph is
  512), so nothing is padded by more than 2x past the floor.
- **Uniform count is not measurable yet.** A per-call fold count (decided 2026-10-22, A6/B2 in
  [a bound is a value](../language/2026-10-20-a-bound-is-a-value.md)) is unbuilt, so it cannot be
  compared. This floor is the interim and is deleted, with the padding and buckets, when that
  lands and wins. Decision 4 of that note refused a fixed-capacity interim before it had
  numbers; these are the numbers, and the cost of the interim is one constant.

## Not measured

Noto itself (LFS pointer), NEON, and proportional faces with many glyphs above 64 pieces.
