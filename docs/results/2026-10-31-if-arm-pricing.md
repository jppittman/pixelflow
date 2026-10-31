# `If` arm pricing in the DAG price (JP-112)

**Date:** 2026-10-31. **Status:** Measurement record for the choice JP-112 asked for.
**Host:** KVM guest, AVX-512, shared and noisy (emit time alone varies ±20% run to run).

The DAG price and the shared DP used to pay both arms of an `If` in full, while the tree price
averages them. Three ways to pay an arm half were weighed (JP-112):

| option | glyph bake (95 glyphs, tiles 16 and 32) | SDF chain fixture (`saturated_sdf_egraph`, 256 nodes) |
|---|---|---|
| 1. seal each arm as a half-weight scope | emitted bytes and hashes identical to before | DAG price of the tree arm's term **9.86e6**, from under 1e6: an arm that reuses the mask's operand is charged again per nested `If` |
| 2/3. exclusive part only (arm minus mask minus other arm) | emitted bytes and hashes identical to before | under 1e6, as before |

**Chosen: the exclusive-part rule**, applied identically in `Selection::dag_price` and
`shared_dag_dp_pass` so the claim is the price. On the glyph corpus no option changes a single
emitted byte, so the choice is made on the fixture where they differ. Compile time on the glyph
walk, best of five: optimize 184 to 205 ms against 176 to 193 ms before (about +5%, inside the
host's noise; emit, which the change cannot touch, moved as much).

Limits: exclusivity is judged against the `If`'s own operands, so a node an outer parent also reads
is paid in full there and at half inside the `If` (an overcharge, never an undercharge).
The `FONT_PATH` the example reads is an LFS pointer in this checkout; the run used the committed
`DejaVuSansMono-Fallback.ttf` through an uncommitted one-line edit. Reproduce with
`cargo run --release -p pixelflow-pipeline --example glyph_phase_split` after that edit.
