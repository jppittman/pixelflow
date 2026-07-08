# Rule report — first hindsight-labeled episodes (2026-07-08)

Reproduce: `cargo run --release -p pixelflow-search --example rule_report`
(5 representative kernels × `all_rules()` (61 rules) × full saturation →
latency-prior extraction → provenance labels.)

## Aggregate

| rule | fired | load-bearing | ratio |
|---|---|---|---|
| commutative | 26,691 | 23,186 | 0.869 |
| fma-fusion | 21,905 | 15,614 | 0.713 |
| associative | 17,854 | 12,540 | 0.702 |
| reverse-associative | 18,860 | 11,465 | 0.608 |
| doubling | 185 | 98 | 0.530 |
| recip-sqrt | 2 | 1 | 0.500 |
| distribute | 620 | 228 | 0.368 |
| factor | 1,437 | 376 | 0.262 |
| constant-fold | 159 | 24 | 0.151 |
| halving | 303 | 37 | 0.122 |
| canonicalize | 41 | 4 | 0.098 |

## Findings

1. **Saturation blowup is real at 61 rules**: `circle_sdf` (trivial SDF) →
   47,630 applications / 2,579 classes; `(x+y)²+2(x+y)` → 39,987 applications.
   Empirical support for the guided-expansion thesis
   (docs/plans/2026-07-07-guided-saturation-redesign.md) before any ML.
2. **Clear stratification**: assoc/comm/FMA rules dominate load-bearing;
   halving/canonicalize/constant-fold rarely matter; distribute/factor are
   situational.
3. **Caveat — ratios are inflated.** `derivation_ancestors` over-approximates
   union causality (documented in provenance.rs), so absolute ratios read high
   (~75% of ALL applications "load-bearing" on circle_sdf is not credible).
   Treat the ordering as signal, not the magnitudes. Tightening union-causality
   precision is a prerequisite for training the Guide on per-application labels;
   per-rule ordering is already usable.
