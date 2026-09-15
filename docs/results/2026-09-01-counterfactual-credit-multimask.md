> **Retracted/Superseded (2026-09-07), ledger L046.** The multi-mask deltas are tree-cost differences (pre-#1192) on a generated corpus. Verdict and rationale: `docs/results/2026-09-07-claims-ledger.md` (PR #1207); the corrected benchmark and re-validation order: `docs/plans/2026-09-07-benchmark-correction.md`.

# Counterfactual credit: hindsight bounds vs measured Δ (leave-one-out and confluence-aware)

> **Instrument change (2026-09-02 forward-port).** Two things about how a
> Phase 3 anytime curve is measured changed with the port, so a re-run does
> not reproduce the numbers below even if nothing else changed: the
> application budget now binds **mid-scan** rather than between rule sweeps
> (`app_actual == app_target` exactly, no overshoot), and the reported cost
> is the **DAG** cost the emitted kernel pays rather than the extraction DP's
> tree total (#1117). Full statement:
> [docs/results/2026-09-02-phase3-instrument-changes.md](2026-09-02-phase3-instrument-changes.md).

Sample: 30 `sh` + 30 DEV classical expressions, 1095 applications sampled (1095 at ordinal < B=100, seed 0x5eedc0dec0ffee01). wall_clock_ceiling_hit = false. git cf8814a8b4e83ca117bf69be2be44a3f6ddc449e.

Δ distribution: zero 1012/1095 (92.4%), positive 61/1095 (5.6%), negative 22/1095 (2.0%).

Proxies loaded: r2g = "pixelflow-pipeline/data/guide_checkpoint_r2g_regime_v1.json", strict-v1 = "pixelflow-pipeline/data/guide_checkpoint_strict_v1.json", per-rule = "docs/results/2026-09-01-train-guide-report.json". Bootstrap: 1000 paired resamples (seeded).

| proxy | Spearman (pooled) [95% CI] | Δρ vs r2g [95% CI] | Spearman (sh) | Spearman (dev) | Pearson (pooled) | n (excluded) |
|---|---:|---:|---:|---:|---:|---:|
| r2g_linear | 0.102655 [0.042750, 0.157630] | [0.000000, 0.000000] | -0.033096 | 0.196297 | 0.042164 | 1095 (0) |
| strict_v1_linear | 0.170344 [0.102819, 0.234209] | [0.025622, 0.112314] | 0.104863 | 0.190191 | -0.003872 | 1095 (0) |
| per_rule_rate | 0.181789 [0.119447, 0.241787] | [0.040374, 0.120465] | 0.163984 | 0.252674 | 0.040272 | 1095 (0) |
| loose | null [null, null] | [null, null] | null | null | null | 1095 (0) |
| tight | null [null, null] | [null, null] | null | null | null | 1095 (0) |
| strict | 0.389273 [0.304146, 0.476948] | [0.206592, 0.364162] | 0.414912 | 0.352117 | 0.036477 | 1095 (0) |
| strict_by_output_class | 0.004593 [-0.055820, 0.058629] | [-0.175899, -0.015682] | 0.014839 | 0.015221 | -0.020099 | 1095 (0) |

## Confluence-aware credit (multi-mask)

Second mask mode: the seed application AND every later application sharing its `(rule_idx, canonical matched-class content)` are skipped, so an alternative re-derivation cannot silently restore the node leave-one-out removed.

Multi-mask Δ distribution: zero 927/1095 (84.7%), positive 138/1095 (12.6%), negative 30/1095 (2.7%).

**Confluence blindness of leave-one-out: 77 of the 1012 applications leave-one-out scored Δ = 0 become Δ > 0 under the multi-mask (7.6% of them)**; 8 become Δ < 0. Multi-masks that skipped more than the seed: 435/1095 (mean skips 1.57, max 6).

| proxy | Spearman vs multi-mask Δ (pooled) [95% CI] | Δρ vs r2g [95% CI] | Spearman (sh) | Spearman (dev) | Pearson (pooled) | n (excluded) |
|---|---:|---:|---:|---:|---:|---:|
| r2g_linear | 0.248762 [0.186622, 0.303776] | [0.000000, 0.000000] | 0.008559 | 0.426935 | 0.083376 | 1095 (0) |
| strict_v1_linear | 0.344455 [0.279667, 0.406200] | [0.055038, 0.137982] | 0.138819 | 0.472752 | 0.073438 | 1095 (0) |
| per_rule_rate | 0.267140 [0.196777, 0.334938] | [-0.030400, 0.069959] | 0.178487 | 0.432860 | 0.155127 | 1095 (0) |
| loose | null [null, null] | [null, null] | null | null | null | 1095 (0) |
| tight | null [null, null] | [null, null] | null | null | null | 1095 (0) |
| strict | 0.725376 [0.660522, 0.780954] | [0.407354, 0.545459] | 0.638298 | 0.742757 | 0.157245 | 1095 (0) |
| strict_by_output_class | 0.056235 [0.000291, 0.105903] | [-0.267600, -0.109650] | 0.054121 | 0.094154 | 0.000713 | 1095 (0) |

## Per-rule Δ over the sampled applications

| idx | rule | n (sh) | mean Δ | Δ>0 | Δ<0 | mean f_r2g | mean adv_r2g |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | canonicalize | 14 (11) | -0.008166 | 2 | 1 | 0.012605 | 0.016089 |
| 1 | involution | 92 (0) | 0.000376 | 10 | 0 | 0.106070 | -0.033670 |
| 4 | canonicalize | 2 (1) | 0.000000 | 0 | 0 | -0.027017 | 0.001602 |
| 8 | constant-fold | 46 (13) | 0.000874 | 15 | 0 | 0.037346 | 0.013991 |
| 9 | commutative | 198 (70) | -0.030775 | 0 | 2 | 0.071790 | -0.035476 |
| 10 | commutative | 212 (166) | -0.000160 | 0 | 6 | 0.041076 | -0.015425 |
| 11 | commutative | 9 (0) | 0.000000 | 0 | 0 | 0.096942 | -0.039672 |
| 12 | commutative | 9 (0) | 0.000000 | 0 | 0 | 0.092052 | -0.050337 |
| 18 | distribute | 15 (12) | -0.000327 | 1 | 1 | 0.090891 | -0.057782 |
| 19 | factor | 1 (0) | 0.000000 | 0 | 0 | 0.058621 | -0.040593 |
| 20 | doubling | 1 (0) | 0.000000 | 0 | 0 | -0.060651 | 0.079883 |
| 21 | halving | 9 (9) | 0.000000 | 0 | 0 | 0.136904 | -0.124940 |
| 22 | associative | 28 (22) | -0.435253 | 0 | 3 | 0.033552 | -0.004770 |
| 23 | associative | 149 (146) | -0.000205 | 0 | 9 | -0.003859 | 0.026394 |
| 25 | associative | 1 (0) | 0.000000 | 0 | 0 | 0.118850 | -0.060615 |
| 26 | reverse-associative | 31 (25) | 0.000000 | 0 | 0 | 0.060086 | -0.030012 |
| 27 | reverse-associative | 110 (103) | 0.000052 | 3 | 0 | 0.004412 | 0.020459 |
| 29 | reverse-associative | 1 (0) | 0.000000 | 0 | 0 | 0.080319 | -0.021042 |
| 30 | odd-negation | 1 (0) | 0.000000 | 0 | 0 | 0.017239 | 0.040195 |
| 31 | odd-negation | 1 (0) | 0.000000 | 0 | 0 | 0.032345 | 0.000446 |
| 34 | even-negation | 9 (0) | 0.000000 | 0 | 0 | 0.042740 | 0.056834 |
| 35 | even-negation | 67 (0) | 0.000589 | 9 | 0 | 0.000167 | 0.056800 |
| 36 | sin-angle-addition | 1 (0) | 0.000000 | 0 | 0 | 0.025806 | 0.031469 |
| 37 | cos-angle-addition | 3 (1) | 0.000000 | 0 | 0 | 0.047678 | 0.031391 |
| 39 | half-angle-product | 5 (5) | 0.000000 | 0 | 0 | -0.050516 | 0.072564 |
| 51 | power-sqrt | 15 (0) | 0.065439 | 8 | 0 | -0.013482 | 0.057990 |
| 52 | power-recip | 21 (0) | 0.076965 | 10 | 0 | 0.001486 | 0.062807 |
| 53 | power-rsqrt | 3 (0) | 0.095376 | 2 | 0 | -0.034772 | 0.061191 |
| 58 | diff-of-squares | 1 (1) | 0.000000 | 0 | 0 | -0.011023 | 0.032116 |
| 59 | fma-fusion | 38 (15) | 0.000000 | 0 | 0 | 0.005037 | 0.052930 |
| 60 | recip-sqrt | 2 (0) | 0.006046 | 1 | 0 | 0.012004 | 0.050388 |

11 expressions had fewer than 20 state-changing applications and contributed all available instead:

- dev_b01_f02_00035 (7 state-changing applications < 20 target)
- dev_b09_f00_00237 (6 state-changing applications < 20 target)
- dev_b16_f00_00347 (15 state-changing applications < 20 target)
- dev_b01_f01_00019 (5 state-changing applications < 20 target)
- dev_b01_f05_00077 (8 state-changing applications < 20 target)
- dev_b06_f07_00219 (16 state-changing applications < 20 target)
- dev_b33_f00_00683 (18 state-changing applications < 20 target)
- dev_b09_f06_00315 (10 state-changing applications < 20 target)
- dev_b22_f01_00469 (11 state-changing applications < 20 target)
- dev_b09_f06_00309 (15 state-changing applications < 20 target)
- dev_b01_f06_00090 (4 state-changing applications < 20 target)
