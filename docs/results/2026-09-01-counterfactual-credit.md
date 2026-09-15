> **Retracted/Superseded (2026-09-07), ledger L045.** The leave-one-out deltas are tree-cost differences (pre-#1192) on a generated corpus. Verdict and rationale: `docs/results/2026-09-07-claims-ledger.md` (PR #1207); the corrected benchmark and re-validation order: `docs/plans/2026-09-07-benchmark-correction.md`.

# Counterfactual credit: hindsight bounds vs measured leave-one-out Δ

> **Instrument change (2026-09-02 forward-port).** Two things about how a
> Phase 3 anytime curve is measured changed with the port, so a re-run does
> not reproduce the numbers below even if nothing else changed: the
> application budget now binds **mid-scan** rather than between rule sweeps
> (`app_actual == app_target` exactly, no overshoot), and the reported cost
> is the **DAG** cost the emitted kernel pays rather than the extraction DP's
> tree total (#1117). Full statement:
> [docs/results/2026-09-02-phase3-instrument-changes.md](2026-09-02-phase3-instrument-changes.md).

Sample: 30 `sh` + 30 DEV classical expressions, 1095 applications sampled (1095 at ordinal < B=100, seed 0x5eedc0dec0ffee01). wall_clock_ceiling_hit = false. git 9bc88578ad17a85a400245e948d6868fd3e9ce9b.

Δ distribution: zero 1012/1095 (92.4%), positive 61/1095 (5.6%), negative 22/1095 (2.0%).

Proxies loaded: r2g = "pixelflow-pipeline/data/guide_checkpoint_r2g_v1.json", strict-v1 = "pixelflow-pipeline/data/guide_checkpoint_strict_v1.json", per-rule = "docs/results/2026-09-01-train-guide-report.json". Bootstrap: 1000 paired resamples (seeded).

| proxy | Spearman (pooled) [95% CI] | Δρ vs r2g [95% CI] | Spearman (sh) | Spearman (dev) | Pearson (pooled) | n (excluded) |
|---|---:|---:|---:|---:|---:|---:|
| r2g_linear | -0.003631 [-0.062202, 0.050633] | [0.000000, 0.000000] | -0.082775 | 0.071398 | 0.091839 | 1095 (0) |
| strict_v1_linear | 0.170344 [0.102819, 0.234209] | [0.096410, 0.255111] | 0.104863 | 0.190191 | -0.003872 | 1095 (0) |
| per_rule_rate | 0.181789 [0.119447, 0.241787] | [0.109242, 0.262268] | 0.163984 | 0.252674 | 0.040272 | 1095 (0) |
| loose | null [null, null] | [null, null] | null | null | null | 1095 (0) |
| tight | null [null, null] | [null, null] | null | null | null | 1095 (0) |
| strict | 0.389273 [0.304146, 0.476948] | [0.288309, 0.490928] | 0.414912 | 0.352117 | 0.036477 | 1095 (0) |
| strict_by_output_class | 0.004593 [-0.055820, 0.058629] | [-0.066841, 0.079081] | 0.014839 | 0.015221 | -0.020099 | 1095 (0) |

## Per-rule Δ over the sampled applications

| idx | rule | n (sh) | mean Δ | Δ>0 | Δ<0 | mean f_r2g | mean adv_r2g |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | canonicalize | 14 (11) | -0.008166 | 2 | 1 | -0.029211 | 0.009869 |
| 1 | involution | 92 (0) | 0.000376 | 10 | 0 | -0.000267 | -0.006272 |
| 4 | canonicalize | 2 (1) | 0.000000 | 0 | 0 | -0.081638 | 0.049743 |
| 8 | constant-fold | 46 (13) | 0.000874 | 15 | 0 | -0.010942 | -0.000539 |
| 9 | commutative | 198 (70) | -0.030775 | 0 | 2 | -0.000984 | -0.012026 |
| 10 | commutative | 212 (166) | -0.000160 | 0 | 6 | -0.014910 | -0.003509 |
| 11 | commutative | 9 (0) | 0.000000 | 0 | 0 | -0.001885 | -0.007221 |
| 12 | commutative | 9 (0) | 0.000000 | 0 | 0 | -0.027986 | 0.014844 |
| 18 | distribute | 15 (12) | -0.000327 | 1 | 1 | -0.017548 | -0.001552 |
| 19 | factor | 1 (0) | 0.000000 | 0 | 0 | -0.073835 | 0.039686 |
| 20 | doubling | 1 (0) | 0.000000 | 0 | 0 | -0.036746 | 0.002221 |
| 21 | halving | 9 (9) | 0.000000 | 0 | 0 | -0.042897 | 0.025434 |
| 22 | associative | 28 (22) | -0.435253 | 0 | 3 | -0.010122 | -0.007231 |
| 23 | associative | 149 (146) | -0.000205 | 0 | 9 | -0.028544 | 0.007540 |
| 25 | associative | 1 (0) | 0.000000 | 0 | 0 | -0.075925 | 0.062191 |
| 26 | reverse-associative | 31 (25) | 0.000000 | 0 | 0 | -0.006279 | -0.008836 |
| 27 | reverse-associative | 110 (103) | 0.000052 | 3 | 0 | -0.032534 | 0.006957 |
| 29 | reverse-associative | 1 (0) | 0.000000 | 0 | 0 | -0.079842 | 0.066214 |
| 30 | odd-negation | 1 (0) | 0.000000 | 0 | 0 | -0.013001 | 0.009264 |
| 31 | odd-negation | 1 (0) | 0.000000 | 0 | 0 | 0.001337 | -0.013919 |
| 34 | even-negation | 9 (0) | 0.000000 | 0 | 0 | -0.006867 | 0.005179 |
| 35 | even-negation | 67 (0) | 0.000589 | 9 | 0 | -0.015296 | 0.004573 |
| 36 | sin-angle-addition | 1 (0) | 0.000000 | 0 | 0 | -0.023238 | 0.019691 |
| 37 | cos-angle-addition | 3 (1) | 0.000000 | 0 | 0 | 0.001167 | -0.011052 |
| 39 | half-angle-product | 5 (5) | 0.000000 | 0 | 0 | -0.084433 | 0.053568 |
| 51 | power-sqrt | 15 (0) | 0.065439 | 8 | 0 | -0.011557 | 0.002144 |
| 52 | power-recip | 21 (0) | 0.076965 | 10 | 0 | -0.019262 | 0.011668 |
| 53 | power-rsqrt | 3 (0) | 0.095376 | 2 | 0 | -0.015993 | 0.000581 |
| 58 | diff-of-squares | 1 (1) | 0.000000 | 0 | 0 | -0.073051 | 0.040196 |
| 59 | fma-fusion | 38 (15) | 0.000000 | 0 | 0 | -0.023632 | 0.006362 |
| 60 | recip-sqrt | 2 (0) | 0.006046 | 1 | 0 | -0.017107 | 0.009142 |

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
