> **Retracted/Superseded (2026-09-07), ledger L046.** The regime-restricted Spearman rests on tree-cost deltas on a generated corpus and is not re-taken. Verdict and rationale: `docs/results/2026-09-07-claims-ledger.md` (PR #1207); the corrected benchmark and re-validation order: `docs/plans/2026-09-07-benchmark-correction.md`.

# Guide return-to-go, round 4: training inside the selected regime, and confluence-aware credit

> **Instrument change (2026-09-02 forward-port).** Two things about how a
> Phase 3 anytime curve is measured changed with the port, so a re-run does
> not reproduce the numbers below even if nothing else changed: the
> application budget now binds **mid-scan** rather than between rule sweeps
> (`app_actual == app_target` exactly, no overshoot), and the reported cost
> is the **DAG** cost the emitted kernel pays rather than the extraction DP's
> tree total (#1117). Full statement:
> [docs/results/2026-09-02-phase3-instrument-changes.md](2026-09-02-phase3-instrument-changes.md).

**Date:** 2026-09-01 (run) · **Registration:** `docs/plans/2026-09-01-guide-return-to-go.md` §2b.3 (selection rule) and §7 (comparison) · **Source rev:** `cf8814a8b4e83ca117bf69be2be44a3f6ddc449e`

**Selection rule, fixed before this round's training run** (§2b.3, from round 3's spread measurement alone): node-count band ∈ {101-250, 251-1000}, label budget B = 100 — the five (budget, band) cells where TRAIN and DEV independently agree that fewer than half the expressions show zero spread among the 11 guided orderings. A record enters training iff its source expression has 101 ≤ node count ≤ 1000 **and** `application_ordinal < 100`.

## Verdict

1. **The regime is trainable; the model is still nearly flat.** Restricting training to the selected regime moved the linear R2G off zero — held-out DEV Spearman(predicted return, realized return) rose from 0.099 (round 3, whole population) to **0.2375** — but it explains **0.08%** of the label variance: DEV MSE 1.132686 against a zero-predictor floor of 1.133602. A linear model on these features is not the thing that reads this credit signal.

2. **On the anytime ladder the regime-trained R2G loses to the frozen strict bit in distribution, ties on `sh`, and beats it on `bezier`.** Per-expression head-to-head at B = 100: DEV band 101-1000 **70 wins / 119 losses / 19 ties**; `sh` **43 / 45 / 7**; `bezier` **60 / 0 / 20** (median cost ratio 0.942). Round 3's "ties the strict bit everywhere" is no longer the result — the two heads now disagree, in opposite directions on the two OOD families.

3. **Leave-one-out systematically understates credit, and the amount is measurable.** Masking every application of the seed's `(rule, canonical match)` rather than just the seed itself moves **77 of 1012** (7.6%) of leave-one-out's Δ = 0 applications to Δ > 0. The strict bit's ρ against that truth nearly doubles (0.3893 → 0.7254), and every proxy rises on the unbanded sample. **The Δ = 0 mass leave-one-out reports is substantially re-derivation noise, not irrelevance** — which is exactly the failure mode that would make a credit signal look absent when it is present. Inside the band the two truths separate the bounds from the models: `strict` still rises (0.5096 → 0.5645) while every *model* proxy falls (R2G 0.1987 → 0.1450, per-rule 0.2881 → 0.2020) — the models were tracking the leave-one-out artifact more than the credit.

4. **The strict bit remains the best predictor of the counterfactual, by a wide margin, under both masks.** Nothing in this round displaces it; the R2G head is now non-zero but still ranks below both the strict bit and the per-rule-rate control on the in-regime sample.


## 1. Training run

Checkpoint: `pixelflow-pipeline/data/guide_checkpoint_r2g_regime_v1.json` (md5 `edcb6559e900cbb0c4f589a6a53cc940`; the `data/` tree is gitignored, as it has been for every round's checkpoint) (objective `return-mse`, target `centered`, B = 100, lr 0.0001, grad-clip 1.0, 30 epochs — the hyperparameters round 3's sweep chose on TRAIN loss alone, unchanged).

| | read | dropped by the regime | kept |
|---|---:|---:|---:|
| TRAIN | 18,565,784 | 17,283,072 | 1,282,712 |
| DEV | 3,544,477 | 3,294,877 | 249,600 |

| statistic | value |
|---|---:|
| TRAIN final-epoch mean loss | 1.302296 |
| DEV MSE | 1.132686 |
| **DEV MSE, zero predictor (`f ≡ 0`)** | **1.133602** |
| variance explained over the floor | 0.081% |
| DEV Spearman(predicted, realized) | 0.2375 |

Skew test: docs/results/2026-09-01-skew-test-r2g-regime.json (PASS, max |trainer+deployed| = 0.0 over 5000 DEV records).

**§1.3 was not enforced by the mint.** The minter attaches a trajectory's return to *every* application it records, including those firing long after the checkpoint the return was read at; with round 3's ladder reaching 3,200 applications that mislabels the great majority of records. `--enforce-label-ordinal` (new, default off so pre-round-3 runs reproduce bit-for-bit) is what enforces the registration's "an application with t > B carries no label for that B". Of the 18,565,784 TRAIN records read, 17,283,072 are dropped by the band and ordinal filters together.


## 2. Anytime ladder — medians and quartiles of cost ratio vs unguided at B

Lower is better. `unguided@B` is 1.000 by definition; its regret column is the median regret of the unguided arm at B against the row's best-known cost. The frozen strict-v1 arm is the round-3 run's rows re-aggregated inside the band (identical unguided/per-rule mechanics; regret is measured against each row's own best-known cost, which includes that row's claim arm, so the unguided regret column differs marginally between the strict and R2G row files).

| set | B | n | per-rule ctl q1/med/q3 | strict-v1 q1/med/q3 | **R2G (regime)** q1/med/q3 | regret% med: ung@B / ctl / strict / R2G |
|---|---:|---:|---|---|---|---|
| DEV classical, band 101-1000 | 100 | 208 | 0.003/0.552/0.691 | 0.003/0.512/0.659 | **0.003/0.521/0.670** | 104.34 / 2.34 / 1.12 / 1.43 |
| DEV classical, band 101-1000 | 200 | 208 | 0.015/0.603/0.709 | 0.006/0.589/0.706 | **0.098/0.600/0.719** | 80.39 / 0.60 / 0.16 / 0.40 |
| sh (OOD, all 95) | 100 | 95 | 0.892/0.903/0.921 | 0.892/0.904/0.918 | **0.890/0.900/0.920** | 20.48 / 7.92 / 7.97 / 8.04 |
| sh (OOD, all 95) | 200 | 95 | 0.879/0.894/0.914 | 0.882/0.896/0.916 | **0.883/0.897/0.918** | 20.33 / 6.70 / 6.80 / 6.70 |
| bezier (OOD, all 80) | 100 | 80 | 0.910/0.910/0.949 | 0.910/0.910/0.949 | **0.843/0.857/0.892** | 29.55 / 17.87 / 17.87 / 11.00 |
| bezier (OOD, all 80) | 200 | 80 | 0.910/0.910/0.980 | 0.857/0.885/0.926 | **0.857/0.857/0.914** | 29.55 / 17.87 / 11.00 / 11.00 |

### Head-to-head, per expression (R2G regime vs frozen strict-v1, same expression, same B)

| set | B | n | R2G better | strict better | tie | median cost ratio R2G/strict |
|---|---:|---:|---:|---:|---:|---:|
| DEV classical, band 101-1000 | 100 | 208 | 70 | 119 | 19 | 1.0015 |
| DEV classical, band 101-1000 | 200 | 208 | 24 | 93 | 91 | 1.0000 |
| sh (OOD, all 95) | 100 | 95 | 43 | 45 | 7 | 1.0000 |
| sh (OOD, all 95) | 200 | 95 | 18 | 38 | 39 | 1.0000 |
| bezier (OOD, all 80) | 100 | 80 | 60 | 0 | 20 | 0.9417 |
| bezier (OOD, all 80) | 200 | 80 | 19 | 0 | 61 | 1.0000 |

## 3. Credit check — leave-one-out vs the confluence-aware multi-mask

The second mask mode (`MaskScope::AllMatchingCandidate`) skips the seed application **and every later application sharing its `(rule_idx, canonical matched-class content)`** — the same `CandidateKey` the guided loop dedups on. The key is read off the live graph at the seed ordinal, not supplied by the caller, so it is exactly the key the original trajectory's application matched. Consequence: an alternative re-derivation by that route cannot silently restore the node leave-one-out removed.


### Unbanded (round-3 sample: 30 `sh` + 30 DEV, 20 applications each — directly comparable to round 3's table)

n = 1095 sampled state-changing applications, B = 100.

| mask | Δ = 0 | Δ > 0 | Δ < 0 |
|---|---:|---:|---:|
| leave-one-out | 1012 (92.4%) | 61 (5.6%) | 22 (2.0%) |
| multi-mask | 927 (84.7%) | 138 (12.6%) | 30 (2.7%) |

**Confluence blindness of leave-one-out: 77 of 1012 (7.6%) Δ = 0 applications become Δ > 0 under the multi-mask** (8 become Δ < 0). Multi-masks that skipped more than the seed: 435/1095 (mean 1.57 skips, max 6).

| proxy | ρ vs leave-one-out Δ | ρ vs multi-mask Δ |
|---|---:|---:|
| `r2g_linear` | 0.1027 | 0.2488 |
| `strict_v1_linear` | 0.1703 | 0.3445 |
| `per_rule_rate` | 0.1818 | 0.2671 |
| `loose` | n/a (no variance) | n/a (no variance) |
| `tight` | n/a (no variance) | n/a (no variance) |
| `strict` | 0.3893 | 0.7254 |
| `strict_by_output_class` | 0.0046 | 0.0562 |

### Inside the training regime (band 101-1000: 40 `sh` + 60 DEV, 30 applications each)

n = 3000 sampled state-changing applications, B = 100.

| mask | Δ = 0 | Δ > 0 | Δ < 0 |
|---|---:|---:|---:|
| leave-one-out | 2525 (84.2%) | 340 (11.3%) | 135 (4.5%) |
| multi-mask | 2415 (80.5%) | 431 (14.4%) | 154 (5.1%) |

**Confluence blindness of leave-one-out: 90 of 2525 (3.6%) Δ = 0 applications become Δ > 0 under the multi-mask** (20 become Δ < 0). Multi-masks that skipped more than the seed: 425/3000 (mean 1.16 skips, max 6).

| proxy | ρ vs leave-one-out Δ | ρ vs multi-mask Δ |
|---|---:|---:|
| `r2g_linear` | 0.1987 | 0.1450 |
| `strict_v1_linear` | 0.2463 | 0.2094 |
| `per_rule_rate` | 0.2881 | 0.2020 |
| `loose` | n/a (no variance) | n/a (no variance) |
| `tight` | n/a (no variance) | n/a (no variance) |
| `strict` | 0.5096 | 0.5645 |
| `strict_by_output_class` | 0.0109 | 0.0310 |

`loose`/`tight` have no variance on this sample (every sampled application is load-bearing under both) — reported as n/a, never as 0.

**The two samples disagree about the models, and that is itself the finding.** Unbanded, every proxy ranks higher against the multi-mask truth. In-band, only the hindsight *bounds* (`strict`, `strict_by_output_class`) rise; all three *model* proxies fall. A learned score that tracks leave-one-out better than it tracks the confluence-aware Δ is fitting the instrument's zero-inflation, not the credit — so in-band, where the budget genuinely binds, the models look worse the more accurately the truth is measured.


## 4. What this does and does not settle

- The selection rule was registered from the spread measurement before any weight was trained (§2b.3), and it is a *narrower* population than "TRAIN classical": no expression in this corpus exceeds 1,000 arena nodes, so nothing here speaks to production's classical tail (p90 ≈ 23,799 applications at stop, ≥ 67.6% of real kernels binding the 5,000-class cap). Extending coverage past 1,000 nodes remains a corpus-design question, not a training-target one.

- No production behavior changed. `CostModel::latency_prior()` remains the extraction cost; FINAL is untouched; the frozen strict-v1 checkpoint is byte-identical to round 1's.

- The confluence result is a measurement of the **validation instrument**, not of any model: it says the leave-one-out Δ this program has been scoring proxies against is a biased-toward-zero estimate of counterfactual credit, and quantifies the bias. Every ρ in round 3's table was computed against that biased truth.


## Artifacts

- `docs/results/2026-09-01-guide-r2g-spread-first.{md,json,csv}` (this report)
- `docs/results/2026-09-01-train-guide-r2g-regime-report.{json,md}`, `docs/results/2026-09-01-skew-test-r2g-regime.json`
- `docs/results/2026-09-01-r2g-ladder-regime-{dev,dev-strict,sh,bezier}.{jsonl,json}` + `-report.md`
- `docs/results/2026-09-01-counterfactual-credit-{multimask,regime}.{jsonl,json,md}`
- checkpoint `pixelflow-pipeline/data/guide_checkpoint_r2g_regime_v1.json`

