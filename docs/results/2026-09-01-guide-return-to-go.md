> **Retracted/Superseded (2026-09-07), ledger L045.** The Claim B delta is a tree-cost difference on a generated corpus; the negative result ("the target was not the bottleneck") is not re-taken. Verdict and rationale: `docs/results/2026-09-07-claims-ledger.md` (PR #1207); the corrected benchmark and re-validation order: `docs/plans/2026-09-07-benchmark-correction.md`.

# Guide return-to-go (linear first): the learned credit is not real — R2G ties the strict bit on every set and tier, and its advantage has zero rank correlation with the leave-one-out counterfactual (2026-09-01)

> **Instrument change (2026-09-02 forward-port).** Two things about how a
> Phase 3 anytime curve is measured changed with the port, so a re-run does
> not reproduce the numbers below even if nothing else changed: the
> application budget now binds **mid-scan** rather than between rule sweeps
> (`app_actual == app_target` exactly, no overshoot), and the reported cost
> is the **DAG** cost the emitted kernel pays rather than the extraction DP's
> tree total (#1117). Full statement:
> [docs/results/2026-09-02-phase3-instrument-changes.md](2026-09-02-phase3-instrument-changes.md).

**Registration:** `docs/plans/2026-09-01-guide-return-to-go.md` (§7 unrevised; §7.1 appended there against its gates).
**Authority (unchanged):** `docs/plans/2026-09-01-phase3-registration.md` (Round 1), `docs/plans/2026-09-01-phase3-round1b-domain-shift-registration.md` (Round 1b).
**Branch:** `claude/phase3-r2g` on top of `claude/phase3-domain-shift` (PR #1091). Companion files: `2026-09-01-guide-return-to-go.{json,csv}` (the four-arm ladder, per set × tier × arm), `2026-09-01-counterfactual-credit.{md,json,jsonl}` (the credit table, per sampled application), `2026-09-01-train-guide-r2g-report.{md,json}` + `-ablation-{raw,b200,rank}.json` (training), `2026-09-01-skew-test-linear-return-guide.json`, `2026-09-01-r2g-ladder-{strict,r2g}-{dev,sh,bezier}.{jsonl,json,-report.md}` (the two harness halves), `2026-09-01-r2g-dataset.{md,json}` (the mint), `journal.jsonl` (six harness records + one summary record).

Reproduce (every number is deterministic under `CostModel::latency_prior()`; no wall-clock enters any metric):
```
cargo build --release -p pixelflow-pipeline --features training
T=./target/release/train_guide_r2g; D=pixelflow-pipeline/data; R=docs/results
$T --train $D/r2g_train.jsonl --dev $D/r2g_dev.jsonl --lr 0.0001 --grad-clip 1 --target centered --label-b 100 --loss mse \
   --out-checkpoint $D/guide_checkpoint_r2g_v1.json --report-json $R/2026-09-01-train-guide-r2g-report.json --report-md $R/2026-09-01-train-guide-r2g-report.md
./target/release/skew_test_linear_guide --model return --dev $D/r2g_dev.jsonl --checkpoint $D/guide_checkpoint_r2g_v1.json --n 5000 --report-json $R/2026-09-01-skew-test-linear-return-guide.json
./target/release/counterfactual_credit --r2g-checkpoint $D/guide_checkpoint_r2g_v1.json --strict-checkpoint $D/guide_checkpoint_strict_v1.json --train-guide-report $R/2026-09-01-train-guide-report.json
B=./target/release/phase3_at_budget_eval   # strict-bit half: default --checkpoint; R2G half: --r2g-checkpoint $D/guide_checkpoint_r2g_v1.json
$B --classical-samples 0 --other-samples 0 --out-jsonl $R/2026-09-01-r2g-ladder-strict-dev.jsonl ...      # and --corpus $D/corpus_dev_ood.bin --name-prefix dev_sh_ / dev_bezier_
python3 scripts/r2g_ladder_join.py --strict-prefix $R/2026-09-01-r2g-ladder-strict --r2g-prefix $R/2026-09-01-r2g-ladder-r2g --out-json $R/2026-09-01-guide-return-to-go.json --out-csv $R/2026-09-01-guide-return-to-go.csv --out-md -
```

Inputs: `corpus_train.bin` MD5 `0ed6cf16abcbc006cd7a3ee2365b15b4`, `corpus_dev.bin` `3026133ebba066eeca10f658da554400`, `corpus_dev_ood.bin` `0c7cbe710c50175afb3cd91f60960b64`; frozen Round-1 `guide_checkpoint_strict_v1.json` `dcc79b59cfe00bc62df031924382e279` (used as-is, never retrained); R2G mint `r2g_train.jsonl` FNV-1a64 `f89c385604adc854` (971,972 records, 677 TRAIN classical expressions × 12 policies), `r2g_dev.jsonl` `2b9d7d6472016bb7` (932,261 records; held out, never trained on). Output: `guide_checkpoint_r2g_v1.json` MD5 `73b7db7bf75d13c94824f7826830a021`, weights FNV `298cb839455cfe8f`. FINAL untouched. Family fence checked at every load (0 collisions). Host load ~5–14 on 12 cores during the runs (context only).

## 0. Standing of this run against the pre-registered dataset gate

The mint report (`2026-09-01-r2g-dataset.md`) found **67.2 % of TRAIN classical expressions with zero return spread at B = 100** (68.0 % on DEV; 0 % on `sh`, 25 % on `bezier`). Registration §2 pre-committed that above 50 % *"the finding is that ordering does not move the outcome at the primary tier on this corpus; record it, do not train."* That finding stands and is the primary result of the mint. Training, the counterfactual and the ladder below were run anyway on the orchestrator's explicit direction (2026-09-01) as the exploratory completion of the round, so that the credit question has a measured answer rather than an inferred one. Nothing below revises §7; every gate is scored as written, and the gate's own reading ("no training") is carried into §5.

## 1. The answer

**Claim B (credit) — FAILS.** On the counterfactual sample S (30 `sh` + 30 DEV classical expressions, 1,095 state-changing applications of the unguided trajectory at B = 100, one masked replay each), the linear R2G model's advantage has Spearman ρ = **−0.004** with the leave-one-out Δ (bootstrap 95 % CI [−0.062, +0.051]). Every hand-drawn bound that has variance beats it, and the paired-bootstrap CI of every difference excludes zero: strict **0.389** (Δρ vs R2G [+0.29, +0.49]), per-rule rate 0.182 ([+0.11, +0.26]), strict-v1 linear 0.170 ([+0.10, +0.26]). The strict-by-output-class variant proposed after Round 1b is 0.005 — no better than the R2G model. `loose` and `tight` are 1,095/1,095 true on this sample (zero variance; undefined, not zero).

**Claim A (ordering) — FAILS on `sh`, holds on DEV, and the pre-committed "ties everywhere" row is the one that fires.** `m_r2g^sh(100) = 0.9084` against the registered `< 0.8439`; `m_r2g^sh(200) = 0.9017` against `< 0.8259`. On DEV `0.5493 ≤ 0.5966` and `0.7137 ≤ 0.7659` hold. `|m_r2g − m_linear| ≤ M_B` on all six (set, tier) cells: +0.013 / +0.018 (DEV), +0.005 / +0.006 (`sh`), −0.053 / −0.029 (`bezier`). Registration §7's reading for that row: **the target was not the bottleneck.**

**The strict bit is the best credit proxy this program has, and it is not the model's.** ρ_strict = 0.39 against the truth while the model trained on the "better" target scores 0. Round 1's strict-bit Guide (ρ = 0.17 as a deployed scorer) and the per-rule table (0.18) both carry a real but small share of the counterfactual signal; the R2G regression carries none, and it also fits none of its own target: TRAIN final MSE 0.7047 against a zero-predictor floor of 0.7054 (§3).

**`pythagorean` still does not fire, under any Guide.** 0 firings at any checkpoint under R2G and under the strict bit on `sh` (unguided: 231 firings on 19 expressions, all past B = 200, 0 strict-positive). It is absent from S because the unguided sweep never reaches rule index 40 within B = 100, so its Δ at the primary tier is *unmeasurable on the unguided trajectory*, not measured-zero. R2G fires more trig at B = 100 than the strict bit (`half-angle-product` 545 vs 360, `doubling` 396 vs 295, `sin-angle-addition` 306 vs 219, `reverse-angle-addition` 213 vs 91) with an identical strict-positive count (134 = 134): the additional firings are not load-bearing, which is why the median does not move.

## 2. The credit table (Claim B) — which credit definition is real

Sample S: 1,095 applications (600 `sh`, 495 DEV; 11 DEV expressions had < 20 state-changing applications and contributed all of them). Δ_a = ln cost(τ∖a, B) − ln cost(τ, B) at B = 100 on the unguided trajectory: **92.4 % zero, 5.6 % positive (removing `a` hurt), 2.0 % negative (removing `a` helped)**. Proxy advantages per §4.3 (same-sweep alternatives `A_t` = the other applications recorded in `ApplicationRecord::step`; no row had an empty `A_t`). Features observed against the e-graph at B by the one constructor `CandidateFeatures::observe`. Bootstrap: 1,000 paired resamples, seeded.

| proxy | Spearman ρ vs Δ, pooled [95 % CI] | Δρ vs R2G [95 % CI] | ρ on `sh` (n = 600) | ρ on DEV (n = 495) | Pearson (pooled) |
|---|---:|---:|---:|---:|---:|
| **R2G linear (the claim)** `mean_{A_t} f − f_a` | **−0.004** [−0.062, +0.051] | — | −0.083 | +0.071 | 0.092 |
| strict-v1 linear Guide (Round 1) | 0.170 [0.103, 0.234] | [+0.096, +0.255] | 0.105 | 0.190 | −0.004 |
| per-rule rate (Round-1 control) | 0.182 [0.119, 0.242] | [+0.109, +0.262] | 0.164 | 0.253 | 0.040 |
| loose bound (`EpisodeLabels::compute`) | undefined (1095/1095 true) | — | — | — | — |
| tight bound (`compute_tight`) | undefined (1095/1095 true) | — | — | — | — |
| **strict bound (`compute_strict`)** | **0.389** [0.304, 0.477] | [+0.288, +0.491] | 0.415 | 0.352 | 0.036 |
| strict-by-output-class (proposed after 1b) | 0.005 [−0.056, +0.059] | [−0.067, +0.079] | 0.015 | 0.015 | −0.020 |

Reading. Claim B required the R2G model to exceed *every* bound pooled and on each subset; it exceeds none, on any subset, and the CI of its difference from the three informative proxies excludes zero. The Pearson column is noise on a 92 %-zero Δ and is shown only because the pre-registration named it. The strict bit's ρ ≈ 0.39 is a lower bound on how much of the counterfactual credit *hindsight ancestry* recovers at B = 100; the output-class relaxation that was meant to catch `pythagorean`-shaped unions recovers nothing, because on this sample almost every union touches a class the extraction visits (it fires on 939/1,095 applications and is therefore nearly as uninformative as `loose`).

Per-rule Δ over S (full table in `2026-09-01-counterfactual-credit.md`): the applications that pay at B = 100 are `power-recip` (Δ > 0 on 10/21, mean +0.077), `power-sqrt` (8/15, +0.065), `power-rsqrt` (2/3, +0.095), `constant-fold` (15/46), `involution` (10/92), `even-negation` (9/67); the only rule with a large negative mean is `associative` idx 22 (−0.435: on 3/28 sampled firings removing it freed a slot worth more than the firing). The R2G advantage has the right sign on `power-recip` (+0.012) and on `associative`-22 (−0.007) but at a magnitude that is one rank among ~100 same-sweep alternatives — consistent with ρ ≈ 0.

## 3. Training (§3.2 primary, §3.4 ablations)

Label: `centered_b100 = R(τ,100) − R̄_e(100)`, `R = ln(cost/c*_e)`. Its distribution is the story: on TRAIN the median and both quartiles are 0.000 (33.4 % of records exactly 0; the mint's 67 % zero-spread expressions contribute only zeros), q10/q90 = −0.062/+0.010, but q01/q99 = −4.12/+2.78 with 6.2 % of records at |ŷ| > 1 — the small-`c*_e` tail Round 1 already flagged. The zero-predictor MSE is therefore **0.7054 on TRAIN and 0.4945 on DEV**, and it is dominated by that tail.

The trainer's defaults (`--lr 0.01 --grad-clip 10`, chosen by the trainer's author before any data existed) diverge on this target: TRAIN final loss **0.871 > 0.705** and DEV MSE 0.830 > 0.495 (DEV Spearman 0.032). A three-point sweep was selected **on TRAIN loss only** (DEV was not consulted for the choice): (lr 0.001, clip 10) → 0.7171; (0.001, 1) → 0.7054; **(0.0001, 1) → 0.7047 ← selected**. That is 0.1 % below the zero floor: the candidate-local linear model does not fit the centered return. The canonical checkpoint reproduces the sweep run bit-for-bit (same weights FNV `298cb839455cfe8f`). Weights: bias 0.015, `w_budget` +0.002, `w_match_class` +0.004, `w_neighborhood` −0.001, `w_expr_size` −0.013, `w_rule` ∈ [−0.058, +0.038] (lowest predicted return, fires first: `diff-of-squares`, `canonicalize`, `annihilator`, `half-angle-product`; highest: `reverse-associative`, `exp2-log2-cancel`, `commutative`).

| run | target | B | loss | TRAIN final loss (zero floor) | DEV MSE (zero floor) | DEV Spearman(pred, realized) |
|---|---|---:|---|---:|---:|---:|
| defaults (lr 0.01, clip 10) | centered | 100 | mse | 0.871 (0.705) — diverged | 0.830 (0.495) | 0.032 |
| **primary** (lr 1e-4, clip 1) | centered | 100 | mse | **0.7047** (0.7054) | 0.4946 (0.4945) | 0.099 |
| ablation: raw target | raw | 100 | mse | 2.573 | 1.279 | 0.587 |
| ablation: B = 200 target | centered | 200 | mse | 0.787 | 0.462 | 0.191 |
| ablation: pairwise rank | centered | 100 | logistic (chance = ln 2 = 0.693) | 0.6895 | 0.572 | −0.079 |

The raw-target ablation's DEV Spearman of 0.59 is exactly the nuisance the centering removes: the model learns the expression's absolute regret level from `expr_node_count` (the level *is* predictable), which says nothing about ordering within the expression and is why raw is an ablation and not the primary. The rank objective sits at chance. The B = 200 target is slightly more predictable (0.19) — the tier where more of the outcome is decided by ordering — but was not deployed (the registered primary is B = 100) and would be a post-hoc choice.

**Skew test:** `skew_test_linear_guide --model return`, 5,000 DEV records of the new mint, `|f_trainer + score_deployed|` max 0.000e0, mean 0.000e0, 0/5000 over tol = 1e-6 — **PASS** (bit-exact). The frozen strict-bit checkpoint's skew test also still passes after the `linear.rs` refactor (0/1500).

## 4. The ladder (Claim A) — B = 100 / 200, DEV classical, `sh`, `bezier`

Four arms: (a) unguided, (c) `PerRuleRateGuide` [control], (d) frozen strict-bit `LinearCandidateGuide` (Round-1 checkpoint, unchanged), (e) `LinearReturnGuide` (R2G, `score = −f`). Both guided halves ran through the unchanged `phase3_at_budget_eval` (`--r2g-checkpoint` swaps only the claim-arm scorer; `context.claim_guide` records which ran) on the same grid `[25, 50, 100, 200, 400, 800]`, class cap and cost model as PR #1091; `scripts/r2g_ladder_join.py` joins the two halves per expression and **refuses to join unless (a) and (c) are cost-identical between the halves on every expression** (they were, on all 509). The strict-bit half reproduces every PR #1091 reference median exactly (✓ below). Regret reference = empirical best of all **four** arms at any checkpoint (one arm more than 1b's reference, so (c)/(d) regrets can only be equal or higher than 1b's; ratios are unaffected). Distributions are q1 / median / q3 (p90).

### DEV classical (n = 334)

| B | arm | ratio vs unguided@B | improved / unch / worse | regret vs best | PR #1091 ref |
|---:|---|---|---|---|---:|
| 100 | (c) control | 0.007 / **0.565** / 0.701 (0.841) | 321 / 4 / 9 | 0.00 / 0.64 / 5.63 % | 0.5655 ✓ |
| 100 | (d) strict-bit | 0.004 / **0.537** / 0.681 (0.797) | 323 / 4 / 7 | 0.00 / 0.38 / 1.97 % | 0.5366 ✓ |
| 100 | (e) R2G | 0.005 / **0.549** / 0.736 (0.842) | 324 / 4 / 6 | 0.19 / 1.71 / 15.38 % | — |
| 200 | (c) control | 0.522 / **0.699** / 1.000 (1.000) | 245 / 71 / 18 | 0.00 / 0.20 / 2.85 % | 0.6991 ✓ |
| 200 | (d) strict-bit | 0.512 / **0.696** / 1.000 (1.000) | 245 / 71 / 18 | 0.00 / 0.08 / 0.44 % | 0.6959 ✓ |
| 200 | (e) R2G | 0.525 / **0.714** / 1.000 (1.000) | 244 / 71 / 19 | 0.00 / 0.34 / 2.99 % | — |

Head-to-head (e) vs (d): B = 100 → R2G lower on **25**, equal 102, higher **207**; B = 200 → 10 / 183 / 141. (e) vs (c): 31 / 103 / 200 and 30 / 186 / 118. **Registered:** `m_r2g ≤ 0.5966` → 0.5493 **HOLDS**; `m_r2g ≤ 0.7659` → 0.7137 **HOLDS**. `m_r2g − m_linear` = +0.013 / +0.018, inside M. The median is inside the margin but the per-expression comparison is not a tie: R2G is worse than the strict bit on 62 % of DEV expressions at B = 100 and its q3 regret is 8× the strict bit's.

### `sh` (n = 95) — the domain-shift family

| B | arm | ratio vs unguided@B | improved / unch / worse | regret vs best | PR #1091 ref |
|---:|---|---|---|---|---:|
| 100 | (c) control | 0.892 / **0.903** / 0.921 (0.965) | 95 / 0 / 0 | 2.96 / 7.92 / 17.41 % | 0.9028 ✓ |
| 100 | (d) strict-bit | 0.892 / **0.904** / 0.918 (0.939) | 95 / 0 / 0 | 2.96 / 7.97 / 17.34 % | 0.9039 ✓ |
| 100 | (e) R2G | 0.890 / **0.908** / 0.928 (0.945) | 95 / 0 / 0 | 3.66 / 8.24 / 20.33 % | — |
| 200 | (c) control | 0.879 / **0.894** / 0.914 (0.937) | 95 / 0 / 0 | 1.73 / 6.70 / 16.50 % | 0.8940 ✓ |
| 200 | (d) strict-bit | 0.882 / **0.896** / 0.916 (0.938) | 95 / 0 / 0 | 2.08 / 6.80 / 16.62 % | 0.8959 ✓ |
| 200 | (e) R2G | 0.887 / **0.902** / 0.920 (0.937) | 95 / 0 / 0 | 2.88 / 7.72 / 17.36 % | — |

Head-to-head (e) vs (d): B = 100 → 21 / 1 / **73**; B = 200 → 5 / 15 / **75**. **Registered:** `m_r2g < 0.8439` → 0.9084 **FAILS**; `m_r2g < 0.8259` → 0.9017 **FAILS**. `m_r2g − m_linear` = +0.005 / +0.006 (M = 0.06 / 0.07): a tie by the registered statistic, a small consistent loss per expression.

### `bezier` (n = 80) — reported, no claim

| B | arm | ratio vs unguided@B | improved / unch / worse | regret vs best | PR #1091 ref |
|---:|---|---|---|---|---:|
| 100 | (c) control | 0.910 / **0.910** / 0.949 (0.996) | 80 / 0 / 0 | 13.50 / 17.87 / 35.06 % | 0.9098 ✓ |
| 100 | (d) strict-bit | 0.910 / **0.910** / 0.949 (0.996) | 80 / 0 / 0 | 13.50 / 17.87 / 35.06 % | 0.9098 ✓ |
| 100 | (e) R2G | 0.857 / **0.857** / 0.928 (0.996) | 80 / 0 / 0 | 8.34 / 11.00 / 31.17 % | — |
| 200 | (c) control | 0.910 / **0.910** / 0.980 (1.000) | 60 / 20 / 0 | 13.50 / 17.87 / 30.52 % | 0.9098 ✓ |
| 200 | (d) strict-bit | 0.857 / **0.885** / 0.926 (1.000) | 60 / 20 / 0 | 8.34 / 11.00 / 30.52 % | 0.8855 ✓ |
| 200 | (e) R2G | 0.857 / **0.857** / 0.914 (1.000) | 60 / 20 / 0 | 8.34 / 11.00 / 22.73 % | — |

Head-to-head (e) vs (d): B = 100 → **60** / 20 / 0; B = 200 → 19 / 61 / 0. `m_r2g − m_linear` = −0.053 / −0.029, inside M at both tiers (−0.053 against M_100 = 0.06 is the closest any cell comes to the margin). On the one family where the mint had real spread and no trig, R2G is never worse than the strict bit on any expression, and its B = 100 median (0.857) is below the strict bit's B = 200 median (0.885). This is one polynomial-only family of 80 expressions with an 11-way tie in the mint's own return table; it is reported, not claimed, and it is the only place the R2G ordering did anything the strict bit did not.

### `sh` trig-rule firings (Round-1b §1.3 table for the R2G arm)

`fired (strict-positive)` pooled over 95 expressions; `exprs` = expressions with any firing over the full run.

| idx | rule | unguided @100 / @200 / full (exprs) | strict-bit @100 / @200 / full (exprs) | **R2G** @100 / @200 / full (exprs) |
|---:|---|---|---|---|
| 20 | doubling | 0 / 7 / 37314 (0) (94) | 295 / 575 / 2875 (0) (93) | 396 / 883 / 3533 (0) (93) |
| 36 | sin-angle-addition | 0 / 0 / 1025 (0) (94) | 219 / 332 / 733 (0) (92) | 306 / 478 / 879 (0) (92) |
| 37 | cos-angle-addition | 0 / 0 / 307 (0) (51) | 22 / 37 / 176 (0) (48) | 1 / 46 / 150 (0) (48) |
| 38 | reverse-angle-addition | 0 / 0 / 1134 (50) (94) | 91 / 226 / 619 (0) (92) | 213 / 385 / 785 (0) (92) |
| 39 | half-angle-product | 0 / 0 / 3275 (72) (94) | 360 (134) / 496 (134) / 985 (134) (93) | 545 (134) / 745 (134) / 1147 (134) (93) |
| 40 | **pythagorean** | 0 / 0 / **231 (0) (19)** | **0 / 0 / 0 (0)** | **0 / 0 / 0 (0)** |
| 30–34 | odd/even-negation | 0 everywhere | 0 everywhere | 0 everywhere |

`pythagorean` fired zero times at every checkpoint through 800 applications under both guided policies, and under unguided only past B = 200 — the same picture as Round 1b, now with the R2G ordering added. *Why* the guided loop never fires it (no enumerated match within 800 applications, or a match that is always outranked) was not instrumented in this round and is not asserted here. Whether it *pays* is therefore not answerable by Δ at B = 100 on any trajectory this round produced; the honest cell is "unmeasurable", and getting it to fire at all is the budget-regime question Round 1b promoted, not a label question.

## 5. Verdict against the pre-committed readings (registration §7)

| pre-committed row | fires? | evidence |
|---|---|---|
| A and B hold | no | — |
| A holds, B fails | no (A fails on `sh`) | — |
| A fails on `sh` but B holds | no (B fails) | — |
| **R2G ties the strict bit everywhere** (`\|m_r2g − m_linear\| ≤ M_B` on every set and tier) | **YES** | +0.013, +0.018, +0.005, +0.006, −0.053, −0.029 |
| R2G worse than the strict bit on DEV by > M_B | no | +0.013 / +0.018 < 0.06 / 0.07 (ablations §3.4 were run regardless; none recovers a fit) |
| **dataset gate fires** | **YES** (TRAIN 67.2 %) | `2026-09-01-r2g-dataset.md`; training proceeded on direction, see §0 |

**Reading, as registered: the target was not the bottleneck.** The hindsight return-to-go label — the decision-transformer objective, linear first — produced a Guide that (i) does not fit its own target above the zero floor, (ii) has zero rank correlation with the leave-one-out counterfactual on the very trajectories it was scored on, and (iii) orders saturation indistinguishably from the strict bit by the registered statistic, slightly worse per expression on DEV and `sh`, better only on `bezier`. Meanwhile the strict bit — the hand-drawn proxy this round set out to replace — is the credit definition that correlates best with the counterfactual truth (ρ = 0.39, against 0.17 for the deployed strict-bit Guide's own score and ≈ 0 for the R2G model).

Two things this does not say. It does not say credit is unlearnable: the counterfactual Δ has spread (7.6 % non-zero, with rules that pay by +0.07–0.10 nats and one that costs −0.44), and the strict bit recovers a real part of its rank structure (ρ = 0.39). It says a *candidate-local linear* function of the *trajectory-level* return cannot see it — every application in a trajectory carries the same label (§1.4), so the only discriminating signal is across orderings of the same expression, and 67 % of TRAIN expressions have no such variation at B = 100. That is the gate's finding restated with a trained model attached. And it does not say the transformer rung (§6) is next: §6's pull-forward condition was "the linear R2G model beats every hand-drawn bound on Spearman but that Spearman stays low" — it beat none, so the condition is not met.

**Next lever, per the registration's own row:** context features (`claude/phase3-context`), *scored against the counterfactual Δ before being deployed* — `counterfactual_credit --r2g-checkpoint`/`--strict-checkpoint` now makes that a one-command check for any `SaturationGuide`, and it should gate every future label or feature change before a ladder run. Independently, the B = 100 regime finding from 1b (unguided reaches only the first rule indices within B; `pythagorean` and the trig identities are all past B = 200) is what actually bounds `sh`, and no ordering policy over the current candidate pool moved that.

## 6. Kill-gate accounting

This round trains one new checkpoint from a new label source and counts as one of the five clean rounds under Round 1's kill gate (registration §7: "the label source changed; the gates did not"). Rounds so far: Round 1 (strict bit, DEV claim holds), Round 1b (domain shift, H_null), **this round (R2G, ties-everywhere / gate fired)**.

## 7. What changed in code (no production behavior change)

- `phase3_at_budget_eval --r2g-checkpoint <path>`: the claim arm becomes `LinearReturnGuide`; the arm keeps its `linear` key in rows and aggregates, and `context.claim_guide` + the markdown labels name which Guide ran. Default invocation is byte-identical to before (the strict-bit half of this ladder reproduces PR #1091 to four decimals on all six cells).
- `counterfactual_credit --r2g-checkpoint / --strict-checkpoint / --train-guide-report`: model-advantage proxies per §4.3, seeded paired-bootstrap CIs, per-rule Δ table; the four bounds are computed exactly as before.
- `scripts/r2g_ladder_join.py`: the checked four-arm join (fails loudly on any non-identical shared arm or missing expression).
- `cargo fmt --all` also reformatted `gen_r2g_trajectories.rs` and `pixelflow-search/src/egraph/graph.rs` from earlier commits on this branch (formatting only).
