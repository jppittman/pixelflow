> **Retracted/Superseded (2026-09-07), ledger L035.** The "Guide saves 16%+" headline is a tree-cost number; read in DAG units it is a smaller DEV win and a loss on every structured family, and it was never taken on a shipped shader. Verdict and rationale: `docs/results/2026-09-07-claims-ledger.md` (PR #1207); the corrected benchmark and re-validation order: `docs/plans/2026-09-07-benchmark-correction.md`.

# Phase 3 at-budget evaluation on DEV (ablation ladder against the registered claim)

> **Instrument change (2026-09-02 forward-port).** Two things about how a
> Phase 3 anytime curve is measured changed with the port, so a re-run does
> not reproduce the numbers below even if nothing else changed: the
> application budget now binds **mid-scan** rather than between rule sweeps
> (`app_actual == app_target` exactly, no overshoot), and the reported cost
> is the **DAG** cost the emitted kernel pays rather than the extraction DP's
> tree total (#1117). Full statement:
> [docs/results/2026-09-02-phase3-instrument-changes.md](2026-09-02-phase3-instrument-changes.md).

**Date:** 2026-09-01 · **Registration:** `docs/plans/2026-09-01-phase3-registration.md` (unrevised) · **Authority:** `docs/plans/2026-08-31-guide-design-revision.md` §5

Per-expression rows: `docs/results/2026-09-01-phase3-at-budget-eval.jsonl` (394 expressions: blitz 30, classical 334, rapid 30). Skipped: none.

Cost = `CostModel::latency_prior()` `extract_dag` cost, deterministic; no wall-clock in any number below. All arms go through `egraph::anytime::run_anytime_curve_with`; guided arms via `GuidedSaturation` (dedup set carried across checkpoints). Distributions are q1 / median / q3 (p90). Regret reference = empirical best of ALL arms at ANY checkpoint for that expression. Strict-oracle arm: not run (see the harness module doc — no cheap provenance replay exists for it).

## classical band, B = 100 (REGISTERED: Y = 16.3% → median ratio ≤ 0.837; 4B-approach: median gap ≤ 24.2%)

n = 334.

| arm | ratio vs unguided@B (q1/med/q3, p90) | improved / unchanged / worse | gap vs unguided@4B | regret vs best | gap closed (n with gap) | structural share @B | strict precision @B | rounds to B (med) |
|---|---|---|---|---|---|---|---|---|
| (a) unguided @B | 1.000 (by definition) | — | 0.39% / 51.66% / 178.61% (p90 48115.2%) | 54.19% / 96.17% / 28395.31% (p90 75649.2%) | 0 | 0.589 | 0.0212 | — |
| (b) unguided @4B | — | — | 0 | 0.00% / 1.10% / 57.51% (p90 22210.4%) | 1 | — | — | — |
| (c) PerRuleRateGuide @B [control] | 0.007 / 0.565 / 0.701 (p90 0.841) | 321 / 4 / 9 | -28.68% / 0.00% / 0.21% (p90 1.0%) | 0.00% / 0.64% / 5.63% (p90 24.7%) | 1.000 / 1.000 / 4.784 (p90 202.792) (n=308) | 0.279 | 0.2248 | 4 |
| (d) LinearCandidateGuide @B [claim] | 0.004 / 0.537 / 0.681 (p90 0.797) | 323 / 4 / 7 | -31.27% / 0.00% / 0.00% (p90 0.9%) | 0.00% / 0.38% / 1.97% (p90 10.2%) | 1.000 / 1.000 / 20.613 (p90 388.785) (n=308) | 0.322 | 0.2237 | 3 |

Ladder diagnostics: linear < control on 131 expressions, equal on 134, linear > control on 69. control: quiesced before B on 41 / 334 (ended at 132.250 / 294.000 / 750.250 (p90 800.000) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 189; vs production cost better / equal / worse = 24 / 101 / 209. linear: quiesced before B on 39 / 334 (ended at 134.250 / 293.000 / 744.250 (p90 800.000) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 199; vs production cost better / equal / worse = 37 / 105 / 192. 

**Verdict (B=100):** (d) LinearCandidateGuide median ratio vs unguided-at-B = **0.537** against the registered threshold ≤ 0.837 (Y = 16.3%) — the Y-clause **HOLDS**; median gap vs unguided-at-4B = 0.00% against ≤ 24.2% — the 4B-approach clause **HOLDS**; (d) beats unguided-at-B at all (median ratio < 1.0, kill-gate view). Ladder: (c) control median ratio 0.565, regret 0.64% vs (d) regret 0.38% — (d) **beats** (c).

## classical band, B = 200 (REGISTERED: Y = 9.0% → median ratio ≤ 0.910; 4B-approach: median gap ≤ 11.0%)

n = 334.

| arm | ratio vs unguided@B (q1/med/q3, p90) | improved / unchanged / worse | gap vs unguided@4B | regret vs best | gap closed (n with gap) | structural share @B | strict precision @B | rounds to B (med) |
|---|---|---|---|---|---|---|---|---|
| (a) unguided @B | 1.000 (by definition) | — | 0.00% / 16.89% / 78.93% (p90 29102.3%) | 0.70% / 51.45% / 176.86% (p90 50849.3%) | 0 | 0.596 | 0.0296 | — |
| (b) unguided @4B | — | — | 0 | 0.00% / 0.00% / 5.66% (p90 17630.8%) | 1 | — | — | — |
| (c) PerRuleRateGuide @B [control] | 0.522 / 0.699 / 1.000 (p90 1.000) | 245 / 71 / 18 | 0.00% / 0.00% / 0.23% (p90 2.8%) | 0.00% / 0.20% / 2.85% (p90 16.5%) | 0.994 / 1.000 / 1.000 (p90 2.063) (n=228) | 0.403 | 0.1422 | 5 |
| (d) LinearCandidateGuide @B [claim] | 0.512 / 0.696 / 1.000 (p90 1.000) | 245 / 71 / 18 | -0.27% / 0.00% / 0.09% (p90 0.4%) | 0.00% / 0.08% / 0.44% (p90 5.8%) | 0.999 / 1.000 / 1.001 (p90 1.998) (n=228) | 0.430 | 0.1460 | 5 |

Ladder diagnostics: linear < control on 114 expressions, equal on 202, linear > control on 18. control: quiesced before B on 126 / 334 (ended at 132.250 / 294.000 / 750.250 (p90 800.000) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 189; vs production cost better / equal / worse = 21 / 137 / 176. linear: quiesced before B on 128 / 334 (ended at 134.250 / 293.000 / 744.250 (p90 800.000) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 199; vs production cost better / equal / worse = 29 / 149 / 156. 

**Verdict (B=200):** (d) LinearCandidateGuide median ratio vs unguided-at-B = **0.696** against the registered threshold ≤ 0.910 (Y = 9.0%) — the Y-clause **HOLDS**; median gap vs unguided-at-4B = 0.00% against ≤ 11.0% — the 4B-approach clause **HOLDS**; (d) beats unguided-at-B at all (median ratio < 1.0, kill-gate view). Ladder: (c) control median ratio 0.699, regret 0.20% vs (d) regret 0.08% — (d) **beats** (c).

## rapid band, B = 100 (reported for completeness, NO claim registered)

n = 30.

| arm | ratio vs unguided@B (q1/med/q3, p90) | improved / unchanged / worse | gap vs unguided@4B | regret vs best | gap closed (n with gap) | structural share @B | strict precision @B | rounds to B (med) |
|---|---|---|---|---|---|---|---|---|
| (a) unguided @B | 1.000 (by definition) | — | 0.00% / 0.00% / 0.00% (p90 0.0%) | 0.00% / 0.00% / 0.00% (p90 0.0%) | 0 | 0.452 | 0.0396 | — |
| (b) unguided @4B | — | — | 0 | 0.00% / 0.00% / 0.00% (p90 0.0%) | 1 | — | — | — |
| (c) PerRuleRateGuide @B [control] | 1.000 / 1.000 / 1.000 (p90 1.000) | 2 / 28 / 0 | 0.00% / 0.00% / 0.00% (p90 0.0%) | 0.00% / 0.00% / 0.00% (p90 0.0%) | 0.974 / 0.979 / 0.984 (p90 0.987) (n=2) | 0.377 | 0.0776 | 4 |
| (d) LinearCandidateGuide @B [claim] | 1.000 / 1.000 / 1.000 (p90 1.000) | 2 / 28 / 0 | 0.00% / 0.00% / 0.00% (p90 0.0%) | 0.00% / 0.00% / 0.00% (p90 0.0%) | 0.974 / 0.979 / 0.984 (p90 0.987) (n=2) | 0.387 | 0.0783 | 4 |

Ladder diagnostics: linear < control on 0 expressions, equal on 30, linear > control on 0. control: quiesced before B on 29 / 30 (ended at 18.250 / 29.000 / 50.750 (p90 56.100) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 28; vs production cost better / equal / worse = 0 / 28 / 2. linear: quiesced before B on 29 / 30 (ended at 18.250 / 28.000 / 49.250 (p90 57.300) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 28; vs production cost better / equal / worse = 0 / 28 / 2. 

## rapid band, B = 200 (reported for completeness, NO claim registered)

n = 30.

| arm | ratio vs unguided@B (q1/med/q3, p90) | improved / unchanged / worse | gap vs unguided@4B | regret vs best | gap closed (n with gap) | structural share @B | strict precision @B | rounds to B (med) |
|---|---|---|---|---|---|---|---|---|
| (a) unguided @B | 1.000 (by definition) | — | 0.00% / 0.00% / 0.00% (p90 0.0%) | 0.00% / 0.00% / 0.00% (p90 0.0%) | 0 | 0.498 | 0.0322 | — |
| (b) unguided @4B | — | — | 0 | 0.00% / 0.00% / 0.00% (p90 0.0%) | 1 | — | — | — |
| (c) PerRuleRateGuide @B [control] | 1.000 / 1.000 / 1.000 (p90 1.000) | 0 / 28 / 2 | 0.00% / 0.00% / 0.00% (p90 0.0%) | 0.00% / 0.00% / 0.00% (p90 0.0%) | n=0 (n=0) | 0.416 | 0.0707 | 4 |
| (d) LinearCandidateGuide @B [claim] | 1.000 / 1.000 / 1.000 (p90 1.000) | 0 / 28 / 2 | 0.00% / 0.00% / 0.00% (p90 0.0%) | 0.00% / 0.00% / 0.00% (p90 0.0%) | n=0 (n=0) | 0.416 | 0.0713 | 4 |

Ladder diagnostics: linear < control on 0 expressions, equal on 30, linear > control on 0. control: quiesced before B on 29 / 30 (ended at 18.250 / 29.000 / 50.750 (p90 56.100) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 28; vs production cost better / equal / worse = 0 / 28 / 2. linear: quiesced before B on 29 / 30 (ended at 18.250 / 28.000 / 49.250 (p90 57.300) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 28; vs production cost better / equal / worse = 0 / 28 / 2. 

## blitz band, B = 100 (reported for completeness, NO claim registered)

n = 30.

| arm | ratio vs unguided@B (q1/med/q3, p90) | improved / unchanged / worse | gap vs unguided@4B | regret vs best | gap closed (n with gap) | structural share @B | strict precision @B | rounds to B (med) |
|---|---|---|---|---|---|---|---|---|
| (a) unguided @B | 1.000 (by definition) | — | 0.00% / 0.00% / 0.00% (p90 0.0%) | 0.00% / 0.00% / 0.00% (p90 0.0%) | 0 | 0.432 | 0.0667 | — |
| (b) unguided @4B | — | — | 0 | 0.00% / 0.00% / 0.00% (p90 0.0%) | 1 | — | — | — |
| (c) PerRuleRateGuide @B [control] | 1.000 / 1.000 / 1.000 (p90 1.000) | 0 / 30 / 0 | 0.00% / 0.00% / 0.00% (p90 0.0%) | 0.00% / 0.00% / 0.00% (p90 0.0%) | n=0 (n=0) | 0.372 | 0.0842 | 4 |
| (d) LinearCandidateGuide @B [claim] | 1.000 / 1.000 / 1.000 (p90 1.000) | 0 / 30 / 0 | 0.00% / 0.00% / 0.00% (p90 0.0%) | 0.00% / 0.00% / 0.00% (p90 0.0%) | n=0 (n=0) | 0.342 | 0.0854 | 4 |

Ladder diagnostics: linear < control on 0 expressions, equal on 30, linear > control on 0. control: quiesced before B on 29 / 30 (ended at 7.000 / 9.500 / 11.000 (p90 12.200) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 30; vs production cost better / equal / worse = 0 / 30 / 0. linear: quiesced before B on 29 / 30 (ended at 7.000 / 9.500 / 11.000 (p90 12.100) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 30; vs production cost better / equal / worse = 0 / 30 / 0. 

## blitz band, B = 200 (reported for completeness, NO claim registered)

n = 30.

| arm | ratio vs unguided@B (q1/med/q3, p90) | improved / unchanged / worse | gap vs unguided@4B | regret vs best | gap closed (n with gap) | structural share @B | strict precision @B | rounds to B (med) |
|---|---|---|---|---|---|---|---|---|
| (a) unguided @B | 1.000 (by definition) | — | 0.00% / 0.00% / 0.00% (p90 0.0%) | 0.00% / 0.00% / 0.00% (p90 0.0%) | 0 | 0.496 | 0.0557 | — |
| (b) unguided @4B | — | — | 0 | 0.00% / 0.00% / 0.00% (p90 0.0%) | 1 | — | — | — |
| (c) PerRuleRateGuide @B [control] | 1.000 / 1.000 / 1.000 (p90 1.000) | 0 / 30 / 0 | 0.00% / 0.00% / 0.00% (p90 0.0%) | 0.00% / 0.00% / 0.00% (p90 0.0%) | n=0 (n=0) | 0.423 | 0.0724 | 4 |
| (d) LinearCandidateGuide @B [claim] | 1.000 / 1.000 / 1.000 (p90 1.000) | 0 / 30 / 0 | 0.00% / 0.00% / 0.00% (p90 0.0%) | 0.00% / 0.00% / 0.00% (p90 0.0%) | n=0 (n=0) | 0.413 | 0.0716 | 4 |

Ladder diagnostics: linear < control on 0 expressions, equal on 30, linear > control on 0. control: quiesced before B on 30 / 30 (ended at 7.000 / 9.500 / 11.000 (p90 12.200) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 30; vs production cost better / equal / worse = 0 / 30 / 0. linear: quiesced before B on 30 / 30 (ended at 7.000 / 9.500 / 11.000 (p90 12.100) applications; burned-key share 0.000 / 0.000 / 0.000 (p90 0.000)); reaches the empirical best on 30; vs production cost better / equal / worse = 0 / 30 / 0. 

## Enabler-starvation diagnostics

### classical (n = 334)

Unguided strict-positive applications: 8902 total, 7588 numeric (non-structural). Of the numeric ones, **6290** (82.9%) have a structural application in their tight derivation ancestry ("structurally enabled"), 619 have a direct child whose chosen node a structural rule created.

| guided arm | numeric strict-positive terms unguided reached that this arm's final e-graph contains | of the structurally-enabled ones |
|---|---|---|
| control | 6365 / 7588 (83.9%) | 5070 / 6290 (80.6%) |
| linear | 6346 / 7588 (83.6%) | 5051 / 6290 (80.3%) |

| arm | structural share of applications @100 | @200 | top rules @100 |
|---|---|---|---|
| control | 0.279 | 0.403 | even-negation 10168, commutative 6042, involution 3659, power-sqrt 2650, constant-fold 2263, power-recip 1950, fma-fusion 1505, associative 760 |
| linear | 0.322 | 0.430 | even-negation 8268, commutative 6149, constant-fold 4334, fma-fusion 2783, power-sqrt 2516, power-recip 1970, involution 1932, associative 684 |
| unguided | 0.589 | 0.596 | commutative 23880, involution 9981, constant-fold 5128, canonicalize 792, even-negation 702, reverse-associative 106, associative 96, inverse-annihilation 72 |

- control: distinct candidate keys scored per recorded application over the run (dedup coverage): 1.000 / 1.000 / 1.000 (p90 1.000)
- linear: distinct candidate keys scored per recorded application over the run (dedup coverage): 1.000 / 1.000 / 1.000 (p90 1.000)

### rapid (n = 30)

Unguided strict-positive applications: 82 total, 74 numeric (non-structural). Of the numeric ones, **45** (60.8%) have a structural application in their tight derivation ancestry ("structurally enabled"), 2 have a direct child whose chosen node a structural rule created.

| guided arm | numeric strict-positive terms unguided reached that this arm's final e-graph contains | of the structurally-enabled ones |
|---|---|---|
| control | 71 / 74 (95.9%) | 42 / 45 (93.3%) |
| linear | 71 / 74 (95.9%) | 42 / 45 (93.3%) |

| arm | structural share of applications @100 | @200 | top rules @100 |
|---|---|---|---|
| control | 0.377 | 0.416 | commutative 295, involution 266, even-negation 206, constant-fold 62, power-sqrt 34, fma-fusion 28, power-recip 28, reverse-associative 28 |
| linear | 0.387 | 0.416 | commutative 295, involution 254, even-negation 200, constant-fold 65, power-sqrt 34, fma-fusion 33, power-recip 28, reverse-associative 26 |
| unguided | 0.452 | 0.498 | commutative 710, involution 563, even-negation 228, constant-fold 166, reverse-associative 65, associative 56, power-sqrt 32, fma-fusion 26 |

- control: distinct candidate keys scored per recorded application over the run (dedup coverage): 1.000 / 1.000 / 1.000 (p90 1.000)
- linear: distinct candidate keys scored per recorded application over the run (dedup coverage): 1.000 / 1.000 / 1.000 (p90 1.000)

### blitz (n = 30)

Unguided strict-positive applications: 31 total, 30 numeric (non-structural). Of the numeric ones, **12** (40.0%) have a structural application in their tight derivation ancestry ("structurally enabled"), 1 have a direct child whose chosen node a structural rule created.

| guided arm | numeric strict-positive terms unguided reached that this arm's final e-graph contains | of the structurally-enabled ones |
|---|---|---|
| control | 30 / 30 (100.0%) | 12 / 12 (100.0%) |
| linear | 30 / 30 (100.0%) | 12 / 12 (100.0%) |

| arm | structural share of applications @100 | @200 | top rules @100 |
|---|---|---|---|
| control | 0.372 | 0.423 | involution 100, even-negation 79, commutative 73, reverse-associative 22, associative 19, power-sqrt 14, constant-fold 13, fma-fusion 12 |
| linear | 0.342 | 0.413 | involution 100, even-negation 78, commutative 69, constant-fold 14, fma-fusion 14, power-sqrt 14, reverse-associative 14, associative 13 |
| unguided | 0.432 | 0.496 | commutative 148, involution 128, even-negation 82, associative 25, constant-fold 17, power-sqrt 14, reverse-associative 14, identity 10 |

- control: distinct candidate keys scored per recorded application over the run (dedup coverage): 1.000 / 1.000 / 1.000 (p90 1.000)
- linear: distinct candidate keys scored per recorded application over the run (dedup coverage): 1.000 / 1.000 / 1.000 (p90 1.000)

## Production-units context (exact production saturation call per expression)

`production_saturation_probe` = the same function body `optimize_runtime_arena` runs (`config_for_node_count` → `saturate_with_full_budget`), stop reason READ from the loop. Wall-clock is a stop condition of that call only; `timeout` stops are machine-dependent.

### classical (n = 334, probe returned None for 0)

- stop reasons: quiesced 314, timeout 20
- effective B (applications at stop): 399.750 / 1671.000 / 9441.000 (p90 23798.700) (max 85900)
- share with applications ≥ 100 / 200 / 400 / 800 / 1600: ≥100: 98.5%, ≥200: 91.6%, ≥400: 74.9%, ≥800: 59.6%, ≥1600: 50.3%
- rounds run: 4.000 / 5.000 / 8.000 (p90 10.000); classes at stop: 139.500 / 329.500 / 1053.250 (p90 4062.600)
- production cost / unguided cost@100: 0.236 / 0.536 / 0.688 (p90 0.995)
- production cost / unguided cost@200: 0.520 / 0.708 / 1.000 (p90 1.000)
- production cost / unguided cost@800: 0.996 / 1.000 / 1.000 (p90 1.000)
- production regret vs empirical best: 0.00% / 0.00% / 0.00% (p90 15940.9%)
- equivalent unguided checkpoint (smallest grid B whose unguided cost ≤ production's; 0 = worse than every checkpoint): 0: 4, 25: 22, 50: 1, 100: 1, 200: 75, 400: 72, 800: 57, 1600: 50, 3200: 26, 6400: 8, 12800: 14, 25600: 4
- unguided checkpoint whose app_actual first reaches production's application count: 100: 6, 200: 24, 400: 51, 800: 47, 1600: 28, 3200: 40, 6400: 27, 12800: 27, 25600: 36, 51200: 16, 102400: 6, beyond grid: 26

### rapid (n = 30, probe returned None for 0)

- stop reasons: quiesced 30
- effective B (applications at stop): 23.250 / 49.000 / 118.750 (p90 167.100) (max 1862)
- share with applications ≥ 100 / 200 / 400 / 800 / 1600: ≥100: 30.0%, ≥200: 10.0%, ≥400: 3.3%, ≥800: 3.3%, ≥1600: 3.3%
- rounds run: 2.000 / 2.000 / 3.000 (p90 4.100); classes at stop: 20.250 / 27.500 / 55.000 (p90 58.100)
- production cost / unguided cost@100: 1.000 / 1.000 / 1.000 (p90 1.000)
- production cost / unguided cost@200: 1.000 / 1.000 / 1.000 (p90 1.000)
- production cost / unguided cost@800: 1.000 / 1.000 / 1.000 (p90 1.000)
- production regret vs empirical best: 0.00% / 0.00% / 0.00% (p90 0.0%)
- equivalent unguided checkpoint (smallest grid B whose unguided cost ≤ production's; 0 = worse than every checkpoint): 25: 25, 50: 2, 100: 1, 200: 2
- unguided checkpoint whose app_actual first reaches production's application count: 25: 9, 50: 6, 100: 6, 200: 5, 400: 2, 3200: 1, beyond grid: 1

### blitz (n = 30, probe returned None for 0)

- stop reasons: quiesced 29, timeout 1
- effective B (applications at stop): 9.000 / 11.500 / 13.000 (p90 16.200) (max 6198)
- share with applications ≥ 100 / 200 / 400 / 800 / 1600: ≥100: 3.3%, ≥200: 3.3%, ≥400: 3.3%, ≥800: 3.3%, ≥1600: 3.3%
- rounds run: 2.000 / 2.000 / 2.000 (p90 2.000); classes at stop: 9.000 / 11.000 / 12.750 (p90 13.100)
- production cost / unguided cost@100: 1.000 / 1.000 / 1.000 (p90 1.000)
- production cost / unguided cost@200: 1.000 / 1.000 / 1.000 (p90 1.000)
- production cost / unguided cost@800: 1.000 / 1.000 / 1.000 (p90 1.000)
- production regret vs empirical best: 0.00% / 0.00% / 0.00% (p90 0.0%)
- equivalent unguided checkpoint (smallest grid B whose unguided cost ≤ production's; 0 = worse than every checkpoint): 25: 30
- unguided checkpoint whose app_actual first reaches production's application count: 25: 28, 50: 1, 6400: 1

## Context (not metrics)

- arms: ["unguided", "control", "linear"]
- checkpoint: pixelflow-pipeline/data/guide_checkpoint_strict_v1.json
- guided_grid: [25, 50, 100, 200, 400, 800] (unguided: [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800])
- load_at_end: 13:17  up 51 days,  3:50, 2 users, load averages: 26.10 22.86 15.62
- load_at_start: 13:16  up 51 days,  3:49, 2 users, load averages: 34.06 21.92 14.67
- source_rev: 7cade3112fd947b0ad1e1f46ecc3e8586840f058
- structural_rules: ["commutative", "associative", "reverse-associative", "distribute", "fma-fusion", "identity"]
- train_guide_report: docs/results/2026-09-01-train-guide-report.json
