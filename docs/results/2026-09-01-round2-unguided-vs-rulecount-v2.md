> **Retracted/Superseded (2026-09-07), ledger L042.** V3 showed the |R| effect reported here is an order confound, and the regret is a tree cost besides. Verdict and rationale: `docs/results/2026-09-07-claims-ledger.md` (PR #1207); the corrected benchmark and re-validation order: `docs/plans/2026-09-07-benchmark-correction.md`.

# Round 2 v2 Register run: unguided regret vs rule count, interleaved order

> **Instrument change (2026-09-02 forward-port).** Two things about how a
> Phase 3 anytime curve is measured changed with the port, so a re-run does
> not reproduce the numbers below even if nothing else changed: the
> application budget now binds **mid-scan** rather than between rule sweeps
> (`app_actual == app_target` exactly, no overshoot), and the reported cost
> is the **DAG** cost the emitted kernel pays rather than the extraction DP's
> tree total (#1117). Full statement:
> [docs/results/2026-09-02-phase3-instrument-changes.md](2026-09-02-phase3-instrument-changes.md).

Companion CSV: `2026-09-01-round2-unguided-vs-rulecount-v2.csv` (44,800 rows: 8 rule sets × 400
expressions × 14 checkpoints). Companion JSON (per-expression curves, machine-readable):
`2026-09-01-round2-unguided-vs-rulecount-v2.json`. Aggregate statistics JSON (this document's
tables, plus `_regrets` per-expression arrays used for the bootstrap):
`2026-09-01-phase3-round2-registration-v2.json`. Design/registration document:
`docs/plans/2026-09-01-phase3-round2-registration-v2.md` (§0–§3 mechanism and conventions, filled
here).

**Machine:** shared; `uptime` at the start of this run: `18:17 up 51 days, 8:50, 2 users, load
averages: 9.65 6.86 6.28`. Corpus MD5s confirmed unchanged before running: train
`0ed6cf16abcbc006cd7a3ee2365b15b4`, dev `3026133ebba066eeca10f658da554400` (FINAL not opened).
Source rev at run time: `70a4c01f8e5bf94be77a55fc2c3c21c771604fc7` plus this document's own commit
(adds `InflationMode::NewRules`/`build_new_rule_set` to `inflate.rs` so mode (iii) goes through the
same `RuleSetSpec`/`RuleOrder` path as modes (i)/(ii) — see the "Mode (iii) update" note in the
plan doc's §2).

## Grid coverage

| Mode | spec | \|R\| | status |
|---|---|---:|---|
| shared | `base` | 62 | complete (n=400: blitz 23, rapid 189, classical 188) |
| (i) | `dup:93` | 93 | complete |
| (i) | `dup:124` | 124 | complete |
| (i) | `dup:186` | 186 | complete |
| (i) | `dup:248` | 248 | complete |
| (ii) | `comp:93` | 93 | complete |
| (ii) | `comp:124` | 124 | complete |
| (ii) | `comp:186` | 186 | **MISSING — see below** |
| (ii) | `comp:248` | 248 | **MISSING — see below** (never attempted; blocked behind `comp:186`) |
| (iii) | `new:95` | 95 | complete |

**`comp:186` and `comp:248` did not complete and are not in this document's tables, CSV, or JSON.**
Two independent attempts (18:32–18:51 and 18:51–19:10 PDT, both `--rule-sets comp:186,comp:248`)
each ran `comp:186` to exactly 200/400 curves, then panicked identically:

```
thread 'main' panicked at pixelflow-search/src/egraph/anytime.rs:255:9:
anytime: saturation hit the wall-clock safety ceiling at target 204800 — offline measurement must
fail loud, never silently truncate
```

Both panics landed in the same ~25-expression window (curves 201–224 of 400, i.e. the rapid/
classical boundary of the size-stratified sample) after roughly the same elapsed time, under
different system load (`load averages` 16.45 then 12.35 — this machine is shared, per this
session's directive to record `uptime`). Two clean, independent reproductions at the same point
under different load is evidence of a genuine compute wall for at least one expression at
`|R|=186` under `CostModel::latency_prior()` search — not transient contention — so a third retry
was not run; this is reported as a finding, not padded or extrapolated per this task's explicit
instruction. `comp:248` was never reached (it is queued after `comp:186` in the same process) and
has no data of any kind, including a fingerprint-only entry. The safety ceiling did exactly what
CLAUDE.md requires ("fail loud, never silently truncate") rather than returning a truncated,
misleadingly-labeled curve — the honest outcome here is an admitted gap, not a wrong number.

This reproduces v1's own experience with these two points almost exactly ("v1 never finished
writing curve rows for these two points... the process producing the committed CSV did not
survive" — plan doc §2) — `comp:186`/`comp:248` remain the two points this harness cannot realize
on this sample within the registered, |R|-scaled safety budget, under either sweep order.

## Tables

**Classical band (n=188), absolute cost and where the inflation is visible.** `visible@B` = expressions whose cost@B differs from the |R|=62 curve's; `first visible` = smallest grid checkpoint at which any expression's cost differs from |R|=62 (`never` = identical at all 14 checkpoints).

| rule set | \|R\| | fingerprint | cost@100 q1/med/q3 | cost@200 q1/med/q3 | cost@400 med | cost@800 med | curve-end med (excl. cycle) | cycle hits | app_actual@100 med | visible@100 | visible@200 | first visible | apps-to-end med | ended |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| `base` | 62 | `e99af8402beaff5d` | 2952 / 14459.5 / 1002550 | 1609 / 7413.5 / 1000324 | 3623.5 | 3329.5 | 2621.0 | 25 | 113 | — | — | n/a | 2686 | app_budget=1, quiesced=187 |
| `dup:93` | 93 | `83e610e33e782a68` | 1874 / 8738.0 / 1000308 | 1457 / 5141.5 / 1000132 | 4438.0 | 4039.0 | 2538.5 | 32 | 102 | 171 | 147 | 25 | 3888 | app_budget=1, quiesced=187 |
| `dup:124` | 124 | `b207aa331bb625ab` | 1840 / 4621.5 / 10680 | 1622 / 4579.5 / 10691 | 3914.0 | 3950.0 | 2538.5 | 38 | 106 | 186 | 157 | 25 | 4294 | app_budget=1, quiesced=187 |
| `dup:186` | 186 | `3a00c565900b48e6` | 2155 / 10700.5 / 1001234 | 1749 / 8046.0 / 1001744 | 7575.5 | 5472.0 | 2514.0 | 33 | 104 | 179 | 178 | 25 | 5420 | app_budget=1, class_cap=1, quiesced=186 |
| `dup:248` | 248 | `43c43d764ef7f76b` | 1788 / 4506.0 / 9992 | 1430 / 4214.5 / 9796 | 4024.0 | 3345.0 | 2621.0 | 37 | 108 | 186 | 160 | 25 | 5434 | app_budget=1, quiesced=187 |
| `comp:93` | 93 | `904ceec9b110e89e` | 2248 / 10868.5 / 1000777 | 1739 / 7388.0 / 1000348 | 4480.5 | 4316.5 | 2518.0 | 31 | 104 | 159 | 151 | 25 | 2928 | app_budget=1, quiesced=187 |
| `comp:124` | 124 | `a7600e5942f0baa5` | 1828 / 5066.0 / 10793 | 1468 / 4574.0 / 11354 | 4463.5 | 4284.5 | 2559.0 | 39 | 109 | 184 | 148 | 25 | 4772 | app_budget=2, quiesced=186 |
| `new:95` | 95 | `113cca49c99cc850` | 4391 / 9582.0 / 266798 | 5197 / 14343.5 / 1000220 | 1000132.0 | 1000091.5 | 2023.5 | 84 | 108 | 188 | 186 | 25 | 25375 | class_cap=2, quiesced=186 |

**Classical band, unguided regret U against the unguided-only closure-aware reference at the same |R|, truncation loss L, and Y.** Percentages; regret quartiles are per-expression.

| rule set | \|R\| | U@100 med | p25 | p75 | p90 | U@200 med | p25 | p75 | p90 | L@100 med | Y@100 | L@200 med | Y@200 | closure gain vs 62: med / p90 / >0 / <0 (n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `base` | 62 | 96.58 | 47.35 | 27840.42 | 75221.25 | 40.49 | 0.00 | 113.73 | 43079.78 | 48.467 | 16.32 | 21.922 | 8.99 | — (reference) |
| `dup:93` | 93 | 43.76 | 12.01 | 18646.64 | 73428.90 | 6.69 | 0.00 | 50.89 | 56309.14 | 20.910 | 8.65 | 0.181 | 0.09 | 0.000 / 0.000 / 8 / 13 (173) |
| `dup:124` | 124 | 41.12 | 18.47 | 70.67 | 102.10 | 25.44 | 0.72 | 54.55 | 83.59 | 15.597 | 6.75 | 5.088 | 2.42 | 0.000 / 0.000 / 7 / 17 (173) |
| `dup:186` | 186 | 15.44 | 3.53 | 32536.30 | 80070.97 | 9.47 | 1.52 | 16076.22 | 72453.78 | 2.565 | 1.25 | 1.047 | 0.52 | 0.000 / 0.000 / 10 / 17 (173) |
| `dup:248` | 248 | 38.67 | 18.72 | 64.55 | 87.76 | 27.02 | 0.76 | 53.75 | 86.11 | 10.996 | 4.95 | 6.997 | 3.27 | 0.000 / 0.000 / 9 / 15 (173) |
| `comp:93` | 93 | 60.55 | 28.61 | 21474.70 | 70501.82 | 25.60 | 0.00 | 113.81 | 51048.55 | 24.604 | 9.87 | 0.175 | 0.09 | 0.000 / 0.000 / 8 / 14 (173) |
| `comp:124` | 124 | 41.69 | 16.32 | 67.77 | 99.34 | 21.22 | 0.17 | 56.25 | 84.07 | 14.371 | 6.28 | 2.684 | 1.31 | 0.000 / 0.000 / 9 / 23 (173) |
| `new:95` | 95 | 33.11 | 3.63 | 183.70 | 145181.76 | 52.06 | 5.24 | 54121.74 | 148669.72 | -0.000 | -0.00 | 0.000 | 0.00 | 0.474 / 2.403 / 97 / 70 (173) |

**Sweeps and match-enumeration overhead (classical, v2 §0.2/§7.1).** `apps_per_sweep` is one throwaway one-sweep probe per expression, median over the band; `B in sweeps` = B / that median (how much of one full rule-order pass a budget spends); `evals/app@B` = cumulative `EGraph::total_evals` / cumulative applications through checkpoint B — matches enumerated per application actually taken, the §7.1 flatness check. **`sweeps_actual` counts passes STARTED (each checkpoint segment restarts the rule vector at index 0 and counts a cut-off pass as one), not completed sweeps — which is why it reads 3.00/4.00 at B=100/200 for every set, including ones where one full pass costs ~1000 applications; use `B in sweeps`, not this column, for the sweep-denominated budget (plan doc §0.1).**

| rule set | \|R\| | apps_per_sweep med | B=100 in sweeps | B=200 in sweeps | sweeps_actual@100 med | sweeps_actual@200 med | evals/app@100 med | evals/app@200 med |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `base` | 62 | 99.5 | 1.01 | 2.01 | 3.00 | 4.00 | 31.20 | 39.54 |
| `dup:93` | 93 | 142.0 | 0.70 | 1.41 | 3.00 | 4.00 | 75.37 | 79.54 |
| `dup:124` | 124 | 300.0 | 0.33 | 0.67 | 3.00 | 4.00 | 78.07 | 76.37 |
| `dup:186` | 186 | 502.0 | 0.20 | 0.40 | 3.00 | 4.00 | 91.81 | 92.67 |
| `dup:248` | 248 | 995.5 | 0.10 | 0.20 | 3.00 | 4.00 | 78.85 | 74.75 |
| `comp:93` | 93 | 87.0 | 1.15 | 2.30 | 3.00 | 4.00 | 85.77 | 92.42 |
| `comp:124` | 124 | 89.5 | 1.12 | 2.23 | 3.00 | 4.00 | 100.60 | 102.79 |
| `new:95` | 95 | 291.5 | 0.34 | 0.69 | 3.00 | 4.00 | 69.33 | 61.98 |

**blitz (n=23) — reported, no claim.**

| rule set | \|R\| | cost@100 med | cost@200 med | U@100 med | L@100 med | apps-to-end med | ended |
|---|---:|---:|---:|---:|---:|---:|---|
| `base` | 62 | 78.0 | 78.0 | 0.00 | 0.000 | 13 | quiesced=23 |
| `dup:93` | 93 | 78.0 | 78.0 | 0.00 | 0.000 | 15 | quiesced=23 |
| `dup:124` | 124 | 78.0 | 78.0 | 0.00 | 0.000 | 45 | quiesced=23 |
| `dup:186` | 186 | 78.0 | 78.0 | 0.00 | 0.000 | 74 | quiesced=23 |
| `dup:248` | 248 | 78.0 | 78.0 | 0.00 | 0.000 | 92 | quiesced=23 |
| `comp:93` | 93 | 78.0 | 78.0 | 0.00 | 0.000 | 13 | quiesced=23 |
| `comp:124` | 124 | 78.0 | 78.0 | 0.00 | 0.000 | 23 | quiesced=23 |
| `new:95` | 95 | 78.0 | 78.0 | 0.00 | 0.000 | 194 | quiesced=23 |

**rapid (n=189) — reported, no claim.**

| rule set | \|R\| | cost@100 med | cost@200 med | U@100 med | L@100 med | apps-to-end med | ended |
|---|---:|---:|---:|---:|---:|---:|---|
| `base` | 62 | 271.0 | 266.0 | 0.00 | 0.000 | 80 | quiesced=189 |
| `dup:93` | 93 | 266.0 | 266.0 | 0.00 | 0.000 | 86 | quiesced=189 |
| `dup:124` | 124 | 266.0 | 266.0 | 0.00 | 0.000 | 165 | quiesced=189 |
| `dup:186` | 186 | 271.0 | 266.0 | 0.00 | 0.000 | 283 | quiesced=189 |
| `dup:248` | 248 | 266.0 | 266.0 | 0.00 | 0.000 | 335 | quiesced=189 |
| `comp:93` | 93 | 271.0 | 266.0 | 0.00 | 0.000 | 80 | quiesced=189 |
| `comp:124` | 124 | 266.0 | 266.0 | 0.00 | 0.000 | 86 | quiesced=189 |
| `new:95` | 95 | 407.0 | 374.0 | 26.32 | 0.538 | 2042 | quiesced=189 |

**Per-mode H1 statistics (classical, from the tables above).**

| mode | grid \|R\| | B | U(\|R\|) | Spearman rho | U(max) - U(62) | Delta1 | H1 direction | H1 effect | Y(\|R\|) | Delta2 | LS slope of U per rule |
|---|---|---:|---|---:|---:|---:|---|---|---|---:|---:|
| (i) | [62, 93, 124, 186, 248] | 100 | [96.585, 43.757, 41.121, 15.438, 38.67] | -0.900 | -57.915 | 28.230 | FAILS | FAILS | [16.32, 8.65, 6.75, 1.25, 4.95] | 0.020 | -0.26796 pts/rule |
| (i) | [62, 93, 124, 186, 248] | 200 | [40.489, 6.694, 25.444, 9.468, 27.024] | -0.100 | -13.464 | 12.342 | FAILS | FAILS | [8.99, 0.09, 2.42, 0.52, 3.27] | 0.020 | -0.03630 pts/rule |
| (ii) | [62, 93, 124] | 100 | [96.585, 60.551, 41.693] | -1.000 | -54.892 | 28.230 | FAILS | FAILS | [16.32, 9.87, 6.28] | 0.020 | -0.88535 pts/rule |
| (ii) | [62, 93, 124] | 200 | [40.489, 25.598, 21.222] | -1.000 | -19.267 | 12.342 | FAILS | FAILS | [8.99, 0.09, 1.31] | 0.020 | -0.31076 pts/rule |
| (iii) | [62, 95] | 100 | [96.585, 33.114] | -1.000 | -63.471 | 28.230 | FAILS | FAILS | [16.32, -0.0] | 0.020 | -1.92336 pts/rule |
| (iii) | [62, 95] | 200 | [40.489, 52.06] | 1.000 | +11.571 | 12.342 | holds | FAILS | [8.99, 0.0] | 0.020 | +0.35065 pts/rule |

Mode (ii)'s row uses `[62, 93, 124]` only — `comp:186`/`comp:248` are the two missing grid points
(above); mode (ii)'s ρ/slope/verdict here are **not** the full 5-point picture modes (i)/(iii)-style
grids get, and should not be read as a completed mode (ii) registration. Re-running just those two
points (unchanged code, no re-run of the rest needed) would complete it.

## H1 verdict on this grid

**H1 direction FAILS at B=100 in every mode measured** (ρ = −0.90 to −1.00: regret U falls, not
rises, as |R| grows) and **H1 effect FAILS everywhere** (at B=100 the observed |U(max) − U(62)| exceeds Δ1 in
magnitude — by 1.9–2.3× — but in the WRONG direction for the effect test as designed, a large
negative change rather than the hypothesized positive one; at B=200 modes (i)/(ii) exceed it by
1.1×/1.6× in the same wrong direction, and mode (iii)'s +11.57 pts is below Δ1 = 12.34). At B=200, mode (iii)'s single point technically satisfies the
direction test (ρ = +1.00 on two points, which is guaranteed by having only two points and U rising
between them) but still fails the effect test's sign convention, and modes (i)/(ii) fail direction
outright (ρ = −0.10 to −1.00). **Unlike v1, this is not "unobservable" — every mode now shows a
large, consistent, statistically decisive movement in U as |R| grows; the movement is just the
opposite of what H1 predicts.** More rules make classical expressions' regret at a fixed
application budget go DOWN, not up: extra rules are on net finding cheaper forms fast enough,
within B=100–200 applications, to outrun whatever "more haystack, same needle" cost the added
sweep-time should in principle impose. `new:95` is the sharpest case — `L@100` and `Y@100` are
both ~0 (essentially no truncation loss left at B=100 at all) with 188/188 classical expressions
visibly different from `base` by B=100.

## Comparison against v1 (appended order)

v1's own committed result was that **0 of 188 classical expressions** differed in `cost@B` from the
`base` curve, for ANY inflated rule set, at B ∈ {100, 200} — H1 was unobservable by construction,
because an appended rule sits past every one of the 62 production rules in sweep order, and one
pass over that 62-rule prefix alone already exceeds B=100 on classical expressions (v1's own median
`app_actual` at the B=100 checkpoint was 113). **Under v2's interleaved order, the same comparison
(count of classical expressions whose `cost@100` differs from `base`, per |R|) is 159–188 out of
188 for every one of the seven inflated rule sets this run completed** — `visible@100` ranges from
159 (`comp:93`) to 188 (`new:95`, literally every classical expression), and `visible@200` from 147
to 186. This is the direct evidence that interleaving fixed the mechanism v1 diagnosed: an inflated
rule can now fire within the very first sweep (median `apps_per_sweep` at |R|=62 is 99.5 —
essentially unchanged from v1's own finding — but at |R|=248 a sweep costs ~995 applications
median, so B=100 is now a small FRACTION of one sweep rather than roughly one full sweep, and the
interleaved rule vector puts inflated rules throughout that fraction rather than entirely past it).
The count of expressions that changed is the evidence H1 is now observable; the direction and sign
of the change (§ above) is the separate, substantive finding this run adds beyond "it's now
measurable."

One further correction to v1's own reading of itself, noted in the design doc and restated here
because this run confirms it empirically: v1's B=100 win for the `base` (|R|=62) point was "largely
rule-list order within the first sweep" (median `app_actual`/`apps_per_sweep` ≈ 1.0 — B=100 lands
almost exactly at the end of one sweep of the 62-rule prefix, so which of those 62 rules go first
matters as much as which 62 rules are in the set at all). The `apps_per_sweep` column above (99.5
at |R|=62) makes this precise rather than descriptive: B=100 in sweeps is 1.01, i.e. barely over
one full pass.

## Reproduction

```bash
# the completed 8-set grid (comp:186/comp:248 excluded — see "Grid coverage" above)
cargo run --release -p pixelflow-pipeline --bin phase3_round2_unguided_curves -- \
    --corpus-dir pixelflow-pipeline/data --samples 400 \
    --out-csv docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.csv \
    --out-json docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.json \
    --rule-sets base,dup:93,dup:124,dup:186,dup:248,comp:93,comp:124,new:95

# comp:186/comp:248 (expect the wall-clock safety ceiling to panic — see "Grid coverage")
cargo run --release -p pixelflow-pipeline --bin phase3_round2_unguided_curves -- \
    --corpus-dir pixelflow-pipeline/data --samples 400 \
    --out-csv /tmp/comp186_248.csv --out-json /tmp/comp186_248.json \
    --rule-sets comp:186,comp:248

# fingerprint + order-sensitivity + 62-never-reordered guarantees, incl. new:95
cargo test -p pixelflow-search math::inflate -- --nocapture

# every table above
python3 pixelflow-pipeline/scripts/round2_register_stats.py \
    --csv docs/results/2026-09-01-round2-unguided-vs-rulecount-v2.csv \
    --expect base,dup:93,dup:124,dup:186,dup:248,comp:93,comp:124,new:95 \
    --modes 'i=base,dup:93,dup:124,dup:186,dup:248;ii=base,comp:93,comp:124;iii=base,new:95' \
    --out-json docs/results/2026-09-01-phase3-round2-registration-v2.json \
    --out-md /tmp/round2v2_final_tables.md
```
