> **Retracted/Superseded (2026-09-07), ledger L033, L034.** The registered constants B = 100 / 200 and Y = 16.3% / 9.0% were derived from tree-cost truncation losses on a synthetic corpus, in a regime 85x below where production stops (median 8,446 applications on real kernels); they port as-is and are not re-derivable, so this registration is closed rather than re-run. Verdict and rationale: `docs/results/2026-09-07-claims-ledger.md` (PR #1207); the corrected benchmark and re-validation order: `docs/plans/2026-09-07-benchmark-correction.md`.

# Phase 3 pre-registration: budget tiers, improvement threshold, and gates

**Date:** 2026-09-01
**Status:** REGISTERED — committed before any Guide training exists, per
`docs/plans/2026-08-31-guide-design-revision.md` §5's requirement that B and Y be fixed from
unguided data only, never tuned after seeing a guided result.
**Authority:** `docs/plans/2026-08-31-guide-design-revision.md` (§0 framing, §5 experiment).
**Data sources (unguided only):**
- `phase3_unguided_baseline` run of 2026-09-01 —
  `docs/results/2026-09-01-phase3-unguided-baseline.{csv,json}` (400 curves, TRAIN+DEV).
- Recalibrated `oracle_filtered_budget_curves` pre-flight of 2026-09-01 —
  `docs/results/2026-09-01-oracle-filtered-budget-curves.csv` (context only; no number from it
  sets B or Y).

Nothing in this document may be revised after the first Guide training run except to append
results against the gates below. If the experiment needs different parameters, that is a new
registration, recorded as superseding this one, with the change and its reason stated.

## 1. Environment (fixed)

| Item | Value |
|---|---|
| Corpus | `gen_bench_corpus --target 4000 --seed 42` (defaults otherwise); 4,272 unique expressions: TRAIN 3,359 + DEV 784 (TRAIN+DEV 4,143) + FINAL 129, split by generator family per the manifest. Regeneration verified byte-for-byte reproducible (same MD5s across two independent runs). |
| Corpus identity (MD5) | train `0ed6cf16abcbc006cd7a3ee2365b15b4`, dev `3026133ebba066eeca10f658da554400`, final `810a3e0e32d14d6ed2397c66f696cbae` |
| FINAL discipline | `corpus_final.bin` was NOT opened for any measurement in this registration and stays untouched until a publication run. DEV drives every selection decision. |
| Cost model | `CostModel::latency_prior()` — deterministic static per-op table. No wall-clock timing anywhere in training or evaluation. |
| Work axis | Cumulative **recorded rule applications**, exactly as provenance records them, idempotent re-fires included (91% of recorded applications commit nothing; a Guide pays to score a candidate whether or not it commits, so the honest axis counts them). Checkpoints crossed at rule-sweep granularity; analysis plots against the actual count (`app_actual`), never the target. |
| Checkpoint grid | `APP_CHECKPOINT_GRID` = 25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800 (`pixelflow-search/src/egraph/anytime.rs`) |
| Class cap | The production tier's fixed memory-protection cap (`config_for_node_count`), identical across any curves being compared — environment, not work. |
| Curve runner | `pixelflow_search::egraph::anytime::run_anytime_curve` — the ONE definition of "cost at budget B"; baseline, pre-flight, and any future guided run all import it. Wall-clock exists only as a per-curve safety ceiling that panics if it binds. |
| Tier bands | blitz ≤ 10 nodes, rapid 11–50, classical > 50 (same thresholds as the scoping round). |
| Source rev | Infrastructure committed at `3afb9dee` / `fc3fb32b` on `claude/phase3-guide`. |

Anytime cost is **not guaranteed monotone** along a curve: `extract_dag` is a heuristic DAG
extraction, and growing the graph can change its answer in either direction (observed: classical
mean first-to-final "gap" is negative in the pre-flight because a few curves get worse before
they get better). This is why the regret reference below is the empirical best over all
checkpoints of both curves, never the final state.

## 2. Metric definition (verbatim from the design revision, §5)

> **Deterministic cost-regret metric, no timing**: regret = `(guided_cost - reference_cost) /
> reference_cost` at each budget checkpoint, where `reference_cost` is the best cost either
> guided or unguided reaches at *any* checkpoint for that expression (the same
> empirical-reference convention `oracle_filtered_budget_curves` already uses, once its
> checkpoint grid is fixed per §2.3).

The checkpoint grid is now fixed (§1 above); this registration instantiates that metric at the
budget tiers in §4.

## 3. Measured unguided baseline (the tables B and Y are fixed from)

400 TRAIN+DEV expressions, size-stratified over the full 4,143-expression population
(stride 10.36): blitz n=23, rapid n=189, classical n=188.

### 3.1 How unguided runs end (applications to quiescence/cap)

| Scope | n | quiesced | class_cap | grid_exhausted | ended-at-apps p25 / median / p75 / p95 / max |
|---|---:|---:|---:|---:|---|
| ALL | 400 | 399 | 0 | 1 | 75 / 259 / 2,626 / 30,866 / 206,648 |
| blitz | 23 | 23 | 0 | 0 | 10 / 13 / 30 / 72 / 85 |
| rapid | 189 | 189 | 0 | 0 | 44 / 80 / 143 / 272 / 10,423 |
| classical | 188 | 187 | 0 | 1 | 532 / 2,686 / 14,548 / 44,412 / 206,648 |

### 3.2 Truncation loss: unguided-at-B vs unguided-at-4B

loss% = `(cost@B − cost@4B) / cost@4B × 100`, per expression; `live@B` = run ended strictly
after B applications; `>0 cnt` = expressions with any positive loss. Full tables (all B, p90,
max, mean, B/2 column) in `docs/results/2026-09-01-phase3-unguided-baseline.json`; the rows
that matter for tier selection:

**classical (n=188):**

| B | live@B | median% | p90% | >0 cnt | B/2 median% |
|---:|---:|---:|---:|---:|---:|
| 50 | 188 | 0.51 | 19,386.8 | 171 | 0.53 |
| **100** | **187** | **48.47** | **59,087.7** | **175** | **48.95** |
| **200** | **186** | **21.92** | **41,604.3** | **131** | **80.23** |
| 400 | 158 | 0.00 | 134.9 | 87 | 33.73 |
| 800 | 127 | 0.00 | 6.1 | 52 | 0.00 |
| 1600 | 109 | 0.00 | 0.09 | 24 | 0.00 |

**rapid (n=189):** median loss is 0.000 at every grid B; the largest positive share is 34/189
(18%) at B=50 with p90 = 2.6%. **blitz (n=23):** zero loss everywhere (median time to
quiescence is 13 applications — below the first grid point).

### 3.3 Interpretation (binding for tier selection)

Truncation demonstrably loses quality **only in the classical band**, where it loses a lot:
at B=100 the median classical expression is 48.5% worse than its own 4B state and 93% of
classical expressions show positive loss; at B=200 the median is 21.9% with 70% positive.
By B=800 the median loss is zero everywhere and the experiment would be unfalsifiable at the
median. On blitz and rapid the anticipated shallow-kernel result holds: production-scale
budgets already reach 4B quality, and **the Guide has nothing to buy back there** — this is
the per-band version of the design revision's stop-the-presses condition, and it stops the
presses only for those bands, not for the program: the classical band carries large,
falsifiable headroom.

## 4. Registered budget tiers B (per band)

| Band | Registered B | Rationale |
|---|---|---|
| **classical** | **B = 100 (primary), B = 200 (secondary)** | The two grid points with large median truncation loss and ≥186/188 curves still live at B. B=100 ≈ 3.7% of the median classical run's total work (2,686 apps); B=200 ≈ 7.4%. |
| rapid | none registered | Median truncation loss 0.000 at every feasible B; a median-based claim is unfalsifiable. Reported, not tested. |
| blitz | none registered | Zero truncation loss anywhere; median quiescence at 13 applications is below the first grid point. Reported, not tested. |

The guided experiment runs **only on the classical band**. Rapid/blitz curves may be reported
for completeness but no claim is registered on them.

## 5. Pre-committed improvement threshold Y

Fixed from §3.2's unguided numbers by the rule the design revision names ("Y = half the
truncation loss at B"), converted to a guided-vs-unguided-at-B ratio: if the median unguided
truncation loss at B is L, a Guide that closes half the gap reaches `(1 + L/2)·cost@4B`
against unguided's `(1 + L)·cost@4B`, i.e. an improvement fraction `Y = 1 − (1 + L/2)/(1 + L)`.

| B (classical) | median L (measured) | **Y (registered)** | Equivalent gap-vs-4B target |
|---:|---:|---:|---:|
| 100 | 48.47% | **16.3%** | median guided@100 gap vs unguided@400 ≤ 24.2% |
| 200 | 21.92% | **9.0%** | median guided@200 gap vs unguided@800 ≤ 11.0% |

Y was computed from unguided data only, before any Guide existed, and is not adjustable after
guided results exist.

## 6. Registered claim

At each registered (band, B): a Guide-ordered saturation, given the same rule library, the
same class cap, and a budget of B recorded applications, reaches extraction cost such that

1. **Y-clause:** the median over per-expression ratios
   `guided_cost@B / unguided_cost@B` is ≤ **1 − Y** (Y from §5), and
2. **4B-approach clause:** the median over per-expression gaps
   `(guided_cost@B − unguided_cost@4B) / unguided_cost@4B` is ≤ **L/2** (the "approaches
   unguided-at-4B" half of §5's claim, made concrete: the Guide closes at least half the
   median truncation gap).

Costs at B are read at the same grid checkpoint semantics as the baseline (sample at the first
between-sweeps point with cumulative applications ≥ B; both curves plotted against actual
counts). Zero-cost references follow the established convention: a positive cost against a
zero-cost reference is infinite loss, never 0%.

## 7. Gates

**Accept gate (publication):** the registered claim (§6) holds at B=100 with Y=16.3% on the
**FINAL** tier's classical band (family-held-out; FINAL is opened for the first time at that
run), reported with the full per-expression distribution (median, quartiles, p90, per-expression
CSV) — never a single geomean or a single median without its distribution. The B=200/Y=9.0%
result is reported alongside as the secondary claim. The accept run must state n for FINAL's
classical band; if n < 30 the result is reported as underpowered and no accept is claimed.

**Kill gate:** if after **5 clean training rounds** (mirroring the extraction-head program's
kill/pivot gate, §4.3 of `docs/plans/2026-08-05-egraph-nnue-research-workflow.md`; clean =
current label source per the design revision's §3 option 3, family-held-out split discipline
enforced, corpus at ≥ this registration's scale) the Guide cannot beat unguided-at-B **at all**
on DEV's classical band (median per-expression ratio < 1.0 at either registered B, i.e. any
improvement, far short of Y), stop and record it. "Guided can't match full saturation even
greedily" was always the redesign plan's Phase 3 gate; this makes it numeric. No unbounded
iteration.

**Honest fallback (pre-registered):** if the kill gate fires, the deliverable is the measured
economics per the design revision §5's fallback paragraph — the incrementality result, the
headroom-bound divergence by rule class, the candidate-deduplication requirement, and this
registration's own finding that truncation loss is a deep-band phenomenon (production budgets
already saturate blitz/rapid). A rigorous negative at this budget scale is a publishable
result.

**Training protocol constraints (restated, binding):** supervised on hindsight labels only —
no RL, no critic, no REINFORCE path, categorically. Cold-start first checkpoint. Label source
per the design revision §3 (option 3: strict-label cold start, tightened-labeler refinement);
a mid-program label-source swap re-validates against these same gates, it does not move them.

## 8. Pre-flight context (non-binding, recorded for honesty)

The recalibrated `oracle_filtered_budget_curves` run (2026-09-01, 225 expressions, same grid)
resolves the 2026-08-30 null: 99/100 classical curves now show shape (the old fraction grid
sampled after 97.8% of runs had already ended). Its second finding is a warning this
registration takes seriously but does not gate on: **rule-granularity oracle filtering barely
moves the curve** (classical median regret at B=100: 85.4% oracle-filtered vs 94.9% unguided;
identical from B=200 up at the median). Keeping only hindsight-load-bearing *rules* is not
where the buy-back is; the Guide must discriminate at **candidate granularity** (which match
of which rule at which e-class, per the design revision §4's candidate-local feature argument
and the §2.2 dedup finding), or it will have nothing to show at these budgets. This is
context for Guide design, not part of the claim.

## 9. Results appended against the gates (DEV, round 1, 2026-09-01)

Appended per this document's own rule (results only; nothing above is revised). Full report:
`docs/results/2026-09-01-phase3-at-budget-eval.md` (per-expression rows in the `.jsonl`,
generated tables in `-report.md`). All 334 DEV classical expressions; Guide =
`guide_checkpoint_strict_v1.json` (strict-v1 labels, cold start); control = `PerRuleRateGuide`.

| Tier | Registered | Measured (linear Guide, median per-expression) | Verdict |
|---|---|---|---|
| classical B=100 | ratio ≤ 0.837 (Y=16.3%); gap vs 4B ≤ 24.2% | ratio **0.537** (323 improved / 4 unchanged / 7 worse); gap vs unguided@400 **0.00%** | both clauses hold on DEV |
| classical B=200 | ratio ≤ 0.910 (Y=9.0%); gap vs 4B ≤ 11.0% | ratio **0.696** (245 / 71 / 18); gap vs unguided@800 **0.00%** | both clauses hold on DEV |

Control arm (per-rule TRAIN rates, no candidate-local information): median ratio 0.565 / 0.699 —
most of the effect; the linear model wins the head-to-head ~2:1 where they differ and halves the
p90 regret. Kill gate: not fired (round 1 of 5 clean rounds). Accept gate: not yet run (FINAL
untouched). Production context: classical production stops at a median 1,671 applications
(q1 400, q3 9,441; 314 quiesced / 20 timeout), i.e. beyond 4B on most expressions — the
registered budget is not the regime production runs in.
