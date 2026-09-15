> **Retracted/Superseded (2026-09-07), ledger L035.** The registered claim is judged here in tree cost (pre-#1192) at a budget production never runs in; the DAG reading is a smaller DEV win and a loss on every structured family. Verdict and rationale: `docs/results/2026-09-07-claims-ledger.md` (PR #1207); the corrected benchmark and re-validation order: `docs/plans/2026-09-07-benchmark-correction.md`.

# Phase 3 at-budget evaluation on DEV: the registered claim holds, and per-rule base rates carry most of it (2026-09-01)

> **This page's numbers predate the 2026-09-02 review fixes and a re-run is required.**
> Six corrections landed after this run and each of them changes numbers here: the guided
> loop now applies every node-level match sharing one candidate key and re-resolves a
> matched node by stable identity before applying it; `derivation_ancestors_tight`
> canonicalizes class ids and takes the extraction's complete choice map (the enabler
> diagnostic); `train_guide` excludes `dedup_repeat` rows, sizes its rule table from the
> registered rule set, and applies L2 to every weight — so the checkpoint the guided arms
> deploy is not the one measured here; and `average_precision` now groups tied scores, which
> moves the reported PR-AUCs. The verdict below is recorded as it was reproduced; nothing in
> it has been edited to match the new code. Re-run before citing any figure.

> **Instrument change (2026-09-02 forward-port).** Two things about how a
> Phase 3 anytime curve is measured changed with the port, so a re-run does
> not reproduce the numbers below even if nothing else changed: the
> application budget now binds **mid-scan** rather than between rule sweeps
> (`app_actual == app_target` exactly, no overshoot), and the reported cost
> is the **DAG** cost the emitted kernel pays rather than the extraction DP's
> tree total (#1117). Full statement:
> [docs/results/2026-09-02-phase3-instrument-changes.md](2026-09-02-phase3-instrument-changes.md).

Reproduce:
```
# Prerequisites — the trained checkpoint and the strict-label splits are NOT committed
# (`pixelflow-pipeline/data/*` is gitignored); regenerate them first, in this order:
cargo run --release -p pixelflow-pipeline --features training --bin gen_bench_corpus -- \
    --target 4000 --seed 42
cargo run --release -p pixelflow-pipeline --features training --bin gen_strict_labels
cargo run --release -p pixelflow-pipeline --features training --bin train_guide -- \
    --report-json docs/results/2026-09-01-train-guide-report.json
# ...then the evaluation itself:
cargo run --release -p pixelflow-pipeline --features training --bin phase3_at_budget_eval -- \
    --classical-samples 0 --other-samples 30
# per-expression rows: docs/results/2026-09-01-phase3-at-budget-eval.jsonl
# aggregates:          docs/results/2026-09-01-phase3-at-budget-eval.json
# generated tables:    docs/results/2026-09-01-phase3-at-budget-eval-report.md
```
Every step is seeded and deterministic, so a clean checkout reproduces the same artifacts.
`phase3_at_budget_eval` writes a `…-eval.config.json` fingerprint beside its JSONL and
refuses to resume into rows written under a different guided grid, checkpoint, control
report, or sampling configuration.

This is the §5 experiment of `docs/plans/2026-08-31-guide-design-revision.md`, run against the
committed registration `docs/plans/2026-09-01-phase3-registration.md` (nothing in it was revised;
a results section is appended to it per its own rule). **DEV only** — the tier every selection
decision is allowed to touch; `corpus_final.bin` was not opened. Every number below is a
deterministic `CostModel::latency_prior()` extraction cost or a count; no wall-clock appears in
any metric.

- Corpus: `corpus_dev.bin` from `gen_bench_corpus --target 4000 --seed 42` (784 expressions:
  blitz 55, rapid 395, classical 334). **All 334 DEV classical expressions** were evaluated (the
  registered band); 30 rapid and 30 blitz, size-stratified, are reported for completeness with
  no claim.
- Guide under test: `pixelflow-pipeline/data/guide_checkpoint_strict_v1.json` — the cold-start
  linear Guide trained on strict-v1 hindsight labels (TRAIN families only; DEV AUC 0.9902,
  PR-AUC 0.4595, `docs/results/2026-09-01-train-guide-report.md`). Control: `PerRuleRateGuide`
  built from the same training run's per-rule TRAIN strict-positive rates.
- Source rev: `c3c51758` on `claude/phase3-guide` (harness `phase3_at_budget_eval`, curve
  plumbing in `pixelflow-search/src/egraph/{anytime,saturate}.rs`).

## The ladder

Every arm goes through the ONE anytime-curve definition
(`egraph::anytime::run_anytime_curve_with`); guided arms advance the graph with
`GuidedSaturation` (dedup set carried across checkpoints as one episode), the unguided arm with
`EGraph::saturate_until_applications`. Cost at B is read at the grid checkpoint for B. Regret is
against the empirical best cost any arm reaches at any checkpoint of that expression. Ratios and
gaps are per-expression; the table shows q1 / median / q3 (p90). Zero-cost references follow the
established convention (positive cost against a zero reference is infinite, never 0%); no such
case occurred in the ratio columns.

The strict-oracle arm (e) was **not run**: the strict label lives on one specific run's
`ApplicationId`s, and transporting it onto a differently-ordered guided run's candidates needs a
`(rule, class content at firing time) → label` map the provenance log does not record
(`ApplicationRecord::match_root` is a class id that union/rebuild renumbers). Building that map
means instrumenting the unguided sweep itself, not a cheap replay.

### classical, B = 100 (primary; registered Y = 16.3% → median ratio ≤ 0.837; 4B-approach: median gap ≤ 24.2%), n = 334

| arm | ratio vs (a) unguided@B | improved / unchanged / worse | gap vs (b) unguided@4B | regret vs empirical best | structural share of apps @B | strict precision @B |
|---|---|---|---|---|---|---|
| (a) unguided @B | 1 (by definition) | — | 0.39% / **51.66%** / 178.6% (p90 4.8e4%) | 54.2% / **96.2%** / 2.8e4% (p90 7.6e4%) | 0.589 | 0.021 |
| (b) unguided @4B | — | — | 0 | 0.00% / **1.10%** / 57.5% (p90 2.2e4%) | — | — |
| (c) PerRuleRateGuide @B [control] | 0.007 / **0.565** / 0.701 (p90 0.841) | 321 / 4 / 9 | −28.7% / **0.00%** / 0.21% (p90 1.0%) | 0.00% / **0.64%** / 5.63% (p90 24.7%) | 0.279 | 0.225 |
| (d) LinearCandidateGuide @B [claim] | 0.004 / **0.537** / 0.681 (p90 0.797) | 323 / 4 / 7 | −31.3% / **0.00%** / 0.00% (p90 0.9%) | 0.00% / **0.38%** / 1.97% (p90 10.2%) | 0.322 | 0.224 |

Head-to-head at B=100: (d) < (c) on 131 expressions, equal on 134, (d) > (c) on 69.

### classical, B = 200 (secondary; registered Y = 9.0% → median ratio ≤ 0.910; 4B-approach: median gap ≤ 11.0%), n = 334

| arm | ratio vs (a) unguided@B | improved / unchanged / worse | gap vs (b) unguided@4B | regret vs empirical best | structural share of apps @B | strict precision @B |
|---|---|---|---|---|---|---|
| (a) unguided @B | 1 (by definition) | — | 0.00% / **16.89%** / 78.9% (p90 2.9e4%) | 0.70% / **51.4%** / 176.9% (p90 5.1e4%) | 0.596 | 0.030 |
| (b) unguided @4B | — | — | 0 | 0.00% / **0.00%** / 5.66% (p90 1.8e4%) | — | — |
| (c) PerRuleRateGuide @B [control] | 0.522 / **0.699** / 1.000 (p90 1.000) | 245 / 71 / 18 | 0.00% / **0.00%** / 0.23% (p90 2.8%) | 0.00% / **0.20%** / 2.85% (p90 16.5%) | 0.403 | 0.142 |
| (d) LinearCandidateGuide @B [claim] | 0.512 / **0.696** / 1.000 (p90 1.000) | 245 / 71 / 18 | −0.27% / **0.00%** / 0.09% (p90 0.4%) | 0.00% / **0.08%** / 0.44% (p90 5.8%) | 0.430 | 0.146 |

Head-to-head at B=200: (d) < (c) on 114, equal on 202, (d) > (c) on 18.

The unguided truncation loss reproduces the registration's baseline on this larger, DEV-only
sample (median 51.7% at B=100 vs 48.5% registered; 16.9% at B=200 vs 21.9%), so the band the
claim was registered on is the band that was measured.

### rapid and blitz (no claim registered; reported for completeness)

30 expressions each. Unguided truncation loss is 0.000 at the median in both bands and both
tiers, as the registration predicted; (c) and (d) have median ratio 1.000 with regret 0 at the
median and p90 (rapid B=100: 2 improved / 28 unchanged / 0 worse for both arms; B=200: 0 / 28 /
2; blitz: 30 unchanged everywhere). There is nothing to buy back in these bands, and the Guide
neither buys nor loses anything there.

## The answer

**B = 100 (primary): the registered claim HOLDS on DEV — the linear Guide's median
per-expression cost ratio against unguided-at-100 is 0.537 against the registered ≤ 0.837
(Y = 16.3%), improving 323 of 334 classical expressions (4 unchanged, 7 worse), and its median gap
to unguided-at-400 is 0.00% against the registered ≤ 24.2% — it does not merely approach
unguided-at-4B, its median regret (0.38%) is below unguided-at-4B's (1.10%).**

**B = 200 (secondary): the registered claim HOLDS on DEV — median ratio 0.696 against ≤ 0.910
(Y = 9.0%), 245 improved / 71 unchanged / 18 worse, median gap to unguided-at-800 0.00% against
≤ 11.0%.**

The kill gate (median ratio < 1.0 at either registered B) is nowhere near firing; round 1 is a
clean training round.

**The ladder question: (d) beats (c), but narrowly, and (c) carries most of the effect.** At B=100
the control arm — a per-rule lookup table with no candidate-local information — already reaches a
median ratio of 0.565 against the linear model's 0.537, and at B=200 they are 0.699 vs 0.696.
Where the two differ, the linear model wins about 2:1 (131 vs 69 at B=100, 114 vs 18 at B=200),
and the difference is concentrated in the tail: p90 regret 10.2% vs 24.7% at B=100, 5.8% vs 16.5%
at B=200; the linear arm reaches the empirical best on 199 expressions to the control's 189. So
the honest reading of the label semantics is: **per-rule strict-positive base rates plus
candidate-key deduplication produce nearly all of the buy-back; the candidate-local features
(neighborhood ops, budget state, matched-class size) earn a modest, consistent improvement in the
tail and have not yet earned their place at the median.** That is consistent with the ranking
metrics from training (control-arm AUC 0.937 vs 0.990 — a real gap, but one that mostly reorders
within-rule) and with the pre-flight finding that rule-granularity oracle filtering barely moved
the curve: what moved the curve here is not rule *filtering* but rule *ordering* plus the removal
of the 91% idempotent re-fire traffic the unguided sweep spends its budget on.

## Where the budget goes (diagnosis)

**Enabler starvation is not present.** The concern was that a Guide trained on strict labels,
which give commutative/associative/distribute 0.0% credit, would learn never to fire structural
rules and thereby never build the classes the numeric rules need. Measured:

- Structural share of the first B applications: unguided 0.589 / 0.596 (B=100 / 200) — the
  unguided sweep spends most of its budget re-firing `commutative` (23,880 of its first-100
  applications across the band) and `involution` (9,981); the guided arms spend 0.28–0.32 at
  B=100 and 0.40–0.43 at B=200 on structural rules. They *do* fire them — `commutative` is the
  guided arms' second most-fired rule (6,042 / 6,149) — later in the order and only once per
  `(rule, class content)` key.
- Of unguided's 7,588 numeric strict-positive applications across the 334 classical
  expressions, 6,290 (82.9%) have a structural application in their tight derivation ancestry
  (the "quiet move before the tactic"); the guided arms' final e-graphs contain 83.6–83.9% of all
  those numeric terms and **80.3–80.6% of the structurally-enabled ones** — the same share as for
  the un-enabled ones. Guided ordering reaches the enabling classes; it just does not pay for
  them 24,000 times.
- Strict precision (share of the first B applications that end up on the arm's own final
  extracted path): unguided 2.1% / 3.0%, guided 22% / 14%. Ten times fewer wasted applications
  per useful one at B=100.

**Where the residual regret lives.** Guided runs proceed to a median 294 applications (q1 132,
q3 750; 75–78 of 334 reached the 800-application grid end still live) and reach the empirical
best on 189 (control) / 199 (linear) expressions to unguided's 299. The guided arms' quiescence is
dedup exhaustion — every `(rule, class content)` key resolved once — which is a strictly smaller
closure than the unguided sweep's, so on ~40% of expressions the guided arm's final cost is above
what full unguided saturation finds (median regret at the guided arms' own 800-application grid
end: 0.00%, but a heavy tail — 33–35 expressions above 2× the best; p90 52% for the control
arm and 8.5e3% for the linear arm, the latter a handful of expressions whose best cost is tiny). That is the stage-2 lever, not a bigger model: the tail is the set of expressions where a
single firing per key is not enough and a second pass (or the tightened-label credit for the
enabler) would be.

**A plumbing defect found and fixed mid-evaluation, disclosed.** The first full run of this
harness showed guided runs "quiescing" at a median 56 (control) / 76 (linear) applications, with
a median 39–44% of scored candidate keys never becoming a recorded application, and a
grid-sensitivity check (guided grid `100,200,400,800` vs the default `25,50,100,…`) showed the
coarse grid producing a *better* cost@100 on 26/40 (control) and 30/40 (linear) expressions —
guided-at-B depended on which earlier checkpoints had been sampled. Cause: `GuidedSaturation`
marked every scored survivor's key as seen at enumeration time, so survivors the round never got
to (after a mid-round budget stop at each checkpoint, or with a `node_idx` staled by an earlier
application in the same round) were deduped away for good. Fixed in `0b8a7a1e` (a key becomes
seen only when an application is actually recorded for it; regression test pins the budget-stop
case), the whole evaluation was re-run, and the grid check is now a wash (3/18/19 and 10/14/16
better/same/worse). This is a change to the evaluation machinery's fairness, not to the Guide, the
labels, B, or Y; **the verdict is the same under both runs** (pre-fix: linear median ratio 0.546 /
0.741, control 0.573 / 0.744, both clauses holding at both tiers), and both are recorded here so
the fix cannot be mistaken for tuning. DEV is the tier where such a discovery is allowed to
happen; FINAL remains untouched.

## Production-units context (integration audit, PR #1079)

Production saturates with `config_for_node_count` (classical: 100 rounds / 5,000 classes /
200 ms) and has no application counter, so the registered B is a proxy for a machine production
does not literally run. For every evaluated expression the harness also ran the exact production
saturation step — `pixelflow_search::runtime::production_saturation_probe`, which is the same
function body `optimize_runtime_arena` executes, refactored into one shared helper rather than
copied — and read its stop reason from the loop.

Classical (n = 334): **stop reasons quiesced 314, timeout 20**; effective B (recorded
applications at stop) **q1 400 / median 1,671 / q3 9,441 / p90 23,799 / max 85,900**; share of
expressions with ≥ 100 / 200 / 400 / 800 / 1,600 applications = 98.5% / 91.6% / 74.9% / 59.6% /
50.3%; rounds run median 5 (q3 8, p90 10) of the 100 allowed; classes at stop median 330 (p90
4,063) of the 5,000 cap. Production's cost equals the unguided curve's best on 302 of 334
expressions; its ratio to unguided cost@800 is 1.000 at the median (q1 0.996); to cost@100 it is
0.536. The unguided checkpoint production's cost is equivalent to: 200 (75 expressions), 400
(72), 800 (57), 1,600 (50), ≥ 3,200 (52); 26 expressions are at or below the 25-checkpoint (small
graphs that quiesce early). Rapid and blitz production runs stop at a median 49 and 12
applications respectively, all but one quiesced.

**Production's budget does not bind where the claim lives.** On the classical band production
runs to quiescence on 94% of expressions and spends a median ~17× the primary registered budget
doing it; it is effectively an unguided-at-4B-or-beyond machine, not an at-B one. The registered
claim is therefore an anytime property that production does not currently exercise: the same
Guide, at 100 applications (6% of production's median work), matches or beats production's cost
on 142 of 334 classical expressions (37 better, 105 equal) and at 200 on 178 (29 / 149). The
20 timeout stops (6% of classical) are machine-dependent — the host's load average was ~11 on 12
cores during the run (recorded in the report's context block) — and are the one place production's
wall-clock ceiling was observed to bind; they are a stop condition of that call, not a metric.
Nothing here changes the registered claim; it says what the claim does and does not speak to.

## What this round did not do

- No FINAL run. The accept gate is a FINAL-tier measurement (n ≥ 30 classical, else underpowered:
  FINAL has 129 expressions in total); DEV drove everything here and the FINAL run is a separate,
  deliberate publication step.
- No strict-oracle arm (above). The upper bound on candidate ordering is still unmeasured.
- No tightened-label Guide. Stage 2 of design decision #1 (option 3) is the next lever the
  diagnosis points at, for the tail where single-pass dedup closure loses to full saturation.

## Journal

Record `phase3_at_budget_eval` appended to `docs/results/journal.jsonl` (deterministic metrics
only: per-tier medians, thresholds, verdicts, production effective-B quartiles, output paths).
