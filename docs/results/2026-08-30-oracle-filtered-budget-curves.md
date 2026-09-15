> **Retracted/Superseded (2026-09-07), ledger L031.** The curves are inconclusive by the doc's own reading, sampled after 97.8% of runs had ended, with the class cap unscaled per checkpoint and costs in tree units, and are not evidence for or against a Guide. Verdict and rationale: `docs/results/2026-09-07-claims-ledger.md` (PR #1207); the corrected benchmark and re-validation order: `docs/plans/2026-09-07-benchmark-correction.md`.

# Oracle-filtered anytime budget curves: inconclusive as calibrated (2026-08-30)

> **SUPERSEDED (2026-09-02).** This report's `cost` and `regret_pct` columns are the
> extraction DP's **tree** cost, not the DAG cost the emitted kernel pays (#1117), and its
> checkpoint grid is denominated in sweep fractions rather than applications. Both defects
> are corrected in
> [docs/results/2026-09-02-oracle-filtered-budget-curves.md](2026-09-02-oracle-filtered-budget-curves.md),
> which supersedes this file. Nothing below has been edited; it is kept as measured.

Reproduce:
```
cargo run --release -p pixelflow-search --example oracle_filtered_budget_curves
```
Harness: `pixelflow-search/examples/oracle_filtered_budget_curves.rs` (889 lines, complete —
`fn main` closes cleanly, compiles clean under `cargo check`). Output: prints a summary to
stdout and writes a full per-expression/per-curve/per-work-fraction CSV to
`docs/results/2026-08-30-oracle-filtered-budget-curves.csv` (2,250 rows = 225 expressions x 2
curves x 5 work-fraction checkpoints, a complete cross product — 220 synthetic corpus items
across 10 complexity bands + 5 named realistic shaders).

**This report is written from the existing, already-complete CSV only.** A `bootstrap_extraction_head`
publication run was live in another worktree while this audit was in progress, with a
sentinel-guarded abort budget; an attempted re-run of this harness (to capture the console
summary text, which is not persisted to any file) caused a +307.7% regime-change abort in that
run before it was killed at 160/225 expressions. The CSV it would have overwritten was untouched
(the harness only writes output at the very end of `main`), so no data was lost, but this report
does not include the console-only diagnostics (per-tier quiesce-before-nominal-budget percentages,
the shader in-sample check) that were never captured to disk by whichever agent ran this
originally — only what the CSV itself contains. Do not re-run this harness while another
worktree's timed benchmark may be live; check `pgrep -fl "bootstrap_extraction_head|bench_extraction_3way"`
first.

**Revision note (2026-09-01):** an automated review of PR #1067 (`chatgpt-codex-connector`) found
four defects in this harness across two review passes, all now fixed:

1. *Safety timeout scoped per-iteration, not per-curve.* `run_anytime_curve` re-passed the same
   300s `Duration` to every single-iteration `saturate_with_limits` call, and that function starts
   its own `Instant::now()` clock on every call — so a `ceiling_iters`-long curve could in
   principle run for up to `ceiling_iters * 300s`, not the documented 300s ceiling. Fixed: one
   deadline is now computed per expression, and each call passes the *remaining* duration
   (fail-loud `assert!` if it's ever exhausted). Not re-run because it had no effect on the
   existing data: the full 225-expression run completed in ~5 minutes total, nowhere near any
   iteration individually approaching 300s.
2. *Zero-baseline regret could mask real regret.* `regret(cost)` returned `0.0` whenever
   `global_best == 0`, regardless of `cost` — so an expression whose curve simplifies to a free
   `Const`/`Var` only at a *later* checkpoint than an earlier positive-cost one would have that
   earlier regret silently erased. Fixed: a positive cost over a zero baseline now reports
   `f64::INFINITY`. Not re-run because it had no effect on the existing data either: as this
   report's own "checkpoint-grid miscalibration" finding establishes, every sampled expression's
   cost is flat across all five checkpoints (97.8% never advance past the first one at all) — the
   masking scenario requires cost to *vary* within an expression's checkpoint set, which does not
   happen anywhere in this dataset. This is exactly the failure mode a recalibrated,
   finer-grained re-run (§ "Design implications", point 1) would be vulnerable to if left unfixed.
3. *(P1) Class cap not scaled per checkpoint — this one DOES affect the existing data's
   interpretation.* `run_anytime_curve` capped growth at `ceiling_classes` (the FINAL, 3x-nominal
   class allowance) for every single-iteration call throughout the whole curve, including rounds
   working toward the smaller checkpoints (0.25x/0.5x/1.0x). Since iteration count doesn't bound
   class growth on its own (a rule sweep can pack far more class growth into few iterations than a
   tier's nominal budget intends), a checkpoint labeled "frac=1.0" could reflect an e-graph that
   had already grown to the 3x class allowance. Verified in the review against the committed CSV:
   **59 of 225 unguided rows exceed their tier's nominal class cap at frac=1.0**, some reaching
   close to 3x it — the "standardized budget" comparison this report's methodology section
   describes was not actually enforced for those rows. Fixed: each checkpoint now has its own
   class cap (`target_classes`, parallel to the existing `target_iters`), and growth is capped to
   the *current* unreached checkpoint's budget, growing only once that checkpoint is sampled and
   the next (larger) target's cap takes over. **Not re-run** (same compute-contention constraint
   as the rest of this report), so the existing CSV's frac<3.0 rows carry this confound as
   originally measured. This does not overturn this report's null/inconclusive finding — a
   *looser* class budget than intended could only have made it easier, not harder, for the two
   curves to reach the same cost, so it cannot be hiding a regret difference the tighter, corrected
   budget would show. It sharpens the same "checkpoint-grid miscalibration" diagnosis this report
   already reached: the existing run measured curves that were freer than their labels claimed,
   in both the iteration and (now confirmed) the class dimension, and the recalibrated re-run this
   report already recommends must respect both budgets, not just iteration count.
4. *(P2) The "over-approximation looseness" diagnostic over-claimed what it isolates.* The
   `credited_via_class_only` diagnostic (console-only, never captured into this report — see
   below) was documented as isolating `derivation_ancestors`'s class-membership over-approximation
   specifically, but its check (created node not literally on the chosen path) can't distinguish
   that from the other two named over-approximation axes, or from genuine transitive-enabler
   credit the labeler is correctly assigning. Fixed: renamed to `credited_non_direct` throughout,
   with the module doc and console output corrected to describe what it actually measures ("not
   the literal creator of a chosen node," not "class-membership over-approximation"). This
   diagnostic was never reported in this document (see the top-of-file note on what the CSV vs.
   console output contains), so no numbers here needed correcting.

## Methodology, from the harness's own module doc

The kernel language is strongly normalizing and this optimizer is budget-only by design:
saturation spends a budget and stops, full stop, and reaching quiescence
(`stats.total_unions == 0`) is a diagnostic condition, never a certified fixpoint. So there is no
privileged "full saturation" reference to call `C_full` and regret everything else against.
Instead, for each expression, the harness runs **two curves** — **unguided** (all 62 rules) and
**oracle-filtered** (only the rules a hindsight pass over the unguided run's own best-found
extraction marked load-bearing, filtered at whole-rule granularity, not per-application — see the
source's module doc for why match-level replay is not stable across runs) — from empty to a
generous ceiling (3x the expression's own production-tier iteration budget), sampling extraction
cost at standardized work fractions (0.25x/0.5x/1x/2x/3x of that per-tier nominal budget) along
the way. Regret at any checkpoint is `(cost - best_cost_ever_seen) / best_cost_ever_seen`, where
"best ever seen" is the minimum cost either curve reaches at any checkpoint for that expression —
an empirical reference, never a claimed optimum.

## Headline: no regret was observed anywhere in this run — and the reason why is itself the finding

**All 2,250 rows report `regret_pct = 0.0000`.** For every one of the 225 expressions, the
`unguided` and `oracle` curves reach *identical* extraction cost at *every* one of the five
checkpoints. This is not the intended positive result ("oracle-filtered keeps pace with
unguided, cheaply") landing cleanly — it is the checkpoint grid never getting a chance to see a
difference, for two distinct, corpus-measurable reasons:

| Cause | Share of the 225 expressions | What it looks like in the CSV |
|---|---:|---|
| Saturation reaches quiescence (or the safety class-count cap) **at or before the first checkpoint** (25% of nominal budget) | **97.8%** (220/225) | `iteration`, `classes`, `applications`, `cost` are byte-identical across all 5 `frac` rows for that expression+curve |
| Saturation hits `EGraph::saturate_with_limits`'s class-count safety cap (~15,000 classes for the largest, "classical"-tier expressions) within single-digit-to-teens iterations, freezing the extraction for the rest of the curve | **4.9%** (11/225, all "classical" tier, node_count 1,255-8,511) | class count converges to 14,949-14,998 (a ~15k cap) by iteration 4-14, then stays fixed through frac=3.0 |
| Genuinely converges that fast (small/simple expressions, not capped) | remaining ~93% of the 97.8% | e.g. `shader_redundant` (13 nodes): quiesces at iteration 14, cost 17, unchanged through frac=3.0 |

Only 5/225 expressions (2.2%) even show the *iteration count itself* varying across checkpoints
(meaning the loop was still actively running between frac=0.25 and frac=3.0) — and even for those
5, `cost` is flat across every checkpoint (two are already at their minimum extractable cost from
the first sample; the other three are class-capped mid-run). In no case, for any expression at
any work level, does `unguided` cost ever differ from `oracle` cost.

## What this does and does not tell us

**It does not refute the Guide thesis.** Unguided and oracle never *disagreeing* is consistent
with oracle-filtering being "free" at these particular checkpoints, but the checkpoints tested
are all *past* the point where either curve stops changing — so this run cannot distinguish "the
Guide has no headroom to offer" from "this checkpoint grid is entirely on the flat part of both
curves." The `guide_headroom` measurement (a different harness, same session) independently shows
2.6x-34x headroom exists in *which applications were load-bearing*; that headroom is invisible
here because the coarsest checkpoint already comes after saturation has settled.

**It does not confirm the thesis either.** "Does oracle-filtered replay reach forms unguided
never finds" — the specific question this harness was written to answer — has no evidence either
way in this run: the two curves are identical at every sampled point, which could mean "no,
never" or could mean "the sampled points are all too coarse to see it." This measurement is
**inconclusive**, not negative.

**Root cause: the checkpoint grid (`FRACS` x `config_for_node_count`'s per-tier nominal
iteration budget) is calibrated far above what this corpus actually needs to settle.** The
smallest fraction tested, 25% of nominal, is already enough iterations for 97.8% of expressions
to fully quiesce or hit their safety cap. To see an anytime curve with actual shape, the next
attempt at this measurement should checkpoint at **absolute small iteration counts** (1, 2, 3, 4,
..., 20) rather than fractions of an already-generous nominal budget — the same lesson
`guide_headroom` reports independently (all 800 of its expressions quiesced before either budget
cap; saturation is fast relative to the budgets this codebase currently configures for it).
**This recalibration was not attempted in this round** — CPU-heavy re-runs of this harness are
deferred (see the note at the top about the concurrent publication benchmark), and a config-only
change without re-running would not produce trustworthy numbers to report.

## What is usable from this run: the oracle rule-set reduction itself

Independent of the (uninformative) regret numbers, the CSV does confirm the *magnitude* of rule
filtering a perfect oracle would apply, per tier (`rules_allowed` column, oracle curve):

| Tier | n | median oracle rules kept (of 62) |
|---|---:|---:|
| blitz (smallest, node_count 6-10) | 24 | 3 |
| rapid (node_count 11-47) | 101 | 7 |
| classical (node_count 53-8,511) | 100 | 24 |

This is directionally consistent with `guide_headroom`'s finding that larger expressions have
*more* rules earning "load-bearing" credit under the labeler bound (more distinct rewrite shapes
get exercised as expressions grow) even though the *fraction* of applications that matter shrinks.
It is not, on its own, evidence that filtering to this reduced set preserves quality at a smaller
budget — that is exactly the question the flat curves above failed to test.

## Design implications for Phase 3

1. **This measurement needs to be re-run with absolute-iteration checkpointing before it can
   speak to the anytime-curve question.** As calibrated, it is not evidence for or against the
   Guide thesis; treat it as a null result caused by checkpoint-grid miscalibration, not as
   "the Guide has nothing to offer." Recommend this as explicit follow-up work, not blocking Phase
   3's design (the design doc's go/no-go leans on `guide_headroom` and the saturation-delta
   measurement instead, both of which are decisive and use deterministic per-application counts
   rather than budget-fraction checkpoints). The re-run should also now enforce a **per-checkpoint
   class cap** (fixed in source this round, see "Revision note" item 3 above, not yet exercised by
   a re-run) — absolute-iteration checkpointing alone is not sufficient if class growth per round
   is left uncapped relative to each checkpoint's own tier budget.
2. **The oracle rule-count-kept table (3/7/24 median by tier) is usable now** as a rough sizing
   prior for how aggressively a Guide might restrict the candidate rule set at each tier, but
   should be corroborated once a properly-calibrated anytime run exists.
3. **Any future version of this harness should record "hit safety cap" as a distinct status from
   "quiesced,"** rather than treating both as the same `stats.total_unions == 0` diagnostic — this
   audit could only tell them apart by cross-referencing class counts against the ~15,000 cap after
   the fact; the harness itself does not currently distinguish them.

Full per-expression, per-curve, per-checkpoint data: `docs/results/2026-08-30-oracle-filtered-budget-curves.csv`.
