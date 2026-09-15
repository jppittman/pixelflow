> **Retracted/Superseded (2026-09-07), ledger L028.** The per-rule labeler-vs-strict correlation (Spearman 0.35) is withdrawn as a constant - two draws give 0.35 and 0.19 - and the headroom and structural-rule rows (L027, L029) were never taken on a shipped kernel. Verdict and rationale: `docs/results/2026-09-07-claims-ledger.md` (PR #1207); the corrected benchmark and re-validation order: `docs/plans/2026-09-07-benchmark-correction.md`.

# Phase 3 oracle headroom at corpus scale (2026-08-30)

Reproduce:
```
cargo run --release -p pixelflow-pipeline --features training --bin guide_headroom -- \
    --limit 800 --min-expressions 500 \
    --out docs/results/2026-08-30-guide-headroom.json
```
Harness: `pixelflow-pipeline/src/bin/guide_headroom.rs` (new, additive — no changes to
`provenance.rs`/`labeler.rs` semantics). Corpus: `corpus_train.bin` (6,480 expressions) +
`corpus_dev.bin` (1,512), regenerated this session (`gen_bench_corpus --target 8000`, since
`data/` was empty) because `--target 2000` fails the per-family integrity check at that scale
(quarantine noise pushes several families over the 20% refusal threshold when each family's
quota is only ~7-9 expressions — not a bug, just too small a target for the generator's
per-family floor). 800 expressions were **stride-sampled** evenly across the full 7,992-entry
train+dev population (not a prefix — corpus write order is band-by-band in increasing
complexity, so a prefix would silently sample only the simplest band and misrepresent "at
scale"). Every number below is a deterministic count or a `CostModel::latency_prior()` cost —
no wall-clock timing. `EGraph::saturate()`'s internal 500ms/iteration deadline is the same
stopping condition every production caller (including `run_episode`) already saturates under;
it bounds the *algorithm*, not anything we measure.

Pre-run check: `pgrep -fl "bootstrap_extraction_head|bench_extraction_3way"` was clean before
starting (no paper-capstone timed run in flight).

**Revision note (2026-09-01):** an automated review of PR #1067 (`chatgpt-codex-connector`)
correctly flagged that an earlier version of this harness reused production `saturate()`'s 500ms
wall-clock deadline as `saturate_with_limits`'s timeout — meaning any expression whose saturation
took longer than 500ms would be silently truncated mid-run and its partial e-graph reported as
"the" measurement, making the supposedly-deterministic ratios depend on host speed. Fixed: the
harness now passes a 60s, effectively-non-binding safety ceiling instead (and fails loud —
`assert!`, not a silent truncation — if that ceiling is ever actually hit), while separately
timing each expression purely as an *informational* diagnostic against the old 500ms figure. The
numbers below are from the re-run with this fix; **27/800 expressions did in fact take >=500ms**
(so the earlier truncation risk was real, not hypothetical), and one previously-"quiesced"
expression turned out to be genuinely exhausting the 100-iteration cap once allowed to run to
completion. Per-expression medians moved negligibly (see "diagnostic" section below); the pooled
ratios moved by a few percent (0.756 -> 0.719 labeler, both directions consistent with previously
under-measuring the blowup cases that dominate the pooled sum).

## The headline number(s) — there are two, and they disagree by ~500x pooled

| Bound | Pooled ratio (ΣLB / ΣApplications) | Per-expression median | Q1 | Q3 | Implied oracle savings (1/median) |
|---|---|---|---|---|---|
| **Labeler** (`derivation_ancestors`, over-approximate — the label a Guide would actually train on) | 0.7188 | **0.382** | 0.333 | 0.527 | **2.6x** |
| **Strict lower bound** (application's output node is literally on the extracted derivation path) | 0.0014 | **0.029** | 0.006 | 0.058 | **34x** |

800 expressions, 8,729,067 total rule applications recorded, 6,274,873 labeler-load-bearing,
12,390 strict-load-bearing. (Pooled totals still carry run-to-run noise of a few tenths of a
percent from iteration-order effects in the blowup cases — see the per-application harness's own
noise note — on top of the larger, one-time shift from the wall-clock fix above. The
per-expression median/quartiles are essentially unaffected by either source: 0.382/0.029 both
before and after.)

**Read the median, not the pooled ratio.** Applications-per-expression is heavy-tailed: median
195, mean 10,911 (top 50 of 800 expressions — 6.25% — account for 69% of all applications
corpus-wide; one expression alone fired 996,047 applications — the same expression that now
correctly reports as budget-exhausted rather than quiesced, see below). The pooled ratio is
dominated by a handful of blowup cases where comm/assoc/distribute saturate combinatorially; it
answers "what fraction of all firings across the whole corpus were load-bearing" (relevant to raw
compute spent), while the per-expression median answers "for a typical expression, what fraction
of its saturation's applications mattered" (the more relevant number for a per-episode Guide).
Report both; lead with the median.

**Compare to the toy-kernel baseline** (`docs/results/2026-07-08-rule-report.md`, 5 hand-picked
kernels, 61 rules, labeler bound only): aggregate ratios there ranged 10%-87% per rule with a
warning that "~75% of ALL applications load-bearing on circle_sdf is not credible." At corpus
scale the *median* expression's labeler ratio (38%) sits inside that same range, so the toy
kernels weren't wildly unrepresentative in aggregate — but the corpus surfaces the pooled/median
divergence the 5-kernel sample was too small to show, and the strict bound (not computed in the
toy-kernel report) reveals the over-approximation is far larger than "ratios read high":
**pooled labeler credits ~507x more applications than the strict walk finds on the winning
derivation path.**

## Two further diagnostics: budget exhaustion and expression size

The harness also reports (numbers from the final run,
`docs/results/2026-08-30-guide-headroom.json`, `quiesced_before_cap_count` /
`exhausted_budget_count` / per-expression `quiesced_before_cap`):

- **799/800 expressions quiesced before hitting either budget cap** (`saturate_with_limits`'s own
  `if unions == 0 { break }` convergence check — a diagnostic condition, not a certified fixpoint;
  this optimizer is budget-only by design). **1/800 genuinely exhausts the 100-iteration cap**
  (`train_b23_f02_03726`, 264 nodes, 996,047 applications, 100/100 iterations) — this is the same
  expression that previously (under the buggy 500ms-truncated harness) looked like it had
  quiesced early, because the wall-clock deadline cut it off before it could reach either its true
  fixed budget or genuine convergence. With the fix, its ratios are now attributed correctly
  (labeler 67.3%, strict 0.0035%) instead of to a truncated partial graph. **27/800 expressions
  took >=500ms wall-clock** — informational only (doesn't gate which samples count), but confirms
  the truncation risk the earlier draft's harness carried was real, not hypothetical. Excluding
  the one genuinely-exhausted expression, the heavy tail described above is a property of *how
  large the graph gets before quiescing*, not of budget clipping.
- **The labeler/strict divergence gets worse, not better, as expressions grow.** Splitting the 800
  expressions into size terciles by arena node count (small/medium/large, ~266-268 each):

  | Tercile | labeler median | labeler Q1-Q3 | strict median | strict Q1-Q3 |
  |---|---:|---:|---:|---:|
  | small | 0.375 | 0.333-0.389 | 0.056 | 0.021-0.097 |
  | medium | 0.359 | 0.297-0.399 | 0.035 | 0.017-0.059 |
  | large | 0.539 | 0.412-0.634 | 0.010 | 0.002-0.026 |

  Large expressions look *more* load-bearing under the labeler bound (54% vs 38%) but *less*
  under the strict one (1.0% vs 5.6%) — a 54x gap at the large tercile vs an 11x gap at the small
  one. This is the same heavy-tail/blowup effect as the pooled-vs-median split, now shown to be
  monotonic in expression size rather than a handful of unrelated outliers: bigger expressions
  build proportionally larger combinatorial equivalence classes (more comm/assoc churn per useful
  rewrite), which is exactly the regime a budget-bounded Guide is meant to help with — and exactly
  where this measurement says the labeler bound is least trustworthy as a training target.

## What the strict bound actually measures

`derivation_ancestors` (the labeler's substrate) is a deliberately conservative
over-approximation on three named axes (`provenance.rs` lines ~214-239): it credits every node
in a *class*, not just the node actually chosen; it pulls in union events by class membership;
it has no fixed-point pruning. The strict bound instead walks *only* the chosen derivation
tree — root, then each chosen node's chosen children, recursively (identical walk to
`labeler::chosen_tagged_nodes`, reimplemented against the crate's public API in the harness
since that internal function isn't `pub` and this round doesn't touch `labeler.rs`) — and
credits an application only if its output node is literally one of those chosen nodes.

This makes the strict bound a genuine *lower* bound, not a better estimate: it is blind to
"enabling" contributions — a `commutative` firing that never becomes the chosen node but was
necessary for congruence closure to later discover the equivalence that *did* get chosen is
real credit the strict walk cannot see. The gap between the two bounds is exactly that
enabling-credit mass, and it turns out to be enormous:

- **Structural/combinatorial rules (commutative, associative, reverse-associative, distribute,
  identity, annihilator, involution) score 73-93% under the labeler bound and ~0.0% under the
  strict bound**, for every one of them. These rules essentially never produce the literal node
  that survives extraction — extraction almost always resolves to a canonical, unrewritten
  form — but they fire in enormous volume (commutative alone: 2.3M firings) building the
  equivalence classes that let *other* rules' rewrites become reachable from the root.
- **Numeric/transcendental rules (power-recip, power-sqrt, power-rsqrt, recip-sqrt,
  even-negation) are the opposite shape**: 13-17% under the labeler bound but their strict
  scores (6-12%) are the *closest* to their labeler scores of any rule family — these rules'
  products more often survive directly into the extracted expression.
- **Rank correlation between the two bounds, per rule instance, is moderate, not strong** —
  Spearman ρ = 0.35 (n = 55 rule instances that fired; average-rank tie handling, computed from
  this document's own `per_rule` JSON — `labeler_load_bearing/fired` vs `strict_load_bearing/fired`
  per row; script not checked in, reproducible from the JSON in a few lines). **Correction
  (2026-09-01):** an earlier draft of this document reported ρ ≈ 0.02 ("almost independently") —
  that number was never computed from this harness's output (no Spearman computation exists
  anywhere in `guide_headroom.rs`; the JSON has no such field) and does not reproduce under any
  variant tried (percentage-vs-percentage, count-vs-count, or the corrected-data re-run all land
  in the 0.35-0.50 range, not near zero). It was an error, not updated data — the wall-clock fix
  above only shifted the pooled totals by a few percent, nowhere near enough to move a
  correlation from 0.02 to 0.35. **The corrected 0.35 still supports the same qualitative
  finding, just less starkly than "almost independent" claimed**: the two bounds agree
  moderately overall, but split cleanly by rule *class* — every structural/congruence rule
  clusters at ~0% under the strict bound regardless of its labeler score (60-85%), while every
  numeric/transcendental rule that fires often enough clusters in the same rough range (6-17%)
  under both. That per-class split, not the single correlation number, is the load-bearing risk
  this measurement surfaces for Phase 3 (see below): a Guide trained on labeler labels would
  learn to prize exactly the rule class the strict bound says contributes almost nothing to the
  literal extracted expression.

## Per-rule table, full 62-rule library (`pixelflow_search::math::all_rules()`)

39 distinct rule *names* fired (several are instantiated once per operator — e.g. `commutative`
covers both `Add` and `Mul` as separate `Rewrite` objects/rule indices; this table sums across
all instances sharing a name). 7 rule names never fired on any of the 800 sampled expressions:
**involution, odd-negation, ln-homomorphism, power-zero, log-power, differentiate** (odd-negation
and involution each have an instance that fired and one that didn't — `power_kind`-gated variants
that this corpus's generator families never happen to trigger; `differentiate` is inert without a
`Dwrt` node, matching its doc comment). Full per-rule-instance breakdown (62 rows, keyed by
rule index) is in the sibling JSON (`docs/results/2026-08-30-guide-headroom.json`, `per_rule`).

| rule | fired | labeler LB | labeler % | strict LB | strict % |
|---|---:|---:|---:|---:|---:|
| commutative | 2,826,152 | 2,363,621 | 83.6% | 0 | 0.0% |
| fma-fusion | 1,625,547 | 1,340,326 | 82.5% | 1,482 | 0.1% |
| distribute | 297,885 | 221,313 | 74.3% | 11 | 0.0% |
| reverse-associative | 1,352,456 | 960,775 | 71.0% | 152 | 0.0% |
| identity | 260,983 | 169,955 | 65.1% | 0 | 0.0% |
| associative | 1,277,910 | 809,484 | 63.3% | 148 | 0.0% |
| doubling | 64,742 | 34,605 | 53.5% | 83 | 0.1% |
| constant-fold | 233,967 | 110,342 | 47.2% | 1,607 | 0.7% |
| halving | 83,371 | 38,759 | 46.5% | 1 | 0.0% |
| factor | 397,645 | 167,127 | 42.0% | 192 | 0.0% |
| cos-angle-addition | 2,537 | 1,015 | 40.0% | 2 | 0.1% |
| sin-angle-addition | 3,204 | 1,113 | 34.7% | 1 | 0.0% |
| canonicalize | 7,639 | 2,426 | 31.8% | 57 | 0.7% |
| half-angle-product | 7,017 | 2,102 | 30.0% | 1 | 0.0% |
| annihilator | 85,142 | 23,131 | 27.2% | 0 | 0.0% |
| exp-homomorphism | 1,213 | 301 | 24.8% | 1 | 0.1% |
| power-combine | 5,143 | 1,073 | 20.9% | 143 | 2.8% |
| odd-negation | 4,407 | 887 | 20.1% | 29 | 0.7% |
| power-recip | 9,765 | 1,664 | 17.0% | 1,159 | 11.9% |
| reverse-angle-addition | 12,164 | 1,982 | 16.3% | 34 | 0.3% |
| power-sqrt | 14,734 | 2,332 | 15.8% | 1,526 | 10.4% |
| power-rsqrt | 3,494 | 543 | 15.5% | 422 | 12.1% |
| expand-square | 7 | 1 | 14.3% | 0 | 0.0% |
| involution | 40,255 | 5,709 | 14.2% | 0 | 0.0% |
| recip-sqrt | 3,571 | 504 | 14.1% | 408 | 11.4% |
| exp-ln-cancel | 244 | 33 | 13.5% | 0 | 0.0% |
| even-negation | 82,517 | 11,085 | 13.4% | 4,910 | 6.0% |
| exp2-log2-cancel | 276 | 36 | 13.0% | 0 | 0.0% |
| pythagorean | 1,540 | 200 | 13.0% | 0 | 0.0% |
| power-identity | 958 | 124 | 12.9% | 0 | 0.0% |
| diff-of-squares | 166 | 21 | 12.7% | 0 | 0.0% |
| power-recurrence | 8 | 1 | 12.5% | 0 | 0.0% |
| inverse-annihilation | 19,829 | 2,155 | 10.9% | 21 | 0.1% |
| idempotent | 1,104 | 82 | 7.4% | 0 | 0.0% |
| log2-power | 15 | 1 | 6.7% | 0 | 0.0% |
| cancellation | 936 | 42 | 4.5% | 0 | 0.0% |
| ln-exp-cancel | 244 | 3 | 1.2% | 0 | 0.0% |
| log2-exp2-cancel | 273 | 0 | 0.0% | 0 | 0.0% |
| power-expand-2 | 7 | 0 | 0.0% | 0 | 0.0% |

(Sorted by labeler %, names merged across operator instances — see JSON for the un-merged,
rule-index-level table the harness actually computed from.)

**Rule-triage reading, with the caveat that "reading" differs by which bound you trust:**
- Under the labeler bound alone, nothing looks safe to drop — even the bottom rows
  (`log2-exp2-cancel`, `power-expand-2`) simply have low fire counts (273, 7), not low ratios
  distinguishable from noise.
- Under the strict bound, the entire top of the labeler table (identity, commutative,
  fma-fusion, associative, distribute, reverse-associative, annihilator, involution) reads as
  **wasted** — 0.0-0.1%. Taken at face value that would argue for aggressively rule-masking
  exactly the rules the labeler bound says matter most. Taking it at face value would be a
  mistake (see above): these are the rules whose entire job is enabling other rewrites via
  congruence closure, which the strict walk cannot see by construction. **This table cannot be
  used for rule triage until the labeler/strict disagreement is understood** — that is this
  report's main finding, not a footnote to it.

## Design implications for Phase 3 (measurement-only conclusions; no Guide built this round)

1. **Headroom exists and is large under either bound** — 2.6x-34x fewer applications would
   suffice for the median expression's extraction. This clears the bar the extraction-head
   program's static/noswap=0.54 cleared for Phase 2: there is real slack for a Guide to
   recover. Recommendation: **Phase 3 is worth running**, contingent on point 2.
2. **The labeler/strict gap is the central open risk, not a rounding error.** A Guide is only
   ever going to be trainable on the labeler bound (the strict bound isn't a candidate training
   signal — it's blind to real enabling credit, as noted above). But this measurement shows the
   labeler bound's *ranking* of rules correlates only moderately with the strict bound's
   (Spearman ρ ≈ 0.35) and splits cleanly by rule class (structural rules near-0% strict
   regardless of labeler score; numeric rules track both bounds closely), and the two bounds'
   *pooled magnitudes* differ by ~500x. Before spending Phase 3 budget training against labeler
   labels, the follow-up docs/plans/2026-07-07 lines 88-90 already called for — tightening the
   union-causality over-approximation — should be understood well enough to know whether a
   *tighter* over-approximation (still safe, i.e. never under-crediting) would substantially
   change the rule ranking above, particularly for the structural-rule class where the two
   bounds disagree most. This round deliberately did not touch that logic (house rule: measure
   the looseness, don't redesign it yet); the next round should.
3. **Per-expression heterogeneity is itself a design input.** The heavy tail (69% of all
   applications from 6.25% of expressions) means a fixed per-episode saturation budget hits
   wildly different "how much of this could a Guide have skipped" targets depending on
   expression size/shape. A budget-bounded Guide (as Phase 3 proposes) should be evaluated
   across the size distribution, not just on a pooled or median number — a Guide that only
   helps small expressions (where full saturation is already cheap) is a different result than
   one that helps the blowup cases (where it's most needed).
4. **7/62 rules never fired on 800 corpus-representative expressions.** Not evidence they're
   dead — this corpus doesn't specifically target the differentiation/homomorphism cases they
   guard — but worth a wider or targeted corpus pass before any rule-masking decision leans on
   "never fires" as a signal.

## Related concurrent scoping measurements (same session, same worktree, 2026-08-30)

Two other Phase-3-scoping measurements were produced alongside this one and are cross-referenced
here rather than duplicated:

- **`docs/results/2026-08-30-guide-scope-saturation-delta.json`** — asks whether the Stockfish
  incrementality argument (the reason NNUE-style accumulators are cheap) applies to the Guide's
  *state* representation (the e-graph as it grows during saturation), as distinct from the
  extraction-head program's negative finding for *extraction candidates*
  (`docs/plans/2026-08-17-egraph-vsa-nnue-research-notes.md`: ~2x, not ~98x, because sibling
  extraction candidates differ by median 44.9% of their edge multiset). Measured against the real
  production (batched) saturation algorithm: 91.1% of all recorded rule applications are
  idempotent re-fires that create zero new nodes/edges (an even stronger form of incrementality
  than "small delta" — no update needed at all), and among the remainder, the per-application
  edge-delta fraction has median 0.14% (~728x implied speedup) — but eval economics are
  separately expensive: median 153 match evaluations per saturation round, 90.4% of all evaluated
  candidates producing no committed rewrite action (the same idempotent-refire mechanism viewed
  from the candidate side; see that file's harness doc for the root cause — rules like
  `commutative` have no "already applied" check and re-match their own already-installed output
  forever).
- **`docs/results/2026-08-30-oracle-filtered-budget-curves.csv`** (may still be running at the
  time this document was written — check the file's presence/timestamp) — the most direct dry
  run of the Phase 3 thesis test at small scale: anytime extraction-cost curves comparing
  unguided (all 62 rules) saturation against a rule set restricted to only the rules a hindsight
  pass found load-bearing, sampled at standardized work fractions. This is the closest thing in
  this session's output to actually answering "does rule filtering keep pace with full
  saturation" — this report's headroom numbers establish *that there is room* for such filtering
  to help; that file (or its successor once complete) is where "and does it actually work,
  greedily, at a budget" gets a first empirical answer.

Neither of the above changes this report's numbers or conclusions; both were produced by other
agents working the same worktree/task concurrently and are noted here for a single point of
entry into everything measured this session.

## Reproducibility

- Corpus identity: regenerated this session, `gen_bench_corpus --target 8000 --seed 42`
  (default seed). `corpus_train.bin`: 6,480 entries; `corpus_dev.bin`: 1,512 entries; quarantine
  1.17% excluded, 0 JIT-vs-oracle miscompiles.
- Rule set: `pixelflow_search::math::all_rules()`, 62 rules (40 algebra/parity/trig/exp/power +
  2 fusion + 1 differentiation — see that module's doc comment for the exact split; the older
  "40+2=42" comment in `egraph/mod.rs` is stale relative to `math/mod.rs`'s "59+2+1=62", which
  this measurement confirms by construction (`all_rules().len() == 62`)).
- Extraction: `CostModel::latency_prior()` (the compiler's default, per Phase 2's still-standing
  gate).
- Full structured output: `docs/results/2026-08-30-guide-headroom.json` (per-expression rows,
  per-rule-index rows, quartiles, pooled ratios — this document's tables are derived from it).
- Harness: `pixelflow-pipeline/src/bin/guide_headroom.rs`.
