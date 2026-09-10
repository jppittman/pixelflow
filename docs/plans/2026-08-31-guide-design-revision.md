# Guide design revision: measured economics and a pre-registered Phase 3 experiment

**Date:** 2026-08-31
**Status:** Design revision — synthesizes the 2026-08-30 Phase 3 scoping round
**Builds on:** `docs/plans/2026-07-07-guided-saturation-redesign.md` (the thesis and phase gates),
`docs/plans/2026-08-17-cost-model-domain.md` (the join table and anti-rows), the extraction-head
research workflow (`docs/plans/2026-08-05-egraph-nnue-research-workflow.md`)

## 0. Framing (binding, restated from the redesign plan)

The kernel language is strongly normalizing; this optimizer is **budget-only** by design.
Saturation spends a budget and stops, full stop — it does not detect or rely on reaching a
fixpoint, and quiescence (`stats.total_unions == 0` for a round) is a diagnostic condition, never
a certified closure. Every value claim in this document is **anytime**: best extraction cost
(static-latency-prior units, deterministic, no wall-clock timing) plotted against work performed
(rule applications, saturation rounds). The Guide is a **quality-at-budget** play, uniformly —
never framed as "finding the optimum faster."

## 1. What was measured this round

Three scoping measurements, all additive (no change to `derivation_ancestors`/`labeler.rs`
semantics), all pinned to a static-latency-prior cost model, no wall-clock timing:

| Measurement | Question | Harness | Report |
|---|---|---|---|
| Oracle headroom | If a perfect Guide replayed only load-bearing applications, how much smaller would saturation be? | `pixelflow-pipeline/src/bin/guide_headroom.rs` | `docs/results/2026-08-30-guide-headroom.md` |
| Saturation delta economics | Does the Stockfish incrementality argument (refuted for extraction candidates) hold for the Guide's actual scoring object — the growing e-graph? | `pixelflow-pipeline/src/bin/guide_scope_saturation_delta.rs` | `docs/results/2026-08-30-guide-scope-saturation-delta.md` |
| Oracle-filtered anytime budget curves | Does replaying only oracle-approved rules reach forms unguided saturation never finds, at standardized work fractions? | `pixelflow-search/examples/oracle_filtered_budget_curves.rs` (deleted) | `docs/results/2026-08-30-oracle-filtered-budget-curves.md` |

**Audit note on this round's provenance.** All three artifacts existed on disk from an
interrupted session (three agents that each hit a session limit mid-write) and were audited,
not regenerated from scratch, per the closing task's instruction. Two corrections were made
during the initial audit (2026-08-31) and are recorded in their respective reports:

- `guide-headroom.md`'s pooled-ratio prose (0.7535, 7,061,943 applications, "418x") was
  transcribed from a run slightly earlier than the JSON that shipped; the pooled totals carry
  ~0.1-1% run-to-run noise from a handful of combinatorial-blowup expressions (confirmed by
  re-running the harness), while the per-expression medians/quartiles — the numbers this
  document leads with — reproduced bit-for-bit across re-runs. The prose was corrected to match
  the shipped JSON at the time (0.7557 pooled, ~434x) — now superseded again, see below.
- `guide-scope-saturation-delta`'s harness was rewritten mid-session, *after* its first JSON was
  generated, from a single-application-stepping design to the real production batched algorithm
  (`EGraph::saturate_with_limits`) — a deliberate, well-argued methodology upgrade documented in
  the harness's own module doc, but one whose numbers were never re-captured before the session
  died. This round re-ran the (now-final) harness, found its per-application summary silently
  degenerate (median exactly 0% because 91.1% of recorded applications are literal no-ops), added
  a small additive fix (report the zero-delta share and the delta distribution conditioned on
  state-changing applications separately), and regenerated both the JSON and the markdown. The
  qualitative conclusion is unchanged and *stronger* than the superseded draft; every specific
  number is different. See that report's "Revision note" for the full account.

**A second round of corrections landed 2026-09-01**, from an automated PR review
(`chatgpt-codex-connector` on PR #1067) that caught six issues across the harnesses, two of them
P1: `guide_headroom` was reusing production `saturate()`'s 500ms wall-clock deadline as its own
stopping condition, silently truncating any expression that took longer and making the
"deterministic" ratios depend on host speed (27/800 expressions actually exceeded 500ms — fixed
with a generous, fail-loud safety ceiling instead; pooled ratios moved 0.756->0.719 labeler,
0.0017->0.0014 strict, medians essentially unchanged); and `guide_scope_saturation_delta` was
attributing node/edge deltas to applications whose created node was later pruned by a rare e-graph
rebuild refusal, inconsistently counting the node but zeroing its edge contribution (136 refusal
events, 1,997 origins affected — 0.044% of applications, now excluded consistently; negligible
effect on the headline numbers). Two more P2 findings were also fixed: a stale-median-derived
"0.00x" speedup figure now correctly derives from the state-changing-only median (~728x, not
"731x" — the fix also slightly changed which applications count as state-changing), and a
`saturate_with_limits` cap-vs-quiescence conflation in the same harness (0/1,512 expressions
changed classification in practice, but the harness no longer depends on that being true by
luck). A `oracle_filtered_budget_curves` finding (safety-timeout scoped per-single-iteration
instead of per-curve, and a zero-baseline regret formula that could mask real regret) had no
effect on this round's already-null result (see §2.3) but is fixed for the recalibrated re-run
§2.3 recommends. Every specific number in §2.1 and §2.2 below reflects this second round; see
`docs/results/2026-08-30-guide-headroom.md` and
`docs/results/2026-08-30-guide-scope-saturation-delta.md`'s own revision notes for full detail,
including the correction of an unrelated, unreproducible ρ≈0.02 Spearman-correlation claim the
same review's questions led to catching (the correct value, computed from the shipped JSON, is
ρ≈0.35 — see §2.1).

The third measurement (`oracle_filtered_budget_curves`) needed no *data* correction in the first
audit pass, but its result is a **null** one — see §2.3.

## 2. The economics, honestly

### 2.1 Headroom: the bounds gap is the first-order finding, not a footnote

Over 800 stride-sampled expressions (`corpus_train.bin`/`corpus_dev.bin`, 7,992-entry
population), two bounds on "what fraction of saturation's rule applications were load-bearing
for the winning extraction":

| Bound | Pooled ratio | Per-expression median | Q1 | Q3 | Implied oracle savings (1/median) |
|---|---:|---:|---:|---:|---:|
| **Labeler** (`derivation_ancestors`, over-approximate — the label a Guide would actually train on) | 0.7188 | **0.382** | 0.333 | 0.527 | **2.6x** |
| **Strict lower bound** (application's output node is literally on the extracted derivation path) | 0.0014 | **0.029** | 0.006 | 0.058 | **34x** |

Both bounds clear the bar Phase 2 used to justify building the extraction head
(static/noswap=0.54) — there is real slack for a Guide to recover under either reading. But the
two bounds don't just disagree in magnitude (2.6x vs 34x, a ~13x spread on the headline number
alone; ~507x on the pooled ratios); **they correlate only moderately overall** (Spearman ρ ≈
0.35, n=55 rule instances that fired — average-rank tie handling; see the headroom report's
"revision note" for how this was computed, and for the correction of an earlier draft's
unreproducible ρ≈0.02 claim, which was never actually computed from this harness's output).
The moderate overall correlation masks a clean split by rule *class*:
structural/congruence rules (commutative, fma-fusion, distribute, reverse-associative,
associative, identity) score 63-84% under the labeler bound and ~0% under the strict one,
because their entire job is enabling congruence closure, not surviving into the extracted
expression — invisible to a walk that only credits the literal chosen derivation path. Numeric
rules (power-recip, power-sqrt, recip-sqrt, ...) are the opposite shape and are the only rule
class where the two bounds track each other closely (6-17% both).

**This per-class split, not the single correlation number, is the first-order finding of the
whole round.** A structural/numeric split this clean means "headroom exists" is not by itself a
green light for training a Guide on labeler labels — see §3, where this becomes design decision
#1.

### 2.2 Does incremental evaluation pay in the (monotone-append) saturation setting? Yes, decisively

The extraction-head program already answered this question for a *different* object — sibling
extraction candidates at one e-class — and got a **negative**: candidates differ by a median
44.9% of their edge multiset, buying an incremental accumulator only ~2x
(`docs/plans/2026-08-17-cost-model-domain.md`, anti-row A6). The Guide scores a different object:
the e-graph state itself, as it grows during saturation — monotone, append-only. Measured against
the real production batched algorithm, over 1,512 corpus expressions:

| Quantity | Value |
|---|---:|
| Recorded applications with zero node/edge delta (idempotent re-fires — see below) | **91.1%** |
| Edge-delta fraction, state-changing applications only (median / p90) | **0.14%** / 0.79% |
| Implied speedup among state-changing applications (1/median) | **~728x** |
| Cumulative-vs-incremental ratio, pooled across all applications | **14,422x** |
| Candidate evals per committed rewrite action | 10.4x (90.4% of scored candidates commit nothing) |

**Incremental evaluation pays, and pays harder than the chess-engine analogy this codebase's own
docs use as the aspirational target** (1/728 ≈ 0.14% incremental cost, i.e. ~99.86% savings on
the applications that do anything at all — and 91.1% of applications need literally zero work).
`GraphAccumulator::add_edge`/`remove_edge` (`pixelflow-search/src/nnue/guide/accumulator.rs`,
already implemented, currently dead-code-gated behind the J10/A2 roadmap seam in the cost-model
domain doc) is the right representation for scoring saturation state — not a speculative
recommendation, the measured deltas say so directly.

**But accumulator cost is not the only cost.** 90.4% of match candidates a Guide would ever be
asked to score produce no committed rewrite at all — the same idempotent-refire mechanism from
the other side. `commutative`/`associative`/`reverse-associative`/`even-negation`/`involution`
have no "already applied" check in their `apply()` implementations and re-match their own
already-installed output on every subsequent scan. An O(1)-per-edge incremental score is still
100% wasted on 90% of the candidate pool. **Candidate-level deduplication — skip a `(rule,
canonical class content)` pair once it has fired and resolved — is the more urgent half of the
Guide's per-round cost, not accumulator maintenance.** This is a design requirement, not a nice-
to-have (§4, §5).

### 2.3 The anytime-curve gap: inconclusive this round, not negative

The direct dry run of the Phase 3 thesis — do oracle-filtered rule sets keep pace with unguided
saturation at standardized work fractions, and does either curve reach forms the other never
finds — came back **uninformative as calibrated**. Across all 225 corpus expressions (2,250
rows: 5 work-fraction checkpoints × 2 curves), `regret_pct` is exactly 0.0000 everywhere:
unguided and oracle-filtered curves never differ in extraction cost at any tested checkpoint, for
any expression. The cause is a checkpoint-grid miscalibration, not evidence against the thesis:
97.8% of expressions (220/225) already reach quiescence — or hit the ~15,000-class safety cap for
the largest "classical"-tier expressions — *before* the coarsest checkpoint tested (25% of a
per-tier nominal iteration budget). The curves have no shape left to compare by the time
measurement starts. Full account, including the class-cap mechanism and the corroborating
per-tier oracle rule-count table (median 3/7/24 of 62 rules kept for blitz/rapid/classical), is in
`docs/results/2026-08-30-oracle-filtered-budget-curves.md`.

This measurement needs re-running with **absolute small-iteration checkpoints** (1, 2, 3, ..., 20)
rather than fractions of an already-generous nominal budget, to have a chance of showing the
anytime trajectory's actual shape. That re-run was not attempted this round: a `bootstrap_extraction_head`
publication benchmark was live in another worktree with a sentinel-guarded abort budget, and an
earlier, unnecessary re-run of this same harness (to recover console output that was never
persisted — see the report) cost that run one of its two allowed aborts. Recalibrating and
re-running this measurement is recommended as an early Phase 3 task (§5), not a blocker to
starting.

### 2.4 Go/No-Go

**Go**, on the strength of §2.1 and §2.2, both of which are decisive, deterministic-count
measurements at corpus scale:

- Headroom is large under either bound (2.6x-34x), clearing the bar Phase 2 used.
- Incrementality — the architectural bet the whole NNUE-for-e-graphs framing depends on — holds
  for the Guide's actual scoring object, and holds *harder* than for extraction candidates
  (where it was refuted) and harder than the chess-engine analogy this codebase aspires to.

**With one open evidence gap, not a blocker but a debt**: §2.3's direct anytime-curve dry run is
inconclusive, not confirmatory. The two decisive measurements establish that there *is* headroom
and that scoring it would be *cheap* — they do not yet show that a rule-filtered replay actually
*keeps pace* with unguided saturation at a real budget. Phase 3 should recalibrate and re-run
`oracle_filtered_budget_curves` early (cheap once recalibrated: the existing run took under a
minute) as a pre-flight check before investing in a trained Guide, precisely because it is the
one measurement of the three that could, in principle, still come back negative.

## 3. Design decision #1: label semantics is a prerequisite, not a parallel task

The redesign plan already flagged tightening `derivation_ancestors`'s union-causality
over-approximation as a "known follow-up... before training the Guide on per-application labels"
(Phase 1 entry). This round's finding — that the labeler and strict bounds correlate only
moderately overall (Spearman ρ ≈ 0.35) and disagree sharply for the highest-volume rule class
(structural/congruence rules: 63-84% labeler, ~0% strict) — upgrades that from a nice-to-have
refinement to a **precondition on Phase 3's training signal**: a Guide trained on labeler labels
would learn to prize exactly the rule class the strict bound says contributes almost nothing to
the literal extracted expression, and the labeler bound is the one whose over-approximation axes
are named but not yet bounded (`provenance.rs` documents three: credits every node in a class not
just the chosen one, pulls in union events by class membership, has no fixed-point pruning).
Building Phase 3 on an unexamined training signal risks teaching a confidently wrong policy for
that rule class rather than a merely noisy one.

**Options:**

1. **Strict extracted-path labels only.** Train the Guide exclusively on the strict bound
   (application's output node literally on the extracted derivation path). Conservative and
   sound-by-construction — no over-approximation to distrust — but by its own nature blind to
   "enabling" credit: a `commutative` firing that never becomes the chosen node but was necessary
   for congruence closure to later discover the equivalence that *did* get chosen would be
   trained as "wasted," when structurally it is not. Given commutative/associative/distribute
   dominate application volume (§2.1's per-rule table), training a policy that treats them as
   pure waste risks pruning exactly the rules that make everything else reachable. Cheapest
   option to start Phase 3 with; likely the wrong long-run target.
2. **Tighten `derivation_ancestors` first.** Close (or narrow) the three named
   over-approximation axes before generating any training labels, then re-measure whether the
   rank correlation with the strict bound improves. This directly answers the open question
   ("would a tighter over-approximation reorder the rule-priority ranking a Guide would learn")
   rather than guessing. Highest confidence, highest up-front cost — it is a `provenance.rs` /
   `labeler.rs` change, not a measurement.
3. **Two-stage: strict-label cold start, tightened-labeler refinement.** Bootstrap the Guide on
   strict labels (option 1) to get a first working, sound-if-conservative policy and validate the
   supervised-training pipeline end to end; in parallel, tighten `derivation_ancestors`; once
   tightened, re-label and re-train, and treat any change in strict-vs-tightened-labeler rank
   correlation as a direct, load-bearing measurement of whether the tightening mattered. This
   sequences the (already-planned) prerequisite work concurrently with early Phase 3 engineering
   instead of serializing behind it, at the cost of a possible policy re-training once tightening
   lands.

**Recommendation: option 3.** It does not block Phase 3's engineering start (the supervised
training pipeline, feature extraction, and candidate-deduplication work in §2.2/§4 can all
proceed against strict labels immediately), it produces the tightening-vs-ranking measurement the
July plan already called for as a direct byproduct rather than a separate research detour, and it
avoids committing training compute to labeler labels before their over-approximation is
understood. The pre-registered experiment in §5 is written against whichever label source is
current at the time it runs, and its cold-start/family-held-out protocol (§5) is designed to
re-validate cleanly after a mid-program label-source swap.

## 4. Features, relooked

The existing `GraphAccumulator` design (`pixelflow-search/src/nnue/guide/accumulator.rs`, J10 in
the cost-model domain doc, roadmap-admitted for Phase 3) is a whole-graph summary: four K=32
sections combining marginal parent/child sums and 1-/2-hop binding accumulations into a single
128-dim state vector, incrementally maintainable per §2.2's measured deltas. This section asks
whether that is the right feature shape for the object the Guide actually scores — a decision to
*re-rank a specific candidate rewrite*, not to characterize the whole graph.

**The case for candidate-local features over a whole-graph summary**, argued from this round's
measurements rather than from architecture taste:

- **The dominant cost (§2.2) is candidate-level deduplication, not graph characterization.**
  90.4% of candidates need to be recognized as "already resolved, will produce no state change"
  — a property of *one rule instance at one e-class*, not of the graph as a whole. A whole-graph
  summary embedding does not naturally expose "have I already fired `(commutative, this exact
  e-class's content)`"; a candidate-local feature keyed on `(rule_idx, canonical class content)`
  is exactly the natural representation of the deduplication key the dedup mechanism (§2.2's
  design implication) needs anyway. Building the feature and the dedup key as the same object
  avoids computing the information twice.
- **Deltas are local (§2.2's "why the deltas are this small").** State-changing applications
  create 1-2 nodes against a graph of hundreds-to-thousands of nodes; the graph-level signature
  barely moves per application (0.14% median edge fraction) even when the *local* effect (does
  this specific rewrite fire, and what does it produce) is exactly what a rewrite-ordering
  decision needs. A whole-graph accumulator answers "how has the position changed overall," which
  is the wrong granularity for "should I fire this specific match right now" — a decision that
  depends on this match's own e-class neighborhood and the rule's identity, not the graph's
  aggregate shape.
- **The label-semantics gap (§3) is per-rule, not per-graph.** The labeler/strict rank
  disagreement is a property of *individual rules* (structural rules diverge, numeric rules
  don't) — a policy that conditions on rule identity plus local structure has a natural way to
  learn "trust the labeler bound for these rules, discount it for those," which a single pooled
  graph-level score cannot express.

**Proposed feature shape**: matched e-class neighborhood (the match's own operands/children,
one hop out — cheap to compute incrementally off the same `EGraph`/`Provenance` accessors this
round's harnesses already used) + rule identity (already a small enum) + budget state (fraction
of iteration/class budget consumed so far, since §2.1 shows large expressions have a
qualitatively different labeler/strict divergence than small ones, and a policy that doesn't know
where it is in the budget cannot learn tier-dependent behavior). This is a smaller, cheaper
feature than the current 128-dim whole-graph summary, and it is closer to what a move-ordering
decision (as opposed to a position-evaluation decision) actually needs — the accumulator's
whole-graph summary remains the right shape for the *extraction-cost* Judge (predicting the best
achievable cost from a state), a genuinely different question from "should this match fire now."
No part of this recommendation is a VSA/binding-algebra claim one way or the other; it is a
locality argument from the measured cost structure. Nothing here requires deleting the existing
accumulator work — see §6 (out of scope) for why extending it, rather than replacing it, is the
right sequencing.

### 4.1 Growth is a feature, and it is most of the question

**Direction (JP, 2026-09-09).** State the Guide's question as an economic
one rather than a ranking one: *is this application going to grow the graph
and give me nothing, or grow it a little and give me something useful?*

Growth is **computable before firing, cheaply and exactly**, which is what
makes it a feature rather than a prediction. A rewrite's RHS is a template
of known size, and the e-graph is hash-consed, so the nodes it would add are
the RHS nodes that do not already exist. An application whose RHS is
entirely present adds zero — which is the 90.4% dedup case of §2.2 seen from
the cost side rather than the identity side, and another reason to build the
dedup key and the feature as one object.

This costs nothing to add to §4's proposed feature shape (matched
neighborhood + rule identity + budget state), and it is the term that most
directly expresses what the budget is for.

**Why it stops being optional.** Every rule in the current set adds one or
two nodes, so growth barely varies and a policy that ignored it would lose
little. That changes with
[2026-09-09-composition-is-linking.md](2026-09-09-composition-is-linking.md),
which proposes `Ref(k) ⟷ body(k)` — inlining as a rewrite, with the cost
model choosing between a call and a splice. That rule's growth is *the
entire referent*, and a rewrite fires eagerly: an ungated inline rule puts
the inlined form in the e-class whether or not extraction wants it, so the
graph pays even when the answer is "keep the call."

The magnitude is already measured on the same shape of question. Expanding a
glyph's reduce before saturation — the same "introduce N copies of a body"
move — costs 3.8× on `A`, 5.9× on `O`, and **8.7× on `8`** (87,007 →
758,059 nodes). No rule in the present set has that profile. So
growth-awareness is not a refinement of the Guide for that rule; it is the
precondition for the rule being admissible at all, and the linking design
should be read as blocked on it.

### 4.2 Measured: spending the whole budget is not monotone in extracted cost

Growth is now computed exactly before firing (`EGraph::predicted_growth`,
asserted equal to the measured delta on every application under growth
telemetry — 10,819 real applications, zero mismatches). Wiring it into the
scan-time estimator in place of the flat 0/1/3 guess did what the guess was
preventing: saturation ran to its cap where it had stopped short
(`circle_sdf` 1218 → 2000 classes, `redundant` 428 → 500).

And on the shipped-kernel corpus (`egraph_off_on`, 16 kernels that run),
**that made 4 kernels extract worse** — `mandelbrot_distance` 518 → 556
(+7.3%), `smooth_min_scene` 134 → 142 (+6.0%), `chrome_packed` and
`plasma` by a point — against 4 better (`julia_set` 717 → 689, `metaballs`
155 → 140, `domain_warp_fbm`, `chrome_R`) and 8 unchanged, with 4–16× the
saturation wall clock on the ones that regressed.

**Decision (JP, 2026-09-09): more saturation is better, and this is the
extractor's bug.** The first reading of that table — that the guess was
regularizing by accident and should be kept — is withdrawn. A richer
equivalence class cannot make the true optimum worse; a superset of forms
contains everything the subset did. So an extractor that returns a *worse*
DAG when given *more* choices is not being led astray by the graph, it is
not monotone in what it is given, and that is a defect in extraction —
greedy with swap-refinement, not exact — with a concrete reproducer:
`mandelbrot_distance` 518 → 556 and `smooth_min_scene` 134 → 142 on the
same kernel with a superset graph.

Therefore the estimator now spends the budget it is given
(`action_cost = predicted_growth`), the four regressions are filed against
the extractor rather than absorbed by saturating less, and the exact growth
term stays available to the Guide as a feature. Two consequences for Phase 3
below: the per-candidate feature set of §4 gains an exact growth term at no
cost; and "budget exhausted" must not be read as success in any experiment
here — extracted cost is the only score, and where it moves the wrong way on
a richer graph, the extractor is what is being measured.

## 5. Pre-registered Phase 3 experiment

Stated purely at-budget, per §0's framing — no claim about optimality, no wall-clock timing.

**Setup.** Corpus split by generator family, never by instance (family-held-out — see below).
Cost measured with `CostModel::latency_prior()` (Phase 2's still-standing default), deterministic
and static. Budget denominated in rule applications performed (an anytime x-axis, matching
§2.3's recalibrated checkpointing once that lands).

**Claim.** At budget tier B (a specific corpus-family-appropriate application count, sized from
this round's per-expression application counts — median ~195, heavy-tailed to hundreds of
thousands, per §2.1), a Guide-ordered saturation reaches extraction cost **≤ unguided-at-B cost ×
(1 - Y%)** for a pre-committed Y (to be fixed before the first training run, not tuned after
seeing results), and **approaches unguided-at-4B cost** (i.e. quality Guided achieves at 1x
budget is competitive with quality unguided achieves at 4x budget) — the anytime-curve gap this
whole program exists to demonstrate, made concrete and falsifiable.

**Training protocol:**
- **Supervised on hindsight labels** exclusively — no RL, no critic, no REINFORCE gradient path
  (the July audit's verdict on the RL apparatus stands unconditionally; nothing in this round's
  measurements bears on it either way).
- **Cold-start**: the Guide's first checkpoint is trained from labels only, no warm-start from
  any prior guide/mask-head weights (all of which were deleted per the unified-training-flow
  history) — a clean baseline for the accept/kill gate below.
- **Family-held-out tiers**: DEV/FINAL split by generator family exactly as the extraction-head
  program's split discipline requires (`docs/plans/2026-08-05-...` §0.2) — never select on a
  family the Guide was trained on.
- **Deterministic cost-regret metric, no timing**: regret = `(guided_cost - reference_cost) /
  reference_cost` at each budget checkpoint, where `reference_cost` is the best cost either
  guided or unguided reaches at *any* checkpoint for that expression (the same empirical-reference
  convention `oracle_filtered_budget_curves` already uses, once its checkpoint grid is fixed per
  §2.3).

**Accept gate**: the claim above holds on FINAL (family-held-out), at the pre-committed Y,
recorded with per-benchmark distribution (never a single geomean number) — mirroring the
extraction-head program's own claim-gate discipline.

**Kill gate**: mirroring the extraction-head program's kill/pivot gate (§4.3 of that plan) — if
after a fixed, pre-committed number of clean training rounds (clean = current label source per
§3's chosen option, split discipline enforced, corpus at the scale this round measured) the
Guide still cannot beat unguided-at-B even greedily, stop and record it. "Guided can't match full
saturation even greedily" was always the redesign plan's stated Phase 3 gate
(`docs/plans/2026-07-07-...`, Phase 3 entry) — this section only makes it falsifiable with
specific numbers instead of a qualitative check.

**Honest fallback (pre-registered, mirroring the extraction-head program's §6 pattern)**: if the
kill gate fires, the deliverable becomes the measured economics themselves — the incrementality
result (§2.2, stronger than the chess-engine analogy), the headroom bounds and their per-rule-class
divergence (§2.1, a genuine methodological finding about hindsight-labeling provenance systems
independent of whether a Guide gets built on them), and the candidate-level deduplication
requirement (§2.2) as a load-bearing finding for *any* future guided-saturation attempt, not just
this one. This is not a consolation prize — a rigorous negative on "does labeler-trained guidance
beat greedy full saturation at this budget scale," backed by the bounds analysis explaining *why*
(the highest-volume rule class is exactly where the two label sources disagree most), is itself a
publishable and useful result, exactly as the sibling extraction-head program pre-registered for
its own kill gate.

## 6. Explicitly out of scope for Phase 3

- **RL, in any form.** The July audit's verdict against REINFORCE/critic training stands; nothing
  measured this round revisits it. Supervised hindsight labels only (§5).
- **A learned Judge (extraction-cost NNUE) as Phase 3's reward or reference signal.** Phase 2's
  static-latency-prior default stands; this round did not re-evaluate that choice (per the
  redesign plan, that re-evaluation is explicitly deferred to "whenever cost-model research
  resumes," not bundled into Phase 3). All cost numbers in this document and in the
  pre-registered experiment (§5) are static-table costs.
- **Incrementality work beyond what §2.2 already justifies.** The saturation-delta measurement
  says accumulator-style incremental scoring *pays* for the Guide's state object — that is now
  in scope (§4 builds on it) — but this does not reopen the extraction-candidate incrementality
  question the cost-model domain doc's anti-row A6 already closed (44.9% median churn, ~2x only,
  not worth an O(Δ) extraction path). A6 stands.
- **Anti-rows from the cost-model domain doc stand unchanged** — in particular A2 (a `GraphEdit`
  enum for `GraphAccumulator`'s mutators): §4 recommends *extending* the accumulator surface with
  candidate-local features, not elaborating its mutator API into a new abstraction. Build the
  enum, if at all, only when a concrete Phase 3 caller needs it — not ahead of that consumer.
- **Recalibrating `oracle_filtered_budget_curves` is in scope as an early Phase 3 task (§2.3,
  §2.4), but is not itself part of the pre-registered experiment (§5)** — it is a pre-flight
  sanity check, not the accept/kill measurement.
