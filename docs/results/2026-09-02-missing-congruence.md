> **Retracted/Superseded (2026-09-08), ledger L054.** The "median 8.66%, p90 13.2%" truncation-cost premise this measurement is framed against (`docs/results/2026-09-01-production-saturation-telemetry.md:41-46`) is computed specifically over the 132 `ClassCap` rows against a 4×-lifted, no-clock reference — it is a tree-DP cost whose *sign* the chrome clock contradicted (12× more classes made the kernel 15% slower, L072), the same reason that source document's own banner withdraws it. It is not derived from, and is not made obsolete by, the ClassCap/Timeout stop-reason prevalence shifting from 68.4% to 93% under the current budget (that shift is L053/L056, now marked historical: the 200 ms-timeout regime, pre-#1118). The under-merging *mechanism* this document measures is unaffected and stands; only the 8.66% cost premise is withdrawn, and for the sign problem, not the regime. Verdict and rationale: `docs/results/2026-09-07-claims-ledger.md` (PR #1207); the corrected benchmark: `docs/plans/2026-09-07-benchmark-correction.md`.

# Missing congruence measurement (issue #1106)

Read-only measurement, no fix. THE FINDING under test: `EGraph::union(x, y)`
pushes only the merged class onto the worklist; `rebuild_budgeted` then
canonicalizes and dedups *that class's own nodes*, but nothing walks the
e-nodes elsewhere in the graph that reference `x`/`y` as a child (no e-node
parent list exists — `EGraph::parent` is the union-find parent pointer, not a
parents-of-a-class index). So `Add(x, z)` and `Add(y, z)` merge only if some
later rule sweep happens to re-walk them. This under-merges (sound, not a
correctness bug) — the question is how much, and whether it explains the
production 5,000-class cap binding on most real kernels (#1087: 68.4% of real
kernels, median 8.66% / p90 13.2% truncation cost).

**Method.** For each kernel: run the exact production regime —
`Optimizer::production()` (pixelflow-search#1108's "one optimizer entry
point": the production rule set, `Budget::Production` == `config_for_node_count`'s
tiers + `saturate_with_full_budget`'s semantics, `CostModel::latency_prior`),
exactly as `optimize_runtime_arena_uncached` calls it (`Dwrt` lowered and
`Reduce` unrolled first). Then clone the post-saturation e-graph and run a
**full upward-congruence-closure** sweep to fixpoint on the clone: repeatedly
re-canonicalize every live class's e-nodes through `find` and union any two
live classes whose canonicalized node forms now coincide, until a full pass
finds zero new unions. This is exactly the "walk every e-node that
references a changed class" step production's `union`/`rebuild_budgeted`
never performs. The original (pre-closure) e-graph is untouched — closure
only ever runs on a `.clone()`. Cost is `CostModel::latency_prior()` summed
over the extracted arena's reachable op nodes (not `ExtractedDAG::total_cost`'s
cycle-penalty-inflated DP total).

**Corpus.** 206 real kernels: 12 `shader_bench` ShaderToy ports, 1
hand-transcribed psychedelic shader kernel, 3 packed cell-grid geometries at
the sizes core-term actually compiles (80×24@1×, 80×24@2×, 120×40@2×, each
623 nodes), and 190 glyph arenas (95 printable ASCII glyphs × 2 display
densities) — dumped via the `#[ignore]`d telemetry-dumper tests this probe's
harness borrowed from `claude/rule-order-numeric-first` (cell-grid and glyph
dumpers cherry-picked verbatim onto this branch; the shader/psychedelic
dumper likewise — that branch itself has diverged too far from `main` to
merge wholesale). Plus 200 size-stratified synthetic classical expressions
from `BwdGenerator` (max_depth ∈ {3,5,7,9,11} × 40 seeds, unoptimized/
junkified form — the same generator `gen_bench_corpus`/`bootstrap_extraction_head`
use for extraction-head training data). All numbers below are pooled over
these 406 kernels unless a table says otherwise. Probe code:
`pixelflow-search/src/runtime.rs`, `mod congruence_gap_probe` (`#[ignore]`d
test `missing_congruence_measurement`).

## The five numbers the task asked for

1. **Additional unions closure finds**: **922 total**, pooled across all 406
   kernels — **0.24%** of the pooled live-class count (922 / 389,252 summed
   `live_before`).
2. **Median per-kernel live-class-count reduction**: **0.00%** (p90 0.49%,
   max 10.48% — see "biggest single-kernel effects" below). Most kernels have
   *zero* missing congruence at the point production saturation stops; the
   effect that exists is concentrated in a small tail.
3. **Median extracted-cost change after closure**: **0.00%** (closure
   changes the extracted expression's cost on only 23/406 kernels = 5.7%;
   among those 23, the median change is a **3.96% cost reduction**, and
   **zero** are regressions — closure never made a kernel's extraction
   worse, only occasionally cheaper).
4. **Kernels that hit the class cap that would NOT have with closure**: this
   number needs two readings, because the naive one is misleading — see
   "the cap correction" below. Naive: 229/231 cap-hit kernels (99.1%) show
   `live_after < max_classes`. Corrected: only **2/231** cap-hit kernels
   have `live_before ≥ max_classes` in the first place (the cap binding on
   the *live/semantic* class count, not just the raw allocation count) — and
   closure rescues **0 of those 2**.
5. Split by ClassCap-stopped vs. not: **no difference**. Both groups show
   median 0.00% class reduction and median 0.00% cost change (table below).
   The missing-congruence effect does not concentrate in capped runs.

## The cap correction (why naive #4 is the wrong number)

`OptimizerStats.classes` / the 5,000-class cap check
(`self.classes.len() > max_classes` in `EGraph::saturate_bounded`) is the
**raw allocation count** — `EGraph::classes` is an append-only `Vec` that
never shrinks; `union` merges via the parent pointer but a merged-away
class's slot stays allocated. It is *not* the live/canonical class count
(`find(i) == i`). These two numbers are already very different **before any
closure runs**: among the 231 cap-hit kernels, the median slack
(`max_classes − live_before`) is **3,648 classes** out of a 5,000 cap — the
live/semantic graph is, on median, less than a third the size the raw
allocation count suggests, with no closure involved at all. Only 2 of 231
cap-hit kernels have `live_before ≥ max_classes` — i.e. the cap is binding on
something the closure could plausibly fix — and offline closure does not
bring either of those two under the cap.

So the naive "229/231 would avoid the cap" number is really measuring "the
live class count is usually already far below the raw allocation count,
independent of closure" — a true fact, but not evidence that **missing
upward congruence specifically** is what's inflating the raw count near the
cap. The raw-vs-live gap is dominated by something else: e-graph growth from
rule application (associativity/FMA/commute variants materializing as
distinct, non-duplicate e-classes) that is *not* congruent under any closure
— it's real alternative structure, not redundant structure this defect is
failing to notice as equal.

**This measurement does not support H-a** as a primary driver of the cap
binding early. The additional congruence this defect leaves on the table is
real (922 unions, up to 10.5% class-count reduction on the worst single
kernel, up to ~11% cheaper extraction on a few small glyphs) but too small
and too poorly correlated with cap-hit status to be "what's binding the
5,000-class cap on 68.4% of real kernels" (#1087). The cap appears to bind on
genuine e-graph growth, not primarily on unrecognized duplicates.

## Split by ClassCap-stopped

| | n | median class reduction | median cost change |
|---|---|---|---|
| ClassCap-stopped | 231 | 0.00% | 0.00% |
| not ClassCap-stopped | 175 | 0.00% | 0.00% |

## By category

| category | n | median class reduction | p90 class reduction | cap-hit kernels |
|---|---|---|---|---|
| cellgrid | 3 | 0.56% | 0.56% | 3/3 |
| shader | 12 | 0.00% | 3.42% | 9/12 |
| psychedelic | 1 | 0.85% | 0.85% | 1/1 |
| glyph | 190 | 0.00% | 0.49% | 178/190 |
| synthetic | 200 | 0.00% | 0.32% | 40/200 |

Classical-tier-only (`max_classes == 5000`, the specific "5,000-class cap"
#1087 measured): 198/206 real kernels fall in this tier, and 187/198 (94.4%)
hit the cap — higher than #1087's 68.4% because this corpus is 92% glyph
arenas (178/190 glyph arenas alone hit the cap), a heavier-tailed mix than
#1087's. The rate is corpus-composition-sensitive; the cap-correction finding
above (live class count already far under the cap) held at the same
magnitude within this subset too.

## Biggest single-kernel effects

The largest per-kernel reductions are small punctuation glyphs, not the
big kernels that hit the cap — these never reach the classical (5,000) tier
at all (`node_count` puts them in `blitz`/`rapid`, `max_classes` 500/2000),
so the effect here is orthogonal to the cap story:

| kernel | live_before | closure_unions | reduction | cost_before → cost_after |
|---|---|---|---|---|
| glyph16/32:U+0027 (apostrophe) | 582 | 61 | 10.48% | 171 → 152 (**−11.1%**) |
| glyph16/32:U+002F, U+005C (`/`, `\`) | 447 | 37 | 8.28% | 166 → 147 (**−11.4%**) |
| glyph16/32:U+0022 (quote) | 421 | 32 | 7.60% | 310 → 286 (**−7.7%**) |

These are real, measurable extraction-quality wins the defect costs today —
just on tiny kernels, not the large ones that trip the cap.

## H-b: does rule order change the missing-congruence count?

Cheap check on one kernel per category present: production `all_rules()`
order vs. the pinned numeric-first static reorder
(`docs/results/2026-09-01-rule-order-real-kernels.md`'s
`NUMERIC_FIRST_ORDER`), same production budget, same offline closure.

| kernel | order | live_before | closure_unions | reduction_frac |
|---|---|---|---|---|
| cellgrid:80x24_d1 | production | 1260 | 7 | 0.56% |
| cellgrid:80x24_d1 | numeric-first | 1083 | 3 | 0.28% |
| shader:cosine_palette | production | 474 | 0 | 0.00% |
| shader:cosine_palette | numeric-first | 434 | 1 | 0.23% |
| psychedelic | production | 946 | 8 | 0.85% |
| psychedelic | numeric-first | 1267 | 43 | **3.39%** |
| glyph16:U+0041 | production | 1340 | 0 | 0.00% |
| glyph16:U+0041 | numeric-first | 1511 | 116 | **7.68%** |

Order does change the missing-congruence count, materially on 2/4 sampled
kernels (psychedelic: +2.5pp; glyph U+0041: +7.7pp) and negligibly on the
other 2. But the direction argues **against** the "the order that looks
better is just the one that stumbles into more congruence" reframing of
#1101/#1088: on both kernels where the two orders differ substantially,
**numeric-first leaves *more* congruence on the table** (more closure_unions,
higher reduction_frac) than production order — yet #1101 found numeric-first
best-or-tied on real-kernel anytime cost at every checkpoint. If
numeric-first's advantage were explained by incidentally achieving more
congruence, we'd expect the opposite sign here. n=4 is too small to
generalize, but on this sample H-b's specific mechanism (order effect ==
congruence-completeness effect) is not what's driving #1101's result.

## Verdict

Upward congruence closure is a real, measurable, sound (never regresses
extracted cost) improvement — but on this corpus it is **small** (922
unions total, 0.24% of live classes; median per-kernel effect is exactly
zero) and **does not explain** why the 5,000-class cap binds on most real
kernels (the cap trips on raw allocation growth that is mostly genuine
non-duplicate structure, not unrecognized congruence — only 2/231 cap-hit
kernels have a live class count that even reaches the cap, and closure
rescues neither). It is **not worth implementing as a fix for the cap/H-a
story**; it may still be worth a small, targeted fix (an e-node parent
index, upward-merging on `union`) purely for the ~5–11% extraction-cost wins
it recovers on small kernels — but that is a much narrower case than "this
is why the cap binds."

## Raw data

`2026-09-02-missing-congruence.csv` / `.json` carry every kernel's row
(`live_before`, `live_after`, `closure_unions`, `max_classes`, `hit_class_cap`,
`cost_before`, `cost_after`, ...) — enough to independently reproduce every
number and table above, including the cap-correction analysis.
