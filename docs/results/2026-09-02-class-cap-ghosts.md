# Class-cap ghosts: allocated vs live e-classes (2026-09-02)

`EGraph::union` (graph.rs) merges through the union-find and never removes an entry from `self.classes`. `num_classes()` returns `self.classes.len()` — ALLOCATED classes, monotonically increasing. The live graph is `class_ids()` — canonical AND non-empty. The production budget check (`self.classes.len() > max_classes`, `saturate_bounded`) counts allocated, not live.

Corpus: 206 real kernels (190 glyph + 13 shader/psychedelic + 3 cell-grid) + 200 size-stratified synthetic expressions (5 depth bands x 40 seeds) = 406 total. Production regime: `Optimizer::production()` exactly as `pixelflow-search/src/runtime.rs`'s `optimize_runtime_arena_uncached` calls it. Cost model: `CostModel::latency_prior()`.

## Headline numbers

1. **Allocated/live ratio** (whole corpus, n=406): median **2.61x**, q1=1.81x, q3=4.91x, worst=10.07x.
2. **Cap-hit kernels**: 231/406 (56.9%). Of those, **163** (70.6%) are ghost-bound — live < cap/2 at the moment the cap tripped.
3. **Upper-bound recovery** — re-running each cap-hit kernel with the cap raised to `5000 * (allocated/live ratio observed at the original cap-hit)`, extracted-cost improvement over the 5000-class-cap baseline (n=231): median **2.348%**, p90 **17.433%**, mean 4.908%. 167/231 cap-hit kernels improved; 212/231 still hit the raised cap.
4. **Memory cost of the raised cap** (proxy: end-of-run allocated/nodes/memo — all three are monotonically non-decreasing across a single saturation run: `add()` only appends to `classes`, `union()` never removes an entry, and `rebuild_budgeted`'s `memo.insert` never removes a key either — so end-of-run equals peak for a single run). Worst-case allocated-class growth from raising the cap: **12.27x**.

## The honest tension: memory protection vs. quality

The cap's stated purpose (`graph.rs:1050`, "Bounds memory against e-graph blowup") is memory protection, and memory is consumed by ALLOCATED classes — every `EClass` (a `Vec<ENode>` + parallel `Vec<ENodeId>`) that `add()` ever pushed stays in `self.classes` forever, whether or not it is still canonical. Counting allocated for a *memory* ceiling is defensible. Counting allocated for a *quality* ceiling — "stop searching once you've found 5,000 equalities" — is not: it stops the search once it has *allocated* 5,000 classes, most of which (median 61.8%, `1 - 1/2.61`) are ghosts of classes already merged away. The two purposes want different counters, and the current code has only one.

## A second, sharper finding: raising the cap is not durable

The upper-bound recovery experiment (raise the cap to `5000 * ratio-at-original-cap-hit`, rerun) answers a second question the task didn't originally ask for but the data forced: **is a one-shot cap raise even a stable fix?** No. **212 of 231 (91.8%)** cap-hit kernels *still* hit the raised cap. The allocated/live ratio is not a per-kernel constant — it grows as saturation runs longer (more rounds allocate more ghosts before the live count catches up), so the ratio measured *at* the original 5,000-class cap systematically **under**-estimates the ratio needed to reach quiescence. Concretely, for `cellgrid:120x40_d2`: base ratio 3.63x (live 1,261 of 4,583 allocated) → raised cap 18,172 → the rerun allocates 16,796 classes for only 3,560 live (a *new* ratio of 4.72x) and hits `ClassCap` again with **zero** cost improvement. A static multiplier is a bigger room, not a fix — the ghosts refill it. This is direct evidence against (c) "raise the cap" as anything but a stopgap, and it is the strongest argument *for* (b) over any fixed-ratio tuning: a live-counted budget doesn't need to guess a multiplier, because it is checking the quantity that actually matters at every step, so it can't be out-grown by a drifting ratio the way a static raise can.

## Three candidate fixes

**(a) Compact** — periodically free merged/empty classes and remap ids. This reclaims memory *and* raises the effective budget in one move (post-compaction, allocated ≈ live, so a 5,000-class cap becomes a true 5,000-*live*-class cap with no multiplier to guess). The cost is a full id remap, and the remap surface is large — see the enumeration below. The riskiest holder is `Provenance`: `ApplicationRecord.match_root` and `UnionEvent.class_a/class_b` are `EClassId`s captured *during* saturation and read only *after* it ends (by the observer callback and, on other branches, the hindsight labeler). A compaction that runs mid-saturation (the only place it would earn its cost — compacting only between runs helps nothing, since a single kernel's ghosts are born and die within one run) must actively rewrite every already-recorded `EClassId` in that log through the compaction's id map, not just the live graph structures, or the log silently points at stale/reused ids the moment class 47 gets freed and a *different* class is later minted at index 47. That is exactly the kind of "escape hatch that becomes a bug hunt" this codebase's CLAUDE.md warns about: nothing panics, the log just quietly mis-attributes credit or corrupts `derivation_ancestors`' ancestry walk (`provenance.rs`) into following a stale id into an unrelated node. NO SILENT FAILURES makes this a hard requirement, not a nice-to-have, and it makes (a) a redesign of the provenance/observer contract, not a patch to `union`.

**(b) Live-counted budget** — an incrementally maintained live-class counter, decremented exactly where `union()` performs an actual merge (the one place a class becomes non-canonical in this codebase — `rebuild_budgeted`'s congruence-closure merges route through the same `union()` call, so there is exactly one update site), incremented in `add()`'s not-in-memo branch. Both are O(1): no scan, no new allocation, a single `usize` field alongside `applications: u64`. `debug_assert_eq!(self.live_count, self.class_ids().count())` in tests (and in any harness that already clones/asserts on the graph, e.g. `congruence_gap_probe`) closes the drift risk the task asked to rule out — the invariant is a one-line addition next to an existing O(n) ground truth. Budget the production tiers' 500/2000/5000 against this counter instead of `classes.len()`, and keep a *separate*, higher ceiling (`HARD_CLASS_LIMIT = 100,000` already exists and already serves exactly this role for allocated growth) for memory. This is the smaller change: one field, two update sites, one debug assertion, and the existing `saturate_bounded` cap check swaps which counter it reads. It also doesn't have the provenance problem — nothing about the graph's *identity* changes, only which counter the budget check reads, so every `EClassId` any holder is caching stays valid.

**(c) Raise the cap (tuning only)** — measured above as the baseline for comparison, and the recovery experiment shows it is not even a durable tuning: 91.8% of cap-hit kernels re-hit a 5,000x-ratio-raised cap. Not recommended as a fix in its own right.

## Recommendation: (b), live-counted budget + allocated ceiling

The numbers point at (b):

- It directly fixes the measured defect (the budget check counts the wrong population) without guessing a multiplier — and the recovery experiment shows guessing a multiplier doesn't hold up (91.8% re-hit).
- It is O(1) on the hot path by construction (single decrement at `union`'s one merge site, single increment at `add`'s one insert site), so it costs nothing where `saturate_bounded`'s inner loop already checks a counter every iteration.
- It cannot drift silently: `debug_assert_eq!(live_count, class_ids().count())` is one line, checked against the existing O(n) definition of live, satisfying the "prove it" requirement directly rather than by argument.
- It does not touch `EGraph`'s identity (no id remap), so every holder enumerated below — `Provenance`, `deps.rs`'s `DepsAnalysis`, `optimize.rs`'s `var_to_eclass`, every measurement harness — keeps working unmodified. Compaction (a) would require auditing and likely rewriting most of them.
- Memory: (b) budgets against live classes directly, so for a kernel whose ghosts are proportionally as bad as this corpus's worst case (12.27x observed allocated growth when the cap was raised), reaching a true 5,000-*live*-class search on that kernel could allocate on the order of ~60,000 classes before stopping — well inside `HARD_CLASS_LIMIT` (100,000), but a real multi-x increase in peak per-kernel compile memory versus today's ≤5,000-allocated ceiling. That is the cost (b) has to be evaluated against, and it is the number a reviewer should ask for before shipping it: **is a ~12x worst-case memory increase, bounded by the existing 100k hard ceiling, acceptable for the median ~60%+ live-class recovery this corpus shows?** This measurement does not answer that question — it hands the number to whoever does.

(a) compact is not ruled out — it strictly dominates (b) on memory (bounded allocated ≈ live at all times, rather than allowing up to 12x growth before the hard ceiling), and it is the more "correct" long-term shape (the graph's storage matches its live content). But it is a larger, riskier change: the Provenance log crossing a mid-saturation compaction boundary needs active id-rebasing to avoid the exact silent-corruption failure mode CLAUDE.md forbids, and every other holder below needs the same audit. Recommend (b) now, as the surgical fix the measurement justifies; leave (a) as a follow-up if the ~12x worst-case memory cost of (b) turns out to matter in practice (the raised-cap CSV/JSON rows here are exactly the data to re-check that against once (b) ships).

## EClassId/ENodeId holders — what a compaction (a) would have to rebase

Enumerated by reading every non-test, non-transient use of `EClassId`/`ENodeId` in `pixelflow-search` and its call sites. "Transient" (produced and consumed within one function call on a graph that doesn't mutate in between, e.g. every extraction/DAG-building routine in `extract.rs`, every `RewriteAction` variant in `rewrite.rs`, `deps.rs`'s internal BFS) is *not* listed — a compaction between calls is fine as long as nothing holds an id across the boundary, and these don't.

**Structures a compaction must rebase, in the `EGraph` itself:**
- `classes: Vec<EClass>` and `parent: Vec<EClassId>` — the union-find and class storage themselves; compaction means literally re-indexing both, which is the mechanism, not a holder to fix elsewhere.
- Every `ENode::Op { children: Vec<EClassId>, .. }` inside every live `EClass.nodes` — every operator node's children are `EClassId`s; a compaction remap must rewrite every one, which also means…
- `memo: HashMap<ENode, EClassId>` — `ENode` (the key) embeds `EClassId` children, so remapping ids changes the key's hash. The whole table must be drained and reinserted under the new ids, not mutated in place.
- `const_fact: Vec<Option<u32>>` — indexed by class id like `classes`/`parent`; needs the same re-indexing.
- `worklist: Vec<EClassId>` — pending rebuild work; live only within a `rebuild_budgeted` call, but if compaction can run *while* a worklist is non-empty (between a `union()` and its `rebuild()`), it must rebase these too, or `rebuild_budgeted` finds ids that no longer exist.

**Structures OUTSIDE `EGraph` that hold ids across a call boundary:**
- **`Provenance`** (`provenance.rs`) — the highest-risk holder. `ApplicationRecord.match_root: EClassId` and `UnionEvent.class_a/class_b: EClassId` are recorded *during* saturation and read only after (`Optimizer::run`'s observer loop, and — on research branches — the hindsight labeler's `derivation_ancestors`). A mid-saturation compaction must actively rewrite every already-logged `EClassId` through the compaction's id map. `ENodeId` itself (`provenance.rs:60-61`, "never reused, never renumbered") is compaction-*safe* by construction — it's a mint counter, not a class-id-derived value — so `origins: HashMap<ENodeId, Origin>` and the two harness-side `ENodeId`-keyed maps in `guide_scope_saturation_delta.rs` need no rebasing.
- **`deps.rs`'s `DepsAnalysis { deps: HashMap<EClassId, Variance> }`** — a per-class analysis computed once (for uniform hoisting) and queried later via `.get(&egraph, id)`. Safe today because it's built and consumed after saturation ends, but if compaction ever ran between `DepsAnalysis::compute` and its later `.get()` calls, this map would go stale silently (a `HashMap` doesn't know its keys are dangling — it would just return `Variance::default()` or nothing for a remapped class, no panic). Needs invalidation-on-compaction if compaction becomes routine mid-pipeline, not just mid-saturation.
- **`optimize.rs`'s `EGraphContext.var_to_eclass: HashMap<String, EClassId>`** (pixelflow-compiler) — variable-name-to-class cache built while walking the source AST into the e-graph, read again for every later reference to the same variable in the same compile. Built and consumed within one compile's graph-building phase, before `Optimizer::run` is called, so safe under "compaction only runs inside saturation" — but a holder to catch in review if that assumption is ever relaxed.
- **`Optimized.choices: Vec<Option<usize>>`** (`optimizer.rs`) — indexed by class id, returned from `Optimizer::run` and consumed by `Optimized::to_arena(&egraph, root)` immediately after, on the *same* `egraph`. Safe as long as compaction runs only inside `saturate_bounded` (i.e., strictly before extraction reads `choices`), which is the natural place to put it — extraction already assumes a frozen post-saturation graph.
- **Measurement/training harnesses** (`guide_headroom.rs`, `guide_scope_saturation_delta.rs`, this file's `class_cap_ghosts` and `congruence_gap_probe` modules) — all compute on a *final* (post-`Optimizer::run`) graph in one pass and hold no id across a mutation boundary. Safe by the same "final graph, single call" pattern as `Optimized::to_arena`.
- **`egraph/candidate.rs`'s `ClassContentKey`/`CandidateKey`** (referenced in `docs/plans/2026-09-02-optimizer-api.md:257-260,738`) — **does not exist in this codebase.** `git grep` across every branch finds no `candidate.rs` and no `ClassContentKey`; the only `CandidateKey` hits are in the deleted `train_guide.rs`/`nnue/guide/mod.rs` (removed per the GuideNnue deletion). The design doc predates a refactor that removed this dedup machinery — nothing to enumerate here today, but worth flagging as a stale doc reference.

**Bottom line for (a):** the mechanism (rewriting `classes`/`parent`/`const_fact`/every node's children/`memo`) is a bounded, mechanical rewrite of the graph's own storage. The genuinely dangerous part is `Provenance` — the one structure designed to outlive individual saturation steps and be read by something else (an `Observer`, a hindsight labeler on other branches) — which is why (a) is scoped above as "a redesign of the provenance/observer contract," not a patch to `union`.

## Cap-hit kernels: raised-cap recovery (raw)

| kernel | category | nodes | base cap | ratio | raised cap | base cost | raised cost | improvement | still capped |
|---|---|---|---|---|---|---|---|---|---|
| glyph16:U+0078 | glyph | 343 | 5000 | 5.79x | 28952 | 639 | 431 | 32.55% | true |
| glyph32:U+0078 | glyph | 343 | 5000 | 5.79x | 28952 | 639 | 431 | 32.55% | true |
| glyph16:U+0022 | glyph | 199 | 5000 | 6.86x | 34299 | 310 | 214 | 30.97% | true |
| glyph32:U+0022 | glyph | 199 | 5000 | 6.86x | 34299 | 310 | 214 | 30.97% | true |
| glyph16:U+0058 | glyph | 343 | 5000 | 5.74x | 28718 | 652 | 452 | 30.67% | true |
| glyph32:U+0058 | glyph | 343 | 5000 | 5.74x | 28718 | 652 | 452 | 30.67% | true |
| glyph16:U+005E | glyph | 199 | 5000 | 6.88x | 34417 | 319 | 223 | 30.09% | true |
| glyph32:U+005E | glyph | 199 | 5000 | 6.88x | 34417 | 319 | 223 | 30.09% | true |
| glyph16:U+0059 | glyph | 263 | 5000 | 6.51x | 32573 | 437 | 317 | 27.46% | true |
| glyph32:U+0059 | glyph | 263 | 5000 | 6.51x | 32573 | 437 | 317 | 27.46% | true |
| glyph16:U+003C | glyph | 295 | 5000 | 6.42x | 32107 | 514 | 394 | 23.35% | false |
| glyph32:U+003C | glyph | 295 | 5000 | 6.42x | 32107 | 514 | 394 | 23.35% | false |
| glyph16:U+003E | glyph | 295 | 5000 | 6.44x | 32197 | 519 | 399 | 23.12% | false |
| glyph32:U+003E | glyph | 295 | 5000 | 6.44x | 32197 | 519 | 399 | 23.12% | false |
| glyph16:U+0027 | glyph | 127 | 5000 | 6.13x | 30636 | 171 | 133 | 22.22% | true |
| glyph32:U+0027 | glyph | 127 | 5000 | 6.13x | 30636 | 171 | 133 | 22.22% | true |
| glyph16:U+002F | glyph | 127 | 5000 | 7.70x | 38523 | 166 | 134 | 19.28% | true |
| glyph16:U+005C | glyph | 127 | 5000 | 7.70x | 38523 | 166 | 134 | 19.28% | true |
| glyph32:U+002F | glyph | 127 | 5000 | 7.70x | 38523 | 166 | 134 | 19.28% | true |
| glyph32:U+005C | glyph | 127 | 5000 | 7.70x | 38523 | 166 | 134 | 19.28% | true |
| glyph16:U+0049 | glyph | 391 | 5000 | 6.05x | 30248 | 626 | 506 | 19.17% | true |
| glyph32:U+0049 | glyph | 391 | 5000 | 6.05x | 30248 | 626 | 506 | 19.17% | true |
| glyph16:U+006C | glyph | 323 | 5000 | 6.44x | 32195 | 522 | 431 | 17.43% | true |
| glyph32:U+006C | glyph | 323 | 5000 | 6.44x | 32195 | 522 | 431 | 17.43% | true |
| glyph16:U+0037 | glyph | 191 | 5000 | 7.39x | 36931 | 288 | 240 | 16.67% | true |
| glyph32:U+0037 | glyph | 191 | 5000 | 7.40x | 37016 | 288 | 240 | 16.67% | true |
| glyph16:U+002A | glyph | 559 | 5000 | 4.47x | 22355 | 1143 | 1003 | 12.25% | true |
| glyph32:U+002A | glyph | 559 | 5000 | 4.47x | 22355 | 1143 | 1003 | 12.25% | true |
| glyph16:U+005A | glyph | 255 | 5000 | 7.32x | 36580 | 393 | 345 | 12.21% | true |
| glyph16:U+007A | glyph | 255 | 5000 | 7.32x | 36580 | 393 | 345 | 12.21% | true |
| glyph32:U+005A | glyph | 255 | 5000 | 7.33x | 36648 | 393 | 345 | 12.21% | true |
| glyph32:U+007A | glyph | 255 | 5000 | 7.32x | 36580 | 393 | 345 | 12.21% | true |
| shader:julia_set | shader | 122 | 5000 | 5.53x | 27674 | 716 | 631 | 11.87% | true |
| glyph16:U+004A | glyph | 2425 | 5000 | 2.29x | 11461 | 4339 | 3926 | 9.52% | true |
| glyph32:U+004A | glyph | 2425 | 5000 | 2.29x | 11461 | 4339 | 3926 | 9.52% | true |
| glyph16:U+0023 | glyph | 615 | 5000 | 5.22x | 26084 | 1014 | 920 | 9.27% | true |
| glyph32:U+0023 | glyph | 615 | 5000 | 5.22x | 26084 | 1014 | 920 | 9.27% | true |
| glyph16:U+0055 | glyph | 2899 | 5000 | 2.25x | 11270 | 5170 | 4701 | 9.07% | true |
| glyph32:U+0055 | glyph | 2899 | 5000 | 2.25x | 11270 | 5170 | 4701 | 9.07% | true |
| shader:smooth_min_scene | shader | 43 | 2000 | 7.62x | 38094 | 136 | 124 | 8.82% | true |
| glyph16:U+003B | glyph | 3494 | 5000 | 2.10x | 10522 | 6080 | 5609 | 7.75% | true |
| glyph32:U+003B | glyph | 3494 | 5000 | 2.10x | 10522 | 6080 | 5609 | 7.75% | true |
| glyph16:U+0047 | glyph | 4563 | 5000 | 1.85x | 9245 | 8262 | 7642 | 7.50% | true |
| glyph32:U+0047 | glyph | 4563 | 5000 | 1.85x | 9245 | 8262 | 7642 | 7.50% | true |
| glyph16:U+0079 | glyph | 3530 | 5000 | 2.02x | 10088 | 6403 | 5928 | 7.42% | true |
| glyph32:U+0079 | glyph | 3530 | 5000 | 2.02x | 10088 | 6403 | 5928 | 7.42% | true |
| glyph16:U+003A | glyph | 3655 | 5000 | 2.10x | 10497 | 6228 | 5774 | 7.29% | true |
| glyph32:U+003A | glyph | 3655 | 5000 | 2.10x | 10497 | 6228 | 5774 | 7.29% | true |
| glyph16:U+0035 | glyph | 4546 | 5000 | 1.86x | 9283 | 8145 | 7553 | 7.27% | true |
| glyph32:U+0035 | glyph | 4546 | 5000 | 1.86x | 9283 | 8145 | 7553 | 7.27% | true |
| glyph16:U+0042 | glyph | 4470 | 5000 | 1.88x | 9387 | 7970 | 7432 | 6.75% | true |
| glyph32:U+0042 | glyph | 4470 | 5000 | 1.88x | 9387 | 7970 | 7432 | 6.75% | true |
| glyph16:U+007D | glyph | 4598 | 5000 | 1.85x | 9269 | 8215 | 7663 | 6.72% | true |
| glyph32:U+007D | glyph | 4598 | 5000 | 1.85x | 9269 | 8215 | 7663 | 6.72% | true |
| glyph16:U+006A | glyph | 4293 | 5000 | 1.91x | 9571 | 7523 | 7021 | 6.67% | true |
| glyph32:U+006A | glyph | 4293 | 5000 | 1.91x | 9571 | 7523 | 7021 | 6.67% | true |
| glyph16:U+007B | glyph | 4598 | 5000 | 1.85x | 9250 | 8205 | 7663 | 6.61% | true |
| glyph32:U+007B | glyph | 4598 | 5000 | 1.85x | 9250 | 8205 | 7663 | 6.61% | true |
| synth_d11_s23 | synthetic | 221 | 5000 | 4.35x | 21757 | 2569 | 2403 | 6.46% | true |
| glyph16:U+0032 | glyph | 3891 | 5000 | 1.95x | 9761 | 6980 | 6539 | 6.32% | true |
| glyph32:U+0032 | glyph | 3891 | 5000 | 1.95x | 9761 | 6980 | 6539 | 6.32% | true |
| glyph16:U+0044 | glyph | 2606 | 5000 | 2.26x | 11296 | 4506 | 4230 | 6.13% | true |
| glyph32:U+0044 | glyph | 2606 | 5000 | 2.26x | 11296 | 4506 | 4230 | 6.13% | true |
| glyph16:U+0052 | glyph | 2710 | 5000 | 2.25x | 11273 | 4725 | 4443 | 5.97% | true |
| glyph32:U+0052 | glyph | 2710 | 5000 | 2.25x | 11273 | 4725 | 4443 | 5.97% | true |
| glyph16:U+0068 | glyph | 2931 | 5000 | 2.27x | 11328 | 5045 | 4751 | 5.83% | true |
| glyph32:U+0068 | glyph | 2931 | 5000 | 2.27x | 11328 | 5045 | 4751 | 5.83% | true |
| glyph16:U+0069 | glyph | 2123 | 5000 | 3.25x | 16244 | 3510 | 3308 | 5.75% | true |
| glyph32:U+0069 | glyph | 2123 | 5000 | 3.25x | 16244 | 3510 | 3308 | 5.75% | true |
| glyph16:U+0077 | glyph | 3663 | 5000 | 1.98x | 9906 | 6602 | 6225 | 5.71% | true |
| glyph32:U+0077 | glyph | 3663 | 5000 | 1.98x | 9906 | 6602 | 6225 | 5.71% | true |
| glyph16:U+006D | glyph | 5289 | 5000 | 1.75x | 8750 | 9624 | 9128 | 5.15% | true |
| glyph32:U+006D | glyph | 5289 | 5000 | 1.75x | 8750 | 9624 | 9128 | 5.15% | true |
| glyph16:U+0066 | glyph | 2364 | 5000 | 2.90x | 14507 | 3955 | 3754 | 5.08% | true |
| glyph32:U+0066 | glyph | 2364 | 5000 | 2.90x | 14507 | 3955 | 3754 | 5.08% | true |
| shader:domain_warp_fbm | shader | 84 | 5000 | 3.68x | 18382 | 444 | 422 | 4.95% | true |
| glyph16:U+0072 | glyph | 2456 | 5000 | 2.72x | 13614 | 4180 | 3977 | 4.86% | true |
| glyph32:U+0072 | glyph | 2456 | 5000 | 2.72x | 13614 | 4180 | 3977 | 4.86% | true |
| glyph16:U+0074 | glyph | 2421 | 5000 | 2.73x | 13655 | 4109 | 3912 | 4.79% | true |
| glyph32:U+0074 | glyph | 2421 | 5000 | 2.73x | 13655 | 4109 | 3912 | 4.79% | true |
| glyph16:U+0075 | glyph | 2493 | 5000 | 2.71x | 13538 | 4153 | 3955 | 4.77% | true |
| glyph32:U+0075 | glyph | 2493 | 5000 | 2.71x | 13538 | 4153 | 3955 | 4.77% | true |
| glyph16:U+0050 | glyph | 2409 | 5000 | 2.85x | 14236 | 3990 | 3801 | 4.74% | true |
| glyph32:U+0050 | glyph | 2409 | 5000 | 2.85x | 14236 | 3990 | 3801 | 4.74% | true |
| glyph16:U+006E | glyph | 2264 | 5000 | 2.96x | 14795 | 3759 | 3592 | 4.44% | true |
| glyph32:U+006E | glyph | 2264 | 5000 | 2.96x | 14795 | 3759 | 3592 | 4.44% | true |
| glyph16:U+006B | glyph | 1460 | 5000 | 3.42x | 17075 | 2517 | 2409 | 4.29% | true |
| glyph32:U+006B | glyph | 1460 | 5000 | 3.42x | 17075 | 2517 | 2409 | 4.29% | true |
| glyph16:U+002E | glyph | 1855 | 5000 | 3.46x | 17286 | 2881 | 2768 | 3.92% | true |
| glyph32:U+002E | glyph | 1855 | 5000 | 3.46x | 17286 | 2881 | 2768 | 3.92% | true |
| glyph16:U+004D | glyph | 2345 | 5000 | 2.89x | 14427 | 4133 | 3977 | 3.77% | true |
| glyph32:U+004D | glyph | 2345 | 5000 | 2.89x | 14427 | 4133 | 3977 | 3.77% | true |
| glyph16:U+0021 | glyph | 1927 | 5000 | 3.48x | 17393 | 3125 | 3012 | 3.62% | true |
| glyph32:U+0021 | glyph | 1927 | 5000 | 3.48x | 17393 | 3125 | 3012 | 3.62% | true |
| glyph16:U+0060 | glyph | 1722 | 5000 | 3.49x | 17470 | 2827 | 2725 | 3.61% | true |
| glyph32:U+0060 | glyph | 1722 | 5000 | 3.49x | 17470 | 2827 | 2725 | 3.61% | true |
| glyph16:U+0056 | glyph | 1573 | 5000 | 3.50x | 17491 | 2647 | 2552 | 3.59% | true |
| glyph32:U+0056 | glyph | 1573 | 5000 | 3.50x | 17491 | 2647 | 2552 | 3.59% | true |
| glyph16:U+0039 | glyph | 8267 | 5000 | 1.32x | 6624 | 15698 | 15139 | 3.56% | true |
| glyph32:U+0039 | glyph | 8267 | 5000 | 1.32x | 6624 | 15698 | 15139 | 3.56% | true |
| glyph16:U+0031 | glyph | 1452 | 5000 | 3.34x | 16697 | 2520 | 2438 | 3.25% | true |
| glyph32:U+0031 | glyph | 1452 | 5000 | 3.34x | 16697 | 2520 | 2438 | 3.25% | true |
| glyph16:U+002C | glyph | 1694 | 5000 | 3.42x | 17123 | 2779 | 2695 | 3.02% | true |
| glyph32:U+002C | glyph | 1694 | 5000 | 3.42x | 17123 | 2779 | 2695 | 3.02% | true |
| glyph16:U+0034 | glyph | 1637 | 5000 | 3.41x | 17042 | 2751 | 2671 | 2.91% | true |
| glyph32:U+0034 | glyph | 1637 | 5000 | 3.41x | 17042 | 2751 | 2671 | 2.91% | true |
| glyph16:U+0041 | glyph | 1605 | 5000 | 3.36x | 16825 | 2672 | 2598 | 2.77% | true |
| glyph32:U+0041 | glyph | 1605 | 5000 | 3.36x | 16825 | 2672 | 2598 | 2.77% | true |
| glyph16:U+004E | glyph | 1399 | 5000 | 3.51x | 17558 | 2457 | 2390 | 2.73% | true |
| glyph32:U+004E | glyph | 1399 | 5000 | 3.51x | 17558 | 2457 | 2390 | 2.73% | true |
| glyph16:U+0036 | glyph | 8067 | 5000 | 1.37x | 6825 | 15249 | 14836 | 2.71% | true |
| glyph32:U+0036 | glyph | 8067 | 5000 | 1.37x | 6825 | 15249 | 14836 | 2.71% | true |
| glyph16:U+0048 | glyph | 247 | 5000 | 7.57x | 37846 | 314 | 306 | 2.55% | true |
| glyph32:U+0048 | glyph | 247 | 5000 | 7.59x | 37935 | 314 | 306 | 2.55% | true |
| glyph16:U+007E | glyph | 5093 | 5000 | 1.78x | 8883 | 9370 | 9150 | 2.35% | true |
| glyph32:U+007E | glyph | 5093 | 5000 | 1.78x | 8883 | 9370 | 9150 | 2.35% | true |
| glyph16:U+0028 | glyph | 1855 | 5000 | 3.42x | 17110 | 3039 | 2968 | 2.34% | true |
| glyph16:U+0029 | glyph | 1855 | 5000 | 3.42x | 17119 | 3039 | 2968 | 2.34% | true |
| glyph32:U+0028 | glyph | 1855 | 5000 | 3.42x | 17110 | 3039 | 2968 | 2.34% | true |
| glyph32:U+0029 | glyph | 1855 | 5000 | 3.42x | 17119 | 3039 | 2968 | 2.34% | true |
| glyph16:U+0071 | glyph | 5046 | 5000 | 1.78x | 8908 | 9172 | 8958 | 2.33% | true |
| glyph32:U+0071 | glyph | 5046 | 5000 | 1.78x | 8908 | 9172 | 8958 | 2.33% | true |
| glyph16:U+0043 | glyph | 5129 | 5000 | 1.77x | 8827 | 9459 | 9252 | 2.19% | true |
| glyph32:U+0043 | glyph | 5129 | 5000 | 1.77x | 8827 | 9459 | 9252 | 2.19% | true |
| glyph16:U+0062 | glyph | 5129 | 5000 | 1.77x | 8844 | 9434 | 9228 | 2.18% | true |
| glyph32:U+0062 | glyph | 5129 | 5000 | 1.77x | 8844 | 9434 | 9228 | 2.18% | true |
| glyph16:U+006F | glyph | 5716 | 5000 | 1.70x | 8483 | 10262 | 10042 | 2.14% | true |
| glyph32:U+006F | glyph | 5716 | 5000 | 1.70x | 8483 | 10262 | 10042 | 2.14% | true |
| glyph16:U+004F | glyph | 5945 | 5000 | 1.66x | 8317 | 10769 | 10539 | 2.14% | true |
| glyph32:U+004F | glyph | 5945 | 5000 | 1.66x | 8317 | 10769 | 10539 | 2.14% | true |
| glyph16:U+0030 | glyph | 5402 | 5000 | 1.72x | 8612 | 9970 | 9768 | 2.03% | true |
| glyph32:U+0030 | glyph | 5402 | 5000 | 1.72x | 8612 | 9970 | 9768 | 2.03% | true |
| glyph16:U+0051 | glyph | 6053 | 5000 | 1.65x | 8254 | 10749 | 10532 | 2.02% | true |
| glyph32:U+0051 | glyph | 6053 | 5000 | 1.65x | 8254 | 10749 | 10532 | 2.02% | true |
| glyph16:U+0070 | glyph | 5406 | 5000 | 1.72x | 8616 | 9922 | 9722 | 2.02% | true |
| glyph32:U+0070 | glyph | 5406 | 5000 | 1.72x | 8616 | 9922 | 9722 | 2.02% | true |
| glyph16:U+0057 | glyph | 7232 | 5000 | 1.53x | 7643 | 13499 | 13231 | 1.99% | true |
| glyph32:U+0057 | glyph | 7232 | 5000 | 1.53x | 7643 | 13499 | 13231 | 1.99% | true |
| glyph16:U+0064 | glyph | 5864 | 5000 | 1.65x | 8273 | 10763 | 10560 | 1.89% | true |
| glyph32:U+0064 | glyph | 5864 | 5000 | 1.65x | 8273 | 10763 | 10560 | 1.89% | true |
| glyph16:U+0061 | glyph | 5743 | 5000 | 1.66x | 8314 | 10612 | 10414 | 1.87% | true |
| glyph32:U+0061 | glyph | 5743 | 5000 | 1.66x | 8314 | 10612 | 10414 | 1.87% | true |
| glyph16:U+0024 | glyph | 7185 | 5000 | 1.54x | 7702 | 13131 | 12898 | 1.77% | true |
| glyph32:U+0024 | glyph | 7185 | 5000 | 1.54x | 7702 | 13131 | 12898 | 1.77% | true |
| glyph16:U+0026 | glyph | 9420 | 5000 | 1.16x | 5803 | 17979 | 17674 | 1.70% | true |
| glyph32:U+0026 | glyph | 9420 | 5000 | 1.16x | 5803 | 17979 | 17674 | 1.70% | true |
| glyph16:U+0025 | glyph | 7327 | 5000 | 1.57x | 7847 | 12640 | 12430 | 1.66% | true |
| glyph32:U+0025 | glyph | 7327 | 5000 | 1.57x | 7847 | 12640 | 12430 | 1.66% | true |
| glyph16:U+003F | glyph | 6716 | 5000 | 1.57x | 7872 | 12176 | 11985 | 1.57% | true |
| glyph32:U+003F | glyph | 6716 | 5000 | 1.57x | 7872 | 12176 | 11985 | 1.57% | true |
| glyph16:U+0073 | glyph | 6929 | 5000 | 1.56x | 7810 | 12680 | 12487 | 1.52% | true |
| glyph32:U+0073 | glyph | 6929 | 5000 | 1.56x | 7810 | 12680 | 12487 | 1.52% | true |
| glyph16:U+0033 | glyph | 6993 | 5000 | 1.56x | 7791 | 12797 | 12605 | 1.50% | true |
| glyph32:U+0033 | glyph | 6993 | 5000 | 1.56x | 7791 | 12797 | 12605 | 1.50% | true |
| glyph16:U+0067 | glyph | 7009 | 5000 | 1.56x | 7793 | 12799 | 12607 | 1.50% | true |
| glyph32:U+0067 | glyph | 7009 | 5000 | 1.56x | 7793 | 12799 | 12607 | 1.50% | true |
| shader:cosine_palette | shader | 40 | 2000 | 3.73x | 18639 | 296 | 292 | 1.35% | true |
| glyph16:U+0053 | glyph | 7616 | 5000 | 1.46x | 7323 | 14066 | 13957 | 0.77% | true |
| glyph32:U+0053 | glyph | 7616 | 5000 | 1.46x | 7323 | 14066 | 13957 | 0.77% | true |
| glyph16:U+0038 | glyph | 9774 | 5000 | 1.13x | 5656 | 18500 | 18400 | 0.54% | true |
| glyph32:U+0038 | glyph | 9774 | 5000 | 1.13x | 5656 | 18500 | 18400 | 0.54% | true |
| synth_d11_s18 | synthetic | 221 | 5000 | 5.37x | 26873 | 2144 | 2140 | 0.19% | true |
| glyph16:U+0063 | glyph | 4671 | 5000 | 1.85x | 9230 | 8419 | 8409 | 0.12% | true |
| glyph32:U+0063 | glyph | 4671 | 5000 | 1.85x | 9230 | 8419 | 8409 | 0.12% | true |
| glyph16:U+0065 | glyph | 4872 | 5000 | 1.81x | 9068 | 8766 | 8756 | 0.11% | true |
| glyph32:U+0065 | glyph | 4872 | 5000 | 1.81x | 9068 | 8766 | 8756 | 0.11% | true |
| synth_d11_s39 | synthetic | 475 | 5000 | 4.32x | 21581 | 6791 | 6790 | 0.01% | true |
| cellgrid:120x40_d2 | cellgrid | 623 | 5000 | 3.63x | 18172 | 432 | 432 | 0.00% | true |
| cellgrid:80x24_d1 | cellgrid | 623 | 5000 | 3.64x | 18187 | 427 | 427 | 0.00% | true |
| cellgrid:80x24_d2 | cellgrid | 623 | 5000 | 3.63x | 18172 | 432 | 432 | 0.00% | true |
| glyph16:U+002B | glyph | 247 | 5000 | 8.42x | 42111 | 319 | 319 | 0.00% | false |
| glyph16:U+0040 | glyph | 12056 | 5000 | 1.00x | 5001 | 23527 | 23527 | 0.00% | true |
| glyph16:U+0045 | glyph | 247 | 5000 | 7.51x | 37568 | 335 | 335 | 0.00% | true |
| glyph16:U+0046 | glyph | 215 | 5000 | 7.66x | 38293 | 285 | 285 | 0.00% | true |
| glyph16:U+004C | glyph | 151 | 5000 | 7.96x | 39822 | 185 | 185 | 0.00% | true |
| glyph16:U+0054 | glyph | 183 | 5000 | 7.52x | 37618 | 227 | 227 | 0.00% | true |
| glyph16:U+005F | glyph | 119 | 5000 | 9.33x | 46636 | 129 | 129 | 0.00% | true |
| glyph32:U+002B | glyph | 247 | 5000 | 8.42x | 42111 | 319 | 319 | 0.00% | false |
| glyph32:U+0040 | glyph | 12056 | 5000 | 1.00x | 5001 | 23527 | 23527 | 0.00% | true |
| glyph32:U+0045 | glyph | 247 | 5000 | 7.53x | 37656 | 335 | 335 | 0.00% | true |
| glyph32:U+0046 | glyph | 215 | 5000 | 7.68x | 38394 | 285 | 285 | 0.00% | true |
| glyph32:U+004C | glyph | 151 | 5000 | 7.98x | 39905 | 185 | 185 | 0.00% | true |
| glyph32:U+0054 | glyph | 183 | 5000 | 7.54x | 37687 | 227 | 227 | 0.00% | true |
| glyph32:U+005F | glyph | 119 | 5000 | 9.33x | 46636 | 129 | 129 | 0.00% | true |
| shader:plasma | shader | 41 | 2000 | 5.42x | 27113 | 363 | 363 | 0.00% | true |
| shader:star_sdf | shader | 66 | 5000 | 6.06x | 30285 | 169 | 169 | 0.00% | true |
| shader:torus_slice | shader | 42 | 2000 | 7.78x | 38886 | 141 | 141 | 0.00% | false |
| synth_d3_s23 | synthetic | 35 | 2000 | 5.12x | 25586 | 132 | 132 | 0.00% | true |
| synth_d5_s11 | synthetic | 81 | 5000 | 10.07x | 50358 | 851 | 851 | 0.00% | false |
| synth_d7_s8 | synthetic | 122 | 5000 | 8.18x | 40912 | 1387 | 1387 | 0.00% | false |
| synth_d7_s11 | synthetic | 73 | 5000 | 7.31x | 36563 | 970 | 970 | 0.00% | false |
| synth_d7_s13 | synthetic | 217 | 5000 | 4.92x | 24624 | 2386 | 2386 | 0.00% | true |
| synth_d7_s14 | synthetic | 72 | 5000 | 9.21x | 46039 | 680 | 680 | 0.00% | false |
| synth_d7_s17 | synthetic | 137 | 5000 | 6.99x | 34937 | 1803 | 1803 | 0.00% | true |
| synth_d7_s19 | synthetic | 105 | 5000 | 8.81x | 44052 | 1931 | 1931 | 0.00% | false |
| synth_d7_s23 | synthetic | 122 | 5000 | 7.28x | 36423 | 2351 | 2351 | 0.00% | false |
| synth_d7_s26 | synthetic | 123 | 5000 | 4.68x | 23390 | 1786 | 1786 | 0.00% | true |
| synth_d7_s28 | synthetic | 146 | 5000 | 7.87x | 39366 | 2311 | 2311 | 0.00% | false |
| synth_d7_s31 | synthetic | 135 | 5000 | 5.66x | 28304 | 1582 | 1582 | 0.00% | true |
| synth_d9_s0 | synthetic | 318 | 5000 | 4.08x | 20421 | 4257 | 4257 | 0.00% | true |
| synth_d9_s8 | synthetic | 188 | 5000 | 3.97x | 19853 | 1946 | 1946 | 0.00% | true |
| synth_d9_s24 | synthetic | 206 | 5000 | 7.39x | 36943 | 3157 | 3157 | 0.00% | false |
| synth_d11_s1 | synthetic | 206 | 5000 | 6.06x | 30288 | 2962 | 2962 | 0.00% | true |
| synth_d11_s5 | synthetic | 456 | 5000 | 4.72x | 23577 | 6816 | 6816 | 0.00% | true |
| synth_d11_s8 | synthetic | 372 | 5000 | 4.88x | 24376 | 4916 | 4916 | 0.00% | true |
| synth_d11_s15 | synthetic | 181 | 5000 | 5.83x | 29166 | 2975 | 2975 | 0.00% | true |
| synth_d11_s16 | synthetic | 135 | 5000 | 5.59x | 27934 | 1729 | 1729 | 0.00% | true |
| synth_d11_s20 | synthetic | 252 | 5000 | 3.87x | 19365 | 3283 | 3283 | 0.00% | true |
| synth_d11_s24 | synthetic | 495 | 5000 | 4.56x | 22824 | 7514 | 7514 | 0.00% | true |
| synth_d11_s27 | synthetic | 289 | 5000 | 6.81x | 34036 | 4665 | 4665 | 0.00% | false |
| synth_d11_s29 | synthetic | 808 | 5000 | 4.15x | 20735 | 10296 | 10296 | 0.00% | false |
| synth_d11_s31 | synthetic | 279 | 5000 | 4.89x | 24426 | 3763 | 3763 | 0.00% | true |
| synth_d11_s32 | synthetic | 414 | 5000 | 5.42x | 27091 | 5595 | 5595 | 0.00% | true |
| synth_d11_s37 | synthetic | 216 | 5000 | 5.03x | 25162 | 3496 | 3496 | 0.00% | true |
| synth_d9_s29 | synthetic | 371 | 5000 | 4.24x | 21181 | 4999 | 5000 | -0.02% | true |
| synth_d11_s7 | synthetic | 454 | 5000 | 5.29x | 26426 | 7254 | 7258 | -0.06% | true |
| synth_d11_s3 | synthetic | 835 | 5000 | 3.83x | 19162 | 11719 | 11728 | -0.08% | true |
| synth_d11_s12 | synthetic | 329 | 5000 | 4.97x | 24855 | 5020 | 5024 | -0.08% | true |
| synth_d9_s32 | synthetic | 318 | 5000 | 5.39x | 26925 | 4175 | 4179 | -0.10% | true |
| synth_d11_s34 | synthetic | 265 | 5000 | 5.62x | 28119 | 3781 | 3786 | -0.13% | false |
| synth_d7_s12 | synthetic | 93 | 5000 | 7.24x | 36206 | 1112 | 1115 | -0.27% | true |
| synth_d9_s13 | synthetic | 297 | 5000 | 4.90x | 24502 | 4618 | 4641 | -0.50% | true |
| synth_d9_s25 | synthetic | 764 | 5000 | 3.24x | 16219 | 10496 | 10557 | -0.58% | true |
| shader:mandelbrot_distance | shader | 152 | 5000 | 8.86x | 44313 | 576 | 595 | -3.30% | false |
| synth_d11_s14 | synthetic | 636 | 5000 | 3.66x | 18306 | 7675 | 7974 | -3.90% | true |
| glyph16:U+004B | glyph | 617 | 5000 | 4.95x | 24746 | 1047 | 1106 | -5.64% | true |
| glyph32:U+004B | glyph | 617 | 5000 | 4.95x | 24746 | 1047 | 1106 | -5.64% | true |
| psychedelic | psychedelic | 102 | 5000 | 5.12x | 25618 | 766 | 816 | -6.53% | true |
| glyph16:U+0076 | glyph | 1115 | 5000 | 3.58x | 17919 | 1760 | 1989 | -13.01% | true |
| glyph32:U+0076 | glyph | 1115 | 5000 | 3.58x | 17919 | 1760 | 1989 | -13.01% | true |
| shader:metaballs | shader | 62 | 5000 | 9.03x | 45175 | 158 | 230 | -45.57% | true |

## Raw data

See `2026-09-02-class-cap-ghosts.csv` / `.json` for every kernel's row (406 baseline rows + 231 recovery rows).
