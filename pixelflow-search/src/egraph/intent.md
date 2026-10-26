# pixelflow-search/src/egraph

**Is.** The optimizer's mechanism (DENOTATION §8): an e-graph whose classes are sets of terms equal in
meaning, grown by rewrite rules under a budget counted in rule applications, and read out by extraction
under a cost. `Optimizer::run` is its one entry; the five laws of §8 are its contract.

## Owns
- The store and the runner: `EGraph` (hash-consing, `union`, `rebuild`; a node arena whose index is the
  node's id and its provenance tag, a member list per class threaded through the arena (`next` per node,
  `head` and `tail` per class) and the live set the store restates at each write that can change it, the per-class
  constant and variance facts; beside them the memo, the worklist, the change log and the per-kind class index a
  rule's scan reads, which only saturation reads) and `FrozenGraph`, what `EGraph::freeze` moves out of it:
  everything extraction reads and nothing else, with no `add` and no `&mut self` method, so the graph the shape
  cache keeps cannot be added to. `Saturation` is the store paired with its `RuleSet` and every count denominated
  in that sweep. Rule identity (`RuleId`, `Fingerprint`) and admission (`Vocabulary`, `template_may_name`).
- The budget: `Budget` resolves to `Limits` (iterations, minted ids, applications); `SaturationConfig`
  picks the tier from the input.
- Extraction and cost: the tree and shared DP arms and their arbiter, `Selection`. `cost(If) = cost(m) +
  (cost(a)+cost(b))/2 + cost(select)`; a fold costs `n·body + (n−1)·combine`, plus its value bounds once;
  `n` is read off its node when its range is known (`ENode::known_range`, per node: after `HalveFold` a class
  holds two ranges), else off its bounds' chosen literals, else `Fold::ESTIMATED_TRIPS`; a surviving `Dwrt`
  costs `LEGALIZATION_PRICE`. The fold rewrites and the chain rule are here; other rule bodies are `crate::math`.
- The guided loop, provenance journal, labeler and growth telemetry: research, off in production.

## Never
- The sweep reads no clock: `Limits` holds no duration and `saturate_budgeted` takes none. The one
  clock read is `Optimizer::run`'s assertion afterwards.
- A guide orders and never mutates: `SaturationGuide::score_candidates` sees summaries, and `guided.rs`
  commits only through `apply_single_rule`.
- Unequal constants are never unioned: `EGraph::union` refuses, recording `refused_const_unions`.
- No rule names `Gather`, `Reduce`, `Shl` or `Shr`: a fold is its own node, a shift's count a `ShiftCount`.
- No estimate of how often a mask is true enters a cost. No register, lane, width or ISA tier is named.
- The journal is not in the budget's denominator: `Saturation::application_count` is unconditional.
- A reranker never scores a frozen graph: `Optimizer::extract_over` asserts there is none, because a reranker
  scores an `Extraction`, which lends an `&EGraph`, and the runtime tier sets none.

## Contract
`mod.rs` declares the modules and re-exports `Optimizer`, `Budget`, `Optimized`, `EGraph`, `Saturation`,
`Limits`, `SaturationStop`, `insert`, `Declined`, `Vocabulary`, `Rewrite`, `RewriteAction`, `RuleSet`,
`SaturationConfig`, `CostModel`, `CostFunction`, `extract_dag`, `ExtractedDAG`, `Provenance`.

## Enforced by
- Types: `Selection::of_map` refuses `Missing`, `OutOfBounds`, `Cyclic`; `LiveTable` cannot be indexed
  by an `EClassId`; `template_may_name` is an exhaustive match, so a new `OpKind` fails the build;
  `FrozenGraph` has no `&mut self` method, so nothing adds to a graph whose memo is gone.
- L1, L2: `graph.rs` tests `over_budget_growth_cannot_assert_a_false_equality` and
  `saturation_at_any_budget_never_removes_an_equality`. L3 well-foundedness: `the_dp_map_is_well_founded`.
- Budget: `tests/production_budget_determinism.rs`, `tests/saturation_stop.rs`,
  `tests/saturation_ceiling_env.rs`; constants: `tests/fold_exactness.rs`; rules and vocabulary:
  `tests/root_shapes.rs`, `tests/ir_insert.rs`; cost: `tests/latency_prior_regression.rs`.
- `scripts/check-provenance-journal-scope.sh` (the journal types exist only under `provenance-journal`);
  desloppify `control-plane-64-bit` (`EClassId` is `u32`, its measurement in `node.rs`).

## Behind
- `optimizer.rs` line 34 cites a `laws.rs` that is not in the tree; `guided.rs` cites an L4 test with a
  reversing guide that does not exist; the plan's `optimizer_laws.rs` was deleted. L4 has no test.
- L5's argmin half is violated and two docs lag the code: DENOTATION §8 and the `Optimizer::cost` doc say
  sharing is never priced, but `extract_dag_over` also runs `shared_dag_dp_pass` and keeps the cheaper by
  `dag_cost`. Both arms are greedy (BACKLOG E5); a fold over a known range ties its unrolling and insertion
  order decides (E6); `Reranker` has no implementation outside tests.
- A value bound's pick is unconstrained: a class read as a bound can settle on a member that reads a coordinate,
  and `ExprArena::push_reduce` refuses it by name; pinned in `fold_rules.rs`, left loud until a kernel hits it.
- No test pins `cost(If)` directly; `SETTLEMENT_DIGESTS` in `extract.rs` holds it.
- `Vocabulary` still admits `Gather` and `ENode::Buffer`; the language has no bound buffers
  (kernels-evaluate-kernels, 2026-10-15). `mod.rs` lists an `extraction` module that does not exist and
  says `all_rules` has 42 rules; `rule_order.rs` asserts 62.
- Choice maps and `node_for_tag`'s `(class, tag)` pairs still carry positions and a redundant class (the arena
  plan's §8, JP's decisions), and a class's count or `k`th member is a walk (`nodes(c).count()`, `.nth(k)`),
  used by the reranker, research bins and assertions, never by a sweep.
- The cached `FrozenGraph` still holds the constant and variance facts, which nothing reads after saturation;
  moving them to the saturation-only columns measured another −7.8 MB of peak RSS (the arena plan's §8, JP's
  decision).

## Decided in
`docs/DENOTATION.md` §3, §4, §8; `docs/archive/plans/2026-09-02-optimizer-api.md`, `2026-09-01-production-budget-determinism.md`,
`2026-09-04-ir-as-a-trait.md`, `2026-09-08-extraction-witnesses.md`, `2026-09-09-a-fold-is-a-node.md`,
`2026-10-15-kernels-evaluate-kernels.md` §1; `docs/archive/plans/2026-10-15-the-egraph-is-an-arena.md` (landed).
