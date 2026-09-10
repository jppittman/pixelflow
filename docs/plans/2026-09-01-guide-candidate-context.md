# Guide candidate context: the denotation, the cell, the coverage table, and rule-conditioned generation

**Date:** 2026-09-01
**Status:** DESIGN — JP-approved denotation (2026-09-01), written before any code. The Build phase
implements exactly the types, bucket rules, and entry points in §1–§6 and §8; a departure is a
revision of this document, not a silent divergence in code.
**Authority:** `docs/plans/2026-08-31-guide-design-revision.md` (§0 budget-only framing, §2.2 dedup
finding, §4 candidate-local features, §5 protocol); `docs/plans/2026-09-01-phase3-registration.md`
(Round 1 — FROZEN; B = 100/200, Y, the one curve definition, FINAL untouched);
`docs/plans/2026-09-01-phase3-round2-rule-scaling.md` on `claude/phase3-round2` (Round 2 — rule-set
inflation, per-|R| re-minting, `TemplateRewrite`); `docs/plans/2026-09-01-phase3-round1b-domain-shift-registration.md`
on `claude/phase3-domain-shift` (Round 1b — the domain-shift table, `sh`/`bezier` OOD families).
**Code this document extends (unchanged unless named here):** `pixelflow-search/src/egraph/candidate.rs`
(`CandidateFeatures`, `CandidateKey`, `ClassContentKey`, `Firing`, `REGISTERED_PRIMARY_BUDGET_APPLICATIONS`),
`pixelflow-search/src/nnue/guide/{mod,linear}.rs` (`SaturationGuide`, `CandidateSummary`,
`LinearCandidateGuide`, `PerRuleRateGuide`), `pixelflow-search/src/egraph/saturate.rs`
(`GuidedSaturation`), `pixelflow-pipeline/src/bin/{gen_strict_labels,train_guide,skew_test_linear_guide,phase3_at_budget_eval}.rs`,
`pixelflow-pipeline/src/training/{split,guide_linear,quarantine}.rs`, `pixelflow-pipeline/corpus_split.toml`.

## 0. Why this document exists, in JP's terms

Round 1 (`2026-09-01-phase3-registration.md` §9) showed the registered claim holds on DEV — but the
control arm, a per-rule lookup of global TRAIN strict-positive rates with no expression context,
gets most of the effect (0.565 vs 0.537 median ratio at B = 100). Round 1b registered the question
that exposes what is missing: *"The pythagorean identities are useless when trying to evaluate
bezier curves, but i sure as shit want them firing when we're doing spherical harmonics."* A global
per-rule rate cannot say both. The answer is not a bigger model over the same features and it is
not adaptation to the user's kernels; it is the right **conditional**, and the right conditional is
a function of the *expression*:

> "the Guide's features are the local algebraic context, so its conditional is a function of the
> expression, not of anyone's workload — and the contexts it needs to have seen are enumerable,
> because the op vocabulary and the rule set are finite. Generate the training distribution from
> the rules (rule-conditioned, per-rule balanced, plus real families), ship one fixed model, gate
> it on the domain-shift table."
> — JP, 2026-09-01

Everything below is that sentence made into types, a bucketing rule, a table, a generator, and a
training-distribution policy. **Denote before you build** (CLAUDE.md): §1 is the denotation; §2–§6
are obligations on the implementation; §7 says what is explicitly not the way forward.

Binding rules carried over unchanged from Round 1/2 (restated so nobody has to look them up):
budgets in recorded rule applications, never wall-clock; deterministic cost under
`CostModel::latency_prior()`; no timing in any metric; FINAL untouched; family-held-out tiers via
`SplitManifest` (the fence); one constructor for the feature (`CandidateFeatures::observe`) feeds
every Guide family; the dedup key `CandidateKey = (rule_idx, canonical class content)` is
**unchanged**; features are **additive** (nothing the v1 linear Guide reads is renamed or removed);
everything the trainer consumes is exactly what the deployed Guide consumes — the skew test is
mandatory and bit-exact over ≥ 1,000 DEV records; oracle validation for every generated expression
under the algebraic-validity contract; no silent failures; minimal public API.

## 0.5 Survey: what already exists, and the verdict on each item

JP, 2026-09-01: *"scan the repo, take a gander at what already exists; we still have a lot of
dead code with some decent ideas in it, factored.rs and so on."* Every feature in §1 is stated in
terms of the functions below, so the Build phase composes rather than reimplements. Status is by
grep over the workspace at the time of writing (`LIVE` = called from non-test code; `TEST-ONLY` =
reachable only from tests; `DEAD` = no callers).

### 0.5.1 REUSED — §1's denotation is written in terms of these

| Item | Where | Status | What it is | How this design composes from it |
|---|---|---|---|---|
| `CandidateFeatures::observe`, `CandidateKey`, `ClassContentKey`, `Firing`, private `neighborhood_ops` | `pixelflow-search/src/egraph/candidate.rs` | LIVE | The one constructor; the key; the class-level one-hop child multiset; the budget state | Extended in place (§1.4). `neighborhood_ops` is kept as the class-wide DOWN hop-1 multiset the v1 linear Guide already trains on. |
| `RewriteAction` (every variant names the bound e-classes: `Create(ENode{children})`, `Distribute{a,b,c}`, `Factor{common,unique_l,unique_r}`, `Associate{a,b,c}`, `AngleAddition{a,b}`, `PowerCombine{base,exp_a,exp_b}`, …, `Union(other)`) and `EGraph::find_rewrite_matches` (calls `rule.apply` and discards the action) | `egraph/rewrite.rs`, `egraph/graph.rs` | LIVE | The instantiated RHS, already computed at enumeration | The bindings `b` and `dcost` (§1.2) — exact for all 62 rules, no template dependence. `find_rewrite_matches_with_actions` (§1.4) stops throwing the action away. |
| `GraphAccumulator` — `add_edge_at_depth`, `add_op_node_at_depth`, `add_2hop_edge`, `remove_*`, `normalized`, `GRAPH_ACC_DIM = 4·K` | `nnue/guide/accumulator.rs` (`#![allow(dead_code)]`, all `pub(crate)`) | TEST-ONLY (transitively: its only callers are `scoring.rs` methods that are themselves test-only) | Four K=32 sections: `[0..K)` Σ E[parent]; `[K..2K)` Σ E[child]; `[2K..3K)` Σ E[parent] ⊙ shift(E[child]) — 1-hop binding; `[3K..4K)` Σ E[gp] ⊙ shift(E[p]) ⊙ shift²(E[c]) — 2-hop binding. **Fed edge by edge; it has no whole-graph constructor, so it is already scope-agnostic.** | **UP and DOWN are these four sections evaluated over the candidate's local edge set** (§5.6): DOWN = the child section over `n → operand-class node` edges, UP hop-1 = the parent section and the 1-hop binding over `parent-node → c` edges, hop-2 = the 2-hop binding in both directions. No new histogram-to-vector encoder is written; the histograms in §1.4 are the *recorded stream* the sections are realized from. |
| `OpEmbeddings::get`, `OpEmbeddings::init_with_latency_prior` | `nnue/factored.rs` | LIVE | K=32 per-op embedding; **dimension 0 is the latency prior**: `e[op][0] = ln(1 + cycles) / ln(1 + 1000)` from the same `latency_prior_cycles()` table `CostModel::latency_prior()` reads | Every op multiset in the context is realized through this table, so Σ count · prior — the neighborhood's table cost — is a feature for free, in the same units as `dcost`. |
| `CostEdge` / `EdgeTrace` / `EdgeSink` (PR #1063 typed edge stream), `AccumulatorScratch` | `nnue/factored.rs` | LIVE (`unified_backward.rs`, `bootstrap_extraction_head.rs`) | Record the stream, realize against live embeddings at train time (bit-identical to the walk); reusable buffers across many candidates | The **discipline** §1.4 adopts: `CandidateContext` stores op multisets (raw, embedding-free, hashable, serializable) and the encoders realize them; nothing caches floats that a trainable embedding would have to be backed out of. `AccumulatorScratch` is the allocation pattern for hundreds of candidates per round. |
| `SaturationHead::forward_candidate` / `compute_candidate_embed` / `score_candidate`, `CANDIDATE_INPUT_DIM = K + 1` | `nnue/guide/scoring.rs` (`pub(crate)`) | TEST-ONLY | The candidate tower: `1/sqrt(n)` bag-of-op pooling + one scalar row (`budget_fraction`), through `ExprNnue::apply_trunk`, then the bilinear `(mask, rule)` scorer | The nonlinear Guide family consumes the context here (§5.6): the input widens from `K + 1` to `4·K + N_SCALARS`; the bilinear scorer is unchanged. |
| `ExprNnue::apply_trunk`, `K`, `EMBED_DIM`, `HIDDEN_DIM`, the `"TRIF"` checkpoint magic | `nnue/factored.rs` | LIVE | Shared trunk and dims; the serialization format | The tower routes through the shared trunk exactly as today; the magic is bumped only if the nonlinear family's weights are serialized this round. |
| `extract_dag` / `extract` bottom-up DP — local `best_cost: Vec<Option<usize>>`, `best_node`; `ExtractedDAG::{schedule, choices, total_cost}` | `egraph/extract.rs` | LIVE (13 files) | Per-class best cost under a `CostFunction` is computed and then **dropped**; only the root's `total_cost` escapes | `RoundSnapshot::take` (§1.4) takes the vector (`ExtractedDAG.class_costs`, additive field) and derives `best_path` from `schedule`. One DP per round. |
| `EGraph::extract_with_costs` (fixpoint `HashMap<EClassId,(usize,ENode)>`) | `egraph/graph.rs` | LIVE | The older root-only fixpoint walk | Not used; listed so nobody adds a third cost pass. |
| `Provenance`: `Origin::{Seed, Rule(ApplicationId)}`, `ENodeId` (global monotone), `EGraph::tags(class)`, `ApplicationRecord` | `egraph/provenance.rs`, `graph.rs` | LIVE | Stable node identity and creation ordinals | Class **age** (§1.2) = min over the class's tags of the creating application's ordinal (`Seed` = 0). There is no class-age field and `EClassId` is not stable under union — this is the only honest source. |
| `EpisodeLabels::{compute_strict, compute_tight}`; private `chosen_tagged_nodes` | `egraph/labeler.rs` | LIVE | Hindsight labels; the chosen-node walk | Label unchanged (strict now, tight at stage 2). `chosen_tagged_nodes` is the node-granularity "on the best path" set behind `on_best_path`; §1.4 derives the class bitset from `ExtractedDAG::schedule` instead, so it stays private. |
| `GuidedSaturation::until_applications` — dedup before scoring (`seen_keys`, `round_keys`), key marked seen only on a recorded application; `GuidedEpisodeStats` (Round 2 §8) | `egraph/saturate.rs` | LIVE | The deployed loop | The snapshot is taken after `find_rewrite_matches`, before scoring (§1.3); ε-mixing and fallback (§6) are seams in the apply-order step; two counters are added for the snapshot cost (§1.4 invariant 6). |
| `CandidateSummary::new`, `SaturationGuide::score_candidates`, `LinearCandidateGuide` (every term `black_box`-fenced), `PerRuleRateGuide`, `CheckpointError` | `nnue/guide/{mod,linear}.rs` | LIVE | The scoring seam and the two live Guide families | `CandidateSummary` gains `context`; `ContextLinearGuide` (§5.5) is the new linear family; `PerRuleRateGuide` stays the control; the cell oracle (§3.4) is the second control. |
| `guide_linear::{Record, Sample, to_sample, op_index_table, Model}` | `pixelflow-pipeline/src/training/guide_linear.rs` | LIVE | The one JSONL-row → feature encoding shared by trainer and skew test | `Record` gains the `context` block; `to_sample` encodes it; the skew test covers every new term. |
| `RuleTemplates::build`, `Rewrite::{lhs_template, rhs_template}` (30 of 62 provide them) | `nnue/factored.rs`, `egraph/rewrite.rs` | LIVE | Per-rule LHS/RHS arenas | §4.1 instantiates LHS templates; the Build phase adds templates for the 30 rules §4.1 names. |
| `pattern_match_arena`, `substitute_template_arena` | `nnue/mod.rs` | LIVE (crate-internal) | Arena-native template match and instantiate (`Var(i)` = metavariable) | **The instantiation primitive for §4.1** — not re-derived. |
| `BwdGenerator`, `BwdGenConfig`, `collect_rule_templates`, `family_rng_seed`, `Band`/`BANDS` (38 entries) | `nnue/mod.rs`, `egraph/mod.rs`, `bin/gen_bench_corpus.rs` (deleted) | LIVE | The corpus generator and its family seeding | Supplies random operand subterms and wrapper contexts to §4.1; the context family `(band, seed)` of §4.5 is exactly this unit. |
| `screen_for_oracle`, `quarantine_verdict`, `QuarantineGrid` (64 seeded points), `Conditioning`, `Exclusion`, `Quarantine` | `pixelflow-pipeline/src/training/quarantine.rs` (deleted) | LIVE | The same-form JIT-vs-oracle gate with a compositional per-point error bound | §4.2's hard gate, unchanged. |
| `PointCheck::is_well_conditioned`, `eval_scalar`, `equivalence_tolerance` | `pixelflow-ir/src/eval.rs` (deleted) | LIVE / LIVE / TEST-ONLY (one codegen test) | Per-point conditioning predicate; the scalar oracle; the per-op drift allowance | §4.2's cross-form conditioned gate. `equivalence_tolerance` becomes load-bearing here. |
| Cross-form "compare two forms only where both are well-conditioned" block | `pixelflow-pipeline/src/bin/bench_extraction_3way.rs` (private to the binary, ~lines 789–960) | LIVE, bin-local | The reference implementation of §4.2's second gate | **Lifted** into `quarantine.rs` (deleted) as `pub fn cross_form_agreement(..) -> CrossFormVerdict` (§8); the binary calls the lifted function. |
| `FenceKey::of` | `training/structural.rs` | LIVE | Feature-quotient structural key | The cross-tier dedup ledger for every generated expression, unchanged. |
| `SplitManifest::{load, parse, tier_of}` (private `validate`), `Family { band, seed }`, `TierSpec`, `SeedRange`, `Fence`, `assert_family_integrity` / `MIN_FAMILY_ADMISSIONS = 8` | `training/split.rs`, `bin/gen_bench_corpus.rs` (deleted) | LIVE | The fence and the per-family attrition assertion | Extended per §4.5/§4.6; the per-family panic rule is applied to rule-conditioned families. |
| `Variance`, `compute_arena_variance`, `Extraction::chosen_variance` (`pub(crate)`), `EdgeAccumulator::variance_frac_*` | `pixelflow-ir/src/variance.rs`, `extract.rs`, `factored.rs` | LIVE | Per-node dependence lattice and its histogram | **Not in v1.** A per-class variance summary is a plausible later context feature (loop-aware codegen makes it matter); adding it is a schema bump, not a silent field. Listed so it is not rediscovered. |

### 0.5.2 SUPERSEDED — dead, do not build on

| Item | Where | Superseded by |
|---|---|---|
| `ExprGenerator`, `ExprGenConfig` | `nnue/mod.rs` | `BwdGenerator`/`BwdGenConfig`. `ExprGenerator` has no `generate` method and is never constructed. (Its private `shader_weight` table is harvested — see 0.5.4.) |
| `MAX_DEPTH = 8`, `DEPTH_LIMITED_MAGIC`, `DEPTH_LIMITED_VERSION` | `nnue/mod.rs` | `factored::MAX_DEPTH = 192` and the `"TRIF"` format — a live name collision; no binpack reader/writer survives. |
| `EdgeAccumulator::{add_var_ref_edges, remove_var_ref_edges, reset}`, `from_dag_choices` (no variance), `randomize_weights_only`, `memory_bytes`, `RuleTemplates::{len, is_empty, has_templates, has_root_op}` | `nnue/factored.rs` | The walker emits reload edges itself; `from_dag_choices_with_variance`; nothing. |
| `SaturationHead::{forward_graph, mask_score_all_rules_graph, mask_score_all_rules_with_hidden}`, `GraphSummary` | `nnue/guide/{scoring,mod}.rs` | Whole-graph scoring, segregated to a future extraction-cost Judge (design revision §4). A candidate-scope context reuses the accumulator's *sections* (0.5.1), never the whole-graph *scorer*. |
| Local `Expr` shim in `factored.rs` (memory listed it as remaining debt) | — | Gone; `pixelflow_ir::ExprArena` is the only IR. Nothing to migrate. |
| `docs/superpowers/plans/2026-04-07-team4-backward-training.md`'s plan to delete `unified_backward.rs` | — | Stale: the file is LIVE via `bootstrap_extraction_head`. |

### 0.5.3 NOT-APPLICABLE — live or dead, but not this feature's business

| Item | Where | Why not |
|---|---|---|
| `unified_backward.rs` (`forward_cached`, `backward_value`, `apply_unified_sgd`, `UnifiedGradients`) | `pixelflow-pipeline/src/training/` | LIVE extraction-head training math, not featurization. `GradientClipStats`/`clip_stats` and the five `norm_*` accessors are dead diagnostics — harvestable later, not here. |
| `EdgeAccumulator` (flat 2K + PE-encoded 2K), `IncrementalExtractor`, `ExtractionPolicy::Nnue`, `predict_log_cost_with_features` | `factored.rs`, `extract.rs`, `extraction.rs` | The learned-cost extraction path. Every cost in this document is the static table (Round 1 §1). |
| `achievable_cost_within_budget` | `egraph/saturate.rs` | Root-only and mutating; the snapshot needs per-class costs without mutation. |
| `pixelflow-ml/src/graphics.rs` (`ShFeatureMap`, `HarmonicAttention`, `LinearAttention`) | `pixelflow-ml` (feature-gated) | Spherical-harmonic **feature maps over `Field`** for rendering. No reverse dependency from `pixelflow-search`/`pixelflow-pipeline`; `LinearAttention` has no forward pass. The `sh` name collision with Round 1b's family is coincidental — `ShFeatureMap::project` contains no trig. Do not cite it for the `sh` family or for trig context. |

### 0.5.4 Dead with a good idea (harvest) vs dead and superseded (leave)

**Harvest:**
- `GraphAccumulator`'s four-section layout (`accumulator.rs`) — the exact DOWN / UP-1 / UP-2 encoding, already scope-agnostic and already invertible. Reused in §5.6.
- `SaturationHead::forward_candidate` (`scoring.rs`) — the minimal candidate tower; widened, not replaced, in §5.6.
- `ExprGenerator::shader_weight` (`nnue/mod.rs`, private, reachable only via dead `new`) — the only ShaderToy-derived op-frequency prior in the repo (Mul 50, Add 30, Sub 20, Div/Neg 10, Abs 12, Sin/Cos 8, …). Lifted to `pub(crate) fn shader_op_prior() -> OpMap<u32>` and used as §4.1's subterm op prior.
- `equivalence_tolerance` (`pixelflow-ir/src/eval.rs` (deleted)) — the per-op drift table, one test caller; §4.2 makes it load-bearing.
- `bench_extraction_3way.rs`'s cross-form well-conditioned comparison — correct logic trapped in a 170 KB binary; lifted (§4.2).
- `EdgeTrace`/`EdgeSink` — not dead, but the record-the-stream discipline is the good idea §1.4 borrows.
- `UnifiedGradients::clip_stats`, `norm_*` — real gradient-health diagnostics; harvest when the nonlinear family gets a trainer, not now.

**Leave:** everything in 0.5.2.

### 0.5.5 Two facts the survey established that the design must not pretend otherwise about

- **`EGraph` keeps no parent lists.** `EGraph::parent: Vec<EClassId>` is the union-find array;
  `EClass` has exactly `nodes` and `tags`; `rebuild_budgeted` is worklist- and memo-driven. The UP
  context needs a reverse index built once per round by one scan over all nodes (§1.2's
  implementation note) — O(Σ|nodes|), the order `find_rewrite_matches` already pays, flat in |R|.
- **There is no class-age field, and `EClassId` order is not stable under union.** The stable
  ordinals are `ENodeId` and `ApplicationId`; §1.2 derives age from `Origin` over the class's tags.

## 1. The denotation: a candidate and its local context

### 1.1 What a candidate is

A candidate is `(R, n, b)`: a rewrite rule `R` (an index into the e-graph's rule vector), an e-node
`n` in e-class `c` (the match root — `RewriteTarget { rule_idx, class_id, node_idx }` today), and
the bindings `b : pattern-var → e-class` the match established. Bindings are observed, not
inferred: `Rewrite::apply(&EGraph, class, node) -> Option<RewriteAction>` is evaluated at
enumeration time already (`EGraph::find_rewrite_matches` calls it and discards the action), and the
action *is* the instantiated RHS — every `RewriteAction` variant names the e-classes the rule bound
(`Create(ENode { children })`, `Distribute { a, b, c }`, `Factor { common, unique_l, unique_r }`,
`Associate { a, b, c }`, `AngleAddition { .. }`, `PowerCombine { .. }`, ...). For the 62 production
rules the bound classes are `n`'s operand classes (depth-1 LHS: commutative, identity,
annihilator, idempotent, doubling, the pow-special literals, ...) or `n`'s operands plus one class
one hop below (depth-2 LHS: distribute, factor, associative, parity, angle-addition, the cancels,
homomorphism, pythagorean, fma-fusion, recip-sqrt, power-combine, ...). The `guide_coverage` tool
(§8) reports the depth of every rule's LHS from its template rather than this document counting
them. For `TemplateRewrite` (Round 2 §8) the e-matcher's substitution is the binding directly.

**DOWN summarizes the operand classes of `n`** — the classes the rule matched *through* — in
operand order, `K_DOWN = 3` slots (the largest arity any rule matches: `MulAdd`/`Select`), padded
with `ClassSummary::ABSENT`. A depth-2 rule's second-hop binding is visible through its parent
operand's op-multiset (a `distribute` match's `B + C` class shows `Add` in slot 1's histogram). This
is the depth-1 projection of `b`, chosen because it is one definition for hand-coded and template
rules alike and costs nothing the enumeration does not already pay; the full substitution for
template rules is a possible later refinement, not part of this design.

### 1.2 The three parts of LOCAL CONTEXT (JP's design, verbatim structure)

**DOWN** — each bound class is a *set* of known-equivalent forms, so the feature is a summary of the
set: op-multiset over the class's e-nodes, best cost under `CostModel::latency_prior()`, size
(e-node count), age (created at which application ordinal).

**UP** — `c`'s parent set: parent-op multiset one hop up and two hops up. This is where
"`sin² + cos² → 1` pays when it feeds a `sqrt` or a divide" lives. `O(parents)`, flat in `|R|`.
*Implementation note, stated so it is not discovered later:* this `EGraph` does **not** keep
per-class parent lists (rebuild is memo-driven off a worklist, `graph.rs::rebuild_budgeted`);
the parent index is built once per round in one `O(Σ|nodes|)` pass and amortized over every
candidate scored that round, exactly like the cost pass below. It is still flat in `|R|`.

**STATE** —
- local `dcost` = table cost of the RHS instantiated with the bound classes' best costs, minus `c`'s
  current best. This is the greedy signal; the Guide's job is to be smarter than it — to learn when
  a cost-neutral or cost-raising rewrite enables a later drop. `dcost` is computed from the
  `RewriteAction` already in hand, so it is exact for **all 62 rules** (constant-fold: `Create(Const)`
  → `0 − best(c)`; a `Union(other)` → `best(other) − best(c)`; every structured variant is a fixed
  op tree over named classes).
- `on_best_path(c)` = `c` is reachable in the current best extraction — one bottom-up cost pass per
  round (`extract_dag`'s `best_cost`/`choices` restricted to the root's reachable set), amortized over
  all candidates that round.
- `budget_fraction` = `application_ordinal / REGISTERED_PRIMARY_BUDGET_APPLICATIONS` (exists,
  unchanged, `candidate.rs`).
- `class_age`, `class_size` of `c` itself (same definitions as the DOWN summaries).

**Label** — unchanged: hindsight strict (`EpisodeLabels::compute_strict`) now, tightened
(`compute_tight`) at stage 2 — a property of the e-graph, domain-free.

### 1.3 Observation time: round start, for mint and deploy alike

Round 1's `gen_strict_labels` replays `observe` against the **final** saturated graph
(`candidate.rs` module doc, "Approximation, stated plainly"). That approximation is tolerable for
class content and the one-hop neighborhood; it is **not** tolerable for STATE: `best(c)` read off
the final graph makes `dcost ≤ 0` and `on_best_path` final-truth for every application, which is a
different feature than the deployed loop can ever compute. The at-budget report already named the
consequence for label transport ("needs a `(rule, class content at firing time) → label` map the
provenance log does not record ... instrumenting the unguided sweep itself").

**Binding:** every `CandidateContext` is observed **at the start of the round in which the candidate
is enumerated**, against a `RoundSnapshot` taken once per round, in both places:

- deploy — `GuidedSaturation::until_applications` takes the snapshot after `find_rewrite_matches`,
  before scoring;
- mint — the unguided sweep (`saturate_until_applications`) is run with an observation hook that
  takes the same snapshot at the same point and calls the same `observe` for every enumerated match,
  storing the context keyed by the `ApplicationId` the match receives when it fires; labels are
  joined on `ApplicationId` after `compute_strict`.

Within a round, applications mutate the graph; contexts are *not* refreshed mid-round in either
place (deploy scores once per round; the mint mirrors that). This is the same object at the same
time on both sides — the precondition for the skew test to mean anything.

### 1.4 Types (the Build phase implements these names)

```rust
// pixelflow-search/src/egraph/candidate.rs — additive to the existing file.

/// Largest operand arity any rule matches (MulAdd/Select). DOWN has this many slots.
pub const K_DOWN: usize = 3;

/// Per-op counts over a set of e-nodes' root ops, saturating at u8::MAX. Indexed by OpKind
/// (`OpMap<u8>`); leaves (Var/Const/Buffer) count under their own OpKind.
pub struct OpHistogram(OpMap<u8>);

/// Summary of one e-class as a SET of equivalent forms (§1.2 DOWN).
pub struct ClassSummary {
    pub present: bool,      // false == ClassSummary::ABSENT (slot beyond the rule's arity)
    pub ops: OpHistogram,   // root ops of every e-node in the class
    pub best_cost: u32,     // latency_prior best cost of the class at round start (ClassCosts)
    pub size: u16,          // e-node count (saturating)
    pub age: u32,           // application ordinal at which the class's earliest node was created; 0 = seed
}

/// Per-(op, op) counts, sparse, sorted, no zero entries, saturating at u8::MAX. Keeps the
/// PAIRING a marginal histogram loses: the GraphAccumulator 2-hop binding section
/// (§5.6) needs (parent op, grandparent op) per edge, not two independent marginals.
pub struct OpPairHistogram(Vec<((OpKind, OpKind), u8)>);

/// Parent-op multisets one and two hops above the match root (§1.2 UP).
pub struct ParentHistogram {
    pub hop1: OpHistogram,      // ops of e-nodes that have `c` as a child
    pub hop2: OpPairHistogram,  // (op of hop-1 parent node p, op of node g that has p's class as a child);
                                // the hop-2 marginal is its projection onto the second component
    pub parents1: u16,          // |hop-1 parent e-nodes| (saturating); 0 == c is the root
    pub parents2: u16,
}

/// The greedy signal and the position in the episode (§1.2 STATE).
pub struct CandidateState {
    pub dcost: i32,          // table cycles: action_table_cost(snapshot) - snapshot.best(c); saturating
    pub on_best_path: bool,  // c reachable in the round-start best extraction
    budget_fraction_bits: u32, // existing field, moved here; read via budget_fraction()
    pub class_age: u32,
    pub class_size: u16,
}

pub struct CandidateContext {
    pub down: [ClassSummary; K_DOWN],
    /// (op of operand-class node t, op of node u in a child class of t) — the DOWN two-hop
    /// pairing, for the same reason `ParentHistogram::hop2` is a pair histogram (§5.6).
    /// For a depth-2 rule this is where the second-hop binding's own forms are visible.
    pub down_hop2: OpPairHistogram,
    pub up: ParentHistogram,
    pub state: CandidateState,
}

/// The candidate-local feature. `key` and `neighborhood_ops` are UNCHANGED; `context` is additive.
pub struct CandidateFeatures {
    pub key: CandidateKey,              // UNCHANGED: (rule_idx, ClassContentKey of c)
    pub neighborhood_ops: Vec<OpKind>,  // UNCHANGED (v1 linear Guide reads it)
    pub context: CandidateContext,      // NEW
}

/// Per-round, per-graph state every candidate of the round is observed against (§1.3).
pub struct RoundSnapshot {
    costs: ClassCosts,          // Vec<u32> best cost per canonical class, bottom-up under latency_prior
    best_path: ClassBitSet,     // classes reachable from `root` through `choices`
    parents: ParentIndex,       // canonical class -> Vec<(parent canonical class, OpKind)>
    ages: Vec<u32>,             // canonical class -> min origin ordinal over its nodes
    pub round_ordinal: u32,
}
impl RoundSnapshot {
    /// THE one way to take a snapshot; O(Σ|nodes|). Called once per round in mint and deploy.
    pub fn take(egraph: &EGraph, root: EClassId, costs: &CostModel) -> Self;
}

/// Caller-supplied firing context. `action` is NEW; everything else exists.
pub struct Firing<'a> {
    pub rule_idx: usize,
    pub match_root: EClassId,
    pub node_idx: usize,                 // NEW: which e-node in match_root matched (RewriteTarget::node_idx)
    pub action: &'a RewriteAction,       // NEW: the instantiated RHS, from enumeration
    pub application_ordinal: u64,
    pub registered_budget: usize,
}

impl CandidateFeatures {
    /// THE one constructor. Builds key + neighborhood (unchanged code) + context in one pass.
    pub fn observe(egraph: &EGraph, snapshot: &RoundSnapshot, firing: &Firing<'_>) -> Self;
}

impl RewriteAction {
    /// Table cost of the instantiated RHS: Σ op costs of the nodes the action would create
    /// + Σ snapshot.best(bound class) for its leaves. Exact for every variant.
    pub(crate) fn table_cost(&self, snapshot: &RoundSnapshot, costs: &CostModel) -> u32;
}
```

**Invariants (tests pin each):**
1. `observe` is the only constructor of `CandidateFeatures`; `CandidateContext` has no public
   constructor. The mint hook and `GuidedSaturation` both call it with a `RoundSnapshot` from
   `RoundSnapshot::take`.
2. `features.key` is bit-identical to Round 1's `CandidateKey` for the same `(egraph, rule_idx,
   match_root)` — the dedup set's semantics do not move.
3. `dcost` for `commutative` is exactly `0`; for `identity`/`annihilator`/`involution`/
   `cancellation`/`constant-fold` it is `≤ 0`; for `distribute` it is `≥ cost(Mul)`. (Sanity pins
   on the table-cost function, per rule class.)
4. `on_best_path(root) == true` in every snapshot; a class with `parents1 == 0` is the root.
5. Observing the same candidate twice against the same snapshot is bit-identical.
6. `RoundSnapshot::take` is `O(Σ|nodes|)` and independent of `|R|` — pinned by the Round 2 §7.1
   "scored candidates per recorded application" measurement, which now also reports snapshot
   cost per round in node-visits.

`EGraph::find_rewrite_matches` gains an additive sibling `find_rewrite_matches_with_actions() ->
Vec<(RewriteTarget, RewriteAction)>` so the action is not recomputed a third time (it is computed
at enumeration and again at `apply_single_rule` today). Production `saturate*` paths are untouched.

`CandidateSummary` (`nnue/guide/mod.rs`) gains `pub context: CandidateContext` (cloned from
`features.context` in `CandidateSummary::new`); `LinearCandidateGuide` and `PerRuleRateGuide` ignore
it (they stay bit-exact with their checkpoints). The new context-aware Guide family (§5) reads it.

## 2. The CELL: bucketing a context coarsely, exactly

The feature is full-precision; the **cell** is its coarse image, used for (a) the coverage table
(§3), (b) generation targeting (§4), (c) out-of-support detection (§6). One function,
`CandidateCell::of(&CandidateFeatures) -> CandidateCell`, deterministic, no floats compared.

### 2.1 The op-group alphabet `G` (8 symbols), from the cost table's own clusters

| group | ops | latency_prior cycles |
|---|---|---:|
| `LEAF` | Var, Const, Buffer, Tuple | 0 |
| `LIN` | Add, Sub, Neg, Abs, Min, Max, Select, Lt, Le, Gt, Ge, Eq, Ne, Floor, Ceil, Round, TruncToInt, IntToFloat, IAdd, Shl, Shr, BitAnd, BitOr | 1–4 |
| `MUL` | Mul, MulAdd | 5 |
| `DIV` | Div, Recip | 11–16 |
| `ROOT` | Sqrt, Rsqrt | 15–21 |
| `TRIG` | Sin, Cos, Tan, Asin, Acos, Atan, Atan2 | 70–103 |
| `EXPLOG` | Exp, Exp2, Ln, Log2, Log10, Pow | 69–196 |
| `OTHER` | Dwrt, Gather, RawGather, Reduce | — |

`MUL` is split from `LIN` despite equal cost because it is the pivot of distribute/factor/fma-fusion
and the multiplicative half of every power/trig identity; the rest of the grouping is the table's
own cost clusters. `group_of(OpKind) -> Group` is total over `OpKind::all()` (a test iterates it).

### 2.2 Dominant group of a class / of a parent set

`dominant(ClassSummary) = group_of(root op of the class's best-cost e-node)` — deterministic given
`ClassCosts` (ties broken by the class's node order, which `ClassContentKey` already canonicalizes).
`dominant(OpHistogram) = argmax_g Σ_{op ∈ g} count(op)`, ties broken in the table order above;
`NONE` if the histogram is empty.

### 2.3 The cell

```rust
pub struct CandidateCell {
    pub rule_idx: u16,
    pub down: DownSig,      // sorted multiset of dominant(down[i]) over PRESENT slots, at most 2 kept:
                            //   for ternary rules keep the two slots with the larger best_cost.
                            //   |DownSig| = 8 (unary) | 36 (binary multisets of 8) | 1 (nullary)
    pub up1: Option<Group>, // dominant(up.hop1); None == root
    pub dcost: DcostBucket, // 7 buckets, §2.4
    pub on_path: bool,
    pub budget: BudgetBucket, // 5 buckets, §2.5
}
```

Levels, because coverage is hierarchical (§3): **L1** = `(rule_idx, down)`; **L2** = L1 + `up1`;
**L3** = the full cell. `hop2` is a feature, not part of the cell (it multiplies the cell count by 9
for a marginal the model can still read at full precision).

### 2.4 `dcost` buckets (7), justified from the table

```
DcostBucket = { LeMinus64, Minus63ToMinus9, Minus8ToMinus1, Zero, Plus1ToPlus8, Plus9ToPlus63, GePlus64 }
```

Boundaries at ±1, ±9, ±64 in table cycles. `|Δ| ≤ 8` is "one arithmetic op's worth" — every
`LIN`/`MUL` op costs ≤ 5, so this bucket holds rewrites that add or remove one cheap op (identity,
commutative-then-fma, doubling). `9 ≤ |Δ| ≤ 63` is "one `DIV`/`ROOT` op's worth, or several
arithmetic ops" (recip-sqrt: 16+15 → 21 is Δ = −10; power-sqrt: 196 → 15 is far below). `|Δ| ≥ 64`
is transcendental scale (Sin 70, Exp 75, Ln 128, Pow 196): a rewrite that removes or introduces a
transcendental. Zero gets its own bucket because it is the dominant case (every structural rule)
and the case the greedy signal is silent on — exactly where the Guide has to be smarter than
`dcost`. JP's example grid (`<−8, [−8,−1], 0, [1,8], >8`) is the inner five; the ±64 split is added
because the table's dynamic range is 1–196 and a rule that saves a `Div` and a rule that saves a
`Pow` are not the same event.

### 2.5 `budget_fraction` buckets (5), from the registered tiers

```
BudgetBucket = { B0_25 [0, 0.25), B25_50 [0.25, 0.5), B50_100 [0.5, 1.0), B100_200 [1.0, 2.0), B200Plus [2.0, ∞) }
```

`budget_fraction` is denominated in `REGISTERED_PRIMARY_BUDGET_APPLICATIONS = 100`; the 1.0 and
2.0 boundaries are the primary and secondary registered tiers, and the two sub-tier splits
separate early from late inside the primary tier (the pre-flight showed most of the curve's shape
is inside the first 100 applications for classical).

### 2.6 What is deliberately not in the cell

Class size and age (features, reported as marginals only); `hop2`; the full per-op histogram; the
class content key. The cell is coarse **on purpose** — it exists to be countable (§3) and
targetable (§4), not to be the model's input.

## 3. The COVERAGE TABLE

### 3.1 Reachable cells

Enumerated from the rule set and `G`, under the caps in §2.3, per level:

- **L1 reachable** = Σ_rules |DownSig(arity(R))| where `arity(R)` is the LHS root op's arity
  (`lhs_template` root where present, else the `RewriteTarget` node's op — the tool asserts every
  rule yields one arity, and fails loud otherwise). For the 62 production rules: 19 unary-root
  rules × 8 + 42 binary-root rules × 36 + constant-fold (all-`LEAF` operands, arities 1..3: 3)
  = 152 + 1,512 + 3 = **1,667**.
- **L2 reachable** = L1 × 9 (`up1 ∈ G ∪ {None}`) = **15,003**.
- **L3 reachable** (upper bound; the tool reports the per-rule reachable `dcost` set derived from
  the action shape, which is smaller — commutative reaches only `Zero`) = L2 × 7 × 2 × 5 =
  **1,050,210**.

The unary/binary partition of the 62 (by LHS root): unary = involution ×2 (1, 5), parity ×6
(30–35), angle-addition ×2 (36, 37), exp/log cancels ×4 (41–44), homomorphism ×2 (45, 46),
log-power ×2 (55, 56), recip-sqrt (60); binary = the remaining 42 (0, 2–4, 6, 7, 9–29, 38–40,
47–54, 57–59, 61); constant-fold (8) is any-arity over constants. **The tool computes these
counts from `all_rules()`; if its number differs from the ones above, this section is corrected
to the tool's number and the discrepancy is recorded — the doc does not overrule the code.**
At inflated |R| (Round 2) the same function runs over the inflated rule vector; compositions and
`TemplateRewrite`s have templates by construction.

### 3.2 Seen cells, counts, thresholds

For a mint (one label file), the tool counts per cell: `n` (observations), `pos` (strict
positives), `fired` (n; every observation fired in the unguided sweep), and per rule the same
totals. Thresholds, fixed here:

- `n_thin = 100` observations. A cell-conditional positive rate `p` has standard error
  `√(p(1−p)/n)`; separating a 10% cell from a 1% cell at two standard errors needs `n ≈ 100`.
- `pos_thin = 5` positives — but only for rules with any positives at all (a rule whose strict rate
  is 0 everywhere has no positive cell to be thin in; it is reported as a *zero-rate rule*, which
  is a separate row).
- A cell is **empty** if `n = 0`, **thin** if `0 < n < n_thin` or (`pos < pos_thin` and the rule has
  positives), **covered** otherwise.

### 3.3 Report format

`docs/results/<date>-guide-coverage.{json,md}`, emitted by `guide_coverage` (§8). One report per
mint; the md has:

1. **Header:** rule-set fingerprint (Round 2 §8), label source (strict/tight), corpus files and
   MD5s, tier, `n_thin`/`pos_thin`.
2. **Global row:** per level L1/L2/L3 — reachable, seen, covered, thin, empty; share covered.
3. **Per-rule table** (62 rows, or |R|): idx, name, arity, L2 reachable / seen / covered / thin /
   empty, `n`, `pos`, rate, and the top-5 *empty* L2 cells by a fill priority (§4.4).
4. **Per-tier comparison** when more than one mint is supplied (TRAIN, DEV, `sh`, `bezier`): the
   set difference `cells(S) \ Support(TRAIN)` per rule — the cells a held-out set exercises that
   training never saw. This column is the out-of-support trigger rate (§6.2) before any Guide runs.
5. **L3 support histogram:** how many L3 cells have `n ≥ n_thin`, per rule, and the
   cell-conditional Bayes ceiling (§3.4).

Every Guide number published after this document ships with (a) this table for the mint the
checkpoint was trained on, (b) the Round 1b domain-shift table, (c) the fallback trigger rate on
the evaluation set. A Guide number without them is unreviewable.

### 3.4 The honest limit, as a number

The same candidate in the same local context can pay or not depending on global state.
`on_best_path` and `budget_fraction` capture most of that; the rest is label noise. So a ceiling
below 1.0 is expected and **is not chased**: the tool reports, per mint, the **cell oracle** — the
scorer that assigns each DEV candidate its L3 cell's TRAIN positive rate (unseen cell → the rule's
rate) — and its DEV AUC / PR-AUC next to the model's. No model over these features at cell
granularity can beat the cell oracle on average; a model that approaches it is done, and the gap
between the cell oracle and 1.0 is the label noise this design accepts.

## 4. Rule-conditioned generation

### 4.1 The instance

For each rule `R` in the rule set with an LHS template (`Rewrite::lhs_template`; 30 of 62 today —
the Build phase adds templates for the rules whose LHS is expressible with `Const` leaves: parity
×6, angle-addition ×2, cancels ×4, homomorphism ×2, pow-special ×6 (`Pow(a, Const(k))`),
power-recurrence, log-power ×2, expand-square, diff-of-squares, recip-sqrt; that leaves exactly
{constant-fold, differentiate} untemplated: constant-fold gets a trivial sampler (a random op over
`Const` operands), differentiate is inert without `Dwrt` and is excluded and listed):

1. **Instantiate the LHS:** each pattern variable is bound to a random well-typed subterm drawn from
   the existing `BwdGenerator` with `max_depth ∈ {0, 1, 2}` over the band's variable count (depth 0
   = a leaf, so the LEAF × LEAF cell is reachable; depth 2 lets the operand's dominant group range
   over `G`). Shared variables in the LHS (pythagorean's `x` in `sin²x + cos²x`) are bound once.
   The substitution itself is `nnue::substitute_template_arena(&mut arena, &lhs_arena, lhs_root,
   &bindings)` (0.5.1) — the existing primitive, not a copy. The op prior for subterm draws is
   `shader_op_prior()` (0.5.4, lifted from the dead `ExprGenerator::shader_weight`); when a target
   cell (§4.4) names a group, that group's ops get their weight multiplied by 8, so targeting is a
   reweighting of one table, not a second generator.
2. **Embed** the instance at a random position in a random context of depth `d ∈ {0, 1, 2, 3}`:
   `d = 0` is the bare instance (root; `up1 = None`); `d ≥ 1` wraps it in `d` random ops so that
   `up1` ranges over `G` — the `sqrt`/`div` parents pythagorean needs to pay are reached by
   choosing the immediate wrapper op from a *target* group (§4.4).
3. **Validate** (§4.2). Keep only instances that pass.
4. **Filter for payability** (§4.3).

### 4.2 Oracle validation (the algebraic-validity contract, `2026-08-05-egraph-nnue-research-workflow.md` §0.4)

Two checks, never conflated:

- **Same-form hard gate** — every generated expression passes the numeric quarantine
  (`training::quarantine::Quarantine`, JIT vs `pixelflow_ir::eval_scalar` on the same arena over
  the 64-point seeded grid under `DifferentialCheck`'s compositional bound; `screen_for_oracle`
  runs first, as it must). Mismatch = miscompile = hard failure at the same `MAX_MISMATCH_RATE`
  the corpus uses.
- **Cross-form conditioned gate** — the LHS instance and its RHS instantiation (the rule's
  `rhs_template` under the same bindings) are evaluated by `eval_scalar` at the well-conditioned
  points of the same grid (`PointCheck::is_well_conditioned`, for **both** forms) and must agree
  within the composed per-op tolerance (`equivalence_tolerance`). Disagreement at a
  well-conditioned point **fails** — an implementation error in the rule, the template, or the
  generator — and is written to the run's quarantine log with the point and both values.
  Divergence at ill-conditioned points is metadata on the instance, never an exclusion, never an
  alarm. Algebraically valid rewrites are never called "unsound" and never get domain guards. The
  comparison is `quarantine::cross_form_agreement(lhs, rhs, &grid) -> CrossFormVerdict`, lifted
  from the private block in `bench_extraction_3way.rs` (0.5.1) so there is one definition; the
  binary is re-pointed at it.

### 4.3 Payability: keep instances where the rule can demonstrably pay, or is a designated enabler

An instance is kept iff, under `latency_prior`, **(a)** `dcost(R, instance) < 0` — the RHS is
strictly cheaper than the LHS for this instantiation (a `pay` instance); or **(b)** `R` is a
*designated enabler* and the instance is an enabler case. Enablers are designated from Round 1's
own evidence, not taste: a rule is an enabler if its strict-positive rate is > 0 on DEV while its
`dcost` is ≥ 0 by construction — today: `doubling`/`halving` (20, 21; the only path from
`Sin(2·φ)` to angle addition — Round 1b §4), `canonicalize` (0, 4), `even-negation` (34, 35),
`odd-negation` (30, 31), `constant-fold` (8; `dcost ≤ 0` but often 0), `fma-fusion` (59;
`dcost = 0` on the table — `MulAdd` is priced at `Mul` parity — yet strict rate 0.4–0.5%). An
enabler case is an instance where the enabler's RHS, embedded, matches the LHS of some **pay** rule
`R'` whose `dcost < 0` under the same bindings — checked by running the unguided sweep for two
rounds on the instance and asserting `R'` fires with the enabler's output on its path. Rules with
`dcost = 0` everywhere and strict rate 0 everywhere (commutative ×4, associative ×8, distribute,
involution ×2, identity ×2, annihilator, idempotent ×2, halving as a standalone) are **not**
generated for directly; they are covered as the *context* of pay/enabler instances, and their
cells fill from the ordinary generator bands. The designated-enabler list is a table in the
generator (`ENABLERS: &[usize]`) with the DEV rate that justified each entry, re-derived from each
round's report — a rule leaves the list when its rate drops to 0 under the current labels.

### 4.4 Balance: per rule, then per cell (fill thin cells first)

Targets: `n_rule = 2,000` admitted instances per generated-for rule (≈ 40× `n_thin`, so a rule with
~20 reachable L2 cells of interest can cover each), and within a rule the draws are allocated by
the coverage table of the *previous* mint: the generator reads `guide-coverage.json`, ranks the
rule's L2 cells by `priority = (empty ? 2 : thin ? 1 : 0) × (1 + reachable-frequency prior)`, and
draws the wrapper op / operand depth to target the highest-priority cells first (the targeting is
by construction: to reach `(pythagorean, {TRIG, TRIG}, up1 = ROOT)` the wrapper is drawn from
`ROOT`). Cells that stay empty after `4 × n_thin` targeted attempts are reported as
**unreachable-in-practice** with the attempt count (never padded, never silently dropped) — the
likely cause is that the rule's LHS constrains the operands more than the arity-only bound in
§3.1 admits, which is information for the reachable count, not a generator failure.

### 4.5 Family naming and TRAIN/DEV assignment

A rule-conditioned expression's holdout unit is its **context family**, exactly the existing unit:
the `(band, seed)` of the generator stream its operands and wrapper were drawn from. The rule is
not the family — every rule must appear in TRAIN, and DEV tests it on unseen context families.
Names: `{tier}_rc-{rule_name}_b{band:02}_f{seed:02}_{idx:05}`, where `rule_name` is
`Rewrite::name()` — the rule **family** (the four `commutative` indices share one name and one
prefix; under Round 2 a duplicate `<name>#dup<k>` and a composition carry their own names). The
name, not the index, is in the entry name because an index is rule-set-relative under Round 2's
inflated vectors and the corpus must remain readable across |R| points. Tier follows
`corpus_split.toml`'s existing `[train]`/`[dev]` band tables (the fence parser is extended to the
`rc-` form; the check that a TRAIN entry's family is a TRAIN family is unchanged). Output files are separate —
`corpus_rc_train.bin`, `corpus_rc_dev.bin` — so Round 1's `corpus_{train,dev}.bin` MD5s are
untouched and Round 1's numbers stay comparable. The manifest gains one table:

```toml
[rule_conditioned]
# Rule-conditioned families inherit tier from the [train]/[dev] band tables above.
bands_train = [3, 8, 14, 19, 25, 31]   # subset of [train].bands, one per depth class
bands_dev   = [6, 16, 27]              # subset of [dev].bands
seeds = [[0, 7]]
```

`SplitManifest::validate` rejects a band listed under `bands_train` that is not in `[train].bands`
(and likewise for dev), and any band in both.

### 4.6 How real families join

- `sh`, `bezier` — **DEV, OOD, never TRAIN**, exactly as Round 1b registered them
  (`corpus_dev_ood.bin`). They are the domain-shift table's rows; their coverage report (§3.3 item 4)
  is the "cells DEV needs that TRAIN never saw" column.
- shaders — the 12 ShaderToy/iquilezles ports and the 5 named production kernels are **FINAL**,
  untouched until the publication run.
- Structure-aware **TRAIN** families (the Round 1b H_null consequence: "a structure-aware family
  (SH, rotations, Fourier sums) is required in TRAIN"): `fourier` (seeded sums of
  `a_k·sin(kθ + φ_k)` with shared arguments), `rotation` (2-D/3-D rotation compositions —
  `cos·cos − sin·sin` with shared angles), `normalize` (`v·rsqrt(v·v)` shapes), `horner` (nested
  polynomial evaluation). Named families entered through the same fence + quarantine path as the
  OOD families, registered as `[train] families = [...]`. **Conflict with the sibling branch,
  stated:** Round 1b §3 has `SplitManifest::validate` reject a `families` key under `[train]`
  outright. The invariant that matters is *a named family appears in exactly one tier*; the
  outright rejection is the over-tightened form. This design asks Round 1b's step 1 to implement
  the exactly-one-tier check instead; `sh`/`bezier` remain DEV-only under it. The per-rule
  positive-rate targets in Round 1b §5 (every trig rule with ≥ 100 TRAIN positives) become the
  acceptance metric for these families — read straight off the coverage table's per-rule row.
  **Two form collisions flagged for JP's decision (survey pass), not resolved here:** (i)
  `normalize` is the *name and the form* of one of the five FINAL production kernels
  (`corpus_split.toml` `[final] kernels`; `gen_bench_corpus::named_kernel`), and `horner` is the
  form of the FINAL kernel `poly` — a TRAIN family that reproduces a FINAL kernel's structure
  leaks by construction, and `FenceKey` catches only exact structural duplicates, not the family
  resemblance. (ii) `fourier`'s `sin(kθ + φ_k)` terms are the `sh-direct` form (`Sin(m·φ)` with a
  `Mul(Const, Y)` argument) of the DEV OOD family, which would make `sh` less out-of-distribution
  than Round 1b registered it to be. Candidate replacements that exercise the same contexts
  without either collision: `lighting` (Lambert/Phong — normalized dot products, `max(0, ·)`,
  `pow(·, k)`: the LIN/ROOT/EXPLOG parents) and `complex-mul` (chains of complex products and
  moduli: the expand-square / diff-of-squares / MUL context); `rotation` stands as proposed.
  Whichever set JP picks, the acceptance metric above is unchanged.

## 5. Training-distribution policy for Round 2 (and after)

1. **TRAIN** — two pre-committed arms, because JP's framing demotes the natural-frequency
   corpus and the demotion should be measured rather than assumed:
   - **Primary arm:** rule-conditioned, cell-balanced `corpus_rc_train.bin` (§4) + the
     structure-aware named TRAIN families (§4.6). The natural-frequency TRAIN bands
     (`corpus_train.bin`) are **not** trained on — "natural-frequency generator corpus demoted to
     a DEV baseline" (JP) is read literally.
   - **Secondary arm (ablation):** the primary arm's data + `corpus_train.bin`. It answers one
     question — does the natural corpus add anything once the cells are covered? — and is
     reported next to the primary, never in its place.
   - A third, distribution-only ablation trains Round 1's *feature set* (no context) on the
     primary arm's data, so the feature's effect and the distribution's effect are separable.
   Labels minted with the in-sweep observation hook (§1.3) under the rule set being trained for
   (per-|R| re-minting, Round 2 §7.3, unchanged).
2. **DEV baseline** = the natural-frequency DEV bands (`corpus_dev.bin`, unchanged MD5) — demoted
   from "the distribution" to *one* held-out distribution — plus `corpus_rc_dev.bin` (rule-
   conditioned, unseen context families) plus `corpus_dev_ood.bin` (`sh`, `bezier`). Every DEV
   number is reported per set; the registered Round 1/2 statistics are still computed on
   `corpus_dev.bin`'s classical band so the anchors (`Q(62)` = 0.537 / 0.696) mean what they meant.
3. **Every Guide number ships with three tables:** the domain-shift table (Round 1b, `D_A` per set
   with `M_B`), the coverage table (§3, for the mint behind the checkpoint), and the fallback
   trigger rate (§6.2). The context-aware Guide's *acceptance* on the domain-shift axis is Round 1b's
   H_shift test unchanged (`D_control − D_linear > M_B` on `sh` at B = 100), re-run with the
   context Guide in the linear arm's place — no new statistic is registered by this document.
4. **The model is one fixed, versioned artifact:** checkpoint carries the rule-set fingerprint, the
   mint's corpus MD5s, the coverage table's hash, and the `Support` set (§6.2). Deploying a
   checkpoint under a different fingerprint is a hard error (Round 2 §6, unchanged).
5. **Model family:** the Build phase's first context Guide is the linear model extended with the
   context features — `w_rule[rule_idx]` + per-slot `w_down[slot][group]` over dominant groups +
   `w_down_cost[slot]·log1p(best_cost)` + `w_up1[group]` + `w_up2[group]` + `w_dcost[bucket]` +
   `w_on_path` + `w_budget` + `w_age·log1p(class_age)` + the existing neighborhood/size terms.
   Same trainer, same loss weighting, same skew test extended to the new fields. Rule embeddings
   from templates (Round 2 §7.2's named Round-3 lever) remain out of scope.
6. **Realization through the existing towers — §5.6.** The context is *recorded* as histograms
   (§1.4) and *realized* by each family's encoder; the linear family's realization is item 5, the
   nonlinear family's is §5.6. Both read the same `CandidateContext`; neither restates it.

### 5.6 The nonlinear family: `GraphAccumulator` sections over the candidate's local edge set

The coordinator's instruction, and the survey's finding (0.5.1), is that `GraphAccumulator`'s
four sections *are* an up/down-context representation built whole-graph, so at candidate scope
the UP/DOWN features are stated as those sections evaluated over the candidate's parent set and
bound classes — not as a new encoder. Concretely, per candidate, with `E = ExprNnue::embeddings`
and `op_n` the op of the matched node `n`:

```
acc = GraphAccumulator::new()                          // or reset() on a per-round scratch buffer
for slot in down where present:
    for (o, k) in slot.ops:        k × acc.add_edge_at_depth(E, parent = op_n, child = o, depth = 1)
for ((t, u), k) in down_hop2:      k × acc.add_2hop_edge(E, gp = op_n, parent = t, child = u)
for (p, k) in up.hop1:             k × acc.add_edge_at_depth(E, parent = p, child = op_n, depth = 1)
for ((p, g), k) in up.hop2:        k × acc.add_2hop_edge(E, gp = g, parent = p, child = op_n)
acc = acc.normalized()                                 // per-section L2, exactly as today
input = [acc.values (4·K = 128) ‖ scalars (N_SCALARS = 20)]      // CANDIDATE_INPUT_DIM: K+1 → 148
```

Read off the sections: `[K..2K)` is the DOWN marginal, `[0..K)` the UP hop-1 marginal,
`[2K..3K)` the 1-hop binding in both directions (keyed by which side is `op_n`), `[3K..4K)` the
2-hop binding in both directions — which is why §1.4's hop-2 histograms are *pair* histograms:
the binding section is a sum of products over edges, not a product of marginals. Because
`E[·][0]` is the latency prior (0.5.1), the marginal sections carry Σ count · prior — the
neighborhood's table cost — with no extra feature. The 20 scalars: `budget_fraction`,
`dcost / 64` clipped to ±1, `on_best_path`, `class_age / B`, `log1p(class_size)`,
`log1p(parents1)`, `log1p(parents2)`, `parents1 == 0`, and per DOWN slot (`present`,
`log1p(best_cost)`, `log1p(size)`, `age / B`) × 3. `forward_candidate` keeps its `1/sqrt(n)`
pooling for the pre-existing `neighborhood_ops` row and its path through `apply_trunk` and the
bilinear scorer; only its input width changes. The tower stays `pub(crate)`; a trainer for it is a
later task, and it is **not** in this round's accept gate (the linear family is) — exactly the
sequencing `train_guide`'s module doc argues for.

## 6. Design items specified here, not implemented here

### 6.1 Epsilon-mixing

**Denotation.** Let `S_t` be round `t`'s post-dedup survivors and `π_t` the Guide's descending
score order over `S_t`. Let `σ_t` be the *unguided sweep order* over the same set (rule index, then
class index, then node index — exactly `find_rewrite_matches`'s enumeration order). With
`ε ∈ [0, 1)` fixed at deploy, the round's application sequence is the deterministic interleave in
which every `⌈1/ε⌉`-th slot is filled by the first not-yet-fired candidate of `σ_t` and every other
slot by the first not-yet-fired candidate of `π_t`; `ε = 0` is the pure guided order. No RNG: a
fixed share of budget spent in unguided order, so a rule the Guide suppresses still fires — at
deploy, so a confidently wrong Guide cannot close a rule out of the reachable closure entirely,
and in any future guided-run mint, so that suppressed rules still receive labels.

```rust
pub struct EpsilonMix { pub numerator: u16, pub denominator: u16 }  // ε = numerator/denominator; (0, n) is off
impl GuidedSaturation<'_, G> { pub fn with_epsilon(self, eps: EpsilonMix) -> Self; }
```

**Evaluation it needs.** A ladder arm per ε ∈ {0, 1/10, 1/4} on DEV classical at B = 100/200:
median ratio vs unguided@B (the cost of ε), and per rule the fired count within B for rules the
Guide scores in its bottom decile (the benefit — those are the rules ε exists to keep alive).
Accept an ε for deploy iff its median ratio is within the registered `Y` of ε = 0's and the
bottom-decile fired count is > 0 for every rule with a non-zero DEV strict rate. Reported as a
table; not a registered claim.

### 6.2 Out-of-support fallback

**Denotation.** `Support` = the set of L2 cells with `n ≥ n_thin` in the TRAIN mint the
checkpoint was trained on (stored in the checkpoint). A candidate whose L2 cell
∉ `Support` is **out of support**: it is removed from `π_t` and takes unguided order — the round's
sequence is `[in-support candidates in π_t order] ++ [out-of-support candidates in σ_t order]`, with
ε-slots (§6.1) drawing from `σ_t` over *all* unfired candidates as before. With `ε = 0` the
fallback candidates fire after the guided ones, in sweep order; with `ε > 0` they also share the
ε slots. The Guide's score for an out-of-support candidate is never consulted — an unseen cell is
not a prediction, it is an absence.

```rust
pub struct SupportTable { cells: HashSet<CandidateCell /* L2 projection */>, n_thin: u32, fingerprint: RuleSetFingerprint }
impl GuidedSaturation<'_, G> { pub fn with_support(self, support: &SupportTable) -> Self; }
```

**Evaluation it needs.** (1) The **trigger rate**: share of scored candidates that were out of
support, per evaluation set (DEV bands, `rc_dev`, `sh`, `bezier`; FINAL only at publication) —
this is the coverage table's set-difference column (§3.3 item 4) observed live, and it ships with
every Guide number (§5.3). (2) The ladder with fallback on vs off on `sh` at B = 100: the
domain-shift table gets a column, and the Round 1b H_shift test is read with fallback on (the
deployed configuration). A trigger rate above 10% on any DEV set is a coverage finding: the
generator's fill list (§4.4) for the next mint is exactly those cells.

## 7. Explicitly NOT the way forward: adaptation at compile time

JP, explicit (2026-09-01): adapting the Guide on the user's own kernels at compile time is not the
direction. In JP's terms — three reasons, each sufficient:

- **History-dependent compiler output.** The same kernel would compile differently depending on
  what the compiler had seen before; a compiler's output must be a function of its input.
- **On-policy labels.** Labels minted from the Guide's own ordering are a property of that ordering,
  not of the e-graph (Round 2 §7.3 already names the residual for *unguided*-order labels; making
  the order itself learned turns a residual into a feedback loop).
- **No cold-start fix.** Adaptation does nothing for the first kernel, which is the kernel the
  user is compiling.

Training is offline; the model is one fixed, versioned artifact; the conditional is the local
algebraic context, which is a function of the expression; coverage of that conditional is the
generator's job, gated by the coverage table and the domain-shift table.

## 8. Build-phase entry points (the api_notes, in one place)

| Crate / file | Item | Kind |
|---|---|---|
| `pixelflow-search/src/egraph/candidate.rs` | `K_DOWN`, `OpHistogram`, `OpPairHistogram`, `ClassSummary` (+`ABSENT`), `ParentHistogram { hop1, hop2: OpPairHistogram, parents1, parents2 }`, `CandidateState`, `CandidateContext { down, down_hop2, up, state }`, `RoundSnapshot::take`, `Firing { node_idx, action }`, `CandidateFeatures { context }`, `CandidateFeatures::observe(egraph, snapshot, firing)`; all context types `Serialize`/`Deserialize` under `std` so the JSONL carries the struct whole | additive types + the one constructor |
| `pixelflow-search/src/egraph/extract.rs` | `ExtractedDAG.class_costs: Vec<Option<usize>>` — the DP's own `best_cost` vector, returned instead of dropped | additive field |
| `pixelflow-search/src/nnue/guide/scoring.rs` | `CANDIDATE_INPUT_DIM = 4 * K + N_SCALARS` (148); `forward_candidate` fed by a `GraphAccumulator` over the candidate's local edge set (§5.6); `accumulator.rs` unchanged | nonlinear family (not in this round's gate) |
| `pixelflow-search/src/nnue/mod.rs` | `pub(crate) fn shader_op_prior() -> OpMap<u32>` lifted from `ExprGenerator::shader_weight`; `substitute_template_arena` reused as-is | harvest |
| `pixelflow-pipeline/src/training/quarantine.rs` (deleted) | `pub fn cross_form_agreement(lhs: (&ExprArena, ExprId), rhs: (&ExprArena, ExprId), grid: &QuarantineGrid) -> CrossFormVerdict` lifted from `bench_extraction_3way.rs`, which is re-pointed at it | lift |
| `pixelflow-search/src/egraph/cell.rs` (new) | `Group`, `group_of(OpKind)`, `DownSig`, `DcostBucket`, `BudgetBucket`, `CandidateCell`, `CandidateCell::of(&CandidateFeatures)`, `CandidateCell::l1()/l2()`, `reachable_cells(rules: &[Box<dyn Rewrite>]) -> ReachableCells` | bucketing + enumeration |
| `pixelflow-search/src/egraph/rewrite.rs` | `RewriteAction::table_cost(&RoundSnapshot, &CostModel)` (pub(crate)); `lhs_template`/`rhs_template` for the 32 currently-untemplated rules minus {constant-fold, differentiate} (30 rules) | additive |
| `pixelflow-search/src/egraph/graph.rs` | `find_rewrite_matches_with_actions()` | additive sibling |
| `pixelflow-search/src/egraph/saturate.rs` | mint hook `saturate_until_applications_observed(egraph, root, budget, hook: impl FnMut(ApplicationId, CandidateFeatures))`; `GuidedSaturation::{with_epsilon, with_support}`; snapshot taken once per round before scoring | additive; production `saturate*` untouched |
| `pixelflow-search/src/nnue/guide/mod.rs` | `CandidateSummary { context }`; `ContextLinearGuide` (§5.5) implementing `SaturationGuide` | additive |
| `pixelflow-pipeline/src/training/split.rs` | `[rule_conditioned]` table; `rc` name form in the fence parser; exactly-one-tier check for named families | manifest |
| `pixelflow-pipeline/src/training/rule_conditioned.rs` (new) | `instantiate_lhs`, `embed`, `is_pay`, `is_enabler_case`, `ENABLERS`, `fill_plan(coverage.json)` | generator |
| `pixelflow-pipeline/src/bin/gen_rc_corpus.rs` (new) | `--rule-set`, `--manifest`, `--coverage` (previous report), `--n-per-rule 2000`, `--out-train corpus_rc_train.bin --out-dev corpus_rc_dev.bin`; entry names `{tier}_rc-{rule_name}_b{band:02}_f{seed:02}_{idx:05}` | binary |
| `pixelflow-pipeline/src/bin/gen_strict_labels.rs` | in-sweep observation (§1.3); JSONL record gains the full `CandidateContext` and the L1/L2/L3 cell | changed mint |
| `pixelflow-pipeline/src/bin/guide_coverage.rs` (new) | `--labels <jsonl>... --rule-set --out docs/results/<date>-guide-coverage.{json,md}`; §3.3 format; cell-oracle AUC | binary |
| `pixelflow-pipeline/src/bin/train_guide.rs` | `--model {linear, context-linear}`; checkpoint carries `Support`, fingerprint, corpus MD5s, coverage hash | changed trainer |
| `pixelflow-pipeline/src/bin/skew_test_linear_guide.rs` | extended to `ContextLinearGuide`: trainer forward vs deployed `score_candidates`, bit-exact over ≥ 1,000 DEV records including every context field | mandatory |
| `pixelflow-pipeline/src/bin/phase3_at_budget_eval.rs` | arms for ε ∈ {0, 1/10, 1/4} and fallback on/off; per-set trigger rate; ships the three tables | evaluation |

Order of work: types + cell + coverage tool on the *existing* Round 1 label files first (the
coverage table of the natural-frequency corpus is itself a result — it is the number that says how
much of the reachable context space Round 1 ever saw); then the mint hook and skew test; then the
generator; then the context Guide; then ε/fallback. Each step lands with its tests and with no
change to production saturation.

## 9. Revision log

- **2026-09-01, first commit (`356de421`):** the denotation, the cell, the coverage table,
  rule-conditioned generation, the training-distribution policy, ε-mixing and fallback, and §7.
- **2026-09-01, survey pass (this revision), per JP's instruction to inventory existing code
  before writing types.** Additive changes only; no committed number in §2–§3 moved:
  - §0.5 added: the REUSED / SUPERSEDED / NOT-APPLICABLE inventory, the harvest-vs-leave lists,
    and the two implementation facts (no parent lists; no class age).
  - §1.4: `ParentHistogram::hop2` and the new `CandidateContext::down_hop2` are **pair**
    histograms (`OpPairHistogram`), because the `GraphAccumulator` 2-hop binding section is a sum
    of products over edges and cannot be realized from two marginals. The hop-2 marginals are
    projections of the pairs, so nothing the first commit's linear model (§5.5) reads is lost.
  - §4.1: names `substitute_template_arena` as the instantiation primitive and `shader_op_prior()`
    (lifted from dead code) as the subterm op prior; §4.2: names `cross_form_agreement` lifted from
    `bench_extraction_3way.rs`.
  - §4.5: rule-conditioned entry names carry `Rewrite::name()` (the rule family) rather than the
    rule index, which is rule-set-relative under Round 2.
  - §4.6: flags the `normalize`/`poly`-vs-FINAL and `fourier`-vs-`sh-direct` form collisions for
    JP's decision, with two collision-free candidates; does not change the list.
  - §5.1: splits TRAIN into a primary arm (natural TRAIN bands excluded, per JP's demotion) and a
    secondary ablation arm (included), plus the distribution-only ablation; the first commit's
    single TRAIN definition is the secondary arm.
  - §5.6 added: the nonlinear family's realization as `GraphAccumulator` sections over the
    candidate's local edge set, with the latency prior arriving through `OpEmbeddings` dim 0.
  - §8: rows for `ExtractedDAG.class_costs`, `scoring.rs`, `shader_op_prior`, and
    `cross_form_agreement`.
