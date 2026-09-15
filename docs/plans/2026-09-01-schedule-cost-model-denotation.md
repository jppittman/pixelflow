> **Retracted/Superseded (2026-09-07), ledger L088.** The premise that "the cost model does 95% of the work" and the reranker over swap neighbourhoods as the next step are withdrawn - the number is a cross-population Spearman on a synthetic corpus, and B is not a step toward C (`2026-09-06-egraph-at-production-scale.md` section 7); the denotation of schedule cost as table plus residual stands. Verdict and rationale: `docs/results/2026-09-07-claims-ledger.md` (PR #1207); the corrected benchmark and re-validation order: `docs/plans/2026-09-07-benchmark-correction.md`.

# The schedule cost model: denotation first, code when there is something to learn

**Date:** 2026-09-01
**Status:** denotation. No code is built on it today; it says what PR #1093
keeps as a seam, why, and what would have to be true before anything is
trained again.
**Decision it records (JP, 2026-09-01, one of three options):** *"Delete the
shape, keep the denotation."* Verbatim, same evening: *"That was the design. I
think we cut too early. Because the expressions are growing. The egraph already
has schedules to choose. Egraph extraction is the place where code gen's
schedule choice is going to go."* and *"I don't think what we have was the
correct shape. I think it's right as an idea."*
**Companions:** the closed program's paper
(`docs/paper/2026-08-egraph-nnue-parity.md`, branch `claude/workshop-writeup`,
PR #1072 — not on this branch; see §8), the domain model
(`2026-08-17-cost-model-domain.md`), the dead-code inventory
(`2026-09-01-dead-code-with-ideas.md`), the scheduling designs
(`docs/designs/lattice-scheduling-types.md`,
`docs/designs/BRAINSTORM_VARIANCE_EGRAPH.md`, `docs/designs/LATTICE_EVAL.md`)
and the codegen plans that create the schedule choices
(`2026-09-01-loop-aware-codegen.md` = #1080,
`2026-09-01-register-allocation-escape-hatches.md`).

---

## 1. Two regimes, one boundary

### 1.1 Extraction today: `cost : Node → cycles`, additive, DP-exact

The e-graph holds equivalent *forms* of an expression. Extraction picks one
node per reachable e-class; codegen then schedules whatever was picked
(`plan_collapse_hoist` hoists by variance after the fact — the extractor never
sees a loop). So the object being priced is a form, the price of a form is a
sum over its nodes, and a 50-entry table (`latency_prior_cycles()`) *is* that
function to within measurement noise:

| what | number | where |
|---|---|---|
| corpus-wide ranking, bare transcendental/divide count | Spearman ρ = 0.949 | paper §6 |
| corpus-wide ranking, the handwritten table | ρ = 0.944 | paper §6 |
| corpus-wide ranking, trained NNUE (best checkpoint) | ρ = 0.989 | paper §5.1 |
| Round 2a, NNUE-driven / table-driven extraction, geomean | **1.0037**, 95% CI [0.9982, 1.0089] — tie | paper §5.2 |
| Round 2b (contrastive objective) | 1.0153, CI [1.0097, 1.0213] — regression | paper §5.3 |
| Round 3 (embeddings trained, best calibration) | 1.0082, CI [1.0031, 1.0133] — regression | paper §5.4 |
| A/A noise floor | ±0.07% | paper §5.2 |
| extraction with the table vs no extraction | 0.54 (≈2×) | paper §9 |

The model was a *better ranker* on every intrinsic metric and a *worse
extractor* every time it got sharper. That is not a near miss; it is the
signature of a regime where the objective is additive, the DP already
minimizes it exactly (over unfolded tree cost — paper §6 states the DAG gap),
and the per-decision residual sits at the per-pair measurement floor
(≤84% of paired decisions smaller than their own kernel's repeat spread, an
upper bound — paper §6). On schedule-free kernels there is nothing left for a
learned cost to decide. **The tie is a measurement of the corpus, not a
verdict on the design.**

### 1.2 Extraction with schedules in the e-graph: `cost : Extraction → cycles`, non-additive

The scheduling designs have said since April that the e-graph is where
factoring is discovered (`lattice-scheduling-types.md`, "Everything Is
Factoring": LICM, CSE, tiling, register placement are one operation —
identify what is independent, assign it to a resource — driven by the
variance lattice) and the codegen work of this week is putting the resources
the e-graph would assign to into the compiler: the codegen lattice
(`LoopShape`, loop axes as a compile-time fact — #1080 §1), the three-region
schedule (prologue / row / batch, each with its own back edge — #1080 §3),
hoisted values that may live in registers instead of being unconditionally
spilled to `hoist_slots` (#1080 §4, escape-hatches §D), and the binder loops
of `Kernel::over` (variance bits 4–8) as further levels. Once those are
choices *in the e-graph* — a hoist as a rewrite, a level as an e-node
attribute, a tiling as lattice nesting, a placement as a class of
register-lattice nodes — an extraction is no longer a form; it is a form
**plus a schedule**, and its cost is:

- **non-additive** — the same op costs `cycles(op) × trips(level)` and the
  level of a parent depends on the levels of the children it was given; the
  values hoisted to a level compete for one register file; ops at the same
  level compete for ports; a chain's latency is a max, not a sum;
- **not DP-exact** — the per-class independence the DP needs breaks exactly
  at those interactions.

This is the Halide regime. Adams et al. 2019 (*Learning to Optimize Halide
with Tree Search and Random Programs*, SIGGRAPH) faced the same object — a
schedule over a fixed algorithm — and the shape that worked was a cost model
over **per-stage, schedule-derived features** (loop extents, vector width,
footprint, arithmetic per stage), a small network that weights hand-derived
per-stage terms rather than predicting a total from a bag, **beam search**
over schedules with that model reranking, and **random programs** for
coverage. Every part of that maps onto what §2 keeps.

### 1.3 The boundary is a property of the corpus, not of the model

The only difference between the regimes is whether the kernels being extracted
have *schedule alternatives*. A kernel with one admissible level per node is
in regime 1 whatever the model; a kernel where a distributivity rewrite moves
`sin(Z)` from the pixel loop to the frame prologue (`sin(Z)·X + sin(Z)·Y →
sin(Z)·(X+Y)`, `BRAINSTORM_VARIANCE_EGRAPH.md`) is in regime 2. The closed
program trained and evaluated exclusively on regime-1 kernels. Its residual
was empty by construction.

---

## 2. The denotation

```
cost(E) = analytic(E) + residual(E)          for an extraction E
```

### 2.1 `analytic` — exact for the schedule the e-graph encodes

```
analytic(E) = Σ_{op ∈ E}  cycles(op) · trips(level(op))
```

- `cycles : OpKind → cycles` is the measured table, `latency_prior_cycles()`,
  re-derived by `measure_latency_prior` whenever the lowering changes (the
  register-allocator work forces a re-measurement before anything is compared
  across it — paper §9).
- `level : op → {const, frame, scanline, pixel, binder₄…₇}` is the shallowest
  scope that binds the op's variance under the codegen lattice — the
  `Variance → scope` map of `BRAINSTORM_VARIANCE_EGRAPH.md` §"Two-Lattice
  Architecture", made concrete by `LoopShape` (#1080 §1) and the three-region
  schedule (#1080 §3). It is a property of the **chosen** nodes (P1(c) in the
  domain model: the class-wide meet lies once a rewrite merges a constant
  into a pixel-varying class), never of the e-class.
- `trips : level → count` is the lattice's extent product for that level:
  1 for the prologue, `rows` for the row region, `rows·groups` for the batch
  region, the binder extent for a reduction. Under JIT-first the geometry is
  known at compile time (recompile-per-resize is deliberate —
  `jit-first-rationale`), so `trips` is a number; where an extent is
  `Bound::Unknown` until call time, `analytic` is a vector indexed by level
  and extraction orders it lexicographically, innermost level first.

This is the objective of the ILP in `lattice-scheduling-types.md` §"The
Solver Spectrum" — `minimize Σ evals_per_frame(scope(e)) · cost(node(e))` —
**without** its pressure constraint. Without pressure it is DP-friendly: the
DP state becomes `(e-class, level)` instead of `(e-class)`, at most five or so
levels wide, the recurrence is
`best(c, ℓ) = min over nodes n of c with level(n) ⊑ ℓ of cycles(n)·trips(ℓ) + Σ_children best(child, ℓ_child)`,
and it is exact for `analytic` for the same reason today's `extract_dag` is
exact for the table. Nothing learned is needed to get this far, and it is the
part that is worth the 2× today plus whatever hoisting is worth tomorrow.

**Where `analytic` stops being DP-friendly** — the interactions, which is also
the list of what `residual` is for:

| interaction | why the per-class recurrence cannot see it |
|---|---|
| register pressure per level | `Σ hoisted_to(ℓ) ≤ registers(ℓ)`; the (k+1)-th hoisted value at a level costs a spill+reload per trip of the level below, which depends on *every other* choice at that level (escape-hatches §D: today every hoisted value is spilled unconditionally, so the term is a constant; after #1080 it is a choice) |
| port contention / ILP at a level | the cost of two independent ops on the same execution port is not the sum of their latencies; a chain's cost is its critical path |
| sharing | the DP charges a shared subexpression once per reference and `choices_to_arena` materialises it once (paper §6); the reload traffic for the shares is a cross-node quantity |
| footprint | spill slots and hoist slots are memory; their cost is a function of the whole level's working set |

Each row is a function of a *set* of nodes, not of one node. That is the
definition of non-additive, and the reason no table entry can absorb it.

### 2.2 `residual` — learned, reranks, structured by level and by interaction

```
residual(E) ≈ measured(E) − analytic(E)
```

- **Role: reranker, never replacement.** The analytic DP produces its
  optimum and either a top-k of near-optimal extractions or the swap
  neighbourhood around the optimum (the search core `IncrementalExtractor`
  had: enumerate per-class alternatives, cycle-check each via
  `Extraction::try_swap`, accept the best improvement, iterate to fixpoint).
  `residual` reorders those. It cannot move the extraction to a point the
  analytic part did not already admit, so the table's ~95% is never at risk
  — the failure mode of Rounds 2b/3 (a sharper model walking to its own
  optimistic minimum, paper §6 "proxy-consumption gap") is made
  unrepresentable by the interface rather than guarded against.
- **Inputs, by level.** For each level ℓ: op counts and edge sums restricted
  to the nodes at ℓ — the per-level sectioned edge accumulator of §4
  (`Σ E[parent]`, `Σ E[child]`, node and edge counts, per level); a pressure
  estimate at ℓ (live hoisted values vs the level's register budget); chain
  depth at ℓ (critical path in table cycles); footprint at ℓ (spill/hoist
  slots). Halide's per-stage features, with "stage" = "level".
- **Inputs, by interaction.** Edges that cross levels (a pixel-level op
  reading a frame-hoisted value is a reload edge; the deleted walker already
  emitted `Var` reload edges for shared references), and the same-level
  co-occurrence the port model needs. These are the features that a
  fixed-width bag-of-edges provably blurs (paper §6: two schedules with
  different port pressure "can still land close together").
- **Training set: only kernels with real schedule alternatives**, i.e. those
  whose e-graph admits ≥2 extractions with distinct level assignments and
  analytic costs within a band narrow enough that the residual has to
  decide. Family-held-out tiers, feature-quotient fencing, schema-derived
  identities, the same-form oracle gate, sentinel-normalised labels — the
  harness the paper describes as its reusable contribution (§4), unchanged.
- **Target: the residual**, in log-ns or table cycles, never the total. A
  model that predicts `measured − analytic` and is wrong by its whole output
  degrades to the analytic DP; a model that predicts `measured` and is wrong
  by 1% loses to it (that is Round 3).

### 2.3 The interface the seam has to carry

```rust
pub trait Reranker {
    fn score(&self, extraction: &Extraction, arena: &ExprArena) -> f64;
}
```

`score` returns the full `cost(E)`; the analytic part is computed inside it
from the table and the levels so the caller cannot combine the two halves
wrongly. The swap search is generic over `Reranker`; the trivial
implementation (score = `analytic`, i.e. the table times trips, with trips ≡ 1
until levels exist) must reproduce `extract_dag`'s choice on every corpus
kernel — that identity is the regression test that says the search core is
correct independent of any model. No learned implementation ships in #1093.

---

## 3. Why the previous shape lost, item by item

| what the shape did | what the denotation does instead | why it matters |
|---|---|---|
| predicted **total** cost from a bag of edges and tried to **replace** the additive table (`ExtractionPolicy::Nnue` swapped the cost function wholesale) | `analytic` is computed, not learned; `residual` is the only learned term and only reranks | every time the model got sharper it deviated further from the additive truth on the 95% it could not improve and lost end to end — Round 2b 1.0153, Round 3 1.0082, both with the CI clear of parity (paper §5.3–5.4) |
| **multiset** features — flat and depth-encoded parent/child sums over the whole DAG; variance entered only as four scalar node *fractions* | features indexed **by level** and **by interaction** — what sits in which loop, pressure at that loop, chain depth, cross-level reloads | aggregation into one fixed-width vector cannot separate two schedules with the same edge multiset and different port pressure (paper §6); the residual is *made of* exactly those separations |
| trained on **schedule-free** kernels (one admissible level per node) | trained only on kernels with ≥2 schedule-distinct extractions | there was no residual in the labels; the model learned the table plus noise, and noise is what it spent its deviations on (Round 3's flip analysis: preferring near-doubled trees it scored as cheaper, paper §5.4) |
| **replaced the DP** with local search from the model's own scores | DP over `(class, level)` stays exact for `analytic`; local search only reorders the DP's neighbourhood | walking toward a learned minimum concentrates decisions where the model is wrong-and-optimistic; walking a fixed neighbourhood bounds the damage to the neighbourhood |
| justified the architecture by **incrementality** (NNUE's O(Δ) update) | incrementality is irrelevant to a reranker over k candidates; it is kept only where it pays (the Guide, §6) | measured Δ at extraction is 44.9% of edges (≈2×, paper §5.6), so the signature capability bought nothing here |
| measured **decisions at the noise floor** | the schedule-alternative corpus is selected for decisions the analytic part cannot make and the measurement can resolve | a residual that exists only below ±0.07% is not a residual; the corpus definition in §2.2 is what makes the target measurable |

What was **right** and is kept: the per-level idea (the four variance
fractions were per-level features in embryo — Halide's per-stage features
before there were stages), the typed edge stream (a feature walk that is a
value can be replayed for training and for parity tests), the prior-seeded op
embeddings (an untrained op already carries the table's cost), the swap search
as the reranking primitive, and the harness.

---

## 4. What #1093 keeps, and why; what it deletes, and why

### Kept (the seam)

| item | where | role in the denotation |
|---|---|---|
| `OpEmbeddings` + `init_with_latency_prior` / `new_with_latency_prior` | `pixelflow-search/src/nnue/factored.rs` | the op vocabulary for `residual`'s features and for the Guide; dimension 0 seeded from `latency_prior_cycles()` so the learned and analytic parts share a scale from step 0 and cannot drift (`cost.rs` header) |
| typed edge stream `CostEdge` / `PeSlot` / `EdgeTrace` / `EdgeSink`, one walker over `CostDag` (arena and extraction adapters) | `nnue/factored.rs` | the feature walk as a value: the same fold the forward pass runs is what training differentiates and what the train/deploy parity test replays (`arena_and_extraction_walks_record_the_same_edge_stream`) |
| per-node variance classification — `variance_histogram(arena)` (`pub(crate)`, over `pixelflow_ir::variance::compute_arena_variance`; the rule: `is_const` → const, x-invariant and y-independent → frame, x-invariant → scanline, else pixel) and `Extraction::chosen_variance()` (classifies the *chosen* nodes by materialising them through `choices_to_arena`, P1(c)) | `pixelflow-search/src/nnue/factored.rs`, `egraph/extract.rs` | the level assignment §2.2's by-level features index on. **No accumulator ships.** #1093 (b) first restored the old `EdgeAccumulator` verbatim under the name `LevelSectionedEdges` — four sections that were flat/depth-encoded × parent/child edge sums over the *whole* DAG, plus this histogram as four scalars; level-*aware*, never level-*indexed* — which was the deleted shape under a new name, and the follow-up commit (`ba042c41`) removed it again, keeping only the classification and its arena-vs-extraction parity test. The level-indexed accumulator — sections indexed by `const, frame, scanline, pixel` (extensible to binders), each holding its own edge sums, classification taken from the chosen nodes — is step 2 of §5.1, built when a level is an e-node attribute, not before |
| `Extraction` + swap search behind `Reranker` | `pixelflow-search/src/egraph/extract.rs` | the reranking primitive: `try_swap`'s cycle-checked candidate construction and the best-improvement loop, generic over `Reranker` (`IncrementalExtractor<'a, R: Reranker>::new(&reranker, top_k)`), with **no implementation shipped** — the test-only `TableReranker` (the table cost itself) is the trivial §2.3 reranker, and `swap_search_reproduces_extract_dags_choice_on_a_cost_ambiguous_class` is the `extract_dag`-identity contract |
| `jit_bench` / `BenchSession` / sentinels, `training/{corpus,split,quarantine,structural}`, `mint`'s normalisation half, `journal` | `pixelflow-pipeline` | the label source and the self-censoring harness; `LocalNs`/`SessionNs` make drift-then-overhead the only well-typed order |
| `measure_latency_prior` | `pixelflow-pipeline/examples/measure_latency_prior.rs` | re-derives `analytic`'s table on the JIT's own lowered form; must be re-run behind the register-allocator work before any comparison crosses it |
| provenance / labeler / saturate, `GraphAccumulator`, `SaturationHead` | `pixelflow-search/src/egraph`, `nnue/guide` | the Guide's substrate (§6) |

### Deleted (the shape)

| item | why it does not survive |
|---|---|
| `ExprNnue` value head, TRIF checkpoint, `PIXELFLOW_NNUE_WEIGHTS`, `ExtractionPolicy::Nnue` | a learned *total* cost replacing the table — the one thing the denotation forbids |
| `bootstrap_extraction_head`, `unified_backward`, `training/episodes`, `bench_extraction_3way`, `profile_extraction`/`egraph::profile` | trainer, gradient and gate for that head, on a regime-1 corpus; the Guide branches (`train_guide`, `gen_strict_labels`, `phase3_*` on `claude/phase3-guide`) import none of them — `unified_backward` and `mint` appear there only in doc comments; the one real conflict is `nnue/guide/{mod,scoring}.rs`'s import of `EdgeAccumulator`/`ExprNnue`, which #1093's `Guide`-holds-`OpEmbeddings` shape resolves (`guide/mod.rs` now imports only `{EMBED_DIM, K, OpEmbeddings}`; PR body, conflicts table) |
| the NNUE-specific scoring inside the swap search | replaced by the `Reranker` seam; the search itself is kept |
| `CostModel` TOML persistence, `calibrate_costs`, the four scalar examples, `nnue/mod.rs` legacy constants, `NNUE_TRAINING.md` (inventory D3) | dead, or a `$HOME`-probing silent override of the production cost model |

Anything on this list that a future round needs is in VCS at `origin/main`
before #1093; the paper's `NUMBERS.md` and `docs/results/journal.jsonl` keep
the measurements.

---

## 5. Sequencing, the trigger, the first experiment, the kill

### 5.1 Nothing to learn yet

Today no rewrite in the e-graph changes a level; levels are assigned after
extraction by `plan_collapse_hoist`, and every hoisted value goes to memory.
`residual` over that corpus is identically the noise the closed program
learned. So: **no model, no trainer, no corpus until the e-graph encodes a
schedule choice.** The order of work is fixed by that dependency:

1. #1077 / #1082 (the kernel ABI as a type; one compile entry — both merged
   2026-09-01) → #1080 stages 0–2 (`LoopShape`
   in the cache key; three-region schedule; hoisted values allocatable to
   registers). This is where schedule *alternatives* first exist in codegen.
2. Level as an e-graph object: the variance analysis already exists
   (`pixelflow-ir/src/variance.rs`, `DepsAnalysis`); the step is making the
   level a per-node attribute the extractor prices, and the `(class, level)`
   DP of §2.1 with `trips` from the lattice. This is `analytic` — worth
   building and measuring on its own, with no learned part.
3. Only after (2) shows a measurable gap between the analytic DP's choice and
   the best measured extraction on some corpus — the residual made visible —
   does the reranker get an implementation.

### 5.2 The trigger, stated so it can be checked

The reranker work opens when **both** hold: (a) an e-graph over a
production-shaped kernel admits ≥2 extractions with distinct level
assignments, and (b) on a family-held-out DEV tier of such kernels the
best-of-k *measured* extraction beats the analytic DP's choice by a geomean
whose 95% CI clears the ±5% band with the A/A floor re-established. (b) is the
oracle headroom; without it there is no residual worth a model.

### 5.3 The first experiment

Three arms on kernels **with** schedule alternatives, same protocol as the
paper's Rounds (median-of-samples, sentinel record-and-normalise with
regime-change abort, same-form oracle gate, family-held-out tiers,
feature-quotient fence, pre-registered ±5% band, verdict that self-censors on
censoring rate and gate failures, journal line per run):

| arm | what chooses the extraction |
|---|---|
| A | analytic `(class, level)` DP alone |
| B | analytic DP + `residual` reranking its top-k / swap neighbourhood |
| C | measured oracle: the best of the same k candidates by direct measurement |

Report B/A and B/C (how much of the oracle headroom the residual recovers),
never a Spearman on a fixed distribution as the headline — intrinsic metrics
failed to transfer three times (paper §6). A pre-registered within-kernel
pairwise ranking accuracy on held-out families is the gate for *running* the
end-to-end arm, not a substitute for it.

**Data it needs:** a corpus minted by `gen_bench_corpus`'s generator with a
band whose templates contain hoistable structure (frame-uniform and
scanline-uniform subterms under distributivity/factoring rules — the
`sin(Z)·X + sin(Z)·Y` family, binder loops once `Kernel::over` is in scope);
per kernel, the k candidate extractions with their level assignments,
`analytic(E)` for each, and a paired measured label for each (k labels per
kernel, not one) — so `residual` targets exist at the decision granularity
that failed to be measurable before; all minted after `measure_latency_prior`
is re-run behind the register allocator, with the source-rev config key
forbidding comparison across that boundary.

### 5.4 The kill condition

Close the reranker again — and record it as a second honest negative in the
same paper — if any of:

- the trigger's (b) fails: the oracle C does not beat A outside the band on
  two independent DEV rounds (there is no residual to learn at this level of
  scheduling; revisit only when a new schedule dimension lands);
- B/A's CI sits **above** parity once with gates green (the 2b/3 signature:
  a residual that reranks toward slower code has learned the wrong thing and
  the interface is not enough of a guard);
- B recovers < 25% of the C headroom on two rounds after the model-side
  levers pre-registered for it are spent (the per-level features are not
  separating the schedules; go back to §2.2's feature list before any
  architecture change);
- the per-kernel paired intervals show the decisions B flips against A are
  themselves unresolved at the measurement floor (the residual is real but
  unmeasurable at this k; a bigger k or a different corpus, not a bigger
  model).

---

## 6. Relationship to the Guide: the same shape, one layer up

The Guide (`2026-08-31-guide-design-revision.md`, branch `claude/phase3-guide`)
has the same denotation with the roles rotated:

| | extraction (this doc) | saturation (the Guide) |
|---|---|---|
| the exact part | the analytic `(class, level)` DP | the saturation loop and the hindsight labeler (provenance as an audit log) |
| the learned part | `residual` reranks the DP's neighbourhood | the Guide reorders which candidate rewrites to apply under a budget |
| what it decides | the **schedule** — which form, at which level | **what enters the e-graph** — which rewrites are explored at all |
| training signal | `measured − analytic` on schedule-alternative kernels | strict / labeler provenance labels, cold-start, supervised, no critic |
| where incrementality pays | nowhere (44.9% Δ) — a reranker recomputes | here (91.1% no-op applications, 0.14% median Δ — paper §5.6) |
| evaluation | geomean at parity band, paired, self-censoring | quality-at-budget anytime curves, cost regret, never "reaches optimum faster" |

The shared substrate is deliberately the same objects: `OpEmbeddings`
(prior-seeded), per-level / per-neighbourhood features built from the same
typed edge walk, the budget discipline (pre-registered band, family-held-out
tiers, journal), and the rule that the learned part *reorders* what an exact
part produces and never replaces it. The Guide chooses which rewrites to
explore; extraction chooses the schedule among what was explored. A Guide
that learns to prize the rewrites that create hoistable structure and a
reranker that prices the hoists are two halves of one loop, and the domain
doc's J10 retirement clause applies to both: if there is nothing to reorder,
the seam is deleted whole.

---

## 7. Parked (scope cut, on purpose)

- **Whole-extraction pricing.** No model predicts `cost(E)` end to end;
  the total is `analytic + residual` or it is nothing. Revisit only if the
  residual is shown to dominate the analytic part on some corpus, which
  would be a different compiler.
- **A transformer (or any sequence model) over extractions.** The by-level
  features are small and structured because the decisions are few and
  local; a model that needs the whole extraction as a token sequence is
  answering the whole-extraction question above.
- **Generality beyond this compiler.** The denotation is written against
  this lattice (X innermost, Y/Z/W, binders), this table, this harness. It is
  Halide's shape specialised to a monomorphised, intermediate-free pipeline
  where the only schedule decision is *which scope each subexpression
  evaluates at* (`lattice-scheduling-types.md`, "The Monomorphization
  Advantage"); nothing here claims to transfer.
- **The self-hosting north star** (the cost model written as a pixelflow
  kernel, optimised by the e-graph it serves — paper §9) stands, and is
  reached through this seam, not around it: a `Reranker` that is a kernel is
  the first version that would be worth writing that way.

---

## 8. The paper's future-work paragraph

`docs/paper/2026-08-egraph-nnue-parity.md` is not on this branch (it lives on
`claude/workshop-writeup`, PR #1072). When it is amended, its future-work
must read: **closed for extraction over expressions; reopens as a residual
schedule-cost model when codegen has schedules to choose** — with the paper's own §9 and its four
"if revisited" items (re-run 2b, predict the residual over the DP, top-k
rerank, relaxation as a training mechanism) pointed at §2 of this document,
which is their denotation.

---

## 9. What `analytic` has to be fitted to (added 2026-09-05)

§2.1 defines `analytic(E) = Σ cycles(op) · trips(level(op))` and §2.2 lists
register pressure, footprint and cross-level reloads as the interactions
`residual` exists to absorb. Both halves are claims about what predicts time,
and until 2026-09-05 neither had been checked against a clock at a real shape.
It now has been, one layer down — over allocations of a fixed schedule rather
than over schedules — and the measurement is the thing any future version of
this document must be fitted to.

**The harness.** `pixelflow-pipeline`'s `collapse_cost` binary and its
`collapse_bench` module (see
`2026-09-01-register-allocation-escape-hatches.md`, the 2026-09-05 block, for
the full tables). It captures a corpus of kernels **with the shapes they are
baked at** as a fixture, compiles each exactly as `Lattice::bake` does, times
the emitted collapse kernel, and records the emitted code's static features
**per scope** of the collapse nest — so a trip count can weight them — as one
JSONL row per (kernel, allocation, tier, pass). `analyze` scores closed-form
predictors against those rows on two things: rank correlation across kernels,
and the **sign of the delta between two builds of the same kernel**, which is
the decision a cost model is actually asked to make.

That second score is the reusable part of the method here. Five allocations ×
two tiers × 208 kernels found predictors that rank kernels at ρ = 0.98 and get
the *direction* of a paired difference right 27% of the time — worse than a
coin — because ranking is dominated by how big a kernel is and the decision is
not. §5.3's Round-1 lesson ("report B/A and B/C, never a Spearman on a fixed
distribution as the headline") is the same statement about the layer above;
this is a worked instance with numbers, and the paired sign test is what makes
it visible cheaply, before an end-to-end arm is worth running.

**What it found, and what that means for `analytic`.** The quantity three
register-allocator policies were built to minimise — dynamic memory operations
per call — is *anti-correlated* with time on the comparisons those policies
turned on. The term that fixes it is **rematerialization**: a value the
allocator rebuilds instead of reloading is not a memory operation and was
therefore invisible to the metric, while being 3× larger under the policy that
lost. `Σ scopes (loads + stores + remats) × trips` gets the sign right 99.1%
(AVX-512) / 97.9% (SSE2).

Three consequences for §2:

1. **`trips` is load-bearing and already measurable.** Weighting by the scope's
   trip count is what separates a 98% predictor from a 78% one on the same
   counts (`static_mem_ops` against `dyn_traffic`), and `LatticeShape` already
   reaches the compiler — it is the emitter it does not reach. §2.1's `trips`
   is not a modelling assumption to be validated later; it is the largest
   single term measured so far.
2. **The footprint row of §2.2's interaction table is understated.** It says
   "spill slots and hoist slots are memory". The measurement says the cost of
   the (k+1)-th hoisted value is *not* only its memory traffic — an allocator
   that avoids the slot by rebuilding the value pays a different, larger price
   — so a pressure term written in loads and stores will mis-price exactly the
   trade it exists to price. Whatever `analytic` charges for pressure has to
   charge for rematerialization too.
3. **The candidate list has to come from what the emitter emits.** Every
   predictor tried before this one was a variation on the losing quantity, and
   the winner was not reachable from that vocabulary. `emit::traffic` counts
   loads, kept loads, stores and rematerializations apart from each other for
   this reason: it does not assume which of them costs, so the next such
   question is answered by the data rather than by the metric that was already
   chosen.

**What the predictor cannot see, and the first term `residual` is for (added
2026-09-06).** `Σ (loads + stores + remats) × trips` weights every schedule
entry as if it executes, so it is blind by construction to control flow inside
the collapse body — and the emitter has some: a `Select` whose arm is exclusive
to it is skipped, per batch, when the mask is uniform. S3b of
[the lattice plan](2026-09-06-kernel-with-a-lattice.md) made that fire on a
scene (642 of 401 body entries under a guard, ranges nested) and the harness
scored the change at **+40.8% (SSE2) / +32.1% (AVX-512)** trip-weighted memory
operations while the clock it was validated against moved **−7.2% / −8.3%**.
Both halves of that are real: a guard forces values live across the branch into
slots, which the metric counts in full, and it removes a whole range from most
batches, which the metric cannot count at all.

That is not a defect to patch with a branch-probability guess. It names the
first concrete **profile-dependent** term the learned residual of §2.2 exists
for: **mask coherence** — how often a mask is uniform across a batch, and how
predictable that is. It is a property of the *data* a kernel runs on, invisible
to any static analysis of the expression, and it is exactly what decides
whether the same guard is a 3× win (a sphere's silhouette: uniformly false in
97% of batches) or a 3× loss (a glyph's coverage: varying per lane almost
everywhere). Codegen ships without it by bounding the downside instead —
refusing a guard whose arm costs less than a mispredict — which is sound and
leaves the whole upside unclaimed. Claiming it needs a measured coherence per
mask, per shape, which is a residual over the analytic table and nothing the
table can hold.

Two method notes it also settles, cheaply: `analytic` will have to carry a
`trips`-like weight that a *branch* can reduce, or the two regimes will
disagree in sign the moment codegen chooses control flow; and the harness's
per-kernel wall clock is not trustworthy below ~10% on a shared host — two runs
of byte-identical machine code measured 4× apart on the smallest glyph kernels,
against an A/A floor inside each run of ~1%, so the corpus's aggregate is the
number to read and a per-kernel ratio is not.

**The trigger this does not satisfy.** §5.2 opens the reranker work when an
e-graph over a production kernel admits ≥2 extractions with distinct *level*
assignments and a measured oracle beats the analytic DP. Nothing here creates
a level alternative — the five variants share one schedule and differ only in
allocation — so §5.1's "nothing to learn yet" stands unchanged. What has
changed is that when levels do become choices, the instrument that says whether
a level assignment was a good one already exists, and it measures at the shape
rather than at a fixed 64-tuple input buffer the way `BenchSession` does.
