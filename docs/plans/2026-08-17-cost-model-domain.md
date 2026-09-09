> **Superseded in large part — see the 2026-09-01 closure note immediately below,**
> which lists row by row what has no object left. Roughly a dozen files this
> document's module map names were deleted with the extraction-head program
> (`nnue/edge.rs`, `nnue/head.rs`, `nnue/embeddings.rs`, `nnue/serialize.rs`,
> `src/eval/gates.rs`, `src/eval/verdict.rs`, `src/bench/*`, `oracle_compare.rs`,
> `pixelflow-ml/src/nnue.rs`). The rows marked live in that note (J2, J7/J8, J10,
> J11, J15) are the part still worth reading.

# Cost-Model Domain Model and Reorganization Plan (2026-08-17)

Synthesis of three independent analyses over the cost-model pipeline
(pixelflow-search/src/{nnue,egraph}, pixelflow-pipeline/src/{training,bin,jit_bench.rs}):

> **2026-09-01 closure note (annotation, not a rewrite).** The extraction-head program this
> domain model was written for is closed honest-negative (workshop paper on branch
> `claude/workshop-writeup`, PR #1072, closed without merging — not in this tree; see
> `2026-09-01-schedule-cost-model-denotation.md`) and deleted per JP's ruling. Rows that are now dead-letter because their object no longer
> exists: **J3/J4** survive in `jit_bench` (harness, kept); **J6** (`VariantSet`) — PR #1044
> closed unmerged; **J9** — the weights sidecar half is deleted, `SchemaIdentity` survives
> for the corpus format only; **J12** (`RoundVerdict`) — `bench_extraction_3way` deleted;
> **A6** — `EdgeAccumulator` and its `remove_*` API deleted (the O(Δ) question is moot);
> §3 module-map rows `nnue/edge.rs`, `nnue/head.rs`, `src/eval/verdict.rs`, `src/eval/gates.rs`
> and the §4(b) "bench_extraction_3way.rs harness-library split" have no object left to
> split. **J2** (`Extraction`), **J7/J8** (`Fence`, `FenceKey`), **J10** (`SaturationGuide`),
> **J11**, **J15** (`JournalEntry`) remain live.

1. **Domain description + noun inventory** — what each stage MEANS, and whether each noun has a type, a convention, or nothing.
2. **Name/argument analysis** — 1,330 fns parsed; namespace-in-name families and argument clusters that always travel together.
3. **Dead inventory** — caller-verified live/dead disposition of the NNUE surface, plus the serialization consequence.

**Method (the join):** the analyses are independent probes of the same missing structure. A
candidate type is **CONFIRMED** when it appears in at least two probes — a prose noun that also
shows up as an argument cluster and/or a name-suffix family compensating for it. A single-source
candidate is checked against the roadmap (§0 north star of the 2026-08-05 research workflow;
guided-saturation Phase 3, 2026-07-07; Round-2 contrastive direction; research-notes §6 adoption
plan): roadmap-demanded → **denote now, implement only what the live path consumes**
(ROADMAP-ADMITTED, roadmap item cited so the seam can be retired if the roadmap changes).
Single-source without roadmap demand → unconfirmed; do not build. Disagreements between probes
are findings in their own right (§0.2).

Open P1 defect classes this reorganization must make **unrepresentable**:
- **(a)** cached ValueSample accumulators go stale while SGD mutates embeddings each batch
- **(b)** TRIE checkpoint schema never bumped when feature meaning changed
- **(c)** variance computed from class-wide meet instead of the chosen nodes
- **(d)** holdout fence keyed on structure while features collapse literals

---

## 0. The join table

Every later section references these rows by number.

| # | Candidate type | Domain noun (probe 1) | Argument cluster (probe 2) | Name family (probe 2) | Kills | Verdict |
|---|---|---|---|---|---|---|
| J1 | `Expr<'a>` (rooted arena) | "expression" §1 — *the* object of the whole pipeline; currently two positional params | `(arena, root)` — 37 sites, 12 files, both crates | `*_arena` suffix family (`benchmark_jit_arena`, `optimize_runtime_arena`, `from_arena_dag`, …) | argument-threading errors; the `*_arena` namespace smell | **CONFIRMED** |
| J2 | `Extraction` | "choice map" + "well-founded choice state" — both Convention; a validated choice vector is indistinguishable from an arbitrary `Vec<Option<usize>>` | `(egraph, root, choices)` — 10 sites, 3 files | `from_dag_choices` / `from_dag_choices_with_variance` / `choices_to_arena` / `backfill_well_founded` | **P1(a)**, **P1(c)** | **CONFIRMED** |
| J3 | `CostLabel` | "the label itself" — **Nothing**; five parts exist as separate pieces (bare f64, file-level `MintMetadata`, run-level `NormalizationStats`) | `ns` / `adjusted_ns` / `call_overhead_ns` / `SentinelContext` / `mode` traveling together on `BenchResult` | `normalized_label_ns`, `.normalization()` | drift-before-overhead ordering bug class; order-correlated label contamination | **CONFIRMED** |
| J4 | `SessionNs` / `LocalNs` clock newtypes | "clock context" — **Nothing**; the exact CLAUDE.md pattern (one f64, several clocks) | same fields as J3 | same | wrong-clock scaling by a future caller | **CONFIRMED** |
| J5 | `BenchRequest` | §4 measurement mechanics (mode, repetition, compile-cache state) | — | 8 `benchmark_*`/`measure_*` fns in jit_bench.rs (subject × repetition × compile spelled as suffixes) | every new measurement mode minting a new suffixed fn | **CONFIRMED** (denote now, implement next round — see §4b) |
| J6 | `VariantSet` | "rewrite-variant family / same-expression variant set" — **Nothing**; the central missing noun: the population ranking must actually operate over | absent from code (that absence IS the finding) | absent | the ranking-target mismatch — the measured 1.8% extraction regression | **ROADMAP-ADMITTED** [Round-2 contrastive training; research notes §1.5] |
| J7 | `Fence<T: TierMarker>` + `Family` newtype | "holdout fence" (guarantee = Convention), "generator family" (bare `(usize, u64)`), "leakage" (**Nothing**) | `report_holdout_fence(fence, live_counts, dropped, requested_synthetic, sidecar)` — 5 params | — | backwards-fence misuse (compiles fine today); family-split leakage | **CONFIRMED** |
| J8 | `FenceKey` = feature-quotient key | leakage precisely defined: structure the model has effectively seen appearing on the certifying side | — | — | **P1(d)** | **ROADMAP/DEFECT-ADMITTED** [P1(d); Round-2 needs a sound DEV tier] |
| J9 | `SchemaIdentity` (trait, §2) | "weights sidecar" typed but `weights_identity` is a one-off convention; "config hash" **Nothing** | — | "TRIE" magic never bumped; dead sections written unconditionally | **P1(b)** | **CONFIRMED** |
| J10 | `SaturationGuide` contract | §5 dead apparatus: graph accumulator, mask/policy head, rule encoders — zero non-test callers, zero gradient producers | `(parent_op, child_op, emb[, depth\|pe])` ×10 | 18 `{add,remove}_{edge,op_node,2hop_edge,leaf}[_at_depth\|_with_pe]` mutators + 6 scoring methods (`*_with_hidden`/`*_graph`) | the decay-of-dead-weights defect (Phase-3 weights randomly initialized, then weight-decayed with no opposing gradient, every run) | **ROADMAP-ADMITTED** [Phase 3, 2026-07-07 guided-saturation redesign] |
| J11 | `SaturationBudget` | saturation (implicit in §5/§7) | `(max_iters, class limits, timeout)` | `saturate_with_{budget,full_budget}` (saturate.rs) vs `saturate_with_{limit,limits}` (graph.rs) — **two independent implementations of the same loop** | silent drift between duplicate saturation loops (the `sin` anecdote, live) | **CONFIRMED** |
| J12 | `RoundVerdict` | DEV verdict, FINAL gate, "underpowered", noise floor — all **Nothing**/Convention; DEV-vs-FINAL enforced by control flow, not type | `Option<(f64,f64)>` CI + prose precedence in `verdict_text` | three colliding "Verdict"s (`quarantine::Verdict`, `PointVerdict`, the e2e `String`) | silently swapping a DEV geomean for a FINAL one (both plain `f64`); gate-precedence bugs | **CONFIRMED** |
| J13 | `EClassRef<'a>` | "a class inside this egraph" | `(egraph, class)` — 16 sites, 6 files | — | — | **CONFIRMED**, low priority |
| J14 | shared `run_scalar` (one copy) | oracle cross-check §3 | `(code, x, y, z, w)` — 8 sites, byte-identical bodies in ≥3 files (quarantine.rs comment admits "Verbatim from …") | — | a fix to one copy silently not propagating — NO SILENT FAILURES violation in waiting | **CONFIRMED** |
| J15 | `JournalEntry` + `ConfigHash` | "run-level research journal" — **Nothing**; a file-format convention two binaries could violate differently | — | — | unreproducible claims; structurally divergent journal lines | **ROADMAP-ADMITTED** [2026-08-05 workflow requires the journal for every round] |

### 0.1 Anti-rows — clusters/families the join says NOT to build

| # | Candidate | Why not |
|---|---|---|
| A1 | `Match{id, node}` for `Rewrite::apply` | Largest raw cluster (37 impls) but **already unified by the trait**. Both probes agree it needs nothing. Churn without payoff. |
| A2 | `GraphEdit` enum for GraphAccumulator's 18 mutators | Name family with **no live domain noun** — the roadmap (Phase 3) explains the gap. Elaborating a dead surface is implementation ahead of its consumer. Encapsulate (J10); if Phase 3 wants the enum, mint it then. |
| A3 | generic `Layer` MLP struct | The 3× hand-copied backprop (`backprop_{mask,rule,value}_mlp`) mostly **dies by deletion**: rule_mlp is legacy-dead, mask_mlp goes behind the Guide contract. Only `value_mlp` stays live — one copy, no abstraction needed. Subtract before you add. |
| A4 | fixing `set_rule` (9 params, worst offender) | Belongs to `RuleFeatures`, which is DELETE-shaped (superseded by `encode_rule_from_arena`). The sharpest argument-count violation in the corpus is resolved by subtraction, not a struct. |
| A5 | `GateKind` (same-form vs cross-form as one enum) | Prose-only, single-source, no cluster, no direct roadmap demand. The two gates live in different modules with different failure semantics; unifying them is a doc-level seam (§3 notes where each lives), not a type. Unconfirmed — do not build. |
| A6 | O(Δ) incremental extraction (`EdgeAccumulator::remove_*` consumers) | Research notes defect #4: measured median edge-multiset symmetric difference is 44.9%, so a correct O(Δ) path buys ~2×, not the chess-analogy ~98%. Keep the `remove_*` API public as-is (KEEP-uncertain, adoption-plan step 5); build nothing on it now. |

### 0.2 Disagreement findings (mismatches between probes)

- **Code grew structure the domain doesn't need yet:** the entire Guide surface (J10). The
  roadmap explains it; the disposition is encapsulation, not deletion and not elaboration.
- **The domain has a vivid noun with zero code footprint:** "leakage" and the "variant set" (J6).
  Leakage is a *property*, not a type — it becomes a theorem obligation discharged by J7+J8
  (family-typed split + feature-quotient fence). The variant set is a genuine missing type.
- **A convention the prose treats as load-bearing that code treats as free:** the clock a
  nanosecond is denominated in (J4). Nothing in the type system stops scaling the wrong field;
  the doc comment on `normalized_label_ns` is the only guard.
- **Stale ground truth caught by the dead-inventory probe:** `RuleTemplates` (corpus
  junkification) is LIVE and unrelated to the dead `RuleFeatures` despite the similar name —
  the collision itself misled the earlier audit. `pixelflow-ml/src/nnue.rs` and the MCTS policy
  samplers are already deleted; those items are closed, not open.

---

## 1. The denotations

What each load-bearing noun IS, as a mathematical object. Types listed here are obligations; the
implementation is then obliged to them.

**Expr (J1).** An expression is a rooted term graph: a pair (arena, root) where the arena is the
DAG storage and the root picks the term out of it. Neither half means anything alone.
`struct Expr<'a> { arena: &'a ExprArena, root: ExprId }` (name open; `RootedExpr` if `Expr` is
too loaded post-deletion of the old Arc `Expr`). Exactly the object threaded as two positional
params through 37 signatures today.

**Extraction (J2).** An extraction is a *witnessed selection*: (egraph, root, choices) where
choices is a well-founded (cycle-free, bottom-up realizable) function from reachable e-class to
chosen node. Constructing one **is** validating well-foundedness — an `Extraction` value cannot
exist un-backfilled; `backfill_well_founded` becomes the smart constructor's internals, and a bare
`Vec<Option<usize>>` no longer crosses a public boundary.
- **Kills P1(c):** variance is a property *of the chosen nodes*, so `Extraction` (not the e-class)
  is the only thing a variance analysis may be computed from. A class-wide meet is
  unrepresentable because the class-wide view is never handed to the variance code.
- **Kills P1(a)** (with the epoch tag): the feature vector is a pure function
  `features(extraction, θ_emb)`. A cached feature vector is valid only for the θ it was computed
  under, so `Features` carries an `EmbeddingsEpoch` — a counter the SGD step increments — and
  every consumer asserts epoch equality, panicking on mismatch (NO SILENT FAILURES). Staleness
  becomes a loud crash today and a type error once the recompute policy lands (§4b).

**CostLabel (J3) + clocks (J4).** A label is a measurement *in a clock context*:
`(expr_ref, value: SessionNs, mode: BenchMode, drift: DriftFactor, order: BenchPosition,
calibration: SentinelContext)`. Two labels are comparable only when denominated in the same
clock. Raw readings are `LocalNs` (the clock running when the sample was taken); the canonical
label value is `SessionNs` (the session-opening clock); the only conversion is
`LocalNs::normalize(drift) -> SessionNs`, and call overhead is a `SessionNs` that can only be
subtracted from a `SessionNs` — so "apply drift, then subtract overhead" stops being an ordering
convention and becomes the only well-typed composition. A measurement without sentinel context
cannot be constructed at all (today it hard-panics; the type removes the branch).

**BenchRequest (J5).** A benchmark is a function of a request:
`{ subject: Arena | Compiled, repetition: Once | Repeated(n) | Both, compile: Cached | Fresh }`
consumed by one `benchmark(req) -> BenchResult`. The 8 suffixed fns are its currying.
Denoted now; consolidated next round (§4b) because it touches the measurement path PR #984 just
hardened.

**VariantSet (J6).** The object the search actually needs ranked: a finite set of expressions
*known equivalent by construction* — either the two sides of a junkify pair or the members of one
saturated e-class. `VariantSet = { class_witness, members: Vec<Expr> }`, with the invariant that
all members denote the same function. The live training loop currently destroys this structure
(each side of `BwdTrainingPairArena` becomes an independent absolute-regression sample); Round 2's
contrastive objective consumes it. **This round mints the type and threads the pairing through to
the training-data files; the loss change is next round.** Load-bearing consequence, from the
measured program: across-expression Spearman ρ≈0.98 is saturated by size (a transcendental-count
baseline gets 0.95) — within-VariantSet ranking is the only metric that discriminates models.

**Tier fence (J7) + FenceKey (J8).** A corpus tier is a set of expressions **closed under the
fence equivalence**. The split unit is the `Family` (one `(Band, Seed)` stream — newtype, not
tuple), never the individual draw. The fence guarantee is directional: a fence built from
DEV∪FINAL keys rejects TRAIN-stream candidates; today `StructuralKeySet` carries no record of
which tiers built it, so using it backwards compiles. `Fence<Dev>` / `Fence<Final>` (phantom tier
marker) makes direction a type.
- **Kills P1(d):** leakage is precisely "an expression whose structure the model has effectively
  seen, on the certifying side." "Effectively seen" is defined by the *feature map*, not by raw
  structure: if features collapse literals, two literal-differing expressions are the same point
  to the model. Obligation: the fence equivalence must be **at least as coarse as** the feature
  equivalence — `FenceKey = structural_key ∘ feature_quotient`, i.e. the key is computed from the
  same abstraction the features see, by construction (one function, imported by both, never
  restated). The type seam lands this round; the key-function change plus corpus regeneration is
  next round (§4b) because it changes which samples are admitted.

**SchemaIdentity (J9).** See §2. Kills P1(b).

**SaturationGuide (J10).** The Guide's contractual role, from the Phase-3 design: given the
e-graph's current state and a set of candidate rule applications, produce per-candidate
move-ordering scores; trained later, purely supervised, on hindsight provenance labels from
`egraph::labeler`. That sentence is the entire public contract — one trait, one constructor:

```rust
pub trait SaturationGuide {
    fn score_candidates(&self, graph: &GraphSummary, candidates: &[RuleCandidate]) -> Vec<f32>;
}
```

(`GraphSummary` wraps today's `GraphAccumulator`; `RuleCandidate` wraps the arena-template rule
encoding.) Everything else — the accumulator's 18 mutators, `forward_graph`,
`compute_graph_embed`, `bilinear_score`, both `mask_score_all_rules_*` variants — is private
machinery behind it. The Guide's parameters leave the live checkpoint (§2) and stop receiving
weight decay. Marked ROADMAP: if Phase 3 is cancelled, this module is deleted whole.

**SaturationBudget (J11).** One saturation loop: `saturate(&mut self, budget: SaturationBudget)
-> SaturationResult` where budget = {max_iters, max_classes, timeout}. The two current
implementations (saturate.rs:91/104 free fns; graph.rs:807/818 methods) are the "one definition,
imported, not restated" violation, live; one of them becomes the definition, the other a caller.

**RoundVerdict (J12).** A verdict is a statement *conditional on gates*, and the gate order is
part of its meaning: correctness first (numeric gate failure ⇒ no timing claim of any kind),
censoring second (>10% policy failure ⇒ no directional claim), margin last. As a type:

```rust
enum RoundVerdict {
    GateFailed(GateFailure),          // numeric or censoring — no claim printable
    Underpowered { ci: Ci, gate: f64 }, // CI straddles the line: more kernels, not a re-run
    Promote { ci: Ci },
    Reject  { ci: Ci },
}
```

produced by exactly one gate pipeline; `verdict_text` becomes a Display impl. The DEV/FINAL
asymmetry becomes structural: only DEV evaluation returns a `RoundVerdict`; FINAL evaluation
returns a `PublicationReport` that *has no verdict field* — the type, not a CLI-flag branch, is
what refuses to answer "should we promote" with FINAL data. The A/A noise floor is a field of the
report, not an inline comparison.

**Naming collisions (findings probe).** Two renames, both mechanical:
- `labeler::Label` (LoadBearing/Wasted) → `HindsightLabel`. The word "label" belongs to the cost
  target (J3); the provenance object is a different noun in a different module.
- the e2e bench's `String` verdict → `RoundVerdict` (above). `quarantine::Verdict` and
  `PointVerdict` keep their names — properly namespaced, distinct objects.

**JournalEntry + ConfigHash (J15).** A claim's run provenance is a value:
`JournalEntry { config: ConfigHash, weights: ArtifactId, metrics: … }`, serde-defined, appended
by one writer function. `ConfigHash` is the SchemaIdentity-style content hash of the full run
configuration. A number without one of these attached is not a result.

---

## 2. The traits

**`SchemaIdentity`** — things-with-schema-identity: NNUE checkpoints, mint sidecars, corpus tier
files, journal entries. One derived **content hash of a layout descriptor** (field names, dims,
feature-slot meanings, magic) replaces hand-maintained version integers — the TRIE bump that
never happened (P1(b)) cannot fail to happen when the identity is *derived from* the layout.
Contract: every loader asserts `stored_identity == Self::schema_identity()` and fails loudly on
mismatch; `MintMetadata::weights_identity` (content-hash-binds sidecar to weight bytes) is the
existing instance, generalized. Deliberately **no format-evolution ceremony**: a scratch/derived
artifact that mismatches is regenerated, not migrated. Never silently consume a stale artifact;
never build migration machinery for files that are cheaper to remint.

**`CostDag`** (exists, keep as-is) — the single seam through which features are computed
(`ArenaCostDag`, `ChoicesCostDag`, unified per PR #984). All feature construction goes through it;
`Extraction` (J2) plugs in as the choices-backed instance. One walker, one input function —
now also the enforcement point for the FenceKey feature-quotient (J8).

**`SaturationGuide`** (J10) — the Phase-3 seam. The only trait minted *for* the roadmap.

**Deliberately not traits** (subtract before you add):
- Bench subject/mode: closed sets → enums inside `BenchRequest` (J5).
- Clock denomination: two newtypes (J4), not a `Clock` trait — there are exactly two clocks.
- Tier: the existing `Tier` enum plus a phantom marker on `Fence` (J7); no tier polymorphism.

---

## 3. The module map

### pixelflow-search

`src/nnue/factored.rs` (3,782 lines) dissolves:

| New module | Contents | Visibility |
|---|---|---|
| `nnue/embeddings.rs` | `OpEmbeddings`, `depth_pe`, `EmbeddingsEpoch`, one `InitStrategy` enum (`Zero \| Random(seed) \| LatencyPrior(seed)`) replacing the 4-fn init family | pub(crate) internals, pub type |
| `nnue/edge.rs` | `EdgeAccumulator` + `CostDag` instances; keeps the currently-unexercised `remove_*` API (A6) | as today |
| `nnue/head.rs` | `ExprNnue` reduced to the live extraction head: backbone + value MLP, `forward_expr_only`, `predict_log_cost_with_features`; `RandomizeScope` collapses (mask scope moves to guide) | pub |
| `nnue/serialize.rs` | live-only checkpoint format + `SchemaIdentity` impl (§4a-2) | pub load/save |
| `nnue/guide/` | **private**: `accumulator.rs` (GraphAccumulator + mutators), `scoring.rs` (mask MLP, bilinear, `encode_rule_from_arena`); `mod.rs` = `SaturationGuide` trait + one constructor, nothing else pub | trait-only surface |
| `nnue/mod.rs` (gen) | stays: `BwdGenerator`, `ExprGenerator`, `BwdTrainingPairArena` → produces `VariantSet` (J6); `RuleTemplates` (LIVE — do not confuse with deleted `RuleFeatures`) | as today |

Deleted outright (zero callers, zero roadmap): `Edge` struct, `ExprNnue::from_factored`,
`RuleFeatures` + `set_rule` + `rule_mlp_*` + `encode_rule`/`encode_all_rules`/
`encode_all_rules_from_templates` (legacy sub-tier superseded by `encode_rule_from_arena`).

`src/egraph/`: one `saturate(budget)` (J11) — `saturate.rs` keeps the definition,
`graph.rs::saturate_with_limit{,s}` become callers or die. `extract.rs` gains `Extraction` (J2)
and loses bare `Vec<Option<usize>>` from its public signatures. `labeler.rs::Label` →
`HindsightLabel`. `Expr<'a>` (J1) lives in pixelflow-ir-adjacent shared location — pragmatically:
define in `pixelflow-search` (it already sits above `ExprArena`) and re-export; do **not** modify
pixelflow-ir.

### pixelflow-pipeline

Bins currently re-own library logic; the library takes it back:

| Module | Contents | Extracted from |
|---|---|---|
| `src/bench/label.rs` | `CostLabel`, `SessionNs`, `LocalNs`, `DriftFactor` (J3/J4); `SentinelContext`, `BenchMode` move here | jit_bench.rs |
| `src/bench/request.rs` | `BenchRequest` (J5) — type this round, consolidation next | jit_bench.rs |
| `src/eval/verdict.rs` | `RoundVerdict`, `PublicationReport`, noise-floor field, gate pipeline (J12) | bench_extraction_3way.rs (4,262 lines → thin orchestration bin) |
| `src/eval/gates.rs` | censoring (`PolicyFailureRates` moves in), geomean-with-precondition | bench_extraction_3way.rs |
| `src/oracle_compare.rs` | the ONE `run_scalar` (J14) — quarantine.rs, bench_extraction_3way.rs, and tests/prod_kernel_jit.rs all import it | 3 verbatim copies |
| `src/training/split.rs` | `Family` newtype, `Band` (moves in from gen_bench_corpus.rs — it is a cross-bin noun), `Fence<T>`, `FenceKey` seam (J7/J8) | split.rs + gen_bench_corpus.rs |
| `src/journal.rs` | `JournalEntry`, `ConfigHash` (J15) | nothing (new; currently prose-only) |

The same-form gate stays in `training/quarantine.rs`; the cross-form conditioned gate stays in
`eval/` — per A5 they share a doc cross-reference, not a type.

---

## 4. Ranked execution list

### (a) THIS ROUND — encapsulation, deletion, mechanical moves

Ordered; each item is independently landable.

1. **Guide encapsulation (J10) + legacy deletion.** Create `nnue/guide/` with the
   `SaturationGuide` trait + constructor as the entire public surface; move GraphAccumulator and
   the mask/graph/rule-proj fields/methods behind it, private. Delete the legacy sub-tier
   (`RuleFeatures`, `rule_mlp_*`, `encode_rule`, `encode_all_rules`,
   `encode_all_rules_from_templates`) and the zero-caller vestiges (`Edge`, `from_factored`).
   `apply_unified_sgd` and `UnifiedGradients` drop every `d_mask_*`/`d_graph_*` field — this
   **fixes the confirmed decay defect** (Phase-3 weights silently decaying toward zero with no
   opposing gradient). Keep the characterization tests, relocated with the module. This is a
   contract boundary, NOT a code move into a side directory.
2. **Live-only serialization + `SchemaIdentity` (J9) — kills P1(b).** The checkpoint carries only
   the 16,897 live params (today's format is a 49%/51% live/noise split by param count). New
   magic; identity = derived content hash of the layout descriptor; every loader asserts it.
   The Guide gets its **own** format with an explicit trained/untrained marker when Phase 3
   first consumes it — "Guide not yet trained" becomes *section absent*, not float noise.
   `MintMetadata.weights_identity` generalizes onto the trait.
3. **`Extraction` (J2) — kills P1(c), arms the P1(a) kill.** Smart constructor absorbs
   `backfill_well_founded`; variance analysis takes `&Extraction` only; `Features` gains the
   `EmbeddingsEpoch` tag with a loud assert on mismatch (fail-fast half of P1(a); the recompute
   policy is 4b-5).
4. **`Expr<'a>` adoption (J1).** Introduce the type; convert the 37 `(arena, root)` sites.
   Purely mechanical, both crates.
5. **`run_scalar` dedup (J14).** One shared fn in `oracle_compare.rs`; all ≥3 copies import it.
6. **Saturation unification (J11).** `SaturationBudget` + one loop; delete the duplicate.
7. **Verdict types + renames (J12).** `RoundVerdict` enum with gate-ordered construction;
   `PublicationReport` without a verdict field for `--final-eval`; `labeler::Label` →
   `HindsightLabel`. Behavior-preserving: same thresholds, same precedence, now as variants.
8. **Fence/Family types (J7).** `Family` newtype, `Band` to the library, `Fence<Dev>`/`Fence<Final>`
   phantom tags, fence constructor parameterized on the key function (the J8 *seam* — the key
   function itself is unchanged this round).
9. **`VariantSet` minting (J6).** Type + thread the junkify pairing through to training-data
   files so the equivalence structure survives to disk. Loss function unchanged.
10. **`JournalEntry`/`ConfigHash` (J15).** Small serde structs + one writer fn; wire the existing
    journal-writing call sites through them.

### (b) NEXT ROUND — behavior-changing or measurement-gated

- **Contrastive/ranking objective over `VariantSet`** — the Round-2 experiment proper. Needs DEV
  measurement, the ±5% CI gate, and the A/A floor; it is research, not reorganization.
- **`FenceKey` feature-quotient change + corpus regeneration** — the behavioral half of P1(d).
  Changes which samples are admitted to which tier; requires reminting tiers and re-baselining.
- **`BenchRequest` consolidation** of jit_bench's 8 suffixed fns — touches the measurement path
  PR #984 just hardened; land under A/A noise-floor verification that labels are bit-stable.
- **bench_extraction_3way.rs harness-library split** — the large move (4,262 lines) lands after
  the `eval/` types (4a-7) exist to receive it, so it is a move, not a rewrite.
- **Features recompute/cache-invalidation policy** — the full P1(a) fix beyond the epoch assert
  (recompute-per-batch vs epoch-keyed caches has a training-throughput cost to measure).

### What Execute must NOT attempt

- **No Guide internals work**: no `GraphEdit` enum (A2), no `Layer` abstraction (A3), no training
  code for the mask head. Implementation ahead of its consumer is how this surface got here.
- **No O(Δ) incremental extraction** (A6); do not delete `EdgeAccumulator::remove_*` either —
  intended API, adoption-plan step 5, measured ceiling ~2×.
- **No `Rewrite::apply` signature change** (A1) — 37 impls, already unified.
- **No pixelflow-ir modifications** (read-only context; `DifferentialCheck`/`PointCheck` stay put).
- **No changes to measured-label semantics** (drift math, overhead subtraction, sentinel cadence)
  outside item 4b-3's gated consolidation — the newtypes in 4a must be representation-only.
- **No renaming `quarantine::Verdict` or `PointVerdict`** — the collision is resolved by the other
  two renames.
- **No `GateKind` unification** (A5) and no speculative `EClassRef` sweep (J13 is approved but
  opportunistic — convert call sites only when a touched file already changes).
- **No deletion of Guide characterization tests** — they are the regression floor for whatever
  Phase 3 consumes.

---

*Retirement notes for roadmap-admitted seams: J6/J8 fall out if Round-2 contrastive training is
abandoned; J10 (and its tests) is deleted whole if Phase 3 is cancelled; J15 follows the 2026-08-05
workflow's lifetime. Every other row is demanded by the live path and carries no such dependency.*
