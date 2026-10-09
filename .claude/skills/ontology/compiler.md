# The compiler and search

### P (the pipeline)

- **Is:** one function, `P : Kernel × LatticeShape × Isa → (Bytes, Link)`.
  Since O1, `P(k, s, t) = emit_t ∘ legalize_t ∘ L_s(k)`, where `L_s`
  optimizes each unit by itself and links it back. It is the only place
  anything is optimized. "The JIT is P run at runtime; the build-time
  compile is P run at build time." The law is
  `bytes(P_build) = bytes(P_run)`.
- **Is not:** two tiers. Not dependent on anything but `(k, s, t)`. Today it
  also depends on `isa::detect()` inside the emitter, the mmap inside the
  emitter, double saturation, the global `KernelStore`,
  `saturation-switch`, silent fallbacks, and arena-order canonicalization
  (the last fixed by a post-order walk). Superseded: P starting at
  `expand_refs` (O1 replaced it with the unit walk); the closing phase
  (deleted 2026-09-29).
- **Follows:**
  - The ISA tier is an explicit input (`EmitCtx { isa }`), and the caller
    maps the bytes.
  - Gates compare two computations rather than committed digests.
  - `pipeline::compile` stays outside the law, by name.
- **Lives:** `pixelflow-codegen/src/{pipeline,jit_cache}.rs`,
  `pixelflow-search/src/runtime.rs`. Decided in one-pipeline §1.1, amended
  by the-language-is-kernel O1. Today `detect()` is called inside
  `emit/mod.rs` (`compile_native`), `EmitCtx` carries only `max_regs`,
  `jit_cache::compile` emits a declined term as given, and the
  `program`/`preload` entry points do not exist.

### Stage (open value, closed program)

- **Is:** a position in P: compose (open: no optimizer; `Dwrt` and folds
  kept) or compile (closed: "the only place anything is optimized").
  "Compose, then compile: a stage, not a tier."
- **Is not:** a tier. "Under the pipeline splits, a stage (compose versus
  compile) was mistaken for a tier (macro versus runtime). Under the loop
  splits, when a loop is built was mistaken for what it is."
- **Follows:** with one optimizer that runs only at compile, there is no
  tier. So `Tier`, its stderr prefix, the telemetry `"tier"` field,
  `DwrtFree` and `macro_tier()` have no job (M1–M3). The chain rule survives
  composition, and `derivative_under_warp.rs` tests the stage.
- **Lives:** one-pipeline §1.2, §2; the-language-is-kernel D2. Today
  `pixelflow-search/src/tier.rs` (`Tier::{Runtime, Macro}`) and
  `pixelflow-compiler/src/lib.rs` (`macro_tier`, `DwrtFree`) still hold the
  tier split.

### Tier. Homonym

- **Is:** several unrelated senses:
  1. The **ISA tier**: AVX2, AVX-512 or NEON, chosen at startup. This is the
     only tier in the denotation.
  2. The **saturation budget tier**: blitz, rapid or classical, by input
     size.
  3. The **corpus tier**: TRAIN, DEV, FINAL.
  4. A variance level ("row-tier work"; mask_support's "symbolic tier").
  5. Retired: the macro and runtime tiers, the combinator/"legacy" tier,
     the interpreter tier, and the frame/row/body hoist tiers.
- **Is not:** one concept. Sense 5 is not a tier at all; it was a stage.
- **Follows:** code that reads "tier" must say which.
- **Lives:** `pixelflow_codegen::isa::Isa` (`pixelflow-codegen/src/isa/mod.rs`);
  `pixelflow_search::tier::Tier` (`pixelflow-search/src/tier.rs`, to delete,
  M3); `SaturationConfig` (`pixelflow-search/src/egraph/saturate.rs`).

### Program, template, instance

- **Is:** a **program** is what P produces, bytes plus links:
  `⟦P(k, s, t)⟧ : f32^m → (L_s → f32)`. Lowering produces a **template**,
  "a replay of `ExprArena` pushes", and instantiating it with structural
  values gives a program. An **instance** is one instantiation, with
  identity by instance. A font at a zoom level is one program per piece
  count: exact N, unbucketed (JP, 2026-10-09; see terminal.md, "Font
  programs").
- **Is not:** per glyph (95), per bucket (6; JP: unbucketed), or one program
  holding every glyph under an `if id < k` tree (superseded 2026-10-09). Not
  unrolling: "Composing instances is the host walking data."
- **Follows:** structural parameters and the shape key the program, and
  uniforms bind through its block. The host chooses a program by its
  structural parameters, for the font the glyph's piece count, and then
  writes the glyph's block into it. Sharing one `id` instance across a tree
  walk was a convention of the host walk, not a type (O3); superseded
  2026-10-09, since no program holds more than one glyph (terminal.md, "Font
  programs"). Before extracting in a template, give template inputs a
  meaning of their own (B6).
- **Lives:** the-language-is-kernel §1.1, §1.4, §1.7–§1.8, D9. Templates are
  `Staged` and the `Param` holes of `pixelflow-compiler/src/{emit,lower}.rs`.

### Pass, legalize, Optimize

- **Is:** a pass is an endomorphism on (arena, root). **Legalize** runs
  after extraction and link, in an order forced by meaning:
  `expand_refs → lower_dwrt → collapse(extent) → pack(L) → expand_gather →
  expand_transcendentals`. You cannot differentiate a name; you must
  differentiate before collapse substitutes X; d sin = cos; and addresses
  are built over binders. Each pass is idempotent. `expand_refs` is the
  unit walk with every optimization the identity. **Optimize** is a
  denotation-preserving endomorphism that answers `Changed`, `Unchanged` or
  `Declined`. "Not optimizing is a value" (`Identity`).
- **Is not:**
  - An order hand-copied at call sites ("An order that has to be retyped is
    an order that can be forgotten").
  - Target-aware by `cfg`.
  - An unroller.
  - Bypassable (`CompileWorkspace` ran no passes).
  - Followed by a second optimizer: "A constant-fold-and-CSE stage after
    `legalize`" was refused.
  - Run first: legalize-first (`pipeline![LowerDwrt, ExpandReduce,
    Saturate]`) "has it the other way round".
- **Follows:** legalize is "a fallback in fact as well as in name". What the
  graph resolved stays resolved, and what it declined, the legalizer lowers.
  An op that reaches a backend without an encoding is a compiler bug.
- **Lives:** `pixelflow_ir::passes::legalize` (`pixelflow-ir/src/passes.rs`),
  `pixelflow_ir::optimize::{Optimize, Then, Identity, Rewritten}` and
  `pipeline!` (`pixelflow-ir/src/optimize.rs`). Decided in
  a-kept-structure-is-control-flow §4, a-fold-is-a-node, and
  collapse-is-a-fold §2.3. Today `optimize.rs`'s module doc ("The runtime
  tier is `lower Dwrt`, then `unroll Reduce`, then `saturate`") is stale.

### Legality

- **Is:** whether a backend can emit a node. Three categories exhaust
  `OpKind`: Legal, Expand (lowered first) and Structural (never scheduled).
  "Legality attaches to the resolved form, not the opcode": `Shl` with a
  `Const` right-hand side is `ShiftImm` and legal, and with any other
  right-hand side it is illegal everywhere.
- **Is not:** a test-only constant, or four restated predicates.
- **Follows:** one table that the passes read. A phase-typed arena
  (`Arena<Surface> → Arena<NoDwrt> → Arena<Legal>`) would make "a Dwrt
  reached codegen" unrepresentable. It was proposed ("Honest cost: it is
  added machinery") and is unbuilt, so the surviving-Dwrt check is still a
  runtime refusal.
- **Lives:** `pixelflow-codegen/src/emit/coverage.rs` (still
  `#[cfg(test)] pub(crate) mod coverage`, "test-only infrastructure"); the
  2026-08-02 ir-layering plan, Phase 6. `OpKind::category()` is unbuilt.

### Decline

- **Is:** an optimizer's answer "this term is outside what I model; compile
  it unoptimized". It is distinct from "nothing to do", and always available,
  because optimization is never required for correctness. "A decline
  narrows": a unit the e-graph cannot hold is linked as written while the
  rest optimize.
- **Is not:** a pass-through that lets later steps run on an unlowered term.
  Not a permanent home for a construct. Not silent: the target is "a
  reported decline, pinned at zero for production kernels" (M12).
- **Follows:** each decline marks a vocabulary gap to close (Buffer became an
  opaque leaf, Reduce a typed node, Param `ENode::Param`). A node that every
  consumer declines is at the wrong layer (`Acc`).
- **Lives:** `Rewritten::Declined` (`pixelflow-ir/src/optimize.rs`),
  `Declined` (`pixelflow-search/src/egraph/insert.rs`), `record_decline`
  (`pixelflow-search/src/telemetry.rs`); one-pipeline M12. Today
  `jit_cache` emits a declined term silently, with telemetry only.

### Rule set and vocabulary

- **Is:** "One rule set, one vocabulary. R is today's `RuleSet::runtime()`,
  which becomes the only set. Its phases are subsets of it, not other sets."
- **Is not:** a production/runtime split. Not a `Vocabulary`
  (`Templates`/`Runtime`): "Naming the vocabulary makes the choice visible"
  named an accident of the tier split.
- **Follows:** a lint fails on `RuleSet::new` or `Optimizer` construction
  outside research modules (CL6). `Gather` and the identity-bearing leaves
  stay representable but opaque.
- **Lives:** one-pipeline §1.1, M4, M5. Today `RuleSet::production()`/
  `runtime()` (`pixelflow-search/src/egraph/rules.rs`) and `enum Vocabulary`
  (`pixelflow-search/src/egraph/ops.rs`) still exist.

### E-graph (e-node, e-class, IR trait)

- **Is:** an e-node's children are classes, not nodes: "That indirection is
  the e-graph." "An e-class is a semantic equivalence class and `add` is a
  homomorphism from the term algebra onto the e-graph quotient." The graph
  is an audit log. "An IR is a term language that can be destructured into,
  and rebuilt from, the signature the e-graph speaks" (`Ir`: project and
  embed over `Shape<R>`).
- **Is not:** mergeable with `ExprNode` through a generic `Node<I>`. Not a
  place for metadata kept as child classes. `EClassId` is not stable under
  union.
- **Follows:** hash-consing is identity. Leaves that carry identity (Buffer,
  Uniform, Param, a Ref unit) are hash-consed by it, and no rule matches
  them. There is one reachable-only `insert<I: Ir>` that declines rather
  than panics. `add` is total: the over-limit `EClassId(0)` sentinel
  asserted false unions and is fixed.
- **Lives:** `pixelflow-search/src/egraph/{node,graph,insert}.rs`;
  `pixelflow-ir/src/term.rs` (`trait Ir`, `Shape`); the 2026-09-04
  ir-as-a-trait plan; a-fold-is-a-node §6. `EClassId(pub(crate) u32)` is
  narrower than the 64-bit rule.

### Hash-consing

- **Is:** two different things. One is a memo private to one `ExprArena`:
  every push interns, and copies collapse as they arrive. The other is a
  shared store that every kernel indexes, so composition interns instead of
  splicing; that one is deferred. In the e-graph, hash-consing at insertion
  "is the product".
- **Is not:** measured against a bare push. Not a source of fresh ids
  (`push_unique` is the escape hatch, and nothing uses it).
- **Follows:** the CSE-only arm alone gives −41% bytes on glyphs, −70% on
  the cell grid and −77% on psychedelic, and chrome does not compile without
  it. The rewrite rules add only about −4% on glyphs. `NodeData::Const` is
  keyed on bits.
- **Lives:** `pixelflow-ir/src/dag.rs` (`Builder::intern`,
  `Builder::push_unique`, used only by `internal_test_support.rs`),
  `ExprArena::intern` (`pixelflow-ir/src/arena.rs`), the e-graph memo;
  exprarena-on-dag §5 and Stage C (2026-09-20);
  `2026-09-07-benchmark-correction.md` §A.

### Saturation

- **Is:** growing an e-graph by firing rules under a deterministic budget.
  "Budget-only by design. Saturation spends a budget and stops, full stop."
  It is monotone: "Saturation only ever adds equalities", so the larger
  budget's graph holds a superset of the terms. All its phases are
  shape-free, ISA-free, and cached by structure. The proposed phases are
  subsets of R run in sequence: `saturate_R = folds_R ∘ main_(R∖folds)`
  (M15). The fold phase admits whole folds, smallest predicted growth first,
  under `HARD_CLASS_LIMIT`.
- **Is not:**
  - A fixpoint search ("quiescence … is a diagnostic condition, never a
    certified closure").
  - Host-speed dependent.
  - Where meaning changes.
  - A source of correctness (see the integral).
  - Run twice (macro saturation followed by JIT saturation is an impurity of
    P).
- **Follows:**
  - Every value claim is anytime.
  - "More saturation is better": "a superset of forms contains everything
    the subset did", so a worse result from a richer graph is the
    extractor's defect (E5).
  - Truncating early can never make the graph unsound.
  - The currency is the **rule application**: one recorded rewrite,
    idempotent re-fires included. `EGraph::application_count()` is
    unconditional, "the budget must not depend on whether anyone is
    watching". It is not `Provenance::recorded_count`.
  - Zero rounds is silent today, so C1's gate requires that the largest
    saturation run at least one round.
- **Lives:** `pixelflow-search/src/egraph/{saturate,graph,optimizer}.rs`
  (`saturate_bounded` is in `graph.rs` and `optimizer.rs`),
  `pixelflow-search/src/saturate_pass.rs`; `HARD_CLASS_LIMIT` (`graph.rs`);
  the 2026-08-31 guide-design-revision §0 and §4.2; one-pipeline §1.1, §1.4,
  M15.

### Budget (and class cap)

- **Is:** what stops saturation: rule applications, e-classes and iterations,
  "all three deterministic functions of the input". It is tiered
  blitz/rapid/classical by size: 20,000 and 80,000 applications; classical
  200,000, rising to 2,000,000 past 625 inserted classes. The classical
  class cap is `clamp(8 × inserted, floor, ceiling)`. `HARD_CLASS_LIMIT`
  guards memory, not meaning.
- **Is not:**
  - Wall clock. `safety_ceiling` (30 s/120 s, classical 300–3,000 s) panics
    rather than truncating.
  - Env-tunable: "an env-tunable budget would put the nondeterminism back".
  - A measure of what a term denotes. Inserted classes describe "how
    compactly the term is written". A fold's size is `len × body`.
  - A cap below the input: "Below that the cap is not a budget, it is a
    truncation of the input's own rewrite frontier."
  - Superseded: wall-clock budgets of 10/50/200 ms; `hard_timeout` reporting
    `saturated: true`; the flat 5,000-class cap.
- **Follows:** the same kernel is produced on every host. A ceiling hit is
  investigated, not raised. `PIXELFLOW_SATURATION_CEILING_MS` can change
  whether the build panics, never which kernel is emitted. "If a tier is too
  slow for a user, the lever is `max_applications` … never the ceiling."
- **Lives:** `SaturationConfig { max_iterations, max_classes,
  max_applications, safety_ceiling }` and `CLASSICAL_CLASSES_PER_INSERTED_CLASS`,
  `CLASSICAL_CLASS_FLOOR`, `CLASSICAL_CLASS_CEILING`
  (`pixelflow-search/src/egraph/saturate.rs`); `Budget`
  (`pixelflow-search/src/egraph/optimizer.rs`) — not the allocator's
  unrelated `type Budget = [usize; 2]`, the carry budget per register class
  in `pixelflow-codegen/src/emit/regalloc.rs`;
  `2026-09-01-production-budget-determinism.md` (revised 2026-09-08,
  unpinned 2026-09-29); CLAUDE.md "A kernel built differently on two
  machines?".

### Extraction

- **Is:** choosing one node per reachable class at a shape. It is where
  equivalent forms are decided. JP, 2026-09-12: "Decisions between
  equivalent forms will be made at extraction." That includes a fold
  against its unrolling. "Extraction reads `s`. It prices every fold by its
  trips."
  - The result is "a witnessed selection: (egraph, root, choices) where
    choices is a well-founded … function from reachable e-class to chosen
    node".
  - There are two DP arms, tree (a child priced at every use) and shared
    (each class priced once), plus an arbiter.
  - Two costs are always reported and never confused. `dag_cost` is the
    latency prior summed once per reachable node. `objective` weights each
    node by `S.evals(variance)` and is what the extractor minimizes.
  - A **witness** is a term the extractor provably holds and walked past.
- **Is not:**
  - A stage followed by repair or reranking.
  - Monotone in graph richness: chrome's dag_cost was +42% at a 100k cap.
  - Argmin: "The witnesses say the failure is **argmin**, not cost."
  - A bare `Vec<Option<usize>>`.
  - Where guard against blend is decided (#1313).
  - Merely a form once schedules are choices: then it is "a form plus a
    schedule".
  - Superseded: the NNUE extraction head (deleted after tying the table);
    pricing macro extraction at a POINT shape.
- **Follows:**
  - Settling is a fixpoint (Knuth's AND-OR Dijkstra), acyclic by
    construction.
  - Saturate once per structure, extract once per shape.
  - 7 of 56 witnesses are one swap away and none is reachable by a sequence
    of swaps, so a swap-neighbourhood search provably misses the answer.
  - A learned component "emits a decision, not a number" and sits inside
    the DP as a residual (extraction-judge, unbuilt).
- **Lives:** `pixelflow-search/src/egraph/extract.rs` (`Extraction`,
  `tree_dp_pass`, `shared_dag_dp_pass`, `settle_in_cost_order`),
  `pixelflow-search/src/egraph/witness.rs`. Decided in 2026-08-17
  cost-model-domain J2, `2026-09-08-extraction-witnesses.md`, and
  one-pipeline §1.1, §1.4. Today `extract.rs` calls the tree arm "the
  control arm of the objective A/B", but it is what finds the unrolled term
  (CL9). CLAUDE.md's `env_extraction_policy()` no longer exists.

### Tie-break

- **Is:** the typed policy the DP applies to an exact tie. In production it
  is `Insertion`: "the winner is whichever node the class happened to hold
  first."
- **Is not:** an accident ("a `TieBreak` impl is a type"). Not a correction
  to latency: compile cost and code size are "a second cost axis, not a
  correction to the first".
- **Follows:** a fold and its unrolling tie under the tree objective. The DAG
  objective, which decides, adds each class's cost once with no trip count,
  so a kept fold is under-priced about 12×. Today "Emitting 34,993
  straight-line instructions versus a loop currently turns on e-node
  insertion order" (E6). Superseded: "the loop is cheaper to compile and
  smaller in I-cache while costing the same to run" (2026-09-11). Measured
  later, the unrolled form runs 2–10× faster per glyph (one-pipeline §1.4).
- **Lives:** `TieBreak` (a `pub(crate)` trait), `Insertion`, `Canonical`
  (`pixelflow-search/src/egraph/extract.rs`). There is no `Content` type; the
  content-ordered research arm that a-surviving-reduce-is-a-loop "What
  decides 2c" calls `Content` is `Canonical` in code. one-pipeline §1.4.

### Cost model (latency prior, fold price)

- **Is:** the static per-op cycle table, `CostModel::latency_prior()`, "the
  only policy". Its denotation is `cost(E) = analytic(E) + residual(E)`,
  with `analytic = Σ cycles(op)·trips(level(op))`, where the level is a
  property of the *chosen* nodes. A fold is priced by one formula,
  `fold_cost = len·body + (len−1)·combine`, shared by the guard analysis
  and the extractor (D2). The proposed price (M14) has the arbiter price
  each settled term "by what codegen executes, with one rule for kernel and
  lattice folds alike".
- **Is not:**
  - A learned total-cost predictor: that shape tied the table and was
    deleted.
  - Additive once schedules are choices.
  - A price for code (measured unsound).
  - Critical-path aware (E2).
  - Zero for a fold (`Reduce => 0` in the guard's arm cost was a bug).
  - Superseded: "the cost model does 95%" and a reranker as the next step
    (ledger L088, 2026-09-07). The denotation stands.
- **Follows:**
  - Mask coherence (how often a mask is uniform across a batch) is the first
    profile-dependent residual term. It is a property of the data: about 97%
    uniform for a sphere silhouette, almost none for glyph coverage.
  - Factoring Σ stays cost-neutral until a binder's trip count is priced.
  - "When a metric has been optimized against and lost, the missing term is
    likely to be something that metric's own definition ruled out of
    scope."
- **Lives:** `pixelflow-search/src/egraph/cost.rs` (`CostModel`,
  `CostModel::fold_cost`, called by `pixelflow-codegen/src/program/guards.rs`);
  `2026-09-01-schedule-cost-model-denotation.md` §2, §9; one-pipeline M14.
  Today `cost.rs`'s "the DP multiplies" is true only of the tree DP.

### Reranker

- **Is:** a seam kept in code with no implementation. A learned term enters
  "as a term in the cost function, not as a stage after extraction". JP:
  "The whole reranker concept is bringing traditional passes where they
  don't belong."
- **Is not:** where a choice between equal forms is made: "The choice
  between two equal forms is the cost function's, evaluated inside the DP,
  once."
- **Follows:** the seam's justification "should be revisited rather than
  quietly relied on".
- **Lives:** `Reranker` (`pixelflow-search/src/egraph/extract.rs`);
  emit-should-just-emit §4 (JP, 2026-09-12). Today CLAUDE.md "Cost Model and
  the Guide" still presents the seam as the place for the schedule cost
  model.

### Variance

- **Is:** the set of binders (coordinates and fold binders) a value depends
  on, as a `u64` bitset. Law: if `b ∉ var(v)`, then `⟦v⟧` is constant along
  `b`. Children union; "a fold removes its own index". It is the Boolean
  shadow of the forward derivative. For an e-class, `var(C) = ⋂ var(n)`, a
  greatest fixpoint from ALL. The placement rule: "the shallowest scope that
  binds every variable … That one rule is loop-invariant code motion,
  hoisting out of a reduction, and constant folding."
- **Is not:** constant-ness. `CONST` means lattice-invariant; a uniform and a
  context pointer are both `CONST`. Not a scheduling property of a whole
  e-class: "the class-wide meet lies once a rewrite merges a constant into a
  pixel-varying class". Not a `u8`. Not reachability. Not demand, which is
  its dual. Superseded: `DepsAnalysis`/`Variance::meet` (where
  `X.meet(Y) = X`).
- **Follows:**
  - Variance decides hoisting, factoring side conditions, the
    lane-uniform/gather split, and `evals` pricing.
  - A stale class fact over-approximates, so "a rule can miss an opportunity
    but can never fire wrongly".
  - A unit's leaf carries its variance.
  - Retired bits 2–3 must be refused.
- **Lives:** `pixelflow_ir::Variance(u64)` (`pixelflow-ir/src/variance.rs`),
  `EGraph::variance` (`pub(crate)`, `pixelflow-search/src/egraph/graph.rs`),
  `Extraction::chosen_variance` (`extract.rs`); one-name-bound-later §2;
  an-integral-is-a-fold §1 and §3 (that part survived the retraction). Today
  `chosen_variance` returns `[f32; SCALAR_FEATURE_COUNT]`, an NNUE
  variance histogram, not a `Variance` per chosen node.

### Demand

- **Is:** `dem(v) ⊆ Ω`, "the points at which v is read". Law: if
  `ω ∉ dem(v)`, replacing `⟦v⟧(ω)` leaves the output at ω unchanged. It is
  computed backward from the root: an `If` passes `dem ∧ m` and `dem ∧ ¬m`
  to its arms, and consumers join with ∨. It is control dependence, the
  Boolean shadow of the adjoint, "a property of the graph — not of any
  select". As a predicate it is DNF over mask literals, capped and widened
  to `true`.
- **Is not:**
  - An e-graph fact: it is exact only on the extracted DAG.
  - Exclusivity: a value read by both selects' true arms has demand `m` and
    is exclusive to neither.
  - A selector: "demand only decides what is computed, never what is
    selected".
  - An ordering: "predicate strength is not an ordering". The superset
    invariant holds for unconditional edges only.
  - Semantics, since arms are total.
- **Follows:** a guard covers a region of equal demand, so `m₁ ∨ ¬m₂` would
  become guardable. A static demand fraction belongs in the cost,
  `cost·P(demanded)` (D6), and guards in the graph need it (D7). A Y-range
  demand belongs in a loop bound, not a per-row guard. Placement that reads
  demand is the next speed step (D1).
- **Lives:** nowhere resident. `pixelflow-ir/src/passes/demand.rs` was
  deleted 2026-10-03 as "a diagnostic that nothing reads" (ce3e9177, #1307;
  last present at `0459d2b3`). `pixelflow-codegen/src/program/ownership.rs`
  computes the single-`If` case, and regions form a tree, so `m₁ ∨ ¬m₂`
  still falls in the enclosing region. Decided in
  `2026-09-07-demand-is-a-dag-property.md` (JP: "Why isn't the DAG over the
  whole program?") and one-conditional §4, §6.

### Widening and narrowing

- **Is:** the direction every soundness argument about predicates and ranges
  takes. Replacing one with a weaker or larger one is always sound.
  Replacing one with a stronger or smaller one never is. "Every unsoundness
  available to this design is a narrowing, and a narrowing deletes pixels
  without failing."
- **Is not:** a precision dial that turns both ways.
- **Follows:** clause caps widen to `true`. A buffer-dependent conjunct
  contributes the full extent. An op with no bound yields the full interval.
  A derived range is computed over the values a uniform can take. The gate
  is containment, not equality. Precision is a compile-budget knob, and
  correctness never depends on one: "No tuned constant stands in for a
  bound." Three constants have had stated derivations that did not survive
  measurement: `EXTENT_SLOP`, `CLUSTER_ROUNDS_PER_SELECT` and
  `DISC_BAND_ULPS`.
- **Lives:** `pixelflow-ir/src/mask_support.rs` (its contract section);
  demand §1, §7; one-conditional §2, §3, §9.

### Compile cache key (canonical)

- **Is:** a kernel's canonical bytes, a post-order walk from the root that
  hash-conses structurally equal subterms, plus its shape. Buffer and
  uniform identities enter as dense slots by first occurrence and come back
  in a link table. "An over-specific key misses sharing and never shares
  wrongly."
- **Is not:** absent (2025's "No cache" is retracted). Not a function of
  construction history (push order was fixed by the post-order walk). Not
  inclusive of uniform values or defaults. Not an ABI ("The cache is a memo,
  not an ABI").
- **Follows:** a thousand circles is one compile. An extent change
  recompiles "by decision". The saturated graph is cached per structure. The
  cache never evicts; caching was put "later". A unit's key digests minted
  identities, so a program of units needs a structural key before E1.
- **Lives:** `pixelflow-codegen/src/jit_cache.rs` (`cache_key`,
  `UNIT_PROGRAM_TAG`), `pixelflow-ir/src/key.rs` (`canonical`);
  uniform-slot-identity "Link order"; composition-is-linking §2.1;
  one-pipeline §1.1.

### Numeric contract (precision versus range)

- **Is:** "the language gives you the instruction." Precision is tunable;
  "Precision is on the table; range is not." "The contract line is algebraic
  validity, not IEEE value-identity: pixelflow is a math library over the
  reals." Where a function cannot be computed over the whole input type, it
  gets a documented domain and returns NaN outside it.
- **Is not:** edge-case IEEE conformance. Not a clamp ("a clamped value is a
  wrong answer wearing a right answer's clothes"). Not a licence to
  hand-roll something worse (the retired `Round`). Not a licence to fold a
  target-divergent row on the build host: ConstantFold declines
  `fold_is_platform_specific(args)`, which is value-aware (`min(1, 2)`
  folds, `min(-0.0, 0.0)` does not).
- **Follows:** range is asserted with no tolerance. Within one target the
  optimizer may still change an answer where none was promised (Min/Max
  commutativity). Interval transfer functions round outward and are
  target-aware.
- **Lives:** CLAUDE.md "Floating point at the edges" and "Precision is on the
  table; range is not"; `OpKind::fold_is_platform_specific`
  (`pixelflow-ir/src/kind.rs`);
  `pixelflow-codegen/tests/{transcendental_jit,trig_range_jit}.rs`.

### External oracle and the gate

- **Is:** "Green CI is permission to submit … the gate itself." "Every gate
  compares two computations at the same commit." An external oracle is an
  independent implementation sharing no code or constants with ours
  (FreeType, the `f64` winding oracle), and it is the only instrument that
  sees a shared-definition bug.
- **Is not:** one signal among others, or a place for prose caveats ("A gap
  in CI is a check to write, not a caveat to attach"). Not a same-form check
  ("a regression corpus is a change-detector, not an oracle"). Not a
  committed digest ("a gate that cries wolf gets deleted"). Superseded:
  JIT-vs-interpreter goldens, and 08-05's "same-form hard gate … closes the
  reward-hacking channel". The interpreter and every same-form suite built
  on it are deleted.
- **Follows:** run the oracle before writing a conclusion. When a bug ships
  green, the retrospective is about the gate. Goldens are quantized to the
  renderer's 8-bit form with tolerance, because bit-exact f32 differs across
  ISA tiers. A gate blind to code size needs its own check
  (`glyph_optimization_cost.rs`).
- **Lives:** CLAUDE.md "CI is the gate"; `.github/workflows`,
  `scripts/check-*.sh`; `pixelflow-graphics/tests/freetype_oracle.rs`,
  `pixelflow-graphics/tests/glyph_optimization_cost.rs`; one-pipeline §5.

### Guide (candidate, growth)

- **Is:** "An ordering policy over the rule applications available this
  round. A guide may only order and truncate. It mints no nodes and performs
  no unions, which is why any guide is sound by construction … and a guide
  PR argues quality, never correctness." It is trained offline, supervised,
  on hindsight labels. JP: "is this application going to grow the graph and
  give me nothing, or grow it a little and give me something useful?"
  - A **candidate** is `(R, n, b)`: a rule, a matched node and its bindings,
    observed at round start and keyed by `(rule, canonical class content)`.
  - **Growth** is the number of e-nodes an application would add. It is
    "computable before firing, cheaply and exactly, which is what makes it a
    feature rather than a prediction".
- **Is not:** an RL policy, critic or REINFORCE head (removed July 2026 as
  methodologically unsound). Not a second saturation loop. Not "finding the
  optimum faster". Not a whole-graph evaluator. Not trained at compile time.
- **Follows:** 90.4% of candidates commit nothing, so dedup is the urgent
  cost. An inlining rule is inadmissible until growth gates it. An unseen
  feature cell takes the unguided order. The model is one versioned
  artifact, keyed by the rule-set fingerprint.
- **Lives:** `pixelflow_search::nnue::guide::SaturationGuide` (a trait,
  `pixelflow-search/src/nnue/guide/mod.rs`),
  `pixelflow-search/src/egraph/{guided,growth,candidate}.rs`,
  `EGraph::predicted_growth` (`pub(crate)`, `graph.rs`). Decided in the
  2026-07-07 guided-saturation redesign, 2026-08-31 guide-design-revision,
  and the 2026-09-02 optimizer-api §1.L4.

### Provenance and the hindsight label. Homonym of codegen's Label

- **Is:** the e-graph as an audit log: node origins plus a union journal.
  After an episode, every fired application gets an observed label,
  load-bearing or wasted, at three bounds: labeler (`derivation_ancestors`,
  over-approximating in the safe direction), tight, and strict.
- **Is not:** credit. "Credit is counterfactual":
  `Δ_a = R(τ \ a, B) − R(τ, B)`, leave-one-out with a confluence-aware mask.
  Not exact, and not codegen's `Label`. The 2026-08-17 J3 ruling gave the
  word "label" to the cost target and asked `labeler::Label` to become
  `HindsightLabel`; that has not landed. Not present without the
  `provenance-journal` feature ("they don't exist as types").
- **Follows:** the strict bit remains the best predictor (round 4). "The
  library ships the records, not the interpretations." Production builds
  must not journal.
- **Lives:** `pixelflow-search/src/egraph/{provenance,labeler}.rs`
  (`labeler::Label` still carries the old name); CLAUDE.md "Need rule
  provenance"; the 2026-09-01 guide-return-to-go plan;
  `scripts/check-provenance-journal-scope.sh`.

### Corpus, fence and held-out

- **Is:** "Real shaders are the corpus." DEV is every member any decision
  has touched. HELD-OUT is what has been used for nothing
  (NotoSansMono-Bold, `bench_scene_chrome`). A corpus tier is "a set of
  expressions closed under the fence equivalence". The split unit is the
  Family, one `(Band, Seed)` stream.
- **Is not:** a draw from an op distribution. A synthetic corpus is never a
  headline. Not pooled across DEV and HELD-OUT. Not a fence keyed on raw
  structure when features collapse literals.
- **Follows:** a member stops being held out once any decision reads it. The
  FenceKey is one imported function, and `Fence<Dev>`/`Fence<Final>` make
  direction a type. A new corpus needs a new acceptance criterion
  (`gen_bench_corpus` was deleted with the interpreter).
- **Lives:** `pixelflow_pipeline::training::{split, structural}`
  (`Fence<T: HoldoutSide>` in `pixelflow-pipeline/src/training/split.rs`,
  `FenceKey` in `structural.rs`); benchmark-correction §A–§C; 2026-08-17
  cost-model-domain J7–J8.

### Cost label, sentinel, journal entry, registration

- **Is:**
  - A **cost label** is "a measurement in a clock context": a value in
    `SessionNs`, with drift factor and sentinel context. The only
    conversion is `LocalNs::normalize(drift) → SessionNs`.
  - A **sentinel** is "a calibration signal, not a tripwire". Only a ≥50%
    regime change aborts.
  - A **journal entry** is a claim's provenance as a value: "A number
    without one of these attached is not a result."
  - A **registration** is a protocol committed before any training, to
    which only results may be appended.
- **Is not:** a bare `f64`. Not comparable across clocks. Not editable after
  a guided run; a change is a superseding registration.
- **Follows:** "Apply drift, then subtract overhead" is the only composition
  that type-checks. Never re-run a configuration.
- **Lives:** `pixelflow_pipeline::jit_bench::{CostLabel, LocalNs, SessionNs,
  SentinelContext}` (`pixelflow-pipeline/src/jit_bench.rs`),
  `journal::{JournalEntry, ConfigHash}` (`pixelflow-pipeline/src/journal.rs`);
  `2026-08-05-egraph-nnue-research-workflow.md` §0.3; 2026-08-17 J3, J4,
  J15; the 2026-09-01 phase3 registrations.
