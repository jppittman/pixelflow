# PixelFlow documentation

Three pages are the whole map; everything else hangs off them.

- **[`DENOTATION.md`](DENOTATION.md)** — what every object *means*: the type grammar, the laws, what is not a concept. Where any other document disagrees with it, the other document is history.
- **[`BACKLOG.md`](BACKLOG.md)** — what is open right now, one line per item, each pointing at the document that owns the detail. Edit it in the same CL as the work.
- **Each module's `intent.md`**, beside its `mod.rs` (or `Cargo.toml` for a crate) — what that module owns, never does, and is behind on, in its own words (`process/2026-10-15-intent-lives-with-the-module.md`). The ontology (`.claude/skills/ontology/`) says per term what the code treats it as and cites both.

## Live documents, by concept

A live document is one that still decides something or owns open work. Its status is one of the template's words (`Draft`, `Proposed`, `Registered`, `Plan of record`, `Landed`, `Closed`, `Superseded`, `Historical`; see [`process/DESIGN_DOC.md`](process/DESIGN_DOC.md)) and it records the commit it was verified against. A `Superseded` document stays live only while BACKLOG points at it as the owner of an open item.

### The language

*what a kernel is, how kernels compose, and the rulings that settled it.*

- [`language/2026-07-20-kernel-unification.md`](language/2026-07-20-kernel-unification.md) — `Plan of record`. The JIT-first Kernel migration plan: direction still agreed, P6 tail and P7–P12 unbuilt, partly superseded by the 2026-10-15 ruling, and not indexed in BACKLOG.
- [`language/2026-07-24-totality-and-the-cost-model.md`](language/2026-07-24-totality-and-the-cost-model.md) — `Plan of record`. Axiom essay (total kernel language, typed fields, cost-as-re-denotation) whose field-typing and kernel-algebra framing is replaced by the 2026-10-15 kernels-evaluate-kernels ruling.
- [`language/2026-09-09-composition-is-linking.md`](language/2026-09-09-composition-is-linking.md) — `Plan of record`. Plan of record for composition-as-linking: L1-L3 landed, L4-L6 (growth-gated Ref rule, call convention, tabulation) still open in BACKLOG, with the BufferIdentity/bind link step partly superseded.
- [`language/2026-09-10-one-name-bound-later.md`](language/2026-09-10-one-name-bound-later.md) — `Superseded`. Proposed one-kind-of-name denotation replacing Buffer/Gather/bind; its bind direction is overruled by kernels-evaluate-kernels §2, but BACKLOG N1 still carries its "one kind of name" goal as open.
- [`language/2026-09-24-one-pipeline.md`](language/2026-09-24-one-pipeline.md) — `Proposed`. Proposed one-pipeline denotation P(k,s,t) with a lattice folded before extraction; replaced in part by the 2026-10-15 kernels ruling and the-language-is-kernel, with its glyph and CL stages still unbuilt.
- [`language/2026-09-25-the-language-is-kernel.md`](language/2026-09-25-the-language-is-kernel.md) — `Plan of record`. Agreed front-end direction (one kernel! syntax, no tables or arrays, If, kernel-typed args) with C2-E1 still open; partly superseded on If, D1 and P by kernels-evaluate-kernels.
- [`language/2026-10-15-kernels-evaluate-kernels.md`](language/2026-10-15-kernels-evaluate-kernels.md) — `Accepted`. the binding rulings: If has one shape, kernels evaluate kernels, the schedule is per backend; everything is a kernel
- [`language/2026-10-17-the-arena-is-typed.md`](language/2026-10-17-the-arena-is-typed.md) — `Proposed`. Implements §2's types in the IR: a signature per op, a type column and typed handles on the arena, typed constants, explicit reinterpretation; deletes the mask and bitwise-domain conventions. Six CLs, byte-identical but the fold binder's.
- [`language/2026-10-20-a-bound-is-a-value.md`](language/2026-10-20-a-bound-is-a-value.md) — `Plan of record`. A fold's bounds are values placed by their variance, never a function of a coordinate; an index is a discrete kernel's application; a font at a tile extent is one program. Reverses the 2026-10-09 one-program-per-piece-count ruling and CLAUDE.md's recompile-per-count rule.

### The compiler and the search

*the optimizer, its laws, budgets, the cost model and the Guide.*

- [`compiler/2026-07-07-guided-saturation-redesign.md`](compiler/2026-07-07-guided-saturation-redesign.md) — `Plan of record`. Supervised guided-saturation direction: provenance and latency prior landed, trained Guide and Phase 3 thesis test still open; keep in place.
- [`compiler/2026-09-01-guide-candidate-context.md`](compiler/2026-09-01-guide-candidate-context.md) — `Plan of record`. JP-approved design for candidate-local context, cell buckets, coverage table and rule-conditioned generation; the Build phase is unstarted, not indexed in BACKLOG.
- [`compiler/2026-09-01-schedule-cost-model-denotation.md`](compiler/2026-09-01-schedule-cost-model-denotation.md) — `Plan of record`. Denotation cost(E)=analytic+residual for schedule-aware extraction; reranker unbuilt, owns open BACKLOG item E2; partly superseded by the 2026-10-15 ruling.
- [`compiler/2026-09-07-benchmark-correction.md`](compiler/2026-09-07-benchmark-correction.md) — `Plan of record`. Plan of record for real-shader benchmarking; §E re-validation items 0 and 2-6 remain unrun, and the 2026-10-15 ruling supersedes only two metrics.
- [`compiler/2026-09-07-demand-is-a-dag-property.md`](compiler/2026-09-07-demand-is-a-dag-property.md) — `Superseded`. Demand-as-DAG-property plan, superseded by the one-conditional and kernels-evaluate-kernels plans; kept in place because BACKLOG C2 (the open '8' waist bug) cites its section 9.
- [`compiler/2026-09-08-extraction-judge-registration.md`](compiler/2026-09-08-extraction-judge-registration.md) — `Registered`. Pre-registration for a learned per-class node scorer as the extraction DP's local objective; no run yet, §9 results empty.
- [`research-problems/2026-10-15-the-egraph.md`](research-problems/2026-10-15-the-egraph.md) — `Draft`. The open research problem: what saturation is for when it never finishes — budget allocation, the Guide's inconclusive results, linear vs bilinear unsettled, fonts a held-out domain only.

### Codegen and the schedule

*control flow, loops, the ISA tier, and the schedule as a factorization.*

- [`codegen/2026-09-08-one-conditional-three-lowerings.md`](codegen/2026-09-08-one-conditional-three-lowerings.md) — `Superseded`. Three-lowering conditional plan replaced by the 2026-10-15 kernels ruling (If has one shape); D1 landed then deleted, D2 withdrawn.
- [`codegen/2026-09-10-a-kept-structure-is-control-flow.md`](codegen/2026-09-10-a-kept-structure-is-control-flow.md) — `Plan of record`. Kept-structure-as-control-flow plan: its If domain-split is superseded, but the open fold-hoist remainder (N5/H4, now under H6) keeps it live.
- [`codegen/2026-09-10-a-surviving-reduce-is-a-loop.md`](codegen/2026-09-10-a-surviving-reduce-is-a-loop.md) — `Plan of record`. Plan to turn surviving Reduce folds into codegen loops; R0, 2a, 2b and the ExpandReduce deletion landed, but BACKLOG still lists 2c and E6 as open.
- [`codegen/2026-09-22-the-isa-is-decided-at-startup.md`](codegen/2026-09-22-the-isa-is-decided-at-startup.md) — `Plan of record`. Records the startup CPUID ISA decision and its deletions; the CI check rename it names is still open, so it stays in place.
- [`codegen/2026-10-15-a-schedule-is-a-factorization.md`](codegen/2026-10-15-a-schedule-is-a-factorization.md) — `Proposed`. the schedule is a factorization of the domain; algorithm / schedule / backend; why there is no matmul

### Graphics

*glyphs and the frame.*

- [`graphics/2026-09-09-a-glyph-is-a-circle.md`](graphics/2026-09-09-a-glyph-is-a-circle.md) — `Superseded`. Glyph-as-two-folds denotation; partly superseded by 2026-10-15 kernels ruling; §B hoist still referenced by open BACKLOG item H4.
- [`graphics/2026-09-09-glyph-as-a-fold-execution.md`](graphics/2026-09-09-glyph-as-a-fold-execution.md) — `Plan of record`. Staged plan to make a glyph one reduce-fold over a piece table; S0/S1 landed, S3 done, S2 cells-deletion still open per BACKLOG.

### The runtime and actors

*drivers, the engine troupe, the scheduler.*

- [`runtime/2026-09-03-wayland-driver.md`](runtime/2026-09-03-wayland-driver.md) — `Proposed`. Proposes a Wayland display driver on wayland-client with a staged DriverOut; phase 1 spike done, phases 2-5 unbuilt and undecided.
- [`docs/runtime/actor-scheduler-mealy-transducer.md`](runtime/actor-scheduler-mealy-transducer.md) — `Plan of record`. Actor design of record: effects-as-return-values Mealy transducer; scheduler primitives (Credit, Transducer lanes, Host, ports!) are built, the pixelflow-runtime engine migration in §9 is still open.
- [`docs/runtime/pixelflow-runtime-engine-mesh-migration.md`](runtime/pixelflow-runtime-engine-mesh-migration.md) — `Plan of record`. Engine-to-mesh runtime plan: rollout steps 1-3 and most of 5 are in the tree, but EngineHandler, the driver/PlatformOps conversion and the Known-unsettled items remain open.

### How we work

*tests, intent, style, CI, the design template.*

- [`process/2026-10-13-tests-check-the-as-if-contract.md`](process/2026-10-13-tests-check-the-as-if-contract.md) — `Plan of record`. Standing test-policy contract: tests check the as-if promise (values within tolerance, refusals, no crash), not optimizer mechanism; still cited by CLAUDE.md and code, so keep in place.
- [`process/2026-10-15-intent-lives-with-the-module.md`](process/2026-10-15-intent-lives-with-the-module.md) — `Plan of record`. one intent.md per module directory; the personas fold in; the check and the desloppify hook
- [`process/DESIGN_DOC.md`](process/DESIGN_DOC.md). the design template and the status vocabulary
- [`process/POSTSUBMIT.md`](process/POSTSUBMIT.md). what runs after merge and why
- [`process/STYLE.md`](process/STYLE.md). coding and review conventions

## Evidence

- [`results/`](results/) — recorded measurements: extraction, rewrite rules, JIT cost, actor benchmarks, the claims ledger. Point-in-time; heed each report's corrections.
- [`bugs/`](bugs/) — defect reports and the test-quality audit series (2026-07-20 → 2026-09-07, a chain: read the newest, [`bugs/2026-09-07-test-quality-audit-followup.md`](bugs/2026-09-07-test-quality-audit-followup.md), which carries the live backlog). Still open there: [`bugs/2026-07-15-pty-fork-malloc-deadlock.md`](bugs/2026-07-15-pty-fork-malloc-deadlock.md).

## Archive

Everything under [`archive/`](archive/) is history: landed, closed, superseded or kept for rationale. Each file opens with a banner naming its status and, where there is one, what replaced it. The thinking is often still worth reading; none of the code it describes should be copied without checking the tree. One line each, newest first.

### Plans

- [`archive/plans/2026-10-15-the-egraph-is-an-arena.md`](archive/plans/2026-10-15-the-egraph-is-an-arena.md) — `Landed` The e-graph's store as an arena (node ids are tags, membership an intrusive list, per-id columns, a per-kind class index, a cache value with no `add`), byte-identical by gate: all 13 CLs landed (#1416, #1418, #1420, #1422, #1423), Ir -32.2% and peak RSS -32.7% on the 29 font programs; §7 and the research problem hold what remains, §8 is JP's.
- [`archive/plans/2026-10-12-placement-reads-demand.md`](archive/plans/2026-10-12-placement-reads-demand.md) — `Superseded` Proposed D1 placement-by-demand design; its literals, demand and projected regions are withdrawn by the 2026-10-15 kernels ruling.
- [`archive/plans/2026-10-08-selection-is-a-phase.md`](archive/plans/2026-10-08-selection-is-a-phase.md) — `Landed` Selection phase plan (A–G series) fully landed on main through #1409; its If shape, arm guards and pass-chain-per-backend are replaced by the 2026-10-15 kernels-evaluate-kernels ruling.
- [`archive/plans/2026-09-23-an-integral-is-a-fold.md`](archive/plans/2026-09-23-an-integral-is-a-fold.md) — `Superseded` Interval-fold integral design, superseded by JP's 2026-09-29 deletion of all integral machinery; demand and mispredict-priced If parts superseded by kernels-evaluate-kernels.
- [`archive/plans/2026-09-23-a-glyph-is-a-formula.md`](archive/plans/2026-09-23-a-glyph-is-a-formula.md) — `Superseded` Glyph-as-formula denotation: the §1 exact-area formula is built, while the §3-§4 compiler/tree direction is replaced by the 2026-10-15 kernels ruling.
- [`archive/plans/2026-09-22-a-pointer-is-a-value.md`](archive/plans/2026-09-22-a-pointer-is-a-value.md) — `Landed` Pointer-class register allocation landed on main (carried-pointer loads cut 45 to 3 per call); the Buffer-leaf wording is superseded in part by the 2026-10-15 kernels ruling.
- [`archive/plans/2026-09-16-collapse-is-a-fold.md`](archive/plans/2026-09-16-collapse-is-a-fold.md) — `Superseded` Lattice-as-three-folds design; steps 1-6 landed, but the 2026-10-15 ruling (kernels-evaluate-kernels §3) rejects wrapping collapse as folds in pixelflow-core, so the direction is replaced.
- [`archive/plans/2026-09-12-emit-should-just-emit.md`](archive/plans/2026-09-12-emit-should-just-emit.md) — `Superseded` Guard-node and soft/hard-If plan replaced by one-shape If; its goal landed via #1309-#1313 and G1/G2 were retired.
- [`archive/plans/2026-09-09-the-graph-differentiates.md`](archive/plans/2026-09-09-the-graph-differentiates.md) — `Superseded` Proposed the e-graph differentiate through tabulations via a Gather materialization rewrite pair; replaced by the 2026-10-15 kernels-evaluate-kernels ruling, which refuses Dwrt of discrete kernels and makes Gather a backend realization.
- [`archive/plans/2026-09-09-exprarena-on-dag.md`](archive/plans/2026-09-09-exprarena-on-dag.md) — `Landed` Stages A and C of the ExprArena-on-Dag port landed; optional Stage D (retiring ExprId) was never started and is not on BACKLOG, so the doc is Landed and archivable.
- [`archive/plans/2026-09-09-a-run-is-a-glyph.md`](archive/plans/2026-09-09-a-run-is-a-glyph.md) — `Landed` Run-as-monoid glyph design, executed: one buffer slot for runs, Glyph::over and Kernel::fold landed; measurement and coverage shape have since moved on.
- [`archive/plans/2026-09-09-a-fold-is-a-node.md`](archive/plans/2026-09-09-a-fold-is-a-node.md) — `Superseded` Phases 1-2 (Fold node, peel/empty rules) landed; the Phase 3 legalize-reorder was never landed and its ExpandReduce legalizer was deleted, so the surviving-fold-is-a-loop plan and kernels-evaluate-kernels replace it.
- [`archive/plans/2026-09-08-macro-tier-is-arena-native.md`](archive/plans/2026-09-08-macro-tier-is-arena-native.md) — `Landed` Macro tier moved onto ExprArena: expansion-time Dwrt differentiation, PARAM_VAR_BASE and the AST optimizer are deleted, and ENode::Param replaces the encoding. All five steps are visibly done in the tree.
- [`archive/plans/2026-09-08-loop-blinn-glyph.md`](archive/plans/2026-09-08-loop-blinn-glyph.md) — `Superseded` Chord/shadow winding-plus-distance glyph design, replaced in execution by the closed-form pixel-area formulation; §7 findings remain as rationale.
- [`archive/plans/2026-09-08-extraction-witnesses.md`](archive/plans/2026-09-08-extraction-witnesses.md) — `Closed` Pre-registered extraction-witness program (2026-09-08) that ran to its recorded end; the results doc answered it and its one mechanical fix (DP fixpoint) landed.
- [`archive/plans/2026-09-08-egraph-cpu-memory-profile.md`](archive/plans/2026-09-08-egraph-cpu-memory-profile.md) — `Historical` Engine-level e-graph CPU and memory profile from 2026-09-08; several of its ranked findings are now built or stale against the tree.
- [`archive/plans/2026-09-06-uniform-slot-identity.md`](archive/plans/2026-09-06-uniform-slot-identity.md) — `Landed` Uniform leaf landed as U0 (#1183) and later stages; buffer/ABI direction replaced by kernels-evaluate-kernels, so archive as Superseded.
- [`archive/plans/2026-09-06-lattice-is-the-index.md`](archive/plans/2026-09-06-lattice-is-the-index.md) — `Landed` All five stages (U0, L1-L4) landed; G1 ran as its own plan; §3/§5 guard and domain-split text is superseded in part by the 10-15 If ruling.
- [`archive/plans/2026-09-06-kernel-with-a-lattice.md`](archive/plans/2026-09-06-kernel-with-a-lattice.md) — `Superseded` Lattice/kernel collapse plan; S1-S4a landed, but its bound-buffer and If-mispredict direction is replaced by the 2026-10-15 ruling.
- [`archive/plans/2026-09-06-egraph-at-production-scale.md`](archive/plans/2026-09-06-egraph-at-production-scale.md) — `Superseded` Measured chrome-scene e-graph use case whose guard/mispredict direction is replaced by the 2026-10-15 If ruling; schedule-domain question moved to a schedule-factorization note.
- [`archive/plans/2026-09-04-ir-as-a-trait.md`](archive/plans/2026-09-04-ir-as-a-trait.md) — `Landed` Proposed the Ir trait (project/embed) and one generic arena-to-e-graph insert with per-tier Vocabulary; all of it is in the tree at HEAD 2a99d933, with the Buffer/Gather parts superseded by the 2026-10-15 plan.
- [`archive/plans/2026-09-02-phase3-forward-port.md`](archive/plans/2026-09-02-phase3-forward-port.md) — `Landed` Phase 3 forward-port landed (Phase A/B and docs, commit 418542d7); only deferred item A10 lacks a BACKLOG owner.
- [`archive/plans/2026-09-02-optimizer-api.md`](archive/plans/2026-09-02-optimizer-api.md) — `Landed` Records the optimizer-API decisions and the five laws; the Optimizer, Budget and RuleSet shipped, with guide loop and JIT-cache keying still open.
- [`archive/plans/2026-09-02-bilinear-guide-registration.md`](archive/plans/2026-09-02-bilinear-guide-registration.md) — `Closed` Pre-registered test of whether a bilinear rule-by-context Guide beats the additive one under shift; run completed with H_form-null, verdict later retracted under L047.
- [`archive/plans/2026-09-01-register-allocation-escape-hatches.md`](archive/plans/2026-09-01-register-allocation-escape-hatches.md) — `Superseded` Allocator escape-hatch plan; its goal is continued by selection-is-a-phase and the CPU pass chain of kernels-evaluate-kernels; several landed claims no longer hold on main.
- [`archive/plans/2026-09-01-production-budget-determinism.md`](archive/plans/2026-09-01-production-budget-determinism.md) — `Landed` Budgets are denominated in rule applications and wall clock is a panicking safety ceiling; the design is in the tree, with the classical cap unpinned 2026-09-29.
- [`archive/plans/2026-09-01-phase3-round2-rule-scaling.md`](archive/plans/2026-09-01-phase3-round2-rule-scaling.md) — `Closed` Round 2 H1/H2 pre-registration design for rule-count scaling; its registration chain (v1 to v2 to v3) replaced it, and v3 is the one to read.
- [`archive/plans/2026-09-01-phase3-round2-registration.md`](archive/plans/2026-09-01-phase3-round2-registration.md) — `Superseded` Round 2 pre-registration of regret-at-budget vs rule count under append-order sweeps; replaced by v2 (interleaved order) and v3 (H1 reading), v3 is the one to read.
- [`archive/plans/2026-09-01-phase3-round2-registration-v3.md`](archive/plans/2026-09-01-phase3-round2-registration-v3.md) — `Superseded` Round 2 v3 registered a static numeric-first reorder as a 96.58% to 1.12% regret quick win; the 2026-09-07 ledger retracted that production reading and the benchmark correction replaces it.
- [`archive/plans/2026-09-01-phase3-round2-registration-v2.md`](archive/plans/2026-09-01-phase3-round2-registration-v2.md) — `Superseded` Interleaved-order Round 2 registration; its H1 reading (unguided regret rises/falls with rule count) was superseded by v3, while its measured data stand.
- [`archive/plans/2026-09-01-phase3-round1b-domain-shift-registration.md`](archive/plans/2026-09-01-phase3-round1b-domain-shift-registration.md) — `Superseded` Round 1b domain-shift pre-registration; its H_null verdict was retracted (L038) and the Guide question moved to the corrected benchmark plan.
- [`archive/plans/2026-09-01-phase3-registration.md`](archive/plans/2026-09-01-phase3-registration.md) — `Superseded` Round 1 Guide pre-registration (B=100/200, Y=16.3%/9.0%) whose constants were retracted on 2026-09-07 as tree-unit artifacts; replaced by benchmark correction and claims ledger.
- [`archive/plans/2026-09-01-loop-aware-codegen.md`](archive/plans/2026-09-01-loop-aware-codegen.md) — `Superseded` Loop-aware codegen design (lattice as compile-time binder nest in the e-graph); replaced by the 2026-10-15 ruling that moves the lattice traversal into the CPU schedule, with BACKLOG H6 carrying the implementation.
- [`archive/plans/2026-09-01-guide-return-to-go.md`](archive/plans/2026-09-01-guide-return-to-go.md) — `Closed` Pre-registered linear return-to-go Guide program; Rounds 1-4 ran to end, verdict retracted as tree-unit invalid by 2026-09-07 ledger.
- [`archive/plans/2026-09-01-dead-code-with-ideas.md`](archive/plans/2026-09-01-dead-code-with-ideas.md) — `Historical` A 2026-09-01 dead-code audit whose DELETE verdicts were largely executed; its open D1-D6 decisions are not BACKLOG-owned.
- [`archive/plans/2026-08-31-guide-design-revision.md`](archive/plans/2026-08-31-guide-design-revision.md) — `Superseded` Guide design revision with a pre-registered Phase 3 experiment; its §5 framing was withdrawn 2026-09-07 as wrong-regime, so the claims ledger and benchmark correction replace it.
- [`archive/plans/2026-08-17-egraph-vsa-nnue-research-notes.md`](archive/plans/2026-08-17-egraph-vsa-nnue-research-notes.md) — `Historical` Research notes on e-graph NNUE, VSA and literature; the learned head and its measured PR #984 fixes no longer exist on the tree, so archive as Historical.
- [`archive/plans/2026-08-17-cost-model-domain.md`](archive/plans/2026-08-17-cost-model-domain.md) — `Superseded` Cost-model domain model for the closed extraction-head NNUE program; its live rows are superseded by the 2026-09-01 denotation doc.
- [`archive/plans/2026-08-08-egraph-constant-domain-spike.md`](archive/plans/2026-08-08-egraph-constant-domain-spike.md) — `Historical` Spike recommending ENode split into Num(Dyadic)/Bits(u32) for an exact dyadic fold; the split was never built and the doc's tree claims have drifted.
- [`archive/plans/2026-08-05-egraph-nnue-research-workflow.md`](archive/plans/2026-08-05-egraph-nnue-research-workflow.md) — `Closed` Closed NNUE extraction-head workflow (tie with latency prior); north-star matmul-over-Gather direction replaced by the 10-15 kernels ruling.
- [`archive/plans/2026-08-02-ir-layering.md`](archive/plans/2026-08-02-ir-layering.md) — `Historical` Ten-phase IR layering plan; most structural phases landed (backend moved, passes.rs, CompileWorkspace gone), the rest superseded by the 2026-10-15 kernels ruling.
- [`archive/plans/2026-07-28-jit-performance-parity.md`](archive/plans/2026-07-28-jit-performance-parity.md) — `Superseded` Scoping of JIT performance parity; its loop-in-JIT collapse scaffold and emitter-level LICM were replaced by collapse-as-a-fold, and its Gather/Select claims by the kernels-evaluate-kernels ruling.
- [`archive/plans/2026-02-25-unified-training-plan.md`](archive/plans/2026-02-25-unified-training-plan.md)
- [`archive/plans/2026-02-25-unified-training-design.md`](archive/plans/2026-02-25-unified-training-design.md)
- [`archive/plans/2025-02-21-kernel-jit-feature-parity.md`](archive/plans/2025-02-21-kernel-jit-feature-parity.md) — `Superseded` Superseded April-era implementation plan for kernel_jit! param baking; the macro and its files are gone, replaced by kernel-unification.
- [`archive/plans/2025-02-21-kernel-jit-feature-parity-design.md`](archive/plans/2025-02-21-kernel-jit-feature-parity-design.md) — `Superseded` Superseded design for kernel_jit! parity with kernel!; retired by kernel-unification, keeping only its param-baking semantics.

### Designs

- [`archive/designs/pty-actor-troupe.md`](archive/designs/pty-actor-troupe.md) — `Landed` Unifies PTY reader, parser and writer as a troupe! of actors; Phases 1-2 are in the tree, Phase 3 was declined.
- [`archive/designs/opkind-numbering-is-private.md`](archive/designs/opkind-numbering-is-private.md) — `Landed` Records that OpKind numbering is private and marshal/unmarshal is the byte boundary, now implemented in kind.rs with no open work left.
- [`archive/designs/lattice-scheduling-types.md`](archive/designs/lattice-scheduling-types.md) — `Superseded` Apr 2026 thesis that lattice types are the schedule and variance is a type parameter; superseded 2026-10-15 by kernels-evaluate-kernels, which keeps schedule and traversal per backend.
- [`archive/designs/compiler-architecture-2026.md`](archive/designs/compiler-architecture-2026.md) — `Superseded` Proposed 2026 IR unification (pixelflow-nnue, OpType, backends in pixelflow-ir); already bannered Superseded by the ir-layering plan and the kernel rulings.
- [`archive/designs/assembler-as-functor.md`](archive/designs/assembler-as-functor.md) — `Superseded` Proposed a one-functor, one-shared-regalloc back-end and a builder API; the 2026-10-15 ruling replaces the shared-backend premise and the split landed elsewhere.
- [`docs/archive/designs/actor-scheduler-supervisor-migration.md`](archive/designs/actor-scheduler-supervisor-migration.md) — `Superseded` Kubernetes/kubelet supervision design for actor-scheduler, superseded by the Mealy-transducer actor design; none of its types were built.
- [`docs/archive/designs/actor-scheduler-backpressure.md`](archive/designs/actor-scheduler-backpressure.md) — `Landed` Explains why actor sends back off on one shared ladder and why blown ladders panic; the mechanism it describes is built.
- [`archive/designs/REDUCTIONS_AND_FOLDS.md`](archive/designs/REDUCTIONS_AND_FOLDS.md) — `Historical` April 2026 reductions/folds design; its variance rule shipped, but its NNUE, materialization and lattice-loop claims were replaced, so archive as Historical.
- [`archive/designs/ML_AUTODIFF_PIPELINE.md`](archive/designs/ML_AUTODIFF_PIPELINE.md) — `Superseded` April-era sketch of ML training via e-graph autodiff over Lattice inputs; the Lattice-as-input model is replaced by the 2026-10-15 kernels plan, and the MCTS/NNUE and Dual parts were never built or were removed.
- [`archive/designs/ML_AND_LINEAR_ALGEBRA.md`](archive/designs/ML_AND_LINEAR_ALGEBRA.md) — `Historical` Twelve-line April stub on functional ML as pure linear maps; self-declared Historical, owns no open BACKLOG item, and its idea lives in the totality design doc.
- [`archive/designs/LATTICE_EVAL.md`](archive/designs/LATTICE_EVAL.md) — `Historical` Apr 2026 design making Lattice the evaluator with loop scheduling in pixelflow-core; replaced by kernels-evaluate-kernels (collapsed lattice is a discrete kernel, loop nest per backend). Keeps the index(collapse(f))=f law.
- [`archive/designs/KERNELS_AND_LATTICES.md`](archive/designs/KERNELS_AND_LATTICES.md) — `Superseded` Language-level bound-buffer memory design (Buffer/Gather/Frozen/Pinned); its direction is replaced by the 2026-10-15 kernels-evaluate-kernels ruling, with M1-M2 and the internal-loop collapse landed.
- [`archive/designs/BRAINSTORM_VARIANCE_EGRAPH.md`](archive/designs/BRAINSTORM_VARIANCE_EGRAPH.md) — `Historical` April 2026 brainstorm on e-graph variance analysis, Lattice-as-schedule and NNUE/ILP extraction; its schedule-by-lattice direction is replaced by the 2026-10-15 kernels ruling, and its variance-as-analysis idea is live in pixelflow-ir.
- [`archive/designs/2026-08-31-preemption-as-transaction-abort.md`](archive/designs/2026-08-31-preemption-as-transaction-abort.md) — `Historical` Exploratory, never-adopted idea to preempt green actors by aborting steps to the sweep loop; nothing built, gated on an unmeasured overrun.
- [`archive/designs/2026-08-31-hardware-sandboxed-kernel-preemption.md`](archive/designs/2026-08-31-hardware-sandboxed-kernel-preemption.md) — `Historical` Exploratory KVM-based kernel preemption idea, explicitly not adopted and unbuilt; kept only for rationale and its prototype traps.
- [`archive/designs/2026-07-25-two-level-ir-and-backend-completeness.md`](archive/designs/2026-07-25-two-level-ir-and-backend-completeness.md) — `Superseded` Proposed two-level IR staging and per-ISA loop emission; its code paths were rewritten, and the binder-as-loop direction now lives in the kernels-evaluate-kernels ruling and collapse/reduce plans.
- [`archive/designs/2026-07-23-lower-realize-boundary.md`](archive/designs/2026-07-23-lower-realize-boundary.md) — `Superseded` Proposed Lower/Lattice::realize boundary for P5; landed 2026-07-23, realize plumbing later deleted, and direction superseded by kernel-unification and kernels-evaluate-kernels.
- [`archive/designs/2026-07-23-jit-orthodoxy-survey.md`](archive/designs/2026-07-23-jit-orthodoxy-survey.md) — `Historical` Surveys PixelFlow's JIT against V8/LuaJIT/HotSpot and Halide; its branchless-If claim is superseded by the 2026-10-15 kernels-evaluate-kernels plan, and its code-cache gap is closed.

### Notes and analyses

- [`archive/notes/nnue-training-pipeline.md`](archive/notes/nnue-training-pipeline.md) — `Historical`
- [`archive/notes/lample_charton_2019_symbolic_math.md`](archive/notes/lample_charton_2019_symbolic_math.md)
- [`archive/notes/function-namespace-audit.md`](archive/notes/function-namespace-audit.md) — `Historical` Generated function-name namespace audit at 6a7930e5; its main target, pixelflow-ir's backend/emit tree, was moved or deleted, so it is rationale only.
- [`archive/notes/function-coordinates.tsv`](archive/notes/function-coordinates.tsv)
- [`archive/notes/function-audit.tsv`](archive/notes/function-audit.tsv)
- [`archive/notes/SEARCH_PIPELINE_DESIGN.md`](archive/notes/SEARCH_PIPELINE_DESIGN.md) — `Superseded` Unshipped MCTS/REINFORCE composable-search interface proposal, replaced by the supervised guided-saturation redesign; its traits never existed in the tree.
- [`archive/notes/NNUE_INTEGRATION_STATUS.md`](archive/notes/NNUE_INTEGRATION_STATUS.md) — `Superseded` Current-state stub for the deleted NNUE extraction-head program; its kept denotation lives in the 2026-09-01 schedule-cost plan.
- [`archive/notes/MESSAGE_CUJ_COVERAGE.md`](archive/notes/MESSAGE_CUJ_COVERAGE.md) — `Historical` Coverage plan for actor message CUJs; its own banner marks it Historical, and no open BACKLOG item depends on it, so it can be archived.
- [`archive/notes/KERNEL_PARAM_LIMIT_INVESTIGATION.md`](archive/notes/KERNEL_PARAM_LIMIT_INVESTIGATION.md) — `Historical` A January 2026 investigation of the >3-parameter kernel! trait-bound failure; the combinator tier it analysed is deleted and the arena kernel path replaced it, so it is archival history.
- [`archive/notes/GNN_REWRITE_GUIDANCE_VISION.md`](archive/notes/GNN_REWRITE_GUIDANCE_VISION.md) — `Historical`
- [`archive/notes/FLAT_CONTEXT_TUPLE_PROTOTYPE.md`](archive/notes/FLAT_CONTEXT_TUPLE_PROTOTYPE.md) — `Superseded` Flat WithContext prototype for nested-Let trait explosion; the combinator tier it targets was retired by the Kernel/ExprArena architecture, so it is Superseded.
- [`archive/notes/EGRAPH_SEARCH_INTEGRATION.md`](archive/notes/EGRAPH_SEARCH_INTEGRATION.md) — `Historical` Short 2026-08-05 current-state stub on the e-graph compiler path and pointers to NNUE/guided-saturation plans; its tree paths are now stale.
- [`archive/notes/EGRAPH_OPTIMIZATION_ARCHITECTURE.md`](archive/notes/EGRAPH_OPTIMIZATION_ARCHITECTURE.md) — `Superseded` Describes the deleted NNUE extraction head and unified training spine; replaced by the static latency prior, the Reranker seam, and the saturation Guide.
- [`archive/notes/COMPILER_OPPORTUNITIES.md`](archive/notes/COMPILER_OPPORTUNITIES.md) — `Historical` January 2026 ranked compiler wishlist; its ideas were realized differently or are dead APIs, and BACKLOG does not own it.
- [`archive/notes/COMPILER_ANALYSIS.md`](archive/notes/COMPILER_ANALYSIS.md) — `Historical` January 2026 compiler-pipeline audit; its IR-integration and cost-calibration findings were overtaken by arena-native optimization and the 2026-09-01 dead-code deletions.
- [`archive/notes/AUTODIFF_RENDERING.md`](archive/notes/AUTODIFF_RENDERING.md) — `Superseded` Proposed forward-mode Jet3 autodiff for CPU rendering and Catmull-Clark subdivision; the Jet tier was retired in favour of symbolic Dwrt/Kernel::dx.

### The April 2026 rewrite plans

- [`archive/superpowers/plans/2026-04-07-team4-backward-training.md`](archive/superpowers/plans/2026-04-07-team4-backward-training.md)
- [`archive/superpowers/plans/2026-04-07-team3-forward-pass.md`](archive/superpowers/plans/2026-04-07-team3-forward-pass.md)
- [`archive/superpowers/plans/2026-04-07-team2-ir-pullbacks.md`](archive/superpowers/plans/2026-04-07-team2-ir-pullbacks.md)
- [`archive/superpowers/plans/2026-04-07-team1-lattice-extensions.md`](archive/superpowers/plans/2026-04-07-team1-lattice-extensions.md)
- [`archive/superpowers/specs/2026-04-07-pipeline-pixelflow-rewrite-design.md`](archive/superpowers/specs/2026-04-07-pipeline-pixelflow-rewrite-design.md)

### Papers

- [`archive/papers/type-class-morphisms-long.pdf`](archive/papers/type-class-morphisms-long.pdf)
- [`archive/papers/fop-conal.pdf`](archive/papers/fop-conal.pdf)
- [`archive/papers/2425_deep_learning_for_symbolic_mat.pdf`](archive/papers/2425_deep_learning_for_symbolic_mat.pdf)

