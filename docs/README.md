# PixelFlow Documentation

This page is the landing page for repository documentation. A document's classification
describes how it should be read today; directory names alone do not establish currency.

## Classification vocabulary

- **Current architecture** — describes the system as it exists now.
- **Plan of record** — approved or active direction for work not yet fully implemented.
- **Experiment/result** — an observation, audit, benchmark, investigation, or research
  reference; it is evidence rather than a promise about current architecture.
- **Historical/superseded** — retained for rationale or archaeology, but not authoritative.
- **Active bug** — a known unresolved defect or quality gap.
- **Fixed bug** — a defect report whose described fault has been corrected.

Classification is separate from design workflow status. **Every future plan and design must
use exactly one status from the vocabulary in
[`templates/DESIGN_DOC.md`](templates/DESIGN_DOC.md)** — `Draft`, `Proposed`, `Registered`,
`Plan of record`, `Landed`, `Closed`, `Superseded` or `Historical` — and must record a
**`Verified against:` commit sha** if it makes claims about the tree. Use `Supersedes` or
`Superseded by` metadata to record lineage *in addition to* the status.

That vocabulary was widened on 2026-09-09. It used to be `Draft | Review | Approved |
Implemented`, which 3 of 59 plans and designs actually used; the rest had invented about
twenty words of their own, several of which said more than any of the four. The rule now
matches what the corpus needs rather than asking 56 documents to lie about themselves.

## Current architecture

- [`designs/2026-07-24-totality-and-the-cost-model.md`](designs/2026-07-24-totality-and-the-cost-model.md)
  — design-of-record axiom for the total kernel language. Some consequences it describes
  (typed discrete fields and cost re-denotation) remain plan-of-record work.
- [`designs/KERNELS_AND_LATTICES.md`](designs/KERNELS_AND_LATTICES.md) — implemented kernel
  and lattice substrate. Its milestone list is a point-in-time snapshot; use the newer
  kernel-unification plan for consumer migration status.
- [`designs/2026-07-25-two-level-ir-and-backend-completeness.md`](designs/2026-07-25-two-level-ir-and-backend-completeness.md)
  — audit of the lowering/backend boundary and the first landed instruction/assembler split;
  follow-up work is identified explicitly in the document.
- [`designs/pty-actor-troupe.md`](designs/pty-actor-troupe.md) — implemented PTY actor wiring.
- [`STYLE.md`](STYLE.md) — current coding and review conventions.

## Plan of record

- [`plans/2026-09-02-optimizer-api.md`](plans/2026-09-02-optimizer-api.md) — the optimizer
  entry point (`Optimizer`/`RuleSet`/`Budget` + optional `SaturationGuide`, `Reranker`,
  `Observer`), the five denotational laws as audited against the code, and the gap table
  (G1–G8) for the cost-model research sitting behind it. Nothing in the surface has landed;
  the audit findings and the archive moves have.
- [`plans/2026-07-20-kernel-unification.md`](plans/2026-07-20-kernel-unification.md) — the
  active migration from type-level combinator emission to arena-backed `Kernel` values. Its
  phase annotations distinguish landed slices from future work.
- [`plans/2026-07-07-guided-saturation-redesign.md`](plans/2026-07-07-guided-saturation-redesign.md)
  — supervised guided-saturation research direction. Provenance and the static latency prior
  are landed; a trained guide and guided-saturation thesis test are not.
- [`designs/actor-scheduler-mealy-transducer.md`](designs/actor-scheduler-mealy-transducer.md)
  and [`designs/pixelflow-runtime-engine-mesh-migration.md`](designs/pixelflow-runtime-engine-mesh-migration.md)
  — draft actor/runtime designs. The scheduler primitives have landed; the runtime mesh
  migration has not.
- [`designs/2026-08-31-preemption-as-transaction-abort.md`](designs/2026-08-31-preemption-as-transaction-abort.md)
  and [`designs/2026-08-31-hardware-sandboxed-kernel-preemption.md`](designs/2026-08-31-hardware-sandboxed-kernel-preemption.md)
  — two alternative designs for preempting a green actor mid-step, which the mealy-transducer
  design's §5 rules out. Neither is adopted and nothing has landed; they are recorded reasoning,
  gated on a step actually being observed to overrun.
- [`designs/LATTICE_EVAL.md`](designs/LATTICE_EVAL.md),
  [`designs/lattice-scheduling-types.md`](designs/lattice-scheduling-types.md),
  [`designs/REDUCTIONS_AND_FOLDS.md`](designs/REDUCTIONS_AND_FOLDS.md),
  [`designs/ML_AND_LINEAR_ALGEBRA.md`](designs/ML_AND_LINEAR_ALGEBRA.md), and
  [`designs/ML_AUTODIFF_PIPELINE.md`](designs/ML_AUTODIFF_PIPELINE.md) — related language
  and scheduling designs that remain directional rather than descriptions of completed code.

## Experiments and results

- [`results/`](results/) — recorded extraction, rewrite-rule, JIT-cost, and actor benchmark
  results. These are point-in-time evidence; heed corrections in each report.
- [`COMPILER_ANALYSIS.md`](COMPILER_ANALYSIS.md),
  [`COMPILER_OPPORTUNITIES.md`](COMPILER_OPPORTUNITIES.md),
  [`KERNEL_PARAM_LIMIT_INVESTIGATION.md`](KERNEL_PARAM_LIMIT_INVESTIGATION.md), and
  [`function-namespace-audit.md`](function-namespace-audit.md) — analyses and audits.
- [`designs/2026-07-23-jit-orthodoxy-survey.md`](designs/2026-07-23-jit-orthodoxy-survey.md),
  [`AUTODIFF_RENDERING.md`](AUTODIFF_RENDERING.md), and the papers and paper notes in this
  directory — research context, not implementation contracts.
- [`designs/BRAINSTORM_VARIANCE_EGRAPH.md`](designs/BRAINSTORM_VARIANCE_EGRAPH.md) and
  [`archive/GNN_REWRITE_GUIDANCE_VISION.md`](archive/GNN_REWRITE_GUIDANCE_VISION.md) — exploratory ideas
  (the GNN vision is archived; its offline-teacher framing survives in
  [`plans/2026-08-17-egraph-vsa-nnue-research-notes.md`](plans/2026-08-17-egraph-vsa-nnue-research-notes.md)).

## Historical or superseded

- [`EGRAPH_OPTIMIZATION_ARCHITECTURE.md`](EGRAPH_OPTIMIZATION_ARCHITECTURE.md) — obsolete
  critic/REINFORCE training spine. Its “current” wording predates and is superseded by the
  guided-saturation redesign; do not use it as the live architecture.
- [`SEARCH_PIPELINE_DESIGN.md`](SEARCH_PIPELINE_DESIGN.md) — unshipped MCTS/REINFORCE
  interface proposal, superseded by the guided-saturation redesign.
- [`plans/archive/2026-02-25-unified-training-design.md`](plans/archive/2026-02-25-unified-training-design.md)
  and [`plans/archive/2026-02-25-unified-training-plan.md`](plans/archive/2026-02-25-unified-training-plan.md)
  — obsolete critic/REINFORCE training path (`NNUE_TRAINING_RECIPE.md`, which described the
  same deleted system, was removed 2026-08-05).
- [`archive/nnue-training-pipeline.md`](archive/nnue-training-pipeline.md) — earlier curriculum
  and best-first proposal.
- [`NNUE_INTEGRATION_STATUS.md`](NNUE_INTEGRATION_STATUS.md) and
  [`EGRAPH_SEARCH_INTEGRATION.md`](EGRAPH_SEARCH_INTEGRATION.md) — short current-state
  stubs (rewritten 2026-08-05) pointing at the guided-saturation and research-workflow plans.
- [`plans/2026-08-05-egraph-nnue-research-workflow.md`](plans/2026-08-05-egraph-nnue-research-workflow.md)
  — the extraction-head research workflow; closed 2026-09-01 when the workshop paper it targets
  found a tie rather than a win, see
  [`plans/2026-09-01-schedule-cost-model-denotation.md`](plans/2026-09-01-schedule-cost-model-denotation.md)
  for the outcome.
- [`designs/actor-scheduler-supervisor-migration.md`](designs/actor-scheduler-supervisor-migration.md)
  — explicitly superseded by the Mealy-transducer design.
- [`designs/2026-07-23-lower-realize-boundary.md`](designs/2026-07-23-lower-realize-boundary.md)
  — transitional `Lower`/`realize` boundary, superseded by the JIT-first `Kernel` course
  correction recorded in the kernel-unification plan.
- [`plans/2025-02-21-kernel-jit-feature-parity-design.md`](plans/2025-02-21-kernel-jit-feature-parity-design.md),
  [`plans/2025-02-21-kernel-jit-feature-parity.md`](plans/2025-02-21-kernel-jit-feature-parity.md),
  and [`superpowers/`](superpowers/) — completed or superseded implementation-era plans.
- [`FLAT_CONTEXT_TUPLE_PROTOTYPE.md`](FLAT_CONTEXT_TUPLE_PROTOTYPE.md),
  [`MESSAGE_CUJ_COVERAGE.md`](MESSAGE_CUJ_COVERAGE.md), and
  [`designs/compiler-architecture-2026.md`](designs/compiler-architecture-2026.md) — retained
  point-in-time proposals; verify their claims against current code before use.

## Bugs

### Active bugs

- [`bugs/2026-07-15-pty-fork-malloc-deadlock.md`](bugs/2026-07-15-pty-fork-malloc-deadlock.md)
  — diagnosed and explicitly not yet fixed.
- [`bugs/2026-07-20-test-quality-audit.md`](bugs/2026-07-20-test-quality-audit.md) — mixed
  audit report with unresolved test-quality findings; classified active until those gaps are
  dispositioned.

### Fixed bugs

- [`bugs/2026-07-21-openpty-not-thread-safe.md`](bugs/2026-07-21-openpty-not-thread-safe.md)
  — concurrent-test contract fix.
- [`bugs/2026-07-22-trig-chebyshev-coefficients-wrong.md`](bugs/2026-07-22-trig-chebyshev-coefficients-wrong.md)
  — trigonometric approximation and regression-test fixes.

### The test-quality audit series — read the newest one only

`bugs/` holds 23 documents from one scheduled series running 2026-07-20 → 2026-09-07. They
are a **chain**: each pass names its predecessor, re-checks that pass's "Recommended next
steps" against the tree, and carries forward whatever is still open. So the newest document
carries the live backlog and the older ones are the audit trail — reading all 23 to find out
what is open is the mistake the chain's shape invites.

Start at [`bugs/2026-09-07-test-quality-audit-followup.md`](bugs/2026-09-07-test-quality-audit-followup.md).
Two standing cautions the series records about itself: the 2026-08-07 pass found that two
passes had swept the *same* commit range because each derived its window from what the
previous document named rather than from `main` (they independently fixed the same
violation), and the 2026-09-01 pass notes that several findings live in PRs that never
merged (#1049, #1050, #1054, #1154), so a "closed" item may not be closed on `main`.

## Complete index of plans and designs

The curated sections above are the *start here* view and are deliberately partial. This is
the full list, so nothing is invisible. Every entry's own header states its status; where a
document is historical it now says so in a banner on line 1, which
[`scripts/check-doc-paths.sh`](../scripts/check-doc-paths.sh) treats as load-bearing — a
plan or design may only name a source file the tree lacks if it is bannered, says the file is
gone, or is recorded in `scripts/doc-paths-baseline.txt`.

**Current direction — read these first.** `plans/2026-09-06-kernel-with-a-lattice.md`
(plan of record), `plans/2026-09-06-lattice-is-the-index.md` (landed),
`plans/2026-09-06-uniform-slot-identity.md` (U0 landed),
`plans/2026-09-08-one-conditional-three-lowerings.md` (draft, nothing built),
`plans/2026-09-09-exprarena-on-dag.md` (proposed),
`plans/2026-09-08-macro-tier-is-arena-native.md`,
`designs/2026-07-24-totality-and-the-cost-model.md` (axiom layer),
`plans/2026-07-20-kernel-unification.md` (plan of record).

**Compiler, codegen, optimizer.** `plans/2026-09-02-optimizer-api.md` (the five laws and
the G1–G8 gap table), `plans/2026-09-01-loop-aware-codegen.md` (stage 0 landed),
`plans/2026-09-01-register-allocation-escape-hatches.md`,
`plans/2026-09-01-production-budget-determinism.md` (revised 2026-09-08 for the class cap),
`plans/2026-09-06-egraph-at-production-scale.md` (measured facts, not a plan),
`plans/2026-09-08-egraph-cpu-memory-profile.md`, `plans/2026-09-04-ir-as-a-trait.md`,
`plans/2026-08-02-ir-layering.md`, `plans/2026-08-08-egraph-constant-domain-spike.md`,
`designs/2026-07-25-two-level-ir-and-backend-completeness.md`,
`designs/assembler-as-functor.md`, `designs/opkind-numbering-is-private.md`.

**Cost model and the Guide** — read the claims ledger (PR #1207) before citing any number
from these. `plans/2026-07-07-guided-saturation-redesign.md`,
`plans/2026-08-31-guide-design-revision.md`, `plans/2026-09-01-schedule-cost-model-denotation.md`,
`plans/2026-09-01-guide-candidate-context.md`, `plans/2026-09-01-guide-return-to-go.md`,
`plans/2026-09-02-phase3-forward-port.md`, and the pre-registration series:
`plans/2026-09-01-phase3-registration.md`,
`plans/2026-09-01-phase3-round1b-domain-shift-registration.md`,
`plans/2026-09-01-phase3-round2-rule-scaling.md`, and
`plans/2026-09-01-phase3-round2-registration.md` →
`plans/2026-09-01-phase3-round2-registration-v2.md` →
`plans/2026-09-01-phase3-round2-registration-v3.md` (each supersedes the last, so
v3 is the one to read), `plans/2026-09-02-bilinear-guide-registration.md`.

**Actors, runtime, platform.** `designs/actor-scheduler-mealy-transducer.md` (design of
record), `designs/actor-scheduler-backpressure.md`,
`designs/pixelflow-runtime-engine-mesh-migration.md` (not landed),
`designs/pty-actor-troupe.md` (implemented), `plans/2026-09-03-wayland-driver.md`, and the
two preemption alternatives `designs/2026-08-31-preemption-as-transaction-abort.md` /
`designs/2026-08-31-hardware-sandboxed-kernel-preemption.md` — neither adopted, both gated
on a step actually being observed to overrun.

**Historical — bannered, kept for rationale.** `designs/LATTICE_EVAL.md`,
`designs/lattice-scheduling-types.md`, `designs/REDUCTIONS_AND_FOLDS.md`,
`designs/KERNELS_AND_LATTICES.md`, `designs/BRAINSTORM_VARIANCE_EGRAPH.md`,
`designs/ML_AUTODIFF_PIPELINE.md`, `designs/ML_AND_LINEAR_ALGEBRA.md`,
`designs/compiler-architecture-2026.md`, `designs/2026-07-23-jit-orthodoxy-survey.md`,
`designs/2026-07-23-lower-realize-boundary.md`,
`designs/actor-scheduler-supervisor-migration.md`,
`plans/2025-02-21-kernel-jit-feature-parity{,-design}.md`,
`plans/2026-07-28-jit-performance-parity.md`,
`plans/2026-08-05-egraph-nnue-research-workflow.md` (closed 2026-09-01),
`plans/2026-08-17-cost-model-domain.md`, `plans/2026-08-17-egraph-vsa-nnue-research-notes.md`
(the §5 reading list is the live half), `plans/2026-09-01-dead-code-with-ideas.md`,
`plans/2026-09-07-demand-is-a-dag-property.md`, `plans/archive/`, `archive/`,
`superpowers/`.

### A caution about the four oldest designs

`LATTICE_EVAL.md`, `lattice-scheduling-types.md`, `REDUCTIONS_AND_FOLDS.md` and
`BRAINSTORM_VARIANCE_EGRAPH.md` are April 2026 and state ideas that are still load-bearing
in APIs that no longer exist — `FrameLattice`/`ScanlineLattice`/`PointLattice`, a
per-point `Manifold::eval`, variance as a type parameter, an NNUE or ILP choosing the
schedule. Their banners say which half is which. The thinking is worth reading; none of the
code is worth copying.
