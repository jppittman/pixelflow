# A literature review of PixelFlow's own record

**Date:** 2026-09-09
**Verified against:** `9b9ac41` ("feat(pixelflow-ir): generic arena-backed `Dag` (#1230)", 2026-09-08),
plus `origin/claude/claims-ledger` @ #1207 and `origin/claude/benchmark-correction` @ #1215, which were
unmerged when this was written and are cited as such throughout.

> **Corrected 2026-09-09, and the error is this document's own subject.** The first revision of this
> line read `main @ 8b6e3ce4`. That was the local `main` ref, which was five commits stale; the tree
> actually surveyed was `9b9ac41`, which additionally contains #1206, #1224, #1226, #1229 and #1230.
> Nothing in the survey changes — but a document whose §4 names "no record of *which tree* was
> verified" as the cheapest unfixed failure mode got its own provenance wrong on the first try, by
> trusting a ref name instead of resolving it. Naming a commit is not enough; the commit has to be
> the one you actually read. Recorded rather than quietly amended, per §4.
**Corpus:** 304 files under `docs/` (179 markdown), 57 GitHub issues (8 open), 11 open pull requests.
**Status:** Draft. A survey, not a plan. It proposes nothing and retracts nothing; where it disagrees
with a document it says so and cites the source that settles it.

This document exists because the program has 179 markdown documents and no map of what they
collectively *claim*. `docs/README.md` is an index — it says how to read a document. This is a review —
it says what the documents found, in what order, which findings survive, and where the open questions
are. It sits **above** the claims ledger (`2026-09-07-claims-ledger.md`), which adjudicates individual
numbers, and does not restate its verdicts except where the arc needs them.

**Read depth, stated because a review that pretends to uniform coverage is the failure mode this
repository catalogues.** Read in full: the claims ledger and its CSV, the three 2026-09-08 sweep
documents, `2026-09-07-egraph-off-vs-on-real-shaders.md`, `2026-09-07-benchmark-correction.md`,
`2026-09-07-corpus-structural-gaps.md`, all 8 open issues, and the bodies of 5 of 11 open PRs. Read as
headers and metadata (title, status, date, supersession, first section): all 59 plans and designs. Read
as title and classification only: the remaining ~60 results artifacts and the 23-document test-quality
audit series. §8 marks the depth per row.

---

## 1. The census

| where | files | markdown | what it holds |
|---|---:|---:|---|
| `docs/` (root) | 20 | 15 | landing page, style, two survey/analysis docs, four historical stubs, 3 PDFs |
| `docs/designs/` | 21 | 21 | design docs — language, compiler, actor, runtime |
| `docs/plans/` | 38 | 38 | plans of record, pre-registrations, scoping |
| `docs/plans/archive/` | 2 | 2 | the deleted RL training path |
| `docs/results/` | 181 | 69 | measurements, audits, triage; 112 of these are `.csv`/`.json`/`.jsonl` row data |
| `docs/bugs/` | 26 | 26 | 23 of them one scheduled test-quality audit series |
| `docs/archive/` | 2 | 2 | GNN vision, NNUE curriculum |
| `docs/superpowers/` | 5 | 5 | April 2026 four-team pipeline rewrite |
| `docs/templates/` | 1 | 1 | `DESIGN_DOC.md` |
| **total** | **304** | **179** | |

Two structural facts about the corpus, both grep-detectable and therefore (per CLAUDE.md's own rule)
checks somebody could write rather than caveats somebody keeps attaching:

- **The `DESIGN_DOC.md` status vocabulary is followed by 3 of 59 plans and designs.** `docs/README.md`
  states: "Every future plan and design must use exactly one of `Draft`, `Review`, `Approved`, or
  `Implemented` in its metadata." 31 of 59 carried a `**Status:**` line at all; 3 of those use a
  sanctioned word. The rest have invented ~20 status words in the field — `Plan of record`,
  `REGISTERED`, `Landed`, `Proposed`, `Closed`, `Scoping`, `denotation`, `DESIGN + PRE-REGISTRATION`,
  `Reconnaissance only`, `Ported`, `Superseded`. Several of those are genuinely more informative than
  the sanctioned four, which is the actual finding: **the vocabulary is wrong, not the authors.** A
  pre-registration is not a draft and never becomes "implemented"; a plan of record is not "approved".
  The fix is to widen the vocabulary to what the corpus actually needs and then enforce it, not to
  rewrite 56 headers into words that fit none of them. *(Partly addressed in this change: 52 of 59 now
  carry a status line or a supersession banner. The template's vocabulary itself is still unfixed.)*
- **33 of 59 plans and designs were not linked from `docs/README.md`**, including every
  current-direction document written since 2026-09-06 (`kernel-with-a-lattice`, `lattice-is-the-index`,
  `egraph-at-production-scale`, `one-conditional-three-lowerings`, `exprarena-on-dag`) and the entire
  Phase 3 registration series. The index described the tree as it stood roughly a week ago. It was
  accurate about what it covered, which is the more dangerous kind of stale. *(Fixed in this change:
  the README now carries a complete index and 0 are unlisted.)*
- **17 plans and designs named source files the tree does not have**, 90 mentions across 48 distinct
  dead paths — `FrameLattice`-era types, `ir_bridge.rs` (split into `lower.rs` + `emit.rs` by #1206
  the day before this review), the whole deleted extraction-head module tree. A reader following one
  lands on nothing. *(Fixed in this change: 19 documents gained supersession banners saying which half
  of them is still true, and `scripts/check-doc-paths.sh` now fails CI on a new one.)*

---

## 2. The arc, in seven threads

The program is one question asked at seven altitudes. Roughly chronologically, with the load-bearing
documents named.

### I. The algebra — what a kernel *means*

The oldest and healthiest thread, and the only one whose documents mostly still describe the code.

`2026-07-24-totality-and-the-cost-model.md` is the axiom layer: the kernel language is total, and the
cost model exists because totality forbids the alternative. `designs/KERNELS_AND_LATTICES.md` and
`designs/LATTICE_EVAL.md` establish the representable-functor framing — `index(collapse(f)) = f` — that
lets a buffer *be* a manifold rather than back one.

The last two months are that framing being taken seriously enough to hurt:

- `2026-07-20-kernel-unification.md` (plan of record) — retire the type-level combinator emitter for
  arena-backed `Kernel` values.
- `2026-09-06-kernel-with-a-lattice.md` — three objects, one verb. Records its own course correction:
  the first draft kept a per-batch `Manifold::eval` trait as "the semantics", which is exactly what
  forces a Rust loop around the JIT. `collapse` was the right verb aimed at the wrong type.
- `2026-09-06-lattice-is-the-index.md` (**Landed**, U0/L1–L4 merged) — a lattice is an extent, full
  stop. The origin is deleted; a coordinate frame is a contramap. This is the strongest instance of
  the repo's "subtract before you add" rule: the law held only with a side condition, so the field
  causing the side condition was removed.
- `2026-09-08-one-conditional-three-lowerings.md` (Draft, nothing built) — `Union` is a missing
  compiler capability that leaked into the public API. A rectangle is two selects. Supersedes
  `2026-09-07-demand-is-a-dag-property.md` on framing while keeping its analysis.
- `2026-09-09-exprarena-on-dag.md` (Proposed) — the newest document in the tree. Notable for what it
  does to its predecessor: `dag.rs` shipped with a written reason for not porting `ExprArena`, and this
  doc's §2 is titled "The lifecycle objection was wrong" — the reason described the *signatures* while
  the *usage* was already pure.

**State:** healthy. Direction is stated, stages land, and superseded documents get amended rather than
left to rot (#1198 exists solely to do that). This thread is where the repo's stated method actually
holds.

### II. Codegen — the assembler, the schedule, the ISA

`designs/assembler-as-functor.md` and `2026-07-25-two-level-ir-and-backend-completeness.md` frame
lowering as a functor and audit where the loop binder belongs. `2026-09-01-loop-aware-codegen.md`
(stage 0 landed, #1092) gives the register allocator the lattice.

The thread's importance is disproportionate to its document count, because **it is where the measured
wins are.** Per the claims ledger: S3 shipped the chrome scene at 0.32×; S3b recovered 3.6× via one
select per colour, arm clustering, and a mispredict bound — "none of it an e-node". CLAUDE.md's
"Floating point at the edges" section is the accumulated ISA-divergence knowledge, and it is unusually
good: it states the trade (speed for edge-case conformance), tables what differs per target, and names
the one thing the trade does not license (folding a target-divergent op on the build host).

**State:** the quiet success. Its risk is that it has the least research apparatus pointed at it and
therefore the least protection: issue #1133 (`avx512f+dq` built and linted but never executed) was
exactly this, and closed.

### III. The e-graph — engine and objective

`EGRAPH_OPTIMIZATION_ARCHITECTURE.md` and `SEARCH_PIPELINE_DESIGN.md` are the historical layer (MCTS,
REINFORCE, a transformer critic), both explicitly superseded. What replaced them is engineering, and
2026-09 is when it started producing real answers:

- **The objective was wrong.** Issue #1116: `extract_dag` minimized *tree* cost while the kernel pays
  *DAG* cost — ~95% of the measured extraction gap. Fixed by #1192 (sharing-aware extraction): 55 of
  206 real kernels improved, 0 worsened, chrome's schedule 401 → 385. This is one of exactly **two**
  optimizer improvements in the whole program that the ledger certifies as helping a shipped shader.
- **Budgets are deterministic by construction.** `2026-09-01-production-budget-determinism.md`:
  saturation budgets denominated in rule applications, with wall clock demoted to a fail-loud ceiling
  that panics the build rather than silently truncating. A kernel cannot differ between two machines.
- **More budget makes extraction worse.** `2026-09-08-class-cap-sweep.md`: raising the class cap
  improves Σ `dag_cost` −16.7% on 44 glyphs but costs a 95-glyph warm 6.6 s → 25.5 s, and puts ink on
  the `'8'` waist where FreeType has none. **Not shipped — pinned at the floor, so production is
  byte-identical.** The prerequisite is a tangency fix in the quadratic winding kernel.
- **Why it gets worse, answered.** PR #1236 (extraction witnesses): of 334 classified frontier
  classes, **29% are CYCLE-PRICED — the DP never priced either candidate**, and 24 of 56 first
  divergences were not made by the DP at all (18 by `repair_choices_well_founded`, which has no cost
  model). "The extractor does not get worse at choosing; it gets more places where it does not choose."
  It also undercuts the successor program's own premise: of 56 objective witnesses, 7 are **one** swap
  from greedy's term, **0** are a sequence of swaps away, and 49 are neither — so the
  `Reranker`-over-swap-refinement seam is a neighbourhood holding 7 of 56 answers and no path to the
  rest.
- **Width is not the missing axis.** PR #1238: `Beam` at width 64 on `chrome_packed` returns
  byte-identical machine code to width 1 at a 50,000-class cap. Sixty-four seats over 10,256 live
  classes changed no choice.
- **The e-graph is not monotone in the class cap** (PR #1236): 33 of 139 candidate pairs fail lookup —
  a term minted at 10k is *absent* at 20k/50k/100k. A different cap is a different trajectory, not a
  prefix extension. This quietly invalidates a class of "just raise the budget" reasoning.

**State:** the most productive thread right now, and the one where negative results are landing fastest
and most honestly. Three of this week's PRs exist to say "not this, and here is the measurement".

### IV. The learned model — four architectures, four nulls

The longest arc and, read end to end, the program's central cautionary tale.

1. **Self-play / RL** (Jan–Jul 2026; `plans/archive/2026-02-25-unified-training-*.md`, issues
   #330–#342). Removed July 2026 after a four-agent audit found it methodologically unsound.
   `2026-07-07-guided-saturation-redesign.md` is the post-mortem and the pivot to supervised.
2. **The NNUE extraction head** (Jul–Sep 2026; `2026-08-05-egraph-nnue-research-workflow.md`,
   `NNUE_INTEGRATION_STATUS.md`). Ran to completion and **tied** the static table on schedule-free
   kernels. Shape deleted; denotation kept behind the `Reranker` seam
   (`2026-09-01-schedule-cost-model-denotation.md`). The workshop paper (PR #1072) closed unmerged.
3. **The saturation Guide** (Aug–Sep 2026): `2026-08-31-guide-design-revision.md`, then a dense
   pre-registration series — `phase3-registration`, `round1b-domain-shift`, `round2-registration`
   v1/v2/v3, `round2-rule-scaling`, `guide-return-to-go`, `bilinear-guide-registration`. Nine
   registration documents, roughly 20 result artifacts.
4. **The rules × nodes filter** (Sep 2026): PR #1228 (seam), PR #1240 (the Optuna sweep).

The pre-registration discipline in (3) is real and worth preserving — constants committed before data,
supersession recorded, protocols opened once. **It did not save the results.** Per the ledger, groups D
through F (Guide scoping, Round 1, Round 1b/2/R2G/bilinear) hold 26 rows, of which **5 are HELD**; 9 are
UNITS INVALID because every cost number predates #1192, and the registered constants B=100/200 and
Y=16.3%/9.0% were derived from tree-cost labels and are "port as-is, not re-derivable".

Then two compounding findings, both from this week:

- **The regime was wrong.** The Guide was registered at B=100–200 applications. Production's median at
  the shipped budget is **5,422 applications with 93% stopping on the class cap** — 54× the registered
  budget. Every quality-at-budget claim is about a regime production never enters.
- **The models were never tuned.** JP, 2026-09-08: *"none of my models worked at all until I ran
  optuna. We haven't run optuna once since I stopped following this closely."* PR #1240 swept 250 TPE
  trials over the rules filter. The null survives unmoved — but **the registered configuration ranks
  204th of 250**, below the median trial. Every learned null in the program was reported at a
  self-imposed handicap. The resulting rule (ledger §7 item 11): no learned model's extrinsic number
  is quoted without an Optuna sweep first, with the untuned config enqueued as trial 0.

**State:** four architectures, zero shipped. The honest reading is not "learning does not work here" —
that is exactly the inference an untuned null cannot support — but that **the program has never once
tested a learned model under conditions where a win would have been detectable**: right units, right
regime, right corpus, tuned. It has not yet run the experiment it has spent nine months registering.

### V. The measurement crisis — 2026-09-07 to now

The most consequential week in the record, triggered by one observation (JP, 2026-09-07): *"It's an
egraph for shaders project, and we have an app with shaders and the moment we pointed it at the real
shaders it sucked."*

Four documents, three of them unmerged:

- **`2026-09-07-claims-ledger.md`** (PR #1207) — 91 rows, one per quantitative claim since the
  self-play era. **52 do not stand.** Adjudicated below in §3.
- **`2026-09-07-corpus-structural-gaps.md`** (#1212, merged) — 777 kernels × 90 columns, two
  populations. The synthetic corpus contains **zero** selects, compares, gathers, buffers or uniforms;
  shares at a median ratio of 1.0 against real's 5.2; is 32 nodes against 1,215; and 45% of it names
  coordinate axes the emitter refuses. This is the mechanism behind every collapsed headline, stated
  as counts rather than argued.
- **`2026-09-07-egraph-off-vs-on-real-shaders.md`** (#1210, merged) — saturation on vs off through the
  production path on all 208 shipped kernels, with `cse-only` and `with-select-hoist` arms. The most
  important measurement in the tree, and its headline is uncomfortable: **hash-consing is the product.**
  The zero-rewrite-round arm delivers −41% of glyph bytes, −70% on the cell grid, −77% on psychedelic;
  the rewrite rules add ≈−4% beyond that and on the scenes *raise* bytes slightly while `dag_cost`
  falls. Without the e-graph the chrome scene does not compile at all — a 335,411-node tree whose
  un-saturated schedule overflows the aarch64 branch range.
- **`2026-09-07-benchmark-correction.md`** (PR #1215) — the corrected corpus, the family-aware DEV /
  HELD-OUT split, per-family metrics, and 28 retraction banners. Blocked on #1207 landing first.

**State:** the correction is written and largely unmerged. §7 covers the dependency.

### VI. Runtime, actors, platform

`designs/pty-actor-troupe.md` (implemented), `actor-scheduler-mealy-transducer.md` (the design of
record), `actor-scheduler-backpressure.md`, `actor-scheduler-supervisor-migration.md` (superseded),
`pixelflow-runtime-engine-mesh-migration.md` (not landed). Two 2026-08-31 preemption designs — KVM
sandboxing and transaction-abort — are recorded reasoning that the Mealy design's §5 rules out;
**neither is adopted and both are gated on a step actually being observed to overrun.** That is good
practice worth naming: a design written, costed, and explicitly parked on a trigger.

Live work: `2026-09-03-wayland-driver.md` (a second Linux driver, spending some of `PlatformOps`'
purity deliberately), issue #1204 (migrate two hand-written `*Out`/`*Wiring` pairs to the `ports!`
macro — the macro has tests but no production caller), PR #994 (macOS signed/notarized DMG, parked on
five Apple credentials that do not exist as repo secrets).

**State:** stable, low-drama, under-documented relative to its size. The `ports!` gap in #1204 is the
"trait-first" rule from CLAUDE.md applied in reverse — the generator exists and the two sites that pay
for it still hand-roll the pattern.

### VII. The terminal, and quality process

core-term is the application, and it is the thread with the least research and the most conventional
engineering. `2026-09-07-csi-audit.md` is the exemplar: a requested audit with no bug report that found
three real defects, the largest being that a malformed CSI sequence wrote garbage to the screen rather
than a log line. Fixed and pinned (#1216).

`docs/POSTSUBMIT.md` documents a genuinely strong pipeline: 5× flake detection per (OS, suite), ISA
matrix, benchmark regression against a `gh-pages` baseline, and **automatic revert** on consistent
failure. The `docs/bugs/` directory is 23 documents of one scheduled test-quality audit series running
from 2026-07-20 to 2026-09-07.

**State:** the process apparatus is more mature than most of what it guards. See §5 for the two places
it has a hole.

---

## 3. What the evidence actually says

The claims ledger, recomputed from `2026-09-07-claims-ledger.csv` on branch `claude/claims-ledger`
(91 rows). Counts here are computed from the CSV, not restated from the prose — the ledger's own §1
warns that earlier revisions of its headline were wrong four separate times.

| group | HELD | FAILED | UNITS INVALID | INSTR. DEFECT | NEVER TESTED | total |
|---|---:|---:|---:|---:|---:|---:|
| A. self-play era | 3 | 0 | 0 | 3 | 2 | 8 |
| B. harness audit 2026-08-05 | 2 | 0 | 0 | 0 | 0 | 2 |
| C. extraction head / paper | 5 | 1 | 0 | 6 | 4 | 16 |
| D. Guide scoping 2026-08-30/31 | 0 | 0 | 0 | 1 | 5 | 6 |
| E. Phase 3 Round 1 | 0 | 0 | 3 | 1 | 1 | 5 |
| F. Round 1b / 2 / R2G / bilinear | 5 | 1 | 6 | 1 | 2 | 15 |
| **G. Real-kernel A/Bs (09-01/02)** | **13** | 0 | 6 | 1 | 0 | 20 |
| **H. Lattice programme (real, Sep 6)** | **11** | 1 | 0 | 5 | 0 | 17 |
| I. Frame-level | 0 | 2 | 0 | 0 | 0 | 2 |
| **total** | **39** | **5** | **15** | **18** | **14** | **91** |

By corpus: 31 rows are `real`, 50 are synthetic or structured-synthetic, 2 toy, 8 not applicable.

**The shape of it is unmistakable.** Groups A–F — every learned-model and synthetic-corpus programme,
9 months of work, 52 rows — hold **15 HELD** between them. Groups G and H — the real-kernel A/Bs and
the lattice programme, both from the last three weeks, both measured on shipped kernels — hold **24 of
the 39**. The dividing line is not sophistication, seniority of the idea, or how carefully it was
pre-registered. It is whether the number was taken on a kernel the product compiles.

### What stands

Short enough to state completely:

- **Sharing-aware extraction** (#1192, L067): 55/206 real kernels improved, 0 worse. One of two
  optimizer changes that helps a shipped shader.
- **The S3b schedule win** (L073): 3.6× on chrome from one select per colour, arm clustering, and a
  mispredict bound — codegen, not the e-graph.
- **The cost table is not the lever on glyphs** (L066): a 33× perturbation changes 0 of 190 extracted
  terms. The saturated graph holds no alternative lowering to choose.
- **Saturation is class-cap bound on real kernels** — 93% at the shipped budget, median 5,422
  applications, median 2 iterations (L090, L091) — and the cap binds on the *spliced input*, not on
  rewrite-minted classes.
- **More saturation sometimes extracts worse code** (L055), and on the one clocked real shader 12× more
  classes cost 30× compile and +15% ns/px (L072).
- **Additivity is ISA-conditional** (L083): slope 1.03 where AVX-512 is throughput-bound, wrong by up
  to 1.9× where latency is exposed — and the tile-mode instrument *inverts* against production's
  scanline loop.
- **Budget determinism** (L069): applications, not wall clock; a kernel cannot differ between machines.

### What is conspicuously absent

**No claim that a learned optimizer improvement helps a shipped shader stands anywhere in the record.**
Not one, across four architectures. The nearest is L081 — a C1 guard change that nearly halves row time
on shipped `O`/`S` glyph kernels — and it was never merged, so nothing on `main` carries its number.

---

## 4. Failure modes the record keeps rediscovering

The ledger names five. The 2026-09-08 sweep documents found three more. Unified, with the check that
would catch each — because CLAUDE.md's own rule is that a gap in CI is a check to write, not a caveat
to attach, and **five of these eight have no check.**

| # | mode | canonical instance | check today |
|---|---|---|---|
| 1 | **Units** — a tree cost is not a cost | 15 rows priced pre-#1192; `julia_set` is 1.4e7 tree against 716 DAG | ✅ #1192 changed the objective; issue **#1239** says a reported *column* still disagrees with it |
| 2 | **Regime** — registered budget ≠ production budget | Guide registered at B=100; production median 5,422 (54×) | ❌ none |
| 3 | **Corpus** — synthetic headline, real collapse | rule order 86× → 0.97× and *reversed* on psychedelic | ⚠️ `corpus_gaps` measures the gap; nothing gates on it |
| 4 | **Instrument** — the instrument is a claim too | 41.67× then 4× timebase; null context so `Gather` was never priced; tile mode inverting against scanline | ❌ none |
| 5 | **Provenance** — no document records *which tree* it was verified against | four stale rows in one sweep, two invalidated by that sweep's own merges | ❌ none — and it is one grep |
| 6 | **Oracle** — same-form cannot see a shared-definition bug | the `sin` range-reduction bug: JIT and `eval_scalar` ran the same expansion and agreed bit-for-bit on garbage | ⚠️ range bounds exist (`trig_range.rs`); cross-form gates cover ~⅔ of rules (**#1112**) |
| 7 | **Untuned** — a null from an untuned model is weak evidence | registered filter config ranks 204/250 | ✅ new rule, ledger §7.11 — first applied in #1240 |
| 8 | **Masking** — a timeout plus fail-fast conceal the suite | #1213: 2 tests timed out, **45% of the suite never ran** and a regression there would be invisible | ❌ none |

**Mode 5 is the cheapest and highest-leverage unfixed one.** Every stale row found in the 2026-09-08
sweep — the audit backlog, L057's provenance, three ledger rows, the sweep's own triage table — would
have announced itself with one field. `2026-09-08-open-pr-triage.md` says so in its own last section,
and then the document immediately below it went stale within the hour for the same reason. Two
documents diagnosing a failure mode exhibited it while doing so. That is not carelessness; it is
evidence the fix is not diligence.

**Modes 2 and 4 have no check and are the two that invalidated the most rows** (9 UNITS INVALID from
the pre-#1192 objective plus 18 INSTRUMENT DEFECT = 33 of 52 non-standing rows trace to units,
instrument, or regime).

One meta-observation worth separating out. The 2026-09-08 sweep found that nine of twelve open review
threads on #1207/#1215 were **a document disagreeing with itself or with a file already in the tree** —
a verdict against its own definition, a 5% decision threshold sitting below the document's own stated
10% noise floor, an acceptance rule that does not test the gap its own row states. None needed a
measurement to settle. That is a different class from modes 1–8: not bad evidence, but internally
inconsistent documents, and it is what happens when documents are long, pre-committed, and revised in
place under time pressure.

---

## 5. Where CI is the gate, and the two holes in it

The repository's stated doctrine is strong — green CI is permission to submit, a gap in CI is a check
to write, and when a bug ships green the retrospective is about the gate rather than the author. The
apparatus backing it is real: presubmit workspace tests, Clippy, rustfmt, feature matrix, ISA matrix,
behavior contracts, four metadata jobs, and a postsubmit pipeline with 5× flake detection and automatic
revert. The doctrine visibly works: `scripts/check-bin-declarations.sh` (~6 s, no toolchain) exists
because one missing `[[bin]]` entry turned `main` red, and the response was a check rather than a note.

Two holes are documented and open:

- **Issue #1193 — no presubmit compiles a production scene or a glyph warm.**
  `2026-09-06-egraph-at-production-scale.md` §5.3 states the compile budget (a scene under ~250 ms; a
  glyph warm is 95 kernels per font size, so +5 ms each is +0.5 s of startup) and **nothing measures
  either**. A change that pushes chrome past its budget, or adds 5 ms to every glyph, merges green.
  This is the single most load-bearing missing check in the tree: it is the gate that would have caught
  the class-cap sweep's 6.6 s → 25.5 s warm automatically, and it is where the benchmark correction's
  §B.5 deterministic-column diff wants to live.
- **Issue #1112 — roughly a third of rewrite rules can never fire under the numeric oracle.** No test
  expression anywhere contains `Pow`, `Ln`, `Log2`, `Exp2`, `Tan`, `Asin`, `Atan`, `Min` or `Max`, so
  all 11 `power_rules`, the exponential rules and the parity rules are validated by review only. L1
  (every rule preserves denotation) is the law the whole optimizer rests on — L4, Guide neutrality, is
  proved *from* it. It is blocked on **#1098** (`Pow`/`Log` return finite garbage outside their domain
  instead of NaN), which is a genuine dependency and not an excuse.

A third, undocumented as an issue: the masking failure of mode 8. #1213's timeout hid 45% of the suite
and nothing detects that shape. The fix that PR #1235 took is the right one — make the glyph pipeline
faster (three O(n²) loops: 86 s → 1.15 s on legalize, 571 s → 187 s on `loop_blinn_winding`) rather than
raise the limit — but the *class* of failure has no detector.

---

## 6. The issue tracker, read as a corpus

57 issues, 8 open. The distribution is itself a finding:

| category | count | note |
|---|---:|---|
| machine-filed postsubmit failures / automatic reverts | 19 | all closed |
| machine-filed performance regressions | 4 | all closed |
| machine-filed flaky-test reports | 3 | all closed |
| Jan 2026 MCTS/NNUE/WASM task issues | 11 | all closed with their programme |
| substantive engineering, closed | 12 | incl. #1116, #1111, #1105 — the extraction-objective trio |
| **substantive engineering, open** | **8** | below |

**26 of 49 closed issues (53%) were filed by automation.** The tracker is not primarily a human backlog;
it is mostly the postsubmit pipeline's output. Human findings live in `docs/` — which is why a review
of "our docs and issues" is really a review of the docs, with the issues as a thin, current top layer.

The 8 open issues, with what each blocks:

| # | title | thread | blocks |
|---|---|---|---|
| **#1239** | cap sweep's `dag_cost` column is not the objective the extractor minimized | e-graph | the class-cap sweep's per-family signs. Found while landing #1238; **cheap** — rows re-derivable from committed dumps without re-saturating |
| **#1193** | no presubmit compiles a production scene or a glyph warm | CI | the benchmark correction's §B.5 gate; every compile-budget claim |
| **#1112** | ⅓ of rewrite rules can never fire under the numeric oracle | correctness | L1 enforcement; the queued 33-rule batch would land under the same gap |
| **#1098** | Pow/Log return finite garbage outside their domain | correctness | #1112. Labeled `jules` |
| **#1106** | e-graph rebuild does no upward congruence closure | e-graph | nothing — correctly scoped as *measure first*: it under-merges, which L2 permits, so the ask is a count of missed merges before the parent-list machinery is bought |
| **#1099** | e-matching cost grows 2.2–3.2× per application as the rule set grows | e-graph | the "much larger rule set" roadmap. No index; every added rule costs a scan whether or not it can fire |
| **#1104** | Phase 3 at-budget comparison is asymmetric between arms | Guide | any re-run of the Phase 3 registration — the fix is an amended registration, not a review round |
| **#1204** | migrate `coordinator_node.rs`/`vsync_actor.rs` to the `ports!` macro | runtime | nothing. Labeled `jules` |

Two are correctness (#1098 → #1112), three are e-graph engineering, one is CI, one is a research
protocol defect, one is a cleanup. **None is a research question** — the research questions all live in
plan documents and open PRs, not in the tracker.

---

## 7. What a TPM would put on the board

The genuine dependency structure, which no single document currently states.

**The critical path is documentary, not technical.**

```
#1207 (claims ledger)  ──must land first──▶  #1215 (benchmark correction, 28 retraction banners)
     │                                              │
     │ 8 open threads, all research verdicts        │ 10 open threads
     │ each has a written recommendation in         │ #1215's own thread says
     │ 2026-09-08-open-thread-decisions.md          │ "#1207 first"
     │ awaiting one accept/reject pass              │
     ▼                                              ▼
  every downstream doc's retraction banner ────▶ the corrected corpus is the
  points at #1207; without it they dangle        precondition for re-taking
                                                 anything in §5 of the ledger
```

Both PRs are green, rebased and unconflicted. Neither is blocked on work. **They are blocked on
decisions**, and the decisions are already staffed: `2026-09-08-open-thread-decisions.md` carries a
recommendation with evidence for each of the twelve original threads plus six later ones, explicitly so
the call is one pass rather than a re-derivation. This is the highest-value hour available anywhere in
the program, and it needs a human, not an agent.

**Second-order consequence, worth stating plainly:** until #1207 lands, 28 documents in `docs/` state
retracted numbers with no banner, and three of them are actively being read as live findings —
`guided-saturation-redesign.md` still presents ρ ≈ 0.35, `phase3-round2-registration-v3.md` still
carries a "Production quick-win" at B=100, `2026-09-02-missing-congruence.md` still opens on the retired
68.4% regime as its premise. New readers (human or agent) will keep picking those up.

**The other four live fronts, in dependency order:**

1. **#1239 before the cap sweep is cited again.** The sweep's headline (extraction gets worse as the
   cap rises) may well survive, but its per-kernel signs are in a column that is not the objective the
   extractor minimized. Cheap to fix and it gates re-reading a document already in the tree.
2. **#1235 (G1: a glyph is two folds over one table)** — the largest live PR (85 files), currently
   `dirty` and draft. It carries the fix for the timeout that killed #1213, and its measured wins are
   real and not about the e-graph at all: legalize 86 s → 1.15 s, glyph kernel build 12 ms → ~1 ms flat
   in piece count. It also deletes `Union` (−1,291 lines), which is the first repayment of the
   one-conditional-three-lowerings thesis. **Highest-value technical work in flight.**
3. **#1236 / #1238 (extraction witnesses, the Extractor seam)** — both say "not this" with
   measurements, and both are unblocked. #1236 is the more consequential: it kills the reranker-over-
   swaps seam that `2026-09-01-schedule-cost-model-denotation.md` names as the successor program.
   **That plan needs an amendment.**
4. **#1193** — the compile-budget gate. Write it before the next cap or extraction change, or the next
   one lands the same way this one did: measured by hand, on one host, at unknown load.

**Parked correctly and needing nothing:** #994 (macOS release, blocked on credentials that do not
exist), the two preemption designs (gated on an overrun nobody has observed), #1106 (correctly framed
as measure-before-build).

**The strategic question the board does not answer.** Groups A–F spent nine months and produced 15
standing rows. Groups G–H spent three weeks and produced 24. The corrected benchmark exists precisely
so the learned-model programme can be re-run under conditions where a win would be visible — and §5 of
the ledger orders that re-take **fifth**, after F, rule order, class-cap, and the latency prior, with
the expectation stated as "Expected null; a null closes the question honestly." That ordering is
defensible and the honesty is admirable. It is also worth saying out loud that the program has now
built a great deal of apparatus for asking a question it increasingly expects to answer "no", while
the two things that demonstrably moved a shipped kernel — sharing-aware extraction and the S3b
schedule — were both plain engineering found by looking at what the machine actually pays.

---

## 8. Annotated bibliography

Legend for read depth: **F** = read in full for this review · **H** = header, status and first section ·
**T** = title and classification only.

### Plans (38 + 2 archived)

| document | status as written | depth | one line |
|---|---|:-:|---|
| `2025-02-21-kernel-jit-feature-parity-design.md` | Approved | H | `kernel_jit!` parity; param-baking scope done, dual-backend framing superseded |
| `2025-02-21-kernel-jit-feature-parity.md` | — | T | implementation plan for the above |
| `2026-07-07-guided-saturation-redesign.md` | Plan | H | the RL post-mortem and the pivot to supervised guidance; still carries a retracted ρ ≈ 0.35 |
| `2026-07-20-kernel-unification.md` | Plan of record | H | retire the combinator emitter for arena-backed `Kernel` values |
| `2026-07-28-jit-performance-parity.md` | Scoping | T | pre-implementation scoping; "Surface" lane superseded by kernel-with-a-lattice |
| `2026-08-02-ir-layering.md` | Proposed | H | subtract then decide; complements kernel-unification P6/P7+ |
| `2026-08-05-egraph-nnue-research-workflow.md` | **Closed 2026-09-01** | H | the extraction-head workflow; ran to completion and found a tie |
| `2026-08-08-egraph-constant-domain-spike.md` | Reconnaissance only | T | `ENode` constant representation for the dyadic-exact fold |
| `2026-08-17-cost-model-domain.md` | — | T | cost-model domain model and reorganization |
| `2026-08-17-egraph-vsa-nnue-research-notes.md` | Research notes | T | external literature survey; carries the archived GNN vision's offline-teacher framing |
| `2026-08-31-guide-design-revision.md` | Design revision | H | measured economics + the pre-registered Phase 3 experiment; the Guide's design of record |
| `2026-09-01-dead-code-with-ideas.md` | — | H | KEEP / REUSE-R2 / SEGREGATE / DELETE audit with `file:line` evidence per row |
| `2026-09-01-guide-candidate-context.md` | DESIGN (approved) | T | context cells, coverage table, rule-conditioned generation |
| `2026-09-01-guide-return-to-go.md` | DESIGN + PRE-REGISTRATION | T | hindsight return as training target; counterfactual replay as credit validation |
| `2026-09-01-loop-aware-codegen.md` | design; stage 0 landed | H | give the register allocator the lattice (#1092) |
| `2026-09-01-phase3-registration.md` | REGISTERED | H | budget tiers, improvement threshold, gates — the parent registration; **#1104 against it** |
| `2026-09-01-phase3-round1b-domain-shift-registration.md` | REGISTERED | T | does the Guide's advantage survive shift toward trig-dominant kernels |
| `2026-09-01-phase3-round2-registration.md` | REGISTERED | T | regret at budget vs rule count (v1) |
| `2026-09-01-phase3-round2-registration-v2.md` | REGISTERED | T | supersedes v1 — inflated rules |
| `2026-09-01-phase3-round2-registration-v3.md` | REGISTERED | T | supersedes v2's H1 reading; `\|R\|` effect and order effect separated. **Carries a retracted B=100 quick-win** |
| `2026-09-01-phase3-round2-rule-scaling.md` | DESIGN | T | does the Guide's advantage grow with rule count |
| `2026-09-01-production-budget-determinism.md` | — | H | budgets in rule applications; wall clock demoted to a fail-loud ceiling. Carries the class-cap block |
| `2026-09-01-register-allocation-escape-hatches.md` | — | T | register allocation outside the register allocator |
| `2026-09-01-schedule-cost-model-denotation.md` | denotation | H | the successor cost model: analytic table + learned residual. **§ premise refuted by #1236** |
| `2026-09-02-bilinear-guide-registration.md` | REGISTERED | T | does a rule-by-context interaction buy a domain-conditional advantage |
| `2026-09-02-optimizer-api.md` | design | H | `Optimizer`/`RuleSet`/`Budget` + the five denotational laws; source of **#1106** and **#1112** |
| `2026-09-02-phase3-forward-port.md` | Ported | T | inventory and map of the Phase 3 forward-port |
| `2026-09-03-wayland-driver.md` | — | H | a second Linux driver; spends some of `PlatformOps`' purity deliberately and says the price |
| `2026-09-04-ir-as-a-trait.md` | — | T | naming the term language the e-graph speaks |
| `2026-09-06-egraph-at-production-scale.md` | measured facts, not a plan | H | the chrome scene as the case to optimize around; §5.3 is the compile budget **#1193** guards |
| `2026-09-06-kernel-with-a-lattice.md` | plan of record | H | three objects, one verb; records its own course correction on `collapse` |
| `2026-09-06-lattice-is-the-index.md` | **Landed** (U0, L1–L4) | H | a lattice is an extent; the origin is deleted, a frame is a contramap |
| `2026-09-06-uniform-slot-identity.md` | — | T | a scalar parameter that is invariant without being known |
| `2026-09-07-demand-is-a-dag-property.md` | **Superseded** | H | analysis survives; framing replaced by one-conditional-three-lowerings |
| `2026-09-08-egraph-cpu-memory-profile.md` | measured facts | H | where the engine's cycles and bytes go — the engineering half's sequel to production-scale |
| `2026-09-08-macro-tier-is-arena-native.md` | — | T | the macro tier optimizes on the arena, not the AST |
| `2026-09-08-one-conditional-three-lowerings.md` | **Draft, nothing built** | H | `Union` is a missing compiler capability that leaked into the public API |
| `2026-09-09-exprarena-on-dag.md` | Proposed | H | staging the port; §2 records that the predecessor's stated blocker was wrong |
| `archive/2026-02-25-unified-training-design.md` | — | T | temporal credit assignment via sequence transformer (deleted path) |
| `archive/2026-02-25-unified-training-plan.md` | — | T | implementation plan for the above |

### Designs (21)

| document | depth | one line |
|---|:-:|---|
| `2026-07-23-jit-orthodoxy-survey.md` | T | is the PixelFlow JIT orthodox? against V8, LuaJIT, HotSpot, Halide-class |
| `2026-07-23-lower-realize-boundary.md` | H | superseded by the JIT-first `Kernel` course correction |
| `2026-07-24-totality-and-the-cost-model.md` | H | **design of record, axiom layer** — totality is the root axiom |
| `2026-07-25-two-level-ir-and-backend-completeness.md` | H | lowering/backend boundary audit + the first Instruction/assembler split |
| `2026-08-31-hardware-sandboxed-kernel-preemption.md` | T | KVM sandboxing — recorded reasoning, **not adopted**, gated on an observed overrun |
| `2026-08-31-preemption-as-transaction-abort.md` | T | the alternative — same status |
| `BRAINSTORM_VARIANCE_EGRAPH.md` | T | variance analysis via saturation — exploratory |
| `KERNELS_AND_LATTICES.md` | H | **current architecture** — the implemented kernel/lattice substrate |
| `LATTICE_EVAL.md` | T | lattice as representable functor; unified scheduling |
| `ML_AND_LINEAR_ALGEBRA.md` | T | functional ML and linear algebra — directional |
| `ML_AUTODIFF_PIPELINE.md` | T | denotational ML and symbolic autodiff — directional |
| `REDUCTIONS_AND_FOLDS.md` | T | reductions, folds, dimension collapse — directional |
| `actor-scheduler-backpressure.md` | T | backpressure is a protocol, not a politeness |
| `actor-scheduler-mealy-transducer.md` | T | **the actor design of record**; its §5 rules out mid-step preemption |
| `actor-scheduler-supervisor-migration.md` | T | explicitly superseded by the Mealy design |
| `assembler-as-functor.md` | T | formalizing the codegen pipeline |
| `compiler-architecture-2026.md` | T | point-in-time IR unification proposal; verify before use |
| `lattice-scheduling-types.md` | T | extraction as factoring |
| `opkind-numbering-is-private.md` | T | the op numbering is private; `marshal` is how you get bytes |
| `pixelflow-runtime-engine-mesh-migration.md` | T | mediator → mesh; **not landed** |
| `pty-actor-troupe.md` | T | **implemented** PTY actor wiring |

### Root documents (15)

`README.md` (the index — see §1 on its coverage) · `STYLE.md` (current conventions) ·
`POSTSUBMIT.md` (the three postsubmit workflows and the automatic-revert policy) ·
`AUTODIFF_RENDERING.md` (the thesis: forward-mode AD for real-time CPU graphics is an unexplored gap;
positions the project against Elliott 2009 and IQ's finite-difference shader idiom) ·
`COMPILER_ANALYSIS.md` (2026-01-30 pipeline analysis, 749 lines) ·
`COMPILER_OPPORTUNITIES.md` (ranked compiler improvements) ·
`KERNEL_PARAM_LIMIT_INVESTIGATION.md` (>3 params failed under the old trait-bound scheme; root cause
identified — obsoleted by the arena architecture) ·
`FLAT_CONTEXT_TUPLE_PROTOTYPE.md` (nested-`Let` trait-bound explosion; same era) ·
`MESSAGE_CUJ_COVERAGE.md` (actor message CUJ test strategy) ·
`function-namespace-audit.md` (functions whose name prefix is really a namespace — the source of
CLAUDE.md's "name vs namespace" rule) ·
`lample_charton_2019_symbolic_math.md` + `2425_deep_learning_for_symbolic_mat.pdf` (external paper
notes) · `fop-conal.pdf`, `type-class-morphisms-long.pdf` (Elliott, the denotational-design lineage) ·
**historical stubs:** `NNUE_INTEGRATION_STATUS.md`, `EGRAPH_SEARCH_INTEGRATION.md`,
`SEARCH_PIPELINE_DESIGN.md`, `EGRAPH_OPTIMIZATION_ARCHITECTURE.md` — all four carry dated supersession
banners and point forward correctly.

### Results (69 markdown + 112 row-data files), by campaign

- **Extraction and cost model** — `2026-07-08-extraction-3way`, `-rule-report`, `2026-09-02-extraction-gap`,
  `-extraction-objective`, `2026-09-06-extraction-objective-rebase`, `2026-09-06-horner-vs-estrin`,
  `2026-09-08-class-cap-sweep`. The arc from "the objective is wrong" to "the objective is right and
  budget still hurts".
- **Guide / Phase 3** (~30 files) — `2026-08-30-guide-headroom`, `-guide-scope-saturation-delta`,
  `-oracle-filtered-budget-curves`; `2026-09-01-phase3-*`, `-r2g-*`, `-round2-*`, `-train-guide-*`,
  `-counterfactual-credit*`, `-control-guide-comparison`, `-tightened-labeler-rank`,
  `-strict-label-*`; `2026-09-02-bilinear-guide*`, `-guided-regression-bisect`. **Most of this campaign
  is UNITS INVALID or NEVER TESTED ON REAL** per §3; read the ledger row before citing any of it.
- **Real-kernel measurement (the correction)** — `2026-09-07-corpus-structural-gaps`,
  `-egraph-off-vs-on-real-shaders` (+ 8 row files), `-claims-ledger` (**unmerged, #1207**),
  `2026-09-01/02-rule-order-real-kernels`, `-missing-congruence`, `-rebuild-writeback-orphan`,
  `-production-saturation-telemetry`.
- **Instrument audits** — `2026-08-05-bench-harness-integrity-audit`, `2026-07-20-jit-compile-cost`,
  `2026-09-02-gradcheck-isa-postmortem`, `2026-09-01-integration-audit`,
  `2026-09-02-phase3-instrument-changes`.
- **Process / triage** — `2026-09-01-open-pr-triage`, `-open-pr-sweep-followup`,
  `2026-09-08-open-pr-triage`, `-open-thread-decisions`, `-pr-sweep-close`. The three 2026-09-08
  documents are the best worked example of the provenance failure mode in the tree, including on
  themselves.
- **Application** — `2026-09-07-csi-audit` (three real terminal defects, fixed and pinned).
- `journal.jsonl` — the structured event log. **Git LFS; a session without LFS push access cannot
  append to it**, which is why three recent documents landed without a journal entry and said so.

### Bugs (26)

23 are one scheduled test-quality audit series (2026-07-20 → 2026-09-07), each pass taking named files
and reporting STYLE.md naming compliance plus `cargo mutants` gaps. Two passes are unmerged or
retro-annotated (#1054 salvaged, #1154 repaired), and both carry dated supersession banners saying a
full mutation re-run is still owed before "0 real gaps" is restated. The other three:
`2026-07-15-pty-fork-malloc-deadlock` (**active** — diagnosed, not fixed),
`2026-07-21-openpty-not-thread-safe` (fixed), `2026-07-22-trig-chebyshev-coefficients-wrong` (fixed).

### Archive, superpowers, templates (8)

`archive/GNN_REWRITE_GUIDANCE_VISION.md` and `archive/nnue-training-pipeline.md` — the two pre-2026-07
learned-guidance visions. `superpowers/` — the April 2026 four-team pipeline rewrite (lattice
extensions, IR pullbacks, forward pass, backward training) plus its spec; completed or superseded.
`templates/DESIGN_DOC.md` — the status vocabulary §1 finds is followed by 3 of 59 documents.

---

## 9. Standing recommendations

Five, ordered by value per hour. None requires a measurement.

1. **Decide the 18 open threads on #1207 and #1215 and land both.** Recommendations with evidence are
   already written in `2026-09-08-open-thread-decisions.md`. Everything downstream — 28 retraction
   banners, the corrected corpus, every re-take in ledger §5 — is behind this one pass, and three
   documents are currently being read as live that should not be.
2. **Add a `Verified against:` field to the design-doc template and grep for it.** Failure mode 5 costs
   more than any other unfixed one, it recurred four times in a single sweep, and the check is one
   script in the metadata jobs (which already run in ~6 s with no toolchain — `check-bin-declarations.sh`
   is the precedent). *(Half done: the field is now required by `templates/DESIGN_DOC.md`, which also
   says to resolve the sha rather than name a ref — the mistake this document itself made. Nothing
   enforces it yet; that is the remaining half.)*
3. ~~**Fix the status vocabulary rather than the 56 documents.**~~ **Done.**
   `templates/DESIGN_DOC.md` now defines `Draft / Proposed / Registered / Plan of record / Landed /
   Closed / Superseded / Historical`, and `docs/README.md` states the same list and indexes all 59
   documents. `Registered` is the one worth naming: a pre-registration commits its constants before the
   data exists and is never revisable afterwards, which none of the old four words could express. Still
   unenforced by a job.
4. **Write #1193.** It is the gate that makes the benchmark correction's §B.5 enforceable and the one
   that would have caught the class-cap warm regression without a human noticing.
5. **Amend `2026-09-01-schedule-cost-model-denotation.md`.** Its successor program is a reranker over
   swap refinement; #1236 measured that neighbourhood and found 7 of 56 objective witnesses one swap
   inside it and **none** reachable by any longer sequence. The seam can stay; the claim that it is the
   next thing to build should not, unretracted, be the most recent word on the subject.

---

**Provenance.** Written 2026-09-09 against `main` @ `8b6e3ce4`. Claim counts are computed from
`2026-09-07-claims-ledger.csv` on `origin/claude/claims-ledger`, not restated from prose; corpus counts
are `find`/`grep` over `docs/` at that commit; issue and PR state is the GitHub API at the time of
writing. Where this document and a live source disagree, the source is right — and per its own §4 row 5,
that sentence is only useful because the commit is named above it.
