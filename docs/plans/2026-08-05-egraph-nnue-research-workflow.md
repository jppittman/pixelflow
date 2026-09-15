> **Closed 2026-09-01 — the workflow ran and its result was a tie.** Kept as the
> record of how the extraction-head program was run, not as work to do. The paper
> it targets closed unmerged (PR #1072, branch `claude/workshop-writeup`, not in
> this tree) and the architecture was deleted; its denotation survives behind the
> `Reranker` seam in
> [`2026-09-01-schedule-cost-model-denotation.md`](2026-09-01-schedule-cost-model-denotation.md).
> Note that #1236 has since measured that seam's neighbourhood and found the answer
> mostly outside it. `bench_extraction_3way.rs` and `bootstrap_extraction_head.rs`
> are deleted.

# E-graph NNUE research workflow: iterate to a publishable extraction-head result

**Status:** Closed, 2026-09-01. The workflow below ran to completion: the workshop paper it
targets (branch `claude/workshop-writeup`, PR #1072, closed without merging — not in this tree)
found the NNUE extraction head ties `CostModel::latency_prior()` rather than beating it. JP's
ruling ("delete the shape, keep the denotation") removed the extraction-head code this plan
built toward — `bootstrap_extraction_head.rs` and `bench_extraction_3way.rs`, linked below, no
longer exist. See
[`2026-09-01-schedule-cost-model-denotation.md`](2026-09-01-schedule-cost-model-denotation.md)
for the outcome, citations, and what the closure kept as a seam, and
[`../NNUE_INTEGRATION_STATUS.md`](../NNUE_INTEGRATION_STATUS.md) for the current-state summary.
The rest of this file is left as it was written, as the record of the workflow that was run.

**Date:** 2026-08-05
**Target:** workshop/preprint first (EGRAPHS @ PLDI, or NeurIPS/ICML ML-for-Systems workshop; arXiv preprint alongside).
**Primary claim:** the NNUE extraction head produces measurably faster extracted kernels than
`CostModel::latency_prior()` on held-out expressions, at acceptable extraction overhead.
**Prior result being challenged:** 2026-07-08 3-way bench — NNUE lost, 6.7% slower geomean at
~31× extraction cost (recorded in `docs/results/2026-07-08-extraction-3way.md`, restored after
the canonical copy was deleted by `dc726864`). That run predates the 41.67× timebase fix, used ~4,000 training
samples, and had no train/test split. The deficit is plausibly recoverable; the workflow below is
designed to either recover it or produce an honest, publishable negative + architecture paper.

This plan synthesizes three surveys (2026-08-05): the autonomous-research frontier
(AlphaEvolve/ShinkaEvolve/MLE-STAR/AIRA/DGM design lessons), the learned-cost-model and
guided-eqsat literature, and a code-level map of the current pipeline. Citations live in the
survey reports; the load-bearing lessons are inlined here as **[L#]** tags matching the frontier
report's numbering.

---

## 0. North star: the ML is written in the language it optimizes

**The eventual goal is that every ML component in this repo — the extraction head, the
saturation head, the accumulators, the training arithmetic — is written in the pixelflow
kernel language itself, not in hand-written Rust that happens to sit beside it.** The optimizer
algebra becomes an expression in the algebra it optimizes.

This is the same move a compiler makes when it becomes self-hosting, and it is the endpoint
that the rest of this plan is walking toward. Nothing in Phase 0 or Phase 1 depends on it; it
is the direction of travel that decides which local choices are right.

**Why it is the right target, not just a cute one:**

- **The cost model's own cost becomes a fixed point.** A cost model expressed as a kernel is an
  expression the e-graph can optimize, using the cost model. The thing that predicts "how long
  will this kernel take" is itself a kernel whose time it predicts. That closes a loop no
  learned-cost-model paper in the 2026 survey closes: the model can be optimized by its own
  extraction, and — more sharply — the accuracy of the model on *itself* is a directly
  measurable, non-circular quantity (predict its own ns; measure its own ns).
- **NNUE inference is already pull-based.** Sampling an evaluation is exactly `Manifold`
  semantics: nothing computes until coordinates arrive. The accumulator's incremental
  add/remove is a coordinate warp over a sparse feature lattice, and a matmul is a `Reduce`
  over a `Gather`. The architecture was chosen for game engines because it is
  add-a-column/subtract-a-column arithmetic on a huge sparse first layer — which is the same
  shape the engine is built to fuse. The impedance mismatch is small on purpose.
- **We stop maintaining two numeric stacks.** Today the forward pass is hand-written Rust
  arrays and the kernels are `kernel!`; the SIMD backend, the JIT, the ISA matrix, the
  constant folder, and the e-graph all serve only the second one. Written as kernels, the ML
  inherits every one of them for free — and inherits future backends automatically, which is
  the whole reason [[kernel-language-unity]] treats parity as the goal.
- **It is a forcing function on the language.** Whatever the ML cannot express is a real gap in
  the algebra. Expressing an NNUE demands exactly the primitives a serious array language owes
  (`Reduce`, `Gather`, binding tables, batched contraction) and will find the places where the
  eDSL is a shader language pretending to be a tensor language. Per
  [[prefer-kernel-macro]], those gaps are bugs to be filed, not reasons to retreat to Rust.
- **Training, eventually, is the same story.** The backward pass is a `Dwrt` away — the e-graph
  already carries differentiation as a rewrite, and `Jet2` dual numbers already do
  autodiff for antialiasing. A gradient step is a kernel. That is the last piece, and the
  least urgent.

**Sequencing (deliberately not now).** Inference before training; extraction head before
saturation head; a kernel-expressed forward pass validated bit-for-bit against the current Rust
implementation before the Rust one is deleted. The current supervised loop must produce a
trustworthy result *first* — a self-hosted cost model that is wrong is harder to debug than a
Rust one that is wrong, because the failure can now be in the optimizer, in the model, or in
the interaction. Phase 1's ablation table gets a row for it when the loop is stable, not
before.

**For the paper.** If the extraction head is expressed in the language it costs, that is a
second contribution beyond the incremental-accumulator one in §1, and a more distinctive one:
not "we learned a cost model" but "the cost model is a program in the object language, so
optimizing it and applying it are the same operation." Worth stating as future work even in
the workshop version.

---

## 1. What the literature says we have (and don't)

**Unclaimed territory (per the 2026 survey — no published occupant found):**
1. NNUE-style *incrementally updatable* cost evaluation inside e-graph search. Every published
   guide re-encodes state per step (GNN forward per action) or amortizes on GPU (SmoothE).
   An O(Δ) accumulator evaluated inside a proc-macro is an unclaimed architectural contribution.
2. VSA (binding/shift) graph featurization for compiler ML. Prior art is uniformly
   message-passing GNNs or transformers.
3. Hindsight provenance labels (union journal → load-bearing unions → supervised targets) as a
   supervision source that retires RL for eqsat guidance. Closest relative: EggMind (2026,
   proof-derived motif caching for an LLM agent) — must cite and differentiate.

**Not novel — do not claim:** learned cost models for extraction per se (SmoothE, e-boost),
RL-guided rule application per se (MCTS-GEB, Omelette, PACT'24), measured-runtime corpus +
regression head (Halide 2019 is the template; cite it as such), rule phasing (Isaria).

**The 2026 evaluation protocol** a paper in this space must satisfy (details in §5):
intrinsic model quality (MAPE, Spearman ρ, top-k regret) + end-to-end geomean *with
per-benchmark distribution* + quality-vs-budget curves + noise floor stated relative to model
error + ablations + by-family (not random) held-out splits + censoring reported. For the
incremental-model claim specifically: incremental == full-recompute equivalence test and
evals/sec comparison.

---

## 2. Loop architecture

We are in AlphaEvolve's problem class — the objective (JIT-measured ns) is automatically
verifiable — so the template is the evolutionary/archive loop, not the paper-factory pipeline.
The consensus design, adapted:

```
             ┌────────────────────────────────────────────────────────┐
             │                    RESEARCH JOURNAL                    │
             │   append-only: config hash → all metrics, per round    │
             └────────────────────────────────────────────────────────┘
                    ▲                                        │
                    │ record                                 │ never re-run a config [L8]
                    │                                        ▼
  PROPOSE ──► TRAIN ──► EVAL-INTRINSIC ──► EVAL-E2E ──► SELECT ──► ADVERSARIAL MINE ─┐
  (one change   (bootstrap_   (held-out MAE,   (3-way bench    (archive, not  (worst |pred−meas|│
   per round,    extraction_   Spearman,        on frozen       hill-climb     exprs → corpus)  │
   ablation-     head, fast)   top-k regret)    holdout:        [L7])          [L3]             │
   targeted                                     geomean vs                                      │
   [L5])                                        prior +                                         │
                                                overhead)                                       │
       ▲                                                                                        │
       └────────────────────────────────────────────────────────────────────────────────────────┘

  HUMAN GATE [L10]: JP accepts a weights checkpoint only on e2e geomean improvement on the
  never-trained-on holdout. The FINAL eval set (§4.3) is touched only for publication claims.
```

Key departures from a naive train-eval loop, each traceable to a documented failure mode:

- **One change per round, chosen by ablation** [L5, MLE-STAR]: each round begins by ablating the
  candidate dimensions (accumulator sections, corpus mix, loss, top_k, feature set) on the
  *previous* round's checkpoint and spending the round's effort only on the dimension that
  carries signal. This is also the leak detector — the `node_count` leak class of bug surfaces
  as an ablation anomaly.
- **Adversarial mining** [L3, AIRA generalization gap]: the extractor searches for minimum
  *predicted* cost, so it concentrates probability mass exactly where the model is
  wrong-and-optimistic. After each round, mine the expressions with worst |predicted − measured|
  (and expressions where the NNUE-extracted variant measured slower than the prior-extracted
  one), benchmark them, and append to the training corpus. This is the cheapest known defense
  against a search consuming its own learned proxy.
- **Novelty rejection before benchmarking** [L6, ShinkaEvolve]: JIT-bench is the expensive step
  (~10 min compile time per 50k-sample run before measurement). Refuse to benchmark an
  expression whose EdgeAccumulator embedding is a near-duplicate of one already in the corpus;
  static screen (compiles, node count in range, not trivially foldable, passes numeric check)
  before every timed run.
- **Harness sentinel** [L1, AI CUDA Engineer post-mortem]: every bench batch interleaves a
  known-cost reference kernel; if its measured ns drifts outside a band, the whole batch fails
  loudly (no silent failures). Also reject any measurement faster than its dependency-chain
  floor.
- **Archive, not hill-climb** [L7, DGM]: keep all checkpoints + configs, sample comparisons
  against more than just the current best. Concretely: the journal plus `data/checkpoints/`
  keyed by config hash.

### Division of labor [L9, L10]

Frontier calibration: agents cannot invent the cost model (no frontier model reproduces even a
documented NanoGPT speedrun from pseudocode), but they reliably run inner loops against a
trusted harness. So:

- **Agents (subagent-driven, per round):** corpus generation, training runs, ablation sweeps,
  bench execution, journal entries, per-round summary reports, adversarial mining.
- **JP (PI):** harness invariants, the one-change-per-round selection, accept/reject on
  checkpoints, problem selection, kill/pivot calls, paper writing.

Mechanically: each round is one Claude session (or one Workflow invocation fanning out the
ablation cells in parallel), reading the journal first, writing the journal last. The round
report is the session's deliverable; JP reviews between rounds.

---

## 3. Phase 0 — Foundation repairs (one-time, before any loop iteration)

The 2026-07-08 loss is not yet evidence about the architecture — it's evidence about a harness
with three known defects. No training run counts until these are fixed. Ordered by dependency:

### 0.1 Retire contaminated data (blocking)
- All pre-2026-07-20 labels/weights are on a 41.67× wrong absolute scale (timebase bug,
  `docs/results/2026-07-20-jit-compile-cost.md`). Delete any surviving `.bin` weights and
  `corpus_bench.jsonl`; regenerate everything with the fixed `nanos_now()`
  ([jit_bench.rs:181](pixelflow-pipeline/src/jit_bench.rs)).
- Fix the trainer→bench filename mismatch: `bootstrap_extraction_head` writes
  `judge_bootstrapped.bin`; `bench_extraction_3way.rs:39` reads `expr_nnue_trid.bin`. One name,
  no manual rename step.
- Fix the `bootstrap_judge` / `bootstrap_extraction_head` clap-name inconsistency
  ([bootstrap_extraction_head.rs:12](pixelflow-pipeline/src/bin/bootstrap_extraction_head.rs)).

### 0.2 Split discipline (blocking — currently NO split exists anywhere)
- Three-tier split, enforced in code via a checked-in split manifest (generator seeds + family
  IDs), not by convention:
  - **TRAIN**: synthetic corpus + mined adversarial expressions. Grows every round.
  - **DEV holdout**: held out *by generator band/seed family*, never trained on; drives the
    per-round SELECT decision. Random splits over one generator are leakage [protocol §C9].
  - **FINAL eval**: real pixelflow kernels (the 5 named production kernels + real `kernel!`
    call sites harvested from the repo + validated ShaderToy imports) + one synthetic family.
    Never used for any selection decision; touched only for the paper's claimed numbers.
- Dedup by structural hash across all tiers at corpus build time.

### 0.3 Measurement-integrity hardening (blocking)

A dedicated audit (2026-08-05) of `jit_bench.rs`, the 3-way bench, and the label-minting
binaries ranked the holes. Full report:
`docs/results/2026-08-05-bench-harness-integrity-audit.md`; findings referenced as H/M/L#.
The high-severity fixes, in leverage order:

1. **Sentinel + QoS pinning (H4).** No core pinning, no QoS class, no calibration kernel exist
   anywhere in the pipeline. A 50k-expression bootstrap run spans hours; an E-core placement or
   thermal throttle mid-run silently shifts every subsequent label, and median-of-20 only
   defends *within* one expression, not *between* them. Fix: set
   `QOS_CLASS_USER_INTERACTIVE` at bench start; benchmark one fixed reference kernel every N
   expressions.
   **Revised 2026-08-08 (JP):** the sentinel is a *calibration signal, not a tripwire*. The
   first end-to-end run showed macOS's own post-build daemons reliably produce 11–19% drift,
   and JP's position — varied data averages contention noise out the same way more runs do —
   holds provided benchmark order is randomized so drift decorrelates from expression
   structure. So: record local sentinel ns + a normalization factor
   (`calibration/local`) on every batch for post-hoc correction; shuffle benchmark order with
   a recorded seed; hard-abort only at ≥50% drift (the E-core regime-change detector — that
   class is a 2–3× step that genuinely poisons everything after it). The paired interleaved
   protocol in the 3-way gate is already drift-robust and keeps its own A/A noise floor.
2. **Independent correctness reference + real check grid (H1/H2) — the actual reward-hacking
   channel.** Today the training paths check equivalence at **one point** (`(0.5,0.7,1.3,-0.2)`
   splatted — the "4-lane" check is 4 identical lanes, [jit_bench.rs:279](pixelflow-pipeline/src/jit_bench.rs));
   the 3-way gate checks 8 easy points in `[-1.2,1.2]` at ABS_TOL 0.2 on O(1) outputs. A
   sloppier-but-faster lowering (fewer Newton steps, dropped low-order term) passes and gets
   rewarded. Worse, the reference is the same JIT backend as the thing under test
   ([bench_extraction_3way.rs:331](pixelflow-pipeline/src/bin/bench_extraction_3way.rs)), and
   the checked code object isn't the timed one. Fix: scalar-`f64` arena interpreter as the
   independent reference; randomized 64+ point grid with magnitude sweeps (1e-4…1e4, negatives,
   ±0.0); tolerance ~1e-3 relative with per-op allowances for documented transcendental
   divergence; time the same `ExecutableCode` that was checked.
3. **Timer floor + dispersion (H5/M1/M5).** One `mach_absolute_time` tick = 41.67 ns; with the
   only-used `repeat_batches=1`, tiny kernels carry 20–40% quantization error — and the tiny
   band is most of the corpus. The heavier `benchmark_jit_arena_repeated` path exists but no
   label-minting caller uses it. Fix: auto-scale `repeat_batches` until the timed sample
   ≥ ~100 ticks; record IQR/MAD in `BenchResult` and persist it (today variance is measured
   then thrown away) so training can inverse-variance-weight or reject jittery labels; measure
   an identity kernel and subtract per-eval call overhead (otherwise `log_ns` turns the shared
   constant into a nonlinear floor exactly where extraction decisions happen).
4. **Serialize the dependency chain (H3).** `eval100!` issues 100 independent calls on one
   fixed input built once — the CPU overlaps them, so the harness measures throughput under
   perfect ILP, not latency. Labels teach the NNUE that deep chains are cheap; the gate, built
   on the same harness, confirms it. Self-consistent and wrong for production (per-pixel
   varying coordinates). Fix: feed the previous output back into one input lane to serialize,
   and cycle inputs through a small varied-coordinate buffer (which also widens the correctness
   check to ~100 points for free).
5. **Plausibility floor (M4).** Validation rejects only NaN/negative/>1s. Add a per-expression
   lower bound (op count × min reciprocal throughput); a measurement below it is a harness bug,
   fail loudly.
6. **Censoring fixes (M2).** Zero-ns (constant-folded) results are dropped as `jit_failed` even
   though the harness documents zero as valid — the fastest expressions are censored from
   training. Episode minting returns `None` on compile/equivalence failure, so the policy is
   never penalized for driving expressions into failure regions — the July post-mortem's
   censored-failure pathology surviving in miniature. Fix: structured exclusion sidecar JSONL
   (name, reason, error), end-of-run assertion that the exclusion rate is under a threshold,
   and keep equivalence-failure episodes as maximal-cost labels.
7. **Gate protocol (M3).** The 3-way bench runs no-swap → heavy saturation (thermal
   perturbation) → static → NNUE in fixed order, once each, with a ±5% verdict band — the same
   order of magnitude as the drift H4/H5 allow. Fix: interleave policies round-robin, ≥3
   repeats, randomized order, per-policy medians.
8. **CI (M6, non-blocking).** `benchmark_regression.yaml` compares against a ratcheting
   baseline on shared runners: repeated 10–14% regressions each re-baseline and compound with
   no alert. Fix: bench base and head in the same job on the same runner; keep a long-horizon
   fixed anchor.

Also from the audit, for the paper's methodology section: what the harness already does right —
JIT-fn-pointer + `black_box` genuinely closes rustc-level DCE/const-prop; the timebase handling
is correct and documented; loop-counter bias is eliminated by full unrolling with the rationale
written down; failures are excluded loudly rather than averaged in; extraction overhead is
measured separately from kernel runtime; corpus is structurally deduped and seed-reproducible.
These become the methodology section's spine, with the fixes above as its "pitfalls" half.

### 0.3b Eval metrics + journal (blocking)
- `bench_extraction_3way` grows machine-readable JSONL output (config hash, per-expression ns
  per policy, failures, overhead µs) and a run-to-run **noise floor** measurement (same policy
  twice; all model claims stated relative to it).
- Add intrinsic metrics to the trainer's eval path: held-out MAE, **Spearman ρ**, and
  **top-k regret** (measure the model's top-k extraction candidates, report best-of-k vs
  measured-best). Ranking metrics matter more than MAPE because the consumer is a search.
  Note the trainer's current "tail MAE" is computed on *training* samples
  ([bootstrap_extraction_head.rs:322](pixelflow-pipeline/src/bin/bootstrap_extraction_head.rs)) —
  it is not a validation signal at all until 0.2 lands.
- Research journal: `docs/results/journal.jsonl`, append-only, one line per (config, metrics).
  Plus restore the two deleted results docs into `docs/results/` proper — deleting recorded
  negative results was the wrong call; the paper needs that provenance.

### 0.4 Numeric integrity gate (blocking for corpus integrity) — reframed 2026-08-06
The contract line is **algebraic validity**, not IEEE value-identity: pixelflow is a math
library over the reals, not an IEEE library (JP, 2026-08-06). An algebraically valid rewrite is
within contract even where it changes computed values at or near singularities (`x/x → 1` at
zero, overflow paths at ~1e40) — so the 07-08 run's 7/40 cross-form disagreements, to the
extent they are of that shape, are *contract*, not a "rule soundness gap". What a gate must
catch is implementation error, which splits the check in two:

- **Same-form hard gate:** scalar reference interpreter vs JIT on the *same arena*, per-op
  tolerances (`equivalence_tolerance` in pixelflow-ir/src/eval.rs). Divergence here is a
  miscompile or lowering bug — panic-worthy at any meaningful rate. This alone closes the
  reward-hacking channel (H1): a sloppier-but-faster lowering diverges from the reference on
  *its own* form, no cross-form comparison needed.
- **Cross-form conditioned gate:** extracted form vs original, compared only at
  well-conditioned points (no near-singular intermediates, finite moderate outputs).
  Disagreement *there* indicates an e-graph/extraction bug and fails hard. Divergence at
  ill-conditioned points is recorded as metadata — never an exclusion, never an alarm.

The previously spun-off "rule soundness investigation" is withdrawn: its premise
(algebraically-valid-but-unsound) was a category error.

### 0.5 Repo hygiene (partly blocking — the duplicates are live binaries)
- Delete the 26 committed `"* 2.*"` Finder duplicates (after 0.3b restores the results docs).
  Not cosmetic: `src/bin/*.rs` auto-discovery compiles `bench_extraction_3way 2.rs` (an older
  version missing the AVX2 `run_scalar` variant — wrong-ABI risk on AVX2 hosts) and
  `episodes 2.rs` as live Cargo targets; tab-completing the stale gate binary reports stale
  numbers (audit L1).
- Small loud-failure fixes from the audit: assert `v > 0.0` before `ln` in the geomean (one
  zero currently makes it 0/−inf silently, L2); use `serde_json` for corpus JSONL instead of
  unescaped `format!` (L3).
- Fix or delete the three stale docs (`NNUE_INTEGRATION_STATUS.md`, `NNUE_TRAINING_RECIPE.md`,
  `EGRAPH_SEARCH_INTEGRATION.md`) that document deleted systems.
- Fix the stale param-count comments in `factored.rs:2487-2517`.

**Phase 0 exit criterion:** `gen_bench_corpus → bootstrap_extraction_head → bench_extraction_3way`
runs end-to-end from a fresh checkout with one command per stage, produces a journal entry, and
the noise floor is measured and recorded. Then round 1 begins.

---

## 4. Phase 1 — The iteration loop

### 4.1 Round structure (one session each)

1. **Read journal**; pick the round's single change via the standing ablation table.
2. **Train** (fast path: ~4k samples trained in 48s previously; scale up as ablations justify —
   the corpus supports 360k and the supervised core was never the bottleneck).
3. **Eval-intrinsic** on DEV: MAE, Spearman, top-k regret.
4. **Eval-e2e** on DEV: 3-way bench geomean vs latency prior + extraction overhead µs.
5. **Journal + round report** (agent deliverable): what changed, all metrics, the ablation table
   for next round, mined adversarial expressions added to TRAIN.
6. **Human gate**: JP accepts/rejects the checkpoint.

### 4.2 Candidate dimensions for PROPOSE (initial ablation table)

Ordered by prior expected value given the diagnosed defects of the losing run:

| # | Dimension | Rationale |
|---|-----------|-----------|
| 1 | Clean-scale labels + 10-50× more training data | The losing run: 4k samples on 41.67× mis-scaled labels. Cheapest possible fix. |
| 2 | Ranking loss (pairwise/LambdaRank) alongside regression | Consumer is a search; AutoTVM/TLP lineage. Directly targets extraction decisions rather than absolute ns. |
| 3 | Corpus mix: real-kernel-shaped vs synthetic bands; adversarially mined exprs | McNamara-fallacy guard [L h]: re-derive op distribution from actual repo kernels, not just ShaderToy weights. |
| 4 | Accumulator section ablation (EdgeAcc; GraphAcc parent/child/1-hop/2-hop) | Which features carry signal for *cost* (vs the mask head's needs); leak detection. |
| 5 | `top_k` in `IncrementalExtractor` (currently 8) + beam shape | Directly trades extraction overhead (the 31×) vs quality. |
| 6 | Target-awareness: per-ISA-level labels/features | Platform-divergent ops (CLAUDE.md tables) make a single cross-target model suspect; `isa-matrix` enumerates the levels. |
| 7 | Extraction overhead engineering: incremental-update path profiling | The 31× overhead is its own claim-killer independent of quality; also feeds the evals/sec paper section. |

### 4.3 Gates

- **Round accept gate** (existing bench verdict, kept): >+5% geomean over latency prior on DEV
  → promote checkpoint. Within ±5% → iterate. The bench already prints this verdict.
- **Claim gate** (publication): >+5% geomean on FINAL (never used for selection), extraction
  overhead within the proc-macro time-control budget
  ([optimize.rs:150-186](pixelflow-compiler/src/optimize.rs)), noise floor below the claimed
  margin, all per §5's checklist.
- **Kill/pivot gate**: if after **5 clean rounds** (clean labels, split, scaled corpus) the e2e
  geomean still loses to the latency prior, stop iterating on the win and pivot the paper to
  the honest alternative (§6) — which is still publishable at the chosen bar. The prior's
  60-entry cycle table is a strong baseline for scalar-cost DAG extraction; losing to it
  informatively is a result. No unbounded iteration.

### 4.4 What is explicitly OUT of scope for this loop

- The saturation head / Guide (Phase 3 of the redesign plan). The mask backward pass doesn't
  exist (`unified_backward.rs:1-29`), the union-causality over-approximation must be tightened
  first (`redesign.md:89-90`), and mixing it in doubles the loop's variables. It is the
  *second* paper (or the second half of this one, only if the extraction claim lands early).
- Memory ops (`Buffer`/`Gather`) — not representable in the e-graph yet
  ([graph.rs:501-514](pixelflow-search/src/egraph/graph.rs)); the paper's scope statement says
  arithmetic kernels.
- Any RL machinery. The literature converged where the July audit did: supervised regression on
  measured ground truth. [L j]

---

## 5. Phase 2 — Paper assembly (checklist form)

Claim framing: **"NNUE for e-graphs"** — e-graph search has game-tree search's profile (huge
numbers of evaluations of slightly-perturbed sparse states, on CPU, under a latency budget);
port the game-engine solution. Halide 2019 is the acknowledged template for the
measured-corpus + small-net recipe; SmoothE/e-boost are the extraction-quality SOTA to position
against (workshop bar: cite and situate, full-venue bar: run them on extraction-gym).

Reviewer checklist assembled from the survey (§2 of the lit report):

- [ ] MAPE **and** Spearman ρ **and** top-k regret on DEV and FINAL, per target ISA level
- [ ] Geomean speedup vs latency prior **plus per-benchmark distribution** (never geomean alone)
- [ ] Quality-vs-budget curve (extraction time budget ↔ kernel quality; the proc-macro framing
      is a strength — MLGO's framing)
- [ ] Noise floor stated, claims stated relative to it; median-of-N, warmup, timebase method
      documented (BHive lineage)
- [ ] Split-by-family documented; FINAL = real kernels, never selected on
- [ ] Censoring: compile failures / numeric-check exclusions reported with counts
- [ ] Ablations: per accumulator section; incremental-vs-full-recompute **equivalence test** +
      evals/sec comparison (the architectural novelty's evidence)
- [ ] Candid pitfalls subsection: node_count leak, timebase bug, loop-counter bias — documented
      pitfalls are a contribution
- [ ] Differentiate from: SmoothE, e-boost, EggMind, Omelette/MCTS-GEB, Isaria, Halide 2019;
      cite HER for hindsight-labeling lineage
- [ ] Artifact: corpus + harness releasable (a miniature Tenset; strengthens any venue)

**Venue timeline:** EGRAPHS @ PLDI 2027 deadline lands ~April 2027 (2026's was April 17);
ML-for-Systems workshops (NeurIPS ~May/Jun, ICML ~Jan/Feb deadlines) are intermediate targets.
arXiv preprint as soon as the FINAL-gate numbers exist. A workshop talk needs the incrementality
trick + provenance idea + honest numbers; it does not need the full SmoothE comparison matrix —
that's the later full-venue upgrade (CGO/C4ML or MLSys).

---

## 6. The honest-negative fallback (pre-registered)

If the kill gate fires, the paper becomes: *"An incrementally-updatable cost model for e-graph
extraction: architecture, and why a 60-entry latency table is hard to beat"* — the NNUE-for-egraphs
architecture + equivalence/evals-per-sec results + the characterized VSA topology-blindness
tradeoff + a rigorous negative on learned-vs-static extraction at proc-macro budgets + the
provenance/hindsight-labeling substrate as the forward-looking contribution. EGRAPHS-shaped,
credible, and it pre-commits us against the temptation to iterate until noise produces a win
[L2: assume the scalar you optimize is adversarially targeted — including by yourself].

---

## 7. Immediate next actions (Phase 0 work items, delegable)

1. **W0-A** Retire stale data + filename/clap fixes (0.1) — small, mechanical.
2. **W0-B** Split manifest + corpus dedup + tiered corpus build (0.2).
3. **W0-C** Harness integrity, tier 1 (0.3 items 1–3): sentinel + QoS pinning; scalar-f64
   interpreter reference + randomized wide check grid; tick-floor auto-scaling + IQR recording
   + identity-kernel overhead subtraction.
4. **W0-D** Harness integrity, tier 2 (0.3 items 4–7): dependency-chain serialization,
   plausibility floor, censoring sidecar, interleaved gate protocol.
5. **W0-E** Eval metrics + journal (0.3b): 3-way JSONL output, noise-floor run, Spearman/top-k
   in trainer eval, `docs/results/journal.jsonl`.
6. **W0-F** Corpus-time numeric quarantine (0.4); spin off rule-soundness investigation.
7. **W0-G** Repo hygiene sweep: duplicates (live-binary risk first), stale docs, stale
   comments, geomean/JSON loud-failure fixes (0.5).
8. **W0-H** Restore the two deleted results docs into `docs/results/` with a short preamble
   noting the timebase caveat.
9. **W0-I** CI fix (0.3 item 8): same-runner base/head comparison + fixed long-horizon anchor.
   Non-blocking for round 1.

W0-A/G/H are independent; W0-C is the highest-leverage single item (protects every label ever
minted); W0-B/C/D/E/F can proceed in parallel after A. One session each, or one Workflow
fan-out. Round 1 (ablation row #1: clean labels at scale) starts when the Phase-0 exit
criterion holds — which now includes: sentinel active, independent correctness reference in
place, tick-floor scaling on, and the noise floor recorded in the journal.
