> **Partly superseded (annotated 2026-09-09) — the reading list is the live half.**
> §5's literature map, §3's VSA analysis and §6's adoption order are what this
> document is for and they stand. §1's defect list describes a tree that no longer
> exists (see the 2026-08-17 reconciliation note below), and the files it names —
> `pixelflow-ml/src/nnue.rs`, `unified_backward.rs`, `extraction.rs` — were deleted
> with the extraction-head program in Sep 2026.

# E-graph × VSA × NNUE: Review Findings and Literature Survey

**Date:** 2026-08-17
**Status:** Research notes — companion to `2026-07-07-guided-saturation-redesign.md`
and `2026-08-05-egraph-nnue-research-workflow.md`.
Records a full review of `pixelflow-search/src/nnue/` plus a literature sweep, so the
findings and the reading list survive context loss.

> **Reconciled 2026-08-17 with the measurement branch (PR #984).** These notes were
> written against `main` while the harness/experiment work sat unlanded, so §1's
> defect list and §4's framing describe a tree that no longer exists. Corrections are
> marked **[LANDED]** / **[MEASURED]** inline; §1.5 records the Round-1 result the
> notes predate. The literature map (§5), the VSA analysis (§3) and the adoption
> order (§6) survive intact and are the live reading list — with the §3 consequence
> now binding on how we describe the work publicly.

---

## 1. Where the system actually is

One-fifth of the NNUE surface is live; four-fifths is Phase-3 scaffolding with no
consumer and no training gradient.

**Live path (the only one that runs):**

```
train: BwdGenerator → EdgeAccumulator::from_arena_dedup → forward → value_mlp → MSE on log-ns
serve: extract.rs → EdgeAccumulator::from_dag_choices_with_variance → forward_expr_only → value_mlp
```

**Dead end-to-end** (zero callers outside `factored.rs`, zero gradient producers):
`GraphAccumulator` + 1-hop/2-hop VSA bindings, mask MLP + bilinear scoring,
`RuleTemplates`/`RuleFeatures` encoders, Bernoulli policy / MCTS support,
`forward_graph`, `mask_score_all_rules_*`, the entire incremental
`add_edge`/`remove_edge` API. All of it is serialized into `TRIE` weight files as
untrained noise.

### Live-path defects (fix before building anything on top)

| # | Defect | Where | Status |
|---|--------|-------|--------|
| 1 | **`w1[128..132]` slot collision.** Trained against `log2(1+count)` scalars (`unified_backward.rs:131-134`), served with variance fractions (`factored.rs` `forward_expr_only`). One weight row, two domains — the exact "one f32 lane, several meanings" failure CLAUDE.md warns about. | `factored.rs`, `unified_backward.rs` | **[LANDED]** PR #984 |
| 2 | **Train/serve accumulator skew.** Training uses `from_arena_dedup` (no var-ref reload edges, variance fracs always 0); serving uses `from_dag_choices_with_variance` (reload edges + real variance). Distribution shift is worst on CSE-heavy expressions — the ones extraction exists for. | `bootstrap_extraction_head.rs:134,207` vs `extract.rs:67,130` | **[LANDED]** PR #984 |
| 3 | **Embeddings frozen but decaying.** `backward_through_accumulator` has no callers, so `d_embeddings ≡ 0`; unconditional weight decay still erodes the latency prior in dim 0. Training actively destroys the embeddings' only signal. | `unified_backward.rs:984-1050,1123-1129` | **OPEN — top Round-2 candidate** |
| 4 | **"Incremental" extractor is O(n) per candidate.** Full accumulator rebuild + full ref-count + cycle check per candidate swap (`extract.rs:79-83` admits it). The NNUE's raison d'être — O(Δ) updates — is unused. | `extract.rs:124-138` | **[MEASURED]** still true, but see below |

**#1 and #2 are fixed.** PR #984 unified both paths on the *deployment*
representation: one walker (`EdgeAccumulator::from_cost_dag`) and one input vector
(`extraction_input()`), with `from_arena_dedup` and the log2-scalar path deleted so
the drift vectors are unrepresentable rather than merely discouraged. The parity test
these notes ask for exists and passes: `train_and_deploy_feature_paths_agree`.

**#3 is the most valuable finding in these notes** — both prior measurement rounds
missed it. One refinement on inspection: the *decay* half is largely neutralised,
since weight decay is a uniform per-element shrink and the post-SGD unit-sphere
projection (`unified_backward.rs`, "Post-SGD embedding normalization") renormalises
it away. The core claim stands and is worse than stated: `backward_through_accumulator`
has no callers, so **the embeddings never learn at all** — the model can only reweight
frozen per-op features through `w1`. Worse, the unit-sphere projection itself flattens
the latency-prior initialisation, compressing the dim-0 cost ratios between ops. A
model that cannot sharpen per-op cost distinctions beyond a linear readout of
distorted constants is a plausible contributor to the conservatism in §1.5.

**#4 is confirmed but its economics are now measured, and they are bad for the
premise.** The extractor really does perform zero incremental updates (2,323 full
rebuilds, 0 delta updates; the incremental API is called only from unit tests). But
the median base-vs-candidate edge-multiset symmetric difference is **44.9%** (p25
17.6%, p75 100%). E-class alternatives are restructured subtrees, not local piece
moves, so a correct O(Δ) implementation saves roughly half the work — not the ~98%
the chess analogy promises. See §4.

### 1.5 What the fixed path actually measures (Round 1, PR #984)

Recorded here because these notes' §6 ordering assumes the answer is unknown. It is
partly known now.

**The head is well-calibrated after the skew fix, and extraction still loses.**

| | before (skewed) | after (unified) |
|---|---|---|
| DEV Spearman ρ | 0.9624 | **0.9799** |
| DEV MAE (log-ns) | 0.9753 | **0.1810** |
| prediction bias | −0.886 | **+0.020** |
| pred/true σ ratio | 0.68 (compressed) | **0.93** |

End-to-end on the full DEV tier (392 kernels, 316 paired comparisons, corrected
latency prior, A/A noise floor ±0.05%): **nnue/static geomean 1.0181, bootstrap 95%
CI [1.0106, 1.0254], median 1.0182, 73W/238L/5T, leave-one-out range
[1.0171, 1.0198]**. NNUE-guided extraction is ~1.8% *slower*, and the interval is
tight enough that no single kernel drives it. (Round 0, before the fix: 1.0389,
7W/18L.) That run self-invalidated on the cross-form correctness gate (12.31% > 10%),
so it is not a publishable verdict — but the estimator is stable and the sign is not
in doubt.

Two mechanisms, both measured:

1. **NNUE is the *conservative* policy** — the opposite of the intuition. Across 12
   re-extracted kernels the static prior substitutes 26 hardware primitives
   (Recip/Rsqrt/MulAdd/Sqrt) where NNUE substitutes 14; static drives `Neg` to zero in
   8/12 kernels while NNUE never gets below 2; NNUE's DAGs are larger and deeper on
   12/12. The per-pair speed ratio is near-monotone in the estimate-op delta.
2. **Corpus-level ranking does not transfer to variant ranking.** ρ ≈ 0.98 across
   *different* expressions, but extraction needs ranking among *rewrite variants of
   the same expression*, where true cost differences sit near the measurement floor.
   For calibration: on the same DEV set the static prior scores ρ = 0.9438 and a bare
   count of transcendental/divide ops scores 0.9486 — the corpus-wide metric is
   saturated by expression size and cannot distinguish these models at all.

Also measured, and relevant to §6's ordering: `static/noswap = 0.51`. E-graph
extraction *with the static prior* is worth ~2× on this corpus. The extraction
machinery pays; the learned head is what does not, yet.

Two incidental results worth keeping: the latency prior was mispriced
(`Sqrt = 15 > Pow = 12`, while `Pow` lowers to log2+mul+exp2 — measured 60.2 ns vs
21.3 ns on the same kernel), now re-derived from measurement; and the full-DEV run
flushed out three extraction bugs the 40-kernel runs never reached (a 2.7 GB OOM on
cyclic choice sets and two non-terminating cycle breakers).

---

## 2. Lineage (what was tried before, and why it went away)

Recorded so nobody re-derives or re-fears these.

1. **HalfEP NNUE** (`pixelflow-ml/src/nnue.rs`, see `docs/archive/GNN_REWRITE_GUIDANCE_VISION.md`):
   O(ops²) one-hot (perspective_op, descendant_op, depth, path) features, 401k inputs.
   Superseded by the factored O(ops) embedding scheme in
   `pixelflow-search/src/nnue/factored.rs` — same move chess NNUE made going from
   HalfKP to factored feature sets.
2. **GNN vision** (`docs/archive/GNN_REWRITE_GUIDANCE_VISION.md`): message passing over
   e-classes, teacher = full saturation, student = GNN predicting productive
   rewrites. Never built in Rust; the constraint (no-alloc, CPU, incremental,
   proc-macro time) is why. The idea survives in a different position — see §5
   (teacher-student distillation).
3. **Unified RL loop** (`docs/plans/archive/2026-02-25-unified-training-*`,
   `docs/NNUE_TRAINING_RECIPE.md`): Rust actor + Python **Causal Sequence
   Transformer critic** (`critic_server.py`, `graph_teacher.py`, PyTorch) doing
   temporal credit assignment for REINFORCE. Removed 2026-07 after the four-agent
   audit (deterministic policy under REINFORCE, advantage collapse, censored
   failures, unconsumed policy). Important nuance: **the critic was cut because the
   RL estimator was unsound, not because teacher-student is a bad idea.** A
   supervised teacher (label states offline, distill into the NNUE) shares none of
   the audited flaws.
4. **Current plan** (`2026-07-07-guided-saturation-redesign.md`): Judge/Guide/Search,
   supervised only, hindsight provenance labels. Phase 2 (static prior default,
   NNUE opt-in) is done and matches `extraction.rs`. Phase 3 (greedy guided
   saturation — the actual thesis test) has not started; the mask head currently
   reads as live code but is inert.

---

## 3. VSA capacity — the "paradox of centrality," pinned down

The half-remembered capacity math: bundle N quasi-orthogonal vectors in dimension D
and the bundle's similarity to any one component decays like **1/√N**, while the
bundle itself drifts toward the centroid of the space — weakly similar to
everything, diagnostic of nothing. Reliable retrieval capacity scales **linearly
with D** (Plate's HRR analysis; Clarkson et al.'s capacity theory; Kleyko's
two-part survey is the modern reference).

Consequences for this codebase at **K=32**:

- A 32-dim bundle holds a handful of bindings before crosstalk swamps it. Our
  accumulators bundle hundreds of edges. As *symbolic memory*, the VSA framing is
  not operative at this dimension — no unbinding/recovery guarantee holds.
- The design survives anyway because **nothing ever unbinds**: a trained decoder
  (`w1`/`graph_w1`) reads the bundle. What we actually have is a learned
  multiplicative feature map with permutation-based asymmetry. That is fine — but
  claims should say "learned features," not "VSA," and capacity intuitions from the
  VSA literature must not be load-bearing.
- Cyclic-shift depth binding aliases mod K (`shift_by(x, 32) == x`); depths ≥ 32
  collide with depth 0.
- **Cheap upgrade if honest binding is ever wanted:** the depth encoding in
  `EdgeAccumulator` is already FHRR — unit-modulus complex phasor rotation. The op
  embeddings are the unconstrained part. Constraining each complex pair of an op
  embedding to unit modulus would make every binding a norm-preserving, exactly
  invertible rotation at zero extra dimension cost.

---

## 4. The chess-engine framing, made precise

Position ≈ expression, move ≈ rewrite, eval ≈ cost model, NNUE ≈ incrementally
updatable eval under small state deltas. Three ways our game is *easier* than chess:

1. **No adversary.** Planning, not minimax; no self-play instability.
2. **Objective ground truth.** JIT the expression and time it — chess never gets
   this. Labels are expensive but *true*.
3. **Monotone moves.** E-graph rewrites only add equivalences; nothing is ever
   lost. Ordering matters only under budget. ("AlphaDev, not chess" — already the
   plan doc's conclusion.)

The one way it is *harder*: chess gets a free reward every game; each of our labels
costs a JIT + benchmark run and is noisy. Label economics, not search depth, is the
binding constraint (see two-tier labels, §5).

The analogy pays rent only when incremental evaluation actually runs (defect #4)
and, later, when weights are quantized (real NNUE speed is int8/int16 SIMD dot
products; ours is all-f32).

**[MEASURED] A fourth difference, and it is the one that matters: our deltas are not
small.** NNUE exists because a chess move perturbs a handful of features out of
thousands, so O(Δ) refresh is ~50× cheaper than recompute. Measured on our workload,
the median edge-multiset symmetric difference between a base DAG and a candidate is
**44.9%** (p25 17.6%, p75 100%, p90 180%); only 11% of candidates change under 10% of
edges, and stripping the depth encoding barely moves it (51.5% vs 56.8% mean), so it
is not a positional-encoding artifact. E-class alternatives are *restructured
subtrees*, not piece moves. Even a perfect incremental implementation buys ~2×.

That is a load-bearing negative result, not a caveat: the architectural argument for
NNUE-in-e-graphs — the thing our own literature survey listed as unclaimed novelty —
rests on a premise this workload does not satisfy. Extraction overhead is also
second-order in practice (NNUE extraction is ~13% of the e-graph pass; p99 is 4.5% of
a blitz compile budget), so §6 step 5 should be understood as a modest constant-factor
win, and any paper should state the delta measurement rather than the analogy.

---

## 5. Literature map (2021–2026): what exists, where it plugs in

### Extraction

| Work | What it is | Relevance |
|------|-----------|-----------|
| **SmoothE** (ASPLOS 2025, best paper) | Differentiable e-graph extraction: per-class softmax over e-nodes, continuous relaxation, gradient descent on the cost; built for complex/nonlinear cost models, GPU-friendly | **Highest-value import.** Our cost model is already differentiable (hand-written backprop). Replaces the fake-incremental refinement loop in `extract.rs` with a principled optimizer. |
| **e-boost** (arXiv 2508.13020) | Parallel heuristic extraction warm-starting an ILP solver; adaptive pruning | Practical middle path for the static-prior default. Extraction dominates end-to-end eqsat runtime (~89% in their measurements). |
| DAG extraction NP-completeness; extraction-gym | Hardness result + benchmark suite | Justifies heuristic/learned extraction at proc-macro time; gym gives baselines to compare `IncrementalExtractor` against. |

### Cost models (the Judge)

| Work | What it is | Relevance |
|------|-----------|-----------|
| **Ithemal** (2018) / **BHive** | LSTM basic-block throughput model + measurement harness/benchmark | Methodology for noisy-label benchmarking; corpus design. |
| **GRANITE** (2022) | GNN over instruction dependency graphs; ~6.9% error, beats Ithemal | Validates graph-structured representation. Our 2-hop binding is a "1-round GNN" — the literature says 1 round is short. Constraint says we can't ship a GNN → use it **offline as a teacher** and distill into the NNUE. This resurrects `docs/archive/GNN_REWRITE_GUIDANCE_VISION.md` in a sound position (supervised, offline, Python allowed — unlike the removed RL critic). |
| **llvm-mca / uiCA** | Analytical/simulated x86 throughput predictors | **Two-tier labels:** pretrain on millions of free analytical labels, fine-tune on measured ns from `jit_bench`. Multiplies effective corpus by orders of magnitude; biggest bang for zero architecture change. |
| Halide autoscheduler (Adams 2019); TenSet/MetaSchedule | Learned cost models beating hand-tuned schedules in narrow SIMD/tensor domains | The existence proof for "our niche": closed op set + cheap ground truth ⇒ learned model beats hand tuning. The goal is beating LLVM *here*, not in general. |

### Guidance / saturation (the Guide)

| Work | What it is | Relevance |
|------|-----------|-----------|
| **Omelette** (Singh, Cambridge MPhil 2022) | PPO agent choosing which rewrite rules to fire per epoch on an e-graph | Functionally *is* our mask head, tried with RL. Read for failure modes before Phase 3. |
| **MCTS-GEB** (MLSys workshop 2023) | MCTS planning of e-graph construction under budget | Same motivation: unsaturable graphs re-introduce phase ordering. Planning beats greedy rule firing. |
| **Sketch-Guided EqSat** (2022) / **Guided EqSat** (POPL 2024, Koehler et al.) | Decompose one big saturation into small ones aimed at human-written sketches | The decomposition our budget problem wants. |
| **ML-Guided EqSat** (EGRAPHS 2025) | Learn the guides/sketches instead of hand-writing them | Philosophically our hindsight-provenance labeling: mine successful runs for intermediate targets. `egraph/labeler.rs` is the right substrate; this line says the idea is current. |
| **LLM-guided** (LGuess, arXiv 2511.00403; ASPEN, MLCAD 2025) | LLM proposes rewrite checkpoints/strategies, e-graph fills in chains | Not viable at proc-macro time; interesting offline for strategy synthesis. |
| **Ruler** (OOPSLA 2021) | Automatic rewrite-rule inference via eqsat | Grows the rule library without hand-writing rules — directly serves the "scale to hundreds of rules" thesis, arguably ahead of the Guide itself. |
| "Rewrite System Showdown: Stochastic Search vs. EqSat" (arXiv 2605.19005, 2026) | Head-to-head STOKE-style stochastic search vs equality saturation | Strategy-level sanity check on when eqsat is even the right tool. Unread — flagged for follow-up. |

---

## 6. Prioritized adoption plan

Ordered; each step is independently shippable. 0 is a prerequisite for everything.

0. **Fix the live path** (§1 defects 1–4) and add the train/serve forward parity
   test. Every idea below sits on top of a trustworthy train/serve path.
   → **defects 1, 2 and the parity test are [LANDED] (PR #984).** What remains of
   step 0 is **defect 3 — enable embedding gradients** (wire
   `backward_through_accumulator`, and revisit whether the unit-sphere projection
   should apply to op embeddings at all, since it flattens the latency-prior init it
   is meant to preserve). This is now the single highest-value unstarted item: the
   model currently cannot learn per-op cost structure, which is exactly the structure
   §1.5 shows it failing to exploit.
1. **Two-tier labels:** pretrain the extraction head on llvm-mca (and/or uiCA)
   predictions over a large generated corpus; fine-tune on measured ns. No
   architecture change.
   → **Re-aim per §1.5:** corpus-level ranking is already saturated (ρ 0.98; a bare
   op-count scores 0.949), so more labels *of the same kind* buy little. The gap is
   ranking rewrite variants of one expression, where measured deltas sit at the noise
   floor and analytical labels are deterministic and free. Spend the mca tier on
   **contrastive pairs drawn from within e-graph candidate sets**, with measured ns
   for calibration — same import, aimed at the actual deficit.
2. **SmoothE-style differentiable extraction:** softmax-relaxed per-class choices,
   descend on `predict_log_cost`. Retire the per-candidate full rebuild.
   → **Use it as a *training* mechanism first, not a serving one.** As a serving
   optimizer it converges harder onto a head that cannot rank variants (and extraction
   overhead is second-order anyway, §4). But soft per-class choices make extraction
   differentiable end-to-end, which is what lets the head be trained *through*
   extraction against measured outcomes — the fine-grained objective step 1 is trying
   to approximate. Train with it, then serve with it.
3. **Phase 3 prep:** read Omelette + MCTS-GEB failure modes; train the Guide
   *supervised* on hindsight provenance labels (per the redesign plan), never
   REINFORCE. Consider GRANITE-style offline GNN teacher → NNUE student
   distillation for the Judge at the same time.
4. **Ruler-style rule inference** to grow the rule library — the thesis is about
   scaling rules; infrastructure without rules tests nothing.
5. **Quantization (int8/int16) + true incremental updates** once extraction is
   incremental — the actual Stockfish trick, in its proper order.

**Non-goals, recorded deliberately:** beating LLVM outside the pixel-kernel niche;
any return of REINFORCE/self-play; treating K=32 bundles as symbolic memory;
building Phase-3 machinery ahead of its consumer (the dead surface in §1 should
shrink, not grow, until Phase 3 starts).

**Consequence of §3 for how this work is described.** Our own literature survey
listed "VSA graph featurization for compiler ML" as unclaimed novelty. Per §3 that
claim should be withdrawn: at K=32 bundling hundreds of edges, no retrieval guarantee
holds and nothing ever unbinds. Describe it as a **learned multiplicative feature map
with permutation-based asymmetry** — still unoccupied territory in compiler ML, and
defensible to a reviewer who knows Plate/Kleyko. If honest binding is ever wanted, the
FHRR upgrade in §3 (unit-modulus op embeddings) is the cheap route, and note the depth
encoding aliases mod K.

**Pre-registered exit.** `2026-08-05-egraph-nnue-research-workflow.md` §6 commits to
an honest-negative writeup if the learned head cannot beat the prior. §1.5 is two
evidenced losses with a mechanism; steps 0 (defect 3) and 1–2 (contrastive objective)
are the remaining model-side levers. If those do not move the sign, the negative
result is the deliverable — a well-calibrated learned cost model losing to a measured
50-entry table at e-graph extraction, plus the 44.9%-delta finding in §4, is a
stronger and more useful paper than a marginal win would have been.

---

## References

- SmoothE: <https://dl.acm.org/doi/10.1145/3669940.3707262> · PDF: <https://www.csl.cornell.edu/~zhiruz/pdfs/smoothe-asplos2025.pdf>
- e-boost: <https://arxiv.org/abs/2508.13020>
- Extraction NP-complete: <https://effect.systems/blog/egraph-extraction.html>
- GRANITE: <https://arxiv.org/abs/2210.03894>
- Ithemal: <https://github.com/ithemal/Ithemal>
- MCTS-GEB: <https://arxiv.org/abs/2303.04651>
- Omelette (Deep RL for EqSat): <https://www.cl.cam.ac.uk/~ey204/pubs/MPHIL_P3/2022_Zak.pdf>
- Guided Equality Saturation (POPL 2024): <https://dl.acm.org/doi/abs/10.1145/3632900>
- Sketch-Guided EqSat: <https://arxiv.org/abs/2111.13040>
- ML-Guided EqSat (EGRAPHS 2025): <https://pldi25.sigplan.org/details/egraphs-2025-papers/6/Machine-Learning-Guided-Equality-Saturation>
- LLM-Guided EqSat (LGuess): <https://arxiv.org/abs/2511.00403>
- ASPEN: <https://www.csl.cornell.edu/~zhiruz/pdfs/aspen-mlcad2025.pdf>
- Ruler: <https://dl.acm.org/doi/abs/10.1145/3485496>
- Kleyko HDC/VSA survey I: <https://arxiv.org/abs/2111.06077> · II: <https://arxiv.org/abs/2112.15424>
- Clarkson, VSA capacity: <https://redwood.berkeley.edu/seminars/ken-clarkson-apr-2023>
- Stochastic search vs EqSat: <https://arxiv.org/abs/2605.19005>
