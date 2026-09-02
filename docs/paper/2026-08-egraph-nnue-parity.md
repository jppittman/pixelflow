# A Well-Calibrated Learned Cost Model Ties a Measured Latency Table at E-Graph Extraction (and Two Targeted Attempts to Beat It Backfired)

**Status:** workshop paper, final form (EGRAPHS @ PLDI / ML-for-Systems bar), 2026-08-31.
**Provenance:** every number in this paper maps, via
[`NUMBERS.md`](NUMBERS.md), to a line of `docs/results/journal.jsonl`, a
named results/plan document, or a per-kernel data artifact. Two classes of
exception are marked there and are worth stating up front rather than
burying in a table:

- **Round 2b (§5.3) is carried from a session draft**, not from a journal
  line — that worktree no longer exists on disk and the line was never
  committed. Those numbers underpin §5.3's central result, so treat them as
  the paper's weakest trace until the run is re-executed.
- **The per-kernel `D2a`/`D3` JSONL artifacts are not committed to this
  repository.** Figures re-derived from them (extraction overheads, floor
  statistics, gate counts, the flip analysis) were verified against those
  files in-session, but a reader with only this repository cannot reproduce
  them; `journal.jsonl` carries the run-level numbers and not the per-kernel
  rows.

---

## Abstract

We ask whether a learned cost model can beat a handwritten per-op latency
table at e-graph extraction time, inside a Rust proc-macro, for a SIMD
pixel-kernel DSL. The answer, measured to the noise floor, is no — and the
mechanisms behind the "no" are the contribution.

An NNUE-style cost model (factored per-op embeddings, an incrementally
updatable edge accumulator, a small MLP head) was trained on JIT-measured
latencies and reached excellent calibration — ultimately Spearman
ρ = 0.9887 and MAE 0.133 log-ns on a family-held-out DEV tier, better than
the static table (ρ = 0.9438) or a bare op count (ρ = 0.9486) by every
intrinsic metric — though those two baselines were measured on Round 1's
392-kernel DEV tier, not Round 3's 784, so the comparison is across
populations and Spearman depends on the population (see §5.0). Driving extraction with it produces kernels that **tie**
the static table: nnue/static geomean 1.0037, 95% CI [0.9982, 1.0089] over
719 paired kernels (276W/409L/34T), leave-one-out geomean range
[1.0031, 1.0044], against an A/A noise floor of ±0.07% (Round 2a). This was
the first run in five attempts that our own harness did not censor or
invalidate.

Two pre-registered model-side levers were then pulled, and both backfired
measurably. (1) A contrastive ranking loss over within-e-graph variant sets
targeted the model's one measured behavioral gap — it declines
hardware-estimate-op substitutions the table takes. The intervention moved
the mechanism (57.2% of the 187 conservative kernels moved toward static's
substitution count, 45.5% matched or exceeded it) and worsened the
outcome (1.0153, 95% CI [1.0097, 1.0213], entirely above parity); only
39.2% of the newly-taken substitutions with a measured outcome were wins,
refuting the assumed mechanism–speed link. (2) Enabling the embedding gradients — the
embeddings had been frozen by a defect for the entire program, so this was
the first time the model could learn per-op cost structure at all —
produced the best calibration ever measured (ρ = 0.9887) and a confirmed
end-to-end regression: 1.0082, 95% CI [1.0031, 1.0133], the entire interval
above parity (Round 3).

The historical arc is the summary: as *our own* measurement defects were
removed, the learned model's deficit marched to parity (1.0389 → 1.0181 →
1.0037), and both attempts to leave parity by improving the model landed on
the slow side of it (1.0153, 1.0082). One defect outlived the writing: the
aggregation never subtracts the 4.272 ns call overhead it measures, which
biases every ratio toward 1 — so "parity" is the reading these numbers most
overstate, while the two negative results only get larger under the
correction (§5.0). Parity behaves like an attractor: the
objective is nearly additive, dynamic programming over the additive table
already optimizes it exactly (over unfolded tree cost — §6 states the gap to
DAG cost), and the residual sits at the per-pair measurement floor.

We also report a negative architectural finding with a positive twin. The
premise that makes NNUE pay off in chess — tiny state deltas — does not
transfer to extraction: the median edge-multiset symmetric difference
between a base DAG and an e-class candidate is 44.9%, so a perfect O(Δ)
update buys about 2×, not ~50×. One layer up, the economics invert: during
*saturation*, 91.1% of rule applications change nothing at all, and the
median edge-delta among state-changing applications is 0.14% (~728×
implied) — the Stockfish argument holds on the monotone e-graph state, which
is where our future work goes.

The measurement harness that made these negatives trustworthy — sentinel
record-and-normalize with regime-change abort, an independent scalar oracle,
compositional error bounds under an algebraic-validity contract,
family-held-out tiers with feature-quotient fencing, schema-derived artifact
identities, and verdicts that self-censor — is described as a first-class,
reusable contribution. Four of our first five end-to-end runs were
invalidated by the harness's own gates; we count that as the system working.

---

## 1. Introduction

pixelflow is a pull-based functional graphics eDSL embedded in Rust: user
code writes algebra over coordinate fields, and a `kernel!` proc-macro
compiles it — through an e-graph — into fused SIMD kernels. Extraction, the
step that picks one implementation out of the saturated e-graph, is driven by
a cost model. The default is `CostModel::latency_prior()`: a handwritten
table of per-op cycle estimates, applied additively over the chosen DAG by
dynamic programming.

The question this paper answers: **can a learned cost model, cheap enough to
run inside a proc-macro, extract measurably faster kernels than that table?**

The question matters beyond this codebase. Learned cost models beat
handwritten ones in adjacent domains (Halide's autoscheduler is the
canonical recipe: measured corpus, small regression net), and e-graph
extraction is exactly the setting where a better evaluation function should
compound — it is a search over huge numbers of slightly-perturbed candidate
states, on CPU, under a latency budget. That profile is chess evaluation's
profile, and chess solved it with NNUE: an incrementally updatable network
whose evaluation cost is proportional to the state delta. Porting that
solution to e-graphs was, per our own literature survey, unclaimed
territory.

The honest answer up front: **no — and the "no" is informative.** After we
removed every measurement defect our harness could find in its own earlier
runs (a 41.67× timebase error, a train/deploy featurization skew, and three
further self-invalidated runs' worth of correctness-gate failures), the
learned model's end-to-end deficit shrank monotonically — 1.0389 → 1.0181 →
1.0037 — and landed exactly at parity. The residual the model could exploit
had been *our own measurement error*, and once it was gone, there was almost
nothing left: the objective is nearly additive (a bare op count achieves
Spearman ρ = 0.949 on the same corpus where the table achieves 0.944),
dynamic programming over an additive table already optimizes it exactly
(over unfolded tree cost, which §6 distinguishes from DAG cost), and the
non-additive residual sits at the per-pair measurement floor. Both pre-registered model-side levers were then pulled —
a contrastive ranking objective aimed at the model's measured behavioral gap
(§5.3), and unfreezing the per-op embeddings so the model could learn cost
structure at all (§5.4). Each demonstrably moved the thing it targeted, and
each made the end-to-end result significantly *worse*. The levers are
exhausted; this pre-registered honest-negative is the deliverable.

Contributions:

1. **A parity result stated to its noise floor** (§5.2): a well-calibrated
   learned cost model ties a measured latency table at e-graph extraction,
   with CIs, leave-one-out ranges, and an A/A floor, on a family-held-out
   corpus, from a harness that had just demonstrated it will invalidate its
   own runs.
2. **Two mechanism-level refutations** (§5.3, §5.4): closing the model's
   conservatism gap on estimate-op substitutions does not close the speed
   gap (57.2% of conservative kernels moved toward static's count, 45.5%
   matched or exceeded it; 39.2% of those with a timing outcome won); and
   the
   best-calibrated checkpoint of the program — the first with trained
   embeddings — is confidently slower end-to-end than its frozen-embedding
   predecessor.
3. **A negative architectural finding with a measured positive twin**
   (§5.6): e-class candidates are restructured subtrees, not piece moves —
   at 44.9% median feature delta, incremental evaluation has a ceiling of
   ~2× at extraction — while saturation's state deltas (91.1% no-ops;
   0.14% median otherwise) are where the incrementality economics actually
   hold.
4. **The harness** (§4): the measurement discipline that makes 1–3
   trustworthy, presented as a reusable checklist, including the pitfalls
   we fell into first.

## 2. System

### 2.1 The compiler pipeline

`kernel!` lowers user algebra to `ExprArena` (a DAG arena; the sole IR),
builds an e-graph, saturates under a rewrite library (associativity,
FMA fusion, estimate-op substitutions, algebraic identities), and extracts.
Two extraction policies exist:

- **static** (default): DP over the DAG with the additive per-op latency
  table.
- **nnue** (opt-in, `PIXELFLOW_NNUE_WEIGHTS`): seed from the static choice,
  then a refinement loop proposes per-e-class alternative choices and
  re-scores full candidate selections with the learned model, keeping
  improvements.

A third, **no-swap**, policy (compile the input form as written, no e-graph)
exists purely as the measurement baseline for how much extraction itself is
worth.

### 2.2 The learned model

The extraction head is NNUE-shaped, adapted from game engines:

- **Factored per-op embeddings** (K = 32 dims per op), initialized from the
  latency prior — the same move chess NNUE made going from HalfKP to
  factored feature sets, replacing an earlier O(ops²) one-hot scheme.
- **An edge accumulator**: a fixed-size vector accumulated over the DAG's
  parent→child edges, combining parent and depth-shifted child embeddings
  multiplicatively. We deliberately describe this as a **learned
  multiplicative feature map with permutation-based asymmetry**, not as a
  Vector Symbolic Architecture, although the construction is VSA-derived:
  at K = 32, bundling hundreds of edges, the VSA capacity analysis (Plate;
  Kleyko) gives no retrieval guarantee — similarity to any one component
  decays like 1/√N and reliable capacity is linear in D. Nothing in our
  pipeline ever unbinds; a trained decoder reads the bundle. Claims about
  the representation must therefore not import capacity intuitions from the
  VSA literature, and ours don't.
- **A small MLP head** mapping the accumulator to predicted log-ns.

**One forward path.** An earlier version of this system trained and served
through *different* featurizations (a deduplicated arena walk at training
time vs. a choices-backed walk with variance features at serving time, plus
a feature-slot collision where one weight row served two domains). Those
skews were unified before any result in this paper: one walker
(`EdgeAccumulator::from_cost_dag`), one input vector, and a parity test
(`train_and_deploy_feature_paths_agree`) that pins train and deploy to the
same bytes. Every number below post-dates that unification.

**Embeddings train as of Round 3.** For Rounds 0–2b the embedding gradient
path had no caller — a defect, not a design — so the model could only
linearly reweight frozen per-op features distorted by weight decay. The fix
(landed 2026-08-31) routes every feature emission through one typed edge
stream (`CostEdge`/`EdgeTrace`/`EdgeSink`, a single emission point), makes
gradient staleness unrepresentable via live-realized traces, validates the
embedding gradients by finite differences, and exempts embeddings from
decay/projection so the latency-prior initialization in dim 0 is preserved
rather than eroded. Round 3 (§5.4) measures the effect of this lever.

A saturation-guidance head (the "Guide") exists behind a single-trait
contract but is untrained and inert; it plays no role in this paper's
results (see §9 for its measured economics).

### 2.3 Training

Supervised regression on JIT-measured latencies (an earlier RL/self-play
loop was audited and removed; nothing here uses it). The corpus is minted by
a seeded generator whose op distribution is weighted toward real shader
workloads, JIT-compiled, and measured by the harness of §4; labels are
median-of-samples, drift-normalized nanoseconds. Cold starts only — no
warm-started checkpoints anywhere in this paper. Round 3's cold start is
additionally **seeded from the latency prior**: a random embedding
initialization trains to good calibration but a measurably worse extraction
policy (§5.4), so the prior-seeded initialization is part of the recipe.
Cold-start training on 1k → 50k synthetic-augmented samples moves DEV MAE
from 0.162 to 0.133 log-ns and Spearman ρ from 0.9860 to 0.9887 (§5.1) —
the head is not underfit, and its ranking quality saturates well above the
static table's.

## 3. Experimental setup

**Corpus tiers.** Three tiers, split **by generator family** (band × seed
stream), never by random draw over one generator — with the fence key
computed from the same abstraction the model's features see (the
feature-quotient), so an expression the model has "effectively seen" cannot
appear on the certifying side merely because a literal differs:

- **TRAIN** — synthetic + mined expressions; grows.
- **DEV** — family-held-out; drives every selection decision in this paper.
- **FINAL** — publication-only: 17 named kernels (5 original production
  kernels + 12 ShaderToy/iquilezles.org ports, Appendix A) + one synthetic
  family. Guarded in code: the FINAL evaluation path refuses to emit a
  selection verdict at all. The tiers hold 3359 (TRAIN) / 784 (DEV) / 129
  (FINAL: 17 named + 112 synthetic band-12) expressions; the capstone
  mint's numeric quarantine checked 4324 expressions and excluded 52
  (1.203%), with **zero JIT-vs-oracle miscompiles**, and mean bound
  coverage 78.1% over all surveyed expressions.

Corpus identity is content-hashed and carried on every journal line. The
TRAIN and DEV tiers are **byte-identical across Rounds 2a and 3**
(`train=c5b6df3a…`, `dev=1daa131a…` on both runs), so those two rounds are
directly comparable kernel-for-kernel. The FINAL tier was re-minted after
the codegen refactors that landed between the rounds (identity
`89c48822…` → `90b84039…`), again with zero miscompiles — the re-mint's
oracle pass doubling as a regression check on the refactored backends.
Round 2b used an intermediate regenerated DEV (1120 kernels); §5.3 carries
the comparability check.

**The FINAL measurement has not run.** The FINAL tier exists, is
quarantined, and is fenced from selection; it was deliberately left
untouched through Round 3. Every end-to-end number in §5 is DEV-tier. We
state this plainly rather than letting a corpus table imply otherwise.

**The 3-way benchmark.** For each DEV kernel: build and saturate the
e-graph once, extract under each policy, JIT-compile each extracted form,
verify it (§4.2), and measure it — interleaved, repeated (5 repeats),
randomized *policy-arm* order under a recorded seed, with bootstrap CIs
(10k resamples)
over per-kernel nnue/static ratios, a leave-one-out geomean range, and an
A/A noise-floor run of the same policy against itself. Verdicts are emitted
by a gate pipeline that checks correctness first, censoring second, margin
last (§4.4). Saturation runs under a fixed budget (40 rounds here);
saturation spends its budget and stops — quiescence is a diagnostic
condition, never a certified closure, and no claim in this paper depends on
one. Environment: aarch64 macOS, baseline ISA, release profile —
**with** fused multiply-add: the aarch64 emitter lowers every
`ResolvedOp::FusedMulAdd` to `FMLA` (`emit/aarch64.rs:2131`). Earlier drafts
said "no FMA", reading `environment_fingerprint`'s `target_feature="fma"`
flag, which is x86-specific and does not describe what this target emits.
That matters here specifically, because `MulAdd` substitutions are one of
the axes §5.3's mechanism analysis is about —
one machine, stated as a limitation (§8).

**Decision band.** Pre-registered: promote only on >+5% DEV geomean with
the CI clear of the band; iterate within ±5%; a kill gate (five clean
rounds without a win) pivots to the honest-negative paper. This document is
that pre-registered fallback, executed.

## 4. Methodology: a harness you can lose trustworthily on

The results in §5 are negatives at the 1% scale measured on a machine whose
background daemons alone cause 11–19% drift. The harness is what makes such
numbers meaningful, and it was built by auditing our own earlier harness
and cataloguing every place gradient descent could exploit it (the full
audit is `docs/results/2026-08-05-bench-harness-integrity-audit.md`). The
design splits into five disciplines.

### 4.1 Sentinels: record and normalize, abort only on regime change

A fixed reference kernel is benchmarked on a cadence throughout every run.
Early designs treated it as a tripwire (fail at >10% drift); the first
end-to-end run showed macOS's post-build daemons reliably produce 11–19%
drift, which would have made every run red. The revised discipline:
**varied data absorbs contention noise the way more repetitions do,
provided benchmark order is randomized so drift decorrelates from
expression structure.**

That proviso is not met for the full-DEV runs this paper reports.
`subsample` returns the tier unchanged when `max_kernels == 0`, and Phase B
walks `prepared.iter()` in that same corpus order on every repeat; the
recorded `ORDER_SEED` shuffles only the four policy arms *within* each
kernel. So arm-versus-arm comparison is protected — which is what the
paired ratios rest on — but corpus-ordered families occupy the same
temporal positions in all five repeats, and the drift defense above does
not hold at the family level. Shuffling the kernel traversal per repeat is
the fix; it was not done here. So the sentinel *records* — local sentinel ns and a
normalization factor accompany every batch, labels are denominated in a
single session clock via that factor — and *aborts* only on regime change
(≥50% drift: the E-core-migration / thermal-step class that genuinely
poisons everything after it). The paired, interleaved 3-way protocol
additionally carries its own A/A floor, which is how a ±0.07% floor coexists
with a machine that drifts 11–19%.

Alongside the sentinel: QoS pinning, timer-tick floor auto-scaling (one
`mach_absolute_time` tick is 41.67 ns; timed samples are scaled until they
span ~100 ticks), identity-kernel call-overhead subtraction, dispersion
(IQR) recorded per label, dependency-chain serialization so the harness
measures latency rather than perfectly-overlapped throughput, and a
per-expression plausibility floor — a measurement faster than its op-count
bound is a harness bug and fails loudly.

### 4.2 Correctness under the algebraic-validity contract

pixelflow is a math library over the reals, not an IEEE-bit-identity
library: an algebraically valid rewrite may legitimately change computed
values at or near singularities. A correctness gate must therefore separate
*implementation error* from *contract*:

- **Same-form gate (hard):** an independent scalar-`f64` arena interpreter
  — not the JIT under test — evaluates the *same* form the JIT compiled,
  over a randomized 64-point grid with magnitude sweeps, under
  **compositional error bounds**: each op propagates a relative-error
  allowance, so the tolerance at the root is derived from the expression,
  not a global constant. Divergence here is a miscompile. Rate across every
  run in this paper: **zero** — including Round 3's 2352 gates run on
  freshly refactored codegen (§5.4).
- **Cross-form gate (conditioned):** extracted form vs. original, compared
  only at well-conditioned points (no near-singular intermediates, finite
  moderate outputs). Disagreement *there* means extraction changed the
  function — a bug. Divergence at ill-conditioned points is recorded as
  metadata, never as an alarm: a clamped or "fixed" value there would be a
  wrong answer wearing a right answer's clothes. The fraction of grid
  points at which bounds could be certified is itself reported (bound
  coverage: 74.2% on the Round 2a run; 74.0% on Round 3; 78.1% on the
  capstone mint survey).

This split closed the actual reward-hacking channel of the earlier harness,
which checked one splatted point at absolute tolerance 0.2 against the same
JIT backend — a sloppier-but-faster lowering could pass it and get rewarded.

### 4.3 Split hygiene: families and feature-quotients

Random splits over one generator are leakage. Tiers split by generator
family; the holdout fence is directional (a fence built from DEV∪FINAL keys
rejects TRAIN-stream candidates, and the type system encodes the
direction); and the fence key is the **feature-quotient** of the structural
key — computed from the same abstraction the features see — because "the
model has effectively seen this expression" is defined by the feature map,
not by syntax. Artifacts (checkpoints, mint sidecars, corpus tiers, journal
lines) carry **schema-derived identities**: a content hash of the layout
descriptor replaces hand-maintained version integers, after we shipped a
checkpoint format whose version tag was never bumped across a feature
meaning change. A stale artifact mismatches loudly and is regenerated,
never migrated, never silently consumed.

### 4.4 Verdicts that self-censor

The benchmark's verdict is produced by a gate pipeline whose order is part
of its meaning: **correctness first** (a tripped numeric gate ⇒ no timing
claim of any kind is derivable), **censoring second** (policy failure rate
>10% ⇒ the surviving kernel set is selected on policy success, so the
geomean cannot ground a directional claim), **margin last**. The journal
retains four runs that this pipeline refused to bless:

| ts | geomean | verdict |
|---|---|---|
| 1786294245 | 0.9285 | CENSORED — nnue failure rate 41.7% |
| 1786298796 | 1.0501 | CENSORED — nnue failure rate 15.0% |
| 1786298839 | 0.8815 | INVALID — cross-form divergence 16.7% |
| 1786732076 | 1.0181 | INVALID — cross-form divergence 12.3% |

Note what censoring bought: the two censored geomeans bracket parity from
*both* sides (0.93 would have been a publishable "win"). Round 2a (§5.2) was
the **first of five attempts to pass its own gates**. We regard the four
refusals as the methodology's strongest evidence.

### 4.5 Pitfalls ledger (the part we owe the reader)

Defects we shipped, found, and fixed — each one had produced a number
somebody briefly believed:

- **Timebase, 41.67×.** All pre-2026-07-20 absolute latencies were
  under-scaled by the mach timebase ratio (125/3). Every label and weight
  file from before the fix was retired; the 2026-07-08 "NNUE loses by 6.7%
  at 31× extraction cost" result predates it and is quoted only as lineage.
- **Train/deploy featurization skew (×3).** Training and serving used
  different accumulator walks and a colliding feature slot (one `w1` row
  trained on log-count scalars, served variance fractions); unified into
  one path with a parity test. The fix moved DEV MAE from 0.975 to 0.181
  log-ns and prediction bias from −0.886 to +0.020 — most of the model's
  apparent badness had been our skew.
- **`node_count` leak lineage.** An earlier head consumed search-metadata
  scalars (including node count) that leaked the label; the live head's
  forward path (`forward_expr_only`) takes expression features only.
- **Benchmark-loop bias.** A loop counter once inflated all measurements
  ~100× and biased small kernels; eliminated by full unrolling with the
  additive-cost rationale documented, and separately the harness measures
  the same code object it correctness-checks.
- **Censoring bugs.** Zero-ns (constant-folded) results were once dropped
  as failures — censoring the fastest expressions from training; failures
  now go to a structured exclusion sidecar with an end-of-run rate
  assertion (that sidecar is where §3's 52/4324 comes from).
- **Embeddings frozen but decaying.** The embedding gradient path had no
  caller, so embeddings never trained, while weight decay eroded their
  initialization — the model could only linearly re-read distorted
  constants. That initialization was `ExprNnue::new_random` (He-scaled
  noise) for every round through 2b: the cold start did not seed from the
  latency prior until the Round-3 commit that §5.4 benches, so what decayed
  in Rounds 0–2b was random init, not a prior. Found by inspection after
  two measurement rounds missed it; **closed 2026-08-31** by the typed edge
  stream (§2.2), and the effect of closing it is measured in §5.4.

## 5. Results

### 5.0 Four threats to every interval in this section

Review of the harness against the claims (2026-09-01) found four ways the
numbers below are weaker than they read. None is repaired here — each needs
a re-run this paper cannot perform retroactively — so each is stated before
the results rather than after them. Two of the four cut *against* the
paper's own framing, which is why they matter.

**1. The reported ratios include call overhead.** `BenchResult::ns` is the
raw measurement; `adjusted_ns` (`raw.ns - call_overhead_ns`) is computed and
serialized but never aggregated — `bench_extraction_3way.rs:2589` pushes
`bench.ns * normalization`. So every geomean and CI here is
`(n + c) / (s + c)` with `c` = 4.272 ns, not `n / s`.

That is a bias *toward* the number 1, algebraically and on both arms: it
shrinks the reported effect whichever way the effect points. The paper's
central claim is a **tie**, and a tie is exactly the result this artifact
manufactures. So §5.2's parity should be read as an upper bound on how close
the two policies really are, and §5.4's regression as a lower bound on how
large it really is — the negative conclusion survives, but "ties the table"
is the reading most exposed. With 4.272 ns against a corpus containing many
short kernels, the correction is plausibly the same size as the 0.4–1.5%
effects being reported. Re-aggregating on `adjusted_ns` is a one-line change
and every interval here should be recomputed before any of them is quoted as
a measurement of the cost model alone.

**2. The two arms differ in search, not only in scoring.** The NNUE arm runs
`IncrementalExtractor::extract_choices_only` (refinement from
`Extraction::from_backfill`, top-k = 8); the static arm runs `egraph::extract`'s
exact DP. Those are different algorithms over different reachable sets, so
every A-versus-B number here measures *cost model plus search policy* and
cannot attribute the difference to the cost model alone. §4.1's protocol
describes seeding the learned arm from the static choices; the harness does
not do that. Until it does, "the learned cost model ties the table" is
properly "the learned extraction *policy* ties the table."

**3. The bootstrap resamples kernels, but the corpus's unit is the family.**
`training::split` defines the split unit as a `(band, seed)` generator
family — "never by random split", because draws within a family reuse
structural motifs — and the DEV tier is 56 families × 14 expressions.
Resampling ~716 surviving kernels as independent understates uncertainty by
whatever the within-family correlation is. The narrow Round-3 interval
carrying the "confirmed regression" verdict is the one most exposed; a
cluster bootstrap over families, with leave-one-family-out sensitivity, is
the correct estimator and is not what produced these numbers.

**4. Round 3 versus Round 2a was never tested directly.** Each checkpoint
has a CI against static; the two intervals ([1.0031, 1.0133] and
[0.9982, 1.0089]) overlap substantially, and they come from separate
sessions. "Significant against 1" for one and "not significant against 1"
for the other does not establish that the checkpoints differ from each
other. Statements below that Round 3 is a regression *from Round 2a*, and
that the trained-embeddings lever is therefore exhausted, need a paired
interleaved comparison that was not run.

### 5.1 Calibration (the model is good, and gets better)

Round-3 cold-start learning curve (trainable embeddings, latency-prior
seed), family-held-out DEV (784 kernels):

| train samples (synthetic aug.) | DEV MAE (log-ns) | Spearman ρ | bias | σ-ratio |
|---|---|---|---|---|
| 4,354 (1k) | 0.162 | 0.9860 | −0.025 | 0.972 |
| 11,312 (8k) | 0.150 | 0.9874 | −0.019 | 0.972 |
| 35,151 (32k) | 0.143 | 0.9873 | −0.014 | 0.973 |
| 53,026 (50k) | **0.133** | **0.9887** | −0.007 | 0.975 |

The random-init ablation trains worse at matched budgets (1k: MAE 0.173,
ρ 0.9827; 8k: MAE 0.164, ρ 0.9856), and the frozen-embedding Round-2a
recipe topped out at MAE 0.145 / ρ 0.9875 at the same 50k budget — so the
trained embeddings and the prior seed both help *calibration*, measurably.

Unbiased, correctly dispersed, and better-ranked than the static table
(ρ = 0.9438) or a bare op count (ρ = 0.9486) — but **not on this corpus**:
both baselines come from Round 1's 392-kernel DEV tier
(`2026-08-17-egraph-vsa-nnue-research-notes.md` §1.5), while the table above
is Round 3's 784. Spearman is a property of the evaluated population, not of
the estimator alone, so this is a cross-population comparison and the margin
is not a measured head-to-head. Recomputing both baselines on the 784-kernel
tier is a cheap run that was not made; until it is, read the ordering as
indicative and the gap as unquantified. By every
intrinsic metric each successive model should win, and the best one should
win most. Neither happens, and §6 explains why the intrinsic metrics were
the wrong ones.

### 5.2 Round 2a: parity, measured to the floor

First fully-valid 3-way run (five repeats, interleaved, randomized order,
full DEV tier; frozen-embedding checkpoint):

| quantity | value |
|---|---|
| nnue/static geomean | **1.0037** (nnue 0.37% slower) |
| 95% bootstrap CI | [0.9982, 1.0089] — straddles 1.0 |
| pairs | 719 (276W / 409L / 34T) |
| per-kernel ratio median [Q1, Q3] | 1.0057 [0.9895, 1.0280] |
| leave-one-out geomean range | [1.0031, 1.0044] |
| A/A noise floor | ±0.07% |
| policy failure rate (static / nnue) | 7.65% / 7.40% |
| same-form miscompiles | 0 |
| cross-form divergence (well-conditioned) | 5.48% (< 10% gate) |
| static/noswap geomean | **0.5440** |

The entire CI sits inside the ±5% decision band: an approx-tie, and the
pre-registered Phase-2 gate fails — the static prior stays default. No
single kernel drives the estimate (LOO range spans 0.13%). Meanwhile the
last row is the perspective line: **e-graph extraction with the static
table is worth ~2× over not extracting at all** (0.5440 here; 0.5444 on the
Round 2b re-mint; 0.5433 on Round 3 — stable across corpora and code
revisions). The machinery pays; the learned head is the part that doesn't.

**Extraction overhead.** From the same run's per-kernel records: median
extraction time 192.6 µs (nnue) vs 14.0 µs (static) — ~14× at the median,
with a heavy tail (nnue mean 5.36 ms, p99 80 ms, vs static mean 92 µs).
In compile-budget terms the NNUE pass is ~13% of the e-graph pass and its
p99 is 4.5% of a blitz compile budget — tolerable, not free. A subsequent
scoping fix (bounding whole-e-graph work to the reachable set) removed
~15–16% of mean extraction time without changing any extracted form
(digest-verified), and did not change any number above. On Round 3 the
means are 3.60 ms (nnue) vs 83.7 µs (static), ~43× — same order, same
conclusion.

### 5.3 Round 2b: the contrastive attempt, and the first refutation

> **Trace caveat.** Every number in this subsection is `R2B` in
> [`NUMBERS.md`](NUMBERS.md): the Round-2b journal line was never committed
> and its worktree no longer exists, so these are carried from that
> session's draft rather than re-derived from a durable artifact. The
> §5.3 result is the paper's weakest-traced claim, and its trainer bug
> (below) is a second, independent reason to treat it as provisional.

Round 1's mechanism analysis had found NNUE to be the *conservative*
policy: on 12 re-inspected kernels it took 14 hardware-estimate-op
substitutions (`Recip`/`Rsqrt`/`MulAdd`/`Sqrt`) where static took 26, and
per-pair speed looked near-monotone in that substitution delta. The
pre-registered Round 2b lever: a contrastive/pairwise ranking loss over
**variant sets** — expressions equivalent by construction, the population
extraction actually ranks — to teach the model within-set order rather
than corpus-wide magnitude.

The training-side diagnostic did improve, reproducibly: within-set pairwise
accuracy rose at λ = 1.0 in 3 of 3 independent seed replicates (+0.038 /
+0.011 / +0.162), without costing DEV MAE or ρ; the checkpoint λ was chosen
by majority vote across those replicates *before* any end-to-end
measurement. The end-to-end result:

| | Round 2a (regression) | Round 2b (contrastive, λ=1.0) |
|---|---|---|
| geomean nnue/static | 1.0037 | **1.0153** |
| 95% CI | [0.9982, 1.0089] | [1.0097, 1.0213] — entirely above 1.0 |
| pairs (W/L/T) | 719 (276/409/34) | 1016 (333/631/52) |
| LOO range | [1.0031, 1.0044] | [1.0138, 1.0158] |
| A/A floor | ±0.07% | ±0.06% |
| static/noswap | 0.5440 | 0.5444 |

A statistically significant **regression**: the contrastive checkpoint
moved the CI clear of parity in the wrong direction. (The DEV corpus was
regenerated between rounds, 784 → 1120 kernels; the near-identical
static/noswap geomeans are the sanity check that corpus difficulty is
comparable, though they do not fully exclude composition effects.)

**The mechanism moved; the speed didn't follow.** A DAG-level count across
all 1120 DEV kernels: Round 2a's checkpoint was *not* conservative in
aggregate (8072 estimate-op instructions vs static's 7591 — the Round-0
n = 12 framing already fails at corpus scale) but was conservative on 187
individual kernels (16.7%). Round 2b's checkpoint moved strictly toward
static's substitution count on 107 of those 187 (57.2%), fully matching or
exceeding it on 85. Cross-referencing those 107 kernels against the run's
own measured timings: **38 wins, 53 losses, 6 ties** — 97 kernels, so
**39.2% wins among those with an outcome**. The remaining 10 carry no
recorded win/loss/tie, and the probe that produced this cross-reference was
run and discarded (§ NUMBERS.md, D2B), so why is no longer recoverable;
quoting 35.5% over all 107 would silently count them as non-wins. The
conservatism gap closed; the speed gap did not follow it. The Round-0
"near-monotone in estimate-op delta" mechanism is refuted — the
substitutions the static table takes are not, individually, where its
performance comes from.

**A caveat, reported plainly.** After the measurement, a deterministic
failure of the pairwise gradient check surfaced in the contrastive trainer
(one shared-trunk gradient entry off by 20% relative against a 5%
tolerance) — a real bug in `backward_pairwise`, in the code that trained
this checkpoint. The Round 2b regression is therefore evidence against
*this trained artifact*, and cannot cleanly separate "the contrastive
objective doesn't work here" from "this implementation of it was buggy."
The mechanism refutation above is unaffected — it compares measured
timings of measured substitutions, not gradients — but a fixed re-run is
required before the objective itself is pronounced dead. Per the
no-silent-failures rule, we publish the caveat with the number.

### 5.4 Round 3: trained embeddings, best calibration, confirmed regression

Round 3 pulls the last pre-registered model-side lever: the embedding
gradients (§2.2's defect, closed by the typed edge stream). For the first
time in the program the model can reshape per-op cost structure instead of
linearly reweighting frozen features. Two cold starts were benched
end-to-end on the full DEV tier (identical corpus, protocol, and seeds as
Round 2a):

| | Round 3, random init (8k) | Round 3, latency-prior seed (50k) |
|---|---|---|
| geomean nnue/static | 1.0118 | **1.0082** |
| 95% CI | [1.0058, 1.0182] | [1.0031, 1.0133] — entirely above 1.0 |
| pairs (W/L/T) | 717 (208/487/22) | 716 (235/449/32) |
| per-kernel median [Q1, Q3] | 1.0110 [0.9948, 1.0306] | 1.0066 [0.9934, 1.0239] |
| LOO range | [1.0102, 1.0126] | [1.0077, 1.0089] |
| A/A floor | ±0.04% | ±0.07% |
| policy failure rate (static / nnue) | 7.65% / 7.78% | 7.65% / 8.16% |
| static/noswap | 0.5437 | 0.5433 |
| DEV calibration of this checkpoint | ρ 0.9856 / MAE 0.164 | **ρ 0.9887 / MAE 0.133** |

The emitted verdict, verbatim: *"NNUE approx-ties static latency prior: the
ENTIRE 95% CI sits inside the ±5% decision band (geomean −0.82%, 95% CI
[−1.33%, −0.31%] over 716 kernels, A/A noise floor ±0.07%) — Phase 2 gate
FAILS per the plan (no meaningful win): keep the static prior as default,
NNUE opt-in only, iterate (plan 4.3)."*

Three readings, in decreasing order of importance:

1. **The lever worked on the objective and backfired on the outcome.** The
   50k prior-seeded checkpoint is the best-calibrated model this program has
   produced (ρ 0.9887 vs Round 2a's 0.9875; MAE 0.133 vs 0.145), and its CI
   sits entirely on the slow side of parity — a confirmed regression from
   Round 2a's tie (1.0037, CI straddling 1.0) on the byte-identical corpus
   under the identical protocol. So does the random-init run in the same
   table, and so did Round 2b, though §5.3's trainer bug binds that verdict
   to its artifact; what is new here is not the sign but that the
   best-calibrated checkpoint the program has produced carries it. Note
   what this does *not* establish: Round 3's interval and Round 2a's overlap
   and were measured in separate sessions, so "a regression from Round 2a"
   is not tested here (§5.0, threat 4). The weaker claim the data supports
   is that trained embeddings did not move the program off the slow side of
   parity — not that they are worse than frozen ones.
2. **The prior seed and the training budget move together, and this pair
   cannot separate them.** Random-init trained to respectable calibration
   (ρ 0.9856) but a worse policy (1.0118); the prior-seeded checkpoint is
   0.36 points of geomean better. The two runs differ in *both*
   initialization and label budget (8k vs 50k), so that gap is not
   attributable to the seed: it is equally consistent with 50k labels
   simply buying a better policy than 8k. Separating them needs a
   random-init 50k run or a prior-seeded 8k one, neither of which was
   benched — the pre-registered lever was "train the embeddings at all,"
   and the budgets are what each cold start happened to be run at. What the
   seed provides — a per-op cost ordering consistent with hardware — may or
   may not be recoverable from corpus labels alone; this program does not
   answer that.
3. **Zero miscompiles on refactored codegen.** Between Rounds 2a and 3 the
   backend was substantially refactored (assembler/encoder unification, a
   retyped AVX2+FMA tier, the edge-stream rework). Round 3's same-form gate
   ran 2352 gates (1568 on extracted forms) with **0 miscompiles** and 0
   compile failures; cross-form divergence 5.87% of extracted forms (under
   the 10% gate; 559 ill-conditioned disagreements recorded as metadata,
   not counted). That rate is `GateTally::cross_form_rate` — 92/1568, over
   *every* extracted-policy gate, the 32 that bounded no grid point
   included; excluding them here would read 5.99%. They are excluded from
   the *bound-coverage* mean only — `same_form_rate` also divides by
   `GateTally::gates`, which `record_noswap`/`record_extracted` increment
   before the outcome is inspected, so all 32 sit in the 2352-gate
   same-form denominator too (its rate is unchanged only because the
   numerator is zero). Mean bound coverage 74.0% (worst kernel 1.4%). The FINAL
   re-mint's oracle pass was likewise clean. The correctness harness is
   doing double duty as the refactor regression net.

**Mechanism: where the trained embeddings changed decisions.** Against the
Round-2a checkpoint on the identical saturated e-graphs, 174 kernels
flipped win↔loss.

Two caveats on this analysis, both structural. First, the flips are not
measured together: `inspect_flip` loads both checkpoints and prints
extracted forms, node counts and predicted costs, but never compiles or
times anything (it contains no benchmark harness at all), so a "flip" here
splices the timing sign from the Round-2a session against the sign from the
Round-3 session. Given §5.2's own finding that most per-kernel differences
are smaller than their repeat spread, and the contention recorded during
Round 3, an individual sign flip can be measurement noise rather than a
consequence of the changed decision. Second, the predicted-cost comparisons
below score each checkpoint only on the form that checkpoint chose — two
models on two arenas. That shows the two disagree; it does not show that
Round 3 *ranked* the fusion below its alternative, which is a within-model
comparison of both forms and was not run. Read the individual flips as
illustrations of the disagreement, not as evidence of its direction; the
aggregate node-delta asymmetry does not depend on either caveat. The asymmetry is directional: of the 109 kernels that
flipped **to losses**, Round 3's extracted tree is *larger* (node count) in
72 (66%), smaller in 13 (12%), unchanged in 24 (22%); of the 65 flips to
wins, larger in 24 (37%), smaller in 26 (40%). Mean node delta over all
flips: +0.81 (median +1). Inspection of individual flips shows the trained
model declining fusions the frozen model and the static table both take —
e.g. `dev_b09_f03_00278` (W→L): Round 3 declines a
`recip`+`recip`+`mul_add` square-reciprocal fusion and keeps a direct
`pow(−2)` form, predicting 4.51 (log-cost units) where Round 2a predicted
3.99, and losing the measurement. The failure is not uniformly
"bigger trees": on `dev_b27_f03_00615` and `dev_b33_f07_00773` Round 3
chose *smaller* trees with higher predicted **and** higher measured cost,
and on `dev_b01_f07_00109` its tree nearly doubles (11→19 nodes) while its
own predicted cost *drops* (2.71→2.51) and the measurement flips to a loss
— the trained model's cost surface visibly disagrees with the hardware on
exactly the decisions extraction asked it to make. Calibration-in-aggregate
and correctness-at-the-margin are different quantities; Round 3 improved
the first and degraded the second.

**Run conditions, disclosed.** This run violated our own machine-quiet
protocol: a sibling experiment's binary (`guide_headroom`) was running at
100% CPU in another worktree throughout (load average 9.45/13.23/14.11,
ps-confirmed; AC power, battery 83% charging). The defenses held by
design rather than by circumstance: the interleaved paired protocol
measures both policies under the same contention, the A/A floor computed
*within this run* is ±0.07% — indistinguishable from Round 2a's
quiet-machine floor — and
the sentinel (calibration 7.446 ns over 231 samples; measured call
overhead 4.272 ns) tripped no regime-change abort. Single clean run, no
retries. We report the contention rather than pretending the protocol was
followed.

### 5.5 The historical arc: parity is an attractor

| round | nnue/static | status | what changed before it |
|---|---|---|---|
| 2026-07-08 | 1.0669 | pre-timebase-fix; unsplit corpus | — |
| Round 0 | 1.0389 (7W/18L) | small n | timebase fix, tiers, harness v2 |
| Round 1 | 1.0181 [1.0106, 1.0254] | INVALID (cross-form 12.3%) | train/deploy skew unified |
| Round 2a | 1.0037 [0.9982, 1.0089] | **valid — the tie** | 3 further skews + gates fixed |
| Round 2b | 1.0153 [1.0097, 1.0213] | valid (§5.3 caveat) | contrastive objective |
| Round 3 (random init) | 1.0118 [1.0058, 1.0182] | valid | embeddings train |
| Round 3 (prior seed) | 1.0082 [1.0031, 1.0133] | valid | + latency-prior cold-start seed |

The arc has two halves and one summary. First half: every *measurement
defect we removed* shrank the gap — the signal the learned model had been
exploiting was largely our own skew, and deleting it marched the result to
parity. Second half: every *model improvement we made* — a ranking
objective that demonstrably improved ranking, an embedding pathway that
demonstrably improved calibration — moved the result away from parity, and
always to the slow side. A harness good enough to delete our defects is
also good enough to certify that what remains is a tie, and that the tie is
hard to leave in the desired direction.

### 5.6 Incrementality: the premise fails at extraction, and holds one layer up

NNUE's reason to exist is that a chess move perturbs a handful of features
out of thousands, making O(Δ) re-evaluation ~50× cheaper than recompute.
Measured on our workload: the median edge-multiset symmetric difference
between a base DAG and an e-class candidate is **44.9%** (p25 17.6%, p75
100%, p90 180%); only 11% of candidates change fewer than 10% of edges;
stripping the depth encoding barely moves the distribution (51.5% vs 56.8%
mean), so it is not a positional-encoding artifact. E-class alternatives
are restructured subtrees, not piece moves. Consistently, the extractor —
which ships a complete incremental add/remove API — performs **zero**
incremental updates in production traces (2,323 full rebuilds, 0 deltas).
A perfect O(Δ) implementation would buy roughly 2×, not 50×; extraction
overhead is in any case second-order (§5.2). The architectural argument
our own survey listed as unclaimed novelty rests on a premise this
workload does not satisfy. We report that as a result, not a caveat.

The same measurement, pointed at the Guide's scoring object instead — the
monotonically growing e-graph during *saturation* — finds the opposite
economics (4.49M rule applications over the production batched algorithm):
**91.1% of applications change zero nodes and zero edges** (idempotent
re-fires), and among state-changing applications the median edge-delta is
**0.14%** of the graph (p90 0.79%) — an implied ~728× incremental saving,
deltas ~320× smaller than extraction's 44.9%. The Stockfish argument does
transfer; it was aimed at the wrong layer. (The same scoping run found the
bigger cost lever first: 90.4% of scored candidates commit nothing, so
candidate deduplication outranks accumulator maintenance. §9 takes this
up.)

## 6. Why the table is (so) hard to beat

**The objective is nearly additive, and DP over an additive table already
optimizes it exactly.** A bare count of
transcendental/divide ops reaches Spearman ρ = 0.9486; the handwritten
table reaches 0.9438; the NNUE reaches 0.9887. The first two are Round 1's
392-kernel DEV tier and the third is Round 3's 784, so read them as three
points about how little headroom there is above a size proxy, not as a
ranked head-to-head on one population (§5.1). Corpus-wide
ranking is saturated by expression size — nearly all between-expression
variance is "how many expensive ops" — and extraction with the static
table already optimizes an additive quantity by dynamic programming.

The argument this section makes does not depend on the margin: it needs
only that a bare op count already sits near the ceiling, which is true on
whichever tier it is measured.

Two qualifications on "optimally", both of which narrow the claim without
changing its force. The DP recurrence in `egraph::extract` sums each
child's full best cost *per reference*, so a subexpression reachable twice
is charged twice — while `choices_to_arena` afterwards materializes each
shared e-class once. The quantity minimized exactly is therefore the
**unfolded tree cost**, which coincides with the executed DAG's additive
cost only when nothing is shared; and `repair_choices_well_founded` may
replace recorded choices after that cost is computed. So the baseline is an
exact optimizer of a close proxy, not of the additive DAG cost itself. It
is still not a strawman — it is ~95% of the truth applied by an exact
optimizer of something near it — and closing this gap needs a DAG-aware
extractor, which would raise the bar the learned model has to clear rather
than lower it.

What remains learnable is the non-additive residual: schedule effects —
ILP, port contention, dependency-chain latency. The deployed
`EdgeAccumulator` is not a flat edge multiset: half of it is depth-encoded,
binding each parent/child embedding to an effective depth
(`tree_depth * MAX_ARITY + child_slot`) through a sinusoidal positional
encoding, and the DAG walker emits distinct `Var` reload edges for shared
references. Graphs whose flat edge counts agree but whose depths, child
positions or sharing differ therefore do *not* produce identical inputs.
The limitation is weaker than "cannot see topology by construction," and
weaker is the honest word: aggregation into a fixed-width accumulator
discards enough structure that two schedules with genuinely different port
pressure can still land close together, and the head has no term for
contention regardless. The model had blurry eyes, not no eyes, for the only
part of the problem left — and §5.4 measures that blur costing more than it
buys.

**The decisions sit at the per-pair noise floor.** Derived from the
Round 2a run's own per-kernel records: the median |nnue−static| per-kernel
delta is 2.5% (IQR 1.0–5.4%), while the median per-kernel repeat spread
(range over the five repeats as a fraction of the median, static policy) is
9.0% — **84% of individual paired decisions are smaller than their own
kernel's measurement spread** (under the stricter IQR-of-repeats definition
the spread is 3.4% and 61% of decisions still sit under it).

Both fractions compare a policy difference against the spread of *individual
measurements*, which is not an uncertainty interval for that difference: the
range in particular is an extreme statistic and inflates with repeat count,
so these numbers overstate how many decisions are genuinely unresolved. The
estimator the claim wants is a per-kernel paired interval or test built from
the five repeats, reporting the fraction whose paired effect does not clear
zero; that needs the per-kernel repeat rows, which are in the uncommitted
`D3` artifact (§5.0). Read 84%/61% as an upper bound on the unresolved
fraction, not a measurement of it. Only pairing and aggregation over 719
kernels resolves the 0.37% mean effect against the ±0.07% A/A floor. A
cost model asked to out-rank the table must make thousands of decisions
whose individual ground truth is at best marginally measurable in isolation;
the labels it trains on carry the same floor. ("Unmeasurable" is stronger
than the statistic above supports — see the caveat.) This is a structural argument that
marginal cost-model improvements here are undetectable-by-construction —
and it generalizes to any extraction setting where candidate variants
differ by a few instructions.

**Intrinsic metrics do not transfer to search outcomes — three times.**
Round 2b improved held-out within-set pairwise accuracy on *minted*
variant sets in 3/3 seeds, then regressed end-to-end. Round 3 produced the
program's best corpus calibration (ρ 0.9887) and a confidently slower
end-to-end result. And Round 3's flip analysis (§5.4) shows
the failure at the decision level: the trained model's predicted costs
disagree with measured latency on precisely the marginal candidates
extraction asks about, up to preferring a near-doubled tree it scores as
cheaper. The extractor does not sample from any evaluation distribution —
it walks toward the model's own minimum, concentrating probability mass
exactly where the model is wrong-and-optimistic (the classic
proxy-consumption gap). A ranking metric on any fixed distribution is an
unreliable predictor of search outcome under the model being ranked.

**And the incrementality economics are absent at this layer** (§5.6), so
the setting does not even reward the architecture's signature capability.
The chess analogy fails quantitatively at both ends: the baseline
evaluation function is already ~95% of the truth, and the state deltas are
45%, not 0.1%.

## 7. Related work

**Extraction.** SmoothE (ASPLOS 2025) makes extraction differentiable via
per-class softmax relaxation, built for complex cost models on GPU;
e-boost (2025) warm-starts ILP from parallel heuristic extraction; DAG
extraction is NP-complete, and extraction-gym provides shared baselines.
We cite these as the extraction-quality SOTA to position against; at
workshop scope we situate rather than benchmark (§8). Our result is
complementary: it characterizes when the *cost model* (not the extractor)
is the binding constraint — under an additive objective, it isn't.

**Learned cost models.** Ithemal/BHive and GRANITE are the basic-block
throughput lineage — GRANITE's GNN validates graph-structured features,
and its offline position suggests the teacher-student route (§9) rather
than serving a GNN at proc-macro time. Halide's 2019 autoscheduler is the
acknowledged recipe template (measured corpus + small net) that our
pipeline instantiates; TenSet/MetaSchedule scale the same recipe. Our
contribution to this line is a carefully-measured *boundary case*: a
domain where the recipe's preconditions hold (closed op set, cheap ground
truth) and the learned model still cannot pay, because the handwritten
baseline sits inside the noise floor of the objective.

**Guided saturation.** Tensat scales equality saturation to tensor
programs (the setting where extraction-cost modeling first bit hard);
Omelette and MCTS-GEB apply RL/MCTS to rule application; Guided/Sketch-
Guided EqSat (POPL 2024) decomposes saturation toward targets; ML-Guided
EqSat (EGRAPHS 2025) learns the guides; Isaria phases rules; Ruler infers
them. Our provenance/hindsight-labeling substrate (union journal →
load-bearing unions → supervised targets) is our forward-looking bet in
this line — supervised, offline, never REINFORCE — with HER as the
hindsight-relabeling ancestor. EggMind (2026) caches proof-derived motifs
for an LLM agent; ours differs in producing dense supervised labels for a
sub-millisecond in-process guide, not context for an LLM.

**What we do not claim.** Learned cost models for extraction per se,
RL-guided rule application per se, and measured-corpus regression heads
are all occupied territory. We also explicitly withdraw the "VSA
featurization" novelty claim our own earlier survey made (§2.2): at
K = 32 with no unbinding, the honest name is a learned feature map.

## 8. Limitations

- **One machine, one ISA.** All numbers: aarch64 macOS, baseline ISA
  level, release profile, FMA present as `FMLA` (see §4.1 — the recorded
  `target_feature="fma"=false` is an x86 flag, not a statement about this
  target). The per-op latency landscape differs
  across the ISA matrix (several ops are documented as
  platform-divergent), and both the table and the learned model would
  shift. Nothing here licenses a cross-ISA conclusion.
- **FINAL tier not consumed.** The publication tier (129 kernels incl.
  the 12 withheld shader ports) is minted, quarantined, fenced, and was
  deliberately left untouched through Round 3. All end-to-end claims are
  DEV-tier; the FINAL pass is the remaining pre-publication action.
- **Corpus lineage.** DEV/TRAIN are outputs of one seeded generator with
  a shader-derived op mix; family fencing prevents leakage but not
  monoculture. The 12 real-shader ports mitigate this in FINAL only. The
  DEV corpus was regenerated for Round 2b only (784 → 1120);
  static/noswap stability (0.5440 / 0.5444 / 0.5437 / 0.5433 across all
  four valid runs) is our comparability check, and Rounds 2a and 3 share
  a byte-identical DEV tier.
- **Shader ports are formula ports, not verbatim GLSL** (no textures,
  buffers, or data-dependent loops in the DSL; raymarch loops became
  closed-form fields). Attribution and per-kernel simplification notes:
  Appendix A. No GLSL was copied; licenses are treated conservatively as
  CC BY-NC-SA where unconfirmed.
- **Workshop-scope baselines.** SmoothE/e-boost are cited and situated,
  not run; the full extraction-gym comparison is the full-venue upgrade.
- **Checklist deltas at workshop scope.** Against the program plan's own
  reviewer checklist: top-k regret and per-accumulator-section ablations
  were not run, no extraction-time quality-vs-budget curve was measured
  (only the overhead distribution of §5.2), and calibration error is
  reported as MAE in log-ns rather than MAPE. The flip analysis (§5.4)
  and the per-pair floor analysis (§6) are the decision-level evidence
  offered in their place.
- **Round 2b carries a trainer bug** (§5.3): that regression verdict
  binds to the artifact, not yet to the objective.
- **Round 3 ran on a contended machine** (§5.4): a sibling CPU-bound
  experiment violated the machine-quiet protocol. The paired interleaved
  design and the run's own A/A floor (±0.07%, matching the quiet-machine
  floor) are the evidence the result survived it; we disclose rather than
  discard.
- **Battery-honesty note.** The capstone corpus statistics were recorded
  with the report machine at 17% battery, discharging — noted because
  thermal/power state is part of this paper's threat model; no timed
  measurement was taken in that state.

## 9. Future work

**Where the headroom actually is.** Extraction with the static table is
already worth ~2× (static/noswap 0.54, four runs); the cost model is at
parity and both model-side levers are spent; therefore the leverage is in
what gets *into* the e-graph, not how it is scored. The Phase-3 scoping
round (2026-08-30/31, `docs/plans/2026-08-31-guide-design-revision.md`)
measured the Guide's economics before committing to it, and they are the
mirror image of extraction's:

- **Oracle headroom exists at two bounds that disagree usefully.** Over
  800 stride-sampled corpus expressions (8.73M rule applications), the
  hindsight-labeler bound says the median expression needed only 38.2% of
  its saturation's applications (implied 2.6×); the strict
  on-the-derivation-path bound says 2.9% (implied 34×). Per rule, the two
  bounds correlate only moderately (Spearman ρ = 0.35 over the 55 fired
  rule instances; an earlier draft's "ρ ≈ 0.02, almost independent" was an
  arithmetic error, corrected 2026-09-01 in the source document) and split
  cleanly by rule *class*: structural/congruence rules score 60–85% under
  the labeler bound and ~0% under the strict bound, while
  numeric/transcendental rules land in the same 6–17% range under both. A
  Guide trained on labeler labels would learn to prize exactly the rule
  class the strict bound says contributes almost nothing to the extracted
  expression — that per-class split, not the correlation number, is the
  finding about hindsight-provenance labeling.
- **The incrementality economics hold here** (§5.6): 91.1% exact no-op
  applications, 0.14% median edge-delta among state-changers (~728×), vs
  extraction's 44.9% (~2×). The accumulator architecture built for the
  wrong layer is the right architecture for this one.
- **The bigger discovered cost is candidate churn:** 90.4% of scored
  candidates commit nothing, so deduplication of already-seen candidates
  outranks incremental accumulator maintenance in expected savings.
- **A null result, kept:** oracle-filtered budget curves showed nothing —
  97.8% of the probed corpus quiesces or hits the class-count safety cap
  before the coarsest budget checkpoint (quiescence being a diagnostic,
  not a certified closure; the optimizer is budget-only by design). The
  curve methodology needs recalibration toward capped/deep expressions
  before it can discriminate guided from unguided saturation.

The plan is two-stage: cold-start the Guide on strict labels while
tightening union-causality in parallel, evaluated as quality-at-budget
(anytime curves), never as "reaching the optimum faster."

**If the extraction cost model is ever revisited:** (1) fix the pairwise
gradient bug and re-run Round 2b to separate objective from artifact;
(2) predict the **residual over the DP-additive table** rather than
absolute cost, with schedule-aware features (dependency depth, port
pressure) that survive aggregation into a fixed-width accumulator — the
additive part is already solved exactly, over unfolded tree cost, and a
DAG-aware extractor would close the remaining gap in the baseline rather
than in the model; (3) **top-k rerank**: extract k candidates under
the table and rerank with the learned model, confining the model to
decisions the table cannot make; (4) use SmoothE-style relaxation as a
*training* mechanism (differentiate through extraction against measured
outcomes) before using it to serve. A register-allocator overhaul is also
in progress; the latency prior must be re-measured and the corpus
re-minted behind it, and no run may be compared across that boundary (the
journal's source-rev config keys enforce the separation).

**The north star** remains self-hosting: the cost model written as a
pixelflow kernel, optimized by the e-graph it serves, its accuracy on
itself a directly measurable quantity. NNUE inference is already
pull-based — an accumulator update is a coordinate warp, a matmul is a
reduce over a gather — so the impedance mismatch is small by design. A
cost model that is a program in the object language makes "optimizing it"
and "applying it" the same operation; that, not a percent on a geomean,
is the durable payoff of this line of work.

## 10. Conclusion

We set out to beat a handwritten latency table with a learned cost model
and instead measured, to a ±0.07% floor, that a well-calibrated one ties
it — then made it worse twice, from opposite directions: once by teaching
it to rank the variants it actually chooses among (and learning the
targeted mechanism was not causal), and once by letting it learn per-op
cost structure at all (achieving the program's best calibration and a
confirmed end-to-end regression). The additive objective, the exact
DP extractor, the noise-floor decision granularity, and the 44.9% state
deltas together explain why this is not a near-miss but a structural
result — and the same instruments locate where the economics do work:
saturation, where deltas are 0.14% and an oracle bound leaves 2.6–34× on
the table. The harness that survived four of its own invalidations is the
transferable artifact; the pre-registered honest-negative you are reading
is its output. Negative results with mechanisms are the product.

---

## Appendix A: withheld shader attribution

The 12 FINAL-tier ports (module: `pixelflow-pipeline/src/shader_bench.rs`;
each kernel's doc comment carries exact source formula and a "what was
simplified" note). All fetched/searched 2026-08-27; ShaderToy pages sit
behind bot-verification, so no GLSL was copied and unconfirmed licenses
are treated as the site default (CC BY-NC-SA 3.0).

| Kernel | Source work | Author | Source | License | Category |
|---|---|---|---|---|---|
| `cosine_palette` | Cosine Color Palette (article) | Inigo Quilez | iquilezles.org/articles/palettes | CC BY-NC-SA 3.0 (default, unconfirmed) | transcendental-heavy |
| `smooth_min_scene` | smin — smooth minimum (article) | Inigo Quilez | iquilezles.org/articles/smin | CC BY-NC-SA 3.0 (default, unconfirmed) | SDF-composition |
| `mandelbrot_distance` | Distance to the Mandelbrot set (article) | Inigo Quilez | iquilezles.org/articles/distancefractals | CC BY-NC-SA 3.0 (default, unconfirmed) | select/branch-heavy |
| `star_sdf` | 2D distance functions — sdPentagram | Inigo Quilez | iquilezles.org/articles/distfunctions2d | CC BY-NC-SA 3.0 (default, unconfirmed) | SDF-composition |
| `gyroid_slice` | gyroid SDF | zzggbb | shadertoy.com/view/wtfSRS (2019) | CC BY-NC-SA 3.0 (default, unconfirmed) | SDF-composition |
| `plasma` | Plasma 90x | bitek | shadertoy.com/view/4ssGR7 (2013) | CC BY-NC-SA 3.0 (default, unconfirmed) | transcendental-heavy |
| `domain_warp_fbm` | Domain Warping (article) | Inigo Quilez | iquilezles.org/articles/warp | CC BY-NC-SA 3.0 (default, unconfirmed) | transcendental-heavy |
| `kaleidoscope_fold` | Kaleidoscope Tutorial | deliaev | shadertoy.com/view/WdcSRr (2020) | CC BY-NC-SA 3.0 (default, unconfirmed) | select/branch-heavy |
| `metaballs` | Metaball | (unresolved handle) | shadertoy.com/view/Xdl3Wl (2013); technique: Blinn 1982 | CC BY-NC-SA 3.0 (default, unconfirmed) | select/branch-heavy |
| `julia_set` | Julia — Distance 2 | Inigo Quilez | shadertoy.com/view/3llyzl | CC BY-NC-SA 3.0 (confirmed) | select/branch-heavy |
| `smoothstep_vignette` | smoothstep (glossary) | Patricio Gonzalez Vivo | thebookofshaders.com/glossary | CC BY-NC-SA (site-wide) | polynomial/FMA-friendly |
| `torus_slice` | Signed Distance Functions — sdTorus | Inigo Quilez | iquilezles.org/articles/distfunctions | CC BY-NC-SA 3.0 (default, unconfirmed) | SDF-composition |

## Appendix B: reproduction

Pipeline (all `pixelflow-pipeline`, `--features training`):
`gen_bench_corpus` (tiered mint + quarantine sidecar) →
`bootstrap_extraction_head` (cold-start, journal entry per run) →
`bench_extraction_3way` (interleaved 3-way, JSONL per-kernel records,
journal verdict line). Every run appends a journal line keyed by config
hash, source rev, corpus identity, weights identity, and environment
string; `NUMBERS.md` maps every number in this paper to its line. The
Round-3 flip analysis is `pixelflow-pipeline/examples/inspect_flip.rs`
(both checkpoints replayed on identical saturated e-graphs). The corpus,
harness, and journal are the releasable artifact.

## References

- SmoothE: differentiable e-graph extraction. ASPLOS 2025. <https://dl.acm.org/doi/10.1145/3669940.3707262>
- e-boost: boosted e-graph extraction with ILP warm-start. <https://arxiv.org/abs/2508.13020>
- E-graph extraction NP-completeness. <https://effect.systems/blog/egraph-extraction.html>
- Ithemal (2018) / BHive: learned basic-block throughput + benchmark. <https://github.com/ithemal/Ithemal>
- GRANITE: GNN basic-block throughput model (2022). <https://arxiv.org/abs/2210.03894>
- Adams et al., Learning to Optimize Halide with Tree Search and Random Programs. SIGGRAPH 2019.
- Yang et al., Equality Saturation for Tensor Graph Superoptimization (Tensat). MLSys 2021.
- Omelette: Deep RL for equality saturation. Cambridge MPhil, 2022. <https://www.cl.cam.ac.uk/~ey204/pubs/MPHIL_P3/2022_Zak.pdf>
- MCTS-GEB: MCTS-planned e-graph construction (2023). <https://arxiv.org/abs/2303.04651>
- Koehler et al., Guided Equality Saturation. POPL 2024. <https://dl.acm.org/doi/abs/10.1145/3632900>
- Sketch-Guided Equality Saturation (2022). <https://arxiv.org/abs/2111.13040>
- ML-Guided Equality Saturation. EGRAPHS 2025. <https://pldi25.sigplan.org/details/egraphs-2025-papers/6/>
- Ruler: rewrite-rule inference via equality saturation. OOPSLA 2021. <https://dl.acm.org/doi/abs/10.1145/3485496>
- Andrychowicz et al., Hindsight Experience Replay. NeurIPS 2017.
- Plate, Holographic Reduced Representations; Kleyko et al., HDC/VSA surveys I & II. <https://arxiv.org/abs/2111.06077>, <https://arxiv.org/abs/2112.15424>
- EggMind: proof-derived motif caching for LLM-agent equality saturation (2026).
