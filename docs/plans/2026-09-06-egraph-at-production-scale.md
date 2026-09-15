> **Superseded (2026-09-08), ledger L092.** §2's framing that hash-consing alone lands the
> ~4,900 classes "within 90 of the cap" and that "almost the whole budget is spent
> representing the input" was pre-#1146 telemetry on chrome alone. `docs/results/2026-09-07-corpus-structural-gaps.csv`
> and the completed off-vs-on measurement (`docs/results/2026-09-07-egraph-off-vs-on-real-shaders.md`)
> now show scenes and shaders enter the e-graph small (chrome 465, psychedelic 112, mandelbrot
> 128, julia 110 hash-consed nodes) and the RULES do most of the expanding — 10–40× to the
> class cap in 2–5 iterations — and most glyphs reach the cap the same way: the 95-glyph
> population (`glyph16`) hash-conses at a median 1,049 nodes (p10 96, p90 3,014) and needs a
> median 2 saturation iterations and 4.6× rule expansion (4,707 of 5,000 classes at stop) to
> get there — 43 glyphs stop after one iteration, 21 after two, 30 after three or more; 89 of
> 95 stop on the class cap, 6 quiesce first. Only the largest few enter near or above the cap
> on hash-consing alone — `U+0040` at 5,226 hash-consed nodes and two others (`U+0026`,
> `U+0038`) above 4,000 — and that is not the population. The rest of §2's chrome measurements
> (arena nodes, compile time, stop reason, extracted schedule, emitted bytes, runtime) stand;
> only the input-vs-rules attribution is withdrawn. Verdict and rationale:
> `docs/results/2026-09-07-claims-ledger.md` (PR #1207, row L092); the corrected benchmark:
> `docs/plans/2026-09-07-benchmark-correction.md`.

# The e-graph at production scale: the chrome scene as the case to optimize around

**Audience:** the research arm. **Status:** a use case and a set of measured facts,
not a plan. Everything numeric here was measured during the kernel-with-a-lattice
programme (S1–S4b-2, `2026-09-06-kernel-with-a-lattice.md`) and is reproducible
from the tree at `main`; the recipe for each number is named where it appears.

## 0. The ask in one paragraph

The JIT is now the only tier, and it wins on every kernel we ship. The first
production kernel large enough to hit the e-graph's class cap is the chrome scene
in `bench_scene_chrome`: it saturates for two iterations and stops at 4,913 of 5,000
classes, and **giving it more classes made its code slower** (5,000 → 60,000: 30× the
compile time, 15% slower). So the scaling problem is not "let saturation run
longer"; it is that the additive latency prior plus swap refinement cannot rank a
larger space, on exactly the kernel shape — guarded selects, hoisted invariants,
shared geometry across channels — where cost is non-additive. That is the regime the
schedule-cost denotation names and the Guide programme was designed for, and this
kernel is the first production instance of it. The 3.6× that was recovered on it came
from the *schedule* — arm contiguity, whether a guard is emitted — which is not an
e-node and which no function of the extracted DAG can see. So the cost model's domain
is the open design decision, and it is the research arm's: §5.1 states the question,
§7 compares the candidate shapes from the side of someone who just wrote some math
and wants it fast. What we want from research is an extraction (and, second, a
guided saturation) that turns headroom into ns/px on this kernel without moving its
compile time, and that leaves the 193 small glyph kernels untouched.

## 1. The two production shapes

| shape | example | count | size (schedule entries) | what matters |
|---|---|---:|---:|---|
| **A: one scene kernel per frame shape** | chrome scene, psychedelic shader, the terminal's cell grid | one per scene | 401 (chrome), ~200 (psychedelic) | **code quality.** Compiled once per lattice shape (time rides on `W`, so no per-frame recompile), runs every frame, every stripe. Compile is a budget, not a metric: ~220 ms today. |
| **B: many small kernels at startup** | the glyph atlas — `atlas.warm(' '..='~')`, 95 kernels, at every font size | hundreds per session | 37–87 | **compile latency**, summed. A glyph bakes once and is cached; its runtime is a rounding error. |

The measurement corpus (`collapse_cost`, 208 kernels: 193 production glyph bakes, 15
synthetic) is shape B. Every extraction and cost-model result to date was obtained on
shape B. The chrome scene is shape A and is the case this document is about; any
result must hold on both.

## 2. The chrome kernel, exactly

Four channel kernels sharing one geometry — `Ray::through_screen` → `Sphere::hit` →
Householder reflection → the world (checker floor with a box-filtered edge, sky) — with
exact `dx()`/`dy()` partials of the hit point for the checker's antialiasing, packed to
one `u32` inside the kernel and selected on packed words (S3b). Source:
`pixelflow-graphics/src/scene3d.rs`; gate: `pixelflow-runtime/examples/bench_scene_chrome.rs`.

| quantity | value | how measured |
|---|---:|---|
| arena nodes at construction | 416,420 (S3b tree; 429,395 in S3) | `Kernel::parts()` before compile |
| build time (construction, no compile) | 14–27 ms | gate example |
| e-classes after hash-consing at e-graph build | ~4,900 (from ~193k distinct nodes in the S3 telemetry) | `--features saturation-telemetry` |
| tier | classical (node count > 50): 100 iterations, 5,000 classes, 200,000 applications, 300 s ceiling | `SaturationConfig::classical()` |
| stop reason | **class cap, after 2 iterations, 4,913/5,000** | telemetry `stop` field |
| compile (saturate + extract + emit) | 181–229 ms | gate example |
| extracted schedule | 401 entries, 17 selects, 642 entries under a guard | `PIXELFLOW_GUARD_TELEMETRY=1` |
| emitted code | 9,760 B SSE2 / 7,645 B AVX-512 | `PackedManifold::code_bytes()` |
| runtime, 1920×1080, single thread | 12.2 ns/px SSE2 / 4.8 AVX-512 | gate, median of 9 |
| against the retired template tier | 1.13–1.75× faster at every thread count | S3b landing block |

**Where the 416k nodes come from.** Not the derivative pass: `lower_dwrt`
differentiates the DAG with a per-node memo (`pixelflow-ir/src/passes.rs`,
`differentiate`). They come from **construction**: `Kernel` composition splices copies
(`at`, `select`, per-channel `Rgba` operations each re-splice the hit point), so the
tree the e-graph receives is ~40× its own DAG. Hash-consing at e-graph build folds
it, and that fold alone lands within 90 classes of the cap. This is the reason the
cap binds on iteration 2: almost the whole budget is spent representing the input.

## 3. The evidence that more saturation is not the lever

All from the S3 landing block, SSE2, single thread, pre-S3b tree (43.9 ns/px baseline):

| experiment | compile | runtime | reading |
|---|---:|---:|---|
| class cap 5,000 → 60,000 | **30×** | **51 ns/px (+15%)** | more classes, worse extraction |
| `world(select(m, R, D))` (one world, select on the ray) | 0.2 s → 2.0 s; 4.6M nodes | −10% | a construction-shape change, superseded by S3b's one-select colour |
| S3b: one select per colour + guard clustering + cost bound | unchanged | **12.2 ns/px (−72%)** | the win came from the *schedule*, not from saturation |

The third row is the important one. The 3.6× improvement on this kernel came from
making the emitter's guard analysis see a contiguous arm and from bounding when a
guard pays for its branch — schedule decisions the e-graph does not represent and the
additive prior cannot see. `dyn_memory_ops`, the corpus predictor validated on shape
B, reported +40% on that change while the clock improved 7–8%: it trip-weights every
schedule entry as executed and is blind to a skipped range. That is the concrete
failure of an additive cost on this kernel.

## 4. What is engineering, not research — do these first, so research measures against the right baseline

1. **Hash-cons at construction.** The deferred "shared-store" note on `Kernel::sum`:
   one interned arena that composition indexes rather than copies. It removes the
   40× before the e-graph sees it, so the class cap stops binding on the input and
   starts binding on rewrites — which is the only regime in which "raise the cap"
   is a meaningful experiment. Measure: arena node count, build time, classes at
   stop, and the emitted bytes must be identical.
2. **The `Sqrt` derivative rule** pushes a fresh `Rsqrt(u)` estimate chain rather
   than sharing anything with the value's `Sqrt(u)`; every square root on the hit
   path pays it once per partial. A constant factor on the ~0.8× that two partials
   cost today (S4b-2 open items). Measure with `collapse_cost`.
3. **A class-cap diagnosis**, using the provenance journal
   (`--features provenance-journal`): of the ~4,900 classes at stop, how many are
   the hash-consed input, how many are numeric-rule expansions (`recip`/`sqrt`
   families on the ray normalizations) that never reach an extraction, and how many
   are genuine alternatives. This tells research whether the cap is guarding memory
   (its stated purpose, `saturate.rs`) or meaning.

## 5. What to optimize around

Each item has a metric, an oracle, and the document that already denotes it.

**5.1 The cost model's domain is the schedule — the first-order ask.**
§3 says what the cost of this kernel is a function of: which arm is contiguous,
whether a guard is emitted, whether its mask is coherent per batch, what is hoisted to
which level. None of those are e-nodes. Extraction today produces a DAG and scores it
additively; the schedule is chosen afterwards by static rules in the emitter (the
linearization, `cluster_select_arms`, `MISPREDICT_PENALTY_CYCLES`, the LICM levels).
So a model that reranks DAG candidates by *any* function of the DAG cannot see the
3.6×: every candidate it ranks is the same DAG under a different schedule. The ask is
therefore not a better score for DAGs. It is a cost model whose domain includes the
schedule, and an extraction that ranges over it — the regime-2 statement
`cost : Extraction → cycles` in `2026-09-01-schedule-cost-model-denotation.md`, read
with "extraction" meaning a scheduled DAG. Two ways to get there:

- (i) **schedule choices as e-graph alternatives**: levels already are, in the
  denotation; guard emission and arm clustering would join them, so extraction
  chooses the schedule and the cost model scores what it chose;
- (ii) **extract a DAG, enumerate its admissible schedules, score the pairs**: the
  e-graph stays as it is and the search moves to the schedule.

Either way the model is non-additive and the oracle is measurement (arm C of §5.3
there: the best of k candidates by the clock). The `Reranker` trait in
`pixelflow-search/src/egraph/extract.rs` is one place such a model could plug in *if
its candidates carry their schedule*; it is a seam, not the shape, and the shape
written around it in that document — an additive backbone with a learned residual
over swap neighbourhoods — is narrower than the ruling it quotes and, by §3, cannot
express this kernel's win. **Mask coherence** is the first term that is data rather
than a property of the kernel: the sphere's silhouette mask is uniform in 97% of
batches, a glyph's coverage mask in almost none, and the static bound refuses guards
under 16 cycles precisely because it cannot know. It enters either as a profile
(§7, E) or as a prior learned over mask shapes.

The trigger stays as written there: (a) the chrome e-graph admits ≥2 extractions with
distinct schedule assignments, and (b) the measured best-of-k beats the static
choice by a geomean clearing ±5%. Metric: ns/px on `bench_scene_chrome` at both tiers
with compile time unchanged. The shape is the research arm's to choose; §7 is the
comparison it should choose against.

**5.2 Guided saturation on a kernel that hits the cap.**
The Guide programme (`2026-08-31-guide-design-revision.md`) measured a 2.6× oracle
savings bound (labeler) on the shape-B corpus. The chrome kernel is the case where the
budget that binds is classes rather than applications, so the question changes shape:
not "which applications were load-bearing" but "which *classes* were". Ask: the
anytime curve of extracted cost versus classes admitted, with the chrome kernel as the
held-out production instance in the pre-registered Phase 3 experiment. Metric: cost
at a fixed class budget; then ns/px.

**5.3 The compile budget, stated so it can be checked.**
Shape A: a scene compiles at ~220 ms today and a resize recompiles; anything under
~250 ms is invisible, 2 s is not (the `world(select)` row). Shape B: 95 glyph compiles
at startup per font size; a research result that adds 5 ms per kernel adds half a
second to startup. Any change must report both, and the glyph kernels must stay
byte-identical unless the change is meant for them.

## 6. How to measure, and the traps already found

- **Runtime:** `cargo run --release -p pixelflow-runtime --example bench_scene_chrome`,
  SSE2 default and `RUSTFLAGS="-C target-feature=+avx512f,+avx512dq"`, threads
  1 / cores / 12, idle host, load stated. Pixel agreement before timing.
- **Saturation:** `--features saturation-telemetry` writes a JSONL record per
  production saturation (tier, node count, stop reason, classes, applications,
  wall clock, extracted cost) to `$PIXELFLOW_SATURATION_TELEMETRY`.
- **Guards:** `PIXELFLOW_GUARD_TELEMETRY=1` prints, per compiled program, schedule
  entries, selects, arm-exclusive entries, and entries under a guard.
- **Corpus:** `collapse_cost` (pixelflow-pipeline), 208 kernels, both tiers.
  **Compare two builds' rows, never a hash against prose** — two stages recorded
  column hashes that did not reproduce. Per-kernel clock ratios are not trustworthy
  below ~10% on the 4-core host (byte-identical code measured 4× apart between runs,
  A/A floor ~1% within a run); the aggregate sign is the claim.
- **Determinism:** budgets are rule applications, classes, and iterations — never
  wall clock (`2026-09-01-production-budget-determinism.md`). The 300 s safety
  ceiling panics rather than truncating; a panic is a finding, not a knob.

## 7. The alternatives, from the consumer's side

The consumer is someone who wrote some math in a `kernel!`, composed a few of them,
baked it over a lattice, and wants it to be fast. They do not know what a schedule, a
guard, a lane or an e-class is, and the language promises they never will. Every
column below is judged from there.

| | A. **current** — additive DP + swap refinement; static schedule rules after | B. **residual reranker** — A, plus a learned score over top-k swap candidates (the denotation doc's shape) | C. **schedule-valued cost** — extraction ranges over scheduled DAGs, non-additive model, measured oracle (§5.1) | D. **measured search** — compile k candidates, time them, keep the best | E. **profile-guided** — count mask coherence per select at runtime, recompile with it | F. **no e-graph** — hash-cons, FMA peephole, static schedule rules |
|---|---|---|---|---|---|---|
| my kernel runs fast | yes on every shipped kernel: ahead of the retired templates 1.1–2.9×; but only after a *codegen* fix found the 3.6× that extraction never saw | no better than A on this kernel: the win is not in the candidate set (§3) | the one shape that can find the 3.6× by design rather than by a hand-written rule; unproven | yes, by construction — it measures | recovers exactly the coherence-dependent part (guards) and nothing else | unknown: never measured (see below) |
| my scene compiles quickly | ~220 ms, once per shape | + a model forward pass per candidate, ×k | + a schedule enumeration per candidate; must be bounded | ×k compiles plus k timings — seconds, and per host | two compiles and a warm frame | fastest possible; no saturation |
| my glyphs don't slow startup | 95 compiles, small kernels, saturate fully | small kernels have few alternatives; near A | same, if the enumeration is bounded by kernel size | ×k on 95 kernels: the worst case of D | a profile per glyph is pointless (its runtime is a rounding error) | fastest |
| same code on every machine | yes — budgets are applications/classes/iterations, never a clock | yes, if the model is a fixed artifact | yes, if the model is a fixed artifact | **no** — it chooses by this host's clock; the determinism rule (`2026-09-01-production-budget-determinism.md`) is violated unless the choice is recorded and replayed | no — the profile is this run's data | yes |
| no cliffs (adding a `select` doesn't make it 3× slower) | **this is A's failure mode**: S3 shipped the chrome scene at 0.32× and it took two codegen changes to fix; the next kernel shape with a cost the prior cannot see will do it again | same cliffs as A; the reranker cannot see them either | cliffs become model error, which is measurable and trainable, not a hand-written rule to discover | no cliffs it can measure; cliffs it cannot enumerate (k is finite) | removes the coherence cliff only | more cliffs: every algebraic rewrite the e-graph makes today is left on the table |
| what I have to know | nothing | nothing | nothing | nothing, but my build is slow and my binary differs from my colleague's | nothing, but the first frame is slow | nothing |
| when it is wrong, what happens | silently slower; a bench catches it, if someone runs one | same | same, plus the model's error is a number that can be tracked per kernel | it is not wrong about *this* host | wrong on a scene whose masks change (the animated sphere's silhouette moves) | silently slower |
| evidence in the tree today | every gate S1–S4b-2; the `dyn_memory_ops` +40% vs −8% inversion; cap 5,000 → 60,000 made code 15% slower | tied the static table on a corpus with no schedule alternatives; the residual was "empty by construction" | none yet; the trigger in §5.1 is the first experiment | `collapse_cost` *is* D over a corpus — it is the oracle harness, not a product | none; `PIXELFLOW_GUARD_TELEMETRY` counts what a profile would count | **none — the raw-versus-optimized delta has never been measured on a shipped kernel** |

Three things the table says that the prose above does not:

1. **A is the right default and its cost is cliffs, not speed.** On every kernel we
   ship it is deterministic, cheap to compile, and ahead. Its failure is that the
   next kernel shape whose cost the additive prior cannot see ships slow, and is
   fixed by someone finding the rule by hand — which is what S3b was. That is the
   thing research is for, and it is why the target is C and not a faster A.
2. **B is not a step towards C.** It keeps the DAG as the domain, so its ceiling on
   this kernel is A's. Building it would spend the research budget proving the
   parity result again in regime 2.
3. **D and E are instruments, not products.** D is the oracle every candidate model
   is measured against and already exists as `collapse_cost`; making it the compiler
   trades the determinism rule for a per-host binary and a ×k build. E is how mask
   coherence, the one term that is data, gets measured for training C's prior; as a
   product it needs a warm frame and breaks on scenes whose masks move.

One gap the table exposes: **F has never been measured.** Every stage measured the
optimized kernel against a different implementation, never against the same kernel
with saturation off (`kernel_raw!` exists for this). Before any research spend, the
raw-versus-optimized delta on `bench_scene_chrome`, `bench_scene_psychedelic` and the
glyph corpus is the number that says what the e-graph buys today on the kernels that
exist. If it is small, the research problem is *extraction of schedules* and the
saturation half can wait; if it is large, both halves matter.

## 8. Pointers

`pixelflow-search/src/egraph/{saturate,extract,cost,provenance,labeler}.rs`;
`pixelflow-search/src/runtime.rs` (`optimize_runtime_arena`: `LowerDwrt` →
`ExpandReduce` → `Saturate::runtime`); `pixelflow-codegen/src/emit/guards.rs`;
`pixelflow-graphics/src/scene3d.rs`; the landing blocks S3, S3b and S4b-2 in
`2026-09-06-kernel-with-a-lattice.md`.
