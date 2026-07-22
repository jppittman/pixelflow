# Extraction 3-way bench: NNUE vs static latency prior vs no-swap (2026-07-08)

> **⚠ Correction (2026-07-20): absolute ns columns under-scaled 41.67x.**
> The harness timer (`pixelflow-pipeline/src/jit_bench.rs::nanos_now`) reported
> raw `mach_absolute_time()` ticks as nanoseconds, but on native Apple Silicon
> the timebase is 125/3 (one tick = 41.67ns; 1:1 holds only on Intel Macs and
> under Rosetta). Every **`ns NNUE` / `ns static` / `ns no-swap`** figure below
> is therefore 41.67x smaller than the true wall-clock time (e.g. swirl's
> "0.290ns" is really ~12.1ns). Because the scale factor is uniform, **all
> ratios, geomeans, orderings, and the Phase 2 gate verdict are unaffected**.
> The `extract μs` columns were measured with `std::time::Instant` and were
> always correct. See docs/results/2026-07-20-jit-compile-cost.md, "Timer
> correction" section, for the fix and empirical verification.

Phase 2 gate of `docs/plans/2026-07-07-guided-saturation-redesign.md` — the
deferred February 3-way experiment (HCE vs Judge vs Guided), redone with the
freshly retrained Judge and no HCE (deleted in the April squash; NNUE vs the
static latency prior is the closest still-buildable analogue).

Reproduce: `cargo run --release -p pixelflow-pipeline --features training --bin bench_extraction_3way`

## Setup

- **Machine**: Apple M2 Max (Apple Silicon MacBook), aarch64, NEON JIT ABI.
  `mach_absolute_time()` timing (1 tick = 1 ns on this hardware). *[Correction
  2026-07-20: this assumption was wrong — the timebase on this machine is
  125/3, so 1 tick = 41.67ns. See the correction note at the top.]*
- **Judge weights**: `pixelflow-pipeline/data/expr_nnue_trid.bin` (138,120
  bytes, TRID format), loaded via `ExprNnue::from_bytes` — the same loader
  verified by `pixelflow-pipeline/tests/judge_weights_load.rs`.
- **Corpus**:
  - 5 named, production-shaped kernels (verbatim from
    `pixelflow-search/examples/rule_report.rs` / the swirl shape in
    `pixelflow-search/tests/prod_kernel_jit.rs`): `swirl`, `circle_sdf`,
    `poly`, `redundant`, `normalize`.
  - 40 synthetic expressions from `BwdGenerator` (`BwdGenConfig::default()`,
    seed `0xB0BA2026`) — the same generator `bootstrap_extraction_head`
    trains the Judge on. The generator's junkified ("unoptimized") form is
    used as the input expression, mimicking an as-authored kernel body.
    Confirmed 0/40 synthetics contain `Gather`/`RawGather`/`Buffer`
    (`BwdGenerator` has no code path that emits memory ops), so all corpus
    entries are directly comparable — none exercise the aarch64 gather path.
- **Pipeline per kernel**: one `EGraph::with_rules(all_rules())`,
  `saturate_with_limit(40)` (matches `prod_kernel_jit.rs`). From that single
  saturated e-graph:
  - **(a) NNUE** — `IncrementalExtractor::new(&judge, top_k=8)` →
    `extract_choices_only` → `choices_to_arena`.
  - **(b) STATIC** — `extract_dag(&eg, root, &CostModel::latency_prior())` →
    `choices_to_arena`.
  - **(c) NO-SWAP** — the original, un-extracted arena/root, compiled as-is.
- **JIT**: `compile_arena_dag` (same path as `prod_kernel_jit.rs`), benchmarked
  with `pixelflow-pipeline/src/jit_bench.rs`'s `benchmark_jit_arena` at its
  defaults (20 timed samples, 64 warmup iters, 100x manually-unrolled inner
  loop per sample, median-of-20 reported).
- **Correctness gate**: before trusting any timing, NNUE and STATIC forms are
  cross-checked against the NO-SWAP JIT output on an 8-point coordinate grid
  (zero, small, negative, larger-magnitude values across all 4 input
  registers), tolerance `max(0.2 absolute, 5% relative)`. A form that fails
  this check is excluded from that policy's timing and from the geomean —
  loudly, with the exact grid point and values printed (see Correctness
  failures below) — never silently averaged in.

## Results

| kernel | nodes | ns NNUE | ns static | ns no-swap | NNUE/static | static/no-swap | extract μs NNUE | extract μs static |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| swirl | 15 | 0.290 | 0.290 | 0.300 | 1.000 | 0.967 | 86.9 | 20.0 |
| circle_sdf | 18 | 0.030 | 0.020 | 0.020 | 1.500 | 1.000 | 525.6 | 74.5 |
| poly | 11 | 0.020 | 0.010 | 0.020 | 2.000 | 0.500 | 52.3 | 6.2 |
| redundant | 13 | 0.010 | 0.010 | 0.010 | 1.000 | 1.000 | 345.5 | 48.1 |
| normalize | 10 | 0.020 | 0.020 | 0.020 | 1.000 | 1.000 | 28.9 | 3.3 |
| **synthetic (n=40, geomean)** | 40 | 1.479 | 1.349 | 2.215 | 1.096 | 0.609 | 1454.7 | 48.3 |

**GEOMEAN** over all valid pairs (5 named + 33 synthetic = 38 NNUE/static
pairs; 5 named + 35 synthetic = 40 static/no-swap pairs):

- **NNUE / static = 1.0669** — NNUE is ~6.7% *slower* than the static latency
  prior.
- **static / no-swap = 0.6676** — the static prior's extraction is ~33%
  faster than not extracting at all (e-graph rewriting + a decent cost model
  clearly earns its keep; that part of the pipeline works).
- **Extraction overhead**: NNUE costs a mean 2078 μs to choose per kernel vs
  67 μs for the static prior — **NNUE extraction is ~31x more expensive to
  run** than the static path it's trying to beat, on top of losing on the
  timing it's meant to improve.

Named-kernel failures: 0/5 for both policies. Synthetic failures: NNUE 7/40,
static 5/40 (union 7/40 — every static failure is also an NNUE failure on
that same kernel, plus two NNUE-only failures).

## Correctness failures (excluded from geomean, not blockers)

7 of 40 synthetic expressions produced a numeric mismatch for at least one
extraction policy (`synth_004`, `synth_007`, `synth_013`, `synth_022`,
`synth_024`, `synth_030`, `synth_037`). All failures are at coordinates
including `(0,0,0,0)` or involve very large magnitudes (one pair diverges at
~1e40). This pattern — divergence concentrated at the origin and at
astronomically large values, affecting static and NNUE symmetrically on the
same kernels — points to **numerically unstable rewrites in the randomly
junkified synthetic corpus** (e.g. algebraically-valid but floating-point-
unsound simplifications through near-singularities: division/pow/exp chains
that are exact in real arithmetic but diverge under IEEE-754 rounding once
the expression shape changes), not a policy-specific extraction bug. It
affects both cost models at a similar rate (5/40 vs 7/40) and is therefore
orthogonal to the NNUE-vs-static comparison — but it is a real rewrite-rule
soundness gap worth its own investigation before the corpus is used for
further NNUE training data.

## Interpretation — Phase 2 gate verdict

**NNUE loses to the static latency prior by ~6.7% geomean — the gate FAILS.**
Per the plan: *"if NNUE ≤ latency prior, port the prior into the static
`CostModel` as the default and keep NNUE opt-in only."* That is the
recommended next action. Two independent findings support closing this
gate rather than reopening it:

1. The named-kernel comparisons are dominated by measurement-floor noise —
   these are 10-18 node expressions with sub-nanosecond runtimes, so a 1.5x
   or 2x ratio there (circle_sdf, poly) is plausibly a couple of JIT-emitted
   cycles, not a systematic effect. The 40-synthetic aggregate (larger,
   more diverse expressions, up to hundreds of nodes pre-junkification) is
   the more statistically meaningful number, and it still shows NNUE ~10%
   slower than static there.
2. NNUE extraction is also ~31x more expensive to *run* than the static
   path (2078 μs vs 67 μs mean, both negligible next to a full kernel
   compile but not free). A learned extractor that is both slower to choose
   and produces slower code has no case for staying in the default path.
   The static prior, on the other hand, clearly outperforms not extracting
   at all (33% faster geomean over no-swap) — the e-graph + cost-model
   machinery works; the *learned* half of it, as currently trained, does
   not yet add value over the handcrafted table it was supposed to
   supersede.

This does not indict the guided-saturation thesis itself (Phase 3, which
tests rule-library scale under a learned Guide, is a different question from
"is the current Judge's extraction head better than a handcrafted table on
already-saturated small graphs"). It does mean Phase 2's specific
recommendation — demote NNUE to opt-in, keep the static prior as the shipped
default — should be carried out before further Judge-only investment.
