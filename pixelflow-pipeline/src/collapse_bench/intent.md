# pixelflow-pipeline/src/collapse_bench

**Is.** The observer of `P(k, s, t)` at the collapse (DENOTATION §8): for a corpus
of kernels, each with the shape it is baked at, it compiles as `Lattice::bake`
does, times the one emitted collapse call, and writes a row per (kernel, tier,
git ref, pass): the wall clock beside the static traffic of the code that made
it. The clock is the outcome; the traffic is a column.

## Owns
- The corpus as a fixture with a diff: `CollapseKernel`, the text format and its
  versioned header, `synthetic()` (`corpus.rs`).
- Measuring a collapse body: `CollapseSession` (pin, burn-in, sentinel
  re-measured every `SENTINEL_INTERVAL` kernels, median of `SAMPLES` with its
  IQR), the fixed visit order `shuffled`, and what a row is (`row.rs`), with
  `features_of` the one map from `CompileResult::traffic` to its columns.
- Scoring a static predictor against the clock (`predict.rs`): Spearman rho per
  tier, and sign agreement on the delta between two allocations of one kernel.

## Never
- No second compile path: every kernel is built by `compile_as_baked`
  (`optimize_runtime_arena`, then `emit::compile`).
- No static quantity is a verdict: a `Predictor` is a closed-form
  `fn(&Row) -> f64`, nothing fitted or captured, scored against the clock.
- No lane count or tier of its own: `lanes()` is `jit_vector_bytes() / 4`,
  `tier()` is `isa::detect().name()`.
- No timestamp, path or host name in a row, and no mixed schemas (`points`
  panics); two runs of one build differ only in what was measured.
- No estimate shown as a count: `features_of` panics on a kernel with a fold
  whose range is not known, so a value-bounded kernel is refused at measurement.

## Contract
`mod.rs` declares `corpus`, `predict`, `row` and offers `lanes`, `tier`,
`compile_as_baked`, `CollapseSession` (`open`, `context`, `measure`),
`features_of`, `run_corpus`, `write_jsonl`, `read_jsonl`, and re-exports `Stat`.
Consumers: `src/bin/collapse_cost.rs`, `src/bin/corpus_gaps.rs`.

## Enforced by
- Types: `Predictor::eval` is a `fn` pointer (`predict.rs`); the corrected figure
  is a `LocalNs` normalized by a `DriftFactor` (`jit_bench.rs`).
- Tests: `the_visit_order_is_a_permutation_and_does_not_move` (`mod.rs`),
  `a_written_corpus_reads_back_identical` (`corpus.rs`).
- The as-if contract: `EmitTraffic` is an instrument the benchmarks read, not a
  test target (`docs/process/2026-10-13-tests-check-the-as-if-contract.md`).
- `scripts/check-bin-declarations.sh`, `scripts/check-doc-paths.sh` (rust.yaml).

## Behind
- 2026-10-15: the corpus replays bound buffers (`B`/`D` lines, `buffer_data`,
  `dummy_context`); a bound buffer is not a concept (DENOTATION §11). Closes with
  the discrete-kernel lift (kernels-evaluate-kernels §2).
- Comments name what is gone: `emit_skip_if_all_false`, an `sse2` tier (`row.rs`).
- `hoisting_puts_work_in_a_prologue` asserts a hoist happened: a mechanism in a
  test, which the as-if plan sends to a benchmark.
- `Trips::of` panics on an extent narrower than one batch, though a compiled
  kernel fills its remainder (placement-reads-demand, line 547).
- `features_of` takes `scopes[1]` as the outermost lattice fold, a convention of
  today's pass order; it moves with the schedule (kernels-evaluate-kernels §3).

## Decided in
`docs/archive/plans/2026-09-01-register-allocation-escape-hatches.md`,
`docs/process/2026-10-13-tests-check-the-as-if-contract.md`,
`docs/compiler/2026-09-01-schedule-cost-model-denotation.md`.
