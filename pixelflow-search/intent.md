# pixelflow-search

**Is.** The optimizer, `P`'s only optimizing step (DENOTATION §8): a function
from an `ExprArena` to an equal one, by saturating an e-graph and extracting
under a cost. Five laws hold it: rules preserve denotation, saturation only
adds equalities, extraction round-trips, a guide changes cost and never
meaning, and a non-additive cost enters only as a `Reranker`.

## Owns
- The rule set and its order (`RuleSet`, `Fingerprint`), the e-graph, and the
  budget: `SaturationConfig` counts rule applications, minted class ids and
  iterations, all deterministic in the input; wall clock is `safety_ceiling`,
  an assertion that panics the build.
- The cost model and extraction: `CostModel::latency_prior()` is the only
  policy; choosing among equivalent forms happens here.
- The structure-keyed saturation cache and the unit walk (`runtime.rs`): a
  named kernel is saturated alone, then linked back. The provenance journal
  and telemetry record, both off by default.

## Never
- No workspace dependency but `pixelflow-ir`; nothing names a register, a
  lane, a width or an ISA tier.
- No budget reads a clock; `PIXELFLOW_SATURATION_CEILING_MS` changes whether a
  build panics, never which kernel it emits.
- No env-driven policy: `Optimizer::production()` is the static prior, unguided.
- No fold whose answer depends on the build host (`fold_is_platform_specific`
  declines in `math/algebra.rs`); unequal constants are never unioned.
- No journal in a production build or in the budget's denominator
  (`Saturation::application_count` is unconditional). A failed telemetry
  write panics.

## Contract
`src/lib.rs`: `egraph` (`Optimizer`, `Budget`, `EGraph`, `insert`, `CostModel`,
`Reranker`, `SaturationConfig`), `math` (rules), `nnue` (the Guide, research),
`runtime` (`optimize_runtime_arena`; `None` when declined), `saturate_pass`
(`Saturate`, the `Optimize` both front ends run), `tier`, `telemetry` (feature
`saturation-telemetry`).

## Enforced by
- `scripts/check-provenance-journal-scope.sh` (CI): `cargo tree` shows no
  `provenance-journal` under `pixelflow-compiler` or `core-term`.
- `tests/production_budget_determinism.rs` (caps pinned, stops independent of
  CPU load), `tests/saturation_ceiling_env.rs` (the override moves only the
  panic), `tests/growth_telemetry_determinism.rs` (telemetry cannot move the
  budget), `tests/fold_exactness.rs` (conflicting constants refused);
  `egraph/graph.rs` pins law 2.

## Behind
- Laws 1, 3, 4 have no test of their own: `egraph/optimizer.rs` cites a laws
  module that is not in the tree (optimizer-api audit, DENOTATION §8).
- Law 5's first half is violated: extraction prices a tree, never sharing
  (JP-15); `Reranker` has no implementation.
- `Tier`, `Saturate::macro_tier` and the telemetry `"tier"` field remain
  though the pipeline has no tier (one-pipeline M1, M3). A declined term
  compiles unoptimized, unreported (`jit_cache` takes the raw arena; M12).

## Decided in
`docs/DENOTATION.md` §8, `docs/archive/plans/2026-09-02-optimizer-api.md`,
`docs/archive/plans/2026-09-01-production-budget-determinism.md`, `docs/language/2026-09-24-one-pipeline.md`.
