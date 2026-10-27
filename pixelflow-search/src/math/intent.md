# pixelflow-search/src/math

**Is.** The rule set `R` of the optimizer: each rule an equality of meaning
between two terms, so a rewrite never changes what a kernel denotes on the
domain of the ops it touches (DENOTATION §8, law 1).

## Owns
- Which rewrites exist and their order: `all_rules()` is 62 rules in
  declaration order (59 math, 2 fusion, 1 derivative); `Fingerprint` names it.
- `ConstantFold`, the one rule that evaluates, so the place the platform-
  divergence rows of CLAUDE.md are declined (`OpKind::fold_is_platform_specific`).
- Rules derived from structure (`InversePair`, `Parity`, `FunctionInverse`,
  `AngleAddition`), fusion, and the research sets `inflate` and `round2_rules`.

## Never
- No rule's right side denotes what its left does not, on the domain of the
  ops involved. No guard makes a rule safer than its ops; a side condition
  matches a literal, never tests IEEE behavior.
- `ConstantFold` never folds when `fold_is_platform_specific(args)`, never
  bakes a non-finite arithmetic result, and folds a comparison to a mask.
- No rule in `all_rules()` but `ConstantFold` has a comparison, `If` or
  bitwise op as its root: a mask is a pattern, not a number.
- No code names a register, lane, width or ISA tier.
- Nothing reaches `inflate` or `round2_rules` from `all_rules` or production.

## Contract
`mod.rs`: `all_rules`, `all_math_rules`, and a module per family (`algebra`,
`parity`, `trig`, `exp`, `power`, `fusion`) each with a `*_rules()` collection;
`inflate` and `round2_rules::experimental_rules` (33) serve research bins. A
rule implements `Rewrite`. `derivative_rules` and `fold_rules` live in `egraph/`.

## Enforced by
- `Rewrite::root_shapes` is required; `pixelflow-search/tests/root_shapes.rs`
  fails a rule that answers `Some` outside its claim.
- `algebra.rs` `platform_specific_fold_tests` (ties, signed zero, NaN
  `Min`/`Max`, reciprocal estimates declined; `MulAdd` and masks fold), over
  the classification `pixelflow-codegen/tests/transcendental_jit.rs` pins.
- `pixelflow-search/tests/fold_exactness.rs`: a fold and an algebraic rule that
  disagree never put two constants in one class.
- `pict_rewrite_tests.rs`: pairwise sweep of `all_rules()`, optimized against
  original at finite points, exact transcendentals as the oracle.
- `mod.rs` `all_rules_count`; `round2_rules.rs` `experimental_rules_not_in_all_rules`;
  desloppify `mask-is-not-a-number`, `hardware-instruction-first`, `simd-is-codegens`.

## Behind
- Law 1 has one test for 62 rules (2026-10-15): the sweep above. `trig`,
  `power`, `exp`, `parity` and `fusion` have none, and `Rewrite` carries no
  domain, so it is prose (`exp_rules`: "Inf/NaN inputs are UB"). No Linear issue
  item tracks it; the as-if plan (2026-10-13) says what the test is.
- Docs cite what is gone: `eval_scalar` (deleted) in `round2_rules.rs` and
  `pict_rewrite_tests.rs`, a `round2_oracle_check` and `math::oracle`. The 33
  experimental rules have no semantic test.
- Research code ships in the production crate: `inflate.rs` and
  `round2_rules.rs` (~100 KB) are `pub`, used by `phase3_round2_*` bins; their
  grids pin `all_rules`, so fold rules live in `egraph/fold_rules.rs`.

## Decided in
`docs/DENOTATION.md` §8, `docs/archive/plans/2026-09-02-optimizer-api.md` (L1),
`docs/language/2026-09-24-one-pipeline.md` §1.1.
