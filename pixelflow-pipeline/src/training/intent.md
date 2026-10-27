# pixelflow-pipeline/src/training

**Is.** The supervised half of the optimizer's guide (DENOTATION §8, guide
neutrality): it fits an ordering policy over rule applications to hindsight
labels read off the e-graph's provenance, and keeps the data it is judged on
apart from the data it is fit to. A guide only orders and truncates, so nothing
here can change what a kernel means.

## Owns
- The corpus: binary format and derived schema identity (`corpus.rs`), the tier
  split by family, never by draw (`split.rs`, `corpus_split.toml`), and the
  literal-blind key both fences quotient by (`structural.rs`, `FenceKey::of`).
- Which side of a holdout a decision may read: `Fence<T: HoldoutSide>`, sealed to
  `DevSide` and `FinalSide`; `Fence::build` takes no data from its caller.
- What a record means as a feature vector and what the linear forward computes
  (`guide_linear.rs`: `Record`, `to_sample`, `Model`), the bilinear checkpoint
  (`guide_bilinear.rs`), the return-to-go record and target (`r2g.rs`).
- Label minting (`mint.rs`); the OOD generators (`sh_family.rs`, `bezier_family.rs`).

## Never
- No training at compile time; no production crate depends on this one. No
  critic, policy gradient, self-play or extraction-head trainer (removed 2026).
- A fence is never built from TRAIN; every dedup and fence calls `FenceKey::of`,
  the one notion of "the same expression".
- No second encoding of a record: a trainer and its skew test share `to_sample`
  and `Model`. No label is minted from a measurement with no sentinel context.

## Contract
`mod.rs` declares, behind feature `training`: `corpus`, `split`, `structural`,
`mint`, `factored`, `guide_linear`, `guide_bilinear`, `r2g`, `sh_family`,
`bezier_family`. Consumers are the `gen_*`, `train_guide*` and `skew_test_*` bins.

## Enforced by
- Types: `HoldoutSide` is sealed through `private::Sealed`; the `compile_fail`
  doctest on `Fence` (`split.rs`) fails if a `Train` side can be spelled, and CI
  runs doctests (`.github/workflows/rust.yaml`).
- Tests: `cross_tier_overlap_is_rejected` and
  `fence_of_x_times_2_blocks_x_times_3_the_review_case` (`split.rs`),
  `feature_quotient_collapses_literals` (`structural.rs`),
  `missing_sentinel_is_a_loud_error_not_a_skipped_correction` (`mint.rs`).
- `scripts/check-bin-declarations.sh`: every bin has a `[[bin]]` entry, where
  `required-features = ["training"]` is set.

## Behind
- `mod.rs` and `../lib.rs` still list a `quarantine` module that does not exist.
  `bezier_family.rs` documents `Quarantine` against `eval_scalar`; neither is
  defined anywhere in the tree, and its `oracle_tests` holds one test (a lerp
  count), not the per-form reference it promises. Closes with JP-29.
- `Fence` has no caller outside `split.rs`'s tests; `gen_bench_corpus` went (C4).
- `mint.rs` and `../journal.rs` cite `MintMetadata` and `weights_identity`, gone
  with the extraction head (2026-09-01). `r2g.rs` says the mint "may not have
  landed"; `gen_r2g_trajectories` is in the tree.
- guide-design-revision §5 is retracted (2026-09-07, ledger L087); only its
  re-validation item 5 may be re-taken.

## Decided in
`docs/archive/plans/2026-08-31-guide-design-revision.md`, `docs/archive/plans/2026-09-01-guide-return-to-go.md`
(design and pre-registration), `docs/archive/plans/2026-09-02-bilinear-guide-registration.md`,
`docs/archive/plans/2026-08-17-cost-model-domain.md` (J7, J8), `docs/compiler/2026-07-07-guided-saturation-redesign.md`.
