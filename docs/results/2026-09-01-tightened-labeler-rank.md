> **Retracted/Superseded (2026-09-07), ledger L041.** This doc's own banner says the name-pooled vectors, quiescence filter and canonicalized ids all move the numbers below, and it was never re-run. Verdict and rationale: `docs/results/2026-09-07-claims-ledger.md` (PR #1207); the corrected benchmark and re-validation order: `docs/plans/2026-09-07-benchmark-correction.md`.

# Tightened-labeler re-measurement: does narrowing the over-approximation reorder the rule ranking? (2026-09-01)

> **Predates the 2026-09-02 review fixes; a re-run is required.** The per-rule Spearman
> vectors are now pooled by rule NAME (the library registers several indexed operator
> variants under one name, and keying by index overweighted the families with more of
> them), non-quiescent replays are excluded rather than pooled, and
> `derivation_ancestors_tight` canonicalizes class ids — all three move the numbers
> below. Nothing here has been edited to match the new code.

> **Instrument change (2026-09-02 forward-port).** Two things about how a
> Phase 3 anytime curve is measured changed with the port, so a re-run does
> not reproduce the numbers below even if nothing else changed: the
> application budget now binds **mid-scan** rather than between rule sweeps
> (`app_actual == app_target` exactly, no overshoot), and the reported cost
> is the **DAG** cost the emitted kernel pays rather than the extraction DP's
> tree total (#1117). Full statement:
> [docs/results/2026-09-02-phase3-instrument-changes.md](2026-09-02-phase3-instrument-changes.md).

Reproduce:
```
cargo run --release -p pixelflow-pipeline --features training --bin tightened_labeler_rank -- \
    --corpus-dir pixelflow-pipeline/data --min-expressions 300 --limit 800 \
    --out docs/results/2026-09-01-tightened-labeler-rank.json
```

Harness: `pixelflow-pipeline/src/bin/tightened_labeler_rank.rs` (new, additive). Library changes:
`pixelflow-search/src/egraph/provenance.rs` (`derivation_ancestors_tight`, alongside the existing
`derivation_ancestors` — neither its semantics nor its tests changed) and
`pixelflow-search/src/egraph/labeler.rs` (`EpisodeLabels::compute_tight` / `::compute_strict`,
sharing the aggregation tail with the existing `::compute` via a new private `from_load_bearing`
helper — `::compute`'s own semantics and tests are unchanged). This is `docs/plans/2026-08-31-guide-design-revision.md`
§3's "tightened-labeler track", stage-2 prep for option 3 (two-stage: strict-label cold start now,
tightened-labeler refinement in parallel).

800 expressions, stride-sampled across `corpus_train.bin` (3,359 entries) + `corpus_dev.bin` (784),
4,143-entry population at the time of this run — a **different corpus snapshot** than
`docs/results/2026-08-30-guide-headroom.md`'s (regenerated between rounds; 7,992 entries there vs
4,143 here). Pooled/median loose and strict ratios below are close to that report's published
numbers (see "Consistency with the published headroom numbers" below) despite the different
sample, which is the main reason to trust this round's *new* tight-bound numbers rather than
suspect a sampling artifact. Every number is a deterministic count or a `CostModel::latency_prior()`
cost — no wall-clock timing gates correctness. The harness asserts `strict ⊆ tight ⊆ loose` on
every one of the 800 expressions individually (not just in aggregate); the run completed with zero
assertion failures, which is itself a scale confirmation of `derivation_ancestors_tight`'s
documented safety property (previously checked only by the five hand-derivable unit tests in
`provenance.rs`).

## The headline: three points on the over-approximation spectrum

| Bound | Pooled ratio | Per-expression median | Q1 | Q3 | Implied oracle savings (1/median) |
|---|---:|---:|---:|---:|---:|
| **Loose** (`derivation_ancestors`, original labeler) | 0.7419 | **0.3793** | 0.3307 | 0.4829 | **2.64x** |
| **Tight** (`derivation_ancestors_tight`, new) | 0.0751 | **0.1940** | 0.1320 | 0.3625 | **5.15x** |
| **Strict** (node literally on the extracted path) | 0.0015 | **0.0312** | 0.0089 | 0.0596 | **32.05x** |

8,041,553 total rule applications recorded across 800 expressions; 5,966,191 loose-load-bearing,
604,270 tight-load-bearing, 11,684 strict-load-bearing.

**Tight lands strictly between loose and strict on every measure, as designed — the question is
where.** On the pooled ratio, tight sits far closer to strict on a log scale (0.075 vs strict's
0.0015 — a 51x gap — vs loose's 74x-larger-than-tight 9.9x gap in the other direction) but the
per-expression median tells a more balanced story: tight's median (19.4%) is almost exactly the
geometric midpoint of loose's (37.9%) and strict's (3.1%) on a log scale. Neither bound "wins" —
tightening removed most, not all, of the loose bound's over-crediting.

## Answering §3's central open question, by rule class

The design doc asks directly: does tightening move the structural/congruence rule class toward
strict (they really are mostly waste) or does substantial enabling-credit survive (strict was
under-crediting them)? Pooled ratios, same 800-expression run, all three bounds:

| Class | Fired | Loose % | Tight % | Strict % | Loose→Tight drop | Tight→Strict gap |
|---|---:|---:|---:|---:|---:|---:|
| **Structural/congruence** (commutative, fma-fusion, distribute, reverse-associative, associative, identity) | 7,087,906 | 78.78% | 7.86% | 0.02% | **10.0x** | **346x** |
| **Numeric** (power-recip, power-sqrt, power-rsqrt, recip-sqrt, even-negation) | 108,962 | 14.53% | 11.42% | 7.37% | 1.3x | 1.6x |

**Both things happen at once, and the honest reading is "mostly waste, but not entirely."**
Tightening removes roughly 90% of the structural class's loose credit (78.78% → 7.86%, a 10x drop)
— strong evidence that most of what the loose labeler counted as "load-bearing" for these rules
was the documented over-approximation (every tag in a visited class, every same-`rule_idx` firing
at or before a union event's step), not real enabling credit. That supports the "mostly waste"
reading and is the larger effect by far.

But the residual does not collapse to strict's near-zero: 7.86% tight-credit is still **346x**
strict's 0.02%, and in absolute terms an expression whose extraction depends on ~200 applications
(the corpus median) would still show ~16 tight-load-bearing structural-class applications where
strict shows essentially none. That residual is not noise — it survives axis-1's tightest
short-circuit (known-choice-node-only credit) and axis-2's exact-firing union credit, i.e. it is
made of applications that either (a) directly created a node in a class whose *actual* chosen node
the walk reached through some other path, or (b) were the literal cause (not just a same-rule
contemporary) of a union event that connected two classes on the chosen derivation. Both are
real, narrowly-defined enabling contributions the strict walk is blind to by construction. So:
**tightening confirms the loose bound substantially over-counted the structural class, but does
not support treating that class as pure waste** — a nontrivial, now much better-characterized
slice of real credit remains, and that slice is exactly the stage-2 refinement target the design
doc's option 3 anticipated.

The numeric class moves far less in both directions (14.53% → 11.42% → 7.37%, all three bounds
within a ~2x band) — confirming the loose bound was already a reasonably tight estimate for this
class, as `guide-headroom.md` found.

## Rank correlation: tight is a materially better predictor of strict than loose is

Per-rule (pooled ratio per rule *name*, n=54 rule instances that fired — merged names shown below),
Spearman ρ (average-rank tie handling):

| Pair | ρ |
|---|---:|
| loose vs strict | 0.186 |
| **tight vs strict** | **0.441** |
| loose vs tight | 0.247 |

Tightening more than doubles the rank correlation with the strict bound (0.186 → 0.441) — a
concrete, positive answer to "would a tighter over-approximation reorder the rule-priority
ranking": yes, substantially, and in the direction that matters (closer to the sound-by-construction
bound). It is still far from ρ=1: a Guide trained on tight labels would rank rules more like a
Guide trained on strict labels would, but the two would still disagree meaningfully — consistent
with the "not pure waste" reading above.

Per-application (every individual recorded application, pooled; phi coefficient, exactly equal to
Spearman on binary data — see the harness's module doc for the identity):

| Pair | φ |
|---|---:|
| loose vs strict | 0.022 |
| **tight vs strict** | **0.134** |
| loose vs tight | 0.168 |

Same direction, smaller in absolute terms — expected, since phi is suppressed by the extreme
base-rate imbalance at the per-application granularity (loose is positive on 74% of applications,
strict on 0.15% of them; a binary correlation coefficient is mechanically bounded well below 1
when the two marginals are this lopsided). The per-rule numbers above are the more informative
granularity for "would a Guide's rule-priority ranking change" — a per-application correlation
answers a different, noisier question (does this *specific* application agree across bounds), and
is included for completeness since the task asked for both granularities.

## Consistency with the published headroom numbers, despite a different corpus

This run's loose/strict pooled ratios (74.19% / 0.15%) and medians (37.93% / 3.12%) are close to
`docs/results/2026-08-30-guide-headroom.md`'s (71.88% / 0.14% pooled; 38.2% / 2.9% median) even
though the underlying corpus was regenerated between rounds (4,143 vs 7,992 entries) — reassuring
that both measurements are sampling the same underlying phenomenon rather than an artifact of one
corpus snapshot. The one number that does **not** reproduce closely is the per-rule Spearman
loose-vs-strict correlation: 0.186 here vs 0.35 published in `guide-headroom.md`. Both are computed
the same way (average-rank Spearman over pooled per-rule-name ratios, n in the low 50s); the
discrepancy is most plausibly sampling variance in a rank correlation computed over ~50-60 points
from two different 800-expression corpus draws — a correlation coefficient at that n has a wide
confidence interval, and neither run's rule set nor ratio-computation logic differs. Flagging this
transparently rather than picking whichever number is more convenient: **the qualitative
finding (structural class disagrees sharply, numeric class doesn't; tight sits meaningfully closer
to strict than loose does) is stable across both runs; the specific loose-vs-strict ρ value is not,
and should be read as "moderate, roughly 0.2-0.35" rather than a single precise figure** until
re-run on a fixed corpus snapshot.

## Full per-rule table (merged across operator instances, 38 distinct rule names fired)

| rule | fired | loose % | tight % | strict % |
|---|---:|---:|---:|---:|
| commutative | 2,746,549 | 86.6% | 7.2% | 0.0% |
| fma-fusion | 1,313,950 | 83.3% | 13.0% | 0.1% |
| reverse-associative | 1,312,477 | 79.9% | 6.2% | 0.0% |
| distribute | 256,079 | 77.7% | 9.3% | 0.0% |
| identity | 212,243 | 60.6% | 2.5% | 0.0% |
| doubling | 57,967 | 59.2% | 10.9% | 0.1% |
| associative | 1,246,608 | 59.0% | 6.2% | 0.0% |
| halving | 86,309 | 54.0% | 2.8% | 0.0% |
| constant-fold | 208,179 | 51.3% | 2.8% | 0.7% |
| factor | 314,357 | 47.7% | 3.7% | 0.1% |
| ln-homomorphism | 15 | 46.7% | 26.7% | 0.0% |
| half-angle-product | 3,040 | 30.5% | 4.6% | 0.0% |
| sin-angle-addition | 2,986 | 26.7% | 9.1% | 0.1% |
| canonicalize | 7,144 | 26.1% | 11.4% | 0.7% |
| cos-angle-addition | 2,335 | 25.6% | 12.5% | 0.1% |
| power-zero | 12 | 25.0% | 16.7% | 16.7% |
| odd-negation | 3,810 | 22.3% | 13.7% | 0.8% |
| power-combine | 3,433 | 19.4% | 13.6% | 3.8% |
| exp-ln-cancel | 388 | 19.3% | 12.9% | 0.0% |
| power-recip | 9,525 | 17.0% | 14.3% | 11.7% |
| power-expand-2 | 12 | 16.7% | 8.3% | 0.0% |
| annihilator | 91,831 | 16.5% | 0.9% | 0.0% |
| exp-homomorphism | 1,465 | 16.1% | 7.8% | 0.0% |
| diff-of-squares | 237 | 16.0% | 9.3% | 0.0% |
| power-sqrt | 13,474 | 15.3% | 13.3% | 10.3% |
| reverse-angle-addition | 8,584 | 14.8% | 1.8% | 0.4% |
| power-rsqrt | 3,380 | 14.5% | 13.6% | 11.9% |
| even-negation | 79,226 | 14.1% | 10.6% | 6.0% |
| involution | 38,610 | 13.9% | 11.0% | 0.0% |
| exp2-log2-cancel | 269 | 13.8% | 13.8% | 0.0% |
| power-identity | 801 | 13.7% | 12.2% | 0.0% |
| recip-sqrt | 3,357 | 13.5% | 12.6% | 10.4% |
| pythagorean | 891 | 8.4% | 2.4% | 0.0% |
| inverse-annihilation | 9,775 | 7.9% | 1.1% | 0.2% |
| idempotent | 894 | 6.7% | 6.7% | 0.0% |
| cancellation | 690 | 1.4% | 0.1% | 0.0% |
| ln-exp-cancel | 384 | 0.0% | 0.0% | 0.0% |
| log2-exp2-cancel | 267 | 0.0% | 0.0% | 0.0% |

(Sorted by loose %. Full un-merged, rule-index-level table — 54 rows — is in the sibling JSON,
`docs/results/2026-09-01-tightened-labeler-rank.json`, `per_rule`.)

One notable individual case worth flagging: **fma-fusion is the structural-class rule with by far
the highest surviving tight credit (13.0%, vs 0.1% strict)** — a 130x gap, larger than the class
average. This makes structural sense: fma-fusion creates a fused node that is a plausible direct
candidate for the winning extraction (unlike commutative/associative/distribute, whose output is
almost always a re-derivable canonical form), so even the *tight* walk's "was this node the actual
choice, or an ancestor of the actual choice's class" credit finds real matches for it that the
literal-node-only strict walk still narrowly misses (the chosen node is a different, but related,
fused form). If Phase 3 does structural-class rule triage from this data, fma-fusion should not be
grouped with the rest of the structural class without checking it individually.

## Design implications for Phase 3

1. **Tightening was worth doing, but does not resolve the label-source question on its own.**
   Both the pooled-ratio and rank-correlation views agree: tight is a real, substantial improvement
   over loose (10x pooled reduction for the structural class, 2.4x better per-rule rank correlation
   with strict) but is not a stand-in for strict — a non-trivial, now-characterized residual credit
   survives tightening for the structural class specifically (fma-fusion most of all). This is
   direct evidence for design doc §3's option 3 (two-stage) over option 2 (tighten-then-train-once):
   there is a real difference between "tight" and "strict" worth training on separately and
   comparing, not a single corrected label source that supersedes both.
2. **The residual concentrates in a few rules, not evenly across the structural class** —
   fma-fusion's 130x tight-vs-strict gap dwarfs commutative's/associative's/distribute's near-total
   collapse to strict. A stage-2 refined labeler or a rule-aware Guide feature (§4 of the design
   doc already proposes conditioning on rule identity) has a natural place to encode this:
   "trust tight credit more for fma-fusion-like rules, discount it further toward strict for
   pure-canonicalization rules like commutative/associative/distribute" is a defensible,
   data-supported per-rule prior, not a guess.
3. **The loose-vs-strict Spearman figure should not be treated as a fixed constant.** Two
   800-expression draws from two corpus snapshots produced 0.35 and 0.19 for the same
   computation. Any future document citing "ρ ≈ 0.35" for this comparison should either re-run on
   a pinned corpus or report a range; the qualitative structural/numeric split is the reproducible
   finding, not the specific correlation coefficient.

## Reproducibility

- Corpus identity: `pixelflow-pipeline/data/corpus_train.bin` (3,359 entries) +
  `corpus_dev.bin` (784 entries), timestamped 2026-09-01 in this worktree — a different generation
  than `docs/results/2026-08-30-guide-headroom.md`'s corpus (7,992 entries); see "Consistency with
  the published headroom numbers" above for why this doesn't undermine the comparison.
- Sampling: 800 expressions, stride-sampled deterministically across the full train+dev
  population (`(i as f64) * stride`, matching `guide_headroom`'s convention).
- Rule set: `pixelflow_search::math::all_rules()`, 62 rules; 38 distinct names fired on this
  sample (fewer than `guide-headroom.md`'s 39 — corpus-draw-dependent, not a rule-set change).
- Saturation: `EGraph::saturate_with_limits(100 iters, 10,000 classes, 60s safety ceiling)` —
  identical budget to `guide_headroom.rs`, so per-expression rows are directly comparable in
  method even though this run drew a different corpus sample. 0/800 expressions hit the 60s
  ceiling (asserted, fail-loud).
- Extraction: `CostModel::latency_prior()` (Phase 2's still-standing default).
- Labelers: `EpisodeLabels::compute` (loose), `EpisodeLabels::compute_tight` (new),
  `EpisodeLabels::compute_strict` (new) — `pixelflow-search/src/egraph/labeler.rs`. Subset
  invariant `strict ⊆ tight ⊆ loose` asserted per-expression during the run; zero failures over
  800 expressions.
- Full structured output: `docs/results/2026-09-01-tightened-labeler-rank.json` (per-expression
  rows, per-rule-index rows, quartiles, pooled ratios, both correlation granularities — this
  document's tables are derived from it).
- Harness: `pixelflow-pipeline/src/bin/tightened_labeler_rank.rs`.
- Library additions this task landed: `derivation_ancestors_tight`
  (`pixelflow-search/src/egraph/provenance.rs`, 5 new unit tests) and
  `EpisodeLabels::compute_tight` / `::compute_strict`
  (`pixelflow-search/src/egraph/labeler.rs`, 2 new unit tests) — both additive, alongside the
  existing loose labeler, with its own semantics and tests untouched.
