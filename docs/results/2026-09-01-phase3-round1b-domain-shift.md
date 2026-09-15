> **Retracted/Superseded (2026-09-07), ledger L038.** The H_null verdict compared two arms that both lose on the real metric (DAG cost), on generator families rather than shipped shaders; the trig finding that half-angle-product pays on sh (L039) stands. Verdict and rationale: `docs/results/2026-09-07-claims-ledger.md` (PR #1207); the corrected benchmark and re-validation order: `docs/plans/2026-09-07-benchmark-correction.md`.

# Phase 3 round 1b — domain shift: H_null on `sh`, and the shift is not about trig at all (`bezier` moves the same amount) (2026-09-01)

> **Instrument change (2026-09-02 forward-port).** Two things about how a
> Phase 3 anytime curve is measured changed with the port, so a re-run does
> not reproduce the numbers below even if nothing else changed: the
> application budget now binds **mid-scan** rather than between rule sweeps
> (`app_actual == app_target` exactly, no overshoot), and the reported cost
> is the **DAG** cost the emitted kernel pays rather than the extraction DP's
> tree total (#1117). Full statement:
> [docs/results/2026-09-02-phase3-instrument-changes.md](2026-09-02-phase3-instrument-changes.md).

Reproduce (same binary, same checkpoint, three corpora; every run is deterministic and resumable):
```
cargo run --release -p pixelflow-pipeline --features training --bin gen_bezier_corpus     # sh entries preserved, bezier appended
B=./target/release/phase3_at_budget_eval
$B --stratify-by-ops --classical-samples 0 --other-samples 0 \
   --out-jsonl docs/results/2026-09-01-phase3-round1b-dev.jsonl    --out-json docs/results/2026-09-01-phase3-round1b-dev.json    --out-md docs/results/2026-09-01-phase3-round1b-dev-report.md
$B --corpus pixelflow-pipeline/data/corpus_dev_ood.bin --name-prefix dev_sh_     --stratify-by-ops --classical-samples 0 --other-samples 0 \
   --out-jsonl docs/results/2026-09-01-phase3-round1b-sh.jsonl     --out-json docs/results/2026-09-01-phase3-round1b-sh.json     --out-md docs/results/2026-09-01-phase3-round1b-sh-report.md
$B --corpus pixelflow-pipeline/data/corpus_dev_ood.bin --name-prefix dev_bezier_ --stratify-by-ops --classical-samples 0 --other-samples 0 \
   --out-jsonl docs/results/2026-09-01-phase3-round1b-bezier.jsonl --out-json docs/results/2026-09-01-phase3-round1b-bezier.json --out-md docs/results/2026-09-01-phase3-round1b-bezier-report.md
# combined tables: 2026-09-01-phase3-round1b-domain-shift.{json,csv}
```

Registration: `docs/plans/2026-09-01-phase3-round1b-domain-shift-registration.md` — nothing in §1–§6 was
revised; §7 is appended there against its gates. Round-1 discipline unchanged: budgets in rule
applications, the one curve definition `egraph::anytime::run_anytime_curve[_with]`, deterministic
`CostModel::latency_prior()` `extract_dag` cost, no wall-clock in any number, FINAL untouched
(`corpus_final.bin` never opened). **Model under test: `guide_checkpoint_strict_v1.json` exactly as
Round 1 trained it — no retraining, no checkpoint change.** Control arm: `PerRuleRateGuide` from the
same training run's per-rule TRAIN strict-positive rates. Same guided grid `[25, 50, 100, 200, 400, 800]`.

- `corpus_dev.bin` MD5 `3026133ebba066eeca10f658da554400` (unchanged, matches Round 1). All 334 DEV
  classical expressions were **re-run** through the ladder (not read from Round 1's rows) so that every
  arm carries the new per-rule-index firing/strict histograms; the re-run reproduces Round 1 exactly
  (control 0.5655 / linear 0.5366 at B=100, 0.6991 / 0.6959 at B=200 — `D_A^DEV = 0.0000` at both
  tiers, so the harness is deterministic across revisions).
- `corpus_dev_ood.bin` (regenerated: `gen_sh_corpus` then `gen_bezier_corpus`, 180 entries, MD5
  `0c7cbe710c50175afb3cd91f60960b64`): `sh` 100 entries, of which **95 classical** (5 have ≤ 50 arena
  nodes and fall in the rapid band by the harness's `tier_name`; the registered tiers are classical, so
  n = 95); `bezier` 80 entries, all classical. Both families fence-checked against `corpus_train.bin`
  (3,359 structures) at generation AND at load: 0 collisions. `sh` stratifies as 95/95 `trig-heavy`,
  `bezier` as 80/80 `polynomial-only`, by construction.
- Source rev: see `context.source_rev` in each report JSON (harness `phase3_at_budget_eval` on
  `claude/phase3-domain-shift`). Host load average during the runs ~6–7 on 12 cores (context for the
  production probe's wall-clock stop only; no metric depends on it).

## The answer

**H_shift — did the global prior break under shift while candidate-local features held? No.** On `sh`
at B=100 (primary), D_control = +0.337 and D_linear = +0.367: both arms lose the same ~0.35 of ratio,
D_control − D_linear = −0.030, inside M_100 = 0.06 → the pre-committed verdict is **H_null**, and the
sign is the wrong way for H_shift (the linear model moved slightly *more* than the control). Same at
B=200 (D_control +0.195, D_linear +0.200, diff −0.005 vs M_200 = 0.07 → H_null).

**H_null — did the corpus teach the linear model the same global prior through its features? Yes,
as far as this test can see.** The candidate-local model and the per-rule lookup table are statistically
indistinguishable on trig-dominant kernels at both tiers (head-to-head at B=100: linear < control on
30, equal on 43, > on 22; at B=200: 3 / 55 / 37). The `neighborhood_op_hist` feature had no
trig-positive examples to learn from and did not learn to use trig context. **Training-distribution
coverage — a structure-aware TRAIN family with shared-argument trig — is the next lever, not
architecture.** But read the next two findings before spending it.

**H_inv — is the neighborhood feature actively harmful under shift? No.** D_linear − D_control =
+0.030 / +0.005, inside M at both tiers.

**The Bézier prediction FAILS, for both arms, at both tiers — and by the same amount as `sh`.**
Registration §1.2 predicted D_A ≤ 0 (point) and D_A ≤ +M_B (binding) on the trig-free family, because
the global prior is *right* there. Measured: D_control = +0.344, D_linear = +0.373 at B=100 (+0.211 /
+0.190 at B=200). A polynomial-only family with zero trig matches shifted the ratio by the same +0.34
as the trig-dominant family did. **The shift in D is therefore not about which rules the prior
suppresses; it is about the regime real structured kernels put the unguided baseline in** (next
section). Per §5 this promotes the dedup-closure / budget-regime investigation ahead of labels for
Round 2, alongside the coverage fix.

### D against the registered margins

| set | n | B | m_control^S | m_linear^S | D_control | D_linear | D_control − D_linear | M_B | verdict | §1.2 poly prediction (D ≤ 0 both / D ≤ M both) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| DEV (re-run, reference) | 334 | 100 | 0.5655 | 0.5366 | −0.0000 | −0.0000 | −0.0000 | 0.06 | H_null (identity check) | — |
| DEV (re-run, reference) | 334 | 200 | 0.6991 | 0.6959 | +0.0000 | −0.0000 | +0.0001 | 0.07 | H_null (identity check) | — |
| **`sh` (primary)** | 95 | **100** | 0.9028 | 0.9039 | **+0.3373** | **+0.3673** | **−0.0300** | 0.06 | **H_null** | — |
| `sh` | 95 | 200 | 0.8940 | 0.8959 | +0.1949 | +0.2000 | −0.0051 | 0.07 | H_null | — |
| `bezier` | 80 | 100 | 0.9098 | 0.9098 | +0.3443 | +0.3732 | −0.0289 | 0.06 | H_null | **FAILS / FAILS** |
| `bezier` | 80 | 200 | 0.9098 | 0.8855 | +0.2107 | +0.1896 | +0.0212 | 0.07 | H_null | **FAILS / FAILS** |
| DEV `trig-heavy` stratum | 287 | 100 | 0.5701 | 0.5399 | +0.0046 | +0.0033 | +0.0013 | 0.06 | H_null | — |
| DEV `trig-heavy` stratum | 287 | 200 | 0.6724 | 0.6654 | −0.0267 | −0.0305 | +0.0038 | 0.07 | H_null | — |
| DEV `transcendental-heavy` stratum | 47 | 100 | 0.5089 | 0.5197 | −0.0566 | −0.0169 | −0.0398 | 0.06 | H_null | — |
| DEV `transcendental-heavy` stratum | 47 | 200 | 1.0000 | 1.0000 | +0.3009 | +0.3041 | −0.0032 | 0.07 | H_null | — |

Every set with n ≥ 30 returns H_null; no set returns H_shift or H_inv; the two OOD families fail the
§1.2 prediction identically.

**The DEV stratification has no power on this corpus.** The §2 rule puts 287 of 334 DEV classical
expressions (86%) in `trig-heavy` and the remaining 47 in `transcendental-heavy`; `polynomial-only`,
`sqrt-recip-heavy` and `mixed` are empty. With trig ops at 11% of generator draws, a > 50-node
expression almost always carries ≥ 3 trig nodes with random arguments, so "trig-heavy DEV" is DEV.
This is why §3a exists, and why the OOD families carry the answer.

## The ladder on each set

Ratios are per-expression `arm_cost@B / unguided_cost@B` (q1 / median / q3); regret is against the
empirical best any arm reached at any checkpoint of that expression; structural share and strict
precision are pooled over the first B applications. Full tables per set: the three `*-report.md`
files; per-(set, B, arm) rows: the `.csv`.

| set | B | arm | ratio q1 / med / q3 | improved / unch / worse | regret vs best (med) | gap vs unguided@4B (med) | structural share @B | strict precision @B | reaches best | ended at (med apps) |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| DEV | 100 | unguided@B | 1 | — | 96.17% | +51.66% | 0.589 | 0.021 | — | — |
| DEV | 100 | unguided@4B | — | — | 1.10% | 0 | — | — | — | — |
| DEV | 100 | control | 0.007 / **0.565** / 0.701 | 321 / 4 / 9 | 0.64% | 0.00% | 0.279 | 0.225 | 189 / 334 | 294 |
| DEV | 100 | linear | 0.004 / **0.537** / 0.681 | 323 / 4 / 7 | 0.38% | 0.00% | 0.322 | 0.224 | 199 / 334 | 293 |
| DEV | 200 | unguided@B | 1 | — | 51.45% | +16.89% | 0.596 | 0.030 | — | — |
| DEV | 200 | control | 0.522 / **0.699** / 1.000 | 245 / 71 / 18 | 0.20% | 0.00% | 0.403 | 0.142 | 189 | 294 |
| DEV | 200 | linear | 0.512 / **0.696** / 1.000 | 245 / 71 / 18 | 0.08% | 0.00% | 0.430 | 0.146 | 199 | 293 |
| `sh` | 100 | unguided@B | 1 | — | 20.48% | +0.18% | 0.906 | 0.011 | — | — |
| `sh` | 100 | unguided@4B | — | — | 17.37% | 0 | — | — | — | — |
| `sh` | 100 | control | 0.892 / **0.903** / 0.921 | 95 / 0 / 0 | 7.92% | **−7.88%** | 0.743 | 0.154 | 6 / 95 | 800 (all 95) |
| `sh` | 100 | linear | 0.892 / **0.904** / 0.918 | 95 / 0 / 0 | 7.97% | **−7.92%** | 0.764 | 0.147 | 6 / 95 | 800 (all 95) |
| `sh` | 200 | unguided@B | 1 | — | 20.33% | +2.24% | 0.913 | 0.013 | — | — |
| `sh` | 200 | control | 0.879 / **0.894** / 0.914 | 95 / 0 / 0 | 6.70% | −6.45% | 0.780 | 0.102 | 6 | 800 |
| `sh` | 200 | linear | 0.882 / **0.896** / 0.916 | 95 / 0 / 0 | 6.80% | −5.86% | 0.791 | 0.099 | 6 | 800 |
| `bezier` | 100 | unguided@B | 1 | — | 29.55% | +5.60% | 0.708 | 0.017 | — | — |
| `bezier` | 100 | unguided@4B | — | — | 22.68% | 0 | — | — | — | — |
| `bezier` | 100 | control | 0.910 / **0.910** / 0.949 | 80 / 0 / 0 | 17.87% | −2.64% | 0.750 | 0.082 | 0 / 80 | 800 |
| `bezier` | 100 | linear | 0.910 / **0.910** / 0.949 | 80 / 0 / 0 | 17.87% | −3.92% | 0.759 | 0.093 | 0 / 80 | 800 |
| `bezier` | 200 | unguided@B | 1 | — | 29.55% | +9.85% | 0.768 | 0.012 | — | — |
| `bezier` | 200 | control | 0.910 / **0.910** / 0.980 | 60 / 20 / 0 | 17.87% | 0.00% | 0.826 | 0.049 | 0 | 800 |
| `bezier` | 200 | linear | 0.857 / **0.885** / 0.926 | 60 / 20 / 0 | 11.00% | 0.00% | 0.822 | 0.047 | 0 | 800 |

Head-to-head (linear < / = / > control): DEV 131 / 134 / 69 (B=100), 114 / 202 / 18 (B=200); `sh`
30 / 43 / 22, 3 / 55 / 37; `bezier` 15 / 65 / 0, **41 / 39 / 0**. The one place the candidate-local
model separates from the control is `bezier` at B=200 (median ratio 0.885 vs 0.910, regret 11.0% vs
17.9%, never worse) — the polynomial neighborhoods (`Add`/`Mul`/`MulAdd`) are exactly what its
DEV-learned features are about. On `sh` at B=200 it is slightly *behind* the control (37 worse vs 3
better; within M).

Production context (the exact `production_saturation_probe`, wall-clock stop only): on `sh` and
`bezier` production quiesces on 95/95 and 80/80 expressions at a median **10,997 / 11,197**
applications (DEV: 1,671; 316 quiesced / 18 timeout of 334), i.e. ~110× the primary budget, and its
cost is 0.877 / 0.650 of unguided@800 at the median. Both guided arms at B=100 are worse than
production on 91/95 (`sh`) and 80/80 (`bezier`) expressions — on DEV they match or beat it on
125–142 of 334.

## Why D moved: the denominator, not the Guides

`D_A` is a shift in `m_A = median(cost_A@B / cost_unguided@B)`. On DEV the unguided anytime curve
has an enormous dynamic range between B and 4B — median truncation loss +51.7% at B=100, regret 96%
at B vs 1.1% at 4B — so a Guide that reaches the 4B-quality answer in B applications scores a ratio
near 0.5 (and a quarter of DEV expressions score ≈ 0.005). On both OOD families the unguided curve is
**flat and far from converged**: truncation loss +0.18% (`sh`) / +5.6% (`bezier`) between 100 and 400
applications, while regret vs the empirical best stays at 20% / 30% at B *and* 17% / 23% at 4B. The
unguided sweep spends 91% (`sh`) / 71% (`bezier`) of its first 100 applications on structural rules
(`doubling` alone: 37,314 unguided firings over the full `sh` runs), buys almost nothing per
application (strict precision 1.1% / 1.7%), and does not saturate these graphs until ~11,000
applications. A ratio of 0.90 against that baseline is a Guide that is 10% better than a baseline
that barely improved from 100 to 400 — every one of the 175 OOD expressions improved (0 unchanged,
0 worse), the guided arms at B=100 **beat unguided-at-4B** on both families (median gap −7.9% / −2.6%
to −3.9%; the registered 4B-approach clause holds everywhere), and they are still improving when the
800-application grid ends (0 of 175 quiesced, vs a median 294-application quiescence on DEV).

So the +0.34 shift is the same on a trig-dominant and a trig-free family because it measures the
same thing on both: **real structured kernels are large, slowly-saturating e-graphs with many live
rewrites, and at B=100–200 the binding constraint is the budget, not the ordering.** The registration's
fifth §5 row anticipated the mirror image ("both arms improve on `sh` → easier at this budget"); what
happened is the reverse — both arms *retain* a uniform ~10% advantage while the baseline's failure mode
changed from "catastrophic at B, converged at 4B" to "mediocre at every budget up to 4B". The
registered Y-clause (median ratio ≤ 0.837) fails on both families at B=100 and holds on both at B=200
(0.896 / 0.885 ≤ 0.910) — reported, not claimed: no claim was registered on OOD families.

## The trig diagnostic: do the identities pay on SH, and does either Guide let them fire?

Pooled over the set; cells are `fired (strict-positive, strict rate)` — firings within the first 100
applications, and over the whole run; `exprs` = expressions the arm fired the rule on at any point.
Strict-positive = the firing's output node is on that arm's own final extracted path. Rule indices are
`all_rules()` order (the parity rules share a `name()`, so the index is the key).

| idx | rule | arm | DEV @100 | DEV full run | DEV exprs | `sh` @100 | `sh` full run | `sh` exprs |
|---:|---|---|---|---|---:|---|---|---:|
| 39 | `half-angle-product` | unguided | 0 | 3,171 (0, 0.0%) | 17 | 0 | 3,275 (72, 2.2%) | 94 |
| 39 | | control | 0 | 23 (0, 0.0%) | 6 | **393 (127, 32.3%)** | 1,050 (134, 12.8%) | 93 |
| 39 | | linear | 0 | 23 (0, 0.0%) | 7 | **360 (134, 37.2%)** | 985 (134, 13.6%) | 93 |
| 38 | `reverse-angle-addition` | unguided | 0 | 5,714 (19, 0.3%) | 64 | 0 | 1,134 (50, 4.4%) | 94 |
| 38 | | control | 26 (11, 42.3%) | 197 (20, 10.2%) | 61 | 134 (0, 0.0%) | 662 (0, 0.0%) | 93 |
| 38 | | linear | 36 (16, 44.4%) | 209 (20, 9.6%) | 61 | 91 (0, 0.0%) | 619 (0, 0.0%) | 92 |
| 36 | `sin-angle-addition` | unguided | 0 | 1,953 (2, 0.1%) | 64 | 0 | 1,025 (0, 0.0%) | 94 |
| 36 | | control | 28 (0, 0.0%) | 278 (1, 0.4%) | 61 | 257 (0, 0.0%) | 779 (0, 0.0%) | 93 |
| 36 | | linear | 31 (0, 0.0%) | 286 (1, 0.3%) | 61 | 219 (0, 0.0%) | 733 (0, 0.0%) | 92 |
| 37 | `cos-angle-addition` | unguided | 1 (0) | 1,766 (5, 0.3%) | 69 | 0 | 307 (0, 0.0%) | 51 |
| 37 | | control | 51 (1, 2.0%) | 412 (5, 1.2%) | 65 | 22 (0, 0.0%) | 202 (0, 0.0%) | 48 |
| 37 | | linear | 48 (1, 2.1%) | 401 (4, 1.0%) | 65 | 22 (0, 0.0%) | 176 (0, 0.0%) | 48 |
| 40 | **`pythagorean`** | unguided | 0 | 225 (0, 0.0%) | 5 | 0 | **231 (0, 0.0%)** | **19** |
| 40 | | control | 0 | **0** | 0 | 0 | **0** | 0 |
| 40 | | linear | 0 | **0** | 0 | 0 | **0** | 0 |
| 34 | `even-negation` (Cos) | unguided | 99 (30, 30.3%) | 6,039 (396, 6.6%) | 257 | 0 | 0 | 0 |
| 34 | | control | 892 (371, 41.6%) | 1,659 (400, 24.1%) | 257 | 0 | 0 | 0 |
| 34 | | linear | 1,135 (395, 34.8%) | 1,764 (400, 22.7%) | 257 | 0 | 0 | 0 |
| 20 | `doubling` (enabler) | unguided | 8 (0) | 31,203 (66, 0.2%) | 202 | 0 | 37,314 (0, 0.0%) | 94 |
| 20 | | control | 162 (17, 10.5%) | 1,796 (54, 3.0%) | 197 | 286 (0, 0.0%) | 3,003 (0, 0.0%) | 93 |
| 20 | | linear | 355 (32, 9.0%) | 1,835 (50, 2.7%) | 196 | 295 (0, 0.0%) | 2,875 (0, 0.0%) | 93 |

Idx 30/31 (`odd-negation` Sin/Tan) never match on `sh` (no negated arguments in the family); idx
32/33 never fire anywhere. On `bezier` every trig rule fires 0 times in every arm (precondition
confirmed); only `doubling` matches there (unguided 1,480 / guided 360 full-run firings on 20
expressions, 0 strict-positive).

**Do trig identities pay on SH kernels? One does, decisively.** `half-angle-product`
(sin x·cos x → sin 2x / 2) has a strict-positive rate of **32–37% inside the first 100 applications**
on `sh` — the highest of any rule in that prefix (the guided arms' pooled strict precision there is
15%) — and lands on the final extracted path of 134 firings across the 95 expressions, for every arm
including unguided (72 of its 3,275). On DEV the same rule fired 3,171 times under unguided and was
strict-positive **zero** times. `reverse-angle-addition` pays for unguided on `sh` (50 positives, 4.4%)
but for neither Guide (0 of 662 / 619) although both fire it on the same 92–93 expressions — a
plausible reading is that the Guides fire it before the `doubling`-built classes it needs exist and
single-firing dedup never re-offers the key, but that ordering is not measured here. The two
angle-addition expansions and the enabler never pay on `sh` for anyone.

**Does either Guide let them fire? Yes — both, equally, and this is the part of the registered story
that was wrong.** The registration read the control's 0.0045 score for `half-angle-product` as "never
fire trig identities". But `PerRuleRateGuide` is an *ordering*, not a filter, and the strict labels
give the structural rules exactly 0.0; on a trig-dense candidate pool a rule scored 0.0045 sorts
ahead of `commutative`/`associative`/`distribute` and fires immediately — 393 / 360 pooled firings in
the first 100 applications, ~4 per expression, at 32–37% precision, where unguided fires it 0 times
in its first 200. The prior did not suppress the paying rule; it was already correct about it in
*rank* even though its *rate* was learned from a corpus where the rule never paid. The linear model's
context feature had nothing to add because the control already had the ordering right.

**The one rule the prior does suppress is `pythagorean`, and it never pays for anyone.** Its control
score is exactly 0.0 (0 firings in 4,143 TRAIN/DEV expressions), it ties with the structural rules at
the bottom of every candidate list, and neither Guide fires it once in 800 applications on `sh`.
The family does offer the match: the unguided sweep fires it 231 times across 19 `sh` expressions
(the `sh-power` band-energy forms) — and **0 of 231 firings are strict-positive**, even at full
saturation, with the same 0 / 225 on DEV. Under this labeler, sin²x + cos²x → 1 is never on the
extracted path. A plausible mechanical reason, flagged for the labeling stream rather than resolved
here: the rule's output is the literal constant `1`, an e-node that almost always already exists in
the graph, so the firing's effect is a *union* with a pre-existing node whose provenance is not this
application — the strict label (output node on the extracted path) cannot credit a rewrite whose
output is a pre-existing term, by construction. If that is what is happening, it is a labeler blind
spot on precisely the identities JP named (the same class as the structural-enabler 0% the tightened
labeler was designed for), and no amount of corpus coverage will lift `pythagorean`'s base rate
above 0 until it is fixed. That is a correctness question for the tightened-label stage; it is
recorded here as a referral with the numbers, not chased.

## What this means for Round 2 (per registration §5)

- **H_null row (primary).** Architecture is not the lever; do not add capacity. The corpus taught the
  candidate-local model the same global prior — but note *what* that prior gets right: on `sh` the
  paying identity already fires under both arms. The coverage fix (a structure-aware TRAIN family
  with shared-argument trig — SH, rotations, Fourier sums; `sh`/`bezier` themselves stay DEV) is
  still the right next lever for the *rates* (§4 acceptance metric: every trig rule ≥ 100 TRAIN
  positives), and after it this registration's test is re-run unchanged.
- **Bézier-fails row.** The +0.34 shift is family-structure, not rule-prior. Both OOD families run the
  guided arms to the 800-application grid end still improving, with 8–18% regret left against a best
  that production reaches only after ~11,000 applications. The dedup-closure / budget-regime
  investigation (Round 1 §"Where the residual regret lives") is promoted ahead of labels: the
  question is what the anytime curve of a guided arm looks like at B = 800–3,200 on structured
  kernels, and whether a second pass over resolved keys or the tightened-label enabler credit closes
  the 8–18%.
- **Referral (labels).** `pythagorean` at 0 / 456 strict-positives over 24 expressions where it
  matched is either a true non-payer under the latency prior or a strict-label blind spot for
  constant-output rewrites; decide which before designing the coverage family around it.
- **The DEV `trig-heavy` stratum is DEV** (287 / 334) — future registrations should not rely on §2's
  count rule for a trig stratum on this generator; use a named family.
- Kill semantics unchanged: this round is an evaluation; it neither fires nor advances the
  5-clean-rounds gate.

## What this round did not do

- No retraining, no checkpoint change, no label change (registered constraint).
- No FINAL run; `corpus_final.bin` was not opened.
- No strict-oracle arm (as in Round 1).
- No per-form breakdown of `sh` (`sh-direct` / `sh-expanded` / `sh-power`): the corpus entries do not
  carry the form, and the generator was not modified to record it (it would have re-derived the
  same 100 entries, but it is a generator change and out of scope for an evaluation round).
- The 5 `sh` entries with ≤ 50 arena nodes were not evaluated (rapid band; no claim registered).

## Files

- `docs/results/2026-09-01-phase3-round1b-domain-shift.{md,json,csv}` — this document; the combined
  per-set / per-tier / per-arm tables and D statistics; one CSV row per (set, stratum, B, arm).
- `docs/results/2026-09-01-phase3-round1b-{dev,sh,bezier}.{jsonl,json}` and `*-report.md` — the
  harness's per-expression rows (with per-rule-index histograms), aggregate JSON, and generated
  report for each corpus.
- `docs/results/journal.jsonl` — one `phase3_round1b_domain_shift` record.
- `docs/plans/2026-09-01-phase3-round1b-domain-shift-registration.md` §7 — results against the gates.

## Harness corrections landed after these numbers were minted (2026-09-02)

The 2026-09-02 review round found five defects in the harness code this PR adds. All are fixed on
this branch; the numbers above were produced *before* the fixes, and are left as recorded rather than
silently restated under new code. What each one can and cannot move:

| Fix | What was wrong | Effect on the numbers above |
|---|---|---|
| Guided loop: siblings sharing a `CandidateKey` are now **all** attempted before the key is marked seen (`egraph/saturate.rs`) | Several nodes of one e-class can match one rule and share the key. Only the first was applied; if it recorded an application that changed nothing — 91% of recorded applications are exact no-ops (`2026-08-30-guide-scope-saturation-delta.md`) — the key was marked resolved and a productive sibling was never attempted, so the loop could report `Quiesced` with applicable work left. | **Can move both guided arms** (`control`, `linear`), never the `unguided` arm, which does not go through this loop. Direction: guided arms could only have been *under*-saturated, so both D statistics were measured against a slightly weakened guided side. The registered comparison is control-vs-linear and both arms shared the defect, so the H_null verdict is not obviously at risk — but it has not been re-measured, and that is the honest statement. |
| `saturate_with_limits` records a mid-sweep `ClassCap` instead of falling through to `Quiesced` (`egraph/graph.rs`) | The per-rule loop breaks on `batch.node_count() > max_classes`; a truncated sweep with zero unions then reported `Quiesced`. The cap is checked against a different quantity than the loop head, so the loop head does not catch it. | Stop-reason reporting only — no cost or ratio changes. Affects `production_saturation_probe` rows and any `ended` field sourced from this loop. |
| `live_at_B` is classified from each checkpoint's own stop reason, not `ended_at_apps > b` (`phase3_unguided_baseline.rs`) | The budget is crossed at rule-sweep granularity, so a run can quiesce while overshooting B (a B=100 checkpoint finishing quiesced at 150 applications). That finalized checkpoint was counted live. | Changes `live@B`, `live_med%`, and `live>0 cnt` in `2026-09-01-phase3-unguided-baseline.{md,csv,json}` — the live-conditioned columns were inflated. The unconditioned truncation-loss columns are unaffected. |
| `average_precision` advances the PR curve once per distinct score (`training/guide_linear.rs`) | Ties were processed in input order, so AP depended on row order — worst for `PerRuleRateGuide`, where every candidate of a rule carries the same score. | Changes the reported `dev_pr_auc` for the control arm in `2026-09-01-control-guide-comparison.{md,json}` and `2026-09-01-train-guide-report.{md,json}`. Does not touch AUC-ROC (which already averaged ranks within a tie group) and does not touch any at-budget ratio. |
| `gen_sh_corpus` filters on the compacted unique-node count and merges into `corpus_dev_ood.bin` instead of overwriting it (`bin/gen_sh_corpus.rs`) | `node_count_subtree` counts a shared node once per parent; SH deliberately shares its trigonometric basis, so the filter admitted entries the evaluator then classified below `classical` — the 95-of-100 disclosed above. Separately, the generator overwrote the shared OOD corpus, deleting `dev_bezier_*` if run second. | A regenerated `sh` corpus will differ from the one behind these numbers (the 5 rapid-band entries would not be admitted, and replacements would be drawn). **`corpus_dev_ood.bin` MD5 `0c7cbe710c50175afb3cd91f60960b64` is the corpus these results were measured on**; regenerating it invalidates the reproduce line at the top of this document. |

Re-running the round under the corrected harness is Round 2's first item, not a change made inside a
results document.
