# NUMBERS.md — provenance map for `2026-08-egraph-nnue-parity.md`

Every number in the paper traces to one of the sources below. Numbers with no
valid trace do not appear in the paper. Last reconciled: 2026-09-01 (Round-3
final form).

## Reproduction debt: what is named here but not in the repository

Four things this paper cites are not recoverable from this commit. Listed
together so a reader hits the whole list once rather than discovering it a
row at a time, and so a future run knows exactly what to commit.

| Missing | Cited by | Why it is gone |
|---|---|---|
| `D2a` / `D3` per-kernel JSONL | extraction overheads, gate counts, floor restatements, mechanism cross-reference, flip analysis | written under `pixelflow-pipeline/data/` (gitignored) on the run machine; those worktrees no longer exist |
| The Round-2a and Round-3 checkpoints | Appendix B's `inspect_flip` recipe, which requires `--r2a-weights` and `--r3-weights` | never committed; the journal records weight *identities* and former paths, not the model bytes |
| `inspect_flip` itself | Appendix B's recipe cannot be run even with the checkpoints | the tool inspected the extraction head, and #1093 deleted that program — `ExprNnue`, `IncrementalExtractor` and `EGraph::saturate_with_limit` are all gone, so the example no longer compiles and was removed rather than left broken. Recovering §5.4's flip analysis means resurrecting the head, not just the weights. |
| Source revision `b706cb67-dirty` | the §5.1 learning curve and the best-calibration checkpoint | the base revision is absent from `git rev-list --all` and the uncommitted patch was never stored; a diff hash identifies a patch, it cannot reconstruct one |
| The Round-2b journal line | every `R2B` figure in §5.3 | that worktree was discarded before the line was committed |

Consequences, stated plainly: **Appendix B's flip reproduction cannot be run
from this commit**, retraining will not reproduce the exact forms and
predicted costs §5.4 quotes (different weights, different decisions), and
the §5.1 curve cannot be audited against the trainer that produced it.

The fix is the same in every row and is not a re-derivation from what is
here: re-execute with the per-kernel output, the checkpoints, and a
committed source revision, and commit all three to LFS beside
`journal.jsonl`. Until then these figures are attributable but not
reproducible, which is a weaker guarantee than the header of this file used
to claim.

## Harness defects affecting every timing row (2026-09-01 review)

Four, all confirmed in code and none repaired retroactively — see §5.0 of the
paper. In short: reported ratios include the 4.272 ns call overhead
(`bench_extraction_3way.rs:2589` aggregates `bench.ns`, not `adjusted_ns`),
which biases every ratio toward 1; the NNUE and static arms use different
*search* algorithms, not only different cost models; the bootstrap resamples
kernels though `training::split` defines the `(band, seed)` family as the
split unit; and no Round-3-versus-Round-2a paired comparison was run. Every
geomean, CI and LOO range below inherits all four.

## Artifact availability (read this before trusting a `D2a`/`D3` row)

`docs/results/journal.jsonl` is committed (LFS) and carries the run-level
numbers. **The per-kernel `D2a` and `D3` JSONL files are not in this
repository** — they were written under `pixelflow-pipeline/data/` on the run
machine, which `.gitignore` excludes, and the Round-2a/Round-3 worktrees no
longer exist. `J@…​.data_jsonl` records their filenames, which is a name, not
a copy.

Every row sourced `D2a`, `D3`, `RUN3` or `FLIP` was therefore re-derived
against those files *in-session* and cannot be re-derived by a reader with
only this repository. That covers the extraction-overhead figures, the
gate/divergence counts, the A/A floor restatements, the mechanism
cross-reference and the flip analysis. The paper's header says so too.

Closing this needs the runs re-executed with the per-kernel output committed
(LFS, alongside `journal.jsonl`), not a re-derivation from what is here.

## Source legend

| Key | Source |
|---|---|
| `J@<ts>` | `docs/results/journal.jsonl`, the line with that `ts_unix` (this worktree, branch `claude/workshop-writeup`) |
| `D3` | Round-3 per-kernel data: `pixelflow-pipeline/data/bench_extraction_3way_1788248016_9e275056fb7cc710.jsonl` (referenced by `J@1788248059.data_jsonl`). **Not committed** — see the artifact note below. |
| `D2a` | Round-2a per-kernel data: `bench_extraction_3way_1786997003_4ac3aa76ff033f2e.jsonl` (referenced by `J@1786997047.data_jsonl`). **Not committed** — see the artifact note below. |
| `FLIP` | `pixelflow-pipeline/examples/inspect_flip.rs` pass over `D3` + the Round-2a checkpoint on identical saturated e-graphs (Round-3 session report, 2026-08-31) |
| `NOTES` | `docs/plans/2026-08-17-egraph-vsa-nnue-research-notes.md` |
| `PLAN` | `docs/plans/2026-08-05-egraph-nnue-research-workflow.md` |
| `GUIDE` | `docs/plans/2026-08-31-guide-design-revision.md` (branch `claude/guide-scoping`, PR #1067) and its results docs `docs/results/2026-08-30-guide-headroom.md`, `2026-08-30-guide-scope-saturation-delta.md`, `2026-08-30-oracle-filtered-budget-curves.md` — re-verified against `origin/claude/guide-scoping` 2026-09-01 |
| `AUDIT` | `docs/results/2026-08-05-bench-harness-integrity-audit.md` |
| `R0708` | `docs/results/2026-07-08-extraction-3way.md` |
| `R2B` | **Round-2b session (2026-08-27).** Its journal line is *not* present in this worktree's `journal.jsonl` (the Round-2b worktree no longer exists on disk); numbers are carried verbatim from that session's draft of this paper and the project memory record. Flagged, not silently blended. |
| `RUN3` | Round-3 unattended-run report (2026-08-31): run-level metadata not serialized into the journal line (sentinel calibration, machine load, mechanism sample), computed from `D3` and the run log in-session |

## Round 3 (final result — §5.4, abstract, arc table)

| Number | Value | Source |
|---|---|---|
| geomean nnue/static | 1.0082 (1.0082271584…) | `J@1788248059.geomean_nnue_static` |
| 95% bootstrap CI | [1.0031, 1.0133] | `J@1788248059.geomean_nnue_static_ci_lo/hi` |
| pairs / W / L / T | 716 / 235 / 449 / 32 | `J@1788248059.nnue_static_pairs/ratio_wins/ratio_losses` (T = 716−235−449) |
| ratio median / Q1 / Q3 | 1.0066 / 0.9934 / 1.0239 | `J@1788248059.ratio_median/ratio_q1/ratio_q3` |
| LOO geomean range | [1.0077, 1.0089]; most influential `dev_b22_f03_00503`, 0.07% shift | `J@1788248059.loo_geomean_min/max/loo_most_influential`; shift = `RUN3` |
| A/A noise floor | ±0.07% (0.0729%) | `J@1788248059.noise_floor_pct`. (The draft's "MAD ±0.03%; worst per-kernel \|Δ\| 4.80%" were `RUN3`-session values that did not reproduce from `D3` under any tried estimator — removed from the paper 2026-09-01.) |
| static/noswap geomean | 0.5433 | `J@1788248059.geomean_static_noswap` (n=724 per `RUN3`) |
| failure rates noswap/static/nnue | 0.0% / 7.65% (60/784) / 8.16% (64/784) | `J@1788248059.failure_rate_*`, `syn_static_fail`, `syn_nnue_fail` |
| kernels excluded at phase-A gate | 0/784 | `J@1788248059.kernels_excluded`, `corpus_entries` |
| same-form miscompiles | 0 of 2352 gates (1568 extracted) | rate: `J@1788248059.same_form_miscompile_rate`=0.0; gate counts: `RUN3`/`D3` |
| cross-form divergence | 92/1568 = 5.87% of extracted = 3.91% of all gates | `J@1788248059.cross_form_divergence_rate`=0.058673…; counts: `RUN3`/`D3` |
| ill-conditioned disagreements (metadata) | 559 | `RUN3`/`D3` |
| gates that bounded nothing | 32 | `RUN3`/`D3`. Excluded from the bound-coverage and same-form-miscompile denominators, **not** from the cross-form one above: `GateTally::cross_form_rate` divides by every extracted-policy gate, so 5.87% is 92/1568 with these 32 included (92/1536 would read 5.99%). |
| compile failures / oracle-unsupported | 0 / 0 | `RUN3`/`D3` |
| bound coverage | mean 74.0% of 72 grid points over 2228 gated policies; worst 1.4% | `J@1788248059.bound_coverage_mean_pct`; per-policy count + worst: `RUN3` |
| extraction overhead (mean, n=784) | nnue 3595.06 µs, static 83.67 µs (~43×) | `RUN3`/`D3` |
| verdict string (quoted verbatim in §5.4) | "NNUE approx-ties… (geomean −0.82%, 95% CI [−1.33%, −0.31%]…)" | `J@1788248059.verdict` (`verdict_censored`=false, `gate_alarms`=[]) |
| protocol (saturate=40, top_k=8, repeats=5, boot 10000, seeds, tuples=64, grid=8) | — | `J@1788248059.config.protocol` |
| environment (aarch64 macOS, baseline ISA, no FMA, release) | — | `J@1788248059.config.environment` |
| weights identity / path | ea54dfc3470702d8 | `J@1788248059.weights` |
| sentinel calibration / call overhead | 7.446 ns (231 samples) / 4.272 ns | `RUN3` |
| machine conditions (load 9.45/13.23/14.11; `guide_headroom` at 100% CPU, ps-confirmed; battery 83% AC; no regime abort; single clean run) | — | `RUN3` |
| checkpoint DEV calibration (ρ 0.9887, MAE 0.133) | — | `J@1788247810` (same `weights_identity` ea54dfc3… as the bench) |

## Round 3, random-init intermediate bench (§5.4 table, arc table)

| Number | Value | Source |
|---|---|---|
| geomean / CI | 1.0118 [1.0058, 1.0182] | `J@1788231974.geomean_nnue_static` + ci_lo/hi |
| pairs / W / L / T | 717 / 208 / 487 / 22 | `J@1788231974` (T = 717−208−487) |
| median [Q1, Q3] | 1.0110 [0.9948, 1.0306] | `J@1788231974.ratio_median/q1/q3` |
| LOO range | [1.0102, 1.0126] | `J@1788231974.loo_geomean_min/max` |
| A/A floor | ±0.04% (0.0431%) | `J@1788231974.noise_floor_pct` |
| static/noswap | 0.5437 | `J@1788231974.geomean_static_noswap` |
| failure rates static/nnue | 7.65% / 7.78% | `J@1788231974.failure_rate_static/nnue` |
| checkpoint calibration (ρ 0.9856, MAE 0.164) | — | `J@1788230784` (same `weights_identity` bbf86c96… as this bench) |

## Round-3 cold-start learning curve (§5.1, §2.3)

Latency-prior-seeded (source_rev b706cb67-dirty):

| Row | MAE / ρ / bias / σ-ratio | Source |
|---|---|---|
| 4,354 (1k) | 0.162 / 0.9860 / −0.025 / 0.972 | `J@1788246968` |
| 11,312 (8k) | 0.150 / 0.9874 / −0.019 / 0.972 | `J@1788247107` |
| 35,151 (32k) | 0.143 / 0.9873 / −0.014 / 0.973 | `J@1788247368` |
| 53,026 (50k) | 0.133 / 0.9887 / −0.007 / 0.975 | `J@1788247810` |

Random-init ablation (source_rev 614e116c-dirty): 1k → MAE 0.173, ρ 0.9827
(`J@1788230699`); 8k → MAE 0.164, ρ 0.9856 (`J@1788230784`).

Round-2a-recipe 50k reference (frozen embeddings): MAE 0.145, ρ 0.9875
(`J@1786996772`); its full curve rows 0.184/0.9801 (`J@1786995939`),
0.189/0.9794 (`J@1786996015`), 0.166/0.9829 (`J@1786996150`),
0.156/0.9858 (`J@1786996403`).

## Round-3 mechanism / flip analysis (§5.4)

All from `FLIP` (nodes-only pass, both checkpoints on identical saturated
e-graphs; per-kernel classes and predicted costs recorded in the Round-3
session report):

- 174 W↔L flips vs Round-2a checkpoint; W→L = 109 (R3 tree bigger 72 = 66%,
  smaller 13 = 12%, same 24 = 22%); L→W = 65 (bigger 24 = 37%, smaller 26 =
  40%, same 15 = 23%); mean node delta +0.81, median +1.
- Named examples: `dev_b09_f03_00278` (W→L, 35→31 nodes, predicted
  3.986→4.507, declines recip+recip+mul_add fusion, keeps pow(−2));
  `dev_b27_f03_00615` (W→L, 440→433, 5.605→6.074);
  `dev_b33_f07_00773` (W→L, 712→679, 6.303→6.824);
  `dev_b01_f07_00109` (W→L, 11→19 nodes, predicted cost 2.708→2.514 —
  model predicts cheaper, measurement flips to loss).

## Round 2a (§5.2, arc table)

| Number | Value | Source |
|---|---|---|
| geomean / CI | 1.0037 [0.9982, 1.0089] | `J@1786997047.geomean_nnue_static` + ci_lo/hi |
| pairs / W / L / T | 719 / 276 / 409 / 34 | `J@1786997047` |
| median [Q1, Q3] | 1.0057 [0.9895, 1.0280] | `J@1786997047.ratio_median/q1/q3` |
| LOO range | [1.0031, 1.0044] | `J@1786997047.loo_geomean_min/max` |
| A/A floor | ±0.07% (0.0723%) | `J@1786997047.noise_floor_pct` |
| failure rates static/nnue | 7.65% / 7.40% | `J@1786997047.failure_rate_*` |
| same-form miscompiles / cross-form | 0 / 5.48% | `J@1786997047.same_form_miscompile_rate/cross_form_divergence_rate` |
| static/noswap | 0.5440 | `J@1786997047.geomean_static_noswap` |
| bound coverage | 74.2% | `J@1786997047.bound_coverage_mean_pct` |
| extraction overhead (median 192.6 µs vs 14.0 µs; means 5.36 ms vs 92 µs; p99 80 ms; ~13% of e-graph pass; p99 4.5% of blitz budget) | — | `D2a` `policy_prepared` records, successful extractions only (nnue n=726, static n=724) — re-derived exactly from `D2a` 2026-09-01 (5357.12/91.56/192.58/13.94 µs, p99 80,194 µs). Round-3's counterpart in §5.2 (3.60 ms / 83.7 µs) is over all 784 kernels incl. failed policies' `extract_us` — also exact from `D3`. Budget-share framings (~13%, 4.5%) are 2026-08-27 draft-session arithmetic. |
| per-pair floor stats (median \|Δ\| 2.5%, IQR 1.0–5.4%; median repeat spread 9.0% range-based, 84% of decisions under own spread; 3.4%/61% IQR-based) | §6 | `D2a` measurement records, re-derived 2026-09-01: per-kernel median-of-5 adjusted_ns; \|Δ\| median 2.52%, IQR 1.04–5.41%; spread = (max−min)/median of static repeats → median 8.97%, 84.3% under; IQR-of-repeats spread → 3.42%, 60.6% under. (The draft's 6.4%/78% used an unrecorded estimator and was replaced with these reproducible ones.) |
| scoping fix (−15–16% mean extraction time, digest-verified no form change) | §5.2 | 2026-08-27 draft session (code landed; not a journal quantity) |

## Round 2b (§5.3, arc table) — carried, flagged

All Round-2b numbers are `R2B` (journal line not present in this worktree —
see legend): geomean 1.0153, CI [1.0097, 1.0213]; 1016 pairs
(333W/631L/52T); LOO [1.0138, 1.0158]; A/A ±0.06%; static/noswap 0.5444;
DEV re-mint 784→1120; pairwise-accuracy replicates +0.038/+0.011/+0.162
(3/3 seeds at λ=1.0); aggregate estimate-op counts 8072 vs 7591;
conservative on 187 kernels (16.7%); moved toward static on 107/187
(57.2%), matched/exceeded on 85; of the 107: 38 wins, 53 losses, 6 ties
= 97 with an outcome (39.2% wins; 35.5% only if the 10 unaccounted are
counted as non-wins — see the R2B arithmetic note); `backward_pairwise` gradient-check failure (one
shared-trunk entry 20% relative vs 5% tolerance).

## Rounds 0–1 and lineage (§4.4, §4.5, §5.5, NOTES §1.5)

| Number | Value | Source |
|---|---|---|
| 2026-07-08 result | 1.0669 (~6.7% slower, ~31× extraction cost); pre-timebase-fix | `R0708` |
| Round 0 | 1.0389, 7W/18L | `NOTES` §1.5 |
| Round 1 geomean / CI / pairs / W/L/T / LOO | 1.0181 [1.0106, 1.0254] / 316 / 73W/238L/5T / [1.0171, 1.0198] | `J@1786732076` (also `NOTES` §1.5); T=5 from NOTES |
| Round 1 status | INVALID — cross-form 12.31% > 10% | `J@1786732076.gate_alarms/verdict` |
| Round 1 A/A | ±0.05% (0.0460%) | `J@1786732076.noise_floor_pct` |
| Round 1 static/noswap | 0.51 (0.5091) | `J@1786732076.geomean_static_noswap` |
| censored runs table | 0.9285 (nnue fail 41.7%), 1.0501 (nnue fail 15.0%), 0.8815 (cross-form 16.7%) | `J@1786294245`, `J@1786298796`, `J@1786298839` |
| skew fix deltas (MAE 0.975→0.181, bias −0.886→+0.020, ρ 0.9624→0.9799, σ-ratio 0.68→0.93) | §4.5, NOTES table | `J@1786421873` (before) vs `J@1786421923`/`J@1786730044` (after); `NOTES` §1.5 |
| Round-1 conservatism counts (26 vs 14 substitutions, n=12; Neg to zero 8/12; larger DAGs 12/12) | §5.3 | `NOTES` §1.5 |
| Sqrt/Pow misprice (Sqrt=15 > Pow=12; measured 60.2 ns vs 21.3 ns) | §4.5 context | `NOTES` §1.5 |
| timebase ratio 125/3 = 41.67× / tick 41.67 ns | §4.1, §4.5 | `AUDIT` (tick, H-items); timebase fix lineage `PLAN` §0.1 |
| 11–19% post-build daemon drift; ≥50% regime-abort threshold | §4.1 | `PLAN` §0.3 (line ~214) |

## Corpus and tiers (§3)

| Number | Value | Source |
|---|---|---|
| tier sizes TRAIN/DEV/FINAL | 3359 / 784 / 129 (17 named + 112 band-12) | capstone mint (2026-08-27 session; DEV size confirmed by `corpus_entries`=784 on `J@1786997047` and `J@1788248059`) |
| quarantine 52/4324 (1.203%), 0 miscompiles, 78.1% mean bound coverage | capstone mint quarantine sidecar (2026-08-27 session) |
| TRAIN/DEV byte-identical across 2a and 3 | `train=c5b6df3a…;dev=1daa131a…` | `J@1786995939…1786997047.config.corpus_identity` vs `J@1788246968…1788248059` |
| FINAL re-mint post-refactor, 0 miscompiles | `final=89c48822…` → `final=90b84039…` | `J@1786995939` vs `J@1788230699` `config.corpus_identity`; 0-miscompile: Round-3 mint report (`RUN3` header: "corpus freshly minted post-refactor … 0 miscompiles") |
| decision band ±5%, kill gate 5 rounds, honest-negative fallback | `PLAN` §4.3, §6 |
| corpus-ranking baselines: static table ρ 0.9438, bare op count ρ 0.9486 | `NOTES` §1.5 — **Round 1's 392-kernel DEV tier**, not Round 3's 784. Any comparison against a Round-3 ρ is cross-population; see §5.1. |

## Incrementality and Guide economics (§5.6, §9)

| Number | Value | Source |
|---|---|---|
| extraction candidate delta: median 44.9%, p25 17.6%, p75 100%, p90 180%; 11% under 10%; depth-strip 51.5% vs 56.8%; 2,323 rebuilds / 0 deltas | `NOTES` §4 (marked [MEASURED]) |
| saturation: 91.1% zero-delta applications (4,096,878/4,495,274); state-changer median edge-delta 0.14%, p90 0.79%; ~728× implied (doc's own 1/median figure); deltas ~320× smaller than extraction's 44.9% (paper-side division) | `GUIDE` (`2026-08-30-guide-scope-saturation-delta.md`, 1,512 expressions) |
| 90.4% of scored candidates commit nothing (431,787 committed of 4,495,274 evals) | `GUIDE` (same doc) |
| oracle headroom: labeler median 38.2% (2.6×), Q1 0.333 Q3 0.527; strict median 2.9% (34×), Q1 0.006 Q3 0.058; 800 expressions, 8,729,067 applications, 6,274,873 labeler-LB, 12,390 strict-LB; 62-rule library; bounds Spearman ρ = 0.35 (n=55 fired rule instances); per-class split (structural/congruence 60–85% labeler vs ~0% strict; numeric/transcendental 6–17% under both) | `GUIDE` (`2026-08-30-guide-headroom.md` + plan §2.1). The source doc's 2026-09-01 correction retracts its earlier ρ ≈ 0.02 as an error never computed from the harness; paper updated to 0.35 the same day. |
| budget-curves null: 97.8% (220/225) quiesce or hit ~15,000-class cap before coarsest checkpoint | `GUIDE` (plan §2.3 / `2026-08-30-oracle-filtered-budget-curves.md`) |
| two-stage plan (cold-start on strict labels ∥ tighten union-causality); quality-at-budget framing | `GUIDE` plan §0, §3 |

## Discipline notes

- **Budget-only framing:** no claim in the paper depends on saturation
  reaching a fixpoint; quiescence is described as a diagnostic only
  (`GUIDE` plan §0 is the binding statement).
- **FINAL untouched:** no journal line consumes `corpus_final.bin` with
  `final_eval=true`; §3 and §8 state it.
- **R2B arithmetic note:** of the 107 kernels that moved toward static's
  substitution count, the carried outcome split is 38W/53L/6T = 97; the 10
  unaccounted kernels presumably lacked surviving paired timings (policy
  failures), but that reason is unrecoverable without the R2B journal line
  and is therefore not asserted in the paper. §5.3 accordingly reports the
  win rate over the 97 kernels that have an outcome (39.2%), not over all
  107 (35.5%) — the latter silently counts the 10 unaccounted as non-wins.
- **R2B trace debt:** the Round-2b journal line should be recovered from
  the 2026-08-27 session's worktree backup if one exists, or the run
  re-executed after the regalloc boundary; until then every `R2B` number
  is draft-carried and labeled as such above.
- **Regalloc boundary:** per the research-workflow memory and §9, runs must
  not be compared across the register-allocator overhaul; `config.source_rev`
  on every journal line is the separator.
