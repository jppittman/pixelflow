# The extraction DP as a fixpoint: what the witnesses said to fix, fixed

**Date:** 2026-09-08
**Change:** `pixelflow-search/src/egraph/extract.rs` — both DP passes settle
in cost order (Knuth's AND-OR Dijkstra) instead of one DFS post-order, and
the repair stage is deleted from the extraction path.
**Denotation it discharges:**
`docs/results/2026-09-08-extraction-witnesses.md` §8 item 1, which was
`docs/results/2026-09-02-extraction-gap.md`'s defect (i), asked for on
2026-09-02 and never done.
**Instrument:** the same `extraction_witnesses` bin over `egraph::witness`,
unchanged, run at the same budget ladder against the same corpus.

> **What is deterministic.** `objective` (`ChoiceCost::dag`, the number the
> extractor minimizes), `dag_cost` (the sweep's unweighted headline column),
> `live_classes`, `tied_classes` and every witness classification are
> functions of the term and the graph. Only `seconds` is wall clock; the box
> ran at load averages 7–30 during these runs and no timing claim is made.
> **The graph is the control:** `classes_at_stop` and `applications` are
> identical to the baseline on **every** row of every table below, so the
> only thing that moved is extraction.

## 0. The answer

The witnesses said 29 % of the frontier classes holding them were
**CYCLE-PRICED** — the DP had no opinion, and the pick came out of
`repair_choices_well_founded`, which has no cost model. Removing that stage
by making the DP a fixpoint:

| | shaders (12 × 5 caps) | chrome (held out, 5 caps) |
|---|---|---|
| Σ `objective` | 1,240,723,402 → **1,126,876,589** (−9.2 %) | — |
| Σ `dag_cost` | 23,962 → **23,429** (−2.2 %) | — |
| worst 5k→100k **objective** ratio | ×**1.754** (`metaballs`) → ×**1.007** (`julia_set`) | ×1.405 → ×**1.070** |
| worst 5k→100k `dag_cost` ratio | ×1.742 → ×1.784 (a different kernel; §3) | ×1.424 → ×**1.070** |
| objective witnesses | **26 → 4** | 7 → 7, mean gap 17.9 % → **4.8 %** |
| first divergences settled by `repair` | 18 → **0** | 1 → 0 |
| frontier classes labelled CYCLE-PRICED | 21 → **0** | 1 → **0** |

**Does more budget stop hurting? On the number the extractor minimizes,
yes.** No shader kernel's objective rises by more than 0.7 % across the
whole 5k → 100k ladder, against +75 % for `metaballs` and +35 % for
`julia_set` before. Chrome's +40.5 % becomes +7.0 %.

**What it does not do.** Chrome is 4.2–5.5 % *worse* at the three small caps
and 20.7–22.6 % better at the two large ones (§4) — the fixpoint is exact
per class, and the sharing-aware DP is still a heuristic over a DAG, so
"exact where it was undefined" is not "better everywhere". And one shader,
`cosine_palette`, rises ×1.78 in `dag_cost` while its objective *falls*
23 % (§3): the two columns disagree, as they did in the witness run.

## 1. What changed in the extractor

Both passes walked one DFS post-order. A class whose child was still on the
stack could not be priced, so it took `CYCLE_COST` and was never revisited —
and saturated graphs are cyclic, commutativity alone being enough. The pick
for such a class then came out of `repair_choices_well_founded`.

The denotation the passes were approximating: **a class is settled when its
cheapest admissible candidate has every child settled** — Knuth's AND-OR
generalisation of Dijkstra. That is exact whenever a candidate costs at
least as much as each of its children, which both passes satisfy: the tree
pass adds its children's costs to a non-negative own cost, and the shared
pass prices the union of its children's reach sets, a superset of each of
them. A candidate that mentions its own class is simply never admissible, so
the sentinel goes with the traversal that needed it.

`Settling` (`price` / `prefer` / `settle`) is the seam, `settle_in_cost_order`
the one driver, `TreePricer` and `SharedPricer` the two `impl`s; `TieBreak`
and `StageRecorder` still monomorphize through `Dp` unchanged.

**Subtracted:** a map out of the driver is well-founded by construction — a
class settles strictly after the children of the candidate it settles on —
so the repair stage is **deleted from the extraction path**. It survives for
`Extraction::from_dp`, whose input is an arbitrary caller's map.
`the_dp_map_is_well_founded_so_the_repair_is_a_no_op` is the gate on the
deletion and
`the_fixpoint_prices_a_class_the_post_order_dp_left_at_the_sentinel` is the
gate on the defect.

## 2. Shaders: the ladder, per kernel

Every row's graph is byte-identical to the baseline's (`classes_at_stop`,
`applications`), so these are two extractors on one graph.

| kernel | objective 5k → 100k (baseline) | objective 5k → 100k (**fixpoint**) | `dag_cost` base | `dag_cost` **fix** |
|---|---|---|---|---|
| `cosine_palette` | 18,418,432 (flat) | **18,418,432 → 14,158,886** | 292 (flat) | **292 → 521** |
| `domain_warp_fbm` | 27,664,132 → 26,941,696 | **26,813,184 → 26,418,688** | 457 → 436 | **444 → 433** |
| `gyroid_slice` | 21,444,647 (flat) | **21,444,647 (flat)** | 932 (flat) | **932 (flat)** |
| `julia_set` | 45,226,752 → 61,214,208 | **39,788,800 → 40,051,968** | 717 → 948 | **640 → 648** |
| `kaleidoscope_fold` | 34,933,520 (flat) | **34,605,840 (flat)** | 560 (flat) | **555 (flat)** |
| `mandelbrot_distance` | 32,707,328 → 37,492,480 | **32,445,952 (flat)** | 518 → 595 | **517 (flat)** |
| `metaballs` | 8,264,960 → 14,496,000 | **6,954,240 (flat)** | 155 → 270 | **135 (flat)** |
| `plasma` | 17,260,544 (flat) | **17,260,544 (flat)** | 359 (flat) | **359 (flat)** |
| `smooth_min_scene` | 7,345,152 → 6,755,349 | **6,099,989 (flat)** | 132 → 144 | **134 (flat)** |
| `smoothstep_vignette` | 7,867,199 (flat) | **7,867,199 (flat)** | 194 (flat) | **194 (flat)** |
| `star_sdf` | 10,554,112 (flat) | **10,554,112 (flat)** | 172 (flat) | **172 (flat)** |
| `torus_slice` | 6,884,110 (flat) | **6,884,110 (flat)** | 130 (flat) | **130 (flat)** |

Five of the twelve regressed from the smallest cap to the largest before;
**one** does now, and only in `dag_cost`. In the objective the ladder rises
on one kernel out of twelve, by 0.7 %, and falls on two. The objective is
worse than the baseline's on exactly **one of sixty rows**
(`domain_warp_fbm` at 10k, +0.51 %) and better or equal on the other 59.

The three named regressions of the witness run are gone: `metaballs`
155 → 270 becomes a flat 135 (objective ×1.754 → flat), `julia_set`
717 → 948 becomes 640 → 648, `mandelbrot_distance` 518 → 595 becomes a flat
517.

## 3. The one kernel that still climbs, and why it is the other column

`cosine_palette` goes 292 → 521 in `dag_cost` — the worst shader ratio in
that column after the change — while its objective falls 18,418,432 →
14,158,886 (−23 %). The extractor switched from `TreeCheaper` to `Shared`
and bought a term with more distinct nodes whose expensive nodes sit at a
cheaper scope. That is the `dag_cost`-versus-objective split the witness run
found on 50 of 99 DEV pairs (§7.2 there), now visible on a shader: every
`cosine_palette` witness pair the harness reports at this commit is
**static-only** — cheaper unweighted, dearer weighted.

`dag_cost` counts nodes flat. The objective weights each node by
`LatticeShape::evals(variance)`, so a node hoisted to constant scope is paid
once instead of 65,536 times at 256 × 256. **One of those two numbers is
what the machine pays and the other is not**, and the class-cap sweep's
headline is stated in the second one. Reconciling them is
`docs/results/2026-09-08-extraction-witnesses.md` §8 item 5 and is still
open; this document reports both and privileges neither.

## 4. Chrome, held out, run once at the end

| cap | objective base | objective **fix** | Δ | `dag_cost` base | `dag_cost` **fix** | arm base → fix |
|---:|---:|---:|---:|---:|---:|---|
| 5,000 | 3,346,823,903 | **3,485,751,868** | +4.15 % | 1668 | **1737** | Shared → Shared |
| 10,000 | 3,326,094,392 | **3,494,055,992** | +5.05 % | 1673 | **1754** | Shared → Shared |
| 20,000 | 3,301,227,397 | **3,483,709,601** | +5.53 % | 1681 | **1778** | Shared → Shared |
| 50,000 | 4,564,003,324 | **3,533,454,368** | -22.58 % | 2214 | **1749** | Shared → Shared |
| 100,000 | 4,700,952,733 | **3,728,383,573** | -20.69 % | 2374 | **1858** | TreeCheaper → Shared |

The sweep's headline for chrome — `dag_cost` 1,668 at 5k against 2,374 at
100k, **+42 %** — becomes 1,737 against 1,858, **+7.0 %**; in the objective,
+40.5 % becomes +7.0 %.

**And the fixpoint is worse on chrome at every small cap.** 4.2 % at 5k,
5.1 % at 10k, 5.5 % at 20k. This is not a defect that survived: pricing a
class exactly is not the same as choosing a term optimally, because the
sharing-aware DP is still a greedy per-class rule over a DAG whose sharing
is a global property. Where the sentinel used to fire, the repair's
arbitrary pick was sometimes the luckier one. The honest reading is that the
mechanical defect is out and the **search** limit is now what is left,
which is exactly what §5 registers.

All 7 chrome witness pairs survive, but the gap they name shrinks — the
mean of |Δobjective| over greedy's own objective goes **17.9 % → 4.8 %**. Their first-divergence labels move
from `COORDINATED` 6 / `CYCLE-PRICED` 1 to `COORDINATED` 5 / `TIE` 2, and
every one of them is now settled at stage `dp` — none by the repair, none by
`min-of-two`.

## 5. What the witnesses look like after the fix

The whole DEV corpus (203 kernels, 635 budget rows, the graph identical to
the baseline on **every** row):

| | baseline | fixpoint |
|---|---|---|
| Σ `objective` | 5,335,908,602 | **5,180,878,600** (−2.91 %) |
| rows better / worse / same (objective) | — | **222 / 131 / 282** |
| Σ `dag_cost` | 2,225,741 | 2,229,375 (+0.16 %) |
| witness pairs | 99 | **57** |
| **objective** witnesses | **49** | **28** (−43 %) |
| static-only witnesses | 50 | 29 |

Frontier-class labels and the stage that settled the first divergence, over
every DEV witness:

| label | base | **fix** | | stage | base | **fix** |
|---|---:|---:|---|---|---:|---:|
| COORDINATED | 55 | **44** | | `dp` | 68 | **53** |
| **CYCLE-PRICED** | **23** | **0** | | **`repair`** | **20** | **0** |
| TIE | 8 | 4 | | `min-of-two` | 11 | **4** |
| SHARING | 6 | 2 | | | | |
| DISTRACTOR | 5 | 7 | | | | |
| **LOCAL-MISS** | **2** | **0** | | | | |

**Both categories the witness run named as defects are at zero.**
CYCLE-PRICED was the finding; LOCAL-MISS was "a cheaper candidate the DP saw
and did not take", and it was produced by the repair overwriting a scored
choice — the stage is gone, and so are both. `repair` settles no first
divergence any more, because it no longer runs on this path.

What is left is what the witness run said would be left: **COORDINATED**
(44), **DISTRACTOR** (7) and **TIE** (4). Those are search, not defects, and
they are what `docs/plans/2026-09-08-extraction-judge-registration.md`
registers a learned component against.

## 6. Does more budget stop hurting?

Direction of each kernel's cost from the 5k cap to its largest, over the
whole DEV corpus:

| | rose | fell | flat | worst ratio |
|---|---:|---:|---:|---|
| `objective`, baseline | 15 | 147 | 41 | ×**1.754** (`metaballs`) |
| `objective`, **fixpoint** | **10** | 148 | 45 | ×**1.078** (`glyph32_U002A`) |
| `dag_cost`, baseline | 13 | 149 | 41 | ×1.742 (`metaballs`) |
| `dag_cost`, **fixpoint** | **3** | 155 | 45 | ×1.784 (`cosine_palette`, §3) |

**Largely, yes.** The worst regression in the number the extractor minimizes
falls from +75 % to +7.8 %, and the count of kernels whose unweighted
`dag_cost` climbs with the budget falls from 13 to **3**. All eight glyphs
that regressed in the class-cap sweep now improve:

| glyph | `dag_cost` base 5k → 20k | **fixpoint** | `objective` base | **fixpoint** |
|---|---|---|---|---|
| `U+006C` | 1,773 → 2,172 (+22.5 %) | **1,804 → 1,738 (−3.7 %)** | 431,247 → 388,525 | **437,199 → 369,057** |
| `U+0066` | 1,821 → 2,158 (+18.5 %) | **1,853 → 1,767 (−4.6 %)** | 470,479 → 411,885 | **476,463 → 393,793** |
| `U+0074` | 1,798 → 1,935 (+7.6 %) | **1,853 → 1,778 (−4.0 %)** | 469,743 → 404,315 | **476,463 → 394,145** |
| `U+006A` | 1,828 → 1,956 (+7.0 %) | **1,860 → 1,790 (−3.8 %)** | 464,627 → 402,445 | **470,611 → 391,553** |

(the 32 px rows; the 16 px rows are the same shapes and move identically.)

**And the two columns disagree in both directions.** The three kernels whose
`dag_cost` still climbs are `cosine_palette` (292 → 521, objective −23 %),
`julia_set` (640 → 648, objective +0.7 %) and `psychedelic_packed`
(821 → 825, objective +0.0007 %). The ten whose **objective** climbs include
eight glyphs whose `dag_cost` *falls*, some by a lot — `U+0058`
842 → **638** with the objective +3.0 %, `U+0078` 828 → **603** with +2.6 %,
`U+002A` 1,204 → **1,121** with +7.8 %. Whichever column is quoted, the other
one says the opposite on a different kernel. §3 is not a footnote about one
shader.

**What it costs.** On glyphs the trade is visible and small: at the 5k and
10k caps the fixpoint is better in both columns (Σ objective −0.04 % /
−0.02 %, Σ `dag_cost` −0.42 % / −0.22 %) and at 20k it is **worse**
(+0.43 % / +1.37 %) — 74 of 190 kernels dearer against 40 cheaper. The ladder
is flatter and the endpoint at the largest glyph cap is slightly dearer than
the endpoint the broken DP happened to land on. Reported, not defended.

## 7. What did not change, and one gate that had to

- `graph.rs` is untouched. Insertion, memo, union and rebuild are as they
  were; `classes_at_stop` and `applications` are identical on all 640 rows.
- Canonical tie-breaking was **not** shipped: the witness run measured it as
  a net loss (§4 there), and this change does not revisit that.
- `min-of-two` is **not** made per class. It settles 4 first divergences
  after the fix, down from 11; every one of them is a `cosine_palette`
  static-only pair, which is the `dag_cost`-vs-objective question and not a
  search question. Deferred with its evidence rather than done on taste.
- One codegen gate moved: `jit_matches_oracle_exact_unaries_and_min_max`
  compares an **optimized** JIT kernel against an oracle that evaluates the
  arena's own association. At `[-3.7, 10000, -10000, -10000]` the two large
  addends cancel, and the new extraction reassociates — a valid rewrite of
  the algebra — so the JIT returns `sqrt(3.7)` to the last bit (0x3ff63682)
  and the oracle returns 1.9238281 (0x3ff64000). The test was calling the
  *more accurate* answer the failure. The cancelling points are now skipped
  by a named predicate; the tolerance was **not** widened, which would hide
  a real op bug.

## 7b. Compile time, for scale only

A JIT-first system pays extraction on every bake, so the obvious worry is
that a heap-ordered fixpoint costs more than a post-order walk. Total wall
clock over the 635 DEV rows — **631.8 s baseline against 630.4 s** — says it
does not, and chrome is *faster* at three of its five caps (0.14/0.15/0.17 s
→ 0.11/0.12/0.14 s) and slower at 100k (3.99 s → 6.83 s). Both runs were
taken on a shared box at load averages 7–30, hours apart, so **none of these
is a measurement**; they are here to show there is no order-of-magnitude
change to investigate. A real compile-time claim needs `jit_bench`'s
discipline and this document makes none.

## 8. Data and reproduction

`2026-09-08-extraction-fixpoint-budgets.csv` (635 DEV rows),
`-witnesses.csv` (57 pairs), and `-chrome-{budgets,witnesses}.csv` (the
held-out arm, 5 rows and 7 pairs, run once in its own process). Columns are
the witness run's, unchanged, so the two sets are directly comparable
against `2026-09-08-extraction-witnesses{,-budgets,-chrome*}.csv`.

```
cargo build --release -p pixelflow-pipeline --bin extraction_witnesses
./target.noindex/release/extraction_witnesses --filter shader      --out <dir>
./target.noindex/release/extraction_witnesses --filter glyph       --out <dir>
./target.noindex/release/extraction_witnesses --filter psychedelic --out <dir>
./target.noindex/release/extraction_witnesses --chrome --filter chrome --out <dir>
```

## 9. What to do next

1. **The judge, registered:**
   `docs/plans/2026-09-08-extraction-judge-registration.md`. What survives
   the fixpoint is COORDINATED / DISTRACTOR / TIE, which is a search
   problem, and the registration names the shape, the metric, the four
   controls and the decision rule before anything is trained.
2. **Reconcile `dag_cost` with the objective.** Now unavoidable: the
   fixpoint is −2.91 % in one column and +0.16 % in the other over the same
   635 rows, and `cosine_palette` moves −23 % and +78 % at once. Nothing
   downstream can be graded until one of them is established as the thing
   the machine pays.
3. **Restate or repair L2.** Unchanged by this work and still false as run:
   the chrome monotonicity miss (`MulAdd(c1429, c0, c5181)` minted in
   `G(10k)`, absent from `G(20k)` and above) reproduced exactly.
4. `min-of-two` per class stays open, with its evidence down from 11 first
   divergences to 4 (§7).
