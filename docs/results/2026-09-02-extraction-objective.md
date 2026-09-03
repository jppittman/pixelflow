# Making extraction price sharing (#1116)

`extract_dag` summed each child's `best_cost` — a TREE cost — so a subterm used ten times was charged ten times in the objective and emitted once in the kernel. This is the measurement of replacing that objective with the DAG cost the kernel actually pays, scored against the same two exact references #1115 built (`2026-09-02-extraction-gap.md`): Knuth's exact tree optimum, and the branch-and-bound DAG optimum where it closes.

Cost table: the latency prior as re-measured on current main (#1134 — Sin 70->95, Cos 75->103, Tan 87->117, Sqrt 15->13, Select 4->3). Every number below is deterministic: `CostModel::latency_prior` through `ExtractedDAG::dag_cost`.

## Headline

- **Gap closed: 95.9%** of the measured gap — 187 of 195 pooled cost units — on the 3 kernel(s) where the branch and bound proved an optimum *and* the old extractor missed it. That population is small by construction: the B&B stops closing at ~100-200 reachable classes, and greedy was already exactly optimal on most of what it does close.
- Exactly DAG-optimal on the 93 solved kernels: **90 -> 92**.
- Real kernels (206): **55 improved, 151 unchanged, 0 worse**. Pooled `dag_cost` **-0.23%**, median +0.00%, best -10.43%, 2303 cost units saved.

**Zero regressions is structural, not lucky.** `extract_dag_scoped` runs both objectives and returns the cheaper term by true `dag_cost`, ties going to the tree arm, so the returned cost is a minimum over a set that contains the old answer. The probe asserts it rather than reporting it.

**The gap denominator is this run's own, not #1115's.** #1115 pooled 195 units over the 89 kernels its branch and bound closed, under the pre-#1134 cost table. This run closes 93 kernels under the refreshed table and pools 195 units over them. Where the two numbers agree that is arithmetic coincidence, not the same set re-measured, and the fraction above is computed entirely within this run.

### The gap kernels, one row each

| kernel | reachable classes | old (tree) | new | exact DAG optimum | closed |
|---|---:|---:|---:|---:|---:|
| shader:smooth_min_scene | 159 | 132 | 130 | 122 | 20% |
| synth_d4_s7 | 53 | 638 | 633 | 633 | 100% |
| synth_d11_s4 | 89 | 1848 | 1668 | 1668 | 100% |

## Real kernels, by group

| group | n | improved | unchanged | worse | median Δ dag_cost | best Δ | units saved |
|---|---:|---:|---:|---:|---:|---:|---:|
| cellgrid | 3 | 3 | 0 | 0 | -0.23% | -0.24% | 3 |
| glyph | 190 | 48 | 142 | 0 | +0.00% | -1.97% | 2232 |
| psychedelic | 1 | 0 | 1 | 0 | +0.00% | +0.00% | 0 |
| shader | 12 | 4 | 8 | 0 | +0.00% | -10.43% | 68 |
| **all real** | **206** | **55** | **151** | **0** | **+0.00%** | **-10.43%** | **2303** |

## Which arm wins, and how often

Neither DP is optimal — both choose greedily bottom-up — so the sharing arm is not uniformly better, and reporting only the min would hide that. Over the 206 real kernels: the sharing arm is strictly cheaper on **55**, ties on **61**, and is strictly *dearer* on **90**.

The 90 are the honest cost of a bottom-up chooser: a class that is DAG-cheap in isolation need not compose into a DAG-cheap parent. Running both arms and taking the min is what turns that from a regression into a no-op, and is why the second pass buys robustness as well as cost.

The worst of them, so the shape of the failure is on the record rather than averaged away:

| kernel | group | tree dag | sharing dag | Δ if taken alone |
|---|---|---:|---:|---:|
| psychedelic | psychedelic | 841 | 1388 | +65.04% |
| shader:julia_set | shader | 706 | 848 | +20.11% |
| glyph16:U+0059 | glyph | 422 | 488 | +15.64% |
| glyph32:U+0059 | glyph | 422 | 488 | +15.64% |
| glyph16:U+006C | glyph | 507 | 585 | +15.38% |
| glyph32:U+006C | glyph | 507 | 585 | +15.38% |
| glyph16:U+003E | glyph | 503 | 575 | +14.31% |
| glyph32:U+003E | glyph | 503 | 575 | +14.31% |

## Where the objective change bites

| kernel | group | reachable classes | tree dag | sharing dag | Δ | tree TREE cost | sharing TREE cost |
|---|---|---:|---:|---:|---:|---:|---:|
| shader:mandelbrot_distance | shader | 415 | 556 | 498 | -10.43% | 843379 | 844891 |
| shader:torus_slice | shader | 165 | 133 | 128 | -3.76% | 285 | 301 |
| glyph16:U+004A | glyph | 1868 | 4203 | 4120 | -1.97% | 28258 | 43335 |
| glyph32:U+004A | glyph | 1868 | 4203 | 4120 | -1.97% | 28258 | 43335 |
| shader:metaballs | shader | 286 | 157 | 154 | -1.91% | 414 | 406 |
| glyph16:U+0055 | glyph | 2079 | 5007 | 4920 | -1.74% | 33290 | 51364 |
| glyph32:U+0055 | glyph | 2079 | 5007 | 4920 | -1.74% | 33290 | 51364 |
| shader:smooth_min_scene | shader | 159 | 132 | 130 | -1.52% | 426 | 444 |
| glyph16:U+003B | glyph | 2307 | 5889 | 5800 | -1.51% | 41440 | 63568 |
| glyph32:U+003B | glyph | 2307 | 5889 | 5800 | -1.51% | 41440 | 63568 |
| glyph16:U+007E | glyph | 2812 | 9083 | 8998 | -0.94% | 65732 | 96086 |
| glyph32:U+007E | glyph | 2812 | 9083 | 8998 | -0.94% | 65732 | 96086 |
| glyph16:U+003A | glyph | 2312 | 6031 | 5976 | -0.91% | 42492 | 66014 |
| glyph32:U+003A | glyph | 2312 | 6031 | 5976 | -0.91% | 42492 | 66014 |
| glyph16:U+0071 | glyph | 2802 | 8892 | 8811 | -0.91% | 59409 | 89986 |
| glyph32:U+0071 | glyph | 2802 | 8892 | 8811 | -0.91% | 59409 | 89986 |
| glyph16:U+0043 | glyph | 2830 | 9166 | 9084 | -0.89% | 65251 | 94613 |
| glyph32:U+0043 | glyph | 2830 | 9166 | 9084 | -0.89% | 65251 | 94613 |
| glyph16:U+0062 | glyph | 2824 | 9140 | 9059 | -0.89% | 64036 | 92971 |
| glyph32:U+0062 | glyph | 2824 | 9140 | 9059 | -0.89% | 64036 | 92971 |
| glyph16:U+0030 | glyph | 2903 | 9658 | 9595 | -0.65% | 71063 | 97572 |
| glyph32:U+0030 | glyph | 2903 | 9658 | 9595 | -0.65% | 71063 | 97572 |
| glyph16:U+0070 | glyph | 2901 | 9614 | 9553 | -0.63% | 72381 | 98404 |
| glyph32:U+0070 | glyph | 2901 | 9614 | 9553 | -0.63% | 72381 | 98404 |
| glyph16:U+006F | glyph | 2947 | 9944 | 9885 | -0.59% | 79507 | 105594 |

The two right-hand columns are the point: where the objective change bites, the TREE cost **rises**. That is not a regression — it is the old objective being abandoned, visible.

## Cost

Extraction wall time against the extractor it replaces, over the 206 real kernels: **median 2.23x, p90 2.29x, max 2.45x, pooled 2.23x**. Each kernel is the median of 9 alternating pairs timed back to back in one process on one saturated graph, so the absolute nanoseconds belong to the machine and only the ratio is claimed. Deterministically: two DP traversals where there was one.

Two DP passes over the e-graph instead of one, plus one bit per class per class of scratch for the sharing pass (~385 KB at the median production glyph's 1,755 reachable classes; ~12.5 MB at the 10,000-class production ceiling, allocated once per extraction and dropped at the end of it). Extraction runs **once** per compile against thousands of rule applications in saturation, which is why this is the cheap place to spend.

The sharing arm returned a different term on 227 of 302 kernels; on the rest the second pass is pure overhead and the tree arm's answer is returned unchanged.

## The gate

The bar this change was held to was: no real kernel regresses in `dag_cost`, and extraction runs in under 2x. The first is met and is structural. **The second is not** — 2.23x median. The second DP pass is not free and the compacted bitset only took it from 2.57x to 2.23x, because the sharing arm also pays a repair and a costing pass of its own. So this is not self-arming: the trade is 2303 cost units and 95.9% of the closable gap against 2.23x on a phase that runs once per compile, and that is a judgement call, not a threshold.

## What this obliges downstream

Every Guide label and every registered constant in the Phase-3 program was minted under tree cost. #1128's bisect already showed the consequence: guides trained on tree-cost labels steer toward UNSHARED terms (tree/dag sharing ratio 4.25 unguided vs 3.29 guided on `sh`) and lose to unguided on the real metric. With the objective changed, that chain restarts — re-mint, retrain, re-evaluate — and any registered constant carried across it is in stale units.

## Reproduction

```sh
PIXELFLOW_EXTRACTION_GAP_ARENA_DIR=<dir of .arena dumps> \
  PIXELFLOW_EXTRACTION_GAP_SECS=4 \
  PIXELFLOW_EXTRACTION_GAP_EXPANSIONS=15000000 \
  RUST_MIN_STACK=268435456 \
  cargo test -p pixelflow-search --release --lib -- --ignored extraction_objective_measurement
```

`2026-09-02-extraction-objective.csv` / `.json` carry every kernel's row.
