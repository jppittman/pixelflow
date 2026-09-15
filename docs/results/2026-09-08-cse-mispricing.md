# The extractor was under-pricing CSE by up to 100%, and that under-pricing *was* the class-cap regression

**The hypothesis (JP, 2026-09-08) was right.** The extraction DP's internal
cost for the choice map it selected did not equal the `dag_cost` of the term
that map materializes, and the error grew with graph size. On 96 (kernel, cap)
cells of the DEV corpus the old extractor's claim was exact on **36**; on the
chrome scene at a 50,000-class cap it claimed **281** for a term costing
**4,564,003,324**.

**The mechanism.** Both DP passes walked one DFS post-order. On a cyclic
e-graph — and commutativity alone makes every saturated e-graph cyclic — that
reaches some classes *before* the descendants closing the cycle. The class was
priced at the cycle sentinel, one of its nodes was recorded anyway, and
`shared_dag_dp_pass` gave it **a reach set of just itself**. Every ancestor
then paid that class's own cost and got its entire sub-DAG **free**. The
sentinel-priced classes grew **8 → 10 → 1,620** on chrome as the cap grew
5,000 → 20,000 → 50,000; the repair then rewrote 1,361 picks with no cost model
at all. An *under*-estimate concentrated exactly where the graph is biggest is
how more budget bought a worse term.

**The fix landed while this measurement ran**, from the extraction-witnesses
workflow: `96da8f5c fix(search): settle extraction in cost order, and delete
the repair stage` — Knuth's AND-OR generalisation of Dijkstra, so a class is
settled when its cheapest admissible candidate has every child settled, the
sentinel goes with the traversal that needed it, and the map is well-founded by
construction. **This CL does not re-fix it. It measures it, and turns the
property into a gate.**

**After the fix the claim is exact on 96 of 96 cells**, in every family, at
every cap — median and maximum signed error 0.00%. And the cap regression is
essentially gone: `chrome_R` +46.21% → **−0.77%**, `chrome_packed` +36.37% →
**+1.37%**, `shader:julia_set` +23.78% → **−2.63%**,
`shader:mandelbrot_distance` +14.63% → **0.00%**, `shader:metaballs` +15.32% →
**0.00%**.

**One thing to hand back.** At the shipped budget the new extractor's honest
objective picks a **dearer** term for chrome: `chrome_packed` +4.15%
(unweighted arena `dag_cost` 1,668 → 1,737) and `chrome_R` +5.22% (1,418 →
1,486). Those are the only two of 32 kernels that got dearer; shader is −3.95%,
psychedelic −1.10%, glyph16 −0.11%, cellgrid identical. Both extractors are
greedy — one settled after all DFS descendants, the other on the cheapest
candidate available at the moment — and neither is optimal, so a *correct*
objective does not by itself buy a better term. That is a search question, and
it is now cleanly separable from the arithmetic for the first time.

## The gate this CL adds

`Optimized::extraction` carries a `ClaimAudit`: the winning DP arm's own
minimized value at the root, and the `CostScale` (`Tree` or `Dag`) it is on —
because the two arms minimize *different quantities* and a claim compared
against the wrong column of `ChoiceCost` means nothing. `costed()` then
`debug_assert`s that the claim equals the price of the term returned.

**That makes every extraction any test in the workspace performs a
self-consistency check** — which is how the defect was caught here: the
assertion fired on `core-term`'s terminal scene, and no fixture
`pixelflow-search` owned reproduced it. `blind_dfs_egraph` is now that scene as
six e-classes, and `the_claim_is_exact_on_the_graph_a_dfs_order_could_not_price`
asserts its own premise (that a DFS order really is blind there) rather than
remembering it.

The instrument is `egraph_off_on consistency --out CSV --class-cap 5000 20000
50000`: saturate each kernel at each cap, then write `claimed` beside
`cost_of_choices` of the term returned — same cost model, same shape the
extraction ran at (`for_lattice`, not `POINT`). Nothing is timed; every column
is a deterministic function of the input.

Corpus: 32 DEV kernels × 3 caps = 96 cells — the 12 `shader_bench` ports, cell
grid 80×24 @2×, psychedelic, and 16 DejaVu glyphs at tile 16
(`U+0040`–`U+004F`) — plus chrome and chrome_R held out and reported
separately. Host load 7–31 throughout; it enters no number here.

## Check 1 — self-consistency

`error = claimed − price` on the winning arm's own scale.

| family | cells | exact before | exact after | worst cell before |
|---|---:|---:|---:|---|
| chrome (held out) | 3 | 2 | **3** | `@50000`: claimed **281** vs dag **4,564,003,324** (−100.0%) |
| chrome_channel (held out) | 3 | 2 | **3** | `@50000`: claimed **738** vs dag **4,126,473,723** (−100.0%) |
| shader | 36 | 25 | **36** | `metaballs@20000`: claimed **4,611,686,018,427,387,903** vs tree 17,204,736 |
| glyph16 | 48 | 1 | **48** | `U+004D@50000`: claimed **18,446,744,073,709,551,615** vs tree 65,744 |
| psychedelic | 3 | 3 | **3** | — |
| cellgrid | 3 | 3 | **3** | — |
| **total** | **96** | **36** | **96** | median \|err\| **3.84%** → **0.00%** |

Two directions, one shape:

- **Under-estimate (`shared` arm).** The sentinel-priced class's reach set was
  `{itself}`, so its sub-DAG was invisible. This is the one that changes
  choices, and it is the class-cap regression. Worst cases, with the
  sentinel/repair counts that drove them:

  | cell | claimed | dag_cost | live | sentinel | repaired |
  |---|---:|---:|---:|---:|---:|
  | `chrome_R@50000` | 738 | 4,126,473,723 | 10,205 | 1,863 | 1,177 |
  | `chrome_packed@50000` | 281 | 4,564,003,324 | 10,256 | 1,620 | 1,361 |
  | `shader:julia_set@50000` | 155,916 | 55,980,544 | 4,499 | 1,135 | 1,077 |
  | `shader:metaballs@5000` | 1,048,576 | 8,264,960 | 286 | 25 | 2 |
  | `glyph16_U0044@50000` | 121,631 | 183,528 | 8,316 | 1,409 | 1,185 |

- **Over-estimate (`tree` arm).** `tree_dp_pass` charged `CYCLE_COST` for an
  unsettled child, so its claim sat at `usize::MAX / 4` or, on any real graph,
  at `usize::MAX` outright — a tree cost is exponential in the sharing it
  refuses to price. Above that ceiling every candidate ties and the DP keeps
  whichever the tie-break prefers: **the tree arm's node choice is cost-blind
  on every glyph**, then and now (pinned:
  `the_tree_arms_objective_saturates_on_a_real_sized_graph`). It cannot corrupt
  the returned term, because the arms are chosen between by re-costed `dag` —
  see check 2.

The error grew with the graph, which is the hypothesis's own prediction:

| live classes | cells | exact | median \|error\| |
|---|---:|---:|---:|
| < 500 | 24 | 13 | 0.00% |
| 500 – 2,000 | 19 | 9 | 32.98% |
| 2,000 – 5,000 | 33 | 10 | 1.13% |
| ≥ 5,000 | 20 | 4 | 20.59% |

## Check 2 — arm comparability: **the arms were already compared like with like**

`extract_dag_scoped` never compares the two DPs' own tables. Both arms are
settled, re-costed by `cost_of_choices` under the *same* cost model and the
*same* shape, and the min is taken on `ChoiceCost::dag`. Now pinned
(`the_arms_are_compared_on_the_price_not_on_their_claims`): the returned term is
`min(tree, shared)` by true `dag_cost`, ties to tree, and the returned map is
that arm's.

So the `tree_cheaper` verdicts in the class-cap sweep were not a scale
confusion, and #1239's worry about "as many as three cost computations"
resolves to two plus a harness column — `ChoiceCost::dag` is weighted by
`LatticeShape::evals` of each chosen form's variance, the harness's `dag_cost`
column is the unweighted latency-prior sum over the materialized arena, and
they agree exactly at `POINT` (`dag_cost_equals_the_materialized_arenas_cost`).
Both are in the CSV (`dag_cost`, `arena_dag_unweighted`), so no reader has to
guess which is quoted.

What *was* wrong is that the scale lived in a comment. It is now in the type:
`CostScale::{Tree, Dag}` is carried on every claim, and `CostScale::of(cost)`
is the only way to read the column it names.

Objective flips over the 96 cells: `tree_cheaper → shared` **11**,
`shared → tree_cheaper` **6**, unchanged 79.

## Check 3 — reach sets

Extended rather than newly written, since #1229 already had a dense reference:

- `shared_pass_matches_the_dense_reference_on_a_saturated_graph` compared the
  sparse/dense hybrid's **choices**. It now also compares the **root cost** — a
  split that agreed on the map while disagreeing on its price was invisible to
  it.
- `the_shared_dp_prices_a_doubly_reached_class_exactly_once`: on `sin(X)·sin(X)`
  the pass's root cost is `own(Mul) + own(Sin) + own(X)` against a hand-computed
  sum — once, not twice, and not the zero times the sentinel produced — while
  `tree_dp_pass` on the same graph is that sum with `sin(X)` charged twice.
- `the_budget_fallback_reports_the_tree_scale_it_actually_used`: over
  `SHARED_DAG_PASS_BYTE_BUDGET` the pass returns no map and the caller reports
  `TreeOnly` **on the `Tree` scale** — never a shared-priced answer under a tree
  label.

## The cap reading after the fix

`dag_cost` at a 50,000-class cap relative to the same kernel at 5,000. Negative
means more budget buys a better term. Cells that move by less than 0.01% in
both columns are omitted.

| kernel | before | after |
|---|---:|---:|
| `chrome_R` (held out) | **+46.21%** | **−0.77%** |
| `chrome_packed` (held out) | **+36.37%** | **+1.37%** |
| `shader:julia_set` | **+23.78%** | **−2.63%** |
| `shader:metaballs` | **+15.32%** | **0.00%** |
| `shader:mandelbrot_distance` | **+14.63%** | **0.00%** |
| `shader:cosine_palette` | 0.00% | **−23.13%** |
| `shader:smooth_min_scene` | −8.03% | 0.00% |
| `shader:domain_warp_fbm` | −2.85% | −1.47% |
| `glyph16_U0044`, `U+004A` | −5.4%, −5.8% | **−14.2%, −14.3%** |
| `glyph16` (other 12 sampled) | −0.13% … −81.61% | −0.13% … −81.64% |
| `cellgrid` | −0.26% | −0.26% |

**Extraction no longer gets worse with budget**, with `chrome_packed`'s residual
+1.37% the only cell above zero — down from +36.37%, and now within the range
that greedy settling order alone can explain.

## What changed in production, and the one thing to hand back

At the shipped classical budget (`CLASSICAL_CLASS_CEILING` is pinned at the
5,000 floor, so the cap-5,000 rows are what ships):

| family | kernels | terms changed | Σ `dag_cost` | Σ arena `dag_cost` (unweighted) |
|---|---:|---:|---:|---:|
| shader | 12 | 6 | **−3.95%** | −2.47% |
| psychedelic | 1 | 1 | **−1.10%** | −0.48% |
| glyph16 (16 sampled) | 16 | 6 | −0.11% | −0.61% |
| cellgrid | 1 | 0 | 0.00% | 0.00% |
| **chrome** (held out) | 1 | 1 | **+4.15%** | 1,668 → **1,737** |
| **chrome_channel** (held out) | 1 | 1 | **+5.22%** | 1,418 → **1,486** |

Chrome is the only kernel that got dearer, and it did so at the shipped budget.
An honest objective does not by itself buy a better term: both extractors are
greedy — the old one settled a class after all its DFS descendants, the new one
settles on the cheapest candidate admissible at that moment — and on chrome the
second greedy rule lands 4–5% worse by the metric that is now trustworthy. This
is a **search** finding, not a costing one, and it belongs with the witnesses
run and #1238's `Beam`: with the arithmetic fixed, a beam win on chrome is now
readable as a search win rather than as noise around a broken objective.

Full workspace suite green with the claim/price `debug_assert` armed (127 test
binaries, 0 failed), `freetype_oracle.rs`'s optimized-arm `'8'`-waist check
included.

## Files

- `docs/results/2026-09-08-cse-mispricing.csv` — 192 rows, `extractor` ∈
  {`dfs-post-order`, `cost-order-settling`} × 96 (kernel, cap) cells.
  `sentinel_classes` and `repaired_picks` are populated only for
  `dfs-post-order`: neither quantity exists any more, which is the point.
- `pixelflow-search/src/egraph/extract.rs` — `ClaimAudit`, `CostScale`, the
  `debug_assert`, `blind_dfs_egraph`, and the five checks.
- `pixelflow-pipeline/src/bin/egraph_off_on.rs` — the `consistency` subcommand.
