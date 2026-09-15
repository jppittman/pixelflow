# Extraction witnesses: where the extractor walks past a term it provably holds

**Date:** 2026-09-08
**Status:** DENOTED — written before the instrument. Results land in
`docs/results/2026-09-08-extraction-witnesses.{md,csv,json}`; nothing below
is revised after the first run except to append what it found.
**Authority:** JP, 2026-09-08 — *"we're back to the extraction judge that we
threw away earlier. Let's pivot our research in that direction. We have some
kernels with known good extractions that we can't access yeah?"*
**Constraint:** `graph.rs`'s insertion, memo, union and rebuild code is under
JP's own reading and is not touched. Everything here is `extract.rs` and
harnesses.

## 0. Why this, now

Two measurements this week say the **extractor**, not the graph and not the
cost model, is the bottleneck:

1. The class-cap sweep (`docs/results/2026-09-08-class-cap-sweep.md`, #1229):
   raising the class cap makes extraction *worse in the extractor's own
   objective* on every family — chrome 1,668 at the 5k cap vs 2,374 at 100k
   (+42 %), `julia_set` 717 → 948, `metaballs` 155 → 270,
   `mandelbrot_distance` 518 → 595 with the 20k+ graph **quiesced** at 237
   live classes.
2. The rules × nodes filter (#1228/#1231): a uniform-random filter at matched
   keep-rate equals a trained bilinear one; the whole `dag_cost` effect is
   *fewer applications*, and `Identity` reaches its own full-budget cost at
   `B/8` on 40 of 94 glyphs.

The e-graph is monotone (L2, `docs/plans/2026-09-02-optimizer-api.md`): the
run at a larger budget performs every application the smaller run performed
and then more, so the larger graph represents a superset of the terms. A
cheaper term found at the smaller budget is therefore **in** the larger
graph, and the extractor walked past it. Those terms are the witnesses.

`docs/results/2026-09-02-extraction-gap.md` already named the suspect — *"(i)
single DFS, not a fixpoint — a class whose child is `on_stack` is scored
`CYCLE_COST` and never revisited … Fix (i) first"* — and it was never fixed:
`tree_dp_pass` and `shared_dag_dp_pass` still run one `post_order` DFS and
price any node whose child is still on the DFS stack at `CYCLE_COST`. That
document measured the gap against a Knuth reference; this one goes the other
way round — it takes the extractor's *own* cheaper outputs at smaller budgets,
maps them into the bigger graph, and asks at which class, and by which stage,
the extractor let go of them.

## 1. Denotation

### 1.1 Objects

- **K** — a DEV kernel: its legalized arena (`LowerDwrt`, `ExpandReduce`) and
  its lattice shape `S` (the production `for_lattice(shape)`).
- **G(b)** — the e-graph after `Optimizer::production().for_lattice(S)` under
  `Budget::Explicit { iterations: tier, classes: b, applications: 40·b }` —
  exactly the sweep's `cap{b}-app{40b}` arm — a deterministic function of
  `(K, b)`. `G(B/8)` and `G(B)` name the filter registration's pair:
  `Budget::Applications(n)` with `B` the applications production fires on
  `K`.
- **Term** — a well-founded choice map `C : classes(G) ⇀ node index`, total
  on the classes reachable from the root under itself. `choices_to_arena`
  materializes it; `cost_of_choices` prices it.
- **Two costs of a term**, both reported, never confused:
  - `dag_cost` — the sweep's column: the latency prior summed over the
    reachable nodes of the materialized arena, each once, **unweighted**.
  - `objective` — `ChoiceCost::dag` at `S`: the same sum with each node
    weighted by `S.evals(variance)`. This is the number the extractor
    minimizes; a per-row subterm is priced once per row, a constant once.
  A term that is cheaper in `dag_cost` but not in `objective` indicts the
  objective's weighting, not the extractor's search, and is tabulated apart.
- **greedy(b)** = `extract_dag_scoped(G(b), root, latency_prior, S)` — what
  production returns: tree DP → sharing-aware DP → cheaper of the two by
  `objective` (ties to tree) → `repair_choices_well_founded`. There is **no
  swap refinement on the production path**: `IncrementalExtractor` runs only
  under a `Reranker`, and none ships. The stages that can make a choice are
  therefore four — tree pass, shared pass, min-of-two, repair.

### 1.2 Witness

For `b_lo < b_hi`, the term `T = greedy(b_lo)` is a **witness against
`greedy(b_hi)`** iff `objective(T) < objective(greedy(b_hi))`. (Where only
`dag_cost(T) < dag_cost(greedy(b_hi))` holds it is a *static-only witness*,
counted separately.)

**Monotonicity, made checkable.** `G(b_hi)`'s run is a prefix-extension of
`G(b_lo)`'s — same rules, same order, same graph until `b_lo`'s cap binds —
so every e-node and every union of `G(b_lo)` is in `G(b_hi)`. Hence every
subterm of `T` has an e-class in `G(b_hi)`. The instrument does not assume
this: it **looks each subterm up** and fails loudly on a miss.

**The induced choice map `C_T`.** Walk `T`'s arena in post-order. A leaf maps
to the class holding `ENode::Var/Const/Buffer/Uniform`; an op node maps to
the class holding `ENode::Op { kind, [class(child_i)] }` with children
canonicalized by `find`. The lookup is a table built read-only from
`class_ids()` × `nodes()` with children canonicalized — the memo's content,
without touching the memo. `C_T(class) = ` the index of that node in
`nodes(class)`. Two of `T`'s arena nodes may land in **one** class of
`G(b_hi)` (equal there, distinct at `b_lo`): then `C_T` keeps the first in
post-order, and the map names a term at least as cheap as `T` (one class
materializes once). The count of such merges is recorded.

### 1.3 Divergence

`C_G` is `greedy(b_hi)`'s choice map. The **divergence set** is

```text
D = { c reachable from root under C_T : C_T(c) ≠ C_G(c) }
```

(`C_G` is total on root-reachable classes, so the comparison is always
defined.) The **frontier** `F ⊆ D` is the divergent classes none of whose
`C_T`-descendants diverge: at a frontier class both choosers agree on
everything below, so the extractor's local comparison there is directly
examinable. The **first divergence** is the first member of `F` in `T`'s
post-order — the deepest place the extractor left the witness.

### 1.4 A class-level explanation

For each frontier class `c` with `w = C_T(c)` (the witness's node) and
`g = C_G(c)` (greedy's):

| field | meaning |
|---|---|
| `own_w`, `own_g` | each node's weighted own cost at `S` |
| `tree_w`, `tree_g` | the cost the **tree pass** assigned each candidate — `CYCLE_COST` when a child was on the DFS stack |
| `shared_w`, `shared_g` | the cost the **shared pass** assigned each candidate |
| `stage` | which stage produced `g`: `tree` / `shared` (the winning arm's raw DP chose it), `min-of-two` (the *losing* arm had `w`), `repair` (the raw DP had something else and repair rewrote it) |
| `rule_g` | the rule that minted `g` (`tags` → `Provenance::origin` → `ApplicationRecord::rule`), or `seed` |
| `swap_delta` | `objective(C_G[c ↦ w]) − objective(C_G)`: what one swap at `c` is worth from greedy's term, if well-founded |

### 1.5 Classification (one label per frontier class)

| label | test | what it says about the extractor |
|---|---|---|
| **CYCLE-PRICED** | `tree_w == CYCLE_COST` (or `shared_w`) while `w` is well-founded in the witness | the DP never priced `w`: its child was on the DFS stack. A single-DFS artifact — extraction-gap's defect (i) |
| **TIE** | `tree_w == tree_g` (finite) | equal local cost; `<` kept the earlier index — insertion order decided |
| **LOCAL-MISS** | `tree_w < tree_g` (finite) | the DP saw a cheaper candidate and did not take it — a bug, if it exists |
| **SHARING** | `tree_w > tree_g` but `swap_delta < 0` | tree-locally `g` is cheaper, DAG-globally `w` is: sharing elsewhere makes `w`'s subterm free. Says why the shared pass missed it: it priced `w` at `CYCLE_COST` too, or it chose `w` and lost min-of-two, or its reach-set greedy choice below differed |
| **DISTRACTOR** | SHARING and `rule_g ≠ seed` | the rule-minted `g` has lower local cost and is dearer in the DAG — the rule is named |
| **COORDINATED** | `swap_delta ≥ 0` yet `objective(C_T) < objective(C_G)` | no single swap from greedy's term reaches the witness; it needs `k > 1` simultaneous changes — a local search cannot see it |

A frontier class gets the first label that matches, in the order listed.
`REALIZABLE-1` / `REALIZABLE-k` / `COORDINATED` is recorded per witness as
well: does one accepted frontier swap already beat greedy; does greedily
accepting improving frontier swaps reach `objective(C_T)`; or neither.

### 1.6 The cheap fix, A/B'd first

Ties break to insertion order (`if this_node_cost < min_cost`), and the
`'8'` bisect showed extraction moving under semantically-null input changes.
Before anything else, the same DP with a **canonical tie-break** — on equal
tree cost prefer lower own cost, then fewer children, then the smaller
`OpKind` ordinal, then smaller canonical child ids — is run on DEV at 5k and
50k and diffed against production. If it recovers witnesses, that is a
mechanical fix and goes first. The count of classes where a tie occurred at
all is reported beside it: if ties are rare, tie-breaking cannot be the
mechanism.

The second variant is the one extraction-gap asked for: the tree DP as a
**fixpoint** (Knuth's AND-OR Dijkstra — a class settles when its cheapest
candidate has every child settled, whatever the DFS order), and the same
with canonical ties. Both are measured, not shipped: the results document
says what each recovers and what it costs, and the fix is a separate change.

## 2. The instrument

`pixelflow-pipeline/src/bin/extraction_witnesses.rs`, built with the crate's
`provenance-journal` feature (already unconditional there). Per DEV kernel:

1. Saturate at each budget of the ladder in one process, keeping every
   `(G(b), Optimized)`: `{5k, 10k, 20k}` for glyphs (`50k` for the named
   glyphs), `{5k, 10k, 20k, 50k, 100k}` for shaders and the scene kernels,
   plus `Applications(B/8)` and `Applications(B)`.
2. For every pair `(b_lo, b_hi)` with `objective(greedy(b_lo)) <
   objective(greedy(b_hi))` (and separately for `dag_cost`): map, diff,
   classify, record.
3. Run the DP variants of §1.6 on every `G(b)` and record their costs beside
   production's.

The only addition to `pixelflow-search` is a feature-gated research module,
`egraph::extract::witness` (`#[cfg(feature = "provenance-journal")]`, the
feature every research harness already builds with and
`scripts/check-provenance-journal-scope.sh` keeps out of downstream builds),
which returns the two raw DP choice maps, their per-candidate cost tables
and the repaired maps as a `StageTrace`, and runs the §1.6 variants. The two
production passes gain a monomorphized `StageRecorder` parameter whose
production instance is `()` — no branch, no allocation, byte-identical
output. Nothing new is `pub` outside the feature.

## 3. What gets reported

- The witness table: kernel, family, `b_lo`, `b_hi`, `live_classes` at each,
  `objective` and `dag_cost` of witness and greedy, delta, `|D|`, `|F|`,
  merges, realizability.
- The classification histogram, by label and by family; the top-5 rules
  minting DISTRACTORs.
- The first-divergence explanations for `mandelbrot_distance`, the two worst
  glyphs by relative loss, `U+004B` (#1114's regression), and — once, at the
  end, labelled held-out — chrome.
- The §1.6 A/B: per budget and family, production vs canonical-ties vs
  fixpoint vs both, Σ objective and the count of kernels changed, and how
  many witnesses each variant recovers.
- The one-paragraph answer to "why does the extractor get worse with more
  budget", written from the histogram, not before it.
