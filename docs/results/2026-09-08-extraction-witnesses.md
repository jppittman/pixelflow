# Extraction witnesses: the extractor walks past terms it provably holds

**Date:** 2026-09-08
**Denotation:** `docs/plans/2026-09-08-extraction-witnesses.md` — written
before the instrument; nothing in it was revised after the run.
**Instrument:** `pixelflow-pipeline`'s `extraction_witnesses` bin over
`pixelflow-search`'s `egraph::witness` (feature-gated on
`provenance-journal`).
**Data:** `2026-09-08-extraction-witnesses.{csv,json}` — witness rows and
per-class explanations — and `2026-09-08-extraction-witnesses-budgets.csv` —
the budget ladder, including the tie-break A/B.

> **What is deterministic.** `dag_cost`, `objective`, `live_classes`,
> `tied_classes`, both choice maps and every classification are functions of
> the term and the graph, so every column below is exact and reproducible.
> Only `seconds` is wall clock, and it was taken on a shared box whose load
> average ran between 24 and 168 during the run — it is reported for scale,
> never as a claim.

## 0. The answer, first

**Why does the extractor get worse with more budget?** Because on the
kernels that regress, most of its decisions are not decisions. Over the 56
objective-witness pairs the run found, the frontier classes where greedy and
the witness disagree classify like this:

| label | count | share |
|---|---:|---:|
| COORDINATED — needs `k > 1` simultaneous changes | 191 | 57 % |
| **CYCLE-PRICED — the DP never priced either candidate** | **98** | **29 %** |
| DISTRACTOR — a rule-minted node, locally cheaper, dearer in the DAG | 28 | 8 % |
| TIE — equal DP cost, insertion order decided | 15 | 4 % |
| LOCAL-MISS — a cheaper candidate the DP saw and did not take | 2 | 0.6 % |
| SHARING — tree-cheaper but DAG-dearer, and the shared pass missed it | 0 | 0 % |

On the regressing families it is starker: **72 of 99** frontier classes on
the shaders and **18 of 35** on chrome are CYCLE-PRICED — both candidates
scored at the cycle sentinel, so no comparison happened and the pick came
out of `repair_choices_well_founded`, whose job is well-foundedness, not
cost. Half of the shader first-divergences (18 of 32 whose stage is not
`dp`) were settled by that repair pass, and 6 more by `min-of-two` throwing
away the arm that had priced the class correctly.

A bigger graph makes this worse in two ways at once. It adds cost-equal
rewrites, so the strict `<` fires less often and the pick falls to insertion
order — the fraction of live classes so decided rises 43.6 % → 57.6 % on the
shaders and 45.0 % → 59.7 % on chrome across the ladder. And those same
rewrites close cycles (commutativity alone is enough), so more classes land
where the single-DFS DP has no opinion at all. **The extractor does not get
worse at choosing; it gets more places where it does not choose.**

Two further results decide what to build next.

- **A reranker over local moves cannot fix this.** Of 56 objective
  witnesses, **7** are reachable by a single swap from greedy's term — all
  seven on shaders — **none** by a greedy sequence of swaps, and 49 by
  neither. Zero on glyphs, zero on chrome. The `Reranker` seam is over the
  wrong search.
- **The e-graph is not monotone in the class cap** (§1b). 33 of 139
  candidate pairs — 24 % — have a subterm that is simply absent from the
  bigger graph.

And the cheap mechanical fix was tried: canonical tie-breaking is a **net
loss** (§4). It helps the shaders by 0.5–1.7 % and costs the glyphs
0.5–2.4 % and chrome 0.6–3.1 %.

## 1. What a witness is

The e-graph was *expected* to be monotone: a run at a larger class cap
performs every application the smaller run performed and then more, so the
larger graph represents a superset of the terms (L2,
`docs/plans/2026-09-02-optimizer-api.md`). §1b is what happened to that
assumption. Where it does hold — checked, not assumed — the extractor's own
cheaper output at the smaller cap is a term present in the bigger graph that
the extractor walked past. The
instrument looks each of its subterms up in the bigger graph by hash-cons —
read-only, and a miss is a loud failure rather than a skip, since
monotonicity is exactly what is being tested — which yields a second choice
map `C_T` over the same classes. Greedy's is `C_G`. The **divergence set**
is where they differ; the **frontier** is the divergent classes below which
they agree, and at a frontier class the extractor's own comparison can be
read off directly.

**Two witness kinds, never conflated.** An *objective* witness is cheaper in
`ChoiceCost::dag` — the shape-weighted number the extractor actually
minimizes — and indicts the search. A *static-only* witness is cheaper only
in the sweep's unweighted `dag_cost` column while being dearer under the
weighting; that indicts the objective's weighting, not the search, and
realizability is not a question one can ask of it. Both are tabulated,
apart.

## 1b. A finding that arrived before any classification: **the e-graph is not monotone in the class cap**

The whole method rests on L2 (`docs/plans/2026-09-02-optimizer-api.md`): a
bigger budget yields a graph that represents a superset of the terms. The
instrument does not assume it — it looks each subterm up and fails loudly on
a miss — and it **found misses**.

| corpus | candidate pairs where the smaller budget's term was cheaper | mapped | subterm **absent** from the bigger graph |
|---|---:|---:|---:|
| shaders + glyphs (DEV) | 129 | 99 | **30** (23 %) |
| chrome (held out) | 10 | 7 | **3** (30 %) |

Every chrome miss is the same node — `MulAdd(c1429, c0, c5181)`, minted in
`G(10k)` and absent from `G(20k)`, `G(50k)` and `G(100k)` — and the glyph
misses are the same shape: an `fma-fusion` output, or a `Select`, present at
the smaller cap and never minted at the larger one.

That is not a bug in the lookup; it is the claim being false as the loop is
actually run. **A different class cap is a different trajectory, not a
prefix-extension of the same one.** The cap changes which classes get
allocated and in what order, which changes which candidates the scan reaches
before a round ends, which changes which rules fire — so `G(b_hi)` is a
*different* graph that happens to be bigger, not a superset of `G(b_lo)`.

Two consequences, both load-bearing:

- **For the witness argument.** Roughly three quarters of the cheaper-at-a-
  smaller-budget pairs do map, and for those the monotonicity conclusion
  holds constructively — the term *is* in the bigger graph, by exhibition
  rather than by appeal to L2. Those are the witnesses this document
  reports. The rest are not witnesses at all, and are excluded rather than
  assumed.
- **For the optimizer's own contract.** L2 is stated as a property and is
  used as one. It should be either restated (monotone in *applications with
  a fixed cap*, which is a much weaker and probably true claim) or made
  true. Right now the class cap is a budget dimension that silently changes
  the answer's *shape*, not only its quality — which is the same class of
  problem `docs/plans/2026-09-01-production-budget-determinism.md` closed for
  wall clock.

## 2. The shader family: the ladder

Every column is exact. `tied` is the count of live classes where two or more
candidates shared the winning DP cost, so insertion order decided.

| kernel | live 5k → 100k | dag 5k | 10k | 20k | 50k | 100k | tied 5k | tied 100k |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `metaballs` | 50 → 73 | **155** | 140 | 136 | 255 | **270** | 25/50 (50 %) | 50/73 (68 %) |
| `julia_set` | 123 → 174 | **717** | 791 | 697 | 904 | **948** | 77/123 (63 %) | 144/174 (83 %) |
| `mandelbrot_distance` | 96 → 115 | **518** | 558 | 595 | 595 | **595** | 45/96 (47 %) | 93/115 (81 %) |
| `smooth_min_scene` | 39 → 39 | **132** | 132 | 132 | 144 | **144** | 19/39 | 20/39 |
| `psychedelic_packed` | 112 → 111 | **825** | 826 | 829 | 829 | **829** | 33/112 | 34/111 |
| `domain_warp_fbm` | 63 → 58 | **457** | 431 | 446 | 439 | **436** | 22/63 | 19/58 |
| `cosine_palette` | 25 → 25 | 292 | 292 | 292 | 292 | 292 | 7/25 | 9/25 |
| `star_sdf` | 58 → 58 | 172 | 172 | 172 | 172 | 172 | 21/58 | 21/58 |
| `gyroid_slice` | 33 → 33 | 932 | 932 | 932 | 932 | 932 | 12/33 | 12/33 |
| `plasma` | 29 → 29 | 359 | 359 | 359 | 359 | 359 | 9/29 | 9/29 |
| `kaleidoscope_fold` | 40 → 40 | 560 | 560 | 560 | 560 | 560 | 12/40 | 12/40 |
| `smoothstep_vignette` | 49 → 49 | 194 | 194 | 194 | 194 | 194 | 17/49 | 17/49 |
| `torus_slice` | 31 → 31 | 130 | 130 | 130 | 130 | 130 | 11/31 | 11/31 |

Five of thirteen regress from the smallest cap to the largest —
`metaballs` by **+74 %**, `julia_set` by **+32 %**, `mandelbrot_distance` by
**+15 %** — one improves, and seven are flat because saturation quiesced
inside the smallest cap and the ladder never changed the graph. Note
`julia_set` is not even monotone in the budget: 717 → 791 → **697** → 904 →
948. A search whose result moves like that with the size of the space it is
searching is not converging on anything.

## 3. Why: the class-level explanation for `mandelbrot_distance`

The kernel: `dag_cost` 518 at the 5k cap, 595 at 20k and above (the graph
quiesces at 6,287 allocated classes / 115 live). The witness is
greedy's own 5k term, mapped into `G(20k)`.

| | |
|---|---|
| witness `objective` / `dag_cost` | 32,707,328 / 518 |
| greedy at 20k | 37,492,480 / 595 |
| divergence set `\|D\|` | 27 classes |
| frontier `\|F\|` | **3** classes |
| every frontier class's label | **CYCLE-PRICED** (3/3) |
| realizable by local search | **no** — best single swap 37,230,336; greedy swaps 37,230,080 after 2 accepted moves; the witness is 32,707,328 |

The first divergence, class 84:

| | greedy | witness |
|---|---|---|
| node | `Mul(19, 7)` | `Add(19, 19)` |
| weighted own cost | 327,680 | 262,144 |
| **tree-pass DP cost** | **CYCLE** | **CYCLE** |
| shared-pass DP cost | 589,824 | 524,288 |
| minted by | `seed` | — |
| stage that settled it | **repair** | — |
| one-swap delta from greedy | −256 | |

Read that row carefully, because it contains the whole finding.

1. The **tree pass priced both candidates at the cycle sentinel**. A class
   whose child is still on the DFS stack is not scored; `x + x` and `x · 2`
   are in one class here, and commutativity has closed a cycle through it.
   So the tree pass expressed *no preference at all* — the pick came out of
   `repair_choices_well_founded`, whose job is well-foundedness, not cost.
2. The **shared pass did price them, and preferred the witness's node**
   (524,288 < 589,824). The information was in the extractor, in the same
   run.
3. **`min-of-two` threw that arm away.** At 5k the winning objective is
   `Shared`; at 10k, 20k, 50k and 100k it is `TreeCheaper`. As the graph
   grows the arm that cycle-prices classes wins the comparison more often,
   and the arm that priced this class correctly is discarded whole. The
   choice is per-extraction, not per-class: there is no way for the shared
   arm's opinion about class 84 to survive when the tree arm's total is
   lower.
4. The witness is **not reachable by local search**. One swap buys 262 k of
   the 4.79 M gap; greedily accepting improving frontier swaps buys 262 k
   and stops. A reranker over single swaps could not have found this term.

### 3b. The two worst glyphs are a different failure — and not the extractor's

`U+006C` (1,773 → 2,172, +22.5 %) and `U+0066` (1,821 → 2,158, +18.5 %) are
the glyph family's worst regressions in the sweep's `dag_cost` column. Every
witness pair either produces is **static-only**: cheaper unweighted, *dearer*
under the shape weighting the extractor minimizes. `U+0066` 5k → 20k:
`dag_cost` 1,821 → 2,158 but `objective` 455,183 → **411,885**. `U+006C`
5k → 20k: 1,773 → 2,172 but 430,112 → **388,525**.

**On these kernels the extractor got better and the sweep's headline column
says it got worse.** Half of all DEV witness pairs (50 of 99) are like this,
and every one of them is a glyph. Before any more work goes into the glyph
regressions, one of those two numbers has to be shown to be the one the
machine pays.

Their frontier classes are still informative about the extractor. `U+0066`'s
first divergence, class 16:

| | greedy | witness |
|---|---|---|
| node | `MulAdd(7, 15, 1)` | `MulAdd(15, 5, 1049)` |
| tree- and shared-pass DP cost | **CYCLE** | **160** |
| minted by | `fma-fusion` | — |
| stage | **repair** | — |

Greedy holds a node the DP never priced while a **finite-cost** candidate sat
in the same class — a `LOCAL-MISS` produced by the repair pass overwriting a
scored choice. That is the same defect as §3, seen from the other side: the
repair pass is making cost decisions, and it has no cost model.

## 4. The mechanical fix, A/B'd: canonical tie-breaking is a net loss

Ties break to the first admissible node, i.e. insertion order — which is why
the `'8'` bisect saw extraction move under a semantically-null change to the
input. The variant replaces that with a total order on the node's own
content (leaf/op tag, then payload or `OpKind` ordinal, then arity, then
canonical child ids), so the pick no longer depends on when a node was
inserted. It is the same DP otherwise, and `Ties::Insertion` is pinned
byte-identical to production by
`insertion_tie_break_is_productions_extraction`.

The A/B, per family × cap, Σ `dag_cost` over every kernel in the family
(`n` kernels; `better` / `worse` count kernels, not classes):

| family | cap | n | better | worse | same | Σ production → canonical |
|---|---:|---:|---:|---:|---:|---|
| shader | 5k | 12 | 4 | 1 | 7 | 4,618 → **4,595** (−0.50 %) |
| shader | 10k | 12 | 4 | 0 | 8 | 4,691 → **4,632** (−1.26 %) |
| shader | 20k | 12 | 5 | 0 | 7 | 4,645 → **4,593** (−1.12 %) |
| shader | 50k | 12 | 4 | 0 | 8 | 4,976 → **4,894** (−1.65 %) |
| shader | 100k | 12 | 3 | 1 | 8 | 5,032 → **4,978** (−1.07 %) |
| glyph | 5k | 190 | 0 | 70 | 120 | 785,519 → 789,381 (**+0.49 %**) |
| glyph | 10k | 190 | 4 | 92 | 94 | 747,888 → 762,418 (**+1.94 %**) |
| glyph | 20k | 190 | 20 | 76 | 94 | 664,234 → 679,968 (**+2.37 %**) |
| chrome | 5k–100k | 1 | 0 | 5 | 0 | worse at every cap (+0.63 % … +3.12 %) |
| psychedelic | all | 1 | 0 | 0 | 5 | unchanged |

The per-kernel wins on shaders are real and large — `smooth_min_scene`
144 → **124** (−13.9 %) at 50k and 100k, `smoothstep_vignette` 194 → **178**
(−8.2 %) at every cap, `mandelbrot_distance` 595 → **576** (−3.2 %) at 20k
and above, `julia_set` 791 → **760** at 10k — but they do not survive the
glyph family, where 70–92 of 190 kernels get *worse*.

**Answer to "does canonical tie-breaking recover anything": a little, and
not where it matters.** It never recovers a whole witness —
`mandelbrot_distance` goes 595 → 576, not 595 → 518 — because the classes it
decides are ones the DP *scored*, and 29 % of the frontier classes holding
the witnesses were scored at `CYCLE`, where a tie-break has nothing to
compare. Ship it only if the goal is determinism under semantically-null
input changes; it is not the fix for the regression, and on the shipped
glyph corpus it is a regression of its own.

## 5. Glyphs: the budget mostly helps, and where it does not it is the same story

Glyphs are the counter-example that keeps the headline honest. Over the 190
glyph kernels (95 warm-range glyphs × two tile sizes), raising the cap from
5k to 20k takes Σ `dag_cost` from **785,519 to 664,234 (−15.4 %)**; 148
improve, 34 are flat (quiesced inside the smallest cap), and **8 — four
glyphs at both tile sizes — regress**:

| glyph | dag 5k | 10k | 20k | live 5k → 20k | tied 5k → 20k |
|---|---:|---:|---:|---|---|
| `U+006C` (`l`) | 1,773 | 1,827 | **2,172** (+22.5 %) | 432 → 526 | 162 → 185 |
| `U+0066` (`f`) | 1,821 | 1,875 | **2,158** (+18.5 %) | 447 → 526 | 162 → 184 |
| `U+0074` (`t`) | 1,798 | 1,907 | **1,935** (+7.6 %) | 441 → 484 | 165 → 178 |
| `U+006A` (`j`) | 1,828 | 1,958 | **1,956** (+7.0 %) | 450 → 488 | 170 → 182 |

and the biggest wins are large: `U+0024` 12,119 → 9,848 (−18.7 %), `U+0053`
11,756 → 9,669, `U+0067` 11,540 → 9,485.

(The two tile sizes give identical `dag_cost` on every glyph — the kernel
differs only in scale constants — so the family is 95 distinct shapes
counted twice, and every glyph total above is over 190 rows.)

**Tie density falls with the budget on glyphs** — 44.9 % → 38.9 % → 35.0 %
of live classes — the opposite of the shaders. So "more ties" is not a law
of bigger graphs; it is what happens when the rules a kernel's shape admits
are predominantly cost-neutral ones. Glyph kernels are winding-number
arithmetic over many Bézier segments, where saturation finds real
factorizations; the regressing shaders are transcendental-heavy, where the
reachable rewrites are mostly reassociations of the same instruction count.

**Which arm wins is a coin flip on glyphs** — `Shared` on 88, 96 and 94 of
190 kernels at the three caps, `TreeCheaper` on the rest — so the
`min-of-two` decision is load-bearing on roughly half of every glyph
bake.

## 6. Method notes and what this does not show

- Every budget arm is production's optimizer at its own tier's round cap
  with the class cap and the `40 ×` application cap moved — the sweep's
  `cap{b}-app{40b}` arms — with the safety ceiling disabled, since the
  ceiling asserts something about *production's* budget and firing it here
  would say nothing about the extractor.
- The traced re-run asserts its cost equals `Optimizer::run`'s on every pair
  analysed, so a drift between the instrument and the shipped extractor
  aborts rather than being quietly analysed. `Ties::Insertion` is separately
  pinned byte-identical to `extract_dag_scoped` by a unit test.
- **Chrome is held out** and was run once, at the end, in its own process
  (`--chrome`); its rows are in the `-chrome` files. Its `dag_cost` ladder
  reproduces the class-cap sweep exactly: 1,668 at 5k → 2,374 at 100k
  (+42 %).
- The classification order in the denotation put `SHARING` before
  `DISTRACTOR`, which would have made `DISTRACTOR` unreachable since it is a
  refinement of `SHARING`. The implemented order tests `DISTRACTOR` first
  (`SHARING` with a rule-minted node) and `SHARING` for a seed node. Noted
  because the denotation was written first and this is the one place it was
  wrong.
- An `UNPRICED` label exists for a frontier class the winning arm's table has
  no entry for; it never fired.
- This measures `dag_cost` and the extraction objective, **not** the emitted
  kernel's wall clock. `docs/plans/2026-09-07-benchmark-correction.md` is
  where a clock claim would come from, and none is made here.
- The `seconds` column in the budget CSV was taken on a shared box under
  load averages between 9 and 168, including a concurrent `cargo test
  --workspace`. It is not a timing measurement.

## 7. The numbers

**Corpus.** 12 ShaderToy kernels at 256×256, `psychedelic_packed` at
1920×1080, 190 glyph kernels (95 warm-range glyphs × two tile sizes), and —
held out, reported once — `chrome_packed` at 1920×1080. 635 budget rows on
the DEV corpus, 5 on chrome. Ladder: `{5k, 10k, 20k}` class caps for glyphs
and `{5k, 10k, 20k, 50k, 100k}` for the rest, each with the sweep's `40 ×`
application cap.

### 7.1 Direction of `dag_cost`, smallest cap → largest

| | shader | psychedelic | glyph | chrome |
|---|---:|---:|---:|---:|
| rose | 4 | 1 | 8 | **1** |
| fell | 1 | 0 | 148 | 0 |
| flat (quiesced inside the smallest cap) | 7 | 0 | 34 | 0 |

Worst by ratio: `metaballs` ×1.74, `chrome_packed` ×1.42, `julia_set` ×1.32,
`U+006C` ×1.23, `U+0066` ×1.19, `mandelbrot_distance` ×1.15.

### 7.2 Witness pairs

| | pairs analysed | objective witnesses | static-only witnesses | monotonicity misses |
|---|---:|---:|---:|---:|
| DEV | 99 | 49 | 50 | 30 |
| chrome | 7 | 7 | 0 | 3 |

Half the DEV pairs are **static-only** — cheaper in the sweep's unweighted
`dag_cost` column and *dearer* under the shape weighting the extractor
minimizes. Every one of those is a glyph. That is not a search failure; it
is the sweep's headline column and the extractor's objective disagreeing,
and it is worth its own investigation: if the weighting is right, the
sweep's `dag_cost` column overstates the glyph regressions.

### 7.3 Frontier-class labels

Over the 334 frontier classes of the 56 objective witnesses (see §0 for the
totals), and over all 796 frontier classes including static-only pairs:

| label | objective only | all pairs |
|---|---:|---:|
| COORDINATED | 191 | 505 |
| CYCLE-PRICED | 98 | 112 |
| DISTRACTOR | 28 | 82 |
| TIE | 15 | 49 |
| LOCAL-MISS | 2 | 14 |
| SHARING | 0 | 34 |

By family, objective witnesses only:

| family | CYCLE-PRICED | COORDINATED | DISTRACTOR | TIE | LOCAL-MISS |
|---|---:|---:|---:|---:|---:|
| shader | **72** | 16 | 10 | 0 | 1 |
| chrome | **18** | 13 | 2 | 1 | 1 |
| glyph | 8 | **162** | 16 | 14 | 0 |

`SHARING` is empty among objective witnesses and small (34) overall: the
sharing-aware pass is *not* where these terms are being lost. `LOCAL-MISS`
is 2 among objective witnesses — the DP is not failing to take a candidate
it scored as cheaper. Both negatives matter: they rule out the two
hypotheses that would have been cheapest to fix.

### 7.4 The first divergence, per witness

| | |
|---|---|
| label | COORDINATED 31, CYCLE-PRICED 22, DISTRACTOR 3 |
| **stage that settled greedy's node** | **`dp` 32, `repair` 18, `min-of-two` 6** |
| rule that minted greedy's node | `constant-fold` 22, `fma-fusion` 10, `seed` 7, `associative(Mul)` 7, `associative(Add)` 3, `distribute` 3, `factor` 2, `reverse-associative(Add)` 2 |

**24 of 56 first divergences were not made by the DP at all** — 18 by the
repair pass and 6 by `min-of-two` discarding the arm that had the witness's
node.

### 7.5 Top distractor rules

A DISTRACTOR is a rule-minted node with a lower local DP cost that is dearer
once sharing is accounted for. Counted over every frontier class of every
witness:

| rule | DEV | chrome |
|---|---:|---:|
| `associative(Add)` | **53** | 0 |
| `fma-fusion` | **23** | 0 |
| `reverse-associative(Add)` | 4 | 0 |
| `associative(Mul)` | 0 | 2 |

The two suspects named in advance — fma-fusion and distribute — are half
right: `fma-fusion` is second, `distribute` never mints one, and
`associative(Add)` is more than twice `fma-fusion`. Reassociation is the
rule family that most often hands the DP a locally cheaper node whose real
cost lives in the sharing it broke.

### 7.6 Realizability: can local search reach the witness?

| | REALIZABLE-1 | REALIZABLE-k | PARTIAL | COORDINATED |
|---|---:|---:|---:|---:|
| shader | **7** | 0 | 24 | 2 |
| glyph | 0 | 0 | 2 | 14 |
| chrome | 0 | 0 | 4 | 3 |
| **total** | **7** | **0** | **30** | **19** |

`REALIZABLE-1` means one accepted frontier swap from greedy's term already
reaches the witness's cost. `PARTIAL` means greedily accepting improving
swaps buys *something* but not enough; `COORDINATED` means it buys nothing.
Seven of fifty-six. **Nothing is `REALIZABLE-k`** — where one swap is not
enough, a sequence of single swaps never gets there either, because each
individual move is uphill.

### 7.7 Which arm `min-of-two` kept

| family | 5k | 10k | 20k | 50k | 100k |
|---|---|---|---|---|---|
| shader (of 12) | 3 Shared | 3 | 1 | 3 | 2 |
| glyph (of 190) | 88 Shared | 96 | 94 | — | — |
| chrome | Shared | Shared | Shared | Shared | **TreeCheaper** |

The decision is load-bearing on roughly half of every glyph bake, and on
`mandelbrot_distance` and `chrome_packed` it *flips to the tree arm at the
largest cap* — the arm that cycle-prices classes wins the whole-term
comparison exactly where the graph has grown enough to make cycle-pricing
common.

## 8. What this says to do next

Ordered by what the data supports, not by what is easiest.

1. **Make the DP a fixpoint.** `docs/results/2026-09-02-extraction-gap.md`
   asked for this and named it defect (i); it was never done, and 29 % of
   the frontier classes that hold witnesses — 73 % on the shaders — are
   classes the single-DFS DP never scored. Knuth's AND-OR Dijkstra settles a
   class when its cheapest candidate has every child settled, whatever the
   DFS order, and needs no repair pass at all. `extraction_gap.rs` already
   has an unweighted implementation to lift.
2. **Make `min-of-two` per class, not per term.** Six first-divergences were
   lost because the losing arm held the right node. Both arms produce a full
   choice map over the same classes; there is no reason the winner must be
   chosen whole.
3. **Do not build a reranker over swap refinement.** 7 of 56 witnesses are
   one swap away, 0 are a sequence of swaps away. The `Reranker` /
   `IncrementalExtractor` seam is over a neighbourhood that does not contain
   the answer. If a learned component belongs anywhere here it is in the
   *fixpoint's* candidate ordering, or in choosing among coordinated
   multi-class changes — which is a different search.
4. **Restate or repair L2.** The class cap changes the answer's shape, not
   only its quality (§1b). Either the monotonicity claim gets qualified to
   "at a fixed class cap", or the loop is changed so a bigger cap really is
   a prefix-extension.
5. **Reconcile `dag_cost` with the extraction objective.** Half the DEV
   witness pairs are cheaper in one and dearer in the other, all of them
   glyphs. One of the two numbers is describing something the machine does
   not pay for.

Not on the list: canonical tie-breaking as a performance fix (§4), and
anything about the sharing-aware pass or a DP bug (§7.3 rules both out).
