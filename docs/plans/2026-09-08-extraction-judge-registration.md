# Pre-registration: can a learned per-class node scorer, used as the extraction DP's local objective, beat the static table's argmin?

**Date:** 2026-09-08
**Status:** REGISTERED — committed before any scorer, any training run, or
any guided extraction exists. No number in §3–§8 may be revised after the
first guided extraction under this registration. Results are appended in §9
against the gates below. A different statistic, threshold, family, cap or
metric is a NEW registration that supersedes this one.

**Authority (inherited, binding):**
- `docs/plans/2026-09-01-phase3-registration.md` — budgets in **rule
  applications**, deterministic cost under `CostModel::latency_prior()`,
  **no timing in any metric**, family-held-out discipline, FINAL untouched.
- `docs/plans/2026-09-01-schedule-cost-model-denotation.md` — the record of
  what the deleted extraction head was and why its shape closed.
- `docs/results/2026-09-08-extraction-witnesses.md` — the witness data every
  claim below is argued from.
- `docs/results/2026-09-08-extraction-fixpoint.md` — the mechanical fix that
  had to land first, and the residual it leaves.

This document cites those and adds a claim; it moves none of their gates.

---

## 0. Why this is not the head that was deleted

The extraction head removed in September 2026 (#1093) **predicted the total
cost of an extracted expression** and was used to rank whole extractions. It
tied the static latency-prior table on schedule-free kernels, and CLAUDE.md's
"Cost Model and the Guide" records the reading: *the cost model was never the
failure.* Re-proposing a cost predictor here would be re-running a closed
experiment.

The witnesses say the failure is **argmin**, not cost. Take
`mandelbrot_distance` at the 20k cap in the witness run: the extractor's own
DP had the witness's node priced *cheaper* than the one it kept (524,288 vs
589,824) in the arm it then discarded. Nothing about the cost model was
wrong; the search threw the answer away. So the learned component registered
here **emits a decision, not a number**: which node of a class the DP keeps.
Its output is never read as a cost, never summed, and never compared against
the table's cost — the table remains the objective the *result* is scored by,
which is what makes the experiment falsifiable at all.

## 1. What the witnesses have already ruled out

Two options were on the table for this session. The data closes one of them
and constrains the other.

**(b) A reranker over the swap-refinement search — DECLINED.** The
`Reranker` / `IncrementalExtractor` seam exists and no `impl` ships. Of the
56 objective witnesses,

| | REALIZABLE-1 | REALIZABLE-k | PARTIAL | COORDINATED |
|---|---:|---:|---:|---:|
| all families | 7 | **0** | 30 | 19 |

7 of 56 are one accepted swap from greedy's term, **zero** are a sequence of
swaps away, and after the fixpoint landed every surviving objective witness
on the shaders and on chrome is `PARTIAL`. A reranker ranks candidates a
search proposes; this search proposes a neighbourhood that provably does not
contain the answer. Building a model to rerank it would be measuring the
model against the wrong hypothesis space, which is the methodological error
`docs/plans/2026-07-07-guided-saturation-redesign.md` closed a whole program
for. **The seam stays unimplemented, and this registration does not use it.**

**(c) A learned pruner making #1115's exact branch-and-bound fit the compile
budget — NOT FIRST, but it is the ceiling and the label source.** It is the
strongest form of JP's registered research question ("can we use CPU neural
nets to scale an e-graph beyond the traditional limitations?"): exact argmin
made tractable. It is not first because its training signal does not exist
yet at the scale it needs — `extraction_gap.rs`'s exact DAG search completed
on **93 of 302** kernels — and because a pruner is only measurable against
an exact baseline it is itself trying to approximate. §6 registers what (a)
must produce before (c) is worth registering separately.

## 2. The candidate: a per-class node scorer as the DP's local objective

**Denotation.** `settle_in_cost_order` settles a class on
`argmin_{n ∈ class} price(n)`. The candidate replaces that with

```text
argmin_{n ∈ class}  price(n) · (1 + λ · σ(s(n)))
```

where `s(n) ∈ ℝ` is the learned score of node `n` in its class, `σ` is
`tanh`, and `λ` is a fixed, registered scale. The form is a **residual on
the table**, never a replacement for it: at `λ = 0` the arm is byte-identical
to production, which is the null arm and is checked as such (§7 control C0).

**Why this shape and not a per-class independent classifier.** 57 % of the
frontier classes holding witnesses are `COORDINATED` — the witness needs
`k > 1` classes to change together, so no rule that decides one class in
isolation and leaves the rest alone can reach it. A local objective inside
the DP is not such a rule: the DP recomputes the **whole** choice map under
it, so one weight change moves many classes in one pass, and the map stays
well-founded by construction. That is the only one of the three options with
this property, and it is why the witness data picks it.

**Features per candidate node `n` in class `c`** — all read from state the
DP has already computed when it prices `n`, so the scorer adds no traversal:

| feature | source |
|---|---|
| `OpKind` of `n` (one-hot, projected through `OpEmbeddings`) | `ENode` |
| the rule that minted `n`, or `seed` | `Provenance::origins` |
| `n`'s weighted own cost, log-scaled | `weighted_own` |
| each child class's settled cost and variance class | the DP tables |
| `n`'s arity, and how many of its children are already shared | the DP tables |
| the local-vs-shared spread: `price_tree(n) − price_shared(n)` | both passes |

`OpEmbeddings` is prior-seeded and already in the tree; the typed edge
stream and `Extraction::chosen_variance` are the surviving seams from the
deleted head and are reused rather than rebuilt.

**Labels.** For each witness pair, the witness's choice map `C_T` restricted
to the divergence set. A class in `C_T` labels its witness node `+1` and
every other candidate of that class `−1`; the loss is a per-class softmax
cross-entropy over candidates, which is a ranking loss within a class and
says nothing about costs across classes. Where `extraction_gap.rs`'s exact
DAG search completes, its optimum supersedes the witness as the label.

## 3. The claim, stated so it can fail

**H₁:** a scorer of this shape, trained on held-out families' witness choice
maps, lowers the extraction objective at the **largest** class cap, relative
to the post-fixpoint greedy DP, by more than a uniform-random tie-break does.

**H₀:** it does not — the whole effect is "something other than insertion
order broke the tie", which is what the rules-by-nodes filter found for
saturation (`docs/plans/2026-09-08-rules-filter-bilinear-registration.md`: a
uniform-random filter at matched keep-rate equalled a trained bilinear one).

## 4. Metric — fixed here

**Primary, per family:** `R_f` = **witness-recovery rate** — the fraction of
that family's objective witness pairs (as defined by
`egraph::witness`, at the same budget ladder) that no longer exist, i.e.
where the arm's term at the higher cap is no dearer in `ChoiceCost::dag` than
its own term at the lower cap. Intrinsic, deterministic, and it is the
quantity the whole program is named after.

**Secondary, per family:** `S_f` = Σ `ChoiceCost::dag` (the *objective*) over
every kernel at the largest cap in that family's ladder, reported as a ratio
to the greedy arm's.

**Reported, never a gate:** unweighted `dag_cost`; the two disagree on 50 of
99 DEV witness pairs and the disagreement is unresolved
(`2026-09-08-extraction-witnesses.md` §7.2). Quoting the column that is not
the objective as a win or a loss is what this line forbids.

**Constraint, hard:** emitted `bytes` no worse than the greedy arm's on any
kernel, by more than 1 %. An extraction that wins the objective by emitting
a materially bigger kernel has moved the cost, not removed it.

**Reported, not a gate:** added extraction wall clock, as a percentage of
the greedy arm's, at the largest DEV kernel. Timing is never a metric here
(inherited from the phase-3 registration) but a scorer that doubles compile
time is a different proposal and the number must be visible.

## 5. Decision rule — fixed here

Families: `shader` (12 kernels), `glyph` (190), `chrome` (1, held out).

1. **Train on glyphs, evaluate on shaders. Train on shaders, evaluate on
   glyphs.** Chrome is touched **once**, at the end, after both directions
   are reported, and never trained on.
2. **H₁ is accepted** iff, in **both** held-out directions:
   `R_f(learned) − R_f(C2) ≥ 0.15` **and** `S_f(learned) ≤ 0.98 · S_f(C0)`,
   with the bytes constraint met on every kernel.
3. **H₀ is recorded** in every other case, including "better than greedy but
   not better than random-among-ties" — that outcome is a finding about the
   tie structure, not about the model, and it is reported as such.
4. A margin missed in one direction and cleared in the other is **H₀**, with
   the asymmetry reported. Rules are domain-conditional
   (`feedback_rules-are-domain-conditional`), so a single-family win is
   exactly the result that has fooled this project before.
5. Hyperparameters are swept on the **intrinsic** metric only (the per-class
   ranking loss on held-out witness classes), before any extrinsic run, and
   **one** extrinsic run per direction is registered. An untuned null is not
   a null (`feedback_optuna-before-quoting-any-learned-null`).

### 5.1 Where the margin comes from

`0.15` is not a round number picked for feel. After the fixpoint there are
**28** objective witness pairs on DEV (`shader` 4, `glyph16` 8, `glyph32` 9,
`psychedelic` 7) and **7** on chrome. A margin of 0.15 over C2 is therefore
"at least four more DEV pairs eliminated than uniform-random-among-ties
manages", which is above the spread five C2 seeds can produce on a
population that small — the seed spread is measured **before** the learned
arm runs, and if it turns out to exceed 0.15 the margin is raised to it and
that raise is recorded here as an amendment with its date, before any
learned number exists.

`S_f ≤ 0.98 · S_f(C0)` is the same discipline in the other statistic: the
fixpoint itself moved Σ objective by −2.91 % over 635 rows, so a learned
component that cannot find 2 % on top of a mechanical fix has not earned the
capacity.

## 6. Controls — all four run, all four reported

| id | arm | what a tie with it means |
|---|---|---|
| **C0** | post-fixpoint greedy DP (`λ = 0`) | the null; the arm must be byte-identical here, and a unit test pins it |
| **C1** | canonical tie-break alone (`Ties::Canonical`, already implemented) | the win is determinism, not learning — and this control is already known to be a net loss on glyphs (+0.5…+2.4 %) and chrome |
| **C2** | uniform-random choice among the argmin set, 5 seeds, median | **the null that matters**: the effect is "not insertion order", and no model is needed |
| **C3** | the exact DAG optimum from `extraction_gap.rs` where it completes (93 kernels) | the ceiling — `R_f` and `S_f` for a perfect argmin, so the gap the learned arm closed is a *fraction*, not an absolute |

C2 is the control the rules-filter result demands. C3 is what makes any
result interpretable: "recovered 40 % of the witnesses" means nothing until
the reachable maximum is on the same table.

## 7. Cost budget — CPU-resident, fixed here

The scorer runs inside a bake. Its budget is stated per candidate because
that is what the DP iterates:

- **≤ 1,536 MACs per candidate node.** Two layers, ≤ 32 hidden dims, over a
  feature vector of ≤ 48 entries. No layer wider than 32.
- **≤ 3 × 10⁷ MACs per extraction** at the largest DEV kernel. The observed
  live-class counts are 25–174 (shaders), ~100–530 (glyphs), 410–442
  (chrome); at ≤ 20 candidates per class this bounds candidates per
  extraction at ~10⁴.
- **No allocation per candidate.** The accumulator is reused across
  candidates the way `EdgeAccumulator` already is.
- Anything that does not fit is **not this proposal** and needs its own
  registration.

## 8. What is explicitly not being built

- No reranker over swap refinement (§1).
- No cost predictor, no value head, no critic, no RL — the loss is a
  supervised per-class ranking loss over hindsight labels, the same shape as
  the saturation Guide's (`docs/plans/2026-08-31-guide-design-revision.md`).
- No change to `graph.rs`'s insertion, memo, union or rebuild.
- No claim about emitted-kernel wall clock. That claim comes from
  `docs/plans/2026-09-07-benchmark-correction.md` and is a separate exercise.

## 9. Results

*(empty — appended after the registered runs)*
