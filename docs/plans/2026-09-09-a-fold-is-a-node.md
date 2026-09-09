# A fold is a node, its decompositions are rules, and legalize comes last

**Date:** 2026-09-09
**Status:** Phases 1 and 2 executed — `Reduce` is a typed node, representable
in the e-graph, and its decompositions are rules the runtime tier holds.
**Phase 3 (the pipeline reorder) is written, measured, and not landed**: it
costs 23–42% more emitted nodes on every production glyph. §9 has the numbers
and the mechanism.
**Author:** JP (the direction), Claude (draft)
**Related:** [a-run-is-a-glyph](2026-09-09-a-run-is-a-glyph.md),
[macro-tier-is-arena-native](2026-09-08-macro-tier-is-arena-native.md),
[schedule-cost-model-denotation](2026-09-01-schedule-cost-model-denotation.md).

---

## 0. The method

The one from [a-run-is-a-glyph](2026-09-09-a-run-is-a-glyph.md) §0: grep the
clusters, write the problem in plain English, read the parts of speech off,
cross-reference against the draft for YAGNI, then look for the mathematical
structure. What follows is that, applied to `Reduce`.

## 1. The clusters

- **Three `Const(f32)` children carrying metadata.** `push_reduce` wrote
  `Nary(Reduce, [Const(combiner.index()), Const(var), Const(extent), body])`.
  Four readers decoded it — `passes::unroll_reduce`, `variance`'s
  `bound_index_slot`, `eval`'s `reduce`, `kernel`'s `lowest_free_index_slot` —
  each with its own failure mode: a panic, a `None` that meant "claim maximal
  dependency", a `checked_sub` with a comment explaining what wrapping would
  cost.
- **An arity assertion inside a variadic node.** `debug_assert_eq!(l, 4,
  "Reduce has 4 children")`, and `OpKind::Reduce => 4` in the arity table.
- **`OpKind::Reduce` in `op_from_kind`'s `None` arm**, grouped with
  `RawGather` under "lowered before/after the e-graph and never appear in one".
- **`OpKind::Reduce => 0` in the cost table**, commented "lowered (unrolled)
  before costing" — a cost that is only true because of the pipeline order.
- **`pipeline![ExpandRefs, LowerDwrt, ExpandReduce, Saturate]`**, whose own
  comment argues the *opposite* case for `LowerDwrt`: "Lowering after the
  e-graph leaves those folds permanently on the table, because nothing folds
  post-extraction."

CLAUDE.md already names the first of these as a defect — "`push_reduce`
encodes an `OpKind` as a `Const(f32)`" is in its list of places where the
meaning lives in a comment instead of a type.

## 2. The sentence

> A reduction folds a body over a contiguous range of an index, under a
> monoid. It means the same thing computed as one term peeled off a shorter
> reduction, as a split into two, or as a fully unrolled chain. The compiler
> should be free to choose among those by cost, the way it chooses between
> `a*b+c` and `fma(a,b,c)` — which means the choice belongs in the e-graph,
> and the unrolling that has to happen anyway belongs after it, as the
> fallback for whatever the graph declined.

## 3. Parts of speech

- **Nouns**: reduction, body, index, range, monoid, term.
- **Verbs**: *folds*, *peels*, *splits*.
- **Sum**: none new.
- **a/b/c or letters**: the combiner is one of `+ × min max ∧ ∨` — answered as
  letters long ago; `Monoid` already exists and is exactly this.

Two nouns have no type: **index** (a `u8` in a magic range) and **range**
(an extent with an implicit lower bound of zero).

## 4. Cross-reference, with rejections

| candidate | demand in the draft | verdict |
|---|---|---|
| `Monoid` for the combiner | already exists; `assert!(is_monoid)` at construction plus `OpKind::from_index(…).expect(…)` at two decode sites | **use it** — no new type at all |
| `Binder` | `4..8` asserted at construction, decoded with a silent-`None` fallback, scanned for in a third place; a wrong value substitutes a literal for a *coordinate* | **build** |
| range `[lo, hi)` | the decompositions need both ends (`sum(x over(1,10))`); today `lo` is implicit | **build** — as fields of `Fold`, not a type of its own |
| `Fold` grouping the three | `push_reduce` would take five arguments; the three travel together at every site, and `Fold` *is* the node's metadata | **build** |
| `BinderRange` as its own newtype | one constructor, one accessor, no site holds a range without the rest of the fold | **reject, YAGNI** — two fields inside `Fold` |
| `Fold::split` | a real law (`⊕_{[lo,hi)} = ⊕_{[lo,m)} ⊕ ⊕_{[m,hi)}`), written and tested, then found to have no caller | **rejected in flight** — build it when a rule wants it |
| `Monoid::tag()` returning a serializable byte | proposed for the JIT-cache key; then the canonical key and the macro emitter wanted the same thing for the *whole* fold | **rejected in flight** — replaced by `Fold::to_bits`/`from_bits`, one method for three callers |

**Why the range is the load-bearing half.** Over an extent, peeling the first
term off `⊕_{[0,n)} f` leaves `⊕_{[0,n-1)} f(·+1)` — a substitution through the
whole body, which in an e-graph means a rebuilt subgraph per peel. Over a
range it leaves `⊕_{[1,n)} f`: the body is unchanged and therefore *shared*.
One end of an interval is the difference between a rule that fires and a rule
nobody can afford. The user wrote it in the range form
(`sum(x over(1,10)) = 1 + sum(x over(2,10))`) before any of this was built.

**Why the binder gets a type and the trip count does not.** CLAUDE.md's
priority rule: a wrong value that is *silently* representable deserves a type;
one that panics on the next line does not. A binder index below the base names
a lattice coordinate, so substituting a literal for it replaces every `X` in
the body and produces plausible, wrong pixels. A trip count out of range
panics at the constructor.

## 5. Structure

```text
Reduce { fold, body }  ⟦·⟧ = ⊕_{k ∈ fold.range()} ⟦body⟧[fold.binder() := k]
```

**The decompositions, as laws:**

```text
⊕_{[lo,hi)} f  =  ⊕_{[lo,hi-1)} f ⊕ f(hi-1)        (peel, from the back)
⊕_{[lo,lo)} f  =  identity(⊕)                       (empty)
```

The identity is what makes `empty` a *value* rather than a special case.
A `split` at an interior point is the third law and is **not built**: no rule
wants it yet, and a law with a test but no caller is machinery ahead of demand.

**The observation that unifies two mechanisms:** `expand_reduce` is
`Fold::peel_back` run to exhaustion. They are not a legalizer and a rule; they
are one operation at two budgets, and they call the same method — which is
not ceremony, because while each did its own range arithmetic the two produced
*opposite associations*, and an e-graph then had to spend reassociation rules
reaching the shape the legalizer built directly.

**The binder is the only thing in the language that *shrinks* a variance set.**
Every other node unions its children's; a fold removes its own index. That is
what makes `Σ_k f(k)` frame-uniform when `f` reads no coordinate, and it is
what the old encoding could only report by giving up.

## 6. What changes

**Phase 1 — a fold is a node.** *(done)*

- `pixelflow-ir/src/fold.rs`: `Monoid` (moved), `Binder`, `Fold`, with
  `peel`, `split`, `is_empty`, and `to_bits`/`from_bits`.
- `ExprNode::Reduce { fold, body }` — **one** child, not four, so nothing that
  walks operands can reach the metadata, fold it, or cost it. Fits the 16-byte
  `ExprNode` cap with a `u16` range (see §9).
- `Shape::Reduce` and `ENode::Reduce`: the metadata is part of the node's
  *identity*, so hash-consing is on exactly the fold's meaning.
- `op_from_kind` no longer needs an arm: a fold is not an op, so there is no
  `Op` for a rule to name and no vocabulary decision to make.
- Cost: the prohibitive sentinel, for `Dwrt`'s reason (§9).

**Phase 2 — the decompositions are rules.** *(done)*
`PeelFold` and `EmptyFold` in `egraph/fold_rules.rs`, on the **runtime tier's**
rule set (`RuleSet::runtime`) rather than in `all_rules`: `kernel!` has no
syntax that builds a fold, so at the macro tier they can never fire, and
adding them to `all_rules` would perturb the pinned rule-set grids
`math::inflate`'s inflation study measures against for no measurement's sake.

They are dynamic rules with constructors, not templates, because peel has to
build `body[binder := lo]` — a substitution over an *e-class*, which is the
standard place e-graphs and binders meet. Two things make it sound and
affordable: a **representative suffices** (`f ≡ g ⟹ f[x:=c] ≡ g[x:=c]`, the
same argument `ChainRule` makes when it differentiates one representative),
and **hash-consing is the invariance test** (a subtree that never mentions the
binder rebuilds to the identical node, so the rule names its class instead of
copying it). A class that reaches itself makes the walk decline, which costs
completeness and never soundness.

**Phase 3 — legalize is the fallback.** *(written, measured, not landed)*
`pipeline![ExpandRefs, Optional(Saturate), LowerDwrt, ExpandReduce]`. What the
graph resolved, it resolved; what it declined, the legalizer lowers, because
codegen has no iteration binder and no `Dwrt`. See §9 for what it measured and
why it is not in the tree.

## 7. Gates

- `pixelflow-search/tests/a_fold_is_a_node.rs` (deleted) — a fold inserts; hash-conses
  on exactly its meaning (algebra, binder, range, body); keeps its binder out
  of the variance it reports; round-trips through extraction; an empty fold is
  its identity.
- `pixelflow-ir/src/fold.rs`'s own tests — peel to exhaustion *is* unrolling,
  split partitions, a reversed or oversized range is refused.
- Unchanged and green: `loop_blinn_winding`, `run_is_a_glyph`,
  `kernel_glyph_golden`, `glyph_atlas_golden`, `freetype_oracle`,
  `collapse_loop`'s matmul, `jit_cache`'s identity tests.
- Phase 2 will need: peel and the original land in one e-class; unrolling by
  rule agrees with `expand_reduce` pixel for pixel.
- Phase 3 will need: the glyph goldens unchanged, *and* a measurement that
  saturation still folds across the unrolled winding terms — the whole risk of
  the reorder.

## 8. What this does not do

- No rule matches a fold yet. Phase 1 buys hash-consing and CSE of a fold's
  body, and the *possibility* of Phases 2–3; it does not change what any
  production kernel compiles to.
- Does not touch `Kernel::over`'s surface. `sum_over(n, f)` still means
  `[0, n)`; the lower bound exists for the rules, not for users.
- Does not give `Dwrt` a rule for a fold. `d(⊕_k f) = ⊕_k d(f)` is linearity —
  true for `Σ`, false for `Π`, `min`, `max` — and it belongs in the rule set
  where the graph can decline it, not in the fallback lowering. Not reachable
  today: production glyph kernels put the `Dwrt` *inside* the fold's body
  (`Distance::in_pixels` is called per piece), never around it.

## 9. What execution found

**The 16-byte cap decided the range's width.** `ExprNode` is capped at 16
bytes and its alignment is 8 (`Ref(KernelKey)`), so a `Reduce` payload of 12
bytes plus a tag rounds to 24. A `u16` range fits exactly. That is not a
truncation to regret: **a `Reduce` is not a loop.** The language is a DAG with
no iteration binder, so a fold that survives to codegen is emitted as
`len()` copies of its body; a trip count is bounded by what one can afford to
*emit*, and the e-graph's own class budget dies orders of magnitude below
2¹⁶. `Fold::new` rejects a larger range rather than wrapping.

**The cost model cannot express a fold's cost, and saying so is the finding.**
`CostFunction::node_cost(&ENode, parent)` sees a node, never its children's
costs, and the extractor sums node costs over the DAG. A fold's cost is
`len × cost(body)` — *not additive in its operands*, the one shape this
interface cannot say. Phase 1 prices it at `Dwrt`'s prohibitive sentinel, and
the reason is exact rather than approximate: a surviving fold gets unrolled
afterwards by a legalizer that runs after everything that could have folded or
CSE'd across the copies, so **any decomposition sharing the e-class is the
same machine code plus the optimizations**, and extraction should take it
whenever one exists. This is precisely the non-additive schedule cost the
`Reranker` seam exists for
([schedule-cost-model-denotation](2026-09-01-schedule-cost-model-denotation.md)).

**Thirty walkers matched a case to read a property.** Adding a variant to
`ENode` compiled everywhere the code used a `match`, and silently skipped a
fold's body everywhere it used `if let ENode::Op { children, .. }`. The first
symptom was `post_order` never visiting a fold's body, so `cost_of_choices`
panicked on an unvisited child — a *test* failure, but only because the test
existed. Every one of those thirty sites was asking "give me this node's
children", which `ENode::children_slice` already answers with no case at all.
Folding them (and adding `children_slice_mut` for the two canonicalizers) is
"fold before you dispatch" applied to about thirty peers at once, and it is
what makes the *next* node variant a compile error everywhere instead of a
silent skip in two dozen walkers.

**One method replaced three.** The JIT-cache key, `key::canonical`, and the
`kernel!` emitter each needed a fold as bytes; the emitter also needed the way
back. `Monoid::tag()` — drafted for the first of them — was deleted in favour
of `Fold::to_bits`/`from_bits`, which is total in both directions, so a
corrupt cache key or a hand-edited token stream is a `None` at the boundary
rather than a fold that means something else.

**Assertions deleted rather than moved.** `push_reduce` asserted that the
combiner was a monoid and the var index a binder; `variance` had a
"malformed binder" arm that returned `Variance::ALL`; `expand_reduce` had a
"reduce extent must be a Const" panic with a test pinning it. All four are
gone, and none was replaced by a check somewhere else — the argument types
refuse the values. The test that pinned the extent panic is now a test that
the extent is not reachable from the node's children at all.

**Phase 3 measured, and not landed.** The reorder is small — `Optional` (a
step whose declining means "unoptimized", not "unlegalized", so the legalizing
suffix still runs) and four lines of pipeline. On every production glyph it
emits **23% to 42% more nodes**:

| glyph | before | after | Δ |
|---|---|---|---|
| A@16 | 1041 | 1479 | +42% |
| O@16 | 2938 | 3689 | +26% |
| S@16 | 3178 | 3949 | +24% |
| 8@32 | 6747 | 8369 | +24% |
| g@32 | 4810 | 6029 | +25% |

Three candidate explanations were tested and eliminated:

- **Not the class budget.** At 8× the cap and 9× the wall clock the gap closes
  by 1.5%.
- **Not the association.** `expand_reduce` builds a left-leaning chain and the
  first peel built a right-leaning one; `Fold::peel_back` fixes that and
  changes nothing. (It is kept, because matching the legalizer's shape is
  right regardless.)
- **Not a partially-unrolled term being extracted.** Saturation telemetry
  shows the run *quiescing* — no rule left to fire — with the folds intact,
  and the extracted arena identical for every glyph at 154 nodes: the folded
  form. `ExpandReduce` then unrolls after extraction, which is after
  everything that could have folded or CSE'd across the terms. That is the
  whole 23–42%.

What is *not* explained: why `PeelFold` stops firing at that scale, when a
40-term fold whose body reads a bound table unrolls fine
(`fold_rules::production_shape_tests`). The two candidates left are the growth
gate (`saturate_bounded` stops the whole scan at the first action that would
exceed the class cap) and a cycle in the substitution walk, which declines.
Until that is answered the reorder is a 30% regression, so the order stands
and the legalizers still run first.

**The retrospective is about the gate.** Every existing glyph test passed
under the reorder — the goldens, the JIT-vs-interpreter oracle, the winding
oracle, the FreeType oracle. None of them reads the *size* of the code that
draws the pixels, so a 42% regression was invisible to CI, and the only reason
it was caught is that this change happened to come with a measurement. That is
a gap in the gate, not a lucky catch, and the fix is a check:
`pixelflow-graphics/tests/glyph_optimization_cost.rs` bounds the emitted node
count for three glyphs, and fails on all three under the reorder.
`examples/glyph_saturation_cost.rs` is the full sweep behind it.

**One more place a fold's cost is not additive.** §9's note about `node_cost`
has a twin in the budget. `Budget::Production` picks its tier from the input's
node count and sizes the classical class cap from its *inserted class* count —
both of which describe how compactly the term is written, not what it denotes.
A folded glyph arena inserts to 154 classes where its unrolled form inserts to
1193–2967, so under the reorder the same kernel asks for a budget an order of
magnitude smaller. That is not what sank the reorder (the budget experiment
above rules it out) but it is wrong on its own terms, and it is the same
mistake in a second place: a fold's size, like a fold's cost, is `len ×
body`.
