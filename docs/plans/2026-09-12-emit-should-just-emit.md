# Emit should just emit

**JP, 2026-09-12.** Lift the guard decision out of the emitter, give the graph
a hard and a soft `Select`, and let extraction choose between them.

## The measurement that forces this

callgrind on `'@'` at tile 16, at `9643f3b` — *after* the three guard
optimisations that already took emit down 76%:

| | share of the program |
|---|---|
| `emit::compile` | 89.7% |
| **`guards::select_arms`** | **80.5%** |
| `guards::cluster_select_arms` | 60.6% |
| `LinearScan::allocate_nest` | 25.9% |
| the assembler | does not appear |

**What calls itself "emit" is four fifths a search.** The three fixes made the
search cheaper per operation — 5.09e9 instructions to 3.06e9 — and did not
change its share, because they optimised the reconstruction instead of
removing it.

Netting the search out: `'@'` would emit 655,866 bytes in roughly 36 ms, about
**18 MB/s**, which is an ordinary fast backend. The emitter is already fast. It
is carrying a passenger.

And the passenger exists for a reason the backlog states as its opening
pattern: *a structure the language has is destroyed early by an unconditional
pass, then a later stage spends real work partially reconstructing it.* A
`Select` **is** an if (CLAUDE.md, "Select contains an if"). The schedule is
flattened, the branch structure is thrown away, and `cluster_select_arms`
spends 60% of a compile searching for contiguous skippable runs to get it back.

A compiler whose front end hands the backend real control flow never does this.
Ours has to, because it discards its own.

## 1. Two nodes, one value

```
Select(m, a, b)   soft — a blend. Both arms evaluated, bitwise select.
Guard (m, a, b)   hard — a branch. Only the taken arm's body runs.
```

**They denote the same function.** That is the whole point: being equal, they
belong in one e-class, and choosing between them is extraction's job rather
than codegen's. `Select`'s value semantics do not change — the demand plan's
constraint stands, *"demand only decides what is computed, never what is
selected."*

The distinction is **which lowering**, not **what value**.

## 2. A guarded arm is a `Ref`, and that is what makes the emitter dumb

A branch may only skip what its arm **exclusively owns**. Today that is
`closed_exclusive`: a fixpoint over consumers, because in a flattened schedule
an arm's extent is implicit and has to be recovered.

Make the arm a `Ref(KernelKey)` and its extent *is* its body — exactly the
named kernel, nothing else. A value shared with the world outside the arm lives
outside it, and the arm holds a reference to it. **Exclusivity stops being a
property to infer and becomes a property of the representation.**

This is what D7 means by making H5's partition *unsayable rather than merely
cheap*, and it is why this composes with N1 rather than competing with it.

Note the `Ref` here is a **naming device for extent, not a call**. A guarded arm
is inlined at its site as a region; it does not need L5's calling convention.
L5 remains a separate question about a `Ref` that survives *unguarded*.

### The arms are fields, not children (JP, 2026-09-12)

> *"Do we have inline kernel refs vs reference kernel refs? We should do the
> same thing here as hard vs soft select."*

**We do not** — and that is a trap this design has to step around rather than
build on. `passes::expand_refs` runs **unconditionally in every compile entry
point** (`jit_cache.rs`), so no `Ref` ever reaches codegen. Inline-versus-by-
reference is not a choice today; making it one is L4, and L5 is what a
survivor would mean. Both are pending.

So if a `Guard`'s arms were `ExprNode::Ref` *children*, `expand_refs` would
splice them in before codegen and destroy the very boundary this rests on.
The arms are therefore **fields**:

```rust
ExprNode::Guard { mask: ExprId, on: KernelKey, off: KernelKey }
```

One child — the mask — and two names. `expand_refs` rewrites `ExprNode::Ref`
and nothing else, so it cannot reach them; codegen expands them itself, as
regions. It also makes a non-`Ref` arm **unrepresentable** rather than
forbidden by a comment, which is the same move `Fold` made when it stopped
encoding an `OpKind` as a `Const(f32)`.

This also names the pattern properly. `expand_refs` inlines every `Ref`,
`ExpandReduce` unrolls every fold, and `guards` reconstructs every branch:
**three unconditional passes that destroy structure, and three later stages
that pay to rebuild it.** That is N5, and it is the backlog's opening
sentence. This plan removes the third; L4 and 2c remove the others.

## 3. What each consumer becomes

**The emitter** reads the node and emits:

- `Select` → the blend it already emits.
- `Guard` → a mask test, a branch to a label, the arm's body, the join. R0
  already gave it labels, so there is no new mechanism.

No partition, no closure, no cone, no demand. Most of `guards.rs` deletes.

**The allocator** still may not let a live range span a branch that might not
have run a definition — that constraint is real and does not go away. But it
**reads** the regions off the structure (`Guard`'s arms are its scopes) instead
of calling `analyze_select_guards`. 2b already made the nest a tree with
`Scope::Fold`; a guarded arm is the same shape of scope with a different
binder.

**Extraction** gains the decision it should always have had.

## 4. Who decides

Extraction, priced as:

```
soft = cost(a) + cost(b) + blend
hard = test + branch + P·cost(a) + (1−P)·cost(b) + (1−coherence)·MISPREDICT
```

`MISPREDICT_PENALTY_CYCLES` stays the single analytic profitability bound, with
its existing derivation — no tuned constant stands in for it.

`P` — how coherent a mask is across a batch — is **not a static property of the
graph**, so it is the part a learned model would supply.

**It supplies it as a term in the cost function, not as a stage after
extraction** (JP, 2026-09-12):

> *"Decisions between equivalent forms will be made at extraction. The whole
> reranker concept is bringing traditional passes where they don't belong."*

This supersedes the framing an earlier draft of this document used. A
`Reranker` re-orders candidates *after* the DP has chosen — which is a second
pass bolted onto the first, and reintroduces exactly the extract-then-repair
shape this plan exists to delete. The choice between two equal forms is the
cost function's, evaluated inside the DP, once.

The consequence is worth stating plainly: **this plan does not use the
`Reranker` seam**, and the seam's own justification (CLAUDE.md, "Cost Model and
the Guide") should be revisited rather than quietly relied on. Until a learned
term exists, a static coherence prior stands in, and it must be documented as a
placeholder rather than as a bound.

## 5. The question this withdraws

An earlier draft of this work asked whether the e-graph needs **consumer
edges**, so demand could be tracked incrementally from consumer to producer.

**It does not, and the question is withdrawn.** That design was needed only to
*infer* which values an arm owns. Once the graph carries `Guard` with `Ref`
arms, nothing is inferred — the graph **declares** it. Demand analysis stops
being load-bearing, and the e-graph keeps its shape.

## 6. Stages

| | | gate |
|---|---|---|
| **G1** | `Guard` node, `Ref` arms, constructible but never chosen | byte-identical: extraction always picks soft |
| **G2** | emitter emits `Guard`; allocator reads regions off structure; delete the analysis | byte-identical where soft is still chosen, plus a hand-built `Guard` that emits and runs |
| **G3** | extraction may pick hard, priced by the table + `MISPREDICT` | behaviour: goldens, ISA matrix, and the bake measurement |
| **G4** | the learned residual for coherence | measured against G3, not asserted |

G1 and G2 are byte-identity gated, which is the same gate that carried R0 and
the three guard fixes. G3 is the first stage that may move a byte, and it is
the first that can pay.

## 7. Constraints

- **`Select` stays a blend.** Its value semantics do not change.
- **`Guard` and `Select` are equal**, and must be provably so — the e-graph may
  rewrite either into the other without a cost argument.
- **`MISPREDICT_PENALTY_CYCLES` stays the one bound.** A learned term may
  rerank; it may not replace the analytic floor.
- **The emitter gains no analysis.** If a stage needs one, it belongs upstream.
  Hash-consing and other free-at-the-point-of-use folds are fine; a search is
  not.
- **The frame prologue is never guarded.** It runs once per call.

## 8. Open, and not for me to settle

1. ~~Does a `Guard` arm always become a `Ref`, or only when it is large enough
   to pay?~~ **Settled (JP, 2026-09-12): every arm is a `Ref`.** No threshold,
   so no tuned constant, and §2's unrepresentability argument holds for every
   guard rather than most of them. `KernelStore` pressure is a consequence to
   measure, not a reason to special-case.
2. **The `'8'` waist bug (C2) lives in this area** and is open on `main`. G3
   changes what is computed under a mask, so it may move that bug in either
   direction. It should be measured across G3 rather than discovered later.
