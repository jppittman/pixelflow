# A kept structure is control flow

## Metadata
- **Author**: JP (direction), Claude (draft)
- **Status**: `Draft`
- **Created**: 2026-09-10
- **Verified against**: `4408d89`
- **Companion to**: [one-name-bound-later](2026-09-10-one-name-bound-later.md).
  That one is about names and when they are bound; this one is about
  structures and when they are flattened. They are the same observation on
  the two halves of a function: its parameters and its control flow.

**Decision it records (JP, 2026-09-10):**

> *"our ability to hoist out of range loops should be the same machinery as
> our ability to hoist out of folds... out of any kind of loop is the same
> machinery as needing to hoist out of the loops that we have for iterating
> over the scene or the lines. We really have only a handful of things going
> on. You know, we have variables and bindings with no higher level
> abstraction, and we have loops and hoisting with no higher level
> abstraction and shared machinery around those."*

---

## 1. The hoist is already one mechanism

This part is **done**, and it is worth writing down because it looks undone.
`pixelflow-codegen/src/emit/mod.rs`:

```rust
/// Split a schedule by scope over `binders`, given innermost first.
///
/// One rule, applied once per binder from the outside in: a value is lifted
/// out of a binder when its variance does not name that binder or any binder
/// inside it. That is loop-invariant code motion, hoisting out of a
/// reduction, and constant folding — the same question asked at each level,
/// which is why this is a loop over binders rather than a tier per scope.
fn partition_by_scope(
    schedule: Vec<regalloc::Def>,
    variance: &[pixelflow_ir::variance::Variance],
    binders: &[u8],
) -> regalloc::ScopedSchedule
```

Generic over an arbitrary binder list. `Variance` already carries a bit per
`Var(0..8)`, with bits 4..8 reserved for the four reduction binder slots. So
"hoist out of a fold" needs no new machinery at all — it needs a longer array.

## 2. It is handed a two-element array, and cannot be handed a longer one

The single production call site:

```rust
const COLLAPSE_BINDERS: [u8; 2] = [0, 1];
let mut scoped = partition_by_scope(schedule, &variance, &COLLAPSE_BINDERS);
```

X and Y. Never a reduction binder — and not because the hoist refuses one.
**There is no fold binder at schedule time.** `ExpandReduce` is the last pass
of the runtime pipeline (`pixelflow-search/src/runtime.rs`) and it runs
unconditionally, so every fold is flattened into a combiner chain before
codegen sees it. Codegen contains **zero** `OpKind::Reduce` cases; CLAUDE.md
states the invariant directly — *"The language is a DAG: no iteration binder.
A fixed-count iteration is unrolled at construction."*

So the shared machinery exists and is unreachable, and the reason is one
level up.

## 3. The decision that does not exist

JP, on the fold rules, 2026-09-09:

> *"this is our loop unroller. and it needs to run on loop unroller logic of
> how big is the loop, how much is it gonna explode the code, and what is the
> performance gain from it."*

A loop unroller answers a question. This one cannot: **there is only one
answer available.** Unroll. Extraction may not keep a fold, because codegen
cannot emit one, so the cost model has nothing to choose between and the
"how big / how much / what gain" reasoning has no place to live.

That is the same shape as the `Ref` story. L5 exists because extraction *may*
keep a `Ref` and codegen has no way to emit one, so today the linker always
inlines. Same sentence, different structure.

## 4. The generalisation

**Extraction chooses a form. Codegen emits control flow for whatever survives.**
Three structures, one rule:

| survives extraction | codegen emits | today |
|---|---|---|
| `Ref(k)` | a **call** | never survives — `expand_refs` always inlines (L5) |
| `Reduce { fold }` | a **loop** | never survives — `ExpandReduce` always unrolls |
| `Select(m, a, b)` with a derived range | a **domain split** | never derived — no mask ⟹ range (D1/D2) |

In all three the flattened form is the *fallback*, and today it is the only
form. Each has a real cost on the other side — a call boundary the register
allocator does not model, a loop's induction variable and trip count, a
second emitted program for the complement — which is exactly why the choice
belongs to the cost model rather than to a pass that always says the same
thing.

Two consequences fall out immediately:

- **`partition_by_scope(schedule, variance, &[0, 1, 4])` is the whole of ask
  B**, once a fold can survive. The gradient magnitude in
  `pixelflow-graphics/src/fonts/loop_blinn.rs`'s `Distance::in_pixels` depends
  on the fold binder and not on X or Y, so it lifts into the per-term scope by
  the rule already written. No `contains_gather` relaxation, no new pass.
- **Legalization's position is already right.** `ExpandReduce` running last is
  what makes "keep the fold" *representable* through saturation and
  extraction; it is only the unconditional part that forecloses the choice.
  The pipeline does not need reordering — it needs the last pass to be a
  fallback in fact as well as in name.

## 5. What this costs, honestly

An emitted loop is not free and this document should not pretend otherwise.

- **Trip count and induction.** A fold's range is a compile-time constant, so
  the loop is counted, not conditional. That is the easy case.
- **Register allocation across a back edge.** The allocator is linear-scan
  over a straight-line schedule (`RegisterAllocator`/`LinearScan`). A loop
  body's live ranges wrap, which linear scan handles badly without a loop
  model. **This is the real cost**, and it is the same boundary L5 names for
  a call.
- **The accumulator.** A fold's accumulator is loop-carried, which is the one
  genuinely new liveness shape.
- **It competes with the unroll.** An unrolled chain gets CSE and reassociation
  across the copies; a loop gets none of that. For small trip counts the
  unroll is simply better, which is the answer a cost model should be allowed
  to give.

So the order is: make it *representable* first and measure the loop against
the unroll on a real glyph, before letting the cost model choose. A rule that
can only be exercised by trusting an unvalidated cost term is not a rule.

## 6. Relation to E1

The geometric `SplitFold` (`⊕_{[lo,hi)} = ⊕_{[lo,mid)} ⊕ ⊕_{[mid,hi)}`,
backlog E1) is the *partial* answer to the same question, and it is available
without any of §5's costs: it needs no substitution, is exactly
cost-preserving in both extraction arms, and expresses "unroll a chunk, keep
the rest folded" — which is precisely what a loop unroller does. It is worth
landing first for that reason, and it does not compete with this document; a
split fold whose halves both survive is two loops.
