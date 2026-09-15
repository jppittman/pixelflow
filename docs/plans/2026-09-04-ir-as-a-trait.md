# `Ir` as a trait: naming the term language the e-graph speaks

## 0. The inversion

An optimizer is an endomorphism on the IR. The type named `Optimizer`
(`pixelflow-search/src/egraph/optimizer.rs:263`) is not one — it is the
*policy* of one step: rule set, budget, cost model, extractor, guide. The
endomorphism itself (insert → saturate → extract → materialise) has no name
and no type, so each tier assembles it by hand.

That is why "skip optimization" cannot be spelled as an identity optimizer:
there is no type for it to inhabit. `kernel_raw!` is a branch that declines to
call a function (`pixelflow-compiler/src/lib.rs:186`), not a value.

The plan that introduced `Optimizer` specified
`run(&ExprArena, root) -> Optimized { arena, root }` — the endomorphism —
and what shipped was `run(&mut EGraph, root) -> choices`
(`docs/plans/2026-09-02-optimizer-api.md:606` vs `optimizer.rs:591`). The
retreat is unrecorded in §6.2. Its cause is stated at `optimizer.rs:257-262`:
the three tiers insert their terms differently, so no single input type fit.

This document names what actually differs, which is smaller than "the IR".

## 1. The denotation

> An **IR** is a term language that can be destructured into, and rebuilt
> from, the signature the e-graph speaks.

Two operations, mutually inverse up to the equalities saturation adds:

- `project`: one node, children already resolved to references. The unfold.
- `embed`: build one node from already-built children. The fold.

`Shape<R>` is the signature — the pattern functor of `ExprNode`
(`pixelflow-ir/src/arena.rs:94`) with `R` for `ExprId` and `BufferDecl` for
`BufferId`, because e-classes outlive any one arena (the reason
`ENode::Buffer` already carries the full decl, `egraph/node.rs:30-38`).

**This trait is worth having with exactly one implementor.** It names the API
the e-graph is entitled to use, which is the API the tests should be calling.
Today no such boundary exists, and the consequence is measurable — see §2.

## 2. What the missing boundary already cost

Two arena→e-graph inserts exist, for one IR, and they disagree:

| | `EGraph::add_arena` (`graph.rs:1036`) | `arena_to_egraph` (`runtime.rs:432`) |
|---|---|---|
| op with no static `Op` | **panics** | declines (`None`) |
| `Param` | **panics** | declines |
| `IAdd` / `Shl` / `Shr` | **panics** — absent from `op_from_kind` | handled (`runtime_op_from_kind:415`) |
| `BitAnd` / `BitOr` / `TruncToInt` / `IntToFloat` | **panics** | handled (`:411-414`) |
| `Gather` | special-cased inline (`:1122-1127`) | via the resolver (`:418`) |
| `Nary` | inserted | declines (`:490`) |
| traversal | **every node** in the arena, `0..n` | reachable from `root` only |

The last row is a live defect. The `Dwrt` tier picks its saturation budget
from `reachable_node_count` (`ir_bridge.rs:737`) but inserts via `add_arena`,
which walks `0..n`. Unreachable construction garbage therefore consumes
`max_classes` — a budget dimension — that the budget never counted. The class
cap binds earlier than the tier's own preset intends, and nothing reports it.

The op rows are two notions wearing one name:

- **an op a rewrite template may name** — `ops::op_from_kind`. `Gather` is
  deliberately absent so no rewrite can match it.
- **an op the graph may hold** — a superset, including opaque ones whose only
  participation is hash-consing CSE.

The second had no name, so `runtime.rs` grew a private `runtime_op_from_kind`
and `add_arena` hand-rolled a partial copy that omits five kinds.

It is **not** one global superset, and that is the important part. The mask
and integer-domain ops are sound in a runtime-tier graph and *unsound* in a
macro-tier one: that e-graph runs before composition, where resolving the
`Dwrt` nodes they travel with is wrong — a leaf's `DX` is 1 only until an
enclosing `.at()` warp scales it, and the fonts' density-dependent AA ramp
broke exactly this way when these ops were briefly global (`runtime.rs:316`).

So the holdable set is per-tier, and it was previously encoded in *which
private helper a call site happened to call* — a convention, checked by
nothing, which is how the two helpers drifted apart on five kinds. It becomes
`ops::Vocabulary { Templates, Runtime }`: an argument at each insertion, with
the soundness rule as a test rather than a comment.

## 3. Surface

```rust
// pixelflow-ir/src/term.rs
pub enum Children<'a, R> { Zero, One(R), Two(R, R), Three(R, R, R), Many(&'a [R]) }

pub enum Shape<'a, R> {
    Var(u8),
    Const(f32),
    Param(u8),
    Buffer(BufferDecl),
    Op(OpKind, Children<'a, R>),
}

pub trait Ir {
    type Ref: Copy + Eq + core::hash::Hash;
    fn project(&self, r: Self::Ref) -> Shape<'_, Self::Ref>;
    fn embed(&mut self, shape: Shape<'_, Self::Ref>) -> Self::Ref;
}
```

`Children` mirrors the existing `ExprChildren` (`arena.rs:116`) so the
projection of an inline-child node borrows nothing it must not.

Not in the trait, each for a reason:

- **No cost.** `CostModel` is over `OpKind`, not over the term language.
- **No evaluation, no codegen.** Those are the backend's, and they are where
  the tiers genuinely differ.
- **No spans.** The AST has them and the arena does not; requiring them would
  make `ExprArena` unable to implement its own IR trait.
- **No `optimize`.** That is the endomorphism, a separate trait, and it is not
  in this change (§5).

## 4. What this change does

1. `pixelflow-ir/src/term.rs` — `Children`, `Shape`, `Ir`.
2. `impl Ir for ExprArena`. `embed` owns buffer-slot dedup by identity: one
   slot per distinct `BufferIdentity`, which is what "declare a buffer here"
   means for an arena and is currently open-coded inside extraction
   (`extract.rs:1256-1265`).
3. `ops::Vocabulary` — which ops a graph may hold, per tier, with the seven
   runtime-only op structs moved from `runtime.rs:330-394` into `ops.rs`
   beside it, comments included.
4. One generic `insert<I: Ir>(&I, I::Ref, &mut EGraph, Vocabulary)
   -> Result<EClassId, Declined>`, reachable-only, iterative, declining rather
   than panicking. It replaces both `add_arena` and `arena_to_egraph`.
5. `choices_to_arena` keeps its cycle detection, choice lookup and
   shift-count pinning — those are the e-graph's — and materialises through
   `Ir::embed` instead of an open-coded `ENode` match.
6. Deletions: `EGraph::add_arena`, `runtime::arena_to_egraph`,
   `runtime_op_from_kind`, and `ir_bridge`'s `representable` pre-check
   (`ir_bridge.rs:710-720`), which existed only to avoid `add_arena`'s panic
   and is subsumed by `insert` returning `Option`.

### Behaviour changes, deliberate

- Every tier stops inserting unreachable nodes. Strictly fewer e-classes for
  the same budget. `anytime.rs`'s assertion that callers pre-compact, and the
  `compact_subtree` helper in `examples/oracle_filtered_budget_curves.rs` (deleted),
  both existed only to work around this; the assertion goes.
- A zero-arity `Op` now panics in `embed` where `choices_to_arena` silently
  materialised the constant `0.0` — a wrong answer wearing a right answer's
  clothes, and unreachable for any well-formed graph.
- Nothing that optimized before declines now, and nothing that declined before
  now panics.

### Equivalence, measured

`optimizer_equivalence` over the twelve `shader_bench` kernels, release build,
digesting the extracted arena — the same harness
docs/plans/2026-09-02-optimizer-api.md §6.3 used, and a stronger check than
cost (equal arenas have equal cost under every model; equal costs can hide a
different term).

| arm | combined digest |
|---|---|
| base `b3b3895` | `66efbe1a7133c5f4` |
| this change | `66efbe1a7133c5f4` |

All twelve match individually — same input size, same extracted node count,
same digest, kernel by kernel.

## 5. Not in this change

The `Optimize` endomorphism trait (`Identity` for `kernel_raw!`, `Saturate`
for the e-graph, composition with a folded fingerprint) is the point of the
exercise and is the next step. It needs this trait first: an endomorphism on
"the IR" is not writable while "the IR" is two hand-rolled conversions with
different failure modes.

The macro tier's AST↔e-graph path (`EGraphContext`, `optimize.rs:494-1221`)
is untouched here. Once `Optimize` exists, the subtraction is to route the
macro tier through `ir_bridge::ast_to_arena` — which already exists and is
already used by the JIT backend — and delete `EGraphContext` entirely, making
CLAUDE.md's "`ExprArena` is the sole IR" true rather than aspirational.
