# A surviving Reduce is a loop

## Metadata
- **Author**: JP (direction), Claude (draft)
- **Status**: `Plan of record`
- **Created**: 2026-09-10
- **Verified against**: `01f3d00`
- **Executes**: the `Reduce` row of
  [a-kept-structure-is-control-flow](2026-09-10-a-kept-structure-is-control-flow.md) §4.

**Decision it records (JP, 2026-09-10):**

> *"Let's start tearing all the shit that made it into the emitter back up to
> the egraph. … So now we can emit reduces that survive the graph as regular
> ass loops?"*

---

## 1. Why, with the number

Compiling one glyph at 32 px takes **2.1 s**, and 99.8% of a bake is compile.
The reason is not that any pass is slow: it is that the emitter is handed
**34,993 straight-line instructions for one glyph**.

Where they come from, measured (`docs/BACKLOG.md`, X1):

- a glyph is a fold over N pieces with a ~250-node body;
- `passes::expand_reduce` unrolls it **unconditionally** into N copies;
- `expand_gather` and `expand_transcendentals` then multiply each copy ~4×
  (post-optimize arena 8,548 nodes → schedule 34,993);
- and `emit` is **~O(n^1.6)** across the three measured points (6,055→133 ms,
  17,521→673 ms, 34,993→2,088 ms; the exponent fits 1.53 and 1.63 on the two
  intervals).

A loop makes the emitted program the **body** — ~1,030 schedule entries
instead of 34 copies of it. Extrapolating the same exponent well below its
measured range, which is why this is an order of magnitude and not a number:
**~9 ms** against 2,088 ms.

It also removes the reason S3 ("one program for the font") needed padding at
all. Padding exists only because a fold's trip count is compile-time, so 34
pieces and 11 pieces are different kernels; §5 records what it would take for
the count to stop being part of the key.

## 2. What is already true

- **Extraction can keep a fold.** The `usize::MAX / 4` sentinel on
  `ENode::Reduce` was removed in `e6b3990`; it is priced at
  `(len − 1) × cost(combiner)`. `arena_to_schedule`'s panic comment still says
  a surviving `Reduce` "is priced out of extraction rather than emitted" —
  that clause is stale.
- **Legalization runs last**, so a fold reaches extraction folded. The pipeline
  does not need reordering; only `ExpandReduce`'s unconditionality is in the
  way.
- **Codegen already emits counted loops.** `IsaBackend::emit_collapse_loop`
  builds the X/Y nest: `counter_clear` → top → `branch_if_counter_done` → body
  → `counter_step` → `emit_jump` → `patch_branch`. Every primitive a reduce
  loop needs exists and is per-ISA already.

## 3. The design decision that makes this tractable

**The accumulator lives in a slot, not a register.**

`emit_collapse_loop` keeps its loop-carried value — the coordinate — in a
frame slot: `slot_load` at the top of each iteration, `slot_store` at the
bottom. So **no value is live across the back edge**, and `LinearScan`, which
is straight-line over a flat schedule and has no CFG, never has to represent
one.

That is the whole reason this does not need the allocator work L5 names for a
call. It costs one load and one store per iteration — against a body of ~1,000
instructions, nothing.

The same trick supplies the binder: the body reads `Var(4..8)` as an `f32`
index, and the loop broadcasts its counter into that register at the top of
each iteration, exactly as the collapse nest reloads `coord_reg(k)`.

**Do not "improve" this to keep the accumulator in a register** without first
giving the allocator live ranges with holes. That is the same boundary L5
names, and it is not needed here.

## 4. Stages

**R0 — a position gets a name.** *Landed.* Codegen had no way to *say* "here",
so every branch was placed as a fixup token the emitter carried by hand to a
`patch_branch` call, against an offset it read off `code.len()` at the one
point in the sequence where that was correct. A loop's back edge is the same
shape, so building one meant more of that.

A program is a sequence of **items**: an instruction (fixed bytes, position-
independent — which is why a branch could never be an `AsmInsn`), a **binding**
of a name to this position (no bytes), or a **reference** to a name (bytes that
depend on where it lands). `Label`, `LabelScope`, `Item`, `AsmBranch`,
`Resolver`, `assemble_labeled` in `pixelflow-codegen/src/emit/mod.rs`; per-ISA
`Branch` enums named by the condition they test, never by an opcode byte.

`AsmBranch::resolve` takes `self` because the field is the *branch's*: x86
spells every displacement `rel32`, aarch64's `B` carries imm26 and its
`B.cond`/`CBZ` an imm19 five bits up. Two front ends, one mechanism —
`assemble_labeled` for a program that can be a value, `Resolver` streaming for
an emitter whose every verb is a `&mut self` method and so cannot build that
sequence at all.

Both loops in the emitter moved: the collapse nest and the `Select`
short-circuit. That deleted `emit_jump`, `patch_branch`, `emit_skip_if_all_false`,
`emit_skip_if_all_true`, `IsaBackend::Branch`, `Aarch64Branch`, `Cond19`,
`Rel26`, `emit_jmp_rel32` and `patch_rel32` — the whole fixup-token mechanism,
five impls of it. The two skip verbs folded into one `compare_mask(.., arm)`,
because they differed only in which uniform mask lets an arm go, which is what
`SelectArm` already names. Emitted bytes are unchanged, which is what the
goldens are for.

**R1 — emit one.** `arena_to_schedule` grows a `Reduce` arm; the backend grows
a reduce-loop scaffold beside `emit_collapse_loop`; `ExpandReduce` gains a way
to be told not to unroll. The back edge is now `labels.branch(B::JUMP, top,
&mut code)` in the same `Resolver` the guards use, so the loop is composition
rather than a third copy of the fixup dance. Gate: a hand-built `⊕_{[0,n)}`
over a table produces the same buffer looped as unrolled, on both ISAs, for
`SUM` and `MIN`.

*What R1 looks like in this tree, read off the code rather than guessed at
(2026-09-10):*

- **The fold's region is `partition_by_scope`, with no new pass.** §4 of
  [a-kept-structure-is-control-flow](2026-09-10-a-kept-structure-is-control-flow.md)
  guessed `&[0, 1, 4]`; the binder list is innermost-first (`COLLAPSE_BINDERS
  = [0, 1]` is X then Y, and `plan_collapse_hoist` is called with the mask of
  `binders[..=j]` from the outside in), so the fold binder goes at the *front*:
  `&[4, 0, 1]`. Then `plan.body` is exactly the values that vary with the
  binder, which is exactly the loop body — and the `Reduce` node itself does
  not vary with the binder it binds, so it lands outside the loop, which is
  where an accumulator's final read belongs.
- **Variance, not reachability, is the right criterion**, and it is already
  computed. Nothing outside the fold can read the binder, so a body node
  shared with the outer graph is loop-invariant by construction — the sharing
  question answers itself.
- **`plan_collapse_hoist`'s gather refusal does not block this.** It excludes
  gather-bearing values from the *hoisted* set, so the cost is invariant table
  reads staying in the loop — a missed optimization (ask B), not a wrong
  split.
- **A fold region is shaped like a guard region**, which is the part that
  makes this tractable: `select_guards` is already a side table of regions
  with `branch_starts[sched_idx]`/`branch_ends[sched_idx]`, walked in schedule
  order and bound as the walk passes. A fold differs in two ways only — the
  branch is a back edge, and there is an accumulator.
- **The loop's own state can be one slot, not two.** The binder is broadcast
  across lanes (the body reads it as `Var(4)`), `add_scalar(reg, scratch,
  1.0)` already steps a broadcast vector by one, and a `u16` trip count is
  exact in `f32` — so the binder *is* the counter, and the termination test is
  a scalar compare of its low lane against `hi`. That needs one new backend
  verb (`ucomiss`+`jae` on x86, `fcmp`+`b.hs` on aarch64 — with no NaN in
  range, `HS` after `FCMP` is `>=`, so both reuse `Branch::IfAboveOrEqual`).
  No new GPR, and so no callee-saved push in the prologue.
- **What is genuinely new**: `ScheduledOp::Reduce` and its `resolve_operands`
  case, the accumulate step, and a third nesting level in a scaffold that is
  hardcoded two deep.

**R2 — the cost model prices a loop.** Today `node_op_cost` for a `Reduce` is
the *unrolled* cost, so extraction has no reason to keep one. A loop costs
`len × (body + step + branch)` in time but **`body + scaffold` in code size**,
and code size is what compile time is superlinear in. This is the loop-unroller
question JP named: how big is the body, how much does it explode, what is the
gain. Gate: extraction keeps the fold for a glyph and unrolls a 2-term one.

**R3 — `ExpandReduce` becomes a fallback.** It unrolls what survived only when
codegen cannot take it, which after R1 is nothing. Gate: the glyph suites and
goldens unmoved; `emit` wall clock on `8`@32 falls by the order §1 predicts, or
the prediction is wrong and this document says so.

**R4 — partial unrolling** (backlog **E1**): geometric `SplitFold`,
`⊕_{[lo,hi)} = ⊕_{[lo,mid)} ⊕ ⊕_{[mid,hi)}`, bisecting. Needs no substitution
(both halves share the body e-class) and is exactly cost-preserving in both
extraction arms. Inert before R2 — split and unsplit price identically under
`latency_prior()` — which is why it is last and not first.

## 5. What this does not do

- **The trip count stays compile-time.** `Fold` holds `lo: u16, hi: u16`, and
  `key.rs` encodes it: *"Two folds over the same body under different
  algebras, binders or ranges are different kernels."* So a 34-piece glyph and
  an 11-piece glyph are still two compiles. Making the count a **uniform** is
  what would collapse the font to one program with no padding, and it is a
  separate change: the key must stop carrying the range, and the loop must
  compare against a loaded value rather than an immediate.
- **No nested reduce loops.** One binder at a time until R1's gate is green on
  a single level.
- **No change to `Select`, `Ref`, or the fold's denotation.** A fold means what
  it meant; this is about what codegen does with one that survives.
