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

~~**The accumulator lives in a slot, not a register.**~~ **Superseded
2026-09-10 — see §3a.** The original claim was that the accumulator should be
pinned to a frame slot, loaded at the top of each iteration and stored at the
bottom, so that no value is live across the back edge and `LinearScan` never
has to represent one. It costs one load and one store per iteration, which
against a ~1,000-instruction body is nothing, and the same trick was to supply
the binder.

## 3a. Correction: placement is the allocator's job (JP, 2026-09-10)

> *"Register allocation is the job of the register allocator. I want to
> eliminate fixed, single case oob, bespoke allocations."*

§3 is a **bespoke allocation** wearing a performance argument. It decides,
outside the allocator and for one value, that the accumulator is in memory —
which is precisely the class of thing step 3 exists to delete, alongside the
X/Y precoloring, `SCAFFOLD_ACC`, and the `Counter` registers.

The accumulator is a **value**. It has a live range that spans the loop, and
the question "register or slot" is the allocator's, asked and answered the
same way for every other value that crosses a scope boundary: `allocate_nest`
already ranks a scope's roots by use count against the pool budget and parks
each one in a carried register or a slot accordingly. A fold's accumulator
goes through that path or the path is wrong for everything else too.

Which is the argument *for* §4b's shape rather than a complication of it. A
fold bracketed inside the body scope has no park mechanism available and needs
a bespoke answer; a fold that is a scope of the nest has the general one
already. The pinned slot was never a design decision — it was the workaround
the wrong shape forced, and it disappears with the shape.

What §3 got right and keeps: **nothing here needs live ranges with holes.**
That was the stated reason for pinning, and it survives without the pinning,
because a park is a whole-scope answer rather than a hole in one.

## 4. Stages

**R0 — a position gets a name.** *Landed.* Codegen had no way to *say* "here",
so every branch was placed as a fixup token the emitter carried by hand to a
`patch_branch` call, against an offset it read off `code.len()` at the one point
in the sequence where that was correct. A loop's back edge is the same shape, so
building one meant more of that.

**The assembler is an assembler.** A program is a flat sequence of `Item`s:
either an instruction, or a **label** bound to this position. Not an AST —
assembly is not context-sensitive and there is nothing to nest.

- **A label is an item, not a field on an instruction.** A label names a
  *position*, and positions are not owned by instructions: a loop's exit label
  sits past the last body instruction, where there is nothing to hang it on, and
  two labels may name the same position. Both are free when the label is its own
  item; both need a dummy `nop` or a `Vec<Label>` per instruction otherwise.
- **A label *reference* is an operand.** `Jcc::je(exit)` is `je exit` — the label
  is that instruction's argument, like a register or an immediate. So positions
  are items and references are fields; two relationships, two spellings.
- **A branch is an ordinary instruction.** `struct Jmp { target }`,
  `struct Jcc { condition, target }` on x86; `B`, `BCond`, `CbzW16` on aarch64 —
  each an `AsmInsn` like any other, wrapped by the backend's `Inst` enum exactly
  as `MovLoadPtr` already was. `AsmInsn::label_ref` is the one method that was
  added: it says which label the instruction is waiting on and how to fill in
  the displacement. That is the whole of what a branch adds.
- **The condition is the opcode's own field, and all sixteen exist.** x86 `jcc`
  is `0F 8x rel32` and A64 `B.cond` is `0101 0100 imm19 0 cond`; in both the
  condition *is* a nibble of the encoding, so `Cond` is a 16-variant
  `#[repr(u8)]` enum whose discriminants are the manual's values, and encoding
  is `0x80 | condition as u8`. It replaced a private `jcc(code, cc: u8)` with
  three hand-picked mnemonics and a doc comment defending the magic byte.
  `Jcc::je` / `BCond::hs` and friends are `const fn` sugar over the one encoder,
  so a call site still reads like assembly without sixteen types that differ by
  a constant.
- **Where an instruction landed is the assembler's bookkeeping.** The branch
  emitters used to return a position; they return nothing now, because the
  assembler wrote down `code.len()` before calling `emit_into`.

Two front ends, one mechanism: `AsmProgram::from([...]).assemble(code)` for a
program that is a value, and `Assembly` — push, bind, finish — for the emitter,
which discovers its instructions while walking a schedule and so cannot hand
over a finished list. Both keep one label map and both resolve the same way.

Both loops in the emitter moved onto it: the collapse nest and the `Select`
short-circuit. That deleted `emit_jump`, `patch_branch`,
`emit_skip_if_all_false`, `emit_skip_if_all_true`, `IsaBackend::Branch`,
`Aarch64Branch`, `Cond19`, `Rel26`, `Rel32`, `emit_jmp_rel32` and `patch_rel32`
— the whole fixup-token mechanism, five impls of it. The two skip verbs folded
into one `branch_if_arm_is_dead(.., MaskTest, Label)`, because they differed
only in which uniform mask lets an arm go, which is what `SelectArm` already
names. Emitted bytes are unchanged, which is what the goldens are for.

**R1 — emit one.** `arena_to_schedule` grows a `Reduce` arm; the backend grows
a reduce-loop scaffold beside `emit_collapse_loop`; `ExpandReduce` gains a way
to be told not to unroll. The back edge is `asm.push(Jmp { target: top })` in
the same `Assembly` the guards use, so the loop is composition rather than a
third copy of the fixup dance. `IsaBackend::loop_open`/`loop_close` already
exist for it: `emit_loop` takes a closure, which a nest wants, and the
open/close pair is the same loop for a *linear walk*, which is what a schedule
walk is. Gate: a hand-built `⊕_{[0,n)}`
over a table produces the same buffer looped as unrolled, on both ISAs, for
`SUM` and `MIN`.

*What R1 looks like in this tree, read off the code rather than guessed at
(2026-09-10):*

**The loop-carried accumulator is a phi, and the park is how you spell one.**
`LinearScan` is straight-line over a flat schedule and has no phis. This was
written as "which is why §3 puts the accumulator in a slot"; §3a retracts the
pinning — a park is the general spelling, and the allocator chooses between a
carried register and a slot the way it already does for every root. What
stands is that this needs *no new machinery at all*: it is three ordinary
schedule defs.

```text
acc_init = Const(identity)                    // before the loop
  <body defs — the ones that vary with the binder>
acc_next = Binary(monoid_op, acc_init, body_root)
```

- `acc_init` and `acc_next` are the **same park**: the fold scope's live-in and
  its live-out are one place, and that aliasing *is* the phi — the bottom of
  the iteration writes where the top reads. Which place is the allocator's
  call, not this plan's (§3a): a carried register when the budget affords one,
  a slot when it does not, ranked by use count like every other root.
- If the park is a slot, `acc_next` reaching it is already automatic.
  `store_after_def[i]` is set for any value that has a slot and whose def-point
  binding is a register — *"every definition writes a register, so this is the
  only place a value reaches its slot"* — so the emitter stores it with no new
  verb. If the park is a register, there is nothing to store.
- `acc_init` inside the loop, and the binder's `Var(4..8)` def, are read from
  the park by the ordinary machinery. That is exactly what `HoistCtx::Body`
  already does for a parked value: *"mapped values are never emitted; their
  locations are overridden — to a carried register where the allocator found
  one, and otherwise to the hoist slot, where every consumer reloads through
  the ordinary spill machinery."* Both halves of that sentence apply here; the
  retracted §3 only ever used the second.

So the accumulate is an **ordinary `Binary` def**, not a loop verb. No scratch
register has to be reserved for it, no `combine` verb is added to `IsaBackend`,
and the allocator learns nothing about folds. That was the part that looked
expensive and is not.

**The region is a span, and a fold region is shaped like a guard region.**
`select_guards` is already a side table of regions with
`branch_starts[sched_idx]`/`branch_ends[sched_idx]`, walked in schedule order
and bound as the walk passes; `IsaBackend::loop_open`/`loop_close` are the same
loop as `emit_loop` for a walk that cannot nest closures. The span runs from the
first def whose `Variance` includes the binder through `acc_next`.

Variance, not reachability, decides it, and it is already computed: nothing
outside a fold can read its binder, so a body node shared with the outer graph
is loop-invariant by construction. A loop-invariant node that happens to sit
*inside* the span is recomputed per iteration — correct, and merely the
optimization ask B is about, not a wrong split.

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

## 4a. Correction, 2026-09-10: a fold has to be a *scope*

§3 says the accumulator in a slot means "no value is live across the back edge,
and `LinearScan` never has to represent one." **That is true of the accumulator
and the binder, and false of everything else in the body.**

`LinearScan` is straight-line: it evicts a value and reloads it at a later
index, on the assumption that each index executes once. A back edge breaks that
assumption for *any* value the allocator relocated inside the loop — the reload
was emitted once, at a point the second iteration reaches with the register
already reused.

Codegen already has the answer, and it is per **scope**, not per value:

> The scope's head, where the previous iteration's tail flows back in. A value
> live across this scope's back edge may end an iteration somewhere other than
> where the next one expects to find it; this is what puts it back, once per
> iteration — the cost the eviction that moved it was charged.

That reconciliation runs at a scope's head, and `allocate_nest` gives one to
each region of `ScopedSchedule`. A fold emitted *inside* the body scope has a
back edge that no scope owns, so nothing reconciles it.

**So the fold's loop must be a region of the same nest as X and Y**, not a
bracket the emitter draws inside the body. Which is exactly the unification JP
asked for — *"collapse that distinction and parameterize them"* — and it turns
out not to be an aesthetic preference: it is what makes a fold's register
allocation correct. X, Y and a binder are three scopes of one nest, differing
in what steps them and what ends them, and
[loop-aware-codegen](2026-09-01-loop-aware-codegen.md) said so on 2026-09-02:
*"They are not a second anything."*

Measured on the way to this: with the fold bracketed inside the body scope,
`pixelflow-core` is green (62/62, nested folds included) and `pixelflow-graphics`
fails 2 of 89 — both *value* differences in glyph coverage, not crashes, which
is the signature of a register relocated across an unreconciled back edge rather
than of wrong arithmetic.

Two bugs found and fixed on the way, both real and both worth keeping whatever
shape the loop ends up:

- **Sibling folds may share a binder index.** The language has eight binders and
  nothing bounds a kernel to eight folds, so a region search that scans the
  whole prefix for "first def mentioning binder 4" finds the *previous* fold's
  body. It must stop at the previous `Reduce` over the same binder.
- **A nest opening at one index must be pushed outermost-first**, since it
  closes inside-out.

And one that is the same mistake twice: the loop cannot borrow `SCAFFOLD_ACC`
/`SCAFFOLD_SCRATCH`. Those are `Reg(0)`/`Reg(1)` — **X and Y** — and the
scaffold may clobber them only because it runs *between* iterations of the
collapse loop, where the next iteration reloads them. A fold runs inside the
body, where they are live. Its registers have to be reserved by the allocator,
as a guard's are.

## 4b. Correction to the correction: the nest is a *tree*, and a placement is *per scope*

§4a concludes "the fold's loop must be a region of the same nest as X and Y".
The diagnosis is right — a fold needs a scope, because reconciliation is per
scope — but "a region of the same nest" is wrong about what a region *is*.

`ScopedSchedule` is a **chain**, and a region is a **prologue**:

```text
regions[0]              once per call, parks roots
  loop y {
    regions[1]          once per row, parks roots
      loop x {
        body            once per sample, produces the result
      }
  }
```

Region *i* encloses region *i+1*; its roots flow **inward**, read by
everything inside. A fold is the other shape. It sits *inside* the body, and
its accumulator flows **outward** to the defs after it:

```text
body_pre                the defs the fold reads
  loop i { fold_body }  parks the accumulator
body_post               the defs that read it
```

There is no position in the chain for `body_post`. Adding a fourth link puts
the fold's loop *around* the code that consumes it, which is not what a fold
means. **The nest is a tree**, and the chain is the special case where every
scope has one child and it is last.

### What actually blocks the tree

Not the emitter — `emit_nest` already counts no levels. It is `Point`:

```rust
pub struct Point { pub scope: Scope, pub index: usize }   // Ord: lexicographic
```

Lexicographic `(scope, index)` is execution order only because the chain has
both properties a tree lacks: scopes are totally ordered by nesting, **and**
all of an outer scope's code precedes all of an inner scope's. In a tree the
second is false — `body`'s own defs sit on both sides of the fold's — so
`Placement`'s "strictly increasing sequence of spans" stops being a sequence
at all, and `Placement::at` answers with a location the value had already
left.

### The fix is a subtraction: a placement is per scope

Give up nest-wide placements. A value's life is a life *within one scope*,
because a scope is a loop body and a loop body is what a scan reasons about.
What crosses a scope boundary is already carried by something else — `parked`,
a slot or a carried register — and that mechanism is untouched.

This is the grain of the code rather than something imposed on it. Every
consumer already reads placements per scope:

- `Allocation::transitions` **filters** spans by `s.from.scope == scope` and
  throws the rest away.
- `record` takes one `scope` and stamps every range it writes with it.
- `parked` already holds the cross-scope answer, and is what an inner scan is
  given.

So `Point` loses its `scope` field and becomes what its name says — a position
in a schedule — and `Scope` becomes a key rather than half a coordinate.
Nothing needs cross-scope ordering afterwards, which is why the tree stops
being hard.

One consumer changes meaning, and the new meaning is the correct one.
Head reconciliation asks `placement.at(Point::TAIL) != at_head` — *"did this
value end the nest somewhere other than where the next iteration expects it"*.
The honest question is about **this scope's** tail, since this scope's back
edge is the one being reconciled. For the collapse body the two coincide
(the body is the innermost scope, so its tail is the nest's), which is why
today's version is right — by coincidence. For a fold they differ.

### Measured: the head reconciliation has never run

§4a says "codegen already has the answer" and quotes the reconciliation. It
does have it. **It has also never executed.** A `panic!` in that branch, run
over `pixelflow-codegen`, `pixelflow-core` and `pixelflow-graphics` — glyph
bakes and all — comes back green: 20 suites, 0 failures, no hit.

It cannot fire, and the reason is structural rather than accidental. `record`
skips a parked value, so a root's ranges in the body are *only* the park; the
park is one span; so `at(TAIL)` and `at_head` are the same span by
construction. The neighbouring `resident_throughout` check
(`spans().all(|s| s.from <= inside || s.at == head)`) is dead for the same
reason — there are no spans after the park to disagree.

Both were written for a shape the chain cannot produce: a value the scopes
*inside* a park relocate. Nothing relocates one, because a carried root's
register is removed from the inner pool and a slot-parked root is reloaded per
use.

This matters twice over:

1. **It corrects the premise.** A fold is not slotting into working
   machinery; it will be the first thing to run this code, and the first
   thing to depend on it being right. Unexercised code that has been correct
   for months is not evidence.
2. **It says what actually fixes the bracket bug.** The 2/89 glyph value
   failures came from the body's own scan relocating values across a back
   edge in the middle of its schedule — where there was no park at all. What
   fixes that is *giving the fold a park set*, not running the reconciliation.
   With a park, a fold's live-ins are pinned for the whole of it, exactly as a
   region's roots are, and the reconciliation stays dead.

So the reconciliation should not be trusted as the answer, and should not be
deleted as dead either until 2c settles whether a fold's own back edge needs
it. Whichever way that lands, it needs a test that reaches it — a nest where
an inner scope genuinely moves a parked value — rather than another few months
of looking correct.

### Sequencing

`ExpandReduce` still runs, so no `Reduce` reaches codegen and none of this is
load-bearing until the last step. That buys a gate for each piece:

| | | gate |
|---|---|---|
| **2a** | placements become per-scope | byte-identity — **done**, `cb740e4` |
| **2b** | a surviving `Reduce` is a def plus a scope | additive — no kernel has one yet |
| **2c** | delete `ExpandReduce`; the e-graph decides | behaviour |

2a and 2b were planned as three steps, with a middle one that added the tree
types and no producer for them. That middle step is folded into 2b: types with
no caller are the machinery this plan is supposed to be removing, and the
byte-identity gate cannot see them either way.

### The shape 2b adds

A parent pointer, not recursion. Storage stays flat — one `ScopeCode` per
scope, which is what the dense placement vectors want — and the tree is the
`parent` field:

```rust
pub struct FoldScope {
    /// The scope whose schedule holds this loop's def.
    pub parent: Scope,
    /// Which def — the `Reduce` this is the body of.
    pub at: usize,
    pub schedule: Vec<Def>,
}
```

with `Scope::Fold(usize)` indexing them. A fold inside a fold is
`parent: Scope::Fold(j)`; nothing special-cases depth.

`within()` becomes a subtree walk rather than a suffix of the chain, and that
is the whole of the tree in the allocator.

### A label should be keyed by the node it names (JP, 2026-09-10)

The emitter keeps two maps and only one of them is doing work:

| | | |
|---|---|---|
| `Assembly.bound: Map<Label, usize>` | label → position | irreducible; this *is* what an assembler does |
| `pending_binds: Map<(guard_idx, arm), Label>` | site → label | exists only because `asm.label()` mints an opaque id |

The second is bookkeeping to remember which id was minted for which site, and
its key is *positional* — `guard_idx` is an index into a scratch
`Vec<SelectGuard>` — where the DAG node is the actual identity. Make the
label's identity the site and the map goes, along with its insert, its remove,
and the `assert!(pending_binds.is_empty())` that checks the bookkeeping was
kept.

The reason this belongs in *this* plan rather than in a tidy-up: `emit_loop`'s
head and exit are per-`Reduce`, and **sibling folds sharing a binder index has
already been a bug here once** (§4a). It was fixed with a search rule — stop
at the previous `Reduce` over the same binder. A label keyed by the Reduce's
`ValueId` cannot alias a sibling's at all. A rule in a comment versus a key
that is unrepresentable when wrong is the trade CLAUDE.md keeps naming, and
this is a place to take it.

The cost, stated honestly: not every label has a node. The constant-pool
anchor has none, and the collapse scaffold's loops are not in the schedule
(until step 3 puts them there). So either `Label` becomes a sum that mentions
`ValueId` — and the assembler stops being usable for any little program, which
is what it was built to be — or the label type becomes a parameter,
`Assembly<L: Ord>`. The latter is a different axis from the `Assembly<Branch>`
that was rejected: that parameterized the *instruction*, where there is one
answer; this parameterizes what names a position, which the assembler has no
opinion about.

### Where step 3 lands

Uniformity, and it is already visible from here. A region *is* the prologue of
the loop that contains the next scope, and `body` is the innermost loop's
body — so the chain is the same (loop, its body) pairing the folds use, with
the loop kept in the scaffold instead of in the schedule. Deleting the X/Y
precoloring turns those two loops into defs, at which point `regions` and
`body` stop being separate concepts from `folds`, and `ScopedSchedule` is one
tree of scopes with one kind of node.

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
