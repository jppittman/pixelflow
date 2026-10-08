# Codegen

### Phase. Homonym

- **Is:** a function from one representation to the next that owns the
  decisions about its output.
- **Is not:** told in advance what a later phase will need; deciding what an
  earlier phase owns. Other senses of the word: a saturation phase (a subset
  of one rule set), and plan or research phases (A–E, "Phase 3").
- **Follows:** a count, reservation or size handed to an earlier stage by a
  later one's needs means a phase is missing between them. Codegen's phases
  are IR → instruction selection (instructions over values) → register
  allocation (and frame layout, and its own spill code) → assembly (labels,
  sections) → encoding. The same rule applies upstream: an ordering stated
  as a type removes the guard that defends it (the phase-typed arena, see
  Legality).
- **Lives:** `pixelflow-codegen/src/pipeline.rs` (legalize → lower
  (`arena_to_schedule`) → scopes and layout (`ScopedSchedule::from_schedule`)
  → allocate and emit (`compile_native`)); `scripts/check-emit-boundary.sh`.
  Instruction selection is the missing phase.

### Lowering

- **Is:** the boundary "where an algebra becomes an instruction". Iteration
  and a fold's combine enter here, after extraction decides that the fold
  survives: "lowering targets the schedule, so the carried value is a
  `ValueId`."
- **Is not:** an e-graph step. Not a place where codegen consults the
  algebra.
- **Follows:** the e-graph reasons about the monoidal form, and the lowered
  form may fuse `acc + a[i]*b[i]` into an FMA. `arena_to_schedule` reads
  variance once to choose between gather and broadcast.
- **Lives:** `pixelflow-codegen/src/program/lower.rs` (`arena_to_schedule`);
  a-surviving-reduce-is-a-loop 2a″. Today the combine is emitted by the
  `Reduce` arm in `emit/mod.rs` outside the schedule (see Loop), which
  contradicts this entry.

### Schedule (Def). Homonym

- **Is:** in codegen, a scope's evaluation order: a topological `Vec<Def>`,
  where each `Def` defines exactly one value (SSA). It is "the first layer
  that has" a previous iteration. Other senses: the Halide sense, a form
  plus placement ("an extraction is no longer a form; it is a form **plus a
  schedule**"); and the loop nest ("the lattice is part of the schedule",
  and schedule loops are folds).
- **Is not:** an annotation on a previous schedule. Not sorted by demand
  (09-07, retracted 09-08). Not recovered by a search after flattening.
- **Follows:** layout places each block after its inputs and its mask, and
  "Nothing else moves". Schedule choices make cost non-additive.
- **Lives:** `pixelflow-codegen/src/program/mod.rs` (`Def`, `ScheduledOp`,
  `ScopedSchedule`), `program/layout.rs`. Today `ScopeCode`
  (`emit/regalloc.rs`) says "The schedule is an *output* because choosing it
  is part of allocating", and `emit_scope` says "The allocator chooses the
  evaluation order", while `RegisterAllocator::allocate_nest` says "The nest
  is handed over finished … returns it unchanged". The docs disagree.

### Scope and the nest

- **Is:** in the language, a binder ("A scope is something that binds a
  variable"). In codegen, a loop body of the nest: `Scope::Body`, run once
  per call, or `Scope::Fold(i)`. A scope is "a **name**, not a coordinate".
  "The nest is a **tree**", with folds hanging off whichever scope holds
  their def: "there is the body, run once per call, and folds all the way
  down." Lanes are a scope.
- **Is not:** a chain of prologues (`Scope::Region(i)` was deleted). Not a
  hoist tier. Not ordered by nesting (the derived `Ord` is for keying). Not
  half of a program point. Not the assembler's concept.
- **Follows:** a fold must be a scope, or a relocated register crosses a
  back edge nobody owns (the 2/89 glyph failures). Placements are per
  scope. What crosses a boundary is a park or a carry. Each back edge is
  reconciled at its own head. The frame is a tree. A scope's children are
  found by scanning `folds` for their parent.
- **Lives:** `pixelflow-codegen/src/program/mod.rs` (`Scope`,
  `ScopedSchedule { body, folds }`, `ScopeFold`, `ScopeRegion`),
  `program/scopes.rs` (`extract_folds`); a-surviving-reduce-is-a-loop
  §4a–§4b; collapse-is-a-fold §2.2, step 5.

### Placement (Span, Point). Homonym

- **Is:** two senses. **Scope placement** is which scope computes a value:
  no further out than the outermost scope binding every bit of its variance,
  and no deeper than the common ancestor of its consumers, with cost
  choosing between. The allocator's **`Placement`** is where a value lives
  at every point of one scope: "a non-empty, strictly increasing sequence of
  `Span`s". A `Point` is an index within one scope.
- **Is not:** one location for a value's whole life. Not nest-wide
  (`Point { scope, index }` was retracted). Not a stack address ("choosing
  that a value spills and choosing *where* it spills are different
  decisions"). Not a separate hoist planner ("`plan_collapse_hoist` is
  placement, and placement exists").
- **Follows:** eviction splits a life. The frame is laid out after every
  placement is known. Today `place_roots` always takes the outer bound and
  reads no demand, which lifted each piece's row work above its band select
  (79% of executed instructions). Placement that reads demand is D1.
- **Lives:** `pixelflow-codegen/src/emit/regalloc.rs` (`Placement`, `Span`,
  `Point`, `Where`), `pixelflow-codegen/src/program/scopes.rs`
  (`place_roots`); a-surviving-reduce-is-a-loop §4b; an-integral-is-a-fold
  §4.

### Root. Homonym

- **Is:** a scope's roots are "the values it computes for the scopes inside
  it: placed here, by the outermost scope binding every binder the value
  depends on". A fold's binder and accumulator are its roots. So are
  constants and `Context` pointers. Other senses: a kernel's root, the arena
  node a `Kernel` names; and a candidate's match root, its e-class.
- **Is not:** reserved by fiat. Not emitted in the scopes inside, where it
  is a placeholder read from its park. Not a value with zero reads.
- **Follows:** every root is carried or parked by one ranking. A root's
  handoff counts as a read at its definition. An arm may own a root when it
  owns every scope that reads it (a-glyph-is-a-formula §4.3, confirmed
  "still needed" by an-integral §6.6).
- **Lives:** `ScopeRegion::roots`, `ScopeFold::roots`
  (`pixelflow-codegen/src/program/mod.rs`), `FoldRoots`
  (`emit/regalloc.rs`), `place_roots` (`program/scopes.rs`). Today
  `program/ownership.rs` says "A scope's root is read by the scope itself,
  so no arm owns it", which contradicts the arm rule.

### Hoist (LICM)

- **Is:** placing a value in the outermost scope binding every binder it
  depends on. "Our ability to hoist out of range loops should be the same
  machinery as our ability to hoist out of folds" (JP). Factoring is the
  same transformation that also moves the operation (`FactorFold`:
  `⊕_i (c ⊗ f) = c ⊗ ⊕_i f` when `i ∉ var(c)`).
- **Is not:** a pass over a flat schedule (`plan_collapse_hoist`,
  `HoistCtx`, `hoist_slots`, all deleted). Not a trip to memory ("Hoisting a
  value out of a loop into a load is a far weaker optimisation than hoisting
  it into a register"). Not a licence to strip arm exclusivity. Not
  `IfHoistUnary`, a different "hoist" that pulls shared work out of `If`
  arms.
- **Follows:** a nested fold invariant in its enclosing binder runs once:
  `8`@32 went from 536,960 bytes to 12,390. Trip-count weighting multiplies
  cost-model error by the trip count. A leaf is never hoisted.
- **Lives:** `pixelflow-codegen/src/program/scopes.rs`; `FactorFold`
  (`pixelflow-search/src/egraph/fold_rules.rs`); `IfHoistUnary`
  (`pixelflow-search/src/math/round2_rules.rs`);
  a-kept-structure-is-control-flow §1; collapse-is-a-fold §2.2, step 1.

### Park. Homonym of the actor park

- **Is:** where a root waits for the scopes inside: "the slot the scope
  computing it writes after the def, and the scopes inside read it from —
  unless it is carried into them in a register." For a fold's accumulator
  the park is the phi: "`acc_init` and `acc_next` are the **same park**."
- **Is not:** a slot pinned outside the allocator. "The accumulator lives in
  a slot" (a-surviving-reduce §3) was "a **bespoke allocation** wearing a
  performance argument", retracted the same day. Not a hole in a live range
  ("a park is a whole-scope answer rather than a hole in one"). Never a
  mask.
- **Follows:** nothing inside a scope relocates a parked value, so head
  reconciliation stays dead. "A skipped region leaves zero in its parks.
  Never a mask." That rule is dissolved today, because no arm owns a root.
  If arms come to own roots (see Root), the zero-park rule comes back with
  them.
- **Lives:** `NestAllocation::parks`, `Allocation::park`
  (`pixelflow-codegen/src/emit/regalloc.rs`), `place_roots`
  (`program/scopes.rs`); a-surviving-reduce-is-a-loop §3a (JP: "Register
  allocation is the job of the register allocator"), R1; demand §2 "Parks";
  one-conditional §4.

### Carry

- **Is:** keeping a root, or a fold's binder or accumulator, in a register
  across every scope inside the one that computes it. "A root is carried iff
  its placement at the body's first point is `Reg(_)`." Every carry is
  decided in one plan, by "reads saved per batch" weighted by trip counts,
  "under one constraint, that no scope has more carried across it than the
  pool has above the floor", with one budget per register class.
- **Is not:** a reservation by fiat, or a temp held across a body. Not a side
  channel (the `carries` map is gone). Not frequency-weighted eviction
  (+37% code, reverted).
- **Follows:** each carry takes one register from every inner scope's pool
  (`file.inside(carried)`). Anything inside a loop that writes a carried
  register is a miscompile visible only in pixels. The coldest root, the
  outermost accumulator, goes to a slot first.
- **Lives:** `plan_carries`, `CarryPlan`, `Carried`, `RegisterFile::inside`,
  the per-class `type Budget = [usize; 2]`
  (`pixelflow-codegen/src/emit/regalloc.rs`); escape-hatches step 4;
  collapse-is-a-fold step 2½; a-pointer-is-a-value §2.

### Back edge (head reconciliation)

- **Is:** the branch from a loop's tail to its head. It breaks linear scan's
  assumption that each index executes once. Head reconciliation, at a
  scope's head, would restore a value that ended an iteration elsewhere.
- **Is not:** working machinery: "It has also never executed." Parked values
  are skipped, so it cannot fire. Not a term of the language: `Acc` was a
  back edge in a DAG.
- **Follows:** "Unexercised code that has been correct for months is not
  evidence." Either reach it with a test or show it is unneeded and delete
  it.
- **Lives:** `pixelflow-codegen/src/emit/regalloc.rs` (`Point::TAIL`);
  a-surviving-reduce-is-a-loop §4b.

### Value

- **Is:** a name for something computed: one definition, its reads, a class.
- **Is not:** a register; a resource.
- **Follows:** names copy, resources move — a value may be `Copy`, a register
  may not. Constants are values, pointers are values, and the accumulator is
  a value, so "register or slot" is asked and answered the same way for all
  of them. A reload result and an instruction temp are values the input DAG
  did not contain. A value live into a block from several predecessors is a
  block parameter.
- **Lives:** `ValueId(pub u64)`, `Def` (`pixelflow-codegen/src/program/mod.rs`).
  Today an effect (`Write`, `Seq`) is a `Def` with a `ValueId` that defines
  no value, and a leaf scheduled in two scopes is "two definitions of one
  name" (09-04). Both contradict "one definition", which is why lives are
  kept per definition.

### Class

- **Is:** `Vector` (one batch of `f32` lanes, in a `Reg`) or `Pointer` (an
  address, in a `PtrReg`), "a function of the defining op". Every consumer
  knows which class it expects by position. "The classes never compete for
  a register."
- **Is not:** a reason for a second algorithm ("One algorithm, two pools").
- **Follows:** carry budgets are per class. Vector-only scratch (temps,
  guard registers, the result role) does not apply to pointers.
- **Lives:** `program::Class`, `ScheduledOp::class`
  (`pixelflow-codegen/src/program/mod.rs`); a-pointer-is-a-value §1–§2.

### Pointer (Context)

- **Is:** "A value in a schedule has a **class**: it is a vector of `f32`
  lanes, or it is an address." `ScheduledOp::Context(k)` is "the `k`-th
  pointer of the context the kernel is called with": buffer bases, then the
  link's uniform block, then the origin block. It has variance `CONST`.
- **Is not:** a slot immediate on the reading op (`slot`, `ctx_slot` are
  deleted). Not reloaded at every read (`MovLoadPtr` per read is deleted).
  Not a vector. Not allocated outside the allocator. JP: "it should know
  what a pointer register is, and our assembler should require one."
- **Follows:** `Gather`, `Broadcast` and `Uniform` take the base as an
  operand. `ResolvedOp` types the base as `PtrReg`, so "a wrong class at
  resolution is a panic naming the allocator, never a silently wrong
  address". Loads fell from 45 per batch to 3 per call. Open: fold binders
  as integers in GPRs, the pool anchor as a value, and callee-saved GPRs.
- **Lives:** `ScheduledOp::Context(u16)` (`pixelflow-codegen/src/program/mod.rs`),
  `Where::Ptr`, `RegisterFile::pointers` (`emit/regalloc.rs`);
  `docs/plans/2026-09-22-a-pointer-is-a-value.md`.

### Constant pool (constants in codegen)

- **Is:** "Constants are values": born `in_slot` (`Where::Remat`, a slot that
  "is the instruction stream"), evicted by distance, re-kept on a re-read,
  and parked or carried like any root. They are read from one pool per
  function, which is a data section of the assembly program with its own
  label.
- **Is not:** a tier below every other value. Three hand rules ("each written
  to hide the thrash the one before it caused") were removed in step 5½.
  Not inline data jumped over in the instruction stream. Not "rebuilt in
  one instruction".
- **Follows:** a rematerialization is a memory operation, so counted memory
  traffic rose honestly while bytes fell. The pool's anchor is a pointer
  value the allocator should place.
- **Lives:** `CONST_POOL`, `CONST_POOL_ALIGN` (`pixelflow-codegen/src/emit/mod.rs`),
  `POOL_BASE` (`emit/x86_64.rs`), `X17` (`emit/aarch64.rs`);
  collapse-is-a-fold step 5½. Today the anchor is pinned to `r8`/`X17` by
  convention (a-pointer-is-a-value §5, open) and the pool's label is the
  string constant `CONST_POOL = "const_pool"`. Both contradict Register and
  Label.

### Register

- **Is:** a finite resource of a machine, in a class (vector, general
  purpose, mask, flags). A resource, not a name.
- **Is not:** `Copy`; constructible anywhere; reserved by convention; chosen
  by an encoder. Not a contiguous range: "A representation that cannot say
  the true thing rounds the true thing down."
- **Follows:** registers are tokens minted once, by the backend's
  register-file declaration, and owned by the allocator. The allocator binds
  values to them by move and lends them to encoders by borrow. Two live
  values in one register, or a register chosen outside the allocator, are
  compile errors. The ABI is initial ownership: the entry block's parameters
  own the argument registers. There is no allocation outside the allocator,
  and no exception.
- **Requirement versus choice:** a register the machine or ABI fixes — the
  argument registers, `sp`, x86's shift count in `cl`, `div`'s `rax`/`rdx` —
  is a requirement, stated as a constraint on a *value* ("this value is in
  `cl` here") that the allocator satisfies with moves. A register the
  hardware would accept any member of the class for — the branch test in
  `w16`, the pool base in `x17` or `r8`, `k1` — is a choice, and choices are
  the allocator's. AAPCS64's IP0/IP1 say only that a linker veneer may
  clobber `x16`/`x17` across a call; a leaf kernel makes none. Whoever owns a
  resource makes every choice about it; whoever needs a specific one states
  the requirement, not the choice.
- **Lives:** today `Reg`/`Gpr`/`PtrReg`/`KReg(pub u8)`
  (`pixelflow-codegen/src/emit/mod.rs`), all `Copy` with public fields, so
  any code can construct one (`POOL_BASE: PtrReg = PtrReg(8)`,
  `X17: PtrReg = PtrReg(17)`), which contradicts this entry; `RegisterFile`,
  `RegSet`, `GprSet`, `MaskSet` (`emit/regalloc.rs`). Escape hatches:
  `docs/plans/2026-09-01-register-allocation-escape-hatches.md`. `fixed` is
  empty on every backend. The live hatch is `POOL_BASE`.

### Temp

- **Is:** nothing. A "temp" is a value with a short life, or a register some
  stage needed and could not ask the allocator for.
- **Is not:** a fixed register held out of the pool (`X86_SCRATCH`,
  `UNARY_SCRATCH`, all removed). Not spillable ("a temp cannot spill"),
  which is the only reason `MIN_SCRATCH` exists.
- **Follows:** where code reserves temps (`temps_for`, `REDUCE_TEMPS`), a
  phase is missing — the stage that would have made them values runs after
  the one that needed them (see Phase). "A reload result and an instruction
  temp are the same thing — a value the input DAG did not contain."
- **Lives:** the `RegisterFile::{temps_for, gpr_temps_for, mask_temps_for}`
  fields, filled by each backend's `temps_for` in
  `pixelflow-codegen/src/emit/{avx2,avx512,aarch64}.rs`;
  `Scratch::{REDUCE_TEMPS, MAX_TEMPS}` and `RegisterFile::MIN_SCRATCH`
  (`emit/regalloc.rs`). Today `MAX_TEMPS` is still sized for the deleted
  scalar-insert gather ("No encoding asks for more than two now").
  escape-hatches §A, §C, step 2.

### Spill, reload, remat

- **Is:** eviction as a split of a value's life: "The loser of an eviction
  keeps the register it held up to that point; its life continues in its
  slot." A spilled value is stored right after its definition.
- **Is not:** an annotation covering a value's whole life. Not a store at
  the eviction point, which a guard can skip.
- **Follows:** the store sits "at its definition, which a guard cannot skip
  without skipping every read". Splitting raises the slot count while
  lowering traffic. Residency is final once the reader is allocated, so
  reload targets are per-instruction roles.
- **Lives:** `Where::{Spilled, Remat}`, `LinearScan`
  (`pixelflow-codegen/src/emit/regalloc.rs`); escape-hatches step 2 and the
  2026-09-05 block.

### Slot. Homonym

- **Is:** a position in a table someone else owns. The senses:
  1. **IR table slot**: `UniformId`/`BufferId`, "Slot index into one arena's
     uniform table. Not an identity". The link chooses each slot's offset.
  2. **Binder slot**: the k-th reduction index.
  3. **Frame slot**: a stack location for a spill, a park, or a fold's
     accumulator or binder.
  4. **Context slot**: the `k` of `Context(k)`.
  5. **Atlas slot**.
- **Is not:** an identity. Not a register. Not a `Copy` token anyone may
  construct.
- **Follows:** splicing merges by identity, never by slot. Equal shapes share
  code because uniforms are keyed by dense slot. Two sibling folds share one
  binder slot, so anything keyed by it aliases. A fold's frame slots are
  based at its parent's top: "The frame is a tree, not a max." When they
  were not, the atlas came out blank. The layout owner owns the offsets.
- **Lives:** `UniformId(pub u64)`, `BufferId(pub u16)`
  (`pixelflow-ir/src/arena.rs`), `Binder(u8)` (`pixelflow-ir/src/fold.rs`),
  `emit::storage::Slot` (`pixelflow-codegen/src/emit/storage.rs`),
  `NestAllocation::{parks, accumulator_slots, binder_slots}`
  (`emit/regalloc.rs`). Today `Slot::new` is a public `const fn` on a `Copy`
  type, which contradicts denotational-diagnosis step 8. `FrameLayout` was
  deleted 2026-10-06 ("The allocator lays the frame out itself").

### Frame. Homonym

- **Is:** in codegen, the stack frame the allocator lays out beside its
  placements: regions and the body at 0, each fold at its parent's top.
  Other senses: graphics' `PackedFrame` ("a *frame* is exactly the bound
  form"); a displayed frame (see Frame (display)); and the retired frame
  prologue or frame tier.
- **Is not:** a max over scopes. Not laid out by the emitter ("the emitter
  reads them and computes none").
- **Follows:** "The frame prologue is never guarded." No per-frame heap
  allocation.
- **Lives:** `NestAllocation::new` (`pixelflow-codegen/src/emit/regalloc.rs`),
  `StackFrame` (`emit/storage.rs`); the
  `a_folds_spill_slots_do_not_alias_its_parents` guard (`emit/mod.rs`).

### Arm ownership. Homonym of Arm

- **Is:** an `If` arm as a region of the DAG. "Every `If` opens two
  **regions**, one per arm, nested in the region the `If` itself belongs to
  … A value belongs to the **lowest common ancestor** of the regions that
  read it." An arm owns exactly what only it reads, including folds, each
  priced at its trips (`FoldReads`). Other senses of "arm": the extractor's
  tree and shared DP arms, and a remainder arm (the code for an extent that
  is not a lane multiple).
- **Is not:** a contiguous run recovered by searching a flat schedule
  (`select_arms`, 80.5% of emit; `cluster_if_arms`; both deleted). Not
  exclusive by reachability alone. The old "reaches the arm" rule was a
  latent miscompile; exclusivity is a closure, "my consumers are skipped
  with me". Not a `KernelKey` field (G1/G2, retired).
- **Follows:** what skipping saves is everything the arm owns. Hoisting must
  not strip exclusivity, which is a DAG fact computed before partitioning.
  An arm may own a root when it owns every scope that reads it (see Root).
- **Lives:** `pixelflow-codegen/src/program/ownership.rs` (#1311),
  `program/layout.rs` (#1312, #1313), `program/guards.rs` (`FoldReads`);
  CLAUDE.md "`If` contains an if".

### Guard. Homonym

- **Is:** five senses:
  1. A **guard clause**, the strongest fold.
  2. An **`If` guard**: the branch that skips an arm's owned block when a
     batch's mask is uniform (`IfGuard`). It is lowering 2 of `If`.
  3. A **runtime guard**: a check "defending what a type should have made
     unrepresentable" (`retired_axis` in `emit::compile`).
  4. The retired IR node `ExprNode::Guard { mask, on, off }`, and D7's open
     denotation `If(m,a,b) = Guard(m,a) ⊕ Guard(¬m,b)`, a one-armed
     conditional the e-graph could choose.
  5. `FastMathGuard`, a scoped FTZ/DAZ switch.
- **Is not:** an analysis the emitter runs. Ever in the frame prologue. A
  structural saving: "It is a data-dependent win". Not equal to `If` (the
  retired node equalled `If` "only for a batch-uniform mask"; retired
  2026-10-05, G3 withdrawn).
- **Follows:** a guard's mask is read at the guard site, so liveness covers
  it. A skipped arm leaves zero in its parks. A runtime guard is a symptom:
  fold it into a type wherever the wrong value would be silently
  representable.
- **Lives:** `pixelflow-codegen/src/program/{guards,layout}.rs`, `IfGuard`
  (`program/mod.rs`); `retired_axis` checked in `compile`
  (`pixelflow-codegen/src/pipeline.rs`); `FastMathGuard`
  (`pixelflow-core/src/fastmath.rs`); emit-should-just-emit status
  (2026-10-05); CLAUDE.md "Denote before you build".

### Mispredict bound

- **Is:** `MISPREDICT_PENALTY_CYCLES` (16), "the single analytic
  profitability bound" an arm's price must clear to be laid out as its own
  block.
- **Is not:** a tuned constant. Not replaceable by a learned term ("A learned
  term may rerank; it may not replace the analytic floor"). Not to be met by
  a fold priced at zero.
- **Follows:** the guard analysis and the extractor price a fold with one
  function, `fold_cost` (D2). Whether this bound decides that an `If` jumps
  at all is open (see If).
- **Lives:** `pixelflow-codegen/src/program/guards.rs` (the constant),
  `program/layout.rs` (its one use); one-conditional §9;
  an-integral-is-a-fold §6.6, D2.

### Loop

- **Is:** a block whose last instruction may branch back to its own label. A
  fold that survives optimization is a loop.
- **Is not:** an instruction; a scaffold of verbs around a body. Not a
  different kind of thing from the lattice's loops, which are folds. Not
  free: allocation across the back edge, the loop-carried accumulator, and
  lost CSE across copies are its costs.
- **Follows:** the header is the block's label. The binder and accumulator are
  block parameters. The trip test, the step and the accumulate are ordinary
  instructions. Nothing about a loop needs reserved registers. A fold inside
  a fold is a loop inside a loop. The lane fold is a loop executed by lanes,
  with no counter and no back edge.
- **Lives:** decided in `docs/plans/2026-09-10-a-surviving-reduce-is-a-loop.md`
  (R1: "the accumulate is an ordinary `Binary` def … the allocator learns
  nothing about folds"); today emitted by verbs in
  `pixelflow-codegen/src/emit/mod.rs`'s `Reduce` arm ("seed, test, body,
  combine, step", with `Scratch::REDUCE_TEMPS`, labels
  `reduce{vid}_top`/`reduce{vid}_exit`), which contradicts it.

### Block. Homonym

- **Is:** a label, its parameters, and a sequence of instructions ending in
  one whose label operands are the block's successors. Other senses: a
  layout block (an earning arm region laid out as one run), a `kernel!`
  block of items, and a program's uniform block.
- **Is not:** a run found by permuting a flat schedule.
- **Follows:** a value live into a block from more than one predecessor is a
  block parameter; that is what a phi is. "Emit the arms as blocks and there
  is nothing to search for."
- **Lives:** to be built at the assembly level. Today
  `pixelflow-codegen/src/program/layout.rs`'s blocks have no label,
  parameters or terminating instruction.

### Label. Homonym of the hindsight label and the cost label

- **Is:** the name of an address — an abstraction over a position in a
  program, nothing more. Things *have* labels: a block has one, the constant
  pool (a data section) has one. A label is minted with the thing that lives
  at its address, so it always names something and no two things share one.
- **Is not:** a string; an instruction; a field on an instruction; a sum of
  the things that can be labelled (`Label::ConstPool | Head(..) | Join(..)`
  inverts the dependency — the assembler would know codegen's vocabulary, and
  every new labelled thing would edit the label type); derived from the id of
  what it names (two sibling folds carry one `ValueId`). Not a hindsight
  label (`labeler::Label`, a training target on a rule application) or a
  `CostLabel` (a measurement), which are different nouns in different
  modules. Not a fixup token returned by `emit_jump` and patched by hand
  (deleted in R0).
- **Follows:** an operand can be a label, so a branch is an ordinary
  instruction whose argument is a label. The assembler maps labels to
  addresses and owns nothing else about them. Because the label is born with
  its block, the code that creates a block holds its label — no map from
  sites to labels, and no label bound twice or bound nowhere. A loop header is
  its block's label. A label is a name, so it is `Copy`; it has no public
  constructor, so it cannot dangle; it is 64-bit, like every id.
- **Lives:** `emit::Label` (`pixelflow-codegen/src/emit/mod.rs`) — today a
  31-byte string built by `format!`, which contradicts this entry. Its names
  are derived from the id of what they name (`format!("reduce{}_top",
  vid.0)`, `format!("v{}_past_{side}", …)`), and `Label::new(&str)` is a
  public constructor. Its doc says "There is nothing to mint, nothing to
  keep, and no table to look a name up in". Clashes are prevented by naming
  convention, not by type. History: R0 of
  `docs/plans/2026-09-10-a-surviving-reduce-is-a-loop.md` made a label an
  item and its reference an operand; "A label should be keyed by the node it
  names" in the same plan removed the site-to-label map by keying on the
  node — minting the label with the block removes it without the label
  knowing about nodes. The hindsight label's rename to `HindsightLabel`
  (2026-08-17 J3) has not landed.

### Operand

- **Is:** an argument of an instruction: a register (a value with an access —
  read, written, both — and a constraint), a label, an immediate, an address
  (with register operands inside), or a frame slot.
- **Is not:** split into special methods by kind (no `target()` beside the
  operands). Not an immediate standing in for a value the allocator should
  place (the `slot`/`ctx_slot` immediates, deleted). Not left to the cost
  model: a shift count must be an immediate, so legality attaches to the
  resolved form.
- **Follows:** each phase reads the operand kinds it owns — the allocator
  binds registers and frame slots, the assembler resolves labels, the encoder
  writes all of them. A block's successors are the label operands of its last
  instruction. `pin_shift_counts`, which re-pins a shift count's extraction
  choice to a `Const`, is the symptom of a missing operand-kind type.
- **Lives:** to be built (`docs/plans/2026-10-08-selection-is-a-phase.md`,
  named but not yet in the tree). Today `program::operands()` yields only
  vector reads, the base comes from a separate `pointer_operand()` (both in
  `pixelflow-codegen/src/program/mod.rs`), and branches expose their target
  through `AsmInsn::label_ref()` (`emit/mod.rs`). All three contradict this
  entry. `pin_shift_counts` lives in `pixelflow-search/src/egraph/extract.rs`.

### Instruction

- **Is:** an operation of the machine applied to operands: a value, chosen by
  instruction selection, encoded last.
- **Is not:** bytes written into a buffer by the code that chose it.
- **Follows:** every register an instruction touches is one of its operands —
  including the ones it clobbers and the flags — so nothing about it is hidden
  from the allocator. A branch is an ordinary instruction.
- **Lives:** `AsmInsn` (`pixelflow-codegen/src/emit/mod.rs`), the backends'
  `Inst` enums (`emit/x86_64.rs`, `emit/aarch64.rs`). Today most emission is
  `IsaBackend` verbs writing bytes inline ("Instructions: bytes written inline
  by ~236 functions, not values", denotational-diagnosis; e.g.
  `emit_write(&mut self, code: &mut Vec<u8>, …)`), which contradicts this
  entry.

### Branch

- **Is:** "A branch is an ordinary instruction": `Jmp { target }` and
  `Jcc { condition, target }` on x86; `B`, `BCond` and `BranchIfW16Zero` on
  aarch64. The condition is the opcode's own field (`Cond`, whose
  discriminants are the manual's values).
- **Is not:** something that returns a position. Not hand-picked mnemonics
  over a private `jcc(code, cc: u8)`.
- **Follows:** a loop's back edge is `push(Jmp { target: top })`. Arm skips
  are one verb, `branch_if_arm_is_dead(.., MaskTest, Label)`, "because they
  differed only in which uniform mask lets an arm go, which is what `IfArm`
  already names". A fold's trip test reuses `IfArm::False`'s test.
- **Lives:** `pixelflow-codegen/src/emit/{x86_64,aarch64}.rs` (the
  instructions), `IsaBackend::branch_if_arm_is_dead` (`emit/mod.rs`,
  implemented in `emit/{avx2,avx512,aarch64}.rs`), `IfArm`
  (`program/mod.rs`); a-surviving-reduce-is-a-loop R0.

### Assembly program

- **Is:** one per kernel: a value made of sections, each a sequence of
  items — instructions, and label bindings — with the constant pool as a data
  section. It is the namespace of its labels, and it mints them. "Not an
  AST — assembly is not context-sensitive."
- **Is not:** a buffer of bytes; split per scope and spliced together (one
  program, one namespace — a scope is the allocator's concept, not the
  assembler's). Superseded: R0's "two front ends, one mechanism" (a value
  `AsmProgram` beside a push/bind/finish `Assembly`).
- **Follows:** whatever builds the program — instruction selection, then the
  allocator inserting its own code — builds a value; nothing writes bytes.
- **Lives:** to be built. Today `emit_scope`
  (`pixelflow-codegen/src/emit/mod.rs`) returns each scope's bytes as a
  `Vec<u8>` and the parent splices them in
  (`asm.code.extend_from_slice(&fold_code)`), which contradicts this entry.

### Assembler

- **Is:** a function from an assembly program to binary: lay the sections
  out, map each label to an address, encode each instruction given the
  addresses of the labels it names. Here, a very small in-memory one: no
  object files, no relocations or symbols beyond labels.
- **Is not:** stateful; a builder; a code buffer other code writes bytes
  into; aware of registers, values, scopes or the IR.
- **Follows:** it stands alone — its module imports nothing from the rest of
  the crate, and everything above it depends on it, never the reverse.
- **Lives:** `emit::Assembly` (a push/bind/finish builder with a public `code`
  field the driver writes into) and `AsmProgram`/`assemble` in
  `pixelflow-codegen/src/emit/mod.rs` — two front ends where the program is
  the value and `assemble` the function; today's builder contradicts this
  entry. `docs/designs/assembler-as-functor.md`.

### Emitter

- **Is:** the last stage, handed a finished `ScopedSchedule`. "The emitter
  gains no analysis. If a stage needs one, it belongs upstream. Hash-consing
  and other free-at-the-point-of-use folds are fine; a search is not."
- **Is not:** a search ("What calls itself 'emit' is four fifths a search").
  Not the builder of the loop nest. Not allowed to name upstream stages.
- **Follows:** lowering, scoping, ownership and layout live in `program/`.
  Emit time is about O(n^1.6) in straight-line instruction count, which is
  why code size is a cost axis.
- **Lives:** `pixelflow-codegen/src/emit/`; CI job `emit-boundary`
  (`scripts/check-emit-boundary.sh`, `scripts/check_emit_boundary.py`);
  `2026-09-12-emit-should-just-emit.md`.

### Scaffold (retired)

- **Is:** the collapse loop the emitter used to build around a kernel
  (`emit_collapse_loop`, `Level`, `CollapseBody`, `Counter`, `HoistCtx`). It
  "is not the fold machinery, so every property the fold machinery gets from
  the DAG … the scaffold has to recover by hand, with its own axioms … and
  its own carve-outs."
- **Is not:** a loop in this file's sense.
- **Follows:** it was deleted in collapse-is-a-fold step 5. Precoloured X and
  Y, `SCAFFOLD_ACC`/`SCAFFOLD_SCRATCH` and `RegisterFile::inputs` went with
  it.
- **Lives:** deleted. The residue is the per-fold verbs in the `Reduce` arm
  of `pixelflow-codegen/src/emit/mod.rs` (see Loop).

### ISA tier (vector width)

- **Is:** "The ISA tier is a property of the process, not of the build: a
  fact the CPU reports, read once." `Isa` has three inhabitants: `Avx2`
  (AVX2+FMA, the floor), `Avx512` (needs avx512f+dq) and `Neon`. The backend
  is a function of the tier, and the width is `Isa::vector_bytes`.
- **Is not:** a build flag (`RUSTFLAGS`/`target-cpu` change nothing). Not
  SSE2: "It was the default only because the build flag was unset". Not a
  ladder of fallbacks: a host below the floor is refused, naming the missing
  feature. Not downgradable: `PIXELFLOW_ISA` only narrows to a tier the host
  can run, "a diagnostic, never a fallback".
- **Follows:** an AVX-512 host and an AVX2 host emit different code by
  design. "Target" is finer than architecture (`Recip` is about 12 bits on
  one tier and about 14 on another). `xtask isa-matrix` runs both x86 tiers
  from one build. The tier must be an input to P.
- **Lives:** `pixelflow_codegen::isa::{Isa, detect}`, `jit_vector_bytes`
  (`pixelflow-codegen/src/isa/mod.rs`), `emit::compile_native`
  (`pub(crate)`, `emit/mod.rs`, which calls `detect()` itself);
  `2026-09-22-the-isa-is-decided-at-startup.md`; CLAUDE.md "SIMD Backend
  Selection".
