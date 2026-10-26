# Codegen

### Phase. Homonym

- **Is:** a function from one representation to the next that owns the
  decisions about its output.
- **Is not:** told in advance what a later phase will need; deciding what an
  earlier phase owns. Other senses of the word: a saturation phase (a subset
  of one rule set), and plan or research phases (A–E, "Phase 3").
- **Follows:** a count, reservation or size handed to an earlier stage by a
  later one's needs means a phase is missing between them. The CPU backend's
  phases are IR → instruction selection (instructions over values) → register
  allocation (and frame layout, and its own spill code) → assembly (labels,
  sections) → encoding; it alone allocates registers, and each backend is its
  own module with its own chain (note §3). The same rule applies upstream: an ordering stated
  as a type removes the guard that defends it (the phase-typed arena, see
  Legality).
- **Lives:** `pixelflow-codegen/src/pipeline.rs` (`compile`: the chain's
  front, `Legalize → Lower → Nest → Own → Layout`) and `emit::native` (its
  back, `Select → Allocate → Assemble`, on the tier's backend; see Pass);
  `scripts/check-emit-boundary.sh`.

### Pass. Homonym of the language's pass

- **Is:** one stage of the compiler as a value: a type with an `Input`, an
  `Output` and `run(self, input) -> Result<Output, CompileError>`, carrying
  its own parameters (shape, lanes, backend) as fields. Stages compose with
  `then`, Kleisli composition in `Result`: associative, and the first error
  ends the chain. The CPU module's chain is `Nest → Own → Layout → Select →
  Allocate → Assemble`, after `Legalize → Lower`; a GPU backend is another
  module with its own chain (note §3). Each stage's output type is the next's
  input (JP, 2026-10-12: "I want all the passes pipelined/monadic/laid
  neatly", "I want a pass trait").
- **Is not:** a pass manager. There is no `dyn Pass`, registry or list of
  passes at run time: a chain is a type, built where its parameters are
  known, and the backend is chosen once, when it is built. Not a stage that
  calls another stage (`Flow::of` used to run the backward ownership pass
  inside the forward layout, inside the scoping). Not the language's pass
  (an endomorphism on (arena, root): `legalize`'s steps, `Optimize`; see
  compiler.md), nor a stage of P (open value, closed program).
- **Follows:** a fact two later stages need is a stage of its own, with its
  own output type: ownership is `Own`, between scoping and layout, so
  `ScopedSchedule<A>` carries `()`, then each scope's `Ownership`, then its
  `Flow`, and a stage cannot be handed a nest its predecessor has not run on.
- **Lives:** `pixelflow-codegen/src/pass.rs` (`Pass`, `Then`);
  `pipeline.rs` (`Legalize`, `Lower`); `program/{scopes,ownership,layout}.rs`
  (`Nest`, `Own`, `Layout`); `emit/select.rs`, `emit/regalloc/local.rs`,
  `emit/mod.rs` (`Select`, `Allocate`, `Assemble`);
  docs/archive/plans/2026-10-08-selection-is-a-phase.md §0.14.

### Lowering

- **Is:** the boundary "where an algebra becomes an instruction". Iteration
  and a fold's combine enter here, after extraction decides that the fold
  survives: "lowering targets the schedule, so the carried value is a
  `ValueId`."
- **Is not:** an e-graph step. Not a place where codegen consults the
  algebra.
- **Follows:** the e-graph reasons about the monoidal form, and the lowered
  form may fuse `acc + a[i]*b[i]` into an FMA. `arena_to_schedule` reads the
  arena's variance once, and the variance chooses the realization of a
  discrete-kernel evaluation (broadcast, contiguous load or gather; note §2),
  and every `Def` carries its node's variance to scoping, which places by it.
  Nothing recomputes variance on the schedule.
- **Lives:** `pixelflow-codegen/src/program/lower.rs` (`arena_to_schedule`);
  a-surviving-reduce-is-a-loop 2a″. Today the combine is emitted by the
  `Reduce` arm in `emit/mod.rs` outside the schedule (see Loop), which
  contradicts this entry.

### Schedule (Def). Homonym

- **Is:** in codegen, a scope's evaluation order: a topological `Vec<Def>`,
  where each `Def` defines exactly one value (SSA) and carries that value's
  variance. It is "the first layer
  that has" a previous iteration. Other senses: the Halide sense, a form
  plus placement ("an extraction is no longer a form; it is a form **plus a
  schedule**"); and the loop nest ("the lattice is part of the schedule",
  and the lattice's traversal, which is not folds the kernel is wrapped in: kernels-evaluate-kernels §3).
- **Is not:** an annotation on a previous schedule. Not sorted by demand
  (09-07, retracted 09-08). Not recovered by a search after flattening.
- **Follows:** the schedule is never reordered: values stay in the order
  they were constructed, and an `If`'s arm blocks go immediately before it
  (the `Flow`). Schedule choices make cost non-additive.
- **Lives:** `pixelflow-codegen/src/program/mod.rs` (`Def`, `ScheduledOp`,
  `ScopedSchedule`), `program/layout.rs`. The schedule
  reaches the emitter finished: selection walks its `Flow`, and
  the `Allocate` pass (`regalloc::local`) chooses registers and slots, never the order.

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
  `ScopedSchedule<A> { body, folds }`, `ScopeFold`, `ScopeRegion`),
  `program/scopes.rs` (`Nest`, `extract_folds`); a-surviving-reduce-is-a-loop
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
  placement is known. `place_roots` (deleted in the selection series) took
  the outer bound and read no demand, which lifted each piece's row work above its band select
  (79% of executed instructions). Placement reads no demand: demand weighting
  is deleted (note §1).
- **Lives:** `pixelflow-codegen/src/emit/regalloc/mod.rs` (`Placement`, `Span`,
  `Point`, `Where`), `pixelflow-codegen/src/program/scopes.rs` (`Nest`;
  `place_roots` is deleted); a-surviving-reduce-is-a-loop §4b; an-integral-is-a-fold
  §4.

### Root. Homonym

- **Is:** what a scope computes for the scopes inside it: a sibling fold's
  accumulator, a constant, a `Context` pointer. Derived, not stored: the
  defs of a scope that a fold inside it names in `ScopeFold::outer`, each
  computed by the outermost scope binding every binder it depends on. Other
  senses: a kernel's root, the arena node a `Kernel` names; and a candidate's
  match root, its e-class.
- **Is not:** reserved by fiat. Not in the schedule of the scopes inside,
  which hold only their own definitions (`ScopeFold::outer` names what they
  read from outside). Not a value with zero reads. Not pinned to its scope: a
  loop reads its roots at its `Reduce` (`FoldReads`), and a root is owned
  like any value, by the lowest common ancestor of its readers' regions.
- **Follows:** an arm owns a root when it owns every scope that reads it
  (a-glyph-is-a-formula §4.3, an-integral §6.6): a batch that skips the arm
  computes none of it. That is sound because a fold's `outer` is closed —
  what it reads from outside, its parent defines ahead of the fold or reads
  from outside in turn, and it reads nothing it does not define or name
  there — which `Own` asserts of every scope (`assert_closed`).
- **Lives:** `ScopeFold::outer` (`pixelflow-codegen/src/program/mod.rs`),
  `Nest` and `assert_closed` (`program/scopes.rs`; `place_roots` is deleted), `FoldReads`
  (`program/ownership.rs`). Selection has no root or park of its own: a value
  live into a loop's head is carried in a register or stored and dropped
  there (`plan_carries`, `emit/regalloc/local.rs`). The `roots` fields were
  deleted with ownership's "no arm owns a root" pin, their one reader.

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
  Never a mask." (demand §7, one-conditional §9) is retired: arms own roots,
  and nothing reads what a skipped arm left behind. An arm-owned value is
  read only inside the arm's block, which one branch skips whole. Its life,
  its slot's lease and its store end inside the block, a loop's exit store
  included, so the join after the arm never holds it. The `If` reads the
  skipped arm's result only through the select, which takes no lane from
  it (selection §0.12; note §1).
- **Lives:** deleted with the legacy allocator (`NestAllocation::parks`,
  `Allocation::park`); `Nest` (`program/scopes.rs`; `place_roots` is deleted)
  removes a root from the folds that read it and names it in their `outer`;
  a-surviving-reduce-is-a-loop §3a (JP: "Register
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
- **Lives:** `plan_carries` (`pixelflow-codegen/src/emit/regalloc/local.rs`), the
  per-file `type Budget = [usize; 4]` (`.../regalloc/policy.rs`); escape-hatches step 4;
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
- **Lives:** the backward branch of a loop's latch, a sequentialized parallel
  move (`pixelflow-codegen/src/emit/regalloc/local.rs`), with join invariant 1
  asserted at each latch; a-surviving-reduce-is-a-loop §4b.

### Value

- **Is:** a name for something computed: one definition, its reads, a class.
- **Is not:** a register; a resource.
- **Follows:** names copy, resources move — a value may be `Copy`, a register
  may not. Constants are values, pointers are values, and the accumulator is
  a value, so "register or slot" is asked and answered the same way for all
  of them. A reload result and an instruction temp are values the input DAG
  did not contain. A value live into a block from several predecessors is a
  block parameter.
- **Lives:** `ValueId(pub u64)`, `Def` (`pixelflow-codegen/src/program/mod.rs`);
  `emit::Value<C>`, a typed name, and `build::Def<C>`, the right to define
  one (`emit/{mod,build}.rs`).
  Today an effect (`Write`, `Seq`) is a `Def` with a `ValueId` that defines
  no value, and a leaf scheduled in two scopes is "two definitions of one
  name" (09-04). Both contradict "one definition", which is why lives are
  kept per definition.

### Class

- **Is:** there are two, and the second replaces the first. `program::Class`
  is an enum, `Vector` (one batch of `f32` lanes) or `Pointer` (an address),
  "a function of the defining op". Every consumer
  knows which class it expects by position. "The classes never compete for
  a register." `emit::Class` (`emit/mod.rs`) is a sealed trait with the markers
  `Vector`, `Pointer`, `Integer`, `Opmask` and `Flags` (`ClassId` is the same
  set as data, on a `ValueName`), the type an instruction's
  operand field is declared with, so a base address cannot be handed a
  `row * pitch` product. The flags are a class. D2 deletes
  `program::Class`.
- **Is not:** a reason for a second algorithm ("One algorithm, two pools").
- **Follows:** carry budgets are per class. Vector-only scratch (temps,
  guard registers, the result role) does not apply to pointers.
- **Lives:** `program::Class`, `ScheduledOp::class`
  (`pixelflow-codegen/src/program/mod.rs`), `emit::Class`;
  a-pointer-is-a-value §1–§2.

### Pointer (Context)

- **Is:** "A value in a schedule has a **class**: it is a vector of `f32`
  lanes, or it is an address." `ScheduledOp::Context(k)` is "the `k`-th
  pointer of the context the kernel is called with": buffer bases, then the
  link's uniform block, then the origin block. It has variance `CONST`.
- **Is not:** a slot immediate on the reading op (`slot`, `ctx_slot` are
  deleted). Not reloaded at every read (`MovLoadPtr` per read is deleted).
  Not a vector. Not allocated outside the allocator. JP: "it should know
  what a pointer register is, and our assembler should require one."
- **Follows:** `Gather` and `Broadcast` (the backend's realizations of a
  discrete-kernel evaluation, note §2) and `Uniform` take the base as an
  operand. Selection types the base as `Value<Pointer>`, so a wrong class
  is a type error, never a silently wrong address. Loads fell from 45 per batch to 3 per call. Open: fold binders
  as integers in GPRs, the pool anchor as a value, and callee-saved GPRs.
- **Lives:** `ScheduledOp::Context(u16)` (`pixelflow-codegen/src/program/mod.rs`),
  `emit::Pointer`, the general file's `Members::general` (`emit/register_file.rs`);
  `docs/archive/plans/2026-09-22-a-pointer-is-a-value.md`.

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
- **Lives:** `CONST_POOL_ALIGN` (`pixelflow-codegen/src/emit/mod.rs`),
  `Constants` (`emit/mod.rs`); collapse-is-a-fold step 5½. x86 reads each
  entry RIP-relative by its own label and has no anchor; NEON's pool base is
  a `Value<Pointer>` selection defines once (`IsaBackend::Anchors`). It was pinned to
  `r8`/`X17` by convention before selection (a-pointer-is-a-value §5).

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
- **Lives:** `Reg`, `Pool` and `Lease` (`pixelflow-codegen/src/emit/regalloc/resource.rs`),
  minted only by `Pool::mint` and `Leases::new`; `In`, `Out` and `InOut`, the
  borrows an instruction's operand fields hold; `RegisterFile`
  (`emit/register_file.rs`), each backend's `IsaBackend::FILE`. Escape hatches:
  `docs/archive/plans/2026-09-01-register-allocation-escape-hatches.md`.

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
- **Lives:** nowhere: every intermediate an instruction needs is a value
  selection defines (`emit/select.rs`, `emit/build.rs`), and the allocator
  gives it a register like any other. `temps_for`, `REDUCE_TEMPS` and
  `MIN_SCRATCH` were the legacy allocator's predictions of them, deleted with
  it (D1). escape-hatches §A, §C, step 2.

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
- **Lives:** the `Allocate` pass (`regalloc::local`)
  (`pixelflow-codegen/src/emit/regalloc/local.rs`); escape-hatches step 2 and the
  2026-09-05 block. It does this one value at a time: a value stays in
  its register until the register is needed, the cheapest to give up by
  `EvictionRank` goes, and its store is placed after its definition when it is
  first evicted. A loop head still flushes every live value to its slot.

### Slot. Homonym

- **Is:** a position in a table someone else owns. The senses:
  1. **IR table slot**: `UniformId`, "Slot index into one arena's
     uniform table. Not an identity". The link chooses each slot's offset.
     A discrete kernel's table is not an IR slot: the IR names the kernel,
     and the table is the backend's (note §2).
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
  `FrameSlot` and `SlotName` (`pixelflow-codegen/src/emit/regalloc/resource.rs`):
  a `FrameSlot` is minted only by `Frame`, not `Copy`, at an offset fixed for
  its life. The legacy `Slot` (`emit/storage.rs`) and `NestAllocation::{parks,
  accumulator_slots, binder_slots}` were deleted with the legacy allocator
  (D1). `FrameLayout` was deleted 2026-10-06 ("The allocator lays the frame
  out itself").

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
- **Lives:** `regalloc::resource::Frame` (`pixelflow-codegen/src/emit/regalloc/resource.rs`),
  laid out by the `Allocate` pass (`regalloc::local`) from value intervals before any
  instruction is bound; the `a_folds_spill_slots_do_not_alias_its_parents`
  guard (`emit/mod.rs`).

### Arm ownership. Homonym of Arm

- **Is:** an `If` arm as a region of the DAG. "Every `If` opens two
  **regions**, one per arm, nested in the region the `If` itself belongs to
  … A value belongs to the **lowest common ancestor** of the regions that
  read it." An arm owns exactly what only it reads, including folds
  (`FoldReads`). Other senses of "arm": the extractor's
  tree and shared DP arms, and a remainder arm (the code for an extent that
  is not a lane multiple).
- **Is not:** a contiguous run recovered by searching a flat schedule
  (`select_arms`, 80.5% of emit; `cluster_if_arms`; both deleted), or
  placed by permuting one (the layout's anchors, deleted 2026-10-12). Not
  exclusive by reachability alone. The old "reaches the arm" rule was a
  latent miscompile; exclusivity is a closure, "my consumers are skipped
  with me". Not a `KernelKey` field (G1/G2, retired).
- **Follows:** what skipping saves is everything the arm owns. Hoisting must
  not strip exclusivity, which is a DAG fact computed before partitioning.
  An arm owns a root when it owns every scope that reads it (see Root).
- **Lives:** `pixelflow-codegen/src/program/ownership.rs` (#1311; `Own`, the
  backward pass), `program/layout.rs` (#1312, #1313; `Layout`, `Flow::of`: an
  `If`'s arms go immediately before it);
  CLAUDE.md "`If` guards its arms".

### Guard. Homonym

- **Is:** five senses:
  1. A **guard clause**, the strongest fold.
  2. An **`If` guard**: the branch that runs an arm's owned block when some
     lane takes it: `if any(m)` before the true arm's block, `if !all(m)`
     before the false arm's (an `ArmBlock` of the scope's `Flow`; note §1). An
     arm that owns nothing has none, and the `If` ends in one select.
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
  it. A skipped arm leaves nothing anyone reads (see Park). A runtime guard
  is a symptom: fold it into a type wherever the wrong value would be
  silently representable.
- **Lives:** `pixelflow-codegen/src/program/layout.rs`, `Flow`
  and `ArmBlock` (`program/mod.rs`); `retired_axis` checked in `compile`
  (`pixelflow-codegen/src/pipeline.rs`); `FastMathGuard`
  (`pixelflow-core/src/fastmath.rs`); emit-should-just-emit status
  (2026-10-05); CLAUDE.md "Denote before you build".

### Loop

- **Is:** a block whose last instruction may branch back to its own label. A
  fold that survives optimization is a loop.
- **Is not:** an instruction; a scaffold of verbs around a body. Not a
  different kind of thing from a fold's loop. The lattice's traversal is not a fold (kernels-evaluate-kernels §3). Not
  free: allocation across the back edge, the loop-carried accumulator, and
  lost CSE across copies are its costs.
- **Follows:** the header is the block's label. The binder and accumulator are
  block parameters. The trip test, the step and the accumulate are ordinary
  instructions. Nothing about a loop needs reserved registers. A fold inside
  a fold is a loop inside a loop. The lane fold is a loop executed by lanes,
  with no counter and no back edge.
- **Lives:** decided in `docs/codegen/2026-09-10-a-surviving-reduce-is-a-loop.md`
  (R1: "the accumulate is an ordinary `Binary` def … the allocator learns
  nothing about folds"); selected as blocks by `Selector::fold`
  (`pixelflow-codegen/src/emit/select.rs`, B3 of
  `docs/archive/plans/2026-10-08-selection-is-a-phase.md`), whose
  `Loop::trips` is a `Trips`: the body's runs per call when every bound is a
  literal, an estimate otherwise. The verbs that once emitted it inline
  (`Reduce` arm, `Scratch::REDUCE_TEMPS`) were deleted with the legacy driver
  (D1).

### Block. Homonym

- **Is:** a label, its parameters, and a sequence of instructions ending in
  one whose label operands are the block's successors. Other senses: an
  arm block (an arm region that owns a value, placed immediately before its `If`), a `kernel!`
  block of items, and a program's uniform block.
- **Is not:** a run found by permuting a flat schedule. A block's place is
  not computed: it is the `If` it belongs to.
- **Follows:** a value live into a block from more than one predecessor is a
  block parameter; that is what a phi is. "Emit the arms as blocks and there
  is nothing to search for."
- **Lives:** `Block`, `Function` and `build::Builder`
  (`pixelflow-codegen/src/emit/{mod,build}.rs`), built by B1 of
  `docs/archive/plans/2026-10-08-selection-is-a-phase.md`.
  `pixelflow-codegen/src/program/layout.rs`'s blocks have no label,
  parameters or terminating instruction; selection makes the `Block`s from
  them (D3).

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
- **Lives:** `emit::asm::Label` (`pixelflow-codegen/src/emit/asm.rs`) — a
  `u64` minted by `Labels::mint`, its only constructor, owned by the program
  (`Function::labels`). Selection mints a block's label with the block, and
  the constant pool's with the pool. Nothing is derived from the id of what
  it names. History: R0 of
  `docs/codegen/2026-09-10-a-surviving-reduce-is-a-loop.md` made a label an
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
  instruction. The e-graph keeps a shift's count out of its operands (it is
  a `ShiftCount` parameter of `ENode::Shift`), so nothing there re-pins it
  after extraction; codegen's operand kinds still have no type for an
  immediate, and `shift_immediate` checks the `Const` at lowering.
- **Lives:** `Operand`, `Access` and `operands` (`emit/mod.rs`), built by B1
  of `docs/archive/plans/2026-10-08-selection-is-a-phase.md`. The schedule still has the
  split below: `program::operands()` yields only vector reads, and the base
  comes from a separate `pointer_operand()` (both in
  `pixelflow-codegen/src/program/mod.rs`); selection joins them
  (`all_operands`). A shift's count is `ShiftCount` in
  `pixelflow-ir/src/shift.rs`, the one definition of a valid count that the
  constant folder, the e-graph's `ENode::Shift` (`pixelflow-search/src/egraph/node.rs`)
  and `shift_immediate` all ask; the e-graph's direction tag and its reading of
  the arena's spelling are in `pixelflow-search/src/egraph/shift.rs`.

### Instruction

- **Is:** an operation of the machine applied to operands: a value, chosen by
  instruction selection, encoded last.
- **Is not:** bytes written into a buffer by the code that chose it.
- **Follows:** every register an instruction touches is one of its operands —
  including the ones it clobbers and the flags — so nothing about it is hidden
  from the allocator. A branch is an ordinary instruction.
- **Lives:** the backends'
  instruction enums: `Gp<S>`, x86's general-register instructions, generic
  over a `Stage` and with named fields typed by `Class` (`emit/x86_64.rs`);
  `Inst<S>` in `emit/avx2.rs` and `emit/avx512.rs`, the VEX and EVEX
  instructions, over the same vocabulary, and `avx2::Op<S>`, the instruction
  AVX2's selection puts in a block; `Inst` in `emit/aarch64.rs`. The encoders are
  written for the allocator's `Bound` stage, the only one with registers placed.
  ("Instructions: bytes written inline by ~236 functions, not values",
  denotational-diagnosis, described the legacy `LegacyBackend` verbs, deleted
  at D1.)

### Branch

- **Is:** "A branch is an ordinary instruction": `Gp::Jmp { to }` and
  `Gp::Jcc { cond, flags, taken, next }` and `Gp::Fallthrough` on x86; `B` and `CbzFar` on
  aarch64. The condition is the opcode's own field (`Cond`, whose
  discriminants are the manual's values).
- **Is not:** something that returns a position. Not hand-picked mnemonics
  over a private `jcc(code, cc: u8)`.
- **Follows:** a loop's back edge is `Gp::Jmp { to: top }`. Arm guards are one
  verb, `IsaBackend::branch(.., Test, Edges)`, with one test per arm: `any(m)`
  for the true arm, `!all(m)` for the false arm (note §1). A fold's trip test
  reuses `IfArm::False`'s test.
- **Lives:** `pixelflow-codegen/src/emit/{x86_64,aarch64}.rs` (the
  instructions), `IsaBackend::{branch, jump}` (`emit/mod.rs`, implemented in
  `emit/{avx2,avx512,aarch64}.rs`), `IfArm` (`program/mod.rs`);
  a-surviving-reduce-is-a-loop R0.

### Assembly program

- **Is:** one per kernel: a value made of sections, each a sequence of
  items — instructions, and label bindings — with the constant pool as a data
  section. It is the namespace of its labels, and it mints them. "Not an
  AST — assembly is not context-sensitive."
- **Is not:** a buffer of bytes; split per scope and spliced together (one
  program, one namespace — a scope is the allocator's concept, not the
  assembler's). R0's "two front ends, one mechanism" (a value `AsmProgram`
  beside a push/bind/finish `Assembly`) is what this is now: the builder
  accumulates the value.
- **Follows:** whatever builds the program — instruction selection, then the
  allocator inserting its own code — builds a value; nothing writes bytes.
- **Lives:** `asm::AsmProgram` (`pixelflow-codegen/src/emit/asm.rs`) is the
  value: `text` and `data` sections of `Item`s, and the `Labels` that minted
  every label in it. The constant pool is its data section. The allocator
  builds the value from the blocks (`Allocated::program`); the push/bind/finish
  `Assembly` front end was deleted with the legacy driver (D1).

### Assembler

- **Is:** a function from an assembly program to binary: lay the sections
  out, map each label to an address, encode each instruction given the
  addresses of the labels it names. Here, a very small in-memory one: no
  object files, no relocations or symbols beyond labels.
- **Is not:** stateful; a builder; a code buffer other code writes bytes
  into; aware of registers, values, scopes or the IR. Not a checker of
  meaning: it accepts exactly the legal programs. What the machine refuses,
  such as an operand of the wrong register class or an immediate the
  encoding cannot hold, its types or its result refuse. What the machine
  accepts, it accepts. `and` of a mask with `1.0` is legal NEON, and refusing
  it is the IR's job.
- **Follows:** it stands alone — its module imports nothing from the rest of
  the crate, and everything above it depends on it, never the reverse.
- **Lives:** `asm::assemble` (`pixelflow-codegen/src/emit/asm.rs`), which
  imports nothing from the crate (`scripts/check_emit_boundary.py` rule A) and
  is handed the encoder as a closure, so it names no instruction type. It is
  called once, by the `Assemble` pass, on the program the allocator builds.
  `docs/archive/designs/assembler-as-functor.md`.

### Emitter

- **Is:** the last stages (`Select`, `Allocate`, `Assemble`), handed a
  finished `ScopedSchedule<Flow>` by the front of the chain (`Legalize` to
  `Layout`). "The emitter
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
- **Lives:** deleted. A fold is a loop of blocks (see Loop).

### ISA tier (vector width)

- **Is:** "The ISA tier is a property of the process, not of the build: a
  fact the CPU reports, read once." `Isa` has three inhabitants: `Avx2`
  (AVX2+FMA, the floor), `Avx512` (needs avx512f+dq) and `Neon`. The backend
  is a function of the tier, and the width is that backend's register file
  (`B::FILE.vector_bytes()`), read publicly through `jit_vector_bytes()`.
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
  (`pixelflow-codegen/src/isa/mod.rs`), `emit::native`
  (`pub(crate)`, `emit/mod.rs`, which calls `detect()` itself);
  `2026-09-22-the-isa-is-decided-at-startup.md`; CLAUDE.md "SIMD Backend
  Selection".
