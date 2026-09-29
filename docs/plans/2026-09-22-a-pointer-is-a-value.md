# A pointer is a value

## Metadata
- **Author**: JP (decision), Claude (draft)
- **Status**: `Done` — decided 2026-09-22; one PR
- **Created**: 2026-09-22
- **Verified against**: `735cb9de` (main with #1286 and #1289)
- **Continues**: [register-allocation-escape-hatches](2026-09-01-register-allocation-escape-hatches.md)
  (no register allocation outside the register allocator) and
  [collapse-is-a-fold](2026-09-16-collapse-is-a-fold.md) step 6, whose
  measurement left 35 loads of one pointer per batch.

**Decision it records (JP, 2026-09-22):**

> *"Let the register allocator allocate the registers, and it should know
> what a pointer register is, and our assembler should require one."*

The assembler already requires one: `Mem`'s base is a `PtrReg`, and a `Gpr`
holding an index cannot be passed as an address. What was missing sat above
it. The allocator had no pointer class at all — `RegisterFile::gpr_scratch`'s
own doc: *"nothing here ever carries a value across instructions, no
GPR-class liveness, no spilling and no eviction"* — so every `Gather`,
`Broadcast` and `Uniform` reloaded its buffer base from the context into
instruction scratch, at every read. The `8`@32 glyph after step 6 spends 75
`mov rax, [rdi + slot*8]` per batch on three pointers that never change.

---

## 1. The denotation

A value in a schedule has a **class**: it is a vector of `f32` lanes, or it is
an address. The class is a function of the defining op, and every consumer
of a value knows which it expects by position.

- **`ScheduledOp::Context(k)`** is the `k`-th pointer of the context the
  kernel is called with: a buffer's base for `k` below the buffer count, the
  link's uniform block and the origin block after. Pointer class, no
  operands, variance `CONST` — so placement puts it in the per-call scope
  and `plan_carries` ranks it like any other root, weighted by its reads
  times the trips of every scope reading it. A base read once per trip of
  the innermost fold outranks nearly everything.
- **`Gather(idx, base)`, `Broadcast(idx, base)`, `Uniform(base, offset)`**
  take the pointer as an operand, in place of the `slot`/`ctx_slot`
  immediates they carried. The `Buffer` leaf, which used to be a dead
  `Const(0.0)` placeholder folded into the immediate, *is* the `Context` def.
  A `Uniform`'s block has no arena node, so `arena_to_schedule` synthesizes
  one `Context` def per block on first use.
- **`Where::Ptr(PtrReg)`** beside `Where::Reg(Reg)`: a pointer value lives
  in a pointer register, in a slot, or nowhere else. `ResolvedOp` types the
  base operand `PtrReg`, so the assembler's requirement reaches the
  schedule: a vector register cannot be handed to a memory operand, and a
  wrong class at resolution is a panic naming the allocator, never a silently
  wrong address.

## 2. The allocator

**One algorithm, two pools.** The classes never compete for a register, so
[`LinearScan`] runs its one forward pass twice over one schedule, once per
class, and the two answers are merged: the vector pass sees vector defs and
vector operands, the pointer pass sees pointer defs and pointer operands, and
each is blind to the other. Everything the pass does — Belady eviction with
splitting, kept reloads, the destination contest, the pre-emptive eviction at
a `Reduce` or `Guard`, the arm-bounded reload ranges — applies unchanged to a
pointer. What does not apply is the vector-only scratch: temps, guard
registers, the result role, `operand_sources`. The pointer pass has one role,
`Scratch::ptr_reload`, because no instruction reads more than one pointer.

`RegisterFile::pointers: GprSet` is the pool: the caller-saved GPRs the ABI
and the backend's own encodings leave free — `r9`–`r11` on x86-64 (`rdi`,
`rsi`, `rdx` carry the ABI, `rax`/`rcx` are instruction scratch, `r8` anchors
the constant pool), `x3`–`x8` and `x12`–`x15` on aarch64. `checked` proves
the pool misses the ABI registers and `gpr_scratch`. Callee-saved GPRs are
not in it yet: adding them is a prologue/epilogue change and a number to
measure, not a design question.

**Carries and parks per class.** `plan_carries` ranks every root of every
scope by the same weight it always did and keeps one budget per class; a
pointer carried across a fold takes one register from the pointer pool of
every scope inside, as a vector carry takes one from the vector pool. A
root the plan does not carry is parked in a slot — a frame slot like any
other, at the vector stride, stored and loaded through the backend's
pointer forms.

## 3. What is subtracted

- `slot: u16` on `Gather`/`Broadcast`, `UniformLoad { ctx_slot, offset }`.
- `MovLoadPtr` per read, `emit_load_ptr_from_ctx`, `GatherScratch::{base_gpr,
  ctx_gpr}`, `BroadcastGprs::{base, ctx}`, `RegisterFile::gpr_ctx`'s role in
  every encoder but `Context`'s own.
- One GPR from every gather's, broadcast's and uniform's scratch count.

## 4. Measure

`8`@32, this host's 128-bit tier, the mnemonic histogram of
[collapse-is-a-fold](2026-09-16-collapse-is-a-fold.md) §1, against `735cb9de`:

| | before | after |
|---|---|---|
| `mov base, [rdi + slot*8]` | 45 per batch (35 + 8 + 2, one per read) | **3** per call (one per pointer) |
| bytes | 5,160 | **4,376** |
| instructions to `ret` | 1,199 | **1,041** |

The three pointers are read in the folds and carried: the pool has two
registers above its floor and the origin block, read only by the body, is
never a candidate. One `mov r10, r9` hands a def to its carry. Pinned by
`pointer_class::a_context_pointer_is_loaded_once_per_call` (the x86 bytes,
on every host) and `a_base_read_inside_a_fold_is_carried_into_it` (the
placement, on every backend).

## 4½. What the fold carve had to stop doing

The first allocation parked the `Uniform`s' base in a slot for a fold that
never read it. `extract_folds_bound_by` walked a fold's closure *through*
every hoisted value into its operands, so a body-computed `Uniform` dragged
its `Context` in as a placeholder no def in the fold read: a root with zero
reads, which `plan_carries` rightly refuses and `park_roots` then stores,
once per call, for nobody. The walk now ends at a value the enclosing scope
computes — what such a value is built from is that scope's business. A
`Guard` is the one exception, because `stays_put` has a fold emit one
wherever it reaches it, so its mask is the fold's to read.

## 5. Not here

- The fold binders as integers in GPRs (the `cvttss2si` per address). Same
  class, same machinery, its own measurement.
- The constant pool's anchor as a pointer value rather than a pinned `r8`.
- Callee-saved GPRs in the pool.
