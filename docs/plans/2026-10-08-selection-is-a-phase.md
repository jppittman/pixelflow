# Selection is a phase

## Metadata

- **Author**: JP (design), Claude (draft)
- **Status**: `Plan of record`. This is revision 3.
  - Revision 2 answered three reviews (allocation, closure and landing), and Appendix A records what was done with each point.
  - Revision 3 carries JP's decisions of 2026-10-08, which override anything below that disagrees with them (§0).
- **Created**: 2026-10-08
- **Verified against**: `origin/main` at `bdee3900` (#1335). At `bdee3900`, `pixelflow-codegen` is the same as at `fc1da611` except for `emit/coverage.rs`.
- **Assumes the narrowing lands first.** The narrowing is worktree `/home/user/pixelflow-narrow`, branch `codegen-narrow-emit`, at `2da80bba` plus uncommitted work. It is based on `9bd3df6c` and must be rebased over #1330–#1335. After it lands:
  - `emit/`'s submodules are private.
  - The register newtypes have private fields.
  - `EmitCtx` is gone. There is no register cap any more, and pressure tests use wide kernels and the test-only `AtFloor` backend.
  - Only `compile`, `CompileResult`, `CompiledKernel`, `EmitTraffic` and `ScopeTraffic` leave the crate.
- **Citations**: written as `file:line` at `bdee3900`, relative to `pixelflow-codegen/src/`, and always with the symbol. The narrowing moves every line in `emit/`, so the symbol is how a moved line is found.
- **Ontology**: `.claude/skills/ontology`, as of #1334 (JP), plus its `codegen.md` part. Wherever this plan meets an ontology entry (Label, Assembly program, Assembler, Operand, Register, Temp, Loop, Block, Spill), the plan follows the entry and cites it.
- **Supersedes**: the scratch loop plan's §6, specifically L1–L5 and its §6.4 lowering (`Acc`, `Phi`, `Loop`).
  - In this design, selection builds the loop's per-iteration code (the latch), so `program/` needs no loop change.
  - Labels follow the ontology, not L1's keyed enum.
- **Continues**:
  - [register-allocation-escape-hatches](2026-09-01-register-allocation-escape-hatches.md). Its goal was "no register allocation outside the register allocator". This plan reaches it by removing the need for predictions.
  - [a-surviving-reduce-is-a-loop](2026-09-10-a-surviving-reduce-is-a-loop.md) and [a-kept-structure-is-control-flow](2026-09-10-a-kept-structure-is-control-flow.md): a loop is a block that branches back to its own label.
  - [emit-should-just-emit](2026-09-12-emit-should-just-emit.md): encoding is last and makes no decisions.
  - [collapse-is-a-fold](2026-09-16-collapse-is-a-fold.md): every kernel is a loop nest.
  - [a-pointer-is-a-value](2026-09-22-a-pointer-is-a-value.md): an address is a value of its own class, and the assembler requires one.
  - [demand-is-a-dag-property](2026-09-07-demand-is-a-dag-property.md): an arm's region is read off the structure.

**The request (JP, 2026-10-08):**

> *"And like, idk how you designed the register allocator, but if id done it,
> registers wouldn't be copy and I'd be using the rust ownership semantics
> extensively."*

**The diagnosis (agreed with JP).**

Today's pipeline is `IR → schedule → register allocation → emit`. `emit` does three jobs: it selects instructions, encodes them, and makes whatever register choices are left over.

So the allocator allocates for instructions that do not exist yet. `RegisterFile` carries five predictions of what selection will need:

- `temps_for`
- `gpr_temps_for`
- `mask_temps_for`
- `guard_temps`
- `mask_guard_temps`

Every register chosen outside the allocator is a place where one of those predictions ran out. The fix is to add the missing phase.

**Three departures from the letter of the binding design.** Each keeps its substance. JP should confirm each one.

1. **Labels are minted, not keyed by what they name.**
   - The task text says labels are keyed by what they name. JP's later #1334 says the opposite.
   - The ontology's Label entry rejects "a sum of the things that can be labelled". It records that minting the label together with its block "removes [the site-to-label map] without the label knowing about nodes".
   - This plan follows #1334.
2. **Register tokens are minted once per compile, not once per process.**
   - The register-file *declaration* is still fixed once, at startup, by `isa::detect`.
   - The tokens are minted from that declaration, by the only constructor, once per compile.
   - Why per compile:
     - `jit_cache` deliberately compiles outside its lock so that distinct kernels do not serialize (`jit_cache.rs:176-179`).
     - With one process-wide pool, either every compile serializes behind it, or the pool is shared read-only. If it is shared read-only, uniqueness moves into per-allocation leases anyway.
     - So the per-allocation lease set is where uniqueness lives in every variant.
     - A thread-local pool would only add a `RefCell` panic and a `!Send` marker.
3. **x86 reads constants RIP-relative, so there is no pool-base value on x86.**
   - The binding design says the pool's address is `lea`/`adrp` with a label operand. That is still true on aarch64.
   - On x86, a constant's address operand holds the label itself: "an address with a label inside", which is one of the binding design's operand kinds. So there is nothing to put in a register.
   - The cost: `[rip + disp32]` is never longer than `[r8 + disp32]`, the `lea` disappears, and a general register comes back.
   - If JP prefers the letter of the design, B8 selects `LeaLabel` plus `[p + disp32]` instead. That is one selection arm and changes nothing else.

---

## 0. JP's decisions of 2026-10-08 (revision 3)

These are binding. Where a later section disagrees, this section wins, and the implementer fixes the later section in the same commit.

1. **How a lane value is held is the backend's associated type.** `IsaBackend::Lane: Copy` is what one IR lane value is on this machine:
   - AVX2 and NEON: `Value<Vector>`. They have one file for lanes, and a mask is a vector there; their instructions take nothing else.
   - AVX-512: `enum { Vector(Value<Vector>), Opmask(Value<Opmask>) }`. A comparison selects `vcmpps k` and yields an `Opmask` lane; `BitAnd`/`BitOr` of two `Opmask` lanes select `kandw`/`korw`.

   Consequences:
   - The driver binds each IR value to a `B::Lane` and hands lanes back to the backend. It never asks whether a value is a mask: only AVX-512 knows that a lane can live in `k`. A block parameter takes its lane's class (`IsaBackend::lane_class`).
   - There is no conversion, as a concept or as a trait method. Where an IR value is read in the other file, AVX-512's selection picks an instruction that reads it where it is:
     - an `If` whose condition is an `Opmask` lane is `vblendmps zmm{k}`; on a `Vector` lane it is `vpternlogd 0xCA`, today's blend;
     - a guard on an `Opmask` lane is `kortestw`; on a `Vector` lane it is `vptestmd`, then `kortestw`;
     - an `Opmask` lane read by an instruction with no `k` form (a mask written out, or used as bits) is first `vpmovm2d`, the instruction that writes lanes from `k`.
     Each is an ordinary instruction over values.
   - Why the other file is ever read: the IR has one lane type, so it lets a comparison's result go anywhere a number goes, and any value be a condition. Today's production masks come from comparisons, combine with `and`/`or` (`fonts/loop_blinn.rs`, `fonts/cache.rs`, `scene3d.rs`) and end in an `If`, so on AVX-512 they stay in `k` throughout. `BitAnd`/`BitOr` on data, such as `ln`'s mantissa extraction in `pixelflow-ir`'s `passes.rs`, are vector instructions.
   - So the `vcmpps k1` → `vpmovm2d` round trip is gone wherever a compare feeds a blend, a guard or mask logic.
   - The class named `Mask` in revision 2 is renamed `Opmask` (file `OpmaskFile`).
2. **The assembler accepts exactly the legal programs.** Its types refuse what the machine refuses, such as an operand of the wrong class or an immediate the encoding cannot hold. They refuse nothing the machine accepts.
   - "A mask is not a number" is the IR's question, not the backend's. On NEON and AVX2 the machine cannot tell the two apart.
   - So no `LaneMask` class over `VectorFile` is built (§6 keeps that refusal).
3. **Labels:** an opaque `Label(u64)`, minted by the one program a kernel has. The program owns the mint (`Labels`): the `Builder` while selecting, then the `AsmProgram`. A thing *has* a label: a block, the pool section, a pool entry. There is no label enum, no scope and no key. This matches §2.8.
4. **The assembler stands alone.** `emit/asm.rs` imports nothing from the crate, and `scripts/check_emit_boundary.py` gains that rule, with a self-test case (A8). It is a stateless function from program to binary. There is no per-scope splicing of bytes: A8's front end threads one program through every scope, so nothing is spliced, and the selection pipeline builds one program.
5. **Fixed registers: requirement versus choice.**
   - A register the machine or ABI requires (the entry arguments, `sp`) is a constraint on a value, and the allocator satisfies it.
   - A register the hardware would accept any member of the class for (`x16`, `x17`, `r8`, `w16`, `eax`, `k1`) is the allocator's choice. No exception.
6. **Tests drive the production API, and production decides the API.**
   - No `#[cfg(test)]` item, accessor, constructor, counter or sample table is added to `src/` for a test. New tests live in `pixelflow-codegen/tests/` (or the consuming crate's `tests/`) and observe `compile`, `CompileResult`, `CompiledKernel` (values, `code_bytes`) and `EmitTraffic`.
   - This revises B1's and B2's unit tests and B10's exhaustive encoder test (`#[cfg(test)] fn samples()`):
     - an invariant that no production input can break is asserted in production code, not unit-tested through a back door;
     - the register coverage B10 wants is a pressure kernel per tier, whose live values outnumber each file's members, checked by its values.
   - `GOLDEN_SELECTED` lives in `tests/`, built through the production API.
   - Existing in-file tests that a commit touches may stay where they are. Each commit's tests must still fail when the commit's change is reverted.
7. **CI: no new required check.**
   - Branch protection cannot be edited from here. The selection-pipeline legs are steps in jobs that are already required: the `isa-matrix` job for x86 tiers, and the `test` matrix job on macOS for NEON.
   - Each new step in the `test` matrix job carries the docs-only guard (CLAUDE.md, "CI is the gate").
8. **Less code.** The series ends with `pixelflow-codegen/src` smaller than at its start, the post-narrowing tree. Lines are counted over tracked `*.rs` files.
   - Phases A–C may grow it, because the new pipeline is built beside the old one.
   - D1–D3 must bring it below the starting count. If they do not, the series is not done: subtract more before declaring it.
   - Every commit message states its line delta.
9. **NEON runs locally.** The gate **Q** (§5) runs the NEON suites under `qemu-aarch64`. So M (macOS CI) confirms; it no longer discovers.
10. **Departures 2 and 3 above are accepted:** tokens are minted per compile, and x86 constants are RIP-relative, with no pool base. Departure 1 is §0.3.

## 1. What each phase is

This section is written from first principles, without reference to the code.

### 1.1 Instruction selection

```text
select : (a scoped schedule) → a machine function over values
```

**What it guarantees.** For every assignment of distinct storage to the function's live values that respects the operand constraints, running the function computes what the IR denotes.

**What it decides.** Which instructions the machine runs, and in which blocks.

- It is the only phase that knows that an `Add` is `vaddps`.
- It knows that a constant FMOV cannot encode is a pool load, and that a NEON gather is four lane loads.
- It knows that a uniform past `imm12`'s reach on aarch64 is address arithmetic.
- It builds a fold's per-iteration code from ordinary instructions: the trip test, the step and the accumulate.
- It builds an `If` guard the same way, from ordinary instructions.

**What it owes.** Every register an instruction touches is one of its operands, and every register operand is a value with a definition, reads and a class. This includes the flags it destroys and the mask a gather zeroes. A "scratch register" is not a concept here: it is a value with one definition and a short life (see the ontology entry Temp).

**What it never decides.** A physical register. Where the hardware dictates something about registers, selection states it as a *constraint* on the operand:

- **Tied**: this definition overwrites that read in place.
- **Early**: this definition is written before the reads are done.

The ABI's argument registers are a requirement, not a choice. They are stated on the entry block's parameters (see the ontology entry Register, "Requirement versus choice").

**Selection is structured.** A loop is a contiguous run of blocks whose last block branches back to its first. A forward branch skips a contiguous run of blocks within one scope. `Builder::finish` asserts both. This is what lets the allocator compute liveness as intervals in layout order (§1.3).

### 1.2 The machine IR

A function is a sequence of blocks in layout order.

**A block** is a label, its parameters, and its instructions.

- The label names a position. It is minted together with the block.
- The parameters are values defined on entry to the block. A parameter is what a phi is.
- The block's successors are the target operands of its last instruction. A block with no successors is the function's exit, and there is exactly one.

**An operand** is one of:

- **a register**: a value with an access (`Read`, `Write`, `Early`, or `Tied` to a read);
- **a target**: a label, plus the values passed to the target block's parameters;
- **an immediate**;
- **an address**: register operands inside it, and possibly a data label;
- **a frame slot**.

**A conditional branch has two targets**, `taken` and `next`.

- `next` must be the block laid out next.
- Its encoding is a zero-width field that the assembler checks lands exactly at the end of the instruction (§1.4).
- So every successor is an operand, and a fall-through costs nothing.
- An unconditional transfer to the next block is the instruction `Fallthrough { to }`, which encodes to nothing. Selection chooses it, because selection emits the blocks in their layout order.

**Two instructions that differ only in physical registers are the same instruction.**

**The flags are a register class with one member.** A compare defines a flags value and a conditional branch reads it. "Nothing clobbers the flags in between" is therefore a liveness fact, not an adjacency convention.

### 1.3 Register allocation, frame layout and spill code

```text
allocate : machine function over values → machine function over registers and slots
```

**What it guarantees.** At every program point:

1. The map from live value to location is injective.
2. Every constraint holds.
3. Every instruction the allocator inserts (spill, reload, remat, copy) preserves each value from its definition to every read of it.

**Registers and frame slots are resources.**

- A register is leased to at most one live value at a time. A frame slot likewise.
- The frame is a second register file. It is unbounded, laid out by the allocator, and capped at `MAX_FRAME` (2 MiB).

**A spilled value is stored right after its definition.** It is never stored at the eviction point, because a guard can skip that point (see the ontology entry Spill). The allocator inserts the store retroactively when it first evicts the value. From then on, the value's slot is valid on every path from its definition.

**Liveness is intervals in layout order, plus two rules.**

- A value's interval runs from its definition to its last read.
- A value live into a loop's head and read inside the loop is extended over the whole loop.
- Liveness is never propagated backwards through a branch. That is what makes the one relaxation in §2.9 sound: a value defined inside an arm and read by its `If`'s blend is not live before its definition.

**Three invariants hold at control-flow joins. The allocator asserts all three.**

1. **At a loop head.** A value live across the loop has one location for the loop's whole extent:
   - either it keeps its register lease from the head to the latch,
   - or its home is its slot, and its reloads inside the loop are split values that die before the latch.
2. **At a forward join.** For each live value defined on every incoming path, its location is the same on every path; otherwise the allocator drops its residency and it is read from its slot. A value undefined on some path takes its location from the paths where it is defined.
   - This is the intersection of the predecessors' states.
   - On a guarded arm, it gives today's rule: a reload made inside the arm dies at the arm's end.
3. **No `Flags` value is live at a label.**

**The allocator writes its own code.**

- It asks the backend for a copy, a spill or a reload at the slot it chose. When the offset does not encode, the backend answers with address arithmetic over a fresh `Pointer` value, which the allocator allocates on the spot.
- The frame lays out the narrow classes (`Pointer`, `Integer`, `Opmask`, 8 bytes each) first, sized from their peak live count. So every narrow slot encodes on every backend: x86 `disp32` and aarch64's scaled `imm12` (4,095 × 8 bytes).
- The vector region starts at `align_up(8·N, max(16, vector_bytes))`.
- Therefore:
  - a narrow spill never needs a temporary;
  - a deep vector slot needs exactly one `Pointer` temporary, which lives across one instruction and is never spilled (asserted);
  - freeing a register for that temporary costs at most one narrow spill, which needs no temporary.
- There is no fixed point to iterate to.

**The allocator's own instructions never write the flags.** Their rights mint only `Spill` classes, which excludes `Flags`. So a copy placed between a `Test` and its `Jcc` is safe because of the types.

**Allocation is total.**

- It returns `Err(BudgetExceeded)` when the frame outgrows `MAX_FRAME`, or when the narrow region would pass 4,095 slots.
- It panics, naming the instruction, when the function cannot be allocated: for example, an instruction that holds more registers of a class than the class has. That is a selection bug, never a fact about a kernel.
- It never picks a register outside its leases.

### 1.4 Assembly

The assembly program (ontology: Assembly program) is one per kernel. It is made of sections of items:

- label bindings;
- instructions;
- alignment;
- bytes.

The constant pool is a data section with its own label, and every entry in it also has a label. The program is the namespace of its labels, and it mints them.

The assembler (ontology: Assembler) is a function:

```text
assemble : assembly program → binary
```

It lays out the sections, maps each label to an address, encodes each instruction, and patches each label field. It imports nothing from the rest of the crate.

**Who owns a label field.** A label field belongs to the instruction that contains it, and the instruction's encoder supplies its patch function:

- x86: `rel32`;
- aarch64: `imm19`, `imm26`, and `adrp`'s page plus `add`'s low 12 bits;
- the zero-width fall-through check.

**Block order.** The order is selection's. A function's blocks are already in layout order when they reach the assembler. The allocator never reorders them.

### 1.5 Encoding

```text
encode : one allocated instruction → bytes, plus label fields
```

Encoding is total and makes no choices.

- Selection proved every immediate encodable. If one does not fit, the encoder panics and names the instruction.
- Choosing between encodings that differ only in which register is involved is the encoder's business, not a register choice. Example: `cmp al, imm8` versus `cmp sil, imm8` (which needs REX) versus `cmp r9b, imm8`.

---

## 2. The types

**Where things are declared.**

- `emit/mod.rs` is the contract. It holds the traits, the types they mention, and `pub(in crate::emit) use` lines.
- A type whose privacy carries a guarantee is declared in a *leaf* module and re-exported. Rust lets every child module see the private fields of a type declared in `emit/mod.rs` (landing B1), so a leaf is the only place a private constructor is really private.
- Nothing here is `pub(crate)`. Nothing outside `emit` uses any of it.

| File | Contents | Leaf? |
|---|---|---|
| `emit/mod.rs` | the contract | — |
| `emit/asm.rs` | `Label`, `Labels`, `AsmProgram`, `Item`, `Encoding`, `assemble`. Imports nothing from the crate | yes |
| `emit/build.rs` | `Def`, `Early`, `Tie`, `Builder`, `Pending`, `Spiller` | yes |
| `emit/select.rs` | the generic selection driver, with its scoped `Bindings` | — |
| `emit/regalloc/mod.rs` | `RegisterAllocator`, `LinearScan`, `Allocated` | — |
| `emit/regalloc/resource.rs` | `Reg`, `Pool`, `Lease`, `In`/`Out`/`InOut`, `FrameSlot`, `Frame`, `SlotLease` | yes |
| `emit/regalloc/policy.rs` | `EvictionRank`, carry pricing (retargeted from today's `LinearScan`) | — |
| `emit/{x86_64,avx2,avx512,aarch64}.rs`, `emit/aarch64/table.rs` | the backends | — |

### 2.1 Register files and value classes

```rust
/// A physical register file of the machine. Sealed: these four are all any
/// target here has. A backend whose machine lacks one declares it empty, and
/// no instruction of that backend has a field in it.
pub(in crate::emit) trait File: sealed::File + 'static { const ID: FileId; }
/// `ymm`, `zmm`, `v`.
pub(in crate::emit) enum VectorFile {}
/// The 64-bit general-purpose registers.
pub(in crate::emit) enum GeneralFile {}
/// AVX-512's `k1`–`k7`. `k0` is not a member: an EVEX `aaa` of 0 means "no mask".
pub(in crate::emit) enum OpmaskFile {}
/// x86 `EFLAGS`, aarch64 `NZCV`: one member.
pub(in crate::emit) enum FlagsFile {}
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::emit) enum FileId { Vector, General, Opmask, Flags }

/// What a value is at the machine level, and the file it lives in.
///
/// A class is a type, not a register file: `Pointer` and `Integer` share
/// `GeneralFile`, and an address operand's base is a `Read<Pointer>`, so a
/// truncated binder index or a `row·pitch` product cannot be passed as an
/// address. This is the guarantee `Mem.base: PtrReg` gives today
/// (`aarch64/table.rs:486-497`, x86 `Mem<D, P = PtrReg>` `x86_64.rs:956`), and
/// that JP asked for ("our assembler should require one",
/// a-pointer-is-a-value).
pub(in crate::emit) trait Class: sealed::Class + Copy + 'static {
    type File: File;
    const ID: ClassId;
}
/// One batch of `f32` lanes; on AVX2 and NEON also a mask.
#[derive(Copy, Clone, Debug)] pub(in crate::emit) enum Vector {}  // File = VectorFile
/// An address.
#[derive(Copy, Clone, Debug)] pub(in crate::emit) enum Pointer {} // File = GeneralFile
/// An integer: an index, a product, a movemask's bits.
#[derive(Copy, Clone, Debug)] pub(in crate::emit) enum Integer {} // File = GeneralFile
/// An AVX-512 opmask: where that backend's comparisons put their lanes.
#[derive(Copy, Clone, Debug)] pub(in crate::emit) enum Opmask {}  // File = OpmaskFile
/// The condition flags.
#[derive(Copy, Clone, Debug)] pub(in crate::emit) enum Flags {}   // File = FlagsFile
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::emit) enum ClassId { Vector, Pointer, Integer, Opmask, Flags }

/// The classes a frame slot can hold. `Flags` cannot be stored, so the
/// allocator's own verbs (`IsaBackend::{copy, spill, reload}`) do not accept
/// it: calling one with `Flags` is a type error.
pub(in crate::emit) trait Spill: Class {}  // Vector, Pointer, Integer, Opmask
```

### 2.2 Registers are tokens, and the allocator holds the only leases

These types live in `regalloc/resource.rs`, a leaf module.

```rust
/// A physical register of file `F` on backend `B`. A resource, not a name.
///
/// Not `Copy`, not `Clone`. Its one constructor is `Pool::mint`. Branded by
/// backend, so an AVX-512 token cannot reach an AVX2 encoder: `Vex::rrr`
/// (`avx2.rs:137-150`) would drop bit 4 and encode `zmm20` as `ymm12`.
pub(in crate::emit) struct Reg<B: IsaBackend, F: File> {
    number: u8,
    _brand: PhantomData<(fn() -> B, F)>,
}
impl<B: IsaBackend, F: File> Reg<B, F> {
    /// The number the encoding's register fields take (`ModRM`, `VEX.vvvv`,
    /// `Rd`/`Rn`/`Rm`). A data-plane width the ISA dictates.
    pub(in crate::emit) fn number(&self) -> u8 { self.number }
}

/// Every member of `B::FILE`, minted. One per compile.
pub(in crate::emit) struct Pool<B: IsaBackend> {
    vector: Box<[Reg<B, VectorFile>]>,
    general: Box<[Reg<B, GeneralFile>]>,
    opmask: Box<[Reg<B, OpmaskFile>]>,
    flags: Box<[Reg<B, FlagsFile>]>,
}
impl<B: IsaBackend> Pool<B> {
    /// The only constructor of a `Reg`. Its one caller is `compile_on`.
    pub(in crate::emit) fn mint() -> Self;
}

/// The allocator's exclusive right to one register for one allocation.
///
/// Not `Copy`, not `Clone`. `Leases::new` makes exactly one per member of a
/// pool, and only the allocator can call it. During allocation:
/// - a free list owns a lease;
/// - binding a value moves the lease into the value's state;
/// - eviction and death move it back.
/// So two live values in one register is unrepresentable in the scan's
/// state, and a register chosen anywhere but the allocator does not compile.
pub(super) struct Lease<'m, B: IsaBackend, F: File>(&'m Reg<B, F>);

/// One allocation's leases. The ABI's three are handed out already bound,
/// as initial ownership, so the allocator never looks a register up by
/// number.
pub(super) struct Leases<'m, B: IsaBackend> {
    pub(super) vector: Vec<Lease<'m, B, VectorFile>>,
    pub(super) general: Vec<Lease<'m, B, GeneralFile>>,
    pub(super) opmask: Vec<Lease<'m, B, OpmaskFile>>,
    pub(super) flags: Vec<Lease<'m, B, FlagsFile>>,
    pub(super) entry: EntryLeases<'m, B>,   // ctx, out, pitch
}
impl<'m, B: IsaBackend> Leases<'m, B> {
    /// `pub(super)`: `regalloc` only. Selection, the backends and the
    /// assembler cannot obtain a lease.
    pub(super) fn new(pool: &'m Pool<B>) -> Self;
}

/// A register as one operand of one allocated instruction: a shared borrow
/// of the pool's token, copied out of the lease its value held at that
/// instruction. Three types, so that `walk` cannot put a read where a write
/// belongs. Each is made only by `regalloc`, from a `&Lease`, and each
/// `Deref`s to `Reg<B, C::File>`. They are not `Copy`: each is produced for
/// exactly one operand.
pub(in crate::emit) struct In<'m, B: IsaBackend, C: Class>(&'m Reg<B, C::File>, PhantomData<C>);
pub(in crate::emit) struct Out<'m, B: IsaBackend, C: Class>(&'m Reg<B, C::File>, PhantomData<C>);
pub(in crate::emit) struct InOut<'m, B: IsaBackend, C: Class>(&'m Reg<B, C::File>, PhantomData<C>);
```

**Why the lease, and not a moved `Reg`.** Moving tokens out of a pool while the allocated program holds `&'m Reg` borrowed from the same pool does not borrow-check (allocation F1, closure 17). So:

- the tokens never move;
- the leases move;
- an allocated instruction holds the shared borrow it took from a lease at that point.

The typed guarantee is therefore: one lease per register per allocation, and every bound operand made from the lease its value held at that instruction. Exclusivity is a property of the scan's state. The allocated program holds shared borrows, so §2.13 lists what remains a runtime check.

### 2.3 The register-file declaration

```rust
/// What a backend's register file *is*: the allocatable members of each
/// file, by hardware number, and where the ABI puts the three arguments.
/// Numbers only; nothing here is a register until `Pool::mint`. One `const`
/// per backend. It says nothing about what an instruction needs: selection
/// runs first, so the allocator reads that off the function.
///
/// A register outside every list belongs to the platform or the caller:
/// callee-saved registers (x86 `rbx rbp r12–r15`, NEON `v8–v15`, aarch64
/// `x19–x28`), the stack pointer (owned by `Frame`, §2.4), `x30`, and
/// Apple's `x18`. None of them is a scratch reservation.
///
/// The calling convention is SysV on x86-64 and AAPCS64 on aarch64, because
/// `executable.rs` builds only for Linux and macOS: anywhere else it has no
/// `NativeCodePage` and fails to compile (`executable.rs:134-146`). A Win64
/// port would change these files, because Win64 callee-saves `xmm6–15`.
#[derive(Copy, Clone, Debug)]
pub(in crate::emit) struct RegisterFile {
    pub vector: &'static [u8],
    pub general: &'static [u8],
    pub opmask: &'static [u8],
    pub flags: &'static [u8],
    /// Members of `general` the three arguments arrive in. This is initial
    /// ownership, not a reservation: once a parameter is dead or spilled, its
    /// register is free.
    pub entry: EntryRegisters,   // { ctx: u8, out: u8, pitch: u8 }
    /// Bytes per `Vector` register and per vector frame slot: 16, 32 or 64.
    pub vector_bytes: u64,
}
impl RegisterFile {
    /// Refuse a self-contradictory declaration at compile time:
    /// - files disjoint;
    /// - `entry` names three distinct `general` members;
    /// - `flags` has at most one member;
    /// - `vector_bytes` a power of two, at least 16.
    pub(in crate::emit) const fn checked(self) -> Self;
}
```

| Backend | `vector` | `general` | `opmask` | `flags` | `entry` (ctx, out, pitch) |
|---|---|---|---|---|---|
| AVX2 | `ymm0`–`15` | `rax rcx rdx rsi rdi r8 r9 r10 r11` | — | `EFLAGS` | `rdi rsi rdx` |
| AVX-512 | `zmm0`–`31` | same as AVX2 | `k1`–`k7` | `EFLAGS` | `rdi rsi rdx` |
| NEON | `v0`–`7`, `v16`–`31` | `x0`–`x17` | — | `NZCV` | `x0 x1 x2` |

Notes on the table:

- The vector members are today's (`AVX2_FILE` `avx2.rs:1164`, `AVX512_FILE` `avx512.rs:1233`, `AARCH64_FILE` `aarch64.rs:1881`).
- The general members are the caller-saved registers that today's three hand-split pools (`gpr_scratch`, `pointers`, the pins) partition, now as one file. `r8`, `x16` and `x17` join it.
- AAPCS64's IP0/IP1 matter only across a call, through a linker veneer. A leaf kernel makes no calls (ontology: Register).

### 2.4 The frame

These types live in `regalloc/resource.rs`.

```rust
/// A frame slot: `bytes` at `offset` from the stack pointer. A resource like
/// `Reg`: not `Copy`, not `Clone`, minted only by `Frame`, at an offset fixed
/// for its life.
pub(in crate::emit) struct FrameSlot { name: SlotName, offset: u64, bytes: u64 }
impl FrameSlot {
    /// Bytes from the stack pointer. The control plane is 64-bit. Each
    /// encoder narrows this to its displacement field, and the backend's
    /// `spill`/`reload` turn an offset that does not fit into address
    /// arithmetic.
    pub(in crate::emit) fn offset(&self) -> u64;
    pub(in crate::emit) fn bytes(&self) -> u64;
    pub(in crate::emit) fn name(&self) -> SlotName;
}
/// A slot's name: `Copy`, carried by selected-stage instructions the way
/// `ValueName` is.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::emit) struct SlotName(u64);

/// The allocator's exclusive right to one slot. Affine, with no lifetime:
/// the offset is fixed at mint, so nothing borrows the frame while the scan
/// grows it.
pub(super) struct SlotLease { name: SlotName }

/// One function's frame, and the stack pointer with it.
///
/// - **Layout.** The narrow region (`Pointer`, `Integer`, `Opmask`; 8 bytes
///   each) occupies `[0, 8·N)`, where `N` is the peak number of
///   simultaneously live narrow values, measured by the allocator before the
///   scan. `N > 4095` is `BudgetExceeded`.
/// - The vector region starts at `align_up(8·N, max(16, vector_bytes))` and
///   grows as the scan mints slots, lowest free first.
/// - The frame's size is the vector region's high-water mark. Past
///   `MAX_FRAME` (2 MiB) it is `BudgetExceeded`.
/// - **The stack pointer is the frame's.** Its only readers are frame-slot
///   operands and `SlotAddr` (§2.10). Its only writers are the function's
///   `Enter` and `Ret` instructions. It is in no class.
/// - After the scan, the allocator converts its `&'m mut Frame` into
///   `&'m Frame`, and the allocated function borrows `&'m FrameSlot`s from it.
pub(in crate::emit) struct Frame { /* slots by region, free lists, high-water */ }
impl Frame {
    pub(in crate::emit) fn empty(vector_bytes: u64) -> Self;
    pub(super) fn reserve_narrow(&mut self, peak: u64) -> Result<(), CompileError>;
    pub(super) fn lease(&mut self, class: ClassId) -> Result<SlotLease, CompileError>;
    pub(super) fn release(&mut self, slot: SlotLease);
    pub(in crate::emit) fn slot(&self, name: SlotName) -> &FrameSlot;
    pub(in crate::emit) fn bytes(&self) -> u64;
}
```

### 2.5 Values, and the right to define one

`Value` and `ValueName` are declared in `mod.rs`. `Def`, `Early` and `Tie` are declared in `build.rs`, so their constructors are private.

```rust
/// A name for something computed, of class `C`: one definition, any number
/// of reads. A name, so `Copy`. Names copy; resources move.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::emit) struct Value<C: Class> { id: u64, _class: PhantomData<C> }
/// The same name with its class as data: the allocator's view.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::emit) struct ValueName { pub id: u64, pub class: ClassId }

/// The right to define one fresh value of class `C`.
///
/// Affine: only `Builder::def` mints it, and placing it in an instruction's
/// write field consumes it. A second definition of a value is therefore
/// unrepresentable. Its name is readable (`Def::value`) so that selection
/// can return it. A read before the definition, and an instruction reading
/// what it itself defines, are refused by `Builder::push` at runtime (§2.13).
#[must_use = "a value whose definition is dropped is read but never written"]
pub(in crate::emit) struct Def<C: Class> { value: Value<C> }

/// A definition the instruction writes before it has finished reading: its
/// register is none of this instruction's reads' registers, not even one
/// whose last use is here. Example: `vgatherdps`'s destination.
#[must_use] pub(in crate::emit) struct Early<C: Class> { value: Value<C> }

/// A read that the instruction overwrites in place, and the right to name
/// what it leaves there: `vfmadd231ps`'s addend, NEON `BSL`'s mask, `INS`'s
/// vector, a post-indexed base, `imul`'s destination, a gather's mask.
///
/// The write takes the read's lease. If the read is still live after this
/// instruction, the allocator copies it into a fresh value first and ties
/// the copy instead.
#[must_use] pub(in crate::emit) struct Tie<C: Class> { read: Value<C>, write: Value<C> }
```

A **clobber** is a definition nobody reads, and it holds its lease for exactly one instruction. Examples:

- the `EFLAGS` an `imul` destroys is a `Def<Flags>` with no reads;
- the mask `vgatherdps` zeroes is a `Tie` whose write has no reads.

**Values stay `Copy`.** An affine value would make "read after the last use" a type error. But a value's reads are many and placed by selection, while its liveness is the allocator's to compute. Making names affine would push liveness into selection's types without making any wrong allocation unrepresentable.

### 2.6 Stages: an instruction is generic over what its operands are

```rust
pub(in crate::emit) trait Stage {
    type Write<C: Class>;
    type Early<C: Class>;
    type Read<C: Class>;
    type Tie<C: Class>;
    type Slot;
    type Target;
    type FrameSize;
}

/// After selection: names, and the rights to define them.
pub(in crate::emit) enum Selected {}
impl Stage for Selected {
    type Write<C: Class> = Def<C>;
    type Early<C: Class> = Early<C>;
    type Read<C: Class> = Value<C>;
    type Tie<C: Class> = Tie<C>;
    type Slot = SlotName;      // only instructions the allocator builds carry one
    type Target = Target;      // a label, and the arguments for its parameters
    type FrameSize = FrameSize;
}

/// After allocation: borrowed tokens, borrowed slots, block arguments already
/// placed as moves, and the frame's size.
pub(in crate::emit) struct Bound<'m, B>(Infallible, PhantomData<(&'m (), fn() -> B)>);
impl<'m, B: IsaBackend> Stage for Bound<'m, B> {
    type Write<C: Class> = Out<'m, B, C>;
    type Early<C: Class> = Out<'m, B, C>;
    type Read<C: Class> = In<'m, B, C>;
    type Tie<C: Class> = InOut<'m, B, C>;
    type Slot = &'m FrameSlot;
    type Target = Label;
    type FrameSize = u64;
}

/// The frame's size, as an operand of `Enter`/`Ret`. Selected up front;
/// bound by the allocator once the frame is laid out.
#[derive(Copy, Clone, Debug)] pub(in crate::emit) struct FrameSize;
```

**Each backend declares one instruction enum**, generic over the stage, with named fields.

- **Addresses are per backend.** x86 keeps `Mem<S, D: Disp>`, `Sib4<S>`, `Vsib4<S>` and `RipRel { label }`. aarch64 keeps `Mem<S>` (`[Xn, #imm12]`) and `MemIndexed<S>` (`[Xn, Wm, UXTW #2]`). The shapes the hardware cannot encode stay unrepresentable (#1332, closure 7), and their bases are `S::Read<Pointer>`.
- **Immediates are typed fields.** Examples: `Imm12`, `Imm32`, `Cond`. They are not operands the allocator sees.

```rust
// avx2.rs: representative arms
pub(super) enum Inst<S: Stage> {
    /// `vaddps`, `vsubps`, … `vcmpps`, `vandps`: VEX three-operand.
    Alu { op: VexAlu, dst: S::Write<Vector>, a: S::Read<Vector>, b: S::Read<Vector> },
    /// `vfmadd231ps`: `acc = a·b + acc`.
    Fma231 { acc: S::Tie<Vector>, a: S::Read<Vector>, b: S::Read<Vector> },
    /// `vbroadcastss dst, [rip + at]`: one pool entry.
    LoadConst { dst: S::Write<Vector>, at: RipRel },
    /// `vpcmpeqd dst, dst, dst`: all-ones. Its reads are not operands: the
    /// result does not depend on them.
    Ones { dst: S::Write<Vector> },
    /// `vgatherdps dst, [base + index·4], mask`.
    Gather { dst: S::Early<Vector>, base: S::Read<Pointer>,
             index: S::Read<Vector>, mask: S::Tie<Vector> },
    /// `vcvttss2si r64, xmm`.
    Cvtt { dst: S::Write<Integer>, src: S::Read<Vector> },
    /// `imul r64, r64`. Destroys the flags, so it says so.
    Imul { dst: S::Tie<Integer>, src: S::Read<Integer>, flags: S::Write<Flags> },
    /// `lea p, [base + index·4]`.
    Lea4 { dst: S::Write<Pointer>, base: S::Read<Pointer>, index: S::Read<Integer> },
    /// `vmovmskps g, v`.
    MoveMask { dst: S::Write<Integer>, src: S::Read<Vector> },
    /// `test g32, g32`: ZF iff no lane is set.
    Test { flags: S::Write<Flags>, src: S::Read<Integer> },
    /// `cmp g8, imm8`: ZF iff the low byte is `imm` (`0xFF`: all eight lanes). The
    /// encoder picks the `cmp al` / REX forms.
    CmpByte { flags: S::Write<Flags>, src: S::Read<Integer>, imm: u8 },
    /// `jcc taken`; `next` is the block laid out next and encodes to nothing.
    Jcc { cond: Cond, flags: S::Read<Flags>, taken: S::Target, next: S::Target },
    Jmp { to: S::Target },
    Fallthrough { to: S::Target },
    /// `vmovups [rsp + disp32], v` / the reverse: the allocator's.
    StoreSlot { src: S::Read<Vector>, slot: S::Slot },
    LoadSlot { dst: S::Write<Vector>, slot: S::Slot },
    Copy { dst: S::Write<Vector>, src: S::Read<Vector> },
    /// `sub rsp, imm32`, then one probe per page (C6).
    Enter { size: S::FrameSize, flags: S::Write<Flags> },
    /// `add rsp, imm32; vzeroupper; ret`. The function's one block with no
    /// successors. The allocator asserts nothing is live there, which is what
    /// makes `vzeroupper`'s clobber of every vector register vacuous.
    Ret { size: S::FrameSize, flags: S::Write<Flags> },
    // …
}
```

Named fields stop selection or encoding from swapping `a` and `b` of a `vsubps` without anyone noticing. CLAUDE.md asks for a type exactly where a wrong value would be silently representable.

### 2.7 The allocator's view: one walk, and the operands it yields

```rust
/// One operand, as the allocator and the CFG read it. Produced by
/// `operands`, a fold over `IsaBackend::walk`. Computed once per instruction
/// into the allocator's operand table, never per query.
pub(in crate::emit) enum Operand {
    Reg { value: ValueName, access: Access },
    /// A branch target and the arguments it passes. A block's successors are
    /// the `Target` operands of its last instruction.
    Target(Target),
    /// A frame slot (only on instructions the allocator inserted).
    Frame(SlotName),
}

/// How an instruction touches a register operand. A `Tie` yields two
/// entries: `Read`, and a `Tied` write naming that read's position.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(in crate::emit) enum Access { Read, Write, Early, Tied { read: usize } }
```

**What is not an `Operand` variant, and why.** The ontology names five operand kinds. Two of them are not variants of this view:

- **Immediates** are typed fields that no phase before encoding reads.
- **Addresses** are composites. Their registers are listed here as `Reg` reads, and their data labels are the encoder's.

**Constraints, and where each one lives.** Each constraint lives where it cannot disagree with anything else:

- the class is the value's own (`ValueName::class`);
- tie and early are `Access` variants, so they cannot appear on a read;
- the ABI is `Function::entry`, so it cannot appear on any other operand.

A future requirement on an instruction operand will need a field type of its own: for example "the shift count is in `cl`", or `div`'s `rax`/`rdx` (ontology: Register). No such instruction exists in this tree, so none is built.

```rust
/// How an instruction is rebuilt at stage `T`, one operand at a time. Each
/// method is the only way to turn its kind of field into `T`'s. A `Bound`
/// instruction needs an `In`/`Out`/`InOut` for every register field, and
/// only `regalloc` can make one. So `walk` can neither misreport an access
/// nor skip a register.
pub(in crate::emit) trait Rebind<T: Stage> {
    fn read<C: Class>(&mut self, v: Value<C>) -> T::Read<C>;
    fn write<C: Class>(&mut self, d: &Def<C>) -> T::Write<C>;
    fn early<C: Class>(&mut self, d: &Early<C>) -> T::Early<C>;
    fn tie<C: Class>(&mut self, t: &Tie<C>) -> T::Tie<C>;
    fn slot(&mut self, s: SlotName) -> T::Slot;
    fn target(&mut self, t: &Target) -> T::Target;
    fn frame_size(&mut self) -> T::FrameSize;
}

/// Every operand of `inst`, in walk order: `walk` at a stage whose fields are
/// `()`, recording as it goes. The operand list and the binding are one
/// traversal, so they cannot disagree.
pub(in crate::emit) fn operands<B: IsaBackend>(inst: &B::Inst<Selected>) -> Vec<Operand>;
```

### 2.8 Labels and the assembler (`emit/asm.rs`)

This module imports nothing from the crate.

```rust
/// The name of an address. Minted with the thing at that address: a block, a
/// data section, a pool entry. `Copy`, 64-bit, no public constructor.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::emit) struct Label(u64);

/// A program's label namespace: the only mint. Owned by whoever is building
/// the program (the `Builder`, then the `AsmProgram`), so minting needs
/// `&mut` to it.
#[derive(Default)] // `Labels::default()` is the empty namespace
pub(in crate::emit) struct Labels { next: u64 }
impl Labels {
    pub(in crate::emit) fn mint(&mut self) -> Label;
}

pub(in crate::emit) enum Item<I> { Bind(Label), Inst(I), Align(u64), Bytes(Vec<u8>) }

/// One kernel's assembly program: a text section and a data section.
pub(in crate::emit) struct AsmProgram<I> {
    pub text: Vec<Item<I>>,
    pub data: Vec<Item<I>>,
    pub labels: Labels,
}

/// How a label field is filled once the label's address is known: write the
/// field at `at` (its own offset, not its instruction's) so that it reaches
/// `target`. The encoder that wrote the field supplies it. This is today's
/// `LabelRef::patch` (`emit/mod.rs:310`); A8 moved the field's offset out of
/// the patch and into `Encoding::field`.
pub(in crate::emit) type Patch = fn(code: &mut [u8], at: usize, target: usize);

/// What an encoder writes into: one instruction's bytes and its label fields.
pub(in crate::emit) struct Encoding<'a> { /* the assembler's */ }
impl Encoding<'_> {
    pub(in crate::emit) fn bytes(&mut self, bytes: &[u8]);
    /// A field starting `at` bytes into this instruction that names `label`.
    pub(in crate::emit) fn field(&mut self, at: usize, label: Label, patch: Patch);
    /// A zero-width field at the end of this instruction: `label` must be
    /// bound exactly there. A fall-through; checked, not assumed.
    pub(in crate::emit) fn falls_through(&mut self, label: Label);
}

pub(in crate::emit) struct Assembled { pub code: Vec<u8>, addresses: Vec<usize> }
impl Assembled { pub(in crate::emit) fn address(&self, label: Label) -> usize; }
// A8 lands `assemble` returning the `Vec<u8>` alone. `Assembled` and
// `falls_through` arrive with their first reader (B-series).

/// Lay out `text` then `data`, encode each instruction with `encode`, and
/// patch every field.
///
/// # Panics
/// - a label bound twice, or named but never bound;
/// - a fall-through whose label is not bound where the instruction ends.
pub(in crate::emit) fn assemble<I>(
    program: &AsmProgram<I>,
    encode: impl Fn(&I, &mut Encoding<'_>),
) -> Assembled;
```

**The encoder is passed in, not a trait the assembler declares.** So the assembler names no instruction type. `Assembly` is not generic over the backend, and no `LabelField` associated type exists (closure 10).

### 2.9 Functions, blocks, the builder

```rust
pub(in crate::emit) struct Block<I> {
    pub label: Label,
    /// Values defined on entry: what a phi is.
    pub params: Vec<ValueName>,
    pub insts: Vec<I>,
    /// The nest scope this block's code belongs to, for traffic and trip
    /// weighting.
    pub scope: Scope,
}

/// A label operand before allocation: where to go, and the values for the
/// target's parameters, in order.
pub(in crate::emit) struct Target { pub label: Label, pub args: Vec<ValueName> }

/// The ABI's three arguments: the entry block's parameters, with their classes.
pub(in crate::emit) struct Entry { pub ctx: Value<Pointer>, pub out: Value<Pointer>, pub pitch: Value<Integer> }

/// One surviving fold's loop.
pub(in crate::emit) struct Loop { pub head: Label, pub parent: Option<usize>, pub trips: u64 }

/// The pool section: its label, then each entry with its own label, in order.
pub(in crate::emit) struct Constants<K> { pub label: Label, pub entries: Vec<(Label, K)> }
/// One pool entry: its label (x86 reads it RIP-relative) and its byte offset
/// in the section (aarch64 reads `[pool base, #offset]`).
#[derive(Copy, Clone, Debug)]
pub(in crate::emit) struct Constant { pub label: Label, pub offset: u64 }

/// A whole kernel, in layout order.
///
/// Invariants, established by selection and asserted by `Builder::finish`:
/// - `blocks[0]` is the entry. Its parameters are `entry`'s three.
/// - Every minted label is bound by exactly one block or data item.
/// - Only a block's last instruction has `Target` operands.
/// - Exactly one block has none: the last.
/// - Every `Target`'s arguments match its block's parameters in arity and
///   class.
/// - The `next` target of a conditional branch is the following block and
///   has no parameters.
/// - Every backward branch comes from the last block of a loop in `loops`
///   and targets that loop's head.
/// - Every forward branch skips a contiguous run of blocks within one scope.
/// - A conditional branch passes arguments on at most one target.
///
/// **Dominance is relaxed in one way, and only one.**
/// - A value an `If` arm defines is read by that `If`'s blend.
/// - On the path where a uniform mask skipped the arm, the wrapper goes to
///   that `If`'s single-arm block, and the blend does not run.
/// - So a definition dominates every read that *executes*, not every
///   syntactic read.
/// - Interval liveness (§1.3) never carries the value above its definition.
///   Join invariant 2 takes its location from the path that defines it.
/// - No phase inserts a phi for it.
pub(in crate::emit) struct Function<B: IsaBackend> {
    pub blocks: Vec<Block<B::Inst<Selected>>>,
    pub entry: Entry,
    pub loops: Vec<Loop>,
    pub constants: Constants<B::Constant>,
    pub labels: Labels,
}
```

The builder lives in `build.rs`, a leaf module.

```rust
/// How selection writes a function, one block at a time.
pub(in crate::emit) struct Builder<B: IsaBackend> { /* … */ }
impl<B: IsaBackend> Builder<B> {
    /// A function whose entry block is open, and its three parameters.
    pub(in crate::emit) fn new() -> (Self, Entry);
    pub(in crate::emit) fn def<C: Class>(&mut self) -> Def<C>;
    pub(in crate::emit) fn early<C: Class>(&mut self) -> Early<C>;
    pub(in crate::emit) fn tie<C: Class>(&mut self, read: Value<C>) -> Tie<C>;
    /// Append `inst` to the open block.
    ///
    /// # Panics
    /// - a write not minted by this builder, or already defined;
    /// - a read of a value not yet defined;
    /// - a read of a value this same instruction defines.
    pub(in crate::emit) fn push(&mut self, inst: B::Inst<Selected>);
    /// A block to be entered later. Its label is minted now, with it.
    pub(in crate::emit) fn block(&mut self, params: &[ClassId], scope: Scope) -> Pending;
    /// Open `block`. It is consumed, so it cannot be entered twice.
    pub(in crate::emit) fn enter(&mut self, block: Pending);
    /// The pool entry for `k`, deduplicated.
    ///
    /// # Errors
    /// `BudgetExceeded` past the backend's reach (`IsaBackend::POOL_REACH`).
    pub(in crate::emit) fn constant(&mut self, k: B::Constant) -> Result<Constant, CompileError>;
    pub(in crate::emit) fn open_loop(&mut self, head: Label, parent: Option<usize>, trips: u64) -> usize;
    /// Per-function selection state the backend keeps (aarch64: the pool base value).
    pub(in crate::emit) fn anchors(&mut self) -> &mut B::Anchors;
    pub(in crate::emit) fn finish(self) -> Function<B>;
}
/// A block minted but not yet entered. Affine; `finish` panics on one never
/// entered.
#[must_use] pub(in crate::emit) struct Pending { label: Label, params: Vec<ValueName>, scope: Scope }
impl Pending {
    pub(in crate::emit) fn label(&self) -> Label;
    /// Parameter `i` as a `Value<C>`. Panics on the wrong class.
    pub(in crate::emit) fn param<C: Class>(&self, i: usize) -> Value<C>;
}

/// The rights the allocator's own verbs get. `def` mints only `Spill`
/// classes, so a copy, spill or reload cannot write the flags.
pub(in crate::emit) struct Spiller<'a, B: IsaBackend> { /* … */ }
impl<B: IsaBackend> Spiller<'_, B> {
    pub(in crate::emit) fn def<C: Spill>(&mut self) -> Def<C>;
    pub(in crate::emit) fn push(&mut self, inst: B::Inst<Selected>);
}
```

**Bindings are private to `select.rs`, and scoped.**

- They form a stack of frames, one per open scope, and a lookup searches the innermost frame first. Each sibling fold pushes and pops its own frame.
- So the `ValueId`s that carved siblings share (`regalloc.rs:1045-1053`; `sibling_column_folds_share_a_reduce_and_its_slots`, `mod.rs:6984`) never collide.
- A binder resolves to the innermost enclosing fold that binds it, as `binder_at` does today (`mod.rs:1375`).
- The backend never sees a `ValueId`: the driver resolves every operand to a `Value<C>` before calling the backend. The class check at that boundary is the driver's, once.

### 2.10 The machine: `IsaBackend`

```rust
/// Everything about a target. The driver, the allocator and the assembler
/// are generic over it, and none of them names a register, an opcode or an
/// encoding. `compile_native` monomorphizes it once per tier: one `match`
/// per kernel.
pub(in crate::emit) trait IsaBackend: Sized + 'static {
    type Inst<S: Stage>;
    /// One pool entry: x86 `u32` (each load broadcasts a scalar), aarch64
    /// `[u32; 4]`.
    type Constant: Copy + Ord;
    /// Per-function selection state: aarch64's pool-base value; `()` on x86.
    type Anchors: Default;
    /// One IR lane value as this machine holds it (§0.1): `Value<Vector>` on
    /// AVX2 and NEON; a `Vector` or an `Opmask` value on AVX-512.
    type Lane: Copy;
    const FILE: RegisterFile;
    /// How many pool entries an instruction can reach. aarch64:
    /// `ldr q, [p, #imm12·16]`, so 4096. x86: `i32::MAX / 4`.
    const POOL_REACH: u64;

    // Selection. Operands are values; the driver resolved them.
    fn lane(b: &mut Builder<Self>, op: LaneOp<Self>) -> Result<Self::Lane, CompileError>;
    /// The class of a block parameter that carries `lane`.
    fn lane_class(lane: Self::Lane) -> ClassId;
    /// A block parameter, read back as a lane.
    fn param_lane(param: ValueName) -> Self::Lane;
    fn context(b: &mut Builder<Self>, ctx: Value<Pointer>, slot: u64) -> Result<Value<Pointer>, CompileError>;
    fn store(b: &mut Builder<Self>, store: Store<Self>);
    /// End the block: go to `edges.taken` when no lane of `test.cond` selects
    /// `test.dead`'s arm; otherwise fall through to `edges.next`.
    fn branch(b: &mut Builder<Self>, test: Test<Self>, edges: Edges);
    /// End the block: go to `to`. A `Fallthrough` when `to` is `next`.
    fn jump(b: &mut Builder<Self>, to: Target, next: Label);
    /// The entry block's `Enter`, and the backend's per-function values
    /// (aarch64's pool base).
    fn enter(b: &mut Builder<Self>);
    fn ret(b: &mut Builder<Self>);

    // The allocator's own instructions.
    fn copy<C: Spill>(b: &mut Spiller<'_, Self>, src: Value<C>) -> Value<C>;
    /// Store `src` to `slot`. When `slot.offset()` does not encode: a
    /// `SlotAddr` into a fresh `Pointer`, then the store through it.
    fn spill<C: Spill>(b: &mut Spiller<'_, Self>, src: Value<C>, slot: &FrameSlot);
    fn reload<C: Spill>(b: &mut Spiller<'_, Self>, slot: &FrameSlot) -> Value<C>;
    /// Whether the allocator may recompute this instruction's definition
    /// instead of storing it. It must read no value that is not itself
    /// rematerializable, have no effect, and write no `Flags`:
    /// - pool loads;
    /// - `movi`/`fmov` immediates;
    /// - `adrp+add` of a label;
    /// - the zero and all-ones idioms.
    fn rematerializable(inst: &Self::Inst<Selected>) -> bool;

    /// Rebuild `inst` at stage `T`, visiting each operand once, in field
    /// order. The only per-instruction traversal.
    fn walk<T: Stage>(inst: &Self::Inst<Selected>, f: &mut impl Rebind<T>) -> Self::Inst<T>;

    /// Encode one allocated instruction.
    fn encode(inst: &Self::Inst<Bound<'_, Self>>, out: &mut Encoding<'_>);
}

/// An operation producing one lane value, with its operands already lanes.
/// Comparisons and `BitAnd`/`BitOr` are `Binary`: which file their result
/// lives in is the backend's choice (§0.1).
pub(in crate::emit) enum LaneOp<B: IsaBackend> {
    Const(f32),
    /// `[0, 1, …, L−1]`.
    Lanes,
    Unary(OpKind, B::Lane),
    Binary(OpKind, B::Lane, B::Lane),
    /// `a·b + c`: one rounding, every target.
    MulAdd(B::Lane, B::Lane, B::Lane),
    /// The `If`'s lane-varying path.
    Blend { cond: B::Lane, if_true: B::Lane, if_false: B::Lane },
    Shift(OpKind, B::Lane, u8),
    Gather { base: Value<Pointer>, index: B::Lane },
    Broadcast { base: Value<Pointer>, index: B::Lane },
    Uniform { base: Value<Pointer>, element: u64 },
}
/// The lattice's effect: `value`'s first `lanes` lanes at
/// `out + 4·(row·pitch + col)`.
pub(in crate::emit) struct Store<B: IsaBackend> {
    pub out: Value<Pointer>,
    pub pitch: Value<Integer>,
    pub row: B::Lane,
    pub col: B::Lane,
    pub value: B::Lane,
    pub lanes: u32,
}
pub(in crate::emit) struct Test<B: IsaBackend> { pub cond: B::Lane, pub dead: IfArm }
pub(in crate::emit) struct Edges { pub taken: Target, pub next: Label }
```

**How the driver uses these** (in `select.rs`).

The driver binds every IR value to a `B::Lane` and never inspects one (§0.1). The arithmetic ops, the comparisons, `Context`, `Uniform` and `Gather` call `lane` or `context`.

1. `enter`.
2. The body's defs, in schedule order:
   - `Outer` placeholders (A5) and non-opening `Reduce` placeholders are reads resolved through `Bindings`, not definitions.
   - A `Var(n)` resolves to a binder.
   - `Seq` emits nothing.
   - `Write` calls `store`.
   - `Context`, `Uniform`, `Gather` and the arithmetic ops call `context` or `lane`.
3. A `Reduce` that opens fold `F` becomes:
   1. The preheader selects the seeds `lo` and the identity as constants, then `jump(Head(F) [lo, identity])`.
   2. Block `Head(F)` has parameters `(binder, acc)`, or just `(binder)` for a `SEQ` fold, which has no accumulator.
   3. Then `F`'s schedule.
   4. Then the latch, as ordinary `lane` calls:
      - `acc' = lane(Binary(fold.combine_op(), acc, body))`;
      - `step = lane(Binary(Add, binder, stride))`;
      - `done = lane(Binary(Ge, step, hi))`, which AVX-512 holds in `k`.
   5. Then `branch(Test { cond: done, dead: True }, Edges { taken: Head(F) [step, acc'], next: exit })`. A broadcast `done` is uniform, so "no lane is true" is exactly "not done".
4. The `Reduce`'s value is `acc'`. It dominates the exit, which has one predecessor, so no exit parameter is needed (closure 14).
5. An arm guarded by `IfGuard` is `branch(Test { cond, dead: arm }, Edges { taken: past, next: arm_block })`. Then the arm's defs, then `jump(past, past)`.
6. A guarded `If`'s uniform wrapper:
   - `branch(dead: True → only_false)`;
   - `branch(dead: False → only_true)`;
   - the blend block, ending `jump(join [blend])`;
   - `only_false`, ending `jump(join [f])`;
   - `only_true`, ending `jump(join [t], next = join)`;
   - `join`, with one parameter.
7. `ret`.

### 2.11 Allocation

```rust
/// Where every value lives at every point. One implementation, `LinearScan`,
/// retargeted:
/// - its policies move to `regalloc/policy.rs` and are kept (Belady eviction
///   ranked store-then-distance, rematerialization, fold-aware carry
///   pricing);
/// - its mechanics, which allocate scheduled ops plus reservations, are
///   replaced by a scan over instructions.
pub(in crate::emit) trait RegisterAllocator {
    /// Bind `function`'s values to leases of `pool` and to slots of `frame`,
    /// inserting the spill, reload, remat and copy instructions that make the
    /// binding hold, and binding `Enter`/`Ret`'s frame size.
    ///
    /// # Errors
    /// `BudgetExceeded` when the narrow region passes 4,095 slots or the frame
    /// passes `MAX_FRAME`.
    ///
    /// # Panics
    /// A selection bug, never a fact about a kernel:
    /// - an instruction holding more registers of a file than the file has;
    /// - a `Flags` value live at a label or across another flags write;
    /// - a broken join invariant (§1.3);
    /// - a block-argument target parameter live into the other successor.
    fn allocate<'m, B: IsaBackend>(
        &self,
        function: Function<B>,
        pool: &'m Pool<B>,
        frame: &'m mut Frame,
    ) -> Result<Allocated<'m, B>, CompileError>;
}

/// An allocated kernel: every register a borrowed token, every slot a
/// borrowed slot, every block argument a placed move.
pub(in crate::emit) struct Allocated<'m, B: IsaBackend> {
    pub blocks: Vec<Block<Placed<'m, B>>>,
    pub loops: Vec<Loop>,
    pub constants: Constants<B::Constant>,
    pub labels: Labels,
    pub frame_bytes: u64,
}
pub(in crate::emit) struct Placed<'m, B: IsaBackend> { pub inst: B::Inst<Bound<'m, B>>, pub origin: Origin }
/// Why an instruction is there. `EmitTraffic` counts these per scope and
/// weights them by trips.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(in crate::emit) enum Origin { Selected, Spill, Reload, Remat, Copy }
```

**The scan.** It is one pass in layout order, so a fold is scanned inside its parent (closure 17).

1. **Operand table and liveness.** `operands()` once per instruction. Intervals with loop extension (§1.3). Next-read positions for Belady.
2. **Frame.** The peak live count of the narrow classes goes to `Frame::reserve_narrow`.
3. **Carry plan.** Per loop, the candidates are every value live into `Head(F)` plus `Head(F)`'s parameters (closure 15). They are priced by today's carry pricing: reads inside the loop × trips, plus one latch copy per trip for a parameter.
   - Budget per file: `members − reserve`.
   - Vector: `CARRY_RESERVE = 7`. This is the fitted constant from escape-hatches 2026-09-04/05, which three principled replacements lost to.
   - General: `GENERAL_CARRY_RESERVE = 5`. That gives x86 4 carries: today's 2 pointer carries plus `out` and `pitch`, the pool base being gone. It gives aarch64 13: today's 9 plus `out`, `pitch`, the pool base, and one more.
   - Opmask: 0. An AVX-512 predicate live into a loop head is spilled (`kmovw`). On AVX2 and NEON, a predicate is a `Vector` and is priced like one.
   - Flags: never carried.
   - A carried value takes a lease at the preheader and holds it to the latch.
4. **Per instruction `i`:**
   1. Leases of values with no read at or after `i` return to the free lists.
   2. Each read not resident at `i` gets a reload (or a remat) inserted before `i`, into a fresh split value. It takes a free lease, or evicts by `EvictionRank`. Eviction never touches a value read at `i` or a carried value inside its loop. The first eviction of a value inserts its store right after its definition.
   3. A `Tie` takes its read's lease if the read dies at `i`. Otherwise a copy is inserted first, and the copy is tied.
   4. An `Early` write takes a lease that no read of `i` holds.
   5. A plain write may take the lease of a read that dies at `i`. It prefers the register of the block parameter it flows to (a hint), so a latch value usually needs no back-edge copy.
   6. A write nobody reads releases its lease after `i`.
5. **Block arguments** become a parallel move into the parameters' locations, placed before the terminator. The moves are flags-safe by type.
   - They are sequentialized.
   - A register cycle is broken through a fresh value, allocated like any other (closure 21, landing B6).
   - They are sound because no target parameter is live into the other successor (asserted).
6. **Joins** are checked against §1.3's three invariants. Forward joins take the intersection of their predecessors' states, taken from snapshots recorded at each forward branch.
7. **Bind.** The scan records, per operand, the `In`/`Out`/`InOut` made from the lease it held. After the scan, `&'m mut Frame` becomes `&'m Frame`, and one `walk` per instruction builds `Inst<Bound>`, with `frame_size()` set to `frame.bytes()`.

### 2.12 The driver

```rust
/// The selection pipeline: select, allocate, assemble.
fn compile_on<B: IsaBackend>(scoped: &ScopedSchedule) -> Result<CompileResult, CompileError> {
    let function = select::<B>(scoped)?;
    let pool = Pool::<B>::mint();
    let mut frame = Frame::empty(B::FILE.vector_bytes);
    let allocated = LinearScan.allocate(function, &pool, &mut frame)?;
    let program = allocated.to_asm();          // text: blocks; data: the pool
    let assembled = asm::assemble(&program, |i, out| B::encode(&i.inst, out));
    let traffic = EmitTraffic::of(&allocated, &assembled);
    CompileResult::new(&assembled.code, traffic, allocated.frame_bytes)
}
```

**`CompileResult` changes:**

- `spill_bytes: u32` becomes `frame_bytes: u64`.
- `spill_count` is redefined as frame slots minted.
- `hoisted_values` is redefined as values read inside a loop nested in the scope that defines them.

**`EmitTraffic`'s scope counts** are counted from `Origin` per `Block::scope` and weighted by `Loop::trips`:

- `instructions` still means the scheduled ops a scope selected. The driver tallies them per block.
- `bytes` comes from `Assembled::address`.

### 2.13 What ownership cannot say, and the check that stands in

| Property | Why a type cannot carry it | What checks it |
|---|---|---|
| A dropped `Def` leaves a name with no definition | Rust is affine, not linear | `#[must_use]`; `Builder::finish` panics naming it |
| A read before its definition; an instruction reading its own result | `Def::value` must be readable for selection to return it (closure 21, landing B4) | `Builder::push` panics |
| A `Target`'s arguments match its block's parameters | A `Label` is a name, not a typed handle | `Builder::finish` |
| A `Pending` parameter's class | Parameter lists are data | `Pending::param::<C>` panics |
| Exclusivity inside the allocated program: an `Early`'s register is none of its instruction's reads | The allocated program holds shared borrows, by design (§2.2) | The scan's lease state, plus an assertion at binding |
| No `Flags` live across a flags write or at a label | Liveness is a function property | The allocator panics. Spilling flags is a type error (`Spill`) |
| A rematerialized instruction writes no `Flags` | A cloning walk must mint any class | The allocator asserts it at remat |
| A `program::ValueId`'s class at the boundary | `ValueId` is untyped by design (`scripts/check-emit-boundary.sh`) | One check, in `select.rs`'s `Bindings` |
| Tokens from two pools of one backend mixed | A generative brand costs more than it buys: both pools encode the same numbers, and `allocate` takes one | — |
| The relaxed dominance across a uniform guard | A correlated branch is not in the CFG | Interval liveness and join invariant 2; documented on `Function` |
| Structured control flow | A CFG type would not stop a stray back edge | `Builder::finish` |

---

## 3. How each escape hatch and each prediction dissolves

In the new pipeline, none of these is ever written. The legacy code is deleted in D1 unless a row says otherwise.

| # | Today (`bdee3900`) | What it is | Dissolves into | Commit |
|---|---|---|---|---|
| P1 | `RegisterFile::temps_for` (`regalloc.rs:310`); tables `avx2.rs:478`, `avx512.rs:281`, `aarch64.rs:782` | A prediction of the vector registers an encoding destroys | Intermediates are values the selected instructions define | B3–B9 / D1 |
| P2 | `gpr_temps_for` (`regalloc.rs:357`; `avx2.rs:509`, `avx512.rs:306`, `aarch64.rs:812`); `gpr_scratch` (`regalloc.rs:352`) | A prediction, plus a second pool that holds nothing across instructions | `Integer`/`Pointer` values: a store's row and column, a broadcast's index, the iota | B3, B8 / D1 |
| P3 | `mask_temps_for` (`regalloc.rs:379`; `avx512.rs:321`); `mask_scratch = {k1}` (`avx512.rs:1267`) | A prediction over a one-register pool | `Opmask` values over `k1`–`k7` | C2 / D1 |
| P4 | `guard_temps` (`regalloc.rs:285`); `Scratch::guard_mask`/`guard_temp` (`regalloc.rs:856-861`); `guard_scratch` (`aarch64.rs:1947`) | A prediction for code emitted between instructions | `branch` selects ordinary instructions. NEON's `UMAXV` result is a `Vector` value | B9, C4 / D1 |
| P5 | `mask_guard_temps` (`regalloc.rs:386`, used at `3552`) | The same, for `vptestmd` | A guard on an `Opmask` lane is `KorTest(k)`, defining `Flags`; on a `Vector` lane, `Ptestm` first (§0.1) | C2 / D1 |
| P6 | `Scratch::REDUCE_TEMPS` (`regalloc.rs:916`); `Reduce` arms `avx2.rs:490`, `avx512.rs:291`, `aarch64.rs:792` | A prediction for the loop driver's `t0`/`t1` | The latch is ordinary `lane` calls | B3 / D1 |
| P7 | `MIN_SCRATCH` (`regalloc.rs:517`), `MIN_POINTERS` (`524`) | A floor computed from predictions | No reservations exist. The per-instruction peak is asserted. Carry budgets are `members − reserve` (§2.11) | B4–B6 / D1 |
| P8 | `operand_sources` (`mod.rs:833`), `reloads_wanted` (`884`) | A prediction of which operands the emitter reloads | A reload is an inserted instruction defining a fresh value | B4 / D1 |
| H1 | `POOL_BASE = PtrReg(8)` (`x86_64.rs:33`); `anchor` (`x86_64.rs:136`) | `r8` held for the whole kernel | `vbroadcastss y, [rip + entry]`; no base value. `r8` joins `General` | B3 |
| H2 | aarch64 `X17` anchor (`aarch64.rs:2140-2145`); pool loads from `X17` (`1851`, `2330`); `X16`/`X17` (`2494-2496`) | IP1 held for the whole kernel | `p = AdrpAdd(pool)` in the entry block: a rematerializable `Pointer`. `x17` joins `General` | C4 |
| H3 | AVX2 guard `avx2.rs`'s `branch_if_arm_is_dead`: `Inst::MoveMask`, `Gp::CmpByte` or `Gp::Test`, `Gp::Jcc` | `rax` clobbered between instructions | `g = MoveMask(m)`, then `f = Test(g)` (a dead `True` arm) or `f = CmpByte(g, 0xFF)` (a dead `False` arm), then `Jcc(f)`. The encoder picks the `cmp al` form | B9 |
| H4 | `FmovToGp` with `Rd = 16` (`table.rs:425-439`); `mvn_w(X16, X16)` (`aarch64.rs:2097`); `CBNZ_W16_OVER_B` (`2745`), `BranchIfW16Zero` (`2770`) | IP0 written and tested by convention | `FmovToGp { dst: Write<Integer> }`, `MvnW { dst: Tie<Integer> }`, `CbzFar { test: Read<Integer>, taken, next }` (`cbnz w, .+8; b taken` with an `imm26` field) | A12b fields; C4 values |
| H5 | `emit_fmov_imm`'s `w16` path (`aarch64.rs:466-477`) | `movz/movk w16; dup` | Every constant is selected: `movi`, `fmov`, or `ldr q, [p, #off]` | C4 |
| H6 | `address_in_ip0` (`table.rs:530`), called by `StrQ` (`562`), `LdrQ` (`617`) and `LdrS` (`672`). `LdrS` also serves `emit_uniform_load` (`aarch64.rs:410`) and `index_into`'s slot read (`2269`). `StrX`/`LdrX` panic past `imm12` (`table.rs:589-594`, `644-648`), and `LdrX` serves `Context` (`aarch64.rs:2390-2401`), so a context slot of 4096 or more panics | IP0 as an address, and two panics | Spills: `spill`/`reload` emit `SlotAddr { dst: Write<Pointer>, slot }` (`add t, sp, #hi, lsl 12; add t, t, #lo`); narrow slots always encode. Non-frame loads (`Uniform`, `Context`): selection emits `t = AddImm(base, #hi, lsl 12)`, `ldr [t, #lo]`, and `movz/movk` plus a register `add` past 16 MiB. The binder read is an ordinary reload. One instruction replaces today's chain of `add #4080` | C4 |
| H7 | `kortestw` bytes hardwiring `k1,k1` (`avx512.rs:647-653`) | An encoder that ignores its register | `KorTest { flags: Write<Flags>, k: Read<Opmask> }` encodes `k.number()` | A11b field; C2 value |
| H8 | Gather mask `mov eax, 0xFFFF; kmovw k1, eax` (`avx512.rs:749`); `aaa = 001` (`789`); `gpr_temps_for(Gather) = 0` | `rax` and `k1` clobbered, undeclared | `k = KOnes` (`kxnorw k, k, k`: no general register), then `Gather { dst: Early<Vector>, mask: Tie<Opmask> }` | C2 |
| H9 | Gather into `temp(1)`, then `vmovaps` (`avx512.rs:1368-1381`) | A temp plus a move standing in for an early def | `dst: S::Early<Vector>`; the move is gone | C2 |
| H10 | Fold `t0`/`t1` (`mod.rs:1798`); seed through `t0` (`1825-1826`); `acc = if body_result == t0 {…}` (`1881`) | The driver aliasing registers by hand | Seeds are preheader `Target` arguments; the accumulate is `Binary(combine, acc, body)` | B3 |
| H11 | `test_ge` borrowing the guard's `k` (`mod.rs:1855`; `avx512.rs:1604-1613`) | One mask register serving two roles | `done` is an ordinary compare; its `Opmask` value is its own | B3, C2 |
| H12 | Loop verbs `slot_store`…`test_ge` (`mod.rs:1078-1140`) | A second instruction set beside the schedule | The latch is selected (§2.10) | B3 / D1 |
| H13 | The `is_reduce` disjunct (`regalloc.rs:3533-3534`) | The allocator reserving for a test it cannot see | The test is in the function | D1 |
| H14 | `Reg(u8::MAX)` (`mod.rs:1987`) | A register-shaped stand-in for "lives nowhere" | A rematerializable definition is placed only where it is read | B7 |
| H15 | `OperandSource::Destination` (`mod.rs:812`, `2145`) | Reload into the destination, sound only by a convention | A reload defines a fresh value. Plain writes may take a dying read's lease | B4 |
| H16 | `setup_mov` (`mod.rs:780`) | A copy chosen while resolving operands | The allocator copies a `Tie`'s live read | B4 |
| H17 | `DecomposedMulAdd`, `c_deferred` (`mod.rs:700-705`) | An instruction choice made by residency | `MulAdd` always fuses | A4 |
| H18 | `gpr_ctx`/`gpr_out`/`gpr_pitch` (`regalloc.rs:331-341`), read by name in `write_address` (`x86_64.rs:862-878`) and `Context` | ABI registers read by name | `Function::entry`, initial ownership via `EntryLeases` | B3 |
| H19 | `write_address`'s positional `gpr_temp(0/1)` (`x86_64.rs:868-869`); AVX-512's remainder mask reusing `gpr_temp(1)` (`avx512.rs:1619`) | Role conventions | Named values: `row`, `col`, `addr`, `m` | B3, C2 |
| H20 | Emitter reconciliation: `store_after_def` (`mod.rs:1477`), `ptr_into` (`1512`), `hand_off` (`1596`), `binder_placeholders` (`1346`), the head reconciliation, the scope-result reload | Allocator decisions executed by the emitter | Inserted instructions (`Origin`), block parameters, and the join invariants | B4–B6 / D1 |
| H21 | `Label` as a 31-byte string (`mod.rs:241`, `CAPACITY` `250`); `format!` (`1439`, `1836-1837`, `1937`); `CONST_POOL: &str` (`432`) | A key flattened to text | `asm::Label(u64)`, minted (ontology: Label) | A2 |
| H22 | About 227 signatures taking `code: &mut Vec<u8>` (`grep -c`, `bdee3900`) | Choice and encoding fused | Per-ISA `Inst<S>`; `encode(&Inst<Bound>)` | A9–A12 |
| H23 | `Where::Ptr` (`regalloc.rs:647`); `pointers` (`370`) beside `gpr_scratch` (`352`) | One physical file split by role | One `GeneralFile`. `Pointer` and `Integer` are *types* of values in it | B3 / D1 |
| H24 | `accumulator_slots`/`binder_slots` keyed by the `Reduce`'s `ValueId`, "the later fold wins" (`regalloc.rs:1050-1054`, pins `1216-1287`) | Two carved siblings sharing slots by accident | Each fold's parameters belong to its own `Head`, and slots are leases | B6 / D1 |
| H25 | The stack pointer: written by `frame_alloc` and `emit_ret`, which frees (`avx2.rs:1465`, `avx512.rs:1552`, `aarch64.rs:2110-2136`), read by every slot (`x86_64.rs:796`, `aarch64.rs:1864`) | A register in no class, used by convention | Owned by `Frame`. Read only through slot operands and `SlotAddr`; written only by `Enter`/`Ret` | B3, C4 |
| H26 | Placeholder defs: `place_roots` rewrites a hoisted vector def to `Const(0.0)` (`program/scopes.rs:277`); a pointer one stays `Context`; `stays_put` (`scopes.rs:205`) | A def whose op lies about what it is. A selected placeholder is a silent 0.0 (the escape-hatches 2026-09-04 miscompile) | `ScheduledOp::Outer(Class)`, a typed read of an enclosing value (A5). D2 deletes the placeholders | A5, D2 |
| H27 | `vzeroupper` and `ret` (`x86_64.rs:159`) | Implicit clobber and read | `Ret` is the exit terminator; nothing is live there (asserted) | B3 |
| H28 | `Write`'s binder read chosen by residency: `vcvttss2si r64, m32` from a slot (`x86_64.rs:846-857`), NEON `ldr s via` (`aarch64.rs:2269-2278`) | A selection made by residency | The binder is a value. A slot-held binder gets a reload, then the register form | B3 (x86), C4 (NEON) |
| H29 | Pool order fixed by emission order (`x86_64.rs:71-80`) and per-scope seeding (`aarch64.rs:1988-2016`, `BUILTIN_HEADROOM` `2010`) | Byte layout depending on emitter order | Entries are deduplicated in selection order. The pool is a data section. The reach check is `POOL_REACH` | B3, C4 |

---

## 4. What is deleted, what is added, and why nothing smaller works

### 4.1 Deleted, by name

The narrowing already removes these, so they are not listed again here: `Item`, `Storage`, `StoreTarget`, `SourceOperand`, `BranchTraffic`, `EmitTraffic::trailing`, `IfGuard::total_guarded_entries`, `EmitCtx`, `capped`, `flat_nest`, `allocate_flat`, `CompileError::{UnboundLabel, DuplicateLabel}`, and the x86 `Inst::{Encoded, Jmp, Jcc, MovLoadPtr}`.

**A2.**
- The string `Label`, `Label::CAPACITY`, `Label::new`, `as_str`, `From<&str>`, `Display`, and the hand-written `Debug`.
- `CONST_POOL`.
- The three `format!` sites.

**A4.**
- `ResolvedOp::DecomposedMulAdd`, `DeferredReload`, and `operand_sources`' both-spilled `MulAdd` arm.
- Each backend's decomposed arm.
- The tests `decomposed_encodes_to_a_multiply_and_an_add` and `resolve_muladd_decomposed_*`, and `a_deferred_c_is_reloaded_between_the_multiply_and_the_add`. `fused_encodes_to_the_targets_fma` is rewritten against the one fused form.

**A6.** `ScopeTraffic::{loads_transient, loads_kept}`, replaced by `loads`.

**A9–A12.** Every free byte-writer in `x86_64.rs`, `avx2.rs`, `avx512.rs`, `aarch64.rs` and `table.rs` becomes an `Inst<S>` arm plus its `encode` arm. Deleted along the way:
- `emit_test_eax`, `emit_movmskps_eax`, `emit_cmp_al_imm8` (gone at A9 and A10b: `Gp::Test`, `Inst::MoveMask`, `Gp::CmpByte`);
- `emit_set_gather_mask`, `emit_mask_flags`, `gather(…, base_gpr: u8, …)`;
- aarch64's positional `Inst` (`aarch64.rs:41`) and `BranchIfW16Zero`'s fixed register.

**D1 — the legacy pipeline.**

`emit/mod.rs`:
- the register newtypes `Reg`, `Gpr`, `PtrReg`, `KReg` (they were `pub u8`; the narrowing made the field private);
- `AsmInsn` (every backend's `Inst<Physical>` implements it) and `Class::Physical`, `Assembly::push` (its one caller), and the legacy `emit::AsmProgram<S>`, which leaves `asm::AsmProgram<I>` the only one;
- `Loc`, `Binding`, `Unary`, `ResolvedOp`, `Reload`, `InstructionPlan`, `OperandSource`;
- `operand_sources`, `reloads_wanted`, `declared_temp`, `declared_gpr_temp`, `declared_mask_temp`;
- `emit_scope`, `resolve_operands`, `location_of`, `binding`;
- `WritePlan`, `MaskTest`;
- the legacy trait (renamed `LegacyBackend` in B1) with every verb: `jump`, `register_file`, `begin`, `emit_plan`, `emit_mov`, `emit_store`, `ptr_store`, `ptr_load`, `ptr_mov`, `emit_resolve`, `branch_if_arm_is_dead`, `frame_alloc`, `anchor`, `finish`, `slot_store`, `slot_load`, `scope_begin`, `scope_end`, `add_scalar`, `load_const`, `alu`, `test_ge`, `emit_write`, `emit_ret`;
- `compile_via_backend`, `AtFloor`'s legacy verbs, `Addressed`, and the legacy `GOLDEN` table;
- the `Codegen` trait, the `Legacy` impl and the `PIXELFLOW_CODEGEN` knob.

`emit/storage.rs`: the whole file (`Slot`, `StackFrame`, `MAX_FRAME`, which moves to `regalloc/resource.rs`).

`emit/regalloc.rs` (legacy):
- `RegSet`, `GprSet`, `MaskSet`;
- the old `RegisterFile` and every one of its fields;
- `MIN_SCRATCH`, `MIN_POINTERS`, `inside`, `Carried`;
- `Scratch` and its constants;
- `Reservations`, `ReadHere`, `guard_sites`, `guarded_arms`, `registers_used`, `no_temps`, `operands_of`, `var_in`;
- `Where`, `FoldRoots`, `NestAllocation`, `Allocation<'a>` and every query on it (`opens_at`, `sibling`, `within`, `fold_opening_at`, `transitions`, `at_head`, `where_at`, `scratch`, `slot_of`, `park`, `placement*`, `if_guards`, `fold_roots`, `spill_slots`, `parks`, `carried`, `fold_reduce_vid`, `fold_parent`, `frame_bytes`, `spill_bytes`);
- `Placement`, `Span`, `Point`, `ScopeCode`, `FoldScope`, `record`, `Boundary`, `Reads`;
- `RegisterAllocator::allocate_nest` and the old `LinearScan::pass`.

`emit/traffic.rs`: `Counting`.

The backends:
- the `Physical` stage, `POOL_BASE`, `x86_64::anchor`, `LeaRip`'s fixed destination, `BroadcastGprs`, `write_address`, `index_into`, `Truncate`;
- `temps_for`, `gpr_temps_for`, `mask_temps_for`, `GatherTemps`, `GatherGprs`;
- `guard_scratch`, `address_in_ip0`, `X16`, `X17`, `CBNZ_W16_OVER_B`;
- `emit_fmov_imm`'s general case, `BUILTIN_HEADROOM`.

Tests deleted because the property is now a type, an invariant, or moot:
- `a_carried_register_is_untouched_by_everything_inside_the_loop`: a carried lease is held to the latch, and join invariant 1 is asserted.
- `a_destination_never_lands_on_a_resident_operand`, `reservations_match_residency_under_pressure`, `a_temp_collides_with_neither_the_operand_nor_the_destination`, `an_if_reserves_a_target_for_each_arm_it_has_to_reload`, `a_resident_operand_reserves_no_reload_target`, `an_op_that_asks_for_no_temp_gets_none`, `a_temp_is_free_again_at_the_next_instruction`: reservations do not exist.
- `sibling_column_folds_share_a_reduce_and_its_slots`: the accident it pins is gone.
- `a_deep_frame_addresses_through_ip0` (`aarch64.rs:1728`): replaced in C4.

**D2 — `program/`.**
- `Outer` placeholder defs (fold schedules hold only their own defs);
- `stays_put`'s placeholder role;
- `program::Class` and `ScheduledOp::class`.

**D3.** `IfGuard`, `ArmPair` (if unused), and the flattening at `program/layout.rs:269-283`.

### 4.2 Added, and why it is the least that works

| Added | Why nothing smaller works |
|---|---|
| A selection phase (`IsaBackend::{lane, context, store, branch, jump, enter, ret}`, `select.rs`) | The allocator cannot see a register an instruction needs until the instruction exists. Any order that allocates first has to be *told*, and being told is a prediction. Escape-hatches step 2 made the predictions more accurate, and every remaining hatch is a place they still run out. |
| A machine IR (`Inst<S>`, `Block`, `Function`, `Target`) | Selection's output must hold chosen instructions with unbound registers. A `Vec<u8>` per op cannot be allocated, and a `ScheduledOp` with a temp count is what exists today. |
| Blocks and labels inside it | If control flow (guards, the wrapper, loops) stayed outside, the code emitting it would run between allocated instructions and need registers. That is H3–H5 and H10–H13 over again. |
| `asm.rs` standing alone | The ontology's Assembler. Today's push/bind/finish builder lets the driver write bytes into it, and that is what H22 is. |
| `Reg`/`Pool`/`Lease`, `FrameSlot`/`Frame`/`SlotLease` | Without them, "only the allocator chooses registers" is enforced by `checked()` asserts and comments, which is today. Ownership is the only mechanism that makes a hand-picked register a compile error. |
| `Stage`, `Rebind`, `walk` | The allocator needs every operand with its access, and the encoder needs named typed fields. One traversal produces both and cannot drift; two hand-written traversals would. |
| `Def`/`Early`/`Tie`, `Spiller` | They put constraints into field types, make "defined twice" unrepresentable, and stop the allocator's own code from writing flags. |
| `Pointer`/`Integer` as distinct classes over `GeneralFile` | Merging today's pointer and scratch pools must not drop `Mem.base: PtrReg`'s guarantee (closure 8). |
| `Origin` | `EmitTraffic` counted by decorating encoders (`Counting`). Counting allocated instructions by why they exist gives the same numbers with no decorator. |
| `ScheduledOp::Outer(Class)` (A5, deleted in D2) | Selection reads the same scoped schedule the legacy pipeline does until D1. A placeholder must say what it is (CLAUDE.md, "extend its type"). |
| The `Codegen` trait and `PIXELFLOW_CODEGEN`, decided once at startup (B4, deleted in D1) | This builds the new pipeline beside the old one, a backend at a time, with each commit live. Per CLAUDE.md, a second implementation of an existing category is a second `impl`. Like `PIXELFLOW_ISA`, an unbuilt choice is refused, never downgraded. |

### 4.3 Considered and refused

- **regalloc2.** Refused by the binding design. The fold-aware carry pricing and its trip-weighted measurements belong to this allocator.
- **A bridge that binds selected instructions with today's plan** (revision 1's S7–S10). It was a prediction with a different spelling. It could not be byte-identical:
  - `Write` is chosen by residency (H28);
  - pool order is emission order (H29);
  - its binding rules were incomplete (landing C).

  Building beside the old pipeline replaces the bridge.
- **Moving tokens before the hatches are gone** (revision 1's S11). That forced `Hatches`, `guard_gpr` and a lookup from number to token. Tokens exist only in the new pipeline, which has no hatches.
- **A `Label` enum keyed by node** (`Head`/`Past`/`Only`/`Join`). The ontology rejects it.
- **`Assembly<B>` with a `LabelField` associated type.** `LabelRef`'s patch function already does this.
- **A shared `Address { base, index, scale, disp }`.** It would undo #1332. Addresses stay per backend.
- **An operand array read by position.** A swapped `vsubps` would go unnoticed.
- **A thread-local pool.** It gives the same per-allocation uniqueness as a per-compile mint, plus a `RefCell` panic and `!Send`.
- **The read-here eviction tiers.** A value read at `i` holds its lease through `i`. That is liveness, not a tier.
- **Loop-plan L3/L4 on the legacy allocator.** That is work on code D1 deletes. Selection builds the latch, so `program/` needs no `Acc`/`Phi`/`Loop`. The latch-order assertion and its hazard ("`Write` reads the binder with no `ValueId` edge") disappear, because the driver emits the latch after the body by construction (closure 3).
- **A memory-operand fold** (`vcvttss2si r64, m32` from a slot, `fold_reload`). The reload plus register form is one more instruction on a rare path. A fold hook is an addition with no measurement behind it.

---

## 5. Commits

### Gates

**G**, for every commit:

```text
cargo test -p pixelflow-codegen -p pixelflow-core -p pixelflow-graphics
cargo xtask isa-matrix --smoke
RUSTFLAGS="-D warnings" cargo check --target aarch64-unknown-linux-gnu --all-targets \
  -p pixelflow-core -p pixelflow-codegen -p pixelflow-ir -p pixelflow-search \
  -p pixelflow-graphics -p pixelflow-pipeline -p pixelflow-compiler   # CI's list, rust.yaml:476-478
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

**Q**, for every commit that touches `aarch64.rs`, `aarch64/table.rs` or anything NEON selects through: the NEON suites run locally under qemu.

```text
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64 -L /usr/aarch64-linux-gnu" \
cargo test --target aarch64-unknown-linux-gnu -p pixelflow-codegen   # + the V suites the commit names
```

**L**, for every commit: the line delta of `pixelflow-codegen/src` (tracked `*.rs`), stated in the commit message. The series must end below its starting count (§0.8).

**M**: CI's `Test on macos-latest`. It is the only gate that runs NEON code and aarch64 clippy (`rust.yaml:565-568`). The cross check runs no tests. A commit that touches `aarch64.rs` or `aarch64/table.rs` is not done until M is green.

**V**, for every commit that moves bytes or values. V is G plus:

| Crate | Suites |
|---|---|
| `pixelflow-codegen/tests/` | `collapse_paths`, `collapse_abi_smoke`, `halve_fold_jit`, `prod_kernel_jit`, `transcendental_jit`, `trig_range_jit`, `muladd_rounding`, `one_compile_per_shape`, `deep_frame`, `empty_fold` |
| `pixelflow-core/tests/` | `fold_factoring`, `guard_fold_price`, `guard_parked_reads`, `guard_sibling_fold`, `reduce_binder_reads_bound_buffer`, `kernel_bake`, `mask_support_of_a_built_kernel` |
| `pixelflow-graphics/tests/` | `glyph_exact_area`, `glyph_area_edge_cases`, `glyph_atlas_golden`, `freetype_oracle`, `loop_blinn_winding`, `rendering_contract`, `pixel_contract` |
| `emit/mod.rs` | `a_surviving_reduce_compiles_and_runs` (`2624`), `a_folds_spill_slots_do_not_alias_its_parents` (`2783`), `a_reduce_three_deep_compiles_and_runs` (`2910`), `sibling_folds_sharing_a_binder_node_read_their_own_counters` (`2962`), `a_folds_roots_are_placed_by_the_budget` (`3055`), `a_fold_owned_by_an_arm_is_guarded` (`4355`), `a_deep_spill_frame_compiles_correctly` (`4797`), `a_uniform_past_the_old_u16_width_loads_on_every_backend` (`5699`) |

**K(tier)**: V run with `PIXELFLOW_CODEGEN=selection PIXELFLOW_ISA=<tier>`, through the production API. In B4–B8, K names the subset that commit makes pass. In B9 and later it is all of V.

**`GOLDEN`** is at `emit/mod.rs:6657`; the test is at `6706`. Its doc says:

> *"a refactor does not edit this table; an intentional byte change does, in a commit of its own that says why."*

- Each byte-moving commit below is its change, plus the re-baselined rows, plus the reason, and nothing else.
- A row's provenance is `git log -L` on it. `GOLDEN`'s doc names no commit hash: a commit cannot name itself.
- A row moving outside the commit's predicted set is a finding, explained in that commit.
- `GOLDEN_SELECTED` is the new pipeline's table. It is born in B10 for AVX2, and gains AVX-512 in C2 and NEON in C4. It obeys the same rule from birth. In D1 it replaces `GOLDEN`.

**Transitional dead code.** A module that only the next one or two commits make live carries `#[expect(dead_code, reason = "live from <commit>")]`. Because it is `expect` and not `allow`, CI fails the commit that makes it live until the attribute is removed.

### Phase A: prepare the legacy pipeline

Every commit in this phase is live in production.

#### A1: Coverage rows in `GOLDEN`

- **Files:** `tests/support/sibling_rows.rs`, `emit/mod.rs` (the `ROWS`/`GOLDEN` tables).
- **Add** rows at `Width::Remainder`, all compiled by today's harness:

  | Row | Contents |
  |---|---|
  | `unary_ops` | every `REQUIRED_UNARY_OPS` op (`coverage.rs`) |
  | `binary_ops` | every `REQUIRED_BINARY_OPS` op, with the comparisons consumed by `BitAnd` and `If` |
  | `shift_muladd_blend` | `Shl`, `Shr`, `MulAdd`, an unguarded `If` |
  | `memory` | a lane-varying gather, a lane-uniform broadcast, a uniform at element 0 and one at element 5,000 |
  | `deep_frame` | `deep_frame(n)` in `tests/support/deep_frame.rs` (`traffic.rs`'s `wide_live_range_kernel` is private to its tests and spills 96 bytes at any `n` on NEON), with `n` past the 64 KiB edge: the NEON spill area alone is past `ldr q`'s reach. `tests/deep_frame.rs` compiles it through `compile`, asserting `spill_bytes` and its values against a scalar reference |

- **Bytes:** nothing moves; the rows are new.
- **Gate:** G.

#### A2: A label is minted with what it names

- **Files:** `emit/mod.rs`, `emit/asm.rs` (new: `Label`, `Labels`, `Patch`), `x86_64.rs`, `avx2.rs`, `avx512.rs`, `aarch64.rs`, `traffic.rs`.
- **Change:**
  - `compile_via_backend` owns one `Labels` per function and passes `&mut Labels` into `emit_scope`.
  - Each `format!` site (`mod.rs:1439`, `1836-1837`, `1937`) mints at the point where the branch is created. The `Past` label travels in `PendingBranch` (`mod.rs:1398`) to its bind point.
  - The pool's label is minted once per function, in `compile_via_backend`, and handed to `IsaBackend::anchor` and `IsaBackend::finish`. (`begin` runs once per scope, not once per function, and a backend holding the label would need an `Option` to mint it on the first call.)
  - `LabelRef { label, patch }` keeps its patch function, typed `asm::Patch`.
- **Remove:** see §4.1, A2.
- **Tests:** label tests mint their labels. `a_label_bound_twice_is_a_bug` binds one minted label twice.
- **Bytes:** identical. Labels emit nothing, and fixups resolve in push order.
- **Gate:** G, with `GOLDEN` not edited.

#### A3: An empty fold is its identity

- **Files:** `program/lower.rs`, `pipeline.rs` (`emit::compile` refuses a zero extent: a zero-width fold reaches `lower.rs` as a body that was never mapped), `emit/mod.rs` (`GOLDEN`'s doc loses its commit hash).
- **Change:**
  - `mark_reachable` (`lower.rs:21`) does not descend into a `Reduce` whose fold `is_empty()`.
  - The `Reduce` arm (`lower.rs:293`) lowers that fold to `Const(fold.monoid().identity())`.
  - A `SEQ` fold is the exception, and the identity constant is not "never read": a `Seq` that names it keeps it live, and the schedule stores a dead `0.0` in the root scope (measured: one store and one instruction more at `POINT`). A `SEQ` fold over nothing has no def, and a `Seq` with such an operand is its other operand (two of them, no def at all). (P1 moved this to `pack`: the only empty `SEQ` folds that reached lowering were its empty main column fold, so `pack` no longer builds one and lowering has no `SEQ` case.)
- **Bytes:** row 0 (`glyph_like_w1`) shrinks on all three backends (1016/984/592 to 724/676/400). It is the only row with an empty main column fold. Any other row moving is a bug in this commit.
- **Tests:** `tests/empty_fold.rs`: through `compile`, a kernel with a surviving `SUM` at width 1 has two scopes fewer than at width 37 (the main column fold and the sum inside it), and its samples, at both widths, equal the closed form; a lattice with no column is refused (`should_panic`, "degenerate extent"). (An empty-`SUM` test through the optimizer would be vacuous: the e-graph's `EmptyFold`, `fold_rules.rs:447-453`, already rewrites it. P1 added one through `emit::compile`, which does not optimize.)
- **Gate:** V, plus a row-0 re-baseline.

#### A4: `MulAdd` always fuses

- **Files:** `emit/mod.rs`, the three backends, `tests/muladd_rounding.rs`, `emit/coverage.rs`, `CLAUDE.md`, and the two comments in `pixelflow-ir` (`kind.rs`, `passes.rs`) that named the decomposed form.
- **Change:** delete the decomposed form (§4.1, A4). The fused form already fits: `a` and `b` reload into the two reload reservations, and `c` reloads into the destination (`operand_sources`, `mod.rs:833-882`).
- **Values:**
  - One rounding on every target, which matches the folder's `libm::fmaf`.
  - `a_spilled_muladd_rounds_twice_on_every_target` becomes `…_rounds_once_on_every_target`, asserting the fused bits.
  - The two `resolve_muladd_decomposed_*` tests become `resolve_muladd_fuses_with_both_multiplicands_spilled` and `resolve_muladd_reloads_a_spilled_addend_into_dst`; `a_deferred_c_is_reloaded_between_the_multiply_and_the_add` goes with `DeferredReload`. (P1 deleted the two in-file tests: they called crate-private `resolve_operands`, and `a_spilled_muladd_rounds_once_on_every_target` now shows the multiplicands' store and reload through `EmitTraffic`.)
  - CLAUDE.md's `MulAdd` row loses "two only where the emitter decomposes it".
- **Bytes:** only kernels where both multiplicands of a `MulAdd` were non-resident. GOLDEN and `glyph_branches` are identical; the chrome sphere in `the_scenes_emit_their_pinned_code` keeps its length on every tier and changes its digest (a different register holds the multiply's inputs).
- **Gate:** V, plus a re-baseline of those cells.

#### A5: A placeholder says what it is

- **Files:** `program/mod.rs`, `program/scopes.rs`, `emit/regalloc.rs`, `emit/mod.rs`.
- **Add:** `ScheduledOp::Outer(Class)`, "a read of a value an enclosing scope computes". `class()` returns the carried class. `operands()` and `structural_children()` yield nothing.
- **Change:** `place_roots` (`scopes.rs:277`) writes `Outer(def.op.class())` for every moved def. Today it writes `Const(0.0)` for vectors and leaves pointers as `Context`. Every legacy match over `ScheduledOp` handles `Outer` as it handled the placeholder: the def is skipped through `parked_by_an_enclosing_scope`.
- **Bytes:** identical.
- **Gate:** G.

#### A6: One `loads` count

- **Files:** `emit/traffic.rs`, and in `pixelflow-pipeline`: `collapse_bench/{mod.rs, row.rs, predict.rs}`. (`bin/corpus_gaps.rs` names neither field.)
- **Change:**
  - `ScopeTraffic::{loads_transient, loads_kept}` become `loads`.
  - `memory_ops` is `loads + stores`.
  - The pipeline's `SCHEMA` (`row.rs:13`) becomes `"collapse-cost-v2"`. Its predictors already use only the sum (`predict.rs:94,104`).
- **Why:** the split is defined by legacy emission paths that the new allocator does not have (closure 6). The new pipeline's `Origin::Reload` gives the sum with the same meaning.
- **Bytes:** identical.
- **Gate:** G.

#### A7: The allocator's policies stand alone

- **Files:** `emit/regalloc.rs` becomes `emit/regalloc/mod.rs`; new `emit/regalloc/policy.rs`.
- **Move:** `EvictionRank` with `ReadHere` (built by `EvictionRank::new`, its fields private), and the pure pricing inside `plan_carries`: `reads_saved` (reads × trips), `loop_state_saved` (a binder's or an accumulator's two accesses per trip), `Candidate` and `carried` (the budget order). The legacy pipeline calls them from their new home. B6's one latch copy per trip for a head parameter is a new pricing beside `loop_state_saved`, not a move.
- **Bytes:** identical.
- **Gate:** G.

#### A8: The assembler stands alone, one program per kernel

- **Files:** `emit/asm.rs` (`Item`, `AsmProgram`, `Encoding`, `assemble`), `emit/mod.rs`, `x86_64.rs`, `aarch64.rs`, the ontology's Assembly program and Assembler entries.
- **Change:**
  - The legacy `Assembly` becomes a front end that accumulates `asm::Item`s. Raw bytes become `Item::Bytes`, and a `LabelRef` becomes an `Encoding::field`. It owns the kernel's `Labels`.
  - There is one `Assembly` per kernel, threaded through `emit_scope(allocation, backend, &mut asm)` and every scope nested in it. Nothing is spliced, items or bytes. (Deviation from the first draft, "`emit_scope` returns its items": the body no longer has to be emitted before the frame around it, because the frame's size is known before it.)
  - A `LabelRef` carries `at`, the field's offset in its instruction, and a `Patch` receives the field's own position.
  - The pool is the data section (`Assembly::pool`: aligned when it holds anything, bound either way).
  - `asm::assemble` produces the code, as a `Vec<u8>`.
  - `scripts/check_emit_boundary.py` gains the rule "`emit/asm.rs` imports nothing from the crate" (no `crate::`, `super::` or `pixelflow_` path), plus a self-test case that a violation is flagged.
- **Bytes:** identical.
- **Gate:** G.

#### A9: x86's general-register instructions are values

- **Files:** `emit/mod.rs`, `x86_64.rs`, and the three `IsaBackend` implementations (`avx2.rs`, `avx512.rs`, `aarch64.rs`) and `traffic.rs` for the `emit_ret` change below.
- **Add:**
  - In `mod.rs`: `Class` and its markers `Pointer`, `Integer` and `Flags`, `Stage`, and an interim stage `Physical` whose register types are the legacy newtypes (`Class` carries a `type Physical` until D1), with `Target = Label` and `FrameSize = u32`. The rest of §2.1's and §2.6's vocabulary arrives with its first reader and is not built here: `File`, `FileId`, `ClassId`, `Spill` and the `sealed` supertraits (B1, B2), the `Vector` marker and `Stage::Early` (A10a, A10b), the `Opmask` marker (A11b), `Stage::Slot` (B-series).
  - In `x86_64.rs`: `Gp<S>`, with named-field variants `Mov`, `MovImm32`, `Movabs`, `Imul`, `Add`, `Lea4`, `MovLoad`, `MovStore`, `Test`, `Jcc { cond, flags, taken }`, `Jmp { to }`, `LeaRip { dst, to }`, `Enter` and `Ret`. Each one that writes `EFLAGS` has a `flags` field (`()` at `Physical`).
  - `Mem<S, D: Disp>` keeps `Disp` typed. Slots are always `disp32`, which matters for EVEX `disp8` scaling (allocation F13).
  - `Gp<Physical>` implements the legacy `AsmInsn` (`emit_into`, and `label_ref` for the three arms with a label field). The `Encoding` form of `encode` arrives when `AsmInsn` retires, because `Assembly::push` is the one caller and takes `AsmInsn`.
- **Change:** today's x86 `Inst` and every free GPR byte-writer become `Gp` arms. Each hardcoded register becomes a literal at its one construction site in the legacy driver, for example `Test { src: gpr::RAX }`. `IsaBackend::frame_free` is merged into `emit_ret(code, bytes)`, which on x86 is the `Ret` arm (`add rsp, size; vzeroupper; ret`), and `Enter` is `frame_alloc`.
- **Deviation from the first draft:**
  - `Cvtt` is not a `Gp` arm. Its bytes are the tier's (VEX or EVEX), so it is A10a's and A11a's convert arm. `Convert` survived A10 only because AVX-512's closures still built byte-writers; A11a replaces it with the `Truncate` trait, which D1 deletes with `write_address` and `index_into`.
  - `Jcc` has no `next` field and there is no `Fallthrough` arm. The legacy driver has no next block to name; the block builder adds both (B1).
  - `Test` is `test r32, r32` alone. `cmp al, 0xFF` is the guard sequence (A10b).
  - `Mov` is a pointer copy (`Pointer`); the allocator's own copy verb is B's.
- **Tests:** keep every SDM byte pin, rewritten to build a `Gp<Physical>`.
- **Bytes:** identical.
- **Gate:** G, Q (the `emit_ret` change touches `aarch64.rs`).

#### A10a / A10b: AVX2's VEX instructions are values

- **Files:** `avx2.rs`, and `mod.rs` and `x86_64.rs` (`Vector`, `Gp::CmpByte`, the guard's `Gp` arms).
- **A10a:** `avx2::Inst<S>` arms for ALU, unary, round, compare, shift, convert (`Cvtt`, `Movq`), copy and `Ones`, and `Fma231`; the `Vector` marker (`Class::Physical = Reg`). A blend is three `Alu` arms, not an arm of its own: it is the and/andn/or sequence, and a `vblendvps` would move bytes.
- **A10b:** memory (`Load`, `Store`, `CvttMem`, `ExtractLane`), broadcast, gather (the first reader of `Stage::Early`), `MoveMask` and the guard sequence (`MoveMask`, `Gp::CmpByte`, `Gp::Jcc`). The masked store is AVX-512's, so A11b's.
- **Change:** each half deletes the byte-writers it replaces, and keeps their pins (for example `emit_movmskps_eax_gathers_…`, `avx2.rs:825`).
- **Bytes:** identical.
- **Gate:** G each.

#### A11a / A11b: AVX-512's EVEX and mask instructions are values

- **Files:** `avx512.rs`.
- **A11a:** `avx512::Inst<S>` arms for every EVEX instruction that does not name a `k` register: ALU, unary, round, shift, `Fma231`, the `If` blend (`Blend`, one `vpternlogd` with the select table), copy, convert (`Cvtt`, `CvttMem`, `Movq`, `InsertHigh`), and memory (`Load`, `Store`, `StoreBatch`, `Broadcast`, `BroadcastIndexed`). The operation enums both x86 tiers name (`Alu`, `Lanewise`, `Rounding`, `Direction`) move to `x86_64.rs`, each tier giving them its own `vex()` or `evex()`; `x86_64::Convert` and `index_into`'s function-pointer pair give way to a `Truncate` trait that the two `Inst`s implement, so `write_address` is generic over the tier.
- **A11b:** the `Opmask`-file fields:
  - `CmpK { dst: Write<Opmask> }`, `Movm2d`, `Ptestm`, `Kand`, `Kor`;
  - `BlendK { dst, mask: Read<Opmask>, a, b }` (`vblendmps zmm{k}`);
  - `KorTest { flags, k: Read<Opmask> }`, which encodes `k.number()`;
  - `Kmovw`, `KOnes`;
  - `Gather { dst: Early, mask: Tie<Opmask> }`, encoding `aaa = mask.number()`;
  - the masked store.

  `eax` and `k1` become literals at the construction site.
- **Deferred from A9, whose first reader is here:** the `Opmask` marker (`Class::Physical = KReg`).
- **Bytes:** identical.
- **Gate:** G each.

#### A12a / A12b: aarch64's instructions are values

- **Files:** `aarch64/table.rs`, `aarch64.rs`.
- **A12a:** the `table.rs` encoding structs, which are already generic in their register types (`Binary<OPCODE, D, L, R>`, `table.rs:225`), are instantiated at `Physical`. A named-field `aarch64::Inst<S>` replaces the 41 positional variants (`aarch64.rs:41`).
- **A12b:** the sequences:
  - `FmovToGp { dst: Write<Integer>, src }`;
  - `MvnW { dst: Tie<Integer> }`;
  - `CbzFar { test, taken, next }` (an `imm26` field);
  - `AddImm`, `AdrpAdd`, `InsFirst`, `InsLane { v: Tie<Vector> }`, `St1Lane { base: Tie<Pointer> }`.

  `address_in_ip0` is called explicitly by the legacy driver, with `X16` written in.
- **Bytes:** identical.
- **Gate:** G and M, each.

### Phase B: the selection pipeline beside the legacy one, AVX2 first

#### B1: The machine IR

- **Files:** `emit/mod.rs` (`Value`, `ValueName`, `Selected`, `Rebind`, `Operand`, `Access`, `Target`, `Block`, `Function`, `Entry`, `Loop`, `Constants`, `Constant`, `LaneOp`, `Store`, `Test`, `Edges`, `IsaBackend`), `emit/build.rs` (§2.9).
- **Change:** rename the legacy trait `IsaBackend` to `LegacyBackend` (mechanical), so that the contract's name is the new trait from the first commit.
- **Deferred from A9, whose first reader is here:** the `sealed` supertraits of `Class`, `Jcc`'s `next` field and the `Fallthrough` arm of `Gp` (the block builder is the first thing with a next block to name).
- **Tests:** none of its own (§0.6). `finish`'s and `push`'s checks are production assertions, and they are exercised by every kernel B3 onward selects.
- **Status:** `expect(dead_code)` until B4.
- **Gate:** G.

#### B2: Registers and frame slots are tokens

- **Files:** `emit/regalloc/resource.rs` (§2.2, §2.4), the `Bound` stage in `mod.rs`.
- **Deferred from A9, whose first reader is here:** `File` and its four files, `FileId`, `ClassId`, `Spill`, and `Stage::Slot` (the frame slot is a token).
- **Tests:** none of its own (§0.6). From B4, the leases and the frame are exercised through `compile`. The narrow region's `BudgetExceeded` is reached by a kernel with more than 4,095 live narrow values, if one can be built through the production API; if none can, the bound is an assertion, not a test.
- **Status:** `expect(dead_code)` until B4.
- **Gate:** G.

#### B3: Selection, AVX2 core

- **Files:** `emit/select.rs` (the driver of §2.10, with scoped `Bindings`), `avx2.rs`.
- **Add to AVX2:**
  - `lane` for `Const` (RIP-relative `LoadConst`; `Ones`; `vxorps` zero), `Lanes`, `Unary`, `Binary`, `MulAdd`, `Blend` and `Shift`;
  - `store` (`Cvtt`, `Imul`, `Add`, `Lea4`, then `vmovups` or the masked remainder);
  - `branch`, `jump`, `enter` and `ret`;
  - `walk`, and `encode` at `Bound`, built through `asm::Encoding`: B3 is the first reader of `encode` at `Bound`, and A9's `Gp<Physical>` implements only the legacy `AsmInsn` until D1 deletes it.
- **The driver covers:** the body, folds as blocks (§2.10), `Outer`/`Var`/`Reduce`/`Seq`/`Write`.
- **The driver refuses**, through `unimplemented_op`, which names the op: `Context`, `Uniform`, `Gather`, `Broadcast`, and guarded `If` arms. These arrive in B8–B9.
- **Tests:** select the `GOLDEN` rows that use only these ops, and assert `finish`'s invariants and the selected loop shape (one backward branch per fold).
- **Status:** `expect(dead_code)` until B4.
- **Gate:** G.

#### B4: The local allocator, and the pipeline goes live behind a knob

- **Files:** `emit/regalloc/mod.rs`, `emit/mod.rs` (`trait Codegen`, `Legacy<B>`, `Selection<B>`, `compile_on`), `emit/traffic.rs` (`EmitTraffic::of(&Allocated, &Assembled)`).
- **The allocator** is the simplest correct one:
  - every value is stored at its definition;
  - every read is reloaded into a fresh value that dies at its instruction;
  - writes take free leases;
  - ties take the dying reload's lease;
  - block parameters live in slots, and arguments are stored to the parameters' slots;
  - `Flags` leases are held from their definition to their read.

  It exercises leases, slots, `Tie`, `Early`, `Spiller`, binding and the per-instruction register-count assertion.
- **The knob:** `PIXELFLOW_CODEGEN=legacy|selection`, read once at the first compile, like `isa::detect`.
  - The default is `legacy`.
  - `selection` on a tier without a selection backend panics, naming the tier. It is refused, never downgraded.
- **`CompileResult`** fields are defined as in §2.12.
- **Bytes:** legacy identical.
- **Gate:** G, plus K(avx2) on `a_surviving_reduce_compiles_and_runs`, `a_reduce_three_deep_compiles_and_runs`, `a_folds_spill_slots_do_not_alias_its_parents`, `sibling_folds_sharing_a_binder_node_read_their_own_counters`, `halve_fold_jit`, `muladd_rounding`, and `collapse_paths`' arithmetic kernels.

#### B5: Residency

- **Files:** `emit/regalloc/mod.rs`.
- **Change:**
  - Values stay in registers between instructions.
  - Eviction by `EvictionRank` (`policy.rs`).
  - The store is inserted retroactively after the definition.
  - Split reloads are kept until evicted.
  - Plain writes take a dying read's lease.
  - Forward joins take the intersection of their predecessors' states (§1.3, invariant 2).
  - Loop heads are still flushed: every live value is slot-homed across a loop until B6.
- **Gate:** K(avx2) as in B4. Add a test that a straight-line kernel with fewer live values than registers allocates no slot.

#### B6: Loops

- **Files:** `emit/regalloc/mod.rs`, `policy.rs`.
- **Change:**
  - The carry plan (§2.11), with `CARRY_RESERVE = 7` and `GENERAL_CARRY_RESERVE = 5` as named constants documented with their derivation.
  - Head parameters live in registers or slots.
  - The back edge's parallel move, with cycles broken through a fresh value.
  - The latch hint.
  - Join invariant 1 asserted at each latch.
- **Gate:** K(avx2) as in B4, plus `a_folds_roots_are_placed_by_the_budget` rewritten as `traffic.loads` before and after a carry-budget-sized root count.

#### B7: Rematerialization

- **Files:** `emit/regalloc/mod.rs`, `avx2.rs` (`rematerializable`).
- **Change:**
  - A rematerializable definition is placed only before a read that needs it in a register, by re-emitting it with a fresh `Def` (`Origin::Remat`). One nobody reads is never placed.
  - Its eviction is free, the remat tier of `EvictionRank`.
- **Gate:** K(avx2) as in B4, plus `constants_are_rematerialized_rather_than_spilled` and `belady_evicts_the_value_used_farthest_out` rewritten against compiled kernels' `traffic`.

#### B8: AVX2 memory operations

- **Files:** `avx2.rs`, `select.rs`.
- **Add:**
  - `context`: `mov p, [ctx + 8·slot]`, refused past `disp32` as today;
  - `Uniform`;
  - `Gather`: `vcvttps2dq`, `Ones` mask, then `Gather { dst: Early, mask: Tie }`;
  - `Broadcast`: `Cvtt`, then `vbroadcastss [base + idx·4]`.
- **Gate:** K(avx2) plus `reduce_binder_reads_bound_buffer`, `glyph_*`, `freetype_oracle`.

#### B9: An `If` is blocks

- **Files:** `select.rs`, `avx2.rs`.
- **Add:**
  - Guarded arms and the uniform wrapper, as in §2.10, read off `IfGuard`.
  - AVX2's `branch` is `MoveMask` → `Test` (`dead: True`) or `CmpByte` (`dead: False`) → `Jcc`, and the encoder picks the `cmp al` form when `g` is in `rax`.
- **Gate:** K(avx2) is now all of V, plus `guard_fold_price`, `guard_parked_reads`, `guard_sibling_fold`, `a_fold_owned_by_an_arm_is_guarded`.

#### B10: The gate for the new pipeline

- **Files:** `emit/mod.rs` tests, `.github/workflows/rust.yaml`.
- **Add:**
  - **`GOLDEN_SELECTED[avx2]`** over every `GOLDEN` and A1 row.
  - **A quality ratchet,** `selection_moves_no_more_memory_than_legacy`. Per row, trip-weighted `loads + stores + remats` of the selection pipeline must be at most 1.25 × legacy's + 8, and the sum over rows at most legacy's. This is the static predictor escape-hatches found tracks wall clock 98–99% of the time.
  - **A compile-size bound,** `selection_stays_linear`. Per row, selected instructions are at most 4 × scheduled ops, and inserted instructions at most 2 × selected. The scan is O(instructions × file size) by construction, so this bounds its `n`.
  - **Register coverage through the production API** (§0.6), in `tests/`: per tier, a pressure kernel whose live vectors outnumber `VectorFile`, and whose live pointers and integers (bound buffers, uniforms, carries) outnumber `GeneralFile`, checked by its values against a scalar reference computed in the test. This is what reaches `cmp sil`/`cmp r9b`'s REX forms (allocation F16). No `samples()` table, no test-only constructor.
  - **CI, in jobs that are already required** (§0.7): the `isa-matrix` job runs V with `PIXELFLOW_CODEGEN=selection PIXELFLOW_ISA=avx2`.
  - **The ratchet compares against a pinned table.** The knob is read once per process, so `selection_moves_no_more_memory_than_legacy` reads legacy's per-row numbers from a table generated once in this commit, and says how to regenerate it.
- **Gate:** G, plus the new steps.

### Phase C: switch each backend

#### C1: AVX2 runs the selection pipeline

- **Files:** `emit/mod.rs` (the knob's default per tier).
- **Bytes:** every AVX2 column of `GOLDEN` moves. Reason: allocation is now over selected instructions, the pool is RIP-relative, and the ABI and pool registers are allocated. `GOLDEN`'s AVX2 column is re-baselined to `GOLDEN_SELECTED`'s.
- **Report** in the commit (evidence for the decision, not a caveat): callgrind instructions of `emit::compile` on `'@'`@16 and `8`@32, and the `font_rendering` bake time against legacy. A rise of more than 20% blocks the commit.
- **Gate:** V, the ratchet, the selection steps, and `isa-matrix --smoke` (avx2 now through selection).

#### C2: AVX-512 selection, beside

- **Files:** `avx512.rs`.
- **Add:**
  - `type Lane` as in §0.1: comparisons are `vcmpps k`; `BitAnd`/`BitOr` of two `Opmask` lanes are `kandw`/`korw`; an `Opmask` lane read by an instruction with no `k` form is `vpmovm2d` first;
  - `Blend` as `vblendmps zmm{k}` on an `Opmask` condition, `vpternlogd 0xCA` on a `Vector` one;
  - `Gather` through `KOnes`;
  - `branch` as `KorTest(k)` → `Jcc` on an `Opmask` condition, with `Ptestm` first on a `Vector` one;
  - the masked remainder store through `MovImm32` → `Kmovw`;
  - a pin that a zmm slot at offset 64 encodes `disp32 = 0x40` (EVEX `disp8` scaling).
- **`GOLDEN_SELECTED[avx512]`.** The `isa-matrix` step adds `PIXELFLOW_ISA=avx512` V.
- **Gate:** G plus the steps.

#### C3: AVX-512 runs the selection pipeline

- **Bytes:** every AVX-512 column moves, for C1's reason, plus predicates kept in `k` (§0.1): each compare feeding a blend, a guard or mask logic loses its `vpmovm2d`, and each guard loses its `vptestmd`.
- **Report and gate:** as C1.

#### C4: NEON selection, beside

- **Files:** `aarch64.rs`, `aarch64/table.rs`, `.github/workflows/rust.yaml` (a macOS step running V with the knob).
- **Add:**
  - `enter` defines the pool base `p = AdrpAdd(pool section)`, rematerializable.
  - Constants: `movi`, `fmov`, or `ldr q, [p, #offset]`. `POOL_REACH = 4096`; past it, `BudgetExceeded`.
  - `Uniform` and `Context` with the address arithmetic of H6.
  - The gather as `UmovW`/`LdrS`/`InsFirst`/`InsLane`.
  - `Rsqrt`/`Recip` as SSA sequences: `e = frsqrte(x)`; `t = fmul(e, e)`; `t' = frsqrts(x, t)` (a `Tie` on `t`); `r = fmul(e, t')` (a `Tie` on `e`).
  - `branch` as `Umaxv`/`Uminv` → `FmovToGp` → (`MvnW`) → `CbzFar`.
  - `spill`/`reload` emitting `SlotAddr` for deep vector slots.
  - The remainder store through `St1Lane` with a `Tie<Pointer>` base.
- **Tests:**
  - New, value: `a_context_slot_past_imm12_loads`, at slot 4,096 (this panics on legacy NEON).
  - New: `a_deep_vector_slot_is_addressed_through_an_allocated_pointer`, replacing `a_deep_frame_addresses_through_ip0`.
  - Rewritten: `a_uniform_past_the_old_u16_width_loads_on_every_backend` (`mod.rs:5699`, asserts at `5785-5789`) asserts `add xN, base, #hi, lsl 12; ldr s, [xN, #lo]`, with `N` read from the bytes, not fixed.
  - **`GOLDEN_SELECTED[neon]`**, including A1's `deep_frame` row, which crosses 64 KiB.
- **Gate:** G and M (plus the knob step).

#### C5: NEON runs the selection pipeline

- **Bytes:** every NEON column moves, for C1's reason plus `x16`/`x17` joining `General`.
- **Report and gate:** as C1, plus M.

#### C6: Probe the stack in `Enter`

- **Files:** `x86_64.rs`, `aarch64.rs`.
- **Change:** a frame larger than one page is entered one page at a time, with no register:
  - x86: `sub rsp, 4096; mov dword [rsp], 0`, repeated, then the remainder;
  - aarch64: `sub sp, sp, #1, lsl 12; str xzr, [sp]`, repeated, then the remainder.

  `MAX_FRAME` (2 MiB) bounds this at 512 pairs.
- **Why:** spawned threads get Rust's 2 MiB default stack behind one guard page, and a single `sub` can skip it (allocation F12).
- **Bytes:** only rows whose frame exceeds 4 KiB move (`parked_roots`, `deep_frame`). Every other row must not move.
- **Gate:** V, M, and a re-baseline of those rows.

### Phase D: delete the legacy pipeline

#### D1: Delete the legacy pipeline and the knob

- **Files:** everything §4.1 lists under D1, `rust.yaml` (the selection steps and the macOS knob step).
- **Change:**
  - `GOLDEN_SELECTED` becomes `GOLDEN`.
  - The quality ratchet becomes a pinned ceiling table per row and backend, holding the selection pipeline's numbers at this commit.
- **Bytes:** identical.
- **Gate:** G and M. `scripts/check-emit-boundary.sh` passes.

#### D2: A fold's schedule holds only its own definitions

- **Files:** `program/scopes.rs`, `program/mod.rs`, `emit/select.rs`.
- **Change:**
  - `place_roots` removes moved defs instead of writing `Outer`.
  - `stays_put`'s non-binder role goes. A `Reduce` placeholder is no longer kept, and the driver resolves it through `Bindings`.
  - Delete `Outer`, `program::Class` and `ScheduledOp::class`. With the placeholders gone, `scopes.rs:276` was the last reader.
- **Bytes:** identical.
- **Gate:** G.

#### D3: Arms are read off the region tree

- **Files:** `program/layout.rs`, `program/mod.rs`, `emit/select.rs`.
- **Change:** `Layout` hands selection the nested blocks it already builds (`Blocks`, `layout.rs:67-109`): items are a def, an arm (its `If`, which arm, its items), or a fold opening. Selection emits the guard and the `past` block around each arm subtree. Delete `IfGuard`, the flattening (`layout.rs:269-283`) and the range bookkeeping.
- **Bytes:** identical.
- **Gate:** G.

---

## 6. Risks, compile-time cost, and what this does not do

### Risks

1. **B5–B7 carry the allocation quality.** The quality ratchet (B10) is the check, and the trip-weighted memory traffic is the measure escape-hatches validated against wall clock. Correctness rests on V; there is no second JIT to compare against, by design (#1320).
2. **The carry reserves are fitted, not derived.** `CARRY_RESERVE = 7` and `GENERAL_CARRY_RESERVE = 5` are kept so that the switch is about representation. aarch64 gains one general carry (13 against today's 12), and that is attributed in C5. Re-deriving both reserves is the first follow-up.
3. **The join invariants are asserted, not built around.** No edge blocks are built. If invariant 1 or 2 fires, the remedy is an edge block on that edge: one `jmp` per trip.
4. **Dominance is relaxed across uniform guards.** An allocator that "repaired" an undefined path with a phi would be wrong. The relaxation is documented on `Function`, and `guard_parked_reads` and `guard_sibling_fold` exercise it.
5. **Affine is not linear.** A dropped `Def` is a panic at `finish`, not a compile error (§2.13).
6. **x86 encodings depend on the register.** `cmp g8, imm8` is two bytes for `al`, three for `cl`/`dl`/`bl`, and four with REX for `sil`/`dil` (`40 80 FE FF`) and for `r8b`–`r11b`. A guard's length now depends on allocation, as every other instruction's already does.
7. **The knob is temporary production surface,** like `PIXELFLOW_ISA`. It is decided once and refuses unbuilt tiers. D1 deletes it. Until then, two pipelines answer for the same kernel, and the job and the ratchet hold both to V.
8. **Instruction-enum size.** The largest variant (the gather) is about 48 bytes at `Selected`, and every `Bound` field is a reference. A glyph body is about 1,030 scheduled ops, which comes to about 2,000 instructions and about 100 KB per compile. That is noise next to saturation.

### Compile time on glyph bakes

Selection, `walk`, binding, encoding and assembly are each one linear pass. The scan is O(n × k) with `n` the number of instructions: about 1.5–2× today's op count on AVX2, about 2× on NEON gathers. `k` is at most 32.

What goes away:

- `emit_scope`'s per-scope `locs`/`moves` tables;
- `Counting`'s indirection;
- the reservation bookkeeping, including `operand_sources`' re-derivations per eviction candidate;
- the operand lists, which are computed once per instruction.

The expectation is that emit plus allocation stays within ±20% of today on `'@'`@16. The checks are `selection_stays_linear` (deterministic, in CI) and the callgrind report in C1, C3 and C5, where a rise over 20% blocks the commit.

### What this does not do

- **Retune the carry reserves,** or price carries by dynamic traffic.
- **Choose the FMA form by which operand dies** (`132`/`213`/`231`). Selection always ties `c`, and the allocator copies `c` when it is still live. Choosing would need a commutation hook in the allocator; that waits for a measurement.
- **Fold a reload into a memory operand** (`vcvttss2si r64, m32`).
- **Bias x86's frame by 8 so that vector slots are 16-byte aligned** (allocation F18). `rsp` is 8 mod 16 after the call. 32/64-byte alignment needs a frame pointer.
- **Test a guard with `vptest`,** removing its general register.
- **Rematerialize `Context` loads from `ctx`** instead of carrying them.
- **Load aarch64 constants with `LDR (literal)`.** Its `imm19` reach of ±1 MiB would become a limit on kernel size; `adrp` reaches ±4 GiB.
- **Add callee-saved registers** to any file (that needs a prologue), compile a `Ref` as a call, or emit a `wasm32` backend. (Predicates in `k` registers *are* done: §0.1.)
- **Give the vector file's masks a class of their own.** A `LaneMask` class over `VectorFile` would make the assembler refuse legal NEON and AVX2 programs (§0.2). "A mask is not a number" belongs with the IR typing that replaces `OpKind::is_bitwise_domain()`.
- **Widen `ScheduledOp::Context(u16)`** to the control plane's 64 bits. It is a separate finding against CLAUDE.md.
- **Make a trip count a runtime value.** It stays compile-time, keyed by shape.

---

## Appendix A: every review point, and what was done with it

### Allocation review

| # | Point | Disposition |
|---|---|---|
| F1 | Moving tokens out of `&'m mut Pool` contradicts `&'m Reg` operands. Frame slots likewise. Brand by backend | **Adopted.** `Lease`, `SlotLease` and a frozen frame (§2.2, §2.4); `Reg<B, F>` |
| F2 | NEON non-frame loads use IP0 or panic (`LdrS` uniform and binder, `LdrX` context) | **Adopted.** H6 names all three callers. C4 selects address arithmetic, and rewrites both tests |
| F3 | Fixed point not one round; offsets versus `SlotName`; alignment | **Adopted.** Narrow region sized from its peak live count at offset 0, vector region aligned after it, no iteration (§1.3, §2.4) |
| F4 | SP is never named | **Adopted.** `Frame` owns SP; `SlotAddr`; `Ret` is the exit terminator (H25, H27) |
| F5 | Frame instructions built after allocation | **Adopted.** `Enter`/`Ret` selected up front with a `FrameSize` operand bound by the allocator |
| F6 | Fall-through successor is not an operand | **Adopted.** `taken`/`next` targets; the zero-width fall-through field is checked |
| F7 | Critical edges | **Adopted** as asserted invariants 1–2 (§1.3). Edge blocks not built (§6, risk 3) |
| F8 | Inserted code clobbering flags by convention | **Adopted.** `Spiller::def<C: Spill>`; `rematerializable` excludes flag writers |
| F9 | Read before definition is representable | **Adopted** as a runtime check in `push`; the claim moved to §2.13 |
| F10 | Flat `Bindings` | **Adopted.** Scoped frames (§2.9) |
| F11 | General carry budget undefined | **Adopted.** `members − GENERAL_CARRY_RESERVE (5)`, derivation stated (§2.11) |
| F12 | No stack probe | **Adopted.** C6 |
| F13 | EVEX `disp8` scaling | **Adopted.** Typed `Disp` kept, slots always `disp32`, pin at offset 64 (C2) |
| F14 | Pool entry type and reach per backend | **Adopted.** `type Constant`, `POOL_REACH`; `BUILTIN_HEADROOM` deleted |
| F15 | Slot reads inside selected instructions | **Adopted** as a byte change attributed in C1/C4. No fold hook (§4.3) |
| F16 | No test that every encoder handles every register | **Adopted.** B10's exhaustive encoder test |
| F17 | A too-low `max_regs` panics with the wrong text | **Moot.** The narrowing removed the cap. The test-only `AtFloor` panic names the file and the instruction |
| F18 | ABI convention by convention; misaligned slots | Convention **documented** on `RegisterFile`, with the build failure that enforces it. The alignment bias is **not done**, with its reason (§6) |
| F19 | Stale citations | **Adopted.** Re-verified at `bdee3900`; symbol plus line |

### Closure review

| # | Point | Disposition |
|---|---|---|
| 1 | Placeholder defs look like real constants | **Adopted.** A5 types them (`Outer`); D2 deletes them |
| 2 | `Bindings` must be scoped | **Adopted** (§2.9) |
| 3 | The latch needs no order assertion | **Adopted.** The driver emits the latch after the body; no assertion, no hazard |
| 4 | Layout flattens a tree that is rebuilt | **Adopted** in D3, after the last flat consumer is gone |
| 5 | Allocator result tables missing from the deletion list | **Adopted** (§4.1, D1) |
| 6 | `Origin` cannot keep the transient/kept split | **Adopted.** A6 merges the two into `loads`, with a schema bump |
| 7 | A shared `Address` undoes #1332; `Imm`/`Mem` operands | **Adopted.** Addresses per backend; the view has no `Imm`/`Mem` variants, with the reason given |
| 8 | Merging pools drops the pointer guarantee | **Adopted.** `Pointer`/`Integer` classes over `GeneralFile` |
| 9 | `Constraint` duplicates data and allows nonsense | **Adopted.** `Access` variants; class from the value; ABI on `Function::entry` |
| 10 | `LabelRef` already is a label field | **Adopted.** `Patch` function; assembler not generic |
| 11 | x86 needs no pool-base value | **Adopted**, flagged to JP (Metadata, departure 3) |
| 12 | Thread-local pool adds checks | **Adopted.** Per-compile mint, flagged to JP (departure 2) |
| 13 | Fall-through blocks unlabelled; branch arity; argument count | **Adopted.** Minted labels for every block, two targets, `Test`/`Edges` structs |
| 14 | How a fold's result reaches the parent | **Adopted.** `acc'` dominates the single-predecessor exit (§2.10) |
| 15 | `Function` lacks the loop nest | **Adopted.** `Function::loops`, `Block::scope`; candidates are live-in at the head |
| 16 | General budget ambiguous | **Adopted** (F11) |
| 17 | The ownership chain does not compile; carried leases across nested scans | **Adopted.** Leases, single layout-order scan, §2.13 row on shared borrows |
| 18 | Uniform loads past `imm12` | **Adopted** (H6, C4) |
| 19 | General-first conflicts with fixed offsets | **Adopted** (F3) |
| 20 | No floor without `MIN_SCRATCH` | **Moot** (F17) |
| 21 | Parallel-copy cycles; overstated `Def` claim; encoder unit tests; implementations in `mod.rs` | **Adopted.** Cycles broken through a fresh value; claim moved to §2.13; numeric encoding helpers keep their pins and B10 adds the exhaustive test; implementations live in leaf modules |
| 22 | S11 before hatches forces throwaway code | **Adopted.** Tokens exist only in the new pipeline |
| 23 | Not verified against main or the narrowing | **Adopted** (Metadata) |

### Landing review

| # | Point | Disposition |
|---|---|---|
| A1 | Stale lines | **Adopted** |
| A2 | `Label` contradicts the ontology at HEAD | **Adopted.** Minted labels; standalone assembler; departure 1 flagged |
| A3 | The narrowing deletes `EmitCtx` and others | **Adopted.** No `Registers` cap; already-deleted items not re-listed |
| A4 | Visibility widens | **Adopted.** `pub(in crate::emit)` throughout |
| B1 | Rust privacy defeats private constructors | **Adopted.** Leaf modules |
| B2 | Lease undeclared | **Adopted** |
| B3 | Frame does not borrow-check; termination | **Adopted.** Affine slot leases, frozen frame, no iteration |
| B4 | A `Def` can be read by its own instruction; thread-local machinery | **Adopted** (§2.13; per-compile mint) |
| B5 | CFG underdetermined | **Adopted.** Two targets |
| B6 | Parallel copy cycles | **Adopted** |
| B7 | `Bindings` must be scoped | **Adopted** |
| C S1 | Redo labels; `Past` in `PendingBranch` | **Adopted** (A2) |
| C S2 | Reachability; vacuous test | **Adopted** (A3) |
| C S3 | Delete two more tests | **Adopted** (A4) |
| C S4/S6 | Too big; replace existing enums; keep pins | **Adopted.** A9–A12 split a/b |
| C S7/S9/S10 | Not byte-identical (Write residency, pool order, bridge rules) | **Adopted by replacing the bridge.** The new pipeline is built beside the old; bytes move only at the switch commits |
| C S8 | Fine | **Superseded** (C2) |
| C S11 | Misordered | **Adopted** (closure 22) |
| C S12 | Does not land | **Superseded.** Legacy is untouched until D1 |
| C S13 | Too big; missing dependency; schema | **Adopted.** Split into B4–B7; binders are values from B3; schema handled in A6 |
| C S14a | Too big; name collision | **Adopted.** Split into B3/B6; `branch` is a new name; legacy renamed `LegacyBackend` |
| C S14b | Nonexistent test | **Adopted.** Real tests named (B9) |
| C S15 | Wrong gate; deep frames untested | **Adopted.** C4 tests, A1 `deep_frame` row, M |
| D1 | Goldens cannot check most claims | **Adopted.** A1 coverage rows; hash updated per re-baseline; harness generic in B10 |
| D2 | aarch64 not in G | **Adopted.** M gate |
| D3 | Compile-time gate not a check | **Adopted.** `selection_stays_linear`; operands computed once |
| E | Proposed split | **Adopted in substance.** S0 → A1; `Write` operands come for free in the new pipeline; S13 and S14a split; S15's three parts land in B3, B8 and C4 |