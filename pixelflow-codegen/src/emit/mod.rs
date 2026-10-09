//! JIT code emission for expression DAGs.
//!
//! ## Register allocation
//!
//! One allocator — `regalloc::LinearScan`, linear scan with Belady eviction
//! and constant rematerialization — parameterised by one description of the
//! target, `regalloc::RegisterFile`. Expressions arrive from e-graph
//! extraction with shared subexpressions, which is why the allocator works on
//! a DAG schedule rather than a tree.
//!
//! All three backends run that same allocator behind the same driver
//! (`IsaBackend`). What a backend contributes is its `RegisterFile` — the
//! allocatable pool, how many registers its encodings and its guards
//! destroy, vector width, the ABI's three pointer registers — and its
//! instruction encodings. Nothing else about a target reaches the
//! allocation, framing, or control-flow logic.
//!
//! ## The loop nest
//!
//! There is no collapse scaffold. A kernel reaches this module already
//! wrapped in the lattice's folds — rows, columns, lanes — by
//! [`pixelflow_ir::passes::legalize`], so the emitted function is one scope
//! (what runs once per call) with folds nested in it, every one emitted by
//! the same `Reduce` arm: seed, test, body, combine, step. The lane fold is
//! the one the `Write` inside names as its lane, and it is executed *by
//! lanes*: its binder is the constant `[0, 1, …, L−1]`, it has no counter
//! and no back edge, and its body is inlined into the column fold's
//! (docs/plans/2026-09-16-collapse-is-a-fold.md §2.2).
//!
//! ## Spilling
//!
//! Values the scratch pool cannot hold go to stack slots. The allocator lays
//! the whole frame out beside its placements, at the backend's vector stride
//! (`regalloc::NestAllocation`), and the emitter reads every address from
//! it (`regalloc::Allocation::slot_of`) and computes none:
//! - A value with a slot is stored to it right after its **definition**, which
//!   every path that reads the value has run — including through an `If`
//!   guard, which can only skip a definition by skipping every read of it.
//! - Reloaded into a register the allocator reserved *for that instruction*
//!   (`regalloc::Scratch`); there is no register outside the pool for this,
//!   and every definition holds a pool register at its own definition.

/// The one way a backend refuses an op.
///
/// Reaching this is never "the target cannot do that". [`pixelflow_ir::passes::legalize`]
/// leaves only ops from the backend-legal set, and every backend owes an
/// encoding for all of them — so arriving here means the pipeline was bypassed
/// or this backend is incomplete. Both are bugs in the compiler rather than
/// facts about the kernel, and neither is something a caller could act on: the
/// only callers that ever saw the old `Err` immediately `.expect()`ed it.
///
/// So it panics, loudly and at the point of failure, naming the op and the
/// backend that owes it. Development gets a stack trace pointing at the missing
/// match arm instead of a `&'static str` surfacing three frames up.
#[cold]
#[inline(never)]
fn unimplemented_op(backend: &str, op: pixelflow_ir::kind::OpKind) -> ! {
    panic!(
        "{backend} has no encoding for {op:?} — `passes::legalize` leaves only \
         backend-legal ops, so this is a missing implementation or a bypassed \
         pipeline, not a bad kernel"
    )
}

mod aarch64;
mod asm;
mod avx2;
mod avx512;
#[cfg(test)]
mod coverage;
mod encoded;
mod executable;
mod regalloc;
mod storage;
mod traffic;
mod x86_64;

// The whole public surface: the driver (it lives in pipeline.rs, and keeps
// this path), what it hands back, and the counts inside that.
pub use crate::pipeline::compile;
pub use executable::CompiledKernel;
pub use traffic::{EmitTraffic, ScopeTraffic};

use asm::{Item, Label, Labels, Patch};
use encoded::EncodedInst;
use storage::{Slot, StackFrame};

use pixelflow_ir::kind::OpKind;

use crate::program::IfArm;
use crate::program::IfGuard;
use crate::program::ScheduledOp;
use traffic::Counting;

use alloc::vec::Vec;

use crate::error::CompileError;
use crate::isa::Isa;
use pixelflow_ir::fold::{Binder, Monoid};

/// The one contract every backend's instruction types satisfy.
trait AsmInsn: Copy {
    /// Emit the instruction's encoded bytes into the output buffer.
    fn emit_into(self, code: &mut Vec<u8>);

    /// The position this instruction's bytes depend on, if any.
    ///
    /// Almost every instruction is position-independent and takes the default.
    /// A branch is not: it emits a placeholder displacement in `emit_into` and
    /// says here which [`Label`] it is waiting on, where the placeholder is
    /// and how to fill it in. That is the whole of what a branch adds — it is
    /// an ordinary instruction that takes a name instead of a number.
    #[inline]
    fn label_ref(self) -> Option<LabelRef> {
        None
    }
}

/// A declarative sequence of assembly instructions.
///
/// Written as an array or collection of instructions, then assembled into machine code:
/// ```ignore
/// AsmProgram::from([
///     Inst::Add { dst: RAX, src: RCX },
/// ]).assemble(&mut buff);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct AsmProgram<S> {
    insts: S,
}

impl<S> AsmProgram<S> {
    /// Create a new assembly program wrapping an instruction sequence.
    #[inline(always)]
    const fn new(insts: S) -> Self {
        Self { insts }
    }
}

impl<I: AsmInsn, const N: usize> From<[I; N]> for AsmProgram<[I; N]> {
    #[inline(always)]
    fn from(insts: [I; N]) -> Self {
        Self { insts }
    }
}

impl<I: AsmInsn, S: IntoIterator<Item = I>> AsmProgram<S> {
    /// Assemble the program into the machine-code buffer.
    ///
    /// # Panics
    ///
    /// If an instruction names a label. A name needs a program that binds it:
    /// that is an [`Assembly`], not a sequence. Only this crate writes these
    /// programs, so that is a bug here rather than a fact about the kernel
    /// being compiled.
    #[inline]
    fn assemble(self, code: &mut Vec<u8>) {
        for inst in self.insts {
            assert!(
                inst.label_ref().is_none(),
                "a label field needs an Assembly to bind its label"
            );
            inst.emit_into(code);
        }
    }
}

/// Free-function fold: assemble a declarative sequence directly into `code`.
#[inline]
fn assemble<I: AsmInsn>(code: &mut Vec<u8>, insts: impl IntoIterator<Item = I>) {
    AsmProgram::new(insts).assemble(code);
}

// =============================================================================
// Labels: a name for a position, bound at assembly time
// =============================================================================

/// How an instruction whose bytes depend on a position gets those bytes.
///
/// Returned by [`AsmInsn::label_ref`]. The instruction emits a placeholder in
/// `emit_into`; `patch` fills the displacement in once the label's position is
/// known.
#[derive(Copy, Clone)]
struct LabelRef {
    /// Bytes from the instruction's start to the field.
    at: usize,
    /// The position this instruction is waiting on.
    label: Label,
    /// See [`Patch`].
    patch: Patch,
}

/// An instruction with a label field, as [`Assembly`] hands it to the
/// assembler: the bytes its `emit_into` wrote, and the field in them.
struct Fielded {
    bytes: Vec<u8>,
    field: LabelRef,
}

/// A program being written: the front end of [`asm::assemble`].
///
/// An emitter that walks a schedule cannot hand over a finished list of
/// instructions and labels — it discovers them as it goes, calling `&mut
/// self` backend verbs for each — so it appends to one of these instead. The
/// bytes of a position-independent instruction go into `code`, the run being
/// written; a label ends the run, and so does an instruction with a field in
/// it, which becomes an item of its own. One kernel is one of these, however
/// many scopes it has.
#[derive(Default)]
struct Assembly {
    /// The run being written: what a backend verb emits into.
    code: Vec<u8>,
    text: Vec<Item<Fielded>>,
    /// Bytes of `text` already ended, which `code` is not part of yet.
    ended: usize,
    data: Vec<Item<Fielded>>,
    labels: Labels,
}

impl Assembly {
    /// A new label, named by the position it will be bound to.
    fn mint(&mut self) -> Label {
        self.labels.mint()
    }

    /// Bytes of code so far: what [`Assembly::bind`] would bind a label to.
    fn len(&self) -> usize {
        self.ended + self.code.len()
    }

    /// End the run being written.
    fn end_run(&mut self) {
        if self.code.is_empty() {
            return;
        }
        self.ended += self.code.len();
        self.text.push(Item::Bytes(self.code.clone()));
        self.code.clear();
    }

    /// Write a label here — the name of this position.
    ///
    /// A branch may name a position before it exists, which is every forward
    /// branch and the exit of every loop, so nothing here checks that anything
    /// refers to it. [`Assembly::finish`] is where a name nobody wrote is
    /// reported.
    fn bind(&mut self, label: Label) {
        self.end_run();
        self.text.push(Item::Bind(label));
    }

    /// Emit one instruction, recording the name it waits on if it has one.
    fn push(&mut self, inst: impl AsmInsn) {
        let at = self.code.len();
        inst.emit_into(&mut self.code);
        let Some(field) = inst.label_ref() else {
            return;
        };
        let bytes = self.code.split_off(at);
        self.end_run();
        self.ended += bytes.len();
        self.text.push(Item::Inst(Fielded { bytes, field }));
    }

    /// The constant pool, trailing the code: the data section. It is aligned
    /// when it holds anything, and `pool` is bound where it starts either way
    /// — the anchor names it unconditionally.
    fn pool(&mut self, pool: Label, entries: Vec<u8>) {
        if !entries.is_empty() {
            self.data.push(Item::Align(CONST_POOL_ALIGN as u64));
        }
        self.data.push(Item::Bind(pool));
        self.data.push(Item::Bytes(entries));
    }

    /// Assemble the kernel.
    ///
    /// # Panics
    ///
    /// If a branch names a label nothing bound, or a label is bound twice.
    /// Only this crate writes these programs, so either is a bug here rather
    /// than a fact about the kernel being compiled.
    #[must_use]
    fn finish(mut self) -> Vec<u8> {
        self.end_run();
        asm::assemble(
            &asm::AsmProgram {
                text: self.text,
                data: self.data,
                labels: self.labels,
            },
            |inst, out| {
                out.bytes(&inst.bytes);
                out.field(inst.field.at, inst.field.label, inst.field.patch);
            },
        )
    }
}

/// The constant pool's alignment: one NEON pool entry, so every `LDR Qt` from
/// it is an aligned vector load. x86's four-byte entries need no alignment and
/// take this one for the cache line. The padding that reaches it from the
/// last instruction follows the code's length, which is why a kernel's
/// trailing bytes can differ between two allocations of it by less than this.
const CONST_POOL_ALIGN: usize = 16;

/// Physical vector register index (v0..v31 on AArch64, xmm/ymm/zmm0..zmm31 on x86).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Reg(u8);

/// Physical 64-bit general-purpose register index (x0..x31 on AArch64, rax..r15 on x86).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Gpr(u8);

/// Physical pointer register index holding a memory address (x0..x31/sp on AArch64, rax..r15/rsp on x86).
///
/// Distinct from [`Gpr`] (integers, counters, indices) and [`Reg`] (SIMD vectors).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct PtrReg(u8);

impl PtrReg {
    /// View as general-purpose register for instructions that manipulate pointers as raw 64-bit values.
    #[inline(always)]
    #[must_use]
    const fn as_gpr(self) -> Gpr {
        Gpr(self.0)
    }
}

impl From<PtrReg> for Gpr {
    #[inline(always)]
    fn from(p: PtrReg) -> Self {
        p.as_gpr()
    }
}

/// Physical mask/predicate register index (k0..k7 on AVX-512).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct KReg(u8);

/// A physical location where a value resides: in a register or on the stack.
///
/// Every variant of `Loc` is a writable, addressable storage location.
/// A rematerialized constant has no location; it is a [`Binding`], not a `Loc`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Loc {
    /// Value is in a vector register.
    Reg(Reg),
    /// Value is an address, in a pointer register — a
    /// [`Class::Pointer`](crate::program::Class::Pointer) value's only kind of register.
    Ptr(PtrReg),
    /// Value is spilled to a stack slot.
    Slot(Slot),
}

impl Loc {
    /// Get the vector register, panicking if the value is not in one.
    #[must_use]
    fn reg(self) -> Reg {
        match self {
            Loc::Reg(r) => r,
            Loc::Ptr(p) => panic!("expected a vector register, got pointer register {p:?}"),
            Loc::Slot(s) => panic!("expected register, got stack slot {}", s.offset()),
        }
    }
}

/// The binding of a value after register allocation: a physical location
/// or a constant that is rematerialized at every use.
///
/// `Binding` is the register allocator's full answer — "where did this value
/// end up?" — and includes [`Remat`](Binding::Remat) for constants that live
/// nowhere. For a writable physical location, use [`Loc`] instead.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Binding {
    /// Value lives in a physical location (register or stack slot).
    Loc(Loc),
    /// Value is a constant (these are its `f32` bits): it lives nowhere and is
    /// re-emitted at each use.
    Remat(u32),
}

impl Binding {
    /// Get the register, panicking if the value is not in one.
    #[must_use]
    fn reg(self) -> Reg {
        match self {
            Binding::Loc(loc) => loc.reg(),
            Binding::Remat(bits) => panic!("expected register, got rematerialized {bits:#x}"),
        }
    }
}

impl From<Reg> for Binding {
    #[inline]
    fn from(r: Reg) -> Self {
        Binding::Loc(Loc::Reg(r))
    }
}

impl From<Slot> for Binding {
    #[inline]
    fn from(s: Slot) -> Self {
        Binding::Loc(Loc::Slot(s))
    }
}

/// One unary instruction as a backend's `emit_unary` takes it: the op, its
/// two registers, and the allocator's temp for the instruction, which the ops
/// that build a mask or a correction term write and the rest ignore.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Unary {
    op: OpKind,
    dst: Reg,
    src: Reg,
    temp: Option<Reg>,
}

/// A concrete instruction to emit, with all registers resolved.
/// Pure data — no side effects, no mutation.
#[derive(Clone, Debug, PartialEq, Eq)]
enum ResolvedOp {
    /// No-op (variable already in input register).
    Nop,
    /// Load constant into dst.
    LoadConst { dst: Reg, val_bits: u32 },
    /// Unary: dst = op(src).
    Unary { op: OpKind, dst: Reg, src: Reg },
    /// Integer shift by a compile-time immediate: dst = src `op` amount, where
    /// `op` is `Shl` or `Shr` (the hardware shift encoders are imm-only).
    ShiftImm {
        op: OpKind,
        dst: Reg,
        src: Reg,
        amount: u8,
    },
    /// Binary: dst = op(left, right).
    Binary {
        op: OpKind,
        dst: Reg,
        left: Reg,
        right: Reg,
    },
    /// Fused multiply-add via FMLA: dst = c + a*b.
    /// Requires dst to hold c before FMLA.
    FusedMulAdd { dst: Reg, a: Reg, b: Reg },
    /// BSL select: dst = mask ? if_true : if_false (mask pre-loaded into dst).
    If {
        dst: Reg,
        if_true: Reg,
        if_false: Reg,
    },
    /// Bound-memory gather: `dst = base[idx_lane]`. Every backend implements
    /// it: AVX2 and AVX-512 natively (`vgatherdps`), NEON as four scalar
    /// loads. `base` is the buffer's base pointer,
    /// wherever the allocator keeps that value — a [`PtrReg`] by type, so
    /// nothing but an address can be handed to the memory operand.
    Gather { dst: Reg, idx: Reg, base: PtrReg },
    /// Lane-uniform gather: `dst = splat(base[idx_lane0])`. The one index
    /// every lane holds is truncated out of lane 0 into a GPR (`cvttss2si`,
    /// `fcvtzs`) and the element is read once and broadcast: `vbroadcastss
    /// [base + idx*4]` on every x86 tier, `ldr s` + `dup` on NEON. No
    /// per-lane extract or insert, on any backend.
    Broadcast { dst: Reg, idx: Reg, base: PtrReg },
    /// Uniform broadcast: `dst = splat(base[offset])`, the scalar at
    /// `4 * offset` of the block `base` addresses, broadcast to every lane:
    /// `vbroadcastss` on every x86 tier, `ldr s` + `dup` on NEON. The
    /// offset is the slot at its full control-plane width; each encoder
    /// narrows it to the displacement its instruction has, and refuses one
    /// that does not fit.
    Uniform { dst: Reg, base: PtrReg, offset: u64 },
    /// A context pointer: `dst = ctx[slot]`, one `mov`/`ldr` from the
    /// context array the kernel is called with. The definition of every
    /// [`Class::Pointer`](crate::program::Class::Pointer) value, and the only instruction that
    /// reads `regalloc::RegisterFile::gpr_ctx`.
    Context { dst: PtrReg, slot: u16 },
    /// The lane fold's binder, materialized: `dst = [0, 1, …, L−1]` as
    /// `f32`s, `L` being the backend's lane count. The one vector constant
    /// that is not a broadcast, and the whole of what "executed by lanes"
    /// costs the body.
    Lanes { dst: Reg },
}

/// Reload instruction: load a value into a register.
///
/// Either reload from stack (spilled) or rematerialize a constant.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Reload {
    /// Load from stack slot.
    FromStack { target: Reg, slot: Slot },
    /// Rematerialize a constant (emit FMOV immediate).
    Const { target: Reg, val_bits: u32 },
    /// Load an address from its stack slot into the pointer register the
    /// allocator reserved for this instruction's base
    /// (`regalloc::Scratch::ptr_reload`).
    Ptr { target: PtrReg, slot: Slot },
}

/// Fully resolved instruction: what to reload, and what to compute.
///
/// No store. A destination is always a register now, so the one place a value
/// reaches its slot is the emit loop's store-after-definition — which is what
/// makes the slot valid on every path an `If` guard can take.
#[derive(Clone, Debug)]
struct InstructionPlan {
    /// Reloads to emit before the main op.
    reloads: Vec<Reload>,
    /// The main operation.
    op: ResolvedOp,
    /// Optional MOV to set up accumulator/mask before main op.
    setup_mov: Option<(Reg, Reg)>,
    /// The registers the encoding may destroy for the length of this
    /// instruction.
    ///
    /// Filled exactly as far as the backend asked
    /// ([`regalloc::RegisterFile::temps_for`]); the allocator picked them, so
    /// each holds no live value and is nobody's operand, and all are free again
    /// at the next instruction. An encoding that needs scratch must read this
    /// rather than a `const`, because there is no register reserved for it.
    scratch: regalloc::Scratch,
}

/// Where one operand of an instruction is read from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum OperandSource {
    /// Already in a register; the location table says which.
    Resident,
    /// Not in a register, and reloaded into the **destination**.
    ///
    /// Free because the destination is a register no encoding writes before
    /// its last read, so one operand can always come from it: an `If`'s
    /// mask and an FMA's addend, which the blend and the `231` form consume
    /// from `dst` anyway, and a binary's left, which costs a reservation
    /// otherwise. Sound because the reload lands before the op and nothing
    /// else the instruction reads is resident in `dst` — the allocator's
    /// destination contest never leaves another *resident* operand in the
    /// register it hands out (a displaced one is non-resident at this index
    /// and reloaded elsewhere). That is the whole guarantee: the encoders do
    /// **not** read every source before writing `dst` (`setup_mov` ahead of
    /// an `If` or FMA on every ISA), so this is the one register-level alias
    /// any of them tolerates.
    Destination,
    /// Not in a register, and reloaded into the `k`'th register the allocator
    /// reserved for this instruction (`regalloc::Scratch::reload`).
    Reload(usize),
}

/// Where each operand of `op` is read from, given which of them are in a
/// register at this point.
///
/// **One statement, read twice.** The allocator counts the [`Reload`]s to
/// reserve; the emitter names the register each one lands in. A second copy
/// would be a convention between two files that has to agree
/// register-for-register — the shape this change exists to remove — so it is
/// one function, and residency is final by the time either calls it (eviction
/// splits a live range rather than rewriting one, so a value in a register
/// when its reader is allocated is in a register when its reader is emitted).
///
/// `resident[k]` for an operand this op does not have is ignored.
///
/// [`Reload`]: OperandSource::Reload
#[must_use]
fn operand_sources(op: &ScheduledOp, resident: [bool; 3]) -> [OperandSource; 3] {
    // The operand an encoding wants in the destination, if any.
    let into_dst = match op {
        ScheduledOp::Binary(..) => Some(0),
        ScheduledOp::Ternary(OpKind::MulAdd, ..) => Some(2),
        ScheduledOp::Ternary(OpKind::If, ..) => Some(0),
        _ => None,
    };
    let arity = match op {
        ScheduledOp::Var(_)
        | ScheduledOp::Lanes(_)
        | ScheduledOp::Const(_)
        | ScheduledOp::Context(_)
        // A uniform load's block is its pointer operand, resolved by
        // `resolve_operands` from the pointer class, never a vector reload.
        | ScheduledOp::Uniform(..)
        | ScheduledOp::Reduce(..)
        | ScheduledOp::Outer(_)
        | ScheduledOp::Seq(..) => 0,
        ScheduledOp::Unary(..)
        | ScheduledOp::ShiftImm(..)
        | ScheduledOp::Gather(..)
        | ScheduledOp::Broadcast(..)
        | ScheduledOp::Write { .. } => 1,
        ScheduledOp::Binary(..) => 2,
        ScheduledOp::Ternary(..) => 3,
    };

    let mut sources = [OperandSource::Resident; 3];
    let mut next = 0;
    for (k, source) in sources.iter_mut().enumerate().take(arity) {
        if resident[k] {
            continue;
        }
        if into_dst == Some(k) {
            *source = OperandSource::Destination;
            continue;
        }
        *source = OperandSource::Reload(next);
        next += 1;
    }
    sources
}

/// How many reload registers [`operand_sources`] asked this instruction to
/// reserve.
#[must_use]
fn reloads_wanted(sources: [OperandSource; 3]) -> usize {
    sources
        .iter()
        .filter(|s| matches!(s, OperandSource::Reload(_)))
        .count()
}

/// The temp an encoding declared in [`regalloc::RegisterFile::temps_for`].
///
/// A backend's `temps_for` and its encodings are two halves of one statement
/// about each instruction, and nothing in the types holds them together — this
/// is where they are checked against each other.
///
/// # Panics
/// If the encoding wants a temp its `temps_for` did not ask for.
#[track_caller]
fn declared_temp(temp: Option<Reg>) -> Reg {
    temp.expect("this encoding needs a temp that `RegisterFile::temps_for` did not ask for")
}

/// The GPR-class mirror of [`declared_temp`], for
/// [`regalloc::RegisterFile::gpr_temps_for`].
#[track_caller]
fn declared_gpr_temp(temp: Option<Gpr>) -> Gpr {
    temp.expect("this encoding needs a GPR that `RegisterFile::gpr_temps_for` did not ask for")
}

/// The mask-class mirror of [`declared_temp`], for
/// [`regalloc::RegisterFile::mask_temps_for`].
#[track_caller]
fn declared_mask_temp(temp: Option<KReg>) -> KReg {
    temp.expect(
        "this encoding needs a mask register that `RegisterFile::mask_temps_for` did not ask for",
    )
}

/// What [`compile`] hands back: the code, and what was emitted, counted —
/// the static half of what a measurement harness prices a kernel by.
pub struct CompileResult {
    /// The executable code.
    pub code: CompiledKernel,
    /// Number of spills performed.
    pub spill_count: u64,
    /// Total stack space used for spills (bytes). A frame offset, not a
    /// program count: `StackFrame` refuses a frame past 2 MiB and the
    /// encoders address a slot through a disp32 / imm12, so the width is the
    /// encoding's.
    pub spill_bytes: u32,
    /// Values one scope computes for the scopes inside it and parks in a
    /// slot of their own — the loop-invariant code motion, counted.
    pub hoisted_values: u64,
    /// What was emitted, per scope of the nest — the static half of a cost
    /// model's inputs. Counted, never optimized: see [`EmitTraffic`].
    pub traffic: EmitTraffic,
}

/// The architecture seam for the shared driver.
///
/// [`compile_via_backend`] owns the architecture-INDEPENDENT logic —
/// register allocation, frame layout, the fold loops and the If
/// short-circuit control flow — and calls an `IsaBackend` for the leaf
/// operations that actually differ between x86-64 and aarch64 (instruction
/// encoding, branch encoding, and any arch-specific finalization such as
/// aarch64's constant pool). Both backends therefore run the *same* driver:
/// there is one place that decides when to emit a guard branch, how a loop
/// is seeded and stepped, where a store's address comes from.
///
/// Control flow crosses this seam as *"branch to this [`Label`]"*. It used to
/// cross as an opaque per-backend fixup token that the driver placed with
/// `emit_jump` and later handed back to `patch_branch` along with an offset it
/// had tracked itself — which is a label, minus the name.
trait IsaBackend {
    /// Jump to `label`, unconditionally.
    fn jump(&mut self, asm: &mut Assembly, label: Label);

    /// This backend's register file: the whole of what allocation and frame
    /// layout need to know about the target.
    ///
    /// Backends declare it as a `const` next to their encodings. It is the
    /// only target-dependent input to any of the shared logic here.
    fn register_file(&self) -> regalloc::RegisterFile;

    /// Per-compile setup before any code is emitted (e.g. seed a constant pool).
    fn begin(&mut self, schedule: &[regalloc::Def]) -> Result<(), CompileError>;

    /// Emit one resolved instruction (with its reloads/store).
    fn emit_plan(&mut self, code: &mut Vec<u8>, plan: &InstructionPlan)
    -> Result<(), CompileError>;

    /// Register-to-register move.
    fn emit_mov(&mut self, code: &mut Vec<u8>, dst: Reg, src: Reg);

    /// Spill a register to a frame slot.
    fn emit_store(&mut self, code: &mut Vec<u8>, src: Reg, offset: u32)
    -> Result<(), CompileError>;

    // -------------------------------------------------------------------------
    // The pointer class: an address moves between its register and its slot
    // through these, never through the vector forms above — a pointer is
    // eight bytes in a general register, and the vector encoders would read
    // or write the wrong file.
    // -------------------------------------------------------------------------

    /// Store an address to a frame slot.
    fn ptr_store(&mut self, code: &mut Vec<u8>, src: PtrReg, offset: u32);
    /// Load an address from a frame slot.
    fn ptr_load(&mut self, code: &mut Vec<u8>, dst: PtrReg, offset: u32);
    /// Copy an address between pointer registers.
    fn ptr_mov(&mut self, code: &mut Vec<u8>, dst: PtrReg, src: PtrReg);

    /// Resolve a value to a register, reloading or rematerializing into
    /// `target` if it is not already in one.
    ///
    /// # Errors
    ///
    /// [`CompileError::BudgetExceeded`] when a rematerialized constant has no
    /// addressable place in the backend's constant pool.
    fn emit_resolve(
        &mut self,
        code: &mut Vec<u8>,
        vid: regalloc::ValueId,
        target: Reg,
        locs: &[Option<Binding>],
    ) -> Result<Reg, CompileError>;

    /// Jump to `label` when **no lane selects `test.arm`**, so the arm can be
    /// skipped.
    ///
    /// One verb rather than a `skip_if_all_false`/`skip_if_all_true` pair: the
    /// two differ only in which uniform mask lets an arm go, which is what
    /// [`IfArm`] already names.
    ///
    /// `scratch` is a vector register the backend may destroy, present exactly
    /// when its [`RegisterFile::guard_temps`](regalloc::RegisterFile::guard_temps)
    /// asked for one. Only aarch64 does — reducing a mask with `UMAXV`/`UMINV`
    /// writes a scalar into a vector register before it can reach a GP
    /// register — so the x86 tiers, whose guards go through
    /// `movmskps`/`kortest` and the flags, receive `None` and want nothing.
    fn branch_if_arm_is_dead(&mut self, asm: &mut Assembly, test: MaskTest, label: Label);

    // -------------------------------------------------------------------------
    // The function around the nest: its frame and what trails it.
    // -------------------------------------------------------------------------

    /// Reserve / release `bytes` of stack.
    fn frame_alloc(&mut self, code: &mut Vec<u8>, bytes: u32);
    fn frame_free(&mut self, code: &mut Vec<u8>, bytes: u32);

    /// Anchor whatever the body's constant loads are relative to, once the
    /// frame exists: the register that holds the constant pool's address for
    /// the rest of the function.
    ///
    /// Takes the whole [`Assembly`], not just its `code`, because the anchor
    /// names `pool` — the constant pool's not-yet-known position — rather
    /// than a `code.len()` read off and carried by hand. One function has one
    /// pool, so the driver mints its label once and hands it to both this and
    /// [`IsaBackend::finish`].
    fn anchor(&mut self, asm: &mut Assembly, pool: Label);

    /// Append whatever must trail the emitted function — the constant pool —
    /// and bind `pool` where it lands.
    fn finish(&mut self, asm: &mut Assembly, pool: Label);

    /// Save / restore a value in a slot outside any scope's own spill slots:
    /// a fold's binder or accumulator, a root parked for the scopes inside.
    fn slot_store(&mut self, code: &mut Vec<u8>, src: Reg, offset: u32);
    fn slot_load(&mut self, code: &mut Vec<u8>, dst: Reg, offset: u32);

    /// Bracket one scope's emission, for a decorator that attributes what is
    /// emitted to the scope it runs in. Defaults do nothing.
    fn scope_begin(&mut self) {}
    fn scope_end(&mut self, _scope: regalloc::Scope, _bytes: u64) {}

    // -------------------------------------------------------------------------
    // A surviving `Reduce`'s own loop: the seed, the trip test, the
    // accumulate and the step.
    //
    // None goes through the schedule/`InstructionPlan` machinery — a fold's
    // roots live where `allocate_nest` put them, a carried register or a
    // slot outside any scope's frame (see
    // docs/plans/2026-09-10-a-surviving-reduce-is-a-loop.md), and the trip
    // test compares the binder against a compile-time bound, not another
    // scheduled value. All are ordinary two- and three-register ALU ops, so
    // they are spelled as one verb each rather than a new opcode.
    // -------------------------------------------------------------------------

    /// `dst += scalar` across every lane, clobbering `scratch`.
    ///
    /// # Errors
    ///
    /// [`CompileError::BudgetExceeded`] when the scalar has no addressable
    /// place in the backend's constant pool.
    fn add_scalar(
        &mut self,
        code: &mut Vec<u8>,
        dst: Reg,
        scratch: Reg,
        scalar: f32,
    ) -> Result<(), CompileError>;

    /// Load an `f32` constant, broadcast across every lane.
    ///
    /// # Errors
    ///
    /// [`CompileError::BudgetExceeded`] when the constant has no addressable
    /// place in the backend's constant pool.
    fn load_const(&mut self, code: &mut Vec<u8>, dst: Reg, val: f32) -> Result<(), CompileError>;

    /// `dst = op(srcs[0], srcs[1])`, an ordinary vector ALU op outside the
    /// schedule: the fold loop's accumulate (`op` is the fold's monoid).
    fn alu(&mut self, code: &mut Vec<u8>, op: OpKind, dst: Reg, srcs: [Reg; 2]);

    /// `dst = (srcs[0] >= srcs[1]) ? all-ones : 0` — the fold loop's trip
    /// test.
    ///
    /// A default over [`IsaBackend::alu`]: every backend but AVX-512 computes
    /// a comparison exactly like any other binary op. AVX-512 represents a
    /// comparison's result as a k-register before it is widened to an
    /// ordinary vector mask ([`RegisterFile::mask_guard_temps`](regalloc::RegisterFile::mask_guard_temps)), which
    /// `mask_scratch` supplies — the one other place besides an `If`
    /// guard that needs it — and every other backend ignores.
    fn test_ge(
        &mut self,
        code: &mut Vec<u8>,
        dst: Reg,
        srcs: [Reg; 2],
        mask_scratch: Option<KReg>,
    ) {
        let _ = mask_scratch;
        self.alu(code, OpKind::Ge, dst, srcs);
    }

    // -------------------------------------------------------------------------
    // The lattice's effect: the store.
    // -------------------------------------------------------------------------

    /// Store `write.lanes` lanes of `write.value` at
    /// `out + 4 · (row · pitch + col)`, `row` and `col` being the enclosing
    /// folds' binders wherever their loops keep them (a register, or a slot
    /// — a broadcast, so any lane is the index). `out` and `pitch` are the
    /// ABI's, in [`RegisterFile::gpr_out`](regalloc::RegisterFile::gpr_out)
    /// and [`gpr_pitch`](regalloc::RegisterFile::gpr_pitch). A full batch is
    /// one vector store; a row's remainder stores exactly its lanes —
    /// masked where the ISA has a masked store, one lane at a time where it
    /// does not.
    fn emit_write(&mut self, code: &mut Vec<u8>, write: &WritePlan);

    /// Function return.
    fn emit_ret(&mut self, code: &mut Vec<u8>);
}

/// One `Write`, resolved: the value's register, where the two address
/// binders are, how many lanes to store, and the scratch the allocator
/// reserved for the address arithmetic.
#[derive(Clone, Copy, Debug)]
struct WritePlan {
    /// The value to store, in a register.
    value: Reg,
    /// The row binder — a broadcast index — where its fold keeps it.
    row: Binding,
    /// The column binder, likewise.
    col: Binding,
    /// How many of `value`'s lanes to store, from lane 0: the lane fold's
    /// trip count, the full batch or a row's remainder.
    lanes: u32,
    /// This instruction's reservations: two GPRs for the address, and the
    /// vector or mask temp a backend's remainder store asked for.
    scratch: regalloc::Scratch,
}

/// A guard's question: is this arm dead for the whole batch?
///
/// One struct because the three travel together and mean nothing apart — the
/// mask register is what is reduced, the scratch is what the reduction may
/// destroy, and the arm says which uniform answer lets the arm go.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct MaskTest {
    /// The mask to reduce.
    reg: Reg,
    /// A vector register the reduction may destroy, present exactly when this
    /// backend's [`RegisterFile::guard_temps`](regalloc::RegisterFile::guard_temps)
    /// asked for one. Only aarch64 does — reducing a mask with `UMAXV`/`UMINV`
    /// writes a scalar into a vector register before it can reach a GP register
    /// — so the x86 tiers, whose guards go through `movmskps`/`kortest` and the
    /// flags, receive `None` and want nothing.
    scratch: Option<Reg>,
    /// The mask-class mirror of `scratch`, present exactly when this backend's
    /// [`RegisterFile::mask_guard_temps`](regalloc::RegisterFile::mask_guard_temps)
    /// asked for one. Only AVX-512 does — `vptestmd` writes its result into a
    /// `k`-register before `kortestw` can read it into the flags — so every
    /// other tier receives `None` and wants nothing.
    mask_scratch: Option<KReg>,
    /// Which arm is being skipped.
    arm: IfArm,
}

/// Where `v` lives when the allocator says `at`: the arrow from a
/// [`regalloc::Where`] to the [`Binding`] the emitter encodes. The allocator
/// says a value is in a slot, and its frame says which one
/// ([`regalloc::Allocation::slot_of`]) — total for every value with an
/// address, which is every value spilled anywhere in the scope.
///
/// # Panics
/// If `at` is `Spilled` and `v` has no slot in this scope.
fn binding(
    allocation: regalloc::Allocation<'_>,
    v: regalloc::ValueId,
    at: regalloc::Where,
) -> Binding {
    match at {
        regalloc::Where::Reg(r) => Binding::from(Reg(r.0)),
        regalloc::Where::Ptr(p) => Binding::Loc(Loc::Ptr(p)),
        regalloc::Where::Remat(bits) => Binding::Remat(bits),
        regalloc::Where::Spilled => {
            Binding::from(allocation.slot_of(v).unwrap_or_else(|| {
                panic!("{v:?} is spilled somewhere in this scope but has no slot")
            }))
        }
    }
}

/// Emit one scope from a finished allocation.
///
/// Every address is the allocation's: where each root of the nest is parked
/// ([`regalloc::Allocation::park`]) — the slot the scope computing it writes
/// after the def, and the scopes inside read it from unless the allocator
/// carried it into them in a register, which their placement says. Which
/// roots this scope *reads* (an ancestor computed them: its entries for them
/// are placeholders that emit nothing) and which it *computes* (its own
/// `roots`) are the allocation's answers too.
///
/// A surviving `Reduce`'s accumulator is the same idea, addressed by its own
/// `ValueId` ([`regalloc::Allocation::accumulator_slot`]), at a slot that
/// outlives both this scope's slots and the fold's own (see
/// `ScheduledOp::Reduce`'s arm below, and
/// docs/plans/2026-09-10-a-surviving-reduce-is-a-loop.md's "the design
/// decision that makes this tractable");
/// [`regalloc::Allocation::binder_slot`] likewise for its binder.
///
/// Appends the scope's code to `asm` — the one program the kernel is, which
/// every scope nested in this one appends to in turn — and returns the
/// register the scope's result is in: `None` when the root is an effect and
/// not a value: a `Write`, a `Seq`, a fold over the unit monoid.
fn emit_scope<B: IsaBackend>(
    allocation: regalloc::Allocation<'_>,
    backend: &mut B,
    asm: &mut Assembly,
) -> Result<Option<Reg>, CompileError> {
    let start = asm.len();
    backend.scope_begin();
    // Allocation happened before this call — once per scope, over the whole
    // nest, its frame included. The allocator chooses the evaluation order,
    // so everything here — guard ranges, program points, the emit loop
    // itself — reads the schedule it handed back rather than the one it was
    // given.
    let schedule = allocation.schedule();

    // The binders of this scope's own fold and every enclosing fold, each
    // where that loop keeps it — innermost first, so a binder shadowing an
    // enclosing one is the nearer loop's. A `Write` reads its row and column
    // from here, and a binder's `Var` (found here by the binder's number —
    // sibling folds binding the same slot share one `Var` node, so which
    // counter it names is this scope's question) is a placeholder whose def
    // emits nothing: the loop seeded it, and the allocator's table already
    // names the slot it reads it from when that loop did not carry it.
    let mut enclosing: alloc::vec::Vec<(Binder, Binding)> = alloc::vec::Vec::new();
    let mut binder_placeholders: alloc::vec::Vec<regalloc::ValueId> = alloc::vec::Vec::new();
    let mut opened = allocation;
    while let Some((parent, at)) = opened.opens_at() {
        let parent = opened.sibling(parent);
        let def = &parent.schedule()[at];
        let ScheduledOp::Reduce(fold, _) = &def.op else {
            unreachable!("a fold scope opens at its parent's Reduce def")
        };
        let binder = fold.binder();
        let at_binder = match opened.fold_roots().binder {
            regalloc::Where::Reg(r) => Binding::from(r),
            regalloc::Where::Ptr(_) => unreachable!("a fold's binder is a vector"),
            regalloc::Where::Spilled | regalloc::Where::Remat(_) => {
                Binding::from(Slot::new(allocation.binder_slot(def.value)))
            }
        };
        if !enclosing.iter().any(|(b, _)| *b == binder) {
            enclosing.push((binder, at_binder));
        }
        if let Some(bv) = schedule
            .iter()
            .find(|d| matches!(d.op, ScheduledOp::Var(v) if v == binder.var()))
            .map(|d| d.value)
            && !binder_placeholders.contains(&bv)
        {
            binder_placeholders.push(bv);
        }
        opened = parent;
    }
    let binder_at = |binder: Binder| -> Binding {
        enclosing
            .iter()
            .find(|(b, _)| *b == binder)
            .map(|(_, at)| *at)
            .unwrap_or_else(|| {
                panic!(
                    "a Write names binder slot {} that no enclosing fold binds",
                    binder.slot()
                )
            })
    };

    // If short-circuit guards, read off the allocation rather than
    // recomputed: `schedule` above is `allocation.schedule()` verbatim, and
    // the table is the one this scope was built with (`program::scopes::lay_out`),
    // which the allocator placed split ranges around each arm by (see
    // `regalloc::Allocation::if_guards`). A root this scope parks is never
    // inside an arm — the analysis was told it is read outside the schedule
    // — so a guard can never skip a park.
    let if_guards: &[IfGuard] = allocation.if_guards();
    let sched_len = schedule.len();

    // One side of a guard's branch to the point past an arm: minted where the
    // branch is made, carried to the instruction the arm ends at, where it is
    // bound.
    struct PendingBranch {
        guard_idx: usize,
        arm: IfArm,
        past: Label,
    }
    let mut branch_starts: alloc::vec::Vec<alloc::vec::Vec<PendingBranch>> =
        (0..sched_len).map(|_| alloc::vec::Vec::new()).collect();
    let mut branch_ends: alloc::vec::Vec<alloc::vec::Vec<PendingBranch>> =
        (0..sched_len).map(|_| alloc::vec::Vec::new()).collect();
    // Which guard, if any, belongs to the `If` at each schedule position: dense
    // by position, built once, so each `If` is a lookup rather than a search.
    let mut guard_at: alloc::vec::Vec<Option<usize>> = alloc::vec![None; sched_len];
    for (gi, guard) in if_guards.iter().enumerate() {
        assert!(
            guard_at[guard.if_idx].replace(gi).is_none(),
            "two guards claim the `If` at schedule position {}",
            guard.if_idx
        );
        for arm in IfArm::ALL {
            let range = guard.range(arm);
            if range.0 != range.1 {
                let past = asm.mint();
                branch_starts[range.0].push(PendingBranch {
                    guard_idx: gi,
                    arm,
                    past,
                });
                if range.1 < sched_len {
                    // The arm too, not just the guard: an end used to name the
                    // guard alone and recover the arm by trying both, which
                    // meant a guard whose arms end together was visited twice.
                    branch_ends[range.1].push(PendingBranch {
                        guard_idx: gi,
                        arm,
                        past,
                    });
                }
            }
        }
    }

    // One dense ValueId -> Binding lookup for the hot loop, carried *forward*: a
    // placement is a schedule, so the answer changes at program points, and
    // this is that schedule played out. Each range of each value's life
    // becomes one write here at the point it starts — O(total ranges), not a
    // lookup per operand per instruction.
    //
    // Seeded with where each value is when this scope first reaches it — at
    // its definition, for the values this scope computes. A value an
    // enclosing scope parked is left out: it is live-in, and the head
    // reconciliation below is what brings it to where this scope expects it.
    // A surviving fold's root in memory — an accumulator, or a binder the
    // allocator did not carry — reads as its dedicated fold slot here, which
    // is where the allocator's table puts it; a `Reduce` def's own emission
    // never goes through the ordinary operand/destination machinery this
    // table serves everyone else, and a binder's placeholder emits nothing.
    let len = schedule
        .iter()
        .map(|def| def.value.0 as usize + 1)
        .max()
        .unwrap_or(0);
    let mut locs: alloc::vec::Vec<Option<Binding>> = alloc::vec![None; len];
    for (i, def) in schedule.iter().enumerate() {
        let v = def.value;
        if allocation.parked_by_an_enclosing_scope(v) {
            continue;
        }
        locs[v.0 as usize] = Some(binding(allocation, v, allocation.where_at(v, i)));
    }
    let mut moves: alloc::vec::Vec<alloc::vec::Vec<(regalloc::ValueId, Binding)>> =
        (0..sched_len).map(|_| alloc::vec::Vec::new()).collect();
    // A value that is in a slot anywhere in this scope is stored there right
    // after its definition, from the register the definition wrote. That is
    // the whole of the slot-validity rule: a definition dominates every read,
    // and an `If` guard that skips a definition skips all of its readers
    // too, so there is no path on which a read finds the slot unwritten.
    let mut store_after_def: alloc::vec::Vec<Option<u32>> = alloc::vec![None; sched_len];
    for (i, def) in schedule.iter().enumerate() {
        let v = def.value;
        if allocation.parked_by_an_enclosing_scope(v) {
            // Live-in: an enclosing scope left it somewhere, and the head
            // reconciliation below brings it to where this scope expects it.
            continue;
        }
        for (index, at) in allocation.transitions(v) {
            if index <= i {
                continue; // The definition itself; the instruction writes it.
            }
            moves[index].push((v, binding(allocation, v, at)));
        }
        if let Some(slot) = allocation.slot_of(v)
            && matches!(
                locs[v.0 as usize],
                Some(Binding::Loc(Loc::Reg(_) | Loc::Ptr(_)))
            )
        {
            // Every definition writes a register, so this is the only place a
            // value reaches its slot — and it is the place that makes the slot
            // valid on both sides of every guard.
            store_after_def[i] = Some(slot.offset());
        }
    }

    backend.begin(schedule)?;

    // Bring an address into pointer register `p` from wherever `locs` says
    // it is: its slot, or another pointer register. The pointer class's
    // `emit_resolve`, with no constant to rematerialize.
    let ptr_into = |backend: &mut B,
                    code: &mut Vec<u8>,
                    vid: regalloc::ValueId,
                    p: PtrReg,
                    locs: &[Option<Binding>]| {
        match location_of(locs, vid) {
            Binding::Loc(Loc::Ptr(q)) => {
                if q != p {
                    backend.ptr_mov(code, p, q);
                }
            }
            Binding::Loc(Loc::Slot(slot)) => backend.ptr_load(code, p, slot.offset()),
            other => panic!("{vid:?} is an address but lives at {other:?}"),
        }
    };

    // The scope's head, where the previous iteration's tail flows back in. A
    // value live across this scope's back edge may end an iteration somewhere
    // other than where the next one expects to find it; this is what puts it
    // back, once per iteration — the cost the eviction that moved it was
    // charged.
    //
    // Always *from the slot*, never from whichever register the tail left it
    // in. The head has two predecessors — the back edge, and the fall-through
    // from the scope outside — and the slot is the one place that holds the
    // value on both. It is also why nothing is ever *stored* here: a value in
    // memory at the head is already in memory on both paths, since a value in
    // memory anywhere is stored right after its definition.
    //
    // Walked over the schedule, not over the map: an enclosing scope parks
    // every root it computes, and a scope inside reads only the subset that
    // reaches it.
    for vid in schedule.iter().map(|def| def.value) {
        if !allocation.parked_by_an_enclosing_scope(vid) {
            continue;
        }
        let placement = allocation.placement(vid);
        let at_head = allocation.at_head(vid);
        let head = binding(allocation, vid, at_head);
        if placement.at(regalloc::Point::TAIL) != at_head {
            let in_register = |at: &regalloc::Where| {
                matches!(at, regalloc::Where::Reg(_) | regalloc::Where::Ptr(_))
            };
            match head {
                Binding::Loc(Loc::Reg(r)) => {
                    let from_memory = placement
                        .locations()
                        .find(|at| !in_register(at))
                        .unwrap_or_else(|| {
                            unreachable!(
                                "a value that never leaves a register never changes register"
                            )
                        });
                    locs[vid.0 as usize] = Some(binding(allocation, vid, from_memory));
                    let got = backend.emit_resolve(&mut asm.code, vid, r, &locs)?;
                    debug_assert_eq!(got, r, "a value out of a register reloads into the target");
                }
                Binding::Loc(Loc::Ptr(p)) => {
                    let from_memory = placement
                        .locations()
                        .find(|at| !in_register(at))
                        .unwrap_or_else(|| {
                            unreachable!(
                                "a value that never leaves a register never changes register"
                            )
                        });
                    locs[vid.0 as usize] = Some(binding(allocation, vid, from_memory));
                    ptr_into(backend, &mut asm.code, vid, p, &locs);
                }
                Binding::Loc(Loc::Slot(_)) | Binding::Remat(_) => {}
            }
        }
        locs[vid.0 as usize] = Some(head);
    }

    // Hand a root this scope parks over to the scopes inside, right after
    // its def, while the value is guaranteed live in `at`. The slot is
    // written unless nothing inside will ever read it — which is exactly the
    // case where the value holds one register at every point of every scope
    // within; read off the placements, not off a flag beside them. Every
    // scope within, not just the first: a root parked here is live across
    // all of them, and one of them keeping it somewhere else is what makes
    // the slot load-bearing. A scope that never reads it has no opinion.
    // `at` is the register the definition wrote, of either class.
    let hand_off = |backend: &mut B,
                    code: &mut Vec<u8>,
                    vid: regalloc::ValueId,
                    at: Loc|
     -> Result<(), CompileError> {
        let Some(offset) = allocation.park(vid) else {
            return Ok(());
        };
        let head = allocation
            .within()
            .next()
            .map_or(regalloc::Where::Spilled, |inner| inner.at_head(vid));
        let resident_throughout = matches!(head, regalloc::Where::Reg(_) | regalloc::Where::Ptr(_))
            && allocation.within().all(|inner| {
                inner
                    .placement_of(vid)
                    .is_none_or(|p| p.locations().all(|at| at == head))
            });
        match (at, head) {
            (Loc::Reg(r), head) => {
                if !resident_throughout {
                    backend.emit_store(code, r, offset)?;
                }
                if let regalloc::Where::Reg(head_reg) = head
                    && head_reg != r
                {
                    backend.emit_mov(code, head_reg, r);
                }
            }
            (Loc::Ptr(p), head) => {
                if !resident_throughout {
                    backend.ptr_store(code, p, offset);
                }
                if let regalloc::Where::Ptr(head_ptr) = head
                    && head_ptr != p
                {
                    backend.ptr_mov(code, head_ptr, p);
                }
            }
            (Loc::Slot(_), _) => unreachable!("a definition writes a register"),
        }
        Ok(())
    };

    let is_unit = |op: &ScheduledOp| match op {
        ScheduledOp::Write { .. } | ScheduledOp::Seq(..) => true,
        ScheduledOp::Reduce(fold, _) => fold.monoid() == Monoid::SEQ,
        _ => false,
    };

    for (sched_idx, def) in schedule.iter().enumerate() {
        let (vid, sched_op) = (&def.value, &def.op);

        // Guard branches that end at this instruction, patched to the point
        // *before* this instruction's reconciliation — because that is the
        // join, and the reconciliation belongs to both paths.
        //
        // A skipped arm is still a path through the program, and the location
        // table is what every path after the join agrees on. A reload placed
        // at an arm's end brings a value back for the code that follows the
        // arm, not for the arm; patching the branch after it would let the
        // skipping path arrive with the register unloaded and the table
        // claiming otherwise. Ordering it first costs nothing when there is
        // nothing to reconcile — which is every kernel that reaches this
        // without a split live range.
        for pb in &branch_ends[sched_idx] {
            asm.bind(pb.past);
        }

        // Ranges that begin here. A register range starting away from the
        // value's definition is a reload the allocator chose to keep: the
        // value comes back into a pool register and stays there, instead of
        // being fetched into a scratch at every read.
        for (v, to) in core::mem::take(&mut moves[sched_idx]) {
            match to {
                Binding::Loc(Loc::Reg(r)) => {
                    let src = backend.emit_resolve(&mut asm.code, v, r, &locs)?;
                    if src != r {
                        backend.emit_mov(&mut asm.code, r, src);
                    }
                }
                Binding::Loc(Loc::Ptr(p)) => ptr_into(backend, &mut asm.code, v, p, &locs),
                Binding::Loc(Loc::Slot(_)) | Binding::Remat(_) => {}
            }
            locs[v.0 as usize] = Some(to);
        }

        // The registers this instruction's own guards may use: the allocator
        // reserved them here because a guard runs *between* instructions, at
        // a point the schedule does contain — the head of the arm it skips,
        // and the `If` that owns it.
        let scratch = allocation.scratch(sched_idx);
        let guard_mask = || {
            scratch.guard_mask.expect(
                "a guard's mask is not in a register and the allocator \
                 reserved nothing to reload it into",
            )
        };
        let guard_temp = scratch.guard_temp;
        let mask_guard_temp = scratch.mask_guard_temp;

        for pb in &branch_starts[sched_idx] {
            let (guard_idx, arm) = (pb.guard_idx, pb.arm);
            let guard = &if_guards[guard_idx];
            let mask_reg = match location_of(&locs, guard.mask_vid) {
                Binding::Loc(Loc::Reg(r)) => r,
                _ => backend.emit_resolve(&mut asm.code, guard.mask_vid, guard_mask(), &locs)?,
            };
            let test = MaskTest {
                reg: mask_reg,
                scratch: guard_temp,
                mask_scratch: mask_guard_temp,
                arm,
            };
            backend.branch_if_arm_is_dead(asm, test, pb.past);
        }

        // A parked value's placeholder def emits nothing — the enclosing
        // scope already parked the value in its slot; consumers reload from
        // there.
        if allocation.parked_by_an_enclosing_scope(*vid) {
            continue;
        }

        // A fold's binder, read here through its `Var`: the loop that binds
        // it seeded it and steps it, wherever the allocator keeps it, so its
        // def here is a placeholder too. `locs` already names the register
        // or the slot.
        if binder_placeholders.contains(vid) {
            continue;
        }

        // Sequencing is the unit monoid's combine: two effects, one after
        // the other, which the schedule's order already is. Nothing to emit.
        if let ScheduledOp::Seq(..) = sched_op {
            continue;
        }

        // The store. Its value is an ordinary operand, reloaded into the
        // reservation the allocator made when it is not resident; its row and
        // column are the enclosing folds' binders, wherever those loops keep
        // them; its width is the lane fold's trip count, folded into the def
        // when the lane fold was inlined (`program::lower::arena_to_schedule`).
        if let ScheduledOp::Write {
            row,
            col,
            lanes,
            value,
            ..
        } = sched_op
        {
            let value_reg = match location_of(&locs, *value) {
                Binding::Loc(Loc::Reg(r)) => r,
                _ => {
                    let target = scratch
                        .reload(0)
                        .expect("a Write's value is not resident and no reload was reserved");
                    backend.emit_resolve(&mut asm.code, *value, target, &locs)?
                }
            };
            backend.emit_write(
                &mut asm.code,
                &WritePlan {
                    value: value_reg,
                    row: binder_at(*row),
                    col: binder_at(*col),
                    lanes: *lanes,
                    scratch,
                },
            );
            continue;
        }

        // A surviving fold: seed its roots, run the loop, combine, step,
        // branch back. `vid`'s own placement was forced to its dedicated slot
        // at scan time (see `regalloc::LinearScan`), never a register, so
        // nothing about it needs `resolve_operands` — this is the whole of
        // what a `Reduce` def does.
        if let ScheduledOp::Reduce(fold, _) = sched_op {
            // The body is a scope of its own, opening exactly here — the
            // query `Allocation::opens_at` was built to answer, from the
            // other side, for exactly this walk. A `Reduce` def that opens
            // no scope here is an enclosing scope's fold, read from its slot
            // (`scopes::extract_folds`'s placeholder): that loop ran before this
            // scope began, `locs` already names the slot, and there is
            // nothing to emit.
            let Some(fold_scope) = allocation.fold_opening_at(sched_idx) else {
                continue;
            };
            let fold_alloc = allocation.sibling(fold_scope);
            let acc_slot = allocation.accumulator_slot(*vid);
            let binder_slot = allocation.binder_slot(*vid);
            // Two transient temps (`Scratch::REDUCE_TEMPS`): the trip test's
            // bound and its mask, reused by the combine's reload and the
            // step's scratch. Neither outlives the instruction it serves,
            // so the body's pool is free to hold them too.
            let temp = |i: usize| {
                scratch.temp(i).unwrap_or_else(|| {
                    panic!("{vid:?}'s Reduce def reserved fewer than {} temps", i + 1)
                })
            };
            let (t0, t1) = (temp(0), temp(1));

            // The loop's own roots, where the allocator put them: a register
            // it carries across the body, or a slot. Nothing is reserved for
            // either — at the floor the answer is "slot", which always fits,
            // and a body inside gets the whole pool minus what is carried.
            let carried_in = |at: regalloc::Where| match at {
                regalloc::Where::Reg(r) => Some(r),
                regalloc::Where::Ptr(_) => unreachable!("a fold's roots are vectors"),
                regalloc::Where::Spilled | regalloc::Where::Remat(_) => None,
            };
            let roots = fold_alloc.fold_roots();
            let binder_reg = carried_in(roots.binder);
            let acc_reg = carried_in(roots.accumulator);
            // A fold over the unit monoid has an accumulator nothing reads:
            // its combine emits no bytes, so neither does its seed or its
            // result — the slot it was given stays a dead vector of stack.
            let accumulates = fold.monoid() != Monoid::SEQ;

            let mut seed = |backend: &mut B,
                            at: Option<Reg>,
                            value: f32,
                            slot: u32|
             -> Result<(), CompileError> {
                match at {
                    Some(r) => backend.load_const(&mut asm.code, r, value),
                    None => {
                        backend.load_const(&mut asm.code, t0, value)?;
                        backend.slot_store(&mut asm.code, t0, slot);
                        Ok(())
                    }
                }
            };
            if accumulates {
                seed(backend, acc_reg, fold.monoid().identity(), acc_slot)?;
            }
            seed(backend, binder_reg, fold.range().start as f32, binder_slot)?;

            let (top, exit) = (asm.mint(), asm.mint());
            asm.bind(top);

            // Trip test: exit once every lane agrees the binder has reached
            // `hi` — `IfArm::False`'s test is exactly "every lane true",
            // which is what an all-lanes-equal broadcast compare produces
            // the instant it stops being false. The compare lands in `t0`,
            // which is either the binder's own reload or distinct from its
            // register; `alu` lets a source alias its destination on every
            // backend, so both are sound.
            let binder_now = match binder_reg {
                Some(b) => b,
                None => {
                    backend.slot_load(&mut asm.code, t0, binder_slot);
                    t0
                }
            };
            backend.load_const(&mut asm.code, t1, fold.range().end as f32)?;
            backend.test_ge(&mut asm.code, t0, [binder_now, t1], scratch.mask_guard_temp);
            backend.branch_if_arm_is_dead(
                asm,
                MaskTest {
                    reg: t0,
                    scratch: scratch.guard_temp,
                    mask_scratch: scratch.mask_guard_temp,
                    arm: IfArm::False,
                },
                exit,
            );

            let body_result = emit_scope(fold_alloc, backend, asm)?;

            // Combine: fold the body's result into the accumulator — an
            // ordinary two-register ALU op, outside the schedule (see
            // `IsaBackend::alu`'s doc) — where the accumulator lives. The
            // body's result may be in either temp, since its pool had both;
            // a slot-held accumulator round-trips through the other one.
            if accumulates {
                let body_result =
                    body_result.expect("a fold over a value monoid has a value to combine");
                match acc_reg {
                    Some(a) => backend.alu(&mut asm.code, fold.combine_op(), a, [a, body_result]),
                    None => {
                        let acc = if body_result == t0 { t1 } else { t0 };
                        backend.slot_load(&mut asm.code, acc, acc_slot);
                        backend.alu(&mut asm.code, fold.combine_op(), acc, [acc, body_result]);
                        backend.slot_store(&mut asm.code, acc, acc_slot);
                    }
                }
            }

            // Step and loop: the binder is the counter, so stepping it is
            // the whole of "advance the loop".
            let stride = fold.stride() as f32;
            match binder_reg {
                Some(b) => backend.add_scalar(&mut asm.code, b, t0, stride)?,
                None => {
                    backend.slot_load(&mut asm.code, t0, binder_slot);
                    backend.add_scalar(&mut asm.code, t0, t1, stride)?;
                    backend.slot_store(&mut asm.code, t0, binder_slot);
                }
            }
            backend.jump(asm, top);
            asm.bind(exit);

            // The result is read from the accumulator's slot — where this
            // scope's placement of the def says it is — so a carried
            // accumulator lands there once, on the way out. A scope inside
            // reads it there too: a `Reduce` is never a root (`scopes::stays_put`),
            // so nothing hands it over or carries it.
            if let Some(a) = acc_reg
                && accumulates
            {
                backend.slot_store(&mut asm.code, a, acc_slot);
            }
            continue;
        }

        let dst_loc = location_of(&locs, *vid);
        let plan = resolve_operands(sched_op, dst_loc, &locs, scratch);

        if let ScheduledOp::Ternary(OpKind::If, mask_vid, true_vid, false_vid) = sched_op
            && let Some(guard) = guard_at[sched_idx].map(|gi| &if_guards[gi])
            && guard.has_guarded_arm()
        {
            let mask_reg = match location_of(&locs, *mask_vid) {
                Binding::Loc(Loc::Reg(r)) => r,
                _ => backend.emit_resolve(&mut asm.code, *mask_vid, guard_mask(), &locs)?,
            };
            let dst = dst_loc.reg();
            let in_reg = |v: regalloc::ValueId| match location_of(&locs, v) {
                Binding::Loc(Loc::Reg(r)) => Some(r),
                _ => None,
            };
            let true_reg = in_reg(*true_vid);
            let false_reg = in_reg(*false_vid);

            let (only_false, only_true, join) = (asm.mint(), asm.mint(), asm.mint());

            // Both guards read `mask_reg`, which is why the reduction
            // scratch is a reservation of its own rather than whichever
            // register the mask was resolved into.
            let test = |arm| MaskTest {
                reg: mask_reg,
                scratch: guard_temp,
                mask_scratch: mask_guard_temp,
                arm,
            };
            backend.branch_if_arm_is_dead(asm, test(IfArm::True), only_false);
            backend.branch_if_arm_is_dead(asm, test(IfArm::False), only_true);

            // Mixed lanes: the blend, the path a lane-varying mask takes.
            backend.emit_plan(&mut asm.code, &plan)?;
            backend.jump(asm, join);

            asm.bind(only_false);
            if let Some(freg) = false_reg {
                backend.emit_mov(&mut asm.code, dst, freg);
            } else {
                backend.emit_resolve(&mut asm.code, *false_vid, dst, &locs)?;
            }
            backend.jump(asm, join);

            asm.bind(only_true);
            if let Some(treg) = true_reg {
                backend.emit_mov(&mut asm.code, dst, treg);
            } else {
                backend.emit_resolve(&mut asm.code, *true_vid, dst, &locs)?;
            }

            asm.bind(join);

            if let Some(offset) = store_after_def[sched_idx] {
                backend.emit_store(&mut asm.code, dst, offset)?;
            }
            hand_off(backend, &mut asm.code, *vid, Loc::Reg(dst))?;
            continue;
        }

        backend.emit_plan(&mut asm.code, &plan)?;

        // The register the definition wrote, of whichever class: a `Context`
        // def's is a pointer register, everything else's a vector one.
        let written = match dst_loc {
            Binding::Loc(loc) => loc,
            Binding::Remat(_) => Loc::Reg(Reg(u8::MAX)), // never stored, never handed off
        };
        if let Some(offset) = store_after_def[sched_idx] {
            match written {
                Loc::Reg(r) => backend.emit_store(&mut asm.code, r, offset)?,
                Loc::Ptr(p) => backend.ptr_store(&mut asm.code, p, offset),
                Loc::Slot(_) => unreachable!("a definition writes a register"),
            }
        }

        // Resident by construction: the hand-off is a read at the definition
        // (`regalloc::Pass::new`), so the allocator gave it a register — a
        // constant's definition included, which otherwise emits nothing.
        hand_off(backend, &mut asm.code, *vid, written)?;
    }

    // The scope's result, in a register for the fold around it to combine.
    // Usually the last instruction's own destination; not when the body's
    // root was hoisted out entirely and is read from its park, which is what
    // the allocator reserved a target on the last instruction for. An effect
    // has no result.
    let root_def = schedule.last().expect("empty schedule");
    let result_reg = if is_unit(&root_def.op) {
        None
    } else {
        let root = root_def.value;
        Some(match location_of(&locs, root) {
            Binding::Loc(Loc::Reg(r)) => r,
            _ => {
                let target = allocation.scratch(sched_len - 1).result.expect(
                    "the allocator reserves a result target on every scope's last instruction",
                );
                backend.emit_resolve(&mut asm.code, root, target, &locs)?
            }
        })
    };

    backend.scope_end(allocation.scope(), (asm.len() - start) as u64);
    Ok(result_reg)
}

/// Resolve a scheduled operation into a concrete instruction plan.
///
/// This is a PURE FUNCTION: no mutation, no side effects, no code emission.
/// Given the scheduled op, destination location, register assignments, and
/// spill slots, it computes exactly which registers to use and what
/// reload/store instructions are needed.
///
/// Every register here is the allocator's. The destination is the register it
/// gave this definition — every definition that emits an instruction has one —
/// and each operand not already in a register is reloaded into the register
/// [`operand_sources`] names for it, which is either the destination (safe:
/// every backend reads all of an instruction's sources before writing it) or
/// one of this instruction's own reservations.
///
/// # Panics
/// If the destination is in a stack slot. A definition writes a register or
/// nothing at all; a spilled destination was the fixed `reload[0]`, and there
/// is no such register any more.
///
/// If a `Ternary` names an op other than `MulAdd` or `If`: the arena refuses a
/// ternary node whose op is not ternary when the node is built, `legalize`
/// lowers every `Gather` to `RawGather`, and lowering forwards nodes unchanged,
/// so that is a pipeline bug ([`unimplemented_op`]), not a kernel.
fn resolve_operands(
    op: &ScheduledOp,
    dst_loc: Binding,
    locs: &[Option<Binding>],
    scratch: regalloc::Scratch,
) -> InstructionPlan {
    // The one pointer-class definition, resolved before the vector
    // destination is read: its register is a pointer register by the
    // allocator's own placement, and it has no operands to resolve.
    if let ScheduledOp::Context(slot) = op {
        let dst = match dst_loc {
            Binding::Loc(Loc::Ptr(p)) => p,
            other => panic!(
                "a Context def landed at {other:?} — the allocator owes every \
                 pointer definition a pointer register"
            ),
        };
        return InstructionPlan {
            reloads: Vec::new(),
            op: ResolvedOp::Context { dst, slot: *slot },
            setup_mov: None,
            scratch,
        };
    }

    let dst = match dst_loc {
        Binding::Loc(Loc::Reg(r)) => r,
        // A rematerialized constant: it lives nowhere and is rebuilt at each
        // use, so its definition computes nothing. Emitting a load into a
        // register nobody reads is what the fixed destination register used to
        // buy.
        Binding::Remat(_) => {
            return InstructionPlan {
                reloads: Vec::new(),
                op: ResolvedOp::Nop,
                setup_mov: None,
                scratch,
            };
        }
        Binding::Loc(Loc::Ptr(p)) => panic!(
            "a vector definition landed in pointer register {p:?} — the \
             allocator placed a value in the wrong class's file"
        ),
        Binding::Loc(Loc::Slot(slot)) => panic!(
            "a definition landed in stack slot {} — the allocator owes \
             every definition a register, since there is none outside the pool \
             to compute into",
            slot.offset()
        ),
    };

    let mut reloads = Vec::new();
    let mut setup_mov = None;

    // The address an instruction reads, in a pointer register: where the
    // allocator keeps it, or reloaded from its slot into the one pointer
    // register it reserved for this instruction. Never a constant.
    let base_of = |v: regalloc::ValueId, reloads: &mut Vec<Reload>| -> PtrReg {
        match location_of(locs, v) {
            Binding::Loc(Loc::Ptr(p)) => p,
            Binding::Loc(Loc::Slot(slot)) => {
                let target = scratch.ptr_reload.unwrap_or_else(|| {
                    panic!(
                        "{v:?} is an address in a slot and the allocator reserved no \
                         pointer register to reload it into"
                    )
                });
                reloads.push(Reload::Ptr { target, slot });
                target
            }
            other => panic!("{v:?} is read as an address but lives at {other:?}"),
        }
    };
    // "Not in a register" — a rematerialized value needs a reload target just
    // as a spilled one does, so both answer false here.
    let in_register =
        |v: &regalloc::ValueId| matches!(location_of(locs, *v), Binding::Loc(Loc::Reg(_)));

    // Where each operand comes from, and so which register each reload lands
    // in. The same call the allocator made when it decided how many to
    // reserve — residency is final between the two, so the two answers are the
    // same answer.
    let mut resident = [true; 3];
    for (k, operand) in regalloc::operands(op).enumerate() {
        resident[k] = in_register(&operand);
    }
    let sources = operand_sources(op, resident);
    // The register operand `k` is reloaded into. Resident operands never ask.
    let target_for = |k: usize| -> Reg {
        match sources[k] {
            OperandSource::Resident => {
                unreachable!("a resident operand is read where it is, not reloaded")
            }
            OperandSource::Destination => dst,
            OperandSource::Reload(slot) => scratch.reload(slot).unwrap_or_else(|| {
                panic!(
                    "operand {k} needs reload register {slot}, which the \
                     allocator did not reserve"
                )
            }),
        }
    };

    let resolve = |v: regalloc::ValueId, target: Reg, reloads: &mut Vec<Reload>| -> Reg {
        match location_of(locs, v) {
            Binding::Loc(Loc::Reg(reg)) => reg,
            Binding::Remat(bits) => {
                reloads.push(Reload::Const {
                    target,
                    val_bits: bits,
                });
                target
            }
            Binding::Loc(Loc::Slot(slot)) => {
                reloads.push(Reload::FromStack { target, slot });
                target
            }
            Binding::Loc(Loc::Ptr(p)) => {
                panic!("{v:?} is read as a vector but is an address in {p:?}")
            }
        }
    };
    // Operand `k`, from wherever it is: its own register, or the one
    // [`operand_sources`] reserved for it.
    let operand = |k: usize, v: regalloc::ValueId, reloads: &mut Vec<Reload>| -> Reg {
        match sources[k] {
            OperandSource::Resident => location_of(locs, v).reg(),
            OperandSource::Destination | OperandSource::Reload(_) => {
                resolve(v, target_for(k), reloads)
            }
        }
    };

    let resolved_op = match op {
        ScheduledOp::Var(_) => {
            // A binder's placeholder: the loop seeded it — no code needed.
            ResolvedOp::Nop
        }
        ScheduledOp::Lanes(_) => ResolvedOp::Lanes { dst },
        // Unreachable precondition: both are effects the emit loop handles
        // before this function is ever called, the same way a hoisted
        // placeholder never reaches here either.
        ScheduledOp::Write { .. } | ScheduledOp::Seq(..) => unreachable!(
            "resolve_operands: an effect reached the generic resolver -- \
             emit_scope must special-case it before calling this"
        ),
        ScheduledOp::Const(val) => ResolvedOp::LoadConst {
            dst,
            val_bits: val.to_bits(),
        },
        ScheduledOp::Unary(op_kind, child) => {
            let src = operand(0, *child, &mut reloads);
            ResolvedOp::Unary {
                op: *op_kind,
                dst,
                src,
            }
        }
        ScheduledOp::ShiftImm(op_kind, child, amount) => {
            let src = operand(0, *child, &mut reloads);
            ResolvedOp::ShiftImm {
                op: *op_kind,
                dst,
                src,
                amount: *amount,
            }
        }
        ScheduledOp::Gather(child, base) => {
            let idx = operand(0, *child, &mut reloads);
            let base = base_of(*base, &mut reloads);
            ResolvedOp::Gather { dst, idx, base }
        }
        ScheduledOp::Broadcast(child, base) => {
            let idx = operand(0, *child, &mut reloads);
            let base = base_of(*base, &mut reloads);
            ResolvedOp::Broadcast { dst, idx, base }
        }
        ScheduledOp::Uniform(base, offset) => ResolvedOp::Uniform {
            dst,
            base: base_of(*base, &mut reloads),
            offset: *offset,
        },
        ScheduledOp::Context(_) => unreachable!("resolved above, before the vector destination"),
        ScheduledOp::Outer(_) => unreachable!(
            "resolve_operands: an Outer def reached the generic resolver -- \
             an enclosing scope parks its value, and emit_scope skips it"
        ),
        // Unreachable precondition: a surviving `Reduce`'s def is forced to
        // `Where::Spilled` at scan time (never a register — the `dst` match
        // above already panics on that), and `emit_dag_body_hoisted` special-
        // cases it before this function is ever called, the same way a
        // hoisted placeholder never reaches here either.
        ScheduledOp::Reduce(..) => unreachable!(
            "resolve_operands: a Reduce def reached the generic resolver -- \
             emit_scope must special-case it before calling this"
        ),
        ScheduledOp::Binary(op_kind, left, right) => {
            // `left` goes to `dst` when it needs reloading (`operand_sources`'
            // one free target) and `right` to a reservation. Every backend's
            // binary form is three-operand, so either may alias `dst`.
            let l_reg = operand(0, *left, &mut reloads);
            let r_reg = operand(1, *right, &mut reloads);
            ResolvedOp::Binary {
                op: *op_kind,
                dst,
                left: l_reg,
                right: r_reg,
            }
        }
        ScheduledOp::Ternary(op_kind, a, b, c) => {
            match op_kind {
                OpKind::MulAdd => {
                    // FMLA path: dst += a * b, so dst must hold c first —
                    // which is where `operand_sources` sends a spilled `c`.
                    let c_reg = operand(2, *c, &mut reloads);
                    if dst.0 != c_reg.0 {
                        setup_mov = Some((dst, c_reg));
                    }
                    let a_reg = operand(0, *a, &mut reloads);
                    let b_reg = operand(1, *b, &mut reloads);
                    ResolvedOp::FusedMulAdd {
                        dst,
                        a: a_reg,
                        b: b_reg,
                    }
                }
                OpKind::If => {
                    // BSL/blend is a 3-input RMW: the mask must end up in `dst`,
                    // and if_true / if_false each need their own live register.
                    //
                    // A spilled mask reloads STRAIGHT into `dst`, which is what
                    // `operand_sources` says for operand 0 here. Every reload
                    // emits before `setup_mov`, so routing the mask through a
                    // register a spilled arm also reloads into would overwrite
                    // it before it reached `dst`; one reservation per arm is
                    // why that cannot happen. Both arms spilled at once used
                    // to need a third fixed register (`if_reload`), held
                    // out of every kernel's pool for the rare kernel reaching
                    // it.
                    let a_reg = operand(0, *a, &mut reloads);
                    if dst.0 != a_reg.0 {
                        setup_mov = Some((dst, a_reg));
                    }
                    let b_reg = operand(1, *b, &mut reloads);
                    let c_reg = operand(2, *c, &mut reloads);
                    ResolvedOp::If {
                        dst,
                        if_true: b_reg,
                        if_false: c_reg,
                    }
                }
                _ => unimplemented_op("the ternary resolver", *op_kind),
            }
        }
    };

    InstructionPlan {
        reloads,
        op: resolved_op,
        setup_mov,
        scratch,
    }
}

/// Where a value lives, from the dense slice the emit loop carries.
///
/// One lookup, indexed by `ValueId.0`. It replaced three parallel slices whose
/// disagreement was a runtime check; a [`Binding`] is one answer, so there is
/// nothing left to disagree.
fn location_of(locs: &[Option<Binding>], vid: regalloc::ValueId) -> Binding {
    locs.get(vid.0 as usize)
        .copied()
        .flatten()
        .unwrap_or_else(|| panic!("{vid:?} has no binding"))
}

// =============================================================================
// The one place a target decides anything
// =============================================================================

/// Drive `schedule` to a kernel on the backend the host's CPU selected.
///
/// Every [`IsaBackend`] compiles on every host — emission is a pure function of
/// `(schedule, RegisterFile)` into a `Vec<u8>`, and an x86 machine is perfectly
/// capable of computing NEON instruction words. So the target does not decide
/// which backends *exist*; it decides which one is *instantiated*, here, from
/// the tier [`crate::isa::detect`] read off CPUID at startup. Each arm
/// monomorphizes the driver against one concrete backend, exactly as the
/// `cfg(target_feature)`-selected `Native` alias this replaces did: static
/// dispatch inside the compile, no `dyn`, no vtable. The one `match` is this,
/// and it runs once per kernel, not once per instruction.
///
/// `detect` answers `Neon` only on aarch64 and `Avx2`/`Avx512` only on x86-64,
/// so no arm carries a `cfg`: the other architecture's backend is typechecked,
/// swept for op coverage and unit-tested on this host, and its arm is never
/// taken. There is no tier below AVX2+FMA on x86-64
/// (docs/plans/2026-09-22-the-isa-is-decided-at-startup.md).
///
/// Genuinely host-bound code lives in [`executable`] (the `KernelFn` ABI types
/// and the `mmap`/`mprotect` that makes bytes callable) and nowhere else.
pub(crate) fn compile_native(
    program: regalloc::ScopedSchedule,
) -> Result<CompileResult, CompileError> {
    match crate::isa::detect() {
        Isa::Avx2 => compile_via_backend(program, &mut avx2::driver::Avx2Backend::new()),
        Isa::Avx512 => compile_via_backend(program, &mut avx512::driver::Avx512Backend::new()),
        Isa::Neon => compile_via_backend(program, &mut aarch64::driver::Aarch64Backend::new()),
    }
}

/// Drive a schedule to a complete collapse kernel via an [`IsaBackend`]: the
/// body from [`emit_scope`], which emits every fold nested in it, framed by
/// the function's own frame.
fn compile_via_backend<B: IsaBackend>(
    program: regalloc::ScopedSchedule,
    backend: &mut B,
) -> Result<CompileResult, CompileError> {
    use regalloc::RegisterAllocator;

    let file = backend.register_file();
    let nest = regalloc::LinearScan.allocate_nest(program, &file)?;

    // Every byte below is emitted through this decorator, so the counts it
    // hands back cover the whole function by construction (see `traffic`).
    let mut counting = Counting::new(backend);

    // Every scope shares one stack frame, laid out by the allocator beside
    // its placements (`regalloc::NestAllocation::new`): spill slots below
    // `spill_bytes`, each fold's two slots and the parks above, `frame_bytes`
    // in all. The body's emission reaches every fold nested in it.
    //
    // The function around the body is one program with it: the frame, the
    // anchor for whatever the body's constants are relative to, the return,
    // and what trails it.
    let mut asm = Assembly::default();
    counting.frame_alloc(&mut asm.code, nest.frame_bytes());
    let pool = asm.mint();
    counting.anchor(&mut asm, pool);
    let body_start = asm.len();
    emit_scope(nest.body(), &mut counting, &mut asm)?;
    let body_bytes = asm.len() - body_start;
    counting.frame_free(&mut asm.code, nest.frame_bytes());
    counting.emit_ret(&mut asm.code);
    let scaffold = counting.take((asm.len() - body_bytes) as u64);
    counting.finish(&mut asm, pool);
    let code = asm.finish();
    let scopes = counting.scopes();

    // How many times one call runs each scope: the body once, a fold its
    // trip count times its parent's. A fold's parent is an earlier scope,
    // so each entry is computed after the one it multiplies.
    let mut trips: alloc::vec::Vec<u64> = alloc::vec![1];
    for j in 0..nest.fold_count() {
        let parent = nest.fold_parent(j);
        let vid = nest.fold_reduce_vid(j);
        let len = nest
            .scope(parent)
            .schedule()
            .iter()
            .find_map(|def| match def.op {
                ScheduledOp::Reduce(fold, _) if def.value == vid => Some(u64::from(fold.len())),
                _ => None,
            })
            .expect("a fold opens at its parent's Reduce def");
        let parent_trips = match parent {
            regalloc::Scope::Body => trips[0],
            regalloc::Scope::Fold(p) => trips[p + 1],
        };
        trips.push(parent_trips * len);
    }

    // A parked root that holds a register at the head of the scopes inside
    // its own is carried rather than reloaded per iteration — read off the
    // placement, which is where the answer lives.
    let carried = nest
        .parks()
        .filter(|root| nest.carried(*root).is_some())
        .count() as u64;
    let exec = unsafe { executable::CompiledKernel::from_code(&code)? };
    Ok(CompileResult {
        code: exec,
        spill_count: nest.body().spill_slots(),
        spill_bytes: nest.spill_bytes(),
        hoisted_values: nest.parks().count() as u64,
        traffic: EmitTraffic {
            scopes: EmitTraffic::by_index(scopes, trips.len()),
            trips,
            scaffold,
            vector_bytes: file.vector_bytes,
            pool: file.scratch.len(),
            carried,
        },
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::tests::{BYTES_PER_LANE, schedule_for};
    use pixelflow_ir::LatticeShape;
    use pixelflow_ir::arena::{ExprArena, ExprId, UniformId};

    /// A schedule allocated as a compile allocates it: scoped by the same
    /// [`regalloc::ScopedSchedule::from_schedule`], so a hand-built
    /// loop-free schedule is one body laid out as a compile lays one out.
    pub(super) fn allocate_nest(
        schedule: Vec<regalloc::Def>,
        file: &regalloc::RegisterFile,
    ) -> regalloc::NestAllocation {
        use regalloc::RegisterAllocator;
        regalloc::LinearScan
            .allocate_nest(regalloc::ScopedSchedule::from_schedule(schedule), file)
            .expect("a test nest fits the frame")
    }

    /// [`compile_via_backend`] on a lowered schedule: what a test that compiles
    /// for a chosen backend hands it, scoped the way a compile scopes it.
    pub(super) fn compile_schedule<B: IsaBackend>(
        schedule: Vec<regalloc::Def>,
        backend: &mut B,
    ) -> Result<CompileResult, CompileError> {
        compile_via_backend(regalloc::ScopedSchedule::from_schedule(schedule), backend)
    }

    /// Allocate a straight-line schedule and emit it as one scope's body: the
    /// one-scope view the emitter's own tests are written against.
    pub(super) fn emit_dag_body<B: IsaBackend>(
        schedule: Vec<regalloc::Def>,
        backend: &mut B,
    ) -> Result<(Vec<u8>, Reg), CompileError> {
        let nest = allocate_nest(schedule, &backend.register_file());
        let mut asm = Assembly::default();
        let result = emit_scope(nest.body(), backend, &mut asm)?;
        Ok((
            asm.finish(),
            result.expect("a value-rooted schedule has a result"),
        ))
    }

    /// The register file of the backend [`compile_native`] instantiates on
    /// this host.
    fn native_file() -> regalloc::RegisterFile {
        match crate::isa::detect() {
            Isa::Avx2 => avx2::driver::Avx2Backend::new().register_file(),
            Isa::Avx512 => avx512::driver::Avx512Backend::new().register_file(),
            Isa::Neon => aarch64::driver::Aarch64Backend::new().register_file(),
        }
    }

    /// A backend whose scratch pool is `.1` registers above the floor
    /// ([`regalloc::tests::at_floor`]): how a test reaches, on a kernel small enough to reason
    /// about, the allocation a kernel too wide for the whole pool gets. Every
    /// verb but [`IsaBackend::register_file`] is the wrapped backend's own.
    pub(super) struct AtFloor<B>(pub(super) B, pub(super) u8);

    impl<B: IsaBackend> IsaBackend for AtFloor<B> {
        fn jump(&mut self, asm: &mut Assembly, label: Label) {
            self.0.jump(asm, label);
        }
        fn register_file(&self) -> regalloc::RegisterFile {
            regalloc::tests::at_floor(self.0.register_file(), self.1)
        }
        fn begin(&mut self, schedule: &[regalloc::Def]) -> Result<(), CompileError> {
            self.0.begin(schedule)
        }
        fn emit_plan(
            &mut self,
            code: &mut Vec<u8>,
            plan: &InstructionPlan,
        ) -> Result<(), CompileError> {
            self.0.emit_plan(code, plan)
        }
        fn emit_mov(&mut self, code: &mut Vec<u8>, dst: Reg, src: Reg) {
            self.0.emit_mov(code, dst, src);
        }
        fn emit_store(
            &mut self,
            code: &mut Vec<u8>,
            src: Reg,
            offset: u32,
        ) -> Result<(), CompileError> {
            self.0.emit_store(code, src, offset)
        }
        fn ptr_store(&mut self, code: &mut Vec<u8>, src: PtrReg, offset: u32) {
            self.0.ptr_store(code, src, offset);
        }
        fn ptr_load(&mut self, code: &mut Vec<u8>, dst: PtrReg, offset: u32) {
            self.0.ptr_load(code, dst, offset);
        }
        fn ptr_mov(&mut self, code: &mut Vec<u8>, dst: PtrReg, src: PtrReg) {
            self.0.ptr_mov(code, dst, src);
        }
        fn emit_resolve(
            &mut self,
            code: &mut Vec<u8>,
            vid: regalloc::ValueId,
            target: Reg,
            locs: &[Option<Binding>],
        ) -> Result<Reg, CompileError> {
            self.0.emit_resolve(code, vid, target, locs)
        }
        fn branch_if_arm_is_dead(&mut self, asm: &mut Assembly, test: MaskTest, label: Label) {
            self.0.branch_if_arm_is_dead(asm, test, label);
        }
        fn frame_alloc(&mut self, code: &mut Vec<u8>, bytes: u32) {
            self.0.frame_alloc(code, bytes);
        }
        fn frame_free(&mut self, code: &mut Vec<u8>, bytes: u32) {
            self.0.frame_free(code, bytes);
        }
        fn anchor(&mut self, asm: &mut Assembly, pool: Label) {
            self.0.anchor(asm, pool);
        }
        fn finish(&mut self, asm: &mut Assembly, pool: Label) {
            self.0.finish(asm, pool);
        }
        fn slot_store(&mut self, code: &mut Vec<u8>, src: Reg, offset: u32) {
            self.0.slot_store(code, src, offset);
        }
        fn slot_load(&mut self, code: &mut Vec<u8>, dst: Reg, offset: u32) {
            self.0.slot_load(code, dst, offset);
        }
        fn scope_begin(&mut self) {
            self.0.scope_begin();
        }
        fn scope_end(&mut self, scope: regalloc::Scope, bytes: u64) {
            self.0.scope_end(scope, bytes);
        }
        fn add_scalar(
            &mut self,
            code: &mut Vec<u8>,
            dst: Reg,
            scratch: Reg,
            scalar: f32,
        ) -> Result<(), CompileError> {
            self.0.add_scalar(code, dst, scratch, scalar)
        }
        fn load_const(
            &mut self,
            code: &mut Vec<u8>,
            dst: Reg,
            val: f32,
        ) -> Result<(), CompileError> {
            self.0.load_const(code, dst, val)
        }
        fn alu(&mut self, code: &mut Vec<u8>, op: OpKind, dst: Reg, srcs: [Reg; 2]) {
            self.0.alu(code, op, dst, srcs);
        }
        fn test_ge(
            &mut self,
            code: &mut Vec<u8>,
            dst: Reg,
            srcs: [Reg; 2],
            mask_scratch: Option<KReg>,
        ) {
            self.0.test_ge(code, dst, srcs, mask_scratch);
        }
        fn emit_write(&mut self, code: &mut Vec<u8>, write: &WritePlan) {
            self.0.emit_write(code, write);
        }
        fn emit_ret(&mut self, code: &mut Vec<u8>) {
            self.0.emit_ret(code);
        }
    }

    /// `a` compiled for this host with the native backend at the floor
    /// ([`AtFloor`]).
    pub(super) fn compile_at_floor(
        a: &ExprArena,
        root: ExprId,
        shape: LatticeShape,
    ) -> CompileResult {
        compile_above_floor(a, root, shape, 0)
    }

    /// `a` compiled for this host with the native backend `above` registers
    /// above the floor ([`AtFloor`]).
    fn compile_above_floor(
        a: &ExprArena,
        root: ExprId,
        shape: LatticeShape,
        above: u8,
    ) -> CompileResult {
        let schedule = native_schedule(a, root, shape);
        match crate::isa::detect() {
            Isa::Avx2 => compile_schedule(
                schedule,
                &mut AtFloor(avx2::driver::Avx2Backend::new(), above),
            ),
            Isa::Avx512 => compile_schedule(
                schedule,
                &mut AtFloor(avx512::driver::Avx512Backend::new(), above),
            ),
            Isa::Neon => compile_schedule(
                schedule,
                &mut AtFloor(aarch64::driver::Aarch64Backend::new(), above),
            ),
        }
        .expect("a test kernel compiles at the floor")
    }

    /// Lanes in one SIMD batch at the tier this host selected.
    fn lanes() -> usize {
        crate::jit_vector_bytes() / core::mem::size_of::<f32>()
    }

    /// One sample, so the lattice's origin *is* the point the kernel is
    /// evaluated at: what a test about arithmetic rather than about the loop
    /// nest compiles for.
    const POINT: LatticeShape = LatticeShape::new([1, 1]);

    /// One full batch of one row: `x` runs `x0 .. x0 + lanes()`, which is
    /// what a test about per-lane behaviour needs.
    fn batch() -> LatticeShape {
        LatticeShape::new([lanes() as u32, 1])
    }

    /// Each backend's register file is the width its kernels are legalized
    /// and framed at, and the width is the ISA's: a `ymm` is 32 bytes, a
    /// `zmm` 64 and a NEON `q` register 16, by definition.
    #[test]
    fn every_backends_vector_width_is_its_tiers() {
        let files = [
            (Isa::Avx2, avx2::driver::Avx2Backend::new().register_file()),
            (
                Isa::Avx512,
                avx512::driver::Avx512Backend::new().register_file(),
            ),
            (
                Isa::Neon,
                aarch64::driver::Aarch64Backend::new().register_file(),
            ),
        ];
        for (isa, file) in files {
            let isa_bytes = match isa {
                Isa::Avx2 => 32,
                Isa::Avx512 => 64,
                Isa::Neon => 16,
            };
            assert_eq!(file.vector_bytes, isa_bytes, "{isa:?}");
        }
        // And the width a compile legalizes at (`jit_vector_bytes`) is the
        // width of the backend it instantiates.
        assert_eq!(
            native_file().vector_bytes as usize,
            crate::jit_vector_bytes()
        );
    }

    /// Run the collapse `code` is over a plane of exactly `shape`'s extent,
    /// and hand the plane back.
    ///
    /// `buffers` binds the arena's buffer slots, `uniforms` its uniform
    /// block, and `(x, y)` is where the lattice's sample `(0, 0)` lies. The
    /// pitch is the width, so a sample reads as `out[row * width + col]`.
    fn collapse_into(
        code: &executable::CompiledKernel,
        buffers: &[*const f32],
        uniforms: &[f32],
        (x, y): (f32, f32),
        shape: LatticeShape,
    ) -> Vec<f32> {
        let [width, height] = shape.extent().map(|n| n as usize);
        let mut out = alloc::vec![f32::NAN; width * height];
        let origin = [x, y];
        let mut ctx: Vec<*const f32> = buffers.to_vec();
        ctx.push(uniforms.as_ptr());
        ctx.push(origin.as_ptr());
        // SAFETY: `ctx` binds every buffer the arena declared, then the
        // uniform block and the origin block, each live for the call; `out`
        // is the whole plane the kernel was compiled at.
        unsafe {
            code.call(ctx.as_ptr(), out.as_mut_ptr(), width);
        }
        out
    }

    /// Evaluate a kernel compiled at [`POINT`] at one lattice point.
    fn eval_point(code: &executable::CompiledKernel, x: f32, y: f32) -> f32 {
        collapse_into(code, &[], &[], (x, y), POINT)[0]
    }

    /// Evaluate a kernel compiled at [`batch`]: one row of `lanes()` samples
    /// from `(x, y)`.
    fn eval_batch(
        code: &executable::CompiledKernel,
        buffers: &[*const f32],
        uniforms: &[f32],
        x: f32,
        y: f32,
    ) -> Vec<f32> {
        collapse_into(code, buffers, uniforms, (x, y), batch())
    }

    /// [`schedule_for`] at this host's own lane count.
    fn native_schedule(a: &ExprArena, root: ExprId, shape: LatticeShape) -> Vec<regalloc::Def> {
        schedule_for(a, root, shape, lanes() as u32)
    }

    /// Whether any range of `p`'s life is in a stack slot.
    fn spills(p: &regalloc::Placement) -> bool {
        p.locations().any(|at| at == regalloc::Where::Spilled)
    }

    /// The branches of the kernel at `root`, compiled the way `compile`
    /// compiles it, read off the allocation's own guard tables: `If`s with a
    /// guarded arm, arms guarded, and schedule entries under a guard (an
    /// entry inside two nested arms counts in each).
    fn census(a: &ExprArena, root: ExprId, shape: LatticeShape) -> (usize, usize, usize) {
        let file = native_file();
        let nest = allocate_nest(native_schedule(a, root, shape), &file);
        let scopes = core::iter::once(regalloc::Scope::Body)
            .chain((0..nest.fold_count()).map(regalloc::Scope::Fold));
        let mut census = (0, 0, 0);
        for scope in scopes {
            for guard in nest.scope(scope).if_guards() {
                census.0 += usize::from(guard.has_guarded_arm());
                for arm in IfArm::ALL.into_iter().filter(|&arm| {
                    let (start, end) = guard.range(arm);
                    start != end
                }) {
                    let (start, end) = guard.range(arm);
                    census.1 += 1;
                    census.2 += end - start;
                }
            }
        }
        census
    }

    /// The route that *does* work: the compile entry expands the reference
    /// before it schedules, so a kernel composed by reference emits exactly
    /// what the spliced composition emits.
    #[test]
    fn a_reference_compiles_through_the_entry_point() {
        let body = pixelflow_ir::Kernel::x().mul(&pixelflow_ir::Kernel::constant(3.0));
        let named = body.by_ref().add(&pixelflow_ir::Kernel::y());
        let direct = body.add(&pixelflow_ir::Kernel::y());
        let (n_arena, n_root) = named.parts();
        let (d_arena, d_root) = direct.parts();
        let named_code = compile(n_arena, n_root, POINT).expect("a named kernel compiles");
        let direct_code = compile(d_arena, d_root, POINT).expect("and so does the spliced one");
        for (x, y) in [(0.0f32, 0.0f32), (1.5, -2.0), (-3.25, 7.5)] {
            let want = eval_point(&direct_code.code, x, y);
            let got = eval_point(&named_code.code, x, y);
            assert_eq!(got, want, "at ({x}, {y})");
        }
    }

    /// `passes::legalize` leaves every `Reduce` standing, so a fold reaches
    /// the emitter as a loop; this builds one by hand and compiles it.
    ///
    /// `SUM` over four terms, matched against the closed form
    /// `⊕_{i<4}(X+i) = 4X + 6`; `MIN` over the same range, matched against
    /// whichever term is smallest for a given `X` — two different monoids,
    /// so a bug specific to one algebra's identity or combine opcode has
    /// somewhere to show up.
    #[test]
    fn a_surviving_reduce_compiles_and_runs() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        let binder = Binder::from_slot(0).expect("slot 0 exists");

        // sum_{i=0}^{3} (X + i) = 4*X + (0+1+2+3) = 4*X + 6
        let mut sum_arena = ExprArena::new();
        let x = sum_arena.push_var(0);
        let i = sum_arena.push_var(binder.var());
        let body = sum_arena.push_binary(OpKind::Add, x, i);
        let sum_fold = Fold::new(Monoid::SUM, binder, 0..4);
        let sum_root = sum_arena.push_reduce(sum_fold, body);
        let sum_code =
            compile(&sum_arena, sum_root, POINT).expect("a surviving SUM Reduce compiles");

        for x in [0.0f32, 2.0, -1.5, 10.0] {
            let got = eval_point(&sum_code.code, x, 0.0);
            let want = 4.0 * x + 6.0;
            assert_eq!(got, want, "SUM at x={x}");
        }

        // min_{i=0}^{3} (X - i) = X - 3, always the last term.
        let mut min_arena = ExprArena::new();
        let x = min_arena.push_var(0);
        let i = min_arena.push_var(binder.var());
        let body = min_arena.push_binary(OpKind::Sub, x, i);
        let min_fold = Fold::new(Monoid::MIN, binder, 0..4);
        let min_root = min_arena.push_reduce(min_fold, body);
        let min_code =
            compile(&min_arena, min_root, POINT).expect("a surviving MIN Reduce compiles");

        for x in [0.0f32, 2.0, -1.5, 10.0] {
            let got = eval_point(&min_code.code, x, 0.0);
            let want = x - 3.0;
            assert_eq!(got, want, "MIN at x={x}");
        }
    }

    /// A fold whose result is *not* the kernel's own root, and whose body
    /// shares an invariant leaf with code outside it.
    ///
    /// Both are the two ways `extract_folds` could get this wrong: reading
    /// the `Reduce` as an ordinary operand (most kernels' folds feed further
    /// arithmetic, unlike the previous test's bare fold) exercises the same
    /// slot override every other spilled value's reader already goes
    /// through; the shared `shared = X * X` node exercises that removing a
    /// fold's body must be *variance*-driven, not *reachability*-driven —
    /// reachability would drop `shared`'s def from the body's own schedule
    /// too and orphan its other reader.
    ///
    /// `reduce = sum_{i=0}^{2}(shared + i) = 3*shared + 3`;
    /// `root = reduce + shared + Y = 4*shared + 3 + Y`.
    #[test]
    fn a_surviving_reduce_shares_a_leaf_and_feeds_further_arithmetic() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let shared = a.push_binary(OpKind::Mul, x, x);
        let binder = Binder::from_slot(0).expect("slot 0 exists");
        let i = a.push_var(binder.var());
        let body = a.push_binary(OpKind::Add, shared, i);
        let fold = Fold::new(Monoid::SUM, binder, 0..3);
        let reduce = a.push_reduce(fold, body);
        let plus_shared = a.push_binary(OpKind::Add, reduce, shared);
        let root = a.push_binary(OpKind::Add, plus_shared, y);

        let code = compile(&a, root, POINT).expect("a fold that feeds further arithmetic compiles");

        for (x, y) in [(0.0f32, 0.0f32), (2.0, 1.0), (-1.5, 3.0), (10.0, -4.0)] {
            let got = eval_point(&code.code, x, y);
            let want = 4.0 * x * x + 3.0 + y;
            assert_eq!(got, want, "at (x={x}, y={y})");
        }
    }

    /// A surviving fold's own per-iteration recompute of a shared invariant
    /// leaf must not clobber the outer scope's copy of that same value.
    ///
    /// Same DAG as `a_surviving_reduce_shares_a_leaf_and_feeds_further_arithmetic`
    /// (`shared` read once inside the fold's body, once again after it),
    /// but pushed to the arena in the *interleaved* order unrolling a fold
    /// produces (const, add, const, add, const, add) rather than all three
    /// constants first. That reordering alone, with no change to the DAG's shape,
    /// used to compute `24` instead of `21`: the outer scope's own copy of
    /// `shared` shared a register with the fold's own per-iteration
    /// recompute of it, and the fold's internal register allocation —
    /// `allocate_nest`'s fold scope, recursed into via `Allocation::sibling`
    /// — has no visibility into what the enclosing scope holds resident, so
    /// its last iteration's write clobbered it. Fixed by evicting every
    /// register the enclosing scope holds resident at a `Reduce`'s own
    /// schedule position (`regalloc::Pass::split_out`, called for every
    /// occupied slot right after `pass.expire` in `scan`), exactly as a
    /// call to something that clobbers the whole register file would force
    /// a caller to save first.
    ///
    /// `shared = (X+0)+(X+1)+(X+2) = 3X+3`;
    /// `reduce = sum_{i=0}^{3}(shared + i) = 4*shared + 6`;
    /// `root = shared + reduce = 5*shared + 6`.
    #[test]
    fn a_surviving_reduces_own_recompute_of_a_shared_leaf_does_not_clobber_the_outer_copy() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let c0 = a.push_const(0.0);
        let t0 = a.push_binary(OpKind::Add, x, c0);
        let c1 = a.push_const(1.0);
        let t1 = a.push_binary(OpKind::Add, x, c1);
        let c2 = a.push_const(2.0);
        let t2 = a.push_binary(OpKind::Add, x, c2);
        let s01 = a.push_binary(OpKind::Add, t0, t1);
        let shared = a.push_binary(OpKind::Add, s01, t2);
        let binder = Binder::from_slot(0).expect("slot 0 exists");
        let i = a.push_var(binder.var());
        let body = a.push_binary(OpKind::Add, shared, i);
        let fold = Fold::new(Monoid::SUM, binder, 0..4);
        let reduce = a.push_reduce(fold, body);
        let root = a.push_binary(OpKind::Add, shared, reduce);

        let code = compile(&a, root, POINT).expect("interleaved shared-leaf order compiles");

        for x in [0.0f32, 2.0, -1.5, 10.0] {
            let got = eval_point(&code.code, x, 0.0);
            let shared_val = 3.0 * x + 3.0;
            let want = shared_val + (4.0 * shared_val + 6.0);
            assert_eq!(got, want, "at x={x}");
        }
    }

    /// A fold's spill slots do not alias its parent's.
    ///
    /// The frame used to be sized as a plain `max` over the nest's scopes,
    /// with every one of them handing out slots from offset 0. That is right
    /// for the two collapse prologues and the body — they run one after
    /// another, each parking what the next needs in a *hoist* slot above the
    /// frame, so an earlier scope's own slots are dead by the time a later one
    /// opens. A fold is the scope that is not like that: its loop runs in the
    /// middle of its parent's schedule, and the parent's spilled values are
    /// live across it. Sharing a base meant the fold's body wrote its own
    /// temporaries over them.
    ///
    /// It took real pressure on both sides to see: for a glyph, a 2,472-def
    /// fold body over a 2,130-def parent, which is every slot the parent had,
    /// and the whole atlas came out blank. So this builds that shape rather
    /// than a minimal fold — `K` values live across the loop and `K` more
    /// inside it, all defined before anything consumes them, which is what
    /// forces both scopes past the register file and into slots.
    ///
    /// `p_k = X + k`; `B(i) = Σ_j (i + j)·p_j`; `reduce = Σ_{i<R} B(i)`;
    /// `root = Σ_k (p_k + reduce)`.
    #[test]
    fn a_folds_spill_slots_do_not_alias_its_parents() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        const K: usize = 20;
        const R: u32 = 3;

        // A balanced sum, pushed level by level: every leaf is defined before
        // the first combine, so they are all live at once.
        fn tree_sum(a: &mut ExprArena, mut ids: alloc::vec::Vec<ExprId>) -> ExprId {
            while ids.len() > 1 {
                let mut next = alloc::vec::Vec::new();
                for pair in ids.chunks(2) {
                    next.push(match pair {
                        [l, r] => a.push_binary(OpKind::Add, *l, *r),
                        [only] => *only,
                        _ => unreachable!("chunks(2) yields 1 or 2"),
                    });
                }
                ids = next;
            }
            ids[0]
        }

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        // Live across the loop: defined here, consumed only past the `Reduce`.
        let p: alloc::vec::Vec<ExprId> = (0..K)
            .map(|k| {
                let c = a.push_const(k as f32);
                a.push_binary(OpKind::Add, x, c)
            })
            .collect();

        let binder = Binder::from_slot(0).expect("slot 0 exists");
        let i = a.push_var(binder.var());
        // Live inside the loop, and sharing every `p_j` with the parent.
        let q: alloc::vec::Vec<ExprId> = (0..K)
            .map(|j| {
                let c = a.push_const(j as f32);
                let ij = a.push_binary(OpKind::Add, i, c);
                a.push_binary(OpKind::Mul, ij, p[j])
            })
            .collect();
        let body = tree_sum(&mut a, q);
        let reduce = a.push_reduce(Fold::new(Monoid::SUM, binder, 0..R), body);

        let joined: alloc::vec::Vec<ExprId> = p
            .iter()
            .map(|&pk| a.push_binary(OpKind::Add, pk, reduce))
            .collect();
        let root = tree_sum(&mut a, joined);

        let code = compile(&a, root, POINT).expect("a fold under register pressure compiles");

        for xv in [0.0f32, 1.0, -2.5, 7.0] {
            let got = eval_point(&code.code, xv, 0.0);
            let pv = |k: usize| xv + k as f32;
            let reduce_v: f32 = (0..R)
                .map(|iv| (0..K).map(|j| (iv as f32 + j as f32) * pv(j)).sum::<f32>())
                .sum();
            let want: f32 = (0..K).map(|k| pv(k) + reduce_v).sum();
            // Summation order differs from the emitted tree's, so this is a
            // tolerance on rounding — the bug it guards was off by the whole
            // value, not the last bits.
            let tol = want.abs() * 1e-4 + 1e-3;
            assert!(
                (got - want).abs() <= tol,
                "at x={xv}: got {got}, want {want}"
            );
        }
    }

    /// A fold inside a fold's body is a loop inside a loop.
    ///
    /// `extract_folds` used to refuse this shape (one level, with
    /// `expand_nested_reduce` unrolling the inner fold ahead of it); it
    /// carves the inner fold out of the outer fold's closure now, the same
    /// way it carves the outer one out of the body. The inner body reads
    /// *both* binders — a contraction — which is what exercises the outer
    /// binder staying in its register across the inner loop, and the inner
    /// scope being handed the outer binder's park.
    ///
    /// `inner(j) = Σ_{i<3} (X + i·j) = 3X + 3j`;
    /// `root = Σ_{j<2} inner(j) = 6X + 3`.
    #[test]
    fn a_reduce_inside_a_reduce_compiles_and_runs() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let inner_binder = Binder::from_slot(0).expect("slot 0 exists");
        let outer_binder = Binder::from_slot(1).expect("slot 1 exists");
        let i = a.push_var(inner_binder.var());
        let j = a.push_var(outer_binder.var());
        let ij = a.push_binary(OpKind::Mul, i, j);
        let inner_body = a.push_binary(OpKind::Add, x, ij);
        let inner = a.push_reduce(Fold::new(Monoid::SUM, inner_binder, 0..3), inner_body);
        let root = a.push_reduce(Fold::new(Monoid::SUM, outer_binder, 0..2), inner);

        // With the whole pool and at the floor: the outer body never reads
        // its own binder, only the inner one does, so at the floor the inner
        // scope reads an enclosing loop's counter from that loop's slot.
        for code in [
            compile(&a, root, POINT).expect("a fold inside a fold compiles"),
            compile_at_floor(&a, root, POINT),
        ] {
            for x in [0.0f32, 2.0, -1.5, 10.0] {
                let got = eval_point(&code.code, x, 0.0);
                let want = 6.0 * x + 3.0;
                assert_eq!(got, want, "at x={x}");
            }
        }
    }

    /// Three deep, every level's binder read at the bottom, and the inner
    /// fold's result feeding further arithmetic in its parent's body rather
    /// than being the parent's whole body.
    ///
    /// `Σ_{k<2} (k + Σ_{j<2} (j + Σ_{i<2} (X + i + j + k)))`: the innermost
    /// sums to `2X + 1 + 2j + 2k`, the middle to `Σ_j (2X + 1 + 2k + 3j)` =
    /// `4X + 5 + 4k`, the outer to `Σ_k (4X + 5 + 5k)` = `8X + 15`.
    #[test]
    fn a_reduce_three_deep_compiles_and_runs() {
        let (a, root) = three_deep_contraction();
        let code = compile(&a, root, POINT).expect("a fold three deep compiles");
        assert_three_deep(&code);
    }

    /// The three-deep contraction of the test above, as an arena.
    fn three_deep_contraction() -> (ExprArena, ExprId) {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let bi = Binder::from_slot(0).expect("slot 0 exists");
        let bj = Binder::from_slot(1).expect("slot 1 exists");
        let bk = Binder::from_slot(2).expect("slot 2 exists");
        let i = a.push_var(bi.var());
        let j = a.push_var(bj.var());
        let k = a.push_var(bk.var());
        let xi = a.push_binary(OpKind::Add, x, i);
        let xij = a.push_binary(OpKind::Add, xi, j);
        let xijk = a.push_binary(OpKind::Add, xij, k);
        let inner = a.push_reduce(Fold::new(Monoid::SUM, bi, 0..2), xijk);
        let j_inner = a.push_binary(OpKind::Add, j, inner);
        let middle = a.push_reduce(Fold::new(Monoid::SUM, bj, 0..2), j_inner);
        let k_middle = a.push_binary(OpKind::Add, k, middle);
        let root = a.push_reduce(Fold::new(Monoid::SUM, bk, 0..2), k_middle);
        (a, root)
    }

    /// `8X + 15`, at four points.
    fn assert_three_deep(code: &CompileResult) {
        for x in [0.0f32, 2.0, -1.5, 10.0] {
            let got = eval_point(&code.code, x, 0.0);
            let want = 8.0 * x + 15.0;
            assert_eq!(got, want, "at x={x}");
        }
    }

    /// Two sibling folds binding the same slot read one `Var` node, and
    /// each reads its own counter through it.
    ///
    /// A glyph's distance and winding folds are this shape, and the e-graph
    /// hash-conses their binder leaf into one node. A binder slot keyed by
    /// that node would be one slot for two loops: the fold allocated last
    /// would own it, and the other would step its own counter but test and
    /// read the sibling's. At the floor neither binder is carried, so both
    /// are read from their slots.
    ///
    /// `Σ_{i<3} (X + i) + Σ_{i<2} 2i = (3X + 3) + 2 = 3X + 5`.
    #[test]
    fn sibling_folds_sharing_a_binder_node_read_their_own_counters() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let bi = Binder::from_slot(0).expect("slot 0 exists");
        let i = a.push_var(bi.var());
        let xi = a.push_binary(OpKind::Add, x, i);
        let left = a.push_reduce(Fold::new(Monoid::SUM, bi, 0..3), xi);
        let two = a.push_const(2.0);
        let two_i = a.push_binary(OpKind::Mul, two, i);
        let right = a.push_reduce(Fold::new(Monoid::SUM, bi, 0..2), two_i);
        let root = a.push_binary(OpKind::Add, left, right);

        for code in [
            compile(&a, root, POINT).expect("sibling folds compile"),
            compile_at_floor(&a, root, POINT),
        ] {
            for x in [0.0f32, 2.0, -1.5, 10.0] {
                let got = eval_point(&code.code, x, 0.0);
                let want = 3.0 * x + 5.0;
                assert_eq!(got, want, "at x={x}");
            }
        }
    }

    /// A fold invariant across the lattice is the **body's** own fold — it
    /// opens once per call, outside the lattice's row, column and lane folds
    /// — and its result reaches the scopes inside from wherever its loop left
    /// it.
    ///
    /// The `Reduce` arm ends its def early, past the hand-off, so a carried
    /// fold result used to reach the scopes inside in a register nothing had
    /// loaded. A fold result is never a root now (`stays_put`), so nothing
    /// carries one: the scopes inside read it from its accumulator slot.
    /// `X + Σ_{i<3} 2i = X + 6`.
    #[test]
    fn a_lattice_invariant_fold_is_the_bodys_own() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let bi = Binder::from_slot(0).expect("slot 0 exists");
        let i = a.push_var(bi.var());
        let two = a.push_const(2.0);
        let two_i = a.push_binary(OpKind::Mul, two, i);
        let sum = a.push_reduce(Fold::new(Monoid::SUM, bi, 0..3), two_i);
        let root = a.push_binary(OpKind::Add, x, sum);

        let file = native_file();
        let nest = allocate_nest(native_schedule(&a, root, POINT), &file);
        let j = kernel_fold(&nest, 3).expect("the kernel's fold is three trips");
        assert_eq!(
            nest.fold_parent(j),
            regalloc::Scope::Body,
            "a fold reading no coordinate belongs to the scope that runs once \
             per call, not to a lattice fold that reruns it per sample"
        );

        let code = compile(&a, root, POINT).expect("a hoisted fold compiles");
        for x in [0.0f32, 2.0, -1.5, 10.0] {
            let got = eval_point(&code.code, x, 0.0);
            assert_eq!(got, x + 6.0, "at x={x}");
        }
    }

    /// Which fold of `nest` runs `trips` times — the one a test built, as
    /// opposed to the lattice's own row, column and lane folds, whose trip
    /// counts come from the shape.
    fn kernel_fold(nest: &regalloc::NestAllocation, trips: u32) -> Option<usize> {
        (0..nest.fold_count()).find(|&j| {
            let vid = nest.fold_reduce_vid(j);
            nest.scope(nest.fold_parent(j)).schedule().iter().any(|d| {
                d.value == vid && matches!(d.op, ScheduledOp::Reduce(f, _) if f.len() == trips)
            })
        })
    }

    /// A fold's binder and accumulator are roots the allocator places under
    /// its budget, not registers reserved across the body.
    ///
    /// Two pools, the same three-deep contraction. At the floor
    /// (`MIN_SCRATCH`) the budget is zero at every depth, so every fold's
    /// roots go to slots and the loops step and test them from memory —
    /// the case a reserved binder could not serve past one level, since each
    /// level took a register from the pool below until an instruction there
    /// had nowhere to put its scratch. With headroom the ranking spends it,
    /// deepest loop first, because that is what a carry saves most per call.
    /// Both compile and run.
    #[test]
    fn a_folds_roots_are_placed_by_the_budget() {
        let (a, root) = three_deep_contraction();
        let carried = |file: &regalloc::RegisterFile| -> usize {
            let nest = allocate_nest(native_schedule(&a, root, POINT), file);
            (0..nest.fold_count())
                .flat_map(|j| regalloc::tests::fold_roots(&nest, j))
                .filter(|at| matches!(at, regalloc::Where::Reg(_)))
                .count()
        };
        let floor = regalloc::tests::FLOOR;
        let mut previous = None;
        for above in [0, 5] {
            let file = regalloc::tests::at_floor(native_file(), above);
            assert_eq!(
                file.scratch.len(),
                floor + above,
                "the pool did not cap where the test expects"
            );
            let count = carried(&file);
            if above == 0 {
                assert_eq!(
                    count, 0,
                    "at the floor the budget is zero, so every fold root is in a slot"
                );
            }
            if let Some(fewer) = previous {
                assert!(
                    count > fewer,
                    "{above} registers above the floor carried {count} fold \
                     roots, no more than the smaller pool's {fewer}"
                );
            }
            previous = Some(count);
            assert_three_deep(&compile_above_floor(&a, root, POINT, above));
        }
    }

    /// A fold inside a fold's body that does not depend on the outer binder
    /// is the outer scope's fold, run once, and read inside from its slot —
    /// not a loop per iteration. Both readings of it here: the outer body
    /// reads it under its own binder, and the root reads it again outside.
    ///
    /// `inner = Σ_{i<3} (X + i) = 3X + 3`; `outer = Σ_{j<2} (inner + j) =
    /// 2·inner + 1`; `root = outer + inner = 3·inner + 1 = 9X + 10`.
    #[test]
    fn an_invariant_reduce_inside_a_reduce_is_hoisted_and_shared() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let bi = Binder::from_slot(0).expect("slot 0 exists");
        let bj = Binder::from_slot(1).expect("slot 1 exists");
        let i = a.push_var(bi.var());
        let xi = a.push_binary(OpKind::Add, x, i);
        let inner = a.push_reduce(Fold::new(Monoid::SUM, bi, 0..3), xi);
        let j = a.push_var(bj.var());
        let inner_j = a.push_binary(OpKind::Add, inner, j);
        let outer = a.push_reduce(Fold::new(Monoid::SUM, bj, 0..2), inner_j);
        let root = a.push_binary(OpKind::Add, outer, inner);

        // The structure: two folds, neither inside the other — the inner's
        // def sits in the outer's schedule as a placeholder, opening nothing.
        let file = native_file();
        let nest = allocate_nest(native_schedule(&a, root, POINT), &file);
        let inner_j = kernel_fold(&nest, 3).expect("the inner fold is three trips");
        let outer_j = kernel_fold(&nest, 2).expect("the outer fold is two trips");
        let inner_vid = nest.fold_reduce_vid(inner_j);
        assert_eq!(
            nest.fold_parent(inner_j),
            nest.fold_parent(outer_j),
            "both folds belong to the same scope; the inner one is not run \
             once per trip of the outer"
        );
        let outer_scope = nest.scope(regalloc::Scope::Fold(outer_j));
        let placeholder = outer_scope
            .schedule()
            .iter()
            .find(|d| d.value == inner_vid)
            .expect("the inner fold's def stays in the outer body as a placeholder");
        let ScheduledOp::Reduce(_, inner_body) = placeholder.op else {
            panic!("a fold's def is a Reduce")
        };
        assert!(
            !outer_scope.schedule().iter().any(|d| d.value == inner_body),
            "nothing behind the placeholder is copied in"
        );

        let code = compile(&a, root, POINT).expect("a hoisted fold compiles");
        for x in [0.0f32, 2.0, -1.5, 10.0] {
            let got = eval_point(&code.code, x, 0.0);
            let want = 9.0 * x + 10.0;
            assert_eq!(got, want, "at x={x}");
        }
    }

    /// The same hoist when the invariant fold *is* the outer body: the
    /// outer's schedule is one placeholder, and its result comes from the
    /// slot. `Σ_{j<2} Σ_{i<3} (X + i) = 2·(3X + 3) = 6X + 6`.
    #[test]
    fn an_invariant_reduce_as_a_folds_whole_body_is_hoisted() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let bi = Binder::from_slot(0).expect("slot 0 exists");
        let bj = Binder::from_slot(1).expect("slot 1 exists");
        let i = a.push_var(bi.var());
        let xi = a.push_binary(OpKind::Add, x, i);
        let inner = a.push_reduce(Fold::new(Monoid::SUM, bi, 0..3), xi);
        let root = a.push_reduce(Fold::new(Monoid::SUM, bj, 0..2), inner);

        let code = compile(&a, root, POINT).expect("a fold whose body is a hoisted fold compiles");
        for x in [0.0f32, 2.0, -1.5, 10.0] {
            let got = eval_point(&code.code, x, 0.0);
            let want = 6.0 * x + 6.0;
            assert_eq!(got, want, "at x={x}");
        }
    }

    /// `a_folds_spill_slots_do_not_alias_its_parents`, one level deeper: the
    /// outer fold's body holds `K` values live across the inner loop, and the
    /// inner loop holds `K` more, so both scopes go to slots and the inner
    /// fold's frame is based at the outer fold's top rather than at its
    /// parent's. The outer binder is read on both sides of the inner loop.
    ///
    /// `p_k = X + j + k` (in the outer body); `inner(j) = Σ_{i<R} Σ_k (i +
    /// p_k)`; `outer body = Σ_k (p_k + inner(j))`; `root = Σ_{j<J} outer`.
    #[test]
    fn a_nested_folds_spill_slots_do_not_alias_its_parents() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        const K: usize = 20;
        const R: u32 = 3;
        const J: u32 = 2;

        fn tree_sum(a: &mut ExprArena, mut ids: alloc::vec::Vec<ExprId>) -> ExprId {
            while ids.len() > 1 {
                let mut next = alloc::vec::Vec::new();
                for pair in ids.chunks(2) {
                    next.push(match pair {
                        [l, r] => a.push_binary(OpKind::Add, *l, *r),
                        [only] => *only,
                        _ => unreachable!("chunks(2) yields 1 or 2"),
                    });
                }
                ids = next;
            }
            ids[0]
        }

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let bi = Binder::from_slot(0).expect("slot 0 exists");
        let bj = Binder::from_slot(1).expect("slot 1 exists");
        let i = a.push_var(bi.var());
        let j = a.push_var(bj.var());
        let xj = a.push_binary(OpKind::Add, x, j);
        // Live across the inner loop: defined in the outer body, consumed
        // only past the inner `Reduce`.
        let p: alloc::vec::Vec<ExprId> = (0..K)
            .map(|k| {
                let c = a.push_const(k as f32);
                a.push_binary(OpKind::Add, xj, c)
            })
            .collect();
        let q: alloc::vec::Vec<ExprId> = (0..K)
            .map(|k| a.push_binary(OpKind::Add, i, p[k]))
            .collect();
        let inner_body = tree_sum(&mut a, q);
        let inner = a.push_reduce(Fold::new(Monoid::SUM, bi, 0..R), inner_body);
        let joined: alloc::vec::Vec<ExprId> = p
            .iter()
            .map(|&pk| a.push_binary(OpKind::Add, pk, inner))
            .collect();
        let outer_body = tree_sum(&mut a, joined);
        let root = a.push_reduce(Fold::new(Monoid::SUM, bj, 0..J), outer_body);

        let code = compile(&a, root, POINT).expect("nested folds under register pressure compile");

        for xv in [0.0f32, 1.0, -2.5, 7.0] {
            let got = eval_point(&code.code, xv, 0.0);
            let want: f32 = (0..J)
                .map(|jv| {
                    let pv = |k: usize| xv + jv as f32 + k as f32;
                    let inner_v: f32 = (0..R)
                        .map(|iv| (0..K).map(|k| iv as f32 + pv(k)).sum::<f32>())
                        .sum();
                    (0..K).map(|k| pv(k) + inner_v).sum::<f32>()
                })
                .sum();
            let tol = want.abs() * 1e-4 + 1e-3;
            assert!(
                (got - want).abs() <= tol,
                "at x={xv}: got {got}, want {want}"
            );
        }
    }

    // =========================================================================
    // Cross-host emission: every backend, from whatever host runs the tests
    // =========================================================================

    /// Every backend emits from every host.
    ///
    /// This is the property the ISA files buy. Emission is a pure function of
    /// `(schedule, RegisterFile)` into a `Vec<u8>`, so an x86 box computes NEON
    /// instruction words and an arm box computes AVX-512 ones. Only running
    /// them needs the matching CPU.
    ///
    /// Before the backends stopped being `#[cfg]`-gated into existence, three
    /// of these four could not even be *named* here.
    ///
    /// The arena reads a uniform, so each backend's `ResolvedOp::Uniform`
    /// dispatch arm — not only the encoder behind it — is what emits here.
    #[test]
    fn every_backend_emits_from_this_host() {
        use pixelflow_ir::arena::{UniformDecl, UniformIdentity};
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let u = a.declare_uniform(UniformDecl {
            id: UniformIdentity::mint(),
            default: 1.0,
        });
        let u = a.push_uniform(u);
        let scaled = a.push_binary(OpKind::Mul, y, u);
        let root = a.push_binary(OpKind::Add, a.clone().push_var(0).max(x).min(x), scaled);

        let mut neon = aarch64::driver::Aarch64Backend::new();
        let mut avx2b = avx2::driver::Avx2Backend::new();
        let mut avx512b = avx512::driver::Avx512Backend::new();

        // Each backend is handed a schedule legalized at *its own* lane
        // count: the lattice's lane fold is the one part of a collapse whose
        // shape belongs to the target.
        let for_backend = |file: regalloc::RegisterFile| {
            schedule_for(&a, root, POINT, file.vector_bytes / BYTES_PER_LANE)
        };
        assert!(
            for_backend(neon.register_file())
                .iter()
                .any(|d| matches!(d.op, ScheduledOp::Uniform(..))),
            "the schedule must carry the uniform load for the backends to dispatch on"
        );

        let neon_len = compile_schedule(for_backend(neon.register_file()), &mut neon)
            .expect("NEON emit")
            .code
            .as_bytes()
            .len();
        assert!(
            neon_len > 0 && neon_len.is_multiple_of(4),
            "aarch64 is fixed-width"
        );
        for (name, len) in [
            (
                "AVX2",
                compile_schedule(for_backend(avx2b.register_file()), &mut avx2b)
                    .expect("AVX2")
                    .code
                    .as_bytes()
                    .len(),
            ),
            (
                "AVX-512",
                compile_schedule(for_backend(avx512b.register_file()), &mut avx512b)
                    .expect("AVX-512")
                    .code
                    .as_bytes()
                    .len(),
            ),
        ] {
            assert!(len > 0, "{name} emitted nothing");
        }
    }

    /// Every x86 program leaves through `vzeroupper; ret`, on both tiers.
    ///
    /// The return is found from the driver's structure, not by scanning for
    /// `C3`, which a ModRM byte, an immediate or a pool entry holds just as
    /// well. A program has one return — [`compile_via_backend`] emits it
    /// after releasing the frame, and [`IsaBackend::emit_ret`] is the only
    /// verb that emits one — and every byte before its end is counted by a
    /// scope or by the scaffold, so the return ends where those counts do.
    ///
    /// The return grew in front of the constant pool, whose position two
    /// labels carry: the anchor's displacement and the pool's padding. So the
    /// pool is checked too, found the same way — through the anchor, the
    /// instruction after the frame, whose displacement the label pass
    /// resolved — and must sit at the first aligned byte at or after the
    /// return when it holds anything, at the return's end when it does not,
    /// across padding that is all zeros.
    #[test]
    fn every_x86_return_clears_the_upper_halves_first() {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};

        /// `VZEROUPPER` (`VEX.128.0F.WIG 77`) then `RET` (`C3`), as the SDM
        /// spells them rather than as the encoder under test does.
        const CLEAN_RETURN: [u8; 4] = [0xC5, 0xF8, 0x77, 0xC3];
        /// Bytes in an x86 pool entry: one `f32`'s bits.
        const POOL_ENTRY: usize = 4;
        /// A RIP-relative displacement is its instruction's last four bytes
        /// when no immediate follows it, and none follows one in `lea`.
        const REL32: usize = 4;
        /// Three rows, and a width that leaves a remainder on either tier's
        /// batch of 8 or 16 lanes, so every fold of the lattice emits.
        const PLANE: LatticeShape = LatticeShape::new([37, 3]);

        /// A kernel to compile, by name.
        type Case<'a> = (&'static str, &'a ExprArena, ExprId);

        fn check<B: IsaBackend>(tier: &str, fresh: impl Fn() -> B, kernels: &[Case<'_>]) {
            // The anchor follows the frame's allocation, each as the backend
            // emits them. The frame's size is an imm32 whatever its value, so
            // an empty frame measures the same bytes.
            let mut prologue = Assembly::default();
            let mut probe = fresh();
            probe.frame_alloc(&mut prologue.code, 0);
            let frame_end = prologue.len();
            let pool = prologue.mint();
            probe.anchor(&mut prologue, pool);
            let anchor_end = prologue.len();
            prologue.bind(pool);
            let prologue = prologue.finish();
            let lea = frame_end..anchor_end - REL32;

            for &(name, arena, root) in kernels {
                let mut backend = fresh();
                let lanes = backend.register_file().vector_bytes / BYTES_PER_LANE;
                let result =
                    compile_schedule(schedule_for(arena, root, PLANE, lanes), &mut backend)
                        .unwrap_or_else(|e| panic!("{tier}/{name}: {e:?}"));
                let code = result.code.as_bytes();
                // Every byte up to the return is a scope's or the scaffold's,
                // so what they leave uncounted is what trails the return.
                let counted: u64 = result.traffic.scopes.iter().map(|s| s.bytes).sum::<u64>()
                    + result.traffic.scaffold.bytes;
                let trailing = code.len() - counted as usize;

                let ret_end = code.len() - trailing;
                assert_eq!(
                    code[ret_end - CLEAN_RETURN.len()..ret_end],
                    CLEAN_RETURN,
                    "{tier}/{name}: the return is not `vzeroupper; ret`"
                );

                assert_eq!(
                    code[lea.clone()],
                    prologue[lea.clone()],
                    "{tier}/{name}: the anchor is not where the frame ends"
                );
                let disp = i32::from_le_bytes(
                    code[anchor_end - REL32..anchor_end]
                        .try_into()
                        .expect("a rel32 is four bytes"),
                );
                let pool = anchor_end
                    .checked_add_signed(disp as isize)
                    .unwrap_or_else(|| panic!("{tier}/{name}: the anchor points before the code"));
                // Padding exists only in front of entries, so a pool that
                // trails anything is aligned, and one that trails nothing is
                // bound where the return ends.
                let expected = match trailing {
                    0 => ret_end,
                    _ => ret_end.next_multiple_of(CONST_POOL_ALIGN),
                };
                assert_eq!(pool, expected, "{tier}/{name}: the anchor misses the pool");
                assert!(
                    code[ret_end..pool].iter().all(|&b| b == 0),
                    "{tier}/{name}: the pool's padding is not zeros"
                );
                assert_eq!(
                    (code.len() - pool) % POOL_ENTRY,
                    0,
                    "{tier}/{name}: the pool is not whole entries"
                );
            }
        }

        // Nothing but coordinates: the least a program is.
        let mut plain = ExprArena::new();
        let (x, y) = (plain.push_var(0), plain.push_var(1));
        let plain_root = plain.push_binary(OpKind::Add, x, y);

        // Constants, and an `If` on a comparison: a pool to pad.
        let mut if_arena = ExprArena::new();
        let (x, y) = (if_arena.push_var(0), if_arena.push_var(1));
        let edge = if_arena.push_const(2.5);
        let scale = if_arena.push_const(3.7);
        let bias = if_arena.push_const(0.25);
        let cond = if_arena.push_binary(OpKind::Lt, x, edge);
        let scaled = if_arena.push_binary(OpKind::Mul, x, scale);
        let biased = if_arena.push_binary(OpKind::Add, y, bias);
        let if_root = if_arena.push_ternary(OpKind::If, cond, scaled, biased);

        // A surviving fold: a loop of the kernel's own inside the lattice's.
        let binder = Binder::from_slot(0).expect("slot 0 exists");
        let mut fold = ExprArena::new();
        let x = fold.push_var(0);
        let i = fold.push_var(binder.var());
        let body = fold.push_binary(OpKind::Add, x, i);
        let fold_root = fold.push_reduce(Fold::new(Monoid::SUM, binder, 0..4), body);

        let kernels = [
            ("plain", &plain, plain_root),
            ("if", &if_arena, if_root),
            ("fold", &fold, fold_root),
        ];
        check("AVX2", avx2::driver::Avx2Backend::new, &kernels);
        check("AVX-512", avx512::driver::Avx512Backend::new, &kernels);
    }

    // =========================================================================
    // What the nest does and does not partition
    // =========================================================================

    /// A value an enclosing scope parked is addressed, in every scope that
    /// reads it, at that scope's park: the allocator's table says so, and
    /// the emitter asks nothing else.
    #[test]
    fn a_parked_placeholder_is_addressed_at_its_park() {
        let (a, root) = shared_leaf_kernel();
        let file = native_file();
        let nest = allocate_nest(native_schedule(&a, root, batch()), &file);
        let mut placeholders = 0;
        for j in 0..nest.fold_count() {
            let view = nest.scope(regalloc::Scope::Fold(j));
            for def in view.schedule() {
                if !view.parked_by_an_enclosing_scope(def.value) {
                    continue;
                }
                placeholders += 1;
                let mut parking = view;
                let park = loop {
                    let (parent, _) = parking
                        .opens_at()
                        .expect("an enclosing scope parks the value");
                    parking = parking.sibling(parent);
                    if let Some(park) = parking.park(def.value) {
                        break park;
                    }
                };
                assert_eq!(
                    view.slot_of(def.value),
                    Some(Slot::new(park)),
                    "Fold({j}) addresses {:?} somewhere other than its park",
                    def.value
                );
            }
        }
        assert!(placeholders > 0, "the fixture's folds read no park at all");
    }

    /// The placement is total over every scope's schedule, a parked
    /// placeholder's entry included — which reads the park, the enclosing
    /// scope's answer, rather than a range of this scope's own.
    #[test]
    fn a_shared_leaf_is_placed_once_per_scope() {
        let (a, root) = shared_leaf_kernel();
        let file = native_file();
        let nest = allocate_nest(native_schedule(&a, root, batch()), &file);

        // Every value a scope schedules has an answer at a point in that
        // scope — which is exactly what a single answer per value could not
        // give.
        let mut answered = 0;
        let mut scheduled = 0;
        let scopes = core::iter::once(regalloc::Scope::Body)
            .chain((0..nest.fold_count()).map(regalloc::Scope::Fold));
        for scope in scopes {
            let view = nest.scope(scope);
            for (i, def) in view.schedule().iter().enumerate() {
                scheduled += 1;
                if matches!(
                    view.where_at(def.value, i),
                    regalloc::Where::Reg(_)
                        | regalloc::Where::Ptr(_)
                        | regalloc::Where::Spilled
                        | regalloc::Where::Remat(_)
                ) {
                    answered += 1;
                }
            }
        }
        assert!(scheduled > 0);
        assert_eq!(
            answered, scheduled,
            "the nest-wide map is total over every scope's schedule"
        );
    }

    /// `y·k + x·k`: one constant, read by a row-invariant term and a
    /// column-varying one, so the inner scope needs a value the outer one
    /// computes.
    fn shared_leaf_kernel() -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let k = a.push_const(3.5);
        let invariant = a.push_binary(OpKind::Mul, y, k);
        let varying = a.push_binary(OpKind::Mul, x, k);
        let root = a.push_binary(OpKind::Add, invariant, varying);
        (a, root)
    }

    // =========================================================================
    // resolve_operands unit tests — the spill logic that was buggy
    // =========================================================================

    /// The two reload registers these `resolve_operands` tests hand the
    /// instruction, standing in for the allocator's per-instruction
    /// reservations. Every register an instruction may use is handed to it
    /// in `TEST_SCRATCH`, exactly as the allocator hands one its
    /// reservations.
    const RELOAD: [Reg; 2] = [Reg(11), Reg(12)];

    /// The scratch these `resolve_operands` tests are written against.
    const TEST_SCRATCH: regalloc::Scratch =
        regalloc::tests::scratch(None, [Some(RELOAD[0]), Some(RELOAD[1])]);

    /// Dense `ValueId -> Binding`, as the emit loop builds it.
    fn make_locs(
        assigned: &[(u32, u8)],
        spilled: &[(u32, u32)],
    ) -> alloc::vec::Vec<Option<Binding>> {
        let len = assigned
            .iter()
            .map(|&(v, _)| v)
            .chain(spilled.iter().map(|&(v, _)| v))
            .max()
            .map_or(0, |m| m as usize + 1);
        let mut locs = alloc::vec![None; len];
        for &(v, r) in assigned {
            locs[v as usize] = Some(Binding::Loc(Loc::Reg(Reg(r))));
        }
        for &(v, off) in spilled {
            locs[v as usize] = Some(Binding::Loc(Loc::Slot(Slot::new(off))));
        }
        locs
    }

    #[test]
    fn resolve_binary_no_spills() {
        // left=v4, right=v5, dst=v6 — all in registers
        let locs = make_locs(&[(0, 4), (1, 5), (2, 6)], &[]);
        let op = ScheduledOp::Binary(OpKind::Add, regalloc::ValueId(0), regalloc::ValueId(1));
        let plan = resolve_operands(
            &op,
            Binding::Loc(Loc::Reg(Reg(6))),
            locs.as_slice(),
            TEST_SCRATCH,
        );

        assert!(plan.reloads.is_empty());
        assert_eq!(
            plan.op,
            ResolvedOp::Binary {
                op: OpKind::Add,
                dst: Reg(6),
                left: Reg(4),
                right: Reg(5)
            }
        );
    }

    /// A spilled left operand goes straight to the destination.
    ///
    /// `dst op= right` consumes the left operand from `dst` anyway, so this
    /// costs no reservation at all — which is why a binary never needs two,
    /// however many of its operands are in memory.
    #[test]
    fn resolve_binary_left_spilled() {
        // left spilled at offset 0, right in v5
        let locs = make_locs(&[(1, 5), (2, 6)], &[(0, 0)]);
        let op = ScheduledOp::Binary(OpKind::Add, regalloc::ValueId(0), regalloc::ValueId(1));
        let plan = resolve_operands(
            &op,
            Binding::Loc(Loc::Reg(Reg(6))),
            locs.as_slice(),
            TEST_SCRATCH,
        );

        assert_eq!(plan.reloads.len(), 1);
        assert_eq!(
            plan.reloads[0],
            Reload::FromStack {
                target: Reg(6),
                slot: Slot::new(0),
            }
        );
        assert_eq!(
            plan.op,
            ResolvedOp::Binary {
                op: OpKind::Add,
                dst: Reg(6),
                left: Reg(6),
                right: Reg(5)
            }
        );
    }

    #[test]
    fn resolve_binary_both_spilled() {
        // Both spilled: left → dst (temp trick), right → tmp_op
        let locs = make_locs(&[(2, 6)], &[(0, 0), (1, 16)]);
        let op = ScheduledOp::Binary(OpKind::Mul, regalloc::ValueId(0), regalloc::ValueId(1));
        let plan = resolve_operands(
            &op,
            Binding::Loc(Loc::Reg(Reg(6))),
            locs.as_slice(),
            TEST_SCRATCH,
        );

        assert_eq!(plan.reloads.len(), 2);
        // left → dst (v6), right → tmp_op (v27)
        assert_eq!(
            plan.reloads[0],
            Reload::FromStack {
                target: Reg(6),
                slot: Slot::new(0),
            }
        );
        assert_eq!(
            plan.reloads[1],
            Reload::FromStack {
                target: RELOAD[0],
                slot: Slot::new(16),
            }
        );
        assert_eq!(
            plan.op,
            ResolvedOp::Binary {
                op: OpKind::Mul,
                dst: Reg(6),
                left: Reg(6),
                right: RELOAD[0]
            }
        );
    }

    /// A definition writes a register or nothing at all.
    ///
    /// The spilled destination was the whole job of `reload[0]`: a value that
    /// lost its register at its own definition was computed into a register
    /// outside the pool and stored from there. Every definition holds a pool
    /// register now, so a `Loc::Spill` destination is not a case to handle but
    /// an allocator that broke its contract — and this is where that shows up
    /// as a panic rather than as a register two values share.
    #[test]
    #[should_panic(expected = "a definition landed in stack slot")]
    fn a_spilled_destination_is_not_a_thing_the_allocator_can_produce() {
        let locs = make_locs(&[(0, 4), (1, 5)], &[(2, 32)]);
        let op = ScheduledOp::Binary(OpKind::Add, regalloc::ValueId(0), regalloc::ValueId(1));
        drop(resolve_operands(
            &op,
            Binding::Loc(Loc::Slot(Slot::new(32))),
            locs.as_slice(),
            TEST_SCRATCH,
        ));
    }

    /// A rematerialized constant's definition emits nothing.
    ///
    /// It lives nowhere and is rebuilt at each use, so computing it once into
    /// a register nobody reads is pure waste — which is what a fixed
    /// destination register made invisible.
    #[test]
    fn a_rematerialized_definition_emits_nothing() {
        let locs = make_locs(&[], &[]);
        let op = ScheduledOp::Const(1.5);
        let plan = resolve_operands(
            &op,
            Binding::Remat(1.5f32.to_bits()),
            locs.as_slice(),
            TEST_SCRATCH,
        );
        assert_eq!(plan.op, ResolvedOp::Nop);
        assert!(plan.reloads.is_empty());
    }

    #[test]
    fn resolve_muladd_fmla_path() {
        // a in reg, b in reg, c in reg → FMLA with setup_mov for c→dst
        let locs = make_locs(&[(0, 4), (1, 5), (2, 7), (3, 8)], &[]);
        let op = ScheduledOp::Ternary(
            OpKind::MulAdd,
            regalloc::ValueId(0),
            regalloc::ValueId(1),
            regalloc::ValueId(2),
        );
        let plan = resolve_operands(
            &op,
            Binding::Loc(Loc::Reg(Reg(8))),
            locs.as_slice(),
            TEST_SCRATCH,
        );

        assert!(plan.reloads.is_empty());
        // c=v7 ≠ dst=v8, so setup_mov should copy c → dst
        assert_eq!(plan.setup_mov, Some((Reg(8), Reg(7))));
        assert_eq!(
            plan.op,
            ResolvedOp::FusedMulAdd {
                dst: Reg(8),
                a: Reg(4),
                b: Reg(5)
            }
        );
    }

    #[test]
    fn resolve_var_is_nop() {
        let locs = make_locs(&[(0, 0)], &[]);
        let op = ScheduledOp::Var(0);
        let plan = resolve_operands(
            &op,
            Binding::Loc(Loc::Reg(Reg(0))),
            locs.as_slice(),
            TEST_SCRATCH,
        );
        assert_eq!(plan.op, ResolvedOp::Nop);
        assert!(plan.reloads.is_empty());
    }

    #[test]
    fn resolve_const() {
        let locs = make_locs(&[(0, 6)], &[]);
        let op = ScheduledOp::Const(core::f32::consts::PI);
        let plan = resolve_operands(
            &op,
            Binding::Loc(Loc::Reg(Reg(6))),
            locs.as_slice(),
            TEST_SCRATCH,
        );
        assert_eq!(
            plan.op,
            ResolvedOp::LoadConst {
                dst: Reg(6),
                val_bits: core::f32::consts::PI.to_bits()
            }
        );
    }

    // =========================================================================
    // Arena compilation tests
    // =========================================================================

    #[test]
    fn arena_compile_simple() {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let sum = arena.push_binary(OpKind::Add, x, y);

        let result = compile(&arena, sum, POINT).expect("arena DAG compile failed");
        // Two leaves and one add force nothing to memory: no scope of the
        // nest stores or reloads a value. Not `spill_count`, which counts the
        // frame's slots — the lattice's own folds reserve one the emitted
        // code never touches, on every backend.
        assert_eq!(result.traffic.dynamic_memory_ops(), 0);

        assert_eq!(eval_point(&result.code, 3.0, 4.0), 7.0);
    }

    #[test]
    fn arena_compile_with_constant() {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let two = arena.push_const(2.0);
        let y = arena.push_var(1);
        let prod = arena.push_binary(OpKind::Mul, x, two);
        let sum = arena.push_binary(OpKind::Add, prod, y);

        let result = compile(&arena, sum, POINT).expect("arena DAG compile failed");

        // 3*2 + 4 = 10
        assert_eq!(eval_point(&result.code, 3.0, 4.0), 10.0);
    }

    /// `Σᵢ (X+i)·(Y+i)` for i in 1..=10, summed as a balanced tree: ten
    /// products are live at once, so any pool must spill. Every leaf depends on
    /// X or Y — a uniform-only subtree would be loop-invariant, hoisted out
    /// of the collapse body, and leave nothing to spill.
    ///
    /// The pressure comes from the live ranges rather than from the budget.
    /// This used to be `(X+Y)·(X−Y) + (X·Y)·(X+1)` in a two-register pool,
    /// which keeps at most three values live: it spilled only because two
    /// registers is fewer than three, and no pool is that small any more
    /// (`RegisterFile::MIN_SCRATCH` — an instruction temp cannot spill). Ten
    /// live values outrun every backend's floor ([`AtFloor`]), so the subject
    /// here — what spilling *does* — no longer depends on how small the pool
    /// can be made.
    #[test]
    fn arena_compile_with_spills() {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let mut terms: alloc::vec::Vec<_> = (1..=10u32)
            .map(|i| {
                let c = arena.push_const(i as f32);
                let ax = arena.push_binary(OpKind::Add, x, c);
                let by = arena.push_binary(OpKind::Add, y, c);
                arena.push_binary(OpKind::Mul, ax, by)
            })
            .collect();
        while terms.len() > 1 {
            terms = terms
                .chunks(2)
                .map(|pair| match pair {
                    [l, r] => arena.push_binary(OpKind::Add, *l, *r),
                    _ => pair[0],
                })
                .collect();
        }
        let root = terms[0];

        let result = compile_at_floor(&arena, root, POINT);

        assert!(
            result.spill_count > 0,
            "expected spills under ten live terms"
        );

        // Σᵢ (3+i)·(4+i) = 20+30+42+56+72+90+110+132+156+182 = 890, every
        // term and partial sum exact in f32.
        assert_eq!(eval_point(&result.code, 3.0, 4.0), 890.0);
    }

    // =========================================================================
    // The shared driver's If short-circuit guard, on every backend that
    // has a JIT.
    //
    // `sched_if_guards` below covers this path on whichever tier the
    // host runs, and `avx512_if_guards` covers AVX-512 by name. aarch64
    // had no guard test at all, which mattered because
    // that is the one backend whose guard needs a scratch register: reducing a
    // mask with `UMAXV`/`UMINV` writes a scalar into a vector register, where
    // the x86 tiers use `movmskps`/`kortest` and the flags. So the register
    // that reduction destroys was, on aarch64 alone, an untested choice.
    //
    // These run wherever a backend exists, against the same expected values,
    // so no backend's guard can drift from another's.
    // =========================================================================
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    mod if_guard_driver {
        use super::*;

        /// Padding that makes an arm worth a branch, and what it adds.
        ///
        /// A guard is refused for an arm whose work costs less than the
        /// mispredict penalty it risks, which is right and which means a
        /// fixture's arms have to be arms worth guarding — a two-op arm is
        /// not one. Three adds of distinct constants are 12 latency-prior
        /// cycles, and every point below stays exact in `f32`.
        const PADDING: f32 = 6.0;

        fn worth_a_branch(a: &mut ExprArena, arm: ExprId) -> ExprId {
            (1..=3u32).fold(arm, |acc, i| {
                let c = a.push_const(i as f32);
                a.push_binary(OpKind::Add, acc, c)
            })
        }

        /// `(X > 0) ? B³ : 3B` (plus [`PADDING`] on each arm) over a shared
        /// `B = X·Y` — arms that are exclusive *and* contiguous in the
        /// schedule.
        ///
        /// Both properties are needed and the second is easy to lose: a guard
        /// skips a whole index range, so every index in it must belong to that
        /// arm. Giving each arm its own `Var` looks exclusive but is not
        /// contiguous — the `Var` is scheduled with the other leaves, far
        /// below the arm's body, and the range from there to the arm swallows
        /// the mask. Deriving both arms from one shared value keeps every leaf
        /// out of both arms, which is what leaves the arms' own nodes adjacent.
        fn guarded_if(a: &mut ExprArena) -> ExprId {
            let x = a.push_var(0);
            let y = a.push_var(1);
            let zero = a.push_const(0.0);
            let base = a.push_binary(OpKind::Mul, x, y);
            let cond = a.push_binary(OpKind::Gt, x, zero);
            let bb = a.push_binary(OpKind::Mul, base, base);
            let bbb = a.push_binary(OpKind::Mul, bb, base);
            let bbb = worth_a_branch(a, bbb);
            let b2 = a.push_binary(OpKind::Add, base, base);
            let b3 = a.push_binary(OpKind::Add, b2, base);
            let b3 = worth_a_branch(a, b3);
            let sel = a.push_ternary(OpKind::If, cond, bbb, b3);
            // Live ACROSS the `If` and read after it. Without something in
            // this role the `If` is the root, nothing downstream reads a
            // register, and a guard that clobbered a live one would still
            // produce the right answer — the test would be blind to exactly
            // the mistake it exists to catch.
            let carried = a.push_binary(OpKind::Sub, x, y);
            a.push_binary(OpKind::Add, sel, carried)
        }

        /// How many terms the filler below has.
        const FILLER: usize = 8;

        /// Coprime with [`FILLER`], so the pairing has no short cycle.
        fn pair(i: usize) -> usize {
            (i * 7 + 3) % FILLER
        }

        /// Filler that is live all at once whatever the evaluation order.
        ///
        /// [`FILLER`] terms off `seed`, multiplied in pairs by a permutation,
        /// so each is read twice with the others in between: no order keeps
        /// them all in registers, which is what makes the tests below about a
        /// *spilled* value rather than about an arithmetic identity. Defining
        /// them all before consuming any is not enough on its own —
        /// `passes::lattice::collapse` rebuilds the arena from the root, and
        /// the order it hands the scheduler is its own.
        fn filler(a: &mut ExprArena, seed: ExprId) -> ExprId {
            let terms: alloc::vec::Vec<ExprId> = (0..FILLER)
                .map(|i| {
                    let c = a.push_const(i as f32 + 1.0);
                    a.push_binary(OpKind::Add, seed, c)
                })
                .collect();
            let mut sum = a.push_const(0.0);
            for i in 0..FILLER {
                let product = a.push_binary(OpKind::Mul, terms[i], terms[pair(i)]);
                sum = a.push_binary(OpKind::Add, sum, product);
            }
            sum
        }

        /// [`filler`], in scalar `f32`.
        fn filler_value(seed: f32) -> f32 {
            let term = |i: usize| seed + i as f32 + 1.0;
            (0..FILLER).map(|i| term(i) * term(pair(i))).sum()
        }

        /// Assert a guard region actually formed for `root`.
        ///
        /// Without this the tests below still pass when the guard stops
        /// forming — they would just be testing an ordinary `If`, which is
        /// the silent-decay shape this file has been bitten by before.
        fn assert_guard_forms(a: &ExprArena, root: ExprId) {
            let file = native_file();
            let nest = allocate_nest(native_schedule(a, root, POINT), &file);
            assert!(
                guarded_scope(&nest).is_some(),
                "no If in this nest has an arm-exclusive range, so the \
                 short-circuit guard this test exists for is never emitted"
            );
        }

        /// The scope of an allocated nest whose schedule carries a guarded
        /// `If`, and that guard — every allocation question below is asked
        /// of the scope that actually branches.
        fn guarded_scope(
            nest: &regalloc::NestAllocation,
        ) -> Option<(regalloc::Allocation<'_>, IfGuard)> {
            let scopes = core::iter::once(regalloc::Scope::Body)
                .chain((0..nest.fold_count()).map(regalloc::Scope::Fold));
            scopes.map(|s| nest.scope(s)).find_map(|view| {
                view.if_guards()
                    .iter()
                    .find(|g| g.has_guarded_arm())
                    .map(|g| (view, g.clone()))
            })
        }

        /// An `If` whose true arm contains an `If`, with entries belonging
        /// to the root sitting inside both arms — so NEITHER level is
        /// guardable as scheduled, and both become guardable once the layout
        /// gathers each arm into one run.
        ///
        /// Nesting is the case that can go wrong quietly: an inner `If`'s
        /// arms lie inside an outer arm, so partitioning the outside moves the
        /// inside with it. If that broke an inner guard the kernel would still
        /// be correct and merely slower, which no value test would catch —
        /// hence the assertion on the analysis as well as on the arithmetic.
        ///
        /// The two "intruders" are read by the root, so they are shared with
        /// the world outside the arms and can never be skipped; they are what
        /// makes the arms non-contiguous to begin with.
        fn nested_guarded_ifs(a: &mut ExprArena) -> (ExprId, ExprId, ExprId) {
            let x = a.push_var(0);
            let y = a.push_var(1);
            let zero = a.push_const(0.0);
            let base = a.push_binary(OpKind::Mul, x, y);
            let outer_cond = a.push_binary(OpKind::Gt, x, zero);
            let inner_cond = a.push_binary(OpKind::Gt, y, zero);

            // Inner true arm, split around an entry the root reads.
            let t1 = a.push_binary(OpKind::Mul, base, base);
            let across_inner = a.push_binary(OpKind::Add, x, y);
            let t2 = a.push_binary(OpKind::Mul, t1, base);
            let one = a.push_const(1.0);
            let t3 = a.push_binary(OpKind::Add, t2, one);

            // Inner false arm.
            let f1 = a.push_binary(OpKind::Add, base, base);
            let two = a.push_const(2.0);
            let f2 = a.push_binary(OpKind::Add, f1, two);

            let (t3, f2) = (worth_a_branch(a, t3), worth_a_branch(a, f2));
            let inner = a.push_ternary(OpKind::If, inner_cond, t3, f2);

            // The rest of the outer true arm, split around a second one.
            let three = a.push_const(3.0);
            let o1 = a.push_binary(OpKind::Add, inner, three);
            let four = a.push_const(4.0);
            let across_outer = a.push_binary(OpKind::Mul, x, four);
            let five = a.push_const(5.0);
            let o2 = a.push_binary(OpKind::Mul, o1, five);

            // Outer false arm.
            let six = a.push_const(6.0);
            let p1 = a.push_binary(OpKind::Add, base, six);
            let seven = a.push_const(7.0);
            let p2 = a.push_binary(OpKind::Mul, p1, seven);

            let (o2, p2) = (worth_a_branch(a, o2), worth_a_branch(a, p2));
            let outer = a.push_ternary(OpKind::If, outer_cond, o2, p2);
            let carried = a.push_binary(OpKind::Add, across_inner, across_outer);
            let root = a.push_binary(OpKind::Add, outer, carried);
            (root, outer, inner)
        }

        /// What `nested_guarded_ifs` computes, in scalar `f32` and with no
        /// guard anywhere — every operation exact at the points below.
        fn nested_expected(x: f32, y: f32) -> f32 {
            let base = x * y;
            let inner = PADDING
                + if y > 0.0 {
                    base * base * base + 1.0
                } else {
                    base + base + 2.0
                };
            let outer = PADDING
                + if x > 0.0 {
                    (inner + 3.0) * 5.0
                } else {
                    (base + 6.0) * 7.0
                };
            outer + (x + y) + x * 4.0
        }

        /// The branches of the kernel at `root`, compiled the way `compile`
        /// compiles it, at one batch.
        fn census_of(a: &ExprArena, root: ExprId) -> (usize, usize, usize) {
            census(a, root, batch())
        }

        /// A kernel shaped like the chrome sphere keeps every branch it earns,
        /// and the sphere's silhouette alone earns none.
        ///
        /// Counts the guards the compile's own tables hold, through
        /// `Allocation::if_guards`: what the emitter branches on, and what no
        /// render can see, since a guarded `If` and a blended one produce the
        /// same pixels. A decision that moves a guard's arm out of reach is a
        /// slower kernel with the same picture, and this is where it shows.
        ///
        /// Three numbers, because an `If` count alone is a weak gate. The real
        /// chrome at 1920x1080 reads `(3, 6, 719)` under the scratch probe on
        /// AVX-512 and on AVX2: 3 of its 17 `If`s earn a guard, each over both
        /// arms. With the old clustering search switched off it still read 3
        /// guards, but 4 arms and 292 entries. This kernel is that shape in miniature
        /// — an `If` whose two arms each hold an `If`, with a value both
        /// worlds read first reached inside one of them — and reads the same
        /// three guards over six arms.
        ///
        /// The control is the silhouette mask over arms cheaper than the
        /// mispredict they would risk: one `If` and no guard. (The real sphere
        /// over sky read `(0, 0, 0)` before the layout chose the order and
        /// reads `(1, 1, 37)` since — its sky arm is costlier than this
        /// control's; docs/results/2026-10-03-guard-structure-baseline.md.)
        /// Counting `If`s would not tell it from the other; the arms' cost
        /// does.
        #[test]
        fn a_chrome_shaped_kernel_keeps_its_branches() {
            let mut a = ExprArena::new();
            let chrome = chrome_shaped(&mut a);
            assert_eq!(
                census_of(&a, chrome),
                (3, 6, 70),
                "the chrome-shaped kernel's branches moved"
            );

            let mut b = ExprArena::new();
            let silhouette = silhouette_shaped(&mut b);
            assert_eq!(
                census_of(&b, silhouette),
                (0, 0, 0),
                "arms under the mispredict bound earned a branch"
            );
        }

        fn bin(a: &mut ExprArena, op: OpKind, l: ExprId, r: ExprId) -> ExprId {
            a.push_binary(op, l, r)
        }

        /// The chrome sphere at the scale of one channel: `sphere.select(
        /// world(mirrored), world(ray))`, each `world(r) = floor.select(
        /// checker(r), sky(r))` (`pixelflow-graphics`'s `scene3d`, the scene
        /// `render::packed`'s code pins compile). Three `If`s: the sphere's,
        /// and one per world.
        ///
        /// Arithmetic only. A transcendental expands into `If`s of its own, so
        /// a census over one would move whenever an expansion did, which is
        /// not what it is there to say.
        fn chrome_shaped(a: &mut ExprArena) -> ExprId {
            let x = a.push_var(0);
            let y = a.push_var(1);
            let zero = a.push_const(0.0);
            let one = a.push_const(1.0);
            let two = a.push_const(2.0);

            // The sphere: the primary ray's discriminant, whose sign is the
            // silhouette, and the bounce the mirrored ray takes off it.
            let xx = bin(a, OpKind::Mul, x, x);
            let yy = bin(a, OpKind::Mul, y, y);
            let r2 = bin(a, OpKind::Add, xx, yy);
            let disc = bin(a, OpKind::Sub, one, r2);
            let hit = bin(a, OpKind::Gt, disc, zero);
            let bounce = bin(a, OpKind::Mul, disc, two);
            let bx = bin(a, OpKind::Mul, bounce, x);
            let by = bin(a, OpKind::Mul, bounce, y);
            let mx = bin(a, OpKind::Sub, x, bx);
            let my = bin(a, OpKind::Sub, y, by);

            // What both worlds read and neither owns: the horizon's tint, and
            // where the floor is.
            let tint = bin(a, OpKind::Mul, xx, yy);
            let floor = a.push_const(-0.5);

            let world = |a: &mut ExprArena, rx: ExprId, ry: ExprId| {
                let height = bin(a, OpKind::Mul, rx, ry);
                let on_floor = bin(a, OpKind::Lt, height, floor);
                let u = bin(a, OpKind::Mul, rx, two);
                let v = bin(a, OpKind::Mul, ry, two);
                let uu = bin(a, OpKind::Mul, u, u);
                let vv = bin(a, OpKind::Mul, v, v);
                let checker = bin(a, OpKind::Sub, uu, vv);
                let checker = bin(a, OpKind::Mul, checker, tint);
                let checker = bin(a, OpKind::Add, checker, u);
                let sky = bin(a, OpKind::Mul, ry, tint);
                let sky = bin(a, OpKind::Add, sky, vv);
                let sky = bin(a, OpKind::Mul, sky, sky);
                let (checker, sky) = (worth_a_branch(a, checker), worth_a_branch(a, sky));
                a.push_ternary(OpKind::If, on_floor, checker, sky)
            };
            let mirrored = world(a, mx, my);
            let direct = world(a, x, y);
            let sel = a.push_ternary(OpKind::If, hit, mirrored, direct);
            let carried = bin(a, OpKind::Sub, x, y);
            bin(a, OpKind::Add, sel, carried)
        }

        /// The sphere over the sky and nothing else: the same silhouette
        /// mask, but arms that are a constant and a few instructions —
        /// cheaper than the mispredict they would risk, so no guard anywhere.
        fn silhouette_shaped(a: &mut ExprArena) -> ExprId {
            let x = a.push_var(0);
            let y = a.push_var(1);
            let zero = a.push_const(0.0);
            let one = a.push_const(1.0);
            let xx = bin(a, OpKind::Mul, x, x);
            let yy = bin(a, OpKind::Mul, y, y);
            let r2 = bin(a, OpKind::Add, xx, yy);
            let disc = bin(a, OpKind::Sub, one, r2);
            let hit = bin(a, OpKind::Gt, disc, zero);
            let grey = a.push_const(0.5);
            let sky = bin(a, OpKind::Add, y, one);
            let sel = a.push_ternary(OpKind::If, hit, grey, sky);
            let carried = bin(a, OpKind::Sub, x, y);
            bin(a, OpKind::Add, sel, carried)
        }

        /// Trip count of [`a_fold_owned_by_an_arm_is_guarded`]'s fold.
        const ARM_FOLD_TRIPS: u32 = 64;

        /// `(X > 0) ? Σ_{j<64} |X − j| : 0`, plus a value carried across: an
        /// arm that is a loop and nothing else. All the scope holds of the
        /// loop is its `Reduce` def, which the latency table prices 0; the
        /// arm is priced as the loop it opens (`guards::FoldReads`), clears
        /// the mispredict bound, and is guarded — and the answer on the
        /// batch that skips the loop is the false arm's.
        #[test]
        fn a_fold_owned_by_an_arm_is_guarded() {
            use pixelflow_ir::fold::{Binder, Fold, Monoid};

            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let zero = a.push_const(0.0);
            let cond = a.push_binary(OpKind::Gt, x, zero);
            let binder = Binder::from_slot(0).expect("slot 0 exists");
            let j = a.push_var(binder.var());
            let diff = a.push_binary(OpKind::Sub, x, j);
            let term = a.push_unary(OpKind::Abs, diff);
            let fold = a.push_reduce(Fold::new(Monoid::SUM, binder, 0..ARM_FOLD_TRIPS), term);
            let sel = a.push_ternary(OpKind::If, cond, fold, zero);
            let carried = a.push_binary(OpKind::Sub, x, y);
            let root = a.push_binary(OpKind::Add, sel, carried);

            assert_guard_forms(&a, root);

            let point = compile(&a, root, POINT).expect("a guarded fold compiles");
            for &(px, py) in &[(3.0f32, 4.0f32), (-3.0, 4.0), (40.5, -2.0), (-0.5, 0.0)] {
                let arm = if px > 0.0 {
                    (0..ARM_FOLD_TRIPS).map(|j| (px - j as f32).abs()).sum()
                } else {
                    0.0
                };
                let want = arm + (px - py);
                let got = eval_point(&point.code, px, py);
                assert_eq!(got, want, "at ({px}, {py})");
            }
        }

        /// Both levels of a nested `If` are guarded once the layout chooses the
        /// order, and the order had to move for it: the arms are interleaved
        /// with entries the root reads, so as written neither is one run.
        #[test]
        fn layout_guards_both_levels_of_a_nested_if() {
            let mut a = ExprArena::new();
            let (root, _outer, _inner) = nested_guarded_ifs(&mut a);
            let schedule = native_schedule(&a, root, POINT);
            let written: Vec<regalloc::ValueId> = schedule.iter().map(|d| d.value).collect();
            let nest = allocate_nest(schedule, &native_file());
            // The scopes holding the `If`s: the column fold's, and its
            // remainder's where the lattice strip-mines it — each a copy.
            let mut guarded: Vec<Vec<IfGuard>> = Vec::new();
            let mut moved = false;
            for scope in core::iter::once(regalloc::Scope::Body)
                .chain((0..nest.fold_count()).map(regalloc::Scope::Fold))
            {
                let view = nest.scope(scope);
                let laid: Vec<regalloc::ValueId> =
                    view.schedule().iter().map(|d| d.value).collect();
                let as_written: Vec<regalloc::ValueId> = written
                    .iter()
                    .copied()
                    .filter(|v| laid.contains(v))
                    .collect();
                moved |= laid != as_written;
                if !view.if_guards().is_empty() {
                    guarded.push(view.if_guards().to_vec());
                }
            }
            assert!(
                moved,
                "the arms were already runs as written, which this fixture is not"
            );
            assert!(!guarded.is_empty(), "no scope earned a guard");
            for guards in guarded {
                assert_eq!(
                    guards.len(),
                    2,
                    "both the outer and the inner select must earn a guard, got {guards:?}"
                );
                assert!(
                    guards.iter().all(|g| g.has_guarded_arm()),
                    "a guard with an empty range is not a guard: {guards:?}"
                );
            }
        }

        /// The laid-out kernel's answer, against the same expression
        /// evaluated in scalar `f32` with no guards: uniform masks (which take
        /// the branches) and mixed lanes (which fall through to the blend),
        /// exactly equal — every operation here is exact at these points, so
        /// there is no tolerance to hide a wrong branch in.
        #[test]
        fn a_nested_guarded_if_agrees_lane_for_lane() {
            let mut a = ExprArena::new();
            let (root, _outer, _inner) = nested_guarded_ifs(&mut a);
            let point = compile(&a, root, POINT).expect("nested guarded If compile");

            // One point at a time: all four combinations of the two masks,
            // each of which takes a pair of branches.
            for &(x, y) in &[(3.0f32, 4.0f32), (3.0, -4.0), (-3.0, 4.0), (-3.0, -4.0)] {
                let got = eval_point(&point.code, x, y);
                assert_eq!(
                    got,
                    nested_expected(x, y),
                    "nested guarded If at ({x}, {y})"
                );
            }

            // Mixed lanes: the batch straddles `x = 0`, so the outer mask
            // varies by lane, its guard cannot fire and the blend has to
            // produce every lane. `y` is the row, so the inner mask is
            // uniform over a batch and takes its branch — both paths, in one
            // call.
            let batch = compile(&a, root, batch()).expect("nested guarded If compile");
            let x0 = -(lanes() as f32) / 2.0;
            for y in [4.0f32, -4.0] {
                let got = eval_batch(&batch.code, &[], &[], x0, y);
                for (lane, got) in got.iter().enumerate() {
                    assert_eq!(
                        *got,
                        nested_expected(x0 + lane as f32, y),
                        "lane {lane} of a mixed-mask batch at y={y}"
                    );
                }
            }
        }

        /// Uniform masks take the all-true and all-false branches; a mixed
        /// mask falls through to the blend. All three must agree with the
        /// arithmetic.
        #[test]
        fn a_guarded_if_takes_every_branch() {
            let mut a = ExprArena::new();
            let root = guarded_if(&mut a);
            assert_guard_forms(&a, root);

            let result = compile(&a, root, POINT).expect("guarded If compile");
            for &(x, y) in &[
                (3.0f32, 4.0f32), // all-true  -> B³
                (-2.0, 0.5),      // all-false -> 3B
                (0.5, -1.0),
                (-0.25, 2.0),
            ] {
                let b = x * y;
                let want = if x > 0.0 { b * b * b } else { 3.0 * b } + PADDING + (x - y);
                let got = eval_point(&result.code, x, y);
                assert!(
                    (got - want).abs() <= 1e-3,
                    "guarded If at ({x}, {y}): got {got}, want {want}"
                );
            }
        }

        /// The same, with the mask itself spilled.
        ///
        /// This is the case the guard's two reservations exist for: a spilled
        /// mask is resolved into `Scratch::guard_mask`, *both* guards then read
        /// it, and the reduction has to land somewhere else
        /// (`Scratch::guard_temp`). On aarch64 both used to be registers held
        /// out of every kernel's pool.
        ///
        /// Getting the mask to be the value that spills takes care, and the
        /// test asserts it rather than assuming: eviction is Belady, so the
        /// victim is whatever is used farthest out. The mask is read only at
        /// the `If`, and the [`filler`] between fills the pool — so the
        /// mask is the farthest-out live value there, and it is the one to
        /// go. A plain `spill_count > 0` would pass with the mask still
        /// resident and this path never taken.
        #[test]
        fn a_guarded_if_survives_a_spilled_mask() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let zero = a.push_const(0.0);

            // Read only at the very end: the farthest-out live value.
            let cond = a.push_binary(OpKind::Gt, x, zero);
            let mid = filler(&mut a, x);

            // Shared-base arms, as in `guarded_if`.
            let base = a.push_binary(OpKind::Mul, mid, y);
            let bb = a.push_binary(OpKind::Mul, base, base);
            let bbb = a.push_binary(OpKind::Mul, bb, base);
            let bbb = worth_a_branch(&mut a, bbb);
            let b2 = a.push_binary(OpKind::Add, base, base);
            let b3 = a.push_binary(OpKind::Add, b2, base);
            let b3 = worth_a_branch(&mut a, b3);
            let sel = a.push_ternary(OpKind::If, cond, bbb, b3);
            let carried = a.push_binary(OpKind::Sub, x, y);
            let root = a.push_binary(OpKind::Add, sel, carried);
            assert_guard_forms(&a, root);

            // The mask must actually be the value that spills.
            let file = regalloc::tests::at_floor(native_file(), 0);
            let nest = allocate_nest(native_schedule(&a, root, POINT), &file);
            let (view, guard) = guarded_scope(&nest).expect("a guard formed above");
            assert!(
                spills(view.placement(guard.mask_vid)),
                "the mask stayed in a register, so the spilled-mask path this \
                 test exists for is never reached"
            );

            let result = compile_at_floor(&a, root, POINT);

            for &(px, py) in &[(3.0f32, 2.0f32), (-2.0, 0.5), (0.5, -1.0)] {
                let b = filler_value(px) * py;
                let want = if px > 0.0 { b * b * b } else { 3.0 * b } + PADDING + (px - py);
                let got = eval_point(&result.code, px, py);
                assert!(
                    (got - want).abs() <= 1e-2 * want.abs().max(1.0),
                    "spilled guarded If at ({px}, {py}): got {got}, want {want}"
                );
            }
        }

        /// A value spilled before a guarded arm, brought back into a register
        /// *inside* it, and read again after it.
        ///
        /// This is the shape live-range splitting has to get right and the
        /// previous attempt did not: the arm is code a uniform mask skips, so
        /// a register range that begins at a read inside it names a register
        /// the skipped path never loaded. The rule is that such a range ends
        /// where the arm does — and the value's slot is valid throughout,
        /// because a value in memory anywhere is stored right after its
        /// definition, which is outside the arm.
        ///
        /// The fixture, so the two tests below assert on one shape rather
        /// than each rebuilding it.
        struct SplitFixture {
            arena: ExprArena,
            root: ExprId,
            /// The value that loses its register and is brought back inside
            /// the arm.
            split: regalloc::ValueId,
            /// The `If`'s true-arm range, in `scope`.
            arm: (usize, usize),
            nest: regalloc::NestAllocation,
            /// The scope holding the `If`.
            scope: regalloc::Scope,
        }

        fn split_across_a_guarded_arm() -> SplitFixture {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let zero = a.push_const(0.0);

            // `split` seeds the filler, so it is live across all of it and is
            // the farthest-out live value where the pool fills; its next read
            // after that is inside the arm. It wears an `Abs` so the schedule
            // can be searched for it by op: the coordinates are folds'
            // binders now, and `X·Y` is no longer a product of two `Var`s to
            // look for.
            let cond = a.push_binary(OpKind::Gt, x, zero);
            let xy = a.push_binary(OpKind::Mul, x, y);
            let split = a.push_unary(OpKind::Abs, xy);
            let mid = filler(&mut a, split);

            // Shared-base arms, so neither arm's leaves land outside it and
            // the arms' own nodes stay adjacent (see `guarded_if`).
            let base = a.push_binary(OpKind::Mul, mid, y);
            // The true arm reads `split` twice: one read would be reloaded
            // into a scratch and kept nowhere, which is not the case under
            // test.
            let t1 = a.push_binary(OpKind::Mul, base, split);
            let t2 = a.push_binary(OpKind::Add, t1, split);
            let t3 = a.push_binary(OpKind::Mul, t2, base);
            let t3 = worth_a_branch(&mut a, t3);
            let f1 = a.push_binary(OpKind::Add, base, base);
            let f2 = a.push_binary(OpKind::Add, f1, base);
            let f2 = worth_a_branch(&mut a, f2);
            let sel = a.push_ternary(OpKind::If, cond, t3, f2);
            // Read after the arm, which is what makes the confinement rule
            // load-bearing: on the skipped path this must not name the
            // register the arm would have loaded.
            let after = a.push_binary(OpKind::Add, sel, split);
            let carried = a.push_binary(OpKind::Sub, x, y);
            let root = a.push_binary(OpKind::Add, after, carried);

            let file = regalloc::tests::at_floor(native_file(), 0);
            let nest = allocate_nest(native_schedule(&a, root, POINT), &file);
            let mut scopes = core::iter::once(regalloc::Scope::Body)
                .chain((0..nest.fold_count()).map(regalloc::Scope::Fold));
            let (scope, guard) = scopes
                .find_map(|s| {
                    nest.scope(s)
                        .if_guards()
                        .iter()
                        .find(|g| {
                            let (start, end) = g.range(IfArm::True);
                            start != end
                        })
                        .map(|g| (s, g.clone()))
                })
                .expect("the true arm is exclusive and contiguous, so it is guarded");

            // Which `ValueId` the arena's `split` became: the one `Abs`.
            let split = nest
                .scope(scope)
                .schedule()
                .iter()
                .find(|d| matches!(d.op, ScheduledOp::Unary(OpKind::Abs, _)))
                .map(|d| d.value)
                .expect("|X·Y| is in the schedule");
            let arm = guard.range(IfArm::True);
            SplitFixture {
                arena: a,
                root,
                split,
                arm,
                nest,
                scope,
            }
        }

        /// The value is right after the arm, on the path that skips it.
        #[test]
        fn a_split_range_inside_a_guarded_arm_is_correct_when_the_arm_is_skipped() {
            let f = split_across_a_guarded_arm();
            let view = f.nest.scope(f.scope);
            assert!(
                spills(view.placement(f.split)),
                "the value under test stayed in a register, so nothing is split"
            );
            let kept = regalloc::tests::ranges(view.placement(f.split))
                .into_iter()
                .any(|(from, at)| matches!(at, regalloc::Where::Reg(_)) && from >= f.arm.0);
            assert!(
                kept,
                "the value was never brought back into a register inside the \
                 arm, so the confinement rule this test exists for is not exercised"
            );

            let result = compile_at_floor(&f.arena, f.root, POINT);
            // x < 0 is the all-false mask: the true arm — and the reload
            // inside it — never runs, and the read after it must still be the
            // value.
            for &(px, py) in &[(-2.0f32, 3.0f32), (-0.5, -4.0), (3.0, 2.0), (0.25, 1.5)] {
                let v = (px * py).abs();
                let b = filler_value(v) * py;
                let arm_value = if px > 0.0 {
                    (b * v + v) * b
                } else {
                    (b + b) + b
                };
                let want = arm_value + PADDING + v + (px - py);
                let got = eval_point(&result.code, px, py);
                assert!(
                    (got - want).abs() <= 1e-2 * want.abs().max(1.0),
                    "split across a guarded arm at ({px}, {py}): got {got}, want {want}"
                );
            }
        }

        /// And the range ends exactly where the arm does.
        ///
        /// One index later would be a register the skipped path never wrote;
        /// earlier is merely wasteful. The allocator gets the arm ranges from
        /// the same tables the emitter branches on, which is
        /// what makes "exactly" a statement about one answer rather than two.
        #[test]
        fn a_kept_reload_inside_a_guarded_arm_ends_at_the_arm() {
            let f = split_across_a_guarded_arm();
            let ranges = regalloc::tests::ranges(f.nest.scope(f.scope).placement(f.split));
            let kept = ranges
                .iter()
                .position(|&(from, at)| matches!(at, regalloc::Where::Reg(_)) && from >= f.arm.0)
                .expect("a register range begins inside the arm");
            assert!(
                ranges[kept].0 < f.arm.1,
                "the range begins outside the arm it was confined to"
            );
            let reverted = ranges
                .get(kept + 1)
                .expect("a confined range is followed by the range it reverts to");
            assert_eq!(
                reverted.0, f.arm.1,
                "a register range that begins inside a guarded arm must end \
                 where the arm does: a read after it would name a register the \
                 skipped path never loaded"
            );
            // And it reverts to *memory*, not to another register. A revert
            // places `(end, Spilled)`; if the kept-reload step then re-keeps
            // the value at that same index, `Pass::place` overwrites the
            // same-index range with `(end, Reg)` and the index check above
            // still passes — while the skipped path reads a register it never
            // loaded. This is the assertion that would have caught it.
            assert!(
                !matches!(reverted.1, regalloc::Where::Reg(_)),
                "the range after a confined one must be in memory, not a \
                 register the skipped path never wrote: {:?}",
                reverted.1
            );
        }
    }

    /// Run an arena kernel at `(x, 0)`, on whichever tier this host selected.
    /// The builtin-parity tests below use it.
    fn run1(arena: &ExprArena, root: ExprId, x: f32) -> f32 {
        run_xy(arena, root, x, 0.0)
    }

    /// Eval at `(x, y)`.
    fn run_xy(arena: &ExprArena, root: ExprId, x: f32, y: f32) -> f32 {
        let r = compile(arena, root, POINT).expect("compile failed");
        eval_point(&r.code, x, y)
    }

    /// A `Dwrt`-carrying arena must JIT-compile end-to-end: the compile entry
    /// runs `lower_dwrt`, so `D(√(x²+y²), x)` compiles to `x / √(x²+y²)`
    /// without the caller ever seeing the derivative machinery.
    #[test]
    fn dwrt_compiles_to_analytic_derivative() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let x2 = a.push_binary(OpKind::Mul, x, x);
        let y2 = a.push_binary(OpKind::Mul, y, y);
        let sum = a.push_binary(OpKind::Add, x2, y2);
        let dist = a.push_unary(OpKind::Sqrt, sum);
        let v0 = a.push_const(0.0);
        let root = a.push_binary(OpKind::Dwrt, dist, v0);

        for (px, py) in [(3.0f32, 4.0f32), (1.0, 1.0), (-2.0, 5.0)] {
            let got = run_xy(&a, root, px, py);
            let want = px / (px * px + py * py).sqrt();
            assert!(
                (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                "d/dx dist at ({px},{py}): got {got}, want {want}"
            );
        }
    }

    /// A `Dwrt` over an op with no derivative rule must surface as a compile
    /// error (loud refusal), not a miscompile or a scheduler panic.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dwrt_of_gather_refuses_to_compile() {
        use pixelflow_ir::arena::BufferDecl;
        let mut a = ExprArena::new();
        let buf = a.declare_buffer(BufferDecl {
            id: pixelflow_ir::arena::BufferIdentity::mint(),
            width: 2,
            height: 1,
        });
        let bufleaf = a.push_buffer(buf);
        let x = a.push_var(0);
        let y = a.push_var(1);
        let g = a.push_ternary(OpKind::Gather, bufleaf, x, y);
        let v0 = a.push_const(0.0);
        let root = a.push_binary(OpKind::Dwrt, g, v0);
        assert!(compile(&a, root, POINT).is_err());
    }

    /// A deep spill frame must compile and produce correct results — the
    /// glyph-scale-kernel case that used to refuse with "exceeds 128-byte red
    /// zone". There is no red zone any more, so what is under test is only
    /// that a frame dozens of slots deep is emitted and addressed correctly.
    ///
    /// Forty terms, **paired by a permutation** so that each is read twice,
    /// far apart: no evaluation order keeps them all in registers. Pushing
    /// them all before consuming any is no longer enough on its own, because
    /// `passes::lattice::collapse` rebuilds the arena from the root and the
    /// order it hands the scheduler is its own.
    #[test]
    fn a_deep_spill_frame_compiles_correctly() {
        const TERMS: usize = 40;
        /// Coprime with `TERMS`, so `i -> PAIR(i)` is a permutation with no
        /// short cycle: a term's two readers are far apart in every order.
        fn pair(i: usize) -> usize {
            (i * 7 + 3) % TERMS
        }

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let terms: alloc::vec::Vec<ExprId> = (0..TERMS)
            .map(|i| {
                let c = a.push_const(i as f32 + 1.0);
                let xc = a.push_binary(OpKind::Add, x, c);
                a.push_binary(OpKind::Mul, xc, y)
            })
            .collect();
        let mut root = a.push_const(0.0);
        for i in 0..TERMS {
            let product = a.push_binary(OpKind::Mul, terms[i], terms[pair(i)]);
            root = a.push_binary(OpKind::Add, root, product);
        }

        let result = compile(&a, root, POINT).expect("large spill frame must compile");
        assert!(
            result.spill_bytes > 128,
            "test did not force a deep frame (spill_bytes = {})",
            result.spill_bytes
        );

        for (px, py) in [(1.5f32, -2.0f32), (0.0, 0.0), (3.0, 4.0)] {
            let got = run_xy(&a, root, px, py);
            let term = |i: usize| (px + i as f32 + 1.0) * py;
            let want: f32 = (0..TERMS).map(|i| term(i) * term(pair(i))).sum();
            let tol = 1e-3 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tol,
                "at ({px},{py}): jit {got}, scalar {want}"
            );
        }
    }

    /// Every x86-64 unary transcendental/round op must match its scalar
    /// reference across a range of inputs — these exercise `emit_arena` →
    /// `emit_unary` directly (not the compiler's lowering).
    #[test]
    fn x86_unary_builtins_match_scalar() {
        // Tolerances reflect the shared (with aarch64) minimax-polynomial
        // accuracy over a sensible input range; exact ops use tight bounds.
        // `rel_err = |jit - scalar| / (1 + |scalar|)`.
        type UnaryCase<'a> = (OpKind, fn(f32) -> f32, &'a [f32], f32);
        let unary: &[UnaryCase] = &[
            (
                OpKind::Sqrt,
                |x| x.sqrt(),
                &[0.25, 1.0, 2.0, 9.0, 100.0],
                1e-5,
            ),
            (OpKind::Abs, |x| x.abs(), &[-3.0, -0.5, 0.0, 2.5], 1e-6),
            (OpKind::Neg, |x| -x, &[-3.0, 0.5, 2.5], 1e-6),
            (
                OpKind::Floor,
                |x| x.floor(),
                &[-2.3, -0.1, 0.9, 1.5, 3.99],
                1e-6,
            ),
            (
                OpKind::Ceil,
                |x| x.ceil(),
                &[-2.3, -0.1, 0.9, 1.5, 3.01],
                1e-6,
            ),
            (
                OpKind::Round,
                |x| x.round_ties_even(),
                &[-2.4, -0.4, 0.4, 1.5, 2.6],
                1e-6,
            ),
            // sin/cos: 4-term Chebyshev — accurate well inside [-π, π].
            (
                OpKind::Sin,
                |x| x.sin(),
                &[-2.0, -1.0, -0.3, 0.0, 0.5, 1.5, 2.0],
                6e-3,
            ),
            (
                OpKind::Cos,
                |x| x.cos(),
                &[-1.0, -0.3, 0.0, 0.5, 1.0],
                1.5e-2,
            ),
            (
                OpKind::Tan,
                |x| x.tan(),
                &[-1.0, -0.3, 0.0, 0.3, 1.0],
                2.5e-2,
            ),
            (
                OpKind::Exp,
                |x| x.exp(),
                &[-2.0, -0.5, 0.0, 1.0, 2.0, 3.0],
                5e-3,
            ),
            (
                OpKind::Exp2,
                |x| x.exp2(),
                &[-3.0, -0.5, 0.0, 1.0, 4.0],
                5e-3,
            ),
            (OpKind::Ln, |x| x.ln(), &[0.25, 0.5, 1.0, 2.0, 10.0], 5e-3),
            (
                OpKind::Log2,
                |x| x.log2(),
                &[0.25, 0.5, 1.0, 2.0, 8.0],
                5e-3,
            ),
            (
                OpKind::Log10,
                |x| x.log10(),
                &[0.1, 0.5, 1.0, 10.0, 100.0],
                5e-3,
            ),
            (
                OpKind::Atan,
                |x| x.atan(),
                &[-5.0, -0.5, -0.2, 0.0, 0.2, 0.5, 5.0],
                8e-3,
            ),
            (
                OpKind::Asin,
                |x| x.asin(),
                &[-0.8, -0.5, 0.0, 0.5, 0.8],
                1e-2,
            ),
            (
                OpKind::Acos,
                |x| x.acos(),
                &[-0.8, -0.5, 0.0, 0.5, 0.8],
                1e-2,
            ),
        ];
        for &(op, scalar, inputs, tol) in unary {
            let mut arena = ExprArena::new();
            let x = arena.push_var(0);
            let root = arena.push_unary(op, x);
            for &xv in inputs {
                let got = run1(&arena, root, xv);
                let want = scalar(xv);
                let err = (got - want).abs() / (1.0 + want.abs());
                assert!(
                    err <= tol,
                    "{op:?}({xv}): jit={got} scalar={want} rel_err={err} > {tol}"
                );
            }
        }
    }

    /// Binary transcendentals + comparisons + ternaries, JIT vs scalar.
    #[test]
    fn x86_binary_ternary_builtins_match_scalar() {
        // Helper: compile f(X, Y) and eval at (x, y).
        fn run2(arena: &ExprArena, root: ExprId, x: f32, y: f32) -> f32 {
            run_xy(arena, root, x, y)
        }

        // atan2(y, x): arena Binary(Atan2, Y, X)  (op order: src1=y, src2=x)
        let pts = [
            (0.5, 2.0),
            (2.0, 0.5),
            (-0.5, 2.0),
            (0.5, -2.0),
            (-2.0, -0.5),
            (3.0, -0.5),
        ];
        {
            let mut a = ExprArena::new();
            let y = a.push_var(1);
            let x = a.push_var(0);
            let root = a.push_binary(OpKind::Atan2, y, x);
            for &(yv, xv) in &pts {
                let got = run2(&a, root, xv, yv);
                let want = yv.atan2(xv);
                assert!(
                    (got - want).abs() <= 1.5e-2,
                    "atan2({yv},{xv}): {got} vs {want}"
                );
            }
        }
        // pow(X, Y)
        {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let root = a.push_binary(OpKind::Pow, x, y);
            for &(xv, yv) in &[(2.0f32, 3.0f32), (9.0, 0.5), (4.0, -1.0), (1.5, 2.0)] {
                let got = run2(&a, root, xv, yv);
                let want = xv.powf(yv);
                let err = (got - want).abs() / (1.0 + want.abs());
                assert!(err <= 5e-3, "pow({xv},{yv}): {got} vs {want} err={err}");
            }
        }
        // hypot(X, Y) — the sqrt(x² + y²) composition it denotes.
        {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let xx = a.push_binary(OpKind::Mul, x, x);
            let yy = a.push_binary(OpKind::Mul, y, y);
            let sum = a.push_binary(OpKind::Add, xx, yy);
            let root = a.push_unary(OpKind::Sqrt, sum);
            for &(xv, yv) in &[(3.0f32, 4.0f32), (1.0, 1.0), (0.0, 2.0)] {
                let got = run2(&a, root, xv, yv);
                let want = xv.hypot(yv);
                assert!(
                    (got - want).abs() <= 1e-4,
                    "hypot({xv},{yv}): {got} vs {want}"
                );
            }
        }
        // Min / Max
        for (op, f) in [
            (OpKind::Min, f32::min as fn(f32, f32) -> f32),
            (OpKind::Max, f32::max as fn(f32, f32) -> f32),
        ] {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let root = a.push_binary(op, x, y);
            for &(xv, yv) in &[(1.0f32, 2.0f32), (3.0, -1.0), (-2.0, -5.0)] {
                let got = run2(&a, root, xv, yv);
                assert!((got - f(xv, yv)).abs() <= 1e-6, "{op:?}({xv},{yv})");
            }
        }
        // clamp(X, 0.0, 1.0) — the min/max composition it denotes.
        {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let lo = a.push_const(0.0);
            let hi = a.push_const(1.0);
            let floored = a.push_binary(OpKind::Max, x, lo);
            let root = a.push_binary(OpKind::Min, floored, hi);
            for &xv in &[-0.5f32, 0.25, 0.9, 1.7] {
                let got = run1(&a, root, xv);
                assert!(
                    (got - xv.clamp(0.0, 1.0)).abs() <= 1e-6,
                    "clamp({xv})={got}"
                );
            }
        }
        // If(X >= 0, 1.0, -1.0) == signum-ish
        {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let zero = a.push_const(0.0);
            let cond = a.push_binary(OpKind::Ge, x, zero);
            let pos = a.push_const(1.0);
            let neg = a.push_const(-1.0);
            let root = a.push_ternary(OpKind::If, cond, pos, neg);
            for &xv in &[-2.0f32, -0.1, 0.1, 3.0] {
                let got = run1(&a, root, xv);
                let want = if xv >= 0.0 { 1.0 } else { -1.0 };
                assert!((got - want).abs() <= 1e-6, "select({xv})={got} want={want}");
            }
        }
    }

    // =========================================================================
    // Forward-mode dual (jet) lowering — validated against analytic derivatives.
    // Uses hardware sqrtps/divps (no polynomial approximations), so tolerances
    // are tight.
    /// Transcendental lowering: sin/cos/tan JIT through the shared driver with
    /// no backend ever emitting a transcendental (they expand to arithmetic in
    /// `lowering`). Validated against `f32` on whichever tier this host
    /// selected.
    mod lowering_tests {
        use super::*;
        use pixelflow_ir::arena::ExprArena;

        // The degree-11 Chebyshev in `passes` measures 6e-7 across the whole
        // reduced interval, so this bound sits an order of magnitude above the
        // measured worst case: tight enough to test the polynomial, loose
        // enough not to test the last bit of the build's rounding. A bound in
        // the 1e-2 range would only be able to catch gross logic errors.
        const TRIG_TOL: f32 = 1e-5;

        #[test]
        fn sin_cos_tan_match_scalar() {
            // Range beyond [-π,π] to exercise the floor-based range reduction.
            let pts = [0.0f32, 0.3, 1.0, 2.0, 3.5, -1.7, 6.0, -4.2];
            for &xv in &pts {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let s = a.push_unary(OpKind::Sin, x);
                assert!((run1(&a, s, xv) - xv.sin()).abs() <= TRIG_TOL, "sin({xv})");

                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let c = a.push_unary(OpKind::Cos, x);
                assert!((run1(&a, c, xv) - xv.cos()).abs() <= TRIG_TOL, "cos({xv})");
            }
            // tan away from its poles (ratio of two ~3e-3 approximations).
            for &xv in &[0.0f32, 0.3, 0.7, -0.5, 1.0] {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let t = a.push_unary(OpKind::Tan, x);
                // tan = sin/cos amplifies both errors by 1/cos²(x); from
                // 6e-7 apiece that is ~3e-6 at x=1.
                assert!((run1(&a, t, xv) - xv.tan()).abs() <= 1e-4, "tan({xv})");
            }
        }

        /// exp/exp2/ln/log2/log10 lower to arithmetic via the bit-manip
        /// primitives (TruncToInt/IntToFloat/IAdd/Shl/Shr/BitAnd/BitOr) — the
        /// float↔int twiddling no backend can avoid. Validated vs `f32`.
        #[test]
        fn exp_log_match_scalar() {
            // exp / exp2 over a moderate range.
            for &xv in &[-2.0f32, -0.5, 0.0, 0.7, 1.5, 3.0] {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let e = a.push_unary(OpKind::Exp, x);
                let rel = (run1(&a, e, xv) - xv.exp()).abs() / xv.exp().max(1.0);
                assert!(rel <= 1e-2, "exp({xv})");

                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let e2 = a.push_unary(OpKind::Exp2, x);
                let rel = (run1(&a, e2, xv) - xv.exp2()).abs() / xv.exp2().max(1.0);
                assert!(rel <= 1e-2, "exp2({xv})");
            }
            // ln / log2 / log10 over positive inputs.
            for &xv in &[0.25f32, 0.5, 1.0, 2.0, 5.0, 100.0] {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let l = a.push_unary(OpKind::Ln, x);
                assert!((run1(&a, l, xv) - xv.ln()).abs() <= 3e-2, "ln({xv})");

                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let l2 = a.push_unary(OpKind::Log2, x);
                assert!((run1(&a, l2, xv) - xv.log2()).abs() <= 3e-2, "log2({xv})");
            }
        }

        /// atan/atan2/asin/acos lower to arithmetic + If (atan2 is the core;
        /// the others derive from it). Value path only — atan2 uses If, which
        /// the jet path can't differentiate. Validated vs `f32`.
        #[test]
        fn inverse_trig_match_scalar() {
            // The atan polynomial is minimax: 8.7e-5 across the interval.
            // What sets this bound is therefore not the polynomial but
            // `Recip`, which is a hardware
            // *estimate* — ~12 bits from `rcpps`, ~14 from `vrcp14ps` — so it
            // injects ~1.2e-4 into the ratio and differs by ISA level. That is
            // also why this is looser than the same check in
            // `pixelflow-ir/tests/trig_range.rs`: the scalar oracle's `Recip`
            // is an exact `1.0/x`, so that test bounds the polynomial and this
            // one bounds the polynomial plus the estimate.
            const ATAN_TOL: f32 = 1e-3;

            // atan over a wide range (exercises the |ratio|>1 swap branch).
            for &xv in &[0.0f32, 0.3, 1.0, 2.5, -0.7, -4.0] {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let at = a.push_unary(OpKind::Atan, x);
                assert!(
                    (run1(&a, at, xv) - xv.atan()).abs() <= ATAN_TOL,
                    "atan({xv})"
                );
            }
            // asin/acos on [-1, 1].
            for &xv in &[-0.9f32, -0.4, 0.0, 0.4, 0.9] {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let s = a.push_unary(OpKind::Asin, x);
                assert!(
                    (run1(&a, s, xv) - xv.asin()).abs() <= ATAN_TOL,
                    "asin({xv})"
                );

                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let c = a.push_unary(OpKind::Acos, x);
                assert!(
                    (run1(&a, c, xv) - xv.acos()).abs() <= ATAN_TOL,
                    "acos({xv})"
                );
            }
            // atan2 across quadrants (y in var0, x in var1). The (1,1)/(-1,-1)…
            // cases sit at |ratio|=1, the polynomial's worst point.
            let pts = [
                (1.0f32, 1.0f32),
                (1.0, -1.0),
                (-1.0, -1.0),
                (-1.0, 1.0),
                (0.5, -2.0),
            ];
            for &(yv, xv) in &pts {
                let mut a = ExprArena::new();
                let y = a.push_var(0);
                let x = a.push_var(1);
                let r = a.push_binary(OpKind::Atan2, y, x);
                let got = run_xy(&a, r, yv, xv);
                assert!(
                    (got - yv.atan2(xv)).abs() <= ATAN_TOL,
                    "atan2({yv},{xv}) = {got}"
                );
            }
        }

        /// A transcendental composed inside arithmetic still works: sin(x)·x + 1.
        #[test]
        fn transcendental_in_expression() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let s = a.push_unary(OpKind::Sin, x);
            let sx = a.push_binary(OpKind::Mul, s, x);
            let one = a.push_const(1.0);
            let root = a.push_binary(OpKind::Add, sx, one);
            for &xv in &[0.2f32, 0.9, 2.1, -1.3] {
                let want = xv.sin() * xv + 1.0;
                // sin's ~3e-3 error is scaled by |x|, so allow for that.
                let tol = 3e-3 * (1.0 + xv.abs());
                assert!(
                    (run1(&a, root, xv) - want).abs() <= tol,
                    "sin(x)·x+1 @ {xv}"
                );
            }
        }
    }

    // =========================================================================
    // Shared-pipeline path (schedule → regalloc → spill), on the host's tier.
    // =========================================================================
    mod sched {
        use super::*;

        /// One point of a kernel whose single argument is bound to `u` — the
        /// lattice-invariant third input a test used to spell `Var(2)`.
        fn eval_point_with_arg(code: &executable::CompiledKernel, x: f32, y: f32, u: f32) -> f32 {
            collapse_into(code, &[], &[u], (x, y), POINT)[0]
        }

        /// Declare one argument in `a` and return its leaf.
        fn arg_leaf(a: &mut ExprArena, default: f32) -> ExprId {
            let slot = a.declare_uniform(pixelflow_ir::Uniform::new(default).decl());
            a.push_uniform(slot)
        }

        fn run(res: &CompileResult, x: f32, y: f32) -> f32 {
            eval_point(&res.code, x, y)
        }

        /// How many of its own values the scope that stores spills.
        ///
        /// Not [`CompileResult::spill_count`], which is the body's alone and
        /// which counts a scope's parked roots and its store along with them
        /// — both are in a slot by construction rather than by pressure, so
        /// no kernel ever reaches zero by that measure.
        fn sample_spills(a: &ExprArena, root: ExprId, file: &regalloc::RegisterFile) -> usize {
            let nest = allocate_nest(native_schedule(a, root, POINT), file);
            let scopes = core::iter::once(regalloc::Scope::Body)
                .chain((0..nest.fold_count()).map(regalloc::Scope::Fold));
            scopes
                .map(|s| nest.scope(s))
                .filter(|view| {
                    view.schedule()
                        .iter()
                        .any(|d| matches!(d.op, ScheduledOp::Write { .. }))
                })
                .map(|view| {
                    view.schedule()
                        .iter()
                        .filter(|d| {
                            !matches!(
                                d.op,
                                ScheduledOp::Write { .. }
                                    | ScheduledOp::Seq(..)
                                    | ScheduledOp::Reduce(..)
                            ) && !view.parked_by_an_enclosing_scope(d.value)
                                && spills(view.placement(d.value))
                        })
                        .count()
                })
                .max()
                .expect("a collapse stores somewhere")
        }

        const PTS: &[(f32, f32, f32)] = &[
            (3.0, 4.0, 0.0),
            (1.0, 2.0, 3.0),
            (-2.0, 0.5, 1.5),
            (0.7, -1.3, 2.1),
        ];

        /// An expression that fits in registers compiles without spilling and
        /// computes the right answer.
        ///
        /// This used to compare a "Sethi-Ullman path" against a "scheduled
        /// path". Both arms had long since become the same `compile` call,
        /// so it compiled one function twice and asserted it equalled
        /// itself; only the ground-truth comparison was load-bearing.
        #[test]
        fn sched_no_spill_is_correct() {
            // f = sqrt(X*X + Y*Y) - Y*U, a non-commutative shape whose third
            // input is the kernel's argument rather than a third coordinate.
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let z = arg_leaf(&mut a, 0.0);
            let xx = a.push_binary(OpKind::Mul, x, x);
            let yy = a.push_binary(OpKind::Mul, y, y);
            let sum = a.push_binary(OpKind::Add, xx, yy);
            let dist = a.push_unary(OpKind::Sqrt, sum);
            let yz = a.push_binary(OpKind::Mul, y, z);
            let sub = a.push_binary(OpKind::Sub, dist, yz); // dist - Y*Z
            let root = sub;

            let sched = compile(&a, root, POINT).expect("compile");
            assert_eq!(
                sample_spills(&a, root, &native_file()),
                0,
                "should fit without spilling"
            );

            for &(px, py, pz) in PTS {
                let want = (px * px + py * py).sqrt() - py * pz;
                let got = eval_point_with_arg(&sched.code, px, py, pz);
                assert!((got - want).abs() <= 1e-4, "got {got} want {want}");
            }
        }

        /// A wide expression that exceeds the allocatable registers must spill
        /// and still compute the right answer.
        #[test]
        fn sched_spills_and_is_correct() {
            // sum_{i=1..=10} (X + i) * (Y + i), as a balanced tree, against a
            // pool at the floor: more live at once than seven registers hold.
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let mut terms = alloc::vec::Vec::new();
            for i in 1..=10u32 {
                let c = a.push_const(i as f32);
                let ax = a.push_binary(OpKind::Add, x, c);
                let by = a.push_binary(OpKind::Add, y, c);
                terms.push(a.push_binary(OpKind::Mul, ax, by));
            }
            while terms.len() > 1 {
                let mut next = alloc::vec::Vec::new();
                let it = terms.chunks(2);
                for pair in it {
                    if pair.len() == 2 {
                        next.push(a.push_binary(OpKind::Add, pair[0], pair[1]));
                    } else {
                        next.push(pair[0]);
                    }
                }
                terms = next;
            }
            let root = terms[0];

            let sched = compile_at_floor(&a, root, POINT);
            assert!(
                sample_spills(&a, root, &regalloc::tests::at_floor(native_file(), 0)) > 0,
                "expected spilling; widen the expression if this regresses"
            );

            for &(px, py, _pz) in PTS {
                let mut want = 0.0f32;
                for i in 1..=10u32 {
                    want += (px + i as f32) * (py + i as f32);
                }
                let got = run(&sched, px, py);
                let tol = 1e-3 * want.abs().max(1.0);
                assert!((got - want).abs() <= tol, "spill: got {got} want {want}");
            }
        }

        /// Exercises the shared driver's If short-circuit guard path on x86
        /// (MOVMSKPS all-true/all-false branches): `(X > 0) ? Y*Y*Y : X+X+X`,
        /// with arm-exclusive subexpressions so a guard region forms. Uniform
        /// inputs take the all-true / all-false branches.
        ///
        /// Both arms are per-*lane*, which is what makes them arms: an arm of
        /// the kernel's arguments alone would be lattice-invariant and hoist
        /// out of the body entirely, leaving nothing for a guard to skip.
        #[test]
        fn sched_if_guards() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let zero = a.push_const(0.0);
            let cond = a.push_binary(OpKind::Gt, x, zero); // X > 0 -> mask
            let yy = a.push_binary(OpKind::Mul, y, y);
            let yyy = a.push_binary(OpKind::Mul, yy, y); // true arm: Y^3
            let zz = a.push_binary(OpKind::Add, x, x);
            let zzz = a.push_binary(OpKind::Add, zz, x); // false arm: 3X
            let root = a.push_ternary(OpKind::If, cond, yyy, zzz);

            let sched = compile(&a, root, POINT).expect("scheduled compile");

            // x>0 -> all-true -> Y^3 ; x<=0 -> all-false -> 3X.
            for &(px, py, _pz) in PTS {
                let want = if px > 0.0 { py * py * py } else { 3.0 * px };
                let got = run(&sched, px, py);
                assert!(
                    (got - want).abs() <= 1e-3,
                    "select: ({px},{py}) got {got} want {want}"
                );
            }
        }
    }

    // =========================================================================
    // Backend op-coverage completeness (docs/designs/2026-07-25-two-level-ir-
    // and-backend-completeness.md)
    // =========================================================================
    //
    // Turns "backend X silently doesn't support op Y" into a named, itemized
    // test failure instead of a gap nobody notices until something happens to
    // exercise it (this is exactly how AVX-512's binary-op dispatch sat at
    // 6-of-15 required ops — nothing enumerated "the ops every backend must
    // support" anywhere, so the hole was invisible until 36 tests failed by
    // accident the first time someone compiled with `-C target-feature=
    // +avx512f`).
    //
    // Every backend compiles on every host — emission is a pure function
    // into bytes — so the sweeps below run for all three from whichever
    // machine runs the tests, and a gap in any of them fails every CI job by
    // name rather than only the leg that happens to select that backend.
    mod uniforms {
        use super::*;
        use pixelflow_ir::arena::{UniformDecl, UniformIdentity};

        fn decl(default: f32) -> UniformDecl {
            UniformDecl {
                id: UniformIdentity::mint(),
                default,
            }
        }

        /// `x + u₀ + 2·u₁`, compiled once and run under two blocks: the
        /// values come from the block at the call, and the uniform-only
        /// product was hoisted.
        #[test]
        fn a_block_is_read_at_the_call_not_at_compile() {
            let mut a = ExprArena::new();
            let u0 = a.declare_uniform(decl(0.0));
            let u1 = a.declare_uniform(decl(0.0));
            let x = a.push_var(0);
            let r0 = a.push_uniform(u0);
            let r1 = a.push_uniform(u1);
            let two = a.push_const(2.0);
            let scaled = a.push_binary(OpKind::Mul, r1, two);
            let sum = a.push_binary(OpKind::Add, x, r0);
            let root = a.push_binary(OpKind::Add, sum, scaled);
            let res = compile(&a, root, batch()).expect("compile");
            assert!(res.hoisted_values >= 1, "2·u₁ is per call");

            for block in [[1.0f32, 10.0], [-2.5, 0.25]] {
                let out = eval_batch(&res.code, &[], &block, 0.0, 0.0);
                for (i, got) in out.iter().enumerate() {
                    assert_eq!(
                        *got,
                        i as f32 + block[0] + 2.0 * block[1],
                        "lane {i} under {block:?}"
                    );
                }
            }
        }

        /// The block's pointer is the context entry after the buffer slots:
        /// a kernel over one buffer reads its block from `ctx[1]`.
        #[test]
        fn the_block_pointer_follows_the_buffer_slots() {
            use pixelflow_ir::arena::{BufferDecl, BufferIdentity};
            let data = [10.0f32, 20.0, 30.0, 40.0];
            let mut a = ExprArena::new();
            let buf = a.declare_buffer(BufferDecl {
                id: BufferIdentity::mint(),
                width: 4,
                height: 1,
            });
            let u = a.declare_uniform(decl(0.0));
            let x = a.push_var(0);
            let zero = a.push_const(0.0);
            let g = a.push_gather(buf, x, zero);
            let r = a.push_uniform(u);
            let root = a.push_binary(OpKind::Add, g, r);
            let res = compile(&a, root, batch()).expect("compile");

            let out = eval_batch(&res.code, &[data.as_ptr()], &[0.5f32], 0.0, 0.0);
            for (i, got) in out.iter().enumerate() {
                assert_eq!(*got, data[i.min(3)] + 0.5, "lane {i}");
            }
        }

        /// `select(u > 0, p(t[u]), 0) + (u + 1) + x`, `p` a polynomial long
        /// enough to be worth a branch: an `If` over per-call values, so the
        /// body computes it and clusters its arms, and `u + 1` — read by the
        /// root, not the `If` — is what makes the true arm non-contiguous
        /// until it does. The arm reads the table through its `Context`
        /// pointer, which only the arm's broadcast reads. A pointer operand is
        /// a read to the guard analysis, so clustering keeps that pointer
        /// ahead of the broadcast rather than sinking it past the `If` as a
        /// stranger.
        #[test]
        fn a_per_call_if_reads_its_table_through_a_defined_pointer() {
            use pixelflow_ir::arena::{BufferDecl, BufferIdentity};
            let data = [4.0f32, 1.5, -2.0, 0.5];
            let poly = |t: f32| ((t * t + t) * t + 3.0) * t * t + 1.0;
            let mut a = ExprArena::new();
            let buf = a.declare_buffer(BufferDecl {
                id: BufferIdentity::mint(),
                width: data.len() as u32,
                height: 1,
            });
            let u = a.declare_uniform(decl(0.0));
            let x = a.push_var(0);
            let uu = a.push_uniform(u);
            let zero = a.push_const(0.0);
            let one = a.push_const(1.0);
            let three = a.push_const(3.0);
            let mask = a.push_binary(OpKind::Gt, uu, zero);
            let leaf = a.push_buffer(buf);
            let t = a.push_binary(OpKind::RawGather, leaf, uu);
            let tt = a.push_binary(OpKind::Mul, t, t);
            let p = a.push_binary(OpKind::Add, tt, t);
            let p = a.push_binary(OpKind::Mul, p, t);
            let p = a.push_binary(OpKind::Add, p, three);
            let p = a.push_binary(OpKind::Mul, p, t);
            let p = a.push_binary(OpKind::Mul, p, t);
            let p = a.push_binary(OpKind::Add, p, one);
            let sel = a.push_ternary(OpKind::If, mask, p, zero);
            let intruder = a.push_binary(OpKind::Add, uu, one);
            let lhs = a.push_binary(OpKind::Add, sel, intruder);
            let root = a.push_binary(OpKind::Add, lhs, x);

            let res = compile(&a, root, batch()).expect("compile");
            for block in [1.0f32, 2.0, 3.0, 0.0, -1.0] {
                let arm = if block > 0.0 {
                    poly(data[block as usize])
                } else {
                    0.0
                };
                let out = eval_batch(&res.code, &[data.as_ptr()], &[block], 0.0, 0.0);
                for (i, got) in out.iter().enumerate() {
                    assert_eq!(
                        *got,
                        arm + (block + 1.0) + i as f32,
                        "lane {i}, u = {block}"
                    );
                }
            }
        }

        /// The slot `UniformId` used to stop at, and one past it, in
        /// bytes: offset 3 shifted up by a full 16-bit range, so that the
        /// byte offset `262_156` (`0x0004_000C`) is `0x0C` wrapped to 16 bits
        /// — the load a narrower offset would have emitted for it, reading
        /// argument 3.
        const PAST_U16: u64 = 3 + (u16::MAX as u64 + 1);
        const PAST_U16_BYTES: u32 = 262_156;

        /// The whole path at that width: an arena declaring more arguments
        /// than 16 bits index, reading the last, scheduled and emitted by
        /// each backend from this host. The slot survives `arena_to_schedule`
        /// and `resolve_operands` unnarrowed, and the bytes carry the
        /// displacement of the argument actually read.
        ///
        /// The schedule is read *by block*, the way
        /// `a_uniform_and_what_depends_on_it_alone_land_in_the_body` tells
        /// the kernel's uniform from the origin's: the link's block (context
        /// slot 0, there being no buffers) is read exactly once, at
        /// `PAST_U16`, and the origin's block (slot 1) exactly twice, at 0
        /// and 1 — `x0` and `y0` at the slots `origin_slots` found for them,
        /// which lie past every one of the kernel's own. That is what pins
        /// `origin_slots` at the widened width: were it to narrow its
        /// answer to 16 bits, slots `PAST_U16 + 1` and `+ 2` would come back
        /// as 4 and 5, match no read, and the origin would schedule as two
        /// *link* reads past the end of the block — while the `PAST_U16`
        /// read and its displacement in the bytes stayed exactly as they are.
        #[test]
        fn a_uniform_past_the_old_u16_width_loads_on_every_backend() {
            use pixelflow_ir::arena::{UniformDecl, UniformIdentity};
            const ARGUMENTS: u64 = PAST_U16 + 1;
            let mut a = ExprArena::new();
            let mut last = None;
            for i in 0..ARGUMENTS {
                last = Some(a.declare_uniform(UniformDecl {
                    id: UniformIdentity::mint(),
                    default: i as f32,
                }));
            }
            let last = last.expect("declared");
            assert_eq!(last, UniformId(PAST_U16));
            let x = a.push_var(0);
            let y = a.push_var(1);
            let xy = a.push_binary(OpKind::Add, x, y);
            let u = a.push_uniform(last);
            let root = a.push_binary(OpKind::Add, xy, u);

            let for_backend = |file: regalloc::RegisterFile| {
                schedule_for(&a, root, POINT, file.vector_bytes / BYTES_PER_LANE)
            };
            let mut avx2b = avx2::driver::Avx2Backend::new();
            let mut avx512b = avx512::driver::Avx512Backend::new();
            let mut neon = aarch64::driver::Aarch64Backend::new();

            // Every uniform read, as (the context slot of the block it reads,
            // its offset in that block), sorted.
            let block_reads = |schedule: &[regalloc::Def]| -> Vec<(u16, u64)> {
                let block_of = |base: regalloc::ValueId| {
                    schedule
                        .iter()
                        .find_map(|d| match d.op {
                            ScheduledOp::Context(slot) if d.value == base => Some(slot),
                            _ => None,
                        })
                        .expect("a uniform read's base is a block's Context def")
                };
                let mut reads: Vec<(u16, u64)> = schedule
                    .iter()
                    .filter_map(|d| match d.op {
                        ScheduledOp::Uniform(base, offset) => Some((block_of(base), offset)),
                        _ => None,
                    })
                    .collect();
                reads.sort_unstable();
                reads
            };
            assert_eq!(
                block_reads(&for_backend(avx2b.register_file())),
                [(0, PAST_U16), (1, 0), (1, 1)],
                "the last argument from the link's block, at its full width; \
                 x0 and y0 from the origin's block, at theirs"
            );

            let disp = PAST_U16_BYTES.to_le_bytes();
            for (tier, code) in [
                (
                    "AVX2",
                    compile_schedule(for_backend(avx2b.register_file()), &mut avx2b)
                        .expect("AVX2")
                        .code,
                ),
                (
                    "AVX-512",
                    compile_schedule(for_backend(avx512b.register_file()), &mut avx512b)
                        .expect("AVX-512")
                        .code,
                ),
            ] {
                assert!(
                    code.as_bytes().windows(disp.len()).any(|w| w == disp),
                    "{tier}: no vbroadcastss with disp32 {PAST_U16_BYTES:#x}"
                );
            }

            let neon_code = compile_schedule(for_backend(neon.register_file()), &mut neon)
                .expect("NEON")
                .code;
            let words: Vec<u32> = neon_code
                .as_bytes()
                .chunks(4)
                .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                .collect();
            // The largest 16-byte-aligned step `add`'s 12-bit immediate holds.
            let step = 4080;
            let add_ip0 = 0x9100_0000 | (step << 10) | (16 << 5) | 16;
            let ldr_s_ip0 = |w: u32| (w & !0x1F) == 0xBD40_0000 | (16 << 5);
            assert!(
                words.contains(&add_ip0) && words.iter().copied().any(ldr_s_ip0),
                "NEON: no IP0-addressed load of the argument"
            );
        }
    }

    /// A one-row buffer of `width` samples, declared in `a`.
    fn table(a: &mut ExprArena, width: u32) -> pixelflow_ir::arena::BufferId {
        a.declare_buffer(pixelflow_ir::arena::BufferDecl {
            id: pixelflow_ir::arena::BufferIdentity::mint(),
            width,
            height: 1,
        })
    }

    /// What an `If` arm may own when the values it reads are shared with the
    /// world outside it — through a block pointer the lowering mints.
    mod arm_ownership {
        use super::*;
        use pixelflow_ir::arena::{UniformDecl, UniformIdentity};

        fn decl(default: f32) -> UniformDecl {
            UniformDecl {
                id: UniformIdentity::mint(),
                default,
            }
        }

        /// `x > 0 ? rsqrt(x·u1 + 2) : -x`, plus `x·u2` outside the `If`: two
        /// arguments of one link block, the first read only inside the true
        /// arm and the second outside it.
        ///
        /// The block's base is a `Context` def made at the first `Uniform`
        /// read, which sits inside the arm's span — and is read again by the
        /// second uniform's load, outside it. An arm that owned the base
        /// would skip it on a batch with no true lane, and the second
        /// uniform's load would read a register nothing wrote. So the base
        /// belongs to the scope, the arm keeps its branch, and the kernel is
        /// right on a batch of all-true, all-false and mixed lanes.
        #[test]
        fn a_block_pointer_read_inside_an_arm_and_outside_it_is_the_scopes() {
            let mut a = ExprArena::new();
            let (u1, u2) = (a.declare_uniform(decl(3.0)), a.declare_uniform(decl(5.0)));
            let x = a.push_var(0);
            let zero = a.push_const(0.0);
            let two = a.push_const(2.0);
            let mask = a.push_binary(OpKind::Gt, x, zero);
            let first = a.push_uniform(u1);
            let scaled = a.push_binary(OpKind::Mul, x, first);
            let shifted = a.push_binary(OpKind::Add, scaled, two);
            let heavy = a.push_unary(OpKind::Rsqrt, shifted);
            let light = a.push_unary(OpKind::Neg, x);
            let sel = a.push_ternary(OpKind::If, mask, heavy, light);
            let second = a.push_uniform(u2);
            let tail = a.push_binary(OpKind::Mul, x, second);
            let root = a.push_binary(OpKind::Add, sel, tail);

            let shape = LatticeShape::new([lanes() as u32, 1]);
            let result = compile(&a, root, shape).expect("compiles");
            let (guards, arms_branched, _) = census(&a, root, shape);
            assert_eq!(
                (guards, arms_branched),
                (1, 1),
                "the true arm is worth a branch and the false arm is not"
            );

            let expect = |x: f32| {
                let sel = if x > 0.0 {
                    1.0 / (x * 3.0 + 2.0).sqrt()
                } else {
                    -x
                };
                sel + x * 5.0
            };
            let width = lanes();
            for (label, start) in [
                ("all lanes true", 1.0),
                ("all lanes false", -(width as f32) - 1.0),
                ("mixed lanes", -(width as f32 / 2.0) + 0.5),
            ] {
                let out = collapse_into(&result.code, &[], &[3.0, 5.0], (start, 0.0), shape);
                for (lane, got) in out.iter().enumerate() {
                    let want = expect(start + lane as f32);
                    assert!(
                        (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                        "{label}, lane {lane}: {got} != {want}"
                    );
                }
            }
        }
    }

    /// A gather whose address the lane binder does not reach is one scalar
    /// load broadcast — `ScheduledOp::Broadcast`, split from `Gather` in
    /// `arena_to_schedule` by the index's variance.
    mod broadcast {
        use super::*;

        /// Every lane holds the one element the row names, and another row
        /// another element: the broadcast reads through the same context
        /// slot a gather does, at the index lane 0 holds.
        #[test]
        fn every_lane_holds_the_rows_element() {
            let data: Vec<f32> = (0..8).map(|i| 10.0 * i as f32 + 1.0).collect();
            let mut a = ExprArena::new();
            let buf = table(&mut a, data.len() as u32);
            let y = a.push_var(1);
            let leaf = a.push_buffer(buf);
            let root = a.push_binary(OpKind::RawGather, leaf, y);
            let res = compile(&a, root, batch()).expect("compile");
            for row in [0.0f32, 3.0, 7.0] {
                let out = eval_batch(&res.code, &[data.as_ptr()], &[], 0.0, row);
                assert!(
                    out.iter().all(|&v| v == data[row as usize]),
                    "row {row}: {out:?}"
                );
            }
        }
    }

    /// A buffer's base is a value the allocator places: the `Context` def is
    /// computed once per call and carried into the folds that read through
    /// it, so a gather's own instruction is the read and nothing else
    /// (docs/plans/2026-09-22-a-pointer-is-a-value.md).
    mod pointer_class {
        use super::*;

        /// A table read by the row: the lattice's row fold gathers through
        /// the base every trip, and the body reads the origin block's base
        /// for the row's own coordinate.
        fn gather_by_row() -> (ExprArena, ExprId) {
            let mut a = ExprArena::new();
            let buf = table(&mut a, 8);
            let y = a.push_var(1);
            let leaf = a.push_buffer(buf);
            let root = a.push_binary(OpKind::RawGather, leaf, y);
            (a, root)
        }

        /// Every fold that reads a base finds it in a pointer register at
        /// its head: carried by the body, never parked and reloaded.
        #[test]
        fn a_base_read_inside_a_fold_is_carried_into_it() {
            let (a, root) = gather_by_row();
            let file = native_file();
            let nest = allocate_nest(native_schedule(&a, root, batch()), &file);
            let mut reads = 0;
            for j in 0..nest.fold_count() {
                let view = nest.scope(regalloc::Scope::Fold(j));
                for def in view.schedule() {
                    let base = match def.op {
                        ScheduledOp::Gather(_, base)
                        | ScheduledOp::Broadcast(_, base)
                        | ScheduledOp::Uniform(base, _) => base,
                        _ => continue,
                    };
                    reads += 1;
                    let at = view.at_head(base);
                    assert!(
                        matches!(at, regalloc::Where::Ptr(_)),
                        "Fold({j}) reads {base:?} and finds it at {at:?}"
                    );
                }
            }
            assert!(reads > 0, "the fixture's folds read no base at all");
        }

        /// The `Context` def's load is the only load of a base per call.
        ///
        /// Counted in the AVX2 tier's bytes, emitted on whatever host this
        /// runs on: `mov r9..r11, [rdi + disp32]` is `REX.WR 8B` then a ModRM
        /// of mod=10, reg=1..3, rm=rdi (`8F`/`97`/`9F`) — the same
        /// instruction on every x86 tier — and the pool holds no other
        /// pointer register. One per `Context` def in the schedule,
        /// wherever the def sits; a base parked in a slot would reload from
        /// `rsp` instead, which does not match, and the count would still be
        /// right — what would be wrong is the allocation, and the test above
        /// is the one that says so.
        #[test]
        fn a_context_pointer_is_loaded_once_per_call() {
            // The AVX2 tier's own batch, whatever this host's is: the
            // schedule is packed at the lane count the backend stores.
            const AVX2_LANES: u32 = 8;
            let (a, root) = gather_by_row();
            let schedule = schedule_for(&a, root, LatticeShape::new([AVX2_LANES, 1]), AVX2_LANES);
            let pointers = schedule
                .iter()
                .filter(|d| matches!(d.op, ScheduledOp::Context(_)))
                .count();
            assert!(pointers >= 2, "a buffer and the origin block: {pointers}");
            let res =
                compile_schedule(schedule, &mut avx2::driver::Avx2Backend::new()).expect("compile");
            let loads = res
                .code
                .as_bytes()
                .windows(3)
                .filter(|w| w[0] == 0x4C && w[1] == 0x8B && matches!(w[2], 0x8F | 0x97 | 0x9F))
                .count();
            assert_eq!(loads, pointers, "context pointer loads in the whole kernel");
        }
    }

    mod backend_op_coverage {
        use super::super::coverage::*;
        use super::*;

        /// Run one `ResolvedOp` through a backend and report whether it
        /// emitted.
        ///
        /// Every backend signals an op it cannot encode the same way now — by
        /// panicking through [`unimplemented_op`], because after `legalize`
        /// that is a missing implementation rather than a property of the
        /// kernel. `catch_unwind` is what lets this test report *which* ops a
        /// backend owes, by name and all of them, instead of dying on the
        /// first one.
        fn try_emit<B: IsaBackend>(backend: &mut B, op: ResolvedOp) -> bool {
            let plan = InstructionPlan {
                reloads: alloc::vec::Vec::new(),
                op,
                setup_mov: None,
                // Every shape below uses registers 4-7, so these stand in for
                // whatever scratch the allocator would hand an encoding that
                // asks for some. A backend that wants scratch and finds none
                // panics, which `try_emit` would report as a missing op.
                // `Gpr(9..=11)`/`KReg(1)` stand in the same way for the
                // GPR/mask-class reservations `Gather`/`Uniform`/compare ask
                // for.
                scratch: regalloc::tests::scratch_with_classes(
                    Some([Reg(15), Reg(14), Reg(13), Reg(12)]),
                    [Some(Reg(11)), Some(Reg(10))],
                    Some([Gpr(9), Gpr(10), Gpr(11)]),
                    Some(KReg(1)),
                ),
            };
            let mut code = alloc::vec::Vec::new();
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                backend.emit_plan(&mut code, &plan)
            }))
            .map(|r| r.is_ok())
            .unwrap_or(false)
        }

        /// Sweep the required unary/binary/shift op lists plus the two
        /// bespoke ternary shapes (`MulAdd`, `If`) against `backend`,
        /// collecting every failure instead of stopping at the first one —
        /// a completeness gap is much cheaper to fix as an itemized list
        /// than rediscovered one `cargo test` run per missing op.
        fn assert_covers_required_ops<B: IsaBackend>(backend_name: &str, backend: &mut B) {
            // The explicit `try_emit` calls below for MulAdd/If are this
            // constant, unrolled by hand (each needs its own `ResolvedOp`
            // shape, so they aren't worth a generic loop) — kept in sync
            // deliberately rather than by a shared loop. `MulAdd` unrolls to
            // one: `FusedMulAdd`.
            debug_assert_eq!(REQUIRED_TERNARY_OPS, &[OpKind::MulAdd, OpKind::If]);
            let mut missing = alloc::vec::Vec::new();

            for &op in REQUIRED_UNARY_OPS {
                if !try_emit(
                    backend,
                    ResolvedOp::Unary {
                        op,
                        dst: Reg(4),
                        src: Reg(5),
                    },
                ) {
                    missing.push(alloc::format!("unary {:?}", op));
                }
            }
            for &op in REQUIRED_BINARY_OPS {
                if !try_emit(
                    backend,
                    ResolvedOp::Binary {
                        op,
                        dst: Reg(4),
                        left: Reg(4),
                        right: Reg(5),
                    },
                ) {
                    missing.push(alloc::format!("binary {:?}", op));
                }
            }
            for &op in REQUIRED_SHIFT_OPS {
                if !try_emit(
                    backend,
                    ResolvedOp::ShiftImm {
                        op,
                        dst: Reg(4),
                        src: Reg(4),
                        amount: 1,
                    },
                ) {
                    missing.push(alloc::format!("shift {:?}", op));
                }
            }
            if !try_emit(
                backend,
                ResolvedOp::FusedMulAdd {
                    dst: Reg(4),
                    a: Reg(5),
                    b: Reg(6),
                },
            ) {
                missing.push(alloc::string::String::from("ternary MulAdd"));
            }
            if !try_emit(
                backend,
                ResolvedOp::If {
                    dst: Reg(4),
                    if_true: Reg(5),
                    if_false: Reg(6),
                },
            ) {
                missing.push(alloc::string::String::from("ternary If"));
            }

            assert!(
                missing.is_empty(),
                "{backend_name} is missing required ops: {missing:?} (see \
                 pixelflow-ir/src/backend/emit/coverage.rs for the full \
                 completeness contract)"
            );
        }

        // Ungated, like every sweep below it: these only *encode* — bytes
        // into a Vec, never executed — and every backend now compiles on
        // every host, so a coverage gap in any of the three fails every CI
        // job rather than only the one leg that happens to select it.
        // AVX-512's binary dispatch once shipped 6 of 15 required ops and
        // nothing noticed until someone first built `+avx512f`; that is the
        // hole this closes for all three at once.
        #[test]
        fn avx2_backend_covers_required_ops() {
            assert_covers_required_ops("Avx2Backend", &mut avx2::driver::Avx2Backend::new());
        }

        #[test]
        fn avx512_backend_covers_required_ops() {
            assert_covers_required_ops("Avx512Backend", &mut avx512::driver::Avx512Backend::new());
        }

        #[test]
        fn aarch64_backend_covers_required_ops() {
            let mut backend = aarch64::driver::Aarch64Backend::new();
            assert_covers_required_ops("Aarch64Backend", &mut backend);
        }
    }

    // =========================================================================
    // MulAdd: the encoding behind `FusedMulAdd`.
    //
    // `FusedMulAdd` rounds once on every target: each has an FMA. The shape is
    // pinned as *bytes*, not just as "it emitted something": a backend that
    // quietly encoded a multiply and an add would still satisfy
    // `backend_op_coverage`, still pass every ULP-tolerant equivalence test,
    // and change the last bit of the answer.
    //
    // Ungated, like `backend_op_coverage`: encoding is a pure function into a
    // `Vec<u8>`, so all three backends are checked from whichever host runs
    // the tests.
    // =========================================================================
    mod muladd_encoding {
        use super::*;

        const DST: Reg = Reg(4);
        const SRC_A: Reg = Reg(5);
        const SRC_B: Reg = Reg(6);

        /// A bare plan: no reloads, no setup mov, no store, no temps — just
        /// the op, so the bytes below are the op's encoding and nothing
        /// else. The empty scratch is the assertion that no backend starts
        /// asking for one on a `MulAdd` unnoticed.
        fn plan(op: ResolvedOp) -> InstructionPlan {
            InstructionPlan {
                reloads: alloc::vec::Vec::new(),
                op,
                setup_mov: None,
                scratch: regalloc::tests::scratch(None, [None, None]),
            }
        }

        fn encode<B: IsaBackend>(backend: &mut B, op: ResolvedOp) -> Vec<u8> {
            let mut code = Vec::new();
            backend.emit_plan(&mut code, &plan(op)).expect("emit_plan");
            code
        }

        fn fused() -> ResolvedOp {
            ResolvedOp::FusedMulAdd {
                dst: DST,
                a: SRC_A,
                b: SRC_B,
            }
        }

        /// `dst += a * b` in one instruction, one rounding, on every target:
        /// each of the three has an FMA.
        #[test]
        fn fused_encodes_to_the_targets_fma() {
            // VEX.256.66.0F38.W0 B8 /r — vfmadd231ps ymm4, ymm5, ymm6.
            assert_eq!(
                encode(&mut avx2::driver::Avx2Backend::new(), fused()),
                alloc::vec![0xc4, 0xe2, 0x55, 0xb8, 0xe6],
                "AVX2 fused MulAdd"
            );
            // EVEX.512.66.0F38.W0 B8 /r — vfmadd231ps zmm4, zmm5, zmm6.
            assert_eq!(
                encode(&mut avx512::driver::Avx512Backend::new(), fused()),
                alloc::vec![0x62, 0xf2, 0x55, 0x48, 0xb8, 0xe6],
                "AVX-512 fused MulAdd"
            );
            // FMLA V4.4S, V5.4S, V6.4S.
            let neon = encode(&mut aarch64::driver::Aarch64Backend::new(), fused());
            assert_eq!(
                neon,
                0x4e26_cca4u32.to_le_bytes(), // fmla v4.4s, v5.4s, v6.4s
                "aarch64 fused MulAdd"
            );
        }

        /// A `MulAdd` node really does reach a backend as `FusedMulAdd` when
        /// nothing spills — the property the byte test above assumes, and the
        /// one an upstream change (a legalization pass that decomposed it, an
        /// arena builder that never emitted it) would silently take away.
        #[test]
        fn a_muladd_dag_emits_the_fused_encoding() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let z = a.push_binary(OpKind::Add, y, x);
            let root = a.push_ternary(OpKind::MulAdd, x, y, z);

            let mut backend = avx2::driver::Avx2Backend::new();
            let lanes = backend.register_file().vector_bytes / BYTES_PER_LANE;
            let result = compile_schedule(schedule_for(&a, root, POINT, lanes), &mut backend)
                .expect("AVX2 emit");
            let code = result.code.as_bytes();
            // vfmadd231ps: VEX.256.66.0F38 B8 — the opcode byte after the
            // 3-byte prefix, whose second byte carries the map (`0F38` is
            // `00010`) under three register-extension bits the allocator's
            // choice of registers decides. Nothing else this kernel emits
            // uses the opcode.
            assert!(
                code.windows(4)
                    .any(|w| w[0] == 0xc4 && w[1] & 0x1f == 0x02 && w[3] == 0xb8),
                "a MulAdd DAG did not reach the AVX2 backend as FusedMulAdd \
                 (no VEX.0F38 B8 in {code:02x?})"
            );
        }
    }

    /// What the point-shaped rows cannot reach: a *surviving* `Reduce` under
    /// a lattice with a remainder, where the column fold is strip-mined into
    /// a main fold and a remainder fold and the `Reduce` that varies with the
    /// column is carved into both.
    ///
    /// Two instruments live here, each answering a different question of
    /// the same kernels: [`the_sibling_fold_rows_emit_the_recorded_bytes_on_every_backend`]
    /// (did any byte move, on any of the three emitters, from any host) and
    /// [`sibling_column_folds_share_a_reduce_and_its_slots`] (the slot
    /// aliasing those folds have today, which is byte-visible and must be
    /// reproduced or deliberately changed).
    mod sibling_folds {
        use super::*;
        use regalloc::RegisterAllocator;

        /// The kernels, from `tests/support`; `examples/byte_probe.rs`
        /// includes the same file, so the probe and this module measure one
        /// definition of each row.
        mod rows {
            include!("../../tests/support/sibling_rows.rs");
        }

        /// One of the three emitters, built the way a compile builds it.
        ///
        /// A kernel is legalized at the lane count of the target it is
        /// emitted for, and nothing here asks the host: the count is the
        /// ISA's, stated once in [`Target::lanes`] and checked against the
        /// backend's own register file every time one is built, so a table
        /// cannot name a width the backend does not have. Bytes are only
        /// generated, never run, so every target emits on every host.
        #[derive(Clone, Copy, Debug)]
        enum Target {
            Avx2,
            Avx512,
            Aarch64,
        }

        impl Target {
            const ALL: [Self; 3] = [Self::Avx2, Self::Avx512, Self::Aarch64];

            /// Lanes in one batch: a 256-, 512- or 128-bit vector of `f32`.
            const fn lanes(self) -> u32 {
                match self {
                    Self::Avx2 => 8,
                    Self::Avx512 => 16,
                    Self::Aarch64 => 4,
                }
            }

            /// `subject` compiled for this target.
            fn compile(self, subject: Subject<'_>) -> CompileResult {
                let lanes = self.lanes();
                match self {
                    Self::Avx2 => compile_on(avx2::driver::Avx2Backend::new(), lanes, subject),
                    Self::Avx512 => {
                        compile_on(avx512::driver::Avx512Backend::new(), lanes, subject)
                    }
                    Self::Aarch64 => {
                        compile_on(aarch64::driver::Aarch64Backend::new(), lanes, subject)
                    }
                }
            }
        }

        /// What is compiled: a kernel's arena and root, over a lattice.
        #[derive(Clone, Copy)]
        struct Subject<'a> {
            arena: &'a ExprArena,
            root: ExprId,
            shape: LatticeShape,
        }

        /// `subject` legalized for `lanes` lanes, scheduled, scoped and emitted
        /// by `backend`, which must be the width `lanes` says.
        fn compile_on<B: IsaBackend>(
            mut backend: B,
            lanes: u32,
            subject: Subject<'_>,
        ) -> CompileResult {
            assert_eq!(
                backend.register_file().vector_bytes / BYTES_PER_LANE,
                lanes,
                "the backend is not the width this test legalized for"
            );
            let Subject { arena, root, shape } = subject;
            compile_schedule(schedule_for(arena, root, shape, lanes), &mut backend)
                .expect("a sibling-fold row compiles on every backend")
        }

        /// The width a row is compiled at, which for two of the three is a
        /// fact about the target's lanes and so cannot be one number.
        #[derive(Clone, Copy)]
        enum Width {
            /// One sample: all remainder, no main fold exists.
            One,
            /// Exactly one batch: all main, no remainder fold exists.
            OneBatch,
            /// [`rows::REMAINDER_WIDTH`]: a main fold and a remainder fold.
            Remainder,
        }

        impl Width {
            fn at(self, target: Target) -> LatticeShape {
                let columns = match self {
                    Self::One => 1,
                    Self::OneBatch => target.lanes(),
                    Self::Remainder => rows::REMAINDER_WIDTH,
                };
                LatticeShape::new([columns, rows::ROWS])
            }
        }

        /// A kernel and the width it is compiled at.
        struct Row {
            name: &'static str,
            build: fn() -> (ExprArena, ExprId),
            width: Width,
        }

        fn parked_roots() -> (ExprArena, ExprId) {
            rows::parked_roots(rows::PARKED_TERMS)
        }

        fn deep_frame() -> (ExprArena, ExprId) {
            rows::deep_frame(rows::DEEP_FRAME_TERMS)
        }

        /// The glyph-like fold at the three widths that decide how many
        /// sibling column folds exist (a remainder alone; a main alone; both),
        /// and each other kernel where both exist. After them, the coverage
        /// rows: every op the backends owe (`coverage`), every way a kernel
        /// reads memory, and a frame past what NEON addresses directly, all at
        /// the width with a remainder.
        const ROWS: [Row; 11] = [
            Row {
                name: "glyph_like_w1",
                build: rows::glyph_like,
                width: Width::One,
            },
            Row {
                name: "glyph_like_wL",
                build: rows::glyph_like,
                width: Width::OneBatch,
            },
            Row {
                name: "glyph_like_w37",
                build: rows::glyph_like,
                width: Width::Remainder,
            },
            Row {
                name: "two_sibling_folds_w37",
                build: rows::two_sibling_folds,
                width: Width::Remainder,
            },
            Row {
                name: "parked_roots_w37",
                build: parked_roots,
                width: Width::Remainder,
            },
            Row {
                name: "guarded_if_in_fold_w37",
                build: rows::guarded_if_in_fold,
                width: Width::Remainder,
            },
            Row {
                name: "unary_ops_w37",
                build: rows::unary_ops,
                width: Width::Remainder,
            },
            Row {
                name: "binary_ops_w37",
                build: rows::binary_ops,
                width: Width::Remainder,
            },
            Row {
                name: "shift_muladd_blend_w37",
                build: rows::shift_muladd_blend,
                width: Width::Remainder,
            },
            Row {
                name: "memory_w37",
                build: rows::memory,
                width: Width::Remainder,
            },
            Row {
                name: "deep_frame_w37",
                build: deep_frame,
                width: Width::Remainder,
            },
        ];

        /// A target's emitted code: its length in bytes and the FNV-1a 64
        /// digest of those bytes ([`crate::fnv1a64`]).
        type Bytes = (usize, u64);

        /// `ROWS`' bytes, per target in [`Target::ALL`]'s order (AVX2,
        /// AVX-512, aarch64). A row's provenance is `git log -L` on it: a
        /// hash written here would be the hash of the commit that wrote it,
        /// which no commit can know.
        ///
        /// **A refactor does not edit this table; an intentional byte change
        /// does, in a commit of its own that says why.** A commit that edits
        /// it beside other work cannot be told apart from one that moved
        /// bytes by accident, which is the thing it exists to catch. When it
        /// fails, the failure prints the whole recomputed table.
        const GOLDEN: [[Bytes; 3]; 11] = [
            [
                (724, 0x23fc2ea3c955d064),
                (676, 0xb2ca1015afc4ecf7),
                (400, 0x78f1164693fe0da6),
            ],
            [
                (728, 0x4379d55663a9294e),
                (664, 0xf22ace44c2fda4c8),
                (400, 0xaa96d96596a05551),
            ],
            [
                (1056, 0xf4f28a978e99b9ec),
                (992, 0x66e9f1ebf718d0cd),
                (608, 0x14a21aabd82fe2e8),
            ],
            [
                (1056, 0x15c0a9e0e3472c74),
                (1056, 0x662f9c26c2bbcafc),
                (656, 0x0e7016ed19878723),
            ],
            [
                (247328, 0x919cb6c0efe53a92),
                (278032, 0x282aa77769f9e2ff),
                (188496, 0x578f9987a3386dcc),
            ],
            [
                (3012, 0x90101b60330eb1ce),
                (2932, 0x3943e837c9f115d3),
                (2144, 0x92246f5ac70b7ef7),
            ],
            [
                (1136, 0xb3294ca012954180),
                (1120, 0x2a5133484f338421),
                (704, 0x435919f39c14241c),
            ],
            [
                (932, 0xfa99679abc99f7b8),
                (1108, 0x9bac855b3070b143),
                (736, 0xe94f00f62e4f2b4e),
            ],
            [
                (636, 0x140b048db98bf221),
                (700, 0x18187fcf1e57608f),
                (432, 0xad9d34cf0f162be5),
            ],
            [
                (500, 0xa8675d79b34a94d3),
                (596, 0x261da02fa7c98009),
                (480, 0x4adfc2b1aba3fa1f),
            ],
            [
                (420428, 0x486f93190a3ec20d),
                (450476, 0x894da10795bcf548),
                (605648, 0xff316d78ddef246a),
            ],
        ];

        /// **Every sibling-fold row emits the same bytes on all three
        /// backends, on any host.**
        ///
        /// `byte_probe` answers this for the host's own tier, on a host that
        /// has it; this answers it for AVX2, AVX-512 *and* NEON on whatever
        /// runs the test, since the bytes are only generated. That is what
        /// puts aarch64 byte identity under CI on a Linux x86 runner, and
        /// what makes "did this refactor move bytes" a check that fails for
        /// everyone rather than a diff one person ran once.
        ///
        /// Host-independent by construction, and the tests that follow it
        /// hold it to that: nothing here reads [`crate::isa::detect`],
        /// `PIXELFLOW_ISA`, [`crate::jit_vector_bytes`] or the CPU. Each
        /// target is legalized at its own lane count and emitted by its own
        /// backend with the default context.
        #[test]
        fn the_sibling_fold_rows_emit_the_recorded_bytes_on_every_backend() {
            let mut recomputed = Vec::new();
            let mut moved = Vec::new();
            for (row, pins) in ROWS.iter().zip(GOLDEN) {
                let (a, root) = (row.build)();
                let mut emitted = Vec::new();
                for (target, pin) in Target::ALL.into_iter().zip(pins) {
                    let subject = Subject {
                        arena: &a,
                        root,
                        shape: row.width.at(target),
                    };
                    let result = target.compile(subject);
                    let code = result.code.as_bytes();
                    let actual = (code.len(), crate::fnv1a64(code));
                    if actual != pin {
                        moved.push(format!(
                            "{} on {target:?}: pinned {pin:x?}, emitted {actual:x?}",
                            row.name
                        ));
                    }
                    emitted.push(format!("({}, {:#018x})", actual.0, actual.1));
                }
                recomputed.push(format!("            [{}],", emitted.join(", ")));
            }
            assert!(
                moved.is_empty(),
                "emitted bytes moved from GOLDEN:\n{}\n\n\
                 if the change is intentional, re-baseline GOLDEN in its own \
                 commit with:\n{}",
                moved.join("\n"),
                recomputed.join("\n")
            );
        }

        /// A backend that forwards to another and writes down every frame
        /// offset each scope addresses, scope by scope.
        ///
        /// What a test can see of "which slot" without decoding an
        /// instruction: the driver hands every slot it uses to the backend as
        /// a displacement, so the displacements the backend was given *are*
        /// the frame the emitted code addresses. Seen are the fold loop's own
        /// slot traffic, the reloads and resolves of a value that lives in a
        /// slot, and a `Write`'s binders.
        struct Addressed<'a, B: IsaBackend> {
            inner: &'a mut B,
            open: Vec<Vec<u32>>,
            closed: Vec<(regalloc::Scope, Vec<u32>)>,
        }

        impl<'a, B: IsaBackend> Addressed<'a, B> {
            fn new(inner: &'a mut B) -> Self {
                Self {
                    inner,
                    open: Vec::new(),
                    closed: Vec::new(),
                }
            }

            /// `offset` was addressed by the scope being emitted.
            fn note(&mut self, offset: u32) {
                if let Some(open) = self.open.last_mut() {
                    open.push(offset);
                }
            }

            /// The slot a binding names, if it names one.
            fn note_binding(&mut self, binding: Option<Binding>) {
                if let Some(Binding::Loc(Loc::Slot(slot))) = binding {
                    self.note(slot.offset());
                }
            }

            /// Every offset any scope addressed.
            fn addressed(&self) -> alloc::collections::BTreeSet<u32> {
                self.closed
                    .iter()
                    .flat_map(|(_, offsets)| offsets.iter().copied())
                    .collect()
            }

            /// The offsets `scope` addressed itself, the scopes nested in it
            /// not included.
            fn addressed_by(&self, scope: regalloc::Scope) -> alloc::collections::BTreeSet<u32> {
                self.closed
                    .iter()
                    .filter(|(closed, _)| *closed == scope)
                    .flat_map(|(_, offsets)| offsets.iter().copied())
                    .collect()
            }
        }

        impl<B: IsaBackend> IsaBackend for Addressed<'_, B> {
            fn jump(&mut self, asm: &mut Assembly, label: Label) {
                self.inner.jump(asm, label);
            }

            fn register_file(&self) -> regalloc::RegisterFile {
                self.inner.register_file()
            }

            fn begin(&mut self, schedule: &[regalloc::Def]) -> Result<(), CompileError> {
                self.inner.begin(schedule)
            }

            fn emit_plan(
                &mut self,
                code: &mut Vec<u8>,
                plan: &InstructionPlan,
            ) -> Result<(), CompileError> {
                for reload in &plan.reloads {
                    match reload {
                        Reload::FromStack { slot, .. } | Reload::Ptr { slot, .. } => {
                            self.note(slot.offset());
                        }
                        Reload::Const { .. } => {}
                    }
                }
                self.inner.emit_plan(code, plan)
            }

            fn emit_mov(&mut self, code: &mut Vec<u8>, dst: Reg, src: Reg) {
                self.inner.emit_mov(code, dst, src);
            }

            fn emit_store(
                &mut self,
                code: &mut Vec<u8>,
                src: Reg,
                offset: u32,
            ) -> Result<(), CompileError> {
                self.note(offset);
                self.inner.emit_store(code, src, offset)
            }

            fn ptr_store(&mut self, code: &mut Vec<u8>, src: PtrReg, offset: u32) {
                self.note(offset);
                self.inner.ptr_store(code, src, offset);
            }

            fn ptr_load(&mut self, code: &mut Vec<u8>, dst: PtrReg, offset: u32) {
                self.note(offset);
                self.inner.ptr_load(code, dst, offset);
            }

            fn ptr_mov(&mut self, code: &mut Vec<u8>, dst: PtrReg, src: PtrReg) {
                self.inner.ptr_mov(code, dst, src);
            }

            fn emit_resolve(
                &mut self,
                code: &mut Vec<u8>,
                vid: regalloc::ValueId,
                target: Reg,
                locs: &[Option<Binding>],
            ) -> Result<Reg, CompileError> {
                self.note_binding(locs.get(vid.0 as usize).copied().flatten());
                self.inner.emit_resolve(code, vid, target, locs)
            }

            fn branch_if_arm_is_dead(&mut self, asm: &mut Assembly, test: MaskTest, label: Label) {
                self.inner.branch_if_arm_is_dead(asm, test, label);
            }

            fn frame_alloc(&mut self, code: &mut Vec<u8>, bytes: u32) {
                self.inner.frame_alloc(code, bytes);
            }

            fn frame_free(&mut self, code: &mut Vec<u8>, bytes: u32) {
                self.inner.frame_free(code, bytes);
            }

            fn anchor(&mut self, asm: &mut Assembly, pool: Label) {
                self.inner.anchor(asm, pool);
            }

            fn finish(&mut self, asm: &mut Assembly, pool: Label) {
                self.inner.finish(asm, pool);
            }

            fn slot_store(&mut self, code: &mut Vec<u8>, src: Reg, offset: u32) {
                self.note(offset);
                self.inner.slot_store(code, src, offset);
            }

            fn slot_load(&mut self, code: &mut Vec<u8>, dst: Reg, offset: u32) {
                self.note(offset);
                self.inner.slot_load(code, dst, offset);
            }

            fn scope_begin(&mut self) {
                self.open.push(Vec::new());
                self.inner.scope_begin();
            }

            fn scope_end(&mut self, scope: regalloc::Scope, bytes: u64) {
                let offsets = self.open.pop().expect("scope_end without a scope_begin");
                self.closed.push((scope, offsets));
                self.inner.scope_end(scope, bytes);
            }

            fn add_scalar(
                &mut self,
                code: &mut Vec<u8>,
                dst: Reg,
                scratch: Reg,
                scalar: f32,
            ) -> Result<(), CompileError> {
                self.inner.add_scalar(code, dst, scratch, scalar)
            }

            fn load_const(
                &mut self,
                code: &mut Vec<u8>,
                dst: Reg,
                val: f32,
            ) -> Result<(), CompileError> {
                self.inner.load_const(code, dst, val)
            }

            fn alu(&mut self, code: &mut Vec<u8>, op: OpKind, dst: Reg, srcs: [Reg; 2]) {
                self.inner.alu(code, op, dst, srcs);
            }

            // Forwarded itself, not left to the trait's default over `alu`,
            // for the reason `Counting` states: only AVX-512 overrides it.
            fn test_ge(
                &mut self,
                code: &mut Vec<u8>,
                dst: Reg,
                srcs: [Reg; 2],
                mask_scratch: Option<KReg>,
            ) {
                self.inner.test_ge(code, dst, srcs, mask_scratch);
            }

            fn emit_write(&mut self, code: &mut Vec<u8>, write: &WritePlan) {
                self.note_binding(Some(write.row));
                self.note_binding(Some(write.col));
                self.inner.emit_write(code, write);
            }

            fn emit_ret(&mut self, code: &mut Vec<u8>) {
                self.inner.emit_ret(code);
            }
        }

        /// **Two sibling column folds are one loop run under two parents, and
        /// they share one accumulator slot and one binder slot: the later
        /// fold's.**
        ///
        /// At a width with a remainder the column fold is strip-mined into a
        /// main fold and a remainder fold, and a `Reduce` varying with the
        /// column is carved into *both*. The two `ScopeFold`s carry the same
        /// `Reduce` `ValueId`, and the allocator keys a fold's accumulator
        /// slot and binder slot by that id (`accumulator_slots` and
        /// `binder_slots` in `regalloc::NestAllocation::new`, each a
        /// `collect()` that keeps the last `j` for a repeated key). So the
        /// earlier fold's slots, `m + 2j·vb` and
        /// `m + (2j + 1)·vb`, are never addressed: its loop reads, steps and
        /// stores the later fold's, and its consumers read the later's. The
        /// two folds never run at once, which is the whole reason it is sound.
        ///
        /// Accidental, and byte-visible: it decides every displacement above
        /// the first fold slot (the frame is `m + 2·fold_count·vb` and the
        /// parks follow). A change to scoping, allocation, frames or labels
        /// that gave each fold its own slots would move bytes and be sound,
        /// and must be a deliberate, separately re-baselined change; one that
        /// *meant* not to move them and did is what this catches before the
        /// byte golden has to.
        ///
        /// Seen from both ends. The nest says two folds carry one `Reduce`
        /// and are siblings. The driver, run unmodified under a backend that
        /// records the displacements it is handed, addresses the later fold's
        /// slots from both parents and the earlier fold's from none, at the
        /// floor pool (where neither is carried, so both are in memory) and
        /// at the whole one.
        #[test]
        fn sibling_column_folds_share_a_reduce_and_its_slots() {
            let (a, root) = rows::glyph_like();
            let shape = LatticeShape::new([rows::REMAINDER_WIDTH, rows::ROWS]);
            let schedule = schedule_for(&a, root, shape, Target::Avx2.lanes());

            let whole = avx2::driver::Avx2Backend::new()
                .register_file()
                .scratch
                .len()
                - regalloc::tests::FLOOR;
            for (pool, above) in [("floor", 0), ("whole", whole)] {
                let mut backend = AtFloor(avx2::driver::Avx2Backend::new(), above);
                let file = backend.register_file();
                let nest = regalloc::LinearScan
                    .allocate_nest(
                        regalloc::ScopedSchedule::from_schedule(schedule.clone()),
                        &file,
                    )
                    .expect("the glyph-like row fits the frame");

                // The nest: exactly one `Reduce` is carved into two folds.
                let mut carved: alloc::collections::BTreeMap<regalloc::ValueId, Vec<usize>> =
                    alloc::collections::BTreeMap::new();
                for j in 0..nest.fold_count() {
                    carved.entry(nest.fold_reduce_vid(j)).or_default().push(j);
                }
                let shared: Vec<(regalloc::ValueId, Vec<usize>)> = carved
                    .into_iter()
                    .filter(|(_, folds)| folds.len() > 1)
                    .collect();
                let [(reduce, folds)] = shared.as_slice() else {
                    panic!("{pool}: expected one Reduce carved into several folds: {shared:?}");
                };
                let &[earlier, later] = folds.as_slice() else {
                    panic!("{pool}: {reduce:?} is carved into {folds:?}, not into two folds");
                };

                // Siblings: separate parents, neither inside the other.
                let ancestors = |mut scope: regalloc::Scope| {
                    let mut chain = Vec::new();
                    while let regalloc::Scope::Fold(j) = scope {
                        chain.push(j);
                        scope = nest.fold_parent(j);
                    }
                    chain
                };
                let (earlier_parent, later_parent) =
                    (nest.fold_parent(earlier), nest.fold_parent(later));
                assert_ne!(
                    earlier_parent, later_parent,
                    "{pool}: two folds under one parent would be nested loops, not siblings"
                );
                assert!(
                    !ancestors(earlier_parent).contains(&later)
                        && !ancestors(later_parent).contains(&earlier),
                    "{pool}: one fold is inside the other"
                );

                // The driver, observed.
                let mut recorder = Addressed::new(&mut backend);
                let result = compile_via_backend(
                    regalloc::ScopedSchedule::from_schedule(schedule.clone()),
                    &mut recorder,
                )
                .expect("the glyph-like row compiles");
                // `spill_bytes` is the frame's `m`, where the fold slots begin.
                let root_slot = |j: usize, root: u32| {
                    result.spill_bytes + (2 * j as u32 + root) * file.vector_bytes
                };
                let (own, shared_acc, shared_binder) = (
                    [root_slot(earlier, 0), root_slot(earlier, 1)],
                    root_slot(later, 0),
                    root_slot(later, 1),
                );

                let touched = recorder.addressed();
                for slot in own {
                    assert!(
                        !touched.contains(&slot),
                        "{pool}: the earlier fold's own slot {slot} was addressed; the \
                         series must reproduce the later fold's slots being shared, or \
                         re-baseline that on purpose"
                    );
                }
                // Each parent runs its loop through the later fold's
                // accumulator slot (held there, or stored on the way out when
                // carried), and its binder's where that is not carried.
                for (which, parent) in [("earlier", earlier_parent), ("later", later_parent)] {
                    let by_parent = recorder.addressed_by(parent);
                    assert!(
                        by_parent.contains(&shared_acc),
                        "{pool}: the {which} fold's parent never addressed the shared \
                         accumulator slot {shared_acc} ({by_parent:?})"
                    );
                    if pool == "floor" {
                        assert!(
                            by_parent.contains(&shared_binder),
                            "{pool}: the {which} fold's parent never addressed the shared \
                             binder slot {shared_binder} ({by_parent:?})"
                        );
                    }
                }
            }
        }
    }
}
