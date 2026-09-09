//! JIT code emission for expression DAGs.
//!
//! ## Register allocation
//!
//! One allocator — [`regalloc::LinearScan`], linear scan with Belady eviction
//! and constant rematerialization — parameterised by one description of the
//! target, [`regalloc::RegisterFile`]. Expressions arrive from e-graph
//! extraction with shared subexpressions, which is why the allocator works on
//! a DAG schedule rather than a tree.
//!
//! All four backends run that same allocator behind the same driver
//! ([`IsaBackend`]). What a backend contributes is its `RegisterFile` — input
//! registers, the allocatable pool, how many registers its encodings and its
//! guards destroy, vector width — and its instruction encodings. Nothing else
//! about a target reaches the allocation, framing, or control-flow logic.
//!
//! ## Spilling
//!
//! Values the scratch pool cannot hold go to stack slots, laid out by
//! [`FrameLayout`] at the backend's vector stride:
//! - A value with a slot is stored to it right after its **definition**, which
//!   every path that reads the value has run — including through a `Select`
//!   guard, which can only skip a definition by skipping every read of it.
//! - Reloaded into a register the allocator reserved *for that instruction*
//!   ([`regalloc::Scratch`]); there is no register outside the pool for this,
//!   and every definition holds a pool register at its own definition.
//! - `EmitCtx::max_regs` caps the pool below the target's own count, which is
//!   how register pressure vs. spill tradeoffs are exercised deliberately

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
pub(crate) fn unimplemented_op(backend: &str, op: pixelflow_ir::kind::OpKind) -> ! {
    panic!(
        "{backend} has no encoding for {op:?} — `passes::legalize` leaves only \
         backend-legal ops, so this is a missing implementation or a bypassed \
         pipeline, not a bad kernel"
    )
}

pub mod aarch64;
pub mod avx2;
pub mod avx512;
#[cfg(test)]
pub(crate) mod coverage;
pub mod encoded;
pub mod executable;
mod guards;
pub mod regalloc;
pub mod storage;
pub mod traffic;
pub mod x86_64;

pub use encoded::EncodedInst;
pub use storage::{Slot, SourceOperand, StackFrame, Storage, StoreTarget};

use pixelflow_ir::kind::OpKind;

pub use guards::SelectArm;
use guards::analyze_select_guards;
use traffic::{Counting, EmitTraffic, ScopeTraffic};

use alloc::vec::Vec;

use crate::error::CompileError;

/// The one contract every backend's instruction types satisfy.
pub trait AsmInsn: Copy {
    /// Emit the instruction's encoded bytes into the output buffer.
    fn emit_into(self, code: &mut Vec<u8>);
}

/// A declarative sequence of assembly instructions.
///
/// Written as an array or collection of instructions, then assembled into machine code:
/// ```ignore
/// AsmProgram::from([
///     Inst::Mov { src: AX, dst: RX },
/// ]).assemble(&mut buff);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AsmProgram<S> {
    insts: S,
}

impl<S> AsmProgram<S> {
    /// Create a new assembly program wrapping an instruction sequence.
    #[inline(always)]
    pub const fn new(insts: S) -> Self {
        Self { insts }
    }
}

impl<I: AsmInsn, const N: usize> From<[I; N]> for AsmProgram<[I; N]> {
    #[inline(always)]
    fn from(insts: [I; N]) -> Self {
        Self { insts }
    }
}

impl<I: AsmInsn> From<alloc::vec::Vec<I>> for AsmProgram<alloc::vec::Vec<I>> {
    #[inline(always)]
    fn from(insts: alloc::vec::Vec<I>) -> Self {
        Self { insts }
    }
}

impl<'a, I: AsmInsn> From<&'a [I]> for AsmProgram<&'a [I]> {
    #[inline(always)]
    fn from(insts: &'a [I]) -> Self {
        Self { insts }
    }
}

impl<I: AsmInsn, S: IntoIterator<Item = I>> AsmProgram<S> {
    /// Assemble the program into the machine-code buffer.
    #[inline]
    pub fn assemble(self, code: &mut Vec<u8>) {
        for inst in self.insts {
            inst.emit_into(code);
        }
    }
}

impl<I: AsmInsn, S: IntoIterator<Item = I> + Copy> AsmInsn for AsmProgram<S> {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        self.assemble(code);
    }
}

/// Free-function fold: assemble a declarative sequence directly into `code`.
#[inline]
pub fn assemble<I: AsmInsn>(code: &mut Vec<u8>, insts: impl IntoIterator<Item = I>) {
    AsmProgram::new(insts).assemble(code);
}

/// Physical vector register index (v0..v31 on AArch64, xmm/ymm/zmm0..zmm31 on x86).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Reg(pub u8);

/// Canonical alias for vector values allocated to DAG nodes.
pub type VReg = Reg;

/// Physical 64-bit general-purpose register index (x0..x31 on AArch64, rax..r15 on x86).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Gpr(pub u8);

/// Physical pointer register index holding a memory address (x0..x31/sp on AArch64, rax..r15/rsp on x86).
///
/// Distinct from [`Gpr`] (integers, counters, indices) and [`Reg`] (SIMD vectors).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PtrReg(pub u8);

impl PtrReg {
    /// Conversion to raw register index.
    #[inline(always)]
    #[must_use]
    pub const fn raw(self) -> u8 {
        self.0
    }

    /// View as general-purpose register for instructions that manipulate pointers as raw 64-bit values.
    #[inline(always)]
    #[must_use]
    pub const fn as_gpr(self) -> Gpr {
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
pub struct KReg(pub u8);

/// A physical location where a value resides: in a register or on the stack.
///
/// Every variant of `Loc` is a writable, addressable storage location, which
/// is why `Loc` implements [`StoreTarget`] — the conversion is total.
/// A rematerialized constant has no location; it is a [`Binding`], not a `Loc`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Loc {
    /// Value is in a register.
    Reg(Reg),
    /// Value is spilled to a stack slot.
    Slot(Slot),
}

impl Loc {
    /// Get the register, panicking if the value is not in one.
    #[must_use]
    pub fn reg(self) -> Reg {
        match self {
            Loc::Reg(r) => r,
            Loc::Slot(s) => panic!("expected register, got stack slot {}", s.offset()),
        }
    }

    /// Physical storage location.
    #[must_use]
    pub fn storage(self) -> Storage {
        match self {
            Loc::Reg(r) => Storage::Reg(r),
            Loc::Slot(s) => Storage::Slot(s),
        }
    }
}

impl From<Reg> for Loc {
    #[inline]
    fn from(r: Reg) -> Self {
        Loc::Reg(r)
    }
}

impl From<Slot> for Loc {
    #[inline]
    fn from(s: Slot) -> Self {
        Loc::Slot(s)
    }
}

impl StoreTarget for Loc {
    #[inline]
    fn target_storage(self) -> Storage {
        self.storage()
    }
    #[inline]
    fn target_reg(self) -> Option<Reg> {
        match self {
            Loc::Reg(r) => Some(r),
            Loc::Slot(_) => None,
        }
    }
    #[inline]
    fn target_slot(self) -> Option<Slot> {
        match self {
            Loc::Reg(_) => None,
            Loc::Slot(s) => Some(s),
        }
    }
}

impl SourceOperand for Loc {
    #[inline]
    fn source_storage(self) -> Option<Storage> {
        Some(self.storage())
    }
    #[inline]
    fn source_reg(self) -> Option<Reg> {
        match self {
            Loc::Reg(r) => Some(r),
            Loc::Slot(_) => None,
        }
    }
    #[inline]
    fn source_slot(self) -> Option<Slot> {
        match self {
            Loc::Reg(_) => None,
            Loc::Slot(s) => Some(s),
        }
    }
    #[inline]
    fn source_const(self) -> Option<u32> {
        None
    }
}

/// The binding of a value after register allocation: a physical location
/// or a constant that is rematerialized at every use.
///
/// `Binding` is the register allocator's full answer — "where did this value
/// end up?" — and includes [`Remat`](Binding::Remat) for constants that live
/// nowhere. For a writable physical location, use [`Loc`] instead.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Binding {
    /// Value lives in a physical location (register or stack slot).
    Loc(Loc),
    /// Value is a constant (these are its `f32` bits): it lives nowhere and is
    /// re-emitted at each use.
    Remat(u32),
}

impl Binding {
    /// Get the register, panicking if the value is not in one.
    #[must_use]
    pub fn reg(self) -> Reg {
        match self {
            Binding::Loc(loc) => loc.reg(),
            Binding::Remat(bits) => panic!("expected register, got rematerialized {bits:#x}"),
        }
    }

    /// Physical storage location if not rematerialized.
    #[must_use]
    pub fn as_loc(self) -> Option<Loc> {
        match self {
            Binding::Loc(loc) => Some(loc),
            Binding::Remat(_) => None,
        }
    }

    /// Physical storage as the canonical enum, if not rematerialized.
    #[must_use]
    pub fn as_storage(self) -> Option<Storage> {
        self.as_loc().map(|l| l.storage())
    }

    /// Stack slot if spilled to stack.
    #[must_use]
    pub fn as_slot(self) -> Option<Slot> {
        match self {
            Binding::Loc(Loc::Slot(s)) => Some(s),
            _ => None,
        }
    }
}

impl From<Loc> for Binding {
    #[inline]
    fn from(loc: Loc) -> Self {
        Binding::Loc(loc)
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

impl SourceOperand for Binding {
    #[inline]
    fn source_storage(self) -> Option<Storage> {
        self.as_storage()
    }

    #[inline]
    fn source_reg(self) -> Option<Reg> {
        match self {
            Binding::Loc(Loc::Reg(r)) => Some(r),
            _ => None,
        }
    }

    #[inline]
    fn source_slot(self) -> Option<Slot> {
        match self {
            Binding::Loc(Loc::Slot(s)) => Some(s),
            _ => None,
        }
    }

    #[inline]
    fn source_const(self) -> Option<u32> {
        match self {
            Binding::Remat(bits) => Some(bits),
            _ => None,
        }
    }
}

/// Stack addresses for one scope of an allocation.
///
/// [`regalloc::Where`] says *that* a value spills; this says *where*. The
/// two are separate decisions, and this is the arrow between them: it consumes
/// one scope's [`Allocation`](regalloc::Allocation) and produces the [`Binding`]
/// the emitter encodes for every value in it.
///
/// Slots are laid out at the backend's own vector stride, so every offset
/// downstream is a real displacement. The stride was once a universal 16 that
/// each wider backend divided back out at its every load, store and prologue —
/// a convention that held only so long as nothing handed this a non-multiple
/// of 16, and would have aliased two live values onto one slot the moment
/// something did.
///
/// Per scope, not per nest. A value parked by an enclosing region lives in a
/// **hoist slot**, which outlives every region's frame and is addressed by the
/// collapse driver rather than laid out here — so this skips those, and the
/// driver pins them afterwards. Unifying the two is the next piece of work; it
/// is not this one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameLayout {
    /// Dense by `ValueId.0`: where each value lives when this scope first
    /// reaches it — at its definition for the values this scope computes.
    /// Total over the scope's schedule; the emitter carries it forward from
    /// here as the placement's later ranges take effect.
    locs: alloc::vec::Vec<Option<Binding>>,
    /// Dense by `ValueId.0`: the address of the value's slot, for every value
    /// this scope ever spills.
    ///
    /// Separate from `locs` because a placement is a schedule: a value can
    /// hold a register for part of this scope and its slot for the rest, so
    /// *that* it needs an address is a property of its whole life here, not of
    /// the one point its definition sits at.
    slot: alloc::vec::Vec<Option<Slot>>,
    /// Total frame size in bytes, a whole number of slots.
    pub frame_size: u32,
    /// How many values this frame gives a slot to.
    pub slots: u32,
}

impl FrameLayout {
    /// Give every spilled value in this scope a stack address.
    ///
    /// Pure: (scope allocation, slot stride) → layout. The collapse driver
    /// runs this twice for one region and relies on both runs agreeing.
    pub fn resolve(
        allocation: regalloc::Allocation<'_>,
        vector_bytes: u32,
    ) -> Result<Self, CompileError> {
        let schedule = allocation.schedule();
        let len = schedule
            .iter()
            .map(|def| def.value.0 as usize + 1)
            .max()
            .unwrap_or(0);
        let mut locs: alloc::vec::Vec<Option<Binding>> = alloc::vec![None; len];

        let mut frame = StackFrame::new(vector_bytes);
        let mut slot: alloc::vec::Vec<Option<Slot>> = alloc::vec![None; len];
        let mut slots = 0u32;
        for (i, def) in schedule.iter().enumerate() {
            // A value an enclosing region parked is read here from its hoist
            // slot, which is not this frame's to place. Its entry in this
            // schedule is a placeholder that emits nothing.
            if allocation.parked_by_an_enclosing_scope(def.value) {
                continue;
            }
            let v = def.value;
            // A slot is owed for the whole of this scope if the value is in
            // one at *any* point of it — not only at the point it is defined,
            // which is where a value that keeps its register for a while and
            // then loses it would have been missed.
            let spills_here = allocation.where_at(v, i) == regalloc::Where::Spilled
                || allocation
                    .transitions(v)
                    .any(|(_, at)| at == regalloc::Where::Spilled);
            if spills_here {
                let s = frame.alloc_slot()?;
                slot[v.0 as usize] = Some(s);
                slots += 1;
            }
            locs[v.0 as usize] = Some(match allocation.where_at(v, i) {
                regalloc::Where::Reg(r) => Binding::from(Reg(r.0)),
                regalloc::Where::Remat(bits) => Binding::Remat(bits),
                regalloc::Where::Spilled => Binding::from(
                    slot[v.0 as usize].unwrap_or_else(|| unreachable!("just given a slot")),
                ),
            });
        }

        Ok(Self {
            locs,
            slot,
            frame_size: frame.frame_size(),
            slots,
        })
    }

    /// Where `v` lives when the allocator says `at`.
    ///
    /// The arrow this type *is*: [`regalloc::Where`] says a value is in a slot,
    /// and this says which one. Total for every value with an address —
    /// `resolve` gave one to each value that spills anywhere in this scope,
    /// and the driver pins a hoist slot for each value an enclosing scope
    /// parked.
    ///
    /// # Panics
    /// If `at` is `Spilled` and `v` has no slot in this frame.
    #[must_use]
    pub fn binding(&self, v: regalloc::ValueId, at: regalloc::Where) -> Binding {
        match at {
            regalloc::Where::Reg(r) => Binding::from(Reg(r.0)),
            regalloc::Where::Remat(bits) => Binding::Remat(bits),
            regalloc::Where::Spilled => Binding::from(self.slot_of(v).unwrap_or_else(|| {
                panic!("{v:?} is spilled somewhere in this scope but has no slot")
            })),
        }
    }

    /// The slot of `v`, if it has one here.
    #[must_use]
    pub fn slot_of(&self, v: regalloc::ValueId) -> Option<Slot> {
        self.slot.get(v.0 as usize).copied().flatten()
    }

    /// Where `v` lives.
    ///
    /// # Panics
    /// If `v` is not in the allocation this was resolved from.
    #[must_use]
    pub fn of(&self, v: regalloc::ValueId) -> Binding {
        self.locs
            .get(v.0 as usize)
            .copied()
            .flatten()
            .unwrap_or_else(|| panic!("{v:?} has no binding in this frame"))
    }

    /// Every value's binding, dense by `ValueId.0`, for the hot emit loop.
    #[must_use]
    pub fn bindings(&self) -> &[Option<Binding>] {
        &self.locs
    }

    /// Give `v` a slot this frame did not lay out.
    ///
    /// The collapse-loop LICM parks a hoisted value in a slot the enclosing
    /// prologue wrote, which outlives every region's frame — so a scope inside
    /// reads and writes *that* address rather than one of its own. Only the
    /// address is pinned: where the value is at each point remains the
    /// placement's answer.
    pub fn pin_slot(&mut self, v: regalloc::ValueId, slot: Slot) {
        let idx = v.0 as usize;
        if idx >= self.slot.len() {
            self.slot.resize(idx + 1, None);
        }
        self.slot[idx] = Some(slot);
    }
}

/// A concrete instruction to emit, with all registers resolved.
/// Pure data — no side effects, no mutation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedOp {
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
    /// Decomposed multiply-add: FMUL(dst, a, b) then reload c, then FADD(dst, dst, c).
    /// Used when a and b are both spilled (can't load both + c simultaneously).
    /// `c_deferred`: if Some, c must be reloaded *after* FMUL.
    DecomposedMulAdd {
        dst: Reg,
        a: Reg,
        b: Reg,
        c: Reg,
        c_deferred: Option<DeferredReload>,
    },
    /// BSL select: dst = mask ? if_true : if_false (mask pre-loaded into dst).
    Select {
        dst: Reg,
        if_true: Reg,
        if_false: Reg,
    },
    /// Bound-memory gather: `dst = buffer[slot][idx_lane]`. Every backend
    /// implements it: AVX-512 natively (`vgatherdps`), AVX-2 as two scalar
    /// halves, SSE2 and NEON as four scalar loads. The buffer base pointer is
    /// loaded from the context struct (rdi) at `slot * 8`.
    Gather { dst: Reg, idx: Reg, slot: u16 },
    /// Uniform broadcast: `dst = splat(block[offset])`. The block's base
    /// pointer is loaded from the context struct at `ctx_slot * 8` — the
    /// entry after the last buffer — and the scalar at `4 * offset` is
    /// broadcast to every lane: `vbroadcastss` on every x86 tier, `ldr s` +
    /// `dup` on NEON. Its variance is `CONST`, so it lands in the per-call
    /// prologue.
    Uniform { dst: Reg, load: UniformLoad },
}

/// Where one uniform lives, relative to the context the kernel is called with.
///
/// Two immediates, both fixed at compile time: which context entry holds the
/// block (always the one past the kernel's buffer slots, so a kernel with no
/// uniforms has no such entry and its context is exactly what it was), and
/// the uniform's dense offset within the block, assigned by the link step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UniformLoad {
    /// Index into the context array of the block's base pointer.
    pub ctx_slot: u16,
    /// Index of the value within the block, in `f32`s.
    pub offset: u16,
}

/// A deferred reload: value loaded mid-instruction (after a partial computation).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeferredReload {
    /// Load from stack slot.
    FromStack(Slot),
    /// Rematerialize a constant.
    Const(u32),
}

/// Reload instruction: load a value into a register.
///
/// Either reload from stack (spilled) or rematerialize a constant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reload {
    /// Load from stack slot.
    FromStack { target: Reg, slot: Slot },
    /// Rematerialize a constant (emit FMOV immediate).
    Const { target: Reg, val_bits: u32 },
}

/// Fully resolved instruction: what to reload, and what to compute.
///
/// No store. A destination is always a register now, so the one place a value
/// reaches its slot is the emit loop's store-after-definition — which is what
/// makes the slot valid on every path a `Select` guard can take.
#[derive(Clone, Debug)]
pub struct InstructionPlan {
    /// Reloads to emit before the main op.
    pub reloads: Vec<Reload>,
    /// The main operation.
    pub op: ResolvedOp,
    /// Optional MOV to set up accumulator/mask before main op.
    pub setup_mov: Option<(Reg, Reg)>,
    /// The registers the encoding may destroy for the length of this
    /// instruction.
    ///
    /// Filled exactly as far as the backend asked
    /// ([`regalloc::RegisterFile::temps_for`]); the allocator picked them, so
    /// each holds no live value and is nobody's operand, and all are free again
    /// at the next instruction. An encoding that needs scratch must read this
    /// rather than a `const`, because there is no register reserved for it.
    pub scratch: regalloc::Scratch,
}

/// Where one operand of an instruction is read from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OperandSource {
    /// Already in a register; the location table says which.
    Resident,
    /// Not in a register, and reloaded into the **destination**.
    ///
    /// Sound because every backend here reads all of an instruction's sources
    /// before writing its destination, and free because these are the operands
    /// an encoding needs in the destination anyway: a `Select`'s mask, an
    /// FMA's addend, and a two-operand binary's left, which `dst op= right`
    /// consumes from the destination by definition.
    Destination,
    /// Not in a register, and reloaded into the `k`'th register the allocator
    /// reserved for this instruction ([`regalloc::Scratch::reload`]).
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
pub fn operand_sources(op: &ScheduledOp, resident: [bool; 3]) -> [OperandSource; 3] {
    // The operand an encoding wants in the destination, if any. `MulAdd`'s
    // answer depends on which form the emitter will choose, and it chooses by
    // residency — the decomposed `FMUL`/`FADD` when both multiplicands need
    // reloading, the fused form otherwise — which is the same question this
    // one is answering.
    let into_dst = match op {
        ScheduledOp::Binary(..) => Some(0),
        ScheduledOp::Ternary(OpKind::MulAdd, ..) if !resident[0] && !resident[1] => Some(0),
        ScheduledOp::Ternary(OpKind::MulAdd, ..) => Some(2),
        ScheduledOp::Ternary(OpKind::Select, ..) => Some(0),
        _ => None,
    };
    let arity = match op {
        ScheduledOp::Var(_) | ScheduledOp::Const(_) | ScheduledOp::Uniform(_) => 0,
        ScheduledOp::Unary(..) | ScheduledOp::ShiftImm(..) | ScheduledOp::Gather(..) => 1,
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
pub fn reloads_wanted(sources: [OperandSource; 3]) -> usize {
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
pub(crate) fn declared_temp(temp: Option<Reg>) -> Reg {
    temp.expect("this encoding needs a temp that `RegisterFile::temps_for` did not ask for")
}

/// Emission context with register budget for ML training.
#[derive(Clone, Debug, Default)]
pub struct EmitCtx {
    /// Cap on the allocatable scratch pool, or `None` to use the whole thing.
    ///
    /// Only ever *shrinks* the selected backend's own pool (see
    /// [`regalloc::RegisterFile::capped`]); setting it low is how a caller
    /// forces spilling deliberately.
    ///
    /// `None` rather than "a number at least as large as every pool": that
    /// spelling was a convention no type enforced, and it broke the moment the
    /// pools grew — a default of 10 silently capped AVX-512's 22 registers
    /// back to 10 while its doc comment still claimed it "caps nothing".
    pub max_regs: Option<u8>,
}

impl EmitCtx {
    /// Create context with custom register budget.
    #[must_use]
    pub fn with_max_regs(max_regs: u8) -> Self {
        Self {
            max_regs: Some(max_regs),
        }
    }

    /// Compile an [`ExprArena`] DAG under this configuration.
    ///
    /// The configured spelling of [`compile`]. It is a method rather than a
    /// `compile_with_ctx` free function because the suffix was only ever
    /// standing in for a receiver: the config is the thing that varies, so the
    /// config is what should be on the left.
    ///
    /// # Errors
    ///
    /// If the arena contains a construct no pass can lower, or the emitter
    /// cannot allocate a frame for it.
    pub fn compile(
        self,
        arena: &pixelflow_ir::arena::ExprArena,
        root: pixelflow_ir::arena::ExprId,
    ) -> Result<CompileResult, CompileError> {
        let (arena, root) =
            pixelflow_ir::passes::legalize(arena, root).map_err(CompileError::Legalize)?;
        let schedule = arena_to_schedule(&arena, root);
        compile_via_backend(schedule, &mut Native::new(self))
    }
}

/// The coordinate inputs, in order: X, Y, Z, W.
///
/// Both ABIs deliver the four vector arguments in the first four vector
/// registers, so this half of every [`RegisterFile`] is genuinely shared.
const INPUT_REGS: [Reg; 4] = [Reg(0), Reg(1), Reg(2), Reg(3)];

// =============================================================================
// Functional Emitter (x86-64)
// =============================================================================

// =============================================================================
// High-level API
// =============================================================================

/// Compile result with metadata for ML training.
///
pub struct CompileResult {
    /// The executable code.
    pub code: executable::ExecutableCode,
    /// Number of spills performed.
    pub spill_count: u32,
    /// Total stack space used for spills (bytes).
    pub spill_bytes: u32,
    /// Register budget that was used.
    pub max_regs: u8,
    /// X-invariant values hoisted out of the collapse loop into the
    /// once-per-call prologue (0 for per-batch kernels, and for collapse
    /// kernels with nothing to hoist).
    pub hoisted_values: u32,
    /// What was emitted, per scope of the collapse nest — the static half of
    /// a cost model's inputs. Counted, never optimized: see
    /// [`traffic`](self::traffic).
    pub traffic: EmitTraffic,
}

/// The architecture seam for the shared driver.
///
/// [`compile_via_backend`] owns the architecture-INDEPENDENT logic — schedule,
/// register allocation, frame layout, and the Select short-circuit control flow
/// — and calls an `IsaBackend` for the leaf operations that actually differ
/// between x86-64 and aarch64 (instruction encoding, branch encoding, the
/// collapse-loop scaffold, and any arch-specific finalization such as
/// aarch64's constant pool). Both backends therefore run the *same* driver: there is one
/// place that decides when to emit a guard branch, where the root goes, etc.
///
/// `Branch` is an opaque per-backend fixup token (aarch64 distinguishes CBZ from
/// B; x86 uses a uniform rel32), patched later by `patch_branch`.
trait IsaBackend {
    type Branch;

    /// This backend's register file: the whole of what allocation and frame
    /// layout need to know about the target.
    ///
    /// Backends declare it as a `const` next to their encodings and clamp its
    /// scratch pool to [`EmitCtx::max_regs`] at construction. It is the only
    /// target-dependent input to any of the shared logic here.
    fn register_file(&self) -> regalloc::RegisterFile;

    /// Per-compile setup before any code is emitted (e.g. seed a constant pool).
    fn begin(&mut self, schedule: &[regalloc::Def]) -> Result<(), CompileError>;

    /// Called once the frame layout is known, BEFORE any body instruction is
    /// emitted. Backends whose spill addressing depends on the frame mode
    /// (x86: red zone vs allocated frame) latch it here; `prologue` runs
    /// after the body is produced and can only prepend bytes.
    fn frame_ready(&mut self, _frame_size: u32) {}

    /// Emit one resolved instruction (with its reloads/store).
    fn emit_plan(&mut self, code: &mut Vec<u8>, plan: &InstructionPlan)
    -> Result<(), CompileError>;

    /// Register-to-register move.
    fn emit_mov(&mut self, code: &mut Vec<u8>, dst: Reg, src: Reg);

    /// Spill a register to a frame slot.
    fn emit_store(&mut self, code: &mut Vec<u8>, src: Reg, offset: u32)
    -> Result<(), CompileError>;

    /// Resolve a value to a register, reloading or rematerializing into
    /// `target` if it is not already in one.
    fn emit_resolve(
        &mut self,
        code: &mut Vec<u8>,
        vid: regalloc::ValueId,
        target: Reg,
        locs: &[Option<Binding>],
    ) -> Reg;

    /// Branch taken when `mask_reg` is all-false (skip the true arm).
    ///
    /// `scratch` is a vector register the backend may destroy, present exactly
    /// when its [`RegisterFile::guard_temps`](regalloc::RegisterFile::guard_temps)
    /// asked for one. Only aarch64 does — reducing a mask with `UMAXV`/`UMINV`
    /// writes a scalar into a vector register before it can reach a GP
    /// register — so the x86 tiers, whose guards go through
    /// `movmskps`/`kortest` and the flags, receive `None` and want nothing.
    fn emit_skip_if_all_false(
        &mut self,
        code: &mut Vec<u8>,
        mask_reg: Reg,
        scratch: Option<Reg>,
    ) -> Self::Branch;
    /// Branch taken when `mask_reg` is all-true (skip the false arm). See
    /// [`IsaBackend::emit_skip_if_all_false`] for `scratch`.
    fn emit_skip_if_all_true(
        &mut self,
        code: &mut Vec<u8>,
        mask_reg: Reg,
        scratch: Option<Reg>,
    ) -> Self::Branch;
    /// Unconditional jump.
    fn emit_jump(&mut self, code: &mut Vec<u8>) -> Self::Branch;
    /// Patch a previously emitted branch to land at `target`.
    fn patch_branch(&mut self, code: &mut Vec<u8>, branch: Self::Branch, target: usize);

    // -------------------------------------------------------------------------
    // Collapse-loop scaffold
    //
    // The verbs below exist only to serve `emit_collapse_loop`, which is a
    // provided method: the loop nest, its branch fixups and its coordinate
    // stepping are written once, here, and every backend gets the same one.
    // What a backend supplies is the meaning of each verb on its ISA.
    // -------------------------------------------------------------------------

    /// How many bytes the *body's own* spill frame occupies inside the
    /// scaffold's allocation, given the layout's frame size.
    ///
    /// Defaults to that size. x86-64 overrides it: in red-zone mode the body
    /// spills below `rsp` and allocates nothing, so the scaffold's coordinate
    /// slots start at zero.
    fn body_frame_bytes(&self, frame_size: u32) -> u32 {
        frame_size
    }

    /// Reserve / release `bytes` of stack.
    fn frame_alloc(&mut self, code: &mut Vec<u8>, bytes: u32);
    fn frame_free(&mut self, code: &mut Vec<u8>, bytes: u32);

    /// Anchor whatever the body's constant loads are relative to, once the
    /// frame exists. Default: nothing to anchor (x86 const loads are
    /// self-contained).
    fn scaffold_anchor(&mut self, _code: &mut Vec<u8>) {}

    /// Append whatever must trail the emitted function — a constant pool and
    /// the fixup that points at it. Default: nothing trails.
    fn scaffold_finish(&mut self, _code: &mut Vec<u8>) {}

    /// Save / restore one of the scaffold's coordinate slots.
    ///
    /// Distinct from [`IsaBackend::emit_store`], which addresses the *body's*
    /// spill slots and may reach into x86's red zone. These are always at a
    /// positive offset from the stack pointer.
    fn slot_store(&mut self, code: &mut Vec<u8>, src: Reg, offset: u32);
    fn slot_load(&mut self, code: &mut Vec<u8>, dst: Reg, offset: u32);

    /// Move the caller's loop bounds somewhere the body cannot clobber.
    /// Default: the ABI already put them out of the body's way.
    fn latch_bounds(&mut self, _code: &mut Vec<u8>) {}

    /// `counter = 0`.
    fn counter_clear(&mut self, code: &mut Vec<u8>, counter: Counter);
    /// `counter += 1`.
    fn counter_step(&mut self, code: &mut Vec<u8>, counter: Counter);
    /// Branch taken once `counter` has reached the bound it is compared against.
    fn branch_if_counter_done(&mut self, code: &mut Vec<u8>, counter: Counter) -> Self::Branch;

    /// Store one batch of results through the output pointer.
    fn store_result(&mut self, code: &mut Vec<u8>, src: Reg);
    /// Advance the output pointer.
    fn advance_out(&mut self, code: &mut Vec<u8>, step: OutStep);

    /// `dst += scalar` across every lane, clobbering `scratch`.
    fn add_scalar(&mut self, code: &mut Vec<u8>, dst: Reg, scratch: Reg, scalar: f32);

    /// Function return.
    fn emit_ret(&mut self, code: &mut Vec<u8>);

    /// Wrap a [`CollapseBody`] in the collapse loop scaffold, producing a
    /// complete [`KernelFn`](executable::KernelFn): the
    /// caller's lane-sequential X is an induction value stepped by the batch
    /// width in the inner loop and reset for each row; Y advances by 1.0 in
    /// the outer loop. Each batch's result is stored straight to the output
    /// pointer. The body's branches are self-relative, so inlining it inside
    /// the loop is sound.
    ///
    /// Coordinate state lives in stack slots above the body's spill frame:
    /// the ABI's vector registers are caller-saved scratch to the body, so
    /// each iteration reloads the input registers from the slots and the X
    /// slot alone is stepped.
    ///
    /// The scaffold moves [`INPUT_COORDS`] of them and a body reads two: the
    /// ABI still carries the base coordinates that were Z and W, the caller
    /// passes zero in both, and no arena that became a `Kernel` can name
    /// them. Dropping them changes this scaffold's own stores and loads, and
    /// so every kernel's bytes — L2's step, not L1's
    /// (docs/plans/2026-09-06-lattice-is-the-index.md).
    ///
    /// The two LICM tiers in [`CollapseBody`] park their results in vector
    /// slots directly above the coordinate slots reserved here.
    fn emit_collapse_loop(&mut self, emitted: &CollapseBody<'_>) -> Vec<u8> {
        let vw = self.register_file().vector_bytes;
        let base = self.body_frame_bytes(emitted.frame_size);
        let total = base + (COORD_SLOTS + emitted.hoist_slots) * vw;
        let slot = |k: u32| base + k * vw;
        let mut code: Vec<u8> = Vec::with_capacity(
            emitted.frame_hoist.len()
                + emitted.row_hoist.len()
                + emitted.batch.len()
                + SCAFFOLD_HEADROOM,
        );

        self.frame_alloc(&mut code, total);
        self.scaffold_anchor(&mut code);
        for k in 0..INPUT_COORDS {
            self.slot_store(&mut code, coord_reg(k), slot(k));
        }
        self.slot_store(&mut code, coord_reg(SLOT_X), slot(SLOT_ROW_START_X));
        // Frame LICM: X/Y-invariant values, computed once per call.
        code.extend_from_slice(emitted.frame_hoist);
        self.latch_bounds(&mut code);
        self.counter_clear(&mut code, Counter::Row);

        let row_top = code.len();
        let rows_done = self.branch_if_counter_done(&mut code, Counter::Row);

        // Row LICM: X-invariant values, recomputed once per row. Reload the
        // coordinates first — the previous body and Y-step clobbered them.
        for k in 0..INPUT_COORDS {
            self.slot_load(&mut code, coord_reg(k), slot(k));
        }
        code.extend_from_slice(emitted.row_hoist);
        self.counter_clear(&mut code, Counter::Batch);

        let batch_top = code.len();
        let batches_done = self.branch_if_counter_done(&mut code, Counter::Batch);

        for k in 0..INPUT_COORDS {
            self.slot_load(&mut code, coord_reg(k), slot(k));
        }
        code.extend_from_slice(emitted.batch);

        self.store_result(&mut code, emitted.result);
        self.advance_out(&mut code, OutStep::Batch);

        // X += one batch of lanes. The coordinate registers are reloaded at
        // the top of the next iteration, so they are free scratch here.
        let lanes = (vw / BYTES_PER_LANE) as f32;
        self.slot_load(&mut code, SCAFFOLD_ACC, slot(SLOT_X));
        self.add_scalar(&mut code, SCAFFOLD_ACC, SCAFFOLD_SCRATCH, lanes);
        self.slot_store(&mut code, SCAFFOLD_ACC, slot(SLOT_X));

        self.counter_step(&mut code, Counter::Batch);
        let repeat_batch = self.emit_jump(&mut code);
        self.patch_branch(&mut code, repeat_batch, batch_top);

        let row_end = code.len();
        self.patch_branch(&mut code, batches_done, row_end);

        // Reset X, advance Y, and skip any scalar tail in the output row.
        self.slot_load(&mut code, SCAFFOLD_ACC, slot(SLOT_ROW_START_X));
        self.slot_store(&mut code, SCAFFOLD_ACC, slot(SLOT_X));
        self.slot_load(&mut code, SCAFFOLD_ACC, slot(SLOT_Y));
        self.add_scalar(&mut code, SCAFFOLD_ACC, SCAFFOLD_SCRATCH, 1.0);
        self.slot_store(&mut code, SCAFFOLD_ACC, slot(SLOT_Y));
        self.advance_out(&mut code, OutStep::RowSkip);

        self.counter_step(&mut code, Counter::Row);
        let repeat_row = self.emit_jump(&mut code);
        self.patch_branch(&mut code, repeat_row, row_top);

        let end = code.len();
        self.patch_branch(&mut code, rows_done, end);
        self.frame_free(&mut code, total);
        self.emit_ret(&mut code);
        self.scaffold_finish(&mut code);
        code
    }
}

/// The emitted code a collapse loop wraps: the per-batch body, plus the two
/// LICM tiers lifted out of it and the framing they were laid out against.
///
/// One emit pass produces all six together, and the scaffold needs all six —
/// which is what makes them one argument rather than six.
struct CollapseBody<'a> {
    /// X/Y-invariant code, emitted once per call.
    frame_hoist: &'a [u8],
    /// X-invariant code, re-emitted at the top of every row.
    row_hoist: &'a [u8],
    /// The per-batch body proper.
    batch: &'a [u8],
    /// Where the batch leaves its result.
    result: Reg,
    /// Bytes of spill frame the body was laid out against.
    frame_size: u32,
    /// Vector slots the two hoist tiers park their roots in, directly above
    /// the scaffold's coordinate slots.
    hoist_slots: u32,
}

/// Which of the collapse loop's two counters a scaffold verb addresses.
///
/// Each is compared against a bound the caller passed in a register, which is
/// why the backend — not the scaffold — knows where either lives.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Counter {
    /// Batches within a row, against the caller's group count.
    Batch,
    /// Rows, against the caller's row count.
    Row,
}

/// How far the output pointer moves.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum OutStep {
    /// Past the batch just written — one vector width.
    Batch,
    /// Past whatever tail the row has beyond its last full batch.
    RowSkip,
}

/// Coordinate slots the scaffold reserves above the body's frame: the four
/// the ABI passes, plus a copy of the row's starting X.
const COORD_SLOTS: u32 = 5;
/// The leading slots that are reloaded into the ABI's input registers.
///
/// Four, of which a body reads two: a lattice has X and Y, and the last two
/// base coordinates are passed as zero and named by nothing that reaches the
/// emitter. See [`IsaBackend::emit_collapse_loop`] for why they are still
/// moved.
const INPUT_COORDS: u32 = 4;
const SLOT_X: u32 = 0;
const SLOT_Y: u32 = 1;
/// Where the row's starting X is kept so the inner loop's stepping can be undone.
const SLOT_ROW_START_X: u32 = 4;
/// Slack for the scaffold's own instructions on top of the code it wraps.
const SCAFFOLD_HEADROOM: usize = 160;
/// A lane is one `f32`.
const BYTES_PER_LANE: u32 = 4;

/// The register a coordinate slot is passed and reloaded in. Every ABI here
/// puts the four base coordinates in the first four vector registers, in
/// that order; only the first two are ever read.
const fn coord_reg(slot: u32) -> Reg {
    Reg(slot as u8)
}

/// Scratch the scaffold's own arithmetic uses between iterations. Both
/// registers hold coordinates inside the body, but every coordinate is
/// reloaded from its slot at the top of each iteration, so the scaffold is
/// free to clobber them once the body has run.
const SCAFFOLD_ACC: Reg = Reg(0);
const SCAFFOLD_SCRATCH: Reg = Reg(1);
/// Allocate a straight-line schedule and emit it as a region body.
///
/// Production compiles allocate the whole nest at once
/// ([`regalloc::RegisterAllocator::allocate_nest`]) so every region's frame
/// is known before any of them is emitted; this is the one-region
/// convenience the emitter's own tests are written against.
#[cfg(test)]
fn emit_dag_body<B: IsaBackend>(
    schedule: Vec<regalloc::Def>,
    backend: &mut B,
) -> Result<(Vec<u8>, Reg, u32, u32), CompileError> {
    use regalloc::RegisterAllocator;
    let nest = regalloc::LinearScan.allocate(schedule, &backend.register_file());
    emit_dag_body_hoisted(nest.body(), backend, HoistCtx::None, None)
}

/// Emit one region's body from a finished allocation, with collapse-loop
/// LICM support: a hoist map (see
/// [`HoistCtx`]) and an optional frame-size override. The override replaces
/// the layout's frame size in the `frame_ready` latch and the returned frame
/// size — the collapse driver passes the max of the prologue's and body's
/// frames so both address the shared hoist slots consistently (and, on x86,
/// so both latch the same allocated-frame mode).
fn emit_dag_body_hoisted<B: IsaBackend>(
    allocation: regalloc::Allocation<'_>,
    backend: &mut B,
    hoist: HoistCtx<'_>,
    frame_override: Option<u32>,
) -> Result<(Vec<u8>, Reg, u32, u32), CompileError> {
    use alloc::collections::BTreeMap;

    let file = backend.register_file();
    // Allocation happened before this call — once per region, over the whole
    // nest. The allocator chooses the evaluation order, so everything here —
    // guard ranges, program points, the emit loop itself — reads the schedule
    // it handed back rather than the one it was given.
    let schedule = allocation.schedule();
    let mut layout = FrameLayout::resolve(allocation, file.vector_bytes)?;
    let real_spill_count = layout.slots;

    // A value an enclosing region parked has no address in this frame — its
    // slot is the driver's hoist slot, which outlives every region's frame.
    // Only the address is pinned: whether the value is in that slot or in a
    // register, at each point, is the placement's answer.
    if let Some(hoisted) = hoist.preloaded() {
        for (vid, &offset) in hoisted {
            layout.pin_slot(*vid, Slot::new(offset, file.vector_bytes));
        }
    }

    let frame_size = frame_override.unwrap_or(layout.frame_size);
    if frame_size < layout.frame_size {
        return Err(CompileError::Internal(
            "frame override smaller than the layout's frame",
        ));
    }
    backend.frame_ready(frame_size);

    // Select short-circuit guards (disabled in the prologue — see HoistCtx).
    let select_guards = if hoist.parks_values() {
        Vec::new()
    } else {
        analyze_select_guards(schedule)
    };
    let sched_len = schedule.len();

    struct PendingBranch {
        guard_idx: usize,
        arm: SelectArm,
    }
    let mut branch_starts: alloc::vec::Vec<alloc::vec::Vec<PendingBranch>> =
        (0..sched_len).map(|_| alloc::vec::Vec::new()).collect();
    let mut branch_ends: alloc::vec::Vec<alloc::vec::Vec<usize>> =
        (0..sched_len).map(|_| alloc::vec::Vec::new()).collect();
    for (gi, guard) in select_guards.iter().enumerate() {
        for arm in SelectArm::ALL {
            let range = guard.range(arm);
            if range.0 != range.1 {
                branch_starts[range.0].push(PendingBranch { guard_idx: gi, arm });
                if range.1 < sched_len {
                    branch_ends[range.1].push(gi);
                }
            }
        }
    }

    // One dense ValueId -> Binding lookup for the hot loop, carried *forward*: a
    // placement is a schedule, so the answer changes at program points, and
    // this is that schedule played out. Each range of each value's life
    // becomes one write here at the point it starts — O(total ranges), not a
    // lookup per operand per instruction.
    let mut locs: alloc::vec::Vec<Option<Binding>> = layout.bindings().to_vec();
    let mut moves: alloc::vec::Vec<alloc::vec::Vec<(regalloc::ValueId, Binding)>> =
        (0..sched_len).map(|_| alloc::vec::Vec::new()).collect();
    // A value that is in a slot anywhere in this scope is stored there right
    // after its definition, from the register the definition wrote. That is
    // the whole of the slot-validity rule: a definition dominates every read,
    // and a `Select` guard that skips a definition skips all of its readers
    // too, so there is no path on which a read finds the slot unwritten.
    let mut store_after_def: alloc::vec::Vec<Option<u32>> = alloc::vec![None; sched_len];
    for (i, def) in schedule.iter().enumerate() {
        let v = def.value;
        if hoist.preloaded().is_some_and(|h| h.contains_key(&v)) {
            // Live-in: an enclosing scope left it somewhere, and the head
            // reconciliation below brings it to where this scope expects it.
            continue;
        }
        for (index, at) in allocation.transitions(v) {
            if index <= i {
                continue; // The definition itself; the instruction writes it.
            }
            moves[index].push((v, layout.binding(v, at)));
        }
        if let Some(slot) = layout.slot_of(v)
            && matches!(locs[v.0 as usize], Some(Binding::Loc(Loc::Reg(_))))
        {
            // Every definition writes a register, so this is the only place a
            // value reaches its slot — and it is the place that makes the slot
            // valid on both sides of every guard.
            store_after_def[i] = Some(slot.offset());
        }
    }

    backend.begin(schedule)?;

    // No prologue here — the caller frames the body (see the fn doc).
    let mut code: Vec<u8> = Vec::new();

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
    // Walked over the schedule, not over the map: an enclosing region parks
    // every root it computes, and a scope inside reads only the subset that
    // reaches it.
    if let Some(hoisted) = hoist.preloaded() {
        for vid in schedule.iter().map(|def| def.value) {
            if !hoisted.contains_key(&vid) {
                continue;
            }
            let placement = allocation.placement(vid);
            let at_head = allocation.where_at(vid, 0);
            let head = layout.binding(vid, at_head);
            if let Binding::Loc(Loc::Reg(r)) = head
                && placement.at(regalloc::Point::TAIL) != at_head
            {
                let from_memory = placement
                    .locations()
                    .find(|at| !matches!(at, regalloc::Where::Reg(_)))
                    .unwrap_or_else(|| {
                        unreachable!("a value that never leaves a register never changes register")
                    });
                locs[vid.0 as usize] = Some(layout.binding(vid, from_memory));
                let got = backend.emit_resolve(&mut code, vid, r, &locs);
                debug_assert_eq!(got, r, "a value out of a register reloads into the target");
            }
            locs[vid.0 as usize] = Some(head);
        }
    }

    let mut pending_patches: BTreeMap<(usize, SelectArm), B::Branch> = BTreeMap::new();

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
        for &gi in &branch_ends[sched_idx] {
            let target = code.len();
            for arm in SelectArm::ALL {
                if let Some(branch) = pending_patches.remove(&(gi, arm)) {
                    backend.patch_branch(&mut code, branch, target);
                }
            }
        }

        // Ranges that begin here. A register range starting away from the
        // value's definition is a reload the allocator chose to keep: the
        // value comes back into a pool register and stays there, instead of
        // being fetched into a scratch at every read.
        for (v, to) in core::mem::take(&mut moves[sched_idx]) {
            if let Binding::Loc(Loc::Reg(r)) = to {
                let src = backend.emit_resolve(&mut code, v, r, &locs);
                if src != r {
                    backend.emit_mov(&mut code, r, src);
                }
            }
            locs[v.0 as usize] = Some(to);
        }

        // The registers this instruction's own guards may use: the allocator
        // reserved them here because a guard runs *between* instructions, at
        // a point the schedule does contain — the head of the arm it skips,
        // and the `Select` that owns it.
        let scratch = allocation.scratch(sched_idx);
        let guard_mask = || {
            scratch.guard_mask.expect(
                "a guard's mask is not in a register and the allocator \
                 reserved nothing to reload it into",
            )
        };
        let guard_temp = scratch.guard_temp;

        // Guard branches that begin before this instruction.
        for pb in &branch_starts[sched_idx] {
            let (guard_idx, arm) = (pb.guard_idx, pb.arm);
            let guard = &select_guards[guard_idx];
            let mask_reg = match location_of(&locs, guard.mask_vid) {
                Binding::Loc(Loc::Reg(r)) => r,
                _ => backend.emit_resolve(&mut code, guard.mask_vid, guard_mask(), &locs),
            };
            let branch = match arm {
                SelectArm::True => backend.emit_skip_if_all_false(&mut code, mask_reg, guard_temp),
                SelectArm::False => backend.emit_skip_if_all_true(&mut code, mask_reg, guard_temp),
            };
            pending_patches.insert((guard_idx, arm), branch);
        }

        // A hoisted value's placeholder def emits nothing — the prologue
        // already parked the value in its slot; consumers reload from there.
        if let Some(hoisted) = hoist.preloaded()
            && hoisted.contains_key(vid)
        {
            continue;
        }

        let dst_loc = location_of(&locs, *vid);
        let plan = resolve_operands(sched_op, dst_loc, &locs, scratch)?;

        // Select with a guard region: emit a uniform-mask short-circuit wrapper.
        if let ScheduledOp::Ternary(OpKind::Select, mask_vid, true_vid, false_vid) = sched_op
            && let Some(guard) = select_guards.iter().find(|g| g.select_idx == sched_idx)
            && guard.has_guarded_arm()
        {
            let mask_reg = match location_of(&locs, *mask_vid) {
                Binding::Loc(Loc::Reg(r)) => r,
                _ => backend.emit_resolve(&mut code, *mask_vid, guard_mask(), &locs),
            };
            let dst = dst_loc.reg();
            let in_reg = |v: regalloc::ValueId| match location_of(&locs, v) {
                Binding::Loc(Loc::Reg(r)) => Some(r),
                _ => None,
            };
            let true_reg = in_reg(*true_vid);
            let false_reg = in_reg(*false_vid);

            // Both guards read `mask_reg`, which is why the reduction
            // scratch is a reservation of its own rather than whichever
            // register the mask was resolved into.
            let all_false = backend.emit_skip_if_all_false(&mut code, mask_reg, guard_temp);
            let all_true = backend.emit_skip_if_all_true(&mut code, mask_reg, guard_temp);

            // Mixed lanes: the real select.
            backend.emit_plan(&mut code, &plan)?;
            let skip_end = backend.emit_jump(&mut code);

            // All-false: dst <- false arm.
            let all_false_target = code.len();
            if let Some(freg) = false_reg {
                backend.emit_mov(&mut code, dst, freg);
            } else {
                backend.emit_resolve(&mut code, *false_vid, dst, &locs);
            }
            let skip_end2 = backend.emit_jump(&mut code);

            // All-true: dst <- true arm.
            let all_true_target = code.len();
            if let Some(treg) = true_reg {
                backend.emit_mov(&mut code, dst, treg);
            } else {
                backend.emit_resolve(&mut code, *true_vid, dst, &locs);
            }

            let end_target = code.len();
            backend.patch_branch(&mut code, all_false, all_false_target);
            backend.patch_branch(&mut code, all_true, all_true_target);
            backend.patch_branch(&mut code, skip_end, end_target);
            backend.patch_branch(&mut code, skip_end2, end_target);

            if let Some(offset) = store_after_def[sched_idx] {
                backend.emit_store(&mut code, dst, offset)?;
            }
            continue;
        }

        backend.emit_plan(&mut code, &plan)?;

        if let Some(offset) = store_after_def[sched_idx] {
            backend.emit_store(&mut code, dst_loc.reg(), offset)?;
        }

        // Prologue mode: hand each hoist root over to the scopes inside, right
        // after its def, while the value is guaranteed live. (Guards are
        // disabled in this mode, so every def reaches this point — the
        // guarded-Select early-continue above cannot fire.)
        if let Some(hoisted) = hoist.parked()
            && let Some(&offset) = hoisted.get(vid)
        {
            // Resident by construction: a hoist root is a computed value, not
            // a leaf (`plan_collapse_hoist` refuses to hoist one), so its own
            // definition — the instruction just emitted — wrote it into a
            // register. There is nothing to resolve.
            let r = dst_loc.reg();
            // The slot is written unless nothing inside will ever read it —
            // which is exactly the case where the value holds one register at
            // every point of every scope within. Read off the placement, not
            // off a flag beside it.
            let inside = allocation.inner_head();
            let head = allocation.placement(*vid).at(inside);
            let resident_throughout = matches!(head, regalloc::Where::Reg(_))
                && allocation
                    .placement(*vid)
                    .spans()
                    .all(|s| s.from <= inside || s.at == head);
            if !resident_throughout {
                backend.emit_store(&mut code, r, offset)?;
            }
            if let regalloc::Where::Reg(head_reg) = head
                && head_reg != r
            {
                backend.emit_mov(&mut code, head_reg, r);
            }
        }
    }

    assert!(
        pending_patches.is_empty(),
        "BUG: {} Select short-circuit branches were never patched",
        pending_patches.len()
    );

    // The scope's result, in a register for the scaffold to store. Usually the
    // last instruction's own destination; not when the body's root was hoisted
    // out entirely and is read from its park, which is what the allocator
    // reserved a target on the last instruction for.
    let root = schedule
        .last()
        .map(|def| def.value)
        .expect("empty schedule");
    let result_reg = match location_of(&locs, root) {
        Binding::Loc(Loc::Reg(r)) => r,
        _ => {
            let target = allocation
                .scratch(sched_len - 1)
                .result
                .expect("the allocator reserves a result target on every scope's last instruction");
            backend.emit_resolve(&mut code, root, target, &locs)
        }
    };

    Ok((code, result_reg, frame_size, real_spill_count))
}

/// Info about an operation in the schedule.
#[derive(Debug, Clone)]
pub enum ScheduledOp {
    /// Variable reference (input register)
    Var(u8),
    /// Constant value
    Const(f32),
    /// Unary op with input value
    Unary(OpKind, regalloc::ValueId),
    /// Binary op with input values
    Binary(OpKind, regalloc::ValueId, regalloc::ValueId),
    /// Ternary op with input values
    Ternary(
        OpKind,
        regalloc::ValueId,
        regalloc::ValueId,
        regalloc::ValueId,
    ),
    /// Bit-shift by a compile-time immediate: `op` is `Shl` or `Shr`, the value
    /// is `ValueId`, and the shift count is folded out of the `Const` RHS by
    /// `arena_to_schedule` (so it never becomes a scheduled value / register).
    ShiftImm(OpKind, regalloc::ValueId, u8),
    /// Bound-memory gather: read buffer `slot` at the lane index computed by the
    /// value operand. Lowered from `RawGather(Buffer(slot), index)`; the buffer
    /// leaf is folded out to the `slot` immediate (like `ShiftImm`'s count) so it
    /// never becomes a scheduled value. The index is the one real input.
    Gather(regalloc::ValueId, u16),
    /// Per-call scalar, broadcast from the block: a definition with no
    /// operands — like `Const`, but not a leaf to the hoisting partition,
    /// since the load is an instruction worth doing once per call rather
    /// than once per batch.
    Uniform(UniformLoad),
}

// =============================================================================
// Arena to Schedule (zero-cost linearization)
// =============================================================================

/// Mark nodes reachable from `root` via DFS.
///
/// The arena may contain garbage nodes from junkify passes; only nodes
/// transitively referenced by `root` should appear in the schedule.
fn mark_reachable(
    arena: &pixelflow_ir::arena::ExprArena,
    root: pixelflow_ir::arena::ExprId,
    reachable: &mut [bool],
) {
    let mut stack = alloc::vec![root];
    while let Some(id) = stack.pop() {
        let idx = id.0 as usize;
        if reachable[idx] {
            continue;
        }
        reachable[idx] = true;
        for child in arena.children(id) {
            if !reachable[child.0 as usize] {
                stack.push(child);
            }
        }
    }
}

/// Narrow a `Const` shift count to the `u8` immediate the hardware encoders
/// take, refusing anything a 32-bit lane cannot be shifted by.
///
/// The check belongs HERE, on the `f32`, because the narrowing is lossy in a
/// way that manufactures a legal-looking value: `256.0 as u32 as u8` is `0`,
/// so a count no target can honour would arrive at the encoder disguised as
/// the identity shift. Any later validation is checking the alias, not the
/// operand the kernel actually asked for.
fn shift_immediate(op: OpKind, count: f32) -> u8 {
    assert!(
        (0.0..32.0).contains(&count) && (count as u32) as f32 == count,
        "{op:?} shift count {count} is not an integer in 0..32 — a 32-bit lane \
         has no bits there, and the targets disagree about what to do (x86 \
         zeroes the whole destination, aarch64 re-encodes the element size)"
    );
    count as u8
}

/// Build a schedule directly from an [`ExprArena`].
///
/// The arena stores nodes in topological order (children before parents by
/// construction). We filter to reachable nodes, remap `ExprId` to `ValueId`,
/// and translate `ExprNode` to `ScheduledOp`.
///
/// # Panics
///
/// Panics if a `Param` or `Nary` node is encountered (these are not expected
/// in JIT compilation).
fn arena_to_schedule(
    arena: &pixelflow_ir::arena::ExprArena,
    root: pixelflow_ir::arena::ExprId,
) -> Vec<regalloc::Def> {
    use pixelflow_ir::arena::{ExprId, ExprNode};
    use regalloc::ValueId;

    let len = arena.len();
    let mut reachable = alloc::vec![false; len];
    mark_reachable(arena, root, &mut reachable);

    // ExprId to ValueId mapping. u32::MAX = unmapped (unreachable).
    let mut id_map = alloc::vec![ValueId(u32::MAX); len];
    let mut schedule = Vec::new();
    let mut next_id = 0u32;

    for idx in 0..len {
        if !reachable[idx] {
            continue;
        }
        let expr_id = ExprId(idx as u32);
        let node = arena.node(expr_id);
        let vid = ValueId(next_id);
        next_id += 1;
        id_map[idx] = vid;

        let map_child = |child: &ExprId| -> ValueId {
            let mapped = id_map[child.0 as usize];
            assert!(
                mapped.0 != u32::MAX,
                "arena_to_schedule: child ExprId({}) not yet mapped -- \
                 arena is not in topological order or child is unreachable",
                child.0
            );
            mapped
        };

        let sched_op = match node {
            ExprNode::Var(i) => ScheduledOp::Var(*i),
            ExprNode::Const(v) => ScheduledOp::Const(*v),
            ExprNode::Param(i) => panic!(
                "ExprNode::Param({}) reached the JIT emitter -- \
                 call substitute_params before compile()",
                i
            ),
            // A Buffer leaf is always folded into a `Gather`'s `slot` immediate
            // (below), so any Buffer that survives as its own reachable node is a
            // dead operand — never consumed as a value. Emit a harmless dead
            // placeholder occupying its ValueId slot, exactly as ShiftImm leaves
            // its folded shift-count Const as a dead schedule entry.
            ExprNode::Buffer(_) => ScheduledOp::Const(0.0),
            // The block pointer sits in the context entry after the buffer
            // slots; the value's offset is its slot index — the link step
            // (`jit_cache`) renumbers the table into dense first-occurrence
            // order before anything reaches here, and a caller compiling an
            // arena directly gets the table order it declared.
            ExprNode::Uniform(u) => ScheduledOp::Uniform(UniformLoad {
                ctx_slot: u16::try_from(arena.buffers().len())
                    .expect("buffer table index fits the context slot immediate"),
                offset: u.0,
            }),
            ExprNode::Unary(op, child) => ScheduledOp::Unary(*op, map_child(child)),
            // Shl/Shr fold their Const shift-count operand into an immediate, so
            // the count never becomes a scheduled value (matching the imm-only
            // hardware shift encoders). The count const may still appear as its
            // own schedule entry (harmless/unused) if shared.
            ExprNode::Binary(op @ (OpKind::Shl | OpKind::Shr), a, b) => {
                let amount = match arena.node(*b) {
                    ExprNode::Const(v) => shift_immediate(*op, *v),
                    _ => panic!(
                        "{:?} shift count must be a Const (lowering guarantees this)",
                        op
                    ),
                };
                ScheduledOp::ShiftImm(*op, map_child(a), amount)
            }
            // RawGather folds its Buffer leaf into the `slot` immediate (like a
            // shift count); only the index operand becomes a scheduled value.
            ExprNode::Binary(OpKind::RawGather, buf, idx) => {
                let slot = match arena.node(*buf) {
                    ExprNode::Buffer(id) => id.0,
                    other => panic!("RawGather's first child must be a Buffer leaf, got {other:?}"),
                };
                ScheduledOp::Gather(map_child(idx), slot)
            }
            // Unreachable precondition: every compile entry point runs
            // `passes::lower_dwrt` before scheduling, which either rewrites
            // all `Dwrt` (autodiff) nodes into chain-rule arithmetic or errors
            // loudly on an op it cannot differentiate. A `Dwrt` here means a
            // caller bypassed that pipeline. Fail loudly rather than as a
            // cryptic instruction-emit panic.
            ExprNode::Binary(OpKind::Dwrt, _, _) => panic!(
                "arena_to_schedule: a Dwrt (autodiff) node reached the JIT \
                 emitter. lower_dwrt runs in every compile entry point and \
                 either eliminates Dwrt or refuses to compile, so a survivor \
                 means this schedule was built without the lowering pipeline."
            ),
            ExprNode::Binary(op, a, b) => ScheduledOp::Binary(*op, map_child(a), map_child(b)),
            ExprNode::Ternary(op, a, b, c) => {
                ScheduledOp::Ternary(*op, map_child(a), map_child(b), map_child(c))
            }
            ExprNode::Nary(_, _, _) => panic!("Nary not supported in JIT arena compilation"),
        };
        schedule.push(regalloc::Def {
            value: vid,
            op: sched_op,
        });
    }
    schedule
}

// =============================================================================
// Collapse-loop LICM (X-invariant hoisting)
// =============================================================================

/// Compute [`Variance`](pixelflow_ir::variance::Variance) for every schedule entry.
///
/// The schedule mirrors the arena's topological order, so one forward pass
/// suffices — the dense result is indexed by `ValueId.0`.
fn schedule_variance(schedule: &[regalloc::Def]) -> Vec<pixelflow_ir::variance::Variance> {
    use pixelflow_ir::variance::Variance;
    let max_vid = schedule.iter().map(|def| def.value.0).max().unwrap_or(0) as usize;
    let mut v = alloc::vec![Variance::CONST; max_vid + 1];
    for def in schedule {
        let (vid, op) = (&def.value, &def.op);
        let i = vid.0 as usize;
        v[i] = match op {
            ScheduledOp::Var(idx) if *idx < 8 => Variance::from_var(*idx),
            ScheduledOp::Var(_) => Variance::ALL,
            // Invariant across the lattice; unknown until the call. The
            // `CONST` here is what carries it into the per-call prologue.
            ScheduledOp::Const(_) | ScheduledOp::Uniform(_) => Variance::CONST,
            ScheduledOp::Unary(_, a)
            | ScheduledOp::ShiftImm(_, a, _)
            // A gather reads from a bound buffer, whose contents are fixed for
            // the kernel's lifetime — its variance is its index's variance.
            | ScheduledOp::Gather(a, _) => v[a.0 as usize],
            ScheduledOp::Binary(_, a, b) => v[a.0 as usize].union(v[b.0 as usize]),
            ScheduledOp::Ternary(_, a, b, c) => v[a.0 as usize]
                .union(v[b.0 as usize])
                .union(v[c.0 as usize]),
        };
    }
    v
}

/// The collapse loop's LICM partition: which values leave the X loop, and the
/// two schedules that result.
///
/// `roots[i]` is parked in hoist slot `i`. `prologue` computes the roots (the
/// full X-invariant sub-DAG, original order); `body` is the loop schedule with
/// each root's entry replaced by a `Const(0.0)` placeholder — never emitted,
/// its location overridden to the hoist slot so consumers reload it through
/// the ordinary spill machinery.
struct HoistPlan {
    roots: Vec<regalloc::ValueId>,
    prologue: Vec<regalloc::Def>,
    body: Vec<regalloc::Def>,
}

/// Partition a collapse schedule for LICM.
///
/// A hoist root is an X-invariant, non-leaf value consumed by at least one
/// X-dependent op (or the schedule root itself, when the whole kernel is
/// X-invariant — the loop degenerates to a store). `Gather`s — and anything
/// computed from one — are never hoisted: hoisting moves a value out of any
/// select-guard arm it sits in, and while speculating arithmetic is free,
/// keeping memory reads exactly where the per-batch kernel had them costs
/// nothing today (winding kernels are gather-free). A `Uniform` load is the
/// one memory read that *is* hoisted: it is invariant for the whole call, it
/// cannot fault, and loading it once is the entire point of the leaf.
///
/// Returns `None` when nothing qualifies, leaving the caller on the plain
/// un-hoisted path.
fn plan_collapse_hoist(
    schedule: &[regalloc::Def],
    variance: &[pixelflow_ir::variance::Variance],
    scope_mask: u8,
) -> Option<HoistPlan> {
    use regalloc::ValueId;
    let n = schedule.len();
    if n == 0 {
        return None;
    }
    let max_vid = schedule.iter().map(|def| def.value.0).max().unwrap_or(0) as usize;

    let operands = |op: &ScheduledOp| -> alloc::vec::Vec<ValueId> {
        match op {
            ScheduledOp::Var(_) | ScheduledOp::Const(_) | ScheduledOp::Uniform(_) => {
                alloc::vec![]
            }
            ScheduledOp::Unary(_, a)
            | ScheduledOp::ShiftImm(_, a, _)
            | ScheduledOp::Gather(a, _) => {
                alloc::vec![*a]
            }
            ScheduledOp::Binary(_, a, b) => alloc::vec![*a, *b],
            ScheduledOp::Ternary(_, a, b, c) => alloc::vec![*a, *b, *c],
        }
    };

    // Which values are consumed by an op varying inside this scope, and which contain a
    // gather anywhere in their sub-DAG (forward pass — schedule is topological).
    let mut feeds_varying = alloc::vec![false; max_vid + 1];
    let mut contains_gather = alloc::vec![false; max_vid + 1];
    for def in schedule {
        let (vid, op) = (&def.value, &def.op);
        let i = vid.0 as usize;
        let ops = operands(op);
        contains_gather[i] = matches!(op, ScheduledOp::Gather(_, _))
            || ops.iter().any(|a| contains_gather[a.0 as usize]);
        if variance[i].bits() & scope_mask != 0 {
            for a in &ops {
                feeds_varying[a.0 as usize] = true;
            }
        }
    }

    let is_leaf = |op: &ScheduledOp| matches!(op, ScheduledOp::Var(_) | ScheduledOp::Const(_));
    let root_vid = schedule.last().map(|def| def.value)?;

    let mut is_root = alloc::vec![false; max_vid + 1];
    let mut roots: Vec<ValueId> = Vec::new();
    for def in schedule {
        let (vid, op) = (&def.value, &def.op);
        let i = vid.0 as usize;
        let hoistable = variance[i].bits() & scope_mask == 0
            && !is_leaf(op)
            && !contains_gather[i]
            && (feeds_varying[i] || *vid == root_vid);
        if hoistable {
            is_root[i] = true;
            roots.push(*vid);
        }
    }
    if roots.is_empty() {
        return None;
    }

    // Prologue: the transitive operand closure of the roots (all X-invariant
    // by construction), kept in original topological order.
    let mut in_prologue = alloc::vec![false; max_vid + 1];
    for r in &roots {
        in_prologue[r.0 as usize] = true;
    }
    for def in schedule.iter().rev() {
        if in_prologue[def.value.0 as usize] {
            for a in operands(&def.op) {
                in_prologue[a.0 as usize] = true;
            }
        }
    }
    let prologue: Vec<_> = schedule
        .iter()
        .filter(|def| in_prologue[def.value.0 as usize])
        .cloned()
        .collect();

    // Body: backward reachability from the schedule root, treating hoist roots
    // as leaves (their entries become placeholders; operands not followed).
    let mut in_body = alloc::vec![false; max_vid + 1];
    in_body[root_vid.0 as usize] = true;
    for def in schedule.iter().rev() {
        let i = def.value.0 as usize;
        if in_body[i] && !is_root[i] {
            for a in operands(&def.op) {
                in_body[a.0 as usize] = true;
            }
        }
    }
    let body: Vec<_> = schedule
        .iter()
        .filter(|def| in_body[def.value.0 as usize])
        .map(|def| {
            if is_root[def.value.0 as usize] {
                // placeholder; never emitted
                regalloc::Def {
                    value: def.value,
                    op: ScheduledOp::Const(0.0),
                }
            } else {
                def.clone()
            }
        })
        .collect();

    // Keep only roots the body actually reads (an interior invariant value
    // consumed solely by other hoisted values needs no slot). The schedule
    // root always keeps its slot — the loop stores it.
    let roots: Vec<ValueId> = roots
        .into_iter()
        .filter(|r| in_body[r.0 as usize])
        .collect();
    if roots.is_empty() {
        return None;
    }

    Some(HoistPlan {
        roots,
        prologue,
        body,
    })
}

/// Split a schedule by scope over `binders`, given innermost first.
///
/// One rule, applied once per binder from the outside in: a value is lifted
/// out of a binder when its variance does not name that binder or any binder
/// inside it. That is loop-invariant code motion, hoisting out of a
/// reduction, and constant folding — the same question asked at each level,
/// which is why this is a loop over binders rather than a tier per scope.
///
/// The lifted roots of an outer region are leaves to every region inside it,
/// so each level sees a strictly smaller schedule and the last remainder is
/// the per-sample body.
fn partition_by_scope(
    schedule: Vec<regalloc::Def>,
    variance: &[pixelflow_ir::variance::Variance],
    binders: &[u8],
) -> regalloc::ScopedSchedule {
    let mut remaining = schedule;
    let mut regions = Vec::with_capacity(binders.len());
    // Outermost first: the scope outside binder `j` cannot depend on `j` or
    // on anything bound inside it.
    for j in (0..binders.len()).rev() {
        let mask = binders[..=j].iter().fold(0u8, |m, b| m | (1 << b));
        match plan_collapse_hoist(&remaining, variance, mask) {
            Some(plan) => {
                remaining = plan.body;
                regions.push(regalloc::ScopeRegion {
                    roots: plan.roots,
                    schedule: plan.prologue,
                });
            }
            None => regions.push(regalloc::ScopeRegion {
                roots: Vec::new(),
                schedule: Vec::new(),
            }),
        }
    }
    regalloc::ScopedSchedule {
        regions,
        body: remaining,
    }
}

/// How [`emit_dag_body_hoisted`] treats hoisted values, if any.
enum HoistCtx<'a> {
    /// No hoisting (per-batch kernels, and collapse kernels with nothing to
    /// hoist).
    None,
    /// Emitting the once-per-call prologue: after each mapped value's def,
    /// store it to its hoist slot. Select short-circuit guards are disabled —
    /// a guard could skip a hoist root's def on a uniform mask, leaving its
    /// slot garbage for the loop to read (and the prologue runs once, so the
    /// guard buys nothing).
    Prologue {
        /// Values parked by an enclosing loop and reloaded as leaves here.
        preloaded: Option<&'a alloc::collections::BTreeMap<regalloc::ValueId, u32>>,
        /// Values this prologue computes and parks for its inner loop.
        parked: &'a alloc::collections::BTreeMap<regalloc::ValueId, u32>,
    },
    /// Emitting the loop body: mapped values are never emitted; their
    /// locations are overridden — to a carried register where the allocator
    /// found one, and otherwise to the hoist slot, where every consumer
    /// reloads through the ordinary spill machinery.
    Body {
        slots: &'a alloc::collections::BTreeMap<regalloc::ValueId, u32>,
    },
}

impl<'a> HoistCtx<'a> {
    fn preloaded(&self) -> Option<&'a alloc::collections::BTreeMap<regalloc::ValueId, u32>> {
        match self {
            Self::None => None,
            Self::Prologue { preloaded, .. } => *preloaded,
            Self::Body { slots, .. } => Some(slots),
        }
    }

    fn parked(&self) -> Option<&'a alloc::collections::BTreeMap<regalloc::ValueId, u32>> {
        match self {
            Self::Prologue { parked, .. } => Some(parked),
            Self::None | Self::Body { .. } => None,
        }
    }

    fn parks_values(&self) -> bool {
        matches!(self, Self::Prologue { .. })
    }
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
pub fn resolve_operands(
    op: &ScheduledOp,
    dst_loc: Binding,
    locs: &[Option<Binding>],
    scratch: regalloc::Scratch,
) -> Result<InstructionPlan, CompileError> {
    let dst = match dst_loc {
        Binding::Loc(Loc::Reg(r)) => r,
        // A rematerialized constant: it lives nowhere and is rebuilt at each
        // use, so its definition computes nothing. Emitting a load into a
        // register nobody reads is what the fixed destination register used to
        // buy.
        Binding::Remat(_) => {
            return Ok(InstructionPlan {
                reloads: Vec::new(),
                op: ResolvedOp::Nop,
                setup_mov: None,
                scratch,
            });
        }
        Binding::Loc(Loc::Slot(slot)) => panic!(
            "a definition landed in stack slot {} — the allocator owes \
             every definition a register, since there is none outside the pool \
             to compute into",
            slot.offset()
        ),
    };

    let mut reloads = Vec::new();
    let mut setup_mov = None;

    // Resolve a value to its register, or plan a reload from stack/constant into `target`.
    let loc_of = |v: regalloc::ValueId| -> Binding {
        locs.get(v.0 as usize)
            .copied()
            .flatten()
            .unwrap_or_else(|| panic!("{v:?} has no binding"))
    };
    // "Not in a register" — a rematerialized value needs a reload target just
    // as a spilled one does, so both answer false here.
    let in_register = |v: &regalloc::ValueId| matches!(loc_of(*v), Binding::Loc(Loc::Reg(_)));

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
        match loc_of(v) {
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
        }
    };
    // Operand `k`, from wherever it is: its own register, or the one
    // [`operand_sources`] reserved for it.
    let operand = |k: usize, v: regalloc::ValueId, reloads: &mut Vec<Reload>| -> Reg {
        match sources[k] {
            OperandSource::Resident => loc_of(v).reg(),
            OperandSource::Destination | OperandSource::Reload(_) => {
                resolve(v, target_for(k), reloads)
            }
        }
    };

    let resolved_op = match op {
        ScheduledOp::Var(_) => {
            // Precolored to input register — no code needed.
            ResolvedOp::Nop
        }
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
        ScheduledOp::Gather(child, slot) => {
            let idx = operand(0, *child, &mut reloads);
            ResolvedOp::Gather {
                dst,
                idx,
                slot: *slot,
            }
        }
        ScheduledOp::Uniform(load) => ResolvedOp::Uniform { dst, load: *load },
        ScheduledOp::Binary(op_kind, left, right) => {
            // `left` goes to `dst` when it needs reloading — the two-operand
            // form consumes it from there anyway — and `right` to a
            // reservation.
            let l_reg = operand(0, *left, &mut reloads);
            let r_reg = operand(1, *right, &mut reloads);
            // The two-operand invariant, stated where the registers are
            // chosen rather than defended in the one backend that has no
            // three-operand form. SSE2's `mulps dst, src` computes
            // `dst <- left; dst op= right`, which corrupts `right` when
            // `dst == right` and `dst != left`.
            //
            // That assignment cannot arise. `dst` is a pool register the
            // allocator gave this definition, disjoint by construction from
            // every register this instruction reads: `right` is either a pool
            // register a live operand holds — which a destination never takes
            // — an input register, or one of this instruction's own reload
            // reservations, which the destination is excluded from.
            //
            // So `left` may alias `dst` and the backends may write the
            // destructive form directly — but if the allocator ever stops
            // guaranteeing this, the failure is a silently corrupted operand,
            // which is what this restates in every debug build.
            debug_assert!(
                dst != r_reg || dst == l_reg,
                "{op_kind:?}: dst {dst:?} aliases the right operand without \
                 aliasing the left — the two-operand form would corrupt it"
            );
            ResolvedOp::Binary {
                op: *op_kind,
                dst,
                left: l_reg,
                right: r_reg,
            }
        }
        ScheduledOp::Ternary(op_kind, a, b, c) => {
            let a_spilled = !in_register(a);
            let b_spilled = !in_register(b);

            match op_kind {
                OpKind::MulAdd => {
                    // MulAdd(a, b, c) = a*b + c.
                    if a_spilled && b_spilled {
                        // Decompose: FMUL(dst, a, b) then FADD(dst, dst, c).
                        // `a` lands in `dst`, which the multiply consumes it
                        // from; `b` and `c` each take a reservation of their
                        // own, so deferring `c` past the multiply no longer
                        // depends on `b` having been consumed by then.
                        let a_reg = operand(0, *a, &mut reloads);
                        let b_reg = operand(1, *b, &mut reloads);
                        // c is deferred — don't add to upfront reloads.
                        let (c_reg, c_deferred) = match loc_of(*c) {
                            Binding::Loc(Loc::Reg(reg)) => (reg, None),
                            Binding::Remat(bits) => {
                                (target_for(2), Some(DeferredReload::Const(bits)))
                            }
                            Binding::Loc(Loc::Slot(slot)) => {
                                (target_for(2), Some(DeferredReload::FromStack(slot)))
                            }
                        };
                        ResolvedOp::DecomposedMulAdd {
                            dst,
                            a: a_reg,
                            b: b_reg,
                            c: c_reg,
                            c_deferred,
                        }
                    } else {
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
                }
                OpKind::Select => {
                    // BSL/blend is a 3-input RMW: the mask must end up in `dst`,
                    // and if_true / if_false each need their own live register.
                    //
                    // A spilled mask reloads STRAIGHT into `dst`, which is what
                    // `operand_sources` says for operand 0 here. Every reload
                    // emits before `setup_mov`, so routing the mask through a
                    // register a spilled arm also reloads into would overwrite
                    // it before it reached `dst`; one reservation per arm is
                    // why that cannot happen. Both arms spilled at once used
                    // to need a third fixed register (`select_reload`), held
                    // out of every kernel's pool for the rare kernel reaching
                    // it.
                    let a_reg = operand(0, *a, &mut reloads);
                    if dst.0 != a_reg.0 {
                        setup_mov = Some((dst, a_reg));
                    }
                    let b_reg = operand(1, *b, &mut reloads);
                    let c_reg = operand(2, *c, &mut reloads);
                    ResolvedOp::Select {
                        dst,
                        if_true: b_reg,
                        if_false: c_reg,
                    }
                }
                _ => return Err(CompileError::UnsupportedOp(*op_kind)),
            }
        }
    };

    Ok(InstructionPlan {
        reloads,
        op: resolved_op,
        setup_mov,
        scratch,
    })
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

/// The backend this build emits for.
///
/// Every [`IsaBackend`] compiles on every host — emission is a pure function of
/// `(schedule, RegisterFile)` into a `Vec<u8>`, and an x86 machine is perfectly
/// capable of computing NEON instruction words. So the target does not decide
/// which backends *exist*; it decides which one is *instantiated*, here, once.
///
/// `Native` is a concrete type, so the driver monomorphizes against it exactly
/// as it did when each backend was `#[cfg]`-gated into existence: static
/// dispatch, no `dyn`, no vtable. What changes is that the other three are
/// still typechecked, still swept for op coverage, and still unit-testable on
/// this host — which is what a `#[cfg]` around their definitions was quietly
/// costing.
///
/// Genuinely host-bound code lives in [`executable`] (the `KernelFn` ABI types
/// and the `mmap`/`mprotect` that makes bytes callable) and nowhere else.
#[cfg(target_arch = "aarch64")]
type Native = aarch64::driver::Aarch64Backend;
/// See the aarch64 variant above.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
type Native = avx512::driver::Avx512Backend;
/// See the aarch64 variant above.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
type Native = avx2::driver::Avx2Backend;
/// See the aarch64 variant above.
#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx2"),
    not(target_feature = "avx512f")
))]
type Native = x86_64::driver::X86Backend;

/// Compile an [`ExprArena`] DAG into a **collapse** kernel: the X/Y loop nest is
/// emitted *inside* the code, so one call fills `rows * groups` output batches
/// with no per-row or per-batch Rust↔JIT boundary. This is the internal-loop
/// realization of a lattice collapse.
///
/// The per-batch body (produced by [`emit_dag_body_hoisted`], with derivatives /
/// reductions / gathers / transcendentals already lowered) is wrapped in the
/// build width's
/// [`IsaBackend::emit_collapse_loop`] scaffold: X steps by the batch width and
/// resets per row, Y steps by 1.0, the two dead base coordinates stay as the
/// caller passed them, gathers read buffer bases from the context register,
/// and each batch stores straight to `out`.
/// Matches the
/// [`KernelFn`](executable::KernelFn) ABI
/// `(ctx, out, groups, rows, row_skip_bytes, x0, y0, z, w)`.
///
/// The context is one base pointer per declared buffer, in the arena's slot
/// order, followed — only when the arena declares a uniform — by the uniform
/// block's base pointer: `f32` values in the arena's uniform-slot order, read
/// once per call in the frame prologue.
///
/// # Panics
///
/// Panics if the arena names a retired coordinate axis (`Var(2)`/`Var(3)`,
/// the old Z and W). This is the boundary the check belongs on, because it
/// is the *only* one every route to machine code passes through — the
/// shape-keyed cache is one caller, and the benchmark harnesses, the corpus
/// tools and several tests come straight here. It is also the *diagnostic*
/// place: a panic naming `Var(2)` at the first `cargo test` is worth far
/// more than what the alternative produces, which is a silent numeric
/// disagreement between this kernel and the scalar oracle, surfacing on
/// whichever machine happens to run the comparison.
///
/// A retired axis reaching here is not merely unread. The scaffold passes
/// zero in those two lanes, and `Variance::from_var(2)` sits outside both
/// `COORDS` and `BINDERS` — so the node reads as frame-uniform and LICM
/// lifts it into the per-call prologue. Plausible pixels, computed once,
/// from a lane that means nothing.
pub fn compile(
    arena: &pixelflow_ir::arena::ExprArena,
    root: pixelflow_ir::arena::ExprId,
) -> Result<CompileResult, CompileError> {
    assert!(
        arena.retired_axis(root).is_none(),
        "emit::compile: the arena names Var({:?}), a coordinate axis a \
         lattice no longer has; a per-call scalar is a Uniform",
        arena.retired_axis(root)
    );
    EmitCtx::default().compile(arena, root)
}

/// Drive a schedule to a complete collapse kernel via an
/// [`IsaBackend`]: the body from [`emit_dag_body_hoisted`], framed by the backend's
/// [`IsaBackend::emit_collapse_loop`] scaffold.
fn compile_via_backend<B: IsaBackend>(
    schedule: Vec<regalloc::Def>,
    backend: &mut B,
) -> Result<CompileResult, CompileError> {
    use regalloc::RegisterAllocator;

    let file = backend.register_file();
    let variance = schedule_variance(&schedule);

    // The collapse ABI's nest, innermost first: X steps by the batch width
    // and resets per row, Y steps by one. Z and W are per-call in this ABI,
    // so a value invariant in X and Y is invariant for the whole call.
    // `partition_by_scope` asks one question per binder; the two regions it
    // returns are the per-call and per-row prologues the scaffold frames.
    const COLLAPSE_BINDERS: [u8; 2] = [0, 1];
    let mut scoped = partition_by_scope(schedule, &variance, &COLLAPSE_BINDERS);
    // The body is the only scope whose selects are guarded (the prologues run
    // once, so a branch buys nothing there), and it is the schedule the guard
    // analysis will read — so this is where an arm's entries are worth
    // gathering into one run. A no-op unless it buys a branch.
    scoped.body = guards::cluster_select_arms(scoped.body);

    // One allocation pass over the whole nest. Each region's frame is a
    // function of its own allocation, so the shared frame below is read off
    // these rather than computed by allocating everything a second time.
    let nest = regalloc::LinearScan.allocate_nest(scoped, &file);
    assert_eq!(
        nest.regions(),
        COLLAPSE_BINDERS.len(),
        "one region per collapse binder"
    );
    let frame_alloc = nest.scope(regalloc::Scope::Region(0));
    let row_alloc = nest.scope(regalloc::Scope::Region(1));
    let body_alloc = nest.body();
    let (frame_roots, row_roots) = (frame_alloc.roots(), row_alloc.roots());

    // Every byte below is emitted through this decorator, so the counts it
    // hands back cover the whole function by construction (see `traffic`).
    let mut counting = Counting::new(backend);

    if frame_roots.is_empty() && row_roots.is_empty() {
        // Nothing loop-invariant worth hoisting: the plain loop nest.
        let (body, result_reg, frame_size, spill_count) =
            emit_dag_body_hoisted(body_alloc, &mut counting, HoistCtx::None, None)?;
        let body_traffic = counting.take(body.len() as u32);
        let code = counting.emit_collapse_loop(&CollapseBody {
            frame_hoist: &[],
            row_hoist: &[],
            batch: &body,
            result: result_reg,
            frame_size,
            hoist_slots: 0,
        });
        let scaffold = counting.take(code.len() as u32 - body.len() as u32);
        let exec = unsafe { executable::ExecutableCode::from_code(&code)? };
        return Ok(CompileResult {
            code: exec,
            spill_count,
            spill_bytes: frame_size,
            max_regs: file.scratch.len(),
            hoisted_values: 0,
            traffic: EmitTraffic {
                frame: ScopeTraffic::default(),
                row: ScopeTraffic::default(),
                body: body_traffic,
                scaffold,
                vector_bytes: file.vector_bytes,
                pool: file.scratch.len(),
                carried: 0,
            },
        });
    };

    // The two prologues and the loop body share one stack frame: spill slots
    // in [0, m), the scaffold's five coordinate slots (four base coordinates
    // plus row-start
    // X) at [m, m + 5·vector_bytes), and hoist slots above those. `m` is the
    // max of the three frames — each region is only live while its own code
    // runs, but the hoist slots outlive all of them. Allocation and frame
    // layout are pure, so pre-sizing here computes exactly the frames the
    // emissions below will.
    //
    // The floor keeps x86's SSE2 backend out of red-zone mode: hoist offsets
    // are far past the 128-byte zone, so both emissions must latch
    // allocated-frame (`[rsp + offset]`) addressing.
    const RED_ZONE_FLOOR: u32 = 144;
    let vector_bytes = file.vector_bytes;
    // Rounded to a whole slot so the scaffold's coordinate and hoist slots,
    // which sit at `m + k·vector_bytes`, stay naturally aligned.
    let mut m = RED_ZONE_FLOOR;
    for allocation in [frame_alloc, row_alloc, body_alloc] {
        if allocation.schedule().is_empty() {
            continue;
        }
        m = m.max(FrameLayout::resolve(allocation, vector_bytes)?.frame_size);
    }
    let m = m.next_multiple_of(vector_bytes);
    // Hoist slot k sits above the scaffold's five coordinate slots.
    let hoist_slot = |k: usize| m + (5 + k as u32) * vector_bytes;
    let frame_map: alloc::collections::BTreeMap<regalloc::ValueId, u32> = frame_roots
        .iter()
        .enumerate()
        .map(|(i, vid)| (*vid, hoist_slot(i)))
        .collect();
    let row_map: alloc::collections::BTreeMap<regalloc::ValueId, u32> = row_roots
        .iter()
        .enumerate()
        .map(|(i, vid)| (*vid, hoist_slot(frame_roots.len() + i)))
        .collect();
    let hoist_map: alloc::collections::BTreeMap<regalloc::ValueId, u32> = frame_map
        .iter()
        .chain(&row_map)
        .map(|(vid, offset)| (*vid, *offset))
        .collect();

    let (frame_code, frame_spills) = if frame_alloc.schedule().is_empty() {
        (Vec::new(), 0)
    } else {
        let (code, _, _, spills) = emit_dag_body_hoisted(
            frame_alloc,
            &mut counting,
            HoistCtx::Prologue {
                preloaded: None,
                parked: &frame_map,
            },
            Some(m),
        )?;
        (code, spills)
    };
    let frame_traffic = counting.take(frame_code.len() as u32);
    let (row_code, row_spills) = if row_alloc.schedule().is_empty() {
        (Vec::new(), 0)
    } else {
        let (code, _, _, spills) = emit_dag_body_hoisted(
            row_alloc,
            &mut counting,
            HoistCtx::Prologue {
                preloaded: if frame_map.is_empty() {
                    None
                } else {
                    Some(&frame_map)
                },
                parked: &row_map,
            },
            Some(m),
        )?;
        (code, spills)
    };
    let row_traffic = counting.take(row_code.len() as u32);
    let (body, result_reg, _, body_spills) = emit_dag_body_hoisted(
        body_alloc,
        &mut counting,
        HoistCtx::Body { slots: &hoist_map },
        Some(m),
    )?;
    let body_traffic = counting.take(body.len() as u32);

    let hoisted_values = (frame_roots.len() + row_roots.len()) as u32;
    let code = counting.emit_collapse_loop(&CollapseBody {
        frame_hoist: &frame_code,
        row_hoist: &row_code,
        batch: &body,
        result: result_reg,
        frame_size: m,
        hoist_slots: hoisted_values,
    });
    let emitted = (frame_code.len() + row_code.len() + body.len()) as u32;
    let scaffold = counting.take(code.len() as u32 - emitted);
    // A parked root that holds a register at the head of the scopes inside it
    // is carried rather than reloaded per iteration — read off the placement,
    // which is where the answer lives.
    let carried = [(frame_alloc, frame_roots), (row_alloc, row_roots)]
        .into_iter()
        .flat_map(|(alloc, roots)| {
            roots
                .iter()
                .filter(move |vid| alloc.carried(**vid).is_some())
        })
        .count() as u32;
    let exec = unsafe { executable::ExecutableCode::from_code(&code)? };
    Ok(CompileResult {
        code: exec,
        spill_count: frame_spills + row_spills + body_spills,
        spill_bytes: m,
        max_regs: file.scratch.len(),
        hoisted_values,
        traffic: EmitTraffic {
            frame: frame_traffic,
            row: row_traffic,
            body: body_traffic,
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
    use pixelflow_ir::arena::ExprArena;
    // Only the 128-bit x86 helpers (`run1`, `run_xy`, `run2`) take an `ExprId`
    // at this level; the wider-ISA submodules import their own.
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        not(target_feature = "avx2")
    ))]
    use pixelflow_ir::arena::ExprId;

    // The three helpers below are shared by ISA-gated tests; which subset is
    // live depends on the build's target features, so none is unconditionally
    // used.

    /// Lanes in one SIMD batch for this build.
    #[allow(dead_code)]
    const LANES: usize = crate::JIT_VECTOR_BYTES / core::mem::size_of::<f32>();

    /// Evaluate one batch at arbitrary per-lane coordinates.
    ///
    /// `V` is `[f32; LANES]`, which is the emitted vector type *by size*, so
    /// this needs neither intrinsics nor a hand-written `extern "C"` signature
    /// — the two things every caller in this file used to spell for itself,
    /// once per ISA. `ctx` may be empty: a kernel that declares no buffer never
    /// reads the pointer.
    #[allow(dead_code)]
    fn eval_batch(
        code: &executable::ExecutableCode,
        ctx: &[*const f32],
        origin: executable::Point4<[f32; LANES]>,
    ) -> [f32; LANES] {
        let mut out = [0.0f32; LANES];
        // SAFETY: `out` holds exactly one batch; size_of::<[f32; LANES]>() is
        // JIT_VECTOR_BYTES by construction; `ctx` binds every declared buffer.
        unsafe {
            code.call_collapse(
                ctx.as_ptr(),
                executable::TileSlice::single(out.as_mut_ptr()),
                origin,
            );
        }
        out
    }

    /// Evaluate at a single point (all lanes the same X).
    #[allow(dead_code)]
    fn eval_point(code: &executable::ExecutableCode, x: f32, y: f32, z: f32, w: f32) -> f32 {
        let o = executable::Point4::new([x; LANES], [y; LANES], [z; LANES], [w; LANES]);
        eval_batch(code, &[], o)[0]
    }

    /// A `Dwrt` that reaches the scheduler (a caller bypassed the lowering
    /// pipeline) must fail loudly at the schedule boundary, not as a cryptic
    /// emit panic. The compile entry points run `lower_dwrt` first, so this
    /// exercises calling `arena_to_schedule` directly.
    #[test]
    #[should_panic(expected = "Dwrt (autodiff) node reached the JIT")]
    fn surviving_dwrt_fails_loudly() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let v = a.push_const(0.0);
        let root = a.push_binary(OpKind::Dwrt, x, v);
        let _ = arena_to_schedule(&a, root);
    }

    /// The scaffold's size does not depend on the frame it wraps.
    ///
    /// Every backend now shares one `emit_collapse_loop`, so the loop nest's
    /// branch displacements are computed in exactly one place — and they are
    /// only correct if the instructions between two labels keep their widths
    /// as the frame grows. A slot displacement that silently widened from an
    /// 8-bit to a 32-bit form would move every label after it.
    ///
    /// Checking it needs no host CPU: emission is a pure function into
    /// `Vec<u8>`, so all four backends are measured from whichever host runs
    /// the tests.
    #[test]
    fn scaffold_size_is_independent_of_the_frame() {
        let ctx = EmitCtx::default();
        let batch: Vec<u8> = alloc::vec![0x90; 8];
        let fh: Vec<u8> = alloc::vec![0x90; 4];
        let rh: Vec<u8> = alloc::vec![0x90; 12];
        let wrapped = |result, frame_size, hoist_slots| CollapseBody {
            frame_hoist: &fh,
            row_hoist: &rh,
            batch: &batch,
            result,
            frame_size,
            hoist_slots,
        };
        // Red-zone and allocated frames, with and without hoisted values.
        const SHAPES: [(u32, u32); 3] = [(0, 0), (64, 2), (160, 3)];

        let mut sizes = alloc::vec::Vec::new();
        for (frame_size, hoist_slots) in SHAPES {
            let mut sse2 = x86_64::driver::X86Backend::new(ctx.clone());
            sse2.frame_ready(frame_size);
            let mut avx2b = avx2::driver::Avx2Backend::new(ctx.clone());
            avx2b.frame_ready(frame_size);
            let mut avx512b = avx512::driver::Avx512Backend::new(ctx.clone());
            avx512b.frame_ready(frame_size);
            let mut neon = aarch64::driver::Aarch64Backend::new(ctx.clone());
            neon.frame_ready(frame_size);

            let neon_code = neon.emit_collapse_loop(&wrapped(Reg(16), frame_size, hoist_slots));
            assert!(
                neon_code.len().is_multiple_of(4),
                "aarch64 is fixed-width, got {} bytes",
                neon_code.len()
            );
            sizes.push([
                sse2.emit_collapse_loop(&wrapped(Reg(4), frame_size, hoist_slots))
                    .len(),
                avx2b
                    .emit_collapse_loop(&wrapped(Reg(4), frame_size, hoist_slots))
                    .len(),
                avx512b
                    .emit_collapse_loop(&wrapped(Reg(4), frame_size, hoist_slots))
                    .len(),
                neon_code.len(),
            ]);
        }
        assert!(sizes[0].iter().all(|&n| n > 0), "every backend emits");
        assert_eq!(
            sizes[0], sizes[1],
            "frame mode must not resize the scaffold"
        );
        assert_eq!(
            sizes[1], sizes[2],
            "hoist slots must not resize the scaffold"
        );
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
        use pixelflow_ir::arena::{ExprArena, UniformDecl, UniformIdentity};
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
        let (a, root) = pixelflow_ir::passes::legalize(&a, root).expect("legalize");
        assert!(
            arena_to_schedule(&a, root)
                .iter()
                .any(|d| matches!(d.op, ScheduledOp::Uniform(_))),
            "the schedule must carry the uniform load for the backends to dispatch on"
        );

        let ctx = EmitCtx::default();
        let mut neon = aarch64::driver::Aarch64Backend::new(ctx.clone());
        let mut sse2 = x86_64::driver::X86Backend::new(ctx.clone());
        let mut avx2b = avx2::driver::Avx2Backend::new(ctx.clone());
        let mut avx512b = avx512::driver::Avx512Backend::new(ctx);

        let neon_len = emit_dag_body(arena_to_schedule(&a, root), &mut neon)
            .expect("NEON emit")
            .0
            .len();
        assert!(
            neon_len > 0 && neon_len.is_multiple_of(4),
            "aarch64 is fixed-width"
        );
        for (name, len) in [
            (
                "SSE2",
                emit_dag_body(arena_to_schedule(&a, root), &mut sse2)
                    .expect("SSE2")
                    .0
                    .len(),
            ),
            (
                "AVX2",
                emit_dag_body(arena_to_schedule(&a, root), &mut avx2b)
                    .expect("AVX2")
                    .0
                    .len(),
            ),
            (
                "AVX-512",
                emit_dag_body(arena_to_schedule(&a, root), &mut avx512b)
                    .expect("AVX-512")
                    .0
                    .len(),
            ),
        ] {
            assert!(len > 0, "{name} emitted nothing");
        }
    }

    /// The aarch64 constant pool must APPEND across the two bodies a collapse
    /// compile pushes through one backend, never reset.
    ///
    /// The prologue's bytes already have the first pool's X17-relative offsets
    /// baked in, so a reset leaves them pointing at different constants — the
    /// "macOS glyph-ink regression", which painted glyphs with the wrong ink
    /// and was only ever observable by running the app on a Mac.
    ///
    /// It is an aarch64 bug, not a macOS one, and now it is a sub-millisecond
    /// unit test on every host.
    #[test]
    fn aarch64_const_pool_appends_across_bodies() {
        use pixelflow_ir::arena::ExprArena;

        fn schedule_for(k: f32) -> Vec<regalloc::Def> {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let c = a.push_const(k);
            let root = a.push_binary(OpKind::Mul, x, c);
            let (a, root) = pixelflow_ir::passes::legalize(&a, root).expect("legalize");
            arena_to_schedule(&a, root)
        }

        // Two constants that genuinely need the pool (not FMOV-immediate).
        let (first, second) = (0.123_456_79_f32, 987.654_3_f32);
        assert!(aarch64::needs_const_pool(first));
        assert!(aarch64::needs_const_pool(second));

        let mut backend = aarch64::driver::Aarch64Backend::new(EmitCtx::default());
        emit_dag_body(schedule_for(first), &mut backend).expect("first body");
        let after_first = backend.pool_entries().to_vec();
        emit_dag_body(schedule_for(second), &mut backend).expect("second body");

        assert!(
            backend.pool_entries().starts_with(&after_first),
            "the second body RESET the constant pool: the first body's baked-in \
             X17-relative offsets now name different constants — the glyph-ink \
             regression. Pool was {after_first:?}, became {:?}",
            backend.pool_entries()
        );
    }

    // =========================================================================
    // What the nest does and does not partition
    // =========================================================================

    /// A leaf shared between an invariant expression and a varying one lands
    /// in **both** scopes' schedules, with a location chosen independently in
    /// each.
    ///
    /// It is tempting to assume `partition_by_scope` partitions `ValueId`s —
    /// `arena_to_schedule` numbers them sequentially, and every non-leaf is
    /// either lifted or left behind. Leaves are the exception: `plan_collapse_hoist`
    /// refuses to make one a hoist root (there is nothing to save by parking a
    /// value one instruction rebuilds), so a `Const` feeding both sides is
    /// simply computed twice. A nest-wide placement map that assumed one
    /// answer per value would have to pick one of the two, and the emitter
    /// would then read a register the other scope never wrote.
    #[test]
    fn a_leaf_feeding_both_scopes_is_scheduled_in_both() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        // One constant, read by an X-invariant term and an X-varying one.
        let k = a.push_const(3.5);
        let invariant = a.push_binary(OpKind::Mul, y, k);
        let varying = a.push_binary(OpKind::Mul, x, k);
        let root = a.push_binary(OpKind::Add, invariant, varying);

        let (arena, root) = pixelflow_ir::passes::legalize(&a, root).expect("legalize");
        let schedule = arena_to_schedule(&arena, root);
        let variance = schedule_variance(&schedule);
        let scoped = partition_by_scope(schedule, &variance, &[0u8, 1]);

        let in_body: alloc::vec::Vec<regalloc::ValueId> =
            scoped.body.iter().map(|d| d.value).collect();
        let shared: alloc::vec::Vec<regalloc::ValueId> = scoped
            .regions
            .iter()
            .flat_map(|r| r.schedule.iter().map(|d| d.value))
            .filter(|v| in_body.contains(v) && !scoped.regions.iter().any(|r| r.roots.contains(v)))
            .collect();

        assert!(
            !shared.is_empty(),
            "no value is scheduled in two scopes, so nothing here is testing \
             what a nest-wide placement map has to survive"
        );
    }

    /// The consequence for the allocator: a value in two scopes gets a range
    /// per scope, and each range is that scope's own answer.
    #[test]
    fn a_shared_leaf_is_placed_once_per_scope() {
        use regalloc::{RegisterAllocator, Scope};

        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let k = a.push_const(3.5);
        let invariant = a.push_binary(OpKind::Mul, y, k);
        let varying = a.push_binary(OpKind::Mul, x, k);
        let root = a.push_binary(OpKind::Add, invariant, varying);

        let (arena, root) = pixelflow_ir::passes::legalize(&a, root).expect("legalize");
        let schedule = arena_to_schedule(&arena, root);
        let variance = schedule_variance(&schedule);
        let scoped = partition_by_scope(schedule, &variance, &[0u8, 1]);

        let file = Native::new(EmitCtx::default()).register_file();
        let nest = regalloc::LinearScan.allocate_nest(scoped, &file);

        // Every value the body schedules has an answer at a body point, and
        // every value a region schedules has one at a point in that region —
        // which is exactly what a single answer per value could not give.
        let mut answered = 0;
        let mut scheduled = 0;
        for scope in [Scope::Region(0), Scope::Region(1), Scope::Body] {
            let view = nest.scope(scope);
            for (i, def) in view.schedule().iter().enumerate() {
                scheduled += 1;
                if matches!(
                    view.where_at(def.value, i),
                    regalloc::Where::Reg(_) | regalloc::Where::Spilled | regalloc::Where::Remat(_)
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

    // =========================================================================
    // FrameLayout unit tests — the Placement -> address arrow
    // =========================================================================

    /// Build an allocation with the given placements, in schedule order.
    fn allocation_of(placements: &[(u32, regalloc::Where)]) -> regalloc::NestAllocation {
        use regalloc::{Def, RegisterAllocator, ValueId};
        // Allocate a trivial all-Var schedule to get a well-formed Allocation,
        // then pin each value where the test wants it.
        let schedule: alloc::vec::Vec<Def> = placements
            .iter()
            .map(|&(v, _)| Def {
                value: ValueId(v),
                op: ScheduledOp::Var(0),
            })
            .collect();
        let mut a = regalloc::LinearScan.allocate(schedule, &TEST_FILE);
        for &(v, p) in placements {
            a.place(ValueId(v), p);
        }
        a
    }

    #[test]
    fn an_allocation_with_no_spills_needs_no_frame() {
        let a = allocation_of(&[(0, regalloc::Where::Reg(Reg(4)))]);
        let layout = FrameLayout::resolve(a.body(), 16).unwrap();
        assert_eq!(layout.frame_size, 0);
        assert_eq!(layout.of(regalloc::ValueId(0)), Loc::Reg(Reg(4)).into());
    }

    #[test]
    fn one_spill_takes_one_slot() {
        let a = allocation_of(&[(5, regalloc::Where::Spilled)]);
        let layout = FrameLayout::resolve(a.body(), 16).unwrap();
        assert_eq!(layout.frame_size, 16);
        assert_eq!(
            layout.of(regalloc::ValueId(5)),
            Loc::Slot(Slot::new(0, 16)).into()
        );
    }

    /// Slots are laid out at the backend's own stride, so the offsets a wide
    /// backend encodes are real displacements rather than 16-byte units it has
    /// to scale back up.
    #[test]
    fn slots_are_laid_out_at_the_backends_vector_stride() {
        let spilled = [
            (1, regalloc::Where::Spilled),
            (2, regalloc::Where::Spilled),
            (3, regalloc::Where::Spilled),
        ];
        for (vector_bytes, expected) in
            [(16u32, [0, 16, 32]), (32, [0, 32, 64]), (64, [0, 64, 128])]
        {
            let a = allocation_of(&spilled);
            let layout = FrameLayout::resolve(a.body(), vector_bytes).unwrap();
            assert_eq!(layout.frame_size, 3 * vector_bytes);
            for (i, off) in expected.iter().enumerate() {
                assert_eq!(
                    layout.of(regalloc::ValueId(i as u32 + 1)),
                    Loc::Slot(Slot::new(*off, vector_bytes)).into(),
                    "vector_bytes={vector_bytes}"
                );
            }
        }
    }

    /// A rematerialized constant occupies no slot at all.
    #[test]
    fn rematerialized_values_take_no_frame_space() {
        let a = allocation_of(&[
            (0, regalloc::Where::Remat(1.0f32.to_bits())),
            (1, regalloc::Where::Spilled),
        ]);
        let layout = FrameLayout::resolve(a.body(), 16).unwrap();
        assert_eq!(layout.frame_size, 16, "only the spill takes a slot");
        assert_eq!(
            layout.of(regalloc::ValueId(0)),
            Binding::Remat(1.0f32.to_bits())
        );
        assert_eq!(
            layout.of(regalloc::ValueId(1)),
            Loc::Slot(Slot::new(0, 16)).into()
        );
    }

    /// The collapse LICM pins a hoisted value to the slot its prologue wrote,
    /// which is not one this frame laid out.
    #[test]
    fn a_slot_can_be_pinned_over_the_frames_own_layout() {
        let a = allocation_of(&[(0, regalloc::Where::Reg(Reg(4)))]);
        let mut layout = FrameLayout::resolve(a.body(), 16).unwrap();
        let v = regalloc::ValueId(0);
        assert_eq!(layout.slot_of(v), None, "a resident value needs no slot");
        let pin = Slot::new(256, 16);
        layout.pin_slot(v, pin);
        assert_eq!(layout.slot_of(v), Some(pin));
        assert_eq!(
            layout.binding(v, regalloc::Where::Spilled),
            Loc::Slot(pin).into()
        );
        assert_eq!(
            layout.binding(v, regalloc::Where::Reg(Reg(7))),
            Loc::Reg(Reg(7)).into(),
            "pinning an address says nothing about where the value is"
        );
    }

    // =========================================================================
    // resolve_operands unit tests — the spill logic that was buggy
    // =========================================================================

    /// Helper: build minimal assignment + spill maps for resolve_operands
    /// tests. Every register an instruction may use is handed to it in
    /// `TEST_SCRATCH`, exactly as the allocator hands one its reservations.
    const TEST_FILE: regalloc::RegisterFile = regalloc::RegisterFile {
        fixed: &[],
        inputs: INPUT_REGS,
        scratch: regalloc::RegSet::range(4, regalloc::RegisterFile::MIN_SCRATCH),
        temps_for: regalloc::no_temps,
        guard_temps: 0,
        vector_bytes: 16,
    }
    .checked();

    /// The two reload registers these `resolve_operands` tests hand the
    /// instruction, standing in for the allocator's per-instruction
    /// reservations.
    const RELOAD: [Reg; 2] = [Reg(11), Reg(12)];

    /// The scratch these `resolve_operands` tests are written against.
    const TEST_SCRATCH: regalloc::Scratch =
        regalloc::Scratch::for_test(None, [Some(RELOAD[0]), Some(RELOAD[1])]);

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
            locs[v as usize] = Some(Binding::Loc(Loc::Slot(Slot::new(off, 16))));
        }
        locs
    }

    #[test]
    fn resolve_binary_no_spills() {
        // left=v4, right=v5, dst=v6 — all in registers
        let locs = make_locs(&[(0, 4), (1, 5), (2, 6)], &[]);
        let op = ScheduledOp::Binary(OpKind::Add, regalloc::ValueId(0), regalloc::ValueId(1));
        let plan =
            resolve_operands(&op, Loc::Reg(Reg(6)).into(), locs.as_slice(), TEST_SCRATCH).unwrap();

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
        let plan =
            resolve_operands(&op, Loc::Reg(Reg(6)).into(), locs.as_slice(), TEST_SCRATCH).unwrap();

        assert_eq!(plan.reloads.len(), 1);
        assert_eq!(
            plan.reloads[0],
            Reload::FromStack {
                target: Reg(6),
                slot: Slot::new(0, 16),
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
        let plan =
            resolve_operands(&op, Loc::Reg(Reg(6)).into(), locs.as_slice(), TEST_SCRATCH).unwrap();

        assert_eq!(plan.reloads.len(), 2);
        // left → dst (v6), right → tmp_op (v27)
        assert_eq!(
            plan.reloads[0],
            Reload::FromStack {
                target: Reg(6),
                slot: Slot::new(0, 16),
            }
        );
        assert_eq!(
            plan.reloads[1],
            Reload::FromStack {
                target: RELOAD[0],
                slot: Slot::new(16, 16),
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
            Loc::Slot(Slot::new(32, 16)).into(),
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
        )
        .unwrap();
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
        let plan =
            resolve_operands(&op, Loc::Reg(Reg(8)).into(), locs.as_slice(), TEST_SCRATCH).unwrap();

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
    fn resolve_muladd_decomposed_both_ab_spilled() {
        // a and b both spilled → decomposed FMUL+FADD path
        // c in register
        let locs = make_locs(&[(2, 7), (3, 8)], &[(0, 0), (1, 16)]);
        let op = ScheduledOp::Ternary(
            OpKind::MulAdd,
            regalloc::ValueId(0),
            regalloc::ValueId(1),
            regalloc::ValueId(2),
        );
        let plan =
            resolve_operands(&op, Loc::Reg(Reg(8)).into(), locs.as_slice(), TEST_SCRATCH).unwrap();

        // a → dst, b → tmp_op loaded upfront
        assert_eq!(plan.reloads.len(), 2);
        assert_eq!(
            plan.reloads[0],
            Reload::FromStack {
                target: Reg(8),
                slot: Slot::new(0, 16),
            }
        );
        assert_eq!(
            plan.reloads[1],
            Reload::FromStack {
                target: RELOAD[0],
                slot: Slot::new(16, 16),
            }
        );
        // c is in a register, no deferred reload needed
        match &plan.op {
            ResolvedOp::DecomposedMulAdd {
                dst,
                a,
                b,
                c,
                c_deferred,
            } => {
                assert_eq!(*dst, Reg(8));
                assert_eq!(*a, Reg(8));
                assert_eq!(*b, RELOAD[0]);
                assert_eq!(*c, Reg(7));
                assert_eq!(*c_deferred, None);
            }
            other => panic!("expected DecomposedMulAdd, got {:?}", other),
        }
    }

    #[test]
    fn resolve_muladd_decomposed_all_three_spilled() {
        // a, b, c all spilled → decomposed with deferred c reload
        let locs = make_locs(&[(3, 8)], &[(0, 0), (1, 16), (2, 32)]);
        let op = ScheduledOp::Ternary(
            OpKind::MulAdd,
            regalloc::ValueId(0),
            regalloc::ValueId(1),
            regalloc::ValueId(2),
        );
        let plan =
            resolve_operands(&op, Loc::Reg(Reg(8)).into(), locs.as_slice(), TEST_SCRATCH).unwrap();

        // Only a and b reloads upfront — c is deferred
        assert_eq!(plan.reloads.len(), 2);
        match &plan.op {
            ResolvedOp::DecomposedMulAdd { c, c_deferred, .. } => {
                assert_eq!(*c, RELOAD[1]); // its own reservation, deferred past the FMUL
                assert_eq!(
                    *c_deferred,
                    Some(DeferredReload::FromStack(Slot::new(32, 16)))
                );
            }
            other => panic!("expected DecomposedMulAdd, got {:?}", other),
        }
    }

    #[test]
    fn resolve_var_is_nop() {
        let locs = make_locs(&[(0, 0)], &[]);
        let op = ScheduledOp::Var(0);
        let plan =
            resolve_operands(&op, Loc::Reg(Reg(0)).into(), locs.as_slice(), TEST_SCRATCH).unwrap();
        assert_eq!(plan.op, ResolvedOp::Nop);
        assert!(plan.reloads.is_empty());
    }

    #[test]
    fn resolve_const() {
        let locs = make_locs(&[(0, 6)], &[]);
        let op = ScheduledOp::Const(core::f32::consts::PI);
        let plan =
            resolve_operands(&op, Loc::Reg(Reg(6)).into(), locs.as_slice(), TEST_SCRATCH).unwrap();
        assert_eq!(
            plan.op,
            ResolvedOp::LoadConst {
                dst: Reg(6),
                val_bits: core::f32::consts::PI.to_bits()
            }
        );
    }

    // =========================================================================
    // DAG integration tests — expressions that previously crashed (SIGSEGV)
    // =========================================================================

    /// Test that Select short-circuits: when mask is all-true, the false arm
    /// (which contains a division by zero) must NOT produce NaN in the output.
    /// Test Select with all-false mask: should return false arm.
    /// Test Select with mixed mask: BSL path, both arms evaluated.
    // =========================================================================
    // Arena compilation tests
    // =========================================================================

    // These three tests call the private `arena_to_schedule`/`arena_to_uses`
    // directly rather than through `compile`: value numbering and
    // dead-node filtering are schedule-shape invariants with no output-value
    // signature (a regression here wastes registers/instructions, it doesn't
    // change what a compiled kernel computes), so there is no public
    // black-box assertion that would catch a break here.
    #[test]
    fn arena_to_schedule_simple() {
        use pixelflow_ir::arena::ExprArena;

        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let sum = arena.push_binary(OpKind::Add, x, y);

        let schedule = arena_to_schedule(&arena, sum);

        // Should have 3 values: X, Y, X+Y
        assert_eq!(
            schedule.len(),
            3,
            "expected 3 schedule entries, got {}",
            schedule.len()
        );

        // Verify the operations
        assert!(matches!(schedule[0].op, ScheduledOp::Var(0)));
        assert!(matches!(schedule[1].op, ScheduledOp::Var(1)));
        assert!(matches!(
            schedule[2].op,
            ScheduledOp::Binary(OpKind::Add, _, _)
        ));
    }

    #[test]
    fn arena_to_schedule_filters_unreachable() {
        use pixelflow_ir::arena::ExprArena;

        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let _garbage = arena.push_const(999.0); // unreachable
        let y = arena.push_var(1);
        let sum = arena.push_binary(OpKind::Add, x, y);

        let schedule = arena_to_schedule(&arena, sum);

        // Should have 3 values (garbage node filtered out)
        assert_eq!(
            schedule.len(),
            3,
            "unreachable garbage node should be filtered"
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn arena_compile_simple() {
        use pixelflow_ir::arena::ExprArena;

        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let sum = arena.push_binary(OpKind::Add, x, y);

        let result = compile(&arena, sum).expect("arena DAG compile failed");
        assert_eq!(result.spill_count, 0);

        assert_eq!(eval_point(&result.code, 3.0, 4.0, 0.0, 0.0), 7.0);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn arena_compile_with_constant() {
        use pixelflow_ir::arena::ExprArena;

        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let two = arena.push_const(2.0);
        let y = arena.push_var(1);
        let prod = arena.push_binary(OpKind::Mul, x, two);
        let sum = arena.push_binary(OpKind::Add, prod, y);

        let result = compile(&arena, sum).expect("arena DAG compile failed");

        // 3*2 + 4 = 10
        assert_eq!(eval_point(&result.code, 3.0, 4.0, 0.0, 0.0), 10.0);
    }

    /// `Σᵢ (X+i)·(Y+i)` for i in 1..=10, summed as a balanced tree: ten
    /// products are live at once, so any pool must spill. Every leaf depends on
    /// X or Y — a uniform-only subtree would be loop-invariant, hoisted out
    /// of the collapse body, and leave nothing to spill.
    ///
    /// The pressure comes from the live ranges rather than from the budget.
    /// This used to be `(X+Y)·(X−Y) + (X·Y)·(X+1)` under `max_regs(2)`, which
    /// keeps at most three values live: it spilled only because two registers
    /// is fewer than three, and a two-register pool is no longer a budget a
    /// caller can ask for (`RegisterFile::MIN_SCRATCH` — an instruction temp
    /// cannot spill). Ten live values outrun every backend's floor, so the
    /// subject here — what spilling *does* — no longer depends on how small
    /// the pool can be made.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn arena_compile_with_spills() {
        use pixelflow_ir::arena::ExprArena;

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

        let result = EmitCtx::with_max_regs(4)
            .compile(&arena, root)
            .expect("arena DAG compile with spills failed");

        assert!(
            result.spill_count > 0,
            "expected spills under ten live terms"
        );

        // Σᵢ (3+i)·(4+i) = 20+30+42+56+72+90+110+132+156+182 = 890, every
        // term and partial sum exact in f32.
        assert_eq!(eval_point(&result.code, 3.0, 4.0, 0.0, 0.0), 890.0);
    }

    // =========================================================================
    // The shared driver's Select short-circuit guard, on every backend that
    // has a JIT.
    //
    // `sched_select_guards` below covers this path, but only on SSE2 — its
    // module is gated `not(avx2), not(avx512f)` — and `avx512_select_guards`
    // covers AVX-512. aarch64 had no guard test at all, which mattered because
    // that is the one backend whose guard needs a scratch register: reducing a
    // mask with `UMAXV`/`UMINV` writes a scalar into a vector register, where
    // the x86 tiers use `movmskps`/`kortest` and the flags. So the register
    // that reduction destroys was, on aarch64 alone, an untested choice.
    //
    // These run wherever a backend exists, against the same expected values,
    // so no backend's guard can drift from another's.
    // =========================================================================
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    mod select_guard_driver {
        use super::*;
        use pixelflow_ir::arena::{ExprArena, ExprId};

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
        fn guarded_select(a: &mut ExprArena) -> ExprId {
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
            let sel = a.push_ternary(OpKind::Select, cond, bbb, b3);
            // Live ACROSS the select and read after it. Without something in
            // this role the select is the root, nothing downstream reads a
            // register, and a guard that clobbered a live one would still
            // produce the right answer — the test would be blind to exactly
            // the mistake it exists to catch.
            let carried = a.push_binary(OpKind::Sub, x, y);
            a.push_binary(OpKind::Add, sel, carried)
        }

        /// Assert a guard region actually formed for `root`.
        ///
        /// Without this the tests below still pass when the guard stops
        /// forming — they would just be testing an ordinary `Select`, which is
        /// the silent-decay shape this file has been bitten by before.
        fn assert_guard_forms(a: &ExprArena, root: ExprId) {
            let schedule = arena_to_schedule(a, root);
            let guards = analyze_select_guards(&schedule);
            let guarded = guards.iter().any(|g| g.has_guarded_arm());
            assert!(
                guarded,
                "no Select in this schedule has an arm-exclusive range, so the \
                 short-circuit guard this test exists for is never emitted"
            );
        }

        /// A select whose true arm contains a select, with entries belonging
        /// to the root sitting inside both arms — so NEITHER level is
        /// guardable as scheduled, and both become guardable once
        /// [`guards::cluster_select_arms`] gathers each arm into one run.
        ///
        /// Nesting is the case that can go wrong quietly: an inner select's
        /// arms lie inside an outer arm, so partitioning the outside moves the
        /// inside with it. If that broke an inner guard the kernel would still
        /// be correct and merely slower, which no value test would catch —
        /// hence the assertion on the analysis as well as on the arithmetic.
        ///
        /// The two "intruders" are read by the root, so they are shared with
        /// the world outside the arms and can never be skipped; they are what
        /// makes the arms non-contiguous to begin with.
        fn nested_guarded_selects(a: &mut ExprArena) -> (ExprId, ExprId, ExprId) {
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
            let inner = a.push_ternary(OpKind::Select, inner_cond, t3, f2);

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
            let outer = a.push_ternary(OpKind::Select, outer_cond, o2, p2);
            let carried = a.push_binary(OpKind::Add, across_inner, across_outer);
            let root = a.push_binary(OpKind::Add, outer, carried);
            (root, outer, inner)
        }

        /// What `nested_guarded_selects` computes, in scalar `f32` and with no
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

        /// How many entries each select has under a guard, by schedule
        /// position, for a schedule built the way `compile` builds it.
        fn guarded_entries(a: &ExprArena, root: ExprId, cluster: bool) -> alloc::vec::Vec<usize> {
            let schedule = arena_to_schedule(a, root);
            let schedule = if cluster {
                guards::cluster_select_arms(schedule)
            } else {
                schedule
            };
            analyze_select_guards(&schedule)
                .iter()
                .map(|g| g.total_guarded_entries())
                .collect()
        }

        /// Both levels of a nested select are guarded once the schedule is
        /// clustered, and neither was before — the reordering is the whole
        /// difference.
        #[test]
        fn clustering_guards_both_levels_of_a_nested_select() {
            let mut a = ExprArena::new();
            let (root, _outer, _inner) = nested_guarded_selects(&mut a);

            let before = guarded_entries(&a, root, false);
            let after = guarded_entries(&a, root, true);
            assert!(
                before.iter().sum::<usize>() < after.iter().sum::<usize>(),
                "clustering bought nothing: {before:?} -> {after:?}"
            );
            assert_eq!(
                after.len(),
                2,
                "both the outer and the inner select must earn a guard, got {after:?}"
            );
            assert!(
                after.iter().all(|&entries| entries > 0),
                "a guard with an empty range is not a guard: {after:?}"
            );
        }

        /// The clustered kernel's answer, against the same expression
        /// evaluated in scalar `f32` with no guards: uniform masks (which take
        /// the branches) and mixed lanes (which fall through to the blend),
        /// exactly equal — every operation here is exact at these points, so
        /// there is no tolerance to hide a wrong branch in.
        #[test]
        fn a_nested_guarded_select_agrees_lane_for_lane() {
            let mut a = ExprArena::new();
            let (root, _outer, _inner) = nested_guarded_selects(&mut a);
            let result = compile(&a, root).expect("nested guarded select compile");

            // One point at a time: all four combinations of the two masks,
            // each of which takes a pair of branches.
            for &(x, y) in &[(3.0f32, 4.0f32), (3.0, -4.0), (-3.0, 4.0), (-3.0, -4.0)] {
                let got = eval_point(&result.code, x, y, 0.0, 0.0);
                assert_eq!(
                    got,
                    nested_expected(x, y),
                    "nested guarded select at ({x}, {y})"
                );
            }

            // Mixed lanes: both masks vary within the batch, so neither guard
            // fires and the blend has to produce every lane.
            let xs: [f32; LANES] = core::array::from_fn(|i| if i % 2 == 0 { 3.0 } else { -3.0 });
            let ys: [f32; LANES] = core::array::from_fn(|i| if i % 3 == 0 { 4.0 } else { -4.0 });
            let got = eval_batch(
                &result.code,
                &[],
                executable::Point4::new(xs, ys, [0.0; LANES], [0.0; LANES]),
            );
            for lane in 0..LANES {
                assert_eq!(
                    got[lane],
                    nested_expected(xs[lane], ys[lane]),
                    "lane {lane} of a mixed-mask batch"
                );
            }
        }

        /// Uniform masks take the all-true and all-false branches; a mixed
        /// mask falls through to the blend. All three must agree with the
        /// arithmetic.
        #[test]
        fn a_guarded_select_takes_every_branch() {
            let mut a = ExprArena::new();
            let root = guarded_select(&mut a);
            assert_guard_forms(&a, root);

            let result = compile(&a, root).expect("guarded select compile");
            for &(x, y) in &[
                (3.0f32, 4.0f32), // all-true  -> B³
                (-2.0, 0.5),      // all-false -> 3B
                (0.5, -1.0),
                (-0.25, 2.0),
            ] {
                let b = x * y;
                let want = if x > 0.0 { b * b * b } else { 3.0 * b } + PADDING + (x - y);
                let got = eval_point(&result.code, x, y, 0.0, 0.0);
                assert!(
                    (got - want).abs() <= 1e-3,
                    "guarded select at ({x}, {y}): got {got}, want {want}"
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
        /// victim is whatever is used farthest out. The mask is computed first
        /// and read last, and everything between it and the `Select` is
        /// consumed before the `Select` — so the mask is the farthest-out live
        /// value when the filler fills the pool, and it is the one to go. A
        /// plain `spill_count > 0` would pass with the mask still resident and
        /// this path never taken.
        #[test]
        fn a_guarded_select_survives_a_spilled_mask() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let zero = a.push_const(0.0);

            // Read only at the very end: the farthest-out live value.
            let cond = a.push_binary(OpKind::Gt, x, zero);

            // Filler that is all live at once and all consumed *before* the
            // select, so the mask outlives every one of them.
            let terms: alloc::vec::Vec<ExprId> = (1..=8u32)
                .map(|i| {
                    let c = a.push_const(i as f32);
                    a.push_binary(OpKind::Add, x, c)
                })
                .collect();
            let mid = terms[1..]
                .iter()
                .fold(terms[0], |acc, &t| a.push_binary(OpKind::Add, acc, t));

            // Shared-base arms, as in `guarded_select`.
            let base = a.push_binary(OpKind::Mul, mid, y);
            let bb = a.push_binary(OpKind::Mul, base, base);
            let bbb = a.push_binary(OpKind::Mul, bb, base);
            let bbb = worth_a_branch(&mut a, bbb);
            let b2 = a.push_binary(OpKind::Add, base, base);
            let b3 = a.push_binary(OpKind::Add, b2, base);
            let b3 = worth_a_branch(&mut a, b3);
            let sel = a.push_ternary(OpKind::Select, cond, bbb, b3);
            let carried = a.push_binary(OpKind::Sub, x, y);
            let root = a.push_binary(OpKind::Add, sel, carried);
            assert_guard_forms(&a, root);

            // The mask must actually be the value that spills.
            let file = Native::new(EmitCtx::with_max_regs(regalloc::RegisterFile::MIN_SCRATCH))
                .register_file();
            let allocation = {
                use regalloc::RegisterAllocator;
                regalloc::LinearScan.allocate(arena_to_schedule(&a, root), &file)
            };
            let mask_vid = analyze_select_guards(allocation.body().schedule())
                .first()
                .expect("a guard formed above")
                .mask_vid;
            assert!(
                allocation.placement(mask_vid).spills(),
                "the mask stayed in a register, so the spilled-mask path this \
                 test exists for is never reached"
            );

            let result = EmitCtx::with_max_regs(regalloc::RegisterFile::MIN_SCRATCH)
                .compile(&a, root)
                .expect("spilled guarded select compile");

            for &(px, py) in &[(3.0f32, 2.0f32), (-2.0, 0.5), (0.5, -1.0)] {
                let m: f32 = (1..=8).map(|i| px + i as f32).sum();
                let b = m * py;
                let want = if px > 0.0 { b * b * b } else { 3.0 * b } + PADDING + (px - py);
                let got = eval_point(&result.code, px, py, 0.0, 0.0);
                assert!(
                    (got - want).abs() <= 1e-2 * want.abs().max(1.0),
                    "spilled guarded select at ({px}, {py}): got {got}, want {want}"
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
        /// Returns the arena, the root, the value that gets split, and the
        /// select's true-arm range, so the two tests below can assert on the
        /// same shape rather than each rebuilding it.
        fn split_across_a_guarded_arm() -> (
            ExprArena,
            ExprId,
            regalloc::ValueId,
            (usize, usize),
            regalloc::NestAllocation,
        ) {
            use regalloc::RegisterAllocator;
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let zero = a.push_const(0.0);

            // Computed first, read last: the farthest-out live values, so
            // these are what eviction takes when the filler fills the pool.
            let cond = a.push_binary(OpKind::Gt, x, zero);
            let split = a.push_binary(OpKind::Mul, x, y);

            let terms: alloc::vec::Vec<ExprId> = (1..=8u32)
                .map(|i| {
                    let c = a.push_const(i as f32);
                    a.push_binary(OpKind::Add, x, c)
                })
                .collect();
            let mid = terms[1..]
                .iter()
                .fold(terms[0], |acc, &t| a.push_binary(OpKind::Add, acc, t));

            // Shared-base arms, so neither arm's leaves land outside it and
            // the arms' own nodes stay adjacent (see `guarded_select`).
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
            let sel = a.push_ternary(OpKind::Select, cond, t3, f2);
            // Read after the arm, which is what makes the confinement rule
            // load-bearing: on the skipped path this must not name the
            // register the arm would have loaded.
            let after = a.push_binary(OpKind::Add, sel, split);
            let carried = a.push_binary(OpKind::Sub, x, y);
            let root = a.push_binary(OpKind::Add, after, carried);

            let file = Native::new(EmitCtx::with_max_regs(regalloc::RegisterFile::MIN_SCRATCH))
                .register_file();
            let schedule = arena_to_schedule(&a, root);
            let allocation = regalloc::LinearScan.allocate(schedule, &file);
            let guard = analyze_select_guards(allocation.body().schedule())
                .into_iter()
                .find(|g| g.is_guarded(SelectArm::True))
                .expect("the true arm is exclusive and contiguous, so it is guarded");

            // Which `ValueId` the arena's `split` became. `X·Y` is the only
            // product of two `Var`s in this kernel.
            let body = allocation.body().schedule();
            let is_var = |v: regalloc::ValueId| {
                body.iter()
                    .any(|d| d.value == v && matches!(d.op, ScheduledOp::Var(_)))
            };
            let split_vid = body
                .iter()
                .find(|d| {
                    matches!(d.op, ScheduledOp::Binary(OpKind::Mul, l, r) if is_var(l) && is_var(r))
                })
                .map(|d| d.value)
                .expect("X·Y is in the schedule");
            (a, root, split_vid, guard.true_range(), allocation)
        }

        /// The value is right after the arm, on the path that skips it.
        #[test]
        fn a_split_range_inside_a_guarded_arm_is_correct_when_the_arm_is_skipped() {
            let (a, root, split_vid, arm, allocation) = split_across_a_guarded_arm();
            assert!(
                allocation.placement(split_vid).spills(),
                "the value under test stayed in a register, so nothing is split"
            );
            let kept = allocation
                .placement(split_vid)
                .spans()
                .any(|s| matches!(s.at, regalloc::Where::Reg(_)) && s.from.index >= arm.0);
            assert!(
                kept,
                "the value was never brought back into a register inside the \
                 arm, so the confinement rule this test exists for is not exercised"
            );

            let result = EmitCtx::with_max_regs(regalloc::RegisterFile::MIN_SCRATCH)
                .compile(&a, root)
                .expect("split-across-a-guard compile");
            // x < 0 is the all-false mask: the true arm — and the reload
            // inside it — never runs, and the read after it must still be the
            // value.
            for &(px, py) in &[(-2.0f32, 3.0f32), (-0.5, -4.0), (3.0, 2.0), (0.25, 1.5)] {
                let m: f32 = (1..=8).map(|i| px + i as f32).sum();
                let b = m * py;
                let v = px * py;
                let arm_value = if px > 0.0 {
                    (b * v + v) * b
                } else {
                    (b + b) + b
                };
                let want = arm_value + PADDING + v + (px - py);
                let got = eval_point(&result.code, px, py, 0.0, 0.0);
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
        /// the same `analyze_select_guards` the emitter branches on, which is
        /// what makes "exactly" a statement about one answer rather than two.
        #[test]
        fn a_kept_reload_inside_a_guarded_arm_ends_at_the_arm() {
            let (_, _, split_vid, arm, allocation) = split_across_a_guarded_arm();
            let spans: alloc::vec::Vec<regalloc::Span> =
                allocation.placement(split_vid).spans().collect();
            let kept = spans
                .iter()
                .position(|s| matches!(s.at, regalloc::Where::Reg(_)) && s.from.index >= arm.0)
                .expect("a register range begins inside the arm");
            assert!(
                spans[kept].from.index < arm.1,
                "the range begins outside the arm it was confined to"
            );
            let ends_at = spans
                .get(kept + 1)
                .map(|s| s.from.index)
                .expect("a confined range is followed by the range it reverts to");
            assert_eq!(
                ends_at, arm.1,
                "a register range that begins inside a guarded arm must end \
                 where the arm does: a read after it would name a register the \
                 skipped path never loaded"
            );
        }
    }

    /// Run an arena kernel at `x` (Y = 0) and return lane 0. The
    /// builtin-parity tests below use it. Gated off `+avx512f` (those builtins
    /// aren't in the AVX-512 op set yet anyway).
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        not(target_feature = "avx2")
    ))]
    fn run1(arena: &ExprArena, root: ExprId, x: f32) -> f32 {
        let r = compile(arena, root).expect("compile failed");
        eval_point(&r.code, x, 0.0, 0.0, 0.0)
    }

    /// Eval at (X=x, Y=y, Z=W=0), lane 0. Gated off `+avx512f` like `run1`.
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        not(target_feature = "avx2")
    ))]
    fn run_xy(arena: &ExprArena, root: ExprId, x: f32, y: f32) -> f32 {
        let r = compile(arena, root).expect("compile failed");
        eval_point(&r.code, x, y, 0.0, 0.0)
    }

    /// A `Dwrt`-carrying arena must JIT-compile end-to-end: the compile entry
    /// runs `lower_dwrt`, so `D(√(x²+y²), x)` compiles to `x / √(x²+y²)`
    /// without the caller ever seeing the derivative machinery.
    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        not(target_feature = "avx2")
    ))]
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
        assert!(compile(&a, root).is_err());
    }

    /// A spill frame past the 128-byte red zone must allocate a real frame
    /// (`sub rsp`) and produce correct results — the glyph-scale-kernel case
    /// that used to refuse with "exceeds 128-byte red zone". 40 products are
    /// all pushed before any is consumed, so dozens are simultaneously live
    /// against 6 allocatable registers.
    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        not(target_feature = "avx2")
    ))]
    fn spill_frame_beyond_red_zone_compiles_correctly() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let mut products = alloc::vec::Vec::new();
        for i in 0..40u32 {
            let c = a.push_const(i as f32 + 1.0);
            let xa = a.push_binary(OpKind::Add, x, c);
            let yb = a.push_binary(OpKind::Add, y, c);
            products.push(a.push_binary(OpKind::Mul, xa, yb));
        }
        let mut root = products[0];
        for p in &products[1..] {
            root = a.push_binary(OpKind::Add, root, *p);
        }

        let result = compile(&a, root).expect("large spill frame must compile");
        assert!(
            result.spill_bytes > 128,
            "test did not force a frame beyond the red zone (spill_bytes = {})",
            result.spill_bytes
        );

        for (px, py) in [(1.5f32, -2.0f32), (0.0, 0.0), (3.0, 4.0)] {
            let got = run_xy(&a, root, px, py);
            let want: f32 = (0..40)
                .map(|i| (px + i as f32 + 1.0) * (py + i as f32 + 1.0))
                .sum();
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
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        not(target_feature = "avx2")
    ))]
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
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        not(target_feature = "avx2")
    ))]
    fn x86_binary_ternary_builtins_match_scalar() {
        // Helper: compile f(X, Y) and eval at (x, y).
        fn run2(arena: &ExprArena, root: ExprId, x: f32, y: f32) -> f32 {
            let r = compile(arena, root).expect("compile failed");
            eval_point(&r.code, x, y, 0.0, 0.0)
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
        // Select(X >= 0, 1.0, -1.0) == signum-ish
        {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let zero = a.push_const(0.0);
            let cond = a.push_binary(OpKind::Ge, x, zero);
            let pos = a.push_const(1.0);
            let neg = a.push_const(-1.0);
            let root = a.push_ternary(OpKind::Select, cond, pos, neg);
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
    /// `lowering`). Validated against `f32` on the default (128-bit) build.
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        not(target_feature = "avx2")
    ))]
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

        /// atan/atan2/asin/acos lower to arithmetic + Select (atan2 is the core;
        /// the others derive from it). Value path only — atan2 uses Select, which
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
    // x86 shared-pipeline path (schedule → regalloc → spill).
    // =========================================================================
    // 128-bit build only; gated off `+avx512f` (covered by `avx512_driver`).
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        not(target_feature = "avx2")
    ))]
    mod sched {
        use super::*;

        /// One batch of a kernel whose single argument is bound to `u` — the
        /// lattice-invariant third input a test used to spell `Var(2)`.
        fn eval_point_with_arg(code: &executable::ExecutableCode, x: f32, y: f32, u: f32) -> f32 {
            let block = [u];
            let ctx: [*const f32; 1] = [block.as_ptr()];
            let o = executable::Point4::new([x; LANES], [y; LANES], [0.0; LANES], [0.0; LANES]);
            eval_batch(code, &ctx, o)[0]
        }

        /// Declare one argument in `a` and return its leaf.
        fn arg_leaf(a: &mut ExprArena, default: f32) -> pixelflow_ir::ExprId {
            let slot = a.declare_uniform(pixelflow_ir::Uniform::new(default).decl());
            a.push_uniform(slot)
        }

        use pixelflow_ir::arena::ExprArena;

        fn run(res: &CompileResult, x: f32, y: f32, z: f32, w: f32) -> f32 {
            eval_point(&res.code, x, y, z, w)
        }

        const PTS: &[(f32, f32, f32, f32)] = &[
            (3.0, 4.0, 0.0, 1.0),
            (1.0, 2.0, 3.0, 4.0),
            (-2.0, 0.5, 1.5, -1.0),
            (0.7, -1.3, 2.1, 0.2),
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

            let sched = compile(&a, root).expect("compile");
            assert_eq!(sched.spill_count, 0, "should fit without spilling");

            for &(px, py, pz, _pw) in PTS {
                let want = (px * px + py * py).sqrt() - py * pz;
                let got = eval_point_with_arg(&sched.code, px, py, pz);
                assert!((got - want).abs() <= 1e-4, "got {got} want {want}");
            }
        }

        /// A wide expression that exceeds the 7 allocatable registers must spill
        /// (to the red zone) and still compute the right answer.
        #[test]
        fn sched_spills_and_is_correct() {
            // sum_{i=1..=10} (X + i) * (Y + i), as a balanced tree so the 10
            // products are live together — forcing spills with only 7 regs.
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

            let sched = compile(&a, root).expect("scheduled compile");
            assert!(
                sched.spill_count > 0,
                "expected spilling; widen the expression if this regresses"
            );

            for &(px, py, _pz, _pw) in PTS {
                let mut want = 0.0f32;
                for i in 1..=10u32 {
                    want += (px + i as f32) * (py + i as f32);
                }
                let got = run(&sched, px, py, 0.0, 0.0);
                let tol = 1e-3 * want.abs().max(1.0);
                assert!((got - want).abs() <= tol, "spill: got {got} want {want}");
            }
        }

        /// Exercises the shared driver's Select short-circuit guard path on x86
        /// (MOVMSKPS all-true/all-false branches): `(X > 0) ? Y*Y*Y : X+X+X`,
        /// with arm-exclusive subexpressions so a guard region forms. Uniform
        /// inputs take the all-true / all-false branches.
        ///
        /// Both arms are per-*lane*, which is what makes them arms: an arm of
        /// the kernel's arguments alone would be lattice-invariant and hoist
        /// out of the body entirely, leaving nothing for a guard to skip.
        #[test]
        fn sched_select_guards() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let zero = a.push_const(0.0);
            let cond = a.push_binary(OpKind::Gt, x, zero); // X > 0 -> mask
            let yy = a.push_binary(OpKind::Mul, y, y);
            let yyy = a.push_binary(OpKind::Mul, yy, y); // true arm: Y^3
            let zz = a.push_binary(OpKind::Add, x, x);
            let zzz = a.push_binary(OpKind::Add, zz, x); // false arm: 3X
            let root = a.push_ternary(OpKind::Select, cond, yyy, zzz);

            let sched = compile(&a, root).expect("scheduled compile");

            // x>0 -> all-true -> Y^3 ; x<=0 -> all-false -> 3X.
            for &(px, py, _pz, _pw) in PTS {
                let want = if px > 0.0 { py * py * py } else { 3.0 * px };
                let got = run(&sched, px, py, 0.0, 0.0);
                assert!(
                    (got - want).abs() <= 1e-3,
                    "select: ({px},{py}) got {got} want {want}"
                );
            }
        }
    }

    // =========================================================================
    // 128-bit end-to-end: bound-memory gather through the shared driver, run on
    // the host across 4 lanes. Covers BOTH 128-bit backends — NEON's native
    // `ld1` lanes and x86's scalar-load assembly (no AVX2 `vgatherdps` at 128
    // bits) — against the same interpreter oracle, so the two cannot drift.
    // Mirrors the avx512_driver gather tests at 128-bit width.
    // =========================================================================
    #[cfg(any(
        target_arch = "aarch64",
        all(
            target_arch = "x86_64",
            not(target_feature = "avx512f"),
            not(target_feature = "avx2")
        )
    ))]
    mod gather_driver_128 {
        use super::*;
        use pixelflow_ir::arena::ExprId;

        /// Run a compiled gather kernel over one batch: `ctx` is the array of
        /// buffer base pointers. Arch-independent now that the coordinates are
        /// plain arrays rather than intrinsics.
        fn run4_ctx(
            res: &CompileResult,
            ctx: &[*const f32],
            xs: [f32; LANES],
            ys: [f32; LANES],
        ) -> [f32; LANES] {
            eval_batch(
                &res.code,
                ctx,
                executable::Point4::new(xs, ys, [0.0; LANES], [0.0; LANES]),
            )
        }

        #[allow(clippy::too_many_arguments)] // test helper: 6 distinct params (arena, root, buffers, xs, ys, tag)
        /// Check a compiled gather kernel lane-for-lane against `eval_scalar`,
        /// the reference interpreter, over the same coords and binding. The 16
        /// coordinate pairs run as four 4-lane batches.
        fn check_against_interp(
            arena: &ExprArena,
            root: ExprId,
            buffers: &[&[f32]],
            xs: [f32; 16],
            ys: [f32; 16],
            tag: &str,
        ) {
            let res = compile(arena, root).expect("compile gather kernel");
            let ctx: Vec<*const f32> = buffers.iter().map(|b| b.as_ptr()).collect();
            let bindings = pixelflow_ir::binding::BindingTable::bind(arena, buffers).unwrap();

            for batch in 0..4 {
                let mut cx = [0.0f32; 4];
                let mut cy = [0.0f32; 4];
                cx.copy_from_slice(&xs[batch * 4..batch * 4 + 4]);
                cy.copy_from_slice(&ys[batch * 4..batch * 4 + 4]);
                let got = run4_ctx(&res, &ctx, cx, cy);
                for i in 0..4 {
                    let want =
                        pixelflow_ir::eval::eval_scalar(arena, root, &[cx[i], cy[i]], &bindings);
                    assert_eq!(
                        got[i], want,
                        "{tag} batch {batch} lane {i} (x={}, y={})",
                        cx[i], cy[i]
                    );
                }
            }
        }

        fn idx_lanes() -> ([f32; 16], [f32; 16]) {
            // A spread of in-range, fractional, and out-of-range coordinates so
            // the clamp and floor paths are all exercised.
            let xs = [
                0.0, 1.0, 2.9, 7.0, -3.0, 100.0, 4.0, 5.5, 6.0, 0.1, 3.0, 2.0, 1.9, 7.9, -0.5, 4.4,
            ];
            let ys = [
                0.0, 0.0, 1.0, 1.9, 2.0, 2.0, -1.0, 3.0, 0.5, 2.9, 1.0, 3.9, 0.0, 2.0, 5.0, 1.0,
            ];
            (xs, ys)
        }

        #[test]
        fn gather_jit_matches_interpreter() {
            // 8x4 buffer, gather at (X, Y).
            let (w, h) = (8usize, 4usize);
            let buf: Vec<f32> = (0..(w * h)).map(|i| i as f32 * 2.0 - 3.0).collect();
            let mut a = ExprArena::new();
            let b = a.declare_buffer(pixelflow_ir::arena::BufferDecl {
                id: pixelflow_ir::arena::BufferIdentity::mint(),
                width: w as u32,
                height: h as u32,
            });
            let x = a.push_var(0);
            let y = a.push_var(1);
            let root = a.push_gather(b, x, y);
            let (xs, ys) = idx_lanes();
            check_against_interp(&a, root, &[buf.as_slice()], xs, ys, "gather");
        }

        #[test]
        fn gather_composed_with_arithmetic() {
            // out = gather(buf, X, Y) * 2 + Y — proves gather is a schedulable
            // mid-expression node, not just a whole-kernel root.
            let (w, h) = (8usize, 4usize);
            let buf: Vec<f32> = (0..(w * h)).map(|i| (i as f32).sin()).collect();
            let mut a = ExprArena::new();
            let b = a.declare_buffer(pixelflow_ir::arena::BufferDecl {
                id: pixelflow_ir::arena::BufferIdentity::mint(),
                width: w as u32,
                height: h as u32,
            });
            let x = a.push_var(0);
            let y = a.push_var(1);
            let g = a.push_gather(b, x, y);
            let two = a.push_const(2.0);
            let scaled = a.push_binary(OpKind::Mul, g, two);
            let root = a.push_binary(OpKind::Add, scaled, y);
            let (xs, ys) = idx_lanes();
            check_against_interp(&a, root, &[buf.as_slice()], xs, ys, "gather*2+Y");
        }

        #[test]
        fn gather_two_buffers() {
            // gA(X,Y) + gB(Y,X) with two distinct bound buffers, exercising
            // slot 0 and slot 1 of the context.
            let (w, h) = (6usize, 6usize);
            let buf_a: Vec<f32> = (0..(w * h)).map(|i| i as f32).collect();
            let buf_b: Vec<f32> = (0..(w * h)).map(|i| -(i as f32) * 0.5).collect();
            let mut a = ExprArena::new();
            let ba = a.declare_buffer(pixelflow_ir::arena::BufferDecl {
                id: pixelflow_ir::arena::BufferIdentity::mint(),
                width: w as u32,
                height: h as u32,
            });
            let bb = a.declare_buffer(pixelflow_ir::arena::BufferDecl {
                id: pixelflow_ir::arena::BufferIdentity::mint(),
                width: w as u32,
                height: h as u32,
            });
            let x = a.push_var(0);
            let y = a.push_var(1);
            let ga = a.push_gather(ba, x, y);
            let gb = a.push_gather(bb, y, x);
            let root = a.push_binary(OpKind::Add, ga, gb);
            let (xs, ys) = idx_lanes();
            check_against_interp(
                &a,
                root,
                &[buf_a.as_slice(), buf_b.as_slice()],
                xs,
                ys,
                "2-buf",
            );
        }

        #[test]
        fn matmul_reduce_jit_matches_interpreter() {
            // out(j) = Σ_i W(i,j) * input(i), evaluated per output lane j = X.
            // The reduction over i unrolls to a flat gather/FMA chain (bound
            // extent), and the whole thing runs as one bound-memory kernel.
            //   W is IN×OUT row-major (width=IN, height=OUT); input is length IN.
            let (in_dim, out_dim) = (4usize, 6usize);
            let w: Vec<f32> = (0..(in_dim * out_dim))
                .map(|k| (k as f32) * 0.5 - 2.0)
                .collect();
            let input: Vec<f32> = (0..in_dim).map(|k| k as f32 + 1.0).collect();

            let mut a = ExprArena::new();
            let wb = a.declare_buffer(pixelflow_ir::arena::BufferDecl {
                id: pixelflow_ir::arena::BufferIdentity::mint(),
                width: in_dim as u32,
                height: out_dim as u32,
            });
            let ib = a.declare_buffer(pixelflow_ir::arena::BufferDecl {
                id: pixelflow_ir::arena::BufferIdentity::mint(),
                width: in_dim as u32,
                height: 1,
            });
            // body(i, j=X) = W(i, X) * input(i, 0)
            let i = a.push_var(4);
            let j = a.push_var(0);
            let zero = a.push_const(0.0);
            let wg = a.push_gather(wb, i, j);
            let ig = a.push_gather(ib, i, zero);
            let prod = a.push_binary(OpKind::Mul, wg, ig);
            let root = a.push_reduce(OpKind::Add, 4, in_dim as u32, prod);

            let buffers: &[&[f32]] = &[w.as_slice(), input.as_slice()];
            // Output lanes j = 0..6 (rest clamp to the last row, harmless here).
            let xs = [
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0,
            ];
            let ys = [0.0f32; 16];
            check_against_interp(&a, root, buffers, xs, ys, "matmul");
        }
    }

    // =========================================================================
    // AVX-512 end-to-end: arena -> shared driver -> EVEX zmm kernel, run on the
    // host across all 16 lanes. Built only with +avx512f.
    // =========================================================================
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    mod avx512_driver {
        use super::*;
        use pixelflow_ir::arena::{ExprArena, ExprId};

        /// Run a compiled zmm kernel over 16 distinct lanes per coordinate.
        fn run16(res: &CompileResult, xs: [f32; 16], ys: [f32; 16], zs: [f32; 16]) -> [f32; 16] {
            let o = executable::Point4::new(xs, ys, zs, [0.0; 16]);
            eval_batch(&res.code, &[], o)
        }

        fn lanes() -> ([f32; 16], [f32; 16], [f32; 16]) {
            let mut xs = [0.0; 16];
            let mut ys = [0.0; 16];
            let mut zs = [0.0; 16];
            for i in 0..16 {
                xs[i] = i as f32 - 7.0;
                ys[i] = (i as f32) * 0.5 + 1.0;
                zs[i] = 3.0 - (i as f32) * 0.25;
            }
            (xs, ys, zs)
        }

        fn check(got: [f32; 16], want: impl Fn(usize) -> f32, tag: &str) {
            for (i, &g) in got.iter().enumerate() {
                let w = want(i);
                assert!(
                    (g - w).abs() <= 1e-3,
                    "{tag} lane {i}: got {} want {}",
                    g,
                    w
                );
            }
        }

        // ---- Bound-memory gather: JIT vs reference interpreter ----

        /// Run a compiled gather kernel with `ctx` bound as its buffer bases.
        fn run16_ctx(
            res: &CompileResult,
            ctx: &[*const f32],
            xs: [f32; 16],
            ys: [f32; 16],
        ) -> [f32; 16] {
            let o = executable::Point4::new(xs, ys, [0.0; 16], [0.0; 16]);
            eval_batch(&res.code, ctx, o)
        }

        /// Check a compiled gather kernel lane-for-lane against `eval_scalar`,
        /// the reference interpreter, over the same coords and binding.
        #[allow(clippy::too_many_arguments)] // test helper: 6 distinct params (arena, root, buffers, xs, ys, tag)
        fn check_against_interp(
            arena: &ExprArena,
            root: ExprId,
            buffers: &[&[f32]],
            xs: [f32; 16],
            ys: [f32; 16],
            tag: &str,
        ) {
            let res = compile(arena, root).expect("compile gather kernel");
            let ctx: Vec<*const f32> = buffers.iter().map(|b| b.as_ptr()).collect();
            let got = run16_ctx(&res, &ctx, xs, ys);

            let bindings = pixelflow_ir::binding::BindingTable::bind(arena, buffers).unwrap();
            for (i, &g) in got.iter().enumerate() {
                let want = pixelflow_ir::eval::eval_scalar(arena, root, &[xs[i], ys[i]], &bindings);
                assert_eq!(g, want, "{tag} lane {i} (x={}, y={})", xs[i], ys[i]);
            }
        }

        fn idx_lanes() -> ([f32; 16], [f32; 16]) {
            // A spread of in-range, fractional, and out-of-range coordinates so
            // the clamp and floor paths are all exercised.
            let xs = [
                0.0, 1.0, 2.9, 7.0, -3.0, 100.0, 4.0, 5.5, 6.0, 0.1, 3.0, 2.0, 1.9, 7.9, -0.5, 4.4,
            ];
            let ys = [
                0.0, 0.0, 1.0, 1.9, 2.0, 2.0, -1.0, 3.0, 0.5, 2.9, 1.0, 3.9, 0.0, 2.0, 5.0, 1.0,
            ];
            (xs, ys)
        }

        #[test]
        fn gather_jit_matches_interpreter() {
            // 8x4 buffer, gather at (X, Y).
            let (w, h) = (8usize, 4usize);
            let buf: Vec<f32> = (0..(w * h)).map(|i| i as f32 * 2.0 - 3.0).collect();
            let mut a = ExprArena::new();
            let b = a.declare_buffer(pixelflow_ir::arena::BufferDecl {
                id: pixelflow_ir::arena::BufferIdentity::mint(),
                width: w as u32,
                height: h as u32,
            });
            let x = a.push_var(0);
            let y = a.push_var(1);
            let root = a.push_gather(b, x, y);
            let (xs, ys) = idx_lanes();
            check_against_interp(&a, root, &[buf.as_slice()], xs, ys, "gather");
        }

        #[test]
        fn gather_composed_with_arithmetic() {
            // out = gather(buf, X, Y) * 2 + Y — proves gather is a schedulable
            // mid-expression node, not just a whole-kernel root.
            let (w, h) = (8usize, 4usize);
            let buf: Vec<f32> = (0..(w * h)).map(|i| (i as f32).sin()).collect();
            let mut a = ExprArena::new();
            let b = a.declare_buffer(pixelflow_ir::arena::BufferDecl {
                id: pixelflow_ir::arena::BufferIdentity::mint(),
                width: w as u32,
                height: h as u32,
            });
            let x = a.push_var(0);
            let y = a.push_var(1);
            let g = a.push_gather(b, x, y);
            let two = a.push_const(2.0);
            let scaled = a.push_binary(OpKind::Mul, g, two);
            let root = a.push_binary(OpKind::Add, scaled, y);
            let (xs, ys) = idx_lanes();
            check_against_interp(&a, root, &[buf.as_slice()], xs, ys, "gather*2+Y");
        }

        #[test]
        fn gather_two_buffers() {
            // coverage.select via arithmetic: gA(X,Y) + gB(Y,X) with two distinct
            // bound buffers, exercising slot 0 and slot 1 of the context.
            let (w, h) = (6usize, 6usize);
            let buf_a: Vec<f32> = (0..(w * h)).map(|i| i as f32).collect();
            let buf_b: Vec<f32> = (0..(w * h)).map(|i| -(i as f32) * 0.5).collect();
            let mut a = ExprArena::new();
            let ba = a.declare_buffer(pixelflow_ir::arena::BufferDecl {
                id: pixelflow_ir::arena::BufferIdentity::mint(),
                width: w as u32,
                height: h as u32,
            });
            let bb = a.declare_buffer(pixelflow_ir::arena::BufferDecl {
                id: pixelflow_ir::arena::BufferIdentity::mint(),
                width: w as u32,
                height: h as u32,
            });
            let x = a.push_var(0);
            let y = a.push_var(1);
            let ga = a.push_gather(ba, x, y);
            let gb = a.push_gather(bb, y, x);
            let root = a.push_binary(OpKind::Add, ga, gb);
            let (xs, ys) = idx_lanes();
            check_against_interp(
                &a,
                root,
                &[buf_a.as_slice(), buf_b.as_slice()],
                xs,
                ys,
                "2-buf",
            );
        }

        #[test]
        fn matmul_reduce_jit_matches_interpreter() {
            // out(j) = Σ_i W(i,j) * input(i), evaluated per output lane j = X.
            // The reduction over i unrolls to a flat gather/FMA chain (bound
            // extent), and the whole thing runs as one bound-memory kernel.
            //   W is IN×OUT row-major (width=IN, height=OUT); input is length IN.
            let (in_dim, out_dim) = (4usize, 6usize);
            let w: Vec<f32> = (0..(in_dim * out_dim))
                .map(|k| (k as f32) * 0.5 - 2.0)
                .collect();
            let input: Vec<f32> = (0..in_dim).map(|k| k as f32 + 1.0).collect();

            let mut a = ExprArena::new();
            let wb = a.declare_buffer(pixelflow_ir::arena::BufferDecl {
                id: pixelflow_ir::arena::BufferIdentity::mint(),
                width: in_dim as u32,
                height: out_dim as u32,
            });
            let ib = a.declare_buffer(pixelflow_ir::arena::BufferDecl {
                id: pixelflow_ir::arena::BufferIdentity::mint(),
                width: in_dim as u32,
                height: 1,
            });
            // body(i, j=X) = W(i, X) * input(i, 0)
            let i = a.push_var(4);
            let j = a.push_var(0);
            let zero = a.push_const(0.0);
            let wg = a.push_gather(wb, i, j);
            let ig = a.push_gather(ib, i, zero);
            let prod = a.push_binary(OpKind::Mul, wg, ig);
            let root = a.push_reduce(OpKind::Add, 4, in_dim as u32, prod);

            let buffers: &[&[f32]] = &[w.as_slice(), input.as_slice()];
            // Output lanes j = 0..6 (rest clamp to the last row, harmless here).
            let xs = [
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0,
            ];
            let ys = [0.0f32; 16];
            check_against_interp(&a, root, buffers, xs, ys, "matmul");
        }

        /// sqrt(X*X + Y*Y) - Z, with a non-commutative shape and FMA-able terms,
        /// fitting in registers (no spill).
        #[test]
        fn avx512_arith_no_spill() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let z = a.push_binary(OpKind::Mul, y, x);
            let xx = a.push_binary(OpKind::Mul, x, x);
            let yy = a.push_binary(OpKind::Mul, y, y);
            let sum = a.push_binary(OpKind::Add, xx, yy);
            let dist = a.push_unary(OpKind::Sqrt, sum);
            let root = a.push_binary(OpKind::Sub, dist, z);

            let res = compile(&a, root).expect("avx512 compile");
            assert_eq!(res.spill_count, 0, "should fit without spilling");

            let (xs, ys, zs) = lanes();
            check(
                run16(&res, xs, ys, zs),
                |i| (xs[i] * xs[i] + ys[i] * ys[i]).sqrt() - ys[i] * xs[i],
                "norm-z",
            );
        }

        /// Spilling on AVX-512 must use a real stack frame: a zmm is 64 bytes
        /// and the SSE2 red zone cannot hold one.
        ///
        /// The pool is capped rather than out-sized by the expression. This
        /// test used to lean on a wide DAG "exceeding the 6 allocatable zmm
        /// regs" and stopped spilling the moment the pool grew to 22 — the
        /// subject here is what spilling *does*, not when it happens, so say
        /// so with `with_max_regs` instead of racing the allocator.
        #[test]
        fn avx512_spills_to_real_frame() {
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
                for pair in terms.chunks(2) {
                    if pair.len() == 2 {
                        next.push(a.push_binary(OpKind::Add, pair[0], pair[1]));
                    } else {
                        next.push(pair[0]);
                    }
                }
                terms = next;
            }
            let root = terms[0];

            let res = EmitCtx::with_max_regs(4)
                .compile(&a, root)
                .expect("avx512 compile");
            assert!(res.spill_count > 0, "expected spilling");

            let (xs, ys, zs) = lanes();
            check(
                run16(&res, xs, ys, zs),
                |i| {
                    let mut acc = 0.0f32;
                    for k in 1..=10u32 {
                        acc += (xs[i] + k as f32) * (ys[i] + k as f32);
                    }
                    acc
                },
                "spill",
            );
        }

        /// Compare + select with non-exclusive arms: `(X < Y) ? X : Y` (== min).
        /// No guard region forms, so this is the plain vcmpps->vpmovm2d mask +
        /// vpternlogd blend path.
        #[test]
        fn avx512_compare_select_blend() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let cond = a.push_binary(OpKind::Lt, x, y);
            let root = a.push_ternary(OpKind::Select, cond, x, y);

            let res = compile(&a, root).expect("avx512 compile");
            let (xs, ys, zs) = lanes();
            check(run16(&res, xs, ys, zs), |i| xs[i].min(ys[i]), "lt-select");
        }

        /// Select with arm-exclusive subexpressions: `(X > 0) ? Y*Y*Y : Z+Z+Z`.
        /// Forms guard regions, exercising the vptestmd+kortestw short-circuit
        /// branches (all-false skips Y^3, all-true skips 3Z) plus the per-lane
        /// blend on mixed input.
        #[test]
        fn avx512_select_guards() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let zero = a.push_const(0.0);
            let cond = a.push_binary(OpKind::Gt, x, zero);
            let yy = a.push_binary(OpKind::Mul, y, y);
            let yyy = a.push_binary(OpKind::Mul, yy, y);
            let zz = a.push_binary(OpKind::Add, x, x);
            let zzz = a.push_binary(OpKind::Add, zz, x);
            let root = a.push_ternary(OpKind::Select, cond, yyy, zzz);

            let res = compile(&a, root).expect("avx512 compile");

            let allpos = [2.0f32; 16];
            let allneg = [-2.0f32; 16];
            let ys = core::array::from_fn::<f32, 16, _>(|i| i as f32 * 0.5 + 1.0);
            let zs = core::array::from_fn::<f32, 16, _>(|i| 3.0 - i as f32 * 0.25);
            check(
                run16(&res, allpos, ys, zs),
                |i| ys[i] * ys[i] * ys[i],
                "guard-true",
            );
            let _unused_third_input = zs;
            check(
                run16(&res, allneg, ys, zs),
                |_| 3.0 * allneg[0],
                "guard-false",
            );

            let mixed = core::array::from_fn::<f32, 16, _>(|i| if i % 2 == 0 { 1.0 } else { -1.0 });
            check(
                run16(&res, mixed, ys, zs),
                |i| {
                    if mixed[i] > 0.0 {
                        ys[i] * ys[i] * ys[i]
                    } else {
                        3.0 * mixed[i]
                    }
                },
                "guard-mixed",
            );
        }

        /// Rounding via vrndscaleps (floor/ceil/round), each a single EVEX op.
        #[test]
        fn avx512_rounding() {
            // Mixed fractional/sign inputs so each rounding mode is distinct.
            let xs = core::array::from_fn::<f32, 16, _>(|i| (i as f32 - 8.0) * 0.7);
            let ones = [1.0f32; 16];
            for (op, f, tag) in [
                (OpKind::Floor, f32::floor as fn(f32) -> f32, "floor"),
                (OpKind::Ceil, f32::ceil as fn(f32) -> f32, "ceil"),
                (
                    OpKind::Round,
                    f32::round_ties_even as fn(f32) -> f32,
                    "round",
                ),
            ] {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let root = a.push_unary(op, x);
                let res = compile(&a, root).expect("avx512 compile");
                check(run16(&res, xs, ones, ones), |i| f(xs[i]), tag);
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
    // Each test below is scoped to a backend this build ACTUALLY compiles —
    // `x86_backend_covers_required_ops` always runs on x86-64,
    // `aarch64_backend_covers_required_ops` always runs on aarch64, and
    // `avx512_backend_covers_required_ops` only compiles (and only needs to
    // pass) when built with `avx512f` — the same feature gate
    // `compile` uses to select `Avx512Backend` in production. On a default `cargo test --workspace` (no RUSTFLAGS) on
    // this x86-64 host, that means: the SSE2 test runs and must be green
    // (it is: X86Backend already covers every required op), and the AVX-512
    // test does not even compile — it isn't lying about passing, it simply
    // isn't part of this build. The moment someone builds with
    // `+avx512f` (exactly the multi-ISA completion work tracked separately),
    // this same test starts running and will fail loudly, by name, for every
    // op `avx512::emit_unary`/`emit_binary`/`emit_plan` doesn't yet cover —
    // rather than waiting for an unrelated test to trip over the gap.
    mod uniforms {
        use super::*;
        use pixelflow_ir::arena::{UniformDecl, UniformIdentity};

        fn decl(default: f32) -> UniformDecl {
            UniformDecl {
                id: UniformIdentity::mint(),
                default,
            }
        }

        /// `x + u·u`: the uniform's load and the product that depends on it
        /// alone are per-call work. Asserted on the partition — which region
        /// holds them — not on timing.
        #[test]
        fn a_uniform_and_what_depends_on_it_alone_land_in_the_frame_prologue() {
            let mut a = ExprArena::new();
            let u = a.declare_uniform(decl(3.0));
            let x = a.push_var(0);
            let uu = a.push_uniform(u);
            let sq = a.push_binary(OpKind::Mul, uu, uu);
            let root = a.push_binary(OpKind::Add, x, sq);

            let (arena, root) = pixelflow_ir::passes::legalize(&a, root).expect("legalize");
            let schedule = arena_to_schedule(&arena, root);
            let variance = schedule_variance(&schedule);
            let scoped = partition_by_scope(schedule, &variance, &[0u8, 1]);

            let frame = &scoped.regions[0];
            assert!(
                frame
                    .schedule
                    .iter()
                    .any(|d| matches!(d.op, ScheduledOp::Uniform(_))),
                "the broadcast load is once per call"
            );
            assert!(
                frame
                    .schedule
                    .iter()
                    .any(|d| matches!(d.op, ScheduledOp::Binary(OpKind::Mul, ..))),
                "and so is u·u"
            );
            assert_eq!(frame.roots.len(), 1, "the product is the one parked value");
            assert!(
                !scoped
                    .body
                    .iter()
                    .any(|d| matches!(d.op, ScheduledOp::Uniform(_))),
                "the body reads the parked product, never the block"
            );
            assert!(
                scoped.regions[1].schedule.is_empty(),
                "nothing about a uniform is per row"
            );
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
            let res = compile(&a, root).expect("compile");
            assert!(res.hoisted_values >= 1, "2·u₁ is per call");

            let xs: [f32; LANES] = core::array::from_fn(|i| i as f32);
            for block in [[1.0f32, 10.0], [-2.5, 0.25]] {
                let ctx = [block.as_ptr()];
                let out = eval_batch(
                    &res.code,
                    &ctx,
                    executable::Point4::new(xs, [0.0; LANES], [0.0; LANES], [0.0; LANES]),
                );
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
            let res = compile(&a, root).expect("compile");

            let block = [0.5f32];
            let ctx = [data.as_ptr(), block.as_ptr()];
            let xs: [f32; LANES] = core::array::from_fn(|i| i as f32);
            let out = eval_batch(
                &res.code,
                &ctx,
                executable::Point4::new(xs, [0.0; LANES], [0.0; LANES], [0.0; LANES]),
            );
            for (i, got) in out.iter().enumerate() {
                assert_eq!(*got, data[i.min(3)] + 0.5, "lane {i}");
            }
        }

        /// The bytes, per backend, for `ctx_slot = 2, offset = 3, dst = 5`.
        /// Checked against `llvm-mc --disassemble` (LLVM 18):
        /// `movq 16(%rdi), %rax` then `vbroadcastss 12(%rax), %xmm5` /
        /// `%ymm5` / `%zmm5`; `ldr x9, [x0, #16]`, `ldr s5, [x9, #12]`,
        /// `dup v5.4s, v5.s[0]`.
        #[test]
        fn every_backend_encodes_the_broadcast_load() {
            let load = UniformLoad {
                ctx_slot: 2,
                offset: 3,
            };
            const MOV_RAX_CTX2: [u8; 7] = [0x48, 0x8B, 0x87, 0x10, 0, 0, 0];

            let mut sse = Vec::new();
            x86_64::emit_uniform_load(&mut sse, Reg(5), load);
            assert_eq!(&sse[..7], &MOV_RAX_CTX2);
            assert_eq!(&sse[7..], &[0xC4, 0xE2, 0x79, 0x18, 0xA8, 0x0C, 0, 0, 0]);

            let mut avx2 = Vec::new();
            avx2::emit_uniform_load(&mut avx2, Reg(5), load);
            assert_eq!(&avx2[..7], &MOV_RAX_CTX2);
            assert_eq!(&avx2[7..], &[0xC4, 0xE2, 0x7D, 0x18, 0xA8, 0x0C, 0, 0, 0]);

            let mut avx512 = Vec::new();
            avx512::emit_uniform_load(&mut avx512, Reg(5), load);
            assert_eq!(&avx512[..7], &MOV_RAX_CTX2);
            assert_eq!(
                &avx512[7..],
                &[0x62, 0xF2, 0x7D, 0x48, 0x18, 0xA8, 0x0C, 0, 0, 0]
            );

            let mut neon = Vec::new();
            aarch64::emit_uniform_load(&mut neon, Reg(5), load);
            let words: Vec<u32> = neon
                .chunks(4)
                .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                .collect();
            assert_eq!(words, [0xF940_0809, 0xBD40_0D25, 0x4E04_04A5]);
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
                scratch: regalloc::Scratch::for_test(
                    Some([Reg(15), Reg(14), Reg(13), Reg(12)]),
                    [Some(Reg(11)), Some(Reg(10))],
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
        /// bespoke ternary shapes (`MulAdd`, `Select`) against `backend`,
        /// collecting every failure instead of stopping at the first one —
        /// a completeness gap is much cheaper to fix as an itemized list
        /// than rediscovered one `cargo test` run per missing op.
        fn assert_covers_required_ops<B: IsaBackend>(backend_name: &str, backend: &mut B) {
            // The explicit `try_emit` calls below for MulAdd/Select are this
            // constant, unrolled by hand (each needs its own `ResolvedOp`
            // shape, so they aren't worth a generic loop) — kept in sync
            // deliberately rather than by a shared loop. `MulAdd` unrolls to
            // four: one fused plus one per `DecomposedMulAdd` spelling.
            debug_assert_eq!(REQUIRED_TERNARY_OPS, &[OpKind::MulAdd, OpKind::Select]);
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
                missing.push(alloc::string::String::from("ternary MulAdd (fused)"));
            }
            // `MulAdd` reaches a backend as EITHER shape depending only on how
            // the allocator placed `a` and `b` (see `resolve_operands`), so a
            // backend owes both. Each `c_deferred` spelling is its own arm.
            for (tag, c_deferred) in [
                ("c in a register", None),
                (
                    "c reloaded from the stack",
                    Some(DeferredReload::FromStack(Slot::new(32, 16))),
                ),
                (
                    "c rematerialized",
                    Some(DeferredReload::Const(1.0f32.to_bits())),
                ),
            ] {
                if !try_emit(
                    backend,
                    ResolvedOp::DecomposedMulAdd {
                        dst: Reg(4),
                        a: Reg(5),
                        b: Reg(6),
                        c: Reg(7),
                        c_deferred,
                    },
                ) {
                    missing.push(alloc::format!("ternary MulAdd (decomposed, {tag})"));
                }
            }
            if !try_emit(
                backend,
                ResolvedOp::Select {
                    dst: Reg(4),
                    if_true: Reg(5),
                    if_false: Reg(6),
                },
            ) {
                missing.push(alloc::string::String::from("ternary Select"));
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
        // every host, so a coverage gap in any of the four fails every CI
        // job rather than only the one leg that happens to select it.
        // AVX-512's binary dispatch once shipped 6 of 15 required ops and
        // nothing noticed until someone first built `+avx512f`; that is the
        // hole this closes for all four at once.
        #[test]
        fn x86_backend_covers_required_ops() {
            assert_covers_required_ops(
                "X86Backend (SSE2)",
                &mut x86_64::driver::X86Backend::new(EmitCtx::default()),
            );
        }

        #[test]
        fn avx2_backend_covers_required_ops() {
            assert_covers_required_ops(
                "Avx2Backend",
                &mut avx2::driver::Avx2Backend::new(EmitCtx::default()),
            );
        }

        #[test]
        fn avx512_backend_covers_required_ops() {
            assert_covers_required_ops(
                "Avx512Backend",
                &mut avx512::driver::Avx512Backend::new(EmitCtx::default()),
            );
        }

        #[test]
        fn aarch64_backend_covers_required_ops() {
            let mut backend = aarch64::driver::Aarch64Backend::new(EmitCtx::default());
            assert_covers_required_ops("Aarch64Backend", &mut backend);
        }
    }

    // =========================================================================
    // MulAdd: the encodings behind the two `ResolvedOp` shapes.
    //
    // `MulAdd` is the one row of CLAUDE.md's platform-divergence table whose
    // two answers live inside a single build: `FusedMulAdd` rounds once where
    // the hardware has an FMA, `DecomposedMulAdd` is architecturally a
    // multiply then an add and rounds twice, and which one a node gets is
    // decided by register pressure alone (`resolve_operands`). So the shapes
    // are pinned as *bytes*, not just as "it emitted something": a backend
    // that quietly encoded one where the driver asked for the other would
    // still satisfy `backend_op_coverage`, still pass every ULP-tolerant
    // equivalence test, and change the last bit of the answer.
    //
    // Ungated, like `backend_op_coverage`: encoding is a pure function into a
    // `Vec<u8>`, so all four backends are checked from whichever host runs the
    // tests — including the two (aarch64, AVX-512 decomposed) that no
    // execution test on any single host reaches.
    // =========================================================================
    mod muladd_encoding {
        use super::*;

        const DST: Reg = Reg(4);
        const SRC_A: Reg = Reg(5);
        const SRC_B: Reg = Reg(6);
        const ADDEND: Reg = Reg(7);

        /// The temp the SSE2 fused stand-in multiplies into. Any pool
        /// register disjoint from the operands would do — the allocator picks
        /// it per instruction — so this names one to pin the bytes.
        const TEMP: Reg = Reg(10);

        /// A bare plan: no reloads, no setup mov, no store — just the op, so
        /// the bytes below are the op's encoding and nothing else.
        ///
        /// `temps` is empty for every spelling but SSE2's `FusedMulAdd`, which
        /// has no FMA to fuse into and needs somewhere to put the product; the
        /// empty set elsewhere is the assertion that no other backend starts
        /// asking for scratch unnoticed.
        fn plan(
            op: ResolvedOp,
            temps: Option<[Reg; regalloc::Scratch::MAX_TEMPS]>,
        ) -> InstructionPlan {
            InstructionPlan {
                reloads: alloc::vec::Vec::new(),
                op,
                setup_mov: None,
                scratch: regalloc::Scratch::for_test(temps, [None, None]),
            }
        }

        fn encode<B: IsaBackend>(backend: &mut B, op: ResolvedOp) -> Vec<u8> {
            let mut code = Vec::new();
            backend
                .emit_plan(&mut code, &plan(op, None))
                .expect("emit_plan");
            code
        }

        /// `encode` for the one spelling that asks the allocator for a temp.
        fn encode_with_temp<B: IsaBackend>(backend: &mut B, op: ResolvedOp) -> Vec<u8> {
            let mut code = Vec::new();
            backend
                .emit_plan(
                    &mut code,
                    &plan(op, Some([TEMP; regalloc::Scratch::MAX_TEMPS])),
                )
                .expect("emit_plan");
            code
        }

        fn fused() -> ResolvedOp {
            ResolvedOp::FusedMulAdd {
                dst: DST,
                a: SRC_A,
                b: SRC_B,
            }
        }

        fn decomposed(c_deferred: Option<DeferredReload>) -> ResolvedOp {
            ResolvedOp::DecomposedMulAdd {
                dst: DST,
                a: SRC_A,
                b: SRC_B,
                c: ADDEND,
                c_deferred,
            }
        }

        /// `dst += a * b` in one instruction, one rounding, on the three
        /// targets that have an FMA — and the SSE2 baseline's honest
        /// three-instruction stand-in, which rounds twice because that is all
        /// the hardware offers.
        #[test]
        fn fused_encodes_to_the_targets_fma() {
            // VEX.256.66.0F38.W0 B8 /r — vfmadd231ps ymm4, ymm5, ymm6.
            assert_eq!(
                encode(
                    &mut avx2::driver::Avx2Backend::new(EmitCtx::default()),
                    fused()
                ),
                alloc::vec![0xc4, 0xe2, 0x55, 0xb8, 0xe6],
                "AVX2 fused MulAdd"
            );
            // EVEX.512.66.0F38.W0 B8 /r — vfmadd231ps zmm4, zmm5, zmm6.
            assert_eq!(
                encode(
                    &mut avx512::driver::Avx512Backend::new(EmitCtx::default()),
                    fused()
                ),
                alloc::vec![0x62, 0xf2, 0x55, 0x48, 0xb8, 0xe6],
                "AVX-512 fused MulAdd"
            );
            // FMLA V4.4S, V5.4S, V6.4S.
            let neon = encode(
                &mut aarch64::driver::Aarch64Backend::new(EmitCtx::default()),
                fused(),
            );
            assert_eq!(
                aarch64::disassemble_code(&neon).trim_end(),
                "   0: 4e26cca4  fmla v4.4s, v5.4s, v6.4s",
                "aarch64 fused MulAdd"
            );
            // No FMA at the SSE2 baseline: movaps/mulps into this
            // instruction's temp, then addps into dst. Two roundings, and the
            // only reason CLAUDE.md's `MulAdd` row still has a second column.
            assert_eq!(
                encode_with_temp(
                    &mut x86_64::driver::X86Backend::new(EmitCtx::default()),
                    fused()
                ),
                alloc::vec![
                    0x44, 0x0f, 0x28, 0xd5, // movaps xmm10, xmm5
                    0x44, 0x0f, 0x59, 0xd6, // mulps  xmm10, xmm6
                    0x41, 0x0f, 0x58, 0xe2, // addps  xmm4,  xmm10
                ],
                "SSE2 fused MulAdd"
            );
        }

        /// The decomposed shape is a multiply and an add — never an FMA, on
        /// any target. A backend that "optimized" it back into one instruction
        /// would change the result's last bit while every tolerant test kept
        /// passing.
        #[test]
        fn decomposed_encodes_to_a_multiply_and_an_add() {
            assert_eq!(
                encode(
                    &mut avx2::driver::Avx2Backend::new(EmitCtx::default()),
                    decomposed(None)
                ),
                alloc::vec![
                    0xc4, 0xe1, 0x54, 0x59, 0xe6, // vmulps ymm4, ymm5, ymm6
                    0xc4, 0xe1, 0x5c, 0x58, 0xe7, // vaddps ymm4, ymm4, ymm7
                ],
                "AVX2 decomposed MulAdd"
            );
            assert_eq!(
                encode(
                    &mut avx512::driver::Avx512Backend::new(EmitCtx::default()),
                    decomposed(None)
                ),
                alloc::vec![
                    0x62, 0xf1, 0x54, 0x48, 0x59, 0xe6, // vmulps zmm4, zmm5, zmm6
                    0x62, 0xf1, 0x5c, 0x48, 0x58, 0xe7, // vaddps zmm4, zmm4, zmm7
                ],
                "AVX-512 decomposed MulAdd"
            );
            let neon = encode(
                &mut aarch64::driver::Aarch64Backend::new(EmitCtx::default()),
                decomposed(None),
            );
            assert_eq!(
                aarch64::disassemble_code(&neon).trim_end(),
                "   0: 6e26dca4  fmul v4.4s, v5.4s, v6.4s\n   4: 4e27d484  fadd v4.4s, v4.4s, v7.4s",
                "aarch64 decomposed MulAdd"
            );
            assert_eq!(
                encode(
                    &mut x86_64::driver::X86Backend::new(EmitCtx::default()),
                    decomposed(None)
                ),
                alloc::vec![
                    0x0f, 0x28, 0xe5, // movaps xmm4, xmm5
                    0x0f, 0x59, 0xe6, // mulps  xmm4, xmm6
                    0x0f, 0x58, 0xe7, // addps  xmm4, xmm7
                ],
                "SSE2 decomposed MulAdd"
            );
        }

        /// A deferred `c` must be reloaded *between* the multiply and the add.
        ///
        /// That ordering is the whole reason `DeferredReload` exists: `c`'s
        /// reload target is the same scratch register `b` was loaded into, so
        /// hoisting it up with the other reloads would destroy `b` before the
        /// multiply reads it. The invariant is checked structurally rather
        /// than as another byte literal — the multiply and the add are already
        /// pinned above, so what is left to prove is that the reload landed
        /// strictly between them, on every backend.
        #[test]
        fn a_deferred_c_is_reloaded_between_the_multiply_and_the_add() {
            fn check<B: IsaBackend>(name: &str, backend: &mut B) {
                let undeferred = encode(backend, decomposed(None));
                // `dst = a*b` is everything before the final add; on SSE2 the
                // add is 3 bytes, on VEX 5, on EVEX 6, on NEON 4 — so split by
                // the tail rather than by a per-backend length.
                let (mul, add) = undeferred.split_at(undeferred.len() - tail_len(name));
                for deferred in [
                    DeferredReload::FromStack(Slot::new(32, 16)),
                    DeferredReload::Const(1.0f32.to_bits()),
                ] {
                    let got = encode(backend, decomposed(Some(deferred.clone())));
                    assert!(
                        got.starts_with(mul),
                        "{name}/{deferred:?}: the multiply is no longer first"
                    );
                    assert!(
                        got.ends_with(add),
                        "{name}/{deferred:?}: the add is no longer last"
                    );
                    assert!(
                        got.len() > undeferred.len(),
                        "{name}/{deferred:?}: nothing was emitted for the reload"
                    );
                }
            }

            /// Byte length of the trailing add in `decomposed(None)`.
            fn tail_len(name: &str) -> usize {
                match name {
                    "SSE2" => 3,
                    "AVX2" => 5,
                    "AVX-512" => 6,
                    "aarch64" => 4,
                    other => panic!("unknown backend {other}"),
                }
            }

            check(
                "SSE2",
                &mut x86_64::driver::X86Backend::new(EmitCtx::default()),
            );
            check(
                "AVX2",
                &mut avx2::driver::Avx2Backend::new(EmitCtx::default()),
            );
            check(
                "AVX-512",
                &mut avx512::driver::Avx512Backend::new(EmitCtx::default()),
            );
            check(
                "aarch64",
                &mut aarch64::driver::Aarch64Backend::new(EmitCtx::default()),
            );
        }

        /// A `MulAdd` node really does reach a backend as `FusedMulAdd` when
        /// nothing spills — the property the byte tests above assume, and the
        /// one an upstream change (a legalization pass that decomposed it, an
        /// arena builder that never emitted it) would silently take away.
        #[test]
        fn a_muladd_dag_emits_the_fused_encoding() {
            use pixelflow_ir::arena::ExprArena;

            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let z = a.push_binary(OpKind::Add, y, x);
            let root = a.push_ternary(OpKind::MulAdd, x, y, z);
            let (a, root) = pixelflow_ir::passes::legalize(&a, root).expect("legalize");

            let (code, _, _, _) = emit_dag_body(
                arena_to_schedule(&a, root),
                &mut avx2::driver::Avx2Backend::new(EmitCtx::default()),
            )
            .expect("AVX2 emit");
            // vfmadd231ps: VEX.256.66.0F38 B8 — the opcode byte after the
            // 3-byte prefix. Nothing else this body emits uses it.
            assert!(
                code.windows(4)
                    .any(|w| w[0] == 0xc4 && w[1] == 0xe2 && w[3] == 0xb8),
                "a MulAdd DAG did not reach the AVX2 backend as FusedMulAdd \
                 (no VEX.0F38 B8 in {code:02x?})"
            );
        }
    }
}
