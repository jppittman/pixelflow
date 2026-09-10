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
pub(crate) mod demand;
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

    /// Compile an expression through the DAG-facing API.
    ///
    /// The legacy arena is used only as the private compatibility boundary
    /// required by the current legalization passes.  Once legalization has
    /// finished, scheduling and emission consume `Node` handles exclusively;
    /// callers never need to carry an arena and an index together.
    pub fn compile_dag(
        self,
        root: pixelflow_ir::Node<'_, pixelflow_ir::ExprData>,
        env: &pixelflow_ir::Environment,
    ) -> Result<CompileResult, CompileError> {
        let (legacy, legacy_root) = root.marshal(env);
        let (legalized, legalized_root) =
            pixelflow_ir::passes::legalize(&legacy, legacy_root).map_err(CompileError::Legalize)?;
        let (rooted, legalized_env) =
            pixelflow_ir::Rooted::unmarshal(&legalized, &[legalized_root]);
        let schedule = dag_to_schedule(rooted.entry(), &legalized_env);
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
    /// The DAG scheduler (so it never becomes a scheduled value / register).
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
// DAG to Schedule (zero-cost linearization)
// =============================================================================

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

/// Build a schedule from a rooted expression DAG.
///
/// `Dag` iteration is already children-before-parents, so the scheduler only
/// needs a side table to retain the dense `ValueId` mapping.  This keeps the
/// representation boundary one-way: expression storage remains owned by the
/// IR DAG and the emitter sees borrowed node handles, never raw offsets.
fn dag_to_schedule(
    root: pixelflow_ir::Node<'_, pixelflow_ir::ExprData>,
    env: &pixelflow_ir::Environment,
) -> Vec<regalloc::Def> {
    use pixelflow_ir::ExprData;
    use regalloc::ValueId;

    let dag = root.dag();
    let mut reachable = dag.side_table(false);
    for node in root.descendants() {
        reachable[node] = true;
    }

    let mut ids = dag.side_table(None::<ValueId>);
    let mut schedule = Vec::new();
    for node in dag.iter() {
        if !reachable[node] {
            continue;
        }
        let value = ValueId(schedule.len() as u32);
        ids[node] = Some(value);
        let mut children = node.children();
        let child = |node: pixelflow_ir::Node<'_, ExprData>,
                     ids: &pixelflow_ir::SideTable<Option<ValueId>>| {
            ids[node].expect("dag_to_schedule: child must precede parent")
        };
        let sched_op = match *node {
            ExprData::Var(i) => ScheduledOp::Var(i),
            ExprData::Const(bits) => ScheduledOp::Const(f32::from_bits(bits)),
            ExprData::Param(i) => panic!(
                "ExprData::Param({i}) reached the JIT emitter -- call substitute_params before compile_dag()"
            ),
            ExprData::Buffer(_) => ScheduledOp::Const(0.0),
            ExprData::Uniform(u) => ScheduledOp::Uniform(UniformLoad {
                ctx_slot: u16::try_from(env.buffers.len())
                    .expect("buffer table index fits the context slot immediate"),
                offset: u.0,
            }),
            ExprData::Reduce(_) => {
                panic!("a bounded fold reached the JIT emitter -- run passes::expand_reduce first")
            }
            ExprData::Ref(key) => panic!(
                "dag_to_schedule: {key:?} names a kernel whose body is not in this DAG; expand_refs runs first"
            ),
            ExprData::Op(op) => match node.child_count() {
                1 => ScheduledOp::Unary(op, child(children.next().expect("unary child"), &ids)),
                2 if matches!(op, OpKind::Shl | OpKind::Shr) => {
                    let a = children.next().expect("shift value");
                    let b = children.next().expect("shift count");
                    let amount = match *b {
                        ExprData::Const(bits) => shift_immediate(op, f32::from_bits(bits)),
                        other => panic!("{op:?} shift count must be a Const, got {other:?}"),
                    };
                    ScheduledOp::ShiftImm(op, child(a, &ids), amount)
                }
                2 if op == OpKind::RawGather => {
                    let buf = children.next().expect("gather buffer");
                    let idx = children.next().expect("gather index");
                    let slot = match *buf {
                        ExprData::Buffer(id) => id.0,
                        other => {
                            panic!("RawGather's first child must be a Buffer leaf, got {other:?}")
                        }
                    };
                    ScheduledOp::Gather(child(idx, &ids), slot)
                }
                2 if op == OpKind::Dwrt => panic!(
                    "dag_to_schedule: a Dwrt node reached the JIT emitter; lower_dwrt must eliminate it"
                ),
                2 => {
                    let a = children.next().expect("binary left child");
                    let b = children.next().expect("binary right child");
                    ScheduledOp::Binary(op, child(a, &ids), child(b, &ids))
                }
                3 => {
                    let a = children.next().expect("ternary first child");
                    let b = children.next().expect("ternary second child");
                    let c = children.next().expect("ternary third child");
                    ScheduledOp::Ternary(op, child(a, &ids), child(b, &ids), child(c, &ids))
                }
                arity => panic!("Nary expression with arity {arity} reached the JIT emitter"),
            },
        };
        schedule.push(regalloc::Def {
            value,
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
/// The schedule mirrors the DAG's topological order, so one forward pass
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

/// Compile a rooted expression DAG into a **collapse** kernel: the X/Y loop nest is
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
/// The context is one base pointer per declared buffer, in the environment's slot
/// order, followed — only when the environment declares a uniform — by the uniform
/// block's base pointer: `f32` values in the DAG environment's uniform-slot order, read
/// once per call in the frame prologue.
///
/// # Panics
///
/// Panics if the DAG names a retired coordinate axis (`Var(2)`/`Var(3)`,
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
/// Compile a rooted expression through the DAG-facing codegen boundary.
#[must_use = "code generation errors must be handled"]
pub fn compile(
    root: pixelflow_ir::Node<'_, pixelflow_ir::ExprData>,
    env: &pixelflow_ir::Environment,
) -> Result<CompileResult, CompileError> {
    EmitCtx::default().compile_dag(root, env)
}

/// Explicit spelling for callers that want to make the DAG boundary visible.
#[must_use = "code generation errors must be handled"]
pub fn compile_dag(
    root: pixelflow_ir::Node<'_, pixelflow_ir::ExprData>,
    env: &pixelflow_ir::Environment,
) -> Result<CompileResult, CompileError> {
    compile(root, env)
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
    use pixelflow_ir::{ExprBuilder, ExprData, ExprGraph, OpKind};

    fn schedule(graph: ExprGraph) -> Vec<regalloc::Def> {
        dag_to_schedule(graph.root(), graph.environment())
    }

    #[test]
    #[should_panic(expected = "Dwrt node reached")]
    fn surviving_dwrt_fails_loudly() {
        let mut b = ExprBuilder::new();
        let x = b.var(0);
        let zero = b.constant(0.0);
        let root = b.binary(OpKind::Dwrt, x, zero);
        let graph = b.finish_one(root);
        let _ = schedule(graph);
    }

    #[test]
    #[should_panic(expected = "names a kernel whose body is not in this DAG")]
    fn surviving_reference_fails_loudly() {
        let body = pixelflow_ir::Kernel::x();
        let key = pixelflow_ir::KernelKey::of(body.root(), &body.environment());
        let mut b = ExprBuilder::new();
        let root = b.reference(key);
        let graph = b.finish_one(root);
        let _ = schedule(graph);
    }

    #[test]
    fn scheduler_maps_dag_nodes_in_topological_order() {
        let mut b = ExprBuilder::new();
        let x = b.var(0);
        let y = b.var(1);
        let c = b.constant(2.0);
        let product = b.binary(OpKind::Mul, x, c);
        let root = b.binary(OpKind::Add, product, y);
        let schedule = schedule(b.finish_one(root));
        assert_eq!(schedule.len(), 5);
        assert!(matches!(
            schedule.last().map(|d| &d.op),
            Some(ScheduledOp::Binary(OpKind::Add, _, _))
        ));
    }

    #[test]
    fn scheduler_folds_shift_immediate() {
        let mut b = ExprBuilder::new();
        let x = b.var(0);
        let amount = b.constant(3.0);
        let root = b.binary(OpKind::Shl, x, amount);
        let schedule = schedule(b.finish_one(root));
        assert!(
            schedule
                .iter()
                .any(|d| matches!(d.op, ScheduledOp::ShiftImm(OpKind::Shl, _, 3)))
        );
    }

    #[test]
    fn scheduler_preserves_uniform_environment() {
        let mut b = ExprBuilder::new();
        let x = b.var(0);
        let uniform = b.uniform(pixelflow_ir::arena::UniformDecl {
            id: pixelflow_ir::arena::UniformIdentity::mint(),
            default: 1.0,
        });
        let root = b.binary(OpKind::Mul, x, uniform);
        let graph = b.finish_one(root);
        let schedule = schedule(graph);
        assert!(
            schedule
                .iter()
                .any(|d| matches!(d.op, ScheduledOp::Uniform(_)))
        );
    }

    #[test]
    fn every_backend_emits_a_simple_dag() {
        let mut b = ExprBuilder::new();
        let x = b.var(0);
        let y = b.var(1);
        let root = b.binary(OpKind::Add, x, y);
        let graph = b.finish_one(root);
        let schedule = schedule(graph);
        for (name, result) in [
            (
                "aarch64",
                emit_dag_body(
                    schedule.clone(),
                    &mut aarch64::driver::Aarch64Backend::new(EmitCtx::default()),
                ),
            ),
            (
                "x86_64",
                emit_dag_body(
                    schedule.clone(),
                    &mut x86_64::driver::X86Backend::new(EmitCtx::default()),
                ),
            ),
            (
                "avx2",
                emit_dag_body(
                    schedule.clone(),
                    &mut avx2::driver::Avx2Backend::new(EmitCtx::default()),
                ),
            ),
            (
                "avx512",
                emit_dag_body(
                    schedule.clone(),
                    &mut avx512::driver::Avx512Backend::new(EmitCtx::default()),
                ),
            ),
        ] {
            let (bytes, _, _, _) =
                result.unwrap_or_else(|e| panic!("{name} emission failed: {e:?}"));
            assert!(!bytes.is_empty(), "{name} emitted no instructions");
        }
    }

    #[test]
    fn muladd_reaches_the_backend() {
        let mut b = ExprBuilder::new();
        let x = b.var(0);
        let y = b.var(1);
        let z = b.constant(2.0);
        let root = b.ternary(OpKind::MulAdd, x, y, z);
        let graph = b.finish_one(root);
        let schedule = schedule(graph);
        let (bytes, _, _, _) = emit_dag_body(
            schedule,
            &mut avx2::driver::Avx2Backend::new(EmitCtx::default()),
        )
        .expect("AVX2 emit");
        assert!(!bytes.is_empty());
    }

    #[test]
    fn compile_entry_accepts_a_rooted_graph() {
        let mut b = ExprBuilder::new();
        let x = b.var(0);
        let y = b.var(1);
        let root = b.binary(OpKind::Add, x, y);
        let graph = b.finish_one(root);
        let result = compile(graph.root(), graph.environment());
        assert!(result.is_ok(), "DAG compile failed");
    }

    #[test]
    fn expr_data_is_pure_payload() {
        let mut b = ExprBuilder::new();
        let x = b.var(0);
        let root = b.unary(OpKind::Neg, x);
        let graph = b.finish_one(root);
        assert!(matches!(*graph.root(), ExprData::Op(OpKind::Neg)));
        assert_eq!(graph.root().child_count(), 1);
    }
}
