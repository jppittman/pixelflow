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
//! ([`IsaBackend`]). What a backend contributes is its `RegisterFile` — the
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
//! Values the scratch pool cannot hold go to stack slots, laid out by
//! [`FrameLayout`] at the backend's vector stride:
//! - A value with a slot is stored to it right after its **definition**, which
//!   every path that reads the value has run — including through an `If`
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

use pixelflow_ir::fold::{Fold, RangeFold};
use pixelflow_ir::kind::OpKind;

pub use guards::IfArm;
// Production code reads guards off the allocation (`Allocation::if_guards`)
// rather than calling this directly — see `emit_scope`. Only the tests, which
// exercise the analysis against hand-built schedules the allocator never
// sees, call it themselves.
#[cfg(test)]
use guards::analyze_if_guards;
use traffic::{Counting, EmitTraffic};

use alloc::vec::Vec;

use crate::error::CompileError;
use crate::isa::Isa;
use pixelflow_ir::arena::{UniformDecl, UniformId, UniformIdentity};
use pixelflow_ir::fold::{Binder, Monoid};
use pixelflow_ir::passes::lattice::{Collapse, Domain};
use pixelflow_ir::variance::LatticeShape;

/// The one contract every backend's instruction types satisfy.
pub trait AsmInsn: Copy {
    /// Emit the instruction's encoded bytes into the output buffer.
    fn emit_into(self, code: &mut Vec<u8>);

    /// The position this instruction's bytes depend on, if any.
    ///
    /// Almost every instruction is position-independent and takes the default.
    /// A branch is not: it emits a placeholder displacement in `emit_into` and
    /// says here which [`Label`] it is waiting on and how to fill the
    /// placeholder in. That is the whole of what a branch adds — it is an
    /// ordinary instruction that takes a name instead of a number.
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

impl<I: AsmInsn, const N: usize> From<[I; N]> for AsmProgram<[Item<I>; N]> {
    #[inline(always)]
    fn from(insts: [I; N]) -> Self {
        Self {
            insts: insts.map(Item::Inst),
        }
    }
}

impl<I: AsmInsn, const N: usize> From<[Item<I>; N]> for AsmProgram<[Item<I>; N]> {
    #[inline(always)]
    fn from(insts: [Item<I>; N]) -> Self {
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

impl<I: AsmInsn, S: IntoIterator<Item = Item<I>>> AsmProgram<S> {
    /// Assemble the program into the machine-code buffer.
    ///
    /// Lay the items out, then fill in the displacements that could not be
    /// known until the layout was. A program with no [`Item::Label`] in it
    /// never reaches the second pass, which is why it used to be a one-pass
    /// map — that is the only thing labels changed.
    ///
    /// # Panics
    ///
    /// If a label is bound twice, or a branch names one that nothing bound.
    /// Only this crate writes these programs, so either is a bug here rather
    /// than a fact about the kernel being compiled.
    #[inline]
    pub fn assemble(self, code: &mut Vec<u8>) {
        let mut asm = Assembly::from_code(core::mem::take(code));
        for item in self.insts {
            match item {
                Item::Inst(inst) => asm.push(inst),
                Item::Label(label) => asm.bind(label),
            }
        }
        *code = asm.finish();
    }
}

impl<I: AsmInsn, S: IntoIterator<Item = Item<I>> + Copy> AsmInsn for AsmProgram<S> {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        self.assemble(code);
    }
}

/// Free-function fold: assemble a declarative sequence directly into `code`.
#[inline]
pub fn assemble<I: AsmInsn>(code: &mut Vec<u8>, insts: impl IntoIterator<Item = I>) {
    AsmProgram::new(insts.into_iter().map(Item::Inst)).assemble(code);
}

// =============================================================================
// Labels: a name for a position, bound at assembly time
// =============================================================================

/// A name for a position in the emitted program.
///
/// Most instructions are position-*independent*: they write their own bytes and
/// do not care where they sit. A branch is the exception, and it used to be
/// handled outside the assembler entirely — `emit_jump` returned a fixup token,
/// the caller carried it to a `patch_branch` twenty lines later, and the target
/// was a `code.len()` read off at the one point in the sequence where that was
/// correct.
///
/// A label is the missing name, and it makes a branch an ordinary instruction
/// again: [`x86_64::Jmp`] and friends *take a `Label`*. Assembling is then two
/// passes instead of one — lay the items out, then fill in the displacements
/// that could not be known until the layout was — which is the only thing that
/// changed.
/// A label is a **name**.
///
/// That is the whole of it. You write instructions and labels, you assemble,
/// you get a binary; addresses never come back out, and the caller is not a
/// participant in working them out. Mapping names to hex is the assembler's
/// job, which is the only reason to have one.
///
/// So this is not a handle. There is nothing to mint, nothing to keep, and no
/// table to look a name up in — a branch to `"row_top"` and the `"row_top"`
/// written later in the stream are the same label because they are the same
/// name. Whatever the emitter uses to *build* a name — a schedule index, a
/// `ValueId`, a guard arm — is its own business and stops here.
///
/// The name is inline rather than a `String` so that a label is `Copy`: a
/// branch instruction holds one, and [`AsmInsn`] is `Copy`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Label {
    name: [u8; Self::CAPACITY],
    len: u8,
}

impl Label {
    /// The longest a name may be. Generous for the names this emitter writes
    /// (`"batch_exit"`, `"v1234_past_true"`) and small enough that carrying
    /// one inside an instruction is free.
    pub const CAPACITY: usize = 31;

    /// The label called `name`.
    ///
    /// # Panics
    ///
    /// If `name` is longer than [`Label::CAPACITY`]. Only this crate writes
    /// these programs, and a truncated name is one that silently aliases
    /// another — so it refuses rather than trims.
    #[must_use]
    pub fn new(name: &str) -> Self {
        let bytes = name.as_bytes();
        assert!(
            bytes.len() <= Self::CAPACITY,
            "label {name:?} is longer than {} bytes",
            Self::CAPACITY
        );
        let mut buffer = [0u8; Self::CAPACITY];
        buffer[..bytes.len()].copy_from_slice(bytes);
        Self {
            name: buffer,
            len: bytes.len() as u8,
        }
    }

    /// The name, as written.
    #[must_use]
    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.name[..self.len as usize])
            .unwrap_or_else(|_| unreachable!("built from a &str"))
    }
}

impl From<&str> for Label {
    fn from(name: &str) -> Self {
        Self::new(name)
    }
}

impl core::fmt::Display for Label {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl core::fmt::Debug for Label {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self.as_str())
    }
}

/// How an instruction whose bytes depend on a position gets those bytes.
///
/// Returned by [`AsmInsn::label_ref`]. The instruction emits a placeholder in
/// `emit_into`; `patch` fills the displacement in once the label's position is
/// known. A function pointer rather than a trait object or a type parameter
/// because the encoding is the instruction's own business and nothing else in
/// the assembler needs to know it — an x86 `rel32` four bytes in, an aarch64
/// `imm19` five bits up in the word.
#[derive(Copy, Clone)]
pub struct LabelRef {
    /// The position this instruction is waiting on.
    pub label: Label,
    /// Fill in the displacement of an instruction that was emitted at `at`, so
    /// that it reaches `target`. Both are offsets from the start of the
    /// program.
    pub patch: fn(code: &mut [u8], at: usize, target: usize),
}

/// One item of an assembly program.
///
/// An instruction, or a name bound to this position. That is the whole of what
/// an assembler takes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Item<I> {
    /// Bytes.
    Inst(I),
    /// Bind `label` here. Emits nothing.
    Label(Label),
}

impl<I> From<I> for Item<I> {
    #[inline]
    fn from(i: I) -> Self {
        Item::Inst(i)
    }
}

/// A program being assembled: its bytes, and the names in it.
///
/// The imperative face of the same assembler [`AsmProgram`] is the declarative
/// face of. An emitter that walks a schedule cannot hand over a finished list
/// of [`Item`]s — it discovers them as it goes, calling `&mut self` backend
/// verbs for each — so it pushes into one of these instead. Same label map,
/// same two passes.
#[derive(Default)]
pub struct Assembly {
    /// The bytes so far. Public because emitting into it is what a backend verb
    /// does.
    pub code: Vec<u8>,
    bound: alloc::collections::BTreeMap<Label, usize>,
    pending: Vec<(usize, LabelRef)>,
}

impl Assembly {
    /// An empty program with room for `capacity` bytes.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            code: Vec::with_capacity(capacity),
            ..Self::default()
        }
    }

    /// Continue a program whose first bytes are already emitted.
    ///
    /// Positions are offsets into the whole buffer, not into the part this
    /// program contributed. A displacement cannot tell the difference — it is
    /// `target - at` either way — but a *page* can, and `AdrpAdd` asks for
    /// one, so there is exactly one answer to what a position means here and
    /// this is it.
    #[must_use]
    pub fn from_code(code: Vec<u8>) -> Self {
        Self {
            code,
            ..Self::default()
        }
    }

    /// Write a label here — the name of this position.
    ///
    /// A branch may name a position before it exists, which is every forward
    /// branch and the exit of every loop, so nothing here checks that anything
    /// refers to it. [`Assembly::finish`] is where a name nobody wrote is
    /// reported.
    ///
    /// # Panics
    ///
    /// If this name is already written elsewhere in the program. A name that
    /// means two positions is not a name, and only this crate writes these
    /// programs, so that is a bug here rather than anything about the kernel
    /// being compiled.
    pub fn bind(&mut self, label: impl Into<Label>) {
        let (label, at) = (label.into(), self.code.len());
        let previously = self.bound.insert(label, at);
        assert!(previously.is_none(), "{label} was written twice");
    }

    /// Emit one instruction, recording the name it waits on if it has one.
    pub fn push(&mut self, inst: impl AsmInsn) {
        let at = self.code.len();
        inst.emit_into(&mut self.code);
        if let Some(reference) = inst.label_ref() {
            self.pending.push((at, reference));
        }
    }

    /// Fill in every deferred displacement and hand back the bytes.
    ///
    /// # Panics
    ///
    /// If a branch names a label nothing bound.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        for (at, reference) in core::mem::take(&mut self.pending) {
            let Some(&target) = self.bound.get(&reference.label) else {
                panic!("{} is branched to but never written", reference.label)
            };
            (reference.patch)(&mut self.code, at, target);
        }
        self.code
    }
}

/// What the constant pool is called.
///
/// One name per emitted function, because there is one pool per emitted
/// function: the anchor names it before a single constant is known, and the
/// pool is written where it lands, after the return. Nothing is carried
/// between the two — they agree because they spell the same thing. Every
/// backend uses it: aarch64 anchors `X17` to it and x86 anchors `r8`, and a
/// kernel's constant loads are then one instruction each, base-relative.
pub const CONST_POOL: &str = "const_pool";

/// The constant pool's alignment: one NEON pool entry, so every `LDR Qt` from
/// it is an aligned vector load. x86's four-byte entries need no alignment and
/// take this one for the cache line. The padding that reaches it from the
/// last instruction follows the code's length, which is why a kernel's
/// trailing bytes can differ between two allocations of it by less than this.
pub const CONST_POOL_ALIGN: usize = 16;

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
    /// Value is in a vector register.
    Reg(Reg),
    /// Value is an address, in a pointer register — a
    /// [`regalloc::Class::Pointer`] value's only kind of register.
    Ptr(PtrReg),
    /// Value is spilled to a stack slot.
    Slot(Slot),
}

impl Loc {
    /// Get the vector register, panicking if the value is not in one.
    #[must_use]
    pub fn reg(self) -> Reg {
        match self {
            Loc::Reg(r) => r,
            Loc::Ptr(p) => panic!("expected a vector register, got pointer register {p:?}"),
            Loc::Slot(s) => panic!("expected register, got stack slot {}", s.offset()),
        }
    }

    /// Physical storage location.
    #[must_use]
    pub fn storage(self) -> Storage {
        match self {
            Loc::Reg(r) => Storage::Reg(r),
            Loc::Ptr(p) => Storage::Ptr(p),
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
            Loc::Ptr(_) | Loc::Slot(_) => None,
        }
    }
    #[inline]
    fn target_slot(self) -> Option<Slot> {
        match self {
            Loc::Reg(_) | Loc::Ptr(_) => None,
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
            Loc::Ptr(_) | Loc::Slot(_) => None,
        }
    }
    #[inline]
    fn source_slot(self) -> Option<Slot> {
        match self {
            Loc::Reg(_) | Loc::Ptr(_) => None,
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
    /// Give every spilled value in this scope a stack address, from `base` up.
    ///
    /// Pure: (scope allocation, slot stride, base) → layout. The collapse
    /// driver runs this twice for one region and relies on both runs agreeing.
    ///
    /// `base` is what keeps a nested scope off its parent's slots — see
    /// [`StackFrame::with_base`]. [`Self::frame_size`] is the resulting total
    /// extent, base included, so a parent's frame size is exactly the base to
    /// hand whatever runs inside it.
    pub fn resolve(
        allocation: regalloc::Allocation<'_>,
        vector_bytes: u32,
        base: u32,
    ) -> Result<Self, CompileError> {
        let schedule = allocation.schedule();
        let len = schedule
            .iter()
            .map(|def| def.value.0 as usize + 1)
            .max()
            .unwrap_or(0);
        let mut locs: alloc::vec::Vec<Option<Binding>> = alloc::vec![None; len];

        let mut frame = StackFrame::with_base(vector_bytes, base);
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
                regalloc::Where::Ptr(p) => Binding::Loc(Loc::Ptr(p)),
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
            regalloc::Where::Ptr(p) => Binding::Loc(Loc::Ptr(p)),
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

/// One unary instruction as a backend's `emit_unary` takes it: the op, its
/// two registers, and the allocator's temp for the instruction, which the ops
/// that build a mask or a correction term write and the rest ignore.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Unary {
    pub op: OpKind,
    pub dst: Reg,
    pub src: Reg,
    pub temp: Option<Reg>,
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
    /// [`regalloc::Class::Pointer`] value, and the only instruction that
    /// reads [`regalloc::RegisterFile::gpr_ctx`].
    Context { dst: PtrReg, slot: u16 },
    /// The lane fold's binder, materialized: `dst = [0, 1, …, L−1]` as
    /// `f32`s, `L` being the backend's lane count. The one vector constant
    /// that is not a broadcast, and the whole of what "executed by lanes"
    /// costs the body.
    Lanes { dst: Reg },
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
    /// Load an address from its stack slot into the pointer register the
    /// allocator reserved for this instruction's base
    /// ([`regalloc::Scratch::ptr_reload`]).
    Ptr { target: PtrReg, slot: Slot },
}

/// Fully resolved instruction: what to reload, and what to compute.
///
/// No store. A destination is always a register now, so the one place a value
/// reaches its slot is the emit loop's store-after-definition — which is what
/// makes the slot valid on every path an `If` guard can take.
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
    /// an `If` or FMA on every ISA, the decomposed `MulAdd`'s multiply
    /// before its add), so this is the one register-level alias any of them
    /// tolerates.
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
        | ScheduledOp::Seq(..) => 0,
        ScheduledOp::Unary(..)
        | ScheduledOp::ShiftImm(..)
        | ScheduledOp::Gather(..)
        | ScheduledOp::Broadcast(..)
        | ScheduledOp::Write { .. }
        // A `Guard`'s one register operand is its mask — its own arm
        // resolution never reaches `resolve_operands`/`operand_sources`
        // (`emit_scope` special-cases it before either is called, the same
        // way it special-cases `Reduce`), but the mask is still resolved the
        // ordinary way, from wherever the allocator put it.
        | ScheduledOp::Guard(..) => 1,
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

/// The GPR-class mirror of [`declared_temp`], for
/// [`regalloc::RegisterFile::gpr_temps_for`].
#[track_caller]
pub(crate) fn declared_gpr_temp(temp: Option<Gpr>) -> Gpr {
    temp.expect("this encoding needs a GPR that `RegisterFile::gpr_temps_for` did not ask for")
}

/// The mask-class mirror of [`declared_temp`], for
/// [`regalloc::RegisterFile::mask_temps_for`].
#[track_caller]
pub(crate) fn declared_mask_temp(temp: Option<KReg>) -> KReg {
    temp.expect(
        "this encoding needs a mask register that `RegisterFile::mask_temps_for` did not ask for",
    )
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
        shape: LatticeShape,
    ) -> Result<CompileResult, CompileError> {
        let lanes = native_register_file(self.clone()).vector_bytes / BYTES_PER_LANE;
        let collapse = Collapse {
            domain: Domain {
                shape,
                origin: origin(),
            },
            lanes,
        };
        let (arena, root) = pixelflow_ir::passes::legalize(arena, root, &collapse)
            .map_err(CompileError::Legalize)?;
        let origin_ids = origin_slots(&arena);
        let schedule = arena_to_schedule(&arena, root, Some(origin_ids));
        compile_native(schedule, self)
    }
}

/// A lane is one `f32`.
const BYTES_PER_LANE: u32 = 4;

/// The two per-call scalars every collapse reads: where the lattice's sample
/// `(0, 0)` lies, `x0` then `y0`.
///
/// Declared as uniforms by `passes::lattice::collapse`, so the arena names
/// them the way it names any per-call scalar and the emitter loads them the
/// way it loads any uniform — once per call, broadcast. What is particular
/// to them is *where*: not in the link's block, whose layout is the
/// caller's, but in a block of their own, the context entry after the
/// link's (see [`KernelFn`](executable::KernelFn)). One identity per axis for
/// the whole process, minted once, so every arena declares the same two
/// instances and [`origin_slots`] can find them by identity afterwards.
pub fn origin() -> [UniformDecl; 2] {
    static ORIGIN: std::sync::OnceLock<[UniformDecl; 2]> = std::sync::OnceLock::new();
    *ORIGIN.get_or_init(|| {
        [0.0, 0.0].map(|default| UniformDecl {
            id: UniformIdentity::mint(),
            default,
        })
    })
}

/// The uniform slots [`origin`]'s two instances hold in a legalized arena —
/// the two `passes::lattice::collapse` declared, which is why they are always
/// present.
fn origin_slots(arena: &pixelflow_ir::arena::ExprArena) -> [UniformId; 2] {
    origin().map(|decl| {
        let slot = arena
            .uniforms()
            .iter()
            .position(|d| d.id == decl.id)
            .unwrap_or_else(|| panic!("a legalized arena declares the origin; this one does not"));
        UniformId(slot as u64)
    })
}

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
    /// Values one scope computes for the scopes inside it and parks in a
    /// slot of their own — the loop-invariant code motion, counted.
    pub hoisted_values: u32,
    /// What was emitted, per scope of the nest — the static half of a cost
    /// model's inputs. Counted, never optimized: see [`traffic`](self::traffic).
    pub traffic: EmitTraffic,
}

/// The architecture seam for the shared driver.
///
/// [`compile_via_backend`] owns the architecture-INDEPENDENT logic — schedule,
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
    /// Backends declare it as a `const` next to their encodings and clamp its
    /// scratch pool to [`EmitCtx::max_regs`] at construction. It is the only
    /// target-dependent input to any of the shared logic here.
    fn register_file(&self) -> regalloc::RegisterFile;

    /// Per-compile setup before any code is emitted (e.g. seed a constant pool).
    fn begin(&mut self, schedule: &[regalloc::Def]) -> Result<(), CompileError>;

    /// Called once the frame layout is known, BEFORE any body instruction is
    /// emitted.
    fn frame_ready(&mut self, _frame_size: u32) {}

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
    fn emit_resolve(
        &mut self,
        code: &mut Vec<u8>,
        vid: regalloc::ValueId,
        target: Reg,
        locs: &[Option<Binding>],
    ) -> Reg;

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
    /// names a [`Label`] — the constant pool's not-yet-known position — rather
    /// than a `code.len()` read off and carried by hand.
    fn anchor(&mut self, asm: &mut Assembly);

    /// Append whatever must trail the emitted function — the constant pool
    /// and the label that names it.
    fn finish(&mut self, asm: &mut Assembly);

    /// Save / restore a value in a slot outside any scope's own spill frame:
    /// a fold's binder or accumulator, a root parked for the scopes inside.
    fn slot_store(&mut self, code: &mut Vec<u8>, src: Reg, offset: u32);
    fn slot_load(&mut self, code: &mut Vec<u8>, dst: Reg, offset: u32);

    /// Bracket one scope's emission, for a decorator that attributes what is
    /// emitted to the scope it runs in. Defaults do nothing.
    fn scope_begin(&mut self) {}
    fn scope_end(&mut self, _scope: regalloc::Scope, _bytes: u32) {}

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
    fn add_scalar(&mut self, code: &mut Vec<u8>, dst: Reg, scratch: Reg, scalar: f32);

    /// Load an `f32` constant, broadcast across every lane.
    fn load_const(&mut self, code: &mut Vec<u8>, dst: Reg, val: f32);

    /// `dst = op(srcs[0], srcs[1])`, an ordinary vector ALU op outside the
    /// schedule: the fold loop's accumulate (`op` is the fold's monoid).
    fn alu(&mut self, code: &mut Vec<u8>, op: OpKind, dst: Reg, srcs: [Reg; 2]);

    /// `dst = (srcs[0] >= srcs[1]) ? all-ones : 0` — the fold loop's trip
    /// test.
    ///
    /// A default over [`IsaBackend::alu`]: every backend but AVX-512 computes
    /// a comparison exactly like any other binary op. AVX-512 represents a
    /// comparison's result as a k-register before it is widened to an
    /// ordinary vector mask ([`RegisterFile::mask_guard_temps`]), which
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
pub struct WritePlan {
    /// The value to store, in a register.
    pub value: Reg,
    /// The row binder — a broadcast index — where its fold keeps it.
    pub row: Binding,
    /// The column binder, likewise.
    pub col: Binding,
    /// How many of `value`'s lanes to store, from lane 0: the lane fold's
    /// trip count, the full batch or a row's remainder.
    pub lanes: u32,
    /// This instruction's reservations: two GPRs for the address, and the
    /// vector or mask temp a backend's remainder store asked for.
    pub scratch: regalloc::Scratch,
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

/// Allocate a straight-line schedule and emit it as one scope's body.
///
/// Production compiles allocate the whole nest at once
/// ([`regalloc::RegisterAllocator::allocate_nest`]) so every scope's frame
/// is known before any of them is emitted; this is the one-scope
/// convenience the emitter's own tests are written against.
#[cfg(test)]
fn emit_dag_body<B: IsaBackend>(
    schedule: Vec<regalloc::Def>,
    backend: &mut B,
) -> Result<(Vec<u8>, Reg, u32, u32), CompileError> {
    use regalloc::RegisterAllocator;
    let nest = regalloc::LinearScan.allocate(schedule, &backend.register_file());
    let (code, result, frame, spills) = emit_scope(
        nest.body(),
        backend,
        &alloc::collections::BTreeMap::new(),
        FramePlan {
            override_size: None,
            fold_slots: &alloc::collections::BTreeMap::new(),
            binder_slots: &alloc::collections::BTreeMap::new(),
            guard_slots: &alloc::collections::BTreeMap::new(),
            slot_base: 0,
        },
    )?;
    Ok((
        code,
        result.expect("a value-rooted schedule has a result"),
        frame,
        spills,
    ))
}

/// Where one scope's memory is, as its driver decided it — the answers
/// [`emit_scope`] cannot work out for itself because they are all facts
/// about the *nest*, not about the scope.
#[derive(Clone, Copy)]
struct FramePlan<'a> {
    /// Frame size to latch instead of this scope's own. The driver hands
    /// every scope the same `m` so they all address the shared park slots
    /// consistently. `None` for a scope that is the whole function.
    override_size: Option<u32>,
    /// Each surviving fold's accumulator slot, by its `Reduce`'s own
    /// `ValueId`: a slot outside any single scope's frame, because the scope
    /// that opens the loop and the loop itself both address it. Empty
    /// wherever nothing here can open a fold.
    fold_slots: &'a alloc::collections::BTreeMap<regalloc::ValueId, u32>,
    /// Each surviving fold's binder slot, by the same `Reduce` `ValueId`: the
    /// loop seeds and steps the binder there when the allocator did not carry
    /// it, and a scope inside reads it there through the binder's `Var`. Keyed
    /// by the fold rather than by that `Var` because sibling folds binding
    /// the same slot share one `Var` node — which loop's counter it names is
    /// a fact about the scope reading it, found by walking that scope's
    /// enclosing folds.
    binder_slots: &'a alloc::collections::BTreeMap<regalloc::ValueId, u32>,
    /// Each surviving `Guard`'s result slot, by its `Guard`'s own `ValueId`:
    /// the accumulator-slot analogue for a branch rather than a loop — a slot
    /// outside any single scope's frame, because the scope the `Guard` def
    /// sits in and both of its two arms all address it. Empty wherever
    /// nothing here can open a guard arm.
    guard_slots: &'a alloc::collections::BTreeMap<regalloc::ValueId, u32>,
    /// Where this scope's own spill slots start. Zero for the body, which
    /// has the frame to itself. A fold's body is the case that is not that:
    /// it runs nested inside its parent's schedule, with the parent's
    /// spilled values still live across it, so it is based at the parent's
    /// `layout.frame_size` and the two cannot alias.
    slot_base: u32,
}

/// Emit one scope from a finished allocation.
///
/// `parks` is where every root of the nest is parked, by its `ValueId`: the
/// slot the scope computing it writes after the def, and the scopes inside
/// read it from — unless the allocator carried it into them in a register,
/// which their placement says. Which roots this scope *reads* (an ancestor
/// computed them: its entries for them are placeholders that emit nothing)
/// and which it *computes* (its own `roots`) are the allocation's answers,
/// so one map serves every scope.
///
/// `fold_slots` is the same idea for a surviving `Reduce`'s accumulator,
/// addressed by its own `ValueId`, at a slot that outlives both this scope's
/// frame and the fold's own (see `ScheduledOp::Reduce`'s arm below, and
/// docs/plans/2026-09-10-a-surviving-reduce-is-a-loop.md's "the design
/// decision that makes this tractable"); `binder_slots` likewise for its
/// binder.
///
/// Returns the code, the register the scope's result is in — `None` when
/// the root is an effect and not a value: a `Write`, a `Seq`, a fold over
/// the unit monoid — the frame size latched, and how many values spilled.
fn emit_scope<B: IsaBackend>(
    allocation: regalloc::Allocation<'_>,
    backend: &mut B,
    parks: &alloc::collections::BTreeMap<regalloc::ValueId, u32>,
    frame: FramePlan<'_>,
) -> Result<(Vec<u8>, Option<Reg>, u32, u32), CompileError> {
    let FramePlan {
        override_size: frame_override,
        fold_slots,
        binder_slots,
        guard_slots,
        slot_base,
    } = frame;
    let file = backend.register_file();
    backend.scope_begin();
    // Allocation happened before this call — once per scope, over the whole
    // nest. The allocator chooses the evaluation order, so everything here —
    // guard ranges, program points, the emit loop itself — reads the schedule
    // it handed back rather than the one it was given.
    let schedule = allocation.schedule();
    let mut layout = FrameLayout::resolve(allocation, file.vector_bytes, slot_base)?;
    let real_spill_count = layout.slots;
    // This scope's top is exactly the base for anything nested inside it.
    let nested_slot_base = layout.frame_size;

    // The roots this scope reads from an enclosing scope's park, and the
    // ones it parks for the scopes inside. A value an enclosing scope parked
    // has no address in this frame — its slot is the park, which outlives
    // every scope's frame. Only the address is pinned: whether the value is
    // in that slot or in a register, at each point, is the placement's
    // answer.
    let preloaded: alloc::collections::BTreeMap<regalloc::ValueId, u32> = schedule
        .iter()
        .map(|def| def.value)
        .filter(|v| allocation.parked_by_an_enclosing_scope(*v))
        .map(|v| {
            let slot = *parks
                .get(&v)
                .unwrap_or_else(|| panic!("{v:?} is parked by an enclosing scope but has no slot"));
            (v, slot)
        })
        .collect();
    let parked: alloc::collections::BTreeMap<regalloc::ValueId, u32> = allocation
        .roots()
        .iter()
        .map(|v| {
            let slot = *parks
                .get(v)
                .unwrap_or_else(|| panic!("{v:?} is a root of this scope but has no slot"));
            (*v, slot)
        })
        .collect();
    for (vid, &offset) in &preloaded {
        layout.pin_slot(*vid, Slot::new(offset, file.vector_bytes));
    }
    // A surviving fold's roots, the same idea in the other direction. Its
    // `Reduce` def is a genuine computation *in* this scope (scanned, never a
    // placeholder), but its slot is the driver's dedicated fold slot, not
    // whatever offset this scope's own `FrameLayout::resolve` gave it
    // (`ScheduledOp::Reduce`'s scan-time `Where::Spilled` earns it one
    // regardless, thrown away here). Harmless to pin one this scope never
    // reaches — `pin_slot` on a `ValueId` nothing here reads is simply never
    // read back.
    // A surviving `Guard`'s result, the same idea again: two arms and the
    // scope its def sits in must all agree on one address, so the driver's
    // `guard_slots` overrides whatever this scope's own `FrameLayout::resolve`
    // gave it too.
    let mut fold_pins: alloc::vec::Vec<(regalloc::ValueId, u32)> = fold_slots
        .iter()
        .chain(guard_slots.iter())
        .map(|(vid, &offset)| (*vid, offset))
        .collect();
    // The binders of this scope's own fold and every enclosing fold, each
    // where that loop keeps it — innermost first, so a binder shadowing an
    // enclosing one is the nearer loop's. A `Write` reads its row and column
    // from here, and a binder's `Var` (found here by the binder's number —
    // sibling folds binding the same slot share one `Var` node, so which
    // counter it names is this scope's question) is a placeholder whose def
    // emits nothing: the loop seeded it. Only one the allocator did not
    // carry has a slot to pin; pinning a carried one would earn the register
    // a store nothing reads.
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
                let offset = *binder_slots.get(&def.value).unwrap_or_else(|| {
                    panic!(
                        "{:?}'s fold has no binder slot — the driver did not assign one",
                        def.value
                    )
                });
                Binding::from(Slot::new(offset, file.vector_bytes))
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
            if let Binding::Loc(Loc::Slot(slot)) = at_binder {
                fold_pins.push((bv, slot.offset()));
            }
        }
        opened = parent;
    }
    for &(vid, offset) in &fold_pins {
        layout.pin_slot(vid, Slot::new(offset, file.vector_bytes));
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

    let frame_size = frame_override.unwrap_or(layout.frame_size);
    if frame_size < layout.frame_size {
        return Err(CompileError::Internal(
            "frame override smaller than the layout's frame",
        ));
    }
    backend.frame_ready(frame_size);

    // If short-circuit guards, read off the allocation rather than
    // recomputed: `schedule` above is `allocation.schedule()` verbatim, and
    // the allocator already ran this same analysis against it to place split
    // ranges around each arm (see `regalloc::Allocation::if_guards`). A
    // root this scope parks is never inside an arm — the analysis was told
    // it is read outside the schedule — so a guard can never skip a park.
    let if_guards: &[guards::IfGuard] = allocation.if_guards();
    let sched_len = schedule.len();

    struct PendingBranch {
        guard_idx: usize,
        arm: IfArm,
    }
    let mut branch_starts: alloc::vec::Vec<alloc::vec::Vec<PendingBranch>> =
        (0..sched_len).map(|_| alloc::vec::Vec::new()).collect();
    let mut branch_ends: alloc::vec::Vec<alloc::vec::Vec<PendingBranch>> =
        (0..sched_len).map(|_| alloc::vec::Vec::new()).collect();
    for (gi, guard) in if_guards.iter().enumerate() {
        for arm in IfArm::ALL {
            let range = guard.range(arm);
            if range.0 != range.1 {
                branch_starts[range.0].push(PendingBranch { guard_idx: gi, arm });
                if range.1 < sched_len {
                    // The arm too, not just the guard: an end used to name the
                    // guard alone and recover the arm by trying both, which
                    // meant a guard whose arms end together was visited twice.
                    branch_ends[range.1].push(PendingBranch { guard_idx: gi, arm });
                }
            }
        }
    }

    // What to call the point past one arm of one guard. The `If`'s own
    // `ValueId` rather than its index in `if_guards`, because the node is
    // the identity and the index is a position in a scratch vector — and
    // because two guards can share a mask, so the mask would alias.
    let arm_join = |guard: &guards::IfGuard, arm: IfArm| {
        let if_value = schedule[guard.if_idx].value;
        let side = match arm {
            IfArm::True => "true",
            IfArm::False => "false",
        };
        Label::new(&alloc::format!("v{}_past_{side}", if_value.0))
    };

    // One dense ValueId -> Binding lookup for the hot loop, carried *forward*: a
    // placement is a schedule, so the answer changes at program points, and
    // this is that schedule played out. Each range of each value's life
    // becomes one write here at the point it starts — O(total ranges), not a
    // lookup per operand per instruction.
    let mut locs: alloc::vec::Vec<Option<Binding>> = layout.bindings().to_vec();
    // A surviving fold's root in its slot — an accumulator, or a binder the
    // allocator did not carry — is read back from its dedicated slot rather
    // than wherever this scope's own `FrameLayout::resolve` happened to put
    // it. `resolve` gave it a real address (its scan-time `Where::Spilled`
    // earns one like any other spilled value), but a throwaway one — a
    // `Reduce` def's own emission never goes through the ordinary
    // operand/destination machinery this table serves everyone else, and a
    // binder's placeholder emits nothing, so nothing but this override ever
    // reads or writes it. Only for a value this scope has: a pin this scope
    // never reaches has no entry here to override.
    for &(vid, offset) in &fold_pins {
        if let Some(entry) = locs.get_mut(vid.0 as usize)
            && entry.is_some()
        {
            *entry = Some(Binding::from(Slot::new(offset, file.vector_bytes)));
        }
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
        if preloaded.contains_key(&v) {
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

    // No prologue here — the caller frames the body (see the fn doc).
    let mut asm = Assembly::default();

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
        if !preloaded.contains_key(&vid) {
            continue;
        }
        let placement = allocation.placement(vid);
        let at_head = allocation.at_head(vid);
        let head = layout.binding(vid, at_head);
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
                    locs[vid.0 as usize] = Some(layout.binding(vid, from_memory));
                    let got = backend.emit_resolve(&mut asm.code, vid, r, &locs);
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
                    locs[vid.0 as usize] = Some(layout.binding(vid, from_memory));
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
        let Some(&offset) = parked.get(&vid) else {
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

    // The unit-typed roots — an effect, not a value.
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
            asm.bind(arm_join(&if_guards[pb.guard_idx], pb.arm));
        }

        // Ranges that begin here. A register range starting away from the
        // value's definition is a reload the allocator chose to keep: the
        // value comes back into a pool register and stays there, instead of
        // being fetched into a scratch at every read.
        for (v, to) in core::mem::take(&mut moves[sched_idx]) {
            match to {
                Binding::Loc(Loc::Reg(r)) => {
                    let src = backend.emit_resolve(&mut asm.code, v, r, &locs);
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

        // Guard branches that begin before this instruction.
        for pb in &branch_starts[sched_idx] {
            let (guard_idx, arm) = (pb.guard_idx, pb.arm);
            let guard = &if_guards[guard_idx];
            let mask_reg = match location_of(&locs, guard.mask_vid) {
                Binding::Loc(Loc::Reg(r)) => r,
                _ => backend.emit_resolve(&mut asm.code, guard.mask_vid, guard_mask(), &locs),
            };
            let past_arm = arm_join(guard, arm);
            let test = MaskTest {
                reg: mask_reg,
                scratch: guard_temp,
                mask_scratch: mask_guard_temp,
                arm,
            };
            backend.branch_if_arm_is_dead(&mut asm, test, past_arm);
        }

        // A parked value's placeholder def emits nothing — the enclosing
        // scope already parked the value in its slot; consumers reload from
        // there.
        if preloaded.contains_key(vid) {
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
        // when the lane fold was inlined (`arena_to_schedule`).
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
                    backend.emit_resolve(&mut asm.code, *value, target, &locs)
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
            // (`extract_folds`'s placeholder): that loop ran before this
            // scope began, `locs` already names the slot, and there is
            // nothing to emit.
            let Some(fold_scope) = allocation.fold_opening_at(sched_idx) else {
                continue;
            };
            let fold_alloc = allocation.sibling(fold_scope);
            let acc_slot = *fold_slots.get(vid).unwrap_or_else(|| {
                panic!(
                    "{vid:?}'s Reduce def has no accumulator slot — the driver did not assign one"
                )
            });
            let binder_slot = *binder_slots.get(vid).unwrap_or_else(|| {
                panic!("{vid:?}'s Reduce def has no binder slot — the driver did not assign one")
            });
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

            // Seed: the accumulator starts at the monoid's identity and the
            // binder at `lo`, each where it lives — written through a temp
            // when that is a slot.
            let mut seed = |backend: &mut B, at: Option<Reg>, value: f32, slot: u32| match at {
                Some(r) => backend.load_const(&mut asm.code, r, value),
                None => {
                    backend.load_const(&mut asm.code, t0, value);
                    backend.slot_store(&mut asm.code, t0, slot);
                }
            };
            if accumulates {
                seed(backend, acc_reg, fold.monoid().identity(), acc_slot);
            }
            seed(backend, binder_reg, fold.range().start as f32, binder_slot);

            let top = Label::new(&alloc::format!("reduce{}_top", vid.0));
            let exit = Label::new(&alloc::format!("reduce{}_exit", vid.0));
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
            backend.load_const(&mut asm.code, t1, fold.range().end as f32);
            backend.test_ge(&mut asm.code, t0, [binder_now, t1], scratch.mask_guard_temp);
            backend.branch_if_arm_is_dead(
                &mut asm,
                MaskTest {
                    reg: t0,
                    scratch: scratch.guard_temp,
                    mask_scratch: scratch.mask_guard_temp,
                    arm: IfArm::False,
                },
                exit,
            );

            // The body, in its own scope.
            let (fold_code, body_result, _, _) = emit_scope(
                fold_alloc,
                backend,
                parks,
                FramePlan {
                    override_size: Some(frame_size),
                    fold_slots,
                    binder_slots,
                    guard_slots,
                    slot_base: nested_slot_base,
                },
            )?;
            asm.code.extend_from_slice(&fold_code);

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
                Some(b) => backend.add_scalar(&mut asm.code, b, t0, stride),
                None => {
                    backend.slot_load(&mut asm.code, t0, binder_slot);
                    backend.add_scalar(&mut asm.code, t0, t1, stride);
                    backend.slot_store(&mut asm.code, t0, binder_slot);
                }
            }
            backend.jump(&mut asm, top);
            asm.bind(exit);

            // The result is read from the accumulator's slot — where this
            // scope's placement of the def says it is — so a carried
            // accumulator lands there once, on the way out. A scope inside
            // reads it there too: a `Reduce` is never a root (`stays_put`),
            // so nothing hands it over or carries it.
            if let Some(a) = acc_reg
                && accumulates
            {
                backend.slot_store(&mut asm.code, a, acc_slot);
            }
            continue;
        }

        // A surviving `Guard`: a mask test, a branch, the taken arm's own
        // scope, a join — exactly §3's plan
        // (docs/plans/2026-09-12-emit-should-just-emit.md), and built from
        // the same primitives as the `Reduce` loop just above (recurse into
        // `emit_scope` for a nested scope's code) and the guarded-`If`
        // block below (`branch_if_arm_is_dead`, `Label`, a join). The
        // difference from both: only one arm ever runs (a branch, not a
        // loop), and *neither* arm is this schedule's own code (both are
        // wholly separate scopes, unlike a blend's two operands sitting
        // right here as values). `guard_opening_at` finds each arm's scope
        // by the position of this def, the way `fold_opening_at` finds a
        // fold's; `None` for the True arm means an enclosing scope's
        // `Guard` was carved to a placeholder here (its value read from the
        // pinned slot below, `fold_pins`), exactly as an enclosing scope's
        // `Reduce` is above.
        if let ScheduledOp::Guard(mask_vid, ..) = sched_op {
            let Some(true_scope) = allocation.guard_opening_at(sched_idx, IfArm::True) else {
                continue;
            };
            let false_scope = allocation
                .guard_opening_at(sched_idx, IfArm::False)
                .expect("a Guard's True arm opens here without its False arm");
            let guard_slot = *guard_slots.get(vid).unwrap_or_else(|| {
                panic!("{vid:?}'s Guard def has no result slot — the driver did not assign one")
            });

            let mask_reg = match location_of(&locs, *mask_vid) {
                Binding::Loc(Loc::Reg(r)) => r,
                _ => backend.emit_resolve(&mut asm.code, *mask_vid, guard_mask(), &locs),
            };
            let part = |part: &str| Label::new(&alloc::format!("v{}_{part}", vid.0));
            let (arm_false, join) = (part("arm_false"), part("arm_join"));
            let test = |arm| MaskTest {
                reg: mask_reg,
                scratch: guard_temp,
                mask_scratch: mask_guard_temp,
                arm,
            };
            // Jump to the False arm when every lane agrees the mask is
            // false — the True arm's own test, exactly as a guarded
            // `If`'s "only_false" branch is reached (mask-uniformly-
            // true takes the *other* branch there because both arms sit in
            // the same flat schedule and one is skipped forward over; here
            // there is no flat schedule to skip through, only two separate
            // scopes to choose between, so a single branch on "is the True
            // arm dead" suffices).
            backend.branch_if_arm_is_dead(&mut asm, test(IfArm::True), arm_false);

            let (true_code, true_result, _, _) = emit_scope(
                allocation.sibling(true_scope),
                backend,
                parks,
                FramePlan {
                    override_size: Some(frame_size),
                    fold_slots,
                    binder_slots,
                    guard_slots,
                    slot_base: nested_slot_base,
                },
            )?;
            asm.code.extend_from_slice(&true_code);
            backend.slot_store(
                &mut asm.code,
                true_result.expect("a guard arm computes a value"),
                guard_slot,
            );
            backend.jump(&mut asm, join);

            asm.bind(arm_false);
            let (false_code, false_result, _, _) = emit_scope(
                allocation.sibling(false_scope),
                backend,
                parks,
                FramePlan {
                    override_size: Some(frame_size),
                    fold_slots,
                    binder_slots,
                    guard_slots,
                    slot_base: nested_slot_base,
                },
            )?;
            asm.code.extend_from_slice(&false_code);
            backend.slot_store(
                &mut asm.code,
                false_result.expect("a guard arm computes a value"),
                guard_slot,
            );

            // Every reader finds the result in `guard_slot`: a `Guard` is
            // never a root (`stays_put`), so nothing hands it over.
            asm.bind(join);
            continue;
        }

        let dst_loc = location_of(&locs, *vid);
        let plan = resolve_operands(sched_op, dst_loc, &locs, scratch)?;

        // If with a guard region: emit a uniform-mask short-circuit wrapper.
        if let ScheduledOp::Ternary(OpKind::If, mask_vid, true_vid, false_vid) = sched_op
            && let Some(guard) = if_guards.iter().find(|g| g.if_idx == sched_idx)
            && guard.has_guarded_arm()
        {
            let mask_reg = match location_of(&locs, *mask_vid) {
                Binding::Loc(Loc::Reg(r)) => r,
                _ => backend.emit_resolve(&mut asm.code, *mask_vid, guard_mask(), &locs),
            };
            let dst = dst_loc.reg();
            let in_reg = |v: regalloc::ValueId| match location_of(&locs, v) {
                Binding::Loc(Loc::Reg(r)) => Some(r),
                _ => None,
            };
            let true_reg = in_reg(*true_vid);
            let false_reg = in_reg(*false_vid);

            // Named after the `If` they belong to, so two of these in one
            // schedule cannot collide however they interleave.
            let part = |part: &str| Label::new(&alloc::format!("v{}_{part}", vid.0));
            let (only_false, only_true, join) =
                (part("only_false"), part("only_true"), part("join"));

            // Both guards read `mask_reg`, which is why the reduction
            // scratch is a reservation of its own rather than whichever
            // register the mask was resolved into.
            let test = |arm| MaskTest {
                reg: mask_reg,
                scratch: guard_temp,
                mask_scratch: mask_guard_temp,
                arm,
            };
            backend.branch_if_arm_is_dead(&mut asm, test(IfArm::True), only_false);
            backend.branch_if_arm_is_dead(&mut asm, test(IfArm::False), only_true);

            // Mixed lanes: the blend, the path a lane-varying mask takes.
            backend.emit_plan(&mut asm.code, &plan)?;
            backend.jump(&mut asm, join);

            asm.bind(only_false);
            if let Some(freg) = false_reg {
                backend.emit_mov(&mut asm.code, dst, freg);
            } else {
                backend.emit_resolve(&mut asm.code, *false_vid, dst, &locs);
            }
            backend.jump(&mut asm, join);

            asm.bind(only_true);
            if let Some(treg) = true_reg {
                backend.emit_mov(&mut asm.code, dst, treg);
            } else {
                backend.emit_resolve(&mut asm.code, *true_vid, dst, &locs);
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
        if parked.contains_key(vid) {
            hand_off(backend, &mut asm.code, *vid, written)?;
        }
    }

    // No "did every branch get its landing point" assertion here any more:
    // `Assembly::finish` panics on a name nobody wrote, which is the same
    // check, stated once, for every branch rather than only these.

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
                backend.emit_resolve(&mut asm.code, root, target, &locs)
            }
        })
    };

    let code = asm.finish();
    backend.scope_end(allocation.scope(), code.len() as u32);
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
    /// Bound-memory gather: read the buffer whose base is the second operand
    /// at the lane index computed by the first. Lowered from
    /// `RawGather(Buffer(slot), index)`; the `Buffer` leaf *is* the base — a
    /// [`ScheduledOp::Context`] def, a [`regalloc::Class::Pointer`] value
    /// the allocator places like any other — so the index is the one vector
    /// operand and the base the one pointer operand.
    Gather(regalloc::ValueId, regalloc::ValueId),
    /// A `Gather` whose index is the same in every lane: one scalar load,
    /// broadcast. The same `RawGather(Buffer(slot), index)`, split from
    /// [`ScheduledOp::Gather`] by [`arena_to_schedule`] on the index's
    /// variance — it lacks the lane binder's bit, so lane 0 *is* the index
    /// and the other lanes are copies of it. A glyph's per-piece table
    /// reads are addressed by its fold's own binder and nothing else, which
    /// makes them this and not a gather; the split is what turns a per-lane
    /// address sequence (`vpextrd`/`vinsertps` ×4, `vgatherdps`, four
    /// `umov`/`ldr`/`ins`) into `cvttss2si` + `vbroadcastss [base + idx*4]`.
    /// Index first, base second, as `Gather`.
    Broadcast(regalloc::ValueId, regalloc::ValueId),
    /// Per-call scalar, broadcast from a block: the value at `4 * offset`
    /// from the block whose base is the pointer operand — the link's
    /// uniform block, or the origin's. Not a leaf to the placement, since
    /// the load is an instruction worth doing once per call rather than
    /// once per batch. The offset is a [`UniformId`]'s slot, at its width.
    Uniform(regalloc::ValueId, u64),
    /// The `k`-th pointer of the context the kernel is called with: a
    /// buffer's base for `k` below the buffer count, the link's uniform
    /// block and the origin block after. The definition of every
    /// [`regalloc::Class::Pointer`] value; no operands, variance `CONST`,
    /// so it is placed in the per-call scope and carried into the loops
    /// inside by `plan_carries` on the strength of its reads there — one
    /// load per call where every gather used to reload it
    /// (docs/plans/2026-09-22-a-pointer-is-a-value.md).
    Context(u16),
    /// The lane fold's binder: the constant `[0, 1, …, L−1]`. The fold
    /// whose binder this is executes by lanes (its body is inlined into its
    /// parent's schedule — see [`arena_to_schedule`]), so the binder is a
    /// leaf here rather than a loop counter. Carries the binder so its
    /// variance bit is the fold's, which is what "lane-uniform" is read off.
    Lanes(Binder),
    /// The store the lattice's folds wrap a kernel in: `value`'s first
    /// `lanes` lanes at `out + 4·(row·pitch + col)`, `row` and `col` being
    /// the enclosing folds' binders. This def *is* the lane fold, executed
    /// by lanes: `lane` is that fold's binder, which it closes over the way
    /// any `Reduce` closes over its own, and `lanes` its trip count — the
    /// full batch, or a row's remainder — so two lane folds sharing one
    /// arena `Write` are two defs of different widths reading one value.
    Write {
        row: Binder,
        col: Binder,
        lane: Binder,
        lanes: u32,
        value: regalloc::ValueId,
    },
    /// Two effects, the first then the second: the unit monoid's own
    /// combine, which is what a `SEQ` fold over a row's main batches and its
    /// remainder is. Reads no register — the schedule's order *is* the
    /// sequencing — and defines no value.
    Seq(regalloc::ValueId, regalloc::ValueId),
    /// A surviving bounded fold: `⊕` over `fold`'s visited indices, whose
    /// body is the value named by the second field — in *this schedule's*
    /// numbering (`arena_to_schedule` maps it like any other child), before
    /// [`extract_folds`] carves the body out into its own
    /// [`regalloc::ScopeFold`]. Kept only so [`schedule_variance`] can look
    /// the body's variance up (`Reduce`'s own result is the body's variance
    /// with the binder's own bit removed) and so [`extract_folds`] can find
    /// the body's closure; the emitter never resolves it as an operand —
    /// the loop's result comes from [`regalloc::Allocation::opens_at`]
    /// naming the [`regalloc::Scope::Fold`] this def opens, not from this
    /// `ValueId`.
    ///
    /// A [`RangeFold`], by type: a loop is what this becomes, and only a
    /// range is one. An interval cannot reach a schedule —
    /// `arena_to_schedule` refuses it — so nothing downstream asks.
    Reduce(RangeFold, regalloc::ValueId),
    /// A surviving `Guard`: the mask, and its two arms' names. `mask` is a
    /// real value in *this* schedule (`arena_to_schedule` maps it like any
    /// other child); the two `KernelKey`s are not — they name kernels whose
    /// bodies live in wholly separate arenas, resolved through
    /// `KernelStore::resolve` by `extract_guards`, which schedules each arm
    /// as its own [`regalloc::Scope::GuardArm`], exactly as [`extract_folds`]
    /// carves a [`ScheduledOp::Reduce`]'s body into its own
    /// [`regalloc::Scope::Fold`] — except an arm is not carved *out of*
    /// anything here, since nothing of it was ever in this schedule to carve.
    /// The emitter never resolves this def's operands the ordinary way: its
    /// own `ValueId` is forced to a slot (`regalloc`'s `Scan`, mirroring a
    /// `Reduce`'s accumulator), and the two arms' scopes — found by
    /// `regalloc::Allocation::guard_opening_at` — are each emitted as a
    /// nested scope bracketed by a branch instead of a loop
    /// (docs/plans/2026-09-12-emit-should-just-emit.md §3).
    Guard(
        regalloc::ValueId,
        pixelflow_ir::key::KernelKey,
        pixelflow_ir::key::KernelKey,
    ),
}

impl ScheduledOp {
    /// Which register file the value this op defines lives in: a
    /// [`ScheduledOp::Context`] is an address, everything else is a vector
    /// (an effect's "value" included, which is never placed anywhere).
    #[must_use]
    pub fn class(&self) -> regalloc::Class {
        match self {
            ScheduledOp::Context(_) => regalloc::Class::Pointer,
            _ => regalloc::Class::Vector,
        }
    }
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
/// The arena is a legalized one: wrapped in the lattice's folds, with no
/// coordinate `Var` left. Two of its shapes are not copied one to one:
///
/// - **The lane fold is inlined.** A `Reduce` whose body is a `Write` naming
///   the fold's binder as its lane is the fold executed by lanes; it has no
///   loop, so its body is not carved into a scope of its own — its `Write`
///   becomes the fold's own def, in its parent's schedule, with the fold's
///   trip count folded in as the store width, and the arena's `Write` node
///   itself gets no `ValueId` (nothing but a lane fold names one). Two lane
///   folds sharing one `Write` — a row's main batches and its remainder —
///   are two `Write` defs of different widths reading one value.
/// - **The lane binder is a constant.** Its `Var` becomes
///   [`ScheduledOp::Lanes`], the iota every lane-varying value is built on.
///
/// `origin` is the uniform slots of the two [`origin`] scalars, which read
/// from the context entry after the link's block rather than from it — or
/// `None` for an arena that declares no origin at all, because
/// `passes::lattice::collapse` never wrapped it (a guard's arm, see
/// [`schedule_guard_arm`]). Then every uniform is the link's.
///
/// # Panics
///
/// Panics if a `Param` or `Nary` node is encountered (these are not expected
/// in JIT compilation), or a coordinate `Var` survived `collapse`.
fn arena_to_schedule(
    arena: &pixelflow_ir::arena::ExprArena,
    root: pixelflow_ir::arena::ExprId,
    origin: Option<[UniformId; 2]>,
) -> Vec<regalloc::Def> {
    arena_to_schedule_from(arena, root, origin, 0)
}

/// [`arena_to_schedule`], numbering `ValueId`s from `starting_id` rather than
/// `0`.
///
/// Every production call site schedules one whole nest's worth of
/// `ValueId`s at once, all sharing one numbering — a fold's schedule is
/// *carved out of* its parent's by [`extract_folds`], keeping the parent's
/// ids, so nothing needs a second range. A `Guard`'s arm is the one
/// exception: its schedule is built fresh, from a wholly separate arena
/// (`schedule_guard_arm`), so its own `0..N` would collide with whatever
/// `ValueId`s the enclosing nest already uses — and a collision here is not
/// merely a wrong number, it is `regalloc::Allocation::parked_by_an_enclosing_scope`
/// answering `true` for an arm value that happens to share a number with
/// some unrelated ancestor's root, reading that root's park slot instead of
/// computing its own value. `schedule_guard_arm` is the one caller that needs
/// this, offsetting each arm past every id already in use in the nest so far.
fn arena_to_schedule_from(
    arena: &pixelflow_ir::arena::ExprArena,
    root: pixelflow_ir::arena::ExprId,
    origin: Option<[UniformId; 2]>,
    starting_id: u32,
) -> Vec<regalloc::Def> {
    use pixelflow_ir::arena::{ExprId, ExprNode};
    use regalloc::ValueId;

    let len = arena.len();
    let mut reachable = alloc::vec![false; len];
    mark_reachable(arena, root, &mut reachable);

    // The lane binder: the one every reachable `Write` names as its lane.
    // One slot, by construction (`passes::lattice::pack` builds both lane
    // folds on `collapse`'s), asserted rather than assumed.
    let mut lane: Option<Binder> = None;
    for (idx, _) in reachable.iter().enumerate().filter(|(_, r)| **r) {
        if let ExprNode::Write { lane: l, .. } = arena.node(ExprId(idx as u32)) {
            match lane {
                None => lane = Some(l),
                Some(seen) => assert_eq!(
                    seen, l,
                    "two Writes name different lane binders; a lattice has one lane fold"
                ),
            }
        }
    }

    // Which reads are the same in every lane. A `RawGather` whose index
    // lacks the lane binder's bit is one load broadcast, not a gather; the
    // bit is read here, where the two are split, because the schedule's
    // own variance (`schedule_variance`) is computed after it is built. No
    // lane fold — a schedule built from an arena `collapse` never wrapped,
    // the emit tests' raw arenas — means nothing is known to be
    // lane-uniform, and every read stays a gather.
    let variance = lane.map(|_| pixelflow_ir::variance::compute_arena_variance(arena));
    let lane_uniform = |idx: ExprId| -> bool {
        match (lane, &variance) {
            (Some(lane), Some(variance)) => variance[idx.0 as usize].is_invariant_in(lane.var()),
            _ => false,
        }
    };

    // ExprId to ValueId mapping. u32::MAX = unmapped (unreachable, or a
    // `Write` node, whose defs are its lane folds').
    let mut id_map = alloc::vec![ValueId(u32::MAX); len];
    let mut schedule = Vec::new();
    let mut next_id = starting_id;

    let buffers = u16::try_from(arena.buffers().len())
        .expect("buffer table index fits the context slot immediate");
    // The uniform blocks' base pointers, one `Context` def each, made on
    // first use: a block has no arena node to map, unlike a buffer, whose
    // `Buffer` leaf is its `Context` def.
    let mut blocks: alloc::collections::BTreeMap<u16, ValueId> =
        alloc::collections::BTreeMap::new();

    for idx in 0..len {
        if !reachable[idx] {
            continue;
        }
        let expr_id = ExprId(idx as u32);
        let node = arena.node(expr_id);
        if let ExprNode::Write { .. } = node {
            continue;
        }

        let map_child = |child: ExprId| -> ValueId {
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
            ExprNode::Var(i) => match Binder::from_var(i) {
                None => panic!(
                    "arena_to_schedule: Var({i}) is a coordinate, which \
                     passes::lattice::collapse substitutes away -- this schedule was \
                     built without the lowering pipeline"
                ),
                Some(b) if lane == Some(b) => ScheduledOp::Lanes(b),
                Some(_) => ScheduledOp::Var(i),
            },
            ExprNode::Const(v) => ScheduledOp::Const(v),
            ExprNode::Param(i) => panic!(
                "ExprNode::Param({}) reached the JIT emitter -- \
                 call substitute_params before compile()",
                i
            ),
            // A buffer's base pointer: the `k`-th context entry, a pointer
            // value the gathers reading the buffer take as an operand.
            ExprNode::Buffer(id) => ScheduledOp::Context(id.0),
            // The link's block sits in the context entry after the buffer
            // slots and the origin's in the one after that; a value's offset
            // is its slot index within its block — the link step
            // (`jit_cache`) renumbers the table into dense first-occurrence
            // order before anything reaches here, and the two origin slots
            // are the last two, declared by `collapse` after the relink.
            // The block's base is a `Context` def made here on first use,
            // ahead of this def so the schedule stays topological.
            ExprNode::Uniform(u) => {
                let axis = origin.and_then(|slots| slots.iter().position(|&o| o == u));
                let (ctx_slot, offset) = match axis {
                    Some(axis) => (buffers + 1, axis as u64),
                    None => (buffers, u.0),
                };
                let block = *blocks.entry(ctx_slot).or_insert_with(|| {
                    let base = ValueId(next_id);
                    next_id += 1;
                    schedule.push(regalloc::Def {
                        value: base,
                        op: ScheduledOp::Context(ctx_slot),
                    });
                    base
                });
                ScheduledOp::Uniform(block, offset)
            }
            ExprNode::Unary(op, child) => ScheduledOp::Unary(op, map_child(child)),
            // Shl/Shr fold their Const shift-count operand into an immediate, so
            // the count never becomes a scheduled value (matching the imm-only
            // hardware shift encoders). The count const may still appear as its
            // own schedule entry (harmless/unused) if shared.
            ExprNode::Binary(op @ (OpKind::Shl | OpKind::Shr), a, b) => {
                let amount = match arena.node(b) {
                    ExprNode::Const(v) => shift_immediate(op, v),
                    _ => panic!(
                        "{:?} shift count must be a Const (lowering guarantees this)",
                        op
                    ),
                };
                ScheduledOp::ShiftImm(op, map_child(a), amount)
            }
            // RawGather's buffer leaf is its base pointer's def, mapped like
            // any other child. An index the lane binder does not reach is one
            // address for the whole batch, and the read is a broadcast load
            // rather than a gather (see `ScheduledOp::Broadcast`).
            ExprNode::Binary(OpKind::RawGather, buf, idx) => {
                assert!(
                    matches!(arena.node(buf), ExprNode::Buffer(_)),
                    "RawGather's first child must be a Buffer leaf, got {:?}",
                    arena.node(buf)
                );
                if lane_uniform(idx) {
                    ScheduledOp::Broadcast(map_child(idx), map_child(buf))
                } else {
                    ScheduledOp::Gather(map_child(idx), map_child(buf))
                }
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
            ExprNode::Binary(OpKind::Seq, a, b) => ScheduledOp::Seq(map_child(a), map_child(b)),
            ExprNode::Binary(op, a, b) => ScheduledOp::Binary(op, map_child(a), map_child(b)),
            ExprNode::Ternary(op, a, b, c) => {
                ScheduledOp::Ternary(op, map_child(a), map_child(b), map_child(c))
            }
            // Same unreachable precondition as `Dwrt` above: `passes::legalize`
            // runs `expand_refs` first in every compile entry point, so a
            // reference here means this schedule was built without the
            // lowering pipeline. Refusing is not a limitation to lift — a
            // surviving reference is a *call*, and codegen emits one flat
            // function per kernel with no ABI for one
            // (docs/plans/2026-09-09-composition-is-linking.md §5.2).
            ExprNode::Ref(key) => panic!(
                "arena_to_schedule: {key:?} names a kernel whose body is not in \
                 this arena. expand_refs runs first in every compile entry \
                 point, so a survivor means this schedule was built without \
                 the lowering pipeline."
            ),
            ExprNode::Nary(_, _) => panic!("Nary not supported in JIT arena compilation"),
            // The lane fold, executed by lanes: its body is the store, and
            // the store is this def, with the fold's trip count as its width.
            ExprNode::Reduce {
                fold: Fold::Range(fold),
                body,
            } if matches!(arena.node(body), ExprNode::Write { lane, .. } if lane == fold.binder()) =>
            {
                let ExprNode::Write {
                    row, col, value, ..
                } = arena.node(body)
                else {
                    unreachable!("matched a Write above")
                };
                ScheduledOp::Write {
                    row,
                    col,
                    lane: fold.binder(),
                    lanes: fold.len(),
                    value: map_child(value),
                }
            }
            // A surviving fold: `body` was already walked above (it is an
            // ordinary child, scheduled before its parent by the arena's own
            // topological order), so `map_child(body)` is that per-iteration
            // value's `ValueId` in *this* numbering. `extract_folds` reads
            // it back out into the fold's own `ScopeFold`; nothing after
            // that resolves it as an operand (see `ScheduledOp::Reduce`).
            ExprNode::Reduce {
                fold: Fold::Range(fold),
                body,
            } => ScheduledOp::Reduce(fold, map_child(body)),
            // Unreachable precondition, like `Dwrt` above: every compile
            // entry point runs `passes::resolve`, which replaces an integral
            // no rule closed by its quadrature. An interval is not a loop, so
            // there is nothing here to schedule; a survivor means this
            // schedule was built without the lowering pipeline.
            ExprNode::Reduce {
                fold: fold @ Fold::Interval(_),
                ..
            } => panic!(
                "arena_to_schedule: an interval fold ({fold}) reached the JIT \
                 emitter. An integral is not a loop; passes::resolve replaces \
                 every one by its quadrature in every compile entry point, so \
                 a survivor means this schedule was built without the \
                 lowering pipeline."
            ),
            // G2: a `Guard` is not lowered away like `Reduce`/`Ref` above —
            // it is meant to be *emitted*, not expanded. Its mask is the one
            // real child in this arena, mapped like any other operand; its
            // two arms are names (`KernelKey`s) rather than `ExprId`s, so
            // there is nothing here to `map_child` for them. What emits it
            // is `allocate_nest`'s `extract_guards` (resolves and schedules
            // each arm as its own scope) plus `emit_scope`'s `Guard` arm
            // (the branch itself), not this function.
            ExprNode::Guard { mask, on, off } => ScheduledOp::Guard(map_child(mask), on, off),
            ExprNode::Write { .. } => unreachable!("a Write node is skipped above"),
        };
        // Numbered after the op is built: a `Uniform` may have pushed its
        // block's `Context` def just above, and ids follow schedule order.
        let vid = ValueId(next_id);
        next_id += 1;
        id_map[idx] = vid;
        schedule.push(regalloc::Def {
            value: vid,
            op: sched_op,
        });
    }
    // A `Write` whose lane fold nothing reached is a lane fold that is not
    // where `pack` put it — under a column fold — and the schedule would
    // be silently store-free.
    assert!(
        lane.is_none()
            || schedule
                .iter()
                .any(|d| matches!(d.op, ScheduledOp::Write { .. })),
        "arena_to_schedule: a Write is reachable but no lane fold names it"
    );
    schedule
}

// =============================================================================
// Scopes: which fold computes what
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
            ScheduledOp::Var(idx) if *idx < Variance::VARIABLES => Variance::from_var(*idx),
            ScheduledOp::Var(_) => Variance::ALL,
            ScheduledOp::Lanes(lane) => Variance::from_var(lane.var()),
            // Invariant across the lattice; unknown until the call. The
            // `CONST` here is what carries it into the per-call scope — a
            // context pointer, and a uniform read through one.
            ScheduledOp::Const(_) | ScheduledOp::Context(_) | ScheduledOp::Uniform(..) => {
                Variance::CONST
            }
            ScheduledOp::Unary(_, a)
            | ScheduledOp::ShiftImm(_, a, _)
            // A gather reads from a bound buffer, whose contents are fixed for
            // the kernel's lifetime — its variance is its index's variance.
            // A broadcast is a gather whose index lacks the lane bit, so
            // the same rule places it in the scope its address varies in.
            | ScheduledOp::Gather(a, _)
            | ScheduledOp::Broadcast(a, _) => v[a.0 as usize],
            ScheduledOp::Binary(_, a, b) | ScheduledOp::Seq(a, b) => {
                v[a.0 as usize].union(v[b.0 as usize])
            }
            ScheduledOp::Ternary(_, a, b, c) => v[a.0 as usize]
                .union(v[b.0 as usize])
                .union(v[c.0 as usize]),
            // A completed reduction closes over its own binder: nothing
            // outside the fold can read it (that is what makes it a binder),
            // so the result's variance is the body's, minus that one bit —
            // `Variance::without` is exactly "what a binder does to its own
            // index" (see `variance.rs`'s doc).
            ScheduledOp::Reduce(fold, body) => {
                v[body.0 as usize].without(Variance::from_var(fold.binder().var()))
            }
            // Exactly the mask's, not `Variance::ALL`: an arm's schedule
            // comes from a wholly separate arena (`extract_guards`, which
            // runs after `scope_schedule` — see `allocate_nest`) that
            // `passes::lattice::collapse` never wrapped, so an arm cannot
            // reference *any* binder of the enclosing nest at all — warping
            // does not yet reach into a guard's arms
            // (docs/plans/2026-09-12-emit-should-just-emit.md §8, an open
            // question this stage does not settle). Whichever arm the mask
            // picks, the picked arm's own value is structurally invariant in
            // every binder this schedule has; only the *choice* of arm can
            // vary, and that varies exactly as the mask does.
            //
            // `Variance::ALL` was tried here first and is wrong, not merely
            // imprecise: `Write`/`Reduce`/`Seq` all union their operands'
            // variance, so an `ALL` leaf poisons every structural ancestor up
            // to the root into looking like it depends on binder bits that
            // do not exist in this nest at all. `extract_folds_bound_by`'s
            // scope-placement filter (`bits() & deeper == 0`) then reads that
            // poisoned variance and strips the *ancestor* `Reduce`/`Seq`
            // nodes themselves out of an outer fold's own remaining
            // schedule — not overly conservative, wrong: the outer loop's
            // own structure goes missing, and `attach_folds` fails to find
            // a child fold's `Reduce` def where extraction said it would be.
            ScheduledOp::Guard(mask, ..) => v[mask.0 as usize],
            // A store reads its row and column for the address, so it sits
            // inside both their folds — which is the whole of why the
            // lattice's loops can be placed by the same rule as everything
            // else. And it is the lane fold, so like any `Reduce` it closes
            // over its own binder: the row fold's result must not read as
            // lane-varying, or the body would disown it.
            ScheduledOp::Write {
                row,
                col,
                lane,
                value,
                ..
            } => v[value.0 as usize]
                .without(Variance::from_var(lane.var()))
                .union(Variance::from_var(row.var()))
                .union(Variance::from_var(col.var())),
        };
    }
    v
}

/// Split a flat schedule into its nest, placing every value in the outermost
/// scope that binds all the binders it depends on.
///
/// Two steps, and the second is the placement. [`extract_folds`] carves each
/// surviving `Reduce`'s closure out into a scope of its own, one level per
/// fold; what that leaves in a fold's schedule is everything its body
/// reaches whose variance names no binder deeper than the fold's own — which
/// still includes values invariant *in* the fold, computed in it once per
/// trip. [`place_roots`] then moves each of those up to the outermost scope
/// binding its binders, where it is computed once, and leaves a placeholder
/// behind for the scope that read it. That is loop-invariant code motion
/// out of every fold at once — the lattice's rows and columns included, so
/// a per-call value is computed per call and a per-row one per row
/// (docs/plans/2026-09-16-collapse-is-a-fold.md §2.2) — asked as one
/// question of the variance rather than as a tier per loop.
fn scope_schedule(
    schedule: Vec<regalloc::Def>,
    variance: &[pixelflow_ir::variance::Variance],
) -> regalloc::ScopedSchedule {
    let (body, pending) = extract_folds(schedule, variance);
    // Every scope's `If`s are guarded where a branch pays, so this is
    // where an arm's entries are worth gathering into one run. A no-op
    // unless it buys a branch. Before `attach_folds`, because it is a
    // permutation and a fold's position is a fact about its parent's final
    // order. The folds first: what a fold costs, which decides whether an
    // arm owning it pays for a branch, is made of the folds inside it.
    let (pending, inner): (Vec<PendingFold>, Vec<guards::FoldReads>) =
        pending.into_iter().map(cluster_pending).unzip();
    let reads = pending_reads(&body, &pending, &inner);
    let body = guards::cluster_if_arms(body, &reads);
    let mut scoped = regalloc::ScopedSchedule {
        body: regalloc::ScopeRegion {
            roots: Vec::new(),
            schedule: body,
        },
        folds: Vec::new(),
        // Not `extract_guards`'s job: that runs after this function returns
        // (`allocate_nest`), on the settled body and fold schedules
        // `cluster_if_arms`/`attach_folds`/`place_roots` below produce —
        // see `extract_guards`'s own doc for why it cannot run in here.
        guard_arms: Vec::new(),
    };
    attach_folds(&mut scoped, pending);
    place_roots(&mut scoped, variance);
    scoped
}

/// [`guards::cluster_if_arms`] over a pending fold's schedule and, one
/// level down, each of its children's — innermost first, and handing back
/// the folds the fold's own schedule opens, which the scope it opens in
/// prices it by.
fn cluster_pending(fold: PendingFold) -> (PendingFold, guards::FoldReads) {
    let (children, inner): (Vec<PendingFold>, Vec<guards::FoldReads>) =
        fold.children.into_iter().map(cluster_pending).unzip();
    let reads = pending_reads(&fold.schedule, &children, &inner);
    let clustered = PendingFold {
        reduce_vid: fold.reduce_vid,
        schedule: guards::cluster_if_arms(fold.schedule, &reads),
        children,
    };
    (clustered, reads)
}

/// What each of `folds`, opened in `scope`, reads from it and costs, `inner`
/// being what each fold's own schedule opens, in the same order.
fn pending_reads(
    scope: &[regalloc::Def],
    folds: &[PendingFold],
    inner: &[guards::FoldReads],
) -> guards::FoldReads {
    guards::FoldReads::new(
        scope,
        folds
            .iter()
            .zip(inner)
            .map(|(fold, inner)| (fold.reduce_vid, fold.schedule.as_slice(), inner)),
    )
}

/// Whether a def is a placeholder already, and so not the placement's to
/// park: a binder's `Var` (found where its fold keeps it), a `Reduce` that
/// is not this scope's own (read from its accumulator slot), or a `Guard`
/// (its own `ValueId` forced to a slot, mirroring that accumulator).
///
/// A `Const` used to be here too, as "cheaper rebuilt than reloaded". It is
/// not: rebuilding one is two instructions on x86, and a value parked for the
/// scopes inside is carried in a register when one is free, which is zero.
/// Whether a constant is worth a register is the allocator's question, priced
/// like every other root's, so nothing here answers it.
fn stays_put(op: &ScheduledOp) -> bool {
    matches!(
        op,
        ScheduledOp::Var(_) | ScheduledOp::Reduce(..) | ScheduledOp::Guard(..)
    )
}

/// The second half of [`scope_schedule`]: in every fold, each def whose
/// variance does not name the fold's binder is computed by an enclosing
/// scope — the outermost binding every binder it *does* name — and read
/// here from that scope's park.
///
/// The value is already in the enclosing scope's schedule: a fold's closure
/// was carved out of its parent's, and a value invariant in the fold has no
/// bit deeper than the parent's, so the parent kept it. What changes is that
/// the fold's own copy becomes a placeholder, and the value joins the
/// computing scope's `roots`.
fn place_roots(
    scoped: &mut regalloc::ScopedSchedule,
    variance: &[pixelflow_ir::variance::Variance],
) {
    use pixelflow_ir::variance::Variance;
    use regalloc::Scope;

    // What each scope binds, read off before any def is edited: a fold's own
    // binder. The same index answers for a fold as a scope and as an
    // ancestor.
    //
    // Not the lane binder of a store the scope holds. The lane fold is
    // inlined into the storing scope, so nothing *deeper* may compute a
    // lane-varying value (`extract_folds_bound_by` keeps them out of the
    // folds within) — but the lanes themselves are the same vector in every
    // batch, so a value that varies by lane and by nothing else is invariant
    // over the whole call and belongs to the body, like any other invariant.
    // Counting the lane as bound here made the storing scope keep its own
    // copy of the iota while the body parked another for the scopes inside,
    // and one scope then held the same value as a def and as a live-in.
    let body_binds = Variance::CONST;
    let binds: Vec<Variance> = (0..scoped.folds.len())
        .map(|j| Variance::from_var(binder_of_fold(scoped, j)))
        .collect();
    for j in 0..scoped.folds.len() {
        let own = binds[j];
        // Ancestors, nearest first, each with what it binds; the body is
        // where the chain ends.
        let mut ancestors: Vec<(Scope, Variance)> = Vec::new();
        let mut up = scoped.folds[j].parent;
        loop {
            match up {
                Scope::Body => {
                    ancestors.push((Scope::Body, body_binds));
                    break;
                }
                Scope::Fold(p) => {
                    ancestors.push((up, binds[p]));
                    up = scoped.folds[p].parent;
                }
                Scope::GuardArm(_) => {
                    unreachable!("a fold's parent is never a guard arm (none nest in one)")
                }
            }
        }
        let mut moved: Vec<(Scope, regalloc::ValueId)> = Vec::new();
        for def in &mut scoped.folds[j].schedule {
            let deps = variance[def.value.0 as usize];
            if deps.bits() & own.bits() != 0 || stays_put(&def.op) {
                continue;
            }
            // The outermost ancestor binding every binder `deps` names: the
            // nearest one binding any of them, or the body when none does —
            // an ancestor's binder in `deps` puts the value inside that
            // ancestor, and outside every scope nearer than it.
            let computing = ancestors
                .iter()
                .find(|(_, binds)| deps.bits() & binds.bits() != 0)
                .map_or(Scope::Body, |(scope, _)| *scope);
            moved.push((computing, def.value));
            // The placeholder; never emitted, located at the park. A vector's
            // says nothing about the value it stands for; a pointer's stays
            // its own op, which is operand-free already, so the scope inside
            // still reads the class off it.
            if def.op.class() == regalloc::Class::Vector {
                def.op = ScheduledOp::Const(0.0);
            }
        }
        for (computing, vid) in moved {
            let roots = match computing {
                Scope::Body => &mut scoped.body.roots,
                Scope::Fold(p) => &mut scoped.folds[p].roots,
                Scope::GuardArm(_) => {
                    unreachable!("place_roots never computes an ancestor as a guard arm")
                }
            };
            if !roots.contains(&vid) {
                roots.push(vid);
            }
        }
    }
}

/// The lane binders a scope binds by holding a store: the lane fold is
/// executed by lanes, inlined into the scope its `Write` def sits in, so
/// that scope is where a lane-varying value lives — nothing deeper binds
/// the lane, and nothing shallower may compute a value that varies by it.
fn lane_binders(schedule: &[regalloc::Def]) -> pixelflow_ir::variance::Variance {
    use pixelflow_ir::variance::Variance;
    schedule
        .iter()
        .fold(Variance::CONST, |acc, def| match def.op {
            ScheduledOp::Write { lane, .. } => acc.union(Variance::from_var(lane.var())),
            _ => acc,
        })
}

/// A fold's binder, read off the `Reduce` def it opens at in its parent.
fn binder_of_fold(scoped: &regalloc::ScopedSchedule, j: usize) -> u8 {
    use regalloc::Scope;
    let fold = &scoped.folds[j];
    let def = match fold.parent {
        Scope::Body => &scoped.body.schedule[fold.at],
        Scope::Fold(p) => &scoped.folds[p].schedule[fold.at],
        Scope::GuardArm(_) => {
            unreachable!("a fold's parent is never a guard arm (none nest in one)")
        }
    };
    let ScheduledOp::Reduce(meta, _) = &def.op else {
        panic!("Fold({j}) opens at a def that is not a Reduce")
    };
    meta.binder().var()
}

/// A fold `extract_folds` carved out of a flat schedule, still looking for
/// its position: everything the body's closure computed, in topological
/// order, ending at the value the `Reduce` combines each iteration.
struct PendingFold {
    /// The `Reduce` def's own `ValueId` — a schedule permutation (LICM, arm
    /// clustering) may move it, but never renames it, so this is what
    /// `attach_folds` searches the post-partition schedule for.
    reduce_vid: regalloc::ValueId,
    /// The fold's own per-iteration computation, in topological order,
    /// ending at the body's root.
    schedule: Vec<regalloc::Def>,
    /// The folds whose `Reduce` def sits in `schedule`: a fold inside this
    /// one's body. The nest is a tree, and this is the recursion.
    children: Vec<PendingFold>,
}

/// Carve every surviving `Reduce`'s body out of `schedule`.
///
/// A fold inside a fold's body is carved the same way, one level down: the
/// outer fold's closure is a schedule like any other, and the inner fold is
/// a `Reduce` def in it. `variance` is indexed by `ValueId` (dense and
/// positional in the arena's own schedule — `arena_to_schedule` assigns
/// them in the order it pushes `Def`s), which is what lets one array answer
/// for every level.
fn extract_folds(
    schedule: Vec<regalloc::Def>,
    variance: &[pixelflow_ir::variance::Variance],
) -> (Vec<regalloc::Def>, Vec<PendingFold>) {
    let top = alloc::vec![false; variance.len()];
    extract_folds_bound_by(
        schedule,
        variance,
        pixelflow_ir::variance::Variance::CONST,
        &top,
    )
}

/// [`extract_folds`] for one scope, `bound` being the binders that scope is
/// *inside*: none at the top, a fold's own binder and its ancestors' for
/// that fold's body.
///
/// `bound` is what decides which values leave. Nothing outside a fold can
/// read its own binder — that is what makes it a binder — so a value whose
/// variance names a binder this scope is *not* inside can only belong to a
/// fold nested deeper, and dropping it here is safe by construction. A value
/// a fold's closure also reached but whose variance names no deeper binder
/// (a shared invariant leaf, or one that varies only with an enclosing
/// binder) is not dropped: it stays here too, and stays in the fold's own
/// schedule as well — where [`place_roots`] turns it into a placeholder read
/// from the enclosing scope's park. Getting this backwards — removing the
/// whole closure — would orphan exactly that shared leaf's other consumer.
/// The closure ends at such a value: what it is built from is the computing
/// scope's business, not the fold's, and is not carried in behind it.
///
/// The one thing that is *not* recomputed inside a fold is another fold
/// that does not depend on its binder: a whole loop per iteration is the
/// glyph's winding sum run once per piece of its distance fold, and once
/// more at the top for the coverage that reads it. Such a `Reduce` stays
/// the enclosing scope's fold, and the fold that reads its result keeps its
/// def as a **placeholder** — in the schedule, so the reads resolve, but
/// opening no scope here; `allocate_nest` parks it in its accumulator slot,
/// where the enclosing scope's loop left it before this one began.
/// `placeholder` is that verdict from the level above, by `ValueId`, so a
/// level never mistakes one for a fold of its own.
fn extract_folds_bound_by(
    schedule: Vec<regalloc::Def>,
    variance: &[pixelflow_ir::variance::Variance],
    bound: pixelflow_ir::variance::Variance,
    placeholder: &[bool],
) -> (Vec<regalloc::Def>, Vec<PendingFold>) {
    use pixelflow_ir::variance::Variance;

    // `ValueId` space, not this schedule's length: a fold's schedule is a
    // subset of its parent's, keeping the parent's ids.
    let n = variance.len();
    let mut position: Vec<Option<usize>> = alloc::vec![None; n];
    for (i, def) in schedule.iter().enumerate() {
        position[def.value.0 as usize] = Some(i);
    }
    // The lane fold is executed by lanes, inlined into the scope holding its
    // store — so that scope binds its binder, and a lane-varying value is
    // computed here, not somewhere deeper.
    let bound = bound.union(lane_binders(&schedule));
    // A binder this scope is not inside: what a value carrying one is
    // nested under, and what a fold read from outside does not carry.
    let deeper = Variance::BINDERS.bits() & !bound.bits();
    let hoisted = |v: regalloc::ValueId| {
        placeholder[v.0 as usize] || variance[v.0 as usize].bits() & deeper == 0
    };

    let mut pending: Vec<PendingFold> = Vec::new();
    // A `Reduce` inside another's closure is that one's child, not this
    // scope's own fold. The schedule is topological, so an outer fold's def
    // comes after everything its body reaches; walking it backwards meets
    // the outer fold first, and what its closure claims is skipped here and
    // carved out by the recursion instead.
    let mut claimed = alloc::vec![false; n];

    for def in schedule.iter().rev() {
        let ScheduledOp::Reduce(fold, body_vid) = def.op else {
            continue;
        };
        if claimed[def.value.0 as usize] || placeholder[def.value.0 as usize] {
            continue;
        }
        // The fold's own closure: everything its body needs, reachable by
        // operand from its root. This *includes* any invariant leaf it
        // shares with code outside it (a `Uniform`, a shared sub-expression)
        // — reachability says nothing about whether such a value depends on
        // *this* binder, which is why it is not what decides removal below.
        //
        // It stops *at* such a leaf, though. A value the enclosing scope
        // computes is a placeholder in this fold and in every fold inside
        // it — read from its park, never emitted — so nothing behind it is
        // this fold's to read. Following through would carry its operands
        // in as placeholders no def here reads: roots the computing scope
        // must then park, for nobody. The `Uniform`s' base pointer was that
        // root, stored to the frame once per call for a fold that reads only
        // the uniforms themselves.
        //
        // A nested `Reduce` that depends on a binder bound here is followed
        // *into*: its body is not an operand (`regalloc::operands` says so —
        // the def's own emission never reads it), but it is this closure's
        // to carry — `regalloc::structural_children`, which the walk below
        // follows, yields it — so the recursion below can carve it out again
        // one level down. One that does not is a placeholder like any other
        // hoisted value, remembered so the level below does not mistake it
        // for a fold of its own.
        //
        // A `Guard` is the exception to stopping: `stays_put` says a fold
        // emits one wherever it reaches it, hoisted or not, so its mask is
        // this fold's to read and the walk goes through.
        let mut mark = alloc::vec![false; n];
        let mut placeholder_here = alloc::vec![false; n];
        let op_of = |v: regalloc::ValueId| position[v.0 as usize].map(|p| &schedule[p].op);
        let is_fold = |v: regalloc::ValueId| matches!(op_of(v), Some(ScheduledOp::Reduce(..)));
        let stops =
            |v: regalloc::ValueId| hoisted(v) && !matches!(op_of(v), Some(ScheduledOp::Guard(..)));
        let mut stack = Vec::new();
        mark[body_vid.0 as usize] = true;
        // The root too: a body that *is* another fold's result reads that
        // result from its slot, and the whole schedule is the placeholder.
        if stops(body_vid) {
            placeholder_here[body_vid.0 as usize] = is_fold(body_vid);
        } else {
            stack.push(body_vid);
        }
        while let Some(v) = stack.pop() {
            let at = position[v.0 as usize].unwrap_or_else(|| {
                panic!(
                    "{v:?} is read by {:?}'s body but is not in its scope",
                    def.value
                )
            });
            let op = &schedule[at].op;
            if matches!(op, ScheduledOp::Reduce(..)) {
                claimed[v.0 as usize] = true;
            }
            for operand in regalloc::structural_children(op) {
                if mark[operand.0 as usize] {
                    continue;
                }
                mark[operand.0 as usize] = true;
                if stops(operand) {
                    placeholder_here[operand.0 as usize] = is_fold(operand);
                    continue;
                }
                stack.push(operand);
            }
        }
        let fold_schedule: Vec<regalloc::Def> = schedule
            .iter()
            .filter(|d| mark[d.value.0 as usize])
            .cloned()
            .collect();
        let inside = bound.union(Variance::from_var(fold.binder().var()));
        let (fold_schedule, children) =
            extract_folds_bound_by(fold_schedule, variance, inside, &placeholder_here);
        pending.push(PendingFold {
            reduce_vid: def.value,
            schedule: fold_schedule,
            children,
        });
    }
    // Schedule order, so fold indices are stable whichever way this walked.
    pending.reverse();

    if pending.is_empty() {
        return (schedule, pending);
    }

    let remaining: Vec<regalloc::Def> = schedule
        .into_iter()
        .filter(|def| variance[def.value.0 as usize].bits() & deeper == 0)
        .collect();
    (remaining, pending)
}

/// Locate each [`PendingFold`]'s `Reduce` def in the body's schedule and
/// record it as a [`regalloc::ScopeFold`].
///
/// Searched by value rather than carried through as a position, because
/// [`guards::cluster_if_arms`] is a schedule *permutation* — it moves a
/// `Def`, never renames the `ValueId` it defines.
fn attach_folds(scoped: &mut regalloc::ScopedSchedule, pending: Vec<PendingFold>) {
    for fold in pending {
        let at = scoped
            .body
            .schedule
            .iter()
            .position(|def| def.value == fold.reduce_vid)
            .unwrap_or_else(|| {
                panic!(
                    "{:?}'s Reduce def is not in the body after extract_folds",
                    fold.reduce_vid
                )
            });
        attach_fold(scoped, fold, regalloc::Scope::Body, at);
    }
}

/// Record `fold` as a [`regalloc::ScopeFold`] opening at `at` in `parent`,
/// then each of its children inside it — depth first, so a parent's index is
/// always below its children's, which is the order `allocate_nest` and the
/// frame layout both walk the tree in.
///
/// A child's position is a search of its parent's schedule, for the same
/// reason [`attach_folds`] searches rather than carries: the allocator keeps
/// a fold's evaluation order, but a position is a fact about a schedule and
/// this is the schedule it will be asked of.
fn attach_fold(
    scoped: &mut regalloc::ScopedSchedule,
    fold: PendingFold,
    parent: regalloc::Scope,
    at: usize,
) {
    let PendingFold {
        reduce_vid,
        schedule,
        children,
    } = fold;
    let index = scoped.folds.len();
    scoped.folds.push(regalloc::ScopeFold {
        parent,
        at,
        roots: Vec::new(),
        schedule,
    });
    for child in children {
        let at = scoped.folds[index]
            .schedule
            .iter()
            .position(|def| def.value == child.reduce_vid)
            .unwrap_or_else(|| {
                panic!(
                    "{:?}'s Reduce def is not in {reduce_vid:?}'s body, which \
                     extract_folds said it was nested in",
                    child.reduce_vid
                )
            });
        attach_fold(scoped, child, regalloc::Scope::Fold(index), at);
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
        return Ok(InstructionPlan {
            reloads: Vec::new(),
            op: ResolvedOp::Context { dst, slot: *slot },
            setup_mov: None,
            scratch,
        });
    }

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

    // Resolve a value to its register, or plan a reload from stack/constant into `target`.
    let loc_of = |v: regalloc::ValueId| -> Binding {
        locs.get(v.0 as usize)
            .copied()
            .flatten()
            .unwrap_or_else(|| panic!("{v:?} has no binding"))
    };
    // The address an instruction reads, in a pointer register: where the
    // allocator keeps it, or reloaded from its slot into the one pointer
    // register it reserved for this instruction. Never a constant.
    let base_of = |v: regalloc::ValueId, reloads: &mut Vec<Reload>| -> PtrReg {
        match loc_of(v) {
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
            Binding::Loc(Loc::Ptr(p)) => {
                panic!("{v:?} is read as a vector but is an address in {p:?}")
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
        // Unreachable precondition: a surviving `Reduce`'s def is forced to
        // `Where::Spilled` at scan time (never a register — the `dst` match
        // above already panics on that), and `emit_dag_body_hoisted` special-
        // cases it before this function is ever called, the same way a
        // hoisted placeholder never reaches here either.
        ScheduledOp::Reduce(..) => unreachable!(
            "resolve_operands: a Reduce def reached the generic resolver -- \
             emit_scope must special-case it before calling this"
        ),
        // Same unreachable precondition, for the same reason: `emit_scope`
        // special-cases a `Guard` def (its branch, its two arms, its result
        // store) before this function is ever called.
        ScheduledOp::Guard(..) => unreachable!(
            "resolve_operands: a Guard def reached the generic resolver -- \
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
                            Binding::Loc(Loc::Ptr(p)) => {
                                panic!("{c:?} is read as a vector but is an address in {p:?}")
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
fn compile_native(
    schedule: Vec<regalloc::Def>,
    ctx: EmitCtx,
) -> Result<CompileResult, CompileError> {
    match crate::isa::detect() {
        Isa::Avx2 => compile_via_backend(schedule, &mut avx2::driver::Avx2Backend::new(ctx)),
        Isa::Avx512 => compile_via_backend(schedule, &mut avx512::driver::Avx512Backend::new(ctx)),
        Isa::Neon => compile_via_backend(schedule, &mut aarch64::driver::Aarch64Backend::new(ctx)),
    }
}

/// The register file of the backend [`compile_native`] instantiates: the
/// width a kernel is legalized at before it is scheduled, and what the
/// allocator's tests allocate against without emitting. The same `match`, so
/// the two cannot name different backends.
fn native_register_file(ctx: EmitCtx) -> regalloc::RegisterFile {
    match crate::isa::detect() {
        Isa::Avx2 => avx2::driver::Avx2Backend::new(ctx).register_file(),
        Isa::Avx512 => avx512::driver::Avx512Backend::new(ctx).register_file(),
        Isa::Neon => aarch64::driver::Aarch64Backend::new(ctx).register_file(),
    }
}

/// Compile an [`ExprArena`] DAG into a **collapse** kernel for a lattice of
/// `shape`: the kernel is wrapped in the lattice's folds by
/// [`pixelflow_ir::passes::legalize`], and every fold is emitted as a loop
/// inside the code — one call fills the whole extent with no per-row or
/// per-batch Rust↔JIT boundary. Matches the
/// [`KernelFn`](executable::KernelFn) ABI `(ctx, out, pitch)`.
///
/// The context is one base pointer per declared buffer, in the arena's slot
/// order, followed by the uniform block's base pointer (`f32` values in the
/// arena's uniform-slot order, read once per call) and then the origin
/// block's: `x0`, `y0`.
///
/// # Panics
///
/// Panics if the arena names a retired coordinate axis (`Var(2)`/`Var(3)`,
/// the old Z and W). This is the boundary the check belongs on, because it
/// is the *only* one every route to machine code passes through — the
/// shape-keyed cache is one caller, and the benchmark harnesses, the corpus
/// tools and several tests come straight here. `collapse` substitutes only
/// `X` and `Y`, so a retired axis would survive into the schedule as a `Var`
/// no fold binds, and the allocator's refusal there names a binder, not an
/// axis; this one names the axis.
pub fn compile(
    arena: &pixelflow_ir::arena::ExprArena,
    root: pixelflow_ir::arena::ExprId,
    shape: LatticeShape,
) -> Result<CompileResult, CompileError> {
    assert!(
        arena.retired_axis(root).is_none(),
        "emit::compile: the arena names Var({:?}), a coordinate axis a \
         lattice no longer has; a per-call scalar is a Uniform",
        arena.retired_axis(root)
    );
    EmitCtx::default().compile(arena, root, shape)
}

/// The nest, scoped and allocated: what every scope's frame and code are
/// read off. One allocation pass over the whole nest, so each scope's frame
/// is a function of its own allocation and the shared frame is read off
/// these rather than computed by allocating everything twice.
fn allocate_nest(
    schedule: Vec<regalloc::Def>,
    file: &regalloc::RegisterFile,
) -> regalloc::NestAllocation {
    use regalloc::RegisterAllocator;

    // Variance first, over the arena's *full* schedule — a surviving
    // `Reduce`'s own result depends on its body's, and the body's def is
    // about to move (`extract_folds`, next) out of this array entirely.
    let variance = schedule_variance(&schedule);
    let mut scoped = scope_schedule(schedule, &variance);
    // After `scope_schedule`, not inside it: a guard arm's schedule is not
    // carved out of this nest's own the way a fold's is (`extract_folds`) —
    // it comes from resolving a wholly separate `KernelKey` — and
    // `place_roots`'s LICM is keyed on *this* nest's own binder variance
    // array, which has no meaning for a value in an arm's arena. See
    // `extract_guards`'s own doc.
    extract_guards(&mut scoped);
    regalloc::LinearScan.allocate_nest(scoped, file)
}

/// Resolve every surviving `Guard`'s two arms and attach them to `scoped` as
/// [`regalloc::ScopeGuardArm`]s.
///
/// Runs once over the body and every fold's own schedule (both already
/// settled by [`scope_schedule`] — the folds carved out, their arms
/// clustered, their roots placed), looking for a [`ScheduledOp::Guard`] def
/// at each position. Unlike [`extract_folds`], there is no schedule to carve
/// a subset out of: an arm's `KernelKey` names a kernel in a wholly separate
/// arena, so its schedule is built fresh, from scratch, by
/// [`schedule_guard_arm`] — extraction and attachment are the same step here,
/// which is why this function does both rather than handing a
/// `Vec<PendingGuardArm>` to a second pass the way [`attach_folds`] follows
/// [`extract_folds`].
///
/// Pushes a `Guard`'s `True` arm immediately before its `False` one, which is
/// the pairing [`regalloc::NestAllocation::guard_count`] relies on.
fn extract_guards(scoped: &mut regalloc::ScopedSchedule) {
    // Every arm gets `ValueId`s past every id already in use anywhere in the
    // nest — see `arena_to_schedule_from`'s doc for why a collision is a
    // correctness bug, not merely an odd number. One counter for every arm
    // of every guard, not one per arm: two arms sharing ids with each other
    // is harmless (neither ever parks anything the other reads), but keeping
    // one counter is simpler than arguing that case is fine.
    let max_vid = |schedule: &[regalloc::Def]| {
        schedule
            .iter()
            .map(|d| d.value.0)
            .max()
            .map_or(0, |m| m + 1)
    };
    let mut next_id = max_vid(&scoped.body.schedule);
    for fold in &scoped.folds {
        next_id = next_id.max(max_vid(&fold.schedule));
    }

    let mut arms: alloc::vec::Vec<regalloc::ScopeGuardArm> = alloc::vec::Vec::new();
    let mut schedule_arm = |parent, at, arm, key| {
        let scheduled = schedule_guard_arm(parent, at, arm, key, next_id);
        next_id = scheduled
            .schedule
            .iter()
            .map(|d| d.value.0)
            .max()
            .map_or(next_id, |m| m + 1);
        arms.push(scheduled);
    };
    for (at, def) in scoped.body.schedule.iter().enumerate() {
        if let ScheduledOp::Guard(_, on, off) = def.op {
            schedule_arm(regalloc::Scope::Body, at, guards::IfArm::True, on);
            schedule_arm(regalloc::Scope::Body, at, guards::IfArm::False, off);
        }
    }
    for (j, fold) in scoped.folds.iter().enumerate() {
        for (at, def) in fold.schedule.iter().enumerate() {
            if let ScheduledOp::Guard(_, on, off) = def.op {
                let parent = regalloc::Scope::Fold(j);
                schedule_arm(parent, at, guards::IfArm::True, on);
                schedule_arm(parent, at, guards::IfArm::False, off);
            }
        }
    }
    scoped.guard_arms = arms;
}

/// Resolve `key`, legalize it short of the lattice, and schedule it as one
/// arm of a `Guard`.
///
/// **Short of the lattice**: only `expand_refs`/`lower_dwrt`, never
/// `passes::lattice::{collapse, pack}`. A guard arm is inlined *at its site*
/// in an already-collapsed schedule — it is a value the enclosing kernel
/// consumes, not a second output plane — so wrapping it in its own row/column
/// /lane folds and `Write` would be a second, nonsensical lattice around a
/// value that already lives inside one. `passes::lattice::collapse` already
/// refuses a reachable `Guard` for exactly this reason (its own doc): coordinate
/// warping does not yet reach into a guard's arms at all — a question
/// docs/plans/2026-09-12-emit-should-just-emit.md's §8 leaves open — so an
/// arm referencing a raw coordinate `Var` fails loudly right here, in
/// `arena_to_schedule`'s own "a coordinate that survived collapse" panic,
/// rather than silently.
///
/// # Panics
///
/// - If `key` resolves to nothing (an arm must be interned before it can
///   reach codegen).
/// - If lowering finds a `Dwrt` with no derivative rule.
/// - If the arm's own schedule contains a nested `Reduce` or `Guard` — not
///   supported in this stage (G2's stated non-goal): a guard arm is a
///   straight-line expression, not a second loop nest or a second branch.
fn schedule_guard_arm(
    parent: regalloc::Scope,
    at: usize,
    arm: guards::IfArm,
    key: pixelflow_ir::key::KernelKey,
    starting_id: u32,
) -> regalloc::ScopeGuardArm {
    let kernel = pixelflow_ir::store::KernelStore::resolve(key).unwrap_or_else(|| {
        panic!(
            "schedule_guard_arm: {key:?} names no kernel in the KernelStore -- \
             a Guard's arm must be interned (KernelStore::intern) before it \
             reaches codegen"
        )
    });
    let (arena, root) = kernel.parts();
    let (arena, root) = pixelflow_ir::passes::expand_refs_owned(arena, root);
    let (arena, root) = pixelflow_ir::passes::resolve(&arena, root).unwrap_or_else(|e| {
        panic!("schedule_guard_arm: {key:?}'s arm has no derivative rule: {e}")
    });
    // No origin: the arm's arena was never wrapped by `collapse`, so it
    // declares none, and every uniform it reads is the link's. Said as the
    // type, not as two sentinel slot numbers a real slot could one day reach.
    let schedule = arena_to_schedule_from(&arena, root, None, starting_id);
    for def in &schedule {
        assert!(
            !matches!(def.op, ScheduledOp::Reduce(..) | ScheduledOp::Guard(..)),
            "schedule_guard_arm: {key:?}'s arm schedules a {:?} -- a guard \
             arm nesting a fold or another guard is not supported in this \
             stage (G2, docs/plans/2026-09-12-emit-should-just-emit.md); a \
             guard's arm must be a straight-line expression",
            def.op
        );
    }
    regalloc::ScopeGuardArm {
        parent,
        at,
        arm,
        schedule,
    }
}

/// Drive a schedule to a complete collapse kernel via an [`IsaBackend`]: the
/// body from [`emit_scope`], which emits every fold nested in it, framed by
/// the function's own frame.
fn compile_via_backend<B: IsaBackend>(
    schedule: Vec<regalloc::Def>,
    backend: &mut B,
) -> Result<CompileResult, CompileError> {
    let file = backend.register_file();
    let nest = allocate_nest(schedule, &file);
    let body_alloc = nest.body();

    // Every byte below is emitted through this decorator, so the counts it
    // hands back cover the whole function by construction (see `traffic`).
    let mut counting = Counting::new(backend);

    // Every scope shares one stack frame: spill slots in [0, m), and the
    // park and fold slots above. `m` is the tree max — a fold's frame is
    // based at its parent's top, since its loop runs *in the middle of* its
    // parent's schedule with the parent's spilled values live across it —
    // rounded to a whole slot so the slots above stay naturally aligned.
    // Allocation and frame layout are pure, so pre-sizing here computes
    // exactly the frames the emissions below will.
    let vector_bytes = file.vector_bytes;
    let mut top_of: alloc::collections::BTreeMap<regalloc::Scope, u32> =
        alloc::collections::BTreeMap::new();
    let mut m = 0u32;
    // The body, then the folds in nest order: a fold's parent is always an
    // earlier scope (asserted where the nest is built), so every `top_of`
    // lookup below is already populated.
    let scopes = core::iter::once(regalloc::Scope::Body)
        .chain((0..nest.fold_count()).map(regalloc::Scope::Fold))
        .chain((0..nest.guard_count()).flat_map(|k| {
            [
                regalloc::Scope::GuardArm(2 * k),
                regalloc::Scope::GuardArm(2 * k + 1),
            ]
        }));
    for scope in scopes {
        let base = match scope {
            regalloc::Scope::Body => 0,
            regalloc::Scope::Fold(j) => top_of[&nest.fold_parent(j)],
            // Both of a guard's arms open at the same position, in the same
            // parent, as the fold that would have opened there instead — see
            // `Scope::GuardArm`'s doc.
            regalloc::Scope::GuardArm(i) => top_of[&nest.guard_parent(i / 2)],
        };
        let allocation = nest.scope(scope);
        let top = if allocation.schedule().is_empty() {
            base
        } else {
            FrameLayout::resolve(allocation, vector_bytes, base)?.frame_size
        };
        top_of.insert(scope, top);
        m = m.max(top);
    }
    let m = m.next_multiple_of(vector_bytes);
    // Each surviving fold's two roots get a slot the same way a park does —
    // one that outlives both the scope reading it (wherever the `Reduce`
    // def is) and the fold's own frame: the accumulator's, then the
    // binder's, both by the fold's `Reduce`. Whether either is used is the
    // allocator's answer, read where the loop is emitted.
    let fold_slot = |j: usize, root: usize| m + (2 * j + root) as u32 * vector_bytes;
    let fold_map: alloc::collections::BTreeMap<regalloc::ValueId, u32> = (0..nest.fold_count())
        .map(|j| (nest.fold_reduce_vid(j), fold_slot(j, 0)))
        .collect();
    let binder_map: alloc::collections::BTreeMap<regalloc::ValueId, u32> = (0..nest.fold_count())
        .map(|j| (nest.fold_reduce_vid(j), fold_slot(j, 1)))
        .collect();
    // Each surviving `Guard`'s one result, the same idea, right after the
    // fold slots: an address outside any single scope's frame, because the
    // scope that opens the branch and the two arms that each store into it
    // all address the same one (see `ScheduledOp::Guard`'s doc — this is
    // its accumulator-slot analogue).
    let guard_slot_base = m + 2 * nest.fold_count() as u32 * vector_bytes;
    let guard_slot = |k: usize| guard_slot_base + k as u32 * vector_bytes;
    let guard_map: alloc::collections::BTreeMap<regalloc::ValueId, u32> = (0..nest.guard_count())
        .map(|k| (nest.guard_reduce_vid(k), guard_slot(k)))
        .collect();
    // Every root of every scope, parked above the fold and guard slots. No
    // root is a fold's or a guard's own result — `stays_put` keeps both out
    // of `roots` — so each one takes a park slot of its own. A value two
    // sibling scopes both compute (a row's main batches and its remainder
    // share their closures) is one root with one slot: the two never run at
    // once, and each writes it before its own scopes read it.
    let park_base = guard_slot_base + nest.guard_count() as u32 * vector_bytes;
    let mut parks: alloc::collections::BTreeMap<regalloc::ValueId, u32> =
        alloc::collections::BTreeMap::new();
    let scopes = core::iter::once(regalloc::Scope::Body)
        .chain((0..nest.fold_count()).map(regalloc::Scope::Fold));
    for scope in scopes {
        for &root in nest.scope(scope).roots() {
            // Loud, because the other outcome is silent: `emit_scope`'s
            // `Reduce` and `Guard` arms end their def before the hand-off, so
            // a park for either would never be written and every scope
            // inside would read whatever the slot held.
            assert!(
                !fold_map.contains_key(&root) && !guard_map.contains_key(&root),
                "{root:?} is a fold's or a guard's result, which `stays_put` \
                 keeps out of every scope's roots"
            );
            if parks.contains_key(&root) {
                continue;
            }
            let slot = park_base + parks.len() as u32 * vector_bytes;
            parks.insert(root, slot);
        }
    }
    let total = park_base + parks.len() as u32 * vector_bytes;

    let (body, _, _, spill_count) = emit_scope(
        body_alloc,
        &mut counting,
        &parks,
        FramePlan {
            override_size: Some(m),
            fold_slots: &fold_map,
            binder_slots: &binder_map,
            guard_slots: &guard_map,
            slot_base: 0,
        },
    )?;

    // The function around it: the frame, the anchor for whatever the body's
    // constants are relative to, and what trails the return.
    let mut asm = Assembly::with_capacity(body.len() + FRAME_HEADROOM);
    counting.frame_alloc(&mut asm.code, total);
    counting.anchor(&mut asm);
    asm.code.extend_from_slice(&body);
    counting.frame_free(&mut asm.code, total);
    counting.emit_ret(&mut asm.code);
    let ret_end = asm.code.len();
    counting.finish(&mut asm);
    let trailing = (asm.code.len() - ret_end) as u32;
    let code = asm.finish();
    let scaffold = counting.take(code.len() as u32 - body.len() as u32 - trailing);
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
            regalloc::Scope::GuardArm(_) => {
                unreachable!("a fold's parent is never a guard arm (none nest in one)")
            }
        };
        trips.push(parent_trips * len);
    }
    // Every guard arm, the same idea: it runs at most once per time its
    // parent's own def is reached, so its trip count is its parent's,
    // conservatively — as if the branch always ran that arm, since which
    // arm actually runs is a runtime property this static count does not
    // see (G3's coherence prior is what will eventually price that). Both
    // arms share the same parent, so the same count twice.
    for k in 0..nest.guard_count() {
        let parent_trips = match nest.guard_parent(k) {
            regalloc::Scope::Body => trips[0],
            regalloc::Scope::Fold(p) => trips[p + 1],
            regalloc::Scope::GuardArm(_) => {
                unreachable!("a guard's parent is never a guard arm (none nest in one)")
            }
        };
        trips.push(parent_trips);
        trips.push(parent_trips);
    }

    // A parked root that holds a register at the head of the scopes inside
    // its own is carried rather than reloaded per iteration — read off the
    // placement, which is where the answer lives.
    let carried = parks
        .keys()
        .filter(|root| nest.carried(**root).is_some())
        .count() as u32;
    let exec = unsafe { executable::ExecutableCode::from_code(&code)? };
    Ok(CompileResult {
        code: exec,
        spill_count,
        spill_bytes: m,
        max_regs: file.scratch.len(),
        hoisted_values: parks.len() as u32,
        traffic: EmitTraffic {
            scopes: EmitTraffic::by_index(scopes, trips.len(), nest.fold_count()),
            trips,
            scaffold,
            trailing,
            vector_bytes: file.vector_bytes,
            pool: file.scratch.len(),
            carried,
        },
    })
}

/// Slack for the function's own instructions on top of the body it wraps.
const FRAME_HEADROOM: usize = 64;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::arena::{ExprArena, ExprId};

    /// Lanes in one SIMD batch at the tier this host selected.
    fn lanes() -> usize {
        crate::isa::jit_vector_bytes() / core::mem::size_of::<f32>()
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

    /// `Isa::vector_bytes` is the table `jit_vector_bytes` answers from, and
    /// each backend's register file is the width its kernels are legalized
    /// and framed at. They state one ISA-defined fact twice; this is what
    /// keeps them the same fact.
    #[test]
    fn every_backends_vector_width_is_its_tiers() {
        let ctx = EmitCtx::default;
        let files = [
            (
                Isa::Avx2,
                avx2::driver::Avx2Backend::new(ctx()).register_file(),
            ),
            (
                Isa::Avx512,
                avx512::driver::Avx512Backend::new(ctx()).register_file(),
            ),
            (
                Isa::Neon,
                aarch64::driver::Aarch64Backend::new(ctx()).register_file(),
            ),
        ];
        for (isa, file) in files {
            assert_eq!(file.vector_bytes as usize, isa.vector_bytes(), "{isa:?}");
        }
    }

    /// Run the collapse `code` is over a plane of exactly `shape`'s extent,
    /// and hand the plane back.
    ///
    /// `buffers` binds the arena's buffer slots, `uniforms` its uniform
    /// block, and `(x, y)` is where the lattice's sample `(0, 0)` lies. The
    /// pitch is the width, so a sample reads as `out[row * width + col]`.
    fn collapse_into(
        code: &executable::ExecutableCode,
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
    fn eval_point(code: &executable::ExecutableCode, x: f32, y: f32) -> f32 {
        collapse_into(code, &[], &[], (x, y), POINT)[0]
    }

    /// Evaluate a kernel compiled at [`batch`]: one row of `lanes()` samples
    /// from `(x, y)`.
    fn eval_batch(
        code: &executable::ExecutableCode,
        buffers: &[*const f32],
        uniforms: &[f32],
        x: f32,
        y: f32,
    ) -> Vec<f32> {
        collapse_into(code, buffers, uniforms, (x, y), batch())
    }

    /// `passes::legalize` at `shape` for a target of `lanes` lanes, then
    /// `arena_to_schedule`: everything a compile entry point runs before the
    /// emitter is handed a schedule.
    pub(super) fn schedule_for(
        a: &ExprArena,
        root: ExprId,
        shape: LatticeShape,
        lanes: u32,
    ) -> Vec<regalloc::Def> {
        let collapse = Collapse {
            domain: Domain {
                shape,
                origin: origin(),
            },
            lanes,
        };
        let (a, root) = pixelflow_ir::passes::legalize(a, root, &collapse).expect("legalize");
        let ids = origin_slots(&a);
        arena_to_schedule(&a, root, Some(ids))
    }

    /// [`schedule_for`] at this host's own lane count.
    fn native_schedule(a: &ExprArena, root: ExprId, shape: LatticeShape) -> Vec<regalloc::Def> {
        schedule_for(a, root, shape, lanes() as u32)
    }

    /// The uniform slots a schedule built by hand names for the origin.
    ///
    /// A raw arena declares no uniform at all, so [`origin_slots`] has
    /// nothing to find; the tests below that feed the scheduler an
    /// unlegalized arena on purpose name the slots themselves.
    const RAW_ORIGIN: Option<[UniformId; 2]> = Some([UniformId(0), UniformId(1)]);

    /// A `Dwrt` that reaches the scheduler (a caller bypassed the lowering
    /// pipeline) must fail loudly at the schedule boundary, not as a cryptic
    /// emit panic. The compile entry points run `lower_dwrt` first, so this
    /// exercises calling `arena_to_schedule` directly.
    ///
    /// Its operand is a fold binder rather than `X`: a coordinate `Var` is a
    /// survivor of its own, refused a line earlier (see
    /// `a_surviving_coordinate_fails_loudly`), and would answer for the
    /// `Dwrt` before the `Dwrt` was ever reached.
    #[test]
    #[should_panic(expected = "Dwrt (autodiff) node reached the JIT")]
    fn surviving_dwrt_fails_loudly() {
        let mut a = ExprArena::new();
        let i = a.push_var(Binder::from_slot(0).expect("slot 0 exists").var());
        let v = a.push_const(0.0);
        let root = a.push_binary(OpKind::Dwrt, i, v);
        let _ = arena_to_schedule(&a, root, RAW_ORIGIN);
    }

    /// A coordinate `Var` is the survivor the collapse ABI added: the
    /// lattice's folds substitute `X` and `Y` away, so one reaching the
    /// scheduler is an arena that never went through `passes::lattice`.
    #[test]
    #[should_panic(expected = "is a coordinate")]
    fn a_surviving_coordinate_fails_loudly() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let two = a.push_const(2.0);
        let root = a.push_binary(OpKind::Mul, x, two);
        let _ = arena_to_schedule(&a, root, RAW_ORIGIN);
    }

    /// And the same for an integral: an interval fold is not a loop, and
    /// `passes::resolve` replaces every one by its quadrature before a
    /// schedule is built. A constant integrand, so no coordinate reaches the
    /// scheduler ahead of the fold and trips its own panic first.
    #[test]
    #[should_panic(expected = "an interval fold")]
    fn a_surviving_interval_fails_loudly() {
        let area = pixelflow_ir::Kernel::constant(1.0).area();
        let (arena, root) = area.parts();
        let _ = arena_to_schedule(arena, root, RAW_ORIGIN);
    }

    /// And the same for a `Ref`: its body is not in this arena at all, so a
    /// survivor is a schedule built without `expand_refs`. `compile` runs
    /// `legalize` first, so this too has to call the scheduler directly.
    #[test]
    #[should_panic(expected = "names a kernel whose body is not in")]
    fn a_surviving_reference_fails_loudly() {
        let named = pixelflow_ir::Kernel::x()
            .mul(&pixelflow_ir::Kernel::constant(3.0))
            .by_ref();
        let (arena, root) = named.parts();
        let _ = arena_to_schedule(arena, root, RAW_ORIGIN);
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

    /// **The new path this stage adds.** `passes::legalize` still runs
    /// `expand_reduce` unconditionally — that is 2c's byte-identical gate,
    /// no production kernel reaches codegen with a surviving `Reduce` yet —
    /// so this test is the only thing exercising it, the same way
    /// `surviving_dwrt_fails_loudly` above reaches the scheduler directly to
    /// see a shape `legalize` would otherwise have cleaned up first.
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
        let sum_code = EmitCtx::default()
            .compile(&sum_arena, sum_root, POINT)
            .expect("a surviving SUM Reduce compiles");

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
        let min_code = EmitCtx::default()
            .compile(&min_arena, min_root, POINT)
            .expect("a surviving MIN Reduce compiles");

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

        let code = EmitCtx::default()
            .compile(&a, root, POINT)
            .expect("a fold that feeds further arithmetic compiles");

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
    /// but pushed to the arena in the *interleaved* order a real unroll
    /// produces (const, add, const, add, const, add — see
    /// `unroll_reduce`'s substitution) rather than all three constants
    /// first. That reordering alone, with no change to the DAG's shape,
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

        let code = EmitCtx::default()
            .compile(&a, root, POINT)
            .expect("interleaved shared-leaf order compiles");

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

        let code = EmitCtx::default()
            .compile(&a, root, POINT)
            .expect("a fold under register pressure compiles");

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
        for ctx in [
            EmitCtx::default(),
            EmitCtx::with_max_regs(regalloc::RegisterFile::MIN_SCRATCH),
        ] {
            let code = ctx
                .compile(&a, root, POINT)
                .expect("a fold inside a fold compiles");
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
        let code = EmitCtx::default()
            .compile(&a, root, POINT)
            .expect("a fold three deep compiles");
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

        for ctx in [
            EmitCtx::default(),
            EmitCtx::with_max_regs(regalloc::RegisterFile::MIN_SCRATCH),
        ] {
            let code = ctx.compile(&a, root, POINT).expect("sibling folds compile");
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

        let file = native_register_file(EmitCtx::default());
        let nest = allocate_nest(native_schedule(&a, root, POINT), &file);
        let j = kernel_fold(&nest, 3).expect("the kernel's fold is three trips");
        assert_eq!(
            nest.fold_parent(j),
            regalloc::Scope::Body,
            "a fold reading no coordinate belongs to the scope that runs once \
             per call, not to a lattice fold that reruns it per sample"
        );

        let code = EmitCtx::default()
            .compile(&a, root, POINT)
            .expect("a hoisted fold compiles");
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
                .flat_map(|j| {
                    let roots = nest.fold_roots(j);
                    [roots.binder, roots.accumulator]
                })
                .filter(|at| matches!(at, regalloc::Where::Reg(_)))
                .count()
        };
        let floor = regalloc::RegisterFile::MIN_SCRATCH;
        let mut previous = None;
        for above in [0, 5] {
            let ctx = EmitCtx::with_max_regs(floor + above);
            let file = native_register_file(ctx.clone());
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
            let code = ctx
                .compile(&a, root, POINT)
                .expect("a fold three deep compiles");
            assert_three_deep(&code);
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
        let file = native_register_file(EmitCtx::default());
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

        let code = EmitCtx::default()
            .compile(&a, root, POINT)
            .expect("a hoisted fold compiles");
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

        let code = EmitCtx::default()
            .compile(&a, root, POINT)
            .expect("a fold whose body is a hoisted fold compiles");
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

        let code = EmitCtx::default()
            .compile(&a, root, POINT)
            .expect("nested folds under register pressure compile");

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

        let ctx = EmitCtx::default();
        let mut neon = aarch64::driver::Aarch64Backend::new(ctx.clone());
        let mut avx2b = avx2::driver::Avx2Backend::new(ctx.clone());
        let mut avx512b = avx512::driver::Avx512Backend::new(ctx);

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

        let neon_len = compile_via_backend(for_backend(neon.register_file()), &mut neon)
            .expect("NEON emit")
            .code
            .len();
        assert!(
            neon_len > 0 && neon_len.is_multiple_of(4),
            "aarch64 is fixed-width"
        );
        for (name, len) in [
            (
                "AVX2",
                compile_via_backend(for_backend(avx2b.register_file()), &mut avx2b)
                    .expect("AVX2")
                    .code
                    .len(),
            ),
            (
                "AVX-512",
                compile_via_backend(for_backend(avx512b.register_file()), &mut avx512b)
                    .expect("AVX-512")
                    .code
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
    /// verb that emits one — and [`EmitTraffic::trailing`] counts the bytes
    /// after it, so the return ends `trailing` bytes before the end.
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
            let frame_end = prologue.code.len();
            probe.anchor(&mut prologue);
            let anchor_end = prologue.code.len();
            let lea = frame_end..anchor_end - REL32;

            for &(name, arena, root) in kernels {
                let mut backend = fresh();
                let lanes = backend.register_file().vector_bytes / BYTES_PER_LANE;
                let result =
                    compile_via_backend(schedule_for(arena, root, PLANE, lanes), &mut backend)
                        .unwrap_or_else(|e| panic!("{tier}/{name}: {e:?}"));
                let code = result.code.as_bytes();
                let trailing = result.traffic.trailing as usize;

                let ret_end = code.len() - trailing;
                assert_eq!(
                    code[ret_end - CLEAN_RETURN.len()..ret_end],
                    CLEAN_RETURN,
                    "{tier}/{name}: the return is not `vzeroupper; ret`"
                );

                assert_eq!(
                    code[lea.clone()],
                    prologue.code[lea.clone()],
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
        let ctx = EmitCtx::default;
        check("AVX2", || avx2::driver::Avx2Backend::new(ctx()), &kernels);
        check(
            "AVX-512",
            || avx512::driver::Avx512Backend::new(ctx()),
            &kernels,
        );
    }

    /// The aarch64 constant pool must APPEND across the scopes a collapse
    /// compile pushes through one backend, never reset.
    ///
    /// An earlier scope's bytes already have the first pool's X17-relative
    /// offsets baked in, so a reset leaves them pointing at different
    /// constants — the "macOS glyph-ink regression", which painted glyphs with
    /// the wrong ink and was only ever observable by running the app on a Mac.
    ///
    /// It is an aarch64 bug, not a macOS one, and now it is a sub-millisecond
    /// unit test on every host.
    #[test]
    fn aarch64_const_pool_appends_across_scopes() {
        /// One scope: a uniform (read through its block's pointer) times a
        /// constant only the pool can hold.
        fn scope_for(k: f32) -> Vec<regalloc::Def> {
            alloc::vec![
                regalloc::Def {
                    value: regalloc::ValueId(0),
                    op: ScheduledOp::Context(0),
                },
                regalloc::Def {
                    value: regalloc::ValueId(1),
                    op: ScheduledOp::Uniform(regalloc::ValueId(0), 0),
                },
                regalloc::Def {
                    value: regalloc::ValueId(2),
                    op: ScheduledOp::Const(k),
                },
                regalloc::Def {
                    value: regalloc::ValueId(3),
                    op: ScheduledOp::Binary(
                        OpKind::Mul,
                        regalloc::ValueId(1),
                        regalloc::ValueId(2),
                    ),
                },
            ]
        }

        // Two constants that genuinely need the pool (not FMOV-immediate).
        let (first, second) = (0.123_456_79_f32, 987.654_3_f32);
        assert!(aarch64::needs_const_pool(first));
        assert!(aarch64::needs_const_pool(second));

        let mut backend = aarch64::driver::Aarch64Backend::new(EmitCtx::default());
        emit_dag_body(scope_for(first), &mut backend).expect("first scope");
        let after_first = backend.pool_entries().to_vec();
        assert!(!after_first.is_empty(), "the first scope pooled nothing");
        emit_dag_body(scope_for(second), &mut backend).expect("second scope");

        assert!(
            backend.pool_entries().starts_with(&after_first),
            "the second scope RESET the constant pool: the first scope's \
             baked-in X17-relative offsets now name different constants — the \
             glyph-ink regression. Pool was {after_first:?}, became {:?}",
            backend.pool_entries()
        );
    }

    // =========================================================================
    // What the nest does and does not partition
    // =========================================================================

    /// A constant shared between a lattice-invariant expression and a varying
    /// one is computed by the outer scope and parked for the inner one, like
    /// any other value the inner scope reads but does not vary.
    ///
    /// It used to be computed in both: `place_roots` left a leaf where it was,
    /// on the theory that nothing is saved by parking a value one instruction
    /// rebuilds. Two instructions on x86, per read, per trip — and a parked
    /// root is carried in a register when one is free, which is none.
    #[test]
    fn a_leaf_feeding_both_scopes_is_parked_by_the_outer_one() {
        let (a, root) = shared_leaf_kernel();
        let schedule = native_schedule(&a, root, batch());
        let variance = schedule_variance(&schedule);
        let scoped = scope_schedule(schedule, &variance);

        let k = scoped
            .body
            .schedule
            .iter()
            .find(|d| matches!(d.op, ScheduledOp::Const(v) if v == 3.5))
            .map(|d| d.value)
            .expect("the body computes the constant");
        assert!(
            scoped.body.roots.contains(&k),
            "the body parks it for the fold: roots {:?}",
            scoped.body.roots
        );
        let inner: alloc::vec::Vec<&ScheduledOp> = scoped
            .folds
            .iter()
            .flat_map(|f| f.schedule.iter())
            .filter(|d| d.value == k)
            .map(|d| &d.op)
            .collect();
        assert!(
            !inner.is_empty()
                && inner
                    .iter()
                    .all(|op| matches!(op, ScheduledOp::Const(v) if *v == 0.0)),
            "the fold reads it through a placeholder, never its own copy: {inner:?}"
        );
    }

    /// The placement is total over every scope's schedule, a parked
    /// placeholder's entry included — which reads the park, the enclosing
    /// scope's answer, rather than a range of this scope's own.
    #[test]
    fn a_shared_leaf_is_placed_once_per_scope() {
        let (a, root) = shared_leaf_kernel();
        let file = native_register_file(EmitCtx::default());
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
    // FrameLayout unit tests — the Placement -> address arrow
    // =========================================================================

    /// Build an allocation with the given placements, in schedule order.
    fn allocation_of(placements: &[(u32, regalloc::Where)]) -> regalloc::NestAllocation {
        use regalloc::{Def, RegisterAllocator, ValueId};
        // Allocate a schedule of bare leaves to get a well-formed Allocation,
        // then pin each value where the test wants it.
        let schedule: alloc::vec::Vec<Def> = placements
            .iter()
            .map(|&(v, _)| Def {
                value: ValueId(v),
                // An operand-free vector leaf that is not a constant — see
                // `regalloc::tests::leaf`.
                op: ScheduledOp::Lanes(Binder::from_slot(0).expect("slot 0")),
            })
            .collect();
        let mut a = regalloc::LinearScan.allocate(schedule, &TEST_FILE);
        for &(v, p) in placements {
            a.place(regalloc::Scope::Body, ValueId(v), p);
        }
        a
    }

    #[test]
    fn an_allocation_with_no_spills_needs_no_frame() {
        let a = allocation_of(&[(0, regalloc::Where::Reg(Reg(4)))]);
        let layout = FrameLayout::resolve(a.body(), 16, 0).unwrap();
        assert_eq!(layout.frame_size, 0);
        assert_eq!(layout.of(regalloc::ValueId(0)), Loc::Reg(Reg(4)).into());
    }

    #[test]
    fn one_spill_takes_one_slot() {
        let a = allocation_of(&[(5, regalloc::Where::Spilled)]);
        let layout = FrameLayout::resolve(a.body(), 16, 0).unwrap();
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
            let layout = FrameLayout::resolve(a.body(), vector_bytes, 0).unwrap();
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
        let layout = FrameLayout::resolve(a.body(), 16, 0).unwrap();
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
        let mut layout = FrameLayout::resolve(a.body(), 16, 0).unwrap();
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
        scratch: regalloc::RegSet::range(4, regalloc::RegisterFile::MIN_SCRATCH),
        temps_for: regalloc::no_temps,
        guard_temps: 0,
        vector_bytes: 16,
        gpr_ctx: None,
        gpr_out: None,
        gpr_pitch: None,
        gpr_scratch: regalloc::GprSet::EMPTY,
        gpr_temps_for: regalloc::no_temps,
        pointers: regalloc::GprSet::EMPTY,
        mask_scratch: regalloc::MaskSet::EMPTY,
        mask_temps_for: regalloc::no_temps,
        mask_guard_temps: 0,
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
    // Arena compilation tests
    // =========================================================================

    // These tests call the private `arena_to_schedule` directly rather than
    // through `compile`: value numbering and dead-node filtering are
    // schedule-shape invariants with no output-value signature (a regression
    // here wastes registers/instructions, it doesn't change what a compiled
    // kernel computes), so there is no public black-box assertion that would
    // catch a break here.

    /// Every operand a schedule names is defined earlier in it, and no value
    /// twice: the numbering is topological and total, which is what the emit
    /// loop walks.
    #[test]
    fn arena_to_schedule_defines_every_operand_before_it_is_read() {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let sum = arena.push_binary(OpKind::Add, x, y);

        let schedule = native_schedule(&arena, sum, POINT);
        assert!(!schedule.is_empty(), "a collapse schedules something");
        let mut defined: alloc::vec::Vec<regalloc::ValueId> = alloc::vec::Vec::new();
        for def in &schedule {
            for operand in regalloc::structural_children(&def.op) {
                assert!(
                    defined.contains(&operand),
                    "{operand:?} is read by {:?} before it is defined",
                    def.value
                );
            }
            assert!(
                !defined.contains(&def.value),
                "{:?} is defined twice",
                def.value
            );
            defined.push(def.value);
        }
    }

    /// A node nothing reaches never becomes a schedule entry.
    #[test]
    fn arena_to_schedule_filters_unreachable() {
        let length = |garbage: bool| {
            let mut arena = ExprArena::new();
            let x = arena.push_var(0);
            if garbage {
                let _unreachable = arena.push_const(999.0);
            }
            let y = arena.push_var(1);
            let sum = arena.push_binary(OpKind::Add, x, y);
            native_schedule(&arena, sum, POINT).len()
        };
        assert_eq!(
            length(true),
            length(false),
            "unreachable garbage node should be filtered"
        );
    }

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
    /// This used to be `(X+Y)·(X−Y) + (X·Y)·(X+1)` under `max_regs(2)`, which
    /// keeps at most three values live: it spilled only because two registers
    /// is fewer than three, and a two-register pool is no longer a budget a
    /// caller can ask for (`RegisterFile::MIN_SCRATCH` — an instruction temp
    /// cannot spill). Ten live values outrun every backend's floor, so the
    /// subject here — what spilling *does* — no longer depends on how small
    /// the pool can be made.
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

        let result = EmitCtx::with_max_regs(4)
            .compile(&arena, root, POINT)
            .expect("arena DAG compile with spills failed");

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
            let file = native_register_file(EmitCtx::default());
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
        ) -> Option<(regalloc::Allocation<'_>, guards::IfGuard)> {
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
        /// guardable as scheduled, and both become guardable once
        /// [`guards::cluster_if_arms`] gathers each arm into one run.
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

        /// How many entries each `If` has under a guard, by schedule
        /// position, for a schedule built the way `compile` builds it.
        fn guarded_entries(a: &ExprArena, root: ExprId, cluster: bool) -> alloc::vec::Vec<usize> {
            let schedule = native_schedule(a, root, POINT);
            // Flat, not scoped: no fold is carved out, so none reads anything.
            let folds = guards::FoldReads::default();
            let schedule = if cluster {
                guards::cluster_if_arms(schedule, &folds)
            } else {
                schedule
            };
            analyze_if_guards(&schedule, &[], &folds)
                .iter()
                .map(|g| g.total_guarded_entries())
                .collect()
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

        /// Both levels of a nested `If` are guarded once the schedule is
        /// clustered, and neither was before — the reordering is the whole
        /// difference.
        #[test]
        fn clustering_guards_both_levels_of_a_nested_if() {
            let mut a = ExprArena::new();
            let (root, _outer, _inner) = nested_guarded_ifs(&mut a);

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
            let ctx = EmitCtx::with_max_regs(regalloc::RegisterFile::MIN_SCRATCH);
            let file = native_register_file(ctx.clone());
            let nest = allocate_nest(native_schedule(&a, root, POINT), &file);
            let (view, guard) = guarded_scope(&nest).expect("a guard formed above");
            assert!(
                view.placement(guard.mask_vid).spills(),
                "the mask stayed in a register, so the spilled-mask path this \
                 test exists for is never reached"
            );

            let result = ctx
                .compile(&a, root, POINT)
                .expect("spilled guarded If compile");

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

            let file =
                native_register_file(EmitCtx::with_max_regs(regalloc::RegisterFile::MIN_SCRATCH));
            let nest = allocate_nest(native_schedule(&a, root, POINT), &file);
            let mut scopes = core::iter::once(regalloc::Scope::Body)
                .chain((0..nest.fold_count()).map(regalloc::Scope::Fold));
            let (scope, guard) = scopes
                .find_map(|s| {
                    nest.scope(s)
                        .if_guards()
                        .iter()
                        .find(|g| g.is_guarded(IfArm::True))
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
            let arm = guard.true_range();
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
                view.placement(f.split).spills(),
                "the value under test stayed in a register, so nothing is split"
            );
            let kept = view
                .placement(f.split)
                .spans()
                .any(|s| matches!(s.at, regalloc::Where::Reg(_)) && s.from.index >= f.arm.0);
            assert!(
                kept,
                "the value was never brought back into a register inside the \
                 arm, so the confinement rule this test exists for is not exercised"
            );

            let result = EmitCtx::with_max_regs(regalloc::RegisterFile::MIN_SCRATCH)
                .compile(&f.arena, f.root, POINT)
                .expect("split-across-a-guard compile");
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
        /// the same `analyze_if_guards` the emitter branches on, which is
        /// what makes "exactly" a statement about one answer rather than two.
        #[test]
        fn a_kept_reload_inside_a_guarded_arm_ends_at_the_arm() {
            let f = split_across_a_guarded_arm();
            let spans: alloc::vec::Vec<regalloc::Span> =
                f.nest.scope(f.scope).placement(f.split).spans().collect();
            let kept = spans
                .iter()
                .position(|s| matches!(s.at, regalloc::Where::Reg(_)) && s.from.index >= f.arm.0)
                .expect("a register range begins inside the arm");
            assert!(
                spans[kept].from.index < f.arm.1,
                "the range begins outside the arm it was confined to"
            );
            let reverted = spans
                .get(kept + 1)
                .expect("a confined range is followed by the range it reverts to");
            assert_eq!(
                reverted.from.index, f.arm.1,
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
                !matches!(reverted.at, regalloc::Where::Reg(_)),
                "the range after a confined one must be in memory, not a \
                 register the skipped path never wrote: {:?}",
                reverted.at
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
        fn eval_point_with_arg(code: &executable::ExecutableCode, x: f32, y: f32, u: f32) -> f32 {
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
        fn sample_spills(a: &ExprArena, root: ExprId, ctx: EmitCtx) -> usize {
            let file = native_register_file(ctx);
            let nest = allocate_nest(native_schedule(a, root, POINT), &file);
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
                                && view.placement(d.value).spills()
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
                sample_spills(&a, root, EmitCtx::default()),
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

            let ctx = EmitCtx::with_max_regs(regalloc::RegisterFile::MIN_SCRATCH);
            let sched = ctx
                .clone()
                .compile(&a, root, POINT)
                .expect("scheduled compile");
            assert!(
                sample_spills(&a, root, ctx) > 0,
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

        /// `x + u·u`: the uniform's load and the product that depends on it
        /// alone are per-call work. Asserted on the nest — which scope holds
        /// them — not on timing.
        #[test]
        fn a_uniform_and_what_depends_on_it_alone_land_in_the_body() {
            let mut a = ExprArena::new();
            let u = a.declare_uniform(decl(3.0));
            let x = a.push_var(0);
            let uu = a.push_uniform(u);
            let sq = a.push_binary(OpKind::Mul, uu, uu);
            let root = a.push_binary(OpKind::Add, x, sq);

            let schedule = native_schedule(&a, root, batch());
            let variance = schedule_variance(&schedule);
            let scoped = scope_schedule(schedule, &variance);

            // The kernel's own uniform, told from the origin's two by the
            // block it is read from: the link's, at the context slot after
            // the (empty) buffer table, rather than the origin block after
            // that. The block's pointer is a `Context` def of its own,
            // loaded once per call in the body.
            let link_block = scoped
                .body
                .schedule
                .iter()
                .find(|d| matches!(d.op, ScheduledOp::Context(0)))
                .map(|d| d.value)
                .expect("the link's block pointer is loaded once per call");
            let kernel_uniform = |op: &ScheduledOp| matches!(op, ScheduledOp::Uniform(base, _) if *base == link_block);
            assert!(
                scoped.body.schedule.iter().any(|d| kernel_uniform(&d.op)),
                "the broadcast load is once per call"
            );
            let sq_vid = scoped
                .body
                .schedule
                .iter()
                .find(|d| match d.op {
                    ScheduledOp::Binary(OpKind::Mul, l, r) => l == r,
                    _ => false,
                })
                .map(|d| d.value)
                .expect("u·u is computed once per call");
            assert!(
                scoped.body.roots.contains(&sq_vid),
                "the product is parked for the scopes inside, not recomputed"
            );
            assert!(
                !scoped
                    .folds
                    .iter()
                    .flat_map(|f| f.schedule.iter())
                    .any(|d| kernel_uniform(&d.op)),
                "the folds read the parked product, never the block"
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

        /// The bytes, per backend, for `offset = 3, dst = 5` through the
        /// block in `rax` / `x9`. Checked against `llvm-mc --disassemble`
        /// (LLVM 18): `vbroadcastss 12(%rax), %ymm5` / `%zmm5`;
        /// `ldr s5, [x9, #12]`, `dup v5.4s, v5.s[0]`. The block's address is
        /// a pointer-class value the allocator placed, so no load of it
        /// appears here: that is the `Context` def's, once per call.
        #[test]
        fn every_backend_encodes_the_broadcast_load() {
            let mut avx2 = Vec::new();
            avx2::emit_uniform_load(&mut avx2, Reg(5), x86_64::ptr::RAX, 3).expect("fits");
            assert_eq!(avx2, [0xC4, 0xE2, 0x7D, 0x18, 0xA8, 0x0C, 0, 0, 0]);

            let mut avx512 = Vec::new();
            avx512::emit_uniform_load(&mut avx512, Reg(5), x86_64::ptr::RAX, 3).expect("fits");
            assert_eq!(avx512, [0x62, 0xF2, 0x7D, 0x48, 0x18, 0xA8, 0x0C, 0, 0, 0]);

            let mut neon = Vec::new();
            aarch64::emit_uniform_load(&mut neon, Reg(5), aarch64::ptr::X9, 3).expect("fits");
            let words: Vec<u32> = neon
                .chunks(4)
                .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                .collect();
            assert_eq!(words, [0xBD40_0D25, 0x4E04_04A5]);
        }

        /// The slot `UniformId` used to stop at, and one past it, in
        /// bytes: `65_539` is the offset above, shifted up by a full 16-bit
        /// range, so that the byte offset `262_156` (`0x0004_000C`) is
        /// `0x0C` wrapped to 16 bits — the load a narrower offset would
        /// have emitted for it, reading argument 3.
        const PAST_U16: u64 = 3 + (u16::MAX as u64 + 1);
        const PAST_U16_BYTES: u32 = 262_156;

        /// The same load with the offset past the old width: the x86 tiers
        /// carry the full `disp32` (same prefix and ModRM as the offset-3
        /// bytes above, only the displacement changes), and NEON, whose
        /// scaled immediate stops at 4095 elements, computes the address
        /// into IP0 in `add`-immediate steps and reads `[x16]` — the same
        /// path a deep spill frame takes.
        #[test]
        fn every_backend_encodes_a_load_past_the_old_u16_offset() {
            let mut avx2 = Vec::new();
            avx2::emit_uniform_load(&mut avx2, Reg(5), x86_64::ptr::RAX, PAST_U16).expect("fits");
            assert_eq!(avx2, [0xC4, 0xE2, 0x7D, 0x18, 0xA8, 0x0C, 0x00, 0x04, 0x00]);

            let mut avx512 = Vec::new();
            avx512::emit_uniform_load(&mut avx512, Reg(5), x86_64::ptr::RAX, PAST_U16)
                .expect("fits");
            assert_eq!(
                avx512,
                [0x62, 0xF2, 0x7D, 0x48, 0x18, 0xA8, 0x0C, 0x00, 0x04, 0x00]
            );

            let mut neon = Vec::new();
            aarch64::emit_uniform_load(&mut neon, Reg(5), aarch64::ptr::X9, PAST_U16)
                .expect("fits");
            let words: Vec<u32> = neon
                .chunks(4)
                .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                .collect();
            let step = aarch64::table::MAX_ADD_IMM;
            let full_adds = PAST_U16_BYTES / step;
            let remainder = PAST_U16_BYTES % step;
            let add = |src: u32, imm: u32| 0x9100_0000 | (imm << 10) | (src << 5) | 16;
            let mut want = alloc::vec![add(9, step)];
            want.extend(core::iter::repeat_n(add(16, step), full_adds as usize - 1));
            want.push(add(16, remainder));
            want.push(0xBD40_0000 | (16 << 5) | 5); // ldr s5, [x16]
            want.push(0x4E04_04A5); // dup v5.4s, v5.s[0]
            assert_eq!(words, want);
        }

        /// The width is the encoder's, and an offset past it is refused,
        /// never wrapped: a wrapped displacement would be a load of some
        /// other argument, with plausible pixels. x86's `disp32` is signed,
        /// so the last element it reaches is at `i32::MAX / 4`; NEON's
        /// [`aarch64::Mem`] holds a 32-bit byte offset.
        #[test]
        fn an_offset_past_the_displacement_is_refused_on_every_backend() {
            const LAST_DISP32: u64 = i32::MAX as u64 / 4;
            const LAST_NEON: u64 = u32::MAX as u64 / 4;
            let refused = |r: Result<(), CompileError>| {
                assert!(
                    matches!(r, Err(CompileError::BudgetExceeded(_))),
                    "expected a refusal, got {r:?}"
                );
            };

            let mut code = Vec::new();
            avx2::emit_uniform_load(&mut code, Reg(0), x86_64::ptr::RAX, LAST_DISP32)
                .expect("the last element a disp32 reaches");
            refused(avx2::emit_uniform_load(
                &mut code,
                Reg(0),
                x86_64::ptr::RAX,
                LAST_DISP32 + 1,
            ));
            avx512::emit_uniform_load(&mut code, Reg(0), x86_64::ptr::RAX, LAST_DISP32)
                .expect("the last element a disp32 reaches");
            refused(avx512::emit_uniform_load(
                &mut code,
                Reg(0),
                x86_64::ptr::RAX,
                LAST_DISP32 + 1,
            ));
            refused(aarch64::emit_uniform_load(
                &mut code,
                Reg(0),
                aarch64::ptr::X9,
                LAST_NEON + 1,
            ));
            // And nothing wrapped: a refused offset emits no bytes at all.
            refused(avx2::emit_uniform_load(
                &mut Vec::new(),
                Reg(0),
                x86_64::ptr::RAX,
                u64::MAX,
            ));
        }

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

            let ctx = EmitCtx::default();
            let for_backend = |file: regalloc::RegisterFile| {
                schedule_for(&a, root, POINT, file.vector_bytes / BYTES_PER_LANE)
            };
            let mut avx2b = avx2::driver::Avx2Backend::new(ctx.clone());
            let mut avx512b = avx512::driver::Avx512Backend::new(ctx.clone());
            let mut neon = aarch64::driver::Aarch64Backend::new(ctx);

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
                    compile_via_backend(for_backend(avx2b.register_file()), &mut avx2b)
                        .expect("AVX2")
                        .code,
                ),
                (
                    "AVX-512",
                    compile_via_backend(for_backend(avx512b.register_file()), &mut avx512b)
                        .expect("AVX-512")
                        .code,
                ),
            ] {
                assert!(
                    code.as_bytes().windows(disp.len()).any(|w| w == disp),
                    "{tier}: no vbroadcastss with disp32 {PAST_U16_BYTES:#x}"
                );
            }

            let neon_code = compile_via_backend(for_backend(neon.register_file()), &mut neon)
                .expect("NEON")
                .code;
            let words: Vec<u32> = neon_code
                .as_bytes()
                .chunks(4)
                .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                .collect();
            let step = aarch64::table::MAX_ADD_IMM;
            let add_ip0 = 0x9100_0000 | (step << 10) | (16 << 5) | 16;
            let ldr_s_ip0 = |w: u32| (w & !0x1F) == 0xBD40_0000 | (16 << 5);
            assert!(
                words.contains(&add_ip0) && words.iter().copied().any(ldr_s_ip0),
                "NEON: no IP0-addressed load of the argument"
            );
        }

        /// The `Context` def's own instruction, per backend: `mov r9, [rdi +
        /// 16]` (`REX.WR 8B /r`) and `ldr x3, [x0, #16]` for context slot 2.
        #[test]
        fn every_backend_reads_a_context_pointer_once() {
            let mut x86 = Vec::new();
            AsmProgram::from([x86_64::MovLoadPtr {
                dst: PtrReg(9),
                base: x86_64::ptr::RDI,
                disp: 2 * x86_64::PTR_BYTES,
            }
            .encode()])
            .assemble(&mut x86);
            assert_eq!(x86, [0x4C, 0x8B, 0x8F, 0x10, 0, 0, 0]);

            let mut neon = Vec::new();
            AsmProgram::from([aarch64::Inst::ldr_x(
                PtrReg(3),
                aarch64::Mem {
                    base: aarch64::ptr::X0,
                    offset: 16,
                },
            )])
            .assemble(&mut neon);
            assert_eq!(neon, 0xF940_0803u32.to_le_bytes());
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

    /// A gather whose address the lane binder does not reach is one scalar
    /// load broadcast — `ScheduledOp::Broadcast`, split from `Gather` in
    /// `arena_to_schedule` by the index's variance.
    mod broadcast {
        use super::*;

        fn count(schedule: &[regalloc::Def], pred: fn(&ScheduledOp) -> bool) -> usize {
            schedule.iter().filter(|d| pred(&d.op)).count()
        }

        /// The split: a read addressed by the row alone is a `Broadcast`,
        /// one addressed by the column — which the lane binder reaches — a
        /// `Gather`. The same arena one leaf apart.
        #[test]
        fn the_lane_bit_decides_broadcast_or_gather() {
            for (axis, want) in [(1u8, (1, 0)), (0u8, (0, 1))] {
                let mut a = ExprArena::new();
                let buf = table(&mut a, 8);
                let idx = a.push_var(axis);
                let leaf = a.push_buffer(buf);
                let root = a.push_binary(OpKind::RawGather, leaf, idx);
                let schedule = native_schedule(&a, root, batch());
                let got = (
                    count(&schedule, |op| matches!(op, ScheduledOp::Broadcast(..))),
                    count(&schedule, |op| matches!(op, ScheduledOp::Gather(..))),
                );
                assert_eq!(got, want, "(broadcasts, gathers) for Var({axis})");
            }
        }

        /// A schedule with no lane fold — an arena `collapse` never
        /// wrapped, which the scheduler still accepts — knows nothing to be
        /// lane-uniform, so every read stays a gather, a constant address
        /// included.
        #[test]
        fn without_a_lane_fold_every_read_is_a_gather() {
            let mut a = ExprArena::new();
            let buf = table(&mut a, 8);
            let idx = a.push_const(3.0);
            let leaf = a.push_buffer(buf);
            let root = a.push_binary(OpKind::RawGather, leaf, idx);
            let schedule = arena_to_schedule(&a, root, RAW_ORIGIN);
            assert_eq!(
                count(&schedule, |op| matches!(op, ScheduledOp::Gather(..))),
                1
            );
            assert_eq!(
                count(&schedule, |op| matches!(op, ScheduledOp::Broadcast(..))),
                0
            );
        }

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

        /// The bytes, per backend, for `dst = 5, idx = 6` through the base in
        /// `rax` and the index in `rcx` — `vcvttss2si rcx, xmm6`,
        /// `vbroadcastss ymm5/zmm5, [rax + rcx*4]` — and through `x9`
        /// and `x10`: `fcvtzs x10, s6`, `ldr s5, [x9, w10, uxtw #2]`, `dup
        /// v5.4s, v5.s[0]`. The x86 encodings were checked against
        /// `objdump -M intel`. The base's own load is the `Context` def's,
        /// once per call, not this instruction's.
        #[test]
        fn every_backend_encodes_the_lane_uniform_read() {
            let gprs = x86_64::BroadcastGprs {
                base: x86_64::ptr::RAX,
                index: x86_64::gpr::RCX,
            };

            let mut avx2 = Vec::new();
            avx2::emit_broadcast_load(&mut avx2, Reg(5), Reg(6), gprs);
            assert_eq!(&avx2[..5], &[0xC4, 0xE1, 0xFE, 0x2C, 0xCE]);
            assert_eq!(&avx2[5..], &[0xC4, 0xE2, 0x7D, 0x18, 0x2C, 0x88]);

            let mut avx512 = Vec::new();
            avx512::emit_broadcast_load(&mut avx512, Reg(5), Reg(6), gprs);
            assert_eq!(&avx512[..6], &[0x62, 0xF1, 0xFE, 0x48, 0x2C, 0xCE]);
            assert_eq!(&avx512[6..], &[0x62, 0xF2, 0x7D, 0x48, 0x18, 0x2C, 0x88]);

            let mut neon = Vec::new();
            aarch64::emit_broadcast_load(
                &mut neon,
                Reg(5),
                Reg(6),
                aarch64::BroadcastGprs {
                    base: aarch64::ptr::X9,
                    index: aarch64::gpr::X10,
                },
            );
            let words: Vec<u32> = neon
                .chunks(4)
                .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                .collect();
            assert_eq!(words, [0x9E38_00CA, 0xBC6A_5925, 0x4E04_04A5]);
        }

        /// A base in a pointer register past the low eight, and one past the
        /// low eight of the index: `vbroadcastss ymm5, [r9 + r11*4]` sets
        /// `X` and `B` in the prefix (clear, inverted), per tier, after a
        /// `vcvttss2si r11, xmm6` whose VEX.R carries the GPR's high bit.
        #[test]
        fn the_broadcast_addresses_high_pointer_registers() {
            let gprs = x86_64::BroadcastGprs {
                base: PtrReg(9),
                index: Gpr(11),
            };
            let mut avx2 = Vec::new();
            avx2::emit_broadcast_load(&mut avx2, Reg(5), Reg(6), gprs);
            assert_eq!(&avx2[..5], &[0xC4, 0x61, 0xFE, 0x2C, 0xDE]);
            assert_eq!(&avx2[5..], &[0xC4, 0x82, 0x7D, 0x18, 0x2C, 0x99]);

            let mut avx512 = Vec::new();
            avx512::emit_broadcast_load(&mut avx512, Reg(5), Reg(6), gprs);
            assert_eq!(&avx512[6..], &[0x62, 0x92, 0x7D, 0x48, 0x18, 0x2C, 0x99]);
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
            let file = native_register_file(EmitCtx::default());
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
            let res = compile_via_backend(
                schedule,
                &mut avx2::driver::Avx2Backend::new(EmitCtx::default()),
            )
            .expect("compile");
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
                scratch: regalloc::Scratch::for_test_with_classes(
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
            // four: one fused plus one per `DecomposedMulAdd` spelling.
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
    // `Vec<u8>`, so all three backends are checked from whichever host runs
    // the tests — including the two (aarch64, AVX-512 decomposed) that no
    // execution test on any single host reaches.
    // =========================================================================
    mod muladd_encoding {
        use super::*;

        const DST: Reg = Reg(4);
        const SRC_A: Reg = Reg(5);
        const SRC_B: Reg = Reg(6);
        const ADDEND: Reg = Reg(7);

        /// A bare plan: no reloads, no setup mov, no store, no temps — just
        /// the op, so the bytes below are the op's encoding and nothing
        /// else. The empty scratch is the assertion that no backend starts
        /// asking for one on a `MulAdd` unnoticed.
        fn plan(op: ResolvedOp) -> InstructionPlan {
            InstructionPlan {
                reloads: alloc::vec::Vec::new(),
                op,
                setup_mov: None,
                scratch: regalloc::Scratch::for_test(None, [None, None]),
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

        fn decomposed(c_deferred: Option<DeferredReload>) -> ResolvedOp {
            ResolvedOp::DecomposedMulAdd {
                dst: DST,
                a: SRC_A,
                b: SRC_B,
                c: ADDEND,
                c_deferred,
            }
        }

        /// `dst += a * b` in one instruction, one rounding, on every target:
        /// each of the three has an FMA.
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
                // `dst = a*b` is everything before the final add; on VEX the
                // add is 5 bytes, on EVEX 6, on NEON 4 — so split by the
                // tail rather than by a per-backend length.
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
                    "AVX2" => 5,
                    "AVX-512" => 6,
                    "aarch64" => 4,
                    other => panic!("unknown backend {other}"),
                }
            }

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
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let z = a.push_binary(OpKind::Add, y, x);
            let root = a.push_ternary(OpKind::MulAdd, x, y, z);

            let mut backend = avx2::driver::Avx2Backend::new(EmitCtx::default());
            let lanes = backend.register_file().vector_bytes / BYTES_PER_LANE;
            let result = compile_via_backend(schedule_for(&a, root, POINT, lanes), &mut backend)
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

    /// G2's end-to-end gate: a hand-built `Guard` reaching the JIT emitter
    /// compiles, and both of its arms run and produce the right value —
    /// chosen at *runtime*, by a per-call uniform, so one compiled kernel
    /// exercises both "every lane takes the True arm" and "every lane takes
    /// the False arm" (docs/plans/2026-09-12-emit-should-just-emit.md).
    mod guard_arms {
        use super::*;
        use pixelflow_ir::kernel::Uniform;
        use pixelflow_ir::passes::lattice;
        use pixelflow_ir::{Kernel, KernelStore};

        /// A binder slot the lattice's row/col/lane folds never claim — far
        /// past the three `collapse` picks — used only as a placeholder
        /// `Var` while `collapse`/`pack` build the `Write` around it.
        ///
        /// Why a placeholder at all: `lattice::collapse` refuses any
        /// *reachable* `Guard` outright (its own doc — coordinate warping
        /// does not reach into a guard's arms yet, this plan's §8), so a
        /// `Guard` cannot be the kernel `collapse` wraps. `Write` is
        /// `pub(crate)` in `pixelflow-ir` on purpose ("Constructible only by
        /// the legalize passes", CLAUDE.md's own citation of it), so nothing
        /// outside that crate may build one directly either. What *is*
        /// public is `substitute_vars_with` — the same primitive
        /// `Kernel::at` warps coordinates with — so this builds the `Write`
        /// around an inert marker first and substitutes the `Guard` in
        /// afterward, the one route to a guarded `Write` this stage has.
        const MARKER: u8 = 40;

        /// Compile a `Guard(flag == 1.0, on, off)` at [`POINT`], and return a
        /// closure that runs it for a given `flag` value.
        fn compile_guard(on: Kernel, off: Kernel) -> impl Fn(f32) -> f32 {
            let on_key = KernelStore::intern(&on);
            let off_key = KernelStore::intern(&off);

            // `Uniform::new` is the properly-declared route to a fresh
            // uniform slot — `ExprArena::uniform_slot_for` is `pub(crate)`,
            // and `push_uniform` asserts its `UniformId` is one the table
            // already knows, so a raw slot number picked by hand is refused
            // rather than silently aliasing `collapse`'s own two (x0, y0).
            // `Uniform::kernel()` builds its own tiny arena declaring it;
            // cloning that arena inherits the declaration and gives a
            // legitimately mutable `ExprArena` to keep building on.
            let flag = Uniform::new(0.0);
            let flag_kernel = flag.kernel();
            let (flag_arena, flag_root) = flag_kernel.parts();
            let mut arena = flag_arena.clone();

            // The mask: `flag == 1.0` — `Variance::CONST`, so every lane
            // agrees on it by construction, which is what makes both a
            // "true" and a "false" run reachable from the same compiled
            // kernel.
            let one = arena.push_const(1.0);
            let mask = arena.push_binary(OpKind::Eq, flag_root, one);
            let guard = arena.push_guard(mask, on_key, off_key);

            let marker = arena.push_var(MARKER);
            let domain = lattice::Domain {
                shape: POINT,
                origin: origin(),
            };
            let write_root = lattice::collapse(&mut arena, marker, domain);
            let packed_root = lattice::pack(&mut arena, write_root, lanes() as u32);
            let root = arena.substitute_vars_with(packed_root, &[(MARKER, guard)]);

            let ids = origin_slots(&arena);
            let schedule = arena_to_schedule(&arena, root, Some(ids));
            let code = compile_native(schedule, EmitCtx::default())
                .expect("a hand-built Guard should compile")
                .code;

            move |flag_value: f32| -> f32 {
                let uniforms = [flag_value];
                let origin_vals = [0.0f32, 0.0f32];
                let mut out = [f32::NAN; 1];
                let ctx: [*const f32; 2] = [uniforms.as_ptr(), origin_vals.as_ptr()];
                // SAFETY: this arena declares no buffers and one uniform
                // (`flag`, at slot 0), so `ctx[0]` is a one-`f32` uniform
                // block and `ctx[1]` the origin block; `out` holds the one
                // sample a `POINT`-shaped lattice writes.
                unsafe {
                    code.call(ctx.as_ptr(), out.as_mut_ptr(), 1);
                }
                out[0]
            }
        }

        /// The mask uniformly true: every lane (there is one, at `POINT`)
        /// takes the `on` arm.
        #[test]
        fn a_uniformly_true_mask_takes_the_on_arm() {
            let run = compile_guard(Kernel::constant(6.0), Kernel::constant(9.0));
            assert_eq!(run(1.0), 6.0);
        }

        /// The mask uniformly false: every lane takes the `off` arm — the
        /// same compiled kernel as above, a different runtime value, which
        /// is the whole point of a *runtime* branch (as opposed to a
        /// compile-time choice extraction would have made for a constant
        /// mask).
        #[test]
        fn a_uniformly_false_mask_takes_the_off_arm() {
            let run = compile_guard(Kernel::constant(6.0), Kernel::constant(9.0));
            assert_eq!(run(0.0), 9.0);
        }

        /// Each arm doing real (if small) arithmetic, not just naming a
        /// `Const` — so the test exercises an arm's own scope actually
        /// computing something, not merely handing back a leaf.
        #[test]
        fn each_arm_computes_its_own_arithmetic() {
            let on = Kernel::constant(2.0).mul(&Kernel::constant(3.0));
            let off = Kernel::constant(10.0).sub(&Kernel::constant(1.0));
            let run = compile_guard(on, off);
            assert_eq!(run(1.0), 6.0, "on arm: 2.0 * 3.0");
            assert_eq!(run(0.0), 9.0, "off arm: 10.0 - 1.0");
        }
    }
}
