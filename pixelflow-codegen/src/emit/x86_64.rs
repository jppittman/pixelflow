//! x86-64 leaf encoders: what the AVX2 and AVX-512 tiers share below the
//! vector width.
//!
//! Every function here emits raw machine code for one instruction (or a
//! small fixed sequence) of the *architecture*: the general-register
//! instructions the loop nest and the store's address arithmetic are made
//! of, the branches, the memory-operand tail (`Mem`, `Disp`) every vector
//! encoder's ModRM/SIB is built from, the pointer class's loads and stores,
//! and the constant pool with its anchor. Nothing here names a vector
//! width: the `ymm`/`zmm` encodings live in `avx2.rs` and `avx512.rs`, each
//! with its own `IsaBackend` driver, and the 128-bit tier that used to sit
//! in this file is gone (docs/plans/2026-09-22-the-isa-is-decided-at-startup.md §7).

use super::{
    AsmInsn, AsmProgram, Assembly, Binding, CONST_POOL, CONST_POOL_ALIGN, EncodedInst, Gpr, Label,
    LabelRef, Loc, PtrReg, Reg, WritePlan, regalloc,
};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

// =============================================================================
// Constants
// =============================================================================

/// The register that holds the constant pool's address for the whole kernel:
/// `r8`, SysV's fifth integer argument, which the collapse ABI does not pass,
/// and caller-saved, so nothing preserves it. The x86 counterpart of
/// aarch64's `X17`. A pointer register because that is what it holds; each
/// tier's register file (`avx2::driver::AVX2_FILE`, `avx512::driver::AVX512_FILE`)
/// keeps its GPR roles clear of it the way they stay clear of the three
/// arguments.
pub const POOL_BASE: PtrReg = PtrReg(8);

/// A kernel's constants, deduplicated, laid out after its `ret`.
///
/// One per emitted function, shared by every scope of the nest: a compile
/// emits every scope through one backend, so the offsets an outer scope baked
/// in stay valid as inner scopes append. Entry `k` is at `[POOL_BASE + 4k]`;
/// [`anchor`] materializes `POOL_BASE` once, after the frame, and
/// [`ConstPool::finish`] writes the entries after the return, binding the
/// label the anchor names.
///
/// Four bytes per constant, not a register's worth: every tier loads a
/// constant with `vbroadcastss` from a 32-bit source, so the splat is the
/// instruction's business and the pool holds the scalar. This replaced two
/// idioms that were each two instructions per read: a `jmp` over sixteen
/// bytes of inline data followed by a RIP-relative load on the 128-bit tier,
/// and a store to the red zone followed by a broadcast from it on the wide
/// ones — a store-forward on the critical path of every constant read.
#[derive(Default)]
pub struct ConstPool {
    /// The entries, in pool order.
    entries: Vec<u32>,
    /// Each entry's byte offset, by its bits.
    index: BTreeMap<u32, u32>,
}

impl ConstPool {
    /// The memory operand of the constant with these bits: `[POOL_BASE +
    /// offset]`, entering it into the pool on first use.
    ///
    /// Always a `disp32`, never a `disp8`: EVEX scales a `disp8` by the
    /// operand's tuple size, VEX does not, and one form for both is worth
    /// three bytes per load.
    pub fn operand(&mut self, bits: u32) -> Mem<Imm32> {
        let offset = *self.index.entry(bits).or_insert_with(|| {
            let offset = (self.entries.len() * 4) as u32;
            self.entries.push(bits);
            offset
        });
        Mem {
            base: POOL_BASE,
            disp: Imm32(offset as i32),
        }
    }

    /// Append the pool after the return and bind [`CONST_POOL`] where it
    /// lands.
    ///
    /// The label is written whether or not there is anything to append: the
    /// anchor names it unconditionally, and `Assembly::finish` panics on a
    /// name nobody wrote.
    pub fn finish(&self, asm: &mut Assembly) {
        if !self.entries.is_empty() {
            while !asm.code.len().is_multiple_of(CONST_POOL_ALIGN) {
                asm.code.push(0);
            }
        }
        asm.bind(CONST_POOL);
        for &bits in &self.entries {
            asm.code.extend_from_slice(&bits.to_le_bytes());
        }
    }
}

/// `lea dst, [rip + target]` — a position's address, in one instruction.
///
/// `REX.W 8D /r` with the RIP-relative ModRM, the displacement patched once
/// the label lands. What every x86 tier's [`anchor`] is made of.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LeaRip {
    /// Where the address is materialized.
    pub dst: PtrReg,
    /// The position it is the address of.
    pub target: Label,
}

/// Bytes from a `LeaRip`'s start to its displacement field: REX, opcode,
/// ModRM.
const LEA_RIP_DISP: usize = 3;

impl AsmInsn for LeaRip {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        code.push(0x48 | (((self.dst.0 >> 3) & 1) << 2));
        code.push(0x8D);
        code.push(((self.dst.0 & 7) << 3) | RM_RIP_AT_MOD0);
        code.extend_from_slice(&[0, 0, 0, 0]);
    }

    #[inline]
    fn label_ref(self) -> Option<LabelRef> {
        Some(LabelRef {
            label: self.target,
            patch: |code, at, target| patch_rel32(code, at + LEA_RIP_DISP, target),
        })
    }
}

/// Every x86 tier's anchor: `POOL_BASE = &pool`, once, after the frame.
pub fn anchor(asm: &mut Assembly) {
    asm.push(LeaRip {
        dst: POOL_BASE,
        target: Label::new(CONST_POOL),
    });
}

// =============================================================================
// The pointer class: an address between a general register and memory
// =============================================================================

/// 64-bit pointer load: `mov dst, [base + disp32]`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MovLoadPtr {
    pub dst: PtrReg,
    pub base: PtrReg,
    pub disp: i32,
}

impl MovLoadPtr {
    /// `REX.W 8B /r` with a `disp32` memory operand: any of the sixteen
    /// GPRs on either side, `rsp`'s SIB and `rbp`'s displacement form
    /// included — the tail is [`mem_operand_into`]'s, the same as every
    /// other memory operand here.
    #[must_use]
    #[inline]
    pub fn encode(self) -> EncodedInst {
        let mut inst = EncodedInst::new();
        inst.push(rex_w(self.dst.as_gpr(), self.base.as_gpr()));
        inst.push(0x8B);
        mem_operand_into(
            &mut inst,
            self.dst.0,
            Mem {
                base: self.base,
                disp: Imm32(self.disp),
            },
        );
        inst
    }
}

impl AsmInsn for MovLoadPtr {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        self.encode().emit_into(code);
    }
}

/// `mov [base + disp32], src` — `REX.W 89 /r`: an address to a frame slot,
/// the pointer class's spill store. [`MovLoadPtr`]'s mirror.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MovStorePtr {
    pub src: PtrReg,
    pub base: PtrReg,
    pub disp: i32,
}

impl MovStorePtr {
    #[must_use]
    #[inline]
    pub fn encode(self) -> EncodedInst {
        let mut inst = EncodedInst::new();
        inst.push(rex_w(self.src.as_gpr(), self.base.as_gpr()));
        inst.push(0x89);
        mem_operand_into(
            &mut inst,
            self.src.0,
            Mem {
                base: self.base,
                disp: Imm32(self.disp),
            },
        );
        inst
    }
}

impl AsmInsn for MovStorePtr {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        self.encode().emit_into(code);
    }
}

/// Bytes per pointer in the context array.
pub const PTR_BYTES: i32 = 8;

/// The GPRs a broadcast load runs through: the buffer's address, wherever
/// the allocator keeps that pointer value, and the one index, this
/// instruction's `RegisterFile::gpr_scratch` reservation. Each tier's
/// `emit_broadcast_load` truncates lane 0 into the index and reads the
/// element once, `vbroadcastss [base + index*4]`, into every lane.
#[derive(Clone, Copy)]
pub struct BroadcastGprs {
    /// The buffer base pointer.
    pub base: PtrReg,
    /// Receives the truncated index.
    pub index: Gpr,
}

// =============================================================================
// Branches — for the shared driver's Select short-circuit guards.
// =============================================================================

/// TEST eax, eax (sets ZF iff eax == 0).
pub fn emit_test_eax(code: &mut Vec<u8>) {
    code.extend_from_slice(&[0x85, 0xC0]);
}

#[cfg(test)]
mod label_tests {
    use super::{Cond, Inst, Jcc, Jmp};
    use crate::emit::{AsmProgram, Item, Label};
    use alloc::vec::Vec;

    fn assemble(items: impl IntoIterator<Item = Item<Inst>>) -> Vec<u8> {
        let mut code = Vec::new();
        AsmProgram::new(items).assemble(&mut code);
        code
    }

    /// The one thing a label does that a fixup token could not: name a
    /// position that does not exist yet.
    #[test]
    fn a_forward_branch_names_a_position_bound_later() {
        let end = Label::new("end");
        let code = assemble([
            Item::Inst(Inst::from(Jmp { target: end })),
            Item::Inst(Inst::ret()),
            Item::Label(end),
        ]);

        // `jmp rel32` is five bytes; `ret` is one; the label lands at 6. The
        // displacement is measured from the end of the branch, so it is 1.
        assert_eq!(code.len(), 6);
        assert_eq!(code[0], 0xE9);
        assert_eq!(i32::from_le_bytes([code[1], code[2], code[3], code[4]]), 1);
    }

    /// A back edge — the shape a loop is made of, and the reason the
    /// resolution pass is separate from the layout pass.
    #[test]
    fn a_back_edge_resolves_to_a_negative_displacement() {
        let top = Label::new("end");
        let code = assemble([
            Item::Label(top),
            Item::Inst(Inst::ret()),
            Item::Inst(Inst::from(Jmp { target: top })),
        ]);

        // `ret` at 0, `jmp` at 1..6. Target 0, origin 6, so the displacement
        // is -6 — and getting this sign backwards is the classic way a loop
        // becomes an infinite one.
        assert_eq!(code.len(), 6);
        assert_eq!(i32::from_le_bytes([code[2], code[3], code[4], code[5]]), -6);
    }

    /// A label may name a position no instruction occupies — the end of the
    /// program. That is why a label is an item of its own rather than a field
    /// on an instruction: there is nothing here to hang it on.
    #[test]
    fn a_label_can_end_the_program() {
        let end = Label::new("end");
        let code = assemble([
            Item::Inst(Inst::from(Jmp { target: end })),
            Item::Label(end),
        ]);
        assert_eq!(code.len(), 5);
        assert_eq!(i32::from_le_bytes([code[1], code[2], code[3], code[4]]), 0);
    }

    /// And two labels may name the same position, for the same reason.
    #[test]
    fn two_labels_can_share_a_position() {
        let (a, b) = (Label::new("end"), Label::new("other"));
        let code = assemble([
            Item::Inst(Inst::from(Jcc {
                condition: Cond::E,
                target: a,
            })),
            Item::Inst(Inst::from(Jmp { target: b })),
            Item::Label(a),
            Item::Label(b),
        ]);
        // `je` is 6 bytes, `jmp` 5, both landing at 11.
        assert_eq!(code.len(), 11);
        assert_eq!(i32::from_le_bytes([code[2], code[3], code[4], code[5]]), 5);
        assert_eq!(i32::from_le_bytes([code[7], code[8], code[9], code[10]]), 0);
    }

    /// Offsets are relative to the program, not the buffer, so a program can
    /// be assembled after bytes that are already there.
    #[test]
    fn a_program_is_position_independent() {
        let end = Label::new("end");
        let items = [
            Item::Inst(Inst::from(Jmp { target: end })),
            Item::Label(end),
        ];

        let mut offset = alloc::vec![0x90u8; 7];
        AsmProgram::new(items).assemble(&mut offset);
        assert_eq!(&assemble(items)[..], &offset[7..]);
    }

    #[test]
    #[should_panic(expected = "never written")]
    fn an_unbound_label_is_a_bug_and_not_a_jump_to_itself() {
        let _ = assemble([Item::Inst(
            Jmp {
                target: Label::new("end"),
            }
            .into(),
        )]);
    }

    #[test]
    #[should_panic(expected = "written twice")]
    fn a_label_bound_twice_is_a_bug() {
        let twice = Label::new("end");
        let _ = assemble([Item::Label(twice), Item::Label(twice)]);
    }
}

// =============================================================================
// General-purpose registers
// =============================================================================

// The x86-64 general register file.
//
// A distinct type from [`Reg`], which names the *vector* file. They are
// different register files that happen to be numbered the same way, so
// `Gpr(10)` is `r10` and `Reg(10)` is `xmm10`, and nothing can silently pass
// one where the other belongs. Before this existed the general file had no
// type at all: it appeared as bare `u8` in a few encoder signatures and as
// raw opcode bytes everywhere else.
//
// A SIMD language barely touches these — loop counters, pointers, and the
// scalar half of a broadcast — which is why the vocabulary below is nine
// instructions rather than an assembler.

/// The general registers the emitted kernels name. Which is *for* what is
/// the register file's to say (`avx2::driver::AVX2_FILE`,
/// `avx512::driver::AVX512_FILE`), not a constant's.
pub mod gpr {
    use super::Gpr;

    /// Scratch / `movmskps` destination.
    pub const RAX: Gpr = Gpr(0);
    /// Scratch aliases for RAX.
    pub const AX: Gpr = RAX;
    pub const RX: Gpr = RAX;
    /// Scratch; SysV's 4th integer argument, which the kernel ABI does not use.
    pub const RCX: Gpr = Gpr(1);
    /// 3rd integer argument: the pitch.
    pub const RDX: Gpr = Gpr(2);
    /// 2nd integer argument: the output plane.
    pub const RSI: Gpr = Gpr(6);
    /// 1st integer argument: the context pointer — the array of bound buffer
    /// bases, then the uniform and origin blocks. Read-only for the whole
    /// kernel.
    pub const RDI: Gpr = Gpr(7);
    /// The stack pointer.
    pub const RSP: Gpr = Gpr(4);
    /// The extended registers, named for the encoders' tests.
    pub const R8: Gpr = Gpr(8);
    pub const R9: Gpr = Gpr(9);
    pub const R10: Gpr = Gpr(10);
    pub const R11: Gpr = Gpr(11);
}

/// SysV argument and scratch pointer registers.
pub mod ptr {
    use super::PtrReg;

    /// Scratch / base pointer register (`rax`).
    pub const RAX: PtrReg = PtrReg(0);
    /// Output pointer register (`rsi`).
    pub const RSI: PtrReg = PtrReg(6);
    /// Context pointer register (`rdi`).
    pub const RDI: PtrReg = PtrReg(7);
    /// Stack pointer register (`rsp`).
    pub const RSP: PtrReg = PtrReg(4);
}

/// First-class x86-64 instruction.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Inst {
    Mov { dst: Gpr, src: Gpr },
    Xor { dst: Gpr, src: Gpr },
    Cmp { lhs: Gpr, rhs: Gpr },
    Inc { dst: Gpr },
    Add { dst: Gpr, src: Gpr },
    AddImm8 { dst: Gpr, imm: Imm8 },
    AddImm32 { dst: Gpr, imm: Imm32 },
    SubImm32 { dst: Gpr, imm: Imm32 },
    Ret,
    MovLoadPtr(MovLoadPtr),
    Encoded(EncodedInst),
    Jmp(Jmp),
    Jcc(Jcc),
}

impl Inst {
    #[must_use]
    #[inline(always)]
    pub const fn mov(dst: Gpr, src: Gpr) -> Self {
        Self::Mov { dst, src }
    }
    #[must_use]
    #[inline(always)]
    pub const fn xor(dst: Gpr, src: Gpr) -> Self {
        Self::Xor { dst, src }
    }
    #[must_use]
    #[inline(always)]
    pub const fn ret() -> Self {
        Self::Ret
    }
}

impl From<EncodedInst> for Inst {
    #[inline(always)]
    fn from(e: EncodedInst) -> Self {
        Inst::Encoded(e)
    }
}

impl From<Jmp> for Inst {
    #[inline(always)]
    fn from(j: Jmp) -> Self {
        Inst::Jmp(j)
    }
}

impl From<Jcc> for Inst {
    #[inline(always)]
    fn from(j: Jcc) -> Self {
        Inst::Jcc(j)
    }
}

impl From<MovLoadPtr> for Inst {
    #[inline(always)]
    fn from(m: MovLoadPtr) -> Self {
        Inst::MovLoadPtr(m)
    }
}

impl AsmInsn for Inst {
    #[inline]
    fn label_ref(self) -> Option<LabelRef> {
        // A branch is an ordinary instruction whose operand happens to be a
        // name: it emits a placeholder displacement above, and this says which
        // label the assembler should measure it against.
        match self {
            Inst::Jmp(j) => j.label_ref(),
            Inst::Jcc(j) => j.label_ref(),
            _ => None,
        }
    }

    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        match self {
            Inst::Mov { dst, src } => mov(code, dst, src),
            Inst::Xor { dst, src } => xor(code, dst, src),
            Inst::Cmp { lhs, rhs } => cmp(code, lhs, rhs),
            Inst::Inc { dst } => inc(code, dst),
            Inst::Add { dst, src } => add(code, dst, src),
            Inst::AddImm8 { dst, imm } => add(code, dst, imm),
            Inst::AddImm32 { dst, imm } => add(code, dst, imm),
            Inst::SubImm32 { dst, imm } => sub(code, dst, imm),
            Inst::Ret => ret(code),
            Inst::Jmp(j) => j.emit_into(code),
            Inst::Jcc(j) => j.emit_into(code),
            Inst::MovLoadPtr(m) => m.emit_into(code),
            Inst::Encoded(e) => e.emit_into(code),
        }
    }
}

/// A sign-extended 8-bit immediate.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Imm8(pub i8);

/// `REX.W` plus the extension bits for a two-register form.
///
/// `R` extends the ModRM.reg field (the source here), `B` extends ModRM.rm
/// (the destination).
#[inline(always)]
const fn rex_w(reg: Gpr, rm: Gpr) -> u8 {
    0x48 | (((reg.0 >> 3) & 1) << 2) | ((rm.0 >> 3) & 1)
}

/// ModRM for the register-direct form: `mod = 11`.
#[inline(always)]
const fn modrm_rr(reg: u8, rm: Gpr) -> u8 {
    0xC0 | ((reg & 7) << 3) | (rm.0 & 7)
}

/// Emit one `REX.W opcode /r` instruction with both operands in registers.
#[inline(always)]
fn rr(code: &mut Vec<u8>, opcode: u8, dst: Gpr, src: Gpr) {
    code.extend_from_slice(&[rex_w(src, dst), opcode, modrm_rr(src.0, dst)]);
}

/// `mov dst, src`
#[inline(always)]
pub fn mov(code: &mut Vec<u8>, dst: Gpr, src: Gpr) {
    rr(code, 0x89, dst, src);
}

/// `xor dst, src` — the idiomatic zeroing form when `dst == src`.
#[inline(always)]
pub fn xor(code: &mut Vec<u8>, dst: Gpr, src: Gpr) {
    rr(code, 0x31, dst, src);
}

/// `cmp lhs, rhs` — sets the flags a following [`jae`] reads.
#[inline(always)]
pub fn cmp(code: &mut Vec<u8>, lhs: Gpr, rhs: Gpr) {
    rr(code, 0x39, lhs, rhs);
}

/// `inc dst`
#[inline(always)]
pub fn inc(code: &mut Vec<u8>, dst: Gpr) {
    code.extend_from_slice(&[rex_w(Gpr(0), dst), 0xFF, modrm_rr(0, dst)]);
}

/// What an [`add`] can add: another register, or a small immediate.
///
/// The operand's *type* picks the encoding, so callers write `add(c, RSI, R8)`
/// and `add(c, RSI, Imm8(16))` rather than choosing between differently-named
/// functions — which would put the operand kinds back in the name.
pub trait AddSrc {
    /// Emit `add dst, self`.
    fn add_into(self, code: &mut Vec<u8>, dst: Gpr);
}

impl AddSrc for Gpr {
    #[inline(always)]
    fn add_into(self, code: &mut Vec<u8>, dst: Gpr) {
        rr(code, 0x01, dst, self);
    }
}

impl AddSrc for Imm8 {
    #[inline(always)]
    fn add_into(self, code: &mut Vec<u8>, dst: Gpr) {
        code.extend_from_slice(&[rex_w(Gpr(0), dst), 0x83, modrm_rr(0, dst), self.0 as u8]);
    }
}

/// `add dst, src`
#[inline(always)]
pub fn add(code: &mut Vec<u8>, dst: Gpr, src: impl AddSrc) {
    src.add_into(code, dst);
}

/// A 32-bit immediate.
///
/// Distinct from [`Imm8`] because x86 gives them different opcodes — `81 /n id`
/// versus the sign-extended `83 /n ib`. The caller writes `add(c, RSP,
/// Imm32(n))` or `add(c, RSI, Imm8(n))` and the operand type picks; nothing
/// upstream has to know which opcode that implies.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Imm32(pub i32);

/// `REX.W 81 /ext id` — the immediate group with a 32-bit operand.
#[inline(always)]
fn ri32(code: &mut Vec<u8>, ext: u8, dst: Gpr, imm: i32) {
    code.extend_from_slice(&[rex_w(Gpr(0), dst), 0x81, modrm_rr(ext, dst)]);
    code.extend_from_slice(&imm.to_le_bytes());
}

impl AddSrc for Imm32 {
    #[inline(always)]
    fn add_into(self, code: &mut Vec<u8>, dst: Gpr) {
        ri32(code, 0, dst, self.0);
    }
}

/// `sub dst, imm32`
#[inline(always)]
pub fn sub(code: &mut Vec<u8>, dst: Gpr, Imm32(imm): Imm32) {
    ri32(code, 5, dst, imm);
}

/// `ret`
#[inline(always)]
pub fn ret(code: &mut Vec<u8>) {
    code.push(0xC3);
}

/// The 4-bit condition an x86 `jcc` tests — the whole field, not a selection.
///
/// `0F 8x rel32` is one instruction whose low opcode nibble *is* this value, so
/// the assembler encodes it by casting rather than by dispatching to one
/// hand-written mnemonic per condition. Named by the ISA's own mnemonics, with
/// their aliases, because that is what a reader checks against the manual.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Cond {
    /// `jo` — overflow.
    O = 0x0,
    /// `jno` — no overflow.
    No = 0x1,
    /// `jb` / `jc` / `jnae` — CF set; unsigned `<`.
    B = 0x2,
    /// `jae` / `jnb` / `jnc` — CF clear; unsigned `>=`.
    Ae = 0x3,
    /// `je` / `jz` — ZF set.
    E = 0x4,
    /// `jne` / `jnz` — ZF clear.
    Ne = 0x5,
    /// `jbe` / `jna` — unsigned `<=`.
    Be = 0x6,
    /// `ja` / `jnbe` — unsigned `>`.
    A = 0x7,
    /// `js` — sign set.
    S = 0x8,
    /// `jns` — sign clear.
    Ns = 0x9,
    /// `jp` / `jpe` — parity even; set by an unordered float compare.
    P = 0xA,
    /// `jnp` / `jpo` — parity odd.
    Np = 0xB,
    /// `jl` / `jnge` — signed `<`.
    L = 0xC,
    /// `jge` / `jnl` — signed `>=`.
    Ge = 0xD,
    /// `jle` / `jng` — signed `<=`.
    Le = 0xE,
    /// `jg` / `jnle` — signed `>`.
    G = 0xF,
}

/// `jmp rel32 target` — an unconditional branch to a [`Label`].
///
/// A struct, like every other instruction here, and its label is an operand
/// like any other. It emits a zero displacement; the assembler writes the real
/// one once the label lands, which is what [`AsmInsn::label_ref`] tells it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Jmp {
    /// Where it goes.
    pub target: Label,
}

impl AsmInsn for Jmp {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        code.extend_from_slice(&[0xE9, 0, 0, 0, 0]);
    }

    #[inline]
    fn label_ref(self) -> Option<LabelRef> {
        Some(LabelRef {
            label: self.target,
            patch: |code, at, target| patch_rel32(code, at + JMP_DISP, target),
        })
    }
}

/// `jcc rel32 target` — a conditional branch to a [`Label`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Jcc {
    /// What must hold for the branch to be taken.
    pub condition: Cond,
    /// Where it goes.
    pub target: Label,
}

impl Jcc {
    /// One constructor per mnemonic, so a call site reads like the assembly it
    /// is: `Jcc::je(exit)` for `je exit`.
    ///
    /// Sugar over the one encoder, not sixteen types: `0F 8x rel32` is a
    /// single instruction whose low opcode nibble is [`Cond`], and sixteen
    /// structs would be sixteen copies of one `emit_into` differing by a
    /// constant. The mnemonics are the assembler's names for the field.
    #[must_use]
    #[inline(always)]
    pub const fn je(target: Label) -> Self {
        Self::on(Cond::E, target)
    }
    /// `jne` / `jnz`.
    #[must_use]
    #[inline(always)]
    pub const fn jne(target: Label) -> Self {
        Self::on(Cond::Ne, target)
    }
    /// `jb` / `jc` / `jnae` — unsigned `<`.
    #[must_use]
    #[inline(always)]
    pub const fn jb(target: Label) -> Self {
        Self::on(Cond::B, target)
    }
    /// `jae` / `jnb` / `jnc` — unsigned `>=`.
    #[must_use]
    #[inline(always)]
    pub const fn jae(target: Label) -> Self {
        Self::on(Cond::Ae, target)
    }
    /// `jbe` / `jna` — unsigned `<=`.
    #[must_use]
    #[inline(always)]
    pub const fn jbe(target: Label) -> Self {
        Self::on(Cond::Be, target)
    }
    /// `ja` / `jnbe` — unsigned `>`.
    #[must_use]
    #[inline(always)]
    pub const fn ja(target: Label) -> Self {
        Self::on(Cond::A, target)
    }
    /// `jl` / `jnge` — signed `<`.
    #[must_use]
    #[inline(always)]
    pub const fn jl(target: Label) -> Self {
        Self::on(Cond::L, target)
    }
    /// `jge` / `jnl` — signed `>=`.
    #[must_use]
    #[inline(always)]
    pub const fn jge(target: Label) -> Self {
        Self::on(Cond::Ge, target)
    }
    /// `jle` / `jng` — signed `<=`.
    #[must_use]
    #[inline(always)]
    pub const fn jle(target: Label) -> Self {
        Self::on(Cond::Le, target)
    }
    /// `jg` / `jnle` — signed `>`.
    #[must_use]
    #[inline(always)]
    pub const fn jg(target: Label) -> Self {
        Self::on(Cond::G, target)
    }
    /// `js` — sign set.
    #[must_use]
    #[inline(always)]
    pub const fn js(target: Label) -> Self {
        Self::on(Cond::S, target)
    }
    /// `jns` — sign clear.
    #[must_use]
    #[inline(always)]
    pub const fn jns(target: Label) -> Self {
        Self::on(Cond::Ns, target)
    }
    /// `jo` — overflow.
    #[must_use]
    #[inline(always)]
    pub const fn jo(target: Label) -> Self {
        Self::on(Cond::O, target)
    }
    /// `jno` — no overflow.
    #[must_use]
    #[inline(always)]
    pub const fn jno(target: Label) -> Self {
        Self::on(Cond::No, target)
    }
    /// `jp` / `jpe` — parity even; set by an unordered float compare.
    #[must_use]
    #[inline(always)]
    pub const fn jp(target: Label) -> Self {
        Self::on(Cond::P, target)
    }
    /// `jnp` / `jpo` — parity odd.
    #[must_use]
    #[inline(always)]
    pub const fn jnp(target: Label) -> Self {
        Self::on(Cond::Np, target)
    }

    /// The branch on a condition chosen at run time, where no single mnemonic
    /// names it.
    #[must_use]
    #[inline(always)]
    pub const fn on(condition: Cond, target: Label) -> Self {
        Self { condition, target }
    }
}

impl AsmInsn for Jcc {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        code.extend_from_slice(&[0x0F, 0x80 | self.condition as u8, 0, 0, 0, 0]);
    }

    #[inline]
    fn label_ref(self) -> Option<LabelRef> {
        Some(LabelRef {
            label: self.target,
            patch: |code, at, target| patch_rel32(code, at + JCC_DISP, target),
        })
    }
}

/// Bytes from a `jmp`'s start to its displacement field: one opcode byte.
const JMP_DISP: usize = 1;
/// Bytes from a `jcc`'s start to its displacement field: `0F` plus the
/// condition byte.
const JCC_DISP: usize = 2;

/// Write a `rel32` at `pos` so the instruction it belongs to reaches `target`.
///
/// `rel32` is measured from the *end* of the instruction, which is the end of
/// the displacement field itself.
fn patch_rel32(code: &mut [u8], pos: usize, target: usize) {
    let rel = (target as i64) - (pos as i64 + 4);
    let rel = i32::try_from(rel).expect("an x86 rel32 spans \u{00b1}2 GiB");
    code[pos..pos + 4].copy_from_slice(&rel.to_le_bytes());
}

// =============================================================================
// The store's address arithmetic and the iota, in the general file
// =============================================================================

/// `movabs dst, imm64` — `REX.W B8+rd io`.
#[inline(always)]
pub fn movabs(code: &mut Vec<u8>, dst: Gpr, imm: u64) {
    code.push(0x48 | ((dst.0 >> 3) & 1));
    code.push(0xB8 | (dst.0 & 7));
    code.extend_from_slice(&imm.to_le_bytes());
}

/// `mov r32, imm32` — `B8+rd id`, zero-extended into the 64-bit register.
#[inline(always)]
pub fn mov_imm32(code: &mut Vec<u8>, dst: Gpr, imm: u32) {
    if dst.0 >= 8 {
        code.push(0x41);
    }
    code.push(0xB8 | (dst.0 & 7));
    code.extend_from_slice(&imm.to_le_bytes());
}

/// `imul dst, src` — `REX.W 0F AF /r`, the two-operand 64-bit multiply.
#[inline(always)]
pub fn imul(code: &mut Vec<u8>, dst: Gpr, src: Gpr) {
    code.extend_from_slice(&[rex_w(dst, src), 0x0F, 0xAF, modrm_rr(dst.0, src)]);
}

/// `lea dst, [base + index*4]` — `REX.W 8D /r` with a SIB: the element
/// address of a plane of `f32`s, in one instruction.
///
/// `rbp`/`r13` have no `mod = 00` form as a SIB base (that encoding means
/// "no base"), so those two take `mod = 01` with a zero `disp8`.
#[inline(always)]
pub fn lea_scaled4(code: &mut Vec<u8>, dst: Gpr, base: Gpr, index: Gpr) {
    debug_assert!(index.0 & 7 != RM_SIB, "rsp/r12 cannot index a SIB");
    let rex = 0x48 | (((dst.0 >> 3) & 1) << 2) | (((index.0 >> 3) & 1) << 1) | ((base.0 >> 3) & 1);
    let disp8_form = base.0 & 7 == RM_RIP_AT_MOD0;
    code.push(rex);
    code.push(0x8D);
    code.push(if disp8_form { 0x40 } else { 0x00 } | ((dst.0 & 7) << 3) | RM_SIB);
    code.push((0b10 << 6) | ((index.0 & 7) << 3) | (base.0 & 7));
    if disp8_form {
        code.push(0);
    }
}

/// A slot in the allocated spill frame. Kernels are leaves with no base
/// pointer, so a slot *is* `rsp + offset`, on every x86 tier.
pub(in crate::emit) const fn frame_slot(offset: u32) -> Mem<Imm32> {
    Mem {
        base: ptr::RSP,
        disp: Imm32(offset as i32),
    }
}

/// One tier's `cvttss2si` pair — the VEX or EVEX spelling of the same
/// instruction, which is the only thing that varies between the tiers'
/// address arithmetic.
#[derive(Clone, Copy)]
pub(in crate::emit) struct Convert {
    pub from_xmm: fn(&mut Vec<u8>, Gpr, Reg),
    pub from_mem: fn(&mut Vec<u8>, Gpr, Mem<Imm32>),
}

/// `dst = trunc(index)` as a 64-bit integer, wherever a fold keeps its
/// binder: a broadcast, so lane 0 of a register or the first word of a
/// slot is the index. Shared by the x86 tiers, which differ only in the
/// *vector* encoding this reads through — the GPR half is the
/// architecture's.
pub(in crate::emit) fn index_into(code: &mut Vec<u8>, dst: Gpr, at: Binding, convert: Convert) {
    match at {
        Binding::Loc(Loc::Reg(r)) => (convert.from_xmm)(code, dst, r),
        Binding::Loc(Loc::Slot(slot)) => (convert.from_mem)(code, dst, frame_slot(slot.offset())),
        Binding::Loc(Loc::Ptr(_)) => unreachable!("a fold's binder is a vector"),
        // A fold whose binder folded to a constant: the trip count was
        // one and the allocator rematerialized it. Truncate on the host,
        // which is what the instruction would have done.
        Binding::Remat(bits) => movabs(code, dst, f32::from_bits(bits) as i64 as u64),
    }
}

/// The store's address into `scratch[0]`: `out + 4 · (row · pitch + col)`,
/// leaving `scratch[1]` free. The row and column indices are converted
/// through `convert`, the tier's own `cvttss2si`.
pub(in crate::emit) fn write_address(
    code: &mut Vec<u8>,
    file: &regalloc::RegisterFile,
    write: &WritePlan,
    convert: Convert,
) -> Gpr {
    let row = crate::emit::declared_gpr_temp(write.scratch.gpr_temp(0));
    let col = crate::emit::declared_gpr_temp(write.scratch.gpr_temp(1));
    let out = file.gpr_out.expect("x86's store needs the output pointer");
    let pitch = file.gpr_pitch.expect("x86's store needs the pitch");
    index_into(code, row, write.row, convert);
    imul(code, row, pitch);
    index_into(code, col, write.col, convert);
    AsmProgram::from([Inst::Add { dst: row, src: col }]).assemble(code);
    lea_scaled4(code, row, out, row);
    row
}

// =============================================================================
// Memory operands
// =============================================================================

/// The displacement half of a [`Mem`] — and, because x86 spells the
/// displacement's *width* in the ModRM `mod` field, the addressing mode.
///
/// `mod = 00 / 01 / 10` are three modes rather than three spellings of one:
/// they cost a different number of bytes, and `mod = 00` is not "a
/// displacement of zero" (see [`NoDisp`]). So the width is picked by the
/// operand's TYPE, exactly as [`AddSrc`] picks `83 /0 ib` over `81 /0 id` —
/// never by the caller reaching for a differently-named function, which is
/// where that choice used to live.
pub trait Disp: Copy {
    /// The ModRM `mod` field this displacement implies.
    const MOD: u8;
    /// Append the displacement bytes, if the mode has any.
    fn emit(self, code: &mut Vec<u8>);
    /// Append the displacement bytes into an `EncodedInst`.
    fn emit_inst(self, inst: &mut EncodedInst);
}

/// No displacement — the `mod = 00` form, `[base]`.
///
/// A mode of its own, not `Imm8(0)`: it is a byte shorter, and it is not
/// available for every base. `mod = 00` with `rm = 101` (rbp/r13) means
/// RIP-relative, a different address entirely, so those two registers have no
/// bare `[base]` form and must spell it `Imm8(0)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NoDisp;

impl Disp for NoDisp {
    const MOD: u8 = 0x00;
    #[inline(always)]
    fn emit(self, _code: &mut Vec<u8>) {}
    #[inline(always)]
    fn emit_inst(self, _inst: &mut EncodedInst) {}
}

impl Disp for Imm8 {
    const MOD: u8 = 0x40;
    #[inline(always)]
    fn emit(self, code: &mut Vec<u8>) {
        code.push(self.0 as u8);
    }
    #[inline(always)]
    fn emit_inst(self, inst: &mut EncodedInst) {
        inst.push(self.0 as u8);
    }
}

impl Disp for Imm32 {
    const MOD: u8 = 0x80;
    #[inline(always)]
    fn emit(self, code: &mut Vec<u8>) {
        code.extend_from_slice(&self.0.to_le_bytes());
    }
    #[inline(always)]
    fn emit_inst(self, inst: &mut EncodedInst) {
        inst.extend(&self.0.to_le_bytes());
    }
}

/// A register usable as the base of a memory address ([`Mem`]).
pub trait BaseReg: Copy {
    fn reg_num(self) -> u8;
}

impl BaseReg for Gpr {
    #[inline(always)]
    fn reg_num(self) -> u8 {
        self.0
    }
}

impl BaseReg for PtrReg {
    #[inline(always)]
    fn reg_num(self) -> u8 {
        self.0
    }
}

/// An address spelled `[base + disp]`.
///
/// The base being a [`Gpr`] or [`PtrReg`] is the point: `rsp` is a value here. It used to be
/// the `_rsp` and `_base` suffixes of five separate functions that all encoded
/// the same `movups`, where nothing could check it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Mem<D, P = PtrReg> {
    /// The register the displacement is measured from.
    pub base: P,
    /// The displacement — and, through its type, the mode (see [`Disp`]).
    pub disp: D,
}

/// ModRM `rm` meaning "a SIB byte follows" — also the low three bits of
/// `rsp`/`r12`, which is why exactly those two bases always take one.
const RM_SIB: u8 = 0b100;

/// ModRM `rm` that means RIP-relative when `mod = 00` — also the low three
/// bits of `rbp`/`r13`.
const RM_RIP_AT_MOD0: u8 = 0b101;

/// SIB naming the base register alone: scale 1, index `100` (none).
const SIB_BASE_ONLY: u8 = 0x24;

/// Write the ModRM/SIB tail for `[base + index*4]` into an `EncodedInst`:
/// `mod = 00`, a SIB with scale 4 and no displacement — the element of a
/// plane of `f32`s, addressed in one instruction. The tail is the
/// architecture's, not the prefix's, so the VEX and EVEX tiers share it the
/// way they share [`mem_operand_into`].
pub(in crate::emit) fn scaled4_operand_into(
    inst: &mut EncodedInst,
    reg: u8,
    base: Gpr,
    index: Gpr,
) {
    debug_assert!(index.0 & 7 != RM_SIB, "rsp/r12 cannot index a SIB");
    sib4_tail_into(inst, reg, base, index.0);
}

/// [`scaled4_operand_into`] with a *vector* index — the VSIB a gather
/// addresses through, `[base + ymm*4]`. Same bytes; the one rule that does
/// not carry over is the GPR one, because SIB index `100` means "no index"
/// only when the index is a general register: as a vector number it is
/// `ymm4`/`ymm12`, which a gather may perfectly well be indexed by. The
/// prefix's X bit carries the index's high bit either way.
pub(in crate::emit) fn vsib4_operand_into(inst: &mut EncodedInst, reg: u8, base: Gpr, index: Reg) {
    sib4_tail_into(inst, reg, base, index.0);
}

/// The ModRM/SIB bytes both scaled-index forms share.
fn sib4_tail_into(inst: &mut EncodedInst, reg: u8, base: Gpr, index: u8) {
    debug_assert!(
        base.0 & 7 != RM_RIP_AT_MOD0,
        "[rbp/r13 + index*4] has no mod=00 form: that base means no base"
    );
    inst.push(((reg & 7) << 3) | RM_SIB);
    inst.push((0b10 << 6) | ((index & 7) << 3) | (base.0 & 7));
}

/// Write the ModRM/SIB/disp tail into an `EncodedInst`.
pub(in crate::emit) fn mem_operand_into<D: Disp, P: BaseReg>(
    inst: &mut EncodedInst,
    reg: u8,
    addr: Mem<D, P>,
) {
    let rm = addr.base.reg_num() & 7;
    debug_assert!(
        D::MOD != NoDisp::MOD || rm != RM_RIP_AT_MOD0,
        "[rbp]/[r13] has no mod=00 form: that encoding is RIP-relative"
    );
    inst.push(D::MOD | ((reg & 7) << 3) | rm);
    if rm == RM_SIB {
        inst.push(SIB_BASE_ONLY);
    }
    addr.disp.emit_inst(inst);
}

#[cfg(test)]
mod gpr_tests {
    use super::gpr::*;
    use super::*;

    fn asm(f: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
        let mut c = Vec::new();
        f(&mut c);
        c
    }

    fn one(inst: EncodedInst) -> Vec<u8> {
        asm(|c| AsmProgram::from([inst]).assemble(c))
    }

    /// A vehicle for the memory-operand tail: `movups [addr], xmm` (`0F 11
    /// /r`), the simplest legacy instruction that takes a [`Mem`]. Test-only
    /// — no tier emits it — so what is under test is [`mem_operand_into`]
    /// and the REX bits, not a 128-bit store.
    fn movups_store<D: Disp, P: BaseReg>(src: Reg, addr: Mem<D, P>) -> EncodedInst {
        let mut inst = EncodedInst::new();
        let rex = 0x40 | (u8::from(src.0 >= 8) << 2) | u8::from(addr.base.reg_num() >= 8);
        if rex != 0x40 {
            inst.push(rex);
        }
        inst.push(0x0F);
        inst.push(0x11);
        mem_operand_into(&mut inst, src.0, addr);
        inst
    }

    /// Each encoding checked against the Intel SDM's form for that mnemonic.
    #[test]
    fn encodings_match_the_manual() {
        // REX.W 89 /r — MOV r/m64, r64
        assert_eq!(asm(|c| mov(c, R10, RCX)), [0x49, 0x89, 0xCA]);
        // REX.W 31 /r — XOR r/m64, r64
        assert_eq!(asm(|c| xor(c, R11, R11)), [0x4D, 0x31, 0xDB]);
        assert_eq!(asm(|c| xor(c, R9, R9)), [0x4D, 0x31, 0xC9]);
        // REX.W 39 /r — CMP r/m64, r64
        assert_eq!(asm(|c| cmp(c, R11, R10)), [0x4D, 0x39, 0xD3]);
        assert_eq!(asm(|c| cmp(c, R9, RDX)), [0x49, 0x39, 0xD1]);
        // REX.W FF /0 — INC r/m64
        assert_eq!(asm(|c| inc(c, R9)), [0x49, 0xFF, 0xC1]);
        assert_eq!(asm(|c| inc(c, R11)), [0x49, 0xFF, 0xC3]);
        // REX.W 01 /r — ADD r/m64, r64
        assert_eq!(asm(|c| add(c, RSI, R8)), [0x4C, 0x01, 0xC6]);
        // REX.W 83 /0 ib — ADD r/m64, imm8
        assert_eq!(asm(|c| add(c, RSI, Imm8(16))), [0x48, 0x83, 0xC6, 0x10]);
        assert_eq!(asm(|c| add(c, RSI, Imm8(64))), [0x48, 0x83, 0xC6, 0x40]);
        // C3 — RET
        assert_eq!(asm(ret), [0xC3]);
        // REX.W B8+rd io — MOV r64, imm64
        assert_eq!(
            asm(|c| movabs(c, RAX, 0x3F80_0000_0000_0000)),
            [0x48, 0xB8, 0, 0, 0, 0, 0, 0, 0x80, 0x3F]
        );
        assert_eq!(asm(|c| movabs(c, R9, 1))[..2], [0x49, 0xB9]);
        // B8+rd id — MOV r32, imm32
        assert_eq!(asm(|c| mov_imm32(c, RCX, 7)), [0xB9, 7, 0, 0, 0]);
        assert_eq!(asm(|c| mov_imm32(c, R9, 7)), [0x41, 0xB9, 7, 0, 0, 0]);
        // REX.W 0F AF /r — IMUL r64, r/m64
        assert_eq!(asm(|c| imul(c, RAX, RDX)), [0x48, 0x0F, 0xAF, 0xC2]);
        // REX.W 8D /r — LEA r64, [base + index*4]
        assert_eq!(
            asm(|c| lea_scaled4(c, RAX, RSI, RAX)),
            [0x48, 0x8D, 0x04, 0x86]
        );
        assert_eq!(
            asm(|c| lea_scaled4(c, RCX, Gpr(5), RCX)),
            [0x48, 0x8D, 0x4C, 0x8D, 0x00],
            "rbp as a base takes the disp8 form"
        );
        assert_eq!(
            asm(|c| lea_scaled4(c, R9, RSI, R10))[0],
            0x4E,
            "REX.R and REX.X"
        );
    }

    #[test]
    fn asm_program_declarative_array() {
        let mut buff = Vec::new();
        AsmProgram::from([Inst::Mov { src: AX, dst: RX }]).assemble(&mut buff);
        assert_eq!(buff, [0x48, 0x89, 0xC0]);

        let mut seq = Vec::new();
        AsmProgram::from([
            Inst::Mov { src: AX, dst: RX },
            Inst::Xor { dst: R11, src: R11 },
            Inst::Inc { dst: R9 },
            Inst::Ret,
        ])
        .assemble(&mut seq);
        assert_eq!(
            seq,
            [
                0x48, 0x89, 0xC0, // mov rax, rax
                0x4D, 0x31, 0xDB, // xor r11, r11
                0x49, 0xFF, 0xC1, // inc r9
                0xC3, // ret
            ]
        );
    }

    /// REX.R extends the source, REX.B the destination; a register above r7
    /// on either side must set its own bit and no other.
    #[test]
    fn rex_extends_each_operand_independently() {
        assert_eq!(asm(|c| mov(c, RCX, RDX))[0], 0x48, "neither extended");
        assert_eq!(
            asm(|c| mov(c, R9, RDX))[0],
            0x49,
            "destination extended → B"
        );
        assert_eq!(asm(|c| mov(c, RCX, R9))[0], 0x4C, "source extended → R");
        assert_eq!(asm(|c| mov(c, R9, R10))[0], 0x4D, "both extended");
    }

    /// A branch's displacement is measured from the *next* instruction, and a
    /// backward one is negative. Asserted through the assembler, because that
    /// is who writes it: the mnemonics emit a zero placeholder and report
    /// nothing.
    #[test]
    fn branches_patch_relative_to_the_next_instruction() {
        use crate::emit::{AsmProgram, Item, Label};

        let end = Label::new("end");
        let mut c = Vec::new();
        AsmProgram::new([
            Item::Inst(Inst::from(Jmp { target: end })),
            // Eleven bytes of padding, so the label lands at 16.
            Item::Inst(Inst::Encoded(EncodedInst::from_slice(&[0x90; 11]))),
            Item::Label(end),
        ])
        .assemble(&mut c);
        assert_eq!(c.len(), 16);
        assert_eq!(c[0], 0xE9, "E9 + rel32");
        assert_eq!(
            &c[1..5],
            &(16i32 - 5).to_le_bytes(),
            "rel is from the next insn"
        );

        let top = Label::new("end");
        let mut c = Vec::new();
        AsmProgram::new([Item::Label(top), Item::Inst(Inst::from(Jcc::jae(top)))]).assemble(&mut c);
        assert_eq!(c[..2], [0x0F, 0x83]);
        assert_eq!(&c[2..6], &(-6i32).to_le_bytes(), "a back edge is negative");
    }

    /// The pointer class's loads and stores reach every GPR and address the
    /// frame: `mov r9, [rdi + 96]` (REX.R for the high destination), `mov
    /// r10, [rsp + 32]` (REX.R, and `rsp`'s SIB), `mov [rsp + 32], r11`.
    #[test]
    fn pointer_loads_and_stores_encode_high_registers_and_the_frame() {
        let mut code = Vec::new();
        AsmProgram::from([
            MovLoadPtr {
                dst: PtrReg(9),
                base: PtrReg(7),
                disp: 96,
            }
            .encode(),
            MovLoadPtr {
                dst: PtrReg(10),
                base: ptr::RSP,
                disp: 32,
            }
            .encode(),
            MovStorePtr {
                src: PtrReg(11),
                base: ptr::RSP,
                disp: 32,
            }
            .encode(),
        ])
        .assemble(&mut code);
        assert_eq!(
            code,
            vec![
                0x4C, 0x8B, 0x8F, 96, 0, 0, 0, // mov r9, [rdi + 96]
                0x4C, 0x8B, 0x94, 0x24, 32, 0, 0, 0, // mov r10, [rsp + 32]
                0x4C, 0x89, 0x9C, 0x24, 32, 0, 0, 0, // mov [rsp + 32], r11
            ]
        );
    }

    /// REX extends each side of a memory operand independently: R for the
    /// register, B for the base, both, or neither.
    #[test]
    fn a_memory_operands_rex_bits_follow_its_two_registers() {
        let store = |reg, base| {
            one(movups_store(
                reg,
                Mem {
                    base: Gpr(base),
                    disp: NoDisp,
                },
            ))
        };
        assert_eq!(store(Reg(3), 3), [0x0F, 0x11, ((3 & 7) << 3) | 3]);
        assert_eq!(store(Reg(9), 3), [0x44, 0x0F, 0x11, ((9 & 7) << 3) | 3]);
        assert_eq!(
            store(Reg(2), 11),
            [0x41, 0x0F, 0x11, ((2 & 7) << 3) | (11 & 7)]
        );
        assert_eq!(
            store(Reg(11), 14),
            [0x45, 0x0F, 0x11, ((11 & 7) << 3) | (14 & 7)]
        );
    }

    /// One instruction, three bases: `[rsp+8]`, `[rax+8]` and `[r10+8]`
    /// differ only in ModRM.rm (plus the SIB `rsp` implies and the REX.B `r10`
    /// does). That is why the base is an operand and not a name suffix.
    #[test]
    fn the_base_register_is_an_operand() {
        let store = |base| {
            one(movups_store(
                Reg(1),
                Mem {
                    base,
                    disp: Imm8(8),
                },
            ))
        };
        // 0F 11 /r, mod=01: rsp takes a SIB byte, rax and r10 do not.
        assert_eq!(store(RSP), [0x0F, 0x11, 0x4C, 0x24, 0x08]);
        assert_eq!(store(RAX), [0x0F, 0x11, 0x48, 0x08]);
        assert_eq!(store(R10), [0x41, 0x0F, 0x11, 0x4A, 0x08]);
    }

    /// The displacement's *type* picks the encoding, so the same address at the
    /// same offset is a 5-byte or an 8-byte instruction depending on which
    /// operand the caller built — and `NoDisp` is a third, shorter mode, not
    /// `Imm8(0)`.
    #[test]
    fn the_displacement_type_picks_the_encoding() {
        let d8 = one(movups_store(
            Reg(0),
            Mem {
                base: RSP,
                disp: Imm8(16),
            },
        ));
        let d32 = one(movups_store(
            Reg(0),
            Mem {
                base: RSP,
                disp: Imm32(16),
            },
        ));
        assert_eq!(d8, [0x0F, 0x11, 0x44, 0x24, 0x10], "mod=01, disp8");
        assert_eq!(
            d32,
            [0x0F, 0x11, 0x84, 0x24, 0x10, 0x00, 0x00, 0x00],
            "mod=10, disp32"
        );

        let bare = one(movups_store(
            Reg(0),
            Mem {
                base: RSI,
                disp: NoDisp,
            },
        ));
        let zero = one(movups_store(
            Reg(0),
            Mem {
                base: RSI,
                disp: Imm8(0),
            },
        ));
        assert_eq!(bare, [0x0F, 0x11, 0x06], "mod=00 is its own mode");
        assert_eq!(zero, [0x0F, 0x11, 0x46, 0x00], "and a byte longer than it");
    }

    /// The frame slot every x86 tier spills through: `[rsp + disp32]`.
    #[test]
    fn a_frame_slot_is_rsp_plus_a_disp32() {
        let slot = frame_slot(64);
        assert_eq!(slot.base, ptr::RSP);
        assert_eq!(slot.disp, Imm32(64));
        assert_eq!(
            one(movups_store(Reg(9), slot)),
            [0x44, 0x0F, 0x11, 0x8C, 0x24, 64, 0, 0, 0]
        );
    }

    /// `Gpr` and `Reg` name different files; the same index is a different
    /// register in each, which is why they are different types.
    #[test]
    fn the_two_register_files_are_not_interchangeable() {
        // r10 and xmm10 share an index and nothing else.
        assert_eq!(R10.0, Reg(10).0);
        // `mov` takes Gpr; passing Reg(10) would not compile. Encoding r10 as
        // the destination sets REX.B, which a vector encoder never emits here.
        assert_eq!(asm(|c| mov(c, R10, RAX))[0] & 1, 1);
    }
}
