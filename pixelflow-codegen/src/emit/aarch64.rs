//! ARM64/NEON instruction encoding.
//!
//! Each function emits raw machine code bytes for one instruction (or a small fixed sequence).
//! These are the "atoms" that compound operations are built from.

use super::{AsmInsn, AsmProgram, Gpr, Label, LabelRef, PtrReg, Reg, unimplemented_op};
use crate::error::CompileError;
use alloc::vec::Vec;
use pixelflow_ir::kind::OpKind;

mod table;
use table::*;

// =============================================================================
// Instruction Encoding Helpers
// =============================================================================

/// An A64 instruction is one 32-bit word, and a branch displacement counts
/// them.
const WORD_BYTES: usize = 4;

/// Write a 32-bit instruction to the code buffer.
#[inline]
fn emit32(code: &mut Vec<u8>, inst: u32) {
    code.extend_from_slice(&inst.to_le_bytes());
}

// =============================================================================
// First-Class AArch64 Instructions
// =============================================================================

/// A concrete ARM64 instruction.
///
/// Denotationally, every single-word ARM64 instruction is a pure `u32` value.
/// Compound or fallback instructions (like `LdrQ` with large displacements)
/// are assembled into code via [`AsmInsn::emit_into`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Inst {
    // Vector floating-point arithmetic (single instruction)
    Fadd(Reg, Reg, Reg),
    Fsub(Reg, Reg, Reg),
    Fmul(Reg, Reg, Reg),
    Fdiv(Reg, Reg, Reg),
    Fmla(Reg, Reg, Reg),
    Fmin(Reg, Reg, Reg),
    Fmax(Reg, Reg, Reg),
    Fsqrt(Reg, Reg),
    Fabs(Reg, Reg),
    Fneg(Reg, Reg),
    Not(Reg, Reg),
    Frintm(Reg, Reg),
    Frintp(Reg, Reg),
    Frinta(Reg, Reg),
    Frsqrte(Reg, Reg),
    Frsqrts(Reg, Reg, Reg),
    Frecpe(Reg, Reg),
    Frecps(Reg, Reg, Reg),

    // Comparisons (result is vector mask)
    Fcmgt(Reg, Reg, Reg),
    Fcmge(Reg, Reg, Reg),
    Fcmeq(Reg, Reg, Reg),

    // Selection
    Bsl(Reg, Reg, Reg),

    // Memory transfers
    LdrQ(LdrQ),
    LdrX(LdrX),
    LdrS(LdrS),
    LdrW(LdrW),
    LdrSIndexed(LdrSIndexed),
    StrQ(StrQ),
    StrX(StrX),

    // Integer & lane operations
    DupLane0(Reg, Reg),
    UmovW { dst: Gpr, src: Reg, lane: u8 },
    InsW { dst: Reg, lane: u8, src: Gpr },
    MvnW { dst: Gpr, src: Gpr },
    Fcvtzs(Reg, Reg),
    FcvtzsX { dst: Gpr, src: Reg },
    Scvtf(Reg, Reg),
    AddI32(Reg, Reg, Reg),
    And(Reg, Reg, Reg),
    Orr(Reg, Reg, Reg),
    Mov(Reg, Reg),

    // If guard masks
    Uminv(Reg, Reg),
    Umaxv(Reg, Reg),
    FmovToGp(Reg),

    // Control & GPR
    Ret,
    Raw(u32),
}

impl Inst {
    #[must_use]
    #[inline(always)]
    fn ldr_q(dst: Reg, addr: Mem) -> Self {
        Self::LdrQ(LdrQ { dst, addr })
    }
    #[must_use]
    #[inline(always)]
    fn ldr_x(dst: PtrReg, addr: Mem) -> Self {
        Self::LdrX(LdrX { dst, addr })
    }
    #[must_use]
    #[inline(always)]
    fn ldr_s(dst: Reg, addr: Mem) -> Self {
        Self::LdrS(LdrS {
            dst: SReg(dst),
            addr,
        })
    }
    #[must_use]
    #[inline(always)]
    fn ldr_w(dst: Gpr, addr: MemIndexed) -> Self {
        Self::LdrW(LdrW { dst, addr })
    }
    #[must_use]
    #[inline(always)]
    fn ldr_s_indexed(dst: Reg, addr: MemIndexed) -> Self {
        Self::LdrSIndexed(LdrSIndexed {
            dst: SReg(dst),
            addr,
        })
    }
    #[must_use]
    #[inline(always)]
    fn str_q(src: Reg, addr: Mem) -> Self {
        Self::StrQ(StrQ { src, addr })
    }
    #[must_use]
    #[inline(always)]
    fn str_x(src: PtrReg, addr: Mem) -> Self {
        Self::StrX(StrX { src, addr })
    }
    #[must_use]
    #[inline(always)]
    fn mov(dst: Reg, src: Reg) -> Self {
        Self::Mov(dst, src)
    }
    #[must_use]
    #[inline(always)]
    fn umov_w(dst: Gpr, src: Reg, lane: u8) -> Self {
        Self::UmovW { dst, src, lane }
    }
    #[must_use]
    #[inline(always)]
    fn ins_w(dst: Reg, lane: u8, src: Gpr) -> Self {
        Self::InsW { dst, lane, src }
    }
    #[must_use]
    #[inline(always)]
    fn mvn_w(dst: impl Into<Gpr>, src: impl Into<Gpr>) -> Self {
        Self::MvnW {
            dst: dst.into(),
            src: src.into(),
        }
    }

    /// Pure encoding of single-word instructions into a 32-bit machine word.
    #[must_use]
    #[inline]
    fn encode(self) -> u32 {
        match self {
            Inst::Fadd(dst, s1, s2) => Fadd::new(dst, s1, s2).encode(),
            Inst::Fsub(dst, s1, s2) => Fsub::new(dst, s1, s2).encode(),
            Inst::Fmul(dst, s1, s2) => Fmul::new(dst, s1, s2).encode(),
            Inst::Fdiv(dst, s1, s2) => Fdiv::new(dst, s1, s2).encode(),
            Inst::Fmla(dst, s1, s2) => Fmla::new(dst, s1, s2).encode(),
            Inst::Fmin(dst, s1, s2) => Fmin::new(dst, s1, s2).encode(),
            Inst::Fmax(dst, s1, s2) => Fmax::new(dst, s1, s2).encode(),
            Inst::Fsqrt(dst, src) => Fsqrt::new(dst, src).encode(),
            Inst::Fabs(dst, src) => Fabs::new(dst, src).encode(),
            Inst::Fneg(dst, src) => Fneg::new(dst, src).encode(),
            Inst::Not(dst, src) => Not::new(dst, src).encode(),
            Inst::Frintm(dst, src) => Frintm::new(dst, src).encode(),
            Inst::Frintp(dst, src) => Frintp::new(dst, src).encode(),
            Inst::Frinta(dst, src) => Frinta::new(dst, src).encode(),
            Inst::Frsqrte(dst, src) => Frsqrte::new(dst, src).encode(),
            Inst::Frsqrts(dst, s1, s2) => Frsqrts::new(dst, s1, s2).encode(),
            Inst::Frecpe(dst, src) => Frecpe::new(dst, src).encode(),
            Inst::Frecps(dst, s1, s2) => Frecps::new(dst, s1, s2).encode(),
            Inst::Fcmgt(dst, s1, s2) => Fcmgt::new(dst, s1, s2).encode(),
            Inst::Fcmge(dst, s1, s2) => Fcmge::new(dst, s1, s2).encode(),
            Inst::Fcmeq(dst, s1, s2) => Fcmeq::new(dst, s1, s2).encode(),
            Inst::Bsl(mask, if_true, if_false) => Bsl::new(mask, if_true, if_false).encode(),
            Inst::LdrQ(_)
            | Inst::LdrX(_)
            | Inst::LdrS(_)
            | Inst::LdrW(_)
            | Inst::LdrSIndexed(_)
            | Inst::StrQ(_)
            | Inst::StrX(_) => {
                panic!("Ldr and Str must be emitted via emit_into or AsmProgram")
            }
            Inst::DupLane0(dst, src) => DupLane0::new(dst, src).encode(),
            Inst::UmovW { dst, src, lane } => UmovW::new(dst, src, lane).encode(),
            Inst::InsW { dst, lane, src } => InsW::new(dst, lane, src).encode(),
            Inst::MvnW { dst, src } => table::MvnW::new(dst, src).encode(),
            Inst::Fcvtzs(dst, src) => Fcvtzs::new(dst, src).encode(),
            Inst::FcvtzsX { dst, src } => FcvtzsX::new(dst, src).encode(),
            Inst::Scvtf(dst, src) => Scvtf::new(dst, src).encode(),
            Inst::AddI32(dst, s1, s2) => AddI32::new(dst, s1, s2).encode(),
            Inst::And(dst, s1, s2) => And::new(dst, s1, s2).encode(),
            Inst::Orr(dst, s1, s2) => Orr::new(dst, s1, s2).encode(),
            Inst::Mov(dst, src) => Orr::new(dst, src, src).encode(),
            Inst::Uminv(dst, src) => Uminv::new(dst, src).encode(),
            Inst::Umaxv(dst, src) => Umaxv::new(dst, src).encode(),
            Inst::FmovToGp(src) => FmovToGp::new(src).encode(),
            Inst::Ret => Ret.encode(),
            Inst::Raw(w) => w,
        }
    }
}

impl crate::emit::AsmInsn for Inst {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        match self {
            Inst::LdrQ(l) => l.emit_into(code),
            Inst::LdrX(l) => l.emit_into(code),
            Inst::LdrS(l) => l.emit_into(code),
            Inst::LdrW(l) => l.emit_into(code),
            Inst::LdrSIndexed(l) => l.emit_into(code),
            Inst::StrQ(s) => s.emit_into(code),
            Inst::StrX(s) => s.emit_into(code),
            Inst::Mov(dst, src) => {
                if dst != src {
                    emit32(code, Orr::new(dst, src, src).encode());
                }
            }
            _ => emit32(code, self.encode()),
        }
    }
}

// =============================================================================
// Load / Store
// =============================================================================

/// `dst = splat(base[offset])`: `ldr s<dst>, [base, #offset*4]` reads the
/// value and `dup` spreads it. `base` is the block's address, wherever the
/// allocator keeps that pointer value. An element past the 12-bit scaled
/// immediate is addressed through IP0, as any deep displacement is
/// (`table::address_in_ip0`).
///
/// # Errors
///
/// [`CompileError::BudgetExceeded`] when the element's byte offset is past
/// the 32-bit offset [`Mem`] carries — refused, never wrapped into an
/// address that reads some other argument. That bound is the operand
/// type's, not the instruction's: `imm12` is the instruction's, and the IP0
/// fallback covers everything past it up to `Mem`'s.
fn emit_uniform_load(
    code: &mut Vec<u8>,
    dst: Reg,
    base: PtrReg,
    offset: u64,
) -> Result<(), CompileError> {
    let bytes = offset
        .checked_mul(u64::from(S_BYTES))
        .and_then(|bytes| u32::try_from(bytes).ok())
        .ok_or(CompileError::BudgetExceeded(
            "uniform byte offset past the 32-bit offset `Mem` carries",
        ))?;
    AsmProgram::from([
        Inst::ldr_s(
            dst,
            Mem {
                base,
                offset: bytes,
            },
        ),
        Inst::DupLane0(dst, dst),
    ])
    .assemble(code);
    Ok(())
}

// =============================================================================
// Constants
// =============================================================================

/// Load a floating-point constant into a vector register.
///
/// Strategy (in priority order):
/// 1. Zero: MOVI Vd.4S, #0 (1 instruction)
/// 2. FMOV-encodable: FMOV Vd.4S, #imm8 (1 instruction)
/// 3. General: MOVZ W16 + MOVK W16 + DUP Vd.4S, W16 (3 instructions)
///
/// TODO: Use a constant pool with LDR for better performance on general case.
fn emit_fmov_imm(code: &mut Vec<u8>, dst: Reg, val: f32) {
    let bits = val.to_bits();

    if bits == 0 {
        emit32(code, 0x4F000400 | (dst.0 as u32));
        return;
    }

    if let Some(imm8) = try_encode_fmov_imm8(val) {
        let abc = ((imm8 as u32) >> 5) & 0x7;
        let defgh = (imm8 as u32) & 0x1F;
        emit32(
            code,
            0x4F00_F400 | (abc << 16) | (defgh << 5) | (dst.0 as u32),
        );
        return;
    }

    // General case: load via GP register (W16)
    // This is 3 instructions but works for any f32 value.
    // Use W16 (IP0) as scratch - it's caller-saved and not used for arguments
    let lo16 = bits & 0xFFFF;
    let hi16 = bits >> 16;

    emit32(code, 0x52800010 | (lo16 << 5));

    emit32(code, 0x72A00010 | (hi16 << 5));

    emit32(code, 0x4E040C00 | (dst.0 as u32) | (16 << 5));
}

/// Try to encode an f32 as an ARM64 FMOV (vector, immediate) 8-bit value.
///
/// An f32 is FMOV-encodable when its bit pattern matches:
///   `[a] [NOT(b)] [bbbbb] [cdefgh] [19 zeros]`
/// producing imm8 = `abcdefgh`.
///
/// This covers values of the form `(-1)^a * 2^n * (1.0 + frac/64)`
/// where n is in [-3, +4] and frac is in [0, 63].
/// Common examples: 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, 1.5, etc.
///
/// Returns `None` for non-encodable values (including ±0.0, denormals, NaN, Inf).
#[must_use]
fn try_encode_fmov_imm8(val: f32) -> Option<u8> {
    let bits = val.to_bits();

    // Low 19 bits must be zero
    if bits & 0x7_FFFF != 0 {
        return None;
    }

    // ±0.0 is not FMOV-encodable (would require b=0 giving exp=0 which is denormal)
    if bits & 0x7FFF_FFFF == 0 {
        return None;
    }

    // bits[29:25] must all equal b, where NOT(b) = bit[30]
    let not_b = (bits >> 30) & 1;
    let b = not_b ^ 1;
    let rep5 = if b == 1 { 0x1F } else { 0x00 };
    let actual = (bits >> 25) & 0x1F;
    if actual != rep5 {
        return None;
    }

    let a = (bits >> 31) & 1;
    let c = (bits >> 24) & 1;
    let d = (bits >> 23) & 1;
    let e = (bits >> 22) & 1;
    let f = (bits >> 21) & 1;
    let g = (bits >> 20) & 1;
    let h = (bits >> 19) & 1;
    let imm8 = (a << 7) | (b << 6) | (c << 5) | (d << 4) | (e << 3) | (f << 2) | (g << 1) | h;
    Some(imm8 as u8)
}

// =============================================================================
// Constant Pool Support
// =============================================================================

// The pool's alignment is every backend's, not this one's; its label is minted
// once by `compile_via_backend` and handed to `anchor` and `finish`.

/// Returns true if the given f32 needs a constant pool entry (not zero, not FMOV-encodable).
#[must_use]
fn needs_const_pool(val: f32) -> bool {
    val.to_bits() != 0 && try_encode_fmov_imm8(val).is_none()
}

/// `adrp xd, #0` + `add xd, xd, #0`, sharing one [`Label`]: materialize the
/// constant pool's address in `dst`.
///
/// **Two instructions, because A64 is fixed 32-bit** — no single instruction
/// holds a 64-bit address. `ADR` reaches ±1 MiB by spending a 21-bit *byte*
/// displacement; `ADRP` spends the same 21 bits on 4 KiB *pages* instead —
/// `(PC & !0xFFF) + (imm21 << 12)`, ±4 GiB — and hands back the base of the
/// target's page, not the target itself. The `ADD`'s 12-bit immediate is
/// exactly one page wide, so it recovers the low bits `ADRP` had to discard.
///
/// **Always both, never the one-instruction `ADR`.** Which one is reachable
/// depends on the distance to the pool; the pool's position depends on where
/// every instruction ahead of it landed; and this instruction's own size is
/// one of those — so choosing the short form is branch relaxation, and
/// resolving it needs layout iterated to a fixed point to save four bytes
/// once per compiled function. The alternative this replaced tried to dodge
/// that fixed point instead of running it: emit `ADR` optimistically,
/// *estimate* the distance to the not-yet-emitted pool with a magic margin,
/// and — when the estimate crossed it — splice four bytes into the middle of
/// already-emitted code to widen it to `ADRP`+`ADD` after the fact. That
/// splice was sound only because the anchor sits above the whole loop nest,
/// so every branch in the body had both endpoints on the same side of it; a
/// branch spanning it would have broken silently. `AdrpAdd` has no estimate
/// and nothing to splice — its size is fixed before a single byte is laid
/// out, like every other instruction here.
///
/// # Alignment invariant
///
/// [`AsmInsn::label_ref`]'s patch below computes pages by masking *buffer
/// offsets* — positions within the `Vec<u8>` this crate is building, not
/// runtime addresses. That is correct only because the executable mapping
/// this buffer is copied into starts on a 4 KiB boundary: `(map + pos) &
/// !0xFFF == map + (pos & !0xFFF)` for every `pos` exactly when `map` is a
/// multiple of 4 KiB, which is what lets a page found by masking an offset
/// stand in for the page a masked address would find. Copy these bytes to an
/// address that is not 4 KiB-aligned and every `ADRP` here is off by a page.
///
/// Two things hold it up, and both are checked rather than assumed:
/// `CodePage::from_code`
/// writes the buffer at offset 0 of a mapping whose size — and therefore
/// whose base — is a whole number of pages, pinned by
/// `page_size_is_a_sane_power_of_two`; and an [`Assembly`](crate::emit::Assembly) position is an
/// offset into the *whole* kernel rather than into the scope that emitted it,
/// because one kernel is one `Assembly`, so there is no base to subtract. A
/// displacement cannot tell those two apart. A page can.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct AdrpAdd {
    /// Where the address is materialized.
    dst: Gpr,
    /// The constant pool's position.
    target: Label,
}

impl AsmInsn for AdrpAdd {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        let rd = u32::from(self.dst.0);
        // ADRP Xd, #0 — page immediate patched by `label_ref` below.
        emit32(code, 0x9000_0000 | rd);
        // ADD Xd, Xd, #0 (64-bit immediate form) — within-page offset
        // patched by `label_ref` below.
        emit32(code, 0x9100_0000 | (rd << 5) | rd);
    }

    #[inline]
    fn label_ref(self) -> Option<LabelRef> {
        Some(LabelRef {
            at: 0,
            label: self.target,
            // `at` is the ADRP word `emit_into` placed; the ADD it placed
            // right after sits at `at + 4`. Both words already carry their
            // opcode and register fields, so this only ORs the immediate in.
            patch: |code, at, target| {
                let page = |pos: usize| (pos as i64) & !0xFFF;
                let pages = (page(target) - page(at)) >> 12;
                assert!(
                    (-(1i64 << 20)..(1i64 << 20)).contains(&pages),
                    "ADRP page offset {pages} out of range (±4GiB)"
                );
                let imm = (pages as u32) & 0x1F_FFFF;
                let immlo = imm & 0x3;
                let immhi = (imm >> 2) & 0x7_FFFF;
                let adrp = u32::from_le_bytes(code[at..at + 4].try_into().unwrap());
                code[at..at + 4]
                    .copy_from_slice(&(adrp | (immlo << 29) | (immhi << 5)).to_le_bytes());

                let within_page = (target as u32) & 0xFFF;
                let add = u32::from_le_bytes(code[at + 4..at + 8].try_into().unwrap());
                code[at + 4..at + 8].copy_from_slice(&(add | (within_page << 10)).to_le_bytes());
            },
        })
    }
}

/// One constant pool entry: the four words of a 128-bit NEON register, lane
/// 0 first. A splat is the common one; the lattice's iota is the other.
type PoolEntry = [u32; 4];

/// Emit a constant pool entry — 16 bytes, lane 0 first.
fn emit_pool_entry(code: &mut Vec<u8>, entry: PoolEntry) {
    for word in entry {
        code.extend_from_slice(&word.to_le_bytes());
    }
}

/// The iota `[0, 1, 2, 3]`, as a pool entry.
const IOTA: PoolEntry = [
    0.0f32.to_bits(),
    1.0f32.to_bits(),
    2.0f32.to_bits(),
    3.0f32.to_bits(),
];

// =============================================================================
// Bound-Memory Gather (scalar-load lowering — NEON has no native gather)
// =============================================================================

/// GP scratch the scalar-load gather sequence clobbers.
struct GatherGprs {
    /// Scratch: one extracted lane index at a time. Clobbered.
    idx: Gpr,
    /// Scratch: one loaded value at a time. Clobbered.
    val: Gpr,
}

/// `dst.4S = base[idx_int.S[lane]]` for each lane — the NEON gather: four scalar
/// loads through GP scratch. `base` is the buffer's address, wherever the
/// allocator keeps that pointer value; `idx_int` holds int32 lane indices
/// (already converted and in-bounds by the `expand_gather` lowering).
/// Clobbers `gprs.idx` and `gprs.val`.
fn emit_gather(code: &mut Vec<u8>, dst: Reg, idx_int: Reg, base: PtrReg, gprs: GatherGprs) {
    let mem = MemIndexed {
        base,
        index: gprs.idx,
    };
    AsmProgram::from([
        Inst::umov_w(gprs.idx, idx_int, 0),
        Inst::ldr_w(gprs.val, mem),
        Inst::ins_w(dst, 0, gprs.val),
        Inst::umov_w(gprs.idx, idx_int, 1),
        Inst::ldr_w(gprs.val, mem),
        Inst::ins_w(dst, 1, gprs.val),
        Inst::umov_w(gprs.idx, idx_int, 2),
        Inst::ldr_w(gprs.val, mem),
        Inst::ins_w(dst, 2, gprs.val),
        Inst::umov_w(gprs.idx, idx_int, 3),
        Inst::ldr_w(gprs.val, mem),
        Inst::ins_w(dst, 3, gprs.val),
    ])
    .assemble(code);
}

/// The GP registers a broadcast load runs through: the buffer's address,
/// wherever the allocator keeps that pointer value, and the one index, this
/// instruction's `RegisterFile::gpr_scratch` reservation.
struct BroadcastGprs {
    /// The buffer base pointer.
    base: PtrReg,
    /// Receives the truncated index.
    index: Gpr,
}

/// `dst = splat(base[idx])`, the index being the same in every lane of
/// `idx`: `fcvtzs x<index>, s<idx>` truncates lane 0, `ldr s<dst>, [base,
/// w<index>, uxtw #2]` reads the element and `dup` spreads it. Three
/// instructions where the gather is thirteen; `dst` may alias `idx`, since
/// the index is in a GPR before `dst` is written.
fn emit_broadcast_load(code: &mut Vec<u8>, dst: Reg, idx: Reg, gprs: BroadcastGprs) {
    AsmProgram::from([
        Inst::FcvtzsX {
            dst: gprs.index,
            src: idx,
        },
        Inst::ldr_s_indexed(
            dst,
            MemIndexed {
                base: gprs.base,
                index: gprs.index,
            },
        ),
        Inst::DupLane0(dst, dst),
    ])
    .assemble(code);
}

// =============================================================================
// Integer Vector Operations (for bit manipulation in transcendentals)
// =============================================================================

/// USHR Vd.4S, Vn.4S, #shift (unsigned shift right by immediate)
fn emit_ushr(code: &mut Vec<u8>, dst: Reg, src: Reg, shift: u8) {
    // A shift by zero is the identity, and USHR cannot encode it: `64 - 0` is
    // 64, which does not fit the 6-bit immediate field. Emit the move instead
    // of refusing a perfectly portable operation — `fold_is_platform_specific`
    // classifies a count of 0 as agreeing on every target, so the encoder has
    // to honour it.
    if shift == 0 {
        AsmProgram::from([Inst::mov(dst, src)]).assemble(code);
        return;
    }
    // `immh` selects the element size: 01xx is .4S, 001x is .8H, 1xxx is .2D.
    // Only shifts in 1..=32 keep `64 - shift` inside 01xx, so anything else
    // silently encodes a DIFFERENT element size and crosses lane boundaries.
    assert!(
        shift <= 32,
        "aarch64 USHR .4S: shift {shift} exceeds 32 — the immediate would \
         encode a different element size, not a 32-bit lane shift"
    );
    let immhb = (64 - shift as u32) & 0x3F; // USHR uses (immh:immb) = (size*2 - shift)
    let inst = 0x6F200400 | (dst.0 as u32) | ((src.0 as u32) << 5) | (immhb << 16);
    emit32(code, inst);
}

/// SHL Vd.4S, Vn.4S, #shift (shift left by immediate)
fn emit_shl(code: &mut Vec<u8>, dst: Reg, src: Reg, shift: u8) {
    // For .4S: immh:immb = shift + 32. At shift >= 32 that carries into
    // `immh`, making it 1xxx — which the ARM ARM decodes as .2D, a 64-bit
    // element shift that leaks bits across the 32-bit lane boundary. Refuse
    // loudly rather than emit a silently different instruction.
    assert!(
        shift < 32,
        "aarch64 SHL .4S: shift {shift} is out of range for a 32-bit lane — \
         the immediate would encode .2D and cross lane boundaries"
    );
    let immhb = (shift as u32) + 32;
    let inst = 0x4F005400 | (dst.0 as u32) | ((src.0 as u32) << 5) | (immhb << 16);
    emit32(code, inst);
}

// =============================================================================
// Binary Transcendental Builtins
// =============================================================================

// =============================================================================
// Compound Operations (emit full instruction sequences)
// =============================================================================

/// How many registers this backend's encodings need beyond their operands.
///
/// Only the reciprocal estimates: `FRECPE`/`FRSQRTE` are estimates, and the
/// Newton-Raphson step that refines them needs somewhere to hold the
/// correction. `Neg` and `Abs` are single instructions here (`FNEG`, `FABS`),
/// unlike the x86 backends where they materialize a sign mask, and `BSL`
/// blends an `If` from its three operands.
fn temps_for(op: &super::ScheduledOp) -> u8 {
    use super::ScheduledOp;
    match op {
        ScheduledOp::Unary(OpKind::Rsqrt | OpKind::Recip, _) => 1,
        // The gather's truncated-index lanes.
        ScheduledOp::Gather(..) => 1,
        // A surviving fold's own loop: two transient registers for the trip
        // test and the accumulate — see `emit_scope`'s `Reduce` arm. The
        // binder and the accumulator are the fold's roots, placed by the
        // allocator, not scratch.
        ScheduledOp::Reduce(..) => super::regalloc::Scratch::REDUCE_TEMPS as u8,
        // A binder the allocator left in a slot has to pass through a vector
        // register on its way to `FCVTZS`: there is no memory-operand
        // convert on this ISA. Whether either binder is in a slot is the
        // allocation's answer, so the register is reserved regardless.
        ScheduledOp::Write { .. } => 1,
        _ => 0,
    }
}

/// How many GPRs this backend's encoding of `op` needs beyond
/// [`regalloc::RegisterFile::gpr_ctx`](crate::emit::regalloc::RegisterFile::gpr_ctx).
///
/// `Gather`'s scalar-load sequence needs a per-lane index and a loaded
/// value, each a GPR; `Broadcast` its one index, the element landing
/// straight in a vector lane; `Uniform` none — the base each addresses is a
/// pointer value the allocator carries, not scratch. A `Write` converts its
/// row and column into one each before combining them into the address. All
/// were `x9`/`x10`/`x11` chosen by hand before this work and are
/// `RegisterFile::gpr_scratch` reservations now.
fn gpr_temps_for(op: &super::ScheduledOp) -> u8 {
    use super::ScheduledOp;
    match op {
        ScheduledOp::Gather(..) | ScheduledOp::Write { .. } => 2,
        ScheduledOp::Broadcast(..) => 1,
        _ => 0,
    }
}

/// `dst = op(src)`.
///
/// The temp is the allocator's for this instruction; only the reciprocal
/// estimates use it, to hold the Newton-Raphson correction.
fn emit_unary(code: &mut Vec<u8>, unary: super::Unary) {
    let super::Unary { op, dst, src, temp } = unary;
    match op {
        OpKind::Neg => AsmProgram::from([Inst::Fneg(dst, src)]).assemble(code),
        OpKind::Abs => AsmProgram::from([Inst::Fabs(dst, src)]).assemble(code),
        OpKind::Sqrt => AsmProgram::from([Inst::Fsqrt(dst, src)]).assemble(code),
        OpKind::Rsqrt => {
            let temp = super::declared_temp(temp);
            AsmProgram::from([
                Inst::Frsqrte(dst, src),
                Inst::Fmul(temp, dst, dst),
                Inst::Frsqrts(temp, src, temp),
                Inst::Fmul(dst, dst, temp),
            ])
            .assemble(code);
        }
        OpKind::Recip => {
            let temp = super::declared_temp(temp);
            AsmProgram::from([
                Inst::Frecpe(dst, src),
                Inst::Frecps(temp, src, dst),
                Inst::Fmul(dst, dst, temp),
            ])
            .assemble(code);
        }
        OpKind::Floor => AsmProgram::from([Inst::Frintm(dst, src)]).assemble(code),
        OpKind::Ceil => AsmProgram::from([Inst::Frintp(dst, src)]).assemble(code),
        OpKind::Round => AsmProgram::from([Inst::Frinta(dst, src)]).assemble(code),

        OpKind::TruncToInt => AsmProgram::from([Inst::Fcvtzs(dst, src)]).assemble(code),
        OpKind::IntToFloat => AsmProgram::from([Inst::Scvtf(dst, src)]).assemble(code),

        // Transcendentals (sin/cos/tan/exp/exp2/ln/log2/log10/atan/asin/acos) are
        // expanded to primitive arithmetic by `lowering` before codegen, so they
        // never reach a backend. Reaching here means lowering was skipped.
        _ => unimplemented_op("aarch64", op),
    }
}

/// Emit a logical shift of i32 lanes by a compile-time immediate.
/// `Shl` -> `SHL`, `Shr` -> `USHR` (logical right). NEON shifts are imm-form.
fn emit_shift_imm(code: &mut Vec<u8>, op: OpKind, dst: Reg, src: Reg, amount: u8) {
    match op {
        OpKind::Shl => emit_shl(code, dst, src, amount),
        OpKind::Shr => emit_ushr(code, dst, src, amount),
        _ => unimplemented_op("aarch64", op),
    }
}

/// Emit binary operation
fn emit_binary(code: &mut Vec<u8>, op: OpKind, dst: Reg, src1: Reg, src2: Reg) {
    match op {
        OpKind::Add => AsmProgram::from([Inst::Fadd(dst, src1, src2)]).assemble(code),
        OpKind::Sub => AsmProgram::from([Inst::Fsub(dst, src1, src2)]).assemble(code),
        OpKind::Mul => AsmProgram::from([Inst::Fmul(dst, src1, src2)]).assemble(code),
        OpKind::Div => AsmProgram::from([Inst::Fdiv(dst, src1, src2)]).assemble(code),
        OpKind::Min => AsmProgram::from([Inst::Fmin(dst, src1, src2)]).assemble(code),
        OpKind::Max => AsmProgram::from([Inst::Fmax(dst, src1, src2)]).assemble(code),

        OpKind::Gt => AsmProgram::from([Inst::Fcmgt(dst, src1, src2)]).assemble(code),
        OpKind::Ge => AsmProgram::from([Inst::Fcmge(dst, src1, src2)]).assemble(code),
        OpKind::Lt => AsmProgram::from([Inst::Fcmgt(dst, src2, src1)]).assemble(code),
        OpKind::Le => AsmProgram::from([Inst::Fcmge(dst, src2, src1)]).assemble(code),
        OpKind::Eq => AsmProgram::from([Inst::Fcmeq(dst, src1, src2)]).assemble(code),
        OpKind::Ne => {
            AsmProgram::from([Inst::Fcmeq(dst, src1, src2), Inst::Not(dst, dst)]).assemble(code);
        }

        OpKind::IAdd => AsmProgram::from([Inst::AddI32(dst, src1, src2)]).assemble(code),
        OpKind::BitAnd => AsmProgram::from([Inst::And(dst, src1, src2)]).assemble(code),
        OpKind::BitOr => AsmProgram::from([Inst::Orr(dst, src1, src2)]).assemble(code),

        _ => unimplemented_op("aarch64", op),
    }
}

// =============================================================================
// Prologue / Epilogue
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::Assembly;

    /// `code`'s instruction words.
    fn words(code: &[u8]) -> Vec<u32> {
        code.chunks(WORD_BYTES)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect()
    }

    /// Offset 3 shifted up by a full 16-bit range: where a 16-bit slot used
    /// to wrap back to argument 3.
    const PAST_U16: u64 = 3 + (u16::MAX as u64 + 1);

    /// The uniform read for `offset = 3, dst = 5` through the block in `x9`:
    /// `ldr s5, [x9, #12]`, `dup v5.4s, v5.s[0]` (checked against `llvm-mc
    /// --disassemble`, LLVM 18). The block's address is a pointer-class
    /// value the allocator placed, so no load of it appears here: that is the
    /// `Context` def's, once per call.
    #[test]
    fn a_uniform_read_is_a_scalar_load_and_a_dup() {
        let mut code = Vec::new();
        emit_uniform_load(&mut code, Reg(5), ptr::X9, 3).expect("fits");
        assert_eq!(words(&code), [0xBD40_0D25, 0x4E04_04A5]);
    }

    /// Past the old 16-bit width the scaled immediate (4095 elements) no
    /// longer reaches, so the address is computed into IP0 in `add`-immediate
    /// steps and `[x16]` is read — the same path a deep spill frame takes.
    #[test]
    fn a_uniform_read_past_the_scaled_immediate_goes_through_ip0() {
        const PAST_U16_BYTES: u32 = 262_156;
        let mut code = Vec::new();
        emit_uniform_load(&mut code, Reg(5), ptr::X9, PAST_U16).expect("fits");
        let step = MAX_ADD_IMM;
        let full_adds = PAST_U16_BYTES / step;
        let remainder = PAST_U16_BYTES % step;
        let add = |src: u32, imm: u32| 0x9100_0000 | (imm << 10) | (src << 5) | 16;
        let mut want = alloc::vec![add(9, step)];
        want.extend(core::iter::repeat_n(add(16, step), full_adds as usize - 1));
        want.push(add(16, remainder));
        want.push(0xBD40_0000 | (16 << 5) | 5); // ldr s5, [x16]
        want.push(0x4E04_04A5); // dup v5.4s, v5.s[0]
        assert_eq!(words(&code), want);
    }

    /// The width is the encoder's, and an offset past it is refused, never
    /// wrapped: a wrapped displacement would be a load of some other
    /// argument, with plausible pixels. [`Mem`] holds a 32-bit byte offset.
    #[test]
    fn an_offset_past_the_byte_offset_is_refused() {
        const LAST: u64 = u32::MAX as u64 / 4;
        let refused = emit_uniform_load(&mut Vec::new(), Reg(0), ptr::X9, LAST + 1);
        assert!(
            matches!(refused, Err(CompileError::BudgetExceeded(_))),
            "expected a refusal, got {refused:?}"
        );
    }

    /// The lane-uniform read for `dst = 5, idx = 6` through `x9` and `x10`:
    /// `fcvtzs x10, s6`, `ldr s5, [x9, w10, uxtw #2]`, `dup v5.4s, v5.s[0]`.
    /// The base's own load is the `Context` def's, once per call.
    #[test]
    fn a_lane_uniform_read_truncates_then_loads_and_dups() {
        let mut code = Vec::new();
        emit_broadcast_load(
            &mut code,
            Reg(5),
            Reg(6),
            BroadcastGprs {
                base: ptr::X9,
                index: gpr::X10,
            },
        );
        assert_eq!(words(&code), [0x9E38_00CA, 0xBC6A_5925, 0x4E04_04A5]);
    }

    /// The one word `f` emits.
    fn shift_word(f: impl FnOnce(&mut Vec<u8>)) -> u32 {
        let mut code = Vec::new();
        f(&mut code);
        let word: [u8; WORD_BYTES] = code.as_slice().try_into().expect("one instruction");
        u32::from_le_bytes(word)
    }

    /// The immediate shifts' words, as the ARM ARM spells them: `immh:immb`
    /// is `64 - shift` for `USHR .4S` and `32 + shift` for `SHL .4S`, with
    /// `immh` = `01xx` selecting the 32-bit lane. A field one bit off decodes
    /// as `.8H` or `.2D` — a different instruction that crosses lanes — and
    /// executing a shift on the host cannot see that anywhere but aarch64.
    #[test]
    fn immediate_shifts_encode_a_32_bit_lane() {
        // ushr v0.4s, v0.4s, #23
        assert_eq!(
            shift_word(|c| emit_ushr(c, Reg(0), Reg(0), 23)),
            0x6F29_0400
        );
        // ushr v3.4s, v7.4s, #32: the widest count `.4S` holds.
        assert_eq!(
            shift_word(|c| emit_ushr(c, Reg(3), Reg(7), 32)),
            0x6F20_04E3
        );
        // shl v1.4s, v2.4s, #8
        assert_eq!(shift_word(|c| emit_shl(c, Reg(1), Reg(2), 8)), 0x4F28_5441);
        // shl v4.4s, v5.4s, #31: the widest count `.4S` holds.
        assert_eq!(shift_word(|c| emit_shl(c, Reg(4), Reg(5), 31)), 0x4F3F_54A4);
    }

    #[test]
    fn fmov_imm8_common_values() {
        // Encodable values — imm8 derived from ARM ARM bit layout:
        //   f32 = [a][NOT(b)][bbbbb][cdefgh][19 zeros]
        //   imm8 = a:b:c:d:e:f:g:h
        assert_eq!(try_encode_fmov_imm8(2.0), Some(0x00)); // 0x40000000
        assert_eq!(try_encode_fmov_imm8(0.5), Some(0x60)); // 0x3F000000
        assert_eq!(try_encode_fmov_imm8(1.0), Some(0x70)); // 0x3F800000
        assert_eq!(try_encode_fmov_imm8(1.5), Some(0x78)); // 0x3FC00000
        assert_eq!(try_encode_fmov_imm8(-1.0), Some(0xF0)); // 0xBF800000
        assert_eq!(try_encode_fmov_imm8(-0.5), Some(0xE0)); // 0xBF000000
        assert_eq!(try_encode_fmov_imm8(-2.0), Some(0x80)); // 0xC0000000
        assert_eq!(try_encode_fmov_imm8(4.0), Some(0x10)); // 0x40800000

        // More encodable values
        assert_eq!(try_encode_fmov_imm8(3.0), Some(0x08)); // 0x40400000
        assert_eq!(try_encode_fmov_imm8(0.25), Some(0x50)); // 0x3E800000
        assert_eq!(try_encode_fmov_imm8(0.125), Some(0x40)); // 0x3E000000

        // Non-encodable values
        assert_eq!(try_encode_fmov_imm8(0.0), None);
        assert_eq!(try_encode_fmov_imm8(-0.0), None);
        assert_eq!(try_encode_fmov_imm8(0.1), None);
        assert_eq!(try_encode_fmov_imm8(f32::NAN), None);
        assert_eq!(try_encode_fmov_imm8(f32::INFINITY), None);
        assert_eq!(try_encode_fmov_imm8(100.0), None);
    }

    /// Read an `ADRP`+`ADD` pair back to the buffer offset it materializes.
    ///
    /// Written from the ARM ARM's field layout rather than from `AdrpAdd`'s
    /// patch, so agreeing with it is evidence rather than a tautology.
    /// `ADRP` is `1 immlo 10000 immhi Rd` with the 21-bit immediate split
    /// across bits 30:29 and 23:5, counting *pages* from the one holding the
    /// instruction; `ADD (immediate)` is `1 0 0 100010 sh imm12 Rn Rd`.
    fn decode_adrp_add(code: &[u8], at: usize) -> (Gpr, i64) {
        let word = |i: usize| u32::from_le_bytes(code[i..i + 4].try_into().unwrap());

        let adrp = word(at);
        assert_eq!(adrp & 0x9F00_0000, 0x9000_0000, "not an ADRP: {adrp:#010x}");
        let imm21 = i64::from(((adrp >> 5) & 0x7_FFFF) << 2 | (adrp >> 29) & 0x3);
        // Sign-extend from bit 20 — the reach is ±4 GiB, not +8 GiB.
        let pages = (imm21 << 43) >> 43;
        let page_base = ((at as i64) & !0xFFF) + (pages << 12);

        let add = word(at + 4);
        assert_eq!(add & 0xFFC0_0000, 0x9100_0000, "not an ADD: {add:#010x}");
        let dst = Gpr((add & 0x1F) as u8);
        assert_eq!(
            Gpr(((add >> 5) & 0x1F) as u8),
            dst,
            "the ADD must accumulate into the register ADRP wrote"
        );

        (dst, page_base + i64::from((add >> 10) & 0xFFF))
    }

    /// The pair reaches its label whichever side of a page boundary the label
    /// falls on — which is the whole reason it is a pair, and which the
    /// `ADR`-with-a-margin scheme it replaced only ever exercised for pools
    /// past 1 MiB, i.e. never.
    #[test]
    fn adrp_add_reaches_across_pages() {
        // Distances chosen around 0x1000 so the page delta is 0, then 1, then
        // more; the last is far enough that no `ADR` would have reached it
        // under the old scheme's margin either.
        for gap in [0, 4, 0xFFC, 0x1000, 0x1004, 0x2000, 3 << 20] {
            let mut asm = Assembly::default();
            let pool = asm.mint();
            asm.push(AdrpAdd {
                dst: Gpr(17),
                target: pool,
            });
            asm.code.resize(gap, 0);
            asm.bind(pool);

            let code = asm.finish();
            assert_eq!(
                decode_adrp_add(&code, 0),
                (Gpr(17), (8 + gap) as i64),
                "gap {gap:#x}"
            );
        }
    }

    /// A label bound *before* the pair still resolves: the page delta is
    /// negative, and its 21 bits are two's complement rather than a magnitude.
    /// Nothing emits this today — the pool always trails the anchor — but the
    /// sign is the easy half of the encoding to get wrong, and it costs one
    /// test to find out here instead of the first time a label moves.
    #[test]
    fn adrp_add_reaches_backwards() {
        for gap in [0usize, 4, 0x1000, 0x2004] {
            let mut asm = Assembly::default();
            let pool = asm.mint();
            asm.bind(pool);
            asm.code.resize(gap, 0);
            asm.push(AdrpAdd {
                dst: Gpr(17),
                target: pool,
            });

            let code = asm.finish();
            assert_eq!(decode_adrp_add(&code, gap), (Gpr(17), 0), "gap {gap:#x}");
        }
    }

    #[test]
    fn fmov_imm8_roundtrip() {
        // Every valid imm8 should encode a value that round-trips
        for imm8 in 0..=255u8 {
            let a = (imm8 >> 7) & 1;
            let b = (imm8 >> 6) & 1;
            let not_b = b ^ 1;
            let cdefgh = imm8 & 0x3F;

            let mut bits: u32 = 0;
            bits |= (a as u32) << 31;
            bits |= (not_b as u32) << 30;
            // bits[29:25] = bbbbb
            let rep5 = if b == 1 { 0x1F_u32 } else { 0x00 };
            bits |= rep5 << 25;
            bits |= (cdefgh as u32) << 19;
            // bits[18:0] = 0

            let val = f32::from_bits(bits);
            let result = try_encode_fmov_imm8(val);
            assert_eq!(
                result,
                Some(imm8),
                "imm8={imm8:#04x} -> f32={val} ({bits:#010x}) did not roundtrip"
            );
        }
    }

    #[test]
    fn emit_fmov_imm_uses_single_instruction_for_encodable() {
        let mut code = Vec::new();
        let dst = Reg(0);

        // 1.0 is FMOV-encodable → should emit exactly 1 instruction (4 bytes)
        emit_fmov_imm(&mut code, dst, 1.0);
        assert_eq!(
            code.len(),
            4,
            "FMOV-encodable value should emit 1 instruction"
        );

        // Verify the encoding: 0x4F00F400 | (abc<<16) | (defgh<<5) | Rd
        // imm8=0x70=0b01110000, abc=011=3, defgh=10000=16
        let inst = u32::from_le_bytes(code[..4].try_into().unwrap());
        assert_eq!(inst, 0x4F03_F600, "FMOV V0.4S, #1.0 encoding");
    }

    #[test]
    fn emit_fmov_imm_zero_is_movi() {
        let mut code = Vec::new();
        emit_fmov_imm(&mut code, Reg(0), 0.0);
        assert_eq!(code.len(), 4, "zero should emit 1 instruction (MOVI)");
    }

    #[test]
    fn emit_fmov_imm_fallback_for_non_encodable() {
        let mut code = Vec::new();
        emit_fmov_imm(&mut code, Reg(0), core::f32::consts::PI);
        assert_eq!(
            code.len(),
            12,
            "non-encodable should emit 3 instructions (MOVZ+MOVK+DUP)"
        );
    }

    /// Encodings cross-checked against clang: `fcvtzs v28.4s, v5.4s` etc.,
    /// assembled with `clang -c -arch arm64` and dumped with objdump.
    #[test]
    fn gather_primitive_encodings() {
        fn one(f: impl FnOnce(&mut Vec<u8>)) -> u32 {
            let mut code = Vec::new();
            f(&mut code);
            assert_eq!(code.len(), 4);
            u32::from_le_bytes(code[..4].try_into().unwrap())
        }

        // fcvtzs v28.4s, v5.4s
        assert_eq!(
            one(|c| AsmProgram::from([Inst::Fcvtzs(Reg(28), Reg(5))]).assemble(c)),
            0x4EA1B8BC
        );
        // ldr x9, [x0, #8]
        assert_eq!(
            one(|c| AsmProgram::from([Inst::ldr_x(
                ptr::X9,
                Mem {
                    base: ptr::X0,
                    offset: 8,
                },
            )])
            .assemble(c)),
            0xF9400409
        );
        // ldr x9, [x0]
        assert_eq!(
            one(|c| AsmProgram::from([Inst::ldr_x(
                ptr::X9,
                Mem {
                    base: ptr::X0,
                    offset: 0,
                },
            )])
            .assemble(c)),
            0xF9400009
        );
        // umov w10, v28.s[0..3]
        assert_eq!(
            one(|c| AsmProgram::from([Inst::umov_w(Gpr(10), Reg(28), 0)]).assemble(c)),
            0x0E043F8A
        );
        assert_eq!(
            one(|c| AsmProgram::from([Inst::umov_w(Gpr(10), Reg(28), 1)]).assemble(c)),
            0x0E0C3F8A
        );
        assert_eq!(
            one(|c| AsmProgram::from([Inst::umov_w(Gpr(10), Reg(28), 2)]).assemble(c)),
            0x0E143F8A
        );
        assert_eq!(
            one(|c| AsmProgram::from([Inst::umov_w(Gpr(10), Reg(28), 3)]).assemble(c)),
            0x0E1C3F8A
        );
        // ldr w11, [x9, w10, uxtw #2]
        assert_eq!(
            one(|c| AsmProgram::from([Inst::ldr_w(
                Gpr(11),
                MemIndexed {
                    base: ptr::X9,
                    index: Gpr(10),
                },
            )])
            .assemble(c)),
            0xB86A592B
        );
        // ins v6.s[0..3], w11
        assert_eq!(
            one(|c| AsmProgram::from([Inst::ins_w(Reg(6), 0, Gpr(11))]).assemble(c)),
            0x4E041D66
        );
        assert_eq!(
            one(|c| AsmProgram::from([Inst::ins_w(Reg(6), 1, Gpr(11))]).assemble(c)),
            0x4E0C1D66
        );
        assert_eq!(
            one(|c| AsmProgram::from([Inst::ins_w(Reg(6), 2, Gpr(11))]).assemble(c)),
            0x4E141D66
        );
        assert_eq!(
            one(|c| AsmProgram::from([Inst::ins_w(Reg(6), 3, Gpr(11))]).assemble(c)),
            0x4E1C1D66
        );
    }

    #[test]
    fn gather_compound_is_four_scalar_loads() {
        let mut code = Vec::new();
        emit_gather(
            &mut code,
            Reg(6),
            Reg(28),
            ptr::X9,
            GatherGprs {
                idx: Gpr(10),
                val: Gpr(11),
            },
        );
        // 4 lanes x (umov + ldr + ins) = 12 instructions.
        assert_eq!(code.len(), 12 * 4);
    }

    /// The base register and the addressing mode are operands, so one `ldr q`
    /// covers what used to be `_voff`, `_sp` and `_x17`: the same word, with
    /// `Rn` coming from the [`Mem`] instead of from the function's name.
    #[test]
    fn the_base_register_is_an_operand_not_a_suffix() {
        fn one(f: impl FnOnce(&mut Vec<u8>)) -> u32 {
            let mut code = Vec::new();
            f(&mut code);
            assert_eq!(code.len(), 4, "aarch64 instructions are fixed-width");
            u32::from_le_bytes(code[..4].try_into().unwrap())
        }
        // ldr q0, [x0, #32] / [sp, #32] / [x17, #32] — one encoder, three bases.
        for base in [ptr::X0, ptr::SP, ptr::X17] {
            let word = one(|c| {
                AsmProgram::from([Inst::ldr_q(Reg(0), Mem { base, offset: 32 })]).assemble(c)
            });
            assert_eq!(word & !(0x1F << 5), 0x3DC0_0800, "same instruction");
            assert_eq!((word >> 5) & 0x1F, u32::from(base.0), "Rn is the base");
        }
        // str q1, [sp, #48]
        assert_eq!(
            one(|c| AsmProgram::from([Inst::str_q(
                Reg(1),
                Mem {
                    base: ptr::SP,
                    offset: 48
                }
            )])
            .assemble(c)),
            0x3D80_0FE1
        );
    }

    /// An offset past the 12-bit scaled immediate is computed into IP0 first,
    /// in `add`-immediate-sized steps, and the transfer then reads `[x16]`.
    #[test]
    fn a_deep_frame_addresses_through_ip0() {
        let mut code = Vec::new();
        // 65536 = 16 * 4096, one slot past the largest encodable displacement.
        AsmProgram::from([Inst::ldr_q(
            Reg(3),
            Mem {
                base: ptr::SP,
                offset: 65536,
            },
        )])
        .assemble(&mut code);
        let words: Vec<u32> = code
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        // 65536 = 16 * 4080 + 256, so sixteen full adds plus the remainder.
        assert_eq!(words.len(), 17 + 1, "adds then the load");
        assert_eq!(words[0], 0x9100_0000 | (4080 << 10) | (31 << 5) | 16);
        assert_eq!(words[1], 0x9100_0000 | (4080 << 10) | (16 << 5) | 16);
        assert_eq!(*words.last().unwrap(), 0x3DC0_0000 | (16 << 5) | 3);
    }
}

// =============================================================================
// The NEON `IsaBackend` driver
// =============================================================================

/// The aarch64 half of code generation: the [`IsaBackend`](crate::emit::IsaBackend)
/// implementation and the constant pool it needs.
///
/// **This file is where aarch64-specific bugs live, and the only place they
/// can.** Emission is a pure function into `Vec<u8>`, so everything here
/// compiles, typechecks and is swept for op coverage on every host — an x86
/// machine computes NEON instruction words perfectly well. Only
/// `compile_native` in `emit` decides which backend a process instantiates
/// — from the tier `crate::isa` read off the CPU — and only
/// [`executable`](crate::emit::executable) needs the matching CPU.
///
/// The consequence worth stating: a change that does not touch an ISA file
/// cannot introduce a platform-specific bug. That is the same bargain `unsafe`
/// makes — confine what cannot be checked, so the rest is checked by
/// construction.
pub(super) mod driver {
    use super::super::*;
    use super::Mem;
    use super::ptr;
    use super::ptr::*;
    use super::*;
    use crate::error::CompileError;
    use alloc::vec::Vec;

    /// Constant pool: maps f32 bit patterns to pool indices.
    ///
    /// Non-zero, non-FMOV-encodable constants are stored in a data section after
    /// the RET instruction. Each entry is 16 bytes (the f32 splatted 4x to fill
    /// a 128-bit NEON register). During code emission, these constants are loaded
    /// with a single `LDR Qt, [X17, #offset]` instead of the 3-instruction
    /// MOVZ+MOVK+DUP sequence.
    struct ConstPool {
        /// Deduplicated entries, in pool order.
        entries: Vec<PoolEntry>,
        /// Map from entry → pool index.
        index: alloc::collections::BTreeMap<PoolEntry, u16>,
    }
    impl ConstPool {
        /// Create an empty constant pool.
        fn new() -> Self {
            Self {
                entries: Vec::new(),
                index: alloc::collections::BTreeMap::new(),
            }
        }

        /// Insert an f32 into the pool as a splat (deduplicating by bit
        /// pattern) and return the byte offset for an `LDR Qt, [X17,
        /// #offset]` load.
        ///
        /// Zero and FMOV-encodable constants are NOT filtered here — callers
        /// that want the fast path should check `needs_const_pool` first.
        /// Builtin emitters call this unconditionally because every constant
        /// they use benefits from the pool (they are transcendental coefficients,
        /// never zero or FMOV-encodable).
        fn push_f32(&mut self, val: f32) -> Result<u16, CompileError> {
            self.push(PoolEntry::from([val.to_bits(); 4]))
        }

        /// Insert one register's worth of words, deduplicated, and return
        /// the byte offset for an `LDR Qt, [X17, #offset]` load.
        fn push(&mut self, entry: PoolEntry) -> Result<u16, CompileError> {
            if let Some(&idx) = self.index.get(&entry) {
                return Ok(idx * 16);
            }
            let idx = self.entries.len();
            if idx >= 4096 {
                return Err(CompileError::BudgetExceeded(
                    "constant pool overflow: exceeded 12-bit LDR offset limit (max 4095 entries)",
                ));
            }
            self.entries.push(entry);
            self.index.insert(entry, idx as u16);
            Ok((idx * 16) as u16)
        }

        /// Get the byte offset for an entry, or None if it's not in the pool.
        fn offset_for(&self, entry: PoolEntry) -> Option<u16> {
            self.index.get(&entry).map(|&idx| idx * 16)
        }
    }
    /// Emit a constant load, using the constant pool when available.
    ///
    /// Falls back to `emit_fmov_imm` for zero and FMOV-encodable values.
    fn emit_const_load(code: &mut Vec<u8>, dst: Reg, val_bits: u32, pool: &ConstPool) {
        if let Some(offset) = pool.offset_for([val_bits; 4]) {
            AsmProgram::from([Inst::ldr_q(
                dst,
                Mem {
                    base: ptr::X17,
                    offset: offset.into(),
                },
            )])
            .assemble(code);
        } else {
            super::emit_fmov_imm(code, dst, f32::from_bits(val_bits));
        }
    }
    /// The address of a frame slot. Kernels are leaf functions with no frame
    /// pointer, so every slot — spill or scaffold — is `sp` plus its offset;
    /// naming that here keeps `sp` a fact about the frame instead of a suffix
    /// on the load and store that reach it.
    const fn frame_slot(offset: u32) -> Mem {
        Mem {
            base: ptr::SP,
            offset,
        }
    }

    /// The aarch64 (NEON) register file.
    ///
    /// AAPCS64 callee-saves the low 64 bits of v8-v15. These kernels are leaf
    /// functions emitted with no prologue that preserves them, so the scratch pool
    /// must steer clear of that range entirely — handing the allocator one of
    /// v8-v15 would silently corrupt whatever the *caller* had live there across
    /// the JIT call.
    ///
    ///   v8-v15:  callee-saved, never allocatable
    ///   v0-v7, v16-v31: allocatable scratch — everything else
    const AARCH64_FILE: regalloc::RegisterFile = regalloc::RegisterFile {
        // v0-v7 and v16-v31: twenty-four of thirty-two. AAPCS64 callee-saves
        // the low 64 bits of v8-v15 and these are leaf kernels with no
        // prologue that preserves them, so v8-v15 stay out. v0-v3 are the
        // last to join: they carried the coordinate vectors of the per-batch
        // ABI, which a call no longer passes.
        scratch: regalloc::RegSet::range(16, 16).union(regalloc::RegSet::range(0, 8)),
        // Nothing. v30 is the gather's truncated-index register, a `temps_for`
        // answer since the gathers landed; v29 used to be `UNARY_SCRATCH`,
        // reserved whole-kernel so a reciprocal estimate could borrow it. The
        // `If` needs none either: `BSL` reads its three operands directly,
        // and `FNEG`/`FABS` are single instructions.
        fixed: &[],
        temps_for: super::temps_for,
        // `UMAXV`/`UMINV` reduce the mask into a vector register before
        // `FMOV` can move it to a general one. It used to be v28, held out of
        // every kernel's pool; it is now a reservation on the instruction the
        // guard is emitted before.
        guard_temps: 1,
        vector_bytes: 16,
        // AAPCS64's first three integer arguments, in the ABI's order: the
        // context (the array of buffer base pointers, then the uniform and
        // origin blocks), the output plane, its pitch. Declaring them here is
        // what lets `checked` prove `gpr_scratch` misses all three, rather
        // than a comment asserting the constants never collide.
        gpr_ctx: Some(ptr::X0.as_gpr()),
        gpr_out: Some(ptr::X1.as_gpr()),
        gpr_pitch: Some(gpr::X2),
        // x9-x11: the gather's base pointer and per-lane index/value GPRs,
        // the store's row and column — chosen by hand before this work and
        // now `Scratch` reservations — clear of the branch guard (w16) and
        // the const-pool anchor (x17).
        gpr_scratch: regalloc::GprSet::of(&[ptr::X9.as_gpr(), gpr::X10, gpr::X11]),
        gpr_temps_for: super::gpr_temps_for,
        // x3-x8 and x12-x15: the caller-saved GPRs AAPCS64 leaves after the
        // three arguments, the three scratch, the intra-procedure pair
        // (x16 the branch guard, x17 the const-pool anchor) and the
        // platform register x18. The pointer class's pool — buffer bases
        // and block addresses are carried here across the loops that read
        // them (docs/plans/2026-09-22-a-pointer-is-a-value.md).
        pointers: regalloc::GprSet::of(&[
            gpr::X3,
            gpr::X4,
            gpr::X5,
            gpr::X6,
            gpr::X7,
            gpr::X8,
            gpr::X12,
            gpr::X13,
            gpr::X14,
            gpr::X15,
        ]),
        // No mask-register file on this tier: masks are ordinary vectors.
        mask_scratch: regalloc::MaskSet::EMPTY,
        mask_temps_for: regalloc::no_temps,
        mask_guard_temps: 0,
    }
    .checked();

    /// The register a guard reduces its mask into.
    ///
    /// `UMAXV`/`UMINV` write a scalar into a vector register, so this tier's
    /// guard needs one that is neither the mask nor anything live — which is
    /// what `RegisterFile::guard_temps` asks the allocator for, and what makes
    /// the two assertions here statements about the allocator rather than
    /// about a hand-picked constant.
    fn guard_scratch(scratch: Option<Reg>, mask_reg: Reg) -> Reg {
        let scratch = scratch
            .expect("aarch64's guard declares `guard_temps: 1`; the allocator owes it a register");
        debug_assert_ne!(scratch, mask_reg, "the reduce would destroy its own input");
        scratch
    }

    pub(in crate::emit) struct Aarch64Backend {
        consts: ConstPool,
    }

    impl Aarch64Backend {
        pub(in crate::emit) fn new() -> Self {
            Self {
                consts: ConstPool::new(),
            }
        }
    }

    impl IsaBackend for Aarch64Backend {
        fn jump(&mut self, asm: &mut Assembly, label: Label) {
            asm.push(B { target: label });
        }

        fn register_file(&self) -> regalloc::RegisterFile {
            AARCH64_FILE
        }

        fn begin(&mut self, schedule: &[regalloc::Def]) -> Result<(), CompileError> {
            // Seed by APPENDING into the existing pool, never replacing it: a
            // compile emits every scope of the nest through one backend, and
            // an outer scope's bytes have the pool's X17-relative offsets
            // baked in — resetting here left them pointing into an inner
            // scope's rebuilt pool (wrong constants; the macOS glyph-ink
            // regression). `push` dedups, and each compile constructs a fresh
            // backend, so appending is reset-equivalent for a single scope.
            for def in schedule {
                match def.op {
                    ScheduledOp::Const(val) if super::needs_const_pool(val) => {
                        self.consts.push_f32(val)?;
                    }
                    ScheduledOp::Lanes(_) => {
                        self.consts.push(IOTA)?;
                    }
                    _ => {}
                }
            }
            // Builtins add up to ~60 polynomial coefficients during emission; bail
            // if the expression constants + headroom would exceed the 12-bit LDR
            // offset limit.
            const BUILTIN_HEADROOM: usize = 128;
            if self.consts.entries.len() + BUILTIN_HEADROOM > 4095 {
                return Err(CompileError::BudgetExceeded(
                    "expression too large: constant pool would exceed 12-bit LDR offset limit",
                ));
            }
            Ok(())
        }

        fn emit_plan(
            &mut self,
            code: &mut Vec<u8>,
            plan: &InstructionPlan,
        ) -> Result<(), CompileError> {
            emit_instruction_plan(code, plan, &mut self.consts)
        }

        fn emit_mov(&mut self, code: &mut Vec<u8>, dst: Reg, src: Reg) {
            AsmProgram::from([Inst::mov(dst, src)]).assemble(code);
        }

        fn emit_store(
            &mut self,
            code: &mut Vec<u8>,
            src: Reg,
            offset: u32,
        ) -> Result<(), CompileError> {
            AsmProgram::from([Inst::str_q(src, frame_slot(offset))]).assemble(code);
            Ok(())
        }

        fn emit_resolve(
            &mut self,
            code: &mut Vec<u8>,
            vid: regalloc::ValueId,
            target: Reg,
            locs: &[Option<Binding>],
        ) -> Result<Reg, CompileError> {
            match location_of(locs, vid) {
                Binding::Loc(Loc::Reg(reg)) => Ok(reg),
                Binding::Remat(bits) => {
                    emit_const_load(code, target, bits, &self.consts);
                    Ok(target)
                }
                Binding::Loc(Loc::Slot(slot)) => {
                    AsmProgram::from([Inst::ldr_q(target, frame_slot(slot.offset()))])
                        .assemble(code);
                    Ok(target)
                }
                Binding::Loc(Loc::Ptr(p)) => {
                    unreachable!("{vid:?} is an address in {p:?}; the pointer class resolves it")
                }
            }
        }

        fn ptr_store(&mut self, code: &mut Vec<u8>, src: PtrReg, offset: u32) {
            AsmProgram::from([Inst::str_x(src, frame_slot(offset))]).assemble(code);
        }

        fn ptr_load(&mut self, code: &mut Vec<u8>, dst: PtrReg, offset: u32) {
            AsmProgram::from([Inst::ldr_x(dst, frame_slot(offset))]).assemble(code);
        }

        fn ptr_mov(&mut self, code: &mut Vec<u8>, dst: PtrReg, src: PtrReg) {
            AsmProgram::from([table::MovX::new(dst, src)]).assemble(code);
        }

        /// `scratch` is this instruction's own reservation, live for these two
        /// instructions only — the allocator makes it because this backend's
        /// `guard_temps` asks for one.
        ///
        /// Both polarities end in [`BranchIfW16Zero`] — `cbnz w16, .+8; b
        /// label`, two words whatever the arm's length — so the arm chooses
        /// the *reduction*: a horizontal max is zero exactly when no lane is
        /// set, and an inverted horizontal min is zero exactly when every lane
        /// is.
        fn branch_if_arm_is_dead(&mut self, asm: &mut Assembly, test: MaskTest, label: Label) {
            let scratch = guard_scratch(test.scratch, test.reg);
            match test.arm {
                IfArm::True => {
                    AsmProgram::from([Inst::Umaxv(scratch, test.reg), Inst::FmovToGp(scratch)])
                        .assemble(&mut asm.code);
                }
                IfArm::False => {
                    AsmProgram::from([
                        Inst::Uminv(scratch, test.reg),
                        Inst::FmovToGp(scratch),
                        Inst::mvn_w(X16, X16),
                    ])
                    .assemble(&mut asm.code);
                }
            }
            asm.push(BranchIfW16Zero { target: label });
        }

        // AAPCS64: x0 = ctx (read-only in the body's gathers and uniform
        // loads), x1 = out, x2 = pitch; the body's scratch GPRs are x9-x11
        // (gather, store address), w16 (branch tests), x17 (pool anchor) —
        // all disjoint, and `checked` says so for the ones it can see.

        fn frame_alloc(&mut self, code: &mut Vec<u8>, bytes: u32) {
            let mut remaining = bytes;
            while remaining > 0 {
                let chunk = remaining.min(table::MAX_ADD_IMM);
                AsmProgram::from([table::SubI64::new(
                    ptr::SP,
                    ptr::SP,
                    table::Imm12(chunk as u16),
                )])
                .assemble(code);
                remaining -= chunk;
            }
        }

        fn frame_free(&mut self, code: &mut Vec<u8>, bytes: u32) {
            let mut remaining = bytes;
            while remaining > 0 {
                let chunk = remaining.min(table::MAX_ADD_IMM);
                AsmProgram::from([table::AddI64::new(
                    ptr::SP,
                    ptr::SP,
                    table::Imm12(chunk as u16),
                )])
                .assemble(code);
                remaining -= chunk;
            }
        }

        /// Every scope's constant loads are X17-relative, so the anchor has
        /// to be inside the emitted function, after the frame.
        fn anchor(&mut self, asm: &mut Assembly, pool: Label) {
            asm.push(AdrpAdd {
                dst: X17.into(),
                target: pool,
            });
        }

        /// Append the constant pool after the final `RET`.
        ///
        /// `anchor` branches to `pool` unconditionally — whether
        /// this compile needed the pool is not known until every constant
        /// has been emitted — so the name is bound even when there is nothing
        /// to append: the assembler panics on a name nobody wrote, and an
        /// unpatched `AdrpAdd` would leave X17 pointing at itself, same as
        /// the unpatched `ADR` this replaced.
        fn finish(&mut self, asm: &mut Assembly, pool: Label) {
            let mut entries = Vec::new();
            for &entry in &self.consts.entries {
                super::emit_pool_entry(&mut entries, entry);
            }
            asm.pool(pool, entries);
        }

        fn slot_store(&mut self, code: &mut Vec<u8>, src: Reg, offset: u32) {
            AsmProgram::from([Inst::str_q(src, frame_slot(offset))]).assemble(code);
        }

        fn slot_load(&mut self, code: &mut Vec<u8>, dst: Reg, offset: u32) {
            AsmProgram::from([Inst::ldr_q(dst, frame_slot(offset))]).assemble(code);
        }

        fn add_scalar(
            &mut self,
            code: &mut Vec<u8>,
            dst: Reg,
            scratch: Reg,
            scalar: f32,
        ) -> Result<(), CompileError> {
            super::emit_fmov_imm(code, scratch, scalar);
            AsmProgram::from([Inst::Fadd(dst, dst, scratch)]).assemble(code);
            Ok(())
        }

        fn load_const(
            &mut self,
            code: &mut Vec<u8>,
            dst: Reg,
            val: f32,
        ) -> Result<(), CompileError> {
            super::emit_fmov_imm(code, dst, val);
            Ok(())
        }

        fn alu(&mut self, code: &mut Vec<u8>, op: OpKind, dst: Reg, srcs: [Reg; 2]) {
            super::emit_binary(code, op, dst, srcs[0], srcs[1]);
        }

        /// A full batch is one `STR Q`. A remainder is `ST1 {V.S}[k]` per
        /// lane, post-indexed by the lane's four bytes.
        fn emit_write(&mut self, code: &mut Vec<u8>, write: &WritePlan) {
            let row = crate::emit::declared_gpr_temp(write.scratch.gpr_temp(0));
            let col = crate::emit::declared_gpr_temp(write.scratch.gpr_temp(1));
            let via = crate::emit::declared_temp(write.scratch.temp(0));
            let out = AARCH64_FILE
                .gpr_out
                .expect("NEON's store needs the output pointer");
            let pitch = AARCH64_FILE
                .gpr_pitch
                .expect("NEON's store needs the pitch");
            index_into(code, row, write.row, via);
            index_into(code, col, write.col, via);
            AsmProgram::from([
                // madd row, row, pitch, col
                Inst::Raw(
                    0x9B00_0000
                        | (u32::from(pitch.0) << 16)
                        | (u32::from(col.0) << 10)
                        | (u32::from(row.0) << 5)
                        | u32::from(row.0),
                ),
                // add row, out, row, lsl #2
                Inst::Raw(
                    0x8B00_0000
                        | (u32::from(row.0) << 16)
                        | (2 << 10)
                        | (u32::from(out.0) << 5)
                        | u32::from(row.0),
                ),
            ])
            .assemble(code);
            if write.lanes == 4 {
                AsmProgram::from([Inst::str_q(
                    write.value,
                    Mem {
                        base: PtrReg(row.0),
                        offset: 0,
                    },
                )])
                .assemble(code);
                return;
            }
            for lane in 0..write.lanes {
                // st1 {value.s}[lane], [row], #4 — the lane index is Q:S.
                let q = (lane >> 1) & 1;
                let s_bit = lane & 1;
                AsmProgram::from([Inst::Raw(
                    0x0D9F_8000
                        | (q << 30)
                        | (s_bit << 12)
                        | (u32::from(row.0) << 5)
                        | u32::from(write.value.0),
                )])
                .assemble(code);
            }
        }

        fn emit_ret(&mut self, code: &mut Vec<u8>) {
            AsmProgram::from([Inst::Ret]).assemble(code);
        }
    }

    /// `dst = trunc(index)` as a 64-bit integer, wherever a fold keeps its
    /// binder: `FCVTZS` from lane 0 of its register, or of `via` after a
    /// scalar load when the allocator left it in a slot.
    fn index_into(code: &mut Vec<u8>, dst: Gpr, at: Binding, via: Reg) {
        let from = match at {
            Binding::Loc(Loc::Reg(r)) => r,
            Binding::Loc(Loc::Slot(slot)) => {
                AsmProgram::from([Inst::ldr_s(via, frame_slot(slot.offset()))]).assemble(code);
                via
            }
            Binding::Loc(Loc::Ptr(_)) => unreachable!("a fold's binder is a vector"),
            // `emit_scope` hands a rematerialized binder over as its slot, so
            // the only caller, `emit_write`, never holds a constant here.
            Binding::Remat(bits) => unreachable!(
                "a fold's binder is read from a register or a slot, never rematerialized ({bits:#x})"
            ),
        };
        AsmProgram::from([Inst::FcvtzsX { dst, src: from }]).assemble(code);
    }
    /// Emit machine code for a resolved instruction plan.
    ///
    /// This is a DETERMINISTIC DISPATCH: given a plan, emit the exact
    /// instructions. No decisions are made here — all decisions were
    /// made by resolve_operands.
    fn emit_instruction_plan(
        code: &mut Vec<u8>,
        plan: &InstructionPlan,
        pool: &mut ConstPool,
    ) -> Result<(), CompileError> {
        use super::*;

        for reload in &plan.reloads {
            match reload {
                Reload::FromStack { target, slot } => {
                    AsmProgram::from([Inst::ldr_q(*target, frame_slot(slot.offset()))])
                        .assemble(code);
                }
                Reload::Const { target, val_bits } => {
                    emit_const_load(code, *target, *val_bits, pool);
                }
                Reload::Ptr { target, slot } => {
                    AsmProgram::from([Inst::ldr_x(*target, frame_slot(slot.offset()))])
                        .assemble(code);
                }
            }
        }

        if let Some((dst, src)) = plan.setup_mov {
            AsmProgram::from([Inst::mov(dst, src)]).assemble(code);
        }

        match &plan.op {
            ResolvedOp::Nop => {}
            ResolvedOp::LoadConst { dst, val_bits } => {
                emit_const_load(code, *dst, *val_bits, pool);
            }
            // The iota is one pool entry, seeded by `begin`.
            ResolvedOp::Lanes { dst } => {
                let offset = pool
                    .offset_for(IOTA)
                    .expect("begin seeds the iota for every Lanes def");
                AsmProgram::from([Inst::ldr_q(
                    *dst,
                    Mem {
                        base: ptr::X17,
                        offset: offset.into(),
                    },
                )])
                .assemble(code);
            }
            ResolvedOp::Unary { op, dst, src } => {
                // The shared driver's `Unary`, not `table::Unary` (an
                // encoding row), which this module also sees.
                emit_unary(
                    code,
                    crate::emit::Unary {
                        op: *op,
                        dst: *dst,
                        src: *src,
                        temp: plan.scratch.temp(0),
                    },
                );
            }
            ResolvedOp::ShiftImm {
                op,
                dst,
                src,
                amount,
            } => {
                super::emit_shift_imm(code, *op, *dst, *src, *amount);
            }
            ResolvedOp::Gather { dst, idx, base } => {
                // dst = base[idx], via four scalar loads (NEON has no native
                // gather). `base` is the buffer's address wherever the
                // allocator keeps it; x10/x11 are `AARCH64_FILE.gpr_scratch`'s
                // reservations for this instruction, clear of the branch
                // guard (w16) and the const-pool anchor (x17).
                let idx_int = crate::emit::declared_temp(plan.scratch.temp(0));
                AsmProgram::from([Inst::Fcvtzs(idx_int, *idx)]).assemble(code);
                super::emit_gather(
                    code,
                    *dst,
                    idx_int,
                    *base,
                    super::GatherGprs {
                        idx: crate::emit::declared_gpr_temp(plan.scratch.gpr_temp(0)),
                        val: crate::emit::declared_gpr_temp(plan.scratch.gpr_temp(1)),
                    },
                );
            }
            ResolvedOp::Broadcast { dst, idx, base } => {
                super::emit_broadcast_load(
                    code,
                    *dst,
                    *idx,
                    super::BroadcastGprs {
                        base: *base,
                        index: crate::emit::declared_gpr_temp(plan.scratch.gpr_temp(0)),
                    },
                );
            }
            ResolvedOp::Uniform { dst, base, offset } => {
                super::emit_uniform_load(code, *dst, *base, *offset)?;
            }
            ResolvedOp::Context { dst, slot } => {
                // The one read of the context pointer (x0 per AAPCS64):
                // `ldr dst, [x0, #slot*8]`, once per call for a value the
                // allocator then carries or parks like any other.
                AsmProgram::from([Inst::ldr_x(
                    *dst,
                    Mem {
                        base: ptr::X0,
                        offset: u32::from(*slot) * X_BYTES,
                    },
                )])
                .assemble(code);
            }
            ResolvedOp::Binary {
                op,
                dst,
                left,
                right,
            } => {
                // Every transcendental is expanded to arithmetic by
                // `expand_transcendentals` before codegen, so only primitives
                // reach here.
                emit_binary(code, *op, *dst, *left, *right);
            }
            ResolvedOp::FusedMulAdd { dst, a, b } => {
                // setup_mov already placed c into dst
                AsmProgram::from([Inst::Fmla(*dst, *a, *b)]).assemble(code);
            }
            ResolvedOp::If {
                dst,
                if_true,
                if_false,
            } => {
                // setup_mov already placed mask into dst
                AsmProgram::from([Inst::Bsl(*dst, *if_true, *if_false)]).assemble(code);
            }
        }

        Ok(())
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::emit::regalloc;

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
            assert!(needs_const_pool(first));
            assert!(needs_const_pool(second));

            let mut backend = Aarch64Backend::new();
            crate::emit::tests::emit_dag_body(scope_for(first), &mut backend).expect("first scope");
            let after_first = backend.consts.entries.to_vec();
            assert!(!after_first.is_empty(), "the first scope pooled nothing");
            crate::emit::tests::emit_dag_body(scope_for(second), &mut backend)
                .expect("second scope");

            assert!(
                backend.consts.entries.starts_with(&after_first),
                "the second scope RESET the constant pool: the first scope's \
                 baked-in X17-relative offsets now name different constants — the \
                 glyph-ink regression. Pool was {after_first:?}, became {:?}",
                backend.consts.entries
            );
        }
    }
}

// =============================================================================
// =============================================================================
// General-purpose and Pointer registers
// =============================================================================

/// Physical pointer registers used by AAPCS64 emitted kernels.
mod ptr {
    use super::PtrReg;

    /// 1st argument: context pointer — array of bound buffer bases, then
    /// the uniform and origin blocks.
    pub(super) const X0: PtrReg = PtrReg(0);
    /// 2nd argument: the output plane.
    pub(super) const X1: PtrReg = PtrReg(1);
    /// Scratch: a gather's base pointer, a store's address.
    pub(super) const X9: PtrReg = PtrReg(9);
    /// IP0, intra-procedure scratch (displacement fallback).
    pub(super) const X16: PtrReg = PtrReg(16);
    /// IP1, intra-procedure scratch (constant-pool anchor).
    pub(super) const X17: PtrReg = PtrReg(17);
    /// The stack pointer — spill slots are addressed from it.
    pub(super) const SP: PtrReg = PtrReg(31);
}

/// AAPCS64 general-purpose registers (integers, indices, the pitch).
mod gpr {
    use super::Gpr;

    /// 3rd argument: the pitch.
    pub(super) const X2: Gpr = Gpr(2);
    /// The pointer pool (`RegisterFile::pointers`), and the encoders' tests.
    pub(super) const X3: Gpr = Gpr(3);
    pub(super) const X4: Gpr = Gpr(4);
    pub(super) const X5: Gpr = Gpr(5);
    pub(super) const X6: Gpr = Gpr(6);
    pub(super) const X7: Gpr = Gpr(7);
    pub(super) const X8: Gpr = Gpr(8);
    /// Scratch: a gather's index, a store's column.
    pub(super) const X10: Gpr = Gpr(10);
    /// Scratch: a gather's value.
    pub(super) const X11: Gpr = Gpr(11);
    /// The pointer pool, continued past the scratch.
    pub(super) const X12: Gpr = Gpr(12);
    pub(super) const X13: Gpr = Gpr(13);
    pub(super) const X14: Gpr = Gpr(14);
    pub(super) const X15: Gpr = Gpr(15);
}

// =============================================================================
// Branches as program items
// =============================================================================

/// Where a branch keeps its displacement, and how far it reaches.
///
/// A64 has no single `rel32`: `B` carries a 26-bit word displacement in the low
/// bits, and `B.cond`/`CBZ`/`CBNZ` carry a 19-bit one starting at bit 5. Two
/// fields, two ranges — so which one a branch uses is part of what the branch
/// *is*, and a branch whose reach must not depend on what it spans is a `B`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct DispField {
    /// Bit position of the field's low end.
    shift: u32,
    /// Field width in bits.
    bits: u32,
}

impl DispField {
    /// `B`'s imm26, at bit 0 — ±128 MiB.
    const IMM26: Self = Self { shift: 0, bits: 26 };
    /// `B.cond`'s and `CBZ`/`CBNZ`'s imm19, at bit 5 — ±1 MiB.
    const IMM19: Self = Self { shift: 5, bits: 19 };

    /// Overwrite this field of the instruction word at `at` so the branch
    /// reaches `target`, leaving every other bit — opcode, condition, register
    /// — exactly as placed.
    ///
    /// # Panics
    ///
    /// If the displacement is not a whole number of instructions, or does not
    /// fit. The first is a bug in this crate; the second is a real limit of the
    /// encoding. A guard no longer meets it: [`BranchIfW16Zero`] jumps with a
    /// `B`, ±128 MiB (2026-10-05; before that an arm past `CBZ`'s 262,144
    /// words panicked here), and no emitter verb branches with a `B.cond`.
    fn write(self, code: &mut [u8], at: usize, target: usize) {
        let bytes = target as i64 - at as i64;
        assert!(
            bytes % 4 == 0,
            "aarch64 branch displacement is not a whole number of instructions"
        );
        let words = bytes / 4;
        let limit = 1i64 << (self.bits - 1);
        assert!(
            (-limit..limit).contains(&words),
            "branch displacement {words} does not fit {} bits",
            self.bits
        );
        let mask = ((1u32 << self.bits) - 1) << self.shift;
        let field = ((words as u32) << self.shift) & mask;
        let existing = u32::from_le_bytes([code[at], code[at + 1], code[at + 2], code[at + 3]]);
        code[at..at + 4].copy_from_slice(&((existing & !mask) | field).to_le_bytes());
    }
}

/// `b target` — an unconditional branch to a [`Label`], ±128 MiB.
///
/// A struct, like every other instruction here, and its label is an operand
/// like any other. It emits a zero displacement; the assembler writes the real
/// one once the label lands, which is what [`AsmInsn::label_ref`] tells it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct B {
    /// Where it goes.
    target: Label,
}

impl AsmInsn for B {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        emit32(code, 0x1400_0000);
    }

    #[inline]
    fn label_ref(self) -> Option<LabelRef> {
        Some(LabelRef {
            at: 0,
            label: self.target,
            patch: |code, at, target| DispField::IMM26.write(code, at, target),
        })
    }
}

/// The branch-test scratch's register number — W16, which the guard path
/// reduces a mask into with `umaxv`/`uminv` + `fmov`.
const BRANCH_TEST_REG: u32 = 16;

/// `CBZ Wt`'s opcode with the register and displacement fields clear:
/// `sf = 0` (a W register) and `011010`, then `op` in bit 24.
const CBZ_W_OPCODE: u32 = 0x3400_0000;
/// `op`, bit 24 of `CBZ`/`CBNZ`: clear branches on zero, set on non-zero. The
/// whole difference between the two instructions.
const CB_OP_NONZERO: u32 = 1 << 24;

/// How far the `CBNZ` of a [`BranchIfW16Zero`] jumps, in words: over the `B`
/// that follows it, to the instruction after the pair.
const CBNZ_SKIPS_B: u32 = 2;

/// `cbnz w16, .+8` — the first word of a [`BranchIfW16Zero`]. Never patched:
/// its displacement is the pair's own size, not a label's position.
const CBNZ_W16_OVER_B: u32 =
    CBZ_W_OPCODE | CB_OP_NONZERO | (CBNZ_SKIPS_B << DispField::IMM19.shift) | BRANCH_TEST_REG;

/// `cbnz w16, .+8` + `b target`, sharing one [`Label`]: branch to `target`
/// when W16 is zero, from anywhere within `B`'s ±128 MiB.
///
/// **Two instructions, because `CBZ` does not reach.** It spends 19 bits on a
/// word displacement — ±1 MiB — and an arm is as long as the code it owns, a
/// bound nothing here puts under 1 MiB: a text run's glyph kernels under one
/// guard are megabytes of NEON. `CBNZ` steps over the `B` when the mask says
/// the arm is live, and falls into the `B` when it says the arm is dead, so
/// this is exactly `cbz w16, target` at `B`'s reach.
///
/// **Always both, never the one-word `CBZ` where it would reach.** Which form
/// fits depends on the arm's length, which depends on where every instruction
/// inside it landed, and this instruction's own size is one of those — so
/// choosing the short form is branch relaxation, layout iterated to a fixed
/// point to save four bytes per guard. The pair's size is fixed before a
/// single byte is laid out, like `AdrpAdd`'s, and there is nothing to relax.
///
/// W16 rather than a register operand because W16 *is* the branch-test scratch
/// in this backend's ABI: the guard path reduces a mask into it with
/// `umaxv`/`uminv` + `fmov`, and nothing else may hold a value there. A
/// register parameter would suggest a choice the ABI does not offer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct BranchIfW16Zero {
    /// Where it goes.
    target: Label,
}

impl AsmInsn for BranchIfW16Zero {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        emit32(code, CBNZ_W16_OVER_B);
        // The jump is `B`'s own word, so there is one encoding of `B` here and
        // `label_ref` below only says where its displacement lives.
        B {
            target: self.target,
        }
        .emit_into(code);
    }

    #[inline]
    fn label_ref(self) -> Option<LabelRef> {
        Some(LabelRef {
            // The `CBNZ` is the first word; the `B` it steps over sits one
            // word later, and a displacement is measured from the branch's
            // own address.
            at: WORD_BYTES,
            label: self.target,
            patch: |code, at, target| DispField::IMM26.write(code, at, target),
        })
    }
}

#[cfg(test)]
mod label_tests {
    use super::*;
    use crate::emit::{Assembly, IfArm, IsaBackend, MaskTest};

    /// One known word, so a test can measure distances in instructions without
    /// depending on any real encoding.
    const NOP: Inst = Inst::Raw(0xD503_201F);

    /// The one-word `cbz w16, .+0` that a guard used to end in: `Rt = 16`,
    /// `op = 0`. What the pair replaced, spelled out so the test can say what
    /// changed about it.
    const OLD_CBZ_W16: u32 = 0x3400_0010;

    fn word_at(code: &[u8], at: usize) -> u32 {
        u32::from_le_bytes([code[at], code[at + 1], code[at + 2], code[at + 3]])
    }

    #[test]
    fn forward_branch_counts_instructions_not_bytes() {
        let mut asm = Assembly::default();
        let end = asm.mint();
        asm.push(B { target: end });
        asm.push(NOP);
        asm.push(NOP);
        asm.bind(end);
        let code = asm.finish();
        // Three instructions ahead of the branch's own address.
        assert_eq!(word_at(&code, 0) & 0x03FF_FFFF, 3);
    }

    #[test]
    fn a_back_edge_is_negative() {
        let mut asm = Assembly::default();
        let top = asm.mint();
        asm.bind(top);
        asm.push(NOP);
        asm.push(NOP);
        asm.push(B { target: top });
        let code = asm.finish();
        // The branch sits two instructions past the label, so -2 words, in
        // imm26's two's complement.
        assert_eq!(
            word_at(&code, 8) & 0x03FF_FFFF,
            (-2i32 as u32) & 0x03FF_FFFF
        );
    }

    /// A branch displacement field read back with its sign, from the ARM ARM's
    /// layout rather than from `DispField`, so agreeing with it is evidence.
    fn imm19(word: u32) -> i64 {
        i64::from((((word >> 5) & 0x7_FFFF) << 13) as i32 >> 13)
    }

    fn imm26(word: u32) -> i64 {
        i64::from(((word & 0x03FF_FFFF) << 6) as i32 >> 6)
    }

    /// Where control goes after the `CBNZ`+`B` pair at word `at`, given W16 —
    /// the instructions' manual semantics, read off the emitted words: `CBNZ`
    /// branches when its register is non-zero, `B` always branches.
    fn pair_lands_at(words: &[u32], at: usize, w16: u32) -> i64 {
        let (cb, b) = (words[at], words[at + 1]);
        let branches_on_nonzero = (cb >> 24) & 1 == 1;
        if (w16 != 0) == branches_on_nonzero {
            return at as i64 + imm19(cb);
        }
        at as i64 + 1 + imm26(b)
    }

    #[test]
    fn a_branch_on_w16_is_cbnz_over_b() {
        let mut asm = Assembly::default();
        let exit = asm.mint();
        asm.push(BranchIfW16Zero { target: exit });
        asm.push(NOP);
        asm.bind(exit);
        let code = asm.finish();
        assert_eq!(code.len(), 3 * WORD_BYTES, "cbnz, b, nop");
        let (cbnz, b) = (word_at(&code, 0), word_at(&code, 4));

        assert_eq!(imm19(cbnz), 2, "steps over the B");
        // The CBZ it replaces with its opcode bit flipped: same width, same
        // register, same field — only the sense of the test changed.
        assert_eq!(cbnz ^ (OLD_CBZ_W16 | (2 << 5)), 1 << 24);
        assert_eq!(cbnz & 0xFF00_001F, 0x3500_0010, "cbnz w16");

        assert_eq!(b & 0xFC00_0000, 0x1400_0000, "an unconditional B");
        assert_eq!(imm26(b), 2, "from the B itself, two words to the label");
    }

    /// The words a guard emits for an arm `filler` instructions long, built
    /// the way the emitter builds it: the backend's verb, the arm's body, the
    /// label bound past it, then the assembler's patch pass.
    fn guard_over_arm(arm: IfArm, filler: usize) -> Vec<u32> {
        let mut backend = driver::Aarch64Backend::new();
        let mut asm = Assembly::default();
        let past_arm = asm.mint();
        let test = MaskTest {
            reg: Reg(0),
            scratch: Some(Reg(1)),
            mask_scratch: None,
            arm,
        };
        backend.branch_if_arm_is_dead(&mut asm, test, past_arm);
        for _ in 0..filler {
            asm.push(NOP);
        }
        asm.bind(past_arm);
        asm.finish()
            .as_chunks::<WORD_BYTES>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect()
    }

    /// The guard's jump is `B`, so an arm's length is not bounded by `CBZ`'s
    /// ±1 MiB: the same two words, with the same `CBNZ`, whether the arm is
    /// three instructions or past what `imm19` can say.
    #[test]
    fn a_guard_reaches_past_what_cbz_could() {
        // The first forward distance `imm19` cannot hold, in words.
        let imm19_limit = 1usize << (DispField::IMM19.bits - 1);
        // `umaxv; fmov` for a true arm; `uminv; fmov; mvn` for a false one.
        for (arm, reduction) in [(IfArm::True, 2), (IfArm::False, 3)] {
            for filler in [3, imm19_limit + 17] {
                let words = guard_over_arm(arm, filler);
                let (cbnz_at, b_at) = (reduction, reduction + 1);
                assert_eq!(
                    words.len(),
                    b_at + 1 + filler,
                    "{arm:?}/{filler}: two words"
                );

                assert_eq!(imm19(words[cbnz_at]), 2, "{arm:?}/{filler}");
                assert_eq!(
                    words[cbnz_at], CBNZ_W16_OVER_B,
                    "{arm:?}/{filler}: the CBNZ is never patched"
                );
                assert_eq!(words[b_at] & 0xFC00_0000, 0x1400_0000, "{arm:?}/{filler}");

                // The label sits one word past the last filler word, and the
                // B measures from itself.
                let exact = words.len() - b_at;
                assert_eq!(imm26(words[b_at]), exact as i64, "{arm:?}/{filler}");
                assert_eq!(exact, filler + 1, "{arm:?}/{filler}");
                assert_eq!(
                    exact >= imm19_limit,
                    filler > imm19_limit,
                    "{arm:?}/{filler}: the large arm, and only it, is past imm19"
                );

                // Jump to the label iff W16 is zero — the old CBZ's meaning.
                let label = words.len() as i64;
                let arm_body = (b_at + 1) as i64;
                assert_eq!(pair_lands_at(&words, cbnz_at, 0), label, "{arm:?}/{filler}");
                for live in [1, 0x8000_0000, u32::MAX] {
                    assert_eq!(
                        pair_lands_at(&words, cbnz_at, live),
                        arm_body,
                        "{arm:?}/{filler}/{live:#x}"
                    );
                }
            }
        }
    }

    /// A label may name a position no instruction occupies — the end of the
    /// program. That is why binding a label is a step of its own rather than
    /// a field on an instruction: there is nothing here to hang it on.
    #[test]
    fn a_label_can_end_the_program() {
        let mut asm = Assembly::default();
        let end = asm.mint();
        asm.push(B { target: end });
        asm.bind(end);
        let code = asm.finish();
        assert_eq!(code.len(), 4);
        assert_eq!(word_at(&code, 0) & 0x03FF_FFFF, 1);
    }

    #[test]
    fn a_program_is_position_independent() {
        let program = |prefix: &[u8]| {
            let mut asm = Assembly::default();
            asm.code.extend_from_slice(prefix);
            let end = asm.mint();
            asm.push(B { target: end });
            asm.push(NOP);
            asm.bind(end);
            asm.finish()
        };
        let offset = program(&[0xAA; 4]);
        assert_eq!(&program(&[])[..], &offset[4..]);
    }

    #[test]
    #[should_panic(expected = "does not fit 19 bits")]
    fn a_conditional_refuses_what_it_cannot_reach() {
        let mut code = alloc::vec![0u8; 4];
        DispField::IMM19.write(&mut code, 0, 1 << 20);
    }

    /// A64 is fixed-width, so a displacement that is not a whole number of
    /// instructions means this crate laid something out unaligned.
    #[test]
    #[should_panic(expected = "not a whole number of instructions")]
    fn an_unaligned_displacement_is_a_bug() {
        let mut code = alloc::vec![0u8; 4];
        DispField::IMM26.write(&mut code, 0, 2);
    }
}

#[cfg(test)]
mod xr_tests {
    use super::gpr::*;
    use super::ptr::*;
    use super::*;

    fn word(f: impl FnOnce(&mut Vec<u8>)) -> u32 {
        let mut c = Vec::new();
        f(&mut c);
        assert_eq!(c.len(), 4, "aarch64 instructions are fixed-width");
        u32::from_le_bytes([c[0], c[1], c[2], c[3]])
    }

    /// Each encoding checked against the ARM ARM's form for that mnemonic.
    /// These are the exact words the collapse-loop scaffold used to spell
    /// inline, which is what makes the replacement provably byte-identical.
    #[test]
    fn encodings_match_the_manual() {
        // ADD Xd, Xn, #imm12
        let add = |dst, src, imm| {
            word(|c| AsmProgram::from([table::AddI64::new(dst, src, Imm12(imm))]).assemble(c))
        };
        assert_eq!(add(X1.as_gpr(), X1.as_gpr(), 16), 0x9100_4021);
        assert_eq!(add(X5, X5, 1), 0x9100_04A5);
        assert_eq!(add(X6, X6, 1), 0x9100_04C6);
        // STR Qt, [Xn]
        assert_eq!(
            word(|c| AsmProgram::from([Inst::str_q(
                Reg(0),
                Mem {
                    base: X1,
                    offset: 0,
                },
            )])
            .assemble(c)),
            0x3D80_0020
        );
        // RET
        assert_eq!(
            word(|c| AsmProgram::from([Inst::Ret]).assemble(c)),
            0xD65F_03C0
        );
    }

    /// The `Context` def's own instruction: `ldr x3, [x0, #16]` for context
    /// slot 2.
    #[test]
    fn a_context_pointer_is_read_by_one_ldr() {
        assert_eq!(
            word(|c| AsmProgram::from([Inst::ldr_x(
                PtrReg(3),
                Mem {
                    base: X0,
                    offset: 16,
                },
            )])
            .assemble(c)),
            0xF940_0803
        );
    }

    /// `Gpr` and `Reg` name different files; the same index is a different
    /// register in each, which is why they are different types.
    #[test]
    fn the_two_register_files_are_not_interchangeable() {
        assert_eq!(X1.0, Reg(1).0);
        // `Inst::str_q` takes both, in their own positions: the vector operand
        // lands in Rt and the address's base in Rn, so swapping them cannot
        // typecheck. `Inst::ldr_x` is the mirror — an `Xr` destination, because
        // it is a load on the general file, not the vector one.
        assert_eq!(
            word(|c| AsmProgram::from([Inst::str_q(
                Reg(3),
                Mem {
                    base: X1,
                    offset: 0,
                },
            )])
            .assemble(c)),
            0x3D80_0023
        );
        assert_eq!(
            word(|c| AsmProgram::from([Inst::ldr_x(
                PtrReg(3),
                Mem {
                    base: X1,
                    offset: 0,
                },
            )])
            .assemble(c)),
            0xF940_0023
        );
    }
}
