//! ARM64/NEON instruction encoding.
//!
//! Each function emits raw machine code bytes for one instruction (or a small fixed sequence).
//! These are the "atoms" that compound operations are built from.

use super::{AsmInsn, AsmProgram, Gpr, PtrReg, Reg, assemble, unimplemented_op};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use pixelflow_ir::kind::OpKind;

pub mod table;
pub use table::*;

// =============================================================================
// Instruction Encoding Helpers
// =============================================================================

/// Write a 32-bit instruction to the code buffer.
#[inline]
pub fn emit32(code: &mut Vec<u8>, inst: u32) {
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
pub enum Inst {
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
    Ldr(Ldr),
    Str(Str),

    // Integer & lane operations
    DupLane0(Reg, Reg),
    UmovW { dst: Gpr, src: Reg, lane: u8 },
    InsW { dst: Reg, lane: u8, src: Gpr },
    MvnW { dst: Gpr, src: Gpr },
    Fcvtzs(Reg, Reg),
    Scvtf(Reg, Reg),
    AddI32(Reg, Reg, Reg),
    And(Reg, Reg, Reg),
    Orr(Reg, Reg, Reg),
    Mov(Reg, Reg),

    // Select guard masks
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
    pub fn ldr(dst: impl Into<LdrReg>, addr: impl Into<Addr>) -> Self {
        Self::Ldr(Ldr::new(dst, addr))
    }
    #[must_use]
    #[inline(always)]
    pub fn ldr_q(dst: Reg, addr: Mem) -> Self {
        Self::Ldr(Ldr::q(dst, addr))
    }
    #[must_use]
    #[inline(always)]
    pub fn ldr_x(dst: PtrReg, addr: Mem) -> Self {
        Self::Ldr(Ldr::x(dst, addr))
    }
    #[must_use]
    #[inline(always)]
    pub fn ldr_s(dst: Reg, addr: Mem) -> Self {
        Self::Ldr(Ldr::s(dst, addr))
    }
    #[must_use]
    #[inline(always)]
    pub fn ldr_w(dst: Gpr, addr: MemIndexed) -> Self {
        Self::Ldr(Ldr::w(dst, addr))
    }
    #[must_use]
    #[inline(always)]
    pub fn str(src: impl Into<StrReg>, addr: Mem) -> Self {
        Self::Str(Str::new(src, addr))
    }
    #[must_use]
    #[inline(always)]
    pub fn str_q(src: Reg, addr: Mem) -> Self {
        Self::Str(Str::q(src, addr))
    }
    #[must_use]
    #[inline(always)]
    pub fn str_x(src: PtrReg, addr: Mem) -> Self {
        Self::Str(Str::x(src, addr))
    }
    #[must_use]
    #[inline(always)]
    pub fn mov(dst: Reg, src: Reg) -> Self {
        Self::Mov(dst, src)
    }
    #[must_use]
    #[inline(always)]
    pub fn umov_w(dst: Gpr, src: Reg, lane: u8) -> Self {
        Self::UmovW { dst, src, lane }
    }
    #[must_use]
    #[inline(always)]
    pub fn ins_w(dst: Reg, lane: u8, src: Gpr) -> Self {
        Self::InsW { dst, lane, src }
    }
    #[must_use]
    #[inline(always)]
    pub fn mvn_w(dst: impl Into<Gpr>, src: impl Into<Gpr>) -> Self {
        Self::MvnW {
            dst: dst.into(),
            src: src.into(),
        }
    }
    #[must_use]
    #[inline(always)]
    pub fn dup_lane0(dst: Reg, src: Reg) -> Self {
        Self::DupLane0(dst, src)
    }

    /// Pure encoding of single-word instructions into a 32-bit machine word.
    #[must_use]
    #[inline]
    pub fn encode(self) -> u32 {
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
            Inst::Ldr(_) | Inst::Str(_) => {
                panic!("Ldr and Str must be emitted via emit_into or AsmProgram")
            }
            Inst::DupLane0(dst, src) => DupLane0::new(dst, src).encode(),
            Inst::UmovW { dst, src, lane } => UmovW::new(dst, src, lane).encode(),
            Inst::InsW { dst, lane, src } => InsW::new(dst, lane, src).encode(),
            Inst::MvnW { dst, src } => table::MvnW::new(dst, src).encode(),
            Inst::Fcvtzs(dst, src) => Fcvtzs::new(dst, src).encode(),
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

impl From<Ldr> for Inst {
    #[inline(always)]
    fn from(l: Ldr) -> Self {
        Inst::Ldr(l)
    }
}

impl From<Str> for Inst {
    #[inline(always)]
    fn from(s: Str) -> Self {
        Inst::Str(s)
    }
}

impl From<DupLane0> for Inst {
    #[inline(always)]
    fn from(d: DupLane0) -> Self {
        Inst::DupLane0(d.dst, d.src)
    }
}

impl From<UmovW> for Inst {
    #[inline(always)]
    fn from(u: UmovW) -> Self {
        Inst::UmovW {
            dst: u.dst,
            src: u.src,
            lane: u.lane,
        }
    }
}

impl From<InsW> for Inst {
    #[inline(always)]
    fn from(i: InsW) -> Self {
        Inst::InsW {
            dst: i.dst,
            lane: i.lane,
            src: i.src,
        }
    }
}

impl From<table::MvnW> for Inst {
    #[inline(always)]
    fn from(m: table::MvnW) -> Self {
        Inst::MvnW {
            dst: m.dst,
            src: m.src,
        }
    }
}

impl crate::emit::AsmInsn for Inst {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        match self {
            Inst::Ldr(ldr) => ldr.emit_into(code),
            Inst::Str(str) => str.emit_into(code),
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

/// `dup v<dst>.4s, v<src>.s[0]` — broadcast lane 0 to every lane.
#[inline]
pub fn emit_dup_lane0(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    assemble(code, [DupLane0::new(dst, src)]);
}

/// `dst = splat(block[offset])`: `ldr x9, [x0, #ctx_slot*8]` fetches the
/// block's base out of the context, `ldr s<dst>, [x9, #offset*4]` the value,
/// and `dup` spreads it. `x9` is the base scratch the gather already claims.
pub fn emit_uniform_load(code: &mut Vec<u8>, dst: Reg, load: super::UniformLoad) {
    const BASE_GPR: PtrReg = ptr::X9;
    AsmProgram::from([
        Inst::ldr_x(
            BASE_GPR,
            Mem {
                base: ptr::X0,
                offset: u32::from(load.ctx_slot) * X_BYTES,
            },
        ),
        Inst::ldr_s(
            dst,
            Mem {
                base: BASE_GPR,
                offset: u32::from(load.offset) * S_BYTES,
            },
        ),
        Inst::DupLane0(dst, dst),
    ])
    .assemble(code);
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
pub fn emit_fmov_imm(code: &mut Vec<u8>, dst: Reg, val: f32) {
    let bits = val.to_bits();

    if bits == 0 {
        // MOVI Vd.4S, #0 - single instruction for zero
        emit32(code, 0x4F000400 | (dst.0 as u32));
        return;
    }

    // Try FMOV Vd.4S, #imm8 for common float constants (1 instruction)
    if let Some(imm8) = try_encode_fmov_imm8(val) {
        let abc = ((imm8 as u32) >> 5) & 0x7;
        let defgh = (imm8 as u32) & 0x1F;
        // FMOV Vd.4S, #imm8: 0x4F00F400 | abc<<16 | defgh<<5 | Rd
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

    // MOVZ W16, #lo16
    emit32(code, 0x52800010 | (lo16 << 5));

    // MOVK W16, #hi16, LSL #16
    emit32(code, 0x72A00010 | (hi16 << 5));

    // DUP Vd.4S, W16
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
pub fn try_encode_fmov_imm8(val: f32) -> Option<u8> {
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

    // Extract imm8 = a:b:c:d:e:f:g:h
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

/// Returns true if the given f32 needs a constant pool entry (not zero, not FMOV-encodable).
#[must_use]
pub fn needs_const_pool(val: f32) -> bool {
    val.to_bits() != 0 && try_encode_fmov_imm8(val).is_none()
}

/// Emit `ADR X17, #0` as a placeholder. Returns the code offset for later patching.
///
/// ADR encodes a PC-relative offset into X17 (IP1, platform scratch register).
/// The offset is patched after the constant pool position is known.
pub fn emit_adr_x17_placeholder(code: &mut Vec<u8>) -> usize {
    let pos = code.len();
    // ADR X17, #0 — will be patched. Encoding: 0x10000011 (Rd=X17=17, imm=0)
    emit32(code, 0x10000011);
    pos
}

/// Patch a previously emitted `ADR X17` placeholder at `adr_pos` to point to `target_pos`.
/// If `is_adrp` is true, assumes 8 bytes are reserved and patches `ADRP X17` + `ADD X17`.
pub fn patch_adr_or_adrp(code: &mut [u8], adr_pos: usize, target_pos: usize, is_adrp: bool) {
    if is_adrp {
        assert!(
            adr_pos + 8 <= code.len(),
            "patch_adr_or_adrp: adr_pos {} + 8 exceeds code length {}",
            adr_pos,
            code.len()
        );

        let pc_page = (adr_pos as i64) & !0xFFF;
        let target_page = (target_pos as i64) & !0xFFF;
        let page_offset = (target_page - pc_page) >> 12;

        assert!(
            (-(1 << 20)..(1 << 20)).contains(&page_offset),
            "ADRP page offset {} out of range (±4GB)",
            page_offset
        );

        // 1. Patch ADRP
        let imm_bits = (page_offset as u32) & 0x1F_FFFF;
        let immlo = imm_bits & 0x3;
        let immhi = (imm_bits >> 2) & 0x7FFFF;
        let adrp_inst = 0x90000011 | (immlo << 29) | (immhi << 5);
        code[adr_pos..adr_pos + 4].copy_from_slice(&adrp_inst.to_le_bytes());

        // 2. Patch ADD (immediate)
        // ADD X17, X17, #target_pos_within_page
        let page_inner_offset = (target_pos as u32) & 0xFFF;
        let add_inst = 0x91000231 | (page_inner_offset << 10);
        code[adr_pos + 4..adr_pos + 8].copy_from_slice(&add_inst.to_le_bytes());
    } else {
        assert!(
            adr_pos + 4 <= code.len(),
            "patch_adr_or_adrp: adr_pos {} + 4 exceeds code length {}",
            adr_pos,
            code.len()
        );
        let offset = (target_pos as i64) - (adr_pos as i64);
        assert!(
            (-(1 << 20)..(1 << 20)).contains(&offset),
            "ADR offset {} out of range (±1MB)",
            offset
        );
        let offset_bits = (offset as u32) & 0x1F_FFFF;
        let immlo = offset_bits & 0x3;
        let immhi = (offset_bits >> 2) & 0x7FFFF;
        let inst = 0x10000011 | (immlo << 29) | (immhi << 5);
        code[adr_pos..adr_pos + 4].copy_from_slice(&inst.to_le_bytes());
    }
}

/// Emit a constant pool entry: 16 bytes = f32 value splatted 4x (fills a 128-bit NEON register).
pub fn emit_pool_entry(code: &mut Vec<u8>, val_bits: u32) {
    let bytes = val_bits.to_le_bytes();
    for _ in 0..4 {
        code.extend_from_slice(&bytes);
    }
}

// =============================================================================
// Bound-Memory Gather (scalar-load lowering — NEON has no native gather)
// =============================================================================

/// GP registers used by the scalar-load gather sequence.
pub struct GatherGprs {
    /// Holds the buffer base pointer (survives the whole sequence).
    pub base: PtrReg,
    /// Scratch: one extracted lane index at a time. Clobbered.
    pub idx: Gpr,
    /// Scratch: one loaded value at a time. Clobbered.
    pub val: Gpr,
}

/// dst.4S = base[idx_int.S[lane]] for each lane — the NEON gather: four scalar
/// loads through GP scratch. `gprs.base` holds the buffer base pointer;
/// `idx_int` holds int32 lane indices (already converted and in-bounds by the
/// `expand_gather` lowering). Clobbers `gprs.idx` and `gprs.val`.
pub fn emit_gather(code: &mut Vec<u8>, dst: Reg, idx_int: Reg, gprs: GatherGprs) {
    let mem = MemIndexed {
        base: gprs.base,
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

// =============================================================================
// Integer Vector Operations (for bit manipulation in transcendentals)
// =============================================================================

/// USHR Vd.4S, Vn.4S, #shift (unsigned shift right by immediate)
fn emit_ushr(code: &mut Vec<u8>, dst: Reg, src: Reg, shift: u8) {
    // Encoding: 0x6F200400 | ((32 - shift) << 16) as immh:immb
    // For .4S: immh = 001x, so (32-shift) in bits [19:16]
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
/// blends a select from its three operands.
pub(crate) fn temps_for(op: &super::ScheduledOp) -> u8 {
    use super::ScheduledOp;
    match op {
        ScheduledOp::Unary(OpKind::Rsqrt | OpKind::Recip, _) => 1,
        // The gather's truncated-index lanes.
        ScheduledOp::Gather(..) => 1,
        _ => 0,
    }
}

#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
/// `dst = op(src)`.
///
/// `temp` is the allocator's temp for this instruction; only the reciprocal
/// estimates use it, to hold the Newton-Raphson correction.
pub(crate) fn emit_unary(code: &mut Vec<u8>, op: OpKind, dst: Reg, src: Reg, temp: Option<Reg>) {
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

        // Bit-manip primitives (integer-domain conversions).
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
pub fn emit_shift_imm(code: &mut Vec<u8>, op: OpKind, dst: Reg, src: Reg, amount: u8) {
    match op {
        OpKind::Shl => emit_shl(code, dst, src, amount),
        OpKind::Shr => emit_ushr(code, dst, src, amount),
        _ => unimplemented_op("aarch64", op),
    }
}

/// Emit binary operation
pub fn emit_binary(code: &mut Vec<u8>, op: OpKind, dst: Reg, src1: Reg, src2: Reg) {
    match op {
        OpKind::Add => AsmProgram::from([Inst::Fadd(dst, src1, src2)]).assemble(code),
        OpKind::Sub => AsmProgram::from([Inst::Fsub(dst, src1, src2)]).assemble(code),
        OpKind::Mul => AsmProgram::from([Inst::Fmul(dst, src1, src2)]).assemble(code),
        OpKind::Div => AsmProgram::from([Inst::Fdiv(dst, src1, src2)]).assemble(code),
        OpKind::Min => AsmProgram::from([Inst::Fmin(dst, src1, src2)]).assemble(code),
        OpKind::Max => AsmProgram::from([Inst::Fmax(dst, src1, src2)]).assemble(code),

        // Comparisons (result is mask in dst)
        OpKind::Gt => AsmProgram::from([Inst::Fcmgt(dst, src1, src2)]).assemble(code),
        OpKind::Ge => AsmProgram::from([Inst::Fcmge(dst, src1, src2)]).assemble(code),
        OpKind::Lt => AsmProgram::from([Inst::Fcmgt(dst, src2, src1)]).assemble(code), // swap args
        OpKind::Le => AsmProgram::from([Inst::Fcmge(dst, src2, src1)]).assemble(code),
        OpKind::Eq => AsmProgram::from([Inst::Fcmeq(dst, src1, src2)]).assemble(code),
        OpKind::Ne => {
            // Ne = not Eq: FCMEQ then bitwise NOT
            AsmProgram::from([Inst::Fcmeq(dst, src1, src2), Inst::Not(dst, dst)]).assemble(code);
        }

        // Bit-manip primitives (integer-domain).
        OpKind::IAdd => AsmProgram::from([Inst::AddI32(dst, src1, src2)]).assemble(code),
        OpKind::BitAnd => AsmProgram::from([Inst::And(dst, src1, src2)]).assemble(code),
        OpKind::BitOr => AsmProgram::from([Inst::Orr(dst, src1, src2)]).assemble(code),

        _ => unimplemented_op("aarch64", op),
    }
}

// =============================================================================
// Prologue / Epilogue
// =============================================================================

// =============================================================================
// Disassembly support
// =============================================================================

/// Disassemble a raw code buffer into a human-readable string.
///
#[must_use]
pub fn disassemble_code(code: &[u8]) -> String {
    let mut out = String::new();
    for (i, chunk) in code.chunks(4).enumerate() {
        if chunk.len() < 4 {
            break;
        }
        let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let offset = i * 4;
        let mnemonic = decode_aarch64_mnemonic(word);
        out.push_str(&format!("{:4x}: {:08x}  {}\n", offset, word, mnemonic));
    }
    out
}

/// Decode a 32-bit AArch64 instruction into a mnemonic string.
///
/// Covers the instructions actually emitted by this JIT:
/// - NEON floating-point (fadd, fmul, fsub, fdiv, fabs, fneg, fsqrt, fmin, fmax)
/// - NEON integer (add, sub, shl, ushr, and, orr, not)
/// - Comparisons (fcmgt, fcmge, fcmeq)
/// - Selection (bsl)
/// - Move/dup (mov v, dup)
/// - Memory (ldr, str, ldp, stp)
/// - Control (ret, cbz, cbnz, b, b.cond, subs)
/// - Constants (movi, fmov imm, movz, movk)
///
/// Everything else returns "unknown".
fn decode_aarch64_mnemonic(word: u32) -> String {
    let rd = word & 0x1F;
    let rn = (word >> 5) & 0x1F;
    let rm = (word >> 16) & 0x1F;

    // RET
    if word == 0xD65F03C0 {
        return "ret".into();
    }

    // MOVI Vd.4S, #0 (common zero-fill)
    if word & 0xFFFFFC00 == 0x4F000400 {
        return format!("movi v{}.4s, #0", rd);
    }

    // FMOV Vd.4S, #imm8
    if word & 0xFFC0FC00 == 0x4F00F400 {
        return format!("fmov v{}.4s, #imm8", rd);
    }

    // DUP Vd.4S, W16 (from GP)
    if word & 0xFFFFFC00 == 0x4E040C00 {
        return format!("dup v{}.4s, w{}", rd, rn);
    }

    // DUP Vd.4S, Vn.S[0] (scalar dup)
    if word & 0xFFFFFC00 == 0x4E040400 {
        return format!("dup v{}.4s, v{}.s[0]", rd, rn);
    }

    // MOVZ Wd, #imm16
    if word & 0xFFE0001F == 0x52800010 {
        let imm16 = (word >> 5) & 0xFFFF;
        return format!("movz w16, #0x{:x}", imm16);
    }
    if word & 0xFFE00000 == 0x52800000 {
        let imm16 = (word >> 5) & 0xFFFF;
        return format!("movz w{}, #0x{:x}", rd, imm16);
    }
    // MOVZ Xd, #imm16 (64-bit)
    if word & 0xFFE00000 == 0xD2800000 {
        let imm16 = (word >> 5) & 0xFFFF;
        return format!("movz x{}, #0x{:x}", rd, imm16);
    }

    // MOVK Wd, #imm16, LSL #16
    if word & 0xFFE00000 == 0x72A00000 {
        let imm16 = (word >> 5) & 0xFFFF;
        return format!("movk w{}, #0x{:x}, lsl #16", rd, imm16);
    }

    // MOV Xd, Xm (ORR Xd, XZR, Xm)
    if word & 0xFFE0FFE0 == 0xAA0003E0 {
        return format!("mov x{}, x{}", rd, rm);
    }

    // NEON 3-same (binary vector ops) — top bits determine the operation
    // Extract opcode bits for classification
    let top11 = word >> 21;

    // ORR Vd.16B, Vn.16B, Vm.16B (also MOV when Vn==Vm)
    if word & 0xFFE0FC00 == 0x4EA01C00 {
        if rn == rm {
            return format!("mov v{}.16b, v{}.16b", rd, rn);
        }
        return format!("orr v{}.16b, v{}.16b, v{}.16b", rd, rn, rm);
    }

    // FADD Vd.4S, Vn.4S, Vm.4S
    if word & 0xFFE0FC00 == 0x4E20D400 {
        return format!("fadd v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }
    // FSUB Vd.4S, Vn.4S, Vm.4S
    if word & 0xFFE0FC00 == 0x4EA0D400 {
        return format!("fsub v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }
    // FMUL Vd.4S, Vn.4S, Vm.4S
    if word & 0xFFE0FC00 == 0x6E20DC00 {
        return format!("fmul v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }
    // FDIV Vd.4S, Vn.4S, Vm.4S
    if word & 0xFFE0FC00 == 0x6E20FC00 {
        return format!("fdiv v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }
    // FMLA Vd.4S, Vn.4S, Vm.4S (fused multiply-add)
    if word & 0xFFE0FC00 == 0x4E20CC00 {
        return format!("fmla v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }
    // FMIN Vd.4S, Vn.4S, Vm.4S
    if word & 0xFFE0FC00 == 0x4EA0F400 {
        return format!("fmin v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }
    // FMAX Vd.4S, Vn.4S, Vm.4S
    if word & 0xFFE0FC00 == 0x4E20F400 {
        return format!("fmax v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }

    // FCMGT Vd.4S, Vn.4S, Vm.4S
    if word & 0xFFE0FC00 == 0x6EA0E400 {
        return format!("fcmgt v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }
    // FCMGE Vd.4S, Vn.4S, Vm.4S
    if word & 0xFFE0FC00 == 0x6E20E400 {
        return format!("fcmge v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }
    // FCMEQ Vd.4S, Vn.4S, Vm.4S
    if word & 0xFFE0FC00 == 0x4E20E400 {
        return format!("fcmeq v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }

    // BSL Vd.16B, Vn.16B, Vm.16B
    if word & 0xFFE0FC00 == 0x6E601C00 {
        return format!("bsl v{}.16b, v{}.16b, v{}.16b", rd, rn, rm);
    }

    // AND Vd.16B, Vn.16B, Vm.16B
    if word & 0xFFE0FC00 == 0x4E201C00 {
        return format!("and v{}.16b, v{}.16b, v{}.16b", rd, rn, rm);
    }

    // ADD Vd.4S, Vn.4S, Vm.4S (integer)
    if word & 0xFFE0FC00 == 0x4EA08400 {
        return format!("add v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }
    // SUB Vd.4S, Vn.4S, Vm.4S (integer)
    if word & 0xFFE0FC00 == 0x6EA08400 {
        return format!("sub v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }

    // FRSQRTS Vd.4S, Vn.4S, Vm.4S
    if word & 0xFFE0FC00 == 0x4EA0FC00 {
        return format!("frsqrts v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }
    // FRECPS Vd.4S, Vn.4S, Vm.4S
    if word & 0xFFE0FC00 == 0x4E20FC00 {
        return format!("frecps v{}.4s, v{}.4s, v{}.4s", rd, rn, rm);
    }

    // 2-reg misc (unary vector ops)
    // FSQRT Vd.4S, Vn.4S
    if word & 0xFFFFFC00 == 0x6EA1F800 {
        return format!("fsqrt v{}.4s, v{}.4s", rd, rn);
    }
    // FABS Vd.4S, Vn.4S
    if word & 0xFFFFFC00 == 0x4EA0F800 {
        return format!("fabs v{}.4s, v{}.4s", rd, rn);
    }
    // FNEG Vd.4S, Vn.4S
    if word & 0xFFFFFC00 == 0x6EA0F800 {
        return format!("fneg v{}.4s, v{}.4s", rd, rn);
    }
    // NOT Vd.16B, Vn.16B
    if word & 0xFFFFFC00 == 0x2E205800 {
        return format!("not v{}.16b, v{}.16b", rd, rn);
    }
    // FRINTM Vd.4S, Vn.4S (floor)
    if word & 0xFFFFFC00 == 0x4E219800 {
        return format!("frintm v{}.4s, v{}.4s", rd, rn);
    }
    // FRINTP Vd.4S, Vn.4S (ceil)
    if word & 0xFFFFFC00 == 0x4EA18800 {
        return format!("frintp v{}.4s, v{}.4s", rd, rn);
    }
    // FRINTA Vd.4S, Vn.4S (round)
    if word & 0xFFFFFC00 == 0x6E218800 {
        return format!("frinta v{}.4s, v{}.4s", rd, rn);
    }
    // FRSQRTE Vd.4S, Vn.4S
    if word & 0xFFFFFC00 == 0x6EA1D800 {
        return format!("frsqrte v{}.4s, v{}.4s", rd, rn);
    }
    // FRECPE Vd.4S, Vn.4S
    if word & 0xFFFFFC00 == 0x4EA1D800 {
        return format!("frecpe v{}.4s, v{}.4s", rd, rn);
    }
    // FCVTZS Vd.4S, Vn.4S
    if word & 0xFFFFFC00 == 0x4EA1B800 {
        return format!("fcvtzs v{}.4s, v{}.4s", rd, rn);
    }
    // SCVTF Vd.4S, Vn.4S
    if word & 0xFFFFFC00 == 0x4E21D800 {
        return format!("scvtf v{}.4s, v{}.4s", rd, rn);
    }
    // UMINV Sd, Vn.4S
    if word & 0xFFFFFC00 == 0x6EB1A800 {
        return format!("uminv s{}, v{}.4s", rd, rn);
    }
    // UMAXV Sd, Vn.4S
    if word & 0xFFFFFC00 == 0x6E30A800 {
        return format!("umaxv s{}, v{}.4s", rd, rn);
    }

    // FMOV Wd, Sn (SIMD to GP)
    if word & 0xFFFFFC00 == 0x1E260000 {
        return format!("fmov w{}, s{}", rd, rn);
    }

    // LDR Qt, [Xn, #imm] (128-bit unsigned offset)
    if word & 0xFFC00000 == 0x3DC00000 {
        let imm12 = (word >> 10) & 0xFFF;
        let byte_offset = imm12 * 16;
        return format!("ldr q{}, [x{}, #{}]", rd, rn, byte_offset);
    }
    // STR Qt, [Xn, #imm] (128-bit unsigned offset)
    if word & 0xFFC00000 == 0x3D800000 {
        let imm12 = (word >> 10) & 0xFFF;
        let byte_offset = imm12 * 16;
        return format!("str q{}, [x{}, #{}]", rd, rn, byte_offset);
    }

    // LDR Qt, [Xn], #16 (post-index)
    if word & 0xFFFFFC00 == 0x3CC10400 {
        return format!("ldr q{}, [x{}], #16", rd, rn);
    }
    // STR Qt, [Xn], #16 (post-index)
    if word & 0xFFFFFC00 == 0x3C810400 {
        return format!("str q{}, [x{}], #16", rd, rn);
    }

    // STP Xt1, Xt2, [Xn, #imm]! (pre-index GP pair)
    if word & 0xFFC00000 == 0xA9800000 | (0b11 << 23) {
        let rt2 = (word >> 10) & 0x1F;
        let imm7 = ((word >> 15) & 0x7F) as i32;
        let offset = (if imm7 >= 64 { imm7 - 128 } else { imm7 }) * 8;
        return format!("stp x{}, x{}, [x{}, #{}]!", rd, rt2, rn, offset);
    }

    // STP Xt1, Xt2, [Xn, #imm] (signed offset GP pair, no writeback)
    if word & 0xFFC00000 == 0xA9000000 {
        let rt2 = (word >> 10) & 0x1F;
        let imm7 = ((word >> 15) & 0x7F) as i32;
        let offset = (if imm7 >= 64 { imm7 - 128 } else { imm7 }) * 8;
        return format!("stp x{}, x{}, [x{}, #{}]", rd, rt2, rn, offset);
    }

    // LDP Xt1, Xt2, [Xn], #imm (post-index GP pair)
    if word & 0xFFC00000 == 0xA8C00000 {
        let rt2 = (word >> 10) & 0x1F;
        let imm7 = ((word >> 15) & 0x7F) as i32;
        let offset = (if imm7 >= 64 { imm7 - 128 } else { imm7 }) * 8;
        return format!("ldp x{}, x{}, [x{}], #{}", rd, rt2, rn, offset);
    }

    // LDP Xt1, Xt2, [Xn, #imm] (signed offset GP pair)
    if word & 0xFFC00000 == 0xA9400000 {
        let rt2 = (word >> 10) & 0x1F;
        let imm7 = ((word >> 15) & 0x7F) as i32;
        let offset = (if imm7 >= 64 { imm7 - 128 } else { imm7 }) * 8;
        return format!("ldp x{}, x{}, [x{}, #{}]", rd, rt2, rn, offset);
    }

    // STP Qt1, Qt2, [Xn, #imm] (NEON pair, signed offset)
    if word & 0xFFC00000 == 0xAD000000 {
        let rt2 = (word >> 10) & 0x1F;
        let imm7 = ((word >> 15) & 0x7F) as i32;
        let offset = (if imm7 >= 64 { imm7 - 128 } else { imm7 }) * 16;
        return format!("stp q{}, q{}, [x{}, #{}]", rd, rt2, rn, offset);
    }

    // LDP Qt1, Qt2, [Xn, #imm] (NEON pair, signed offset)
    if word & 0xFFC00000 == 0xAD400000 {
        let rt2 = (word >> 10) & 0x1F;
        let imm7 = ((word >> 15) & 0x7F) as i32;
        let offset = (if imm7 >= 64 { imm7 - 128 } else { imm7 }) * 16;
        return format!("ldp q{}, q{}, [x{}, #{}]", rd, rt2, rn, offset);
    }

    // ADD Xd, Xn, #imm12 (GP immediate)
    if word & 0xFF000000 == 0x91000000 {
        let imm12 = (word >> 10) & 0xFFF;
        return format!("add x{}, x{}, #{}", rd, rn, imm12);
    }

    // SUB Xd, Xn, #imm12 (GP immediate) -- includes SUBS via 0xF1
    if word & 0xFF000000 == 0xD1000000 {
        let imm12 = (word >> 10) & 0xFFF;
        return format!("sub x{}, x{}, #{}", rd, rn, imm12);
    }
    if word & 0xFF000000 == 0xF1000000 {
        let imm12 = (word >> 10) & 0xFFF;
        return format!("subs x{}, x{}, #{}", rd, rn, imm12);
    }

    // USHR Vd.4S, Vn.4S, #shift
    // Mask includes immh[3:2] (bits 22:21) so the .4S arrangement selector
    // (immh = 01xx) is matched; bits 20:16 carry the shift amount and stay free.
    if word & 0xFFE0FC00 == 0x6F200400 {
        let immhb = (word >> 16) & 0x3F;
        let shift = 64u32.wrapping_sub(immhb) & 0x3F;
        return format!("ushr v{}.4s, v{}.4s, #{}", rd, rn, shift);
    }

    // SHL Vd.4S, Vn.4S, #shift
    // Mask includes immh[3:2] (bits 22:21) so only the .4S arrangement
    // (immh = 01xx) matches; bits 20:16 carry the shift amount and stay free.
    // Without this a 64-bit `SHL .2D` (immh = 1xxx) would mis-decode as `.4s`.
    if word & 0xFFE0FC00 == 0x4F205400 {
        let immhb = (word >> 16) & 0x3F;
        let shift = immhb.wrapping_sub(32);
        return format!("shl v{}.4s, v{}.4s, #{}", rd, rn, shift);
    }

    // ADR Xd, #imm
    if word & 0x9F000000 == 0x10000000 {
        return format!("adr x{}, <imm>", rd);
    }
    // ADRP Xd, #imm
    if word & 0x9F000000 == 0x90000000 {
        return format!("adrp x{}, <imm>", rd);
    }

    // CBZ Wt (32-bit)
    if word & 0xFF000000 == 0x34000000 {
        return format!("cbz w{}, <imm>", rd);
    }
    // CBNZ Wt (32-bit)
    if word & 0xFF000000 == 0x35000000 {
        return format!("cbnz w{}, <imm>", rd);
    }
    // CBZ Xt (64-bit)
    if word & 0xFF000000 == 0xB4000000 {
        return format!("cbz x{}, <imm>", rd);
    }
    // CBNZ Xt (64-bit)
    if word & 0xFF000000 == 0xB5000000 {
        return format!("cbnz x{}, <imm>", rd);
    }

    // B.cond
    if word & 0xFF000010 == 0x54000000 {
        let cond = word & 0xF;
        let cond_name = match cond {
            0x0 => "eq",
            0x1 => "ne",
            0x2 => "cs",
            0x3 => "cc",
            0x4 => "mi",
            0x5 => "pl",
            0x8 => "hi",
            0x9 => "ls",
            0xA => "ge",
            0xB => "lt",
            0xC => "gt",
            0xD => "le",
            _ => "??",
        };
        return format!("b.{} <imm>", cond_name);
    }

    // B (unconditional)
    if word & 0xFC000000 == 0x14000000 {
        return "b <imm>".to_string();
    }

    // SUBS Xd, Xn, Xm (register)
    if word & 0xFFE00000 == 0xEB000000 {
        return format!("subs x{}, x{}, x{}", rd, rn, rm);
    }

    let _ = top11; // suppress unused warning
    "unknown".into()
}

// =============================================================================
// dump_jit_asm — compile expression and return disassembly
// =============================================================================

/// Compile an expression from an [`ExprArena`] and return its disassembly.
///
/// This is a diagnostic entry point: it compiles the expression through the
/// normal JIT pipeline, then disassembles the resulting machine code instead
/// of executing it. Useful for inspecting what the JIT generates.
///
/// # Errors
///
/// Returns [`crate::error::CompileError`] if compilation fails (same errors as
/// [`compile`](super::compile)).
#[cfg(target_arch = "aarch64")]
pub fn dump_jit_asm(
    arena: &pixelflow_ir::arena::ExprArena,
    root: pixelflow_ir::arena::ExprId,
) -> Result<String, crate::error::CompileError> {
    let result = super::compile(arena, root)?;
    Ok(disassemble_code(result.code.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // =====================================================================
    // Disassembler tests
    // =====================================================================

    #[test]
    fn disassemble_ret() {
        let mut code = Vec::new();
        ret(&mut code);
        let dis = disassemble_code(&code);
        assert!(
            dis.contains("ret"),
            "disassembly should contain 'ret', got: {dis}"
        );
    }

    #[test]
    fn disassemble_fadd() {
        let mut code = Vec::new();
        AsmProgram::from([Inst::Fadd(Reg(0), Reg(1), Reg(2))]).assemble(&mut code);
        let dis = disassemble_code(&code);
        assert!(
            dis.contains("fadd v0.4s, v1.4s, v2.4s"),
            "expected fadd decode, got: {dis}"
        );
    }

    // Round-trip the NEON shift-by-immediate encoders through the disassembler.
    // These guard the `immh` arrangement bits in the decoder masks: the `.4S`
    // form sets immh = 01xx, so emitted words have bits[23:21] = 001. A mask
    // that ignores those bits either never matches (the USHR bug fixed here) or
    // mis-decodes the 64-bit `.2D` form as `.4s` (the SHL case).
    #[test]
    fn disassemble_ushr() {
        let mut code = Vec::new();
        emit_ushr(&mut code, Reg(0), Reg(0), 23); // used by the log2 lowering
        let dis = disassemble_code(&code);
        assert!(
            dis.contains("ushr v0.4s, v0.4s, #23"),
            "expected ushr decode, got: {dis}"
        );
    }

    #[test]
    fn disassemble_shl() {
        let mut code = Vec::new();
        emit_shl(&mut code, Reg(1), Reg(2), 8);
        let dis = disassemble_code(&code);
        assert!(
            dis.contains("shl v1.4s, v2.4s, #8"),
            "expected shl decode, got: {dis}"
        );
    }

    #[test]
    fn disassemble_mov_vec() {
        let mut code = Vec::new();
        AsmProgram::from([Inst::mov(Reg(5), Reg(3))]).assemble(&mut code);
        let dis = disassemble_code(&code);
        assert!(
            dis.contains("mov v5.16b, v3.16b"),
            "expected mov decode, got: {dis}"
        );
    }

    #[test]
    fn disassemble_sequence() {
        let mut code = Vec::new();
        AsmProgram::from([
            Inst::Fmul(Reg(4), Reg(0), Reg(0)),
            Inst::Fsqrt(Reg(4), Reg(4)),
            Inst::Ret,
        ])
        .assemble(&mut code);
        let dis = disassemble_code(&code);
        assert!(dis.contains("fmul"), "missing fmul in: {dis}");
        assert!(dis.contains("fsqrt"), "missing fsqrt in: {dis}");
        assert!(dis.contains("ret"), "missing ret in: {dis}");
    }

    #[test]
    fn disassemble_zero_const() {
        let mut code = Vec::new();
        emit_fmov_imm(&mut code, Reg(0), 0.0);
        let dis = disassemble_code(&code);
        assert!(
            dis.contains("movi"),
            "zero should decode as movi, got: {dis}"
        );
    }

    #[test]
    fn disassemble_ldr_str() {
        let mut code = Vec::new();
        AsmProgram::from([
            Inst::ldr_q(
                Reg(0),
                Mem {
                    base: ptr::X0,
                    offset: 32,
                },
            ),
            Inst::str_q(
                Reg(1),
                Mem {
                    base: ptr::X0,
                    offset: 48,
                },
            ),
        ])
        .assemble(&mut code);
        let dis = disassemble_code(&code);
        assert!(dis.contains("ldr"), "missing ldr in: {dis}");
        assert!(dis.contains("str"), "missing str in: {dis}");
    }

    #[test]
    fn disassemble_code_empty() {
        let dis = disassemble_code(&[]);
        assert!(
            dis.is_empty(),
            "empty code should produce empty disassembly"
        );
    }

    #[test]
    fn disassemble_code_short_chunk() {
        // Less than 4 bytes should produce nothing
        let dis = disassemble_code(&[0x00, 0x01]);
        assert!(
            dis.is_empty(),
            "short chunk should produce empty disassembly"
        );
    }

    #[test]
    fn disassemble_offsets_are_sequential() {
        let mut code = Vec::new();
        AsmProgram::from([
            Inst::Fadd(Reg(0), Reg(1), Reg(2)),
            Inst::Fsub(Reg(0), Reg(1), Reg(2)),
            Inst::Ret,
        ])
        .assemble(&mut code);
        let dis = disassemble_code(&code);
        // Lines should start with offsets 0, 4, 8
        assert!(
            dis.starts_with("   0:"),
            "first line should start at offset 0, got: {dis}"
        );
        let lines: Vec<&str> = dis.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(
            lines[1].starts_with("   4:"),
            "second line should start at offset 4"
        );
        assert!(
            lines[2].starts_with("   8:"),
            "third line should start at offset 8"
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
            GatherGprs {
                base: ptr::X9,
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
        for base in [xr::X0, xr::SP, xr::X17] {
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
                    base: xr::SP,
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
                base: xr::SP,
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

/// The aarch64 half of code generation: the [`IsaBackend`](super::super::IsaBackend)
/// implementation and the constant pool it needs.
///
/// **This file is where aarch64-specific bugs live, and the only place they
/// can.** Emission is a pure function into `Vec<u8>`, so everything here
/// compiles, typechecks and is swept for op coverage on every host — an x86
/// machine computes NEON instruction words perfectly well. Only
/// [`Native`](super::super::Native) decides which backend a build instantiates,
/// and only [`executable`](super::super::executable) needs the matching CPU.
///
/// The consequence worth stating: a change that does not touch an ISA file
/// cannot introduce a platform-specific bug. That is the same bargain `unsafe`
/// makes — confine what cannot be checked, so the rest is checked by
/// construction.
///
/// Dead only in a build that selected a *different* `Native`. The condition
/// mirrors this backend's `Native` alias, so a genuinely unused item in the
/// backend this build actually compiles still trips `dead_code`; an
/// unconditional allow here would hide it from CI's `clippy -D warnings`.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(crate) mod driver {
    use super::super::*;
    use super::ptr;
    use super::xr::*;
    use super::*;
    use super::{Mem, Xr};
    use crate::error::CompileError;
    use alloc::vec::Vec;

    /// Constant pool: maps f32 bit patterns to pool indices.
    ///
    /// Non-zero, non-FMOV-encodable constants are stored in a data section after
    /// the RET instruction. Each entry is 16 bytes (the f32 splatted 4x to fill
    /// a 128-bit NEON register). During code emission, these constants are loaded
    /// with a single `LDR Qt, [X17, #offset]` instead of the 3-instruction
    /// MOVZ+MOVK+DUP sequence.
    pub(crate) struct ConstPool {
        /// Deduplicated entries: f32 bit patterns in pool order.
        entries: Vec<u32>,
        /// Map from f32 bits → pool index.
        index: alloc::collections::BTreeMap<u32, u16>,
    }
    impl ConstPool {
        /// Create an empty constant pool.
        pub(crate) fn new() -> Self {
            Self {
                entries: Vec::new(),
                index: alloc::collections::BTreeMap::new(),
            }
        }

        /// Insert an f32 into the pool (deduplicating by bit pattern) and return
        /// the byte offset for an `LDR Qt, [X17, #offset]` load.
        ///
        /// Zero and FMOV-encodable constants are NOT filtered here — callers
        /// that want the fast path should check `needs_const_pool` first.
        /// Builtin emitters call this unconditionally because every constant
        /// they use benefits from the pool (they are transcendental coefficients,
        /// never zero or FMOV-encodable).
        pub(crate) fn push_f32(&mut self, val: f32) -> Result<u16, CompileError> {
            let bits = val.to_bits();
            if let Some(&idx) = self.index.get(&bits) {
                return Ok(idx * 16);
            }
            let idx = self.entries.len();
            if idx >= 4096 {
                return Err(CompileError::BudgetExceeded(
                    "constant pool overflow: exceeded 12-bit LDR offset limit (max 4095 entries)",
                ));
            }
            self.entries.push(bits);
            self.index.insert(bits, idx as u16);
            Ok((idx * 16) as u16)
        }

        /// Get the byte offset for a constant, or None if it's not in the pool.
        fn offset_for(&self, val_bits: u32) -> Option<u16> {
            self.index.get(&val_bits).map(|&idx| idx * 16)
        }

        /// Returns true if the pool has any entries.
        fn is_empty(&self) -> bool {
            self.entries.is_empty()
        }
    }
    /// Emit a constant load, using the constant pool when available.
    ///
    /// Falls back to `emit_fmov_imm` for zero and FMOV-encodable values.
    fn emit_const_load(code: &mut Vec<u8>, dst: Reg, val_bits: u32, pool: &ConstPool) {
        if let Some(offset) = pool.offset_for(val_bits) {
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

    /// A pending aarch64 branch: 19-bit conditional or 26-bit unconditional.
    pub(crate) enum Aarch64Branch {
        Cond(super::Cond19),
        Uncond(super::Rel26),
    }

    /// aarch64 implementation of the shared driver's leaf operations.
    ///
    /// Mechanically wraps the existing aarch64 encoders + constant pool, so the
    /// emitted code is the same as the previous bespoke `compile_from_schedule`.
    /// The aarch64 (NEON) register file.
    ///
    /// AAPCS64 callee-saves the low 64 bits of v8-v15. These kernels are leaf
    /// functions emitted with no prologue that preserves them, so the scratch pool
    /// must steer clear of that range entirely — handing the allocator one of
    /// v8-v15 would silently corrupt whatever the *caller* had live there across
    /// the JIT call.
    ///
    ///   v0-v3:   inputs (X, Y, Z, W)
    ///   v8-v15:  callee-saved, never allocatable
    ///   v4-v7, v16-v31: allocatable scratch — everything else
    const AARCH64_FILE: regalloc::RegisterFile = regalloc::RegisterFile {
        inputs: INPUT_REGS,
        // v4-v7 and v16-v31: twenty of thirty-two. AAPCS64 callee-saves the
        // low 64 bits of v8-v15 and these are leaf kernels with no prologue
        // that preserves them, so v8-v15 stay out; v4-v7 are unused argument
        // registers.
        //
        // v26/v27 are the last two to join: they were `reload`, held out of
        // every kernel's pool for a spilled operand and a spilled destination.
        // v28 came before them, holding the `UMAXV`/`UMINV` the Select
        // short-circuit guards reduce a mask into. Both needs arise at points
        // the schedule contains, so both are reservations the allocator makes
        // on an instruction (`Scratch::reload`, `guard_temps`).
        scratch: regalloc::RegSet::range(16, 16).union(regalloc::RegSet::range(4, 4)),
        // Nothing. v30 is the gather's truncated-index register, a `temps_for`
        // answer since the gathers landed; v29 used to be `UNARY_SCRATCH`,
        // reserved whole-kernel so a reciprocal estimate could borrow it. The
        // select needs none either: `BSL` reads its three operands directly,
        // and `FNEG`/`FABS` are single instructions.
        fixed: &[],
        temps_for: super::temps_for,
        // `UMAXV`/`UMINV` reduce the mask into a vector register before
        // `FMOV` can move it to a general one. It used to be v28, held out of
        // every kernel's pool; it is now a reservation on the instruction the
        // guard is emitted before.
        guard_temps: 1,
        vector_bytes: 16,
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

    pub(crate) struct Aarch64Backend {
        pool: ConstPool,
        adr_patch_pos: usize,
        file: regalloc::RegisterFile,
    }

    impl Aarch64Backend {
        /// The constant pool as emitted so far.
        ///
        /// Test-only, and says so in the type: its one reader is the test
        /// pinning that the pool APPENDS across the two bodies a collapse
        /// compile pushes through one backend. A reset there is the glyph-ink
        /// regression.
        #[cfg(test)]
        pub(crate) fn pool_entries(&self) -> &[u32] {
            &self.pool.entries
        }

        pub(crate) fn new(ctx: EmitCtx) -> Self {
            Self {
                pool: ConstPool::new(),
                adr_patch_pos: 0,
                file: AARCH64_FILE.capped(ctx.max_regs),
            }
        }

        /// Append the constant pool after the final RET and patch the ADR anchor
        /// (upgrading to ADRP+ADD when the pool is out of ADR range). Shared by
        /// the per-batch epilogue and the collapse-loop scaffold.
        fn finish_pool(&mut self, code: &mut Vec<u8>) {
            if self.pool.is_empty() {
                return;
            }
            let adr_pos = self.adr_patch_pos;
            let estimated_offset = (code.len() as i64) - (adr_pos as i64);
            let needs_adrp = estimated_offset >= (1 << 20) - 32;
            if needs_adrp {
                code.splice(adr_pos + 4..adr_pos + 4, [0, 0, 0, 0]);
            }
            while !code.len().is_multiple_of(16) {
                code.push(0);
            }
            let pool_start = code.len();
            for &bits in &self.pool.entries {
                super::emit_pool_entry(code, bits);
            }
            super::patch_adr_or_adrp(code, adr_pos, pool_start, needs_adrp);
        }
    }

    impl IsaBackend for Aarch64Backend {
        type Branch = Aarch64Branch;

        fn register_file(&self) -> regalloc::RegisterFile {
            self.file
        }

        fn begin(&mut self, schedule: &[regalloc::Def]) -> Result<(), CompileError> {
            // Seed by APPENDING into the existing pool, never replacing it: a
            // collapse compile emits two bodies through one backend (the LICM
            // prologue, then the loop body), and the prologue's bytes have the
            // first pool's X17-relative offsets baked in — resetting here left
            // them pointing into the body's rebuilt pool (wrong constants; the
            // macOS glyph-ink regression). `push_f32` dedups, and each compile
            // constructs a fresh backend, so appending is reset-equivalent for
            // single-body compiles.
            for def in schedule {
                if let ScheduledOp::Const(val) = def.op
                    && super::needs_const_pool(val)
                {
                    self.pool.push_f32(val)?;
                }
            }
            // Builtins add up to ~60 polynomial coefficients during emission; bail
            // if the expression constants + headroom would exceed the 12-bit LDR
            // offset limit.
            const BUILTIN_HEADROOM: usize = 128;
            if self.pool.entries.len() + BUILTIN_HEADROOM > 4095 {
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
            emit_instruction_plan(code, plan, &mut self.pool)
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
        ) -> Reg {
            match location_of(locs, vid) {
                Binding::Loc(Loc::Reg(reg)) => reg,
                Binding::Remat(bits) => {
                    emit_const_load(code, target, bits, &self.pool);
                    target
                }
                Binding::Loc(Loc::Slot(slot)) => {
                    AsmProgram::from([Inst::ldr_q(target, frame_slot(slot.offset()))])
                        .assemble(code);
                    target
                }
            }
        }

        /// `scratch` is this instruction's own reservation, live for these two
        /// instructions only — the allocator makes it because this backend's
        /// `guard_temps` asks for one.
        fn emit_skip_if_all_false(
            &mut self,
            code: &mut Vec<u8>,
            mask_reg: Reg,
            scratch: Option<Reg>,
        ) -> Aarch64Branch {
            let scratch = guard_scratch(scratch, mask_reg);
            AsmProgram::from([Inst::Umaxv(scratch, mask_reg), Inst::FmovToGp(scratch)])
                .assemble(code);
            Aarch64Branch::Cond(super::cbz_w16(code))
        }

        fn emit_skip_if_all_true(
            &mut self,
            code: &mut Vec<u8>,
            mask_reg: Reg,
            scratch: Option<Reg>,
        ) -> Aarch64Branch {
            let scratch = guard_scratch(scratch, mask_reg);
            AsmProgram::from([
                Inst::Uminv(scratch, mask_reg),
                Inst::FmovToGp(scratch),
                Inst::mvn_w(X16, X16),
            ])
            .assemble(code);
            Aarch64Branch::Cond(super::cbz_w16(code))
        }

        fn emit_jump(&mut self, code: &mut Vec<u8>) -> Aarch64Branch {
            Aarch64Branch::Uncond(super::b_placeholder(code))
        }

        fn patch_branch(&mut self, code: &mut Vec<u8>, branch: Aarch64Branch, target: usize) {
            match branch {
                Aarch64Branch::Cond(c) => c.patch(code, target),
                Aarch64Branch::Uncond(b) => b.patch(code, target),
            }
        }

        // AAPCS64: x0 = ctx (read-only in the body's gathers), x1 = out,
        // x2 = groups, x3 = rows, x4 = row-skip bytes, v0..3 = x0/y0/z/w.
        // Loop registers: x5 = batch counter, x6 = row counter; the body's
        // scratch GPRs are x9-x11 (gather), w16 (branch tests), x17 (pool
        // anchor) — all disjoint. The bounds arrive in registers the body never
        // touches, so `latch_bounds` has nothing to do.

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

        /// The prologue's and body's constant loads are X17-relative, so the
        /// anchor has to be inside the emitted function, after the frame.
        fn scaffold_anchor(&mut self, code: &mut Vec<u8>) {
            self.adr_patch_pos = super::emit_adr_x17_placeholder(code);
        }

        fn scaffold_finish(&mut self, code: &mut Vec<u8>) {
            self.finish_pool(code);
        }

        fn slot_store(&mut self, code: &mut Vec<u8>, src: Reg, offset: u32) {
            AsmProgram::from([Inst::str_q(src, frame_slot(offset))]).assemble(code);
        }

        fn slot_load(&mut self, code: &mut Vec<u8>, dst: Reg, offset: u32) {
            AsmProgram::from([Inst::ldr_q(dst, frame_slot(offset))]).assemble(code);
        }

        fn counter_clear(&mut self, code: &mut Vec<u8>, counter: Counter) {
            AsmProgram::from([table::Movz::new(counter_reg(counter), 0)]).assemble(code);
        }

        fn counter_step(&mut self, code: &mut Vec<u8>, counter: Counter) {
            let r = counter_reg(counter);
            AsmProgram::from([table::AddI64::new(r, r, table::Imm12(1))]).assemble(code);
        }

        fn branch_if_counter_done(
            &mut self,
            code: &mut Vec<u8>,
            counter: Counter,
        ) -> Aarch64Branch {
            AsmProgram::from([table::CmpI64::new(counter_reg(counter), bound_reg(counter))])
                .assemble(code);
            Aarch64Branch::Cond(super::b_hs(code))
        }

        fn store_result(&mut self, code: &mut Vec<u8>, src: Reg) {
            AsmProgram::from([Inst::str_q(
                src,
                Mem {
                    base: X1,
                    offset: 0,
                },
            )])
            .assemble(code);
        }

        fn advance_out(&mut self, code: &mut Vec<u8>, step: OutStep) {
            match step {
                OutStep::Batch => {
                    AsmProgram::from([table::AddI64::new(
                        X1,
                        X1,
                        table::Imm12(self.file.vector_bytes as u16),
                    )])
                    .assemble(code);
                }
                OutStep::RowSkip => {
                    AsmProgram::from([table::AddI64::new(X1, X1, X4)]).assemble(code);
                }
            }
        }

        fn add_scalar(&mut self, code: &mut Vec<u8>, dst: Reg, scratch: Reg, scalar: f32) {
            super::emit_fmov_imm(code, scratch, scalar);
            AsmProgram::from([Inst::Fadd(dst, dst, scratch)]).assemble(code);
        }

        fn emit_ret(&mut self, code: &mut Vec<u8>) {
            AsmProgram::from([Inst::Ret]).assemble(code);
        }
    }

    /// The register each loop counter lives in.
    const fn counter_reg(counter: Counter) -> Xr {
        match counter {
            Counter::Batch => X5,
            Counter::Row => X6,
        }
    }

    /// The register each counter is compared against — both arrive as
    /// arguments and stay put, since nothing in the body writes them.
    const fn bound_reg(counter: Counter) -> Xr {
        match counter {
            Counter::Batch => X2,
            Counter::Row => X3,
        }
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

        // 1. Emit reloads (from stack or rematerialized constants)
        for reload in &plan.reloads {
            match reload {
                Reload::FromStack { target, slot } => {
                    AsmProgram::from([Inst::ldr_q(*target, frame_slot(slot.offset()))])
                        .assemble(code);
                }
                Reload::Const { target, val_bits } => {
                    emit_const_load(code, *target, *val_bits, pool);
                }
            }
        }

        // 2. Emit setup MOV (for FMLA accumulator or BSL mask)
        if let Some((dst, src)) = plan.setup_mov {
            AsmProgram::from([Inst::mov(dst, src)]).assemble(code);
        }

        // 3. Emit main op
        match &plan.op {
            ResolvedOp::Nop => {}
            ResolvedOp::LoadConst { dst, val_bits } => {
                emit_const_load(code, *dst, *val_bits, pool);
            }
            ResolvedOp::Unary { op, dst, src } => {
                emit_unary(code, *op, *dst, *src, plan.scratch.temp(0));
            }
            ResolvedOp::ShiftImm {
                op,
                dst,
                src,
                amount,
            } => {
                super::emit_shift_imm(code, *op, *dst, *src, *amount);
            }
            ResolvedOp::Gather { dst, idx, slot } => {
                // dst = buffer[slot][idx], via four scalar loads (NEON has no
                // native gather). The context pointer (array of buffer base
                // pointers) is caller-provided in x0 per AAPCS64 — disjoint from
                // the coordinate vectors in v0..3 and never touched by the
                // arithmetic/const emit, so it survives to here.
                // v30 is declared in `AARCH64_FILE.fixed`, so
                // `RegisterFile::checked` proves it misses the pool, the reload
                // pair and the guard scratch (v28) rather than a comment claiming
                // it; x9-x11 are caller-saved GPR scratch clear of the branch
                // guard (w16) and the const-pool anchor (x17).
                let idx_int = crate::emit::declared_temp(plan.scratch.temp(0));
                const BASE_GPR: PtrReg = ptr::X9;
                const IDX_GPR: Gpr = gpr::X10;
                const VAL_GPR: Gpr = gpr::X11;
                /// Bytes per pointer in the context array.
                const PTR_BYTES: u32 = 8;
                AsmProgram::from([
                    Inst::Fcvtzs(idx_int, *idx),
                    Inst::ldr_x(
                        BASE_GPR,
                        Mem {
                            base: X0,
                            offset: u32::from(*slot) * PTR_BYTES,
                        },
                    ),
                    Inst::UmovW {
                        dst: IDX_GPR,
                        src: idx_int,
                        lane: 0,
                    },
                    Inst::ldr_w(
                        VAL_GPR,
                        MemIndexed {
                            base: BASE_GPR,
                            index: IDX_GPR,
                        },
                    ),
                    Inst::InsW {
                        dst: *dst,
                        lane: 0,
                        src: VAL_GPR,
                    },
                    Inst::UmovW {
                        dst: IDX_GPR,
                        src: idx_int,
                        lane: 1,
                    },
                    Inst::ldr_w(
                        VAL_GPR,
                        MemIndexed {
                            base: BASE_GPR,
                            index: IDX_GPR,
                        },
                    ),
                    Inst::InsW {
                        dst: *dst,
                        lane: 1,
                        src: VAL_GPR,
                    },
                    Inst::UmovW {
                        dst: IDX_GPR,
                        src: idx_int,
                        lane: 2,
                    },
                    Inst::ldr_w(
                        VAL_GPR,
                        MemIndexed {
                            base: BASE_GPR,
                            index: IDX_GPR,
                        },
                    ),
                    Inst::InsW {
                        dst: *dst,
                        lane: 2,
                        src: VAL_GPR,
                    },
                    Inst::UmovW {
                        dst: IDX_GPR,
                        src: idx_int,
                        lane: 3,
                    },
                    Inst::ldr_w(
                        VAL_GPR,
                        MemIndexed {
                            base: BASE_GPR,
                            index: IDX_GPR,
                        },
                    ),
                    Inst::InsW {
                        dst: *dst,
                        lane: 3,
                        src: VAL_GPR,
                    },
                ])
                .assemble(code);
            }
            ResolvedOp::Uniform { dst, load } => {
                super::emit_uniform_load(code, *dst, *load);
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
            ResolvedOp::DecomposedMulAdd {
                dst,
                a,
                b,
                c,
                c_deferred,
            } => {
                // FMUL(dst, a, b) — consumes a and b (loaded upfront).
                AsmProgram::from([Inst::Fmul(*dst, *a, *b)]).assemble(code);
                // Reload c after FMUL (c may reuse tmp_op which held b).
                emit_deferred(code, *c, c_deferred.as_ref(), pool);
                // FADD(dst, dst, c)
                AsmProgram::from([Inst::Fadd(*dst, *dst, *c)]).assemble(code);
            }
            ResolvedOp::Select {
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
    /// Emit a deferred reload: either from stack or rematerialized constant.
    fn emit_deferred(
        code: &mut Vec<u8>,
        target: Reg,
        deferred: Option<&DeferredReload>,
        pool: &ConstPool,
    ) {
        match deferred {
            Some(DeferredReload::FromStack(slot)) => {
                AsmProgram::from([Inst::ldr_q(target, frame_slot(slot.offset()))]).assemble(code);
            }
            Some(DeferredReload::Const(val_bits)) => {
                emit_const_load(code, target, *val_bits, pool);
            }
            None => {}
        }
    }
}

// =============================================================================
// =============================================================================
// General-purpose and Pointer registers
// =============================================================================

/// The aarch64 general register file (`x0`–`x30`, plus the zero register).
///
/// A distinct type from [`Reg`], which names the *vector* file `v0`–`v31`.
/// They are different files that share a numbering, so `Xr(1)` is `x1` and
/// `Reg(1)` is `v1`, and neither can be passed where the other belongs.
pub type Xr = Gpr;

/// Constructor function for backwards compatibility with `Xr(u8)`.
#[inline(always)]
#[must_use]
#[allow(non_snake_case)]
pub const fn Xr(r: u8) -> Gpr {
    Gpr(r)
}

/// Physical pointer registers used by AAPCS64 emitted kernels.
pub mod ptr {
    use super::PtrReg;

    /// 1st argument: context pointer — array of bound buffer bases.
    pub const X0: PtrReg = PtrReg(0);
    /// 2nd argument: output pointer, advanced per batch and per row.
    pub const X1: PtrReg = PtrReg(1);
    /// Scratch base register for gather.
    pub const X9: PtrReg = PtrReg(9);
    /// IP0, intra-procedure scratch (displacement fallback).
    pub const X16: PtrReg = PtrReg(16);
    /// IP1, intra-procedure scratch (constant-pool anchor).
    pub const X17: PtrReg = PtrReg(17);
    /// The stack pointer — spill slots are addressed from it.
    pub const SP: PtrReg = PtrReg(31);
}

/// AAPCS64 general-purpose registers (integers, counters, indices, bounds).
pub mod gpr {
    use super::Gpr;

    /// The zero register in positions where `xzr` is meant.
    pub const XZR: Gpr = Gpr(31);
    /// 3rd argument: group count (the inner bound).
    pub const X2: Gpr = Gpr(2);
    /// 4th argument: row count (the outer bound).
    pub const X3: Gpr = Gpr(3);
    /// 5th argument: row-skip in bytes.
    pub const X4: Gpr = Gpr(4);
    /// Inner (batch) loop counter.
    pub const X5: Gpr = Gpr(5);
    /// Outer (row) loop counter.
    pub const X6: Gpr = Gpr(6);
    /// Scratch: gather index.
    pub const X10: Gpr = Gpr(10);
    /// Scratch: gather value.
    pub const X11: Gpr = Gpr(11);
}

/// AAPCS64 registers the emitted kernels use.
pub mod xr {
    pub use super::gpr::*;
    pub use super::ptr::*;
}

pub use table::Imm12;

/// `movz dst, #imm16` — also how `mov dst, xzr` is spelled, as `movz dst, #0`.
#[inline(always)]
pub fn movz(code: &mut Vec<u8>, dst: impl Into<Gpr>, imm: u16) {
    let dst = dst.into();
    emit32(code, 0xD280_0000 | ((imm as u32) << 5) | dst.0 as u32);
}

/// `cmp lhs, rhs` — `subs xzr, lhs, rhs`, setting the flags [`b_hs`] reads.
#[inline(always)]
pub fn cmp(code: &mut Vec<u8>, lhs: impl Into<Gpr>, rhs: impl Into<Gpr>) {
    let lhs = lhs.into();
    let rhs = rhs.into();
    emit32(
        code,
        0xEB00_0000 | ((rhs.0 as u32) << 16) | ((lhs.0 as u32) << 5) | 31,
    );
}

/// What an [`add`] can add: another register, a pointer register, or a 12-bit immediate.
///
/// As on x86, the operand's *type* selects the encoding, so the mnemonic stays
/// one name instead of splitting into `add_reg` / `add_imm`.
pub trait AddOperand {
    /// Emit `add dst, src, self`.
    fn add_into(self, code: &mut Vec<u8>, dst: Gpr, src: Gpr);
}

impl AddOperand for Gpr {
    #[inline(always)]
    fn add_into(self, code: &mut Vec<u8>, dst: Gpr, src: Gpr) {
        table::AddI64::new(dst, src, self).emit_into(code);
    }
}

impl AddOperand for PtrReg {
    #[inline(always)]
    fn add_into(self, code: &mut Vec<u8>, dst: Gpr, src: Gpr) {
        table::AddI64::new(dst, src, self.as_gpr()).emit_into(code);
    }
}

impl AddOperand for Imm12 {
    #[inline(always)]
    fn add_into(self, code: &mut Vec<u8>, dst: Gpr, src: Gpr) {
        table::AddI64::new(dst, src, self).emit_into(code);
    }
}

/// `add dst, src, operand`
#[inline(always)]
pub fn add(code: &mut Vec<u8>, dst: impl Into<Gpr>, src: impl Into<Gpr>, operand: impl AddOperand) {
    operand.add_into(code, dst.into(), src.into());
}

/// `mvn w<dst>, w<src>` — bitwise NOT of a 32-bit general register.
///
/// `ORN Wd, WZR, Wm`; the guard path uses it to turn "all lanes set" into
/// zero so a following `cbz` tests it.
#[inline(always)]
pub fn mvn_w(code: &mut Vec<u8>, dst: impl Into<Gpr>, src: impl Into<Gpr>) {
    let dst = dst.into();
    let src = src.into();
    emit32(code, 0x2A20_03E0 | ((src.0 as u32) << 16) | dst.0 as u32);
}

/// `ret`
#[inline(always)]
pub fn ret(code: &mut Vec<u8>) {
    emit32(code, 0xD65F_03C0);
}

/// A conditional branch whose 19-bit displacement is not filled in yet.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[must_use = "an unpatched branch falls through to itself"]
pub struct Cond19(usize);

impl Cond19 {
    /// Point the branch at `target`, a byte offset into the same buffer.
    #[inline(always)]
    pub fn patch(self, code: &mut [u8], target: usize) {
        let offset = ((target as i64 - self.0 as i64) / 4) as i32;
        assert!(
            (-(1 << 18)..(1 << 18)).contains(&offset),
            "19-bit branch offset {offset} out of range (±1MB)"
        );
        let imm19 = (offset as u32) & 0x7FFFF;
        let existing = u32::from_le_bytes([
            code[self.0],
            code[self.0 + 1],
            code[self.0 + 2],
            code[self.0 + 3],
        ]);
        let patched = (existing & 0xFF00_001F) | (imm19 << 5);
        code[self.0..self.0 + 4].copy_from_slice(&patched.to_le_bytes());
    }
}

/// An unconditional branch whose 26-bit displacement is not filled in yet.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[must_use = "an unpatched branch falls through to itself"]
pub struct Rel26(usize);

impl Rel26 {
    /// Point the branch at `target`, a byte offset into the same buffer.
    #[inline(always)]
    pub fn patch(self, code: &mut [u8], target: usize) {
        let offset = ((target as i64 - self.0 as i64) / 4) as i32;
        assert!(
            (-(1 << 25)..(1 << 25)).contains(&offset),
            "26-bit branch offset {offset} out of range (±128MB)"
        );
        let imm26 = (offset as u32) & 0x03FF_FFFF;
        let existing = u32::from_le_bytes([
            code[self.0],
            code[self.0 + 1],
            code[self.0 + 2],
            code[self.0 + 3],
        ]);
        let patched = (existing & 0xFC00_0000) | imm26;
        code[self.0..self.0 + 4].copy_from_slice(&patched.to_le_bytes());
    }
}

/// `cbz w16, #0` — branch if W16 == 0 (mask all-false), awaiting patch.
#[inline(always)]
pub fn cbz_w16(code: &mut Vec<u8>) -> Cond19 {
    let at = code.len();
    emit32(code, 0x3400_0010);
    Cond19(at)
}

/// `b.hs` — taken when the previous [`cmp`] found `lhs >= rhs` unsigned.
#[inline(always)]
pub fn b_hs(code: &mut Vec<u8>) -> Cond19 {
    let at = code.len();
    emit32(code, 0x5400_0002);
    Cond19(at)
}

/// `b #0` — unconditional forward branch placeholder awaiting patch.
#[inline(always)]
pub fn b_placeholder(code: &mut Vec<u8>) -> Rel26 {
    let at = code.len();
    emit32(code, 0x1400_0000);
    Rel26(at)
}

/// `b target` — an unconditional branch to an already-known offset.
///
/// Unlike the x86 counterpart this needs no fixup token: every use in the
/// scaffold jumps *backwards* to a label already emitted.
#[inline(always)]
pub fn b(code: &mut Vec<u8>, target: usize) {
    let rel = ((target as i64 - code.len() as i64) / 4) as i32;
    emit32(code, 0x1400_0000 | ((rel as u32) & 0x03FF_FFFF));
}

#[cfg(test)]
mod xr_tests {
    use super::xr::*;
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
        // MOVZ Xd, #imm16 — `mov xN, xzr` is `movz xN, #0`.
        assert_eq!(word(|c| movz(c, X6, 0)), 0xD280_0006);
        assert_eq!(word(|c| movz(c, X5, 0)), 0xD280_0005);
        // SUBS XZR, Xn, Xm
        assert_eq!(word(|c| cmp(c, X6, X3)), 0xEB03_00DF);
        assert_eq!(word(|c| cmp(c, X5, X2)), 0xEB02_00BF);
        // ADD Xd, Xn, #imm12
        assert_eq!(word(|c| add(c, X1, X1, Imm12(16))), 0x9100_4021);
        assert_eq!(word(|c| add(c, X5, X5, Imm12(1))), 0x9100_04A5);
        assert_eq!(word(|c| add(c, X6, X6, Imm12(1))), 0x9100_04C6);
        // ADD Xd, Xn, Xm
        assert_eq!(word(|c| add(c, X1, X1, X4)), 0x8B04_0021);
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
        assert_eq!(word(ret), 0xD65F_03C0);
    }

    /// The immediate and register forms of `add` are different instructions
    /// reached through one name; the operand type is what chooses.
    #[test]
    fn the_operand_type_selects_the_add_encoding() {
        assert_eq!(
            word(|c| add(c, X1, X1, Imm12(16))) >> 24,
            0x91,
            "immediate form"
        );
        assert_eq!(word(|c| add(c, X1, X1, X4)) >> 24, 0x8B, "register form");
    }

    /// A conditional branch is emitted as a placeholder and patched to a
    /// forward target; the displacement counts instructions, not bytes.
    #[test]
    fn conditional_branches_patch_forward_in_instructions() {
        let mut c = Vec::new();
        let br = b_hs(&mut c);
        c.resize(24, 0); // three more instructions
        br.patch(&mut c, 24);
        let w = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        assert_eq!((w >> 5) & 0x7FFFF, 6, "24 bytes ahead is 6 instructions");
        assert_eq!(w & 0xF, 0x2, "cond = HS");
    }

    /// An unconditional backward branch encodes a negative instruction count.
    #[test]
    fn unconditional_branches_go_backwards() {
        let mut c = vec![0u8; 16];
        b(&mut c, 0);
        let w = u32::from_le_bytes([c[16], c[17], c[18], c[19]]);
        assert_eq!(w >> 26, 0x05, "B opcode");
        // -4 instructions, in 26-bit two's complement.
        assert_eq!(w & 0x03FF_FFFF, (-4i32 as u32) & 0x03FF_FFFF);
    }

    /// `Xr` and `Reg` name different files; the same index is a different
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
