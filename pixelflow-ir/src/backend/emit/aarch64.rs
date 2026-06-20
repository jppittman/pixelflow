//! ARM64/NEON instruction encoding.
//!
//! Each function emits raw machine code bytes for one instruction (or a small fixed sequence).
//! These are the "atoms" that compound operations are built from.

use super::Reg;
use crate::kind::OpKind;

// =============================================================================
// Instruction Encoding Helpers
// =============================================================================

/// Encode a NEON 3-same instruction (binary vector ops).
/// Format: Vd.4S, Vn.4S, Vm.4S
#[inline]
fn encode_3same(opcode: u32, dst: Reg, src1: Reg, src2: Reg) -> u32 {
    opcode | (dst.0 as u32 & 0x1F) | ((src1.0 as u32 & 0x1F) << 5) | ((src2.0 as u32 & 0x1F) << 16)
}

/// Encode a NEON 2-reg misc instruction (unary vector ops).
/// Format: Vd.4S, Vn.4S
#[inline]
fn encode_2misc(opcode: u32, dst: Reg, src: Reg) -> u32 {
    opcode | (dst.0 as u32 & 0x1F) | ((src.0 as u32 & 0x1F) << 5)
}

/// Write a 32-bit instruction to the code buffer.
#[inline]
pub fn emit32(code: &mut Vec<u8>, inst: u32) {
    code.extend_from_slice(&inst.to_le_bytes());
}

// =============================================================================
// Load / Store
// =============================================================================

/// LDR Vd, [X0, #offset] - Load 128-bit vector from base + offset
pub fn emit_ldr_voff(code: &mut Vec<u8>, dst: Reg, offset: u16) {
    // LDR Qt, [Xn, #imm] - 128-bit load
    // Encoding: 0x3DC00000 | (imm12 << 10) | (Rn << 5) | Rt
    // imm12 is offset/16 for 128-bit loads
    let imm12 = (offset / 16) as u32;
    let inst = (0x3DC00000 | (imm12 << 10)) | (dst.0 as u32); // X0 as base
    emit32(code, inst);
}

/// STR Vt, [X0, #offset] - Store 128-bit vector to base + offset
pub fn emit_str_voff(code: &mut Vec<u8>, src: Reg, offset: u16) {
    let imm12 = (offset / 16) as u32;
    let inst = (0x3D800000 | (imm12 << 10)) | (src.0 as u32);
    emit32(code, inst);
}

/// LDR Vd, [SP, #offset] - Load 128-bit vector from stack.
///
/// Small offsets (< 65536, 16-byte aligned) use single-instruction scaled
/// immediate addressing. Large offsets use X16 (IP0) as scratch to compute
/// the address, then load from [X16].
pub fn emit_ldr_sp(code: &mut Vec<u8>, dst: Reg, offset: u32) {
    assert!(
        offset.is_multiple_of(16),
        "emit_ldr_sp: offset {offset} not 16-byte aligned"
    );
    let imm12 = offset / 16;
    if imm12 <= 4095 {
        // LDR Qt, [SP, #imm12*16]
        let inst = 0x3DC00000 | (imm12 << 10) | (31 << 5) | (dst.0 as u32);
        emit32(code, inst);
    } else {
        // ADD X16, SP, #offset  (may need multiple instructions)
        emit_add_x16_sp(code, offset);
        // LDR Qt, [X16]
        let inst = 0x3DC00000 | (16 << 5) | (dst.0 as u32);
        emit32(code, inst);
    }
}

/// STR Vt, [SP, #offset] - Store 128-bit vector to stack.
///
/// Small offsets use scaled immediate. Large offsets use X16 scratch.
pub fn emit_str_sp(code: &mut Vec<u8>, src: Reg, offset: u32) {
    assert!(
        offset.is_multiple_of(16),
        "emit_str_sp: offset {offset} not 16-byte aligned"
    );
    let imm12 = offset / 16;
    if imm12 <= 4095 {
        // STR Qt, [SP, #imm12*16]
        let inst = 0x3D800000 | (imm12 << 10) | (31 << 5) | (src.0 as u32);
        emit32(code, inst);
    } else {
        emit_add_x16_sp(code, offset);
        // STR Qt, [X16]
        let inst = 0x3D800000 | (16 << 5) | (src.0 as u32);
        emit32(code, inst);
    }
}

/// ADD X16, SP, #offset - Compute stack address in scratch register.
///
/// ARM64 ADD immediate is 12-bit (max 4095). For larger offsets, emit
/// multiple ADD instructions.
fn emit_add_x16_sp(code: &mut Vec<u8>, offset: u32) {
    // First: MOV X16, SP  (ADD X16, SP, #0 — but we'll fold the first chunk)
    let mut remaining = offset;
    let first_chunk = remaining.min(4080);
    // ADD X16, SP, #first_chunk
    let inst = 0x91000000 | (first_chunk << 10) | (31 << 5) | 16;
    emit32(code, inst);
    remaining -= first_chunk;
    while remaining > 0 {
        let chunk = remaining.min(4080);
        // ADD X16, X16, #chunk
        let inst = 0x91000000 | (chunk << 10) | (16 << 5) | 16;
        emit32(code, inst);
        remaining -= chunk;
    }
}

/// SUB SP, SP, #imm - Allocate stack frame.
///
/// ARM64 ADD/SUB immediate has a 12-bit field (max 4095). For larger frames,
/// we emit multiple instructions, each subtracting up to 4080 (largest
/// 16-byte-aligned value in 12 bits).
pub fn emit_sub_sp(code: &mut Vec<u8>, size: u32) {
    let mut remaining = size;
    while remaining > 0 {
        let chunk = remaining.min(4080);
        assert!(chunk <= 4095, "ARM64 immediate overflow in emit_sub_sp");
        let inst = 0xD10003FF | (chunk << 10);
        emit32(code, inst);
        remaining -= chunk;
    }
}

/// ADD SP, SP, #imm - Deallocate stack frame.
///
/// See `emit_sub_sp` for why we emit multiple instructions.
pub fn emit_add_sp(code: &mut Vec<u8>, size: u32) {
    let mut remaining = size;
    while remaining > 0 {
        let chunk = remaining.min(4080);
        assert!(chunk <= 4095, "ARM64 immediate overflow in emit_add_sp");
        let inst = 0x910003FF | (chunk << 10);
        emit32(code, inst);
        remaining -= chunk;
    }
}

// =============================================================================
// Arithmetic - Single Instructions
// =============================================================================

/// FADD Vd.4S, Vn.4S, Vm.4S
pub fn emit_fadd(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x4E20D400, dst, src1, src2));
}

/// FSUB Vd.4S, Vn.4S, Vm.4S
pub fn emit_fsub(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x4EA0D400, dst, src1, src2));
}

/// FMUL Vd.4S, Vn.4S, Vm.4S
pub fn emit_fmul(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x6E20DC00, dst, src1, src2));
}

/// FDIV Vd.4S, Vn.4S, Vm.4S
pub fn emit_fdiv(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x6E20FC00, dst, src1, src2));
}

/// FMLA Vd.4S, Vn.4S, Vm.4S (fused multiply-add: Vd += Vn * Vm)
pub fn emit_fmla(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x4E20CC00, dst, src1, src2));
}

/// FMIN Vd.4S, Vn.4S, Vm.4S
pub fn emit_fmin(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x4EA0F400, dst, src1, src2));
}

/// FMAX Vd.4S, Vn.4S, Vm.4S
pub fn emit_fmax(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x4E20F400, dst, src1, src2));
}

/// FSQRT Vd.4S, Vn.4S
pub fn emit_fsqrt(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit32(code, encode_2misc(0x6EA1F800, dst, src));
}

/// FABS Vd.4S, Vn.4S
pub fn emit_fabs(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit32(code, encode_2misc(0x4EA0F800, dst, src));
}

/// FNEG Vd.4S, Vn.4S
pub fn emit_fneg(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit32(code, encode_2misc(0x6EA0F800, dst, src));
}

/// NOT Vd.16B, Vn.16B (bitwise NOT, 2-register miscellaneous)
pub fn emit_not(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit32(code, encode_2misc(0x2E205800, dst, src));
}

/// FRINTM Vd.4S, Vn.4S (floor)
pub fn emit_frintm(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit32(code, encode_2misc(0x4E219800, dst, src));
}

/// FRINTP Vd.4S, Vn.4S (ceil)
pub fn emit_frintp(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit32(code, encode_2misc(0x4EA18800, dst, src));
}

// =============================================================================
// Approximate operations (estimate + refinement)
// =============================================================================

/// FRSQRTE + FRSQRTS refinement (~3 instructions for rsqrt)
pub fn emit_frsqrt(code: &mut Vec<u8>, dst: Reg, src: Reg, scratch: Reg) {
    // est = frsqrte(src)
    emit32(code, encode_2misc(0x6EA1D800, dst, src));
    // scratch = est * est
    emit32(code, encode_3same(0x6E20DC00, scratch, dst, dst));
    // scratch = frsqrts(src, scratch) = (3 - src * scratch) / 2
    emit32(code, encode_3same(0x4EA0FC00, scratch, src, scratch));
    // dst = est * scratch (refined)
    emit32(code, encode_3same(0x6E20DC00, dst, dst, scratch));
}

/// FRECPE + FRECPS refinement (~3 instructions for recip)
pub fn emit_frecip(code: &mut Vec<u8>, dst: Reg, src: Reg, scratch: Reg) {
    // est = frecpe(src)
    emit32(code, encode_2misc(0x4EA1D800, dst, src));
    // scratch = frecps(src, est) = 2 - src * est
    emit32(code, encode_3same(0x4E20FC00, scratch, src, dst));
    // dst = est * scratch (refined)
    emit32(code, encode_3same(0x6E20DC00, dst, dst, scratch));
}

// =============================================================================
// Comparisons
// =============================================================================

/// FCMGT Vd.4S, Vn.4S, Vm.4S (greater than)
pub fn emit_fcmgt(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x6EA0E400, dst, src1, src2));
}

/// FCMGE Vd.4S, Vn.4S, Vm.4S (greater or equal)
pub fn emit_fcmge(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x6E20E400, dst, src1, src2));
}

/// FCMEQ Vd.4S, Vn.4S, Vm.4S (equal)
pub fn emit_fcmeq(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x4E20E400, dst, src1, src2));
}

// =============================================================================
// Selection / Blending
// =============================================================================

/// BSL Vd.16B, Vn.16B, Vm.16B (bitwise select: Vd = (Vd & Vn) | (~Vd & Vm))
pub fn emit_bsl(code: &mut Vec<u8>, mask: Reg, if_true: Reg, if_false: Reg) {
    emit32(code, encode_3same(0x6E601C00, mask, if_true, if_false));
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
pub fn emit_fmov_imm(code: &mut Vec<u8>, dst: Reg, val: f32, _scratch: [Reg; 4]) {
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

/// Duplicate scalar to all lanes: DUP Vd.4S, Vn.S[0]
pub fn emit_dup_s0(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit32(code, 0x4E040400 | (dst.0 as u32) | ((src.0 as u32) << 5));
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
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AdrMode {
    Adr,
    Adrp,
}

impl From<bool> for AdrMode {
    fn from(b: bool) -> Self {
        if b { AdrMode::Adrp } else { AdrMode::Adr }
    }
}

pub fn patch_adr_or_adrp(code: &mut [u8], adr_pos: usize, target_pos: usize, mode: AdrMode) {
    let is_adrp = mode == AdrMode::Adrp;
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

/// Emit `LDR Qt, [X17, #imm]` — 128-bit load from constant pool base + offset.
///
/// Encoding: `0x3DC00000 | (imm12 << 10) | (Rn << 5) | Rt`
/// where imm12 = byte_offset / 16, Rn = 17 (X17).
pub fn emit_ldr_q_x17(code: &mut Vec<u8>, dst: Reg, byte_offset: u16) {
    assert!(
        byte_offset.is_multiple_of(16),
        "constant pool offset {} not 16-byte aligned",
        byte_offset
    );
    let imm12 = (byte_offset / 16) as u32;
    assert!(imm12 < 4096, "constant pool offset too large");
    emit32(
        code,
        0x3DC00000 | (imm12 << 10) | (17 << 5) | (dst.0 as u32),
    );
}

/// Emit a constant pool entry: 16 bytes = f32 value splatted 4x (fills a 128-bit NEON register).
pub fn emit_pool_entry(code: &mut Vec<u8>, val_bits: u32) {
    let bytes = val_bits.to_le_bytes();
    for _ in 0..4 {
        code.extend_from_slice(&bytes);
    }
}

// =============================================================================
// Integer Vector Operations (for bit manipulation in transcendentals)
// =============================================================================

/// USHR Vd.4S, Vn.4S, #shift (unsigned shift right by immediate)
fn emit_ushr(code: &mut Vec<u8>, dst: Reg, src: Reg, shift: u8) {
    // Encoding: 0x6F200400 | ((32 - shift) << 16) as immh:immb
    // For .4S: immh = 001x, so (32-shift) in bits [19:16]
    let immhb = (64 - shift as u32) & 0x3F; // USHR uses (immh:immb) = (size*2 - shift)
    let inst = 0x6F200400 | (dst.0 as u32) | ((src.0 as u32) << 5) | (immhb << 16);
    emit32(code, inst);
}

/// SHL Vd.4S, Vn.4S, #shift (shift left by immediate)
fn emit_shl(code: &mut Vec<u8>, dst: Reg, src: Reg, shift: u8) {
    // For .4S: immh:immb = shift + 32
    let immhb = (shift as u32) + 32;
    let inst = 0x4F005400 | (dst.0 as u32) | ((src.0 as u32) << 5) | (immhb << 16);
    emit32(code, inst);
}

/// SUB Vd.4S, Vn.4S, Vm.4S (integer subtract)
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
fn emit_sub_i32(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x6EA08400, dst, src1, src2));
}

/// ADD Vd.4S, Vn.4S, Vm.4S (integer add)
fn emit_add_i32(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x4EA08400, dst, src1, src2));
}

/// AND Vd.16B, Vn.16B, Vm.16B (bitwise AND)
fn emit_and(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x4E201C00, dst, src1, src2));
}

/// ORR Vd.16B, Vn.16B, Vm.16B (bitwise OR)
fn emit_orr(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit32(code, encode_3same(0x4EA01C00, dst, src1, src2));
}

/// FCVTZS Vd.4S, Vn.4S (float to signed int, round toward zero)
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
fn emit_fcvtzs(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit32(code, encode_2misc(0x4EA1B800, dst, src));
}

/// SCVTF Vd.4S, Vn.4S (signed int to float)
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
fn emit_scvtf(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit32(code, encode_2misc(0x4E21D800, dst, src));
}

/// FRINTA Vd.4S, Vn.4S (round to nearest, ties away from zero)
fn emit_frinta(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit32(code, encode_2misc(0x6E218800, dst, src));
}

/// MOV Vd.16B, Vn.16B (register copy via ORR)
fn emit_mov(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    if dst.0 != src.0 {
        emit_orr(code, dst, src, src);
    }
}

// =============================================================================
// Constant Loading Helpers
// =============================================================================

/// Load a 32-bit constant (as raw bits) into all 4 lanes of a vector register.
/// Uses W16 (IP0) as a GP scratch register.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
fn emit_u32_const(code: &mut Vec<u8>, dst: Reg, bits: u32) {
    if bits == 0 {
        // MOVI Vd.4S, #0
        emit32(code, 0x4F000400 | (dst.0 as u32));
        return;
    }
    let lo16 = bits & 0xFFFF;
    let hi16 = bits >> 16;
    // MOVZ W16, #lo16
    emit32(code, 0x52800010 | (lo16 << 5));
    // MOVK W16, #hi16, LSL #16
    emit32(code, 0x72A00010 | (hi16 << 5));
    // DUP Vd.4S, W16
    emit32(code, 0x4E040C00 | (dst.0 as u32) | (16 << 5));
}

// =============================================================================
// Transcendental Builtins — inline polynomial sequences
// =============================================================================
//
// Each builtin translates a pixelflow-core NEON implementation into direct
// machine code emission. Same coefficients, same algorithms.
//
// Register contract:
//   dst  — output register (also used as Horner accumulator)
//   src  — input register (read-only, never clobbered)
//   s0-s2 — scratch registers from scratch[0..2] (clobbered)
//   s3    — scratch[3], used by composition builtins (cos, tan, pow)

/// log2(x) — base-2 logarithm via bit manipulation + polynomial.
///
/// Translated from pixelflow-core arm.rs F32x4::log2().
/// Uses exponent extraction and 5-coefficient polynomial on [√2/2, √2].
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(crate) fn emit_log2_builtin(
    code: &mut Vec<u8>,
    pool: &mut super::ConstPool,
    dst: Reg,
    src: Reg,
    scratch: [Reg; 4],
) -> Result<(), &'static str> {
    let s0 = scratch[0]; // n (exponent as float), then Horner scratch
    let s1 = scratch[1]; // f (normalized mantissa), alive through Horner
    let s2 = scratch[2]; // general scratch

    // Phase 1: Extract exponent as float
    // exp_bits = src_as_u32 >> 23
    emit_ushr(code, s0, src, 23);
    // bias = 127 (as integer)
    emit_u32_const(code, s2, 127);
    // n_i32 = exp_bits - 127 (integer subtraction)
    emit_sub_i32(code, s0, s0, s2);
    // n = float(n_i32) — convert to float, result in dst (will hold n)
    emit_scvtf(code, dst, s0);

    // Phase 2: Extract mantissa in [1, 2)
    // mantissa_mask = 0x007FFFFF
    emit_u32_const(code, s2, 0x007FFFFF);
    // s1 = src & mantissa_mask (extract mantissa bits)
    emit_and(code, s1, src, s2);
    // one_bits = 0x3F800000 (1.0 as bits)
    emit_u32_const(code, s2, 0x3F800000);
    // s1 = mantissa_bits | 1.0_bits → f in [1, 2)
    emit_orr(code, s1, s1, s2);

    // Phase 3: Reduce to [√2/2, √2] for better accuracy
    // mask = (f >= √2)
    let sqrt2 = pool.push_f32(core::f32::consts::SQRT_2)?;
    emit_ldr_q_x17(code, s2, sqrt2);
    emit_fcmge(code, s2, s1, s2); // s2 = mask (all-ones where f >= √2)
    // adjust = 1.0 & mask
    let one = pool.push_f32(1.0)?;
    emit_ldr_q_x17(code, s0, one);
    emit_and(code, s0, s0, s2); // s0 = adjust (1.0 where f >= √2, 0 elsewhere)
    // n += adjust
    emit_fadd(code, dst, dst, s0);
    // If f >= √2: multiply by 0.5 (divide by 2).
    // Use BSL: s1 = mask ? f*0.5 : f
    // Compute f*0.5 into s0, then BSL s2 (mask), s0 (if_true), s1 (if_false)
    let half = pool.push_f32(0.5)?;
    emit_ldr_q_x17(code, s0, half);
    emit_fmul(code, s0, s1, s0); // s0 = f * 0.5
    emit_bsl(code, s2, s0, s1); // s2 = mask ? f*0.5 : f
    emit_mov(code, s1, s2); // s1 = adjusted f

    // Phase 4: Polynomial log2(f) on [√2/2, √2]
    // Subtract 1 so argument is centered at 0 for the polynomial
    // Actually, the arm.rs impl doesn't subtract 1 — it uses a polynomial
    // fitted to the [√2/2, √2] interval directly. The Horner chain is:
    //   poly = ((((c4*f + c3)*f + c2)*f + c1)*f + c0)
    // Result = n + poly (not n + poly*f)
    //
    // Horner: alternate dst and s2 as accumulator, s1 = f throughout.

    // p = c4*f + c3
    let c4 = pool.push_f32(-0.320_043_5_f32)?;
    emit_ldr_q_x17(code, s2, c4);
    let c3 = pool.push_f32(1.797_496_9_f32)?;
    emit_ldr_q_x17(code, s0, c3);
    emit_fmla(code, s0, s2, s1); // s0 = c3 + c4*f

    // p = p*f + c2
    let c2 = pool.push_f32(-4.198_805_f32)?;
    emit_ldr_q_x17(code, s2, c2);
    emit_fmla(code, s2, s0, s1); // s2 = c2 + p*f

    // p = p*f + c1
    let c1 = pool.push_f32(5.727_023_f32)?;
    emit_ldr_q_x17(code, s0, c1);
    emit_fmla(code, s0, s2, s1); // s0 = c1 + p*f

    // p = p*f + c0
    let c0 = pool.push_f32(-3.005_614_8_f32)?;
    emit_ldr_q_x17(code, s2, c0);
    emit_fmla(code, s2, s0, s1); // s2 = c0 + p*f

    // result = n + poly
    emit_fadd(code, dst, dst, s2); // dst = n + poly

    Ok(())
}

/// exp2(x) — base-2 exponential via floor + polynomial + bit scaling.
///
/// Translated from pixelflow-core arm.rs F32x4::exp2().
/// Uses 5-coefficient minimax polynomial on [0,1) fractional part,
/// then scales by 2^n via integer exponent bit manipulation.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(crate) fn emit_exp2_builtin(
    code: &mut Vec<u8>,
    pool: &mut super::ConstPool,
    dst: Reg,
    src: Reg,
    scratch: [Reg; 4],
) -> Result<(), &'static str> {
    let s0 = scratch[0]; // n = floor(x)
    let s1 = scratch[1]; // f = frac(x), then integer scratch
    let s2 = scratch[2]; // Horner alternating accumulator

    // Phase 1: Split into integer and fractional parts
    // n = floor(x)
    emit_frintm(code, s0, src);
    // f = x - n (fractional part in [0, 1))
    emit_fsub(code, s1, src, s0);

    // Phase 2: Horner polynomial for 2^f
    // p = ((((c4*f + c3)*f + c2)*f + c1)*f + c0)
    // Alternate between dst and s2 as accumulator, s1 = f.

    // p = c4*f + c3
    let c4 = pool.push_f32(0.0135557_f32)?;
    emit_ldr_q_x17(code, s2, c4);
    let c3 = pool.push_f32(0.0520323_f32)?;
    emit_ldr_q_x17(code, dst, c3);
    emit_fmla(code, dst, s2, s1); // dst = c3 + c4*f

    // p = p*f + c2
    let c2 = pool.push_f32(0.2413793_f32)?;
    emit_ldr_q_x17(code, s2, c2);
    emit_fmla(code, s2, dst, s1); // s2 = c2 + p*f

    // p = p*f + c1
    let c1 = pool.push_f32(core::f32::consts::LN_2)?;
    emit_ldr_q_x17(code, dst, c1);
    emit_fmla(code, dst, s2, s1); // dst = c1 + p*f

    // p = p*f + c0 — 1.0 is FMOV-encodable but pool dedup is harmless
    let c0 = pool.push_f32(1.0_f32)?;
    emit_ldr_q_x17(code, s2, c0);
    emit_fmla(code, s2, dst, s1); // s2 = c0 + p*f
    // Polynomial result now in s2.

    // Phase 3: Compute 2^n via bit manipulation
    // 2^n = reinterpret_f32((int(n) + 127) << 23)
    emit_fcvtzs(code, s1, s0); // s1 = int(n) (s1 was f, no longer needed)
    emit_u32_const(code, dst, 127); // dst = 127 (as integer)
    emit_add_i32(code, s1, s1, dst); // s1 = int(n) + 127
    emit_shl(code, s1, s1, 23); // s1 = (int(n) + 127) << 23 = 2^n as IEEE bits

    // Phase 4: result = poly * 2^n
    emit_fmul(code, dst, s2, s1); // dst = poly * scale = 2^x

    Ok(())
}

/// round(x) — round to nearest, ties away from zero. ARM64 FRINTA instruction.
pub fn emit_round_builtin(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_frinta(code, dst, src);
}

/// fract(x) = x - floor(x).
pub fn emit_fract_builtin(code: &mut Vec<u8>, dst: Reg, src: Reg, scratch: [Reg; 4]) {
    let s0 = scratch[0];
    emit_frintm(code, s0, src); // s0 = floor(x)
    emit_fsub(code, dst, src, s0); // dst = x - floor(x)
}

// =============================================================================
// Binary Transcendental Builtins
// =============================================================================

/// pow(x, y) = exp2(y * log2(x)).
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_pow_builtin(
    code: &mut Vec<u8>,
    pool: &mut super::ConstPool,
    dst: Reg,
    src1: Reg,
    src2: Reg,
    scratch: [Reg; 4],
) -> Result<(), &'static str> {
    let s3 = scratch[3];
    // log2(x) → s3 (uses s0, s1, s2 as scratch, reads src1)
    emit_log2_builtin(code, pool, s3, src1, scratch)?;
    // s3 = y * log2(x)
    emit_fmul(code, s3, src2, s3);
    // exp2(s3) → dst (uses s0, s1, s2 as scratch)
    emit_exp2_builtin(code, pool, dst, s3, scratch)
}

/// hypot(x, y) = sqrt(x*x + y*y).
pub fn emit_hypot_builtin(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    // dst = x * x
    emit_fmul(code, dst, src1, src1);
    // dst = dst + y * y  (FMLA: dst += y * y)
    emit_fmla(code, dst, src2, src2);
    // dst = sqrt(dst)
    emit_fsqrt(code, dst, dst);
}

// =============================================================================
// Compound Operations (emit full instruction sequences)
// =============================================================================

/// Emit unary operation - dispatches to appropriate instruction(s)
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_unary(
    code: &mut Vec<u8>,
    _pool: &mut super::ConstPool,
    op: OpKind,
    dst: Reg,
    src: Reg,
    scratch: [Reg; 4],
) -> Result<(), &'static str> {
    match op {
        OpKind::Neg => emit_fneg(code, dst, src),
        OpKind::Abs => emit_fabs(code, dst, src),
        OpKind::Sqrt => emit_fsqrt(code, dst, src),
        OpKind::Rsqrt => emit_frsqrt(code, dst, src, scratch[0]),
        OpKind::Recip => emit_frecip(code, dst, src, scratch[0]),
        OpKind::Floor => emit_frintm(code, dst, src),
        OpKind::Ceil => emit_frintp(code, dst, src),
        OpKind::Round => emit_round_builtin(code, dst, src),
        OpKind::Fract => emit_fract_builtin(code, dst, src, scratch),

        // Bit-manip primitives (integer-domain conversions).
        OpKind::TruncToInt => emit_fcvtzs(code, dst, src), // f32 -> i32 (truncate)
        OpKind::IntToFloat => emit_scvtf(code, dst, src),  // i32 -> f32

        // Transcendentals (sin/cos/tan/exp/exp2/ln/log2/log10/atan/asin/acos) are
        // expanded to primitive arithmetic by `lowering` before codegen, so they
        // never reach a backend. Reaching here means lowering was skipped.
        _ => return Err("unary emit not implemented for this op (lowering not run?)"),
    }
    Ok(())
}

/// Emit a logical shift of i32 lanes by a compile-time immediate.
/// `Shl` -> `SHL`, `Shr` -> `USHR` (logical right). NEON shifts are imm-form.
pub fn emit_shift_imm(
    code: &mut Vec<u8>,
    op: OpKind,
    dst: Reg,
    src: Reg,
    amount: u8,
) -> Result<(), &'static str> {
    match op {
        OpKind::Shl => emit_shl(code, dst, src, amount),
        OpKind::Shr => emit_ushr(code, dst, src, amount),
        _ => return Err("aarch64 emit_shift_imm: not a shift op"),
    }
    Ok(())
}

/// Emit binary operation
pub fn emit_binary(code: &mut Vec<u8>, op: OpKind, dst: Reg, src1: Reg, src2: Reg) {
    match op {
        OpKind::Add => emit_fadd(code, dst, src1, src2),
        OpKind::Sub => emit_fsub(code, dst, src1, src2),
        OpKind::Mul => emit_fmul(code, dst, src1, src2),
        OpKind::Div => emit_fdiv(code, dst, src1, src2),
        OpKind::Min => emit_fmin(code, dst, src1, src2),
        OpKind::Max => emit_fmax(code, dst, src1, src2),

        // Comparisons (result is mask in dst)
        OpKind::Gt => emit_fcmgt(code, dst, src1, src2),
        OpKind::Ge => emit_fcmge(code, dst, src1, src2),
        OpKind::Lt => emit_fcmgt(code, dst, src2, src1), // swap args
        OpKind::Le => emit_fcmge(code, dst, src2, src1),
        OpKind::Eq => emit_fcmeq(code, dst, src1, src2),
        OpKind::Ne => {
            // Ne = not Eq: FCMEQ then bitwise NOT
            emit_fcmeq(code, dst, src1, src2);
            emit_not(code, dst, dst);
        }

        // Bit-manip primitives (integer-domain).
        OpKind::IAdd => emit_add_i32(code, dst, src1, src2),
        OpKind::BitAnd => emit_and(code, dst, src1, src2),
        OpKind::BitOr => emit_orr(code, dst, src1, src2),

        _ => panic!("binary emit not implemented for {:?}", op),
    }
}

/// Emit binary transcendental operation (needs scratch registers).
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_binary_transcendental(
    code: &mut Vec<u8>,
    pool: &mut super::ConstPool,
    op: OpKind,
    dst: Reg,
    src1: Reg,
    src2: Reg,
    scratch: [Reg; 4],
) -> Result<(), &'static str> {
    match op {
        OpKind::Pow => emit_pow_builtin(code, pool, dst, src1, src2, scratch),
        OpKind::Hypot => {
            emit_hypot_builtin(code, dst, src1, src2);
            Ok(())
        }
        // Atan2 is lowered to arithmetic before codegen; only Pow/Hypot still
        // reach a backend (not yet lowered).
        _ => Err("binary transcendental emit not implemented for this op"),
    }
}

/// Emit ternary operation
#[allow(clippy::too_many_arguments)]
pub fn emit_ternary(code: &mut Vec<u8>, op: OpKind, dst: Reg, a: Reg, b: Reg, c: Reg) {
    match op {
        OpKind::MulAdd => {
            // dst = a * b + c
            // FMLA does: dst = dst + src1 * src2
            //
            // Problem: if dst == a and dst != c, copying c to dst would clobber a
            // before we can use it. In that case, use FMUL + FADD instead.
            if (dst.0 == a.0 || dst.0 == b.0) && dst.0 != c.0 {
                // dst overlaps with a or b, can't use FMLA safely
                // Use FMUL + FADD: dst = a * b, then dst = dst + c
                emit_fmul(code, dst, a, b);
                emit_fadd(code, dst, dst, c);
            } else {
                // Safe to use FMLA
                if dst.0 != c.0 {
                    // MOV dst, c first
                    emit32(
                        code,
                        0x4EA01C00 | (dst.0 as u32) | ((c.0 as u32) << 5) | ((c.0 as u32) << 16),
                    );
                }
                emit_fmla(code, dst, a, b);
            }
        }

        OpKind::Select => {
            // dst = a ? b : c (a is mask)
            // Need to move mask to dst first for BSL
            if dst.0 != a.0 {
                emit32(
                    code,
                    0x4EA01C00 | (dst.0 as u32) | ((a.0 as u32) << 5) | ((a.0 as u32) << 16),
                );
            }
            emit_bsl(code, dst, b, c);
        }

        OpKind::Clamp => {
            // dst = clamp(a, b, c) = max(min(a, c), b)
            emit_fmin(code, dst, a, c); // dst = min(a, hi)
            emit_fmax(code, dst, dst, b); // dst = max(dst, lo)
        }

        _ => panic!("ternary emit not implemented for {:?}", op),
    }
}

// =============================================================================
// Select Short-Circuit Helpers
// =============================================================================

/// UMINV Sd, Vn.4S — horizontal unsigned minimum across all 4 lanes.
/// Result is in lane 0 of dst (scalar Sd).
/// If mask is all-ones (0xFFFFFFFF per lane), result = 0xFFFFFFFF.
pub fn emit_uminv(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    // UMINV Vd.4S: 0x6EB1A800 | Rd | (Rn << 5)
    // Encoding: 0 1 1 0 1 1 1 0 1 0 1 1 0 0 0 1  1 0 1 0 1 0 0 0  Rn:5 Rd:5
    emit32(code, 0x6EB1A800 | (dst.0 as u32) | ((src.0 as u32) << 5));
}

/// UMAXV Sd, Vn.4S — horizontal unsigned maximum across all 4 lanes.
/// If mask is all-zeros, result = 0x00000000.
pub fn emit_umaxv(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    // UMAXV Vd.4S: 0x6E30A800 | Rd | (Rn << 5)
    emit32(code, 0x6E30A800 | (dst.0 as u32) | ((src.0 as u32) << 5));
}

/// FMOV Wd, Sn — move lane 0 of SIMD register to GP register W16.
/// We always use W16 as the GP scratch for Select branching.
pub fn emit_fmov_to_gp(code: &mut Vec<u8>, src: Reg) {
    // FMOV W16, Sn: 0x1E260000 | (Rn << 5) | Rd
    // where Rd=16 (W16), Rn is the SIMD register number
    emit32(code, 0x1E260000 | ((src.0 as u32) << 5) | 16);
}

/// CBZ W16, #offset — branch if W16 == 0 (mask all-false).
/// `offset` is in bytes, must be aligned to 4, range ±1MB.
/// Returns the index in `code` where the offset is encoded (for patching).
pub fn emit_cbz_w16(code: &mut Vec<u8>) -> usize {
    let patch_pos = code.len();
    // CBZ W16, #0 (placeholder offset)
    // Encoding: 0 0110100 imm19 Rt
    // Rt = 16 (W16)
    emit32(code, 0x34000010); // imm19 = 0, will be patched
    patch_pos
}

/// CBNZ W16, #offset — branch if W16 != 0 (mask has some true lanes).
/// Returns the index in `code` where the offset is encoded (for patching).
pub fn emit_cbnz_w16(code: &mut Vec<u8>) -> usize {
    let patch_pos = code.len();
    // CBNZ W16, #0 (placeholder offset)
    emit32(code, 0x35000010); // imm19 = 0, will be patched
    patch_pos
}

/// B #offset — unconditional branch (for skipping past else-arm).
/// Returns the index in `code` where the offset is encoded (for patching).
pub fn emit_b(code: &mut Vec<u8>) -> usize {
    let patch_pos = code.len();
    // B #0 (placeholder)
    emit32(code, 0x14000000); // imm26 = 0, will be patched
    patch_pos
}

/// Patch a CBZ/CBNZ instruction at `patch_pos` to branch to `target_pos`.
/// Both positions are byte offsets into the code buffer.
pub fn patch_cbz_cbnz(code: &mut [u8], patch_pos: usize, target_pos: usize) {
    let offset = (target_pos as i64 - patch_pos as i64) / 4;
    assert!(
        (-(1 << 18)..(1 << 18)).contains(&offset),
        "CBZ/CBNZ branch offset {} out of range (±1MB)",
        offset
    );
    let imm19 = (offset as u32) & 0x7FFFF;
    let existing = u32::from_le_bytes([
        code[patch_pos],
        code[patch_pos + 1],
        code[patch_pos + 2],
        code[patch_pos + 3],
    ]);
    let patched = (existing & 0xFF00001F) | (imm19 << 5);
    code[patch_pos..patch_pos + 4].copy_from_slice(&patched.to_le_bytes());
}

/// Patch an unconditional B instruction at `patch_pos` to branch to `target_pos`.
pub fn patch_b(code: &mut [u8], patch_pos: usize, target_pos: usize) {
    let offset = (target_pos as i64 - patch_pos as i64) / 4;
    assert!(
        (-(1 << 25)..(1 << 25)).contains(&offset),
        "B branch offset {} out of range (±128MB)",
        offset
    );
    let imm26 = (offset as u32) & 0x3FFFFFF;
    let existing = u32::from_le_bytes([
        code[patch_pos],
        code[patch_pos + 1],
        code[patch_pos + 2],
        code[patch_pos + 3],
    ]);
    let patched = (existing & 0xFC000000) | imm26;
    code[patch_pos..patch_pos + 4].copy_from_slice(&patched.to_le_bytes());
}

// =============================================================================
// Prologue / Epilogue
// =============================================================================

/// Emit function prologue
pub fn emit_prologue(_code: &mut Vec<u8>) {
    // For a simple JIT kernel, we might not need much
    // Input pointer already in X0
    // Just ensure we're aligned
}

/// Emit function epilogue (return)
pub fn emit_epilogue(code: &mut Vec<u8>, result: Reg) {
    // Move result to V0 if not already there
    if result.0 != 0 {
        // MOV V0, Vresult (ORR Vd.16B, Vn.16B, Vn.16B)
        emit32(
            code,
            0x4EA01C00 | ((result.0 as u32) << 5) | ((result.0 as u32) << 16),
        );
    }
    // RET
    emit32(code, 0xD65F03C0);
}

// =============================================================================
// Scanline loop wrapper
// =============================================================================

/// Encode an ARM64 GP register move: MOV Xd, Xm (= ORR Xd, XZR, Xm).
fn encode_mov_x(dst: u8, src: u8) -> u32 {
    // ORR Xd, XZR, Xm: sf=1, opc=01, shift=00, N=0
    // 0xAA0003E0 | (Rm << 16) | Rd
    0xAA0003E0u32 | ((src as u32) << 16) | (dst as u32)
}

/// Encode a NEON register move: MOV Vd.16B, Vn.16B (= ORR Vd.16B, Vn.16B, Vn.16B).
fn encode_mov_v(dst: u8, src: u8) -> u32 {
    0x4EA01C00u32 | ((src as u32) << 5) | ((src as u32) << 16) | (dst as u32)
}

/// Encode ADD Xd, Xn, #imm12 (64-bit).
#[allow(dead_code)]
fn encode_add_x_imm(dst: u8, src: u8, imm12: u32) -> u32 {
    assert!(imm12 <= 4095, "ADD immediate {imm12} exceeds 12 bits");
    0x91000000u32 | (imm12 << 10) | ((src as u32) << 5) | (dst as u32)
}

/// Encode SUBS XZR, Xn, Xm (CMP Xn, Xm — sets flags, discards result).
#[allow(dead_code)]
fn encode_cmp_x(n: u8, m: u8) -> u32 {
    // SUBS Xd=XZR(31), Xn, Xm
    0xEB000000u32 | ((m as u32) << 16) | ((n as u32) << 5) | 31
}

/// Encode LDR Qt, [Xn], #16 (128-bit post-index load, 16-byte increment).
fn encode_ldr_q_post16(rt: u8, rn: u8) -> u32 {
    // LDR Qt, [Xn], #16 (post-index immediate)
    // size=00, V=1, opc=01, imm9=16=0x010, post-index=01
    // 00_111_1_00_01_0_000010000_01_Rn_Rt
    // Base for imm9=16, post-index: 0x3CC10400
    0x3CC10400u32 | ((rn as u32) << 5) | (rt as u32)
}

/// Encode STR Qt, [Xn], #16 (128-bit post-index store, 16-byte increment).
fn encode_str_q_post16(rt: u8, rn: u8) -> u32 {
    // STR Qt, [Xn], #16 (post-index immediate)
    // size=00, V=1, opc=00, imm9=16=0x010, post-index=01
    // 00_111_1_00_00_0_000010000_01_Rn_Rt
    // Base for imm9=16, post-index: 0x3C810400
    0x3C810400u32 | ((rn as u32) << 5) | (rt as u32)
}

/// Encode STP Xt1, Xt2, [Xn, #imm]! (pre-index, signed offset in multiples of 8).
fn encode_stp_pre(rt1: u8, rt2: u8, rn: u8, imm_bytes: i32) -> u32 {
    // STP Xt1, Xt2, [Xn, #imm]! (pre-index)
    // opc=10, V=0, L=0, pre-index
    // imm7 = imm_bytes / 8
    let imm7 = ((imm_bytes / 8) as u32) & 0x7F;
    0xA9800000u32
        | (imm7 << 15)
        | ((rt2 as u32) << 10)
        | ((rn as u32) << 5)
        | (rt1 as u32)
        | (0b11 << 23) // pre-index = opc 10, bit23=1 for pre-index writeback
}

/// Encode LDP Xt1, Xt2, [Xn], #imm (post-index, signed offset in multiples of 8).
fn encode_ldp_post(rt1: u8, rt2: u8, rn: u8, imm_bytes: i32) -> u32 {
    // LDP Xt1, Xt2, [Xn], #imm (post-index)
    let imm7 = ((imm_bytes / 8) as u32) & 0x7F;
    0xA8C00000u32 | (imm7 << 15) | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt1 as u32)
}

/// Encode LDP Xt1, Xt2, [Xn, #imm] (signed offset, no writeback).
fn encode_ldp_offset(rt1: u8, rt2: u8, rn: u8, imm_bytes: i32) -> u32 {
    let imm7 = ((imm_bytes / 8) as u32) & 0x7F;
    0xA9400000u32 | (imm7 << 15) | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt1 as u32)
}

/// Encode STP Xt1, Xt2, [Xn, #imm] (signed offset, no writeback).
fn encode_stp_offset(rt1: u8, rt2: u8, rn: u8, imm_bytes: i32) -> u32 {
    let imm7 = ((imm_bytes / 8) as u32) & 0x7F;
    0xA9000000u32 | (imm7 << 15) | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt1 as u32)
}

/// Encode MOVZ Xd, #imm16 (zero and load immediate).
#[allow(dead_code)]
fn encode_movz_x(dst: u8, imm16: u16) -> u32 {
    // MOVZ Xd, #imm16, LSL #0: sf=1
    0xD2800000u32 | ((imm16 as u32) << 5) | (dst as u32)
}

/// Encode B.cond with imm19 offset (in instructions, not bytes).
fn encode_bcond(imm19: u32, cond: u8) -> u32 {
    0x54000000u32 | ((imm19 & 0x7FFFF) << 5) | (cond as u32)
}

/// Encode CBZ Xt, #imm19 (64-bit GP register).
fn encode_cbz_x(rt: u8) -> u32 {
    // CBZ Xt, #0 (placeholder) — sf=1 for 64-bit
    0xB4000000u32 | (rt as u32)
}

/// Emit a scanline loop prologue for ARM64.
///
/// ## Calling convention
///
/// `ScanlineKernelFn(ptr, f32x4, f32x4, f32x4, ptr, usize)`:
///
/// ```text
///   x0 = input pointer     (GP arg 0)
///   v0 = Y broadcast       (SIMD arg 0)
///   v1 = Z broadcast       (SIMD arg 1)
///   v2 = W broadcast       (SIMD arg 2)
///   x1 = output pointer    (GP arg 1)
///   x2 = count             (GP arg 2)
/// ```
///
/// The kernel body expects X in v0, Y in v1, Z in v2, W in v3.
/// The prologue shuffles SIMD regs (v0/v1/v2 -> v1/v2/v3) and sets up
/// loop variables in callee-saved GP registers x19-x22.
///
/// Returns [`ScanlinePrologue`] with offsets for branch patching.
pub fn emit_scanline_prologue(code: &mut Vec<u8>) -> ScanlinePrologue {
    // Save callee-saved GP regs we use: x19, x20, x21, x22
    // STP x29, x30, [SP, #-48]!  (48 = 6*8, room for x29/x30/x19/x20/x21/x22)
    emit32(code, encode_stp_pre(29, 30, 31, -48));
    // STP x19, x20, [SP, #16]
    emit32(code, encode_stp_offset(19, 20, 31, 16));
    // STP x21, x22, [SP, #32]
    emit32(code, encode_stp_offset(21, 22, 31, 32));

    // Save Y/Z/W from ABI positions v0/v1/v2 into callee-saved NEON regs v29/v30/v31.
    // These survive the kernel body (which freely clobbers v0-v27).
    // We save them here so we can restore v1/v2/v3 at the top of each loop iteration.
    emit32(code, encode_mov_v(29, 0)); // V29 = Y (from ABI v0)
    emit32(code, encode_mov_v(30, 1)); // V30 = Z (from ABI v1)
    emit32(code, encode_mov_v(31, 2)); // V31 = W (from ABI v2)

    // Set up loop variables in callee-saved GP registers.
    // x19 = input pointer (post-incremented per iteration)
    // x20 = output pointer (post-incremented per iteration)
    // x21 = remaining count (decremented per iteration)
    emit32(code, encode_mov_x(19, 0)); // MOV x19, x0 (input ptr)
    emit32(code, encode_mov_x(20, 1)); // MOV x20, x1 (output ptr)
    emit32(code, encode_mov_x(21, 2)); // MOV x21, x2 (count)

    // Early exit if count == 0 (patched later to jump past the epilogue).
    let early_exit_patch = code.len();
    emit32(code, encode_cbz_x(21));

    // Loop header: reload Y/Z/W from callee-saved copies (kernel body may clobber v1-v3),
    // then load X[i] into v0, post-incrementing the input pointer.
    let loop_header = code.len();
    emit32(code, encode_mov_v(1, 29)); // V1 = Y (from saved V29)
    emit32(code, encode_mov_v(2, 30)); // V2 = Z (from saved V30)
    emit32(code, encode_mov_v(3, 31)); // V3 = W (from saved V31)
    // LDR Q0, [x19], #16 (load X[i] and advance pointer)
    emit32(code, encode_ldr_q_post16(0, 19));

    ScanlinePrologue {
        early_exit_patch,
        loop_header,
    }
}

/// Metadata returned by [`emit_scanline_prologue`] for patching branches.
pub struct ScanlinePrologue {
    /// Code offset of the CBZ (early exit if count==0) — patch target is the epilogue.
    pub early_exit_patch: usize,
    /// Code offset of the loop header (LDR X[i]) — back-edge target.
    pub loop_header: usize,
}

/// Emit the scanline loop tail: store result, decrement count, branch back to the
/// loop header, then restore callee-saved registers and return.
///
/// Uses post-index addressing: the output pointer (x20) is advanced by 16 bytes
/// per iteration (one `float32x4_t`). The count (x21) is decremented.
///
/// `result_reg` is the NEON register holding the kernel's output for this pixel.
pub fn emit_scanline_epilogue(code: &mut Vec<u8>, prologue: &ScanlinePrologue, result_reg: Reg) {
    // Store result: STR Q_result, [x20], #16 (store and advance output pointer)
    emit32(code, encode_str_q_post16(result_reg.0, 20));

    // Decrement remaining count: SUBS x21, x21, #1 (sets flags)
    // SUBS Xd, Xn, #imm12: 0xF1000000 | (imm12 << 10) | (Rn << 5) | Rd
    emit32(code, 0xF1000000u32 | (1 << 10) | (21 << 5) | 21);

    // B.NE loop_header (branch back if count > 0)
    let branch_pos = code.len();
    let offset_instr = (prologue.loop_header as i64 - branch_pos as i64) / 4;
    assert!(
        (-(1 << 18)..(1 << 18)).contains(&offset_instr),
        "scanline loop back-edge offset {offset_instr} out of ±1MB range"
    );
    let imm19 = (offset_instr as u32) & 0x7FFFF;
    emit32(code, encode_bcond(imm19, 0x01)); // cond=NE (0b0001)

    // --- Loop exit ---
    // Patch the early-exit CBZ to land here.
    let epilogue_pos = code.len();
    patch_cbz_x(code, prologue.early_exit_patch, epilogue_pos);

    // Restore callee-saved registers (reverse order of saves).
    // LDP x21, x22, [SP, #32]
    emit32(code, encode_ldp_offset(21, 22, 31, 32));
    // LDP x19, x20, [SP, #16]
    emit32(code, encode_ldp_offset(19, 20, 31, 16));
    // LDP x29, x30, [SP], #48
    emit32(code, encode_ldp_post(29, 30, 31, 48));

    // RET
    emit32(code, 0xD65F03C0);
}

// =============================================================================
// Hoisted scanline prologue / epilogue
// =============================================================================

/// AAPCS64 callee-saved NEON registers: v8-v15 (8 registers).
/// We use v8..v(7+num_hoisted) for loop-invariant values hoisted out of the
/// inner pixel loop.
const CALLEE_SAVED_NEON_BASE: u8 = 8;

/// Maximum number of NEON callee-saved registers available for hoisting.
const MAX_HOISTED_NEON: u8 = 8; // v8-v15

/// Encode STP Qt1, Qt2, [Xn, #imm] (128-bit NEON pair, signed offset, no writeback).
fn encode_stp_q_offset(rt1: u8, rt2: u8, rn: u8, imm_bytes: i32) -> u32 {
    let imm7 = ((imm_bytes / 16) as u32) & 0x7F;
    0xAD000000u32 | (imm7 << 15) | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt1 as u32)
}

/// Encode LDP Qt1, Qt2, [Xn, #imm] (128-bit NEON pair, signed offset, no writeback).
fn encode_ldp_q_offset(rt1: u8, rt2: u8, rn: u8, imm_bytes: i32) -> u32 {
    let imm7 = ((imm_bytes / 16) as u32) & 0x7F;
    0xAD400000u32 | (imm7 << 15) | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt1 as u32)
}

/// Encode STR Qt, [Xn, #imm] (single 128-bit NEON store, unsigned offset).
fn encode_str_q_offset(rt: u8, rn: u8, imm_bytes: u32) -> u32 {
    let imm12 = imm_bytes / 16;
    assert!(imm12 <= 4095, "STR Q offset exceeds 12-bit range");
    0x3D800000u32 | (imm12 << 10) | ((rn as u32) << 5) | (rt as u32)
}

/// Encode LDR Qt, [Xn, #imm] (single 128-bit NEON load, unsigned offset).
fn encode_ldr_q_offset(rt: u8, rn: u8, imm_bytes: u32) -> u32 {
    let imm12 = imm_bytes / 16;
    assert!(imm12 <= 4095, "LDR Q offset exceeds 12-bit range");
    0x3DC00000u32 | (imm12 << 10) | ((rn as u32) << 5) | (rt as u32)
}

/// Metadata returned by [`emit_scanline_prologue_hoisted`] for patching branches.
pub struct HoistedScanlinePrologue {
    /// Code offset of the CBZ (early exit if count==0) -- patch target is the epilogue.
    pub early_exit_patch: usize,
    /// Code offset of the loop header (reload Y/Z/W + LDR X[i]) -- back-edge target.
    pub loop_header: usize,
    /// Number of NEON callee-saved registers preserved (v8..v(7+num_hoisted)).
    pub num_hoisted: u8,
    /// Total SP adjustment for callee saves (GP + NEON), 16-byte aligned.
    pub callee_frame_size: u32,
}

/// Emit scanline loop prologue with additional callee-saved NEON registers for
/// hoisted (loop-invariant) values.
///
/// Saves v8..v(7+num_hoisted) in addition to the GP callee saves (x19-x22,
/// x29, x30). The setup block (emitted by the caller between this prologue
/// and the loop header) writes hoisted results into v8..v(7+num_hoisted).
///
/// # Panics
///
/// Panics if `num_hoisted > 8` (AAPCS64 only guarantees v8-v15 as callee-saved).
pub fn emit_scanline_prologue_hoisted(
    code: &mut Vec<u8>,
    num_hoisted: u8,
) -> HoistedScanlinePrologue {
    assert!(
        num_hoisted <= MAX_HOISTED_NEON,
        "num_hoisted={num_hoisted} exceeds AAPCS64 callee-saved NEON limit of {MAX_HOISTED_NEON} (v8-v15)"
    );

    // Compute total frame size for GP saves (48 bytes) + NEON saves.
    // GP frame: x29, x30, x19, x20, x21, x22 = 6 * 8 = 48 bytes.
    // NEON frame: num_hoisted * 16 bytes, rounded up to 16-byte alignment.
    let neon_save_bytes = (num_hoisted as u32) * 16;
    // Align NEON save area to 16 bytes (already guaranteed since 16*n is always aligned).
    let neon_frame = (neon_save_bytes + 15) & !15;
    let gp_frame: u32 = 48;
    let total_frame = gp_frame + neon_frame;

    // Save callee-saved GP regs: STP x29, x30, [SP, #-total_frame]!
    emit32(code, encode_stp_pre(29, 30, 31, -(total_frame as i32)));
    // STP x19, x20, [SP, #16]
    emit32(code, encode_stp_offset(19, 20, 31, 16));
    // STP x21, x22, [SP, #32]
    emit32(code, encode_stp_offset(21, 22, 31, 32));

    // Save NEON callee-saved registers in pairs at [SP, #gp_frame + offset].
    let mut neon_offset = gp_frame as i32;
    let mut i = 0u8;
    while i + 1 < num_hoisted {
        // Save pair: STP v(8+i), v(8+i+1), [SP, #neon_offset]
        emit32(
            code,
            encode_stp_q_offset(
                CALLEE_SAVED_NEON_BASE + i,
                CALLEE_SAVED_NEON_BASE + i + 1,
                31,
                neon_offset,
            ),
        );
        neon_offset += 32; // 2 * 16 bytes
        i += 2;
    }
    if i < num_hoisted {
        // Odd register: single STR v(8+i), [SP, #neon_offset]
        emit32(
            code,
            encode_str_q_offset(CALLEE_SAVED_NEON_BASE + i, 31, neon_offset as u32),
        );
    }

    // Move ABI inputs to callee-saved locations (same as non-hoisted prologue).
    // Y/Z/W from v0/v1/v2 into v29/v30/v31 (survive kernel body).
    emit32(code, encode_mov_v(29, 0)); // V29 = Y
    emit32(code, encode_mov_v(30, 1)); // V30 = Z
    emit32(code, encode_mov_v(31, 2)); // V31 = W

    // Set up loop variables in callee-saved GP registers.
    emit32(code, encode_mov_x(19, 0)); // x19 = input pointer
    emit32(code, encode_mov_x(20, 1)); // x20 = output pointer
    emit32(code, encode_mov_x(21, 2)); // x21 = count

    // --- Setup block emitted by caller here ---
    // The caller emits the hoisted (X-invariant) computation between this
    // point and the loop header. Results land in v8..v(7+num_hoisted) via
    // precolored register allocation.
    //
    // We defer the early-exit CBZ until after the setup block so that
    // the setup code runs unconditionally (callee saves must be balanced).
    // The CBZ is emitted as a placeholder; the caller patches it.
    let early_exit_patch = code.len();
    emit32(code, encode_cbz_x(21));

    // Loop header: reload Y/Z/W, load X[i].
    let loop_header = code.len();
    emit32(code, encode_mov_v(1, 29)); // V1 = Y
    emit32(code, encode_mov_v(2, 30)); // V2 = Z
    emit32(code, encode_mov_v(3, 31)); // V3 = W
    emit32(code, encode_ldr_q_post16(0, 19)); // V0 = X[i], x19 += 16

    HoistedScanlinePrologue {
        early_exit_patch,
        loop_header,
        num_hoisted,
        callee_frame_size: total_frame,
    }
}

/// Emit scanline loop epilogue with callee-saved NEON register restoration.
///
/// Mirrors [`emit_scanline_prologue_hoisted`]: store result, loop back, then
/// restore v8..v(7+num_hoisted) and GP callee saves before returning.
pub fn emit_scanline_epilogue_hoisted(
    code: &mut Vec<u8>,
    prologue: &HoistedScanlinePrologue,
    result_reg: Reg,
) {
    let num_hoisted = prologue.num_hoisted;
    let total_frame = prologue.callee_frame_size;

    // Store result: STR Q_result, [x20], #16
    emit32(code, encode_str_q_post16(result_reg.0, 20));

    // Decrement count: SUBS x21, x21, #1
    emit32(code, 0xF1000000u32 | (1 << 10) | (21 << 5) | 21);

    // B.NE loop_header
    let branch_pos = code.len();
    let offset_instr = (prologue.loop_header as i64 - branch_pos as i64) / 4;
    assert!(
        (-(1 << 18)..(1 << 18)).contains(&offset_instr),
        "scanline loop back-edge offset {offset_instr} out of +/-1MB range"
    );
    let imm19 = (offset_instr as u32) & 0x7FFFF;
    emit32(code, encode_bcond(imm19, 0x01)); // cond=NE

    // --- Loop exit ---
    let epilogue_pos = code.len();
    patch_cbz_x(code, prologue.early_exit_patch, epilogue_pos);

    // Restore NEON callee-saved registers (reverse order of saves).
    let gp_frame: u32 = 48;
    let mut neon_offset = gp_frame as i32;
    let mut i = 0u8;
    // We need to restore in the same order (pairs first, then odd), but from the frame.
    while i + 1 < num_hoisted {
        emit32(
            code,
            encode_ldp_q_offset(
                CALLEE_SAVED_NEON_BASE + i,
                CALLEE_SAVED_NEON_BASE + i + 1,
                31,
                neon_offset,
            ),
        );
        neon_offset += 32;
        i += 2;
    }
    if i < num_hoisted {
        emit32(
            code,
            encode_ldr_q_offset(CALLEE_SAVED_NEON_BASE + i, 31, neon_offset as u32),
        );
    }

    // Restore GP callee-saved registers.
    emit32(code, encode_ldp_offset(21, 22, 31, 32));
    emit32(code, encode_ldp_offset(19, 20, 31, 16));
    emit32(code, encode_ldp_post(29, 30, 31, total_frame as i32));

    // RET
    emit32(code, 0xD65F03C0);
}

/// Patch a 64-bit CBZ instruction at `patch_pos` to branch to `target_pos`.
fn patch_cbz_x(code: &mut [u8], patch_pos: usize, target_pos: usize) {
    let offset = (target_pos as i64 - patch_pos as i64) / 4;
    assert!(
        (-(1 << 18)..(1 << 18)).contains(&offset),
        "CBZ branch offset {offset} out of ±1MB range"
    );
    let imm19 = (offset as u32) & 0x7FFFF;
    let existing = u32::from_le_bytes([
        code[patch_pos],
        code[patch_pos + 1],
        code[patch_pos + 2],
        code[patch_pos + 3],
    ]);
    let patched = (existing & 0xFF00001F) | (imm19 << 5);
    code[patch_pos..patch_pos + 4].copy_from_slice(&patched.to_le_bytes());
}

// =============================================================================
// Aarch64Asm — typed assembler layer over raw encode_* functions
// =============================================================================

/// A thin assembler wrapper that owns a `Vec<u8>` code buffer and exposes
/// typed methods for each AArch64/NEON instruction.
///
/// The existing `emit_*` free functions are the correctness layer. Every
/// method on `Aarch64Asm` delegates to the corresponding free function,
/// adding nothing but ergonomics and buffer ownership.
///
/// ```text
/// Aarch64Asm::fadd(dst, a, b)  →  emit_fadd(&mut self.code, dst, a, b)
/// ```
pub struct Aarch64Asm {
    code: Vec<u8>,
}

impl Default for Aarch64Asm {
    fn default() -> Self {
        Self::new()
    }
}

impl Aarch64Asm {
    /// Create a new assembler with an empty code buffer.
    #[must_use]
    pub fn new() -> Self {
        Self { code: Vec::new() }
    }

    /// Create a new assembler with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            code: Vec::with_capacity(n),
        }
    }

    /// Current code offset in bytes (useful for branch patching).
    #[inline]
    #[must_use]
    pub fn offset(&self) -> usize {
        self.code.len()
    }

    /// Borrow the code buffer.
    #[inline]
    #[must_use]
    pub fn code(&self) -> &[u8] {
        &self.code
    }

    /// Mutable borrow of the code buffer (for branch patching).
    #[inline]
    pub fn code_mut(&mut self) -> &mut Vec<u8> {
        &mut self.code
    }

    /// Consume the assembler, returning the code buffer.
    #[inline]
    #[must_use]
    pub fn into_code(self) -> Vec<u8> {
        self.code
    }

    // =========================================================================
    // Arithmetic — NEON float32x4
    // =========================================================================

    /// FADD Vd.4S, Vn.4S, Vm.4S
    #[inline]
    pub fn fadd(&mut self, dst: Reg, a: Reg, b: Reg) {
        emit_fadd(&mut self.code, dst, a, b);
    }

    /// FSUB Vd.4S, Vn.4S, Vm.4S
    #[inline]
    pub fn fsub(&mut self, dst: Reg, a: Reg, b: Reg) {
        emit_fsub(&mut self.code, dst, a, b);
    }

    /// FMUL Vd.4S, Vn.4S, Vm.4S
    #[inline]
    pub fn fmul(&mut self, dst: Reg, a: Reg, b: Reg) {
        emit_fmul(&mut self.code, dst, a, b);
    }

    /// FDIV Vd.4S, Vn.4S, Vm.4S
    #[inline]
    pub fn fdiv(&mut self, dst: Reg, a: Reg, b: Reg) {
        emit_fdiv(&mut self.code, dst, a, b);
    }

    /// FMLA Vd.4S, Vn.4S, Vm.4S (fused multiply-add: Vd += Vn * Vm)
    #[inline]
    pub fn fmla(&mut self, dst: Reg, a: Reg, b: Reg) {
        emit_fmla(&mut self.code, dst, a, b);
    }

    /// FMIN Vd.4S, Vn.4S, Vm.4S
    #[inline]
    pub fn fmin(&mut self, dst: Reg, a: Reg, b: Reg) {
        emit_fmin(&mut self.code, dst, a, b);
    }

    /// FMAX Vd.4S, Vn.4S, Vm.4S
    #[inline]
    pub fn fmax(&mut self, dst: Reg, a: Reg, b: Reg) {
        emit_fmax(&mut self.code, dst, a, b);
    }

    // =========================================================================
    // Unary — NEON float32x4
    // =========================================================================

    /// FSQRT Vd.4S, Vn.4S
    #[inline]
    pub fn fsqrt(&mut self, dst: Reg, src: Reg) {
        emit_fsqrt(&mut self.code, dst, src);
    }

    /// FABS Vd.4S, Vn.4S
    #[inline]
    pub fn fabs(&mut self, dst: Reg, src: Reg) {
        emit_fabs(&mut self.code, dst, src);
    }

    /// FNEG Vd.4S, Vn.4S
    #[inline]
    pub fn fneg(&mut self, dst: Reg, src: Reg) {
        emit_fneg(&mut self.code, dst, src);
    }

    /// NOT Vd.16B, Vn.16B
    #[inline]
    pub fn not(&mut self, dst: Reg, src: Reg) {
        emit_not(&mut self.code, dst, src);
    }

    /// FRINTM Vd.4S, Vn.4S (floor)
    #[inline]
    pub fn frintm(&mut self, dst: Reg, src: Reg) {
        emit_frintm(&mut self.code, dst, src);
    }

    /// FRINTP Vd.4S, Vn.4S (ceil)
    #[inline]
    pub fn frintp(&mut self, dst: Reg, src: Reg) {
        emit_frintp(&mut self.code, dst, src);
    }

    /// FRINTA Vd.4S, Vn.4S (round to nearest, ties away from zero)
    #[inline]
    pub fn frinta(&mut self, dst: Reg, src: Reg) {
        emit_frinta(&mut self.code, dst, src);
    }

    // =========================================================================
    // Approximate operations (estimate + refinement)
    // =========================================================================

    /// FRSQRTE + FRSQRTS refinement (~4 instructions for reciprocal sqrt)
    #[inline]
    pub fn frsqrt(&mut self, dst: Reg, src: Reg, scratch: Reg) {
        emit_frsqrt(&mut self.code, dst, src, scratch);
    }

    /// FRECPE + FRECPS refinement (~3 instructions for reciprocal)
    #[inline]
    pub fn frecip(&mut self, dst: Reg, src: Reg, scratch: Reg) {
        emit_frecip(&mut self.code, dst, src, scratch);
    }

    // =========================================================================
    // Move / Duplicate
    // =========================================================================

    /// MOV Vd.16B, Vn.16B (vector register copy via ORR)
    #[inline]
    pub fn mov_vec(&mut self, dst: Reg, src: Reg) {
        emit_mov(&mut self.code, dst, src);
    }

    /// DUP Vd.4S, Vn.S[0] (duplicate scalar lane 0 to all lanes)
    #[inline]
    pub fn dup_s0(&mut self, dst: Reg, src: Reg) {
        emit_dup_s0(&mut self.code, dst, src);
    }

    // =========================================================================
    // Comparison
    // =========================================================================

    /// FCMGT Vd.4S, Vn.4S, Vm.4S (greater than)
    #[inline]
    pub fn fcmgt(&mut self, dst: Reg, a: Reg, b: Reg) {
        emit_fcmgt(&mut self.code, dst, a, b);
    }

    /// FCMGE Vd.4S, Vn.4S, Vm.4S (greater or equal)
    #[inline]
    pub fn fcmge(&mut self, dst: Reg, a: Reg, b: Reg) {
        emit_fcmge(&mut self.code, dst, a, b);
    }

    /// FCMEQ Vd.4S, Vn.4S, Vm.4S (equal)
    #[inline]
    pub fn fcmeq(&mut self, dst: Reg, a: Reg, b: Reg) {
        emit_fcmeq(&mut self.code, dst, a, b);
    }

    // =========================================================================
    // Bitwise / Selection
    // =========================================================================

    /// BSL Vd.16B, Vn.16B, Vm.16B (bitwise select)
    #[inline]
    pub fn bsl(&mut self, mask: Reg, if_true: Reg, if_false: Reg) {
        emit_bsl(&mut self.code, mask, if_true, if_false);
    }

    // =========================================================================
    // Memory — Load / Store
    // =========================================================================

    /// LDR Vd, [X0, #offset] (128-bit vector load from base X0)
    #[inline]
    pub fn ldr_voff(&mut self, dst: Reg, offset: u16) {
        emit_ldr_voff(&mut self.code, dst, offset);
    }

    /// STR Vt, [X0, #offset] (128-bit vector store to base X0)
    #[inline]
    pub fn str_voff(&mut self, src: Reg, offset: u16) {
        emit_str_voff(&mut self.code, src, offset);
    }

    /// LDR Vd, [SP, #offset] (128-bit vector load from stack)
    #[inline]
    pub fn ldr_sp(&mut self, dst: Reg, offset: u32) {
        emit_ldr_sp(&mut self.code, dst, offset);
    }

    /// STR Vt, [SP, #offset] (128-bit vector store to stack)
    #[inline]
    pub fn str_sp(&mut self, src: Reg, offset: u32) {
        emit_str_sp(&mut self.code, src, offset);
    }

    /// LDR Qt, [X17, #byte_offset] (128-bit load from constant pool)
    #[inline]
    pub fn ldr_q_x17(&mut self, dst: Reg, byte_offset: u16) {
        emit_ldr_q_x17(&mut self.code, dst, byte_offset);
    }

    /// SUB SP, SP, #imm (allocate stack frame)
    #[inline]
    pub fn sub_sp(&mut self, size: u32) {
        emit_sub_sp(&mut self.code, size);
    }

    /// ADD SP, SP, #imm (deallocate stack frame)
    #[inline]
    pub fn add_sp(&mut self, size: u32) {
        emit_add_sp(&mut self.code, size);
    }

    // =========================================================================
    // Constants
    // =========================================================================

    /// Load a floating-point constant into a vector register.
    ///
    /// Strategy: zero -> MOVI, FMOV-encodable -> FMOV, else MOVZ+MOVK+DUP.
    #[inline]
    pub fn load_const(&mut self, dst: Reg, value: f32) {
        emit_fmov_imm(
            &mut self.code,
            dst,
            value,
            [Reg(28), Reg(29), Reg(30), Reg(31)],
        );
    }

    /// Emit a constant pool entry (f32 splatted 4x = 16 bytes).
    #[inline]
    pub fn pool_entry(&mut self, val_bits: u32) {
        emit_pool_entry(&mut self.code, val_bits);
    }

    // =========================================================================
    // Branch / Control
    // =========================================================================

    /// Emit RET instruction.
    #[inline]
    pub fn ret(&mut self) {
        emit32(&mut self.code, 0xD65F03C0);
    }

    /// Emit function prologue (currently a no-op for simple kernels).
    #[inline]
    pub fn prologue(&mut self) {
        emit_prologue(&mut self.code);
    }

    /// Emit function epilogue: move result to V0 if needed, then RET.
    #[inline]
    pub fn epilogue(&mut self, result: Reg) {
        emit_epilogue(&mut self.code, result);
    }

    /// CBZ W16, #0 (placeholder). Returns patch position.
    #[inline]
    pub fn cbz_w16(&mut self) -> usize {
        emit_cbz_w16(&mut self.code)
    }

    /// CBNZ W16, #0 (placeholder). Returns patch position.
    #[inline]
    pub fn cbnz_w16(&mut self) -> usize {
        emit_cbnz_w16(&mut self.code)
    }

    /// B #0 (unconditional branch placeholder). Returns patch position.
    #[inline]
    pub fn b(&mut self) -> usize {
        emit_b(&mut self.code)
    }

    /// FMOV Wd, Sn -> W16 (move SIMD lane 0 to GP register W16).
    #[inline]
    pub fn fmov_to_gp(&mut self, src: Reg) {
        emit_fmov_to_gp(&mut self.code, src);
    }

    /// UMINV Sd, Vn.4S (horizontal unsigned min across all lanes).
    #[inline]
    pub fn uminv(&mut self, dst: Reg, src: Reg) {
        emit_uminv(&mut self.code, dst, src);
    }

    /// UMAXV Sd, Vn.4S (horizontal unsigned max across all lanes).
    #[inline]
    pub fn umaxv(&mut self, dst: Reg, src: Reg) {
        emit_umaxv(&mut self.code, dst, src);
    }

    // =========================================================================
    // Branch patching
    // =========================================================================

    /// Patch a CBZ/CBNZ at `patch_pos` to branch to `target_pos`.
    #[inline]
    pub fn patch_cbz_cbnz(&mut self, patch_pos: usize, target_pos: usize) {
        patch_cbz_cbnz(&mut self.code, patch_pos, target_pos);
    }

    /// Patch an unconditional B at `patch_pos` to branch to `target_pos`.
    #[inline]
    pub fn patch_b(&mut self, patch_pos: usize, target_pos: usize) {
        patch_b(&mut self.code, patch_pos, target_pos);
    }

    /// Patch an ADR or ADRP+ADD at `adr_pos` to point to `target_pos`.
    #[inline]
    pub fn patch_adr_or_adrp(&mut self, adr_pos: usize, target_pos: usize, mode: AdrMode) {
        patch_adr_or_adrp(&mut self.code, adr_pos, target_pos, mode);
    }

    /// Emit ADR X17, #0 placeholder. Returns patch position.
    #[inline]
    pub fn adr_x17_placeholder(&mut self) -> usize {
        emit_adr_x17_placeholder(&mut self.code)
    }

    // =========================================================================
    // Compound dispatch (delegates to emit_unary / emit_binary / emit_ternary)
    // =========================================================================

    /// Emit a binary operation by opcode.
    #[inline]
    pub fn binary(&mut self, op: OpKind, dst: Reg, src1: Reg, src2: Reg) {
        emit_binary(&mut self.code, op, dst, src1, src2);
    }

    /// Emit a ternary operation by opcode.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn ternary(&mut self, op: OpKind, dst: Reg, a: Reg, b: Reg, c: Reg) {
        emit_ternary(&mut self.code, op, dst, a, b, c);
    }

    // =========================================================================
    // Scanline loop
    // =========================================================================

    /// Emit the scanline loop prologue. Returns metadata for patching.
    #[inline]
    pub fn scanline_prologue(&mut self) -> ScanlinePrologue {
        emit_scanline_prologue(&mut self.code)
    }

    /// Emit the scanline loop epilogue (store, decrement, back-edge, restore, ret).
    #[inline]
    pub fn scanline_epilogue(&mut self, prologue: &ScanlinePrologue, result_reg: Reg) {
        emit_scanline_epilogue(&mut self.code, prologue, result_reg);
    }

    /// Emit hoisted scanline prologue with callee-saved NEON registers.
    #[inline]
    pub fn scanline_prologue_hoisted(&mut self, num_hoisted: u8) -> HoistedScanlinePrologue {
        emit_scanline_prologue_hoisted(&mut self.code, num_hoisted)
    }

    /// Emit hoisted scanline epilogue with NEON register restoration.
    #[inline]
    pub fn scanline_epilogue_hoisted(
        &mut self,
        prologue: &HoistedScanlinePrologue,
        result_reg: Reg,
    ) {
        emit_scanline_epilogue_hoisted(&mut self.code, prologue, result_reg);
    }

    // =========================================================================
    // Raw emission
    // =========================================================================

    /// Write a raw 32-bit instruction word.
    #[inline]
    pub fn emit_raw(&mut self, inst: u32) {
        emit32(&mut self.code, inst);
    }

    // =========================================================================
    // Disassembly
    // =========================================================================

    /// Disassemble the code buffer into a human-readable string.
    ///
    /// Decodes common NEON floating-point, memory, move, comparison, branch,
    /// and control instructions. Unknown encodings are shown as "unknown".
    #[must_use]
    pub fn disassemble(&self) -> String {
        disassemble_code(&self.code)
    }
}

// =============================================================================
// Disassembly support
// =============================================================================

/// Disassemble a raw code buffer into a human-readable string.
///
/// Public so that `dump_jit_asm` and other callers can disassemble code
/// produced by the free-function emitters without constructing an `Aarch64Asm`.
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
/// Returns an error string if compilation fails (same errors as `compile_arena`).
#[cfg(target_arch = "aarch64")]
pub fn dump_jit_asm(
    arena: &crate::arena::ExprArena,
    root: crate::arena::ExprId,
) -> Result<String, &'static str> {
    let result = super::compile_arena_dag(arena, root)?;
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
        let scratch = [Reg(16), Reg(17), Reg(18), Reg(19)];

        // 1.0 is FMOV-encodable → should emit exactly 1 instruction (4 bytes)
        emit_fmov_imm(&mut code, dst, 1.0, scratch);
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
        emit_fmov_imm(&mut code, Reg(0), 0.0, [Reg(16), Reg(17), Reg(18), Reg(19)]);
        assert_eq!(code.len(), 4, "zero should emit 1 instruction (MOVI)");
    }

    #[test]
    fn emit_fmov_imm_fallback_for_non_encodable() {
        let mut code = Vec::new();
        emit_fmov_imm(
            &mut code,
            Reg(0),
            core::f32::consts::PI,
            [Reg(16), Reg(17), Reg(18), Reg(19)],
        );
        assert_eq!(
            code.len(),
            12,
            "non-encodable should emit 3 instructions (MOVZ+MOVK+DUP)"
        );
    }

    // =====================================================================
    // Aarch64Asm tests
    // =====================================================================

    #[test]
    fn asm_new_is_empty() {
        let asm = Aarch64Asm::new();
        assert_eq!(asm.offset(), 0);
        assert!(asm.code().is_empty());
    }

    #[test]
    fn asm_with_capacity() {
        let asm = Aarch64Asm::with_capacity(256);
        assert_eq!(asm.offset(), 0);
    }

    #[test]
    fn asm_fadd_matches_free_function() {
        let mut asm = Aarch64Asm::new();
        asm.fadd(Reg(0), Reg(1), Reg(2));

        let mut raw = Vec::new();
        emit_fadd(&mut raw, Reg(0), Reg(1), Reg(2));

        assert_eq!(
            asm.code(),
            raw.as_slice(),
            "Aarch64Asm::fadd must match emit_fadd"
        );
    }

    #[test]
    fn asm_sequence_matches_free_functions() {
        let mut asm = Aarch64Asm::new();
        asm.fmul(Reg(4), Reg(0), Reg(0));
        asm.fmla(Reg(4), Reg(1), Reg(1));
        asm.fsqrt(Reg(4), Reg(4));
        asm.epilogue(Reg(4));

        let mut raw = Vec::new();
        emit_fmul(&mut raw, Reg(4), Reg(0), Reg(0));
        emit_fmla(&mut raw, Reg(4), Reg(1), Reg(1));
        emit_fsqrt(&mut raw, Reg(4), Reg(4));
        emit_epilogue(&mut raw, Reg(4));

        assert_eq!(
            asm.code(),
            raw.as_slice(),
            "Aarch64Asm sequence must be byte-identical to free function sequence"
        );
    }

    #[test]
    fn asm_into_code_returns_buffer() {
        let mut asm = Aarch64Asm::new();
        asm.ret();
        let code = asm.into_code();
        assert_eq!(code.len(), 4);
        let word = u32::from_le_bytes(code[..4].try_into().unwrap());
        assert_eq!(word, 0xD65F03C0, "RET encoding");
    }

    #[test]
    fn asm_load_const_zero() {
        let mut asm = Aarch64Asm::new();
        asm.load_const(Reg(5), 0.0);
        assert_eq!(
            asm.offset(),
            4,
            "zero constant should be 1 instruction (MOVI)"
        );
    }

    #[test]
    fn asm_load_const_one() {
        let mut asm = Aarch64Asm::new();
        asm.load_const(Reg(5), 1.0);
        assert_eq!(asm.offset(), 4, "1.0 should be 1 instruction (FMOV imm)");
    }

    #[test]
    fn asm_load_const_general() {
        let mut asm = Aarch64Asm::new();
        asm.load_const(Reg(5), core::f32::consts::PI);
        assert_eq!(
            asm.offset(),
            12,
            "PI should be 3 instructions (MOVZ+MOVK+DUP)"
        );
    }

    #[test]
    fn asm_all_arithmetic_ops() {
        let mut asm = Aarch64Asm::new();
        let d = Reg(4);
        let a = Reg(0);
        let b = Reg(1);
        asm.fadd(d, a, b);
        asm.fsub(d, a, b);
        asm.fmul(d, a, b);
        asm.fdiv(d, a, b);
        asm.fmin(d, a, b);
        asm.fmax(d, a, b);
        // 6 instructions = 24 bytes
        assert_eq!(asm.offset(), 24);
    }

    #[test]
    fn asm_all_unary_ops() {
        let mut asm = Aarch64Asm::new();
        let d = Reg(4);
        let s = Reg(0);
        asm.fsqrt(d, s);
        asm.fabs(d, s);
        asm.fneg(d, s);
        asm.not(d, s);
        asm.frintm(d, s);
        asm.frintp(d, s);
        asm.frinta(d, s);
        // 7 instructions = 28 bytes
        assert_eq!(asm.offset(), 28);
    }

    #[test]
    fn asm_comparison_and_select() {
        let mut asm = Aarch64Asm::new();
        let d = Reg(4);
        let a = Reg(0);
        let b = Reg(1);
        let c = Reg(2);
        asm.fcmgt(d, a, b);
        asm.fcmge(d, a, b);
        asm.fcmeq(d, a, b);
        asm.bsl(d, b, c);
        assert_eq!(asm.offset(), 16);
    }

    #[test]
    fn asm_branch_patching() {
        let mut asm = Aarch64Asm::new();
        asm.fadd(Reg(0), Reg(1), Reg(2)); // offset 0
        let patch = asm.cbz_w16(); // offset 4
        asm.fadd(Reg(0), Reg(1), Reg(2)); // offset 8
        let target = asm.offset(); // offset 12
        asm.patch_cbz_cbnz(patch, target);

        // Verify the CBZ was patched (offset from 4 to 12 = 8 bytes = 2 instructions)
        let word = u32::from_le_bytes(asm.code()[4..8].try_into().unwrap());
        let imm19 = (word >> 5) & 0x7FFFF;
        assert_eq!(imm19, 2, "CBZ should encode +2 instructions forward");
    }

    // =====================================================================
    // Disassembler tests
    // =====================================================================

    #[test]
    fn disassemble_ret() {
        let mut asm = Aarch64Asm::new();
        asm.ret();
        let dis = asm.disassemble();
        assert!(
            dis.contains("ret"),
            "disassembly should contain 'ret', got: {dis}"
        );
    }

    #[test]
    fn disassemble_fadd() {
        let mut asm = Aarch64Asm::new();
        asm.fadd(Reg(0), Reg(1), Reg(2));
        let dis = asm.disassemble();
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
        let mut asm = Aarch64Asm::new();
        asm.mov_vec(Reg(5), Reg(3));
        let dis = asm.disassemble();
        assert!(
            dis.contains("mov v5.16b, v3.16b"),
            "expected mov decode, got: {dis}"
        );
    }

    #[test]
    fn disassemble_sequence() {
        let mut asm = Aarch64Asm::new();
        asm.fmul(Reg(4), Reg(0), Reg(0));
        asm.fsqrt(Reg(4), Reg(4));
        asm.ret();
        let dis = asm.disassemble();
        assert!(dis.contains("fmul"), "missing fmul in: {dis}");
        assert!(dis.contains("fsqrt"), "missing fsqrt in: {dis}");
        assert!(dis.contains("ret"), "missing ret in: {dis}");
    }

    #[test]
    fn disassemble_zero_const() {
        let mut asm = Aarch64Asm::new();
        asm.load_const(Reg(0), 0.0);
        let dis = asm.disassemble();
        assert!(
            dis.contains("movi"),
            "zero should decode as movi, got: {dis}"
        );
    }

    #[test]
    fn disassemble_ldr_str() {
        let mut asm = Aarch64Asm::new();
        asm.ldr_voff(Reg(0), 32);
        asm.str_voff(Reg(1), 48);
        let dis = asm.disassemble();
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
        let mut asm = Aarch64Asm::new();
        asm.fadd(Reg(0), Reg(1), Reg(2));
        asm.fsub(Reg(0), Reg(1), Reg(2));
        asm.ret();
        let dis = asm.disassemble();
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
}
