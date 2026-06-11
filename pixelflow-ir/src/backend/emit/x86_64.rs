//! x86-64 SSE/AVX instruction encoding.
//!
//! Each function emits raw machine code bytes for one instruction (or a small fixed sequence).
//!
//! Two encoding strategies:
//! - **Legacy SSE** (2-operand, destructive): `dst op= src`
//! - **VEX** (3-operand, non-destructive): `dst = op(src1, src2)`
//!
//! Transcendental builtins (atan2, atan, asin, acos) use VEX encoding for the
//! 3-operand form which avoids extra MOV instructions in multi-step sequences.

use super::Reg;
use crate::kind::OpKind;

// =============================================================================
// Encoding Helpers
// =============================================================================

/// Emit a VEX-encoded 3-operand instruction (AVX style).
/// VEX.128.0F: dst = op(src1, src2)
fn emit_vex_128_0f(code: &mut Vec<u8>, opcode: u8, dst: Reg, src1: Reg, src2: Reg) {
    // 3-byte VEX prefix for xmm0-xmm15
    // VEX.128.0F: C4 RXB.01111 W.vvvv.L.pp
    let r = if dst.0 >= 8 { 0 } else { 0x80 };
    let x = 0x40; // X not used for register-register
    let b = if src2.0 >= 8 { 0 } else { 0x20 };
    let vvvv = (!src1.0 & 0xF) << 3;

    code.push(0xC4);
    code.push(r | x | b | 0x01); // map = 0F
    code.push(vvvv | 0x00); // W=0, L=0 (128-bit), pp=00
    code.push(opcode);
    code.push(0xC0 | ((dst.0 & 7) << 3) | (src2.0 & 7)); // ModRM
}

/// Emit a VEX-encoded 3-operand instruction with an immediate byte.
/// VEX.128.0F: dst = op(src1, src2, imm8)
fn emit_vex_128_0f_imm(code: &mut Vec<u8>, opcode: u8, dst: Reg, src1: Reg, src2: Reg, imm8: u8) {
    emit_vex_128_0f(code, opcode, dst, src1, src2);
    code.push(imm8);
}

/// Emit SSE instruction (legacy encoding, 2-operand: dst op= src)
fn emit_sse_rr(code: &mut Vec<u8>, prefix: Option<u8>, opcode: &[u8], dst: Reg, src: Reg) {
    if let Some(p) = prefix {
        code.push(p);
    }

    // REX prefix if needed (for xmm8-xmm15)
    let rex = 0x40 | (if dst.0 >= 8 { 0x04 } else { 0 }) | (if src.0 >= 8 { 0x01 } else { 0 });
    if rex != 0x40 {
        code.push(rex);
    }

    code.extend_from_slice(opcode);
    code.push(0xC0 | ((dst.0 & 7) << 3) | (src.0 & 7));
}

// =============================================================================
// Load / Store
// =============================================================================

/// MOVAPS xmm, [rdi + offset] - Load 128-bit aligned
pub fn emit_movaps_load(code: &mut Vec<u8>, dst: Reg, offset: u16) {
    // REX if needed
    if dst.0 >= 8 {
        code.push(0x44); // REX.R
    }
    code.push(0x0F);
    code.push(0x28);

    if offset == 0 {
        code.push(0x07 | ((dst.0 & 7) << 3)); // [rdi]
    } else if offset < 128 {
        code.push(0x47 | ((dst.0 & 7) << 3)); // [rdi + disp8]
        code.push(offset as u8);
    } else {
        code.push(0x87 | ((dst.0 & 7) << 3)); // [rdi + disp32]
        code.extend_from_slice(&(offset as u32).to_le_bytes());
    }
}

/// MOVAPS [rdi + offset], xmm - Store 128-bit aligned
pub fn emit_movaps_store(code: &mut Vec<u8>, src: Reg, offset: u16) {
    if src.0 >= 8 {
        code.push(0x44);
    }
    code.push(0x0F);
    code.push(0x29);

    if offset == 0 {
        code.push(0x07 | ((src.0 & 7) << 3));
    } else if offset < 128 {
        code.push(0x47 | ((src.0 & 7) << 3));
        code.push(offset as u8);
    } else {
        code.push(0x87 | ((src.0 & 7) << 3));
        code.extend_from_slice(&(offset as u32).to_le_bytes());
    }
}

/// MOVAPS xmm, xmm - Register-to-register copy
pub fn emit_movaps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x28], dst, src);
}

// =============================================================================
// Arithmetic (SSE legacy 2-operand)
// =============================================================================

/// ADDPS xmm, xmm
pub fn emit_addps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x58], dst, src);
}

/// SUBPS xmm, xmm
pub fn emit_subps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x5C], dst, src);
}

/// MULPS xmm, xmm
pub fn emit_mulps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x59], dst, src);
}

/// DIVPS xmm, xmm
pub fn emit_divps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x5E], dst, src);
}

/// SQRTPS xmm, xmm
pub fn emit_sqrtps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x51], dst, src);
}

/// RSQRTPS xmm, xmm (approximate)
pub fn emit_rsqrtps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x52], dst, src);
}

/// RCPPS xmm, xmm (approximate reciprocal)
pub fn emit_rcpps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x53], dst, src);
}

/// MINPS xmm, xmm
pub fn emit_minps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x5D], dst, src);
}

/// MAXPS xmm, xmm
pub fn emit_maxps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x5F], dst, src);
}

// =============================================================================
// Arithmetic (VEX 3-operand)
// =============================================================================

/// VADDPS dst, src1, src2
fn emit_vaddps(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit_vex_128_0f(code, 0x58, dst, src1, src2);
}

/// VSUBPS dst, src1, src2
fn emit_vsubps(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit_vex_128_0f(code, 0x5C, dst, src1, src2);
}

/// VMULPS dst, src1, src2
fn emit_vmulps(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit_vex_128_0f(code, 0x59, dst, src1, src2);
}

/// VSQRTPS dst, src (VEX unary — src1 field is 0b1111 i.e. unused)
fn emit_vsqrtps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    // For unary VEX, vvvv = 0b1111 (src1 = Reg(15) inverted → all ones)
    emit_vex_128_0f(code, 0x51, dst, Reg(0), src);
}

// =============================================================================
// Bitwise (VEX 3-operand)
// =============================================================================

/// VANDPS dst, src1, src2 — bitwise AND
fn emit_vandps(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit_vex_128_0f(code, 0x54, dst, src1, src2);
}

/// VANDNPS dst, src1, src2 — bitwise NOT(src1) AND src2
fn emit_vandnps(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit_vex_128_0f(code, 0x55, dst, src1, src2);
}

/// VORPS dst, src1, src2 — bitwise OR
fn emit_vorps(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit_vex_128_0f(code, 0x56, dst, src1, src2);
}

/// VXORPS dst, src1, src2 — bitwise XOR
fn emit_vxorps(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit_vex_128_0f(code, 0x57, dst, src1, src2);
}

// =============================================================================
// Bitwise (SSE legacy 2-operand)
// =============================================================================

/// XORPS xmm, xmm (also used for negation via sign bit flip)
pub fn emit_xorps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x57], dst, src);
}

/// ANDPS xmm, xmm
pub fn emit_andps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, None, &[0x0F, 0x54], dst, src);
}

// =============================================================================
// Comparisons (VEX)
// =============================================================================

/// VCMPPS predicates
const CMP_LT: u8 = 1; // Less than (ordered, non-signaling)
const CMP_NLE: u8 = 6; // Not less-or-equal, i.e. greater than (unordered)

/// VCMPPS dst, src1, src2, imm8 — packed float comparison
///
/// Result is all-ones mask where predicate is true, all-zeros where false.
/// Predicate 1 = LT, Predicate 6 = NLE (greater than).
fn emit_vcmpps(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg, predicate: u8) {
    emit_vex_128_0f_imm(code, 0xC2, dst, src1, src2, predicate);
}

// =============================================================================
// Constants
// =============================================================================

/// Load a splat f32 constant into an XMM register via RIP-relative load.
///
/// Strategy: emit a JMP over 16 bytes of inline constant data, then load
/// with MOVAPS [RIP + disp]. This avoids needing GP scratch registers.
///
/// Layout in code stream:
/// ```text
///   JMP +16          ; 2 bytes (EB 10)
///   <16 bytes data>  ; 4x f32 splatted
///   MOVAPS dst, [RIP + disp32]  ; RIP-relative load
/// ```
fn emit_f32_const(code: &mut Vec<u8>, dst: Reg, val: f32) {
    let bits = val.to_bits();

    // Fast path: zero constant
    if bits == 0 {
        emit_vxorps(code, dst, dst, dst);
        return;
    }

    // JMP rel8 over 16 bytes of constant data
    code.push(0xEB);
    code.push(0x10); // jump +16

    // Emit 16 bytes: 4 copies of the f32
    for _ in 0..4 {
        code.extend_from_slice(&bits.to_le_bytes());
    }

    // MOVUPS dst, [RIP + disp32]
    // The displacement is relative to the end of this instruction.
    // MOVUPS (unaligned load) is required here: the constant is embedded inline
    // in the code stream at an arbitrary byte offset, so its address is not
    // guaranteed 16-byte aligned. MOVAPS would #GP-fault on a misaligned load.
    // Opcode 0F 10, ModRM = 0x05 | (dst.0 << 3), then disp32.
    // Total instruction length = (optional REX) + 2(opcode) + 1(ModRM) + 4(disp32) = 7 or 8 bytes.
    // RIP points to end of instruction, so disp32 = -(16 + instruction_length).

    let needs_rex = dst.0 >= 8;
    let inst_len: i32 = if needs_rex { 8 } else { 7 };
    let disp: i32 = -(16 + inst_len);

    if needs_rex {
        code.push(0x44); // REX.R
    }
    code.push(0x0F);
    code.push(0x10);
    code.push(0x05 | ((dst.0 & 7) << 3)); // ModRM: mod=00, rm=101 (RIP-relative)
    code.extend_from_slice(&disp.to_le_bytes());
}

/// Load constant into register (placeholder for the high-level emit dispatch).
///
/// Uses RIP-relative constant embedding for non-zero values, VXORPS for zero.
pub fn emit_const(code: &mut Vec<u8>, dst: Reg, val: f32, _scratch: [Reg; 4]) {
    emit_f32_const(code, dst, val);
}

// =============================================================================
// VEX integer / convert / round primitives
// =============================================================================
//
// The transcendental builtins below are faithful ports of the aarch64 (NEON)
// implementations in `aarch64.rs` — same algorithms and coefficients — emitted
// with AVX (VEX.128) encodings. AVX gives us `vroundps` plus 128-bit integer
// ops (`vcvttps2dq`, `vpslld`, `vpaddd`, ...) needed for exp/log bit twiddling.

/// General 3-byte VEX encoder for 128-bit ops.
///
/// `pp`: 0=none, 1=0x66, 2=0xF3, 3=0xF2. `mmmmm`: 1=0F, 2=0F38, 3=0F3A.
/// `reg` is the ModRM.reg operand (a register, or a `/digit` opcode extension
/// passed as `Reg(digit)`); `vvvv` is the inverted extra source (pass `Reg(0)`
/// when unused — that encodes the required `1111`); `rm` is the ModRM.rm reg.
fn emit_vex(
    code: &mut Vec<u8>,
    pp: u8,
    mmmmm: u8,
    w: u8,
    reg: Reg,
    vvvv: Reg,
    rm: Reg,
    opcode: u8,
) {
    let rbit = if reg.0 >= 8 { 0x00 } else { 0x80 };
    let xbit = 0x40;
    let bbit = if rm.0 >= 8 { 0x00 } else { 0x20 };
    code.push(0xC4);
    code.push(rbit | xbit | bbit | mmmmm);
    code.push((w << 7) | ((!vvvv.0 & 0xF) << 3) | pp);
    code.push(opcode);
    code.push(0xC0 | ((reg.0 & 7) << 3) | (rm.0 & 7));
}

/// VROUNDPS dst, src, imm8 — round packed f32 (imm: 0=nearest, 1=floor, 2=ceil, 3=trunc).
fn emit_vroundps(code: &mut Vec<u8>, dst: Reg, src: Reg, imm: u8) {
    emit_vex(code, 1, 3, 0, dst, Reg(0), src, 0x08); // VEX.128.66.0F3A.WIG 08 /r ib
    code.push(imm);
}

/// VCVTTPS2DQ dst, src — convert packed f32 → i32 with truncation.
fn emit_vcvttps2dq(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_vex(code, 2, 1, 0, dst, Reg(0), src, 0x5B); // VEX.128.F3.0F.WIG 5B /r
}

/// VCVTDQ2PS dst, src — convert packed i32 → f32.
fn emit_vcvtdq2ps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_vex(code, 0, 1, 0, dst, Reg(0), src, 0x5B); // VEX.128.0F.WIG 5B /r
}

/// VPADDD dst, src1, src2 — packed i32 add.
fn emit_vpaddd(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit_vex(code, 1, 1, 0, dst, src1, src2, 0xFE); // VEX.128.66.0F.WIG FE /r
}

/// VPSUBD dst, src1, src2 — packed i32 subtract.
fn emit_vpsubd(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit_vex(code, 1, 1, 0, dst, src1, src2, 0xFA); // VEX.128.66.0F.WIG FA /r
}

/// VPSLLD dst, src, imm8 — packed i32 shift-left-logical by immediate.
fn emit_vpslld_imm(code: &mut Vec<u8>, dst: Reg, src: Reg, imm: u8) {
    // VEX.128.66.0F.WIG 72 /6 ib ; dst = vvvv, src = rm, /6 in ModRM.reg.
    emit_vex(code, 1, 1, 0, Reg(6), dst, src, 0x72);
    code.push(imm);
}

/// VPSRLD dst, src, imm8 — packed i32 shift-right-logical by immediate.
fn emit_vpsrld_imm(code: &mut Vec<u8>, dst: Reg, src: Reg, imm: u8) {
    // VEX.128.66.0F.WIG 72 /2 ib.
    emit_vex(code, 1, 1, 0, Reg(2), dst, src, 0x72);
    code.push(imm);
}

/// Bit-select (NEON BSL analogue): `dst = (mask & if_true) | (~mask & if_false)`.
///
/// `tmp` must differ from `dst`, `mask`, `if_true`, and `if_false`.
fn emit_blend(code: &mut Vec<u8>, dst: Reg, mask: Reg, if_true: Reg, if_false: Reg, tmp: Reg) {
    emit_vandps(code, tmp, mask, if_true); // tmp = mask & if_true
    emit_vandnps(code, dst, mask, if_false); // dst = ~mask & if_false
    emit_vorps(code, dst, tmp, dst); // dst = blended
}

// VCMPPS ordered predicates (subset).
const CMP_EQ: u8 = 0; // EQ_OQ
const CMP_LE: u8 = 2; // LE_OS
const CMP_NEQ: u8 = 4; // NEQ_UQ
const CMP_GE: u8 = 5; // NLT_US (>=)

// =============================================================================
// Transcendental Builtins — inline polynomial sequences
// =============================================================================
//
// Faithful ports of the aarch64 builtins (same algorithms / coefficients).
//
// Register contract:
//   dst  — output register
//   src  — input register (read-only; never clobbered)
//   scratch[0..4] — clobbered scratch (4 distinct registers)

/// log2(x) core — exponent extraction + mantissa reduction + 5-term Horner.
fn emit_log2_body(code: &mut Vec<u8>, dst: Reg, src: Reg, s0: Reg, s1: Reg, s2: Reg) {
    // Phase 1: n = float(exponent_bits - 127)
    emit_vpsrld_imm(code, s0, src, 23); // s0 = bits >> 23
    emit_f32_const(code, s2, f32::from_bits(127)); // s2 = 127 (int splat)
    emit_vpsubd(code, s0, s0, s2); // s0 = exp - 127
    emit_vcvtdq2ps(code, dst, s0); // dst = n

    // Phase 2: f = mantissa in [1, 2)
    emit_f32_const(code, s2, f32::from_bits(0x007F_FFFF));
    emit_vandps(code, s1, src, s2); // s1 = mantissa bits
    emit_f32_const(code, s2, f32::from_bits(0x3F80_0000));
    emit_vorps(code, s1, s1, s2); // s1 = f

    // Phase 3: branchless reduction to [√2/2, √2]
    // mask = (f >= √2); adjust = 1.0 & mask; n += adjust; f *= (1 - 0.5*adjust)
    emit_f32_const(code, s2, 1.414_213_56_f32);
    emit_vcmpps(code, s2, s1, s2, CMP_GE); // s2 = mask(f >= √2)
    emit_f32_const(code, s0, 1.0);
    emit_vandps(code, s0, s0, s2); // s0 = adjust (1.0 or 0.0)
    emit_vaddps(code, dst, dst, s0); // n += adjust
    emit_f32_const(code, s2, 0.5);
    emit_vmulps(code, s2, s0, s2); // s2 = 0.5*adjust
    emit_f32_const(code, s0, 1.0);
    emit_vsubps(code, s2, s0, s2); // s2 = 1 - 0.5*adjust = factor
    emit_vmulps(code, s1, s1, s2); // f *= factor   (s1 = reduced f)

    // Phase 4: poly = ((((c4*f + c3)*f + c2)*f + c1)*f + c0); result = n + poly
    emit_f32_const(code, s0, -0.320_043_52_f32);
    emit_f32_const(code, s2, 1.797_496_9_f32);
    emit_vmulps(code, s0, s0, s1);
    emit_vaddps(code, s0, s0, s2);
    emit_f32_const(code, s2, -4.198_804_6_f32);
    emit_vmulps(code, s0, s0, s1);
    emit_vaddps(code, s0, s0, s2);
    emit_f32_const(code, s2, 5.727_023_f32);
    emit_vmulps(code, s0, s0, s1);
    emit_vaddps(code, s0, s0, s2);
    emit_f32_const(code, s2, -3.005_614_7_f32);
    emit_vmulps(code, s0, s0, s1);
    emit_vaddps(code, s0, s0, s2);
    emit_vaddps(code, dst, dst, s0); // n + poly
}

/// exp2(x) core — floor/frac split + 5-term Horner + 2^n bit scaling.
fn emit_exp2_body(code: &mut Vec<u8>, dst: Reg, src: Reg, s0: Reg, s1: Reg, s2: Reg) {
    emit_vroundps(code, s0, src, 1); // s0 = n = floor(x)
    emit_vsubps(code, s1, src, s0); // s1 = f = x - n

    // poly (accumulator s2): ((((c4*f + c3)*f + c2)*f + c1)*f + c0)
    emit_f32_const(code, s2, 0.013_555_7_f32);
    emit_f32_const(code, dst, 0.052_032_3_f32);
    emit_vmulps(code, s2, s2, s1);
    emit_vaddps(code, s2, s2, dst);
    emit_f32_const(code, dst, 0.241_379_3_f32);
    emit_vmulps(code, s2, s2, s1);
    emit_vaddps(code, s2, s2, dst);
    emit_f32_const(code, dst, 0.693_147_2_f32);
    emit_vmulps(code, s2, s2, s1);
    emit_vaddps(code, s2, s2, dst);
    emit_f32_const(code, dst, 1.0_f32);
    emit_vmulps(code, s2, s2, s1);
    emit_vaddps(code, s2, s2, dst); // s2 = poly(2^f)

    // 2^n = reinterpret((int(n) + 127) << 23)
    emit_vcvttps2dq(code, s1, s0); // s1 = int(n)
    emit_f32_const(code, dst, f32::from_bits(127)); // 127 (int splat)
    emit_vpaddd(code, s1, s1, dst);
    emit_vpslld_imm(code, s1, s1, 23); // s1 = 2^n bits
    emit_vmulps(code, dst, s2, s1); // dst = poly * 2^n
}

/// MOVUPS [rsp+disp8], xmm — red-zone spill store (unaligned, leaf-safe).
pub fn emit_movups_store_rsp(code: &mut Vec<u8>, src: Reg, disp: i8) {
    if src.0 >= 8 {
        code.push(0x44); // REX.R
    }
    code.push(0x0F);
    code.push(0x11);
    code.push(0x44 | ((src.0 & 7) << 3)); // mod=01, reg=src, rm=100 (SIB)
    code.push(0x24); // SIB: base=rsp, no index
    code.push(disp as u8);
}

/// MOVUPS xmm, [rsp+disp8] — red-zone reload (unaligned, leaf-safe).
pub fn emit_movups_load_rsp(code: &mut Vec<u8>, dst: Reg, disp: i8) {
    if dst.0 >= 8 {
        code.push(0x44);
    }
    code.push(0x0F);
    code.push(0x10);
    code.push(0x44 | ((dst.0 & 7) << 3));
    code.push(0x24);
    code.push(disp as u8);
}

// ---------------------------------------------------------------------------
// Public unary builtin entry points
// ---------------------------------------------------------------------------

pub fn emit_floor_builtin(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_vroundps(code, dst, src, 1);
}

pub fn emit_ceil_builtin(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_vroundps(code, dst, src, 2);
}

pub fn emit_round_builtin(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_vroundps(code, dst, src, 0); // round to nearest (even)
}

pub fn emit_fract_builtin(code: &mut Vec<u8>, dst: Reg, src: Reg, sc: [Reg; 4]) {
    emit_vroundps(code, sc[0], src, 1); // floor
    emit_vsubps(code, dst, src, sc[0]); // x - floor(x)
}

/// pow(x, y) = exp2(y * log2(x)).
pub fn emit_pow_builtin(code: &mut Vec<u8>, dst: Reg, base: Reg, exp: Reg, sc: [Reg; 4]) {
    emit_log2_body(code, sc[3], base, sc[0], sc[1], sc[2]); // sc[3] = log2(x)
    emit_vmulps(code, sc[3], sc[3], exp); // sc[3] = y * log2(x)
    emit_exp2_body(code, dst, sc[3], sc[0], sc[1], sc[2]); // dst = 2^(...)
}

/// hypot(x, y) = sqrt(x² + y²).
pub fn emit_hypot_builtin(code: &mut Vec<u8>, dst: Reg, x: Reg, y: Reg, sc: [Reg; 4]) {
    emit_vmulps(code, sc[0], x, x);
    emit_vmulps(code, dst, y, y);
    emit_vaddps(code, dst, dst, sc[0]);
    emit_vsqrtps(code, dst, dst);
}

/// select(cond, if_true, if_false) — bit blend (`cond` is an all-ones/zeros mask).
///
/// `tmp` must differ from `cond`, `if_true`, and `if_false`.
pub fn emit_select(code: &mut Vec<u8>, dst: Reg, cond: Reg, if_true: Reg, if_false: Reg, tmp: Reg) {
    emit_blend(code, dst, cond, if_true, if_false, tmp);
}

// =============================================================================
// High-level dispatch
// =============================================================================

/// Emit unary operation
pub fn emit_unary(code: &mut Vec<u8>, op: OpKind, dst: Reg, src: Reg, scratch: [Reg; 4]) {
    match op {
        OpKind::Sqrt => emit_sqrtps(code, dst, src),
        OpKind::Rsqrt => {
            emit_rsqrtps(code, dst, src);
            // TODO: Newton-Raphson refinement
        }
        OpKind::Recip => emit_rcpps(code, dst, src),

        // Negation: flip the sign bit (dst = src XOR 0x80000000).
        OpKind::Neg => {
            let mask = scratch[0];
            emit_f32_const(code, mask, f32::from_bits(0x8000_0000));
            emit_vxorps(code, dst, src, mask);
        }

        // Absolute value: clear the sign bit (dst = src AND 0x7FFFFFFF).
        OpKind::Abs => {
            let mask = scratch[0];
            emit_f32_const(code, mask, f32::from_bits(0x7FFF_FFFF));
            emit_vandps(code, dst, src, mask);
        }

        // Rounding (AVX vroundps)
        OpKind::Floor => emit_floor_builtin(code, dst, src),
        OpKind::Ceil => emit_ceil_builtin(code, dst, src),
        OpKind::Round => emit_round_builtin(code, dst, src),
        OpKind::Fract => emit_fract_builtin(code, dst, src, scratch),

        // Bit-manip primitives (integer-domain). Single instructions.
        OpKind::TruncToInt => emit_vcvttps2dq(code, dst, src),
        OpKind::IntToFloat => emit_vcvtdq2ps(code, dst, src),

        // Transcendentals (sin/cos/tan/exp/exp2/ln/log2/log10/atan/asin/acos) are
        // expanded to primitive arithmetic by `lowering` before codegen, so they
        // never reach a backend. Reaching here means the lowering pass was
        // skipped — a bug; fall through to the panic.
        _ => panic!(
            "x86_64 unary emit not implemented for {:?} (lowering not run?)",
            op
        ),
    }
}

/// Emit a logical shift of i32 lanes by a compile-time immediate.
/// `Shl` -> `vpslld`, `Shr` -> `vpsrld` (logical). VEX form is 3-operand
/// (`dst = src << imm`), so there is no two-operand hazard.
pub fn emit_shift_imm(code: &mut Vec<u8>, op: OpKind, dst: Reg, src: Reg, amount: u8) {
    match op {
        OpKind::Shl => emit_vpslld_imm(code, dst, src, amount),
        OpKind::Shr => emit_vpsrld_imm(code, dst, src, amount),
        _ => panic!("x86_64 emit_shift_imm: not a shift op: {:?}", op),
    }
}

/// Emit binary operation
pub fn emit_binary(code: &mut Vec<u8>, op: OpKind, dst: Reg, src1: Reg, src2: Reg) {
    // SSE is 2-operand, so we may need to move first
    if dst.0 != src1.0 {
        emit_sse_rr(code, None, &[0x0F, 0x28], dst, src1); // MOVAPS dst, src1
    }

    match op {
        OpKind::Add => emit_addps(code, dst, src2),
        OpKind::Sub => emit_subps(code, dst, src2),
        OpKind::Mul => emit_mulps(code, dst, src2),
        OpKind::Div => emit_divps(code, dst, src2),
        OpKind::Min => emit_minps(code, dst, src2),
        OpKind::Max => emit_maxps(code, dst, src2),

        // Comparisons → all-ones / all-zeros mask (ordered predicates).
        // `dst` already holds src1 (moved above), so compare in place.
        OpKind::Eq => emit_cmp_tail(code, dst, src2, CMP_EQ),
        OpKind::Ne => emit_cmp_tail(code, dst, src2, CMP_NEQ),
        OpKind::Lt => emit_cmp_tail(code, dst, src2, CMP_LT),
        OpKind::Le => emit_cmp_tail(code, dst, src2, CMP_LE),
        OpKind::Gt => emit_cmp_tail(code, dst, src2, CMP_NLE),
        OpKind::Ge => emit_cmp_tail(code, dst, src2, CMP_GE),

        // Bit-manip primitives. `dst` already holds src1 (moved above); the VEX
        // 3-operand encoders take it as the vvvv source so this is in-place.
        OpKind::IAdd => emit_vpaddd(code, dst, dst, src2),
        OpKind::BitAnd => emit_vandps(code, dst, dst, src2),
        OpKind::BitOr => emit_vorps(code, dst, dst, src2),

        _ => panic!("x86_64 binary emit not implemented for {:?}", op),
    }
}

/// Emit the trailing `CMPPS dst, src2, imm8` of an in-place compare (dst already
/// holds src1). Produces an all-ones / all-zeros mask.
fn emit_cmp_tail(code: &mut Vec<u8>, dst: Reg, src2: Reg, pred: u8) {
    let rex = 0x40 | (if dst.0 >= 8 { 0x04 } else { 0 }) | (if src2.0 >= 8 { 0x01 } else { 0 });
    if rex != 0x40 {
        code.push(rex);
    }
    code.push(0x0F);
    code.push(0xC2);
    code.push(0xC0 | ((dst.0 & 7) << 3) | (src2.0 & 7));
    code.push(pred);
}

/// Emit binary transcendental operation (needs scratch registers).
pub fn emit_binary_transcendental(
    code: &mut Vec<u8>,
    op: OpKind,
    dst: Reg,
    src1: Reg,
    src2: Reg,
    scratch: [Reg; 4],
) {
    match op {
        // Atan2 is lowered to arithmetic before codegen; only Pow/Hypot still
        // reach a backend (not yet lowered).
        OpKind::Pow => emit_pow_builtin(code, dst, src1, src2, scratch),
        OpKind::Hypot => emit_hypot_builtin(code, dst, src1, src2, scratch),
        _ => panic!(
            "x86_64 binary transcendental emit not implemented for {:?}",
            op
        ),
    }
}

/// Emit ternary operation
pub fn emit_ternary(code: &mut Vec<u8>, op: OpKind, dst: Reg, a: Reg, b: Reg, c: Reg) {
    match op {
        OpKind::MulAdd => {
            // Without FMA: dst = a * b; dst = dst + c
            if dst.0 != a.0 {
                emit_sse_rr(code, None, &[0x0F, 0x28], dst, a);
            }
            emit_mulps(code, dst, b);
            emit_addps(code, dst, c);
        }

        OpKind::Clamp => {
            // max(min(a, c), b)
            if dst.0 != a.0 {
                emit_sse_rr(code, None, &[0x0F, 0x28], dst, a);
            }
            emit_minps(code, dst, c);
            emit_maxps(code, dst, b);
        }

        _ => panic!("x86_64 ternary emit not implemented for {:?}", op),
    }
}

// =============================================================================
// Prologue / Epilogue
// =============================================================================

/// Emit function prologue
pub fn emit_prologue(_code: &mut Vec<u8>) {
    // Input pointer in rdi (System V) or rcx (Windows)
    // For now, assume System V
}

/// Emit function epilogue
pub fn emit_epilogue(code: &mut Vec<u8>, result: Reg) {
    // Move result to xmm0 if not already there
    if result.0 != 0 {
        emit_sse_rr(code, None, &[0x0F, 0x28], Reg(0), result);
    }
    // RET
    code.push(0xC3);
}

// =============================================================================
// Branches — for the shared driver's Select short-circuit guards.
// =============================================================================

/// MOVMSKPS eax, xmm — gather the 4 lane sign bits into eax (0b0000..0b1111).
/// For a select mask (lanes all-ones or all-zeros), eax == 0 means all-false
/// and eax == 0xF means all-true.
pub fn emit_movmskps_eax(code: &mut Vec<u8>, src: Reg) {
    if src.0 >= 8 {
        code.push(0x41); // REX.B
    }
    code.push(0x0F);
    code.push(0x50);
    code.push(0xC0 | (src.0 & 7)); // mod=11, reg=eax(0), rm=src
}

/// Emit `jcc rel32` with a zero placeholder; returns the offset of the rel32
/// field (pass to [`patch_rel32`]). `cc` is the 0x8_ condition byte (0x84 = je/jz,
/// 0x85 = jne/jnz).
pub fn emit_jcc_rel32(code: &mut Vec<u8>, cc: u8) -> usize {
    code.push(0x0F);
    code.push(cc);
    let pos = code.len();
    code.extend_from_slice(&[0, 0, 0, 0]);
    pos
}

/// Emit `jmp rel32` with a zero placeholder; returns the rel32 field offset.
pub fn emit_jmp_rel32(code: &mut Vec<u8>) -> usize {
    code.push(0xE9);
    let pos = code.len();
    code.extend_from_slice(&[0, 0, 0, 0]);
    pos
}

/// Patch a rel32 branch displacement (emitted by [`emit_jcc_rel32`] /
/// [`emit_jmp_rel32`]) so it lands at `target`.
pub fn patch_rel32(code: &mut [u8], pos: usize, target: usize) {
    let rel = (target as i64) - (pos as i64 + 4);
    code[pos..pos + 4].copy_from_slice(&(rel as i32).to_le_bytes());
}

/// TEST eax, eax (sets ZF iff eax == 0).
pub fn emit_test_eax(code: &mut Vec<u8>) {
    code.extend_from_slice(&[0x85, 0xC0]);
}

/// CMP eax, imm8 (sign-extended).
pub fn emit_cmp_eax_imm8(code: &mut Vec<u8>, imm: u8) {
    code.extend_from_slice(&[0x83, 0xF8, imm]);
}
