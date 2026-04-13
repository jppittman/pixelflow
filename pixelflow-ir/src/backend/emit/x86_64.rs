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

/// VDIVPS dst, src1, src2
fn emit_vdivps(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    emit_vex_128_0f(code, 0x5E, dst, src1, src2);
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

    // MOVAPS dst, [RIP + disp32]
    // The displacement is relative to the end of this instruction.
    // This instruction is 4 bytes (REX? + 0F 28 ModRM + disp32) or 3+4=7.
    // Actually: opcode 0F 28, ModRM = 0x05 | (dst.0 << 3), then disp32.
    // Total instruction length = (optional REX) + 2(opcode) + 1(ModRM) + 4(disp32) = 7 or 8 bytes.
    // RIP points to end of instruction, so disp32 = -(16 + instruction_length).

    let needs_rex = dst.0 >= 8;
    let inst_len: i32 = if needs_rex { 8 } else { 7 };
    let disp: i32 = -(16 + inst_len);

    if needs_rex {
        code.push(0x44); // REX.R
    }
    code.push(0x0F);
    code.push(0x28);
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
// Transcendental Builtins — inline polynomial sequences
// =============================================================================
//
// Each builtin translates the pixelflow-core SSE2 implementation into direct
// machine code emission. Same coefficients, same algorithms as x86.rs.
//
// Register contract:
//   dst  — output register
//   src  — input register (read-only, never clobbered)
//   s0-s2 — scratch registers from scratch[0..2] (clobbered)
//   s3    — scratch[3], used by composition builtins
//
// All operations use VEX encoding (3-operand, non-destructive) to minimize
// register pressure in long polynomial sequences.

/// atan2(y, x) — Chebyshev polynomial approximation.
///
/// Translated from pixelflow-ir backend/x86.rs F32x4::atan2().
/// Algorithm:
///   1. r = y / x, r_abs = |r|
///   2. Horner polynomial: poly = ((c7*t² + c5)*t² + c3)*t² + c1
///   3. atan_approx = poly * r_abs
///   4. Large-angle correction: if |r| > 1, atan = π/2 - atan(1/r)
///   5. Sign correction: multiply by sign(y)
///   6. Quadrant correction: subtract π*sign(y) if x < 0
pub fn emit_atan2_builtin(code: &mut Vec<u8>, dst: Reg, src_y: Reg, src_x: Reg, scratch: [Reg; 4]) {
    let s0 = scratch[0];
    let s1 = scratch[1];
    let s2 = scratch[2];
    let s3 = scratch[3];

    // ---- Phase 1: Compute r = y / x, r_abs = |r| ----

    // s0 = y / x
    emit_vdivps(code, s0, src_y, src_x);

    // s1 = abs_mask = 0x7FFFFFFF (splat)
    emit_f32_const(code, s1, f32::from_bits(0x7FFFFFFF));

    // s2 = r_abs = |r| = r & abs_mask
    emit_vandps(code, s2, s0, s1);

    // ---- Phase 2: Horner polynomial atan(t) ≈ t * ((c7*t²+c5)*t²+c3)*t²+c1) ----
    // where t = r_abs

    // dst = t² = r_abs * r_abs
    emit_vmulps(code, dst, s2, s2);

    // Load Chebyshev coefficients and evaluate Horner chain
    // poly = c7
    emit_f32_const(code, s3, -0.142857143_f32);

    // poly = c7 * t² + c5
    emit_f32_const(code, s1, 0.2_f32);
    emit_vmulps(code, s3, s3, dst); // s3 = c7 * t²
    emit_vaddps(code, s3, s3, s1); // s3 = c7*t² + c5

    // poly = (c7*t²+c5) * t² + c3
    emit_f32_const(code, s1, -0.333333333_f32);
    emit_vmulps(code, s3, s3, dst); // s3 = prev * t²
    emit_vaddps(code, s3, s3, s1); // s3 = prev*t² + c3

    // poly = (...) * t² + c1
    emit_f32_const(code, s1, 0.999999999_f32);
    emit_vmulps(code, s3, s3, dst); // s3 = prev * t²
    emit_vaddps(code, s3, s3, s1); // s3 = poly = prev*t² + c1

    // atan_approx = poly * r_abs
    emit_vmulps(code, s3, s3, s2); // s3 = atan_approx

    // ---- Phase 3: Handle |r| > 1 case ----
    // atan_large = π/2 - (1/r_abs) * atan_approx

    // s1 = 1.0
    emit_f32_const(code, s1, 1.0_f32);

    // mask_large = r_abs > 1.0  (all-ones where |r| > 1)
    emit_vcmpps(code, dst, s2, s1, CMP_NLE); // dst = mask_large

    // s1 = 1.0 / r_abs
    emit_vdivps(code, s1, s1, s2); // s1 = recip_r = 1/r_abs

    // s1 = recip_r * atan_approx
    emit_vmulps(code, s1, s1, s3); // s1 = recip_r * atan_approx

    // s2 = π/2
    emit_f32_const(code, s2, core::f32::consts::FRAC_PI_2);

    // s1 = π/2 - recip_r * atan_approx = atan_large
    emit_vsubps(code, s1, s2, s1); // s1 = atan_large

    // ---- Phase 4: Blend large/small case ----
    // atan_val = mask_large ? atan_large : atan_approx
    // Using AND/ANDN/OR pattern:
    //   s2 = dst & s1           (mask & atan_large)
    //   dst = ~dst & s3         (ANDN: ~mask & atan_approx)
    //   s2 = s2 | dst           (combine)
    emit_vandps(code, s2, dst, s1); // s2 = mask & atan_large
    emit_vandnps(code, dst, dst, s3); // dst = ~mask & atan_approx
    emit_vorps(code, s2, s2, dst); // s2 = atan_val (blended result)

    // ---- Phase 5: Sign correction (multiply by sign of y) ----
    // sign_y = y / |y|  (preserves sign, NaN-safe for zero handled by quadrant)

    // s1 = abs_mask
    emit_f32_const(code, s1, f32::from_bits(0x7FFFFFFF));
    // s3 = |y|
    emit_vandps(code, s3, src_y, s1);
    // s3 = sign_y = |y| / y ... actually we want y / |y|
    emit_vdivps(code, s3, src_y, s3); // s3 = sign_y = y / |y|

    // s2 = atan_signed = atan_val * sign_y
    emit_vmulps(code, s2, s2, s3); // s2 = atan_signed

    // ---- Phase 6: Quadrant correction for negative x ----
    // if x < 0: result = atan_signed - π * sign_y

    // dst = 0.0
    emit_vxorps(code, dst, dst, dst);

    // s1 = mask_neg_x = (x < 0)
    emit_vcmpps(code, s1, src_x, dst, CMP_LT); // s1 = mask where x < 0

    // s0 = π * sign_y
    emit_f32_const(code, s0, core::f32::consts::PI);
    emit_vmulps(code, s0, s0, s3); // s0 = π * sign_y

    // dst = atan_signed - correction = atan_signed - π * sign_y
    emit_vsubps(code, dst, s2, s0); // dst = corrected result

    // Blend: result = mask_neg_x ? corrected : atan_signed
    //   s0 = s1 & dst           (mask & corrected)
    //   s1 = ~s1 & s2           (ANDN: ~mask & atan_signed)
    //   dst = s0 | s1
    emit_vandps(code, s0, s1, dst); // s0 = mask & corrected
    emit_vandnps(code, s1, s1, s2); // s1 = ~mask & atan_signed
    emit_vorps(code, dst, s0, s1); // dst = final result
}

/// atan(x) = atan2(x, 1.0)
pub fn emit_atan_builtin(code: &mut Vec<u8>, dst: Reg, src: Reg, scratch: [Reg; 4]) {
    let s3 = scratch[3];
    // Load 1.0 into s3
    emit_f32_const(code, s3, 1.0_f32);
    // atan(x) = atan2(x, 1.0)
    emit_atan2_builtin(code, dst, src, s3, scratch);
}

/// asin(x) = atan2(x, sqrt(1 - x²))
pub fn emit_asin_builtin(code: &mut Vec<u8>, dst: Reg, src: Reg, scratch: [Reg; 4]) {
    let s0 = scratch[0];
    let s1 = scratch[1];

    // s0 = x²
    emit_vmulps(code, s0, src, src);

    // s1 = 1.0
    emit_f32_const(code, s1, 1.0_f32);

    // s0 = 1.0 - x²
    emit_vsubps(code, s0, s1, s0);

    // s0 = sqrt(1 - x²)
    emit_vsqrtps(code, s0, s0);

    // atan2(x, sqrt(1 - x²))
    emit_atan2_builtin(code, dst, src, s0, scratch);
}

/// acos(x) = atan2(sqrt(1 - x²), x)
pub fn emit_acos_builtin(code: &mut Vec<u8>, dst: Reg, src: Reg, scratch: [Reg; 4]) {
    let s0 = scratch[0];
    let s1 = scratch[1];

    // s0 = x²
    emit_vmulps(code, s0, src, src);

    // s1 = 1.0
    emit_f32_const(code, s1, 1.0_f32);

    // s0 = 1.0 - x²
    emit_vsubps(code, s0, s1, s0);

    // s0 = sqrt(1 - x²)
    emit_vsqrtps(code, s0, s0);

    // atan2(sqrt(1 - x²), x)
    emit_atan2_builtin(code, dst, s0, src, scratch);
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

        // Inverse trigonometric builtins
        OpKind::Atan => emit_atan_builtin(code, dst, src, scratch),
        OpKind::Asin => emit_asin_builtin(code, dst, src, scratch),
        OpKind::Acos => emit_acos_builtin(code, dst, src, scratch),

        _ => panic!("x86_64 unary emit not implemented for {:?}", op),
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

        _ => panic!("x86_64 binary emit not implemented for {:?}", op),
    }
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
        OpKind::Atan2 => emit_atan2_builtin(code, dst, src1, src2, scratch),
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
