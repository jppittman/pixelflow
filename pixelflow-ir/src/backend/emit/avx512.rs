//! x86-64 AVX-512 (EVEX) JIT encoder — 512-bit, 16-lane `zmm` kernels.
//!
//! This is the wide counterpart to the SSE2 (`x86_64.rs`) leaf encoders. It
//! targets the full `zmm0..zmm31` register file via EVEX, so it can also use the
//! extended registers (`zmm16..31`) that VEX cannot reach.
//!
//! Scope (Stage 1): arithmetic, FMA, sqrt/recip/rsqrt, min/max, bitwise, and
//! constant broadcast — enough for the arithmetic core of a kernel. Comparison
//! masks / select (the k-register class) and the transcendental polynomials are
//! separate stages; the backend rejects those up front so nothing here is
//! reached for an unsupported op.
//!
//! Spills use a real stack frame (a `zmm` is 64 bytes — far past the 128-byte
//! red zone the SSE2 path relies on).

use super::Reg;
use crate::OpKind;
use alloc::vec::Vec;

// =============================================================================
// EVEX encoder
// =============================================================================

/// Opcode escape map (EVEX `mm`).
#[derive(Clone, Copy)]
enum Map {
    /// `0F`
    M0F = 1,
    /// `0F38`
    M0F38 = 2,
    /// `0F3A`
    M0F3A = 3,
}

/// Mandatory prefix (EVEX `pp`).
#[derive(Clone, Copy)]
enum Pp {
    /// none — packed single
    None = 0,
    /// `66`
    P66 = 1,
    /// `F3`
    F3 = 2,
    /// `F2`
    F2 = 3,
}

/// Emit a 512-bit EVEX 3-operand register form: `op zmmDST, zmmSRC1, zmmSRC2`,
/// where SRC1 is the non-destructive EVEX.vvvv source and SRC2 is the ModRM
/// r/m. Any of `zmm0..zmm31` is valid. `w` sets EVEX.W.
fn evex_rrr(
    code: &mut Vec<u8>,
    map: Map,
    pp: Pp,
    w: bool,
    opcode: u8,
    dst: u8,
    src1: u8,
    src2: u8,
) {
    // EVEX stores the high register bits inverted.
    let r = ((dst >> 3) & 1) ^ 1; // ModRM.reg bit3
    let rp = ((dst >> 4) & 1) ^ 1; // ModRM.reg bit4 (R')
    let b = ((src2 >> 3) & 1) ^ 1; // ModRM.r/m bit3
    let x = ((src2 >> 4) & 1) ^ 1; // ModRM.r/m bit4 (EVEX.X extends r/m reg)
    let vvvv = (!src1) & 0x0F;
    let vp = ((src1 >> 4) & 1) ^ 1; // vvvv bit4 (V')

    let p0 = (r << 7) | (x << 6) | (b << 5) | (rp << 4) | (map as u8);
    let p1 = ((w as u8) << 7) | (vvvv << 3) | (1 << 2) | (pp as u8);
    // z=0, L'L=10 (512-bit), b(roadcast)=0, V', aaa=0 (no mask).
    let p2 = (0b10 << 5) | (vp << 3);

    code.push(0x62);
    code.push(p0);
    code.push(p1);
    code.push(p2);
    code.push(opcode);
    code.push(0xC0 | ((dst & 7) << 3) | (src2 & 7));
}

/// Emit a 512-bit EVEX `op zmmDST, [rsp + disp32]` (load/store reg + memory).
/// Used for spills/reloads and constant broadcast from the stack. `disp` is a
/// signed displacement from `rsp`.
fn evex_rm_rsp(code: &mut Vec<u8>, map: Map, pp: Pp, w: bool, opcode: u8, reg: u8, disp: i32) {
    let r = ((reg >> 3) & 1) ^ 1;
    let rp = ((reg >> 4) & 1) ^ 1;
    // Memory operand via SIB with base = rsp (encoding 4, bit3 = 0) and no
    // index. EVEX.B/X are stored INVERTED: base bit3 = 0 -> B encoded 1; the
    // "no index" SIB index field is 4 -> X encoded 1. (Encoding B = 0 here was
    // the spill-path bug: it set the base's bit3, addressing r12 instead of
    // rsp and faulting on a garbage pointer.)
    let b = 1u8; // base rsp: logical bit3 0 -> encoded 1
    let x = 1u8; // no index -> encoded 1
    let vvvv = 0x0F; // unused -> all ones
    let vp = 1u8; // V' unused -> 1

    let p0 = (r << 7) | (x << 6) | (b << 5) | (rp << 4) | (map as u8);
    let p1 = ((w as u8) << 7) | (vvvv << 3) | (1 << 2) | (pp as u8);
    let p2 = (0b10 << 5) | (vp << 3);

    code.push(0x62);
    code.push(p0);
    code.push(p1);
    code.push(p2);
    code.push(opcode);
    // ModRM: mod=10 (disp32), reg=reg, r/m=100 (SIB follows).
    code.push(0x80 | ((reg & 7) << 3) | 0b100);
    // SIB: scale=0, index=100 (none), base=100 (rsp).
    code.push(0x24);
    code.extend_from_slice(&disp.to_le_bytes());
}

// --- packed-single arithmetic (0F, no prefix, W0) ---
fn vaddps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    evex_rrr(c, Map::M0F, Pp::None, false, 0x58, d, s1, s2);
}
fn vsubps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    evex_rrr(c, Map::M0F, Pp::None, false, 0x5C, d, s1, s2);
}
fn vmulps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    evex_rrr(c, Map::M0F, Pp::None, false, 0x59, d, s1, s2);
}
fn vdivps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    evex_rrr(c, Map::M0F, Pp::None, false, 0x5E, d, s1, s2);
}
fn vminps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    evex_rrr(c, Map::M0F, Pp::None, false, 0x5D, d, s1, s2);
}
fn vmaxps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    evex_rrr(c, Map::M0F, Pp::None, false, 0x5F, d, s1, s2);
}

// --- bitwise (0F, 66 prefix for the integer-domain forms; use ps forms) ---
fn vandps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    evex_rrr(c, Map::M0F, Pp::None, false, 0x54, d, s1, s2);
}
fn vorps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    evex_rrr(c, Map::M0F, Pp::None, false, 0x56, d, s1, s2);
}
fn vxorps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    evex_rrr(c, Map::M0F, Pp::None, false, 0x57, d, s1, s2);
}
fn vandnps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    evex_rrr(c, Map::M0F, Pp::None, false, 0x55, d, s1, s2);
}

/// Sentinel for the EVEX `vvvv`/`V'` source field on instructions that have no
/// second source (2-operand forms): the field must read as *unused*, which the
/// hardware encodes as `vvvv = 1111` AND `V' = 1`. In `evex_rrr` both are
/// derived from the `src1` index by inversion, so the index that yields
/// `vvvv=1111, V'=1` is **0** (not 0x1F — that has bit4 set, giving `V'=0` and a
/// `#UD` / SIGILL).
const UNUSED_VVVV: u8 = 0;

/// Scratch register for unary mask materialization (neg/abs). zmm15 is outside
/// the backend's allocatable range (zmm4-9), reload regs (zmm11-12), and inputs
/// (zmm0-3), so it is always free here. Lets neg/abs handle `dst == src`.
const UNARY_SCRATCH: Reg = Reg(15);

// --- unary (one source; no second source -> UNUSED_VVVV) ---
/// vsqrtps zmmD, zmmS — EVEX.512.0F.W0 51 /r ; vvvv unused.
fn vsqrtps(c: &mut Vec<u8>, d: u8, s: u8) {
    evex_rrr(c, Map::M0F, Pp::None, false, 0x51, d, UNUSED_VVVV, s);
}

/// vrndscaleps zmmD, zmmS, imm8 — EVEX.512.66.0F3A.W0 08 /r ib ; vvvv unused.
/// (Opcode 08 = packed-single; 09 is packed-double and needs W1.) Round each
/// lane per `imm8` (see the Floor/Ceil/Round arms for the bit layout).
fn vrndscaleps(c: &mut Vec<u8>, d: u8, s: u8, imm: u8) {
    evex_rrr_imm(c, Map::M0F3A, Pp::P66, false, 0x08, d, UNUSED_VVVV, s, imm);
}

// --- FMA (0F38, 66 prefix, W0). 213: dst = src1*dst + src2. ---
fn vfmadd213ps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    evex_rrr(c, Map::M0F38, Pp::P66, false, 0xA8, d, s1, s2);
}

/// vmovaps zmmDST, zmmSRC — register copy (EVEX.512.0F.W0 28 /r).
pub fn emit_mov(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    if dst.0 == src.0 {
        return;
    }
    evex_rrr(
        code,
        Map::M0F,
        Pp::None,
        false,
        0x28,
        dst.0,
        UNUSED_VVVV,
        src.0,
    );
}

/// vmovups zmmDST, [rsp+disp] — 512-bit reload (EVEX.512.0F.W0 10 /r).
/// `vmovups` has NO mandatory prefix; `F3 0F 10` would be the *scalar* `vmovss`.
pub fn emit_load_rsp(code: &mut Vec<u8>, dst: Reg, disp: i32) {
    evex_rm_rsp(code, Map::M0F, Pp::None, false, 0x10, dst.0, disp);
}

/// vmovups [rsp+disp], zmmSRC — 512-bit spill store (EVEX.512.0F.W0 11 /r).
/// `vmovups` has NO mandatory prefix; `F3 0F 11` would be the *scalar* `vmovss`
/// (which caused the spill-path SIGSEGV: a scalar store to a garbage SIB base).
pub fn emit_store_rsp(code: &mut Vec<u8>, src: Reg, disp: i32) {
    evex_rm_rsp(code, Map::M0F, Pp::None, false, 0x11, src.0, disp);
}

/// Broadcast an f32 constant to all 16 lanes of `dst`.
///
/// Writes the bit pattern to `[rsp-4]` then `vbroadcastss zmm, [rsp-4]`
/// (EVEX.512.66.0F38.W0 18 /r). Touches only the red zone below rsp; no GP/zmm
/// clobber. Safe in a leaf, and unaffected by any spill frame (which lives at
/// `[rsp .. rsp+frame)`, i.e. above this).
pub fn emit_const(code: &mut Vec<u8>, dst: Reg, val: f32) {
    let bits = val.to_bits();
    // mov dword [rsp-4], imm32  ->  C7 44 24 FC <imm32>
    code.extend_from_slice(&[0xC7, 0x44, 0x24, 0xFC]);
    code.extend_from_slice(&bits.to_le_bytes());
    // vbroadcastss zmm, [rsp-4]
    evex_rm_rsp_broadcast(code, dst.0, -4);
}

/// `vbroadcastss zmm, [rsp+disp32]` (EVEX.512.66.0F38.W0 18 /r).
///
/// Uses a full `disp32` (`mod=10`) rather than EVEX compressed `disp8`: the
/// compressed form scales the byte by the tuple element size (4 for a
/// `vbroadcastss` scalar source), so a `disp8` of `-4` would address `[rsp-16]`,
/// not `[rsp-4]`. `disp32` is never scaled, so the displacement is literal.
fn evex_rm_rsp_broadcast(code: &mut Vec<u8>, reg: u8, disp: i32) {
    let r = ((reg >> 3) & 1) ^ 1;
    let rp = ((reg >> 4) & 1) ^ 1;
    let p0 = (r << 7) | (1 << 6) | (1 << 5) | (rp << 4) | (Map::M0F38 as u8);
    let p1 = (0x0F << 3) | (1 << 2) | (Pp::P66 as u8);
    let p2 = (0b10 << 5) | (1 << 3);
    code.push(0x62);
    code.push(p0);
    code.push(p1);
    code.push(p2);
    code.push(0x18);
    // mod=10 (disp32), reg=reg, r/m=100 (SIB) ; SIB base=rsp ; disp32
    code.push(0x80 | ((reg & 7) << 3) | 0b100);
    code.push(0x24);
    code.extend_from_slice(&disp.to_le_bytes());
}

// =============================================================================
// Stack frame (real frame; zmm spills are 64 bytes)
// =============================================================================

/// `sub rsp, imm32`.
pub fn emit_sub_rsp(code: &mut Vec<u8>, size: u32) {
    code.extend_from_slice(&[0x48, 0x81, 0xEC]);
    code.extend_from_slice(&size.to_le_bytes());
}

/// `add rsp, imm32`.
pub fn emit_add_rsp(code: &mut Vec<u8>, size: u32) {
    code.extend_from_slice(&[0x48, 0x81, 0xC4]);
    code.extend_from_slice(&size.to_le_bytes());
}

/// `ret`.
pub fn emit_ret(code: &mut Vec<u8>) {
    code.push(0xC3);
}

// =============================================================================
// Op dispatch
// =============================================================================

/// Emit `dst = op(src1, src2)` for a binary arithmetic op.
///
/// EVEX is 3-operand and non-destructive, so unlike SSE there is no
/// two-operand hazard: `src1`/`src2` are never clobbered and may alias `dst`.
/// Returns `Err` for ops not in the Stage-1 arithmetic subset.
pub fn emit_binary(
    code: &mut Vec<u8>,
    op: OpKind,
    dst: Reg,
    src1: Reg,
    src2: Reg,
) -> Result<(), &'static str> {
    let (d, s1, s2) = (dst.0, src1.0, src2.0);
    match op {
        OpKind::Add => vaddps(code, d, s1, s2),
        OpKind::Sub => vsubps(code, d, s1, s2),
        OpKind::Mul => vmulps(code, d, s1, s2),
        OpKind::Div => vdivps(code, d, s1, s2),
        OpKind::Min => vminps(code, d, s1, s2),
        OpKind::Max => vmaxps(code, d, s1, s2),
        _ => return Err("avx512: binary op not in Stage-1 subset"),
    }
    Ok(())
}

// =============================================================================
// Masks & select — a mask is an ordinary vector (all-ones / all-zeros lanes) in
// the regular zmm register file, exactly like NEON. It flows through the shared
// allocator as a normal value; the k-register (k1) is only transient scratch
// inside these encoders, never an allocatable class. This is the trait's job
// (IsaBackend::emit_plan), not the allocator's.
// =============================================================================

/// `vcmpps`/`vpternlog` predicate (imm8). Same ordering as the SSE2 path.
const CMP_EQ: u8 = 0;
const CMP_LT: u8 = 1;
const CMP_LE: u8 = 2;
const CMP_NEQ: u8 = 4;
const CMP_GE: u8 = 5;
const CMP_GT: u8 = 6;

/// Transient k-register used to receive a `vcmpps` result before it is widened
/// to a vector mask. Never allocated — scratch internal to compare emission.
const SCRATCH_K: u8 = 1;

/// Like [`evex_rrr`] but appends an `imm8` (for `vcmpps`, `vpternlogd`).
fn evex_rrr_imm(
    code: &mut Vec<u8>,
    map: Map,
    pp: Pp,
    w: bool,
    opcode: u8,
    dst: u8,
    src1: u8,
    src2: u8,
    imm: u8,
) {
    evex_rrr(code, map, pp, w, opcode, dst, src1, src2);
    code.push(imm);
}

/// Map a comparison `OpKind` to its `vcmpps` predicate imm8.
fn cmp_pred(op: OpKind) -> Option<u8> {
    Some(match op {
        OpKind::Eq => CMP_EQ,
        OpKind::Ne => CMP_NEQ,
        OpKind::Lt => CMP_LT,
        OpKind::Le => CMP_LE,
        OpKind::Gt => CMP_GT,
        OpKind::Ge => CMP_GE,
        _ => return None,
    })
}

/// Whether `op` is a comparison handled by [`emit_compare`].
#[must_use]
pub fn is_compare(op: OpKind) -> bool {
    cmp_pred(op).is_some()
}

/// Emit `dst = (src1 <op> src2) ? all-ones : all-zeros` as a vector mask.
///
/// `vcmpps k1, src1, src2, pred` (EVEX.512.0F.W0 C2 /r ib) writes a k-register;
/// `vpmovm2d dst, k1` (EVEX.512.F3.0F38.W0 38 /r) widens it to a per-lane
/// all-ones/all-zeros vector occupying the allocator-assigned `dst` zmm.
pub fn emit_compare(
    code: &mut Vec<u8>,
    op: OpKind,
    dst: Reg,
    src1: Reg,
    src2: Reg,
) -> Result<(), &'static str> {
    let pred = cmp_pred(op).ok_or("avx512: not a comparison op")?;
    // vcmpps k1, src1, src2, pred  (k-dest in ModRM.reg)
    evex_rrr_imm(
        code,
        Map::M0F,
        Pp::None,
        false,
        0xC2,
        SCRATCH_K,
        src1.0,
        src2.0,
        pred,
    );
    // vpmovm2d dst, k1  (widen mask -> vector)
    evex_rrr(
        code,
        Map::M0F38,
        Pp::F3,
        false,
        0x38,
        dst.0,
        UNUSED_VVVV,
        SCRATCH_K,
    );
    Ok(())
}

/// Emit `dst = mask ? if_true : if_false`, with the vector mask already in
/// `dst` (placed there by `setup_mov`, matching the SSE2/NEON convention).
///
/// One `vpternlogd dst, if_true, if_false, 0xCA` (EVEX.512.66.0F3A.W0 25 /r ib):
/// the truth table 0xCA computes `A?B:C` per bit with A=dst(mask), B=if_true,
/// C=if_false, i.e. a per-lane select for an all-ones/all-zeros mask.
pub fn emit_select(code: &mut Vec<u8>, dst: Reg, if_true: Reg, if_false: Reg) {
    evex_rrr_imm(
        code,
        Map::M0F3A,
        Pp::P66,
        false,
        0x25,
        dst.0,
        if_true.0,
        if_false.0,
        0xCA,
    );
}

/// Set flags from a vector mask for the Select short-circuit guards.
///
/// `vptestmd k1, mask, mask` sets `k1[i]` for each nonzero lane; `kortestw k1,k1`
/// then sets ZF iff `k1 == 0` (all lanes false) and CF iff `k1 == 0xFFFF` (all
/// 16 lanes true). The caller follows with `jz` (all-false) or `jc` (all-true).
pub fn emit_mask_flags(code: &mut Vec<u8>, mask: Reg) {
    // vptestmd k1, mask, mask  (EVEX.512.66.0F38.W0 27 /r)
    evex_rrr(
        code,
        Map::M0F38,
        Pp::P66,
        false,
        0x27,
        SCRATCH_K,
        mask.0,
        mask.0,
    );
    // kortestw k1, k1  (VEX.L0.0F.W0 98 /r) -> C5 F8 98 C9
    code.extend_from_slice(&[0xC5, 0xF8, 0x98, 0xC9]);
}

/// Emit `dst = op(src)` for a unary op (Stage-1 subset).
pub fn emit_unary(code: &mut Vec<u8>, op: OpKind, dst: Reg, src: Reg) -> Result<(), &'static str> {
    match op {
        OpKind::Sqrt => vsqrtps(code, dst.0, src.0),
        OpKind::Neg => {
            // dst = src XOR (-0.0 broadcast). Build the mask in a scratch reg,
            // not dst: dst may alias src, and writing the mask into dst first
            // would clobber the source before the xor reads it.
            emit_const(code, UNARY_SCRATCH, f32::from_bits(0x8000_0000));
            vxorps(code, dst.0, src.0, UNARY_SCRATCH.0);
        }
        OpKind::Abs => {
            // dst = src AND (0x7FFFFFFF broadcast). Same aliasing concern.
            emit_const(code, UNARY_SCRATCH, f32::from_bits(0x7FFF_FFFF));
            vandps(code, dst.0, src.0, UNARY_SCRATCH.0);
        }
        // Rounding: a single EVEX instruction (vrndscaleps), no polynomial.
        // imm8 bit layout: bits[7:4] = scale (0 = integer), bits[3:0] = rounding
        // mode (0 = nearest-even, 1 = toward -inf/floor, 2 = toward +inf/ceil).
        OpKind::Floor => vrndscaleps(code, dst.0, src.0, 0x01),
        OpKind::Ceil => vrndscaleps(code, dst.0, src.0, 0x02),
        OpKind::Round => vrndscaleps(code, dst.0, src.0, 0x00),
        _ => return Err("avx512: unary op not in Stage-1 subset"),
    }
    Ok(())
}

/// Emit a fused multiply-add `dst = a*b + c` where `dst` already holds `c`.
/// (213 form: `vfmadd213ps dst, a, b` == `dst = a*dst + b`; caller arranges
/// operands so this computes the intended `a*b + c`.)
pub fn emit_fmadd_c_in_dst(code: &mut Vec<u8>, dst: Reg, a: Reg, b: Reg) {
    // dst currently = c. We want a*b + c. vfmadd231ps dst, a, b => dst = a*b + dst.
    // 231: EVEX.512.66.0F38.W0 B8 /r.
    evex_rrr(code, Map::M0F38, Pp::P66, false, 0xB8, dst.0, a.0, b.0);
}

/// Bitwise helpers exposed for completeness / future mask emulation.
pub fn emit_and(code: &mut Vec<u8>, dst: Reg, s1: Reg, s2: Reg) {
    vandps(code, dst.0, s1.0, s2.0);
}
pub fn emit_or(code: &mut Vec<u8>, dst: Reg, s1: Reg, s2: Reg) {
    vorps(code, dst.0, s1.0, s2.0);
}
pub fn emit_xor(code: &mut Vec<u8>, dst: Reg, s1: Reg, s2: Reg) {
    vxorps(code, dst.0, s1.0, s2.0);
}
pub fn emit_andn(code: &mut Vec<u8>, dst: Reg, s1: Reg, s2: Reg) {
    vandnps(code, dst.0, s1.0, s2.0);
}
pub fn emit_fmadd213(code: &mut Vec<u8>, dst: Reg, s1: Reg, s2: Reg) {
    vfmadd213ps(code, dst.0, s1.0, s2.0);
}

#[cfg(test)]
mod tests {
    //! Hardware validation. The byte-level EVEX encodings for 2-operand forms,
    //! memory forms, FMA231, and the stack frame are hand-derived; these JIT
    //! real `zmm` kernels and execute them on the host (all 16 lanes), so a bad
    //! byte fails loudly. Runtime tests require `+avx512f`.
    #![allow(clippy::needless_range_loop)]

    #[cfg(target_feature = "avx512f")]
    mod runtime {
        use super::super::*;
        use crate::backend::emit::executable::ExecutableCode;
        use core::arch::x86_64::*;

        type K = unsafe extern "C" fn(__m512, __m512, __m512, __m512) -> __m512;

        fn run(body: &[u8], xs: [f32; 16], ys: [f32; 16], zs: [f32; 16]) -> [f32; 16] {
            let mut code = body.to_vec();
            emit_ret(&mut code);
            let exec = unsafe { ExecutableCode::from_code(&code).expect("mmap") };
            unsafe {
                let f: K = exec.as_fn();
                let r = f(
                    _mm512_loadu_ps(xs.as_ptr()),
                    _mm512_loadu_ps(ys.as_ptr()),
                    _mm512_loadu_ps(zs.as_ptr()),
                    _mm512_setzero_ps(),
                );
                let mut out = [0.0f32; 16];
                _mm512_storeu_ps(out.as_mut_ptr(), r);
                out
            }
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
            for i in 0..16 {
                let w = want(i);
                assert!(
                    (got[i] - w).abs() <= 1e-3,
                    "{tag} lane {i}: got {} want {}",
                    got[i],
                    w
                );
            }
        }

        const X: Reg = Reg(0);
        const Y: Reg = Reg(1);
        const Z: Reg = Reg(2);

        #[test]
        fn binary_ops() {
            let (xs, ys, zs) = lanes();
            let cases: &[(OpKind, fn(f32, f32) -> f32)] = &[
                (OpKind::Add, |a, b| a + b),
                (OpKind::Sub, |a, b| a - b),
                (OpKind::Mul, |a, b| a * b),
                (OpKind::Div, |a, b| a / b),
                (OpKind::Min, |a, b| a.min(b)),
                (OpKind::Max, |a, b| a.max(b)),
            ];
            for &(op, f) in cases {
                let mut c = Vec::new();
                emit_binary(&mut c, op, X, X, Y).unwrap();
                check(run(&c, xs, ys, zs), |i| f(xs[i], ys[i]), "binary");
            }
        }

        #[test]
        fn high_register() {
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            emit_binary(&mut c, OpKind::Mul, Reg(20), X, Y).unwrap();
            emit_mov(&mut c, X, Reg(20));
            check(run(&c, xs, ys, zs), |i| xs[i] * ys[i], "mul via zmm20");
        }

        #[test]
        fn sqrt_op() {
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            emit_unary(&mut c, OpKind::Sqrt, X, Y).unwrap(); // Y > 0
            check(run(&c, xs, ys, zs), |i| ys[i].sqrt(), "sqrt");
        }

        #[test]
        fn neg_abs() {
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            emit_unary(&mut c, OpKind::Neg, X, X).unwrap();
            check(run(&c, xs, ys, zs), |i| -xs[i], "neg");
            let mut c = Vec::new();
            emit_unary(&mut c, OpKind::Abs, X, X).unwrap();
            check(run(&c, xs, ys, zs), |i| xs[i].abs(), "abs");
        }

        #[test]
        fn const_broadcast() {
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            emit_const(&mut c, Reg(5), 2.5);
            emit_binary(&mut c, OpKind::Add, X, X, Reg(5)).unwrap();
            check(run(&c, xs, ys, zs), |i| xs[i] + 2.5, "const+add");
        }

        #[test]
        fn fma_231() {
            let (xs, ys, zs) = lanes();
            // emit_fmadd_c_in_dst(dst, a, b): dst = a*b + dst.
            let mut c = Vec::new();
            emit_mov(&mut c, Reg(5), Z);
            emit_fmadd_c_in_dst(&mut c, Reg(5), X, Y);
            emit_mov(&mut c, X, Reg(5));
            check(run(&c, xs, ys, zs), |i| xs[i] * ys[i] + zs[i], "fma231");
        }

        #[test]
        fn spill_frame_roundtrip() {
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            emit_sub_rsp(&mut c, 64);
            emit_binary(&mut c, OpKind::Mul, Reg(6), X, Y).unwrap();
            emit_store_rsp(&mut c, Reg(6), 0);
            emit_binary(&mut c, OpKind::Add, Reg(6), X, X).unwrap(); // clobber
            emit_load_rsp(&mut c, X, 0);
            emit_add_rsp(&mut c, 64);
            check(run(&c, xs, ys, zs), |i| xs[i] * ys[i], "spill roundtrip");
        }
    }
}
