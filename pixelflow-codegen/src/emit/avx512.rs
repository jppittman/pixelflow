//! x86-64 AVX-512 (EVEX) JIT encoder — 512-bit, 16-lane `zmm` kernels.
//!
//! This is the wide counterpart to the SSE2 (`x86_64.rs`) leaf encoders. It
//! targets the full `zmm0..zmm31` register file via EVEX, so it can also use the
//! extended registers (`zmm16..31`) that VEX cannot reach.
//!
//! Scope: arithmetic, FMA, sqrt/recip/rsqrt, min/max, bitwise, comparisons,
//! select, constant broadcast, the integer bit-manipulation atoms
//! (`IAdd`/`BitAnd`/`BitOr`/`TruncToInt`/`IntToFloat`), and `ShiftImm` (see
//! `emit_shift_imm`) — so the exp/log lowering reaches this backend intact.
//! Comparisons go through the k-register class (`vcmpps` -> `vpmovm2d`, see
//! `emit_compare` below) so every downstream consumer still sees an ordinary
//! all-ones/all-zeros vector, exactly like every other backend — the allocator
//! never learns k-registers exist. Note `vpmovm2d` is AVX-512**DQ**, not F: an
//! F-only part would fault on any kernel containing a comparison.
//!
//! Transcendentals themselves are still a separate lowering stage; ops with no
//! rule here are refused up front rather than mis-emitted.
//!
//! Spills use a real stack frame (a `zmm` is 64 bytes — far past the 128-byte
//! red zone the SSE2 path relies on).

use super::x86_64;
use super::x86_64::{Disp, Imm32, Mem, MovLoadPtr, NoDisp, ptr};
use super::{AsmProgram, EncodedInst, KReg, Reg, assemble, unimplemented_op};
use alloc::vec::Vec;
use pixelflow_ir::OpKind;

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
}

/// The identity of one EVEX-512 instruction: opcode map, mandatory prefix, W
/// bit, opcode byte. This quadruple is *which instruction* — it is constant
/// per mnemonic, so each mnemonic below states it exactly once and the
/// operand form (`rrr`/`rm`) supplies the per-call parts.
///
/// The 128- and 256-bit twins are `x86_64::Vex` and `avx2::Vex`.
#[derive(Clone, Copy)]
struct Evex {
    map: Map,
    pp: Pp,
    w: bool,
    opcode: u8,
}

impl Evex {
    const fn new(map: Map, pp: Pp, opcode: u8) -> Self {
        Self {
            map,
            pp,
            w: false,
            opcode,
        }
    }
    /// Map `0F`, no prefix — the packed-single family.
    const fn m0f(opcode: u8) -> Self {
        Self::new(Map::M0F, Pp::None, opcode)
    }
    /// Map `0F`, `66` — the integer-domain family.
    const fn m0f_66(opcode: u8) -> Self {
        Self::new(Map::M0F, Pp::P66, opcode)
    }
    /// Map `0F`, `F3`.
    const fn m0f_f3(opcode: u8) -> Self {
        Self::new(Map::M0F, Pp::F3, opcode)
    }
    /// Map `0F38`, `66`.
    const fn m0f38_66(opcode: u8) -> Self {
        Self::new(Map::M0F38, Pp::P66, opcode)
    }
    /// Map `0F38`, `F3` — the mask-to-vector widening family.
    const fn m0f38_f3(opcode: u8) -> Self {
        Self::new(Map::M0F38, Pp::F3, opcode)
    }
    /// Map `0F3A`, `66` — the imm8 family (round, ternlog).
    const fn m0f3a_66(opcode: u8) -> Self {
        Self::new(Map::M0F3A, Pp::P66, opcode)
    }

    /// Attach an imm8 (`vcmpps` predicate, rounding mode, shift count,
    /// `vpternlogd` truth table); the returned value emits it after the
    /// instruction.
    const fn imm(self, imm: u8) -> EvexImm {
        EvexImm { evex: self, imm }
    }

    /// 3-operand register form: `op zmmDST, zmmSRC1, zmmSRC2`, where SRC1 is
    /// the non-destructive EVEX.vvvv source and SRC2 is the ModRM r/m. Any of
    /// `zmm0..zmm31` is valid.
    fn rrr(self, dst: u8, src1: u8, src2: u8) -> EncodedInst {
        let mut inst = EncodedInst::new();
        // EVEX stores the high register bits inverted.
        let r = ((dst >> 3) & 1) ^ 1; // ModRM.reg bit3
        let rp = ((dst >> 4) & 1) ^ 1; // ModRM.reg bit4 (R')
        let b = ((src2 >> 3) & 1) ^ 1; // ModRM.r/m bit3
        let x = ((src2 >> 4) & 1) ^ 1; // ModRM.r/m bit4 (EVEX.X extends r/m reg)
        let vvvv = (!src1) & 0x0F;
        let vp = ((src1 >> 4) & 1) ^ 1; // vvvv bit4 (V')

        self.prefix_into(
            &mut inst,
            (r << 7) | (x << 6) | (b << 5) | (rp << 4),
            vvvv,
            vp,
        );
        inst.push(0xC0 | ((dst & 7) << 3) | (src2 & 7));
        inst
    }

    /// `op zmmREG, [addr]` — the memory form used for spills, reloads,
    /// constant broadcast and the collapse loop's output store.
    ///
    /// EVEX.R/B are stored INVERTED, and X likewise: there is no index
    /// register in any of these forms, so X is always the encoded 1.
    /// (Encoding B = 0 for an `rsp` base was the spill-path bug: it set the
    /// base's bit 3, addressing r12 and faulting on a garbage pointer.)
    /// The ModRM/SIB/displacement tail is the architecture's, not EVEX's, so
    /// it comes from `x86_64::mem_operand`.
    fn rm<D: Disp>(self, reg: u8, addr: Mem<D>) -> EncodedInst {
        let mut inst = EncodedInst::new();
        let r = ((reg >> 3) & 1) ^ 1;
        let rp = ((reg >> 4) & 1) ^ 1;
        let b = ((addr.base.0 >> 3) & 1) ^ 1;
        let x = 1u8; // no index -> encoded 1

        self.prefix_into(
            &mut inst,
            (r << 7) | (x << 6) | (b << 5) | (rp << 4),
            0x0F,
            1,
        );
        x86_64::mem_operand_into(&mut inst, reg, addr);
        inst
    }

    /// The 4-byte EVEX prefix plus the opcode byte, shared by both forms.
    /// `reg_ext` is the assembled `R X B R'` nibble of P0; `vvvv`/`vp` are the
    /// extra-source fields. Every one of them is already inverted by the
    /// caller, as the encoding requires.
    fn prefix_into(self, inst: &mut EncodedInst, reg_ext: u8, vvvv: u8, vp: u8) {
        inst.push(0x62);
        inst.push(reg_ext | (self.map as u8));
        inst.push(((self.w as u8) << 7) | (vvvv << 3) | (1 << 2) | (self.pp as u8));
        // z=0, L'L=10 (512-bit), b(roadcast)=0, V', aaa=0 (no mask).
        inst.push((0b10 << 5) | (vp << 3));
        inst.push(self.opcode);
    }
}

/// An [`Evex`] instruction carrying its imm8.
#[derive(Clone, Copy)]
struct EvexImm {
    evex: Evex,
    imm: u8,
}

impl EvexImm {
    /// Register form with the imm8 appended.
    fn rrr(self, dst: u8, src1: u8, src2: u8) -> EncodedInst {
        let mut inst = self.evex.rrr(dst, src1, src2);
        inst.push(self.imm);
        inst
    }
}

// --- packed-single arithmetic (0F, no prefix, W0) ---
fn vaddps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    assemble(c, [Evex::m0f(0x58).rrr(d, s1, s2)]);
}
fn vsubps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    assemble(c, [Evex::m0f(0x5C).rrr(d, s1, s2)]);
}
fn vmulps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    assemble(c, [Evex::m0f(0x59).rrr(d, s1, s2)]);
}
fn vdivps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    assemble(c, [Evex::m0f(0x5E).rrr(d, s1, s2)]);
}
fn vminps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    assemble(c, [Evex::m0f(0x5D).rrr(d, s1, s2)]);
}
fn vmaxps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    assemble(c, [Evex::m0f(0x5F).rrr(d, s1, s2)]);
}

// --- bitwise (0F, 66 prefix for the integer-domain forms; use ps forms) ---
fn vandps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    assemble(c, [Evex::m0f(0x54).rrr(d, s1, s2)]);
}
fn vorps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    assemble(c, [Evex::m0f(0x56).rrr(d, s1, s2)]);
}
fn vxorps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    assemble(c, [Evex::m0f(0x57).rrr(d, s1, s2)]);
}
/// Sentinel for the EVEX `vvvv`/`V'` source field on instructions that have no
/// second source (2-operand forms): the field must read as *unused*, which the
/// hardware encodes as `vvvv = 1111` AND `V' = 1`. In `evex_rrr` both are
/// derived from the `src1` index by inversion, so the index that yields
/// `vvvv=1111, V'=1` is **0** (not 0x1F — that has bit4 set, giving `V'=0` and a
/// `#UD` / SIGILL).
const UNUSED_VVVV: u8 = 0;

/// How many registers this backend's encodings need beyond their operands.
///
/// Only the `Neg`/`Abs` sign mask: EVEX is non-destructive and `vpternlogd`
/// blends a select with no temporary.
pub(crate) fn temps_for(op: &super::ScheduledOp) -> u8 {
    use super::ScheduledOp;
    match op {
        ScheduledOp::Unary(OpKind::Neg | OpKind::Abs, _) => 1,
        // The gather's truncated-index lanes and its destination.
        ScheduledOp::Gather(..) => 2,
        _ => 0,
    }
}

// --- unary (one source; no second source -> UNUSED_VVVV) ---
/// vsqrtps zmmD, zmmS — EVEX.512.0F.W0 51 /r ; vvvv unused.
fn vsqrtps(c: &mut Vec<u8>, d: u8, s: u8) {
    assemble(c, [Evex::m0f(0x51).rrr(d, UNUSED_VVVV, s)]);
}

/// vrndscaleps zmmD, zmmS, imm8 — EVEX.512.66.0F3A.W0 08 /r ib ; vvvv unused.
/// (Opcode 08 = packed-single; 09 is packed-double and needs W1.) Round each
/// lane per `imm8` (see the Floor/Ceil/Round arms for the bit layout).
fn vrndscaleps(c: &mut Vec<u8>, d: u8, s: u8, imm: u8) {
    assemble(c, [Evex::m0f3a_66(0x08).imm(imm).rrr(d, UNUSED_VVVV, s)]);
}

/// vrcp14ps zmmD, zmmS — EVEX.512.66.0F38.W0 4C /r ; vvvv unused. AVX-512F's
/// replacement for AVX's `vrcpps` (EVEX has no `0F 53` form); ~2^-14 relative
/// error, matching `Recip`'s existing "approximate reciprocal" contract on
/// every other backend (SSE2's `rcpps`, AVX2's `vrcpps`).
fn vrcp14ps(c: &mut Vec<u8>, d: u8, s: u8) {
    assemble(c, [Evex::m0f38_66(0x4C).rrr(d, UNUSED_VVVV, s)]);
}

/// vrsqrt14ps zmmD, zmmS — EVEX.512.66.0F38.W0 4E /r ; vvvv unused.
/// AVX-512F's replacement for AVX's `vrsqrtps`, same accuracy tier as
/// `vrcp14ps` above.
fn vrsqrt14ps(c: &mut Vec<u8>, d: u8, s: u8) {
    assemble(c, [Evex::m0f38_66(0x4E).rrr(d, UNUSED_VVVV, s)]);
}

// --- integer-domain primitives (exp/log lowering) ---
// Same opcodes as the AVX2 backend's VEX forms, EVEX-wrapped at 512 bits.

/// vcvttps2dq zmmD, zmmS — EVEX.512.F3.0F.W0 5B /r ; vvvv unused.
fn vcvttps2dq(c: &mut Vec<u8>, d: u8, s: u8) {
    assemble(c, [Evex::m0f_f3(0x5B).rrr(d, UNUSED_VVVV, s)]);
}

/// vcvtdq2ps zmmD, zmmS — EVEX.512.0F.W0 5B /r ; vvvv unused.
fn vcvtdq2ps(c: &mut Vec<u8>, d: u8, s: u8) {
    assemble(c, [Evex::m0f(0x5B).rrr(d, UNUSED_VVVV, s)]);
}

/// vpaddd zmmD, zmmS1, zmmS2 — EVEX.512.66.0F.W0 FE /r.
fn vpaddd(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    assemble(c, [Evex::m0f_66(0xFE).rrr(d, s1, s2)]);
}

/// vpslld zmmD, zmmS, imm8 — EVEX.512.66.0F.W0 72 /6 ib. The shift-by-imm
/// group encodes the operation in ModRM.reg (/6 = left) and the DESTINATION
/// in vvvv, with the source in r/m — reg/vvvv swap roles vs. ordinary rrr.
fn vpslld_imm(c: &mut Vec<u8>, d: u8, s: u8, imm: u8) {
    assemble(c, [Evex::m0f_66(0x72).imm(imm).rrr(6, d, s)]);
}

/// vpsrld zmmD, zmmS, imm8 — EVEX.512.66.0F.W0 72 /2 ib (logical, zero-fill).
fn vpsrld_imm(c: &mut Vec<u8>, d: u8, s: u8, imm: u8) {
    assemble(c, [Evex::m0f_66(0x72).imm(imm).rrr(2, d, s)]);
}

/// vmovaps zmmDST, zmmSRC — register copy (EVEX.512.0F.W0 28 /r).
pub fn emit_mov(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    if dst.0 == src.0 {
        return;
    }
    assemble(code, [Evex::m0f(0x28).rrr(dst.0, UNUSED_VVVV, src.0)]);
}

/// A slot in the allocated spill frame. AVX-512 kernels are leaves with no
/// base pointer, so a slot *is* `rsp + offset`.
const fn frame_slot(offset: u32) -> Mem<Imm32> {
    Mem {
        base: ptr::RSP,
        disp: Imm32(offset as i32),
    }
}

/// Where [`emit_const`] stages an f32 before broadcasting it: four bytes of
/// red zone below `rsp`, never touched by a spill frame (which lives at
/// `[rsp .. rsp+N)`).
///
/// A full `disp32`, not EVEX's compressed `disp8`: the compressed form scales
/// the byte by the tuple element size (4 for a `vbroadcastss` scalar source),
/// so a `disp8` of -4 would address `[rsp-16]`. `disp32` is never scaled.
const RED_ZONE_CONST: Mem<Imm32> = Mem {
    base: ptr::RSP,
    disp: Imm32(-4),
};

/// Broadcast an f32 constant to all 16 lanes of `dst`.
///
/// Writes the bit pattern to `[rsp-4]` then `vbroadcastss zmm, [rsp-4]`
/// (EVEX.512.66.0F38.W0 18 /r). Touches only the red zone below rsp; no GP/zmm
/// clobber. Safe in a leaf, and unaffected by any spill frame (which lives at
/// `[rsp .. rsp+frame)`, i.e. above this).
pub fn emit_const(code: &mut Vec<u8>, dst: Reg, val: f32) {
    let bits = val.to_bits();
    let mut mov_imm = EncodedInst::new();
    // mov dword [rsp-4], imm32  ->  C7 44 24 FC <imm32>
    mov_imm.extend(&[0xC7, 0x44, 0x24, 0xFC]);
    mov_imm.extend(&bits.to_le_bytes());
    // vbroadcastss zmm, [rsp-4]
    assemble(
        code,
        [mov_imm, Evex::m0f38_66(0x18).rm(dst.0, RED_ZONE_CONST)],
    );
}

/// `dst = splat(block[offset])` at 512 bits: `mov rax, [rdi + ctx_slot*8]`
/// then `vbroadcastss zmm<dst>, [rax + 4*offset]` (EVEX.512.66.0F38.W0 18
/// /r). A full `disp32`, as [`emit_const`]'s is, so EVEX's compressed-`disp8`
/// scaling never enters into it. See `x86_64::emit_uniform_load` for the
/// register contract.
pub fn emit_uniform_load(code: &mut Vec<u8>, dst: Reg, load: super::UniformLoad) {
    AsmProgram::from([
        MovLoadPtr {
            dst: ptr::RAX,
            base: ptr::RDI,
            disp: i32::from(load.ctx_slot) * 8,
        }
        .encode(),
        Evex::m0f38_66(0x18).rm(
            dst.0,
            Mem {
                base: ptr::RAX,
                disp: Imm32(i32::from(load.offset) * 4),
            },
        ),
    ])
    .assemble(code);
}

// =============================================================================
// Stack frame (real frame; zmm spills are 64 bytes)
// =============================================================================

// =============================================================================
// Op dispatch
// =============================================================================

/// Emit `dst = op(src1, src2)` for a binary arithmetic op.
///
/// EVEX is 3-operand and non-destructive, so unlike SSE there is no
/// two-operand hazard: `src1`/`src2` are never clobbered and may alias `dst`.
/// Returns `Err` for ops not in the Stage-1 arithmetic subset.
pub fn emit_binary(code: &mut Vec<u8>, op: OpKind, dst: Reg, src1: Reg, src2: Reg) {
    let (d, s1, s2) = (dst.0, src1.0, src2.0);
    match op {
        OpKind::Add => vaddps(code, d, s1, s2),
        OpKind::Sub => vsubps(code, d, s1, s2),
        OpKind::Mul => vmulps(code, d, s1, s2),
        OpKind::Div => vdivps(code, d, s1, s2),
        OpKind::Min => vminps(code, d, s1, s2),
        OpKind::Max => vmaxps(code, d, s1, s2),
        OpKind::BitAnd => vandps(code, d, s1, s2),
        OpKind::BitOr => vorps(code, d, s1, s2),
        // Integer add on lane bit patterns (exp/log exponent arithmetic).
        OpKind::IAdd => vpaddd(code, d, s1, s2),
        _ => unimplemented_op("avx-512", op),
    }
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
const SCRATCH_K: KReg = KReg(1);

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
pub fn emit_compare(code: &mut Vec<u8>, op: OpKind, dst: Reg, src1: Reg, src2: Reg) {
    let Some(pred) = cmp_pred(op) else {
        unimplemented_op("avx-512", op)
    };
    // vcmpps k1, src1, src2, pred  (k-dest in ModRM.reg)
    // vpmovm2d dst, k1  (widen mask -> vector)
    assemble(
        code,
        [
            Evex::m0f(0xC2).imm(pred).rrr(SCRATCH_K.0, src1.0, src2.0),
            Evex::m0f38_f3(0x38).rrr(dst.0, UNUSED_VVVV, SCRATCH_K.0),
        ],
    );
}

/// Emit `dst = mask ? if_true : if_false`, with the vector mask already in
/// `dst` (placed there by `setup_mov`, matching the SSE2/NEON convention).
///
/// One `vpternlogd dst, if_true, if_false, 0xCA` (EVEX.512.66.0F3A.W0 25 /r ib):
/// the truth table 0xCA computes `A?B:C` per bit with A=dst(mask), B=if_true,
/// C=if_false, i.e. a per-lane select for an all-ones/all-zeros mask.
pub fn emit_select(code: &mut Vec<u8>, dst: Reg, if_true: Reg, if_false: Reg) {
    assemble(
        code,
        [Evex::m0f3a_66(0x25)
            .imm(0xCA)
            .rrr(dst.0, if_true.0, if_false.0)],
    );
}

/// Set flags from a vector mask for the Select short-circuit guards.
///
/// `vptestmd k1, mask, mask` sets `k1[i]` for each nonzero lane; `kortestw k1,k1`
/// then sets ZF iff `k1 == 0` (all lanes false) and CF iff `k1 == 0xFFFF` (all
/// 16 lanes true). The caller follows with `jz` (all-false) or `jc` (all-true).
pub fn emit_mask_flags(code: &mut Vec<u8>, mask: Reg) {
    assemble(
        code,
        [
            // vptestmd k1, mask, mask  (EVEX.512.66.0F38.W0 27 /r)
            Evex::m0f38_66(0x27).rrr(SCRATCH_K.0, mask.0, mask.0),
            // kortestw k1, k1  (VEX.L0.0F.W0 98 /r) -> C5 F8 98 C9
            EncodedInst::from_slice(&[0xC5, 0xF8, 0x98, 0xC9]),
        ],
    );
}

/// Emit `dst = op(src)` for a unary op (Stage-1 subset).
/// Emit `dst = src << amount` / `dst = src >> amount` (logical, zero-fill)
/// on lane bit patterns. The amount is a compile-time immediate — the
/// schedule folds the `Const` RHS out (`ScheduledOp::ShiftImm`).
pub fn emit_shift_imm(code: &mut Vec<u8>, op: OpKind, dst: Reg, src: Reg, amount: u8) {
    match op {
        OpKind::Shl => vpslld_imm(code, dst.0, src.0, amount),
        OpKind::Shr => vpsrld_imm(code, dst.0, src.0, amount),
        _ => unimplemented_op("avx-512", op),
    }
}

/// `dst = op(src)`.
///
/// `temp` is the allocator's temp for this instruction; only `Neg` and `Abs`
/// use it, to hold the sign mask.
pub fn emit_unary(code: &mut Vec<u8>, op: OpKind, dst: Reg, src: Reg, temp: Option<Reg>) {
    match op {
        OpKind::Sqrt => vsqrtps(code, dst.0, src.0),
        OpKind::Neg => {
            // dst = src XOR (-0.0 broadcast). Build the mask in the temp, not
            // dst: dst may alias src, and writing the mask into dst first would
            // clobber the source before the xor reads it.
            let mask = super::declared_temp(temp);
            emit_const(code, mask, f32::from_bits(0x8000_0000));
            vxorps(code, dst.0, src.0, mask.0);
        }
        OpKind::Abs => {
            // dst = src AND (0x7FFFFFFF broadcast). Same aliasing concern.
            let mask = super::declared_temp(temp);
            emit_const(code, mask, f32::from_bits(0x7FFF_FFFF));
            vandps(code, dst.0, src.0, mask.0);
        }
        // Rounding: a single EVEX instruction (vrndscaleps), no polynomial.
        // imm8 bit layout: bits[7:4] = scale (0 = integer), bits[3:0] = rounding
        // mode (0 = nearest-even, 1 = toward -inf/floor, 2 = toward +inf/ceil).
        OpKind::Floor => vrndscaleps(code, dst.0, src.0, 0x01),
        OpKind::Ceil => vrndscaleps(code, dst.0, src.0, 0x02),
        OpKind::Round => vrndscaleps(code, dst.0, src.0, 0x00),
        OpKind::Recip => vrcp14ps(code, dst.0, src.0),
        OpKind::Rsqrt => vrsqrt14ps(code, dst.0, src.0),
        // Int/float domain crossings, exactly the hardware's cvttps2dq /
        // cvtdq2ps — the primitives exp/log lower to.
        OpKind::TruncToInt => vcvttps2dq(code, dst.0, src.0),
        OpKind::IntToFloat => vcvtdq2ps(code, dst.0, src.0),
        _ => unimplemented_op("avx-512", op),
    }
}

/// Emit a fused multiply-add `dst = a*b + c` where `dst` already holds `c`.
/// (213 form: `vfmadd213ps dst, a, b` == `dst = a*dst + b`; caller arranges
/// operands so this computes the intended `a*b + c`.)
pub fn emit_fmadd_c_in_dst(code: &mut Vec<u8>, dst: Reg, a: Reg, b: Reg) {
    // dst currently = c. We want a*b + c. vfmadd231ps dst, a, b => dst = a*b + dst.
    // 231: EVEX.512.66.0F38.W0 B8 /r.
    assemble(code, [Evex::m0f38_66(0xB8).rrr(dst.0, a.0, b.0)]);
}

/// Bitwise helpers exposed for completeness / future mask emulation.
pub fn emit_and(code: &mut Vec<u8>, dst: Reg, s1: Reg, s2: Reg) {
    vandps(code, dst.0, s1.0, s2.0);
}
// =============================================================================
// Bound-memory gather (RawGather lowering target)
//
// `vgatherdps zmm{k1}, [base_gpr + zmm_index*4]` reads one f32 per lane from a
// bound buffer. The lowered index is a float (`clamp(floor(x))·1 + …`), so it is
// first truncated to signed int32 lanes with `vcvttps2dq`. The writemask k1 must
// be all-ones going in (the instruction clears completed lanes), so it is reset
// before every gather.
// =============================================================================

/// `vcvttps2dq zmmDST, zmmSRC` — truncate packed f32 → signed int32 lanes
/// (EVEX.512.F3.0F.W0 5B /r). The lowered gather index is an exact non-negative
/// integer in float form, so truncation is lossless and matches the reference
/// interpreter's `floorf(index) as usize`.
pub fn emit_cvttps2dq(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    assemble(code, [Evex::m0f_f3(0x5B).rrr(dst.0, UNUSED_VVVV, src.0)]);
}

/// Set the gather writemask `k1` to all-ones (`mov eax, 0xFFFF; kmovw k1, eax`).
///
/// A gather requires a non-zero writemask and *clears* the bits it completes, so
/// this must run before each gather. Clobbers `eax` (caller-saved scratch).
pub fn emit_set_gather_mask(code: &mut Vec<u8>) {
    assemble(
        code,
        [
            // mov eax, 0x0000FFFF
            EncodedInst::from_slice(&[0xB8, 0xFF, 0xFF, 0x00, 0x00]),
            // kmovw k1, eax  (VEX.L0.0F.W0 92 /r ; ModRM 11 001 000)
            EncodedInst::from_slice(&[0xC5, 0xF8, 0x92, 0xC8]),
        ],
    );
}

/// `mov dstGPR, [ctxGPR + disp32]` (REX.W 8B /r) — load a 64-bit buffer base
/// pointer out of the context struct. `dst_gpr`/`ctx_gpr` are raw GP register
/// numbers (e.g. `rax`=0, `rdi`=7). Only low-8 GPRs are supported (no REX.B),
/// which is all the emitter needs (`rax` dest, `rdi` context).
pub fn emit_load_ptr_from_ctx(code: &mut Vec<u8>, dst_gpr: u8, ctx_gpr: u8, disp: i32) {
    debug_assert!(
        dst_gpr < 8 && ctx_gpr < 8,
        "emit_load_ptr_from_ctx: GPR8 only"
    );
    // REX.W ; 8B ; mod=10 reg=dst r/m=ctx ; disp32
    let mut inst = EncodedInst::new();
    inst.push(0x48);
    inst.push(0x8B);
    inst.push(0x80 | ((dst_gpr & 7) << 3) | (ctx_gpr & 7));
    inst.extend(&disp.to_le_bytes());
    assemble(code, [inst]);
}

/// `vgatherdps zmmDST{k1}, [baseGPR + zmmINDEX*4]`
/// (EVEX.512.66.0F38.W0 92 /vsib, mask = k1, scale = 4).
///
/// Gathers one f32 per lane at `base + index_lane*4`. The caller must ensure
/// `k1` is all-ones ([`emit_set_gather_mask`]), the index lanes are int32
/// ([`emit_cvttps2dq`]), and `dst != index` (the instruction forbids the
/// destination and index vectors aliasing). `base_gpr` must not be rbp/r13
/// (mod=00 SIB base restriction) — the emitter uses `rax`.
/// Pure encoding for `vgatherdps zmmDST{k1}, [baseGPR + zmmINDEX*4]`
#[must_use]
pub fn gather(dst: Reg, base_gpr: u8, index: Reg) -> EncodedInst {
    let d = dst.0;
    let idx = index.0;
    let base = base_gpr;
    debug_assert!(d != idx, "vgatherdps: dst and index must differ");
    debug_assert!(
        base != 5 && base != 13,
        "vgatherdps: base must not be rbp/r13"
    );

    let r = ((d >> 3) & 1) ^ 1; // dst bit3  -> EVEX.R
    let rp = ((d >> 4) & 1) ^ 1; // dst bit4  -> EVEX.R'
    let x = ((idx >> 3) & 1) ^ 1; // index bit3 -> EVEX.X
    let b = ((base >> 3) & 1) ^ 1; // base bit3  -> EVEX.B
    let vp = ((idx >> 4) & 1) ^ 1; // index bit4 -> EVEX.V'
    let vvvv = 0x0F; // unused -> encoded 1111

    let p0 = (r << 7) | (x << 6) | (b << 5) | (rp << 4) | (Map::M0F38 as u8);
    // W=0 (bit7 clear): gather uses signed dword indices.
    let p1 = (vvvv << 3) | (1 << 2) | (Pp::P66 as u8);
    // z=0, L'L=10 (512-bit), b=0, V' = index bit4, aaa=001 (k1).
    let p2 = (0b10 << 5) | (vp << 3) | 0b001;

    let mut inst = EncodedInst::new();
    inst.push(0x62);
    inst.push(p0);
    inst.push(p1);
    inst.push(p2);
    inst.push(0x92);
    // ModRM: mod=00, reg=dst[2:0], r/m=100 (SIB follows).
    inst.push(((d & 7) << 3) | 0b100);
    // SIB: scale=10 (*4), index=idx[2:0], base=base[2:0].
    inst.push((0b10 << 6) | ((idx & 7) << 3) | (base & 7));
    inst
}

pub fn emit_gather(code: &mut Vec<u8>, dst: Reg, base_gpr: u8, index: Reg) {
    AsmProgram::from([gather(dst, base_gpr, index)]).assemble(code);
}

#[cfg(test)]
mod tests {
    //! Hardware validation. The byte-level EVEX encodings for 2-operand forms,
    //! memory forms, FMA231, and the stack frame are hand-derived; these JIT
    //! real `zmm` kernels and execute them on the host (all 16 lanes), so a bad
    //! byte fails loudly. Runtime tests require `+avx512f`.
    #![allow(clippy::needless_range_loop)]
    use super::*;

    #[test]
    fn emit_and_emits_the_same_bytes_as_vandps() {
        // `emit_and` is `pub fn` (bitwise helpers exposed for completeness /
        // future mask emulation, per its doc comment) but nothing in this
        // file or the driver calls it — pin it directly against the private
        // `vandps` it wraps, whose own correctness is already proven by
        // `emit_unary_negates_and_takes_the_absolute_value_of_every_lane`'s
        // Abs case.
        let mut via_and = Vec::new();
        emit_and(&mut via_and, Reg(3), Reg(1), Reg(2));
        let mut via_vandps = Vec::new();
        vandps(&mut via_vandps, 3, 1, 2);
        assert_eq!(via_and, via_vandps);
    }

    #[test]
    fn emit_load_ptr_from_ctx_encodes_arbitrary_low_gpr_pairs() {
        // The only production caller (`emit_uniform_load`) always passes
        // dst_gpr=rax(0), ctx_gpr=rdi(7) — values that happen to make this
        // function's `&`/`<<` bit ops indistinguishable from `|`/`>>`
        // (0 shifted either direction is 0; `7 & 7 == 7 | 7`). Pin the byte
        // encoding directly with values that don't share that coincidence.
        let mut c = Vec::new();
        emit_load_ptr_from_ctx(&mut c, 5, 3, 0x10);
        assert_eq!(
            c,
            vec![0x48, 0x8B, 0x80 | (5 << 3) | 3, 0x10, 0x00, 0x00, 0x00],
            "REX.W mov r5, [r3 + 0x10]"
        );
    }

    #[cfg(target_feature = "avx512f")]
    mod runtime {
        use super::super::*;
        use crate::emit::executable::ExecutableCode;
        use crate::emit::{Gpr, PtrReg};
        use core::arch::x86_64::*;

        // Passing __m512 by value IS the emitted ABI (SysV: zmm0-7), so
        // not-FFI-safe is a false positive here, as for `executable`'s aliases.
        #[allow(improper_ctypes_definitions)]
        type K = unsafe extern "C" fn(__m512, __m512, __m512, __m512) -> __m512;

        fn run(body: &[u8], xs: [f32; 16], ys: [f32; 16], zs: [f32; 16]) -> [f32; 16] {
            let mut code = body.to_vec();
            crate::emit::x86_64::ret(&mut code);
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
        /// Standing in for the allocator's instruction temp: any register
        /// disjoint from the operands each case uses.
        const TEMP: Reg = Reg(15);
        const Z: Reg = Reg(2);

        /// One row of the binary-op table: the op and its scalar reference.
        type BinaryCase = (OpKind, fn(f32, f32) -> f32);

        #[test]
        fn emit_binary_matches_the_scalar_reference_for_every_arithmetic_op() {
            let (xs, ys, zs) = lanes();
            let cases: &[BinaryCase] = &[
                (OpKind::Add, |a, b| a + b),
                (OpKind::Sub, |a, b| a - b),
                (OpKind::Mul, |a, b| a * b),
                (OpKind::Div, |a, b| a / b),
                (OpKind::Min, |a, b| a.min(b)),
                (OpKind::Max, |a, b| a.max(b)),
            ];
            for &(op, f) in cases {
                let mut c = Vec::new();
                emit_binary(&mut c, op, X, X, Y);
                check(run(&c, xs, ys, zs), |i| f(xs[i], ys[i]), "binary");
            }
        }

        #[test]
        fn emit_binary_writes_a_high_numbered_register() {
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            emit_binary(&mut c, OpKind::Mul, Reg(20), X, Y);
            emit_mov(&mut c, X, Reg(20));
            check(run(&c, xs, ys, zs), |i| xs[i] * ys[i], "mul via zmm20");
        }

        #[test]
        fn emit_load_and_emit_store_address_a_high_numbered_base_register_correctly() {
            // Every production caller in this file addresses memory through
            // rsp or rax (both < r8), so `Evex::rm`'s B-bit inversion for a
            // >= r8 base has no other coverage. Move the incoming pointer
            // into r9 (rbp/r13 have their own mod=00 RIP-relative special
            // case, which `mem_operand` refuses outright) and round-trip
            // through it to pin that bit.
            #[allow(improper_ctypes_definitions)]
            type F = unsafe extern "C" fn(*mut f32);

            let r9 = Gpr(9);
            let mut c = Vec::new();
            x86_64::mov(&mut c, r9, x86_64::gpr::RDI);
            let via_r9 = Mem {
                base: PtrReg(9),
                disp: NoDisp,
            };
            AsmProgram::from([Evex::m0f(0x10).rm(X.0, via_r9)]).assemble(&mut c);
            emit_const(&mut c, Reg(5), 1.0);
            emit_binary(&mut c, OpKind::Add, X, X, Reg(5));
            AsmProgram::from([Evex::m0f(0x11).rm(X.0, via_r9)]).assemble(&mut c);
            AsmProgram::from([crate::emit::x86_64::Inst::Ret]).assemble(&mut c);

            let mut buf = [0.0f32; 16];
            for (i, v) in buf.iter_mut().enumerate() {
                *v = i as f32;
            }
            let exec = unsafe { ExecutableCode::from_code(&c).expect("mmap") };
            unsafe {
                let f: F = exec.as_fn();
                f(buf.as_mut_ptr());
            }
            for (i, &v) in buf.iter().enumerate() {
                assert_eq!(v, i as f32 + 1.0, "lane {i}");
            }
        }

        #[test]
        fn emit_unary_computes_sqrt_of_a_positive_operand() {
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            emit_unary(&mut c, OpKind::Sqrt, X, Y, None); // Y > 0
            check(run(&c, xs, ys, zs), |i| ys[i].sqrt(), "sqrt");
        }

        #[test]
        fn emit_unary_negates_and_takes_the_absolute_value_of_every_lane() {
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            emit_unary(&mut c, OpKind::Neg, X, X, Some(TEMP));
            check(run(&c, xs, ys, zs), |i| -xs[i], "neg");
            let mut c = Vec::new();
            emit_unary(&mut c, OpKind::Abs, X, X, Some(TEMP));
            check(run(&c, xs, ys, zs), |i| xs[i].abs(), "abs");
        }

        #[test]
        fn emit_const_broadcasts_and_adds_to_every_lane() {
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            emit_const(&mut c, Reg(5), 2.5);
            emit_binary(&mut c, OpKind::Add, X, X, Reg(5));
            check(run(&c, xs, ys, zs), |i| xs[i] + 2.5, "const+add");
        }

        #[test]
        fn emit_fmadd_c_in_dst_computes_the_fused_multiply_add() {
            let (xs, ys, zs) = lanes();
            // emit_fmadd_c_in_dst(dst, a, b): dst = a*b + dst.
            let mut c = Vec::new();
            emit_mov(&mut c, Reg(5), Z);
            emit_fmadd_c_in_dst(&mut c, Reg(5), X, Y);
            emit_mov(&mut c, X, Reg(5));
            check(run(&c, xs, ys, zs), |i| xs[i] * ys[i] + zs[i], "fma231");
        }

        /// The FMA bytes really are an FMA: **one** rounding, not a multiply
        /// followed by an add.
        ///
        /// `emit_fmadd_c_in_dst_computes_the_fused_multiply_add`'s 1e-3 tolerance cannot tell those apart — the whole
        /// difference is the last mantissa bit — so a stand-in built out of a
        /// multiply and an add would pass it. `1.0000001 * 4097 + 4097` is one
        /// of the inputs CLAUDE.md's `MulAdd` row is about, where the two
        /// forms genuinely disagree, and this asserts the bits.
        #[test]
        fn emit_fmadd_c_in_dst_rounds_once_not_twice() {
            let xs = [1.000_000_1f32; 16];
            let ys = [4097.0f32; 16];
            let zs = [4097.0f32; 16];
            let one = xs[0].mul_add(ys[0], zs[0]);
            // `black_box` stops LLVM contracting the reference into the very
            // instruction it exists to be different from.
            let two = core::hint::black_box(xs[0] * ys[0]) + zs[0];
            assert_ne!(
                one.to_bits(),
                two.to_bits(),
                "this input no longer separates one rounding from two"
            );

            let mut c = Vec::new();
            emit_mov(&mut c, Reg(5), Z);
            emit_fmadd_c_in_dst(&mut c, Reg(5), X, Y);
            emit_mov(&mut c, X, Reg(5));
            for (i, &g) in run(&c, xs, ys, zs).iter().enumerate() {
                assert_eq!(
                    g.to_bits(),
                    one.to_bits(),
                    "lane {i}: {g} rounded twice; the fused answer is {one}"
                );
            }
        }

        #[test]
        fn emit_gather_reads_the_value_at_each_lanes_index() {
            // JIT a function: fn(*const f32 base [rdi], __m512 idx_float [zmm0]) -> __m512
            // that truncates the float indices, sets the mask, and gathers
            // base[idx] per lane. Validates the VSIB vgatherdps bytes on hardware.
            // Passing __m512 by value IS the emitted ABI (SysV: zmm0-7), so
            // not-FFI-safe is a false positive here, as for `executable`'s aliases.
            #[allow(improper_ctypes_definitions)]
            type G = unsafe extern "C" fn(*const f32, __m512) -> __m512;

            let mut c = Vec::new();
            emit_cvttps2dq(&mut c, Reg(13), Reg(0)); // zmm13 = (i32) idx_float
            emit_set_gather_mask(&mut c); // k1 = 0xFFFF
            emit_gather(&mut c, Reg(14), 7, Reg(13)); // zmm14{k1} = [rdi + zmm13*4]
            emit_mov(&mut c, Reg(0), Reg(14)); // return in zmm0
            crate::emit::x86_64::ret(&mut c);

            let buf: Vec<f32> = (0..64).map(|i| (i as f32) * 1.5 + 0.25).collect();
            // Distinct per-lane indices, including repeats and the ends.
            let idx: [f32; 16] = [
                0.0, 63.0, 1.0, 2.0, 10.0, 10.0, 5.0, 32.0, 7.0, 8.0, 63.0, 0.0, 20.0, 21.0, 40.0,
                41.0,
            ];

            let exec = unsafe { ExecutableCode::from_code(&c).expect("mmap") };
            let out = unsafe {
                let f: G = exec.as_fn();
                let r = f(buf.as_ptr(), _mm512_loadu_ps(idx.as_ptr()));
                let mut out = [0.0f32; 16];
                _mm512_storeu_ps(out.as_mut_ptr(), r);
                out
            };

            for i in 0..16 {
                let want = buf[idx[i] as usize];
                assert_eq!(out[i], want, "gather lane {i}: idx {}", idx[i]);
            }
        }

        #[test]
        fn emit_gather_addresses_high_numbered_vector_registers_and_gpr_base() {
            // The production driver always gathers through rax (base_gpr=0)
            // with dst/idx below zmm16 in every kernel this test suite
            // compiles, so `emit_gather_reads_the_value_at_each_lanes_index`
            // never sets the R'/B/V' extension bits this emitter also has to
            // encode. Move the base pointer into r9 (>= r8) and gather
            // into/from zmm registers >= 16 to pin them, mirroring
            // `emit_binary_writes_a_high_numbered_register`'s zmm20 case.
            #[allow(improper_ctypes_definitions)]
            type G = unsafe extern "C" fn(*const f32, __m512) -> __m512;

            let mut c = Vec::new();
            x86_64::mov(&mut c, Gpr(9), x86_64::gpr::RDI);
            emit_cvttps2dq(&mut c, Reg(21), Reg(0)); // zmm21 = (i32) idx_float
            emit_set_gather_mask(&mut c);
            emit_gather(&mut c, Reg(20), 9, Reg(21)); // zmm20{k1} = [r9 + zmm21*4]
            emit_mov(&mut c, Reg(0), Reg(20));
            crate::emit::x86_64::ret(&mut c);

            let buf: Vec<f32> = (0..64).map(|i| (i as f32) * 1.5 + 0.25).collect();
            let idx: [f32; 16] = [
                0.0, 63.0, 1.0, 2.0, 10.0, 10.0, 5.0, 32.0, 7.0, 8.0, 63.0, 0.0, 20.0, 21.0, 40.0,
                41.0,
            ];

            let exec = unsafe { ExecutableCode::from_code(&c).expect("mmap") };
            let out = unsafe {
                let f: G = exec.as_fn();
                let r = f(buf.as_ptr(), _mm512_loadu_ps(idx.as_ptr()));
                let mut out = [0.0f32; 16];
                _mm512_storeu_ps(out.as_mut_ptr(), r);
                out
            };

            for i in 0..16 {
                let want = buf[idx[i] as usize];
                assert_eq!(out[i], want, "gather lane {i}: idx {}", idx[i]);
            }
        }

        #[test]
        fn emit_load_after_emit_store_recovers_the_spilled_value() {
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            AsmProgram::from([crate::emit::x86_64::Inst::SubImm32 {
                dst: crate::emit::x86_64::gpr::RSP,
                imm: crate::emit::x86_64::Imm32(64),
            }])
            .assemble(&mut c);
            emit_binary(&mut c, OpKind::Mul, Reg(6), X, Y);
            AsmProgram::from([Evex::m0f(0x11).rm(6, frame_slot(0))]).assemble(&mut c);
            emit_binary(&mut c, OpKind::Add, Reg(6), X, X); // clobber
            AsmProgram::from([Evex::m0f(0x10).rm(X.0, frame_slot(0))]).assemble(&mut c);
            AsmProgram::from([crate::emit::x86_64::Inst::AddImm32 {
                dst: crate::emit::x86_64::gpr::RSP,
                imm: crate::emit::x86_64::Imm32(64),
            }])
            .assemble(&mut c);
            check(run(&c, xs, ys, zs), |i| xs[i] * ys[i], "spill roundtrip");
        }
    }
}

// =============================================================================
// The AVX-512 `IsaBackend` driver
// =============================================================================

/// The AVX-512 half of code generation.
///
/// **This file is where AVX-512-specific bugs live, and the only place they
/// can.** Emission is a pure function into `Vec<u8>`, so everything here
/// compiles, typechecks and is swept for op coverage on every host, whatever
/// CPU it has. Only [`Native`](super::super::Native) decides which backend a
/// build instantiates, and only [`executable`](super::super::executable) needs
/// the matching hardware.
///
/// The consequence worth stating: a change that does not touch an ISA file
/// cannot introduce a platform-specific bug. That is the bargain `unsafe`
/// makes — confine what cannot be checked, so the rest is checked by
/// construction.
///
/// Dead only in a build that selected a *different* `Native`. The condition
/// mirrors this backend's `Native` alias, so a genuinely unused item in the
/// backend this build actually compiles still trips `dead_code`; an
/// unconditional allow here would hide it from CI's `clippy -D warnings`.
#[cfg_attr(
    not(all(target_arch = "x86_64", target_feature = "avx512f")),
    allow(dead_code)
)]
pub(crate) mod driver {
    use super::super::*;
    use super::{AsmProgram, Evex, Mem, NoDisp, UNUSED_VVVV, frame_slot};
    use crate::emit::x86_64 as x86;
    use crate::emit::x86_64::driver::SSE2_FILE;
    use crate::error::CompileError;
    use alloc::vec::Vec;
    use pixelflow_ir::kind::OpKind;

    /// The AVX-512 register file (zmm, 512-bit).
    ///
    /// Identical register *roles* to SSE2 — the shared driver depends on that —
    /// at four times the width, so only `vector_bytes` differs.
    const AVX512_FILE: regalloc::RegisterFile = regalloc::RegisterFile {
        // zmm4-31: twenty-eight of thirty-two, which is every register the ABI
        // does not use for an argument. The pool was *six* when this work
        // started, because a contiguous range could not reach past the reload
        // pair and the gather's scratch — sixteen registers were untouched by
        // anything at all.
        scratch: regalloc::RegSet::range(4, 28),
        // Nothing. Every register this backend's encodings destroy is now a
        // per-instruction reservation: zmm15 for a sign-flip's mask, zmm14 and
        // zmm16 for the gather's destination and truncated indices. All three
        // are in the pool above, borrowed only across the one instruction that
        // needs them. The select needs none — `vpternlogd` consumes its three
        // operands.
        fixed: &[],
        temps_for: super::temps_for,
        vector_bytes: 64,
        ..SSE2_FILE
    }
    .checked();

    /// AVX-512 implementation of the shared driver's leaf operations.
    pub(crate) struct Avx512Backend {
        file: regalloc::RegisterFile,
    }

    impl Avx512Backend {
        pub(crate) fn new(ctx: EmitCtx) -> Self {
            Self {
                file: AVX512_FILE.capped(ctx.max_regs),
            }
        }

        fn reload(code: &mut Vec<u8>, reload: &Reload) {
            match reload {
                Reload::FromStack { target, slot } => {
                    AsmProgram::from([Evex::m0f(0x10).rm(target.0, frame_slot(slot.offset()))])
                        .assemble(code);
                }
                Reload::Const { target, val_bits } => {
                    super::emit_const(code, *target, f32::from_bits(*val_bits));
                }
            }
        }
    }

    impl IsaBackend for Avx512Backend {
        type Branch = usize;

        fn register_file(&self) -> regalloc::RegisterFile {
            self.file
        }

        fn begin(&mut self, _schedule: &[regalloc::Def]) -> Result<(), CompileError> {
            Ok(()) // const broadcast is self-contained; no pool.
        }

        fn emit_plan(
            &mut self,
            code: &mut Vec<u8>,
            plan: &InstructionPlan,
        ) -> Result<(), CompileError> {
            for r in &plan.reloads {
                Self::reload(code, r);
            }
            if let Some((dst, src)) = plan.setup_mov
                && dst != src
            {
                AsmProgram::from([Evex::m0f(0x28).rrr(dst.0, UNUSED_VVVV, src.0)]).assemble(code);
            }
            match &plan.op {
                ResolvedOp::Nop => {}
                ResolvedOp::LoadConst { dst, val_bits } => {
                    super::emit_const(code, *dst, f32::from_bits(*val_bits));
                }
                ResolvedOp::Unary { op, dst, src } => {
                    super::emit_unary(code, *op, *dst, *src, plan.scratch.temp(0));
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
                    // dst = buffer[slot][idx]. The context pointer (array of buffer
                    // base pointers) is caller-provided in rdi; arithmetic/const emit
                    // never touches rdi, so it survives to here. Both are
                    // declared in AVX512_FILE.fixed, so `RegisterFile::checked`
                    // proves they miss the pool, the reload pair and
                    // the allocator's `arm_reload`.
                    let idx_int = crate::emit::declared_temp(plan.scratch.temp(0));
                    let gather_dst = crate::emit::declared_temp(plan.scratch.temp(1));
                    const RAX: u8 = 0;
                    const RDI: u8 = 7;
                    AsmProgram::from([
                        Evex::m0f_f3(0x5B).rrr(idx_int.0, UNUSED_VVVV, idx.0),
                        EncodedInst::from_slice(&[0xB8, 0xFF, 0xFF, 0x00, 0x00]),
                        EncodedInst::from_slice(&[0xC5, 0xF8, 0x92, 0xC8]),
                        x86::MovLoadPtr {
                            dst: PtrReg(RAX),
                            base: PtrReg(RDI),
                            disp: (*slot as i32) * 8,
                        }
                        .encode(),
                        super::gather(gather_dst, RAX, idx_int),
                    ])
                    .assemble(code);
                    if *dst != gather_dst {
                        AsmProgram::from([Evex::m0f(0x28).rrr(dst.0, UNUSED_VVVV, gather_dst.0)])
                            .assemble(code);
                    }
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
                    // EVEX 3-operand: no two-operand hazard, emit directly.
                    // Comparisons produce a vector mask (vcmpps -> vpmovm2d).
                    if super::is_compare(*op) {
                        super::emit_compare(code, *op, *dst, *left, *right);
                    } else {
                        super::emit_binary(code, *op, *dst, *left, *right);
                    }
                }
                ResolvedOp::FusedMulAdd { dst, a, b } => {
                    // dst holds c (setup_mov); real FMA231: dst = a*b + dst.
                    super::emit_fmadd_c_in_dst(code, *dst, *a, *b);
                }
                ResolvedOp::DecomposedMulAdd {
                    dst,
                    a,
                    b,
                    c,
                    c_deferred,
                } => {
                    // dst = a*b, reload c (after the multiply if deferred), dst += c.
                    super::emit_binary(code, OpKind::Mul, *dst, *a, *b);
                    match c_deferred {
                        Some(DeferredReload::FromStack(slot)) => {
                            AsmProgram::from([Evex::m0f(0x10).rm(c.0, frame_slot(slot.offset()))])
                                .assemble(code);
                        }
                        Some(DeferredReload::Const(bits)) => {
                            super::emit_const(code, *c, f32::from_bits(*bits));
                        }
                        None => {}
                    }
                    super::emit_binary(code, OpKind::Add, *dst, *dst, *c);
                }
                ResolvedOp::Select {
                    dst,
                    if_true,
                    if_false,
                } => {
                    // setup_mov already placed the vector mask in dst; one vpternlogd.
                    AsmProgram::from([Evex::m0f3a_66(0x25)
                        .imm(0xCA)
                        .rrr(dst.0, if_true.0, if_false.0)])
                    .assemble(code);
                }
            }
            Ok(())
        }

        fn emit_mov(&mut self, code: &mut Vec<u8>, dst: Reg, src: Reg) {
            if dst != src {
                AsmProgram::from([Evex::m0f(0x28).rrr(dst.0, UNUSED_VVVV, src.0)]).assemble(code);
            }
        }

        fn emit_store(
            &mut self,
            code: &mut Vec<u8>,
            src: Reg,
            offset: u32,
        ) -> Result<(), CompileError> {
            AsmProgram::from([Evex::m0f(0x11).rm(src.0, frame_slot(offset))]).assemble(code);
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
                    super::emit_const(code, target, f32::from_bits(bits));
                    target
                }
                Binding::Loc(Loc::Slot(slot)) => {
                    AsmProgram::from([Evex::m0f(0x10).rm(target.0, frame_slot(slot.offset()))])
                        .assemble(code);
                    target
                }
            }
        }

        // Select short-circuit guards: reduce the vector mask to flags (vptestmd +
        // kortestw) and branch. jz = all-false (skip true arm); jc = all-true (skip
        // false arm). Mirrors the SSE2 MOVMSKPS guards, k-register-based.
        /// `_scratch` is unused: this tier's guard reduces the mask with
        /// `movmskps`/`kortest` into the flags, needing no vector register.
        fn emit_skip_if_all_false(
            &mut self,
            code: &mut Vec<u8>,
            mask_reg: Reg,
            _scratch: Option<Reg>,
        ) -> usize {
            super::emit_mask_flags(code, mask_reg);
            x86_64::je(code).field() // ZF set when k1 == 0 (all false)
        }
        /// `_scratch` is unused: this tier's guard reduces the mask with
        /// `movmskps`/`kortest` into the flags, needing no vector register.
        fn emit_skip_if_all_true(
            &mut self,
            code: &mut Vec<u8>,
            mask_reg: Reg,
            _scratch: Option<Reg>,
        ) -> usize {
            super::emit_mask_flags(code, mask_reg);
            x86_64::jc(code).field() // CF set when k1 == 0xFFFF (all true)
        }
        fn emit_jump(&mut self, code: &mut Vec<u8>) -> usize {
            x86_64::emit_jmp_rel32(code)
        }
        fn patch_branch(&mut self, code: &mut Vec<u8>, branch: usize, target: usize) {
            x86_64::patch_rel32(code, branch, target);
        }

        // Same scaffold register roles as SSE2 — see `x86_64::scaffold` — at
        // this vector width. Unlike SSE2 there is no red-zone mode: the body
        // always spills into an allocated frame, and the scaffold's coordinate
        // slots sit above it.

        fn frame_alloc(&mut self, code: &mut Vec<u8>, bytes: u32) {
            AsmProgram::from([x86::Inst::SubImm32 {
                dst: x86::gpr::RSP,
                imm: x86::Imm32(bytes as i32),
            }])
            .assemble(code);
        }

        fn frame_free(&mut self, code: &mut Vec<u8>, bytes: u32) {
            AsmProgram::from([x86::Inst::AddImm32 {
                dst: x86::gpr::RSP,
                imm: x86::Imm32(bytes as i32),
            }])
            .assemble(code);
        }

        fn slot_store(&mut self, code: &mut Vec<u8>, src: Reg, offset: u32) {
            AsmProgram::from([Evex::m0f(0x11).rm(src.0, frame_slot(offset))]).assemble(code);
        }

        fn slot_load(&mut self, code: &mut Vec<u8>, dst: Reg, offset: u32) {
            AsmProgram::from([Evex::m0f(0x10).rm(dst.0, frame_slot(offset))]).assemble(code);
        }

        fn latch_bounds(&mut self, code: &mut Vec<u8>) {
            x86::scaffold::latch_bounds(code);
        }

        fn counter_clear(&mut self, code: &mut Vec<u8>, counter: Counter) {
            x86::scaffold::counter_clear(code, counter);
        }

        fn counter_step(&mut self, code: &mut Vec<u8>, counter: Counter) {
            x86::scaffold::counter_step(code, counter);
        }

        fn branch_if_counter_done(&mut self, code: &mut Vec<u8>, counter: Counter) -> usize {
            x86::scaffold::branch_if_counter_done(code, counter)
        }

        fn store_result(&mut self, code: &mut Vec<u8>, src: Reg) {
            AsmProgram::from([Evex::m0f(0x11).rm(
                src.0,
                Mem {
                    base: x86::scaffold::OUT_PTR,
                    disp: NoDisp,
                },
            )])
            .assemble(code);
        }

        fn advance_out(&mut self, code: &mut Vec<u8>, step: OutStep) {
            x86::scaffold::advance_out(code, step, self.file.vector_bytes);
        }

        fn add_scalar(&mut self, code: &mut Vec<u8>, dst: Reg, scratch: Reg, scalar: f32) {
            super::emit_const(code, scratch, scalar);
            super::emit_binary(code, OpKind::Add, dst, dst, scratch);
        }

        fn emit_ret(&mut self, code: &mut Vec<u8>) {
            AsmProgram::from([x86::Inst::Ret]).assemble(code);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// `AVX512_FILE`'s field values are restated (not delegated to
        /// `..SSE2_FILE`) because they genuinely differ — twenty-eight
        /// registers and this backend's own `temps_for` rather than SSE2's
        /// twelve and its select-reload-aware one — so nothing catches a
        /// regression back to the SSE2 shape except a direct assertion.
        #[test]
        fn avx512_file_reserves_zmm4_through_27_with_no_fixed_registers() {
            assert_eq!(AVX512_FILE.scratch, regalloc::RegSet::range(4, 28));
            assert!(AVX512_FILE.fixed.is_empty());
            assert_eq!(
                AVX512_FILE.temps_for as *const () as usize,
                super::super::temps_for as *const () as usize
            );
        }
    }
}
