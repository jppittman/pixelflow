//! x86-64 AVX2 (VEX.256) JIT encoder — 256-bit, 8-lane `ymm` kernels.
//!
//! The middle width between the SSE2 leaf encoders (`x86_64.rs`, 128-bit) and
//! the AVX-512 EVEX encoders (`avx512.rs`, 512-bit). Register numbering is
//! identical to SSE2 (ymm0-15, no extended file — AVX2 has no REX2/EVEX), so
//! this backend reuses the register file `X86Backend` declares and only the
//! instruction *encoding* changes.
//!
//! Unlike legacy SSE2, VEX is 3-operand and non-destructive — same property
//! AVX-512's EVEX has — so there is no two-operand hazard to route around
//! (the SSE2 tier's two-operand form has no such freedom).
//! Comparisons are simpler here than on AVX-512: `vcmpps` writes an ordinary
//! all-ones/all-zeros `ymm` directly (no k-register, no mask-to-vector
//! conversion) — the same representation NEON and SSE2 already use.
//!
//! Spills use a real stack frame, not the red zone: mirrors `avx512.rs`'s
//! reasoning (a `ymm` slot is 32 bytes; keeping the red-zone arithmetic exact
//! for two different slot widths is not worth it for a bit of frame reuse on
//! tiny kernels).
//!
//! Gather has no direct AVX2 hardware analogue reused here: `vgatherdps`'s
//! VSIB + vector-mask-with-clearing semantics are a bigger lift than this
//! backend's scope warrants, so — like `X86Backend` — a gather is assembled
//! from scalar loads via the existing 128-bit lane-insert sequence
//! (`x86_64::emit_gather_scalar`), run once per 128-bit half and combined
//! with `vinsertf128`.

use super::x86_64;
use super::x86_64::{Disp, Imm32, Mem, NoDisp, ptr};
use super::{AsmProgram, EncodedInst, Gpr, PtrReg, Reg, SourceOperand, assemble, unimplemented_op};
use alloc::vec::Vec;
use pixelflow_ir::OpKind;

// The AVX2 tier requires FMA3, and `crate::isa` refuses a host without it:
// AVX2-without-FMA is not a narrower tier but a paper configuration, and the
// probe's doc says why (no shipping CPU has one without the other, and the
// two-rounding fork it once forced on `emit_fmadd_c_in_dst` put two
// materially different kernels under one environment fingerprint).

// =============================================================================
// VEX.256 encoder
// =============================================================================

/// Which legacy-prefix byte the VEX prefix implies (the `pp` field).
#[derive(Clone, Copy)]
enum Pp {
    /// No implied prefix.
    None = 0,
    /// `66`
    P66 = 1,
    /// `F3`
    F3 = 2,
}

/// Which opcode map the instruction lives in (the field Intel calls
/// `mmmmm` — a map *selector*, nothing more).
#[derive(Clone, Copy)]
enum Map {
    /// `0F`
    M0F = 1,
    /// `0F38`
    M0F38 = 2,
    /// `0F3A`
    M0F3A = 3,
}

/// The identity of one VEX instruction: opcode map, implied legacy prefix, W
/// bit, opcode byte, and whether it is the 256-bit form. This is *which
/// instruction* — it is constant per mnemonic, so each mnemonic below states
/// it exactly once and the operand form (`rrr`/`imm`/`rm`) supplies the
/// per-call parts.
#[derive(Clone, Copy)]
struct Vex {
    map: Map,
    pp: Pp,
    w: bool,
    /// `VEX.L`: set for the `ymm` form, clear for the few `xmm`-only
    /// instructions this tier needs (`vmovq`, `vextractps`), which `#UD` at
    /// `L = 1`.
    l256: bool,
    opcode: u8,
}

/// Sentinel for an unused VEX.vvvv source (2-operand forms): index 0 inverts
/// to `1111`, the required "unused" encoding.
const UNUSED_VVVV: u8 = 0;

impl Vex {
    const fn new(map: Map, pp: Pp, opcode: u8) -> Self {
        Self {
            map,
            pp,
            w: false,
            l256: true,
            opcode,
        }
    }
    /// The 128-bit (`VEX.L = 0`) form of this instruction.
    const fn xmm(self) -> Self {
        Self {
            l256: false,
            ..self
        }
    }
    /// The 64-bit-operand (`VEX.W = 1`) form of this instruction.
    const fn w1(self) -> Self {
        Self { w: true, ..self }
    }
    /// Map `0F`, no prefix — the packed-single arithmetic family.
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
    /// Map `0F3A`, `66` — the imm8 family (round, insert/extract).
    const fn m0f3a_66(opcode: u8) -> Self {
        Self::new(Map::M0F3A, Pp::P66, opcode)
    }

    /// Attach an imm8 (`vcmpps` predicate, rounding mode, shift count, lane
    /// index); the returned value emits it after the instruction.
    const fn imm(self, imm: u8) -> VexImm {
        VexImm { vex: self, imm }
    }

    /// Register-register-register form: `op dst, vvvv, rm`.
    fn rrr(self, dst: u8, vvvv: u8, rm: u8) -> EncodedInst {
        let mut inst = EncodedInst::new();
        let rbit = if dst >= 8 { 0x00 } else { 0x80 };
        let xbit = 0x40;
        let bbit = if rm >= 8 { 0x00 } else { 0x20 };
        inst.push(0xC4);
        inst.push(rbit | xbit | bbit | self.map as u8);
        inst.push(
            ((self.w as u8) << 7) | ((!vvvv & 0xF) << 3) | ((self.l256 as u8) << 2) | self.pp as u8,
        );
        inst.push(self.opcode);
        inst.push(0xC0 | ((dst & 7) << 3) | (rm & 7));
        inst
    }

    /// `op reg, [addr]` — the memory-operand form, for any base and any
    /// displacement mode. The prefix is VEX's; the ModRM/SIB/displacement tail
    /// is the architecture's, so it comes from `x86_64::mem_operand`.
    fn rm<D: Disp>(self, reg: u8, addr: Mem<D>) -> EncodedInst {
        let mut inst = EncodedInst::new();
        // R and B are stored inverted; X is unused (no index register).
        let rbit = if reg >= 8 { 0x00 } else { 0x80 };
        let bbit = if addr.base.0 >= 8 { 0x00 } else { 0x20 };
        inst.push(0xC4);
        inst.push(rbit | 0x40 | bbit | self.map as u8);
        inst.push(((self.w as u8) << 7) | (0xF << 3) | ((self.l256 as u8) << 2) | self.pp as u8); // vvvv unused
        inst.push(self.opcode);
        x86_64::mem_operand_into(&mut inst, reg, addr);
        inst
    }

    /// `op reg, [base + index*4]` — the SIB form with a scaled index, which
    /// a broadcast load reads one element of a plane through. X carries the
    /// index's high bit, inverted like R and B.
    fn rm_scaled4(self, reg: u8, base: Gpr, index: Gpr) -> EncodedInst {
        let mut inst = EncodedInst::new();
        let rbit = if reg >= 8 { 0x00 } else { 0x80 };
        let xbit = if index.0 >= 8 { 0x00 } else { 0x40 };
        let bbit = if base.0 >= 8 { 0x00 } else { 0x20 };
        inst.push(0xC4);
        inst.push(rbit | xbit | bbit | self.map as u8);
        inst.push(((self.w as u8) << 7) | (0xF << 3) | ((self.l256 as u8) << 2) | self.pp as u8); // vvvv unused
        inst.push(self.opcode);
        x86_64::scaled4_operand_into(&mut inst, reg, base, index);
        inst
    }

    /// `op dst, vvvv, [addr]` — 3-operand VEX.256 with memory operand.
    #[allow(dead_code)]
    fn rrm<D: Disp>(self, dst: u8, vvvv: u8, addr: Mem<D>) -> EncodedInst {
        let mut inst = EncodedInst::new();
        let rbit = if dst >= 8 { 0x00 } else { 0x80 };
        let bbit = if addr.base.0 >= 8 { 0x00 } else { 0x20 };
        inst.push(0xC4);
        inst.push(rbit | 0x40 | bbit | self.map as u8);
        inst.push(
            ((self.w as u8) << 7) | ((!vvvv & 0xF) << 3) | ((self.l256 as u8) << 2) | self.pp as u8,
        );
        inst.push(self.opcode);
        x86_64::mem_operand_into(&mut inst, dst, addr);
        inst
    }

    /// Generic 3-operand form: `op dst, vvvv, rm` where `rm` can be a register or stack slot.
    #[allow(dead_code)]
    fn rro<S: SourceOperand>(self, dst: u8, vvvv: u8, rm: S) -> Option<EncodedInst> {
        if let Some(r) = rm.source_reg() {
            return Some(self.rrr(dst, vvvv, r.0));
        }
        let slot = rm.source_slot()?;
        Some(self.rrm(
            dst,
            vvvv,
            Mem {
                base: ptr::RSP,
                disp: Imm32(slot.offset() as i32),
            },
        ))
    }
}

/// A [`Vex`] instruction carrying its imm8.
#[derive(Clone, Copy)]
struct VexImm {
    vex: Vex,
    imm: u8,
}

impl VexImm {
    /// Register form with the imm8 appended.
    fn rrr(self, dst: u8, vvvv: u8, rm: u8) -> EncodedInst {
        let mut inst = self.vex.rrr(dst, vvvv, rm);
        inst.push(self.imm);
        inst
    }
    /// Memory form with the imm8 appended.
    fn rm<D: Disp>(self, reg: u8, addr: Mem<D>) -> EncodedInst {
        let mut inst = self.vex.rm(reg, addr);
        inst.push(self.imm);
        inst
    }
}

// --- packed-single arithmetic (0F, no prefix, W0) ---
fn vaddps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    assemble(c, [Vex::m0f(0x58).rrr(d, s1, s2)]);
}
fn vsubps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    assemble(c, [Vex::m0f(0x5C).rrr(d, s1, s2)]);
}
fn vmulps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    assemble(c, [Vex::m0f(0x59).rrr(d, s1, s2)]);
}
fn vdivps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    assemble(c, [Vex::m0f(0x5E).rrr(d, s1, s2)]);
}
fn vminps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    assemble(c, [Vex::m0f(0x5D).rrr(d, s1, s2)]);
}
fn vmaxps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    assemble(c, [Vex::m0f(0x5F).rrr(d, s1, s2)]);
}
fn vsqrtps(c: &mut Vec<u8>, d: u8, s: u8) {
    assemble(c, [Vex::m0f(0x51).rrr(d, UNUSED_VVVV, s)]);
}
fn vrsqrtps(c: &mut Vec<u8>, d: u8, s: u8) {
    assemble(c, [Vex::m0f(0x52).rrr(d, UNUSED_VVVV, s)]);
}
fn vrcpps(c: &mut Vec<u8>, d: u8, s: u8) {
    assemble(c, [Vex::m0f(0x53).rrr(d, UNUSED_VVVV, s)]);
}

// --- bitwise (0F, no prefix, W0) ---
fn vandps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    assemble(c, [Vex::m0f(0x54).rrr(d, s1, s2)]);
}
fn vandnps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    assemble(c, [Vex::m0f(0x55).rrr(d, s1, s2)]);
}
fn vorps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    assemble(c, [Vex::m0f(0x56).rrr(d, s1, s2)]);
}
fn vxorps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    assemble(c, [Vex::m0f(0x57).rrr(d, s1, s2)]);
}

// --- comparisons (0F, no prefix, W0; imm8 predicate) ---
const CMP_EQ: u8 = 0;
const CMP_LT: u8 = 1;
const CMP_LE: u8 = 2;
const CMP_NEQ: u8 = 4;
const CMP_GE: u8 = 5;
const CMP_NLE: u8 = 6; // > (unordered-safe "not less-or-equal")

fn vcmpps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8, pred: u8) {
    assemble(c, [Vex::m0f(0xC2).imm(pred).rrr(d, s1, s2)]);
}

fn cmp_pred(op: OpKind) -> Option<u8> {
    Some(match op {
        OpKind::Eq => CMP_EQ,
        OpKind::Ne => CMP_NEQ,
        OpKind::Lt => CMP_LT,
        OpKind::Le => CMP_LE,
        OpKind::Gt => CMP_NLE,
        OpKind::Ge => CMP_GE,
        _ => return None,
    })
}

/// Whether `op` is a comparison handled by [`emit_binary`].
#[must_use]
pub fn is_compare(op: OpKind) -> bool {
    cmp_pred(op).is_some()
}

// --- rounding (0F3A, 66 prefix, W0; imm8) ---
fn vroundps(c: &mut Vec<u8>, d: u8, s: u8, imm: u8) {
    assemble(c, [Vex::m0f3a_66(0x08).imm(imm).rrr(d, UNUSED_VVVV, s)]);
}

// --- int/float convert (0F, W0) ---
fn vcvttps2dq(c: &mut Vec<u8>, d: u8, s: u8) {
    assemble(c, [Vex::m0f_f3(0x5B).rrr(d, UNUSED_VVVV, s)]); // F3 prefix
}
fn vcvtdq2ps(c: &mut Vec<u8>, d: u8, s: u8) {
    assemble(c, [Vex::m0f(0x5B).rrr(d, UNUSED_VVVV, s)]); // no prefix
}

// --- integer-domain (66 prefix, 0F, W0) ---
fn vpaddd(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    assemble(c, [Vex::m0f_66(0xFE).rrr(d, s1, s2)]);
}
fn vpslld_imm(c: &mut Vec<u8>, d: u8, s: u8, imm: u8) {
    assemble(c, [Vex::m0f_66(0x72).imm(imm).rrr(6, d, s)]); // /6, dst=vvvv, src=rm
}
fn vpsrld_imm(c: &mut Vec<u8>, d: u8, s: u8, imm: u8) {
    assemble(c, [Vex::m0f_66(0x72).imm(imm).rrr(2, d, s)]); // /2
}

// --- lane insert/extract between 256-bit and 128-bit (0F3A, 66 prefix, W0) ---
/// `vinsertf128 ymmDST, ymmSRC1, xmmSRC2, imm8[0]` — copy `src1`, then place
/// `src2` into the low (`imm=0`) or high (`imm=1`) 128 bits.
fn vinsertf128(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8, imm: u8) {
    assemble(c, [Vex::m0f3a_66(0x18).imm(imm).rrr(d, s1, s2)]);
}
/// `vextractf128 xmmDST, ymmSRC, imm8[0]` — extract the low (`imm=0`) or high
/// (`imm=1`) 128 bits of `src` into `dst`.
fn vextractf128(c: &mut Vec<u8>, d: u8, s: u8, imm: u8) {
    // VEX.256.66.0F3A.W0 19 /r ib — note dst is the ModRM.rm operand here
    // (the reverse of the usual direction: register source, register/mem dest).
    assemble(c, [Vex::m0f3a_66(0x19).imm(imm).rrr(s, UNUSED_VVVV, d)]);
}

/// `vmovaps ymmDST, ymmSRC` — register copy.
pub fn emit_mov(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    if dst.0 == src.0 {
        return;
    }
    assemble(code, [Vex::m0f(0x28).rrr(dst.0, UNUSED_VVVV, src.0)]);
}

/// A slot in the allocated spill frame. AVX2 kernels are leaves with no base
/// pointer, so a slot *is* `rsp + offset`.
const fn frame_slot(offset: u32) -> Mem<Imm32> {
    Mem {
        base: ptr::RSP,
        disp: Imm32(offset as i32),
    }
}

/// `dst = splat(val)`: `vbroadcastss ymm, [pool]` (VEX.256.66.0F38.W0 18 /r),
/// one instruction from the kernel's constant pool. Zero is `vxorps`.
pub fn emit_const(code: &mut Vec<u8>, dst: Reg, val: f32, pool: &mut x86_64::ConstPool) {
    let bits = val.to_bits();
    if bits == 0 {
        vxorps(code, dst.0, dst.0, dst.0);
        return;
    }
    assemble(code, [Vex::m0f38_66(0x18).rm(dst.0, pool.operand(bits))]);
}

/// `dst = splat(base[offset])` at 256 bits: `vbroadcastss ymm<dst>, [base +
/// 4*offset]` (VEX.256.66.0F38.W0 18 /r). See `x86_64::emit_uniform_load`.
pub fn emit_uniform_load(code: &mut Vec<u8>, dst: Reg, base: PtrReg, offset: u16) {
    AsmProgram::from([Vex::m0f38_66(0x18).rm(
        dst.0,
        Mem {
            base,
            disp: Imm32(i32::from(offset) * 4),
        },
    )])
    .assemble(code);
}

/// `dst = splat(base[idx])` at 256 bits, the index being the same in every
/// lane of `idx`: `vcvttss2si index, xmm<idx>`, `vbroadcastss ymm<dst>,
/// [base + index*4]` (VEX.256.66.0F38.W0 18 /r). See
/// `x86_64::emit_broadcast_load` for the register contract.
pub fn emit_broadcast_load(code: &mut Vec<u8>, dst: Reg, idx: Reg, gprs: x86_64::BroadcastGprs) {
    AsmProgram::from([
        vcvttss2si_xmm(gprs.index, idx),
        Vex::m0f38_66(0x18).rm_scaled4(dst.0, gprs.base.as_gpr(), gprs.index),
    ])
    .assemble(code);
}

// =============================================================================
// Stack frame (real frame; a ymm spill is 32 bytes)
// =============================================================================

// =============================================================================
// Op dispatch
// =============================================================================

/// `dst = op(src1, src2)`. VEX is 3-operand/non-destructive: operands are
/// never clobbered and may alias `dst`. Comparisons produce an ordinary
/// all-ones/all-zeros vector directly (no k-register step, unlike AVX-512).
pub fn emit_binary(code: &mut Vec<u8>, op: OpKind, dst: Reg, src1: Reg, src2: Reg) {
    let (d, s1, s2) = (dst.0, src1.0, src2.0);
    if let Some(pred) = cmp_pred(op) {
        vcmpps(code, d, s1, s2, pred);
        return;
    }
    match op {
        OpKind::Add => vaddps(code, d, s1, s2),
        OpKind::Sub => vsubps(code, d, s1, s2),
        OpKind::Mul => vmulps(code, d, s1, s2),
        OpKind::Div => vdivps(code, d, s1, s2),
        OpKind::Min => vminps(code, d, s1, s2),
        OpKind::Max => vmaxps(code, d, s1, s2),
        OpKind::BitAnd => vandps(code, d, s1, s2),
        OpKind::BitOr => vorps(code, d, s1, s2),
        OpKind::IAdd => vpaddd(code, d, s1, s2),
        _ => unimplemented_op("avx2", op),
    }
}

/// `dst = op(src)`.
///
/// The temp is the allocator's for this instruction; only `Neg` and `Abs`
/// use it, to hold the sign mask they XOR or AND with, which comes from the
/// kernel's constant pool like any other constant.
pub fn emit_unary(code: &mut Vec<u8>, unary: super::Unary, pool: &mut x86_64::ConstPool) {
    let super::Unary { op, dst, src, temp } = unary;
    match op {
        OpKind::Sqrt => vsqrtps(code, dst.0, src.0),
        OpKind::Rsqrt => vrsqrtps(code, dst.0, src.0),
        OpKind::Recip => vrcpps(code, dst.0, src.0),
        OpKind::Neg => {
            let mask = super::declared_temp(temp);
            emit_const(code, mask, f32::from_bits(0x8000_0000), pool);
            vxorps(code, dst.0, src.0, mask.0);
        }
        OpKind::Abs => {
            let mask = super::declared_temp(temp);
            emit_const(code, mask, f32::from_bits(0x7FFF_FFFF), pool);
            vandps(code, dst.0, src.0, mask.0);
        }
        // imm8: bits[3:0] = rounding mode (0=nearest, 1=floor, 2=ceil).
        OpKind::Floor => vroundps(code, dst.0, src.0, 0x01),
        OpKind::Ceil => vroundps(code, dst.0, src.0, 0x02),
        OpKind::Round => vroundps(code, dst.0, src.0, 0x00),
        OpKind::TruncToInt => vcvttps2dq(code, dst.0, src.0),
        OpKind::IntToFloat => vcvtdq2ps(code, dst.0, src.0),
        _ => unimplemented_op("avx2", op),
    }
}

/// How many registers this backend's encodings need beyond their operands.
///
/// `Neg`/`Abs` build a sign mask, and the select blends through a temporary;
/// every other encoding here is a single non-destructive VEX instruction.
pub(crate) fn temps_for(op: &super::ScheduledOp) -> u8 {
    use super::ScheduledOp;
    match op {
        ScheduledOp::Unary(OpKind::Neg | OpKind::Abs, _) => 1,
        ScheduledOp::Ternary(OpKind::Select, ..) => 1,
        // A 256-bit gather is two 128-bit halves: the half-sequence's own
        // index and value registers, plus one of each to carry the high half
        // while the low one is assembled in `dst`.
        ScheduledOp::Gather(..) => 4,
        // A surviving fold's own loop: two transient registers for the trip
        // test and the accumulate — see `emit_scope`'s `Reduce` arm. The
        // binder and the accumulator are the fold's roots, placed by the
        // allocator, not scratch.
        ScheduledOp::Reduce(..) => super::regalloc::Scratch::REDUCE_TEMPS as u8,
        // A remainder past the low half stores its upper lanes out of the
        // high 128 bits, extracted into one temp; one that fits the low half
        // reads them straight out of the value, and a full batch is one
        // `vmovups`.
        ScheduledOp::Write { lanes, .. } if *lanes > 4 && *lanes < 8 => 1,
        _ => 0,
    }
}

// =============================================================================
// The store, and the iota
// =============================================================================

/// `vcvttss2si r64, xmm` — `VEX.LIG.F3.0F.W1 2C /r`: lane 0, truncated to a
/// 64-bit integer.
#[must_use]
fn vcvttss2si_xmm(dst: Gpr, src: Reg) -> EncodedInst {
    Vex::m0f_f3(0x2C).w1().rrr(dst.0, UNUSED_VVVV, src.0)
}

/// `vcvttss2si r64, m32` — the same, reading the first word of a slot.
#[must_use]
fn vcvttss2si_mem<D: Disp>(dst: Gpr, addr: Mem<D>) -> EncodedInst {
    Vex::m0f_f3(0x2C).w1().rm(dst.0, addr)
}

/// `vmovq xmm, r64` — `VEX.128.66.0F.W1 6E /r`: eight bytes into the low
/// lanes, the rest zeroed.
#[must_use]
fn vmovq_xmm_r64(dst: Reg, src: Gpr) -> EncodedInst {
    Vex::m0f_66(0x6E).w1().xmm().rrr(dst.0, UNUSED_VVVV, src.0)
}

/// `vpmovzxbd ymm, xmm` — `VEX.256.66.0F38.WIG 31 /r`: eight bytes widened
/// to eight dword lanes.
#[must_use]
fn vpmovzxbd(dst: Reg, src: Reg) -> EncodedInst {
    Vex::m0f38_66(0x31).rrr(dst.0, UNUSED_VVVV, src.0)
}

/// `vextractps m32, xmm, lane` — `VEX.128.66.0F3A.WIG 17 /r ib`: one lane of
/// the low half, stored.
#[must_use]
fn vextractps_store<D: Disp>(addr: Mem<D>, src: Reg, lane: u8) -> EncodedInst {
    debug_assert!(lane < 4, "vextractps reads the low 128 bits");
    Vex::m0f3a_66(0x17).xmm().imm(lane).rm(src.0, addr)
}

/// The bytes `0..8`, little end first: what one `movabs` carries in for
/// `vpmovzxbd` to widen into the iota.
const IOTA_BYTES: u64 = 0x0706_0504_0302_0100;

/// Emit a shift of i32 lanes by a compile-time immediate.
pub fn emit_shift_imm(code: &mut Vec<u8>, op: OpKind, dst: Reg, src: Reg, amount: u8) {
    match op {
        OpKind::Shl => vpslld_imm(code, dst.0, src.0, amount),
        OpKind::Shr => vpsrld_imm(code, dst.0, src.0, amount),
        _ => unimplemented_op("avx2", op),
    }
}

/// `dst = mask ? if_true : if_false` (bit-select; mask already in `dst`, same
/// convention as SSE2/AVX-512).
///
/// `tmp` is the allocator's temp for this instruction, which it picks disjoint
/// from every operand — the `debug_assert` restates that here, where the
/// instruction would silently blend garbage if it ever failed.
pub fn emit_select(code: &mut Vec<u8>, dst: Reg, if_true: Reg, if_false: Reg, tmp: Option<Reg>) {
    let tmp = super::declared_temp(tmp);
    debug_assert!(tmp.0 != dst.0 && tmp.0 != if_true.0 && tmp.0 != if_false.0);
    vandps(code, tmp.0, dst.0, if_true.0); // tmp = mask & if_true
    vandnps(code, dst.0, dst.0, if_false.0); // dst = ~mask & if_false
    vorps(code, dst.0, tmp.0, dst.0); // dst = blended
}

/// `vmovmskps eax, ymmSRC` — gather the 8 lane sign bits into eax[7:0].
pub fn emit_movmskps_eax(code: &mut Vec<u8>, src: Reg) {
    assemble(code, [Vex::m0f(0x50).rrr(0, UNUSED_VVVV, src.0)]);
}

/// `cmp al, imm8` — unlike `cmp eax, imm8` (sign-extending `0x83`), this
/// compares the raw byte pattern, which is what an 8-lane all-true check
/// (`eax == 0xFF`) needs (`0x83`'s sign-extension would compare against
/// `0xFFFFFFFF`, which `vmovmskps`'s zero-extended result can never equal).
pub fn emit_cmp_al_imm8(code: &mut Vec<u8>, imm: u8) {
    code.push(0x3C);
    code.push(imm);
}

/// `vfmadd231ps ymmD, ymmA, ymmB` — `dst = a*b + dst` (231 form: dst is the
/// addend going in, `a`/`b` the product). VEX.256.66.0F38.W0 B8 /r, same
/// opcode as `avx512.rs`'s EVEX form, just VEX-encoded at 256 bits. FMA3 is
/// not implied by AVX2 in CPUID, which is why `crate::isa`'s x86-64 probe
/// asks for both before this backend can be selected.
fn vfmadd231ps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    assemble(c, [Vex::m0f38_66(0xB8).rrr(d, s1, s2)]);
}

/// Fused multiply-add: `dst` already holds `c`; computes `dst = a*b + dst`.
///
/// Always real hardware FMA: the AVX2 tier requires FMA3 (`crate::isa`
/// refuses a host without it), so there is no software mul+add fallback to
/// choose between here. This rounds once, as the folder does
/// (`libm::fmaf`); a software two-step mul-then-add would round twice and
/// disagree in the last bit.
///
/// The two-roundings case still exists, just not in *this* function: it is
/// what `DecomposedMulAdd` does on every tier, this one included, whenever
/// register pressure pulls `a` and `b` apart from `c`. Both are pinned as
/// bytes by `emit::tests::muladd_encoding` and as values by
/// `tests/muladd_rounding.rs`.
pub fn emit_fmadd_c_in_dst(code: &mut Vec<u8>, dst: Reg, a: Reg, b: Reg) {
    vfmadd231ps(code, dst.0, a.0, b.0);
}

// =============================================================================
// Bound-memory gather (RawGather lowering target)
//
// No native vgatherdps here (see the module doc): truncate all 8 lanes at
// once, split into two 128-bit halves, run the existing SSE2/AVX scalar-load
// sequence (`x86_64::emit_gather_scalar`) on each half (it only ever touches
// the low 128 bits of whatever register it's given — ymm0's low 128 IS
// xmm0), then recombine with vinsertf128.
// =============================================================================

/// Scratch the 256-bit gather clobbers: the 128-bit sequence's own scratch,
/// which both halves reuse, plus the two vector registers that carry the high
/// half while the low half is being assembled. All of it must be distinct from
/// the gather's `dst` and `idx`.
#[derive(Clone, Copy)]
pub struct GatherScratch {
    /// Scratch for one 128-bit half — see [`x86_64::GatherScratch`].
    pub half: x86_64::GatherScratch,
    /// Vector register receiving lanes 4..8 of the float indices.
    pub idx_hi: Reg,
    /// Vector register receiving the high half's gathered values.
    pub res_hi: Reg,
}

/// `dst = base[idx_lane]` for 8 lanes. `idx` holds FLOAT indices (already
/// clamped in range by the `Gather` lowering — `x86_64::emit_gather_scalar`
/// does its own float->int truncation per half, so `idx` must not be
/// pre-truncated here); `base` the buffer's address. Clobbers everything in
/// `s`.
pub fn emit_gather_scalar(code: &mut Vec<u8>, dst: Reg, idx: Reg, base: PtrReg, s: GatherScratch) {
    // idx's low 128 already holds lanes 0..4 (float); split off lanes 4..8
    // into idx_hi before either gather call touches idx/dst (which may alias).
    vextractf128(code, s.idx_hi.0, idx.0, 1);

    // Low half: lanes 0..4. May write dst == idx (the callee handles that:
    // it converts idx to int in scratch before ever writing dst).
    x86_64::emit_gather_scalar(code, dst, idx, base, s.half);
    // High half: lanes 4..8, into res_hi (a 128-bit scratch distinct from dst).
    x86_64::emit_gather_scalar(code, s.res_hi, s.idx_hi, base, s.half);

    // Recombine: dst[0..4] already holds the low half; splice in the high.
    vinsertf128(code, dst.0, dst.0, s.res_hi.0, 1);
}

#[cfg(test)]
mod tests {
    //! Hardware validation, mirroring `avx512.rs`'s runtime test tier: JIT real
    //! `ymm` kernels and execute them on the host.
    use super::*;

    #[test]
    fn is_compare_is_true_only_for_the_six_ordered_comparison_ops() {
        for op in [
            OpKind::Eq,
            OpKind::Ne,
            OpKind::Lt,
            OpKind::Le,
            OpKind::Gt,
            OpKind::Ge,
        ] {
            assert!(is_compare(op), "{op:?} should be a compare");
        }
        for op in [
            OpKind::Add,
            OpKind::Sub,
            OpKind::Mul,
            OpKind::Div,
            OpKind::Min,
            OpKind::Max,
            OpKind::BitAnd,
            OpKind::BitOr,
            OpKind::IAdd,
        ] {
            assert!(!is_compare(op), "{op:?} should not be a compare");
        }
    }

    /// Executes the bytes on this host's CPU, so every test first asks
    /// whether it can (`skip_unless_host_runs!`); the encodings themselves
    /// are pinned bytewise on every host by the tests above. The `extern
    /// "C"` kernels take `ymm` values, which the ABI only lets a caller
    /// compiled with AVX pass — hence `#[target_feature]` on the functions
    /// that call them, and nowhere else.
    #[cfg(target_arch = "x86_64")]
    mod runtime {
        use super::super::*;
        use crate::emit::executable::ExecutableCode;
        use crate::isa::{Isa, skip_unless_host_runs};
        use core::arch::x86_64::*;

        #[allow(improper_ctypes_definitions)]
        type K = unsafe extern "C" fn(__m256, __m256, __m256, __m256) -> __m256;

        fn run(body: &[u8], xs: [f32; 8], ys: [f32; 8], zs: [f32; 8]) -> [f32; 8] {
            let mut code = body.to_vec();
            crate::emit::x86_64::ret(&mut code);
            // SAFETY: every caller is a test that checked the host runs AVX2.
            unsafe { run_code(&code, xs, ys, zs) }
        }

        /// `run`, for a body that read constants from `pool`: the anchor
        /// ahead of it and the pool behind its `ret`, as the driver lays a
        /// kernel out.
        fn run_pooled(
            body: &[u8],
            pool: &x86_64::ConstPool,
            xs: [f32; 8],
            ys: [f32; 8],
            zs: [f32; 8],
        ) -> [f32; 8] {
            let mut asm = crate::emit::Assembly::default();
            x86_64::anchor(&mut asm);
            asm.code.extend_from_slice(body);
            crate::emit::x86_64::ret(&mut asm.code);
            pool.finish(&mut asm);
            // SAFETY: every caller is a test that checked the host runs AVX2.
            unsafe { run_code(&asm.finish(), xs, ys, zs) }
        }

        /// # Safety
        ///
        /// The host must execute AVX2: `code` is `ymm` code, and this function
        /// is compiled with AVX enabled to be allowed to pass `ymm` values.
        #[target_feature(enable = "avx2")]
        unsafe fn run_code(code: &[u8], xs: [f32; 8], ys: [f32; 8], zs: [f32; 8]) -> [f32; 8] {
            let exec = unsafe { ExecutableCode::from_code(code).expect("mmap") };
            unsafe {
                let f: K = exec.as_fn();
                let r = f(
                    _mm256_loadu_ps(xs.as_ptr()),
                    _mm256_loadu_ps(ys.as_ptr()),
                    _mm256_loadu_ps(zs.as_ptr()),
                    _mm256_setzero_ps(),
                );
                let mut out = [0.0f32; 8];
                _mm256_storeu_ps(out.as_mut_ptr(), r);
                out
            }
        }

        fn lanes() -> ([f32; 8], [f32; 8], [f32; 8]) {
            let mut xs = [0.0; 8];
            let mut ys = [0.0; 8];
            let mut zs = [0.0; 8];
            for i in 0..8 {
                xs[i] = i as f32 - 3.5;
                ys[i] = (i as f32) * 0.5 + 1.0;
                zs[i] = 3.0 - (i as f32) * 0.25;
            }
            (xs, ys, zs)
        }

        /// One row of the binary-op table: the op and its scalar reference.
        type BinaryCase = (OpKind, fn(f32, f32) -> f32);

        fn check(got: [f32; 8], want: impl Fn(usize) -> f32, tag: &str) {
            for (i, &g) in got.iter().enumerate() {
                let w = want(i);
                assert!((g - w).abs() <= 1e-3, "{tag} lane {i}: got {g} want {w}");
            }
        }

        /// Bit-exact check for mask results: an all-ones/all-zeros lane is
        /// NaN under float subtraction, so `check`'s epsilon comparison can't
        /// be used for compares/selects' underlying mask bit pattern.
        fn check_bits(got: [f32; 8], want: impl Fn(usize) -> u32, tag: &str) {
            for (i, &g) in got.iter().enumerate() {
                let w = want(i);
                assert_eq!(
                    g.to_bits(),
                    w,
                    "{tag} lane {i}: got {:#x} want {:#x}",
                    g.to_bits(),
                    w
                );
            }
        }

        const X: Reg = Reg(0);
        const Y: Reg = Reg(1);
        const Z: Reg = Reg(2);
        /// Standing in for the allocator's instruction temp: any register
        /// disjoint from the operands each case uses.
        const TEMP: Reg = Reg(15);

        #[test]
        fn emit_binary_matches_the_scalar_reference_for_every_arithmetic_op() {
            skip_unless_host_runs!(Isa::Avx2);
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
        fn emit_binary_produces_an_all_ones_mask_when_lt_holds() {
            skip_unless_host_runs!(Isa::Avx2);
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            emit_binary(&mut c, OpKind::Lt, X, X, Y);
            check_bits(
                run(&c, xs, ys, zs),
                |i| if xs[i] < ys[i] { 0xFFFF_FFFF } else { 0 },
                "lt mask",
            );
        }

        #[test]
        fn emit_movmskps_eax_gathers_the_lanewise_compare_mask_sign_bits() {
            skip_unless_host_runs!(Isa::Avx2);
            #[allow(improper_ctypes_definitions)]
            type MaskCheck = unsafe extern "C" fn(__m256, __m256) -> i32;

            /// # Safety
            ///
            /// The host must execute AVX2 (checked above).
            #[target_feature(enable = "avx2")]
            unsafe fn run_mask(body: &[u8], xs: [f32; 8], ys: [f32; 8]) -> i32 {
                let mut code = body.to_vec();
                crate::emit::x86_64::ret(&mut code);
                let exec = unsafe { ExecutableCode::from_code(&code).expect("mmap") };
                unsafe {
                    let f: MaskCheck = exec.as_fn();
                    f(_mm256_loadu_ps(xs.as_ptr()), _mm256_loadu_ps(ys.as_ptr()))
                }
            }

            // Half true, half false, so the result exercises every mask bit
            // rather than only the all-true/all-false extremes.
            let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
            let ys = [9.0, 9.0, 9.0, 9.0, -9.0, -9.0, -9.0, -9.0];
            let mut c = Vec::new();
            emit_binary(&mut c, OpKind::Lt, X, X, Y);
            emit_movmskps_eax(&mut c, X);
            // SAFETY: the host runs AVX2, checked at the top of this test.
            let got = unsafe { run_mask(&c, xs, ys) };
            assert_eq!(got, 0b0000_1111, "lt mask, lanes 0-3 true");

            // The complementary comparison, to pin the other half of eax
            // independently of the first assertion.
            let mut c = Vec::new();
            emit_binary(&mut c, OpKind::Gt, X, X, Y);
            emit_movmskps_eax(&mut c, X);
            // SAFETY: as above.
            let got = unsafe { run_mask(&c, xs, ys) };
            assert_eq!(got, 0b1111_0000, "gt mask, lanes 4-7 true");
        }

        #[test]
        fn emit_unary_computes_sqrt_neg_and_abs_per_lane() {
            skip_unless_host_runs!(Isa::Avx2);
            let (xs, ys, zs) = lanes();
            let unary = |op, src, temp| crate::emit::Unary {
                op,
                dst: X,
                src,
                temp,
            };
            let mut pool = x86_64::ConstPool::default();
            let mut c = Vec::new();
            emit_unary(&mut c, unary(OpKind::Sqrt, Y, None), &mut pool);
            check(run_pooled(&c, &pool, xs, ys, zs), |i| ys[i].sqrt(), "sqrt");

            let mut pool = x86_64::ConstPool::default();
            let mut c = Vec::new();
            emit_unary(&mut c, unary(OpKind::Neg, X, Some(TEMP)), &mut pool);
            check(run_pooled(&c, &pool, xs, ys, zs), |i| -xs[i], "neg");

            let mut pool = x86_64::ConstPool::default();
            let mut c = Vec::new();
            emit_unary(&mut c, unary(OpKind::Abs, X, Some(TEMP)), &mut pool);
            check(run_pooled(&c, &pool, xs, ys, zs), |i| xs[i].abs(), "abs");
        }

        #[test]
        fn emit_select_blends_if_true_and_if_false_by_the_mask() {
            skip_unless_host_runs!(Isa::Avx2);
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            emit_binary(&mut c, OpKind::Lt, Reg(5), X, Y); // mask
            emit_mov(&mut c, Reg(6), Reg(5));
            emit_select(&mut c, Reg(6), X, Y, Some(TEMP)); // dst = mask ? x : y
            emit_mov(&mut c, X, Reg(6));
            check(
                run(&c, xs, ys, zs),
                |i| if xs[i] < ys[i] { xs[i] } else { ys[i] },
                "select",
            );
        }

        /// Two constants, the first read twice: the pool holds each once, and
        /// every read is one broadcast from it.
        #[test]
        fn emit_const_broadcasts_and_adds_to_every_lane() {
            skip_unless_host_runs!(Isa::Avx2);
            let (xs, ys, zs) = lanes();
            let mut pool = x86_64::ConstPool::default();
            let mut c = Vec::new();
            emit_const(&mut c, Reg(5), 2.5, &mut pool);
            emit_binary(&mut c, OpKind::Add, X, X, Reg(5));
            emit_const(&mut c, Reg(6), -1.0, &mut pool);
            emit_binary(&mut c, OpKind::Add, X, X, Reg(6));
            emit_const(&mut c, Reg(5), 2.5, &mut pool);
            emit_binary(&mut c, OpKind::Add, X, X, Reg(5));
            check(
                run_pooled(&c, &pool, xs, ys, zs),
                |i| xs[i] + 4.0,
                "const+add",
            );
        }

        #[test]
        fn emit_fmadd_c_in_dst_computes_the_fused_multiply_add() {
            skip_unless_host_runs!(Isa::Avx2);
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            emit_mov(&mut c, Reg(5), Z);
            emit_fmadd_c_in_dst(&mut c, Reg(5), X, Y);
            emit_mov(&mut c, X, Reg(5));
            check(run(&c, xs, ys, zs), |i| xs[i] * ys[i] + zs[i], "fma231");
        }

        /// The FMA bytes really are an FMA: **one** rounding, not a multiply
        /// followed by an add.
        ///
        /// `const_broadcast_and_fma`'s 1e-3 tolerance cannot tell those apart — the whole
        /// difference is the last mantissa bit — so a stand-in built out of a
        /// multiply and an add would pass it. `1.0000001 * 4097 + 4097` is one
        /// of the inputs CLAUDE.md's `MulAdd` row is about, where the two
        /// forms genuinely disagree, and this asserts the bits.
        #[test]
        fn emit_fmadd_c_in_dst_rounds_once_not_twice() {
            skip_unless_host_runs!(Isa::Avx2);
            let xs = [1.000_000_1f32; 8];
            let ys = [4097.0f32; 8];
            let zs = [4097.0f32; 8];
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
        fn emit_load_after_emit_store_recovers_the_spilled_value() {
            skip_unless_host_runs!(Isa::Avx2);
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            AsmProgram::from([crate::emit::x86_64::Inst::SubImm32 {
                dst: crate::emit::x86_64::gpr::RSP,
                imm: crate::emit::x86_64::Imm32(32),
            }])
            .assemble(&mut c);
            emit_binary(&mut c, OpKind::Mul, Reg(6), X, Y);
            AsmProgram::from([Vex::m0f(0x11).rm(6, frame_slot(0))]).assemble(&mut c);
            emit_binary(&mut c, OpKind::Add, Reg(6), X, X); // clobber
            AsmProgram::from([Vex::m0f(0x10).rm(X.0, frame_slot(0))]).assemble(&mut c);
            AsmProgram::from([crate::emit::x86_64::Inst::AddImm32 {
                dst: crate::emit::x86_64::gpr::RSP,
                imm: crate::emit::x86_64::Imm32(32),
            }])
            .assemble(&mut c);
            check(run(&c, xs, ys, zs), |i| xs[i] * ys[i], "spill roundtrip");
        }

        #[test]
        fn emit_gather_scalar_reads_the_value_at_each_lanes_index() {
            skip_unless_host_runs!(Isa::Avx2);
            // Matches the production ABI (mod.rs's `ResolvedOp::Gather`): the
            // base is a pointer register the allocator placed — here the
            // first argument, `rdi`, holding the buffer's own address — not a
            // context slot the gather loads it from.
            #[allow(improper_ctypes_definitions)]
            type G = unsafe extern "C" fn(*const f32, __m256) -> __m256;

            /// # Safety
            ///
            /// The host must execute AVX2 (checked above).
            #[target_feature(enable = "avx2")]
            unsafe fn gather(exec: &ExecutableCode, base: *const f32, idx: [f32; 8]) -> [f32; 8] {
                unsafe {
                    let f: G = exec.as_fn();
                    let r = f(base, _mm256_loadu_ps(idx.as_ptr()));
                    let mut out = [0.0f32; 8];
                    _mm256_storeu_ps(out.as_mut_ptr(), r);
                    out
                }
            }

            let mut c = Vec::new();
            // idx (zmm/ymm0) -> int truncate happens inside emit_gather_scalar.
            let s = x86_64::GatherScratch {
                index_gpr: 1, // rcx
                idx_lanes: Reg(13),
                value: Reg(14),
            };
            emit_gather_scalar(
                &mut c,
                Reg(0),
                Reg(0),
                x86_64::ptr::RDI,
                GatherScratch {
                    half: s,
                    idx_hi: Reg(9),
                    res_hi: Reg(8),
                },
            );
            crate::emit::x86_64::ret(&mut c);

            let buf: Vec<f32> = (0..64).map(|i| (i as f32) * 1.5 + 0.25).collect();
            let idx: [f32; 8] = [0.0, 63.0, 1.0, 2.0, 10.0, 5.0, 32.0, 7.0];

            let exec = unsafe { ExecutableCode::from_code(&c).expect("mmap") };
            // SAFETY: the host runs AVX2, checked at the top of this test.
            let out = unsafe { gather(&exec, buf.as_ptr(), idx) };

            for i in 0..8 {
                let want = buf[idx[i] as usize];
                assert_eq!(out[i], want, "gather lane {i}: idx {}", idx[i]);
            }
        }
    }
}

// =============================================================================
// The AVX2 `IsaBackend` driver
// =============================================================================

/// The AVX2 half of code generation.
///
/// **This file is where AVX2-specific bugs live, and the only place they
/// can.** Emission is a pure function into `Vec<u8>`, so everything here
/// compiles, typechecks and is swept for op coverage on every host, whatever
/// CPU it has. Only `compile_native` in `emit` decides which backend a
/// process instantiates — from the tier `crate::isa` read off the CPU — and
/// only [`executable`](super::super::executable) needs the matching hardware.
///
/// The consequence worth stating: a change that does not touch an ISA file
/// cannot introduce a platform-specific bug. That is the bargain `unsafe`
/// makes — confine what cannot be checked, so the rest is checked by
/// construction.
pub(crate) mod driver {
    use super::super::*;
    use super::{
        AsmProgram, IOTA_BYTES, Mem, NoDisp, UNUSED_VVVV, Vex, frame_slot, vcvttss2si_mem,
        vcvttss2si_xmm, vextractf128, vextractps_store, vmovq_xmm_r64, vpmovzxbd,
    };
    use crate::emit::x86_64 as x86;
    use crate::emit::x86_64::driver::{Convert, SSE2_FILE, write_address};
    use crate::error::CompileError;
    use alloc::vec::Vec;
    use pixelflow_ir::kind::OpKind;

    /// The AVX2 register file (ymm, 256-bit).
    ///
    /// The same sixteen registers as SSE2's at twice the width. The gather
    /// borrows four of them across its own sequence (the high half's index
    /// and result beside the low half's pair), the sign mask and the select
    /// blend borrow one — all reservations the allocator makes for one
    /// instruction, so all of them are its the rest of the time.
    const AVX2_FILE: regalloc::RegisterFile = regalloc::RegisterFile {
        scratch: regalloc::RegSet::range(0, 16),
        fixed: &[],
        temps_for: super::temps_for,
        vector_bytes: 32,
        ..SSE2_FILE
    }
    .checked();

    /// AVX2 implementation of the shared driver's leaf operations.
    pub(crate) struct Avx2Backend {
        consts: x86::ConstPool,
        file: regalloc::RegisterFile,
    }

    impl Avx2Backend {
        pub(crate) fn new(ctx: EmitCtx) -> Self {
            Self {
                consts: x86::ConstPool::default(),
                file: AVX2_FILE.capped(ctx.max_regs),
            }
        }

        fn reload(&mut self, code: &mut Vec<u8>, reload: &Reload) {
            match reload {
                Reload::FromStack { target, slot } => {
                    AsmProgram::from([Vex::m0f(0x10).rm(target.0, frame_slot(slot.offset()))])
                        .assemble(code);
                }
                Reload::Const { target, val_bits } => {
                    super::emit_const(code, *target, f32::from_bits(*val_bits), &mut self.consts);
                }
                Reload::Ptr { target, slot } => self.ptr_load(code, *target, slot.offset()),
            }
        }
    }

    impl IsaBackend for Avx2Backend {
        fn jump(&mut self, asm: &mut Assembly, label: Label) {
            asm.push(x86::Jmp { target: label });
        }

        fn register_file(&self) -> regalloc::RegisterFile {
            self.file
        }

        /// Nothing to seed: the pool fills as constants are emitted.
        fn begin(&mut self, _schedule: &[regalloc::Def]) -> Result<(), CompileError> {
            Ok(())
        }

        fn emit_plan(
            &mut self,
            code: &mut Vec<u8>,
            plan: &InstructionPlan,
        ) -> Result<(), CompileError> {
            for r in &plan.reloads {
                self.reload(code, r);
            }
            if let Some((dst, src)) = plan.setup_mov
                && dst != src
            {
                AsmProgram::from([Vex::m0f(0x28).rrr(dst.0, UNUSED_VVVV, src.0)]).assemble(code);
            }
            match &plan.op {
                ResolvedOp::Nop => {}
                ResolvedOp::LoadConst { dst, val_bits } => {
                    super::emit_const(code, *dst, f32::from_bits(*val_bits), &mut self.consts);
                }
                // The iota: the bytes `0..8` in through a GPR, widened to
                // dwords, converted. No vector temp — `dst` is every stage's.
                ResolvedOp::Lanes { dst } => {
                    let gpr = crate::emit::declared_gpr_temp(plan.scratch.gpr_temp(0));
                    x86::movabs(code, gpr, IOTA_BYTES);
                    AsmProgram::from([
                        vmovq_xmm_r64(*dst, gpr),
                        vpmovzxbd(*dst, *dst),
                        Vex::m0f(0x5B).rrr(dst.0, UNUSED_VVVV, dst.0),
                    ])
                    .assemble(code);
                }
                ResolvedOp::Unary { op, dst, src } => {
                    let unary = Unary {
                        op: *op,
                        dst: *dst,
                        src: *src,
                        temp: plan.scratch.temp(0),
                    };
                    super::emit_unary(code, unary, &mut self.consts);
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
                    // `base` is the buffer's address wherever the allocator
                    // keeps it; the index GPR is `AVX2_FILE.gpr_scratch`'s
                    // reservation; the four vector temps are the two halves'
                    // index and value registers (see
                    // `super::emit_gather_scalar`).
                    super::emit_gather_scalar(
                        code,
                        *dst,
                        *idx,
                        *base,
                        super::GatherScratch {
                            half: x86_64::GatherScratch {
                                index_gpr: crate::emit::declared_gpr_temp(plan.scratch.gpr_temp(0))
                                    .0,
                                idx_lanes: crate::emit::declared_temp(plan.scratch.temp(0)),
                                value: crate::emit::declared_temp(plan.scratch.temp(1)),
                            },
                            idx_hi: crate::emit::declared_temp(plan.scratch.temp(2)),
                            res_hi: crate::emit::declared_temp(plan.scratch.temp(3)),
                        },
                    );
                }
                ResolvedOp::Broadcast { dst, idx, base } => {
                    super::emit_broadcast_load(
                        code,
                        *dst,
                        *idx,
                        x86::BroadcastGprs {
                            base: *base,
                            index: crate::emit::declared_gpr_temp(plan.scratch.gpr_temp(0)),
                        },
                    );
                }
                ResolvedOp::Uniform { dst, base, offset } => {
                    super::emit_uniform_load(code, *dst, *base, *offset);
                }
                ResolvedOp::Context { dst, slot } => {
                    let ctx = self
                        .file
                        .gpr_ctx
                        .expect("x86's context read needs the GPR context input");
                    AsmProgram::from([x86::MovLoadPtr {
                        dst: *dst,
                        base: PtrReg(ctx.0),
                        disp: i32::from(*slot) * x86::PTR_BYTES,
                    }
                    .encode()])
                    .assemble(code);
                }
                ResolvedOp::Binary {
                    op,
                    dst,
                    left,
                    right,
                } => {
                    // VEX 3-operand: no two-operand hazard, emit directly.
                    super::emit_binary(code, *op, *dst, *left, *right);
                }
                ResolvedOp::FusedMulAdd { dst, a, b } => {
                    super::emit_fmadd_c_in_dst(code, *dst, *a, *b);
                }
                ResolvedOp::DecomposedMulAdd {
                    dst,
                    a,
                    b,
                    c,
                    c_deferred,
                } => {
                    super::emit_binary(code, OpKind::Mul, *dst, *a, *b);
                    match c_deferred {
                        Some(DeferredReload::FromStack(slot)) => {
                            AsmProgram::from([Vex::m0f(0x10).rm(c.0, frame_slot(slot.offset()))])
                                .assemble(code);
                        }
                        Some(DeferredReload::Const(bits)) => {
                            super::emit_const(code, *c, f32::from_bits(*bits), &mut self.consts);
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
                    // setup_mov already placed the vector mask in dst.
                    super::emit_select(code, *dst, *if_true, *if_false, plan.scratch.temp(0));
                }
            }
            Ok(())
        }

        fn emit_mov(&mut self, code: &mut Vec<u8>, dst: Reg, src: Reg) {
            if dst != src {
                AsmProgram::from([Vex::m0f(0x28).rrr(dst.0, UNUSED_VVVV, src.0)]).assemble(code);
            }
        }

        fn emit_store(
            &mut self,
            code: &mut Vec<u8>,
            src: Reg,
            offset: u32,
        ) -> Result<(), CompileError> {
            AsmProgram::from([Vex::m0f(0x11).rm(src.0, frame_slot(offset))]).assemble(code);
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
                    super::emit_const(code, target, f32::from_bits(bits), &mut self.consts);
                    target
                }
                Binding::Loc(Loc::Slot(slot)) => {
                    AsmProgram::from([Vex::m0f(0x10).rm(target.0, frame_slot(slot.offset()))])
                        .assemble(code);
                    target
                }
                Binding::Loc(Loc::Ptr(p)) => {
                    unreachable!("{vid:?} is an address in {p:?}; the pointer class resolves it")
                }
            }
        }

        fn ptr_store(&mut self, code: &mut Vec<u8>, src: PtrReg, offset: u32) {
            AsmProgram::from([x86::MovStorePtr {
                src,
                base: x86::ptr::RSP,
                disp: offset as i32,
            }
            .encode()])
            .assemble(code);
        }

        fn ptr_load(&mut self, code: &mut Vec<u8>, dst: PtrReg, offset: u32) {
            AsmProgram::from([x86::MovLoadPtr {
                dst,
                base: x86::ptr::RSP,
                disp: offset as i32,
            }
            .encode()])
            .assemble(code);
        }

        fn ptr_mov(&mut self, code: &mut Vec<u8>, dst: PtrReg, src: PtrReg) {
            x86::mov(code, dst.as_gpr(), src.as_gpr());
        }

        fn anchor(&mut self, asm: &mut Assembly) {
            x86::anchor(asm);
        }

        fn finish(&mut self, asm: &mut Assembly) {
            self.consts.finish(asm);
        }

        // Select short-circuit guards: vmovmskps -> eax[7:0], same shape as
        // X86Backend's MOVMSKPS guards but 8 lanes wide (al == 0xFF for
        // all-true, not 0x0F — see `super::emit_cmp_al_imm8`'s doc for why the
        // sign-extending `cmp eax, imm8` X86Backend uses doesn't work here).
        /// [`MaskTest::scratch`] and [`MaskTest::mask_scratch`] are both
        /// unused: this tier reduces the mask with `movmskps` into the
        /// flags, needing neither a vector nor a mask register.
        fn branch_if_arm_is_dead(&mut self, asm: &mut Assembly, test: MaskTest, label: Label) {
            super::emit_movmskps_eax(&mut asm.code, test.reg);
            match test.arm {
                // ZF set when eax == 0: no lane is true, so the true arm is dead.
                SelectArm::True => x86_64::emit_test_eax(&mut asm.code),
                // ZF set when al == 0xFF: every lane is true, so the false arm is.
                SelectArm::False => super::emit_cmp_al_imm8(&mut asm.code, 0xFF),
            }
            asm.push(x86::Jcc::je(label));
        }

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
            AsmProgram::from([Vex::m0f(0x11).rm(src.0, frame_slot(offset))]).assemble(code);
        }

        fn slot_load(&mut self, code: &mut Vec<u8>, dst: Reg, offset: u32) {
            AsmProgram::from([Vex::m0f(0x10).rm(dst.0, frame_slot(offset))]).assemble(code);
        }

        fn add_scalar(&mut self, code: &mut Vec<u8>, dst: Reg, scratch: Reg, scalar: f32) {
            super::emit_const(code, scratch, scalar, &mut self.consts);
            super::emit_binary(code, OpKind::Add, dst, dst, scratch);
        }

        fn load_const(&mut self, code: &mut Vec<u8>, dst: Reg, val: f32) {
            super::emit_const(code, dst, val, &mut self.consts);
        }

        fn alu(&mut self, code: &mut Vec<u8>, op: OpKind, dst: Reg, srcs: [Reg; 2]) {
            super::emit_binary(code, op, dst, srcs[0], srcs[1]);
        }

        /// A full batch is one `vmovups`. A remainder is `vextractps` per
        /// lane: the low four straight out of the value, the rest out of its
        /// high half extracted into the reserved temp.
        fn emit_write(&mut self, code: &mut Vec<u8>, write: &WritePlan) {
            let addr = write_address(
                code,
                &self.file,
                write,
                Convert {
                    from_xmm: |code, dst, src| {
                        AsmProgram::from([vcvttss2si_xmm(dst, src)]).assemble(code)
                    },
                    from_mem: |code, dst, addr| {
                        AsmProgram::from([vcvttss2si_mem(dst, addr)]).assemble(code)
                    },
                },
            );
            let base = PtrReg(addr.0);
            let lanes = self.file.vector_bytes / 4;
            if write.lanes == lanes {
                AsmProgram::from([Vex::m0f(0x11).rm(write.value.0, Mem { base, disp: NoDisp })])
                    .assemble(code);
                return;
            }
            let mut half = write.value;
            for lane in 0..write.lanes {
                if lane == 4 {
                    half = crate::emit::declared_temp(write.scratch.temp(0));
                    vextractf128(code, half.0, write.value.0, 1);
                }
                AsmProgram::from([vextractps_store(
                    Mem {
                        base,
                        disp: x86::Imm8((lane * 4) as i8),
                    },
                    half,
                    (lane % 4) as u8,
                )])
                .assemble(code);
            }
        }

        fn emit_ret(&mut self, code: &mut Vec<u8>) {
            AsmProgram::from([x86::Inst::Ret]).assemble(code);
        }
    }
}
