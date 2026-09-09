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

use super::{
    AsmInsn, AsmProgram, Counter, EncodedInst, Gpr, OutStep, PtrReg, Reg, SourceOperand, assemble,
    unimplemented_op,
};
use alloc::vec::Vec;
use pixelflow_ir::kind::OpKind;

// =============================================================================
// Encoding Helpers
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

/// A `/digit` opcode extension: the ModRM.reg field carries an opcode bit
/// group where a register would otherwise go. Distinct from [`Reg`] because
/// it names no register at all.
#[derive(Clone, Copy)]
struct Digit(u8);

/// The identity of one VEX-128 instruction: opcode map, implied legacy
/// prefix, W bit, opcode byte. This quadruple is *which instruction* — it is
/// constant per mnemonic, so each mnemonic below states it exactly once and
/// the operand form (`rrr`/`rr`/`digit_rm`/...) supplies the per-call parts.
///
/// The 256-bit twin of this type is `avx2::Vex`; the only encoding difference
/// is VEX.L, which is 0 here.
#[derive(Clone, Copy)]
struct Vex {
    map: Map,
    pp: Pp,
    w: bool,
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
    /// Map `0F3A`, `66` — the imm8 family (round, insert/extract).
    const fn m0f3a_66(opcode: u8) -> Self {
        Self::new(Map::M0F3A, Pp::P66, opcode)
    }
    /// Map `0F38`, `66` — the broadcast family.
    const fn m0f38_66(opcode: u8) -> Self {
        Self::new(Map::M0F38, Pp::P66, opcode)
    }

    /// Attach an imm8 (rounding mode, shift count, lane index); the returned
    /// value emits it after the instruction.
    const fn imm(self, imm: u8) -> VexImm {
        VexImm { vex: self, imm }
    }

    /// `op reg, [addr]` — the memory-operand form, for any base and any
    /// displacement mode. The prefix is VEX's (L=0); the ModRM/SIB/
    /// displacement tail is the architecture's, so it comes from
    /// [`mem_operand`]. VEX.B extends the *base* here, not an r/m register.
    fn rm<D: Disp, P: BaseReg>(self, reg: Reg, addr: Mem<D, P>) -> EncodedInst {
        let mut inst = EncodedInst::new();
        self.head_into(&mut inst, reg.0, UNUSED_VVVV, addr.base.reg_num());
        mem_operand_into(&mut inst, reg.0, addr);
        inst
    }

    /// `op reg, vvvv, [addr]` — 3-operand VEX with memory operand.
    #[allow(dead_code)]
    fn rrm<D: Disp, P: BaseReg>(self, reg: Reg, vvvv: Reg, addr: Mem<D, P>) -> EncodedInst {
        let mut inst = EncodedInst::new();
        self.head_into(&mut inst, reg.0, vvvv.0, addr.base.reg_num());
        mem_operand_into(&mut inst, reg.0, addr);
        inst
    }

    /// Generic 3-operand form: `op reg, vvvv, rm` where `rm` can be a register or stack slot.
    #[allow(dead_code)]
    fn rro<S: SourceOperand>(self, reg: Reg, vvvv: Reg, rm: S) -> Option<EncodedInst> {
        if let Some(r) = rm.source_reg() {
            return Some(self.rrr(reg, vvvv, r));
        }
        let slot = rm.source_slot()?;
        Some(self.rrm(
            reg,
            vvvv,
            Mem {
                base: gpr::RSP,
                disp: Imm32(slot.offset() as i32),
            },
        ))
    }

    /// Generic 2-operand form: `op reg, rm` where `rm` can be a register or stack slot.
    #[allow(dead_code)]
    fn ro<S: SourceOperand>(self, reg: Reg, rm: S) -> Option<EncodedInst> {
        if let Some(r) = rm.source_reg() {
            return Some(self.rr(reg, r));
        }
        let slot = rm.source_slot()?;
        Some(self.rm(
            reg,
            Mem {
                base: gpr::RSP,
                disp: Imm32(slot.offset() as i32),
            },
        ))
    }

    /// Register-register-register form: `op reg, vvvv, rm`.
    fn rrr(self, reg: Reg, vvvv: Reg, rm: Reg) -> EncodedInst {
        let mut inst = EncodedInst::new();
        self.head_into(&mut inst, reg.0, vvvv.0, rm.0);
        inst.push(0xC0 | ((reg.0 & 7) << 3) | (rm.0 & 7));
        inst
    }

    /// Two-operand form: `op reg, rm`, VEX.vvvv unused.
    fn rr(self, reg: Reg, rm: Reg) -> EncodedInst {
        let mut inst = EncodedInst::new();
        self.head_into(&mut inst, reg.0, UNUSED_VVVV, rm.0);
        inst.push(0xC0 | ((reg.0 & 7) << 3) | (rm.0 & 7));
        inst
    }

    /// `/digit` form: `op vvvv, rm, ...` with the opcode extension in
    /// ModRM.reg (the shift-by-immediate group).
    fn digit_rm(self, ext: Digit, vvvv: Reg, rm: Reg) -> EncodedInst {
        let mut inst = EncodedInst::new();
        self.head_into(&mut inst, ext.0, vvvv.0, rm.0);
        inst.push(0xC0 | ((ext.0 & 7) << 3) | (rm.0 & 7));
        inst
    }

    /// `op reg, rmGPR` — the r/m operand names a *general* register, the
    /// reverse of the usual direction (`vpextrd`).
    fn rr_gpr(self, reg: Reg, rm_gpr: u8) -> EncodedInst {
        debug_assert!(rm_gpr < 8, "Vex::rr_gpr: GPR8 only");
        let mut inst = EncodedInst::new();
        self.head_into(&mut inst, reg.0, UNUSED_VVVV, rm_gpr);
        inst.push(0xC0 | ((reg.0 & 7) << 3) | (rm_gpr & 7));
        inst
    }

    /// `op reg, [baseGPR + indexGPR*4]` — SIB, scale 4, no displacement.
    fn rm_scaled4(self, reg: Reg, base_gpr: u8, index_gpr: u8) -> EncodedInst {
        debug_assert!(base_gpr < 8 && index_gpr < 8, "Vex::rm_scaled4: GPR8 only");
        debug_assert!(base_gpr != 5, "base rbp/r13 would force a disp form");
        let mut inst = EncodedInst::new();
        self.head_into(&mut inst, reg.0, UNUSED_VVVV, base_gpr);
        inst.push(((reg.0 & 7) << 3) | 0b100); // mod=00, rm=SIB
        inst.push((0b10 << 6) | ((index_gpr & 7) << 3) | (base_gpr & 7)); // scale=4
        inst
    }

    /// The 3-byte VEX prefix plus the opcode byte, shared by every form
    /// above: `C4 RXB.mmmmm W.vvvv.L.pp opcode`, with L=0 (128-bit).
    fn head_into(self, inst: &mut EncodedInst, reg: u8, vvvv: u8, rm: u8) {
        let rbit = if reg >= 8 { 0x00 } else { 0x80 };
        let xbit = 0x40; // X unused: no index register in these forms
        let bbit = if rm >= 8 { 0x00 } else { 0x20 };
        inst.push(0xC4);
        inst.push(rbit | xbit | bbit | self.map as u8);
        inst.push(((self.w as u8) << 7) | ((!vvvv & 0xF) << 3) | self.pp as u8);
        inst.push(self.opcode);
    }
}

/// A [`Vex`] instruction carrying its imm8.
#[derive(Clone, Copy)]
struct VexImm {
    vex: Vex,
    imm: u8,
}

impl VexImm {
    /// Two-operand form with the imm8 appended.
    fn rr(self, reg: Reg, rm: Reg) -> EncodedInst {
        let mut inst = self.vex.rr(reg, rm);
        inst.push(self.imm);
        inst
    }

    /// Register-register-register form with the imm8 appended.
    fn rrr(self, reg: Reg, vvvv: Reg, rm: Reg) -> EncodedInst {
        let mut inst = self.vex.rrr(reg, vvvv, rm);
        inst.push(self.imm);
        inst
    }

    /// `/digit` form with the imm8 appended.
    fn digit_rm(self, ext: Digit, vvvv: Reg, rm: Reg) -> EncodedInst {
        let mut inst = self.vex.digit_rm(ext, vvvv, rm);
        inst.push(self.imm);
        inst
    }

    /// `op reg, rmGPR, imm8` — see [`Vex::rr_gpr`].
    fn rr_gpr(self, reg: Reg, rm_gpr: u8) -> EncodedInst {
        let mut inst = self.vex.rr_gpr(reg, rm_gpr);
        inst.push(self.imm);
        inst
    }
}

/// Emit SSE instruction (legacy encoding, 2-operand: dst op= src)
fn sse_rr(opcode: &[u8], dst: Reg, src: Reg) -> EncodedInst {
    let mut inst = EncodedInst::new();
    let rex = 0x40 | (if dst.0 >= 8 { 0x04 } else { 0 }) | (if src.0 >= 8 { 0x01 } else { 0 });
    if rex != 0x40 {
        inst.push(rex);
    }
    inst.extend(opcode);
    inst.push(0xC0 | ((dst.0 & 7) << 3) | (src.0 & 7));
    inst
}

fn emit_sse_rr(code: &mut Vec<u8>, opcode: &[u8], dst: Reg, src: Reg) {
    assemble(code, [sse_rr(opcode, dst, src)]);
}

// =============================================================================
// Load / Store
// =============================================================================

/// MOVAPS xmm, xmm - Register-to-register copy
pub fn emit_movaps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, &[0x0F, 0x28], dst, src);
}

/// `movups` between `xmm<reg>` and `[addr]`. Load and store are one encoding
/// a single opcode byte apart, so they share everything below.
fn movups<D: Disp, P: BaseReg>(opcode: u8, reg: Reg, addr: Mem<D, P>) -> EncodedInst {
    let mut inst = EncodedInst::new();
    // REX only when an operand needs extending: R for the xmm, B for the base.
    let rex = 0x40 | (u8::from(reg.0 >= 8) << 2) | u8::from(addr.base.reg_num() >= 8);
    if rex != 0x40 {
        inst.push(rex);
    }
    inst.push(0x0F);
    inst.push(opcode);
    mem_operand_into(&mut inst, reg.0, addr);
    inst
}

/// `movups xmm<dst>, [addr]` — unaligned 128-bit load.
#[must_use]
#[inline(always)]
pub fn movups_load<D: Disp, P: BaseReg>(dst: Reg, addr: Mem<D, P>) -> EncodedInst {
    movups(0x10, dst, addr)
}

/// `movups [addr], xmm<src>` — unaligned 128-bit store.
///
/// The address comes first because that is where it sits in the instruction:
/// the direction is the operand *order*, and `0F 10` versus `0F 11` is the
/// only thing the two mnemonics do not share.
#[must_use]
#[inline(always)]
pub fn movups_store<D: Disp, P: BaseReg>(src: Reg, addr: Mem<D, P>) -> EncodedInst {
    movups(0x11, src, addr)
}

// =============================================================================
// Arithmetic (SSE legacy 2-operand)
// =============================================================================

/// ADDPS xmm, xmm
pub fn emit_addps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, &[0x0F, 0x58], dst, src);
}

/// SUBPS xmm, xmm
pub fn emit_subps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, &[0x0F, 0x5C], dst, src);
}

/// MULPS xmm, xmm
pub fn emit_mulps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, &[0x0F, 0x59], dst, src);
}

/// DIVPS xmm, xmm
pub fn emit_divps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, &[0x0F, 0x5E], dst, src);
}

/// SQRTPS xmm, xmm
pub fn emit_sqrtps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, &[0x0F, 0x51], dst, src);
}

/// RSQRTPS xmm, xmm (approximate)
pub fn emit_rsqrtps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, &[0x0F, 0x52], dst, src);
}

/// RCPPS xmm, xmm (approximate reciprocal)
pub fn emit_rcpps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, &[0x0F, 0x53], dst, src);
}

/// MINPS xmm, xmm
pub fn emit_minps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, &[0x0F, 0x5D], dst, src);
}

/// MAXPS xmm, xmm
pub fn emit_maxps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    emit_sse_rr(code, &[0x0F, 0x5F], dst, src);
}

// =============================================================================
// Arithmetic (VEX 3-operand)
// =============================================================================

// =============================================================================
// Bitwise (VEX 3-operand)
// =============================================================================

/// VANDPS dst, src1, src2 — bitwise AND
fn emit_vandps(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    assemble(code, [Vex::m0f(0x54).rrr(dst, src1, src2)]);
}

/// VANDNPS dst, src1, src2 — bitwise NOT(src1) AND src2
fn emit_vandnps(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    assemble(code, [Vex::m0f(0x55).rrr(dst, src1, src2)]);
}

/// VORPS dst, src1, src2 — bitwise OR
fn emit_vorps(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    assemble(code, [Vex::m0f(0x56).rrr(dst, src1, src2)]);
}

/// VXORPS dst, src1, src2 — bitwise XOR
fn emit_vxorps(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    assemble(code, [Vex::m0f(0x57).rrr(dst, src1, src2)]);
}

// =============================================================================
// Comparisons (VEX)
// =============================================================================

/// VCMPPS predicates
const CMP_LT: u8 = 1; // Less than (ordered, non-signaling)
const CMP_NLE: u8 = 6; // Not less-or-equal, i.e. greater than (unordered)

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
pub fn emit_const(code: &mut Vec<u8>, dst: Reg, val: f32) {
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

/// VROUNDPS dst, src, imm8 — round packed f32 (imm: 0=nearest, 1=floor, 2=ceil, 3=trunc).
fn emit_vroundps(code: &mut Vec<u8>, dst: Reg, src: Reg, imm: u8) {
    assemble(code, [Vex::m0f3a_66(0x08).imm(imm).rr(dst, src)]); // VEX.128.66.0F3A.WIG 08 /r ib
}

/// VCVTTPS2DQ dst, src — convert packed f32 → i32 with truncation.
fn emit_vcvttps2dq(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    assemble(code, [Vex::m0f_f3(0x5B).rr(dst, src)]); // VEX.128.F3.0F.WIG 5B /r
}

/// VCVTDQ2PS dst, src — convert packed i32 → f32.
fn emit_vcvtdq2ps(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    assemble(code, [Vex::m0f(0x5B).rr(dst, src)]); // VEX.128.0F.WIG 5B /r
}

// ─────────────────────────── bound-memory gather ────────────────────────────
//
// This build has no AVX2, so there is no `vgatherdps` at 128 bits. A gather is
// therefore four independent scalar loads assembled into a lane vector:
// extract each lane's integer index to a GPR, load that element, and insert it
// into the destination lane. All four instructions below are plain AVX (the
// VEX encodings of SSE4.1 ops), the same tier the rest of this backend already
// emits (`vroundps`, `vandnps`).

/// `mov dstGPR, [ctxGPR + disp32]` — load a buffer base pointer out of the
/// 64-bit pointer load: `mov dst, [base + disp32]`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MovLoadPtr {
    pub dst: PtrReg,
    pub base: PtrReg,
    pub disp: i32,
}

impl MovLoadPtr {
    #[must_use]
    #[inline]
    pub fn encode(self) -> EncodedInst {
        debug_assert!(self.dst.0 < 8 && self.base.0 < 8, "MovLoadPtr: GPR8 only");
        let mut inst = EncodedInst::new();
        inst.push(0x48);
        inst.push(0x8B);
        inst.push(0x80 | ((self.dst.0 & 7) << 3) | (self.base.0 & 7));
        inst.extend(&self.disp.to_le_bytes());
        inst
    }
}

impl AsmInsn for MovLoadPtr {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        self.encode().emit_into(code);
    }
}

/// Fetch a pointer out of the caller's context: `mov dst, [ctx + disp32]`.
/// Used by gather to load a buffer's base, and by `emit_uniform_load` to load
/// the uniform block's base. Read-only on the context struct. Mirrors the
/// AVX-512 backend's loader.
pub fn emit_load_ptr_from_ctx(code: &mut Vec<u8>, dst_gpr: u8, ctx_gpr: u8, disp: i32) {
    MovLoadPtr {
        dst: PtrReg(dst_gpr),
        base: PtrReg(ctx_gpr),
        disp,
    }
    .emit_into(code);
}

/// `vpextrd r32, xmmSRC, lane` — move one 32-bit lane into a GP register.
fn emit_vpextrd_to_gpr(code: &mut Vec<u8>, dst_gpr: u8, src: Reg, lane: u8) {
    debug_assert!(lane < 4, "vpextrd lane must be 0..4");
    // VEX.128.66.0F3A.W0 16 /r ib — note the *xmm* is the ModRM.reg operand and
    // the GPR is r/m, the reverse of the usual direction.
    assemble(code, [Vex::m0f3a_66(0x16).imm(lane).rr_gpr(src, dst_gpr)]);
}

/// `vmovss xmmDST, [baseGPR + indexGPR*4]` — load one f32 element, zeroing the
/// upper lanes.
fn emit_vmovss_load_scaled(code: &mut Vec<u8>, dst: Reg, base_gpr: u8, index_gpr: u8) {
    // VEX.LIG.F3.0F.WIG 10 /r, mod=00 rm=100 (SIB), SIB scale=4.
    assemble(
        code,
        [Vex::m0f_f3(0x10).rm_scaled4(dst, base_gpr, index_gpr)],
    );
}

/// `vinsertps xmmDST, xmmSRC1, xmmSRC2, imm8` — place lane 0 of `src2` into
/// lane `dst_lane` of the result, keeping `src1`'s other lanes.
fn emit_vinsertps(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg, dst_lane: u8) {
    debug_assert!(dst_lane < 4, "vinsertps lane must be 0..4");
    // VEX.128.66.0F3A.WIG 21 /r ib. imm8: [7:6] source lane, [5:4] dest lane,
    // [3:0] zero mask (none).
    assemble(
        code,
        [Vex::m0f3a_66(0x21).imm(dst_lane << 4).rrr(dst, src1, src2)],
    );
}

/// Scratch the scalar gather sequence clobbers. The vector pair must be
/// distinct from each other and from the gather's index operand; the GPRs must
/// be free across the sequence.
#[derive(Clone, Copy)]
pub struct GatherScratch {
    /// GPR receiving the buffer base pointer.
    pub base_gpr: u8,
    /// GPR receiving each lane's element index.
    pub index_gpr: u8,
    /// GPR holding the caller's context pointer (read-only).
    pub ctx_gpr: u8,
    /// Vector register for the truncated integer indices.
    pub idx_lanes: Reg,
    /// Vector register for one loaded element.
    pub value: Reg,
}

/// Bytes per pointer in the context array.
const PTR_BYTES: i32 = 8;
/// Bytes per value in the uniform block.
const UNIFORM_BYTES: i32 = 4;

/// `dst = splat(block[offset])` — one uniform, broadcast to every lane:
/// `mov rax, [rdi + ctx_slot*8]` to fetch the block's base out of the
/// context, then `vbroadcastss xmm<dst>, [rax + 4*offset]`
/// (VEX.128.66.0F38.W0 18 /r). `rax` is the scratch the gather already
/// claims; `rdi` is the context, read-only for the whole kernel.
pub fn emit_uniform_load(code: &mut Vec<u8>, dst: Reg, load: super::UniformLoad) {
    AsmProgram::from([
        MovLoadPtr {
            dst: ptr::RAX,
            base: ptr::RDI,
            disp: i32::from(load.ctx_slot) * PTR_BYTES,
        }
        .encode(),
        Vex::m0f38_66(0x18).rm(
            dst,
            Mem {
                base: ptr::RAX,
                disp: Imm32(i32::from(load.offset) * UNIFORM_BYTES),
            },
        ),
    ])
    .assemble(code);
}

/// `dst = base[idx_lane]` for each lane — the whole gather sequence.
///
/// `idx` holds the *float* indices (the lowering already clamped them in
/// range). `dst` may alias `idx`: the indices are converted into scratch before
/// the first write to `dst`.
pub fn emit_gather_scalar(code: &mut Vec<u8>, dst: Reg, idx: Reg, slot: u16, s: GatherScratch) {
    debug_assert!(s.idx_lanes.0 != s.value.0 && s.idx_lanes.0 != idx.0);
    emit_vcvttps2dq(code, s.idx_lanes, idx);
    emit_load_ptr_from_ctx(code, s.base_gpr, s.ctx_gpr, i32::from(slot) * 8);
    for lane in 0..4u8 {
        emit_vpextrd_to_gpr(code, s.index_gpr, s.idx_lanes, lane);
        if lane == 0 {
            // Lane 0 seeds the vector (vmovss zeroes lanes 1..4).
            emit_vmovss_load_scaled(code, dst, s.base_gpr, s.index_gpr);
        } else {
            emit_vmovss_load_scaled(code, s.value, s.base_gpr, s.index_gpr);
            emit_vinsertps(code, dst, dst, s.value, lane);
        }
    }
}

/// VPADDD dst, src1, src2 — packed i32 add.
fn emit_vpaddd(code: &mut Vec<u8>, dst: Reg, src1: Reg, src2: Reg) {
    assemble(code, [Vex::m0f_66(0xFE).rrr(dst, src1, src2)]); // VEX.128.66.0F.WIG FE /r
}

/// VPSLLD dst, src, imm8 — packed i32 shift-left-logical by immediate.
fn emit_vpslld_imm(code: &mut Vec<u8>, dst: Reg, src: Reg, imm: u8) {
    // VEX.128.66.0F.WIG 72 /6 ib ; dst = vvvv, src = rm, /6 in ModRM.reg.
    assemble(
        code,
        [Vex::m0f_66(0x72).imm(imm).digit_rm(Digit(6), dst, src)],
    );
}

/// VPSRLD dst, src, imm8 — packed i32 shift-right-logical by immediate.
fn emit_vpsrld_imm(code: &mut Vec<u8>, dst: Reg, src: Reg, imm: u8) {
    // VEX.128.66.0F.WIG 72 /2 ib.
    assemble(
        code,
        [Vex::m0f_66(0x72).imm(imm).digit_rm(Digit(2), dst, src)],
    );
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

/// `dst = mask ? if_true : if_false` (bit-select; mask already in `dst`, same
/// convention as AVX2/AVX-512).
///
/// `tmp` is the allocator's temp for this instruction, which it picks disjoint
/// from the destination and every operand — the `debug_assert` restates that
/// here, where the blend would otherwise read back its own scratch.
pub fn emit_select(code: &mut Vec<u8>, dst: Reg, if_true: Reg, if_false: Reg, tmp: Option<Reg>) {
    let tmp = super::declared_temp(tmp);
    debug_assert!(tmp.0 != dst.0 && tmp.0 != if_true.0 && tmp.0 != if_false.0);
    emit_vandps(code, tmp, dst, if_true); // tmp = mask & if_true
    emit_vandnps(code, dst, dst, if_false); // dst = ~mask & if_false
    emit_vorps(code, dst, tmp, dst); // dst = blended
}

// =============================================================================
// High-level dispatch
// =============================================================================

/// How many registers this backend's encodings need beyond their operands.
///
/// On more ops than the VEX/EVEX tiers, because SSE2's two-operand form has
/// no non-destructive spelling: `Neg`/`Abs` build a sign mask, the select
/// blends through a temporary, and `MulAdd` — which this tier has no
/// instruction for — multiplies into one before adding.
pub(crate) fn temps_for(op: &super::ScheduledOp) -> u8 {
    use super::ScheduledOp;
    match op {
        ScheduledOp::Unary(OpKind::Neg | OpKind::Abs, _) => 1,
        ScheduledOp::Ternary(OpKind::Select, ..) => 1,
        // `FusedMulAdd` here is a `movaps`/`mulps`/`addps` stand-in: the
        // product needs somewhere to live that is neither `dst` (which already
        // holds the addend) nor an operand. The decomposed form needs none,
        // but which of the two a node gets is decided at emit time by whether
        // its multiplicands are resident, so the demand is stated for both.
        ScheduledOp::Ternary(OpKind::MulAdd, ..) => 1,
        // The gather truncates the float indices into one vector register and
        // loads each element through another.
        ScheduledOp::Gather(..) => 2,
        _ => 0,
    }
}

/// `dst = op(src)`.
///
/// `temp` is the allocator's temp for this instruction; only `Neg` and `Abs`
/// use it, to hold the sign mask they XOR or AND with.
pub fn emit_unary(code: &mut Vec<u8>, op: OpKind, dst: Reg, src: Reg, temp: Option<Reg>) {
    match op {
        OpKind::Sqrt => emit_sqrtps(code, dst, src),
        OpKind::Rsqrt => {
            emit_rsqrtps(code, dst, src);
            // TODO: Newton-Raphson refinement
        }
        OpKind::Recip => emit_rcpps(code, dst, src),

        // Negation: flip the sign bit (dst = src XOR 0x80000000).
        OpKind::Neg => {
            let mask = super::declared_temp(temp);
            emit_f32_const(code, mask, f32::from_bits(0x8000_0000));
            emit_vxorps(code, dst, src, mask);
        }

        // Absolute value: clear the sign bit (dst = src AND 0x7FFFFFFF).
        OpKind::Abs => {
            let mask = super::declared_temp(temp);
            emit_f32_const(code, mask, f32::from_bits(0x7FFF_FFFF));
            emit_vandps(code, dst, src, mask);
        }

        // Rounding (AVX vroundps)
        OpKind::Floor => emit_floor_builtin(code, dst, src),
        OpKind::Ceil => emit_ceil_builtin(code, dst, src),
        OpKind::Round => emit_round_builtin(code, dst, src),

        // Bit-manip primitives (integer-domain). Single instructions.
        OpKind::TruncToInt => emit_vcvttps2dq(code, dst, src),
        OpKind::IntToFloat => emit_vcvtdq2ps(code, dst, src),

        _ => unimplemented_op("x86-64", op),
    }
}

/// Emit a logical shift of i32 lanes by a compile-time immediate.
/// `Shl` -> `vpslld`, `Shr` -> `vpsrld` (logical). VEX form is 3-operand
/// (`dst = src << imm`), so there is no two-operand hazard.
pub fn emit_shift_imm(code: &mut Vec<u8>, op: OpKind, dst: Reg, src: Reg, amount: u8) {
    match op {
        OpKind::Shl => emit_vpslld_imm(code, dst, src, amount),
        OpKind::Shr => emit_vpsrld_imm(code, dst, src, amount),
        _ => unimplemented_op("x86-64", op),
    }
}

// =============================================================================
// Instruction selection: OpKind -> a closed set of binary-op mnemonics
// =============================================================================
//
// This used to be one flat `match OpKind { Add => emit_addps(...), ...,
// _ => panic!() }`: "which ops exist" and "how do I encode this op" were the
// same match, so a missing arm only showed up as a runtime panic on whatever
// input happened to exercise it — exactly how AVX-512's binary-op match sat
// at 6-of-15 ops for a release (see avx512.rs `emit_binary`, and
// docs/designs/2026-07-25-two-level-ir-and-backend-completeness.md).
// Splitting selection from encoding makes "does this backend support op Y"
// a question with one authoritative answer, not an emergent property of a
// match arm nobody re-checked:
//
//   1. **Selection** (`X86BinaryInsn::select`): OpKind -> Option<X86BinaryInsn>.
//      Still partial over the full `OpKind` (most of its variants are unary,
//      ternary, or eliminated by `lowering` before they ever reach a backend
//      — see `lowering.rs`), so this keeps a `_ => None`. But every op this
//      backend claims to encode is named exactly once, here, as data — a
//      completeness test enumerates against this function directly instead
//      of poking the flat dispatch and hoping to hit every case.
//   2. **Encoding** (`X86BinaryInsn::encode`): X86BinaryInsn -> bytes. This
//      match has NO wildcard: it is exhaustive over the closed mnemonic
//      enum, so adding a variant without teaching `encode` how to emit it is
//      a compile error, not a silently-missing arm.
//
// An instruction is a value, not a function call: constructing
// `X86BinaryInsn::AddPs` and encoding it are two separate steps, so
// selection (which op maps to which mnemonic) is testable independently of
// encoding (which mnemonic maps to which bytes).

/// A binary SSE mnemonic this backend knows how to encode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum X86BinaryInsn {
    AddPs,
    SubPs,
    MulPs,
    DivPs,
    MinPs,
    MaxPs,
    /// `vcmpps` ordered predicate immediate (`CMP_EQ`/`CMP_LT`/...).
    CmpPs(u8),
    /// Packed i32 lane add (`IAdd`).
    PAddD,
    AndPs,
    OrPs,
}

impl X86BinaryInsn {
    /// Select the mnemonic for `op`, or `None` if this backend has no binary
    /// encoding for it. `None` covers every non-binary `OpKind` (unary,
    /// ternary, structural) as well as ops eliminated by `lowering` before
    /// scheduling (transcendentals, `Dwrt`, `Reduce`, `Gather`) — none of
    /// those can reach `emit_binary` from a correctly-lowered arena, so
    /// `None` reaching a caller is an upstream invariant violation, not a
    /// missing feature.
    pub(crate) const fn select(op: OpKind) -> Option<Self> {
        match op {
            OpKind::Add => Some(Self::AddPs),
            OpKind::Sub => Some(Self::SubPs),
            OpKind::Mul => Some(Self::MulPs),
            OpKind::Div => Some(Self::DivPs),
            OpKind::Min => Some(Self::MinPs),
            OpKind::Max => Some(Self::MaxPs),
            OpKind::Eq => Some(Self::CmpPs(CMP_EQ)),
            OpKind::Ne => Some(Self::CmpPs(CMP_NEQ)),
            OpKind::Lt => Some(Self::CmpPs(CMP_LT)),
            OpKind::Le => Some(Self::CmpPs(CMP_LE)),
            OpKind::Gt => Some(Self::CmpPs(CMP_NLE)),
            OpKind::Ge => Some(Self::CmpPs(CMP_GE)),
            OpKind::IAdd => Some(Self::PAddD),
            OpKind::BitAnd => Some(Self::AndPs),
            OpKind::BitOr => Some(Self::OrPs),
            _ => None,
        }
    }

    /// Encode `dst = dst <mnemonic> src2` (the two-operand SSE setup —
    /// `dst` already holding `src1` — runs in the caller, `emit_binary`).
    /// Exhaustive over the closed `X86BinaryInsn` set: no wildcard arm.
    fn encode(self, code: &mut Vec<u8>, dst: Reg, src2: Reg) {
        match self {
            Self::AddPs => emit_addps(code, dst, src2),
            Self::SubPs => emit_subps(code, dst, src2),
            Self::MulPs => emit_mulps(code, dst, src2),
            Self::DivPs => emit_divps(code, dst, src2),
            Self::MinPs => emit_minps(code, dst, src2),
            Self::MaxPs => emit_maxps(code, dst, src2),
            // Comparisons -> all-ones / all-zeros mask (ordered predicates).
            Self::CmpPs(pred) => emit_cmp_tail(code, dst, src2, pred),
            // Bit-manip primitives: the VEX 3-operand encoders take `dst` as
            // the vvvv source, so this is in-place (dst already holds src1).
            Self::PAddD => emit_vpaddd(code, dst, dst, src2),
            Self::AndPs => emit_vandps(code, dst, dst, src2),
            Self::OrPs => emit_vorps(code, dst, dst, src2),
        }
    }
}

/// Emit binary operation
pub fn emit_binary(code: &mut Vec<u8>, op: OpKind, dst: Reg, src1: Reg, src2: Reg) {
    // SSE is 2-operand, so we may need to move first
    if dst.0 != src1.0 {
        emit_sse_rr(code, &[0x0F, 0x28], dst, src1); // MOVAPS dst, src1
    }

    match X86BinaryInsn::select(op) {
        Some(insn) => insn.encode(code, dst, src2),
        None => unimplemented_op("x86-64", op),
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

// =============================================================================
// Prologue / Epilogue
// =============================================================================

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
/// Emit `jmp rel32` with a zero placeholder; returns the rel32 field offset.
pub fn emit_jmp_rel32(code: &mut Vec<u8>) -> usize {
    code.push(0xE9);
    let pos = code.len();
    code.extend_from_slice(&[0, 0, 0, 0]);
    pos
}

/// Patch a rel32 branch displacement (emitted by the `jcc` forms /
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `X86BinaryInsn::select` is `emit_binary`'s completeness contract made
    /// checkable: every op `crate::emit::coverage::
    /// REQUIRED_BINARY_OPS` lists must select `Some`, or this backend would
    /// panic the moment the scheduler handed it that op. Cheaper and more
    /// precise than the emit-and-catch-panic sweep in `emit/mod.rs`'s
    /// `backend_op_coverage` tests, because it checks the *selection* step
    /// directly instead of inferring "not supported" from a caught panic.
    #[test]
    fn selects_every_required_binary_op() {
        use crate::emit::coverage::REQUIRED_BINARY_OPS;

        let unselected: alloc::vec::Vec<OpKind> = REQUIRED_BINARY_OPS
            .iter()
            .copied()
            .filter(|&op| X86BinaryInsn::select(op).is_none())
            .collect();

        assert!(
            unselected.is_empty(),
            "X86BinaryInsn::select has no mnemonic for: {unselected:?}"
        );
    }

    // The tests below target this file's real API boundary: given operands,
    // what bytes come out. Most of that boundary's byte-construction ORs
    // together bit-disjoint fields (a REX bit, a ModRM.reg nibble in bits
    // 3-5, a ModRM.rm nibble in bits 0-2, ...) — that's what makes ModRM/VEX
    // encoding decodable at all — so `cargo mutants` will keep reporting a
    // `|` → `^` replacement there as missed no matter what a test asserts:
    // OR and XOR of operands that can never share a set bit compute the same
    // byte for every input. That is a real equivalent, not a gap.

    // A test of `Vex { w: true }` stood here and was removed: `Vex::new` is
    // the only construction on any production path and hardcodes `w: false`,
    // so the assertion pinned a state no emitted kernel can reach. Mutation
    // coverage of unreachable internal state is not coverage.
    //
    // The subtraction that follows from it is deliberately NOT taken here,
    // to keep this a test-only change: `Vex::w` is dead as written (read
    // once, at the `(self.w as u8) << 7` in the 3-byte head, and never set),
    // so it could go, with the bit hardcoded to 0. That is an encoder change
    // and wants its own CL.

    /// `emit_movups_store_base` (a raw `[base]`-only store, no displacement)
    /// no longer exists as its own function post-refactor — it is
    /// `emit_movups_store` called with a `NoDisp` address, which is exactly
    /// what these tests now drive it through.
    #[test]
    fn emit_movups_store_omits_rex_for_a_no_disp_address_when_both_registers_are_low() {
        let mut code = Vec::new();
        AsmProgram::from([movups_store(
            Reg(3),
            Mem {
                base: Gpr(3),
                disp: NoDisp,
            },
        )])
        .assemble(&mut code);
        assert_eq!(code, vec![0x0F, 0x11, ((3 & 7) << 3) | 3]);
    }

    #[test]
    fn emit_movups_store_sets_rex_r_for_a_no_disp_address_when_only_the_source_register_is_high() {
        let mut code = Vec::new();
        AsmProgram::from([movups_store(
            Reg(9),
            Mem {
                base: Gpr(3),
                disp: NoDisp,
            },
        )])
        .assemble(&mut code);
        assert_eq!(code, vec![0x44, 0x0F, 0x11, ((9 & 7) << 3) | 3]);
    }

    #[test]
    fn emit_movups_store_sets_rex_b_for_a_no_disp_address_when_only_the_base_register_is_high() {
        let mut code = Vec::new();
        AsmProgram::from([movups_store(
            Reg(2),
            Mem {
                base: Gpr(11),
                disp: NoDisp,
            },
        )])
        .assemble(&mut code);
        assert_eq!(code, vec![0x41, 0x0F, 0x11, ((2 & 7) << 3) | (11 & 7)]);
    }

    #[test]
    fn emit_movups_store_sets_both_rex_bits_for_a_no_disp_address_when_both_registers_are_high() {
        let mut code = Vec::new();
        AsmProgram::from([movups_store(
            Reg(11),
            Mem {
                base: Gpr(14),
                disp: NoDisp,
            },
        )])
        .assemble(&mut code);
        assert_eq!(code, vec![0x45, 0x0F, 0x11, ((11 & 7) << 3) | (14 & 7)]);
    }

    #[test]
    fn emit_load_ptr_from_ctx_masks_both_gprs_into_disjoint_modrm_fields() {
        let mut code = Vec::new();
        emit_load_ptr_from_ctx(&mut code, 3, 2, 96);
        assert_eq!(code, vec![0x48, 0x8B, 0x80 | (3 << 3) | 2, 96, 0, 0, 0]);
    }

    #[test]
    fn emit_movmskps_eax_emits_rex_b_for_a_high_source_register() {
        let mut code = Vec::new();
        emit_movmskps_eax(&mut code, Reg(11));
        assert_eq!(code, vec![0x41, 0x0F, 0x50, 0xC0 | (11 & 7)]);
    }

    #[test]
    fn emit_movmskps_eax_omits_rex_for_a_low_source_register() {
        let mut code = Vec::new();
        emit_movmskps_eax(&mut code, Reg(2));
        assert_eq!(code, vec![0x0F, 0x50, 0xC0 | 2]);
    }

    #[test]
    fn emit_cmp_eax_imm8_emits_the_cmp_opcode_and_the_immediate_byte() {
        let mut code = Vec::new();
        emit_cmp_eax_imm8(&mut code, 0x0F);
        assert_eq!(code, vec![0x83, 0xF8, 0x0F]);
    }
}

// =============================================================================
// The SSE2 `IsaBackend` driver
// =============================================================================

/// The SSE2 half of code generation.
///
/// **This file is where SSE2-specific bugs live, and the only place they
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
    not(all(
        target_arch = "x86_64",
        not(target_feature = "avx2"),
        not(target_feature = "avx512f")
    )),
    allow(dead_code)
)]
pub(crate) mod driver {
    use super::super::*;
    use super::{
        AsmProgram, Imm8, Imm32, Inst, Mem, NoDisp, gpr, movups_load, movups_store, ptr, scaffold,
    };
    use crate::error::CompileError;
    use alloc::vec::Vec;
    use pixelflow_ir::kind::OpKind;

    /// The SSE2 register file (xmm, 128-bit).
    ///
    /// SysV has no callee-saved XMM registers, so every register past the
    /// inputs is fair game — and every one of them is now the allocator's.
    pub(crate) const SSE2_FILE: regalloc::RegisterFile = regalloc::RegisterFile {
        inputs: INPUT_REGS,
        // xmm4-15: twelve of sixteen, which is every register the ABI does
        // not use for an argument. xmm11 and xmm12 are the last two to join —
        // they were `reload`, held out of every kernel's pool so that a
        // spilled operand had somewhere to land and a spilled destination had
        // somewhere to be computed. Both are per-instruction reservations now
        // (`Scratch::reload`), as xmm13 (`select_reload`), xmm14/15 (the
        // gather's) and xmm10 (the sign mask, the select blend, the `MulAdd`
        // stand-in's product) became before them.
        scratch: regalloc::RegSet::range(4, 12),
        // Nothing. This was the last backend holding a register for its own
        // encodings, and the one that held it for a *register* rather than an
        // op: `emit_binary_safe` stashed `right` whenever the allocator chose
        // `dst == right` for a non-commutative binary. It never does — see the
        // invariant `resolve_operands` states — so the case was not a demand
        // to reserve for but a fallback for something that cannot happen, and
        // the three real demands are `temps_for` answers.
        fixed: &[],
        temps_for: super::temps_for,
        // A guard reduces its mask with `movmskps` into the flags, which costs
        // no vector register at all.
        guard_temps: 0,
        vector_bytes: 16,
    }
    .checked();

    /// A slot in the allocated spill frame. Kernels are leaves with no base
    /// pointer, so a slot *is* `rsp + offset`; the `disp32` is what makes the
    /// frame mode able to address a frame the red zone could not.
    const fn frame_slot(offset: u32) -> Mem<Imm32> {
        Mem {
            base: ptr::RSP,
            disp: Imm32(offset as i32),
        }
    }

    /// Map a `FrameLayout` spill offset to a red-zone `[rsp+disp8]` displacement.
    fn x86_redzone_disp(offset: u32) -> Result<i8, CompileError> {
        // Slots live below rsp: offset 0 -> [rsp-16], 16 -> [rsp-32], ...
        // Only called in red-zone mode (the prologue switches to an allocated
        // frame when the layout exceeds the zone), so overflow here is an
        // internal invariant violation, not a kernel-size limit.
        let disp = -(offset as i64 + 16);
        if disp < -128 {
            return Err(CompileError::Internal(
                "red-zone offset exceeded 128 bytes despite frame fitting the zone",
            ));
        }
        Ok(disp as i8)
    }

    /// x86-64 implementation of the shared driver's leaf operations.
    ///
    /// `frame_bytes` is set by `prologue`: 0 = spills fit the 128-byte red zone
    /// (no frame is allocated), otherwise the size of the allocated frame and
    /// spill slots are `[rsp + offset]`.
    pub(crate) struct X86Backend {
        frame_bytes: u32,
        file: regalloc::RegisterFile,
    }

    impl X86Backend {
        pub(crate) fn new(ctx: EmitCtx) -> Self {
            Self {
                frame_bytes: 0,
                file: SSE2_FILE.capped(ctx.max_regs),
            }
        }

        fn spill_store(&self, code: &mut Vec<u8>, src: Reg, offset: u32) {
            match self.red_zone_slot(offset) {
                Some(addr) => AsmProgram::from([movups_store(src, addr)]).assemble(code),
                None => AsmProgram::from([movups_store(src, frame_slot(offset))]).assemble(code),
            }
        }

        fn spill_load(&self, code: &mut Vec<u8>, dst: Reg, offset: u32) {
            match self.red_zone_slot(offset) {
                Some(addr) => AsmProgram::from([movups_load(dst, addr)]).assemble(code),
                None => AsmProgram::from([movups_load(dst, frame_slot(offset))]).assemble(code),
            }
        }

        /// The slot at `offset` as a red-zone address, when the body is in
        /// red-zone mode. `None` means an allocated frame, whose slots are
        /// [`frame_slot`]s — a disp32 away, not a disp8 below `rsp`.
        fn red_zone_slot(&self, offset: u32) -> Option<Mem<Imm8>> {
            if self.frame_bytes != 0 {
                return None;
            }
            let disp = x86_redzone_disp(offset).expect("red-zone mode implies fitting offsets");
            Some(Mem {
                base: ptr::RSP,
                disp: Imm8(disp),
            })
        }
    }

    impl IsaBackend for X86Backend {
        /// rel32 field offset of the branch (uniform for jcc/jmp on x86).
        type Branch = usize;

        fn register_file(&self) -> regalloc::RegisterFile {
            self.file
        }

        fn begin(&mut self, _schedule: &[regalloc::Def]) -> Result<(), CompileError> {
            Ok(()) // x86 const loads are self-contained; no pool.
        }

        fn frame_ready(&mut self, frame_size: u32) {
            // Red zone when it fits (max slot offset frame_size-16 maps to disp
            // -(frame_size) >= -128); otherwise an allocated frame, with slots at
            // [rsp + offset]. Latched here so the body's spill addressing agrees
            // with the prologue emitted afterwards.
            self.frame_bytes = if frame_size <= 128 { 0 } else { frame_size };
        }

        fn emit_plan(
            &mut self,
            code: &mut Vec<u8>,
            plan: &InstructionPlan,
        ) -> Result<(), CompileError> {
            use super::*;
            for reload in &plan.reloads {
                match reload {
                    Reload::FromStack { target, slot } => {
                        self.spill_load(code, *target, slot.offset());
                    }
                    Reload::Const { target, val_bits } => {
                        emit_const(code, *target, f32::from_bits(*val_bits));
                    }
                }
            }
            if let Some((dst, src)) = plan.setup_mov {
                emit_movaps(code, dst, src);
            }
            match &plan.op {
                ResolvedOp::Nop => {}
                ResolvedOp::LoadConst { dst, val_bits } => {
                    emit_const(code, *dst, f32::from_bits(*val_bits));
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
                    emit_shift_imm(code, *op, *dst, *src, *amount);
                }
                ResolvedOp::Gather { dst, idx, slot } => {
                    // No AVX2 here, so no 128-bit vgatherdps: assemble the lanes
                    // from four scalar loads. The context pointer (array of buffer
                    // base pointers) is caller-provided in rdi and never touched by
                    // arithmetic/const emission, so it survives to here; rax/rcx
                    // are caller-saved and unused by the rest of the body.
                    super::emit_gather_scalar(
                        code,
                        *dst,
                        *idx,
                        *slot,
                        super::GatherScratch {
                            base_gpr: 0,  // rax
                            index_gpr: 1, // rcx
                            ctx_gpr: 7,   // rdi
                            idx_lanes: crate::emit::declared_temp(plan.scratch.temp(0)),
                            value: crate::emit::declared_temp(plan.scratch.temp(1)),
                        },
                    );
                }
                ResolvedOp::Uniform { dst, load } => {
                    super::emit_uniform_load(code, *dst, *load);
                }
                ResolvedOp::Binary {
                    op,
                    dst,
                    left,
                    right,
                } => emit_binary(code, *op, *dst, *left, *right),
                ResolvedOp::Select {
                    dst,
                    if_true,
                    if_false,
                } => {
                    // setup_mov already placed the mask in `dst`; blend in place.
                    emit_select(code, *dst, *if_true, *if_false, plan.scratch.temp(0));
                }
                ResolvedOp::FusedMulAdd { dst, a, b } => {
                    // No hardware FMA assumed: `dst` already holds c (setup_mov);
                    // compute a*b in this instruction's temp, then add. The
                    // temp is disjoint from `a`, `b` and `dst` by construction,
                    // and `a` is copied out before any write, so c==a / c==b
                    // are handled.
                    let product = crate::emit::declared_temp(plan.scratch.temp(0));
                    emit_movaps(code, product, *a);
                    emit_binary(code, OpKind::Mul, product, product, *b);
                    emit_binary(code, OpKind::Add, *dst, *dst, product);
                }
                ResolvedOp::DecomposedMulAdd {
                    dst,
                    a,
                    b,
                    c,
                    c_deferred,
                } => {
                    // dst = a*b, reload c (after the multiply, if deferred), dst += c.
                    // Both destructive two-operand forms are safe by the
                    // invariant `resolve_operands` states: this shape is only
                    // chosen when both multiplicands are spilled, so `a` was
                    // reloaded into `dst`, and the add's left operand is `dst`.
                    emit_binary(code, OpKind::Mul, *dst, *a, *b);
                    match c_deferred {
                        Some(DeferredReload::FromStack(slot)) => {
                            self.spill_load(code, *c, slot.offset());
                        }
                        Some(DeferredReload::Const(bits)) => {
                            emit_const(code, *c, f32::from_bits(*bits));
                        }
                        None => {}
                    }
                    emit_binary(code, OpKind::Add, *dst, *dst, *c);
                }
            }
            Ok(())
        }

        fn emit_mov(&mut self, code: &mut Vec<u8>, dst: Reg, src: Reg) {
            super::emit_movaps(code, dst, src);
        }

        fn emit_store(
            &mut self,
            code: &mut Vec<u8>,
            src: Reg,
            offset: u32,
        ) -> Result<(), CompileError> {
            self.spill_store(code, src, offset);
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
                    self.spill_load(code, target, slot.offset());
                    target
                }
            }
        }

        /// `_scratch` is unused: this tier's guard reduces the mask with
        /// `movmskps`/`kortest` into the flags, needing no vector register.
        fn emit_skip_if_all_false(
            &mut self,
            code: &mut Vec<u8>,
            mask_reg: Reg,
            _scratch: Option<Reg>,
        ) -> usize {
            super::emit_movmskps_eax(code, mask_reg);
            super::emit_test_eax(code);
            super::je(code).field() // ZF set when eax == 0 (all lanes false)
        }

        /// `_scratch` is unused: this tier's guard reduces the mask with
        /// `movmskps`/`kortest` into the flags, needing no vector register.
        fn emit_skip_if_all_true(
            &mut self,
            code: &mut Vec<u8>,
            mask_reg: Reg,
            _scratch: Option<Reg>,
        ) -> usize {
            super::emit_movmskps_eax(code, mask_reg);
            super::emit_cmp_eax_imm8(code, 0x0F);
            super::je(code).field() // ZF set when eax == 0xF (all lanes true)
        }

        fn emit_jump(&mut self, code: &mut Vec<u8>) -> usize {
            super::emit_jmp_rel32(code)
        }

        fn patch_branch(&mut self, code: &mut Vec<u8>, branch: usize, target: usize) {
            super::patch_rel32(code, branch, target);
        }

        // SysV: rdi = ctx (read-only in the body's gathers), rsi = out,
        // rdx = groups, rcx = rows, r8 = row-skip bytes, xmm0..3 = x0/y0/z/w.
        // Loop registers: r9 = batch counter, r10 = preserved row count, r11 =
        // row counter; the body's scratch GPRs are rax/rcx (gather, movmskps)
        // — disjoint.

        /// In red-zone mode the body spills *below* `rsp` and allocates
        /// nothing, so the scaffold's slots start at zero rather than above a
        /// frame that does not exist.
        fn body_frame_bytes(&self, _frame_size: u32) -> u32 {
            self.frame_bytes
        }

        fn frame_alloc(&mut self, code: &mut Vec<u8>, bytes: u32) {
            AsmProgram::from([Inst::SubImm32 {
                dst: gpr::RSP,
                imm: Imm32(bytes as i32),
            }])
            .assemble(code);
        }

        fn frame_free(&mut self, code: &mut Vec<u8>, bytes: u32) {
            AsmProgram::from([Inst::AddImm32 {
                dst: gpr::RSP,
                imm: Imm32(bytes as i32),
            }])
            .assemble(code);
        }

        // The scaffold's slots are always at a positive displacement, unlike
        // `emit_store`/`emit_resolve`, which follow the body's frame mode.
        fn slot_store(&mut self, code: &mut Vec<u8>, src: Reg, offset: u32) {
            AsmProgram::from([movups_store(src, frame_slot(offset))]).assemble(code);
        }

        fn slot_load(&mut self, code: &mut Vec<u8>, dst: Reg, offset: u32) {
            AsmProgram::from([movups_load(dst, frame_slot(offset))]).assemble(code);
        }

        /// Preserve the row count away from `rcx`, which the body's gather and
        /// select guards may clobber.
        fn latch_bounds(&mut self, code: &mut Vec<u8>) {
            scaffold::latch_bounds(code);
        }

        fn counter_clear(&mut self, code: &mut Vec<u8>, counter: Counter) {
            scaffold::counter_clear(code, counter);
        }

        fn counter_step(&mut self, code: &mut Vec<u8>, counter: Counter) {
            scaffold::counter_step(code, counter);
        }

        fn branch_if_counter_done(&mut self, code: &mut Vec<u8>, counter: Counter) -> usize {
            scaffold::branch_if_counter_done(code, counter)
        }

        fn store_result(&mut self, code: &mut Vec<u8>, src: Reg) {
            AsmProgram::from([movups_store(
                src,
                Mem {
                    base: scaffold::OUT_PTR,
                    disp: NoDisp,
                },
            )])
            .assemble(code);
        }

        fn advance_out(&mut self, code: &mut Vec<u8>, step: OutStep) {
            scaffold::advance_out(code, step, self.file.vector_bytes);
        }

        fn add_scalar(&mut self, code: &mut Vec<u8>, dst: Reg, scratch: Reg, scalar: f32) {
            super::emit_const(code, scratch, scalar);
            super::emit_binary(code, OpKind::Add, dst, dst, scratch);
        }

        fn emit_ret(&mut self, code: &mut Vec<u8>) {
            AsmProgram::from([Inst::Ret]).assemble(code);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn x86_redzone_disp_negates_the_offset_and_biases_by_the_red_zone_size() {
            assert_eq!(x86_redzone_disp(0), Ok(-16));
            assert_eq!(x86_redzone_disp(16), Ok(-32));
            // 112 is the largest offset production can reach: `frame_ready`
            // picks red-zone mode only for `frame_size <= 128`, and
            // `FrameLayout::resolve` advances by `vector_bytes`, so the last
            // slot sits at `frame_size - 16`. -(112 + 16) == -128, the last
            // value disp8 represents.
            assert_eq!(x86_redzone_disp(112), Ok(-128));
        }

        // A test that `x86_redzone_disp(113)` returns the internal
        // out-of-range error stood here, and was removed for the same reason
        // as the `Vex { w: true }` one below: 113 is not a reachable offset.
        // Offsets advance in `vector_bytes` steps from 0, so they are all
        // multiples of 16, and red-zone mode caps them at 112 besides. The
        // assertion pinned an internal error message rather than any
        // behaviour a kernel can produce. The reachable boundary it was
        // wrapped around — 112 — is kept above, where it belongs.

        // Two tests characterizing `emit_binary_safe` stood here. They were
        // deleted, not ported: main's #1177/#1183 removed the function along
        // with the fixed-scratch-register model they encoded (`X86_SCRATCH`,
        // one reserved `Reg(10)`), which is now an allocator-managed pool
        // (`scratch: RegSet::range(4, 12)`, reached as `plan.scratch.temp(n)`).
        // There is no translation of "stashes `right` into the scratch
        // register" into a world with no such register. The only occurrence of
        // the name left in this file is the past-tense comment above line 1058.

        // `X86Backend::prologue`/`epilogue` — and the `if self.frame_bytes > 0`
        // conditional they gated — were deleted outright by main's #1082
        // ("one kernel ABI, one compile entry"), which replaced them with
        // `frame_alloc`/`frame_free`: unconditional `emit_sub_rsp`/`emit_add_rsp`
        // delegations called by the shared collapse-loop scaffold on a `total`
        // that always includes the scaffold's own coordinate slots and so is
        // never zero. There is no surviving red-zone-omits-the-adjustment
        // branch to test; `frame_ready`'s red-zone bookkeeping now only feeds
        // `red_zone_slot`'s body-spill addressing (covered by
        // `x86_redzone_disp`'s tests above), not whether a prologue/epilogue is
        // emitted at all. Removed rather than rewritten against a coincidence.
    }
}

// =============================================================================
// The SysV collapse-loop scaffold
// =============================================================================

/// The general-register half of the collapse loop, shared by every x86 vector
/// width.
///
/// Counters, bounds and the output pointer are the same registers stepped by
/// the same instructions whether the body is SSE2, AVX2 or AVX-512 — these are
/// general-register ops, and the width only reaches them as the batch stride.
pub(in crate::emit) mod scaffold {
    use super::gpr::*;
    use super::ptr;
    use super::{AsmProgram, Counter, Gpr, Imm8, Inst, OutStep, PtrReg};
    use alloc::vec::Vec;

    /// The output pointer the scaffold writes through.
    pub(in crate::emit) const OUT_PTR: PtrReg = ptr::RSI;

    /// The register each loop counter lives in.
    const fn counter_reg(counter: Counter) -> Gpr {
        match counter {
            Counter::Batch => R9,
            Counter::Row => R11,
        }
    }

    /// The register each counter is compared against: the caller's group count
    /// arrives in `rdx`, and the row count is latched out of `rcx`.
    const fn bound_reg(counter: Counter) -> Gpr {
        match counter {
            Counter::Batch => RDX,
            Counter::Row => R10,
        }
    }

    /// Preserve the row count away from `rcx`, which the body's gather and
    /// select guards may clobber.
    #[inline(always)]
    pub(in crate::emit) fn latch_bounds(code: &mut Vec<u8>) {
        AsmProgram::from([Inst::Mov { dst: R10, src: RCX }]).assemble(code);
    }

    #[inline(always)]
    pub(in crate::emit) fn counter_clear(code: &mut Vec<u8>, counter: Counter) {
        let r = counter_reg(counter);
        AsmProgram::from([Inst::Xor { dst: r, src: r }]).assemble(code);
    }

    #[inline(always)]
    pub(in crate::emit) fn counter_step(code: &mut Vec<u8>, counter: Counter) {
        AsmProgram::from([Inst::Inc {
            dst: counter_reg(counter),
        }])
        .assemble(code);
    }

    /// The loop's exit test: unsigned `counter >= bound`.
    #[inline(always)]
    pub(in crate::emit) fn branch_if_counter_done(code: &mut Vec<u8>, counter: Counter) -> usize {
        AsmProgram::from([Inst::Cmp {
            lhs: counter_reg(counter),
            rhs: bound_reg(counter),
        }])
        .assemble(code);
        super::jae(code).field()
    }

    #[inline(always)]
    pub(in crate::emit) fn advance_out(code: &mut Vec<u8>, step: OutStep, vector_bytes: u32) {
        match step {
            OutStep::Batch => {
                AsmProgram::from([Inst::AddImm8 {
                    dst: RSI,
                    imm: Imm8(vector_bytes as i8),
                }])
                .assemble(code);
            }
            OutStep::RowSkip => {
                AsmProgram::from([Inst::Add { dst: RSI, src: R8 }]).assemble(code);
            }
        }
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
// scalar half of a gather — which is why the vocabulary below is nine
// instructions rather than an assembler.

/// SysV argument and scratch registers the emitted kernels use.
pub mod gpr {
    use super::Gpr;

    /// Scratch / `movmskps` destination.
    pub const RAX: Gpr = Gpr(0);
    /// Scratch aliases for RAX.
    pub const AX: Gpr = RAX;
    pub const RX: Gpr = RAX;
    /// 4th integer argument; the collapse loop's row count on entry.
    pub const RCX: Gpr = Gpr(1);
    /// 3rd integer argument: group count.
    pub const RDX: Gpr = Gpr(2);
    /// 2nd integer argument: the output pointer, advanced per batch.
    pub const RSI: Gpr = Gpr(6);
    /// 1st integer argument: the context pointer — the array of bound buffer
    /// bases and the uniform block a gather or a uniform load reads its
    /// slot from. Read-only for the whole kernel.
    pub const RDI: Gpr = Gpr(7);
    /// The stack pointer.
    pub const RSP: Gpr = Gpr(4);
    /// 5th integer argument: row-skip in bytes.
    pub const R8: Gpr = Gpr(8);
    /// Inner (batch) loop counter.
    pub const R9: Gpr = Gpr(9);
    /// Preserved copy of the row count, away from `rcx`.
    pub const R10: Gpr = Gpr(10);
    /// Outer (row) loop counter.
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

impl From<MovLoadPtr> for Inst {
    #[inline(always)]
    fn from(m: MovLoadPtr) -> Self {
        Inst::MovLoadPtr(m)
    }
}

impl AsmInsn for Inst {
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

/// A branch whose 32-bit displacement is not known yet.
///
/// Holds the offset of the displacement field, so the target can be filled in
/// once its address is known. Returned by [`jae`] and [`jmp`] so a caller
/// cannot emit a branch and forget it needs patching.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[must_use = "an unpatched branch jumps to itself"]
pub struct Rel32(usize);

impl Rel32 {
    /// Point the branch at `target`, a byte offset into the same buffer.
    #[inline(always)]
    pub fn patch(self, code: &mut [u8], target: usize) {
        let rel = (target as i32) - (self.0 as i32 + 4);
        code[self.0..self.0 + 4].copy_from_slice(&rel.to_le_bytes());
    }

    /// The offset of the displacement field.
    #[must_use]
    #[inline(always)]
    pub const fn field(self) -> usize {
        self.0
    }
}

/// `jae rel32` — taken when the previous [`cmp`] found `lhs >= rhs` unsigned.
#[inline(always)]
pub fn jae(code: &mut Vec<u8>) -> Rel32 {
    jcc(code, 0x83)
}

/// `je rel32` — taken when the previous compare or test set ZF.
#[inline(always)]
pub fn je(code: &mut Vec<u8>) -> Rel32 {
    jcc(code, 0x84)
}

/// `jc rel32` — taken when the previous operation set CF.
#[inline(always)]
pub fn jc(code: &mut Vec<u8>) -> Rel32 {
    jcc(code, 0x82)
}

/// The shared body of the `jcc rel32` forms. Private: a condition is chosen by
/// calling the mnemonic, never by handing a byte to a generic emitter.
#[inline(always)]
fn jcc(code: &mut Vec<u8>, cc: u8) -> Rel32 {
    code.extend_from_slice(&[0x0F, cc]);
    let at = code.len();
    code.extend_from_slice(&[0, 0, 0, 0]);
    Rel32(at)
}

/// `jmp rel32`
#[inline(always)]
pub fn jmp(code: &mut Vec<u8>) -> Rel32 {
    code.push(0xE9);
    let at = code.len();
    code.extend_from_slice(&[0, 0, 0, 0]);
    Rel32(at)
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

    /// Each encoding checked against the Intel SDM's form for that mnemonic.
    /// These are the exact bytes the collapse-loop scaffold used to spell
    /// inline, which is what makes the replacement provably byte-identical.
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

    /// A branch reports where its displacement lives, and patching aims it at
    /// a byte offset in the same buffer.
    #[test]
    fn branches_patch_relative_to_the_next_instruction() {
        let mut c = Vec::new();
        let br = jmp(&mut c);
        assert_eq!(c.len(), 5, "E9 + rel32");
        assert_eq!(br.field(), 1);
        // Jump forward to the end of a 16-byte buffer.
        c.resize(16, 0x90);
        br.patch(&mut c, 16);
        assert_eq!(
            &c[1..5],
            &(16i32 - 5).to_le_bytes(),
            "rel is from the next insn"
        );

        let mut c = Vec::new();
        let br = jae(&mut c);
        assert_eq!(c[..2], [0x0F, 0x83]);
        // A backward jump is negative.
        c.resize(10, 0x90);
        br.patch(&mut c, 0);
        assert_eq!(&c[2..6], &(-6i32).to_le_bytes());
    }

    /// One instruction, three bases: `movups [rsp+8]`, `[rax+8]` and `[r10+8]`
    /// differ only in ModRM.rm (plus the SIB `rsp` implies and the REX.B `r10`
    /// does). That is why the base is an operand and not a name suffix.
    #[test]
    fn the_base_register_is_an_operand() {
        let store = |base| {
            asm(|c| {
                AsmProgram::from([movups_store(
                    Reg(1),
                    Mem {
                        base,
                        disp: Imm8(8),
                    },
                )])
                .assemble(c);
            })
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
        let d8 = asm(|c| {
            AsmProgram::from([movups_load(
                Reg(0),
                Mem {
                    base: RSP,
                    disp: Imm8(16),
                },
            )])
            .assemble(c);
        });
        let d32 = asm(|c| {
            AsmProgram::from([movups_load(
                Reg(0),
                Mem {
                    base: RSP,
                    disp: Imm32(16),
                },
            )])
            .assemble(c);
        });
        assert_eq!(d8, [0x0F, 0x10, 0x44, 0x24, 0x10], "mod=01, disp8");
        assert_eq!(
            d32,
            [0x0F, 0x10, 0x84, 0x24, 0x10, 0x00, 0x00, 0x00],
            "mod=10, disp32"
        );

        let bare = asm(|c| {
            AsmProgram::from([movups_load(
                Reg(0),
                Mem {
                    base: RSI,
                    disp: NoDisp,
                },
            )])
            .assemble(c);
        });
        let zero = asm(|c| {
            AsmProgram::from([movups_load(
                Reg(0),
                Mem {
                    base: RSI,
                    disp: Imm8(0),
                },
            )])
            .assemble(c);
        });
        assert_eq!(bare, [0x0F, 0x10, 0x06], "mod=00 is its own mode");
        assert_eq!(zero, [0x0F, 0x10, 0x46, 0x00], "and a byte longer than it");
    }

    /// Load and store are the same encoding one opcode byte apart; the address
    /// is on whichever side of the transfer the mnemonic puts it.
    #[test]
    fn load_and_store_differ_only_in_the_opcode() {
        let addr = Mem {
            base: RSP,
            disp: Imm32(64),
        };
        let mut load = asm(|c| AsmProgram::from([movups_load(Reg(9), addr)]).assemble(c));
        let store = asm(|c| AsmProgram::from([movups_store(Reg(9), addr)]).assemble(c));
        assert_eq!(load[2], 0x10);
        assert_eq!(store[2], 0x11);
        load[2] = 0x11;
        assert_eq!(load, store);
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
