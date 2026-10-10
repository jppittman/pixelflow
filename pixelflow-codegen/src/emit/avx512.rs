//! x86-64 AVX-512 (EVEX) JIT encoder — 512-bit, 16-lane `zmm` kernels.
//!
//! The widest of the x86-64 tiers (`crate::isa`), above the AVX2 VEX
//! encoders (`avx2.rs`, 256-bit). It targets the full `zmm0..zmm31` register
//! file via EVEX, so it can also use the extended registers (`zmm16..31`)
//! that VEX cannot reach; the general-register half of every kernel is
//! `x86_64.rs`'s, shared with AVX2.
//!
//! Scope: arithmetic, FMA, sqrt/recip/rsqrt, min/max, bitwise, comparisons,
//! `If`, constant broadcast, the integer bit-manipulation atoms
//! (`IAdd`/`BitAnd`/`BitOr`/`TruncToInt`/`IntToFloat`), and `ShiftImm` (see
//! `emit_shift_imm`) — so the exp/log lowering reaches this backend intact.
//! Comparisons go through the k-register class (`vcmpps` -> `vpmovm2d`, see
//! `emit_compare` below) so every downstream consumer still sees an ordinary
//! all-ones/all-zeros vector, exactly like every other backend — the DAG's
//! values are still all vectors, even though the k-register itself is now an
//! allocated `RegisterFile::mask_scratch` reservation rather than a hardcoded
//! transient. Note `vpmovm2d` is AVX-512**DQ**, not F: an F-only part would
//! fault on any kernel containing a comparison.
//!
//! Transcendentals themselves are still a separate lowering stage; ops with no
//! rule here are refused up front rather than mis-emitted.
//!
//! Spills use a real stack frame (a `zmm` is 64 bytes — far past the 128-byte
//! red zone).

use super::x86_64;
use super::x86_64::{
    Alu, Direction, Disp, Imm32, Lanewise, Mem, NoDisp, Pred, Rounding, Truncate, frame_slot,
};
use super::{
    AsmInsn, AsmProgram, EncodedInst, Flags, Gpr, Integer, KReg, Opmask, Physical, Pointer, PtrReg,
    Reg, Stage, Vector, unimplemented_op,
};
use crate::error::CompileError;
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

/// The identity of one EVEX instruction: opcode map, mandatory prefix, W
/// bit, opcode byte, vector length, and the writemask. This is *which
/// instruction* — it is constant per mnemonic, so each mnemonic below states
/// it exactly once and the operand form (`rrr`/`rm`) supplies the per-call
/// parts.
///
/// The 256-bit twin is `avx2::Vex`.
#[derive(Clone, Copy)]
struct Evex {
    map: Map,
    pp: Pp,
    w: bool,
    /// `EVEX.L'L`: `10` for the `zmm` form, `00` for the few `xmm`-only
    /// instructions this tier needs (`vmovq`, `vpinsrq`).
    ll: u8,
    /// `EVEX.aaa`: the writemask register, or 0 for none. Merge-masking
    /// (`z = 0`), which for a store means the masked-off lanes are left in
    /// memory untouched.
    aaa: u8,
    opcode: u8,
}

/// `EVEX.L'L` for a 512-bit operation.
const LL_512: u8 = 0b10;

impl Evex {
    const fn new(map: Map, pp: Pp, opcode: u8) -> Self {
        Self {
            map,
            pp,
            w: false,
            ll: LL_512,
            aaa: 0,
            opcode,
        }
    }
    /// The 128-bit (`L'L = 00`) form of this instruction.
    const fn xmm(self) -> Self {
        Self { ll: 0, ..self }
    }
    /// The 64-bit-operand (`EVEX.W = 1`) form of this instruction.
    const fn w1(self) -> Self {
        Self { w: true, ..self }
    }
    /// This instruction under writemask `k`.
    const fn masked(self, k: KReg) -> Self {
        Self { aaa: k.0, ..self }
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
    fn rm<D: Disp>(self, reg: u8, addr: Mem<Physical, D>) -> EncodedInst {
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
        x86_64::mem_operand_into(&mut inst, reg, addr.base.0, addr.disp);
        inst
    }

    /// `op zmmREG, [base + index*4]` — the SIB form with a scaled index,
    /// which a broadcast load reads one element of a plane through. X is
    /// the index's high bit here, inverted like R and B.
    fn rm_scaled4(self, reg: u8, base: Gpr, index: Gpr) -> EncodedInst {
        let mut inst = EncodedInst::new();
        let r = ((reg >> 3) & 1) ^ 1;
        let rp = ((reg >> 4) & 1) ^ 1;
        let b = ((base.0 >> 3) & 1) ^ 1;
        let x = ((index.0 >> 3) & 1) ^ 1;

        self.prefix_into(
            &mut inst,
            (r << 7) | (x << 6) | (b << 5) | (rp << 4),
            0x0F,
            1,
        );
        x86_64::scaled4_operand_into(&mut inst, reg, base.0, index.0);
        inst
    }

    /// `op zmmREG{k}, [base + zmm_index*4]` — the VSIB form a gather
    /// addresses through. The index's high bits ride in X and V', inverted
    /// like R and B, and `vvvv` is unused.
    fn vsib_scaled4(self, reg: u8, base: Gpr, index: Reg) -> EncodedInst {
        let mut inst = EncodedInst::new();
        let r = ((reg >> 3) & 1) ^ 1;
        let rp = ((reg >> 4) & 1) ^ 1;
        let b = ((base.0 >> 3) & 1) ^ 1;
        let x = ((index.0 >> 3) & 1) ^ 1;
        let vp = ((index.0 >> 4) & 1) ^ 1;

        self.prefix_into(
            &mut inst,
            (r << 7) | (x << 6) | (b << 5) | (rp << 4),
            0x0F,
            vp,
        );
        x86_64::vsib4_operand_into(&mut inst, reg, base.0, index.0);
        inst
    }

    /// The 4-byte EVEX prefix plus the opcode byte, shared by every form.
    /// `reg_ext` is the assembled `R X B R'` nibble of P0; `vvvv`/`vp` are the
    /// extra-source fields. Every one of them is already inverted by the
    /// caller, as the encoding requires.
    fn prefix_into(self, inst: &mut EncodedInst, reg_ext: u8, vvvv: u8, vp: u8) {
        inst.push(0x62);
        inst.push(reg_ext | (self.map as u8));
        inst.push(((self.w as u8) << 7) | (vvvv << 3) | (1 << 2) | (self.pp as u8));
        // z=0 (merge), L'L, b(roadcast)=0, V', aaa.
        inst.push((self.ll << 5) | (vp << 3) | self.aaa);
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

impl Alu {
    const fn evex(self) -> Evex {
        match self {
            Alu::Add => Evex::m0f(0x58),
            Alu::Sub => Evex::m0f(0x5C),
            Alu::Mul => Evex::m0f(0x59),
            Alu::Div => Evex::m0f(0x5E),
            Alu::Min => Evex::m0f(0x5D),
            Alu::Max => Evex::m0f(0x5F),
            Alu::And => Evex::m0f(0x54),
            Alu::AndNot => Evex::m0f(0x55),
            Alu::Or => Evex::m0f(0x56),
            Alu::Xor => Evex::m0f(0x57),
            Alu::IAdd => Evex::m0f_66(0xFE),
        }
    }
}

impl Lanewise {
    /// `rsqrt` and `recip` are AVX-512F's `vrsqrt14ps` and `vrcp14ps`: EVEX
    /// has no `0F 52`/`0F 53`, and ~2^-14 relative error matches `Recip`'s
    /// "approximate reciprocal" contract on every other backend.
    const fn evex(self) -> Evex {
        match self {
            Lanewise::Sqrt => Evex::m0f(0x51),
            Lanewise::Rsqrt => Evex::m0f38_66(0x4E),
            Lanewise::Recip => Evex::m0f38_66(0x4C),
            Lanewise::ToInt => Evex::m0f_f3(0x5B),
            Lanewise::FromInt => Evex::m0f(0x5B),
            Lanewise::WidenBytes => Evex::m0f38_66(0x31),
        }
    }
}

/// `vpternlogd`'s truth table for `A ? B : C` per bit, with `A` the
/// destination (the mask), `B` the first source and `C` the second.
const TERNLOG_SELECT: u8 = 0xCA;

/// An AVX-512 instruction: EVEX, and the VEX-encoded mask-register forms.
///
/// Generic over what its operands are ([`Stage`]), like [`x86_64::Gp`] and
/// AVX2's `Inst`. EVEX is three-operand and non-destructive, so an operand may
/// be the register the result is written to; the instructions below whose
/// destination is also a source say so with a `Tie`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Inst<S: Stage> {
    /// `op dst, a, b`
    Alu {
        op: Alu,
        dst: S::Write<Vector>,
        a: S::Read<Vector>,
        b: S::Read<Vector>,
    },
    /// `op dst, src`
    Unary {
        op: Lanewise,
        dst: S::Write<Vector>,
        src: S::Read<Vector>,
    },
    /// `vrndscaleps dst, src, mode`: scale 0, so an integer. (Opcode `08` is
    /// packed-single; `09` is packed-double and needs `W1`.)
    Round {
        mode: Rounding,
        dst: S::Write<Vector>,
        src: S::Read<Vector>,
    },
    /// `vpslld`/`vpsrld dst, src, amount`
    Shift {
        direction: Direction,
        dst: S::Write<Vector>,
        src: S::Read<Vector>,
        amount: u8,
    },
    /// `vfmadd231ps acc, a, b`: `acc = a·b + acc`, one rounding.
    Fma231 {
        acc: S::Tie<Vector>,
        a: S::Read<Vector>,
        b: S::Read<Vector>,
    },
    /// `vpternlogd dst, if_true, if_false, 0xCA`: `dst = dst ? if_true :
    /// if_false` per bit, `dst` holding an all-ones/all-zeros mask going in.
    Blend {
        dst: S::Tie<Vector>,
        if_true: S::Read<Vector>,
        if_false: S::Read<Vector>,
    },
    /// `vmovaps dst, src`
    Mov {
        dst: S::Write<Vector>,
        src: S::Read<Vector>,
    },
    /// `vcvttss2si dst, src`: lane 0, truncated to a 64-bit integer. EVEX
    /// rather than VEX so the source may be `zmm16..31`.
    Cvtt {
        dst: S::Write<Integer>,
        src: S::Read<Vector>,
    },
    /// `vcvttss2si dst, [src]`: the first word of a slot.
    CvttMem {
        dst: S::Write<Integer>,
        src: Mem<S, Imm32>,
    },
    /// `vmovq dst, src`: eight bytes into the low lanes, the rest zeroed.
    Movq {
        dst: S::Write<Vector>,
        src: S::Read<Integer>,
    },
    /// `vpinsrq dst, dst, src, 1`: eight bytes into the high half of the low
    /// 128 bits.
    InsertHigh {
        dst: S::Tie<Vector>,
        src: S::Read<Integer>,
    },
    /// `vcmpps dst, a, b, pred`: a bit where it holds, clear where it does not.
    CmpK {
        pred: Pred,
        dst: S::Write<Opmask>,
        a: S::Read<Vector>,
        b: S::Read<Vector>,
    },
    /// `vpmovm2d dst, k`: an all-ones lane where the bit is set, all-zero
    /// where it is not. AVX-512**DQ**, not F: an F-only part would fault.
    Movm2d {
        dst: S::Write<Vector>,
        k: S::Read<Opmask>,
    },
    /// `vptestmd dst, a, b`: a bit where the lanes' AND is nonzero.
    Ptestm {
        dst: S::Write<Opmask>,
        a: S::Read<Vector>,
        b: S::Read<Vector>,
    },
    /// `kortestw k, k`: ZF iff no bit is set, CF iff all sixteen are.
    KorTest {
        flags: S::Write<Flags>,
        k: S::Read<Opmask>,
    },
    /// `kmovw dst, src`: the low sixteen bits of a general register.
    Kmovw {
        dst: S::Write<Opmask>,
        src: S::Read<Integer>,
    },
    /// `vgatherdps dst{mask}, [base + index*4]`: one `f32` per lane whose
    /// `mask` bit is set. The instruction clears the bits as it completes
    /// lanes, and `#UD`s if `dst` and `index` are one register.
    Gather {
        dst: S::Early<Vector>,
        base: S::Read<Pointer>,
        index: S::Read<Vector>,
        mask: S::Tie<Opmask>,
    },
    /// `vmovups [dst]{mask}, src`: the lanes whose bit is set, the rest of
    /// memory left untouched.
    StoreMasked {
        dst: Mem<S, NoDisp>,
        src: S::Read<Vector>,
        mask: S::Read<Opmask>,
    },
    /// `vmovups dst, [src]`: a slot.
    Load {
        dst: S::Write<Vector>,
        src: Mem<S, Imm32>,
    },
    /// `vmovups [dst], src`: a slot.
    Store {
        dst: Mem<S, Imm32>,
        src: S::Read<Vector>,
    },
    /// `vmovups [dst], src` with no displacement: a whole batch to the
    /// address the store's arithmetic computed.
    StoreBatch {
        dst: Mem<S, NoDisp>,
        src: S::Read<Vector>,
    },
    /// `vbroadcastss dst, [src]`: one `f32`, a pool entry or a uniform, into
    /// every lane. A full `disp32`, never EVEX's compressed `disp8`, which
    /// scales the byte by the tuple's element size.
    Broadcast {
        dst: S::Write<Vector>,
        src: Mem<S, Imm32>,
    },
    /// `vbroadcastss dst, [base + index*4]`: one element of a plane.
    BroadcastIndexed {
        dst: S::Write<Vector>,
        base: S::Read<Pointer>,
        index: S::Read<Integer>,
    },
}

impl Inst<Physical> {
    fn encode(self) -> EncodedInst {
        match self {
            Inst::Alu { op, dst, a, b } => op.evex().rrr(dst.0, a.0, b.0),
            Inst::Unary { op, dst, src } => op.evex().rrr(dst.0, UNUSED_VVVV, src.0),
            Inst::Round { mode, dst, src } => {
                Evex::m0f3a_66(0x08)
                    .imm(mode as u8)
                    .rrr(dst.0, UNUSED_VVVV, src.0)
            }
            // The destination is `vvvv` and the `/digit` is `reg`.
            Inst::Shift {
                direction,
                dst,
                src,
                amount,
            } => Evex::m0f_66(0x72)
                .imm(amount)
                .rrr(direction as u8, dst.0, src.0),
            Inst::Fma231 { acc, a, b } => Evex::m0f38_66(0xB8).rrr(acc.0, a.0, b.0),
            Inst::Blend {
                dst,
                if_true,
                if_false,
            } => Evex::m0f3a_66(0x25)
                .imm(TERNLOG_SELECT)
                .rrr(dst.0, if_true.0, if_false.0),
            Inst::Mov { dst, src } => Evex::m0f(0x28).rrr(dst.0, UNUSED_VVVV, src.0),
            Inst::Cvtt { dst, src } => Evex::m0f_f3(0x2C).w1().rrr(dst.0, UNUSED_VVVV, src.0),
            Inst::CvttMem { dst, src } => Evex::m0f_f3(0x2C).w1().rm(dst.0, src),
            Inst::Movq { dst, src } => Evex::m0f_66(0x6E).w1().xmm().rrr(dst.0, UNUSED_VVVV, src.0),
            Inst::InsertHigh { dst, src } => Evex::m0f3a_66(0x22)
                .w1()
                .xmm()
                .imm(1)
                .rrr(dst.0, dst.0, src.0),
            Inst::CmpK { pred, dst, a, b } => Evex::m0f(0xC2).imm(pred as u8).rrr(dst.0, a.0, b.0),
            Inst::Movm2d { dst, k } => Evex::m0f38_f3(0x38).rrr(dst.0, UNUSED_VVVV, k.0),
            Inst::Ptestm { dst, a, b } => Evex::m0f38_66(0x27).rrr(dst.0, a.0, b.0),
            // `VEX.L0.0F.W0 98 /r`, both operands the one mask register.
            Inst::KorTest { flags: (), k } => {
                EncodedInst::from_slice(&[0xC5, 0xF8, 0x98, 0xC0 | (k.0 << 3) | k.0])
            }
            // `VEX.L0.0F.W0 92 /r`, in the three-byte prefix.
            Inst::Kmovw { dst, src } => {
                // B̄ is inverted: set when the source needs no extension bit.
                let no_extension = if src.0 < 8 { 0x20 } else { 0x00 };
                let modrm = 0xC0 | ((dst.0 & 7) << 3) | (src.0 & 7);
                EncodedInst::from_slice(&[0xC4, 0xC1 | no_extension, 0x78, 0x92, modrm])
            }
            Inst::Gather {
                dst,
                base,
                index,
                mask,
            } => {
                debug_assert!(dst != index, "vgatherdps: dst and index must differ");
                // `base` is never `rbp`/`r13` (the pointer pool is `r9`-`r11`),
                // so the SIB's no-base encoding is unreachable.
                Evex::m0f38_66(0x92)
                    .masked(mask)
                    .vsib_scaled4(dst.0, base.as_gpr(), index)
            }
            Inst::StoreMasked { dst, src, mask } => Evex::m0f(0x11).masked(mask).rm(src.0, dst),
            Inst::Load { dst, src } => Evex::m0f(0x10).rm(dst.0, src),
            Inst::Store { dst, src } => Evex::m0f(0x11).rm(src.0, dst),
            Inst::StoreBatch { dst, src } => Evex::m0f(0x11).rm(src.0, dst),
            Inst::Broadcast { dst, src } => Evex::m0f38_66(0x18).rm(dst.0, src),
            Inst::BroadcastIndexed { dst, base, index } => {
                Evex::m0f38_66(0x18).rm_scaled4(dst.0, base.as_gpr(), index)
            }
        }
    }
}

impl AsmInsn for Inst<Physical> {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        self.encode().emit_into(code);
    }
}

impl Truncate for Inst<Physical> {
    fn from_xmm(dst: Gpr, src: Reg) -> Self {
        Inst::Cvtt { dst, src }
    }

    fn from_slot(dst: Gpr, src: Mem<Physical, Imm32>) -> Self {
        Inst::CvttMem { dst, src }
    }
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
/// blends an `If` with no temporary.
fn temps_for(op: &super::ScheduledOp) -> u8 {
    use super::ScheduledOp;
    match op {
        ScheduledOp::Unary(OpKind::Neg | OpKind::Abs, _) => 1,
        // The gather's truncated-index lanes and its destination.
        ScheduledOp::Gather(..) => 2,
        // A surviving fold's own loop: two transient registers for the trip
        // test and the accumulate — see `emit_scope`'s `Reduce` arm. The
        // binder and the accumulator are the fold's roots, placed by the
        // allocator, not scratch.
        ScheduledOp::Reduce(..) => super::regalloc::Scratch::REDUCE_TEMPS as u8,
        _ => 0,
    }
}

/// How many GPRs this backend's encoding of `op` needs beyond
/// [`regalloc::RegisterFile::gpr_ctx`](crate::emit::regalloc::RegisterFile::gpr_ctx).
///
/// `Uniform` needs none: the base it addresses is a pointer value the
/// allocator carries. `Gather` takes its indices as a vector, so its one is
/// for the writemask's all-ones. `Broadcast` needs one for its index, since it
/// addresses the element through a SIB. A `Write` converts its row and column
/// into one each before combining them into the address, and the remainder's
/// writemask rides in through the second once the address is done with it;
/// the iota carries each eight bytes in through one.
fn gpr_temps_for(op: &super::ScheduledOp) -> u8 {
    use super::ScheduledOp;
    match op {
        ScheduledOp::Write { .. } => 2,
        ScheduledOp::Broadcast(..) | ScheduledOp::Lanes(_) | ScheduledOp::Gather(..) => 1,
        _ => 0,
    }
}

/// How many mask registers this backend's encoding of `op` needs.
///
/// A comparison's `vcmpps` destination — `k1`, chosen by hand before this
/// work and now a `RegisterFile::mask_scratch` reservation — a gather's
/// writemask, and a remainder store's writemask. Every other op either has no
/// mask (arithmetic) or reads the mask as an ordinary vector (`If`).
fn mask_temps_for(op: &super::ScheduledOp) -> u8 {
    use super::ScheduledOp;
    match op {
        ScheduledOp::Binary(op_kind, ..) if is_compare(*op_kind) => 1,
        ScheduledOp::Gather(..) => 1,
        ScheduledOp::Write { lanes, .. } if *lanes < 16 => 1,
        _ => 0,
    }
}

// =============================================================================
// The store, and the iota
// =============================================================================

/// Set the gather's writemask `mask` to all-ones: `mov ones, 0xFFFF`, then
/// `kmovw mask, ones` in the two-byte VEX prefix, which [`Inst::Kmovw`] (the
/// three-byte form) does not encode. A gather clears the bits it completes, so
/// this runs before each one, and it clobbers `ones`, the gather's declared
/// GPR temp. A `kxnorw k1, k1, k1` would need neither a GPR nor a second
/// instruction, but it reads the mask the previous gather is still clearing,
/// which chains each gather behind the last. C2 allocates the mask and picks.
fn set_gather_mask(code: &mut Vec<u8>, mask: KReg, ones: Gpr) {
    assert!(
        ones.0 < 8,
        "the two-byte VEX prefix has no bit to extend r/m"
    );
    x86_64::Gp::MovImm32 {
        dst: ones,
        imm: 0xFFFF,
    }
    .emit_into(code);
    // ModRM `11 mask ones`: the mask in the reg field, `ones` in r/m.
    let modrm = 0xC0 | (mask.0 << 3) | ones.0;
    EncodedInst::from_slice(&[0xC5, 0xF8, 0x92, modrm]).emit_into(code);
}

/// The bytes `0..8` and `8..16`, little end first: what two `movabs` carry
/// in for `vpmovzxbd` to widen into the iota.
const IOTA_BYTES: [u64; 2] = [0x0706_0504_0302_0100, 0x0F0E_0D0C_0B0A_0908];

/// vmovaps zmmDST, zmmSRC — register copy (EVEX.512.0F.W0 28 /r).
fn emit_mov(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    if dst.0 == src.0 {
        return;
    }
    Inst::Mov { dst, src }.emit_into(code);
}

/// `dst = splat(val)`: `vbroadcastss zmm, [pool]` (EVEX.512.66.0F38.W0 18
/// /r), one instruction from the kernel's constant pool. Zero is `vxorps`.
///
/// The pool's operand is a full `disp32`, not EVEX's compressed `disp8`: the
/// compressed form scales the byte by the tuple element size (4 for a
/// `vbroadcastss` scalar source), and `disp32` is never scaled.
///
/// # Errors
///
/// [`CompileError::BudgetExceeded`] when the pool outgrows a `disp32`
/// ([`x86_64::ConstPool::operand`]).
fn emit_const(
    code: &mut Vec<u8>,
    dst: Reg,
    val: f32,
    pool: &mut x86_64::ConstPool,
) -> Result<(), CompileError> {
    let bits = val.to_bits();
    if bits == 0 {
        Inst::Alu {
            op: Alu::Xor,
            dst,
            a: dst,
            b: dst,
        }
        .emit_into(code);
        return Ok(());
    }
    Inst::Broadcast {
        dst,
        src: pool.operand(bits)?,
    }
    .emit_into(code);
    Ok(())
}

/// `dst = splat(base[offset])` at 512 bits: `vbroadcastss zmm<dst>, [base +
/// 4*offset]` (EVEX.512.66.0F38.W0 18 /r). A full `disp32`, as
/// [`emit_const`]'s is, so EVEX's compressed-`disp8` scaling never enters
/// into it. `base` is the block's address, wherever the allocator keeps
/// that pointer value.
///
/// # Errors
///
/// [`CompileError::BudgetExceeded`] when the element lies past a `disp32`
/// ([`x86_64::block_element`]).
fn emit_uniform_load(
    code: &mut Vec<u8>,
    dst: Reg,
    base: PtrReg,
    offset: u64,
) -> Result<(), CompileError> {
    let element = x86_64::block_element(base, offset)?;
    Inst::Broadcast { dst, src: element }.emit_into(code);
    Ok(())
}

/// `dst = splat(base[idx])` at 512 bits, the index being the same in every
/// lane of `idx`: `vcvttss2si index, xmm<idx>`, `vbroadcastss zmm<dst>,
/// [base + index*4]` (EVEX.512.66.0F38.W0 18 /r). Two instructions, no
/// writemask, no `vgatherdps`. See [`x86_64::BroadcastGprs`] for the
/// register contract; `dst` may alias `idx`, since the index is in a GPR
/// before `dst` is written.
fn emit_broadcast_load(code: &mut Vec<u8>, dst: Reg, idx: Reg, gprs: x86_64::BroadcastGprs) {
    AsmProgram::from([
        Inst::Cvtt {
            dst: gprs.index,
            src: idx,
        },
        Inst::BroadcastIndexed {
            dst,
            base: gprs.base,
            index: gprs.index,
        },
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
/// EVEX is 3-operand and non-destructive: `src1`/`src2` are never clobbered
/// and may alias `dst`.
/// Returns `Err` for ops not in the Stage-1 arithmetic subset.
fn emit_binary(code: &mut Vec<u8>, op: OpKind, dst: Reg, a: Reg, b: Reg) {
    let op = match op {
        OpKind::Add => Alu::Add,
        OpKind::Sub => Alu::Sub,
        OpKind::Mul => Alu::Mul,
        OpKind::Div => Alu::Div,
        OpKind::Min => Alu::Min,
        OpKind::Max => Alu::Max,
        OpKind::BitAnd => Alu::And,
        OpKind::BitOr => Alu::Or,
        // Integer add on lane bit patterns (exp/log exponent arithmetic).
        OpKind::IAdd => Alu::IAdd,
        _ => unimplemented_op("avx-512", op),
    };
    Inst::Alu { op, dst, a, b }.emit_into(code);
}

// =============================================================================
// Masks & `If` — a mask is an ordinary vector (all-ones / all-zeros lanes) in
// the regular zmm register file, exactly like NEON. It flows through the shared
// allocator as a normal value; the k-register these encoders use transiently
// (a `vcmpps`/`vptestmd` destination, immediately widened or read into the
// flags) is `RegisterFile::mask_scratch`'s allocated reservation for the one
// instruction that needs it, named through `Scratch::mask_temp`/
// `mask_guard_temp` rather than a hardcoded constant.
// =============================================================================

/// Map a comparison `OpKind` to its `vcmpps` predicate.
fn cmp_pred(op: OpKind) -> Option<Pred> {
    Some(match op {
        OpKind::Eq => Pred::Eq,
        OpKind::Ne => Pred::Ne,
        OpKind::Lt => Pred::Lt,
        OpKind::Le => Pred::Le,
        OpKind::Gt => Pred::Nle,
        OpKind::Ge => Pred::Ge,
        _ => return None,
    })
}

/// Whether `op` is a comparison handled by [`emit_compare`].
#[must_use]
fn is_compare(op: OpKind) -> bool {
    cmp_pred(op).is_some()
}

/// Emit `dst = (srcs[0] <op> srcs[1]) ? all-ones : all-zeros` as a vector
/// mask.
///
/// `vcmpps k, src1, src2, pred` writes a k-register — this instruction's
/// `RegisterFile::mask_scratch` reservation, `k` — and `vpmovm2d dst, k`
/// widens it to a per-lane all-ones/all-zeros vector occupying the
/// allocator-assigned `dst` zmm.
///
/// `srcs` is a pair rather than two more positional args to stay inside this
/// crate's 5-argument ceiling.
fn emit_compare(code: &mut Vec<u8>, op: OpKind, dst: Reg, srcs: [Reg; 2], k: KReg) {
    let Some(pred) = cmp_pred(op) else {
        unimplemented_op("avx-512", op)
    };
    let [a, b] = srcs;
    AsmProgram::from([Inst::CmpK { pred, dst: k, a, b }, Inst::Movm2d { dst, k }]).assemble(code);
}

/// Emit `dst = src << amount` / `dst = src >> amount` (logical, zero-fill)
/// on lane bit patterns. The amount is a compile-time immediate — the
/// schedule folds the `Const` RHS out (`ScheduledOp::ShiftImm`).
fn emit_shift_imm(code: &mut Vec<u8>, op: OpKind, dst: Reg, src: Reg, amount: u8) {
    let direction = match op {
        OpKind::Shl => Direction::Left,
        OpKind::Shr => Direction::Right,
        _ => unimplemented_op("avx-512", op),
    };
    Inst::Shift {
        direction,
        dst,
        src,
        amount,
    }
    .emit_into(code);
}

/// `dst = op(src)`.
///
/// The temp is the allocator's for this instruction; only `Neg` and `Abs`
/// use it, to hold the sign mask, which comes from the kernel's constant pool
/// like any other constant.
///
/// # Errors
///
/// [`CompileError::BudgetExceeded`] when that mask's pool entry lies past a
/// `disp32` ([`emit_const`]).
fn emit_unary(
    code: &mut Vec<u8>,
    unary: super::Unary,
    pool: &mut x86_64::ConstPool,
) -> Result<(), CompileError> {
    let super::Unary { op, dst, src, temp } = unary;
    let lanewise = |op| Inst::Unary { op, dst, src };
    let round = |mode| Inst::Round { mode, dst, src };
    // `dst` may alias `src`, so the mask goes in the temp, not `dst`:
    // writing it into `dst` first would clobber the source before it is read.
    let mut signed = |op, bits| {
        let mask = super::declared_temp(temp);
        emit_const(code, mask, f32::from_bits(bits), pool)?;
        Ok::<_, CompileError>(Inst::Alu {
            op,
            dst,
            a: src,
            b: mask,
        })
    };
    let inst = match op {
        OpKind::Sqrt => lanewise(Lanewise::Sqrt),
        OpKind::Neg => signed(Alu::Xor, 0x8000_0000)?,
        OpKind::Abs => signed(Alu::And, 0x7FFF_FFFF)?,
        OpKind::Floor => round(Rounding::Floor),
        OpKind::Ceil => round(Rounding::Ceil),
        OpKind::Round => round(Rounding::Nearest),
        OpKind::Recip => lanewise(Lanewise::Recip),
        OpKind::Rsqrt => lanewise(Lanewise::Rsqrt),
        // Int/float domain crossings, exactly the hardware's cvttps2dq /
        // cvtdq2ps — the primitives exp/log lower to.
        OpKind::TruncToInt => lanewise(Lanewise::ToInt),
        OpKind::IntToFloat => lanewise(Lanewise::FromInt),
        _ => unimplemented_op("avx-512", op),
    };
    inst.emit_into(code);
    Ok(())
}

/// Emit a fused multiply-add `dst = a*b + c` where `dst` already holds `c`:
/// `vfmadd231ps dst, a, b` (EVEX.512.66.0F38.W0 B8 /r), which is
/// `dst = a*b + dst`. The 231 form is the one whose accumulator is the
/// destination, so `c` needs no move.
fn emit_fmadd_c_in_dst(code: &mut Vec<u8>, dst: Reg, a: Reg, b: Reg) {
    Inst::Fma231 { acc: dst, a, b }.emit_into(code);
}

#[cfg(test)]
mod tests {
    //! Hardware validation. The byte-level EVEX encodings for 2-operand forms,
    //! memory forms, FMA231, and the stack frame are hand-derived; these JIT
    //! real `zmm` kernels and execute them on the host (all 16 lanes), so a bad
    //! byte fails loudly. Runtime tests require `+avx512f`.
    #![allow(clippy::needless_range_loop)]
    use super::*;

    /// Offset 3 shifted up by a full 16-bit range: where a 16-bit slot used
    /// to wrap back to argument 3.
    const PAST_U16: u64 = 3 + (u16::MAX as u64 + 1);

    /// The uniform read for `dst = 5` through the block in `rax`: offset 3
    /// is `vbroadcastss zmm5, [rax + 12]` (checked against `llvm-mc
    /// --disassemble`, LLVM 18), and an offset past the old 16-bit width
    /// carries its full `disp32` with the same prefix and ModRM. The block's
    /// address is a pointer-class value the allocator placed, so no load of
    /// it appears here: that is the `Context` def's, once per call.
    #[test]
    fn a_uniform_read_is_one_broadcast_load() {
        let mut code = Vec::new();
        emit_uniform_load(&mut code, Reg(5), PtrReg(0), 3).expect("fits");
        assert_eq!(code, [0x62, 0xF2, 0x7D, 0x48, 0x18, 0xA8, 0x0C, 0, 0, 0]);

        let mut code = Vec::new();
        emit_uniform_load(&mut code, Reg(5), PtrReg(0), PAST_U16).expect("fits");
        assert_eq!(
            code,
            [0x62, 0xF2, 0x7D, 0x48, 0x18, 0xA8, 0x0C, 0x00, 0x04, 0x00]
        );
    }

    /// The width is the encoder's, and an offset past it is refused, never
    /// wrapped: a wrapped displacement would be a load of some other argument,
    /// with plausible pixels. `disp32` is signed, so the last element it
    /// reaches is at `i32::MAX / 4` — and a refused offset emits nothing.
    #[test]
    fn an_offset_past_the_disp32_is_refused() {
        const LAST_DISP32: u64 = i32::MAX as u64 / 4;
        let mut code = Vec::new();
        emit_uniform_load(&mut code, Reg(0), PtrReg(0), LAST_DISP32)
            .expect("the last element a disp32 reaches");
        for offset in [LAST_DISP32 + 1, u64::MAX] {
            let mut code = Vec::new();
            let refused = emit_uniform_load(&mut code, Reg(0), PtrReg(0), offset);
            assert!(
                matches!(refused, Err(CompileError::BudgetExceeded(_))),
                "offset {offset}: expected a refusal, got {refused:?}"
            );
            assert!(code.is_empty(), "offset {offset}: a refusal emitted bytes");
        }
    }

    /// The lane-uniform read for `dst = 5, idx = 6`: `vcvttss2si rcx, xmm6`
    /// then `vbroadcastss zmm5, [rax + rcx*4]` (checked against `objdump
    /// -M intel`); and with the base and index past the low eight, `[r9 +
    /// r11*4]` sets `X` and `B` in the prefix (clear, inverted). The base's
    /// own load is the `Context` def's, once per call, not this instruction's.
    #[test]
    fn a_lane_uniform_read_truncates_then_broadcasts() {
        let low = x86_64::BroadcastGprs {
            base: PtrReg(0),
            index: x86_64::gpr::RCX,
        };
        let mut code = Vec::new();
        emit_broadcast_load(&mut code, Reg(5), Reg(6), low);
        assert_eq!(
            code,
            [
                0x62, 0xF1, 0xFE, 0x48, 0x2C, 0xCE, 0x62, 0xF2, 0x7D, 0x48, 0x18, 0x2C, 0x88
            ]
        );

        let high = x86_64::BroadcastGprs {
            base: PtrReg(9),
            index: Gpr(11),
        };
        let mut code = Vec::new();
        emit_broadcast_load(&mut code, Reg(5), Reg(6), high);
        assert_eq!(&code[6..], [0x62, 0x92, 0x7D, 0x48, 0x18, 0x2C, 0x99]);
    }

    /// Executes the bytes on this host's CPU, so every test first asks
    /// whether this process emits AVX-512 (`skip_unless_avx512_is_selected!`).
    /// The `extern
    /// "C"` kernels take `zmm` values, which the ABI only lets a caller
    /// compiled with AVX-512 pass — hence `#[target_feature]` on the
    /// functions that call them, and nowhere else.
    #[cfg(target_arch = "x86_64")]
    mod runtime {
        use super::super::*;

        /// `ret` (`C3`): the end of a hand-assembled test kernel, which
        /// returns its `__m256`/`__m512` in the vector register a
        /// `vzeroupper` would clear.
        const RET: u8 = 0xC3;
        use crate::emit::executable::CompiledKernel;
        use crate::emit::{AsmInsn, PtrReg};

        /// In a test: return early, with a note on stderr, unless this process
        /// emits AVX-512 ([`crate::isa::detect`]) — so `xtask isa-matrix` runs
        /// these on the pass that selects it. The harness has no skip, so an
        /// early `return` with a note is what "not this tier" looks like:
        /// never a silent pass, and never a failure for a fact about the
        /// machine.
        macro_rules! skip_unless_avx512_is_selected {
            () => {
                let selected = crate::isa::detect();
                if selected != crate::isa::Isa::Avx512 {
                    std::eprintln!("skipped: this process emits {selected:?}, not AVX-512");
                    return;
                }
            };
        }
        use core::arch::x86_64::*;

        // Passing __m512 by value IS the emitted ABI (SysV: zmm0-7), so
        // not-FFI-safe is a false positive here, as for `executable`'s aliases.
        #[allow(improper_ctypes_definitions)]
        type K = unsafe extern "C" fn(__m512, __m512, __m512, __m512) -> __m512;

        fn run(body: &[u8], xs: [f32; 16], ys: [f32; 16], zs: [f32; 16]) -> [f32; 16] {
            let mut code = body.to_vec();
            code.push(RET);
            // SAFETY: every caller is a test that checked the host runs AVX-512.
            unsafe { run_code(&code, xs, ys, zs) }
        }

        /// `run`, for a body that read constants from `pool`: the anchor
        /// ahead of it and the pool behind its `ret`, as the driver lays a
        /// kernel out.
        fn run_pooled(
            body: &[u8],
            pool: &x86_64::ConstPool,
            xs: [f32; 16],
            ys: [f32; 16],
            zs: [f32; 16],
        ) -> [f32; 16] {
            let mut asm = crate::emit::Assembly::default();
            let pool_label = asm.mint();
            x86_64::anchor(&mut asm, pool_label);
            asm.run.extend_from_slice(body);
            asm.run.push(RET);
            pool.finish(&mut asm, pool_label);
            // SAFETY: every caller is a test that checked the host runs AVX-512.
            unsafe { run_code(&asm.finish(), xs, ys, zs) }
        }

        /// # Safety
        ///
        /// The host must execute AVX-512: `code` is `zmm` code, and this
        /// function is compiled with AVX-512F enabled to be allowed to pass
        /// `zmm` values.
        #[target_feature(enable = "avx512f")]
        unsafe fn run_code(code: &[u8], xs: [f32; 16], ys: [f32; 16], zs: [f32; 16]) -> [f32; 16] {
            let exec = unsafe { CompiledKernel::from_code(code).expect("mmap") };
            unsafe {
                let f: K = core::mem::transmute(exec.as_bytes().as_ptr());
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

        /// Call `exec` as `fn(*const f32 base, zmm float indices) -> zmm`:
        /// the gather tests' ABI.
        ///
        /// # Safety
        ///
        /// The host must execute AVX-512 (every caller checked).
        #[target_feature(enable = "avx512f")]
        unsafe fn gather(exec: &CompiledKernel, base: *const f32, idx: [f32; 16]) -> [f32; 16] {
            #[allow(improper_ctypes_definitions)]
            type G = unsafe extern "C" fn(*const f32, __m512) -> __m512;
            unsafe {
                let f: G = core::mem::transmute(exec.as_bytes().as_ptr());
                let r = f(base, _mm512_loadu_ps(idx.as_ptr()));
                let mut out = [0.0f32; 16];
                _mm512_storeu_ps(out.as_mut_ptr(), r);
                out
            }
        }

        /// Set `k1` to all-ones and gather `dst = [base + index*4]` under it,
        /// as the driver does before every gather.
        fn gather_through_k1(c: &mut Vec<u8>, dst: Reg, base: PtrReg, index: Reg) {
            let mask = KReg(1);
            set_gather_mask(c, mask, x86_64::gpr::RAX);
            Inst::Gather {
                dst,
                base,
                index,
                mask,
            }
            .emit_into(c);
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
            skip_unless_avx512_is_selected!();
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
            skip_unless_avx512_is_selected!();
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            emit_binary(&mut c, OpKind::Mul, Reg(20), X, Y);
            emit_mov(&mut c, X, Reg(20));
            check(run(&c, xs, ys, zs), |i| xs[i] * ys[i], "mul via zmm20");
        }

        #[test]
        fn emit_load_and_emit_store_address_a_high_numbered_base_register_correctly() {
            skip_unless_avx512_is_selected!();
            // Every production caller in this file addresses memory through
            // rsp or rax (both < r8), so `Evex::rm`'s B-bit inversion for a
            // >= r8 base has no other coverage. Move the incoming pointer
            // into r9 (rbp/r13 have their own mod=00 RIP-relative special
            // case, which `mem_operand` refuses outright) and round-trip
            // through it to pin that bit. The load addresses it with a
            // disp32 (`Inst::Load` has no `NoDisp` form), the store with
            // none.
            #[allow(improper_ctypes_definitions)]
            type F = unsafe extern "C" fn(*mut f32);

            let mut pool = x86_64::ConstPool::default();
            let mut asm = crate::emit::Assembly::default();
            let pool_label = asm.mint();
            x86_64::anchor(&mut asm, pool_label);
            let c = &mut asm.run;
            x86_64::Gp::Mov {
                dst: PtrReg(9),
                src: PtrReg(x86_64::gpr::RDI.0),
            }
            .emit_into(c);
            let via_r9 = Mem {
                base: PtrReg(9),
                disp: NoDisp,
            };
            Inst::Load {
                dst: X,
                src: Mem {
                    base: PtrReg(9),
                    disp: x86_64::Imm32(0),
                },
            }
            .emit_into(c);
            emit_const(c, Reg(5), 1.0, &mut pool).unwrap();
            emit_binary(c, OpKind::Add, X, X, Reg(5));
            Inst::StoreBatch {
                dst: via_r9,
                src: X,
            }
            .emit_into(c);
            x86_64::Gp::Ret { size: 0, flags: () }.emit_into(c);
            pool.finish(&mut asm, pool_label);
            let c = asm.finish();

            let mut buf = [0.0f32; 16];
            for (i, v) in buf.iter_mut().enumerate() {
                *v = i as f32;
            }
            let exec = unsafe { CompiledKernel::from_code(&c).expect("mmap") };
            unsafe {
                let f: F = core::mem::transmute(exec.as_bytes().as_ptr());
                f(buf.as_mut_ptr());
            }
            for (i, &v) in buf.iter().enumerate() {
                assert_eq!(v, i as f32 + 1.0, "lane {i}");
            }
        }

        fn unary(op: OpKind, src: Reg, temp: Option<Reg>) -> crate::emit::Unary {
            crate::emit::Unary {
                op,
                dst: X,
                src,
                temp,
            }
        }

        #[test]
        fn emit_unary_computes_sqrt_of_a_positive_operand() {
            skip_unless_avx512_is_selected!();
            let (xs, ys, zs) = lanes();
            let mut pool = x86_64::ConstPool::default();
            let mut c = Vec::new();
            emit_unary(&mut c, unary(OpKind::Sqrt, Y, None), &mut pool).unwrap(); // Y > 0
            check(run_pooled(&c, &pool, xs, ys, zs), |i| ys[i].sqrt(), "sqrt");
        }

        #[test]
        fn emit_unary_negates_and_takes_the_absolute_value_of_every_lane() {
            skip_unless_avx512_is_selected!();
            let (xs, ys, zs) = lanes();
            let mut pool = x86_64::ConstPool::default();
            let mut c = Vec::new();
            emit_unary(&mut c, unary(OpKind::Neg, X, Some(TEMP)), &mut pool).unwrap();
            check(run_pooled(&c, &pool, xs, ys, zs), |i| -xs[i], "neg");
            let mut pool = x86_64::ConstPool::default();
            let mut c = Vec::new();
            emit_unary(&mut c, unary(OpKind::Abs, X, Some(TEMP)), &mut pool).unwrap();
            check(run_pooled(&c, &pool, xs, ys, zs), |i| xs[i].abs(), "abs");
        }

        /// Two constants, the first read twice: the pool holds each once, and
        /// every read is one broadcast from it.
        #[test]
        fn emit_const_broadcasts_and_adds_to_every_lane() {
            skip_unless_avx512_is_selected!();
            let (xs, ys, zs) = lanes();
            let mut pool = x86_64::ConstPool::default();
            let mut c = Vec::new();
            emit_const(&mut c, Reg(5), 2.5, &mut pool).unwrap();
            emit_binary(&mut c, OpKind::Add, X, X, Reg(5));
            emit_const(&mut c, Reg(6), -1.0, &mut pool).unwrap();
            emit_binary(&mut c, OpKind::Add, X, X, Reg(6));
            emit_const(&mut c, Reg(5), 2.5, &mut pool).unwrap();
            emit_binary(&mut c, OpKind::Add, X, X, Reg(5));
            check(
                run_pooled(&c, &pool, xs, ys, zs),
                |i| xs[i] + 4.0,
                "const+add",
            );
        }

        #[test]
        fn emit_fmadd_c_in_dst_computes_the_fused_multiply_add() {
            skip_unless_avx512_is_selected!();
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
            skip_unless_avx512_is_selected!();
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
        fn a_gather_reads_the_value_at_each_lanes_index() {
            skip_unless_avx512_is_selected!();
            // JIT a function: fn(*const f32 base [rdi], __m512 idx_float [zmm0]) -> __m512
            // that truncates the float indices, sets the mask, and gathers
            // base[idx] per lane. Validates the VSIB vgatherdps bytes on hardware.
            let mut c = Vec::new();
            Inst::Unary {
                op: Lanewise::ToInt,
                dst: Reg(13),
                src: Reg(0),
            }
            .emit_into(&mut c); // zmm13 = (i32) idx_float
            gather_through_k1(&mut c, Reg(14), PtrReg(7), Reg(13)); // zmm14{k1} = [rdi + zmm13*4]
            emit_mov(&mut c, Reg(0), Reg(14)); // return in zmm0
            c.push(RET);

            let buf: Vec<f32> = (0..64).map(|i| (i as f32) * 1.5 + 0.25).collect();
            // Distinct per-lane indices, including repeats and the ends.
            let idx: [f32; 16] = [
                0.0, 63.0, 1.0, 2.0, 10.0, 10.0, 5.0, 32.0, 7.0, 8.0, 63.0, 0.0, 20.0, 21.0, 40.0,
                41.0,
            ];

            let exec = unsafe { CompiledKernel::from_code(&c).expect("mmap") };
            // SAFETY: the host runs AVX-512, checked at the top of this test.
            let out = unsafe { gather(&exec, buf.as_ptr(), idx) };

            for i in 0..16 {
                let want = buf[idx[i] as usize];
                assert_eq!(out[i], want, "gather lane {i}: idx {}", idx[i]);
            }
        }

        #[test]
        fn a_gather_addresses_high_numbered_vector_registers_and_a_gpr_base() {
            skip_unless_avx512_is_selected!();
            // The production driver's gather base is a pointer in r9-r11 (rax
            // is clobbered just before the gather), and its dst/idx are below
            // zmm16 in every kernel this test suite compiles, so
            // `a_gather_reads_the_value_at_each_lanes_index` never sets the
            // R'/B/V' extension bits this emitter also has to encode. Move the
            // base pointer into r9 (>= r8) and gather into/from zmm registers
            // >= 16 to pin them, mirroring
            // `emit_binary_writes_a_high_numbered_register`'s zmm20 case.
            let mut c = Vec::new();
            x86_64::Gp::Mov {
                dst: PtrReg(9),
                src: PtrReg(x86_64::gpr::RDI.0),
            }
            .emit_into(&mut c);
            Inst::Unary {
                op: Lanewise::ToInt,
                dst: Reg(21),
                src: Reg(0),
            }
            .emit_into(&mut c); // zmm21 = (i32) idx_float
            gather_through_k1(&mut c, Reg(20), PtrReg(9), Reg(21)); // zmm20{k1} = [r9 + zmm21*4]
            emit_mov(&mut c, Reg(0), Reg(20));
            c.push(RET);

            let buf: Vec<f32> = (0..64).map(|i| (i as f32) * 1.5 + 0.25).collect();
            let idx: [f32; 16] = [
                0.0, 63.0, 1.0, 2.0, 10.0, 10.0, 5.0, 32.0, 7.0, 8.0, 63.0, 0.0, 20.0, 21.0, 40.0,
                41.0,
            ];

            let exec = unsafe { CompiledKernel::from_code(&c).expect("mmap") };
            // SAFETY: the host runs AVX-512, checked at the top of this test.
            let out = unsafe { gather(&exec, buf.as_ptr(), idx) };

            for i in 0..16 {
                let want = buf[idx[i] as usize];
                assert_eq!(out[i], want, "gather lane {i}: idx {}", idx[i]);
            }
        }

        #[test]
        fn emit_load_after_emit_store_recovers_the_spilled_value() {
            skip_unless_avx512_is_selected!();
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            crate::emit::x86_64::Gp::Enter {
                size: 64,
                flags: (),
            }
            .emit_into(&mut c);
            emit_binary(&mut c, OpKind::Mul, Reg(6), X, Y);
            Inst::Store {
                dst: frame_slot(0),
                src: Reg(6),
            }
            .emit_into(&mut c);
            emit_binary(&mut c, OpKind::Add, Reg(6), X, X); // clobber
            Inst::Load {
                dst: X,
                src: frame_slot(0),
            }
            .emit_into(&mut c);
            // `add rsp, 64` (REX.W 81 /0 id), not `Gp::Ret`: this kernel
            // returns its answer in a zmm register, which a `vzeroupper`
            // would clear.
            AsmProgram::from([EncodedInst::from_slice(&[0x48, 0x81, 0xC4, 64, 0, 0, 0])])
                .assemble(&mut c);
            check(run(&c, xs, ys, zs), |i| xs[i] * ys[i], "spill roundtrip");
        }
    }
}

// =============================================================================
// The AVX-512 `LegacyBackend` driver
// =============================================================================

/// The AVX-512 half of code generation.
///
/// **This file is where AVX-512-specific bugs live, and the only place they
/// can.** Emission is a pure function into `Vec<u8>`, so everything here
/// compiles, typechecks and is swept for op coverage on every host, whatever
/// CPU it has. Only `compile_native` in `emit` decides which backend a
/// process instantiates — from the tier `crate::isa` read off the CPU — and
/// only [`executable`](crate::emit::executable) needs the matching hardware.
///
/// The consequence worth stating: a change that does not touch an ISA file
/// cannot introduce a platform-specific bug. That is the bargain `unsafe`
/// makes — confine what cannot be checked, so the rest is checked by
/// construction.
pub(super) mod driver {
    use super::super::*;
    use super::{AsmProgram, IOTA_BYTES, Inst, Lanewise, Mem, NoDisp, frame_slot};
    use crate::emit::x86_64 as x86;
    use crate::emit::x86_64::write_address;
    use crate::error::CompileError;
    use alloc::vec::Vec;
    use pixelflow_ir::kind::OpKind;

    /// The AVX-512 register file (zmm, 512-bit).
    ///
    /// The same GPR roles as AVX2's — SysV's, and the shared driver depends
    /// on them — at twice the width, over the whole extended vector file.
    const AVX512_FILE: regalloc::RegisterFile = regalloc::RegisterFile {
        // zmm0-31, all thirty-two. The pool was *six* when this work
        // started, because a contiguous range could not reach past the reload
        // pair and the gather's scratch — sixteen registers were untouched by
        // anything at all.
        scratch: regalloc::RegSet::range(0, 32),
        // Nothing. Every register this backend's encodings destroy is a
        // per-instruction reservation, borrowed only across the one
        // instruction that needs it. The `If` needs none — `vpternlogd`
        // consumes its three operands.
        fixed: &[],
        temps_for: super::temps_for,
        // A guard reduces its mask through `k1` and `kortestw` into the
        // flags, which costs no vector register.
        guard_temps: 0,
        vector_bytes: 64,
        // SysV's first three integer arguments, in the ABI's order: the
        // context, the output plane, its pitch — declared so `checked`
        // proves `gpr_scratch` misses all three.
        gpr_ctx: Some(x86::gpr::RDI),
        gpr_out: Some(x86::gpr::RSI),
        gpr_pitch: Some(x86::gpr::RDX),
        // rax/rcx: the broadcast's index, the store's row and column, the
        // iota's bytes, a gather's all-ones. `vgatherdps`'s native addressing
        // needs no per-lane index GPR.
        gpr_scratch: regalloc::GprSet::of(&[x86::gpr::RAX, x86::gpr::RCX]),
        gpr_temps_for: super::gpr_temps_for,
        // r9-r11: the pointer class's pool, the caller-saved GPRs left after
        // the arguments, the scratch and `r8` (the constant pool's anchor).
        pointers: regalloc::GprSet::of(&[x86::gpr::R9, x86::gpr::R10, x86::gpr::R11]),
        // AVX-512's mask-register file: k1, transient scratch for a
        // compare's `vcmpps` destination, a guard's `vptestmd` destination,
        // a gather's writemask and a remainder store's writemask, never the
        // same instruction's use of two at once.
        mask_scratch: regalloc::MaskSet::of(&[KReg(1)]),
        mask_temps_for: super::mask_temps_for,
        // `vptestmd`'s k-register destination, reduced to flags by
        // `kortestw` — the mask-class mirror of a vector `guard_temps`,
        // needed because this tier's guard (unlike AVX2's
        // `vmovmskps`/flags) goes through the mask-register file.
        mask_guard_temps: 1,
    }
    .checked();

    /// AVX-512 implementation of the shared driver's leaf operations.
    pub(in crate::emit) struct Avx512Backend {
        consts: x86::ConstPool,
    }

    impl Avx512Backend {
        pub(in crate::emit) fn new() -> Self {
            Self {
                consts: x86::ConstPool::default(),
            }
        }

        fn reload(&mut self, code: &mut Vec<u8>, reload: &Reload) -> Result<(), CompileError> {
            match reload {
                Reload::FromStack { target, slot } => {
                    self.slot_load(code, *target, slot.offset());
                }
                Reload::Const { target, val_bits } => {
                    super::emit_const(code, *target, f32::from_bits(*val_bits), &mut self.consts)?;
                }
                Reload::Ptr { target, slot } => self.ptr_load(code, *target, slot.offset()),
            }
            Ok(())
        }
    }

    impl LegacyBackend for Avx512Backend {
        fn jump(&mut self, asm: &mut Assembly, label: Label) {
            asm.push(x86::Gp::Jmp { to: label });
        }

        fn register_file(&self) -> regalloc::RegisterFile {
            AVX512_FILE
        }

        /// Nothing to seed: the pool fills as constants are emitted.
        fn begin(&mut self, _schedule: &[regalloc::Def]) -> Result<(), CompileError> {
            Ok(())
        }

        fn reads_pool(&self, val_bits: u32) -> bool {
            val_bits != 0
        }

        fn emit_plan(
            &mut self,
            code: &mut Vec<u8>,
            plan: &InstructionPlan,
        ) -> Result<(), CompileError> {
            for r in &plan.reloads {
                self.reload(code, r)?;
            }
            if let Some((dst, src)) = plan.setup_mov {
                super::emit_mov(code, dst, src);
            }
            match &plan.op {
                ResolvedOp::Nop => {}
                ResolvedOp::LoadConst { dst, val_bits } => {
                    super::emit_const(code, *dst, f32::from_bits(*val_bits), &mut self.consts)?;
                }
                // The iota: the bytes `0..16` in through a GPR eight at a
                // time, widened to dwords, converted. No vector temp — `dst`
                // is every stage's.
                ResolvedOp::Lanes { dst } => {
                    let gpr = crate::emit::declared_gpr_temp(plan.scratch.gpr_temp(0));
                    x86::Gp::Movabs {
                        dst: gpr,
                        imm: IOTA_BYTES[0],
                    }
                    .emit_into(code);
                    let (dst, src) = (*dst, gpr);
                    Inst::Movq { dst, src }.emit_into(code);
                    x86::Gp::Movabs {
                        dst: gpr,
                        imm: IOTA_BYTES[1],
                    }
                    .emit_into(code);
                    AsmProgram::from([
                        Inst::InsertHigh { dst, src },
                        Inst::Unary {
                            op: Lanewise::WidenBytes,
                            dst,
                            src: dst,
                        },
                        Inst::Unary {
                            op: Lanewise::FromInt,
                            dst,
                            src: dst,
                        },
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
                    super::emit_unary(code, unary, &mut self.consts)?;
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
                    // dst = base[idx]: `vgatherdps` through the mask temp, `base`
                    // being the buffer's address wherever the allocator
                    // keeps it. The lowered index is a float, so it is
                    // truncated to int32 lanes first, and the mask is all-ones
                    // going in (the instruction clears the bits it
                    // completes), so it is reset before every gather.
                    let idx_int = crate::emit::declared_temp(plan.scratch.temp(0));
                    let gather_dst = crate::emit::declared_temp(plan.scratch.temp(1));
                    Inst::Unary {
                        op: Lanewise::ToInt,
                        dst: idx_int,
                        src: *idx,
                    }
                    .emit_into(code);
                    let mask = crate::emit::declared_mask_temp(plan.scratch.mask_temp(0));
                    let ones = crate::emit::declared_gpr_temp(plan.scratch.gpr_temp(0));
                    super::set_gather_mask(code, mask, ones);
                    Inst::Gather {
                        dst: gather_dst,
                        base: *base,
                        index: idx_int,
                        mask,
                    }
                    .emit_into(code);
                    super::emit_mov(code, *dst, gather_dst);
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
                    super::emit_uniform_load(code, *dst, *base, *offset)?;
                }
                ResolvedOp::Context { dst, slot } => {
                    let ctx = AVX512_FILE
                        .gpr_ctx
                        .expect("x86's context read needs the GPR context input");
                    x86::Gp::MovLoad {
                        dst: *dst,
                        src: Mem {
                            base: PtrReg(ctx.0),
                            disp: x86::Imm32(i32::from(*slot) * x86::PTR_BYTES),
                        },
                    }
                    .emit_into(code);
                }
                ResolvedOp::Binary {
                    op,
                    dst,
                    left,
                    right,
                } => {
                    // EVEX 3-operand: either source may alias `dst`.
                    // Comparisons produce a vector mask (vcmpps -> vpmovm2d),
                    // through the mask-register temp `mask_temps_for` reserved.
                    if super::is_compare(*op) {
                        let k = crate::emit::declared_mask_temp(plan.scratch.mask_temp(0));
                        super::emit_compare(code, *op, *dst, [*left, *right], k);
                    } else {
                        super::emit_binary(code, *op, *dst, *left, *right);
                    }
                }
                ResolvedOp::FusedMulAdd { dst, a, b } => {
                    // dst holds c (setup_mov); real FMA231: dst = a*b + dst.
                    super::emit_fmadd_c_in_dst(code, *dst, *a, *b);
                }
                ResolvedOp::If {
                    dst,
                    if_true,
                    if_false,
                } => {
                    // setup_mov already placed the vector mask in dst; one vpternlogd.
                    Inst::Blend {
                        dst: *dst,
                        if_true: *if_true,
                        if_false: *if_false,
                    }
                    .emit_into(code);
                }
            }
            Ok(())
        }

        fn emit_mov(&mut self, code: &mut Vec<u8>, dst: Reg, src: Reg) {
            super::emit_mov(code, dst, src);
        }

        fn emit_store(
            &mut self,
            code: &mut Vec<u8>,
            src: Reg,
            offset: u32,
        ) -> Result<(), CompileError> {
            self.slot_store(code, src, offset);
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
                    super::emit_const(code, target, f32::from_bits(bits), &mut self.consts)?;
                    Ok(target)
                }
                Binding::Loc(Loc::Slot(slot)) => {
                    self.slot_load(code, target, slot.offset());
                    Ok(target)
                }
                Binding::Loc(Loc::Ptr(p)) => {
                    unreachable!("{vid:?} is an address in {p:?}; the pointer class resolves it")
                }
            }
        }

        fn ptr_store(&mut self, code: &mut Vec<u8>, src: PtrReg, offset: u32) {
            x86::Gp::MovStore {
                dst: frame_slot(offset),
                src,
            }
            .emit_into(code);
        }

        fn ptr_load(&mut self, code: &mut Vec<u8>, dst: PtrReg, offset: u32) {
            x86::Gp::MovLoad {
                dst,
                src: frame_slot(offset),
            }
            .emit_into(code);
        }

        fn ptr_mov(&mut self, code: &mut Vec<u8>, dst: PtrReg, src: PtrReg) {
            x86::Gp::Mov { dst, src }.emit_into(code);
        }

        fn anchor(&mut self, asm: &mut Assembly, pool: Label) {
            x86::anchor(asm, pool);
        }

        fn finish(&mut self, asm: &mut Assembly, pool: Label) {
            self.consts.finish(asm, pool);
        }

        // If short-circuit guards: reduce the vector mask to flags (vptestmd +
        // kortestw) and branch. jz = all-false (skip true arm); jc = all-true (skip
        // false arm). The k-register spelling of AVX2's `vmovmskps` guards.
        /// [`MaskTest::scratch`] is unused: this tier reduces the mask with
        /// `kortest` into the flags, needing no *vector* register. It is the
        /// one tier that wants [`MaskTest::mask_scratch`], because `vptestmd`
        /// lands in a `k`-register before `kortestw` can read it.
        fn branch_if_arm_is_dead(&mut self, asm: &mut Assembly, test: MaskTest, label: Label) {
            let k = crate::emit::declared_mask_temp(test.mask_scratch);
            asm.push(Inst::Ptestm {
                dst: k,
                a: test.reg,
                b: test.reg,
            });
            asm.push(Inst::KorTest { flags: (), k });
            // One `kortest` sets both answers at once, so the arm picks the
            // condition rather than a different reduction.
            asm.push_branch(|next| match test.arm {
                // ZF set when k1 == 0: no lane is true, so the true arm is dead.
                IfArm::True => x86::Gp::je(label, next),
                // CF set when k1 == 0xFFFF: every lane is, so the false arm is.
                IfArm::False => x86::Gp::jb(label, next),
            });
        }

        fn frame_alloc(&mut self, code: &mut Vec<u8>, bytes: u32) {
            x86::Gp::Enter {
                size: bytes,
                flags: (),
            }
            .emit_into(code);
        }

        fn slot_store(&mut self, code: &mut Vec<u8>, src: Reg, offset: u32) {
            Inst::Store {
                dst: frame_slot(offset),
                src,
            }
            .emit_into(code);
        }

        fn slot_load(&mut self, code: &mut Vec<u8>, dst: Reg, offset: u32) {
            Inst::Load {
                dst,
                src: frame_slot(offset),
            }
            .emit_into(code);
        }

        fn add_scalar(
            &mut self,
            code: &mut Vec<u8>,
            dst: Reg,
            scratch: Reg,
            scalar: f32,
        ) -> Result<(), CompileError> {
            super::emit_const(code, scratch, scalar, &mut self.consts)?;
            super::emit_binary(code, OpKind::Add, dst, dst, scratch);
            Ok(())
        }

        fn load_const(
            &mut self,
            code: &mut Vec<u8>,
            dst: Reg,
            val: f32,
        ) -> Result<(), CompileError> {
            super::emit_const(code, dst, val, &mut self.consts)
        }

        fn alu(&mut self, code: &mut Vec<u8>, op: OpKind, dst: Reg, srcs: [Reg; 2]) {
            super::emit_binary(code, op, dst, srcs[0], srcs[1]);
        }

        // `emit_binary` has no comparison arm on this tier — a comparison's
        // result is a k-register before `vpmovm2d` widens it to an ordinary
        // vector, which is what `emit_compare` does and `alu` cannot.
        fn test_ge(
            &mut self,
            code: &mut Vec<u8>,
            dst: Reg,
            srcs: [Reg; 2],
            mask_scratch: Option<KReg>,
        ) {
            let k = mask_scratch
                .expect("AVX-512's Ge needs a k-register scratch (RegisterFile::mask_guard_temps)");
            super::emit_compare(code, OpKind::Ge, dst, srcs, k);
        }

        /// A full batch is one `vmovups`; a remainder is the same store
        /// under a writemask of its lanes, built in the GPR the address
        /// arithmetic has finished with.
        fn emit_write(&mut self, code: &mut Vec<u8>, write: &WritePlan) {
            let addr = write_address::<Inst<Physical>>(code, &AVX512_FILE, write);
            let at = Mem {
                base: addr,
                disp: NoDisp,
            };
            let lanes = AVX512_FILE.vector_bytes / 4;
            if write.lanes == lanes {
                Inst::StoreBatch {
                    dst: at,
                    src: write.value,
                }
                .emit_into(code);
                return;
            }
            let mask = crate::emit::declared_gpr_temp(write.scratch.gpr_temp(1));
            let k = crate::emit::declared_mask_temp(write.scratch.mask_temp(0));
            x86::Gp::MovImm32 {
                dst: mask,
                imm: (1u32 << write.lanes) - 1,
            }
            .emit_into(code);
            AsmProgram::from([
                Inst::Kmovw { dst: k, src: mask },
                Inst::StoreMasked {
                    dst: at,
                    src: write.value,
                    mask: k,
                },
            ])
            .assemble(code);
        }

        fn emit_ret(&mut self, code: &mut Vec<u8>, bytes: u32) {
            x86::Gp::Ret {
                size: bytes,
                flags: (),
            }
            .emit_into(code);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// `AVX512_FILE` states every field itself rather than borrowing
        /// AVX2's through `..`, because the two genuinely differ — thirty-two
        /// registers and this backend's own `temps_for` rather than AVX2's
        /// sixteen — so nothing catches a regression back to the narrower
        /// shape except a direct assertion.
        #[test]
        fn avx512_file_reserves_the_whole_zmm_file_with_no_fixed_registers() {
            assert_eq!(AVX512_FILE.scratch, regalloc::RegSet::range(0, 32));
            assert!(AVX512_FILE.fixed.is_empty());
            assert_eq!(
                AVX512_FILE.temps_for as *const () as usize,
                super::super::temps_for as *const () as usize
            );
        }

        /// The remainder's writemask path, byte for byte against the SDM.
        #[test]
        fn a_masked_store_and_its_mask_encode_as_the_manual_says() {
            let mut c = Vec::new();
            // kmovw k1, ecx — VEX.L0.0F.W0 92 /r
            Inst::Kmovw {
                dst: KReg(1),
                src: x86::gpr::RCX,
            }
            .emit_into(&mut c);
            assert_eq!(c, [0xC4, 0xE1, 0x78, 0x92, 0xC9]);
            // kmovw k2, r9d — the source past r8 clears B
            let mut c = Vec::new();
            Inst::Kmovw {
                dst: KReg(2),
                src: Gpr(9),
            }
            .emit_into(&mut c);
            assert_eq!(c, [0xC4, 0xC1, 0x78, 0x92, 0xD1]);
            // vmovups [rax]{k1}, zmm4 — EVEX.512.0F.W0 11 /r, aaa = 001
            let mut c = Vec::new();
            Inst::StoreMasked {
                dst: Mem {
                    base: PtrReg(0),
                    disp: NoDisp,
                },
                src: Reg(4),
                mask: KReg(1),
            }
            .emit_into(&mut c);
            assert_eq!(c, [0x62, 0xF1, 0x7C, 0x49, 0x11, 0x20]);
            // kortestw k1, k1 and k5, k5 — VEX.L0.0F.W0 98 /r, the register in both fields
            let mut c = Vec::new();
            for k in [1, 5] {
                Inst::KorTest {
                    flags: (),
                    k: KReg(k),
                }
                .emit_into(&mut c);
            }
            assert_eq!(c, [0xC5, 0xF8, 0x98, 0xC9, 0xC5, 0xF8, 0x98, 0xED]);
            // vmovq xmm20, rax — EVEX.128.66.0F.W1 6E /r, an extended register
            let mut c = Vec::new();
            Inst::Movq {
                dst: Reg(20),
                src: x86::gpr::RAX,
            }
            .emit_into(&mut c);
            assert_eq!(c, [0x62, 0xE1, 0xFD, 0x08, 0x6E, 0xE0]);
        }
    }
}
