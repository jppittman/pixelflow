//! x86-64 AVX2 (VEX.256) JIT encoder — 256-bit, 8-lane `ymm` kernels.
//!
//! The floor of the x86-64 tiers (`crate::isa`), below the AVX-512 EVEX
//! encoders (`avx512.rs`, 512-bit). Sixteen registers, `ymm0-15` — AVX2 has
//! no extended file — and the general-register half of every kernel (the
//! loop nest, the store's address, the pointer class, the constant pool) is
//! `x86_64.rs`'s, shared with AVX-512; only the vector *encoding* is this
//! file's.
//!
//! VEX is 3-operand and non-destructive — the same property AVX-512's EVEX
//! has — so an operand may alias the destination in every encoding here.
//! Comparisons are simpler here than on AVX-512: `vcmpps` writes an ordinary
//! all-ones/all-zeros `ymm` directly (no k-register, no mask-to-vector
//! conversion) — the same representation NEON uses.
//!
//! Spills use a real stack frame, not the red zone: mirrors `avx512.rs`'s
//! reasoning (a `ymm` slot is 32 bytes; keeping the red-zone arithmetic exact
//! for two different slot widths is not worth it for a bit of frame reuse on
//! tiny kernels).
//!
//! A gather is the hardware's: `vgatherdps ymm, [base + ymm*4], ymm` reads
//! one element per lane through a VSIB, under a vector mask the instruction
//! clears as it completes lanes — so each gather sets its mask to all-ones
//! first, from a register the allocator reserved for it, the way the
//! AVX-512 tier resets `k1`.

use super::x86_64;
use super::x86_64::{
    Alu, Direction, Disp, Imm8, Imm32, Lanewise, Mem, NoDisp, Pred, Rounding, Truncate, frame_slot,
};
use super::{
    AsmInsn, AsmProgram, EncodedInst, Gpr, Integer, Physical, Pointer, PtrReg, Reg, Stage, Vector,
    unimplemented_op,
};
use crate::error::CompileError;
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
    fn rm<D: Disp>(self, reg: u8, addr: Mem<Physical, D>) -> EncodedInst {
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

    /// `op reg, [base + ymmINDEX*4], vvvv` — the VSIB form, whose index is
    /// a *vector* register: one address per lane, scale 4, no displacement.
    /// X carries the index's high bit exactly as it does for a GPR index;
    /// the SIB tail is the same bytes with a vector number in the index
    /// field (`x86_64::vsib4_operand_into`, which knows `ymm4`/`ymm12` are
    /// not `rsp`/`r12`). `vvvv` is the gather's mask.
    fn vsib_scaled4(self, reg: u8, vvvv: u8, base: Gpr, index: Reg) -> EncodedInst {
        let mut inst = EncodedInst::new();
        let rbit = if reg >= 8 { 0x00 } else { 0x80 };
        let xbit = if index.0 >= 8 { 0x00 } else { 0x40 };
        let bbit = if base.0 >= 8 { 0x00 } else { 0x20 };
        inst.push(0xC4);
        inst.push(rbit | xbit | bbit | self.map as u8);
        inst.push(
            ((self.w as u8) << 7) | ((!vvvv & 0xF) << 3) | ((self.l256 as u8) << 2) | self.pp as u8,
        );
        inst.push(self.opcode);
        x86_64::vsib4_operand_into(&mut inst, reg, base, index);
        inst
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
    fn rm<D: Disp>(self, reg: u8, addr: Mem<Physical, D>) -> EncodedInst {
        let mut inst = self.vex.rm(reg, addr);
        inst.push(self.imm);
        inst
    }
}

impl Alu {
    const fn vex(self) -> Vex {
        match self {
            Alu::Add => Vex::m0f(0x58),
            Alu::Sub => Vex::m0f(0x5C),
            Alu::Mul => Vex::m0f(0x59),
            Alu::Div => Vex::m0f(0x5E),
            Alu::Min => Vex::m0f(0x5D),
            Alu::Max => Vex::m0f(0x5F),
            Alu::And => Vex::m0f(0x54),
            Alu::AndNot => Vex::m0f(0x55),
            Alu::Or => Vex::m0f(0x56),
            Alu::Xor => Vex::m0f(0x57),
            Alu::IAdd => Vex::m0f_66(0xFE),
        }
    }
}

impl Lanewise {
    const fn vex(self) -> Vex {
        match self {
            Lanewise::Sqrt => Vex::m0f(0x51),
            Lanewise::Rsqrt => Vex::m0f(0x52),
            Lanewise::Recip => Vex::m0f(0x53),
            Lanewise::ToInt => Vex::m0f_f3(0x5B),
            Lanewise::FromInt => Vex::m0f(0x5B),
            Lanewise::WidenBytes => Vex::m0f38_66(0x31),
        }
    }
}

/// An AVX2 (VEX.256) vector instruction.
///
/// Generic over what its operands are ([`Stage`]), like [`x86_64::Gp`]. VEX is
/// three-operand and non-destructive, so no destination here is tied to a
/// source, and an operand may be the register the result is written to.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Inst<S: Stage> {
    /// `op dst, a, b`
    Alu {
        op: Alu,
        dst: S::Write<Vector>,
        a: S::Read<Vector>,
        b: S::Read<Vector>,
    },
    /// `vcmpps dst, a, b, pred`: an all-ones lane where it holds, all-zero
    /// where it does not.
    Cmp {
        pred: Pred,
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
    /// `vroundps dst, src, mode`
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
    /// `vfmadd231ps acc, a, b`: `acc = a·b + acc`, one rounding. FMA3 is not
    /// implied by AVX2 in CPUID, which is why `crate::isa`'s x86-64 probe asks
    /// for both before this backend can be selected.
    Fma231 {
        acc: S::Tie<Vector>,
        a: S::Read<Vector>,
        b: S::Read<Vector>,
    },
    /// `vpcmpeqd dst, dst, dst`: all-ones, whatever `dst` held. Its reads are
    /// not operands, because the result does not depend on them.
    Ones { dst: S::Write<Vector> },
    /// `vmovaps dst, src`
    Mov {
        dst: S::Write<Vector>,
        src: S::Read<Vector>,
    },
    /// `vcvttss2si dst, src`: lane 0, truncated to a 64-bit integer.
    Cvtt {
        dst: S::Write<Integer>,
        src: S::Read<Vector>,
    },
    /// `vmovq dst, src`: eight bytes into the low lanes, the rest zeroed.
    Movq {
        dst: S::Write<Vector>,
        src: S::Read<Integer>,
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
    /// every lane.
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
    /// `vgatherdps dst, [base + index*4], mask`: one `f32` per lane, for
    /// every lane whose `mask` sign bit is set. The instruction clears the
    /// mask as it completes lanes, and `#UD`s unless `dst`, `index` and `mask`
    /// are three registers.
    Gather {
        dst: S::Early<Vector>,
        base: S::Read<Pointer>,
        index: S::Read<Vector>,
        mask: S::Tie<Vector>,
    },
    /// `vmovmskps dst, src`: the lane sign bits, in the low bits of `dst`.
    MoveMask {
        dst: S::Write<Integer>,
        src: S::Read<Vector>,
    },
    /// `vcvttss2si dst, [src]`: the first word of a slot.
    CvttMem {
        dst: S::Write<Integer>,
        src: Mem<S, Imm32>,
    },
    /// `vextractf128 dst, src, 1`: the high 128 bits.
    ExtractHigh {
        dst: S::Write<Vector>,
        src: S::Read<Vector>,
    },
    /// `vextractps [dst], src, lane`: one lane of the low half, stored.
    ExtractLane {
        dst: Mem<S, Imm8>,
        src: S::Read<Vector>,
        lane: u8,
    },
}

impl Inst<Physical> {
    fn encode(self) -> EncodedInst {
        match self {
            Inst::Alu { op, dst, a, b } => op.vex().rrr(dst.0, a.0, b.0),
            Inst::Cmp { pred, dst, a, b } => Vex::m0f(0xC2).imm(pred as u8).rrr(dst.0, a.0, b.0),
            Inst::Unary { op, dst, src } => op.vex().rrr(dst.0, UNUSED_VVVV, src.0),
            Inst::Round { mode, dst, src } => {
                Vex::m0f3a_66(0x08)
                    .imm(mode as u8)
                    .rrr(dst.0, UNUSED_VVVV, src.0)
            }
            // The destination is `vvvv` and the `/digit` is `reg`.
            Inst::Shift {
                direction,
                dst,
                src,
                amount,
            } => Vex::m0f_66(0x72)
                .imm(amount)
                .rrr(direction as u8, dst.0, src.0),
            Inst::Fma231 { acc, a, b } => Vex::m0f38_66(0xB8).rrr(acc.0, a.0, b.0),
            Inst::Ones { dst } => Vex::m0f_66(0x76).rrr(dst.0, dst.0, dst.0),
            Inst::Mov { dst, src } => Vex::m0f(0x28).rrr(dst.0, UNUSED_VVVV, src.0),
            Inst::Cvtt { dst, src } => Vex::m0f_f3(0x2C).w1().rrr(dst.0, UNUSED_VVVV, src.0),
            Inst::Movq { dst, src } => Vex::m0f_66(0x6E).w1().xmm().rrr(dst.0, UNUSED_VVVV, src.0),
            Inst::Load { dst, src } => Vex::m0f(0x10).rm(dst.0, src),
            Inst::Store { dst, src } => Vex::m0f(0x11).rm(src.0, dst),
            Inst::StoreBatch { dst, src } => Vex::m0f(0x11).rm(src.0, dst),
            Inst::Broadcast { dst, src } => Vex::m0f38_66(0x18).rm(dst.0, src),
            Inst::BroadcastIndexed { dst, base, index } => {
                Vex::m0f38_66(0x18).rm_scaled4(dst.0, base.as_gpr(), index)
            }
            Inst::Gather {
                dst,
                base,
                index,
                mask,
            } => {
                debug_assert!(
                    dst != index && dst != mask && index != mask,
                    "vgatherdps: dst, index and mask must be three registers"
                );
                // `base` is never `rbp`/`r13` (the pointer pool is `r9`-`r11`),
                // so the SIB's no-base encoding is unreachable.
                Vex::m0f38_66(0x92).vsib_scaled4(dst.0, mask.0, base.as_gpr(), index)
            }
            Inst::MoveMask { dst, src } => Vex::m0f(0x50).rrr(dst.0, UNUSED_VVVV, src.0),
            Inst::CvttMem { dst, src } => Vex::m0f_f3(0x2C).w1().rm(dst.0, src),
            // The destination is the `rm` operand here, the reverse of the
            // usual direction.
            Inst::ExtractHigh { dst, src } => {
                Vex::m0f3a_66(0x19).imm(1).rrr(src.0, UNUSED_VVVV, dst.0)
            }
            Inst::ExtractLane { dst, src, lane } => {
                debug_assert!(lane < 4, "vextractps reads the low 128 bits");
                Vex::m0f3a_66(0x17).xmm().imm(lane).rm(src.0, dst)
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

/// `vmovaps ymmDST, ymmSRC` — register copy.
fn emit_mov(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    if dst.0 == src.0 {
        return;
    }
    Inst::Mov { dst, src }.emit_into(code);
}

/// `dst = splat(val)`: `vbroadcastss ymm, [pool]` (VEX.256.66.0F38.W0 18 /r),
/// one instruction from the kernel's constant pool. Zero is `vxorps`.
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

/// `dst = splat(base[offset])` at 256 bits: `vbroadcastss ymm<dst>, [base +
/// 4*offset]` (VEX.256.66.0F38.W0 18 /r). `base` is the block's address,
/// wherever the allocator keeps that pointer value.
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

/// `dst = splat(base[idx])` at 256 bits, the index being the same in every
/// lane of `idx`: `vcvttss2si index, xmm<idx>`, `vbroadcastss ymm<dst>,
/// [base + index*4]` (VEX.256.66.0F38.W0 18 /r). See
/// [`x86_64::BroadcastGprs`] for the register contract; `dst` may alias
/// `idx`, since the index is in a GPR before `dst` is written.
fn emit_broadcast_load(code: &mut Vec<u8>, dst: Reg, idx: Reg, gprs: x86_64::BroadcastGprs) {
    Inst::Cvtt {
        dst: gprs.index,
        src: idx,
    }
    .emit_into(code);
    Inst::BroadcastIndexed {
        dst,
        base: gprs.base,
        index: gprs.index,
    }
    .emit_into(code);
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
fn emit_binary(code: &mut Vec<u8>, op: OpKind, dst: Reg, a: Reg, b: Reg) {
    if let Some(pred) = cmp_pred(op) {
        Inst::Cmp { pred, dst, a, b }.emit_into(code);
        return;
    }
    let op = match op {
        OpKind::Add => Alu::Add,
        OpKind::Sub => Alu::Sub,
        OpKind::Mul => Alu::Mul,
        OpKind::Div => Alu::Div,
        OpKind::Min => Alu::Min,
        OpKind::Max => Alu::Max,
        OpKind::BitAnd => Alu::And,
        OpKind::BitOr => Alu::Or,
        OpKind::IAdd => Alu::IAdd,
        _ => unimplemented_op("avx2", op),
    };
    Inst::Alu { op, dst, a, b }.emit_into(code);
}

/// `dst = op(src)`.
///
/// The temp is the allocator's for this instruction; only `Neg` and `Abs`
/// use it, to hold the sign mask they XOR or AND with, which comes from the
/// kernel's constant pool like any other constant.
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
        OpKind::Rsqrt => lanewise(Lanewise::Rsqrt),
        OpKind::Recip => lanewise(Lanewise::Recip),
        OpKind::Neg => signed(Alu::Xor, 0x8000_0000)?,
        OpKind::Abs => signed(Alu::And, 0x7FFF_FFFF)?,
        OpKind::Floor => round(Rounding::Floor),
        OpKind::Ceil => round(Rounding::Ceil),
        OpKind::Round => round(Rounding::Nearest),
        OpKind::TruncToInt => lanewise(Lanewise::ToInt),
        OpKind::IntToFloat => lanewise(Lanewise::FromInt),
        _ => unimplemented_op("avx2", op),
    };
    inst.emit_into(code);
    Ok(())
}

/// How many registers this backend's encodings need beyond their operands.
///
/// `Neg`/`Abs` build a sign mask, and the `If` blends through a temporary;
/// every other encoding here is a single non-destructive VEX instruction.
fn temps_for(op: &super::ScheduledOp) -> u8 {
    use super::ScheduledOp;
    match op {
        ScheduledOp::Unary(OpKind::Neg | OpKind::Abs, _) => 1,
        ScheduledOp::Ternary(OpKind::If, ..) => 1,
        // The gather's truncated-index lanes and its all-ones mask, which the
        // instruction requires distinct from each other and from `dst`.
        ScheduledOp::Gather(..) => 2,
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

/// How many GPRs this backend's encoding of `op` needs beyond
/// [`regalloc::RegisterFile::gpr_ctx`](crate::emit::regalloc::RegisterFile::gpr_ctx).
///
/// `Gather` and `Uniform` need none: the base each addresses is a pointer
/// value the allocator carries, and `vgatherdps` takes its indices as a
/// vector. `Broadcast` needs one for its index, since it addresses the
/// element through a SIB. A `Write` converts its row and column into one
/// each before combining them into the address; the iota carries its eight
/// bytes in through one.
fn gpr_temps_for(op: &super::ScheduledOp) -> u8 {
    use super::ScheduledOp;
    match op {
        ScheduledOp::Write { .. } => 2,
        ScheduledOp::Broadcast(..) | ScheduledOp::Lanes(_) => 1,
        _ => 0,
    }
}

// =============================================================================
// The store, and the iota
// =============================================================================

/// The bytes `0..8`, little end first: what one `movabs` carries in for
/// `vpmovzxbd` to widen into the iota.
const IOTA_BYTES: u64 = 0x0706_0504_0302_0100;

/// Emit a shift of i32 lanes by a compile-time immediate.
fn emit_shift_imm(code: &mut Vec<u8>, op: OpKind, dst: Reg, src: Reg, amount: u8) {
    let direction = match op {
        OpKind::Shl => Direction::Left,
        OpKind::Shr => Direction::Right,
        _ => unimplemented_op("avx2", op),
    };
    Inst::Shift {
        direction,
        dst,
        src,
        amount,
    }
    .emit_into(code);
}

/// `dst = mask ? if_true : if_false` (bit-select; mask already in `dst`, same
/// convention as AVX-512 and NEON).
///
/// `tmp` is the allocator's temp for this instruction, which it picks disjoint
/// from every operand — the `debug_assert` restates that here, where the
/// instruction would silently blend garbage if it ever failed.
fn emit_if(code: &mut Vec<u8>, dst: Reg, if_true: Reg, if_false: Reg, tmp: Option<Reg>) {
    let tmp = super::declared_temp(tmp);
    debug_assert!(tmp.0 != dst.0 && tmp.0 != if_true.0 && tmp.0 != if_false.0);
    let alu = |op, dst, a, b| Inst::Alu { op, dst, a, b };
    AsmProgram::from([
        alu(Alu::And, tmp, dst, if_true),
        alu(Alu::AndNot, dst, dst, if_false),
        alu(Alu::Or, dst, tmp, dst),
    ])
    .assemble(code);
}

/// Fused multiply-add: `dst` already holds `c`; computes `dst = a*b + dst`.
///
/// Always real hardware FMA: the AVX2 tier requires FMA3 (`crate::isa`
/// refuses a host without it), so there is no software mul+add fallback to
/// choose between here. This rounds once, as the folder does
/// (`libm::fmaf`); a software two-step mul-then-add would round twice and
/// disagree in the last bit.
///
/// Pinned as bytes by `emit::tests::muladd_encoding` and as values by
/// `tests/muladd_rounding.rs`.
fn emit_fmadd_c_in_dst(code: &mut Vec<u8>, dst: Reg, a: Reg, b: Reg) {
    Inst::Fma231 { acc: dst, a, b }.emit_into(code);
}

// =============================================================================
// Bound-memory gather (RawGather lowering target)
//
// `vgatherdps ymmDST, [base + ymmIDX*4], ymmMASK` reads one f32 per lane from
// a bound buffer. The lowered index is a float (`clamp(floor(x))·1 + …`), so
// it is first truncated to signed int32 lanes with `vcvttps2dq`. The mask
// must have every lane's sign bit set going in — the instruction clears the
// lanes it completes — so it is set to all-ones before every gather.
// =============================================================================

/// The two registers a gather destroys beside its destination: the
/// truncated indices and the all-ones mask. The instruction `#UD`s unless
/// all three are distinct, and the allocator's temps are disjoint from
/// the destination and each other by construction.
#[derive(Clone, Copy)]
struct GatherTemps {
    /// Vector register for the truncated integer indices.
    idx_int: Reg,
    /// Vector register for the mask, all-ones going in and cleared on exit.
    mask: Reg,
}

/// `dst = base[idx_lane]` for 8 lanes — the whole gather sequence. `idx`
/// holds the *float* indices (the lowering already clamped them in range);
/// `base` the buffer's address. `dst` may alias `idx`: the indices are
/// truncated into `t.idx_int` before the first write to `dst`.
fn emit_gather(code: &mut Vec<u8>, dst: Reg, idx: Reg, base: PtrReg, t: GatherTemps) {
    debug_assert!(
        t.idx_int != idx,
        "the truncated indices must not overwrite the float ones"
    );
    AsmProgram::from([
        Inst::Unary {
            op: Lanewise::ToInt,
            dst: t.idx_int,
            src: idx,
        },
        Inst::Ones { dst: t.mask },
        Inst::Gather {
            dst,
            base,
            index: t.idx_int,
            mask: t.mask,
        },
    ])
    .assemble(code);
}

#[cfg(test)]
mod tests {
    //! Hardware validation, mirroring `avx512.rs`'s runtime test tier: JIT real
    //! `ymm` kernels and execute them on the host.
    use super::*;

    /// Offset 3 shifted up by a full 16-bit range: where a 16-bit slot used
    /// to wrap back to argument 3.
    const PAST_U16: u64 = 3 + (u16::MAX as u64 + 1);

    /// The uniform read for `dst = 5` through the block in `rax`: offset 3
    /// is `vbroadcastss ymm5, [rax + 12]` (checked against `llvm-mc
    /// --disassemble`, LLVM 18), and an offset past the old 16-bit width
    /// carries its full `disp32` with the same prefix and ModRM. The block's
    /// address is a pointer-class value the allocator placed, so no load of
    /// it appears here: that is the `Context` def's, once per call.
    #[test]
    fn a_uniform_read_is_one_broadcast_load() {
        let mut code = Vec::new();
        emit_uniform_load(&mut code, Reg(5), PtrReg(0), 3).expect("fits");
        assert_eq!(code, [0xC4, 0xE2, 0x7D, 0x18, 0xA8, 0x0C, 0, 0, 0]);

        let mut code = Vec::new();
        emit_uniform_load(&mut code, Reg(5), PtrReg(0), PAST_U16).expect("fits");
        assert_eq!(code, [0xC4, 0xE2, 0x7D, 0x18, 0xA8, 0x0C, 0x00, 0x04, 0x00]);
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
    /// then `vbroadcastss ymm5, [rax + rcx*4]` (checked against `objdump
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
                0xC4, 0xE1, 0xFE, 0x2C, 0xCE, 0xC4, 0xE2, 0x7D, 0x18, 0x2C, 0x88
            ]
        );

        let high = x86_64::BroadcastGprs {
            base: PtrReg(9),
            index: Gpr(11),
        };
        let mut code = Vec::new();
        emit_broadcast_load(&mut code, Reg(5), Reg(6), high);
        // `vcvttss2si r11, xmm6`, whose VEX.R carries the GPR's high bit.
        assert_eq!(&code[..5], [0xC4, 0x61, 0xFE, 0x2C, 0xDE]);
        assert_eq!(&code[5..], [0xC4, 0x82, 0x7D, 0x18, 0x2C, 0x99]);
    }

    /// Executes the bytes on this host's CPU — AVX2 with FMA is the floor
    /// every x86-64 host the JIT runs on has (`isa::detect` refuses one
    /// below it); the encodings themselves are pinned bytewise on every host
    /// by the tests above. The `extern
    /// "C"` kernels take `ymm` values, which the ABI only lets a caller
    /// compiled with AVX pass — hence `#[target_feature]` on the functions
    /// that call them, and nowhere else.
    #[cfg(target_arch = "x86_64")]
    mod runtime {
        use super::super::*;

        /// `ret` (`C3`): the end of a hand-assembled test kernel, which
        /// returns its `__m256`/`__m512` in the vector register a
        /// `vzeroupper` would clear.
        const RET: u8 = 0xC3;
        use crate::emit::AsmInsn;
        use crate::emit::executable::CompiledKernel;
        use core::arch::x86_64::*;

        #[allow(improper_ctypes_definitions)]
        type K = unsafe extern "C" fn(__m256, __m256, __m256, __m256) -> __m256;

        fn run(body: &[u8], xs: [f32; 8], ys: [f32; 8], zs: [f32; 8]) -> [f32; 8] {
            let mut code = body.to_vec();
            code.push(RET);
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
            let pool_label = asm.mint();
            x86_64::anchor(&mut asm, pool_label);
            asm.run.extend_from_slice(body);
            asm.run.push(RET);
            pool.finish(&mut asm, pool_label);
            // SAFETY: every caller is a test that checked the host runs AVX2.
            unsafe { run_code(&asm.finish(), xs, ys, zs) }
        }

        /// # Safety
        ///
        /// The host must execute AVX2: `code` is `ymm` code, and this function
        /// is compiled with AVX enabled to be allowed to pass `ymm` values.
        #[target_feature(enable = "avx2")]
        unsafe fn run_code(code: &[u8], xs: [f32; 8], ys: [f32; 8], zs: [f32; 8]) -> [f32; 8] {
            let exec = unsafe { CompiledKernel::from_code(code).expect("mmap") };
            unsafe {
                let f: K = core::mem::transmute(exec.as_bytes().as_ptr());
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
        /// be used for compares/`If`s' underlying mask bit pattern.
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
            #[allow(improper_ctypes_definitions)]
            type MaskCheck = unsafe extern "C" fn(__m256, __m256) -> i32;

            /// # Safety
            ///
            /// The host must execute AVX2 (checked above).
            #[target_feature(enable = "avx2")]
            unsafe fn run_mask(body: &[u8], xs: [f32; 8], ys: [f32; 8]) -> i32 {
                let mut code = body.to_vec();
                code.push(RET);
                let exec = unsafe { CompiledKernel::from_code(&code).expect("mmap") };
                unsafe {
                    let f: MaskCheck = core::mem::transmute(exec.as_bytes().as_ptr());
                    f(_mm256_loadu_ps(xs.as_ptr()), _mm256_loadu_ps(ys.as_ptr()))
                }
            }

            // Half true, half false, so the result exercises every mask bit
            // rather than only the all-true/all-false extremes.
            let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
            let ys = [9.0, 9.0, 9.0, 9.0, -9.0, -9.0, -9.0, -9.0];
            let mut c = Vec::new();
            emit_binary(&mut c, OpKind::Lt, X, X, Y);
            Inst::MoveMask {
                dst: x86_64::gpr::RAX,
                src: X,
            }
            .emit_into(&mut c);
            // SAFETY: the host runs AVX2, checked at the top of this test.
            let got = unsafe { run_mask(&c, xs, ys) };
            assert_eq!(got, 0b0000_1111, "lt mask, lanes 0-3 true");

            // The complementary comparison, to pin the other half of eax
            // independently of the first assertion.
            let mut c = Vec::new();
            emit_binary(&mut c, OpKind::Gt, X, X, Y);
            Inst::MoveMask {
                dst: x86_64::gpr::RAX,
                src: X,
            }
            .emit_into(&mut c);
            // SAFETY: as above.
            let got = unsafe { run_mask(&c, xs, ys) };
            assert_eq!(got, 0b1111_0000, "gt mask, lanes 4-7 true");
        }

        #[test]
        fn emit_unary_computes_sqrt_neg_and_abs_per_lane() {
            let (xs, ys, zs) = lanes();
            let unary = |op, src, temp| crate::emit::Unary {
                op,
                dst: X,
                src,
                temp,
            };
            let mut pool = x86_64::ConstPool::default();
            let mut c = Vec::new();
            emit_unary(&mut c, unary(OpKind::Sqrt, Y, None), &mut pool).unwrap();
            check(run_pooled(&c, &pool, xs, ys, zs), |i| ys[i].sqrt(), "sqrt");

            let mut pool = x86_64::ConstPool::default();
            let mut c = Vec::new();
            emit_unary(&mut c, unary(OpKind::Neg, X, Some(TEMP)), &mut pool).unwrap();
            check(run_pooled(&c, &pool, xs, ys, zs), |i| -xs[i], "neg");

            let mut pool = x86_64::ConstPool::default();
            let mut c = Vec::new();
            emit_unary(&mut c, unary(OpKind::Abs, X, Some(TEMP)), &mut pool).unwrap();
            check(run_pooled(&c, &pool, xs, ys, zs), |i| xs[i].abs(), "abs");
        }

        #[test]
        fn emit_if_blends_if_true_and_if_false_by_the_mask() {
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            emit_binary(&mut c, OpKind::Lt, Reg(5), X, Y); // mask
            emit_mov(&mut c, Reg(6), Reg(5));
            emit_if(&mut c, Reg(6), X, Y, Some(TEMP)); // dst = mask ? x : y
            emit_mov(&mut c, X, Reg(6));
            check(
                run(&c, xs, ys, zs),
                |i| if xs[i] < ys[i] { xs[i] } else { ys[i] },
                "if",
            );
        }

        /// Two constants, the first read twice: the pool holds each once, and
        /// every read is one broadcast from it.
        #[test]
        fn emit_const_broadcasts_and_adds_to_every_lane() {
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
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            crate::emit::x86_64::Gp::Enter {
                size: 32,
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
            // `add rsp, 32` (REX.W 81 /0 id), not `Gp::Ret`: this kernel
            // returns its answer in a ymm register, which a `vzeroupper`
            // would clear.
            AsmProgram::from([EncodedInst::from_slice(&[0x48, 0x81, 0xC4, 32, 0, 0, 0])])
                .assemble(&mut c);
            check(run(&c, xs, ys, zs), |i| xs[i] * ys[i], "spill roundtrip");
        }

        /// The gather sequence executes: `dst` may alias the float index
        /// register, and every lane reads its own element. Matches the
        /// production ABI (mod.rs's `ResolvedOp::Gather`): the base is a
        /// pointer register the allocator placed — here the first argument,
        /// `rdi`, holding the buffer's own address.
        #[test]
        fn emit_gather_reads_the_value_at_each_lanes_index() {
            #[allow(improper_ctypes_definitions)]
            type G = unsafe extern "C" fn(*const f32, __m256) -> __m256;

            /// # Safety
            ///
            /// The host must execute AVX2 (checked above).
            #[target_feature(enable = "avx2")]
            unsafe fn run_gather(
                exec: &CompiledKernel,
                base: *const f32,
                idx: [f32; 8],
            ) -> [f32; 8] {
                unsafe {
                    let f: G = core::mem::transmute(exec.as_bytes().as_ptr());
                    let r = f(base, _mm256_loadu_ps(idx.as_ptr()));
                    let mut out = [0.0f32; 8];
                    _mm256_storeu_ps(out.as_mut_ptr(), r);
                    out
                }
            }

            let buf: Vec<f32> = (0..64).map(|i| (i as f32) * 1.5 + 0.25).collect();
            let idx: [f32; 8] = [0.0, 63.0, 1.0, 2.0, 10.0, 5.0, 32.0, 7.0];
            // Twice: into a register of its own, and over the float index.
            for dst in [Reg(5), Reg(0)] {
                let mut c = Vec::new();
                emit_gather(
                    &mut c,
                    dst,
                    Reg(0),
                    PtrReg(7),
                    GatherTemps {
                        idx_int: Reg(13),
                        mask: Reg(14),
                    },
                );
                if dst != Reg(0) {
                    emit_mov(&mut c, Reg(0), dst);
                }
                c.push(RET);

                let exec = unsafe { CompiledKernel::from_code(&c).expect("mmap") };
                // SAFETY: the host runs AVX2, checked at the top of this test.
                let out = unsafe { run_gather(&exec, buf.as_ptr(), idx) };
                for i in 0..8 {
                    let want = buf[idx[i] as usize];
                    assert_eq!(
                        out[i], want,
                        "gather into {dst:?}, lane {i}: idx {}",
                        idx[i]
                    );
                }
            }
        }
    }

    /// The gather's bytes against `objdump -M intel` (binutils 2.42) and
    /// `llvm-mc`: `vpcmpeqd ymm7, ymm7, ymm7`, `vcvttps2dq ymm6, ymm0`,
    /// then `vgatherdps ymm5, [r9 + ymm6*4], ymm7`. VEX.R, X and B each
    /// carry one operand's high bit — the destination's, the vector
    /// index's and the base's — pinned by the all-high and mixed forms.
    #[test]
    fn the_gather_encodes_as_the_manual_says() {
        let mut c = Vec::new();
        emit_gather(
            &mut c,
            Reg(5),
            Reg(0),
            PtrReg(9),
            GatherTemps {
                idx_int: Reg(6),
                mask: Reg(7),
            },
        );
        assert_eq!(
            c,
            [
                0xC4, 0xE1, 0x7E, 0x5B, 0xF0, // vcvttps2dq ymm6, ymm0
                0xC4, 0xE1, 0x45, 0x76, 0xFF, // vpcmpeqd ymm7, ymm7, ymm7
                0xC4, 0xC2, 0x45, 0x92, 0x2C, 0xB1, // vgatherdps ymm5, [r9+ymm6*4], ymm7
            ]
        );
        let mut c = Vec::new();
        let gather = |dst, base, index, mask| Inst::Gather {
            dst: Reg(dst),
            base: PtrReg(base),
            index: Reg(index),
            mask: Reg(mask),
        };
        AsmProgram::from([gather(13, 11, 14, 15), gather(0, 7, 13, 14)]).assemble(&mut c);
        assert_eq!(
            c,
            [
                0xC4, 0x02, 0x05, 0x92, 0x2C, 0xB3, // vgatherdps ymm13, [r11+ymm14*4], ymm15
                0xC4, 0xA2, 0x0D, 0x92, 0x04, 0xAF, // vgatherdps ymm0, [rdi+ymm13*4], ymm14
            ]
        );
    }

    /// A VSIB index of `ymm4` or `ymm12` puts `100` in the SIB's index
    /// field — which for a GPR index would mean `rsp`, "no index", and is
    /// refused there. As a vector number it is just a register, and the
    /// cell grid's gathers are indexed by whichever the allocator picked.
    /// The bytes are `objdump`'s: `vgatherdps ymm5, [r9 + ymm4*4], ymm7` and
    /// the same through `ymm12`, which differ only in the prefix's X bit.
    #[test]
    fn a_vsib_index_may_be_the_fourth_or_twelfth_register() {
        let mut c = Vec::new();
        let gather = |index| Inst::Gather {
            dst: Reg(5),
            base: PtrReg(9),
            index: Reg(index),
            mask: Reg(7),
        };
        AsmProgram::from([gather(4), gather(12)]).assemble(&mut c);
        assert_eq!(
            c,
            [
                0xC4, 0xC2, 0x45, 0x92, 0x2C, 0xA1, // vgatherdps ymm5, [r9+ymm4*4], ymm7
                0xC4, 0x82, 0x45, 0x92, 0x2C, 0xA1, // vgatherdps ymm5, [r9+ymm12*4], ymm7
            ]
        );
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

    /// The AVX2 register file (ymm, 256-bit).
    ///
    /// SysV has no callee-saved vector registers and the collapse ABI passes
    /// no vector, so every one of the sixteen is the allocator's. The gather
    /// borrows two of them across its own sequence (the truncated indices
    /// and the mask), the sign mask and the `If` blend borrow one — all
    /// reservations the allocator makes for one instruction, so all of them
    /// are its the rest of the time.
    const AVX2_FILE: regalloc::RegisterFile = regalloc::RegisterFile {
        scratch: regalloc::RegSet::range(0, 16),
        // Nothing: every register an encoding destroys is a `temps_for`
        // reservation for that one instruction.
        fixed: &[],
        temps_for: super::temps_for,
        // A guard reduces its mask with `vmovmskps` into the flags, which
        // costs no vector register at all.
        guard_temps: 0,
        vector_bytes: 32,
        // SysV's first three integer arguments, in the ABI's order: the
        // context (the array of buffer base pointers, then the uniform and
        // origin blocks), the output plane, its pitch. Declared here so
        // `checked` proves `gpr_scratch` misses all three, rather than a
        // comment asserting the constants never collide.
        gpr_ctx: Some(x86::gpr::RDI),
        gpr_out: Some(x86::gpr::RSI),
        gpr_pitch: Some(x86::gpr::RDX),
        // rax/rcx: the broadcast's index, the store's row and column, the
        // iota's bytes — `Scratch` reservations like every vector temp.
        // `vgatherdps` addresses through a vector index, so the gather needs
        // none.
        gpr_scratch: regalloc::GprSet::of(&[x86::gpr::RAX, x86::gpr::RCX]),
        gpr_temps_for: super::gpr_temps_for,
        // r9-r11: the caller-saved GPRs SysV leaves after the three
        // arguments, the two scratch and `r8` (the constant pool's anchor).
        // The pointer class's pool — buffer bases and block addresses are
        // carried here across the loops that read them
        // (docs/plans/2026-09-22-a-pointer-is-a-value.md). The callee-saved
        // six would double it at the price of a prologue; not yet measured.
        pointers: regalloc::GprSet::of(&[x86::gpr::R9, x86::gpr::R10, x86::gpr::R11]),
        // No mask-register file on this tier: masks are ordinary vectors.
        mask_scratch: regalloc::MaskSet::EMPTY,
        mask_temps_for: regalloc::no_temps,
        mask_guard_temps: 0,
    }
    .checked();

    /// AVX2 implementation of the shared driver's leaf operations.
    pub(in crate::emit) struct Avx2Backend {
        consts: x86::ConstPool,
    }

    impl Avx2Backend {
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

    impl IsaBackend for Avx2Backend {
        fn jump(&mut self, asm: &mut Assembly, label: Label) {
            asm.push(x86::Gp::Jmp { to: label });
        }

        fn register_file(&self) -> regalloc::RegisterFile {
            AVX2_FILE
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
                // The iota: the bytes `0..8` in through a GPR, widened to
                // dwords, converted. No vector temp — `dst` is every stage's.
                ResolvedOp::Lanes { dst } => {
                    let gpr = crate::emit::declared_gpr_temp(plan.scratch.gpr_temp(0));
                    x86::Gp::Movabs {
                        dst: gpr,
                        imm: IOTA_BYTES,
                    }
                    .emit_into(code);
                    let (dst, src) = (*dst, gpr);
                    AsmProgram::from([
                        Inst::Movq { dst, src },
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
                    // dst = base[idx]: `vgatherdps` under an all-ones mask,
                    // `base` being the buffer's address wherever the
                    // allocator keeps it; the two vector temps are this
                    // instruction's reservations (`super::temps_for`).
                    super::emit_gather(
                        code,
                        *dst,
                        *idx,
                        *base,
                        super::GatherTemps {
                            idx_int: crate::emit::declared_temp(plan.scratch.temp(0)),
                            mask: crate::emit::declared_temp(plan.scratch.temp(1)),
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
                    super::emit_uniform_load(code, *dst, *base, *offset)?;
                }
                ResolvedOp::Context { dst, slot } => {
                    let ctx = AVX2_FILE
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
                    // VEX 3-operand: either source may alias `dst`.
                    super::emit_binary(code, *op, *dst, *left, *right);
                }
                ResolvedOp::FusedMulAdd { dst, a, b } => {
                    super::emit_fmadd_c_in_dst(code, *dst, *a, *b);
                }
                ResolvedOp::If {
                    dst,
                    if_true,
                    if_false,
                } => {
                    // setup_mov already placed the vector mask in dst.
                    super::emit_if(code, *dst, *if_true, *if_false, plan.scratch.temp(0));
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

        // If short-circuit guards: vmovmskps -> eax[7:0], then a test
        // of the low byte (al == 0xFF for all-true — see [`x86::Gp::CmpByte`]
        // for why the sign-extending `cmp eax, imm8` would not do).
        /// [`MaskTest::scratch`] and [`MaskTest::mask_scratch`] are both
        /// unused: this tier reduces the mask with `movmskps` into the
        /// flags, needing neither a vector nor a mask register.
        fn branch_if_arm_is_dead(&mut self, asm: &mut Assembly, test: MaskTest, label: Label) {
            let mask = x86::gpr::RAX;
            asm.push(Inst::MoveMask {
                dst: mask,
                src: test.reg,
            });
            asm.push(match test.arm {
                // ZF set when eax == 0: no lane is true, so the true arm is dead.
                IfArm::True => x86::Gp::Test {
                    flags: (),
                    src: mask,
                },
                // ZF set when al == 0xFF: every lane is true, so the false arm is.
                IfArm::False => x86::Gp::CmpByte {
                    flags: (),
                    src: mask,
                    imm: 0xFF,
                },
            });
            asm.push(x86::Gp::je(label));
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

        /// A full batch is one `vmovups`. A remainder is `vextractps` per
        /// lane: the low four straight out of the value, the rest out of its
        /// high half extracted into the reserved temp.
        fn emit_write(&mut self, code: &mut Vec<u8>, write: &WritePlan) {
            let base = write_address::<Inst<Physical>>(code, &AVX2_FILE, write);
            let lanes = AVX2_FILE.vector_bytes / 4;
            if write.lanes == lanes {
                Inst::StoreBatch {
                    dst: Mem { base, disp: NoDisp },
                    src: write.value,
                }
                .emit_into(code);
                return;
            }
            let mut half = write.value;
            for lane in 0..write.lanes {
                if lane == 4 {
                    half = crate::emit::declared_temp(write.scratch.temp(0));
                    Inst::ExtractHigh {
                        dst: half,
                        src: write.value,
                    }
                    .emit_into(code);
                }
                Inst::ExtractLane {
                    dst: Mem {
                        base,
                        disp: x86::Imm8((lane * 4) as i8),
                    },
                    src: half,
                    lane: (lane % 4) as u8,
                }
                .emit_into(code);
            }
        }

        fn emit_ret(&mut self, code: &mut Vec<u8>, bytes: u32) {
            x86::Gp::Ret {
                size: bytes,
                flags: (),
            }
            .emit_into(code);
        }
    }
}
