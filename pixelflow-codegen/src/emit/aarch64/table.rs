//! Denotational AArch64 instruction shapes and instruction table.
//!
//! Instructions are categorized by their algebraic morphism:
//! - [`Binary<const OPCODE: u32>`]: $V \times V \to V$
//! - [`Unary<const OPCODE: u32>`]: $V \to V$
//! - [`Reduce<const OPCODE: u32>`]: $V \to S$
//!
//! Magic hex opcodes live strictly on the typed instruction definitions in this table.

use crate::emit::{AsmInsn, Gpr, PtrReg, Reg};
use alloc::vec::Vec;

// =============================================================================
// Algebraic Instruction Shapes
// =============================================================================

/// 12-bit unsigned immediate for AArch64 arithmetic instructions.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct Imm12(pub(super) u16);

/// 64-bit integer addition of an immediate: `ADD Xd, Xn, #imm12`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct AddI64 {
    dst: Gpr,
    src: Gpr,
    operand: Imm12,
}

impl AddI64 {
    #[must_use]
    #[inline]
    pub(super) fn new(dst: impl Into<Gpr>, src: impl Into<Gpr>, operand: Imm12) -> Self {
        Self {
            dst: dst.into(),
            src: src.into(),
            operand,
        }
    }
}

impl AsmInsn for AddI64 {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        let w = 0x9100_0000
            | ((self.operand.0 as u32 & 0xFFF) << 10)
            | ((self.src.0 as u32 & 0x1F) << 5)
            | (self.dst.0 as u32 & 0x1F);
        code.extend_from_slice(&w.to_le_bytes());
    }
}

/// 64-bit integer subtraction of an immediate: `SUB Xd, Xn, #imm12`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct SubI64 {
    dst: Gpr,
    src: Gpr,
    operand: Imm12,
}

impl SubI64 {
    #[must_use]
    #[inline]
    pub(super) fn new(dst: impl Into<Gpr>, src: impl Into<Gpr>, operand: Imm12) -> Self {
        Self {
            dst: dst.into(),
            src: src.into(),
            operand,
        }
    }
}

impl AsmInsn for SubI64 {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        let w = 0xD100_0000
            | ((self.operand.0 as u32 & 0xFFF) << 10)
            | ((self.src.0 as u32 & 0x1F) << 5)
            | (self.dst.0 as u32 & 0x1F);
        code.extend_from_slice(&w.to_le_bytes());
    }
}

/// Bitwise NOT of 32-bit general-purpose register: `MVN Wd, Wm` (encoded as `ORN Wd, WZR, Wm`)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct MvnW {
    dst: Gpr,
    src: Gpr,
}

impl MvnW {
    #[must_use]
    #[inline]
    pub(super) fn new(dst: impl Into<Gpr>, src: impl Into<Gpr>) -> Self {
        Self {
            dst: dst.into(),
            src: src.into(),
        }
    }

    #[must_use]
    #[inline]
    pub(super) fn encode(self) -> u32 {
        0x2A20_03E0 | ((self.src.0 as u32 & 0x1F) << 16) | (self.dst.0 as u32 & 0x1F)
    }
}

/// Binary vector operation: V × V → V
///
/// Denotes `dst = lhs ⊗ rhs` across all vector lanes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct Binary<const OPCODE: u32, D = Reg, L = Reg, R = Reg> {
    dst: D,
    lhs: L,
    rhs: R,
}

impl<const OPCODE: u32> Binary<OPCODE, Reg, Reg, Reg> {
    #[must_use]
    #[inline]
    pub(super) const fn new(dst: Reg, lhs: Reg, rhs: Reg) -> Self {
        Self { dst, lhs, rhs }
    }

    #[must_use]
    #[inline]
    pub(super) fn encode(self) -> u32 {
        OPCODE
            | (self.dst.0 as u32 & 0x1F)
            | ((self.lhs.0 as u32 & 0x1F) << 5)
            | ((self.rhs.0 as u32 & 0x1F) << 16)
    }
}

/// Unary vector operation: V → V
///
/// Denotes `dst = f(src)` across all vector lanes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct Unary<const OPCODE: u32, D = Reg, S = Reg> {
    dst: D,
    src: S,
}

impl<const OPCODE: u32> Unary<OPCODE, Reg, Reg> {
    #[must_use]
    #[inline]
    pub(super) const fn new(dst: Reg, src: Reg) -> Self {
        Self { dst, src }
    }

    #[must_use]
    #[inline]
    pub(super) fn encode(self) -> u32 {
        OPCODE | (self.dst.0 as u32 & 0x1F) | ((self.src.0 as u32 & 0x1F) << 5)
    }
}

/// Vector-to-scalar horizontal reduction: V → S
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct Reduce<const OPCODE: u32, D = Reg, S = Reg> {
    dst: D,
    src: S,
}

impl<const OPCODE: u32> Reduce<OPCODE, Reg, Reg> {
    #[must_use]
    #[inline]
    pub(super) const fn new(dst: Reg, src: Reg) -> Self {
        Self { dst, src }
    }

    #[must_use]
    #[inline]
    pub(super) fn encode(self) -> u32 {
        OPCODE | (self.dst.0 as u32 & 0x1F) | ((self.src.0 as u32 & 0x1F) << 5)
    }
}

/// Bitwise select: `mask = (mask & if_true) | (~mask & if_false)`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct Bsl<M = Reg, T = Reg, F = Reg> {
    mask: M,
    if_true: T,
    if_false: F,
}

impl Bsl<Reg, Reg, Reg> {
    #[must_use]
    #[inline]
    pub(super) const fn new(mask: Reg, if_true: Reg, if_false: Reg) -> Self {
        Self {
            mask,
            if_true,
            if_false,
        }
    }

    #[must_use]
    #[inline]
    pub(super) fn encode(self) -> u32 {
        0x6E60_1C00
            | (self.mask.0 as u32 & 0x1F)
            | ((self.if_true.0 as u32 & 0x1F) << 5)
            | ((self.if_false.0 as u32 & 0x1F) << 16)
    }
}

/// Broadcast a lane of a vector register across all lanes: `DUP Vd.4S, Vn.s[0]`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct DupLane0 {
    dst: Reg,
    src: Reg,
}

impl DupLane0 {
    #[must_use]
    #[inline]
    pub(super) const fn new(dst: Reg, src: Reg) -> Self {
        Self { dst, src }
    }

    #[must_use]
    #[inline]
    pub(super) fn encode(self) -> u32 {
        0x4E04_0400 | ((self.src.0 as u32) << 5) | (self.dst.0 as u32)
    }
}

/// Move vector lane 0 to general purpose register: `FMOV X16, D<src>`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct FmovToGp {
    src: Reg,
}

impl FmovToGp {
    #[must_use]
    #[inline]
    pub(super) const fn new(src: Reg) -> Self {
        Self { src }
    }

    #[must_use]
    #[inline]
    pub(super) fn encode(self) -> u32 {
        0x1E26_0000 | ((self.src.0 as u32) << 5) | 16
    }
}

/// Return from subroutine: `RET`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct Ret;

impl Ret {
    const OPCODE: u32 = 0xD65F_03C0;

    #[must_use]
    #[inline]
    pub(super) const fn encode(self) -> u32 {
        Self::OPCODE
    }
}

impl AsmInsn for Ret {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        code.extend_from_slice(&self.encode().to_le_bytes());
    }
}

/// A 32-bit scalar float register (the `s0`..`s31` view of a vector register).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct SReg(pub(super) Reg);

/// Bytes moved by a `q` (128-bit vector) access — also the scale of its offset.
const Q_BYTES: u32 = 16;
/// Bytes moved by an `x` (64-bit general/pointer) access.
pub(super) const X_BYTES: u32 = 8;
/// Bytes moved by an `s` (32-bit scalar SIMD&FP) access.
pub(super) const S_BYTES: u32 = 4;
/// The largest value a 12-bit scaled immediate holds.
const MAX_IMM12: u32 = 4095;
/// The largest 16-byte-aligned displacement `add`'s own 12-bit immediate holds.
pub(in crate::emit) const MAX_ADD_IMM: u32 = 4080;

/// An address spelled `[base, #offset]` — the scaled-immediate addressing mode.
///
/// `offset` is in BYTES. aarch64 encodes it divided by the access size.
/// The base being a [`PtrReg`] guarantees an integer counter or index cannot
/// be mistakenly passed as an address.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(in crate::emit) struct Mem {
    /// The register holding the base address.
    pub(in crate::emit) base: PtrReg,
    /// Displacement in bytes; must be a multiple of the access size.
    pub(in crate::emit) offset: u32,
}

/// An address spelled `[base, w<index>, uxtw #2]` — a 32-bit index register,
/// zero-extended to 64 bits and scaled by 4.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct MemIndexed {
    /// The register holding the buffer base pointer.
    pub(super) base: PtrReg,
    /// The element index, read as the 32-bit `w<index>`.
    pub(super) index: Gpr,
}

/// Rewrite `addr` as `[x16]`, computing `base + offset` into IP0 first.
///
/// The fallback for a displacement past the 12-bit scaled immediate — a spill
/// frame deeper than 64 KiB. `add`'s immediate is 12 bits too, so a large
/// displacement takes several of them.
fn address_in_ip0(code: &mut Vec<u8>, Mem { base, offset }: Mem) -> Mem {
    let mut remaining = offset;
    let first = remaining.min(MAX_ADD_IMM);
    AddI64::new(Gpr(16), base.as_gpr(), Imm12(first as u16)).emit_into(code);
    remaining -= first;
    while remaining > 0 {
        let chunk = remaining.min(MAX_ADD_IMM);
        AddI64::new(Gpr(16), Gpr(16), Imm12(chunk as u16)).emit_into(code);
        remaining -= chunk;
    }
    Mem {
        base: PtrReg(16),
        offset: 0,
    }
}

/// `STR Qt, [Xn, #imm12*16]`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct StrQ {
    pub(super) src: Reg,
    pub(super) addr: Mem,
}

impl AsmInsn for StrQ {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        assert!(
            self.addr.offset.is_multiple_of(Q_BYTES),
            "128-bit access offset {} is not 16-byte aligned",
            self.addr.offset
        );
        let a = if self.addr.offset / Q_BYTES > MAX_IMM12 {
            address_in_ip0(code, self.addr)
        } else {
            self.addr
        };
        let w = 0x3D80_0000
            | ((a.offset / Q_BYTES) << 10)
            | ((a.base.0 as u32) << 5)
            | (self.src.0 as u32);
        code.extend_from_slice(&w.to_le_bytes());
    }
}

/// `STR Xt, [Xn, #imm12*8]`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct StrX {
    pub(super) src: PtrReg,
    pub(super) addr: Mem,
}

impl AsmInsn for StrX {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        assert!(
            self.addr.offset.is_multiple_of(X_BYTES),
            "pointer store offset {} not 8-byte aligned",
            self.addr.offset
        );
        let imm12 = self.addr.offset / X_BYTES;
        assert!(
            imm12 <= MAX_IMM12,
            "pointer store offset {} exceeds STR imm12 range",
            self.addr.offset
        );
        let w =
            0xF900_0000 | (imm12 << 10) | ((self.addr.base.0 as u32) << 5) | (self.src.0 as u32);
        code.extend_from_slice(&w.to_le_bytes());
    }
}

/// `LDR Qt, [Xn, #imm12*16]`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct LdrQ {
    pub(super) dst: Reg,
    pub(super) addr: Mem,
}

impl AsmInsn for LdrQ {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        assert!(
            self.addr.offset.is_multiple_of(Q_BYTES),
            "128-bit access offset {} is not 16-byte aligned",
            self.addr.offset
        );
        let a = if self.addr.offset / Q_BYTES > MAX_IMM12 {
            address_in_ip0(code, self.addr)
        } else {
            self.addr
        };
        let w = 0x3DC0_0000
            | ((a.offset / Q_BYTES) << 10)
            | ((a.base.0 as u32) << 5)
            | (self.dst.0 as u32);
        code.extend_from_slice(&w.to_le_bytes());
    }
}

/// `LDR Xt, [Xn, #imm12*8]`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct LdrX {
    pub(super) dst: PtrReg,
    pub(super) addr: Mem,
}

impl AsmInsn for LdrX {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        assert!(
            self.addr.offset.is_multiple_of(X_BYTES),
            "pointer load offset {} not 8-byte aligned",
            self.addr.offset
        );
        let imm12 = self.addr.offset / X_BYTES;
        assert!(
            imm12 <= MAX_IMM12,
            "pointer load offset {} exceeds LDR imm12 range",
            self.addr.offset
        );
        let w =
            0xF940_0000 | (imm12 << 10) | ((self.addr.base.0 as u32) << 5) | (self.dst.0 as u32);
        code.extend_from_slice(&w.to_le_bytes());
    }
}

/// `LDR St, [Xn, #imm12*4]`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct LdrS {
    pub(super) dst: SReg,
    pub(super) addr: Mem,
}

impl AsmInsn for LdrS {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        assert!(
            self.addr.offset.is_multiple_of(S_BYTES),
            "32-bit access offset {} is not 4-byte aligned",
            self.addr.offset
        );
        let a = if self.addr.offset / S_BYTES > MAX_IMM12 {
            address_in_ip0(code, self.addr)
        } else {
            self.addr
        };
        let w = 0xBD40_0000
            | ((a.offset / S_BYTES) << 10)
            | ((a.base.0 as u32) << 5)
            | (self.dst.0.0 as u32);
        code.extend_from_slice(&w.to_le_bytes());
    }
}

/// `LDR Wt, [Xn, Wm, UXTW #2]`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct LdrW {
    pub(super) dst: Gpr,
    pub(super) addr: MemIndexed,
}

impl AsmInsn for LdrW {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        let w = 0xB860_5800
            | ((self.addr.index.0 as u32) << 16)
            | ((self.addr.base.0 as u32) << 5)
            | (self.dst.0 as u32);
        code.extend_from_slice(&w.to_le_bytes());
    }
}

/// `ldr s<dst>, [base, w<index>, uxtw #2]` — one element of a plane of
/// `f32`s straight into lane 0, where a `dup` can spread it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct LdrSIndexed {
    pub(super) dst: SReg,
    pub(super) addr: MemIndexed,
}

impl AsmInsn for LdrSIndexed {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        // The same register-offset form as `LdrW` with the SIMD&FP bit
        // (bit 26) set: `LDR St, [Xn, Wm, UXTW #2]`.
        let w = 0xBC60_5800
            | ((self.addr.index.0 as u32) << 16)
            | ((self.addr.base.0 as u32) << 5)
            | (self.dst.0.0 as u32);
        code.extend_from_slice(&w.to_le_bytes());
    }
}

/// UMOV Wd, Vn.S[lane] — extract a 32-bit vector lane into a GP register.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct UmovW {
    dst: Gpr,
    src: Reg,
    lane: u8,
}

impl UmovW {
    #[must_use]
    #[inline]
    pub(super) const fn new(dst: Gpr, src: Reg, lane: u8) -> Self {
        Self { dst, src, lane }
    }

    #[must_use]
    #[inline]
    pub(super) fn encode(self) -> u32 {
        // A lane of 4 carries out of imm5 into bit 21 and encodes a different
        // instruction. Emission is compile time, so the check costs no render.
        assert!(
            self.lane < 4,
            "UMOV Wd, Vn.S[{}]: a 128-bit register has four 32-bit lanes",
            self.lane
        );
        let imm5 = ((self.lane as u32) << 3) | 0b100;
        0x0E00_3C00 | (imm5 << 16) | ((self.src.0 as u32) << 5) | (self.dst.0 as u32)
    }
}

/// MOV Xd, Xm — `ORR Xd, XZR, Xm`: an address between pointer registers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct MovX {
    dst: PtrReg,
    src: PtrReg,
}

impl MovX {
    #[must_use]
    #[inline]
    pub(super) const fn new(dst: PtrReg, src: PtrReg) -> Self {
        Self { dst, src }
    }

    #[must_use]
    #[inline]
    fn encode(self) -> u32 {
        0xAA00_03E0 | ((self.src.0 as u32) << 16) | (self.dst.0 as u32)
    }
}

impl AsmInsn for MovX {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        code.extend_from_slice(&self.encode().to_le_bytes());
    }
}

/// FCVTZS Xd, Sn — truncate the scalar float in lane 0 to a signed 64-bit
/// integer in a GP register: how a fold's binder, or a broadcast load's
/// index, becomes an address. The scalar form of [`Fcvtzs`], whose `.4S`
/// result stays in the vector file.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct FcvtzsX {
    dst: Gpr,
    src: Reg,
}

impl FcvtzsX {
    #[must_use]
    #[inline]
    pub(super) const fn new(dst: Gpr, src: Reg) -> Self {
        Self { dst, src }
    }

    #[must_use]
    #[inline]
    pub(super) fn encode(self) -> u32 {
        0x9E38_0000 | ((self.src.0 as u32) << 5) | (self.dst.0 as u32)
    }
}

/// INS Vd.S[lane], Wn — insert a GP register into a 32-bit vector lane.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct InsW {
    dst: Reg,
    lane: u8,
    src: Gpr,
}

impl InsW {
    #[must_use]
    #[inline]
    pub(super) const fn new(dst: Reg, lane: u8, src: Gpr) -> Self {
        Self { dst, lane, src }
    }

    #[must_use]
    #[inline]
    pub(super) fn encode(self) -> u32 {
        // As `UmovW::encode`: lane 4 would encode a different instruction.
        assert!(
            self.lane < 4,
            "INS Vd.S[{}], Wn: a 128-bit register has four 32-bit lanes",
            self.lane
        );
        let imm5 = ((self.lane as u32) << 3) | 0b100;
        0x4E00_1C00 | (imm5 << 16) | ((self.src.0 as u32) << 5) | (self.dst.0 as u32)
    }
}

// =============================================================================
// Instruction Table (Type Aliases with Local Opcodes)
// =============================================================================

// Binary arithmetic (V × V → V)
pub(super) type Fadd = Binary<0x4E20_D400>;
pub(super) type Fsub = Binary<0x4EA0_D400>;
pub(super) type Fmul = Binary<0x6E20_DC00>;
pub(super) type Fdiv = Binary<0x6E20_FC00>;
pub(super) type Fmla = Binary<0x4E20_CC00>;
pub(super) type Fmin = Binary<0x4EA0_F400>;
pub(super) type Fmax = Binary<0x4E20_F400>;

// Unary arithmetic (V → V)
pub(super) type Fsqrt = Unary<0x6EA1_F800>;
pub(super) type Fabs = Unary<0x4EA0_F800>;
pub(super) type Fneg = Unary<0x6EA0_F800>;
pub(super) type Not = Unary<0x2E20_5800>;

// Rounding
pub(super) type Frintm = Unary<0x4E21_9800>; // floor
pub(super) type Frintp = Unary<0x4EA1_8800>; // ceil
pub(super) type Frinta = Unary<0x6E21_8800>; // round

// Reciprocal estimate / steps
pub(super) type Frsqrte = Unary<0x6EA1_D800>;
pub(super) type Frsqrts = Binary<0x4EA0_FC00>;
pub(super) type Frecpe = Unary<0x4EA1_D800>;
pub(super) type Frecps = Binary<0x4E20_FC00>;

// Vector comparisons (result is bit mask)
pub(super) type Fcmgt = Binary<0x6EA0_E400>;
pub(super) type Fcmge = Binary<0x6E20_E400>;
pub(super) type Fcmeq = Binary<0x4E20_E400>;

// Integer vector operations
pub(super) type AddI32 = Binary<0x4EA0_8400>;
pub(super) type And = Binary<0x4E20_1C00>;
pub(super) type Orr = Binary<0x4EA0_1C00>;

// Conversions
pub(super) type Fcvtzs = Unary<0x4EA1_B800>; // float -> signed int32
pub(super) type Scvtf = Unary<0x4E21_D800>; // signed int32 -> float

// Reductions (V → S)
pub(super) type Uminv = Reduce<0x6EB1_A800>;
pub(super) type Umaxv = Reduce<0x6E30_A800>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_i64_encodes_an_immediate() {
        let mut code_imm = Vec::new();
        AddI64::new(Gpr(0), Gpr(1), Imm12(42)).emit_into(&mut code_imm);
        assert_eq!(
            code_imm,
            (0x9100_0000u32 | (42 << 10) | (1 << 5)).to_le_bytes()
        );
    }
}
