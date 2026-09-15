//! Denotational AArch64 instruction shapes and instruction table.
//!
//! Instructions are categorized by their algebraic morphism:
//! - [`Binary<const OPCODE: u32>`]: $V \times V \to V$
//! - [`Unary<const OPCODE: u32>`]: $V \to V$
//! - [`Reduce<const OPCODE: u32>`]: $V \to S$
//!
//! Magic hex opcodes live strictly on the typed instruction definitions in this table.

use crate::emit::{AsmInsn, Gpr, PtrReg, Reg, SourceOperand, StoreTarget};
use alloc::vec::Vec;

// =============================================================================
// Algebraic Instruction Shapes
// =============================================================================

/// 12-bit unsigned immediate for AArch64 arithmetic instructions.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Imm12(pub u16);

/// 64-bit integer addition: `ADD Xd, Xn, Xm` or `ADD Xd, Xn, #imm12`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AddI64<O = Gpr> {
    pub dst: Gpr,
    pub src: Gpr,
    pub operand: O,
}

impl<O> AddI64<O> {
    #[must_use]
    #[inline]
    pub const fn new_raw(dst: Gpr, src: Gpr, operand: O) -> Self {
        Self { dst, src, operand }
    }

    #[must_use]
    #[inline]
    pub fn new(dst: impl Into<Gpr>, src: impl Into<Gpr>, operand: O) -> Self {
        Self {
            dst: dst.into(),
            src: src.into(),
            operand,
        }
    }
}

impl AsmInsn for AddI64<Gpr> {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        let w = 0x8B00_0000
            | ((self.operand.0 as u32 & 0x1F) << 16)
            | ((self.src.0 as u32 & 0x1F) << 5)
            | (self.dst.0 as u32 & 0x1F);
        code.extend_from_slice(&w.to_le_bytes());
    }
}

impl AsmInsn for AddI64<Imm12> {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        let w = 0x9100_0000
            | ((self.operand.0 as u32 & 0xFFF) << 10)
            | ((self.src.0 as u32 & 0x1F) << 5)
            | (self.dst.0 as u32 & 0x1F);
        code.extend_from_slice(&w.to_le_bytes());
    }
}

/// 64-bit integer subtraction: `SUB Xd, Xn, Xm` or `SUB Xd, Xn, #imm12`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SubI64<O = Gpr> {
    pub dst: Gpr,
    pub src: Gpr,
    pub operand: O,
}

impl<O> SubI64<O> {
    #[must_use]
    #[inline]
    pub const fn new_raw(dst: Gpr, src: Gpr, operand: O) -> Self {
        Self { dst, src, operand }
    }

    #[must_use]
    #[inline]
    pub fn new(dst: impl Into<Gpr>, src: impl Into<Gpr>, operand: O) -> Self {
        Self {
            dst: dst.into(),
            src: src.into(),
            operand,
        }
    }
}

impl AsmInsn for SubI64<Gpr> {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        let w = 0xCB00_0000
            | ((self.operand.0 as u32 & 0x1F) << 16)
            | ((self.src.0 as u32 & 0x1F) << 5)
            | (self.dst.0 as u32 & 0x1F);
        code.extend_from_slice(&w.to_le_bytes());
    }
}

impl AsmInsn for SubI64<Imm12> {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        let w = 0xD100_0000
            | ((self.operand.0 as u32 & 0xFFF) << 10)
            | ((self.src.0 as u32 & 0x1F) << 5)
            | (self.dst.0 as u32 & 0x1F);
        code.extend_from_slice(&w.to_le_bytes());
    }
}

/// 64-bit integer compare: `CMP Xn, Xm` (encoded as `SUBS XZR, Xn, Xm`)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CmpI64 {
    pub lhs: Gpr,
    pub rhs: Gpr,
}

impl CmpI64 {
    #[must_use]
    #[inline]
    pub const fn new_raw(lhs: Gpr, rhs: Gpr) -> Self {
        Self { lhs, rhs }
    }

    #[must_use]
    #[inline]
    pub fn new(lhs: impl Into<Gpr>, rhs: impl Into<Gpr>) -> Self {
        Self {
            lhs: lhs.into(),
            rhs: rhs.into(),
        }
    }
}

impl AsmInsn for CmpI64 {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        let w = 0xEB00_0000
            | ((self.rhs.0 as u32 & 0x1F) << 16)
            | ((self.lhs.0 as u32 & 0x1F) << 5)
            | 31;
        code.extend_from_slice(&w.to_le_bytes());
    }
}

/// Move wide with zero: `MOVZ Xd, #imm16`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Movz {
    pub dst: Gpr,
    pub imm: u16,
}

impl Movz {
    #[must_use]
    #[inline]
    pub const fn new_raw(dst: Gpr, imm: u16) -> Self {
        Self { dst, imm }
    }

    #[must_use]
    #[inline]
    pub fn new(dst: impl Into<Gpr>, imm: u16) -> Self {
        Self {
            dst: dst.into(),
            imm,
        }
    }
}

impl AsmInsn for Movz {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        let w = 0xD280_0000 | ((self.imm as u32) << 5) | (self.dst.0 as u32 & 0x1F);
        code.extend_from_slice(&w.to_le_bytes());
    }
}

/// Bitwise NOT of 32-bit general-purpose register: `MVN Wd, Wm` (encoded as `ORN Wd, WZR, Wm`)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MvnW {
    pub dst: Gpr,
    pub src: Gpr,
}

impl MvnW {
    #[must_use]
    #[inline]
    pub const fn new_raw(dst: Gpr, src: Gpr) -> Self {
        Self { dst, src }
    }

    #[must_use]
    #[inline]
    pub fn new(dst: impl Into<Gpr>, src: impl Into<Gpr>) -> Self {
        Self {
            dst: dst.into(),
            src: src.into(),
        }
    }

    #[must_use]
    #[inline]
    pub fn encode(self) -> u32 {
        0x2A20_03E0 | ((self.src.0 as u32 & 0x1F) << 16) | (self.dst.0 as u32 & 0x1F)
    }
}

impl AsmInsn for MvnW {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        code.extend_from_slice(&self.encode().to_le_bytes());
    }
}

/// Binary vector operation: V × V → V
///
/// Denotes `dst = lhs ⊗ rhs` across all vector lanes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Binary<const OPCODE: u32, D = Reg, L = Reg, R = Reg> {
    pub dst: D,
    pub lhs: L,
    pub rhs: R,
}

impl<const OPCODE: u32> Binary<OPCODE, Reg, Reg, Reg> {
    #[must_use]
    #[inline]
    pub const fn new(dst: Reg, lhs: Reg, rhs: Reg) -> Self {
        Self { dst, lhs, rhs }
    }

    #[must_use]
    #[inline]
    pub fn encode(self) -> u32 {
        OPCODE
            | (self.dst.0 as u32 & 0x1F)
            | ((self.lhs.0 as u32 & 0x1F) << 5)
            | ((self.rhs.0 as u32 & 0x1F) << 16)
    }

    /// Attempt to construct a concrete register binary op from abstract storage operands.
    #[must_use]
    #[inline]
    pub fn from_operands<D: StoreTarget, L: SourceOperand, R: SourceOperand>(
        dst: D,
        lhs: L,
        rhs: R,
    ) -> Option<Self> {
        let d = dst.target_reg()?;
        let l = lhs.source_reg()?;
        let r = rhs.source_reg()?;
        Some(Self::new(d, l, r))
    }
}

impl<const OPCODE: u32> AsmInsn for Binary<OPCODE, Reg, Reg, Reg> {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        code.extend_from_slice(&self.encode().to_le_bytes());
    }
}

/// Unary vector operation: V → V
///
/// Denotes `dst = f(src)` across all vector lanes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Unary<const OPCODE: u32, D = Reg, S = Reg> {
    pub dst: D,
    pub src: S,
}

impl<const OPCODE: u32> Unary<OPCODE, Reg, Reg> {
    #[must_use]
    #[inline]
    pub const fn new(dst: Reg, src: Reg) -> Self {
        Self { dst, src }
    }

    #[must_use]
    #[inline]
    pub fn encode(self) -> u32 {
        OPCODE | (self.dst.0 as u32 & 0x1F) | ((self.src.0 as u32 & 0x1F) << 5)
    }

    /// Attempt to construct a concrete register unary op from abstract storage operands.
    #[must_use]
    #[inline]
    pub fn from_operands<D: StoreTarget, S: SourceOperand>(dst: D, src: S) -> Option<Self> {
        let d = dst.target_reg()?;
        let s = src.source_reg()?;
        Some(Self::new(d, s))
    }
}

impl<const OPCODE: u32> AsmInsn for Unary<OPCODE, Reg, Reg> {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        code.extend_from_slice(&self.encode().to_le_bytes());
    }
}

/// Vector-to-scalar horizontal reduction: V → S
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Reduce<const OPCODE: u32, D = Reg, S = Reg> {
    pub dst: D,
    pub src: S,
}

impl<const OPCODE: u32> Reduce<OPCODE, Reg, Reg> {
    #[must_use]
    #[inline]
    pub const fn new(dst: Reg, src: Reg) -> Self {
        Self { dst, src }
    }

    #[must_use]
    #[inline]
    pub fn encode(self) -> u32 {
        OPCODE | (self.dst.0 as u32 & 0x1F) | ((self.src.0 as u32 & 0x1F) << 5)
    }

    /// Attempt to construct a concrete register reduction op from abstract storage operands.
    #[must_use]
    #[inline]
    pub fn from_operands<D: StoreTarget, S: SourceOperand>(dst: D, src: S) -> Option<Self> {
        let d = dst.target_reg()?;
        let s = src.source_reg()?;
        Some(Self::new(d, s))
    }
}

impl<const OPCODE: u32> AsmInsn for Reduce<OPCODE, Reg, Reg> {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        code.extend_from_slice(&self.encode().to_le_bytes());
    }
}

/// Bitwise select: `mask = (mask & if_true) | (~mask & if_false)`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Bsl<M = Reg, T = Reg, F = Reg> {
    pub mask: M,
    pub if_true: T,
    pub if_false: F,
}

impl Bsl<Reg, Reg, Reg> {
    #[must_use]
    #[inline]
    pub const fn new(mask: Reg, if_true: Reg, if_false: Reg) -> Self {
        Self {
            mask,
            if_true,
            if_false,
        }
    }

    #[must_use]
    #[inline]
    pub fn encode(self) -> u32 {
        0x6E60_1C00
            | (self.mask.0 as u32 & 0x1F)
            | ((self.if_true.0 as u32 & 0x1F) << 5)
            | ((self.if_false.0 as u32 & 0x1F) << 16)
    }

    /// Attempt to construct a concrete register Bsl from abstract storage operands.
    #[must_use]
    #[inline]
    pub fn from_operands<M: SourceOperand, T: SourceOperand, F: SourceOperand>(
        mask: M,
        if_true: T,
        if_false: F,
    ) -> Option<Self> {
        let m = mask.source_reg()?;
        let t = if_true.source_reg()?;
        let f = if_false.source_reg()?;
        Some(Self::new(m, t, f))
    }
}

impl AsmInsn for Bsl<Reg, Reg, Reg> {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        code.extend_from_slice(&self.encode().to_le_bytes());
    }
}

/// Broadcast a lane of a vector register across all lanes: `DUP Vd.4S, Vn.s[0]`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DupLane0 {
    pub dst: Reg,
    pub src: Reg,
}

impl DupLane0 {
    #[must_use]
    #[inline]
    pub const fn new(dst: Reg, src: Reg) -> Self {
        Self { dst, src }
    }

    #[must_use]
    #[inline]
    pub fn encode(self) -> u32 {
        0x4E04_0400 | ((self.src.0 as u32) << 5) | (self.dst.0 as u32)
    }
}

impl AsmInsn for DupLane0 {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        code.extend_from_slice(&self.encode().to_le_bytes());
    }
}

/// Move vector lane 0 to general purpose register: `FMOV X16, D<src>`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FmovToGp {
    pub src: Reg,
}

impl FmovToGp {
    #[must_use]
    #[inline]
    pub const fn new(src: Reg) -> Self {
        Self { src }
    }

    #[must_use]
    #[inline]
    pub fn encode(self) -> u32 {
        0x1E26_0000 | ((self.src.0 as u32) << 5) | 16
    }
}

impl AsmInsn for FmovToGp {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        code.extend_from_slice(&self.encode().to_le_bytes());
    }
}

/// Return from subroutine: `RET`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Ret;

impl Ret {
    pub const OPCODE: u32 = 0xD65F_03C0;

    #[must_use]
    #[inline]
    pub const fn encode(self) -> u32 {
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
pub struct SReg(pub Reg);

impl From<Reg> for SReg {
    #[inline(always)]
    fn from(r: Reg) -> Self {
        SReg(r)
    }
}

impl From<SReg> for Reg {
    #[inline(always)]
    fn from(s: SReg) -> Self {
        s.0
    }
}

/// Bytes moved by a `q` (128-bit vector) access — also the scale of its offset.
pub const Q_BYTES: u32 = 16;
/// Bytes moved by an `x` (64-bit general/pointer) access.
pub const X_BYTES: u32 = 8;
/// Bytes moved by an `s` (32-bit scalar SIMD&FP) access.
pub const S_BYTES: u32 = 4;
/// The largest value a 12-bit scaled immediate holds.
pub const MAX_IMM12: u32 = 4095;
/// The largest 16-byte-aligned displacement `add`'s own 12-bit immediate holds.
pub const MAX_ADD_IMM: u32 = 4080;

/// An address spelled `[base, #offset]` — the scaled-immediate addressing mode.
///
/// `offset` is in BYTES. aarch64 encodes it divided by the access size.
/// The base being a [`PtrReg`] guarantees an integer counter or index cannot
/// be mistakenly passed as an address.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Mem {
    /// The register holding the base address.
    pub base: PtrReg,
    /// Displacement in bytes; must be a multiple of the access size.
    pub offset: u32,
}

impl Mem {
    #[must_use]
    #[inline]
    pub const fn new(base: PtrReg, offset: u32) -> Self {
        Self { base, offset }
    }
}

/// An address spelled `[base, w<index>, uxtw #2]` — a 32-bit index register,
/// zero-extended to 64 bits and scaled by 4.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MemIndexed {
    /// The register holding the buffer base pointer.
    pub base: PtrReg,
    /// The element index, read as the 32-bit `w<index>`.
    pub index: Gpr,
}

impl MemIndexed {
    #[must_use]
    #[inline]
    pub const fn new(base: PtrReg, index: Gpr) -> Self {
        Self { base, index }
    }
}

/// Rewrite `addr` as `[x16]`, computing `base + offset` into IP0 first.
///
/// The fallback for a displacement past the 12-bit scaled immediate — a spill
/// frame deeper than 64 KiB. `add`'s immediate is 12 bits too, so a large
/// displacement takes several of them.
pub fn address_in_ip0(code: &mut Vec<u8>, Mem { base, offset }: Mem) -> Mem {
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

/// Register operand for a store instruction.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum StrReg {
    /// 128-bit vector register (`Qt`).
    Q(Reg),
    /// 64-bit pointer/GP register (`Xt`).
    X(PtrReg),
}

impl From<Reg> for StrReg {
    #[inline(always)]
    fn from(r: Reg) -> Self {
        StrReg::Q(r)
    }
}

impl From<PtrReg> for StrReg {
    #[inline(always)]
    fn from(p: PtrReg) -> Self {
        StrReg::X(p)
    }
}

impl From<Gpr> for StrReg {
    #[inline(always)]
    fn from(g: Gpr) -> Self {
        StrReg::X(PtrReg(g.0))
    }
}

/// Store register: `STR src, [addr]`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Str {
    pub src: StrReg,
    pub addr: Mem,
}

impl Str {
    #[must_use]
    #[inline]
    pub const fn new_raw(src: StrReg, addr: Mem) -> Self {
        Self { src, addr }
    }

    #[must_use]
    #[inline]
    pub fn new(src: impl Into<StrReg>, addr: Mem) -> Self {
        Self {
            src: src.into(),
            addr,
        }
    }

    #[must_use]
    #[inline]
    pub const fn q(src: Reg, addr: Mem) -> Self {
        Self {
            src: StrReg::Q(src),
            addr,
        }
    }

    #[must_use]
    #[inline]
    pub const fn x(src: PtrReg, addr: Mem) -> Self {
        Self {
            src: StrReg::X(src),
            addr,
        }
    }
}

impl AsmInsn for Str {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        match self.src {
            StrReg::Q(src) => {
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
                    | (src.0 as u32);
                code.extend_from_slice(&w.to_le_bytes());
            }
            StrReg::X(src) => {
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
                    0xF900_0000 | (imm12 << 10) | ((self.addr.base.0 as u32) << 5) | (src.0 as u32);
                code.extend_from_slice(&w.to_le_bytes());
            }
        }
    }
}

/// Register destination for a load instruction.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LdrReg {
    /// 128-bit vector register (`Qt`).
    Q(Reg),
    /// 64-bit pointer/GP register (`Xt`).
    X(PtrReg),
    /// 32-bit scalar float in vector lane 0 (`St`).
    S(Reg),
    /// 32-bit general-purpose word (`Wt`).
    W(Gpr),
}

impl From<Reg> for LdrReg {
    #[inline(always)]
    fn from(r: Reg) -> Self {
        LdrReg::Q(r)
    }
}

impl From<PtrReg> for LdrReg {
    #[inline(always)]
    fn from(p: PtrReg) -> Self {
        LdrReg::X(p)
    }
}

impl From<SReg> for LdrReg {
    #[inline(always)]
    fn from(s: SReg) -> Self {
        LdrReg::S(s.0)
    }
}

impl From<Gpr> for LdrReg {
    #[inline(always)]
    fn from(g: Gpr) -> Self {
        LdrReg::W(g)
    }
}

/// Addressing mode for a load instruction.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Addr {
    Offset(Mem),
    Indexed(MemIndexed),
}

impl From<Mem> for Addr {
    #[inline(always)]
    fn from(m: Mem) -> Self {
        Addr::Offset(m)
    }
}

impl From<MemIndexed> for Addr {
    #[inline(always)]
    fn from(m: MemIndexed) -> Self {
        Addr::Indexed(m)
    }
}

/// Load register: `LDR dst, [addr]`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Ldr {
    pub dst: LdrReg,
    pub addr: Addr,
}

impl Ldr {
    #[must_use]
    #[inline]
    pub const fn new_raw(dst: LdrReg, addr: Addr) -> Self {
        Self { dst, addr }
    }

    #[must_use]
    #[inline]
    pub fn new(dst: impl Into<LdrReg>, addr: impl Into<Addr>) -> Self {
        Self {
            dst: dst.into(),
            addr: addr.into(),
        }
    }

    #[must_use]
    #[inline]
    pub const fn q(dst: Reg, addr: Mem) -> Self {
        Self {
            dst: LdrReg::Q(dst),
            addr: Addr::Offset(addr),
        }
    }

    #[must_use]
    #[inline]
    pub const fn x(dst: PtrReg, addr: Mem) -> Self {
        Self {
            dst: LdrReg::X(dst),
            addr: Addr::Offset(addr),
        }
    }

    #[must_use]
    #[inline]
    pub const fn s(dst: Reg, addr: Mem) -> Self {
        Self {
            dst: LdrReg::S(dst),
            addr: Addr::Offset(addr),
        }
    }

    #[must_use]
    #[inline]
    pub const fn w(dst: Gpr, addr: MemIndexed) -> Self {
        Self {
            dst: LdrReg::W(dst),
            addr: Addr::Indexed(addr),
        }
    }
}

impl AsmInsn for Ldr {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        match (self.dst, self.addr) {
            (LdrReg::Q(dst), Addr::Offset(addr)) => {
                assert!(
                    addr.offset.is_multiple_of(Q_BYTES),
                    "128-bit access offset {} is not 16-byte aligned",
                    addr.offset
                );
                let a = if addr.offset / Q_BYTES > MAX_IMM12 {
                    address_in_ip0(code, addr)
                } else {
                    addr
                };
                let w = 0x3DC0_0000
                    | ((a.offset / Q_BYTES) << 10)
                    | ((a.base.0 as u32) << 5)
                    | (dst.0 as u32);
                code.extend_from_slice(&w.to_le_bytes());
            }
            (LdrReg::X(dst), Addr::Offset(addr)) => {
                assert!(
                    addr.offset.is_multiple_of(X_BYTES),
                    "pointer load offset {} not 8-byte aligned",
                    addr.offset
                );
                let imm12 = addr.offset / X_BYTES;
                assert!(
                    imm12 <= MAX_IMM12,
                    "pointer load offset {} exceeds LDR imm12 range",
                    addr.offset
                );
                let w = 0xF940_0000 | (imm12 << 10) | ((addr.base.0 as u32) << 5) | (dst.0 as u32);
                code.extend_from_slice(&w.to_le_bytes());
            }
            (LdrReg::S(dst), Addr::Offset(addr)) => {
                assert!(
                    addr.offset.is_multiple_of(S_BYTES),
                    "32-bit access offset {} is not 4-byte aligned",
                    addr.offset
                );
                let a = if addr.offset / S_BYTES > MAX_IMM12 {
                    address_in_ip0(code, addr)
                } else {
                    addr
                };
                let w = 0xBD40_0000
                    | ((a.offset / S_BYTES) << 10)
                    | ((a.base.0 as u32) << 5)
                    | (dst.0 as u32);
                code.extend_from_slice(&w.to_le_bytes());
            }
            (LdrReg::W(dst), Addr::Indexed(addr)) => {
                let w = 0xB860_5800
                    | ((addr.index.0 as u32) << 16)
                    | ((addr.base.0 as u32) << 5)
                    | (dst.0 as u32);
                code.extend_from_slice(&w.to_le_bytes());
            }
            _ => panic!("unsupported Ldr combination: {:?}", (self.dst, self.addr)),
        }
    }
}

/// UMOV Wd, Vn.S[lane] — extract a 32-bit vector lane into a GP register.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct UmovW {
    pub dst: Gpr,
    pub src: Reg,
    pub lane: u8,
}

impl UmovW {
    #[must_use]
    #[inline]
    pub const fn new(dst: Gpr, src: Reg, lane: u8) -> Self {
        Self { dst, src, lane }
    }

    #[must_use]
    #[inline]
    pub fn encode(self) -> u32 {
        debug_assert!(self.lane < 4);
        let imm5 = ((self.lane as u32) << 3) | 0b100;
        0x0E00_3C00 | (imm5 << 16) | ((self.src.0 as u32) << 5) | (self.dst.0 as u32)
    }
}

impl AsmInsn for UmovW {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        code.extend_from_slice(&self.encode().to_le_bytes());
    }
}

/// INS Vd.S[lane], Wn — insert a GP register into a 32-bit vector lane.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct InsW {
    pub dst: Reg,
    pub lane: u8,
    pub src: Gpr,
}

impl InsW {
    #[must_use]
    #[inline]
    pub const fn new(dst: Reg, lane: u8, src: Gpr) -> Self {
        Self { dst, lane, src }
    }

    #[must_use]
    #[inline]
    pub fn encode(self) -> u32 {
        debug_assert!(self.lane < 4);
        let imm5 = ((self.lane as u32) << 3) | 0b100;
        0x4E00_1C00 | (imm5 << 16) | ((self.src.0 as u32) << 5) | (self.dst.0 as u32)
    }
}

impl AsmInsn for InsW {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        code.extend_from_slice(&self.encode().to_le_bytes());
    }
}

// =============================================================================
// Instruction Table (Type Aliases with Local Opcodes)
// =============================================================================

// Binary arithmetic (V × V → V)
pub type Fadd = Binary<0x4E20_D400>;
pub type Fsub = Binary<0x4EA0_D400>;
pub type Fmul = Binary<0x6E20_DC00>;
pub type Fdiv = Binary<0x6E20_FC00>;
pub type Fmla = Binary<0x4E20_CC00>;
pub type Fmin = Binary<0x4EA0_F400>;
pub type Fmax = Binary<0x4E20_F400>;

// Unary arithmetic (V → V)
pub type Fsqrt = Unary<0x6EA1_F800>;
pub type Fabs = Unary<0x4EA0_F800>;
pub type Fneg = Unary<0x6EA0_F800>;
pub type Not = Unary<0x2E20_5800>;

// Rounding
pub type Frintm = Unary<0x4E21_9800>; // floor
pub type Frintp = Unary<0x4EA1_8800>; // ceil
pub type Frinta = Unary<0x6E21_8800>; // round

// Reciprocal estimate / steps
pub type Frsqrte = Unary<0x6EA1_D800>;
pub type Frsqrts = Binary<0x4EA0_FC00>;
pub type Frecpe = Unary<0x4EA1_D800>;
pub type Frecps = Binary<0x4E20_FC00>;

// Vector comparisons (result is bit mask)
pub type Fcmgt = Binary<0x6EA0_E400>;
pub type Fcmge = Binary<0x6E20_E400>;
pub type Fcmeq = Binary<0x4E20_E400>;

// Integer vector operations
pub type AddI32 = Binary<0x4EA0_8400>;
pub type And = Binary<0x4E20_1C00>;
pub type Orr = Binary<0x4EA0_1C00>;

// Conversions
pub type Fcvtzs = Unary<0x4EA1_B800>; // float -> signed int32
pub type Scvtf = Unary<0x4E21_D800>; // signed int32 -> float

// Reductions (V → S)
pub type Uminv = Reduce<0x6EB1_A800>;
pub type Umaxv = Reduce<0x6E30_A800>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::Loc;

    #[test]
    fn binary_and_unary_from_abstract_operands() {
        let dst = Reg(0);
        let lhs = Loc::Reg(Reg(1));
        let rhs = Loc::Reg(Reg(2));

        let add = Fadd::from_operands(dst, lhs, rhs).expect("all in registers");
        assert_eq!(add.encode(), Fadd::new(Reg(0), Reg(1), Reg(2)).encode());

        let sqrt = Fsqrt::from_operands(dst, lhs).expect("all in registers");
        assert_eq!(sqrt.encode(), Fsqrt::new(Reg(0), Reg(1)).encode());
    }

    #[test]
    fn add_i64_encodes_register_and_immediate() {
        let mut code_reg = Vec::new();
        AddI64::new(Gpr(0), Gpr(1), Gpr(2)).emit_into(&mut code_reg);
        assert_eq!(
            code_reg,
            (0x8B00_0000u32 | (2 << 16) | (1 << 5)).to_le_bytes()
        );

        let mut code_imm = Vec::new();
        AddI64::new(Gpr(0), Gpr(1), Imm12(42)).emit_into(&mut code_imm);
        assert_eq!(
            code_imm,
            (0x9100_0000u32 | (42 << 10) | (1 << 5)).to_le_bytes()
        );
    }
}
