//! The AArch64 operation vocabulary: the enums whose discriminants are the
//! opcodes, the memory operands and the sizes their immediates scale by.
//!
//! The instructions themselves are [`Inst`]'s arms, grouped by their algebraic
//! morphism:
//! - [`Alu`]: $V \times V \to V$
//! - [`Lanewise`]: $V \to V$, and the horizontal reductions $V \to S$
//!
//! Magic hex opcodes live strictly on the enums below, and on the arms of
//! `Inst::encode` whose words carry no operation to choose.

use super::Inst;
use crate::emit::{AsmInsn, Integer, Physical, Pointer, Stage};
use alloc::vec::Vec;

// =============================================================================
// Algebraic Instruction Shapes
// =============================================================================

/// 12-bit unsigned immediate for AArch64 arithmetic instructions.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct Imm12(pub(super) u16);

/// Two-source vector operations: `dst = a ⊗ b` across all lanes. The
/// discriminant is the opcode with its register fields clear.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(super) enum Alu {
    Fadd = 0x4E20_D400,
    Fsub = 0x4EA0_D400,
    Fmul = 0x6E20_DC00,
    Fdiv = 0x6E20_FC00,
    Fmin = 0x4EA0_F400,
    Fmax = 0x4E20_F400,
    Frsqrts = 0x4EA0_FC00,
    Frecps = 0x4E20_FC00,
    /// A comparison's result is a bit mask: all-ones where it holds.
    Fcmgt = 0x6EA0_E400,
    Fcmge = 0x6E20_E400,
    Fcmeq = 0x4E20_E400,
    AddI32 = 0x4EA0_8400,
    And = 0x4E20_1C00,
    Orr = 0x4EA0_1C00,
}

/// One-source vector operations: `dst = f(src)` across all lanes. The
/// discriminant is the opcode with its register fields clear. `Uminv` and
/// `Umaxv` are the horizontal reductions: the scalar lands in lane 0.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(super) enum Lanewise {
    Fsqrt = 0x6EA1_F800,
    Fabs = 0x4EA0_F800,
    Fneg = 0x6EA0_F800,
    Not = 0x2E20_5800,
    /// Floor.
    Frintm = 0x4E21_9800,
    /// Ceil.
    Frintp = 0x4EA1_8800,
    /// Round, ties away.
    Frinta = 0x6E21_8800,
    Frsqrte = 0x6EA1_D800,
    Frecpe = 0x4EA1_D800,
    /// Float to signed int32.
    Fcvtzs = 0x4EA1_B800,
    /// Signed int32 to float.
    Scvtf = 0x4E21_D800,
    Uminv = 0x6EB1_A800,
    Umaxv = 0x6E30_A800,
}

/// Bytes moved by a `q` (128-bit vector) access — also the scale of its offset.
pub(super) const Q_BYTES: u32 = 16;
/// Bytes moved by an `x` (64-bit general/pointer) access.
pub(super) const X_BYTES: u32 = 8;
/// Bytes moved by an `s` (32-bit scalar SIMD&FP) access.
pub(super) const S_BYTES: u32 = 4;
/// The largest value a 12-bit scaled immediate holds.
pub(super) const MAX_IMM12: u32 = 4095;
/// The largest 16-byte-aligned displacement `add`'s own 12-bit immediate holds.
pub(super) const MAX_ADD_IMM: u32 = 4080;

/// An address spelled `[base, #offset]` — the scaled-immediate addressing mode.
///
/// `offset` is in BYTES. aarch64 encodes it divided by the access size.
/// The base being a pointer-class operand guarantees an integer counter or
/// index cannot be mistakenly passed as an address.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct Mem<S: Stage> {
    /// The register holding the base address.
    pub(super) base: S::Read<Pointer>,
    /// Displacement in bytes; must be a multiple of the access size.
    pub(super) offset: u32,
}

/// An address spelled `[base, w<index>, uxtw #2]` — a 32-bit index register,
/// zero-extended to 64 bits and scaled by 4.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct MemIndexed<S: Stage> {
    /// The register holding the buffer base pointer.
    pub(super) base: S::Read<Pointer>,
    /// The element index, read as the 32-bit `w<index>`.
    pub(super) index: S::Read<Integer>,
}

impl Mem<Physical> {
    /// The offset in units of the access it addresses: what `imm12` holds.
    ///
    /// # Panics
    ///
    /// If the offset is not a multiple of the access size.
    pub(super) fn scaled(self, access: u32) -> u32 {
        assert!(
            self.offset.is_multiple_of(access),
            "{}-bit access offset {} is not {access}-byte aligned",
            access * 8,
            self.offset
        );
        self.offset / access
    }

    /// `self`, or `[x16]` after the adds that compute it when the offset is
    /// past the 12-bit scaled immediate of an `access`-byte transfer.
    pub(super) fn near(self, code: &mut Vec<u8>, access: u32) -> Self {
        if self.scaled(access) > MAX_IMM12 {
            return address_in_ip0(code, self);
        }
        self
    }
}

/// Rewrite `addr` as `[x16]`, computing `base + offset` into IP0 first.
///
/// The fallback for a displacement past the 12-bit scaled immediate — a spill
/// frame deeper than 64 KiB. `add`'s immediate is 12 bits too, so a large
/// displacement takes several of them.
fn address_in_ip0(code: &mut Vec<u8>, Mem { base, offset }: Mem<Physical>) -> Mem<Physical> {
    let ip0 = super::ptr::X16;
    let mut remaining = offset;
    let first = remaining.min(MAX_ADD_IMM);
    Inst::AddImm {
        dst: ip0,
        src: base,
        imm: Imm12(first as u16),
    }
    .emit_into(code);
    remaining -= first;
    while remaining > 0 {
        let chunk = remaining.min(MAX_ADD_IMM);
        Inst::AddImm {
            dst: ip0,
            src: ip0,
            imm: Imm12(chunk as u16),
        }
        .emit_into(code);
        remaining -= chunk;
    }
    Mem {
        base: ip0,
        offset: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::PtrReg;

    #[test]
    fn add_i64_encodes_an_immediate() {
        let mut code_imm = Vec::new();
        Inst::AddImm {
            dst: PtrReg(0),
            src: PtrReg(1),
            imm: Imm12(42),
        }
        .emit_into(&mut code_imm);
        assert_eq!(
            code_imm,
            (0x9100_0000u32 | (42 << 10) | (1 << 5)).to_le_bytes()
        );
    }
}
