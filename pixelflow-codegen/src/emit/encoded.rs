//! Encoded instruction byte representation for variable-length ISAs (x86-64).
//!
//! Architectural limit on x86-64: no instruction can exceed 15 bytes.
//! [`EncodedInst`] is a 16-byte `Copy` stack struct: 15 payload bytes and a 1-byte length.
//! It fits in a single 128-bit register and incurs zero heap allocations.

use alloc::vec::Vec;
use core::ops::Deref;

/// A stack-allocated encoded instruction (up to 15 bytes).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct EncodedInst {
    pub bytes: [u8; 15],
    pub len: u8,
}

impl EncodedInst {
    /// Create an empty encoded instruction.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self {
            bytes: [0u8; 15],
            len: 0,
        }
    }

    /// Push a single byte into the instruction buffer.
    #[inline]
    pub fn push(&mut self, byte: u8) {
        assert!(self.len < 15, "x86 instruction cannot exceed 15 bytes");
        self.bytes[self.len as usize] = byte;
        self.len += 1;
    }

    /// Extend the instruction buffer by a byte slice.
    #[inline]
    pub fn extend(&mut self, slice: &[u8]) {
        assert!(
            self.len as usize + slice.len() <= 15,
            "x86 instruction cannot exceed 15 bytes"
        );
        let start = self.len as usize;
        self.bytes[start..start + slice.len()].copy_from_slice(slice);
        self.len += slice.len() as u8;
    }

    /// Create an encoded instruction from a fixed-size byte array (`N <= 15`).
    #[must_use]
    #[inline]
    pub const fn from_array<const N: usize>(arr: [u8; N]) -> Self {
        assert!(N <= 15, "x86 instruction cannot exceed 15 bytes");
        let mut bytes = [0u8; 15];
        let mut i = 0;
        while i < N {
            bytes[i] = arr[i];
            i += 1;
        }
        Self {
            bytes,
            len: N as u8,
        }
    }

    /// Create from a slice (`slice.len() <= 15`).
    #[must_use]
    #[inline]
    pub fn from_slice(slice: &[u8]) -> Self {
        assert!(slice.len() <= 15, "x86 instruction cannot exceed 15 bytes");
        let mut bytes = [0u8; 15];
        bytes[..slice.len()].copy_from_slice(slice);
        Self {
            bytes,
            len: slice.len() as u8,
        }
    }

    /// Return the encoded instruction bytes as a slice.
    #[must_use]
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }

    /// Length in bytes.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.len as usize
    }

    /// Whether the instruction is empty.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Append this instruction's bytes directly to a code buffer.
    #[inline]
    pub fn emit(&self, code: &mut Vec<u8>) {
        code.extend_from_slice(self.as_bytes());
    }
}

impl Deref for EncodedInst {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_bytes()
    }
}

impl AsRef<[u8]> for EncodedInst {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl crate::emit::AsmInsn for EncodedInst {
    #[inline]
    fn emit_into(self, code: &mut Vec<u8>) {
        code.extend_from_slice(self.as_bytes());
    }
}

impl Default for EncodedInst {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
