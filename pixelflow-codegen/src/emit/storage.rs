//! Physical storage locations, stack slots, and frame allocation.
//!
//! A vector value on the physical machine resides in one of two physical
//! storage classes:
//! - In a hardware vector [`Reg`]
//! - In an aligned stack frame [`Slot`]
//!
//! This module models these entities as first-class domain objects, providing
//! capability traits for writing ([`StoreTarget`]) and reading ([`SourceOperand`])
//! across registers and memory, along with a recycling [`StackFrame`] slot allocator.

use super::Reg;
use alloc::vec::Vec;

/// An aligned slot in the stack frame.
///
/// A `Slot` represents a concrete stack address: it knows its byte displacement
/// relative to the stack/frame pointer and the vector width (in bytes).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Slot {
    offset: u32,
    bytes: u32,
}

impl Slot {
    /// Create a new stack slot with the given byte displacement and size.
    #[inline]
    #[must_use]
    pub const fn new(offset: u32, bytes: u32) -> Self {
        Self { offset, bytes }
    }

    /// Displacement in bytes from the stack/frame pointer (e.g. `[rsp + offset]`).
    #[inline]
    #[must_use]
    pub const fn offset(self) -> u32 {
        self.offset
    }

    /// Size of the vector slot in bytes (16 for SSE/NEON, 32 for AVX2, 64 for AVX-512).
    #[inline]
    #[must_use]
    pub const fn bytes(self) -> u32 {
        self.bytes
    }
}

/// A physical location where a vector value can reside: in a register or on the stack.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Storage {
    /// A hardware vector register.
    Reg(Reg),
    /// A stack frame slot.
    Slot(Slot),
}

impl From<Reg> for Storage {
    #[inline]
    fn from(r: Reg) -> Self {
        Storage::Reg(r)
    }
}

impl From<Slot> for Storage {
    #[inline]
    fn from(s: Slot) -> Self {
        Storage::Slot(s)
    }
}

/// Capability: a physical storage location that can be written to by an instruction or transfer.
pub trait StoreTarget: Copy + core::fmt::Debug {
    /// Convert to the canonical [`Storage`] enum.
    fn target_storage(self) -> Storage;

    /// Extract the register if this target is in-register.
    fn target_reg(self) -> Option<Reg>;

    /// Extract the stack slot if this target is on-stack.
    fn target_slot(self) -> Option<Slot>;
}

impl StoreTarget for Reg {
    #[inline]
    fn target_storage(self) -> Storage {
        Storage::Reg(self)
    }
    #[inline]
    fn target_reg(self) -> Option<Reg> {
        Some(self)
    }
    #[inline]
    fn target_slot(self) -> Option<Slot> {
        None
    }
}

impl StoreTarget for Slot {
    #[inline]
    fn target_storage(self) -> Storage {
        Storage::Slot(self)
    }
    #[inline]
    fn target_reg(self) -> Option<Reg> {
        None
    }
    #[inline]
    fn target_slot(self) -> Option<Slot> {
        Some(self)
    }
}

impl StoreTarget for Storage {
    #[inline]
    fn target_storage(self) -> Storage {
        self
    }
    #[inline]
    fn target_reg(self) -> Option<Reg> {
        match self {
            Storage::Reg(r) => Some(r),
            Storage::Slot(_) => None,
        }
    }
    #[inline]
    fn target_slot(self) -> Option<Slot> {
        match self {
            Storage::Reg(_) => None,
            Storage::Slot(s) => Some(s),
        }
    }
}

/// Capability: a location or constant that can be read from as an operand.
pub trait SourceOperand: Copy + core::fmt::Debug {
    /// Extract the physical storage location if this operand resides in memory or register.
    fn source_storage(self) -> Option<Storage>;

    /// Extract the register if this operand resides in a register.
    fn source_reg(self) -> Option<Reg>;

    /// Extract the stack slot if this operand resides on the stack.
    fn source_slot(self) -> Option<Slot>;

    /// Extract constant bit pattern if this operand is a rematerialized immediate.
    fn source_const(self) -> Option<u32>;
}

impl SourceOperand for Reg {
    #[inline]
    fn source_storage(self) -> Option<Storage> {
        Some(Storage::Reg(self))
    }
    #[inline]
    fn source_reg(self) -> Option<Reg> {
        Some(self)
    }
    #[inline]
    fn source_slot(self) -> Option<Slot> {
        None
    }
    #[inline]
    fn source_const(self) -> Option<u32> {
        None
    }
}

impl SourceOperand for Slot {
    #[inline]
    fn source_storage(self) -> Option<Storage> {
        Some(Storage::Slot(self))
    }
    #[inline]
    fn source_reg(self) -> Option<Reg> {
        None
    }
    #[inline]
    fn source_slot(self) -> Option<Slot> {
        Some(self)
    }
    #[inline]
    fn source_const(self) -> Option<u32> {
        None
    }
}

impl SourceOperand for Storage {
    #[inline]
    fn source_storage(self) -> Option<Storage> {
        Some(self)
    }
    #[inline]
    fn source_reg(self) -> Option<Reg> {
        match self {
            Storage::Reg(r) => Some(r),
            Storage::Slot(_) => None,
        }
    }
    #[inline]
    fn source_slot(self) -> Option<Slot> {
        match self {
            Storage::Reg(_) => None,
            Storage::Slot(s) => Some(s),
        }
    }
    #[inline]
    fn source_const(self) -> Option<u32> {
        None
    }
}

/// A recycling stack frame slot allocator.
///
/// Manages allocation and deallocation of vector stack slots at a fixed byte stride,
/// reusing slots that have been released to minimize stack frame size.
#[derive(Clone, Debug)]
pub struct StackFrame {
    vector_bytes: u32,
    allocated_bytes: u32,
    free_pool: Vec<Slot>,
}

impl StackFrame {
    /// Create a new stack frame manager for vector slots of `vector_bytes` stride.
    #[inline]
    #[must_use]
    pub const fn new(vector_bytes: u32) -> Self {
        Self {
            vector_bytes,
            allocated_bytes: 0,
            free_pool: Vec::new(),
        }
    }

    /// Allocate a slot in the frame, reusing a previously freed slot if available.
    pub fn alloc_slot(&mut self) -> Result<Slot, crate::error::CompileError> {
        const MAX_FRAME: u32 = 2 * 1024 * 1024;
        if let Some(reused) = self.free_pool.pop() {
            Ok(reused)
        } else {
            if self.allocated_bytes > MAX_FRAME - self.vector_bytes {
                return Err(crate::error::CompileError::BudgetExceeded(
                    "spill frame overflow: exceeds 2MB stack limit",
                ));
            }
            let offset = self.allocated_bytes;
            self.allocated_bytes += self.vector_bytes;
            Ok(Slot::new(offset, self.vector_bytes))
        }
    }

    /// Release a slot back to the free pool so it can be reused by a later value.
    pub fn free_slot(&mut self, slot: Slot) {
        debug_assert_eq!(slot.bytes(), self.vector_bytes);
        self.free_pool.push(slot);
    }

    /// Total stack frame size in bytes, aligned to 16 bytes per standard ABI.
    #[inline]
    #[must_use]
    pub const fn frame_size(&self) -> u32 {
        (self.allocated_bytes + 15) & !15
    }

    /// The vector stride in bytes.
    #[inline]
    #[must_use]
    pub const fn vector_bytes(&self) -> u32 {
        self.vector_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::{Binding, Loc};

    #[test]
    fn storage_capabilities_for_reg_and_slot() {
        let r = Reg(3);
        let s = Slot::new(32, 16);

        // StoreTarget
        assert_eq!(r.target_reg(), Some(Reg(3)));
        assert_eq!(r.target_slot(), None);
        assert_eq!(r.target_storage(), Storage::Reg(Reg(3)));

        assert_eq!(s.target_reg(), None);
        assert_eq!(s.target_slot(), Some(Slot::new(32, 16)));
        assert_eq!(s.target_storage(), Storage::Slot(Slot::new(32, 16)));

        // SourceOperand
        assert_eq!(r.source_const(), None);
        assert_eq!(s.source_const(), None);
    }

    #[test]
    fn storage_capabilities_for_loc() {
        let l_reg = Loc::Reg(Reg(4));
        let l_slot = Loc::Slot(Slot::new(64, 32));

        // StoreTarget is total for Loc: every Loc is a writable physical location.
        assert_eq!(l_reg.target_storage(), Storage::Reg(Reg(4)));
        assert_eq!(l_reg.target_reg(), Some(Reg(4)));
        assert_eq!(l_reg.target_slot(), None);

        assert_eq!(l_slot.target_storage(), Storage::Slot(Slot::new(64, 32)));
        assert_eq!(l_slot.target_reg(), None);
        assert_eq!(l_slot.target_slot(), Some(Slot::new(64, 32)));

        // SourceOperand
        assert_eq!(l_reg.source_reg(), Some(Reg(4)));
        assert_eq!(l_reg.source_slot(), None);
        assert_eq!(l_reg.source_const(), None);

        assert_eq!(l_slot.source_reg(), None);
        assert_eq!(l_slot.source_slot(), Some(Slot::new(64, 32)));
        assert_eq!(l_slot.source_const(), None);

        // Binding SourceOperand capabilities (including Remat)
        let b_reg = Binding::from(l_reg);
        let b_slot = Binding::from(l_slot);
        let b_remat = Binding::Remat(0x3F80_0000);

        assert_eq!(b_reg.source_reg(), Some(Reg(4)));
        assert_eq!(b_slot.source_slot(), Some(Slot::new(64, 32)));
        assert_eq!(b_remat.source_reg(), None);
        assert_eq!(b_remat.source_slot(), None);
        assert_eq!(b_remat.source_const(), Some(0x3F80_0000));
        assert_eq!(b_remat.source_storage(), None);
    }

    #[test]
    fn stack_frame_allocates_and_reuses_slots() {
        let mut frame = StackFrame::new(16);
        let s0 = frame.alloc_slot().unwrap();
        let s1 = frame.alloc_slot().unwrap();
        assert_eq!(s0.offset(), 0);
        assert_eq!(s1.offset(), 16);
        assert_eq!(frame.frame_size(), 32);

        frame.free_slot(s0);
        let s2 = frame.alloc_slot().unwrap();
        assert_eq!(s2.offset(), 0);
        assert_eq!(frame.frame_size(), 32);
    }
}
