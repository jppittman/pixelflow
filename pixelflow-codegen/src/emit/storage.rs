//! Physical storage locations, stack slots, and frame allocation.
//!
//! A vector value on the physical machine resides in one of two physical
//! storage classes:
//! - In a hardware vector [`Reg`](super::Reg)
//! - In an aligned stack frame [`Slot`]
//!
//! This module models the slot, along with the [`StackFrame`] slot allocator.

/// An aligned slot in the stack frame.
///
/// A `Slot` represents a concrete stack address: it knows its byte displacement
/// relative to the stack/frame pointer.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Slot {
    offset: u32,
}

impl Slot {
    /// Create a new stack slot with the given byte displacement.
    #[inline]
    #[must_use]
    pub(super) const fn new(offset: u32) -> Self {
        Self { offset }
    }

    /// Displacement in bytes from the stack/frame pointer (e.g. `[rsp + offset]`).
    #[inline]
    #[must_use]
    pub(super) const fn offset(self) -> u32 {
        self.offset
    }
}

/// The largest frame a kernel may lay out. [`StackFrame::alloc_slot`] refuses
/// a spill past it and the nest's layout refuses the whole frame (spills, fold
/// roots and parks) past it, so every slot offset is below it by construction.
pub(super) const MAX_FRAME: u32 = 2 * 1024 * 1024;

/// A stack frame slot allocator.
///
/// Manages allocation of vector stack slots at a fixed byte stride.
#[derive(Clone, Debug)]
pub(super) struct StackFrame {
    vector_bytes: u32,
    allocated_bytes: u32,
}

impl StackFrame {
    /// A frame for vector slots of `vector_bytes` stride, handing out slots
    /// from `base` upward.
    ///
    /// A scope that runs to completion before the next one needs its slots can
    /// reuse the same offsets — that is why the two collapse prologues and the
    /// body all start at 0. A scope that runs *nested inside* another cannot:
    /// its parent's values are still live in their slots across it, so a fold's
    /// frame is based at its parent's [`Self::frame_size`] and the two never
    /// alias. [`frame_size`](Self::frame_size) stays the total extent — base
    /// included — so a parent's top is exactly its child's base.
    #[inline]
    #[must_use]
    pub(super) const fn with_base(vector_bytes: u32, base: u32) -> Self {
        Self {
            vector_bytes,
            allocated_bytes: base,
        }
    }

    /// Allocate a slot in the frame.
    pub(super) fn alloc_slot(&mut self) -> Result<Slot, crate::error::CompileError> {
        if self.allocated_bytes > MAX_FRAME - self.vector_bytes {
            return Err(crate::error::CompileError::BudgetExceeded(
                "spill frame overflow: exceeds 2MB stack limit",
            ));
        }
        let offset = self.allocated_bytes;
        self.allocated_bytes += self.vector_bytes;
        Ok(Slot::new(offset))
    }

    /// Total stack frame size in bytes, aligned to 16 bytes per standard ABI.
    #[inline]
    #[must_use]
    pub(super) const fn frame_size(&self) -> u32 {
        (self.allocated_bytes + 15) & !15
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stack_frame_allocates_slots_at_the_stride() {
        let mut frame = StackFrame::with_base(16, 0);
        let s0 = frame.alloc_slot().unwrap();
        let s1 = frame.alloc_slot().unwrap();
        assert_eq!(s0.offset(), 0);
        assert_eq!(s1.offset(), 16);
        assert_eq!(frame.frame_size(), 32);
    }
}
