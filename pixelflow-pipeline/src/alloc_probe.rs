//! A counting global allocator for the measurement bins: bytes allocated,
//! bytes freed, and the peak of their difference since the last reset.
//!
//! Three relaxed atomics per allocation on top of the system allocator —
//! cheap against `malloc` itself, and uniform across everything a bin
//! measures, so the compile times such a bin reports are comparable with
//! each other and a few percent above an uncounted build's. A bin opts in
//! by declaring it:
//!
//! ```ignore
//! #[global_allocator]
//! static GLOBAL: pixelflow_pipeline::alloc_probe::CountingAlloc = CountingAlloc;
//! ```
//!
//! [`reset`] zeroes the counters, so [`peak_bytes`] afterwards is the peak
//! **net growth** of live heap since the reset — what a saturate+extract
//! call held at its widest over and above everything already allocated
//! when it began — and not the process's resident size.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// The allocator. See the module docs.
pub struct CountingAlloc;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let now = ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            let live = now.saturating_sub(DEALLOCATED.load(Ordering::Relaxed));
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }
}

/// Zero all three counters.
pub fn reset() {
    ALLOCATED.store(0, Ordering::Relaxed);
    DEALLOCATED.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
}

/// Bytes allocated since the last [`reset`].
#[must_use]
pub fn allocated_bytes() -> usize {
    ALLOCATED.load(Ordering::Relaxed)
}

/// Peak net heap growth since the last [`reset`].
#[must_use]
pub fn peak_bytes() -> usize {
    PEAK.load(Ordering::Relaxed)
}
