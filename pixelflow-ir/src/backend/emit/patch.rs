//! Inline code patching and W^X memory management.
//!
//! Provides utilities to safely mutate executable memory (e.g., for inline caching
//! or patching direct branches to JIT-compiled kernels).
//!
//! On Apple Silicon (macOS), this uses the fast thread-local `pthread_jit_write_protect_np`
//! to flip W^X permissions instantly without the heavy syscall overhead of `mprotect`.

use core::ffi::c_void;

#[cfg(target_os = "macos")]
extern "C" {
    fn pthread_jit_write_protect_np(enabled: core::ffi::c_int);
    fn sys_icache_invalidate(start: *mut c_void, len: usize);
}

/// Temporarily disable W^X protection for the current thread to allow patching
/// executable memory.
///
/// # Safety
/// This function bypasses security protections. The caller must ensure that
/// memory modifications are safe and that `end_patching` is called immediately
/// after modifications are complete.
#[inline(always)]
pub unsafe fn begin_patching() {
    #[cfg(target_os = "macos")]
    {
        // 0 = false (write enabled, execute disabled for the current thread)
        pthread_jit_write_protect_np(0);
    }
}

/// Re-enable W^X protection for the current thread and flush the instruction cache.
///
/// # Safety
/// Must be paired with a preceding `begin_patching` call.
#[inline(always)]
pub unsafe fn end_patching(start: *mut u8, len: usize) {
    #[cfg(target_os = "macos")]
    {
        // 1 = true (write disabled, execute enabled for the current thread)
        pthread_jit_write_protect_np(1);

        // Flush the instruction cache for the modified region.
        // Apple requires this to be called AFTER restoring execute permissions.
        sys_icache_invalidate(start as *mut c_void, len);
    }
}
