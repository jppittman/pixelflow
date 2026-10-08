//! macOS executable code page implementation.
//!
//! Handles `mmap`, `mprotect`, and Apple Silicon instruction cache invalidation.

use core::ptr;
use libc::{
    _SC_PAGESIZE, MAP_ANON, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE, mmap, mprotect, munmap,
    sysconf,
};

use super::{CodePage, ExecutableCode};
use crate::error::CompileError;

/// A mapped, writable code page on macOS.
pub struct MacOsCodePage {
    ptr: *mut u8,
    capacity: usize,
}

#[cfg(target_arch = "aarch64")]
fn sync_instruction_cache(ptr: *mut u8, len: usize) {
    unsafe extern "C" {
        fn sys_icache_invalidate(start: *mut core::ffi::c_void, size: usize);
    }
    unsafe { sys_icache_invalidate(ptr.cast::<core::ffi::c_void>(), len) };
}

#[cfg(not(target_arch = "aarch64"))]
fn sync_instruction_cache(_ptr: *mut u8, _len: usize) {}

impl CodePage for MacOsCodePage {
    fn page_size() -> usize {
        let n = unsafe { sysconf(_SC_PAGESIZE) };
        assert!(n > 0, "sysconf(_SC_PAGESIZE) failed");
        n as usize
    }

    fn map(capacity: usize) -> Result<Self, CompileError> {
        let ptr = unsafe {
            mmap(
                ptr::null_mut(),
                capacity,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(CompileError::Mmap);
        }
        Ok(Self {
            ptr: ptr.cast::<u8>(),
            capacity,
        })
    }

    fn write(&mut self, code: &[u8]) {
        // `assert!`, not `debug_assert!`: this trait's methods are safe, so a
        // caller reaches this `copy_nonoverlapping` from safe code. A
        // debug-only guard compiles out of release, where the overrun would
        // write past the mapping.
        assert!(
            code.len() <= self.capacity,
            "code buffer ({} bytes) exceeds the mapped page ({} bytes)",
            code.len(),
            self.capacity,
        );
        unsafe { ptr::copy_nonoverlapping(code.as_ptr(), self.ptr, code.len()) };
    }

    fn finish(self, len: usize) -> Result<ExecutableCode, CompileError> {
        // Before anything reads `len` bytes: `sync_instruction_cache` walks
        // that range, and the `ExecutableCode` this returns hands it to the
        // safe `as_bytes`, which builds a slice from it. An unchecked `len`
        // past `capacity` is therefore out-of-bounds through safe code.
        if len > self.capacity {
            return Err(CompileError::Internal(
                "finish: code length exceeds the mapped page",
            ));
        }

        let rc = unsafe {
            mprotect(
                self.ptr.cast::<libc::c_void>(),
                self.capacity,
                PROT_READ | PROT_EXEC,
            )
        };
        if rc != 0 {
            return Err(CompileError::Mprotect);
        }

        sync_instruction_cache(self.ptr, len);

        let me = core::mem::ManuallyDrop::new(self);
        Ok(ExecutableCode {
            ptr: me.ptr,
            len,
            capacity: me.capacity,
        })
    }
}

impl Drop for MacOsCodePage {
    fn drop(&mut self) {
        // As `ExecutableCode::drop`: the failure is a leak at best and nothing
        // can be done about it here, but it must not pass unremarked. This
        // drop is reached on `finish`'s error paths too.
        let rc = unsafe { munmap(self.ptr.cast::<libc::c_void>(), self.capacity) };
        debug_assert_eq!(
            rc, 0,
            "munmap failed for {:?} ({} bytes)",
            self.ptr, self.capacity
        );
    }
}

#[cfg(test)]
pub(crate) fn test_sync_empty() {
    let mut byte = 0u8;
    sync_instruction_cache(&raw mut byte, 0);
}
