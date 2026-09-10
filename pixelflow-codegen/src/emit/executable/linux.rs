//! Linux executable code page implementation.
//!
//! Handles `mmap`, `mprotect`, and Linux ARM64 architectural instruction cache maintenance.

use core::ptr;
use libc::{
    _SC_PAGESIZE, MAP_ANON, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE, mmap, mprotect, munmap,
    sysconf,
};

use super::{CodePage, ExecutableCode};
use crate::error::CompileError;

/// A mapped, writable code page on Linux.
pub struct LinuxCodePage {
    ptr: *mut u8,
    capacity: usize,
}

#[cfg(target_arch = "aarch64")]
fn sync_instruction_cache(ptr: *mut u8, len: usize) {
    use core::arch::asm;

    if len == 0 {
        return;
    }
    let start = ptr as usize;
    let end = start + len;

    let ctr: u64;
    unsafe {
        asm!("mrs {ctr}, ctr_el0", ctr = out(reg) ctr, options(nomem, nostack, preserves_flags))
    };
    let dline = 4usize << ((ctr >> 16) & 0xF);
    let iline = 4usize << (ctr & 0xF);

    let mut addr = start & !(dline - 1);
    while addr < end {
        unsafe { asm!("dc cvau, {addr}", addr = in(reg) addr, options(nostack, preserves_flags)) };
        addr += dline;
    }
    unsafe { asm!("dsb ish", options(nostack, preserves_flags)) };

    let mut addr = start & !(iline - 1);
    while addr < end {
        unsafe { asm!("ic ivau, {addr}", addr = in(reg) addr, options(nostack, preserves_flags)) };
        addr += iline;
    }
    unsafe { asm!("dsb ish", "isb", options(nostack, preserves_flags)) };
}

#[cfg(not(target_arch = "aarch64"))]
fn sync_instruction_cache(_ptr: *mut u8, _len: usize) {}

impl CodePage for LinuxCodePage {
    fn page_size() -> usize {
        let n = unsafe { sysconf(_SC_PAGESIZE) };
        if n > 0 { n as usize } else { 4096 }
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
        // `assert!`, not `debug_assert!`: this trait is publicly exported and
        // its methods are safe, so a downstream caller reaches this
        // `copy_nonoverlapping` from safe code. A debug-only guard compiles
        // out of release, where the overrun would write past the mapping.
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

impl Drop for LinuxCodePage {
    fn drop(&mut self) {
        unsafe { munmap(self.ptr.cast::<libc::c_void>(), self.capacity) };
    }
}

#[cfg(test)]
pub(crate) fn test_sync_empty() {
    let mut byte = 0u8;
    sync_instruction_cache(&raw mut byte, 0);
}

#[cfg(test)]
mod tests {
    use super::LinuxCodePage;
    use crate::emit::executable::CodePage;

    /// `CodePage` is publicly exported and every one of its methods is safe, so
    /// this sequence is reachable from a downstream crate without `unsafe`.
    /// Before the guard in `finish`, it returned an `ExecutableCode` whose
    /// `len` exceeded the mapping — and `ExecutableCode::as_bytes` is a safe
    /// `from_raw_parts` over exactly that `len`.
    #[test]
    fn sealing_past_the_mapping_is_refused_rather_than_returning_an_oob_slice() {
        // `map` records the requested capacity verbatim — mmap rounds up to a
        // page internally, but `self.capacity` is what was asked for, and that
        // is the bound `as_bytes` would be trusted with.
        const REQUESTED: usize = 64;

        assert!(
            LinuxCodePage::map(REQUESTED)
                .expect("map")
                .finish(REQUESTED)
                .is_ok(),
            "sealing exactly the requested capacity must still be allowed",
        );
        assert!(
            LinuxCodePage::map(REQUESTED)
                .expect("map")
                .finish(REQUESTED + 1)
                .is_err(),
            "one byte past the mapping must be refused, not handed to as_bytes",
        );
    }

    /// The release-mode half of the same hole: `write` guarded
    /// `copy_nonoverlapping` with a `debug_assert!`, which is absent from the
    /// build that ships.
    #[test]
    #[should_panic(expected = "exceeds the mapped page")]
    fn writing_past_the_mapping_panics_rather_than_overrunning_it() {
        let mut page = LinuxCodePage::map(64).expect("map");
        let oversized = vec![0x90u8; 65];
        page.write(&oversized);
    }

    /// `ExecutableCode`'s `Drop` (not this file's `LinuxCodePage::drop` —
    /// `finish` wraps `self` in `ManuallyDrop` before that impl ever runs) is
    /// what owns unmapping the flipped page, and nothing in this crate reads
    /// it back to notice if that stopped happening: a `Drop` reduced to a
    /// no-op would leak every kernel this process ever compiled, silently.
    ///
    /// Ask the kernel for the exact address back with `MAP_FIXED_NOREPLACE`
    /// rather than scanning `/proc/self/maps`: reading that file is itself an
    /// allocation, large enough on a real process to plausibly reuse the
    /// single freed page by coincidence before it's inspected, which is
    /// exactly the false "still mapped" a scan can't tell apart from a real
    /// one. `MAP_FIXED_NOREPLACE` asks the kernel to place a mapping at this
    /// precise address and refuses instead of overlapping if anything is
    /// already there — the two outcomes `Drop` running or not would produce.
    #[test]
    fn dropping_the_executable_code_unmaps_the_page() {
        // Content is irrelevant: this page is only ever mapped, never executed.
        let code = [0x90u8];
        let exec = LinuxCodePage::from_code(&code).expect("map + flip");
        let addr = exec.as_bytes().as_ptr();
        let len = <LinuxCodePage as CodePage>::page_size();
        drop(exec);

        let remap = unsafe {
            libc::mmap(
                addr.cast::<libc::c_void>().cast_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED_NOREPLACE,
                -1,
                0,
            )
        };
        assert_ne!(
            remap,
            libc::MAP_FAILED,
            "{addr:p} is still mapped after drop"
        );
        assert_eq!(
            remap.cast::<u8>(),
            addr.cast_mut(),
            "kernel placed the remap elsewhere despite MAP_FIXED_NOREPLACE"
        );
        unsafe { libc::munmap(remap, len) };
    }
}
