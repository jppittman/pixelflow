//! Executable memory management for JIT.
//!
//! This module handles the mmap/mprotect dance to create executable code at runtime.

use crate::error::CompileError;

/// One kernel's emitted code, at one lattice shape, held as executable memory.
///
/// The memory is allocated as read-write, code is written to it, then it is
/// flipped to read-execute (W^X). The loop nest is inside the code — the
/// lattice's rows, batches and lanes are folds the kernel was wrapped in
/// before it was scheduled (docs/plans/2026-09-16-collapse-is-a-fold.md) — so
/// [`call`](Self::call) is the whole collapse, and the lattice shape is what
/// the code *is*: its loop bounds are the extent it was compiled at. No cache;
/// the caller decides its lifetime.
pub struct CompiledKernel {
    ptr: *mut u8,
    len: usize,
    capacity: usize,
}

// SAFETY: The code is immutable after compilation and can be shared across threads.
unsafe impl Send for CompiledKernel {}
unsafe impl Sync for CompiledKernel {}

impl CompiledKernel {
    /// Compile a code buffer into executable memory.
    ///
    /// # Safety
    /// The caller must ensure the code buffer contains valid machine code
    /// for the current architecture.
    #[cfg(unix)]
    pub(super) unsafe fn from_code(code: &[u8]) -> Result<Self, CompileError> {
        NativeCodePage::from_code(code)
    }

    /// The emitted machine code. The bytes are the artifact, not an ABI.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl CompiledKernel {
    /// Run the collapse this code is: every lattice point the kernel was
    /// compiled for, stored into `out`, whose rows are `pitch` elements apart.
    ///
    /// # Safety
    ///
    /// - `ctx` must hold one valid base pointer per buffer the kernel
    ///   declared, in slot order, then the uniform block's base (when the
    ///   kernel declares a uniform; anything otherwise) and then the origin
    ///   block's — two `f32`s, `x0` then `y0` — each live for the call.
    /// - `out` must be writable for `(height - 1) * pitch + width` elements of
    ///   the extent the kernel was compiled at.
    #[inline(always)]
    pub unsafe fn call(&self, ctx: *const *const f32, out: *mut f32, pitch: usize) {
        // SAFETY: the code is a `KernelFn` — every compile emits one — and the
        // caller upholds its contract, above.
        let func: KernelFn = unsafe { core::mem::transmute_copy(&self.ptr) };
        func(ctx, out, pitch)
    }
}

impl Drop for CompiledKernel {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            let rc = libc::munmap(self.ptr as *mut libc::c_void, self.capacity);
            // A failing `munmap` means the mapping this type believed it owned
            // was not the mapping the kernel had — a leak at best. Nothing
            // useful can be done about it while unwinding, but it must not
            // pass unremarked.
            debug_assert_eq!(
                rc, 0,
                "munmap failed for {:?} ({} bytes)",
                self.ptr, self.capacity
            );
        }
    }
}

// =============================================================================
// Page preparation & platform abstraction
// =============================================================================

/// The lifecycle of a writable JIT code page transitioning to executable memory.
trait CodePage: Sized {
    /// Get the system page size for this platform.
    fn page_size() -> usize;

    /// Map a writable page of at least `capacity` bytes.
    fn map(capacity: usize) -> Result<Self, CompileError>;

    /// Copy code into the page.
    fn write(&mut self, code: &[u8]);

    /// Seal the page to Read+Execute, synchronize instruction caches,
    /// and return the executable handle.
    fn finish(self, len: usize) -> Result<CompiledKernel, CompileError>;

    /// Compile a code buffer into executable memory.
    fn from_code(code: &[u8]) -> Result<CompiledKernel, CompileError> {
        let page_size = Self::page_size();
        let capacity = (code.len() + page_size - 1) & !(page_size - 1);

        let mut page = Self::map(capacity)?;
        page.write(code);
        page.finish(code.len())
    }
}

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
type NativeCodePage = macos::MacOsCodePage;

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
type NativeCodePage = linux::LinuxCodePage;

// =============================================================================
// The one kernel ABI
// =============================================================================

/// A JIT-compiled collapse: `fn(ctx, out, pitch)`.
///
/// One signature on every target, because the loop nest, the coordinates and
/// the lane width are all inside the emitted code — the lattice's folds wrap
/// the kernel before it is scheduled (docs/plans/2026-09-16-collapse-is-a-fold.md
/// §2.1), so nothing about the batch reaches the boundary. What does:
///
/// - `ctx`: one base pointer per declared buffer, in slot order; then the
///   uniform block's base (`f32`s in the link's order); then the origin
///   block's base — `[x0, y0]`, where the lattice's sample `(0, 0)` lies.
///   `rdi` under SysV, `x0` under AAPCS64; read-only for the whole call.
/// - `out`: the plane the samples land in. `rsi` / `x1`.
/// - `pitch`: elements between the starts of two rows of `out`. `rdx` / `x2`.
///
/// Every lane of every batch the code computes is stored: the width is the
/// extent's, and a row's final partial batch stores exactly its remainder.
type KernelFn = extern "C" fn(*const *const f32, *mut f32, usize);

// =============================================================================
// Tests
// =============================================================================

/// Page preparation, independent of ISA level and architecture.
///
/// Ungated on purpose: `CodePages`, `page_size` and `sync_instruction_cache`
/// are exercised by every build, whatever tier the host runs.
#[cfg(all(test, unix))]
mod page_tests {
    use super::*;

    /// A single `ret` for the host, so the buffer really is valid machine code
    /// and `from_code`'s safety contract holds —
    /// `host_ret_is_actually_a_valid_return_instruction` below is what proves
    /// that claim by executing it; every other test here only reads the
    /// bytes back.
    fn host_ret() -> Vec<u8> {
        #[cfg(target_arch = "x86_64")]
        {
            alloc::vec![0xC3]
        }
        #[cfg(target_arch = "aarch64")]
        {
            0xD65F_03C0u32.to_le_bytes().to_vec()
        }
    }

    /// Asked of the machine, not assumed from the OS.
    ///
    /// The lower bound and the power of two together are what aarch64's
    /// [`AdrpAdd`](crate::emit::aarch64::AdrpAdd) rests on: they make every
    /// mapping a multiple of 4 KiB, which is the only reason masking a
    /// *buffer offset* finds the same page that masking the runtime address
    /// would. A 2 KiB page here and every `ADRP` this crate emits is off by
    /// one.
    #[test]
    fn page_size_is_a_sane_power_of_two() {
        let n = NativeCodePage::page_size();
        assert!(n >= 4096, "page size {n} below the smallest we run on");
        assert!(n <= 1 << 20, "page size {n} implausibly large");
        assert!(n.is_power_of_two(), "page size {n} is not a power of two");
    }

    /// The whole `CodePages` path: map writable, write, flip to executable,
    /// sync the instruction cache. Reads the bytes back rather than running
    /// them, so it is meaningful on every host.
    #[test]
    fn code_survives_the_w_xor_x_flip() {
        let code = host_ret();
        // SAFETY: `code` is a single valid `ret` for this architecture.
        let exec = unsafe { CompiledKernel::from_code(&code) }.expect("map + flip");
        assert_eq!(
            exec.as_bytes(),
            code.as_slice(),
            "bytes changed across the flip"
        );
    }

    /// The mapping is rounded up to a whole page, so a one-byte kernel and a
    /// page-sized one both work and neither reports padding as code.
    #[test]
    fn length_reported_is_the_code_not_the_mapping() {
        let mut code = host_ret();
        let ret_len = code.len();
        code.resize(NativeCodePage::page_size() + ret_len, 0);
        code.rotate_right(ret_len); // keep the `ret` first
        // SAFETY: entry point is a valid `ret`; the padding is never executed.
        let exec = unsafe { CompiledKernel::from_code(&code) }.expect("map + flip");
        assert_eq!(
            exec.as_bytes().len(),
            code.len(),
            "len must be the code, not the page"
        );
    }

    /// Every other test in this module reads `host_ret`'s bytes back rather
    /// than running them, so nothing else here actually proves it is a valid
    /// instruction for the host rather than a placeholder that happens to
    /// round-trip. Executing it and returning control to the test process is
    /// the only way to prove that.
    #[test]
    fn host_ret_is_actually_a_valid_return_instruction() {
        type NoOp = unsafe extern "C" fn();
        let code = host_ret();
        // SAFETY: about to prove `code` is a valid `ret` by executing it — a
        // bare `ret` touches no memory and no register but the program
        // counter, so calling it with no arguments and discarding any return
        // value is sound as long as it really is one.
        let exec = unsafe { CompiledKernel::from_code(&code) }.expect("map + flip");
        let func: NoOp = unsafe { core::mem::transmute(exec.as_bytes().as_ptr()) };
        unsafe { func() };
    }

    /// The `- 1` in `(len + page_size - 1) & !(page_size - 1)` is what stops
    /// an exact multiple of the page size from spilling into an extra page;
    /// pin the arithmetic at that boundary specifically; a naive round-up
    /// without it would double the mapping for `exact` below. `capacity` is
    /// private to this crate but not to this module — [`page_tests`] is a
    /// descendant of `executable`.
    #[test]
    fn capacity_rounds_up_to_the_page_size_without_overshooting_an_exact_multiple() {
        let page = NativeCodePage::page_size();

        let exact = vec![0u8; page];
        let exec = NativeCodePage::from_code(&exact).expect("map");
        assert_eq!(
            exec.capacity, page,
            "an exact page multiple must not round up to a second page"
        );

        let over = vec![0u8; page + 1];
        let exec = NativeCodePage::from_code(&over).expect("map");
        assert_eq!(
            exec.capacity,
            2 * page,
            "one byte past a page boundary must round up to the next page"
        );
    }
}

#[cfg(test)]
mod abi_tests {
    use super::*;

    /// `mov [rsi], rdx ; ret` — the second and third arguments, SysV.
    #[cfg(target_arch = "x86_64")]
    mod hand_assembled {
        pub(super) const STORE_PITCH_THROUGH_OUT: &[u8] = &[0x48, 0x89, 0x16, 0xC3];
    }

    /// `str x2, [x1] ; ret` — the second and third arguments, AAPCS64.
    #[cfg(target_arch = "aarch64")]
    mod hand_assembled {
        pub(super) const STORE_PITCH_THROUGH_OUT: &[u8] =
            &[0x22, 0x00, 0x00, 0xF9, 0xC0, 0x03, 0x5F, 0xD6];
    }

    /// A kernel that stores its `pitch` argument through `out` and returns:
    /// the three-argument ABI, exercised without an emitter in the way.
    #[test]
    fn a_hand_assembled_kernel_reads_the_three_arguments() {
        let code = hand_assembled::STORE_PITCH_THROUGH_OUT;
        let mut out = [0u32; 2];
        // SAFETY: the code writes exactly eight bytes at `out` and returns.
        unsafe {
            let exec = CompiledKernel::from_code(code).expect("map + flip");
            exec.call(
                core::ptr::null(),
                out.as_mut_ptr().cast::<f32>(),
                0x1234_5678,
            );
        }
        assert_eq!(out, [0x1234_5678, 0]);
    }
}
