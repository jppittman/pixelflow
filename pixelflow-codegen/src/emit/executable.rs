//! Executable memory management for JIT.
//!
//! This module handles the mmap/mprotect dance to create executable code at runtime.

use crate::error::CompileError;

/// A region of executable memory containing JIT-compiled code.
///
/// The memory is allocated as read-write, code is written to it,
/// then it's flipped to read-execute (W^X security).
pub struct ExecutableCode {
    ptr: *mut u8,
    len: usize,
    capacity: usize,
}

// SAFETY: The code is immutable after compilation and can be shared across threads.
unsafe impl Send for ExecutableCode {}
unsafe impl Sync for ExecutableCode {}

impl ExecutableCode {
    /// Compile a code buffer into executable memory.
    ///
    /// # Safety
    /// The caller must ensure the code buffer contains valid machine code
    /// for the current architecture.
    #[cfg(unix)]
    pub unsafe fn from_code(code: &[u8]) -> Result<Self, CompileError> {
        NativeCodePage::from_code(code)
    }

    /// Get a function pointer to the compiled code.
    ///
    /// # Safety
    /// The caller must ensure the code implements the correct calling convention
    /// and signature for type `F`.
    #[inline]
    #[must_use]
    pub unsafe fn as_fn<F>(&self) -> F {
        // SAFETY: Caller guarantees F matches the compiled code's signature.
        unsafe { core::mem::transmute_copy(&self.ptr) }
    }

    /// Get the code as a byte slice (for debugging).
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Length of the compiled code in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the code is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl ExecutableCode {
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
        let func: KernelFn = unsafe { self.as_fn() };
        func(ctx, out, pitch)
    }
}

impl Drop for ExecutableCode {
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
pub trait CodePage: Sized {
    /// Get the system page size for this platform.
    fn page_size() -> usize;

    /// Map a writable page of at least `capacity` bytes.
    fn map(capacity: usize) -> Result<Self, CompileError>;

    /// Copy code into the page.
    fn write(&mut self, code: &[u8]);

    /// Seal the page to Read+Execute, synchronize instruction caches,
    /// and return the executable handle.
    fn finish(self, len: usize) -> Result<ExecutableCode, CompileError>;

    /// Compile a code buffer into executable memory.
    fn from_code(code: &[u8]) -> Result<ExecutableCode, CompileError> {
        if code.is_empty() {
            return Err(CompileError::EmptyCodeBuffer);
        }

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
pub use macos::MacOsCodePage;
#[cfg(target_os = "macos")]
pub type NativeCodePage = macos::MacOsCodePage;

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub use linux::LinuxCodePage;
#[cfg(target_os = "linux")]
pub type NativeCodePage = linux::LinuxCodePage;

pub mod mock;
pub use mock::MockCodePage;

/// Get the system page size for the native target.
#[must_use]
#[inline]
pub fn page_size() -> usize {
    NativeCodePage::page_size()
}

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
pub type KernelFn = extern "C" fn(*const *const f32, *mut f32, usize);

// =============================================================================
// Tests
// =============================================================================

/// Page preparation, independent of ISA level and architecture.
///
/// Ungated on purpose: `CodePages`, `page_size` and `sync_instruction_cache`
/// are exercised by every build, not just the SSE2 one the module below is
/// limited to.
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

    /// Asked of the machine, not assumed from the OS. A bad `sysconf` fallback
    /// would show up here as a non-power-of-two or an absurd size.
    ///
    /// The lower bound and the power of two together are what aarch64's
    /// [`AdrpAdd`](crate::emit::aarch64::AdrpAdd) rests on: they make every
    /// mapping a multiple of 4 KiB, which is the only reason masking a
    /// *buffer offset* finds the same page that masking the runtime address
    /// would. A 2 KiB page here and every `ADRP` this crate emits is off by
    /// one.
    #[test]
    fn page_size_is_a_sane_power_of_two() {
        let n = page_size();
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
        let exec = unsafe { ExecutableCode::from_code(&code) }.expect("map + flip");
        assert_eq!(exec.len(), code.len());
        assert!(!exec.is_empty());
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
        code.resize(page_size() + ret_len, 0);
        code.rotate_right(ret_len); // keep the `ret` first
        // SAFETY: entry point is a valid `ret`; the padding is never executed.
        let exec = unsafe { ExecutableCode::from_code(&code) }.expect("map + flip");
        assert_eq!(exec.len(), code.len(), "len must be the code, not the page");
    }

    #[test]
    fn an_empty_buffer_is_refused() {
        // SAFETY: empty slice; rejected before anything is mapped.
        match unsafe { ExecutableCode::from_code(&[]) } {
            Err(e) => assert_eq!(e, CompileError::EmptyCodeBuffer),
            Ok(_) => panic!("an empty buffer must not map"),
        }
    }

    /// `sync_instruction_cache` must accept an empty range without touching
    /// memory — the aarch64 path computes a loop bound from it.
    #[test]
    fn syncing_an_empty_range_is_a_no_op() {
        #[cfg(target_os = "macos")]
        macos::test_sync_empty();
        #[cfg(target_os = "linux")]
        linux::test_sync_empty();
    }

    #[test]
    fn mock_code_page_exercises_lifecycle() {
        let code = host_ret();
        let exec = MockCodePage::from_code(&code).expect("mock map + flip");
        assert_eq!(exec.len(), code.len());
        assert_eq!(exec.as_bytes(), code.as_slice());
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
        let exec = unsafe { ExecutableCode::from_code(&code) }.expect("map + flip");
        let func: NoOp = unsafe { exec.as_fn() };
        unsafe { func() };
    }

    /// `from_code` refuses an empty buffer, so `is_empty` can never observe
    /// the zero-length case through it; go around it the way `from_code`
    /// itself is built — `map` then `finish` directly — to construct the
    /// case `is_empty` exists to report.
    #[test]
    fn is_empty_reports_a_zero_length_page() {
        let exec = MockCodePage::map(page_size())
            .expect("map")
            .finish(0)
            .expect("finish");
        assert!(exec.is_empty());
        assert_eq!(exec.len(), 0);
    }

    /// The `- 1` in `(len + page_size - 1) & !(page_size - 1)` is what stops
    /// an exact multiple of the page size from spilling into an extra page;
    /// pin the arithmetic at that boundary specifically; a naive round-up
    /// without it would double the mapping for `exact` below. `capacity` is
    /// private to this crate but not to this module — [`page_tests`] is a
    /// descendant of `executable`, same as `MockCodePage`'s own construction
    /// of it.
    #[test]
    fn capacity_rounds_up_to_the_page_size_without_overshooting_an_exact_multiple() {
        let page = page_size();

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
            let exec = ExecutableCode::from_code(code).expect("map + flip");
            exec.call(
                core::ptr::null(),
                out.as_mut_ptr().cast::<f32>(),
                0x1234_5678,
            );
        }
        assert_eq!(out, [0x1234_5678, 0]);
    }
}
