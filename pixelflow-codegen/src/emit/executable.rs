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
    /// Execute the collapse kernel over a 2D tile destination.
    ///
    /// # Safety
    ///
    /// - `tile.out` must have space for `tile.rows * (tile.groups * LANES)` floats plus row skip padding.
    /// - `size_of::<V>() == JIT_VECTOR_BYTES`.
    #[inline(always)]
    pub unsafe fn call_collapse<V: Copy>(
        &self,
        ctx: *const *const f32,
        tile: TileSlice,
        origin: Point4<V>,
    ) {
        debug_assert_eq!(core::mem::size_of::<V>(), crate::JIT_VECTOR_BYTES);
        let func: KernelFn = unsafe { self.as_fn() };
        unsafe {
            func(
                ctx,
                tile.out,
                tile.groups,
                tile.rows,
                tile.row_skip_bytes,
                core::mem::transmute_copy(&origin.x),
                core::mem::transmute_copy(&origin.y),
                core::mem::transmute_copy(&origin.z),
                core::mem::transmute_copy(&origin.w),
            )
        }
    }
}

/// The base coordinate a collapse call starts from.
///
/// Four wide because the ABI is: a lattice has two axes, so `z` and `w` are
/// dead weight passed as zero and read by nothing — an arena that names
/// `Var(2)` or `Var(3)` is refused by
/// [`compile`](crate::emit::compile) before it can become a kernel.
/// Narrowing this narrows the emitted scaffold too, which moves every
/// kernel's bytes, so it is L2's step
/// (docs/plans/2026-09-06-lattice-is-the-index.md).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Point4<T> {
    /// X coordinate (1st lattice axis).
    pub x: T,
    /// Y coordinate (2nd lattice axis).
    pub y: T,
    /// Retired: the Z axis. Passed, never read.
    pub z: T,
    /// Retired: the W axis. Passed, never read.
    pub w: T,
}

impl<T> Point4<T> {
    /// Create a new 4D point with coordinates `(x, y, z, w)`.
    #[must_use]
    #[inline(always)]
    pub const fn new(x: T, y: T, z: T, w: T) -> Self {
        Self { x, y, z, w }
    }
}

impl<T: Copy> Point4<T> {
    /// Create a 4D point by splatting a single value to all coordinates.
    #[must_use]
    #[inline(always)]
    pub const fn splat(val: T) -> Self {
        Self {
            x: val,
            y: val,
            z: val,
            w: val,
        }
    }
}

impl<T> From<(T, T, T, T)> for Point4<T> {
    #[inline(always)]
    fn from((x, y, z, w): (T, T, T, T)) -> Self {
        Self { x, y, z, w }
    }
}

impl<T> From<[T; 4]> for Point4<T> {
    #[inline(always)]
    fn from([x, y, z, w]: [T; 4]) -> Self {
        Self { x, y, z, w }
    }
}

/// 2D rectangular grid dimensions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Extent2D {
    /// Width in elements.
    pub width: usize,
    /// Height in elements.
    pub height: usize,
}

impl Extent2D {
    /// Create new 2D grid dimensions.
    #[must_use]
    #[inline(always)]
    pub const fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    /// Total number of elements (`width * height`).
    ///
    /// # Panics
    ///
    /// If the product overflows `usize`. Callers size buffers by this figure,
    /// so a wrapped one is a buffer-too-small check that passes on a buffer
    /// that is too small — the one failure mode the check exists to stop.
    #[must_use]
    #[inline(always)]
    pub const fn len(self) -> usize {
        match self.width.checked_mul(self.height) {
            Some(n) => n,
            None => panic!("Extent2D::len overflowed usize"),
        }
    }

    /// True if either dimension is 0.
    #[must_use]
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }
}

impl From<(usize, usize)> for Extent2D {
    #[inline(always)]
    fn from((width, height): (usize, usize)) -> Self {
        Self { width, height }
    }
}

impl From<[usize; 2]> for Extent2D {
    #[inline(always)]
    fn from([width, height]: [usize; 2]) -> Self {
        Self { width, height }
    }
}

/// Destination layout for a 2D collapse evaluation pass.
#[derive(Copy, Clone, Debug)]
pub struct TileSlice {
    /// Pointer to the beginning of the writable output buffer.
    pub out: *mut f32,
    /// Number of full SIMD groups per row.
    pub groups: usize,
    /// Number of rows in the tile.
    pub rows: usize,
    /// Padding bytes to skip after each row before starting the next row.
    pub row_skip_bytes: usize,
}

impl TileSlice {
    /// Create a new tile destination layout.
    #[must_use]
    #[inline]
    pub const fn new(out: *mut f32, groups: usize, rows: usize, row_skip_bytes: usize) -> Self {
        Self {
            out,
            groups,
            rows,
            row_skip_bytes,
        }
    }

    /// Create a contiguous tile destination layout (zero padding bytes between rows).
    #[must_use]
    #[inline]
    pub const fn contiguous(out: *mut f32, groups: usize, rows: usize) -> Self {
        Self {
            out,
            groups,
            rows,
            row_skip_bytes: 0,
        }
    }

    /// Create a single-batch destination layout (1 group, 1 row, 0 padding).
    #[must_use]
    #[inline]
    pub const fn single(out: *mut f32) -> Self {
        Self {
            out,
            groups: 1,
            rows: 1,
            row_skip_bytes: 0,
        }
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
// Kernel type aliases for JIT-compiled functions
// =============================================================================

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::float32x4_t;

// __m128 is only the kernel width on the SSE2 build: the deleted scanline
// ABI was the last 128-bit-at-every-width user.
#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    not(target_feature = "avx2")
))]
use core::arch::x86_64::__m128;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
use core::arch::x86_64::__m256;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use core::arch::x86_64::__m512;

/// JIT-compiled *collapse* kernel (ARM64). See the AVX-512
/// [`KernelFn`] doc for the contract. AAPCS64 argument registers:
/// `x0` = context, `x1` = `out`, `x2` = `groups`, `x3` = `rows`, `x4` =
/// `row_skip_bytes`, `v0..3` = x0/y0/z/w. Batch width is 4 lanes; the
/// per-iteration X step is 4.0 and Y advances by 1.0 per row.
#[allow(improper_ctypes_definitions)]
#[cfg(target_arch = "aarch64")]
pub type KernelFn = extern "C" fn(
    *const *const f32,
    *mut f32,
    usize,
    usize,
    usize,
    float32x4_t,
    float32x4_t,
    float32x4_t,
    float32x4_t,
);

/// JIT-compiled *collapse* kernel (x86-64 AVX-512): the whole 2D loop nest is
/// inside the emitted code, so one call fills `rows * groups` output batches —
/// no per-row or per-batch Rust↔JIT boundary. X is an induction value reset at
/// each row: the kernel starts
/// from the caller's lane-sequential `x0` and adds the batch width (16.0)
/// per group; Y starts at `y0` and advances by 1.0 per row. The remaining two
/// base coordinates are the retired axes: passed, never read.
///
/// SysV argument registers: `rdi` = context (array of buffer base pointers,
/// one per declared buffer in slot order, followed by the uniform block's
/// base pointer when the arena declares a uniform — `f32` values in
/// uniform-slot order, read once per call — pass a dangling-free null-less
/// array or anything when the arena declares neither; such a kernel never
/// reads it),
/// `rsi` = `out` (output, written 64 bytes at a time; must hold at least
/// `groups * 16` f32s per row), `rdx` = `groups`, `rcx` = `rows`, `r8` =
/// bytes to skip after each row's final full group, `zmm0..3` = x0/y0/z/w.
#[allow(improper_ctypes_definitions)]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub type KernelFn =
    extern "C" fn(*const *const f32, *mut f32, usize, usize, usize, __m512, __m512, __m512, __m512);

/// JIT-compiled *collapse* kernel (x86-64 AVX2). See the AVX-512
/// [`KernelFn`] doc for the contract; batch width is 8 lanes and the
/// per-iteration X step is 8.0.
#[allow(improper_ctypes_definitions)]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub type KernelFn =
    extern "C" fn(*const *const f32, *mut f32, usize, usize, usize, __m256, __m256, __m256, __m256);

/// JIT-compiled *collapse* kernel (x86-64, 128-bit). See the AVX-512
/// [`KernelFn`] doc for the contract; batch width is 4 lanes and the
/// per-iteration X step is 4.0.
#[allow(improper_ctypes_definitions)]
#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    not(target_feature = "avx2")
))]
pub type KernelFn =
    extern "C" fn(*const *const f32, *mut f32, usize, usize, usize, __m128, __m128, __m128, __m128);

// =============================================================================
// Tests
// =============================================================================

// These tests hand-assemble SSE2 byte sequences and call them through the
// 128-bit `KernelFn`, so they are specific to the SSE2 (no AVX2, no AVX-512)
// ABI. Under `+avx512f` or `+avx2`, `KernelFn` is `__m512`/`__m256` and these
// `__m128` call sites don't type check; those paths are covered by the
/// `Extent2D` arithmetic, independent of ISA level and architecture.
#[cfg(test)]
mod extent_tests {
    use super::*;

    #[test]
    fn len_is_the_product() {
        assert_eq!(Extent2D::new(7, 5).len(), 35);
        assert_eq!(Extent2D::new(0, 5).len(), 0);
    }

    /// A wrapped product is the failure this check exists to prevent: callers
    /// size output buffers by `len()` and assert against it, so `2^32 x 2^32`
    /// wrapping to 0 let an empty buffer satisfy a request for 2^64 pixels
    /// while the kernel was still told to fill 2^32 rows.
    #[test]
    #[should_panic(expected = "Extent2D::len overflowed usize")]
    fn len_refuses_to_wrap() {
        assert_eq!(
            Extent2D::new(1 << 32, 1 << 32).len(),
            0,
            "unreachable: len panics"
        );
    }
}

/// Page preparation, independent of ISA level and architecture.
///
/// Ungated on purpose: `CodePages`, `page_size` and `sync_instruction_cache`
/// are exercised by every build, not just the SSE2 one the module below is
/// limited to.
#[cfg(all(test, unix))]
mod page_tests {
    use super::*;

    /// A single `ret` for the host, so the buffer really is valid machine code
    /// and `from_code`'s safety contract holds. Nothing here executes it.
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
}

#[cfg(all(test, not(target_feature = "avx512f"), not(target_feature = "avx2")))]
mod tests {
    // These tests hand-assemble instruction words as `base | Rd | (Rn << 5) | ...`;
    // the `| 0` / `(0 << 5)` terms document zero register fields on purpose.
    #![allow(clippy::identity_op)]

    use super::*;

    #[allow(improper_ctypes_definitions)]
    #[cfg(target_arch = "aarch64")]
    type KernelFn = extern "C" fn(
        core::arch::aarch64::float32x4_t,
        core::arch::aarch64::float32x4_t,
        core::arch::aarch64::float32x4_t,
        core::arch::aarch64::float32x4_t,
    ) -> core::arch::aarch64::float32x4_t;

    #[allow(improper_ctypes_definitions)]
    #[cfg(target_arch = "x86_64")]
    type KernelFn = extern "C" fn(
        core::arch::x86_64::__m128,
        core::arch::x86_64::__m128,
        core::arch::x86_64::__m128,
        core::arch::x86_64::__m128,
    ) -> core::arch::x86_64::__m128;

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn jit_return_x() {
        // Simplest kernel: return X (already in v0)
        // Just RET - input X is already in v0, which is the return register!

        let mut code = Vec::new();

        crate::emit::aarch64::ret(&mut code);

        unsafe {
            let exec = ExecutableCode::from_code(&code).expect("failed to create executable");
            let func: KernelFn = exec.as_fn();

            use core::arch::aarch64::*;
            let x = vdupq_n_f32(42.0);
            let y = vdupq_n_f32(0.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let result = func(x, y, z, w);

            // Extract first lane
            let val = vgetq_lane_f32(result, 0);
            assert_eq!(val, 42.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn jit_add_xy() {
        // kernel: X + Y
        // v0 = X, v1 = Y, return v0 + v1

        let mut code = Vec::new();

        // FADD V0.4S, V0.4S, V1.4S
        // Encoding: 0x4E20D400 | Rd | (Rn << 5) | (Rm << 16)
        let fadd = 0x4E20D400u32 | 0 | (0 << 5) | (1 << 16);
        code.extend_from_slice(&fadd.to_le_bytes());

        crate::emit::aarch64::ret(&mut code);

        unsafe {
            let exec = ExecutableCode::from_code(&code).expect("failed to create executable");
            let func: KernelFn = exec.as_fn();

            use core::arch::aarch64::*;
            let x = vdupq_n_f32(10.0);
            let y = vdupq_n_f32(32.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let result = func(x, y, z, w);
            let val = vgetq_lane_f32(result, 0);
            assert_eq!(val, 42.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn jit_complex_expr() {
        // kernel: (X + Y) * Z
        // Uses register allocation:
        //   v0=X, v1=Y, v2=Z, v3=W
        //   v0 = v0 + v1  (X + Y)
        //   v0 = v0 * v2  ((X+Y) * Z)
        //   ret

        let mut code = Vec::new();

        // FADD V0.4S, V0.4S, V1.4S
        let fadd = 0x4E20D400u32 | 0 | (0 << 5) | (1 << 16);
        code.extend_from_slice(&fadd.to_le_bytes());

        // FMUL V0.4S, V0.4S, V2.4S
        let fmul = 0x6E20DC00u32 | 0 | (0 << 5) | (2 << 16);
        code.extend_from_slice(&fmul.to_le_bytes());

        crate::emit::aarch64::ret(&mut code);

        unsafe {
            let exec = ExecutableCode::from_code(&code).expect("failed to create executable");
            let func: KernelFn = exec.as_fn();

            use core::arch::aarch64::*;
            let x = vdupq_n_f32(2.0);
            let y = vdupq_n_f32(5.0);
            let z = vdupq_n_f32(6.0);
            let w = vdupq_n_f32(0.0);

            let result = func(x, y, z, w);
            let val = vgetq_lane_f32(result, 0);
            assert_eq!(val, 42.0); // (2 + 5) * 6 = 42
        }
    }

    // =========================================================================
    // Integration tests using the compile() API
    // =========================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn jit_const_05_raw() {
        // Test raw constant loading for 0.5
        // MOVZ W16, #0
        // MOVK W16, #0x3F00, LSL #16  (0x3F000000 = 0.5f)
        // DUP V0.4S, W16
        // RET

        let mut code = Vec::new();

        // MOVZ W16, #0  (lo16 = 0)
        code.extend_from_slice(&0x52800010u32.to_le_bytes());

        // MOVK W16, #0x3F00, LSL #16
        code.extend_from_slice(&(0x72A00010u32 | (0x3F00 << 5)).to_le_bytes());

        // DUP V0.4S, W16
        code.extend_from_slice(&(0x4E040C00u32 | (16 << 5) | 0).to_le_bytes());

        crate::emit::aarch64::ret(&mut code);

        unsafe {
            let exec = ExecutableCode::from_code(&code).expect("failed to create executable");
            let func: KernelFn = exec.as_fn();

            use core::arch::aarch64::*;
            let x = vdupq_n_f32(0.0);
            let y = vdupq_n_f32(0.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let result = func(x, y, z, w);
            assert_eq!(vgetq_lane_f32(result, 0), 0.5);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn jit_return_x_x86() {
        // Simplest kernel: return X (already in xmm0)

        let mut code = Vec::new();

        crate::emit::x86_64::ret(&mut code);

        unsafe {
            let exec = ExecutableCode::from_code(&code).expect("failed to create executable");
            let func: KernelFn = exec.as_fn();

            use core::arch::x86_64::*;
            let x = _mm_set1_ps(42.0);
            let y = _mm_setzero_ps();
            let z = _mm_setzero_ps();
            let w = _mm_setzero_ps();

            let result = func(x, y, z, w);
            let val = _mm_cvtss_f32(result);
            assert_eq!(val, 42.0);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn jit_add_xy_x86() {
        // kernel: X + Y

        let mut code = Vec::new();

        crate::emit::x86_64::emit_addps(&mut code, crate::emit::Reg(0), crate::emit::Reg(1));

        crate::emit::x86_64::ret(&mut code);

        unsafe {
            let exec = ExecutableCode::from_code(&code).expect("failed to create executable");
            let func: KernelFn = exec.as_fn();

            use core::arch::x86_64::*;
            let x = _mm_set1_ps(10.0);
            let y = _mm_set1_ps(32.0);
            let z = _mm_setzero_ps();
            let w = _mm_setzero_ps();

            let result = func(x, y, z, w);
            let val = _mm_cvtss_f32(result);
            assert_eq!(val, 42.0);
        }
    }
}
