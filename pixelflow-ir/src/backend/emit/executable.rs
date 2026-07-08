//! Executable memory management for JIT.
//!
//! This module handles the mmap/mprotect dance to create executable code at runtime.

use core::ptr;

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
    pub unsafe fn from_code(code: &[u8]) -> Result<Self, &'static str> {
        use libc::{MAP_ANON, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE, mmap, mprotect};

        if code.is_empty() {
            return Err("empty code buffer");
        }

        // Round up to page size (usually 4KB, but 16KB on Apple Silicon)
        let page_size = page_size();
        let capacity = (code.len() + page_size - 1) & !(page_size - 1);

        // SAFETY: All syscalls below are safe given valid arguments (which we ensure).
        unsafe {
            // 1. Allocate read-write memory
            let ptr = mmap(
                ptr::null_mut(),
                capacity,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            );

            if ptr == libc::MAP_FAILED {
                return Err("mmap failed");
            }

            let ptr = ptr as *mut u8;

            // 2. Copy code into the buffer
            ptr::copy_nonoverlapping(code.as_ptr(), ptr, code.len());

            // 3. Flip to read-execute (W^X)
            let result = mprotect(ptr as *mut libc::c_void, capacity, PROT_READ | PROT_EXEC);

            if result != 0 {
                libc::munmap(ptr as *mut libc::c_void, capacity);
                return Err("mprotect failed");
            }

            // Instruction cache coherence on Apple Silicon: the I-cache is not
            // automatically coherent with the D-cache, so code written via the
            // store above can otherwise be invisible (or stale) to the fetch
            // unit until explicitly invalidated.
            #[cfg(target_os = "macos")]
            {
                unsafe extern "C" {
                    fn sys_icache_invalidate(start: *mut core::ffi::c_void, size: usize);
                }
                sys_icache_invalidate(ptr as *mut core::ffi::c_void, code.len());
            }

            Ok(Self {
                ptr,
                len: code.len(),
                capacity,
            })
        }
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

impl Drop for ExecutableCode {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.capacity);
        }
    }
}

/// Get the system page size.
#[cfg(unix)]
fn page_size() -> usize {
    // Apple Silicon uses 16KB pages
    #[cfg(target_os = "macos")]
    {
        16384
    }
    #[cfg(not(target_os = "macos"))]
    unsafe {
        libc::sysconf(libc::_SC_PAGESIZE) as usize
    }
}

// =============================================================================
// Reusable code buffer — eliminates mmap/munmap per compile
// =============================================================================

/// A reusable region of executable memory for JIT compilation.
///
/// Allocates a single page (or multiple pages) once via `mmap`, then reuses it
/// across compiles. This eliminates the ~10-20µs mmap/munmap syscall overhead
/// per compile, replacing it with cheaper mprotect toggles (~2-5µs) or
/// `pthread_jit_write_protect_np` on Apple Silicon (~0.5µs).
///
/// # Usage
///
/// ```ignore
/// let mut buf = CodeBuffer::new(65536).expect("mmap failed");
/// let func: KernelFn = buf.write_code(&machine_code)?;
/// // ... call func ...
/// // Next compile reuses the same memory:
/// let func2: KernelFn = buf.write_code(&other_code)?;
/// ```
///
/// # Safety
///
/// The caller must ensure that no references to previously returned function
/// pointers are used after a subsequent `write_code` call (the old code is
/// overwritten).
pub struct CodeBuffer {
    ptr: *mut u8,
    capacity: usize,
    len: usize,
}

// SAFETY: The code buffer is owned by a single thread at a time.
// The caller must ensure no concurrent access to write_code.
unsafe impl Send for CodeBuffer {}

impl CodeBuffer {
    /// Allocate a reusable code buffer of at least `capacity` bytes.
    ///
    /// The actual capacity is rounded up to the system page size.
    /// Returns an error if mmap fails.
    #[cfg(unix)]
    pub fn new(capacity: usize) -> Result<Self, &'static str> {
        use libc::{MAP_ANON, MAP_PRIVATE, PROT_READ, PROT_WRITE, mmap};

        if capacity == 0 {
            return Err("CodeBuffer capacity must be > 0");
        }

        let ps = page_size();
        let capacity = (capacity + ps - 1) & !(ps - 1);

        // On macOS with JIT support, use MAP_JIT for pthread_jit_write_protect_np.
        #[cfg(target_os = "macos")]
        let flags = MAP_PRIVATE | MAP_ANON | libc::MAP_JIT;
        #[cfg(not(target_os = "macos"))]
        let flags = MAP_PRIVATE | MAP_ANON;

        let ptr = unsafe {
            mmap(
                ptr::null_mut(),
                capacity,
                PROT_READ | PROT_WRITE,
                flags,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err("CodeBuffer: mmap failed");
        }

        // Immediately flip to RX so it's in a safe default state.
        #[cfg(not(target_os = "macos"))]
        {
            let rc = unsafe { libc::mprotect(ptr, capacity, libc::PROT_READ | libc::PROT_EXEC) };
            if rc != 0 {
                unsafe {
                    libc::munmap(ptr, capacity);
                }
                return Err("CodeBuffer: initial mprotect to RX failed");
            }
        }

        #[cfg(target_os = "macos")]
        {
            // With MAP_JIT on macOS, the memory starts RW. Toggle to RX.
            unsafe {
                toggle_jit_write(JitWriteState::Executable);
            }
        }

        Ok(Self {
            ptr: ptr as *mut u8,
            capacity,
            len: 0,
        })
    }

    /// Write machine code into the buffer and return a function pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// 1. `code` contains valid machine code for the current architecture.
    /// 2. No previously returned function pointers are called after this.
    /// 3. `code.len() <= self.capacity`.
    #[cfg(unix)]
    pub unsafe fn write_code<F: Copy>(&mut self, code: &[u8]) -> Result<F, &'static str> {
        if code.is_empty() {
            return Err("CodeBuffer: empty code");
        }
        if code.len() > self.capacity {
            return Err("CodeBuffer: code exceeds buffer capacity");
        }

        // SAFETY: All unsafe operations below are protected by the function's
        // safety contract: valid machine code, no concurrent access, code fits.
        unsafe {
            // Toggle to writable.
            #[cfg(target_os = "macos")]
            {
                toggle_jit_write(JitWriteState::Writable);
            }
            #[cfg(not(target_os = "macos"))]
            {
                let rc = libc::mprotect(
                    self.ptr as *mut libc::c_void,
                    self.capacity,
                    libc::PROT_READ | libc::PROT_WRITE,
                );
                if rc != 0 {
                    return Err("CodeBuffer: mprotect to RW failed");
                }
            }

            // Copy code.
            ptr::copy_nonoverlapping(code.as_ptr(), self.ptr, code.len());
            self.len = code.len();

            // Toggle to executable.
            #[cfg(target_os = "macos")]
            {
                toggle_jit_write(JitWriteState::Executable);
                // Instruction cache coherence on Apple Silicon.
                // sys_icache_invalidate is needed after writing code on ARM.
                unsafe extern "C" {
                    fn sys_icache_invalidate(start: *mut core::ffi::c_void, size: usize);
                }
                sys_icache_invalidate(self.ptr as *mut core::ffi::c_void, code.len());
            }
            #[cfg(not(target_os = "macos"))]
            {
                let rc = libc::mprotect(
                    self.ptr as *mut libc::c_void,
                    self.capacity,
                    libc::PROT_READ | libc::PROT_EXEC,
                );
                if rc != 0 {
                    return Err("CodeBuffer: mprotect to RX failed");
                }
            }

            Ok(core::mem::transmute_copy(&self.ptr))
        }
    }

    /// Current code length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer contains no code.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total capacity in bytes.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Drop for CodeBuffer {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.capacity);
        }
    }
}

/// Toggle JIT write protection on macOS (Apple Silicon).
///
/// When `writable` is true, the current thread can write to MAP_JIT memory.
/// When false, the memory is executable but not writable (W^X).
///
/// This is much cheaper than mprotect (~0.5µs vs ~5µs) and is per-thread,
/// so it doesn't affect other threads' ability to execute the code.
#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JitWriteState {
    Writable,
    Executable,
}

#[cfg(target_os = "macos")]
unsafe fn toggle_jit_write(state: JitWriteState) {
    // pthread_jit_write_protect_np(true) = write-protect (executable)
    // pthread_jit_write_protect_np(false) = writable (not executable)
    // Note: the semantics are inverted from what you'd expect!
    unsafe extern "C" {
        fn pthread_jit_write_protect_np(enabled: core::ffi::c_int);
    }

    let enabled = match state {
        JitWriteState::Executable => 1,
        JitWriteState::Writable => 0,
    };

    // SAFETY: pthread_jit_write_protect_np is always safe to call — it only
    // affects the calling thread's JIT write permission.
    unsafe {
        pthread_jit_write_protect_np(enabled);
    }
}

// =============================================================================
// Kernel type aliases for JIT-compiled functions
// =============================================================================

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::float32x4_t;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::__m128;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use core::arch::x86_64::__m512;

/// JIT-compiled kernel signature for ARM64.
/// Args: X in v0, Y in v1, Z in v2, W in v3; returns result in v0.
/// Each arg/result is a SIMD vector, so one call computes one pixel per lane
/// (4 pixels for a 128-bit vector), not a single pixel; the caller loops.
#[allow(improper_ctypes_definitions)]
#[cfg(target_arch = "aarch64")]
pub type KernelFn =
    extern "C" fn(float32x4_t, float32x4_t, float32x4_t, float32x4_t) -> float32x4_t;

/// JIT-compiled kernel that reads bound memory (ARM64).
///
/// Identical to [`KernelFn`] plus a leading context pointer: an array of buffer
/// base pointers, one per declared [`BufferId`](crate::arena::BufferId) in slot
/// order. AAPCS64 places this integer-class pointer in `x0`, disjoint from the
/// coordinate vectors in `v0..3`, so the emitted body is byte-for-byte the same
/// as a `KernelFn` — only kernels containing a `Gather` read `x0`. The caller
/// picks this type iff the arena declared buffers.
#[allow(improper_ctypes_definitions)]
#[cfg(target_arch = "aarch64")]
pub type CtxKernelFn = extern "C" fn(
    *const *const f32,
    float32x4_t,
    float32x4_t,
    float32x4_t,
    float32x4_t,
) -> float32x4_t;

/// JIT-compiled scanline kernel signature for ARM64.
///
/// Processes an entire scanline in a single call with no per-batch Rust-JIT boundary.
/// Y/Z/W stay in registers across the entire loop (loop-invariant hoisting by construction).
///
/// Args:
///   x0 = pointer to input X array (128-bit aligned `float32x4_t` values)
///   v1 = Y (broadcast, loop-invariant)
///   v2 = Z (broadcast, loop-invariant)
///   v3 = W (broadcast, loop-invariant)
///   x1 = pointer to output array (128-bit aligned `float32x4_t` values)
///   x2 = count (number of SIMD groups to process)
// SIMD vector types are not nominally FFI-safe, but both sides of this
// boundary are our own JIT-emitted code using the platform vector ABI.
#[allow(improper_ctypes_definitions)]
#[cfg(target_arch = "aarch64")]
pub type ScanlineKernelFn = extern "C" fn(
    *const float32x4_t, // x_array
    float32x4_t,        // y (broadcast)
    float32x4_t,        // z (broadcast)
    float32x4_t,        // w (broadcast)
    *mut float32x4_t,   // output array
    usize,              // count
);

/// JIT-compiled per-batch kernel signature for x86-64.
///
/// Args: X/Y/Z/W in the first four vector registers; returns the result in the
/// first. One call computes one pixel per lane; the caller loops.
///
/// The width tracks the build's selected SIMD: 512-bit `__m512` (16 lanes) when
/// compiled with AVX-512, else 128-bit `__m128` (4 lanes, SSE2). This MUST match
/// `pixelflow-core`'s `Field`; the `kernel_jit!` wrapper const-asserts
/// `size_of::<Field>() == JIT_VECTOR_BYTES`. The looping variant is
/// `ScanlineKernelFn`, which stays 128-bit (the scanline emitter is still SSE2).
#[allow(improper_ctypes_definitions)]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub type KernelFn = extern "C" fn(__m512, __m512, __m512, __m512) -> __m512;

/// JIT-compiled kernel that reads bound memory (x86-64 AVX-512).
///
/// Identical to [`KernelFn`] plus a leading context pointer: an array of buffer
/// base pointers, one per declared [`BufferId`](crate::arena::BufferId) in slot
/// order. System V places this integer-class pointer in `rdi`, disjoint from the
/// coordinate vectors in `zmm0..3`, so the emitted body is byte-for-byte the
/// same as a `KernelFn` — only kernels containing a `Gather` read `rdi`. The
/// caller picks this type iff the arena declared buffers.
#[allow(improper_ctypes_definitions)]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub type CtxKernelFn = extern "C" fn(*const *const f32, __m512, __m512, __m512, __m512) -> __m512;

/// JIT-compiled *collapse* kernel (x86-64 AVX-512): the whole lattice domain
/// loop is inside the emitted code, so one call fills the entire output — no
/// per-batch Rust↔JIT boundary. This is the internal-loop realization of the
/// lattice: coordinates are induction values, not arguments.
///
/// SysV argument registers: `rdi` = context (array of buffer base pointers),
/// `rsi` = `xs` (domain X coordinates, 16-lane groups), `rdx` = `out` (output,
/// 16-lane groups), `rcx` = `groups` (number of 16-lane groups). Y/Z/W are zero
/// inside the kernel. `xs`/`out` are read/written 64 bytes at a time and must
/// hold at least `groups * 16` f32s.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub type CollapseKernelFn = extern "C" fn(*const *const f32, *const f32, *mut f32, usize);

#[allow(improper_ctypes_definitions)]
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
pub type KernelFn = extern "C" fn(__m128, __m128, __m128, __m128) -> __m128;

/// JIT-compiled scanline kernel signature for x86-64 (128-bit; the scanline
/// emitter is SSE2 only, independent of the per-batch width).
#[allow(improper_ctypes_definitions)]
#[cfg(target_arch = "x86_64")]
pub type ScanlineKernelFn = extern "C" fn(
    *const __m128, // x_array
    __m128,        // y (broadcast)
    __m128,        // z (broadcast)
    __m128,        // w (broadcast)
    *mut __m128,   // output array
    usize,         // count
);

// =============================================================================
// Tests
// =============================================================================

// These tests hand-assemble SSE2 byte sequences and call them through the
// 128-bit `KernelFn`, so they are specific to the non-AVX-512 ABI. Under
// `+avx512f`, `KernelFn` is `__m512` and these `__m128` call sites don't type
// check; the AVX-512 path is covered by the `avx512` tests in `mod.rs`.
#[cfg(all(test, not(target_feature = "avx512f")))]
mod tests {
    // These tests hand-assemble instruction words as `base | Rd | (Rn << 5) | ...`;
    // the `| 0` / `(0 << 5)` terms document zero register fields on purpose.
    #![allow(clippy::identity_op)]

    use super::*;

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn jit_return_x() {
        // Simplest kernel: return X (already in v0)
        // Just RET - input X is already in v0, which is the return register!

        let mut code = Vec::new();

        // RET
        code.extend_from_slice(&0xD65F03C0u32.to_le_bytes());

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

        // RET
        code.extend_from_slice(&0xD65F03C0u32.to_le_bytes());

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

        // RET
        code.extend_from_slice(&0xD65F03C0u32.to_le_bytes());

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

        // RET
        code.extend_from_slice(&0xD65F03C0u32.to_le_bytes());

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

        // RET
        code.push(0xC3);

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

        // ADDPS xmm0, xmm1
        code.extend_from_slice(&[0x0F, 0x58, 0xC1]);

        // RET
        code.push(0xC3);

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
