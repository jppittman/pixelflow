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
        use libc::{mmap, mprotect, MAP_ANON, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE};

        if code.is_empty() {
            return Err("empty code buffer");
        }

        // Round up to page size (usually 4KB, but 16KB on Apple Silicon)
        let page_size = page_size();
        let capacity = (code.len() + page_size - 1) & !(page_size - 1);

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

        Ok(Self {
            ptr,
            len: code.len(),
            capacity,
        })
    }

    /// Get a function pointer to the compiled code.
    ///
    /// # Safety
    /// The caller must ensure the code implements the correct calling convention
    /// and signature for type `F`.
    #[inline]
    pub unsafe fn as_fn<F>(&self) -> F {
        core::mem::transmute_copy(&self.ptr)
    }

    /// Get the code as a byte slice (for debugging).
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Length of the compiled code in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the code is empty.
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
// Kernel type aliases for JIT-compiled functions
// =============================================================================

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::float32x4_t;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::__m128;

/// JIT-compiled kernel signature for ARM64.
/// Args: X in v0, Y in v1, Z in v2, W in v3
/// Returns: result in v0
#[cfg(target_arch = "aarch64")]
pub type KernelFn =
    extern "C" fn(float32x4_t, float32x4_t, float32x4_t, float32x4_t) -> float32x4_t;

/// JIT-compiled kernel signature for x86-64.
/// Args: X in xmm0, Y in xmm1, Z in xmm2, W in xmm3
/// Returns: result in xmm0
#[cfg(target_arch = "x86_64")]
pub type KernelFn = extern "C" fn(__m128, __m128, __m128, __m128) -> __m128;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_jit_return_x() {
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
    fn test_jit_add_xy() {
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
    fn test_jit_complex_expr() {
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
    fn test_compile_return_x() {
        use crate::backend::emit::compile;
        use crate::expr::Expr;

        // Simplest: just return X
        let expr = Expr::Var(0);
        let exec = compile(&expr).expect("compile failed");

        unsafe {
            let func: KernelFn = exec.as_fn();

            use core::arch::aarch64::*;
            let x = vdupq_n_f32(123.0);
            let y = vdupq_n_f32(0.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let result = func(x, y, z, w);
            assert_eq!(vgetq_lane_f32(result, 0), 123.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_compile_return_y() {
        use crate::backend::emit::compile;
        use crate::expr::Expr;

        // Return Y (needs MOV to v0)
        let expr = Expr::Var(1);
        let exec = compile(&expr).expect("compile failed");

        unsafe {
            let func: KernelFn = exec.as_fn();

            use core::arch::aarch64::*;
            let x = vdupq_n_f32(0.0);
            let y = vdupq_n_f32(456.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let result = func(x, y, z, w);
            assert_eq!(vgetq_lane_f32(result, 0), 456.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_compile_add_xy() {
        use crate::backend::emit::compile;
        use crate::expr::Expr;
        use crate::kind::OpKind;

        // X + Y
        let expr = Expr::Binary(OpKind::Add, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let exec = compile(&expr).expect("compile failed");

        unsafe {
            let func: KernelFn = exec.as_fn();

            use core::arch::aarch64::*;
            let x = vdupq_n_f32(10.0);
            let y = vdupq_n_f32(32.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let result = func(x, y, z, w);
            assert_eq!(vgetq_lane_f32(result, 0), 42.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_compile_complex() {
        use crate::backend::emit::compile;
        use crate::expr::Expr;
        use crate::kind::OpKind;

        // (X + Y) * Z
        let expr = Expr::Binary(
            OpKind::Mul,
            Box::new(Expr::Binary(
                OpKind::Add,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
            )),
            Box::new(Expr::Var(2)),
        );
        let exec = compile(&expr).expect("compile failed");

        unsafe {
            let func: KernelFn = exec.as_fn();

            use core::arch::aarch64::*;
            let x = vdupq_n_f32(2.0);
            let y = vdupq_n_f32(5.0);
            let z = vdupq_n_f32(6.0);
            let w = vdupq_n_f32(0.0);

            let result = func(x, y, z, w);
            assert_eq!(vgetq_lane_f32(result, 0), 42.0); // (2+5)*6 = 42
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_compile_const() {
        use crate::backend::emit::compile;
        use crate::expr::Expr;

        // Return a constant
        let expr = Expr::Const(42.0);
        let exec = compile(&expr).expect("compile failed");

        unsafe {
            let func: KernelFn = exec.as_fn();

            use core::arch::aarch64::*;
            let x = vdupq_n_f32(0.0);
            let y = vdupq_n_f32(0.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let result = func(x, y, z, w);
            assert_eq!(vgetq_lane_f32(result, 0), 42.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_compile_floor() {
        use crate::backend::emit::compile;
        use crate::expr::Expr;
        use crate::kind::OpKind;

        // floor(X)
        let expr = Expr::Unary(OpKind::Floor, Box::new(Expr::Var(0)));
        let exec = compile(&expr).expect("compile failed");

        unsafe {
            let func: KernelFn = exec.as_fn();

            use core::arch::aarch64::*;
            let x = vdupq_n_f32(42.7);
            let y = vdupq_n_f32(0.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let result = func(x, y, z, w);
            assert_eq!(vgetq_lane_f32(result, 0), 42.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_compile_mul_add() {
        use crate::backend::emit::compile;
        use crate::expr::Expr;
        use crate::kind::OpKind;

        // X * Y + Z (FMA)
        let expr = Expr::Ternary(
            OpKind::MulAdd,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
            Box::new(Expr::Var(2)),
        );
        let exec = compile(&expr).expect("compile failed");

        unsafe {
            let func: KernelFn = exec.as_fn();

            use core::arch::aarch64::*;
            let x = vdupq_n_f32(6.0);
            let y = vdupq_n_f32(7.0);
            let z = vdupq_n_f32(0.0); // 6*7 + 0 = 42
            let w = vdupq_n_f32(0.0);

            let result = func(x, y, z, w);
            assert_eq!(vgetq_lane_f32(result, 0), 42.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_compile_const_negative() {
        use crate::backend::emit::compile;
        use crate::expr::Expr;

        // Return a negative constant
        let expr = Expr::Const(-42.0);
        let exec = compile(&expr).expect("compile failed");

        unsafe {
            let func: KernelFn = exec.as_fn();

            use core::arch::aarch64::*;
            let x = vdupq_n_f32(0.0);
            let y = vdupq_n_f32(0.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let result = func(x, y, z, w);
            assert_eq!(vgetq_lane_f32(result, 0), -42.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_compile_const_pi() {
        use crate::backend::emit::compile;
        use crate::expr::Expr;

        // Return π (not a simple constant)
        let expr = Expr::Const(core::f32::consts::PI);
        let exec = compile(&expr).expect("compile failed");

        unsafe {
            let func: KernelFn = exec.as_fn();

            use core::arch::aarch64::*;
            let x = vdupq_n_f32(0.0);
            let y = vdupq_n_f32(0.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let result = func(x, y, z, w);
            let val = vgetq_lane_f32(result, 0);
            assert!(
                (val - core::f32::consts::PI).abs() < 0.0001,
                "PI = {}, expected ~3.14159",
                val
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_jit_const_05_raw() {
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
    #[cfg(target_arch = "aarch64")]
    fn test_compile_x_plus_const() {
        use crate::backend::emit::compile;
        use crate::expr::Expr;
        use crate::kind::OpKind;

        // X + 0.5
        let expr = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(0.5)),
        );
        let exec = compile(&expr).expect("compile failed");

        unsafe {
            let func: KernelFn = exec.as_fn();

            use core::arch::aarch64::*;
            let x = vdupq_n_f32(41.5);
            let y = vdupq_n_f32(0.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let result = func(x, y, z, w);
            assert_eq!(vgetq_lane_f32(result, 0), 42.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_compile_floor_add() {
        use crate::backend::emit::compile;
        use crate::expr::Expr;
        use crate::kind::OpKind;

        // floor(X + 0.5) - common rounding pattern
        let expr = Expr::Unary(
            OpKind::Floor,
            Box::new(Expr::Binary(
                OpKind::Add,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(0.5)),
            )),
        );
        let exec = compile(&expr).expect("compile failed");

        unsafe {
            let func: KernelFn = exec.as_fn();

            use core::arch::aarch64::*;
            let x = vdupq_n_f32(41.3); // floor(41.3 + 0.5) = floor(41.8) = 41
            let y = vdupq_n_f32(0.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let result = func(x, y, z, w);
            assert_eq!(vgetq_lane_f32(result, 0), 41.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_compile_horner() {
        use crate::backend::emit::compile;
        use crate::expr::Expr;
        use crate::kind::OpKind;

        // Simple Horner: c1 * x + c0 with x=0 should give c0=5.0
        // mul_add(c1=2.0, x, c0=5.0) = 2*0 + 5 = 5
        let expr = Expr::Ternary(
            OpKind::MulAdd,
            Box::new(Expr::Const(2.0)), // c1
            Box::new(Expr::Var(0)),     // x
            Box::new(Expr::Const(5.0)), // c0
        );
        let exec = compile(&expr).expect("compile failed");

        unsafe {
            let func: KernelFn = exec.as_fn();

            use core::arch::aarch64::*;
            let x = vdupq_n_f32(0.0); // 2*0 + 5 = 5
            let y = vdupq_n_f32(0.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let result = func(x, y, z, w);
            assert_eq!(vgetq_lane_f32(result, 0), 5.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_compile_horner_with_x() {
        use crate::backend::emit::compile;
        use crate::expr::Expr;
        use crate::kind::OpKind;

        // c1 * x + c0 with x=3 should give 2*3 + 5 = 11
        let expr = Expr::Ternary(
            OpKind::MulAdd,
            Box::new(Expr::Const(2.0)),
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(5.0)),
        );
        let exec = compile(&expr).expect("compile failed");

        unsafe {
            let func: KernelFn = exec.as_fn();

            use core::arch::aarch64::*;
            let x = vdupq_n_f32(3.0); // 2*3 + 5 = 11
            let y = vdupq_n_f32(0.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let result = func(x, y, z, w);
            assert_eq!(vgetq_lane_f32(result, 0), 11.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_compile_sin_lowered() {
        use crate::backend::emit::compile;
        use crate::expr::Expr;
        use crate::kind::OpKind;

        // sin(X) - should be lowered to polynomial
        let expr = Expr::Unary(OpKind::Sin, Box::new(Expr::Var(0)));
        let exec = compile(&expr).expect("compile failed");

        unsafe {
            let func: KernelFn = exec.as_fn();

            use core::arch::aarch64::*;
            // sin(0) = 0
            let x = vdupq_n_f32(0.0);
            let y = vdupq_n_f32(0.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let result = func(x, y, z, w);
            let val = vgetq_lane_f32(result, 0);
            assert!((val - 0.0).abs() < 0.01, "sin(0) = {}, expected ~0", val);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_jit_return_x_x86() {
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
    fn test_jit_add_xy_x86() {
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
