//! JIT code execution and code-patching evaluators.
//!
//! Provides inline-cached execution of JIT kernels bypassing the C ABI.

use crate::{Field, PARALLELISM};

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::float32x4_t;

/// Patch the inline cache in the evaluator loop to point to a new JIT kernel.
///
/// # Safety
/// This mutates the `.text` segment. It is inherently thread-unsafe if multiple
/// threads attempt to evaluate *different* JIT kernels concurrently using the
/// global inline cache.
pub unsafe fn patch_evaluator(target_ptr: *const u8) {
    #[cfg(target_arch = "aarch64")]
    {
        extern "C" {
            // The symbol defined in our inline assembly block
            fn _pixelflow_jit_patch_site();
        }

        let site_addr = _pixelflow_jit_patch_site as usize;
        let target_addr = target_ptr as usize;

        // Calculate PC-relative offset in words (instructions)
        let diff = (target_addr as isize) - (site_addr as isize);
        
        if diff < -(128 * 1024 * 1024) || diff > (128 * 1024 * 1024 - 4) {
            panic!("JIT kernel is outside the +/- 128MB branch range for BL instruction");
        }

        let offset_words = (diff / 4) as i32;

        // Construct BL instruction: 0x94000000 | (offset & 0x03FFFFFF)
        let bl_inst = 0x94000000u32 | ((offset_words as u32) & 0x03FFFFFF);

        // Patch the memory
        pixelflow_ir::backend::emit::patch::begin_patching();
        let ptr = site_addr as *mut u32;
        ptr.write_volatile(bl_inst);
        pixelflow_ir::backend::emit::patch::end_patching(site_addr as *mut u8, 4);
    }
}

/// Evaluates a JIT-compiled kernel over a 1D stripe using a patched inline cache.
///
/// You MUST call `patch_evaluator(jit)` before calling this, and ensure no other
/// thread patches it while this runs.
pub fn evaluate_stripe_jit_patched(target: &mut [f32], y: usize, width: usize) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut packed = [0.0f32; PARALLELISM];
        let mut x = 0;
        
        let mut xs = Field::sequential(0.0);
        let ys = Field::from(y as f32);
        let zero = Field::from(0.0);
        let step = Field::from(PARALLELISM as f32);
        
        use crate::numeric::Numeric;

        while x + PARALLELISM <= width {
            let mut result: float32x4_t;
            let vx = core::mem::transmute::<Field, float32x4_t>(xs);
            let vy = core::mem::transmute::<Field, float32x4_t>(ys);
            let vz = core::mem::transmute::<Field, float32x4_t>(zero);
            let vw = core::mem::transmute::<Field, float32x4_t>(zero);

            core::arch::asm!(
                // We define the global patch site.
                // It defaults to a BRK (breakpoint) so if it's unpatched, it crashes loudly.
                ".global _pixelflow_jit_patch_site",
                "_pixelflow_jit_patch_site:",
                "brk #1", 
                inout("v0") vx => result,
                in("v1") vy,
                in("v2") vz,
                in("v3") vw,
                // The JIT kernel clobbers v4-v27 for computation, and v28-v31 for scratch.
                // We don't need to clobber x0-x15 because the JIT only uses x16/x17.
                out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                out("v24") _, out("v25") _, out("v26") _, out("v27") _,
                out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                out("x16") _, out("x17") _,
                options(nomem, nostack, preserves_flags)
            );

            let res_field = core::mem::transmute::<float32x4_t, Field>(result);
            res_field.store(&mut packed);
            target[x..x + PARALLELISM].copy_from_slice(&packed);

            x += PARALLELISM;
            xs = xs.raw_add(step);
        }

        if x < width {
            let mut result: float32x4_t;
            let vx = core::mem::transmute::<Field, float32x4_t>(xs);
            let vy = core::mem::transmute::<Field, float32x4_t>(ys);
            let vz = core::mem::transmute::<Field, float32x4_t>(zero);
            let vw = core::mem::transmute::<Field, float32x4_t>(zero);

            core::arch::asm!(
                "bl _pixelflow_jit_patch_site", // Use local BL to the site we patched above
                inout("v0") vx => result,
                in("v1") vy,
                in("v2") vz,
                in("v3") vw,
                out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                out("v24") _, out("v25") _, out("v26") _, out("v27") _,
                out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                out("x16") _, out("x17") _,
                options(nomem, nostack, preserves_flags)
            );

            let res_field = core::mem::transmute::<float32x4_t, Field>(result);
            res_field.store(&mut packed);
            let tail_len = width - x;
            target[x..width].copy_from_slice(&packed[..tail_len]);
        }
    }
}
