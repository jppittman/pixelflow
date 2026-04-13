//! # Parallel Evaluator
//!
//! Evaluates a pure manifold kernel over discrete memory grids (Lattices).
//! This decouples memory access and scheduling from the kernel logic itself.
//! The kernel remains a pure mathematical function, while the host orchestrates
//! memory loading, SIMD chunking, and multithreading.

use crate::{Field, Manifold, PARALLELISM};

#[cfg(feature = "std")]
use std::sync::atomic::{AtomicUsize, Ordering};

/// Configuration for parallel evaluation.
#[derive(Copy, Clone, Debug)]
pub struct EvalOptions {
    /// Number of threads to use. 0 or 1 means single-threaded.
    pub num_threads: usize,
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self { num_threads: 1 }
    }
}

#[cfg(feature = "std")]
#[derive(Copy, Clone)]
struct SendPtr<T>(*mut T);

#[cfg(feature = "std")]
unsafe impl<T> Send for SendPtr<T> {}
#[cfg(feature = "std")]
unsafe impl<T> Sync for SendPtr<T> {}

#[cfg(feature = "std")]
const STACK_SIZE: usize = 2 * 1024 * 1024;

/// Evaluate a 2D kernel into a continuous slice of memory.
///
/// The kernel is evaluated at integer coordinates `(x, y)` corresponding to
/// the indices of the 2D grid. The resulting values are written into `data`
/// in row-major order.
///
/// If `options.num_threads > 1` and the `std` feature is enabled, this uses
/// work-stealing parallelism.
pub fn evaluate_2d<M>(
    data: &mut [f32],
    width: usize,
    height: usize,
    manifold: &M,
    options: EvalOptions,
) where
    M: Manifold<(Field, Field, Field, Field), Output = Field> + Sync,
{
    assert!(data.len() >= width * height, "Buffer too small for dimensions");

    #[cfg(feature = "std")]
    if options.num_threads > 1 && height > 1 {
        evaluate_2d_parallel(data, width, height, manifold, options.num_threads);
        return;
    }

    // Single-threaded fallback
    for y in 0..height {
        let offset = y * width;
        let row_slice = &mut data[offset..offset + width];
        evaluate_stripe_2d(manifold, row_slice, y, width);
    }
}

#[cfg(feature = "std")]
fn evaluate_2d_parallel<M>(
    data: &mut [f32],
    width: usize,
    height: usize,
    manifold: &M,
    num_threads: usize,
) where
    M: Manifold<(Field, Field, Field, Field), Output = Field> + Sync,
{
    let next_row = AtomicUsize::new(0);
    let buffer_ptr = SendPtr(data.as_mut_ptr());

    std::thread::scope(|s| {
        for _ in 0..num_threads {
            std::thread::Builder::new()
                .stack_size(STACK_SIZE)
                .spawn_scoped(s, || {
                    let ptr = buffer_ptr;
                    loop {
                        let row = next_row.fetch_add(1, Ordering::Relaxed);
                        if row >= height {
                            break;
                        }

                        let offset = row * width;
                        let row_slice =
                            unsafe { core::slice::from_raw_parts_mut(ptr.0.add(offset), width) };

                        evaluate_stripe_2d(manifold, row_slice, row, width);
                    }
                })
                .expect("Failed to spawn evaluation thread");
        }
    });
}

fn evaluate_stripe_2d<M>(manifold: &M, target: &mut [f32], y: usize, width: usize)
where
    M: Manifold<(Field, Field, Field, Field), Output = Field> + ?Sized,
{
    let mut packed = [0.0f32; PARALLELISM];
    let mut x = 0;

    // We evaluate exactly at the integer indices of the grid
    let ys = Field::from(y as f32);
    let mut xs = Field::sequential(0.0);
    let step = Field::from(PARALLELISM as f32);
    let zero = Field::from(0.0);

    use crate::numeric::Numeric;

    #[cfg(feature = "alloc")]
    if let Some(func_ptr) = manifold.jit_ptr() {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use core::arch::aarch64::float32x4_t;
            while x + PARALLELISM <= width {
                let mut result: float32x4_t;
                let vx = core::mem::transmute::<Field, float32x4_t>(xs);
                let vy = core::mem::transmute::<Field, float32x4_t>(ys);
                let vz = core::mem::transmute::<Field, float32x4_t>(zero);
                let vw = core::mem::transmute::<Field, float32x4_t>(zero);

                core::arch::asm!(
                    "blr {func}",
                    func = in(reg) func_ptr,
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
                    "blr {func}",
                    func = in(reg) func_ptr,
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
        
        #[cfg(not(target_arch = "aarch64"))]
        {
            // Fallback for non-aarch64 (like x86_64) - could implement naked x86 call here too.
            // For now, we'll let it drop through to the AOT path, 
            // though ideally it would cast func_ptr to extern "C" fn.
        }
        
        #[cfg(target_arch = "aarch64")]
        return;
    }

    // SIMD hot path (AOT Rust or non-JIT path)
    while x + PARALLELISM <= width {
        let result = manifold.eval((xs, ys, zero, zero));
        result.store(&mut packed);
        target[x..x + PARALLELISM].copy_from_slice(&packed);

        x += PARALLELISM;
        xs = xs.raw_add(step);
    }

    // SIMD tail (no scalar fallback required!)
    if x < width {
        let result = manifold.eval((xs, ys, zero, zero));
        result.store(&mut packed);
        let tail_len = width - x;
        target[x..width].copy_from_slice(&packed[..tail_len]);
    }
}
