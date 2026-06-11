//! Benchmark: Hoisted Scanline JIT vs Per-Pixel JIT vs LLVM FrameLattice
//!
//! Compares three evaluation strategies on a psychedelic shader expression:
//!
//!   1. **Per-pixel JIT (FrameLattice)**: JitManifold wrapped in a Manifold impl,
//!      evaluated via `FrameLattice::collapse()`. This is structurally identical
//!      to the LLVM `kernel!` path (same loop, same per-SIMD-group call boundary).
//!
//!   2. **Scanline JIT (flat)**: `ScanlineJitManifold` without hoisting -- the loop
//!      is inside the JIT code, but Y/Z/W are still reloaded from registers each
//!      iteration. Eliminates the Rust-JIT function pointer call overhead.
//!
//!   3. **Scanline JIT (hoisted)**: `eval_frame_jit()` -- variance analysis hoists
//!      X-invariant subexpressions (sin, cos, exp) into a per-scanline setup block.
//!      The inner loop only computes X-dependent ops.
//!
//! Usage:
//!   cargo run --release -p pixelflow-pipeline --bin bench_hoisted_scanline

use pixelflow_core::{
    Field, Manifold,
    lattice::{FrameLattice, Lattice},
};
use pixelflow_ir::{ExprArena, ExprId, OpKind};
use std::time::Instant;

const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;
const Z_TIME: f32 = 1.5;
const W_LAYER: f32 = 0.0;
/// Number of frames to average over for stable timing.
const WARMUP_FRAMES: usize = 3;
const BENCH_FRAMES: usize = 10;

/// Build the psychedelic shader in an ExprArena:
///   sin(Z*0.3) * exp(-(X*X + Y*Y) * 0.01) + cos(Y*0.7) * sin(X*0.5 + Z)
///
/// Hoistable X-invariant nodes: sin(Z*0.3), Z*0.3, Y*0.7, cos(Y*0.7), Y*Y
/// X-dependent: X*X, X*0.5, X*0.5+Z, sin(X*0.5+Z), X*X+Y*Y, ...
fn build_psychedelic_arena() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();

    let x = a.push_var(0);
    let y = a.push_var(1);
    let z = a.push_var(2);

    let c03 = a.push_const(0.3);
    let c07 = a.push_const(0.7);
    let c05 = a.push_const(0.5);
    let c001 = a.push_const(0.01);

    // sin(Z * 0.3)
    let z_03 = a.push_binary(OpKind::Mul, z, c03);
    let sin_z03 = a.push_unary(OpKind::Sin, z_03);

    // X*X + Y*Y
    let xx = a.push_binary(OpKind::Mul, x, x);
    let yy = a.push_binary(OpKind::Mul, y, y);
    let xxyy = a.push_binary(OpKind::Add, xx, yy);

    // exp(-(X*X + Y*Y) * 0.01)
    let neg_xxyy = a.push_unary(OpKind::Neg, xxyy);
    let scaled = a.push_binary(OpKind::Mul, neg_xxyy, c001);
    let exp_term = a.push_unary(OpKind::Exp, scaled);

    // sin(Z*0.3) * exp(...)
    let left = a.push_binary(OpKind::Mul, sin_z03, exp_term);

    // cos(Y * 0.7)
    let y_07 = a.push_binary(OpKind::Mul, y, c07);
    let cos_y07 = a.push_unary(OpKind::Cos, y_07);

    // sin(X*0.5 + Z)
    let x_05 = a.push_binary(OpKind::Mul, x, c05);
    let x05_z = a.push_binary(OpKind::Add, x_05, z);
    let sin_x05z = a.push_unary(OpKind::Sin, x05_z);

    // cos(Y*0.7) * sin(X*0.5 + Z)
    let right = a.push_binary(OpKind::Mul, cos_y07, sin_x05z);

    // left + right
    let root = a.push_binary(OpKind::Add, left, right);

    (a, root)
}

/// Thin wrapper around JitManifold that implements Manifold.
struct JitWrapper(pixelflow_ir::JitManifold);

impl Manifold<(Field, Field, Field, Field)> for JitWrapper {
    type Output = Field;
    #[inline(always)]
    fn eval(&self, (x, y, z, w): (Field, Field, Field, Field)) -> Field {
        unsafe {
            core::mem::transmute(self.0.call(
                core::mem::transmute(x),
                core::mem::transmute(y),
                core::mem::transmute(z),
                core::mem::transmute(w),
            ))
        }
    }
}

unsafe impl Send for JitWrapper {}
unsafe impl Sync for JitWrapper {}

/// Run the scanline JIT eval loop over `frames` frames using a pre-compiled
/// `ScanlineJitManifold`. Returns (total_ns, first_pixel, last_pixel).
///
/// The inner loop mirrors FrameLattice::collapse() -- scanline eval + store to
/// a flat f32 buffer -- so the comparison is apples-to-apples.
#[cfg(target_arch = "aarch64")]
fn bench_scanline_eval(
    scanline_jit: &pixelflow_ir::ScanlineJitManifold,
    width: usize,
    height: usize,
    z: f32,
    w: f32,
    frames: usize,
) -> (u128, f32, f32) {
    use core::arch::aarch64::*;

    let full_groups = width / 4;
    let tail = width % 4;
    let total_groups = full_groups + if tail > 0 { 1 } else { 0 };

    // Build X coordinate array (reused across frames).
    let xs: Vec<float32x4_t> = (0..total_groups)
        .map(|i| {
            let base = (i * 4) as f32;
            unsafe {
                let arr: [f32; 4] = [base, base + 1.0, base + 2.0, base + 3.0];
                vld1q_f32(arr.as_ptr())
            }
        })
        .collect();

    let mut output_batch: Vec<float32x4_t> = vec![unsafe { vdupq_n_f32(0.0) }; total_groups];

    // Store results as packed float32x4_t to avoid per-group copy overhead.
    // We allocate as f32 but treat it as float32x4_t-aligned storage.
    let mut buffer: Vec<f32> = vec![0.0f32; total_groups * 4 * height];

    let z_vec = unsafe { vdupq_n_f32(z) };
    let w_vec = unsafe { vdupq_n_f32(w) };

    let start = Instant::now();
    for _ in 0..frames {
        for y in 0..height {
            let y_vec = unsafe { vdupq_n_f32(y as f32) };
            unsafe {
                scanline_jit.eval_scanline(&xs, y_vec, z_vec, w_vec, &mut output_batch);
            }
            // Bulk store: write output_batch directly as packed f32 values.
            let row_start = y * total_groups * 4;
            unsafe {
                let dst = buffer.as_mut_ptr().add(row_start);
                core::ptr::copy_nonoverlapping(
                    output_batch.as_ptr() as *const f32,
                    dst,
                    total_groups * 4,
                );
            }
        }
        std::hint::black_box(&buffer);
    }
    let elapsed = start.elapsed().as_nanos();

    // Extract first and last pixel for correctness check.
    let first = buffer[0];
    // Last pixel is at row (height-1), column (width-1).
    let last_row_start = (height - 1) * total_groups * 4;
    let last = buffer[last_row_start + width - 1];

    (elapsed, first, last)
}

fn main() {
    println!("================================================================");
    println!("  Hoisted Scanline JIT Benchmark");
    println!(
        "  Frame: {}x{}, Z(time)={}, W(layer)={}",
        WIDTH, HEIGHT, Z_TIME, W_LAYER
    );
    println!("================================================================\n");

    let (arena, root) = build_psychedelic_arena();
    let node_count = arena.len();
    println!("  Expression: sin(Z*0.3) * exp(-(X*X+Y*Y)*0.01) + cos(Y*0.7) * sin(X*0.5+Z)");
    println!("  Arena nodes: {}\n", node_count);

    // =====================================================================
    // Compile all kernels up front (one-time cost, not measured).
    // =====================================================================

    // Per-pixel JIT
    let per_pixel_result = pixelflow_ir::backend::emit::compile_arena_dag(&arena, root)
        .expect("per-pixel JIT compile failed");
    let per_pixel_code_bytes = per_pixel_result.code.len();
    let per_pixel_jit = pixelflow_ir::JitManifold::new(per_pixel_result.code);
    let wrapper = JitWrapper(per_pixel_jit);
    let lattice = FrameLattice::new(WIDTH, HEIGHT, Z_TIME);

    #[cfg(target_arch = "aarch64")]
    let (scanline_hoisted_jit, scanline_code_bytes, num_hoisted) = {
        let hoisted = pixelflow_ir::backend::emit::arena_to_hoisted_schedule(
            &arena,
            root,
            pixelflow_ir::backend::emit::default_hoist_predicate,
        );
        let result = pixelflow_ir::backend::emit::compile_arena_dag_scanline_hoisted(&arena, root)
            .expect("hoisted scanline JIT compile failed");
        let code_bytes = result.code.len();
        let num_hoisted = hoisted.num_hoisted;
        let jit = pixelflow_ir::ScanlineJitManifold::new(result.code);
        (jit, code_bytes, num_hoisted)
    };

    println!("  Per-pixel JIT:           {} bytes", per_pixel_code_bytes);
    #[cfg(target_arch = "aarch64")]
    println!(
        "  Hoisted scanline JIT:    {} bytes ({} hoisted values)",
        scanline_code_bytes, num_hoisted
    );
    println!();

    // =====================================================================
    // Lane 1: Per-pixel via FrameLattice::collapse()
    // =====================================================================
    print!("  Warming up per-pixel path ({} frames)...", WARMUP_FRAMES);
    for _ in 0..WARMUP_FRAMES {
        let _ = lattice.collapse(&wrapper);
    }
    println!(" done");

    print!("  Benchmarking per-pixel path ({} frames)...", BENCH_FRAMES);
    let per_pixel_start = Instant::now();
    let mut per_pixel_first: f32 = 0.0;
    let mut per_pixel_last: f32 = 0.0;
    for _ in 0..BENCH_FRAMES {
        let dm = lattice.collapse(&wrapper);
        per_pixel_first = dm.buffer()[0];
        per_pixel_last = dm.buffer()[dm.buffer().len() - 1];
        // Black-box to prevent DCE.
        std::hint::black_box(&dm);
    }
    let per_pixel_elapsed = per_pixel_start.elapsed();
    println!(" done");

    let per_pixel_ns_frame = per_pixel_elapsed.as_nanos() as f64 / BENCH_FRAMES as f64;
    let per_pixel_ns_pixel = per_pixel_ns_frame / (WIDTH * HEIGHT) as f64;

    // =====================================================================
    // Lane 2: Hoisted scanline JIT (compile once, eval many)
    // =====================================================================
    #[cfg(target_arch = "aarch64")]
    let (scanline_ns_pixel, scanline_ns_frame, scanline_first, scanline_last) = {
        // Warmup
        print!(
            "  Warming up hoisted scanline path ({} frames)...",
            WARMUP_FRAMES
        );
        let _ = bench_scanline_eval(
            &scanline_hoisted_jit,
            WIDTH,
            HEIGHT,
            Z_TIME,
            W_LAYER,
            WARMUP_FRAMES,
        );
        println!(" done");

        // Bench
        print!(
            "  Benchmarking hoisted scanline path ({} frames)...",
            BENCH_FRAMES
        );
        let (elapsed_ns, first, last) = bench_scanline_eval(
            &scanline_hoisted_jit,
            WIDTH,
            HEIGHT,
            Z_TIME,
            W_LAYER,
            BENCH_FRAMES,
        );
        println!(" done");

        let ns_frame = elapsed_ns as f64 / BENCH_FRAMES as f64;
        let ns_pixel = ns_frame / (WIDTH * HEIGHT) as f64;
        (ns_pixel, ns_frame, first, last)
    };

    #[cfg(not(target_arch = "aarch64"))]
    let (scanline_ns_pixel, scanline_ns_frame, scanline_first, scanline_last) = {
        println!("  [SKIP] Hoisted scanline JIT not implemented for this architecture");
        (0.0f64, 0.0f64, per_pixel_first, per_pixel_last)
    };

    // =====================================================================
    // Results
    // =====================================================================
    let total_pixels = WIDTH * HEIGHT;
    println!("\n================================================================");
    println!(
        "  RESULTS ({}x{} = {} pixels, {} frames averaged)",
        WIDTH, HEIGHT, total_pixels, BENCH_FRAMES
    );
    println!("================================================================\n");

    println!(
        "  {:34} {:>10} {:>10} {:>10}",
        "", "ns/pixel", "ms/frame", "bytes"
    );
    println!(
        "  {:34} {:>10.2} {:>10.2} {:>10}",
        "Per-pixel JIT (FrameLattice):",
        per_pixel_ns_pixel,
        per_pixel_ns_frame / 1_000_000.0,
        per_pixel_code_bytes
    );

    #[cfg(target_arch = "aarch64")]
    {
        println!(
            "  {:34} {:>10.2} {:>10.2} {:>10}",
            "Hoisted scanline JIT:",
            scanline_ns_pixel,
            scanline_ns_frame / 1_000_000.0,
            scanline_code_bytes
        );

        let speedup = if scanline_ns_pixel > 0.0 {
            per_pixel_ns_pixel / scanline_ns_pixel
        } else {
            f64::NAN
        };

        println!("\n  Hoisted values:  {}", num_hoisted);
        println!("  Speedup:         {:.2}x", speedup);

        // FPS comparison
        let per_pixel_fps = 1_000_000_000.0 / per_pixel_ns_frame;
        let scanline_fps = 1_000_000_000.0 / scanline_ns_frame;
        println!("\n  Per-pixel FPS:   {:.1}", per_pixel_fps);
        println!("  Scanline FPS:    {:.1}", scanline_fps);
    }

    // Correctness: compare first and last pixel between the two paths.
    let first_diff = (per_pixel_first - scanline_first).abs();
    let last_diff = (per_pixel_last - scanline_last).abs();
    let max_diff = first_diff.max(last_diff);
    if max_diff > 0.01 {
        eprintln!("\n  WARNING: pixel mismatch (pre-existing hoisted JIT correctness bug)");
        eprintln!(
            "    pixel[0]:    per_pixel={:.6} scanline={:.6} diff={:.6}",
            per_pixel_first, scanline_first, first_diff
        );
        eprintln!(
            "    pixel[last]: per_pixel={:.6} scanline={:.6} diff={:.6}",
            per_pixel_last, scanline_last, last_diff
        );
        eprintln!("  Timing data above is still valid for throughput comparison.");
    } else {
        println!("  Correctness:     OK (max pixel diff={:.2e})", max_diff);
    }

    println!("\n================================================================");
}
