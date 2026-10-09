//! A frame past what NEON addresses directly, compiled and executed: the
//! spill area alone is past the reach of `ldr q`'s scaled 12-bit displacement
//! (`4095 · 16` bytes), so on NEON every slot beyond it is addressed through a
//! register, and the values must still be the kernel's own.
//!
//! `a_deep_spill_frame_compiles_correctly` forces more than 128 bytes; this is
//! the value test of a frame past 64 KiB, run on NEON under qemu and natively
//! on aarch64 hosts, and on the host's x86 tier elsewhere, where the same
//! kernel's spill area is wider still.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_codegen::emit::compile;
use pixelflow_ir::LatticeShape;

mod rows {
    include!("support/deep_frame.rs");
}

/// The reach of a 12-bit displacement scaled by a 16-byte vector.
const REACH: u32 = 64 * 1024;
const WIDTH: usize = 11;
const ROWS: usize = 3;
const ORIGIN: [f32; 2] = [2.0, 5.0];
/// A balanced `f32` sum of `DEEP_FRAME_TERMS` positive terms, squared, errs
/// about `1e-6` relative, and one misaddressed far slot moves it by `4e-4`.
const TOLERANCE: f64 = 1e-5;

/// `(Σ tₖ)²`, `tₖ = x·c_{k mod s} + y·d_{k div s}`, in `f64`.
fn reference(terms: usize, x: f32, y: f32) -> f64 {
    let side = terms.isqrt() + 1;
    let sum: f64 = (0..terms)
        .map(|k| {
            let c = 0.25 + (k % side) as f32 * 0.125;
            let d = 0.5 + (k / side) as f32 * 0.0625;
            f64::from(x) * f64::from(c) + f64::from(y) * f64::from(d)
        })
        .sum();
    sum * sum
}

#[test]
fn a_spill_area_past_64_kib_holds_the_kernels_values() {
    let (arena, root) = rows::deep_frame(rows::DEEP_FRAME_TERMS);
    let shape = LatticeShape::new([WIDTH as u32, ROWS as u32]);
    let compiled = compile(&arena, root, shape).expect("a deep frame compiles");
    assert!(
        compiled.frame_bytes > u64::from(REACH),
        "the fixture no longer forces a frame past reach (frame_bytes = {})",
        compiled.frame_bytes
    );

    let mut out = vec![f32::NAN; ROWS * WIDTH];
    let uniforms: [f32; 0] = [];
    let ctx = [uniforms.as_ptr(), ORIGIN.as_ptr()];
    unsafe {
        compiled.code.call(ctx.as_ptr(), out.as_mut_ptr(), WIDTH);
    }
    for row in 0..ROWS {
        for col in 0..WIDTH {
            let (x, y) = (ORIGIN[0] + col as f32, ORIGIN[1] + row as f32);
            let got = f64::from(out[row * WIDTH + col]);
            let want = reference(rows::DEEP_FRAME_TERMS, x, y);
            assert!(
                (got - want).abs() <= TOLERANCE * want.abs(),
                "row {row} col {col} (x={x}, y={y}): got {got}, want {want}"
            );
        }
    }
}
