//! Headless proof that a colored (`Discrete`) scene renders through the real
//! `rasterize()` path with **JIT-compiled** channel kernels, matching the
//! combinator render.
//!
//! The rasterizer is polymorphic over `dyn Manifold<Output = Discrete>`, and a
//! color scene is `ColorCube::default().at(red, green, blue, alpha)` where each
//! channel is just a `Field` manifold. So feeding `kernel_jit!`-compiled
//! channels into the same `ColorCube` yields a JIT-backed scene that renders
//! through the unchanged engine/rasterizer — no per-pixel `Manifold::eval`.
//!
//! This is the "the actual scene renders via JIT" milestone, verified headless:
//! the JIT frame must match the combinator frame to within the JIT's
//! polynomial-approximation accuracy for `sin`/`sqrt`.

use pixelflow_compiler::{kernel, kernel_jit};
use pixelflow_core::{X, Y};
use pixelflow_core::ManifoldExt;
use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::rasterize;
use pixelflow_runtime::platform::ColorCube;

const W: u32 = 64;
const H: u32 = 64;

#[test]
fn colored_scene_renders_via_jit_matching_combinators() {
    // Channel expressions are pure functions of the pixel coordinates (X, Y),
    // so each compiles once — exactly the compile-on-resize model, no per-frame
    // recompilation.
    //   red   = X * 0.015            (horizontal ramp)
    //   green = Y * 0.015            (vertical ramp)
    //   blue  = sqrt(X*X + Y*Y) * 0.011  (radial)
    //
    // These use only arithmetic and sqrt, which the combinator and JIT paths
    // both evaluate accurately, so the combinator render is a faithful
    // reference. (Transcendentals like `sin` are validated numerically against
    // the analytic result elsewhere; here the goal is the render-path wiring,
    // and the combinator's own low-degree `sin` is too coarse to serve as a
    // reference.)

    // Combinator scene: channels evaluated as combinator trees (the status quo).
    let combo = ColorCube::default().at(
        kernel!(|| X * 0.015)(),
        kernel!(|| Y * 0.015)(),
        kernel!(|| (X * X + Y * Y).sqrt() * 0.011)(),
        1.0,
    );

    // JIT scene: same expressions, each channel compiled to native machine code.
    let jit = ColorCube::default().at(
        kernel_jit!(|| X * 0.015),
        kernel_jit!(|| Y * 0.015),
        kernel_jit!(|| (X * X + Y * Y).sqrt() * 0.011),
        1.0,
    );

    let mut frame_combo = Frame::<Rgba8>::new(W, H);
    let mut frame_jit = Frame::<Rgba8>::new(W, H);

    // The real render path: work-stealing parallel rasterizer over SIMD lanes.
    rasterize(&combo, &mut frame_combo, 1);
    rasterize(&jit, &mut frame_jit, 1);

    // Compare every pixel. The JIT lowers sin/sqrt to Chebyshev/rsqrt
    // approximations, so a few 8-bit quantization levels of drift are expected;
    // a real failure (wrong scene, wrong wiring) would be off by tens or more.
    let mut max_diff = 0u8;
    let mut worst = (0usize, 0u8, 0u8);
    for (i, (a, b)) in frame_combo.data.iter().zip(frame_jit.data.iter()).enumerate() {
        for (ca, cb) in [(a.r(), b.r()), (a.g(), b.g()), (a.b(), b.b())] {
            let d = ca.abs_diff(cb);
            if d > max_diff {
                max_diff = d;
                worst = (i, ca, cb);
            }
        }
    }

    eprintln!(
        "[jit-render] {W}x{H} colored scene rendered via JIT through rasterize(); \
         max 8-bit channel diff vs combinators = {max_diff} (worst pixel {} combo={} jit={})",
        worst.0, worst.1, worst.2
    );

    assert!(
        max_diff <= 2,
        "JIT render diverged from combinator render by {max_diff} levels (>2)"
    );

    // Sanity: the scene is non-trivial (not all one color), so the test is real.
    let first = frame_jit.data[0];
    let last = frame_jit.data[(W * H - 1) as usize];
    assert!(
        first != last,
        "scene should vary across the frame; got uniform output"
    );
}
