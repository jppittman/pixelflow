//! The S2 gate: the psychedelic shader as a `Scene::Packed`.
//!
//! `cargo run --release -p pixelflow-runtime --example bench_scene_psychedelic`
//!
//! Three channels plus an opaque alpha as four `Kernel`s compiled at the
//! frame's lattice shape with the pixel pack inside the kernel, collapsed one
//! call per stripe.
//!
//! It used to time this against `Scene::Surface` — a monomorphized `kernel!`
//! channel per colour, packed by `ColorCube` and evaluated once per SIMD batch
//! by the work-stealing rasterizer — and compare the two lanes pixel by pixel
//! before timing either. S4a deleted that lane and S4b-2 the tier it belonged
//! to, so there is one lane and this is the whole-frame measurement of it; the
//! cross-tier comparisons are in the plan's landing blocks
//! (`docs/plans/2026-09-06-kernel-with-a-lattice.md`).

use std::time::Instant;

use pixelflow_core::{Kernel, Uniform};
use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::scene::{compile_packed_for, Scene};
use pixelflow_graphics::render::Frame;
use pixelflow_graphics::scene3d::Rgba;

const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;

// ── the shader, as four channel kernels ──

/// The `y` weight is the only thing that separates the three channels.
fn psych_channel(y_weight: f32, clock: Uniform) -> Kernel {
    let k = Kernel::constant;
    let scale = 2.0 / 1080.0;
    let x = Kernel::x().sub(&k(960.0)).mul(&k(scale));
    let y = k(540.0).sub(&Kernel::y()).mul(&k(scale));
    let time = clock.kernel().add(&k(1.3));
    let r_sq = x.mul(&x).add(&y.mul(&y));
    let radial = r_sq.sub(&k(0.7)).abs();
    let swirl_scale = k(1.0).sub(&radial).mul(&k(5.0));
    let vx = x.mul(&swirl_scale);
    let vy = y.mul(&swirl_scale);
    let phase = time.mul(&k(0.5));
    let sin_w03 = time.mul(&k(0.3)).sin();
    let sin_w20 = time.mul(&k(2.0)).sin();
    let vxp = vx.add(&phase);
    let swirl = vxp
        .sin()
        .add(&k(1.0))
        .mul(&vxp.sub(&vy.add(&phase.mul(&k(0.7)))).abs())
        .mul(&k(0.2))
        .add(&k(0.001));
    let pulse = k(1.0).add(&sin_w20.mul(&k(0.1)));
    let radial_factor = radial.mul(&k(-4.0)).mul(&pulse).exp();
    let raw = y
        .mul(&k(y_weight))
        .add(&sin_w03.mul(&k(0.2)))
        .exp()
        .mul(&radial_factor)
        .div(&swirl);
    raw.div(&raw.abs().add(&k(1.0))).add(&k(1.0)).mul(&k(0.5))
}

fn packed_scene() -> Scene {
    // One clock, read by all three channels: one argument in the fused
    // program, and the bench holds it at its default.
    let clock = Uniform::new(0.0);
    let channels = [
        psych_channel(1.0, clock),
        psych_channel(-1.0, clock),
        psych_channel(-2.0, clock),
        Kernel::constant(1.0),
    ];
    let color = Rgba::from(channels);
    let compiled = Instant::now();
    let program = compile_packed_for::<Rgba8>(&color, [WIDTH as u32, HEIGHT as u32]);
    let compile_time = compiled.elapsed();
    // The compile time, because a scene's compile is a budget (~250 ms, plan
    // §5.3) that an extraction change must report on both production shapes.
    println!("  packed kernel: compiled in {compile_time:?}");
    let block = program.block();
    Scene::Packed(program.bind_with(&[], &block))
}

/// Median ns/pixel over `runs` frames, after `warm` warm-up frames.
fn time_ns_per_px(scene: &Scene, threads: usize, warm: usize, runs: usize) -> f64 {
    let mut frame = Frame::<Rgba8>::new(WIDTH as u32, HEIGHT as u32);
    for _ in 0..warm {
        scene.render(&mut frame, threads);
        std::hint::black_box(&frame.data[0]);
    }
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        scene.render(&mut frame, threads);
        samples.push(t.elapsed().as_nanos());
        std::hint::black_box(&frame.data[0]);
    }
    samples.sort_unstable();
    samples[samples.len() / 2] as f64 / (WIDTH * HEIGHT) as f64
}

fn main() {
    const WARM: usize = 3;
    const RUNS: usize = 9;
    /// `PerformanceConfig::default().render_threads`.
    const RUNTIME_THREADS: usize = 12;

    let packed = packed_scene();

    println!(
        "psychedelic shader as a Scene, {WIDTH}x{HEIGHT}, median of {RUNS} frames (ns/pixel)\n"
    );

    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("  threads   packed (Scene::Packed)");
    for threads in [1, cores, RUNTIME_THREADS] {
        let p = time_ns_per_px(&packed, threads, WARM, RUNS);
        println!("  {threads:>7}   {p:>20.2}");
    }
}
