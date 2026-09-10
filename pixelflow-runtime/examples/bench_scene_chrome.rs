//! The S3/S3b gate: the chrome sphere as a `Scene::Packed`.
//!
//! `cargo run --release -p pixelflow-runtime --example bench_scene_chrome`
//!
//! The scene is `scene3d`'s constructors — four channel `Kernel`s compiled
//! once at the frame's lattice shape with the pixel pack inside the kernel,
//! collapsed one call per stripe. What it reports is the compile cost (arena
//! nodes, build and compile time, emitted bytes) and ns/pixel at one, `cores`
//! and twelve threads.
//!
//! It used to time this against `Scene::Surface` — the `Manifold<Jet3>`
//! combinators of `scene3d_surface`, evaluated once per SIMD batch through a
//! `dyn` boundary — and the two lanes' agreement rows were how S3 established
//! that the packed lane draws the same picture (matte, mirror and chrome, all
//! three explained in
//! [the plan](../../docs/plans/2026-09-06-kernel-with-a-lattice.md)). S4a
//! deleted that lane, so the comparison lives in the plan's landing blocks
//! and this example times what remains. `scene3d`'s own tests are what now
//! pin the picture — `a_reflection_off_a_sphere_is_a_unit_ray`,
//! `the_floor_footprint_is_one_pixel_wide` — and the goldens pin the pixels.

use std::time::Instant;

use pixelflow_core::Kernel;
use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::scene::{compile_packed_for, Scene};
use pixelflow_graphics::render::Frame;
use pixelflow_graphics::scene3d::{checker, sky, Hit, Plane, Ray, Rgba, Sphere};

const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;

/// Sphere centre and radius.
const CENTER: (f32, f32, f32) = (0.0, 0.0, 4.0);
const RADIUS: f32 = 1.0;
/// The checker floor's height.
const FLOOR: f32 = -1.0;

fn k(v: f32) -> Kernel {
    Kernel::constant(v)
}

fn ray() -> Ray {
    Ray::through_screen(WIDTH as f32, HEIGHT as f32)
}

fn sphere(ray: &Ray) -> Hit {
    Sphere::new([k(CENTER.0), k(CENTER.1), k(CENTER.2)], k(RADIUS)).hit(ray)
}

/// The checker floor under the sky, seen along `ray`, filtered over one pixel
/// — which is what a screen-space footprint is.
fn world(ray: &Ray) -> Rgba {
    let floor = Plane::at_height(k(FLOOR)).hit(ray);
    floor.select(
        &checker(&floor.point()[0], &floor.point()[2], &floor.footprint()),
        &sky(ray),
    )
}

fn chrome() -> Rgba {
    let ray = ray();
    let sphere = sphere(&ray);
    let mirrored = ray.reflected(sphere.normal());
    sphere.select(&world(&mirrored), &world(&ray))
}

fn packed_scene(color: &Rgba) -> Scene {
    Scene::Packed(compile_packed_for::<Rgba8>(color, [WIDTH as u32, HEIGHT as u32]).bind(&[]))
}

/// What fraction of the frame the sphere covers — the area whose reflected
/// world the select's guard gets to skip in every other batch.
fn sphere_coverage() -> f64 {
    let ray = ray();
    let mask = sphere(&ray).mask().select(&k(1.0), &k(0.0));
    let scene = packed_scene(&Rgba::from([mask.clone(), mask.clone(), mask, k(1.0)]));
    let mut frame = Frame::<Rgba8>::new(WIDTH as u32, HEIGHT as u32);
    scene.render(&mut frame, 4);
    let hits = frame.data.iter().filter(|p| p.r() > 0).count();
    hits as f64 / (WIDTH * HEIGHT) as f64
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

    // Compile cost: building the scene and JIT-compiling it, once.
    let built = Instant::now();
    let color = chrome();
    let build_time = built.elapsed();
    let nodes: usize = color.fold(
        &|channels: &[Kernel; 4]| {
            channels.iter().map(|c| c.term().root().node_count()).sum()
        },
        &|mask: &Kernel, if_true: usize, if_false: usize| {
            mask.term().root().node_count() + if_true + if_false
        },
    );
    let compiled = Instant::now();
    let program = compile_packed_for::<Rgba8>(&color, [WIDTH as u32, HEIGHT as u32]);
    let compile_time = compiled.elapsed();
    let code = program.code_bytes().len();
    let packed = Scene::Packed(program.bind(&[]));

    println!("chrome sphere as a Scene, {WIDTH}x{HEIGHT}, median of {RUNS} frames (ns/pixel)\n");
    println!(
        "  compile: {nodes} arena nodes over four channels, built in {build_time:?}, \
         compiled in {compile_time:?} to {code} bytes of code"
    );
    println!(
        "  the sphere covers {:.2}% of the frame\n",
        sphere_coverage() * 100.0
    );

    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("  threads   packed (Scene::Packed)");
    for threads in [1, cores, RUNTIME_THREADS] {
        let p = time_ns_per_px(&packed, threads, WARM, RUNS);
        println!("  {threads:>7}   {p:>20.2}");
    }
}
