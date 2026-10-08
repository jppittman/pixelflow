//! Chrome sphere, rendered as a packed scene at 1920x1080.
//!
//! `cargo run --release -p pixelflow-runtime --example chrome_sphere`
//!
//! The scene is four channel kernels — a sphere's quadratic, a plane's
//! division, a Householder reflection, a checkerboard filtered over one pixel
//! — compiled once at the frame's shape with the pixel pack inside the
//! kernel, then collapsed one call per stripe across threads.
//!
//! For the same scene measured against the legacy `Manifold<Jet3>` tier, see
//! `bench_scene_chrome.rs`.

use std::fs::File;
use std::io::Write;
use std::time::Instant;

use pixelflow_core::Kernel;
use pixelflow_graphics::render::color::PlatformPixel;
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::scene::{compile_platform_packed, Scene};
use pixelflow_graphics::scene3d::{checker, sky, Plane, Ray, Rgba, Sphere};

const W: usize = 1920;
const H: usize = 1080;

fn k(v: f32) -> Kernel {
    Kernel::constant(v)
}

/// The checker floor under the sky, seen along `ray`.
fn world(ray: &Ray) -> Rgba {
    let floor = Plane::at_height(k(-1.0)).hit(ray);
    floor.select(
        &checker(&floor.point()[0], &floor.point()[2], &floor.footprint()),
        &sky(ray),
    )
}

/// A chrome sphere at (0, 0, 4) reflecting that world.
fn scene() -> (Scene, std::time::Duration) {
    let ray = Ray::through_screen(W as f32, H as f32);
    let sphere = Sphere::new([k(0.0), k(0.0), k(4.0)], k(1.0)).hit(&ray);
    let mirrored = ray.reflected(sphere.normal());
    let color = sphere.select(&world(&mirrored), &world(&ray));

    let start = Instant::now();
    let program = compile_platform_packed(&color, [W as u32, H as u32]);
    let compile = start.elapsed();
    (Scene::Packed(program.bind(&[])), compile)
}

fn save(name: &str, frame: &Frame<PlatformPixel>) {
    let path = std::env::temp_dir().join(name);
    let mut file = File::create(&path).expect("cannot write the image");
    writeln!(file, "P6\n{W} {H}\n255").expect("cannot write the header");
    for p in &frame.data {
        file.write_all(&[p.r(), p.g(), p.b()])
            .expect("cannot write a pixel");
    }
    println!("  Saved: {}", path.display());
}

fn main() {
    println!("Chrome Sphere (packed scene)");
    println!("============================");
    println!(
        "Resolution: {W}x{H} ({:.1}M pixels)",
        (W * H) as f64 / 1_000_000.0
    );

    let (scene, compile) = scene();
    println!("Compiled once in {compile:?}\n");

    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("Available CPU threads: {cores}\n");

    let mut frame = Frame::<PlatformPixel>::new(W as u32, H as u32);
    scene.render(&mut frame, 1);

    let single = {
        let start = Instant::now();
        scene.render(&mut frame, 1);
        let elapsed = start.elapsed();
        println!(
            "Single-threaded: {:>7.2}ms ({:>5.1} Mpix/s, {:>5.1} FPS)",
            elapsed.as_secs_f64() * 1000.0,
            (W * H) as f64 / elapsed.as_secs_f64() / 1_000_000.0,
            1.0 / elapsed.as_secs_f64()
        );
        save("chrome_sphere_single.ppm", &frame);
        elapsed
    };
    println!();

    for threads in [2, 4, 8, cores].iter().filter(|&&t| t <= cores) {
        let start = Instant::now();
        scene.render(&mut frame, *threads);
        let elapsed = start.elapsed();
        println!(
            "{threads:>2}-threaded:      {:>7.2}ms ({:>5.1} Mpix/s, {:>5.1} FPS) - {:.2}x speedup",
            elapsed.as_secs_f64() * 1000.0,
            (W * H) as f64 / elapsed.as_secs_f64() / 1_000_000.0,
            1.0 / elapsed.as_secs_f64(),
            single.as_secs_f64() / elapsed.as_secs_f64()
        );
        if *threads == cores {
            save("chrome_sphere_parallel.ppm", &frame);
        }
    }
}
