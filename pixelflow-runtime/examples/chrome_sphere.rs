//! Chrome Sphere Parallel Rendering Demo
//!
//! Compares single-threaded vs parallel rasterization of the 3D chrome sphere scene.
//! Uses the mullet architecture: geometry once, colors as packed Discrete.

use pixelflow_core::combinators::At;
use pixelflow_core::jet::Jet3;
use pixelflow_core::{Discrete, Field, Manifold, ManifoldCompat};
use pixelflow_compiler::ManifoldExpr;

type Field4 = (Field, Field, Field, Field);
type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);
use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::rasterize;
use pixelflow_graphics::scene3d::{
    plane, ColorChecker, ColorReflect, ColorScreenToDir, ColorSky, ColorSurface,
};
use pixelflow_runtime::platform::ColorCube;

/// Sphere at given center with radius (local to this example).
#[derive(Clone, Copy, ManifoldExpr)]
struct SphereAt {
    center: (f32, f32, f32),
    radius: f32,
}

impl Manifold<Jet3_4> for SphereAt {
    type Output = Jet3;

    #[inline]
    fn eval(&self, p: Jet3_4) -> Jet3 {
        let (rx, ry, rz, _w) = p;
        let cx = Jet3::constant(Field::from(self.center.0));
        let cy = Jet3::constant(Field::from(self.center.1));
        let cz = Jet3::constant(Field::from(self.center.2));

        let d_dot_c = rx * cx + ry * cy + rz * cz;
        let c_sq = cx * cx + cy * cy + cz * cz;
        let r_sq = Jet3::constant(Field::from(self.radius * self.radius));
        let discriminant = d_dot_c * d_dot_c - (c_sq - r_sq);

        // Smooth fix for grazing angles
        let epsilon_sq = Jet3::constant(Field::from(0.0001));
        d_dot_c - (discriminant + epsilon_sq).sqrt()
    }
}
use std::fs::File;
use std::io::Write;
use std::time::Instant;

const W: usize = 1920;
const H: usize = 1080;

/// Remap pixel coordinates to normalized screen coordinates for ~60° FOV.
#[derive(Clone, ManifoldExpr)]
struct ColorScreenRemap<M> {
    inner: M,
    width: f32,
    height: f32,
}

impl<M: ManifoldCompat<Field, Output = Discrete>> Manifold<Field4> for ColorScreenRemap<M> {
    type Output = Discrete;

    fn eval(&self, p: Field4) -> Discrete {
        let (px, py, z, w) = p;
        let width = Field::from(self.width);
        let height = Field::from(self.height);

        let scale = Field::from(2.0) / height;
        let x = (px - width * Field::from(0.5)) * scale.clone();
        let y = (height * Field::from(0.5) - py) * scale;

        At {
            inner: &self.inner,
            x,
            y,
            z,
            w,
        }
        .collapse()
    }
}

/// Build the color scene using the mullet architecture.
/// Geometry runs once, colors flow as packed RGBA.
fn build_scene() -> impl Manifold<Output = Discrete> + Clone + Sync {
    let color_cube = ColorCube::default();
    let world = ColorSurface {
        geometry: plane(-1.0),
        material: ColorChecker::new(color_cube.clone()),
        background: ColorSky::new(color_cube),
    };

    let scene = ColorSurface {
        geometry: SphereAt {
            center: (0.0, 0.0, 4.0),
            radius: 1.0,
        },
        material: ColorReflect {
            inner: world.clone(),
        },
        background: world,
    };

    ColorScreenRemap {
        inner: ColorScreenToDir { inner: scene },
        width: W as f32,
        height: H as f32,
    }
}

fn main() {
    println!("Chrome Sphere Parallel Rendering Demo");
    println!("=====================================");
    println!(
        "Resolution: {}x{} ({:.1}M pixels)",
        W,
        H,
        (W * H) as f64 / 1_000_000.0
    );
    println!();

    // Build the scene using mullet architecture (geometry once, colors as packed Discrete)
    let scene = build_scene();

    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    println!("Available CPU threads: {}", num_cpus);
    println!();

    // Warm-up run
    {
        let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
        rasterize(&scene, &mut frame, 1);
    }

    // Single-threaded benchmark
    let single_time = {
        let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
        let start = Instant::now();
        rasterize(&scene, &mut frame, 1);
        let elapsed = start.elapsed();

        let mpps = (W * H) as f64 / elapsed.as_secs_f64() / 1_000_000.0;
        let fps = 1.0 / elapsed.as_secs_f64();
        println!(
            "Single-threaded: {:>7.2}ms ({:>5.1} Mpix/s, {:>5.1} FPS)",
            elapsed.as_secs_f64() * 1000.0,
            mpps,
            fps
        );

        // Save the image
        let path = std::env::temp_dir().join("chrome_sphere_single.ppm");
        let mut file = File::create(&path).unwrap();
        writeln!(file, "P6\n{} {}\n255", W, H).unwrap();
        for p in &frame.data {
            file.write_all(&[p.r(), p.g(), p.b()]).unwrap();
        }
        println!("  Saved: {}", path.display());

        elapsed
    };

    println!();

    // Parallel benchmarks with different thread counts
    for threads in [2, 4, 8, num_cpus].iter().filter(|&&t| t <= num_cpus) {
        let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);

        let start = Instant::now();
        rasterize(&scene, &mut frame, *threads);
        let elapsed = start.elapsed();

        let mpps = (W * H) as f64 / elapsed.as_secs_f64() / 1_000_000.0;
        let fps = 1.0 / elapsed.as_secs_f64();
        let speedup = single_time.as_secs_f64() / elapsed.as_secs_f64();

        println!(
            "{:>2}-threaded:      {:>7.2}ms ({:>5.1} Mpix/s, {:>5.1} FPS) - {:.2}x speedup",
            threads,
            elapsed.as_secs_f64() * 1000.0,
            mpps,
            fps,
            speedup
        );

        if *threads == num_cpus {
            // Save the parallel result
            let path = std::env::temp_dir().join("chrome_sphere_parallel.ppm");
            let mut file = File::create(&path).unwrap();
            writeln!(file, "P6\n{} {}\n255", W, H).unwrap();
            for p in &frame.data {
                file.write_all(&[p.r(), p.g(), p.b()]).unwrap();
            }
            println!("  Saved: {}", path.display());
        }
    }

    println!();
    println!("Done! Images saved to /tmp/");
}
