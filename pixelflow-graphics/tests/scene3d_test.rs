//! Test: Analytic 3D rendering with three-layer architecture
//!
//! Three Layers:
//! 1. Geometry: Returns `t` (Jet3) - UnitSphere, PlaneGeometry
//! 2. Surface: Warps `P = ray * t` - creates tangent frame via chain rule
//! 3. Material: Reconstructs normal from derivatives - Reflect, Checker, Sky

use pixelflow_core::combinators::At;
use pixelflow_core::jet::Jet3;
use pixelflow_core::{Discrete, Field, Manifold, ManifoldCompat, ManifoldExt};
use pixelflow_compiler::ManifoldExpr;

type Field4 = (Field, Field, Field, Field);
type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);
use pixelflow_graphics::render::color::{Rgba8, RgbaColorCube};
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::rasterize;
use pixelflow_graphics::scene3d::{
    Checker, ColorChecker, ColorReflect, ColorScreenToDir, ColorSky, ColorSurface, plane,
    Reflect, ScreenToDir, sky, Surface,
};
use std::fs::File;
use std::io::Write;

/// Sphere at given center with radius (local to this test).
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

        let epsilon_sq = Jet3::constant(Field::from(0.0001));
        d_dot_c - (discriminant + epsilon_sq).sqrt()
    }
}

/// Convert grayscale Field to Discrete RGBA
struct GrayToRgba<M> {
    inner: M,
}

impl<M: ManifoldCompat<Field, Output = Field>> Manifold<Field4> for GrayToRgba<M> {
    type Output = Discrete;

    fn eval(&self, p: Field4) -> Discrete {
        let (x, y, z, w) = p;
        let gray = self.inner.eval_raw(x, y, z, w);
        Discrete::pack(gray, gray, gray, Field::from(1.0))
    }
}

/// Remap pixel coordinates to normalized screen coordinates for ~60° FOV.
/// Transforms [0, width] × [0, height] → [-aspect, aspect] × [-1, 1]
/// where the values represent tan(angle) from optical axis.
struct ScreenRemap<M> {
    inner: M,
    width: f32,
    height: f32,
}

impl<M: ManifoldCompat<Field, Output = Field> + ManifoldExt> Manifold<Field4> for ScreenRemap<M> {
    type Output = Field;

    fn eval(&self, p: Field4) -> Field {
        let (px, py, z, w) = p;
        let width = Field::from(self.width);
        let height = Field::from(self.height);

        // Scale so that screen Y range is [-1, 1]
        // This gives ~53° vertical FOV (since atan(1) = 45°, the full range is 90°)
        let scale = Field::from(2.0) / height;
        let x = (px - width * Field::from(0.5)) * scale.clone();
        let y = (height * Field::from(0.5) - py) * scale; // Flip Y

        self.inner.eval_at(x, y, z, w)
    }
}

// ============================================================================
// TESTS
// ============================================================================

/// Test: Chrome sphere at z=4 reflecting floor and sky.
/// Uses the new three-layer architecture:
/// - ScreenToDir: seeds jets with screen derivatives, normalizes direction
/// - Surface<SphereAt, Reflect<world>, world>: sphere reflecting world
/// - world = Surface<plane, Checker, Sky>: floor + sky
#[test]
fn test_chrome_unit_sphere() {
    const W: usize = 400;
    const H: usize = 300;

    // World = floor + sky
    // PlaneGeometry at y=-1: returns t = -1 / ry (negative plane)
    let world = Surface {
        geometry: plane(-1.0),
        material: Checker,
        background: sky(),
    };

    // Scene = chrome sphere at (0, 0, 4) + world background
    // SphereAt solves quadratic for intersection distance
    let scene = Surface {
        geometry: SphereAt {
            center: (0.0, 0.0, 4.0),
            radius: 1.0,
        },
        material: Reflect { inner: world.clone() },
        background: world,
    };

    // ScreenToDir: pixel coords → direction jets with derivatives
    let screen = ScreenRemap {
        inner: ScreenToDir { inner: scene },
        width: W as f32,
        height: H as f32,
    };

    // Wrap in GrayToRgba for rendering
    let renderable = GrayToRgba { inner: screen };

    // Render
    let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
    rasterize(&renderable, &mut frame, 1);

    // Save PPM
    let path = std::env::temp_dir().join("pixelflow_chrome_unit_sphere.ppm");
    let mut file = File::create(&path).unwrap();
    writeln!(file, "P6\n{} {}\n255", W, H).unwrap();
    for p in &frame.data {
        file.write_all(&[p.r(), p.g(), p.b()]).unwrap();
    }
    println!("Saved: {}", path.display());

    // Debug: print some pixel values
    let center = &frame.data[(H / 2) * W + (W / 2)];
    let bottom_sphere = &frame.data[(H * 5 / 8) * W + (W / 2)]; // Lower part
    let top_sphere = &frame.data[(H * 3 / 8) * W + (W / 2)]; // Upper part
    let corner = &frame.data[0]; // Top-left (sky)

    println!("Chrome center: r={}", center.r());
    println!("Chrome bottom: r={}", bottom_sphere.r());
    println!("Chrome top: r={}", top_sphere.r());
    println!("Corner (sky): r={}", corner.r());

    // Sanity checks
    assert!(
        center.r() > 10,
        "Center should not be black: r={}",
        center.r()
    );
    // Sky gradient goes from 0.1 (dark) to 0.9 (bright) = 25 to 229
    assert!(
        corner.r() > 20,
        "Corner should be sky (not black): r={}",
        corner.r()
    );
}

/// Test: Just the sky (no geometry)
#[test]
fn test_sky_only() {
    const W: usize = 200;
    const H: usize = 150;

    // Sky as the only "scene" - wraps it in a dummy Surface that always misses
    struct SkyOnly;

    impl Manifold<Jet3_4> for SkyOnly {
        type Output = Field;

        fn eval(&self, p: Jet3_4) -> Field {
            let (_x, y, _z, _w) = p;
            // Same as Sky: gradient based on Y direction
            let t = (y.val * Field::from(0.5) + Field::from(0.5))
                .max(Field::from(0.0))
                .min(Field::from(1.0));
            (Field::from(0.1) + t * Field::from(0.8)).constant()
        }
    }

    let screen = ScreenRemap {
        inner: ScreenToDir { inner: SkyOnly },
        width: W as f32,
        height: H as f32,
    };

    let renderable = GrayToRgba { inner: screen };

    let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
    rasterize(&renderable, &mut frame, 1);

    // Save
    let path = std::env::temp_dir().join("pixelflow_sky_only.ppm");
    let mut file = File::create(&path).unwrap();
    writeln!(file, "P6\n{} {}\n255", W, H).unwrap();
    for p in &frame.data {
        file.write_all(&[p.r(), p.g(), p.b()]).unwrap();
    }
    println!("Saved: {}", path.display());

    // Top should be brighter than bottom (gradient)
    let top = &frame.data[(H / 4) * W + (W / 2)];
    let bottom = &frame.data[(3 * H / 4) * W + (W / 2)];
    println!("Sky top: r={}", top.r());
    println!("Sky bottom: r={}", bottom.r());

    // Top looks "up" (positive y direction), should be brighter
    assert!(top.r() > bottom.r(), "Sky should be brighter at top");
}

/// Test: Floor only (plane with checker pattern)
#[test]
fn test_floor_only() {
    const W: usize = 400;
    const H: usize = 300;

    // Just floor + sky (no sphere)
    let scene = Surface {
        geometry: plane(-1.0),
        material: Checker,
        background: sky(),
    };

    let screen = ScreenRemap {
        inner: ScreenToDir { inner: scene },
        width: W as f32,
        height: H as f32,
    };

    let renderable = GrayToRgba { inner: screen };

    let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
    rasterize(&renderable, &mut frame, 1);

    // Save
    let path = std::env::temp_dir().join("pixelflow_floor_only.ppm");
    let mut file = File::create(&path).unwrap();
    writeln!(file, "P6\n{} {}\n255", W, H).unwrap();
    for p in &frame.data {
        file.write_all(&[p.r(), p.g(), p.b()]).unwrap();
    }
    println!("Saved: {}", path.display());

    // Bottom half should hit floor, top half should hit sky
    let floor_pixel = &frame.data[(3 * H / 4) * W + (W / 2)];
    let sky_pixel = &frame.data[(H / 4) * W + (W / 2)];
    println!("Floor: r={}", floor_pixel.r());
    println!("Sky: r={}", sky_pixel.r());

    // Floor should have checkerboard values (either light or dark)
    assert!(
        floor_pixel.r() < 80 || floor_pixel.r() > 180,
        "Floor should be checker (dark or light): r={}",
        floor_pixel.r()
    );
}

/// Test: Color chrome sphere with blue sky (MULLET ARCHITECTURE)
/// Geometry runs ONCE, colors flow as packed RGBA. 3x speedup!
#[test]
fn test_color_chrome_sphere() {
    const W: usize = 1920;
    const H: usize = 1080;

    // World = floor + sky (using Color* types that output Discrete)
    let world = ColorSurface {
        geometry: plane(-1.0),
        material: ColorChecker::<RgbaColorCube>::default(),
        background: ColorSky::<RgbaColorCube>::default(),
    };

    // Scene = chrome sphere reflecting world
    let scene = ColorSurface {
        geometry: SphereAt {
            center: (0.0, 0.0, 4.0),
            radius: 1.0,
        },
        material: ColorReflect { inner: world.clone() },
        background: world,
    };

    // ScreenRemap but for Discrete output
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

    let renderable = ColorScreenRemap {
        inner: ColorScreenToDir { inner: scene },
        width: W as f32,
        height: H as f32,
    };

    // Render
    let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
    let start = std::time::Instant::now();
    rasterize(&renderable, &mut frame, 1);
    let elapsed = start.elapsed();
    let mpps = (W * H) as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    println!("Color render (mullet): {:?} ({:.2} Mpix/s)", elapsed, mpps);

    // Save PPM
    let path = std::env::temp_dir().join("pixelflow_color_chrome.ppm");
    let mut file = File::create(&path).unwrap();
    writeln!(file, "P6\n{} {}\n255", W, H).unwrap();
    for p in &frame.data {
        file.write_all(&[p.r(), p.g(), p.b()]).unwrap();
    }
    println!("Saved: {}", path.display());

    // Debug pixels
    let center = &frame.data[(H / 2) * W + (W / 2)];
    let sky = &frame.data[0];
    println!("Center: r={} g={} b={}", center.r(), center.g(), center.b());
    println!("Sky: r={} g={} b={}", sky.r(), sky.g(), sky.b());

    // Sky should be blue-ish (B > R)
    assert!(
        sky.b() > sky.r(),
        "Sky should be blue: r={} b={}",
        sky.r(),
        sky.b()
    );
}

/// Test: Compare 3-channel vs mullet rendering to ensure they match.
/// This verifies the mullet architecture produces identical results.
#[test]
fn test_mullet_vs_3channel_comparison() {
    const W: usize = 200;
    const H: usize = 150;

    // ============================================================
    // OLD APPROACH: Run geometry 3 times (once per channel)
    // ============================================================

    /// Per-channel sky (inline version since we deleted BlueSky)
    #[derive(Clone, Copy, ManifoldExpr)]
    struct ChannelSky {
        channel: u8,
    }
    impl Manifold<Jet3_4> for ChannelSky {
        type Output = Field;
        fn eval(&self, p: Jet3_4) -> Field {
            let (_x, y, _z, _w) = p;
            let t = y.val * Field::from(0.5) + Field::from(0.5);
            let t = t.max(Field::from(0.0)).min(Field::from(1.0)).constant();
            match self.channel {
                0 => (Field::from(0.7) - t * Field::from(0.5)).constant(),
                1 => (Field::from(0.85) - t * Field::from(0.45)).constant(),
                _ => (Field::from(1.0) - t * Field::from(0.2)).constant(),
            }
        }
    }

    /// Per-channel checker (inline version)
    #[derive(Clone, Copy, ManifoldExpr)]
    struct ChannelChecker {
        channel: u8,
    }
    impl Manifold<Jet3_4> for ChannelChecker {
        type Output = Field;
        fn eval(&self, p: Jet3_4) -> Field {
            let (x, _y, z, _w) = p;
            let cell_x = x.val.floor().constant();
            let cell_z = z.val.floor().constant();
            let sum = (cell_x + cell_z).constant();
            let half = (sum * Field::from(0.5)).constant();
            let fract_half = (half.clone() - half.floor()).constant();
            let is_even = fract_half.abs().lt(Field::from(0.25));

            let (a, b) = match self.channel {
                0 => (0.95, 0.2),
                1 => (0.9, 0.25),
                _ => (0.8, 0.3),
            };

            let color_a = Field::from(a);
            let color_b = Field::from(b);
            let base_color = is_even.clone().select(color_a, color_b);

            let fx = (x.val - cell_x).constant();
            let fz = (z.val - cell_z).constant();
            let dx_edge = (fx - Field::from(0.5)).abs();
            let dz_edge = (fz - Field::from(0.5)).abs();
            let dist_to_edge = (Field::from(0.5) - dx_edge).min(Field::from(0.5) - dz_edge);

            let grad_x = (x.dx * x.dx + x.dy * x.dy + x.dz * x.dz).sqrt().constant();
            let grad_z = (z.dx * z.dx + z.dy * z.dy + z.dz * z.dz).sqrt().constant();
            let pixel_size = (grad_x.max(grad_z) + Field::from(0.001)).constant();

            let coverage = (dist_to_edge / pixel_size)
                .min(Field::from(1.0))
                .max(Field::from(0.0));
            let neighbor_color = is_even.select(color_b, color_a);
            (base_color * coverage.clone() + neighbor_color * (Field::from(1.0) - coverage))
                .constant()
        }
    }

    fn build_channel_scene(channel: u8) -> impl Manifold<Output = Field> {
        let world = Surface {
            geometry: plane(-1.0),
            material: ChannelChecker { channel },
            background: ChannelSky { channel },
        };

        ScreenRemap {
            inner: ScreenToDir {
                inner: Surface {
                    geometry: SphereAt {
                        center: (0.0, 0.0, 4.0),
                        radius: 1.0,
                    },
                    material: Reflect { inner: world.clone() },
                    background: world,
                },
            },
            width: W as f32,
            height: H as f32,
        }
    }

    struct ThreeChannelRenderer<R, G, B> {
        r: R,
        g: G,
        b: B,
    }
    impl<R, G, B> Manifold<Field4> for ThreeChannelRenderer<R, G, B>
    where
        R: ManifoldCompat<Field, Output = Field>,
        G: ManifoldCompat<Field, Output = Field>,
        B: ManifoldCompat<Field, Output = Field>,
    {
        type Output = Discrete;
        fn eval(&self, p: Field4) -> Discrete {
            let (x, y, z, w) = p;
            let r = self.r.eval_raw(x, y, z, w);
            let g = self.g.eval_raw(x, y, z, w);
            let b = self.b.eval_raw(x, y, z, w);
            Discrete::pack(r, g, b, Field::from(1.0))
        }
    }

    let old_renderer = ThreeChannelRenderer {
        r: build_channel_scene(0),
        g: build_channel_scene(1),
        b: build_channel_scene(2),
    };

    let mut old_frame = Frame::<Rgba8>::new(W as u32, H as u32);
    let old_start = std::time::Instant::now();
    rasterize(&old_renderer, &mut old_frame, 1);
    let old_elapsed = old_start.elapsed();

    // ============================================================
    // NEW APPROACH: Mullet architecture (geometry once, Discrete colors)
    // ============================================================

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

    let world = ColorSurface {
        geometry: plane(-1.0),
        material: ColorChecker::<RgbaColorCube>::default(),
        background: ColorSky::<RgbaColorCube>::default(),
    };

    let new_renderer = ColorScreenRemap {
        inner: ColorScreenToDir {
            inner: ColorSurface {
                geometry: SphereAt {
                    center: (0.0, 0.0, 4.0),
                    radius: 1.0,
                },
                material: ColorReflect { inner: world.clone() },
                background: world,
            },
        },
        width: W as f32,
        height: H as f32,
    };

    let mut new_frame = Frame::<Rgba8>::new(W as u32, H as u32);
    let new_start = std::time::Instant::now();
    rasterize(&new_renderer, &mut new_frame, 1);
    let new_elapsed = new_start.elapsed();

    // ============================================================
    // COMPARE
    // ============================================================

    println!("3-channel: {:?}", old_elapsed);
    println!("Mullet:    {:?}", new_elapsed);
    println!(
        "Speedup:   {:.2}x",
        old_elapsed.as_secs_f64() / new_elapsed.as_secs_f64()
    );

    // Compare all pixels
    let mut max_diff = 0i32;
    let mut diff_count = 0usize;
    for (i, (old_p, new_p)) in old_frame.data.iter().zip(new_frame.data.iter()).enumerate() {
        let dr = (old_p.r() as i32 - new_p.r() as i32).abs();
        let dg = (old_p.g() as i32 - new_p.g() as i32).abs();
        let db = (old_p.b() as i32 - new_p.b() as i32).abs();
        let d = Ord::max(Ord::max(dr, dg), db);
        if d > max_diff {
            max_diff = d;
            let x = i % W;
            let y = i / W;
            println!(
                "New max diff {} at ({}, {}): old=({},{},{}) new=({},{},{})",
                d,
                x,
                y,
                old_p.r(),
                old_p.g(),
                old_p.b(),
                new_p.r(),
                new_p.g(),
                new_p.b()
            );
        }
        if d > 0 {
            diff_count += 1;
        }
    }

    println!("Max diff: {} (out of 255)", max_diff);
    println!("Pixels with diff: {} / {}", diff_count, W * H);

    // Allow small differences due to FP ordering, but they should be identical
    assert!(
        max_diff <= 1,
        "Max diff too large: {} (expected 0-1 for FP rounding)",
        max_diff
    );
}

/// Benchmark: Compare work-stealing vs single-threaded at 1080p
#[test]
fn test_work_stealing_benchmark() {
    const W: usize = 1920;
    const H: usize = 1080;

    // Build scene
    let world = ColorSurface {
        geometry: plane(-1.0),
        material: ColorChecker::<RgbaColorCube>::default(),
        background: ColorSky::<RgbaColorCube>::default(),
    };

    let scene = ColorSurface {
        geometry: SphereAt {
            center: (0.0, 0.0, 4.0),
            radius: 1.0,
        },
        material: ColorReflect { inner: world.clone() },
        background: world,
    };

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

    let renderable = ColorScreenRemap {
        inner: ColorScreenToDir { inner: scene },
        width: W as f32,
        height: H as f32,
    };

    // Single-threaded baseline
    let mut frame1 = Frame::<Rgba8>::new(W as u32, H as u32);
    let start1 = std::time::Instant::now();
    rasterize(&renderable, &mut frame1, 1);
    let single = start1.elapsed();

    // Work-stealing with 12 threads
    let mut frame2 = Frame::<Rgba8>::new(W as u32, H as u32);
    let start2 = std::time::Instant::now();
    rasterize(&renderable, &mut frame2, 12);
    let parallel = start2.elapsed();

    let speedup = single.as_secs_f64() / parallel.as_secs_f64();
    let mpps = (W * H) as f64 / parallel.as_secs_f64() / 1_000_000.0;

    println!("Single-threaded: {:?}", single);
    println!("Work-stealing (12 threads): {:?}", parallel);
    println!("Speedup: {:.2}x", speedup);
    println!("Throughput: {:.2} Mpix/s", mpps);

    // Verify correctness
    assert_eq!(
        frame1.data, frame2.data,
        "Parallel output must match single-threaded"
    );
}
