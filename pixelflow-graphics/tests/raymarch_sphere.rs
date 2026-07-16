//! Test: 3D scene rendering with sphere + floor using the scene3d architecture

use pixelflow_core::combinators::At;
use pixelflow_core::jet::Jet3;
use pixelflow_core::{Discrete, Field, Manifold, ManifoldCompat};
use pixelflow_compiler::ManifoldExpr;

type Field4 = (Field, Field, Field, Field);
type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);
use pixelflow_graphics::render::color::{Rgba8, RgbaColorCube};
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::rasterize;
use pixelflow_graphics::scene3d::{
    ColorChecker, ColorReflect, ColorScreenToDir, ColorSky, ColorSurface, plane,
};

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
use std::fs::File;
use std::io::Write;

/// Remap pixel coordinates to normalized screen coordinates.
/// Transforms [0, width] × [0, height] → normalized coordinates
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

        // Map pixel coords to normalized screen coordinates (flip y)
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

#[test]
fn test_sphere_on_floor() {
    const W: usize = 400;
    const H: usize = 300;

    // Floor + sky as the world (background)
    let world = ColorSurface {
        geometry: plane(-0.5),
        material: ColorChecker::<RgbaColorCube>::default(),
        background: ColorSky::<RgbaColorCube>::default(),
    };

    // Chrome sphere at (0.0, 0.5, 4.0) with radius 1.0, reflecting the world
    let scene = ColorSurface {
        geometry: SphereAt {
            center: (0.0, 0.5, 4.0),
            radius: 1.0,
        },
        material: ColorReflect { inner: world.clone() },
        background: world,
    };

    // Wrap with screen coordinate transformation
    let renderable = ColorScreenRemap {
        inner: ColorScreenToDir { inner: scene },
        width: W as f32,
        height: H as f32,
    };

    let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
    rasterize(&renderable, &mut frame, 1);

    // Save PPM
    let path = std::env::temp_dir().join("pixelflow_raymarch_sh.ppm");
    let mut file = File::create(&path).unwrap();
    writeln!(file, "P6\n{} {}\n255", W, H).unwrap();
    for p in &frame.data {
        file.write_all(&[p.r(), p.g(), p.b()]).unwrap();
    }
    println!("Saved: {}", path.display());

    // Basic sanity: center should hit something (not pure sky)
    let center = &frame.data[(H / 2) * W + (W / 2)];
    // Sky is blue-ish, sphere reflection should be different
    assert!(
        center.r() > 10 || center.g() > 10 || center.b() > 10,
        "Center pixel should have some color"
    );
}

/// Test with solid gray material (non-reflective)
#[test]
fn test_sphere_on_matte_floor() {


    const W: usize = 400;
    const H: usize = 300;

    // Simple solid gray material
    #[derive(Copy, Clone, ManifoldExpr)]
    struct SolidGray;

    impl pixelflow_core::Manifold<Jet3_4> for SolidGray {
        type Output = Discrete;

        fn eval(&self, _p: Jet3_4) -> Discrete {
            let gray = Field::from(0.5);
            Discrete::pack(gray, gray, gray, Field::from(1.0))
        }
    }

    // Floor + sky as the world (background)
    let world = ColorSurface {
        geometry: plane(-0.5),
        material: SolidGray,
        background: ColorSky::<RgbaColorCube>::default(),
    };

    // Matte gray sphere at (0.0, 0.5, 4.0) with radius 1.0
    let scene = ColorSurface {
        geometry: SphereAt {
            center: (0.0, 0.5, 4.0),
            radius: 1.0,
        },
        material: SolidGray,
        background: world,
    };

    let renderable = ColorScreenRemap {
        inner: ColorScreenToDir { inner: scene },
        width: W as f32,
        height: H as f32,
    };

    let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
    rasterize(&renderable, &mut frame, 1);

    // Save PPM
    let path = std::env::temp_dir().join("pixelflow_raymarch_matte.ppm");
    let mut file = File::create(&path).unwrap();
    writeln!(file, "P6\n{} {}\n255", W, H).unwrap();
    for p in &frame.data {
        file.write_all(&[p.r(), p.g(), p.b()]).unwrap();
    }
    println!("Saved: {}", path.display());

    // Center should hit geometry (gray sphere)
    let center = &frame.data[(H / 2) * W + (W / 2)];
    // Should be grayish (around 127-128 for 0.5 * 255)
    assert!(
        center.r() > 100 && center.r() < 150,
        "Center pixel should be gray: r={}",
        center.r()
    );
}

/// Chrome sphere on checkerboard floor
#[test]
fn test_chrome_sphere_on_checkerboard() {
    const WIDTH: usize = 400;
    const HEIGHT: usize = 300;

    // Floor with checkerboard pattern + sky background
    let world = ColorSurface {
        geometry: plane(-0.5),
        material: ColorChecker::<RgbaColorCube>::default(),
        background: ColorSky::<RgbaColorCube>::default(),
    };

    // Chrome sphere at (0.0, 0.5, 4.0) reflecting the checkerboard floor
    let scene = ColorSurface {
        geometry: SphereAt {
            center: (0.0, 0.5, 4.0),
            radius: 1.0,
        },
        material: ColorReflect { inner: world.clone() },
        background: world,
    };

    let renderable = ColorScreenRemap {
        inner: ColorScreenToDir { inner: scene },
        width: WIDTH as f32,
        height: HEIGHT as f32,
    };

    let mut frame = Frame::<Rgba8>::new(WIDTH as u32, HEIGHT as u32);
    rasterize(&renderable, &mut frame, 1);

    // Save PPM
    let path = std::env::temp_dir().join("pixelflow_chrome_checker.ppm");
    let mut file = File::create(&path).unwrap();
    writeln!(file, "P6\n{} {}\n255", WIDTH, HEIGHT).unwrap();
    for p in &frame.data {
        file.write_all(&[p.r(), p.g(), p.b()]).unwrap();
    }
    println!("Saved: {}", path.display());

    // Debug: print some pixel values
    let center = &frame.data[(HEIGHT / 2) * WIDTH + (WIDTH / 2)];
    println!(
        "Center pixel (sphere): r={} g={} b={}",
        center.r(),
        center.g(),
        center.b()
    );
    let bottom = &frame.data[(HEIGHT * 3 / 4) * WIDTH + (WIDTH / 2)];
    println!(
        "Bottom pixel (floor): r={} g={} b={}",
        bottom.r(),
        bottom.g(),
        bottom.b()
    );
    let top = &frame.data[(HEIGHT / 4) * WIDTH + (WIDTH / 2)];
    println!("Top pixel (sky): r={} g={} b={}", top.r(), top.g(), top.b());

    // Verify: center should be chrome sphere (reflective)
    let center = &frame.data[(HEIGHT / 2) * WIDTH + (WIDTH / 2)];
    assert!(
        center.r() > 10 || center.g() > 10 || center.b() > 10,
        "Center pixel should hit chrome sphere"
    );

    // Verify: bottom should be checkerboard floor
    let bottom = &frame.data[(HEIGHT * 3 / 4) * WIDTH + (WIDTH / 2)];
    assert!(
        bottom.r() > 10 || bottom.g() > 10 || bottom.b() > 10,
        "Bottom pixel should hit floor"
    );
}
