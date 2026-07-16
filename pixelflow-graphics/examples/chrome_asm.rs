//! ASM inspection - mirrors the exact benchmark pattern
//!
//! Run: cargo-asm -p pixelflow-graphics --example chrome_asm eval_one_pixel --release

use pixelflow_core::combinators::At;
use pixelflow_core::jet::Jet3;
use pixelflow_core::{Discrete, Field, Manifold, ManifoldCompat};
use pixelflow_graphics::render::color::RgbaColorCube;
use pixelflow_compiler::ManifoldExpr;
use pixelflow_graphics::scene3d::{
    ColorChecker, ColorReflect, ColorScreenToDir, ColorSky, ColorSurface, plane,
};

type Field4 = (Field, Field, Field, Field);
type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);
use std::hint::black_box;

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

        let epsilon_sq = Jet3::constant(Field::from(0.0001));
        d_dot_c - (discriminant + epsilon_sq).sqrt()
    }
}

// This is the EXACT pattern from the benchmark
#[derive(Copy, Clone)]
struct ColorScreenRemap<M> {
    inner: M,
    width: f32,
    height: f32,
}

impl<M: ManifoldCompat<Field, Output = Discrete>> Manifold<Field4> for ColorScreenRemap<M> {
    type Output = Discrete;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Discrete {
        let (x, y, z, w) = p;
        let width = Field::from(self.width);
        let height = Field::from(self.height);
        let scale = Field::from(2.0) / height;
        let sx = (x - width * Field::from(0.5)) * scale.clone();
        let sy = (height * Field::from(0.5) - y) * scale;
        // Use At combinator to evaluate at transformed coordinates
        At {
            inner: &self.inner,
            x: sx,
            y: sy,
            z,
            w,
        }
        .collapse()
    }
}

/// Evaluate one pixel - this is what we want ASM for
#[inline(never)]
pub fn eval_one_pixel(x: Field, y: Field) -> Discrete {
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

    let renderable = ColorScreenRemap {
        inner: ColorScreenToDir { inner: scene },
        width: 1920.0,
        height: 1080.0,
    };

    renderable.eval((x, y, Field::from(0.0), Field::from(0.0)))
}

fn main() {
    let x = Field::sequential(960.0);
    let y = Field::from(540.0);
    let result = eval_one_pixel(black_box(x), black_box(y));
    black_box(result);
}
