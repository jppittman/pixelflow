//! Compile-and-evaluate tests for e-graph optimization type-stability.
//!
//! Two historical defects forced font/bilinear kernels onto `kernel_raw!`:
//!
//! 1. **Type instability across the V/DX/DY projection boundary**: kernel
//!    bodies mixing domain-space math (e.g. `Jet2`) with Field-space math
//!    (post-`V`/`DX`/`DY` projection) were rewritten by the e-graph into forms
//!    with fresh literals (e.g. `a/b + c -> mul_add(a, 1/b, c)`); codegen
//!    lifted those literals to the kernel's domain scalar type, producing
//!    ill-typed `Jet2 op Field` expressions.
//!
//! 2. **Extraction reused non-Copy subexpressions**: a kernel with a manifold
//!    param tapped via `.at()` is not Copy (taps capture `&self.param`), but
//!    extraction could emit a form that binds a tap once and uses it twice,
//!    failing E0382.
//!
//! These tests assert both shapes now compile under the optimizing `kernel!`
//! macro and evaluate identically to the unoptimized `kernel_raw!` form.

use pixelflow_compiler::{kernel, kernel_raw};
use pixelflow_core::jet::Jet2;
use pixelflow_core::{DiscreteManifold, Field, Manifold};

/// Extract the first lane from a Field as f32.
fn field_extract(f: Field) -> f32 {
    unsafe { core::mem::transmute_copy(&f) }
}

// ============================================================================
// Defect 1: V/DX/DY projection boundary (gradient-normalized coverage ramp)
// ============================================================================

/// The line-crossing coverage ramp from the font renderer, stamped per
/// evaluation domain. This is exactly the shape that used to extract to
/// ill-typed code: `V(d) / (grad + V(min_grad)) + V(0.5)` invites the
/// `a/b + c -> mul_add(a, recip(b), c)` rewrite, whose fresh `1.0` literal
/// must stay in Field space (the ramp is Field math after projection).
macro_rules! ramp_kernel {
    ($n:ty) => {
        kernel!(|x0: f32, y0: f32, dx_over_dy: f32, min_grad: f32| -> $n {
            let d = X - ((Y - y0) * dx_over_dy + x0);
            let grad = (DX(d.clone()) * DX(d.clone()) + DY(d.clone()) * DY(d.clone())).sqrt();
            (V(d) / (grad + V(min_grad)) + V(0.5))
                .max(V(0.0))
                .min(V(1.0))
        })
    };
}

macro_rules! ramp_kernel_raw {
    ($n:ty) => {
        kernel_raw!(|x0: f32, y0: f32, dx_over_dy: f32, min_grad: f32| -> $n {
            let d = X - ((Y - y0) * dx_over_dy + x0);
            let grad = (DX(d.clone()) * DX(d.clone()) + DY(d.clone()) * DY(d.clone())).sqrt();
            (V(d) / (grad + V(min_grad)) + V(0.5))
                .max(V(0.0))
                .min(V(1.0))
        })
    };
}

#[test]
fn ramp_kernel_field_domain_matches_raw() {
    let opt = ramp_kernel!(Field)(2.0, 0.5, 0.25, 1e-3);
    let raw = ramp_kernel_raw!(Field)(2.0, 0.5, 0.25, 1e-3);

    for &(x, y) in &[(0.0, 0.0), (2.5, 1.0), (1.9, 0.3), (-4.0, 7.5)] {
        let p = (
            Field::from(x),
            Field::from(y),
            Field::from(0.0_f32),
            Field::from(0.0_f32),
        );
        let got = field_extract(opt.eval(p));
        let want = field_extract(raw.eval(p));
        assert!(
            (got - want).abs() < 1e-5,
            "Field domain mismatch at ({x}, {y}): kernel! gave {got}, kernel_raw! gave {want}"
        );
    }
}

#[test]
fn ramp_kernel_jet2_domain_matches_raw() {
    let opt = ramp_kernel!(Jet2)(2.0, 0.5, 0.25, 1e-3);
    let raw = ramp_kernel_raw!(Jet2)(2.0, 0.5, 0.25, 1e-3);

    for &(x, y) in &[(0.0, 0.0), (2.5, 1.0), (1.9, 0.3), (-4.0, 7.5)] {
        let p = (
            Jet2::x(Field::from(x)),
            Jet2::y(Field::from(y)),
            Jet2::constant(Field::from(0.0_f32)),
            Jet2::constant(Field::from(0.0_f32)),
        );
        let got = field_extract(opt.eval(p));
        let want = field_extract(raw.eval(p));
        assert!(
            (got - want).abs() < 1e-5,
            "Jet2 domain mismatch at ({x}, {y}): kernel! gave {got}, kernel_raw! gave {want}"
        );
    }

    // Sanity: over Jet2 the ramp is gradient-normalized, so a point one pixel
    // inside the crossing must have coverage strictly between hard 0/1 only
    // near the edge; far away it saturates. Check saturation far inside.
    let far_inside = (
        Jet2::x(Field::from(100.0_f32)),
        Jet2::y(Field::from(0.0_f32)),
        Jet2::constant(Field::from(0.0_f32)),
        Jet2::constant(Field::from(0.0_f32)),
    );
    let v = field_extract(opt.eval(far_inside));
    assert!(
        (v - 1.0).abs() < 1e-6,
        "coverage must saturate to 1 far inside the crossing, got {v}"
    );
}

// ============================================================================
// Defect 2: non-Copy manifold-param taps shared by extraction
// ============================================================================

// The 4-tap bilinear kernel shape: `tex` taps are non-Copy (they capture
// `&self.tex`), and the optimizer is free to rewrite the weighted sum into a
// form that references a tap more than once. That must compile (clone/borrow
// legally), not fail E0382.
kernel!(pub struct BilerpShape = |tex: kernel| Field -> Field {
    let x0 = X.floor();
    let y0 = Y.floor();
    let fx = X - x0;
    let fy = Y - y0;

    let c00 = tex.at(x0, y0, Z, W);
    let c10 = tex.at(x0 + 1.0, y0, Z, W);
    let c01 = tex.at(x0, y0 + 1.0, Z, W);
    let c11 = tex.at(x0 + 1.0, y0 + 1.0, Z, W);

    c00 * ((1.0 - fx) * (1.0 - fy))
        + c10 * (fx * (1.0 - fy))
        + c01 * ((1.0 - fx) * fy)
        + c11 * (fx * fy)
});

fn sample<M>(m: &M, x: f32, y: f32) -> f32
where
    M: Manifold<(Field, Field, Field, Field), Output = Field>,
{
    field_extract(m.eval((
        Field::from(x),
        Field::from(y),
        Field::from(0.0_f32),
        Field::from(0.0_f32),
    )))
}

#[test]
fn bilerp_shape_constant_field_is_identity() {
    let tex = DiscreteManifold::new(vec![0.75; 9], 3, 3);
    let bilerp = BilerpShape::new(tex);

    for &(x, y) in &[(0.0, 0.0), (0.5, 0.5), (1.25, 0.75), (1.9, 1.1)] {
        let v = sample(&bilerp, x, y);
        assert!(
            (v - 0.75).abs() < 1e-6,
            "constant field not reproduced at ({x}, {y}): got {v}"
        );
    }
}

#[test]
fn bilerp_shape_linear_gradient_reproduced() {
    // f(x, y) = x + 2y sampled at integer coords, 4x4. Bilinear interpolation
    // of an affine function is exact everywhere.
    let mut buf = Vec::with_capacity(16);
    for y in 0..4 {
        for x in 0..4 {
            buf.push(x as f32 + 2.0 * y as f32);
        }
    }
    let bilerp = BilerpShape::new(DiscreteManifold::new(buf, 4, 4));

    for &(x, y) in &[(0.5, 0.5), (1.25, 2.75), (0.1, 0.9), (2.5, 1.0)] {
        let expect = x + 2.0 * y;
        let v = sample(&bilerp, x, y);
        assert!(
            (v - expect).abs() < 1e-5,
            "gradient not reproduced at ({x}, {y}): got {v}, want {expect}"
        );
    }
}
