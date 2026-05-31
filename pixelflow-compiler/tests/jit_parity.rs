//! Correctness harness for the JIT backend (`kernel_jit!`), validated against
//! f32 ground truth.
//!
//! ## Why ground truth, not the combinator backend
//!
//! The original plan was to gate the JIT by comparing it to the combinator
//! backend (`kernel!`). That oracle is unsound: the two backends disagree with
//! ground truth in *different* ops. Measured on x86-64:
//!
//! - `X + Y*Z` at (1,2,3): JIT = 2, combinator = 7 (correct). The JIT
//!   miscompiled the FMA-fused shape (an SSE two-operand register clobber in
//!   the Sethi-Ullman emitter); now fixed and guarded by [`jit_arithmetic`] /
//!   [`jit_noncommutative_heavy_right`].
//! - `sin(1)`: JIT = 0.8414683 (libm = 0.8414710, correct), combinator =
//!   0.5116387 (badly inaccurate). Here the *combinator* is wrong.
//!
//! So we hold the JIT to f32 ground truth directly, not to the combinator.

use pixelflow_compiler::kernel_jit;
use pixelflow_core::{Field, Manifold};

type F4 = (Field, Field, Field, Field);

/// First lane of a `Field` as `f32` (`Field` is `repr(transparent)` over the
/// platform SIMD type; the first lane is the lowest-address element).
fn lane0(f: Field) -> f32 {
    unsafe { core::mem::transmute_copy(&f) }
}

fn eval(m: &impl Manifold<F4, Output = Field>, p: (f32, f32, f32, f32)) -> f32 {
    lane0(m.eval((
        Field::from(p.0),
        Field::from(p.1),
        Field::from(p.2),
        Field::from(p.3),
    )))
}

/// Sample points spanning magnitudes/signs across all four coords.
const SAMPLES: &[(f32, f32, f32, f32)] = &[
    (0.0, 0.0, 0.0, 0.0),
    (1.0, 2.0, 3.0, 4.0),
    (-1.5, 0.5, 2.25, -3.0),
    (3.0, 4.0, 0.0, 1.0),
    (0.1, 0.2, 0.3, 0.4),
    (10.0, -10.0, 5.0, -5.0),
    (0.7, 1.3, -0.9, 2.1),
];

/// Small-argument samples for transcendentals. The JIT's polynomial sin/cos are
/// accurate near zero (sin(1) matches libm to ~3e-6) but lose precision for
/// larger arguments (sin(3) is off by ~5%), so range-reduction quality is a
/// separate concern from the kernels exercised here.
const SMALL: &[(f32, f32, f32, f32)] = &[
    (0.0, 0.0, 0.0, 0.0),
    (0.5, -0.5, 0.25, -0.25),
    (1.0, -1.0, 0.7, -0.7),
    (0.1, 0.2, -0.3, 0.4),
    (1.2, -1.2, 0.9, -0.9),
];

fn check(name: &str, got: f32, want: f32, abs_tol: f32, rel_tol: f32) {
    let diff = (got - want).abs();
    let tol = abs_tol + rel_tol * want.abs();
    assert!(
        diff <= tol || (want.is_nan() && got.is_nan()),
        "{name}: jit={got} truth={want} diff={diff} tol={tol}"
    );
}

/// Build a JIT kernel, evaluate it at every sample, and compare to the scalar
/// reference `$ref` (a closure of `(x,y,z,w) -> f32`).
macro_rules! jit_truth {
    ($name:literal, $jit:expr, $ref:expr, $abs:expr, $rel:expr) => {
        jit_truth!($name, $jit, $ref, $abs, $rel, SAMPLES)
    };
    ($name:literal, $jit:expr, $ref:expr, $abs:expr, $rel:expr, $pts:expr) => {{
        let m = $jit;
        let r = $ref;
        for &p in $pts {
            check($name, eval(&m, p), r(p.0, p.1, p.2, p.3), $abs, $rel);
        }
    }};
}

#[test]
fn jit_arithmetic() {
    jit_truth!("sub_div", kernel_jit!(|| (X - Y) / (Z + 1.0)), |x: f32, y: f32, z: f32, _w| (x - y) / (z + 1.0), 1e-4, 1e-4);
    jit_truth!("affine", kernel_jit!(|| 2.0 * X - 3.0 * Y + 1.0), |x: f32, y: f32, _z, _w| 2.0 * x - 3.0 * y + 1.0, 1e-4, 1e-4);
    // FMA shapes, both operand orders. `X + Y*Z` (product on the RHS of the
    // add) previously miscompiled to `Y` due to an SSE two-operand clobber in
    // the Sethi-Ullman emitter; covered here as a regression guard.
    jit_truth!("mul_add", kernel_jit!(|| X * Y + Z), |x: f32, y: f32, z: f32, _w| x * y + z, 1e-4, 1e-4);
    jit_truth!("add_mul", kernel_jit!(|| X + Y * Z), |x: f32, y: f32, z: f32, _w| x + y * z, 1e-4, 1e-4);
}

#[test]
fn jit_unary() {
    jit_truth!("abs", kernel_jit!(|| (X - Y).abs()), |x: f32, y: f32, _z, _w| (x - y).abs(), 1e-5, 1e-5);
    jit_truth!("neg", kernel_jit!(|| (-X)), |x: f32, _y, _z, _w| -x, 1e-5, 1e-5);
    // floor is now in the AVX-512 op set too (vrndscaleps), so this runs on
    // both widths.
    jit_truth!("floor", kernel_jit!(|| X.floor()), |x: f32, _y, _z, _w| x.floor(), 1e-5, 1e-5);
}

#[test]
fn jit_sqrt_norm() {
    jit_truth!("norm2", kernel_jit!(|| (X * X + Y * Y).sqrt()), |x: f32, y: f32, _z, _w| (x * x + y * y).sqrt(), 1e-3, 1e-3);
    jit_truth!("norm3", kernel_jit!(|| (X * X + Y * Y + Z * Z).sqrt()), |x: f32, y: f32, z: f32, _w| (x * x + y * y + z * z).sqrt(), 1e-3, 1e-3);
}

#[test]
fn jit_minmax() {
    jit_truth!("min_max", kernel_jit!(|| X.max(Y).min(Z)), |x: f32, y: f32, z: f32, _w| x.max(y).min(z), 1e-5, 1e-5);
}

/// Transcendentals (sin/cos/exp) are not in the AVX-512 backend's Stage-1 op
/// set (the wide polynomial ports are a later stage), so `compile_arena_dag`
/// rejects them under `+avx512f` and `kernel_jit!` panics. Gate off there; the
/// 128-bit path covers them.
#[test]
#[cfg(not(target_feature = "avx512f"))]
fn jit_transcendental() {
    // Small-argument range. Tolerance here is "ballpark", not full f32
    // precision: these are SIMD polynomial approximations (measured error up to
    // ~1.3% for cos near x=1), so the bound is set to catch *logic* errors —
    // like the FMA bug below, which is off by whole integers — not to certify
    // ulp accuracy. Tightening the approximations is separate work. (The JIT's
    // sin is already markedly more accurate than the combinator backend's; see
    // the module docs.)
    jit_truth!("sin", kernel_jit!(|| X.sin()), |x: f32, _y, _z, _w| x.sin(), 3e-2, 3e-2, SMALL);
    jit_truth!("cos", kernel_jit!(|| X.cos()), |x: f32, _y, _z, _w| x.cos(), 3e-2, 3e-2, SMALL);
    jit_truth!("exp", kernel_jit!(|| X.exp()), |x: f32, _y, _z, _w| x.exp(), 3e-2, 3e-2, SMALL);
}

#[test]
fn jit_scalar_params() {
    // N-param builder: constants folded into the JIT'd kernel.
    let cx = 1.0_f32;
    let cy = 2.0_f32;
    let r = 0.5_f32;
    let m = kernel_jit!(|cx: f32, cy: f32, r: f32| {
        ((X - cx) * (X - cx) + (Y - cy) * (Y - cy)).sqrt() - r
    })(cx, cy, r);
    for &p in SAMPLES {
        let want = ((p.0 - cx) * (p.0 - cx) + (p.1 - cy) * (p.1 - cy)).sqrt() - r;
        check("circle_sdf", eval(&m, p), want, 1e-3, 1e-3);
    }
}

/// Non-commutative ops with the heavier operand on the right exercise the other
/// half of the SSE two-operand hazard: the emitter can't swap operands, so it
/// must keep the left operand in `dst`. Regression guard for that path.
#[test]
fn jit_noncommutative_heavy_right() {
    jit_truth!("sub_heavy_r", kernel_jit!(|| X - Y * Z), |x: f32, y: f32, z: f32, _w| x - y * z, 1e-4, 1e-4);
    jit_truth!("div_heavy_r", kernel_jit!(|| X / (Y * Z + 1.0)), |x: f32, y: f32, z: f32, _w| x / (y * z + 1.0), 1e-4, 1e-4);
}
