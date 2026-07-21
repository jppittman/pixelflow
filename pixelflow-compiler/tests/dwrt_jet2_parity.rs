//! Parity proof for Phase 2 of the kernel-unification plan
//! (docs/plans/2026-07-20-kernel-unification.md): a font-ramp-shaped coverage
//! expression whose gradient magnitude comes from symbolic `Dwrt` nodes,
//! JIT-compiled through `pixelflow_ir::lower_dwrt`, must match forward-mode
//! automatic differentiation (`pixelflow_core::jet::Jet2`) over the exact op
//! set the font path exercises: sub, mul, div, sqrt, select, comparisons,
//! clamp.
//!
//! Jet2 is the combinator path's ground truth for derivatives (the
//! antialiasing path in pixelflow-graphics evaluates coverage exactly this
//! way), so agreement here is the acceptance criterion for retiring the
//! emit-time jet mode: derivatives are ordinary expressions, lowered before
//! scheduling, compiled by the same backend.

use pixelflow_core::Field;
use pixelflow_core::jet::Jet2;
use pixelflow_ir::backend::emit::compile_arena_dag;
use pixelflow_ir::{BindingTable, ExprArena, ExprId, OpKind, eval_scalar};

/// First lane of a `Field` as `f32`.
fn lane0(f: Field) -> f32 {
    unsafe { core::mem::transmute_copy(&f) }
}

/// Push `Dwrt(e, Const(var))`.
fn dwrt(a: &mut ExprArena, e: ExprId, var: u8) -> ExprId {
    let v = a.push_const(var as f32);
    a.push_binary(OpKind::Dwrt, e, v)
}

/// Arena side: the coverage expression with `Dwrt`-based gradient
/// normalization.
///
///   g   = X − (2Y + 0.5)/(Y + 3)
///   f   = select(Y > 0, g, g·0.5)
///   cov = clamp(0.5 − f/√(DX(f)² + DY(f)²), 0, 1)
fn coverage_arena() -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let two = a.push_const(2.0);
    let half = a.push_const(0.5);
    let three = a.push_const(3.0);
    let zero = a.push_const(0.0);
    let one = a.push_const(1.0);

    let t = a.push_binary(OpKind::Mul, two, y);
    let num = a.push_binary(OpKind::Add, t, half);
    let den = a.push_binary(OpKind::Add, y, three);
    let ramp = a.push_binary(OpKind::Div, num, den);
    let g = a.push_binary(OpKind::Sub, x, ramp);

    let gate = a.push_binary(OpKind::Gt, y, zero);
    let g_half = a.push_binary(OpKind::Mul, g, half);
    let f = a.push_ternary(OpKind::Select, gate, g, g_half);

    let fx = dwrt(&mut a, f, 0);
    let fy = dwrt(&mut a, f, 1);
    let fx2 = a.push_binary(OpKind::Mul, fx, fx);
    let fy2 = a.push_binary(OpKind::Mul, fy, fy);
    let mag2 = a.push_binary(OpKind::Add, fx2, fy2);
    let mag = a.push_unary(OpKind::Sqrt, mag2);

    let ratio = a.push_binary(OpKind::Div, f, mag);
    let dist = a.push_binary(OpKind::Sub, half, ratio);
    let cov = a.push_ternary(OpKind::Clamp, dist, zero, one);
    (a, cov)
}

/// Jet2 side: the same `f` computed by forward-mode automatic differentiation
/// on seeded `Jet2` coordinates — exactly how the stamped font manifolds
/// evaluate over the `Jet2` domain (generic combinators cannot mix `f32`
/// literals into a `Jet2` domain, which is why the fonts stamp per-domain
/// impls; this test does the same inline). The coverage normalization is then
/// applied to the resulting (val, dx, dy) triple in scalar arithmetic.
fn coverage_jet2(px: f32, py: f32) -> f32 {
    let c = |v: f32| Jet2::constant(Field::from(v));
    let x = Jet2::x(Field::from(px));
    let y = Jet2::y(Field::from(py));

    let ramp = (y * c(2.0) + c(0.5)) / (y + c(3.0));
    let g = x - ramp;
    let f = Jet2::select(y.gt(c(0.0)), g, g * c(0.5));

    let (val, dx, dy) = (lane0(f.val), lane0(f.dx), lane0(f.dy));
    let mag = (dx * dx + dy * dy).sqrt();
    (0.5 - val / mag).clamp(0.0, 1.0)
}

/// Grid avoiding the select boundary (y = 0) and the ramp pole (y = -3).
const GRID_X: &[f32] = &[-1.5, -0.4, 0.0, 0.3, 1.2, 2.0];
const GRID_Y: &[f32] = &[-1.5, -0.6, 0.4, 1.1, 2.3];

fn check(name: &str, got: f32, want: f32, x: f32, y: f32) {
    let tol = 1e-5 + 1e-5 * want.abs();
    assert!(
        (got - want).abs() <= tol,
        "{name} at ({x}, {y}): got {got}, want {want}"
    );
}

/// Interpreter (post-`lower_dwrt`) vs Jet2 forward-mode AD.
#[test]
fn dwrt_interpreter_matches_jet2() {
    let (arena, root) = coverage_arena();
    let bindings = BindingTable::empty();
    for &y in GRID_Y {
        for &x in GRID_X {
            let ir = eval_scalar(&arena, root, &[x, y, 0.0, 0.0], &bindings);
            let jet = coverage_jet2(x, y);
            check("interpreter vs Jet2", ir, jet, x, y);
        }
    }
}

/// JIT-compiled (post-`lower_dwrt`) vs Jet2 forward-mode AD — the parity proof
/// for the font op set.
#[test]
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", not(target_feature = "avx512f"))
))]
fn dwrt_jit_matches_jet2() {
    use pixelflow_ir::backend::emit::executable;

    let (arena, root) = coverage_arena();
    let compiled = compile_arena_dag(&arena, root).expect("JIT compile of Dwrt arena failed");

    let jit_eval = |x: f32, y: f32| -> f32 {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use core::arch::aarch64::*;
            let f: executable::KernelFn = compiled.code.as_fn();
            let out = f(
                vdupq_n_f32(x),
                vdupq_n_f32(y),
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
            );
            vgetq_lane_f32(out, 0)
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use core::arch::x86_64::*;
            let f: executable::KernelFn = compiled.code.as_fn();
            let out = f(
                _mm_set1_ps(x),
                _mm_set1_ps(y),
                _mm_set1_ps(0.0),
                _mm_set1_ps(0.0),
            );
            _mm_cvtss_f32(out)
        }
    };

    for &y in GRID_Y {
        for &x in GRID_X {
            let jit = jit_eval(x, y);
            let jet = coverage_jet2(x, y);
            check("JIT vs Jet2", jit, jet, x, y);
        }
    }
}

/// The macro surface end-to-end: `kernel_jit!` bodies with `V`/`DX`/`DY`
/// derivative accessors are accepted (mapped to `Dwrt` nodes by `ir_bridge`)
/// and JIT-compile through `lower_dwrt`.
#[test]
fn kernel_jit_accepts_derivative_accessors() {
    use pixelflow_compiler::kernel_jit;
    use pixelflow_core::Manifold;

    let eval =
        |m: &dyn Manifold<(Field, Field, Field, Field), Output = Field>, x: f32, y: f32| -> f32 {
            lane0(m.eval((
                Field::from(x),
                Field::from(y),
                Field::from(0.0),
                Field::from(0.0),
            )))
        };

    // d(x²+y²)/dx + y = 2x + y ; V is the identity projection.
    let m = kernel_jit!(|| DX(X * X + Y * Y) + V(Y));
    for &(x, y) in &[(3.0f32, 4.0f32), (1.0, 2.0), (-2.0, 0.5)] {
        let got = eval(&m, x, y);
        let want = 2.0 * x + y;
        assert!(
            (got - want).abs() < 1e-4,
            "DX at ({x},{y}): got {got}, want {want}"
        );
    }

    // d(x·y²)/dy = 2xy.
    let m = kernel_jit!(|| DY(X * Y * Y));
    for &(x, y) in &[(3.0f32, 4.0f32), (1.0, 2.0), (-2.0, 0.5)] {
        let got = eval(&m, x, y);
        let want = 2.0 * x * y;
        assert!(
            (got - want).abs() < 1e-3,
            "DY at ({x},{y}): got {got}, want {want}"
        );
    }
}
