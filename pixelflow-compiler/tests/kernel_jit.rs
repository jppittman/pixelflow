//! Integration tests for the kernel_jit! macro.
//!
//! These tests verify the full pipeline from macro input to executable JIT code.

use pixelflow_compiler::kernel_jit;
use pixelflow_core::{Field, Manifold};

// ============================================================================
// Helpers
// ============================================================================

/// Extract the first lane from a Field as f32.
/// Field is repr(transparent) over the platform SIMD type; the first lane
/// is always the lowest-address element.
fn field_extract(f: Field) -> f32 {
    unsafe { core::mem::transmute_copy(&f) }
}

fn eval1(m: &impl Manifold<(Field, Field, Field, Field), Output = Field>, x: f32) -> f32 {
    field_extract(m.eval((
        Field::from(x),
        Field::from(0.0_f32),
        Field::from(0.0_f32),
        Field::from(0.0_f32),
    )))
}

fn eval2(
    m: &impl Manifold<(Field, Field, Field, Field), Output = Field>,
    x: f32,
    y: f32,
) -> f32 {
    field_extract(m.eval((
        Field::from(x),
        Field::from(y),
        Field::from(0.0_f32),
        Field::from(0.0_f32),
    )))
}

fn eval3(
    m: &impl Manifold<(Field, Field, Field, Field), Output = Field>,
    x: f32,
    y: f32,
    z: f32,
) -> f32 {
    field_extract(m.eval((
        Field::from(x),
        Field::from(y),
        Field::from(z),
        Field::from(0.0_f32),
    )))
}

// ============================================================================
// Basic arithmetic
// ============================================================================

#[test]
fn test_jit_macro_return_x() {
    let m = kernel_jit!(|| X);
    assert_eq!(eval1(&m, 42.0), 42.0);
}

#[test]
fn test_jit_macro_add_xy() {
    let m = kernel_jit!(|| X + Y);
    assert_eq!(eval2(&m, 10.0, 32.0), 42.0);
}

#[test]
fn test_jit_macro_complex_expr() {
    // (X + Y) * Z
    let m = kernel_jit!(|| (X + Y) * Z);
    assert_eq!(eval3(&m, 2.0, 5.0, 6.0), 42.0); // (2+5)*6 = 42
}

#[test]
fn test_jit_macro_subtraction() {
    let m = kernel_jit!(|| X - Y);
    assert_eq!(eval2(&m, 100.0, 58.0), 42.0);
}

#[test]
fn test_jit_macro_division() {
    let m = kernel_jit!(|| X / Y);
    assert_eq!(eval2(&m, 84.0, 2.0), 42.0);
}

#[test]
fn test_jit_macro_negation() {
    let m = kernel_jit!(|| -X);
    assert_eq!(eval1(&m, -42.0), 42.0);
}

// ============================================================================
// Transcendentals (lowered via polynomial)
// ============================================================================

#[test]
fn test_jit_macro_sin() {
    // sin(0) = 0
    let m = kernel_jit!(|| X.sin());
    let val = eval1(&m, 0.0);
    assert!((val - 0.0).abs() < 0.001, "sin(0) = {val}, expected ~0");
}

#[test]
fn test_jit_macro_sin_pi_half() {
    // sin(π/2) ≈ 1
    let m = kernel_jit!(|| X.sin());
    let val = eval1(&m, core::f32::consts::FRAC_PI_2);
    assert!((val - 1.0).abs() < 0.01, "sin(π/2) = {val}, expected ~1");
}

#[test]
fn test_jit_macro_cos() {
    // cos(0) = 1
    let m = kernel_jit!(|| X.cos());
    let val = eval1(&m, 0.0);
    assert!((val - 1.0).abs() < 0.01, "cos(0) = {val}, expected ~1");
}

#[test]
fn test_jit_macro_sqrt() {
    // sqrt(1764) = 42
    let m = kernel_jit!(|| X.sqrt());
    assert_eq!(eval1(&m, 1764.0), 42.0);
}

#[test]
fn test_jit_macro_abs() {
    let m = kernel_jit!(|| X.abs());
    assert_eq!(eval1(&m, -42.0), 42.0);
}

#[test]
fn test_jit_macro_min_max() {
    let m_min = kernel_jit!(|| X.min(Y));
    let m_max = kernel_jit!(|| X.max(Y));
    assert_eq!(eval2(&m_min, 10.0, 42.0), 10.0);
    assert_eq!(eval2(&m_max, 10.0, 42.0), 42.0);
}

// ============================================================================
// Parameter tests — builder closure API
// ============================================================================

#[test]
fn kernel_jit_no_params_is_manifold() {
    // Zero-param case: returns JitManifold implementing Manifold
    let m = kernel_jit!(|| X + Y);
    let result = m.eval((
        Field::from(10.0_f32),
        Field::from(32.0_f32),
        Field::from(0.0_f32),
        Field::from(0.0_f32),
    ));
    assert!((field_extract(result) - 42.0).abs() < 1e-5);
}

#[test]
fn kernel_jit_one_param_builder() {
    // Single param returns builder closure |offset: f32| -> JitManifold
    let builder = kernel_jit!(|offset: f32| X + offset);
    let m = builder(32.0_f32);
    let result = m.eval((
        Field::from(10.0_f32),
        Field::from(0.0_f32),
        Field::from(0.0_f32),
        Field::from(0.0_f32),
    ));
    assert!((field_extract(result) - 42.0).abs() < 1e-5);
}

#[test]
fn kernel_jit_two_params_builder() {
    let builder = kernel_jit!(|cx: f32, r: f32| (X - cx) * r);
    let m = builder(1.0_f32, 2.0_f32);
    // X=5.0: (5.0 - 1.0) * 2.0 = 8.0
    let result = m.eval((
        Field::from(5.0_f32),
        Field::from(0.0_f32),
        Field::from(0.0_f32),
        Field::from(0.0_f32),
    ));
    assert!((field_extract(result) - 8.0).abs() < 1e-5);
}

#[test]
fn kernel_jit_same_semantics_as_kernel() {
    use pixelflow_compiler::kernel;

    let jit_builder = kernel_jit!(|cx: f32| X - cx);
    let ct_builder = kernel!(|cx: f32| X - cx);

    let jit_m = jit_builder(3.0_f32);
    let ct_m = ct_builder(3.0_f32);

    for x_val in [0.0_f32, 1.0, 5.0, -2.0, 100.0] {
        let p = (
            Field::from(x_val),
            Field::from(0.0_f32),
            Field::from(0.0_f32),
            Field::from(0.0_f32),
        );
        let jit_result = field_extract(jit_m.eval(p));
        let ct_result = field_extract(ct_m.eval(p));
        assert!(
            (jit_result - ct_result).abs() < 1e-5,
            "mismatch at x={x_val}: jit={jit_result} ct={ct_result}"
        );
    }
}

// ============================================================================
// Inverse trigonometric functions
// ============================================================================

#[test]
fn test_jit_macro_atan2_basic() {
    let m = kernel_jit!(|| Y.atan2(X));
    // atan2(1, 1) = π/4 — polynomial has ~0.06 error at t=1 boundary
    let val = eval2(&m, 1.0, 1.0);
    assert!(
        (val - std::f32::consts::FRAC_PI_4).abs() < 0.07,
        "atan2(1, 1) = {val}, expected ~{}", std::f32::consts::FRAC_PI_4
    );

    // atan2(1, 2) = atan(0.5) ≈ 0.4636 — well inside polynomial range
    let val2 = eval2(&m, 2.0, 1.0);
    let expected2 = 1.0_f32.atan2(2.0);
    assert!(
        (val2 - expected2).abs() < 0.02,
        "atan2(1, 2) = {val2}, expected ~{expected2}"
    );
}

#[test]
fn test_jit_macro_atan2_quadrants() {
    let m = kernel_jit!(|| Y.atan2(X));

    // Use ratio = 0.5 (well inside polynomial range) for quadrant tests
    // atan2(1, 2): Q1 — atan(0.5) ≈ 0.4636
    let q1 = eval2(&m, 2.0, 1.0);
    let expected_q1 = 1.0_f32.atan2(2.0);
    assert!(
        (q1 - expected_q1).abs() < 0.02,
        "Q1: atan2(1, 2) = {q1}, expected ~{expected_q1}"
    );

    // atan2(1, -2): Q2 — π - atan(0.5) ≈ 2.678
    let q2 = eval2(&m, -2.0, 1.0);
    let expected_q2 = 1.0_f32.atan2(-2.0);
    assert!(
        (q2 - expected_q2).abs() < 0.07,
        "Q2: atan2(1, -2) = {q2}, expected ~{expected_q2}"
    );

    // atan2(-1, -2): Q3 — -(π - atan(0.5)) ≈ -2.678
    let q3 = eval2(&m, -2.0, -1.0);
    let expected_q3 = (-1.0_f32).atan2(-2.0);
    assert!(
        (q3 - expected_q3).abs() < 0.07,
        "Q3: atan2(-1, -2) = {q3}, expected ~{expected_q3}"
    );

    // atan2(-1, 2): Q4 — -atan(0.5) ≈ -0.4636
    let q4 = eval2(&m, 2.0, -1.0);
    let expected_q4 = (-1.0_f32).atan2(2.0);
    assert!(
        (q4 - expected_q4).abs() < 0.02,
        "Q4: atan2(-1, 2) = {q4}, expected ~{expected_q4}"
    );
}

#[test]
fn test_jit_macro_atan() {
    let m = kernel_jit!(|| X.atan());
    // atan(0.5) ≈ 0.4636 — well within polynomial range
    let val = eval1(&m, 0.5);
    let expected = 0.5_f32.atan();
    assert!(
        (val - expected).abs() < 0.02,
        "atan(0.5) = {val}, expected ~{expected}"
    );
    // atan(0) = 0
    let val0 = eval1(&m, 0.0);
    assert!(
        val0.abs() < 0.01,
        "atan(0) = {val0}, expected ~0"
    );
}

#[test]
fn test_jit_macro_asin() {
    let m = kernel_jit!(|| X.asin());
    // asin(0) = 0
    let val0 = eval1(&m, 0.0);
    assert!(
        val0.abs() < 0.01,
        "asin(0) = {val0}, expected ~0"
    );
    // asin(0.5) = π/6 ≈ 0.5236 — ratio < 1, polynomial is accurate
    let val_half = eval1(&m, 0.5);
    let expected = 0.5_f32.asin();
    assert!(
        (val_half - expected).abs() < 0.02,
        "asin(0.5) = {val_half}, expected ~{expected}"
    );
}

#[test]
fn test_jit_macro_acos() {
    let m = kernel_jit!(|| X.acos());
    // acos(0.5) = π/3 ≈ 1.047 — exercises large-ratio path (ratio ≈ 1.73)
    let val_half = eval1(&m, 0.5);
    let expected = 0.5_f32.acos();
    assert!(
        (val_half - expected).abs() < 0.05,
        "acos(0.5) = {val_half}, expected ~{expected}"
    );
    // acos(0) = π/2
    let val0 = eval1(&m, 0.0);
    assert!(
        (val0 - std::f32::consts::FRAC_PI_2).abs() < 0.07,
        "acos(0) = {val0}, expected ~π/2"
    );
}
