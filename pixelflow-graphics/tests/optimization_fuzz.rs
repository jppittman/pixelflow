//! Fuzz tests for optimization correctness.
//!
//! These tests verify that the kernel! macro's optimizations preserve
//! numerical semantics within floating-point epsilon tolerance.
//!
//! Strategy:
//! 1. Define kernel expressions that exercise various optimizations
//! 2. Define hand-written reference implementations (no macro magic)
//! 3. Use proptest to generate random inputs
//! 4. Assert kernel output matches reference within epsilon

use pixelflow_core::{Field, Manifold};
use pixelflow_compiler::kernel;
use proptest::prelude::*;

type Field4 = (Field, Field, Field, Field);

/// Maximum relative error tolerance for floating-point comparisons.
/// We use a relatively loose epsilon because:
/// - FMA fusion changes rounding behavior
/// - rsqrt approximations have ~1e-6 relative error
const EPSILON: f32 = 1e-4;

/// Absolute tolerance for values near zero where relative error explodes.
const ABS_EPSILON: f32 = 1e-6;

fn field4(x: f32, y: f32, z: f32, w: f32) -> Field4 {
    (Field::from(x), Field::from(y), Field::from(z), Field::from(w))
}

/// Extract first lane from Field for comparison.
fn field_val(f: Field) -> f32 {
    // Field is SIMD, all lanes have same value for broadcast inputs
    // Use the Debug representation to extract value
    let debug = format!("{:?}", f);
    // Parse "Field(F32x16([1.0, 1.0, ...]))" or similar
    if let Some(start) = debug.find('[') {
        if let Some(end) = debug.find(',') {
            if let Ok(v) = debug[start + 1..end].trim().parse::<f32>() {
                return v;
            }
        }
        // Single value case
        if let Some(end) = debug.find(']') {
            if let Ok(v) = debug[start + 1..end].trim().parse::<f32>() {
                return v;
            }
        }
    }
    panic!("Failed to parse Field value from: {}", debug);
}

/// Check if two f32 values are approximately equal.
fn approx_eq(a: f32, b: f32) -> bool {
    if a.is_nan() && b.is_nan() {
        return true; // Both NaN is considered equal for our purposes
    }
    if a.is_nan() || b.is_nan() {
        return false;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }

    let abs_diff = (a - b).abs();

    // For values near zero, use absolute tolerance
    if a.abs() < ABS_EPSILON && b.abs() < ABS_EPSILON {
        return abs_diff < ABS_EPSILON;
    }

    // Otherwise use relative tolerance
    let max_abs = a.abs().max(b.abs());
    abs_diff / max_abs < EPSILON
}

// ============================================================================
// Test Cases: Each tests a specific optimization pattern
// ============================================================================

proptest! {
    /// Test that basic arithmetic preserves values.
    #[test]
    fn fuzz_basic_arithmetic(x in -100.0f32..100.0, y in -100.0f32..100.0) {
        let add_kernel = kernel!(|| X + Y);
        let sub_kernel = kernel!(|| X - Y);
        let mul_kernel = kernel!(|| X * Y);

        let p = field4(x, y, 0.0, 0.0);

        let add_result = field_val(add_kernel().eval(p));
        let sub_result = field_val(sub_kernel().eval(p));
        let mul_result = field_val(mul_kernel().eval(p));

        prop_assert!(approx_eq(add_result, x + y),
            "add: kernel={} ref={}", add_result, x + y);
        prop_assert!(approx_eq(sub_result, x - y),
            "sub: kernel={} ref={}", sub_result, x - y);
        prop_assert!(approx_eq(mul_result, x * y),
            "mul: kernel={} ref={}", mul_result, x * y);
    }

    /// Test division (avoiding div by zero).
    #[test]
    fn fuzz_division(x in -100.0f32..100.0, y in prop::num::f32::NORMAL.prop_filter("non-zero", |v| v.abs() > 0.001)) {
        let div_kernel = kernel!(|| X / Y);
        let p = field4(x, y, 0.0, 0.0);

        let kernel_result = field_val(div_kernel().eval(p));
        let reference = x / y;

        prop_assert!(approx_eq(kernel_result, reference),
            "div: kernel={} ref={}", kernel_result, reference);
    }

    /// Test sqrt optimization: sqrt(x) for positive x.
    #[test]
    fn fuzz_sqrt(x in 0.001f32..1000.0) {
        let sqrt_kernel = kernel!(|| X.sqrt());
        let p = field4(x, 0.0, 0.0, 0.0);

        let kernel_result = field_val(sqrt_kernel().eval(p));
        let reference = x.sqrt();

        prop_assert!(approx_eq(kernel_result, reference),
            "sqrt: kernel={} ref={}", kernel_result, reference);
    }

    /// Test sqrt of expression (exercises precedence handling).
    #[test]
    fn fuzz_sqrt_of_sum(x in 0.001f32..100.0, y in 0.001f32..100.0) {
        let kernel = kernel!(|| (X + Y).sqrt());
        let p = field4(x, y, 0.0, 0.0);

        let kernel_result = field_val(kernel().eval(p));
        let reference = (x + y).sqrt();

        prop_assert!(approx_eq(kernel_result, reference),
            "sqrt(X+Y): kernel={} ref={} x={} y={}", kernel_result, reference, x, y);
    }

    /// Test FMA fusion: a*b + c should give same result as unfused.
    #[test]
    fn fuzz_fma_pattern(a in -100.0f32..100.0, b in -100.0f32..100.0, c in -100.0f32..100.0) {
        let fma_kernel = kernel!(|| X * Y + Z);
        let p = field4(a, b, c, 0.0);

        let kernel_result = field_val(fma_kernel().eval(p));
        // Reference: unfused multiply-add (may differ by 1 ULP due to FMA)
        let reference = a * b + c;

        prop_assert!(approx_eq(kernel_result, reference),
            "fma: kernel={} ref={}", kernel_result, reference);
    }

    /// Test identity removal: x + 0.0 should equal x.
    #[test]
    fn fuzz_add_zero_identity(x in -1000.0f32..1000.0) {
        let kernel = kernel!(|| X + 0.0);
        let p = field4(x, 0.0, 0.0, 0.0);

        let kernel_result = field_val(kernel().eval(p));

        prop_assert!(approx_eq(kernel_result, x),
            "x+0: kernel={} ref={}", kernel_result, x);
    }

    /// Test identity removal: x * 1.0 should equal x.
    #[test]
    fn fuzz_mul_one_identity(x in -1000.0f32..1000.0) {
        let kernel = kernel!(|| X * 1.0);
        let p = field4(x, 0.0, 0.0, 0.0);

        let kernel_result = field_val(kernel().eval(p));

        prop_assert!(approx_eq(kernel_result, x),
            "x*1: kernel={} ref={}", kernel_result, x);
    }

    /// Test zero propagation: x * 0.0 should equal 0.0.
    #[test]
    fn fuzz_mul_zero_propagation(x in -1000.0f32..1000.0) {
        let kernel = kernel!(|| X * 0.0);
        let p = field4(x, 0.0, 0.0, 0.0);

        let kernel_result = field_val(kernel().eval(p));

        // Note: x * 0.0 can be -0.0 for negative x, but we treat 0.0 == -0.0
        prop_assert!(approx_eq(kernel_result, 0.0),
            "x*0: kernel={} ref=0.0", kernel_result);
    }

    /// Test abs function.
    #[test]
    fn fuzz_abs(x in -1000.0f32..1000.0) {
        let kernel = kernel!(|| X.abs());
        let p = field4(x, 0.0, 0.0, 0.0);

        let kernel_result = field_val(kernel().eval(p));
        let reference = x.abs();

        prop_assert!(approx_eq(kernel_result, reference),
            "abs: kernel={} ref={}", kernel_result, reference);
    }

    /// Test floor function.
    #[test]
    fn fuzz_floor(x in -1000.0f32..1000.0) {
        let kernel = kernel!(|| X.floor());
        let p = field4(x, 0.0, 0.0, 0.0);

        let kernel_result = field_val(kernel().eval(p));
        let reference = x.floor();

        prop_assert!(approx_eq(kernel_result, reference),
            "floor: kernel={} ref={}", kernel_result, reference);
    }

    /// Test negation.
    #[test]
    fn fuzz_neg(x in -1000.0f32..1000.0) {
        let kernel = kernel!(|| (-X));
        let p = field4(x, 0.0, 0.0, 0.0);

        let kernel_result = field_val(kernel().eval(p));
        let reference = -x;

        prop_assert!(approx_eq(kernel_result, reference),
            "neg: kernel={} ref={}", kernel_result, reference);
    }

    /// Test complex expression: distance formula sqrt(x^2 + y^2).
    #[test]
    fn fuzz_distance_2d(x in -100.0f32..100.0, y in -100.0f32..100.0) {
        let kernel = kernel!(|| (X * X + Y * Y).sqrt());
        let p = field4(x, y, 0.0, 0.0);

        let kernel_result = field_val(kernel().eval(p));
        let reference = (x * x + y * y).sqrt();

        prop_assert!(approx_eq(kernel_result, reference),
            "dist2d: kernel={} ref={}", kernel_result, reference);
    }

    /// Test chained operations: (x + y) * z - w.
    #[test]
    fn fuzz_chained_ops(x in -50.0f32..50.0, y in -50.0f32..50.0, z in -50.0f32..50.0, w in -50.0f32..50.0) {
        let kernel = kernel!(|| (X + Y) * Z - W);
        let p = field4(x, y, z, w);

        let kernel_result = field_val(kernel().eval(p));
        let reference = (x + y) * z - w;

        prop_assert!(approx_eq(kernel_result, reference),
            "chain: kernel={} ref={}", kernel_result, reference);
    }

    /// Test kernel with scalar parameter.
    #[test]
    fn fuzz_scalar_param(x in -100.0f32..100.0, param in -100.0f32..100.0) {
        let kernel_factory = kernel!(|offset: f32| X + offset);
        let kernel = kernel_factory(param);
        let p = field4(x, 0.0, 0.0, 0.0);

        let kernel_result = field_val(kernel.eval(p));
        let reference = x + param;

        prop_assert!(approx_eq(kernel_result, reference),
            "scalar_param: kernel={} ref={}", kernel_result, reference);
    }

    /// Test method after binary op (the bug we just fixed).
    #[test]
    fn fuzz_method_after_binop(x in 0.001f32..100.0, offset in 0.001f32..100.0) {
        let kernel_factory = kernel!(|val: f32| (X + val).sqrt());
        let kernel = kernel_factory(offset);
        let p = field4(x, 0.0, 0.0, 0.0);

        let kernel_result = field_val(kernel.eval(p));
        let reference = (x + offset).sqrt();

        prop_assert!(approx_eq(kernel_result, reference),
            "method_after_binop: kernel={} ref={} x={} offset={}",
            kernel_result, reference, x, offset);
    }

    /// Test multiple methods chained after binary ops.
    #[test]
    fn fuzz_chained_methods(x in 0.001f32..100.0, y in 0.001f32..100.0) {
        let kernel = kernel!(|| (X + Y).sqrt().abs());
        let p = field4(x, y, 0.0, 0.0);

        let kernel_result = field_val(kernel().eval(p));
        let reference = (x + y).sqrt().abs();

        prop_assert!(approx_eq(kernel_result, reference),
            "chained_methods: kernel={} ref={}", kernel_result, reference);
    }

    /// Test nested binary ops with methods.
    #[test]
    fn fuzz_nested_binop_methods(x in 0.001f32..50.0, y in 0.001f32..50.0, z in 0.001f32..50.0) {
        let kernel = kernel!(|| (X * Y).sqrt() + Z);
        let p = field4(x, y, z, 0.0);

        let kernel_result = field_val(kernel().eval(p));
        let reference = (x * y).sqrt() + z;

        prop_assert!(approx_eq(kernel_result, reference),
            "nested: kernel={} ref={}", kernel_result, reference);
    }
}

// ============================================================================
// Regression Tests: Specific expressions that have caused bugs
// ============================================================================

#[test]
fn regression_sqrt_with_param() {
    // This was the bug: (X + val).sqrt() was being emitted as X + val.sqrt()
    let kernel_factory = kernel!(|val: f32| (X + val).sqrt());
    let kernel = kernel_factory(7.0);

    let result = field_val(kernel.eval(field4(9.0, 0.0, 0.0, 0.0)));
    let expected = (9.0_f32 + 7.0).sqrt(); // sqrt(16) = 4

    assert!(
        approx_eq(result, expected),
        "sqrt with param: got {} expected {}", result, expected
    );
}

#[test]
fn regression_mul_then_method() {
    // (X * Y).abs() should not become X * Y.abs()
    let kernel = kernel!(|| (X * Y).abs());

    // Test with negative values where the bug would be visible
    let result = field_val(kernel().eval(field4(-3.0, 4.0, 0.0, 0.0)));
    let expected = (-3.0_f32 * 4.0).abs(); // |-12| = 12

    assert!(
        approx_eq(result, expected),
        "mul then abs: got {} expected {}", result, expected
    );
}

#[test]
fn regression_sub_then_method() {
    // (X - Y).floor() should not become X - Y.floor()
    let kernel = kernel!(|| (X - Y).floor());

    let result = field_val(kernel().eval(field4(5.7, 2.3, 0.0, 0.0)));
    let expected = (5.7_f32 - 2.3).floor(); // floor(3.4) = 3

    assert!(
        approx_eq(result, expected),
        "sub then floor: got {} expected {}", result, expected
    );
}
