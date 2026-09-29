//! Fuzz tests for optimization correctness.
//!
//! The optimizer must preserve meaning: `kernel!` saturates an e-graph and
//! extracts a form the source never wrote, and the only thing making that
//! sound is that the new form computes the same number. So the property is
//! checked against an oracle outside the compiler entirely — scalar `f32`
//! arithmetic in Rust, not the IR's own `eval_scalar`, which shares its
//! definitions with the thing under test.
//!
//! Strategy:
//! 1. Define kernel expressions that exercise various optimizations
//! 2. Evaluate each at one point with `Lattice::eval_at` — the one evaluation
//!    entry there is
//! 3. Use proptest to generate random inputs
//! 4. Assert the baked value matches scalar `f32` within epsilon
//!
//! A kernel's parameters are its uniforms (§1.4 of
//! docs/plans/2026-09-25-the-language-is-kernel.md), never constants folded
//! into it, so a property with parameters (`|c: f32| X * Y + c`) is one
//! compiled program, and its random parameters fuzz that program's
//! arguments, not the rewrites a particular constant would enable. Before,
//! each case was a program of its own. The constant-specific rewrites are
//! the properties that spell their constant in the body (`X * 1.0`).

use pixelflow_compiler::kernel;
use pixelflow_core::{Kernel, Lattice};
use proptest::prelude::*;

/// Maximum relative error tolerance for floating-point comparisons.
/// We use a relatively loose epsilon because the optimizer is free to rewrite
/// `sqrt(x)` into the faster `x * rsqrt(x)` form (and to contract FMAs). The
/// hardware `rsqrt` with one Newton step carries ~1-2e-4 relative error, which
/// compounds across nested expressions; a few rsqrt-backed sqrts stay within
/// 5e-4. A semantics-breaking rewrite would miss by orders of magnitude more.
const EPSILON: f32 = 5e-4;

/// Absolute tolerance for values near zero where relative error explodes.
const ABS_EPSILON: f32 = 1e-6;

/// Evaluate a kernel at one point.
fn bake(k: &Kernel, x: f32, y: f32) -> f32 {
    Lattice::eval_at(k, x, y)
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


        let add_result = bake(&add_kernel, x, y);
        let sub_result = bake(&sub_kernel, x, y);
        let mul_result = bake(&mul_kernel, x, y);

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

        let kernel_result = bake(&div_kernel, x, y);
        let reference = x / y;

        prop_assert!(approx_eq(kernel_result, reference),
            "div: kernel={} ref={}", kernel_result, reference);
    }

    /// Test sqrt optimization: sqrt(x) for positive x.
    #[test]
    fn fuzz_sqrt(x in 0.001f32..1000.0) {
        let sqrt_kernel = kernel!(|| X.sqrt());

        let kernel_result = bake(&sqrt_kernel, x, 0.0);
        let reference = x.sqrt();

        prop_assert!(approx_eq(kernel_result, reference),
            "sqrt: kernel={} ref={}", kernel_result, reference);
    }

    /// Test sqrt of expression (exercises precedence handling).
    #[test]
    fn fuzz_sqrt_of_sum(x in 0.001f32..100.0, y in 0.001f32..100.0) {
        let kernel = kernel!(|| (X + Y).sqrt());

        let kernel_result = bake(&kernel, x, y);
        let reference = (x + y).sqrt();

        prop_assert!(approx_eq(kernel_result, reference),
            "sqrt(X+Y): kernel={} ref={} x={} y={}", kernel_result, reference, x, y);
    }

    /// Test FMA fusion: a*b + c should give same result as unfused.
    #[test]
    fn fuzz_fma_pattern(a in -100.0f32..100.0, b in -100.0f32..100.0, c in -100.0f32..100.0) {
        // The addend is a parameter, not a third axis: a lattice has two.
        let fma_kernel = kernel!(|c: f32| X * Y + c)(c);

        let kernel_result = bake(&fma_kernel, a, b);
        // Reference: unfused multiply-add (may differ by 1 ULP due to FMA)
        let reference = a * b + c;

        prop_assert!(approx_eq(kernel_result, reference),
            "fma: kernel={} ref={}", kernel_result, reference);
    }

    /// Test identity removal: x + 0.0 should equal x.
    #[test]
    fn fuzz_add_zero_identity(x in -1000.0f32..1000.0) {
        let kernel = kernel!(|| X + 0.0);

        let kernel_result = bake(&kernel, x, 0.0);

        prop_assert!(approx_eq(kernel_result, x),
            "x+0: kernel={} ref={}", kernel_result, x);
    }

    /// Test identity removal: x * 1.0 should equal x.
    #[test]
    fn fuzz_mul_one_identity(x in -1000.0f32..1000.0) {
        let kernel = kernel!(|| X * 1.0);

        let kernel_result = bake(&kernel, x, 0.0);

        prop_assert!(approx_eq(kernel_result, x),
            "x*1: kernel={} ref={}", kernel_result, x);
    }

    /// Test zero propagation: x * 0.0 should equal 0.0.
    #[test]
    fn fuzz_mul_zero_propagation(x in -1000.0f32..1000.0) {
        let kernel = kernel!(|| X * 0.0);

        let kernel_result = bake(&kernel, x, 0.0);

        // Note: x * 0.0 can be -0.0 for negative x, but we treat 0.0 == -0.0
        prop_assert!(approx_eq(kernel_result, 0.0),
            "x*0: kernel={} ref=0.0", kernel_result);
    }

    /// Test abs function.
    #[test]
    fn fuzz_abs(x in -1000.0f32..1000.0) {
        let kernel = kernel!(|| X.abs());

        let kernel_result = bake(&kernel, x, 0.0);
        let reference = x.abs();

        prop_assert!(approx_eq(kernel_result, reference),
            "abs: kernel={} ref={}", kernel_result, reference);
    }

    /// Test floor function.
    #[test]
    fn fuzz_floor(x in -1000.0f32..1000.0) {
        let kernel = kernel!(|| X.floor());

        let kernel_result = bake(&kernel, x, 0.0);
        let reference = x.floor();

        prop_assert!(approx_eq(kernel_result, reference),
            "floor: kernel={} ref={}", kernel_result, reference);
    }

    /// Test negation.
    #[test]
    fn fuzz_neg(x in -1000.0f32..1000.0) {
        let kernel = kernel!(|| (-X));

        let kernel_result = bake(&kernel, x, 0.0);
        let reference = -x;

        prop_assert!(approx_eq(kernel_result, reference),
            "neg: kernel={} ref={}", kernel_result, reference);
    }

    /// Test complex expression: distance formula sqrt(x^2 + y^2).
    #[test]
    fn fuzz_distance_2d(x in -100.0f32..100.0, y in -100.0f32..100.0) {
        let kernel = kernel!(|| (X * X + Y * Y).sqrt());

        let kernel_result = bake(&kernel, x, y);
        let reference = (x * x + y * y).sqrt();

        prop_assert!(approx_eq(kernel_result, reference),
            "dist2d: kernel={} ref={}", kernel_result, reference);
    }

    /// Test chained operations: (x + y) * z - w.
    #[test]
    fn fuzz_chained_ops(x in -50.0f32..50.0, y in -50.0f32..50.0, z in -50.0f32..50.0, w in -50.0f32..50.0) {
        let kernel = kernel!(|z: f32, w: f32| (X + Y) * z - w)(z, w);

        let kernel_result = bake(&kernel, x, y);
        let reference = (x + y) * z - w;

        prop_assert!(approx_eq(kernel_result, reference),
            "chain: kernel={} ref={}", kernel_result, reference);
    }

    /// Test kernel with scalar parameter.
    #[test]
    fn fuzz_scalar_param(x in -100.0f32..100.0, param in -100.0f32..100.0) {
        let kernel_factory = kernel!(|offset: f32| X + offset);
        let kernel = kernel_factory(param);

        let kernel_result = bake(&kernel, x, 0.0);
        let reference = x + param;

        prop_assert!(approx_eq(kernel_result, reference),
            "scalar_param: kernel={} ref={}", kernel_result, reference);
    }

    /// Test method after binary op (the bug we just fixed).
    #[test]
    fn fuzz_method_after_binop(x in 0.001f32..100.0, offset in 0.001f32..100.0) {
        let kernel_factory = kernel!(|val: f32| (X + val).sqrt());
        let kernel = kernel_factory(offset);

        let kernel_result = bake(&kernel, x, 0.0);
        let reference = (x + offset).sqrt();

        prop_assert!(approx_eq(kernel_result, reference),
            "method_after_binop: kernel={} ref={} x={} offset={}",
            kernel_result, reference, x, offset);
    }

    /// Test multiple methods chained after binary ops.
    #[test]
    fn fuzz_chained_methods(x in 0.001f32..100.0, y in 0.001f32..100.0) {
        let kernel = kernel!(|| (X + Y).sqrt().abs());

        let kernel_result = bake(&kernel, x, y);
        let reference = (x + y).sqrt().abs();

        prop_assert!(approx_eq(kernel_result, reference),
            "chained_methods: kernel={} ref={}", kernel_result, reference);
    }

    /// Test nested binary ops with methods.
    #[test]
    fn fuzz_nested_binop_methods(x in 0.001f32..50.0, y in 0.001f32..50.0, z in 0.001f32..50.0) {
        let kernel = kernel!(|z: f32| (X * Y).sqrt() + z)(z);

        let kernel_result = bake(&kernel, x, y);
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

    let result = bake(&kernel, 9.0, 0.0);
    let expected = (9.0_f32 + 7.0).sqrt(); // sqrt(16) = 4

    assert!(
        approx_eq(result, expected),
        "sqrt with param: got {} expected {}",
        result,
        expected
    );
}

#[test]
fn regression_mul_then_method() {
    // (X * Y).abs() should not become X * Y.abs()
    let kernel = kernel!(|| (X * Y).abs());

    // Test with negative values where the bug would be visible
    let result = bake(&kernel, -3.0, 4.0);
    let expected = (-3.0_f32 * 4.0).abs(); // |-12| = 12

    assert!(
        approx_eq(result, expected),
        "mul then abs: got {} expected {}",
        result,
        expected
    );
}

#[test]
fn regression_sub_then_method() {
    // (X - Y).floor() should not become X - Y.floor()
    let kernel = kernel!(|| (X - Y).floor());

    let result = bake(&kernel, 5.7, 2.3);
    let expected = (5.7_f32 - 2.3).floor(); // floor(3.4) = 3

    assert!(
        approx_eq(result, expected),
        "sub then floor: got {} expected {}",
        result,
        expected
    );
}
