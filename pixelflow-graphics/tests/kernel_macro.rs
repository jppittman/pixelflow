//! Integration tests for the kernel! macro.
//!
//! These tests verify that the macro-generated kernels work correctly with
//! the full PixelFlow stack: ManifoldExt, rasterization, and SIMD evaluation.
//!
//! ## Architecture
//!
//! The kernel! macro generates:
//! 1. A struct holding captured parameters (the environment)
//! 2. A ZST expression tree using Var<N> for parameter references (the code)
//! 3. Nested Let::new() bindings that extend the domain with parameter values
//!
//! Parameters use Peano-encoded stack indices:
//! - First param (index 0) → Var<N{n-1}> (deepest in stack)
//! - Last param (index n-1) → Var<N0> (head of stack)
//!
//! This enables **unlimited parameters** (up to 8), compared to the old
//! Z/W coordinate slot approach which was limited to 2 parameters.
//!
//! ## Testing Strategy
//!
//! Since Field is a SIMD type (multiple f32 lanes), we can't extract single
//! values. Instead we test using Field-level comparisons and check that all
//! lanes satisfy the expected condition.

use pixelflow_core::jet::Jet3;
use pixelflow_core::{Field, Manifold, ManifoldExt};
use pixelflow_compiler::kernel;

type Field4 = (Field, Field, Field, Field);

/// Helper: Convert f32 tuple to Field4
fn field4(x: f32, y: f32, z: f32, w: f32) -> Field4 {
    (
        Field::from(x),
        Field::from(y),
        Field::from(z),
        Field::from(w),
    )
}

/// Helper: Check if two Fields are approximately equal across all lanes.
///
/// Since Field operators create AST nodes (not immediate values), we:
/// 1. Build the comparison expression as an AST
/// 2. Evaluate it using `.constant()` to get a concrete Field
/// 3. Use Field's native `.all()` to check all lanes
fn fields_close(a: Field, b: Field, epsilon: f32) -> bool {
    // Build AST: Abs<Sub<Field, Field>>
    let diff_ast = (a - b).abs();
    // Evaluate at origin to collapse AST → Field
    let diff_field = diff_ast.constant();
    let eps = Field::from(epsilon);
    // Now use Field's native lt (returns mask-as-Field)
    Field::lt(diff_field, eps).all()
}

/// Test a kernel with one parameter.
#[test]
fn test_one_param_kernel() {
    // X offset by a parameter
    let offset_x = kernel!(|dx: f32| X + dx);
    let k = offset_x(10.0);

    // At x=5: result should be 5 + 10 = 15
    let result = k.eval(field4(5.0, 0.0, 0.0, 0.0));
    let expected = Field::from(15.0);
    assert!(
        fields_close(result, expected, 0.001),
        "expected 15, got different value"
    );
}

/// Test a kernel with two parameters.
#[test]
fn test_two_param_kernel() {
    // Offset both X and Y
    let offset_xy = kernel!(|dx: f32, dy: f32| (X + dx) + (Y + dy));
    let k = offset_xy(10.0, 20.0);

    // At (5, 3): result should be (5+10) + (3+20) = 38
    let result = k.eval(field4(5.0, 3.0, 0.0, 0.0));
    let expected = Field::from(38.0);
    assert!(
        fields_close(result, expected, 0.001),
        "expected 38, got different value"
    );
}

/// Test a kernel with no parameters.
#[test]
fn test_zero_param_kernel() {
    // Simple distance from origin (no params)
    let dist = kernel!(|| (X * X + Y * Y).sqrt());
    let k = dist();

    // At (3, 4): distance = 5
    let result = k.eval(field4(3.0, 4.0, 0.0, 0.0));
    let expected = Field::from(5.0);
    assert!(
        fields_close(result, expected, 0.001),
        "expected 5, got different value"
    );
}

/// Test method chaining (ManifoldExt integration).
#[test]
fn test_method_chaining() {
    // Clamped value using .max().min()
    let clamp = kernel!(|lo: f32, hi: f32| X.max(lo).min(hi));
    let k = clamp(0.0, 1.0);

    // Below range: clamp(-5) should be 0
    let below = k.eval(field4(-5.0, 0.0, 0.0, 0.0));
    assert!(
        fields_close(below, Field::from(0.0), 0.001),
        "clamp below failed"
    );

    // In range: clamp(0.5) should be 0.5
    let middle = k.eval(field4(0.5, 0.0, 0.0, 0.0));
    assert!(
        fields_close(middle, Field::from(0.5), 0.001),
        "clamp middle failed"
    );

    // Above range: clamp(5) should be 1
    let above = k.eval(field4(5.0, 0.0, 0.0, 0.0));
    assert!(
        fields_close(above, Field::from(1.0), 0.001),
        "clamp above failed"
    );
}

/// Test that kernels are Clone (not Copy, since they hold data).
#[test]
fn test_kernel_is_clone() {
    let scale = kernel!(|factor: f32| X * factor);
    let k1 = scale(2.0);
    let k2 = k1.clone();

    let r1 = k1.eval(field4(5.0, 0.0, 0.0, 0.0));
    let r2 = k2.eval(field4(5.0, 0.0, 0.0, 0.0));

    assert!(fields_close(r1, Field::from(10.0), 0.001));
    assert!(fields_close(r2, Field::from(10.0), 0.001));
}

/// Test that different instantiations are independent.
#[test]
fn test_independent_instantiations() {
    let scale = kernel!(|factor: f32| X * factor);

    let double = scale(2.0);
    let triple = scale(3.0);

    let r_double = double.eval(field4(5.0, 0.0, 0.0, 0.0));
    let r_triple = triple.eval(field4(5.0, 0.0, 0.0, 0.0));

    assert!(
        fields_close(r_double, Field::from(10.0), 0.001),
        "5 * 2 = 10"
    );
    assert!(
        fields_close(r_triple, Field::from(15.0), 0.001),
        "5 * 3 = 15"
    );
}

/// Test sqrt method.
#[test]
fn test_sqrt() {
    let root = kernel!(|val: f32| (X + val).sqrt());
    let k = root(7.0);

    // sqrt(9 + 7) = sqrt(16) = 4
    let result = k.eval(field4(9.0, 0.0, 0.0, 0.0));
    assert!(fields_close(result, Field::from(4.0), 0.001));
}

/// Test floor method.
#[test]
fn test_floor() {
    let floored = kernel!(|| X.floor());
    let k = floored();

    let result = k.eval(field4(3.7, 0.0, 0.0, 0.0));
    assert!(fields_close(result, Field::from(3.0), 0.001));

    let negative = k.eval(field4(-1.3, 0.0, 0.0, 0.0));
    assert!(fields_close(negative, Field::from(-2.0), 0.001));
}

/// Test abs method.
#[test]
fn test_abs() {
    let absolute = kernel!(|offset: f32| (X - offset).abs());
    let k = absolute(5.0);

    // |3 - 5| = 2
    let result = k.eval(field4(3.0, 0.0, 0.0, 0.0));
    assert!(fields_close(result, Field::from(2.0), 0.001));

    // |7 - 5| = 2
    let result2 = k.eval(field4(7.0, 0.0, 0.0, 0.0));
    assert!(fields_close(result2, Field::from(2.0), 0.001));
}

/// Test that the generated expression tree is ZST-based (Copy).
/// This is verified by the fact that the kernel compiles at all -
/// if parameters were injected directly, the expression wouldn't be Copy.
#[test]
fn test_zst_expression_is_copy() {
    // This kernel uses the parameter twice in the expression.
    // If the expression weren't Copy (ZST-based), this wouldn't compile
    // because the parameter would be moved on first use.
    let square_offset = kernel!(|d: f32| (X - d) * (X - d) + (Y - d) * (Y - d));
    let k = square_offset(1.0);

    // (3-1)² + (4-1)² = 4 + 9 = 13
    let result = k.eval(field4(3.0, 4.0, 0.0, 0.0));
    assert!(fields_close(result, Field::from(13.0), 0.001));
}

// ============================================================================
// Tests for >2 parameters (using Let/Var binding system)
// ============================================================================

/// Test a kernel with three parameters.
/// This demonstrates the new Let/Var binding system since Z/W slots only support 2.
#[test]
fn test_three_param_kernel() {
    // Translate point by (dx, dy) and add z offset
    let translate_3d = kernel!(|dx: f32, dy: f32, dz: f32| (X + dx) + (Y + dy) + dz);
    let k = translate_3d(10.0, 20.0, 30.0);

    // At (5, 3): result should be (5+10) + (3+20) + 30 = 68
    let result = k.eval(field4(5.0, 3.0, 0.0, 0.0));
    let expected = Field::from(68.0);
    assert!(
        fields_close(result, expected, 0.001),
        "expected 68, got different value"
    );
}

/// Test a kernel with four parameters.
/// Demonstrates full 4-parameter support.
#[test]
fn test_four_param_kernel() {
    // Combine all four parameters with coordinates
    let quad_combine = kernel!(|a: f32, b: f32, c: f32, d: f32| a + b + c + d + X + Y);
    let k = quad_combine(1.0, 2.0, 3.0, 4.0);

    // At (5, 6): result should be 1 + 2 + 3 + 4 + 5 + 6 = 21
    let result = k.eval(field4(5.0, 6.0, 0.0, 0.0));
    let expected = Field::from(21.0);
    assert!(
        fields_close(result, expected, 0.001),
        "expected 21, got different value"
    );
}

/// Test a 4-parameter sphere SDF kernel.
/// This is a practical example: signed distance from a sphere at (cx, cy, cz) with radius r.
#[test]
fn test_sphere_sdf_kernel() {
    // Sphere SDF: distance from center minus radius
    let sphere_sdf = kernel!(|cx: f32, cy: f32, cz: f32, r: f32| {
        let dx = X - cx;
        let dy = Y - cy;
        let dz = Z - cz;
        (dx * dx + dy * dy + dz * dz).sqrt() - r
    });

    // Sphere at (2, 3, 4) with radius 5
    let k = sphere_sdf(2.0, 3.0, 4.0, 5.0);

    // Point at origin (0, 0, 0): distance = sqrt(4 + 9 + 16) - 5 = sqrt(29) - 5 ≈ 0.385
    let result = k.eval(field4(0.0, 0.0, 0.0, 0.0));
    let expected = Field::from((29.0f32).sqrt() - 5.0);
    assert!(
        fields_close(result, expected, 0.01),
        "sphere SDF at origin should be ~0.385"
    );

    // Point at (2, 3, 4) (center): distance = 0 - 5 = -5
    let result_center = k.eval(field4(2.0, 3.0, 4.0, 0.0));
    let expected_center = Field::from(-5.0);
    assert!(
        fields_close(result_center, expected_center, 0.001),
        "sphere SDF at center should be -5"
    );
}

/// Test parameter ordering with 3 parameters.
/// Verifies that parameters are correctly bound (first param deepest in stack).
#[test]
fn test_parameter_ordering_three() {
    // Each parameter has a different multiplier to verify correct binding
    let order_test = kernel!(|a: f32, b: f32, c: f32| a * 100.0 + b * 10.0 + c);
    let k = order_test(1.0, 2.0, 3.0);

    // Result should be 1*100 + 2*10 + 3 = 123
    let result = k.eval(field4(0.0, 0.0, 0.0, 0.0));
    let expected = Field::from(123.0);
    assert!(
        fields_close(result, expected, 0.001),
        "expected 123 (a=1, b=2, c=3)"
    );
}

/// Test that parameters can be used multiple times in 3+ param kernels.
#[test]
fn test_param_reuse_three() {
    // Use each parameter twice
    let reuse = kernel!(|a: f32, b: f32, c: f32| (a + a) + (b + b) + (c + c));
    let k = reuse(1.0, 2.0, 3.0);

    // Result should be 2 + 4 + 6 = 12
    let result = k.eval(field4(0.0, 0.0, 0.0, 0.0));
    let expected = Field::from(12.0);
    assert!(
        fields_close(result, expected, 0.001),
        "expected 12 (each param doubled)"
    );
}

// ============================================================================
// Tests for Jet3 output (automatic differentiation)
// ============================================================================

type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);

/// Helper: Convert f32 tuple to Jet3_4 (constant jets)
fn jet3_4(x: f32, y: f32, z: f32, w: f32) -> Jet3_4 {
    (
        Jet3::constant(Field::from(x)),
        Jet3::constant(Field::from(y)),
        Jet3::constant(Field::from(z)),
        Jet3::constant(Field::from(w)),
    )
}

/// Test a simple Jet3 kernel (domain inferred from output type).
#[test]
fn test_jet3_simple() {
    // X + Y with Jet3 output → domain is Jet3_4
    let add_xy = kernel!(|| -> Jet3 X + Y);
    let k = add_xy();

    // At (3, 4, 0, 0): result should be 7
    let result = k.eval(jet3_4(3.0, 4.0, 0.0, 0.0));
    let expected = Jet3::constant(Field::from(7.0));

    // Compare the val component
    let diff = (result.val - expected.val).abs();
    let eps = Field::from(0.001);
    assert!(
        Field::lt(diff.constant(), eps).all(),
        "Jet3 X + Y at (3,4) should be 7"
    );
}

/// Test Jet3 kernel with parameters (sphere SDF).
#[test]
fn test_jet3_sphere_sdf() {
    // Sphere SDF: distance from center minus radius
    // Using Jet3 for automatic differentiation (normals)
    let sphere_sdf = kernel!(|cx: f32, cy: f32, cz: f32, r: f32| -> Jet3 {
        let dx = X - cx;
        let dy = Y - cy;
        let dz = Z - cz;
        (dx * dx + dy * dy + dz * dz).sqrt() - r
    });

    // Sphere at (2, 3, 4) with radius 5
    let k = sphere_sdf(2.0, 3.0, 4.0, 5.0);

    // Point at origin (0, 0, 0): distance = sqrt(4 + 9 + 16) - 5 = sqrt(29) - 5 ≈ 0.385
    let result = k.eval(jet3_4(0.0, 0.0, 0.0, 0.0));
    let expected_val = (29.0f32).sqrt() - 5.0;
    let expected = Jet3::constant(Field::from(expected_val));

    let diff = (result.val - expected.val).abs();
    let eps = Field::from(0.01);
    assert!(
        Field::lt(diff.constant(), eps).all(),
        "Jet3 sphere SDF at origin should be ~0.385"
    );
}

// ============================================================================
// Tests for kernel composition (manifold parameters)
// ============================================================================

/// Test basic kernel composition with a single manifold parameter.
/// This is the core use case: composing a distance function with a circle SDF.
#[test]
fn test_simple_kernel_composition() {
    // Distance from a point (parametric)
    let dist = kernel!(|cx: f32, cy: f32| {
        let dx = X - cx;
        let dy = Y - cy;
        (dx * dx + dy * dy).sqrt()
    });

    // Circle SDF: takes a distance manifold and subtracts radius
    let circle = kernel!(|inner: kernel, r: f32| inner - r);

    // Compose: circle centered at origin with radius 1
    let c = circle(dist(0.0, 0.0), 1.0);

    // At (2, 0): distance from origin is 2, minus radius 1 = 1
    let result = c.eval(field4(2.0, 0.0, 0.0, 0.0));
    let expected = Field::from(1.0);
    assert!(
        fields_close(result, expected, 0.001),
        "circle SDF at (2,0) should be 1.0"
    );

    // At (0, 0): distance from origin is 0, minus radius 1 = -1 (inside)
    let result_center = c.eval(field4(0.0, 0.0, 0.0, 0.0));
    let expected_center = Field::from(-1.0);
    assert!(
        fields_close(result_center, expected_center, 0.001),
        "circle SDF at center should be -1.0"
    );
}

/// Test kernel composition with offset centers.
#[test]
fn test_kernel_composition_with_offset() {
    let dist = kernel!(|cx: f32, cy: f32| {
        let dx = X - cx;
        let dy = Y - cy;
        (dx * dx + dy * dy).sqrt()
    });

    let circle = kernel!(|inner: kernel, r: f32| inner - r);

    // Circle at (3, 4) with radius 5
    let c = circle(dist(3.0, 4.0), 5.0);

    // At origin (0, 0): distance from (3,4) is 5, minus radius 5 = 0 (on surface)
    let result = c.eval(field4(0.0, 0.0, 0.0, 0.0));
    let expected = Field::from(0.0);
    assert!(
        fields_close(result, expected, 0.001),
        "circle SDF at origin should be 0 (on surface)"
    );

    // At (3, 4): center, SDF = -5
    let result_center = c.eval(field4(3.0, 4.0, 0.0, 0.0));
    let expected_center = Field::from(-5.0);
    assert!(
        fields_close(result_center, expected_center, 0.001),
        "circle SDF at center should be -5"
    );
}

/// Test multiple manifold parameters (SDF union).
///
/// TODO: Two-manifold case needs ManifoldBind2 or similar for type inference.
/// Currently uses Computed fallback which breaks type inference.
/// The pattern `|a: kernel, b: kernel| a.min(b)` requires either:
/// 1. ManifoldBind2<M1, M2, Body> that carries both manifold types
/// 2. Explicit type annotations in the generated closure
/// 3. A different codegen strategy (e.g., leveled evaluation)
#[test]
#[ignore = "two-manifold params require ManifoldBind2 implementation"]
fn test_two_manifold_params() {
    // Test body commented out until ManifoldBind2 is implemented
    // See the original test for the intended behavior:
    //
    // let circle_sdf = kernel!(|cx: f32, cy: f32, r: f32| { ... });
    // let sdf_union = kernel!(|a: kernel, b: kernel| a.min(b));
    // let union = sdf_union(circle1, circle2);
}

/// Test mixed manifold and scalar parameters.
#[test]
fn test_mixed_manifold_scalar_params() {
    // Scale an SDF by a factor
    let scale_sdf = kernel!(|inner: kernel, factor: f32| inner * factor);

    // Simple distance from origin
    let dist = kernel!(|| (X * X + Y * Y).sqrt());

    // Scale the distance by 2
    let scaled = scale_sdf(dist(), 2.0);

    // At (3, 4): distance = 5, scaled = 10
    let result = scaled.eval(field4(3.0, 4.0, 0.0, 0.0));
    assert!(
        fields_close(result, Field::from(10.0), 0.001),
        "scaled distance at (3,4) should be 10"
    );
}

/// Test chained kernel composition (three levels deep).
#[test]
fn test_chained_composition() {
    // Basic X coordinate
    let get_x = kernel!(|| X);

    // Add a constant to a manifold
    let add_const = kernel!(|inner: kernel, val: f32| inner + val);

    // Multiply a manifold by a constant
    let mul_const = kernel!(|inner: kernel, val: f32| inner * val);

    // Chain: (X + 5) * 2
    let composed = mul_const(add_const(get_x(), 5.0), 2.0);

    // At x=3: (3 + 5) * 2 = 16
    let result = composed.eval(field4(3.0, 0.0, 0.0, 0.0));
    assert!(
        fields_close(result, Field::from(16.0), 0.001),
        "(3 + 5) * 2 should be 16"
    );
}

/// Test that composed kernels can be cloned (the inner kernel is owned).
#[test]
fn test_composed_kernel_ownership() {
    let dist = kernel!(|cx: f32, cy: f32| {
        let dx = X - cx;
        let dy = Y - cy;
        (dx * dx + dy * dy).sqrt()
    });

    let circle = kernel!(|inner: kernel, r: f32| inner - r);

    // Create a composed kernel
    let c = circle(dist(0.0, 0.0), 1.0);

    // Evaluate multiple times (the kernel is borrowed, not moved)
    let r1 = c.eval(field4(2.0, 0.0, 0.0, 0.0));
    let r2 = c.eval(field4(0.0, 2.0, 0.0, 0.0));

    assert!(fields_close(r1, Field::from(1.0), 0.001));
    assert!(fields_close(r2, Field::from(1.0), 0.001));
}

/// Test Jet3 kernel with numeric literals in expressions.
/// This tests the annotation-based literal handling for Jet domains.
/// Previously, literals like `1.0` would be wrapped as `Jet3::constant(...)`,
/// creating type mismatches with ZST expression tree types.
/// With the annotation pass, literals become Var<N> references bound via Let.
#[test]
fn test_jet3_with_literals() {
    // Inverse square root with literal numerator
    // This expression: 1.0 / (X*X + Y*Y + Z*Z).sqrt()
    // Tests that the literal 1.0 is properly handled in Jet3 mode
    let inv_sqrt = kernel!(|| -> Jet3 {
        let len_sq = X * X + Y * Y + Z * Z;
        1.0 / len_sq.sqrt()
    });
    let k = inv_sqrt();

    // At (1, 0, 0): 1.0 / sqrt(1) = 1.0
    let result = k.eval(jet3_4(1.0, 0.0, 0.0, 0.0));
    let expected = Jet3::constant(Field::from(1.0));
    let diff = (result.val - expected.val).abs();
    let eps = Field::from(0.01);
    assert!(
        Field::lt(diff.constant(), eps).all(),
        "1.0/sqrt(1) should be 1.0"
    );

    // At (2, 0, 0): 1.0 / sqrt(4) = 0.5
    let result2 = k.eval(jet3_4(2.0, 0.0, 0.0, 0.0));
    let expected2 = Jet3::constant(Field::from(0.5));
    let diff2 = (result2.val - expected2.val).abs();
    assert!(
        Field::lt(diff2.constant(), eps).all(),
        "1.0/sqrt(4) should be 0.5"
    );
}

/// Test Jet3 kernel with multiple literals.
/// Verifies that multiple literals each get their own Var<N> binding.
#[test]
fn test_jet3_multiple_literals() {
    // Expression with multiple literals: 2.0 * X + 3.0
    let affine = kernel!(|| -> Jet3 2.0 * X + 3.0);
    let k = affine();

    // At x=5: 2.0 * 5 + 3.0 = 13.0
    let result = k.eval(jet3_4(5.0, 0.0, 0.0, 0.0));
    let expected = Jet3::constant(Field::from(13.0));
    let diff = (result.val - expected.val).abs();
    let eps = Field::from(0.01);
    assert!(
        Field::lt(diff.constant(), eps).all(),
        "2.0 * 5 + 3.0 should be 13.0"
    );
}

/// Test Jet3 kernel with literals AND parameters.
/// Both literals and params become Var<N> - they should coexist correctly.
#[test]
fn test_jet3_literals_and_params() {
    // offset + 2.0 * X - 0.5
    let kernel = kernel!(|offset: f32| -> Jet3 offset + 2.0 * X - 0.5);
    let k = kernel(10.0);

    // At x=3: 10.0 + 2.0*3 - 0.5 = 10.0 + 6.0 - 0.5 = 15.5
    let result = k.eval(jet3_4(3.0, 0.0, 0.0, 0.0));
    let expected = Jet3::constant(Field::from(15.5));
    let diff = (result.val - expected.val).abs();
    let eps = Field::from(0.01);
    assert!(
        Field::lt(diff.constant(), eps).all(),
        "10.0 + 2.0*3 - 0.5 should be 15.5"
    );
}

// ============================================================================
// Tests for Fused Derivative Combinators
// ============================================================================
//
// Fused combinators evaluate the inner manifold ONCE and extract derived quantities:
// - GradientMag2D(m) → √(dx² + dy²)
// - GradientMag3D(m) → √(dx² + dy² + dz²)
// - Antialias2D(m)   → val / √(dx² + dy²)
// - Antialias3D(m)   → val / √(dx² + dy² + dz²)
// - Normalized2D(m)  → (dx, dy) / √(dx² + dy²)
// - Normalized3D(m)  → (dx, dy, dz) / √(dx² + dy² + dz²)

use pixelflow_core::jet::Jet2;
use pixelflow_core::{GradientMag2D, GradientMag3D, Antialias2D, Antialias3D, Normalized2D};

type Jet2_4 = (Jet2, Jet2, Jet2, Jet2);

/// Helper: Create Jet2_4 with proper derivative seeds
fn jet2_4_seeded(x: f32, y: f32) -> Jet2_4 {
    (
        Jet2::x(Field::from(x)),
        Jet2::y(Field::from(y)),
        Jet2::constant(Field::from(0.0)),
        Jet2::constant(Field::from(0.0)),
    )
}

/// Helper: Create Jet3_4 with proper derivative seeds
fn jet3_4_seeded(x: f32, y: f32, z: f32) -> Jet3_4 {
    (
        Jet3::x(Field::from(x)),
        Jet3::y(Field::from(y)),
        Jet3::z(Field::from(z)),
        Jet3::constant(Field::from(0.0)),
    )
}

/// Test GradientMag2D computes √(dx² + dy²) with single eval.
#[test]
fn test_gradient_mag_2d() {
    // For f(x,y) = sqrt(x² + y²), the gradient is (x/r, y/r) where r = sqrt(x²+y²)
    // Gradient magnitude is always 1.0 for distance fields
    let dist = (pixelflow_core::X * pixelflow_core::X
        + pixelflow_core::Y * pixelflow_core::Y)
        .sqrt();

    let grad_mag = GradientMag2D(dist);

    // At (3, 4): r = 5, gradient = (3/5, 4/5), magnitude = 1.0
    let result = grad_mag.eval(jet2_4_seeded(3.0, 4.0));
    assert!(
        fields_close(result, Field::from(1.0), 0.01),
        "GradientMag2D at (3,4) should be 1.0"
    );
}

/// Test GradientMag3D computes √(dx² + dy² + dz²) with single eval.
#[test]
fn test_gradient_mag_3d() {
    // For f(x,y,z) = sqrt(x² + y² + z²), gradient magnitude is 1.0
    let dist = (pixelflow_core::X * pixelflow_core::X
        + pixelflow_core::Y * pixelflow_core::Y
        + pixelflow_core::Z * pixelflow_core::Z)
        .sqrt();

    let grad_mag = GradientMag3D(dist);

    // At (1, 2, 2): r = 3, gradient magnitude = 1.0
    let result = grad_mag.eval(jet3_4_seeded(1.0, 2.0, 2.0));
    assert!(
        fields_close(result, Field::from(1.0), 0.01),
        "GradientMag3D at (1,2,2) should be 1.0"
    );
}

/// Test Antialias2D computes val / √(dx² + dy²) with single eval.
#[test]
fn test_antialias_2d() {
    // Circle SDF using kernel! macro (handles literal promotion)
    // At (2, 0): val = 1.0, gradient = (1, 0), so antialias = 1.0 / 1.0 = 1.0
    let circle_sdf = kernel!(|| -> Jet2 {
        (X * X + Y * Y).sqrt() - 1.0
    });
    let sdf = circle_sdf();

    let aa = Antialias2D(sdf);

    let result = aa.eval(jet2_4_seeded(2.0, 0.0));
    assert!(
        fields_close(result, Field::from(1.0), 0.01),
        "Antialias2D at (2,0) for circle SDF should be 1.0"
    );
}

/// Test Antialias3D computes val / √(dx² + dy² + dz²) with single eval.
#[test]
fn test_antialias_3d() {
    // Sphere SDF using kernel! macro
    // At (2, 0, 0): val = 1.0, gradient = (1, 0, 0), so antialias = 1.0 / 1.0 = 1.0
    let sphere_sdf = kernel!(|| -> Jet3 {
        (X * X + Y * Y + Z * Z).sqrt() - 1.0
    });
    let sdf = sphere_sdf();

    let aa = Antialias3D(sdf);

    let result = aa.eval(jet3_4_seeded(2.0, 0.0, 0.0));
    assert!(
        fields_close(result, Field::from(1.0), 0.01),
        "Antialias3D at (2,0,0) for sphere SDF should be 1.0"
    );
}

/// Test Normalized2D returns unit gradient vector with single eval.
#[test]
fn test_normalized_2d() {
    // Distance field: sqrt(x² + y²)
    // At (3, 4): gradient = (3/5, 4/5) = (0.6, 0.8)
    let dist = (pixelflow_core::X * pixelflow_core::X
        + pixelflow_core::Y * pixelflow_core::Y)
        .sqrt();

    let normal = Normalized2D(dist);

    let (nx, ny) = normal.eval(jet2_4_seeded(3.0, 4.0));
    assert!(
        fields_close(nx, Field::from(0.6), 0.01),
        "Normalized2D.x at (3,4) should be 0.6"
    );
    assert!(
        fields_close(ny, Field::from(0.8), 0.01),
        "Normalized2D.y at (3,4) should be 0.8"
    );
}

/// Test fused combinators with kernel-composed manifolds.
#[test]
fn test_fused_combinators_with_kernel_composition() {
    // Create a circle SDF kernel
    let circle_sdf = kernel!(|cx: f32, cy: f32, r: f32| -> Jet2 {
        let dx = X - cx;
        let dy = Y - cy;
        (dx * dx + dy * dy).sqrt() - r
    });

    // Instantiate: circle at (0, 0) with radius 1
    let sdf = circle_sdf(0.0, 0.0, 1.0);

    // Use fused combinators - evaluates sdf ONCE per combinator
    let grad_mag = GradientMag2D(sdf.clone());
    let aa = Antialias2D(sdf);

    // At (2, 0): gradient magnitude = 1.0, antialias = 1.0 / 1.0 = 1.0
    let grad_result = grad_mag.eval(jet2_4_seeded(2.0, 0.0));
    let aa_result = aa.eval(jet2_4_seeded(2.0, 0.0));

    assert!(
        fields_close(grad_result, Field::from(1.0), 0.01),
        "GradientMag2D(circle_sdf) at (2,0) should be 1.0"
    );
    assert!(
        fields_close(aa_result, Field::from(1.0), 0.01),
        "Antialias2D(circle_sdf) at (2,0) should be 1.0"
    );
}

// ============================================================================
// Tests for Simple Derivative Accessor Combinators
// ============================================================================
//
// Simple accessors extract individual Jet components:
// - V(m)   → val  (the function value)
// - DX(m)  → dx   (∂f/∂x)
// - DY(m)  → dy   (∂f/∂y)
// - DZ(m)  → dz   (∂f/∂z, Jet3 only)
//
// **Design Note**: These are EXTRACTORS for individual components.
// For composed operations (gradient magnitude, antialiasing), use the
// FUSED COMBINATORS (GradientMag2D, Antialias2D, etc.) which evaluate
// the inner manifold once and compute derived quantities efficiently.

use pixelflow_core::{V, DX, DY, DZ};

/// Test V accessor extracts the value component from Jet2.
#[test]
fn test_v_accessor() {
    // Distance from origin: sqrt(x² + y²)
    let dist = (pixelflow_core::X * pixelflow_core::X
        + pixelflow_core::Y * pixelflow_core::Y)
        .sqrt();

    // Extract just the value
    let val_only = V(dist);

    // At (3, 4): distance = 5
    let result = val_only.eval(jet2_4_seeded(3.0, 4.0));
    assert!(
        fields_close(result, Field::from(5.0), 0.01),
        "V(dist) at (3,4) should be 5.0"
    );
}

/// Test DX accessor extracts ∂f/∂x from Jet2.
#[test]
fn test_dx_accessor() {
    // Distance from origin: sqrt(x² + y²)
    // ∂dist/∂x = x / sqrt(x² + y²) = x / dist
    let dist = (pixelflow_core::X * pixelflow_core::X
        + pixelflow_core::Y * pixelflow_core::Y)
        .sqrt();

    // Extract ∂f/∂x
    let dx_only = DX(dist);

    // At (3, 4): dist = 5, ∂dist/∂x = 3/5 = 0.6
    let result = dx_only.eval(jet2_4_seeded(3.0, 4.0));
    assert!(
        fields_close(result, Field::from(0.6), 0.01),
        "DX(dist) at (3,4) should be 0.6"
    );
}

/// Test DY accessor extracts ∂f/∂y from Jet2.
#[test]
fn test_dy_accessor() {
    // Distance from origin: sqrt(x² + y²)
    // ∂dist/∂y = y / sqrt(x² + y²) = y / dist
    let dist = (pixelflow_core::X * pixelflow_core::X
        + pixelflow_core::Y * pixelflow_core::Y)
        .sqrt();

    // Extract ∂f/∂y
    let dy_only = DY(dist);

    // At (3, 4): dist = 5, ∂dist/∂y = 4/5 = 0.8
    let result = dy_only.eval(jet2_4_seeded(3.0, 4.0));
    assert!(
        fields_close(result, Field::from(0.8), 0.01),
        "DY(dist) at (3,4) should be 0.8"
    );
}

/// Test DZ accessor extracts ∂f/∂z from Jet3.
#[test]
fn test_dz_accessor() {
    // 3D distance from origin: sqrt(x² + y² + z²)
    // ∂dist/∂z = z / sqrt(x² + y² + z²)
    let dist = (pixelflow_core::X * pixelflow_core::X
        + pixelflow_core::Y * pixelflow_core::Y
        + pixelflow_core::Z * pixelflow_core::Z)
        .sqrt();

    // Extract ∂f/∂z
    let dz_only = DZ(dist);

    // At (1, 2, 2): dist = 3, ∂dist/∂z = 2/3 ≈ 0.667
    let result = dz_only.eval(jet3_4_seeded(1.0, 2.0, 2.0));
    assert!(
        fields_close(result, Field::from(2.0 / 3.0), 0.01),
        "DZ(dist) at (1,2,2) should be 2/3"
    );
}

/// Test composed gradient magnitude using DX and DY accessors.
///
/// With the 0-fill rule and specific domain impls, composed operations like
/// `(DX(sdf) * DX(sdf) + DY(sdf) * DY(sdf)).sqrt()` now work with ManifoldExt.
/// CSE (Phase 6) will optimize away redundant evaluations.
#[test]
fn test_manual_gradient_magnitude() {
    // Distance from origin
    let dist = (pixelflow_core::X * pixelflow_core::X
        + pixelflow_core::Y * pixelflow_core::Y)
        .sqrt();

    // Manual gradient magnitude using DX/DY accessors
    let grad_mag = (DX(dist) * DX(dist) + DY(dist) * DY(dist)).sqrt();

    // At (3, 4): gradient magnitude = 1.0 for distance fields
    let result = grad_mag.eval(jet2_4_seeded(3.0, 4.0));
    assert!(
        fields_close(result, Field::from(1.0), 0.01),
        "Manual gradient magnitude at (3,4) should be 1.0"
    );
}

// NOTE: This test requires HasDerivatives bound detection for manifold params that use V/DX/DY.
// See Phase 5 plan for derivative accessor implementation details.
// The kernel! macro doesn't yet add HasDerivatives bounds when derivative accessors are used.
//
// #[test]
// fn test_manual_antialias() {
//     let circle = kernel!(|| -> Jet2 { (X * X + Y * Y).sqrt() - 1.0 });
//     let sdf = circle();
//     let antialias = kernel!(|m: kernel| V(m) / (DX(m) * DX(m) + DY(m) * DY(m)).sqrt());
//     let aa = antialias(sdf);
//     let result = aa.eval(jet2_4_seeded(2.0, 0.0));
//     assert!(fields_close(result, Field::from(1.0), 0.01));
// }

// NOTE: This test requires HasDerivatives bound detection for manifold params that use V/DX/DY.
// See Phase 5 plan for derivative accessor implementation details.
// The kernel! macro doesn't yet add HasDerivatives bounds when derivative accessors are used.
//
// #[test]
// fn test_derivative_accessors_with_composition() {
//     let circle_sdf = kernel!(|cx: f32, cy: f32, r: f32| -> Jet2 {
//         let dx = X - cx;
//         let dy = Y - cy;
//         (dx * dx + dy * dy).sqrt() - r
//     });
//     let sdf = circle_sdf(0.0, 0.0, 1.0);
//     let antialias = kernel!(|m: kernel| V(m) / (DX(m) * DX(m) + DY(m) * DY(m)).sqrt());
//     let aa = antialias(sdf);
//     let result = aa.eval(jet2_4_seeded(2.0, 0.0));
//     assert!(fields_close(result, Field::from(1.0), 0.01));
//
//     let sdf2 = circle_sdf(0.0, 0.0, 1.0);
//     let get_val = kernel!(|m: kernel| V(m));
//     let get_dx = kernel!(|m: kernel| DX(m));
//     let get_dy = kernel!(|m: kernel| DY(m));
//     let val_result = get_val(sdf2.clone()).eval(jet2_4_seeded(2.0, 0.0));
//     let gx_result = get_dx(sdf2.clone()).eval(jet2_4_seeded(2.0, 0.0));
//     let gy_result = get_dy(sdf2).eval(jet2_4_seeded(2.0, 0.0));
//     assert!(fields_close(val_result, Field::from(1.0), 0.01));
//     assert!(fields_close(gx_result, Field::from(1.0), 0.01));
//     assert!(fields_close(gy_result, Field::from(0.0), 0.01));
// }
