//! Integration tests for the `kernel!` macro.
//!
//! These verify the full pipeline from macro input to numbers: parse, sema,
//! e-graph, arena, and then the one evaluation entry there is — compiled at a
//! lattice's shape and collapsed.

use pixelflow_compiler::{kernel, kernel_raw};
use pixelflow_core::{Kernel, Lattice};

// ============================================================================
// Helpers
// ============================================================================

/// Tabulate a kernel over a one-point lattice and read the value back.
///
/// This is the whole evaluation surface: a kernel plus a lattice, baked. There
/// is no per-batch entry to call instead.
fn bake1(k: &Kernel, x: f32, y: f32) -> f32 {
    Lattice::eval_at(k, x, y)
}

fn eval1(k: &Kernel, x: f32) -> f32 {
    bake1(k, x, 0.0)
}

fn eval2(k: &Kernel, x: f32, y: f32) -> f32 {
    bake1(k, x, y)
}

// ============================================================================
// Basic arithmetic
// ============================================================================

#[test]
fn macro_return_x() {
    let m = kernel!(|| X);
    assert_eq!(eval1(&m, 42.0), 42.0);
}

#[test]
fn macro_add_xy() {
    let m = kernel!(|| X + Y);
    assert_eq!(eval2(&m, 10.0, 32.0), 42.0);
}

#[test]
fn macro_complex_expr() {
    // (X + Y) * k, where the third input is a parameter rather than a third
    // axis: a lattice has two.
    let m = kernel!(|k: f32| (X + Y) * k);
    assert_eq!(eval2(&m(6.0), 2.0, 5.0), 42.0); // (2+5)*6 = 42
}

#[test]
fn macro_subtraction() {
    let m = kernel!(|| X - Y);
    assert_eq!(eval2(&m, 100.0, 58.0), 42.0);
}

#[test]
fn macro_division() {
    let m = kernel!(|| X / Y);
    assert_eq!(eval2(&m, 84.0, 2.0), 42.0);
}

#[test]
fn macro_negation() {
    let m = kernel!(|| -X);
    assert_eq!(eval1(&m, -42.0), 42.0);
}

// ============================================================================
// Transcendentals (lowered via polynomial)
// ============================================================================

#[test]
#[cfg(not(target_feature = "avx512f"))] // transcendentals: not in AVX-512 Stage-1 op set
fn macro_sin() {
    // sin(0) = 0
    let m = kernel!(|| X.sin());
    let val = eval1(&m, 0.0);
    assert!((val - 0.0).abs() < 0.001, "sin(0) = {val}, expected ~0");
}

#[test]
#[cfg(not(target_feature = "avx512f"))] // transcendentals: not in AVX-512 Stage-1 op set
fn macro_sin_pi_half() {
    // sin(π/2) ≈ 1
    let m = kernel!(|| X.sin());
    let val = eval1(&m, core::f32::consts::FRAC_PI_2);
    assert!((val - 1.0).abs() < 0.01, "sin(π/2) = {val}, expected ~1");
}

#[test]
#[cfg(not(target_feature = "avx512f"))] // transcendentals: not in AVX-512 Stage-1 op set
fn macro_cos() {
    // cos(0) = 1
    let m = kernel!(|| X.cos());
    let val = eval1(&m, 0.0);
    assert!((val - 1.0).abs() < 0.01, "cos(0) = {val}, expected ~1");
}

#[test]
fn macro_sqrt() {
    // sqrt(1764) = 42
    let m = kernel!(|| X.sqrt());
    assert_eq!(eval1(&m, 1764.0), 42.0);
}

#[test]
fn macro_abs() {
    let m = kernel!(|| X.abs());
    assert_eq!(eval1(&m, -42.0), 42.0);
}

#[test]
fn macro_min_returns_smaller_and_max_returns_larger() {
    let m_min = kernel!(|| X.min(Y));
    let m_max = kernel!(|| X.max(Y));
    assert_eq!(eval2(&m_min, 10.0, 42.0), 10.0);
    assert_eq!(eval2(&m_max, 10.0, 42.0), 42.0);
}

// ============================================================================
// Primitive methods the arena lowering used to have no arm for
// (`round`/`log10`/`pow` are real `OpKind`s, but the hand-written method-call
// match was missing all three, so a kernel body calling them failed to
// compile with "Unsupported method" even though sema and the e-graph both
// accepted them). Fixed by resolving every primitive method through
// `OpKind::from_method_call` instead of re-listing names by hand.
// ============================================================================

#[test]
fn macro_round() {
    // round(41.6) = 42, ties to even (round_ties_to_even_matching_x86s_vroundps
    // in pixelflow-ir pins the exact tie behavior; this just needs an ordinary
    // non-tie value).
    let m = kernel!(|| X.round());
    assert_eq!(eval1(&m, 41.6), 42.0);
}

#[test]
#[cfg(not(target_feature = "avx512f"))] // transcendentals: not in AVX-512 Stage-1 op set
fn macro_log10() {
    let m = kernel!(|| X.log10());
    let val = eval1(&m, 1000.0);
    let expected = 1000.0_f32.log10();
    assert!(
        (val - expected).abs() < 0.02,
        "log10(1000) = {val}, expected ~{expected}"
    );
}

#[test]
#[cfg(not(target_feature = "avx512f"))] // transcendentals: not in AVX-512 Stage-1 op set
fn macro_pow() {
    let m = kernel!(|| X.pow(Y));
    let val = eval2(&m, 2.0, 10.0);
    let expected = 2.0_f32.powf(10.0);
    assert!(
        (val - expected).abs() / expected < 0.02,
        "pow(2, 10) = {val}, expected ~{expected}"
    );
}

// ============================================================================
// Library methods: fixed compositions of primitive ops, not an `OpKind`
// themselves. Previously accepted by the AST-tier e-graph pass but
// rejected by `sema.rs` as "unknown method" before ever reaching it — the
// e-graph's `fract`/`hypot`/`clamp` decomposition was unreachable dead code.
// Fixed by validating against the same `LIBRARY_METHODS` list the two
// backends' dispatch reads.
// ============================================================================

#[test]
fn macro_fract_is_x_minus_its_floor() {
    let m = kernel!(|| X.fract());
    let val = eval1(&m, 42.75);
    assert!(
        (val - 0.75).abs() < 1e-5,
        "fract(42.75) = {val}, expected ~0.75"
    );
}

#[test]
fn macro_hypot_is_the_euclidean_norm() {
    let m = kernel!(|| X.hypot(Y));
    // 3-4-5 triangle.
    assert!((eval2(&m, 3.0, 4.0) - 5.0).abs() < 1e-5);
}

#[test]
fn macro_clamp_bounds_the_value_on_both_sides() {
    let m = kernel!(|lo: f32, hi: f32| X.clamp(lo, hi));
    let clamped = m(0.0, 1.0);
    assert_eq!(eval1(&clamped, -5.0), 0.0);
    assert_eq!(eval1(&clamped, 0.5), 0.5);
    assert_eq!(eval1(&clamped, 5.0), 1.0);
}

// `kernel_raw!` skips the e-graph entirely and goes straight from sema to
// the arena lowering — the one path that used to have no arm for
// `round`/`log10`/`pow` at all, and no decomposition for `hypot`/`fract`
// (only the AST-tier e-graph pass had those, so `kernel!` could express
// them once sema stopped rejecting the names, but `kernel_raw!` had no
// second implementation to fall back on). Proven here directly, without the
// optimizer in the way.
#[test]
fn kernel_raw_supports_the_same_primitive_and_library_methods_as_kernel() {
    assert_eq!(eval1(&kernel_raw!(|| X.round()), 41.6), 42.0);
    assert!((eval1(&kernel_raw!(|| X.fract()), 42.75) - 0.75).abs() < 1e-5);
    assert!((eval2(&kernel_raw!(|| X.hypot(Y)), 3.0, 4.0) - 5.0).abs() < 1e-5);
    assert_eq!(
        eval1(
            &kernel_raw!(|lo: f32, hi: f32| X.clamp(lo, hi))(0.0, 1.0),
            5.0
        ),
        1.0
    );
}

#[test]
#[cfg(not(target_feature = "avx512f"))] // transcendentals: not in AVX-512 Stage-1 op set
fn kernel_raw_supports_pow_and_log10() {
    let pow = eval2(&kernel_raw!(|| X.pow(Y)), 2.0, 10.0);
    assert!((pow - 2.0_f32.powf(10.0)).abs() / pow < 0.02);

    let log10 = eval1(&kernel_raw!(|| X.log10()), 1000.0);
    assert!((log10 - 1000.0_f32.log10()).abs() < 0.02);
}

// ============================================================================
// Parameter tests — builder closure API
// ============================================================================

#[test]
fn no_params_is_a_kernel() {
    // Zero-param case: the macro evaluates to a `Kernel` value directly.
    let k = kernel!(|| X + Y);
    assert!((eval2(&k, 10.0, 32.0) - 42.0).abs() < 1e-5);
}

#[test]
fn one_param_is_a_builder() {
    // A single scalar param returns a builder closure |offset: f32| -> Kernel.
    let builder = kernel!(|offset: f32| X + offset);
    let k = builder(32.0_f32);
    assert!((eval1(&k, 10.0) - 42.0).abs() < 1e-5);
}

#[test]
fn two_params_is_a_builder() {
    let builder = kernel!(|cx: f32, r: f32| (X - cx) * r);
    let k = builder(1.0_f32, 2.0_f32);
    // X=5.0: (5.0 - 1.0) * 2.0 = 8.0
    assert!((eval1(&k, 5.0) - 8.0).abs() < 1e-5);
}

// ============================================================================
// Inverse trigonometric functions
// ============================================================================

#[test]
#[cfg(not(target_feature = "avx512f"))] // transcendentals: not in AVX-512 Stage-1 op set
fn macro_atan2_matches_reference_at_boundary_and_interior_points() {
    let m = kernel!(|| Y.atan2(X));
    // atan2(1, 1) = π/4 — polynomial has ~0.06 error at t=1 boundary
    let val = eval2(&m, 1.0, 1.0);
    assert!(
        (val - std::f32::consts::FRAC_PI_4).abs() < 0.07,
        "atan2(1, 1) = {val}, expected ~{}",
        std::f32::consts::FRAC_PI_4
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
#[cfg(not(target_feature = "avx512f"))] // transcendentals: not in AVX-512 Stage-1 op set
fn macro_atan2_quadrants() {
    let m = kernel!(|| Y.atan2(X));

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
#[cfg(not(target_feature = "avx512f"))] // transcendentals: not in AVX-512 Stage-1 op set
fn macro_atan() {
    let m = kernel!(|| X.atan());
    // atan(0.5) ≈ 0.4636 — well within polynomial range
    let val = eval1(&m, 0.5);
    let expected = 0.5_f32.atan();
    assert!(
        (val - expected).abs() < 0.02,
        "atan(0.5) = {val}, expected ~{expected}"
    );
    // atan(0) = 0
    let val0 = eval1(&m, 0.0);
    assert!(val0.abs() < 0.01, "atan(0) = {val0}, expected ~0");
}

#[test]
#[cfg(not(target_feature = "avx512f"))] // transcendentals: not in AVX-512 Stage-1 op set
fn macro_asin() {
    let m = kernel!(|| X.asin());
    // asin(0) = 0
    let val0 = eval1(&m, 0.0);
    assert!(val0.abs() < 0.01, "asin(0) = {val0}, expected ~0");
    // asin(0.5) = π/6 ≈ 0.5236 — ratio < 1, polynomial is accurate
    let val_half = eval1(&m, 0.5);
    let expected = 0.5_f32.asin();
    assert!(
        (val_half - expected).abs() < 0.02,
        "asin(0.5) = {val_half}, expected ~{expected}"
    );
}

#[test]
#[cfg(not(target_feature = "avx512f"))] // transcendentals: not in AVX-512 Stage-1 op set
fn macro_acos() {
    let m = kernel!(|| X.acos());
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

// ============================================================================
// Builders choose fold or uniform by the argument's type
// ============================================================================

/// `f32` folds, exactly as it always has: the built kernel declares no
/// argument.
#[test]
fn an_f32_argument_still_folds() {
    let k = kernel!(|cx: f32, r: f32| (X - cx) * r)(1.0, 2.0);
    assert!(k.parts().0.uniforms().is_empty());
    assert!((eval1(&k, 5.0) - 8.0).abs() < 1e-5);
}

/// A `Uniform` handle makes the same parameter an argument of the compiled
/// kernel: the bake reads its default, a block moves it, and one handle
/// passed twice is one argument.
#[test]
fn a_uniform_argument_is_bound_per_call() {
    use pixelflow_core::{Manifold, Uniform};
    let cx = Uniform::new(1.0);
    let k = kernel!(|cx: f32, r: f32| (X - cx) * r)(cx, 2.0);
    assert_eq!(k.parts().0.uniforms(), &[cx.decl()]);
    assert!((eval1(&k, 5.0) - 8.0).abs() < 1e-5, "default cx = 1");

    let program = Manifold::compile(&k, [1, 1]);
    let mut block = program.block();
    block.set(cx, 3.0).expect("cx is the argument");
    let moved = program.bind(&[]).with_uniforms(&block).eval_at(5.0, 0.0);
    assert!((moved - 4.0).abs() < 1e-5, "(5 − 3)·2");

    let twice = kernel!(|a: f32, b: f32| a * b)(cx, cx);
    assert_eq!(twice.parts().0.uniforms().len(), 1);
}
