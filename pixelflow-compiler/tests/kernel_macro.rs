//! Integration tests for the `kernel!` macro.
//!
//! These verify the full pipeline from macro input to numbers: parse, sema,
//! e-graph, arena, and then the one evaluation entry there is — compiled at a
//! lattice's shape and collapsed.

use pixelflow_compiler::{kernel, kernel_raw};
use pixelflow_core::{Kernel, Lattice};

// ============================================================================
// Primitive methods: each resolves through `OpKind::from_method_call`, so a
// name is listed in no second table.
// ============================================================================

#[test]
fn round_takes_a_non_tie_to_the_nearest_integer() {
    // round(41.6) = 42, ties to even (round_ties_to_even_matching_x86s_vroundps
    // in pixelflow-ir pins the exact tie behavior; this just needs an ordinary
    // non-tie value).
    let m = kernel!(|| X.round());
    assert_eq!(Lattice::eval_at(&m, 41.6, 0.0), 42.0);
}

#[test]
fn log10_of_a_thousand_is_three() {
    let m = kernel!(|| X.log10());
    let val = Lattice::eval_at(&m, 1000.0, 0.0);
    let expected = 1000.0_f32.log10();
    assert!(
        (val - expected).abs() < 0.02,
        "log10(1000) = {val}, expected ~{expected}"
    );
}

/// `2^10` and `10^2` differ, so a swapped receiver and argument is seen.
#[test]
fn pow_takes_its_receiver_as_the_base_and_its_argument_as_the_exponent() {
    let m = kernel!(|| X.pow(Y));
    let val = Lattice::eval_at(&m, 2.0, 10.0);
    let expected = 2.0_f32.powf(10.0);
    assert!(
        (val - expected).abs() / expected < 0.02,
        "pow(2, 10) = {val}, expected ~{expected}"
    );
}

// ============================================================================
// Library methods: fixed compositions of primitive ops, not an `OpKind`
// themselves. `sema` validates a name against the `LIBRARY_METHODS` list that
// lowering reads.
// ============================================================================

#[test]
fn macro_fract_is_x_minus_its_floor() {
    let m = kernel!(|| X.fract());
    let val = Lattice::eval_at(&m, 42.75, 0.0);
    assert!(
        (val - 0.75).abs() < 1e-5,
        "fract(42.75) = {val}, expected ~0.75"
    );
}

#[test]
fn macro_hypot_is_the_euclidean_norm() {
    let m = kernel!(|| X.hypot(Y));
    // 3-4-5 triangle.
    assert!((Lattice::eval_at(&m, 3.0, 4.0) - 5.0).abs() < 1e-5);
}

#[test]
fn macro_clamp_bounds_the_value_on_both_sides() {
    let m = kernel!(|lo: f32, hi: f32| X.clamp(lo, hi));
    let clamped = m(0.0, 1.0);
    assert_eq!(Lattice::eval_at(&clamped, -5.0, 0.0), 0.0);
    assert_eq!(Lattice::eval_at(&clamped, 0.5, 0.0), 0.5);
    assert_eq!(Lattice::eval_at(&clamped, 5.0, 0.0), 1.0);
}

// `kernel_raw!` skips the e-graph and goes straight from sema to the arena
// lowering, so this proves the front end's methods without the optimizer in
// the way.
#[test]
fn kernel_raw_supports_the_same_primitive_and_library_methods_as_kernel() {
    assert_eq!(
        Lattice::eval_at(&kernel_raw!(|| X.round()), 41.6, 0.0),
        42.0
    );
    assert!((Lattice::eval_at(&kernel_raw!(|| X.fract()), 42.75, 0.0) - 0.75).abs() < 1e-5);
    assert!((Lattice::eval_at(&kernel_raw!(|| X.hypot(Y)), 3.0, 4.0) - 5.0).abs() < 1e-5);
    assert_eq!(
        Lattice::eval_at(
            &kernel_raw!(|lo: f32, hi: f32| X.clamp(lo, hi))(0.0, 1.0),
            5.0,
            0.0
        ),
        1.0
    );
}

#[test]
fn kernel_raw_supports_pow_and_log10() {
    let pow = Lattice::eval_at(&kernel_raw!(|| X.pow(Y)), 2.0, 10.0);
    assert!((pow - 2.0_f32.powf(10.0)).abs() / pow < 0.02);

    let log10 = Lattice::eval_at(&kernel_raw!(|| X.log10()), 1000.0, 0.0);
    assert!((log10 - 1000.0_f32.log10()).abs() < 0.02);
}

// ============================================================================
// Inverse trigonometric functions
// ============================================================================

fn assert_within(name: &str, got: f32, want: f32, tolerance: f32) {
    assert!(
        (got - want).abs() < tolerance,
        "{name} = {got}, expected ~{want}"
    );
}

/// `Y.atan2(X)` takes `y` as the receiver and `x` as the argument, in every
/// quadrant. The ratio `½` is well inside the polynomial's range, and the
/// ratio `1` at its boundary, where the polynomial is off by about 0.06.
#[test]
fn atan2_takes_y_as_the_receiver_and_x_as_the_argument_in_every_quadrant() {
    let m = kernel!(|| Y.atan2(X));
    for (x, y, tolerance) in [
        (1.0_f32, 1.0_f32, 0.07), // the boundary: π/4
        (2.0, 1.0, 0.02),         // Q1: atan(0.5)
        (-2.0, 1.0, 0.07),        // Q2: π − atan(0.5)
        (-2.0, -1.0, 0.07),       // Q3: −(π − atan(0.5))
        (2.0, -1.0, 0.02),        // Q4: −atan(0.5)
    ] {
        assert_within(
            &format!("atan2({y}, {x})"),
            Lattice::eval_at(&m, x, y),
            y.atan2(x),
            tolerance,
        );
    }
}

#[test]
fn atan_asin_and_acos_match_std_inside_their_polynomial_range() {
    let atan = kernel!(|| X.atan());
    let asin = kernel!(|| X.asin());
    let acos = kernel!(|| X.acos());
    let at = |k: &Kernel, x: f32| Lattice::eval_at(k, x, 0.0);

    assert_within("atan(0)", at(&atan, 0.0), 0.0, 0.01);
    assert_within("atan(0.5)", at(&atan, 0.5), 0.5_f32.atan(), 0.02);

    assert_within("asin(0)", at(&asin, 0.0), 0.0, 0.01);
    assert_within("asin(0.5)", at(&asin, 0.5), 0.5_f32.asin(), 0.02);

    // acos(0.5) exercises the large-ratio path (ratio ≈ 1.73).
    assert_within("acos(0.5)", at(&acos, 0.5), 0.5_f32.acos(), 0.05);
    assert_within("acos(0)", at(&acos, 0.0), std::f32::consts::FRAC_PI_2, 0.07);
}

// ============================================================================
// Every parameter is a uniform; a constant is a `const` item
// (docs/language/2026-09-25-the-language-is-kernel.md §1.4)
// ============================================================================

kernel! {
    const CX: f32 = 1.0;
    const R: f32 = 2.0;
    /// The closure below with its values spelled as constants of the program.
    pub fn folded() -> f32 { (X - CX) * R }
}

/// A constant is spelled as a `const` item, and it folds: the kernel declares
/// no argument. An `f32` argument is a uniform with the call's value as its
/// default, and a bake reads that value.
#[test]
fn a_const_item_folds_and_an_f32_argument_is_a_uniform() {
    let folded = folded();
    assert!(folded.uniforms().is_empty());
    assert!((Lattice::eval_at(&folded, 5.0, 0.0) - 8.0).abs() < 1e-5);

    let k = kernel!(|cx: f32, r: f32| (X - cx) * r)(1.0, 2.0);
    let defaults: Vec<f32> = k.uniforms().iter().map(|u| u.default).collect();
    assert_eq!(defaults, [1.0, 2.0], "each parameter is an argument");
    assert!((Lattice::eval_at(&k, 5.0, 0.0) - 8.0).abs() < 1e-5);
}

/// A parameter is an argument of the compiled kernel: the bake reads the
/// call's value, a block moves it, and a parameter read twice is one
/// argument. The closure form has no `Args` record, so its program is
/// rebound by position, through the one core method.
#[test]
fn a_uniform_argument_is_bound_per_call() {
    use pixelflow_core::Manifold;
    let k = kernel!(|cx: f32, r: f32| (X - cx) * r)(1.0, 2.0);
    assert!(
        (Lattice::eval_at(&k, 5.0, 0.0) - 8.0).abs() < 1e-5,
        "default cx = 1"
    );

    let program = Manifold::compile(&k, [1, 1]);
    let mut block = program.block();
    block.set_declared([3.0, 2.0]).expect("cx and r");
    let moved = program.bind(&[]).with_uniforms(&block).eval_at(5.0, 0.0);
    assert!((moved - 4.0).abs() < 1e-5, "(5 − 3)·2");

    let squared = kernel!(|a: f32| a * a)(3.0);
    assert_eq!(
        squared.uniforms().len(),
        1,
        "one parameter read twice is one argument"
    );
    assert_eq!(Lattice::eval_at(&squared, 0.0, 0.0), 9.0);
}
