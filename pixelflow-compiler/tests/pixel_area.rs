//! Shader expressions denote pixel integrals, not host-side sampling loops.

use pixelflow_compiler::{kernel, kernel_raw};
use pixelflow_core::{Kernel, Lattice};

fn check(kernel: &Kernel, x: f32, y: f32, want: f32) {
    let got = Lattice::eval_at(kernel, x, y);
    assert!(
        (got - want).abs() < 1e-4,
        "at ({x}, {y}): got {got}, expected {want}"
    );
}

#[test]
fn integrate_an_indicator_to_fractional_coverage_in_both_macros() {
    let method = kernel!(|edge: f32| (X < edge).select(1.0, 0.0).area());
    let function = kernel!(|edge: f32| area((X < edge).select(1.0, 0.0)));
    let raw_method = kernel_raw!(|edge: f32| (X < edge).select(1.0, 0.0).area());
    let raw_function = kernel_raw!(|edge: f32| area((X < edge).select(1.0, 0.0)));
    for edge in [-0.25_f32, 0.0, 0.75] {
        for k in [method(edge), function(edge), raw_method(edge), raw_function(edge)] {
            for x in [-1.0_f32, -0.25, 0.0, 0.25, 0.5, 1.5] {
                // The length of [x-1/2, x+1/2] to the left of edge.
                check(&k, x, 2.0, (edge - x + 0.5).clamp(0.0, 1.0));
            }
        }
    }
}

#[test]
fn express_a_glyph_chords_coverage_as_an_integral_in_the_shader() {
    let build = kernel!(|lo: f32, hi: f32, edge: f32| {
        let band = (Y >= lo).select(1.0, 0.0) * (Y < hi).select(1.0, 0.0);
        area(band * (X < edge).select(1.0, 0.0))
    });
    let raw = kernel_raw!(|lo: f32, hi: f32, edge: f32| {
        let band = (Y >= lo).select(1.0, 0.0) * (Y < hi).select(1.0, 0.0);
        area(band * (X < edge).select(1.0, 0.0))
    });
    for k in [build(-0.25, 0.25, 0.25), raw(-0.25, 0.25, 0.25)] {
        check(&k, 0.0, 0.0, 0.375);
        check(&k, 1.0, 0.0, 0.0);
        check(&k, -1.0, 0.0, 0.5);
        check(&k, 0.0, 1.0, 0.0);
    }
}

#[test]
fn substitute_coordinates_simultaneously_without_changing_a_shared_field() {
    let k = kernel!(|| {
        let field = X + 2.0 * Y;
        field + field.at(Y, X)
    });
    let raw = kernel_raw!(|| {
        let field = X + 2.0 * Y;
        field + field.at(Y, X)
    });
    for k in [k, raw] {
        check(&k, 3.0, 5.0, 24.0);
        check(&k, -2.0, 0.5, -4.5);
    }
}

#[test]
fn distinguish_integrating_a_warped_field_from_warping_its_integral() {
    let screen = kernel!(|| (X < 1.0).select(1.0, 0.0).at(2.0 * X, Y).area());
    let carried = kernel!(|| (X < 1.0).select(1.0, 0.0).area().at(2.0 * X, Y));
    let raw_screen = kernel_raw!(|| (X < 1.0).select(1.0, 0.0).at(2.0 * X, Y).area());
    let raw_carried = kernel_raw!(|| (X < 1.0).select(1.0, 0.0).area().at(2.0 * X, Y));
    // Screen: ink ends at x=1/2, so 0.7 of the pixel about x=0.3.
    // Carried: the unit pixel about 2x=0.6 ends at 1.1, so 0.9 under ink.
    for k in [screen, raw_screen] {
        check(&k, 0.3, 0.0, 0.7);
    }
    for k in [carried, raw_carried] {
        check(&k, 0.3, 0.0, 0.9);
    }
}

#[test]
fn keep_derivatives_symbolic_through_composition_inside_the_shader() {
    let k = kernel!(|| DX(X * X).at(2.0 * X, Y));
    let raw = kernel_raw!(|| DX(X * X).at(2.0 * X, Y));
    for k in [k, raw] {
        check(&k, 3.0, 0.0, 24.0);
    }
}

#[test]
fn preserve_the_existing_midpoint_fallback_for_an_unclosed_integral() {
    let k = kernel!(|| area(X.sin()));
    let raw = kernel_raw!(|| X.sin().area());
    for k in [k, raw] {
        check(&k, 0.25, 0.0, 0.25_f32.sin());
    }
}
