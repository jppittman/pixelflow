//! Kernels compose as **values**, through the builder.
//!
//! A `kernel!` body used to be able to declare a manifold-typed parameter
//! (`|sdf: kernel|`), and the builder spliced the argument's fragment into its
//! own arena through `Lower`. That was a second way to say what
//! `Kernel::at`/`sum`/`select`/arithmetic already say at the value level, and
//! it went with the tier that needed it
//! (docs/plans/2026-09-06-kernel-with-a-lattice.md, S4b-2). These are the same
//! claims, made against the builder's surface: composition nests, derivatives
//! see through the whole chain, a shared kernel used several times stays one
//! DAG, and `at` warps coordinates per site.
//!
//! The language's own composition surface is a kernel-typed parameter,
//! `k: impl Fn(f32, f32) -> f32`, applied `k(x, y)` — the builder's `at` —
//! and that is `kernel_typed_parameters.rs`'s (Phase D-a of
//! docs/plans/2026-09-25-the-language-is-kernel.md). These stay until the
//! builder's constructors leave the public surface (Phase D-d).

use pixelflow_compiler::kernel;
use pixelflow_core::{Kernel, Lattice};

/// Tabulate a kernel over a one-point lattice and read the value back.
fn eval(k: &Kernel, x: f32, y: f32) -> f32 {
    Lattice::eval_at(k, x, y)
}

fn check(name: &str, got: f32, want: f32) {
    let diff = (got - want).abs();
    let tol = 1e-3 + 1e-3 * want.abs();
    assert!(
        diff <= tol,
        "{name}: got={got} want={want} diff={diff} tol={tol}"
    );
}

/// The font architecture end to end: a gradient-normalized AA ramp composed
/// over a circle SDF. The gradient of a (unit-speed) SDF is 1, so
/// coverage = clamp(d/(1+mg) + 0.5, 0, 1).
#[test]
fn aa_ramp_over_circle_sdf() {
    let circle = kernel!(|cx: f32, cy: f32, r: f32| {
        ((X - cx) * (X - cx) + (Y - cy) * (Y - cy)).sqrt() - r
    })(0.25, -0.5, 1.0);

    let min_grad = Kernel::constant(0.001);
    let grad = circle
        .dx()
        .mul(&circle.dx())
        .add(&circle.dy().mul(&circle.dy()))
        .sqrt();
    let coverage = circle
        .div(&grad.add(&min_grad))
        .add(&Kernel::constant(0.5))
        .clamp(&Kernel::constant(0.0), &Kernel::constant(1.0));

    for (x, y) in [
        (0.25f32, 0.5f32), // on the ramp (d = 0)
        (0.35, 0.55),
        (1.6, -0.5),        // fully outside
        (0.25, -0.5 + 0.1), // deep inside
        (-1.0, 0.4),
    ] {
        let d = ((x - 0.25).powi(2) + (y + 0.5).powi(2)).sqrt() - 1.0;
        let want = (d / (1.0 + 0.001) + 0.5).clamp(0.0, 1.0);
        check("aa_over_circle", eval(&coverage, x, y), want);
    }
}

/// Composition chains: the output of one composed kernel is itself a kernel
/// and composes into the next.
#[test]
fn composition_nests() {
    let plane = kernel!(|k: f32| X * k - Y)(2.0);
    let doubled = plane.mul(&Kernel::constant(3.0));
    let shifted = doubled.add(&Kernel::constant(10.0));

    for (x, y) in [(1.0f32, 0.5f32), (-2.0, 4.0), (0.0, 0.0)] {
        let want = (x * 2.0 - y) * 3.0 + 10.0;
        check("nested", eval(&shifted, x, y), want);
    }
}

/// Derivatives see through the whole composed chain, not just one layer:
/// `dx` of a composed-and-scaled SDF is the scaled derivative.
#[test]
fn derivative_of_nested_composition() {
    let dist = kernel!(|| (X * X + Y * Y).sqrt());
    let scaled = dist.mul(&Kernel::constant(5.0));
    let ddx = scaled.dx();

    for (x, y) in [(3.0f32, 4.0f32), (1.0, 1.0), (-2.0, 5.0)] {
        let want = 5.0 * x / (x * x + y * y).sqrt();
        check("d_nested", eval(&ddx, x, y), want);
    }
}

/// One kernel used at several sites is one DAG, not several copies: the
/// arenas hash-cons, so `g*g + g` evaluates `g` once and consistently.
#[test]
fn a_kernel_used_multiple_times_stays_one_dag() {
    let f = kernel!(|| X * Y);
    let combined = f.mul(&f).add(&f);

    for (x, y) in [(2.0f32, 3.0f32), (-1.0, 4.0)] {
        let v = x * y;
        check("multi_use", eval(&combined, x, y), v * v + v);
    }
}

/// `Kernel::at` samples a kernel at warped coordinates, one warp per site.
/// The central difference of `f = x²·y` is exactly `2xy` (the quadratic's
/// second differences cancel).
#[test]
fn at_sites_warp_coordinates_per_site() {
    let f = kernel!(|| X * X * Y);
    let right = f.at(&Kernel::x().add(&Kernel::constant(1.0)), &Kernel::y());
    let left = f.at(&Kernel::x().sub(&Kernel::constant(1.0)), &Kernel::y());
    let central_dx = right.sub(&left).mul(&Kernel::constant(0.5));

    for (x, y) in [(2.0f32, 3.0f32), (-1.5, 0.5), (0.0, 4.0)] {
        check("central_dx", eval(&central_dx, x, y), 2.0 * x * y);
    }
}

/// A bare use and an `at` site of the same kernel coexist: the bare use is
/// the unwarped fragment, the site its own warp.
#[test]
fn bare_and_at_sites_mix() {
    let f = kernel!(|| X + Y * 10.0);
    let swapped = f.at(&Kernel::y(), &Kernel::x());
    let m = f.add(&swapped);

    for (x, y) in [(1.0f32, 2.0f32), (-3.0, 0.5)] {
        let want = (x + y * 10.0) + (y + x * 10.0);
        check("bare_plus_at", eval(&m, x, y), want);
    }
}
