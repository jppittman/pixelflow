//! Fonts on the JIT-first path: leaf outline segments become [`Kernel`] values
//! (`AnalyticalLine::kernel` / `AnalyticalQuad::kernel`), compose into a glyph's
//! coverage with `Kernel::sum` + `abs().min(1)` (the winding rule), and bake
//! **once** over a lattice — no combinator types, no `Lower`, no arena in sight.
//!
//! This is the P5 proving ground: the whole path (leaf → compose → bake) runs
//! through the language's value surface, and the antialiasing ramp's `DX`/`DY`
//! resolve to symbolic `Dwrt` at bake (no jet domain).

use pixelflow_core::{Kernel, Lattice};
use pixelflow_graphics::fonts::ttf_curve_analytical::{AnalyticalLine, AnalyticalQuad};

/// Coverage kernel for a set of outline segments: `min(|Σ contributions|, 1)`.
/// The sum is the winding number; `abs().min(1)` folds it to inside/outside.
fn coverage(segments: &[Kernel]) -> Kernel {
    Kernel::sum(segments).abs().min(&Kernel::constant(1.0))
}

/// Sample a baked coverage field at texel `(i, j)` of a unit-spaced lattice
/// whose origin is at `(0.5, 0.5)` — i.e. the sample coordinate is `(i, j)`
/// plus the origin. Extent is a plain square.
fn bake_square(cov: &Kernel, n: u32) -> Vec<f32> {
    let lattice = Lattice {
        extent: [n, n, 1, 1],
        origin: [0.5, 0.5, 0.0, 0.0],
    };
    lattice.bake(cov).buffer().to_vec()
}

#[test]
fn triangle_coverage_bakes_from_line_kernels() {
    // Triangle A=(2,2) B=(14,2) C=(8,14). The bottom edge A–B is horizontal and
    // contributes nothing to a horizontal-ray winding count (from_points drops
    // it); the two slanted edges carry the crossings.
    let edges = [
        AnalyticalLine::from_points([14.0, 2.0], [8.0, 14.0]).expect("B→C non-horizontal"),
        AnalyticalLine::from_points([8.0, 14.0], [2.0, 2.0]).expect("C→A non-horizontal"),
    ];
    let cov = coverage(&edges.iter().map(AnalyticalLine::kernel).collect::<Vec<_>>());

    let n = 16u32;
    let buf = bake_square(&cov, n);
    assert_eq!(buf.len(), (n * n) as usize);
    let at = |i: usize, j: usize| buf[j * n as usize + i];

    // Interior (7.5, 5.5): between the two edge crossings (x≈3.75 and x≈12.25),
    // several units clear of both, so the ~1px AA ramp has saturated to full.
    assert!(at(7, 5) > 0.9, "interior coverage {} should be ~1", at(7, 5));

    // Exterior, far left (0.5, 5.5): left of both crossings → no contribution.
    assert!(at(0, 5) < 0.1, "left-exterior coverage {} should be ~0", at(0, 5));

    // Exterior, below the triangle (7.5, 0.5): y < 2, outside both edges' Y
    // extent → the in_y mask zeroes every contribution.
    assert!(at(7, 0) < 0.1, "below-exterior coverage {} should be ~0", at(7, 0));

    // Every texel is a real, bounded coverage value in [0, 1] (± AA slack).
    for &v in &buf {
        assert!((-0.01..=1.01).contains(&v), "coverage out of range: {v}");
    }
}

#[test]
fn quad_leaf_bakes_as_kernel_value() {
    // A single quadratic Bezier bulging right: P0=(4,2) P1=(12,8) P2=(4,14).
    // Smoke-proof that AnalyticalQuad::kernel() lowers, composes, and bakes —
    // the curve branch (non-degenerate ay) exercises the analytical root solver
    // and its Dwrt-resolved ramp end to end.
    let quad = AnalyticalQuad::new([4.0, 2.0], [12.0, 8.0], [4.0, 14.0]);
    let cov = coverage(&[quad.kernel()]);

    let n = 16u32;
    let buf = bake_square(&cov, n);
    assert_eq!(buf.len(), (n * n) as usize);

    // The curve spans y∈[2,14]; at the mid-scanline y≈8 it reaches x≈8. A point
    // to the LEFT of the curve near mid-height sees the crossing to its right →
    // no contribution; a point to the right of x=4 but left of the bulge is
    // inside the single open crossing. We only assert the field is finite and
    // in range — the winding of one open curve is orientation-dependent, and the
    // point of this test is that the leaf bakes, not a filled-shape golden.
    for &v in &buf {
        assert!(v.is_finite(), "quad coverage produced non-finite value");
        assert!((-1.01..=1.01).contains(&v), "quad coverage out of range: {v}");
    }
    // At least one interior scanline must register a crossing (non-zero field),
    // proving the solver + ramp actually fired rather than masking everything.
    assert!(
        buf.iter().any(|&v| v.abs() > 0.5),
        "quad kernel produced an all-zero field — the root solver never fired"
    );
}
