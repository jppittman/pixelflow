//! Tests for gradient-normalized crossing-ramp antialiasing in the font path.
//!
//! Antialiasing is intrinsic to the glyph coverage `Kernel`: each crossing's
//! `DX`/`DY` become symbolic `Dwrt` resolved at bake, so the coverage ramp is
//! ~1 *screen* pixel wide at any glyph scale (the chain rule runs through
//! every coordinate warp). There is no separate hard/AA mode — the old
//! Field-domain "hard step" was a degenerate mode of the retired combinator
//! pipeline.

use pixelflow_core::{Kernel, Lattice};
use pixelflow_graphics::fonts::ttf_curve_analytical::{AnalyticalLine, AnalyticalQuad};
use pixelflow_graphics::fonts::Font;

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// Evaluate a coverage kernel at a single point (the compile cache makes
/// repeated samples of the same kernel cheap).
fn sample(k: &Kernel, x: f32, y: f32) -> f32 {
    Lattice::point(x, y, 0.0, 0.0).bake(k).into_buffer()[0]
}

/// Winding coverage for segment kernels: `min(|Σ|, 1)`.
fn coverage(segments: &[Kernel]) -> Kernel {
    Kernel::sum(segments).abs().min(&Kernel::constant(1.0))
}

/// A square from (100,100) to (500,500) built from line segments.
///
/// Horizontal edges contribute nothing to the winding number, so
/// `from_points` rejects them; the inside test is determined by the two
/// vertical edges.
fn square_coverage() -> Kernel {
    let segs: Vec<Kernel> = [
        AnalyticalLine::from_points([100.0, 100.0], [500.0, 100.0]),
        AnalyticalLine::from_points([500.0, 100.0], [500.0, 500.0]),
        AnalyticalLine::from_points([500.0, 500.0], [100.0, 500.0]),
        AnalyticalLine::from_points([100.0, 500.0], [100.0, 100.0]),
    ]
    .into_iter()
    .flatten()
    .map(|l| l.kernel())
    .collect();
    coverage(&segs)
}

/// A dome: quadratic from (0,0) up to the control point (50,100) and back to
/// (100,0), closed by the (rejected, horizontal) chord. The curve's
/// Y-extremum — the tangent point of the horizontal scanline — is at (50, 50).
fn dome_coverage() -> Kernel {
    coverage(&[AnalyticalQuad::new([0.0, 0.0], [50.0, 100.0], [100.0, 0.0]).kernel()])
}

// =============================================================================
// AA behavior: monotonic ~1px ramp across a straight edge
// =============================================================================

#[test]
fn aa_ramp_across_straight_edge() {
    let smooth = square_coverage();

    // Far from the edge, coverage saturates.
    assert!(sample(&smooth, 98.0, 300.0) < 0.01, "far outside must be 0");
    assert!(sample(&smooth, 102.0, 300.0) > 0.99, "far inside must be 1");

    // On the edge, coverage is one half.
    let mid = sample(&smooth, 100.0, 300.0);
    assert!(
        (mid - 0.5).abs() < 0.05,
        "coverage on the edge should be ~0.5, got {mid}"
    );

    // Across the edge: monotonic, with genuinely intermediate values confined
    // to ~1px around the edge.
    let step = 0.125f32;
    let mut prev = -1.0f32;
    let mut intermediate = 0;
    for i in 0..33 {
        let x = 98.0 + i as f32 * step;
        let c = sample(&smooth, x, 300.0);
        assert!(c >= prev - 1e-4, "AA ramp must be monotonic at x={x}");
        prev = c;
        if c > 0.05 && c < 0.95 {
            intermediate += 1;
            assert!(
                (x - 100.0).abs() <= 0.75,
                "intermediate coverage {c} outside the 1px band at x={x}"
            );
        }
    }
    assert!(
        intermediate >= 3,
        "expected several intermediate samples across the edge, got {intermediate}"
    );
}

// =============================================================================
// Scale invariance: the ramp is ~1 *screen* pixel wide at any em size
// =============================================================================

/// Measure the width (in screen pixels) of the first left-to-right coverage
/// ramp on the given scanline, by scanning at 1/16px resolution and measuring
/// the contiguous run of intermediate values around the 0.5 crossing.
fn first_ramp_width(k: &Kernel, y: f32, x_max: f32) -> f32 {
    const STEP: f32 = 1.0 / 16.0;
    let n = (x_max / STEP) as usize;
    let cov: Vec<f32> = (0..n).map(|i| sample(k, i as f32 * STEP, y)).collect();

    // First 0.5 crossing.
    let crossing = cov
        .windows(2)
        .position(|w| w[0] < 0.5 && w[1] >= 0.5)
        .expect("no coverage edge found on scanline");

    // Contiguous run of intermediate samples containing the crossing.
    let is_mid = |c: f32| c > 0.02 && c < 0.98;
    let mut lo = crossing;
    while lo > 0 && is_mid(cov[lo]) {
        lo -= 1;
    }
    let mut hi = crossing + 1;
    while hi < n - 1 && is_mid(cov[hi]) {
        hi += 1;
    }
    (hi - lo - 1) as f32 * STEP
}

#[test]
fn ramp_width_is_one_screen_pixel_at_any_scale() {
    let font = Font::parse(FONT_BYTES).unwrap();

    let mut widths = Vec::new();
    for size in [16.0f32, 64.0f32] {
        let glyph = font.glyph_kernel_scaled('H', size).unwrap();
        let w = first_ramp_width(&glyph, size * 0.5, size);
        assert!(
            (0.4..=1.6).contains(&w),
            "ramp width should be ~1 screen px at {size}px em, got {w}"
        );
        widths.push(w);
    }

    // The symbolic chain-rule payoff: 4x the glyph scale, same screen ramp.
    let diff = (widths[0] - widths[1]).abs();
    assert!(
        diff <= 0.5,
        "ramp width must not scale with em size: 16px -> {}, 64px -> {}",
        widths[0],
        widths[1]
    );
}

// =============================================================================
// Coverage saturates away from edges
// =============================================================================

#[test]
fn coverage_saturates_far_from_edges() {
    let font = Font::parse(FONT_BYTES).unwrap();
    let size = 32.0f32;
    let glyph = font.glyph_kernel_scaled('H', size).unwrap();

    let n = size as usize;
    let baked = Lattice {
        extent: [n as u32, n as u32, 1, 1],
        origin: [0.5, 0.5, 0.0, 0.0],
    }
    .bake(&glyph);
    let buf = baked.buffer();
    let cov = |i: usize, j: usize| buf[j * n + i];

    let mut saturated_in = 0;
    for j in 1..n - 1 {
        for i in 1..n - 1 {
            // Where the whole 3x3 neighborhood agrees on inside/outside —
            // i.e. at least one pixel from any edge — coverage must saturate.
            let neighborhood = (-1i32..=1)
                .flat_map(|dj| (-1i32..=1).map(move |di| (di, dj)))
                .map(|(di, dj)| cov((i as i32 + di) as usize, (j as i32 + dj) as usize));
            let (mut all_in, mut all_out) = (true, true);
            for h in neighborhood {
                all_in &= h > 0.5;
                all_out &= h <= 0.5;
            }
            let c = cov(i, j);
            if all_in {
                assert!(c > 0.9, "interior coverage {c} not saturated at ({i},{j})");
                saturated_in += 1;
            }
            if all_out {
                assert!(c < 0.1, "exterior coverage {c} not clear at ({i},{j})");
            }
        }
    }
    assert!(saturated_in > 20, "'H' at 32px should have a solid interior");
}

// =============================================================================
// Quadratics: finite coverage everywhere, including the tangent point
// =============================================================================

#[test]
fn quad_tangent_point_is_finite() {
    let smooth = dome_coverage();

    // Dense grid over the dome, including the exact Y-extremum row (y = 50,
    // where the quadratic discriminant is 0) and the tangent point (50, 50).
    for j in 0..=120 {
        let y = j as f32 * 0.5;
        for i in 0..=100 {
            let x = i as f32;
            let c = sample(&smooth, x, y);
            assert!(c.is_finite(), "coverage not finite at ({x},{y}): {c}");
            assert!(
                (-1e-3..=1.0 + 1e-3).contains(&c),
                "coverage out of range at ({x},{y}): {c}"
            );
        }
    }

    // Sanity: the dome interior is actually covered.
    assert!(sample(&smooth, 50.0, 25.0) > 0.99);
}

#[test]
fn quad_heavy_glyph_is_finite_everywhere() {
    let font = Font::parse(FONT_BYTES).unwrap();
    let size = 32.0f32;
    let glyph = font.glyph_kernel_scaled('O', size).unwrap();

    // Sample at half-pixel steps: warp coordinates by 1/2 and bake on the
    // integer grid — texel (i, j) holds coverage at (i/2, j/2).
    let n = size as usize * 2 + 1;
    let half = glyph.at(
        &Kernel::x().mul(&Kernel::constant(0.5)),
        &Kernel::y().mul(&Kernel::constant(0.5)),
        &Kernel::z(),
        &Kernel::w(),
    );
    let baked = Lattice {
        extent: [n as u32, n as u32, 1, 1],
        origin: [0.0, 0.0, 0.0, 0.0],
    }
    .bake(&half);

    let mut covered = 0;
    for &c in baked.buffer() {
        assert!(c.is_finite(), "'O' coverage not finite: {c}");
        assert!(
            (-1e-3..=1.0 + 1e-3).contains(&c),
            "'O' coverage out of range: {c}"
        );
        if c > 0.5 {
            covered += 1;
        }
    }
    assert!(covered > 50, "'O' at 32px should cover many samples");
}
