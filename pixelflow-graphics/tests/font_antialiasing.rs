//! Tests for gradient-normalized crossing-ramp antialiasing in the font path.
//!
//! The glyph pipeline is generic over the evaluation domain:
//! - `Field` coordinates → hard 0/1 coverage (legacy behavior).
//! - `Jet2` coordinates (via `Antialiased`) → a coverage ramp exactly
//!   ~1 screen pixel wide, at any glyph scale (autodiff chain rule).

use pixelflow_core::combinators::{At, Texture};
use pixelflow_core::{Field, Manifold};
use pixelflow_graphics::fonts::ttf::{
    make_line, make_quad, Geometry, Line, LineKernel, Quad, QuadKernel,
};
use pixelflow_graphics::fonts::Font;
use pixelflow_graphics::render::aa::Antialiased;
use std::sync::Arc;

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

type Field4 = (Field, Field, Field, Field);

/// Evaluate a scalar coverage manifold at a single point.
fn sample<M: Manifold<Field4, Output = Field>>(m: &M, x: f32, y: f32) -> f32 {
    let bound = At {
        inner: m,
        x,
        y,
        z: 0.0f32,
        w: 0.0f32,
    };
    let tex = Texture::from_manifold(&bound, 1, 1);
    tex.data()[0]
}

/// A square from (100,100) to (500,500) built from line segments.
///
/// Horizontal edges contribute nothing to the winding number, so `make_line`
/// rejects them; the inside test is determined by the two vertical edges.
fn square_geometry() -> Geometry<Line<LineKernel>, Quad<QuadKernel>> {
    let lines: Vec<Line<LineKernel>> = [
        make_line([[100.0, 100.0], [500.0, 100.0]]),
        make_line([[500.0, 100.0], [500.0, 500.0]]),
        make_line([[500.0, 500.0], [100.0, 500.0]]),
        make_line([[100.0, 500.0], [100.0, 100.0]]),
    ]
    .into_iter()
    .flatten()
    .collect();
    Geometry {
        lines: Arc::from(lines),
        quads: Arc::from(Vec::<Quad<QuadKernel>>::new()),
    }
}

/// A dome: quadratic from (0,0) up to the control point (50,100) and back to
/// (100,0), closed by the (rejected, horizontal) chord. The curve's
/// Y-extremum — the tangent point of the horizontal scanline — is at (50, 50).
fn dome_geometry() -> Geometry<Line<LineKernel>, Quad<QuadKernel>> {
    let quad = make_quad([[0.0, 0.0], [50.0, 100.0], [100.0, 0.0]]);
    Geometry {
        lines: Arc::from(Vec::<Line<LineKernel>>::new()),
        quads: Arc::from(vec![quad]),
    }
}

// =============================================================================
// Hard degeneration: Field-domain evaluation is still a binary step
// =============================================================================

#[test]
fn hard_coverage_is_binary() {
    let geo = square_geometry();

    // Interior and exterior points.
    assert_eq!(sample(&geo, 300.0, 300.0), 1.0, "interior must be 1");
    assert_eq!(sample(&geo, 50.0, 300.0), 0.0, "exterior must be 0");
    assert_eq!(sample(&geo, 600.0, 300.0), 0.0, "exterior must be 0");

    // Scan across the left edge: every sample is 0 or 1, and the step happens
    // at the edge. (The ramp denominator ε = 1e-3 makes the transition band
    // half a *milli*-pixel wide, so any sample off the exact edge is binary;
    // the 0.05 offset keeps the scan off the measure-zero d = 0 point, where
    // coverage is 0.5 by construction.)
    let mut prev = 0.0f32;
    for i in 0..80 {
        let x = 96.05 + i as f32 * 0.1;
        let c = sample(&geo, x, 300.0);
        assert!(
            !(1e-3..=1.0 - 1e-3).contains(&c),
            "hard coverage must be binary, got {c} at x={x}"
        );
        assert!(c >= prev - 1e-3, "hard step must be monotonic at x={x}");
        prev = c;
    }
    assert_eq!(sample(&geo, 99.9, 300.0), 0.0);
    assert_eq!(sample(&geo, 100.1, 300.0), 1.0);
}

#[test]
fn hard_glyph_still_works_for_existing_consumers() {
    let font = Font::parse(FONT_BYTES).unwrap();
    let glyph = font.glyph_scaled('H', 32.0).unwrap();

    // Glyph: Manifold<Field4, Output = Field> keeps working unchanged.
    let mut inside = 0;
    for j in 0..32 {
        for i in 0..32 {
            let c = sample(&glyph, i as f32 + 0.5, j as f32 + 0.5);
            assert!(c.is_finite(), "hard glyph coverage NaN/inf at ({i},{j})");
            if c > 0.5 {
                inside += 1;
            }
        }
    }
    assert!(inside > 50, "'H' at 32px should cover many pixels");
}

// =============================================================================
// AA behavior: monotonic ~1px ramp across a straight edge
// =============================================================================

#[test]
fn aa_ramp_across_straight_edge() {
    let geo = square_geometry();
    let smooth = Antialiased::new(&geo);

    // Far from the edge, AA matches hard coverage.
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
fn first_ramp_width<M: Manifold<Field4, Output = Field>>(m: &M, y: f32, x_max: f32) -> f32 {
    const STEP: f32 = 1.0 / 16.0;
    let n = (x_max / STEP) as usize;
    let cov: Vec<f32> = (0..n).map(|i| sample(m, i as f32 * STEP, y)).collect();

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
        let glyph = font.glyph_scaled('H', size).unwrap();
        let smooth = Antialiased::new(glyph);
        let w = first_ramp_width(&smooth, size * 0.5, size);
        assert!(
            (0.4..=1.6).contains(&w),
            "ramp width should be ~1 screen px at {size}px em, got {w}"
        );
        widths.push(w);
    }

    // The autodiff chain-rule payoff: 4x the glyph scale, same screen ramp.
    let diff = (widths[0] - widths[1]).abs();
    assert!(
        diff <= 0.5,
        "ramp width must not scale with em size: 16px -> {}, 64px -> {}",
        widths[0],
        widths[1]
    );
}

// =============================================================================
// AA agrees with hard coverage away from edges
// =============================================================================

#[test]
fn aa_matches_hard_far_from_edges() {
    let font = Font::parse(FONT_BYTES).unwrap();
    let size = 32.0f32;
    let glyph = font.glyph_scaled('H', size).unwrap();
    let smooth = Antialiased::new(&glyph);

    let n = size as usize;
    // hard[j][i] at pixel centers.
    let hard: Vec<Vec<f32>> = (0..n)
        .map(|j| {
            (0..n)
                .map(|i| sample(&glyph, i as f32 + 0.5, j as f32 + 0.5))
                .collect()
        })
        .collect();

    for j in 1..n - 1 {
        for i in 1..n - 1 {
            // Only compare where the whole 3x3 pixel neighborhood agrees —
            // i.e. at least one pixel away from any edge.
            let neighborhood = (-1i32..=1)
                .flat_map(|dj| (-1i32..=1).map(move |di| (di, dj)))
                .map(|(di, dj)| hard[(j as i32 + dj) as usize][(i as i32 + di) as usize]);
            let (mut all_in, mut all_out) = (true, true);
            for h in neighborhood {
                all_in &= h > 0.5;
                all_out &= h <= 0.5;
            }
            let c = sample(&smooth, i as f32 + 0.5, j as f32 + 0.5);
            if all_in {
                assert!(c > 0.9, "AA {c} disagrees with hard interior at ({i},{j})");
            }
            if all_out {
                assert!(c < 0.1, "AA {c} disagrees with hard exterior at ({i},{j})");
            }
        }
    }
}

// =============================================================================
// Quadratics: finite coverage everywhere, including the tangent point
// =============================================================================

#[test]
fn quad_tangent_point_is_finite() {
    let geo = dome_geometry();
    let smooth = Antialiased::new(&geo);

    // Dense grid over the dome, including the exact Y-extremum row (y = 50,
    // where the quadratic discriminant is 0) and the tangent point (50, 50).
    for j in 0..=120 {
        let y = j as f32 * 0.5;
        for i in 0..=100 {
            let x = i as f32;
            for (name, c) in [("hard", sample(&geo, x, y)), ("aa", sample(&smooth, x, y))] {
                assert!(
                    c.is_finite(),
                    "{name} coverage not finite at ({x},{y}): {c}"
                );
                assert!(
                    (-1e-3..=1.0 + 1e-3).contains(&c),
                    "{name} coverage out of range at ({x},{y}): {c}"
                );
            }
        }
    }

    // Sanity: the dome interior is actually covered.
    assert!(sample(&geo, 50.0, 25.0) > 0.99);
    assert!(sample(&smooth, 50.0, 25.0) > 0.99);
}

#[test]
fn quad_heavy_glyph_is_finite_everywhere() {
    let font = Font::parse(FONT_BYTES).unwrap();
    let size = 32.0f32;
    let glyph = font.glyph_scaled('O', size).unwrap();
    let smooth = Antialiased::new(&glyph);

    let mut covered = 0;
    for j in 0..=(size as usize * 2) {
        let y = j as f32 * 0.5;
        for i in 0..=(size as usize * 2) {
            let x = i as f32 * 0.5;
            let c = sample(&smooth, x, y);
            assert!(
                c.is_finite(),
                "'O' AA coverage not finite at ({x},{y}): {c}"
            );
            assert!(
                (-1e-3..=1.0 + 1e-3).contains(&c),
                "'O' AA coverage out of range at ({x},{y}): {c}"
            );
            if c > 0.5 {
                covered += 1;
            }
        }
    }
    assert!(covered > 50, "'O' at 32px should cover many samples");
}
