//! Regression tests for font rasterization on the kernel path.
//!
//! Guards the behaviors that have broken before:
//! - Winding mask errors (mask AND vs multiply, x-intersection direction)
//! - Missing Y-offset for ascent in scaled glyphs
//! - Whole-pipeline blank output

use pixelflow_core::{Kernel, Lattice};
use pixelflow_graphics::fonts::ttf_curve_analytical::AnalyticalLine;
use pixelflow_graphics::fonts::{text, Font};

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// Evaluate a coverage kernel at a single point.
fn sample(k: &Kernel, x: f32, y: f32) -> f32 {
    Lattice::point(x, y, 0.0, 0.0).bake(k).into_buffer()[0]
}

/// Winding coverage for segment kernels: `min(|Σ|, 1)`.
fn coverage(segments: &[Kernel]) -> Kernel {
    Kernel::sum(segments).abs().min(&Kernel::constant(1.0))
}

// =============================================================================
// Regression: winding masks must combine correctly
// =============================================================================

/// A 400x400 square: interior coverage saturates, all four exteriors are
/// clear. Horizontal edges contribute nothing to the winding number, so
/// `from_points` correctly rejects them; the inside test is fully determined
/// by the two vertical edges.
#[test]
fn regression_mask_and_not_multiply() {
    let segs: Vec<Kernel> = [
        AnalyticalLine::from_points([100.0, 100.0], [500.0, 100.0]), // bottom (horizontal → None)
        AnalyticalLine::from_points([500.0, 100.0], [500.0, 500.0]), // right
        AnalyticalLine::from_points([500.0, 500.0], [100.0, 500.0]), // top (horizontal → None)
        AnalyticalLine::from_points([100.0, 500.0], [100.0, 100.0]), // left
    ]
    .into_iter()
    .flatten()
    .map(|l| l.kernel())
    .collect();
    let cov = coverage(&segs);

    assert!(
        sample(&cov, 300.0, 300.0) > 0.8,
        "center of square should be inside"
    );
    assert!(sample(&cov, 50.0, 300.0) < 0.2, "left should be outside");
    assert!(sample(&cov, 600.0, 300.0) < 0.2, "right should be outside");
    assert!(sample(&cov, 300.0, 50.0) < 0.2, "above should be outside");
    assert!(sample(&cov, 300.0, 600.0) < 0.2, "below should be outside");
}

/// Winding uses left-ray casting: for a vertical segment at x=500, points to
/// the RIGHT register the crossing and get full contribution; points to the
/// LEFT see no crossing.
#[test]
fn regression_line_x_intersection_test() {
    let line = AnalyticalLine::from_points([500.0, 100.0], [500.0, 500.0]).unwrap();
    let cov = coverage(&[line.kernel()]);

    assert!(
        sample(&cov, 600.0, 300.0) > 0.8,
        "point right of line should get high contribution"
    );
    assert!(
        sample(&cov, 100.0, 300.0) < 0.2,
        "point left of line should get low contribution"
    );
}

// =============================================================================
// Regression: scaled glyphs must include the ascent Y-offset
// =============================================================================

/// Without the Y-offset fix, glyphs render above y=0 (outside the visible
/// area) and the visible frame is blank.
#[test]
fn regression_glyph_ascent_offset() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");
    let kernel = text(&font, "A", 100.0);

    let (width, height) = (80u32, 120u32);
    let baked = Lattice {
        extent: [width, height, 1, 1],
        origin: [0.5, 0.5, 0.0, 0.0],
    }
    .bake(&kernel);
    let buf = baked.buffer();

    let inked = buf.iter().filter(|&&v| v > 0.0).count();
    assert!(
        inked > 500,
        "Expected at least 500 inked texels, got {inked} (glyph may be outside visible area)"
    );
    let clear = buf.iter().filter(|&&v| v == 0.0).count();
    assert!(
        clear > 1000,
        "Expected at least 1000 clear texels for background, got {clear}"
    );
}

// =============================================================================
// Full pipeline regression tests
// =============================================================================

/// The full text pipeline produces visible, bounded output.
#[test]
fn regression_text_rendering_pipeline() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");
    let kernel = text(&font, "HELLO", 20.0);

    let (width, height) = (100u32, 30u32);
    let baked = Lattice {
        extent: [width, height, 1, 1],
        origin: [0.5, 0.5, 0.0, 0.0],
    }
    .bake(&kernel);
    let buf = baked.buffer();

    let bright = buf.iter().filter(|&&v| v > 0.5).count();
    let dark = buf.iter().filter(|&&v| v < 0.5).count();
    assert!(
        bright > 50,
        "Expected at least 50 bright texels for 'HELLO', got {bright}"
    );
    assert!(
        dark > 500,
        "Expected at least 500 dark texels for background, got {dark}"
    );
}

/// All printable ASCII characters produce scaled glyph kernels and advances.
#[test]
fn regression_all_printable_ascii_render() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    for ch in ' '..='~' {
        let glyph = font.glyph_kernel_scaled(ch, 16.0);
        assert!(
            glyph.is_some(),
            "Character '{}' (0x{:02X}) should have a scaled glyph kernel",
            ch,
            ch as u32
        );

        let advance = font.advance_scaled(ch, 16.0);
        assert!(
            advance.is_some(),
            "Character '{}' should have advance width",
            ch
        );
    }
}

/// Glyph metrics are reasonable.
#[test]
fn regression_font_metrics() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    assert!(font.units_per_em >= 1000, "units_per_em should be >= 1000");
    assert!(font.ascent > 0, "ascent should be positive");
    assert!(font.descent < 0, "descent should be negative");
}

/// Advance width is consistent for a monospace font.
#[test]
fn regression_monospace_advance() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    let advance_a = font.advance_scaled('A', 16.0).unwrap();
    let advance_m = font.advance_scaled('M', 16.0).unwrap();
    let advance_i = font.advance_scaled('i', 16.0).unwrap();

    assert!(
        (advance_a - advance_m).abs() < 0.01,
        "Monospace font should have equal advances: A={advance_a}, M={advance_m}"
    );
    assert!(
        (advance_a - advance_i).abs() < 0.01,
        "Monospace font should have equal advances: A={advance_a}, i={advance_i}"
    );
}
