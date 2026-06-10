//! Regression tests for font rasterization.
//!
//! These tests ensure the font rasterization system works correctly and
//! catches bugs like:
//! - Mask AND using `*` instead of `&` (SIMD mask multiplication gives NaN)
//! - Missing Y-offset for ascent in glyph_scaled
//! - Winding number calculation errors

use pixelflow_core::{materialize_discrete, PARALLELISM};
use pixelflow_graphics::fonts::ttf::{make_line, Geometry, Line, LineKernel, Quad, QuadKernel};
use pixelflow_graphics::fonts::{text, Font};
use pixelflow_graphics::render::color::{Grayscale, Rgba8};
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::rasterize;
use std::sync::Arc;

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

// =============================================================================
// Regression: SIMD mask AND must use `&` not `*`
// =============================================================================

/// Test that SIMD mask AND works correctly (bug: using `*` gave NaN).
///
/// This test creates a simple square and verifies that points inside
/// have high coverage and points outside have low coverage.
/// Note: With analytical AA, coverage is smooth 0.0-1.0, not hard 0/1.
#[test]
fn regression_mask_and_not_multiply() {
    // Create a 400x400 square from (100,100) to (500,500)
    // Use Geometry with lines (which now produce smooth AA coverage)
    // Horizontal edges (bottom/top) contribute nothing to the winding number,
    // so `make_line` correctly rejects them (returns None). The square's inside
    // test is fully determined by the two vertical edges.
    let lines: Vec<Line<LineKernel>> = [
        make_line([[100.0, 100.0], [500.0, 100.0]]), // bottom (horizontal → None)
        make_line([[500.0, 100.0], [500.0, 500.0]]), // right
        make_line([[500.0, 500.0], [100.0, 500.0]]), // top (horizontal → None)
        make_line([[100.0, 500.0], [100.0, 100.0]]), // left
    ]
    .into_iter()
    .flatten()
    .collect();
    let geo: Geometry<Line<LineKernel>, Quad<QuadKernel>> = Geometry {
        lines: Arc::from(lines),
        quads: Arc::from(vec![]),
    };
    let lifted = Grayscale(geo);

    // Test center (should be inside, coverage > 200)
    let mut center_pixels = [0u32; PARALLELISM];
    materialize_discrete(&lifted, 300.0, 300.0, &mut center_pixels);
    let center_coverage = center_pixels[0] & 0xFF;
    assert!(
        center_coverage > 200,
        "Center of square should be inside (coverage > 200), got {}",
        center_coverage
    );

    // Test outside left (should be outside, coverage < 50)
    let mut left_pixels = [0u32; PARALLELISM];
    materialize_discrete(&lifted, 50.0, 300.0, &mut left_pixels);
    let left_coverage = left_pixels[0] & 0xFF;
    assert!(
        left_coverage < 50,
        "Left of square should be outside (coverage < 50), got {}",
        left_coverage
    );

    // Test outside right
    let mut right_pixels = [0u32; PARALLELISM];
    materialize_discrete(&lifted, 600.0, 300.0, &mut right_pixels);
    let right_coverage = right_pixels[0] & 0xFF;
    assert!(
        right_coverage < 50,
        "Right of square should be outside (coverage < 50), got {}",
        right_coverage
    );

    // Test outside above
    let mut above_pixels = [0u32; PARALLELISM];
    materialize_discrete(&lifted, 300.0, 50.0, &mut above_pixels);
    let above_coverage = above_pixels[0] & 0xFF;
    assert!(
        above_coverage < 50,
        "Above square should be outside (coverage < 50), got {}",
        above_coverage
    );

    // Test outside below
    let mut below_pixels = [0u32; PARALLELISM];
    materialize_discrete(&lifted, 300.0, 600.0, &mut below_pixels);
    let below_coverage = below_pixels[0] & 0xFF;
    assert!(
        below_coverage < 50,
        "Below square should be outside (coverage < 50), got {}",
        below_coverage
    );
}

/// Test that line segment winding calculation correctly handles the x < x_intersection test.
/// Note: With analytical AA, we get smooth coverage rather than hard 0/1.
#[test]
fn regression_line_x_intersection_test() {
    // Vertical line at x=500, going from (500,100) to (500,500)
    let line = make_line([[500.0, 100.0], [500.0, 500.0]]).unwrap();
    let geo: Geometry<Line<LineKernel>, Quad<QuadKernel>> = Geometry {
        lines: Arc::from(vec![line]),
        quads: Arc::from(vec![]),
    };
    let lifted = Grayscale(geo);

    // Winding uses left-ray casting: a point contributes when the segment's
    // crossing lies at or to its left (X >= x_int). For this vertical segment
    // at x=500, that means points to the RIGHT register the crossing and get
    // full coverage; points to the LEFT see no crossing.
    let mut right_pixels = [0u32; PARALLELISM];
    materialize_discrete(&lifted, 600.0, 300.0, &mut right_pixels);
    let right_value = right_pixels[0] & 0xFF;
    assert!(
        right_value > 200,
        "Point right of line should get high contribution, got {}",
        right_value
    );

    let mut left_pixels = [0u32; PARALLELISM];
    materialize_discrete(&lifted, 100.0, 300.0, &mut left_pixels);
    let left_value = left_pixels[0] & 0xFF;
    assert!(
        left_value < 50,
        "Point left of line should get low contribution, got {}",
        left_value
    );
}

// =============================================================================
// Regression: glyph_scaled must include Y-offset for ascent
// =============================================================================

/// Test that glyphs are rendered within the visible area.
///
/// Without the Y-offset fix, glyphs would render above y=0 (outside visible area).
#[test]
fn regression_glyph_ascent_offset() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    // Render 'A' at size 100
    let _glyph = font.glyph_scaled('A', 100.0).expect("No glyph 'A'");
    let glyph = text(&font, "A", 100.0);
    let lifted = Grayscale(glyph);

    // Create a framebuffer
    let width = 80;
    let height = 120;
    let mut frame = Frame::<Rgba8>::new(width as u32, height as u32);

    rasterize(&lifted, &mut frame, 1);

    let pixels = frame.data;

    // Count non-black pixels (with AA, we have smooth gradients)
    let white_pixels = pixels.iter().filter(|p| p.r() > 0).count();

    // There should be a significant number of non-black pixels (glyph area)
    // A typical 'A' at 100px would cover at least 1000 pixels
    assert!(
        white_pixels > 500,
        "Expected at least 500 non-black pixels, got {} (glyph may be outside visible area)",
        white_pixels
    );

    // There should also be many black pixels (background)
    let black_pixels = pixels.iter().filter(|p| p.r() == 0).count();
    assert!(
        black_pixels > 1000,
        "Expected at least 1000 black pixels for background, got {}",
        black_pixels
    );
}

// =============================================================================
// Full pipeline regression tests
// =============================================================================

/// Test that the full text rendering pipeline produces visible output.
#[test]
fn regression_text_rendering_pipeline() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    let glyph = text(&font, "HELLO", 20.0);
    let lifted = Grayscale(glyph);

    let width = 100;
    let height = 30;
    let mut frame = Frame::<Rgba8>::new(width as u32, height as u32);

    rasterize(&lifted, &mut frame, 1);
    let pixels = frame.data;

    // Count pixels by brightness
    let bright_count = pixels.iter().filter(|p| p.r() > 128).count();
    let dark_count = pixels.iter().filter(|p| p.r() < 128).count();

    // Text should take up some space but not fill the entire buffer
    // With AA, we expect smooth gradients at edges
    assert!(
        bright_count > 50,
        "Expected at least 50 bright pixels for 'HELLO', got {}",
        bright_count
    );
    assert!(
        dark_count > 500,
        "Expected at least 500 dark pixels for background, got {}",
        dark_count
    );
}

/// Test that all printable ASCII characters can be rendered.
#[test]
fn regression_all_printable_ascii_render() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    for ch in ' '..='~' {
        let glyph = font.glyph_scaled(ch, 16.0);
        assert!(
            glyph.is_some(),
            "Character '{}' (0x{:02X}) should have a scaled glyph",
            ch,
            ch as u32
        );

        // Also verify we can get advance width
        let advance = font.advance_scaled(ch, 16.0);
        assert!(
            advance.is_some(),
            "Character '{}' should have advance width",
            ch
        );
    }
}

/// Test that glyph metrics are reasonable.
#[test]
fn regression_font_metrics() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    // NotoSansMono should have these approximate values (fields are public on Font)
    assert!(font.units_per_em >= 1000, "units_per_em should be >= 1000");
    assert!(font.ascent > 0, "ascent should be positive");
    assert!(font.descent < 0, "descent should be negative");
}

/// Test that the advance width is consistent for monospace font.
#[test]
fn regression_monospace_advance() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    let advance_a = font.advance_scaled('A', 16.0).unwrap();
    let advance_m = font.advance_scaled('M', 16.0).unwrap();
    let advance_i = font.advance_scaled('i', 16.0).unwrap();

    // For a monospace font, all advances should be equal
    assert!(
        (advance_a - advance_m).abs() < 0.01,
        "Monospace font should have equal advances: A={}, M={}",
        advance_a,
        advance_m
    );
    assert!(
        (advance_a - advance_i).abs() < 0.01,
        "Monospace font should have equal advances: A={}, i={}",
        advance_a,
        advance_i
    );
}
