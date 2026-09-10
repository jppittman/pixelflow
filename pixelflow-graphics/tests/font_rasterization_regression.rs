//! Regression tests for font rasterization on the kernel path.
//!
//! Guards the behaviors that have broken before:
//! - Winding mask errors (mask AND vs multiply, crossing direction)
//! - Missing Y-offset for ascent in scaled glyphs
//! - Whole-pipeline blank output

use pixelflow_core::{Kernel, Lattice};
use pixelflow_graphics::fonts::{loop_blinn, text, Contour, Font, Glyph, Outline, Segment};

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// Evaluate a coverage kernel at a single point, binding `glyph`'s winding
/// table if it has one (`glyph`'s `Kernel::sum_over` winding sum reads a
/// bound piece table — S1a of
/// docs/plans/2026-09-09-glyph-as-a-fold-execution.md).
fn sample(glyph: &Glyph, x: f32, y: f32) -> f32 {
    glyph.bound(&glyph.kernel(), [1, 1]).eval_at(x, y)
}

/// Bake `kernel` (derived from `glyph` by a coordinate contramap, so it
/// declares the same winding table) over a `width x height` frame.
fn bake(width: u32, height: u32, kernel: &Kernel, glyph: &Glyph) -> Vec<f32> {
    let lattice = Lattice::frame(width as usize, height as usize);
    glyph.bake(kernel, lattice).into_buffer()
}

// =============================================================================
// Regression: winding masks must combine correctly
// =============================================================================

/// A 400x400 square: interior coverage saturates, all four exteriors are
/// clear, in either winding direction.
#[test]
fn regression_mask_and_not_multiply() {
    for clockwise in [false, true] {
        let mut corners = [
            [100.0, 100.0],
            [500.0, 100.0],
            [500.0, 500.0],
            [100.0, 500.0],
        ];
        if clockwise {
            corners.reverse();
        }
        let segments = (0..4)
            .map(|i| Segment::Line {
                from: corners[i],
                to: corners[(i + 1) % 4],
            })
            .collect();
        let cov = loop_blinn::glyph(&Outline {
            contours: vec![Contour::new(segments).expect("the square closes")],
        });

        assert!(
            sample(&cov, 300.0, 300.0) > 0.8,
            "center of square should be inside"
        );
        assert!(sample(&cov, 50.0, 300.0) < 0.2, "left should be outside");
        assert!(sample(&cov, 600.0, 300.0) < 0.2, "right should be outside");
        assert!(sample(&cov, 300.0, 50.0) < 0.2, "above should be outside");
        assert!(sample(&cov, 300.0, 600.0) < 0.2, "below should be outside");
        // A horizontal edge is an edge like any other: half covered on it.
        let on_top = sample(&cov, 300.0, 100.0);
        assert!(
            (on_top - 0.5).abs() < 0.05,
            "on the horizontal edge coverage should be ~0.5, got {on_top}"
        );
    }
}

// =============================================================================
// Regression: scaled glyphs must include the ascent Y-offset
// =============================================================================

/// Without the Y-offset fix, glyphs render above y=0 (outside the visible
/// area) and the visible frame is blank.
#[test]
fn regression_glyph_ascent_offset() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");
    // `text` is convention-agnostic; land on pixel centers as a contramap.
    let glyph = text(&font, "A", 100.0);
    let kernel = glyph.kernel().at(
        &Kernel::x().add(&Kernel::constant(0.5)),
        &Kernel::y().add(&Kernel::constant(0.5)),
    );

    let (width, height) = (80u32, 120u32);
    let buf = bake(width, height, &kernel, &glyph);

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
    // `text` is convention-agnostic; land on pixel centers as a contramap.
    let glyph = text(&font, "HELLO", 20.0);
    let kernel = glyph.kernel().at(
        &Kernel::x().add(&Kernel::constant(0.5)),
        &Kernel::y().add(&Kernel::constant(0.5)),
    );

    let (width, height) = (100u32, 30u32);
    let buf = bake(width, height, &kernel, &glyph);

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
