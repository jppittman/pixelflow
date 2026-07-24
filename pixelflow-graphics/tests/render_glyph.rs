//! Tests for the TTF parser and glyph kernel rendering.

use pixelflow_core::Lattice;
use pixelflow_graphics::fonts::Font;

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

#[test]
fn parse_font_and_get_glyph() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    // Test metrics (direct field access)
    assert!(font.units_per_em > 0, "Font should have units_per_em");
    assert!(font.ascent > 0, "Font should have positive ascent");

    // Test getting glyph kernels
    let glyph_a = font
        .glyph_kernel_scaled('A', 64.0)
        .expect("Glyph 'A' not found");
    let advance = font
        .advance_scaled('A', 64.0)
        .expect("Advance for 'A' not found");
    assert!(advance > 0.0, "Glyph should have positive advance");

    // Bake the glyph and confirm it renders real ink in range.
    let baked = Lattice {
        extent: [64, 64, 1, 1],
        origin: [0.5, 0.5, 0.0, 0.0],
    }
    .bake(&glyph_a);
    let buf = baked.buffer();
    assert_eq!(buf.len(), 64 * 64);
    let ink: f32 = buf.iter().sum();
    assert!(ink > 10.0, "glyph 'A' baked with almost no ink: {ink}");
    for &v in buf {
        assert!((-0.01..=1.01).contains(&v), "coverage out of range: {v}");
    }
}

#[test]
fn all_printable_ascii_glyphs_exist() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    for ch in ' '..='~' {
        let glyph = font.glyph_kernel(ch);
        assert!(
            glyph.is_some(),
            "Printable ASCII character '{}' (0x{:02X}) should exist",
            ch,
            ch as u32
        );
    }
}

#[test]
fn advance_and_kern() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    let advance_a = font.advance_scaled('A', 16.0).unwrap();
    let advance_w = font.advance_scaled('W', 16.0).unwrap();

    assert!(advance_a > 0.0, "Advance for 'A' should be positive");
    assert!(advance_w > 0.0, "Advance for 'W' should be positive");

    // In a monospace font, all advances should be equal
    assert!(
        (advance_a - advance_w).abs() < 0.01,
        "Monospace font should have equal advances"
    );

    // Kerning - monospace fonts typically have 0 kerning
    let kern = font.kern_scaled('A', 'V', 16.0);
    println!("Kerning for 'AV': {}", kern);
}
