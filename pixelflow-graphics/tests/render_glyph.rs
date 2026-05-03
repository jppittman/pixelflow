//! Tests for the TTF parser and glyph rendering.

use pixelflow_core::{Field, ManifoldCompat};
use pixelflow_graphics::fonts::Font;

const FONT_BYTES: &[u8] = include_bytes!("../assets/NotoSansMono-Regular.ttf");

#[test]
fn parse_font_and_get_glyph() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    // Test metrics (direct field access)
    assert!(font.units_per_em > 0, "Font should have units_per_em");
    assert!(font.ascent > 0, "Font should have positive ascent");

    // Test getting glyphs
    let glyph_a = font.glyph_scaled('A', 64.0).expect("Glyph 'A' not found");
    let advance = font
        .advance_scaled('A', 64.0)
        .expect("Advance for 'A' not found");
    assert!(advance > 0.0, "Glyph should have positive advance");

    // Test that we can evaluate the glyph
    let val = glyph_a.eval_raw(
        Field::from(32.0),
        Field::from(32.0),
        Field::from(0.0),
        Field::from(0.0),
    );
    println!("Glyph 'A' evaluated at (32,32): {:?}", val);
    println!("Glyph advance: {}", advance);
}



#[test]
fn all_printable_ascii_glyphs_exist() {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");

    for ch in ' '..='~' {
        let glyph = font.glyph(ch);
        assert!(
            glyph.is_some(),
            "Printable ASCII character '{}' (0x{:02X}) should exist",
            ch,
            ch as u32
        );
    }

    println!("All printable ASCII characters found in font");
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
