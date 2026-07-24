//! Test that glyphs render with correct vertical orientation.
//!
//! This is a regression test for the font Y-axis orientation bug where
//! glyphs were rendering upside-down. The text run is one fused coverage
//! `Kernel` baked over a lattice; assertions measure the coverage grid.

use pixelflow_core::Lattice;
use pixelflow_graphics::fonts::{text, Font};

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

const WIDTH: usize = 60;
const HEIGHT: usize = 70;
/// Coverage threshold matching the old 8-bit `32/255` edge threshold.
const THRESHOLD: f32 = 32.0 / 255.0;

/// Bake a one-string text kernel over the test grid at pixel centers.
fn bake_text(s: &str) -> Vec<f32> {
    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");
    let kernel = text(&font, s, 48.0);
    Lattice {
        extent: [WIDTH as u32, HEIGHT as u32, 1, 1],
        origin: [0.5, 0.5, 0.0, 0.0],
    }
    .bake(&kernel)
    .into_buffer()
}

/// Width of rendered content at a given Y row (0 if the row is empty).
fn row_width(buf: &[f32], y: usize) -> usize {
    let row = &buf[y * WIDTH..(y + 1) * WIDTH];
    let Some(left) = row.iter().position(|&v| v > THRESHOLD) else {
        return 0;
    };
    let right = row
        .iter()
        .rposition(|&v| v > THRESHOLD)
        .expect("rposition exists when position does");
    right - left + 1
}

/// Vertical bounds (top_row, bottom_row) of rendered content.
fn vertical_bounds(buf: &[f32]) -> (usize, usize) {
    let mut top_row = None;
    let mut bottom_row = None;
    for y in 0..HEIGHT {
        if row_width(buf, y) > 0 {
            if top_row.is_none() {
                top_row = Some(y);
            }
            bottom_row = Some(y);
        }
    }
    (
        top_row.expect("Glyph should have rendered content"),
        bottom_row.expect("Glyph should have rendered content"),
    )
}

#[test]
fn letter_a_apex_is_at_top() {
    // The letter 'A' has a triangular shape:
    // - NARROW apex at the TOP
    // - WIDE legs at the BOTTOM
    // If the glyph is rendered upside-down, the wide part will be at the top.
    let buf = bake_text("A");

    let (top_row, bottom_row) = vertical_bounds(&buf);
    let glyph_height = bottom_row - top_row + 1;
    assert!(
        glyph_height > 10,
        "Glyph should be tall enough to measure (got {glyph_height} rows)"
    );

    let top_quarter_y = top_row + glyph_height / 4;
    let bottom_quarter_y = bottom_row - glyph_height / 4;
    let top_width = row_width(&buf, top_quarter_y);
    let bottom_width = row_width(&buf, bottom_quarter_y);

    assert!(
        top_width < bottom_width,
        "Letter 'A' should be narrower at top (apex) than at bottom (legs).\n\
         Top quarter (y={top_quarter_y}) width: {top_width}\n\
         Bottom quarter (y={bottom_quarter_y}) width: {bottom_width}\n\
         This suggests the glyph is rendered upside-down."
    );
}

#[test]
fn letter_a_has_crossbar() {
    // The letter 'A' has a horizontal crossbar connecting the two legs,
    // significantly wider than the apex.
    let buf = bake_text("A");

    let (top_row, bottom_row) = vertical_bounds(&buf);
    let glyph_height = bottom_row - top_row + 1;

    // The crossbar is roughly 30-50% down from the top.
    let search_start = top_row + glyph_height / 4;
    let search_end = top_row + 2 * glyph_height / 3;

    let mut max_width = 0;
    let mut crossbar_row = search_start;
    for y in search_start..search_end {
        let w = row_width(&buf, y);
        if w > max_width {
            max_width = w;
            crossbar_row = y;
        }
    }

    let apex_width = row_width(&buf, top_row + 2);
    assert!(
        max_width > apex_width,
        "Letter 'A' should have a crossbar wider than the apex.\n\
         Crossbar (y={crossbar_row}) width: {max_width}\n\
         Apex width: {apex_width}"
    );
}

#[test]
fn letter_v_point_is_at_bottom() {
    // The letter 'V' is the opposite of 'A': WIDE at the TOP, NARROW point at
    // the BOTTOM — verifies we didn't just invert everything.
    let buf = bake_text("V");

    let (top_row, bottom_row) = vertical_bounds(&buf);
    let glyph_height = bottom_row - top_row + 1;
    assert!(glyph_height > 10, "Glyph should be tall enough to measure");

    let top_quarter_y = top_row + glyph_height / 4;
    let bottom_quarter_y = bottom_row - glyph_height / 4;
    let top_width = row_width(&buf, top_quarter_y);
    let bottom_width = row_width(&buf, bottom_quarter_y);

    assert!(
        top_width > bottom_width,
        "Letter 'V' should be wider at top than at bottom (point).\n\
         Top quarter (y={top_quarter_y}) width: {top_width}\n\
         Bottom quarter (y={bottom_quarter_y}) width: {bottom_width}\n\
         This suggests the glyph is rendered upside-down."
    );
}
