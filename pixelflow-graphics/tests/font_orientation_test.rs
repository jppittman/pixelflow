//! Test that glyphs render with correct vertical orientation.
//!
//! This is a regression test for the font Y-axis orientation bug where
//! glyphs were rendering upside-down.

use pixelflow_graphics::fonts::{text, Font};
use pixelflow_graphics::render::color::{Grayscale, Rgba8};
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::rasterize;

const FONT_BYTES: &[u8] = include_bytes!("../assets/NotoSansMono-Regular.ttf");

/// Measure the horizontal extent of rendered pixels at a given Y row.
/// Returns (leftmost_x, rightmost_x) of pixels above the threshold, or None if row is empty.
fn measure_row_extent(frame: &Frame<Rgba8>, y: usize, threshold: u8) -> Option<(usize, usize)> {
    let width = frame.width;
    let row_start = y * width;
    let row = &frame.data[row_start..row_start + width];

    let left = row.iter().position(|p| p.r() > threshold)?;
    let right = row.iter().rposition(|p| p.r() > threshold)?;

    Some((left, right))
}

/// Calculate the width of rendered content at a given Y row.
fn row_width(frame: &Frame<Rgba8>, y: usize, threshold: u8) -> usize {
    match measure_row_extent(frame, y, threshold) {
        Some((left, right)) => right - left + 1,
        None => 0,
    }
}

#[test]
fn letter_a_apex_is_at_top() {
    // The letter 'A' has a triangular shape:
    // - NARROW apex at the TOP
    // - WIDE legs at the BOTTOM
    // - Crossbar connecting the legs in the MIDDLE
    //
    // If the glyph is rendered upside-down, the wide part will be at the top.

    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");
    let glyph = text(&font, "A", 48.0);
    let color_manifold = Grayscale(glyph);

    let width = 60;
    let height = 70;
    let mut frame = Frame::<Rgba8>::new(width as u32, height as u32);
    rasterize(&color_manifold, &mut frame, 1);
    let pixels = &frame.data;

    // Debug: print the rendered 'A'
    println!("\nRendered 'A' at size 48 ({}x{}):", width, height);
    for y in 0..height {
        print!("{:2} | ", y);
        for x in 0..width {
            let intensity = pixels[y * width + x].r();
            let ch = if intensity > 128 {
                '#'
            } else if intensity > 32 {
                '.'
            } else {
                ' '
            };
            print!("{}", ch);
        }
        println!(" | width={}", row_width(&frame, y, 32));
    }

    // Find the vertical bounds of the rendered glyph (use threshold 32 for cleaner edges)
    let threshold = 32;
    let mut top_row = None;
    let mut bottom_row = None;
    for y in 0..height {
        if row_width(&frame, y, threshold) > 0 {
            if top_row.is_none() {
                top_row = Some(y);
            }
            bottom_row = Some(y);
        }
    }

    let top_row = top_row.expect("Glyph should have rendered content");
    let bottom_row = bottom_row.expect("Glyph should have rendered content");
    let glyph_height = bottom_row - top_row + 1;

    println!(
        "\nGlyph bounds: top={}, bottom={}, height={}",
        top_row, bottom_row, glyph_height
    );

    assert!(
        glyph_height > 10,
        "Glyph should be tall enough to measure (got {} rows)",
        glyph_height
    );

    // Measure width at top quarter and bottom quarter of the glyph
    let top_quarter_y = top_row + glyph_height / 4;
    let bottom_quarter_y = bottom_row - glyph_height / 4;

    let top_width = row_width(&frame, top_quarter_y, threshold);
    let bottom_width = row_width(&frame, bottom_quarter_y, threshold);

    println!("Top quarter (y={}): width={}", top_quarter_y, top_width);
    println!(
        "Bottom quarter (y={}): width={}",
        bottom_quarter_y, bottom_width
    );

    // The apex (top) should be NARROWER than the legs (bottom)
    assert!(
        top_width < bottom_width,
        "Letter 'A' should be narrower at top (apex) than at bottom (legs).\n\
         Top quarter (y={}) width: {}\n\
         Bottom quarter (y={}) width: {}\n\
         This suggests the glyph is rendered upside-down.",
        top_quarter_y,
        top_width,
        bottom_quarter_y,
        bottom_width
    );
}

#[test]
fn letter_a_has_crossbar() {
    // The letter 'A' has a horizontal crossbar connecting the two legs.
    // The crossbar should be filled across its width (high intensity).

    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");
    let glyph = text(&font, "A", 48.0);
    let color_manifold = Grayscale(glyph);

    let width = 60;
    let height = 70;
    let mut frame = Frame::<Rgba8>::new(width as u32, height as u32);
    rasterize(&color_manifold, &mut frame, 1);

    let threshold = 32;

    // Find vertical bounds
    let mut top_row = None;
    let mut bottom_row = None;
    for y in 0..height {
        if row_width(&frame, y, threshold) > 0 {
            if top_row.is_none() {
                top_row = Some(y);
            }
            bottom_row = Some(y);
        }
    }

    let top_row = top_row.expect("Glyph should have rendered content");
    let bottom_row = bottom_row.expect("Glyph should have rendered content");
    let glyph_height = bottom_row - top_row + 1;

    // The crossbar should be roughly in the upper half of the glyph
    // (for 'A', the crossbar is typically 30-50% down from the top)
    let search_start = top_row + glyph_height / 4;
    let search_end = top_row + 2 * glyph_height / 3;

    // Find the row with maximum width (the crossbar has solid fill)
    let mut max_width = 0;
    let mut crossbar_row = search_start;
    for y in search_start..search_end {
        let w = row_width(&frame, y, threshold);
        if w > max_width {
            max_width = w;
            crossbar_row = y;
        }
    }

    // The crossbar should be significantly wider than the apex (top)
    let apex_width = row_width(&frame, top_row + 2, threshold);

    assert!(
        max_width > apex_width,
        "Letter 'A' should have a crossbar wider than the apex.\n\
         Crossbar (y={}) width: {}\n\
         Apex width: {}",
        crossbar_row,
        max_width,
        apex_width
    );
}

#[test]
fn letter_v_point_is_at_bottom() {
    // The letter 'V' has an inverted triangular shape:
    // - WIDE at the TOP
    // - NARROW point at the BOTTOM
    //
    // This is the opposite of 'A' and helps verify we didn't just invert everything.

    let font = Font::parse(FONT_BYTES).expect("Failed to parse font");
    let glyph = text(&font, "V", 48.0);
    let color_manifold = Grayscale(glyph);

    let width = 60;
    let height = 70;
    let mut frame = Frame::<Rgba8>::new(width as u32, height as u32);
    rasterize(&color_manifold, &mut frame, 1);

    // Find the vertical bounds (use threshold 32 for cleaner edges)
    let threshold = 32;
    let mut top_row = None;
    let mut bottom_row = None;
    for y in 0..height {
        if row_width(&frame, y, threshold) > 0 {
            if top_row.is_none() {
                top_row = Some(y);
            }
            bottom_row = Some(y);
        }
    }

    let top_row = top_row.expect("Glyph should have rendered content");
    let bottom_row = bottom_row.expect("Glyph should have rendered content");
    let glyph_height = bottom_row - top_row + 1;

    assert!(glyph_height > 10, "Glyph should be tall enough to measure");

    // Measure width at top quarter and bottom quarter
    let top_quarter_y = top_row + glyph_height / 4;
    let bottom_quarter_y = bottom_row - glyph_height / 4;

    let top_width = row_width(&frame, top_quarter_y, threshold);
    let bottom_width = row_width(&frame, bottom_quarter_y, threshold);

    // The top should be WIDER than the bottom (opposite of 'A')
    assert!(
        top_width > bottom_width,
        "Letter 'V' should be wider at top than at bottom (point).\n\
         Top quarter (y={}) width: {}\n\
         Bottom quarter (y={}) width: {}\n\
         This suggests the glyph is rendered upside-down.",
        top_quarter_y,
        top_width,
        bottom_quarter_y,
        bottom_width
    );
}
