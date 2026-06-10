//! Text rendering as a category of composition.
//!
//! We map a string into a Sum of Translated, Scaled Glyphs.

use super::ttf::{Font, Glyph, Line, LineKernel, Quad, QuadKernel, Sum};
use crate::transform::Translate;
use std::sync::Arc;

/// Create a text manifold from a string.
///
/// This is a scan (prefix sum) operation over the character stream,
/// lifting each character into the Manifold category.
///
/// Returns a Sum of translated glyphs.
#[must_use]
pub fn text(
    font: &Font,
    text_str: &str,
    size: f32,
) -> Sum<Translate<Glyph<Line<LineKernel>, Quad<QuadKernel>>>> {
    // The Scan: Accumulate X position while mapping chars to glyphs
    // Optimized to perform a single CMAP lookup per character
    let mut cursor = 0.0;
    let terms: Vec<_> = text_str
        .chars()
        .map(|ch| {
            // Single CMAP lookup!
            let id = font.cmap_lookup(ch).unwrap_or(0);

            // Fetch scaled glyph and advance using the ID
            let glyph = font.glyph_scaled_by_id(id, size).unwrap_or(Glyph::Empty);
            let scaled_advance = font.advance_scaled_by_id(id, size).unwrap_or(0.0);

            let pos = cursor;
            cursor += scaled_advance;

            // The Morphism: Translate the pre-scaled glyph
            Translate {
                manifold: glyph,
                offset: [pos, 0.0],
            }
        })
        .collect();

    // The Monoid: Sum the terms
    Sum(Arc::from(terms))
}
