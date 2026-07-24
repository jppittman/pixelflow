//! Text layout as Kernel composition.
//!
//! A string is a scan (prefix sum) over character advances, each glyph's
//! coverage [`Kernel`] translated to its pen position and summed. The result
//! is ONE fused coverage kernel for the whole run — composed at layout time,
//! compiled once at bake.

use super::ttf::Font;
use pixelflow_core::Kernel;

/// Lay out uncached analytical text as a single coverage [`Kernel`].
///
/// Advance-based (kerning-free) layout: each glyph is scaled to `size` and
/// translated by the accumulated advance. Antialiasing comes from the glyph
/// kernels' `Dwrt` ramps at bake.
#[must_use]
pub fn text(font: &Font, text_str: &str, size: f32) -> Kernel {
    let mut cursor = 0.0f32;
    let terms: Vec<Kernel> = text_str
        .chars()
        .map(|ch| {
            // Single CMAP lookup per character.
            let id = font.cmap_lookup(ch).unwrap_or(0);

            let glyph = font
                .glyph_kernel_scaled_by_id(id, size)
                .unwrap_or_else(|| Kernel::constant(0.0));
            let scaled_advance = font.advance_scaled_by_id(id, size).unwrap_or(0.0);

            let pos = cursor;
            cursor += scaled_advance;

            // Translate: sample the glyph at (X - pos, Y).
            glyph.at(
                &Kernel::x().sub(&Kernel::constant(pos)),
                &Kernel::y(),
                &Kernel::z(),
                &Kernel::w(),
            )
        })
        .collect();

    Kernel::sum(&terms)
}
