//! Text layout: a run of characters as one glyph.
//!
//! A string is a scan (prefix sum) over character advances with each glyph's
//! outline placed at its pen position. **A laid-out string is one glyph**,
//! and its coverage is the winding number of all those contours together —
//! the same non-zero rule TrueType already specifies *within* a glyph, which
//! is what fills a `B` drawn as two overlapping shapes.
//!
//! That one rule is load-bearing, because **coverage is not additive**.
//! Summing per-glyph coverages reaches 2 where two glyphs' ink overlaps, and
//! 2 is not a coverage; it also makes overlapping glyphs a different
//! function from overlapping contours, which are the same situation one
//! level apart.
//!
//! An earlier version drew the wrong conclusion from that and merged every
//! character's contours into a single [`Outline`] before building any
//! kernel. The premise rules out combining *coverages*; it says nothing
//! about combining *windings*, which are additive and which vanish outside
//! a closed contour. [`loop_blinn::run`] folds each character separately and
//! combines those instead, so each character keeps its own bounding box and a
//! sample only pays for the characters whose box contains it — the binning is
//! exact, and it is what the merge threw away
//! (docs/plans/2026-09-09-a-run-is-a-glyph.md).
//!
//! The *coefficient table* is still shared, though: one table for the run,
//! with each character folding over its own row range. Per-character tables
//! would need one bound buffer slot each, and a frame binds at most
//! `MAX_BOUND_BUFFERS` without allocating — so a five-character run would
//! not compile.

use super::loop_blinn::{self, Glyph};
use super::outline::Outline;
use super::ttf::Font;

/// Lay out uncached analytical text as a single coverage [`Glyph`], in raw
/// glyph space.
///
/// Advance-based (kerning-free) layout: each glyph is scaled to `size` and
/// placed at the accumulated advance. Antialiasing comes from the kernel's
/// `Dwrt` ramps at bake.
///
/// This is the denotation — the function a laid-out string *is*. Like any
/// other glyph kernel, it carries no coordinate frame: a caller wanting
/// pixel-center sampling applies `.at(&(X + 0.5), &(Y + 0.5))` to
/// [`Glyph::kernel`] before baking, same as a raw glyph kernel would. Each
/// character's piece table travels with its own kernel, so there is nothing
/// to bind separately.
#[must_use]
pub fn text(font: &Font, text_str: &str, size: f32) -> Glyph {
    let placed: Vec<Outline> = layout(font, text_str, size).collect();
    loop_blinn::run(&placed)
}

/// The scan: every glyph's outline in the screen frame, translated to its
/// pen position, in order.
///
/// Placement stays host-side, on the outline, so each character's
/// coefficient table is already in run coordinates and no kernel-level warp
/// is needed to move it.
fn layout<'a>(font: &'a Font, text_str: &'a str, size: f32) -> impl Iterator<Item = Outline> + 'a {
    let mut cursor = 0.0f32;
    text_str.chars().map(move |ch| {
        // Single CMAP lookup per character.
        let id = font.cmap_lookup(ch).unwrap_or(0);
        let outline = font.outline_scaled_by_id(id, size);
        let advance = font.advance_scaled_by_id(id, size).unwrap_or(0.0);
        let pen = cursor;
        cursor += advance;
        outline
            .map(|o| o.translated([pen, 0.0]))
            .unwrap_or_default()
    })
}
