//! # GlyphAtlas: baked glyph coverage tiles in one buffer
//!
//! The glyph-side half of the JIT cell-grid scene
//! ([`pixelflow_core::CellGridProgram`]): a uniform grid of square tiles,
//! one baked glyph per tile, in a single row-major `f32` coverage buffer a
//! cell-grid kernel gathers from. This module owns everything with a `char`
//! in it — slot assignment, bake-on-miss, the em-box convention — so the
//! core sampler only ever sees floats.
//!
//! ## Tile layout
//!
//! Every tile is `tile_px × tile_px` texels of content (the glyph's square
//! em-box bake, [`CachedGlyph`](super::cache::CachedGlyph)'s convention at
//! texel centers) inside a `(tile_px + 2)²` slot: a one-texel zero apron on
//! every side, which is exactly the reach of the cell-grid kernel's
//! bilinear taps — coverage fades to zero at tile edges instead of bleeding
//! into a neighbor.
//!
//! ## Growth
//!
//! Slots are assigned on first use ([`GlyphAtlas::uv`] bakes on miss). A
//! full atlas doubles its slot rows into a fresh buffer; the `epoch` bumps
//! so scene programs compiled against the old extents know to recompile.
//! The buffer is handed out as an `Arc` snapshot — frames in flight keep
//! whatever buffer they were built with.

use crate::fonts::ttf::Font;
use crate::fonts::PIXEL_CENTER;
use pixelflow_core::{Kernel, Lattice};
use std::collections::HashMap;
use std::sync::Arc;

/// Zero-coverage apron, in texels, on every side of a tile — the reach of a
/// bilinear tap at the tile's edge.
const PAD: usize = 1;

/// Baked glyph coverage tiles in one gatherable buffer.
pub struct GlyphAtlas {
    /// Row-major coverage texels, `width × height`. `Arc` so frames in
    /// flight pin the buffer they were built against; baking a new glyph
    /// clones on write when shared.
    buffer: Arc<Vec<f32>>,
    /// Buffer extents in texels.
    width: usize,
    height: usize,
    /// Content texels per tile edge (the glyph bake extent).
    tile_px: usize,
    /// Slot pitch in texels (`tile_px + 2·PAD`).
    slot_px: usize,
    /// Slots per atlas row.
    slots_per_row: usize,
    /// Total slot capacity.
    capacity: usize,
    /// `char` → slot. `None` records "font has no glyph" so a missing char
    /// is looked up (and logged by the caller) once, not per frame.
    ///
    /// An atlas is bound to ONE font: the map is keyed by `char` alone, so
    /// feeding `uv` different fonts would serve stale tiles. Callers that
    /// switch fonts build a new atlas (core-term does, via the same path
    /// that rebuilds on cell-size and density changes).
    slots: HashMap<char, Option<usize>>,
    /// Next unassigned slot. Slot 0 is the reserved blank.
    next_slot: usize,
    /// Point-space glyph square (the cell height, by convention).
    size_pt: f32,
    /// Texels per point the tiles are baked at.
    density: f32,
    /// Bumped whenever the buffer's EXTENTS change (growth). A scene
    /// program compiled against this atlas must recompile when it moves.
    epoch: u64,
}

impl GlyphAtlas {
    /// An atlas for glyphs baked at `size_pt × density` texels, with room
    /// for `capacity` glyphs before the first growth. Slot 0 is a reserved
    /// all-zero tile ([`GlyphAtlas::blank_uv`]).
    ///
    /// # Panics
    ///
    /// Panics on a non-positive size or density.
    #[must_use]
    pub fn new(size_pt: f32, density: f32, capacity: usize) -> Self {
        assert!(
            size_pt > 0.0 && size_pt.is_finite(),
            "invalid atlas glyph size: {size_pt}"
        );
        assert!(
            density > 0.0 && density.is_finite(),
            "invalid atlas density: {density}"
        );
        let tile_px = (size_pt * density).round().max(1.0) as usize;
        let slot_px = tile_px + 2 * PAD;
        // Squarish layout: slots_per_row ≈ √capacity, then round capacity up
        // to fill whole rows.
        let capacity = capacity.max(2);
        let slots_per_row = (capacity as f32).sqrt().ceil() as usize;
        let slot_rows = capacity.div_ceil(slots_per_row);
        let capacity = slots_per_row * slot_rows;
        let width = slots_per_row * slot_px;
        let height = slot_rows * slot_px;
        Self {
            buffer: Arc::new(vec![0.0; width * height]),
            width,
            height,
            tile_px,
            slot_px,
            slots_per_row,
            capacity,
            slots: HashMap::new(),
            next_slot: 1, // slot 0 stays blank
            size_pt,
            density,
            epoch: 0,
        }
    }

    /// Atlas width in texels.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Atlas height in texels.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Texels per point the tiles are baked at.
    #[must_use]
    pub fn density(&self) -> f32 {
        self.density
    }

    /// Point-space glyph square the tiles hold.
    #[must_use]
    pub fn size_pt(&self) -> f32 {
        self.size_pt
    }

    /// Content texels per tile edge (the glyph bake extent) — the scene
    /// metric's `tile_w`/`tile_h`.
    #[must_use]
    pub fn tile_px(&self) -> usize {
        self.tile_px
    }

    /// Bumped whenever the buffer's extents change; a scene program
    /// compiled against other extents must recompile.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// A snapshot of the coverage buffer for binding into a frame.
    #[must_use]
    pub fn buffer(&self) -> Arc<Vec<f32>> {
        self.buffer.clone()
    }

    /// Texel origin of the reserved all-zero tile.
    #[must_use]
    pub fn blank_uv(&self) -> (f32, f32) {
        self.slot_origin(0)
    }

    /// Texel origin of `ch`'s tile content, baking it on first use. Returns
    /// [`GlyphAtlas::blank_uv`] for characters the font has no glyph for.
    pub fn uv(&mut self, font: &Font, ch: char) -> (f32, f32) {
        if let Some(&slot) = self.slots.get(&ch) {
            return match slot {
                Some(s) => self.slot_origin(s),
                None => self.blank_uv(),
            };
        }
        let baked = font
            .glyph_kernel_scaled(ch, self.tile_px as f32)
            .map(|glyph| {
                let n = self.tile_px as u32;
                // This crate's shared pixel-center convention (`fonts/mod.rs`):
                // texel (i, j) holds coverage at (i + PIXEL_CENTER, j +
                // PIXEL_CENTER). Used to be the lattice's own origin; a
                // contramap on the kernel now that a lattice is a pure index.
                let centered = glyph.kernel().at(
                    &Kernel::x().add(&Kernel::constant(PIXEL_CENTER)),
                    &Kernel::y().add(&Kernel::constant(PIXEL_CENTER)),
                );
                let lattice = Lattice { extent: [n, n] };
                // `glyph.kernel()` declares the winding table `glyph` built as
                // a slot (S1a), so a bare `Lattice::bake` (which binds
                // nothing itself) would panic on it — `Glyph::bake` goes
                // through `Manifold::compile`/`bind` instead, and the table
                // travels with `centered` itself (`Kernel::with_buffer_data`),
                // so there is nothing to bind explicitly; a glyph with no
                // outline declares no buffer and bakes the same way.
                glyph.bake(&centered, lattice)
            });
        let Some(baked) = baked else {
            self.slots.insert(ch, None);
            return self.blank_uv();
        };

        if self.next_slot == self.capacity {
            self.grow();
        }
        let slot = self.next_slot;
        self.next_slot += 1;
        self.slots.insert(ch, Some(slot));

        let (u, v) = self.slot_origin(slot);
        let (u, v) = (u as usize, v as usize);
        let buf = Arc::make_mut(&mut self.buffer);
        let src = baked.buffer();
        for row in 0..self.tile_px {
            let dst_start = (v + row) * self.width + u;
            let src_start = row * self.tile_px;
            buf[dst_start..dst_start + self.tile_px]
                .copy_from_slice(&src[src_start..src_start + self.tile_px]);
        }
        (u as f32, v as f32)
    }

    /// Pre-bake a set of characters (typically ASCII at startup).
    pub fn warm(&mut self, font: &Font, chars: impl IntoIterator<Item = char>) {
        for ch in chars {
            let _ = self.uv(font, ch);
        }
    }

    /// Content origin (inside the apron) of a slot, in texels.
    fn slot_origin(&self, slot: usize) -> (f32, f32) {
        let sx = (slot % self.slots_per_row) * self.slot_px + PAD;
        let sy = (slot / self.slots_per_row) * self.slot_px + PAD;
        (sx as f32, sy as f32)
    }

    /// Double the slot rows into a fresh buffer, copying existing texels.
    ///
    /// # Panics
    ///
    /// Panics if growth would push the buffer past 2^24 texels — the
    /// cell-grid kernel computes gather indices in `f32`, which is exact
    /// only below that, so a larger atlas would silently alias texels.
    fn grow(&mut self) {
        let old_rows = self.height / self.slot_px;
        let new_rows = old_rows * 2;
        let new_height = new_rows * self.slot_px;
        assert!(
            self.width * new_height <= 1 << 24,
            "GlyphAtlas: growing past 2^24 texels ({}x{new_height}) would \
             exceed the exactly f32-indexable gather range",
            self.width
        );
        let mut grown = vec![0.0; self.width * new_height];
        grown[..self.buffer.len()].copy_from_slice(&self.buffer);
        self.buffer = Arc::new(grown);
        self.height = new_height;
        self.capacity = self.slots_per_row * new_rows;
        self.epoch += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FONT_BYTES: &[u8] = include_bytes!("../../assets/DejaVuSansMono-Fallback.ttf");

    fn font() -> Font<'static> {
        Font::parse(FONT_BYTES).expect("parse font")
    }

    #[test]
    fn slot_zero_is_blank_and_missing_chars_map_to_it() {
        let font = font();
        let mut atlas = GlyphAtlas::new(16.0, 1.0, 8);
        // A codepoint no monospace font covers.
        let uv = atlas.uv(&font, '\u{ffff}');
        assert_eq!(uv, atlas.blank_uv());
        // The blank tile really is blank (content + apron).
        let (u, v) = atlas.blank_uv();
        let (u, v) = (u as usize, v as usize);
        let buf = atlas.buffer();
        for row in 0..18 {
            for col in 0..18 {
                let texel = buf[(v - PAD + row) * atlas.width() + (u - PAD + col)];
                assert_eq!(texel, 0.0, "blank tile has ink at ({col},{row})");
            }
        }
    }

    #[test]
    fn baked_glyph_has_ink_and_a_zero_apron() {
        let font = font();
        let mut atlas = GlyphAtlas::new(16.0, 1.0, 8);
        let (u, v) = atlas.uv(&font, 'A');
        let (u, v) = (u as usize, v as usize);
        let buf = atlas.buffer();
        let tile = 16;
        let ink: f32 = (0..tile)
            .flat_map(|r| (0..tile).map(move |c| (r, c)))
            .map(|(r, c)| buf[(v + r) * atlas.width() + (u + c)])
            .sum();
        assert!(ink > 1.0, "glyph 'A' baked with almost no ink: {ink}");
        // The apron stays zero on all four sides.
        for k in 0..tile + 2 {
            let w = atlas.width();
            assert_eq!(buf[(v - 1) * w + (u - 1 + k)], 0.0, "top apron");
            assert_eq!(buf[(v + tile) * w + (u - 1 + k)], 0.0, "bottom apron");
            assert_eq!(buf[(v - 1 + k) * w + (u - 1)], 0.0, "left apron");
            assert_eq!(buf[(v - 1 + k) * w + (u + tile)], 0.0, "right apron");
        }
    }

    #[test]
    fn uv_is_stable_and_memoized() {
        let font = font();
        let mut atlas = GlyphAtlas::new(16.0, 1.0, 8);
        let first = atlas.uv(&font, 'Q');
        let second = atlas.uv(&font, 'Q');
        assert_eq!(first, second);
    }

    #[test]
    fn growth_preserves_tiles_and_bumps_epoch() {
        let font = font();
        let mut atlas = GlyphAtlas::new(8.0, 1.0, 2);
        let before_epoch = atlas.epoch();
        let a_uv = atlas.uv(&font, 'a');
        // Force growth past the tiny capacity.
        for ch in 'b'..='m' {
            let _ = atlas.uv(&font, ch);
        }
        assert!(atlas.epoch() > before_epoch, "growth must bump the epoch");
        // 'a' keeps its slot and its texels.
        assert_eq!(atlas.uv(&font, 'a'), a_uv);
        let (u, v) = (a_uv.0 as usize, a_uv.1 as usize);
        let buf = atlas.buffer();
        let ink: f32 = (0..8)
            .flat_map(|r| (0..8).map(move |c| (r, c)))
            .map(|(r, c)| buf[(v + r) * atlas.width() + (u + c)])
            .sum();
        assert!(ink > 0.5, "'a' lost its ink across growth: {ink}");
    }

    #[test]
    fn frames_in_flight_keep_their_buffer() {
        let font = font();
        let mut atlas = GlyphAtlas::new(8.0, 1.0, 4);
        let _ = atlas.uv(&font, 'x');
        let snapshot = atlas.buffer();
        let before: Vec<f32> = snapshot.as_ref().clone();
        // New bakes clone-on-write; the snapshot must not change.
        let _ = atlas.uv(&font, 'y');
        assert_eq!(*snapshot, before, "in-flight frame's buffer mutated");
    }
}
