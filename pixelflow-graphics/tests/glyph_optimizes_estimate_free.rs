//! **Every glyph optimizes, and to exact arithmetic.** The runtime tier takes
//! every glyph and every run, and what it gives back holds no estimate.
//!
//! A glyph's coverage is one fold over its piece table whose body is each
//! piece's area in closed form (`fonts/loop_blinn.rs`,
//! `RisingArc::pixel_area`). Its quotients are exact divides; a reciprocal
//! estimate in their place (`Recip`, `Rsqrt`: 12–14 bits) would move a
//! texel's coverage while leaving it in range and the ink where it was.
//!
//! For every printable ASCII glyph at 7, 16 and 32 px, `HELLO` at 16, and
//! every printable glyph as one run at 16, each composed and shaped as the
//! atlas bakes it (texel centres, `tile_px × tile_px`):
//!
//! - `optimize_runtime_arena` optimizes it — `None` would mean the tier
//!   declined, and the arena compiled would be the one written;
//! - and the optimized term holds no reciprocal estimate and no `Dwrt`.
//!
//! `glyph_exact_area` judges a glyph's texels, and cannot see a decline:
//! a declined glyph's texels are still its area. The run of every glyph is
//! the long graph here, 94 folds in one saturation.
//!
//! Saturation is keyed by structure, and a glyph's structure is its bucketed
//! trip count — the pieces are data — so the hundreds of glyphs here are a
//! handful of saturations. One test per size all the same, the way
//! `loop_blinn_winding` bands its sweeps, so no single process carries the
//! whole font in a debug build.

use pixelflow_core::Kernel;
use pixelflow_graphics::fonts::{text, Font, Glyph};
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};
use pixelflow_ir::{LatticeShape, OpKind};
use pixelflow_search::runtime::optimize_runtime_arena;

const FONT_DATA: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// The atlas's texel centre, `fonts::PIXEL_CENTER` (crate-private).
const PIXEL_CENTER: f32 = 0.5;

/// `glyph`'s coverage as the atlas bakes it: texel `(i, j)` samples
/// `(i + ½, j + ½)`.
fn at_texel_centres(glyph: &Glyph) -> Kernel {
    glyph.kernel().at(
        &Kernel::x().add(&Kernel::constant(PIXEL_CENTER)),
        &Kernel::y().add(&Kernel::constant(PIXEL_CENTER)),
    )
}

/// How many nodes reachable from `root` are a reciprocal estimate or a
/// `Dwrt`.
fn estimates_and_derivatives(arena: &ExprArena, root: ExprId) -> usize {
    let mut seen = vec![false; arena.len()];
    let mut stack = vec![root];
    let mut n = 0;
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        n += usize::from(matches!(
            arena.node(id),
            ExprNode::Unary(OpKind::Recip | OpKind::Rsqrt, _)
                | ExprNode::Binary(OpKind::Dwrt, _, _)
        ));
        stack.extend(arena.children(id));
    }
    n
}

/// Every claim of the module docs, for `glyph` at a `tile × tile` lattice.
/// `name` labels the failure. A glyph with no ink (a space) is the literal
/// 0 and still optimizes.
fn assert_optimizes_estimate_free(name: &str, glyph: &Glyph, tile: u32) {
    let kernel = at_texel_centres(glyph);
    let (arena, root) = kernel.linked_parts();
    let shape = LatticeShape::new([tile, tile]);
    let optimized = optimize_runtime_arena(&arena, root, shape)
        .unwrap_or_else(|| panic!("{name}: the runtime tier declined the glyph"));
    let (out, out_root) = &*optimized;
    assert_eq!(
        estimates_and_derivatives(out, *out_root),
        0,
        "{name}: the optimized term holds a reciprocal estimate or a Dwrt"
    );
}

/// Every printable ASCII glyph at `size` px, on the atlas's tile.
fn every_glyph_optimizes_at(size: u32) {
    let font = Font::parse(FONT_DATA).expect("parse font");
    for ch in ' '..='~' {
        let glyph = font
            .glyph_kernel_scaled(ch, size as f32)
            .unwrap_or_else(|| panic!("the font has no glyph for {ch:?}"));
        assert_optimizes_estimate_free(&format!("{ch:?}@{size}"), &glyph, size);
    }
}

#[test]
fn every_glyph_optimizes_estimate_free_at_7px() {
    every_glyph_optimizes_at(7);
}

#[test]
fn every_glyph_optimizes_estimate_free_at_16px() {
    every_glyph_optimizes_at(16);
}

#[test]
fn every_glyph_optimizes_estimate_free_at_32px() {
    every_glyph_optimizes_at(32);
}

/// A run is one fold per character over one table, each over its own range
/// of rows.
#[test]
fn a_run_optimizes_estimate_free() {
    let font = Font::parse(FONT_DATA).expect("parse font");
    assert_optimizes_estimate_free("HELLO@16", &text(&font, "HELLO", 16.0), 16);
}

/// **How long a run is decides nothing.** A run of every printable glyph is
/// 94 folds in one graph, and the tier takes it whole and leaves no
/// estimate in it.
#[test]
fn a_run_of_every_glyph_optimizes_estimate_free() {
    let font = Font::parse(FONT_DATA).expect("parse font");
    let every: String = ('!'..='~').collect();
    assert_optimizes_estimate_free(
        &format!("{} glyphs@16", every.len()),
        &text(&font, &every, 16.0),
        16,
    );
}
