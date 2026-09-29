//! **Every glyph is written closed.** No glyph's arena holds an integral,
//! and none is left for the compiler to close.
//!
//! A glyph's coverage is the area of the pixel under ink, one fold over its
//! piece table whose body is each piece's area **in closed form**
//! (`fonts/loop_blinn.rs`, `RisingArc::pixel_area`). It was once written as
//! the integral it is, `area(χ_p)` — two interval folds per piece — and
//! closed by e-graph rules, with one-point quadrature legalizing whatever
//! they left open. That fallback point-sampled an edge while the coverage
//! stayed in range and the ink stayed where it was, so nothing downstream
//! could notice a glyph the budget had not closed. Written closed, there is
//! nothing to close and nothing to fall back to — and this is what keeps it
//! so.
//!
//! For every printable ASCII glyph at 7, 16 and 32 px, `HELLO` at 16, and
//! every printable glyph as one run at 16, each composed and shaped as the
//! atlas bakes it (texel centres, `tile_px × tile_px`):
//!
//! - the arena as written, links resolved, holds **no** interval fold;
//! - `optimize_runtime_arena` optimizes it — `None` would mean the tier
//!   declined, and the arena compiled would be the one written;
//! - and the optimized term holds no reciprocal estimate (`Recip`,
//!   `Rsqrt`: 12–14 bits, where the closed form's quotients are exact
//!   divides), no `Dwrt`, and no interval fold.
//!
//! Saturation is keyed by structure, and a glyph's structure is its bucketed
//! trip count — the pieces are data — so the hundreds of glyphs here are a
//! handful of saturations. One test per size all the same, the way
//! `loop_blinn_winding` bands its sweeps, so no single process carries the
//! whole font in a debug build.

use pixelflow_core::Kernel;
use pixelflow_graphics::fonts::{text, Font, Glyph};
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};
use pixelflow_ir::{Fold, LatticeShape, OpKind};
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

/// How many nodes reachable from `root` satisfy `is`.
fn count(arena: &ExprArena, root: ExprId, is: impl Fn(ExprNode) -> bool) -> usize {
    let mut seen = vec![false; arena.len()];
    let mut stack = vec![root];
    let mut n = 0;
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        n += usize::from(is(arena.node(id)));
        stack.extend(arena.children(id));
    }
    n
}

fn is_integral(node: ExprNode) -> bool {
    matches!(
        node,
        ExprNode::Reduce {
            fold: Fold::Interval(_),
            ..
        }
    )
}

fn is_estimate_or_derivative(node: ExprNode) -> bool {
    matches!(
        node,
        ExprNode::Unary(OpKind::Recip | OpKind::Rsqrt, _) | ExprNode::Binary(OpKind::Dwrt, _, _)
    )
}

/// Every claim of the module docs, for `glyph` at a `tile × tile` lattice.
/// `name` labels the failure. A glyph with no ink (a space) is the literal
/// 0 and still optimizes.
fn assert_closed(name: &str, glyph: &Glyph, tile: u32) {
    let kernel = at_texel_centres(glyph);
    let (arena, root) = kernel.linked_parts();
    assert_eq!(
        count(&arena, root, is_integral),
        0,
        "{name}: the glyph is written with an integral"
    );
    let shape = LatticeShape::new([tile, tile]);
    let optimized = optimize_runtime_arena(&arena, root, shape)
        .unwrap_or_else(|| panic!("{name}: the runtime tier declined the glyph"));
    let (out, out_root) = &*optimized;
    assert_eq!(
        count(out, *out_root, is_integral),
        0,
        "{name}: optimization introduced an integral"
    );
    assert_eq!(
        count(out, *out_root, is_estimate_or_derivative),
        0,
        "{name}: the optimized term holds a reciprocal estimate or a Dwrt"
    );
}

/// Every printable ASCII glyph at `size` px, on the atlas's tile.
fn every_glyph_is_closed_at(size: u32) {
    let font = Font::parse(FONT_DATA).expect("parse font");
    for ch in ' '..='~' {
        let glyph = font
            .glyph_kernel_scaled(ch, size as f32)
            .unwrap_or_else(|| panic!("the font has no glyph for {ch:?}"));
        assert_closed(&format!("{ch:?}@{size}"), &glyph, size);
    }
}

#[test]
fn every_glyph_is_closed_at_7px() {
    every_glyph_is_closed_at(7);
}

#[test]
fn every_glyph_is_closed_at_16px() {
    every_glyph_is_closed_at(16);
}

#[test]
fn every_glyph_is_closed_at_32px() {
    every_glyph_is_closed_at(32);
}

/// A run is one fold per character over one table, each over its own range
/// of rows.
#[test]
fn a_run_is_closed() {
    let font = Font::parse(FONT_DATA).expect("parse font");
    assert_closed("HELLO@16", &text(&font, "HELLO", 16.0), 16);
}

/// **How long a run is decides nothing.** A run of every printable glyph is
/// 94 folds in one graph, and it holds no integral whatever saturation's
/// class cap reaches — the case that once left a third of a 50-character
/// run's integrals to quadrature when each character's term was closed by
/// rule (docs/results/2026-09-23-glyph-is-a-formula.md, §3).
#[test]
fn a_run_of_every_glyph_is_closed() {
    let font = Font::parse(FONT_DATA).expect("parse font");
    let every: String = ('!'..='~').collect();
    assert_closed(
        &format!("{} glyphs@16", every.len()),
        &text(&font, &every, 16.0),
        16,
    );
}
