//! **Every glyph's area closes in the e-graph.** No integral a glyph is
//! written with reaches the emitter, and none is left to quadrature.
//!
//! A glyph is written as the area under each pixel
//! (`fonts/loop_blinn.rs`): one fold over its piece table whose body holds
//! `area(χ_p)` — two interval folds, the pixel's — and the e-graph closes
//! them by rule (`FactorFold`, `NarrowInterval`, `ArcMoment`;
//! docs/plans/2026-09-23-an-integral-is-a-fold.md §3, §8). Whatever the
//! rules leave open is still legal: `passes::resolve` replaces it by its
//! one-point quadrature before anything is emitted. That fallback is exactly
//! what must not happen to a glyph — the quadrature of a crossing is a
//! point sample, an aliased edge — and nothing downstream would notice: the
//! coverage stays in range and the ink stays where it was. So this counts.
//!
//! For every printable ASCII glyph at 7, 16 and 32 px, and `HELLO` at 16,
//! each composed and shaped as the atlas bakes it (texel centres,
//! `tile_px × tile_px`):
//!
//! - `optimize_runtime_arena` optimizes it — `None` would mean the tier
//!   declined, and the arena compiled would be the one written;
//! - the term extraction chose holds **no** interval fold
//!   (`runtime::unclosed_integrals`), so `resolve` has no quadrature to do;
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
use pixelflow_search::runtime::{optimize_runtime_arena, unclosed_integrals};

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
/// 0 and has nothing to close.
fn assert_closed(name: &str, glyph: &Glyph, tile: u32) {
    let kernel = at_texel_centres(glyph);
    let (arena, root) = kernel.linked_parts();
    let written = count(&arena, root, is_integral);
    if glyph.support.is_empty() {
        assert_eq!(written, 0, "{name}: an empty glyph holds an integral");
        return;
    }
    assert!(
        written > 0,
        "{name}: the glyph is not written as an area, so this closes nothing"
    );
    let shape = LatticeShape::new([tile, tile]);
    assert_eq!(
        unclosed_integrals(&arena, root, shape),
        Some(0),
        "{name}: extraction left integrals for quadrature"
    );
    let optimized = optimize_runtime_arena(&arena, root, shape)
        .unwrap_or_else(|| panic!("{name}: the runtime tier declined the glyph"));
    let (out, out_root) = &*optimized;
    assert_eq!(
        count(out, *out_root, is_integral),
        0,
        "{name}: an integral survived optimization"
    );
    assert_eq!(
        count(out, *out_root, is_estimate_or_derivative),
        0,
        "{name}: the optimized term holds a reciprocal estimate or a Dwrt"
    );
}

/// Every printable ASCII glyph at `size` px, on the atlas's tile.
fn every_glyph_closes_at(size: u32) {
    let font = Font::parse(FONT_DATA).expect("parse font");
    for ch in ' '..='~' {
        let glyph = font
            .glyph_kernel_scaled(ch, size as f32)
            .unwrap_or_else(|| panic!("the font has no glyph for {ch:?}"));
        assert_closed(&format!("{ch:?}@{size}"), &glyph, size);
    }
}

#[test]
fn every_glyph_closes_at_7px() {
    every_glyph_closes_at(7);
}

#[test]
fn every_glyph_closes_at_16px() {
    every_glyph_closes_at(16);
}

#[test]
fn every_glyph_closes_at_32px() {
    every_glyph_closes_at(32);
}

/// A run is one fold per character over one table, so its closure is five
/// integrals' closure in one graph — the case the closing phase's class cap
/// would decide first.
#[test]
fn a_run_closes() {
    let font = Font::parse(FONT_DATA).expect("parse font");
    assert_closed("HELLO@16", &text(&font, "HELLO", 16.0), 16);
}
