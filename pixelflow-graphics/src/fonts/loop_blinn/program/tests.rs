//! **The font's programs draw the builder's glyphs.**
//!
//! Phase C of docs/plans/2026-09-25-the-language-is-kernel.md deletes the
//! builder ([`crate::fonts::loop_blinn::glyph`]); until then there are two
//! definitions of a glyph, the builder's and the `kernel!` block the font's
//! programs are written in, and a copy is a future divergence. So the two
//! are pinned against each other:
//!
//! - **as terms**: one piece over uniforms — [`one_piece`], an entry over a
//!   record of ten `f32`s — is one canonical key through either, with the
//!   same value in each uniform slot, so the JIT compiles one program for
//!   them and they cannot differ in a bit;
//! - **as glyphs**: every printable ASCII glyph at 7, 16 and 32 px, drawn by
//!   its piece count's program over its block ([`FontPrograms::draw`]),
//!   draws the builder's pixels, to the closed form's own error bound; the
//!   one glyph without ink (space) is pinned by name, at every size. Nothing
//!   of the builder is on that path: the pieces' rows are the font's data
//!   ([`pieces`], [`piece_row`]), the box is the outline's own
//!   ([`Outline::bounds`]), and how far past it coverage reaches is the
//!   block's `inside`, not the builder's [`Support`](super::super::Support);
//! - **as optimized terms**: every count's program optimizes, and to exact
//!   arithmetic — no reciprocal estimate, no `Dwrt` — the claim
//!   `tests/glyph_optimizes_estimate_free.rs` makes for the builder's glyph.

use super::*;
use crate::fonts::loop_blinn::{MonotoneQuad, Piece};
use crate::fonts::{Font, PIXEL_CENTER};
use pixelflow_core::Uniform;
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};
use pixelflow_ir::key::canonical;
use pixelflow_ir::{LatticeShape, OpKind};
use pixelflow_search::runtime::optimize_runtime_arena;
use std::collections::BTreeSet;

/// The crate's font.
const FONT_DATA: &[u8] = include_bytes!("../../../../assets/DejaVuSansMono-Fallback.ttf");

/// The pixel sizes the glyphs are drawn at, each its tile's side: the sizes
/// the exact-area ratchet holds every coverage to
/// (`tests/glyph_exact_area.rs`), so this gate and that one read the same
/// tiles.
const SIZES: [u32; 3] = [7, 16, 32];

/// The glyphs: printable ASCII, every one of which the font must have.
const ASCII: core::ops::RangeInclusive<char> = ' '..='~';

/// The glyphs with no ink, at every size: no pieces, so [`blank`] draws
/// them. Pinned by name and asserted per size, so a glyph losing its
/// outline at any size is a failure here, not a skipped case.
const EMPTY: [char; 1] = [' '];

/// `ch`'s outline at `size` px, in the screen frame both definitions read.
fn outline(font: &Font, ch: char, size: u32) -> Outline {
    let id = font
        .cmap_lookup(ch)
        .unwrap_or_else(|| panic!("the font has no glyph for {ch:?}"));
    font.outline_scaled_by_id(id, size as f32)
        .unwrap_or_else(|| panic!("the font has no outline for {ch:?}"))
}

/// One piece's term through either definition is one program: the same
/// canonical key, and the same value in each uniform slot. A curve and a
/// line, since only values differ between them.
#[test]
fn a_piece_is_one_term_through_either_definition() {
    let arcs = [
        MonotoneQuad::split([1.25, -3.0], [2.75, -1.0], [3.25, 2.5])[0],
        MonotoneQuad::split([4.0, 1.0], [4.5, 3.0], [5.0, 5.0])[0],
    ];
    for arc in arcs {
        let piece = Piece::oriented(arc).expect("neither arc is horizontal");
        let row = piece_row(piece);
        let columns: [Uniform; PIECE_ROW_COLS] = core::array::from_fn(|k| Uniform::new(row[k]));
        let built = super::super::piece_term(&|k| columns[k].kernel());
        let written = one_piece(record(row));
        let (built_arena, built_root) = built.parts();
        let (written_arena, written_root) = written.parts();
        let built_form = canonical(built_arena, built_root);
        let written_form = canonical(written_arena, written_root);
        assert_eq!(
            built_form.key,
            written_form.key,
            "built:   {}\nwritten: {}",
            built_arena.display(built_root),
            written_arena.display(written_root)
        );
        let values = |form: &pixelflow_ir::key::Canonical| -> Vec<u32> {
            form.uniforms.iter().map(|u| u.default.to_bits()).collect()
        };
        assert_eq!(values(&built_form), values(&written_form), "{row:?}");
    }
}

/// Every printable ASCII glyph, drawn by its piece count's program over its
/// block, draws the builder's pixels at 7, 16 and 32 px, to twice the closed
/// form's error bound, which every coverage is held to against the exact
/// area (`tests/glyph_exact_area.rs`). The equivalence the builder's
/// deletion (Phase C) stands on: the plan's B7, now through the path a frame
/// will call.
///
/// Not to the bit. A piece is one term through either definition (above),
/// but a fold over a table and a sum of instances over uniforms are two
/// programs, and the optimizer is free to extract them differently — the
/// sum's association, which products fuse, what a batch hoists. Measured,
/// on the AVX-512 and AVX2 tiers alike: they differ in the low bits of some
/// texels, by at most `6.9·10⁻⁷` at 7 px, `2.0·10⁻⁶` at 16 px and
/// `1.4·10⁻⁶` at 32 px, where the bound here is `7.6·10⁻⁶`, `1.6·10⁻⁵` and
/// `3.1·10⁻⁵` at its smallest; the run prints each size's spread.
///
/// One function, not one test per glyph: each `#[test]` is its own process
/// under nextest, and a process compiles each program once, so one sweep
/// compiles each piece count's program once per size, where a test per
/// glyph would compile every glyph again.
#[test]
fn every_ascii_glyph_drawn_by_its_count_s_program_draws_the_builders_pixels() {
    let font = Font::parse(FONT_DATA).expect("parse font");
    for size in SIZES {
        let side = size as usize;
        let mut programs = FontPrograms::new([size, size]);
        let mut spread = 0.0f64;
        let mut empties: Vec<char> = Vec::new();
        for ch in ASCII {
            let outline = outline(&font, ch, size);
            let rows = GlyphRows::of(&outline);

            let built = super::super::glyph(&outline);
            let centred = built.kernel().at(
                &Kernel::x().add(&super::super::constant(PIXEL_CENTER)),
                &Kernel::y().add(&super::super::constant(PIXEL_CENTER)),
            );
            let by_the_builder = built
                .bake(&centred, Lattice::frame(side, side))
                .into_buffer();
            let by_the_program = programs.draw(&rows).into_buffer();

            if rows.pieces() == 0 {
                assert!(
                    by_the_program.iter().all(|&v| v == 0.0),
                    "{ch:?} at {size} px has no pieces, yet its program drew ink"
                );
                assert!(
                    by_the_builder.iter().all(|&v| v == 0.0),
                    "{ch:?} at {size} px has no pieces, yet the builder drew ink"
                );
                empties.push(ch);
                continue;
            }

            assert!(
                by_the_builder.iter().any(|&v| v > 0.0),
                "{ch:?} at {size} px drew nothing to compare"
            );
            for (k, (&a, &b)) in by_the_builder.iter().zip(&by_the_program).enumerate() {
                let (i, j) = (k % side, k / side);
                let difference = (f64::from(a) - f64::from(b)).abs();
                assert!(
                    difference <= 2.0 * arithmetic_bound(side, [i, j]),
                    "{ch:?} at {size} px, texel ({i}, {j}): {a} by the builder, {b} by its program"
                );
                spread = spread.max(difference);
            }
        }
        eprintln!("at {size} px the builder and the programs differ by at most {spread:e}");
        assert_eq!(empties, EMPTY, "the glyphs with no pieces at {size} px");
    }
}

/// Every piece count printable ASCII has, at every size, has a program that
/// optimizes — `None` would mean the runtime tier declined it, and the
/// program compiled would be the one written — and to exact arithmetic: no
/// reciprocal estimate, no `Dwrt`. The block's `monotone_root` writes
/// `1.0 / (…)`, the shape that would become a 12–14-bit `Recip` estimate and
/// move a texel's coverage while leaving it in range.
#[test]
fn every_count_s_program_optimizes_estimate_free() {
    let font = Font::parse(FONT_DATA).expect("parse font");
    for size in SIZES {
        let counts: BTreeSet<usize> = ASCII
            .map(|ch| GlyphRows::of(&outline(&font, ch, size)).pieces())
            .collect();
        for &count in &counts {
            let (arena, root) = program(count).linked_parts();
            let optimized = optimize_runtime_arena(&arena, root, LatticeShape::new([size, size]))
                .unwrap_or_else(|| {
                    panic!("{count} pieces at {size} px: the runtime tier declined the program")
                });
            let (out, out_root) = &*optimized;
            assert_eq!(
                estimates_and_derivatives(out, *out_root),
                0,
                "{count} pieces at {size} px: the optimized program holds a reciprocal estimate or a Dwrt"
            );
        }
    }
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

/// The closed form's error bound at texel `(i, j)` of a `size`-px tile,
/// no arc longer than the tile: `2⁻²²·(1 + X + Y + 2·size)` at the texel's
/// centre (`tests/glyph_exact_area.rs`, `arithmetic_bound`).
fn arithmetic_bound(size: usize, [i, j]: [usize; 2]) -> f64 {
    const ARC_TOLERANCE_UNIT: f64 = 1.0 / 4_194_304.0;
    let (x, y) = (i as f64 + 0.5, j as f64 + 0.5);
    ARC_TOLERANCE_UNIT * (1.0 + x + y + 2.0 * size as f64)
}
