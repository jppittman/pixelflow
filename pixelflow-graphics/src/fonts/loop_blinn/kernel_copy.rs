//! **The `kernel!` copy of a glyph is the builder's.**
//!
//! Phase C of docs/plans/2026-09-25-the-language-is-kernel.md writes the
//! font in `kernel!` and deletes this builder; until then there are two
//! definitions of a glyph, [`glyph`] and §1.7's block
//! (`pixelflow-compiler/tests/common/section_1_7.rs`). A copy is a future
//! divergence, so the two are pinned against each other twice:
//!
//! - **as terms**: one piece over uniforms — the block's `one_piece`, an
//!   entry over a record of ten `f32`s — is one canonical key through
//!   either, with the same value in each uniform slot, so the JIT compiles
//!   one program for them and they cannot differ in a bit; and
//! - **as glyphs**: every printable ASCII glyph with ink at 7, 16 and 32 px
//!   (the plan's B7), composed the way §1.7 says the host composes one —
//!   each piece one `one_piece` instance, the instances summed into one ink
//!   by the block's `sum2` as a balanced tree, and the block's
//!   `glyph(ink, bounds)` over it — draws [`glyph`]'s pixels, to the closed
//!   form's own error bound; the one glyph without ink (space) is pinned by
//!   name, at every size. Nothing of the builder is on that path: the
//!   pieces' rows are the font's data ([`pieces`], [`piece_row`]), the box
//!   is the outline's own ([`Outline::bounds`]), and how far past it
//!   coverage reaches is the block's `inside`, not the builder's
//!   [`Support`].

use super::*;
use crate::fonts::{Font, PIXEL_CENTER};
use pixelflow_ir::key::canonical;

/// §1.7's block, expanded. This module calls `one_piece`, `sum2` and
/// `glyph`; the `Args` record of `one_piece` goes unused here.
#[allow(dead_code)]
mod block {
    use pixelflow_compiler::kernel;

    include!("../../../../pixelflow-compiler/tests/common/section_1_7.rs");

    section_1_7!(kernel);
}

use block::{one_piece, Bounds, Row};

/// The crate's font.
const FONT_DATA: &[u8] = include_bytes!("../../../assets/DejaVuSansMono-Fallback.ttf");

/// The pixel sizes the glyphs are drawn at, each its tile's side: the
/// sizes the exact-area ratchet holds every coverage to
/// (`tests/glyph_exact_area.rs`), so this gate and that one read the same
/// tiles.
const SIZES: [usize; 3] = [7, 16, 32];

/// The glyphs: printable ASCII, every one of which the font must have.
const ASCII: core::ops::RangeInclusive<char> = ' '..='~';

/// The glyphs with no ink, at every size: no pieces to compose, so the
/// language side has nothing to say. Pinned by name and asserted per size,
/// so a glyph losing its outline at any size is a failure here, not a
/// skipped case. (The builder's tile is all zero for these by the same
/// predicate `run` uses, so the name is the pin, not the tile.)
const EMPTY: [char; 1] = [' '];

/// `row` as the block's record: the columns, in declaration order.
fn record(row: [f32; PIECE_ROW_COLS]) -> Row {
    Row {
        x0: row[COL_X0],
        e0x: row[COL_E0X],
        e1x: row[COL_E1X],
        y0: row[COL_Y0],
        e0y: row[COL_E0Y],
        e1y: row[COL_E1Y],
        sigma: row[COL_SIGMA],
        s: row[COL_S],
        lo: row[COL_ROWS_LO],
        hi: row[COL_ROWS_HI],
    }
}

/// The sum of `terms` by the block's `sum2` alone: a balanced tree, halved
/// by index, so a glyph's pieces compose to one shape however many there
/// are, and its ink declares their uniforms in piece order.
fn ink(terms: &[Kernel]) -> Kernel {
    match terms {
        [] => panic!("a glyph with no pieces has no ink to compose"),
        [term] => term.clone(),
        _ => {
            let (left, right) = terms.split_at(terms.len() / 2);
            block::sum2(&ink(left), &ink(right))
        }
    }
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
        let built = piece_term(&|k| columns[k].kernel());
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

/// Every printable ASCII glyph with ink, composed in the language — a
/// `one_piece` instance per piece, summed by `sum2`, under `glyph`'s box —
/// draws [`glyph`]'s pixels at 7, 16 and 32 px, to twice the closed form's
/// error bound, which every coverage is held to against the exact area
/// (`tests/glyph_exact_area.rs`). The plan's B7: the equivalence gate the
/// builder's deletion (Phase C) stands on.
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
/// The box is the outline's own: the block's `inside` reaches the half
/// pixel past it that coverage does, where the builder's [`Support`] is the
/// box already dilated.
///
/// The composed program's uniforms are the glyph's box, then each piece's
/// ten in piece order: the entry's own first, then each argument's in
/// parameter order, which the balanced tree keeps in piece order — the
/// order a positional binding of the font's block reads (O3 of the plan),
/// pinned here at every piece count printable ASCII has.
///
/// One function, not one test per glyph: each `#[test]` is its own
/// process under nextest, and the JIT cache is per process, so one sweep
/// compiles each distinct program once per size — the builder's per
/// trip-count bucket, the language's per piece count — where a test per
/// glyph would saturate every glyph again. 57 s in a debug build.
#[test]
fn every_ascii_glyph_composed_in_the_language_draws_the_builders_pixels() {
    let font = Font::parse(FONT_DATA).expect("parse font");
    for size in SIZES {
        let mut spread = 0.0f64;
        let mut empties: Vec<char> = Vec::new();
        for ch in ASCII {
            let id = font
                .cmap_lookup(ch)
                .unwrap_or_else(|| panic!("the font has no glyph for {ch:?}"));
            let outline = font
                .outline_scaled_by_id(id, size as f32)
                .unwrap_or_else(|| panic!("the font has no outline for {ch:?}"));
            let rows: Vec<[f32; PIECE_ROW_COLS]> =
                pieces(&outline).into_iter().map(piece_row).collect();

            let built = glyph(&outline);
            let centred = built.kernel().at(
                &Kernel::x().add(&constant(PIXEL_CENTER)),
                &Kernel::y().add(&constant(PIXEL_CENTER)),
            );
            let by_the_builder = built
                .bake(&centred, Lattice::frame(size, size))
                .into_buffer();

            let (Some([x0, y0, x1, y1]), false) = (outline.bounds(), rows.is_empty()) else {
                assert!(
                    by_the_builder.iter().all(|&v| v == 0.0),
                    "{ch:?} at {size} px has no pieces, yet the builder drew ink"
                );
                empties.push(ch);
                continue;
            };

            let instances: Vec<Kernel> = rows.iter().map(|row| one_piece(record(*row))).collect();
            let written = block::glyph(&ink(&instances), Bounds { x0, y0, x1, y1 });
            let declared: Vec<f32> = written.uniforms().iter().map(|u| u.default).collect();
            let in_order: Vec<f32> = [x0, y0, x1, y1]
                .into_iter()
                .chain(rows.iter().flatten().copied())
                .collect();
            assert_eq!(
                declared, in_order,
                "{ch:?} at {size} px: the box, then each piece's row"
            );
            let in_the_language = Lattice::frame(size, size).bake(&written).into_buffer();

            assert!(
                by_the_builder.iter().any(|&v| v > 0.0),
                "{ch:?} at {size} px drew nothing to compare"
            );
            for (k, (&a, &b)) in by_the_builder.iter().zip(&in_the_language).enumerate() {
                let (i, j) = (k % size, k / size);
                let difference = (f64::from(a) - f64::from(b)).abs();
                assert!(
                    difference <= 2.0 * arithmetic_bound(size, [i, j]),
                    "{ch:?} at {size} px, texel ({i}, {j}): {a} by the builder, {b} in the language"
                );
                spread = spread.max(difference);
            }
        }
        eprintln!("at {size} px the builder and the language differ by at most {spread:e}");
        assert_eq!(empties, EMPTY, "the glyphs with no pieces at {size} px");
    }
}

/// The closed form's error bound at texel `(i, j)` of a `size`-px tile,
/// no arc longer than the tile: `2⁻²²·(1 + X + Y + 2·size)` at the texel's
/// centre (`tests/glyph_exact_area.rs`, `arithmetic_bound`).
fn arithmetic_bound(size: usize, [i, j]: [usize; 2]) -> f64 {
    const ARC_TOLERANCE_UNIT: f64 = 1.0 / 4_194_304.0;
    let (x, y) = (i as f64 + 0.5, j as f64 + 0.5);
    ARC_TOLERANCE_UNIT * (1.0 + x + y + 2.0 * size as f64)
}
