//! **The `kernel!` copy of a piece's term is the builder's.**
//!
//! Phase C of docs/plans/2026-09-25-the-language-is-kernel.md writes the
//! font in `kernel!` (§1.7) and deletes this builder; until then there are
//! two definitions of a piece's term, [`piece_term`] and the §1.7 block's
//! `piece_term`/`piece_area` (`pixelflow-compiler/tests/common/section_1_7.rs`,
//! included here as that crate's tests include it). A copy is a future
//! divergence, so the two are pinned against each other twice:
//!
//! - **as terms**: one piece over uniforms is one canonical key through
//!   either, with the same value in each uniform slot — so the JIT compiles
//!   one program for them, and they cannot differ in a bit; and
//! - **as glyphs**: real glyphs' rows, through [`glyph`]'s fold over its
//!   table and through §1.7's `glyph::<N>` over `N` copies of uniforms, bake
//!   the same pixels, to the closed form's own error bound.

use super::*;
use crate::fonts::{Font, PIXEL_CENTER};
use pixelflow_ir::key::canonical;

/// §1.7's block, expanded. Its other entry and its `Args` records are the
/// compiler's tests to exercise; this module calls `glyph` and `one_piece`.
#[allow(dead_code)]
mod block {
    use pixelflow_compiler::kernel;

    include!("../../../../pixelflow-compiler/tests/common/section_1_7.rs");

    section_1_7!(kernel);
}

use block::{one_piece, Bounds, Row};

/// The crate's font.
const FONT_DATA: &[u8] = include_bytes!("../../../assets/DejaVuSansMono-Fallback.ttf");

/// The pixel size the glyphs are drawn at, and their tile's side.
const SIZE: usize = 16;

/// The family's count: every glyph below has at most this many pieces, and
/// is padded to it with [`padding_row`]s, as [`run`] pads to its bucket.
const PIECES: usize = 32;

/// `row` as §1.7's record: the columns, in declaration order.
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

/// Real glyphs' rows bake the same pixels through [`glyph`] — the fold over
/// one table — and through §1.7's `glyph::<N>`, `N` copies of the term over
/// each row's uniforms, the box the glyph's [`Support`]: the same to the
/// closed form's own error bound, which every coverage is held to against
/// the exact area (`tests/glyph_exact_area.rs`), taken twice.
///
/// Not to the bit. A piece is one term through either definition (above),
/// but a fold over a table and a sum of copies over uniforms are two
/// programs, and the optimizer is free to extract them differently — the
/// sum's association, which products fuse, what a batch hoists. Measured:
/// they differ in the last bits of some texels, by at most `4.2·10⁻⁷` on
/// the AVX-512 and AVX2 tiers alike, where the bound here is `1.6·10⁻⁵` at
/// its smallest.
#[test]
fn real_glyphs_bake_the_same_pixels_through_either_definition() {
    let font = Font::parse(FONT_DATA).expect("parse font");
    let mut spread = 0.0f64;
    for ch in ['A', 'O', 'S', 'g', '8', 'Q'] {
        let id = font
            .cmap_lookup(ch)
            .unwrap_or_else(|| panic!("the font has no glyph for {ch:?}"));
        let outline = font
            .outline_scaled_by_id(id, SIZE as f32)
            .unwrap_or_else(|| panic!("the font has no outline for {ch:?}"));
        let rows: Vec<[f32; PIECE_ROW_COLS]> =
            pieces(&outline).into_iter().map(piece_row).collect();
        assert!(
            rows.len() <= PIECES,
            "{ch:?} has {} pieces, more than the family's {PIECES}",
            rows.len()
        );

        let built = glyph(&outline);
        let centred = built.kernel().at(
            &Kernel::x().add(&constant(PIXEL_CENTER)),
            &Kernel::y().add(&constant(PIXEL_CENTER)),
        );
        let by_fold = built
            .bake(&centred, Lattice::frame(SIZE, SIZE))
            .into_buffer();

        let family: [Row; PIECES] =
            core::array::from_fn(|k| record(rows.get(k).copied().unwrap_or_else(padding_row)));
        let [x0, y0, x1, y1] = built.support.bounds();
        let written = block::glyph::<PIECES>(family, Bounds { x0, y0, x1, y1 });
        let by_copies = Lattice::frame(SIZE, SIZE).bake(&written).into_buffer();

        assert!(
            by_fold.iter().any(|&v| v > 0.0),
            "{ch:?} drew nothing to compare"
        );
        for (k, (&a, &b)) in by_fold.iter().zip(&by_copies).enumerate() {
            let (i, j) = (k % SIZE, k / SIZE);
            let difference = (f64::from(a) - f64::from(b)).abs();
            assert!(
                difference <= 2.0 * arithmetic_bound([i, j]),
                "{ch:?} texel ({i}, {j}): {a} by the fold, {b} by the copies"
            );
            spread = spread.max(difference);
        }
    }
    eprintln!("the fold and the copies differ by at most {spread:e}");
}

/// The closed form's error bound at texel `(i, j)` of a [`SIZE`]-px tile,
/// no arc longer than the tile: `2⁻²²·(1 + X + Y + 2·SIZE)` at the texel's
/// centre (`tests/glyph_exact_area.rs`, `arithmetic_bound`).
fn arithmetic_bound([i, j]: [usize; 2]) -> f64 {
    const ARC_TOLERANCE_UNIT: f64 = 1.0 / 4_194_304.0;
    let (x, y) = (i as f64 + 0.5, j as f64 + 0.5);
    ARC_TOLERANCE_UNIT * (1.0 + x + y + 2.0 * SIZE as f64)
}
