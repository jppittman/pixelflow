//! **The `kernel!` copy of a piece's term is the builder's.**
//!
//! Phase C of docs/plans/2026-09-25-the-language-is-kernel.md writes the
//! font in `kernel!` and deletes this builder; until then there are two
//! definitions of a piece's term, [`piece_term`] and the `kernel!` block's
//! `one_piece` (`pixelflow-compiler/tests/common/section_1_7.rs`, an entry
//! over a record of ten `f32`s, each a uniform). A copy is a future
//! divergence, so the two are pinned against each other: one piece over
//! uniforms is one canonical key through either, with the same value in
//! each uniform slot — so the JIT compiles one program for them, and they
//! cannot differ in a bit.

use super::*;
use pixelflow_ir::key::canonical;

/// The `kernel!` block's piece, expanded. This module calls `one_piece`;
/// its `Args` record goes unused here.
#[allow(dead_code)]
mod block {
    use pixelflow_compiler::kernel;

    include!("../../../../pixelflow-compiler/tests/common/section_1_7.rs");

    section_1_7!(kernel);
}

use block::{one_piece, Row};

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
