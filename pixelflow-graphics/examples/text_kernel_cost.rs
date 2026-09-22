//! What a laid-out string costs to *build* and to *legalize*, as the string
//! grows: arena size, reachable nodes, piece count, and two node counts
//! downstream. One line per length. Every column must stay linear in the
//! piece count — the measurement behind
//! docs/plans/2026-09-09-composition-is-linking.md §7.
//!
//! The two downstream columns are different questions and the difference
//! matters, because the runtime pipeline splits between them:
//!
//! - `saturation_sees` is reference-linked and nothing else — exactly what
//!   `pixelflow_search::runtime` hands the e-graph, since legalization
//!   (`LowerDwrt`, `ExpandReduce`) runs *after* saturation as the fallback
//!   for shapes the rule set declined. It is the lever on optimization cost,
//!   and holding it down is why the legalizer sits at the end.
//! - `legalized` is the whole of `legalize` on the *unoptimized* arena, so
//!   it also expands every `Gather` into address arithmetic. Nothing folds
//!   that here, whereas in the pipeline saturation runs first and CSEs the
//!   repeated reads — so this column over-counts what the JIT emits, and it
//!   over-counts more the more the kernel reads a table.
//!
//! Run: `cargo run --release -p pixelflow-graphics --example text_kernel_cost`

use std::time::Instant;

use pixelflow_core::Kernel;
use pixelflow_graphics::fonts::{text, Font};
use pixelflow_ir::arena::{ExprArena, ExprId};
use pixelflow_ir::passes::lattice::{Collapse, Domain};
use pixelflow_ir::passes::legalize;
use pixelflow_ir::LatticeShape;

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// The piece table both folds read has this many columns per piece
/// (`loop_blinn::PIECE_ROW_COLS`, private to that module).
const PIECE_ROW_COLS: usize = 22;

/// The lattice `legalize` wraps the kernel for, held fixed across every
/// string length: this file measures how node counts scale with the piece
/// count, not with the canvas, so any shape would do as long as it is the
/// same one for every row.
const MEASURE_SHAPE: LatticeShape = LatticeShape::new([64, 64]);

fn reachable(arena: &ExprArena, root: ExprId) -> usize {
    let mut seen = vec![false; arena.len()];
    let mut stack = vec![root];
    let mut n = 0;
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        n += 1;
        stack.extend(arena.children(id));
    }
    n
}

fn main() {
    let font = Font::parse(FONT_BYTES).expect("parse font");
    let size = 32.0f32;
    let alphabet = "abcdefghijklmnopqrstuvwxyz";
    println!(
        "chars  construct_us  arena_len  reachable  pieces  saturation_sees  \
         legalized_reachable  legalize_us"
    );
    for n in [1usize, 2, 4, 8, 13, 26] {
        let s = &alphabet[..n];
        let t0 = Instant::now();
        let glyph = text(&font, s, size);
        let kernel = glyph.kernel().at(
            &Kernel::x().add(&Kernel::constant(0.5)),
            &Kernel::y().add(&Kernel::constant(0.5)),
        );
        let construct = t0.elapsed();
        let (arena, root) = kernel.parts();
        let pieces: usize = kernel
            .buffer_data()
            .map(|(_, data)| data.len())
            .sum::<usize>()
            / PIECE_ROW_COLS;
        let (linked, linked_root) = kernel.linked_parts();
        let t1 = Instant::now();
        let collapse = Collapse {
            domain: Domain {
                shape: MEASURE_SHAPE,
                origin: pixelflow_codegen::emit::origin(),
            },
            lanes: (pixelflow_codegen::jit_vector_bytes() / 4) as u32,
        };
        let (legal, legal_root) = legalize(arena, root, &collapse).expect("legalize");
        let legalize_t = t1.elapsed();
        println!(
            "{n:>5}  {:>12}  {:>9}  {:>9}  {pieces:>6}  {:>15}  {:>19}  {:>11}",
            construct.as_micros(),
            arena.len(),
            reachable(arena, root),
            reachable(&linked, linked_root),
            reachable(&legal, legal_root),
            legalize_t.as_micros(),
        );
    }
}
