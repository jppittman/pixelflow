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
use pixelflow_ir::passes::{expand_refs_owned, legalize};
use pixelflow_ir::{Environment, Rooted};

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// The piece table both folds read has this many columns per piece
/// (`loop_blinn::PIECE_ROW_COLS`, private to that module).
const PIECE_ROW_COLS: usize = 22;

fn reachable(root: pixelflow_ir::Node<'_, pixelflow_ir::ExprData>) -> usize {
    root.node_count()
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
        let env = Environment {
            buffers: kernel.buffers().to_vec(),
            uniforms: kernel.uniforms().to_vec(),
        };
        let (legacy, legacy_root) = kernel.root().marshal(&env);
        let pieces: usize = kernel
            .buffer_data()
            .map(|(_, data)| data.len())
            .sum::<usize>()
            / PIECE_ROW_COLS;
        let (linked, linked_root) = expand_refs_owned(&legacy, legacy_root);
        let t1 = Instant::now();
        let (legal, legal_root) = legalize(&legacy, legacy_root).expect("legalize");
        let legalize_t = t1.elapsed();
        println!(
            "{n:>5}  {:>12}  {:>9}  {:>9}  {pieces:>6}  {:>15}  {:>19}  {:>11}",
            construct.as_micros(),
            kernel.root().dag().len(),
            kernel.root().node_count(),
            Rooted::unmarshal(&linked, &[linked_root])
                .0
                .entry()
                .node_count(),
            Rooted::unmarshal(&legal, &[legal_root])
                .0
                .entry()
                .node_count(),
            legalize_t.as_micros(),
        );
    }
}
