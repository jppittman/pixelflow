//! What the runtime tier costs on a production glyph: how long saturation
//! takes, and how big the arena it hands the emitter is.
//!
//! The measurement the pipeline *order* turns on. `optimize_runtime_arena` is
//! a composition — link, saturate, legalize — and moving the legalizing steps
//! from before the e-graph to after it changes both columns without changing
//! the pixels, so neither goldens nor `kernel_glyph_optimize` can tell you
//! whether the move was worth making. This can.
//!
//! Run: `cargo run --release -p pixelflow-graphics --example glyph_saturation_cost`

use std::time::Instant;

use pixelflow_graphics::fonts::Font;
use pixelflow_ir::{Environment, ExprData, LatticeShape};

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// The characters and size `collapse_cost`'s corpus benches, plus two the
/// atlas warms. `'A'` is line-segment only, `'O'` is all quadratics, `'8'`
/// crosses zero twice per scanline, `'g'` has a descender and two contours.
const CHARS: [char; 5] = ['A', 'O', 'S', '8', 'g'];
const SIZES: [usize; 2] = [16, 32];

fn reachable(root: pixelflow_ir::Node<'_, ExprData>) -> usize {
    root.node_count()
}

/// Whether a binder survived to the emitter's input. It must not: codegen has
/// no iteration binder, so a fold here is a legalizer that did not run.
fn has_fold(root: pixelflow_ir::Node<'_, ExprData>) -> bool {
    root.descendants()
        .any(|node| matches!(*node, ExprData::Reduce(_)))
}

fn main() {
    let font = Font::parse(FONT_BYTES).expect("parse font");
    println!("char\tsize\tbuilt\toptimized\tms\tfold_survived");
    for size in SIZES {
        for ch in CHARS {
            let Some(glyph) = font.glyph_kernel_scaled(ch, size as f32) else {
                continue;
            };
            let coverage = glyph.kernel();
            let env = Environment {
                buffers: coverage.buffers().to_vec(),
                uniforms: coverage.uniforms().to_vec(),
            };
            let built = reachable(coverage.root());
            let shape = LatticeShape::new([size as u32, size as u32]);

            let t0 = Instant::now();
            let optimized =
                pixelflow_search::runtime::optimize_runtime_dag(coverage.rooted(), &env, shape);
            let ms = t0.elapsed().as_secs_f64() * 1e3;

            let out_root = match optimized.as_deref() {
                Some((rooted, _)) => rooted.entry(),
                // The pipeline declined outright — which after the reorder
                // would mean the emitter gets an un-legalized arena, so it is
                // worth seeing rather than averaging away.
                None => {
                    println!("{ch}\t{size}\t{built}\tDECLINED\t{ms:.1}\t?");
                    continue;
                }
            };
            println!(
                "{ch}\t{size}\t{built}\t{}\t{ms:.1}\t{}",
                reachable(out_root),
                has_fold(out_root)
            );
        }
    }
}
