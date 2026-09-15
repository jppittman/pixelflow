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
use pixelflow_ir::arena::{ExprArena, ExprId};
use pixelflow_ir::{ExprNode, LatticeShape};

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// The characters and size `collapse_cost`'s corpus benches, plus two the
/// atlas warms. `'A'` is line-segment only, `'O'` is all quadratics, `'8'`
/// crosses zero twice per scanline, `'g'` has a descender and two contours.
const CHARS: [char; 5] = ['A', 'O', 'S', '8', 'g'];
const SIZES: [usize; 2] = [16, 32];

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

/// Whether a binder survived to the emitter's input. It must not: codegen has
/// no iteration binder, so a fold here is a legalizer that did not run.
fn has_fold(arena: &ExprArena, root: ExprId) -> bool {
    let mut seen = vec![false; arena.len()];
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        if matches!(arena.node(id), ExprNode::Reduce { .. }) {
            return true;
        }
        stack.extend(arena.children(id));
    }
    false
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
            let (arena, root) = coverage.parts();
            let built = reachable(arena, root);
            let shape = LatticeShape::new([size as u32, size as u32]);

            let t0 = Instant::now();
            let optimized = pixelflow_search::runtime::optimize_runtime_arena(arena, root, shape);
            let ms = t0.elapsed().as_secs_f64() * 1e3;

            let (out, out_root) = match optimized.as_deref() {
                Some((a, r)) => (a.clone(), *r),
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
                reachable(&out, out_root),
                has_fold(&out, out_root)
            );
        }
    }
}
