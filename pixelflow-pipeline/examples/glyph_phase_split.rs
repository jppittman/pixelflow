//! Where a glyph's compile time actually goes, and how many distinct
//! programs a font needs.
//!
//! ```text
//! cargo run --release -p pixelflow-pipeline --example glyph_phase_split
//! ```
//!
//! Two questions, one walk over the glyphs `GlyphAtlas::warm` bakes at
//! startup.
//!
//! **Where the time goes.** `compile_as_baked` is two phases —
//! `optimize_runtime_arena` (saturate, then extract) and `compile` (emit) —
//! and which of them dominates decides which backlog item is worth doing.
//! `emit` is ~O(n^1.6) in straight-line instruction count, so if it dominates
//! then keeping a fold as a loop is the lever; if optimization dominates then
//! the e-graph items are. Nobody has split this since the guard-partition fix
//! took a bake 31.5 s → 20.0 s, and the last two guesses about this hump were
//! both wrong (`cluster_select_arms`, not regalloc; compile, not collapse), so
//! it is measured rather than reasoned about.
//!
//! **How many programs.** A fold's trip count is a `Fold` field, hashed into
//! the `KernelKey` the JIT cache keys on, so *a different trip count is a
//! different program* — that is the language's premise, not an accident
//! (CLAUDE.md: "a trip count that must change is a recompile"). One program
//! for the whole font therefore needs every glyph padded to the font-wide
//! maximum, and every glyph then pays the worst glyph's work at every pixel,
//! because a constant trip count cannot exit early and a `Select` computes
//! both arms.
//!
//! So this counts the **distinct trip counts**, which is the number of
//! programs a font actually needs, and what bucketing them to a power of two
//! would cost: the padding factor is bounded below 2× per glyph, against
//! `max/min` for one global program.
//!
//! Compare two builds' rows, never a row against prose: per-kernel clocks on
//! a shared host are only trustworthy as a paired difference.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use pixelflow_graphics::fonts::{Font, GlyphAtlas};
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};

/// core-term's font asset (`terminal_app.rs`: `FONT_FILENAME`).
const FONT_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../pixelflow-graphics/assets/NotoSansMono-Regular.ttf"
);
/// What `GlyphAtlas::warm` bakes on startup.
const WARM_RANGE: std::ops::RangeInclusive<char> = ' '..='~';
/// Cell height core-term asks the atlas for, in points.
const CELL_HEIGHT_PT: f32 = 16.0;
/// Display densities core-term builds an atlas at.
const DENSITIES: [f32; 2] = [1.0, 2.0];
const ATLAS_CAPACITY: usize = 128;

fn tile_for(density: f32) -> u32 {
    GlyphAtlas::new(CELL_HEIGHT_PT, density, ATLAS_CAPACITY).tile_px() as u32
}

/// Every fold trip count reachable from `root`, largest first.
///
/// These are the numbers baked into the program. Two glyphs agree on a
/// program only if they agree on this whole multiset, so it — not the piece
/// count, which is a proxy for it — is what a cache key partitions on.
fn trip_counts(arena: &ExprArena, root: ExprId) -> Vec<u32> {
    let mut seen = vec![false; arena.len()];
    let mut stack = vec![root];
    let mut trips = Vec::new();
    while let Some(id) = stack.pop() {
        if core::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        if let ExprNode::Reduce { fold, .. } = arena.node(id) {
            trips.push(fold.len());
        }
        stack.extend(arena.children(id));
    }
    trips.sort_unstable_by(|a, b| b.cmp(a));
    trips
}

/// The next power of two at or above `n` — the bucket a quantized trip count
/// would round up to.
fn bucket(n: u32) -> u32 {
    n.max(1).next_power_of_two()
}

fn main() {
    let data = std::fs::read(FONT_PATH).unwrap_or_else(|e| panic!("read {FONT_PATH}: {e}"));
    let font = Font::parse(&data).expect("parse the production font");

    println!("kernel\ttile\toptimize_ms\temit_ms\tnodes_in\tnodes_out\tbytes\ttrips");
    for density in DENSITIES {
        let tile = tile_for(density);
        let mut opt_ms = 0.0f64;
        let mut emit_ms = 0.0f64;
        let mut kernels = 0usize;
        // The distinct programs a font needs: one per distinct trip-count
        // multiset, and one per distinct bucketed multiset.
        let mut exact: BTreeSet<Vec<u32>> = BTreeSet::new();
        let mut bucketed: BTreeSet<Vec<u32>> = BTreeSet::new();
        // Padding cost of one global program: every glyph runs the maximum.
        let mut max_trip = 0u32;
        let mut trip_histogram: BTreeMap<u32, usize> = BTreeMap::new();

        for ch in WARM_RANGE {
            let Some(glyph) = font.glyph_kernel_scaled(ch, tile as f32) else {
                continue;
            };
            let coverage = glyph.kernel();
            let (arena, root) = coverage.parts();
            let nodes_in = arena.len();
            let trips = trip_counts(arena, root);

            // Phase 1: saturate and extract.
            let started = Instant::now();
            let optimized = pixelflow_search::runtime::optimize_runtime_arena(
                arena,
                root,
                pixelflow_ir::LatticeShape::new([tile, tile]),
            );
            let opt = started.elapsed().as_secs_f64() * 1e3;
            let (oarena, oroot) = optimized
                .as_deref()
                .map(|(a, r)| (a, *r))
                .unwrap_or((arena, root));
            let nodes_out = oarena.len();

            // Phase 2: emit. Compiled through the same entry the bake uses,
            // so the two phases sum to what `compile_as_baked` costs.
            let started = Instant::now();
            let result = pixelflow_codegen::emit::compile(oarena, oroot)
                .expect("a production glyph compiles");
            let emit = started.elapsed().as_secs_f64() * 1e3;

            for &t in &trips {
                *trip_histogram.entry(t).or_insert(0) += 1;
                max_trip = max_trip.max(t);
            }
            exact.insert(trips.clone());
            bucketed.insert(trips.iter().copied().map(bucket).collect());

            println!(
                "glyph{tile}_U{:04X}\t{tile}\t{opt:.3}\t{emit:.3}\t{nodes_in}\t{nodes_out}\t{}\t{:?}",
                ch as u32,
                result.code.len(),
                trips
            );
            opt_ms += opt;
            emit_ms += emit;
            kernels += 1;
        }

        let total = opt_ms + emit_ms;
        println!(
            "# tile {tile} (density {density}): {kernels} kernels, {total:.1} ms total — \
             optimize {opt_ms:.1} ms ({:.1}%), emit {emit_ms:.1} ms ({:.1}%)",
            100.0 * opt_ms / total,
            100.0 * emit_ms / total,
        );
        let min_trip = trip_histogram.keys().copied().next().unwrap_or(0);
        println!(
            "# tile {tile}: {kernels} glyphs need {} distinct programs exactly, \
             {} bucketed to powers of two; trips {min_trip}..={max_trip}",
            exact.len(),
            bucketed.len(),
        );
        println!("# tile {tile}: trip histogram {trip_histogram:?}");
    }
}
