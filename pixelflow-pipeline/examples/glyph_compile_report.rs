//! Shape B of `docs/plans/2026-09-06-egraph-at-production-scale.md` §5.3:
//! the compile cost of a glyph warm, per kernel, with the emitted bytes.
//!
//! ```text
//! cargo run --release -p pixelflow-pipeline --example glyph_compile_report          # rows
//! PIXELFLOW_GLYPH_REPORT_MODE=warm \
//!   cargo run --release -p pixelflow-pipeline --example glyph_compile_report        # atlas.warm
//! ```
//!
//! The default mode prints one TSV row per production glyph bake — the
//! printable ASCII range `GlyphAtlas::warm` bakes on startup, at core-term's
//! 16 pt cell height and both densities it builds an atlas at (tile 16 and
//! tile 32) — carrying the compile time through the same path `Lattice::bake`
//! takes (`compile_as_baked`: runtime optimization at the bake shape, then
//! emit), the emitted byte count, and a hash of the emitted bytes. Each kernel
//! is compiled **once**: `optimize_runtime_arena` memoizes per process, so a
//! second pass in the same process would time the emitter alone. Repeat the
//! process and take medians per kernel. Two builds' rows diffed on the hash
//! column say *which* glyph kernels a change touched; diffed on the time
//! column, what it cost at startup.
//!
//! `warm` mode times one cold `GlyphAtlas::warm` per density — the number a
//! user actually waits on — and nothing else, so the JIT and optimizer caches
//! are empty when it runs.
//!
//! Compare two builds' rows, never a row against prose: per-kernel clocks on
//! a shared host are only trustworthy as a paired difference.

use std::time::Instant;

use pixelflow_graphics::fonts::{Font, GlyphAtlas};
use pixelflow_pipeline::collapse_bench::compile_as_baked;

/// core-term's font asset (`terminal_app.rs`: `FONT_FILENAME`).
const FONT_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../pixelflow-graphics/assets/NotoSansMono-Regular.ttf"
);
/// What `GlyphAtlas::warm` bakes on startup.
const WARM_RANGE: std::ops::RangeInclusive<char> = ' '..='~';
/// Cell height core-term asks the atlas for, in points.
const CELL_HEIGHT_PT: f32 = 16.0;
/// Display densities core-term builds an atlas at: 1.0 on startup, then the
/// window's own (2.0 on a Retina display).
const DENSITIES: [f32; 2] = [1.0, 2.0];
const ATLAS_CAPACITY: usize = 128;
const MODE_VAR: &str = "PIXELFLOW_GLYPH_REPORT_MODE";

enum Mode {
    Kernels,
    Warm,
}

fn mode() -> Mode {
    match std::env::var(MODE_VAR) {
        Ok(v) if v == "kernels" => Mode::Kernels,
        Ok(v) if v == "warm" => Mode::Warm,
        Ok(v) => panic!("{MODE_VAR}={v:?}: expected \"kernels\" or \"warm\""),
        Err(std::env::VarError::NotPresent) => Mode::Kernels,
        Err(e) => panic!("{MODE_VAR}: {e}"),
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    bytes
        .iter()
        .fold(OFFSET, |h, &b| (h ^ u64::from(b)).wrapping_mul(PRIME))
}

fn tile_for(density: f32) -> u32 {
    // The atlas decides the tile size, and the tile size is the bake extent
    // — ask it rather than restating its arithmetic.
    GlyphAtlas::new(CELL_HEIGHT_PT, density, ATLAS_CAPACITY).tile_px() as u32
}

fn kernels(font: &Font) {
    println!("kernel\ttile\tcompile_ms\tbytes\tcode_fnv1a64");
    for density in DENSITIES {
        let tile = tile_for(density);
        let mut kernels = 0usize;
        let mut sum_ms = 0.0f64;
        let mut sum_bytes = 0usize;
        for ch in WARM_RANGE {
            // Production bakes nothing for these either (the atlas leaves the
            // slot blank), so they are not kernels.
            let Some(glyph) = font.glyph_kernel_scaled(ch, tile as f32) else {
                continue;
            };
            let coverage = glyph.kernel();
            let (arena, root) = coverage.parts();
            let started = Instant::now();
            let result = compile_as_baked(arena, root, [tile, tile]);
            let ms = started.elapsed().as_secs_f64() * 1e3;
            let bytes = result.code.len();
            println!(
                "glyph{tile}_U{:04X}\t{tile}\t{ms:.3}\t{bytes}\t{:016x}",
                ch as u32,
                fnv1a64(result.code.as_bytes())
            );
            kernels += 1;
            sum_ms += ms;
            sum_bytes += bytes;
        }
        println!(
            "# tile {tile} (density {density}): {kernels} kernels, {sum_ms:.1} ms of compile, \
             {sum_bytes} bytes of code"
        );
    }
}

fn warm(font: &Font) {
    for density in DENSITIES {
        let mut atlas = GlyphAtlas::new(CELL_HEIGHT_PT, density, ATLAS_CAPACITY);
        let started = Instant::now();
        atlas.warm(font, WARM_RANGE);
        println!(
            "# atlas.warm tile {} (density {density}): {:.1} ms, cold",
            atlas.tile_px(),
            started.elapsed().as_secs_f64() * 1e3
        );
    }
}

fn main() {
    let data = std::fs::read(FONT_PATH).unwrap_or_else(|e| panic!("read {FONT_PATH}: {e}"));
    let font = Font::parse(&data).expect("parse the production font");
    match mode() {
        Mode::Kernels => kernels(&font),
        Mode::Warm => warm(&font),
    }
}
