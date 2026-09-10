//! Wall clock for collapse bodies at real shapes, and the first scoring of
//! static predictors against it.
//!
//! Three subcommands, in the order they are used:
//!
//! ```text
//! # once, on any build — writes the fixture the other two replay
//! collapse_cost capture --out corpus/collapse
//!
//! # once per (allocation variant × ISA tier), built from that variant's ref
//! collapse_cost bench --corpus corpus/collapse --out rows/main.sse2.jsonl \
//!     --git-ref main --passes 2
//!
//! # once, over every jsonl the sweep produced
//! collapse_cost analyze --rows rows/*.jsonl --out report.md
//! ```
//!
//! Why it is shaped this way: the corpus has to be *identical* across the
//! variants being compared, and a corpus generated inside each run would not
//! be — it would be whatever that build's font parsing and kernel
//! construction produced. Capturing once to files makes the input a fixture
//! with a diff. See `pixelflow_pipeline::collapse_bench` for what is timed and
//! why, and `docs/plans/2026-09-01-register-allocation-escape-hatches.md` for
//! the measured contradiction that asked for it.

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use pixelflow_graphics::fonts::Glyph;
use pixelflow_ir::arena::ExprArena;
use pixelflow_pipeline::collapse_bench::{
    self,
    corpus::{self, CollapseKernel},
};

/// Printable ASCII: what `GlyphAtlas::warm` bakes on startup.
const WARM_RANGE: std::ops::RangeInclusive<char> = ' '..='~';
/// Cell height core-term asks the atlas for, in points.
const CELL_HEIGHT_PT: f32 = 16.0;
/// Display densities core-term builds an atlas at: 1.0 on startup, then the
/// window's own (2.0 on a Retina display).
const DENSITIES: [f32; 2] = [1.0, 2.0];
const ATLAS_CAPACITY: usize = 128;

/// The three characters and the lattice `pixelflow-graphics`'s
/// `font_rendering` bench bakes — the exact measurement the 13–18% AVX-512
/// regression was reported on, so the corpus contains the thing the
/// contradiction was observed on and not only its neighbours.
const BENCH_CHARS: [(&str, char); 3] =
    [("A_linear", 'A'), ("O_quadratic", 'O'), ("S_complex", 'S')];
const BENCH_PT: f32 = 32.0;
const BENCH_EXTENT: [u32; 2] = [40, 45];

#[derive(Parser)]
#[command(
    name = "collapse_cost",
    about = "Time collapse kernels at their bake shapes and score static predictors against the clock"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Write the corpus fixture: the production glyph bakes plus the
    /// synthetic pressure families.
    Capture {
        /// Directory to write `.collapse` files into.
        #[arg(long)]
        out: PathBuf,
        /// The TTF to bake glyphs from. Defaults to the face
        /// `pixelflow-graphics`'s own `font_rendering` bench uses.
        #[arg(long)]
        font: Option<PathBuf>,
    },
    /// Compile and time every corpus kernel on this build.
    Bench {
        #[arg(long)]
        corpus: PathBuf,
        /// JSONL destination, one row per (kernel, pass).
        #[arg(long)]
        out: PathBuf,
        /// Human name of the allocation variant this build came from.
        #[arg(long)]
        git_ref: String,
        /// The commit that variant is. Defaults to this checkout's HEAD,
        /// which is right when the binary runs where it was built and wrong
        /// when a sweep builds several variants and runs them side by side —
        /// so a sweep passes it, rather than recording a plausible lie.
        #[arg(long)]
        git_sha: Option<String>,
        /// Sweeps of the corpus. Two or more give the A/A noise floor.
        #[arg(long, default_value_t = 2)]
        passes: u32,
        /// Recorded in every row; say which cargo profile built this.
        #[arg(long, default_value = "release")]
        profile: String,
    },
    /// Score the candidate predictors over one or more benched JSONL files.
    Analyze {
        #[arg(long, num_args = 1..)]
        rows: Vec<PathBuf>,
        /// Markdown destination. Written to stdout as well.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Which timing estimator to score against: `median` (the headline),
        /// `min`, or `drift` (the sentinel-corrected median).
        #[arg(long, default_value = "median")]
        stat: String,
    },
}

fn main() {
    match Cli::parse().command {
        Command::Capture { out, font } => capture(&out, font.as_deref()),
        Command::Bench {
            corpus,
            out,
            git_ref,
            git_sha,
            passes,
            profile,
        } => bench(
            &corpus,
            &out,
            &Build {
                git_ref,
                git_sha: git_sha.unwrap_or_else(head_sha),
                profile,
            },
            passes,
        ),
        Command::Analyze { rows, out, stat } => analyze(&rows, out.as_deref(), &stat),
    }
}

/// Captured contents for each buffer `arena` declares, matched to the data
/// `glyph.kernel` itself carries
/// ([`Kernel::buffer_data`](pixelflow_ir::Kernel::buffer_data)) by
/// [`BufferIdentity`](pixelflow_ir::arena::BufferIdentity) — the winding
/// sum's real piece table, not a restatement of its shape. `None` at a slot
/// means this glyph declared a buffer its kernel carries no data for, which
/// `dummy_context` reports loudly at replay rather than silently zeroing.
fn buffer_data_for(arena: &ExprArena, glyph: &Glyph) -> Vec<Option<Arc<Vec<f32>>>> {
    arena
        .buffers()
        .iter()
        .map(|decl| {
            glyph
                .kernel()
                .buffer_data()
                .find(|(id, _)| *id == decl.id)
                .map(|(_, data)| Arc::new(data.to_vec()))
        })
        .collect()
}

fn capture(out: &std::path::Path, font: Option<&std::path::Path>) {
    // core-term itself loads `NotoSansMono-Regular.ttf`; that asset is stored
    // in large-file storage and is a pointer file in checkouts without it, so
    // the default is the fallback face the `font_rendering` bench already
    // bakes. The corpus's shapes — the atlas tile sizes and the bench lattice
    // — are core-term's either way, which is what the measurement rests on.
    let default = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../pixelflow-graphics/assets/DejaVuSansMono-Fallback.ttf");
    let font_path = font.unwrap_or(&default);
    let data =
        std::fs::read(font_path).unwrap_or_else(|e| panic!("read {}: {e}", font_path.display()));
    let parsed = pixelflow_graphics::fonts::Font::parse(&data).expect("parse the production font");

    let mut kernels = corpus::synthetic();
    let synthetic_count = kernels.len();
    let mut missing = 0usize;
    for density in DENSITIES {
        // The atlas is what decides the tile size, and the tile size is the
        // bake extent — so ask it rather than restating its arithmetic.
        let atlas =
            pixelflow_graphics::fonts::GlyphAtlas::new(CELL_HEIGHT_PT, density, ATLAS_CAPACITY);
        let tile = atlas.tile_px() as u32;
        for ch in WARM_RANGE {
            let Some(glyph) = parsed.glyph_kernel_scaled(ch, tile as f32) else {
                // Production bakes nothing for these either (atlas.rs leaves
                // the slot blank), so they are not kernels.
                missing += 1;
                continue;
            };
            // Linked: the winding sum is composed by reference, and the
            // corpus holds the arena the optimizer sees — the referent
            // spliced in, declaring the table it reads.
            let coverage = glyph.kernel();
            let (arena, root) = coverage.parts();
            let (arena, root) = pixelflow_ir::passes::expand_refs_owned(arena, root);
            let buffer_data = buffer_data_for(&arena, &glyph);
            kernels.push(CollapseKernel {
                name: format!("glyph{tile}_U{:04X}", ch as u32),
                family: format!("glyph{tile}"),
                arena,
                root,
                extent: [tile, tile],
                buffer_data,
            });
        }
    }
    for (label, ch) in BENCH_CHARS {
        let glyph = parsed
            .glyph_kernel_scaled(ch, BENCH_PT)
            .unwrap_or_else(|| panic!("the font has no glyph for {ch:?}"));
        let coverage = glyph.kernel();
        let (arena, root) = coverage.parts();
        let (arena, root) = pixelflow_ir::passes::expand_refs_owned(arena, root);
        let buffer_data = buffer_data_for(&arena, &glyph);
        kernels.push(CollapseKernel {
            name: format!("bench_{label}"),
            family: "bench".to_string(),
            arena,
            root,
            extent: BENCH_EXTENT,
            buffer_data,
        });
    }
    corpus::write_dir(out, &kernels);
    println!(
        "wrote {} kernels ({synthetic_count} synthetic, {} glyph) to {}; \
         the font has no glyph for {missing} of the warmed characters",
        kernels.len(),
        kernels.len() - synthetic_count,
        out.display()
    );
}

/// Which build produced the rows — recorded in every one of them.
struct Build {
    git_ref: String,
    git_sha: String,
    profile: String,
}

fn bench(corpus_dir: &std::path::Path, out: &std::path::Path, build: &Build, passes: u32) {
    let kernels = corpus::read_dir(corpus_dir);
    assert!(
        !kernels.is_empty(),
        "{}: empty corpus",
        corpus_dir.display()
    );
    let usable: Vec<CollapseKernel> = kernels
        .into_iter()
        .filter(|k| k.extent[0] >= collapse_bench::LANES as u32)
        .collect();
    eprintln!(
        "benching {} kernels on {} ({} lanes), {passes} pass(es)",
        usable.len(),
        collapse_bench::tier(),
        collapse_bench::LANES
    );
    let rows = collapse_bench::run_corpus(&usable, passes);
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|e| panic!("create {}: {e}", parent.display()));
    }
    collapse_bench::write_jsonl(out, &rows, &build.git_ref, &build.git_sha, &build.profile);
    println!("wrote {} rows to {}", rows.len(), out.display());
}

fn analyze(rows: &[PathBuf], out: Option<&std::path::Path>, stat: &str) {
    let stat = collapse_bench::Stat::parse(stat).unwrap_or_else(|e| panic!("--stat: {e}"));
    let rows = collapse_bench::read_jsonl(rows);
    let report = collapse_bench::predict::report(&rows, stat);
    if let Some(path) = out {
        std::fs::write(path, &report).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    }
    print!("{report}");
}

/// The commit this binary was built from, or `unknown` outside a checkout.
fn head_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short=8", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}
