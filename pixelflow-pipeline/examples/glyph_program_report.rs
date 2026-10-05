//! What compiling a font's glyphs as **one program** costs, per glyph count:
//! the N-glyph row of the CL6 baseline
//! (`docs/results/2026-10-05-cl6-baseline.md`).
//!
//! ```text
//! PIXELFLOW_PROBE_FONT=/path/to/NotoSansMono-Regular.ttf \
//!   cargo run --release -p pixelflow-pipeline --example glyph_program_report -- 32 [tile]
//! ```
//!
//! One count per process, on purpose. `optimize_runtime_arena` memoizes per
//! process and `jit_cache` caches what it compiled, so a sweep of 4, 8, 16, 32
//! in one process would hand each later row the earlier rows' work, and
//! `VmHWM` (the process's peak resident set) only ever rises, so after the
//! first row it would name the largest row so far and no other. Run it once
//! per count:
//!
//! ```text
//! for n in 4 8 16 32; do glyph_program_report $n; done
//! ```
//!
//! The program is the first `N` inked glyphs of the printable ASCII range at
//! core-term's 16 pt cell height (the atlas's tile at density 1.0), each the
//! production kernel `Font::glyph_kernel_scaled` builds — a fold over its
//! piece table, with the table travelling in the kernel — shifted to texel
//! centres the way `GlyphAtlas` bakes it, made a unit with `Kernel::by_ref`,
//! and chosen between by a balanced `if id < k` tree over one uniform. That
//! is the shape `docs/plans/2026-09-25-the-language-is-kernel.md` §1.7 compiles
//! a font as. Each glyph's fold varies with the column, so it is a surviving
//! fold of its own to scope, allocate and frame, which is what makes this the
//! fixture for a change to how those scale (the optimizer unrolls a glyph with
//! few enough pieces, so a few fewer than `N` survive). The tile is a square
//! of `tile` texels, 16 by default (core-term's atlas at density 1.0): a
//! multiple of the lane count of both x86 tiers, so **at the default there is
//! no remainder column fold** and each glyph's fold is carved once. A `tile`
//! that is not a multiple (`17`) gives the lattice a remainder, carves each
//! glyph's fold into the main and the remainder column folds, and about
//! doubles the folds: the sibling-fold shape.
//!
//! It goes through `jit_cache::compile`, the production entry (what
//! `Manifold::compile` calls), and not `Manifold::compile` itself: that
//! refuses a kernel declaring more than `MAX_BOUND_BUFFERS` buffers, and `N`
//! glyphs declare `N` piece tables. Nothing here is bound or run; this is a
//! compile cost.
//!
//! One TSV row. The columns up to `emit_frame_bytes` are deterministic and are
//! the ones to diff between two builds (cut from `optimize_ms` on, as
//! `glyph_compile_report`'s rows are compared with its time column cut); the
//! rest are a measurement and move run to run. Three stages, each timed and
//! counted ([`pixelflow_pipeline::alloc_probe`]: bytes requested, and the
//! peak net growth of the live heap over the stage's start):
//!
//! | stage | what runs |
//! |---|---|
//! | `optimize_*` | `optimize_runtime_arena`, cold. Its answer is memoized and stays resident, so the stages after it start from it |
//! | `compile_*` | `jit_cache::compile`, the production entry: with the optimizer's answer found in its memo, the canonical key, the link and the emit. Its bytes are the row's |
//! | `emit_*` | `emit::compile` alone over the optimized arena, for what only `CompileResult` carries; not the production bytes, since it skips the link's renumbering of the program's buffers and uniform |
//!
//! | column | meaning |
//! |---|---|
//! | `n` `tile` | glyphs in the program; its lattice's edge |
//! | `arena_nodes` | the program's arena before any optimization (units are leaves in it) |
//! | `bytes` `fnv` | the production code, and `fnv1a64` of it |
//! | `guards` `arms` `entries` | what it branched over (`BranchTraffic`) |
//! | `optimized` | whether the optimizer took the program (a declined one still compiles, unoptimized) |
//! | `emit_scopes` | scopes in the nest (the body plus every surviving fold) |
//! | `emit_hoisted` `emit_frame_bytes` | values parked for inner scopes; the frame's `m` |
//! | `*_ms` `*_alloc_mib` `*_peak_mib` | each stage's wall time, bytes requested and peak net heap growth |
//! | `vm_hwm_mib` | the process's peak resident set after the `compile` stage, from `/proc/self/status` (the `emit` stage runs after it) |
//!
//! The stages inside the emitter (scoping, allocation, emission) are not
//! reachable from an example; the codegen crate's own measurement
//! (`emit::tests::sibling_folds::sibling_scopes_allocation`) splits those.
//!
//! The font is a Git LFS object in the tree; where the checkout holds only
//! its pointer, name a real file in `PIXELFLOW_PROBE_FONT`.

use std::time::Instant;

use pixelflow_codegen::jit_cache;
use pixelflow_graphics::fonts::Font;
use pixelflow_ir::{Kernel, LatticeShape, Uniform};
use pixelflow_pipeline::alloc_probe::{self, CountingAlloc};

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// core-term's font asset (`terminal_app.rs`: `FONT_FILENAME`).
const FONT_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../pixelflow-graphics/assets/NotoSansMono-Regular.ttf"
);
/// Overrides [`FONT_PATH`].
const FONT_VAR: &str = "PIXELFLOW_PROBE_FONT";
/// What a Git LFS pointer file begins with.
const LFS_POINTER: &[u8] = b"version https://git-lfs";
/// The printable ASCII range `GlyphAtlas::warm` bakes.
const WARM_RANGE: std::ops::RangeInclusive<char> = ' '..='~';
/// The glyph tile at core-term's 16 pt cell height and density 1.0
/// (`glyph_compile_report`'s `tile_for(1.0)`).
const ATLAS_TILE: u32 = 16;
/// `fonts::PIXEL_CENTER`: texel `(i, j)` holds the coverage at
/// `(i + 0.5, j + 0.5)`. That constant is crate-private; the atlas's bake is
/// what this restates.
const PIXEL_CENTER: f32 = 0.5;
const BYTES_PER_MIB: f64 = (1u64 << 20) as f64;

/// The font's bytes, or why there are none.
fn font_bytes() -> Vec<u8> {
    let path = std::env::var(FONT_VAR).unwrap_or_else(|_| FONT_PATH.to_owned());
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert!(
        !data.starts_with(LFS_POINTER),
        "{path} is a Git LFS pointer, not a font: fetch the asset, or set {FONT_VAR} to a real file"
    );
    data
}

/// The first `count` inked glyphs as units of one program: the glyph's own
/// kernel at texel centres, held by reference.
fn units(font: &Font, tile: u32, count: usize) -> Vec<Kernel> {
    let shifted = |kernel: Kernel| {
        kernel.at(
            &Kernel::x().add(&Kernel::constant(PIXEL_CENTER)),
            &Kernel::y().add(&Kernel::constant(PIXEL_CENTER)),
        )
    };
    let units: Vec<Kernel> = WARM_RANGE
        .filter_map(|ch| font.glyph_kernel_scaled(ch, tile as f32))
        // A space draws nothing, and production bakes it as a blank tile.
        .filter(|glyph| !glyph.support.is_empty())
        .take(count)
        .map(|glyph| shifted(glyph.kernel()).by_ref())
        .collect();
    assert_eq!(
        units.len(),
        count,
        "the font has {} inked glyphs in the printable range, not {count}",
        units.len()
    );
    units
}

/// `units` under a balanced tree of `if id < k`: the font program's id tree,
/// `first` being the id of `units[0]`.
fn id_tree(id: &Kernel, units: &[Kernel], first: usize) -> Kernel {
    if units.len() == 1 {
        return units[0].clone();
    }
    let half = units.len() / 2;
    id.lt(&Kernel::constant((first + half) as f32)).select(
        &id_tree(id, &units[..half], first),
        &id_tree(id, &units[half..], first + half),
    )
}

/// The process's peak resident set so far, in bytes, where the OS says.
fn peak_resident_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / BYTES_PER_MIB
}

/// What one stage of the compile cost.
struct Stage {
    ms: f64,
    requested: usize,
    peak: usize,
}

/// Run `work` between a counter reset and a read: its wall time, the bytes it
/// requested and the peak net growth of the live heap over its start.
fn staged<R>(work: impl FnOnce() -> R) -> (R, Stage) {
    alloc_probe::reset();
    let started = Instant::now();
    let result = work();
    let ms = started.elapsed().as_secs_f64() * 1e3;
    let stage = Stage {
        ms,
        requested: alloc_probe::allocated_bytes(),
        peak: alloc_probe::peak_bytes(),
    };
    (result, stage)
}

impl Stage {
    /// `ms`, requested MiB, peak MiB: the three columns a stage owns.
    fn columns(&self) -> String {
        format!(
            "{:.1}\t{:.1}\t{:.1}",
            self.ms,
            mib(self.requested),
            mib(self.peak)
        )
    }
}

fn main() {
    const USAGE: &str = "usage: glyph_program_report <glyph count> [tile]";
    let mut args = std::env::args().skip(1);
    let count: usize = args
        .next()
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("{USAGE}"));
    let tile: u32 = args.next().map_or(ATLAS_TILE, |t| {
        t.parse().unwrap_or_else(|_| panic!("{USAGE}"))
    });
    let data = font_bytes();
    let font = Font::parse(&data).expect("parse the font");
    let program = id_tree(&Uniform::new(0.0).kernel(), &units(&font, tile, count), 0);
    let shape = LatticeShape::new([tile, tile]);
    let (arena, root) = program.parts();
    let arena_nodes = arena.len();

    println!(
        "n\ttile\tarena_nodes\tbytes\tfnv\tguards\tarms\tentries\t\
         optimized\temit_scopes\temit_hoisted\temit_frame_bytes\t\
         optimize_ms\toptimize_alloc_mib\toptimize_peak_mib\t\
         compile_ms\tcompile_alloc_mib\tcompile_peak_mib\tvm_hwm_mib\t\
         emit_ms\temit_alloc_mib\temit_peak_mib"
    );

    // Stage 1: the optimizer, cold. Its answer is memoized and stays resident,
    // so what follows starts from it and its peaks are growth over it. A
    // program the optimizer declines still compiles, unoptimized.
    let (optimized, optimize) =
        staged(|| pixelflow_search::runtime::optimize_runtime_arena(arena, root, shape));

    // Stage 2: the production entry, which finds the optimizer's answer in its
    // memo and so costs the canonical key, the link and the emit. These are
    // the row's bytes.
    let (linked, compile) =
        staged(|| jit_cache::compile(&program, shape).expect("the glyph program compiles"));
    let hwm = peak_resident_bytes()
        .map_or_else(|| "n/a".to_owned(), |b| format!("{:.1}", mib(b as usize)));

    // Stage 3: the emitter alone over the optimized arena, for the facts only
    // `CompileResult` carries (how many scopes the nest has, how much it
    // parked, the frame). Not the production bytes: this skips the link's
    // renumbering of the program's buffers and uniform.
    let (emitted, emit) = staged(|| {
        let (arena, root) = optimized
            .as_deref()
            .map_or((arena, root), |(arena, root)| (arena, *root));
        pixelflow_codegen::emit::compile(arena, root, shape).expect("the glyph program emits")
    });

    let bytes = linked.kernel.code_bytes();
    let branches = linked.kernel.branches();
    println!(
        "{count}\t{tile}\t{arena_nodes}\t{}\t{:016x}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{hwm}\t{}",
        bytes.len(),
        pixelflow_codegen::fnv1a64(bytes),
        branches.guards,
        branches.arms_branched,
        branches.arm_entries,
        optimized.is_some(),
        emitted.traffic.scopes.len(),
        emitted.hoisted_values,
        emitted.spill_bytes,
        optimize.columns(),
        compile.columns(),
        emit.columns(),
    );
}
