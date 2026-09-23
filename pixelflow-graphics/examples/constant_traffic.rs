//! Where a kernel's constants go: per-scope instruction and constant-load
//! counts for the two production kernels, through the same optimize-and-emit
//! path a bake or a frame uses, on this host's tier. Run it at two commits
//! and diff.
//!
//! The columns are `EmitTraffic`'s, per scope of the nest — the body first,
//! then every fold in nest order, the lattice's row and column folds among
//! them. `trips` is how many times one call runs the scope, so a count in a
//! row with a large `trips` is the hot loop's. `remats` is constants brought
//! into a register from outside the register file (from the pool, or before
//! the pool, rebuilt inline); `loads_k` includes a root reloaded from its
//! park at a scope's head.
//!
//! Fixtures:
//! - The glyph `8` at 32px composed with `.at(x+0.5, y+0.5)`, at its own
//!   tile, as `GlyphAtlas` bakes it.
//! - The cell grid's red channel at one stripe of an 80×24 terminal frame:
//!   what a worker runs per stripe, minus the packing of the other three
//!   channels. The packed four-channel program's size follows, since that is
//!   the kernel a frame actually runs.
//!
//! ```bash
//! cargo run --release -p pixelflow-graphics --example constant_traffic
//! ```

use pixelflow_codegen::emit;
use pixelflow_core::{CellGridShape, Kernel};
use pixelflow_graphics::fonts::Font;
use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::scene::compile_packed_for;
use pixelflow_graphics::scene3d::Rgba;
use pixelflow_ir::LatticeShape;
use pixelflow_search::runtime::optimize_runtime_arena;

const FONT_DATA: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// Rows per stripe, as `render::scene` hands them to workers.
const STRIPE_ROWS: u32 = 8;

fn report(label: &str, kernel: &Kernel, shape: LatticeShape) {
    let (arena, root) = kernel.linked_parts();
    let optimized = optimize_runtime_arena(&arena, root, shape);
    let (a, r) = optimized
        .as_deref()
        .map_or((&arena, root), |(a, r)| (a, *r));
    let result = emit::compile(a, r, shape).expect("emit");
    let t = &result.traffic;
    let [w, h] = shape.extent();
    println!(
        "{label} at {w}x{h}: {} bytes, {} spill slots, {} parked roots, {} memory ops per call",
        t.bytes(),
        result.spill_count,
        result.hoisted_values,
        t.dynamic_memory_ops()
    );
    println!("  scope    trips  bytes  instr  remats  loads_t  loads_k  stores  writes");
    for (i, (s, trips)) in t.scopes.iter().zip(&t.trips).enumerate() {
        println!(
            "  {i:>5} {trips:>8} {:>6} {:>6} {:>7} {:>8} {:>8} {:>7} {:>7}",
            s.bytes, s.instructions, s.remats, s.loads_transient, s.loads_kept, s.stores, s.writes
        );
    }
}

fn main() {
    let font = Font::parse(FONT_DATA).expect("parse font");
    let glyph = font
        .glyph_kernel_scaled('8', 32.0)
        .expect("'8' glyph at 32px");
    let half = Kernel::constant(0.5);
    let warped = glyph
        .kernel()
        .at(&Kernel::x().add(&half), &Kernel::y().add(&half));
    report("glyph 8", &warped, LatticeShape::new([32, 32]));

    let shape = CellGridShape {
        cols: 80,
        rows: 24,
        atlas_width: 512,
        atlas_height: 512,
        frame_w: 960,
        frame_h: 480,
    };
    let kernels = shape.channel_kernels();
    report(
        "cell grid, red channel",
        &kernels.channels[0],
        LatticeShape::new([shape.frame_w, STRIPE_ROWS]),
    );
    let packed = compile_packed_for::<Rgba8>(
        &Rgba::from(&kernels.channels),
        [shape.frame_w, shape.frame_h],
    );
    println!(
        "cell grid, four channels packed at {}x{}: {} bytes",
        shape.frame_w,
        shape.frame_h,
        packed.code_bytes().len()
    );
}
