//! Production saturation telemetry, stage 1 of 2
//! (docs/results/2026-09-01-production-saturation-telemetry.md): write the
//! glyph arenas core-term actually bakes, so the measurer in
//! `pixelflow-search/src/runtime.rs` (`production_telemetry`) can replay
//! `optimize_runtime_arena`'s calls on them and keep the `SaturationResult`
//! production discards.
//!
//! What core-term bakes (`core-term/src/terminal_app.rs`):
//! - font: `FONT_FILENAME = "NotoSansMono-Regular.ttf"` (`:54`), the asset in
//!   this crate;
//! - `GlyphAtlas::new(cell_height = 16, density, ATLAS_CAPACITY = 128)` at
//!   density 1.0 on startup (`:204`) and again at the display density after
//!   `WindowCreated` (`:243`) — 2.0 on a Retina Mac;
//! - `atlas.warm(&font, ' '..='~')` (`:205,244`), which bakes
//!   `font.glyph_kernel_scaled(ch, tile_px)` through `Lattice::bake`
//!   (`pixelflow-graphics/src/fonts/atlas.rs:168-184`), and `Lattice::bake`
//!   hands `kernel.parts()` to `jit_cache::compile` unchanged
//!   (`pixelflow-core/src/lattice/mod.rs:402`).
//!
//! Also cross-checks the atlas arithmetic the cell-grid dumper in
//! `pixelflow-core/src/lattice/cell_grid.rs` restates, so the two dumpers
//! cannot drift apart silently.

use pixelflow_graphics::fonts::{Font, GlyphAtlas};
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};

const FONT_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/NotoSansMono-Regular.ttf"
);
const CELL_HEIGHT_PT: f32 = 16.0;
const ATLAS_CAPACITY: usize = 128;

#[test]
#[ignore = "telemetry dumper: PIXELFLOW_TELEMETRY_DIR=<dir> cargo test -p pixelflow-graphics --release --test production_glyph_arena_dump -- --ignored"]
fn dump_production_glyph_arenas() {
    let dir = std::path::PathBuf::from(
        std::env::var("PIXELFLOW_TELEMETRY_DIR").expect("PIXELFLOW_TELEMETRY_DIR must be set"),
    );
    std::fs::create_dir_all(&dir).expect("create dump dir");
    let data = std::fs::read(FONT_PATH).unwrap_or_else(|e| panic!("read {FONT_PATH}: {e}"));
    let font = Font::parse(&data).expect("parse production font");

    let mut missing: Vec<(f32, char)> = Vec::new();
    let mut dumped = 0usize;
    for density in [1.0f32, 2.0] {
        let atlas = GlyphAtlas::new(CELL_HEIGHT_PT, density, ATLAS_CAPACITY);
        // The cell-grid dumper's restatement of atlas.rs:90-98 (PAD = 1,
        // 12 slots per row, 11 rows for capacity 128).
        let tile_px = (CELL_HEIGHT_PT * density).round().max(1.0) as usize;
        assert_eq!(
            atlas.tile_px(),
            tile_px,
            "tile_px arithmetic drifted from GlyphAtlas::new"
        );
        assert_eq!(
            atlas.width(),
            12 * (tile_px + 2),
            "atlas width arithmetic drifted"
        );
        assert_eq!(
            atlas.height(),
            11 * (tile_px + 2),
            "atlas height arithmetic drifted"
        );

        for ch in ' '..='~' {
            let Some(kernel) = font.glyph_kernel_scaled(ch, atlas.tile_px() as f32) else {
                missing.push((density, ch));
                continue;
            };
            let (arena, root) = kernel.parts();
            let name = format!("glyph{tile_px}:U+{:04X}", ch as u32);
            let path = dir.join(format!("glyph{tile_px}_U{:04X}.arena", ch as u32));
            dump_arena(arena, root, &name, &path);
            dumped += 1;
        }
    }
    println!("dumped {dumped} glyph arenas to {}", dir.display());
    if !missing.is_empty() {
        // Production skips these too (atlas.rs:180-183: slot None, blank tile),
        // so they are not kernels — but say so out loud rather than dropping
        // them from the count silently.
        println!("font has no glyph for {} (density, char) pairs; production bakes nothing for them: {missing:?}", missing.len());
    }
    assert!(dumped > 0, "dumped nothing");
}

/// Text dump of the subgraph reachable from `root`: nodes in ascending
/// original id order (children precede parents), ids remapped dense,
/// constants as bit patterns, buffer identities as dense ordinals. The loader
/// in `pixelflow-search/src/runtime.rs` is the inverse. Duplicated verbatim
/// from `pixelflow-core/src/lattice/cell_grid.rs`'s test module rather than
/// shared, because the only crate both dumpers can see is `pixelflow-ir`,
/// which must not grow a test-only serializer.
fn dump_arena(arena: &ExprArena, root: ExprId, name: &str, path: &std::path::Path) {
    use std::fmt::Write as _;
    let len = arena.nodes_raw().len();
    let mut reachable = vec![false; len];
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut reachable[id.0 as usize], true) {
            continue;
        }
        stack.extend(arena.children(id));
    }
    let mut out = String::new();
    writeln!(out, "# pixelflow arena dump v1").expect("fmt");
    writeln!(out, "name {name}").expect("fmt");
    let mut idents: Vec<pixelflow_ir::arena::BufferIdentity> = Vec::new();
    for decl in arena.buffers() {
        let ord = match idents.iter().position(|i| *i == decl.id) {
            Some(p) => p,
            None => {
                idents.push(decl.id);
                idents.len() - 1
            }
        };
        writeln!(out, "buf {ord} {} {}", decl.width, decl.height).expect("fmt");
    }
    let mut dense: Vec<u32> = vec![u32::MAX; len];
    let mut next = 0u32;
    let d = |dense: &[u32], id: ExprId| -> u32 {
        let v = dense[id.0 as usize];
        assert_ne!(v, u32::MAX, "child dumped before parent");
        v
    };
    for idx in 0..len {
        if !reachable[idx] {
            continue;
        }
        let id = ExprId(idx as u32);
        match arena.node(id) {
            ExprNode::Var(i) => writeln!(out, "V {i}"),
            ExprNode::Const(v) => writeln!(out, "C {}", v.to_bits()),
            ExprNode::Buffer(b) => writeln!(out, "B {}", b.0),
            ExprNode::Unary(k, a) => writeln!(out, "U {k:?} {}", d(&dense, *a)),
            ExprNode::Binary(k, a, b) => writeln!(out, "Bi {k:?} {} {}", d(&dense, *a), d(&dense, *b)),
            ExprNode::Ternary(k, a, b, c) => {
                writeln!(out, "T {k:?} {} {} {}", d(&dense, *a), d(&dense, *b), d(&dense, *c))
            }
            other @ (ExprNode::Param(_) | ExprNode::Nary(..)) => {
                panic!("{name}: production arena contains {other:?}, which optimize_runtime_arena bails on")
            }
        }
        .expect("fmt");
        dense[idx] = next;
        next += 1;
    }
    writeln!(out, "root {}", d(&dense, root)).expect("fmt");
    std::fs::write(path, out).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}
