//! # CellGrid: a JIT-compiled tile-grid sampler
//!
//! A grid of uniformly sized cells, each referencing a tile in a coverage
//! atlas and carrying two RGBA colors; every pixel blends the cell's colors
//! by the atlas coverage under the tile's bilinear filter. This is the
//! tilemap primitive: sprite grids, tile-based scenes, and text grids are
//! all instances (a terminal is a text grid whose atlas holds glyphs — that
//! mapping lives with the caller; this module only ever sees floats).
//!
//! The whole scene is ONE compiled program per color channel — cell-index
//! arithmetic, per-cell data gathers, a 4-tap atlas read, and the color
//! blend — over two bound buffers (the per-cell data and the atlas). The
//! per-frame update rewrites the cell buffer; a geometry change (grid
//! dimensions, cell size, atlas extents) recompiles the program — a few
//! milliseconds through the JIT, dominated by the e-graph optimizer that
//! the compile path always runs, and paid only on resize. This replaces the only
//! alternative the combinator layer offered — a per-frame tree of boxed
//! `Select`s sized by the runtime grid — with a program whose size is
//! independent of the grid's.
//!
//! ## Cell data layout
//!
//! Row-major, [`CELL_STRIDE`] `f32`s per cell:
//!
//! ```text
//! [u0, v0, fg_r, fg_g, fg_b, fg_a, bg_r, bg_g, bg_b, bg_a]
//! ```
//!
//! `(u0, v0)` is the cell's tile origin in atlas texels (the caller points
//! it at the first *content* texel — any padding inset is the caller's).
//! Colors are linear [0, 1] floats.
//!
//! ## Coordinate convention
//!
//! Queries are in point space, like [`CachedGlyph`-style samplers]: cell
//! `(c, r)` spans `[c·cell_w, (c+1)·cell_w) × [r·cell_h, (r+1)·cell_h)`
//! points; within a cell, the atlas is addressed at
//! `(u0 + lx·density − ½, v0 + ly·density − ½)` texels, where `(lx, ly)`
//! is the point-space offset into the cell — the same texel-center
//! embedding the glyph cache uses, so a density-matched sample grid reads
//! tiles losslessly. Queries outside the grid return the default
//! background color. The bilinear taps reach one texel beyond the sampled
//! region on each side, so tiles need a one-texel apron in the atlas
//! (zero coverage for "nothing outside the tile").

use alloc::sync::Arc;
use alloc::vec::Vec;

use super::{BilinearSampler, DiscreteManifold};
use crate::Field;
use pixelflow_codegen::JitManifold;
use pixelflow_ir::arena::BufferIdentity;
use pixelflow_ir::{ExprArena, Kernel};

/// `f32`s per cell in the cell-data buffer.
pub const CELL_STRIDE: usize = 10;

/// The compiled-against shape of a [`CellGridProgram`]: grid dimensions,
/// cell extents, atlas extents in texels, and the atlas sample density.
/// Changing any field means recompiling.
///
/// ## Coordinate units
///
/// Cell extents and `density` are in the CALLER'S sampling space — the
/// coordinates the compiled kernels will be evaluated at. The program
/// applies no unit conversion of its own, so whoever drives the bake
/// decides what a coordinate means and must express every field in that
/// one space. In particular, `pixelflow-runtime` renders cell-grid scenes
/// on the frame buffer's DEVICE-PIXEL grid without any DPI contramap: a
/// scene destined for the runtime bakes its display scale in here
/// (extents × scale, density in texels per device pixel), as core-term
/// does.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CellGridGeometry {
    /// Grid width in cells.
    pub cols: u32,
    /// Grid height in cells.
    pub rows: u32,
    /// Cell width, in sampling-space coordinate units (see the type docs).
    pub cell_w: f32,
    /// Cell height, in sampling-space coordinate units.
    pub cell_h: f32,
    /// Atlas texels per sampling-space coordinate unit.
    pub density: f32,
    /// Atlas width in texels.
    pub atlas_width: u32,
    /// Atlas height in texels.
    pub atlas_height: u32,
    /// Tile content width in texels. Samples past it clamp into the
    /// zero apron between slots instead of reaching a neighbor's content
    /// (a cell wider than its tile shows background there, exactly as the
    /// per-glyph sampler's out-of-extent zero did).
    pub tile_w: u32,
    /// Tile content height in texels (see `tile_w`).
    pub tile_h: u32,
    /// Width in pixels of the frame this program renders. Every
    /// [`PlaneRegion`] baked through it lies within `frame_w × frame_h`;
    /// pixels past the grid's own edge (a frame is rarely a whole number of
    /// cells) show the default background. Part of the geometry because the
    /// compiled kernels are specialized to the frame's lattice: a resize is
    /// a recompile.
    pub frame_w: u32,
    /// Height in pixels of the frame this program renders (see `frame_w`).
    pub frame_h: u32,
}

impl CellGridGeometry {
    /// Cell-data buffer length this geometry implies.
    ///
    /// # Panics
    ///
    /// Panics if the product overflows `usize`. A wrapped length would be
    /// SMALL, so `frame` would accept a correspondingly small buffer while
    /// the compiled kernel still declared the true row width — and the
    /// gathers would read billions of elements past the end through an
    /// entirely safe API.
    #[must_use]
    pub fn cells_len(&self) -> usize {
        (self.cols as usize)
            .checked_mul(self.rows as usize)
            .and_then(|cells| cells.checked_mul(CELL_STRIDE))
            .expect("CellGridGeometry::cells_len overflows usize")
    }

    /// Atlas buffer length this geometry implies.
    ///
    /// # Panics
    ///
    /// Panics if the product overflows `usize` — see [`Self::cells_len`].
    #[must_use]
    pub fn atlas_len(&self) -> usize {
        (self.atlas_width as usize)
            .checked_mul(self.atlas_height as usize)
            .expect("CellGridGeometry::atlas_len overflows usize")
    }

    /// Width of the cell-data buffer as the kernel declares it, in `f32`s.
    ///
    /// # Panics
    ///
    /// Panics if it overflows `u32`. `channel_kernel` computed this as a
    /// wrapping `u32` product, so a wrapped value silently declared a buffer
    /// of a completely different shape than the one bound to it.
    #[must_use]
    pub(crate) fn cells_row_width(&self) -> u32 {
        self.cols
            .checked_mul(CELL_STRIDE as u32)
            .expect("CellGridGeometry: cell-buffer row width overflows u32")
    }
}

/// One channel's program: everything from pixel coordinate to blended channel
/// value, composed from the two samplers rather than assembled node by node.
///
/// Keyed on extents, not data. A sampler's fragment carries only its buffer
/// declaration, so the whole program composes from geometry alone and the
/// per-frame buffers bind later.
fn channel_kernel(
    geom: &CellGridGeometry,
    bufs: GridBuffers,
    channel: usize,
    default_bg: f32,
) -> Kernel {
    let k = Kernel::constant;
    let (x, y, z, w) = (Kernel::x(), Kernel::y(), Kernel::z(), Kernel::w());

    // Which cell, and where inside it.
    let col = x.mul(&k(1.0 / geom.cell_w)).floor();
    let row = y.mul(&k(1.0 / geom.cell_h)).floor();
    let lx = x.sub(&col.mul(&k(geom.cell_w)));
    let ly = y.sub(&row.mul(&k(geom.cell_h)));

    // Per-cell data: gathers clamp to the buffer edge, so out-of-grid queries
    // read the border cell — harmless, since the final select replaces them
    // with the default background anyway.
    let cells = DiscreteManifold::kernel_for(bufs.cells, geom.cells_row_width(), geom.rows);
    let cx = col.mul(&k(CELL_STRIDE as f32));
    let field = |offset: usize| {
        let idx = if offset == 0 {
            cx.clone()
        } else {
            cx.add(&k(offset as f32))
        };
        cells.at(&idx, &row, &z, &w)
    };
    let u0 = field(0);
    let v0 = field(1);
    let fg = field(2 + channel);
    let bg = field(6 + channel);

    // Atlas coordinates: point offset into the cell, scaled to texels, shifted
    // to texel centres. Clamping to half a texel past the tile content keeps
    // the taps on this slot's apron and (at most) the neighbor's — both zero —
    // so a cell larger than its tile fades to background rather than showing
    // fragments of the adjacent tile.
    let au = u0.add(&lx.mul(&k(geom.density)).min(&k(geom.tile_w as f32 + 0.5)));
    let av = v0.add(&ly.mul(&k(geom.density)).min(&k(geom.tile_h as f32 + 0.5)));
    let atlas = BilinearSampler::kernel_for(bufs.atlas, geom.atlas_width, geom.atlas_height);
    let cov = atlas.at(&au.sub(&k(0.5)), &av.sub(&k(0.5)), &z, &w);

    // blended = bg + cov·(fg − bg), and the default background off-grid.
    let blended = bg.add(&cov.mul(&fg.sub(&bg)));
    let in_grid = x
        .ge(&k(0.0))
        .and(&x.lt(&k(geom.cols as f32 * geom.cell_w)))
        .and(&y.ge(&k(0.0)))
        .and(&y.lt(&k(geom.rows as f32 * geom.cell_h)));
    in_grid.select(&blended, &k(default_bg))
}

/// The whole screen as ONE program: the four channel kernels, each packed to
/// a byte exactly as `Pixel::from_rgba` does — `(x·255).clamp(0, 255)` then
/// truncate toward zero — shifted to its byte lane and OR-folded into a
/// `u32` pixel bit pattern at the root.
///
/// `shifts[c]` is the bit position of channel `c` in `(r, g, b, a)` order.
/// Both pixel orders are little-endian byte arrays wrapping a `u32`, so byte
/// `i` sits at bit `8·i`: `Rgba8` is `[r, g, b, a]` → `[0, 8, 16, 24]`, and
/// `Bgra8` is `[b, g, r, a]` → `[16, 8, 0, 24]`.
///
/// Truncation (not rounding) is deliberate twice over: it is bit-exact with
/// the scalar pack's `as u8`, and `cvttps2dq`/`fcvtzs` both truncate toward
/// zero — no per-target tie divergence to inherit.
fn packed_kernel(
    geom: &CellGridGeometry,
    bufs: GridBuffers,
    default_bg: [f32; 4],
    shifts: [u32; 4],
) -> Kernel {
    let k = Kernel::constant;
    let byte = |c: usize| {
        channel_kernel(geom, bufs, c, default_bg[c])
            .mul(&k(255.0))
            .clamp(&k(0.0), &k(255.0))
            .trunc_to_int()
            .shl(shifts[c])
    };
    // Fold in the bit domain, then take the single named exit: the packed
    // pattern IS the kernel's output.
    byte(0).or(&byte(1)).or(&byte(2)).or(&byte(3)).into_kernel()
}

/// The two blocks of memory a cell-grid program reads, identified so that
/// composing many reads of one buffer still merges to one slot, and so binding
/// can say which slot is which without inferring it from extents.
#[derive(Clone, Copy, Debug)]
struct GridBuffers {
    cells: BufferIdentity,
    atlas: BufferIdentity,
}

impl GridBuffers {
    fn mint() -> Self {
        Self {
            cells: BufferIdentity::mint(),
            atlas: BufferIdentity::mint(),
        }
    }
}

/// Which buffer a slot of the composed arena reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SlotReads {
    Cells,
    Atlas,
}

/// Attribute each declared slot to one of the program's two buffers.
///
/// Matching is on [`BufferIdentity`], so it is a fact rather than an inference
/// — two buffers of equal extent would be indistinguishable by shape, and
/// guessing there would bind one pointer for both.
///
/// # Panics
///
/// Panics on a slot belonging to neither, which would mean a fragment declared
/// memory this program cannot bind.
fn slot_roles(arena: &ExprArena, bufs: GridBuffers) -> Vec<SlotReads> {
    arena
        .buffers()
        .iter()
        .map(|decl| {
            if decl.id == bufs.cells {
                SlotReads::Cells
            } else if decl.id == bufs.atlas {
                SlotReads::Atlas
            } else {
                panic!("cell-grid kernel declared an unbindable buffer {decl:?}")
            }
        })
        .collect()
}

/// The geometry preconditions every cell-grid compile shares — the
/// four-channel and packed paths refuse the same shapes for the same reasons.
///
/// # Panics
///
/// Panics on a degenerate geometry (zero cells, non-positive cell extent or
/// density, empty atlas), when this build's `Field` width does not match the
/// JIT's emitted width, or on buffers too large to index exactly in f32.
fn assert_compilable(geom: &CellGridGeometry) {
    assert!(
        geom.cols > 0 && geom.rows > 0 && geom.atlas_width > 0 && geom.atlas_height > 0,
        "cell-grid compile: degenerate geometry {geom:?}"
    );
    assert!(
        geom.cell_w > 0.0 && geom.cell_h > 0.0 && geom.density > 0.0,
        "cell-grid compile: non-positive cell extent or density {geom:?}"
    );
    assert_eq!(
        core::mem::size_of::<Field>(),
        pixelflow_codegen::JIT_VECTOR_BYTES,
        "cell-grid compile: Field width does not match the JIT's emitted width"
    );
    // Both products must be computable before anything downstream trusts
    // them: cells_len/cells_row_width panic on overflow rather than wrapping,
    // and a wrapped length paired with an unwrapped row width is how a safe
    // API ends up gathering past the end of the bound buffer.
    // (Both accessors panic on overflow; calling them here is the check.)
    assert!(
        geom.cells_len() > 0 && geom.cells_row_width() > 0,
        "cell-grid compile: empty cell buffer for {geom:?}"
    );

    // Gather computes its row-major linear index in f32, which is exact
    // only below 2^24 — beyond that adjacent texels alias. Refuse
    // loudly instead of rendering corrupted gathers.
    const EXACT_F32_INDEX: usize = 1 << 24;
    assert!(
        geom.atlas_len() <= EXACT_F32_INDEX,
        "cell-grid compile: atlas of {} texels exceeds the exactly \
         f32-indexable range (2^24); gathers would alias adjacent texels",
        geom.atlas_len()
    );
    assert!(
        geom.cells_len() <= EXACT_F32_INDEX,
        "cell-grid compile: cell buffer of {} floats exceeds the \
         exactly f32-indexable range (2^24)",
        geom.cells_len()
    );
}

/// The four compiled channel kernels for one grid/atlas geometry.
///
/// Compile once per geometry ([`CellGridProgram::compile`]); stamp per-frame
/// data into it with [`CellGridProgram::frame`]. A resize is: build the new
/// geometry, compile, done — the program's size and compile time are
/// independent of the grid's dimensions.
pub struct CellGridProgram {
    geom: CellGridGeometry,
    jits: [Arc<JitManifold>; 4],
    /// What each buffer slot of the compiled arenas reads — see [`SlotReads`].
    /// Shared so a [`CellGridFrame`] stays cheap to clone.
    slots: Arc<[SlotReads]>,
}

/// Upper bound on buffer slots in a composed channel kernel, so binding can
/// build its context array on the stack: no per-frame allocation.
///
/// Splicing merges by [`BufferIdentity`], so this is one slot per *distinct*
/// buffer however many times each is read — the program reads two. A third
/// would fail the check in `compile` rather than silently overflow.
const MAX_SLOTS: usize = 2;

impl CellGridProgram {
    /// JIT-compile the four channel programs for `geom`, with `default_bg`
    /// (linear RGBA) painted outside the grid.
    ///
    /// # Panics
    ///
    /// Panics on a degenerate geometry (zero cells, non-positive cell extent
    /// or density, empty atlas), when this build's `Field` width does not
    /// match the JIT's emitted width, or if compilation fails.
    #[must_use]
    pub fn compile(geom: CellGridGeometry, default_bg: [f32; 4]) -> Self {
        assert_compilable(&geom);
        let bufs = GridBuffers::mint();
        let kernels: [Kernel; 4] =
            core::array::from_fn(|c| channel_kernel(&geom, bufs, c, default_bg[c]));
        let slots = slot_roles(kernels[0].parts().0, bufs);
        for (c, kernel) in kernels.iter().enumerate().skip(1) {
            assert_eq!(
                slot_roles(kernel.parts().0, bufs),
                slots,
                "cell-grid channel {c} disagrees with channel 0 on slot layout"
            );
        }
        assert!(
            slots.len() <= MAX_SLOTS,
            "cell-grid kernel needs {} buffer slots, over the {MAX_SLOTS} a \
             frame can bind without allocating",
            slots.len()
        );

        let jits = core::array::from_fn(|c| {
            let (arena, root) = kernels[c].parts();
            // Collapse-mode (internal 2D loop): one call bakes a whole run
            // of rows for a channel. Bound-memory arenas are uncacheable
            // (the code bakes buffer slot metadata); the cache recognizes
            // that and compiles fresh.
            pixelflow_codegen::jit_cache::compile(
                arena,
                root,
                pixelflow_ir::LatticeShape::new([geom.frame_w, geom.frame_h, 1, 1]),
            )
            .expect("cell-grid channel failed to compile")
        });
        Self {
            geom,
            jits,
            slots: slots.into(),
        }
    }

    /// The geometry this program was compiled for.
    #[must_use]
    pub fn geometry(&self) -> &CellGridGeometry {
        &self.geom
    }

    /// Bind one frame's data. `cells` is [`CELL_STRIDE`] `f32`s per cell,
    /// row-major; `atlas` is the coverage atlas, row-major texels. Both are
    /// `Arc`s so a frame in flight keeps its data alive while the caller
    /// prepares the next one.
    ///
    /// # Panics
    ///
    /// Panics if either buffer's length does not match the geometry.
    #[must_use]
    pub fn frame(&self, cells: Arc<Vec<f32>>, atlas: Arc<Vec<f32>>) -> CellGridFrame {
        assert_eq!(
            cells.len(),
            self.geom.cells_len(),
            "cell buffer length does not match geometry {:?}",
            self.geom
        );
        assert_eq!(
            atlas.len(),
            self.geom.atlas_len(),
            "atlas buffer length does not match geometry {:?}",
            self.geom
        );
        CellGridFrame {
            jits: self.jits.clone(),
            slots: self.slots.clone(),
            frame: [self.geom.frame_w, self.geom.frame_h],
            cells,
            atlas,
        }
    }
}

/// One frame of a cell grid: the compiled channels plus the data they read.
/// Cheap to clone (four `Arc`s and two `Arc`s).
#[derive(Clone)]
pub struct CellGridFrame {
    jits: [Arc<JitManifold>; 4],
    slots: Arc<[SlotReads]>,
    /// The pixel frame the kernels were compiled for (`geom.frame_w/frame_h`).
    frame: [u32; 2],
    cells: Arc<Vec<f32>>,
    atlas: Arc<Vec<f32>>,
}

/// A horizontal band of pixel rows to bake: `width` pixels across, rows
/// `y0 .. y0 + rows`.
#[derive(Clone, Copy, Debug)]
pub struct PlaneRegion {
    /// Pixels per row.
    pub width: usize,
    /// First pixel row.
    pub y0: usize,
    /// Number of rows.
    pub rows: usize,
}

/// The kernels were compiled for `frame` (`geom.frame_w × geom.frame_h`); a
/// region outside it will run the collapse loop past the lattice they were
/// specialized to.
///
/// `debug_assert`, matching [`JitManifold::call_collapse`]'s own check of the
/// same promise: today's emitted code takes its loop bounds from the tile at
/// run time, so a wider region is merely a stale cache key, not wrong pixels.
/// It becomes load-bearing when the emitted code specializes on the extents,
/// and is promoted with that change rather than ahead of it — a release panic
/// for a promise nothing yet relies on is a new way for a terminal to die.
fn assert_region_in_frame(what: &str, frame: [u32; 2], region: PlaneRegion) {
    let (fw, fh) = (frame[0] as usize, frame[1] as usize);
    debug_assert!(
        region.width <= fw && region.y0.saturating_add(region.rows) <= fh,
        "{what}: region {}×{} at row {} lies outside the {fw}×{fh} frame this program was compiled for",
        region.width,
        region.rows,
        region.y0
    );
}

impl CellGridFrame {
    /// The padded row stride (in `f32`s) that [`CellGridFrame::bake_channel_rows`]
    /// writes: `width` rounded up to whole SIMD batches. Planes are laid out
    /// at this stride; the padding lanes hold whatever the kernel computed
    /// past the right edge (out-of-grid pixels — the default background) and
    /// are simply not read back.
    #[must_use]
    pub fn padded_width(width: usize) -> usize {
        let lanes = pixelflow_codegen::JIT_VECTOR_BYTES / core::mem::size_of::<f32>();
        width.div_ceil(lanes).max(1) * lanes
    }

    /// Bake one color channel (0 = R, 1 = G, 2 = B, 3 = A) over the pixel
    /// rows `y0 .. y0 + rows` at pixel-center coordinates, into a plane of
    /// [`CellGridFrame::padded_width`]`(width)`-stride rows.
    ///
    /// ONE call into the channel's 2D collapse kernel per invocation: the
    /// loop over batches and rows lives inside the JIT, with the two-level
    /// LICM prologues (per-call and per-row) active. This is the production
    /// render path — a frame is four of these per stripe, then a pack.
    ///
    /// # Panics
    ///
    /// Panics if `channel >= 4`, the region's width is zero, or `out` is
    /// shorter than `rows * padded_width(width)`.
    pub fn bake_channel_rows(&self, channel: usize, region: PlaneRegion, out: &mut [f32]) {
        let PlaneRegion { width, y0, rows } = region;
        assert!(width > 0, "bake_channel_rows: zero width");
        assert_region_in_frame("bake_channel_rows", self.frame, region);
        let stride = Self::padded_width(width);
        // Checked for the same reason as `bake_packed_rows`: a wrapped
        // product would admit an undersized `out` while the unsafe collapse
        // below still wrote the real row count past its end.
        let needed = rows
            .checked_mul(stride)
            .expect("bake_channel_rows: rows * stride overflows usize");
        assert!(
            out.len() >= needed,
            "bake_channel_rows: plane of {} f32s cannot hold {rows} rows at stride {stride}",
            out.len()
        );
        if rows == 0 {
            return;
        }
        let groups = stride / (pixelflow_codegen::JIT_VECTOR_BYTES / core::mem::size_of::<f32>());
        // One base pointer per declared slot, in slot order. Stack-allocated
        // against the MAX_SLOTS bound `compile` checked, so binding a frame
        // allocates nothing; trailing entries stay null and are never read
        // because the kernel only addresses slots it declared.
        let mut ctx = [core::ptr::null::<f32>(); MAX_SLOTS];
        for (dst, slot) in ctx.iter_mut().zip(self.slots.iter()) {
            *dst = match slot {
                SlotReads::Cells => self.cells.as_ptr(),
                SlotReads::Atlas => self.atlas.as_ptr(),
            };
        }
        // Pixel centers: the rasterizer convention (x + ½, y + ½).
        let x0 = Field::sequential(0.5);
        let y = Field::from(y0 as f32 + 0.5);
        let zero = Field::from(0.0);
        // SAFETY: `compile` checked size_of::<Field>() == JIT_VECTOR_BYTES;
        // every slot the kernel declared was attributed to the cell buffer or
        // the atlas by `slot_roles` (which refuses any other declaration), and
        // `CellGridProgram::frame` asserted both lengths against this same
        // geometry; `ctx` binds their live base pointers for the duration of
        // the call; and `out` holds
        // `rows` full rows of `groups` batches (asserted above) with no
        // scalar tail (row_skip = 0 — the stride IS whole batches).
        unsafe {
            self.jits[channel].call_collapse(
                ctx.as_ptr(),
                pixelflow_codegen::TileSlice::contiguous(out.as_mut_ptr(), groups, rows),
                pixelflow_codegen::Point4::new(x0, y, zero, zero),
            );
        }
    }
}

/// The packed-pixel sibling of [`CellGridProgram`]: ONE compiled program
/// whose root is a packed `u32` pixel, so the collapse loop stores finished
/// pixels directly — no per-channel planes, no pack pass. A sibling type
/// rather than a fifth entry in `jits` because the two paths have different
/// output types, and the type should refuse baking a packed program into an
/// `f32` plane (or vice versa) rather than a runtime check catching it.
pub struct CellGridPackedProgram {
    geom: CellGridGeometry,
    /// The byte lanes this kernel packed for. A frame stores raw words, so
    /// whoever consumes them must confirm their format agrees.
    shifts: [u32; 4],
    jit: Arc<JitManifold>,
    /// What each buffer slot of the compiled arena reads — see [`SlotReads`].
    slots: Arc<[SlotReads]>,
}

impl CellGridPackedProgram {
    /// JIT-compile the packed program for `geom`, with `default_bg` (linear
    /// RGBA) painted outside the grid and `shifts[c]` giving the bit position
    /// of channel `c` in `(r, g, b, a)` order — `[0, 8, 16, 24]` for `Rgba8`,
    /// `[16, 8, 0, 24]` for `Bgra8` (see [`packed_kernel`]'s derivation).
    ///
    /// # Panics
    ///
    /// Panics on the same geometry preconditions as
    /// [`CellGridProgram::compile`], on shifts that are not a permutation of
    /// the four byte lanes (channels would overlap), if the composed kernel's
    /// buffer reads do not merge to exactly one cell slot and one atlas slot,
    /// or if compilation fails.
    #[must_use]
    pub fn compile(geom: CellGridGeometry, default_bg: [f32; 4], shifts: [u32; 4]) -> Self {
        assert_compilable(&geom);
        {
            let mut lanes = shifts;
            lanes.sort_unstable();
            assert_eq!(
                lanes,
                [0, 8, 16, 24],
                "CellGridPackedProgram::compile: shifts {shifts:?} are not a \
                 permutation of the byte lanes; channels would overlap"
            );
        }
        let bufs = GridBuffers::mint();
        let kernel = packed_kernel(&geom, bufs, default_bg, shifts);
        let slots = slot_roles(kernel.parts().0, bufs);
        // Identity-merge across the four channels' splices is load-bearing:
        // all cell reads and atlas taps must land in the same two slots a
        // frame can bind. A third slot means splice stopped merging — refuse.
        assert_eq!(
            slots.iter().filter(|r| **r == SlotReads::Cells).count(),
            1,
            "packed cell-grid kernel did not merge its cell reads to one slot"
        );
        assert_eq!(
            slots.iter().filter(|r| **r == SlotReads::Atlas).count(),
            1,
            "packed cell-grid kernel did not merge its atlas reads to one slot"
        );
        let (arena, root) = kernel.parts();
        let jit = pixelflow_codegen::jit_cache::compile(
            arena,
            root,
            pixelflow_ir::LatticeShape::new([geom.frame_w, geom.frame_h, 1, 1]),
        )
        .expect("packed cell-grid kernel failed to compile");
        Self {
            geom,
            shifts,
            jit,
            slots: slots.into(),
        }
    }

    /// The byte lanes this program's kernel packs into.
    #[must_use]
    pub fn shifts(&self) -> [u32; 4] {
        self.shifts
    }

    /// The packed kernel's emitted bytes (research/profiling harness).
    #[cfg(test)]
    fn jits_code_bytes(&self) -> &[u8] {
        self.jit.code_bytes()
    }

    /// The geometry this program was compiled for.
    #[must_use]
    pub fn geometry(&self) -> &CellGridGeometry {
        &self.geom
    }

    /// Bind one frame's data — the same buffers, layout, and length checks as
    /// [`CellGridProgram::frame`].
    ///
    /// # Panics
    ///
    /// Panics if either buffer's length does not match the geometry.
    #[must_use]
    pub fn frame(&self, cells: Arc<Vec<f32>>, atlas: Arc<Vec<f32>>) -> CellGridPackedFrame {
        assert_eq!(
            cells.len(),
            self.geom.cells_len(),
            "cell buffer length does not match geometry {:?}",
            self.geom
        );
        assert_eq!(
            atlas.len(),
            self.geom.atlas_len(),
            "atlas buffer length does not match geometry {:?}",
            self.geom
        );
        CellGridPackedFrame {
            shifts: self.shifts,
            jit: self.jit.clone(),
            slots: self.slots.clone(),
            frame: [self.geom.frame_w, self.geom.frame_h],
            cells,
            atlas,
        }
    }
}

/// One frame of a packed cell grid: the compiled program plus the data it
/// reads. Cheap to clone.
#[derive(Clone)]
pub struct CellGridPackedFrame {
    /// The byte lanes the kernel packed for; a consumer storing these words
    /// must confirm its pixel format agrees.
    shifts: [u32; 4],
    jit: Arc<JitManifold>,
    slots: Arc<[SlotReads]>,
    /// The pixel frame the kernel was compiled for (`geom.frame_w/frame_h`).
    frame: [u32; 2],
    cells: Arc<Vec<f32>>,
    atlas: Arc<Vec<f32>>,
}

impl CellGridPackedFrame {
    /// The byte lanes these words are packed into.
    #[must_use]
    pub fn shifts(&self) -> [u32; 4] {
        self.shifts
    }

    /// Bake packed pixels over the pixel rows `y0 .. y0 + rows` at
    /// pixel-center coordinates, into a plane of
    /// [`CellGridFrame::padded_width`]`(width)`-stride rows — the same
    /// region, stride, and whole-batch contract as
    /// [`CellGridFrame::bake_channel_rows`], with the pack already done.
    ///
    /// The kernel's root is int-domain: each lane already holds a packed
    /// pixel's bit pattern. The collapse store is a raw vector store —
    /// type-blind bit movement — so writing through the ABI's `*mut f32`
    /// into a `u32` plane is exact: no float operation touches the value
    /// between the OR-fold and memory.
    ///
    /// # Panics
    ///
    /// Panics if the region's width is zero or `out` is shorter than
    /// `rows * padded_width(width)`.
    pub fn bake_packed_rows(&self, region: PlaneRegion, out: &mut [u32]) {
        let PlaneRegion { width, y0, rows } = region;
        assert!(width > 0, "bake_packed_rows: zero width");
        assert_region_in_frame("bake_packed_rows", self.frame, region);
        let stride = CellGridFrame::padded_width(width);
        // Checked: `rows * stride` wraps in release for a caller-supplied
        // region large enough, and a wrapped product would let an undersized
        // `out` pass this guard while the collapse call below still received
        // the real (enormous) row count and wrote past the slice. The
        // documented panic must fire before any unsafe call, not after.
        let needed = rows
            .checked_mul(stride)
            .expect("bake_packed_rows: rows * stride overflows usize");
        assert!(
            out.len() >= needed,
            "bake_packed_rows: plane of {} u32s cannot hold {rows} rows at stride {stride}",
            out.len()
        );
        if rows == 0 {
            return;
        }
        let groups = stride / (pixelflow_codegen::JIT_VECTOR_BYTES / core::mem::size_of::<f32>());
        // One base pointer per declared slot, in slot order — identical to
        // `bake_channel_rows`, against the same MAX_SLOTS bound (`compile`
        // proved exactly two slots).
        let mut ctx = [core::ptr::null::<f32>(); MAX_SLOTS];
        for (dst, slot) in ctx.iter_mut().zip(self.slots.iter()) {
            *dst = match slot {
                SlotReads::Cells => self.cells.as_ptr(),
                SlotReads::Atlas => self.atlas.as_ptr(),
            };
        }
        // Pixel centers: the rasterizer convention (x + ½, y + ½).
        let x0 = Field::sequential(0.5);
        let y = Field::from(y0 as f32 + 0.5);
        let zero = Field::from(0.0);
        // SAFETY: as in `bake_channel_rows` — `assert_compilable` checked
        // size_of::<Field>() == JIT_VECTOR_BYTES; every declared slot was
        // attributed to the cell buffer or the atlas by `slot_roles`, and
        // `frame` asserted both lengths against this same geometry; `ctx` binds
        // their live base pointers for the duration of the call; `out` holds
        // `rows` full rows of `groups` batches (asserted above) with no
        // scalar tail (row_skip = 0). The pointer cast is sound because
        // `u32` and `f32` share size and alignment and the store is a raw
        // vector store of the root's bit pattern.
        unsafe {
            self.jit.call_collapse(
                ctx.as_ptr(),
                pixelflow_codegen::TileSlice::contiguous(
                    out.as_mut_ptr().cast::<f32>(),
                    groups,
                    rows,
                ),
                pixelflow_codegen::Point4::new(x0, y, zero, zero),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// A 2×1 grid over a 2-tile atlas: tile A solid coverage 1, tile B a
    /// half-coverage field. Colors chosen so every channel distinguishes
    /// fg, bg, and the default background.
    fn tiny_scene() -> (CellGridProgram, Arc<Vec<f32>>, Arc<Vec<f32>>) {
        // Atlas: two 4×4-content tiles with 1-texel aprons, laid out side by
        // side → 12×6 texels.
        let (aw, ah, slot) = (12usize, 6usize, 6usize);
        let mut atlas = vec![0.0f32; aw * ah];
        for r in 0..4 {
            for c in 0..4 {
                atlas[(1 + r) * aw + 1 + c] = 1.0; // tile 0: solid
                atlas[(1 + r) * aw + slot + 1 + c] = 0.5; // tile 1: half
            }
        }
        let geom = CellGridGeometry {
            cols: 2,
            rows: 1,
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: aw as u32,
            atlas_height: ah as u32,
            tile_w: 4,
            tile_h: 4,
            frame_w: 12,
            frame_h: 6,
        };
        let program = CellGridProgram::compile(geom, [0.9, 0.8, 0.7, 0.6]);
        // Cell 0 → tile 0, red-on-black; cell 1 → tile 1, white-on-blue.
        let cells = vec![
            1.0, 1.0, /* uv */ 1.0, 0.0, 0.0, 1.0, /* fg */ 0.0, 0.0, 0.0,
            1.0, /* bg */
            7.0, 1.0, /* uv */ 1.0, 1.0, 1.0, 1.0, /* fg */ 0.0, 0.0, 1.0,
            1.0, /* bg */
        ];
        (program, Arc::new(cells), Arc::new(atlas))
    }

    /// Research harness: dump the packed kernel's machine code and hot-loop
    /// it in one process, so a sampling profiler's addresses correlate to
    /// the dumped bytes exactly. `PIXELFLOW_CODE_DUMP=<path>` receives the
    /// raw bytes; the base pointer prints to stdout.
    #[test]
    #[ignore = "manual profiling harness"]
    fn profile_dump_packed_kernel() {
        let geom = CellGridGeometry {
            cols: 213,
            rows: 66,
            cell_w: 12.0,
            cell_h: 24.0,
            density: 1.0,
            atlas_width: 64,
            atlas_height: 32,
            tile_w: 12,
            tile_h: 24,
            frame_w: 2560,
            frame_h: 1584,
        };
        let program = CellGridPackedProgram::compile(geom, [0.1, 0.1, 0.1, 1.0], [0, 8, 16, 24]);
        let code = program.jits_code_bytes();
        std::println!(
            "CODE_BASE=0x{:x} CODE_LEN={}",
            code.as_ptr() as usize,
            code.len()
        );
        if let Ok(path) = std::env::var("PIXELFLOW_CODE_DUMP") {
            std::fs::write(&path, code).expect("dump write failed");
        }
        let mut atlas = alloc::vec![0.0f32; geom.atlas_len()];
        for (i, t) in atlas.iter_mut().enumerate() {
            *t = ((i * 7) % 11) as f32 / 10.0;
        }
        let mut cells = alloc::vec![0.0f32; geom.cells_len()];
        for (i, c) in cells.iter_mut().enumerate() {
            *c = ((i * 13) % 17) as f32 / 16.0;
        }
        let frame = program.frame(Arc::new(cells), Arc::new(atlas));
        let (w, h) = (2560usize, 1584usize);
        let stride = CellGridFrame::padded_width(w);
        let mut out = alloc::vec![0u32; stride * h];
        for _ in 0..150 {
            frame.bake_packed_rows(
                PlaneRegion {
                    width: w,
                    y0: 0,
                    rows: h,
                },
                &mut out,
            );
            core::hint::black_box(&out);
        }
    }

    /// Splicing merges buffer declarations by identity, so the nine reads in
    /// a channel program (four cell fields — `bg` twice — plus the atlas)
    /// occupy exactly two slots: one per distinct buffer.
    ///
    /// Nodes are a different story, and this pins the gap. 146 against the 77
    /// the hand-written arena built, because every *use* of a Kernel value
    /// re-splices its whole fragment: `row` is used by four cell reads, `cx`
    /// by four, `bg` by two. Identity fixed the slots; it cannot fix this.
    /// Node sharing is an inlining decision, and it belongs to the e-graph —
    /// which cannot see this arena at all while `Gather` is unrepresentable
    /// there. This number falling is the sign that landed.
    #[test]
    fn reads_of_one_buffer_share_a_slot_but_not_nodes() {
        let geom = CellGridGeometry {
            cols: 2,
            rows: 1,
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: 12,
            atlas_height: 6,
            tile_w: 4,
            tile_h: 4,
            frame_w: 8,
            frame_h: 4,
        };
        let bufs = GridBuffers::mint();
        let kernel = channel_kernel(&geom, bufs, 0, 0.0);
        let (arena, root) = kernel.parts();

        let roles = slot_roles(arena, bufs);
        assert_eq!(
            roles.iter().filter(|r| **r == SlotReads::Cells).count(),
            1,
            "five cell reads, one buffer, one slot"
        );
        assert_eq!(
            roles.iter().filter(|r| **r == SlotReads::Atlas).count(),
            1,
            "one slot for the atlas"
        );
        assert_eq!(
            reachable_nodes(arena, root),
            146,
            "composed node count (hand-built was 77)"
        );
    }

    fn reachable_nodes(arena: &ExprArena, root: pixelflow_ir::ExprId) -> usize {
        let mut seen = vec![false; arena.nodes_raw().len()];
        let mut stack = vec![root];
        let mut count = 0;
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            count += 1;
            stack.extend(arena.children(id));
        }
        count
    }

    /// Row-major index into an 8-wide plane.
    fn px(row: usize, col: usize) -> usize {
        row * 8 + col
    }

    /// Bake one channel over pixel centers and unpad to a dense w×h plane.
    fn plane(frame: &CellGridFrame, channel: usize, w: usize, h: usize) -> alloc::vec::Vec<f32> {
        let stride = CellGridFrame::padded_width(w);
        let mut padded = vec![0.0f32; h * stride];
        frame.bake_channel_rows(
            channel,
            PlaneRegion {
                width: w,
                y0: 0,
                rows: h,
            },
            &mut padded,
        );
        let mut dense = alloc::vec::Vec::with_capacity(w * h);
        for row in 0..h {
            dense.extend_from_slice(&padded[row * stride..row * stride + w]);
        }
        dense
    }

    #[test]
    fn interior_texels_blend_fg_over_bg_by_coverage() {
        let (program, cells, atlas) = tiny_scene();
        let frame = program.frame(cells, atlas);
        let r = plane(&frame, 0, 8, 4);
        let b = plane(&frame, 2, 8, 4);
        // Cell 0 interior: coverage 1 → pure fg (red).
        assert!((r[px(1, 1)] - 1.0).abs() < 1e-5, "cell0 R = {}", r[9]);
        assert!(b[px(1, 1)].abs() < 1e-5, "cell0 B = {}", b[9]);
        // Cell 1 interior: coverage 0.5 → half fg, half bg.
        // R: 0.5·1 + 0.5·0 = 0.5;  B: 0.5·1 + 0.5·1 = 1.0.
        assert!((r[px(1, 5)] - 0.5).abs() < 1e-5, "cell1 R = {}", r[13]);
        assert!((b[px(1, 5)] - 1.0).abs() < 1e-5, "cell1 B = {}", b[13]);
    }

    #[test]
    fn outside_the_grid_is_the_default_background() {
        let (program, cells, atlas) = tiny_scene();
        let frame = program.frame(cells, atlas);
        // Sample a lattice wider and taller than the 8×4 grid.
        let g = plane(&frame, 1, 12, 6);
        // (10.5, 0.5) is right of the grid; (0.5, 5.5) below it.
        assert!((g[10] - 0.8).abs() < 1e-5, "right of grid G = {}", g[10]);
        assert!(
            (g[5 * 12] - 0.8).abs() < 1e-5,
            "below grid G = {}",
            g[5 * 12]
        );
        // Alpha channel outside: 0.6.
        let a = plane(&frame, 3, 12, 6);
        assert!((a[11] - 0.6).abs() < 1e-5, "outside A = {}", a[11]);
    }

    #[test]
    fn aligned_pixel_centers_read_tiles_losslessly() {
        let (program, cells, atlas) = tiny_scene();
        let frame = program.frame(cells, atlas);
        let r = plane(&frame, 0, 8, 4);
        // Density-matched pixel centers land exactly on texel centers, so
        // the bilinear read degenerates to lookup: the LAST pixel column of
        // cell 0 still reads pure coverage 1 (its own last content texel),
        // and the FIRST pixel column of cell 1 reads its tile's 0.5.
        assert!((r[px(1, 3)] - 1.0).abs() < 1e-5, "cell0 edge R = {}", r[11]);
        assert!((r[px(1, 4)] - 0.5).abs() < 1e-5, "cell1 edge R = {}", r[12]);
    }

    #[test]
    fn misaligned_samples_fade_through_the_apron_not_into_neighbors() {
        // Density 0.8 over a 5-point cell: pixel centers land BETWEEN texel
        // centers, so the bilinear filter engages. The last pixel of the
        // cell (x = 4.5 → tile-space 3.6) taps the last content texel
        // (weight 0.9) and the zero APRON (weight 0.1) — never the next
        // slot's content, which sits two texels further.
        let (aw, ah, slot) = (12usize, 6usize, 6usize);
        let mut atlas = vec![0.0f32; aw * ah];
        for r in 0..4 {
            for c in 0..4 {
                atlas[(1 + r) * aw + 1 + c] = 1.0; // tile 0: solid
                atlas[(1 + r) * aw + slot + 1 + c] = 1.0; // tile 1: ALSO solid
            }
        }
        let geom = CellGridGeometry {
            cols: 1,
            rows: 1,
            cell_w: 5.0,
            cell_h: 5.0,
            density: 0.8,
            atlas_width: aw as u32,
            atlas_height: ah as u32,
            tile_w: 4,
            tile_h: 4,
            frame_w: 5,
            frame_h: 5,
        };
        let program = CellGridProgram::compile(geom, [0.0; 4]);
        let cells = vec![
            1.0, 1.0, /* fg */ 1.0, 1.0, 1.0, 1.0, /* bg */ 0.0, 0.0, 0.0, 1.0,
        ];
        let frame = program.frame(Arc::new(cells), Arc::new(atlas));
        let r = plane(&frame, 0, 5, 5);
        // Row 2 (tile-space y = 2.0·... = interior): x = 4.5 → 0.9 content,
        // 0.1 apron. If the tap bled into tile 1's solid content instead,
        // this would read 1.0.
        assert!(
            (r[2 * 5 + 4] - 0.9).abs() < 1e-5,
            "fractional edge sample R = {}",
            r[2 * 5 + 4]
        );
    }

    #[test]
    fn cells_wider_than_their_tile_fade_to_background() {
        // 1x1 grid, 8-point-wide cell over a 4-texel tile: the right half
        // of the cell has no tile content. The clamp must land those taps
        // in the zero apron (background), never in a neighboring slot.
        let (aw, ah, slot) = (12usize, 6usize, 6usize);
        let mut atlas = vec![0.0f32; aw * ah];
        for r in 0..4 {
            for c in 0..4 {
                atlas[(1 + r) * aw + 1 + c] = 1.0; // tile 0: solid
                atlas[(1 + r) * aw + slot + 1 + c] = 1.0; // tile 1: ALSO solid
            }
        }
        let geom = CellGridGeometry {
            cols: 1,
            rows: 1,
            cell_w: 8.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: aw as u32,
            atlas_height: ah as u32,
            tile_w: 4,
            tile_h: 4,
            frame_w: 8,
            frame_h: 4,
        };
        let program = CellGridProgram::compile(geom, [0.0; 4]);
        let cells = vec![
            1.0, 1.0, /* fg */ 1.0, 1.0, 1.0, 1.0, /* bg */ 0.0, 0.0, 0.0, 1.0,
        ];
        let frame = program.frame(Arc::new(cells), Arc::new(atlas));
        let r = plane(&frame, 0, 8, 4);
        // Left half: tile content, fg. Right half: past the tile — if the
        // clamp failed, taps would reach tile 1 (also solid) and read 1.0.
        assert!(
            (r[px(1, 1)] - 1.0).abs() < 1e-5,
            "tile content R = {}",
            r[px(1, 1)]
        );
        for col in 5..8 {
            assert!(
                r[px(1, col)].abs() < 1e-5,
                "cell past its tile must be background at col {col}, got {}",
                r[px(1, col)]
            );
        }
    }

    #[test]
    #[should_panic(expected = "exactly")]
    fn oversized_atlas_is_refused() {
        let geom = CellGridGeometry {
            cols: 1,
            rows: 1,
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: 1 << 13,
            atlas_height: 1 << 12, // 2^25 texels: past exact f32 indexing
            tile_w: 4,
            tile_h: 4,
            frame_w: 4,
            frame_h: 4,
        };
        let _refused = CellGridProgram::compile(geom, [0.0; 4]);
    }

    #[test]
    #[should_panic(expected = "cell buffer length")]
    fn mismatched_cell_buffer_is_refused() {
        let (program, _, atlas) = tiny_scene();
        // Binding satisfies unused_must_use; the call panics first.
        let _refused = program.frame(Arc::new(alloc::vec![0.0; 3]), atlas);
    }

    #[test]
    fn cells_len_and_atlas_len_match_declared_geometry() {
        let geom = CellGridGeometry {
            cols: 3,
            rows: 2,
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: 12,
            atlas_height: 6,
            tile_w: 4,
            tile_h: 4,
            frame_w: 12,
            frame_h: 8,
        };
        assert_eq!(geom.cells_len(), 3 * 2 * CELL_STRIDE);
        assert_eq!(geom.atlas_len(), 12 * 6);
    }

    #[test]
    fn padded_width_rounds_up_to_whole_simd_batches() {
        let lanes = pixelflow_codegen::JIT_VECTOR_BYTES / core::mem::size_of::<f32>();
        assert_eq!(CellGridFrame::padded_width(1), lanes);
        assert_eq!(CellGridFrame::padded_width(lanes), lanes);
        assert_eq!(CellGridFrame::padded_width(lanes + 1), lanes * 2);
    }

    #[test]
    fn green_channel_reads_its_own_bg_field_not_a_neighboring_one() {
        // In `tiny_scene`, cell 1 has fg_g = 1.0, bg_g = 0.0, coverage 0.5,
        // so the correct blend is 0.5. Every neighboring field at cell 1
        // (fg_a, bg_r, bg_b) is 1.0 or 0.0 in a way that a wrong bg-field
        // offset would still coincidentally read 0.5-ish through cell 0's
        // R/B channels the other tests already check — G is the one channel
        // whose neighbors in the field layout (fg_a at offset 5, bg_r at
        // offset 6) both read 1.0/0.0 in a way that distinguishes a bad
        // offset from the correct one.
        let (program, cells, atlas) = tiny_scene();
        let frame = program.frame(cells, atlas);
        let g = plane(&frame, 1, 8, 4);
        assert!(
            (g[px(1, 5)] - 0.5).abs() < 1e-5,
            "cell1 G = {}",
            g[px(1, 5)]
        );
    }

    #[test]
    fn cells_taller_than_their_tile_fade_to_background() {
        // Mirrors `cells_wider_than_their_tile_fade_to_background` in the
        // row direction: a 1x1 grid with a cell taller than its tile. The
        // bottom half has no tile content; the vertical apron clamp must
        // land those taps in the zero apron, never a neighboring slot.
        let (aw, ah, slot) = (12usize, 6usize, 6usize);
        let mut atlas = vec![0.0f32; aw * ah];
        for r in 0..4 {
            for c in 0..4 {
                atlas[(1 + r) * aw + 1 + c] = 1.0; // tile 0: solid
                atlas[(1 + r) * aw + slot + 1 + c] = 1.0; // tile 1: ALSO solid
            }
        }
        let geom = CellGridGeometry {
            cols: 1,
            rows: 1,
            cell_w: 4.0,
            cell_h: 8.0,
            density: 1.0,
            atlas_width: aw as u32,
            atlas_height: ah as u32,
            tile_w: 4,
            tile_h: 4,
            frame_w: 4,
            frame_h: 8,
        };
        let program = CellGridProgram::compile(geom, [0.0; 4]);
        let cells = vec![
            1.0, 1.0, /* fg */ 1.0, 1.0, 1.0, 1.0, /* bg */ 0.0, 0.0, 0.0, 1.0,
        ];
        let frame = program.frame(Arc::new(cells), Arc::new(atlas));
        let r = plane(&frame, 0, 4, 8);
        // Top half: tile content, fg. Bottom half: past the tile — if the
        // clamp failed, taps would reach tile 1 (also solid) and read 1.0.
        assert!((r[5] - 1.0).abs() < 1e-5, "tile content R = {}", r[5]);
        for row in 5..8 {
            assert!(
                r[row * 4 + 1].abs() < 1e-5,
                "cell past its tile must be background at row {row}, got {}",
                r[row * 4 + 1]
            );
        }
    }

    #[test]
    fn grid_extent_is_cell_count_times_cell_size_not_their_sum() {
        // A 3x3 grid of 2x2 cells: grid_w = grid_h = 6.0. If the boundary
        // check ever computed cols + cell_w (5.0) instead of cols * cell_w,
        // the last row/column of real cells (spanning [4, 6)) would be
        // wrongly treated as outside the grid.
        let (aw, ah) = (6usize, 6usize);
        let mut atlas = vec![0.0f32; aw * ah];
        for r in 0..4 {
            for c in 0..4 {
                atlas[(1 + r) * aw + 1 + c] = 1.0; // solid tile
            }
        }
        let geom = CellGridGeometry {
            cols: 3,
            rows: 3,
            cell_w: 2.0,
            cell_h: 2.0,
            density: 1.0,
            atlas_width: aw as u32,
            atlas_height: ah as u32,
            tile_w: 4,
            tile_h: 4,
            frame_w: 8,
            frame_h: 8,
        };
        let program = CellGridProgram::compile(geom, [0.4; 4]);
        let block: [f32; CELL_STRIDE] = [
            1.0, 1.0, /* fg */ 1.0, 0.0, 0.0, 1.0, /* bg */ 0.0, 0.0, 0.0, 1.0,
        ];
        let mut cells = vec![];
        for _ in 0..9 {
            cells.extend_from_slice(&block);
        }
        let frame = program.frame(Arc::new(cells), Arc::new(atlas));
        let r = plane(&frame, 0, 8, 8);
        // (5.5, 5.5) is interior to the last (col 2, row 2) cell — must
        // read pure fg (1.0), not the default background (0.4) a
        // cols+cell_w-sized boundary would wrongly clip it to.
        assert!(
            (r[5 * 8 + 5] - 1.0).abs() < 1e-5,
            "last cell R = {}",
            r[5 * 8 + 5]
        );
    }

    #[test]
    fn baking_a_row_offset_region_samples_the_correct_absolute_row() {
        // A 1x2 grid (one column, two rows) with visibly different content
        // per row. Baking a region whose `PlaneRegion::y0` starts mid-grid
        // must read the ABSOLUTE row that `y0` implies. A constant error in
        // the pixel-center offset shifts every sampled row by the same
        // amount, so a self-consistency check (comparing this bake against
        // a full bake sliced at the same relative offset) can't see it —
        // this asserts a concrete expected value instead.
        let (aw, ah, slot) = (12usize, 6usize, 6usize);
        let mut atlas = vec![0.0f32; aw * ah];
        for r in 0..4 {
            for c in 0..4 {
                atlas[(1 + r) * aw + 1 + c] = 1.0; // tile 0: solid
                atlas[(1 + r) * aw + slot + 1 + c] = 0.5; // tile 1: half
            }
        }
        let geom = CellGridGeometry {
            cols: 1,
            rows: 2,
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: aw as u32,
            atlas_height: ah as u32,
            tile_w: 4,
            tile_h: 4,
            frame_w: 4,
            frame_h: 8,
        };
        let program = CellGridProgram::compile(geom, [0.0; 4]);
        let cells = vec![
            1.0, 1.0, /* row0 fg */ 1.0, 0.0, 0.0, 1.0, /* bg */ 0.0, 0.0, 0.0, 1.0, 7.0,
            1.0, /* row1 fg */ 0.0, 0.0, 1.0, 1.0, /* bg */ 1.0, 1.0, 1.0, 1.0,
        ];
        let frame = program.frame(Arc::new(cells), Arc::new(atlas));

        let stride = CellGridFrame::padded_width(4);
        let mut padded = vec![0.0f32; 4 * stride];
        frame.bake_channel_rows(
            0,
            PlaneRegion {
                width: 4,
                y0: 4,
                rows: 4,
            },
            &mut padded,
        );
        // y0 = 4 → the first baked row samples absolute y = 4.5, inside row
        // 1's cell (coverage 0.5, R: bg 1.0 -> fg 0.0 blends to 0.5). An
        // off-by-one-pixel-center bug would instead sample y = 3.5, still
        // inside row 0's cell (solid fg, R = 1.0).
        assert!(
            (padded[1] - 0.5).abs() < 1e-5,
            "row-offset bake read the wrong absolute row: R = {}",
            padded[1]
        );
    }

    /// `Rgba8`'s byte lanes: little-endian `[r, g, b, a]`.
    const RGBA_SHIFTS: [u32; 4] = [0, 8, 16, 24];
    /// `Bgra8`'s byte lanes: little-endian `[b, g, r, a]`.
    const BGRA_SHIFTS: [u32; 4] = [16, 8, 0, 24];

    /// The scalar pack the kernel must be bit-exact with:
    /// `Pixel::from_rgba`'s per-channel `(x·255).clamp(0, 255) as u8`,
    /// composed with the same shifts.
    fn pack_expected(rgba: [f32; 4], shifts: [u32; 4]) -> u32 {
        rgba.iter()
            .zip(shifts)
            .map(|(&x, s)| u32::from((x * 255.0).clamp(0.0, 255.0) as u8) << s)
            .fold(0, |acc, lane| acc | lane)
    }

    /// Bake packed pixels over pixel centers and unpad to a dense w×h plane.
    fn packed_plane(frame: &CellGridPackedFrame, w: usize, h: usize) -> alloc::vec::Vec<u32> {
        let stride = CellGridFrame::padded_width(w);
        let mut padded = vec![0u32; h * stride];
        frame.bake_packed_rows(
            PlaneRegion {
                width: w,
                y0: 0,
                rows: h,
            },
            &mut padded,
        );
        let mut dense = alloc::vec::Vec::with_capacity(w * h);
        for row in 0..h {
            dense.extend_from_slice(&padded[row * stride..row * stride + w]);
        }
        dense
    }

    #[test]
    fn packed_bake_is_bit_exact_with_channel_bakes_under_both_byte_orders() {
        // 12×6 over tiny_scene's 8×4 grid: the right and bottom margins are
        // off-grid, so default_bg flows through the pack too. THE invariant:
        // for every pixel, the packed bake equals the four channel bakes
        // composed with the scalar pack — exact u32 equality, no epsilon.
        let (program, cells, atlas) = tiny_scene();
        let frame = program.frame(cells.clone(), atlas.clone());
        let (w, h) = (12, 6);
        let planes: [alloc::vec::Vec<f32>; 4] = core::array::from_fn(|c| plane(&frame, c, w, h));
        for shifts in [RGBA_SHIFTS, BGRA_SHIFTS] {
            let packed =
                CellGridPackedProgram::compile(*program.geometry(), [0.9, 0.8, 0.7, 0.6], shifts);
            let got = packed_plane(&packed.frame(cells.clone(), atlas.clone()), w, h);
            for i in 0..w * h {
                let expected = pack_expected(
                    [planes[0][i], planes[1][i], planes[2][i], planes[3][i]],
                    shifts,
                );
                assert_eq!(
                    got[i], expected,
                    "pixel {i} under shifts {shifts:?}: {:#010x} != {:#010x}",
                    got[i], expected
                );
            }
        }
    }

    #[test]
    fn packed_bake_parity_holds_on_an_asymmetric_grid() {
        // The 1×2 grid from the row-offset test (cols ≠ rows, per-row
        // colors), sampled 6×10 so the right and bottom margins exercise a
        // non-gray default background through every channel's shift.
        let (aw, ah, slot) = (12usize, 6usize, 6usize);
        let mut atlas = vec![0.0f32; aw * ah];
        for r in 0..4 {
            for c in 0..4 {
                atlas[(1 + r) * aw + 1 + c] = 1.0; // tile 0: solid
                atlas[(1 + r) * aw + slot + 1 + c] = 0.5; // tile 1: half
            }
        }
        let geom = CellGridGeometry {
            cols: 1,
            rows: 2,
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: aw as u32,
            atlas_height: ah as u32,
            tile_w: 4,
            tile_h: 4,
            frame_w: 6,
            frame_h: 10,
        };
        let default_bg = [0.25, 0.5, 0.75, 1.0];
        let cells = Arc::new(vec![
            1.0, 1.0, /* row0 fg */ 1.0, 0.0, 0.0, 1.0, /* bg */ 0.0, 0.0, 0.0, 1.0, 7.0,
            1.0, /* row1 fg */ 0.0, 0.0, 1.0, 1.0, /* bg */ 1.0, 1.0, 1.0, 1.0,
        ]);
        let atlas = Arc::new(atlas);
        let (w, h) = (6, 10);
        let program = CellGridProgram::compile(geom, default_bg);
        let frame = program.frame(cells.clone(), atlas.clone());
        let planes: [alloc::vec::Vec<f32>; 4] = core::array::from_fn(|c| plane(&frame, c, w, h));
        let packed = CellGridPackedProgram::compile(geom, default_bg, RGBA_SHIFTS);
        let got = packed_plane(&packed.frame(cells, atlas), w, h);
        for i in 0..w * h {
            let expected = pack_expected(
                [planes[0][i], planes[1][i], planes[2][i], planes[3][i]],
                RGBA_SHIFTS,
            );
            assert_eq!(
                got[i], expected,
                "pixel {i}: {:#010x} != {:#010x}",
                got[i], expected
            );
        }
    }

    /// The packed kernel splices four channel fragments (each with its own
    /// cell reads and atlas taps), and identity-merge must still land them
    /// all in exactly two slots — one per distinct buffer.
    #[test]
    fn packed_kernel_reads_land_in_exactly_two_slots() {
        let geom = CellGridGeometry {
            cols: 2,
            rows: 1,
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: 12,
            atlas_height: 6,
            tile_w: 4,
            tile_h: 4,
            frame_w: 8,
            frame_h: 4,
        };
        let bufs = GridBuffers::mint();
        let kernel = packed_kernel(&geom, bufs, [0.0; 4], RGBA_SHIFTS);
        let roles = slot_roles(kernel.parts().0, bufs);
        assert_eq!(
            roles.iter().filter(|r| **r == SlotReads::Cells).count(),
            1,
            "all four channels' cell reads, one buffer, one slot"
        );
        assert_eq!(
            roles.iter().filter(|r| **r == SlotReads::Atlas).count(),
            1,
            "all four channels' atlas taps, one slot"
        );
        assert_eq!(roles.len(), 2, "exactly {{Cells, Atlas}}, nothing else");
    }

    /// Pins the packed program's size, in the spirit of
    /// `reads_of_one_buffer_share_a_slot_but_not_nodes`: 623 = 4 × 146 (each
    /// `or` re-splices a whole channel fragment) + 4 × 9 (each channel's pack
    /// chain: const 255, mul, const 0, max, const 255, min, trunc, const
    /// shift, shl) + 3 ors. Node sharing is the e-graph's call once it can
    /// see `Gather`; this number falling is the sign that landed.
    #[test]
    fn packed_kernel_node_count_is_the_channel_kernels_plus_the_pack() {
        let geom = CellGridGeometry {
            cols: 2,
            rows: 1,
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: 12,
            atlas_height: 6,
            tile_w: 4,
            tile_h: 4,
            frame_w: 8,
            frame_h: 4,
        };
        let bufs = GridBuffers::mint();
        let kernel = packed_kernel(&geom, bufs, [0.0; 4], RGBA_SHIFTS);
        let (arena, root) = kernel.parts();
        assert_eq!(
            reachable_nodes(arena, root),
            623,
            "composed packed node count (4 channels of 146, plus the pack)"
        );
    }

    #[test]
    fn packed_row_offset_region_matches_the_full_bake() {
        // Mirrors `baking_a_row_offset_region_samples_the_correct_absolute_row`
        // for the packed path: a bake starting at y0 = 4 must reproduce rows
        // 4..8 of the full bake exactly (u32 equality — these are packed
        // pixels, not floats).
        let (aw, ah, slot) = (12usize, 6usize, 6usize);
        let mut atlas = vec![0.0f32; aw * ah];
        for r in 0..4 {
            for c in 0..4 {
                atlas[(1 + r) * aw + 1 + c] = 1.0; // tile 0: solid
                atlas[(1 + r) * aw + slot + 1 + c] = 0.5; // tile 1: half
            }
        }
        let geom = CellGridGeometry {
            cols: 1,
            rows: 2,
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: aw as u32,
            atlas_height: ah as u32,
            tile_w: 4,
            tile_h: 4,
            frame_w: 4,
            frame_h: 8,
        };
        let program = CellGridPackedProgram::compile(geom, [0.0; 4], RGBA_SHIFTS);
        let cells = Arc::new(vec![
            1.0, 1.0, /* row0 fg */ 1.0, 0.0, 0.0, 1.0, /* bg */ 0.0, 0.0, 0.0, 1.0, 7.0,
            1.0, /* row1 fg */ 0.0, 0.0, 1.0, 1.0, /* bg */ 1.0, 1.0, 1.0, 1.0,
        ]);
        let frame = program.frame(cells, Arc::new(atlas));

        let full = packed_plane(&frame, 4, 8);

        let stride = CellGridFrame::padded_width(4);
        let mut padded = vec![0u32; 4 * stride];
        frame.bake_packed_rows(
            PlaneRegion {
                width: 4,
                y0: 4,
                rows: 4,
            },
            &mut padded,
        );
        for row in 0..4 {
            for col in 0..4 {
                assert_eq!(
                    padded[row * stride + col],
                    full[(row + 4) * 4 + col],
                    "offset bake row {row} col {col} disagrees with the full bake"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "permutation of the byte lanes")]
    fn overlapping_pack_shifts_are_refused() {
        let geom = CellGridGeometry {
            cols: 1,
            rows: 1,
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            atlas_width: 12,
            atlas_height: 6,
            tile_w: 4,
            tile_h: 4,
            frame_w: 4,
            frame_h: 4,
        };
        // R and B both at bit 0: the OR would blend two channels' bytes.
        let _refused = CellGridPackedProgram::compile(geom, [0.0; 4], [0, 8, 0, 24]);
    }

    /// Production saturation telemetry, stage 1 of 2
    /// (docs/results/2026-09-01-production-saturation-telemetry.md): write
    /// the packed cell-grid arena core-term actually compiles, for the
    /// geometries it actually compiles it at, so the measurer in
    /// `pixelflow-search/src/runtime.rs` (`production_telemetry`) can replay
    /// `optimize_runtime_arena`'s calls on it and keep the `SaturationResult`
    /// production discards. Lives here because `packed_kernel` and
    /// `GridBuffers::mint` are private, and stays private itself.
    ///
    /// Geometry is core-term's, not a fixture's:
    /// `core-term/src/terminal_app.rs:362-372` builds `CellGridGeometry` from
    /// the snapshot's cell size (config defaults `cell_width_px: 10`,
    /// `cell_height_px: 16`, `config.rs`) times the display density, with
    /// `density: 1.0`, and the atlas extents of
    /// `GlyphAtlas::new(cell_height, density, ATLAS_CAPACITY = 128)`
    /// (`terminal_app.rs:204,243`), whose arithmetic (`atlas.rs:90-98`, PAD = 1)
    /// is restated here and cross-checked against a real `GlyphAtlas` by the
    /// glyph dumper. Shifts are `PlatformColorCube::PACKED_SHIFTS` on macOS
    /// (`RgbaColorCube`, `[0, 8, 16, 24]`); `default_bg` is the default
    /// `ColorScheme.background` (black). Density 1.0 is the startup compile,
    /// 2.0 the Retina recompile after `WindowCreated`; 120x40 is one resize.
    #[test]
    #[ignore = "telemetry dumper: PIXELFLOW_TELEMETRY_DIR=<dir> cargo test -p pixelflow-core --release -- --ignored dump_production_cell_grid_arenas"]
    fn dump_production_cell_grid_arenas() {
        let dir = std::path::PathBuf::from(
            std::env::var("PIXELFLOW_TELEMETRY_DIR").expect("PIXELFLOW_TELEMETRY_DIR must be set"),
        );
        std::fs::create_dir_all(&dir).expect("create dump dir");
        const CELL_W_PT: f32 = 10.0;
        const CELL_H_PT: f32 = 16.0;
        const ATLAS_SLOTS_PER_ROW: u32 = 12; // ceil(sqrt(128))
        const ATLAS_SLOT_ROWS: u32 = 11; // ceil(128 / 12)
        const ATLAS_PAD: u32 = 1;
        const DEFAULT_BG_BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
        for (label, cols, rows, density) in [
            ("80x24_d1", 80u32, 24u32, 1.0f32),
            ("80x24_d2", 80, 24, 2.0),
            ("120x40_d2", 120, 40, 2.0),
        ] {
            let tile_px = (CELL_H_PT * density).round().max(1.0) as u32;
            let slot_px = tile_px + 2 * ATLAS_PAD;
            let cell_w = CELL_W_PT * density;
            let cell_h = CELL_H_PT * density;
            let geom = CellGridGeometry {
                cols,
                rows,
                cell_w,
                cell_h,
                density: 1.0,
                atlas_width: ATLAS_SLOTS_PER_ROW * slot_px,
                atlas_height: ATLAS_SLOT_ROWS * slot_px,
                tile_w: tile_px,
                tile_h: tile_px,
                frame_w: (cols as f32 * cell_w).round() as u32,
                frame_h: (rows as f32 * cell_h).round() as u32,
            };
            // Same preconditions `CellGridPackedProgram::compile` enforces.
            assert_compilable(&geom);
            let kernel = packed_kernel(&geom, GridBuffers::mint(), DEFAULT_BG_BLACK, RGBA_SHIFTS);
            let (arena, root) = kernel.parts();
            let name = alloc::format!("cellgrid:{label}");
            let path = dir.join(alloc::format!("cellgrid_{label}.arena"));
            dump_arena(arena, root, &name, &path);
            std::println!(
                "{name}: {} reachable nodes -> {}",
                reachable_nodes(arena, root),
                path.display()
            );
        }
    }

    /// Text dump of the subgraph reachable from `root`: nodes in ascending
    /// original id order (children precede parents), ids remapped dense,
    /// constants as bit patterns, buffer identities as dense ordinals. The
    /// loader in `pixelflow-search/src/runtime.rs` is the inverse. Duplicated
    /// verbatim in `pixelflow-graphics/tests/production_glyph_arena_dump.rs`
    /// rather than shared, because the only crate both dumpers can see is
    /// `pixelflow-ir`, which must not grow a test-only serializer.
    fn dump_arena(
        arena: &ExprArena,
        root: pixelflow_ir::ExprId,
        name: &str,
        path: &std::path::Path,
    ) {
        use core::fmt::Write as _;
        use pixelflow_ir::arena::ExprNode;
        let len = arena.nodes_raw().len();
        let mut reachable = vec![false; len];
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut reachable[id.0 as usize], true) {
                continue;
            }
            stack.extend(arena.children(id));
        }
        let mut out = std::string::String::new();
        writeln!(out, "# pixelflow arena dump v1").expect("fmt");
        writeln!(out, "name {name}").expect("fmt");
        let mut idents: alloc::vec::Vec<pixelflow_ir::arena::BufferIdentity> =
            alloc::vec::Vec::new();
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
        let mut dense: alloc::vec::Vec<u32> = vec![u32::MAX; len];
        let mut next = 0u32;
        let d = |dense: &[u32], id: pixelflow_ir::ExprId| -> u32 {
            let v = dense[id.0 as usize];
            assert_ne!(v, u32::MAX, "child dumped before parent");
            v
        };
        for idx in 0..len {
            if !reachable[idx] {
                continue;
            }
            let id = pixelflow_ir::ExprId(idx as u32);
            match arena.node(id) {
                ExprNode::Var(i) => writeln!(out, "V {i}"),
                ExprNode::Const(v) => writeln!(out, "C {}", v.to_bits()),
                ExprNode::Buffer(b) => writeln!(out, "B {}", b.0),
                ExprNode::Unary(k, a) => writeln!(out, "U {k:?} {}", d(&dense, *a)),
                ExprNode::Binary(k, a, b) => {
                    writeln!(out, "Bi {k:?} {} {}", d(&dense, *a), d(&dense, *b))
                }
                ExprNode::Ternary(k, a, b, c) => writeln!(
                    out,
                    "T {k:?} {} {} {}",
                    d(&dense, *a),
                    d(&dense, *b),
                    d(&dense, *c)
                ),
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
}
