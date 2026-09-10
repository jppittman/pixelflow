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
//! per-frame update rewrites the cell buffer. This replaces the only
//! alternative the combinator layer offered — a per-frame tree of boxed
//! `Select`s sized by the runtime grid — with a program whose size is
//! independent of the grid's.
//!
//! ## What recompiles, and what does not
//!
//! The two halves of a cell grid's geometry have different types, because
//! they have different lifetimes in the machine code:
//!
//! - A [`CellGridShape`] is nothing but **extents** — the cell buffer's, the
//!   atlas's, and the frame lattice's. An extent is compiled against: it is a
//!   loop bound and a gather stride, an immediate in the emitted code. A new
//!   shape is a new program.
//! - A [`CellGridMetrics`] is the **metric** — how big a cell is, how many
//!   atlas texels a point is worth, how much of a tile has content, and how
//!   the caller's point space relates to the frame's device pixels. None of
//!   that is an extent, so none of it is compiled against: it is a block of
//!   [`Uniform`]s ([`CellGridSlots`]), and changing it is a parameter write.
//!
//! So a zoom, a font-size change, a DPI change, or any other change of
//! *metric* reuses the compiled program; only a change of *shape* — the grid
//! gaining a column, the atlas growing, the window changing size — compiles
//! anything.
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
//! The lattice is the frame's **device-pixel** index. Everything a
//! [`CellGridMetrics`] measures is in the caller's **point** space, and
//! [`CellGridMetrics::scale`] — device pixels per point — is the contramap
//! between them: the kernel samples the point `(i + ½)/scale`. A display
//! scale is a coordinate transform, not a property of the domain, and it is
//! spelled as one here rather than folded into every extent by the caller.
//!
//! In point space, cell `(c, r)` spans
//! `[c·cell_w, (c+1)·cell_w) × [r·cell_h, (r+1)·cell_h)`; within a cell, the
//! atlas is addressed at `(u0 + lx·density − ½, v0 + ly·density − ½)` texels,
//! where `(lx, ly)` is the point-space offset into the cell — the same
//! texel-center embedding the glyph cache uses, so a density-matched sample
//! grid reads tiles losslessly. The bilinear taps reach one texel beyond the
//! sampled region on each side, so tiles need a one-texel apron in the atlas
//! (zero coverage for "nothing outside the tile").
//!
//! ## Outside the grid is a range, not a mask
//!
//! A frame is rarely a whole number of cells, so some of it lies outside the
//! grid. That used to be `in_grid.select(blended, default_bg)`: a mask
//! multiplied through a program every pixel of the frame ran. It is an
//! [`IndexRange`] now — [`CellGridShape::grid_range`] — and the pixels
//! outside it are filled with the default background instead of computed.
//! An extent on the domain needs no mask, no guard and no coherence
//! question: the loop does not go there.

use alloc::sync::Arc;
use alloc::vec::Vec;

use super::manifold::{BoundManifold, Manifold, PlaneRegion, UniformBlock, UnknownUniform};
use super::union::IndexRange;
use super::{BilinearSampler, DiscreteManifold};
use pixelflow_ir::decl::BufferIdentity;
use pixelflow_ir::{Kernel, Uniform};

/// `f32`s per cell in the cell-data buffer.
pub const CELL_STRIDE: usize = 10;

/// Half a sample: a pixel index's offset from its own center, which is where
/// a frame collapse samples ([`PlaneRegion::rows`]).
const SAMPLE_CENTER: f32 = 0.5;

/// **The extents a cell-grid program is compiled against, and nothing else.**
///
/// Every field is the shape of something the emitted code addresses: the cell
/// buffer's row width and height, the atlas's, and the frame lattice's. A
/// change to any of them is a different program.
///
/// What is deliberately *not* here is the grid's metric — cell size, sample
/// density, tile content size, display scale. Those are
/// [`CellGridMetrics`], they arrive through a [`UniformBlock`], and changing
/// them compiles nothing.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CellGridShape {
    /// Grid width in cells. The cell buffer's row width is
    /// `cols × CELL_STRIDE`.
    pub cols: u32,
    /// Grid height in cells, and the cell buffer's height.
    pub rows: u32,
    /// Atlas width in texels.
    pub atlas_width: u32,
    /// Atlas height in texels.
    pub atlas_height: u32,
    /// Width in device pixels of the frame this program renders. The lattice
    /// the four channel kernels are compiled at, so a window resize is a
    /// recompile — an extent is not a uniform.
    pub frame_w: u32,
    /// Height in device pixels of the frame this program renders (see
    /// `frame_w`).
    pub frame_h: u32,
}

/// **The metric of a cell grid: a block of parameters, never an extent.**
///
/// Measured in the caller's POINT space, with [`Self::scale`] relating that
/// space to the frame's device-pixel lattice. Writing a new value of this
/// type into a program's [`UniformBlock`] is the whole of a zoom, a font-size
/// change or a DPI change — see the [module docs](self).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CellGridMetrics {
    /// Cell width in points.
    pub cell_w: f32,
    /// Cell height in points.
    pub cell_h: f32,
    /// Atlas texels per point.
    pub density: f32,
    /// Tile content width in texels. Samples past it clamp into the zero
    /// apron between slots instead of reaching a neighbor's content (a cell
    /// wider than its tile shows background there).
    pub tile_w: u32,
    /// Tile content height in texels (see `tile_w`).
    pub tile_h: u32,
    /// Device pixels per point: the frame lattice's scale relative to the
    /// point space every other field is measured in. `1.0` samples points
    /// and device pixels alike; `2.0` is a Retina frame of a point-space
    /// scene.
    pub scale: f32,
}

impl CellGridMetrics {
    /// The metric preconditions a program's arguments must satisfy.
    ///
    /// **Asserted on the values that reach the block, not only on the fields
    /// that were passed.** A subnormal `cell_w` — `1e-42` — is positive and
    /// finite and passes any check on the input, and then `1/cell_w`
    /// overflows to `+inf` and every pixel of the frame reads cell 0. The
    /// slots are private precisely so no half-denoted state is reachable, and
    /// an input-only check leaves one reachable: what must be finite is what
    /// the kernel loads.
    ///
    /// # Panics
    ///
    /// Panics on a non-positive or non-finite cell extent, density or scale;
    /// on a cell extent or scale whose reciprocal is not finite; and on a
    /// tile with no content, which denotes a cell that can show nothing.
    ///
    /// **This does not close the metric space, and should not be read as
    /// doing so.** The tile extents have a lower bound here and no upper one,
    /// because the bound that matters — `tile_w <= atlas_width` — is a fact
    /// about the *shape*, which a metric-local check cannot see. A `tile_w`
    /// larger than the atlas is accepted, its `tile_w + ½` clamp never binds,
    /// and a cell wider than its tile walks into the adjacent slot's content:
    /// exactly the failure `cells_wider_than_their_tile_fade_to_background`
    /// says the clamp prevents, arriving as a plausible wrong picture rather
    /// than a crash. Refusing it belongs where the shape is in hand —
    /// `assert_compilable` or [`CellGridShape::grid_extent`]'s callers — and
    /// is not done yet.
    fn assert_writable(&self) {
        let finite_positive = |v: f32| v.is_finite() && v > 0.0;
        assert!(
            finite_positive(self.cell_w)
                && finite_positive(self.cell_h)
                && finite_positive(self.density)
                && finite_positive(self.scale),
            "cell-grid metrics: non-positive or non-finite {self:?}"
        );
        assert!(
            (1.0 / self.cell_w).is_finite()
                && (1.0 / self.cell_h).is_finite()
                && (1.0 / self.scale).is_finite(),
            "cell-grid metrics: non-positive or non-finite reciprocal of {self:?}"
        );
        assert!(
            self.tile_w > 0 && self.tile_h > 0,
            "cell-grid metrics: a tile with no content in {self:?}"
        );
    }
}

impl CellGridShape {
    /// Cell-data buffer length this shape implies.
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
            .expect("CellGridShape::cells_len overflows usize")
    }

    /// Atlas buffer length this shape implies.
    ///
    /// # Panics
    ///
    /// Panics if the product overflows `usize` — see [`Self::cells_len`].
    #[must_use]
    pub fn atlas_len(&self) -> usize {
        (self.atlas_width as usize)
            .checked_mul(self.atlas_height as usize)
            .expect("CellGridShape::atlas_len overflows usize")
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
            .expect("CellGridShape: cell-buffer row width overflows u32")
    }

    /// **The frame indices the grid covers at `metrics`** — the domain-side
    /// encoding of "outside the grid", and what used to be the `in_grid`
    /// mask.
    ///
    /// A pixel index `i` samples the point `(i + ½)/scale`, and the grid
    /// spans `[0, cols·cell_w)` points, so the range is the `i` with
    /// `i + ½ < cols·cell_w·scale`, clipped to the frame. Anchored at the
    /// origin on both axes: a pixel index is never negative, which is the
    /// other half of what the mask tested and the half a range does not need
    /// to state.
    ///
    /// The range is a function of the *metric*, not of the shape alone, so it
    /// belongs to a frame rather than to the compiled program — which is
    /// exactly why it can be a range at all: nothing about it is compiled
    /// against.
    #[must_use]
    pub fn grid_range(&self, metrics: &CellGridMetrics) -> IndexRange {
        let [w, h] = self.grid_extent(metrics);
        IndexRange::new(0, 0, w as usize, h as usize)
    }

    /// [`Self::grid_range`] as an extent.
    ///
    /// The range is anchored at the frame origin by construction — a pixel
    /// index is never negative and cell `(0, 0)` starts at the origin — so
    /// two of a range's four numbers are structurally zero. This is the form
    /// a frame stores; `grid_range` is the form it is asked about.
    #[must_use]
    pub fn grid_extent(&self, metrics: &CellGridMetrics) -> [u32; crate::lattice::AXES] {
        // The same refusal `CellGridSlots::write` makes, for the same reason:
        // this is public, and an infinite `cell_w` would report a full-frame
        // range from a metric no program will accept.
        metrics.assert_writable();
        let covered = |cells: u32, cell: f32, frame: u32| -> u32 {
            // Device-pixel extent of the grid on this axis. `ceil(g - ½)` is
            // the count of pixel centres strictly inside `[0, g)`.
            let g = cells as f32 * cell * metrics.scale;
            let n = (g - SAMPLE_CENTER).ceil();
            let n = if n > 0.0 { n as u32 } else { 0 };
            n.min(frame)
        };
        [
            covered(self.cols, metrics.cell_w, self.frame_w),
            covered(self.rows, metrics.cell_h, self.frame_h),
        ]
    }

    /// The four channel kernels this shape denotes — everything from a point
    /// coordinate to a blended channel value — over freshly minted identities
    /// for the two buffers they read and freshly minted slots for the metric
    /// they are parameterized by.
    ///
    /// This is the whole of the cell grid as a *program*: what a consumer does
    /// with the four channels — bake them into planes ([`CellGridProgram`]) or
    /// compose them into one packed-pixel kernel (`pixelflow-graphics`, which
    /// is where byte lanes and pixel formats live) — is the consumer's.
    ///
    /// The kernels answer for the grid and for nothing else. A consumer that
    /// samples a whole frame paints [`Self::grid_range`]'s complement itself;
    /// [`CellGridFrame`] is the instance that does.
    ///
    /// # Panics
    ///
    /// Panics on a degenerate shape: zero cells or an empty atlas.
    #[must_use]
    pub fn channel_kernels(&self) -> CellGridKernels {
        assert_compilable(self);
        let buffers = CellGridBuffers::mint();
        let slots = CellGridSlots::mint();
        CellGridKernels {
            channels: core::array::from_fn(|c| channel_kernel(self, buffers, &slots, c)),
            buffers,
            slots,
        }
    }
}

/// **The handles a cell-grid program's [`CellGridMetrics`] is written
/// through** — one [`Uniform`] per value the kernel reads.
///
/// There are more slots than there are metric fields, because the kernel
/// reads reciprocals and pre-offset clamps rather than dividing per pixel.
/// That is why the fields are private and [`Self::write`] is the only way in:
/// a caller that could set `cell_w` without setting `1/cell_w` could put the
/// program in a state no metric denotes, and the picture would be plausible.
#[derive(Clone, Copy, Debug)]
pub struct CellGridSlots {
    /// Points per device pixel — the display contramap, `1/scale`.
    points_per_pixel: Uniform,
    /// Cell width in points.
    cell_w: Uniform,
    /// Cells per point along X, `1/cell_w`. A reciprocal rather than a
    /// divide: `Recip` is an estimate (~12 bits, CLAUDE.md), and the host
    /// has an exact `f32` division to hand.
    cells_per_point_x: Uniform,
    /// Cell height in points.
    cell_h: Uniform,
    /// Cells per point along Y, `1/cell_h`.
    cells_per_point_y: Uniform,
    /// Atlas texels per point.
    density: Uniform,
    /// The largest intra-cell U offset a tap may reach, `tile_w + ½` texels.
    tile_u_max: Uniform,
    /// The largest intra-cell V offset a tap may reach, `tile_h + ½` texels.
    tile_v_max: Uniform,
}

impl CellGridSlots {
    /// Eight arguments distinct from every other in this process.
    ///
    /// The defaults are the identity metric — one-point cells, one texel per
    /// point, no display scale — so a program collapsed without a block is
    /// well defined and visibly not the caller's grid. Nothing production
    /// reads them: [`CellGridProgram::frame`] takes a block, and
    /// `PackedManifold::bind` refuses a program that has arguments.
    fn mint() -> Self {
        Self {
            points_per_pixel: Uniform::new(1.0),
            cell_w: Uniform::new(1.0),
            cells_per_point_x: Uniform::new(1.0),
            cell_h: Uniform::new(1.0),
            cells_per_point_y: Uniform::new(1.0),
            density: Uniform::new(1.0),
            tile_u_max: Uniform::new(SAMPLE_CENTER),
            tile_v_max: Uniform::new(SAMPLE_CENTER),
        }
    }

    /// Write `metrics` into `block` — the whole of a resize, a zoom or a DPI
    /// change.
    ///
    /// # Errors
    ///
    /// [`UnknownUniform`] when `block` was laid out for a program that does
    /// not declare these slots — a composition mistake, never ignored.
    ///
    /// # Panics
    ///
    /// Panics on a non-positive or non-finite metric
    /// ([`CellGridMetrics::assert_writable`]).
    pub fn write(
        &self,
        block: &mut UniformBlock,
        metrics: &CellGridMetrics,
    ) -> Result<(), UnknownUniform> {
        metrics.assert_writable();
        let values = [
            (self.points_per_pixel, 1.0 / metrics.scale),
            (self.cell_w, metrics.cell_w),
            (self.cells_per_point_x, 1.0 / metrics.cell_w),
            (self.cell_h, metrics.cell_h),
            (self.cells_per_point_y, 1.0 / metrics.cell_h),
            (self.density, metrics.density),
            (self.tile_u_max, metrics.tile_w as f32 + SAMPLE_CENTER),
            (self.tile_v_max, metrics.tile_h as f32 + SAMPLE_CENTER),
        ];
        // Every slot is probed before any is written. `set` on a foreign
        // handle would otherwise leave the block half-written — the exact
        // state the private fields exist to make unrepresentable, reachable
        // again through a partial failure.
        for (slot, _) in values {
            block.get(slot)?;
        }
        for (slot, value) in values {
            block
                .set(slot, value)
                .expect("every slot was probed above and cannot now be unknown");
        }
        Ok(())
    }
}

/// One channel's program: everything from a device-pixel index to a blended
/// channel value, composed from the two samplers rather than assembled node by
/// node.
///
/// Keyed on extents, not data and not metric. A sampler's fragment carries
/// only its buffer declaration, and every measurement is a [`Uniform`], so the
/// whole program composes from a [`CellGridShape`] alone.
fn channel_kernel(
    shape: &CellGridShape,
    bufs: CellGridBuffers,
    slots: &CellGridSlots,
    channel: usize,
) -> Kernel {
    let k = Kernel::constant;
    // The display contramap, precomposed onto the coordinates rather than
    // applied to the finished kernel: `Kernel::at` on the root would
    // substitute into every one of the reads below and double the arena, the
    // cost L1 measured on `scene3d::checker`. Precomposing first costs one
    // extra node per *use* of a coordinate instead.
    let ppp = slots.points_per_pixel.kernel();
    let (x, y) = (Kernel::x().mul(&ppp), Kernel::y().mul(&ppp));

    // Which cell, and where inside it.
    let col = x.mul(&slots.cells_per_point_x.kernel()).floor();
    let row = y.mul(&slots.cells_per_point_y.kernel()).floor();
    let lx = x.sub(&col.mul(&slots.cell_w.kernel()));
    let ly = y.sub(&row.mul(&slots.cell_h.kernel()));

    // Per-cell data. The gathers clamp to the buffer edge, which is why the
    // grid's own index range is the frame's business: a query outside it
    // would read the border cell rather than the background, so nothing asks.
    let cells = DiscreteManifold::kernel_for(bufs.cells, shape.cells_row_width(), shape.rows);
    let cx = col.mul(&k(CELL_STRIDE as f32));
    let field = |offset: usize| {
        let idx = if offset == 0 {
            cx.clone()
        } else {
            cx.add(&k(offset as f32))
        };
        cells.at(&idx, &row)
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
    let density = slots.density.kernel();
    let au = u0.add(&lx.mul(&density).min(&slots.tile_u_max.kernel()));
    let av = v0.add(&ly.mul(&density).min(&slots.tile_v_max.kernel()));
    let atlas = BilinearSampler::kernel_for(bufs.atlas, shape.atlas_width, shape.atlas_height);
    let cov = atlas.at(&au.sub(&k(SAMPLE_CENTER)), &av.sub(&k(SAMPLE_CENTER)));

    // blended = bg + cov·(fg − bg). What lies outside the grid is not this
    // kernel's answer to give: it is an index range nothing collapses.
    bg.add(&cov.mul(&fg.sub(&bg)))
}

/// The two blocks of memory a cell-grid program reads, identified so that
/// composing many reads of one buffer still merges to one slot, and so binding
/// can say which slot is which without inferring it from extents.
#[derive(Clone, Copy, Debug)]
pub struct CellGridBuffers {
    /// The per-cell data buffer: [`CELL_STRIDE`] `f32`s per cell, row-major.
    pub cells: BufferIdentity,
    /// The coverage atlas, row-major texels.
    pub atlas: BufferIdentity,
}

impl CellGridBuffers {
    /// Two identities distinct from every other in this process.
    #[must_use]
    pub fn mint() -> Self {
        Self {
            cells: BufferIdentity::mint(),
            atlas: BufferIdentity::mint(),
        }
    }
}

/// A shape's four channel kernels, the buffers they read, and the slots their
/// metric is written through ([`CellGridShape::channel_kernels`]).
pub struct CellGridKernels {
    /// One kernel per channel, in `(r, g, b, a)` order.
    pub channels: [Kernel; 4],
    /// The identities those kernels declared, for binding data to them.
    pub buffers: CellGridBuffers,
    /// The arguments those kernels declared, for writing a
    /// [`CellGridMetrics`] into a block laid out for them.
    pub slots: CellGridSlots,
}

/// The shape preconditions every cell-grid compile shares.
///
/// # Panics
///
/// Panics on a degenerate shape: zero cells or an empty atlas. Buffers too
/// large to index exactly in `f32` are refused by [`Manifold::compile`],
/// which sees every declared buffer's real extents rather than this shape's
/// idea of them. Metric preconditions are [`CellGridMetrics`]'s, checked
/// where a metric is written.
fn assert_compilable(shape: &CellGridShape) {
    assert!(
        shape.cols > 0 && shape.rows > 0 && shape.atlas_width > 0 && shape.atlas_height > 0,
        "cell-grid compile: degenerate shape {shape:?}"
    );
    // Both products must be computable before anything downstream trusts
    // them: cells_len/cells_row_width panic on overflow rather than wrapping,
    // and a wrapped length paired with an unwrapped row width is how a safe
    // API ends up gathering past the end of the bound buffer.
    // (Both accessors panic on overflow; calling them here is the check.)
    assert!(
        shape.cells_len() > 0 && shape.cells_row_width() > 0,
        "cell-grid compile: empty cell buffer for {shape:?}"
    );
}

/// The four compiled channel kernels for one grid/atlas shape, each baking
/// into its own `f32` plane.
///
/// The production frame path is the *packed* program a layer up
/// (`pixelflow-graphics`), which folds these four channels into one kernel
/// whose root is a finished pixel; this one is the per-channel form — the
/// parity oracle that pack is checked against, and the way to read one
/// channel as a field.
///
/// Compile once per shape ([`CellGridProgram::compile`]); lay out a metric
/// with [`CellGridProgram::params`] and stamp per-frame data into it with
/// [`CellGridProgram::frame`]. **A change of metric is a parameter write, not
/// a compile**: params for a new cell size or display scale go through the
/// same `CellGridProgram`, and every `Manifold` in it is the one that was
/// there before.
pub struct CellGridProgram {
    shape: CellGridShape,
    buffers: CellGridBuffers,
    slots: CellGridSlots,
    default_bg: [f32; 4],
    channels: [Manifold; 4],
}

/// One metric, laid out for one program: the argument values every channel
/// reads, and the index range the grid covers at that metric.
///
/// Made by [`CellGridProgram::params`] and handed to
/// [`CellGridProgram::frame`]. Making one is where the arithmetic of a resize
/// happens; a frame that reuses it costs nothing but refcounts.
pub struct CellGridParams {
    blocks: [UniformBlock; 4],
    grid: [u32; crate::lattice::AXES],
}

impl CellGridParams {
    /// The frame indices the grid covers at this metric — see
    /// [`CellGridShape::grid_range`].
    #[must_use]
    pub fn grid_range(&self) -> IndexRange {
        grid_range_of(self.grid)
    }
}

/// A grid extent as the index range it names, anchored at the frame origin.
fn grid_range_of(extent: [u32; crate::lattice::AXES]) -> IndexRange {
    IndexRange::new(0, 0, extent[0] as usize, extent[1] as usize)
}

impl CellGridProgram {
    /// JIT-compile the four channel programs for `shape`, with `default_bg`
    /// (linear RGBA) painted over the frame indices outside the grid.
    ///
    /// `default_bg` is not compiled into anything — it is the value the
    /// border range is filled with — so two programs differing only in it
    /// share one compiled kernel.
    ///
    /// # Panics
    ///
    /// Panics on a degenerate shape (zero cells, empty atlas), when this
    /// build's `Field` width does not match the JIT's emitted width, or if
    /// compilation fails.
    #[must_use]
    pub fn compile(shape: CellGridShape, default_bg: [f32; 4]) -> Self {
        let CellGridKernels {
            channels,
            buffers,
            slots,
        } = shape.channel_kernels();
        let extent = [shape.frame_w, shape.frame_h];
        Self {
            shape,
            buffers,
            slots,
            default_bg,
            channels: channels.map(|kernel| Manifold::compile(&kernel, extent)),
        }
    }

    /// The shape this program was compiled for.
    #[must_use]
    pub fn shape(&self) -> &CellGridShape {
        &self.shape
    }

    /// Lay `metrics` out for this program: one block per channel, plus the
    /// grid's index range at that metric. **This is the whole of a resize
    /// that keeps the shape** — no arena, no saturation, no compile.
    ///
    /// # Panics
    ///
    /// Panics on a non-positive or non-finite metric.
    #[must_use]
    pub fn params(&self, metrics: &CellGridMetrics) -> CellGridParams {
        CellGridParams {
            blocks: core::array::from_fn(|c| {
                let mut block = self.channels[c].block();
                self.slots
                    .write(&mut block, metrics)
                    .expect("the slots were minted with these kernels");
                block
            }),
            grid: self.shape.grid_extent(metrics),
        }
    }

    /// Bind one frame's data. `cells` is [`CELL_STRIDE`] `f32`s per cell,
    /// row-major; `atlas` is the coverage atlas, row-major texels. Both are
    /// `Arc`s so a frame in flight keeps its data alive while the caller
    /// prepares the next one.
    ///
    /// `params` is required rather than defaulted: a frame drawn at the
    /// slots' defaults would be a one-point-cell grid, which is a picture,
    /// and a plausible-looking wrong picture is the worst failure there is.
    ///
    /// # Panics
    ///
    /// Panics if either buffer's length does not match the shape, or if
    /// `params` was laid out for a different program.
    #[must_use]
    pub fn frame(
        &self,
        params: &CellGridParams,
        cells: Arc<Vec<f32>>,
        atlas: Arc<Vec<f32>>,
    ) -> CellGridFrame {
        let bound = [(self.buffers.cells, cells), (self.buffers.atlas, atlas)];
        CellGridFrame {
            channels: core::array::from_fn(|c| {
                self.channels[c]
                    .bind(&bound)
                    .with_uniforms(&params.blocks[c])
            }),
            grid: params.grid,
            default_bg: self.default_bg,
        }
    }
}

/// One frame of a cell grid: the compiled channels, the data and metric they
/// read, the index range the grid covers, and the background outside it.
/// Cheap to clone.
#[derive(Clone)]
pub struct CellGridFrame {
    channels: [BoundManifold; 4],
    /// The grid's device-pixel extent — see [`CellGridShape::grid_extent`]
    /// for why a frame holds an extent rather than the range itself.
    grid: [u32; crate::lattice::AXES],
    default_bg: [f32; 4],
}

impl CellGridFrame {
    /// The frame indices the grid covers.
    #[must_use]
    pub fn grid_range(&self) -> IndexRange {
        grid_range_of(self.grid)
    }

    /// Collapse one color channel (0 = R, 1 = G, 2 = B, 3 = A) over the pixel
    /// index band `band` into a plane whose rows are `stride` samples apart.
    ///
    /// The band is an [`IndexRange`] and not a coordinate: what the samples
    /// of a frame are is the frame's business (pixel centers), and what a
    /// caller chooses is *which* samples. The grid's kernel is collapsed over
    /// `band ∩ grid_range` and the rest of the band is painted with the
    /// default background — two disjoint ranges, no mask.
    ///
    /// # Panics
    ///
    /// Panics if `channel >= 4`, the band starts at a column other than 0,
    /// its width is zero, `stride` is less than it, or `out` cannot hold the
    /// band.
    pub fn collapse_channel_rows(
        &self,
        channel: usize,
        band: IndexRange,
        out: &mut [f32],
        stride: usize,
    ) {
        assert_eq!(
            band.x0(),
            0,
            "CellGridFrame::collapse_channel_rows: a frame band starts at column 0"
        );
        let claim = self.grid_range().intersect(&band);
        if !claim.is_empty() {
            self.channels[channel].collapse_subrect(
                PlaneRegion::rows(claim.width(), claim.y0(), claim.rows()),
                out,
                stride,
            );
        }
        band.paint_complement(claim, out, stride, self.default_bg[channel]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// The 2x1 grid's shape: a 2-tile atlas of 4x4-content tiles with
    /// 1-texel aprons, laid out side by side (12x6 texels), over a 12x6
    /// frame.
    const TINY_SHAPE: CellGridShape = CellGridShape {
        cols: 2,
        rows: 1,
        atlas_width: 12,
        atlas_height: 6,
        frame_w: 12,
        frame_h: 6,
    };

    /// The 2x1 grid's metric: 4x4-point cells, one texel per point, no
    /// display scale.
    const TINY_METRICS: CellGridMetrics = CellGridMetrics {
        cell_w: 4.0,
        cell_h: 4.0,
        density: 1.0,
        tile_w: 4,
        tile_h: 4,
        scale: 1.0,
    };

    /// Two 4x4-content tiles with 1-texel aprons: tile 0 solid coverage 1,
    /// tile 1 the coverage `fill` field.
    fn two_tile_atlas(fill: f32) -> Vec<f32> {
        let (aw, slot) = (12usize, 6usize);
        let mut atlas = vec![0.0f32; aw * 6];
        for r in 0..4 {
            for c in 0..4 {
                atlas[(1 + r) * aw + 1 + c] = 1.0;
                atlas[(1 + r) * aw + slot + 1 + c] = fill;
            }
        }
        atlas
    }

    /// A 2x1 grid over a 2-tile atlas: tile A solid coverage 1, tile B a
    /// half-coverage field. Colors chosen so every channel distinguishes
    /// fg, bg, and the default background.
    fn tiny_scene() -> (CellGridProgram, Arc<Vec<f32>>, Arc<Vec<f32>>) {
        let program = CellGridProgram::compile(TINY_SHAPE, [0.9, 0.8, 0.7, 0.6]);
        // Cell 0 → tile 0, red-on-black; cell 1 → tile 1, white-on-blue.
        let cells = vec![
            1.0, 1.0, /* uv */ 1.0, 0.0, 0.0, 1.0, /* fg */ 0.0, 0.0, 0.0,
            1.0, /* bg */
            7.0, 1.0, /* uv */ 1.0, 1.0, 1.0, 1.0, /* fg */ 0.0, 0.0, 1.0,
            1.0, /* bg */
        ];
        (program, Arc::new(cells), Arc::new(two_tile_atlas(0.5)))
    }

    /// Splicing merges buffer declarations by identity, so the nine reads in
    /// a channel program (four cell fields — `bg` twice — plus the atlas)
    /// occupy exactly two slots: one per distinct buffer.
    ///
    /// Nodes are a different story, and this pins the gap: every *use* of a
    /// Kernel value re-splices its whole fragment, and the display contramap
    /// makes each use of a coordinate two nodes rather than one. Identity
    /// fixed the slots; it cannot fix this. Node sharing is an inlining
    /// decision, and it belongs to the e-graph — which cannot see this arena
    /// at all while `Gather` is unrepresentable there. This number falling is
    /// the sign that landed.
    #[test]
    fn reads_of_one_buffer_share_a_slot_but_not_nodes() {
        let shape = CellGridShape {
            frame_w: 8,
            frame_h: 4,
            ..TINY_SHAPE
        };
        let CellGridKernels {
            channels, buffers, ..
        } = shape.channel_kernels();
        let term = channels[0].term();

        let slots = &term.env().buffers;
        assert_eq!(
            slots.iter().filter(|d| d.id == buffers.cells).count(),
            1,
            "five cell reads, one buffer, one slot"
        );
        assert_eq!(
            slots.iter().filter(|d| d.id == buffers.atlas).count(),
            1,
            "one slot for the atlas"
        );
        assert_eq!(slots.len(), 2, "exactly {{Cells, Atlas}}, nothing else");
        assert_eq!(
            term.root().descendants().count(),
            CHANNEL_NODES,
            "composed node count"
        );
    }

    /// Reachable nodes in one channel kernel. Pinned rather than derived:
    /// see `reads_of_one_buffer_share_a_slot_but_not_nodes`.
    pub(crate) const CHANNEL_NODES: usize = 157;

    /// Row-major index into an 8-wide plane.
    fn px(row: usize, col: usize) -> usize {
        row * 8 + col
    }

    /// The whole `w x h` frame as one index band.
    fn whole(w: usize, h: usize) -> IndexRange {
        IndexRange::new(0, 0, w, h)
    }

    /// Bake one channel over a w×h frame into a dense plane.
    fn plane(frame: &CellGridFrame, channel: usize, w: usize, h: usize) -> alloc::vec::Vec<f32> {
        let mut dense = vec![0.0f32; h * w];
        frame.collapse_channel_rows(channel, whole(w, h), &mut dense, w);
        dense
    }

    /// `tiny_scene`'s four channel planes at `TINY_METRICS`, 8x4.
    fn tiny_planes() -> [alloc::vec::Vec<f32>; 4] {
        let (program, cells, atlas) = tiny_scene();
        let frame = program.frame(&program.params(&TINY_METRICS), cells, atlas);
        core::array::from_fn(|c| plane(&frame, c, 8, 4))
    }

    #[test]
    fn interior_texels_blend_fg_over_bg_by_coverage() {
        let p = tiny_planes();
        let (r, b) = (&p[0], &p[2]);
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
        let params = program.params(&TINY_METRICS);
        // The grid is 8x4 device pixels; the 12x6 frame's margin is not.
        assert_eq!(params.grid_range(), IndexRange::new(0, 0, 8, 4));
        let frame = program.frame(&params, cells, atlas);
        let g = plane(&frame, 1, 12, 6);
        // (10, 0) is right of the grid; (0, 5) below it.
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
        let p = tiny_planes();
        let r = &p[0];
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
        let shape = CellGridShape {
            cols: 1,
            rows: 1,
            frame_w: 5,
            frame_h: 5,
            ..TINY_SHAPE
        };
        let metrics = CellGridMetrics {
            cell_w: 5.0,
            cell_h: 5.0,
            density: 0.8,
            ..TINY_METRICS
        };
        let program = CellGridProgram::compile(shape, [0.0; 4]);
        let cells = vec![
            1.0, 1.0, /* fg */ 1.0, 1.0, 1.0, 1.0, /* bg */ 0.0, 0.0, 0.0, 1.0,
        ];
        // Tile 1 ALSO solid: a tap that bled into it would read 1.0.
        let frame = program.frame(
            &program.params(&metrics),
            Arc::new(cells),
            Arc::new(two_tile_atlas(1.0)),
        );
        let r = plane(&frame, 0, 5, 5);
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
        let shape = CellGridShape {
            cols: 1,
            rows: 1,
            frame_w: 8,
            frame_h: 4,
            ..TINY_SHAPE
        };
        let metrics = CellGridMetrics {
            cell_w: 8.0,
            ..TINY_METRICS
        };
        let program = CellGridProgram::compile(shape, [0.0; 4]);
        let cells = vec![
            1.0, 1.0, /* fg */ 1.0, 1.0, 1.0, 1.0, /* bg */ 0.0, 0.0, 0.0, 1.0,
        ];
        let frame = program.frame(
            &program.params(&metrics),
            Arc::new(cells),
            Arc::new(two_tile_atlas(1.0)),
        );
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
        let shape = CellGridShape {
            cols: 1,
            rows: 1,
            atlas_width: 1 << 13,
            atlas_height: 1 << 12, // 2^25 texels: past exact f32 indexing
            frame_w: 4,
            frame_h: 4,
        };
        let _refused = CellGridProgram::compile(shape, [0.0; 4]);
    }

    #[test]
    #[should_panic(expected = "floats bound to slot")]
    fn mismatched_cell_buffer_is_refused() {
        let (program, _, atlas) = tiny_scene();
        // Binding satisfies unused_must_use; the call panics first.
        let _refused = program.frame(
            &program.params(&TINY_METRICS),
            Arc::new(alloc::vec![0.0; 3]),
            atlas,
        );
    }

    #[test]
    #[should_panic(expected = "non-positive or non-finite")]
    fn a_zero_cell_width_is_refused_where_it_is_written() {
        // 1/0 is +inf, so every pixel of the frame would read cell 0 and the
        // picture would be a plausible one. Refused at the write.
        let (program, _, _) = tiny_scene();
        let _refused = program.params(&CellGridMetrics {
            cell_w: 0.0,
            ..TINY_METRICS
        });
    }

    #[test]
    fn cells_len_and_atlas_len_match_declared_shape() {
        let shape = CellGridShape {
            cols: 3,
            rows: 2,
            atlas_width: 12,
            atlas_height: 6,
            frame_w: 12,
            frame_h: 8,
        };
        assert_eq!(shape.cells_len(), 3 * 2 * CELL_STRIDE);
        assert_eq!(shape.atlas_len(), 12 * 6);
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
        let g = &tiny_planes()[1];
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
        let shape = CellGridShape {
            cols: 1,
            rows: 1,
            frame_w: 4,
            frame_h: 8,
            ..TINY_SHAPE
        };
        let metrics = CellGridMetrics {
            cell_h: 8.0,
            ..TINY_METRICS
        };
        let program = CellGridProgram::compile(shape, [0.0; 4]);
        let cells = vec![
            1.0, 1.0, /* fg */ 1.0, 1.0, 1.0, 1.0, /* bg */ 0.0, 0.0, 0.0, 1.0,
        ];
        let frame = program.frame(
            &program.params(&metrics),
            Arc::new(cells),
            Arc::new(two_tile_atlas(1.0)),
        );
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
        // A 3x3 grid of 2x2 cells: grid_w = grid_h = 6.0. If the range ever
        // computed cols + cell_w (5.0) instead of cols * cell_w, the last
        // row/column of real cells (spanning [4, 6)) would be wrongly
        // treated as outside the grid.
        let (aw, ah) = (6usize, 6usize);
        let mut atlas = vec![0.0f32; aw * ah];
        for r in 0..4 {
            for c in 0..4 {
                atlas[(1 + r) * aw + 1 + c] = 1.0; // solid tile
            }
        }
        let shape = CellGridShape {
            cols: 3,
            rows: 3,
            atlas_width: aw as u32,
            atlas_height: ah as u32,
            frame_w: 8,
            frame_h: 8,
        };
        let metrics = CellGridMetrics {
            cell_w: 2.0,
            cell_h: 2.0,
            ..TINY_METRICS
        };
        let program = CellGridProgram::compile(shape, [0.4; 4]);
        let params = program.params(&metrics);
        assert_eq!(params.grid_range(), IndexRange::new(0, 0, 6, 6));
        let block: [f32; CELL_STRIDE] = [
            1.0, 1.0, /* fg */ 1.0, 0.0, 0.0, 1.0, /* bg */ 0.0, 0.0, 0.0, 1.0,
        ];
        let mut cells = vec![];
        for _ in 0..9 {
            cells.extend_from_slice(&block);
        }
        let frame = program.frame(&params, Arc::new(cells), Arc::new(atlas));
        let r = plane(&frame, 0, 8, 8);
        // (5, 5) is interior to the last (col 2, row 2) cell — must read
        // pure fg (1.0), not the default background (0.4) a cols+cell_w
        // range would wrongly clip it to.
        assert!(
            (r[5 * 8 + 5] - 1.0).abs() < 1e-5,
            "last cell R = {}",
            r[5 * 8 + 5]
        );
    }

    #[test]
    fn baking_a_row_offset_band_samples_the_correct_absolute_row() {
        // A 1x2 grid (one column, two rows) with visibly different content
        // per row. Collapsing a band whose `y0` starts mid-grid must read the
        // ABSOLUTE row that `y0` implies. A constant error in the pixel-center
        // offset shifts every sampled row by the same amount, so a
        // self-consistency check (comparing this bake against a full bake
        // sliced at the same relative offset) can't see it — this asserts a
        // concrete expected value instead.
        let shape = CellGridShape {
            cols: 1,
            rows: 2,
            frame_w: 4,
            frame_h: 8,
            ..TINY_SHAPE
        };
        let program = CellGridProgram::compile(shape, [0.0; 4]);
        let cells = vec![
            1.0, 1.0, /* row0 fg */ 1.0, 0.0, 0.0, 1.0, /* bg */ 0.0, 0.0, 0.0, 1.0, 7.0,
            1.0, /* row1 fg */ 0.0, 0.0, 1.0, 1.0, /* bg */ 1.0, 1.0, 1.0, 1.0,
        ];
        let frame = program.frame(
            &program.params(&TINY_METRICS),
            Arc::new(cells),
            Arc::new(two_tile_atlas(0.5)),
        );

        let mut dense = vec![0.0f32; 4 * 4];
        frame.collapse_channel_rows(0, IndexRange::new(0, 4, 4, 4), &mut dense, 4);
        // y0 = 4 → the first collapsed row samples absolute y = 4.5, inside
        // row 1's cell (coverage 0.5, R: bg 1.0 -> fg 0.0 blends to 0.5). An
        // off-by-one-pixel-center bug would instead sample y = 3.5, still
        // inside row 0's cell (solid fg, R = 1.0).
        assert!(
            (dense[1] - 0.5).abs() < 1e-5,
            "row-offset bake read the wrong absolute row: R = {}",
            dense[1]
        );
    }

    /// **A change of metric is a parameter write.** Two programs of one
    /// shape, at different cell sizes, are the same compiled code — and one
    /// program re-parameterized to the other's metric draws the other's
    /// picture, bit for bit.
    ///
    /// The comparison is against a *separately constructed* program rather
    /// than against the same object, so what is asserted is that the metric
    /// reached the kernel, not merely that something changed.
    #[test]
    fn a_new_metric_is_a_parameter_write_not_a_compile() {
        let shape = CellGridShape {
            cols: 2,
            rows: 2,
            frame_w: 16,
            frame_h: 16,
            ..TINY_SHAPE
        };
        let small = CellGridMetrics {
            cell_w: 4.0,
            cell_h: 4.0,
            ..TINY_METRICS
        };
        let large = CellGridMetrics {
            cell_w: 8.0,
            cell_h: 8.0,
            ..TINY_METRICS
        };
        let cells: Vec<f32> = (0..4)
            .flat_map(|i| {
                let f = i as f32;
                [1.0, 1.0, f * 0.25, 0.5, 0.75, 1.0, 0.1, 0.2, 0.3, 1.0]
            })
            .collect();
        let (cells, atlas) = (Arc::new(cells), Arc::new(two_tile_atlas(0.5)));

        let a = CellGridProgram::compile(shape, [0.9, 0.8, 0.7, 0.6]);
        let b = CellGridProgram::compile(shape, [0.9, 0.8, 0.7, 0.6]);
        for c in 0..4 {
            assert!(
                core::ptr::eq(
                    a.channels[c].code_bytes().as_ptr(),
                    b.channels[c].code_bytes().as_ptr()
                ),
                "one shape, one compiled kernel per channel"
            );
        }

        // `a` renders `large` through a parameter write; `b` was built for
        // nothing else. Same program, same picture.
        let from_write = plane(
            &a.frame(&a.params(&large), cells.clone(), atlas.clone()),
            0,
            16,
            16,
        );
        let fresh = plane(
            &b.frame(&b.params(&large), cells.clone(), atlas.clone()),
            0,
            16,
            16,
        );
        assert_eq!(
            from_write.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            fresh.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "a metric written into the block must be the metric the kernel reads"
        );

        // And the two metrics genuinely differ, so the equality above is not
        // vacuous.
        let at_small = plane(&a.frame(&a.params(&small), cells, atlas), 0, 16, 16);
        assert_ne!(at_small, from_write, "the two metrics must not agree");
    }

    /// **A display scale is a contramap.** Scaling the coordinates by `s`
    /// and measuring in points is the same function as measuring in device
    /// pixels at scale 1: `cell_w = c` at `scale = s` and `cell_w = c·s` at
    /// `scale = 1` are two spellings of one program, and they agree on every
    /// sample bit for bit.
    ///
    /// That equality is the whole content of "DPI is a contramap, not a
    /// property of the domain": the display scale composes with the metric
    /// rather than sitting beside it.
    #[test]
    fn a_display_scale_is_a_contramap_on_the_coordinates() {
        let shape = CellGridShape {
            frame_w: 16,
            frame_h: 8,
            ..TINY_SHAPE
        };
        let in_points = CellGridMetrics {
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            scale: 2.0,
            ..TINY_METRICS
        };
        let in_pixels = CellGridMetrics {
            cell_w: 8.0,
            cell_h: 8.0,
            density: 0.5,
            scale: 1.0,
            ..TINY_METRICS
        };
        let cells = Arc::new(vec![
            1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 7.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.0,
        ]);
        let atlas = Arc::new(two_tile_atlas(0.5));
        let program = CellGridProgram::compile(shape, [0.0; 4]);
        assert_eq!(
            program.params(&in_points).grid_range(),
            program.params(&in_pixels).grid_range(),
            "the same grid, measured two ways, covers the same pixels"
        );
        let mut varied = false;
        for channel in 0..4 {
            let a = plane(
                &program.frame(&program.params(&in_points), cells.clone(), atlas.clone()),
                channel,
                16,
                8,
            );
            let b = plane(
                &program.frame(&program.params(&in_pixels), cells.clone(), atlas.clone()),
                channel,
                16,
                8,
            );
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "channel {channel}: the scale did not compose with the metric"
            );
            varied |= a.iter().any(|v| (v - a[0]).abs() > 1e-3);
        }
        // Not vacuous: the picture has structure, not one flat value.
        assert!(
            varied,
            "every channel is constant, so the comparison proves nothing"
        );
    }

    /// A block laid out for one program is refused by another: its offsets
    /// would be a different kernel's, and the pixels would be plausible.
    #[test]
    #[should_panic(expected = "laid out for a different program")]
    fn params_from_another_program_are_refused() {
        let (a, cells, atlas) = tiny_scene();
        let b = CellGridProgram::compile(
            CellGridShape {
                cols: 3,
                ..TINY_SHAPE
            },
            [0.0; 4],
        );
        let _refused = a.frame(&b.params(&TINY_METRICS), cells, atlas);
    }

    /// Every channel reads the whole metric, which is what makes
    /// `Manifold::bind` without a block — and `PackedManifold::bind`, which
    /// refuses a program with arguments outright — the wrong call for a
    /// cell grid.
    #[test]
    fn every_channel_declares_the_whole_metric() {
        let program = CellGridProgram::compile(TINY_SHAPE, [0.0; 4]);
        for (c, channel) in program.channels.iter().enumerate() {
            assert_eq!(
                channel.uniforms().len(),
                METRIC_SLOTS,
                "channel {c} does not read the whole metric"
            );
        }
    }

    /// Arguments a cell-grid channel kernel declares: the display contramap,
    /// two cell extents and their reciprocals, the sample density, and two
    /// tile clamps.
    const METRIC_SLOTS: usize = 8;

    /// One configuration of `the_border_range_agrees_with_the_mask_it_replaced`.
    struct MaskCase {
        cols: u32,
        rows: u32,
        cell_w: f32,
        cell_h: f32,
        /// Device pixels per point. The one axis of this fixture that the
        /// rest of the module never varies.
        scale: f32,
        frame_w: u32,
        frame_h: u32,
        /// The grid extent this case must produce, stated rather than
        /// derived — a case whose expected value came from the code under
        /// test would pin nothing.
        grid: [u32; 2],
    }

    impl MaskCase {
        fn shape(&self) -> CellGridShape {
            CellGridShape {
                cols: self.cols,
                rows: self.rows,
                frame_w: self.frame_w,
                frame_h: self.frame_h,
                ..TINY_SHAPE
            }
        }

        fn metrics(&self) -> CellGridMetrics {
            CellGridMetrics {
                cell_w: self.cell_w,
                cell_h: self.cell_h,
                scale: self.scale,
                ..TINY_METRICS
            }
        }
    }

    /// **The border range is the mask it replaced.** The old kernel totalized
    /// the grid's extent into its range — `in_grid.select(blended,
    /// default_bg)`, evaluated at every pixel of the frame — and this one
    /// restricts the index instead. The two are different programs (the
    /// masked one is a bigger arena, extracted on its own), so this is a
    /// comparison of two encodings and not of one thing with itself.
    ///
    /// Asserted bit-exactly, over configurations chosen so that each of the
    /// range's own arithmetic decisions is load-bearing in at least one:
    ///
    /// | case | grid extent | what it pins |
    /// |---|---|---|
    /// | integral | 12 x 8 | the ordinary case; a border on both axes |
    /// | fractional | 17 x 12 | **the `− ½`**: the grid spans `[0, 17.5)`, so pixel 17 (centre 17.5) is outside, and a range computed as `ceil(g)` would claim it. Also a claim width that is no multiple of any SIMD batch, so `RowTail::Exact`'s scratch path is live. |
    /// | fractional through the scale | 22 x 11 | the same, with the display contramap carrying it: `3 · 5 · 1.5 = 22.5` is fractional though every cell extent on X is integral, and `3 · 2.5 · 1.5 = 11.25` is fractional in both. **Both axes discriminate** — `ceil(g)` gives 23 x 12. At `scale = 2.0`, which this case had first, both products came out integral and it pinned nothing about the half-pixel however fractional its cell extents looked. |
    /// | grid past the frame | 15 x 11 | the clip to the frame, with no border at all |
    ///
    /// Without the fractional rows this test — and every other fixture in
    /// this module — passes with the `− ½` deleted, because `ceil(g − ½)` and
    /// `ceil(g)` agree on every integral `g`. That is a missing fixture, and
    /// these are it.
    #[test]
    fn the_border_range_agrees_with_the_mask_it_replaced() {
        let cases = [
            MaskCase {
                cols: 3,
                rows: 2,
                cell_w: 4.0,
                cell_h: 4.0,
                scale: 1.0,
                frame_w: 15,
                frame_h: 11,
                grid: [12, 8],
            },
            MaskCase {
                cols: 5,
                rows: 5,
                cell_w: 3.5,
                cell_h: 2.5,
                scale: 1.0,
                frame_w: 24,
                frame_h: 20,
                grid: [17, 12],
            },
            MaskCase {
                cols: 3,
                rows: 3,
                // Integral on X, fractional on Y, and a fractional scale over
                // both: `3 · 5 · 1.5 = 22.5` is fractional through the
                // contramap ALONE, which is what this case is here to say.
                // At `scale = 2.0` — what this case had before — every
                // product came out integral and it pinned nothing about the
                // half-pixel however fractional its cell extents looked.
                cell_w: 5.0,
                cell_h: 2.5,
                scale: 1.5,
                frame_w: 40,
                frame_h: 18,
                grid: [22, 11],
            },
            MaskCase {
                cols: 4,
                rows: 4,
                cell_w: 4.0,
                cell_h: 4.0,
                scale: 1.0,
                frame_w: 15,
                frame_h: 11,
                grid: [15, 11],
            },
        ];
        for case in cases {
            let (shape, metrics, want_grid) = (case.shape(), case.metrics(), case.grid);
            let (cols, rows) = (case.cols, case.rows);
            let (cell_w, cell_h, scale) = (case.cell_w, case.cell_h, case.scale);
            let (frame_w, frame_h) = (case.frame_w, case.frame_h);
            let default_bg = [0.9, 0.8, 0.7, 0.6];
            let cells: Vec<f32> = (0..(cols * rows))
                .flat_map(|i| {
                    let f = i as f32;
                    [
                        1.0 + 6.0 * (i % 2) as f32,
                        1.0,
                        (f * 0.15) % 1.0,
                        0.5,
                        0.75,
                        1.0,
                        0.1,
                        0.2,
                        0.3,
                        1.0,
                    ]
                })
                .collect();
            let (cells, atlas) = (Arc::new(cells), Arc::new(two_tile_atlas(0.5)));

            // The range encoding: what this module now does.
            let program = CellGridProgram::compile(shape, default_bg);
            let params = program.params(&metrics);
            assert_eq!(
                params.grid_range(),
                IndexRange::new(0, 0, want_grid[0] as usize, want_grid[1] as usize),
                "{shape:?} at {metrics:?}"
            );
            let ranged = program.frame(&params, cells.clone(), atlas.clone());

            // The value encoding: the same channel kernels under the mask the
            // select used to carry, evaluated over the whole frame.
            let CellGridKernels {
                channels,
                buffers,
                slots,
            } = shape.channel_kernels();
            let k = Kernel::constant;
            let (x, y) = (Kernel::x(), Kernel::y());
            let in_grid = x
                .ge(&k(0.0))
                .and(&x.lt(&k(cols as f32 * cell_w * scale)))
                .and(&y.ge(&k(0.0)))
                .and(&y.lt(&k(rows as f32 * cell_h * scale)));
            let bound = [(buffers.cells, cells), (buffers.atlas, atlas)];
            let (w, h) = (frame_w as usize, frame_h as usize);
            for (c, channel) in channels.iter().enumerate() {
                let masked = in_grid.select(channel, &k(default_bg[c]));
                let manifold = Manifold::compile(&masked, [frame_w, frame_h]);
                let mut block = manifold.block();
                slots.write(&mut block, &metrics).expect("same slots");
                let mut want = vec![0.0f32; w * h];
                manifold.bind(&bound).with_uniforms(&block).collapse_rows(
                    PlaneRegion::rows(w, 0, h),
                    &mut want,
                    w,
                );

                let got = plane(&ranged, c, w, h);
                assert_eq!(
                    got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "channel {c} of {shape:?} at {metrics:?}: the range encoding \
                     and the mask encoding disagree"
                );
            }
        }
    }

    /// A metric whose *derived* values are not finite is refused where it is
    /// written, not where the picture is seen. A subnormal cell extent passes
    /// every check on the input — positive, finite — and then `1/cell_w`
    /// overflows to `+inf` and every pixel of the frame reads cell 0.
    #[test]
    fn a_metric_whose_reciprocal_overflows_is_refused_where_it_is_written() {
        let program = CellGridProgram::compile(TINY_SHAPE, [0.0; 4]);
        let refused = [
            CellGridMetrics {
                cell_w: 0.0,
                ..TINY_METRICS
            },
            CellGridMetrics {
                cell_w: 1e-42,
                ..TINY_METRICS
            },
            CellGridMetrics {
                cell_h: -4.0,
                ..TINY_METRICS
            },
            CellGridMetrics {
                scale: 1e-42,
                ..TINY_METRICS
            },
            CellGridMetrics {
                scale: f32::INFINITY,
                ..TINY_METRICS
            },
            CellGridMetrics {
                density: f32::NAN,
                ..TINY_METRICS
            },
            CellGridMetrics {
                tile_w: 0,
                ..TINY_METRICS
            },
            CellGridMetrics {
                tile_h: 0,
                ..TINY_METRICS
            },
        ];
        for metrics in refused {
            assert!(
                std::panic::catch_unwind(|| program.params(&metrics)).is_err(),
                "{metrics:?} was accepted"
            );
            // The public range accessors refuse it too: reporting a
            // full-frame range for a metric no program will take is exactly
            // the kind of plausible answer this type refuses to give.
            assert!(
                std::panic::catch_unwind(|| TINY_SHAPE.grid_range(&metrics)).is_err(),
                "{metrics:?} was accepted by grid_range"
            );
        }
    }
}
