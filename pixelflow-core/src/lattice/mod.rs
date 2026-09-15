//! # Lattice: Representable Functor for Kernel Evaluation
//!
//! A Lattice is a finite box domain that tabulates a [`Manifold`] — a
//! [`Kernel`] compiled at that domain's shape — into a discrete buffer. This
//! is the `tabulate`/`index` pair from representable functors:
//!
//! - **[`Lattice::collapse`]** = `tabulate`: `(Rep -> a) -> F a` -- the manifold at every point
//! - **[`DiscreteManifold::kernel`]** = `index`: `F a -> Rep -> a` -- a gather
//!   that reads the buffer back by coordinate, composable into any kernel
//! - **Isomorphism**: `index(collapse(m), i) = m(coord(i))` (up to discretization)
//!
//! Nothing computes until a Lattice demands it. A single-point evaluation is
//! not a lattice at all -- it is [`Lattice::eval_at`], a different operation
//! with a different signature: it answers one value, not a domain to
//! tabulate.
//!
//! There is one `Lattice` type, not one per shape. An axis with extent 1 is
//! fixed; an axis with extent > 1 is a loop dimension. The constructors
//! (`frame`, `index2`, `index`) are sugar for common shapes -- the shape is
//! data, not a type. Extents only need to be static at JIT-compile time,
//! which is when the kernel is specialized. A coordinate frame is never part
//! of the shape: pixel-center sampling, a placed glyph, a DPI scale are all
//! contramaps on the kernel (`Kernel::at`), composed before the lattice ever
//! sees a coordinate.
//!
//! A sub-lattice is an [`IndexRange`](crate::IndexRange): a rectangular
//! restriction of the index, never a coordinate — the domain-side encoding of
//! an extent, the counterpart to a `select` mask on the range side.
//!
//! A kernel with a lattice is the evaluation API:
//!
//! ```text
//! kernel ──compile(shape)──▶ manifold ──bind(buffers)──▶ bound ──collapse──▶ buffer
//! ```
//!
//! One kernel is compiled for the whole domain and the loop nest lives inside
//! the emitted code, so the compiler owns the hoisting and the register
//! allocation across all of it. [`Lattice::bake`] is that line for a kernel
//! that reads no memory. Reductions are a binder in the kernel
//! (`Kernel::over` and friends), not a fold the lattice performs.
//!
//! [`Kernel`]: pixelflow_ir::Kernel
//! [`Manifold`]: crate::Manifold

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

// ============================================================================
// DiscreteManifold: The result of collapsing a lattice
// ============================================================================

/// The result of collapsing a Lattice. A buffer of values that IS a Manifold.
///
/// `index`: `F a -> Rep -> a` -- read by coordinate.
/// This closes the representable functor isomorphism:
/// `index(collapse(f)) = f` (up to discretization).
#[derive(Clone, Debug)]
pub struct DiscreteManifold {
    /// Raw value buffer, row-major (y * width + x). `Arc<[f32]>`, not
    /// `Vec<f32>`: the buffer is write-once (no mutable accessor exists), so
    /// every clone of a `DiscreteManifold` — and every [`Self::kernel`] built
    /// from it — shares this one allocation. That is load-bearing for
    /// `Kernel::with_buffer_data`'s merge, not only an optimization: two
    /// kernels that read this manifold twice (a repeated glyph in a run, a
    /// texture sampled from two places) must carry the *same* `Arc` pointer
    /// under their shared `BufferIdentity`, or the merge's `Arc::ptr_eq`
    /// check — the one it can afford, at every composition — reads two
    /// honest tabulations as if they had drifted apart.
    pub(crate) buffer: Arc<[f32]>,
    /// Width of the grid (X dimension).
    pub(crate) width: usize,
    /// Height of the grid (Y dimension).
    pub(crate) height: usize,
    /// Which memory this is, for merging composed arenas. Clones share it,
    /// which is sound because the buffer is write-once — there is no mutable
    /// accessor, so a clone can never diverge from its original.
    pub(crate) id: pixelflow_ir::arena::BufferIdentity,
}

impl DiscreteManifold {
    /// Create a DiscreteManifold from a pre-filled buffer.
    ///
    /// # Panics
    ///
    /// Panics if `buffer.len() != width * height`.
    #[must_use]
    pub fn new(buffer: Vec<f32>, width: usize, height: usize) -> Self {
        assert_eq!(
            buffer.len(),
            width * height,
            "DiscreteManifold buffer size {} does not match dimensions {}x{} = {}",
            buffer.len(),
            width,
            height,
            width * height,
        );
        Self {
            buffer: Arc::from(buffer),
            width,
            height,
            id: pixelflow_ir::arena::BufferIdentity::mint(),
        }
    }

    /// Width of the grid (X extent).
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Height of the grid (Y extent).
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Read-only access to the underlying buffer (row-major).
    #[must_use]
    pub fn buffer(&self) -> &[f32] {
        &self.buffer
    }

    /// Consume the DiscreteManifold and return the buffer, copied out of the
    /// shared `Arc` — [`Self::kernel`] may hold other references to it, and
    /// an unsized `Arc<[f32]>` has no in-place way to reclaim its allocation
    /// even when this is the last one.
    #[must_use]
    pub fn into_buffer(self) -> Vec<f32> {
        self.buffer.to_vec()
    }
}

// ============================================================================
// Lattice: a finite box domain over the two coordinate axes
// ============================================================================

/// A finite box domain over the two coordinate axes (X, Y): the indices
/// `[0, extent[0]) × [0, extent[1])` and nothing else.
///
/// `extent[i]` is the number of samples along axis `i`; an axis with extent 1
/// is a fixed point, extent > 1 a loop dimension. Iteration is row-major with
/// X innermost (SIMD lanes ride X).
///
/// The shape is data, not a type: a frame, a tensor index range, and a sub-
/// lattice ([`IndexRange`](crate::IndexRange)) are all the same `Lattice`
/// shape with different extents. The JIT specializes on the extents at
/// kernel-compile time.
///
/// A `Lattice` is the index, and nothing else — it carries no coordinate
/// frame. It used to: an `origin` field let two lattices of the same extent
/// disagree about where their samples fell, which broke the representable
/// functor's law (`index(collapse(f)) = f`) on paper even though the emitted
/// code already treated the origin as a runtime argument. Where an origin
/// used to go, a contramap does instead — pixel-center sampling is
/// `f ∘ (+½)`, a placed glyph is `f ∘ (−pos)` — composed on the *kernel*
/// with [`Kernel::at`](pixelflow_ir::Kernel::at) before it ever reaches a
/// lattice (docs/plans/2026-09-06-lattice-is-the-index.md). There were also
/// four axes; Z and W left the same way, one stage earlier, because an axis
/// that never varies is a [`Uniform`](pixelflow_ir::Uniform), not an axis.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Lattice {
    /// Samples per axis: `[x, y]`. Extent 1 = fixed axis.
    pub extent: [u32; AXES],
}

/// Coordinate axes a lattice has.
pub const AXES: usize = pixelflow_ir::arena::COORD_AXES;

impl Lattice {
    /// A 2D pixel frame: X varies per pixel, Y per scanline.
    #[must_use]
    pub fn frame(width: usize, height: usize) -> Self {
        Self {
            extent: [width as u32, height as u32],
        }
    }

    /// A 1D index range `[0, len)` over X. Feature indices, not pixels.
    #[must_use]
    pub fn index(len: usize) -> Self {
        Self {
            extent: [len as u32, 1],
        }
    }

    /// A 2D index range `[0, width) x [0, height)` over X and Y.
    /// Weight-matrix indices: X = input dim, Y = output dim.
    #[must_use]
    pub fn index2(width: usize, height: usize) -> Self {
        Self {
            extent: [width as u32, height as u32],
        }
    }

    /// Collapse one row of `kernel`: X loops across `width`, Y is fixed at
    /// row `y0` — the row-range special case of an
    /// [`IndexRange`](crate::IndexRange), compiled at exactly that range's
    /// own extent (`[width, 1]`) rather than a taller frame's, since nothing
    /// above `y0` is ever asked about.
    ///
    /// Not a `Lattice` constructor: a fixed row is a coordinate, and a
    /// coordinate has nowhere to live on a domain that is only an index. A
    /// caller that wants an arbitrary (possibly fractional) fixed Y wants a
    /// contramap instead — `kernel.at(&Kernel::x(), &Kernel::constant(y))`
    /// baked with [`Lattice::frame`], the same move `eval_at` makes for a
    /// point.
    ///
    /// # Panics
    ///
    /// Whatever [`IndexRange::bake`](crate::IndexRange::bake) panics for.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[must_use]
    pub fn scanline(kernel: &pixelflow_ir::Kernel, width: usize, y0: usize) -> DiscreteManifold {
        if width == 0 {
            return DiscreteManifold::new(Vec::new(), 0, 1);
        }
        union::IndexRange::new(0, y0, width, 1).bake(kernel)
    }

    /// Evaluate `kernel` at one coordinate: the degenerate case with both
    /// axes fixed. Not a domain with a one-point extent and a weird origin —
    /// a different operation, answering a single value rather than
    /// tabulating a buffer, compiled once at `[1, 1]` and shared by the
    /// global compile cache across every coordinate this kernel is evaluated
    /// at (the coordinate is a collapse-time argument, never part of the
    /// compiled shape).
    ///
    /// # Panics
    ///
    /// Panics if `kernel` **declares a buffer it carries no data for** (see
    /// [`Lattice::bake`]'s panics for the distinction) — compile and bind it
    /// yourself, then call
    /// [`BoundManifold::eval_at`](crate::BoundManifold::eval_at). Also
    /// panics if this build's `Field` width is not the JIT's, or if
    /// compilation fails.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[must_use]
    pub fn eval_at(kernel: &pixelflow_ir::Kernel, x: f32, y: f32) -> f32 {
        manifold::Manifold::compile(kernel, [1, 1])
            .bind(&[])
            .eval_at(x, y)
    }

    // ───────────────────── domain queries ──────────────────────

    /// Number of points in this domain.
    #[must_use]
    pub fn len(&self) -> usize {
        self.extent.iter().map(|&e| e as usize).product()
    }

    /// Whether the domain is empty (any axis has extent 0).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bitmask of loop axes (extent > 1): bit 0 = X, 1 = Y.
    /// Fixed axes are constants from the kernel's point of view.
    #[must_use]
    pub fn loop_mask(&self) -> u8 {
        let mut mask = 0u8;
        for (i, &e) in self.extent.iter().enumerate() {
            if e > 1 {
                mask |= 1 << i;
            }
        }
        mask
    }

    /// Map a linear index to concrete coordinate values (X fastest).
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    #[must_use]
    pub fn coord(&self, index: usize) -> (f32, f32) {
        assert!(
            index < self.len(),
            "Lattice::coord index {} out of bounds (len = {})",
            index,
            self.len(),
        );
        let [ex, _] = self.extent.map(|e| e as usize);
        let x = index % ex;
        let y = index / ex;
        (x as f32, y as f32)
    }

    // ─────────────────────── collapse ────────────────────────

    /// **Tabulate a bound manifold over this domain**: the one verb, and the
    /// only way a compiled kernel becomes a buffer of numbers.
    ///
    /// The manifold must have been compiled at this lattice's extents
    /// ([`Manifold::compile`](crate::Manifold::compile)) and had every buffer
    /// slot it declared bound ([`Manifold::bind`](crate::Manifold::bind)).
    /// Samples are taken at the index itself on every axis — a plain
    /// coordinate frame is nowhere to be found.
    ///
    /// The X/Y loop nest lives *inside* the emitted code, so the whole domain
    /// is one call rather than one `extern "C"` call per row or SIMD batch.
    /// The result is a [`DiscreteManifold`] of `extent[0]` columns and
    /// `extent[1]` rows — the buffer that IS a manifold, which closes
    /// `index(collapse(f)) = f`.
    ///
    /// An empty domain collapses to an empty buffer without calling anything.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[must_use]
    pub fn collapse(&self, manifold: &manifold::BoundManifold) -> DiscreteManifold {
        let [ex, ey] = self.extent.map(|e| e as usize);
        let mut buffer = vec![0.0f32; self.len()];
        if !self.is_empty() {
            let band = manifold::PlaneRegion::at_index(ex, ey, 0, 0);
            manifold.collapse_rows(band, &mut buffer, ex);
        }
        DiscreteManifold::new(buffer, ex, ey)
    }

    /// Compile a [`Kernel`](pixelflow_ir::Kernel) at this lattice's shape, bind
    /// nothing, and collapse it: `collapse(compile(k).bind(&[]))` in one line,
    /// for the buffer-free case that is most of the tree. Every uniform
    /// argument the kernel has is bound at its default; bit for bit the same
    /// bake as a block holding those defaults.
    ///
    /// Compilation goes through the global cache, and the compile entry runs
    /// the optimizer itself, so a runtime-composed kernel
    /// (`Kernel::over`/`.at()`/arithmetic) — which never sees the
    /// `kernel!` macro's e-graph saturation — still reaches the
    /// backend optimized, with no caller having to remember to ask. `Dwrt`
    /// derivatives are resolved during codegen.
    ///
    /// # Panics
    ///
    /// Panics if the kernel **declares a buffer it carries no data for**:
    /// binding nothing (beyond what the kernel itself carried into
    /// [`Manifold::compile`](crate::Manifold::compile)) leaves that slot
    /// empty, and [`Manifold::bind`](crate::Manifold::bind) refuses it by
    /// name rather than letting the gathers load a base pointer out of an
    /// unbound context. A kernel built over bound memory
    /// (`DiscreteManifold::kernel`, `BilinearSampler::kernel`) carries its
    /// own data and bakes here just fine; one built from
    /// `DiscreteManifold::kernel_for` alone (a shape with no data seeded
    /// yet) does not — compile such a kernel yourself and bind its memory.
    /// Also panics if this build's `Field` width is not the JIT's, or if
    /// compilation fails.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[must_use]
    pub fn bake(&self, kernel: &pixelflow_ir::Kernel) -> DiscreteManifold {
        if self.is_empty() {
            // No samples, so no code: there is no lattice for the JIT to
            // specialize to, and `Manifold::compile` refuses the degenerate
            // extent rather than emitting a loop that runs zero times.
            let [ex, ey] = self.extent.map(|e| e as usize);
            return DiscreteManifold::new(Vec::new(), ex, ey);
        }
        self.collapse(&manifold::Manifold::compile(kernel, self.extent).bind(&[]))
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod cell_grid;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod manifold;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod union;

#[cfg(test)]
mod tests;

// ============================================================================
// BilinearSampler: smooth read-back of a collapsed lattice
// ============================================================================

/// Bilinear read-back of a [`DiscreteManifold`]: the smooth companion to its
/// nearest-neighbour [`DiscreteManifold::kernel`].
///
/// Where the discrete manifold's gather snaps a continuous coordinate to the
/// containing lattice cell, the sampler reads the four surrounding
/// integer-grid texels and blends them with the fractional coordinate
/// weights. The blend is a bound-memory kernel — four `Gather`s plus the
/// weight arithmetic in one arena — that a caller composes into a larger
/// kernel ([`Self::kernel`]) and compiles at the shape it will collapse over.
///
/// # Coordinate convention
///
/// Integer-grid space: a query at `(i + fx, j + fy)` with `fx, fy ∈ [0, 1)`
/// blends the texels at `(i, j)`, `(i+1, j)`, `(i, j+1)`, `(i+1, j+1)`.
/// Consequences:
///
/// - At exact integer coordinates the stored texel is returned untouched
///   (the fractional weights are zero).
/// - A buffer holding samples of a function affine in x and y is reproduced
///   exactly everywhere.
/// - Out-of-range taps clamp to the edge texel, exactly as
///   `DiscreteManifold::eval` does — `Gather`'s reference semantics.
/// - No half-pixel convention is baked in. Callers that store samples at
///   texel *centers* must shift coordinates by −0.5 before sampling.
///
/// Z and W pass through unchanged; interpolation is 2D over X/Y only.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[derive(Clone)]
pub struct BilinearSampler {
    tex: DiscreteManifold,
}

/// The 4-tap bilinear blend over one declared buffer, as an arena fragment:
/// `Σ tap(x?, y?) · weight` with `x0 = floor(X)`, `fx = X − x0`, and the
/// mirrored pair in y. Gather clamps each tap to the buffer edge.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn bilinear_arena(
    id: pixelflow_ir::arena::BufferIdentity,
    width: u32,
    height: u32,
) -> (pixelflow_ir::ExprArena, pixelflow_ir::ExprId) {
    use pixelflow_ir::arena::BufferDecl;
    use pixelflow_ir::{ExprArena, OpKind};

    let mut a = ExprArena::new();
    let buf = a.declare_buffer(BufferDecl { id, width, height });
    let x = a.push_var(0);
    let y = a.push_var(1);
    let one = a.push_const(1.0);

    let x0 = a.push_unary(OpKind::Floor, x);
    let y0 = a.push_unary(OpKind::Floor, y);
    let x1 = a.push_binary(OpKind::Add, x0, one);
    let y1 = a.push_binary(OpKind::Add, y0, one);
    let fx = a.push_binary(OpKind::Sub, x, x0);
    let fy = a.push_binary(OpKind::Sub, y, y0);
    let gx = a.push_binary(OpKind::Sub, one, fx);
    let gy = a.push_binary(OpKind::Sub, one, fy);

    let c00 = a.push_gather(buf, x0, y0);
    let c10 = a.push_gather(buf, x1, y0);
    let c01 = a.push_gather(buf, x0, y1);
    let c11 = a.push_gather(buf, x1, y1);

    let w00 = a.push_binary(OpKind::Mul, gx, gy);
    let w10 = a.push_binary(OpKind::Mul, fx, gy);
    let w01 = a.push_binary(OpKind::Mul, gx, fy);
    let w11 = a.push_binary(OpKind::Mul, fx, fy);

    let t00 = a.push_binary(OpKind::Mul, c00, w00);
    let t10 = a.push_binary(OpKind::Mul, c10, w10);
    let t01 = a.push_binary(OpKind::Mul, c01, w01);
    let t11 = a.push_binary(OpKind::Mul, c11, w11);

    let s0 = a.push_binary(OpKind::Add, t00, t10);
    let s1 = a.push_binary(OpKind::Add, s0, t01);
    let root = a.push_binary(OpKind::Add, s1, t11);
    (a, root)
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl BilinearSampler {
    /// The buffer this sampler reads (row-major f32).
    #[must_use]
    pub fn texture(&self) -> &DiscreteManifold {
        &self.tex
    }

    /// The 4-tap blend over a buffer of these extents, as a composable
    /// fragment — read it at *computed* coordinates inside a larger kernel
    /// with `.at(&u, &v)` instead of only at the caller's own.
    ///
    /// Takes extents rather than a sampler because extents are all the IR
    /// carries; the data binds later. So a program can compose against a
    /// buffer shape it has not filled yet, which is what lets a grid compile
    /// from its geometry alone. `bilinear` compiles from this same builder,
    /// so the composed and called forms cannot drift apart.
    #[must_use]
    pub fn kernel_for(
        id: pixelflow_ir::arena::BufferIdentity,
        width: u32,
        height: u32,
    ) -> pixelflow_ir::Kernel {
        // Same guard as `DiscreteManifold::kernel_for`, for the same reason:
        // `BindingTable::bind` accepts an empty slice against an empty
        // declaration, and gather lowering's `saturating_sub(1)` then clamps
        // every tap to index 0 — so an empty extent reaches the JIT and
        // dereferences a zero-length buffer instead of failing here.
        assert!(
            width > 0 && height > 0,
            "BilinearSampler::kernel_for: empty buffer ({width}x{height})"
        );
        let (arena, root) = bilinear_arena(id, width, height);
        pixelflow_ir::Kernel::from_parts(arena, root)
    }

    /// This sampler's blend as a composable fragment, its own texture
    /// seeded into the result ([`Kernel::with_buffer_data`]) — the data
    /// travels with the value from here, so a caller composing this into a
    /// larger kernel never gathers a binding for it separately. A refcount
    /// clone of [`DiscreteManifold`]'s own `Arc`, not a copy, and the same
    /// pointer every call — which is what lets two reads of one texture
    /// merge by `Arc::ptr_eq` instead of by comparing contents.
    ///
    /// # Panics
    ///
    /// Panics when an extent exceeds `u32`.
    #[must_use]
    pub fn kernel(&self) -> pixelflow_ir::Kernel {
        Self::kernel_for(
            self.tex.id,
            u32::try_from(self.tex.width).expect("buffer width exceeds u32"),
            u32::try_from(self.tex.height).expect("buffer height exceeds u32"),
        )
        .with_buffer_data(self.tex.id, Arc::clone(&self.tex.buffer))
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl DiscreteManifold {
    /// A nearest-neighbour read of a buffer of these extents, as a composable
    /// fragment — `.at(&idx, &row)` reads at *computed* indices.
    ///
    /// One `Gather`, which already carries [`Self::eval`]'s semantics: floor,
    /// clamp to the declared extents, index row-major. Takes extents rather
    /// than `&self` for the same reason as [`BilinearSampler::kernel_for`] —
    /// extents are what the IR holds, the data binds later.
    ///
    /// # Panics
    ///
    /// Panics on a zero extent, which would make every gather clamp onto an
    /// empty buffer.
    #[must_use]
    pub fn kernel_for(
        id: pixelflow_ir::arena::BufferIdentity,
        width: u32,
        height: u32,
    ) -> pixelflow_ir::Kernel {
        assert!(
            width > 0 && height > 0,
            "DiscreteManifold::kernel_for: empty buffer ({width}x{height})"
        );
        let mut a = pixelflow_ir::ExprArena::new();
        let buf = a.declare_buffer(pixelflow_ir::arena::BufferDecl { id, width, height });
        let (x, y) = (a.push_var(0), a.push_var(1));
        let root = a.push_gather(buf, x, y);
        pixelflow_ir::Kernel::from_parts(a, root)
    }

    /// This buffer paired with the identity [`Self::kernel`] declared, ready
    /// for [`Manifold::bind`](crate::Manifold::bind).
    ///
    /// A kernel over a buffer names its memory by identity and nothing else,
    /// so the two halves have to travel together to reach a collapse; this is
    /// the other half. The buffer is copied into the `Arc` a bound manifold
    /// holds — once per bind, not once per band — because a collapsed lattice
    /// owns its samples outright.
    #[must_use]
    pub fn binding(
        &self,
    ) -> (
        pixelflow_ir::arena::BufferIdentity,
        alloc::sync::Arc<Vec<f32>>,
    ) {
        (self.id, alloc::sync::Arc::new(self.buffer.to_vec()))
    }

    /// This buffer's nearest-neighbour read as a composable fragment, this
    /// buffer's own contents seeded into the result
    /// ([`Kernel::with_buffer_data`]) — the data travels with the value
    /// from here, so a caller composing this into a larger kernel never
    /// gathers a binding for it separately (see [`Self::kernel_for`] for
    /// the shape-only fragment this builds on). A refcount clone of this
    /// manifold's own `Arc`, not a copy, and the same pointer every call —
    /// which is what lets two reads of one buffer (a repeated glyph in a
    /// run) merge by `Arc::ptr_eq` instead of by comparing contents.
    ///
    /// # Panics
    ///
    /// Panics on an empty buffer, or when an extent exceeds `u32`.
    #[must_use]
    pub fn kernel(&self) -> pixelflow_ir::Kernel {
        Self::kernel_for(
            self.id,
            u32::try_from(self.width).expect("buffer width exceeds u32"),
            u32::try_from(self.height).expect("buffer height exceeds u32"),
        )
        .with_buffer_data(self.id, Arc::clone(&self.buffer))
    }

    /// Wrap this buffer in a [`BilinearSampler`].
    ///
    /// Nothing is compiled here. The sampler carries the buffer and the shape
    /// of its 4-tap blend; a consumer composes [`BilinearSampler::kernel`]
    /// into the kernel it is building and compiles that once, at the shape it
    /// will collapse over. (This used to JIT the blend at a point shape,
    /// because the sampler was read one SIMD batch at a time through a
    /// per-batch `eval` — a compile per glyph for a call that no longer
    /// exists.)
    ///
    /// # Panics
    ///
    /// Panics on an empty buffer, or when an extent exceeds `u32`.
    #[must_use]
    pub fn bilinear(self) -> BilinearSampler {
        assert!(
            !self.buffer.is_empty(),
            "DiscreteManifold::bilinear on an empty buffer ({}x{})",
            self.width,
            self.height,
        );
        let _ = u32::try_from(self.width).expect("buffer width exceeds u32");
        let _ = u32::try_from(self.height).expect("buffer height exceeds u32");
        BilinearSampler { tex: self }
    }
}
