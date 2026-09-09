//! # A manifold is a kernel compiled at a lattice's shape
//!
//! The middle object of `kernel ──compile(shape)──▶ manifold
//! ──bind(buffers)──▶ bound ──collapse(region)──▶ buffer`. A [`Kernel`] is the
//! description; compiling it at a lattice's extents gives a **manifold**
//! ([`Manifold`]); binding the memory it declared gives a [`BoundManifold`];
//! collapsing that over a band of rows gives numbers. The loop over batches
//! and rows lives inside the emitted code, with the two-level LICM prologues
//! (per call, per row) active, so one collapse call covers a whole band, and
//! whatever memory the kernel declared is bound by identity once and stays
//! bound for every band collapsed from it.
//!
//! ## Rank
//!
//! A manifold is compiled at a [`Lattice`](crate::Lattice)'s whole `[x, y]`
//! extent, which is exactly what a
//! [`LatticeShape`](pixelflow_ir::LatticeShape) is: the extents decide which
//! axes are binders, and therefore what the emitter may hoist out of what. The
//! *collapse ABI* is two-dimensional — one call fills batches across X and
//! rows down Y — and so is the domain, so a lattice is one call. A scalar the
//! kernel needs but the lattice does not vary is an argument
//! ([`UniformBlock`]), not a third axis of extent 1.
//!
//! This is the shape every frame path in the tree already had — the cell
//! grid's four channel programs and its packed sibling were two copies of it
//! — with nothing above it: no colour, no channels, no pixel format. Those
//! live a layer up, in `pixelflow-graphics`, which composes a packed-pixel
//! kernel and compiles it here.
//!
//! ## Output planes
//!
//! A band is written straight into the caller's plane, whose rows are however
//! many elements apart the caller says: the collapse ABI stores whole SIMD
//! batches and steps the output pointer by the leftover bytes between rows,
//! so a destination at any stride is filled in place — no staging plane, no
//! per-row copy. The one thing a batch store cannot do is a row's final
//! *partial* batch when the stride has no room for its overhang; that batch
//! alone goes through a one-batch scratch, and it is the only place in this
//! module where the SIMD width is visible at all.
//!
//! The store is a raw vector store, type-blind bit movement, so a kernel
//! whose root is int-domain (a packed pixel, a mask) collapses through
//! [`BoundManifold::collapse_int_rows`] into a `u32` plane exactly: no float
//! operation touches the value between the root and memory.

use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::Field;
use pixelflow_codegen::CompiledKernel;
use pixelflow_ir::arena::{BufferDecl, BufferIdentity, UniformDecl, UniformIdentity};
use pixelflow_ir::{Kernel, Uniform};

/// Buffer slots a [`BoundManifold`] can bind without allocating: binding builds
/// its base-pointer array on the stack, so the bound is what makes that array
/// a fixed size. A kernel declaring more is refused at compile time rather
/// than silently overflowing it.
pub const MAX_BOUND_BUFFERS: usize = 4;

/// Entries in the context a collapse call hands the kernel: one base pointer
/// per buffer slot, then the uniform block's, which sits in the entry after
/// the kernel's last buffer and exists only when the kernel has an argument.
const CONTEXT_ENTRIES: usize = MAX_BOUND_BUFFERS + 1;

/// The values of a compiled kernel's arguments, laid out as its code reads
/// them: one `f32` per [`Uniform`] the kernel declares, at the offset the
/// link step assigned.
///
/// A block is an *argument of the collapse*, not state on the program: it is
/// built from a [`Manifold`] with every argument at its default
/// ([`Manifold::block`]), written through the handles the scene kept
/// ([`UniformBlock::set`]), and handed to
/// [`BoundManifold::with_uniforms`] — stripes on separate threads all read
/// one block immutably, and nothing is ambient. Setting a value touches no
/// arena, runs no saturation, and compiles nothing.
///
/// The values are shared with every bound manifold that took them, so
/// handing a block to a frame is a refcount and never an allocation.
/// [`UniformBlock::set`] writes in place while the block is the sole
/// holder, and copies the values first when a frame still holds the
/// previous ones — so a consumer that drops last frame's bound manifold
/// before setting next frame's values allocates nothing per frame.
#[derive(Clone, Debug)]
pub struct UniformBlock {
    values: Arc<Vec<f32>>,
    /// The link this block is laid out against, shared with the manifold
    /// that made it; a bound manifold checks it is the very same table.
    link: Arc<[UniformDecl]>,
}

/// A handle that is not one of the program's arguments.
///
/// An error rather than a silent no-op, because the pixels would be
/// plausible: a cursor that never moves looks like a cursor that has not
/// moved yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnknownUniform(pub UniformIdentity);

impl core::fmt::Display for UnknownUniform {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?} is not an argument of this compiled kernel", self.0)
    }
}

impl core::error::Error for UnknownUniform {}

impl UniformBlock {
    fn offset(&self, u: Uniform) -> Result<usize, UnknownUniform> {
        self.link
            .iter()
            .position(|d| d.id == u.identity())
            .ok_or(UnknownUniform(u.identity()))
    }

    /// Bind `v` to the argument `u` names.
    ///
    /// # Errors
    ///
    /// [`UnknownUniform`] when `u` is not one of this program's arguments —
    /// a composition mistake, never ignored.
    pub fn set(&mut self, u: Uniform, v: f32) -> Result<(), UnknownUniform> {
        let i = self.offset(u)?;
        Arc::make_mut(&mut self.values)[i] = v;
        Ok(())
    }

    /// The value currently bound to the argument `u` names.
    ///
    /// # Errors
    ///
    /// [`UnknownUniform`] when `u` is not one of this program's arguments.
    pub fn get(&self, u: Uniform) -> Result<f32, UnknownUniform> {
        self.offset(u).map(|i| self.values[i])
    }

    /// The values in link order — what the kernel reads. An `&[f32]` in
    /// *this* order is not what the oracle takes; see [`Self::entries`].
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Every argument with its value, by identity — the order-free form,
    /// and the one to hand `BindingTable::bind_uniforms` so the oracle and
    /// the kernel read the same block.
    pub fn entries(&self) -> impl Iterator<Item = (UniformIdentity, f32)> + '_ {
        self.link
            .iter()
            .map(|d| d.id)
            .zip(self.values.iter().copied())
    }
}

/// A horizontal band to collapse: `width` samples across, `rows` rows down,
/// starting at some coordinate of the kernel's two-dimensional domain.
///
/// A frame's timestamp used to enter a compiled kernel through here, as the
/// `(z, w)` plane the band lay in. It is a [`UniformBlock`] value now: the
/// program is still compiled once and still collapsed with a new time each
/// frame, but the time is an argument of the call rather than a coordinate no
/// sample varies along.
///
/// The band's own **origin** — the coordinate its first sample is taken at —
/// is what separates the conventions in the tree, and every one of them is
/// an *index*, never an arbitrary coordinate: a pixel band samples *centers*
/// (`i + ½`, `j + ½`), which is what [`PlaneRegion::rows`] builds; a
/// [`Lattice`](crate::Lattice) or an
/// [`IndexRange`](crate::IndexRange) samples the raw index, which is what
/// `PlaneRegion::at_index` builds. A single real-valued point
/// ([`Lattice::eval_at`](crate::Lattice::eval_at)) is the one place an
/// arbitrary coordinate legitimately appears, because there both axes are
/// fixed and there is no index to be one of. All three are the same band
/// mechanism with a different starting coordinate, which is why there is one
/// collapse and not three.
#[derive(Clone, Copy, Debug)]
pub struct PlaneRegion {
    /// Samples per row.
    pub width: usize,
    /// Number of rows.
    pub rows: usize,
    /// The coordinate the first sample is taken at: lane 0 of the first row.
    /// Private so a band is always somewhere — no axis can be set alone or
    /// left undefined — and because which of the sampling conventions
    /// applies is the constructor's to decide, not the caller's to patch
    /// afterwards.
    origin: [f32; crate::lattice::AXES],
}

impl PlaneRegion {
    /// A band of pixel rows `y0 .. y0 + rows` on the origin plane, sampled at
    /// pixel centers: sample `(i, j)` is taken at `(i + ½, j + ½)`.
    #[must_use]
    pub fn rows(width: usize, y0: usize, rows: usize) -> Self {
        Self {
            width,
            rows,
            origin: [SAMPLE_CENTER, y0 as f32 + SAMPLE_CENTER],
        }
    }

    /// A band of `rows` rows whose first sample is at raw index `(x0, y0)`,
    /// with each subsequent sample one unit further along X and each row one
    /// unit further along Y — no coordinate frame, just the index itself.
    /// What [`Lattice::collapse`](crate::Lattice::collapse) and a
    /// [`Union`](crate::Union) summand sample.
    pub(crate) fn at_index(width: usize, rows: usize, x0: usize, y0: usize) -> Self {
        Self {
            width,
            rows,
            origin: [x0 as f32, y0 as f32],
        }
    }

    /// A single sample at the literal coordinate `(x, y)` — not an index and
    /// not a pixel center, the one case where an arbitrary real-valued
    /// coordinate is legitimate: both axes are fixed, so there is no index
    /// left to restrict. What [`Lattice::eval_at`](crate::Lattice::eval_at)
    /// samples.
    pub(crate) fn at_point(x: f32, y: f32) -> Self {
        Self {
            width: 1,
            rows: 1,
            origin: [x, y],
        }
    }
}

/// Half a sample: the offset from a pixel's integer index to its center, which
/// is where a rasterizer samples.
const SAMPLE_CENTER: f32 = 0.5;

/// **A manifold is a [`Kernel`] compiled at a lattice's shape** — the thing
/// you can sample over a domain, and the only compiled object a consumer
/// names.
///
/// Compile once per shape; bind memory per frame ([`Manifold::bind`]). Its
/// size and compile time are independent of the lattice's.
///
/// It has no `eval`: the way a manifold becomes numbers is to bind its memory
/// and collapse it over a region — [`BoundManifold::collapse_rows`] for a band
/// in place, [`Lattice::collapse`](crate::Lattice::collapse) for a whole
/// domain into a buffer.
///
/// **This is also the only compiled object that can bind memory.** A kernel
/// composed over a buffer —
/// [`DiscreteManifold::kernel_for`](crate::DiscreteManifold::kernel_for)'s
/// gather, [`BilinearSampler::kernel_for`](crate::BilinearSampler::kernel_for)'s
/// 4-tap blend, or anything built on them — declares buffer slots, and every
/// slot must be bound before a collapse: the gathers load their base pointers
/// out of the bound context.
///
/// The colour-shaped things in the tree are all this object wearing different
/// numbers of channels: [`Lattice::bake`](crate::Lattice::bake) is the
/// one-channel, buffer-free instance; a field over bound memory is the
/// one-channel instance with its slots filled; and `pixelflow-graphics`'s
/// packed manifold is four channel kernels compiled through here with an
/// integer pack at the root. Not three paths — one, sampled three ways.
pub struct Manifold {
    jit: Arc<CompiledKernel>,
    /// The lattice shape the kernel was specialized to, `[x, y]`. Every band
    /// collapsed through it lies within these extents.
    extent: [u32; crate::lattice::AXES],
    /// The memory the kernel declared, in the slot order its ABI binds.
    /// Shared so a [`BoundManifold`] stays cheap to clone.
    slots: Arc<[BufferDecl]>,
    /// The kernel's arguments, in the offset order its block is read in —
    /// the link. Shared with every block and bound manifold made from here,
    /// so "the same link" is pointer equality.
    link: Arc<[UniformDecl]>,
    /// Every argument at its default, built once here so that `bind` — a
    /// per-frame call, four times a frame on the terminal path — is a
    /// refcount and never an allocation, uniforms or none.
    defaults: Arc<Vec<f32>>,
}

impl Manifold {
    /// JIT-compile `kernel` in collapse mode at a lattice of these extents.
    ///
    /// `extent` is a [`Lattice`](crate::Lattice)'s own `[x, y]` — the shape,
    /// and nothing else, because the shape is what specializes the code and
    /// where a collapse starts is a property of the collapse
    /// ([`PlaneRegion`]), not of the manifold. A frame is `[w, h]`.
    ///
    /// # Panics
    ///
    /// Panics on a degenerate extent, when this build's `Field` width does not
    /// match the JIT's emitted width, when the kernel declares more buffers
    /// than [`MAX_BOUND_BUFFERS`] or one too large to index exactly in `f32`,
    /// or if compilation fails.
    #[must_use]
    pub fn compile(kernel: &Kernel, extent: [u32; crate::lattice::AXES]) -> Self {
        assert!(
            extent.iter().all(|&e| e > 0),
            "Manifold::compile: degenerate extent {extent:?}"
        );
        assert_eq!(
            core::mem::size_of::<Field>(),
            pixelflow_codegen::JIT_VECTOR_BYTES,
            "Manifold::compile: Field width does not match the JIT's emitted width"
        );
        for decl in kernel.buffers() {
            assert!(
                buffer_len(decl) <= EXACT_F32_INDEX,
                "Manifold::compile: buffer of {} elements exceeds the \
                 exactly f32-indexable range (2^24); gathers would alias \
                 adjacent samples",
                buffer_len(decl)
            );
        }
        // The cache keys on structure — buffers and uniforms by dense slot,
        // not identity — so two compositions of one shape share code, and
        // what comes back beside it is *this* composition's link: which
        // identity each slot binds, in the order the code was compiled
        // against. That order, not the arena's declaration order, is the
        // one `bind` fills the context in.
        let linked =
            pixelflow_codegen::jit_cache::compile(kernel, pixelflow_ir::LatticeShape::new(extent))
                .expect("Manifold: kernel failed to compile");
        assert!(
            linked.buffers.len() <= MAX_BOUND_BUFFERS,
            "Manifold::compile: kernel needs {} buffer slots, over the \
             {MAX_BOUND_BUFFERS} a frame can bind without allocating",
            linked.buffers.len()
        );
        let defaults: Vec<f32> = linked.uniforms.iter().map(|d| d.default).collect();
        Self {
            jit: linked.kernel,
            extent,
            slots: linked.buffers.into(),
            link: linked.uniforms.into(),
            defaults: Arc::new(defaults),
        }
    }

    /// The lattice extents this manifold was compiled for, `[x, y]`.
    #[must_use]
    pub fn extent(&self) -> [u32; crate::lattice::AXES] {
        self.extent
    }

    /// The memory this manifold's kernel declared, in slot order. A caller that
    /// minted the identities can say which slot is which without inferring it
    /// from extents.
    #[must_use]
    pub fn buffers(&self) -> &[BufferDecl] {
        &self.slots
    }

    /// The kernel's arguments, in the order the block holds them.
    #[must_use]
    pub fn uniforms(&self) -> &[UniformDecl] {
        &self.link
    }

    /// A block with every argument at its default, laid out per this
    /// manifold's link. Make one once, or once per frame; set what moved
    /// through the handles; hand it to [`BoundManifold::with_uniforms`].
    #[must_use]
    pub fn block(&self) -> UniformBlock {
        UniformBlock {
            values: Arc::clone(&self.defaults),
            link: Arc::clone(&self.link),
        }
    }

    /// The compiled kernel's emitted bytes (research/profiling harness).
    #[must_use]
    pub fn code_bytes(&self) -> &[u8] {
        self.jit.code_bytes()
    }

    /// Bind one frame's memory: each declared slot takes the buffer carrying
    /// its identity. `buffers` may be given in any order and may carry entries
    /// this kernel does not read. Buffers are `Arc`s so a frame in flight
    /// keeps its data alive while the caller prepares the next one.
    ///
    /// Every uniform argument is bound at its default; a frame that moves
    /// one applies a block with [`BoundManifold::with_uniforms`].
    ///
    /// # Panics
    ///
    /// Panics if a declared slot has no buffer bound to it, or if a buffer's
    /// length is not the `width × height` its declaration promised — the
    /// gathers address the declared shape, so a shorter buffer would be read
    /// past its end through an entirely safe API.
    #[must_use]
    pub fn bind(&self, buffers: &[(BufferIdentity, Arc<Vec<f32>>)]) -> BoundManifold {
        let mut bound: [Option<Arc<Vec<f32>>>; MAX_BOUND_BUFFERS] = Default::default();
        for (slot, decl) in bound.iter_mut().zip(self.slots.iter()) {
            let data = buffers
                .iter()
                .find(|(id, _)| *id == decl.id)
                .map(|(_, data)| data)
                .unwrap_or_else(|| panic!("Manifold::bind: nothing bound to slot {decl:?}"));
            assert_eq!(
                data.len(),
                buffer_len(decl),
                "Manifold::bind: buffer of {} floats bound to slot {decl:?}",
                data.len()
            );
            *slot = Some(Arc::clone(data));
        }
        BoundManifold {
            jit: Arc::clone(&self.jit),
            extent: self.extent,
            bound,
            buffer_slots: self.slots.len(),
            link: Arc::clone(&self.link),
            // A refcount, not an allocation: `bind` is per frame, and the
            // invariant is pinned by `tests/bind_allocates_nothing.rs`.
            uniforms: Arc::clone(&self.defaults),
        }
    }
}

/// Gathers compute a row-major linear index in `f32`, which is exact only
/// below 2^24 — beyond that adjacent samples alias.
const EXACT_F32_INDEX: usize = 1 << 24;

/// Elements a declaration promises.
///
/// # Panics
///
/// Panics if the product overflows `usize`. A wrapped length would be SMALL,
/// so `bind` would accept a correspondingly small buffer while the compiled
/// kernel still declared the true row width — and the gathers would read
/// billions of elements past the end through an entirely safe API.
fn buffer_len(decl: &BufferDecl) -> usize {
    (decl.width as usize)
        .checked_mul(decl.height as usize)
        .expect("Manifold: declared buffer length overflows usize")
}

/// A [`Manifold`] with its memory bound: the compiled code plus the buffers it
/// reads. Cheap to clone (one `Arc` for the code, one per bound buffer).
///
/// This is what a collapse takes. A kernel that reads nothing binds the empty
/// slice and is a bound manifold too — there is no second, buffer-free form.
#[derive(Clone)]
pub struct BoundManifold {
    jit: Arc<CompiledKernel>,
    extent: [u32; crate::lattice::AXES],
    /// Bound memory in slot order; entries past the declared slots stay
    /// `None` and are never addressed, because the kernel only reads slots it
    /// declared.
    bound: [Option<Arc<Vec<f32>>>; MAX_BOUND_BUFFERS],
    /// How many of `bound` the kernel declared — the context entry after
    /// them is the block's.
    buffer_slots: usize,
    /// The link the block below is laid out against.
    link: Arc<[UniformDecl]>,
    /// The argument values the kernel reads, in link order — the manifold's
    /// defaults or a block's values, shared rather than copied; empty when
    /// it has none, and then no block pointer is passed at all.
    uniforms: Arc<Vec<f32>>,
}

impl BoundManifold {
    /// The lattice extents the kernel was compiled for, `[x, y]`.
    #[must_use]
    pub fn extent(&self) -> [u32; crate::lattice::AXES] {
        self.extent
    }

    /// This bound memory with the kernel's arguments taken from `block` —
    /// the per-frame step for whatever moved. A refcount on the block's
    /// values, no copy and no allocation.
    ///
    /// # Panics
    ///
    /// Panics if `block` was made by a different program: its offsets would
    /// be another kernel's, and the values would land in the wrong arguments
    /// with entirely plausible pixels.
    #[must_use]
    pub fn with_uniforms(mut self, block: &UniformBlock) -> Self {
        assert!(
            Arc::ptr_eq(&block.link, &self.link),
            "BoundManifold::with_uniforms: the block was laid out for a different program"
        );
        self.uniforms = Arc::clone(&block.values);
        self
    }

    /// Evaluate this manifold at one literal coordinate — a single-sample
    /// collapse, and the mechanism behind
    /// [`Lattice::eval_at`](crate::Lattice::eval_at). Not a domain: a
    /// different operation from [`Self::collapse_rows`], answering one value
    /// instead of filling a buffer. Works at any compiled extent, since a
    /// `1×1` band lies inside every non-degenerate one — the common case is
    /// [`Lattice::eval_at`](crate::Lattice::eval_at), which compiles at
    /// `[1, 1]` so every coordinate this kernel is ever asked about shares
    /// one compiled program.
    #[must_use]
    pub fn eval_at(&self, x: f32, y: f32) -> f32 {
        let mut out = [0.0f32];
        self.collapse_rows(PlaneRegion::at_point(x, y), &mut out, 1);
        out[0]
    }

    /// Collapse the region into `out`, whose rows are `stride` elements apart
    /// and whose first `region.width` elements each row are the samples. Where
    /// the samples are taken is the region's — pixel centers for
    /// [`PlaneRegion::rows`].
    ///
    /// The destination is written in place — the collapse loop's own stores
    /// land in it — so `stride` is whatever the caller's plane already is: a
    /// frame's packed row width, a padded scratch, a sub-rectangle of
    /// something larger.
    ///
    /// # Panics
    ///
    /// Panics if the region's width is zero, `stride` is less than it, or
    /// `out` cannot hold the band.
    pub fn collapse_rows(&self, region: PlaneRegion, out: &mut [f32], stride: usize) {
        let band = self.plan(
            "collapse_rows",
            region,
            Destination::absorbing(stride, out.len()),
        );
        // SAFETY: see `collapse`. `out` is an `f32` plane, which is what the
        // collapse ABI writes, and `plan` proved it holds the band.
        unsafe { self.collapse(region, out.as_mut_ptr(), band) }
    }

    /// [`BoundManifold::collapse_rows`] for a kernel whose root is int-domain:
    /// each lane already holds a bit pattern (a packed pixel, a mask), and the
    /// collapse store is a raw vector store — type-blind bit movement — so
    /// writing through the ABI's `*mut f32` into a `u32` plane is exact.
    ///
    /// # Panics
    ///
    /// Panics if the region's width is zero, `stride` is less than it, or
    /// `out` cannot hold the band.
    pub fn collapse_int_rows(&self, region: PlaneRegion, out: &mut [u32], stride: usize) {
        let band = self.plan(
            "collapse_int_rows",
            region,
            Destination::absorbing(stride, out.len()),
        );
        // SAFETY: see `collapse`. `u32` and `f32` share size and alignment,
        // and the store moves the root's bit pattern without interpreting it.
        unsafe { self.collapse(region, out.as_mut_ptr().cast::<f32>(), band) }
    }

    /// Collapse the region into the `width × rows` sub-rectangle of `out`
    /// whose first sample is `out[0]` and whose rows are `stride` elements
    /// apart, writing **exactly** `region.width` samples per row and leaving
    /// every other element of `out` as it was.
    ///
    /// [`BoundManifold::collapse_rows`] deliberately lets a row's final
    /// partial batch overhang into the stride's spare columns, because for a
    /// frame those columns are padding nobody reads. For a summand of a
    /// [`Union`](crate::Union) they are the *neighbour's* columns, filled by a
    /// different program, so an overhang there is not scratch — it is wrong
    /// samples. That is the whole of the difference: same plan, same loop,
    /// same one-batch scratch for a row's tail, with [`RowTail::Exact`] where
    /// a frame passes [`RowTail::Absorbs`]. No staging plane, and so nothing
    /// to allocate.
    ///
    /// # Panics
    ///
    /// Panics if the region's width is zero, `stride` is less than it, or
    /// `out` cannot hold the sub-rectangle.
    pub(crate) fn collapse_subrect(&self, region: PlaneRegion, out: &mut [f32], stride: usize) {
        let band = self.plan(
            "collapse_subrect",
            region,
            Destination::exact(stride, out.len()),
        );
        // SAFETY: see `collapse`. `out` is an `f32` plane, which is what the
        // collapse ABI writes, and `plan` proved it holds the sub-rectangle —
        // which under `RowTail::Exact` reaches no further than `width` in any
        // row.
        unsafe { self.collapse(region, out.as_mut_ptr(), band) }
    }

    /// [`Self::collapse_subrect`] for a kernel whose root is int-domain — a
    /// packed pixel or a mask — the way [`Self::collapse_int_rows`] is for
    /// [`Self::collapse_rows`].
    ///
    /// Public because the caller that needs it is a *union summand* in
    /// another crate: a program that answers for part of a frame writes
    /// exactly its own columns, and the columns past them belong to whatever
    /// fills the rest. `collapse_int_rows`'s overhang policy — "the caller
    /// owns the padding" — is false there.
    ///
    /// # Panics
    ///
    /// Panics if the region's width is zero, `stride` is less than it, or
    /// `out` cannot hold the sub-rectangle.
    pub fn collapse_int_subrect(&self, region: PlaneRegion, out: &mut [u32], stride: usize) {
        let band = self.plan(
            "collapse_int_subrect",
            region,
            Destination::exact(stride, out.len()),
        );
        // SAFETY: see `collapse`. `u32` and `f32` share size and alignment,
        // the store moves the root's bit pattern without interpreting it, and
        // `plan` proved `out` holds the sub-rectangle — which under
        // `RowTail::Exact` reaches no further than `width` in any row.
        unsafe { self.collapse(region, out.as_mut_ptr().cast::<f32>(), band) }
    }

    /// How a band lands in a destination of a given stride, and the guard that
    /// the destination can hold it.
    ///
    /// # Panics
    ///
    /// Panics if the region's width is zero, `stride` is less than it, the
    /// region leaves the compiled extents, or `out_len` cannot hold the band.
    fn plan(&self, what: &str, region: PlaneRegion, dest: Destination) -> BandPlan {
        let Destination {
            stride,
            len: out_len,
            tail,
        } = dest;
        let (width, rows) = (region.width, region.rows);
        assert!(width > 0, "{what}: zero width");
        assert!(
            stride >= width,
            "{what}: stride {stride} is narrower than the {width} samples a row holds"
        );
        // The kernel was compiled for `extent`; a region outside it would run
        // the collapse loop past the lattice it was specialized to.
        //
        // `debug_assert`, matching `CompiledKernel::call_collapse`'s own check of
        // the same promise: today's emitted code takes its loop bounds from
        // the tile at run time, so a wider region is merely a stale cache key,
        // not wrong samples. It becomes load-bearing when the emitted code
        // specializes on the extents, and is promoted with that change rather
        // than ahead of it — a release panic for a promise nothing yet relies
        // on is a new way for a terminal to die.
        let (fw, fh) = (self.extent[0] as usize, self.extent[1] as usize);
        debug_assert!(
            width <= fw && rows <= fh,
            "{what}: a band of {width}×{rows} lies outside the {fw}×{fh} \
             lattice this manifold was compiled for"
        );
        let plan = BandPlan::new(width, rows, stride, tail);
        // Checked: the span wraps in release for a caller-supplied region
        // large enough, and a wrapped product would let an undersized `out`
        // pass this guard while the collapse call below still received the
        // real (enormous) row count and wrote past the slice. The documented
        // panic must fire before any unsafe call, not after.
        let needed = plan.span().expect("collapse: band span overflows usize");
        assert!(
            out_len >= needed,
            "{what}: plane of {out_len} elements cannot hold {rows} rows at stride {stride}"
        );
        plan
    }

    /// # Safety
    ///
    /// `out` must be writable for `band.span()` 4-byte elements — which
    /// [`BoundManifold::plan`] asserted for the slice it came from.
    unsafe fn collapse(&self, region: PlaneRegion, out: *mut f32, band: BandPlan) {
        if band.rows == 0 {
            return;
        }
        // One base pointer per declared slot, in slot order, then the block's
        // in the entry after them when the kernel has arguments.
        // Stack-allocated against the MAX_BOUND_BUFFERS bound `compile`
        // checked, so baking a band allocates nothing; trailing entries stay
        // null and are never read because the kernel only addresses slots it
        // declared.
        let mut ctx = [core::ptr::null::<f32>(); CONTEXT_ENTRIES];
        for (dst, src) in ctx.iter_mut().zip(self.bound.iter()) {
            if let Some(data) = src {
                *dst = data.as_ptr();
            }
        }
        if !self.uniforms.is_empty() {
            ctx[self.buffer_slots] = self.uniforms.as_ptr();
        }
        // Where the band's first sample lies; X advances by one per lane, Y by
        // one per row, which is what the collapse loop does.
        //
        // The ABI still carries four base coordinates. The last two are dead:
        // the emitter refuses an arena naming `Var(2)` or `Var(3)`, so no
        // emitted body reads them, and passing zero keeps every kernel's
        // bytes exactly what they were. Narrowing the ABI to two is not this
        // stage's either — L2 removed `Lattice::origin`, a type-level change
        // with no effect on the ABI's own shape — because it changes the
        // scaffold's own stores and loads and so every kernel's bytes, which
        // is exactly what an identity gate needs held still.
        let [x0, y0] = region.origin;
        let dead = Field::from(0.0);
        if band.groups > 0 {
            // SAFETY: `compile` checked size_of::<Field>() == JIT_VECTOR_BYTES
            // and that every declared slot fits `ctx`; `bind` bound a buffer of
            // the declared length to each of them and this frame holds those
            // `Arc`s alive for the duration of the call, as it does the block
            // the entry after them points into (one `f32` per argument, in
            // the link's order, which is the order the code was compiled
            // against); the caller's guard
            // proved `out` holds `rows` rows of `groups` whole batches at
            // `stride`, which is what `row_skip_bytes` steps between.
            unsafe {
                self.jit.call_collapse(
                    ctx.as_ptr(),
                    pixelflow_codegen::TileSlice::new(
                        out,
                        band.groups,
                        band.rows,
                        band.row_skip_bytes(),
                    ),
                    pixelflow_codegen::Point4::new(
                        Field::sequential(x0),
                        Field::from(y0),
                        dead,
                        dead,
                    ),
                );
            }
        }
        let Some(tail) = band.tail() else { return };
        // The row's last, partial batch. A whole-batch store there would run
        // past the row into the next one, so it lands in a scratch batch and
        // only the samples that belong to the row are copied back — one extra
        // call per row, and only when the stride left no room for the overhang.
        let mut scratch = [0.0f32; BATCH_LANES];
        for row in 0..band.rows {
            // SAFETY: as above; `scratch` is exactly the one batch
            // `TileSlice::single` writes.
            unsafe {
                self.jit.call_collapse(
                    ctx.as_ptr(),
                    pixelflow_codegen::TileSlice::single(scratch.as_mut_ptr()),
                    pixelflow_codegen::Point4::new(
                        Field::sequential(x0 + band.whole_lanes() as f32),
                        Field::from(y0 + row as f32),
                        dead,
                        dead,
                    ),
                );
            }
            // SAFETY: `plan` proved `out` holds `row * stride + width`
            // elements, and this writes the last `tail` of them.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    scratch.as_ptr(),
                    out.add(row * band.stride + band.whole_lanes()),
                    tail,
                );
            }
        }
    }
}

/// Lanes in one SIMD batch of the emitted code. The only place in this module
/// the width is visible, and it is here for one reason: a row's final partial
/// batch needs somewhere to land that is not the next row.
const BATCH_LANES: usize = pixelflow_codegen::JIT_VECTOR_BYTES / core::mem::size_of::<f32>();

/// The plane a band lands in: how far apart its rows are, how much of it
/// there is, and whether the columns past the band are the caller's.
///
/// One value because the three are one question — *where does this go* — and
/// answering it in three parameters is how a signature stops refusing the
/// wrong combination.
#[derive(Clone, Copy, Debug)]
struct Destination {
    /// Elements between the starts of two rows.
    stride: usize,
    /// Elements the destination holds.
    len: usize,
    /// Whose the columns past the band's width are.
    tail: RowTail,
}

impl Destination {
    /// A plane the caller owns: a final partial batch may overhang.
    fn absorbing(stride: usize, len: usize) -> Self {
        Self {
            stride,
            len,
            tail: RowTail::Absorbs,
        }
    }

    /// A sub-rectangle of somebody else's plane: exactly `width` per row.
    fn exact(stride: usize, len: usize) -> Self {
        Self {
            stride,
            len,
            tail: RowTail::Exact,
        }
    }
}

/// Who owns a destination row's columns past `width`.
///
/// The only thing separating a frame's collapse from a summand's, and so the
/// only reason there are two entry points at all.
#[derive(Clone, Copy, Debug)]
enum RowTail {
    /// A plane the caller owns outright: the columns past `width` are padding
    /// nobody reads back, so a final partial batch may overhang into them and
    /// a whole band is one call.
    Absorbs,
    /// A sub-rectangle of somebody else's plane: exactly `width` samples per
    /// row are written, and a final partial batch goes through the one-batch
    /// scratch rather than into a neighbour's columns.
    Exact,
}

/// How one band of rows is written into a destination plane: whole SIMD
/// batches stored straight through the collapse loop, and whatever partial
/// batch the stride left no room for.
#[derive(Clone, Copy, Debug)]
struct BandPlan {
    /// Samples per row the caller asked for.
    width: usize,
    rows: usize,
    /// Elements between the starts of two destination rows.
    stride: usize,
    /// Whole batches the collapse loop stores per row.
    groups: usize,
}

impl BandPlan {
    /// A final partial batch overhangs `width`, which is harmless exactly when
    /// the destination row both is wide enough to absorb it and is the
    /// caller's to clobber — the padding lanes then hold whatever the kernel
    /// computed past the right edge and are not read back. Otherwise it is
    /// left to the scratch batch, and the loop stores only the batches that
    /// fit.
    fn new(width: usize, rows: usize, stride: usize, tail: RowTail) -> Self {
        let batches = width.div_ceil(BATCH_LANES);
        let groups = match tail {
            RowTail::Absorbs if stride >= batches * BATCH_LANES => batches,
            _ => width / BATCH_LANES,
        };
        Self {
            width,
            rows,
            stride,
            groups,
        }
    }

    /// Samples of each row the collapse loop's whole batches cover.
    fn whole_lanes(self) -> usize {
        self.groups * BATCH_LANES
    }

    /// Bytes the collapse loop steps the output pointer by between rows.
    fn row_skip_bytes(self) -> usize {
        (self.stride - self.whole_lanes()) * core::mem::size_of::<f32>()
    }

    /// Samples per row left for the scratch batch, or `None` when the whole
    /// batches already covered the row.
    fn tail(self) -> Option<usize> {
        self.width
            .checked_sub(self.whole_lanes())
            .filter(|t| *t > 0)
    }

    /// Elements the destination must hold: every row but the last at full
    /// stride, then whichever of the batch overhang and the sampled width
    /// reaches further.
    fn span(self) -> Option<usize> {
        let Some(before_last) = self.rows.checked_sub(1) else {
            return Some(0);
        };
        before_last
            .checked_mul(self.stride)?
            .checked_add(self.whole_lanes().max(self.width))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// `x * 100 + y` — every sample names the coordinate it was taken at, so a
    /// misplaced row or column is visible in the value rather than only in a
    /// difference.
    fn coordinate_kernel() -> Kernel {
        Kernel::x().mul(&Kernel::constant(100.0)).add(&Kernel::y())
    }

    fn expected(col: usize, row: usize) -> f32 {
        (col as f32 + 0.5) * 100.0 + row as f32 + 0.5
    }

    /// Collapse a kernel of the coordinates themselves and read the plane back:
    /// the region's absolute rows, the sample-center convention, and a
    /// destination whose rows are exactly as wide as the samples, in one.
    #[test]
    fn a_band_collapses_its_absolute_rows_at_sample_centers() {
        let program = Manifold::compile(&coordinate_kernel(), [8, 8]);
        let frame = program.bind(&[]);
        let (width, y0, rows) = (5, 4, 3);
        let mut out = vec![0.0f32; rows * width];
        frame.collapse_rows(PlaneRegion::rows(width, y0, rows), &mut out, width);
        for row in 0..rows {
            for col in 0..width {
                let want = expected(col, y0 + row);
                assert!(
                    (out[row * width + col] - want).abs() < 1e-3,
                    "row {row} col {col}: {} != {want}",
                    out[row * width + col]
                );
            }
        }
    }

    /// The destination's rows are `stride` apart because the caller said so,
    /// not because the batch width worked out that way: every row's samples
    /// name their own coordinates, so a band placed at the wrong pitch reads
    /// back the wrong values. This is the case the collapse ABI's
    /// `row_skip_bytes` exists for, and the packed frame path is built on it.
    ///
    /// A spare row past the band stays pristine: the collapse fills the rows
    /// it was given and no more. (Within a row, everything past `width` is
    /// scratch — a final partial batch's lanes land there — which is why only
    /// the samples themselves are read back.)
    #[test]
    fn a_band_lands_at_the_stride_the_caller_asked_for() {
        let program = Manifold::compile(&coordinate_kernel(), [16, 8]);
        let frame = program.bind(&[]);
        let (width, stride, rows) = (9, BATCH_LANES * 4 + 1, 4);
        const UNTOUCHED: f32 = -1.0;
        let mut out = vec![UNTOUCHED; (rows + 1) * stride];
        frame.collapse_rows(PlaneRegion::rows(width, 0, rows), &mut out, stride);
        for row in 0..rows {
            for col in 0..width {
                let want = expected(col, row);
                assert!(
                    (out[row * stride + col] - want).abs() < 1e-3,
                    "row {row} col {col}: {} != {want}",
                    out[row * stride + col]
                );
            }
        }
        assert!(
            out[rows * stride..].iter().all(|&x| x == UNTOUCHED),
            "the collapse wrote past the {rows} rows it was given"
        );
    }

    /// Whatever the stride, the samples are the same: a row's final partial
    /// batch reaches memory through the scratch batch with the values it would
    /// have had from a whole-batch store into a padded row.
    #[test]
    fn a_partial_last_batch_collapses_the_same_samples_at_any_stride() {
        let program = Manifold::compile(&coordinate_kernel(), [32, 8]);
        let frame = program.bind(&[]);
        let (width, rows) = (BATCH_LANES * 2 - 1, 3);
        let padded = BATCH_LANES * 2;
        let mut packed = vec![0.0f32; rows * width];
        let mut spread = vec![0.0f32; rows * padded];
        let region = PlaneRegion::rows(width, 1, rows);
        frame.collapse_rows(region, &mut packed, width);
        frame.collapse_rows(region, &mut spread, padded);
        for row in 0..rows {
            let a = &packed[row * width..(row + 1) * width];
            let b = &spread[row * padded..row * padded + width];
            assert_eq!(a, b, "row {row}: the stride changed the samples");
        }
    }

    /// An int-domain root reaches memory as the bit pattern the kernel built,
    /// with no float operation in between — including through the scratch
    /// batch, which is why the width here is not a whole batch.
    #[test]
    fn an_int_domain_root_collapses_into_a_u32_plane_bit_exactly() {
        let kernel = Kernel::x()
            .trunc_to_int()
            .shl(8)
            .or(&Kernel::y().trunc_to_int())
            .into_kernel();
        let width = BATCH_LANES + 1;
        let program = Manifold::compile(&kernel, [width as u32, 4]);
        let frame = program.bind(&[]);
        let mut out = vec![0u32; 2 * width];
        frame.collapse_int_rows(PlaneRegion::rows(width, 0, 2), &mut out, width);
        for row in 0..2u32 {
            for col in 0..width as u32 {
                assert_eq!(
                    out[row as usize * width + col as usize],
                    (col << 8) | row,
                    "row {row} col {col}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "narrower than")]
    fn a_stride_below_the_sampled_width_is_refused() {
        let program = Manifold::compile(&coordinate_kernel(), [8, 8]);
        let frame = program.bind(&[]);
        let mut out = vec![0.0f32; 32];
        frame.collapse_rows(PlaneRegion::rows(5, 0, 2), &mut out, 4);
    }

    /// The same object with one channel and no pack: a kernel whose root is a
    /// gather over a declared buffer, compiled at the buffer's shape, bound by
    /// identity, and collapsed into one `f32` field. `Lattice::bake` cannot do
    /// this — it binds nothing and refuses a kernel that declares slots — so
    /// this is how a buffer-backed sampler (a glyph's coverage, a texture)
    /// reaches numbers.
    #[test]
    fn a_kernel_over_bound_memory_collapses_its_samples() {
        let buffer = BufferIdentity::mint();
        let (bw, bh) = (4u32, 3u32);
        let data: Vec<f32> = (0..bw * bh).map(|i| i as f32 * 0.25).collect();
        let kernel = crate::lattice::DiscreteManifold::kernel_for(buffer, bw, bh)
            .at(&Kernel::x(), &Kernel::y());
        let program = Manifold::compile(&kernel, [bw, bh]);
        assert_eq!(
            program.buffers().len(),
            1,
            "the sampler's slot must survive to the compiled program"
        );
        let frame = program.bind(&[(buffer, Arc::new(data.clone()))]);
        let (width, rows) = (bw as usize, bh as usize);
        let mut out = vec![0.0f32; rows * width];
        frame.collapse_rows(PlaneRegion::rows(width, 0, rows), &mut out, width);
        assert_eq!(out, data, "the collapse did not read the bound buffer");
    }

    #[test]
    #[should_panic(expected = "nothing bound to slot")]
    fn a_declared_slot_with_no_buffer_is_refused() {
        let buffer = BufferIdentity::mint();
        let kernel = crate::lattice::DiscreteManifold::kernel_for(buffer, 4, 4)
            .at(&Kernel::x(), &Kernel::y());
        let program = Manifold::compile(&kernel, [4, 4]);
        let _refused = program.bind(&[]);
    }

    #[test]
    #[should_panic(expected = "floats bound to slot")]
    fn a_buffer_of_the_wrong_length_is_refused() {
        let buffer = BufferIdentity::mint();
        let kernel = crate::lattice::DiscreteManifold::kernel_for(buffer, 4, 4)
            .at(&Kernel::x(), &Kernel::y());
        let program = Manifold::compile(&kernel, [4, 4]);
        let _refused = program.bind(&[(buffer, Arc::new(vec![0.0f32; 15]))]);
    }
}
