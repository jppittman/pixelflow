//! # A manifold is a kernel compiled at a lattice's shape
//!
//! The middle object of `kernel ──compile(shape)──▶ manifold
//! ──bind(buffers)──▶ bound ──collapse(region)──▶ buffer`. A [`Kernel`] is the
//! description; compiling it at a lattice's extents gives a **manifold**
//! ([`Manifold`]); binding the memory it declared gives a [`BoundManifold`];
//! collapsing that over a band of rows gives numbers. The lattice's folds —
//! rows, columns, lanes — are wrapped around the kernel before it is
//! scheduled and emitted as the loops of the code, with everything invariant
//! in a loop computed outside it, so a band is one call, and whatever memory
//! the kernel declared is bound by identity once and stays bound for every
//! band collapsed from it.
//!
//! ## Rank
//!
//! A manifold is compiled at a [`Lattice`](crate::Lattice)'s `[x, y]` extent,
//! which is exactly what a [`LatticeShape`](pixelflow_ir::LatticeShape) is:
//! the extents are the folds' trip counts, and the code is specialized to
//! them. A band of some other shape — a stripe of a frame, a cell grid's
//! claim on a wider band — is another shape and so other code: the manifold
//! keeps a table of the shapes it has been collapsed at and compiles a new
//! one on first use, through the same shape-keyed cache. A scalar the kernel
//! needs but the lattice does not vary is an argument ([`UniformBlock`]),
//! not a third axis of extent 1.
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
//! many elements apart the caller says: the code's stores address
//! `row · pitch + col`, so a destination at any stride is filled in place —
//! no staging plane, no per-row copy — and exactly `width` samples per row
//! are written, a row's final partial batch through a masked or lane-wise
//! store. The SIMD width is nowhere in this module.
//!
//! The store is a raw vector store, type-blind bit movement, so a kernel
//! whose root is int-domain (a packed pixel, a mask) collapses through
//! [`BoundManifold::collapse_int_rows`] into a `u32` plane exactly: no float
//! operation touches the value between the root and memory.

// The compiled code is executable memory, which is `std`'s business already;
// the table of shapes it is kept in needs a lock, which is `std`'s too.
extern crate std;

use alloc::sync::Arc;
use alloc::vec::Vec;
use std::sync::RwLock;

use crate::Field;
use pixelflow_codegen::CompiledKernel;
use pixelflow_ir::LatticeShape;
use pixelflow_ir::arena::{BufferDecl, BufferIdentity, UniformDecl, UniformIdentity};
use pixelflow_ir::{Kernel, Uniform};

/// Buffer slots a [`BoundManifold`] can bind without allocating: binding builds
/// its base-pointer array on the stack, so the bound is what makes that array
/// a fixed size. A kernel declaring more is refused at compile time rather
/// than silently overflowing it.
pub const MAX_BOUND_BUFFERS: usize = 4;

/// Entries in the context a collapse call hands the kernel: one base pointer
/// per buffer slot, then the uniform block's in the entry after the kernel's
/// last buffer (read only when the kernel has an argument), then the origin
/// block's — where the band's first sample lies.
const CONTEXT_ENTRIES: usize = MAX_BOUND_BUFFERS + 2;

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
    /// What [`Lattice::collapse`](crate::Lattice::collapse) and
    /// [`IndexRange::bake`](crate::IndexRange::bake) sample.
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
    /// The code at this manifold's own extent, compiled up front.
    jit: Arc<CompiledKernel>,
    /// That code and every other shape's, as they are asked for.
    codes: Arc<Codes>,
    /// The lattice shape this manifold was compiled for, `[x, y]`: the shape
    /// [`Lattice::collapse`](crate::Lattice::collapse) fills.
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
    /// Tabulations `kernel` itself carried at compile time
    /// (`Kernel::buffer_data`) — the data that travels with the kernel
    /// rather than being gathered by a caller and threaded to `bind`
    /// separately. A slot named here is already spoken for: `bind` fills it
    /// from this table before it looks at its own `buffers` argument, so a
    /// kernel built over bound memory
    /// (`DiscreteManifold::kernel`/`BilinearSampler::kernel`) binds with
    /// `bind(&[])`. A refcount clone out of the kernel, not a copy — the
    /// copy into `Arc<Vec<f32>>` `bind` needs happens there, once, only for
    /// a slot actually bound.
    carried: Arc<[(BufferIdentity, Arc<[f32]>)]>,
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
        let shape = LatticeShape::new(extent);
        let linked = pixelflow_codegen::jit_cache::compile(kernel, shape)
            .expect("Manifold: kernel failed to compile");
        assert!(
            linked.buffers.len() <= MAX_BOUND_BUFFERS,
            "Manifold::compile: kernel needs {} buffer slots, over the \
             {MAX_BOUND_BUFFERS} a frame can bind without allocating",
            linked.buffers.len()
        );
        let defaults: Vec<f32> = linked.uniforms.iter().map(|d| d.default).collect();
        let carried: Vec<(BufferIdentity, Arc<[f32]>)> = kernel
            .buffer_data()
            .map(|(id, data)| (id, Arc::clone(data)))
            .collect();
        let slots: Arc<[BufferDecl]> = linked.buffers.into();
        let link: Arc<[UniformDecl]> = linked.uniforms.into();
        Self {
            jit: Arc::clone(&linked.kernel),
            codes: Arc::new(Codes {
                kernel: kernel.clone(),
                slots: Arc::clone(&slots),
                link: Arc::clone(&link),
                compiled: RwLock::new(alloc::vec![(shape, linked.kernel)]),
            }),
            extent,
            slots,
            link,
            defaults: Arc::new(defaults),
            carried: carried.into(),
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

    /// The emitted bytes of the code compiled at this manifold's own extent
    /// (research/profiling harness).
    #[must_use]
    pub fn code_bytes(&self) -> &[u8] {
        self.jit.code_bytes()
    }

    /// Bind one frame's memory: each declared slot takes the buffer carrying
    /// its identity — first from what `kernel` itself carried into this
    /// [`Manifold`] at [`Manifold::compile`] (a kernel built over bound
    /// memory needs nothing here), then from `buffers` for anything still
    /// unfilled. `buffers` may be given in any order and may carry entries
    /// this kernel does not read. Buffers are `Arc`s so a frame in flight
    /// keeps its data alive while the caller prepares the next one.
    ///
    /// Every uniform argument is bound at its default; a frame that moves
    /// one applies a block with [`BoundManifold::with_uniforms`].
    ///
    /// # Panics
    ///
    /// Panics if a declared slot has no buffer bound to it (carried or
    /// supplied), or if one's length is not the `width × height` its
    /// declaration promised — the gathers address the declared shape, so a
    /// shorter buffer would be read past its end through an entirely safe
    /// API.
    #[must_use]
    pub fn bind(&self, buffers: &[(BufferIdentity, Arc<Vec<f32>>)]) -> BoundManifold {
        let mut bound: [Option<Arc<Vec<f32>>>; MAX_BOUND_BUFFERS] = Default::default();
        for (slot, decl) in bound.iter_mut().zip(self.slots.iter()) {
            let data: Arc<Vec<f32>> = match self.carried.iter().find(|(id, _)| *id == decl.id) {
                // The kernel's own tabulation: a refcount clone out of it,
                // copied into the `Vec` header `bind`'s ABI needs — once,
                // here, not once per composition the way a caller gathering
                // this by hand would have paid.
                Some((_, data)) => Arc::new(data.to_vec()),
                None => buffers
                    .iter()
                    .find(|(id, _)| *id == decl.id)
                    .map(|(_, data)| Arc::clone(data))
                    .unwrap_or_else(|| panic!("Manifold::bind: nothing bound to slot {decl:?}")),
            };
            assert_eq!(
                data.len(),
                buffer_len(decl),
                "Manifold::bind: buffer of {} floats bound to slot {decl:?}",
                data.len()
            );
            *slot = Some(data);
        }
        BoundManifold {
            codes: Arc::clone(&self.codes),
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

/// The code a manifold's kernel compiles to, per shape it has been collapsed
/// at. The shape is what the code *is* — its loop bounds are the extent —
/// so a band of a new shape is a compile, through the global shape-keyed
/// cache, and a band of a shape seen before is a lookup that allocates
/// nothing. Shared between a [`Manifold`] and every [`BoundManifold`] made
/// from it, so a shape one frame compiled is the next frame's hit.
struct Codes {
    kernel: Kernel,
    /// The link every shape's code binds against — the same for all of
    /// them, since the link is a function of the kernel's structure alone,
    /// and checked to be when a new shape is compiled.
    slots: Arc<[BufferDecl]>,
    link: Arc<[UniformDecl]>,
    compiled: RwLock<Vec<(LatticeShape, Arc<CompiledKernel>)>>,
}

impl Codes {
    /// Run `f` on the code for `shape`, compiling it first on the first
    /// request for that shape.
    fn with<R>(&self, shape: LatticeShape, f: impl FnOnce(&CompiledKernel) -> R) -> R {
        {
            let table = self.compiled.read().expect("Manifold: code table poisoned");
            if let Some((_, code)) = table.iter().find(|(s, _)| *s == shape) {
                return f(code);
            }
        }
        let linked = pixelflow_codegen::jit_cache::compile(&self.kernel, shape)
            .expect("Manifold: kernel failed to compile at a band's shape");
        debug_assert!(
            linked.buffers[..] == self.slots[..] && linked.uniforms[..] == self.link[..],
            "Manifold: a kernel's link changed with the shape it was compiled at"
        );
        let mut table = self
            .compiled
            .write()
            .expect("Manifold: code table poisoned");
        // Another thread may have compiled the same shape meanwhile; the
        // global cache handed both the same code, so keeping either is right.
        if !table.iter().any(|(s, _)| *s == shape) {
            table.push((shape, Arc::clone(&linked.kernel)));
        }
        f(&linked.kernel)
    }
}

/// A [`Manifold`] with its memory bound: the compiled code plus the buffers it
/// reads. Cheap to clone (one `Arc` for the code, one per bound buffer).
///
/// This is what a collapse takes. A kernel that reads nothing binds the empty
/// slice and is a bound manifold too — there is no second, buffer-free form.
#[derive(Clone)]
pub struct BoundManifold {
    codes: Arc<Codes>,
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
    /// The destination is written in place — the code's own stores land in
    /// it — so `stride` is whatever the caller's plane already is: a frame's
    /// packed row width, a padded scratch, a sub-rectangle of something
    /// larger. Exactly `width` samples per row are written and every other
    /// element of `out` is left as it was, so a caller writing one piece of
    /// a plane other pieces share writes only its own columns.
    ///
    /// The region's `width × rows` is a lattice shape, and the code that
    /// fills it is compiled the first time this manifold is collapsed at that
    /// shape (see [`Manifold`]); after that a band of the same shape is a
    /// call that allocates nothing.
    ///
    /// # Panics
    ///
    /// Panics if the region's width or row count is zero, `stride` is less
    /// than the width, or `out` cannot hold the band.
    pub fn collapse_rows(&self, region: PlaneRegion, out: &mut [f32], stride: usize) {
        self.check("collapse_rows", region, stride, out.len());
        // SAFETY: see `collapse`. `out` is an `f32` plane, which is what the
        // code writes, and `check` proved it holds the band.
        unsafe { self.collapse(region, out.as_mut_ptr(), stride) }
    }

    /// [`BoundManifold::collapse_rows`] for a kernel whose root is int-domain:
    /// each lane already holds a bit pattern (a packed pixel, a mask), and the
    /// store is a raw vector store — type-blind bit movement — so writing
    /// through the ABI's `*mut f32` into a `u32` plane is exact.
    ///
    /// # Panics
    ///
    /// Panics if the region's width or row count is zero, `stride` is less
    /// than the width, or `out` cannot hold the band.
    pub fn collapse_int_rows(&self, region: PlaneRegion, out: &mut [u32], stride: usize) {
        self.check("collapse_int_rows", region, stride, out.len());
        // SAFETY: see `collapse`. `u32` and `f32` share size and alignment,
        // and the store moves the root's bit pattern without interpreting it.
        unsafe { self.collapse(region, out.as_mut_ptr().cast::<f32>(), stride) }
    }

    /// The guard that a destination of `out_len` elements at `stride` can
    /// hold the band.
    ///
    /// # Panics
    ///
    /// Panics if the region's width or row count is zero, `stride` is less
    /// than the width, or `out_len` cannot hold the band.
    fn check(&self, what: &str, region: PlaneRegion, stride: usize, out_len: usize) {
        let (width, rows) = (region.width, region.rows);
        assert!(width > 0 && rows > 0, "{what}: empty region {width}×{rows}");
        assert!(
            stride >= width,
            "{what}: stride {stride} is narrower than the {width} samples a row holds"
        );
        // Checked: the span wraps in release for a caller-supplied region
        // large enough, and a wrapped product would let an undersized `out`
        // pass this guard while the code still wrote the real (enormous)
        // band. The documented panic must fire before any unsafe call.
        let needed = (rows - 1)
            .checked_mul(stride)
            .and_then(|before_last| before_last.checked_add(width))
            .expect("collapse: band span overflows usize");
        assert!(
            out_len >= needed,
            "{what}: plane of {out_len} elements cannot hold {rows} rows at stride {stride}"
        );
    }

    /// # Safety
    ///
    /// `out` must be writable for `(rows - 1) * stride + width` 4-byte
    /// elements — which [`BoundManifold::check`] asserted for the slice it
    /// came from.
    unsafe fn collapse(&self, region: PlaneRegion, out: *mut f32, stride: usize) {
        // One base pointer per declared slot, in slot order, then the block's
        // in the entry after them when the kernel has arguments, then the
        // origin's. Stack-allocated against the MAX_BOUND_BUFFERS bound
        // `compile` checked, so a band allocates nothing; entries between
        // stay null and are never read because the kernel only addresses
        // slots it declared.
        let mut ctx = [core::ptr::null::<f32>(); CONTEXT_ENTRIES];
        for (dst, src) in ctx.iter_mut().zip(self.bound.iter()) {
            if let Some(data) = src {
                *dst = data.as_ptr();
            }
        }
        if !self.uniforms.is_empty() {
            ctx[self.buffer_slots] = self.uniforms.as_ptr();
        }
        // Where the band's first sample lies; the code steps X by one per
        // column and Y by one per row from here.
        let origin = region.origin;
        ctx[self.buffer_slots + 1] = origin.as_ptr();
        let shape = LatticeShape::new([region.width as u32, region.rows as u32]);
        self.codes.with(shape, |code| {
            // SAFETY: `compile` checked size_of::<Field>() == JIT_VECTOR_BYTES
            // and that every declared slot fits `ctx`; `bind` bound a buffer
            // of the declared length to each of them and this frame holds
            // those `Arc`s alive for the duration of the call, as it does the
            // block the entry after them points into (one `f32` per argument,
            // in the link's order, which is the order the code was compiled
            // against) and the origin on this stack frame; the caller's
            // guard proved `out` holds `rows` rows of `width` at `stride`,
            // which is exactly what code compiled at `shape` writes.
            unsafe { code.call(ctx.as_ptr(), out, stride) }
        });
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
    /// back the wrong values. The packed frame path is built on this.
    ///
    /// Everything but the samples stays pristine: the columns past `width`
    /// in every row, and the spare row past the band. The collapse writes
    /// exactly what it was given and no more.
    #[test]
    fn a_band_lands_at_the_stride_the_caller_asked_for() {
        let program = Manifold::compile(&coordinate_kernel(), [16, 8]);
        let frame = program.bind(&[]);
        let (width, stride, rows) = (9, 61, 4);
        const UNTOUCHED: f32 = -1.0;
        let mut out = vec![UNTOUCHED; (rows + 1) * stride];
        frame.collapse_rows(PlaneRegion::rows(width, 0, rows), &mut out, stride);
        for row in 0..rows {
            for col in 0..stride {
                let got = out[row * stride + col];
                if col < width {
                    let want = expected(col, row);
                    assert!(
                        (got - want).abs() < 1e-3,
                        "row {row} col {col}: {got} != {want}"
                    );
                } else {
                    assert_eq!(
                        got, UNTOUCHED,
                        "row {row} col {col}: past the width was written"
                    );
                }
            }
        }
        assert!(
            out[rows * stride..].iter().all(|&x| x == UNTOUCHED),
            "the collapse wrote past the {rows} rows it was given"
        );
    }

    /// Whatever the stride, the samples are the same: a row's final partial
    /// batch is stored lane by lane with the values it would have had from a
    /// whole-batch store into a padded row.
    #[test]
    fn a_partial_last_batch_collapses_the_same_samples_at_any_stride() {
        let program = Manifold::compile(&coordinate_kernel(), [32, 8]);
        let frame = program.bind(&[]);
        let (width, rows) = (31, 3);
        let padded = 32;
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

    /// A band of a shape the manifold was not compiled at is compiled on
    /// first use and reused after: the frame path collapses eight-row
    /// stripes of a frame compiled at its whole height.
    #[test]
    fn a_band_of_a_new_shape_compiles_once_and_is_reused() {
        let program = Manifold::compile(&coordinate_kernel(), [16, 24]);
        let frame = program.bind(&[]);
        let mut out = vec![0.0f32; 16 * 24];
        for stripe in 0..3 {
            frame.collapse_rows(
                PlaneRegion::rows(16, stripe * 8, 8),
                &mut out[stripe * 8 * 16..],
                16,
            );
        }
        for row in 0..24 {
            for col in 0..16 {
                let want = expected(col, row);
                assert!(
                    (out[row * 16 + col] - want).abs() < 1e-3,
                    "row {row} col {col}"
                );
            }
        }
        let shapes = program.codes.compiled.read().unwrap().len();
        assert_eq!(
            shapes, 2,
            "the frame's shape and the stripe's, and nothing else"
        );
    }

    /// An int-domain root reaches memory as the bit pattern the kernel built,
    /// with no float operation in between — including through a row's final
    /// partial batch, which is why the width here is not a whole batch.
    #[test]
    fn an_int_domain_root_collapses_into_a_u32_plane_bit_exactly() {
        let kernel = Kernel::x()
            .trunc_to_int()
            .shl(8)
            .or(&Kernel::y().trunc_to_int())
            .into_kernel();
        let width = 17;
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
