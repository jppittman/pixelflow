//! # Scene: what gets rendered into a frame
//!
//! A scene is a [`PackedFrame`] — four channel kernels compiled at the
//! frame's lattice shape, with the pixel pack inside the kernel, which is a
//! compiled manifold with four channels. Rendering it is **one collapse call
//! per stripe**, stripes across threads, the kernel's stores landing **in the
//! frame itself**. No per-batch FFI boundary, no virtual dispatch, no
//! per-pixel scalar pack, no staging plane and no row copy, and the collapse
//! kernel's two-level LICM prologues (per call, per row) active. Byte order
//! comes from the pixel format the frame is stored in
//! ([`compile_packed_for`], [`compile_platform_packed`]), never from
//! application code.
//!
//! The cell grid ([`compile_cell_grid_for`]) is one instance of that: a
//! geometry that denotes four channel kernels over a cell buffer and a
//! coverage atlas. A procedural shader is another: four kernels and nothing
//! bound. There is no second lane: S4a deleted the per-batch `Surface`
//! variant and the work-stealing rasterizer behind it.

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::render::cell_grid::{CellGridPackedFrame, CellGridPackedManifold};
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::render::color::PlatformPixel;
use crate::render::frame::Frame;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::render::packed::{PackedFrame, PackedManifold};
use crate::render::Pixel;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::scene3d::Rgba;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pixelflow_core::{CellGridShape, FastMathGuard, IndexRange, PlaneRegion};
use std::sync::Mutex;

/// A renderable scene: four channel kernels compiled at the frame's lattice
/// shape, packed to a `u32` pixel inside the kernel. The scene collapses it
/// one call per stripe, straight into the frame's own memory.
#[derive(Clone)]
pub enum Scene {
    /// A program that answers for the whole frame.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    Packed(PackedFrame),
    /// A cell grid: a program that answers for the grid's own index range,
    /// with a constant border over the rest of the frame.
    ///
    /// The second variant, and the reason there is one: the range a cell
    /// grid covers is a function of its per-frame *metric*, not of the shape
    /// it was compiled at, so it cannot be folded into the compiled program
    /// the way an extent can. Both variants are architecture-gated, because a
    /// target with no JIT backend has no scene at all rather than a
    /// different one.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    CellGrid(CellGridPackedFrame),
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl From<PackedFrame> for Scene {
    fn from(frame: PackedFrame) -> Self {
        Self::Packed(frame)
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl From<CellGridPackedFrame> for Scene {
    fn from(frame: CellGridPackedFrame) -> Self {
        Self::CellGrid(frame)
    }
}

/// Compose a colour output for the pixel format `P` and compile it at the
/// frame's shape: four channel kernels in `[0, 1]` — red, green, blue, alpha,
/// which is all a colour output ever is — over a `frame[0] × frame[1]` pixel
/// lattice. What comes back is a compiled manifold the scene collapses one
/// call per stripe.
///
/// This is the constructor a shader author uses. Byte order comes from `P`
/// itself, so the format the kernel packs for is the format the frame stores,
/// by construction; the author never sees a shift.
///
/// The channel kernels are sampled at **device pixel centers** of the frame
/// they are compiled for. A shader authored in some other space says so in
/// the language, by precomposing the embedding — `channel.at(&(X * s), &(Y *
/// s))` — before compiling here; there is no separate scale for anything
/// downstream to keep in step.
///
/// # Panics
///
/// Panics for a `P` with no packed RGBA form ([`Pixel::packed_shifts`]
/// returns `None`) — a grayscale or exotic format has no byte lanes to pack
/// into, and silently packing RGBA into it would be garbage.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[must_use]
pub fn compile_packed_for<P: Pixel>(color: &Rgba, frame: [u32; 2]) -> PackedManifold {
    PackedManifold::compile(color, packed_shifts_of::<P>("compile_packed_for"), frame)
}

/// [`compile_packed_for`] with THIS platform's pixel byte order.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[must_use]
pub fn compile_platform_packed(color: &Rgba, frame: [u32; 2]) -> PackedManifold {
    compile_packed_for::<PlatformPixel>(color, frame)
}

/// A solid colour as a packed scene for the pixel format `P`: four constant
/// channel kernels over a `frame[0] × frame[1]` lattice, nothing bound.
///
/// The simplest scene there is, and the shape every scene has — so a caller
/// that only needs one flat colour takes the path production takes rather
/// than a lane of its own.
///
/// # Panics
///
/// Panics for a `P` with no packed RGBA form, on a degenerate frame extent,
/// or if the kernel fails to compile.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[must_use]
pub fn constant_scene_for<P: Pixel>(rgba: [f32; 4], frame: [u32; 2]) -> Scene {
    let color = Rgba::from(rgba.map(pixelflow_core::Kernel::constant));
    Scene::Packed(compile_packed_for::<P>(&color, frame).bind(&[]))
}

/// [`constant_scene_for`] with THIS platform's pixel byte order.
///
/// # Panics
///
/// Panics on a degenerate frame extent, or if the kernel fails to compile.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[must_use]
pub fn constant_platform_scene(rgba: [f32; 4], frame: [u32; 2]) -> Scene {
    constant_scene_for::<PlatformPixel>(rgba, frame)
}

/// The byte lanes `P` packs into.
///
/// # Panics
///
/// Panics for a `P` with no packed RGBA form.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn packed_shifts_of<P: Pixel>(what: &str) -> [u32; 4] {
    P::packed_shifts().unwrap_or_else(|| {
        panic!(
            "{what}: {} has no byte lanes to pack RGBA into",
            core::any::type_name::<P>()
        )
    })
}

/// Compose the cell grid's four channel kernels and compile them at its
/// frame's shape for the pixel format `P`, taking byte order from `P` itself
/// — so the format the kernel packs for is the format the frame stores, by
/// construction. Applications pass a shape and colors; they never see a
/// shift. The grid's metric arrives later, through
/// `CellGridPackedManifold::params`, without recompiling anything.
///
/// Use [`compile_platform_cell_grid`] when the frame is the platform's own
/// format, which is the production path.
///
/// # Panics
///
/// Panics for a `P` with no packed RGBA form ([`Pixel::packed_shifts`]
/// returns `None`) — a grayscale or exotic format has no byte lanes to pack
/// into, and silently packing RGBA into it would be garbage.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[must_use]
pub fn compile_cell_grid_for<P: Pixel>(
    shape: CellGridShape,
    default_bg: [f32; 4],
) -> CellGridPackedManifold {
    CellGridPackedManifold::compile(
        shape,
        default_bg,
        packed_shifts_of::<P>("compile_cell_grid_for"),
    )
}

/// Compile the packed cell-grid program with THIS platform's pixel byte
/// order, which [`PlatformPixel`] is the single statement of.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[must_use]
pub fn compile_platform_cell_grid(
    shape: CellGridShape,
    default_bg: [f32; 4],
) -> CellGridPackedManifold {
    compile_cell_grid_for::<PlatformPixel>(shape, default_bg)
}

impl core::fmt::Debug for Scene {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            Self::Packed(_) => f.debug_tuple("Scene::Packed").finish(),
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            Self::CellGrid(_) => f.debug_tuple("Scene::CellGrid").finish(),
        }
    }
}

impl Scene {
    /// Render this scene into `frame` with up to `num_threads` workers: one
    /// collapse call per stripe, stripes pulled by the workers, finished
    /// pixels landing straight in the frame's memory.
    pub fn render<P: Pixel + Send>(&self, frame: &mut Frame<P>, num_threads: usize) {
        match self {
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            Self::Packed(packed) => {
                render_bands(packed.shifts(), frame, num_threads, &|band, out, stride| {
                    packed.collapse_rows(
                        PlaneRegion::rows(band.width(), band.y0(), band.rows()),
                        out,
                        stride,
                    );
                })
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            Self::CellGrid(grid) => {
                render_bands(grid.shifts(), frame, num_threads, &|band, out, stride| {
                    grid.collapse_rows(band, out, stride);
                });
            }
        }
    }
}

/// Rows one worker claims at a time — the band of the lattice one collapse
/// call covers.
///
/// The unit of scheduling, so the two costs it trades are: a stripe too short
/// pays the collapse call's per-call LICM prologue too often and claims too
/// often, while a stripe too tall makes the last worker's straggling stripe
/// the frame's critical path. Swept at 4 / 8 / 16 / 32 rows on the
/// psychedelic shader at 1920×1080 and the four values came out within 3% of
/// each other — inside the run-to-run spread — so the height is chosen for
/// the property a four-core host cannot show: stripes per worker on a wide
/// machine. Eight rows keeps at least four stripes per worker up to 32 cores
/// at 1080p, where sixteen would leave two.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const STRIPE_ROWS: usize = 8;

/// Collapse a scene over the frame, stripe-parallel, through whatever
/// `collapse` does with one index band.
///
/// The band is an [`IndexRange`] rather than a `PlaneRegion` because this is
/// the domain side: the loop chooses *which* indices a worker claims, and
/// what coordinate a frame samples them at belongs to the collapse it calls.
///
/// Workers pull fixed-height stripes from one shared cursor until the frame is
/// done, and each collapses its stripe with ONE call whose stores land in the
/// frame itself: the collapse ABI steps the output pointer between rows, so
/// the frame's row stride is simply what it is handed. Nothing is staged,
/// nothing is copied, and nothing per-worker is allocated.
///
/// Pulling rather than partitioning is what makes a slow stripe survivable —
/// a row band over a glyph-dense region costs more than one over background,
/// and with a static split the whole frame waits for whoever drew it.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn render_bands<P: Pixel + Send>(
    shifts: [u32; 4],
    frame: &mut Frame<P>,
    num_threads: usize,
    collapse: &(dyn Fn(IndexRange, &mut [u32], usize) + Sync),
) {
    let (width, height) = (frame.width, frame.height);
    if width == 0 || height == 0 {
        return;
    }
    // The kernel packed its word for the shifts it was compiled with;
    // storing that word as a format with different byte order would swap R
    // and B in every pixel. The old per-pixel `from_rgba` path converted, so
    // any `P` worked; this path moves bits, so the formats must match.
    assert_eq!(
        P::packed_shifts(),
        Some(shifts),
        "packed frame format does not match the shifts its kernel packed \
         for — compile it with `compile_packed_for::<P>`"
    );
    let workers = worker_count(num_threads, height.div_ceil(STRIPE_ROWS));
    // A packed pixel IS a `u32` word here — that is what the kernel's root
    // computed and what `packed_shifts` above just agreed on.
    let pixels: &mut [u32] = &mut frame.as_u32_slice_mut()[..width * height];
    // The cursor and the disjointness proof in one: `chunks_mut` hands each
    // claim a `&mut` band that provably overlaps no other, so the borrow
    // checker states the invariant an atomic row counter would only assert.
    let stripes = Mutex::new(pixels.chunks_mut(STRIPE_ROWS * width).enumerate());

    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                // Denormals cost up to 100x a normal operation, and a shader
                // that decays toward zero — an exponential falloff, a
                // reciprocal of something large — produces them by the
                // batch. FTZ/DAZ is per-thread CPU state (MXCSR on x86, FPCR
                // on aarch64, saved and restored across context switches), so
                // it has to be set on the thread that runs the kernel: a
                // guard held where the workers are spawned would not reach
                // them. One guard covers every stripe this worker claims and
                // restores the mode before the scope joins.
                //
                // SAFETY: this thread was spawned to collapse stripes and does
                // nothing else, so nothing here depends on denormals being
                // preserved; the guard restores the caller's mode on drop,
                // and the scope joins before `render` returns.
                let _fast_math = unsafe { FastMathGuard::new() };
                while let Some((stripe, band)) = claim(&stripes) {
                    let rows = band.len() / width;
                    collapse(
                        IndexRange::new(0, stripe * STRIPE_ROWS, width, rows),
                        band,
                        width,
                    );
                }
            });
        }
    });
}

/// How many workers to actually spawn for a request of `num_threads` over
/// `stripes` stripes.
///
/// A request larger than the machine has cores is honoured as *the machine's
/// cores*, not as itself. Stripes are pure compute with no I/O to overlap, so
/// extra threads cannot hide latency; they only add context switches and
/// evict each other's cache lines, and with a shared cursor there is no
/// load-imbalance argument left for oversubscribing either — the pull already
/// covers it. (`PerformanceConfig::default().render_threads` is 12, which on a
/// four-core host used to mean twelve stripes contending for four cores.)
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn worker_count(num_threads: usize, stripes: usize) -> usize {
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    num_threads.clamp(1, cores).min(stripes.max(1))
}

/// Take the next stripe, or `None` once the frame is done.
///
/// Split out so the lock is provably dropped before the collapse: holding it
/// across a stripe would serialize the whole render.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn claim<'a>(
    stripes: &Mutex<core::iter::Enumerate<core::slice::ChunksMut<'a, u32>>>,
) -> Option<(usize, &'a mut [u32])> {
    stripes
        .lock()
        .expect("a render worker panicked holding the stripe cursor")
        .next()
}

#[cfg(test)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::render::color::Rgba8;
    use pixelflow_core::{CellGridMetrics, Kernel};
    use std::sync::Arc;

    /// A 2×2 grid of solid/half tiles; oracle is the scalar blend math.
    fn scene() -> Scene {
        let (aw, ah, slot) = (12usize, 6usize, 6usize);
        let mut atlas = vec![0.0f32; aw * ah];
        for r in 0..4 {
            for c in 0..4 {
                atlas[(1 + r) * aw + 1 + c] = 1.0; // tile 0: solid
                atlas[(1 + r) * aw + slot + 1 + c] = 0.5; // tile 1: half
            }
        }
        let shape = CellGridShape {
            cols: 2,
            rows: 2,
            atlas_width: aw as u32,
            atlas_height: ah as u32,
            frame_w: 10,
            frame_h: 10,
        };
        let metrics = CellGridMetrics {
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            tile_w: 4,
            tile_h: 4,
            scale: 1.0,
        };
        let program = compile_cell_grid_for::<Rgba8>(shape, [0.25, 0.25, 0.25, 1.0]);
        #[rustfmt::skip]
        let cells = vec![
            // (0,0): solid tile, red on black
            1.0, 1.0,  1.0, 0.0, 0.0, 1.0,  0.0, 0.0, 0.0, 1.0,
            // (1,0): half tile, white on blue
            7.0, 1.0,  1.0, 1.0, 1.0, 1.0,  0.0, 0.0, 1.0, 1.0,
            // (0,1): half tile, green on black
            7.0, 1.0,  0.0, 1.0, 0.0, 1.0,  0.0, 0.0, 0.0, 1.0,
            // (1,1): solid tile, black on white
            1.0, 1.0,  0.0, 0.0, 0.0, 1.0,  1.0, 1.0, 1.0, 1.0,
        ];
        Scene::CellGrid(program.frame(&program.params(&metrics), Arc::new(cells), Arc::new(atlas)))
    }

    #[test]
    fn cell_grid_scene_packs_blended_channels() {
        let scene = scene();
        // 10×10 frame: the 8×8 grid plus a margin of default background.
        let mut frame = Frame::<Rgba8>::new(10, 10);
        scene.render(&mut frame, 2);

        let px = |x: usize, y: usize| frame.data[y * 10 + x];
        // Cell (0,0) interior: coverage 1 → pure red.
        assert_eq!((px(1, 1).r(), px(1, 1).g(), px(1, 1).b()), (255, 0, 0));
        // Cell (1,0) interior: coverage ½ of white over blue → (128, 128, 255)ish.
        let p = px(5, 1);
        assert!(p.r() >= 126 && p.r() <= 129, "half blend r = {}", p.r());
        assert_eq!(p.b(), 255);
        // Cell (1,1) interior: black over white at coverage 1 → black.
        assert_eq!(px(5, 5).r(), 0);
        // Outside the grid: the default background (0.25 → ~64).
        let m = px(9, 9);
        assert!(m.r() >= 62 && m.r() <= 66, "margin r = {}", m.r());
    }

    /// Which worker draws which stripe is a scheduling decision, and no
    /// scheduling decision may be visible in the pixels. The width here is
    /// deliberately not a whole number of SIMD batches, so each stripe's last
    /// column is also the partial-batch case, and the height spans more than
    /// one stripe so the cursor is actually contended.
    #[test]
    fn how_the_stripes_are_shared_out_does_not_change_pixels() {
        let (w, h) = (9u32, (STRIPE_ROWS * 3 + 1) as u32);
        // A coordinate ramp: every pixel's value names where it was sampled,
        // so a stripe placed at the wrong row is a different colour, not a
        // coincidentally equal one.
        let ramp = |scale: f32| {
            Kernel::x()
                .add(&Kernel::y().mul(&Kernel::constant(scale)))
                .mul(&Kernel::constant(1.0 / 255.0))
        };
        let channels = [ramp(1.0), ramp(3.0), ramp(7.0), Kernel::constant(1.0)];
        let color = Rgba::from(&channels);
        let scene = Scene::Packed(compile_packed_for::<Rgba8>(&color, [w, h]).bind(&[]));
        let mut one = Frame::<Rgba8>::new(w, h);
        scene.render(&mut one, 1);
        for threads in [2usize, 3, 8, 16] {
            let mut many = Frame::<Rgba8>::new(w, h);
            scene.render(&mut many, threads);
            assert_eq!(
                one.data, many.data,
                "{threads} workers rendered different pixels than one"
            );
        }
    }

    /// A request past the machine's cores is honoured as the machine's cores,
    /// and a frame shorter than one stripe never spawns a worker with nothing
    /// to do.
    #[test]
    fn a_request_beyond_the_cores_is_clamped_to_them() {
        let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        let plenty = usize::MAX;
        assert_eq!(super::worker_count(plenty, plenty), cores);
        assert_eq!(super::worker_count(1, plenty), 1);
        assert_eq!(super::worker_count(plenty, 1), 1);
        assert_eq!(super::worker_count(0, plenty), 1);
    }

    #[test]
    fn platform_wrapper_needs_no_shifts_from_the_caller() {
        let (aw, ah) = (12usize, 6usize);
        let atlas = vec![0.0f32; aw * ah];
        let shape = CellGridShape {
            cols: 2,
            rows: 2,
            atlas_width: aw as u32,
            atlas_height: ah as u32,
            frame_w: 8,
            frame_h: 8,
        };
        let metrics = CellGridMetrics {
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            tile_w: 4,
            tile_h: 4,
            scale: 1.0,
        };
        let program = super::compile_platform_cell_grid(shape, [0.0, 0.0, 0.0, 1.0]);
        // Zero coverage everywhere, so every in-grid pixel shows its cell's
        // BACKGROUND — which must be opaque for the alpha assertion to hold.
        let mut cells = vec![0.0f32; shape.cells_len()];
        for cell in cells.chunks_mut(pixelflow_core::CELL_STRIDE) {
            cell[9] = 1.0; // bg_a
        }
        let frame_data = program.frame(&program.params(&metrics), Arc::new(cells), Arc::new(atlas));
        // The platform wrapper packs for PlatformPixel, so the frame must BE
        // PlatformPixel — pairing it with a hardcoded format is the mistake
        // the render-time format guard exists to catch, and did catch here
        // (this test asserted Rgba8 and failed on Linux, where the platform
        // is Bgra8). Alpha is byte 3 in both orders, so the assertion below
        // is the one claim that holds whichever platform runs it.
        let mut frame = Frame::<crate::render::color::PlatformPixel>::new(8, 8);
        Scene::CellGrid(frame_data).render(&mut frame, 1);
        assert!(
            frame.data.iter().all(|p| p.0 >> 24 == 255),
            "alpha lane must be opaque regardless of platform byte order"
        );
    }

    /// The shader-author path: four channel kernels and a frame size, with
    /// nothing bound. No geometry, no buffers, no `Manifold`.
    #[test]
    fn four_channel_kernels_and_a_frame_size_are_a_scene() {
        let k = Kernel::constant;
        // r ramps across the frame, g down it, b and a constant.
        let channels = [
            Kernel::x().mul(&k(1.0 / 8.0)),
            Kernel::y().mul(&k(1.0 / 8.0)),
            k(0.0),
            k(1.0),
        ];
        let program = compile_packed_for::<Rgba8>(&Rgba::from(&channels), [8, 8]);
        let scene = Scene::Packed(program.bind(&[]));
        let mut frame = Frame::<Rgba8>::new(8, 8);
        scene.render(&mut frame, 2);

        for y in 0..8usize {
            for x in 0..8usize {
                let p = frame.data[y * 8 + x];
                let expect = |v: f32| ((v * 255.0).clamp(0.0, 255.0)) as u8;
                assert_eq!(
                    (p.r(), p.g(), p.b(), p.a()),
                    (
                        expect((x as f32 + 0.5) / 8.0),
                        expect((y as f32 + 0.5) / 8.0),
                        0,
                        255
                    ),
                    "pixel ({x}, {y})"
                );
            }
        }
    }

    /// A shader authored in point space targets a denser device frame by
    /// PRECOMPOSITION — `Kernel::at` on the channel kernels, in the language
    /// — not by a wrapper somewhere downstream. The scale is spent at compile
    /// time, so nothing per-frame has to know about it.
    #[test]
    fn a_point_space_shader_reaches_a_denser_frame_by_precomposition() {
        let k = Kernel::constant;
        let red = Kernel::x().mul(&k(1.0 / 16.0));
        let channels = |r: Kernel| [r, k(0.0), k(0.0), k(1.0)];

        // Authored over a 16-point-wide surface...
        let points = compile_packed_for::<Rgba8>(&Rgba::from(channels(red.clone())), [16, 16]);
        let mut point_frame = Frame::<Rgba8>::new(16, 16);
        Scene::Packed(points.bind(&[])).render(&mut point_frame, 1);

        // ...and sampled on a 2x device grid: x_point = x_device / 2.
        let half = |v: &Kernel| v.mul(&k(0.5));
        let device = compile_packed_for::<Rgba8>(
            &Rgba::from(channels(red.at(&half(&Kernel::x()), &half(&Kernel::y())))),
            [32, 32],
        );
        let mut device_frame = Frame::<Rgba8>::new(32, 32);
        Scene::Packed(device.bind(&[])).render(&mut device_frame, 1);

        for (x, px) in device_frame.data[..32].iter().enumerate() {
            let expected = (((x as f32 + 0.5) * 0.5 / 16.0) * 255.0) as u8;
            assert_eq!(
                px.r(),
                expected,
                "device column {x} does not sample point space"
            );
        }
        // The two agree where they sample the same point: device pixel 2i+…
        // straddles point pixel i, so the point frame's value is between the
        // device frame's two.
        for i in 0..16usize {
            let (lo, hi) = (
                device_frame.data[2 * i].r(),
                device_frame.data[2 * i + 1].r(),
            );
            let mid = point_frame.data[i].r();
            assert!(
                lo <= mid && mid <= hi,
                "point column {i} ({mid}) is not between device columns ({lo}, {hi})"
            );
        }
    }

    #[test]
    fn stripe_split_matches_single_threaded() {
        let scene = scene();
        let mut one = Frame::<Rgba8>::new(9, 8);
        let mut many = Frame::<Rgba8>::new(9, 8);
        scene.render(&mut one, 1);
        scene.render(&mut many, 4);
        assert_eq!(one.data, many.data, "thread count must not change pixels");
    }
}

#[cfg(test)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod pixel_format_tests {
    use super::*;
    use crate::render::color::{Bgra8, Rgba8};
    use pixelflow_core::{CellGridMetrics, CellGridShape};
    use std::sync::Arc;

    /// The kernel packs for one byte order and the frame stores raw words,
    /// so a mismatched frame format would swap R and B in every pixel. The
    /// old per-pixel `from_rgba` path converted; this one must refuse.
    #[test]
    #[should_panic(expected = "does not match the shifts its kernel packed for")]
    fn mismatched_frame_format_is_refused() {
        let (aw, ah) = (12usize, 6usize);
        let shape = CellGridShape {
            cols: 2,
            rows: 2,
            atlas_width: aw as u32,
            atlas_height: ah as u32,
            frame_w: 8,
            frame_h: 8,
        };
        let metrics = CellGridMetrics {
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            tile_w: 4,
            tile_h: 4,
            scale: 1.0,
        };
        // Compiled for Rgba8's lanes...
        let program = compile_cell_grid_for::<Rgba8>(shape, [0.0, 0.0, 0.0, 1.0]);
        let frame_data = program.frame(
            &program.params(&metrics),
            Arc::new(vec![0.0f32; shape.cells_len()]),
            Arc::new(vec![0.0f32; aw * ah]),
        );
        // ...rendered into a Bgra8 frame.
        let mut frame = Frame::<Bgra8>::new(8, 8);
        Scene::CellGrid(frame_data).render(&mut frame, 1);
    }

    /// Both packed formats work when the kernel is compiled for them.
    #[test]
    fn each_packed_format_renders_through_its_own_shifts() {
        let (aw, ah) = (12usize, 6usize);
        let shape = CellGridShape {
            cols: 2,
            rows: 2,
            atlas_width: aw as u32,
            atlas_height: ah as u32,
            frame_w: 8,
            frame_h: 8,
        };
        let metrics = CellGridMetrics {
            cell_w: 4.0,
            cell_h: 4.0,
            density: 1.0,
            tile_w: 4,
            tile_h: 4,
            scale: 1.0,
        };
        // Opaque red background everywhere: coverage is zero (empty atlas),
        // so every in-grid pixel is its cell's background.
        let mut cells = vec![0.0f32; shape.cells_len()];
        for cell in cells.chunks_mut(pixelflow_core::CELL_STRIDE) {
            cell[6] = 1.0; // bg_r
            cell[9] = 1.0; // bg_a
        }
        let atlas = Arc::new(vec![0.0f32; aw * ah]);
        let cells = Arc::new(cells);

        let rgba_program = compile_cell_grid_for::<Rgba8>(shape, [1.0, 0.0, 0.0, 1.0]);
        let rgba = rgba_program.frame(&rgba_program.params(&metrics), cells.clone(), atlas.clone());
        let mut rf = Frame::<Rgba8>::new(8, 8);
        Scene::CellGrid(rgba).render(&mut rf, 1);

        let bgra_program = compile_cell_grid_for::<Bgra8>(shape, [1.0, 0.0, 0.0, 1.0]);
        let bgra = bgra_program.frame(&bgra_program.params(&metrics), cells, atlas);
        let mut bf = Frame::<Bgra8>::new(8, 8);
        Scene::CellGrid(bgra).render(&mut bf, 1);

        // Same color through both formats: the raw words differ (byte order),
        // the decoded channels agree.
        assert_ne!(rf.data[0].0, bf.data[0].0, "byte orders must differ");
        assert_eq!((rf.data[0].r(), rf.data[0].b()), (255, 0));
        assert_eq!((bf.data[0].r(), bf.data[0].b()), (255, 0));
    }
}
