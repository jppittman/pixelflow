//! CLAUDE.md: "Zero allocations — no per-frame heap allocation." `bind` and
//! `collapse_rows` are the per-frame steps (the cell grid binds four
//! manifolds a frame), so they must allocate nothing — with or without
//! uniforms. Counted with a wrapping global allocator; this binary holds
//! nothing else, so the count is this test's alone.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use pixelflow_core::{DiscreteManifold, Kernel, Manifold, PlaneRegion, Uniform};
use pixelflow_ir::decl::BufferIdentity;

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards every call to `System` unchanged; the counter is the only
// addition and touches no allocator state.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocations_during(f: impl FnOnce()) -> usize {
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    f();
    ALLOCATIONS.load(Ordering::Relaxed) - before
}

/// `gather(buf, x) · scale + cx`, over one buffer and (optionally) two
/// uniforms — the terminal's shape and a moving scene's, in one kernel.
fn kernel(buffer: BufferIdentity, with_uniforms: bool) -> Kernel {
    let read = DiscreteManifold::kernel_for(buffer, 4, 1).at(&Kernel::x(), &Kernel::constant(0.0));
    if with_uniforms {
        read.mul(&Uniform::new(2.0).kernel())
            .add(&Uniform::new(0.5).kernel())
    } else {
        read.mul(&Kernel::constant(2.0)).add(&Kernel::constant(0.5))
    }
}

#[test]
fn binding_and_collapsing_allocate_nothing_per_frame() {
    for with_uniforms in [false, true] {
        let buffer = BufferIdentity::mint();
        let program = Manifold::compile(&kernel(buffer, with_uniforms), [4, 2]);
        let data = Arc::new(vec![1.0f32, 2.0, 3.0, 4.0]);
        let mut out = vec![0.0f32; 8];
        let block = program.block();
        // Warm anything lazy (nothing is expected to be), then count.
        program
            .bind(&[(buffer, Arc::clone(&data))])
            .with_uniforms(&block)
            .collapse_rows(PlaneRegion::rows(4, 0, 2), &mut out, 4);
        let allocations = allocations_during(|| {
            let bound = program
                .bind(&[(buffer, Arc::clone(&data))])
                .with_uniforms(&block);
            bound.collapse_rows(PlaneRegion::rows(4, 0, 2), &mut out, 4);
        });
        assert_eq!(
            allocations, 0,
            "bind + with_uniforms + collapse allocated (uniforms: {with_uniforms})"
        );
        assert_eq!(&out[..4], &[2.5, 4.5, 6.5, 8.5]);
    }
    setting_a_sole_holders_block_allocates_nothing();
}

/// Setting a value while no frame holds the previous ones writes in place:
/// a consumer that drops last frame's bound manifold first allocates nothing
/// per frame. (Setting while a frame still holds them copies — that is the
/// price of two frames in flight, paid once per set, never per stripe.)
///
/// Called from the one `#[test]` above rather than being one itself: the
/// counter is process-global, and two tests in one binary would count each
/// other's compiles.
fn setting_a_sole_holders_block_allocates_nothing() {
    let buffer = BufferIdentity::mint();
    let cx = Uniform::new(0.5);
    let k = DiscreteManifold::kernel_for(buffer, 4, 1)
        .at(&Kernel::x(), &Kernel::constant(0.0))
        .add(&cx.kernel());
    let program = Manifold::compile(&k, [4, 1]);
    let mut block = program.block();
    let data = Arc::new(vec![1.0f32; 4]);
    let mut out = vec![0.0f32; 4];
    // First set copies out of the manifold's shared defaults; after that the
    // block is the sole holder between frames.
    block.set(cx, 1.0).expect("argument");
    for frame in 0..3 {
        let allocations = allocations_during(|| {
            block.set(cx, frame as f32).expect("argument");
            program
                .bind(&[(buffer, Arc::clone(&data))])
                .with_uniforms(&block)
                .collapse_rows(PlaneRegion::rows(4, 0, 1), &mut out, 4);
        });
        assert_eq!(allocations, 0, "frame {frame} allocated");
        assert_eq!(out, [1.0 + frame as f32; 4]);
    }
}
