//! CLAUDE.md: "Zero allocations — no per-frame heap allocation." A program
//! compiled once from an entry with a family (docs/plans/2026-09-25-the-
//! language-is-kernel.md §1.6) is rebound per call from its `Args`, whose
//! `write_into` streams the family's fields element-major — flat-mapped, a
//! chain of iterators, no `Vec` — into a block the caller keeps: so a call
//! allocates nothing once the block is the sole holder of its values.
//! `pixelflow-core/tests/bind_allocates_nothing.rs` pins the core's half of
//! this; this pins the half a `kernel!` expansion generates. Counted with a
//! wrapping global allocator; this binary holds nothing else, so the count
//! is this test's alone.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use pixelflow_compiler::kernel;
use pixelflow_core::{Manifold, PlaneRegion};

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

kernel! {
    pub struct Pair { pub a: f32, pub b: f32 }

    /// A family of records and one of `f32`s, beside a scalar.
    pub fn weighted<const N: usize>(pairs: [Pair; N], v: [f32; 2], r: f32) -> f32 {
        pairs.into_iter().map(|p| p.a * X + p.b).sum::<f32>() * r
            + v.into_iter().map(|e| e * Y).sum::<f32>()
    }
}

/// The samples of the one row collapsed: pixel centres, `X = i + ½`, `Y = ½`.
const ROW: usize = 4;

#[test]
fn rebinding_a_family_from_its_args_allocates_nothing() {
    let pairs = [Pair { a: 1.0, b: 0.0 }; 3];
    let program = Manifold::compile(&weighted::<3>(pairs, [0.0, 0.0], 1.0), [ROW as u32, 1]);
    let mut block = program.block();
    let mut out = vec![0.0f32; ROW];
    // The first write copies out of the manifold's shared defaults; after it
    // the block is the sole holder between calls.
    WeightedArgs {
        pairs,
        v: [0.0, 0.0],
        r: 1.0,
    }
    .write_into(&mut block)
    .expect("weighted's arguments");
    for call in 0..3 {
        let args = WeightedArgs {
            pairs: [Pair {
                a: call as f32,
                b: 1.0,
            }; 3],
            v: [0.5, 0.25],
            r: 2.0,
        };
        let allocations = allocations_during(|| {
            args.write_into(&mut block).expect("weighted's arguments");
            program.bind(&[]).with_uniforms(&block).collapse_rows(
                PlaneRegion::rows(ROW, 0, 1),
                &mut out,
                ROW,
            );
        });
        assert_eq!(allocations, 0, "call {call} allocated");
        let want: Vec<f32> = (0..ROW)
            .map(|i| 3.0 * (call as f32 * (i as f32 + 0.5) + 1.0) * 2.0 + 0.75 * 0.5)
            .collect();
        assert_eq!(out, want, "call {call}");
    }
}
