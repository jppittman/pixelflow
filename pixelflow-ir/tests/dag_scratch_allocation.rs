//! Proves `Dag::scratch()` + `Node::descendants_in` amortize to zero
//! allocations across repeated walks — the whole reason `Scratch` exists
//! over the simpler `Node::descendants()`.
//!
//! Needs a process-wide `#[global_allocator]` to count, which is why this
//! lives in its own integration-test binary rather than `dag.rs`'s unit
//! test module: `cargo test` runs a `--lib` binary's tests concurrently by
//! default, and unrelated tests allocating on other threads would pollute a
//! shared counter. A dedicated binary is its own process — nothing else
//! here is allocating into the count.

use pixelflow_ir::internal_test_support::scratch_allocation_fixture as build;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static A: Counting = Counting;

// The fixture itself (`x`, `y`, `add = x + y`, `root = add * x`) lives in
// `pixelflow_ir::internal_test_support`, imported above as `build`:
// `dag::Builder`/`dag::Id` are `pub(crate)`, and this integration test is a
// separate Cargo target that only sees `pub` API, same as any external
// crate. What this test actually exercises — `Scratch`, `descendants_in`,
// the allocation count — never touches `Builder` or `Id` either way.

#[test]
fn repeated_walks_do_not_allocate() {
    let g = build();
    let mut sc = g.scratch();
    let mut sink = 0usize;

    // Measure 1000-call windows until two *consecutive* ones cost exactly
    // the same, instead of asserting a literal zero (which bakes in one
    // platform's allocator behavior) or a fixed warmup count (which just
    // moves the guess elsewhere). Observed on the macOS runner: a 64-call
    // warmup wasn't enough -- window 1 still cost 5 allocations, window 2
    // cost 0 -- i.e. some one-time, per-process allocator setup lands
    // *inside* the first measured window rather than before it, and there
    // is no fixed call count that's guaranteed to precede it. Retrying
    // until consecutive windows agree handles that regardless of how many
    // calls it takes to trigger, on any platform.
    let mut previous_window: Option<usize> = None;
    let mut steady_state = None;
    for _ in 0..50 {
        let before = ALLOCS.load(Ordering::Relaxed);
        for _ in 0..1000 {
            sink += g.entry().descendants_in(&mut sc).count();
        }
        let window = ALLOCS.load(Ordering::Relaxed) - before;
        if previous_window == Some(window) {
            steady_state = Some(window);
            break;
        }
        previous_window = Some(window);
    }
    let steady_state = steady_state.unwrap_or_else(|| {
        panic!("descendants_in's allocations never stabilized across 50 windows of 1000 calls each")
    });

    // Stabilizing is necessary but not sufficient: a regression to one
    // allocation per call also stabilizes, at 1000, and would still clear
    // the `descendants()` comparison below (which lands near 2000, since
    // that variant allocates both a stack and a bitmap every call). The
    // property this test exists for is *zero*, so require it -- the loop
    // above is only there to let a platform's one-time allocator setup
    // land somewhere before the window that has to be clean.
    assert_eq!(
        steady_state, 0,
        "descendants_in should settle at zero allocations per call, not {steady_state} per 1000"
    );

    // The allocating variant, for contrast.
    let before = ALLOCS.load(Ordering::Relaxed);
    for _ in 0..1000 {
        sink += g.entry().descendants().count();
    }
    let allocating = ALLOCS.load(Ordering::Relaxed) - before;
    assert!(
        allocating > steady_state + 100,
        "descendants() (fresh alloc per call) should cost far more than descendants_in \
         (reused scratch): allocating={allocating}, steady_state={steady_state}"
    );
    assert!(sink > 0);
}
