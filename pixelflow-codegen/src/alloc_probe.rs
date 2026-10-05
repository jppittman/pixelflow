//! A counting allocator for this crate's own tests: how many bytes a piece of
//! code requests, in how many allocations, and the most it holds at once.
//!
//! A term that is "linear in the folds" is a claim about memory as much as
//! time, and a scratch table sized to the whole program once per fold is
//! quadratic in a way no pixel and no byte of machine code can show. This is
//! the instrument that can, written once so each claim a later change makes
//! about allocation is a [`measure`] call and an assertion rather than a new
//! allocator.
//!
//! # What is counted
//!
//! Only what the calling thread allocates **inside** a [`measure`] scope.
//! `cargo test` runs this binary's tests on many threads at once, and the
//! allocator is process-global, so the tally is thread-local: another test
//! allocating on another thread is neither counted nor disturbed, and a
//! thread the measured code spawns is not counted either.
//!
//! The tally is const-initialized, `Copy` and has no destructor, so reading
//! or writing it never allocates and never registers a thread-exit callback,
//! which is what lets it live inside an allocator at all. An access from a
//! thread that is already tearing down its thread-locals counts nothing
//! rather than panicking inside `alloc`.
//!
//! # What the numbers mean
//!
//! - `requested`: the sum of every `Layout::size` handed to the allocator in
//!   the scope. A growing `Vec` is an allocation of its new capacity and a
//!   free of its old one (the trait's default `realloc`, deliberately not
//!   overridden, so the answer does not depend on whether the system
//!   allocator could extend a block in place). This is the figure a
//!   `vec![0; n]`-per-fold term shows up in: it is the bytes asked for, and
//!   an upper bound on the bytes zeroed.
//! - `allocations`: how many times.
//! - `peak`: the largest **net growth** of the live heap over the scope's
//!   start: bytes allocated in the scope minus bytes freed in it, at its
//!   highest. A block that predates the scope and is freed inside it lowers
//!   the running figure like any other free, so a scope that consumes a large
//!   input it was handed can under-report its own peak by up to that input's
//!   size; build the input inside the scope when it must be counted.
//!   This is the same definition `pixelflow_pipeline::alloc_probe` gives the
//!   measurement bins, so the two read alike.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

/// The one thread's running count.
#[derive(Clone, Copy)]
struct Tally {
    counting: bool,
    requested: usize,
    allocations: usize,
    /// Net growth so far: signed, because a scope may free more than it
    /// allocated.
    live: isize,
    peak: usize,
}

impl Tally {
    const IDLE: Self = Self {
        counting: false,
        requested: 0,
        allocations: 0,
        live: 0,
        peak: 0,
    };

    const STARTED: Self = Self {
        counting: true,
        ..Self::IDLE
    };
}

thread_local! {
    // `const` init and no `Drop`: no lazy-initialization branch that could
    // allocate, no destructor to register.
    static TALLY: Cell<Tally> = const { Cell::new(Tally::IDLE) };
}

/// Apply `change` to this thread's tally if it is counting.
fn record(change: impl FnOnce(&mut Tally)) {
    // A thread whose thread-locals are gone has no scope to be inside of.
    TALLY
        .try_with(|cell| {
            let mut tally = cell.get();
            if tally.counting {
                change(&mut tally);
                cell.set(tally);
            }
        })
        .unwrap_or_default();
}

/// The allocator: the system's, plus [`record`]. Installed for this crate's
/// test binary only, by the `#[global_allocator]` below.
pub(crate) struct Probe;

// SAFETY: every method forwards to `System` with the arguments it was given
// and returns what `System` returned, so the allocator contract is `System`'s;
// the counting beside it reads and writes a const thread-local `Cell` and
// calls nothing that allocates.
unsafe impl GlobalAlloc for Probe {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded unchanged.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            grew(layout);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // Overridden so a zeroed block stays the system's lazily-mapped one:
        // the default would allocate and then write every byte, and a probe
        // should not make a large `vec![0; n]` touch its pages.
        // SAFETY: forwarded unchanged.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            grew(layout);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded unchanged.
        unsafe { System.dealloc(ptr, layout) };
        record(|tally| tally.live -= layout.size() as isize);
    }
}

/// One block of `layout` came into being.
fn grew(layout: Layout) {
    record(|tally| {
        tally.requested += layout.size();
        tally.allocations += 1;
        tally.live += layout.size() as isize;
        tally.peak = tally.peak.max(tally.live.max(0) as usize);
    });
}

#[global_allocator]
static PROBE: Probe = Probe;

/// What one [`measure`] scope allocated. See the module docs for each field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Measured {
    /// Bytes requested, summed over every allocation.
    pub requested: usize,
    /// Allocations made.
    pub allocations: usize,
    /// Peak net growth of the live heap over the scope's start.
    pub peak: usize,
}

/// Stops the tally when dropped, so a scope that panics leaves its thread
/// counting nothing rather than counting the rest of the test.
struct Counting;

impl Drop for Counting {
    fn drop(&mut self) {
        TALLY.with(|cell| {
            let mut tally = cell.get();
            tally.counting = false;
            cell.set(tally);
        });
    }
}

/// Run `scope` and report what the **calling thread** allocated while it ran,
/// beside what it returned.
///
/// Not reentrant: a scope inside a scope has no second tally to count into,
/// and splitting one would hide exactly the bytes the outer one is for.
///
/// # Panics
///
/// If called from inside another `measure` scope on the same thread.
pub(crate) fn measure<R>(scope: impl FnOnce() -> R) -> (R, Measured) {
    TALLY.with(|cell| {
        assert!(
            !cell.get().counting,
            "alloc_probe::measure is not reentrant: this thread is already inside a scope"
        );
        cell.set(Tally::STARTED);
    });
    let counting = Counting;
    let result = scope();
    drop(counting);
    let tally = TALLY.with(Cell::get);
    let measured = Measured {
        requested: tally.requested,
        allocations: tally.allocations,
        peak: tally.peak,
    };
    (result, measured)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::{Duration, Instant};

    /// Threads allocating and freeing throughout the measured scope.
    const NOISY_THREADS: usize = 4;
    /// A size no background thread allocates, so a leak of their traffic into
    /// the tally could not be mistaken for the scope's own.
    const KNOWN: usize = 4096;
    /// What each noisy allocation asks for.
    const NOISE: usize = 1 << 16;
    /// How long the scope waits for the noisy threads before calling the test
    /// broken rather than hanging a CI job on a thread that died.
    const NOISE_DEADLINE: Duration = Duration::from_secs(30);

    /// **A known allocation reads as exactly its bytes, while other threads
    /// allocate throughout.**
    ///
    /// The noisy threads are started before the scope and joined after it. The
    /// scope does not begin its allocation until every one of them has
    /// finished a round of its own, and they keep looping until it is over, so
    /// their traffic is certain to overlap the measurement rather than
    /// assumed to. Were the tally process-global they would be counted; were
    /// the allocator to disturb them, they would fail the lengths they check
    /// themselves.
    #[test]
    fn a_known_allocation_reads_as_its_bytes_while_other_threads_allocate() {
        let stop = Arc::new(AtomicBool::new(false));
        let ready = Arc::new(Barrier::new(NOISY_THREADS + 1));
        // How many noisy threads have finished a first round: the scope waits
        // for all of them, so "the noise ran" is a precondition, not a hope.
        let warmed = Arc::new(AtomicUsize::new(0));
        let noisy: Vec<_> = (0..NOISY_THREADS)
            .map(|_| {
                let (stop, ready, warmed) = (stop.clone(), ready.clone(), warmed.clone());
                thread::spawn(move || {
                    ready.wait();
                    let mut rounds = 0usize;
                    while !stop.load(Ordering::Relaxed) {
                        let block = black_box(vec![1u8; NOISE]);
                        assert_eq!(block.len(), NOISE);
                        rounds += 1;
                        if rounds == 1 {
                            warmed.fetch_add(1, Ordering::Release);
                        }
                    }
                    rounds
                })
            })
            .collect();
        ready.wait();

        let (kept, measured) = measure(|| {
            // Spinning allocates nothing, so it adds nothing to the tally.
            let deadline = Instant::now() + NOISE_DEADLINE;
            while warmed.load(Ordering::Acquire) < NOISY_THREADS {
                assert!(
                    Instant::now() < deadline,
                    "the noisy threads never started, so this proved nothing"
                );
                thread::yield_now();
            }
            black_box(Vec::<u8>::with_capacity(KNOWN))
        });
        // Past the scope, so neither the keeping nor the freeing is counted.
        assert!(kept.capacity() >= KNOWN);
        drop(kept);

        stop.store(true, Ordering::Relaxed);
        let rounds: usize = noisy
            .into_iter()
            .map(|t| t.join().expect("a noisy thread finished"))
            .sum();
        assert!(rounds > 0, "the noise never ran, so this proved nothing");
        assert_eq!(
            measured,
            Measured {
                requested: KNOWN,
                allocations: 1,
                peak: KNOWN
            }
        );
    }

    /// The peak is a high-water mark of what is held, not a total: a block
    /// freed before the next is allocated is not held with it.
    #[test]
    fn the_peak_is_what_is_held_at_once() {
        const FIRST: usize = 1000;
        const SECOND: usize = 3000;
        const THIRD: usize = 500;
        let ((), measured) = measure(|| {
            let first = black_box(Vec::<u8>::with_capacity(FIRST));
            let second = black_box(Vec::<u8>::with_capacity(SECOND));
            drop(first);
            drop(second);
            // Held alone, after both are gone.
            drop(black_box(Vec::<u8>::with_capacity(THIRD)));
        });
        assert_eq!(
            measured,
            Measured {
                requested: FIRST + SECOND + THIRD,
                allocations: 3,
                peak: FIRST + SECOND
            }
        );
    }

    /// Nothing is counted outside a scope, and a scope that panics does not
    /// leave its thread counting.
    #[test]
    fn a_panicking_scope_stops_counting() {
        let outcome = std::panic::catch_unwind(|| {
            measure(|| {
                drop(black_box(Vec::<u8>::with_capacity(KNOWN)));
                panic!("the scope fails");
            })
        });
        assert!(outcome.is_err());
        let ((), after) = measure(|| ());
        assert_eq!(
            after,
            Measured {
                requested: 0,
                allocations: 0,
                peak: 0
            },
            "a fresh scope starts from zero, and the failed one stopped counting"
        );
    }
}
