//! What the emitter actually emitted, counted per scope.
//!
//! Three allocator policies have now been built and rejected on a static
//! quantity — memory operations, code bytes — that turned out not to predict
//! wall clock (`docs/archive/plans/2026-09-01-register-allocation-escape-hatches.md`,
//! the 2026-09-04 blocks). Fitting a cost model to time needs the static
//! features and the time side by side, per kernel, per allocation, per tier,
//! and this is the static half: the counts the driver produces on its way to
//! machine code, attributed to the scope that executes them, so a trip count
//! can weight them afterwards.
//!
//! It is a *count*, never a decision. Nothing in the emitter or the allocator
//! reads a [`ScopeTraffic`]; it rides out on [`CompileResult`](super::CompileResult)
//! for a measurement harness to record. That is deliberate — an allocator that
//! optimized this number is exactly the thing the measurements above refused.
//!
//! The counting is done by [`EmitTraffic::of`] over the allocated blocks: each
//! instruction the allocator inserted carries the reason it did
//! ([`Origin`]), so an instruction is counted by why it exists and nothing
//! decorates an encoder.

use super::asm::Assembled;
use super::regalloc::{Allocated, Origin};
use super::{FileId, IsaBackend};
use crate::program::Scope;
use alloc::vec::Vec;
use core::fmt;
use pixelflow_ir::fold::{Fold, KnownRange};

/// Emitted traffic within one scope of the nest.
///
/// A scope's counts are *static*: what the scope's code contains, not what a
/// call executes. Multiply by the scope's trip count for the dynamic figure —
/// which is the whole reason the split by scope exists, since an instruction
/// in the column fold runs `rows × batches` times and one in the body once.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScopeTraffic {
    /// Scheduled operations emitted. A constant counts only when the
    /// instruction selected for it is not rematerializable. A
    /// rematerializable one (every AVX2 constant) is placed where it is read,
    /// and counted there: in `remats`, when it reads the pool.
    pub instructions: u64,
    /// Stack loads emitted: an instruction's operand reloads, a range brought
    /// back into a register, a scope head's reconciliation, a guard's mask, a
    /// fold's slot-held root.
    pub loads: u64,
    /// Reads of the kernel's constant pool: a constant brought into a register
    /// by a memory read, whether the instruction folds the operand or a load
    /// brings it in. What a trip
    /// executes is the point, so a zero or all-ones idiom, or an immediate,
    /// is not counted: it reads no memory. A read of the pool, but not of a
    /// slot this kernel wrote, which is why it is counted apart from the loads
    /// and the stores.
    pub remats: u64,
    /// Stack stores emitted: spills, parks, a fold's slot-held roots.
    pub stores: u64,
    /// Bytes of machine code the scope's own instructions occupy, excluding
    /// the scopes nested inside it, so every byte lands in exactly one scope.
    pub bytes: u64,
}

impl ScopeTraffic {
    /// Loads plus stores — the quantity #1150's table reports, and the one
    /// the 2026-09-04 measurements found does not predict AVX-512 time.
    #[must_use]
    const fn memory_ops(&self) -> u64 {
        self.loads + self.stores
    }
}

/// How many times a loop's body runs per call, and whether that is a count.
///
/// A fold whose bounds are literals runs a number of times the program states.
/// One whose bound is a value runs a number the call decides, which nothing
/// here knows: the static model prices it at [`Fold::ESTIMATED_TRIPS`], and
/// that price is carried as an estimate, so that no measurement can read it as
/// a count. A loop nested in an estimate is an estimate too, for it runs that
/// many times that many times, and so is a nest of literals whose product
/// passes `u64`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trips {
    /// Every bound is a literal: the body runs exactly this many times.
    Known(u64),
    /// A bound is a value, here or in an enclosing fold, or the nest's product
    /// passes `u64`: the body is priced as running this many times, and no
    /// count is claimed.
    Estimated(u64),
}

impl Trips {
    /// What the body of a call runs: once.
    pub(in crate::emit) const ONCE: Self = Self::Known(1);

    /// The trips of a fold whose range is `range` (`None` when a bound is a
    /// value), on their own and not yet nested in any loop.
    pub(in crate::emit) fn of(range: Option<KnownRange>) -> Self {
        match range {
            Some(range) => Self::Known(range.len()),
            None => Self::Estimated(Fold::ESTIMATED_TRIPS),
        }
    }

    /// These trips of a loop inside `parent`: their product, an estimate if
    /// either is. A product past `u64` is a weight, not a count to report, so
    /// it is an estimate at `u64::MAX`; a weight is a price to add up, and the
    /// allocator's sums of them saturate.
    pub(in crate::emit) fn nested_in(self, parent: Self) -> Self {
        let product = parent.weight().checked_mul(self.weight());
        match (self, parent, product) {
            (Self::Known(_), Self::Known(_), Some(count)) => Self::Known(count),
            (_, _, product) => Self::Estimated(product.unwrap_or(u64::MAX)),
        }
    }

    /// The number the allocator weighs a read in the loop by: the count, or
    /// the estimate's price. A price, not a count: the allocator adds these up
    /// and its sums saturate.
    pub(in crate::emit) fn weight(self) -> u64 {
        match self {
            Self::Known(n) | Self::Estimated(n) => n,
        }
    }

    /// The count, when it is one.
    #[must_use]
    pub fn known(self) -> Option<u64> {
        match self {
            Self::Known(n) => Some(n),
            Self::Estimated(_) => None,
        }
    }
}

/// A known count prints as the bare number, with the formatter's width and
/// fill honoured; an estimate prints as `~n`.
impl fmt::Display for Trips {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Known(n) => fmt::Display::fmt(n, f),
            Self::Estimated(n) => f.pad(&alloc::format!("~{n}")),
        }
    }
}

/// The whole nest's traffic, plus the target facts a cost model needs to
/// price it (a 64-byte spill is not a 16-byte one).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EmitTraffic {
    /// One entry per scope, the body first and then the folds in nest order
    /// (`Scope::Fold(j)` at `j + 1`) — the lattice's row and column folds
    /// among them, since they are folds like any other.
    pub scopes: Vec<ScopeTraffic>,
    /// How many times one call runs each scope, indexed like `scopes`: the
    /// body once, a fold its trip count times its parent's. A fold whose range
    /// is not known, and every fold inside it, is an estimate.
    pub trips: Vec<Trips>,
    /// The function around the nest: its frame, the body's place in it and
    /// the return. The same code under every allocation of a kernel on a
    /// given target, so it cannot explain a difference between two
    /// allocations — recorded separately rather than folded into a scope so
    /// that stays visible.
    pub scaffold: ScopeTraffic,
    /// Bytes one spilled register occupies: the backend's vector width.
    pub vector_bytes: u32,
    /// Vector registers the allocator hands out: the size of its pool. The
    /// selection pipeline leases the whole file, and counts it so.
    pub pool: u8,
    /// Parked roots that hold a register across the scopes inside them
    /// rather than a slot.
    pub carried: u64,
}

impl EmitTraffic {
    /// What an allocated kernel contains, read off its blocks: each scope's
    /// instructions as the driver tallied them, its loads and stores by why the
    /// allocator inserted them, and its bytes between its blocks' addresses.
    /// The function's frame and return are the body's instructions here, so
    /// the scaffold is empty and nothing is carried. The pool is the whole
    /// vector file, which the allocator leases from.
    #[must_use]
    pub(super) fn of<B: IsaBackend>(allocated: &Allocated<'_, B>, assembled: &Assembled) -> Self {
        let mut scopes = alloc::vec![ScopeTraffic::default(); allocated.scheduled.len()];
        for (scope, &instructions) in scopes.iter_mut().zip(&allocated.scheduled) {
            scope.instructions = instructions;
        }
        let ends = allocated
            .blocks
            .iter()
            .skip(1)
            .map(|block| block.label)
            .chain([allocated.text_end]);
        let mut trips = alloc::vec![Trips::ONCE; scopes.len()];
        for (block, end) in allocated.blocks.iter().zip(ends) {
            let scope = &mut scopes[scope_ix(block.scope)];
            scope.bytes += (assembled.address(end) - assembled.address(block.label)) as u64;
            for emitted in &block.insts {
                match emitted.origin {
                    Origin::Selected | Origin::Copy => {}
                    Origin::Remat => scope.remats += u64::from(B::reads_pool(&emitted.inst)),
                    Origin::Reload => scope.loads += 1,
                    Origin::Spill => scope.stores += 1,
                }
            }
            // A loop's head is the first block of its fold's scope.
            if let Some(looped) = allocated.loops.iter().find(|l| l.head == block.label) {
                trips[scope_ix(block.scope)] = looped.trips;
            }
        }
        Self {
            scopes,
            trips,
            scaffold: ScopeTraffic::default(),
            vector_bytes: B::FILE.vector_bytes() as u32,
            pool: u8::try_from(B::FILE.members(FileId::Vector).len())
                .expect("a vector file has far fewer than 256 registers"),
            carried: 0,
        }
    }

    /// The body's traffic: what runs once per call.
    #[must_use]
    pub fn body(&self) -> ScopeTraffic {
        self.scopes.first().copied().unwrap_or_default()
    }

    /// Memory operations one call executes: each scope's, weighted by how
    /// many times the call runs it. `None` when any scope's trips are an
    /// estimate, so that no measurement shows an estimate as a count.
    ///
    /// The scaffold is excluded: it is the same code under every allocation of
    /// a kernel, so including it only adds a constant to both sides of every
    /// comparison this number exists to make.
    #[must_use]
    pub fn dynamic_memory_ops(&self) -> Option<u64> {
        self.scopes
            .iter()
            .zip(&self.trips)
            .map(|(s, trips)| Some(s.memory_ops() * trips.known()?))
            .sum()
    }
}

/// The index a scope's count is kept under: the body first, then the folds.
pub(super) fn scope_ix(scope: Scope) -> usize {
    match scope {
        Scope::Body => 0,
        Scope::Fold(j) => j + 1,
    }
}

#[cfg(test)]
mod tests {
    use super::{EmitTraffic, ScopeTraffic, Trips};

    /// `memory_ops` is read by a measurement harness, not by anything this
    /// crate itself branches on, so a wrong formula would ship silently
    /// unless a test pins the exact arithmetic against values that cannot
    /// agree by coincidence.
    #[test]
    fn memory_ops_sums_loads_and_stores() {
        let traffic = ScopeTraffic {
            loads: 8,
            stores: 7,
            ..ScopeTraffic::default()
        };
        assert_eq!(traffic.memory_ops(), 15);
    }

    /// Same reasoning as `memory_ops` above, one level up: each scope is
    /// weighted by its own trip count, so the test's inputs are chosen so
    /// every wrong weighting or wrong operator lands on a different total.
    #[test]
    fn dynamic_memory_ops_weights_each_scope_by_its_trip_count() {
        let body = ScopeTraffic {
            loads: 1,
            ..ScopeTraffic::default()
        };
        let rows = ScopeTraffic {
            loads: 2,
            stores: 1,
            ..ScopeTraffic::default()
        };
        let cols = ScopeTraffic {
            loads: 4,
            stores: 1,
            ..ScopeTraffic::default()
        };
        let traffic = EmitTraffic {
            scopes: alloc::vec![body, rows, cols],
            trips: alloc::vec![Trips::Known(1), Trips::Known(6), Trips::Known(42)],
            ..EmitTraffic::default()
        };

        assert_eq!(traffic.dynamic_memory_ops(), Some(1 + 3 * 6 + 5 * 42));
    }

    /// A scope whose trips are an estimate has no count of memory operations,
    /// and neither has the kernel: the sum is `None`, not a number a reader
    /// could take for one.
    #[test]
    fn dynamic_memory_ops_is_no_count_over_an_estimate() {
        let scope = ScopeTraffic {
            loads: 1,
            ..ScopeTraffic::default()
        };
        let traffic = EmitTraffic {
            scopes: alloc::vec![scope, scope],
            trips: alloc::vec![Trips::Known(1), Trips::Estimated(32)],
            ..EmitTraffic::default()
        };

        assert_eq!(traffic.dynamic_memory_ops(), None);
    }

    /// `byte_probe` prints `trips={trips}` and `constant_traffic` a `{:>8}`
    /// column: a known count must print as it did when it was a `u64`.
    #[test]
    fn a_known_trip_count_prints_as_its_number() {
        assert_eq!(alloc::format!("{}", Trips::Known(42)), "42");
        assert_eq!(alloc::format!("{:>5}", Trips::Known(42)), "   42");
    }

    #[test]
    fn an_estimate_prints_marked() {
        assert_eq!(alloc::format!("{}", Trips::Estimated(32)), "~32");
        assert_eq!(alloc::format!("{:>5}", Trips::Estimated(32)), "  ~32");
    }

    /// A loop nested in a loop whose count is not known runs a number of times
    /// that is not known either, and carries the product.
    #[test]
    fn nesting_an_estimate_is_an_estimate() {
        assert_eq!(
            Trips::Estimated(32).nested_in(Trips::Known(4)),
            Trips::Estimated(128)
        );
        assert_eq!(
            Trips::Known(4).nested_in(Trips::Estimated(32)),
            Trips::Estimated(128)
        );
        assert_eq!(Trips::Known(3).nested_in(Trips::Known(4)), Trips::Known(12));
    }

    /// A product of literal trips past `u64` is not a count: it prices at the
    /// most the allocator can weigh, and `dynamic_memory_ops` does not report it.
    #[test]
    fn a_product_past_u64_is_an_estimate() {
        let nested = Trips::Known(u64::MAX).nested_in(Trips::Known(2));
        assert_eq!(nested, Trips::Estimated(u64::MAX));
        assert_eq!(nested.known(), None);
    }
}
