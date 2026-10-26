//! Who a value is computed for: the arm of an `If` that owns it, read off the
//! DAG.
//!
//! An `If` has two cases, and a value computed for exactly one of them is
//! dead weight whenever the other is taken. Which values those are is a fact
//! about the **DAG** — who reads whom — and no order of the schedule changes
//! it. So it is stated here once, as a function of the reads, instead of being
//! recovered from the order the values happen to sit in.
//!
//! The denotation. Every `If` opens two **regions**, one per arm, nested in
//! the region the `If` itself belongs to. The `If` reads its mask in its own
//! region and each arm's operand in that arm's region. A value belongs to the
//! **lowest common ancestor** of the regions that read it: if only one arm
//! reads it, it is that arm's; if both arms, or the mask, or anything outside
//! the `If`, read it, it belongs to the region around the `If`. A loop reads
//! what it reads at its `Reduce` ([`FoldReads`]), so a value only the loops in
//! one arm read is that arm's, and a batch that skips them skips it too.
//!
//! That is the whole relation. It is computed in one reverse pass over the
//! schedule: every reader of a value comes after it, so by the time the pass
//! reaches a value every region that reads it is known.
//!
//! One pass, O(reads × log depth): the lowest common ancestor is a climb on
//! skip pointers, so a ladder of `else if`s a thousand deep costs what a flat
//! scope does.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::error::CompileError;
use crate::pass::Pass;
use crate::program::tree::Tree;
use crate::program::{Def, IfArm, ScheduledOp, ScopedSchedule, ValueId, all_operands};

/// A region of one scope: the scope itself, or one arm of one `If` in it.
///
/// Regions are numbered in the order the reverse pass meets them, so a
/// parent's number is always smaller than its children's.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Region(pub(super) usize);

impl Region {
    /// The scope itself: what no arm owns.
    const SCOPE: Self = Self(0);
}

/// One arm of one `If`, as a region.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct Arm {
    /// The position of the `If` in the schedule.
    pub(super) at: usize,
    /// The `If`'s mask.
    pub(super) mask: ValueId,
    /// Which arm.
    pub(super) arm: IfArm,
    /// The region the arm's values belong to.
    pub(super) region: Region,
}

/// The backward pass: every scope's [`Ownership`], read off the nest.
///
/// Its own stage, between [`Nest`](super::Nest) and [`Layout`](super::Layout),
/// because ownership is a fact about the DAG that layout uses and neither
/// computes the other.
///
/// # Panics
///
/// If a scope is not closed ([`ScopedSchedule::assert_closed`]): ownership
/// sees a loop's reads only at its `Reduce`, so a read the nest did not name
/// would let an arm own a value a loop outside the arm reads.
pub(crate) struct Own;

impl Pass for Own {
    type Input = ScopedSchedule<()>;
    type Output = ScopedSchedule<Ownership>;

    fn run(self, scoped: ScopedSchedule<()>) -> Result<ScopedSchedule<Ownership>, CompileError> {
        Ok(scoped.analyze(|nest, scope| {
            nest.assert_closed(scope);
            let (schedule, _, ()) = nest.at(scope);
            let opened = nest
                .folds
                .iter()
                .filter(|fold| fold.parent == scope)
                .map(|fold| (fold.reduce, fold.reads().collect()))
                .collect();
            Ownership::of(schedule, &opened)
        }))
    }
}

/// What each loop a scope opens reads, keyed by the loop's `Reduce` def: the
/// ids its body names, which the scope's own schedule holds or an enclosing
/// scope does (`ScopeFold::reads`).
///
/// To [`operands`](crate::program::operands) a `Reduce` def reads its value
/// bounds, the seed and the end its loop reads (none for a literal fold), and
/// never its body: once `scopes::extract_folds` has carved a fold's body into
/// a scope of its own, the def is where the loop opens. But the loop *runs*
/// there, and its body reads this scope: a sibling fold's result from its
/// accumulator slot, a value invariant in the fold from the park this scope
/// leaves it in. Without these edges a guard could skip a fold whose result a
/// sibling fold's body still reads, or a value a fold reads, which the fold
/// then finds unwritten (`pixelflow-core/tests/guard_sibling_fold.rs` pins
/// both).
///
/// They take no walk of a body: a fold's schedule keeps its parent's `ValueId`s,
/// so what it reads from its parent is among the ids it names, and a fold
/// nested in it reads nothing its parent does not name
/// (`ScopedSchedule::assert_closed`, run before this). Every named id is an
/// edge, a leaf the body rebuilds for itself included, which over-approximates
/// (such a leaf is kept out of an arm the fold does not run in); one this scope
/// does not define is not a position here and is ignored.
pub(super) type FoldReads = BTreeMap<ValueId, Vec<ValueId>>;

/// Everything the def `def` reads: its operands of both register classes, and,
/// for a `Reduce` that opens a fold, what the fold reads.
pub(super) fn reads<'a>(folds: &'a FoldReads, def: &'a Def) -> impl Iterator<Item = ValueId> + 'a {
    let fold = folds.get(&def.value).into_iter().flatten().copied();
    all_operands(&def.op).chain(fold)
}

/// Ownership of one scope's schedule.
pub(crate) struct Ownership {
    /// The region each schedule position belongs to.
    region_of: Vec<Region>,
    /// The regions, nested as the `If`s are.
    regions: Tree,
    /// Every arm of every `If`, in schedule order, true arm first.
    arms: Vec<Arm>,
}

/// Where a value sits in the scope's schedule, by `ValueId`: dense, because
/// ids are handed out sequentially. Absent for a value this scope does not
/// define (a live-in from an enclosing scope).
pub(super) struct Positions(Vec<Option<usize>>);

impl Positions {
    pub(super) fn of(schedule: &[Def]) -> Self {
        let len = schedule
            .iter()
            .map(|def| def.value.0 as usize + 1)
            .max()
            .unwrap_or(0);
        let mut at = alloc::vec![None; len];
        for (pos, def) in schedule.iter().enumerate() {
            at[def.value.0 as usize] = Some(pos);
        }
        Self(at)
    }

    pub(super) fn get(&self, value: ValueId) -> Option<usize> {
        self.0.get(value.0 as usize).copied().flatten()
    }
}

impl Ownership {
    /// The ownership of `schedule`.
    ///
    /// `folds` is what each loop the scope opens reads from it, which makes
    /// the loop's `Reduce` def a reader of each of those values.
    pub(super) fn of(schedule: &[Def], folds: &FoldReads) -> Self {
        let positions = Positions::of(schedule);
        let mut me = Self {
            region_of: alloc::vec![Region::SCOPE; schedule.len()],
            regions: Tree::rooted(),
            arms: Vec::new(),
        };
        // The region every reader of each position agrees on so far.
        let mut readers: Vec<Option<Region>> = alloc::vec![None; schedule.len()];

        for (pos, def) in schedule.iter().enumerate().rev() {
            // A value nothing reads is the scope's own: the scope's result, or
            // an effect.
            let here = readers[pos].unwrap_or(Region::SCOPE);
            me.region_of[pos] = here;

            let mut read = |me: &mut Self, value: ValueId, region: Region| {
                let Some(at) = positions.get(value) else {
                    return;
                };
                readers[at] = Some(match readers[at] {
                    None => region,
                    Some(known) => Region(me.regions.common_ancestor(known.0, region.0)),
                });
            };

            match &def.op {
                ScheduledOp::If(mask, if_true, if_false) => {
                    let on_true = Region(me.regions.grow(here.0));
                    let on_false = Region(me.regions.grow(here.0));
                    // Pushed false arm first: the sweep meets the `If`s
                    // last-to-first and the list is reversed once at the end.
                    me.arms.push(Arm {
                        at: pos,
                        mask: *mask,
                        arm: IfArm::False,
                        region: on_false,
                    });
                    me.arms.push(Arm {
                        at: pos,
                        mask: *mask,
                        arm: IfArm::True,
                        region: on_true,
                    });
                    read(&mut me, *mask, here);
                    read(&mut me, *if_true, on_true);
                    read(&mut me, *if_false, on_false);
                }
                _ => {
                    for value in reads(folds, def) {
                        read(&mut me, value, here);
                    }
                }
            }
        }

        me.arms.reverse();
        me
    }

    /// Every arm of every `If`, in schedule order, an `If`'s true arm first.
    pub(super) fn arms(&self) -> &[Arm] {
        &self.arms
    }

    /// The region the value at `pos` belongs to.
    pub(super) fn region_of(&self, pos: usize) -> Region {
        self.region_of[pos]
    }

    /// The regions, for a stage that nests something in them.
    pub(super) fn regions(&self) -> &Tree {
        &self.regions
    }
}
