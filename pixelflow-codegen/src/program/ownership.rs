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
//! the `If`, read it, it belongs to the region around the `If`. A scope's
//! roots are read by the scope itself, so no arm owns one — a branch skipping
//! the arm would leave the root unwritten for a loop that runs regardless.
//!
//! That is the whole relation. It is computed in one reverse pass over the
//! schedule: every reader of a value comes after it, so by the time the pass
//! reaches a value every region that reads it is known.
//!
//! One pass, O(reads × log depth): the lowest common ancestor is a climb on
//! skip pointers, so a ladder of `else if`s a thousand deep costs what a flat
//! scope does.
//!
//! This replaced a search. The old analysis recovered the same sets per `If`
//! by closing a cone under "every consumer is in the set", which is a walk of
//! the whole scope for every `If`. The two were asserted equal on every compile
//! (and on 300 random DAGs) until the search was deleted; what keeps this pass
//! honest now is its own tests, which pin the relation case by case, and the
//! layout built on it, whose every branch is checked to be exactly the run of
//! the values its arm owns.

use alloc::vec::Vec;

use pixelflow_ir::kind::OpKind;
use pixelflow_search::egraph::CostModel;

use crate::program::guards::{FoldReads, def_cycles};
use crate::program::tree::Tree;
use crate::program::{Def, IfArm, ScheduledOp, ValueId};

/// A region of one scope: the scope itself, or one arm of one `If` in it.
///
/// Regions are numbered in the order the reverse pass meets them, so a
/// parent's number is always smaller than its children's.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Region(pub(crate) usize);

impl Region {
    /// The scope itself: what no arm owns.
    pub(crate) const SCOPE: Self = Self(0);
}

/// One arm of one `If`, as a region.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Arm {
    /// Schedule position of the `If` (in the order the ownership was read
    /// from; positions are only names for defs here, the relation does not
    /// depend on them).
    pub(crate) if_pos: usize,
    /// The `If`'s mask.
    pub(crate) mask: ValueId,
    /// Which arm.
    pub(crate) arm: IfArm,
    /// The region the arm's values belong to.
    pub(crate) region: Region,
    /// Latency-prior cycles of everything the arm owns, nested arms and the
    /// folds it owns included: what skipping the arm saves, and so what a
    /// branch over it can pay for.
    pub(crate) cycles: usize,
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
pub(crate) struct Positions(Vec<Option<usize>>);

impl Positions {
    pub(crate) fn of(schedule: &[Def]) -> Self {
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

    pub(crate) fn get(&self, value: ValueId) -> Option<usize> {
        self.0.get(value.0 as usize).copied().flatten()
    }
}

impl Ownership {
    /// The ownership of `schedule`.
    ///
    /// `external` are the values read from outside the schedule — a scope's
    /// roots, read by the loops inside it — and `folds` is what each loop the
    /// scope opens reads from it, which makes the loop's `Reduce` def a reader
    /// of each of those values (`FoldReads`). Both are the old analysis'
    /// inputs, so the two answer the same question.
    pub(crate) fn of(schedule: &[Def], external: &[ValueId], folds: &FoldReads) -> Self {
        let positions = Positions::of(schedule);
        let cycles = CostModel::latency_prior();
        let mut me = Self {
            region_of: alloc::vec![Region::SCOPE; schedule.len()],
            regions: Tree::rooted(),
            arms: Vec::new(),
        };
        // The region every reader of each position agrees on so far.
        let mut readers: Vec<Option<Region>> = alloc::vec![None; schedule.len()];

        for root in external {
            if let Some(pos) = positions.get(*root) {
                readers[pos] = Some(Region::SCOPE);
            }
        }

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
                ScheduledOp::Ternary(OpKind::If, mask, if_true, if_false) => {
                    let on_true = Region(me.regions.grow(here.0));
                    let on_false = Region(me.regions.grow(here.0));
                    // Pushed false arm first: the sweep meets the `If`s
                    // last-to-first and the list is reversed once at the end.
                    me.arms.push(Arm {
                        if_pos: pos,
                        mask: *mask,
                        arm: IfArm::False,
                        region: on_false,
                        cycles: 0,
                    });
                    me.arms.push(Arm {
                        if_pos: pos,
                        mask: *mask,
                        arm: IfArm::True,
                        region: on_true,
                        cycles: 0,
                    });
                    read(&mut me, *mask, here);
                    read(&mut me, *if_true, on_true);
                    read(&mut me, *if_false, on_false);
                }
                op => {
                    for value in folds.reads(def.value, op) {
                        read(&mut me, value, here);
                    }
                }
            }
        }

        me.price(schedule, folds, &cycles);
        me.arms.reverse();
        me
    }

    /// Each arm's price: the cycles of every def in its region or one nested
    /// in it. A child's number is larger than its parent's, so one descending
    /// sweep folds every region into its parent.
    fn price(&mut self, schedule: &[Def], folds: &FoldReads, cycles: &CostModel) {
        let mut subtree = alloc::vec![0usize; self.regions.len()];
        for (pos, def) in schedule.iter().enumerate() {
            let region = self.region_of[pos].0;
            subtree[region] = subtree[region].saturating_add(def_cycles(def, folds, cycles));
        }
        for region in (1..subtree.len()).rev() {
            let parent = self.regions.parent(region);
            subtree[parent] = subtree[parent].saturating_add(subtree[region]);
        }
        for arm in &mut self.arms {
            arm.cycles = subtree[arm.region.0];
        }
    }

    /// Every arm of every `If`, in schedule order, an `If`'s true arm first.
    pub(crate) fn arms(&self) -> &[Arm] {
        &self.arms
    }

    /// The region the value at `pos` belongs to.
    pub(crate) fn region_of(&self, pos: usize) -> Region {
        self.region_of[pos]
    }

    /// The regions, for a stage that nests something in them.
    pub(crate) fn regions(&self) -> &Tree {
        &self.regions
    }

    #[cfg(test)]
    /// Whether `inner` is `outer` or nested in it.
    pub(crate) fn is_within(&self, inner: Region, outer: Region) -> bool {
        self.regions.is_within(inner.0, outer.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn def(value: u64, op: ScheduledOp) -> Def {
        Def {
            value: ValueId(value),
            op,
        }
    }

    fn if_of(value: u64, mask: u64, if_true: u64, if_false: u64) -> Def {
        def(
            value,
            ScheduledOp::Ternary(
                OpKind::If,
                ValueId(mask),
                ValueId(if_true),
                ValueId(if_false),
            ),
        )
    }

    /// The positions an arm owns, nested arms included.
    fn owned(own: &Ownership, arm: &Arm, len: usize) -> Vec<usize> {
        (0..len)
            .filter(|&pos| own.is_within(own.region_of(pos), arm.region))
            .collect()
    }

    /// Only the true arm reads `Rsqrt(v1)`, so the true arm owns it; the mask
    /// and the false arm (the mask again) own nothing.
    #[test]
    fn an_arm_owns_what_only_it_reads() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Unary(OpKind::Rsqrt, ValueId(1))),
            if_of(3, 0, 2, 0),
        ];
        let own = Ownership::of(&schedule, &[], &FoldReads::default());
        let [on_true, on_false] = own.arms() else {
            panic!("one If is two arms");
        };
        assert_eq!((on_true.arm, on_false.arm), (IfArm::True, IfArm::False));
        assert_eq!(owned(&own, on_true, schedule.len()), alloc::vec![1, 2]);
        assert_eq!(owned(&own, on_false, schedule.len()), Vec::<usize>::new());
    }

    /// A value both arms read belongs to the region around the `If`, and so
    /// does the mask's cone.
    #[test]
    fn a_value_both_arms_read_belongs_to_neither() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Unary(OpKind::Rsqrt, ValueId(1))),
            def(3, ScheduledOp::Unary(OpKind::Neg, ValueId(2))),
            def(4, ScheduledOp::Unary(OpKind::Sqrt, ValueId(2))),
            if_of(5, 0, 3, 4),
        ];
        let own = Ownership::of(&schedule, &[], &FoldReads::default());
        let [on_true, on_false] = own.arms() else {
            panic!("one If is two arms");
        };
        assert_eq!(owned(&own, on_true, schedule.len()), alloc::vec![3]);
        assert_eq!(owned(&own, on_false, schedule.len()), alloc::vec![4]);
        assert_eq!(own.region_of(2), Region::SCOPE);
    }

    /// A scope's root is read by the scope itself, so no arm owns it, however
    /// many of its readers sit inside one.
    #[test]
    fn no_arm_owns_a_root() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Unary(OpKind::Rsqrt, ValueId(1))),
            if_of(3, 0, 2, 0),
        ];
        let own = Ownership::of(&schedule, &[ValueId(2)], &FoldReads::default());
        let [on_true, _] = own.arms() else {
            panic!("one If is two arms");
        };
        assert_eq!(owned(&own, on_true, schedule.len()), Vec::<usize>::new());
        assert_eq!(own.region_of(2), Region::SCOPE);
    }

    /// An arm that owns an `If` owns what that `If`'s arms own, and prices
    /// them: the outer true arm costs everything nested in it.
    #[test]
    fn a_nested_arm_is_owned_by_the_arm_around_it() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Unary(OpKind::Rsqrt, ValueId(1))),
            def(3, ScheduledOp::Unary(OpKind::Sqrt, ValueId(1))),
            if_of(4, 0, 2, 3),
            def(5, ScheduledOp::Unary(OpKind::Recip, ValueId(1))),
            if_of(6, 0, 4, 5),
        ];
        let own = Ownership::of(&schedule, &[], &FoldReads::default());
        let [inner_true, inner_false, outer_true, outer_false] = own.arms() else {
            panic!("two Ifs are four arms, in schedule order");
        };
        // The inner If sits at position 4, the outer at 6.
        assert_eq!((inner_true.if_pos, outer_true.if_pos), (4, 6));
        assert_eq!(owned(&own, inner_true, schedule.len()), alloc::vec![2]);
        assert_eq!(owned(&own, inner_false, schedule.len()), alloc::vec![3]);
        assert_eq!(
            owned(&own, outer_true, schedule.len()),
            alloc::vec![2, 3, 4],
            "the inner If and everything under it belongs to the arm it is the root of"
        );
        assert_eq!(owned(&own, outer_false, schedule.len()), alloc::vec![5]);
        assert!(outer_true.cycles > inner_true.cycles);
    }

    /// The relation is a property of the reads, so reordering the schedule
    /// within what the reads allow changes no arm's membership.
    #[test]
    fn the_order_of_the_schedule_does_not_change_what_is_owned() {
        let ops = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Unary(OpKind::Rsqrt, ValueId(1))),
            def(3, ScheduledOp::Unary(OpKind::Sqrt, ValueId(1))),
            def(4, ScheduledOp::Unary(OpKind::Neg, ValueId(3))),
            if_of(5, 0, 2, 4),
        ];
        // The same defs with the two arms' work interleaved the other way.
        let swapped = alloc::vec![
            ops[0].clone(),
            ops[1].clone(),
            ops[3].clone(),
            ops[2].clone(),
            ops[4].clone(),
            ops[5].clone(),
        ];
        let by_value = |schedule: &[Def]| {
            let own = Ownership::of(schedule, &[], &FoldReads::default());
            own.arms()
                .iter()
                .map(|arm| {
                    let mut values: Vec<u64> = (0..schedule.len())
                        .filter(|&pos| own.is_within(own.region_of(pos), arm.region))
                        .map(|pos| schedule[pos].value.0)
                        .collect();
                    values.sort_unstable();
                    (arm.arm, values, arm.cycles)
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(by_value(&ops), by_value(&swapped));
    }
}
