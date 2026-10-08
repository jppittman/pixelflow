//! Where each value goes, and which arms are one run: the order of a scope's
//! schedule, decided once from who owns what.
//!
//! A branch skips a *range*, so an arm earns one only if the values it owns
//! sit in one run, and a schedule written in the order a program was
//! constructed rarely puts them there: an arm's values are interleaved with
//! whatever else happened to be built between them. The order used to be
//! repaired after the fact, by a search over the schedule that asked, for each
//! `If`, which entries its arms own and partitioned the region around it —
//! the whole scope re-examined per `If`. Here the order is *chosen*, from the
//! ownership the DAG already states (`ownership`).
//!
//! **What earns a block.** An arm region whose price exceeds
//! [`MISPREDICT_PENALTY_CYCLES`] (strictly: an arm that costs exactly the
//! penalty cannot pay for its own branch) and whose mask is computed in this
//! scope. A cheaper arm is merged into the region around it: it is neither
//! branched over nor moved, it is simply part of its parent's values.
//!
//! **Where a block goes.** Each earning region becomes one *block*: its own
//! values, in the order they already had, with the earning blocks nested in it
//! spliced in. A block is placed in its parent block's sequence immediately
//! before the first of the parent's own values whose original position is at
//! or past the block's *anchor*, the latest of
//!
//! - the block's first original position (nothing moves earlier than it was),
//! - one past the original position of anything the block reads that the
//!   parent block computes itself (it must follow its inputs), and
//! - one past its mask.
//!
//! The `If` is one of the parent's own values and sits past every anchor of
//! its arms, so a block always lands ahead of its `If`, and the scope's last
//! value stays last. Blocks that share an anchor go in order of their `If`,
//! the true arm first.
//!
//! **Nothing else moves.** A block already one run after its mask has its
//! first position as its anchor and lands where it was, so a scope whose arms
//! are all in order is returned as it came: the identity, by construction and
//! not by a check that happens to hold.
//!
//! The order's validity is checked, not argued: the tests below check, over
//! hand-built and random schedules, that each arm a layout branches over is
//! exactly the run of the values that arm owns. Until the search it replaced
//! was deleted, the old analysis ran on every layout's output and had to find
//! the same runs.

use alloc::vec::Vec;

use crate::program::guards::{FoldReads, MISPREDICT_PENALTY_CYCLES};
use crate::program::ownership::{Ownership, Positions};
use crate::program::tree::Tree;
use crate::program::{ArmPair, Def, IfGuard, ValueId};

/// A scope's schedule order and the branches over it.
pub(crate) struct Layout {
    /// The old position of each new position: the schedule to emit is
    /// `order.map(|old| schedule[old])`.
    order: Vec<usize>,
    /// The new position of each old position: how a table of positions into
    /// the old schedule (a fold's `at`) is carried to the new one.
    pub(super) position: Vec<usize>,
    /// The `If`s with a branch, in new positions, ascending.
    pub(crate) guards: Vec<IfGuard>,
}

/// The blocks of one scope: the earning arm regions, nested as they are in the
/// region tree, and the scope itself as block 0.
struct Blocks {
    /// Nesting of the blocks. Numbered in region order, so a parent precedes
    /// its children.
    tree: Tree,
    /// The block each region's values go to: itself if it earns one, else
    /// the nearest enclosing region that does.
    of_region: Vec<usize>,
    /// The arm (an index into `Ownership::arms`) each block is, `None` for the
    /// scope.
    arm: Vec<Option<usize>>,
}

impl Blocks {
    fn of(own: &Ownership, positions: &Positions) -> Self {
        let regions = own.regions();
        let mut arm_of_region = alloc::vec![None; regions.len()];
        for (i, arm) in own.arms().iter().enumerate() {
            arm_of_region[arm.region.0] = Some(i);
        }
        let mut blocks = Self {
            tree: Tree::rooted(),
            of_region: alloc::vec![0; regions.len()],
            arm: alloc::vec![None],
        };
        // Parents are numbered before their children, so a region's enclosing
        // block is known by the time the region is reached.
        for (region, arm) in arm_of_region.iter().enumerate().skip(1) {
            let enclosing = blocks.of_region[regions.parent(region)];
            let earns = arm.filter(|&i| {
                let arm = &own.arms()[i];
                arm.cycles > MISPREDICT_PENALTY_CYCLES && positions.get(arm.mask).is_some()
            });
            blocks.of_region[region] = match earns {
                Some(i) => {
                    blocks.arm.push(Some(i));
                    blocks.tree.grow(enclosing)
                }
                None => enclosing,
            };
        }
        blocks
    }
}

impl Layout {
    /// The layout of `schedule`, from the ownership of its values.
    ///
    /// `external` and `folds` are as for [`Ownership::of`].
    pub(crate) fn of(schedule: &[Def], external: &[ValueId], folds: &FoldReads) -> Self {
        let own = Ownership::of(schedule, external, folds);
        Self::from(schedule, folds, &own)
    }

    fn from(schedule: &[Def], folds: &FoldReads, own: &Ownership) -> Self {
        let len = schedule.len();
        let positions = Positions::of(schedule);
        let blocks = Blocks::of(own, &positions);
        let count = blocks.tree.len();
        let block_at = |pos: usize| blocks.of_region[own.region_of(pos).0];

        let mut members: Vec<Vec<usize>> = alloc::vec![Vec::new(); count];
        for pos in 0..len {
            members[block_at(pos)].push(pos);
        }

        // Where each block starts and how many values it spans, nested blocks
        // included. A child is numbered after its parent, so one descending
        // sweep folds each into the one around it.
        let mut first: Vec<usize> = members
            .iter()
            .map(|own| own.first().copied().unwrap_or(usize::MAX))
            .collect();
        let mut span: Vec<usize> = members.iter().map(Vec::len).collect();
        for block in (1..count).rev() {
            let parent = blocks.tree.parent(block);
            first[parent] = first[parent].min(first[block]);
            span[parent] += span[block];
        }

        // What a block must follow in its parent's sequence: the values its
        // parent computes itself that anything inside it reads. A read of
        // something higher up needs nothing here — the whole parent block
        // already follows that — and a read of something deeper is an `If`
        // reading the root of its own arm, which precedes the `If` by
        // construction.
        let mut after = alloc::vec![0usize; count];
        for (pos, def) in schedule.iter().enumerate() {
            let here = block_at(pos);
            for read in folds.reads(def.value, &def.op) {
                let Some(source) = positions.get(read) else {
                    continue;
                };
                let there = block_at(source);
                if blocks.tree.depth(there) >= blocks.tree.depth(here) {
                    continue;
                }
                let child = blocks.tree.ancestor_at(here, blocks.tree.depth(there) + 1);
                debug_assert_eq!(blocks.tree.parent(child), there);
                after[child] = after[child].max(source + 1);
            }
        }

        let anchor: Vec<usize> = (0..count)
            .map(|block| {
                let Some(arm) = blocks.arm[block] else {
                    return 0;
                };
                let mask = positions
                    .get(own.arms()[arm].mask)
                    .expect("a block's mask is in the scope");
                first[block].max(after[block]).max(mask + 1)
            })
            .collect();

        // Children in the order they are placed: by anchor, then by `If`, the
        // true arm first. The arms are already in `If` order, so a stable
        // counting sort by anchor is the whole sort.
        let in_arm_order: Vec<usize> = {
            let mut by_arm = alloc::vec![None; own.arms().len()];
            for block in 1..count {
                by_arm[blocks.arm[block].expect("a block but the scope is an arm")] = Some(block);
            }
            by_arm.into_iter().flatten().collect()
        };
        let mut next_slot = alloc::vec![0usize; len + 2];
        for &block in &in_arm_order {
            next_slot[anchor[block] + 1] += 1;
        }
        for at in 1..next_slot.len() {
            next_slot[at] += next_slot[at - 1];
        }
        let mut placed = alloc::vec![0usize; in_arm_order.len()];
        for &block in &in_arm_order {
            placed[next_slot[anchor[block]]] = block;
            next_slot[anchor[block]] += 1;
        }
        let mut children: Vec<Vec<usize>> = alloc::vec![Vec::new(); count];
        for block in placed {
            children[blocks.tree.parent(block)].push(block);
        }

        // Emit: each block's own values in order, a child spliced ahead of the
        // first own value at or past its anchor. One explicit stack, because a
        // program may nest as deep as it likes and a call stack may not.
        struct Frame {
            block: usize,
            member: usize,
            child: usize,
        }
        let mut order = Vec::with_capacity(len);
        let mut start = alloc::vec![0usize; count];
        let mut stack = alloc::vec![Frame {
            block: 0,
            member: 0,
            child: 0
        }];
        while let Some(frame) = stack.last_mut() {
            let block = frame.block;
            let next_member = members[block].get(frame.member).copied();
            if let Some(&child) = children[block].get(frame.child)
                && next_member.is_none_or(|member| anchor[child] <= member)
            {
                frame.child += 1;
                start[child] = order.len();
                stack.push(Frame {
                    block: child,
                    member: 0,
                    child: 0,
                });
                continue;
            }
            match next_member {
                Some(member) => {
                    order.push(member);
                    frame.member += 1;
                }
                None => {
                    debug_assert_eq!(
                        order.len(),
                        start[block] + span[block],
                        "a block is one run of exactly the values it owns"
                    );
                    stack.pop();
                }
            }
        }
        debug_assert_eq!(
            order.len(),
            len,
            "a layout moves values, never adds or drops"
        );

        let mut position = alloc::vec![0usize; len];
        for (new, &old) in order.iter().enumerate() {
            position[old] = new;
        }

        let mut run_of_arm = alloc::vec![None; own.arms().len()];
        for block in 1..count {
            let arm = blocks.arm[block].expect("a block but the scope is an arm");
            run_of_arm[arm] = Some((start[block], start[block] + span[block]));
        }
        let mut guards: Vec<IfGuard> = own
            .arms()
            .chunks(2)
            .zip(run_of_arm.chunks(2))
            .filter(|(_, runs)| runs.iter().any(Option::is_some))
            .map(|(pair, runs)| {
                let if_idx = position[pair[0].if_pos];
                let run = |arm: usize| runs[arm].unwrap_or((if_idx, if_idx));
                IfGuard {
                    if_idx,
                    mask_vid: pair[0].mask,
                    ranges: ArmPair::new(run(0), run(1)),
                }
            })
            .collect();
        guards.sort_by_key(|guard| guard.if_idx);

        Self {
            order,
            position,
            guards,
        }
    }

    /// `schedule` in the new order.
    pub(crate) fn apply(&self, schedule: &[Def]) -> Vec<Def> {
        self.order
            .iter()
            .map(|&old| schedule[old].clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::ScheduledOp;
    use crate::program::{IfArm, IfGuard};
    use pixelflow_ir::kind::OpKind;

    impl Layout {
        /// Whether the schedule is returned as it came.
        pub(crate) fn is_identity(&self) -> bool {
            self.order.iter().enumerate().all(|(new, &old)| new == old)
        }

        /// Whether this is a layout: every read still follows the value it reads,
        /// and laying the result out again moves nothing — the property that makes
        /// the tables derived from one pass the tables of the order it chose.
        ///
        /// A schedule that reads a value before it defines it (a hand-built
        /// fixture) has no order to keep, and passes.
        pub(crate) fn is_sound(
            &self,
            schedule: &[Def],
            roots: &[ValueId],
            folds: &FoldReads,
        ) -> bool {
            if !reads_follow(0..schedule.len(), schedule, folds) {
                return true;
            }
            reads_follow(self.order.iter().copied(), schedule, folds)
                && Self::of(&self.apply(schedule), roots, folds).is_identity()
        }
    }

    /// Whether, taking `schedule`'s defs in `order`, every read follows the value
    /// it reads. A value this schedule does not define (a live-in) is not read
    /// from here.
    pub(crate) fn reads_follow(
        order: impl Iterator<Item = usize>,
        schedule: &[Def],
        folds: &FoldReads,
    ) -> bool {
        let positions = Positions::of(schedule);
        let mut at = alloc::vec![0usize; schedule.len()];
        let order: Vec<usize> = order.collect();
        for (new, &old) in order.iter().enumerate() {
            at[old] = new;
        }
        order.iter().enumerate().all(|(new, &old)| {
            let def = &schedule[old];
            folds
                .reads(def.value, &def.op)
                .filter_map(|read| positions.get(read))
                .all(|source| at[source] < new)
        })
    }

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

    fn unary(value: u64, op: OpKind, of: u64) -> Def {
        def(value, ScheduledOp::Unary(op, ValueId(of)))
    }

    /// The layout of `schedule` and the values in its new order, after every
    /// property a layout owes has been checked: it is one (every read follows
    /// its value, and laying it out again moves nothing), and each arm it
    /// branches over is exactly the run of the values that arm owns — no
    /// stranger inside it, nothing the arm owns outside it.
    fn laid_out(schedule: &[Def]) -> (Layout, Vec<u64>) {
        laid_out_with(schedule, &[], &FoldReads::default())
    }

    fn laid_out_with(schedule: &[Def], roots: &[ValueId], folds: &FoldReads) -> (Layout, Vec<u64>) {
        let layout = Layout::of(schedule, roots, folds);
        assert!(layout.is_sound(schedule, roots, folds));
        let own = Ownership::of(schedule, roots, folds);
        for pair in own.arms().chunks(2) {
            let guard = layout
                .guards
                .iter()
                .find(|g| schedule[layout.order[g.if_idx]].value == schedule[pair[0].if_pos].value);
            for arm in pair {
                let owned: Vec<ValueId> = (0..schedule.len())
                    .filter(|&pos| own.is_within(own.region_of(pos), arm.region))
                    .map(|pos| schedule[pos].value)
                    .collect();
                let run: Vec<ValueId> = guard
                    .map(|g| g.range(arm.arm))
                    .filter(|&(start, end)| start != end)
                    .map(|(start, end)| {
                        (start..end)
                            .map(|new| schedule[layout.order[new]].value)
                            .collect()
                    })
                    .unwrap_or_default();
                if run.is_empty() {
                    continue;
                }
                let (mut run, mut owned) = (run, owned);
                run.sort_unstable();
                owned.sort_unstable();
                assert_eq!(
                    run, owned,
                    "the {:?} arm of the If at {} is not exactly the run of what it owns",
                    arm.arm, arm.if_pos
                );
                assert!(arm.cycles > MISPREDICT_PENALTY_CYCLES);
            }
        }
        let values = layout
            .order
            .iter()
            .map(|&old| schedule[old].value.0)
            .collect();
        (layout, values)
    }

    /// Arms already one run after their mask: nothing moves.
    #[test]
    fn a_scope_whose_arms_are_in_order_is_returned_as_it_came() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            unary(2, OpKind::Rsqrt, 1),
            if_of(3, 0, 2, 0),
        ];
        let (layout, values) = laid_out(&schedule);
        assert!(layout.is_identity());
        assert_eq!(values, [0, 1, 2, 3]);
        assert_eq!(layout.guards.len(), 1);
        assert_eq!(layout.guards[0].range(IfArm::True), (1, 3));
        assert_eq!(layout.guards[0].range(IfArm::False), (3, 3));
    }

    /// A stranger between the arm's values: the arm's two values become one
    /// run and the stranger stays where the `If` does not need it moved.
    #[test]
    fn a_stranger_between_an_arms_values_is_left_behind() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            unary(2, OpKind::Sqrt, 1),  // the arm's, first value
            unary(3, OpKind::Neg, 0),   // a stranger: nothing in the arm reads it
            unary(4, OpKind::Rsqrt, 2), // the arm's, second value
            if_of(5, 0, 4, 0),
            def(6, ScheduledOp::Binary(OpKind::Add, ValueId(5), ValueId(3))),
        ];
        let (layout, values) = laid_out(&schedule);
        assert!(!layout.is_identity());
        // The arm is values 2 and 4 and must be adjacent; the stranger is
        // read after the `If` and stays out of the arm's way.
        let at = |v: u64| values.iter().position(|&x| x == v).unwrap();
        assert_eq!(at(4), at(2) + 1, "the arm is one run: {values:?}");
        assert!(at(5) > at(4));
        assert_eq!(
            *values.last().unwrap(),
            6,
            "the scope's last value stays last"
        );
    }

    /// An arm that costs exactly the mispredict penalty earns nothing, and one
    /// a little over earns its block.
    #[test]
    fn the_bound_is_strict() {
        assert_eq!(
            pixelflow_search::egraph::CostModel::latency_prior().cost(OpKind::Recip),
            MISPREDICT_PENALTY_CYCLES,
            "fixture assumes Recip sits exactly on the bound",
        );
        let recip_only = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            unary(2, OpKind::Recip, 1),
            if_of(3, 0, 2, 0),
        ];
        assert!(
            Layout::of(&recip_only, &[], &FoldReads::default())
                .guards
                .is_empty()
        );
        let a_little_more = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            unary(2, OpKind::Recip, 1),
            unary(3, OpKind::Neg, 2),
            if_of(4, 0, 3, 0),
        ];
        let layout = Layout::of(&a_little_more, &[], &FoldReads::default());
        assert_eq!(layout.guards.len(), 1);
    }

    /// Both arms of one `If` each earn a run, and the `If` that is the scope's
    /// root stays last.
    #[test]
    fn both_arms_of_an_if_are_runs_and_the_if_stays_last() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            unary(2, OpKind::Rsqrt, 1),
            unary(3, OpKind::Rsqrt, 0),
            unary(4, OpKind::Sqrt, 2),
            unary(5, OpKind::Sqrt, 3),
            if_of(6, 0, 4, 5),
        ];
        let (layout, values) = laid_out(&schedule);
        assert_eq!(*values.last().unwrap(), 6);
        let guard = &layout.guards[0];
        let (t, f) = (guard.range(IfArm::True), guard.range(IfArm::False));
        assert!(t.0 != t.1 && f.0 != f.1, "both arms branch: {guard:?}");
        assert!(t.1 <= f.0 || f.1 <= t.0, "the two runs do not overlap");
    }

    /// An arm nested in an arm: both are runs, the inner inside the outer.
    #[test]
    fn nested_arms_are_nested_runs() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            unary(2, OpKind::Rsqrt, 1),
            unary(3, OpKind::Rsqrt, 0),
            unary(4, OpKind::Sqrt, 2),
            if_of(5, 0, 4, 3),
            unary(6, OpKind::Recip, 1),
            if_of(7, 0, 5, 6),
        ];
        let (layout, _) = laid_out(&schedule);
        let (inner, outer) = (&layout.guards[0], &layout.guards[1]);
        let inside = |a: (usize, usize), b: (usize, usize)| b.0 <= a.0 && a.1 <= b.1;
        assert!(
            inside(inner.range(IfArm::True), outer.range(IfArm::True)),
            "the inner arm's run is inside the outer's: {inner:?} in {outer:?}"
        );
    }

    /// A block whose input is computed inside what would be its span waits for
    /// it: the block lands after the input, not before.
    #[test]
    fn a_block_follows_an_input_that_sits_inside_its_span() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            unary(2, OpKind::Sqrt, 1), // the arm's
            unary(3, OpKind::Neg, 0),  // shared: the arm reads it below
            def(4, ScheduledOp::Binary(OpKind::Add, ValueId(2), ValueId(3))),
            unary(5, OpKind::Rsqrt, 4), // the arm's root
            def(6, ScheduledOp::Binary(OpKind::Add, ValueId(3), ValueId(0))),
            if_of(7, 0, 5, 6),
        ];
        let (_, values) = laid_out(&schedule);
        let at = |v: u64| values.iter().position(|&x| x == v).unwrap();
        assert!(at(3) < at(4), "the input precedes its reader: {values:?}");
    }

    /// A deterministic stream, so a failure names its seed.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// A random DAG with `Ifs` in it: every value reads earlier ones, an `If`
    /// reads three. Heavy ops (`Rsqrt`, 21 cycles) make arms worth a branch.
    fn random_schedule(seed: u64, len: usize) -> Vec<Def> {
        let mut rng = Lcg(seed);
        let mut schedule: Vec<Def> =
            alloc::vec![def(0, ScheduledOp::Var(0)), def(1, ScheduledOp::Var(1))];
        while schedule.len() < len {
            let value = schedule.len() as u64;
            let earlier = |rng: &mut Lcg| rng.below(value as usize) as u64;
            let op = match rng.below(10) {
                0..=2 => ScheduledOp::Unary(OpKind::Rsqrt, ValueId(earlier(&mut rng))),
                3..=4 => ScheduledOp::Unary(OpKind::Neg, ValueId(earlier(&mut rng))),
                5..=6 => ScheduledOp::Binary(
                    OpKind::Add,
                    ValueId(earlier(&mut rng)),
                    ValueId(earlier(&mut rng)),
                ),
                _ => ScheduledOp::Ternary(
                    OpKind::If,
                    ValueId(earlier(&mut rng)),
                    ValueId(earlier(&mut rng)),
                    ValueId(earlier(&mut rng)),
                ),
            };
            schedule.push(def(value, op));
        }
        schedule
    }

    /// Whatever the DAG, the layout is a layout, and every arm it branches
    /// over is exactly the run of the values that arm owns.
    #[test]
    fn a_random_dag_lays_out_validly() {
        for seed in 0..300 {
            let len = 20 + (seed as usize % 40);
            let schedule = random_schedule(seed, len);
            let roots: Vec<ValueId> = alloc::vec![ValueId((len - 1) as u64)];
            let folds = FoldReads::default();
            let layout = Layout::of(&schedule, &roots, &folds);
            assert!(
                layout.is_sound(&schedule, &roots, &folds),
                "seed {seed}: not a layout"
            );
            laid_out_with(&schedule, &roots, &folds);
        }
    }

    // -- What a loop reads and costs -------------------------------------

    /// A four-trip fold; the scope its body was carved into is not what these
    /// tests look at, so the body names a hole.
    fn reduce() -> ScheduledOp {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};
        let fold = Fold::new(
            Monoid::SUM,
            Binder::from_slot(0).expect("slot 0 exists"),
            0..4,
        );
        ScheduledOp::Reduce(fold, ValueId(99))
    }

    fn at(values: &[u64], v: u64) -> usize {
        values
            .iter()
            .position(|&x| x == v)
            .expect("a layout keeps every def")
    }

    /// `W` is read by the true arm and by a sibling fold's body, which runs
    /// whatever the mask: the arm may not own `W`. Its `Reduce` def names no
    /// operand, so without the sibling's reads the arm owned it, and a
    /// uniformly-false mask skipped the loop the sibling then read.
    #[test]
    fn an_arm_does_not_own_what_a_sibling_fold_reads() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, reduce()),
            unary(2, OpKind::Rsqrt, 1),
            def(3, ScheduledOp::Binary(OpKind::Mul, ValueId(1), ValueId(2))),
            if_of(4, 0, 3, 0),
            def(5, reduce()),
            def(6, ScheduledOp::Binary(OpKind::Add, ValueId(4), ValueId(5))),
        ];
        // The sibling's body holds `W`'s id, as a placeholder read from its
        // accumulator slot.
        let sibling = [def(1, reduce())];
        let folds = FoldReads::new(
            &schedule,
            [(ValueId(5), &sibling[..], &FoldReads::default())],
        );

        let (blind, _) = laid_out(&schedule);
        let (seen, values) = laid_out_with(&schedule, &[], &folds);
        let len = |g: &IfGuard| g.range(IfArm::True).1 - g.range(IfArm::True).0;
        assert_eq!(len(&blind.guards[0]), 3, "the arm owned `W` unseen");
        assert_eq!(len(&seen.guards[0]), 2, "`W` is not the arm's");
        assert!(
            at(&values, 1) < at(&values, 2),
            "and it stays ahead of the arm"
        );
    }

    /// `s` is read only by `W`'s body, and `W` is in the true arm: `s` is the
    /// arm's, and goes with it, ahead of the loop that reads it.
    #[test]
    fn what_a_fold_reads_stays_ahead_of_the_fold() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            unary(2, OpKind::Rsqrt, 1),
            def(3, ScheduledOp::Binary(OpKind::Add, ValueId(1), ValueId(1))),
            def(4, reduce()),
            def(5, ScheduledOp::Binary(OpKind::Mul, ValueId(2), ValueId(4))),
            if_of(6, 0, 5, 0),
            def(7, ScheduledOp::Binary(OpKind::Add, ValueId(6), ValueId(1))),
        ];
        let body = [def(3, ScheduledOp::Const(0.0))];
        let folds = FoldReads::new(&schedule, [(ValueId(4), &body[..], &FoldReads::default())]);
        let (_, values) = laid_out_with(&schedule, &[], &folds);
        assert!(at(&values, 3) < at(&values, 4), "{values:?}");
    }

    /// A gather's base is a pointer operand, not a vector one, and it is a
    /// read all the same: the `Context` the arm's `Broadcast` addresses
    /// through goes ahead of its reader, not behind it.
    #[test]
    fn a_pointer_stays_ahead_of_its_reader() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            unary(2, OpKind::Rsqrt, 1),
            def(3, ScheduledOp::Binary(OpKind::Add, ValueId(1), ValueId(1))),
            def(4, ScheduledOp::Context(0)),
            def(5, ScheduledOp::Broadcast(ValueId(2), ValueId(4))),
            if_of(6, 0, 5, 0),
            def(7, ScheduledOp::Binary(OpKind::Add, ValueId(6), ValueId(3))),
        ];
        let (_, values) = laid_out(&schedule);
        assert!(at(&values, 4) < at(&values, 5), "{values:?}");
    }

    /// A value the schedule never defines is not a position: it must not be
    /// mistaken for one, which would corrupt a run or overflow computing its
    /// end.
    #[test]
    fn a_read_of_a_value_the_schedule_never_defines_is_ignored() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            unary(1, OpKind::Rsqrt, 3), // ValueId(3) has no Def
            if_of(4, 0, 0, 1),
        ];
        let (layout, _) = laid_out(&schedule);
        assert_eq!(layout.guards.len(), 1);
        assert_eq!(layout.guards[0].range(IfArm::False), (1, 2));
    }

    /// The arm fold's trip count below: long enough that pricing the loop as
    /// one instruction, or as nothing, is off by a factor this large.
    const ARM_TRIPS: u32 = 64;

    /// A sum of `trips` terms binding `slot`.
    fn sum_over(slot: u8, trips: u32) -> pixelflow_ir::fold::Fold {
        use pixelflow_ir::fold::{Binder, Fold, Monoid};
        let binder = Binder::from_slot(slot).expect("a live binder slot");
        Fold::new(Monoid::SUM, binder, 0..trips)
    }

    /// `select(X < 20, F, 0) + Y`, with `F` the `Reduce` def `fold` opening at
    /// position 3 — the true arm's only entry of its own.
    fn if_over_a_fold(fold: pixelflow_ir::fold::Fold, body_root: u64) -> Vec<Def> {
        alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Const(20.0)),
            def(2, ScheduledOp::Binary(OpKind::Lt, ValueId(0), ValueId(1))),
            def(3, ScheduledOp::Reduce(fold, ValueId(body_root))),
            def(4, ScheduledOp::Const(0.0)),
            if_of(5, 2, 3, 4),
            def(6, ScheduledOp::Var(1)),
            def(7, ScheduledOp::Binary(OpKind::Add, ValueId(5), ValueId(6))),
        ]
    }

    /// `|x − j|` per trip, `j` the binder of `fold` and `x` read from the
    /// enclosing scope as `ValueId(0)`; the ids from `first` up are the body's
    /// own.
    fn distance_body(fold: pixelflow_ir::fold::Fold, first: u64) -> Vec<Def> {
        let (j, diff, abs) = (first, first + 1, first + 2);
        alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(j, ScheduledOp::Var(fold.binder().var())),
            def(
                diff,
                ScheduledOp::Binary(OpKind::Sub, ValueId(0), ValueId(j))
            ),
            def(abs, ScheduledOp::Unary(OpKind::Abs, ValueId(diff))),
        ]
    }

    /// The price of the one arm of `schedule`'s one `If` that has work.
    fn true_arm_price(schedule: &[Def], roots: &[ValueId], folds: &FoldReads) -> usize {
        let own = Ownership::of(schedule, roots, folds);
        assert_eq!(own.arms().len(), 2);
        own.arms()[0].cycles
    }

    /// An arm that owns a fold is priced by the loop: `n` trips of a `k`-cycle
    /// body are `n·k`, plus the `n − 1` combines — not the `Reduce` def's
    /// table price of 0, which refused the arm a branch however long the
    /// loop. With the price, the arm clears the mispredict bound and is
    /// guarded.
    #[test]
    fn an_arm_that_owns_a_fold_is_priced_by_its_trips() {
        let cycles = pixelflow_search::egraph::CostModel::latency_prior();
        let fold = sum_over(0, ARM_TRIPS);
        let body = distance_body(fold, 10);
        let schedule = if_over_a_fold(fold, 12);
        let folds = FoldReads::new(&schedule, [(ValueId(3), &body[..], &FoldReads::default())]);

        let n = ARM_TRIPS as usize;
        let k = cycles.cost(OpKind::Sub) + cycles.cost(OpKind::Abs);
        let combine = cycles.cost(OpKind::Add);
        assert_eq!(
            true_arm_price(&schedule, &[], &folds),
            n * k + (n - 1) * combine,
            "the arm is its loop: {n} trips of a {k}-cycle body, and a combine between each"
        );
        assert_eq!(
            true_arm_price(&schedule, &[], &folds),
            cycles.fold_cost(fold, k)
        );
        assert_eq!(
            true_arm_price(&schedule, &[], &FoldReads::default()),
            0,
            "a Reduce def that opens no loop here is a slot read, and the table prices it 0"
        );

        let (layout, _) = laid_out_with(&schedule, &[], &folds);
        assert_eq!(
            layout.guards[0].range(IfArm::True),
            (3, 4),
            "the loop is skipped whole"
        );
        assert!(
            Layout::of(&schedule, &[], &FoldReads::default())
                .guards
                .is_empty()
        );
    }

    /// A table read whose address the lane binder does not reach is a
    /// `Broadcast`, and in a fold's body it runs every trip: `Σ_j |x − t[j]|`
    /// is priced `n` reads, as the extractor priced the arena's `RawGather`,
    /// not `n` of a uniform's prologue leaf (0). Its base pointer is the
    /// scope's root and read by the loop, so the arm is the loop alone.
    #[test]
    fn a_table_read_per_trip_is_priced_as_the_read_it_is() {
        let cycles = pixelflow_search::egraph::CostModel::latency_prior();
        let fold = sum_over(0, ARM_TRIPS);
        let (x, ctx, j, t, diff, abs) = (0, 8, 10, 11, 12, 13);
        let schedule = alloc::vec![
            def(x, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Const(20.0)),
            def(2, ScheduledOp::Binary(OpKind::Lt, ValueId(x), ValueId(1))),
            def(ctx, ScheduledOp::Context(0)),
            def(3, ScheduledOp::Reduce(fold, ValueId(abs))),
            def(4, ScheduledOp::Const(0.0)),
            if_of(5, 2, 3, 4),
            def(6, ScheduledOp::Var(1)),
            def(7, ScheduledOp::Binary(OpKind::Add, ValueId(5), ValueId(6))),
        ];
        let body = [
            def(x, ScheduledOp::Var(0)),
            def(ctx, ScheduledOp::Context(0)),
            def(j, ScheduledOp::Var(fold.binder().var())),
            def(t, ScheduledOp::Broadcast(ValueId(j), ValueId(ctx))),
            def(
                diff,
                ScheduledOp::Binary(OpKind::Sub, ValueId(x), ValueId(t)),
            ),
            def(abs, ScheduledOp::Unary(OpKind::Abs, ValueId(diff))),
        ];
        let folds = FoldReads::new(&schedule, [(ValueId(3), &body[..], &FoldReads::default())]);
        let roots = [ValueId(ctx)];

        let k =
            cycles.cost(OpKind::RawGather) + cycles.cost(OpKind::Sub) + cycles.cost(OpKind::Abs);
        assert_eq!(
            true_arm_price(&schedule, &roots, &folds),
            cycles.fold_cost(fold, k)
        );
        let (layout, _) = laid_out_with(&schedule, &roots, &folds);
        assert_eq!(
            layout.guards[0].range(IfArm::True),
            (4, 5),
            "the loop, not its pointer"
        );
    }

    /// A fold nested in the arm's fold is priced by its own trips inside every
    /// trip of the outer one: `m · (n·k + …)`, recursively — the inner loop's
    /// price comes from the outer body's own `FoldReads`.
    #[test]
    fn a_nested_fold_is_priced_by_the_product_of_its_trips() {
        const OUTER_TRIPS: u32 = 4;
        let cycles = pixelflow_search::egraph::CostModel::latency_prior();
        let (outer, inner) = (sum_over(0, OUTER_TRIPS), sum_over(1, ARM_TRIPS));
        // Outer body: `|I − i|`, `I` the inner fold, `i` the outer binder.
        let outer_body = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(20, ScheduledOp::Var(outer.binder().var())),
            def(21, ScheduledOp::Reduce(inner, ValueId(32))),
            def(
                22,
                ScheduledOp::Binary(OpKind::Sub, ValueId(21), ValueId(20))
            ),
            def(23, ScheduledOp::Unary(OpKind::Abs, ValueId(22))),
        ];
        let inner_body = distance_body(inner, 30);
        let inside = FoldReads::new(
            &outer_body,
            [(ValueId(21), &inner_body[..], &FoldReads::default())],
        );
        let schedule = if_over_a_fold(outer, 23);
        let folds = FoldReads::new(&schedule, [(ValueId(3), &outer_body[..], &inside)]);

        let k = cycles.cost(OpKind::Sub) + cycles.cost(OpKind::Abs);
        let inner_loop = cycles.fold_cost(inner, k);
        let outer_loop = cycles.fold_cost(outer, inner_loop + k);
        let price = true_arm_price(&schedule, &[], &folds);
        assert_eq!(price, outer_loop);
        assert!(
            outer_loop >= (OUTER_TRIPS * ARM_TRIPS) as usize * k,
            "every trip of the outer loop runs the whole inner one"
        );
    }
}
