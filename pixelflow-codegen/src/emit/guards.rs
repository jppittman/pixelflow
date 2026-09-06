//! Which schedule entries a `Select`'s short-circuit branch may skip.
//!
//! A property of the **schedule**, not of allocation and not of emission: it
//! asks only which values each arm of a `Select` computes for itself, and the
//! answer is the same whatever registers those values end up in. Both sides
//! read it — the emitter to place the branches, the allocator to keep a split
//! live range from naming a register a skipped arm never loaded — so it lives
//! beside them rather than inside either.

use alloc::vec::Vec;

use pixelflow_ir::kind::OpKind;

use super::ScheduledOp;
use super::regalloc::{Def, ValueId};

/// Describes a Select node's short-circuit structure in the schedule.
///
/// For `Select(mask, if_true, if_false)`, identifies contiguous ranges of
/// schedule entries that are exclusive to each arm (not shared with mask
/// or the other arm). These ranges can be guarded by conditional branches.
#[derive(Debug, Clone)]
pub(crate) struct SelectGuard {
    /// Schedule index of the Select node itself.
    pub(crate) select_idx: usize,
    /// ValueId of the mask operand (already computed before arms).
    pub(crate) mask_vid: ValueId,
    /// Range of schedule indices exclusive to the true arm: [true_start, true_end).
    /// Empty if true_start == true_end.
    pub(crate) true_range: (usize, usize),
    /// Range of schedule indices exclusive to the false arm: [false_start, false_end).
    pub(crate) false_range: (usize, usize),
}

/// Compute the transitive dependencies of a ValueId in the schedule.
///
/// `schedule_ops` is a dense Vec indexed by `ValueId.0`, pre-built by the
/// caller so each lookup is O(1) instead of O(n).
fn transitive_deps(
    vid: ValueId,
    schedule_ops: &[Option<ScheduledOp>],
) -> alloc::collections::BTreeSet<ValueId> {
    use alloc::collections::BTreeSet;

    let mut deps = BTreeSet::new();
    let mut worklist = alloc::vec![vid];
    while let Some(v) = worklist.pop() {
        if !deps.insert(v) {
            continue;
        }
        // O(1) lookup via dense Vec indexed by ValueId.0
        if let Some(Some(sop)) = schedule_ops.get(v.0 as usize) {
            match sop {
                ScheduledOp::Var(_) | ScheduledOp::Const(_) => {}
                ScheduledOp::Unary(_, c)
                | ScheduledOp::ShiftImm(_, c, _)
                | ScheduledOp::Gather(c, _) => {
                    worklist.push(*c);
                }
                ScheduledOp::Binary(_, l, r) => {
                    worklist.push(*l);
                    worklist.push(*r);
                }
                ScheduledOp::Ternary(_, a, b, c) => {
                    worklist.push(*a);
                    worklist.push(*b);
                    worklist.push(*c);
                }
            }
        }
    }
    deps
}
/// Analyze the schedule for Select nodes and compute short-circuit guard ranges.
///
/// For each Select, partitions schedule entries into:
/// - Shared: needed by mask, or by both arms (must always execute)
/// - True-exclusive: only needed by the true arm (skip if mask all-false)
/// - False-exclusive: only needed by the false arm (skip if mask all-true)
///
/// Returns guards sorted by select_idx (ascending).
pub(crate) fn analyze_select_guards(schedule: &[Def]) -> Vec<SelectGuard> {
    use alloc::collections::BTreeSet;

    let mut guards = Vec::new();

    if schedule.is_empty() {
        return guards;
    }

    // Build dense lookup: schedule_ops[vid.0] = Some(&ScheduledOp) for O(1) child traversal.
    // ValueIds are sequential starting from 0 (guaranteed by arena_to_schedule).
    let max_vid = schedule.iter().map(|def| def.value.0).max().unwrap_or(0) as usize;
    let mut schedule_ops: alloc::vec::Vec<Option<ScheduledOp>> = alloc::vec![None; max_vid + 1];
    for def in schedule {
        schedule_ops[def.value.0 as usize] = Some(def.op.clone());
    }

    // Build dense lookup: vid_to_sched_idx[vid.0] = schedule position (u32::MAX = absent).
    let mut vid_to_sched_idx: alloc::vec::Vec<usize> = alloc::vec![usize::MAX; max_vid + 1];
    for (i, def) in schedule.iter().enumerate() {
        vid_to_sched_idx[def.value.0 as usize] = i;
    }

    // Global consumer map: consumers[v.0] = every value that reads v as an
    // operand. A node may only be guarded (skipped when its arm's mask is
    // uniform) if EVERY consumer is inside that arm's subtree (or the select
    // itself) — otherwise an outer/sibling expression reads a register the
    // branch never computed. Subtree-local exclusivity (below) is necessary but
    // NOT sufficient; this is the global check that was missing.
    let mut consumers: alloc::vec::Vec<alloc::vec::Vec<ValueId>> =
        alloc::vec![alloc::vec::Vec::new(); max_vid + 1];
    for def in schedule {
        let vid = def.value;
        let mut add = |child: ValueId| {
            if (child.0 as usize) <= max_vid {
                consumers[child.0 as usize].push(vid);
            }
        };
        match &def.op {
            ScheduledOp::Var(_) | ScheduledOp::Const(_) => {}
            ScheduledOp::Unary(_, c)
            | ScheduledOp::ShiftImm(_, c, _)
            | ScheduledOp::Gather(c, _) => add(*c),
            ScheduledOp::Binary(_, a, b) => {
                add(*a);
                add(*b);
            }
            ScheduledOp::Ternary(_, a, b, c) => {
                add(*a);
                add(*b);
                add(*c);
            }
        }
    }

    for (i, def) in schedule.iter().enumerate() {
        let (sel_vid, sop) = (&def.value, &def.op);
        if let ScheduledOp::Ternary(OpKind::Select, mask_vid, true_vid, false_vid) = sop {
            // Compute transitive deps for each subtree using the dense O(1) lookup
            let mask_deps = transitive_deps(*mask_vid, &schedule_ops);
            let true_deps = transitive_deps(*true_vid, &schedule_ops);
            let false_deps = transitive_deps(*false_vid, &schedule_ops);

            // A node is safe to skip under this arm only if every one of its
            // consumers lies within the arm's subtree or is the select node
            // itself. Otherwise skipping it (uniform-mask short-circuit) leaves a
            // value some other expression still reads uninitialized.
            let only_used_within = |v: ValueId, arm: &BTreeSet<ValueId>| {
                consumers[v.0 as usize]
                    .iter()
                    .all(|c| *c == *sel_vid || arm.contains(c))
            };

            // True-exclusive: in true_deps but NOT in mask_deps and NOT in
            // false_deps, AND used only within the true arm.
            let true_exclusive: BTreeSet<ValueId> = true_deps
                .difference(&mask_deps)
                .copied()
                .collect::<BTreeSet<_>>()
                .difference(&false_deps)
                .copied()
                .filter(|v| only_used_within(*v, &true_deps))
                .collect();

            // False-exclusive: symmetric.
            let false_exclusive: BTreeSet<ValueId> = false_deps
                .difference(&mask_deps)
                .copied()
                .collect::<BTreeSet<_>>()
                .difference(&true_deps)
                .copied()
                .filter(|v| only_used_within(*v, &false_deps))
                .collect();

            // Map to schedule indices using dense O(1) lookup
            let true_indices: BTreeSet<usize> = true_exclusive
                .iter()
                .filter_map(|v| {
                    let idx = *vid_to_sched_idx.get(v.0 as usize)?;
                    if idx == usize::MAX { None } else { Some(idx) }
                })
                .collect();
            let false_indices: BTreeSet<usize> = false_exclusive
                .iter()
                .filter_map(|v| {
                    let idx = *vid_to_sched_idx.get(v.0 as usize)?;
                    if idx == usize::MAX { None } else { Some(idx) }
                })
                .collect();

            // The guard's uniformity test is emitted at the range START and
            // reads the mask's register there, so the mask must already be
            // computed by then. Schedules from the macro pipeline emit the mask
            // before both arms, but arena-composed kernels (`Kernel::select`
            // splicing arbitrary fragments) may schedule an arm BEFORE the
            // mask — guarding such an arm would branch on an uninitialized
            // register. No guard in that case; the select still evaluates
            // correctly through the unconditional BSL/blend path.
            let mask_idx = vid_to_sched_idx
                .get(mask_vid.0 as usize)
                .copied()
                .unwrap_or(usize::MAX);

            // Get contiguous ranges (min..max+1)
            let true_range = if true_indices.is_empty() {
                (i, i) // empty range
            } else {
                let start = *true_indices
                    .iter()
                    .next()
                    .expect("non-empty set has first element");
                let end = *true_indices
                    .iter()
                    .next_back()
                    .expect("non-empty set has last element")
                    + 1;
                // The branch skips the WHOLE range [start, end) when the mask is
                // uniform, so EVERY index in it must be a true-exclusive node.
                // If any in-range index is a shared node (used outside this arm)
                // or a false-exclusive node, skipping it would leave a value some
                // other expression reads uninitialized — fall back to BSL.
                let all_exclusive = (start..end).all(|idx| true_indices.contains(&idx));
                if all_exclusive && mask_idx < start {
                    (start, end)
                } else {
                    (i, i)
                }
            };

            let false_range = if false_indices.is_empty() {
                (i, i)
            } else {
                let start = *false_indices
                    .iter()
                    .next()
                    .expect("non-empty set has first element");
                let end = *false_indices
                    .iter()
                    .next_back()
                    .expect("non-empty set has last element")
                    + 1;
                let all_exclusive = (start..end).all(|idx| false_indices.contains(&idx));
                if all_exclusive && mask_idx < start {
                    (start, end)
                } else {
                    (i, i)
                }
            };

            // Only create a guard if at least one arm has exclusive nodes
            if true_range.0 != true_range.1 || false_range.0 != false_range.1 {
                guards.push(SelectGuard {
                    select_idx: i,
                    mask_vid: *mask_vid,
                    true_range,
                    false_range,
                });
            }
        }
    }

    guards
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::kind::OpKind;

    fn def(value: u32, op: ScheduledOp) -> Def {
        Def {
            value: ValueId(value),
            op,
        }
    }

    /// A `Select` whose true arm alone does work exclusive to it — the false
    /// arm is just the mask again, so it contributes nothing beyond
    /// `mask_deps`. Pins the exact range rather than only "a guard formed
    /// somewhere," which the whole-kernel `assert_guard_forms`-style tests
    /// in `emit/mod.rs` already cover.
    #[test]
    fn range_the_true_arm_when_only_it_is_exclusive() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Unary(OpKind::Neg, ValueId(1))),
            def(
                3,
                ScheduledOp::Ternary(OpKind::Select, ValueId(0), ValueId(2), ValueId(0)),
            ),
        ];

        let guards = analyze_select_guards(&schedule);

        assert_eq!(guards.len(), 1);
        assert_eq!(guards[0].select_idx, 3);
        assert_eq!(guards[0].mask_vid, ValueId(0));
        assert_eq!(guards[0].true_range, (1, 3));
        assert_eq!(guards[0].false_range, (3, 3));
    }

    /// Symmetric to the above: the false arm alone is exclusive.
    #[test]
    fn range_the_false_arm_when_only_it_is_exclusive() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Unary(OpKind::Neg, ValueId(1))),
            def(
                3,
                ScheduledOp::Ternary(OpKind::Select, ValueId(0), ValueId(0), ValueId(2)),
            ),
        ];

        let guards = analyze_select_guards(&schedule);

        assert_eq!(guards.len(), 1);
        assert_eq!(guards[0].select_idx, 3);
        assert_eq!(guards[0].true_range, (3, 3));
        assert_eq!(guards[0].false_range, (1, 3));
    }

    /// An operand reachable only through a value the schedule never defines
    /// (a "hole" — legitimate for a schedule spliced from arbitrary
    /// fragments, per this module's doc comment) must not be mistaken for a
    /// real schedule position. Regression test for treating the sentinel
    /// `usize::MAX` (marking "not in this schedule") as a valid index, which
    /// would corrupt the range or overflow computing its end.
    #[test]
    fn ignore_a_false_operand_missing_from_the_schedule() {
        let schedule = alloc::vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Unary(OpKind::Neg, ValueId(3))), // ValueId(3) has no Def
            def(
                4,
                ScheduledOp::Ternary(OpKind::Select, ValueId(0), ValueId(0), ValueId(1)),
            ),
        ];

        let guards = analyze_select_guards(&schedule);

        assert_eq!(guards.len(), 1);
        assert_eq!(guards[0].select_idx, 2);
        assert_eq!(guards[0].false_range, (1, 2));
    }
}
