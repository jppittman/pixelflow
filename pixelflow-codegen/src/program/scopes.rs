//! Scoping: a flat schedule to the nest the allocator and the emitter walk.
//!
//! Lowering ([`super::lower`]) produces one flat schedule. Scoping carves each
//! surviving `Reduce` out into a scope of its own and moves every value to the
//! outermost scope that binds what it depends on. [`Nest`] is the one way in;
//! nothing here names a register or a byte.

use alloc::vec::Vec;

use pixelflow_ir::fold::{Binder, Bound};
use pixelflow_ir::variance::Variance;

use super::lattice::Traversal;
use super::ownership::Positions;
use super::{
    Def, Looping, Lowered, ScheduledOp, Scope, ScopeFold, ScopeRegion, ScopedSchedule, ValueId,
    all_operands, structural_children,
};
use crate::error::CompileError;
use crate::pass::Pass;

/// Split a flat schedule into its nest, placing every value in the outermost
/// scope that binds all the binders it depends on.
///
/// [`extract_folds`] carves each surviving `Reduce`'s closure out into a scope
/// of its own, one level per fold, and in the same walk sorts the closure in
/// two: a value whose variance names no binder deeper than the fold's parent
/// is invariant in the fold, so the parent computes it once and the fold
/// *reads* it — the fold's `outer`, its free variables — while the fold's
/// schedule holds only what the fold defines. That is loop-invariant code
/// motion out of every fold at once — the lattice's rows and columns
/// included, so a per-call value is computed per call and a per-row one per
/// row (docs/archive/plans/2026-09-16-collapse-is-a-fold.md §2.2) — asked as one
/// question of the variance rather than as a tier per loop.
///
/// The factorization is the first thing `Nest` does: the lattice's row, column
/// and lane loops are written around the kernel's value
/// ([`write_traversal`]), so the carving below sees them as it sees any fold.
pub(crate) struct Nest;

impl Pass for Nest {
    type Input = Lowered;
    type Output = ScopedSchedule<()>;

    fn run(self, lowered: Lowered) -> Result<ScopedSchedule<()>, CompileError> {
        let Lowered {
            mut schedule,
            value,
            traversal,
        } = lowered;
        write_traversal(&mut schedule, &traversal, value);
        // The carving asks for an operand's variance by `ValueId` at every
        // level of the nest, so one table, from the full schedule before it
        // is split. Lowering numbers its defs `0..len`, so every slot is
        // written; an id past `len` panics here rather than read another's.
        let mut variance = alloc::vec![Variance::ALL; schedule.len()];
        for def in &schedule {
            variance[def.value.0 as usize] = def.variance;
        }
        let (body, pending) = extract_folds(schedule, &variance);
        let mut scoped = ScopedSchedule {
            body: ScopeRegion {
                schedule: body,
                analysis: (),
            },
            folds: Vec::new(),
        };
        attach_folds(&mut scoped, pending);
        Ok(scoped)
    }
}

impl<A> ScopedSchedule<A> {
    /// The closure of one scope, which [`Own`](super::Own) asserts of each
    /// scope before it reads off the scope's ownership: every value a def of
    /// the scope reads is a def of it or in its `outer`. Ownership sees a
    /// loop's reads at its `Reduce`, one level down
    /// ([`super::ownership::FoldReads`]), so a read it did not see would let an
    /// arm own a value a loop outside the arm reads.
    ///
    /// What carving makes true by construction (a fold's `outer` is the
    /// hoisted boundary of its closure; its `Reduce` is a def of its parent) is
    /// not asserted again. This is the part that rests on the variance
    /// contract, a def's variance including its operands', which the carving
    /// trusts to decide where an operand goes.
    ///
    /// Linear: this scope's defs and its `outer` are indexed densely by
    /// `ValueId` once.
    pub(super) fn assert_closed(&self, scope: Scope) {
        let (schedule, outer, _) = self.at(scope);
        let defined = Positions::of(schedule);
        // Ids are dense (handed out sequentially), so membership is one index.
        let capacity = outer.iter().map(|id| id.0 as usize + 1).max().unwrap_or(0);
        let mut from_outside = alloc::vec![false; capacity];
        for id in outer {
            from_outside[id.0 as usize] = true;
        }
        let read_from_outside =
            |id: ValueId| from_outside.get(id.0 as usize).copied().unwrap_or(false);

        for def in schedule {
            for read in all_operands(&def.op) {
                assert!(
                    defined.get(read).is_some() || read_from_outside(read),
                    "{:?} reads {read:?}, which {scope:?} neither defines nor reads from outside",
                    def.value
                );
            }
        }
    }
}

/// The traversal's loops, after the kernel's defs: for each of the row's
/// pieces ([`Traversal::pieces`]) a `Write` of `value` under its column loop,
/// the pieces sequenced under the row loop. The lane fold is a store: it is
/// executed by lanes and has no loop, so it is a `Write` of the kernel's value
/// with the piece's trip count as the store width, and a row's main batches and
/// remainder are two `Write`s of different widths reading one value. Each
/// def's variance is the fold's rule (`transfer` in `pixelflow-ir`'s
/// `variance.rs`): what its body reads, less the binder it closes; the kernel's
/// own is the variance of `value`'s def.
fn write_traversal(schedule: &mut Vec<Def>, traversal: &Traversal, value: ValueId) {
    let kernel = schedule[value.0 as usize].variance;
    let bit = |binder: Binder| Variance::from_var(binder.var());
    let mut push = |op: ScheduledOp, variance: Variance| {
        let id = ValueId(schedule.len() as u64);
        schedule.push(Def {
            value: id,
            op,
            variance,
        });
        (id, variance)
    };
    // A store reads its three binders for the address, so it sits inside all
    // three loops; the lane fold it is closes the lane binder.
    let stored = kernel.union(traversal.variance());
    let mut pieces = Vec::new();
    for piece in traversal.pieces() {
        let (write, in_lane) = push(
            ScheduledOp::Write {
                row: traversal.row,
                col: traversal.col,
                lane: traversal.lane,
                lanes: piece.lanes,
                value,
            },
            stored.without(bit(traversal.lane)),
        );
        let [lo, hi] = piece.range;
        pieces.push(push(
            ScheduledOp::Reduce {
                looping: piece.col,
                lo: Bound::Lit(lo),
                hi: Bound::Lit(hi),
                body: write,
            },
            in_lane.without(bit(traversal.col)),
        ));
    }
    let (walk, variance) = pieces
        .into_iter()
        .reduce(|(main, main_variance), (tail, tail_variance)| {
            push(
                ScheduledOp::Seq(main, tail),
                main_variance.union(tail_variance),
            )
        })
        .expect("a traversal has a piece");
    let _ = push(
        ScheduledOp::Reduce {
            looping: Looping::Sweep {
                binder: traversal.row,
                stride: 1,
            },
            lo: Bound::Lit(0),
            hi: Bound::Lit(u64::from(traversal.extent[1])),
            body: walk,
        },
        variance.without(bit(traversal.row)),
    );
}

/// The lane binders a scope binds by holding a store: the lane fold is
/// executed by lanes, inlined into the scope its `Write` def sits in, so
/// that scope is where a lane-varying value lives — nothing deeper binds
/// the lane, and nothing shallower may compute a value that varies by it.
fn lane_binders(schedule: &[Def]) -> pixelflow_ir::variance::Variance {
    use pixelflow_ir::variance::Variance;
    schedule
        .iter()
        .fold(Variance::CONST, |acc, def| match def.op {
            ScheduledOp::Write { lane, .. } => acc.union(Variance::from_var(lane.var())),
            _ => acc,
        })
}

/// A fold `extract_folds` carved out of a flat schedule, still looking for
/// its position: everything the body's closure computed, in topological
/// order, ending at the value the `Reduce` combines each iteration.
struct PendingFold {
    /// The `Reduce` def's own `ValueId` — a schedule permutation (LICM) may
    /// move it, but never renames it, so this is what
    /// `attach_folds` searches the post-partition schedule for.
    reduce_vid: ValueId,
    /// The fold's own per-iteration computation, in topological order,
    /// ending at the body's root unless the root is read from `outer`.
    schedule: Vec<Def>,
    /// What the body reads that its parent computes: the closure's hoisted
    /// boundary, in schedule order.
    outer: Vec<ValueId>,
    /// The folds whose `Reduce` def sits in `schedule`: a fold inside this
    /// one's body. The nest is a tree, and this is the recursion.
    children: Vec<PendingFold>,
}

/// Carve every surviving `Reduce`'s body out of `schedule`.
///
/// A fold inside a fold's body is carved the same way, one level down: the
/// outer fold's closure is a schedule like any other, and the inner fold is
/// a `Reduce` def in it. `variance` is every def's [`Def::variance`] by
/// `ValueId`, built once by [`Nest`]; a fold's
/// schedule keeps its parent's ids, which is what lets one array answer an
/// operand's variance at every level.
fn extract_folds(
    schedule: Vec<Def>,
    variance: &[pixelflow_ir::variance::Variance],
) -> (Vec<Def>, Vec<PendingFold>) {
    let top = alloc::vec![false; variance.len()];
    extract_folds_bound_by(
        schedule,
        variance,
        pixelflow_ir::variance::Variance::CONST,
        &top,
    )
}

/// [`extract_folds`] for one scope, `bound` being the binders that scope is
/// *inside*: none at the top, a fold's own binder and its ancestors' for
/// that fold's body.
///
/// `bound` is what decides which values leave. Nothing outside a fold can
/// read its own binder — that is what makes it a binder — so a value whose
/// variance names a binder this scope is *not* inside can only belong to a
/// fold nested deeper, and dropping it here is safe by construction. A value
/// a fold's closure also reached but whose variance names no deeper binder
/// (a shared invariant leaf, or one that varies only with an enclosing
/// binder) is not dropped: it stays here too, and the fold records it as
/// `outer`, to be read from the enclosing scope. Getting this backwards —
/// removing the whole closure — would orphan exactly that shared leaf's
/// other consumer. The closure ends at such a value: what it is built from
/// is the computing scope's business, not the fold's, and is not carried in
/// behind it.
///
/// The one thing that is *not* recomputed inside a fold is another fold
/// that does not depend on its binder: a whole loop per iteration is the
/// glyph's winding sum run once per piece of its distance fold, and once
/// more at the top for the coverage that reads it. Such a `Reduce` stays
/// the enclosing scope's fold, and the fold that reads its result keeps its
/// def as a **placeholder** — in the schedule, so the closure walk
/// resolves, but opening no scope here; it is one of the fold's `outer`, the
/// result the enclosing scope's loop bound.
/// `placeholder` is that verdict from the level above, by `ValueId`, so a
/// level never mistakes one for a fold of its own.
fn extract_folds_bound_by(
    schedule: Vec<Def>,
    variance: &[pixelflow_ir::variance::Variance],
    bound: pixelflow_ir::variance::Variance,
    placeholder: &[bool],
) -> (Vec<Def>, Vec<PendingFold>) {
    use pixelflow_ir::variance::Variance;

    // `ValueId` space, not this schedule's length: a fold's schedule is a
    // subset of its parent's, keeping the parent's ids.
    let n = variance.len();
    let mut position: Vec<Option<usize>> = alloc::vec![None; n];
    for (i, def) in schedule.iter().enumerate() {
        position[def.value.0 as usize] = Some(i);
    }
    // The lane fold is executed by lanes, inlined into the scope holding its
    // store — so that scope binds its binder, and a lane-varying value is
    // computed here, not somewhere deeper.
    let bound = bound.union(lane_binders(&schedule));
    let deeper = Variance::BINDERS.bits() & !bound.bits();
    let hoisted =
        |v: ValueId| placeholder[v.0 as usize] || variance[v.0 as usize].bits() & deeper == 0;

    let mut pending: Vec<PendingFold> = Vec::new();
    // A `Reduce` inside another's closure is that one's child, not this
    // scope's own fold. The schedule is topological, so an outer fold's def
    // comes after everything its body reaches; walking it backwards meets
    // the outer fold first, and what its closure claims is skipped here and
    // carved out by the recursion instead.
    let mut claimed = alloc::vec![false; n];

    for def in schedule.iter().rev() {
        let ScheduledOp::Reduce {
            looping,
            body: body_vid,
            ..
        } = def.op
        else {
            continue;
        };
        if claimed[def.value.0 as usize] || placeholder[def.value.0 as usize] {
            continue;
        }
        // The fold's own closure: everything its body needs, reachable by
        // operand from its root. This *includes* any invariant leaf it
        // shares with code outside it (a `Uniform`, a shared sub-expression)
        // — reachability says nothing about whether such a value depends on
        // *this* binder, which is why it is not what decides removal below.
        //
        // It stops *at* such a leaf, though. A value the enclosing scope
        // computes is a placeholder in this fold and in every fold inside
        // it — read from its park, never emitted — so nothing behind it is
        // this fold's to read. Following through would carry its operands
        // in as placeholders no def here reads: roots the computing scope
        // must then park, for nobody. The `Uniform`s' base pointer was that
        // root, stored to the frame once per call for a fold that reads only
        // the uniforms themselves.
        //
        // A nested `Reduce` that depends on a binder bound here is followed
        // *into*: its body is not an operand (`operands` says so —
        // the def's own emission never reads it), but it is this closure's
        // to carry — `structural_children`, which the walk below
        // follows, yields it, and its value bounds, which a deeper fold may
        // compute — so the recursion below can carve it out again
        // one level down. One that does not is a placeholder like any other
        // hoisted value, remembered so the level below does not mistake it
        // for a fold of its own.
        let mut mark = alloc::vec![false; n];
        let mut placeholder_here = alloc::vec![false; n];
        let op_of = |v: ValueId| position[v.0 as usize].map(|p| &schedule[p].op);
        let is_fold = |v: ValueId| matches!(op_of(v), Some(ScheduledOp::Reduce { .. }));
        let mut stack = Vec::new();
        mark[body_vid.0 as usize] = true;
        // The root too: a body that *is* another fold's result reads that
        // result from the enclosing scope, and the whole schedule is the placeholder.
        if hoisted(body_vid) {
            placeholder_here[body_vid.0 as usize] = is_fold(body_vid);
        } else {
            stack.push(body_vid);
        }
        while let Some(v) = stack.pop() {
            let at = position[v.0 as usize].unwrap_or_else(|| {
                panic!(
                    "{v:?} is read by {:?}'s body but is not in its scope",
                    def.value
                )
            });
            let op = &schedule[at].op;
            if matches!(op, ScheduledOp::Reduce { .. }) {
                claimed[v.0 as usize] = true;
            }
            for operand in structural_children(op) {
                if mark[operand.0 as usize] {
                    continue;
                }
                mark[operand.0 as usize] = true;
                if hoisted(operand) {
                    placeholder_here[operand.0 as usize] = is_fold(operand);
                    continue;
                }
                stack.push(operand);
            }
        }
        let fold_schedule: Vec<Def> = schedule
            .iter()
            .filter(|d| mark[d.value.0 as usize])
            .cloned()
            .collect();
        let inside = bound.union(Variance::from_var(looping.binder().var()));
        let (fold_schedule, children) =
            extract_folds_bound_by(fold_schedule, variance, inside, &placeholder_here);
        // What the parent computes is read, not recomputed: the closure's
        // hoisted values leave the fold's schedule and become its `outer`.
        let (outer, fold_schedule): (Vec<Def>, Vec<Def>) =
            fold_schedule.into_iter().partition(|d| hoisted(d.value));
        pending.push(PendingFold {
            reduce_vid: def.value,
            schedule: fold_schedule,
            outer: outer.into_iter().map(|d| d.value).collect(),
            children,
        });
    }
    // Schedule order, so fold indices are stable whichever way this walked.
    pending.reverse();

    if pending.is_empty() {
        return (schedule, pending);
    }

    let remaining: Vec<Def> = schedule
        .into_iter()
        .filter(|def| def.variance.bits() & deeper == 0)
        .collect();
    (remaining, pending)
}

/// Record each [`PendingFold`] of the body as a [`ScopeFold`], and its
/// children inside it.
///
/// A fold names the `Reduce` def it opens at by value, not by position: a
/// schedule permutation moves a `Def` without renaming the `ValueId` it
/// defines.
fn attach_folds(scoped: &mut ScopedSchedule<()>, pending: Vec<PendingFold>) {
    for fold in pending {
        attach_fold(scoped, fold, Scope::Body);
    }
}

/// Record `fold` as a [`ScopeFold`] opening in `parent`, then each of its
/// children inside it — depth first, so a parent's index is always below its
/// children's, which is the order the `LinearScan` allocator (`regalloc::local`) and
/// the frame layout both walk the tree in.
fn attach_fold(scoped: &mut ScopedSchedule<()>, fold: PendingFold, parent: Scope) {
    let PendingFold {
        reduce_vid,
        schedule,
        outer,
        children,
    } = fold;
    let index = scoped.folds.len();
    scoped.folds.push(ScopeFold {
        parent,
        reduce: reduce_vid,
        outer,
        schedule,
        analysis: (),
    });
    for child in children {
        attach_fold(scoped, child, Scope::Fold(index));
    }
}
