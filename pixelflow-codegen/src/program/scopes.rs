//! Scoping: a flat schedule to the nest the allocator and the emitter walk.
//!
//! Lowering ([`super::lower`]) produces one flat schedule. Scoping carves each
//! surviving `Reduce` out into a scope of its own, moves every value to the
//! outermost scope that binds what it depends on, and lays every scope out.
//! [`ScopedSchedule::from_schedule`] is the one way in; nothing here names a
//! register or a byte.

use alloc::vec::Vec;

use super::{
    Class, Def, ScheduledOp, Scope, ScopeFold, ScopeRegion, ScopedSchedule, ValueId, guards,
    layout::Layout, structural_children,
};

/// Compute [`Variance`](pixelflow_ir::variance::Variance) for every schedule entry.
///
/// The schedule mirrors the arena's topological order, so one forward pass
/// suffices — the dense result is indexed by `ValueId.0`.
fn schedule_variance(schedule: &[Def]) -> Vec<pixelflow_ir::variance::Variance> {
    use pixelflow_ir::variance::Variance;
    let max_vid = schedule.iter().map(|def| def.value.0).max().unwrap_or(0) as usize;
    let mut v = alloc::vec![Variance::CONST; max_vid + 1];
    for def in schedule {
        let (vid, op) = (&def.value, &def.op);
        let i = vid.0 as usize;
        v[i] = match op {
            // Lowering emits a `Var` only for a binder it found, so an index
            // past the analysis's bits is a hand-built schedule, which `ALL`
            // would let through without a name.
            ScheduledOp::Var(idx) => {
                assert!(
                    *idx < Variance::VARIABLES,
                    "{vid:?} reads Var({idx}), past the {} variables the analysis names",
                    Variance::VARIABLES
                );
                Variance::from_var(*idx)
            }
            ScheduledOp::Lanes(lane) => Variance::from_var(lane.var()),
            // Invariant across the lattice; unknown until the call. The
            // `CONST` here is what carries it into the per-call scope — a
            // context pointer, and a uniform read through one.
            ScheduledOp::Const(_) | ScheduledOp::Context(_) | ScheduledOp::Uniform(..) => {
                Variance::CONST
            }
            ScheduledOp::Unary(_, a)
            | ScheduledOp::ShiftImm(_, a, _)
            // A gather reads from a bound buffer, whose contents are fixed for
            // the kernel's lifetime — its variance is its index's variance.
            // A broadcast is a gather whose index lacks the lane bit, so
            // the same rule places it in the scope its address varies in.
            | ScheduledOp::Gather(a, _)
            | ScheduledOp::Broadcast(a, _) => v[a.0 as usize],
            ScheduledOp::Binary(_, a, b) | ScheduledOp::Seq(a, b) => {
                v[a.0 as usize].union(v[b.0 as usize])
            }
            ScheduledOp::Ternary(_, a, b, c) => v[a.0 as usize]
                .union(v[b.0 as usize])
                .union(v[c.0 as usize]),
            // A completed reduction closes over its own binder: nothing
            // outside the fold can read it (that is what makes it a binder),
            // so the result's variance is the body's, minus that one bit —
            // `Variance::without` is exactly "what a binder does to its own
            // index" (see `variance.rs`'s doc).
            ScheduledOp::Reduce(fold, body) => {
                v[body.0 as usize].without(Variance::from_var(fold.binder().var()))
            }
            // A store reads its row and column for the address, so it sits
            // inside both their folds — which is the whole of why the
            // lattice's loops can be placed by the same rule as everything
            // else. And it is the lane fold, so like any `Reduce` it closes
            // over its own binder: the row fold's result must not read as
            // lane-varying, or the body would disown it.
            ScheduledOp::Write {
                row,
                col,
                lane,
                value,
                ..
            } => v[value.0 as usize]
                .without(Variance::from_var(lane.var()))
                .union(Variance::from_var(row.var()))
                .union(Variance::from_var(col.var())),
        };
    }
    v
}

impl ScopedSchedule {
    /// Split a flat schedule into its nest, placing every value in the outermost
    /// scope that binds all the binders it depends on.
    ///
    /// Two steps, and the second is the placement. [`extract_folds`] carves each
    /// surviving `Reduce`'s closure out into a scope of its own, one level per
    /// fold; what that leaves in a fold's schedule is everything its body
    /// reaches whose variance names no binder deeper than the fold's own — which
    /// still includes values invariant *in* the fold, computed in it once per
    /// trip. [`place_roots`] then moves each of those up to the outermost scope
    /// binding its binders, where it is computed once, and leaves a placeholder
    /// behind for the scope that read it. That is loop-invariant code motion
    /// out of every fold at once — the lattice's rows and columns included, so
    /// a per-call value is computed per call and a per-row one per row
    /// (docs/plans/2026-09-16-collapse-is-a-fold.md §2.2) — asked as one
    /// question of the variance rather than as a tier per loop.
    pub(crate) fn from_schedule(schedule: Vec<Def>) -> Self {
        // Variance first, over the arena's *full* schedule — a surviving
        // `Reduce`'s own result depends on its body's, and the body's def is
        // about to move (`extract_folds`, next) out of this array entirely.
        let variance = schedule_variance(&schedule);
        let (body, pending) = extract_folds(schedule, &variance);
        let mut scoped = ScopedSchedule {
            body: ScopeRegion {
                roots: Vec::new(),
                schedule: body,
                guards: Vec::new(),
            },
            folds: Vec::new(),
        };
        attach_folds(&mut scoped, pending);
        place_roots(&mut scoped, &variance);
        lay_out(&mut scoped);
        scoped
    }
}

/// Put every scope's schedule in its final order and tabulate the branches
/// over it — the last step of [`ScopedSchedule::from_schedule`], and the only place either is
/// decided for a body or a fold.
///
/// Last because both are questions about the scope's *final* contents: which
/// arms are worth a branch depends on what the loops the scope opens read from
/// it and on the roots it parks for them — a skipped arm would leave a park
/// unwritten for a loop that runs regardless — and none of that is settled
/// until [`attach_folds`] and [`place_roots`] have run. The order is *chosen*
/// here, from who owns what (`program::layout`), rather than repaired
/// beforehand and then hoped to survive the placement. Nothing after this
/// edits a schedule or a table: the allocator reads them as its input and the
/// emitter branches over the same ones.
///
/// What a loop costs, which decides whether an arm owning it pays for a
/// branch, is made of the loops inside it ([`guards::FoldReads`]), so the
/// scopes go innermost first: a fold's index is always above its parent's,
/// which makes the reverse of nest order a children-before-parents order. A
/// scope's layout permutes its schedule, and a fold's `at` is a position in
/// its parent's, so each scope's children are carried to the new order as the
/// scope is laid out.
pub(crate) fn lay_out(scoped: &mut ScopedSchedule) {
    let slot = |scope: Scope| match scope {
        Scope::Body => 0,
        Scope::Fold(j) => j + 1,
    };
    let mut reads: Vec<guards::FoldReads> = (0..=scoped.folds.len())
        .map(|_| guards::FoldReads::default())
        .collect();
    for scope in (0..scoped.folds.len())
        .rev()
        .map(Scope::Fold)
        .chain(core::iter::once(Scope::Body))
    {
        let nest = &*scoped;
        let (schedule, roots) = match scope {
            Scope::Body => (&nest.body.schedule, &nest.body.roots),
            Scope::Fold(j) => (&nest.folds[j].schedule, &nest.folds[j].roots),
        };
        let opened = guards::FoldReads::new(
            schedule,
            nest.folds
                .iter()
                .enumerate()
                .filter(|(_, fold)| fold.parent == scope)
                .map(|(k, fold)| {
                    let inner = &reads[slot(Scope::Fold(k))];
                    (schedule[fold.at].value, fold.schedule.as_slice(), inner)
                }),
        );
        let layout = Layout::of(schedule, roots, &opened);
        let (ordered, branches) = (layout.apply(schedule), layout.guards);
        let position = layout.position;
        match scope {
            Scope::Body => {
                scoped.body.schedule = ordered;
                scoped.body.guards = branches;
            }
            Scope::Fold(j) => {
                scoped.folds[j].schedule = ordered;
                scoped.folds[j].guards = branches;
            }
        }
        for fold in scoped.folds.iter_mut().filter(|fold| fold.parent == scope) {
            fold.at = position[fold.at];
        }
        reads[slot(scope)] = opened;
    }
}

/// Whether a def is a placeholder already, and so not the placement's to
/// park: a binder's `Var` (found where its fold keeps it), a `Reduce` that
/// is not this scope's own (read from its accumulator slot).
///
/// A `Const` used to be here too, as "cheaper rebuilt than reloaded". It is
/// not: rebuilding one is two instructions on x86, and a value parked for the
/// scopes inside is carried in a register when one is free, which is zero.
/// Whether a constant is worth a register is the allocator's question, priced
/// like every other root's, so nothing here answers it.
fn stays_put(op: &ScheduledOp) -> bool {
    matches!(op, ScheduledOp::Var(_) | ScheduledOp::Reduce(..))
}

/// The second half of [`ScopedSchedule::from_schedule`]: in every fold, each def whose
/// variance does not name the fold's binder is computed by an enclosing
/// scope — the outermost binding every binder it *does* name — and read
/// here from that scope's park.
///
/// The value is already in the enclosing scope's schedule: a fold's closure
/// was carved out of its parent's, and a value invariant in the fold has no
/// bit deeper than the parent's, so the parent kept it. What changes is that
/// the fold's own copy becomes a placeholder, and the value joins the
/// computing scope's `roots`.
fn place_roots(scoped: &mut ScopedSchedule, variance: &[pixelflow_ir::variance::Variance]) {
    use pixelflow_ir::variance::Variance;

    // What each scope binds, read off before any def is edited: a fold's own
    // binder. The same index answers for a fold as a scope and as an
    // ancestor.
    //
    // Not the lane binder of a store the scope holds. The lane fold is
    // inlined into the storing scope, so nothing *deeper* may compute a
    // lane-varying value (`extract_folds_bound_by` keeps them out of the
    // folds within) — but the lanes themselves are the same vector in every
    // batch, so a value that varies by lane and by nothing else is invariant
    // over the whole call and belongs to the body, like any other invariant.
    // Counting the lane as bound here made the storing scope keep its own
    // copy of the iota while the body parked another for the scopes inside,
    // and one scope then held the same value as a def and as a live-in.
    let body_binds = Variance::CONST;
    let binds: Vec<Variance> = (0..scoped.folds.len())
        .map(|j| Variance::from_var(binder_of_fold(scoped, j)))
        .collect();
    for j in 0..scoped.folds.len() {
        let own = binds[j];
        // Ancestors, nearest first, each with what it binds; the body is
        // where the chain ends.
        let mut ancestors: Vec<(Scope, Variance)> = Vec::new();
        let mut up = scoped.folds[j].parent;
        loop {
            match up {
                Scope::Body => {
                    ancestors.push((Scope::Body, body_binds));
                    break;
                }
                Scope::Fold(p) => {
                    ancestors.push((up, binds[p]));
                    up = scoped.folds[p].parent;
                }
            }
        }
        let mut moved: Vec<(Scope, ValueId)> = Vec::new();
        for def in &mut scoped.folds[j].schedule {
            let deps = variance[def.value.0 as usize];
            if deps.bits() & own.bits() != 0 || stays_put(&def.op) {
                continue;
            }
            // The outermost ancestor binding every binder `deps` names: the
            // nearest one binding any of them, or the body when none does —
            // an ancestor's binder in `deps` puts the value inside that
            // ancestor, and outside every scope nearer than it.
            let computing = ancestors
                .iter()
                .find(|(_, binds)| deps.bits() & binds.bits() != 0)
                .map_or(Scope::Body, |(scope, _)| *scope);
            moved.push((computing, def.value));
            // The placeholder; never emitted, located at the park. A vector's
            // says nothing about the value it stands for; a pointer's stays
            // its own op, which is operand-free already, so the scope inside
            // still reads the class off it.
            if def.op.class() == Class::Vector {
                def.op = ScheduledOp::Const(0.0);
            }
        }
        for (computing, vid) in moved {
            let roots = match computing {
                Scope::Body => &mut scoped.body.roots,
                Scope::Fold(p) => &mut scoped.folds[p].roots,
            };
            if !roots.contains(&vid) {
                roots.push(vid);
            }
        }
    }
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

/// A fold's binder, read off the `Reduce` def it opens at in its parent.
fn binder_of_fold(scoped: &ScopedSchedule, j: usize) -> u8 {
    let fold = &scoped.folds[j];
    let def = match fold.parent {
        Scope::Body => &scoped.body.schedule[fold.at],
        Scope::Fold(p) => &scoped.folds[p].schedule[fold.at],
    };
    let ScheduledOp::Reduce(meta, _) = &def.op else {
        panic!("Fold({j}) opens at a def that is not a Reduce")
    };
    meta.binder().var()
}

/// A fold `extract_folds` carved out of a flat schedule, still looking for
/// its position: everything the body's closure computed, in topological
/// order, ending at the value the `Reduce` combines each iteration.
struct PendingFold {
    /// The `Reduce` def's own `ValueId` — a schedule permutation (LICM, arm
    /// clustering) may move it, but never renames it, so this is what
    /// `attach_folds` searches the post-partition schedule for.
    reduce_vid: ValueId,
    /// The fold's own per-iteration computation, in topological order,
    /// ending at the body's root.
    schedule: Vec<Def>,
    /// The folds whose `Reduce` def sits in `schedule`: a fold inside this
    /// one's body. The nest is a tree, and this is the recursion.
    children: Vec<PendingFold>,
}

/// Carve every surviving `Reduce`'s body out of `schedule`.
///
/// A fold inside a fold's body is carved the same way, one level down: the
/// outer fold's closure is a schedule like any other, and the inner fold is
/// a `Reduce` def in it. `variance` is indexed by `ValueId` (dense and
/// positional in the arena's own schedule — `arena_to_schedule` assigns
/// them in the order it pushes `Def`s), which is what lets one array answer
/// for every level.
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
/// binder) is not dropped: it stays here too, and stays in the fold's own
/// schedule as well — where [`place_roots`] turns it into a placeholder read
/// from the enclosing scope's park. Getting this backwards — removing the
/// whole closure — would orphan exactly that shared leaf's other consumer.
/// The closure ends at such a value: what it is built from is the computing
/// scope's business, not the fold's, and is not carried in behind it.
///
/// The one thing that is *not* recomputed inside a fold is another fold
/// that does not depend on its binder: a whole loop per iteration is the
/// glyph's winding sum run once per piece of its distance fold, and once
/// more at the top for the coverage that reads it. Such a `Reduce` stays
/// the enclosing scope's fold, and the fold that reads its result keeps its
/// def as a **placeholder** — in the schedule, so the reads resolve, but
/// opening no scope here; `allocate_nest` parks it in its accumulator slot,
/// where the enclosing scope's loop left it before this one began.
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
        let ScheduledOp::Reduce(fold, body_vid) = def.op else {
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
        // follows, yields it — so the recursion below can carve it out again
        // one level down. One that does not is a placeholder like any other
        // hoisted value, remembered so the level below does not mistake it
        // for a fold of its own.
        let mut mark = alloc::vec![false; n];
        let mut placeholder_here = alloc::vec![false; n];
        let op_of = |v: ValueId| position[v.0 as usize].map(|p| &schedule[p].op);
        let is_fold = |v: ValueId| matches!(op_of(v), Some(ScheduledOp::Reduce(..)));
        let mut stack = Vec::new();
        mark[body_vid.0 as usize] = true;
        // The root too: a body that *is* another fold's result reads that
        // result from its slot, and the whole schedule is the placeholder.
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
            if matches!(op, ScheduledOp::Reduce(..)) {
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
        let inside = bound.union(Variance::from_var(fold.binder().var()));
        let (fold_schedule, children) =
            extract_folds_bound_by(fold_schedule, variance, inside, &placeholder_here);
        pending.push(PendingFold {
            reduce_vid: def.value,
            schedule: fold_schedule,
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
        .filter(|def| variance[def.value.0 as usize].bits() & deeper == 0)
        .collect();
    (remaining, pending)
}

/// Locate each [`PendingFold`]'s `Reduce` def in the body's schedule and
/// record it as a [`ScopeFold`].
///
/// Searched by value rather than carried through as a position: the scope's
/// layout, which runs after this, permutes the schedule — it moves a `Def`,
/// never renames the `ValueId` it defines — and carries each fold's position
/// along itself ([`lay_out`]).
fn attach_folds(scoped: &mut ScopedSchedule, pending: Vec<PendingFold>) {
    for fold in pending {
        let at = scoped
            .body
            .schedule
            .iter()
            .position(|def| def.value == fold.reduce_vid)
            .unwrap_or_else(|| {
                panic!(
                    "{:?}'s Reduce def is not in the body after extract_folds",
                    fold.reduce_vid
                )
            });
        attach_fold(scoped, fold, Scope::Body, at);
    }
}

/// Record `fold` as a [`ScopeFold`] opening at `at` in `parent`,
/// then each of its children inside it — depth first, so a parent's index is
/// always below its children's, which is the order `allocate_nest` and the
/// frame layout both walk the tree in.
///
/// A child's position is a search of its parent's schedule, for the same
/// reason [`attach_folds`] searches rather than carries: the allocator keeps
/// a fold's evaluation order, but a position is a fact about a schedule and
/// this is the schedule it will be asked of.
fn attach_fold(scoped: &mut ScopedSchedule, fold: PendingFold, parent: Scope, at: usize) {
    let PendingFold {
        reduce_vid,
        schedule,
        children,
    } = fold;
    let index = scoped.folds.len();
    scoped.folds.push(ScopeFold {
        parent,
        at,
        roots: Vec::new(),
        schedule,
        guards: Vec::new(),
    });
    for child in children {
        let at = scoped.folds[index]
            .schedule
            .iter()
            .position(|def| def.value == child.reduce_vid)
            .unwrap_or_else(|| {
                panic!(
                    "{:?}'s Reduce def is not in {reduce_vid:?}'s body, which \
                     extract_folds said it was nested in",
                    child.reduce_vid
                )
            });
        attach_fold(scoped, child, Scope::Fold(index), at);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::schedule_for;
    use pixelflow_ir::LatticeShape;
    use pixelflow_ir::arena::{ExprArena, ExprId, UniformDecl, UniformIdentity};
    use pixelflow_ir::kind::OpKind;

    /// Lanes in one SIMD batch at the tier this host selected.
    fn lanes() -> usize {
        crate::isa::jit_vector_bytes() / core::mem::size_of::<f32>()
    }

    /// One full batch of one row: `x` runs `x0 .. x0 + lanes()`, which is
    /// what a test about per-lane behaviour needs.
    fn batch() -> LatticeShape {
        LatticeShape::new([lanes() as u32, 1])
    }

    /// [`schedule_for`] at this host's own lane count.
    fn native_schedule(a: &ExprArena, root: ExprId, shape: LatticeShape) -> Vec<Def> {
        schedule_for(a, root, shape, lanes() as u32)
    }

    /// `y·k + x·k`: one constant, read by a row-invariant term and a
    /// column-varying one, so the inner scope needs a value the outer one
    /// computes.
    fn shared_leaf_kernel() -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let k = a.push_const(3.5);
        let invariant = a.push_binary(OpKind::Mul, y, k);
        let varying = a.push_binary(OpKind::Mul, x, k);
        let root = a.push_binary(OpKind::Add, invariant, varying);
        (a, root)
    }

    fn decl(default: f32) -> UniformDecl {
        UniformDecl {
            id: UniformIdentity::mint(),
            default,
        }
    }

    /// A constant shared between a lattice-invariant expression and a varying
    /// one is computed by the outer scope and parked for the inner one, like
    /// any other value the inner scope reads but does not vary.
    ///
    /// It used to be computed in both: `place_roots` left a leaf where it was,
    /// on the theory that nothing is saved by parking a value one instruction
    /// rebuilds. Two instructions on x86, per read, per trip — and a parked
    /// root is carried in a register when one is free, which is none.
    #[test]
    fn a_leaf_feeding_both_scopes_is_parked_by_the_outer_one() {
        let (a, root) = shared_leaf_kernel();
        let schedule = native_schedule(&a, root, batch());
        let scoped = ScopedSchedule::from_schedule(schedule);

        let k = scoped
            .body
            .schedule
            .iter()
            .find(|d| matches!(d.op, ScheduledOp::Const(v) if v == 3.5))
            .map(|d| d.value)
            .expect("the body computes the constant");
        assert!(
            scoped.body.roots.contains(&k),
            "the body parks it for the fold: roots {:?}",
            scoped.body.roots
        );
        let inner: alloc::vec::Vec<&ScheduledOp> = scoped
            .folds
            .iter()
            .flat_map(|f| f.schedule.iter())
            .filter(|d| d.value == k)
            .map(|d| &d.op)
            .collect();
        assert!(
            !inner.is_empty()
                && inner
                    .iter()
                    .all(|op| matches!(op, ScheduledOp::Const(v) if *v == 0.0)),
            "the fold reads it through a placeholder, never its own copy: {inner:?}"
        );
    }

    /// `x + u·u`: the uniform's load and the product that depends on it
    /// alone are per-call work. Asserted on the nest — which scope holds
    /// them — not on timing.
    #[test]
    fn a_uniform_and_what_depends_on_it_alone_land_in_the_body() {
        let mut a = ExprArena::new();
        let u = a.declare_uniform(decl(3.0));
        let x = a.push_var(0);
        let uu = a.push_uniform(u);
        let sq = a.push_binary(OpKind::Mul, uu, uu);
        let root = a.push_binary(OpKind::Add, x, sq);

        let schedule = native_schedule(&a, root, batch());
        let scoped = ScopedSchedule::from_schedule(schedule);

        // The kernel's own uniform, told from the origin's two by the
        // block it is read from: the link's, at the context slot after
        // the (empty) buffer table, rather than the origin block after
        // that. The block's pointer is a `Context` def of its own,
        // loaded once per call in the body.
        let link_block = scoped
            .body
            .schedule
            .iter()
            .find(|d| matches!(d.op, ScheduledOp::Context(0)))
            .map(|d| d.value)
            .expect("the link's block pointer is loaded once per call");
        let kernel_uniform =
            |op: &ScheduledOp| matches!(op, ScheduledOp::Uniform(base, _) if *base == link_block);
        assert!(
            scoped.body.schedule.iter().any(|d| kernel_uniform(&d.op)),
            "the broadcast load is once per call"
        );
        let sq_vid = scoped
            .body
            .schedule
            .iter()
            .find(|d| match d.op {
                ScheduledOp::Binary(OpKind::Mul, l, r) => l == r,
                _ => false,
            })
            .map(|d| d.value)
            .expect("u·u is computed once per call");
        assert!(
            scoped.body.roots.contains(&sq_vid),
            "the product is parked for the scopes inside, not recomputed"
        );
        assert!(
            !scoped
                .folds
                .iter()
                .flat_map(|f| f.schedule.iter())
                .any(|d| kernel_uniform(&d.op)),
            "the folds read the parked product, never the block"
        );
    }
}
