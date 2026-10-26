//! Lowering: a legalized arena to the flat schedule the later stages read.
//!
//! The arena arrives in topological order by construction. Lowering keeps what
//! the root reaches, numbers the survivors ([`ValueId`]), and translates each
//! node to the [`ScheduledOp`] that defines it. It reads the arena and nothing
//! else: it names no register, no byte and no target, and takes the two origin
//! slots as a parameter rather than minting them.

use alloc::vec::Vec;

use pixelflow_ir::arena::{ExprId, ExprNode, UniformId};
use pixelflow_ir::fold::{Binder, Bound, BoundEnd, BoundRefusal, Fold, KnownRange, Monoid};
use pixelflow_ir::kind::OpKind;
use pixelflow_ir::variance::{Variance, compute_arena_variance};
use pixelflow_ir::{Lit, ShiftCount};

use super::lattice::Traversal;
use super::{CompareOp, Def, Legal, ScheduledOp, ValueId, lane_pattern, legal, not_legal};

/// Mark nodes reachable from `root` via DFS.
///
/// The arena may contain garbage nodes from junkify passes; only nodes
/// transitively referenced by `root` should appear in the schedule, except
/// through an empty fold, whose body is never evaluated.
fn mark_reachable(
    arena: &pixelflow_ir::arena::ExprArena,
    root: pixelflow_ir::arena::ExprId,
    reachable: &mut [bool],
) {
    let mut stack = alloc::vec![root];
    while let Some(id) = stack.pop() {
        let idx = id.0 as usize;
        if reachable[idx] {
            continue;
        }
        reachable[idx] = true;
        // An empty fold is its identity whatever its body says (see `arena_to_schedule`).
        if is_empty_fold(arena, id) {
            continue;
        }
        for child in arena.children(id) {
            if !reachable[child.0 as usize] {
                stack.push(child);
            }
        }
    }
}

/// Whether `id` is a fold over nothing: its monoid's identity, whatever its
/// body says. A range that is not known is not known to be empty, so it is
/// not one.
fn is_empty_fold(arena: &pixelflow_ir::arena::ExprArena, id: pixelflow_ir::arena::ExprId) -> bool {
    matches!(arena.node(id), pixelflow_ir::arena::ExprNode::Reduce { .. })
        && arena.known_range(id).is_some_and(KnownRange::is_empty)
}

/// What reads a fold whose bound is a value, as far as a schedule has met it.
#[derive(Clone, Copy)]
enum ValueFold {
    /// Not such a fold.
    Not,
    /// Such a fold that nothing reachable has read yet.
    Unread,
    /// Such a fold, and the node that read it first.
    ReadBy(ExprId),
}

/// The folds whose bound is a value that the schedule reaches, and who reads
/// each: the single-reader property placement rests on.
///
/// A fold whose bound is a value arrives inside the guard `passes::legalize_back`
/// ends with, `If(dom, If(Lt(lo, hi), fold, identity), NaN)`, and the fold's
/// one reader is the inner `If`, as its true arm. That makes the fold the
/// arm's alone to own, so its loop sits in an arm block and runs only for a
/// bound that names a count. A second reader, of any shape, would own the
/// fold at the two readers' lowest common ancestor, outside the arm, and the
/// loop would run unguarded: a bound of `+∞` or `2²⁴ + 2` stalls its counter
/// at `2²⁴`, which never reaches the end.
struct ValueFolds {
    /// By arena id; empty when no reachable fold has a value bound.
    readers: Vec<ValueFold>,
}

impl ValueFolds {
    /// The folds `ids`, in an arena of `len` nodes, none read yet.
    fn unread(len: usize, ids: &[ExprId]) -> Self {
        if ids.is_empty() {
            return Self {
                readers: Vec::new(),
            };
        }
        let mut readers = alloc::vec![ValueFold::Not; len];
        for id in ids {
            readers[id.0 as usize] = ValueFold::Unread;
        }
        Self { readers }
    }

    /// Record `reader`'s reads of these folds, and check each.
    ///
    /// # Panics
    ///
    /// If `reader` is a second reader of a fold, or reads one other than as
    /// the true arm of an `If` (not the mask, and not the false arm too):
    /// the arena was not legalized, which is a pipeline bug and not a
    /// kernel's.
    fn read_by(&mut self, arena: &pixelflow_ir::arena::ExprArena, reader: ExprId) {
        // An empty fold's children are not reached (`mark_reachable`).
        if self.readers.is_empty() || is_empty_fold(arena, reader) {
            return;
        }
        for fold in arena.children(reader) {
            match self.readers[fold.0 as usize] {
                ValueFold::Not => continue,
                ValueFold::ReadBy(first) => panic!(
                    "arena_to_schedule: {fold:?}, a fold whose bound is a value, is read by \
                     {first:?} and by {reader:?}: exactly one reader, an If's true arm, is the \
                     guard that owns it (passes::legalize_back ends with the guard, L5)"
                ),
                ValueFold::Unread => self.readers[fold.0 as usize] = ValueFold::ReadBy(reader),
            }
            let as_an_arm = matches!(
                arena.node(reader),
                ExprNode::Ternary(OpKind::If, mask, if_true, if_false)
                    if if_true == fold && mask != fold && if_false != fold
            );
            assert!(
                as_an_arm,
                "arena_to_schedule: {fold:?}, a fold whose bound is a value, is read by \
                 {reader:?} other than as an If's true arm: an arena that was not legalized \
                 (passes::legalize_back ends with the guard that owns such a fold, L5)"
            );
        }
    }

    /// Check, after the walk, that every value-bounded fold found a reader.
    ///
    /// # Panics
    ///
    /// If a fold was read by nothing: it is the root, or its readers were not
    /// reached, and either way no guard owns it.
    fn assert_every_fold_was_read(&self) {
        let unread = self
            .readers
            .iter()
            .position(|fold| matches!(fold, ValueFold::Unread));
        if let Some(fold) = unread {
            panic!(
                "arena_to_schedule: ExprId({fold}), a fold whose bound is a value, is read by \
                 nothing: an arena that was not legalized (passes::legalize_back ends with the \
                 guard that owns such a fold, L5)"
            );
        }
    }
}

/// L3 after collapse: a fold's bound varies with no binder of the lattice.
///
/// A bound reads no lattice binder, so it is fixed before the lattice is
/// traversed: one value for the call or, where it is an enclosing fold's index,
/// one value per trip of that fold, and either way one value in every lane of a
/// batch. `collapse` picks the lattice's binder slots from those no reachable
/// `Var` or fold uses, so this cannot fail on a legalized arena. It is also
/// what keeps every lane of a guard's mask and of a latch's compare at one
/// value: the lane binder would vary them within a batch, and a guard's arm
/// would run on some of its lanes; the row and column binders would vary them
/// between batches, which the law forbids all the same.
fn assert_off_the_lattice(end: BoundEnd, reads: Variance, lattice: Variance) {
    let shared = reads.intersection(lattice);
    assert!(
        shared.is_const(),
        "arena_to_schedule: a fold's {end} bound varies with {shared:?}, a lattice binder's: \
         L3 after collapse (docs/language/2026-10-20-a-bound-is-a-value.md); a pipeline bug, \
         not a kernel's"
    );
}

/// Which values of a schedule are masks ([`ScheduledOp::is_mask`]), worked out
/// as far as a fold's bound asks, in schedule order.
struct MaskValues(Vec<bool>);

impl MaskValues {
    /// Whether `value`, a def of `schedule`, is a mask.
    fn is_mask(&mut self, schedule: &[Def], value: ValueId) -> bool {
        for def in &schedule[self.0.len()..] {
            assert_eq!(
                def.value.0 as usize,
                self.0.len(),
                "a schedule numbers its defs in order"
            );
            let mask = def.op.is_mask(|operand| self.0[operand.0 as usize]);
            self.0.push(mask);
        }
        self.0[value.0 as usize]
    }
}

/// Narrow a `Const` shift count to the `u8` immediate the hardware encoders
/// take, refusing anything a 32-bit lane cannot be shifted by.
///
/// The check belongs HERE, on the `f32`, because the narrowing is lossy in a
/// way that manufactures a legal-looking value: `256.0 as u32 as u8` is `0`,
/// so a count no target can honour would arrive at the encoder disguised as
/// the identity shift. Any later validation is checking the alias, not the
/// operand the kernel actually asked for.
fn shift_immediate(op: OpKind, count: f32) -> u8 {
    let Some(count) = ShiftCount::from_literal(count) else {
        panic!(
            "{op:?} shift count {count:?} is not a ShiftCount (an integer literal in \
             0..{}) — a 32-bit lane has no bits there, and the targets disagree \
             about what to do (x86 zeroes the whole destination, aarch64 \
             re-encodes the element size)",
            ShiftCount::WORD_BITS
        );
    };
    // Below `WORD_BITS`, so it fits.
    count.bits() as u8
}

/// Build a schedule directly from an [`ExprArena`](pixelflow_ir::arena::ExprArena).
///
/// The arena stores nodes in topological order (children before parents by
/// construction). We filter to reachable nodes, remap `ExprId` to `ValueId`,
/// and translate `ExprNode` to `ScheduledOp`.
///
/// The arena is a legalized one with no coordinate `Var` left; `traversal`
/// is how the CPU walks the lattice it is spelled through. The arena holds
/// only the kernel: the traversal's loops are written here, after the
/// kernel's defs, as the schedule the arena used to carry as nodes.
///
/// - **The lane fold is a store.** The lane binder's fold is executed by
///   lanes; it has no loop, so it is a [`ScheduledOp::Write`] def of the
///   kernel's value with the fold's trip count as the store width. A row's
///   main batches and its remainder are two `Write` defs of different widths
///   reading one value, each the body of its column fold, sequenced by a
///   [`ScheduledOp::Seq`] under the row fold.
/// - **The lane binder is a constant.** Its `Var` becomes
///   [`ScheduledOp::Lanes`], the iota every lane-varying value is built on.
///
/// `None` is an arena with no lattice (the lowering tests' raw arenas):
/// nothing is stored, and every read stays a gather.
///
/// A fold whose bound is a value arrives inside its guard (`passes::legalize_back`'s
/// last pass): its bounds are operands ([`operands`](super::operands)), and it
/// is read only as an `If`'s true arm.
///
/// Every def carries the variance of the node it was lowered from, which
/// scoping places it by. A block's base pointer has no node: it is the call's,
/// `CONST`.
///
/// `origin` is the uniform slots of the two origin scalars, which read
/// from the context entry after the link's block rather than from it.
///
/// # Panics
///
/// Panics if a `Param` or `Nary` node is encountered (these are not expected
/// in JIT compilation), or a `Var` that is no fold's binder survived
/// lowering: a coordinate or a retired axis, which means the lowering
/// pipeline was skipped, or an index past the binder slots, which means the
/// kernel is open. Such an index is a fold index that escaped its builder (a
/// placeholder, which the variance this reads types as constant) or an index
/// past the index space; either way this is the refusal that stops one from
/// being placed.
///
/// It also panics, for a fold whose bound is a value, when a bound reads a
/// binder of the lattice (L3 after collapse) or the fold is read other than
/// once, as an `If`'s true arm: an arena that was not legalized, a pipeline
/// bug and not a kernel's. A bound the emitter holds as a mask
/// (`ScheduledOp::is_mask`: a comparison, or a bitwise and/or of two such) is
/// refused by name (`BoundSort`), at either end and on every tier alike: a mask
/// names no count.
pub(crate) fn arena_to_schedule(
    arena: &pixelflow_ir::arena::ExprArena,
    root: pixelflow_ir::arena::ExprId,
    origin: [UniformId; 2],
    traversal: Option<&Traversal>,
) -> Vec<Def> {
    let len = arena.len();
    let mut reachable = alloc::vec![false; len];
    mark_reachable(arena, root, &mut reachable);

    // The lane binder and the binders the lattice's loops bind, from the
    // traversal; and the folds that have a bound that is a value, from a scan.
    let lane = traversal.map(|t| t.lane);
    let lattice = traversal.map_or(Variance::CONST, Traversal::variance);
    let mut value_folds = Vec::new();
    for (idx, _) in reachable.iter().enumerate().filter(|(_, r)| **r) {
        let id = ExprId(idx as u32);
        match arena.node(id) {
            ExprNode::Reduce { lo, hi, .. } if lo.value().is_some() || hi.value().is_some() => {
                value_folds.push(id);
            }
            _ => {}
        }
    }
    let mut folds = ValueFolds::unread(len, &value_folds);
    let mut masks = MaskValues(Vec::new());

    // What every value varies with, read once: each def carries its
    // node's, and a `RawGather` whose index lacks the lane binder's bit is
    // one load broadcast, not a gather. No lane fold (an arena `collapse`
    // never wrapped, the emit tests' raw arenas) means nothing is known to
    // be lane-uniform, and every read stays a gather.
    let variance = compute_arena_variance(arena);
    let lane_uniform =
        |idx: ExprId| lane.is_some_and(|lane| variance[idx.0 as usize].is_invariant_in(lane.var()));

    // ExprId to ValueId mapping. u64::MAX = unmapped (unreachable).
    let mut id_map = alloc::vec![ValueId(u64::MAX); len];
    let mut schedule = Vec::new();
    let mut next_id = 0;

    let buffers = u16::try_from(arena.buffers().len())
        .expect("buffer table index fits the context slot immediate");
    // The uniform blocks' base pointers, one `Context` def each, made on
    // first use: a block has no arena node to map, unlike a buffer, whose
    // `Buffer` leaf is its `Context` def.
    let mut blocks: alloc::collections::BTreeMap<u16, ValueId> =
        alloc::collections::BTreeMap::new();

    for idx in 0..len {
        if !reachable[idx] {
            continue;
        }
        let expr_id = ExprId(idx as u32);
        let node = arena.node(expr_id);
        folds.read_by(arena, expr_id);

        let map_child = |child: ExprId| -> ValueId {
            let mapped = id_map[child.0 as usize];
            assert!(
                mapped.0 != u64::MAX,
                "arena_to_schedule: child ExprId({}) not yet mapped -- \
                 arena is not in topological order or child is unreachable",
                child.0
            );
            mapped
        };

        let sched_op = match node {
            ExprNode::Var(i) => match Binder::from_var(i) {
                None if i < Variance::VARIABLES => panic!(
                    "arena_to_schedule: Var({i}) is a coordinate or a retired axis, which \
                     the lowering pipeline substitutes away or refuses -- this schedule was \
                     built without it"
                ),
                None => panic!(
                    "arena_to_schedule: Var({i}) is past the binder slots: a fold index that \
                     escaped its builder, or an index past the index space. No fold binds it, \
                     so the kernel is open"
                ),
                Some(b) if lane == Some(b) => ScheduledOp::Lanes,
                Some(b) => ScheduledOp::Var(b),
            },
            ExprNode::Const(lit) => ScheduledOp::Const(lane_pattern(lit)),
            ExprNode::Param(i) => {
                panic!("ExprNode::Param({i}) reached the JIT emitter: a `Param` has no value")
            }
            // A buffer's base pointer: the `k`-th context entry, a pointer
            // value the gathers reading the buffer take as an operand.
            ExprNode::Buffer(id) => ScheduledOp::Context(id.0),
            // The link's block sits in the context entry after the buffer
            // slots and the origin's in the one after that; a value's offset
            // is its slot index within its block — the link step
            // (`jit_cache`) renumbers the table into dense first-occurrence
            // order before anything reaches here, and the two origin slots
            // are the last two, declared by `collapse` after the relink.
            // The block's base is a `Context` def made here on first use,
            // ahead of this def so the schedule stays topological.
            ExprNode::Uniform(u) => {
                let (ctx_slot, offset) = match origin.iter().position(|&o| o == u) {
                    Some(axis) => (buffers + 1, axis as u64),
                    None => (buffers, u.0),
                };
                let block = *blocks.entry(ctx_slot).or_insert_with(|| {
                    let base = ValueId(next_id);
                    next_id += 1;
                    schedule.push(Def {
                        value: base,
                        op: ScheduledOp::Context(ctx_slot),
                        // No node: the call's pointer, the same at every sample.
                        variance: Variance::CONST,
                    });
                    base
                });
                ScheduledOp::Uniform(block, offset)
            }
            // A reinterpretation is its operand: the same lanes, read as the
            // other sort. It defines nothing and takes no `ValueId`; whatever
            // reads it reads the operand, so no backend sees it and no `mov`
            // is spent on it.
            ExprNode::Unary(OpKind::Bits | OpKind::Float, child) => {
                let operand = map_child(child);
                id_map[idx] = operand;
                continue;
            }
            ExprNode::Unary(op, child) => ScheduledOp::Unary(legal(op), map_child(child)),
            // Shl/Shr fold their Const shift-count operand into an immediate, so
            // the count never becomes a scheduled value (matching the imm-only
            // hardware shift encoders). The count const may still appear as its
            // own schedule entry (harmless/unused) if shared.
            ExprNode::Binary(op @ (OpKind::Shl | OpKind::Shr), a, b) => {
                let amount = match arena.node(b) {
                    ExprNode::Const(Lit::Real(v)) => shift_immediate(op, v),
                    _ => panic!(
                        "{:?} shift count must be a real Const (lowering guarantees this)",
                        op
                    ),
                };
                ScheduledOp::ShiftImm(legal(op), map_child(a), amount)
            }
            // RawGather's buffer leaf is its base pointer's def, mapped like
            // any other child. An index the lane binder does not reach is one
            // address for the whole batch, and the read is a broadcast load
            // rather than a gather (see `ScheduledOp::Broadcast`).
            ExprNode::Binary(OpKind::RawGather, buf, idx) => {
                assert!(
                    matches!(arena.node(buf), ExprNode::Buffer(_)),
                    "RawGather's first child must be a Buffer leaf, got {:?}",
                    arena.node(buf)
                );
                if lane_uniform(idx) {
                    ScheduledOp::Broadcast(map_child(idx), map_child(buf))
                } else {
                    ScheduledOp::Gather(map_child(idx), map_child(buf))
                }
            }
            // Unreachable precondition: every compile entry point runs
            // `passes::lower_dwrt` before scheduling, which either rewrites
            // all `Dwrt` (autodiff) nodes into chain-rule arithmetic or errors
            // loudly on an op it cannot differentiate. A `Dwrt` here means a
            // caller bypassed that pipeline. Fail loudly rather than as a
            // cryptic instruction-emit panic.
            ExprNode::Binary(OpKind::Dwrt, _, _) => panic!(
                "arena_to_schedule: a Dwrt (autodiff) node reached the JIT \
                 emitter. lower_dwrt runs in every compile entry point and \
                 either eliminates Dwrt or refuses to compile, so a survivor \
                 means this schedule was built without the lowering pipeline."
            ),
            ExprNode::Binary(op, a, b) => match CompareOp::of(op) {
                Some(compare) => ScheduledOp::Compare(compare, map_child(a), map_child(b)),
                None => ScheduledOp::Binary(legal(op), map_child(a), map_child(b)),
            },
            ExprNode::Ternary(OpKind::MulAdd, a, b, c) => {
                ScheduledOp::MulAdd(map_child(a), map_child(b), map_child(c))
            }
            ExprNode::Ternary(OpKind::If, a, b, c) => {
                ScheduledOp::If(map_child(a), map_child(b), map_child(c))
            }
            ExprNode::Ternary(op, ..) => not_legal(op),
            // Same unreachable precondition as `Dwrt` above: `passes::legalize_back`
            // runs `expand_refs` first in every compile entry point, so a
            // reference here means this schedule was built without the
            // lowering pipeline. Refusing is not a limitation to lift — a
            // surviving reference is a *call*, and codegen emits one flat
            // function per kernel with no ABI for one
            // (docs/language/2026-09-09-composition-is-linking.md §5.2).
            ExprNode::Ref(key) => panic!(
                "arena_to_schedule: {key:?} names a kernel whose body is not in \
                 this arena. expand_refs runs first in every compile entry \
                 point, so a survivor means this schedule was built without \
                 the lowering pipeline."
            ),
            ExprNode::Nary(_, _) => panic!("Nary not supported in JIT arena compilation"),
            // Its body was not reached (`mark_reachable`), and the arena's
            // variance for it is `CONST` already: the identity's.
            ExprNode::Reduce { fold, .. } if is_empty_fold(arena, expr_id) => {
                ScheduledOp::Const(lane_pattern(fold.monoid().identity()))
            }
            // A surviving fold: `body` was already walked above (it is an
            // ordinary child, scheduled before its parent by the arena's own
            // topological order), so `map_child(body)` is that per-iteration
            // value's `ValueId` in *this* numbering. `scopes::extract_folds` reads
            // it back out into the fold's own `ScopeFold`; nothing after
            // that resolves it as an operand (see `ScheduledOp::Reduce`). A
            // value bound is an operand of its fold (`operands`): `lo` seeds
            // the loop and `hi` is read at its latch.
            ExprNode::Reduce { fold, lo, hi, body } => {
                for (end, bound) in [(BoundEnd::Lo, lo), (BoundEnd::Hi, hi)] {
                    let Bound::Value(term) = bound else { continue };
                    let reads = variance[term.0 as usize];
                    if let Err(refusal) = fold.admits_bound(end, reads) {
                        panic!("arena_to_schedule: {refusal}");
                    }
                    assert_off_the_lattice(end, reads, lattice);
                    // A mask names no count, at either end, in whichever
                    // file a target keeps it: `is_mask` asks, before
                    // selection, what the emitter will hold as one.
                    assert!(
                        !masks.is_mask(&schedule, map_child(term)),
                        "arena_to_schedule: {}: the bound is a mask (a comparison, or a \
                         bitwise op on masks), which names no count; until a bound is typed \
                         (T6 of docs/language/2026-10-17-the-arena-is-typed.md) it is refused \
                         here, on every tier",
                        BoundRefusal::BoundSort { end }
                    );
                }
                ScheduledOp::Reduce {
                    fold,
                    lo: lo.map(map_child),
                    hi: hi.map(map_child),
                    body: map_child(body),
                }
            }
        };
        // Numbered after the op is built: a `Uniform` may have pushed its
        // block's `Context` def just above, and ids follow schedule order.
        let vid = ValueId(next_id);
        next_id += 1;
        id_map[idx] = vid;
        schedule.push(Def {
            value: vid,
            op: sched_op,
            variance: variance[idx],
        });
    }
    folds.assert_every_fold_was_read();
    if let Some(traversal) = traversal {
        let value = id_map[root.0 as usize];
        write_traversal(&mut schedule, traversal, value, variance[root.0 as usize]);
    }
    schedule
}

/// The traversal's loops, after the kernel's defs: for each of the row's
/// pieces ([`Traversal::pieces`]) a `Write` of `value` (the lane fold,
/// executed by lanes) under its column fold, the pieces sequenced under the
/// row fold. `kernel` is what the kernel's value varies with; each def's
/// variance is the fold's rule (`transfer` in `pixelflow-ir`'s `variance.rs`):
/// what its body reads, less the binder it closes.
fn write_traversal(
    schedule: &mut Vec<Def>,
    traversal: &Traversal,
    value: ValueId,
    kernel: Variance,
) {
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
                fold: piece.col,
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
            fold: Fold::new(Monoid::SEQ, traversal.row),
            lo: Bound::Lit(0),
            hi: Bound::Lit(u64::from(traversal.extent[1])),
            body: walk,
        },
        variance.without(bit(traversal.row)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::operands;
    use pixelflow_ir::arena::ExprArena;
    use pixelflow_ir::fold::{Fold, Monoid};

    /// The uniform slots a schedule built by hand names for the origin.
    ///
    /// A raw arena declares no uniform at all, so `origin_slots` has
    /// nothing to find; the tests below that feed the scheduler an
    /// unlegalized arena on purpose name the slots themselves.
    const RAW_ORIGIN: [UniformId; 2] = [UniformId(0), UniformId(1)];

    /// A `Dwrt` that reaches the scheduler (a caller bypassed the lowering
    /// pipeline) must fail loudly at the schedule boundary, not as a cryptic
    /// emit panic. The compile entry points run `lower_dwrt` first, so this
    /// exercises calling `arena_to_schedule` directly.
    ///
    /// Its operand is a fold binder rather than `X`: a coordinate `Var` is a
    /// survivor of its own, refused a line earlier (see
    /// `a_surviving_coordinate_fails_loudly`), and would answer for the
    /// `Dwrt` before the `Dwrt` was ever reached.
    #[test]
    #[should_panic(expected = "Dwrt (autodiff) node reached the JIT")]
    fn surviving_dwrt_fails_loudly() {
        let mut a = ExprArena::new();
        let i = a.push_var(Binder::from_slot(0).expect("slot 0 exists").var());
        let v = a.push_const(0.0);
        let root = a.push_binary(OpKind::Dwrt, i, v);
        let _ = arena_to_schedule(&a, root, RAW_ORIGIN, None);
    }

    /// A coordinate `Var` is the survivor the collapse ABI added: the
    /// lattice's folds substitute `X` and `Y` away, so one reaching the
    /// scheduler is an arena that never went through `program::lattice`.
    #[test]
    #[should_panic(expected = "is a coordinate")]
    fn a_surviving_coordinate_fails_loudly() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let two = a.push_const(2.0);
        let root = a.push_binary(OpKind::Mul, x, two);
        let _ = arena_to_schedule(&a, root, RAW_ORIGIN, None);
    }

    /// A placeholder is a fold index that escaped its builder: `Variance::of_var`
    /// types it as constant, so placement would not tell it from a uniform, and
    /// the refusal is this `Var` arm's. It answers for the open term before
    /// anything is scheduled, whatever the variance says, and as an open
    /// kernel, not as a schedule built without the lowering pipeline.
    #[test]
    #[should_panic(expected = "so the kernel is open")]
    fn a_surviving_placeholder_fails_loudly() {
        let mut a = ExprArena::new();
        let index = a.push_var(
            pixelflow_ir::fold::Placeholder::nth(0)
                .expect("placeholder 0 exists")
                .var(),
        );
        let two = a.push_const(2.0);
        let root = a.push_binary(OpKind::Mul, index, two);
        let _ = arena_to_schedule(&a, root, RAW_ORIGIN, None);
    }

    /// And the same for a `Ref`: its body is not in this arena at all, so a
    /// survivor is a schedule built without `expand_refs`. `compile` runs
    /// `legalize` first, so this too has to call the scheduler directly.
    #[test]
    #[should_panic(expected = "names a kernel whose body is not in")]
    fn a_surviving_reference_fails_loudly() {
        let named = pixelflow_ir::Kernel::x()
            .mul(&pixelflow_ir::Kernel::constant(3.0))
            .by_ref();
        let (arena, root) = named.parts();
        let _ = arena_to_schedule(arena, root, RAW_ORIGIN, None);
    }

    /// The uniform slots after the one a test below declares for a count, so
    /// that the count is not read as an origin axis.
    const AFTER_ONE_UNIFORM: [UniformId; 2] = [UniformId(1), UniformId(2)];

    /// The bound that is the count itself.
    fn the_count(_: &mut ExprArena, count: ExprId) -> ExprId {
        count
    }

    /// The literal at the end of a range a test bounds by a value at the other.
    const OTHER_END: u64 = 8;

    /// A count, a fold bounded at `at` by `end(count)`, and a guard around it:
    /// `If(Lt(0, count), Σ_i i, 0)`, as `(arena, count, fold, root)`. The fold's
    /// other bound is a literal: `0` below, [`OTHER_END`] above. The `If`'s test
    /// is a stand-in, not the `Lt(lo, hi)` that `passes::legalize_back` builds:
    /// lowering asks only that the fold has one reader, an `If`'s true arm.
    ///
    /// The body is the binder's own `Var`, not `X`: a coordinate is a
    /// survivor refused first, and would answer for the bound.
    fn a_guarded_fold(
        at: BoundEnd,
        end: impl FnOnce(&mut ExprArena, ExprId) -> ExprId,
    ) -> (ExprArena, ExprId, ExprId, ExprId) {
        let mut a = ExprArena::new();
        let slot = a.declare_uniform(pixelflow_ir::Uniform::new(3.0).decl());
        let count = a.push_uniform(slot);
        let end = Bound::Value(end(&mut a, count));
        let (lo, hi) = match at {
            BoundEnd::Lo => (end, Bound::Lit(OTHER_END)),
            BoundEnd::Hi => (Bound::Lit(0), end),
        };
        let binder = Binder::from_slot(0).expect("slot 0 exists");
        let i = a.push_var(binder.var());
        let fold = a.push_reduce(Fold::new(Monoid::SUM, binder), lo, hi, i);
        let zero = a.push_const(0.0);
        let runs = a.push_binary(OpKind::Lt, zero, count);
        let root = a.push_ternary(OpKind::If, runs, fold, zero);
        (a, count, fold, root)
    }

    /// A bound that is a value is an operand of its fold, so the carve,
    /// ownership and the closure check see it as a read of the `Reduce`.
    #[test]
    fn a_value_bound_is_an_operand_of_its_fold() {
        let (a, _, _, root) = a_guarded_fold(BoundEnd::Hi, the_count);
        let schedule = arena_to_schedule(&a, root, AFTER_ONE_UNIFORM, None);

        let reduce = schedule
            .iter()
            .find(|def| matches!(def.op, ScheduledOp::Reduce { .. }))
            .expect("the fold survives");
        let ScheduledOp::Reduce { lo, hi, .. } = reduce.op else {
            unreachable!("matched a Reduce above")
        };
        let Bound::Value(count) = hi else {
            panic!("the end is the count: {hi:?}")
        };
        assert_eq!(lo, Bound::Lit(0));
        let loads = schedule
            .iter()
            .find(|def| def.value == count)
            .map(|d| &d.op);
        assert!(
            matches!(loads, Some(ScheduledOp::Uniform(..))),
            "the end names the uniform's load, not a copy: {loads:?}"
        );
        assert_eq!(operands(&reduce.op).collect::<Vec<_>>(), [count]);
    }

    /// A value-bounded fold read by an `Add` is one nothing guarded, and the
    /// loop would run for any count: the arena was not legalized.
    #[test]
    #[should_panic(expected = "other than as an If's true arm")]
    fn an_unguarded_value_bounded_fold_fails_loudly() {
        let (mut a, _, fold, _) = a_guarded_fold(BoundEnd::Hi, the_count);
        let one = a.push_const(1.0);
        let root = a.push_binary(OpKind::Add, fold, one);
        let _ = arena_to_schedule(&a, root, AFTER_ONE_UNIFORM, None);
    }

    /// An `If`'s false arm is not its true arm: the guard skips the loop for
    /// the wrong mask.
    #[test]
    #[should_panic(expected = "other than as an If's true arm")]
    fn a_value_bounded_fold_in_the_false_arm_fails_loudly() {
        let (mut a, count, fold, _) = a_guarded_fold(BoundEnd::Hi, the_count);
        let zero = a.push_const(0.0);
        let runs = a.push_binary(OpKind::Lt, zero, count);
        let root = a.push_ternary(OpKind::If, runs, zero, fold);
        let _ = arena_to_schedule(&a, root, AFTER_ONE_UNIFORM, None);
    }

    /// Two `If`s that both read the fold as their true arm each pass the
    /// shape check, and own the fold between them at their common ancestor,
    /// outside both arms: the loop would run unguarded, and a count of `+∞`
    /// would stall its counter at `2²⁴` for ever. The reader count is what
    /// catches it, whatever the readers' shape.
    #[test]
    #[should_panic(expected = "and by")]
    fn a_value_bounded_fold_with_two_guards_fails_loudly() {
        let (mut a, count, fold, guarded) = a_guarded_fold(BoundEnd::Hi, the_count);
        let one = a.push_const(1.0);
        let runs = a.push_binary(OpKind::Lt, one, count);
        let other = a.push_ternary(OpKind::If, runs, fold, one);
        let root = a.push_binary(OpKind::Add, guarded, other);
        let _ = arena_to_schedule(&a, root, AFTER_ONE_UNIFORM, None);
    }

    /// A value-bounded fold that is the root has no reader at all.
    #[test]
    #[should_panic(expected = "is read by nothing")]
    fn a_value_bounded_fold_nothing_reads_fails_loudly() {
        let (a, _, fold, _) = a_guarded_fold(BoundEnd::Hi, the_count);
        let _ = arena_to_schedule(&a, fold, AFTER_ONE_UNIFORM, None);
    }

    /// The fold's own guard is in the arena but unreachable from the root:
    /// what a rebuilt arena leaves behind (`rebuild_arena` appends and keeps
    /// the nodes it replaced). Only reachable readers count.
    #[test]
    fn a_reader_that_is_not_reachable_is_not_a_reader() {
        let (mut a, _, fold, guarded) = a_guarded_fold(BoundEnd::Hi, the_count);
        let one = a.push_const(1.0);
        let _stale = a.push_binary(OpKind::Add, fold, one);
        let _ = arena_to_schedule(&a, guarded, AFTER_ONE_UNIFORM, None);
    }

    /// L3 after collapse: a bound reads no coordinate, and collapse spells the
    /// coordinates through the lattice's binders, so a bound reads none of
    /// them. Only the lane binder would vary
    /// a guard's mask within a batch; `row` and `col` would vary it between
    /// batches, which is refused all the same. `collapse` picks its binders
    /// from slots nothing else uses, so no pipeline builds this; it is built by
    /// hand, with the traversal named beside it.
    ///
    /// `Σ_{i<row} i`, `row` being the lattice's.
    #[test]
    #[should_panic(expected = "a lattice binder's")]
    fn a_bound_on_a_lattice_binder_fails_loudly() {
        let slot = |n| Binder::from_slot(n).expect("the slot exists");
        let (i, row, col, lane) = (slot(0), slot(1), slot(2), slot(3));
        let mut a = ExprArena::new();
        let index = a.push_var(i.var());
        let rows = a.push_var(row.var());
        let fold = a.push_reduce(
            Fold::new(Monoid::SUM, i),
            Bound::Lit(0),
            Bound::Value(rows),
            index,
        );
        let zero = a.push_const(0.0);
        let runs = a.push_binary(OpKind::Lt, zero, rows);
        let root = a.push_ternary(OpKind::If, runs, fold, zero);
        let traversal = Traversal {
            row,
            col,
            lane,
            extent: [4, 1],
            lanes: 4,
        };
        let _ = arena_to_schedule(&a, root, RAW_ORIGIN, Some(&traversal));
    }

    /// What lowering a fold bounded at `at` by `end(count)` panics with, or
    /// `None` if it lowers.
    fn refusal_of_the_end(
        at: BoundEnd,
        end: impl FnOnce(&mut ExprArena, ExprId) -> ExprId,
    ) -> Option<String> {
        let (a, _, _, root) = a_guarded_fold(at, end);
        let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = arena_to_schedule(&a, root, AFTER_ONE_UNIFORM, None);
        }))
        .err()?;
        let message = match payload.downcast::<String>() {
            Ok(message) => *message,
            Err(payload) => payload
                .downcast_ref::<&str>()
                .map_or_else(String::new, |message| (*message).to_owned()),
        };
        Some(message)
    }

    /// Whether lowering refuses a fold bounded at `at` by `end(count)` by
    /// name, as a mask, and not for some other reason.
    fn is_refused_as_a_mask(
        at: BoundEnd,
        end: impl FnOnce(&mut ExprArena, ExprId) -> ExprId,
    ) -> bool {
        refusal_of_the_end(at, end).is_some_and(|message| message.contains("BoundSort"))
    }

    /// A mask names no count, so a bound the emitter holds as one is refused
    /// at either end and on every tier alike, by `ScheduledOp::is_mask`, which
    /// AVX-512's selector is held to: a comparison, a bitwise op on two masks, a
    /// reinterpretation of either. A bitwise op on a mask and a number is a
    /// number, and a count.
    #[test]
    fn a_bound_is_refused_exactly_when_the_emitter_holds_it_as_a_mask() {
        let lt = |a: &mut ExprArena, c| {
            let zero = a.push_const(0.0);
            a.push_binary(OpKind::Lt, zero, c)
        };
        let gt = |a: &mut ExprArena, c| {
            let one = a.push_const(1.0);
            a.push_binary(OpKind::Gt, c, one)
        };
        for at in [BoundEnd::Lo, BoundEnd::Hi] {
            assert!(is_refused_as_a_mask(at, lt), "{at}: a comparison is a mask");
            assert!(
                is_refused_as_a_mask(at, |a, c| {
                    let (x, y) = (lt(a, c), gt(a, c));
                    a.push_binary(OpKind::BitAnd, x, y)
                }),
                "{at}: a bitwise and of two masks is a mask"
            );
            assert!(
                is_refused_as_a_mask(at, |a, c| {
                    let (x, y) = (lt(a, c), gt(a, c));
                    let both = a.push_binary(OpKind::BitOr, x, y);
                    let wider = a.push_binary(OpKind::BitAnd, both, y);
                    a.push_binary(OpKind::BitOr, wider, x)
                }),
                "{at}: a bitwise op on masks is a mask, however deep"
            );
            assert!(
                is_refused_as_a_mask(at, |a, c| {
                    let mask = lt(a, c);
                    a.push_unary(OpKind::Bits, mask)
                }),
                "{at}: a reinterpretation defines no value: it is its operand, a mask"
            );
            assert_eq!(
                refusal_of_the_end(at, |a, c| {
                    let mask = lt(a, c);
                    let three = a.push_const(3.0);
                    a.push_binary(OpKind::BitAnd, mask, three)
                }),
                None,
                "{at}: a mask and a number is a number: 3 where it holds, 0 where it does not"
            );
            assert_eq!(
                refusal_of_the_end(at, |a, c| {
                    let one = a.push_const(1.0);
                    a.push_binary(OpKind::Add, c, one)
                }),
                None,
                "{at}: arithmetic is a number"
            );
            assert_eq!(
                refusal_of_the_end(at, the_count),
                None,
                "{at}: a uniform is a number"
            );
        }
    }

    /// The refusal names the law: a mask names no count.
    #[test]
    #[should_panic(expected = "BoundSort")]
    fn a_bound_that_is_a_mask_fails_by_name() {
        let (a, _, _, root) = a_guarded_fold(BoundEnd::Hi, |a, count| {
            let zero = a.push_const(0.0);
            a.push_binary(OpKind::Lt, zero, count)
        });
        let _ = arena_to_schedule(&a, root, AFTER_ONE_UNIFORM, None);
    }
}
