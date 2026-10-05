//! Lowering: a legalized arena to the flat schedule the later stages read.
//!
//! The arena arrives in topological order by construction. Lowering keeps what
//! the root reaches, numbers the survivors ([`ValueId`]), and translates each
//! node to the [`ScheduledOp`] that defines it. It reads the arena and nothing
//! else: it names no register, no byte and no target, and takes the two origin
//! slots as a parameter rather than minting them.

use alloc::vec::Vec;

use pixelflow_ir::arena::UniformId;
use pixelflow_ir::fold::Binder;
use pixelflow_ir::kind::OpKind;

use super::{Def, ScheduledOp, ValueId};

/// Mark nodes reachable from `root` via DFS.
///
/// The arena may contain garbage nodes from junkify passes; only nodes
/// transitively referenced by `root` should appear in the schedule.
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
        for child in arena.children(id) {
            if !reachable[child.0 as usize] {
                stack.push(child);
            }
        }
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
    assert!(
        (0.0..32.0).contains(&count) && (count as u32) as f32 == count,
        "{op:?} shift count {count} is not an integer in 0..32 — a 32-bit lane \
         has no bits there, and the targets disagree about what to do (x86 \
         zeroes the whole destination, aarch64 re-encodes the element size)"
    );
    count as u8
}

/// Build a schedule directly from an [`ExprArena`].
///
/// The arena stores nodes in topological order (children before parents by
/// construction). We filter to reachable nodes, remap `ExprId` to `ValueId`,
/// and translate `ExprNode` to `ScheduledOp`.
///
/// The arena is a legalized one: wrapped in the lattice's folds, with no
/// coordinate `Var` left. Two of its shapes are not copied one to one:
///
/// - **The lane fold is inlined.** A `Reduce` whose body is a `Write` naming
///   the fold's binder as its lane is the fold executed by lanes; it has no
///   loop, so its body is not carved into a scope of its own — its `Write`
///   becomes the fold's own def, in its parent's schedule, with the fold's
///   trip count folded in as the store width, and the arena's `Write` node
///   itself gets no `ValueId` (nothing but a lane fold names one). Two lane
///   folds sharing one `Write` — a row's main batches and its remainder —
///   are two `Write` defs of different widths reading one value.
/// - **The lane binder is a constant.** Its `Var` becomes
///   [`ScheduledOp::Lanes`], the iota every lane-varying value is built on.
///
/// `origin` is the uniform slots of the two origin scalars, which read
/// from the context entry after the link's block rather than from it.
///
/// # Panics
///
/// Panics if a `Param` or `Nary` node is encountered (these are not expected
/// in JIT compilation), or a coordinate `Var` survived `collapse`.
pub(crate) fn arena_to_schedule(
    arena: &pixelflow_ir::arena::ExprArena,
    root: pixelflow_ir::arena::ExprId,
    origin: [UniformId; 2],
) -> Vec<Def> {
    use pixelflow_ir::arena::{ExprId, ExprNode};

    let len = arena.len();
    let mut reachable = alloc::vec![false; len];
    mark_reachable(arena, root, &mut reachable);

    // The lane binder: the one every reachable `Write` names as its lane.
    // One slot, by construction (`passes::lattice::pack` builds both lane
    // folds on `collapse`'s), asserted rather than assumed.
    let mut lane: Option<Binder> = None;
    for (idx, _) in reachable.iter().enumerate().filter(|(_, r)| **r) {
        if let ExprNode::Write { lane: l, .. } = arena.node(ExprId(idx as u32)) {
            match lane {
                None => lane = Some(l),
                Some(seen) => assert_eq!(
                    seen, l,
                    "two Writes name different lane binders; a lattice has one lane fold"
                ),
            }
        }
    }

    // Which reads are the same in every lane. A `RawGather` whose index
    // lacks the lane binder's bit is one load broadcast, not a gather; the
    // bit is read here, where the two are split, because the schedule's
    // own variance (`scopes::schedule_variance`) is computed after it is built. No
    // lane fold — a schedule built from an arena `collapse` never wrapped,
    // the emit tests' raw arenas — means nothing is known to be
    // lane-uniform, and every read stays a gather.
    let variance = lane.map(|_| pixelflow_ir::variance::compute_arena_variance(arena));
    let lane_uniform = |idx: ExprId| -> bool {
        match (lane, &variance) {
            (Some(lane), Some(variance)) => variance[idx.0 as usize].is_invariant_in(lane.var()),
            _ => false,
        }
    };

    // ExprId to ValueId mapping. u32::MAX = unmapped (unreachable, or a
    // `Write` node, whose defs are its lane folds').
    let mut id_map = alloc::vec![ValueId(u32::MAX); len];
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
        if let ExprNode::Write { .. } = node {
            continue;
        }

        let map_child = |child: ExprId| -> ValueId {
            let mapped = id_map[child.0 as usize];
            assert!(
                mapped.0 != u32::MAX,
                "arena_to_schedule: child ExprId({}) not yet mapped -- \
                 arena is not in topological order or child is unreachable",
                child.0
            );
            mapped
        };

        let sched_op = match node {
            ExprNode::Var(i) => match Binder::from_var(i) {
                None => panic!(
                    "arena_to_schedule: Var({i}) is a coordinate, which \
                     passes::lattice::collapse substitutes away -- this schedule was \
                     built without the lowering pipeline"
                ),
                Some(b) if lane == Some(b) => ScheduledOp::Lanes(b),
                Some(_) => ScheduledOp::Var(i),
            },
            ExprNode::Const(v) => ScheduledOp::Const(v),
            ExprNode::Param(i) => panic!(
                "ExprNode::Param({}) reached the JIT emitter -- a template's slot (a `kernel!` \
                 template's structural hole, a rewrite rule's metavariable) is filled before \
                 a Kernel exists",
                i
            ),
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
                    });
                    base
                });
                ScheduledOp::Uniform(block, offset)
            }
            ExprNode::Unary(op, child) => ScheduledOp::Unary(op, map_child(child)),
            // Shl/Shr fold their Const shift-count operand into an immediate, so
            // the count never becomes a scheduled value (matching the imm-only
            // hardware shift encoders). The count const may still appear as its
            // own schedule entry (harmless/unused) if shared.
            ExprNode::Binary(op @ (OpKind::Shl | OpKind::Shr), a, b) => {
                let amount = match arena.node(b) {
                    ExprNode::Const(v) => shift_immediate(op, v),
                    _ => panic!(
                        "{:?} shift count must be a Const (lowering guarantees this)",
                        op
                    ),
                };
                ScheduledOp::ShiftImm(op, map_child(a), amount)
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
            ExprNode::Binary(OpKind::Seq, a, b) => ScheduledOp::Seq(map_child(a), map_child(b)),
            ExprNode::Binary(op, a, b) => ScheduledOp::Binary(op, map_child(a), map_child(b)),
            ExprNode::Ternary(op, a, b, c) => {
                ScheduledOp::Ternary(op, map_child(a), map_child(b), map_child(c))
            }
            // Same unreachable precondition as `Dwrt` above: `passes::legalize`
            // runs `expand_refs` first in every compile entry point, so a
            // reference here means this schedule was built without the
            // lowering pipeline. Refusing is not a limitation to lift — a
            // surviving reference is a *call*, and codegen emits one flat
            // function per kernel with no ABI for one
            // (docs/plans/2026-09-09-composition-is-linking.md §5.2).
            ExprNode::Ref(key) => panic!(
                "arena_to_schedule: {key:?} names a kernel whose body is not in \
                 this arena. expand_refs runs first in every compile entry \
                 point, so a survivor means this schedule was built without \
                 the lowering pipeline."
            ),
            ExprNode::Nary(_, _) => panic!("Nary not supported in JIT arena compilation"),
            // The lane fold, executed by lanes: its body is the store, and
            // the store is this def, with the fold's trip count as its width.
            ExprNode::Reduce { fold, body } if matches!(arena.node(body), ExprNode::Write { lane, .. } if lane == fold.binder()) =>
            {
                let ExprNode::Write {
                    row, col, value, ..
                } = arena.node(body)
                else {
                    unreachable!("matched a Write above")
                };
                ScheduledOp::Write {
                    row,
                    col,
                    lane: fold.binder(),
                    lanes: fold.len(),
                    value: map_child(value),
                }
            }
            // A surviving fold: `body` was already walked above (it is an
            // ordinary child, scheduled before its parent by the arena's own
            // topological order), so `map_child(body)` is that per-iteration
            // value's `ValueId` in *this* numbering. `scopes::extract_folds` reads
            // it back out into the fold's own `ScopeFold`; nothing after
            // that resolves it as an operand (see `ScheduledOp::Reduce`).
            ExprNode::Reduce { fold, body } => ScheduledOp::Reduce(fold, map_child(body)),
            ExprNode::Write { .. } => unreachable!("a Write node is skipped above"),
        };
        // Numbered after the op is built: a `Uniform` may have pushed its
        // block's `Context` def just above, and ids follow schedule order.
        let vid = ValueId(next_id);
        next_id += 1;
        id_map[idx] = vid;
        schedule.push(Def {
            value: vid,
            op: sched_op,
        });
    }
    // A `Write` whose lane fold nothing reached is a lane fold that is not
    // where `pack` put it — under a column fold — and the schedule would
    // be silently store-free.
    assert!(
        lane.is_none()
            || schedule
                .iter()
                .any(|d| matches!(d.op, ScheduledOp::Write { .. })),
        "arena_to_schedule: a Write is reachable but no lane fold names it"
    );
    schedule
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::arena::ExprArena;

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
        let _ = arena_to_schedule(&a, root, RAW_ORIGIN);
    }

    /// A coordinate `Var` is the survivor the collapse ABI added: the
    /// lattice's folds substitute `X` and `Y` away, so one reaching the
    /// scheduler is an arena that never went through `passes::lattice`.
    #[test]
    #[should_panic(expected = "is a coordinate")]
    fn a_surviving_coordinate_fails_loudly() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let two = a.push_const(2.0);
        let root = a.push_binary(OpKind::Mul, x, two);
        let _ = arena_to_schedule(&a, root, RAW_ORIGIN);
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
        let _ = arena_to_schedule(arena, root, RAW_ORIGIN);
    }

    /// A one-row buffer of `width` samples, declared in `a`.
    fn table(a: &mut ExprArena, width: u32) -> pixelflow_ir::arena::BufferId {
        a.declare_buffer(pixelflow_ir::arena::BufferDecl {
            id: pixelflow_ir::arena::BufferIdentity::mint(),
            width,
            height: 1,
        })
    }

    fn count(schedule: &[Def], pred: fn(&ScheduledOp) -> bool) -> usize {
        schedule.iter().filter(|d| pred(&d.op)).count()
    }

    /// A schedule with no lane fold — an arena `collapse` never
    /// wrapped, which the scheduler still accepts — knows nothing to be
    /// lane-uniform, so every read stays a gather, a constant address
    /// included.
    #[test]
    fn without_a_lane_fold_every_read_is_a_gather() {
        let mut a = ExprArena::new();
        let buf = table(&mut a, 8);
        let idx = a.push_const(3.0);
        let leaf = a.push_buffer(buf);
        let root = a.push_binary(OpKind::RawGather, leaf, idx);
        let schedule = arena_to_schedule(&a, root, RAW_ORIGIN);
        assert_eq!(
            count(&schedule, |op| matches!(op, ScheduledOp::Gather(..))),
            1
        );
        assert_eq!(
            count(&schedule, |op| matches!(op, ScheduledOp::Broadcast(..))),
            0
        );
    }
}
