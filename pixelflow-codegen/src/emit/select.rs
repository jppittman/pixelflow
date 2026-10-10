//! Instruction selection: a scoped schedule to a machine function over values.
//!
//! The driver is generic over the machine ([`IsaBackend`]) and names no
//! register, opcode or encoding. It binds every scheduled value to the
//! backend's [`IsaBackend::Lane`] and hands lanes back, never looking inside
//! one. What it owns is the shape: the body, each surviving fold as a loop of
//! blocks, and the lattice's store.

use super::build::{Builder, Pending};
use super::traffic::scope_ix;
use super::{
    Edges, Entry, Function, IsaBackend, LaneOp, Pointer, Store, Target, Test, Value,
    unimplemented_op,
};
use crate::error::CompileError;
use crate::program::{IfArm, IfGuard, ScheduledOp, Scope, ScopedSchedule, ValueId};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::cmp::Reverse;
use pixelflow_ir::fold::{Binder, Fold, Monoid};
use pixelflow_ir::kind::OpKind;

/// What the driver says when a backend must be asked for something it has not
/// been taught yet: the op, named.
const DRIVER: &str = "the selection driver";

/// Select `scoped` as one machine function.
///
/// # Errors
///
/// [`CompileError::BudgetExceeded`] when the backend refuses a constant.
pub(super) fn select<B: IsaBackend>(scoped: &ScopedSchedule) -> Result<Function<B>, CompileError> {
    let (mut b, entry) = Builder::<B>::new();
    B::enter(&mut b);
    let mut selector = Selector {
        scoped,
        b,
        entry,
        frames: Vec::new(),
        binders: Vec::new(),
        loops: Vec::new(),
        scheduled: alloc::vec![0; scoped.folds.len() + 1],
    };
    selector.scope(Scope::Body)?;
    B::ret(&mut selector.b);
    let mut function = selector.b.finish();
    function.scheduled = selector.scheduled;
    Ok(function)
}

/// A guarded arm: skipped when no lane of the mask selects it.
struct Arm {
    arm: IfArm,
    mask: ValueId,
    /// The schedule position its run ends at.
    end: usize,
}

/// The arms of `guards` by the schedule position each begins at, outer arms
/// before the arms nested in them.
fn arms_by_start(guards: &[IfGuard]) -> BTreeMap<usize, Vec<Arm>> {
    let mut starts: BTreeMap<usize, Vec<Arm>> = BTreeMap::new();
    for guard in guards {
        for arm in IfArm::ALL {
            let (start, end) = guard.range(arm);
            if start != end {
                let mask = guard.mask_vid;
                starts
                    .entry(start)
                    .or_default()
                    .push(Arm { arm, mask, end });
            }
        }
    }
    starts
        .values_mut()
        .for_each(|arms| arms.sort_by_key(|arm| Reverse(arm.end)));
    starts
}

/// The loop nest as it is entered: one binding frame per open scope, and the
/// binders and loops the code being selected is inside.
struct Selector<'s, B: IsaBackend> {
    scoped: &'s ScopedSchedule,
    b: Builder<B>,
    entry: Entry,
    /// Values by name, one frame per open scope, innermost last. Sibling folds
    /// share the `ValueId`s of what they carve out, so each binds in a frame
    /// of its own and a lookup searches outward.
    frames: Vec<Frame<B>>,
    /// The binder of each fold the code is inside, innermost last: where a
    /// `Var`, and a store's row and column, find their lane.
    binders: Vec<(Binder, B::Lane)>,
    /// Each open loop's index in the function's loops, and how many times its
    /// body runs per call.
    loops: Vec<(usize, u64)>,
    /// Scheduled ops selected, per scope.
    scheduled: Vec<u64>,
}

/// What one scope has defined. An operand knows by position which class it
/// reads (a `Uniform`'s base is an address, everything else is a lane), so a
/// value is looked up in the table of its class and the class check is that
/// lookup.
struct Frame<B: IsaBackend> {
    lanes: BTreeMap<ValueId, B::Lane>,
    pointers: BTreeMap<ValueId, Value<Pointer>>,
}

impl<B: IsaBackend> Frame<B> {
    fn new() -> Self {
        Self {
            lanes: BTreeMap::new(),
            pointers: BTreeMap::new(),
        }
    }
}

impl<B: IsaBackend> Selector<'_, B> {
    /// Select `scope`'s schedule, and return the lane of its root: `None`
    /// when the root is an effect (a store, a sequence, a fold over the unit
    /// monoid) and not a value.
    fn scope(&mut self, scope: Scope) -> Result<Option<B::Lane>, CompileError> {
        let scoped = self.scoped;
        let (schedule, guards) = match scope {
            Scope::Body => (&scoped.body.schedule, &scoped.body.guards),
            Scope::Fold(j) => (&scoped.folds[j].schedule, &scoped.folds[j].guards),
        };
        self.frames.push(Frame::new());
        let mut starts = arms_by_start(guards);
        // The arms open at this point of the schedule, innermost last, each
        // with the block that follows it.
        let mut open: Vec<(usize, Pending)> = Vec::new();
        for (at, def) in schedule.iter().enumerate() {
            while let Some((_, past)) = open.pop_if(|(end, _)| *end == at) {
                self.close_arm(past);
            }
            for arm in starts.remove(&at).unwrap_or_default() {
                let cond = self.lookup(arm.mask);
                let past = self.open_arm(scope, cond, arm.arm);
                open.push((arm.end, past));
            }
            // A placeholder, a sequence and a binder's alias select nothing;
            // a `Reduce` counts once it is known to open a loop; a constant is
            // defined where it is read, and counted there, as a remat.
            if !matches!(
                def.op,
                ScheduledOp::Const(_)
                    | ScheduledOp::Outer(_)
                    | ScheduledOp::Seq(..)
                    | ScheduledOp::Var(_)
                    | ScheduledOp::Reduce(..)
            ) {
                self.scheduled[scope_ix(scope)] += 1;
            }
            let value = match &def.op {
                // Computed by an enclosing scope, which bound it there.
                ScheduledOp::Outer(_) => continue,
                ScheduledOp::Seq(..) => continue,
                ScheduledOp::Write {
                    row,
                    col,
                    lanes,
                    value,
                    ..
                } => {
                    let store = Store {
                        out: self.entry.out,
                        pitch: self.entry.pitch,
                        row: self.binder(*row),
                        col: self.binder(*col),
                        value: self.lookup(*value),
                        lanes: *lanes,
                    };
                    B::store(&mut self.b, store);
                    continue;
                }
                // A fold this scope opens is a loop here; any other is the
                // result of one an enclosing scope ran, bound there.
                ScheduledOp::Reduce(fold, _) => {
                    let Some(j) = scoped
                        .folds
                        .iter()
                        .position(|f| f.parent == scope && f.at == at)
                    else {
                        continue;
                    };
                    self.scheduled[scope_ix(scope)] += 1;
                    match self.fold(scope, j, *fold)? {
                        Some(result) => result,
                        None => continue,
                    }
                }
                ScheduledOp::Var(var) => self.binder_of_var(*var),
                ScheduledOp::Const(value) => self.lane(LaneOp::Const(*value))?,
                ScheduledOp::Lanes(_) => self.lane(LaneOp::Lanes)?,
                ScheduledOp::Unary(op, a) => {
                    let a = self.lookup(*a);
                    self.lane(LaneOp::Unary(*op, a))?
                }
                ScheduledOp::Binary(op, a, b) => {
                    let (a, b) = (self.lookup(*a), self.lookup(*b));
                    self.lane(LaneOp::Binary(*op, a, b))?
                }
                ScheduledOp::ShiftImm(op, a, amount) => {
                    let a = self.lookup(*a);
                    self.lane(LaneOp::Shift(*op, a, *amount))?
                }
                ScheduledOp::Ternary(OpKind::MulAdd, a, b, c) => {
                    let (a, b, c) = (self.lookup(*a), self.lookup(*b), self.lookup(*c));
                    self.lane(LaneOp::MulAdd(a, b, c))?
                }
                ScheduledOp::Ternary(OpKind::If, cond, if_true, if_false) => {
                    let guarded = guards
                        .iter()
                        .any(|guard| guard.if_idx == at && guard.has_guarded_arm());
                    let (cond, if_true, if_false) = (
                        self.lookup(*cond),
                        self.lookup(*if_true),
                        self.lookup(*if_false),
                    );
                    match guarded {
                        true => self.guarded_if(scope, cond, (if_true, if_false))?,
                        false => self.lane(LaneOp::Blend {
                            cond,
                            if_true,
                            if_false,
                        })?,
                    }
                }
                ScheduledOp::Ternary(op, ..) => unimplemented_op(DRIVER, op),
                ScheduledOp::Context(slot) => {
                    let ctx = self.entry.ctx;
                    let pointer = B::context(&mut self.b, ctx, u64::from(*slot))?;
                    self.frame().pointers.insert(def.value, pointer);
                    continue;
                }
                ScheduledOp::Uniform(base, element) => {
                    let base = self.pointer(*base);
                    self.lane(LaneOp::Uniform {
                        base,
                        element: *element,
                    })?
                }
                ScheduledOp::Gather(index, base) => {
                    let (base, index) = (self.pointer(*base), self.lookup(*index));
                    self.lane(LaneOp::Gather { base, index })?
                }
                ScheduledOp::Broadcast(index, base) => {
                    let (base, index) = (self.pointer(*base), self.lookup(*index));
                    self.lane(LaneOp::Broadcast { base, index })?
                }
            };
            self.bind(def.value, value);
        }
        assert!(
            open.is_empty() && starts.is_empty(),
            "an arm is open at the end of its scope, or begins past it"
        );
        let root = schedule.last().expect("a scope's schedule is not empty");
        let is_unit = match &root.op {
            ScheduledOp::Write { .. } | ScheduledOp::Seq(..) => true,
            ScheduledOp::Reduce(fold, _) => fold.monoid() == Monoid::SEQ,
            _ => false,
        };
        let result = (!is_unit).then(|| self.lookup(root.value));
        self.frames.pop();
        Ok(result)
    }

    /// The loop `fold` opens at `parent`'s `Reduce`, as blocks:
    ///
    /// ```text
    /// preheader: seeds, then into the head
    /// head(binder, acc): the body ... accumulate, step, test; back to the head or on
    /// exit:
    /// ```
    ///
    /// It returns the accumulator after the last trip, which dominates the
    /// exit (one predecessor), so the exit takes no parameter. A fold over
    /// the unit monoid has no accumulator and returns `None`.
    fn fold(
        &mut self,
        parent: Scope,
        j: usize,
        fold: Fold,
    ) -> Result<Option<B::Lane>, CompileError> {
        assert!(
            !fold.is_empty(),
            "an empty fold is its monoid's identity before it reaches selection"
        );
        let accumulates = fold.monoid() != Monoid::SEQ;
        let range = fold.range();

        let lo = self.constant(range.start as f32)?;
        let seeds: Vec<B::Lane> = if accumulates {
            alloc::vec![lo, self.constant(fold.monoid().identity())?]
        } else {
            alloc::vec![lo]
        };
        let names: Vec<_> = seeds.iter().map(|&seed| B::lane_name(seed)).collect();
        let classes: Vec<_> = names.iter().map(|name| name.class).collect();

        let head = self.b.block(&classes, Scope::Fold(j));
        let (parent_loop, parent_trips) = match self.loops.last() {
            Some(&(index, trips)) => (Some(index), trips),
            None => (None, 1),
        };
        let trips = parent_trips * u64::from(fold.len());
        let index = self.b.open_loop(head.label(), parent_loop, trips);
        let head_label = head.label();
        let params: Vec<B::Lane> = head.params().iter().map(|&p| B::param_lane(p)).collect();
        let into_head = Target {
            label: head_label,
            args: names,
        };
        B::jump(&mut self.b, into_head, head_label);
        self.b.enter(head);

        let binder = params[0];
        self.binders.push((fold.binder(), binder));
        self.loops.push((index, trips));
        let body = self.scope(Scope::Fold(j))?;
        self.loops.pop();
        self.binders.pop();

        let accumulated = match (accumulates, body) {
            (true, Some(body)) => Some(self.binary(fold.combine_op(), params[1], body)?),
            (false, _) => None,
            (true, None) => panic!("a fold over a value monoid has a value to combine"),
        };
        let stride = self.constant(fold.stride() as f32)?;
        let step = self.binary(OpKind::Add, binder, stride)?;
        let end = self.constant(range.end as f32)?;
        let done = self.binary(OpKind::Ge, step, end)?;
        let exit = self.b.block(&[], parent);
        let args = [Some(step), accumulated]
            .into_iter()
            .flatten()
            .map(B::lane_name)
            .collect();
        let taken = Target {
            label: head_label,
            args,
        };
        let next = exit.label();
        // A broadcast `done` is uniform, so "no lane is set" is "not done yet".
        B::branch(
            &mut self.b,
            Test {
                cond: done,
                dead: IfArm::True,
            },
            Edges { taken, next },
        );
        self.b.enter(exit);
        Ok(accumulated)
    }

    /// Open the arm `dead` names: skipped, to the block after it, when no lane
    /// of `cond` selects it. It returns that block, to be entered where the
    /// arm ends.
    fn open_arm(&mut self, scope: Scope, cond: B::Lane, dead: IfArm) -> Pending {
        let (arm, past) = (self.b.block(&[], scope), self.b.block(&[], scope));
        let taken = Target {
            label: past.label(),
            args: Vec::new(),
        };
        let edges = Edges {
            taken,
            next: arm.label(),
        };
        B::branch(&mut self.b, Test { cond, dead }, edges);
        self.b.enter(arm);
        past
    }

    /// End the arm in the open block and enter the block after it.
    fn close_arm(&mut self, past: Pending) {
        let label = past.label();
        let to = Target {
            label,
            args: Vec::new(),
        };
        B::jump(&mut self.b, to, label);
        self.b.enter(past);
    }

    /// An `If` with a guarded arm, as blocks:
    ///
    /// ```text
    /// no lane true:   only_false
    /// no lane false:  only_true
    /// blend:          the lane-varying path
    /// join(result)
    /// ```
    ///
    /// An arm a uniform mask skipped did not run, and its value is read only
    /// on the paths where it did, so each path passes the `If`'s value to the
    /// join.
    fn guarded_if(
        &mut self,
        scope: Scope,
        cond: B::Lane,
        (if_true, if_false): (B::Lane, B::Lane),
    ) -> Result<B::Lane, CompileError> {
        let [only_false, only_true, second, blend] = [(); 4].map(|()| self.b.block(&[], scope));
        let to = |block: &Pending| Target {
            label: block.label(),
            args: Vec::new(),
        };
        let first = Edges {
            taken: to(&only_false),
            next: second.label(),
        };
        let dead = |dead| Test { cond, dead };
        B::branch(&mut self.b, dead(IfArm::True), first);
        self.b.enter(second);
        let second = Edges {
            taken: to(&only_true),
            next: blend.label(),
        };
        B::branch(&mut self.b, dead(IfArm::False), second);
        self.b.enter(blend);
        let blended = self.lane(LaneOp::Blend {
            cond,
            if_true,
            if_false,
        })?;
        let join = self.b.block(&[B::lane_name(blended).class], scope);
        let result = B::param_lane(join.params()[0]);
        let into = |lane| Target {
            label: join.label(),
            args: alloc::vec![B::lane_name(lane)],
        };
        B::jump(&mut self.b, into(blended), only_false.label());
        self.b.enter(only_false);
        B::jump(&mut self.b, into(if_false), only_true.label());
        self.b.enter(only_true);
        B::jump(&mut self.b, into(if_true), join.label());
        self.b.enter(join);
        Ok(result)
    }

    fn lane(&mut self, op: LaneOp<B>) -> Result<B::Lane, CompileError> {
        B::lane(&mut self.b, op)
    }

    fn constant(&mut self, value: f32) -> Result<B::Lane, CompileError> {
        self.lane(LaneOp::Const(value))
    }

    fn binary(&mut self, op: OpKind, a: B::Lane, b: B::Lane) -> Result<B::Lane, CompileError> {
        self.lane(LaneOp::Binary(op, a, b))
    }

    fn frame(&mut self) -> &mut Frame<B> {
        self.frames.last_mut().expect("a scope is open")
    }

    fn bind(&mut self, value: ValueId, lane: B::Lane) {
        let bound = self.frame().lanes.insert(value, lane);
        assert!(bound.is_none(), "{value:?} is defined twice in one scope");
    }

    /// The lane `value` lives in: the innermost scope that bound it.
    fn lookup(&self, value: ValueId) -> B::Lane {
        let found = self
            .frames
            .iter()
            .rev()
            .find_map(|frame| frame.lanes.get(&value));
        *found.unwrap_or_else(|| panic!("{value:?} is read as a lane before any scope defines it"))
    }

    /// The address `value` is: the innermost scope that bound it.
    fn pointer(&self, value: ValueId) -> Value<Pointer> {
        let found = self
            .frames
            .iter()
            .rev()
            .find_map(|frame| frame.pointers.get(&value));
        *found.unwrap_or_else(|| {
            panic!("{value:?} is read as an address before any scope defines it")
        })
    }

    /// The lane of `binder`, in the innermost fold that binds it.
    fn binder(&self, binder: Binder) -> B::Lane {
        let found = self.binders.iter().rev().find(|(b, _)| *b == binder);
        let (_, lane) = found.unwrap_or_else(|| {
            panic!(
                "a store names binder slot {} that no enclosing fold binds",
                binder.slot()
            )
        });
        *lane
    }

    /// The lane of the innermost fold binding `Var(var)`.
    fn binder_of_var(&self, var: u8) -> B::Lane {
        let found = self.binders.iter().rev().find(|(b, _)| b.var() == var);
        let (_, lane) =
            found.unwrap_or_else(|| panic!("Var({var}) names no enclosing fold's binder"));
        *lane
    }
}
