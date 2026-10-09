//! Instruction selection: a scoped schedule to a machine function over values.
//!
//! The driver is generic over the machine ([`IsaBackend`]) and names no
//! register, opcode or encoding. It binds every scheduled value to the
//! backend's [`IsaBackend::Lane`] and hands lanes back, never looking inside
//! one. What it owns is the shape: the body, each surviving fold as a loop of
//! blocks, and the lattice's store.
#![expect(dead_code, reason = "live from B4")]

use super::build::Builder;
use super::{
    Edges, Entry, Function, IsaBackend, LaneOp, Pointer, Store, Target, Test, Value,
    unimplemented_op,
};
use crate::error::CompileError;
use crate::program::{IfArm, ScheduledOp, Scope, ScopedSchedule, ValueId};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
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
    };
    selector.scope(Scope::Body)?;
    B::ret(&mut selector.b);
    Ok(selector.b.finish())
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
        for (at, def) in schedule.iter().enumerate() {
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
                    if guards
                        .iter()
                        .any(|guard| guard.if_idx == at && guard.has_guarded_arm())
                    {
                        unimplemented_op(DRIVER, &def.op);
                    }
                    let (cond, if_true, if_false) = (
                        self.lookup(*cond),
                        self.lookup(*if_true),
                        self.lookup(*if_false),
                    );
                    self.lane(LaneOp::Blend {
                        cond,
                        if_true,
                        if_false,
                    })?
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
                ScheduledOp::Gather(..) | ScheduledOp::Broadcast(..) => {
                    unimplemented_op(DRIVER, &def.op)
                }
            };
            self.bind(def.value, value);
        }
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
