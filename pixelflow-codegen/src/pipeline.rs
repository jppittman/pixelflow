//! The driver: from an arena and a lattice's shape to machine code.
//!
//! The stages are [`Pass`] values, run as one chain: the arena is legalized
//! around the lattice's folds ([`Front`], [`Collapse`], [`Pack`], [`Back`];
//! the traversal is this backend's, the halves are `pixelflow_ir::passes`'),
//! lowered to a flat schedule ([`Lower`]), split into its nest ([`Nest`]), its
//! arms' ownership read off backward ([`Own`]) and their blocks laid out
//! ([`Layout`]); then the emitter selects, allocates and assembles it
//! ([`emit::native`]). Nothing here is the emitter's: `emit/` is handed the
//! stages above it as a value and names none of them.

use crate::emit::{self, CompileResult};
use crate::error::CompileError;
use crate::pass::Pass;
use crate::program::{Collapse, Def, Domain, Layout, Nest, Own, Pack, arena_to_schedule};
use core::marker::PhantomData;
use pixelflow_ir::LatticeShape;
use pixelflow_ir::arena::{ExprArena, ExprId, UniformDecl, UniformId, UniformIdentity};

/// The two per-call scalars every collapse reads: where the lattice's sample
/// `(0, 0)` lies, `x0` then `y0`.
///
/// Declared as uniforms by `program::lattice::collapse`, so the arena names
/// them the way it names any per-call scalar and the emitter loads them the
/// way it loads any uniform — once per call, broadcast. What is particular
/// to them is *where*: not in the link's block, whose layout is the
/// caller's, but in a block of their own, the context entry after the
/// link's (see `KernelFn`). One identity
/// per axis for the whole process, minted once, so every arena declares the
/// same two instances and [`origin_slots`] can find them by identity
/// afterwards.
fn origin() -> [UniformDecl; 2] {
    static ORIGIN: std::sync::OnceLock<[UniformDecl; 2]> = std::sync::OnceLock::new();
    *ORIGIN.get_or_init(|| {
        [0.0, 0.0].map(|default| UniformDecl {
            id: UniformIdentity::mint(),
            default,
        })
    })
}

/// The uniform slots [`origin`]'s two instances hold in a legalized arena —
/// the two `program::lattice::collapse` declared, which is why they are always
/// present.
fn origin_slots(arena: &ExprArena) -> [UniformId; 2] {
    origin().map(|decl| {
        let slot = arena
            .uniforms()
            .iter()
            .position(|d| d.id == decl.id)
            .unwrap_or_else(|| panic!("a legalized arena declares the origin; this one does not"));
        UniformId(slot as u64)
    })
}

/// `passes::legalize_front`: names and derivatives, before the lattice.
struct Front<'a>(PhantomData<&'a ExprArena>);

impl<'a> Pass for Front<'a> {
    type Input = (&'a ExprArena, ExprId);
    type Output = (ExprArena, ExprId);

    fn run(self, (arena, root): Self::Input) -> Result<Self::Output, CompileError> {
        pixelflow_ir::passes::legalize_front(arena, root).map_err(CompileError::Legalize)
    }
}

/// `passes::legalize_back`: reads, transcendentals and the value-bound guard,
/// after the lattice.
struct Back;

impl Pass for Back {
    type Input = (ExprArena, ExprId);
    type Output = (ExprArena, ExprId);

    fn run(self, (arena, root): Self::Input) -> Result<Self::Output, CompileError> {
        Ok(pixelflow_ir::passes::legalize_back(&arena, root))
    }
}

/// The CPU's legalization of a collapse of a lattice of `shape` on a target of
/// `lanes` lanes: the kernel comes back wrapped in the lattice's folds, every
/// op a backend can emit. The lattice's traversal sits between the IR's two
/// halves of legalization (`pixelflow_ir::passes`'s module doc for why).
fn legalization<'a>(
    shape: LatticeShape,
    lanes: u32,
) -> impl Pass<Input = (&'a ExprArena, ExprId), Output = (ExprArena, ExprId)> {
    let domain = Domain {
        shape,
        origin: origin(),
    };
    Front(PhantomData)
        .then(Collapse { domain })
        .then(Pack { lanes })
        .then(Back)
}

/// `arena_to_schedule`: a legalized arena to the flat schedule the later
/// stages read.
struct Lower;

impl Pass for Lower {
    type Input = (ExprArena, ExprId);
    type Output = Vec<Def>;

    fn run(self, (arena, root): Self::Input) -> Result<Self::Output, CompileError> {
        Ok(arena_to_schedule(&arena, root, origin_slots(&arena)))
    }
}

/// The kernel at `root` as the CPU compiles it for a lattice of `shape`, one
/// step before scheduling: wrapped in the lattice's folds at this host's lane
/// width, every op legal. For tools that read the arena a compile would
/// lower (telemetry dumps, cost examples) without emitting it.
///
/// # Errors
///
/// [`CompileError::Legalize`] for an expression with no derivative rule.
pub fn legalize(
    arena: &ExprArena,
    root: ExprId,
    shape: LatticeShape,
) -> Result<(ExprArena, ExprId), CompileError> {
    let lanes = (crate::jit_vector_bytes() / core::mem::size_of::<f32>()) as u32;
    legalization(shape, lanes).run((arena, root))
}

/// Compile an [`ExprArena`] DAG into a
/// **collapse** kernel for a lattice of `shape`: the kernel is wrapped in the
/// lattice's folds by
/// `legalization`, and every fold is emitted as a loop
/// inside the code — one call fills the whole extent with no per-row or
/// per-batch Rust↔JIT boundary. Matches the
/// `KernelFn` ABI `(ctx, out, pitch)`.
///
/// The context is one base pointer per declared buffer, in the arena's slot
/// order, followed by the uniform block's base pointer (`f32` values in the
/// arena's uniform-slot order, read once per call) and then the origin
/// block's: `x0`, `y0`.
///
/// A fold's bound may be a value rather than a literal: the fold is then
/// compiled inside its guard (`passes::legalize_back`), computing its fold or NaN
/// on every tier. A bound is read as a number: a select between two masks is a
/// mask in value but no emitter holds it as one, so its bits are read as a
/// float, on every tier alike, which the guard turns into NaN or a count.
///
/// # Panics
///
/// Panics if the arena names a retired coordinate axis (`Var(2)`/`Var(3)`,
/// the old Z and W). This is the boundary the check belongs on, because it
/// is the *only* one every route to machine code passes through — the
/// shape-keyed cache is one caller, and the benchmark harnesses, the corpus
/// tools and several tests come straight here. `collapse` substitutes only
/// `X` and `Y`, so a retired axis would survive into the schedule as a `Var`
/// no fold binds, and the allocator's refusal there names a binder, not an
/// axis; this one names the axis. Also panics on a zero extent, which has
/// no sample to compute. A fold whose bound the emitter holds as a mask (a
/// comparison, or a bitwise and/or of two such: `ScheduledOp::is_mask`) is
/// refused too, by name (`BoundSort`), at either end and on every tier alike: a
/// mask names no count.
pub fn compile(
    arena: &ExprArena,
    root: ExprId,
    shape: LatticeShape,
) -> Result<CompileResult, CompileError> {
    assert!(
        arena.retired_axis(root).is_none(),
        "emit::compile: the arena names Var({:?}), a coordinate axis a \
         lattice no longer has; a per-call scalar is a Uniform",
        arena.retired_axis(root)
    );
    assert!(
        shape.extent().iter().all(|&e| e > 0),
        "emit::compile: degenerate extent {:?}",
        shape.extent()
    );
    let front = |lanes| {
        legalization(shape, lanes)
            .then(Lower)
            .then(Nest)
            .then(Own)
            .then(Layout)
    };
    emit::native(front, (arena, root))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::pass::Pass;

    /// The stages that make a schedule, at a width a test chooses: everything
    /// a compile runs before scoping.
    pub(crate) fn schedule_for(
        a: &ExprArena,
        root: ExprId,
        shape: LatticeShape,
        lanes: u32,
    ) -> Vec<Def> {
        let schedule = legalization(shape, lanes).then(Lower);
        schedule.run((a, root)).expect("legalize")
    }

    /// Where the stages put a fold whose bound is a count: a lattice of two
    /// rows of eight at four lanes, so the column fold is one fold and
    /// nothing is left over to strip-mine (a width that is not a multiple of
    /// the lanes makes a main fold and a remainder that share one store).
    const TWO_BATCHES: LatticeShape = LatticeShape::new([8, 2]);
    const FOUR_LANES: u32 = 4;

    /// The scoped schedule of `kernel` at `TWO_BATCHES` and `FOUR_LANES`.
    fn nest_of(kernel: &pixelflow_ir::Kernel) -> crate::program::ScopedSchedule<()> {
        use crate::program::Nest;

        let (a, root) = kernel.parts();
        let schedule = schedule_for(a, root, TWO_BATCHES, FOUR_LANES);
        Nest.run(schedule).expect("the schedule nests")
    }

    /// The `SUM` fold the kernel wrote, as the scope it opens and the scope
    /// it sits in: not one of the lattice's, which are `SEQ`.
    fn the_written_fold(
        scoped: &crate::program::ScopedSchedule<()>,
    ) -> &crate::program::ScopeFold<()> {
        use crate::program::{ScheduledOp, Scope};
        use pixelflow_ir::fold::Monoid;

        let sits_in = |fold: &crate::program::ScopeFold<()>| match fold.parent {
            Scope::Body => &scoped.body.schedule,
            Scope::Fold(j) => &scoped.folds[j].schedule,
        };
        let mut written = scoped.folds.iter().filter(|fold| {
            sits_in(fold).iter().any(|def| {
                def.value == fold.reduce
                    && matches!(def.op, ScheduledOp::Reduce { fold, .. } if fold.monoid() == Monoid::SUM)
            })
        });
        let fold = written.next().expect("the kernel's own fold survives");
        assert!(written.next().is_none(), "the kernel wrote one fold");
        fold
    }

    /// How many `Compare` and `If` defs `schedule` holds, in that order.
    fn count_guards(schedule: &[Def]) -> (usize, usize) {
        use crate::program::ScheduledOp;

        let compares = schedule
            .iter()
            .filter(|def| matches!(def.op, ScheduledOp::Compare(..)))
            .count();
        let ifs = schedule
            .iter()
            .filter(|def| matches!(def.op, ScheduledOp::If(..)))
            .count();
        (compares, ifs)
    }

    /// `X · Σ_{k<n} (X + k)`: the body varies with the column, the bound
    /// with nothing. The guard's masks vary with nothing either, so they are
    /// computed once per call, in the body, whichever scope the fold is in;
    /// the fold and its two `If`s are the column fold's, which holds the
    /// store, and read the masks from outside.
    #[test]
    fn a_guard_on_a_uniform_count_is_computed_once_per_call() {
        use crate::program::Scope;
        use pixelflow_ir::fold::Monoid;
        use pixelflow_ir::{Kernel, Uniform};

        let n = Uniform::new(7.0).kernel();
        let sum = Kernel::over_values(Monoid::SUM, &Kernel::constant(0.0), &n, |k| {
            Kernel::x().add(k)
        });
        let scoped = nest_of(&Kernel::x().mul(&sum));

        // Three tests of the domain (`0 <= n`, `n <= 2^24`, `floor(n) == n`) and
        // the range's (`0 < n`).
        let (compares, ifs) = count_guards(&scoped.body.schedule);
        assert_eq!((compares, ifs), (4, 0), "the masks are the call's");
        for fold in &scoped.folds {
            assert_eq!(count_guards(&fold.schedule).0, 0, "no mask is the fold's");
        }

        let Scope::Fold(column) = the_written_fold(&scoped).parent else {
            panic!("the fold is inside the lattice's, not in the body")
        };
        let stores = scoped.folds[column]
            .schedule
            .iter()
            .any(|def| matches!(def.op, crate::program::ScheduledOp::Write { .. }));
        assert!(stores, "the fold's parent is the column fold, which stores");
        assert_eq!(
            count_guards(&scoped.folds[column].schedule).1,
            2,
            "both guard `If`s sit in the column fold, beside the fold they own"
        );
    }

    /// `X + Σ_{k<n} k`: nothing the fold reads varies with the lattice, so
    /// the fold, its guard and its masks are all the call's. The carve sees
    /// the outer `If`, not the `Reduce`, as the def a deeper fold reads: the
    /// fold is its guard's alone and is carved at the body level, where the
    /// arm blocks hold it.
    #[test]
    fn a_fold_whose_body_reads_no_coordinate_is_carved_in_the_body() {
        use crate::program::{ScheduledOp, Scope};
        use pixelflow_ir::fold::Monoid;
        use pixelflow_ir::{Kernel, Uniform};

        let n = Uniform::new(7.0).kernel();
        let sum = Kernel::over_values(Monoid::SUM, &Kernel::constant(0.0), &n, |k| k.clone());
        let scoped = nest_of(&Kernel::x().add(&sum));

        let written = the_written_fold(&scoped);
        assert_eq!(written.parent, Scope::Body);
        let (_, ifs) = count_guards(&scoped.body.schedule);
        assert_eq!(ifs, 2, "the guard's two `If`s are the call's");

        let guarded = scoped
            .body
            .schedule
            .iter()
            .find_map(|def| match def.op {
                ScheduledOp::If(_, ran, _)
                    if matches!(
                        scoped.body.schedule.iter().find(|d| d.value == ran),
                        Some(d) if matches!(d.op, ScheduledOp::If(..))
                    ) =>
                {
                    Some(def.value)
                }
                _ => None,
            })
            .expect("the outer `If` of the guard");
        let lattice = scoped.folds.iter().filter(|f| f.reduce != written.reduce);
        let reads: Vec<_> = lattice
            .flat_map(|fold| fold.outer.iter().copied())
            .collect();
        assert!(
            reads.contains(&guarded),
            "the lattice's folds read the outer `If`"
        );
        assert!(
            !reads.contains(&written.reduce),
            "and never the `Reduce`: it is read by its guard alone"
        );
    }

    use pixelflow_ir::arena::ExprNode;
    use pixelflow_ir::fold::{Binder, Bound, Fold, KNOWN_RANGE, Monoid};
    use pixelflow_ir::{Lit, OpKind};

    /// Every `Reduce` reachable from `root`.
    fn reachable_reduces(arena: &ExprArena, root: ExprId) -> usize {
        let mut seen = vec![false; arena.len()];
        let mut stack = vec![root];
        let mut count = 0;
        while let Some(id) = stack.pop() {
            if std::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            if matches!(arena.node(id), ExprNode::Reduce { .. }) {
                count += 1;
            }
            stack.extend(arena.children(id));
        }
        count
    }

    /// A `Reduce` whose body reads another `Reduce`'s result reaches the
    /// backend as written: two folds, one inside the other, both inside the
    /// lattice's three. This used to be the one shape `legalize` still
    /// unrolled (`expand_nested_reduce`, deleted), because codegen carved
    /// fold bodies out one level at a time; it carves a fold inside a fold
    /// now, and the pass went with the restriction it existed for.
    ///
    /// `inner = sum_{i<3}(X+i)`; `outer = sum_{j<2}(inner+j)`.
    #[test]
    fn legalization_leaves_a_nested_reduce_standing_inside_the_lattices_folds() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let inner_binder = Binder::from_slot(0).expect("slot 0 exists");
        let i = a.push_var(inner_binder.var());
        let inner_body = a.push_binary(OpKind::Add, x, i);
        let inner = a.push_reduce(
            Fold::new(Monoid::SUM, inner_binder),
            Bound::Lit(0),
            Bound::Lit(3),
            inner_body,
        );

        let outer_binder = Binder::from_slot(1).expect("slot 1 exists");
        let j = a.push_var(outer_binder.var());
        let outer_body = a.push_binary(OpKind::Add, inner, j);
        let root = a.push_reduce(
            Fold::new(Monoid::SUM, outer_binder),
            Bound::Lit(0),
            Bound::Lit(2),
            outer_body,
        );

        let (legalized, new_root) = legalization(LatticeShape::new([4, 1]), FOUR_LANES)
            .run((&a, root))
            .expect("legalize");
        // The lattice's row, column and lane folds around the kernel's two.
        assert_eq!(
            reachable_reduces(&legalized, new_root),
            5,
            "both folds must survive legalization, inside the lattice's three"
        );
        let ExprNode::Reduce { fold, .. } = legalized.node(new_root) else {
            panic!("root must be the lattice's row fold");
        };
        assert_eq!(fold.monoid(), Monoid::SEQ);
        assert_eq!(
            legalized.known_range(new_root).expect(KNOWN_RANGE).range(),
            0..1
        );
        // A fold's range is read through the arena, which panics on any other
        // node: keep the shape check first.
        let ranges_over = |id: ExprId, range: std::ops::Range<u64>| {
            matches!(legalized.node(id), ExprNode::Reduce { .. })
                && legalized.known_range(id).expect(KNOWN_RANGE).range() == range
        };
        let outer = (0..legalized.len())
            .map(|k| ExprId(k as u32))
            .find(|id| ranges_over(*id, 0..2))
            .expect("the outer kernel fold survives");
        let ExprNode::Reduce { body, .. } = legalized.node(outer) else {
            unreachable!()
        };
        let ExprNode::Binary(OpKind::Add, lhs, _) = legalized.node(body) else {
            panic!("the outer body must still be `inner + j`");
        };
        assert!(
            ranges_over(lhs, 0..3),
            "the inner Reduce must be the outer body's own operand, not unrolled into it"
        );
    }

    /// `legalize` of `X + Σ_{k<u} k` at `[4, 2]` with four lanes: the kernel
    /// comes back wrapped in the lattice's three folds, whose ranges are
    /// literal, and the `Write`'s value reads the sum only through its guard.
    #[test]
    fn legalization_guards_a_value_bounded_fold_inside_the_lattice() {
        let mut a = ExprArena::new();
        let slot = a.declare_uniform(pixelflow_ir::Uniform::new(8.0).decl());
        let u = a.push_uniform(slot);
        let binder = Binder::from_slot(0).expect("slot 0 exists");
        let k = a.push_var(binder.var());
        let sum = a.push_reduce(
            Fold::new(Monoid::SUM, binder),
            Bound::Lit(0),
            Bound::Value(u),
            k,
        );
        let x = a.push_var(0);
        let root = a.push_binary(OpKind::Add, x, sum);

        let (legalized, new_root) = legalization(LatticeShape::new([4, 2]), FOUR_LANES)
            .run((&a, root))
            .expect("legalize");

        // An arena keeps the nodes a rebuild replaced, so only what the root
        // reaches is the legalized kernel.
        let mut seen = vec![false; legalized.len()];
        let mut stack = vec![new_root];
        let mut reachable = Vec::new();
        while let Some(id) = stack.pop() {
            if !std::mem::replace(&mut seen[id.0 as usize], true) {
                reachable.push(id);
                stack.extend(legalized.children(id));
            }
        }
        let (valued, lattice): (Vec<ExprId>, Vec<ExprId>) = reachable
            .iter()
            .copied()
            .filter(|id| matches!(legalized.node(*id), ExprNode::Reduce { .. }))
            .partition(|id| legalized.known_range(*id).is_none());
        assert_eq!(
            valued.len(),
            1,
            "the kernel's fold is the one whose range is a value"
        );
        assert_eq!(lattice.len(), 3, "the row, column and lane folds");
        for fold in lattice {
            let range = legalized.known_range(fold).expect(KNOWN_RANGE);
            assert!(!range.is_empty());
        }

        let write = reachable
            .iter()
            .find_map(|id| match legalized.node(*id) {
                ExprNode::Write { value, .. } => Some(value),
                _ => None,
            })
            .expect("the lattice's store");
        let ExprNode::Binary(OpKind::Add, _, guarded) = legalized.node(write) else {
            panic!("the store's value is `x + sum`");
        };
        let ExprNode::Ternary(OpKind::If, _, ran, nan) = legalized.node(guarded) else {
            panic!("the sum reaches the store through its guard");
        };
        assert!(matches!(
            legalized.node(nan),
            ExprNode::Const(Lit::Real(v)) if v.is_nan()
        ));
        let ExprNode::Ternary(OpKind::If, _, fold, _) = legalized.node(ran) else {
            panic!("the guard's true arm is the range test");
        };
        assert_eq!(fold, valued[0]);
    }

    /// The backstop for an arena that never became a `Kernel`.
    /// `Kernel::from_parts` refuses an arena naming a retired axis
    /// (`pixelflow-ir`'s `an_arena_naming_a_retired_axis_is_not_a_kernel`),
    /// but `compile` takes a bare arena: the benchmark harnesses and corpus
    /// tools call it directly, and a rewrite rule that instantiated one of
    /// its own metavariables would hand it one past that refusal. So the
    /// arena goes straight in here, and the refusal is this function's own.
    #[test]
    #[should_panic(expected = "emit::compile: the arena names Var")]
    fn an_arena_naming_a_retired_axis_never_reaches_the_emitter() {
        let mut a = pixelflow_ir::arena::ExprArena::new();
        let x = a.push_var(0);
        let z = a.push_var(2);
        let root = a.push_binary(pixelflow_ir::OpKind::Add, x, z);
        let _refused = super::compile(&a, root, LatticeShape::new([4, 4]));
    }
}
