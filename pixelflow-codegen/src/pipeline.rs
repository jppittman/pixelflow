//! The driver: from an arena and a lattice's shape to machine code.
//!
//! The stages are [`Pass`] values, run as one chain: the arena is legalized
//! into the lattice's folds ([`Legalize`], `pixelflow_ir::passes::legalize`),
//! lowered to a flat schedule ([`Lower`]), split into its nest ([`Nest`]), its
//! arms' ownership read off backward ([`Own`]) and their blocks laid out
//! ([`Layout`]); then the emitter selects, allocates and assembles it
//! ([`emit::native`]). Nothing here is the emitter's: `emit/` is handed the
//! stages above it as a value and names none of them.

use crate::emit::{self, CompileResult};
use crate::error::CompileError;
use crate::pass::Pass;
use crate::program::{Def, Layout, Nest, Own, arena_to_schedule};
use core::marker::PhantomData;
use pixelflow_ir::LatticeShape;
use pixelflow_ir::arena::{ExprArena, ExprId, UniformDecl, UniformId, UniformIdentity};
use pixelflow_ir::passes::lattice::{Collapse, Domain};

/// The two per-call scalars every collapse reads: where the lattice's sample
/// `(0, 0)` lies, `x0` then `y0`.
///
/// Declared as uniforms by `passes::lattice::collapse`, so the arena names
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
/// the two `passes::lattice::collapse` declared, which is why they are always
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

/// `passes::legalize` for the collapse of a lattice of `shape` on a target of
/// `lanes` lanes: the kernel comes back wrapped in the lattice's folds.
struct Legalize<'a> {
    collapse: Collapse,
    arena: PhantomData<&'a ExprArena>,
}

impl Legalize<'_> {
    fn new(shape: LatticeShape, lanes: u32) -> Self {
        let domain = Domain {
            shape,
            origin: origin(),
        };
        Self {
            collapse: Collapse { domain, lanes },
            arena: PhantomData,
        }
    }
}

impl<'a> Pass for Legalize<'a> {
    type Input = (&'a ExprArena, ExprId);
    type Output = (ExprArena, ExprId);

    fn run(self, (arena, root): Self::Input) -> Result<Self::Output, CompileError> {
        pixelflow_ir::passes::legalize(arena, root, &self.collapse).map_err(CompileError::Legalize)
    }
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

/// Compile an [`ExprArena`] DAG into a
/// **collapse** kernel for a lattice of `shape`: the kernel is wrapped in the
/// lattice's folds by
/// [`pixelflow_ir::passes::legalize`], and every fold is emitted as a loop
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
/// compiled inside its guard (`passes::legalize`), computing its fold or NaN
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
        Legalize::new(shape, lanes)
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
        let schedule = Legalize::new(shape, lanes).then(Lower);
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
