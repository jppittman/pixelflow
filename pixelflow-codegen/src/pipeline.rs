//! The driver: from an arena and a lattice's shape to machine code.
//!
//! One function does what the stages in between are for. The arena is
//! legalized into the lattice's folds (`pixelflow_ir::passes::legalize`),
//! lowered to a flat schedule ([`crate::program::arena_to_schedule`]), split into its nest
//! ([`ScopedSchedule::from_schedule`]), and handed to the emitter, which
//! allocates and emits it ([`compile_native`]). Nothing here is the emitter's:
//! `emit/` is handed a finished program and names none of the stages above it.

use crate::emit::{CompileResult, compile_native};
use crate::error::CompileError;
use crate::program::ScopedSchedule;
use crate::program::arena_to_schedule;
use pixelflow_ir::LatticeShape;
use pixelflow_ir::arena::{UniformDecl, UniformId, UniformIdentity};
use pixelflow_ir::passes::lattice::{Collapse, Domain};

/// A lane is one `f32`.
const BYTES_PER_LANE: u32 = 4;

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
fn origin_slots(arena: &pixelflow_ir::arena::ExprArena) -> [UniformId; 2] {
    origin().map(|decl| {
        let slot = arena
            .uniforms()
            .iter()
            .position(|d| d.id == decl.id)
            .unwrap_or_else(|| panic!("a legalized arena declares the origin; this one does not"));
        UniformId(slot as u64)
    })
}

/// Compile an [`ExprArena`](pixelflow_ir::arena::ExprArena) DAG into a
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
/// # Panics
///
/// Panics if the arena names a retired coordinate axis (`Var(2)`/`Var(3)`,
/// the old Z and W). This is the boundary the check belongs on, because it
/// is the *only* one every route to machine code passes through — the
/// shape-keyed cache is one caller, and the benchmark harnesses, the corpus
/// tools and several tests come straight here. `collapse` substitutes only
/// `X` and `Y`, so a retired axis would survive into the schedule as a `Var`
/// no fold binds, and the allocator's refusal there names a binder, not an
/// axis; this one names the axis.
pub fn compile(
    arena: &pixelflow_ir::arena::ExprArena,
    root: pixelflow_ir::arena::ExprId,
    shape: LatticeShape,
) -> Result<CompileResult, CompileError> {
    assert!(
        arena.retired_axis(root).is_none(),
        "emit::compile: the arena names Var({:?}), a coordinate axis a \
         lattice no longer has; a per-call scalar is a Uniform",
        arena.retired_axis(root)
    );
    let lanes = (crate::jit_vector_bytes() as u32) / BYTES_PER_LANE;
    let schedule = lower(arena, root, shape, lanes)?;
    compile_native(ScopedSchedule::from_schedule(schedule))
}

/// `passes::legalize` at `shape` for a target of `lanes` lanes, then
/// `arena_to_schedule`: everything [`compile`] runs before the emitter is
/// handed a schedule.
fn lower(
    arena: &pixelflow_ir::arena::ExprArena,
    root: pixelflow_ir::arena::ExprId,
    shape: LatticeShape,
    lanes: u32,
) -> Result<Vec<crate::program::Def>, CompileError> {
    let collapse = Collapse {
        domain: Domain {
            shape,
            origin: origin(),
        },
        lanes,
    };
    let (arena, root) =
        pixelflow_ir::passes::legalize(arena, root, &collapse).map_err(CompileError::Legalize)?;
    let origin_ids = origin_slots(&arena);
    Ok(arena_to_schedule(&arena, root, origin_ids))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::LatticeShape;

    /// A lane is one `f32`.
    pub(crate) const BYTES_PER_LANE: u32 = super::BYTES_PER_LANE;

    /// `passes::legalize` at `shape` for a target of `lanes` lanes, then
    /// `arena_to_schedule`: everything a compile runs before the emitter is
    /// handed a schedule, at a width a test chooses.
    pub(crate) fn schedule_for(
        a: &pixelflow_ir::arena::ExprArena,
        root: pixelflow_ir::arena::ExprId,
        shape: LatticeShape,
        lanes: u32,
    ) -> Vec<crate::program::Def> {
        super::lower(a, root, shape, lanes).expect("legalize")
    }
}
