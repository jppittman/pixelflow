# pixelflow-codegen/src/program

**Is.** The schedule of the CPU backend: how `apply(f, s)` factors the domain
(DENOTATION §7), as the front of its typed `Pass` chain. It takes a legalized
arena to a `ScopedSchedule` that evaluates each value at the coarsest scope it
factors through, each scope's arms laid out as the blocks the `If` shape asks
for (§4); that is the input of `select` (§10).

## Owns
- The vocabulary every stage between the arena and the machine shares.
- Which scope computes a value: the outermost one binding every binder in its
  variance. Hoisting is this, the apply rule, for every fold at once (`Nest`).
- Who a value is computed for: the lowest common ancestor of the arm regions
  that read it; a loop reads at its `Reduce` (`Own`).
- Where a block goes: an `If`'s arms that own work, true arm first, go
  immediately before the `If`; values stay in construction order (`Layout`).
- Which nodes define a value: a `Bits`/`Float` reinterpretation defines none
  and takes no `ValueId` — lowering maps it to its operand, so no backend
  meets it and it is zero instructions on every target (`lower.rs`).
- A fold's bounds in the schedule: `ScheduledOp::Reduce` carries the arena
  node's `lo` and `hi`; a value bound is an operand of the fold (`operands`),
  its loop's seed and its latch's end, and selection reads a known range with
  `KnownRange::of`.

## Never
- No register, byte, allocator, ISA tier or driver is named: `program/` does
  not import `crate::emit`, `isa`, `pipeline` or `jit_cache`.
- No emitter module reads behind `mod.rs`: `emit/` names no submodule of `program/`.
- No price: an arm has a block when it owns a value of the scope, however
  little; nothing estimates how often a mask is true.
- No structure recovered from position and no variance recomputed: ownership
  is computed from reads; each `Def` carries the arena's variance.
- No open term: a `Dwrt`, a coordinate `Var` or a `Ref` reaching
  `arena_to_schedule` panics, as does an op outside `UnaryOp`, `BinaryOp`,
  `CompareOp` and `ShiftOp`, a value bound that reads a lattice binder, and a
  value-bounded fold read other than once, as an `If`'s true arm: a bypassed
  pipeline, not a refused kernel. A bound the emitter holds as a mask
  (`ScheduledOp::is_mask`) is refused by name, `BoundSort`, on every tier.

## Contract
`mod.rs` defines what `emit/` reads: `ValueId`, `Def`, `ScheduledOp` (with
`UnaryOp`, `BinaryOp` and its `on_masks` and `MaskOp`, `CompareOp`, `ShiftOp`,
`Legal`, `legal`), `Scope`,
`ScopedSchedule`/`ScopeRegion`/`ScopeFold`, `Flow`/`Item`/`ArmBlock`/`IfArm`,
`operands`, `pointer_operand`, `all_operands`; and the stages `arena_to_schedule`
(`lower.rs`), `Nest` (`scopes.rs`), `Own` (`ownership.rs`), `Layout` (`layout.rs`).

## Enforced by
- Types: `Nest`, `Own`, `Layout` are `Pass`es over `ScopedSchedule<()>`,
  `<Ownership>`, `<Flow>`, so a stage cannot be handed a nest its predecessor
  has not run on; `legal::<T>` makes an emitter's `match` over an op exhaustive.
- `scripts/check-emit-boundary.sh` (CI job "Emit boundary"; rules P and E).
- `ScopedSchedule::assert_closed`, run by `Own` on every scope; desloppify `phases-in-order`.
- Tests: the `*_fails_loudly` tests in `lower.rs` and
  `a_bound_is_refused_exactly_when_the_emitter_holds_it_as_a_mask`; the carve in
  `pipeline.rs`; through the emitter `pixelflow-codegen/tests/shared_arm_value.rs`,
  `tests/collapse_paths.rs` and `tests/value_bounds_jit.rs`.

## Behind
- `collapse` and `pack` still run in `pixelflow-ir/src/passes/lattice.rs`,
  called from `pipeline.rs`'s `Legalize`, so `Nest` receives the lattice
  already folded into the arena; `ScheduledOp::Write`, `Seq`, `Lanes` and the
  lane-fold inlining in `lower.rs` carry that CPU schedule. They move into
  `Nest` (`docs/codegen/2026-10-15-a-schedule-is-a-factorization.md` §1, §4).
- `ScheduledOp::Context(u16)` and `Write { lanes: u32 }` are narrower than the
  control plane's 64 bits with no measurement cited; no plan closes it.

## Decided in
`docs/DENOTATION.md` §4, §7, §10; `docs/archive/plans/2026-10-08-selection-is-a-phase.md`
§0.13, §0.14; `docs/language/2026-10-15-kernels-evaluate-kernels.md` §1, §3;
`docs/codegen/2026-10-15-a-schedule-is-a-factorization.md`.
