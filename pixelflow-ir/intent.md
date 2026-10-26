# pixelflow-ir

**Is.** The language's terms. A kernel is a value of arrow type held as a
rooted term in one hash-consed arena (DENOTATION §1); this crate is that
arena, the `OpKind` vocabulary, the fold, the content key that names a
kernel, and the passes that are endomorphisms on (arena, root). `no_std`
except for the process-global store a `Ref` names into.

## Owns
- The representation of a term: `ExprArena` over `Dag`, interning, the
  dense `ExprId`, the identity tables, and what reachability means. Nothing
  else may depend on how a node is stored.
- The meaning of a fold (`Fold`: monoid, binder, stride; `Bound`: a literal or a
  child; NaN unless each bound is a count in `0..=2²⁴`), of a constant (`Lit`: a
  real by its bits or a mask by its truth, never a lane whose bits mean one or
  the other by convention), of a reinterpretation (`Bits`/`Float`: the same bits
  read as the other sort, compiling to nothing), of the bitwise domain
  (`is_bitwise_domain`), and of a name (`KernelKey`, `Ref`).
- Composition: `Kernel::at` as substitution into `Var` leaves, `by_ref` and
  the linker. Composition substitutes and links; it never evaluates or
  optimizes.
- The exact constant domain the folder uses (`dyadic`), so folding cannot
  contradict the algebraic rewrites.

## Never
- No consumer knows the arena's layout. A walker asks `children` and
  `ExprNode`, never a variant's raw offsets or arity; `Nary`'s offsets are
  not published, and a `push_*` is never assumed to return a fresh id.
- No evaluation, no interpreter, no per-batch entry, no vector, no width, no
  memory read: there is no `eval`, `Gather` is not a language op, and a
  table is a discrete kernel evaluated by the backend.
- No optimization of an open term: a `Dwrt`, a `Ref` and a fold survive
  construction unchanged, so the chain rule holds under a warp and a fold's
  shape is extraction's to choose.
- No schedule: the lattice's traversal is the backend's. `passes::lattice`
  (`collapse`, `pack`), `Write` and `Seq` are the CPU schedule in the wrong
  crate and move out (DENOTATION §7; the schedule note §1).
- No `std` outside `store`.

## Contract
`src/lib.rs` re-exports the surface: `Kernel`, `Uniform`, `Bits`, `OpKind`,
`Fold`/`FoldHeader`/`Bound`/`KnownRange`/`Monoid`/`Binder`, `KernelKey`, `KernelStore`, `Variance`/`LatticeShape`,
`ExprArena`/`ExprId`/`ExprNode`, `Dag`/`Node`/`Rooted`, `Ir`/`Shape`/`Children`,
`Optimize`/`Identity`/`Then`, `passes`, `dyadic`, `library`. Only compiler
crates depend on it (`pixelflow-codegen`, `pixelflow-search`,
`pixelflow-compiler`, `pixelflow-pipeline`); `pixelflow-core` re-exports
`Kernel` to everyone else.

## Enforced by
- `no_std` build in CI ("pixelflow-ir no_std + pixelflow-search
  std-feature-off builds").
- `pixelflow-compiler/tests/derivative_under_warp.rs` (a `Dwrt` survives
  composition); `pixelflow-search/tests/ir_insert.rs` (names and folds reach
  the e-graph intact).
- desloppify `invariant-in-comment` (`overloaded_type`: one lane carrying
  real, integer and mask), `mask-is-not-a-number`, `control-plane-64-bit`.
- The `ExprNode` size assertion is a tripwire against an accident, not a
  width to design against (CLAUDE.md).
- `pixelflow-ir/src/lit.rs`'s tests, and `pixelflow-search/tests/fold_exactness.rs`
  (`a_mask_and_the_nan_sharing_its_lane_are_two_classes`): a true mask and
  the NaN that shares its lane are two literals, two arena nodes and two
  e-classes.

## Behind
- `ExprId(pub u32)` is public and forgeable, and `ExprArena`/`ExprNode` are
  re-exported, so the arena is visible to consumers; `pixelflow-graphics`
  uses `pixelflow_ir` directly in `fonts/loop_blinn.rs` (`ExprArena`) and
  `render/packed.rs` (exprarena-on-dag Stage D, not started; D16).
- A constant is typed (`Lit`), but a node is not: a lane is a real or an integer
  by convention, a computed mask a lane like any other. Node sorts and `Lit::Int`
  are ruled (kernels-evaluate-kernels §2) and wait on the arena-is-typed T3.
- `Var(u8)` means an axis or a binder by range; `Binder` a `u8`, against the
  64-bit control-plane rule.
- `Buffer`, `BoundManifold`'s link tables and `passes::lattice` are in the
  tree and not concepts (DENOTATION §11).

## Decided in
`docs/DENOTATION.md` §1–§3, `docs/language/2026-09-25-the-language-is-kernel.md`,
`docs/archive/plans/2026-09-09-exprarena-on-dag.md`, `docs/archive/plans/2026-09-09-a-fold-is-a-node.md`,
`docs/language/2026-09-09-composition-is-linking.md`, `docs/language/2026-10-15-kernels-evaluate-kernels.md` §2–§3.
