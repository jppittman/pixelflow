# pixelflow-compiler

**Is.** The language's front end: a function from a `kernel!` block's tokens to
the code that rebuilds, at load time, the kernels it denotes. An entry is a term
of arrow type with uniform parameters (DENOTATION §2); the optimizer it hands
that term to is an `Optimize` value, never a branch (§8).

## Owns
- The surface: grammar, types (`Ty`: `f32`, `bool`, `usize`, record, kernel)
  and refusals, each citing its plan section (`PLAN`). The IR keeps one lane
  for a number and a mask; the difference lives here.
- Binding times as the host sees them (structural, uniform, kernel-typed), the
  host `fn`, the `Args` record, and `const` evaluation.
- Which `Optimize` each macro hands `expand` (`kernel!` saturates, `kernel_raw!`
  is `Identity`), and which entries no optimizer sees.

## Never
- No construct is defined here. Lowering calls `pixelflow_ir::library`
  (`fract`, `hypot`, `clamp`, the derivative), `ExprArena::close_over`,
  `Fold::admits` and `ExprArena::apply`; a copy is a future divergence.
- No `Dwrt` is resolved at expansion: `macro_tier` declines an arena that
  carries one, so a warp reaches its operand and the chain rule holds.
- No compilation or evaluation at expansion (`[dependencies]` holds no
  `pixelflow-codegen`), and no fold is unrolled: it lowers to one `Reduce`.
- No stage accepts what a later one refuses: every advertised method name has a
  path through sema, lowering and emission, under both macros.
- No collection type, recursion, `X` or `Y` in a helper, mask read as a number,
  or whole-record arithmetic: each is refused where it is written.
- No call's value becomes a constant: one entry at one instantiation is one program.

## Contract
`src/lib.rs` exports exactly `kernel!` and `kernel_raw!`, both `expand(input,
optimizer)`. Behind them: `parser.rs` (tokens to `ast.rs`),
`sema.rs` (types, consts, call graph, binding times; scopes in `symbol.rs`),
`lower.rs` (AST to a term at a `Site`), `emit.rs` (arena to `TokenStream`;
`Staged` for a kernel-taking entry). Only `pixelflow-graphics` uses it.

## Enforced by
- Types: `Ty` keeps `Bool` and `F32` apart; "do not optimize" is `Identity`;
  desloppify `mask-is-not-a-number` covers `pixelflow-compiler/**`.
- `tests/derivative_under_warp.rs`, `tests/the_library_is_the_builders.rs`,
  `tests/fold_is_kernel_over.rs`, `tests/binding_times.rs`,
  `tests/rustc_is_the_oracle.rs`, `tests/optimized_agrees_with_raw.rs`,
  `tests/bake_vs_scalar_truth.rs`; `lib.rs` `every_advertised_method_compiles`.
- Unit tests: `emit.rs` `an_optimizer_is_offered_only_entries_it_can_read`; `sema.rs`
  `recursion_is_refused`; `parser.rs` `a_collection_spelled_any_way_is_refused_naming_its_sections`.

## Behind
- `macro_tier`, `DwrtFree` and `kernel_raw!` are in `lib.rs`, and `kernel!`
  saturates any entry with no structural or kernel-typed parameter and no
  `Dwrt`; the plan optimizes nothing at expansion unless the instance is
  declared, and deletes all three (the-language-is-kernel §1.8, D2, D14).
- A fold's ends are `u32` (`known_range`) and a structural count read as a
  value rides `Param(u8)`, at most 256, against the 64-bit control plane (A5, D2).

## Decided in
`docs/archive/plans/2026-09-08-macro-tier-is-arena-native.md`,
`docs/language/2026-09-25-the-language-is-kernel.md` (§1.1–§1.5, §1.8, D2, D14),
`docs/language/2026-09-24-one-pipeline.md` (§1.2, M1–M3), `docs/DENOTATION.md` §2, §8.
