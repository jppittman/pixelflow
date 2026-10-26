# pixelflow-compiler/src/lower

**Is.** The front end's third stage: the function from an analyzed entry to a
term — the IR everything downstream speaks (DENOTATION §2) — built at a `Site`, one
walk run at expansion or when the host function is called, or the first refusal.

## Owns
- The one walk (`walk.rs`): `lower_entry` — uniforms declared first in
  `AnalyzedKernel::parameters` order, kernel parameters admitted next, then the
  body children-first; a `let` one shared node, a helper inlined (β) in a frame
  of its own, a `const` the value sema gave it, each fold one `Reduce`.
- Where a term is built (`Site`, `mod.rs`): each step of the walk as a call.
- The expansion site (`site.rs`): `Expansion`, an `ExprArena` at macro
  expansion for entries with no kernel parameters, and its
  output `Lowered`.

## Never
- No construct is defined here: `fract`, `hypot`, `clamp`, the derivative, a
  binder's choice and rename and a kernel's application are `pixelflow_ir`'s,
  called through the `Site`; a copy is a future divergence.
- No checking that is sema's: a name that sema refuses is an `unreachable!`
  here, never a second refusal with a second text.
- No unrolling, no optimization, no token: a fold is one `Reduce`, the arena
  leaves as lowered, and `emit::Staged`, the second `Site`, lives in `emit`.
- No placeholder enters an expanded arena: a `Param` reaching emission
  fails loudly.
- No import of a client: no file under `lower/` names `emit`.

## Contract
`mod.rs`: `Site`, the one real trait, defined there; under `# Contract`,
`lower_entry` from `walk.rs`; `FoldBound` from `mod.rs`; `Expansion` (with
`lower`) and `Lowered` (fields `arena`, `root`, `scalars`) from `site.rs`; each
`pub(crate) use`d there.

## Enforced by
- Types: `Expansion`'s `Site::Argument` is `Infallible`, so no entry taking a
  kernel lowers at expansion; `Scopes<Binding>` holds every name, so a binding
  resolves before a coordinate; `FoldBound` keeps known and value
  counts distinct until a site builds a `Reduce`.
- Visibility: `rg 'pub\(crate\)' pixelflow-compiler/src/lower --glob '!mod.rs'`
  lists only what `mod.rs` re-exports.
- `pixelflow-compiler/src/tests/lower.rs`: `parse`, `analyze`, `Expansion::lower`,
  asserting the `Lowered` term or the refusal text; `tests/the_library_is_the_builders.rs`,
  `tests/fold_is_kernel_over.rs`, `tests/kernel_typed_parameters.rs` pin each construction
  against the builder's. `scripts/check-intent.sh`, `scripts/check-doc-paths.sh`.

## Behind
- Value bounds stay child terms of the fold; known bounds are checked against the exact `f32`
  index limit at expansion.
- `resolve`'s coordinate-in-a-helper and binding-before-coordinate `Err` arms are
  reachable from no program sema admits; the next desloppify pass makes them
  `expect`s naming the sema refusal (the-front-ends-stages §4).
- A fold's bounds are computed by sema and again here (`range_bounds`), since
  the result is not stored on the analyzed fold (the-front-ends-stages §4).

## Decided in
`docs/compiler/2026-10-22-the-front-ends-stages-are-its-modules.md` (§1.1, §1.2, §1.5, §4),
`docs/language/2026-09-25-the-language-is-kernel.md` (§1.2, §1.4, B5, D-a), `docs/DENOTATION.md` §2.
