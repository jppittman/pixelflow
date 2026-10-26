# pixelflow-compiler/src/sema

**Is.** The front end's second stage: the function from a parsed `KernelDef`
to the same definition checked — every name bound, every expression typed,
every `const` evaluated, every fold's range admitted — so lowering's term is
well-formed by construction (DENOTATION §2), or the first refusal at its construct.

## Owns
- The stage's function and output (`analyze.rs`): `analyze`, `AnalyzedKernel`,
  the block's `Items` by name, and `Ty`, the type a body's every name and
  expression has: `f32`, `bool`, a `usize` count, a record, a kernel. The IR
  has one lane for a number and a mask; the difference lives here.
- One body's analysis (`body.rs`): `FnAnalyzer`, the binding kinds (`Symbol`,
  `SymbolKind`) over the vocabulary's `Scopes`, a kernel's move check per path.
- Counts and `const`s (`consts.rs`): a `usize` expression checked as rustc's
  const evaluator checks it; a fold's bounds, known or value; every
  `const`'s value, each operation rounded once in `f32`.
- The call graph (`calls.rs`): a DAG, or the first edge that closes a cycle.
- `AnalyzedKernel::parameters`: the one definition of an entry's uniform order.

## Never
- No parsing: every `Method`, `Projection` and `BinaryOp` arrives resolved.
- No IR and no term: nothing here builds an arena node or names a lane. A
  fold's bounds are checked, not lowered; a `const` is a value, not a node.
- No type leaves the module: `Ty` is `pub(super)`, and no stage downstream
  asks whether an expression is an `f32` or a mask.
- No import of a client: no file under `sema/` names `lower` or `emit`.
- No item but the contract's leaves: every other item is `pub(super)` or private.

## Contract
`mod.rs`, under `# Contract`: `analyze`, `AnalyzedKernel` (with `parameters`),
`ConstValue`, `Parameter`, `Scalar` from `analyze.rs`; `range_bounds`, `Count`,
`RangeScope`, `folds_nested_too_deep` from
`consts.rs`; each `pub(crate) use`d there.

## Enforced by
- Types: `Ty::Bool` against `Ty::F32`, so a number used as a mask is a type
  error; `Scopes<Symbol>` holds every binding, so a name resolves in one place;
  `Count` distinguishes `Known` and `Name`, so lowering keeps
  each bound at its proper binding time.
- Visibility: `rg 'pub\(crate\)' pixelflow-compiler/src/sema --glob '!mod.rs'`
  lists only what `mod.rs` re-exports; `Ty` is `pub(super)`.
- `pixelflow-compiler/src/tests/sema.rs`: fifty-seven tests, each `parse` then
  `analyze`, asserting the `AnalyzedKernel` or the refusal text; the sweep
  `every_advertised_method_compiles` in `pixelflow-compiler/src/tests/front_end.rs`
  types every advertised method through the pipeline. `tests/rustc_is_the_oracle.rs`
  pins the refusal texts and spans. `scripts/check-intent.sh`, `scripts/check-doc-paths.sh`.

## Behind
- The mask/scalar typing duplicates what the typed arena will enforce; T4 of
  `docs/language/2026-10-17-the-arena-is-typed.md` deletes it here, keeping `Usize` and `Record`.
- A fold's bounds are computed here and discarded, so lowering computes them
  again; `range_bounds` and `RangeScope` are on the contract only because the
  result is not stored on the analyzed fold (the-front-ends-stages §4).
- A known bound is a `u64`, checked against the exact `f32` index limit;
  value bounds are checked when their counts are available.

## Decided in
`docs/compiler/2026-10-22-the-front-ends-stages-are-its-modules.md` (§1.1 map,
§1.2 sema contract, §1.3 `Ty` stays, §1.4 `folds_nested_too_deep`, §1.5 tests),
`docs/language/2026-09-25-the-language-is-kernel.md` (§1.2–§1.6, D7, D-a),
`docs/language/2026-10-17-the-arena-is-typed.md` (T4), `docs/DENOTATION.md` §2.
