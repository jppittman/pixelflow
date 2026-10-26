# pixelflow-compiler

**Is.** The language's front end: a function from a `kernel!` block's tokens to
the code that rebuilds, at load time, the kernels it denotes. An entry is a term
of arrow type with uniform parameters (DENOTATION §2); the optimizer it hands
that term to is an `Optimize` value, never a branch (§8).

## Owns
- The surface: grammar, types (`Ty`: `f32`, `bool`, `usize`, record, kernel)
  and refusals, each citing its plan section (`PLAN`); the IR keeps one lane
  for a number and a mask, so the difference lives in `Ty` (Behind).
- Names as the language compares them (`Name`), a method or projection resolved
  once in the parser, how a written name binds (`Scopes`), binding times as the
  host sees them (the host `fn`, the `Args` record, `const` evaluation), and
  which `Optimize` each macro hands `expand` (`kernel!` saturates, `kernel_raw!` is `Identity`).

## Never
- No construct is defined here: lowering calls `pixelflow_ir::library`, `ExprArena::close_over`,
  `KnownRange::admits` and `ExprArena::apply`; a fold lowers to one `Reduce`, never unrolled; a copy is a future divergence.
- No `Dwrt` is resolved, no kernel compiled or evaluated, at expansion:
  `macro_tier` declines an arena carrying one; only `const` items are evaluated.
- No stage accepts what a later one refuses: a collection type, recursion, `X`
  or `Y` in a helper, a mask read as a number, whole-record arithmetic, a call's
  value made a constant — each refused where written. No module imports its
  client: parser → ast; sema → ast; lower → ast, sema; emit → ast, sema, lower.

## Contract
`src/lib.rs` exports exactly `kernel!` and `kernel_raw!`, both `expand(input, optimizer)` over
`front_end`: `parse` → `analyze` → `emit_kernel`, each stage refusing with a `syn::Error` at the
construct (a block with no entry has none and is refused at the invocation). A module's contract
is the `# Contract` of its `mod.rs`, which re-exports exactly:
- `src/ast/mod.rs`, the vocabulary: `PLAN`; `KernelDef`, `ClosureEntry`, `Callable`,
  `RecordId`, `RecordItem`, `RecordField`, `ConstItem`, `FnItem`, `Role`, `Param`;
  `FunctionType`, `bare_name`, `is_f32`; `Name`, `same_name`, `Method`,
  `LibraryMethod`, `Projection`; `Scopes`; `Expr`, `FieldExpr`, `LiteralExpr`, `Literal`,
  `BinaryOp`, `BinaryExpr`, `UnaryOp`, `UnaryExpr`, `MethodCallExpr`, `ProjectExpr`,
  `CallExpr`, `Arity`, `arity_mismatch`, `IfExpr`, `FoldExpr`, `RangeExpr`, `Reduction`,
  `CastExpr`, `Stmt`, `LetStmt`, `BlockExpr`.
- `src/parser/mod.rs`: `parse`.
- `src/sema/mod.rs`: `analyze`, `AnalyzedKernel` (`parameters`), `ConstValue`, `Parameter`,
  `Scalar`; `range_bounds`, `Count`, `RangeScope`, `folds_nested_too_deep`.
- `src/lower/mod.rs`: `Site`, the one real trait, defined there; `lower_entry`;
  `FoldBound`, `Expansion` (`lower`), `Lowered` (`arena`, `root`, `scalars`).
- `src/emit/mod.rs`: `emit_kernel`. Only `pixelflow-graphics` uses the crate.

## Enforced by
- Types: `Ty` keeps `Bool` and `F32` apart; `Name` keys every map of names; a
  `Method` or `Projection` is resolved once; "do not optimize" is `Identity`;
  desloppify `mask-is-not-a-number` covers `pixelflow-compiler/**`.
- Visibility: `rg -n '^\s*pub (fn|struct|enum|trait|const|type|[a-z_]+:)' src`
  lists the two `#[proc_macro]` fns alone; `rg 'pub\(crate\)'` outside `mod.rs`
  and `lib.rs` lists only what some `mod.rs` re-exports (review-grade: Behind).
- `tests/rustc_is_the_oracle.rs` pins every refusal's text and span; the other eleven
  `tests/*.rs` run what the macros emit. Unit tests, `src/tests/`, one file per module, a child
  of the root, drive a stage from the one before (`parse` → `analyze` → `Expansion::lower` →
  `front_end`) and can name no `pub(super)` item: the compiler holds that law. `scripts/check-intent.sh`.

## Behind
- Sema's mask/scalar typing (`Ty::Bool` against `Ty::F32`) is the arena's job:
  T4 has `kernel!` emit typed pushes and deletes the duplicate here (typed-arena §2).
- `macro_tier`, `DwrtFree` and `kernel_raw!` stay in `lib.rs`; the plan optimizes nothing at
  expansion unless the instance is declared, and deletes all three (the-language-is-kernel §1.8, D2,
  D14). Count parameters are uniforms and bound folds by value; no
  structural placeholder is built.
- A fold's bounds are computed by sema and discarded, so lowering computes them again:
  `range_bounds` and `RangeScope` are on sema's contract only because the result is not stored on
  the analyzed fold (the-front-ends-stages §4).
- `node` (`src/emit/arena.rs`) and `Staged` (`src/emit/staged.rs`) are two
  spellings of "an IR call as the statement that makes it"; replaying an arena
  into a `Site` would reduce them to one (the-front-ends-stages §4).
- The visibility greps are review's and desloppify's, not CI's (2026-10-22): the
  first becomes a step of `scripts/check-intent.sh`, over every crate whose
  `mod.rs` directories carry an `intent.md`, in the CL after the-front-ends-stages §3.

## Decided in
`docs/compiler/2026-10-22-the-front-ends-stages-are-its-modules.md`,
`docs/archive/plans/2026-09-08-macro-tier-is-arena-native.md`,
`docs/language/2026-09-25-the-language-is-kernel.md` (§1.1–§1.5, §1.8, D2, D14),
`docs/language/2026-09-24-one-pipeline.md` (§1.2, M1–M3),
`docs/language/2026-10-17-the-arena-is-typed.md` (T4), `docs/DENOTATION.md` §2, §8.
