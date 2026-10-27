# pixelflow-compiler/src/ast

**Is.** The front end's vocabulary: the AST a `kernel!` block parses to — an
entry's spelling before it is a term of arrow type (DENOTATION §2) — and the
one reading of every question two stages ask of it: which record a type names,
which method a name is at an arity, which binding a written name means.

## Owns
- The definition (`def.rs`): `KernelDef`, a sum of the closure and items
  spellings, and `Callable`, the view of an entry or helper both give.
- A declared type's surface (`types.rs`): `FunctionType`, the one function type
  against every misspelling of it; `bare_name`, through a macro's invisible group.
- Names as the language compares them (`name.rs`): `Name`, an identifier without
  its `r#`, and the two the language spells itself; `Method`, resolved with its
  arity once, in the parser; `Projection`, by name.
- How a written name binds (`scopes.rs`): `Scopes<V>`, Rust's lexical scoping
  over any binding, which sema and lowering both resolve through.
- What a literal denotes before its position types it (`expr.rs`); the one
  refusal two stages make, `arity_mismatch`; `PLAN`, the plan every refusal cites.

## Never
- No stage function: nothing here parses, checks, lowers or emits. It holds
  what was written and answers questions about it.
- No semantic type: `Ty` is sema's. This module knows a type's spelling, not
  whether a body's expression is an `f32` or a mask.
- No IR and no evaluation: a fold is a `FoldExpr`, a `const` is its `init`.
  Sema evaluates; lowering builds the arena.
- No import of a client: no file under `ast/` names `parser`, `sema`, `lower`
  or `emit`.
- No second reading: a record's name, a method, a projection and a binding each
  resolve in one function here, and every stage calls it.

## Contract
`mod.rs`, under `# Contract`: every type and function it `pub(crate) use`s.
`def.rs`: `KernelDef`, `ClosureEntry`, `Callable`, `RecordId`, `RecordItem`,
`RecordField`, `ConstItem`, `FnItem`, `Role`, `Param`. `types.rs`:
`FunctionType`, `bare_name`, `is_f32`. `name.rs`: `Name`,
`same_name`, `Method`, `LibraryMethod`, `Projection`. `scopes.rs`: `Scopes`.
`expr.rs`: `Expr`, `FieldExpr`, `LiteralExpr`, `Literal`, `BinaryOp`,
`BinaryExpr`, `UnaryOp`, `UnaryExpr`, `MethodCallExpr`, `ProjectExpr`,
`CallExpr`, `Arity`, `arity_mismatch`, `IfExpr`, `FoldExpr`, `RangeExpr`,
`Reduction`, `CastExpr`, `Stmt`, `LetStmt`, `BlockExpr`. And `PLAN`, defined there.

## Enforced by
- Types: `Name` is the key of every map of names and `Name::of` its one
  constructor; `RecordId` is made only by `KernelDef::record_named`; a
  `MethodCallExpr` stores a `Method`, so no later stage reads the spelling;
  `Scopes<V>` is generic in what is bound, so it cannot depend on sema's kinds.
- Visibility: every field is `pub(crate)`; every helper the contract does not
  list is private; `rg 'pub\(crate\)'` outside `mod.rs` lists only re-exports.
- `pixelflow-compiler/src/tests/ast.rs` drives `bare_name` and `Scopes`;
  `pixelflow-compiler/src/tests/parser.rs`
  drives the operators and `args_record` through `parse`; the integration
  tests drive the macros. `scripts/check-intent.sh`, `scripts/check-doc-paths.sh`.

## Behind
- The mask/scalar distinction a body's expression carries lives in sema's `Ty`,
  not in this vocabulary or the arena; the typed arena (T4) moves it to the IR
  (`docs/language/2026-10-17-the-arena-is-typed.md`).

## Decided in
`docs/compiler/2026-10-22-the-front-ends-stages-are-its-modules.md` (§1.1 model
map, §1.2 ast contract, §1.3 `Scopes` is vocabulary, §1.4 `PLAN`),
`docs/language/2026-09-25-the-language-is-kernel.md` (§1.2–§1.5), `docs/DENOTATION.md` §2.
