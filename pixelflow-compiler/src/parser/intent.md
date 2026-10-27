# pixelflow-compiler/src/parser

**Is.** The front end's first stage: the function from a `kernel!` block's
tokens to the vocabulary's `KernelDef` — an entry's spelling, before it is a
term of arrow type (DENOTATION §2) — or the refusal at the token that spells
nothing the language means.

## Owns
- The grammar (`mod.rs`): which of Rust's syntax is the kernel language, and
  the one list of what is refused here against what is parsed and refused by
  `sema`.
- The two forms (`items.rs`): a `syn::File` of records, consts and `fn`s, or
  `|params| expr`, each item converted in declaration order.
- A body's expressions (`expr.rs`): the one dispatcher every nesting level
  recurses through, each form's arm, a method resolved once with its arity
  (`Method::resolve`) and a projection by name, the operator tables.
- A fold's six spellings (`fold.rs`): the range, the binder, the monoid, and
  the refusal of every other method on a range.
- A block (`block.rs`): `let` of a plain name or a tuple of them, the final
  expression, and the number a literal denotes — a float rounded once, an
  integer kept exact.
- The nesting bound (`MAX_NESTING`, in `mod.rs`): a body chained deeper than
  1,024 terms is refused at the chain's head token, in O(1), on rustc's stack.

## Never
- No semantics: nothing here knows a name's type, whether it is bound, or what
  a `const` evaluates to. Those refusals are `sema`'s, and the module doc lists
  which is whose.
- No pass-through: nothing is carried along for a later stage to refuse. The
  AST holds only what the language means.
- No second resolution: a method, a projection and an operator each resolve
  here, once, into the `Method`, `Projection` or `BinaryOp` the AST stores.
- No import of a client: no file under `parser/` names `sema`, `lower` or `emit`.
- No item but `parse` leaves the module: every other item is `pub(super)` or private.

## Contract
`mod.rs`, under `# Contract`: `parse`, implemented in `items.rs` and
`pub(crate) use`d there. One function, total over the module doc's grammar,
returning `syn::Result<KernelDef>`.

## Enforced by
- Types: the AST stores a `Method`, a `Projection`, a `BinaryOp`, never a
  spelling; `Nesting` is handed down by value, so a finished level takes its
  depth with it; `MethodTail` moves a fold's call out of its receiver.
- Visibility: `rg 'pub\(crate\)' pixelflow-compiler/src/parser` lists `parse`
  alone; `mod.rs` re-exports nothing else.
- `pixelflow-compiler/src/tests/parser.rs`: fifty-four tests, each through
  `parse`, asserting the `KernelDef` or the refusal text; the bound's two spell
  `1024`. `tests/rustc_is_the_oracle.rs` pins the refusal texts and spans.
  `scripts/check-intent.sh`, `scripts/check-doc-paths.sh`.

## Behind
- A nested delimiter (`{{…}}`, `f(f(…))`, `((…))`) overflows `syn`'s own
  recursion 400 to 800 deep, below `MAX_NESTING`, and no bound in this crate
  reaches it; the `MAX_NESTING` doc records the measurements.

## Decided in
`docs/compiler/2026-10-22-the-front-ends-stages-are-its-modules.md` (§1.1 map,
§1.2 parser contract, §1.4 refusals, §1.5 tests),
`docs/language/2026-09-25-the-language-is-kernel.md` (§1.2–§1.6, D7),
`docs/DENOTATION.md` §2.
