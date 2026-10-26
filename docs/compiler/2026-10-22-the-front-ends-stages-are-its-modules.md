# The front end's stages are its modules, and the AST is their vocabulary

**Status:** `Plan of record` (JP, 2026-10-22: "the models that are missing become modules with tight APIs"), implemented as the seven CLs of §3,
every path §1.1 maps now in the tree. Layout and visibility only: what `kernel!` and `kernel_raw!` accept, emit and refuse does not change.
**Verified against:** bab3441c4ed9143708423b42f921564d3b7edc5b. Every `file.rs:line` below is bab3441's; the series renames the six flat files into
directories, so after it the bare names `ast.rs`, `sema.rs`, `emit.rs` and `symbol.rs` are gone.

## 0. The claim

A front end is a pipeline over one vocabulary. Each stage is a function from the stage before it to its own output type, or a refusal spanned at the
construct: `parse` gives a `KernelDef`, `analyze` gives it back checked as an `AnalyzedKernel`, lowering gives a term (`Lowered`), emission gives
tokens. The vocabulary is what those functions pass: the AST, the names in it and what each resolves to, and how a written name binds. A module is a
contract (`mod.rs`) and an implementation behind it, so a *stage* is a module whose contract is its one function and the types it returns, and the
*vocabulary* is a module whose contract is its types and the one reading of each question two stages ask of them. The six flat files under
`pixelflow-compiler/src/lib.rs:51-56` are not that: every item is `pub(crate)` whether one stage reads it or four (109 items, 83 bare `pub` fields);
one model spread over two files imports its client (`symbol.rs:63` reads `sema::Ty`, `sema.rs:78` reads `symbol`); 2,900 lines of tests reach
whatever `use super::*` offers. The four models the desloppify pass made real are types today, and a type's module is the vocabulary's: three are files of it, the fourth is a convention.

## 1. The denotation

### 1.1 The models, translated, and the map (Q1)

| model | denotation | at bab3441 | what the code has that the denotation does not |
|---|---|---|---|
| 1 declared type | `Type → {f32, bool, usize, record, kernel} ∪ refusal`, one reading | `ast.rs:390-502` (`FunctionType`, `bare_name`, `is_f32`, `is_usize`), `:112` (`record_named`); `sema.rs:130` (`Ty::from_syn`) | two halves by design: the surface half is the vocabulary's; `Ty` is sema's and stays there (typed-arena T4 owns it) |
| 2 definition | `KernelDef = Closure(ClosureEntry) \| Items{records, consts, fns}`; `Callable` the view both give | `ast.rs:62-178, 271-362` | nothing: it is the vocabulary's spine |
| 3 name | `Name` (unraw'd); a method at an arity is a `Method`, a projection a `Projection`; a use is the innermost binding in scope, `Scopes<V>` | `ast.rs:189-218, 763-861`; `symbol.rs:108-155` (`Scopes`), `:159-244` (`SymbolTable`) | the scoping rule and sema's *kinds* of binding share a file, hence the cycle; `SymbolTable` is eleven one-line forwards over `Scopes<Symbol>` |
| 4 refusal | `syn::Error::new(span, claim)` by the earliest stage that sees the construct; the pipeline is `front_end` (`lib.rs:398-404`) | every stage; shared claims `ast.rs:901`, `sema.rs:1719`; `PLAN` at `lib.rs:66` | a convention, correctly; `PLAN` sits in the stages' client |

**By stage, with the AST as the vocabulary module.** A model is a type, and a type belongs with the types it is defined in terms of; every stage names
models 1–3 together (the parser imports 37 `ast` items, sema 36, lower 22, emit 8), so a directory per model would add three contracts nothing imports
alone. Two placements are decided from the denotation, not from counts:

- **Lowering is a module**, though `emit.rs:56` is its one reader. `P` is a composition of stages, each a position in the pipeline (ontology compiler.md, "P",
  "Stage"), and "lowering produces a template, a replay of `ExprArena` pushes" (compiler.md, "Program, template, instance"): lowering is the stage whose
  output is the IR, the one representation boundary `lib.rs:32-36` names as where two stages can disagree about what the language is. A contract there is
  the denotation's; a reader count is a fact about today's code. Consequence: `Lowered` is a contract type tested through `Expansion::lower`, and `Holes` must leave it (§1.2, lower).
- **The parser sits beside the vocabulary, not inside it.** `parse` is a stage; the AST is what four stages share. The skill's "the types those traits
  mention" are *listed* in the root, which the worked examples do by import when another module owns them (`core-term/src/term/mod.rs` imports
  `AnsiCommand`; `pixelflow-codegen/src/emit/regalloc/mod.rs` imports `Function`); the ontology names "one syntax and one parser" (language.md, "kernel!").
  `Scopes<V>` is vocabulary: generic in what is bound, dependent on `Name` alone, resolved through by sema and lowering "so the two cannot disagree" (`symbol.rs:53-55`); how a written name binds is part of what the AST says.

```text
pixelflow-compiler/            intent.md: Contract points at the five mod.rs below
  src/lib.rs                   kernel!, kernel_raw!, expand, front_end, macro_tier, DwrtFree; five `mod`s; #[cfg(test)] mod tests;
  src/ast/      mod.rs         THE VOCABULARY: doc (ast.rs:1-47), `mod`s, pub(crate) use of every type below, PLAN                        + intent.md
    def.rs                     model 2: KernelDef, ClosureEntry, Callable, RecordId, RecordItem, RecordField, ConstItem, FnItem, Role, Param
    types.rs | scopes.rs       model 1: FunctionType, bare_name, is_f32, is_usize  |  model 3: Scopes<V>                     <- git mv symbol.rs
    name.rs                    model 3: Name (+coordinate, coordinates; COORDINATES private), same_name, Method, LibraryMethod, Projection
    expr.rs                    Expr, FieldExpr, LiteralExpr, Literal, BinaryOp, BinaryExpr, UnaryOp, UnaryExpr, MethodCallExpr, ProjectExpr, CallExpr, Arity, arity_mismatch, IfExpr, FoldExpr, RangeExpr, Reduction, CastExpr, Stmt, LetStmt, BlockExpr
  src/parser/   mod.rs         PARSE: the grammar and refusal list (parser.rs:1-192); pub(crate) use items::parse                        + intent.md
    items.rs, expr.rs, fold.rs, block.rs       parser.rs:206-700 | :702-1211 (Nesting, convert_expr, methods, binary_op/unary_op) | :1213-1644 | :1646-2103
  src/sema/     mod.rs         ANALYZE: analyze, AnalyzedKernel, ConstValue, Parameter, Scalar, Bounds, StructuralRange, RangeScope, range_bounds, folds_nested_too_deep   + intent.md
    analyze.rs, body.rs, consts.rs, calls.rs   sema.rs:86-611 (Ty, Items, analyze, the output types) | :613-1507 (FnAnalyzer; Symbol, SymbolKind) | :1509-1970 | :1972-2093
  src/lower/    mod.rs         LOWER: trait Site; pub(crate) use site::{Expansion, Lowered}, walk::lower_entry                              + intent.md
    site.rs, walk.rs           lower.rs:195-445 (Expansion, Lowered + range_of, Holes, OPEN_RANGE, HOLE_STRIDE, UNBOUND) | :81-193, 447-914
  src/emit/     mod.rs         EMIT: pub(crate) use host::emit_kernel                                                                          + intent.md
    host.rs, arena.rs, staged.rs    emit.rs:61-122, 168-415 (emit_kernel, entry_arena, the host items) | :124-166, 417-632, 834-841 | :634-832
  src/tests/    mod.rs, ast.rs, parser.rs, sema.rs, lower.rs, front_end.rs                                                                   (§1.5)
```
An implementation file is carved only at a bar the flat file already draws or a model boundary. A `mod.rs` holds the module doc, `mod` lines, `pub(crate)
use`, the contract block and (lower) the one real trait; every other item is `pub(super)` or private. A re-exported item is `pub(crate)` where it is
implemented, since a `pub(crate) use` cannot widen a `pub(super)` item (E0364; the regalloc shape: `pixelflow-codegen/src/emit/regalloc/local.rs:453` is
`pub(in crate::emit) struct LinearScan`, re-exported `pub(super)`). Imports after: parser → ast; sema → ast; lower → ast, sema; emit → ast, sema, lower. No module imports its client; `lib.rs` is imported by nothing.

### 1.2 The contracts (Q2)

Every crate-visible item of bab3441 appears below once, or in §2 as demoted, moved or deleted. Every entry returns `syn::Result`, `Err` a `syn::Error`
spanned at the construct (§1.4). `Site` is the one real `trait` (two impls); the rest are documented shapes in each `mod.rs` doc, per the skill
`desloppify/skills/behavioral-contracts/SKILL.md`.

**ast** (`ast/mod.rs`): the vocabulary. No stage function; the contract is the types the tree lists, each `pub(crate) use`d, and the one reading of
every question two stages ask of them. Fields the parser builds and the stages read are `pub(crate)`.
```rust
pub(crate) const PLAN: &str;                                                    // <- lib.rs:66: the plan every refusal cites
enum KernelDef { Closure(ClosureEntry), Items { records: Vec<RecordItem>, consts: Vec<ConstItem>, fns: Vec<FnItem> } }
impl KernelDef { fn records(&self) -> &[RecordItem]; fn consts(&self) -> &[ConstItem]; fn fns(&self) -> &[FnItem];         // declaration order; a closure's are empty
    fn record_named(&self, ty: &Type) -> Option<RecordId>;  fn record(&self, id: RecordId) -> &RecordItem; }                 // the one resolution of a record's name
impl ClosureEntry { fn callable(&self) -> Callable<'_>; }   impl FnItem { fn callable(&self) -> Callable<'_>;  fn role(&self) -> Role;   // pub ⇒ Entry
    fn takes_a_kernel(&self) -> bool;  fn has_args_record(&self) -> bool;  fn args_record(&self) -> Ident; }   // params non-empty and no kernel; UpperCamelCase + `Args`, a leading digit gains `_`
impl Callable<'_> { fn kernel_parameters(&self) -> impl Iterator<Item = &Ident>; }  impl Param { fn is_kernel(&self) -> bool; }   // declaration order = admission order
impl FunctionType { fn of(ty: &Type) -> FunctionType; }               // Kernel iff `impl Fn(f32, f32) -> f32`, else Misspelled(why), through Paren/Group/Reference
fn bare_name(ty: &Type) -> Option<&Ident>;  fn is_f32(ty: &Type) -> bool;  fn is_usize(ty: &Type) -> bool;   // through a macro's group; None for `::p`, qself, generics
impl Name { fn of(ident: &Ident) -> Name;  fn coordinate(&self) -> Option<Axis>;  fn coordinates() -> impl Iterator<Item = (Name, Axis)>; }   // Eq/Hash unraw'd; X then Y
fn same_name(a: &Ident, b: &Ident) -> bool;   impl Projection { fn from_name(name: &Name) -> Option<Projection>; }   // V DX DY DXX DXY DYY
impl<V> Scopes<V> { fn default() -> Self; fn push_scope(&mut self); fn pop_scope(&mut self); fn bind(&mut self, Name, V); fn lookup(&self, &Name) -> Option<&V>; }
    // Rust's lexical scoping over bindings of `V`: a function scope no pop reaches, one map per open block; `bind` shadows any binding of `name` in scope, the same block's too; `lookup` is the innermost, else the function's
impl Method { fn resolve(name: &str, arity: usize) -> Option<Method>;         // clone/0, OpKind::from_method_call, LibraryMethod::ALL; name and arity together
    fn advertised() -> impl Iterator<Item = (&'static str, usize)>; }         // every (name, arity) a body may call: the parser's and the sweep's oracle
impl Expr { fn span(&self) -> Span;  fn named(&self) -> Option<&Ident>; }  impl CastExpr { fn named(&self) -> Option<&Ident>; }   // a name through parentheses
impl LiteralExpr { fn f32_value(&self) -> syn::Result<f32>;  fn usize_value(&self) -> syn::Result<u64>; }   // Err: > MANTISSA_DIGITS significant bits; a float, or > u64::MAX
impl Arity { fn supplied_verb(self) -> &'static str; }  fn arity_mismatch(span: Span, callee: impl Display, arity: Arity) -> syn::Error;
```
**parser** (`parser/mod.rs`; `parse` is `pub(crate)` in `items.rs`, from `parser.rs:213`):
```rust
trait Parse {   // implemented by `parser::parse`
    /// The `KernelDef` the tokens spell: `|params| expr` is `Closure`; a `syn::File` of `struct`, `const` and `fn` items is `Items`, lists in declaration order.
    /// Total over the module doc's grammar; every other token is `Err` at that token with the listed text, and nothing is carried along for a later stage to
    /// refuse. A method is resolved once (`Method::resolve`), a projection by name at arity one, a float rounded once, an integer kept exact. A body chained
    /// deeper than 1,024 terms is `Err` at the chain's head token, in O(1), on rustc's 8 MiB stack.
    fn parse(input: proc_macro2::TokenStream) -> syn::Result<KernelDef>;
}
```
**sema** (`sema/mod.rs`; `analyze` and the output types in `analyze.rs`, the two functions and `Bounds` in `consts.rs`). Lower reads nine of these
(`lower.rs:67-70`), emit five (`emit.rs:57, 429`); `Ty` leaves the contract (§1.3).
```rust
trait Analyze {
    /// `def` checked and its consts evaluated: every name bound in scope; every expression `f32` or `bool`, a `usize` only as `i as f32`, a record only by
    /// name, a kernel only applied or passed on once per path; every call a helper at its arity; the call graph acyclic; every `const` the value rustc gives;
    /// every fold's range constant, forward and admitted; at least one entry. Returns `def` unchanged. Err: the first refusal, at its construct (a block with
    /// no entry has none, and is refused at the invocation, sema.rs:316).
    fn analyze(def: KernelDef) -> syn::Result<AnalyzedKernel>;
    /// An entry's uniforms in declaration order: an `f32` one, a record one per field, a kernel-typed one omitted — the one definition of a program's uniform order.
    fn AnalyzedKernel::parameters<'a>(&'a self, params: &'a [Param]) -> Vec<Parameter<'a>>;
    /// `lo..hi` over literals, `usize` consts and `scope.structural`: `Known` when both evaluate, run forward and `Fold::admits` them; `Structural` when a
    /// bound names a structural parameter (rustc evaluates it at instantiation). Err: not constant, overflow, backwards, past `Fold::EXACT_BOUND`.
    fn range_bounds(range: &RangeExpr, scope: RangeScope<'_>) -> syn::Result<Bounds>;
    fn folds_nested_too_deep(span: Span) -> syn::Error;   // a fold opened with `Binder::COUNT` already open
    fn StructuralRange::text(&self) -> String;            // `lo..hi` as source: the key two folds over one range share
}
struct AnalyzedKernel { pub(crate) def: KernelDef, pub(crate) consts: HashMap<Name, ConstValue> }   enum ConstValue { F32(f32), Usize(u64) }
enum Parameter<'a> { Scalar(&'a Ident), Record { name: &'a Ident, id: RecordId, fields: &'a [RecordField] } }   enum Scalar { Param(Ident), Field(Ident, Ident) }
enum Bounds { Known(Range<u32>), Structural(StructuralRange) }   struct StructuralRange { pub(crate) lo: TokenStream, pub(crate) hi: TokenStream }
struct RangeScope<'a> { pub(crate) consts: &'a HashMap<Name, ConstValue>, pub(crate) structural: &'a [Ident] }
```
**lower** (`lower/mod.rs` defines `Site`; `Expansion`, `Lowered` in `site.rs`, `lower_entry` in `walk.rs`). `Holes`, `OPEN_RANGE`, `HOLE_STRIDE`,
`UNBOUND` and the `Param(u8)` hole (`lower.rs:199-272, 394-410`) are private to `site.rs`.
```rust
pub(crate) trait Site { type Term: Clone; type Argument: Clone;   // where lowering builds a term, each step of the walk as a call; the two impls differ only in *when* a step runs (`Expansion` at expansion, `emit::Staged` when the host fn is called); reproduced from lower.rs:107-158
    fn constant(&mut self, value: f32) -> Self::Term;  fn coordinate(&mut self, axis: Axis) -> Self::Term;  fn unary(&mut self, op: OpKind, operand: Self::Term) -> Self::Term;
    fn binary(&mut self, op: OpKind, operands: [Self::Term; 2]) -> Self::Term;  fn ternary(&mut self, op: OpKind, operands: [Self::Term; 3]) -> Self::Term;
    fn fract(&mut self, x: Self::Term) -> Self::Term;  fn hypot(&mut self, operands: [Self::Term; 2]) -> Self::Term;  fn clamp(&mut self, x: Self::Term, bounds: [Self::Term; 2]) -> Self::Term;
    fn derivative(&mut self, e: Self::Term, axis: Axis) -> Self::Term;  fn uniform(&mut self, scalar: Scalar) -> Self::Term;
    fn kernel_parameter(&mut self, name: &Ident) -> syn::Result<Self::Argument>;  fn count(&mut self, position: usize, name: &Ident) -> syn::Result<Self::Term>;
    fn open_fold(&mut self, depth: usize, monoid: Monoid, bounds: Bounds) -> Self::Term;  fn close_fold(&mut self, body: Self::Term) -> Self::Term;
    fn apply(&mut self, kernel: Self::Argument, at: [Self::Term; 2]) -> Self::Term; }
trait Lower {
    /// `entry`'s body as a term at `site`: uniforms declared first, every scalar read or not, in `analyzed.parameters(entry.params)` order; kernel parameters
    /// admitted next in `kernel_parameters()` order; then the body, children first, helpers inlined (β), consts folded, each fold one `Reduce`, a `let` one
    /// shared node. Err: a coordinate in a helper; a fold at depth `Binder::COUNT` counting inlined ones.
    fn lower_entry<S: Site>(entry: Callable<'_>, analyzed: &AnalyzedKernel, site: &mut S) -> syn::Result<S::Term>;
    /// `lower_entry` at a fresh arena. Err: `lower_entry`'s, and an entry that takes a kernel.
    fn Expansion::lower(entry: Callable<'_>, analyzed: &AnalyzedKernel) -> syn::Result<Lowered>;
    /// The range of one of this template's open folds; `None` for a known fold; panics on a placeholder no hole of this entry names. (The body of `Holes::range_of`, lower.rs:256-271, moved; no wrapper.)
    fn Lowered::range_of(&self, fold: Fold) -> Option<&StructuralRange>;
}
struct Lowered { pub(crate) arena: ExprArena, pub(crate) root: ExprId, pub(crate) scalars: Vec<Scalar>, holes: Holes /* private */ }
struct Expansion;   // Default; its fields stay private (lower.rs:288-296)
```
Hiding `holes` is not a pure move at `emit.rs:190-224`, which destructures `Lowered` and moves `arena` into the optimizer while `EntryContext` (`:238-242`)
holds `&Holes`. The shape after — the lower CL's one body edit; `node` (`:606`) reads `context.lowered.range_of(fold)`:
```rust
struct EntryContext<'a> { arguments: Vec<TokenStream>, structural: &'a [Ident], lowered: &'a Lowered }
fn entry_arena(entry: Callable<'_>, analyzed: &AnalyzedKernel, optimizer: &mut dyn Optimize) -> syn::Result<(TokenStream, Vec<Scalar>)> {
    let lowered = Expansion::lower(entry, analyzed)?;
    let context = EntryContext { arguments: lowered.scalars.iter().map(argument).collect(), structural: entry.structural, lowered: &lowered };
    let rewritten = match entry.structural.is_empty() { false => None,   // a template is emitted as lowered (emit.rs:185-189)
        true => optimizer.optimize(&lowered.arena, lowered.root).into_changed().map(|(arena, root)| match lowered.arena.uniforms().is_empty() {
            false => arena.relink(root, lowered.arena.buffers(), lowered.arena.uniforms()),
            true => (arena, root) }) };
    let (arena, root) = rewritten.as_ref().map_or((&lowered.arena, lowered.root), |(arena, root)| (arena, *root));
    let tokens = arena_to_tokens(arena, root, &context);
    Ok((tokens, lowered.scalars))
}
```
**emit** (`emit/mod.rs`; implemented in `host.rs`). `Staged` stays in emit: it writes `::pixelflow_core::__macro` paths with emit's snippets
(`emit.rs:758-777`); it is the emission of lowering's steps, the second `Site` impl.
```rust
trait Emit {
    /// The tokens that rebuild `analyzed`'s kernels at load time. Closure form: an expression, a `Kernel` or `move |p: f32, …| -> Kernel`. Items form: a
    /// `#[repr(C)]` struct per record, a host `const` per `pub const`, per entry a `#[must_use]` host `fn` (a kernel param as `&Kernel`, const generics over
    /// structural params) and, when `has_args_record`, its `Args` record with `write_into`. Every uniform is declared with the call's value as its default, in
    /// `parameters` order. `optimizer` is offered exactly the entries with no structural and no kernel-typed parameter (emit.rs:109-118, :206);
    /// `Declined`/`Unchanged` keep the lowered arena, `Changed` is relinked to declaration order; an entry taking a kernel is `Staged`. No `::std` path.
    /// Deterministic: equal input and optimizer answer, byte-equal tokens. Err: lowering's.
    fn emit_kernel(analyzed: &AnalyzedKernel, optimizer: &mut dyn Optimize) -> syn::Result<TokenStream>;
}
```
**lib.rs** keeps `kernel!`, `kernel_raw!` (`:329, :352`), `macro_tier` (`:366`), `DwrtFree` (`:379`), `front_end` (`:398`), `expand` (`:408`); loses
`mod symbol;` (`:56`) and `PLAN` (`:66`); gains `#[cfg(test)] mod tests;`.

### 1.3 Where `symbol` lives (Q3)

Nowhere: `symbol.rs` is renamed to `ast/scopes.rs` once what it held besides `Scopes` has gone home. `Scopes<V>` (`symbol.rs:108-155`) depends on `Name`
alone and is resolved through by sema (`sema.rs:624`) and lowering (`lower.rs:469, 520, 798`): vocabulary. `Symbol` and `SymbolKind` (`:69-101`) carry
`Ty` and are read by sema alone: `sema/body.rs`, private; the `symbol.rs:6-15` table joins sema's module doc. `SymbolTable` (`:159-244`) is eleven
one-line forwards over `Scopes<Symbol>` and is **deleted**: `FnAnalyzer.symbols` becomes `Scopes<Symbol>`, bound at the six `register_*` sites
(`sema.rs:657, 699, 715, 895, 1370`) and in `FnAnalyzer::new` (`:640-666`), which binds `X` and `Y` from `Name::coordinates()` where `SymbolTable::new`
(`:652`) did; `is_intrinsic` (`:743`) is its one-line body at its one caller; `lookup`/`push_scope`/`pop_scope` are `Scopes`'s own. `SymbolTable::coordinate`
(`:170`; callers `sema.rs:406`, `lower.rs:856`) is "a name the language spells itself": `Name::coordinate`, after which `Name::spelled` (`ast.rs:198`)
has one caller in its own file and goes private. The cycle dissolves because its halves were two things; `Ty` is then named by nothing outside sema and goes `pub(super)`.

### 1.4 Where a refusal lives (Q4)

A convention every contract states — every entry returns `syn::Result`, `Err` spanned at the construct, made by the earliest stage that sees it — not a
module: there is no type beyond `syn::Error`, and a `refusal/` would be format strings with no function of their own, a namespace in a name. The two
claims two stages make live in the earlier stage's contract: `Arity`/`arity_mismatch` (`parser.rs:1118`; `sema.rs:1170, 1306`) in `ast`;
`folds_nested_too_deep` (`sema.rs:890`; `lower.rs:656`) in `sema`, a claim about `Binder::COUNT` the vocabulary should not import. `refuse_where_clause`
(`parser.rs:690`; callers `:329, :473, :685`) is parser-private. `PLAN`, the citation every refusal makes, moves from `lib.rs:66` to `ast/mod.rs`, so `parser.rs:194`, `sema.rs:71` and `lower.rs:61` stop importing from their client.

### 1.5 Tests (Q5)

**The law.** A unit test lives in `src/tests/<module>.rs`, declared `#[cfg(test)] mod tests;` from `lib.rs`. A child of the crate root sees what the root
sees: every `pub(crate)` item — exactly what each `mod.rs` re-exports — plus `lib.rs`'s own `front_end` and `macro_tier`, and no `pub(super)` item of any
module. **The compiler holds the law**, which a `<module>/tests.rs` cannot: a child of `sema` sees every `pub(super)` item of `sema/body.rs`. A test drives
a stage from the stage before it (`parse` → `analyze` → `Expansion::lower` → `front_end`), never a hand-built intermediate, and asserts a contract's
return or the pipeline's refusal text. Nothing is widened for a test: the two tests reading `MAX_NESTING` (`parser.rs:2479, 2495`) spell the contract's
`1024`. No `#[cfg(test)]` remains in an implementation file (`rg '#\[cfg\(test\)\]' pixelflow-compiler/src --glob '!src/tests/**' --glob '!lib.rs'` is empty). The 129 unit tests at bab3441:

| file | n | moved unchanged | re-fixtured | deleted (and what observes the behavior) |
|---|---|---|---|---|
| `ast.rs:1062-1156` | 5 | 1 (`a_bare_name_is_seen_through_a_group…`) | 4: the three `from_syn` tests (`:1089, :1111, :1125`) drive `parse` over `\|\| X #op X`, same tables, observing `Expr::Binary`/`Unary` or the refusal text, in `tests/parser.rs`; `an_entrys_args_record…` (`:1135`) parses `pub fn <name>() -> f32 { X }` then calls `args_record()` | — |
| `symbol.rs:246-330` | 5 | — | 1: `a_lookup_sees_the_innermost_binding_in_scope` (`:315`), its `name` helper through `Name::of`, in `tests/ast.rs` | 4 `SymbolTable` tests (`:255-312`): the subject is deleted; sema's `a_local_used_after_its_block_ends_is_refused` (`:2156`), `a_let_named_after_a_coordinate_is_refused` (`:2184`), `a_parameter_shadowed_in_an_inner_block_is_in_scope_after_it` (`:2201`), `a_let_initializer_sees_the_binding_it_shadows` (`:2207`) observe each through `analyze` |
| `parser.rs:2105-3254` | 50 | 48 | 2: the bound spelled `1024` (`:2479, :2495`) | — |
| `sema.rs:2094-3472` | 63 | 57 | 4: the sweep `every_advertised_method_compiles` (`:3366-3460`) to `tests/front_end.rs`, its `expand` helper rewritten (below), the four test bodies unchanged | `every_advertised_op_method_has_a_typing` (`:2847`), `a_bit_pattern_op_without_a_typing_is_refused` (`:2863`): both reach private `method_typing`; the sweep observes the first (an untyped advertised op panics there), the second watches an `unreachable!` no advertised name reaches |
| `lower.rs:916-1025` | 3 | 1 (`a_helper_fold_nests…`, `:995`: `parse`, `analyze`, `Expansion::lower`, `Lowered{arena, root}`, all contract) | — | `a_binding_is_resolved_before_a_coordinate…` (`:967`), `a_coordinate_in_a_helper_is_refused_here_too` (`:979`): `lower_unanalyzed` (`:929-957`) forges an `AnalyzedKernel` sema never produces (`:945`) to watch a defensive arm no production output reaches |
| `emit.rs:843-938` | 3 | 3 (to `tests/front_end.rs`) | — | — |

110 moved, 11 re-fixtured, 8 deleted: 129. The sweep's `expand` needed sema-private `method_typing` (`:3368, :3399`) to shape a well-typed receiver.
Rewritten with least knowledge: the language has two value types, so for each `(name, arity)` of `Method::advertised()` it tries the receiver `X` and then
`X.lt(X)`, every argument `X`, and the method passes iff one compiles under the macro's pipeline. `Method::advertised` stays the oracle (on `ast`'s contract
for the parser, `parser.rs:1117, 1121`), so no second copy of `ast.rs:816-826` exists; the converse guard `a_name_the_front_end_does_not_advertise_is_still_refused`
(`:3449`) refuses on both receivers and keeps it honest. Integration tests (`pixelflow-compiler/tests/*.rs`, twelve files and `tests/common/mod.rs`) drive the macros: unchanged.

## 2. What it deletes (Q7)

| | before (bab3441) | after |
|---|---|---|
| crate-visible items (`pub(crate)`, `PLAN` included) | 109: ast 67, symbol 19, sema 13, lower 7, parser 1, emit 1, lib 1 | 92: ast 72, sema 12, lower 6, parser 1, emit 1 |
| bare `pub` fields | 83: ast 71, sema 6, lower 4, symbol 2 | 0 (80 `pub(crate)`: ast 71, sema 6, lower 3; `Lowered.holes` private; `Symbol`'s two sema-private) |
| `use crate::` edges between modules / cycles | 14 / 1 | 7 / 0 |
| `#[cfg(test)] mod tests` inside implementation files | 6 | 0 |
| unit tests | 129 | 121 |
| files under `src/` | 7 | 30: 7 moved, 5 `mod.rs`, 12 carved, 6 under `tests/` |

Deleted: `SymbolTable` and its eleven methods (`symbol.rs:157-244`, ~95 lines with docs); `mod symbol` (`lib.rs:56`); eight tests (~140 lines);
`lower_unanalyzed`; eleven `use crate::` lines. Demoted: `Ty` with `from_syn`/`of_a_value`/`name` (`sema.rs:112-174`, `pub(super)`); `Holes`, `OPEN_RANGE`,
`HOLE_STRIDE`, `UNBOUND` (private to `lower/site.rs`; `Holes::range_of`'s body becomes `Lowered::range_of`); `Name::spelled` (private); `SymbolKind`, `Symbol`
(sema-private); `COORDINATES` (ast-private). Moved: `BinaryOp::from_syn`, `UnaryOp::from_syn` (`ast.rs:1031-1060`) into the bodies of their only callers,
`binary_op`/`unary_op` (`parser.rs:1649, 1686`); `PLAN` into `ast`. Imports gone: `symbol → sema::Ty` (the cycle), `sema → symbol`, `lower → symbol`,
`{parser, sema, lower} → crate::PLAN`. Added, the only additions the ruling allows: five `mod.rs` of declarations, docs and `pub(crate) use` (~250 lines: the
contract blocks above and the moved module docs), five `intent.md` (≤60 lines each), `src/tests/mod.rs`. Net: more text, less code; no wrapper fn anywhere.

## 3. The series (Q6)

Seven stacked CLs, in this order. Each leaves the tree green and carries the `intent.md` its directory needs (the `Module intent` job,
`scripts/check-intent.sh`, fails on a `mod.rs` directory without one), the Contract and Enforced-by lines of `pixelflow-compiler/intent.md` for the files it
moves, and **every** live citation of the file, whether or not `scripts/check-doc-paths.sh` would catch it: `lower.rs` and `parser.rs` survive as basenames
elsewhere (`pixelflow-codegen/src/program/lower.rs`, `core-term/src/ansi/parser.rs`), so a stale citation of either passes the basename fallback while
pointing at nothing; the check is the floor, not the standard. Two commits per CL where a file becomes a directory: `git mv src/X.rs src/X/mod.rs` (a 100%
rename), then the carve, reviewed with `git diff -M -C`.

Gate, every CL: `cargo test -p pixelflow-compiler`; `cargo test -p pixelflow-graphics` (the macros' only consumer, `pixelflow-graphics/Cargo.toml`;
`pixelflow-graphics/src/fonts/loop_blinn/program.rs:31`); `cargo clippy -p pixelflow-compiler --all-targets -- -D warnings`; `bash scripts/check-intent.sh`;
`bash scripts/check-doc-paths.sh`; and the differential, on the pinned stable toolchain: in a worktree at the parent and at the head,
`RUSTC_BOOTSTRAP=1 RUSTC_WRAPPER= cargo rustc -q -p pixelflow-compiler --test <bin> -- -Zunpretty=expanded | sha256sum` for each of the twelve
`pixelflow-compiler/tests/*.rs` binaries, `-p pixelflow-graphics --lib`, and its `--test optimization_fuzz`: fourteen hashes, equal. `-Zunpretty` prints
tokens without spans, so that is exactly "byte-identical token streams"; refusal texts and spans are `tests/rustc_is_the_oracle.rs`'s. Not a gate:
`-W unreachable_pub` is already clean at bab3441 and does not fire on a `pub` field of a `pub(crate)` struct. Two greps are the reviewer's check per CL,
review-grade until a script lands (§4): `rg -n '^\s*pub (fn|struct|enum|trait|const|type|[a-z_]+:)' pixelflow-compiler/src` lists exactly the two
`#[proc_macro] pub fn`s; `rg -n 'pub\(crate\)' pixelflow-compiler/src --glob '!**/mod.rs' --glob '!lib.rs'` lists only items some `mod.rs` re-exports.

1. **`symbol` dissolves.** The one CL that edits bodies: `Symbol`, `SymbolKind` and the `symbol.rs:6-15` table into sema; `FnAnalyzer.symbols: Scopes<Symbol>`
   bound at the sites §1.3 names; `Name::coordinate`/`coordinates` and `COORDINATES` onto `Name`, `Name::spelled` private; `lower.rs:856` and `sema.rs:406`
   re-pointed; the file keeps `Scopes` and its one test; four tests deleted. The fingerprint and `cargo test` are the behavior pin.
2. **`ast/` and `src/tests/`.** `git mv src/ast.rs src/ast/mod.rs`; carve `def`, `types`, `name`, `expr`; `git mv src/symbol.rs src/ast/scopes.rs` (a rename
   above 50% once CL 1 has shrunk it); `PLAN` in from `lib.rs`, the three `use crate::PLAN` lines re-pointed (`parser.rs:194`, `sema.rs:71`, `lower.rs:61`);
   `from_syn` out to `parser.rs:1649, 1686`; 71 fields `pub(crate)`; `src/tests/{mod,ast,parser}.rs` with ast's and symbol's tests as §1.5 says; `ast/intent.md`; `lib.rs:7-24` names the directory.
3. **`parser/`.** `git mv`; carve `items`, `expr`, `fold`, `block`; the 50 tests to `tests/parser.rs`, two spelling `1024`; `parser/intent.md`; `.claude/skills/ontology/language.md:74`; crate `intent.md:55`.
4. **`sema/`.** `git mv`; carve `analyze`, `body`, `consts`, `calls`; `Ty` `pub(super)`; 6 fields `pub(crate)`; 57 tests to `tests/sema.rs`, two deleted, the
   sweep to `tests/front_end.rs` rewritten; `sema/intent.md`; `docs/language/2026-10-17-the-arena-is-typed.md:33, 161` (the only `sema.rs` in the tree, so the
   basename fallback cannot save them once it is renamed); `language.md:590`; crate `intent.md:53-55`.
5. **`lower/`.** `git mv`; carve `site`, `walk`; `Site` defined in `mod.rs`; `Lowered::range_of`, `holes` private, and the `entry_arena`/`EntryContext` shape of
   §1.2 in emit — the CL's one body edit; 3 fields `pub(crate)`; one test to `tests/lower.rs`, two deleted; `lower/intent.md`; `language.md:162, 373`; `compiler.md:85`.
6. **`emit/`.** `git mv`; carve `host`, `arena`, `staged` (`emitted!` crosses as `pub(super) use emitted;`); three tests to `tests/front_end.rs`; `emit/intent.md`;
   `the-arena-is-typed.md:155` (`compiler/emit.rs:509`, a basename gone after the rename); `language.md:109, 263`; `compiler.md:85`; crate `intent.md:54`.
7. **Docs only** (skips the build jobs, `scripts/ci-change-scope.sh`): `pixelflow-compiler/intent.md` Contract points at the five `mod.rs` and lists their
   entries, Behind gains the two lines of §4; `docs/process/2026-10-15-intent-lives-with-the-module.md:74` drops this crate from its flat-file example; this
   note's Status becomes `Plan of record`, every path it names now existing.

Order is forced at two points only: CL 2 needs CL 1 (the `scopes.rs` rename), CL 6 needs CL 5 (`entry_arena`). The module paths `crate::{ast, parser, sema, lower, emit}` are the same for a file and a directory, so CLs 3–6 would resolve in any order after CL 2; they are stacked linearly so every rebase is trivial.

## 4. What this is not (Q8)

- Not the IR-bound models: `Holes`, `OPEN_RANGE`, `HOLE_STRIDE`, `Param(u8)` (`lower.rs:394-410`) and `u32` fold ends (`sema.rs:1620`) are hidden in `lower/site.rs` and `sema/consts.rs`, not replaced (the-language-is-kernel D2, A5).
- Not the typed arena: `Ty::Bool`/`Ty::F32`, `MethodTyping`, `term_type` stay in sema, now `pub(super)`; T4 deletes the enforcement inside `sema/`, and no
  module exists for the mask/scalar distinction to be deleted with it (`docs/language/2026-10-17-the-arena-is-typed.md` §2, §4).
- Not the tier deletion: `macro_tier`, `DwrtFree`, `kernel_raw!` stay in `lib.rs` (D2, D14; one-pipeline M1–M3).
- Not a change to what the macros accept, emit or refuse: no `quote!` body, refusal text or span moves; the differential is the proof, and the as-if contract
  (`docs/process/2026-10-13-tests-check-the-as-if-contract.md`) is why the eight deleted tests cost no coverage.
- Not the two subtractions the map exposes, since both change a signature: sema computes a fold's `Bounds` at `sema.rs:888` and discards it, so lowering
  recomputes at `lower.rs:648` — `range_bounds` and `RangeScope` are on sema's contract only because the result is not stored on the analyzed fold (sema's
  `intent.md` Behind); and `node` (`emit.rs:526`) and `Staged` (`:727`) are two spellings of "an IR call as the statement that makes it", which replaying an
  arena into a `Site` would reduce to one.
- Not the dead arms the deleted tests watched: `resolve()`'s coordinate-in-a-helper and binding-before-coordinate `Err` arms (`lower.rs:833-865`)
  are reachable from no program sema admits, which is why no test can drive them through a contract. They stay this series (a body edit) and are the
  kind of arm the desloppify pass turned into `expect`s naming the sema refusal; the next pass does the same here.
- Not a new CI check. The test law is held by the compiler (a crate-root test cannot name a `pub(super)` item); the two greps of §3 by review and by desloppify's
  `mod-rs-is-a-contract`, `interface-lives-in-mod-rs` and `tests-target-the-contract` rules reading each `intent.md`. CLAUDE.md says a gap in CI is a check to
  write; the ruling says this series adds `mod.rs` and `intent.md` and nothing else. So the check — the first grep as a step of `scripts/check-intent.sh`, over
  every crate whose `mod.rs` directories carry an `intent.md` — is named in `pixelflow-compiler/intent.md` Behind, dated, as the CL that follows.
