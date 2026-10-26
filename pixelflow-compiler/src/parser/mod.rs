//! # Parser
//!
//! Parses the kernel DSL from token stream to AST.
//!
//! ## Grammar
//!
//! A `kernel!` is a block of items, parsed by syn as a `File`, or one closure
//! entry. A body is parsed by syn as a Rust
//! expression, so precedence and associativity are Rust's. Of that syntax,
//! this is the kernel language — what every stage accepts:
//!
//! ```text
//! kernel  ::= item*                          -- the items form
//!           | '|' params '|' expr            -- the closure form: one entry, its type inferred
//! item    ::= VIS? 'struct' RECORD '{' (field (',' field)* ','?)? '}'
//!                                            -- a record of `f32` fields
//!           | VIS? 'const' IDENT ':' 'f32' '=' cexpr ';'
//!           | VIS? 'const' IDENT ':' 'usize' '=' iexpr ';'
//!           | VIS 'fn' IDENT '(' params ')' '->' type block
//!                                            -- an entry
//!           | 'fn' IDENT '(' params ')' '->' type block
//!                                            -- a helper: private, no generics
//! field   ::= VIS? IDENT ':' 'f32'
//! params  ::= (param (',' param)* ','?)?
//! param   ::= IDENT ':' type
//! type    ::= 'f32' | 'bool' | RECORD        -- an entry's `f32`s and records
//!                                            -- are its uniforms
//!           | 'impl' 'Fn' '(' 'f32' ',' 'f32' ')' '->' 'f32'
//!                                            -- a kernel: an argument the host
//!                                            -- passes at run time (Phase D-a)
//!
//! cexpr   ::= cexpr ('+' | '-' | '*' | '/') cexpr   -- an `f32` const's initializer,
//!           | '-' cexpr | '(' cexpr ')'            -- evaluated at expansion,
//!           | IDENT 'as' 'f32'                     -- per operation in f32
//!           | IDENT | LITERAL
//! iexpr   ::= iexpr ('+' | '-' | '*' | '/') iexpr   -- a `usize`: a range's bounds
//!           | '(' iexpr ')'                        -- or a `usize` const's
//!           | IDENT | INTEGER                      -- initializer; each operation
//!                                                  -- checked as rustc checks it,
//!                                                  -- at expansion
//!
//! expr    ::= expr binop expr
//!           | '-' expr
//!           | expr '.' METHOD '(' (expr (',' expr)*)? ')'
//!           | expr '.' IDENT                       -- a record's field
//!           | PROJECTION '(' expr ')'
//!           | IDENT '(' (expr (',' expr)*)? ')'    -- a helper, inlined
//!           | KERNEL '(' expr ',' expr ')'         -- a kernel applied at
//!                                                  -- `(x, y)`: contramap
//!           | 'if' expr block 'else' (block | 'if' …)   -- the choice
//!           | fold
//!           | IDENT 'as' 'f32'         -- a `usize` (a fold's index, a `usize`
//!                                      -- const or an entry count parameter) as
//!                                      -- a value
//!           | '(' expr ')'
//!           | block
//!           | IDENT                    -- X, Y (in an entry), a parameter, an
//!                                      -- `f32` const, or a `let` in scope
//!           | LITERAL                  -- an integer or a float, as its f32
//! fold    ::= range '.map' '(' binder expr ')' '.sum' F32? '(' ')'       -- Σ
//!           | range '.map' '(' binder expr ')' '.product' F32? '(' ')'   -- Π
//!           | range '.map' '(' binder expr ')'
//!                 '.fold' '(' 'f32::INFINITY' ',' 'f32::min' ')'         -- min
//!           | range '.map' '(' binder expr ')'
//!                 '.fold' '(' 'f32::NEG_INFINITY' ',' 'f32::max' ')'     -- max
//!           | range '.any' '(' binder expr ')'                           -- ∃, of bools
//!           | range '.all' '(' binder expr ')'                           -- ∀, of bools
//! range   ::= '(' iexpr '..' iexpr ')'     -- half-open, constant, forwards
//! binder  ::= '|' IDENT '|'                -- the index: a `usize`
//! F32     ::= '::' '<' 'f32' '>'
//! binop   ::= '+' | '-' | '*' | '/'
//!           | '<' | '<=' | '>' | '>=' | '==' | '!='    -- a comparison: a bool
//!           | '&' | '|'                                -- bools combine
//! block   ::= '{' stmt* expr '}'
//! stmt    ::= 'let' IDENT (':' type)? '=' expr ';'
//!           | 'let' '(' IDENT (',' IDENT)* ','? ')' (':' '(' type (',' type)* ')')?
//!                 '=' '(' expr (',' expr)* ','? ')' ';'
//!                                          -- as many names as expressions, each
//!                                          -- bound to its own, all at once
//!           | expr ';'
//!
//! VIS        -- `pub`, `pub(crate)`, …: a visibility, kept on the host item
//! RECORD     -- the name of one of the block's records
//! METHOD     -- an `OpKind` method, a `LibraryMethod` composition, or `clone`
//! PROJECTION -- V, DX, DY, DXX, DXY, DYY
//! KERNEL     -- a kernel-typed parameter in scope
//! ```
//!
//! A `let` is scoped as Rust scopes it ([`crate::ast::Scopes`]), and so is a fold's
//! index: its body sees the enclosing bindings, an enclosing fold's index
//! among them. Nothing shadows `X`, `Y`, a `const`
//! or a `fn`: a parameter, a `let` or an index of that name is refused. `X`
//! and `Y` appear only in an entry; a helper takes its coordinates as
//! arguments
//! (docs/language/2026-09-25-the-language-is-kernel.md §1.2). A fold
//! (plan §1.5) denotes ⊕ of its body over `i ∈ [a, b)`, and its monoid's
//! identity when the range is empty; the syntax never unrolls it.
//!
//! Binding times (plan §1.4). Every scalar or count parameter is a
//! **uniform**: an `f32` or `usize` is one, a record is
//! one per field, and the host function's arguments are their values for
//! that call — never constants folded in. A record is written by name: a
//! parameter, a `let` alias of one (`let q = p;`), an argument passed on, or
//! the base of a field read.
//!
//! The language has no collection types (plan §1.3, §1.6): no array, slice,
//! list or table, as a parameter, a value or a thing to iterate. Several
//! values are several parameters, a record is its named fields, and the one
//! iteration is a fold, over a range.
//!
//! A tuple is taken apart where it is written, and only there:
//! `let (a, b) = (e1, e2);` binds each name to its expression, every
//! expression read before any name binds, as Rust's does (D7's front half).
//!
//! A record and its fields keep their attributes, re-emitted on the host
//! struct; every other item keeps only its doc comments.
//!
//! Refused here, with a span, at the token: an item that is not a `struct`, a
//! `const` or a `fn`; a record that is generic, a tuple struct or a unit
//! struct, and a `repr` or a `cfg` on a record or its field (§1.3); an
//! attribute other than a doc comment on a `const` or a `fn`; generics on
//! any item; a function, spelled any way, as a
//! parameter of the closure form (rustc's E0562 of an `impl` type: an entry
//! of the items form takes a kernel);
//! a `fn` without a declared return type; an `if` without an
//! `else` or an `if let`; `loop`/`while`/`for`; assignment; `return`; a
//! closure anywhere but a fold's; a tuple
//! anywhere but a tuple `let`'s value, and a tuple's field (a tuple value is
//! Phase D, D7); a record literal (Phase D, D7); every spelling of a
//! collection — a parameter typed as an array or a slice, an array literal,
//! indexing and slicing, `.into_iter()` and `.iter()`, and `.map`, `.any` or
//! `.all` over anything but a range (no collection types, §1.3, §1.6); a
//! range anywhere but a fold's; an inclusive or open-ended range; a method
//! on a range or a mapped range that is not one of the fold's spellings;
//! a `.fold` whose arguments are not one of the two above; a fold's
//! closure with a type annotation, a pattern, more than one
//! parameter, a return type or a qualifier; type arguments on a method
//! (`::<f32>` on `.sum` and `.product` excepted); an `as` to any type but
//! `f32`; a path or a call from outside the block; a method the language
//! does not have, a known one at the wrong arity, and the `Kernel` methods
//! `.at()`, `.constant()` and `.collapse()`; `%` and `!` (no IR op);
//! a `let` whose pattern is neither a plain name nor a tuple of them (`mut`,
//! `ref`, `@`, a record's pattern); a tuple `let` whose value is not a tuple
//! of as many expressions, with a pattern within its pattern, a `_` or a
//! `..`, a name bound twice, no names, or an annotation that is not a tuple
//! of as many types; a `let` without an initializer; `let … else`; an
//! item or a macro inside a block; an operator not in the table above; a
//! literal that is not a number; a literal suffixed with a type other than
//! `f32`; an integer past `u128`; a float past `f32`'s range; and any other
//! Rust expression syntax, named in the refusal. Nothing is passed through
//! for a later stage to refuse: the AST holds only what the language means.
//!
//! Parsed, and refused by `sema`: an unbound name, `Z` and `W` among them; a coordinate in
//! a helper; a call to an entry or to an unknown function; recursion; a
//! type error (every expression is an `f32` or a `bool`; a `usize` is used only as `i as f32`; a record only by name);
//! a record's field that is not an `f32` (§1.3); a
//! record returned, built, chosen by an `if`, or used in arithmetic or a
//! comparison (Phase D, D7); an unknown field; an `as` of anything but a
//! `usize` name; an integer an `f32` does not hold exactly where a value is
//! expected; a `const` whose initializer is not a `cexpr` or an `iexpr`; a
//! negated `usize`; a range whose bounds are not constant or that runs
//! backwards; a block with no final expression; an
//! `Args` record's name taken twice; a function type spelled any way but
//! `impl Fn(f32, f32) -> f32`; a kernel used as anything but applied, with
//! two `f32`s, or passed by name to a helper's kernel-typed parameter — in
//! arithmetic, a comparison, a `let`, an `if`'s arm, a record's field, a
//! return, `as` (Phase D-a); a kernel passed on twice on one path, or used
//! after it is passed on on that path, or passed on inside a fold's body —
//! what rustc refuses of an `impl Fn` it moves (E0382, E0507); and a
//! parameter, `let` or index named after a projection; and a projection called
//! with other than one argument (`type_of_call`: the parser leaves the name a
//! call, so the refusal says what to rename).
//!
//! Refused by `sema`, where the language meets the IR's widths: a range
//! bound past 2²⁴, the last integer bound a fold's index — an `f32` lane —
//! names exactly, and folds nested deeper than the IR has binders (lowering
//! refuses the same depth where an inlined helper's fold adds to it). A range
//! over a per-call count is guarded by the IR; the host entry and `Args`
//! refuse counts past 2²⁴.
//!
//! ## Implementation Note
//!
//! We use syn to parse into its Expr types first, then convert to our AST.
//! This gives us Rust's expression parsing for free while maintaining our
//! own semantic layer.
//!
//! # Contract
//!
//! One function, the stage's: tokens in, the vocabulary's `KernelDef` out, or
//! the refusal at the token that spells nothing. `parse` is `pub(crate)` in
//! `items.rs` and re-exported here; every other item under `parser/` is the
//! implementation and is `pub(super)` or private.
//!
//! ```ignore
//! /// Implemented by `parser::parse`.
//! trait Parse {
//!     /// The `KernelDef` the tokens spell: `|params| expr` is `Closure`; a `syn::File` of
//!     /// `struct`, `const` and `fn` items is `Items`, lists in declaration order. Total over
//!     /// the module doc's grammar; every other token is `Err` at that token with the listed
//!     /// text, and nothing is carried along for a later stage to refuse. A method is resolved
//!     /// once (`Method::resolve`), a projection by name at arity one, a float rounded once,
//!     /// an integer kept exact. A body chained deeper than 1,024 terms is `Err` at the
//!     /// chain's head token, in O(1), on rustc's 8 MiB stack.
//!     fn parse(input: proc_macro2::TokenStream) -> syn::Result<KernelDef>;
//! }
//! ```

mod block;
mod expr;
mod fold;
mod items;

pub(crate) use items::parse;

/// The deepest an expression may nest: a body of `MAX_NESTING` terms in a
/// chain (`a + b + …`, `x.f().g()…`, a long `else if`) converts, one more is
/// refused. Every level costs the stack of [`expr::convert_expr`] and its arm, and
/// then of `sema` and lowering below it, which walk no deeper an AST than
/// this one.
///
/// Measured in a dev build of `kernel_raw!` on rustc's 8 MiB thread stack,
/// the whole pipeline overflows at about 2,058 terms of a sum, 1,736 of a
/// method chain, 2,219 of unary minus and 1,836 of an `else if` chain. The
/// bound is half the smallest (868) rounded to a power of two, 1,024: the
/// stack keeps a margin of 1.7x over the worst shape, for a host whose frames
/// are somewhat larger. A host with a smaller stack than rustc's default can
/// still overflow; that is not a bound this crate can set.
///
/// The bound covers only the shapes `syn` parses flat (a sum, a
/// method chain, `as`, a field, `else if`, and a run of prefix operators,
/// `----X`), which arrive here whole. It does **not** cover a nested
/// delimiter: `syn` recurses by itself on one before this parser sees a node,
/// and it overflows well below `MAX_NESTING`. Measured on `syn::parse2::<Expr>`
/// alone, in a dev build on rustc's 8 MiB thread stack, tokens built
/// iteratively so no `from_str` recursion is involved: `if c { if c { … } … }`
/// holds to about 400 levels, `{{…}}` to about 480, `-(-(…))` to about 590,
/// `f(f(…))` and `X.max(X.max(…))` to about 650, `((…))` to about 770, while a
/// run of prefix operators holds to about 2,390. So a program nested 500 to
/// 800 deep in a delimiter still crashes the compiler with no diagnostic, and
/// no bound in this crate can reach it: a refusal there needs a token pre-scan
/// that restates `syn`'s grammar, or a stack the compiler does not own.
const MAX_NESTING: usize = 1024;
