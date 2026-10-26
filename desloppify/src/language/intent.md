# desloppify/src/language

**Is.** The table from a source language to the tree-sitter grammar and the
queries that cut its files into units (desloppify/README.md, "How a review
runs", step 1 and "Rules"). It finds code and judges nothing: a query
returns the text a rule will be asked about, never whether that text is
right. desloppify is not a pixelflow crate: this cites its README, not DENOTATION.

## Owns
- Which files are reviewable at all: `Language::of` decides by file
  extension, and a path it answers `None` for is skipped by the planner.
- The grammar for each language (`Language::grammar`).
- What each unit kind means in each language (`Language::query` for
  `Part::Functions`, `Part::FunctionSignatures` and `Part::Types`), where a
  function body is (`Language::bodies`, what an outline elides), and what a
  test module is (`Language::is_test_module`, whose contents are never shown).

## Never
- No outcome, verdict, rule, model or System One appears here: the module
  imports only `crate::rule::Part`, and a query's result is text and a line.
- No file is read here. `Language::of` looks at a path's extension and
  nothing else; the caller reads and parses (`snippet/capture.rs`).
- No query names an identifier, a path or a crate, only a syntactic shape.
- A query's captures are `@target` and, where a part is cut short, `@cut`
  (`TARGET_CAPTURE`, `CUT_CAPTURE` in `rule/mod.rs`). Nothing else is read.
- No language is added by a mode flag on an existing one: a new language is a
  new `Language` variant and an arm in each method, which the compiler
  refuses to leave out.

## Contract
`mod.rs` is the whole module: the enum `Language` (`Rust`), `Language::ALL`,
and `of`, `grammar`, `query`, `bodies`, `is_test_module`. The module is
crate-private (`mod language;` in `lib.rs`). Callers: `rule/file.rs`
(compiles every language's query for a rule's unit when the rule loads),
`snippet/` (cuts a file), `review/plan.rs` and `review/module.rs` (pick the
language of a file).

## Enforced by
- `match` over a closed enum with no wildcard arm: a language missing a
  query for a `Part`, or a body query, does not compile.
- A malformed query fails when the rules load, before any call:
  `rule/file.rs` compiles each one and requires `@target`
  (`desloppify/tests/shipped_rules.rs`, `every_shipped_rule_loads`).
- The shapes the queries cut, through the production API, in
  `desloppify/tests/shipped_rules.rs`: signatures shown without bodies, types
  each alone, functions whole, functions inside a `#[cfg(test)]` module never
  shown, an outline keeping every line but function bodies.

## Behind
- `is_test_module` matches the text `cfg(test)` in the module's attributes
  with spaces removed, so `#[cfg(all(test, ...))]` is not recognised as a
  test module and its functions are reviewed (2026-10-15; no plan or
  BACKLOG item names it, and no test covers it).

## Decided in
`desloppify/README.md`, "Rules" ("Units are per-language tree-sitter queries
in `Language::query` ... tree-sitter only finds units; it judges nothing")
and "Layout" (every module's `mod.rs` is its contract). No plan in
`docs/plans/` names this module.
