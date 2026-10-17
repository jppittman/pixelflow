# desloppify/src/snippet

**Is.** The cut from a file to the units a rule asks about
(desloppify/README.md, "How a review runs", step 1, and "Rules"): a function
from (rule, language, source text) to the numbered text of each unit, in
source order. It finds code and judges nothing. desloppify is not a pixelflow
crate: this cites its README, not DENOTATION.

## Owns
- What one call sees of a file, per `Unit`: the whole file (`File`), the file
  with every function body elided (`Outline`), or the parts a query captures
  (`Parts`), each alone (`Group::Each`) or joined into one snippet
  (`Group::File`, `Group::Crate`).
- The line-number prefix (`Snippet::numbered`), the only way the model can
  point at a line, and `Snippet::first_line`. An elided body (`ELIDED`) leaves
  every other line at its own number; joined parts are split by `GAP`.
- That tests are not units: a part inside a test module is dropped.

## Never
- No verdict, outcome, rule id, model, level or System One: the module imports
  only `Rule`, `Unit`, `Group`, `CompiledQuery` and `TARGET_CAPTURE` from
  `rule` and `Language` from `language`.
- No file is read: both entries take `source: &str`; `review/plan.rs` reads.
- No query is written here: queries and the test-module test live in
  `language/`; this module reads `@target` and `@cut` and nothing else.
- No crate is joined here: a `Group::Crate` snippet is one file's parts, and
  `review/plan.rs` puts files under `== path ==` headers.
- No silent empty: a grammar or source that will not parse is `Err`; only a
  language with no query for the part gives no snippet.

## Contract
`mod.rs`: `Snippet` (`first_line`, `numbered`), `snippets` (a rule's units
in a file) and `outline` (the file with bodies elided); `capture.rs` is the
implementation, all `pub(super)`. Callers: `review/plan.rs` (`snippets`, and
`outline` for a module root) and `review/identify.rs` (`outline`).

## Enforced by
- Visibility: `capture.rs` exports `pub(super)` only and `mod snippet;` is not
  `pub`, so nothing outside `snippet/` reaches the cut.
- `rule/file.rs` (`compile`): a part's query with no `@target` fails when the
  rules load, before any cut.
- `desloppify/tests/shipped_rules.rs`, through the production planner: each
  unit kind's shape (`types_are_shown_each_alone`,
  `functions_are_shown_each_alone_and_whole`), signatures joined with `GAP`,
  `functions_inside_a_cfg_test_module_are_never_shown`, and
  `an_outline_keeps_every_line_but_function_bodies_at_its_own_number`.
- `desloppify/tests/review.rs`: `a_crate_wide_rule_makes_one_call_per_crate_and_findings_keep_their_file`.

## Behind
- `Outline` does not skip test modules: a `#[cfg(test)]` module's signatures
  are in the outline, though `Parts` drops its functions (2026-10-15; no plan
  or BACKLOG item names it, and no test covers it).
- `Language::is_test_module` misses `#[cfg(all(test, ...))]`; the cut inherits
  it (`language/intent.md`, Behind).

## Decided in
`desloppify/README.md`: "How a review runs" (tree-sitter cuts the code into
the units a rule asks about), "Rules" (`unit`, `group`) and "Layout" (every
module's `mod.rs` is its contract). No plan in `docs/plans/` names this module.
