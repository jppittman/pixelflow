---
name: style-guide
description: The repository's code style (docs/STYLE.md and CLAUDE.md "Code Style"), distilled.
---

# Style guide

- **Clarity over comments.** If code needs a comment to explain *what* it does,
  refactor it; a helpful "what" comment is a helper function's rustdoc waiting
  to be extracted. Rustdoc (`///`) documents the public contract; `//` explains
  *why* the current code is the way it is. No history in comments ("previously",
  "changed to", "updated to") and no commented-out code — that is version
  control's job.
- **Guard clauses and early returns.** The number of open cases — unhandled
  inputs, unchecked invariants — should only go down as you read down a
  function. The top is the hardest part to understand, because everything is
  still possible; each guard discharges one case and exits, so by the bottom
  exactly one case is left and what to do is obvious. `let ... else`, `?`, and
  `return` delete a case; nothing rejoins, so no `else` accumulates and the
  code doesn't drift rightward.
- **Parse, don't poke at strings.** Splitting, trimming, prefix-stripping and
  `parse::<T>()` on structured text is a parser, whether or not it is called
  one. Parse once, at the boundary, into a type — a `serde` struct over a real
  format (`toml`, `json`), a `FromStr` enum, or a grammar — and let the rest of
  the code match on the type, never on the string.
- **Fold before you dispatch.** A fold leaves fewer live cases than it found
  (`if x > 0.0 { x = -x }` needs no `else`). Dispatch keeps every case alive for
  everything downstream. Collapse cases wherever they collapse; use `match` for
  the ones that genuinely cannot, and prefer `match` over `else if` chains.
  `If(m, a, b)` in the kernel language is dispatch, not a fold.
- **Functions take fewer than 4 arguments**; group related ones into a struct.
- **No boolean arguments**; an enum or two functions says what the call means.
- **Named constants**, not unexplained literals other than 0, 1, 2.
- **Name vs namespace.** A function name stacking concepts
  (`compile_arena_dag_jet`) or an accreting family (`foo`, `foo_with_ctx`,
  `foo_scanline`) wants to be a module, a method, or a builder.
- **New implementation of an existing category → trait first.** A second way of
  doing something is a second `impl`, not a parallel free function, a copy, or a
  mode flag. Dispatch once at construction, not at every use; a `Box<dyn>` pays
  dispatch per call and on a hot path is usually better as an enum or a generic.
- **Errors are handled.** No silent failures: no swallowed `Result`, no default
  standing in for a failure the caller needed to know about. Fail loud.
- **Tests** test public behaviour, and a test's name reads as a sentence after
  "it should".
