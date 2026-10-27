# desloppify/src/rule

**Is.** A rule as a typed decision: a question about one unit of code, with a
closed set of outcomes, each fine or a violation, and the model levels that
answer and explain it (desloppify/README.md, "Rules"). A rule file is read
once into a `Rule` that is already checked, so a review never meets a
malformed one halfway through. desloppify is not a pixelflow crate; DENOTATION
does not apply here.

## Owns
- The shape of a rule file (`RuleFile`) and every refusal of a bad one: an
  unknown field, a level outside 1-4, `explain` below `decide`, no fine or no
  violation outcome, an outcome named `unsure` or named twice, an unknown
  skill, a bad glob, a `group` on a whole-file unit, a query with no `@target`.
- The closed set of outcomes (`Decision`, `Outcome`, `Verdict`); `UNSURE` is
  every decision's own and no file may name it.
- What a call can see: `Unit`, `Part`, `Group`, `Surroundings` and the
  compiled queries. tree-sitter finds units here; it does not judge them.
- Which paths a rule reads (`Files::contains`).

## Never
- No rule is half-loaded: `load_dir` returns every rule or the first failure.
- Nothing here calls a model, builds a prompt or reads code under review; it
  imports only `language`, `model` and `skills`, and reads only rule files.
- A `Rule`'s fields other than `id` and `levels` are `pub(crate)`.
- A rule's judgment lives in its file: no shipped rule id is in `src/`.

## Contract
`mod.rs` holds the types and one entry; `file.rs` reads, `files.rs` filters.
Public: `load_dir`, `Rule`, `Levels`. Crate only:
`Decision`, `Outcome`, `Verdict`, `Files`, `Unit`, `CompiledQuery`, `Part`,
`Group`, `Surroundings`, `TARGET_CAPTURE`, `CUT_CAPTURE`, `UNSURE`; `review/`
and `snippet/` read them.

## Enforced by
- Types: `ModelLevel` deserializes through `TryFrom<u64>`, 1-4 only;
  `RuleFile` and `Levels` carry `deny_unknown_fields`; `Verdict`, `Unit`,
  `Part`, `Group`, `Surroundings` are closed enums.
- `desloppify/tests/rule.rs`: one test per refusal (levels, no violations,
  `unsure`, a repeated outcome, `group` on a whole file, an unknown field or
  skill) and the id and levels of a good rule.
- `desloppify/tests/shipped_rules.rs`: every rule in `desloppify/rules/`
  loads, and each shows a model the unit it claims. Both run in
  `cargo test --workspace`.

## Behind
- `Decision::outcomes` says "each in file order"; `fine` and `violations` are
  `BTreeMap`s, so the order is by outcome name. No plan or Linear issue.
- An outcome named twice is refused across `fine` and `violations` only;
  twice inside one object, serde_json keeps the last and the rule loads. No
  plan or Linear issue.
- `"context": "module_root"` with `"group": "crate"` loads, and the planner
  gives the crate call no root (`review/plan.rs`), so the context is dropped
  silently. No plan or Linear issue.
- `Surroundings::ModuleIntent` stops at the crate's `Cargo.toml`; a module
  whose intent lives in a parent crate gets none, and nothing says which
  crate a nested `Cargo.toml` makes the boundary.

## Decided in
`desloppify/README.md` "Rules"; no plan decides the rule shape.
