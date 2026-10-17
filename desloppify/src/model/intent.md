# desloppify/src/model

**Is.** The model ladder (desloppify/README.md, "Backends"): a total function
from (`Provider`, `ModelLevel`) to the name of the model that answers a
question that hard. A rule says how hard its question is, 1 to 4 (`Lite`,
`Fast`, `Strong`, `Frontier`); the provider says which model that is.

## Owns
- The four levels and their order. `ModelLevel` is the only place a level is
  a number (1–4, `try_from = "u64"`); everywhere else it is the enum.
- Every model id the language-model backends send: the eight arms of
  `Provider::model`, Anthropic and Gemini.
- "One level higher" (`ModelLevel::above`, `None` at `Frontier`): how an
  `unsure` answer escalates.

## Never
- No model id is spelled outside `Provider::model`: no rule, skill, prompt or
  backend names a model, only a level.
- No level is chosen here: the levels a rule asks at (`Levels`) and a
  naming or synthesis asks at belong to `rule/` and `review/`.
- No knowledge of how a backend runs a model: thinking effort, rate limits,
  credentials, retries and tokens are `agent/` and `rate_limit/`.
- No fallback. `Provider::model` has no wildcard arm, so a new level or
  provider is a compile error here, not a default model elsewhere.
- No System One: Jev's model (`TYPESAFE_DEFAULT_MODEL`) is `decide/`'s.

## Contract
`mod.rs` is the whole module. Public: `ModelLevel`, `LevelOutOfRange`,
`Provider` (`Anthropic`, `Gemini`). `pub(crate)`: `ModelLevel::above` (the
`unsure` ladder in `review/run.rs`) and `Provider::model` (`agent/rig.rs`, and
`agent/claude_code.rs` as `Provider::Anthropic`). The README "Backends" table
is the human reading of `Provider::model`.

## Enforced by
- Types: `ModelLevel` is a closed enum, so level 0 or 5 is unrepresentable
  past parsing, where `LevelOutOfRange` refuses it; the `match` in
  `Provider::model` is exhaustive over both.
- `desloppify/tests/rule.rs`: `a_level_outside_one_to_four_is_refused` and
  `an_explain_level_below_the_decide_level_is_refused` (the derived `Ord` is
  what `explain >= decide` compares, in `rule/file.rs`).
- `desloppify/tests/review.rs`: an `unsure` answer climbs Lite, Fast, Strong
  and stops at the rule's explain level; things are named and described at
  Fast, and the one synthesis is asked at Frontier.

## Behind
- No test pins the ladder: nothing fails if an id in `Provider::model` is
  misspelled or retired (three Gemini ids are `-preview` names), so the first
  real call finds out. No plan or BACKLOG item covers it.
- `Provider` derives `clap::ValueEnum`, but `main.rs` selects a backend
  through its own `Backend` enum and never parses a `Provider`; the derive is
  unused.
- A level's meaning is split across two files: which model is here, how hard
  it thinks (`think` in `agent/claude_code.rs`: off at 1–2, medium at 3, high
  at 4) is there, and the API backends set neither.

## Decided in
desloppify/README.md, "How a review runs" (the `unsure` ladder, levels 3 and 4
for diagnosis) and "Backends" (the table); `d835dd48` (Lite is Haiku 5.5).
No plan decides the ladder.
