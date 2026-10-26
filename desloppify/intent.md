# desloppify

**Is.** Rule-driven code review: a function from (rules, files) to a `Report`
(desloppify/README.md, not DENOTATION: this is not a pixelflow crate). A rule is
a typed decision; tree-sitter cuts the units, a cheap decider answers, a model
explains only the violations, small models name what converging findings are
about, and one frontier call diagnoses them as symptoms of a missing model and
writes the review.

## Owns
- The shipped rules (`rules/*.json`, 27) and the skills they pull in
  (`skills/<name>/SKILL.md`): what this repository asks a reviewer.
- The pipeline's shape: plan, decide, explain, identify, synthesize;
  `main.rs` only wires it from flags.
- The exit status: failure on any finding or undecided unit.
- Its own layout: every module is a directory whose `mod.rs` is the contract,
  every other file is `pub(super)`, all tests are in `tests/` and drive the
  API `main` uses.

## Never
- tree-sitter never judges. It is imported only by `language/`, `snippet/`
  and `rule/`, and finds units; every verdict is a decider's or a model's.
- No model id is spelled outside `model/` (`Provider::model`): no rule, skill
  or backend names one. Jev's `jev-latest` is System One's, in `decide/`.
- A module that is handed source never reads a file: `language/` and
  `snippet/` import no `std::fs`; only `skills`, `rule`, `review/plan.rs`,
  `review/module.rs`, `agent/journal.rs`, `agent/claude_code.rs` (its scratch
  directory) and `main.rs` touch the disk.
- No rule's id or outcome is matched in `src/`; judgment lives in the JSON.
- No `#[test]` or `#[cfg(test)]` module in `src/`; tests live in `tests/`.

## Contract
`src/lib.rs` declares nine modules: `agent`, `decide`, `model`, `rate_limit`,
`review`, `rule`, `skills` public; `language` and `snippet` private. Each has
its own `intent.md`. The CLI is `src/main.rs` (`Args`): paths to review;
`--backend` (`anthropic`, `gemini`, `claude-code`, `dry-run`), `--system-one`
(`none`, `jev`, `dry-run`), `--limiter`, `--min-confidence`, `--jobs`,
`--findings-only`, `--out`, `--rules`, `--skills`; it calls `skills::load`,
`rule::load_dir`, `review::plan` and a `review::Session`'s `review`,
`identify`, `synthesize`; `--min-confidence` is 0 to 1.

## Enforced by
- `desloppify/tests/shipped_rules.rs`: every rule in `rules/` loads, and each
  shows a model the unit it claims, read from the prompts a review sends.
- `desloppify/tests/rule.rs`: the loader refuses each kind of bad rule.
- `tests/review.rs`, `agent.rs`, `decide.rs`, `rate_limit.rs`: the rest, via
  the production API. `RuleFile` carries `deny_unknown_fields` (a type).
- `desloppify/tests/transcript.rs`: every prompt, schema, level and reported
  datum of a scripted review is held to `tests/golden/`, so an edit to one is
  a visible diff and journals keyed on prompts stay replayable.
- `.github/workflows/rust.yaml` runs these with the workspace. Nothing checks
  the import and `fs` lines under Never; `grep` does today.

## Behind
- `skills/pixelflow-architecture` and `skills/style-guide` are hand-kept
  distillations of CLAUDE.md and STYLE.md, a tenth of their length, kept so
  by choice: the whole of either in every rule prompt would cost more than
  it buys (plan §5, as amended). Only `skills/ontology` is a symlink.
- Nothing in CI runs desloppify over the tree; only its own tests run.

## Decided in
`desloppify/README.md`; `docs/process/2026-10-15-intent-lives-with-the-module.md`
(§4, §5). No other plan.
