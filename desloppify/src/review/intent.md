# desloppify/src/review

**Is.** The review pipeline of a rule-driven code reviewer: a function from
(rules, files) to a `Report` in five stages, plan, decide, explain, identify,
synthesize. Every model reply is a value of a closed schema or a failure;
small models make many calls, and the one frontier call sees everything.

## Owns
- The plan: which rules ask about which unit of which file, and which share a
  call (same unit and context; a crate-wide rule gets one call per crate).
- The cascade: System One first, a model at the rule's decide level below the
  session's minimum confidence, one level up per `unsure` to the explain
  level; only a unit decided as a violation is explained.
- What each call is shown and its reply's schema (`reply.rs`): every call is
  made from a `Step`, a level, a system prompt and a schema.
- The numbers a reply echoes (`numbering.rs`): a number is a position in the
  list it counts, one a prompt printed or the reply's own; a number that
  names nothing fails its reply instead of shortening it.
- What a module is on disk (`module.rs`): its roots, `intent.md`, outline.
- `identify.rs` names and describes what findings are about with small calls;
  `synthesize.rs` has one Frontier call diagnose, root and write the review.
- Concurrency and order: every unit, rule and module is asked at once; the
  findings are in one total order, so the same answers give the same prompt.

## Never
- No backend is named: a model is reached only through `Ask` at a
  `ModelLevel`, System One through `Decide`; no `Provider`, CLI, key or URL.
- No rule's content is known: a rule is its id, question, outcomes, skills
  and context; no id or outcome name is matched.
- No call is made without a `schema`, no finding is kept under an outcome that
  is not its rule's violation, and no position a model counts reaches a public
  type except as the `FindingIx`, `ThingIx` or `DiagnosisIx` it resolved to.
- The describe step never sees the code: its prompt is the name and context.
- A failed call never stops the others and is never silent: a review or
  naming call is an entry in `Report::failures`, the lead review's an error
  on stderr; the lead is told how many failed.
- The module root is context only: a root file is shown alone.

## Contract
`mod.rs` is the contract; stage files are `pub(super)`, `Source` `pub(crate)`.
`plan` makes `Call`s, each holding its rules; `Session` (who a review asks,
how sure System One must be) has the steps `review`, `identify`, `synthesize`;
they pass `Report` (findings fixed when `review` returns, the rules that
asked), `Finding`, `Thing`, `Diagnosis`, `Root`, `Synthesis` and the positions
`FindingIx`, `ThingIx`, `DiagnosisIx`.

## Enforced by
- `desloppify/tests/transcript.rs`: a scripted review held to `tests/golden/`;
  a change to any prompt, schema, level or reported datum is a golden diff.
- `desloppify/tests/review.rs`, via the production API with scripted agents:
  `a_finding_number_that_names_nothing_is_an_error_not_a_shorter_list`,
  `a_symptom_number_that_names_nothing_fails_its_module_whole`,
  `a_plan_is_reviewed_under_the_rules_it_was_planned_with`,
  `a_session_settles_at_or_above_its_minimum_confidence_and_asks_the_model_below`,
  `a_finding_under_an_outcome_that_is_not_a_violation_is_a_failure`,
  `a_naming_outside_its_schema_still_counts_its_tokens`, and three that fail
  if a stage serialises its asks: `every_rule_about_a_unit_is_asked_at_once`,
  `every_unit_is_reviewed_at_once`, `every_converging_module_is_named_at_once`.
- Types: `Ask` and `Decide` are the only way out; `Surroundings` is closed.

## Behind
- `module_intent` walks up to the first `Cargo.toml`, a file outside any
  crate to the filesystem root; no rule asks for more, so nothing pins the stop.
- Two walks: `module::root_outline` looks only in the file's own directory,
  `module::containing` climbs to the nearest root; they differ for a rootless
  directory and a 2018 module. `module::shape` drops only neighbouring
  duplicates, so a root with symptoms but not the most is outlined twice.
  All pinned by the golden.
- A crate-wide rule never shares a call and gets no root or intent whatever
  its `context` (rule/intent.md, Behind).
- `Session::ask` and `::decide` are `pub` only so tests can read back their
  scripted agents (`Reviewers` had them); `main` never reads them.
- A finding's `line` is unchecked: it prints as its own number, so no list
  resolves it, and 0 can mean the whole file.
- A lead review rejected for a bad number is in no usage row; carrying
  `Usage` on that error is a follow-up.

## Decided in
`desloppify/README.md` ("How a review runs", "Layout"); §4 of
`docs/process/2026-10-15-intent-lives-with-the-module.md`.
