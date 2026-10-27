# desloppify/src/decide

**Is.** System One of a review: a function from one piece of code and a set of
closed-set questions about it to, for each question it can answer, a label and a
confidence in 0..1 (desloppify/README.md, "How a review runs", step 2). It is
the cheap decider asked before any model; an answer it is sure of settles a
rule for a unit, and the rest go to `agent::Ask`.

## Owns
- The shape of a closed-set question and its answer: `Choice` (name,
  instructions, labels), `Decided` (label, confidence), `Decisions` (answers by
  question name, and the `Usage` the request cost).
- The absence of an answer: a question not in `Decisions::answers` is
  unanswered, not any label.
- The Jev wire format and its retry (`POST /v1/systemone`; 408, 429 and 5xx
  retried at most twice after `Retry-After`), the `TYPESAFE_*` environment that
  configures it, and its concurrency bound (`JevConfig::jobs`).

## Never
- No knowledge of rules, units, files or findings: it sees a `state` string
  and `Choice` values, never a `Rule`, a `Call` or a path.
- No model call and no `agent::Ask`; the only item it takes from `agent` is
  `Usage`, to report what a request cost.
- No threshold. Whether a confidence is enough is the review's
  (`min_confidence`); an implementation reports what it computed.
- No hidden failure: `Err` means no question was answered. A failing System One
  is the review's to survive (every question goes to the model and the outage
  is reported), never an empty `Ok` from here.
- No implementation type is nameable outside the module: `Jev`, `DryRun` and
  `Nobody` are `pub(super)` (the compiler enforces it), reached as `impl Decide`.

## Contract
`mod.rs`: the trait `Decide` (`decide(&self, state, questions)`, safe to call
concurrently); the data `Choice`, `Decided`, `Decisions`; `JevConfig` and
`JevConfig::from_env`; the constructors `jev`, `dry_run`, `none`, each returning
`impl Decide`. `jev.rs`, `dry_run.rs` and `none.rs` are implementation behind it.

## Enforced by
- `desloppify/tests/decide.rs`: Jev against a local server speaking its API
  (every question posted as a choice, retries and their limit, a refusal not
  retried, a reply that is not answers is an error), `dry_run` and `none`.
- `desloppify/tests/review.rs`: a confident answer settles a rule without a
  model, a low-confidence one goes to the model, and a System One outage sends
  every question to the model and is reported.
- desloppify rules `mod-rs-is-a-contract`, `interface-lives-in-mod-rs` and
  `behavior-through-the-trait` (desloppify reviews its own tree).

## Behind
- A returned label is not checked against the question's labels. `Jev` passes
  the server's `choice` through, and `review/run.rs` treats an unknown label as
  unsettled, so "one of that question's labels" is held by the caller, not by
  `Decide`. No plan or Linear issue names it.
- `Choice::name` unique within a request is stated in a doc comment and not
  checked; the review satisfies it with rule ids.
- `JevConfig` has public fields, `api_key` among them, and derives `Debug`.

## Decided in
`desloppify/README.md` (How a review runs, steps 2 and 3; Layout; Backends).
No plan in `docs/plans` or Linear issue names this module.
