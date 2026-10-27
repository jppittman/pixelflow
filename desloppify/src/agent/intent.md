# desloppify/src/agent

**Is.** The asking of a model: a function from a `Question` (level, system
prompt, prompt, optional JSON Schema) to an `Answer` (the reply's text and
the tokens it cost), possibly failing (desloppify/README.md, "Backends" and
"Layout"). Every backend is one `impl Ask`; the crate is generic over it.

## Owns
- The trait `Ask` and the types it mentions: `Question`, `Answer`, `Usage`.
- How a level reaches a backend: the model (`Provider::model`), the thinking
  effort (`claude_code::think`), the reply cap (`MAX_REPLY_TOKENS`).
- Each backend's concurrency (a `Semaphore` sized by `jobs`) and, for
  `RigAgent`, retry: it asks its `RateLimiter` before every attempt, and
  `classify` reads a failure for it, so the limiter knows nothing of rig.
- The journal's question key (level, system, prompt, schema) and file format.

## Never
- No file outside `agent/` names `rig_core`, `schemars` or the `claude`
  executable; callers hold an `impl Ask` from a constructor in `mod.rs`.
- No backend parses or rewrites a reply: `Answer::text` is what the model
  returned. With a schema, the provider enforces it.
- Nothing in `agent/` imports `review`, `rule`, `decide` or `skills`: it
  knows no rules, units, findings or decisions.
- No retry of what a retry cannot fix: a non-retryable error is `Err` at once.
- `ClaudeCode` runs with no tools, MCP servers, settings or session, from an
  empty directory: no project's `CLAUDE.md` reaches a call.
- `Journaled` records no reply that breaks the schema it was asked under; a
  torn line is skipped, never an error.

## Contract
`mod.rs` is the contract; `claude_code.rs`, `dry_run.rs`, `journal.rs` and
`rig.rs` are `pub(super)` implementation. Public: `Ask`; `Question`,
`Answer`, `Usage`; constructors returning `impl Ask` (`from_env`,
`claude_code`, `dry_run`, `journaled`); `classify`. `Usage` sums with
`AddAssign` and counts every input token, cached or not.

## Enforced by
- Types: `ClaudeCode`, `DryRun`, `RigAgent` and `Journaled` are `pub(super)`,
  reachable only through constructors returning `impl Ask`.
- `desloppify/tests/agent.rs`: `classify` (a 429 throttles with its
  `Retry-After` seconds, a date is ignored, an outage is a failure) and
  `journal` (a rerun answers without a call, a reply that breaks its schema
  is asked again, a torn last line is skipped).
- `desloppify/tests/review.rs`, `rule.rs`: the pipeline through a scripted
  `Ask` and through `dry_run`.
- desloppify `mod-rs-is-a-contract`, `interface-lives-in-mod-rs` and
  `behavior-through-the-trait` review this directory; CI runs the tests
  (`cargo nextest run --workspace`, `.github/workflows/rust.yaml`).

## Behind
- `mod.rs` says `dry_run` answers every call `[]`; `dry_run.rs` answers an
  object whose enum properties take their first value, as the README says.
  Both docs are stale (2026-10-15; no plan or Linear issue names it).
- `RigAgent::ask` and `ClaudeCode::ask` need a network or the CLI and have no
  test (no plan or Linear issue names it).

## Decided in
desloppify/README.md ("Backends", "Layout", "Rate limiting"); no plan names it.
