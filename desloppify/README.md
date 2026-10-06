# desloppify

Rule-driven code review. tree-sitter finds the code, an agent
([rig](https://github.com/0xPlaygrounds/rig)) reviews it.

```bash
ANTHROPIC_API_KEY=... cargo run -p desloppify -- path/to/src
GEMINI_API_KEY=...    cargo run -p desloppify -- --provider gemini path/to/src
```

Prints `path:line: [rule] message` per finding; exits non-zero on any finding
or any snippet that could not be reviewed.

## Rules — `rules/<id>.json`

```json
{ "level": 2, "scope": "function_names", "review": "together",
  "skills": ["rust-idioms"], "prompt": "Flag ..." }
```

| Field | |
|---|---|
| `level` | 1–4: how capable a model the question needs |
| `scope` | the rule's input: `"file"`; a named part — `"functions"`, `"function_names"`, `"function_bodies"`, `"types"`, `"comments"` — in every language that has it; or `{"language": "rust", "query": "... @target"}` |
| `review` | `"each"` (default): one call per captured part. `"together"`: all of a file's parts in one call, for questions about consistency across them |
| `skills` | optional skill names to put ahead of the prompt |
| `prompt` | what to flag |

Named parts are per-language tree-sitter queries in `Language::query`
(`src/language.rs`); a new language adds a grammar and its queries there.

| Level | Anthropic | Gemini |
|---|---|---|
| 1 | Haiku | Flash-Lite |
| 2 | Sonnet | Flash |
| 3 | Opus | 2.5 Pro |
| 4 | Fable | 3 Pro |

The ladder is `Provider::model` in `src/model.rs`.

## Rate limiting — `src/rate_limit/`

```rust
pub trait RateLimiter: Send + Sync {
    fn wait(&self, last: Option<BoxError>) -> Result<Duration, BoxError>;
}
```

Asked before every call: `None` before a first attempt, `Some(error)` before
retrying a failure. `Ok(wait)` sleeps then calls; `Err` gives up. Errors that
can't succeed on retry (401, malformed request) never reach it.
Two impls, picked with `--limiter`:

- `adaptive` (default): AIMD. Paces calls at a rate starting at `--rpm`; a
  429 halves it (once per minute, so in-flight 429s count once), unthrottled
  busy time grows it back toward `--max-rpm`. Honours `Retry-After`. A 5xx is
  retried at the current pace without cutting it. It reads errors through a
  classifier (`agent::classify` for rig), so it knows nothing of rig.
- `token-bucket`: a token per call, `--burst` deep, one more per `--refill-ms`.

Both refuse a call that would wait past `--max-wait-secs`. At most `--jobs`
calls are in flight, so a new pace takes effect within that many calls.

## Skills — `skills/<name>/SKILL.md`

Shared review knowledge. A rule that names a skill gets its text in the prompt.
Rules and skills are validated at load: a bad query, a missing `@target`, or an
unknown skill fails before any call is made (`tests/shipped_rules.rs`).
