# desloppify

Rule-driven code review. tree-sitter finds the code, an agent
([rig](https://github.com/0xPlaygrounds/rig)) reviews it.

```bash
ANTHROPIC_API_KEY=... cargo run -p desloppify -- path/to/src
GEMINI_API_KEY=...    cargo run -p desloppify -- --provider gemini path/to/src
```

`--dry-run` counts the calls each rule would make and makes none — check it
before pointing a frontier level at a whole tree.

Prints `path:line: [rule] message` per finding; exits non-zero on any finding
or any snippet that could not be reviewed.

## Layout

desloppify follows its own rules: every module is a directory whose `mod.rs`
is the contract — traits (`agent::Ask`, `rate_limit::RateLimiter`,
`rate_limit::Clock`), the types they mention, and constructors returning
`impl Trait` — and every other file is `pub(super)` implementation. All tests
are in `tests/` and use only the exported API; time is an injected `Clock`
so the limiters are tested without sleeping, and `review` is tested with a
scripted `Ask`.

## Rules — `rules/<id>.json`

```json
{ "level": 2, "scope": "function_names", "review": "together",
  "skills": ["rust-idioms"], "prompt": "Flag ..." }
```

| Field | |
|---|---|
| `level` | 1–4: how capable a model the question needs |
| `scope` | the rule's input: `"file"`; a named part — `"functions"`, `"function_names"`, `"function_bodies"`, `"types"`, `"comments"` — in every language that has it; or `{"language": "rust", "query": "... @target"}` |
| `review` | `"each"` (default): one call per captured part. `"together"`: all of a file's parts in one call, for questions about consistency across them. `"file"`: the whole file, naming the captured lines, if it has any — for a match that needs its surroundings |
| `context` | `"module_root"`: each call also sees the root (`mod.rs`/`lib.rs`/`main.rs`) of the reviewed file's module — the contract an implementation file is judged against |
| `paths`, `exclude` | optional globs over paths relative to where the review runs (`pixelflow-*/**`, `**/tests/**`); no `paths` means every file |
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

## Shipped rules

Drawn from `CLAUDE.md`, `AGENTS.md`, `docs/STYLE.md` and `.claude/agents/`.
Only rules needing judgment are here: what a deterministic check can catch
(`cfg` encapsulation, `let _ =` on `#[must_use]`, conventional commits) is
already a CI job or a lint, and stays there.

| Rule | Level | Scope | Source |
|---|---|---|---|
| `boolean-argument` | 1 | `bool` params | STYLE: boolean arguments |
| `too-many-arguments` | 1 | fns with ≥4 params | STYLE: argument count |
| `magic-numbers` | 1 | file | STYLE: magic numbers |
| `panicking-unwrap` | 1 | file, at each `unwrap`/`expect` | CLAUDE.md: no silent failures |
| `test-names-it-should` | 1 | a file's test names together | STYLE: "it should" names |
| `comment-says-why` | 2 | file | STYLE: comments |
| `guard-clauses` | 2 | file | STYLE: guard clauses |
| `silent-failure` | 2 | file | CLAUDE.md: errors handled, fail loud |
| `naming-consistency` | 2 | a file's fn names together | CLAUDE.md: name vs namespace |
| `control-plane-64-bit` | 2 | pixelflow types | CLAUDE.md: control plane is 64-bit |
| `no-terminal-logic-in-pixelflow` | 2 | pixelflow files | CLAUDE.md: no terminal logic |
| `simd-is-codegens` | 2 | pixelflow files outside the emitters | CLAUDE.md: SIMD is an implementation detail |
| `per-frame-allocation` | 2 | render-path crates | CLAUDE.md: zero allocations |
| `actor-lane-choice` | 2 | actor crates | CLAUDE.md: actor lanes |
| `fold-before-dispatch` | 3 | file | CLAUDE.md / STYLE: fold before dispatch |
| `trait-first` | 3 | file | CLAUDE.md / STYLE: trait first |
| `invariant-in-comment` | 3 | file | CLAUDE.md: denote before you build |
| `mask-is-not-a-number` | 3 | kernel crates | CLAUDE.md: floating point at the edges |
| `hardware-instruction-first` | 3 | kernel crates | CLAUDE.md: take what the hardware gives |
| `interface-lives-in-mod-rs` | 1 | impl files, at anything wider than `pub(super)` | behavioral contracts |
| `behavior-through-the-trait` | 2 | public inherent methods | behavioral contracts |
| `tests-target-the-contract` | 2 | test files | behavioral contracts |
| `no-tests-inside-the-implementation` | 1 | test modules under `src/` | behavioral contracts |
| `mod-rs-is-a-contract` | 3 | `mod.rs`, `lib.rs` | behavioral contracts |
| `effects-are-returned` | 3 | traits and trait impls | behavioral contracts |

Skills: `style-guide` (STYLE.md distilled), `pixelflow-architecture`
(CLAUDE.md's constraints) and `behavioral-contracts` (the module shape of
`core-term`'s `ansi/`, `term/` and PTY troupe: a trait contract in the module
root, a private implementation behind it, effects returned as data, tests
against the trait).

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
