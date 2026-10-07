# desloppify

Rule-driven code review. tree-sitter finds the code, a model reviews it.

```bash
cargo run -p desloppify -- --backend dry-run path/to/src      # price it, no calls
cargo run -p desloppify -- --backend claude-code path/to/src  # the Claude Code CLI
ANTHROPIC_API_KEY=... cargo run -p desloppify -- path/to/src
GEMINI_API_KEY=...    cargo run -p desloppify -- --backend gemini path/to/src
```

Every backend implements one contract, `agent::Ask`, and reports the tokens
each call used; a run ends with calls and tokens per rule and per level on
stderr. `dry-run` is a backend like the others — it finds nothing and prices
each call at four characters a token — so a dry run goes through exactly the
pipeline a real one does. `claude-code` runs `claude -p` per call from an
empty directory with no tools, MCP servers or settings; the cheap levels run
with thinking off (it was most of their output, and output costs the most),
the design levels at medium and high effort. Reviewers' replies are held to
a JSON schema by the backend (`--json-schema`, or the API's structured
output), so nothing parses around a model's prose.

The design is many small, cheap calls: most rules send one function at level
1, and only rules that need a whole file, or a whole crate, pay for one.

Each rule is applied by its own call, which sees one rule and a little code,
so no reviewer forgets a rule. Then one level-4 call, the lead, reads every
finding and writes the review: grouped by file, duplicates across rules
merged, trivial or mistaken findings dropped and counted, recurring patterns
called out. `--findings-only` prints the raw findings instead, one
`path:line: [rule] message` per line.

Exits non-zero on any finding
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
| `scope` | the rule's input: `"file"`; a named part — `"functions"`, `"function_signatures"`, `"function_names"`, `"function_bodies"`, `"types"`, `"comments"` — in every language that has it; or `{"language": "rust", "query": "... @target"}` |
| `review` | `"each"` (default): one call per captured part. `"function"`: each function holding a match, naming the matched lines. `"together"`: all of a file's parts in one call. `"file"`: the whole file, naming the matched lines, if it has any. `"crate"`: every part from every file in a crate in one call, under `== path ==` headers |
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

| Rule | Level | Reviews | Source |
|---|---|---|---|
| `boolean-argument` | 1 | each of query matches | STYLE: boolean arguments |
| `comment-says-why` | 1 | each of functions | STYLE: comments |
| `control-plane-64-bit` | 1 | each of types | CLAUDE.md: control plane is 64-bit |
| `guard-clauses` | 1 | each of functions | STYLE: guard clauses; open cases only go down |
| `hidden-parser` | 1 | each function holding query matches | parse, don't poke at strings |
| `interface-lives-in-mod-rs` | 1 | a file's query matches together | behavioral contracts |
| `magic-numbers` | 1 | each of functions | STYLE: magic numbers |
| `no-terminal-logic-in-pixelflow` | 1 | file | CLAUDE.md: no terminal logic |
| `no-tests-inside-the-implementation` | 1 | a file's query matches together | behavioral contracts |
| `panicking-unwrap` | 1 | each function holding query matches | CLAUDE.md: no silent failures |
| `per-frame-allocation` | 1 | each of functions | CLAUDE.md: zero allocations |
| `silent-failure` | 1 | each of functions | CLAUDE.md: fail loud |
| `simd-is-codegens` | 1 | file | CLAUDE.md: SIMD is an implementation detail |
| `test-names-it-should` | 1 | a file's query matches together | STYLE: "it should" names |
| `actor-lane-choice` | 2 | file | CLAUDE.md: actor lanes |
| `behavior-through-the-trait` | 2 | a file's query matches together | behavioral contracts |
| `fold-before-dispatch` | 2 | each of functions | CLAUDE.md / STYLE: fold before dispatch |
| `hardware-instruction-first` | 2 | each of functions | CLAUDE.md: take what the hardware gives |
| `mask-is-not-a-number` | 2 | each of functions | CLAUDE.md: floating point at the edges |
| `tests-target-the-contract` | 2 | file | behavioral contracts |
| `effects-are-returned` | 3 | whole file at query matches | behavioral contracts |
| `function-shapes` | 3 | a crate's function signatures together | names vs namespaces, argument structs, denotation, extraction |
| `invariant-in-comment` | 3 | file | CLAUDE.md: denote before you build |
| `mod-rs-is-a-contract` | 3 | file | behavioral contracts |
| `trait-first` | 3 | file | CLAUDE.md / STYLE: trait first |

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
