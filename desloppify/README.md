# desloppify

Rule-driven code review. Each rule is a typed decision; tree-sitter cuts the
code into the units a rule asks about, a cheap decider answers every rule's
question about a unit at once, and a model explains only the units decided
as violations.

```bash
cargo run -p desloppify -- --backend dry-run --system-one dry-run path/to/src  # price it, no calls
cargo run -p desloppify -- --backend claude-code path/to/src                   # the Claude Code CLI
ANTHROPIC_API_KEY=... cargo run -p desloppify -- path/to/src
GEMINI_API_KEY=...    cargo run -p desloppify -- --backend gemini path/to/src
TYPESAFE_API_KEY=...  cargo run -p desloppify -- --system-one jev path/to/src  # Jev decides first
```

## How a review runs

1. **Plan.** Every rule names its `unit` — a whole file, an outline (the file
   with every function body elided), each function, function signatures,
   each type. Rules that see the same unit of the same file share one call.
2. **Decide, System One** (`decide::Decide`, `--system-one`). One request
   per call carries every rule's question about that unit, each a closed set
   of outcomes. An answer at or above `--min-confidence` (0.7) settles the
   rule for the unit. `jev` is TypeSafe AI's Jev, a model that returns
   calibrated choices rather than text, at a few cents per million input
   tokens and no output; `none` (the default) skips this step.
3. **Decide, model** (`agent::Ask`). A question System One did not settle
   goes to a model at the rule's `levels.decide`, held to a schema whose only
   values are the rule's outcomes or `unsure`. `unsure` asks one level higher,
   up to `levels.explain`.
4. **Explain.** A unit decided as one of the rule's `violations` goes to a
   model at `levels.explain`, told the question and the outcome, which reports
   findings: line, what, and the fix.
5. **Diagnose.** Findings are symptoms. Where they converge on a module —
   at least three, from at least two rules — the review asks what they are
   symptoms *of*, the way a person would: what is this thing, how does it
   work, what does it do, how does it behave? One level-3 call names the
   thing in its domain's words ("an assembler"). One level-4 call describes
   it from first principles and is never shown the code, so it cannot borrow
   the code's model of itself. A last level-4 call translates that
   description's shape onto the code's outline — where each part is, or that
   it is nowhere — and states the diagnosis: the model the code is missing,
   the symptoms it explains, and what falls out once the code has the
   thing's shape. Diagnoses are symptoms too: a module with two or more is
   asked once more, at level 4, for the root beneath them — the deepest
   model that makes several true at once, with its consequences followed to
   closure.
6. **Lead review.** One level-4 call reads every finding and writes the
   review, leading with the roots, then the diagnoses: grouped by file, duplicates across
   rules merged, trivial or mistaken findings dropped and counted, recurring
   patterns called out.
   `--findings-only` prints the raw findings instead, one
   `path:line: [rule/outcome] message` per line.

Each model call sees one rule and a little code, so none forgets a rule; the
fine outcomes, which are most of them, never pay for an explanation. A run
ends with each rule's tally of outcomes and escalations, and calls and tokens
per rule, for System One and for the lead, on stderr. Exits non-zero on any
finding or any unit that could not be decided.

The shape is Jac's meaning-typed `by llm` function: the question and the
outcomes' meanings are the signature, the closed set of outcomes the return
type, so a reply is a value or a schema failure — never prose to parse.

## Backends

Every model backend implements `agent::Ask` and reports the tokens each call
used. `dry-run` is a backend like the others: it answers every schema with
its first value (so every decision is the rule's first fine outcome) and
prices each call at four characters a token, so a dry run goes through
exactly the pipeline a real one does. `claude-code` runs `claude -p` per call
from an empty directory with no tools, MCP servers or settings; levels 1–2
run with thinking off, 3 and 4 at medium and high effort. Replies are held to
a JSON schema by the backend (`--json-schema`, or the API's structured
output).

System One backends implement `decide::Decide`: `jev` (`TYPESAFE_API_KEY`,
and optionally `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL`), `dry-run`
(first label at full confidence), and `none`.

| Level | Anthropic | Gemini |
|---|---|---|
| 1 | Haiku | Flash-Lite |
| 2 | Sonnet | Flash |
| 3 | Opus | 2.5 Pro |
| 4 | Fable | 3 Pro |

The ladder is `Provider::model` in `src/model/mod.rs`.

## Layout

desloppify follows its own rules: every module is a directory whose `mod.rs`
is the contract — traits (`agent::Ask`, `decide::Decide`,
`rate_limit::RateLimiter`, `rate_limit::Clock`), the types they mention, and
constructors returning `impl Trait` — and every other file is `pub(super)`
implementation. All tests are in `tests/` and drive the production API —
what `main` uses — and nothing is public only so a test can reach it: what
a rule shows the model is read from the prompts a review sends, path
scoping from the binary run at the repository root. Time is an injected
`Clock`, so the limiters are tested without sleeping; `review` is tested
with a scripted `Ask` and `Decide`; Jev is tested against a local server
speaking its API.

## Rules — `rules/<id>.json`

```json
{
  "unit": "functions",
  "levels": { "decide": 1, "explain": 2 },
  "exclude": ["**/tests/**"],
  "question": "Does this function swallow a failure where the caller needed to know?",
  "fine": { "fails_loud": "Every failure is returned or panics naming its cause." },
  "violations": { "swallowed": "`.ok()` or `.unwrap_or_default()` on an operation whose failure mattered." },
  "guidance": "Name the failure and who needed it."
}
```

| Field | |
|---|---|
| `unit` | what one call sees: `"file"`, `"outline"`, `"functions"`, `"function_signatures"` or `"types"` |
| `group` | for a part of a file: `"each"` (default, one call per part), `"file"` (a file's parts together) or `"crate"` (a crate's parts together, under `== path ==` headers) |
| `levels` | 1–4: `decide` is who answers the question when System One does not, `explain` who writes the findings and the top of the `unsure` ladder; `explain ≥ decide` |
| `question` | the decision, answered once per unit |
| `fine`, `violations` | the outcomes, each name with its meaning; at least one of each. `unsure` is every rule's own |
| `guidance` | optional, for the explainer: what a finding should say |
| `context` | `"module_root"`: each call also sees the outline of the reviewed file's module root — the contract an implementation file is judged against |
| `paths`, `exclude` | optional globs over paths relative to where the review runs (`pixelflow-*/**`, `**/tests/**`); no `paths` means every file |
| `skills` | optional skill names to put ahead of the question |

Everything is checked when the rules load, before any call
(`tests/rule.rs`, `tests/shipped_rules.rs`). Units are per-language
tree-sitter queries in `Language::query` (`src/language/mod.rs`); a new
language adds a grammar and its queries there. tree-sitter only finds units;
it judges nothing.

## Shipped rules

Drawn from `CLAUDE.md`, `AGENTS.md`, `docs/STYLE.md`, `.claude/agents/` and
`docs/plans/`. Only rules needing judgment are here: what a deterministic
check can catch (`cfg` encapsulation, `let _ =` on `#[must_use]`,
conventional commits) is already a CI job or a lint, and stays there.

| Rule | Levels | Unit | Source |
|---|---|---|---|
| `boolean-argument` | 1→1 | a file's function signatures | STYLE: boolean arguments |
| `no-tests-inside-the-implementation` | 1→2 | outline | behavioral contracts: no test code or test-only API in the source tree |
| `test-names-it-should` | 1→1 | outline | STYLE: "it should" names |
| `actor-lane-choice` | 1→2 | file | CLAUDE.md: actor lanes |
| `behavior-through-the-trait` | 1→2 | outline + module root | behavioral contracts |
| `comment-says-why` | 1→2 | each of functions | STYLE: comments |
| `control-plane-64-bit` | 1→2 | each of types | CLAUDE.md: control plane is 64-bit |
| `fold-before-dispatch` | 1→2 | each of functions | CLAUDE.md / STYLE: fold before dispatch |
| `guard-clauses` | 1→2 | each of functions | STYLE: guard clauses; open cases only go down |
| `hardware-instruction-first` | 1→2 | each of functions | CLAUDE.md: take what the hardware gives |
| `hidden-parser` | 1→2 | each of functions | parse, don't poke at strings |
| `interface-lives-in-mod-rs` | 1→2 | outline + module root | behavioral contracts |
| `magic-numbers` | 1→2 | each of functions | STYLE: magic numbers |
| `mask-is-not-a-number` | 1→2 | each of functions | CLAUDE.md: floating point at the edges |
| `no-terminal-logic-in-pixelflow` | 1→2 | outline | CLAUDE.md: no terminal logic |
| `panicking-unwrap` | 1→2 | each of functions | CLAUDE.md: no silent failures |
| `per-frame-allocation` | 1→2 | each of functions | CLAUDE.md: zero allocations |
| `registers-come-from-the-allocator` | 1→2 | each of functions | register-allocation escape hatches plan |
| `silent-failure` | 1→2 | each of functions | CLAUDE.md: fail loud |
| `simd-is-codegens` | 1→2 | file | CLAUDE.md: SIMD is an implementation detail |
| `tests-target-the-contract` | 1→2 | file | behavioral contracts: tests observe the production API's output |
| `effects-are-returned` | 2→3 | file | behavioral contracts |
| `function-shapes` | 2→3 | a crate's function signatures | names vs namespaces, argument structs, denotation, extraction |
| `invariant-in-comment` | 2→3 | file | CLAUDE.md: denote before you build |
| `mod-rs-is-a-contract` | 2→3 | file | behavioral contracts |
| `trait-first` | 2→3 | file | CLAUDE.md / STYLE: trait first |
| `phases-in-order` | 2→3 | file | a stage predicting what a later one needs, or deciding what an earlier one owns |

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

Asked before every model call: `None` before a first attempt, `Some(error)`
before retrying a failure. `Ok(wait)` sleeps then calls; `Err` gives up.
Errors that can't succeed on retry (401, malformed request) never reach it.
Two impls, picked with `--limiter`:

- `adaptive` (default): AIMD. Paces calls at a rate starting at `--rpm`; a
  429 halves it (once per minute, so in-flight 429s count once), unthrottled
  busy time grows it back toward `--max-rpm`. Honours `Retry-After`. A 5xx is
  retried at the current pace without cutting it. It reads errors through a
  classifier (`agent::classify` for rig), so it knows nothing of rig.
- `token-bucket`: a token per call, `--burst` deep, one more per `--refill-ms`.

Both refuse a call that would wait past `--max-wait-secs`. At most `--jobs`
calls are in flight, so a new pace takes effect within that many calls. Jev
retries 408, 429 and 5xx itself, twice, after the server's `Retry-After`.

## Skills — `skills/<name>/SKILL.md`

Shared review knowledge. A rule that names a skill gets its text ahead of its
question.
