# desloppify

Rule-driven code review. tree-sitter finds the code, a rate-limited agent
([rig](https://github.com/0xPlaygrounds/rig)) reviews it.

```bash
ANTHROPIC_API_KEY=... cargo run -p desloppify -- path/to/src
GEMINI_API_KEY=...    cargo run -p desloppify -- --provider gemini --rpm 30 path/to/src
```

Prints `path:line: [rule] message` per finding; exits non-zero on any finding
or any snippet that could not be reviewed.

## Rules — `rules/<id>.json`

| Field | |
|---|---|
| `level` | 1–4: how capable a model the question needs |
| `language` | `rust` (add a grammar in `src/language.rs` for more) |
| `query` | optional tree-sitter query; each `@target` capture is one review. Absent → the whole file |
| `skills` | optional skill names to put ahead of the prompt |
| `prompt` | what to flag |

| Level | Anthropic | Gemini |
|---|---|---|
| 1 | Haiku | Flash-Lite |
| 2 | Sonnet | Flash |
| 3 | Opus | 2.5 Pro |
| 4 | Fable | 3 Pro |

The ladder is `Provider::model` in `src/model.rs`.

## Skills — `skills/<name>/SKILL.md`

Shared review knowledge. A rule that names a skill gets its text in the prompt.
Rules and skills are validated at load: a bad query, a missing `@target`, or an
unknown skill fails before any call is made (`tests/shipped_rules.rs`).
