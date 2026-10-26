# core-term/src/ansi

**Is.** A total function from the bytes of a PTY to a batch, the sequence
`text₀ cmd₀ text₁ … textₙ` (DENOTATION §1, totality: every byte string has an
answer; §8, the as-if contract: the observable output is the whole contract).
It recognizes and never interprets.

## Owns
- What a byte means as syntax: UTF-8, C0, ESC, CSI, OSC, DCS, PM, APC, and
  the values for malformed input (`U+FFFD`, `AnsiCommand::Error`).
- The batch: text in one buffer, each command at its offset (`AnsiBatch`),
  delivered in order through `AnsiSink`.
- The state a split sequence needs between calls, and its caps (`MAX_PARAMS`,
  `MAX_INTERMEDIATES`, `MAX_OSC_LEN`).

## Never
- No meaning is applied: no cursor, screen, mode, charset or color state. An
  OSC, DCS, PM or APC payload leaves as raw bytes.
- No effect is performed: no I/O, channel, clock or global; the only side
  channel is `log`. Effects are returned as an `AnsiBatch`.
- No dependency on the application: the only `crate::` import is `crate::color`.
- No command per printable character: there is no `Print` command; text is a
  run equal to its characters one by one.
- No malformed CSI's tail becomes text: `CsiIgnore` absorbs it to the final byte.
- `AnsiLexer`, `AnsiToken` and the parser's states are named nowhere outside
  this directory.

## Contract
`mod.rs`: `AnsiParser::process_bytes(&mut self, &[u8]) -> AnsiBatch`, implemented
by `AnsiProcessor` (`new`); `AnsiBatch` (`is_empty`, `drain_into`); `AnsiSink`
(`text`, `command`). `commands`: `AnsiCommand`, `C0Control`, `CsiCommand`,
`EscCommand`, `Attribute`, `SGR_*`. A pending escape, CSI or string sequence
carries to the next call; a pending UTF-8 sequence does not.

## Enforced by
- Types: `AnsiBatch`'s fields are private and its writers `pub(super)`, so
  offsets are ordered and on character boundaries (`batch.rs`); `mod lexer;`
  and `mod parser;` are private in `mod.rs`.
- Desloppify `interface-lives-in-mod-rs`, `mod-rs-is-a-contract`,
  `behavior-through-the-trait`, `effects-are-returned`.
- Tests: `core-term/tests/print_text_equivalence.rs` (run equals characters),
  `core-term/tests/ansi_parser_message_tests.rs` (trait and processor only),
  `core-term/src/ansi/tests.rs` (split sequences; malformed CSI discards its tail).

## Behind
- The `mod.rs` doc says partial UTF-8 is buffered across chunks; the code emits
  `U+FFFD` at the end of each call (pinned by
  `it_should_handle_fragmented_utf8_input_with_intermediate_finalization`),
  though a PTY read can split a character. No plan or Linear issue names it.
- A string past `MAX_OSC_LEN` (1024) is truncated and dispatched unmarked.
- `AnsiProcessor`'s `lexer` and `parser` fields are `pub(super)`, crate-wide;
  nothing outside `ansi` uses them.
- `parser.rs` has a `#[cfg(test)] mod tests` over `is_ground`, `process_token`
  and `is_printable_ascii`, none of them contract.
- `C0Control::from_byte` is an `unsafe` transmute, sound by its variant list.

## Decided in
`desloppify/skills/behavioral-contracts/SKILL.md` (its worked example),
`docs/process/2026-10-13-tests-check-the-as-if-contract.md`; no plan of its own.
