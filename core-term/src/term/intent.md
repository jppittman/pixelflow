# core-term/src/term

**Is.** The terminal as a state machine over one closed input: each
`EmulatorInput` takes the screen state to its successor and returns the effect
the caller must perform, and `TerminalSnapshot` is the value the host reads
that state as, the table it applies a compiled kernel to (DENOTATION §5, §6).
It performs no effect.

## Owns
- The terminal's state: grids, scrollback, cursor, DEC and standard modes,
  character sets G0 to G3, selection, the viewport, and the dirt a snapshot
  reports (`screen.rs`, `cursor.rs`, `modes.rs`, `charset.rs`, `snapshot.rs`).
- The input and effect vocabularies: `EmulatorInput`, `UserInputAction`,
  `ControlEvent` in, `EmulatorAction` out (`action.rs`).
- Cell geometry and zoom (`layout.rs`), character width (`unicode.rs`), OSC 52's
  base64 (`base64.rs`).

## Never
- No effect is performed: nothing under `term/` opens a file or socket, sends
  on a channel, spawns, sleeps or reads a clock. Writing the PTY, the
  clipboard, the title or the bell is a returned `EmulatorAction`.
- No state changes except through `EmulatorInput` (`Ansi`, `User`, `Control`);
  an `AnsiCommand` arrives parsed from `ansi/`, never as raw bytes.
- No rendering: no pixel, font or `Scene`; the `pixelflow-runtime` types named
  are input vocabulary (`MouseButton`, `KeySymbol`, `Modifiers`, `Selection`).
- No reference to `terminal_app`, `io` or `messages`; `unsafe` only in `unicode.rs`.

## Contract
`mod.rs`: `TerminalInterface` with `interpret_input(EmulatorInput) ->
Option<EmulatorAction>` and `get_render_snapshot() -> Option<TerminalSnapshot>`
(`None` under synchronized output); `TerminalEmulator` is its one `impl`
(`emulator/`, own `intent.md`). Re-exports: `ControlEvent`, `EmulatorAction`,
`UserInputAction`, `CharacterSet`, `MouseEncodingParams`, `MouseEventKind`,
`Layout`, `CursorRenderState`, `CursorShape`, `Point`, `Selection`,
`SelectionMode`, `SnapshotLine`, `TerminalSnapshot`.

## Enforced by
- `EmulatorInput` is a closed enum: a new input is a variant every `match` handles.
- Desloppify `effects-are-returned`, `mod-rs-is-a-contract`,
  `interface-lives-in-mod-rs`, `behavior-through-the-trait`,
  `tests-target-the-contract`.
- `core-term/tests/print_text_equivalence.rs`: a text run equals its characters.

## Behind
- Nothing uses `TerminalInterface`. `TerminalApp` holds a `TerminalEmulator`
  and calls inherent methods; the trait impl only forwards to inherent methods
  of the same names, the second door `behavior-through-the-trait` names. It
  also has public methods the trait lacks (`print_text`, `cell_at`,
  `scroll_viewport`, `encode_mouse_event` and eleven more). No plan or Linear
  issue tracks either.
- `get_render_snapshot` takes `&mut self` because it records
  `last_snapshot_view` and `last_cursor_mark`, so a query advances damage state.
- `screen.rs`, `layout.rs` and `charset.rs` carry `#[cfg(test)]` modules inside
  implementation files, against behavioral-contracts.

## Decided in
`desloppify/skills/behavioral-contracts/SKILL.md` ("Tight contracts"),
`docs/process/2026-10-13-tests-check-the-as-if-contract.md`,
`docs/language/2026-09-25-the-language-is-kernel.md` D13, CLAUDE.md.
