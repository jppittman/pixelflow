# core-term/src/term/emulator

**Is.** The host-side (not program) state machine behind the cell table (DENOTATION §5, §6): a
function from (terminal state, parsed command, user input or control event) to (state, an
optional returned `EmulatorAction`), plus the snapshot of cells, cursor and dirt the table is
built from.

## Owns
- `TerminalEmulator`'s state beyond the cells: cursor, DEC modes, G0-G3, pending wrap, focus,
  `Layout`, the viewport offset.
- The meaning of every `AnsiCommand`, `UserInputAction` and `ControlEvent` as a state change
  plus at most one `EmulatorAction`. The effect is the return value.
- What a frame is told is dirty: `get_render_snapshot` stamps per-line dirt, adds the rows a
  moved cursor left and entered (`last_cursor_mark`), dirties all lines when the view moved.
- Key and mouse encoding, a function of DEC modes alone: `translate_key_input` and
  `encode_mouse_event` take `&DecPrivateModes` and the event, never the emulator.

## Never
- No effect is performed: no file, socket, channel send, thread or clock. The one
  `std::io::Write` is a `write!` into a `Vec` (`ansi_handler`).
- No tokenizing of the PTY stream (nothing imports `AnsiParser`), and no pixel, font, kernel
  or colour: the only `pixelflow_*` imports are `pixelflow_runtime::input` types.
- Every entry (`process_ansi_command`, `process_user_input_action`, `process_control_event`)
  clears pending wrap first; printing sets it. A run prints as its characters one by one.
- No snapshot while synchronized output (DEC 2026) is on; no cursor in a scrolled-back view;
  PTY output returns the view to the live screen.
- Nothing outside `term/` names `Screen` or `DecPrivateModes`: `mod emulator` is private and
  `term` re-exports only `TerminalEmulator` and the mouse types.

## Contract
`mod.rs`: `TerminalEmulator::{new, interpret_input, print_text, get_render_snapshot, cell_at}`;
viewport `{scroll_viewport, reset_viewport, viewport_offset, scrollback_len}`; selection
`{start_selection, extend_selection, apply_selection_clear, clear_selection,
get_selected_text}`; mouse `{encode_mouse_event, is_mouse_tracking_active, reports_all_motion,
reports_button_motion}`, `mouse::{MouseEncodingParams, MouseEventKind}`. `TerminalInterface`
(`core-term/src/term/mod.rs`) is `interpret_input` and `get_render_snapshot`. Behavior is
`pub(super)` by concern: `ansi_handler`, `char_processor`, `cursor_handler`, `screen_ops`,
`mode_handler`, `osc_handler`, `input_handler`, `key_translator`, `mouse`.

## Enforced by
- Types: `Screen` and `DecPrivateModes` are `pub(super)`; the struct holds no channel or
  handle, so an effect leaves only as an `EmulatorAction`. Desloppify `effects-are-returned`
  (its exemplar is `EmulatorAction`) and `no-terminal-logic-in-pixelflow`.
- `core-term/tests/print_text_equivalence.rs` (runs equal characters, probed by a trailing one);
  `core-term/src/term/tests.rs`: `scrolled_back_idle_snapshot_is_clean`,
  `returning_to_live_screen_marks_dirty`, `cursor_only_movement_marks_its_rows_dirty`.

## Behind
- `mod.rs` holds the snapshot algorithm (`get_render_snapshot`, lines 189-314), against
  desloppify `mod-rs-is-a-contract`; no plan or Linear issue tracks it.
- `get_render_snapshot(&mut self)` clears the dirt, so the flags live in the snapshot alone
  (`core-term/src/terminal_app.rs`); damage is split over `Screen::dirty` and two marks. Untracked.
- `scrollback_generation` is bumped by convention ("every new mutation site MUST bump it",
  `core-term/src/term/screen.rs`); CSI 3J in `screen_ops` bumps it by hand. Untracked.
- Pending wrap is also cleared by hand in 24 places (`cursor_handler`, `screen_ops`,
  `methods`); `methods` is a miscellany. Untracked.

## Decided in
No plan decides this module's shape: CLAUDE.md "Critical Constraints" and `core-term/intent.md`.
