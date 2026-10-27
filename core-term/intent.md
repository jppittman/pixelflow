# core-term

**Is.** The host that consumes the kernel language (DENOTATION §5, §6): it
walks a PTY's bytes into a cell table and applies a compiled kernel to it,
once per changed frame. All terminal logic lives here; the kernel side gets
uniforms and a shape.

## Owns
- Every terminal concept: the PTY and its child, ANSI/VT parsing, screens
  (primary, alternate, scrollback), cursor, modes, selection, OSC 52, mouse
  and key encoding; CLAUDE.md's first constraint makes this their only home.
- The PTY pipeline: reader, parser and writer actors as one troupe
  (`io/event_monitor_actor`) over epoll or kqueue.
- `TerminalApp`: engine events and PTY batches in, a cell table from the
  snapshot, a `Scene` out, and the decision that nothing changed. Startup
  (`src/main.rs`) runs the engine troupe on the main thread.

## Never
- No `kernel!` is expanded and no arena built here: the crate depends on
  `pixelflow-core`, `-graphics`, `-runtime` and `actor-scheduler`, never on
  the compiler, IR or codegen crates.
- No pixel is written here: a frame leaves as `AppData::RenderSurface(Scene)`.
- No command per printable character: text reaches the emulator as a run
  (`AnsiSink::text`), equivalent to its characters one by one.
- No bulk message on Control or Management, no latency-critical one on Data,
  no discarded send result: PTY bytes are Data, resize is Control.
- No blocking read of the PTY outside an actor's `handle_os`.

## Contract
`src/lib.rs` declares the library; `src/main.rs` is the binary.
- `ansi`: `AnsiCommand`, `AnsiBatch`/`AnsiSink`. `term`: `TerminalEmulator`,
  `TerminalInterface`, `TerminalSnapshot`, `EmulatorAction`, `Layout`.
- `io`: `NixPty`, `PtyTroupe`, `waker`. `terminal_app`: `TerminalApp`,
  `spawn_terminal_app`, `TerminalAppSender`. `messages`: `TerminalData`.
  `config`: `CONFIG`. `glyph`, `color`, `keys` re-export others' types.

## Enforced by
- Desloppify `no-terminal-logic-in-pixelflow` (every `pixelflow-*` file).
- `core-term/Cargo.toml`'s dependency list: a `kernel!` use does not resolve.
- Desloppify `actor-lane-choice` and `per-frame-allocation` (`core-term/src/**`).
- `core-term/tests/print_text_equivalence.rs` (run equals characters);
  `core-term/tests/actor_roundtrip_tests.rs` (`pty_writer_resize_preempts_queued_writes`).

## Behind
- The terminal-shaped cell grid lives in PixelFlow, not here: `CellGridMetrics`,
  `CellGridShape`, `CELL_STRIDE` come from `pixelflow-core/src/lattice/cell_grid.rs`
  and the program from `pixelflow-graphics/src/render/cell_grid.rs`
  (the-language-is-kernel D13: it leaves core; the plan is Proposed).
- `build_scene` in `core-term/src/terminal_app.rs` allocates the cell table
  on every drawn frame (one-pipeline U3). The cell table as a discrete kernel
  (kernels-evaluate-kernels §2) closes it; no Linear issue tracks it.
- `AppEvent` and `RenderRequest` in `core-term/src/messages.rs` are used
  nowhere else, and `core-term/src/color.rs` names a `pixelflow-render` that
  does not exist; no plan names them.

## Decided in
`docs/language/2026-09-25-the-language-is-kernel.md` D13,
`docs/language/2026-10-15-kernels-evaluate-kernels.md` §2,
`docs/language/2026-09-24-one-pipeline.md` U3, CLAUDE.md "Critical Constraints".
