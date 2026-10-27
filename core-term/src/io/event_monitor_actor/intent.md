# core-term/src/io/event_monitor_actor

**Is.** The PTY pipeline of the host: one troupe of three actors that turns the
child's bytes into `AnsiBatch`es for the app, the input of its cell table
(DENOTATION §6), and the app's bytes and resizes into the PTY. DENOTATION has
no section on actors; this module holds no kernel.

## Owns
- The topology and each message's lane: bytes Data, `Resize` Control, `Bind`
  Management. The lifecycle: `new` wires channels, `spawn` sends the `Bind`s
  and plays the troupe, dropping `PtyTroupeHandle` sends `Shutdown` and joins.
- The buffer pool: `POOL_SIZE` buffers of `READ_BUFFER_SIZE`, allocated at
  `Bind`; an empty pool stops reads and the full kernel buffer blocks the shell.
- Each actor's OS bridge, inside `handle_os`: the reader polls `{pty, waker}`,
  the writer `{pty writable, waker}`; the writer queues writes and one
  coalesced resize until `Bind`.

## Never
- No actor names another actor's type: they meet only through `Directory`.
- No blocking outside `handle_os`: the PTY fd is non-blocking; the only wait
  is `monitor.events(.., -1)` after `waker.arm()`.
- No send result is discarded: the module has no `let _ =` and no `.ok()`.
- No terminal semantics: nothing here imports `term` or `terminal_app`; the
  parser only calls `AnsiProcessor::process_bytes` and forwards.
- No bulk message on Control or Management, no resize on Data, no resource
  except by `Bind`.

## Contract
`mod.rs` carries the topology doc; `reader.rs`, `parser.rs`, `writer.rs` are
private modules with `pub(super)` actors.
- `PtyTroupe` (`new`, `writer_handle`, `spawn`), `PtyTroupeHandle`,
  `PtyWriterHandle`, `WriterControl::Resize`, `ParserControl::Reset`, and
  `ReaderManagement`, `ParserManagement`, `WriterManagement` (one `Bind` each).

## Enforced by
- Types: `PtyReader` has `type Control = Infallible`; `spawn(self)` and
  `Troupe::play(self)` consume what they start; each actor's PTY is an
  `Option<Bound*>`, so no use precedes `Bind`.
- Tests through the real surface in `mod.rs` (`troupe_delivers_output_and_child_exit`,
  `shutdown_interrupts_blocked_reader`, `resize_before_bind_is_applied`,
  `it_should_flush_writes_queued_before_bind_once_bound`) and
  `core-term/tests/actor_roundtrip_tests.rs` (`pty_writer_resize_preempts_queued_writes`).
- desloppify `actor-lane-choice`, `silent-failure`, `tests-target-the-contract`.
  Review only: no blocking outside `handle_os`.

## Behind
- `ShutdownMode` has one variant, `Immediate`, so dropping `PtyTroupeHandle`
  loses the writer's queued bytes; the design's §3.5 specified `DrainAll`
  (2026-10-15; no Linear issue).
- `ParserControl::Reset` is handled and sent by nothing; supervised restart,
  its use, was declined in §5 Phase 3 (2026-10-15).
- The app sinks are `Box<dyn PtySender>`, dispatched per send; §3.3 left an
  `ActorHandle` as a follow-up (no Linear issue).
- `ParserControl`, `ReaderManagement`, `ParserManagement` are `pub` with no
  user outside the module; after a write error the writer latches `broken` and
  drops later writes without telling the app (2026-10-15; no plan).

## Decided in
`docs/archive/designs/pty-actor-troupe.md` (Done; Phase 3 not pursued).
