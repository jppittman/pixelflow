# Design Doc: PTY Actor Troupe — Unifying PTY I/O Under the Actor Scheduler

## Metadata
- **Author**: Claude (requested by jppittman)
- **Status**: Phase 1 implemented (hand-wired actors; troupe!/kubelet phases pending)
- **Created**: 2026-07-16
- **Reviewers**: —

---

## 1. Overview

### 1.1 Problem Statement

`EventMonitorActor` is called an actor, but only one of its three threads is one.
The PTY pipeline today:

```
ReadThread (raw thread)          ParserThread (real Actor)        TerminalApp (real Actor)
  epoll/kqueue loop      ──D──▶    ActorScheduler<Vec<u8>,          ActorBuilder, 3 lanes
  blocking read                     NoControl, NoManagement>
       ▲                                │
       └────── mpsc::channel ◀──────────┘        (buffer recycling, unbounded)

TerminalApp ── mpsc::sync_channel(128) ──▶ WriteThread (raw thread)
                 PtyCommand::{Write,Resize}      blocking recv loop, write_all
```

Four different communication primitives coexist in one subsystem:

| Edge | Primitive | Lanes? | Backpressure |
|------|-----------|--------|--------------|
| Read → Parser | `ActorHandle` (Data lane) | yes (2 unused) | yes |
| Parser → Read | `mpsc::channel` (recycler) | no | none (unbounded) |
| Parser → App | `Box<dyn PtySender>` → `ActorHandle` | yes | yes |
| App → Write | `mpsc::sync_channel(128)` | no | blocking |

### 1.2 Concrete Deficiencies

1. **Reader and writer are unaddressable.** Neither has an inbox. The reader can
   only be stopped by closing the PTY FD out from under it; the writer only by
   dropping the sender. There is no way to send either one a control message.

2. **Shutdown is Drop-order choreography.** The module doc spends ~40 lines
   explaining the required teardown sequence (write thread first → FD closes →
   read thread's poll fails → parser's channel closes). Correctness depends on
   field-drop order in a struct. The scheduler already has `Message::Shutdown`
   and `ShutdownMode` — none of it is used here.

3. **`Resize` queues behind bulk writes.** `PtyCommand::Resize` (control
   semantics: `TIOCSWINSZ` + `SIGWINCH`, should apply ASAP) shares a FIFO with
   `PtyCommand::Write` (data semantics). Worse, the writer uses blocking
   `write_all`: if the shell stops draining the PTY during a large paste, the
   write thread wedges and resizes are stuck behind it indefinitely.

4. **Child exit is swallowed.** On PTY EOF the read thread logs and exits.
   Nothing reaches `TerminalApp`, so `exit` in the shell leaves a dead window.

5. **No supervision.** The module doc's own error table says: "Parser thread
   panics → App never receives commands → App frozen." The Kubelet/`ServiceHandle`
   machinery (see `actor-scheduler-supervisor-migration.md`) exists precisely
   for this and is unused.

6. **Unused lanes, dead types.** `NoControl`/`NoManagement` placeholders on the
   parser; the parser cannot be told to reset, reconfigure, or flush.

7. **Buffer pool can silently grow.** If the recycler is slow, the reader
   allocates fresh buffers unboundedly (`unwrap_or_else(|| Vec::with_capacity(n))`),
   violating the zero-allocation goal under exactly the load where it matters.

### 1.3 Goals

- Every PTY thread is an `Actor` on an `ActorScheduler` — one communication
  model, three priority lanes, uniform shutdown.
- Resize preempts queued writes; writer never wedges on a full PTY buffer.
- Child exit propagates to `TerminalApp` as a message.
- True end-to-end backpressure: shell → kernel PTY buffer → reader buffer pool
  → parser data lane → app data lane, with a fixed buffer population.
- Keep the existing performance profile (zero steady-state allocation, <1ms
  read latency).

### 1.4 Non-Goals

- Changing the ANSI parser, `TerminalApp`, or the engine troupe.
- Making the scheduler's doorbell fd-based (see §8, future work).
- Windows/ConPTY support.

---

## 2. Key Insight: `park()` Is Already the OS Bridge

The blocker that (presumably) kept the reader as a raw thread is that
`ActorScheduler::run()` blocks on its mpsc doorbell, while the reader must
block on epoll/kqueue. But the codebase already solved this for X11
(`pixelflow-runtime/src/platform/linux/platform.rs`):

- `park(SystemStatus::Idle)` blocks in `XNextEvent`.
- An `X11Waker` (`WakeHandler`) injects a synthetic event to interrupt it
  whenever a message is sent to the actor.
- `park` returns `ActorStatus::Busy` after handling OS work, so the scheduler
  loop uses `try_recv` on the doorbell instead of blocking — **the OS wait
  becomes the doorbell.**

The PTY reader and writer are the same shape with a different waker:

- **Linux**: an `eventfd` registered in the actor's own epoll set. `WakeHandler::wake()`
  writes 8 bytes; `park` sees it as a ready fd, drains it, returns.
- **macOS**: `EVFILT_USER` kevent; `wake()` triggers it with `NOTE_TRIGGER`.

Call this `FdWaker`. It lives in `core-term/src/io/` next to the existing
`epoll.rs`/`kqueue.rs` wrappers (it is terminal-side plumbing, not pixelflow).

---

## 3. Proposed Design

Three actors replace the three raw/half-raw threads. Naming follows the
troupe idiom:

```
                 ┌──────────────────────────────────────────────────┐
                 │                    PTY Troupe                     │
                 │                                                   │
   buffers (D)   │  ┌────────────┐   bytes (D)   ┌──────────────┐   │  AnsiCommands (D)
  ┌──────────────┼─▶│ PtyReader  │──────────────▶│  PtyParser   │───┼──▶ TerminalApp
  │              │  │ park: epoll│               │ (unchanged   │   │
  │              │  │ {pty,waker}│               │  core logic) │   │
  │              │  └─────┬──────┘               └──────┬───────┘   │
  │              │        │ ChildExited (D→app)         │           │
  └──────────────┼────────┼─────────────────────────────┘           │
                 │        ▼                                          │
                 │  ┌────────────┐                                   │
 TerminalApp ────┼─▶│ PtyWriter  │  D: Write(bytes)                  │
                 │  │ park: epoll│  C: Resize                        │
                 │  │ {out,waker}│  owns primary PTY FD              │
                 │  └────────────┘                                   │
                 └──────────────────────────────────────────────────┘
```

### 3.1 Lane Assignments

| Actor | Data (D) | Control (C) | Management (M) |
|-------|----------|-------------|----------------|
| `PtyReader` | `RecycledBuf(Vec<u8>)` — returned buffers from parser | `ReaderControl::{Pause, Resume}` (flow control; v2) | `NoManagement` |
| `PtyParser` | `Vec<u8>` — raw bytes from reader | `ParserControl::Reset` — clear escape-sequence state | `NoManagement` |
| `PtyWriter` | `Vec<u8>` — bytes for the shell | `WriterControl::Resize(Resize)` | `NoManagement` |

`PtyCommand` and the `mpsc::sync_channel` it rode on are deleted; the lane
split *is* the enum. `TerminalApp.pty_tx` becomes
`ActorHandle<Vec<u8>, WriterControl, NoManagement>`:

```rust
// before
self.pty_tx.send(PtyCommand::Write(bytes))
self.pty_tx.send(PtyCommand::Resize(Resize { cols, rows }))
// after
self.pty_writer.send(Message::Data(bytes))
self.pty_writer.send(Message::Control(WriterControl::Resize(Resize { cols, rows })))
```

Resize now genuinely preempts queued paste data — the scheduler drains Control
before Data by construction.

### 3.2 PtyReader

State: cloned PTY FD, `EventMonitor` (epoll/kqueue), `FdWaker` fd, a fixed
pool of `N` buffers (e.g. 8 × 4 KiB, allocated at spawn), `saw_eof` flag,
and one producer `ActorHandle` each for the parser (bytes) and the app
(`ChildExited`).

- `handle_data(RecycledBuf)`: push buffer back into the pool.
- `handle_control(Pause/Resume)`: toggle EPOLLIN interest (v2; enables
  scrollback-time flow control without touching the shell).
- `park(_)`:
  1. If pool is empty → `epoll_wait({waker})` only. Starvation of buffers
     must not busy-spin on a readable PTY; the kernel PTY buffer holds the
     data and blocks the shell — that is the backpressure working.
  2. Else → `epoll_wait({pty: EPOLLIN, waker})`.
  3. On waker ready: drain the eventfd, return `Busy` (scheduler will drain
     lanes and call `park` again).
  4. On pty ready: pop a pool buffer, `read()` (nonblocking FD). Loop up to a
     small burst (bounded by pool size). Send each to parser via Data lane.
     Return `Busy`.
  5. On `read() == 0` (EOF) or fatal error: send `TerminalData::ChildExited`
     to the app on its dedicated handle, set `saw_eof`, deregister the PTY fd,
     and return `Idle` thereafter (actor drains until `Shutdown` arrives).

The reader **never allocates after spawn** and **never reads without a buffer**
— fixing deficiency #7 and making the pool the explicit flow-control token,
exactly like the vsync token bucket elsewhere in the codebase.

### 3.3 PtyParser

Unchanged in substance — it is already a correct actor. Changes:

- `recycler_tx: Sender<Vec<u8>>` → `reader: ActorHandle<RecycledBuf, …>`;
  recycling becomes `reader.send(Message::Data(RecycledBuf(buf)))`.
- `NoControl` → `ParserControl::Reset` (used on child restart under
  supervision, and handy for tests).
- `Box<dyn PtySender>` stays for now (it decouples spawn order); collapsing it
  to a plain `ActorHandle` is a follow-up once troupe wiring lands (§5).

### 3.4 PtyWriter

State: primary `NixPty` (RAII — drop still closes the FD and kills the shell),
`EventMonitor`, `FdWaker` fd, `pending: VecDeque<Vec<u8>>` plus a cursor into
the front buffer.

- `handle_data(bytes)`: append to `pending`; try an immediate nonblocking
  flush (common case: empty queue, write succeeds, zero extra latency).
- `handle_control(Resize)`: `ioctl(TIOCSWINSZ)` immediately. Never queued
  behind data.
- `park(_)`:
  - If `pending` empty → `epoll_wait({waker})`; on wake return `Busy`.
  - Else → `epoll_wait({pty: EPOLLOUT, waker})`; on writable, flush as much
    as the kernel accepts (`WouldBlock` → keep remainder), return `Busy`.

This removes the `write_all` wedge (deficiency #3): a stalled shell stalls
only the data queue, while Resize, Shutdown, and wakes keep flowing.

Note: the primary FD and the reader's dup share one file description, so
setting `O_NONBLOCK` affects both. The reader already tolerates `WouldBlock`
(it treats it as a spurious wakeup) and becomes strictly more correct with a
nonblocking FD under edge-style polling.

### 3.5 Shutdown

Drop-order choreography is replaced by explicit signals:

1. Owner sends `Message::Shutdown` to writer, reader, parser (any order —
   each exit is now self-contained; reverse-spawn order retained for tidiness).
2. `FdWaker` interrupts any `park()` blocked in epoll/kqueue — this is the
   piece that was impossible before: **the reader can now be told to exit
   without yanking its FD.**
3. Shutdown modes: writer uses `ShutdownMode::DrainAll { timeout: 250ms }`
   (flush queued keystrokes before closing the FD, bounded); parser uses
   `DrainAll` (don't drop already-read output); reader uses `Immediate`.
4. `EventMonitorActor`'s `Drop` keeps joining threads, but its body becomes
   "send Shutdown × 3, join × 3."

EOF path (shell exits): reader sends `ChildExited` → app handles it as Data →
app sends `AppManagement::Quit` to the engine → engine troupe unwinds → main
drops `EventMonitorActor` → clean shutdown. This turns today's dead-window
hang into the correct close behavior, and gives `TerminalApp` the hook that
`EngineEventControl::CloseRequested` (currently `unimplemented!`) also needs.

```rust
// messages.rs
pub enum TerminalData {
    Engine(EngineEventData),
    Pty(Vec<AnsiCommand>),
    ChildExited { status: Option<i32> },   // new
}
```

---

## 4. What Gets Deleted

- `mpsc::channel` recycler (reader Data lane replaces it)
- `mpsc::sync_channel<PtyCommand>` + `PtyCommand` enum (writer lanes replace it)
- `NoControl` (parser gains real control), reader/writer raw `thread::Builder` loops
- ~40 lines of Drop-ordering contract documentation, because the invariant it
  guarded no longer exists
- The `eprintln!("DEBUG: Read thread started!")` while we're in there

---

## 5. Wiring and Bootstrap

### Phase 1 — hand-wired (recommended first step)

Keep `EventMonitorActor::spawn(pty, cmd_tx, …)` as the public entry point, but
build the three schedulers with `ActorBuilder` inside it. The signature
changes only in that `pty_cmd_rx: Receiver<PtyCommand>` disappears; instead
`spawn` **returns** the writer's `ActorHandle` (created via
`writer_builder.add_producer()`), which `main.rs` threads into
`TerminalAppParams` in place of `pty_tx`. No troupe machinery yet; the diff is
confined to `io/` plus the `pty_tx` type in `terminal_app.rs`/`main.rs`.

Reader needs a producer handle to the app for `ChildExited`; `spawn_terminal_app`
already mints per-consumer handles (`builder.add_producer()`), so this is one
more line.

### Phase 2 — `troupe!` (optional, once Phase 1 settles)

```rust
troupe! {
    reader: PtyReader [expose],   // expose: needed? only if app pauses it
    parser: PtyParser,
    writer: PtyWriter [expose],   // app sends writes/resizes
}
```

Two obstacles, both with existing idioms:

- `TroupeActor::new(dir)` takes only a directory, but these actors need the
  `NixPty` and buffer config. `VsyncActor` already established the pattern:
  construct empty, then deliver resources via a Management message
  (`WriterMgmt::Bind(NixPty)`, `ReaderMgmt::Bind { pty, waker }`). `NixPty` is
  `Send`, so it can ride a message.
- The troupe's `play()` blocks on scoped threads, and main already blocks on
  the engine troupe. The PTY troupe therefore nests: `s.spawn(|| pty_troupe.play())`
  — the documented two-phase nesting pattern in `actor-scheduler/src/lib.rs`.

### Phase 3 — supervision (future)

With Phase 2 in place, the parser (stateless between `Reset`s) is the ideal
first Kubelet-managed pod: `RestartPolicy::OnFailure`, `ServiceHandle` from
the reader, `ParserControl::Reset` on restart. Reader/writer own FDs and are
*not* restartable in place; their failure should escalate to `ChildExited`
semantics instead. This aligns with `actor-scheduler-supervisor-migration.md`
and gives the supervision work a real consumer beyond tests.

---

## 6. Performance Notes

- Hot path is unchanged: one epoll_wait + one read per wakeup, buffer handoff
  by move, SPSC rings underneath. The doorbell `try_send` + eventfd write per
  message batch is the only addition, and the reader already paid an
  equivalent cost (`ActorHandle::send` → doorbell) today.
- Burst limits give the reader/parser the same flood protection the old
  `burst: 10` setting did — keep 10/64 as the starting parameters.
- Writer's immediate-flush-in-`handle_data` keeps keystroke latency identical
  to the blocking version when the queue is empty (the overwhelming case).

## 7. Testing

- `FdWaker`: unit test — blocked `epoll_wait` returns within N ms of `wake()`.
- Reader pool starvation: parser that withholds buffers; assert reader stops
  reading (no allocation growth) and resumes on recycle.
- Writer preemption: fill PTY buffer via a non-draining child (`sleep`), queue
  1 MiB of writes, send Resize; assert `TIOCSWINSZ` lands before the queue drains.
- EOF: spawn `/bin/true`; assert app receives `ChildExited`.
- Shutdown: property that `Shutdown` × 3 joins all threads without touching
  the FD first (the thing Drop-choreography could never test).
- Existing `pty_tests.rs` and `write_thread` tests port mechanically.

## 8. Open Questions / Future Work

1. **Fd-backed doorbell in actor-scheduler.** The `FdWaker` pattern (also used
   by X11/Cocoa) suggests the scheduler could offer an optional fd-based
   doorbell so `park()`-blocking actors don't each hand-roll the waker. Worth
   doing only after a third consumer appears; keep it out of this change.
2. **`ReaderControl::Pause` policy.** Wire it to scrollback viewing? Deferred —
   the lane exists either way, which is the point.
3. **Child exit status.** Reader sees EOF, not the wait status. `NixPty`
   already tracks the child PID; a nonblocking `waitpid` in the EOF path can
   populate `status`. v1 may ship `status: None`.
