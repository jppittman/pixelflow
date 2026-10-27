# core-term/src/io

**Is.** The terminal's boundary with the operating system: where this process's
effects on a shell live, outside the language (DENOTATION §5: the language has
no effect and no memory read). It creates the PTY and the child on its slave
side, waits on file descriptors, and turns a blocked poll into one the actor
scheduler can interrupt. It moves bytes and a window size; what they mean is
`ansi`'s and `term`'s.

## Owns
- The PTY and child: `NixPty` holds the master (`Arc<OwnedFd>`), the child pid,
  and the child's fate (SIGHUP on drop, SIGWINCH on resize).
- How the child starts: `posix_spawnp`, `POSIX_SPAWN_SETSID`, the slave opened
  as fd 0 to acquire the controlling terminal, default signal dispositions.
- The platform poll (`EventMonitor` over epoll or kqueue) and the `Event` /
  `EventFlags` names both answer to, so a poll loop is written once.
- The wake protocol between the scheduler's fd-less doorbell and a blocked
  poll (`FdWaker::arm` / `wake` / `disarm`), and its SeqCst ordering.

## Never
- No `fork`, `forkpty` or `pre_exec`: a threaded process spawns by
  `posix_spawn`, with all heap work in the parent before the call.
- No `epoll_*` or `kevent` call outside `epoll.rs` and `kqueue.rs`.
- No terminal meaning: `pty.rs`, `epoll.rs`, `kqueue.rs` and `waker.rs` import
  no `ansi`, `term` or `surface`; only `traits.rs` names `AnsiBatch`.
- No second spawner and no concurrent `spawn_with_config` (`openpty(3)` is not
  thread-safe on macOS): `main` spawns the one PTY before the troupe's threads.
- No drain of the waker pipe outside `disarm()`; no blocking poll without
  `arm()`, and a `false` from `arm()` returns `Busy`.

## Contract
`mod.rs`: `Resize`; `pty` (`PtyConfig`, `PtyChannel`, `NixPty`); `traits`
(`EventSource`, `PtySender`); `waker` (`FdWaker`); `event` (`EventMonitor`,
`Event`, `EventFlags`, aliasing `epoll` or `kqueue`); `event_monitor_actor`
(the three-actor pipeline, its own `intent.md`). `spawn_linux.rs` and
`spawn_macos.rs` are selected once at `mod spawn`.

## Enforced by
- Types: `OwnedFd` closes the master on the last drop; `FdWaker::arm` is
  `#[must_use]`; a clone has `child_pid: None`, so only the original signals.
- `io/pty_tests.rs`: controlling terminal, default SIGPIPE, an `Err` for a
  missing command or unenterable directory, child ended on drop.
- `waker.rs` tests: a wake unblocks an armed monitor; a disarmed wake writes
  no byte and fails the next `arm`.
- `scripts/check-cfg-encapsulation.sh` (CI job `cfg-encapsulation`): the
  platform split is files; `io`'s old violations are baselined in
  `scripts/cfg_encapsulation_baseline.txt`.

## Behind
- The single-spawner rule is a caller contract: `spawn_with_config` does not
  prevent concurrent calls, so callers must serialize PTY creation.

## Decided in
`docs/archive/designs/pty-actor-troupe.md`,
`docs/process/2026-10-15-intent-lives-with-the-module.md`.
