# `openpty(3)` is not thread-safe on macOS

**Status:** fixed 2026-07-21 (single-threaded contract on
`NixPty::spawn_with_config`, upheld in tests by `OPENPTY_TEST_LOCK` in
`core-term/src/io/pty.rs`).
**Impact:** intermittent PTY spawn failures in the **test suite** — a flaky
`io::pty_tests::pty_child_gets_default_sigpipe` under `cargo test --workspace`.

**Not a live production race.** Production opens exactly one PTY per run:
`main.rs` calls `spawn_with_config` once, on the main thread, before the
troupe's threads are spawned, then hands the `NixPty` to `PtyTroupe::new`
and on to the actors via `Bind` messages. Every other call site is inside
`#[cfg(test)]`. No multi-window/multi-PTY design exists. The hazard is real
but only reachable from concurrent callers, which today means tests.

Unrelated to [the fork/malloc deadlock](2026-07-15-pty-fork-malloc-deadlock.md),
which was fixed by moving to `posix_spawn`. This is a second, independent PTY
hazard in the same spawn path.

## Symptom

```
thread 'io::pty_tests::pty_child_gets_default_sigpipe' panicked at
core-term/src/io/pty_tests.rs:230:
  Failed to spawn PTY: Failed to open PTY (nix::pty::openpty call)
Caused by:
    UnknownErrno: Unknown errno
```

Note *where* it fails: line 230 is the `spawn_with_config(...).expect(...)`,
not the read-with-timeout on line 231. This is **not** a timeout — the test's
5-second `READ_TIMEOUT` is never reached, and the SIGPIPE behaviour the test
asserts is not implicated at all. `openpty` itself fails before a child is
ever spawned.

Reproduced under `cargo test --workspace` (1 failure in 15 runs); passes
reliably alone or with `-p core-term --lib`, because the trigger is thread
oversubscription across concurrently-running test binaries.

## Measurement

Standalone C harness, `openpty` in a loop across N threads on a 12-core
machine, counting failures:

| Mode | Threads | Runs | Total calls | Failures |
|------|---------|------|-------------|----------|
| unserialized | 8  | 3 | 9,600  | **0** |
| unserialized | 48 | 3 | 57,600 | **107** |
| serialized (mutex) | 48 | 3 | 57,600 | **0** |

The race only appears once threads meaningfully oversubscribe the cores —
which is exactly what `cargo test --workspace` does (many test binaries ×
many threads each). At 8 threads on 12 cores it does not reproduce, which is
why the flake looked load-dependent and mysterious.

## Why the errno is `UnknownErrno`

The failing calls report errno **-6** — a *negative* value, not a valid errno.
`nix`'s `Errno::from_raw` has no arm for it and falls through to
`_ => UnknownErrno`. So `UnknownErrno` here does **not** mean "errno was 0";
it means libc handed back a corrupted value.

This also rules out the plausible-but-wrong theories:

- **Not pty exhaustion.** `kern.tty.ptmx_max` is 511; exhausting it fails
  cleanly with `ENXIO` (6), a *valid* errno, and the machine had 5 ptys in use.
- **Not fd exhaustion.** `ulimit -n` is 1048576.
- **Not the fork/malloc deadlock.** That path is gone; spawn uses `posix_spawn`.
- **Not a `posix_spawn` signal-attribute race.** `POSIX_SPAWN_SETSIGDEF` +
  `POSIX_SPAWN_SETSIGMASK` are applied atomically per spawn, and the failure
  happens before `posix_spawn` is reached.

## Which internal call races is unconfirmed

`ptsname(3)` is the obvious suspect — `man ptsname` states it "is not
guaranteed to be reentrant or thread safe" (process-global static buffer),
and BSD-derived `openpty` implementations call it internally, whereas glibc's
was fixed to use `ptsname_r`.

But that story does not fully fit: a corrupted replica pathname would make
the subsequent `open()` fail with a *valid* errno (`ENOENT`/`ENXIO`), not -6.
A negative errno looks more like an internal error-propagation path in
libutil writing a negated return code. **The fix is justified by the
measurement, not by the mechanism.**

Note `nix::unistd::ttyname` (used later in `posix_spawn_child`) is *not* a
second instance of this bug — it uses the reentrant `ttyname_r`.

## Fix

`spawn_with_config` documents a **single-threaded contract**, and production
satisfies it structurally: one PTY, spawned on the main thread before the
troupe's threads exist, handed to `PtyTroupe::new` and delivered to the
actors as a `Bind` message. No production lock.

The contract is upheld inside the test binary by a `#[cfg(test)]`
`OPENPTY_TEST_LOCK`, because libtest runs the PTY-spawning tests as parallel
threads. It is gated inside `spawn_with_config` rather than sprinkled across
the tests: spawn sites live in four different test modules (`io::pty_tests`,
`io::event_monitor_actor::{mod, writer}`, `terminal_app`), and a lock a new
test can forget to take would let the flake back in silently. No integration
test spawns a PTY, which is what makes `cfg(test)` a sufficient boundary — if
one ever does, it falls outside this guard.

Lock poisoning is `.expect()`-ed rather than recovered from: a panic inside
that critical section means the PTY layer is in an unknown state, and per
project policy it must fail loudly.

Deliberately *not* done: widening the test's timeout or retrying the spawn.
Both would have masked a genuine libc thread-safety bug behind a timing knob.

### Considered and rejected

- **A production `Mutex`.** Rejected: it is machinery for a race that cannot
  occur today, and a `Mutex<()>` guarding no data enforces the invariant by
  convention anyway.
- **Owning PTY spawning in an actor** (the tidiest answer — an OS-level
  operation with a single owner). Rejected for now as a layering problem:
  the natural home is the engine, but `pixelflow-runtime` is deliberately
  generic and a PTY is terminal-specific.

If a second PTY is ever needed (tabs, splits), the fix is to give spawning a
single owner — not to make `spawn_with_config` reentrant.
