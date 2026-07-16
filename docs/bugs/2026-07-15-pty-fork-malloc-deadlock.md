# PTY fork/malloc deadlock in core-term tests

**Status:** diagnosed 2026-07-15 (live process sample); not yet fixed.
**Impact:** `cargo test --workspace` hangs *indefinitely and intermittently* on
macOS — two multi-hour hangs traced to this (one ran 13 hours at 0.0% CPU).
CI never sees it. Local single-threaded runs never see it.

## Symptom

The `core_term` unit-test binary wedges after the `surface::manifold` tests
with ~0 CPU. The parent `cargo test` sits idle forever. Which test wedges is
racy; the sampled instance was
`io::event_monitor_actor::write_thread::tests::write_thread_handles_resize_command`.

## Evidence (process sample, 2026-07-15)

Test thread:

```
write_thread_handles_resize_command (mod.rs:94)
  → core_term::io::pty::NixPty::spawn_with_config (pty.rs:172)
    → alloc::sync::Arc<T>::new
      → _xzm_xzone_malloc_freelist_outlined (libsystem_malloc)
        → _xzm_fork_lock_wait          ← BLOCKED on the malloc fork lock
```

Main thread (the libtest harness), simultaneously:

```
test::run_tests::get_timed_out_tests
  → RawVec::grow_one → finish_grow
    → mfm_alloc (libsystem_malloc)
      → _os_unfair_lock_lock_slow      ← BLOCKED on a malloc zone lock
        → __ulock_wait2
```

Every thread that touches malloc is stuck; nothing is running. Classic
deadlock signature, not slowness.

## Mechanism

`cargo test` (libtest) runs tests as **threads inside one process**. A PTY
spawn (`NixPty::spawn_with_config`) does a `fork()` (via `forkpty`/openpty +
fork). On macOS, libmalloc registers atfork hooks that take **all malloc zone
locks** around the fork so the child's heap is consistent
(`_malloc_fork_prepare` / `_parent` / `_child`; the xzone allocator's variant
is the `_xzm_fork_lock` seen in the stack).

The race: while one test thread is inside the fork's prepare/parent window,
other test threads are allocating. Lock-order/state gets wedged —
allocating threads pile onto zone locks held for the fork, and the forking
thread waits on `_xzm_fork_lock` that can't make progress. Result: the whole
process freezes at 0% CPU. POSIX is explicit that only async-signal-safe
work is allowed between `fork` and `exec` in a multithreaded process; heap
allocation anywhere near a concurrent fork is exactly the hazard zone.

Two independent hangs in a row on 2026-07-15 vs. clean runs on 2026-07-08:
it's a timing race, so it appears/disappears with build layout and machine
load. Nothing changed in the PTY code between those dates.

## Why CI is immune

CI runs **cargo-nextest** (`.github/workflows/rust.yaml`), which executes
each test in its **own process**. A single-threaded (or nearly so) process
forking a PTY doesn't race anyone for malloc locks. Local
`cargo test -p core-term -- --test-threads=1` is green for the same reason.

So: green CI + hanging local workspace runs is the expected signature of
this bug, not evidence of absence.

## Remedies, ranked (check against the new PTY design)

1. **Don't fork from a threaded process at all — use `posix_spawn`.**
   `posix_spawn` on macOS avoids running user-space code (and malloc atfork
   hooks in the child) between fork and exec; Apple recommends it for
   threaded processes. If the PTY overhaul controls the spawn path, this is
   the structural fix: `openpty` + `posix_spawn` with
   `POSIX_SPAWN_SETSID` + `TIOCSCTTY` handled via file actions, instead of
   `forkpty`. (Note `forkpty` itself is documented as unsafe-after-fork in
   threaded processes for exactly this reason.)
2. **If `fork` stays: make the fork→exec window allocation-free.** No
   `Arc::new`, no `format!`, no logging between `fork()` and `execvp` in the
   child, and minimize the window in the parent. The sampled stack shows an
   `Arc::new` allocation *inside* `spawn_with_config` adjacent to the fork —
   whatever the new design does, keep heap work strictly before the fork.
   This shrinks the window but does not close it (the parent-side atfork
   hooks still serialize against other threads' mallocs).
3. **Serialize PTY-spawning tests.** Mark them `#[serial]` (serial_test
   crate — weigh against suckless-deps rule) or move them to an integration
   test binary that runs single-threaded. Cheap, test-only, doesn't fix the
   production hazard: the real terminal also spawns PTYs from a process with
   threads (actor system), so the production spawn path has the same
   theoretical race if anything else allocates concurrently at spawn time.
4. **Adopt nextest locally** (`cargo nextest run --workspace`) and document
   it as the blessed local runner. Sidesteps the test hang entirely; does
   nothing for the production-path concern in (3).

Recommendation: (1) if the overhaul permits — it fixes tests *and* the
production spawn path; (4) as the immediate local-workflow patch regardless.

## Repro / verification

No deterministic repro (race). Best-effort: loop
`cargo test -p core-term` (default parallel threads) on a loaded machine;
hangs manifest as 0% CPU after the manifold tests. Verify a fix by the same
loop staying green, plus `sample <pid>` showing no `_xzm_fork_lock_wait`
during PTY test runs.
