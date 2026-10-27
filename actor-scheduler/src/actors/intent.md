# actor-scheduler/src/actors

**Is.** Ready-made actors for the runtime outside the language, the part of an
application that waits on the world; DENOTATION has no section for it (§11 lists
`Write`/`Seq` as "not a concept" of the terms, and effects run in this crate).
Today one: `Timer`, a thread that waits, fires a callback and waits again, so an
actor that cannot wait on time is sent a message.

## Owns
- The one place in the crate that waits on a clock. A scheduler blocks on its
  doorbell and an actor may not block, so time enters as a message a `Timer`
  sends, never as a timeout inside a handler.
- A timer's lifetime: it ends at `Timer::stop`, at `ControlFlow::Break` from the
  callback, or at drop, whichever is first. Only `stop` is synchronous: it joins
  the thread, so no fire follows its return.
- The schedule (`Schedule::Every`, `Schedule::After`). `Every` waits between
  fires, not between fire starts, so a slow callback delays the next fire.

## Never
- No domain concept, and no crate type: `timer.rs` imports `std` only. It names
  no `ActorHandle`, `Message`, lane or application type; the callback owns the send.
- No re-entrant callback: one thread runs `on_fire`, and the next wait begins
  after it returns.
- No send of its own, so no lane choice and no send result to discard; what the
  callback sends, and on which lane, is its caller's.
- No error from `set_interval` or `stop` for a timer that already exited; both
  are moot then and handled once in `Timer::command`. `stop` panics only if the
  callback panicked; `spawn` panics only if the OS refuses a thread.

## Contract
`mod.rs` declares `pub mod timer` and re-exports `Schedule` (`Every(Duration)`,
`After(Duration)`) and `Timer` (`spawn(name, schedule, on_fire)`, `set_interval`,
effective on the next wait, and `stop(self)`). Callers today:
`pixelflow-runtime/src/engine_troupe.rs` and `vsync_actor.rs` (vsync clock).

## Enforced by
- Types: `Timer::stop` takes `self`, so no `Timer` is used after it; the callback returns
  `ControlFlow<()>`, so "keep going" is not a `bool` read the wrong way round; `#[must_use]` on
  `Timer` refuses a handle bound to nothing.
- The in-file tests in `timer.rs` (`a_one_shot_timer_fires_exactly_once`,
  `breaking_from_the_callback_stops_a_repeating_timer`,
  `stop_is_safe_before_the_first_fire`, `dropping_a_timer_stops_its_thread`); none
  asserts elapsed time. `postsubmit-flake-detection.yaml` runs the suite 5 times.
- desloppify `actor-lane-choice` (paths include `actor-scheduler/**` and
  `pixelflow-runtime/**`) judges the sends inside the callbacks.

## Behind
- Nothing checks that `actors/` stays `std`-only and free of domain concepts; the
  `mod.rs` doc and review are the guard (2026-10-15; no plan or Linear issue).
- A `Timer` is one OS thread per clock. The Draft transducer design puts timers
  in a reactor on the System lane (§3.3); no reactor exists in
  `actor-scheduler/src` (2026-10-15; that design is open).
- The ontology has no entry for `Timer` (2026-10-15; no plan or Linear issue).

## Decided in
`docs/runtime/actor-scheduler-mealy-transducer.md` (Draft; §3.3, and its note
that the vsync clock's explicit `Tick` messages settle where ticks belong);
`docs/archive/designs/actor-scheduler-backpressure.md` (Accepted; the lanes a callback
sends on).
