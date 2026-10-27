# actor-scheduler

**Is.** The runtime outside the language: a cooperative scheduler for actors on
three priority lanes, Control > Management > Data. DENOTATION has no section for
it; §11 lists effects (`Write`/`Seq`) as "not a concept" of the terms, and this
crate is where the application's effects run.

## Owns
- The lanes: one SPSC ring per producer per lane (`spsc.rs`), drained
  round-robin by a sharded inbox (`sharded.rs`), sealed by `ActorBuilder::build`.
- The doorbell (`doorbell.rs`): a wake is a level, shutdown a latch, in one atomic
  word, so shutdown outranks wake by a bit test, not queue order.
- Send policy: Data yields until the ring has room; Control and Management
  spin, yield, then sleep with jittered backoff and return `SendError::Timeout`.
- Drain order, burst limits, the sleep decision (`SchedulerParams`).

## Never
- No application type in the API: the one workspace dependency is
  `actor-scheduler-macros`, and `actors/` names no domain concept.
- No unbounded lane: Control and Management rings are bounded
  (`control_mgmt_buffer_size`), so Control cannot carry volume.
- No sender in a `Transducer` step: it returns its output word, `Wiring`
  delivers it, a full target parks the rest. A green actor never blocks.
- No caller-chosen send policy (`ActorHandle::try_send` is `pub(crate)`) and no
  shutdown as a message: it sets the latch and no `handle_*` sees it.
- A handler `Err` panics the scheduler thread; so does a flush to a gone target.
- No producer after `build`, no `Clone` on `ActorHandle`, no `unsafe` outside `spsc.rs`.

## Contract
`src/lib.rs`: `Message<D, C, M>`, `Actor`, `ActorTypes`, `TroupeActor`,
`ActorBuilder`, `ActorHandle`, `Waker`, `ActorScheduler`, `DedicatedThread`,
`WakeHandler`, `ActorStatus`, `SystemStatus`, `SchedulerParams`, `SendError`,
`troupe!`, `ports!` re-exported. `mealy.rs`: `Transducer`, `Wiring`, `Node`.
`host.rs`: `Host`, `GreenThread`. `actors/`: `Timer`. An actor implements
`ActorTypes` and `Actor`, plus `TroupeActor` to join a `troupe!`.

## Enforced by
- Types: `ActorBuilder::build(self)` consumes the registry; `Transducer` has no
  sender in scope; `unused_must_use = "deny"` covers a send's `Result`.
- desloppify `actor-lane-choice` (its paths include `actor-scheduler/**`).
- `tests/priority_and_order.rs` (drain order), `tests/dedicated_thread.rs`
  (`a_flooded_lane_does_not_starve_shutdown`, a gone wiring target panics),
  `tests/stress_tests.rs` (`burst_limit_prevents_data_starvation`).
- `SchedulerParams::validate`; `postsubmit-flake-detection.yaml` (5 runs).

## Behind
- The `Message` lane table says Control and Management block "Unlimited" and Data
  "may drop"; `ActorHandle::send` fails "only if the receiver has been dropped".
  The code times out and Data never drops (2026-10-15; no plan or Linear issue).
- The `TroupeActor` doc example (handlers returning `()`) and the documented
  `SchedulerParams` defaults (32, 10, 1024) are stale; `DEFAULT` is 118, 36, 332.
- Two implementations of one priority contract: `ActorScheduler::handle_wake`
  and `mealy::Node`. `mealy.rs` says "Prototype", its design is `Draft`
  (2026-07-24), and `poll_once` leaves the public API once its test callers
  move to `Host`.

## Decided in
`docs/archive/designs/actor-scheduler-backpressure.md` (Accepted);
`docs/runtime/actor-scheduler-mealy-transducer.md` (Draft).
