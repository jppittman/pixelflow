# The runtime and actors

### Priority lane (doorbell). Homonym of the SIMD lane

- **Is:** three lanes, Control > Management > Data, with Shutdown above them
  in the scheduler. "Control/Management prioritize latency over throughput."
  Senders publish into per-producer SPSC lanes and ring a doorbell. The
  receiver drains the lanes in priority order, with burst limits.
- **Is not:** unbounded or never-blocking on Control. "Control creates
  backpressure by timing out senders who are too aggressive. If the timeout
  exceeds a threshold, an error is returned." A SIMD lane.
- **Follows:** a send's result is never ignored (`unused_must_use = "deny"`).
  Bulk or continuous data goes on Data, and a latency-critical event never
  does. In the Mealy design "every port parks" (a sender parks on a full
  target ring and resumes when it is not full), and the
  `Delivery::{Blocking, Droppable}` split is deleted.
- **Lives:** `actor_scheduler::Message<D, C, M>`, `ActorHandle::send`,
  `ActorScheduler` (`actor-scheduler/src/lib.rs`), `SendError::Timeout`
  (`actor-scheduler/src/error.rs`), `doorbell.rs`, `spsc.rs`; CLAUDE.md
  "Actor Model"; `docs/designs/actor-scheduler-mealy-transducer.md`;
  `desloppify/rules/actor-lane-choice.json`. Today
  `.claude/agents/actor-scheduler.md` says Control "Never" blocks and has an
  "Unbounded buffer", which contradicts CLAUDE.md and the crate's
  `SendError::Timeout`. The crate's own `Message` doc table says Control and
  Management blocking is "Unlimited" and Data "may drop if buffer
  overflows", and `ActorHandle::send`'s doc says it "Returns `Err` only if
  the receiver has been dropped"; all three contradict the timeout.

### Actor and troupe

- **Is:** an actor handles messages and OS status:
  `handle_os(SystemStatus) -> Result<ActorStatus, HandlerError>`. A troupe is
  a group of actors with a shared directory and two-phase initialization,
  declared with `troupe!`.
- **Is not:** a loop around `actor.park(hint)` with `ActorStatus` returned
  from park. That is the stale description in
  `.claude/agents/actor-scheduler.md`.
- **Follows:** input is separated from rendering. An idle receiver parks on
  its doorbell (the actor sense of "park").
- **Lives:** `Actor`, `TroupeActor` (`actor-scheduler/src/lib.rs`), `troupe!`
  (`actor-scheduler-macros/src/lib.rs`). Today the `TroupeActor` doc example
  in `actor-scheduler/src/lib.rs` still writes
  `fn handle_os(&mut self, status: SystemStatus) -> ActorStatus` and
  handlers returning `()`, against the trait's `HandlerResult` and
  `Result<ActorStatus, HandlerError>`.

### Driver (PlatformOps)

- **Is:** the display backend seam. "Outbound events are returned via
  `DriverOut` rather than sent, so an implementation can be driven and
  observed with no engine, no scheduler, and no channels in the loop." The
  platform runs on the main thread, which Cocoa requires.
- **Is not:** pure under Wayland: "Does not survive: the driver as a function
  of its inputs alone." Not something a spawned thread may run on macOS.
- **Follows:** under Wayland, `Blitted` becomes asynchronous, so the driver
  needs two buffers. The poll is hand-rolled over the connection fd and a
  wake eventfd. The staging field is a `DriverOut`, not a second type. No
  per-frame heap allocation (ping-pong buffers).
- **Lives:** `PlatformOps`, `DriverOut`
  (`pixelflow-runtime/src/display/ops.rs`), `display/drivers/`,
  `platform/{linux,macos}/`; `docs/plans/2026-09-03-wayland-driver.md`;
  CLAUDE.md "Platform Notes". The Wayland driver is unbuilt.
