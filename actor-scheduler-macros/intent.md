# actor-scheduler-macros

**Is.** The static topology of the effects outside the language (DENOTATION
§11; `actor-scheduler` is where they run). Two proc macros, each a function
from a declaration to code against `actor_scheduler`: `troupe!` takes actors
and yields the troupe (a `Directory` per actor holding a dedicated SPSC handle
to every actor, `ExposedHandles`, `Wakers`, the threads); `ports!` takes an
actor's output ports and yields its output word and its wiring.

## Owns
- The shape of a troupe: which actors exist, which runs on the calling thread
  (`[main]`), which hand a handle to a parent (`[expose]`), which get a
  `WakeHandler` slot (`[waker]`; `[main]` implies one).
- The shape of an output word: one `Option<T>` per port, one `SpscSender<T>`
  per deliverable port, and a `flush` that sends every one of them.
- Shutdown order (reverse declaration) and `TROUPE_DEFAULT_BUFFER_SIZE`.

## Never
- No policy. Lanes, backoff, burst limits and drain order stay in
  `actor-scheduler`; generated code only calls `ActorBuilder`, `add_producer`,
  `build`, `ActorScheduler::run` and `mealy::send_port`.
- No application type, no dependency (`Cargo.toml` has none); every
  generated path starts `::actor_scheduler::`.
- No expansion of a troupe without exactly one `[main]`, or of `ports!` with
  no port or two `[self]` ports (the continuation slot holds one message).
- No droppable port: `[drop]` is a named expansion-time panic. A `[self]`
  port has no wiring field and never goes through `flush`.
- No port left out of `flush`: every deliverable port is sent via
  `mealy::send_port`, so none vanishes unsent.

## Contract
`src/lib.rs`. `troupe!` generates `Directory`, `ExposedHandles`, `Wakers`, `run`
and `Troupe` (`new`, `new_with_wakers`, `new_with_waker`, `exposed`, `play`).
`ports!` generates `{Actor}Out`, `{Actor}Wiring`, `impl Wiring`, and
`take_continuation` for a `[self]` port. Callers use `actor_scheduler::{troupe, ports}`.

## Enforced by
- `actor-scheduler/tests/simple_type.rs`, `generic_type.rs`: a `troupe!`
  expansion compiles against the real traits, generic actors included.
- `actor-scheduler/tests/ports.rs`: generated flush parks on a full target
  and keeps the message; a `[self]` port has no wiring field.
- Types: `Troupe::play(self)` consumes the builders, so no producer follows
  it; drift from `ActorBuilder`, `Wiring` or `TroupeActor` fails to compile
  in `core-term/src/io/event_monitor_actor/mod.rs` and
  `pixelflow-runtime/src/engine_troupe.rs`.
- Refusals: no `trybuild`, no `should_panic` (debt).

## Behind
- `troupe!` drops any stray token or punctuation it does not recognise, and
  duplicate actor names reach rustc as duplicate fields (2026-10-15; no plan
  or Linear issue).
- Generated `Directory::shutdown` discards each send with `let _ =`, which
  `let_underscore_must_use` denies in hand-written code.
- The `troupe!` doc says `exposed(&self)` (code: `&mut self`) and omits
  `Wakers` and `[waker]`.

## Decided in
`docs/archive/designs/pty-actor-troupe.md` (`[waker]`, Done);
`docs/runtime/actor-scheduler-mealy-transducer.md` §7.2 (`ports!`; Draft).
