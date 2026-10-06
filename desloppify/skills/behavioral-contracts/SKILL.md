---
name: behavioral-contracts
description: How modules in this workspace are shaped — a trait-based contract in the module root, a private implementation behind it — with the worked examples in core-term and pixelflow-runtime.
---

# Behavioral contracts

A module is a contract and an implementation, and the file layout says which
is which.

## The module root is the contract

The root file — `mod.rs`, or `lib.rs` for the crate — declares everything any
other module may use, and little else:

- **Traits** that state the module's behavior, each with rustdoc a test could
  be written from: what it denotes, its preconditions and postconditions, and
  its guarantees (order, determinism, completeness, memory bounds, what
  happens on bad input). "Handles input", "may", "for example" and "returns
  `None` if something goes wrong" are not contracts.
- **The types those traits mention**, defined here or in an implementation
  file and re-exported with `pub use`, so the root lists the whole surface.
- **The state type** of the implementation, with `pub(super)` fields, and its
  constructor.
- **The module doc**: topology, lifecycle, blocking model — whatever a caller
  must know that a signature can't say.
- `mod` declarations (private) and `pub use`. No algorithm bodies.

## Implementation files expose nothing

Every other file in the module is implementation. Its items are private or
`pub(super)` — visible to the root, which decides what escapes. A file whose
parent is `lib.rs`/`main.rs` that has anything to offer the crate is a module
with no contract yet: it becomes a directory with a `mod.rs`.

## Tight contracts

A tight contract's signature refuses wrong shapes on its own:

- **One closed input.** `TerminalInterface::interpret_input(EmulatorInput)`:
  every way to affect the emulator is a variant of one enum — its instruction
  set — so the contract is a total function over a known domain.
- **Effects returned, not performed.** `interpret_input` returns an
  `EmulatorAction` instead of writing to the PTY; `AnsiParser::process_bytes`
  returns an `AnsiBatch`; `PlatformOps` steps push into a `DriverOut`. The
  implementation can then be driven and observed with no IO, no scheduler and
  no channels — which is what makes the contract testable.
- **No second door.** Behavior goes through the trait. A public inherent
  method that duplicates or bypasses it (a delegate the trait impl forwards
  to) lets callers depend on the implementation instead of the contract.
  Constructors are the exception.
- **No loose returns.** An `Option` whose `None` the doc doesn't pin down, a
  `bool` mode flag, or `&mut self` on what claims to be a query all admit
  behaviors the contract never stated.

## Tests only ever see the exported API

Every test is a black-box test. Even when what it checks is a specific
internal behavior — the cursor stopping at the margin, the AIMD rate halving
— it drives the module's exported interface and observes what that interface
returns: feed `EmulatorInput`s and read the snapshot or the returned
`EmulatorAction`s; call `RateLimiter::wait` and read the waits. Never a
`pub(super)` field, a private helper, or a `#[cfg(test)] mod tests` inside an
implementation file that can see them.

When a behavior cannot be observed or controlled through the interface, that
is a finding about the interface, not a license to reach in: make the input
explicit (time becomes a `Clock` the constructor takes; randomness a seed)
or the effect a returned value. A test generic over the trait
(`fn contract<P: AnsiParser>(p: P)`) holds every implementation to it.

## Worked examples

- `core-term/src/ansi/`: `AnsiParser` with its full contract in `mod.rs`;
  `lexer` and `parser` private; `tests/ansi_parser_message_tests.rs` uses only
  the trait and `AnsiProcessor`.
- `core-term/src/term/`: `TerminalInterface` + `EmulatorInput` in `term/mod.rs`;
  `TerminalEmulator`'s state in `emulator/mod.rs`, its behavior split by concern
  across private files as `pub(super)` methods.
- `core-term/src/io/event_monitor_actor/`: the PTY troupe's topology and
  lifecycle in the `mod.rs` doc; reader, parser and writer actors are
  `pub(super)`.
- `pixelflow-runtime/src/display/ops.rs`: `PlatformOps` returns effects via
  `DriverOut` (the shape is right; the trait should live in `display/mod.rs`).
