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

- **Traits** that state the module's behavior — the API's *shape* — each with
  rustdoc a test could
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

## A trait is the shape, not necessarily a `trait`

A real `trait` earns its keep when there is more than one implementation (a
second backend, a scripted fake for tests). With exactly one, keep it simple:
the concrete type implements its methods directly, and the module root
documents the shape as a trait in its module docs, so the API still reads at
a glance:

```rust
//! # Contract
//!
//! ```ignore
//! /// Implemented by [`TerminalEmulator`].
//! trait Emulator {
//!     /// Applies `input`; returns what the caller must do, if anything.
//!     fn interpret_input(&mut self, input: EmulatorInput) -> Option<EmulatorAction>;
//!     /// The visible state, for the renderer.
//!     fn snapshot(&self) -> TerminalSnapshot;
//! }
//! ```
```

The documented shape is the contract with the same standard as a real trait:
every public method of the type appears in it with the same signature and a
doc a test could be written from, and nothing public is missing from it.
Because the compiler does not check it, it drifts unless something does —
that is what the rules are for. When a second implementation appears, the
documented trait becomes a real one and nothing else about the module moves.

## Implementation files expose nothing

Every other file in the module is implementation. Its items are private or
`pub(super)` — visible to the root, which decides what escapes — except the
methods the root's contract declares: the documented shape's methods may be
`pub` where they are implemented, because the root already lists them. A file whose
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

## Tests only ever see the production API

Every test is a black-box test of the production API: what the crate's real
callers use. Even when what it checks is a specific internal behavior — the
cursor stopping at the margin, the AIMD rate halving, a register reaching the
allocator — it drives that API and observes what it returns: feed
`EmulatorInput`s and read the snapshot or the returned `EmulatorAction`s; call
`RateLimiter::wait` and read the waits; compile a kernel and run it. Never a
`pub(super)` field, a private helper, or a `#[cfg(test)] mod tests` inside an
implementation file that can see them.

**No API exists for tests.** Nothing is made public, or `pub(crate)`, or
`#[cfg(test)]`-visible, so a test can reach it: no test hooks, accessors,
constructors, re-exports, mocks or `for_test` variants in the source tree.
The reasoning is one step deeper than "don't reach in": a piece of code
earns its place by what it does to the production API's output. If changing
it can change some output a real caller sees, the test observes that output.
If no production output can change, the code is dead or a no-op — delete it;
don't expose it so a test can watch it run. So "this can't be tested through
the interface" is never answered by widening the interface: it is answered by
finding the output the code affects, or by deleting the code.

Production decides the API. An accessor, recorder, counter, mock or fixture
exists only if production needs it; when production does, a test may use it
like any other caller. The one place a test may shape the API is dependency
injection, when we choose it: a dependency the code needs either way — a `Clock`, a seed, a
backend trait — taken by the constructor, so production passes the real one
and a test passes its own. A test generic over a trait
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
