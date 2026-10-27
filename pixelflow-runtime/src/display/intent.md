# pixelflow-runtime/src/display

**Is.** The seam between the lattice and the OS window (DENOTATION §5: a lattice
is an extent and nothing else; a resize is a new extent). The OS reports an extent;
this module makes it the one frame buffer a `collapse` writes into, lends it, takes
it back and shows it. A backend *returns* what happened; one adapter sends it.

## Owns
- The seam: `PlatformOps` and `DriverOut`/`DriverEmit`, the effects a step returns.
- The adapter `PlatformActor<Ops>`, the only holder of the engine handle here and
  so the only place a `DriverEmit` becomes a send; `DriverActor<P>` is the troupe's
  `[main]` actor.
- The buffer's lifecycle: `WindowKeeper` allocates it at the frame extent, keeps it
  at rest, grants it on request, takes it back. `Generation` stamps each buffer and
  `Presented::{Blit, Superseded}` is the one identity check.
- The message vocabulary (`Display{Data,Control,Mgmt,Event}`, `Window`, `Surface`).

## Never
- No Cocoa, X11 or `unsafe` in this directory; the OS is reached only through a
  `PlatformOps` impl in `platform/` or `drivers/`.
- A `PlatformOps` method holds no engine handle and sends nothing: it pushes into
  the `&mut DriverOut` it is given, so it runs with no engine or scheduler.
- A backend never sees `DisplayMgmt::RequestWindow` and never allocates a `Frame`.
- No platform `#[cfg]` on an item: `use_*_display` gates only `mod` and one-line
  `pub use` in `mod.rs` and `drivers/mod.rs`.
- No ignored send: `PlatformActor` panics, naming the send, if the engine refuses it.

## Contract
`mod.rs` is the module list and re-exports (`DriverActor`, `DisplayEvent`, `Platform`,
the active `*DisplayDriver`); it declares no trait. `ops.rs`: `PlatformOps`,
`DriverOut`, `DriverEmit`. `platform.rs`: `Platform` (an actor marker), `PlatformActor`.
`driver.rs`: `DriverActor`. `window_keeper.rs`: `WindowKeeper`, `Presented`.
`messages.rs`: the vocabulary. `MetalOps` and `LinuxOps` live in `platform/`.

## Enforced by
- Types: `DriverOut` is the only way out of a `PlatformOps` step; `Window` moves by
  value (one holder); `Presented` is `#[must_use]` and names the superseded case.
- desloppify `effects-are-returned` (cites `DriverOut`).
- `pixelflow-runtime/tests/platform_actor_tests.rs` (delegation through a mock
  `PlatformOps`; a backend emitting with no engine) and the unit tests in
  `window_keeper.rs` (lend once, resize supersedes, Retina allocates the lattice).
- `scripts/check-cfg-encapsulation.sh` (no baseline entry here);
  `scripts/check-driver-cfg-coverage.sh` (each `use_*_display` has a CI job or gap).

## Behind
- The seam is not in `mod.rs` (2026-10-15): `PlatformOps` is in `ops.rs`, `Platform`
  in `platform.rs`, and `mod.rs`'s doc still describes a `DisplayDriver`. The
  behavioral-contracts skill names it; no Linear issue or plan closes it.
- `driver.rs` still holds the deprecated `DisplayDriver` trait; only
  `drivers/web/mod.rs` implements it, sending `EngineData::PresentComplete`, which no
  longer exists. The web driver does not build (no wasm32 JIT backend); untracked.
- `messages.rs` still documents a `PresentComplete` return; `Present` is the return
  under the pull protocol (mesh-migration design §8).
- `drivers::MetalDisplayDriver` and `platform::PlatformDriver` are the same type on macOS.

## Decided in
`docs/runtime/pixelflow-runtime-engine-mesh-migration.md` (Draft 2026-07-25): §5.1
the send leaves the ops, §7.1 `Window::tear`, §8 the pull protocol;
`docs/runtime/2026-09-03-wayland-driver.md`.
