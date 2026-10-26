# pixelflow-runtime

**Is.** The platform end of the pull: it holds the OS window, takes the lattice's
extent from the OS (a resize is a new extent and a recompile, DENOTATION §5),
lends the frame a `collapse` writes into, and shows the result. It carries no
language: it is the actors, threads and drivers around a collapse.

## Owns
- The windowing platform: `PlatformOps` behind `PlatformActor`, `ActivePlatform`,
  the event loop, and the OS's pixel format (`PlatformPixel`). The driver is the
  troupe's `[main]` actor, so it runs on the thread that built the `Troupe`.
- `platform::PlatformPixel` re-exports the graphics framebuffer format;
  `pixel::PlatformPixel` re-exports that same type.
- The frame buffer: `WindowKeeper` allocates it at a resize, lends it, takes it back.
- The engine's wiring: the actors (`driver`, `engine`, vsync and coordinator
  nodes) and each message's lane.
- The input vocabulary (`KeySymbol`, `Modifiers`), the app API (`Application`,
  `EngineEvent`), and the first `isa::detect()` call (`Troupe::with_config`).

## Never
- No terminal logic (PTY, ANSI, cell) and no dependency on `core-term`.
- No SIMD, lane or width: no `std::arch` or vector type; kernels arrive compiled.
- No Cocoa or X11 call outside `platform/`; Cocoa only on the main thread.
- No per-frame heap allocation: the buffer is recycled by hand-back.
- No decision in a handler: a pure core (`EngineCore`, `VsyncCore`, `CoordinatorCore`,
  `RenderCoordinator`) returns what to emit; the adapter sends.
- No message on the wrong lane, no send result ignored: bulk data rides Data;
  input, resize and close ride Control or Management.
- No platform `#[cfg]` on an item in a platform-agnostic file.

## Contract
`src/lib.rs` re-exports `api::public::*`, `RuntimeError`, `EngineConfig`,
`WindowConfig`, `actor_scheduler` as `actor`, `Troupe` as `EngineTroupe`;
`engine_troupe.rs` is the bootstrap; `engine_core.rs`, `coordinator_node.rs`,
`render_coordinator.rs`, `vsync_actor.rs` are the pure cores and their nodes;
`build.rs` picks the driver cfg (`use_cocoa_display`, `use_x11_display`,
`use_web_display`); `api/`, `display/`, `platform/`, `testing/` have their own `intent.md`.

## Enforced by
- `scripts/check-cfg-encapsulation.sh` (job `cfg-encapsulation`), baselined in
  `scripts/cfg_encapsulation_baseline.txt`; a new violation fails.
- `scripts/check-driver-cfg-coverage.sh`: every `use_*_display` cfg is built by a
  CI job or tracked as a known gap; `scripts/check-wasm-status.sh` holds the web
  driver's failure at its baseline.
- desloppify `actor-lane-choice`, `per-frame-allocation`, `no-terminal-logic-in-pixelflow`.
- `tests/target_topology.rs` (the mesh's edges, via `Topology`),
  `tests/platform_actor_tests.rs` (`PlatformActor` over a mock `PlatformOps`).

## Behind
- `EngineHandler` still relays app input to the coordinator; the mediator is not
  yet a mesh (mesh-migration design, Draft 2026-07-25, §8).
- `traits.rs` holds an `EngineEvent` nothing imports, shadowing
  `api::public::EngineEvent`.
- `lib.rs` carries two `wasm_bindgen` entry points its own comment says belong in
  the platform. The web driver does not build (no wasm32 JIT backend), keeps the
  legacy `DisplayDriver` trait, and discards send results (`display/drivers/web/mod.rs`).

## Decided in
`docs/runtime/pixelflow-runtime-engine-mesh-migration.md`,
`docs/runtime/actor-scheduler-mealy-transducer.md`,
`docs/codegen/2026-09-22-the-isa-is-decided-at-startup.md`; CLAUDE.md "Platform Notes".
