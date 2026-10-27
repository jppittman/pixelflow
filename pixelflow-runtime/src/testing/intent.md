# pixelflow-runtime/src/testing

**Is.** Two fixtures standing where the application and the engine stand in the
pull: `MockEngine` receives the engine's three lanes and records what arrives, and
`black_scene`/`constant_scene` build the smallest `Scene` there is, a constant
kernel compiled at a frame's extent (DENOTATION §5). It is a test API the crate
ships, which the behavioral-contracts skill says should not exist (DENOTATION §8:
a test observes the production API's values).

## Owns
- A recording receiver: `MockEngine` runs an `ActorBuilder` scheduler on its own
  thread and stores each `EngineData`, `EngineControl` and `AppManagement` it is
  sent as a `ReceivedMessage`.
- The flat-colour `Scene`: `constant_scene(rgba, frame)` and `black_scene(frame)`,
  four constant channel kernels bound to nothing.
- Nothing about the language, the platform or what a lane means; both fixtures
  borrow those from `api/` and `pixelflow-graphics`.

## Never
- No terminal logic, no SIMD, lane or width, no OS call: a scene comes from
  `pixelflow_graphics::render::scene::compile_platform_packed`, never by hand.
- No production path depends on it: only `#[cfg(test)]` code in this crate names
  `crate::testing`.
- No message of its own: `ReceivedMessage` has one variant per lane of
  `EngineData`, `EngineControl` and `AppManagement`, and nothing else.

## Contract
`mod.rs` declares `mock_engine` and `scene` and re-exports `MockEngine`,
`black_scene`, `constant_scene`. `MockEngine::new` starts it; `take_handle`
yields its one `EngineActorHandle` (once; a second call panics); `messages` locks
the record. `ReceivedMessage` is `pub` in `mock_engine.rs` and not re-exported
from `mod.rs`.

## Enforced by
- Nothing. `lib.rs` declares `pub mod testing;` with no `cfg`, so both fixtures
  ship in every build as public API of the crate.
- desloppify `tests-target-the-contract` (`test_only_api`: "a fixture made for
  tests") judges test files, not this module's place under `src/`.

## Behind
- This module is an API that exists only so tests can reach it (behavioral-contracts,
  "No API exists for tests"). `MockEngine` is the clearest case: nothing constructs
  it (`pixelflow-runtime/tests/macos_integration_tests.rs` records that its last
  caller was removed), and `MockEngine::new` spawns a thread that is never joined.
  It is dead and is deleted, not gated.
- `constant_scene` has the body of `pixelflow_graphics::render::scene::constant_platform_scene`,
  which the graphics crate already offers. `black_scene` has five call sites,
  all in `#[cfg(test)]` modules (`engine_core.rs`, `engine_troupe.rs`,
  `coordinator_node.rs` twice, `render_coordinator.rs`); they call the graphics
  function directly and this module goes.
- The runtime's real injection seam is `PlatformOps` behind `PlatformActor`, driven
  by a mock in `tests/platform_actor_tests.rs`. A test of an engine actor takes
  that shape, not a recorder in `src/`.
- No plan or Linear issue names this module yet.

## Decided in
`desloppify/skills/behavioral-contracts/SKILL.md` ("Tests only ever see the
production API"), `docs/process/2026-10-13-tests-check-the-as-if-contract.md`. No
plan decides this module's existence.
