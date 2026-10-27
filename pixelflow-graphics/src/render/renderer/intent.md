# pixelflow-graphics/src/render/renderer

**Is.** The thread that applies a compiled scene to a frame on request: a
Mealy machine from (scene, frame, payload) to the frame and payload
(DENOTATION §5, "Evaluation is whole-domain"). It evaluates nothing: a request
is one `Scene::render` call, one collapse per stripe.

## Owns
- The frame's round trip: a `RenderRequest` carries the caller's only render
  buffer in, a `RenderResponse` carries it out; `render_time: None` means not drawn.
- The pause flag and the thread count (`RenderCore`), clamped to at least one.
- The bootstrap handshake: no request can be sent until a response channel is
  registered. Render requests are Data; `Pause`/`Resume` Control; thread count
  and config queries Management.

## Never
- Never drops a frame. `RenderCore::render` returns a `RenderResponse`, which
  needs a `Frame`, and the only frame in scope is the request's; it is
  infallible, so no `Err` path exists that could carry none.
- Never reads `meta`: it is destructured and handed back untouched.
- Never builds or inspects a scene: outside `#[cfg(test)]` no `Kernel`,
  collapse, lattice, byte order or ISA appears; it calls
  `scene.render(&mut frame, threads)` and nothing else of the scene.
- Never knows a pixel format, window or terminal: `P: Pixel`, `Meta` generic.
- Never decides in the adapter: `RendererActor` only sends `RenderCoreOut`;
  pause, resume and thread count live in `RenderCore`.

## Contract
`mod.rs` re-exports `RendererActor`, `RenderRequest`, `RenderResponse`,
`RenderControl`, `RenderManagement`, `RenderConfig`, `RendererHandle`,
`RendererSetup`, `RendererSetupHandle`. `RendererActor::spawn_with_setup`
returns the setup handle; `RendererSetupHandle::register` returns the handle.

## Enforced by
- Types: `RenderCore::render` cannot return without a `Frame`; a
  `RendererHandle` comes only from `RendererSetupHandle::register`
  (`RendererSetup::reply_tx` and `RendererSetupHandle::new` are `pub(crate)`).
- `actor.rs` tests `a_paused_core_returns_the_frame_unrendered`,
  `meta_round_trips_untouched_through_a_render`.
- desloppify `no-terminal-logic-in-pixelflow`, `simd-is-codegens`,
  `per-frame-allocation`. `actor-lane-choice` covers the senders
  (`pixelflow-runtime/**`, `core-term/**`), not this directory.

## Behind
- `Pause`, `Resume`, `SetThreadCount` and `GetConfig` have no sender outside
  this directory (2026-10-15); the runtime fixes the count at `spawn_renderer`.
  No plan or Linear issue gives them a user.
- `RenderControl::Pause` is documented "stop processing Data messages"; the
  code returns the frame unrendered.
- The adapter is an `Actor` shell over a `Transducer`, spawned by handshake
  outside the troupe (`engine_troupe.rs`), not a `Node` on a `Host`; the mesh
  design defers that move (§5, step 2b).
- `Scene::render` spawns workers per request (`render_bands`); the stripe
  schedule is the CPU backend's (kernels-evaluate-kernels §3).

## Decided in
`docs/runtime/pixelflow-runtime-engine-mesh-migration.md` §5 step 3,
`docs/runtime/actor-scheduler-mealy-transducer.md`, `docs/language/2026-10-15-kernels-evaluate-kernels.md` §3.
