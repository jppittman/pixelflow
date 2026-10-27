# pixelflow-runtime/src/display/drivers/web

**Is.** The browser's end of the pull (DENOTATION §5): a worker-side driver that
takes the lattice's extent from `CreateWindow`, blits the engine's frame onto an
`OffscreenCanvas`, and carries the page's input back over a `SharedArrayBuffer`
ring. It does not build today; it is written against the legacy `DisplayDriver`
and API names the runtime has since replaced (Behind).

## Owns
- The worker's browser resources, `OffscreenCanvas` and `SharedArrayBuffer`,
  handed in once by `init_resources` (a thread-local) and taken by `run`.
- The ring's wire format (`ipc::SharedRingBuffer`): two `Int32` indices, then
  length-prefixed `bincode` `DisplayEvent`s. The page writes and never blocks;
  the worker reads and waits with `Atomics`.

## Never
- No `web_sys` or `js_sys` outside this directory, except `lib.rs`'s two
  `wasm_bindgen` entry points (Behind).
- No send result ignored (`let_underscore_must_use = deny`).
- No frame kept on an error path: `Present` returns the buffer even if the blit
  failed. No blocking on the page's thread: a full ring is `RingBufferFull`.

## Contract
`mod.rs` (`#![cfg(use_web_display)]`, selected in `drivers/mod.rs`):
`init_resources`, `WebDisplayDriver` (`new`, `send`, `run`; a clone sends but
cannot run), `pub mod ipc` (`SharedRingBuffer::new`, `blocking_read_timeout`,
`write`).

## Enforced by
- `scripts/check-wasm-status.sh` (job `wasm32-status`, non-blocking) holds this
  driver's failure at `scripts/wasm_status_baseline.txt` ("pixelflow-runtime web
  driver"); `scripts/test_wasm_status.py` tests the script.
- `scripts/check-driver-cfg-coverage.sh` tracks `use_web_display` as a known gap.
- Nothing checks the `web_sys` or ignored-send lines: clippy never reaches this
  file. desloppify `actor-lane-choice` (`ignored_send`) flags the sends in review.

## Behind
- 2026-10-15: no build. `pixelflow-codegen/src/isa/mod.rs` has a `compile_error!`
  on any target but x86-64 and aarch64, and the runtime depends on it through
  `pixelflow-core`. A wasm32 backend is tracked work (CLAUDE.md, "Workspace
  Structure"); no plan names it.
- The baseline hides a second layer: the file is stale on its own. It imports
  `pixelflow_render` (no such crate; the frame is
  `pixelflow_graphics::render::frame::Frame`), names `EngineCommand` (defined
  nowhere), writes `DriverCommand<Rgba>` (not generic), sends
  `EngineData::PresentComplete` (replaced by `WindowGranted`), and runs `bincode`
  on `DisplayEvent` (derives only `Debug`). A wasm32 backend alone would not
  turn the job green.
- Three `let _ = engine_tx.send(...)` break the lint the rest of the crate obeys;
  the job runs `cargo check`, not clippy.
- It implements the deprecated `DisplayDriver` (`display/driver.rs`), not
  `PlatformOps`; `lib.rs` holds its two `wasm_bindgen` entry points under a
  comment saying they belong in the platform. No plan or Linear issue owns the
  port.

## Decided in
No plan decides this driver. Status: CLAUDE.md ("Workspace Structure"),
Linear JP-27 (the wasm status gate). Crate: `pixelflow-runtime/intent.md`.
