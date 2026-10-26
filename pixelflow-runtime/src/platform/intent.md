# pixelflow-runtime/src/platform

**Is.** The seam where the OS is chosen: one `target_os` decision naming the
`PlatformOps` implementation (`ActivePlatform`), the pixel format the frame is
a lattice of (`PlatformPixel`, DENOTATION §5), and the `EventLoopWaker` that
interrupts that platform's blocking poll from another thread. No logic of its own.

## Owns
- The selection of `linux/` or `macos/`, as `#[cfg]` on `mod` items and on
  single-line `pub use` and `pub type` items in `mod.rs`.
- `ActivePlatform` (`PlatformActor<LinuxOps>` or `<MetalOps>`), `PlatformDriver`
  (the `DriverActor` over it), and the re-export of graphics' `PlatformPixel`
  (`Bgra8` on Linux, else `Rgba8`), also re-exported through `pixel.rs`.
- `EventLoopWaker` and its implementations (`X11Waker`, `CocoaWaker`,
  `NoOpWaker`), and the macOS wake-token coalescing (`consume_wake_token`).

## Never
- No X11, Cocoa or Metal call in `mod.rs`; outside the wakers they live in
  `linux/` and `macos/`.
- No `#[cfg]` on a `fn`, `struct`, `impl` or field in `mod.rs`: only on `mod`
  items and single-line dispatch items.
- No event handling, window state or frame logic: that is `PlatformOps`.
- No engine or actor handle in a waker: it imports `RuntimeError` and the
  platform's own types.
- No send result ignored: `EventLoopWaker::wake` returns `Result<(), RuntimeError>`.

## Contract
`mod.rs`: `waker`, `macos`, `linux`, the glob re-export of the selected
sub-module, `PlatformPixel`, `PlatformDriver`, `ActivePlatform`. `waker.rs`:
`EventLoopWaker`, `NoOpWaker`, `X11Waker` (`use_x11_display`), `CocoaWaker`
and `consume_wake_token` (`use_cocoa_display`).

## Enforced by
- `scripts/check-cfg-encapsulation.sh`: a platform `#[cfg]` is allowed on a
  `mod` item or a single-line item in `mod.rs`, and nowhere else; a new
  violation fails.
- `scripts/check-driver-cfg-coverage.sh`: every `use_*_display` cfg that
  `build.rs` selects is built by a CI job or tracked as a known gap.
- `unused_must_use = "deny"` (workspace lints) refuses a dropped `wake` result.
- `tests/platform_actor_tests.rs`: `PlatformActor` over a mock `PlatformOps`,
  no platform module in the loop.

## Behind
- The waker is chosen outside the seam: `engine_troupe.rs` picks `CocoaWaker`,
  `X11Waker` or none by `target_os` (baselined), and `linux/` is gated by
  `target_os` while `X11Waker` is gated by `use_x11_display`.
- `waker.rs` holds both platforms' wakers in one file, and one baselined
  `#[cfg(all(target_os = "macos", not(use_cocoa_display)))]` sits on
  `consume_wake_token`.
- `PlatformDriver` has no user. Off Linux and macOS `ActivePlatform` is
  undefined while `PlatformPixel` falls back to `Rgba8`.

## Decided in
CLAUDE.md "Code Style" (platform `cfg` is encapsulation) and "Platform Notes";
`docs/runtime/2026-09-03-wayland-driver.md` (`PlatformOps` as the driver seam);
`docs/runtime/pixelflow-runtime-engine-mesh-migration.md`.
