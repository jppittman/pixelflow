# pixelflow-codegen/src/isa

**Is.** The `Isa` argument of the optimizer `P : Kernel × LatticeShape × Isa →
code` (DENOTATION §8, §10): the ISA tier, a property of the process rather
than of the build. A fact the CPU reports, read once, it fixes which backend
is instantiated and the one vector width in the workspace.

## Owns
- The tier: `Isa` is `Avx2` (AVX2+FMA, the floor), `Avx512` (`avx512f` and
  `avx512dq`) or `Neon`. `detect` decides on its first call behind a
  `OnceLock`; it never changes for the life of the process, so a compile
  cache keyed on shape may assume it.
- The probe: the CPUID features each tier's encoders need (`AVX2_FLOOR`,
  `AVX512_TIER` in `x86_64.rs`); aarch64 has one tier, NEON (`aarch64.rs`).
- `PIXELFLOW_ISA`: it picks a narrower tier the host can also run. A name
  that is no tier, a tier the host cannot run, and a host below the floor are
  each a panic quoting the value or naming the missing feature.

## Never
- No width, lane count or register is held here. The width is the backend's
  register file, read through `emit::vector_bytes` as `jit_vector_bytes()`.
- No tier is chosen by `cfg(target_feature)`, `RUSTFLAGS` or `target-cpu`;
  `cfg(target_arch)` selects the probe file and nothing else.
- No SSE2 tier, no scalar tier, no downgrade: an override naming a tier the
  host cannot execute is refused, never lowered.
- No second read of `PIXELFLOW_ISA` or CPUID in `pixelflow-codegen`, and no
  knowledge of what a tier emits (no `OpKind`, register or instruction).

## Contract
`mod.rs`: `Isa` (`Avx2`, `Avx512`, `Neon`; `Isa::name`) and `detect() -> Isa`.
Each probe file offers `pub(super) fn runnable()`, widest tier first, or the
floor feature the host lacks. Consumers: `emit::compile_native` and
`emit::vector_bytes` (one `match` on the tier per compile), `jit_vector_bytes()`
in `lib.rs`, and `detect().name()` in `pixelflow-pipeline` and `pixelflow-runtime`.

## Enforced by
- The type: `Isa` is closed and the `match` in `emit/mod.rs` is exhaustive, so
  a fourth tier is a compile error there; `compile_error!` in `mod.rs` on any
  `target_arch` but x86-64 and aarch64.
- Unit tests in `mod.rs`: `the_host_can_execute_the_tier_it_detected`,
  `a_name_round_trips_and_is_case_insensitive` (no `sse2`).
- `pixelflow-codegen/tests/cross_tier.rs` re-executes its binary under
  `PIXELFLOW_ISA=avx2` and compares it with the wide tier.
- desloppify `simd-is-codegens` (`build_time_isa`; excludes this directory).
- `xtask isa-matrix --clippy --smoke` (CI job "ISA matrix"): the smoke tests
  per tier under `PIXELFLOW_ISA=<tier>`.

## Behind
- `emit::compile_native` calls `detect()` itself: the tier is an ambient read
  where DENOTATION §8 has it an argument of `P`. `EmitCtx { isa }` is
  one-pipeline C1 (`Proposed`, `docs/language/2026-09-24-one-pipeline.md`).
- `xtask/src/main.rs` (`ISA_LEVELS`) keeps its own copy of the tier-to-feature
  table, since `xtask` does not depend on `pixelflow-codegen`.
- The CI job is still named "ISA matrix (SSE2/AVX2/AVX-512)" for branch
  protection though SSE2 is gone (JP-27, still open).

## Decided in
`docs/codegen/2026-09-22-the-isa-is-decided-at-startup.md` (JP, 2026-09-22),
`docs/DENOTATION.md` §8 and §10, Linear JP-27.
