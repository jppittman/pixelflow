# pixelflow-core / pixelflow-ir: `sin`/`cos`/`atan2` were badly wrong — 2026-07-22

## Summary

Found while writing public-API correctness tests for `pixelflow-core/src/ops/trig.rs`
as part of a routine test-quality-control pass (mutation-testing follow-up to
`docs/bugs/2026-07-20-test-quality-audit.md`, which had flagged `ops/trig.rs` and
`ops/compare.rs` as the biggest remaining coverage gap but didn't have time to add
real correctness tests). The moment real value-level tests existed, they failed —
badly. Four independent, pre-existing bugs, all in code reachable through the public
`.sin()` / `.cos()` / `.atan2()` `ManifoldExt` combinators:

1. **`cos(0)` returned `π/2 ≈ 1.5708`, not `1`.** The even-polynomial coefficients
   for the cosine Chebyshev/minimax fit were simply wrong — `sin(π/6)` returned
   `0.276` instead of `0.5`, a 45% error well inside the polynomial's claimed
   "principal range."
2. **`atan2(y, x)` returned `NaN` whenever `y == 0`.** `sign_y = y.abs() / y` is
   `0.0 / 0.0` at `y == 0`.
3. **`atan2` was wrong by up to 100% whenever `|y| > |x|`** (the `mask_large`
   branch). The code computed `atan(|r|)` once at `t = |r|`, then for `|r| > 1`
   did `π/2 - (1/|r|) * atan(|r|)` — but the correct identity `atan(r) = π/2 -
   atan(1/r)` requires the polynomial to be *re-evaluated* at `t = 1/|r|`, not
   the `t = |r|` result rescaled. `atan2(2, 1)` returned `2.02` instead of the
   correct `1.107`.
4. **`atan2` had the wrong sign for every `x < 0` quadrant correction.** The
   code combined `correction` and `atan_signed` as `atan_signed - correction`;
   the correct combination (derived from `atan(y/x)` under a sign-flipping
   denominator plus the `±π` branch term) is `correction - atan_signed`.
   `atan2(1, -1)` returned `-2.356` instead of `+2.356` (exact sign flip).

All four bugs were duplicated verbatim across every SIMD backend that hand-rolls
this Chebyshev approximation: `pixelflow-core/src/ops/trig.rs` (the plain-Rust
`Field`-level path) and `pixelflow-ir/src/backend/{x86.rs (SSE2/AVX2/AVX512),
arm.rs (NEON)}` (the JIT/codegen backend). The scalar fallback
(`pixelflow-ir/src/backend/scalar.rs`, used when no SIMD ISA is available) was
**not** affected — it calls `libm::sinf/cosf/atan2f` directly and was always
correct. `Jet2`/`Jet3`/`Dual` autodiff `sin`/`cos`/`atan2` (in `jet2.rs`, `jet3.rs`,
`jet2h.rs`, `path_jet.rs`, `dual.rs`) delegate to the same `Field`-level ops and
inherited the fix automatically; no separate coefficients live there.

Bug #1 (cos coefficients) alone means `cos(0) ≠ 1` — as basis-independent a
correctness failure as a trig approximation can have. This was demonstrably never
caught by any existing test: `ops/trig.rs` had zero unit tests (confirmed by the
2026-07-20 audit), and the only other exerciser, `combinators/spherical.rs`'s
`sh_orthonormality`/`sh_coeffs_dot` tests, evaluates spherical-harmonic basis
functions built from closed-form Cartesian polynomials — it never calls `.sin()`/
`.cos()`/`.atan2()` at all.

## Fix

Derived new coefficients numerically (Chebyshev-node-weighted iterative least
squares, a good approximation to a true Remez/minimax solution) for the same
polynomial *shape* already baked into every backend (odd degree-7 in `t = x/π`
for sin, even degree-6 for cos, degree-7 in `t ∈ [0,1]` for atan), verified against
`f64` reference values on dense grids:

| Function | Max abs error (new) | Max abs/rel error (old, broken) |
|---|---|---|
| `sin` | 2.6e-4 | ~45% at `x = π/6` |
| `cos` | 1.4e-3 | `cos(0)` off by `π/2 - 1 ≈ 0.57` |
| `atan` (atan2's core) | 8.7e-5 | up to 100%+ in the `\|r\|>1` branch |

Also fixed the `atan2` sign-of-y division (replaced `y.abs()/y` with a
comparison-based `y.lt(0.0).select(-1.0, 1.0)`, which is exact and NaN-free at
`y == 0` — `atan(0) == 0` makes the sign choice irrelevant there anyway), the
`|r| > 1` branch (now re-evaluates the same polynomial at `t = 1/|r|` instead of
rescaling the `t = |r|` result), and the `x < 0` quadrant-correction sign.

Applied identically to `pixelflow-core/src/ops/trig.rs` and all four SIMD
implementations in `pixelflow-ir/src/backend/{x86.rs,arm.rs}`. Updated the stale
"~7-8 significant digits" accuracy doc comments to the measured bounds above.

## Verification

- New tests in `pixelflow-core/src/ops/trig.rs` (`sin`/`cos` against `f32::sin`/
  `cos` in the principal range, periodicity across ±37 windings, large-angle
  agreement, and `atan2` across all four quadrants + the `|r|>1` branch) — all
  pass with the fix, all failed before it (that's the point).
- `cargo test --workspace --lib` and `cargo test --workspace --tests`: full
  green, no regressions (438 pixelflow-graphics tests, 101 pixelflow-ir tests
  including `lowering_tests::sin_cos_tan_match_scalar` and
  `inverse_trig_match_scalar`, 91 pixelflow-search tests, core-term's suite,
  all pass).
- `cargo clippy -p pixelflow-core -p pixelflow-ir --tests`, plus
  `RUSTFLAGS="-C target-feature=+avx2,+fma"` / `+avx512f` clippy passes and a
  `--target aarch64-unknown-linux-gnu` check, all clean (fixed two
  `excessive_precision` literals the new coefficients introduced).

## Why this went unnoticed

`ops/trig.rs`'s `cheby_sin`/`cheby_cos`/`cheby_atan2` are `pub(crate)`, reached
only through the public `ManifoldExt::sin()/cos()/atan2()` combinators — nothing
in pixelflow-core, pixelflow-graphics, or core-term's own source currently calls
them (a repo-wide grep found call sites only in `pixelflow-runtime/examples/`
demo programs: `animated_sphere.rs`, `psychedelic_shader.rs`,
`bench_psychedelic.rs`). Anything routed through the `kernel!` macro → JIT
compiler for real-time rendering hits `pixelflow-ir`'s backend `sin`/`cos`/
`atan2` directly, which had the identical bug — but nothing in the terminal
emulator's actual render path appears to use trig today either. That's likely
why 155 FPS terminal rendering was never visibly broken by this, but it means
any future feature reaching for rotation, circular motion, or angle math via
these public entry points would have silently gotten badly wrong numbers.
