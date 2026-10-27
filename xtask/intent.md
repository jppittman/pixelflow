# xtask

**Is.** The workspace's build driver: it fixes the `Isa` argument of
`P : Kernel × LatticeShape × Isa → code` from outside, one `PIXELFLOW_ISA`
per run of one build (DENOTATION §8), and assembles and launches the macOS
bundle. It denotes nothing in the language; its one obligation is that every
tier the host can execute is executed and every tier it cannot is stated.

## Owns
- Which tiers the matrix runs: `ISA_LEVELS` (name, `PIXELFLOW_ISA` value, the
  CPU features that make it executable), and whether a tier is `NotRun`.
- What a PASS claims: `IsaExecutionMode` (`BuildOnly`, `Smoke`,
  `BuildAndTest`) and `test_commands`. Presubmit's smoke set is the crates
  whose output is per-tier machine code (codegen, ir, core, compiler,
  pipeline, graphics' glyph JIT tests and `fonts::loop_blinn::`); the rest is
  postsubmit's. The summary line names the scope.
- The `CoreTerm.app` layout and launch: `bundle`, `bundle-run`,
  `launch-stability-check` (real LaunchServices launches, crash reports).
- `bake-eigen`: Stam's `ccdata50NT.dat` to Rust consts.

## Never
- No `RUSTFLAGS`, `--target` or `target-feature` per tier, and no rebuild per
  tier: `run_cargo` sets `PIXELFLOW_ISA` and `RUST_MIN_STACK` and nothing else.
  The tier is the JIT's startup decision, so one build serves every tier.
- No tier is silently skipped or downgraded. A tier the host cannot run is
  `NotRun` with its reason; xtask never sets a `PIXELFLOW_ISA` it has not probed.
- No dependency on a pixelflow crate: `[dependencies]` is empty, so the tier
  list restates `pixelflow_codegen::isa`'s features and a new tier is added in both.
- No hardcoded build directory: `cargo_target_dir` is the only source.

## Contract
`xtask/src/main.rs` is the whole crate. Run as `cargo run -p xtask -- <cmd>`:
`bundle-run` and `bake-eigen` (also aliases in `.cargo/config.toml`), `bundle`,
`launch-stability-check [--runs N] [--watch-seconds S]`, `isa-matrix [--clippy]
[--smoke | --build-only]` (both modes together are refused). `isa-matrix`
builds and lints once, then runs each tier; off x86-64 it prints a note, exits 0.

## Enforced by
- `.github/workflows/rust.yaml`: job `ISA matrix (SSE2/AVX2/AVX-512)` runs
  `isa-matrix --clippy --smoke`; `macos-launch-stability` runs `launch-stability-check`.
- `postsubmit-flake-detection.yaml` runs `isa-matrix --clippy`; its failure
  makes `automatic-revert.yaml` propose a revert.
- Types: `IsaExecutionMode` makes a contradictory mode unrepresentable past
  `main`; `LevelResult` has no silent case; `host_has_feature` panics on an
  unlisted feature rather than answering false. Nothing pins the first Never.

## Behind
- `bake-eigen` writes `coeffs.rs` into a `subdiv` module of `pixelflow-graphics`
  that is gone, so it emits a file nothing declares. No plan or Linear issue closes it.
- CLAUDE.md writes `cargo xtask isa-matrix`; `.cargo/config.toml` has no `xtask`
  alias, so the working form is `cargo run -p xtask -- isa-matrix`.
- Ten `#[cfg(...)]` sites on items in `main.rs` are baselined in
  `scripts/cfg_encapsulation_baseline.txt`; the matrix belongs in its own file.

## Decided in
`docs/codegen/2026-09-22-the-isa-is-decided-at-startup.md` (xtask lost
`run_with_rustflags`, `host_triple` and the per-level target directory),
`docs/archive/plans/2026-10-08-selection-is-a-phase.md` (legacy-leg suites left xtask),
`docs/process/2026-10-13-tests-check-the-as-if-contract.md`, CLAUDE.md "CI is the gate".
