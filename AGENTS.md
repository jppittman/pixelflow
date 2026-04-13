# Repository Guidelines

## Project Structure & Module Organization
This repository is a Cargo workspace centered on `core-term`, a terminal emulator built on PixelFlow. Primary crates live in `core-term/`, `pixelflow-core/`, `pixelflow-graphics/`, `pixelflow-runtime/`, `pixelflow-compiler/`, `pixelflow-search/`, `pixelflow-pipeline/`, `pixelflow-ml/`, `pixelflow-ir/`, `actor-scheduler/`, and `actor-scheduler-macros/`. Shared docs live in `docs/`, reusable assets in `assets/`, developer scripts in `scripts/`, and automation tasks in `xtask/`. Keep terminal-specific behavior inside `core-term`; do not move terminal logic into PixelFlow crates.

## Build, Test, and Development Commands
Use the stable Rust toolchain from `rust-toolchain.toml`.

- `cargo build --workspace`: build all workspace crates.
- `cargo test --workspace`: run the full test suite.
- `cargo run --release -p core-term`: launch the terminal directly.
- `cargo xtask bundle-run`: build and run the macOS app bundle.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: enforce lint rules.
- `cargo bench -p pixelflow-core` or `cargo bench -p actor-scheduler`: run focused benchmarks.

## Coding Style & Naming Conventions
Follow `docs/STYLE.md`. Prefer clear names over explanatory comments; use Rustdoc for public APIs and regular comments only for design rationale. Keep control flow flat with guard clauses, prefer `match` over long `else if` chains, and avoid boolean parameters when an enum or separate function is clearer. Use `snake_case` for functions/modules/files, `CamelCase` for types, and named constants instead of unexplained numeric literals.

## Testing Guidelines
Place unit tests near the code they exercise and integration tests under each crate’s `tests/` directory. Existing patterns include names such as `ansi_parser_message_tests.rs` and `actor_roundtrip_tests.rs`. Test public behavior rather than internal implementation details. Before opening a PR, run at least `cargo test --workspace`; use crate-targeted runs such as `cargo test -p core-term` while iterating.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commits with optional conventional prefixes, for example `feat(core): ...`, `feat(optimizer): ...`, and `docs: ...`. Prefer a scoped subject when the affected crate is clear. PRs should describe the behavioral change, list the crates touched, reference related issues or design docs, and include screenshots or terminal output when UI/runtime behavior changed.

## Architecture Notes
Preserve the repository boundary: PixelFlow crates stay general-purpose, while PTY, ANSI, and terminal state handling belong in `core-term`. Keep dependencies minimal and handle `Result` values explicitly; workspace lints deny ignored must-use results.
