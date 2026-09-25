# Repository Guidelines

## Project Structure & Module Organization
This repository is a Cargo workspace centered on `core-term`, a terminal emulator built on PixelFlow. Primary crates live in `core-term/`, `pixelflow-core/`, `pixelflow-graphics/`, `pixelflow-runtime/`, `pixelflow-compiler/`, `pixelflow-search/`, `pixelflow-pipeline/`, `pixelflow-ml/`, `pixelflow-ir/`, `pixelflow-codegen/`, `actor-scheduler/`, and `actor-scheduler-macros/`. Shared docs live in `docs/`, reusable assets in `assets/`, developer scripts in `scripts/`, and automation tasks in `xtask/`. Keep terminal-specific behavior inside `core-term`; do not move terminal logic into PixelFlow crates.

## Build, Test, and Development Commands
Use the stable Rust toolchain from `rust-toolchain.toml`.

- `cargo build --workspace`: build all workspace crates.
- `cargo test --workspace`: run the full test suite.
- `cargo run --release -p core-term`: launch the terminal directly.
- `cargo bundle-run`: build and run the macOS app bundle.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: enforce lint rules.
- `cargo bench -p pixelflow-core` or `cargo bench -p actor-scheduler`: run focused benchmarks.

## Coding Style & Naming Conventions
Design denotationally: decide what a thing *means* as a mathematical object, and encode that meaning in the type system, before writing the code that manipulates it (see "Denote before you build" in `CLAUDE.md`). A rule that lives only in a doc comment — "this operand must be a constant", "these bits are a mask, not a number", "these coordinates are in the caller's space" — is an invariant some later pass will break, and the repair always costs more than the type would have.

Fold before you dispatch (see `CLAUDE.md`, and `docs/STYLE.md` Code Structure rule 3). A fold leaves fewer live possibilities than it found; dispatch keeps every case alive for everything downstream to carry, so the two are an ordering rather than a division of labor between `if` and `match` — collapse the cases wherever they collapse, and reach for `match` only where they genuinely cannot. Guard clauses and early returns are that rule at function scope and the strongest form of it: a `return` deletes a case rather than collapsing it, and takes the join point with it, which is why code written this way accumulates no `else` blocks. Flat control flow is the symptom; eliminating cases is the reason. Branchless is the limit — and it is a property of the *denotation*, not the instruction stream, so the short-circuit branch codegen emits around an `If` is not a violation: it changes the work done, never the value.

Adding a second implementation of something the repo already does one way: if that category is already a trait, write another `impl`; if it isn't, extract a trait around the existing implementation *first*, rather than forking it, copying it, or bolting on a mode flag (`RegisterAllocator` with `LinearScan` in `pixelflow-codegen` is the worked example). This is the same factoring at type scope — the case is settled once, where the concrete type is chosen, leaving every use downstream unconditional. A `Box<dyn Trait>` instead pays dispatch at every call, and on a hot path is usually better as a `match` over an enum or static dispatch.

Follow `docs/STYLE.md`. Prefer clear names over explanatory comments; use Rustdoc for public APIs and regular comments only for design rationale. Avoid boolean parameters when an enum or separate function is clearer. Use `snake_case` for functions/modules/files, `CamelCase` for types, and named constants instead of unexplained numeric literals.

## Testing Guidelines
Place unit tests near the code they exercise and integration tests under each crate’s `tests/` directory. Existing patterns include names such as `ansi_parser_message_tests.rs` and `actor_roundtrip_tests.rs`. Test public behavior rather than internal implementation details. Before opening a PR, run at least `cargo test --workspace`; use crate-targeted runs such as `cargo test -p core-term` while iterating.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commits with optional conventional prefixes, for example `feat(core): ...`, `feat(optimizer): ...`, and `docs: ...`. Prefer a scoped subject when the affected crate is clear. PRs should describe the behavioral change, list the crates touched, reference related issues or design docs, and include screenshots or terminal output when UI/runtime behavior changed.

## Architecture Notes
Preserve the repository boundary: PixelFlow crates stay general-purpose, while PTY, ANSI, and terminal state handling belong in `core-term`. Keep dependencies minimal and handle `Result` values explicitly; workspace lints deny ignored must-use results.
