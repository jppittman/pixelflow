//! The front end's unit tests, one file per module, each a black-box test of
//! that module's contract.
//!
//! A child of the crate root sees what the root sees: every `pub(crate)`
//! item — exactly what each `mod.rs` re-exports — and `lib.rs`'s own
//! `front_end` and `macro_tier`, and no `pub(super)` item of any module. So
//! the compiler holds the law a `<module>/tests.rs` could not: a test drives
//! a stage from the stage before it (`parse` → `analyze` → `Expansion::lower`
//! → `front_end`), never a hand-built intermediate, and asserts a contract's
//! return or the pipeline's refusal text
//! (docs/compiler/2026-10-22-the-front-ends-stages-are-its-modules.md §1.5).

mod ast;
mod front_end;
mod lower;
mod parser;
mod sema;
