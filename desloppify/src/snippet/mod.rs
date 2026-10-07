//! The code a single review sees.

mod capture;

use anyhow::Result;

use crate::language::Language;
use crate::rule::Rule;

/// Code shown to one call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snippet {
    /// 1-based line of the snippet's first line in its file.
    pub first_line: u64,
    /// The code, each line prefixed by its line number in the file so the
    /// model can point at lines.
    pub numbered: String,
}

/// The snippets `rule` reviews in `source`, a file in `language`, by the
/// rule's [`Unit`](crate::rule::Unit):
///
/// - `File`: one snippet, the whole file.
/// - `Outline`: one snippet, [`outline`] of the file.
/// - `Parts`: none if the language has no such part; otherwise each part
///   alone, in source order (`Group::Each`), or all of them joined into one
///   snippet (`Group::File`, `Group::Crate` — a crate's files are joined
///   later, by the planner).
///
/// Parts inside a `#[cfg(test)]` module are never units: tests belong outside
/// the source tree, and a rule says so about the module itself.
///
/// # Errors
///
/// tree-sitter cannot load the language's grammar or parse `source`.
pub fn snippets(rule: &Rule, language: Language, source: &str) -> Result<Vec<Snippet>> {
    capture::snippets(rule, language, source)
}

/// `source` with the inside of every function body elided: each body's
/// first line keeps its text up to the brace, followed by `{ … }`, and the
/// lines after it, to the body's end, are dropped. Every kept line keeps its
/// line number in the file.
///
/// # Errors
///
/// tree-sitter cannot load the language's grammar or parse `source`.
pub fn outline(language: Language, source: &str) -> Result<Snippet> {
    capture::outline(language, source)
}
