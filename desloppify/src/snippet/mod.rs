//! The code a single review sees.

mod capture;

use anyhow::Result;

use crate::language::Language;
use crate::rule::Rule;

/// Code shown to one call.
pub struct Snippet {
    /// 1-based line of the snippet's first line in its file.
    pub first_line: u64,
    /// The code, each line prefixed by its line number in the file so the
    /// model can point at lines.
    pub numbered: String,
}

/// The snippets `rule` reviews in `source`, a file in `language`.
///
/// - A whole-file rule: one snippet, the whole file.
/// - A rule with no query for `language`: none.
/// - Otherwise, by the rule's [`Review`](crate::rule::Review): each `@target`
///   capture alone, in source order; all of them joined into one snippet; or,
///   if there are any, the whole file headed by `Review lines a, b.` naming
///   the captures' first lines.
///
/// # Errors
///
/// tree-sitter cannot load the language's grammar or parse `source`.
pub fn snippets(rule: &Rule, language: Language, source: &str) -> Result<Vec<Snippet>> {
    capture::snippets(rule, language, source)
}
