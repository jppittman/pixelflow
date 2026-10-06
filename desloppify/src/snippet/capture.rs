//! Finding snippets with tree-sitter.

use streaming_iterator::StreamingIterator;

use super::Snippet;
use crate::language::Language;
use crate::rule::{CompiledQuery, Review, Rule, Scope};

/// Between captures reviewed together.
const GAP: &str = "  ...\n";

impl Snippet {
    fn new(first_line: u64, text: &str) -> Self {
        let numbered = (first_line..)
            .zip(text.lines())
            .map(|(n, line)| format!("{n:>5} | {line}\n"))
            .collect();
        Self {
            first_line,
            numbered,
        }
    }

    /// All of `source`, headed by the first lines of `captures`; `None` if
    /// there are no captures.
    fn pointing_at(captures: &[Self], source: &str) -> Option<Self> {
        let lines: Vec<String> = captures.iter().map(|c| c.first_line.to_string()).collect();
        if lines.is_empty() {
            return None;
        }
        let whole = Self::new(1, source);
        let numbered = format!("Review lines {}.\n\n{}", lines.join(", "), whole.numbered);
        Some(Self {
            first_line: 1,
            numbered,
        })
    }

    fn join(parts: Vec<Self>) -> Option<Self> {
        let first_line = parts.first()?.first_line;
        let numbered = parts
            .into_iter()
            .map(|p| p.numbered)
            .collect::<Vec<_>>()
            .join(GAP);
        Some(Self {
            first_line,
            numbered,
        })
    }
}

pub(super) fn snippets(
    rule: &Rule,
    language: Language,
    source: &str,
) -> anyhow::Result<Vec<Snippet>> {
    let Scope::Captures { queries, review } = &rule.scope else {
        return Ok(vec![Snippet::new(1, source)]);
    };
    let Some(query) = queries.iter().find(|q| q.language == language) else {
        return Ok(Vec::new());
    };
    let captures = captures(query, source)?;
    Ok(match review {
        Review::Each => captures,
        Review::Together => Snippet::join(captures).into_iter().collect(),
        Review::File => Snippet::pointing_at(&captures, source)
            .into_iter()
            .collect(),
    })
}

fn captures(query: &CompiledQuery, source: &str) -> anyhow::Result<Vec<Snippet>> {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&query.language.grammar())?;
    let tree = parser
        .parse(source, None)
        .ok_or_else(|| anyhow::anyhow!("tree-sitter gave no tree"))?;

    let mut cursor = tree_sitter::QueryCursor::new();
    let mut matches = cursor.captures(&query.query, tree.root_node(), source.as_bytes());
    let mut snippets = Vec::new();
    while let Some((m, index)) = matches.next() {
        let capture = m.captures()[*index];
        if capture.index != query.target {
            continue;
        }
        let node = capture.node;
        snippets.push(Snippet::new(
            node.start_position().row as u64 + 1,
            &source[node.byte_range()],
        ));
    }
    Ok(snippets)
}
