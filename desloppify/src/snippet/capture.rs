//! Finding snippets with tree-sitter.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow};
use streaming_iterator::StreamingIterator;
use tree_sitter::Node;

use super::Snippet;
use crate::language::Language;
use crate::rule::{CompiledQuery, Review, Rule, Scope};

/// Between captures reviewed together.
const GAP: &str = "  ...\n";

/// One `@target`: where it starts and the text it covers.
struct Capture {
    first_line: u64,
    text: String,
    /// The byte range and first line of the function holding it, if any.
    function: Option<(std::ops::Range<usize>, u64)>,
}

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

    /// `text`, starting at `first_line`, headed by the `lines` to review.
    fn pointing_at(lines: &[u64], first_line: u64, text: &str) -> Self {
        let lines: Vec<String> = lines.iter().map(u64::to_string).collect();
        let whole = Self::new(first_line, text);
        Self {
            first_line,
            numbered: format!("Review lines {}.\n\n{}", lines.join(", "), whole.numbered),
        }
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

pub(super) fn snippets(rule: &Rule, language: Language, source: &str) -> Result<Vec<Snippet>> {
    let Scope::Captures { queries, review } = &rule.scope else {
        return Ok(vec![Snippet::new(1, source)]);
    };
    let Some(query) = queries.iter().find(|q| q.language == language) else {
        return Ok(Vec::new());
    };
    let captures = captures(query, source)?;
    if captures.is_empty() {
        return Ok(Vec::new());
    }
    let alone = |c: &Capture| Snippet::new(c.first_line, &c.text);
    Ok(match review {
        Review::Each => captures.iter().map(alone).collect(),
        Review::Together | Review::Crate => Snippet::join(captures.iter().map(alone).collect())
            .into_iter()
            .collect(),
        Review::Function => in_functions(&captures, source),
        Review::File => {
            let lines: Vec<u64> = captures.iter().map(|c| c.first_line).collect();
            vec![Snippet::pointing_at(&lines, 1, source)]
        }
    })
}

/// Each function holding a capture, pointing at its captures' lines; a
/// capture outside every function, alone. In source order.
fn in_functions(captures: &[Capture], source: &str) -> Vec<Snippet> {
    let mut functions: BTreeMap<usize, (std::ops::Range<usize>, u64, Vec<u64>)> = BTreeMap::new();
    let mut snippets = Vec::new();
    for capture in captures {
        let Some((range, first_line)) = &capture.function else {
            snippets.push((
                capture.first_line,
                Snippet::new(capture.first_line, &capture.text),
            ));
            continue;
        };
        functions
            .entry(range.start)
            .or_insert_with(|| (range.clone(), *first_line, Vec::new()))
            .2
            .push(capture.first_line);
    }
    for (range, first_line, mut lines) in functions.into_values() {
        lines.dedup();
        snippets.push((
            first_line,
            Snippet::pointing_at(&lines, first_line, &source[range]),
        ));
    }
    snippets.sort_by_key(|(line, _)| *line);
    snippets.into_iter().map(|(_, snippet)| snippet).collect()
}

fn captures(query: &CompiledQuery, source: &str) -> Result<Vec<Capture>> {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&query.language.grammar())?;
    let tree = parser
        .parse(source, None)
        .ok_or_else(|| anyhow!("tree-sitter gave no tree"))?;

    let mut cursor = tree_sitter::QueryCursor::new();
    let mut matches = cursor.matches(&query.query, tree.root_node(), source.as_bytes());
    let mut captures = Vec::new();
    while let Some(m) = matches.next() {
        let Some(target) = node(m.captures(), query.target) else {
            continue;
        };
        if in_test_module(target, source, query.language) {
            continue;
        }
        let end = query
            .cut
            .and_then(|cut| node(m.captures(), cut))
            .map_or(target.end_byte(), |cut| cut.start_byte());
        captures.push(Capture {
            first_line: line_of(target),
            text: source[target.start_byte()..end].trim_end().to_owned(),
            function: enclosing_function(target, query.language),
        });
    }
    captures.sort_by_key(|c| c.first_line);
    Ok(captures)
}

fn node<'tree>(captures: &[tree_sitter::QueryCapture<'tree>], index: u32) -> Option<Node<'tree>> {
    captures.iter().find(|c| c.index == index).map(|c| c.node)
}

fn enclosing_function(node: Node, language: Language) -> Option<(std::ops::Range<usize>, u64)> {
    let kinds = language.function_kinds();
    let function =
        std::iter::successors(Some(node), Node::parent).find(|n| kinds.contains(&n.kind()))?;
    Some((function.byte_range(), line_of(function)))
}

/// Whether `node` sits inside a test module, whose contents are not reviewed:
/// tests belong outside the source tree, and a rule says so about the module
/// itself.
fn in_test_module(node: Node, source: &str, language: Language) -> bool {
    std::iter::successors(node.parent(), Node::parent)
        .any(|ancestor| language.is_test_module(ancestor, source))
}

fn line_of(node: Node) -> u64 {
    node.start_position().row as u64 + 1
}
