//! Finding units with tree-sitter.

use std::ops::Range;

use anyhow::{Result, anyhow};
use streaming_iterator::StreamingIterator;
use tree_sitter::{Node, Query, QueryCursor, Tree};

use super::Snippet;
use crate::language::Language;
use crate::rule::{CompiledQuery, Group, Rule, TARGET_CAPTURE, Unit};

/// Between parts reviewed together.
const GAP: &str = "  ...\n";

/// What an elided body is shown as.
const ELIDED: &str = "{ … }";

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
    let (group, queries) = match &rule.unit {
        Unit::File => return Ok(vec![Snippet::new(1, source)]),
        Unit::Outline => return Ok(vec![outline(language, source)?]),
        Unit::Parts { group, queries, .. } => (group, queries),
    };
    let Some(query) = queries.iter().find(|q| q.language == language) else {
        return Ok(Vec::new());
    };
    let parts = parts(query, source)?;
    Ok(match group {
        Group::Each => parts,
        Group::File | Group::Crate => Snippet::join(parts).into_iter().collect(),
    })
}

pub(super) fn outline(language: Language, source: &str) -> Result<Snippet> {
    let tree = parse(language, source)?;
    let query = Query::new(&language.grammar(), language.bodies())?;
    let target = query
        .capture_index_for_name(TARGET_CAPTURE)
        .ok_or_else(|| anyhow!("{language:?}'s body query has no @{TARGET_CAPTURE}"))?;
    let mut bodies: Vec<Range<usize>> = Vec::new();
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, tree.root_node(), source.as_bytes());
    while let Some(m) = matches.next() {
        bodies.extend(
            m.captures()
                .iter()
                .filter(|c| c.index == target)
                .map(|c| c.node.byte_range()),
        );
    }
    bodies.sort_by_key(|b| b.start);
    // A body inside another is elided with it.
    let mut outer: Vec<Range<usize>> = Vec::new();
    for body in bodies {
        if outer.last().is_some_and(|o| body.end <= o.end) {
            continue;
        }
        outer.push(body);
    }

    let mut numbered = String::new();
    let mut bodies = outer.iter().peekable();
    let mut offset = 0;
    for (line, text) in (1_u64..).zip(source.split_inclusive('\n')) {
        let span = offset..offset + text.len();
        offset = span.end;
        while bodies.peek().is_some_and(|b| b.end <= span.start) {
            bodies.next();
        }
        let shown = match bodies.peek() {
            Some(body) if body.start <= span.start => continue,
            Some(body) if body.start < span.end => {
                format!("{}{ELIDED}", &source[span.start..body.start])
            }
            _ => text.trim_end_matches('\n').to_owned(),
        };
        numbered.push_str(&format!("{line:>5} | {shown}\n"));
    }
    Ok(Snippet {
        first_line: 1,
        numbered,
    })
}

/// Each `@target`, as its own snippet, in source order.
fn parts(query: &CompiledQuery, source: &str) -> Result<Vec<Snippet>> {
    let tree = parse(query.language, source)?;
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query.query, tree.root_node(), source.as_bytes());
    let mut parts = Vec::new();
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
        let text = source[target.start_byte()..end].trim_end();
        parts.push(Snippet::new(line_of(target), text));
    }
    parts.sort_by_key(|p| p.first_line);
    Ok(parts)
}

fn parse(language: Language, source: &str) -> Result<Tree> {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&language.grammar())?;
    parser
        .parse(source, None)
        .ok_or_else(|| anyhow!("tree-sitter gave no tree"))
}

fn node<'tree>(captures: &[tree_sitter::QueryCapture<'tree>], index: u32) -> Option<Node<'tree>> {
    captures.iter().find(|c| c.index == index).map(|c| c.node)
}

/// Whether `node` sits inside a test module.
fn in_test_module(node: Node, source: &str, language: Language) -> bool {
    std::iter::successors(node.parent(), Node::parent)
        .any(|ancestor| language.is_test_module(ancestor, source))
}

fn line_of(node: Node) -> u64 {
    node.start_position().row as u64 + 1
}
