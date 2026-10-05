//! The code a single review sees.

use streaming_iterator::StreamingIterator;

use crate::rule::{Rule, Target};

pub struct Snippet {
    /// 1-based line of the snippet's first line in its file.
    pub first_line: u64,
    pub text: String,
}

impl Snippet {
    /// The snippet with each line prefixed by its line number in the file, so
    /// the model can point at lines.
    #[must_use]
    pub fn numbered(&self) -> String {
        (self.first_line..)
            .zip(self.text.lines())
            .map(|(n, line)| format!("{n:>5} | {line}\n"))
            .collect()
    }
}

/// The snippets `rule` reviews in `source`.
pub fn snippets(rule: &Rule, source: &str) -> anyhow::Result<Vec<Snippet>> {
    let Target::Query { query, target } = &rule.target else {
        return Ok(vec![Snippet {
            first_line: 1,
            text: source.to_owned(),
        }]);
    };

    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&rule.language.grammar())?;
    let tree = parser
        .parse(source, None)
        .ok_or_else(|| anyhow::anyhow!("tree-sitter gave no tree"))?;

    let mut cursor = tree_sitter::QueryCursor::new();
    let mut captures = cursor.captures(query, tree.root_node(), source.as_bytes());
    let mut snippets = Vec::new();
    while let Some((m, index)) = captures.next() {
        let capture = m.captures()[*index];
        if capture.index != *target {
            continue;
        }
        let node = capture.node;
        snippets.push(Snippet {
            first_line: node.start_position().row as u64 + 1,
            text: source[node.start_byte()..node.end_byte()].to_owned(),
        });
    }
    Ok(snippets)
}
