//! Languages tree-sitter can parse for us, and where each keeps its parts.

use std::path::Path;

use serde::Deserialize;

use crate::rule::Part;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Language {
    Rust,
}

impl Language {
    pub const ALL: [Self; 1] = [Self::Rust];

    /// The language of the file at `path`, judged by its extension.
    #[must_use]
    pub fn of(path: &Path) -> Option<Self> {
        match path.extension()?.to_str()? {
            "rs" => Some(Self::Rust),
            _ => None,
        }
    }

    #[must_use]
    pub fn grammar(self) -> tree_sitter::Language {
        match self {
            Self::Rust => tree_sitter_rust::LANGUAGE.into(),
        }
    }

    /// The tree-sitter query whose `@target` captures are `part` in this
    /// language, or `None` if the language has no such thing.
    #[must_use]
    pub fn query(self, part: Part) -> Option<&'static str> {
        match (self, part) {
            (Self::Rust, Part::Functions) => Some("(function_item) @target"),
            (Self::Rust, Part::FunctionNames) => Some("(function_item name: (identifier) @target)"),
            (Self::Rust, Part::FunctionBodies) => Some("(function_item body: (block) @target)"),
            (Self::Rust, Part::Types) => {
                Some("[(struct_item) (enum_item) (union_item) (trait_item) (type_item)] @target")
            }
            (Self::Rust, Part::Comments) => Some("[(line_comment) (block_comment)] @target"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PARTS: [Part; 5] = [
        Part::Functions,
        Part::FunctionNames,
        Part::FunctionBodies,
        Part::Types,
        Part::Comments,
    ];

    #[test]
    fn every_part_query_compiles_and_captures_target() {
        for language in Language::ALL {
            for part in PARTS {
                let Some(source) = language.query(part) else {
                    continue;
                };
                let query = tree_sitter::Query::new(&language.grammar(), source)
                    .unwrap_or_else(|e| panic!("{language:?} {part:?}: {e}"));
                assert!(
                    query
                        .capture_index_for_name(crate::rule::TARGET_CAPTURE)
                        .is_some()
                );
            }
        }
    }
}
