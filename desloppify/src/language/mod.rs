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
            (Self::Rust, Part::FunctionSignatures) => {
                Some("[(function_item body: (_) @cut) (function_signature_item)] @target")
            }
            (Self::Rust, Part::FunctionNames) => Some("(function_item name: (identifier) @target)"),
            (Self::Rust, Part::FunctionBodies) => Some("(function_item body: (block) @target)"),
            (Self::Rust, Part::Types) => {
                Some("[(struct_item) (enum_item) (union_item) (trait_item) (type_item)] @target")
            }
            (Self::Rust, Part::Comments) => Some("[(line_comment) (block_comment)] @target"),
        }
    }

    /// The node kinds that are a function in this language: what a
    /// [`Review::Function`](crate::rule::Review::Function) capture is shown
    /// inside.
    #[must_use]
    pub fn function_kinds(self) -> &'static [&'static str] {
        match self {
            Self::Rust => &["function_item"],
        }
    }

    /// Whether `node` is a test module: in Rust, a `mod` with a
    /// `#[cfg(test)]` attribute.
    #[must_use]
    pub fn is_test_module(self, node: tree_sitter::Node, source: &str) -> bool {
        match self {
            Self::Rust => {
                node.kind() == "mod_item"
                    && std::iter::successors(
                        node.prev_named_sibling(),
                        tree_sitter::Node::prev_named_sibling,
                    )
                    .take_while(|n| n.kind() == "attribute_item")
                    .any(|attr| {
                        source[attr.byte_range()]
                            .replace(' ', "")
                            .contains("cfg(test)")
                    })
            }
        }
    }
}
