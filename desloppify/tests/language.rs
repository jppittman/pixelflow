//! Every named part a language claims has a query that compiles and names
//! the `@target` capture.

use desloppify::language::Language;
use desloppify::rule::{Part, TARGET_CAPTURE};

const PARTS: [Part; 3] = [Part::Functions, Part::FunctionSignatures, Part::Types];

fn compiles_with_target(language: Language, source: &str) {
    let query = tree_sitter::Query::new(&language.grammar(), source)
        .unwrap_or_else(|e| panic!("{language:?} {source}: {e}"));
    assert!(query.capture_index_for_name(TARGET_CAPTURE).is_some());
}

#[test]
fn every_part_query_compiles_and_captures_target() {
    for language in Language::ALL {
        for part in PARTS {
            if let Some(source) = language.query(part) {
                compiles_with_target(language, source);
            }
        }
        compiles_with_target(language, language.bodies());
    }
}

#[test]
fn a_file_is_rust_by_its_extension() {
    assert_eq!(Language::of("a/b.rs".as_ref()), Some(Language::Rust));
    assert_eq!(Language::of("a/b.py".as_ref()), None);
    assert_eq!(Language::of("Makefile".as_ref()), None);
}
