//! Every named part a language claims has a query that compiles and names
//! the `@target` capture.

use desloppify::language::Language;
use desloppify::rule::{Part, TARGET_CAPTURE};

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
            assert!(query.capture_index_for_name(TARGET_CAPTURE).is_some());
        }
    }
}

#[test]
fn a_file_is_rust_by_its_extension() {
    assert_eq!(Language::of("a/b.rs".as_ref()), Some(Language::Rust));
    assert_eq!(Language::of("a/b.py".as_ref()), None);
    assert_eq!(Language::of("Makefile".as_ref()), None);
}
