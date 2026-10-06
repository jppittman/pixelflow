//! The rules and skills in this crate load, and their queries find what they
//! claim to. Catches a bad query or a dangling skill name before a review does.

use std::path::Path;

use desloppify::language::Language;
use desloppify::rule::Rule;
use desloppify::skills::Skills;
use desloppify::snippet::snippets;

fn shipped() -> Vec<Rule> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let skills = Skills::load(&root.join("skills")).unwrap();
    Rule::load_dir(&root.join("rules"), &skills).unwrap()
}

fn rule(id: &str) -> Rule {
    shipped().into_iter().find(|r| r.id == id).unwrap()
}

#[test]
fn every_shipped_rule_loads() {
    assert!(!shipped().is_empty());
}

#[test]
fn unwrap_query_captures_unwrap_and_expect_only() {
    let source = "fn f() {\n    a.unwrap();\n    b.expect(\"x\");\n    c.len();\n}\n";
    let found = snippets(&rule("panicking-unwrap"), Language::Rust, source).unwrap();
    let lines: Vec<u64> = found.iter().map(|s| s.first_line).collect();
    assert_eq!(lines, [2, 3]);
}

#[test]
fn boolean_argument_query_ignores_other_parameters() {
    let source = "fn a(x: bool) {}\nfn b(x: u32) {}\n";
    let found = snippets(&rule("boolean-argument"), Language::Rust, source).unwrap();
    assert_eq!(found.len(), 1);
    assert!(found[0].numbered.contains("fn a"));
}

#[test]
fn function_names_reviewed_together_are_one_snippet_with_each_line_numbered() {
    let source = "fn get_a() {}\n\nfn fetch_b() {}\n";
    let found = snippets(&rule("naming-consistency"), Language::Rust, source).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].numbered, "    1 | get_a\n  ...\n    3 | fetch_b\n");
}

#[test]
fn functions_scope_reviews_each_function_alone() {
    let source = "fn a() {}\nimpl S { fn b(&self) {} }\n";
    let found = snippets(&rule("narrating-comments"), Language::Rust, source).unwrap();
    assert_eq!(found.len(), 2);
}
