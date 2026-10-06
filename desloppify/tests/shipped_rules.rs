//! The rules and skills in this crate load, and their queries find what they
//! claim to. Catches a bad query or a dangling skill name before a review does.

use std::path::Path;
use std::path::PathBuf;

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
fn types_scope_points_at_each_type_in_the_file() {
    let source = "struct A { id: u32 }\nfn f() {}\nenum B { X }\n";
    let found = snippets(&rule("control-plane-64-bit"), Language::Rust, source).unwrap();
    assert_eq!(found.len(), 1);
    assert!(found[0].numbered.starts_with("Review lines 1, 3.\n"));
}

#[test]
fn too_many_arguments_query_captures_four_parameters_but_not_three_or_self() {
    let source = "fn three(a: u8, b: u8, c: u8) {}\nfn four(a: u8, b: u8, c: u8, d: u8) {}\nimpl S { fn m(&self, a: u8, b: u8, c: u8) {} }\n";
    let found = snippets(&rule("too-many-arguments"), Language::Rust, source).unwrap();
    assert_eq!(found.len(), 1);
    assert!(found[0].numbered.contains("fn four"));
}

#[test]
fn test_names_rule_reviews_only_test_functions_together() {
    let source =
        "#[test]\nfn works() {}\n\nfn helper() {}\n\n#[test]\nfn rejects_empty_input() {}\n";
    let found = snippets(&rule("test-names-it-should"), Language::Rust, source).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(
        found[0].numbered,
        "    2 | works\n  ...\n    7 | rejects_empty_input\n"
    );
}

#[test]
fn path_scoped_rule_reads_only_its_crates() {
    let rule = rule("no-terminal-logic-in-pixelflow");
    assert!(
        rule.files
            .contains(&PathBuf::from("pixelflow-core/src/lib.rs"))
    );
    assert!(
        rule.files
            .contains(&PathBuf::from("./pixelflow-graphics/src/fonts/cache.rs"))
    );
    assert!(!rule.files.contains(&PathBuf::from("core-term/src/main.rs")));
}

#[test]
fn excluded_paths_are_not_read() {
    let simd = rule("simd-is-codegens");
    assert!(
        simd.files
            .contains(&PathBuf::from("pixelflow-codegen/src/jit_cache.rs"))
    );
    assert!(
        !simd
            .files
            .contains(&PathBuf::from("pixelflow-codegen/src/emit/x86.rs"))
    );
    let unwrap = rule("panicking-unwrap");
    assert!(
        !unwrap
            .files
            .contains(&PathBuf::from("desloppify/tests/shipped_rules.rs"))
    );
}

#[test]
fn file_review_sends_the_whole_file_naming_matched_lines_and_skips_files_without_any() {
    let unwrap = rule("panicking-unwrap");
    let source = "fn f() {\n    let a = x.unwrap();\n    let b = y.expect(\"y\");\n}\n";
    let found = snippets(&unwrap, Language::Rust, source).unwrap();
    assert_eq!(found.len(), 1);
    assert!(
        found[0]
            .numbered
            .starts_with("Review lines 2, 3.\n\n    1 | fn f() {\n")
    );
    assert!(
        snippets(&unwrap, Language::Rust, "fn g() {}\n")
            .unwrap()
            .is_empty()
    );
}
