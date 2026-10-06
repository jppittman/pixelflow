//! The rules and skills in this crate load, and their queries find what they
//! claim to. Catches a bad query or a dangling skill name before a review does.

use std::path::Path;
use std::path::PathBuf;

use desloppify::language::Language;
use desloppify::review::plan;
use desloppify::rule::{self, Rule};
use desloppify::skills;
use desloppify::snippet::snippets;

fn shipped() -> Vec<Rule> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let skills = skills::load(&root.join("skills")).unwrap();
    rule::load_dir(&root.join("rules"), &skills).unwrap()
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
fn function_shapes_sees_signatures_without_bodies() {
    let source = "pub fn a(x: u8) -> u8 {\n    x\n}\nfn b() {}\n";
    let found = snippets(&rule("function-shapes"), Language::Rust, source).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(
        found[0].numbered,
        "    1 | pub fn a(x: u8) -> u8\n  ...\n    4 | fn b()\n"
    );
}

#[test]
fn types_scope_reviews_each_type_alone() {
    let source = "struct A { id: u32 }\nfn f() {}\nenum B { X }\n";
    let found = snippets(&rule("control-plane-64-bit"), Language::Rust, source).unwrap();
    assert_eq!(found.len(), 2);
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
fn function_review_sends_each_function_holding_a_match_naming_its_lines() {
    let unwrap = rule("panicking-unwrap");
    let source = "fn f() {\n    let a = x.unwrap();\n    let b = y.expect(\"y\");\n}\nfn g() {}\n";
    let found = snippets(&unwrap, Language::Rust, source).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(
        found[0].numbered,
        "Review lines 2, 3.\n\n    1 | fn f() {\n    2 |     let a = x.unwrap();\n    3 |     let b = y.expect(\"y\");\n    4 | }\n"
    );
    assert!(
        snippets(&unwrap, Language::Rust, "fn g() {}\n")
            .unwrap()
            .is_empty()
    );
}

#[test]
fn interface_rule_shows_signatures_of_everything_wider_than_pub_super() {
    let source = "pub fn a() {\n}\npub(super) fn b() {}\nfn c() {}\npub(crate) struct S;\n";
    let found = snippets(&rule("interface-lives-in-mod-rs"), Language::Rust, source).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(
        found[0].numbered,
        "    1 | pub fn a()\n  ...\n    5 | pub(crate) struct S;\n"
    );
}

#[test]
fn trait_rule_shows_public_inherent_method_signatures_but_not_trait_impls() {
    let source = "impl S {\n    pub fn a() {}\n    pub(super) fn b() {}\n}\nimpl T for S {\n    pub fn c() {}\n}\n";
    let found = snippets(&rule("behavior-through-the-trait"), Language::Rust, source).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].numbered, "    2 | pub fn a()\n");
}

#[test]
fn module_root_context_attaches_the_mod_rs_to_an_implementation_file() {
    let dir = std::env::temp_dir().join(format!("desloppify-root-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("mod.rs"), "mod leaf;\n").unwrap();
    let leaf = dir.join("leaf.rs");
    std::fs::write(&leaf, "pub fn exposed() {}\n").unwrap();

    let rules = shipped();
    let calls = plan(&rules, &[leaf]).unwrap();
    let call = calls
        .iter()
        .find(|c| rules[c.rule].id == "interface-lives-in-mod-rs")
        .unwrap();
    let root = call.root.as_ref().unwrap();
    assert_eq!(root.path, dir.join("mod.rs"));
    assert_eq!(root.text, "mod leaf;\n");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn functions_inside_a_cfg_test_module_are_not_reviewed() {
    let source =
        "fn a() {}\n#[cfg(test)]\nmod tests {\n    fn t() {}\n}\nmod inner {\n    fn b() {}\n}\n";
    let found = snippets(&rule("guard-clauses"), Language::Rust, source).unwrap();
    let names: Vec<_> = found.iter().map(|s| s.first_line).collect();
    assert_eq!(names, [1, 7]);
}
