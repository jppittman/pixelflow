//! The rules and skills in this crate load, and each sees the unit it claims
//! to. Catches a bad rule file or a dangling skill name before a review does.

use std::path::Path;
use std::path::PathBuf;

use desloppify::language::Language;
use desloppify::review::plan;
use desloppify::rule::{self, Rule, Verdict};
use desloppify::skills;
use desloppify::snippet::{outline, snippets};

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
fn every_shipped_rule_lists_its_fine_outcomes_before_its_violations() {
    for rule in shipped() {
        let verdicts: Vec<_> = rule.decision.outcomes.iter().map(|o| o.verdict).collect();
        let first_violation = verdicts
            .iter()
            .position(|v| *v == Verdict::Violation)
            .unwrap();
        assert!(first_violation > 0, "{} has no fine outcome", rule.id);
        assert!(
            verdicts[first_violation..]
                .iter()
                .all(|v| *v == Verdict::Violation),
            "{}",
            rule.id
        );
    }
}

#[test]
fn signatures_are_shown_without_bodies_joined_in_one_snippet() {
    let source = "pub fn a(x: u8) -> u8 {\n    x\n}\nfn b() {}\n";
    let found = snippets(&rule("boolean-argument"), Language::Rust, source).unwrap();
    assert_eq!(
        found,
        [desloppify::snippet::Snippet {
            first_line: 1,
            numbered: "    1 | pub fn a(x: u8) -> u8\n  ...\n    4 | fn b()\n".into()
        }]
    );
}

#[test]
fn types_are_reviewed_each_alone() {
    let source = "struct A { id: u32 }\nfn f() {}\nenum B { X }\n";
    let found = snippets(&rule("control-plane-64-bit"), Language::Rust, source).unwrap();
    let lines: Vec<_> = found.iter().map(|s| s.first_line).collect();
    assert_eq!(lines, [1, 3]);
}

#[test]
fn functions_are_reviewed_each_alone_whole() {
    let source = "fn f() {\n    let a = x.unwrap();\n}\nfn g() {}\n";
    let found = snippets(&rule("panicking-unwrap"), Language::Rust, source).unwrap();
    assert_eq!(found.len(), 2);
    assert_eq!(
        found[0].numbered,
        "    1 | fn f() {\n    2 |     let a = x.unwrap();\n    3 | }\n"
    );
}

#[test]
fn functions_inside_a_cfg_test_module_are_not_reviewed() {
    let source =
        "fn a() {}\n#[cfg(test)]\nmod tests {\n    fn t() {}\n}\nmod inner {\n    fn b() {}\n}\n";
    let found = snippets(&rule("guard-clauses"), Language::Rust, source).unwrap();
    let lines: Vec<_> = found.iter().map(|s| s.first_line).collect();
    assert_eq!(lines, [1, 7]);
}

#[test]
fn an_outline_keeps_every_line_but_function_bodies_at_its_own_number() {
    let source = "#[test]\nfn works() {\n    assert!(true);\n}\n\npub struct S;\nimpl S {\n    fn m(&self) {\n        inner();\n    }\n}\n";
    let shown = outline(Language::Rust, source).unwrap();
    assert_eq!(
        shown.numbered,
        "    1 | #[test]\n    2 | fn works() { … }\n    5 | \n    6 | pub struct S;\n    7 | impl S {\n    8 |     fn m(&self) { … }\n   11 | }\n"
    );
    let found = snippets(&rule("test-names-it-should"), Language::Rust, source).unwrap();
    assert_eq!(found, [shown]);
}

#[test]
fn a_function_nested_in_a_body_is_elided_with_it() {
    let source = "fn outer() {\n    fn inner() {\n    }\n}\nfn after() {}\n";
    let shown = outline(Language::Rust, source).unwrap();
    assert_eq!(
        shown.numbered,
        "    1 | fn outer() { … }\n    5 | fn after() { … }\n"
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
    let registers = rule("registers-come-from-the-allocator");
    assert!(
        registers
            .files
            .contains(&PathBuf::from("pixelflow-codegen/src/emit/x86_64.rs"))
    );
    assert!(
        !registers
            .files
            .contains(&PathBuf::from("pixelflow-codegen/src/emit/regalloc.rs"))
    );
    let unwrap = rule("panicking-unwrap");
    assert!(
        !unwrap
            .files
            .contains(&PathBuf::from("desloppify/tests/shipped_rules.rs"))
    );
}

/// A scratch directory, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("desloppify-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn write(&self, name: &str, text: &str) -> PathBuf {
        let path = self.0.join(name);
        std::fs::write(&path, text).unwrap();
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.0) {
            eprintln!("leaving {}: {error}", self.0.display());
        }
    }
}

#[test]
fn module_root_context_attaches_the_mod_rs_outline_to_an_implementation_file() {
    let dir = Scratch::new("root");
    dir.write("mod.rs", "mod leaf;\npub fn made() -> u8 {\n    1\n}\n");
    let leaf = dir.write("leaf.rs", "pub fn exposed() {}\n");

    let rules = shipped();
    let calls = plan(&rules, &[leaf]).unwrap();
    let call = calls
        .iter()
        .find(|c| {
            c.rules
                .iter()
                .any(|&r| rules[r].id == "interface-lives-in-mod-rs")
        })
        .unwrap();
    let root = call.root.as_ref().unwrap();
    assert_eq!(root.path, dir.0.join("mod.rs"));
    assert_eq!(
        root.text,
        "    1 | mod leaf;\n    2 | pub fn made() -> u8 { … }\n"
    );
}

#[test]
fn rules_that_see_the_same_unit_share_its_calls() {
    let dir = Scratch::new("shared");
    let file = dir.write("a.rs", "fn a() {}\nfn b() {}\n");

    let rules = shipped();
    let calls = plan(&rules, &[file]).unwrap();
    let by_function: Vec<_> = calls
        .iter()
        .filter(|c| c.rules.iter().any(|&r| rules[r].id == "guard-clauses"))
        .collect();
    assert_eq!(by_function.len(), 2, "one call per function");
    for call in by_function {
        let ids: Vec<_> = call.rules.iter().map(|&r| rules[r].id.as_str()).collect();
        assert!(ids.contains(&"magic-numbers"), "{ids:?}");
        assert!(ids.contains(&"silent-failure"), "{ids:?}");
    }
}
