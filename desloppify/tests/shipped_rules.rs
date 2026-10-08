//! The rules and skills in this crate load, and each rule shows a model the
//! unit of code it claims to — observed in what a review sends the model, and
//! for path scoping in what the binary reports from the repository root.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use desloppify::agent::{self, Answer, Ask, Question};
use desloppify::decide;
use desloppify::review::{Reviewers, plan, review};
use desloppify::rule::{self, Rule};
use desloppify::skills::{self, Skills};

const MIN_CONFIDENCE: f64 = 0.7;
const CONTEXT_MARKER: &str = "\nFor context only";

fn manifest() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn shipped_skills() -> Skills {
    skills::load(&manifest().join("skills")).unwrap()
}

#[test]
fn every_shipped_rule_loads() {
    let rules = rule::load_dir(&manifest().join("rules"), &shipped_skills()).unwrap();
    assert!(!rules.is_empty());
}

/// Answers as the dry-run backend does, recording the code each call was
/// shown.
struct Recording<A> {
    inner: A,
    shown: Mutex<Vec<String>>,
}

impl<A: Ask> Ask for Recording<A> {
    async fn ask(&self, question: &Question<'_>) -> Result<Answer> {
        self.shown.lock().unwrap().push(question.prompt.to_owned());
        self.inner.ask(question).await
    }
}

/// A scratch directory, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("desloppify-{name}-{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).unwrap();
        }
        std::fs::create_dir_all(dir.join("rules")).unwrap();
        Self(dir)
    }

    fn write(&self, name: &str, text: &str) -> PathBuf {
        let path = self.0.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, text).unwrap();
        path
    }

    /// Copies the shipped rule `id` into this directory's `rules/`. With
    /// `Scoping::Dropped` its `paths` and `exclude` are removed, so it reads
    /// the scratch files it is given.
    fn copy_rule(&self, id: &str, scoping: Scoping) {
        let shipped = manifest().join("rules").join(format!("{id}.json"));
        let mut json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(shipped).unwrap()).unwrap();
        if scoping == Scoping::Dropped {
            let fields = json.as_object_mut().unwrap();
            fields.remove("paths");
            fields.remove("exclude");
        }
        self.write(&format!("rules/{id}.json"), &json.to_string());
    }

    fn rules(&self) -> Vec<Rule> {
        rule::load_dir(&self.0.join("rules"), &shipped_skills()).unwrap()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.0) {
            eprintln!("leaving {}: {error}", self.0.display());
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Scoping {
    Kept,
    Dropped,
}

/// The prompts a dry review of `files` under the shipped rule `id` shows the
/// model, one per unit, sorted. Each unit is decided once, by the dry-run
/// model's first outcome — a fine one — so nothing is explained.
async fn shown(id: &str, files: &[(&str, &str)]) -> Vec<String> {
    let scratch = Scratch::new(id);
    scratch.copy_rule(id, Scoping::Dropped);
    let paths: Vec<_> = files.iter().map(|(p, t)| scratch.write(p, t)).collect();
    let rules = scratch.rules();
    let calls = plan(&rules, &paths).unwrap();
    let reviewers = Arc::new(Reviewers {
        ask: Recording {
            inner: agent::dry_run(),
            shown: Mutex::new(Vec::new()),
        },
        decide: decide::none(),
        min_confidence: MIN_CONFIDENCE,
    });
    let report = review(reviewers.clone(), Arc::new(rules), calls)
        .await
        .unwrap();
    assert!(report.failures.is_empty() && report.findings.is_empty());
    let mut shown = reviewers.ask.shown.lock().unwrap().clone();
    shown.sort();
    shown
}

/// The numbered code in a prompt: after the `File:` line, before any
/// context.
fn code(prompt: &str) -> &str {
    let (_, code) = prompt.split_once("\n\n").unwrap();
    code.split(CONTEXT_MARKER).next().unwrap()
}

#[tokio::test]
async fn signatures_are_shown_without_bodies_all_of_a_file_in_one_call() {
    let shown = shown(
        "boolean-argument",
        &[("a.rs", "pub fn a(x: u8) -> u8 {\n    x\n}\nfn b() {}\n")],
    )
    .await;
    assert_eq!(shown.len(), 1);
    assert_eq!(
        code(&shown[0]),
        "    1 | pub fn a(x: u8) -> u8\n  ...\n    4 | fn b()\n"
    );
}

#[tokio::test]
async fn types_are_shown_each_alone() {
    let shown = shown(
        "control-plane-64-bit",
        &[("a.rs", "struct A { id: u32 }\nfn f() {}\nenum B { X }\n")],
    )
    .await;
    let codes: Vec<_> = shown.iter().map(|p| code(p)).collect();
    assert_eq!(
        codes,
        ["    1 | struct A { id: u32 }\n", "    3 | enum B { X }\n"]
    );
}

#[tokio::test]
async fn functions_are_shown_each_alone_and_whole() {
    let shown = shown(
        "panicking-unwrap",
        &[("a.rs", "fn f() {\n    let a = x.unwrap();\n}\nfn g() {}\n")],
    )
    .await;
    let codes: Vec<_> = shown.iter().map(|p| code(p)).collect();
    assert_eq!(
        codes,
        [
            "    1 | fn f() {\n    2 |     let a = x.unwrap();\n    3 | }\n",
            "    4 | fn g() {}\n"
        ]
    );
}

#[tokio::test]
async fn functions_inside_a_cfg_test_module_are_never_shown() {
    let shown = shown(
        "guard-clauses",
        &[(
            "a.rs",
            "fn a() {}\n#[cfg(test)]\nmod tests {\n    fn t() {}\n}\nmod inner {\n    fn b() {}\n}\n",
        )],
    )
    .await;
    let codes: Vec<_> = shown.iter().map(|p| code(p)).collect();
    assert_eq!(codes, ["    1 | fn a() {}\n", "    7 | fn b() {}\n"]);
}

#[tokio::test]
async fn an_outline_keeps_every_line_but_function_bodies_at_its_own_number() {
    let shown = shown(
        "test-names-it-should",
        &[(
            "a.rs",
            "#[test]\nfn works() {\n    assert!(true);\n}\n\npub struct S;\nimpl S {\n    fn m(&self) {\n        fn inner() {}\n    }\n}\n",
        )],
    )
    .await;
    assert_eq!(shown.len(), 1);
    assert_eq!(
        code(&shown[0]),
        "    1 | #[test]\n    2 | fn works() { … }\n    5 | \n    6 | pub struct S;\n    7 | impl S {\n    8 |     fn m(&self) { … }\n   11 | }\n"
    );
}

#[tokio::test]
async fn an_implementation_file_is_shown_with_its_module_roots_outline() {
    let shown = shown(
        "interface-lives-in-mod-rs",
        &[
            ("m/mod.rs", "mod leaf;\npub fn made() -> u8 {\n    1\n}\n"),
            ("m/leaf.rs", "pub fn exposed() {}\n"),
        ],
    )
    .await;
    let leaf = shown.iter().find(|p| p.contains("leaf.rs\n")).unwrap();
    assert!(
        leaf.ends_with("mod.rs:\n\n    1 | mod leaf;\n    2 | pub fn made() -> u8 { … }\n"),
        "{leaf}"
    );
}

/// Each rule's decision tally, as the binary prints it, from a dry review of
/// `file` (relative to the repository root) under the shipped rules `ids`
/// with their scoping.
fn decided_from_the_repository_root(ids: &[&str], file: &str) -> String {
    let scratch = Scratch::new("cli");
    for id in ids {
        scratch.copy_rule(id, Scoping::Kept);
    }
    let output = Command::new(env!("CARGO_BIN_EXE_desloppify"))
        .current_dir(manifest().parent().unwrap())
        .args([
            "--backend",
            "dry-run",
            "--system-one",
            "dry-run",
            "--findings-only",
        ])
        .arg("--rules")
        .arg(scratch.0.join("rules"))
        .arg(file)
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    String::from_utf8(output.stderr).unwrap()
}

#[test]
fn a_path_scoped_rule_reads_only_the_paths_it_names_from_the_repository_root() {
    const RULES: [&str; 3] = [
        "simd-is-codegens",
        "registers-come-from-the-allocator",
        "no-terminal-logic-in-pixelflow",
    ];
    let reads = |file: &str| {
        let decided = decided_from_the_repository_root(&RULES, file);
        RULES
            .into_iter()
            .filter(|id| decided.contains(&format!("{id}: ")))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        reads("pixelflow-codegen/src/jit_cache.rs"),
        [
            "simd-is-codegens",
            "registers-come-from-the-allocator",
            "no-terminal-logic-in-pixelflow"
        ]
    );
    assert_eq!(
        reads("pixelflow-codegen/src/emit/x86_64.rs"),
        [
            "registers-come-from-the-allocator",
            "no-terminal-logic-in-pixelflow"
        ]
    );
    assert_eq!(
        reads("pixelflow-codegen/src/emit/regalloc.rs"),
        ["no-terminal-logic-in-pixelflow"]
    );
    assert!(reads("core-term/src/main.rs").is_empty());
}
