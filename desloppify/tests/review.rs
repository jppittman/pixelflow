//! `plan` and `review` driven through their public API, with a scripted
//! `Ask` standing in for a model.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Result, bail};
use desloppify::agent::Ask;
use desloppify::model::ModelLevel;
use desloppify::review::{plan, review};
use desloppify::rule::{self, Rule};
use desloppify::skills;

/// What a call was asked.
struct Asked {
    level: ModelLevel,
    preamble: String,
    prompt: String,
}

/// Answers every prompt with `reply(prompt)`, recording what it was asked.
struct Scripted {
    reply: fn(&str) -> Result<String>,
    asked: Mutex<Vec<Asked>>,
}

impl Scripted {
    fn new(reply: fn(&str) -> Result<String>) -> Arc<Self> {
        Arc::new(Self {
            reply,
            asked: Mutex::new(Vec::new()),
        })
    }
}

impl Ask for Scripted {
    async fn ask(&self, level: ModelLevel, preamble: &str, prompt: &str) -> Result<String> {
        self.asked.lock().unwrap().push(Asked {
            level,
            preamble: preamble.to_owned(),
            prompt: prompt.to_owned(),
        });
        (self.reply)(prompt)
    }
}

/// A scratch tree: `rules/` holding the given rules, plus the given files.
struct Tree(PathBuf);

impl Tree {
    fn new(name: &str, rules: &[(&str, &str)], files: &[(&str, &str)]) -> Self {
        let root = std::env::temp_dir().join(format!("desloppify-{name}-{}", std::process::id()));
        if root.exists() {
            std::fs::remove_dir_all(&root).unwrap();
        }
        for (id, json) in rules {
            write(&root.join("rules").join(format!("{id}.json")), json);
        }
        for (path, text) in files {
            write(&root.join(path), text);
        }
        Self(root)
    }

    fn rules(&self) -> Vec<Rule> {
        let skills = skills::load(&self.0.join("skills")).unwrap();
        rule::load_dir(&self.0.join("rules"), &skills).unwrap()
    }

    fn path(&self, relative: &str) -> PathBuf {
        self.0.join(relative)
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        // A leftover scratch tree is harmless; a panic in drop is not.
        if let Err(error) = std::fs::remove_dir_all(&self.0) {
            eprintln!("leaving {}: {error}", self.0.display());
        }
    }
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

const WHOLE_FILE: &str = r#"{"level": 2, "scope": "file", "prompt": "Flag everything."}"#;

async fn run(tree: &Tree, files: &[&str], agent: Arc<Scripted>) -> desloppify::review::Report {
    let rules = tree.rules();
    let files: Vec<_> = files.iter().map(|f| tree.path(f)).collect();
    let calls = plan(&rules, &files).unwrap();
    review(agent, Arc::new(rules), calls).await.unwrap()
}

#[tokio::test]
async fn findings_come_back_sorted_by_path_then_line_with_their_rule() {
    let tree = Tree::new(
        "sorted",
        &[("everything", WHOLE_FILE)],
        &[("b.rs", "fn b() {}\n"), ("a.rs", "fn a() {}\n")],
    );
    let agent = Scripted::new(|_| {
        Ok(r#"[{"line": 9, "message": "late"}, {"line": 1, "message": "early"}]"#.into())
    });
    let report = run(&tree, &["b.rs", "a.rs"], agent).await;

    let seen: Vec<_> = report
        .findings
        .iter()
        .map(|f| {
            (
                f.path.file_name().unwrap().to_str().unwrap(),
                f.line,
                f.rule.as_str(),
            )
        })
        .collect();
    assert_eq!(
        seen,
        [
            ("a.rs", 1, "everything"),
            ("a.rs", 9, "everything"),
            ("b.rs", 1, "everything"),
            ("b.rs", 9, "everything"),
        ]
    );
    assert!(report.failures.is_empty());
}

#[tokio::test]
async fn a_fenced_reply_is_read_like_a_bare_one() {
    let tree = Tree::new(
        "fenced",
        &[("everything", WHOLE_FILE)],
        &[("a.rs", "fn a() {}\n")],
    );
    let agent = Scripted::new(|_| Ok("```json\n[{\"line\": 1, \"message\": \"m\"}]\n```".into()));
    let report = run(&tree, &["a.rs"], agent).await;
    assert_eq!(report.findings.len(), 1);
    assert_eq!(report.findings[0].message, "m");
}

#[tokio::test]
async fn a_prose_reply_is_a_failure_and_the_other_calls_still_count() {
    let tree = Tree::new(
        "prose",
        &[("everything", WHOLE_FILE)],
        &[("good.rs", "fn good() {}\n"), ("bad.rs", "fn bad() {}\n")],
    );
    let agent = Scripted::new(|prompt| {
        if prompt.contains("bad.rs") {
            return Ok("Looks fine to me!".into());
        }
        Ok(r#"[{"line": 1, "message": "m"}]"#.into())
    });
    let report = run(&tree, &["good.rs", "bad.rs"], agent).await;
    assert_eq!(report.findings.len(), 1);
    assert_eq!(report.failures.len(), 1);
    let failure = format!("{:#}", report.failures[0]);
    assert!(
        failure.contains("everything") && failure.contains("bad.rs"),
        "{failure}"
    );
}

#[tokio::test]
async fn a_failed_call_is_a_failure_not_an_abort() {
    let tree = Tree::new(
        "failed",
        &[("everything", WHOLE_FILE)],
        &[("a.rs", "fn a() {}\n")],
    );
    let agent = Scripted::new(|_| bail!("401 unauthorized"));
    let report = run(&tree, &["a.rs"], agent).await;
    assert!(report.findings.is_empty());
    assert_eq!(report.failures.len(), 1);
    assert!(format!("{:#}", report.failures[0]).contains("401"));
}

#[tokio::test]
async fn each_call_carries_its_rules_level_prompt_and_numbered_code() {
    let tree = Tree::new(
        "prompt",
        &[(
            "frontier",
            r#"{"level": 4, "scope": "file", "prompt": "Find the bug."}"#,
        )],
        &[("a.rs", "fn a() {}\nfn b() {}\n")],
    );
    let agent = Scripted::new(|_| Ok("[]".into()));
    run(&tree, &["a.rs"], agent.clone()).await;

    let asked = agent.asked.lock().unwrap();
    assert_eq!(asked.len(), 1);
    assert_eq!(asked[0].level, ModelLevel::Frontier);
    assert!(asked[0].preamble.ends_with("Find the bug."));
    assert!(asked[0].prompt.contains("a.rs"));
    assert!(asked[0].prompt.contains("    2 | fn b() {}\n"));
}

#[tokio::test]
async fn a_module_root_rule_shows_the_root_as_context_and_a_root_file_alone() {
    let tree = Tree::new(
        "root",
        &[(
            "contract",
            r#"{"level": 1, "scope": "file", "context": "module_root", "prompt": "p"}"#,
        )],
        &[("m/mod.rs", "mod leaf;\n"), ("m/leaf.rs", "fn leaf() {}\n")],
    );
    let agent = Scripted::new(|_| Ok("[]".into()));
    run(&tree, &["m/leaf.rs", "m/mod.rs"], agent.clone()).await;

    let asked = agent.asked.lock().unwrap();
    let leaf = asked
        .iter()
        .find(|a| a.prompt.contains("leaf.rs\n"))
        .unwrap();
    assert!(leaf.prompt.contains("For context only"));
    assert!(leaf.prompt.contains("mod leaf;"));
    let root = asked
        .iter()
        .find(|a| {
            a.prompt
                .starts_with(&format!("File: {}", tree.path("m/mod.rs").display()))
        })
        .unwrap();
    assert!(!root.prompt.contains("For context only"));
}

#[test]
fn files_in_no_known_language_or_outside_a_rules_paths_plan_nothing() {
    let tree = Tree::new(
        "plan",
        &[(
            "scoped",
            r#"{"level": 1, "scope": "file", "paths": ["**/inside/**"], "prompt": "p"}"#,
        )],
        &[
            ("inside/a.rs", "fn a() {}\n"),
            ("outside/b.rs", "fn b() {}\n"),
            ("inside/notes.md", "# notes\n"),
        ],
    );
    let files: Vec<_> = ["inside/a.rs", "outside/b.rs", "inside/notes.md"]
        .iter()
        .map(|f| tree.path(f))
        .collect();
    let calls = plan(&tree.rules(), &files).unwrap();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].path.ends_with("inside/a.rs"));
}
