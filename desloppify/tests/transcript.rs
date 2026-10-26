//! A scripted review of a tree that holds every layout the planner, namer and
//! synthesis treat differently, recorded as the questions asked of a model,
//! the requests made of System One, and the data `report.json` is written from.
//! Compared against `tests/golden/transcript.txt`.
//!
//! The golden is generated, never edited by hand: create it empty, then run
//! with `TRANSCRIPT_OUT=<path to tests/golden/transcript.txt>` to rewrite it.
//! A change that leaves it unchanged has changed no prompt, no schema, no
//! level and no reported datum *for this tree*; a layout the tree does not hold
//! is not pinned, so a new layout a change cares about is a new file here. A
//! change to the golden is a change to what a journal keys on, and wants a
//! reason in its own commit.
//!
//! The tree is built under the temp directory, and the crate-less `loose/`
//! files look upward for a `Cargo.toml` and an `intent.md`, so a temp directory
//! with either above it is refused rather than recorded.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use desloppify::agent::{Answer, Ask, Question, Usage};
use desloppify::decide::{Choice, Decide, Decided, Decisions};
use desloppify::model::ModelLevel;
use desloppify::review::{Session, plan};
use desloppify::rule;
use desloppify::skills;
use serde_json::{Value, json};

/// Prompts longer than this are recorded as their length and a hash.
const LONG: usize = 8 * 1024;
/// A whole file.
const FILE: &str = r#""unit": "file","#;
/// A whole file, scoped to the `edge` directory.
const EDGE: &str = r#""unit": "file", "paths": ["**/edge/**"],"#;
/// A crate's function signatures together.
const CRATE: &str = r#""unit": "function_signatures", "group": "crate","#;
/// Function lines in a root file big enough to overrun the namer's budget.
const BIG: usize = 6100;
/// One line short of the namer's budget (`MAX_SHAPE_LINES`): a one-line file
/// beside it lands exactly on the budget, which is still shown.
const JUST_UNDER: usize = 5999;

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

/// A rule whose unit (and whatever scopes it) is `shape`, a JSON fragment.
fn rule(shape: &str, levels: (u8, u8), id: &str) -> String {
    format!(
        r#"{{{shape} "levels": {{"decide": {}, "explain": {}}},
            "question": "rule:{id}", "fine": {{"clean": "c"}}, "violations": {{"wrong": "w"}}}}"#,
        levels.0, levels.1
    )
}

/// The crate-less `loose/` files look upward for a `Cargo.toml` and an
/// `intent.md`; one above the temp directory would be recorded, and the golden
/// would then be a function of the machine.
fn refuse_a_root_with_a_crate_above(root: &Path) {
    for dir in root.ancestors().skip(1) {
        for marker in ["Cargo.toml", "intent.md"] {
            assert!(
                !dir.join(marker).exists(),
                "{} holds a {marker}, so the crate-less fixture would record it: point TMPDIR elsewhere",
                dir.display()
            );
        }
    }
}

fn fns(count: usize) -> String {
    (0..count)
        .map(|i| format!("pub fn f{i}() {{}}\n"))
        .collect()
}

struct Tree(PathBuf);

impl Tree {
    fn new() -> Self {
        let root =
            std::env::temp_dir().join(format!("desloppify-transcript-{}", std::process::id()));
        refuse_a_root_with_a_crate_above(&root);
        if root.exists() {
            std::fs::remove_dir_all(&root).unwrap();
        }
        let rules = [
            ("alpha", rule(FILE, (1, 2), "alpha")),
            ("bravo", rule(FILE, (1, 2), "bravo")),
            (
                "rooted",
                rule(
                    r#""unit": "file", "context": "module_root","#,
                    (1, 2),
                    "rooted",
                ),
            ),
            (
                "intended",
                rule(
                    r#""unit": "file", "context": "module_intent","#,
                    (1, 2),
                    "intended",
                ),
            ),
            ("climb", rule(FILE, (1, 3), "climb")),
            ("settled", rule(FILE, (1, 2), "settled")),
            ("shapes_a", rule(CRATE, (3, 3), "shapes_a")),
            ("shapes_b", rule(CRATE, (3, 3), "shapes_b")),
            ("stuck", rule(EDGE, (1, 2), "stuck")),
            ("bogus", rule(EDGE, (1, 2), "bogus")),
            ("mislabeled", rule(EDGE, (1, 2), "mislabeled")),
            (
                "shapes_ctx",
                rule(
                    r#""unit": "function_signatures", "group": "crate", "context": "module_intent","#,
                    (3, 3),
                    "shapes_ctx",
                ),
            ),
        ];
        for (id, json) in rules {
            write(&root.join("rules").join(format!("{id}.json")), &json);
        }
        let files: Vec<(&str, String)> = vec![
            ("k/Cargo.toml", "[package]\nname = \"k\"\n".into()),
            ("k/intent.md", "# k\n\n**Is.** The crate.\n".into()),
            (
                "k/src/lib.rs",
                "mod m;\nmod n;\nmod top;\npub fn lib_root() {\n    body();\n}\n".into(),
            ),
            ("k/src/main.rs", "fn main() {\n    run();\n}\n".into()),
            (
                "k/src/top.rs",
                "fn top() {}\nfn top2(x: u8) {}\n// FAIL_S1\n".into(),
            ),
            (
                "k/src/m/mod.rs",
                "mod a;\nmod b;\npub fn m_root() {\n    body();\n}\n".into(),
            ),
            ("k/src/m/a.rs", "pub fn a() {\n    body();\n}\n".into()),
            ("k/src/m/b.rs", "pub fn b(x: u8) {}\n".into()),
            (
                "k/src/m/intent.md",
                "# k/src/m\n\n**Is.** The module.\n".into(),
            ),
            ("k/src/m/deep/c.rs", "fn c() {}\nfn c2() {}\n".into()),
            ("k/src/n.rs", "mod d;\npub fn n_root() {}\n".into()),
            ("k/src/n/d.rs", "fn d() {}\nfn d2() {}\n".into()),
            // A directory holding `mod.rs` beside `main.rs`: its roots are
            // outlined in path order, not in the order the root names are listed.
            (
                "k/src/multi/mod.rs",
                "mod a;\npub fn multi_root() {}\n".into(),
            ),
            ("k/src/multi/main.rs", "fn main() {}\n".into()),
            ("k/src/multi/a.rs", "pub fn multi_a() {}\n".into()),
            // A directory whose root file and 2018-layout sibling both exist:
            // the root in the directory wins.
            ("k/src/both.rs", "mod e;\npub fn both_sibling() {}\n".into()),
            (
                "k/src/both/mod.rs",
                "mod e;\npub fn both_root() {}\n".into(),
            ),
            ("k/src/both/e.rs", "pub fn both_e() {}\n".into()),
            // A crate inside a crate: its files belong to the nearer one.
            (
                "k/vendor/Cargo.toml",
                "[package]\nname = \"vendor\"\n".into(),
            ),
            ("k/vendor/src/lib.rs", "pub fn vendored(x: u8) {}\n".into()),
            ("k/vendor/src/v.rs", "pub fn v() {}\n".into()),
            // A module whose outline lands exactly on the namer's budget.
            ("k/src/bnd/mod.rs", fns(JUST_UNDER)),
            ("k/src/bnd/y.rs", "fn y() {}\n".into()),
            ("k/src/w1/mod.rs", fns(BIG)),
            ("k/src/w1/y.rs", "fn y() {}\n".into()),
            ("k/src/w2/mod.rs", fns(BIG)),
            ("k/src/w2/y.rs", "fn y() {}\n".into()),
            (
                "k/src/edge/mod.rs",
                "mod a;\npub fn edge_root() {}\n".into(),
            ),
            ("k/src/edge/a.rs", "pub fn edge_a() {}\n".into()),
            ("loose/z.rs", "fn z() {}\n".into()),
            ("loose/zz.rs", "fn zz() {}\n// FAIL_EXPLAIN\n".into()),
            ("notes/readme.md", "# not code\n".into()),
            ("other/Cargo.toml", "[package]\nname = \"other\"\n".into()),
        ];
        for (path, text) in files {
            write(&root.join(path), &text);
        }
        write(&root.join("other/src/lib.rs"), "pub fn other(x: u8) {}\n");
        Self(root)
    }

    fn files(&self) -> Vec<PathBuf> {
        let mut found = Vec::new();
        collect(&self.0, &mut found);
        found.retain(|p| !p.starts_with(self.0.join("rules")));
        found.sort();
        found
    }
}

fn collect(dir: &Path, found: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect(&path, found);
        } else {
            found.push(path);
        }
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.0) {
            eprintln!("leaving {}: {error}", self.0.display());
        }
    }
}

/// One question as asked: what a journal keys on.
#[derive(Clone)]
struct Asked {
    level: ModelLevel,
    system: String,
    prompt: String,
    schema: String,
}

struct Scripted {
    asked: Mutex<Vec<Asked>>,
}

fn has(schema: &Value, property: &str) -> bool {
    schema["properties"].get(property).is_some()
}

fn count(prompt: &str, marker: &str) -> usize {
    prompt.lines().filter(|l| l.starts_with(marker)).count()
}

fn reply(question: &Question<'_>) -> String {
    let schema = question.schema.unwrap();
    let (system, prompt) = (question.system, question.prompt);
    if has(schema, "outcome") {
        let outcome = match () {
            () if system.contains("rule:settled") => "clean",
            () if system.contains("rule:climb") && question.level == ModelLevel::Lite => "unsure",
            () if system.contains("rule:stuck") => "unsure",
            () if system.contains("rule:bogus") => "bogus",
            () => "wrong",
        };
        return json!({ "outcome": outcome }).to_string();
    }
    if has(schema, "findings") {
        let called = prompt.lines().find_map(|l| l.strip_prefix("File: "));
        let header = prompt
            .lines()
            .filter_map(|l| l.strip_prefix("== ").and_then(|l| l.strip_suffix(" ==")))
            .find(|p| p.ends_with("b.rs"))
            .or_else(|| {
                prompt
                    .lines()
                    .find_map(|l| l.strip_prefix("== ").and_then(|l| l.strip_suffix(" ==")))
            });
        // These two name no file for the crate `k`, so their findings are
        // attributed to the crate's directory.
        let nameless = called.is_some_and(|file| file.ends_with("/k"))
            && (system.contains("rule:shapes_b") || system.contains("rule:shapes_ctx"));
        let path = header.filter(|_| !nameless);
        let place = |line: u64, message: &str| match path {
            Some(path) => {
                json!({"outcome": "wrong", "path": path, "line": line, "message": message})
            }
            None => json!({"outcome": "wrong", "line": line, "message": message}),
        };
        let mut found = vec![place(1, "one"), place(2, "two")];
        // A model names a file that is not there.
        if let Some(file) =
            called.filter(|f| system.contains("rule:bravo") && f.ends_with("/m/b.rs"))
        {
            let ghost = file.replace("b.rs", "ghost.rs");
            found.push(json!({"outcome": "wrong", "path": ghost, "line": 3, "message": "ghost"}));
        }
        return json!({ "findings": found }).to_string();
    }
    if has(schema, "things") {
        return json!({ "things": [
            {"thing": "an assembler", "context": "in a JIT", "symptoms": [1, 2, 3]},
            {"thing": "a register allocator", "context": "in a JIT", "symptoms": [2, 3]},
            {"thing": "a parser", "context": "in a compiler", "symptoms": [1, 1, 2]},
            {"thing": "a lexer", "context": "in a compiler", "symptoms": [3]},
        ] })
        .to_string();
    }
    if has(schema, "description") {
        return json!({ "description": format!("Described: {prompt}") }).to_string();
    }
    let things = count(prompt, "### T");
    let findings = count(prompt, "- F");
    let diagnoses: Vec<Value> = (1..=things)
        .map(|t| {
            json!({"thing": t, "diagnosis": format!("d{t}"), "falls_out": "f",
                   "explains": [1, findings, 1]})
        })
        .collect();
    json!({
        "diagnoses": diagnoses,
        "roots": [{"root": "r", "falls_out": "f", "explains": [1, things]}],
        "review": "# Review\n",
    })
    .to_string()
}

impl Ask for Scripted {
    async fn ask(&self, question: &Question<'_>) -> Result<Answer> {
        self.asked.lock().unwrap().push(Asked {
            level: question.level,
            system: question.system.to_owned(),
            prompt: question.prompt.to_owned(),
            schema: question.schema.unwrap().to_string(),
        });
        if question.prompt.contains("// FAIL_EXPLAIN") {
            anyhow::bail!("scripted failure");
        }
        Ok(Answer {
            text: reply(question),
            usage: Usage {
                calls: 1,
                input: 100,
                output: 10,
            },
        })
    }
}

/// Settles `settled` as clean, fails on `FAIL_S1`, answers nothing else.
struct SystemOne {
    requested: Mutex<Vec<String>>,
}

impl Decide for SystemOne {
    async fn decide(&self, state: &str, questions: &[Choice<'_>]) -> Result<Decisions> {
        let asked: Vec<String> = questions
            .iter()
            .map(|q| {
                let labels: Vec<String> = q
                    .labels
                    .iter()
                    .map(|(label, meaning)| format!("{label}={meaning}"))
                    .collect();
                format!("{}: {} [{}]", q.name, q.instructions, labels.join("; "))
            })
            .collect();
        self.requested
            .lock()
            .unwrap()
            .push(format!("state {state} | {}", asked.join(" | ")));
        if state.contains("FAIL_S1") {
            anyhow::bail!("503 service unavailable");
        }
        let answers = questions
            .iter()
            .filter_map(|q| {
                let label = match q.name {
                    "settled" => "clean",
                    "mislabeled" => "nonsense",
                    _ => return None,
                };
                Some((
                    q.name.to_owned(),
                    Decided {
                        label: label.into(),
                        confidence: 0.9,
                    },
                ))
            })
            .collect();
        Ok(Decisions {
            answers,
            usage: Usage {
                calls: 1,
                input: 50,
                output: 0,
            },
        })
    }
}

fn fnv(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn shown(text: &str, root: &str) -> String {
    let text = text.replace(root, "<root>");
    if text.len() <= LONG {
        return text;
    }
    format!("<{} bytes, fnv {:016x}>", text.len(), fnv(&text))
}

/// Where `got` first departs from `want`, so a failure on one platform can be
/// read from the log.
fn first_difference(got: &str, want: &str) -> String {
    let clip = |line: Option<&str>| {
        line.map_or_else(
            || "<end of transcript>".to_owned(),
            |l| l.chars().take(300).collect(),
        )
    };
    let (mut got_lines, mut want_lines) = (got.lines(), want.lines());
    let mut number = 1;
    loop {
        let (g, w) = (got_lines.next(), want_lines.next());
        if g != w {
            return format!(
                "the transcript changed at line {number}:\n  got:  {}\n  want: {}\nrerun with TRANSCRIPT_OUT=<path to tests/golden/transcript.txt> and diff",
                clip(g),
                clip(w)
            );
        }
        number += 1;
    }
}

fn usage(u: Usage) -> Value {
    json!([u.calls, u.input, u.output])
}

#[tokio::test]
async fn a_scripted_review_asks_the_recorded_questions_and_reports_the_recorded_data() {
    let tree = Tree::new();
    let root = tree.0.display().to_string();
    let skills = skills::load(&tree.0.join("skills")).unwrap();
    let rules = rule::load_dir(&tree.0.join("rules"), &skills).unwrap();
    let calls = plan(rules, &tree.files()).unwrap();

    let reviewers = Arc::new(Session::new(
        Scripted {
            asked: Mutex::new(Vec::new()),
        },
        SystemOne {
            requested: Mutex::new(Vec::new()),
        },
    ));
    let mut report = reviewers.clone().review(calls).await.unwrap();
    let identify_usage = reviewers.identify(&mut report).await.unwrap();
    let synthesis = reviewers.synthesize(&report).await.unwrap().unwrap();

    let mut questions: Vec<String> = reviewers
        .ask
        .asked
        .lock()
        .unwrap()
        .iter()
        .map(|a| {
            format!(
                "level {} | schema {} | system {} | prompt {}",
                u64::from(a.level),
                a.schema,
                shown(&a.system, &root),
                shown(&a.prompt, &root)
            )
        })
        .collect();
    questions.sort();
    let mut requests: Vec<String> = reviewers
        .decide
        .requested
        .lock()
        .unwrap()
        .iter()
        .map(|request| shown(request, &root))
        .collect();
    requests.sort();
    let mut failures: Vec<String> = report
        .failures
        .iter()
        .map(|f| format!("{f:#}").replace(&root, "<root>"))
        .collect();
    failures.sort();

    let data = json!({
        "findings": report.findings(),
        "things": report.things(),
        "diagnoses": synthesis.diagnoses,
        "roots": synthesis.roots,
        "decisions": report.decisions,
        "escalated": report.escalated,
        "usage": report.usage,
        "failures": failures,
        "system_one": usage(report.system_one),
        "identify": usage(identify_usage),
        "lead": usage(synthesis.usage),
        "review": synthesis.review,
    });
    let data = serde_json::to_string_pretty(&data)
        .unwrap()
        .replace(&root, "<root>");
    let transcript = format!(
        "== system one requests ==\n{}\n\n== questions ({}) ==\n{}\n\n== data ==\n{data}\n",
        requests.join("\n"),
        questions.len(),
        questions.join("\n---\n"),
    );
    if let Ok(path) = std::env::var("TRANSCRIPT_OUT") {
        std::fs::write(path, &transcript).unwrap();
        return;
    }
    let golden = include_str!("golden/transcript.txt");
    if transcript != golden {
        panic!("{}", first_difference(&transcript, golden));
    }
}
