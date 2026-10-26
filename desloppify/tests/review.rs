//! `plan`, `review`, `identify` and `synthesize` driven through their public API, with a
//! scripted `Ask` standing in for a model and a scripted `Decide` for
//! System One.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Result, bail};
use desloppify::agent::{self, Answer, Ask, Question, Usage};
use desloppify::decide::{self, Choice, Decide, Decided, Decisions};
use desloppify::model::ModelLevel;
use desloppify::review::{Report, Session, plan};
use desloppify::rule::{self, Rule};
use desloppify::skills;

/// What every scripted model call reports using.
const ONE_CALL: Usage = Usage {
    calls: 1,
    input: 100,
    output: 10,
};

/// What every scripted System One request reports using.
const ONE_REQUEST: Usage = Usage {
    calls: 1,
    input: 50,
    output: 0,
};

/// What a model call was asked to do, read from the schema it was held to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Decide,
    Explain,
    Identify,
    Describe,
    Synthesize,
}

/// What a model call was asked.
#[derive(Clone, Debug)]
struct Asked {
    level: ModelLevel,
    kind: Kind,
    preamble: String,
    prompt: String,
    schema: serde_json::Value,
}

fn kind(schema: Option<&serde_json::Value>) -> Kind {
    let schema = schema.expect("every call is held to a schema");
    let has = |property: &str| schema["properties"].get(property).is_some();
    match () {
        () if has("outcome") => Kind::Decide,
        () if has("findings") => Kind::Explain,
        () if has("things") => Kind::Identify,
        () if has("description") => Kind::Describe,
        () if has("review") => Kind::Synthesize,
        () => panic!("an unknown schema: {schema}"),
    }
}

/// Answers every model call with `reply(asked)`, recording what it was
/// asked.
struct Scripted {
    reply: fn(&Asked) -> Result<String>,
    asked: Mutex<Vec<Asked>>,
}

impl Scripted {
    fn new(reply: fn(&Asked) -> Result<String>) -> Self {
        Self {
            reply,
            asked: Mutex::new(Vec::new()),
        }
    }

    fn asked(&self) -> Vec<Asked> {
        self.asked.lock().unwrap().clone()
    }
}

impl Ask for Scripted {
    async fn ask(&self, question: &Question<'_>) -> Result<Answer> {
        let asked = Asked {
            level: question.level,
            kind: kind(question.schema),
            preamble: question.system.to_owned(),
            prompt: question.prompt.to_owned(),
            schema: question.schema.cloned().unwrap_or_default(),
        };
        self.asked.lock().unwrap().push(asked.clone());
        Ok(Answer {
            text: (self.reply)(&asked)?,
            usage: ONE_CALL,
        })
    }
}

/// A System One request: its state and its questions' names.
#[derive(Clone, Debug)]
struct Requested {
    state: String,
    questions: Vec<String>,
}

/// Answers each question with `answer(question)`, recording each request.
struct Deciding {
    answer: fn(&Choice<'_>) -> Option<(&'static str, f64)>,
    requested: Mutex<Vec<Requested>>,
}

impl Deciding {
    fn new(answer: fn(&Choice<'_>) -> Option<(&'static str, f64)>) -> Self {
        Self {
            answer,
            requested: Mutex::new(Vec::new()),
        }
    }
}

impl Decide for Deciding {
    async fn decide(&self, state: &str, questions: &[Choice<'_>]) -> Result<Decisions> {
        self.requested.lock().unwrap().push(Requested {
            state: state.to_owned(),
            questions: questions.iter().map(|q| q.name.to_owned()).collect(),
        });
        let answers = questions
            .iter()
            .filter_map(|q| {
                let (label, confidence) = (self.answer)(q)?;
                let decided = Decided {
                    label: label.to_owned(),
                    confidence,
                };
                Some((q.name.to_owned(), decided))
            })
            .collect();
        Ok(Decisions {
            answers,
            usage: ONE_REQUEST,
        })
    }
}

/// A System One that is down.
struct Down;

impl Decide for Down {
    async fn decide(&self, _: &str, _: &[Choice<'_>]) -> Result<Decisions> {
        bail!("503 service unavailable")
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

/// A whole-file rule, decided at level 1 and explained at level 2.
const EVERYTHING: &str = r#"{
    "unit": "file",
    "levels": {"decide": 1, "explain": 2},
    "question": "Is anything wrong here?",
    "fine": {"clean": "Nothing is wrong."},
    "violations": {"wrong": "Something is wrong."},
    "guidance": "Say what to change."
}"#;

const WRONG: &str = r#"{"outcome": "wrong"}"#;
const CLEAN: &str = r#"{"outcome": "clean"}"#;
const UNSURE: &str = r#"{"outcome": "unsure"}"#;

/// Decides every unit `wrong` and explains it with one finding on line 1.
fn wrong_on_line_one(asked: &Asked) -> Result<String> {
    Ok(match asked.kind {
        Kind::Decide => WRONG.into(),
        _ => r#"{"findings": [{"outcome": "wrong", "line": 1, "message": "needs work"}]}"#.into(),
    })
}

fn session<A: Ask, D: Decide>(ask: A, decide: D) -> Arc<Session<A, D>> {
    Arc::new(Session::new(ask, decide))
}

async fn run<A: Ask, D: Decide>(
    tree: &Tree,
    files: &[&str],
    session: &Arc<Session<A, D>>,
) -> Report {
    let files: Vec<_> = files.iter().map(|f| tree.path(f)).collect();
    let calls = plan(tree.rules(), &files).unwrap();
    session.clone().review(calls).await.unwrap()
}

/// Positions as report.json writes them.
fn written<T: serde::Serialize>(positions: &T) -> serde_json::Value {
    serde_json::to_value(positions).unwrap()
}

fn decided(report: &Report, rule: &str, outcome: &str) -> u64 {
    report
        .decisions
        .get(rule)
        .and_then(|d| d.get(outcome))
        .copied()
        .unwrap_or_default()
}

fn model_usage(report: &Report, rule: &str) -> Usage {
    report.usage.get(rule).copied().unwrap_or_default()
}

#[tokio::test]
async fn findings_come_back_sorted_by_path_then_line_with_their_rule_and_outcome() {
    let tree = Tree::new(
        "sorted",
        &[("everything", EVERYTHING)],
        &[("b.rs", "fn b() {}\n"), ("a.rs", "fn a() {}\n")],
    );
    let session = session(
        Scripted::new(|asked| {
            Ok(match asked.kind {
                Kind::Decide => WRONG.into(),
                _ => r#"{"findings": [{"outcome": "wrong", "line": 9, "message": "late"}, {"outcome": "wrong", "line": 1, "message": "early"}]}"#
                    .into(),
            })
        }),
        decide::none(),
    );
    let report = run(&tree, &["b.rs", "a.rs"], &session).await;

    let seen: Vec<_> = report
        .findings()
        .iter()
        .map(|f| {
            (
                f.path.file_name().unwrap().to_str().unwrap(),
                f.line,
                f.rule.as_str(),
                f.outcome.as_str(),
            )
        })
        .collect();
    assert_eq!(
        seen,
        [
            ("a.rs", 1, "everything", "wrong"),
            ("a.rs", 9, "everything", "wrong"),
            ("b.rs", 1, "everything", "wrong"),
            ("b.rs", 9, "everything", "wrong"),
        ]
    );
    assert!(report.failures.is_empty());
}

#[tokio::test]
async fn a_confident_fine_answer_from_system_one_settles_the_rule_without_a_model() {
    let tree = Tree::new(
        "settled",
        &[("everything", EVERYTHING)],
        &[("a.rs", "fn a() {}\n")],
    );
    let session = session(
        Scripted::new(wrong_on_line_one),
        Deciding::new(|_| Some(("clean", 0.9))),
    );
    let report = run(&tree, &["a.rs"], &session).await;

    assert!(session.ask.asked().is_empty());
    assert_eq!(decided(&report, "everything", "clean"), 1);
    assert!(report.escalated.is_empty());
    assert_eq!(model_usage(&report, "everything"), Usage::default());
    assert_eq!(report.system_one, ONE_REQUEST);
}

#[tokio::test]
async fn a_confident_violation_goes_straight_to_the_explainer_with_its_guidance() {
    let tree = Tree::new(
        "straight",
        &[("everything", EVERYTHING)],
        &[("a.rs", "fn a() {}\n")],
    );
    let session = session(
        Scripted::new(wrong_on_line_one),
        Deciding::new(|_| Some(("wrong", 0.95))),
    );
    let report = run(&tree, &["a.rs"], &session).await;

    let asked = session.ask.asked();
    assert_eq!(asked.len(), 1);
    assert_eq!(asked[0].kind, Kind::Explain);
    assert_eq!(asked[0].level, ModelLevel::Fast);
    assert!(asked[0].preamble.contains("Say what to change."));
    assert!(asked[0].preamble.contains("`wrong`: Something is wrong."));
    assert_eq!(report.findings().len(), 1);
    assert!(report.escalated.is_empty());
}

#[tokio::test]
async fn an_answer_below_the_minimum_confidence_goes_to_the_model_at_the_decide_level() {
    let tree = Tree::new(
        "doubtful",
        &[("everything", EVERYTHING)],
        &[("a.rs", "fn a() {}\n")],
    );
    let session = session(
        Scripted::new(|_| Ok(CLEAN.into())),
        Deciding::new(|_| Some(("wrong", 0.5))),
    );
    let report = run(&tree, &["a.rs"], &session).await;

    let asked = session.ask.asked();
    assert_eq!(asked.len(), 1);
    assert_eq!(
        (asked[0].kind, asked[0].level),
        (Kind::Decide, ModelLevel::Lite)
    );
    assert_eq!(report.escalated["everything"], 1);
    assert_eq!(decided(&report, "everything", "clean"), 1);
    assert!(report.findings().is_empty());
}

#[tokio::test]
async fn a_label_that_is_not_one_of_the_rules_outcomes_is_not_an_answer() {
    let tree = Tree::new(
        "bogus",
        &[("everything", EVERYTHING)],
        &[("a.rs", "fn a() {}\n")],
    );
    let session = session(
        Scripted::new(|_| Ok(CLEAN.into())),
        Deciding::new(|_| Some(("bogus", 1.0))),
    );
    let report = run(&tree, &["a.rs"], &session).await;
    assert_eq!(report.escalated["everything"], 1);
    assert_eq!(decided(&report, "everything", "clean"), 1);
}

const ESCALATING: &str = r#"{
    "unit": "file",
    "levels": {"decide": 1, "explain": 3},
    "question": "q",
    "fine": {"clean": "c"},
    "violations": {"wrong": "w"}
}"#;

#[tokio::test]
async fn unsure_asks_one_level_higher_up_to_the_explain_level_then_counts_as_unsure() {
    let tree = Tree::new(
        "unsure",
        &[("escalating", ESCALATING)],
        &[("a.rs", "fn a() {}\n")],
    );
    let session = session(Scripted::new(|_| Ok(UNSURE.into())), decide::none());
    let report = run(&tree, &["a.rs"], &session).await;

    let levels: Vec<_> = session.ask.asked().iter().map(|a| a.level).collect();
    assert_eq!(
        levels,
        [ModelLevel::Lite, ModelLevel::Fast, ModelLevel::Strong]
    );
    assert_eq!(decided(&report, "escalating", "unsure"), 1);
    assert!(report.findings().is_empty() && report.failures.is_empty());
    assert_eq!(report.usage["escalating"].calls, 3);
}

#[tokio::test]
async fn a_violation_decided_after_escalating_is_explained_at_the_explain_level() {
    let tree = Tree::new(
        "escalated",
        &[("escalating", ESCALATING)],
        &[("a.rs", "fn a() {}\n")],
    );
    let session = session(
        Scripted::new(|asked| {
            Ok(match (asked.kind, asked.level) {
                (Kind::Decide, ModelLevel::Lite) => UNSURE.into(),
                (Kind::Decide, _) => WRONG.into(),
                _ => r#"{"findings": [{"outcome": "wrong", "line": 1, "message": "m"}]}"#.into(),
            })
        }),
        decide::none(),
    );
    let report = run(&tree, &["a.rs"], &session).await;

    let asked: Vec<_> = session
        .ask
        .asked()
        .iter()
        .map(|a| (a.kind, a.level))
        .collect();
    assert_eq!(
        asked,
        [
            (Kind::Decide, ModelLevel::Lite),
            (Kind::Decide, ModelLevel::Fast),
            (Kind::Explain, ModelLevel::Strong),
        ]
    );
    assert_eq!(decided(&report, "escalating", "wrong"), 1);
    assert_eq!(report.findings().len(), 1);
}

#[tokio::test]
async fn rules_that_see_the_same_unit_share_one_system_one_request() {
    let tree = Tree::new(
        "shared",
        &[("first", EVERYTHING), ("second", EVERYTHING)],
        &[("a.rs", "fn a() {}\nfn b() {}\n")],
    );
    let session = session(
        Scripted::new(wrong_on_line_one),
        Deciding::new(|_| Some(("clean", 1.0))),
    );
    let report = run(&tree, &["a.rs"], &session).await;

    let requested = session.decide.requested.lock().unwrap().clone();
    assert_eq!(requested.len(), 1);
    assert_eq!(requested[0].questions, ["first", "second"]);
    assert!(requested[0].state.contains("a.rs"));
    assert!(requested[0].state.contains("    2 | fn b() {}\n"));
    assert_eq!(decided(&report, "first", "clean"), 1);
    assert_eq!(decided(&report, "second", "clean"), 1);
}

#[tokio::test]
async fn when_system_one_is_down_every_question_goes_to_the_model_and_the_outage_is_reported() {
    let tree = Tree::new(
        "down",
        &[("first", EVERYTHING), ("second", EVERYTHING)],
        &[("a.rs", "fn a() {}\n")],
    );
    let session = session(Scripted::new(|_| Ok(CLEAN.into())), Down);
    let report = run(&tree, &["a.rs"], &session).await;

    assert_eq!(session.ask.asked().len(), 2);
    assert_eq!(decided(&report, "first", "clean"), 1);
    assert_eq!(decided(&report, "second", "clean"), 1);
    assert_eq!(report.failures.len(), 1);
    let failure = format!("{:#}", report.failures[0]);
    assert!(
        failure.contains("System One") && failure.contains("503"),
        "{failure}"
    );
}

#[tokio::test]
async fn a_decision_outside_the_schema_is_a_failure_and_the_other_units_are_still_decided() {
    let tree = Tree::new(
        "prose",
        &[("everything", EVERYTHING)],
        &[("good.rs", "fn good() {}\n"), ("bad.rs", "fn bad() {}\n")],
    );
    let session = session(
        Scripted::new(|asked| {
            if asked.prompt.contains("bad.rs") {
                return Ok("Looks fine to me!".into());
            }
            Ok(CLEAN.into())
        }),
        decide::none(),
    );
    let report = run(&tree, &["good.rs", "bad.rs"], &session).await;

    assert_eq!(decided(&report, "everything", "clean"), 1);
    assert_eq!(report.failures.len(), 1);
    let failure = format!("{:#}", report.failures[0]);
    assert!(
        failure.contains("everything") && failure.contains("bad.rs"),
        "{failure}"
    );
}

#[tokio::test]
async fn an_explanation_outside_the_findings_schema_is_a_failure_but_the_decision_counts() {
    let tree = Tree::new(
        "schema",
        &[("everything", EVERYTHING)],
        &[("a.rs", "fn a() {}\n")],
    );
    let session = session(
        Scripted::new(|asked| {
            Ok(match asked.kind {
                Kind::Decide => WRONG.into(),
                _ => r#"[{"line": 1, "message": "a bare array"}]"#.into(),
            })
        }),
        decide::none(),
    );
    let report = run(&tree, &["a.rs"], &session).await;

    assert!(report.findings().is_empty());
    assert_eq!(report.failures.len(), 1);
    assert_eq!(decided(&report, "everything", "wrong"), 1);
}

/// A file rule with two ways to violate it.
const TWO_WAYS: &str = r#"{
    "unit": "file",
    "levels": {"decide": 1, "explain": 2},
    "question": "q",
    "fine": {"clean": "c"},
    "violations": {"loud": "It shouts.", "quiet": "It mumbles."}
}"#;

#[tokio::test]
async fn an_explanation_reports_each_place_under_the_violation_it_shows() {
    let tree = Tree::new(
        "two-ways",
        &[("rule", TWO_WAYS)],
        &[("a.rs", "fn a() {}\nfn b() {}\n")],
    );
    let session = session(
        Scripted::new(|asked| {
            Ok(match asked.kind {
                Kind::Decide => r#"{"outcome": "loud"}"#.into(),
                _ => r#"{"findings": [{"outcome": "loud", "line": 1, "message": "a"}, {"outcome": "quiet", "line": 2, "message": "b"}]}"#.into(),
            })
        }),
        decide::none(),
    );
    let report = run(&tree, &["a.rs"], &session).await;

    let outcomes: Vec<_> = report
        .findings()
        .iter()
        .map(|f| f.outcome.as_str())
        .collect();
    assert_eq!(outcomes, ["loud", "quiet"]);
    assert_eq!(decided(&report, "rule", "loud"), 1);
    let explained = session
        .ask
        .asked()
        .into_iter()
        .find(|a| a.kind == Kind::Explain)
        .unwrap();
    for violation in ["`loud`: It shouts.", "`quiet`: It mumbles."] {
        assert!(
            explained.preamble.contains(violation),
            "{}",
            explained.preamble
        );
    }
}

#[tokio::test]
async fn a_finding_under_an_outcome_that_is_not_a_violation_is_a_failure() {
    let tree = Tree::new(
        "not-a-violation",
        &[("rule", TWO_WAYS)],
        &[("a.rs", "fn a() {}\n")],
    );
    let session = session(
        Scripted::new(|asked| {
            Ok(match asked.kind {
                Kind::Decide => r#"{"outcome": "loud"}"#.into(),
                _ => r#"{"findings": [{"outcome": "clean", "line": 1, "message": "a"}]}"#.into(),
            })
        }),
        decide::none(),
    );
    let report = run(&tree, &["a.rs"], &session).await;
    assert!(report.findings().is_empty());
    assert_eq!(report.failures.len(), 1);
    assert!(format!("{:#}", report.failures[0]).contains("clean"));
}

#[tokio::test]
async fn a_failed_call_is_a_failure_not_an_abort_and_uses_nothing() {
    let tree = Tree::new(
        "failed",
        &[("everything", EVERYTHING)],
        &[("a.rs", "fn a() {}\n")],
    );
    let session = session(Scripted::new(|_| bail!("401 unauthorized")), decide::none());
    let report = run(&tree, &["a.rs"], &session).await;

    assert!(report.findings().is_empty());
    assert_eq!(report.failures.len(), 1);
    assert!(format!("{:#}", report.failures[0]).contains("401"));
    assert_eq!(model_usage(&report, "everything"), Usage::default());
}

#[tokio::test]
async fn a_model_decision_is_held_to_the_rules_outcomes_and_shown_the_numbered_code() {
    let tree = Tree::new(
        "prompt",
        &[("everything", EVERYTHING)],
        &[("a.rs", "fn a() {}\nfn b() {}\n")],
    );
    let session = session(Scripted::new(|_| Ok(CLEAN.into())), decide::none());
    run(&tree, &["a.rs"], &session).await;

    let asked = session.ask.asked();
    assert_eq!(asked.len(), 1);
    let preamble = &asked[0].preamble;
    assert!(preamble.contains("Is anything wrong here?"), "{preamble}");
    for outcome in [
        "`clean`: Nothing is wrong.",
        "`wrong`: Something is wrong.",
        "`unsure`",
    ] {
        assert!(preamble.contains(outcome), "{preamble}");
    }
    assert!(preamble.contains(
        "Outcomes:\n- `clean`: Nothing is wrong.\n- `wrong`: Something is wrong.\n- `unsure`: The code shown is not enough to decide.\n"
    ));
    assert_eq!(
        asked[0].schema["properties"]["outcome"]["enum"],
        serde_json::json!(["clean", "wrong", "unsure"])
    );
    assert!(asked[0].prompt.contains("a.rs"));
    assert!(asked[0].prompt.contains("    2 | fn b() {}\n"));
}

#[tokio::test]
async fn a_model_naming_an_outcome_the_rule_lacks_is_a_failure() {
    let tree = Tree::new(
        "unknown",
        &[("everything", EVERYTHING)],
        &[("a.rs", "fn a() {}\n")],
    );
    let session = session(
        Scripted::new(|_| Ok(r#"{"outcome": "bogus"}"#.into())),
        decide::none(),
    );
    let report = run(&tree, &["a.rs"], &session).await;
    assert_eq!(report.failures.len(), 1);
    assert!(format!("{:#}", report.failures[0]).contains("bogus"));
}

#[tokio::test]
async fn a_module_root_rule_shows_the_root_outline_as_context_and_a_root_file_alone() {
    let tree = Tree::new(
        "root",
        &[(
            "contract",
            r#"{"unit": "file", "levels": {"decide": 1, "explain": 1}, "context": "module_root",
                "question": "q", "fine": {"f": "f"}, "violations": {"v": "v"}}"#,
        )],
        &[
            ("m/mod.rs", "mod leaf;\nfn made() {\n    body();\n}\n"),
            ("m/leaf.rs", "fn leaf() {}\n"),
        ],
    );
    let session = session(Scripted::new(|_| Ok(UNSURE.into())), decide::none());
    run(&tree, &["m/leaf.rs", "m/mod.rs"], &session).await;

    let asked = session.ask.asked();
    let leaf = asked
        .iter()
        .find(|a| a.prompt.contains("leaf.rs\n"))
        .unwrap();
    assert!(leaf.prompt.contains("For context only"));
    assert!(
        leaf.prompt.contains("    2 | fn made() { … }\n"),
        "{}",
        leaf.prompt
    );
    assert!(!leaf.prompt.contains("body()"));
    let root = asked
        .iter()
        .find(|a| {
            a.prompt
                .starts_with(&format!("File: {}", tree.path("m/mod.rs").display()))
        })
        .unwrap();
    assert!(!root.prompt.contains("For context only"));
}

#[tokio::test]
async fn a_module_intent_rule_shows_the_nearest_intent_within_the_crate() {
    let tree = Tree::new(
        "intent",
        &[(
            "contract",
            r#"{"unit": "file", "levels": {"decide": 1, "explain": 1}, "context": "module_intent",
                "question": "q", "fine": {"f": "f"}, "violations": {"v": "v"}}"#,
        )],
        &[
            ("Cargo.toml", "[package]\nname = \"x\"\n"),
            ("intent.md", "# x\n\n**Is.** The crate.\n"),
            ("src/lib.rs", "mod m;\nmod top;\n"),
            ("src/top.rs", "fn top() {}\n"),
            ("src/m/mod.rs", "mod leaf;\n"),
            ("src/m/leaf.rs", "fn leaf() {}\n"),
            ("src/m/intent.md", "# x/src/m\n\n**Is.** The module.\n"),
        ],
    );
    let session = session(Scripted::new(|_| Ok(UNSURE.into())), decide::none());
    run(&tree, &["src/m/leaf.rs", "src/top.rs"], &session).await;

    let asked = session.ask.asked();
    let leaf = asked
        .iter()
        .find(|a| a.prompt.contains("leaf.rs\n"))
        .unwrap();
    assert!(
        leaf.prompt.contains("its module's intent"),
        "{}",
        leaf.prompt
    );
    assert!(leaf.prompt.contains("    3 | **Is.** The module.\n"));
    assert!(!leaf.prompt.contains("The crate."));
    assert!(leaf.prompt.contains("the outline of its module root"));
    let top = asked
        .iter()
        .find(|a| a.prompt.contains("top.rs\n"))
        .unwrap();
    assert!(
        top.prompt.contains("    3 | **Is.** The crate.\n"),
        "{}",
        top.prompt
    );
}

#[tokio::test]
async fn files_in_no_known_language_or_outside_a_rules_paths_are_never_shown() {
    let tree = Tree::new(
        "plan",
        &[(
            "scoped",
            r#"{"unit": "file", "levels": {"decide": 1, "explain": 1}, "paths": ["**/inside/**"],
                "question": "q", "fine": {"f": "f"}, "violations": {"v": "v"}}"#,
        )],
        &[
            ("inside/a.rs", "fn a() {}\n"),
            ("outside/b.rs", "fn b() {}\n"),
            ("inside/notes.md", "# notes\n"),
        ],
    );
    let session = session(
        Scripted::new(|_| Ok(r#"{"outcome": "f"}"#.into())),
        decide::none(),
    );
    let report = run(
        &tree,
        &["inside/a.rs", "outside/b.rs", "inside/notes.md"],
        &session,
    )
    .await;

    let asked = session.ask.asked();
    assert_eq!(asked.len(), 1);
    let inside = format!("File: {}\n", tree.path("inside/a.rs").display());
    assert!(asked[0].prompt.starts_with(&inside), "{}", asked[0].prompt);
    assert_eq!(decided(&report, "scoped", "f"), 1);
}

/// A synthesis that diagnoses nothing and writes `# Review`.
const REVIEW: &str = r##"{"diagnoses": [], "roots": [], "review": "# Review\n"}"##;

#[tokio::test]
async fn a_report_without_findings_is_not_synthesized() {
    let lead = Session::new(Scripted::new(|_| Ok(REVIEW.into())), decide::none());
    let synthesis = lead.synthesize(&Report::default()).await.unwrap();
    assert!(synthesis.is_none());
    assert!(lead.ask.asked().is_empty());
}

#[tokio::test]
async fn the_synthesis_is_one_frontier_call_over_every_finding_with_its_rule() {
    let tree = Tree::new(
        "lead",
        &[("everything", EVERYTHING)],
        &[("a.rs", "fn a() {}\n"), ("b.rs", "fn b() {}\n")],
    );
    let session = session(Scripted::new(wrong_on_line_one), decide::none());
    let report = run(&tree, &["a.rs", "b.rs"], &session).await;

    let lead = Session::new(Scripted::new(|_| Ok(REVIEW.into())), decide::none());
    let synthesis = lead.synthesize(&report).await.unwrap().unwrap();
    assert_eq!(synthesis.review, "# Review\n");
    assert!(synthesis.diagnoses.is_empty() && synthesis.roots.is_empty());
    assert_eq!(synthesis.usage, ONE_CALL);

    let asked = lead.ask.asked();
    assert_eq!(asked.len(), 1);
    assert_eq!(
        (asked[0].kind, asked[0].level),
        (Kind::Synthesize, ModelLevel::Frontier)
    );
    let brief = &asked[0].prompt;
    assert!(
        brief.contains("`everything`: Is anything wrong here?"),
        "{brief}"
    );
    assert!(brief.contains("`wrong`: Something is wrong."), "{brief}");
    assert!(!brief.contains("Nothing is wrong."), "{brief}");
    assert!(brief.contains("a.rs") && brief.contains("b.rs"), "{brief}");
    assert!(
        brief.contains("line 1 [everything/wrong]: needs work"),
        "{brief}"
    );
}

#[tokio::test]
async fn a_crate_wide_rule_makes_one_call_per_crate_and_findings_keep_their_file() {
    let tree = Tree::new(
        "crate",
        &[(
            "shapes",
            r#"{"unit": "function_signatures", "group": "crate", "levels": {"decide": 3, "explain": 3},
                "question": "q", "fine": {"f": "f"}, "violations": {"v": "v"}}"#,
        )],
        &[
            ("k/Cargo.toml", "[package]\n"),
            ("k/src/a.rs", "fn a(x: u8) {}\n"),
            ("k/src/b.rs", "fn b() {}\n"),
        ],
    );
    let b = tree.path("k/src/b.rs");
    let session = session(
        Scripted::new(|asked| {
            Ok(match asked.kind {
                Kind::Decide => r#"{"outcome": "v"}"#.into(),
                _ => {
                    let path = asked.prompt.lines().find_map(|l| {
                        l.strip_prefix("== ")
                            .and_then(|l| l.strip_suffix(" =="))
                            .filter(|p| p.ends_with("b.rs"))
                    });
                    format!(
                        r#"{{"findings": [{{"outcome": "v", "path": "{}", "line": 1, "message": "m"}}]}}"#,
                        path.unwrap()
                    )
                }
            })
        }),
        decide::none(),
    );
    let report = run(&tree, &["k/src/a.rs", "k/src/b.rs"], &session).await;
    assert_eq!(report.findings().len(), 1);
    assert_eq!(report.findings()[0].path, b);

    let asked = session.ask.asked();
    let decisions: Vec<_> = asked.iter().filter(|a| a.kind == Kind::Decide).collect();
    assert_eq!(decisions.len(), 1, "one call for the crate");
    let shown = &decisions[0].prompt;
    assert!(
        shown.starts_with(&format!("File: {}\n", tree.path("k").display())),
        "{shown}"
    );
    assert!(shown.contains("a.rs ==\n    1 | fn a(x: u8)\n"), "{shown}");
    assert!(shown.contains("b.rs ==\n    1 | fn b()\n"), "{shown}");
}

#[tokio::test]
async fn model_usage_is_totalled_per_rule_including_replies_that_could_not_be_read() {
    let tree = Tree::new(
        "usage",
        &[("everything", EVERYTHING)],
        &[("good.rs", "fn good() {}\n"), ("bad.rs", "fn bad() {}\n")],
    );
    let session = session(
        Scripted::new(|asked| {
            if asked.prompt.contains("bad.rs") {
                return Ok("not json".into());
            }
            Ok(CLEAN.into())
        }),
        Deciding::new(|_| None),
    );
    let report = run(&tree, &["good.rs", "bad.rs"], &session).await;

    assert_eq!(report.failures.len(), 1);
    assert_eq!(
        report.usage["everything"],
        Usage {
            calls: 2,
            input: 200,
            output: 20
        }
    );
    assert_eq!(
        report.system_one,
        Usage {
            calls: 2,
            input: 100,
            output: 0
        }
    );
}

#[tokio::test]
async fn a_dry_run_decides_every_unit_by_its_first_outcome_and_prices_it_by_its_prompt() {
    let tree = Tree::new(
        "dry",
        &[("everything", EVERYTHING)],
        &[("a.rs", "fn a() {}\n"), ("b.rs", "fn b() {}\n")],
    );
    let model = session(agent::dry_run(), decide::none());
    let report = run(&tree, &["a.rs", "b.rs"], &model).await;
    assert!(report.findings().is_empty() && report.failures.is_empty());
    assert_eq!(decided(&report, "everything", "clean"), 2);
    let usage = report.usage["everything"];
    assert_eq!((usage.calls, usage.output), (2, 0));
    // Each call carries the decider's instructions, the rule and a file.
    assert!(usage.input > 2 * 20, "{usage:?}");

    let system_one = session(agent::dry_run(), decide::dry_run());
    let report = run(&tree, &["a.rs", "b.rs"], &system_one).await;
    assert_eq!(decided(&report, "everything", "clean"), 2);
    assert_eq!(
        model_usage(&report, "everything"),
        Usage::default(),
        "System One settled every unit"
    );
    assert_eq!((report.system_one.calls, report.system_one.output), (2, 0));
    assert!(report.system_one.input > 0);
}

/// Two rules that each find two places wrong in every file: symptoms that
/// converge on a module from two angles.
const CONVERGING: fn(&Asked) -> Result<String> = |asked| {
    Ok(match asked.kind {
        Kind::Decide => WRONG.into(),
        _ => r#"{"findings": [{"outcome": "wrong", "line": 1, "message": "one"}, {"outcome": "wrong", "line": 2, "message": "two"}]}"#.into(),
    })
};

/// A module `m` whose findings converge.
fn converging(name: &str) -> (Tree, Arc<Session<Scripted, impl Decide>>) {
    let tree = Tree::new(
        name,
        &[("first", EVERYTHING), ("second", EVERYTHING)],
        &[
            ("m/mod.rs", "mod a;\npub fn distinctive_root() {}\n"),
            ("m/a.rs", "pub fn distinctive_leaf() {\n    body();\n}\n"),
        ],
    );
    (tree, session(Scripted::new(CONVERGING), decide::none()))
}

/// Names an assembler from symptoms 1-3 and describes it.
fn naming(asked: &Asked) -> Result<String> {
    Ok(match asked.kind {
        Kind::Identify => r#"{"things": [{"thing": "an assembler", "context": "in a JIT", "symptoms": [1, 2, 3]}]}"#.into(),
        Kind::Describe => r#"{"description": "An assembler turns labelled items into bytes."}"#.into(),
        other => bail!("not a naming call: {other:?}"),
    })
}

#[tokio::test]
async fn converging_symptoms_are_named_by_small_models_and_described_without_the_code() {
    let (tree, session) = converging("named");
    let mut report = run(&tree, &["m/mod.rs", "m/a.rs"], &session).await;
    assert_eq!(report.findings().len(), 8);

    let namer = Session::new(Scripted::new(naming), decide::none());
    let usage = namer.identify(&mut report).await.unwrap();
    assert!(report.failures.is_empty());
    assert_eq!(report.things().len(), 1);
    let t = &report.things()[0];
    assert_eq!(t.component, tree.path("m"));
    assert_eq!(
        (t.thing.as_str(), t.context.as_str()),
        ("an assembler", "in a JIT")
    );
    assert_eq!(
        t.description,
        "An assembler turns labelled items into bytes."
    );
    // The module's first three findings are the report's.
    assert_eq!(written(&t.explains), serde_json::json!([0, 1, 2]));
    assert_eq!(usage.calls, 2);

    let asked = namer.ask.asked();
    let steps: Vec<_> = asked.iter().map(|a| (a.kind, a.level)).collect();
    assert_eq!(
        steps,
        [
            (Kind::Identify, ModelLevel::Fast),
            (Kind::Describe, ModelLevel::Fast),
        ]
    );
    for code in ["distinctive_root", "distinctive_leaf"] {
        assert!(asked[0].prompt.contains(code), "the namer sees the outline");
        assert!(
            !asked[1].prompt.contains(code),
            "the description never sees the code"
        );
    }
    assert!(!asked[0].prompt.contains("body()"), "bodies are elided");
    let first = format!(
        "S1. {}:1 [first/wrong] one\n",
        tree.path("m/a.rs").display()
    );
    assert!(asked[0].prompt.contains(&first), "{}", asked[0].prompt);
    assert_eq!(asked[1].prompt, "an assembler, in a JIT");
}

#[tokio::test]
async fn symptoms_from_one_rule_alone_are_not_named() {
    let tree = Tree::new(
        "one-rule",
        &[("first", EVERYTHING)],
        &[("m/mod.rs", "mod a;\n"), ("m/a.rs", "fn a() {}\n")],
    );
    let session = session(Scripted::new(CONVERGING), decide::none());
    let mut report = run(&tree, &["m/mod.rs", "m/a.rs"], &session).await;
    assert_eq!(report.findings().len(), 4);

    let namer = Session::new(Scripted::new(naming), decide::none());
    namer.identify(&mut report).await.unwrap();
    assert!(report.things().is_empty());
    assert!(namer.ask.asked().is_empty());
}

#[tokio::test]
async fn a_thing_only_one_symptom_concerns_is_not_described() {
    let (tree, session) = converging("one-symptom");
    let mut report = run(&tree, &["m/mod.rs", "m/a.rs"], &session).await;

    let namer = Session::new(
        Scripted::new(|asked| {
            Ok(match asked.kind {
                Kind::Identify => {
                    r#"{"things": [{"thing": "a parser", "context": "c", "symptoms": [2]}]}"#.into()
                }
                other => bail!("asked {other:?}"),
            })
        }),
        decide::none(),
    );
    namer.identify(&mut report).await.unwrap();
    assert!(report.things().is_empty() && report.failures.is_empty());
    assert_eq!(namer.ask.asked().len(), 1);
}

#[tokio::test]
async fn a_description_outside_its_schema_is_a_failure_not_an_abort() {
    let (tree, session) = converging("bad-description");
    let mut report = run(&tree, &["m/mod.rs", "m/a.rs"], &session).await;

    let namer = Session::new(
        Scripted::new(|asked| {
            Ok(match asked.kind {
                Kind::Describe => "It is an assembler, probably.".into(),
                _ => naming(asked)?,
            })
        }),
        decide::none(),
    );
    let usage = namer.identify(&mut report).await.unwrap();
    assert!(report.things().is_empty());
    assert_eq!(report.failures.len(), 1);
    assert!(format!("{:#}", report.failures[0]).contains("an assembler"));
    // The naming and the description it could not read were both paid for.
    assert_eq!(usage.calls, 2);
    assert_eq!(
        (usage.input, usage.output),
        (2 * ONE_CALL.input, 2 * ONE_CALL.output)
    );
}

#[tokio::test]
async fn a_naming_outside_its_schema_still_counts_its_tokens() {
    let (tree, session) = converging("bad-naming");
    let mut report = run(&tree, &["m/mod.rs", "m/a.rs"], &session).await;

    let namer = Session::new(
        Scripted::new(|_| Ok("an assembler, I think".into())),
        decide::none(),
    );
    let usage = namer.identify(&mut report).await.unwrap();
    assert!(report.things().is_empty());
    assert_eq!(report.failures.len(), 1);
    assert_eq!(usage, ONE_CALL);
}

/// Two things in `m`, named by small models, ready for the synthesis.
async fn identified(name: &str) -> (Tree, Report) {
    let (tree, session) = converging(name);
    let mut report = run(&tree, &["m/mod.rs", "m/a.rs"], &session).await;
    let namer = Session::new(
        Scripted::new(|asked| {
            Ok(match asked.kind {
                Kind::Identify => r#"{"things": [
                    {"thing": "an assembler", "context": "c", "symptoms": [1, 2]},
                    {"thing": "a register allocator", "context": "c", "symptoms": [3, 4]}
                ]}"#
                .into(),
                Kind::Describe => r#"{"description": "described"}"#.into(),
                other => bail!("asked {other:?}"),
            })
        }),
        decide::none(),
    );
    namer.identify(&mut report).await.unwrap();
    (tree, report)
}

#[tokio::test]
async fn the_synthesis_sees_the_things_and_their_outlines_and_its_numbers_become_indexes() {
    let (_tree, report) = identified("synthesized").await;
    assert_eq!(report.things().len(), 2);

    let lead = Session::new(
        Scripted::new(|_| {
            Ok(r##"{
                "diagnoses": [
                    {"thing": 1, "diagnosis": "a label is a key", "falls_out": "names go", "explains": [1, 2]},
                    {"thing": 2, "diagnosis": "temps are predicted", "falls_out": "f", "explains": [3]}
                ],
                "roots": [{"root": "selection is a phase", "falls_out": "predictions go", "explains": [1, 2]}],
                "review": "# Review\n"
            }"##
            .into())
        }),
        decide::none(),
    );
    let synthesis = lead.synthesize(&report).await.unwrap().unwrap();
    let d = &synthesis.diagnoses;
    assert_eq!(
        written(&(d[0].thing, d[1].thing)),
        serde_json::json!([0, 1])
    );
    assert_eq!(d[0].diagnosis, "a label is a key");
    assert_eq!(written(&d[0].explains), serde_json::json!([0, 1]));
    assert_eq!(written(&d[1].explains), serde_json::json!([2]));
    assert_eq!(
        written(&synthesis.roots[0].explains),
        serde_json::json!([0, 1])
    );

    let brief = &lead.ask.asked()[0].prompt;
    assert!(brief.contains("### T1 — an assembler, c — "), "{brief}");
    assert!(brief.contains("Symptoms: F1, F2"), "{brief}");
    assert!(brief.contains("distinctive_root"), "the outline: {brief}");
    assert!(!brief.contains("body()"), "bodies are elided: {brief}");
    assert!(brief.contains("- F1 line 1 [first/wrong]: one"), "{brief}");
}

#[tokio::test]
async fn a_diagnosis_of_a_thing_never_named_is_an_error() {
    let (_tree, report) = identified("unnamed").await;
    let lead = Session::new(
        Scripted::new(|_| {
            Ok(r#"{"diagnoses": [{"thing": 3, "diagnosis": "d", "falls_out": "f", "explains": []}], "roots": [], "review": "r"}"#.into())
        }),
        decide::none(),
    );
    let error = lead.synthesize(&report).await.unwrap_err();
    assert!(format!("{error:#}").contains("T3"), "{error:#}");
}

/// [`Scripted`], except that calls of `kind` are answered only once two are
/// in flight together, and fail if one waits alone.
struct Together {
    inner: Scripted,
    kind: Kind,
    pair: tokio::sync::Barrier,
}

/// Longer than any pair of concurrent calls takes to meet.
const ALONE: std::time::Duration = std::time::Duration::from_secs(5);

impl Together {
    fn new(kind: Kind, reply: fn(&Asked) -> Result<String>) -> Self {
        Self {
            inner: Scripted::new(reply),
            kind,
            pair: tokio::sync::Barrier::new(2),
        }
    }
}

impl Ask for Together {
    async fn ask(&self, question: &Question<'_>) -> Result<Answer> {
        if kind(question.schema) == self.kind
            && tokio::time::timeout(ALONE, self.pair.wait()).await.is_err()
        {
            bail!("asked alone");
        }
        self.inner.ask(question).await
    }
}

#[tokio::test]
async fn every_rule_about_a_unit_is_asked_at_once() {
    let tree = Tree::new(
        "together",
        &[("first", EVERYTHING), ("second", EVERYTHING)],
        &[("a.rs", "fn a() {}\n")],
    );
    let session = session(
        Together::new(Kind::Decide, |_| Ok(CLEAN.into())),
        decide::none(),
    );
    let report = run(&tree, &["a.rs"], &session).await;
    assert!(report.failures.is_empty(), "{:?}", report.failures);
    assert_eq!(
        decided(&report, "first", "clean") + decided(&report, "second", "clean"),
        2
    );
}

#[tokio::test]
async fn every_converging_module_is_named_at_once() {
    let tree = Tree::new(
        "modules",
        &[("first", EVERYTHING), ("second", EVERYTHING)],
        &[
            ("m/mod.rs", "mod a;\n"),
            ("m/a.rs", "fn a() {}\n"),
            ("n/mod.rs", "mod b;\n"),
            ("n/b.rs", "fn b() {}\n"),
        ],
    );
    let session = session(Scripted::new(CONVERGING), decide::none());
    let mut report = run(
        &tree,
        &["m/mod.rs", "m/a.rs", "n/mod.rs", "n/b.rs"],
        &session,
    )
    .await;

    let namer = Session::new(Together::new(Kind::Identify, naming), decide::none());
    namer.identify(&mut report).await.unwrap();
    assert!(report.failures.is_empty(), "{:?}", report.failures);
    let components: Vec<_> = report
        .things()
        .iter()
        .map(|t| t.component.clone())
        .collect();
    assert_eq!(
        components,
        [tree.path("m"), tree.path("n")],
        "in module order"
    );
}

#[tokio::test]
async fn the_namer_sees_the_root_outline_first_then_the_files_with_the_most_symptoms() {
    let tree = Tree::new(
        "order",
        &[("first", EVERYTHING), ("second", EVERYTHING)],
        &[
            ("m/mod.rs", "mod a;\nmod b;\npub fn distinctive_root() {}\n"),
            ("m/a.rs", "fn distinctive_a() {}\n"),
            ("m/b.rs", "fn distinctive_b() {}\n"),
            ("m/c.rs", "fn distinctive_c() {}\n"),
        ],
    );
    // One, three and two findings: most-first (b, c, a) differs from path order
    // (a, b, c), from its reverse (c, b, a) and from fewest-first (a, c, b).
    let session = session(
        Scripted::new(|asked| {
            Ok(match asked.kind {
                Kind::Decide => WRONG.into(),
                _ if asked.prompt.contains("b.rs") => findings_on_lines(3),
                _ if asked.prompt.contains("c.rs") => findings_on_lines(2),
                _ => findings_on_lines(1),
            })
        }),
        decide::none(),
    );
    let mut report = run(&tree, &["m/a.rs", "m/b.rs", "m/c.rs"], &session).await;

    let namer = Session::new(Scripted::new(naming), decide::none());
    namer.identify(&mut report).await.unwrap();
    let prompt = &namer.ask.asked()[0].prompt;
    let at = |code: &str| {
        prompt
            .find(code)
            .unwrap_or_else(|| panic!("{code}: {prompt}"))
    };
    assert!(at("distinctive_root") < at("distinctive_b"), "{prompt}");
    assert!(at("distinctive_b") < at("distinctive_c"), "{prompt}");
    assert!(at("distinctive_c") < at("distinctive_a"), "{prompt}");
}

/// `count` findings, on lines 1 to `count`, each under the rule's violation.
fn findings_on_lines(count: usize) -> String {
    let each: Vec<String> = (1..=count)
        .map(|line| format!(r#"{{"outcome": "wrong", "line": {line}, "message": "m{line}"}}"#))
        .collect();
    format!(r#"{{"findings": [{}]}}"#, each.join(", "))
}

#[tokio::test]
async fn a_file_in_a_directory_without_a_root_clusters_with_the_enclosing_module() {
    let tree = Tree::new(
        "enclosing",
        &[("first", EVERYTHING), ("second", EVERYTHING)],
        &[
            ("m/mod.rs", "mod sub;\n"),
            ("m/sub/deep.rs", "fn deep() {}\n"),
        ],
    );
    let session = session(Scripted::new(CONVERGING), decide::none());
    let mut report = run(&tree, &["m/mod.rs", "m/sub/deep.rs"], &session).await;

    let namer = Session::new(Scripted::new(naming), decide::none());
    namer.identify(&mut report).await.unwrap();
    let identifying: Vec<_> = namer
        .ask
        .asked()
        .into_iter()
        .filter(|a| a.kind == Kind::Identify)
        .collect();
    assert_eq!(identifying.len(), 1, "one module, not two");
    assert!(
        identifying[0]
            .prompt
            .starts_with(&format!("Module: {}\n", tree.path("m").display()))
    );
    assert!(identifying[0].prompt.contains("deep.rs"));
    assert_eq!(report.things()[0].component, tree.path("m"));
}

#[tokio::test]
async fn a_directory_without_a_root_file_shows_its_files_no_module_root() {
    let tree = Tree::new(
        "no-root",
        &[(
            "contract",
            r#"{"unit": "file", "levels": {"decide": 1, "explain": 1}, "context": "module_root",
                "question": "q", "fine": {"f": "f"}, "violations": {"v": "v"}}"#,
        )],
        &[
            ("m/mod.rs", "mod sub;\n"),
            ("m/sub/deep.rs", "fn deep() {}\n"),
        ],
    );
    let session = session(Scripted::new(|_| Ok(UNSURE.into())), decide::none());
    run(&tree, &["m/sub/deep.rs"], &session).await;

    let asked = session.ask.asked();
    assert_eq!(asked.len(), 1);
    assert!(
        !asked[0].prompt.contains("For context only"),
        "{}",
        asked[0].prompt
    );
}

#[test]
fn a_minimum_confidence_outside_zero_to_one_is_refused() {
    let at = |confidence: f64| {
        Session::new(Scripted::new(|_| Ok(CLEAN.into())), decide::none()).settling_at(confidence)
    };
    assert!(at(0.0).is_ok() && at(1.0).is_ok());
    for outside in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
        assert!(at(outside).is_err(), "{outside}");
    }
}

/// Model calls a review makes when System One answers `clean` at 0.75 and the
/// session settles questions at `minimum` or above.
async fn model_calls_when_settling_at(minimum: f64) -> usize {
    let tree = Tree::new(
        "threshold",
        &[("everything", EVERYTHING)],
        &[("a.rs", "fn a() {}\n")],
    );
    let session = Arc::new(
        Session::new(
            Scripted::new(|_| Ok(CLEAN.into())),
            Deciding::new(|_| Some(("clean", 0.75))),
        )
        .settling_at(minimum)
        .unwrap(),
    );
    let report = run(&tree, &["a.rs"], &session).await;
    assert_eq!(decided(&report, "everything", "clean"), 1);
    session.ask.asked().len()
}

#[tokio::test]
async fn a_session_settles_at_or_above_its_minimum_confidence_and_asks_the_model_below() {
    assert_eq!(model_calls_when_settling_at(0.0).await, 0);
    assert_eq!(model_calls_when_settling_at(0.7).await, 0);
    assert_eq!(model_calls_when_settling_at(0.75).await, 0, "at it settles");
    assert_eq!(model_calls_when_settling_at(0.9).await, 1);
    assert_eq!(model_calls_when_settling_at(1.0).await, 1);
}

#[tokio::test]
async fn every_unit_is_reviewed_at_once() {
    let tree = Tree::new(
        "units-together",
        &[("everything", EVERYTHING)],
        &[("a.rs", "fn a() {}\n"), ("b.rs", "fn b() {}\n")],
    );
    let session = session(
        Together::new(Kind::Decide, |_| Ok(CLEAN.into())),
        decide::none(),
    );
    let report = run(&tree, &["a.rs", "b.rs"], &session).await;
    assert!(report.failures.is_empty(), "{:?}", report.failures);
    assert_eq!(decided(&report, "everything", "clean"), 2);
}

#[tokio::test]
async fn a_plan_is_reviewed_under_the_rules_it_was_planned_with() {
    let tree = Tree::new(
        "planned",
        &[
            ("everything", EVERYTHING),
            (
                "nowhere",
                r#"{"unit": "file", "levels": {"decide": 1, "explain": 1}, "paths": ["**/nowhere/**"],
                    "question": "q", "fine": {"f": "f"}, "violations": {"v": "v"}}"#,
            ),
        ],
        &[("a.rs", "fn a() {}\n")],
    );
    let calls = plan(tree.rules(), &[tree.path("a.rs")]).unwrap();

    // The session that reviews it holds no rules.
    let session = session(Scripted::new(|_| Ok(CLEAN.into())), decide::none());
    let report = session.clone().review(calls).await.unwrap();
    assert_eq!(decided(&report, "everything", "clean"), 1);
    assert_eq!(session.ask.asked().len(), 1);
    let asked: Vec<_> = report.rules().map(|rule| rule.id.as_str()).collect();
    assert_eq!(asked, ["everything"], "a rule that asked nothing is absent");
}

/// The synthesis, answered with `reply`, of two things named in `m`: eight
/// findings, two things, and the reply's own diagnoses, so no two lists share
/// a length.
async fn synthesis_error(name: &str, reply: fn(&Asked) -> Result<String>) -> String {
    let (_tree, report) = identified(name).await;
    let lead = Session::new(Scripted::new(reply), decide::none());
    format!("{:#}", lead.synthesize(&report).await.unwrap_err())
}

#[tokio::test]
async fn a_finding_number_that_names_nothing_is_an_error_not_a_shorter_list() {
    let error = synthesis_error("past-findings", |_| {
        Ok(r#"{"diagnoses": [{"thing": 1, "diagnosis": "d", "falls_out": "f", "explains": [1, 99]}], "roots": [], "review": "r"}"#.into())
    })
    .await;
    assert!(error.contains("F99"), "{error}");
}

#[tokio::test]
async fn a_zero_names_nothing_in_any_list() {
    let error = synthesis_error("zero-finding", |_| {
        Ok(r#"{"diagnoses": [{"thing": 1, "diagnosis": "d", "falls_out": "f", "explains": [0]}], "roots": [], "review": "r"}"#.into())
    })
    .await;
    assert!(error.contains("F0"), "{error}");
    let error = synthesis_error("zero-thing", |_| {
        Ok(r#"{"diagnoses": [{"thing": 0, "diagnosis": "d", "falls_out": "f", "explains": []}], "roots": [], "review": "r"}"#.into())
    })
    .await;
    assert!(error.contains("T0"), "{error}");
    let error = synthesis_error("zero-diagnosis", |_| {
        Ok(r#"{"diagnoses": [{"thing": 1, "diagnosis": "d", "falls_out": "f", "explains": []}], "roots": [{"root": "r", "falls_out": "f", "explains": [0]}], "review": "r"}"#.into())
    })
    .await;
    assert!(error.contains("D0"), "{error}");
}

#[tokio::test]
async fn a_root_numbers_the_replys_own_diagnoses_and_one_past_them_is_an_error() {
    let error = synthesis_error("past-diagnoses", |_| {
        Ok(r#"{"diagnoses": [{"thing": 1, "diagnosis": "d", "falls_out": "f", "explains": []}], "roots": [{"root": "r", "falls_out": "f", "explains": [1, 2]}], "review": "r"}"#.into())
    })
    .await;
    assert!(error.contains("D2"), "{error}");
}

#[tokio::test]
async fn a_symptom_number_that_names_nothing_fails_its_module_whole() {
    let (tree, session) = converging("past-symptoms");
    let mut report = run(&tree, &["m/mod.rs", "m/a.rs"], &session).await;

    let namer = Session::new(
        Scripted::new(|asked| {
            Ok(match asked.kind {
                Kind::Identify => r#"{"things": [
                    {"thing": "an assembler", "context": "c", "symptoms": [1, 2]},
                    {"thing": "a parser", "context": "c", "symptoms": [3, 9]}
                ]}"#
                .into(),
                other => bail!("asked {other:?}"),
            })
        }),
        decide::none(),
    );
    let usage = namer.identify(&mut report).await.unwrap();
    assert!(report.things().is_empty());
    assert_eq!(report.failures.len(), 1);
    let failure = format!("{:#}", report.failures[0]);
    assert!(
        failure.contains("S9") && failure.contains("naming"),
        "{failure}"
    );
    assert_eq!(
        usage.calls, 1,
        "the reply was paid for and nothing was described"
    );
}

#[tokio::test]
async fn a_zero_symptom_number_names_nothing() {
    let (tree, session) = converging("zero-symptom");
    let mut report = run(&tree, &["m/mod.rs", "m/a.rs"], &session).await;

    let namer = Session::new(
        Scripted::new(|asked| {
            Ok(match asked.kind {
                Kind::Identify => {
                    r#"{"things": [{"thing": "a parser", "context": "c", "symptoms": [0, 1, 2]}]}"#
                        .into()
                }
                other => bail!("asked {other:?}"),
            })
        }),
        decide::none(),
    );
    namer.identify(&mut report).await.unwrap();
    assert!(report.things().is_empty());
    let failure = format!("{:#}", report.failures[0]);
    assert!(failure.contains("S0"), "{failure}");
}

#[tokio::test]
async fn a_things_symptoms_come_back_in_report_order_each_once() {
    let (tree, session) = converging("report-order");
    let mut report = run(&tree, &["m/mod.rs", "m/a.rs"], &session).await;

    let namer = Session::new(
        Scripted::new(|asked| {
            Ok(match asked.kind {
                Kind::Identify => {
                    r#"{"things": [{"thing": "a parser", "context": "c", "symptoms": [3, 1, 3, 2]}]}"#
                        .into()
                }
                Kind::Describe => r#"{"description": "described"}"#.into(),
                other => bail!("asked {other:?}"),
            })
        }),
        decide::none(),
    );
    namer.identify(&mut report).await.unwrap();
    assert!(report.failures.is_empty(), "{:?}", report.failures);
    assert_eq!(
        written(&report.things()[0].explains),
        serde_json::json!([0, 1, 2])
    );
}

#[tokio::test]
async fn the_same_symptom_named_twice_counts_once() {
    let (tree, session) = converging("twice");
    let mut report = run(&tree, &["m/mod.rs", "m/a.rs"], &session).await;

    let namer = Session::new(
        Scripted::new(|asked| {
            Ok(match asked.kind {
                Kind::Identify => {
                    r#"{"things": [{"thing": "a parser", "context": "c", "symptoms": [2, 2]}]}"#
                        .into()
                }
                other => bail!("asked {other:?}"),
            })
        }),
        decide::none(),
    );
    namer.identify(&mut report).await.unwrap();
    assert!(report.things().is_empty() && report.failures.is_empty());
}
