//! `plan`, `review` and `synthesize` driven through their public API, with a
//! scripted `Ask` standing in for a model and a scripted `Decide` for
//! System One.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Result, bail};
use desloppify::agent::{self, Answer, Ask, Question, Usage};
use desloppify::decide::{self, Choice, Decide, Decided, Decisions};
use desloppify::model::ModelLevel;
use desloppify::review::{Report, Reviewers, plan, review, synthesize};
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

/// The confidence a review is run with.
const MIN_CONFIDENCE: f64 = 0.7;

/// What a model call was asked to do, read from the schema it was held to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Decide,
    Explain,
    Prose,
}

/// What a model call was asked.
#[derive(Clone, Debug)]
struct Asked {
    level: ModelLevel,
    kind: Kind,
    preamble: String,
    prompt: String,
}

fn kind(schema: Option<&serde_json::Value>) -> Kind {
    let Some(schema) = schema else {
        return Kind::Prose;
    };
    if schema["properties"].get("outcome").is_some() {
        return Kind::Decide;
    }
    assert!(schema["properties"].get("findings").is_some(), "{schema}");
    Kind::Explain
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
        Kind::Explain | Kind::Prose => {
            r#"{"findings": [{"outcome": "wrong", "line": 1, "message": "needs work"}]}"#.into()
        }
    })
}

fn reviewers<D: Decide>(ask: Scripted, decide: D) -> Arc<Reviewers<Scripted, D>> {
    Arc::new(Reviewers {
        ask,
        decide,
        min_confidence: MIN_CONFIDENCE,
    })
}

async fn run<A: Ask, D: Decide>(
    tree: &Tree,
    files: &[&str],
    reviewers: &Arc<Reviewers<A, D>>,
) -> Report {
    let rules = tree.rules();
    let files: Vec<_> = files.iter().map(|f| tree.path(f)).collect();
    let calls = plan(&rules, &files).unwrap();
    review(reviewers.clone(), Arc::new(rules), calls)
        .await
        .unwrap()
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
    let reviewers = reviewers(
        Scripted::new(|asked| {
            Ok(match asked.kind {
                Kind::Decide => WRONG.into(),
                _ => r#"{"findings": [{"outcome": "wrong", "line": 9, "message": "late"}, {"outcome": "wrong", "line": 1, "message": "early"}]}"#
                    .into(),
            })
        }),
        decide::none(),
    );
    let report = run(&tree, &["b.rs", "a.rs"], &reviewers).await;

    let seen: Vec<_> = report
        .findings
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
    let reviewers = reviewers(
        Scripted::new(wrong_on_line_one),
        Deciding::new(|_| Some(("clean", 0.9))),
    );
    let report = run(&tree, &["a.rs"], &reviewers).await;

    assert!(reviewers.ask.asked().is_empty());
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
    let reviewers = reviewers(
        Scripted::new(wrong_on_line_one),
        Deciding::new(|_| Some(("wrong", 0.95))),
    );
    let report = run(&tree, &["a.rs"], &reviewers).await;

    let asked = reviewers.ask.asked();
    assert_eq!(asked.len(), 1);
    assert_eq!(asked[0].kind, Kind::Explain);
    assert_eq!(asked[0].level, ModelLevel::Fast);
    assert!(asked[0].preamble.contains("Say what to change."));
    assert!(asked[0].preamble.contains("`wrong`: Something is wrong."));
    assert_eq!(report.findings.len(), 1);
    assert!(report.escalated.is_empty());
}

#[tokio::test]
async fn an_answer_below_the_minimum_confidence_goes_to_the_model_at_the_decide_level() {
    let tree = Tree::new(
        "doubtful",
        &[("everything", EVERYTHING)],
        &[("a.rs", "fn a() {}\n")],
    );
    let reviewers = reviewers(
        Scripted::new(|_| Ok(CLEAN.into())),
        Deciding::new(|_| Some(("wrong", 0.5))),
    );
    let report = run(&tree, &["a.rs"], &reviewers).await;

    let asked = reviewers.ask.asked();
    assert_eq!(asked.len(), 1);
    assert_eq!(
        (asked[0].kind, asked[0].level),
        (Kind::Decide, ModelLevel::Lite)
    );
    assert_eq!(report.escalated["everything"], 1);
    assert_eq!(decided(&report, "everything", "clean"), 1);
    assert!(report.findings.is_empty());
}

#[tokio::test]
async fn a_label_that_is_not_one_of_the_rules_outcomes_is_not_an_answer() {
    let tree = Tree::new(
        "bogus",
        &[("everything", EVERYTHING)],
        &[("a.rs", "fn a() {}\n")],
    );
    let reviewers = reviewers(
        Scripted::new(|_| Ok(CLEAN.into())),
        Deciding::new(|_| Some(("bogus", 1.0))),
    );
    let report = run(&tree, &["a.rs"], &reviewers).await;
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
    let reviewers = reviewers(Scripted::new(|_| Ok(UNSURE.into())), decide::none());
    let report = run(&tree, &["a.rs"], &reviewers).await;

    let levels: Vec<_> = reviewers.ask.asked().iter().map(|a| a.level).collect();
    assert_eq!(
        levels,
        [ModelLevel::Lite, ModelLevel::Fast, ModelLevel::Strong]
    );
    assert_eq!(decided(&report, "escalating", "unsure"), 1);
    assert!(report.findings.is_empty() && report.failures.is_empty());
    assert_eq!(report.usage["escalating"].calls, 3);
}

#[tokio::test]
async fn a_violation_decided_after_escalating_is_explained_at_the_explain_level() {
    let tree = Tree::new(
        "escalated",
        &[("escalating", ESCALATING)],
        &[("a.rs", "fn a() {}\n")],
    );
    let reviewers = reviewers(
        Scripted::new(|asked| {
            Ok(match (asked.kind, asked.level) {
                (Kind::Decide, ModelLevel::Lite) => UNSURE.into(),
                (Kind::Decide, _) => WRONG.into(),
                _ => r#"{"findings": [{"outcome": "wrong", "line": 1, "message": "m"}]}"#.into(),
            })
        }),
        decide::none(),
    );
    let report = run(&tree, &["a.rs"], &reviewers).await;

    let asked: Vec<_> = reviewers
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
    assert_eq!(report.findings.len(), 1);
}

#[tokio::test]
async fn rules_that_see_the_same_unit_share_one_system_one_request() {
    let tree = Tree::new(
        "shared",
        &[("first", EVERYTHING), ("second", EVERYTHING)],
        &[("a.rs", "fn a() {}\nfn b() {}\n")],
    );
    let reviewers = reviewers(
        Scripted::new(wrong_on_line_one),
        Deciding::new(|_| Some(("clean", 1.0))),
    );
    let report = run(&tree, &["a.rs"], &reviewers).await;

    let requested = reviewers.decide.requested.lock().unwrap().clone();
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
    let reviewers = reviewers(Scripted::new(|_| Ok(CLEAN.into())), Down);
    let report = run(&tree, &["a.rs"], &reviewers).await;

    assert_eq!(reviewers.ask.asked().len(), 2);
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
    let reviewers = reviewers(
        Scripted::new(|asked| {
            if asked.prompt.contains("bad.rs") {
                return Ok("Looks fine to me!".into());
            }
            Ok(CLEAN.into())
        }),
        decide::none(),
    );
    let report = run(&tree, &["good.rs", "bad.rs"], &reviewers).await;

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
    let reviewers = reviewers(
        Scripted::new(|asked| {
            Ok(match asked.kind {
                Kind::Decide => WRONG.into(),
                _ => r#"[{"line": 1, "message": "a bare array"}]"#.into(),
            })
        }),
        decide::none(),
    );
    let report = run(&tree, &["a.rs"], &reviewers).await;

    assert!(report.findings.is_empty());
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
    let reviewers = reviewers(
        Scripted::new(|asked| {
            Ok(match asked.kind {
                Kind::Decide => r#"{"outcome": "loud"}"#.into(),
                _ => r#"{"findings": [{"outcome": "loud", "line": 1, "message": "a"}, {"outcome": "quiet", "line": 2, "message": "b"}]}"#.into(),
            })
        }),
        decide::none(),
    );
    let report = run(&tree, &["a.rs"], &reviewers).await;

    let outcomes: Vec<_> = report.findings.iter().map(|f| f.outcome.as_str()).collect();
    assert_eq!(outcomes, ["loud", "quiet"]);
    assert_eq!(decided(&report, "rule", "loud"), 1);
    let explained = reviewers
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
    let reviewers = reviewers(
        Scripted::new(|asked| {
            Ok(match asked.kind {
                Kind::Decide => r#"{"outcome": "loud"}"#.into(),
                _ => r#"{"findings": [{"outcome": "clean", "line": 1, "message": "a"}]}"#.into(),
            })
        }),
        decide::none(),
    );
    let report = run(&tree, &["a.rs"], &reviewers).await;
    assert!(report.findings.is_empty());
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
    let reviewers = reviewers(Scripted::new(|_| bail!("401 unauthorized")), decide::none());
    let report = run(&tree, &["a.rs"], &reviewers).await;

    assert!(report.findings.is_empty());
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
    let reviewers = reviewers(Scripted::new(|_| Ok(CLEAN.into())), decide::none());
    run(&tree, &["a.rs"], &reviewers).await;

    let asked = reviewers.ask.asked();
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
    let reviewers = reviewers(
        Scripted::new(|_| Ok(r#"{"outcome": "bogus"}"#.into())),
        decide::none(),
    );
    let report = run(&tree, &["a.rs"], &reviewers).await;
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
    let reviewers = reviewers(Scripted::new(|_| Ok(UNSURE.into())), decide::none());
    run(&tree, &["m/leaf.rs", "m/mod.rs"], &reviewers).await;

    let asked = reviewers.ask.asked();
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
    let reviewers = reviewers(
        Scripted::new(|_| Ok(r#"{"outcome": "f"}"#.into())),
        decide::none(),
    );
    let report = run(
        &tree,
        &["inside/a.rs", "outside/b.rs", "inside/notes.md"],
        &reviewers,
    )
    .await;

    let asked = reviewers.ask.asked();
    assert_eq!(asked.len(), 1);
    let inside = format!("File: {}\n", tree.path("inside/a.rs").display());
    assert!(asked[0].prompt.starts_with(&inside), "{}", asked[0].prompt);
    assert_eq!(decided(&report, "scoped", "f"), 1);
}

#[tokio::test]
async fn a_report_without_findings_is_not_synthesized() {
    let lead = Scripted::new(|_| Ok("unused".into()));
    let review = synthesize(&lead, &[], &Report::default()).await.unwrap();
    assert!(review.is_none());
    assert!(lead.asked().is_empty());
}

#[tokio::test]
async fn the_lead_review_is_one_frontier_call_over_every_finding_with_its_rule() {
    let tree = Tree::new(
        "lead",
        &[("everything", EVERYTHING)],
        &[("a.rs", "fn a() {}\n"), ("b.rs", "fn b() {}\n")],
    );
    let reviewers = reviewers(Scripted::new(wrong_on_line_one), decide::none());
    let report = run(&tree, &["a.rs", "b.rs"], &reviewers).await;

    let lead = Scripted::new(|_| Ok("# Review\n".into()));
    let review = synthesize(&lead, &tree.rules(), &report)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(review.text, "# Review\n");
    assert_eq!(review.usage, ONE_CALL);

    let asked = lead.asked();
    assert_eq!(asked.len(), 1);
    assert_eq!(asked[0].level, ModelLevel::Frontier);
    assert_eq!(asked[0].kind, Kind::Prose, "the lead writes prose");
    let brief = &asked[0].prompt;
    assert!(
        brief.contains("`everything`: Is anything wrong here?"),
        "{brief}"
    );
    assert!(brief.contains("`wrong`: Something is wrong."), "{brief}");
    assert!(!brief.contains("Nothing is wrong."), "{brief}");
    assert!(brief.contains("a.rs") && brief.contains("b.rs"), "{brief}");
    assert!(
        brief.contains("- line 1 [everything/wrong]: needs work"),
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
    let reviewers = reviewers(
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
    let report = run(&tree, &["k/src/a.rs", "k/src/b.rs"], &reviewers).await;
    assert_eq!(report.findings.len(), 1);
    assert_eq!(report.findings[0].path, b);

    let asked = reviewers.ask.asked();
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
    let reviewers = reviewers(
        Scripted::new(|asked| {
            if asked.prompt.contains("bad.rs") {
                return Ok("not json".into());
            }
            Ok(CLEAN.into())
        }),
        Deciding::new(|_| None),
    );
    let report = run(&tree, &["good.rs", "bad.rs"], &reviewers).await;

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
    let model = Arc::new(Reviewers {
        ask: agent::dry_run(),
        decide: decide::none(),
        min_confidence: MIN_CONFIDENCE,
    });
    let report = run(&tree, &["a.rs", "b.rs"], &model).await;
    assert!(report.findings.is_empty() && report.failures.is_empty());
    assert_eq!(decided(&report, "everything", "clean"), 2);
    let usage = report.usage["everything"];
    assert_eq!((usage.calls, usage.output), (2, 0));
    // Each call carries the decider's instructions, the rule and a file.
    assert!(usage.input > 2 * 20, "{usage:?}");

    let system_one = Arc::new(Reviewers {
        ask: agent::dry_run(),
        decide: decide::dry_run(),
        min_confidence: MIN_CONFIDENCE,
    });
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
