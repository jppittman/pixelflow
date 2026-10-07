//! A rule file is checked when it loads: every malformed rule is refused,
//! naming its file, before a review starts.

use std::path::PathBuf;

use desloppify::model::ModelLevel;
use desloppify::rule::{self, Rule, Verdict};
use desloppify::skills::Skills;

/// A scratch rules directory holding one rule, removed on drop.
struct Rules(PathBuf);

impl Rules {
    fn holding(name: &str, json: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("desloppify-rule-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(format!("{name}.json")), json).unwrap();
        Self(dir)
    }

    fn load(&self) -> anyhow::Result<Vec<Rule>> {
        rule::load_dir(&self.0, &Skills::default())
    }
}

impl Drop for Rules {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.0) {
            eprintln!("leaving {}: {error}", self.0.display());
        }
    }
}

/// `rule` with `fields` spliced in after its unit.
fn with(fields: &str) -> String {
    format!(
        r#"{{"unit": "functions", {fields} "question": "q", "fine": {{"b_fine": "f", "a_fine": "f"}}, "violations": {{"v": "v"}}}}"#
    )
}

fn refused(name: &str, json: &str, because: &str) {
    let rules = Rules::holding(name, json);
    let Err(error) = rules.load() else {
        panic!("{name} loaded");
    };
    let error = format!("{error:#}");
    assert!(error.contains(&format!("{name}.json")), "{error}");
    assert!(error.contains(because), "{error}");
}

#[test]
fn a_well_formed_rule_loads_with_its_fine_outcomes_first() {
    let rules = Rules::holding("good", &with(r#""levels": {"decide": 1, "explain": 3},"#));
    let loaded = rules.load().unwrap();
    assert_eq!(loaded.len(), 1);
    let rule = &loaded[0];
    assert_eq!(rule.id, "good");
    assert_eq!(
        (rule.levels.decide, rule.levels.explain),
        (ModelLevel::Lite, ModelLevel::Strong)
    );
    let outcomes: Vec<_> = rule
        .decision
        .outcomes
        .iter()
        .map(|o| (o.name.as_str(), o.verdict))
        .collect();
    assert_eq!(
        outcomes,
        [
            ("a_fine", Verdict::Fine),
            ("b_fine", Verdict::Fine),
            ("v", Verdict::Violation)
        ]
    );
}

#[test]
fn an_explain_level_below_the_decide_level_is_refused() {
    refused(
        "backwards",
        &with(r#""levels": {"decide": 3, "explain": 1},"#),
        "levels.explain",
    );
}

#[test]
fn a_level_outside_one_to_four_is_refused() {
    refused(
        "level",
        &with(r#""levels": {"decide": 5, "explain": 5},"#),
        "5",
    );
}

#[test]
fn a_decision_without_violations_is_refused() {
    refused(
        "fine-only",
        r#"{"unit": "file", "levels": {"decide": 1, "explain": 1}, "question": "q", "fine": {"f": "f"}, "violations": {}}"#,
        "violation",
    );
}

#[test]
fn an_outcome_named_unsure_is_refused() {
    refused(
        "unsure",
        r#"{"unit": "file", "levels": {"decide": 1, "explain": 1}, "question": "q", "fine": {"unsure": "f"}, "violations": {"v": "v"}}"#,
        "unsure",
    );
}

#[test]
fn an_outcome_named_twice_is_refused() {
    refused(
        "twice",
        r#"{"unit": "file", "levels": {"decide": 1, "explain": 1}, "question": "q", "fine": {"same": "f"}, "violations": {"same": "v"}}"#,
        "twice",
    );
}

#[test]
fn a_group_on_a_whole_file_unit_is_refused() {
    refused(
        "grouped",
        r#"{"unit": "outline", "group": "crate", "levels": {"decide": 1, "explain": 1}, "question": "q", "fine": {"f": "f"}, "violations": {"v": "v"}}"#,
        "group",
    );
}

#[test]
fn an_unknown_field_or_skill_is_refused() {
    refused(
        "field",
        &with(r#""levels": {"decide": 1, "explain": 1}, "prompt": "old format","#),
        "prompt",
    );
    refused(
        "skill",
        &with(r#""levels": {"decide": 1, "explain": 1}, "skills": ["nope"],"#),
        "nope",
    );
}
