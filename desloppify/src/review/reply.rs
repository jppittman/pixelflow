//! What each model call is told, and reading its reply.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use super::Call;
use crate::agent::{Answer, Ask, Question};
use crate::model::ModelLevel;
use crate::rule::{Outcome, Rule, UNSURE, Verdict};

/// One kind of question to a model: the level it is asked at, what it is
/// told, and the schema its reply is held to.
pub(super) struct Step<'a> {
    pub(super) level: ModelLevel,
    pub(super) system: &'a str,
    pub(super) schema: &'a serde_json::Value,
}

/// `prompt`, put to a model as `step` says. No call is made without a
/// schema.
pub(super) async fn ask<A: Ask>(agent: &A, step: &Step<'_>, prompt: &str) -> Result<Answer> {
    agent
        .ask(&Question {
            level: step.level,
            system: step.system,
            prompt,
            schema: Some(step.schema),
        })
        .await
}

const DECIDER: &str = "\
You answer one question about the code you are shown, by choosing exactly one \
of the outcomes below. Choose `unsure` only if the code shown is not enough \
to decide.";

const EXPLAINER: &str = "\
You explain a decision that the code you are shown violates a rule. Report \
every place in the code that violates it, as a finding: the violation that \
place shows — the one decided, or another of the rule's violations below — \
the line number printed in the code, what is wrong and why, and the fix. \
When the code comes from several files, each under a `== path ==` header, \
give each finding that header's path. If on reflection nothing in the code \
violates the rule, report no findings.";

/// The labels a decision chooses from: the rule's outcomes, each with what
/// it means.
pub(super) fn labels(rule: &Rule) -> Vec<(&str, &str)> {
    rule.decision
        .outcomes
        .iter()
        .map(|o| (o.name.as_str(), o.meaning.as_str()))
        .collect()
}

/// What a decision chose: one of the rule's outcomes, or that the code shown
/// is not enough. The closed set a decision's reply is held to, so the schema
/// and the reading of a reply cannot disagree about it.
#[derive(Clone, Copy)]
pub(super) enum Chosen<'r> {
    Outcome(&'r Outcome),
    Unsure,
}

impl<'r> Chosen<'r> {
    /// The name the reply, the schema and the tally know it by.
    pub(super) fn name(self) -> &'r str {
        match self {
            Self::Outcome(outcome) => &outcome.name,
            Self::Unsure => UNSURE,
        }
    }

    fn meaning(self) -> &'r str {
        match self {
            Self::Outcome(outcome) => &outcome.meaning,
            Self::Unsure => "The code shown is not enough to decide.",
        }
    }
}

/// What a decision may choose: the rule's outcomes, then `unsure`.
fn choices(rule: &Rule) -> impl Iterator<Item = Chosen<'_>> {
    rule.decision
        .outcomes
        .iter()
        .map(Chosen::Outcome)
        .chain([Chosen::Unsure])
}

/// The system prompt for deciding `rule` with a model.
pub(super) fn decide_preamble(rule: &Rule) -> String {
    let outcomes: String = choices(rule)
        .map(|c| format!("- `{}`: {}\n", c.name(), c.meaning()))
        .collect();
    format!(
        "{}{DECIDER}\n\nQuestion: {}\n\nOutcomes:\n{outcomes}",
        rule.skills, rule.decision.question
    )
}

/// The shape a decision's reply must have: one of what it may choose.
pub(super) fn decision_schema(rule: &Rule) -> serde_json::Value {
    let names: Vec<&str> = choices(rule).map(|c| c.name()).collect();
    serde_json::json!({
        "type": "object",
        "properties": { "outcome": { "type": "string", "enum": names } },
        "required": ["outcome"]
    })
}

#[derive(Deserialize)]
struct DecisionReply {
    outcome: String,
}

/// What a decision reply chose.
pub(super) fn parse_decision<'r>(rule: &'r Rule, reply: &str) -> Result<Chosen<'r>> {
    let reply: DecisionReply =
        serde_json::from_str(reply).with_context(|| format!("reply is not a decision: {reply}"))?;
    choices(rule)
        .find(|c| c.name() == reply.outcome)
        .with_context(|| format!("`{}` is not one of the rule's outcomes", reply.outcome))
}

/// The rule's violation outcomes, in file order.
fn violations(rule: &Rule) -> impl Iterator<Item = &Outcome> {
    rule.decision
        .outcomes
        .iter()
        .filter(|o| o.verdict == Verdict::Violation)
}

/// The system prompt for explaining a unit decided `outcome`: every
/// violation of the rule it shows, not only that one.
pub(super) fn explain_preamble(rule: &Rule, outcome: &Outcome) -> String {
    let guidance = rule
        .decision
        .guidance
        .as_deref()
        .map(|g| format!("\n\n{g}"))
        .unwrap_or_default();
    let violations: String = violations(rule)
        .map(|o| format!("- `{}`: {}\n", o.name, o.meaning))
        .collect();
    format!(
        "{}{EXPLAINER}{guidance}\n\nThe question was: {}\nThe code was decided `{}`.\n\nThe rule's violations:\n{violations}",
        rule.skills, rule.decision.question, outcome.name
    )
}

/// The shape an explanation's reply must have: findings, each under one of
/// `rule`'s violation outcomes.
pub(super) fn findings_schema(rule: &Rule) -> serde_json::Value {
    let names: Vec<&str> = violations(rule).map(|o| o.name.as_str()).collect();
    serde_json::json!({
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "outcome": { "type": "string", "enum": names },
                        "path": { "type": "string" },
                        "line": { "type": "integer" },
                        "message": { "type": "string" }
                    },
                    "required": ["outcome", "line", "message"]
                }
            }
        },
        "required": ["findings"]
    })
}

#[derive(Deserialize)]
struct Replies {
    findings: Vec<Reply>,
}

#[derive(Deserialize)]
pub(super) struct Reply {
    /// The violation outcome this place shows.
    pub(super) outcome: String,
    /// Set when the code shown came from several files.
    #[serde(default)]
    pub(super) path: Option<PathBuf>,
    pub(super) line: u64,
    pub(super) message: String,
}

/// The findings in an explanation's reply, each under one of `rule`'s
/// violation outcomes.
pub(super) fn parse_findings(rule: &Rule, reply: &str) -> Result<Vec<Reply>> {
    let replies: Replies = serde_json::from_str(reply)
        .with_context(|| format!("reply is not a findings object: {reply}"))?;
    for finding in &replies.findings {
        if !violations(rule).any(|o| o.name == finding.outcome) {
            bail!("`{}` is not one of the rule's violations", finding.outcome);
        }
    }
    Ok(replies.findings)
}

/// The user prompt for `call` — also System One's state: the file, its
/// snippet, and any context.
pub(super) fn prompt(call: &Call) -> String {
    let mut prompt = format!("File: {}\n\n{}", call.path.display(), call.snippet.numbered);
    if let Some(root) = &call.root {
        prompt.push_str(&format!(
            "\nFor context only, not under review — the outline of its module root, {}:\n\n{}",
            root.path.display(),
            root.text
        ));
    }
    if let Some(intent) = &call.intent {
        prompt.push_str(&format!(
            "\nFor context only, not under review — its module's intent, {}:\n\n{}",
            intent.path.display(),
            intent.text
        ));
    }
    prompt
}
