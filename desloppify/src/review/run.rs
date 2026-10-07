//! Making the calls: System One, then the model, then the explanation.

use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::task::JoinSet;

use super::reply::{
    decide_preamble, decision_schema, explain_preamble, findings_schema, labels, parse_decision,
    parse_findings, prompt,
};
use super::{Call, Finding, Report, Reviewers};
use crate::agent::{Ask, Question, Usage};
use crate::decide::{Choice, Decide, Decisions};
use crate::model::ModelLevel;
use crate::rule::{Outcome, Rule, UNSURE, Verdict};

/// What one rule made of one unit.
struct Judged {
    rule: usize,
    /// Tokens its model calls used.
    usage: Usage,
    escalated: bool,
    /// The outcome's name, `unsure` if undecided, or the reason it failed.
    outcome: Result<String>,
    findings: Result<Vec<Finding>>,
}

/// What one call produced.
struct Reviewed {
    system_one: Usage,
    system_one_failure: Option<anyhow::Error>,
    judged: Vec<Judged>,
}

pub(super) async fn review<A: Ask, D: Decide>(
    reviewers: Arc<Reviewers<A, D>>,
    rules: Arc<Vec<Rule>>,
    plan: Vec<Call>,
) -> Result<Report> {
    let mut calls = JoinSet::new();
    for call in plan {
        let (reviewers, rules) = (reviewers.clone(), rules.clone());
        calls.spawn(async move { reviewed(&reviewers, &rules, &call).await });
    }

    let mut report = Report::default();
    while let Some(joined) = calls.join_next().await {
        let reviewed = joined?;
        report.system_one += reviewed.system_one;
        report.failures.extend(reviewed.system_one_failure);
        for judged in reviewed.judged {
            let id = &rules[judged.rule].id;
            *report.usage.entry(id.clone()).or_default() += judged.usage;
            if judged.escalated {
                *report.escalated.entry(id.clone()).or_default() += 1;
            }
            match judged.outcome {
                Ok(outcome) => {
                    let tally = report.decisions.entry(id.clone()).or_default();
                    *tally.entry(outcome).or_default() += 1;
                }
                Err(failure) => report.failures.push(failure),
            }
            match judged.findings {
                Ok(findings) => report.findings.extend(findings),
                Err(failure) => report.failures.push(failure),
            }
        }
    }
    report
        .findings
        .sort_by(|a, b| (&a.path, a.line, &a.rule).cmp(&(&b.path, b.line, &b.rule)));
    Ok(report)
}

async fn reviewed<A: Ask, D: Decide>(
    reviewers: &Reviewers<A, D>,
    rules: &[Rule],
    call: &Call,
) -> Reviewed {
    let state = prompt(call);
    let labels: Vec<Vec<(&str, &str)>> = call.rules.iter().map(|&r| labels(&rules[r])).collect();
    let questions: Vec<Choice<'_>> = call
        .rules
        .iter()
        .zip(&labels)
        .map(|(&r, labels)| Choice {
            name: &rules[r].id,
            instructions: &rules[r].decision.question,
            labels,
        })
        .collect();
    // If System One fails, every question goes to the model: a slower
    // review, not a smaller one.
    let (decisions, system_one_failure) = match reviewers.decide.decide(&state, &questions).await {
        Ok(decisions) => (decisions, None),
        Err(error) => {
            let error = error.context(format!("System One on {}", place(call)));
            (Decisions::default(), Some(error))
        }
    };

    let unit = Seen {
        call,
        rules,
        state,
        decisions,
    };
    let mut judged = Vec::new();
    for &index in &call.rules {
        judged.push(judge(reviewers, &unit, index).await);
    }
    Reviewed {
        system_one: unit.decisions.usage,
        system_one_failure,
        judged,
    }
}

/// One call's unit as each rule about it is judged, with what System One
/// made of it.
struct Seen<'c> {
    call: &'c Call,
    rules: &'c [Rule],
    /// What every model call about the unit is shown; System One's state.
    state: String,
    decisions: Decisions,
}

async fn judge<A: Ask, D: Decide>(
    reviewers: &Reviewers<A, D>,
    unit: &Seen<'_>,
    index: usize,
) -> Judged {
    let rule = &unit.rules[index];
    let context = || format!("rule {} on {}", rule.id, place(unit.call));
    let settled = unit
        .decisions
        .answers
        .get(&rule.id)
        .filter(|d| d.confidence >= reviewers.min_confidence)
        .and_then(|d| rule.decision.outcomes.iter().find(|o| o.name == d.label));
    let mut asking = Asking {
        model: &reviewers.ask,
        rule,
        unit,
        usage: Usage::default(),
    };
    let decided = match settled {
        Some(outcome) => Ok(Some(outcome)),
        None => asking.decide().await,
    };
    let (outcome, findings) = match decided {
        Err(error) => (Err(error.context(context())), Ok(Vec::new())),
        Ok(Some(outcome)) if outcome.verdict == Verdict::Violation => (
            Ok(outcome.name.clone()),
            asking.explain(outcome).await.with_context(context),
        ),
        Ok(outcome) => {
            let name = outcome.map_or(UNSURE, |o| o.name.as_str());
            (Ok(name.to_owned()), Ok(Vec::new()))
        }
    };
    Judged {
        rule: index,
        usage: asking.usage,
        escalated: settled.is_none(),
        outcome,
        findings,
    }
}

/// One rule's model calls about one unit, and the tokens they used.
struct Asking<'a, A> {
    model: &'a A,
    rule: &'a Rule,
    unit: &'a Seen<'a>,
    usage: Usage,
}

impl<'a, A: Ask> Asking<'a, A> {
    /// The outcome a model decides, asking at the rule's decide level and one
    /// level higher on each `unsure`, up to its explain level; `None` if still
    /// unsure there.
    async fn decide(&mut self) -> Result<Option<&'a Outcome>> {
        let rule = self.rule;
        let (system, schema) = (decide_preamble(rule), decision_schema(rule));
        let mut level = rule.levels.decide;
        loop {
            let reply = self.ask(level, &system, &schema).await?;
            if let Some(outcome) = parse_decision(rule, &reply)? {
                return Ok(Some(outcome));
            }
            match level.above() {
                Some(next) if next <= rule.levels.explain => level = next,
                _ => return Ok(None),
            }
        }
    }

    /// The findings a model at the rule's explain level reports for a unit
    /// decided as `outcome`.
    async fn explain(&mut self, outcome: &Outcome) -> Result<Vec<Finding>> {
        let rule = self.rule;
        let system = explain_preamble(rule, outcome);
        let reply = self
            .ask(rule.levels.explain, &system, &findings_schema())
            .await?;
        let path = &self.unit.call.path;
        Ok(parse_findings(&reply)?
            .into_iter()
            .map(|r| Finding {
                rule: rule.id.clone(),
                outcome: outcome.name.clone(),
                path: r.path.unwrap_or_else(|| path.clone()),
                line: r.line,
                message: r.message,
            })
            .collect())
    }

    async fn ask(
        &mut self,
        level: ModelLevel,
        system: &str,
        schema: &serde_json::Value,
    ) -> Result<String> {
        let question = Question {
            level,
            system,
            prompt: &self.unit.state,
            schema: Some(schema),
        };
        let answer = self.model.ask(&question).await?;
        self.usage += answer.usage;
        Ok(answer.text)
    }
}

fn place(call: &Call) -> String {
    format!("{}:{}", call.path.display(), call.snippet.first_line)
}
