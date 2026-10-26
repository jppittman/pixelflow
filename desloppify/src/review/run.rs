//! Making the calls: System One, then the model, then the explanation.

use std::sync::Arc;

use anyhow::{Context, Result};
use futures::future::join_all;
use tokio::task::JoinSet;

use super::reply::{
    Chosen, Step, ask, decide_preamble, decision_schema, explain_preamble, findings_schema, labels,
    parse_decision, parse_findings, prompt,
};
use super::{Call, Finding, Report, Session};
use crate::agent::{Ask, Usage};
use crate::decide::{Choice, Decide, Decisions};
use crate::model::ModelLevel;
use crate::rule::{Outcome, Rule, Verdict};

/// What one rule made of one unit.
struct Judged {
    rule: Arc<Rule>,
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
    session: Arc<Session<A, D>>,
    calls: Vec<Call>,
) -> Result<Report> {
    let mut report = Report::default();
    let mut tasks = JoinSet::new();
    for call in calls {
        for rule in &call.rules {
            report
                .rules
                .entry(rule.id.clone())
                .or_insert_with(|| rule.clone());
        }
        let session = session.clone();
        tasks.spawn(async move { reviewed(&session, &call).await });
    }

    while let Some(joined) = tasks.join_next().await {
        let reviewed = joined?;
        report.system_one += reviewed.system_one;
        report.failures.extend(reviewed.system_one_failure);
        for judged in reviewed.judged {
            let id = &judged.rule.id;
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
    // A total order: calls finish in any order, and the synthesis numbers
    // the findings in this one.
    report.findings.sort_by(|a, b| {
        (&a.path, a.line, &a.rule, &a.outcome, &a.message)
            .cmp(&(&b.path, b.line, &b.rule, &b.outcome, &b.message))
    });
    Ok(report)
}

async fn reviewed<A: Ask, D: Decide>(session: &Session<A, D>, call: &Call) -> Reviewed {
    let state = prompt(call);
    let labels: Vec<Vec<(&str, &str)>> = call.rules.iter().map(|rule| labels(rule)).collect();
    let questions: Vec<Choice<'_>> = call
        .rules
        .iter()
        .zip(&labels)
        .map(|(rule, labels)| Choice {
            name: &rule.id,
            instructions: &rule.decision.question,
            labels,
        })
        .collect();
    // If System One fails, every question goes to the model: a slower
    // review, not a smaller one.
    let (decisions, system_one_failure) = match session.decide.decide(&state, &questions).await {
        Ok(decisions) => (decisions, None),
        Err(error) => {
            let error = error.context(format!("System One on {}", place(call)));
            (Decisions::default(), Some(error))
        }
    };

    let unit = Seen {
        call,
        state,
        decisions,
    };
    // Each rule's questions about the unit are independent of the others'.
    let judged = join_all(call.rules.iter().map(|rule| judge(session, &unit, rule))).await;
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
    /// What every model call about the unit is shown; System One's state.
    state: String,
    decisions: Decisions,
}

async fn judge<A: Ask, D: Decide>(
    session: &Session<A, D>,
    unit: &Seen<'_>,
    rule: &Arc<Rule>,
) -> Judged {
    let context = || format!("rule {} on {}", rule.id, place(unit.call));
    let settled = unit
        .decisions
        .answers
        .get(&rule.id)
        .filter(|d| d.confidence >= session.min_confidence)
        .and_then(|d| rule.decision.outcomes.iter().find(|o| o.name == d.label))
        .map(Chosen::Outcome);
    let mut asking = Asking {
        model: &session.ask,
        rule,
        unit,
        usage: Usage::default(),
    };
    let decided = match settled {
        Some(chosen) => Ok(chosen),
        None => asking.decide().await,
    };
    let (outcome, findings) = match decided {
        Err(error) => (Err(error.context(context())), Ok(Vec::new())),
        Ok(Chosen::Outcome(outcome)) if outcome.verdict == Verdict::Violation => (
            Ok(outcome.name.clone()),
            asking.explain(outcome).await.with_context(context),
        ),
        Ok(chosen) => (Ok(chosen.name().to_owned()), Ok(Vec::new())),
    };
    Judged {
        rule: rule.clone(),
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
    /// What a model decides, asking at the rule's decide level and one level
    /// higher on each `unsure`, up to its explain level; `Unsure` if still
    /// unsure there.
    async fn decide(&mut self) -> Result<Chosen<'a>> {
        let rule = self.rule;
        let (system, schema) = (decide_preamble(rule), decision_schema(rule));
        let mut level = rule.levels.decide;
        loop {
            let reply = self.ask(level, &system, &schema).await?;
            match parse_decision(rule, &reply)? {
                Chosen::Unsure => match level.above() {
                    Some(next) if next <= rule.levels.explain => level = next,
                    _ => return Ok(Chosen::Unsure),
                },
                settled => return Ok(settled),
            }
        }
    }

    /// The findings a model at the rule's explain level reports for a unit
    /// decided as `outcome`, each under the violation it shows.
    async fn explain(&mut self, outcome: &Outcome) -> Result<Vec<Finding>> {
        let rule = self.rule;
        let system = explain_preamble(rule, outcome);
        let reply = self
            .ask(rule.levels.explain, &system, &findings_schema(rule))
            .await?;
        let path = &self.unit.call.path;
        Ok(parse_findings(rule, &reply)?
            .into_iter()
            .map(|r| Finding {
                rule: rule.id.clone(),
                outcome: r.outcome,
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
        let step = Step {
            level,
            system,
            schema,
        };
        let answer = ask(self.model, &step, &self.unit.state).await?;
        self.usage += answer.usage;
        Ok(answer.text)
    }
}

fn place(call: &Call) -> String {
    format!("{}:{}", call.path.display(), call.snippet.first_line)
}
