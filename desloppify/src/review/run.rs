//! Making the calls.

use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::task::JoinSet;

use super::reply::{parse, preamble, prompt, schema};
use super::{Call, Finding, Report};
use crate::agent::{Ask, Question, Usage};
use crate::rule::Rule;

pub(super) async fn review<A: Ask>(
    agent: Arc<A>,
    rules: Arc<Vec<Rule>>,
    plan: Vec<Call>,
) -> Result<Report> {
    let mut calls = JoinSet::new();
    for call in plan {
        let (agent, rules) = (agent.clone(), rules.clone());
        calls.spawn(async move {
            let rule = &rules[call.rule];
            let (preamble, prompt) = (preamble(rule), prompt(&call));
            let context = || {
                format!(
                    "rule {} on {}:{}",
                    rule.id,
                    call.path.display(),
                    call.snippet.first_line
                )
            };
            // A call that answered was paid for, even if its answer is unreadable.
            let schema = schema();
            let question = Question {
                level: rule.level,
                system: &preamble,
                prompt: &prompt,
                schema: Some(&schema),
            };
            let answer = match agent.ask(&question).await {
                Ok(answer) => answer,
                Err(error) => return (call.rule, Usage::default(), Err(error.context(context()))),
            };
            let findings = parse(&answer.text).with_context(context).map(|replies| {
                replies
                    .into_iter()
                    .map(|r| Finding {
                        rule: rule.id.clone(),
                        path: r.path.unwrap_or_else(|| call.path.clone()),
                        line: r.line,
                        message: r.message,
                    })
                    .collect::<Vec<_>>()
            });
            (call.rule, answer.usage, findings)
        });
    }

    let mut report = Report::default();
    while let Some(joined) = calls.join_next().await {
        let (rule, usage, findings) = joined?;
        *report.usage.entry(rules[rule].id.clone()).or_default() += usage;
        match findings {
            Ok(findings) => report.findings.extend(findings),
            Err(failure) => report.failures.push(failure),
        }
    }
    report
        .findings
        .sort_by(|a, b| (&a.path, a.line, &a.rule).cmp(&(&b.path, b.line, &b.rule)));
    Ok(report)
}
