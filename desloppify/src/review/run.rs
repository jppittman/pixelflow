//! Making the calls.

use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::task::JoinSet;

use super::reply::{parse, preamble, prompt};
use super::{Call, Finding, Report};
use crate::agent::Ask;
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
            let replies = async { parse(&agent.ask(rule.level, &preamble, &prompt).await?) }
                .await
                .with_context(|| {
                    format!(
                        "rule {} on {}:{}",
                        rule.id,
                        call.path.display(),
                        call.snippet.first_line
                    )
                })?;
            Ok::<_, anyhow::Error>(
                replies
                    .into_iter()
                    .map(|r| Finding {
                        rule: rule.id.clone(),
                        path: r.path.unwrap_or_else(|| call.path.clone()),
                        line: r.line,
                        message: r.message,
                    })
                    .collect::<Vec<_>>(),
            )
        });
    }

    let mut report = Report::default();
    while let Some(joined) = calls.join_next().await {
        match joined? {
            Ok(findings) => report.findings.extend(findings),
            Err(failure) => report.failures.push(failure),
        }
    }
    report
        .findings
        .sort_by(|a, b| (&a.path, a.line, &a.rule).cmp(&(&b.path, b.line, &b.rule)));
    Ok(report)
}
