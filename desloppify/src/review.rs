//! A review: every rule over every file it applies to, one agent call per snippet.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::Deserialize;
use tokio::task::JoinSet;

use crate::agent::Agent;
use crate::language::Language;
use crate::rule::Rule;
use crate::snippet::snippets;

const REVIEWER: &str = "\
You are a code reviewer applying exactly one rule, given below. Report only \
violations of that rule, in the code you are shown. Reply with only a JSON \
array, no prose: one {\"line\": <line number>, \"message\": \"<what and why>\"} \
per violation, using the line numbers printed in the code. Reply [] if the \
code does not violate the rule.";

#[derive(Debug)]
pub struct Finding {
    pub rule: String,
    pub path: PathBuf,
    pub line: u64,
    pub message: String,
}

#[derive(Default)]
pub struct Report {
    pub findings: Vec<Finding>,
    /// Snippets that could not be reviewed: a failed call or an unreadable reply.
    pub failures: Vec<anyhow::Error>,
}

#[derive(Deserialize)]
struct Reply {
    line: u64,
    message: String,
}

pub async fn review(agent: Arc<Agent>, rules: Arc<Vec<Rule>>, files: &[PathBuf]) -> Result<Report> {
    let mut calls = JoinSet::new();
    for path in files {
        let Some(language) = Language::of(path) else {
            continue;
        };
        let source =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        for (index, rule) in rules.iter().enumerate() {
            for snippet in snippets(rule, language, &source)
                .with_context(|| format!("parsing {}", path.display()))?
            {
                let (agent, rules, path) = (agent.clone(), rules.clone(), path.clone());
                calls.spawn(async move {
                    let rule = &rules[index];
                    let prompt = format!("File: {}\n\n{}", path.display(), snippet.numbered);
                    let preamble = format!("{REVIEWER}\n\n{}", rule.instructions);
                    let replies =
                        async { parse(&agent.ask(rule.level, &preamble, &prompt).await?) }
                            .await
                            .with_context(|| {
                                format!(
                                    "rule {} on {}:{}",
                                    rule.id,
                                    path.display(),
                                    snippet.first_line
                                )
                            })?;
                    Ok::<_, anyhow::Error>(
                        replies
                            .into_iter()
                            .map(|r| Finding {
                                rule: rule.id.clone(),
                                path: path.clone(),
                                line: r.line,
                                message: r.message,
                            })
                            .collect::<Vec<_>>(),
                    )
                });
            }
        }
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

/// The findings in a reply, tolerating the code fence models like to add.
fn parse(reply: &str) -> Result<Vec<Reply>> {
    let body = reply.trim();
    let body = body
        .strip_prefix("```json")
        .or_else(|| body.strip_prefix("```"))
        .unwrap_or(body);
    let body = body.strip_suffix("```").unwrap_or(body);
    serde_json::from_str(body.trim())
        .with_context(|| format!("reply is not a findings array: {reply}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fenced_and_bare_replies_both_parse() {
        assert!(parse("[]").unwrap().is_empty());
        let fenced = parse("```json\n[{\"line\": 3, \"message\": \"m\"}]\n```").unwrap();
        assert_eq!(fenced[0].line, 3);
    }

    #[test]
    fn prose_reply_is_a_failure() {
        assert!(parse("Looks good to me!").is_err());
    }
}
