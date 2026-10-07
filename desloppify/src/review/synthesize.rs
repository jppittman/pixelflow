//! One frontier call reading every finding and writing the review.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::Result;

use super::{Finding, Report};
use crate::agent::{Answer, Ask, Question};
use crate::model::ModelLevel;
use crate::rule::{Rule, Verdict};

const LEAD: &str = "\
You are the lead reviewer. Other reviewers each applied one rule, given below, \
to one piece of code, and reported what they found. Write the code review the \
author will read, in Markdown:\n\
- Group by file. Within a file, most serious first.\n\
- Merge findings that are the same problem reported under different rules or \
lines, and name every rule involved.\n\
- Drop findings that are trivial or that misread their rule, and say at the end \
how many you dropped and why, in one line.\n\
- For each item kept: `path:line`, what is wrong, the rule(s), and the fix.\n\
- Report nothing no reviewer found.\n\
- Finish with the patterns that recur across files, if any: those are what the \
author most needs to hear.";

pub(super) async fn synthesize<A: Ask>(
    agent: &A,
    rules: &[Rule],
    report: &Report,
) -> Result<Option<Answer>> {
    if report.findings.is_empty() {
        return Ok(None);
    }
    let prompt = brief(rules, report);
    let question = Question {
        level: ModelLevel::Frontier,
        system: LEAD,
        prompt: &prompt,
        schema: None,
    };
    agent.ask(&question).await.map(Some)
}

/// The rules that found something, then the findings by file.
fn brief(rules: &[Rule], report: &Report) -> String {
    let mut by_file: BTreeMap<&PathBuf, Vec<&Finding>> = BTreeMap::new();
    for finding in &report.findings {
        by_file.entry(&finding.path).or_default().push(finding);
    }

    let mut brief = String::from("## Rules\n\n");
    for rule in rules
        .iter()
        .filter(|r| report.findings.iter().any(|f| f.rule == r.id))
    {
        brief.push_str(&format!("- `{}`: {}\n", rule.id, rule.decision.question));
        for outcome in rule
            .decision
            .outcomes
            .iter()
            .filter(|o| o.verdict == Verdict::Violation)
        {
            brief.push_str(&format!("  - `{}`: {}\n", outcome.name, outcome.meaning));
        }
    }
    brief.push_str("\n## Findings\n");
    for (path, findings) in by_file {
        brief.push_str(&format!("\n### {}\n\n", path.display()));
        for f in findings {
            brief.push_str(&format!(
                "- line {} [{}/{}]: {}\n",
                f.line, f.rule, f.outcome, f.message
            ));
        }
    }
    if !report.failures.is_empty() {
        brief.push_str(&format!(
            "\n{} reviews failed; the code they covered was not reviewed.\n",
            report.failures.len()
        ));
    }
    brief
}
