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
author most needs to hear.\n\
- If roots are given, lead with them: each is the model beneath the \
diagnoses it explains. Then the diagnoses, then the files.\n\
- If diagnoses are given, lead with them, before the files: each is the cause \
of the findings it explains. Say what the thing is, the model the code is \
missing, and what falls out once the code has the thing's shape; under it, \
list the findings it explains by `path:line`, briefly, and do not repeat them \
in the file sections.";

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

/// The diagnoses, the rules that found something, then the findings by
/// file, each numbered `F1`… in report order so a diagnosis can name the
/// ones it explains.
fn brief(rules: &[Rule], report: &Report) -> String {
    let mut by_file: BTreeMap<&PathBuf, Vec<(usize, &Finding)>> = BTreeMap::new();
    for (index, finding) in report.findings.iter().enumerate() {
        by_file
            .entry(&finding.path)
            .or_default()
            .push((index, finding));
    }

    let mut brief = String::new();
    if !report.roots.is_empty() {
        brief.push_str("## Roots\n");
        for r in &report.roots {
            let explains: Vec<&str> = r
                .explains
                .iter()
                .filter_map(|&i| report.diagnoses.get(i).map(|d| d.thing.as_str()))
                .collect();
            brief.push_str(&format!(
                "\n### {}\n\nRoot: {}\n\nWhat falls out: {}\n\nExplains the diagnoses of: {}\n",
                r.component.display(),
                r.root,
                r.falls_out,
                explains.join("; ")
            ));
        }
        brief.push('\n');
    }
    if !report.diagnoses.is_empty() {
        brief.push_str("## Diagnoses\n");
        for d in &report.diagnoses {
            let explains: Vec<String> = d.explains.iter().map(|i| format!("F{}", i + 1)).collect();
            brief.push_str(&format!(
                "\n### {} — {}\n\nDiagnosis: {}\n\nThe thing, from first principles:\n{}\n\nIts shape in the code: {}\n\nWith the thing's shape: {}\n\nExplains: {}\n",
                d.thing,
                d.component.display(),
                d.diagnosis,
                d.denotation,
                d.shape,
                d.falls_out,
                explains.join(", ")
            ));
        }
        brief.push('\n');
    }
    brief.push_str("## Rules\n\n");
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
        for (index, f) in findings {
            brief.push_str(&format!(
                "- F{} line {} [{}/{}]: {}\n",
                index + 1,
                f.line,
                f.rule,
                f.outcome,
                f.message
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
