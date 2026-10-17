//! One frontier call over everything the small calls found: it diagnoses the
//! things they named, finds the roots beneath the diagnoses, and writes the
//! review.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use super::identify::shape;
use super::{Diagnosis, Finding, Report, Root, Synthesis};
use crate::agent::{Ask, Question};
use crate::model::ModelLevel;
use crate::rule::{Rule, Verdict};

/// The most outline lines the synthesis is shown, over every module a thing
/// was named in, the modules with the most symptoms first.
const MAX_SHAPE_LINES: usize = 12_000;

const LEAD: &str = "\
You are the lead reviewer of a codebase. Many small reviewers each applied one \
rule, given below, to one piece of code and reported what they found: the \
findings, numbered F1…, each a symptom. Where symptoms converged on a module, \
other reviewers named the things they are about (T1…), each described from \
first principles by a reviewer who never saw the code, and you are shown those \
modules' outlines: items and signatures, bodies elided, at their line numbers. \
You are the only reviewer who sees all of it. Your judgment is the review.\n\
\n\
Diagnose. For each thing worth it, translate its description's shape onto the \
code's outline — where each part and behaviour is: a type, a function, a \
convention kept in a comment, or nowhere — and state the model of the thing \
the code is missing or has wrong, as what the thing is (\"a label is an \
operand\", \"instruction selection is a phase\"), with the findings it \
explains. One diagnosis per thing, the deepest the evidence supports; if it \
explains one finding, look further. Then follow its consequences to closure: \
with the thing's shape, what stops being special, what becomes a type, what \
is deleted, and what follows from that, until nothing more falls out. Refuse \
exceptions: a fix that keeps one is a symptom not yet explained.\n\
\n\
Diagnoses are symptoms too, across modules as well as within one. Find the \
roots: the deepest models whose absence makes several diagnoses true at once \
— a missing phase, a missing type, a wrong boundary — each with the \
diagnoses it explains and its consequences to closure.\n\
\n\
Then write the review the author will read, in Markdown: the roots first, \
then the diagnoses, each with the findings it explains by `path:line`, \
briefly; then the remaining findings grouped by file, most serious first. \
Merge findings that are the same problem under different rules or lines, \
naming every rule involved. Drop findings that are trivial or misread their \
rule, and say at the end how many and why, in one line. For each item kept: \
`path:line`, what is wrong, the rule(s), and the fix. Report nothing no \
reviewer found. Finish with the patterns that recur across files.";

fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "diagnoses": { "type": "array", "items": {
                "type": "object",
                "properties": {
                    "thing": { "type": "integer" },
                    "diagnosis": { "type": "string" },
                    "falls_out": { "type": "string" },
                    "explains": { "type": "array", "items": { "type": "integer" } }
                },
                "required": ["thing", "diagnosis", "falls_out", "explains"]
            } },
            "roots": { "type": "array", "items": {
                "type": "object",
                "properties": {
                    "root": { "type": "string" },
                    "falls_out": { "type": "string" },
                    "explains": { "type": "array", "items": { "type": "integer" } }
                },
                "required": ["root", "falls_out", "explains"]
            } },
            "review": { "type": "string" }
        },
        "required": ["diagnoses", "roots", "review"]
    })
}

#[derive(Deserialize)]
struct Reply {
    diagnoses: Vec<Diagnosed>,
    roots: Vec<Rooted>,
    review: String,
}

#[derive(Deserialize)]
struct Diagnosed {
    /// A thing's number, from 1.
    thing: usize,
    diagnosis: String,
    falls_out: String,
    /// Finding numbers, from 1.
    explains: Vec<usize>,
}

#[derive(Deserialize)]
struct Rooted {
    root: String,
    falls_out: String,
    /// Diagnosis numbers, from 1, in the reply's order.
    explains: Vec<usize>,
}

pub(super) async fn synthesize<A: Ask>(
    agent: &A,
    rules: &[Rule],
    report: &Report,
) -> Result<Option<Synthesis>> {
    if report.findings.is_empty() {
        return Ok(None);
    }
    let prompt = brief(rules, report)?;
    let schema = schema();
    let question = Question {
        level: ModelLevel::Frontier,
        system: LEAD,
        prompt: &prompt,
        schema: Some(&schema),
    };
    let answer = agent.ask(&question).await?;
    let reply: Reply = serde_json::from_str(&answer.text)
        .with_context(|| format!("reply is not a synthesis: {}", answer.text))?;

    let mut diagnoses = Vec::new();
    for d in reply.diagnoses {
        let Some(thing) = d.thing.checked_sub(1).filter(|t| *t < report.things.len()) else {
            bail!(
                "a diagnosis names T{}; there are {} things",
                d.thing,
                report.things.len()
            );
        };
        diagnoses.push(Diagnosis {
            thing,
            diagnosis: d.diagnosis,
            falls_out: d.falls_out,
            explains: within(&d.explains, report.findings.len()),
        });
    }
    let roots = reply
        .roots
        .into_iter()
        .map(|r| Root {
            root: r.root,
            falls_out: r.falls_out,
            explains: within(&r.explains, diagnoses.len()),
        })
        .collect();
    Ok(Some(Synthesis {
        review: reply.review,
        diagnoses,
        roots,
        usage: answer.usage,
    }))
}

/// Numbers from 1 as indexes below `count`; a number past it names nothing.
fn within(numbers: &[usize], count: usize) -> Vec<usize> {
    numbers
        .iter()
        .filter_map(|n| n.checked_sub(1).filter(|i| *i < count))
        .collect()
}

/// The rules that found something, the things named, the outlines of their
/// modules, then the findings by file, each numbered `F1`… in report order.
fn brief(rules: &[Rule], report: &Report) -> Result<String> {
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

    if !report.things.is_empty() {
        brief.push_str("\n## Things\n");
        for (n, t) in report.things.iter().enumerate() {
            let symptoms: Vec<String> = t.explains.iter().map(|i| format!("F{}", i + 1)).collect();
            brief.push_str(&format!(
                "\n### T{} — {}, {} — {}\n\nFrom first principles:\n{}\n\nSymptoms: {}\n",
                n + 1,
                t.thing,
                t.context,
                t.component.display(),
                t.description,
                symptoms.join(", ")
            ));
        }
        brief.push_str("\n## Outlines\n\n");
        brief.push_str(&outlines(report)?);
    }

    let mut by_file: BTreeMap<&PathBuf, Vec<(usize, &Finding)>> = BTreeMap::new();
    for (index, finding) in report.findings.iter().enumerate() {
        by_file
            .entry(&finding.path)
            .or_default()
            .push((index, finding));
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
    Ok(brief)
}

/// The outline of every module a thing was named in, the modules with the
/// most symptoms first, to [`MAX_SHAPE_LINES`] in all.
fn outlines(report: &Report) -> Result<String> {
    let mut symptoms: BTreeMap<&Path, Vec<&Path>> = BTreeMap::new();
    for thing in &report.things {
        let paths = symptoms.entry(thing.component.as_path()).or_default();
        paths.extend(
            thing
                .explains
                .iter()
                .map(|&i| report.findings[i].path.as_path()),
        );
    }
    let mut modules: Vec<(&Path, Vec<&Path>)> = symptoms.into_iter().collect();
    modules.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(b.0)));

    let mut outlines = String::new();
    let mut left = MAX_SHAPE_LINES;
    for (component, paths) in modules {
        if left == 0 {
            outlines.push_str(&format!(
                "== {} == (omitted: over budget)\n",
                component.display()
            ));
            continue;
        }
        let (shape, used) = shape(component, paths.into_iter(), left)?;
        outlines.push_str(&shape);
        left = left.saturating_sub(used);
    }
    Ok(outlines)
}
