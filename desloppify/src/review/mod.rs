//! A review: every rule over every file it applies to, one call per snippet.
//!
//! [`plan`] decides the calls without making any; [`review`] makes them
//! through an [`Ask`] and gathers what comes back; [`synthesize`] has one
//! frontier call read everything gathered and write the review a person
//! reads. Each reviewer sees one rule and a little code, so none forgets a
//! rule; the lead sees every finding, so the review reads as one.

mod plan;
mod reply;
mod run;
mod synthesize;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;

use crate::agent::{Answer, Ask, Usage};
use crate::rule::Rule;
use crate::snippet::Snippet;

/// One call: a rule (by index into the rules planned with) over one snippet
/// of one file.
pub struct Call {
    pub rule: usize,
    pub path: PathBuf,
    pub snippet: Snippet,
    /// The module root, when the rule asks for it and the file has one.
    pub root: Option<Source>,
}

/// A file read for context.
#[derive(Clone)]
pub struct Source {
    pub path: PathBuf,
    pub text: String,
}

/// A violation a model reported.
#[derive(Debug)]
pub struct Finding {
    /// The rule's id.
    pub rule: String,
    pub path: PathBuf,
    /// The line the model named, as numbered in the snippet it was shown.
    pub line: u64,
    pub message: String,
}

/// Everything a review produced.
#[derive(Default)]
pub struct Report {
    /// Sorted by path, then line, then rule.
    pub findings: Vec<Finding>,
    /// One per call that produced no findings list: the call failed, or its
    /// reply was not a JSON findings array. Each names its rule, file and
    /// first line.
    pub failures: Vec<anyhow::Error>,
    /// Tokens used, by rule id: every call that was answered, whether or not
    /// its answer could be read. A call that failed used nothing.
    pub usage: BTreeMap<String, Usage>,
}

/// Every call a review of `files` under `rules` makes, in file order then
/// rule order. Files in no known language, and rules whose `Files` exclude a
/// file, plan nothing; a rule asking for `"module_root"` context gets the
/// root of each file's module (none for a file that is one).
///
/// # Errors
///
/// A file or module root cannot be read, or a file cannot be parsed.
pub fn plan(rules: &[Rule], files: &[PathBuf]) -> Result<Vec<Call>> {
    plan::plan(rules, files)
}

/// Makes every call in `plan` through `agent` and gathers the findings.
///
/// Each call's system prompt is the reviewer's instructions followed by its
/// rule's; its user prompt names the file and holds the snippet, then any
/// module root, marked as context only. Each call asks for a reply in a
/// fixed JSON shape — `{"findings": [{"line", "message", "path"?}]}` — which
/// the backend has the model satisfy; a reply that does not is a failure, as
/// is a call that errs. One call failing does not stop the others.
///
/// # Errors
///
/// A call's task panicked.
pub async fn review<A: Ask>(
    agent: Arc<A>,
    rules: Arc<Vec<Rule>>,
    plan: Vec<Call>,
) -> Result<Report> {
    run::review(agent, rules, plan).await
}

/// The review a person reads, written by one call at
/// [`ModelLevel::Frontier`](crate::model::ModelLevel::Frontier) from every
/// finding in `report`: grouped by file, duplicates across rules merged,
/// trivial or mistaken findings dropped (and counted), recurring patterns
/// called out. The call is told the prompt of each rule that found something
/// and how many reviews failed. It returns the reply verbatim, with its usage.
///
/// `Ok(None)`, with no call made, when `report` has no findings.
///
/// # Errors
///
/// The call failed.
pub async fn synthesize<A: Ask>(
    agent: &A,
    rules: &[Rule],
    report: &Report,
) -> Result<Option<Answer>> {
    synthesize::synthesize(agent, rules, report).await
}
