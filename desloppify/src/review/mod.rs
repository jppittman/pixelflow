//! A review: every rule over every file it applies to, one call per snippet.
//!
//! [`plan`] decides the calls without making any; [`review`] makes them
//! through an [`Ask`] and gathers what comes back.

mod plan;
mod reply;
mod run;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;

use crate::agent::Ask;
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
/// module root, marked as context only. A reply is a JSON array of
/// `{"line", "message"}`, optionally inside a code fence; anything else is a
/// failure, as is a call that errs. One call failing does not stop the others.
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
