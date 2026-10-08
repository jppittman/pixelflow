//! A review: every rule's decision about every unit it applies to, the
//! violations explained, and one lead review over all of it.
//!
//! [`plan`] decides the calls without making any. [`review`] makes them in
//! System One's cascade: each call's unit goes to a [`Decide`] with every
//! applicable rule's question at once; an answer at or above
//! `min_confidence` settles that rule, and the rest are asked of a model
//! through [`Ask`] at the rule's decide level, escalating a level on
//! `unsure`. A unit decided as a violation is shown to a model at the rule's
//! explain level, which writes the findings. [`synthesize`] has one frontier
//! call read every finding and write the review a person reads. Each model
//! call sees one rule and a little code, so none forgets a rule.

mod diagnose;
mod plan;
mod reply;
mod run;
mod synthesize;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;

use crate::agent::{Answer, Ask, Usage};
use crate::decide::Decide;
use crate::rule::Rule;
use crate::snippet::Snippet;

/// One unit of one file, and the rules (by index into the rules planned
/// with) that ask about it.
pub struct Call {
    /// Never empty; every rule here sees the same unit and context.
    pub(crate) rules: Vec<usize>,
    /// The file, or for a crate-wide rule the crate's directory.
    pub(crate) path: PathBuf,
    pub(crate) snippet: Snippet,
    /// The module root's outline, when the rules ask for it and the file has
    /// one.
    pub(crate) root: Option<Source>,
}

/// A file shown for context.
#[derive(Clone, Debug)]
pub(crate) struct Source {
    pub(crate) path: PathBuf,
    /// Numbered, as a snippet is.
    pub(crate) text: String,
}

/// Who a review asks, and how sure System One must be to settle a question.
pub struct Reviewers<A, D> {
    pub ask: A,
    pub decide: D,
    /// An answer below this confidence goes to the model.
    pub min_confidence: f64,
}

/// A violation a model reported.
#[derive(Debug)]
pub struct Finding {
    /// The rule's id.
    pub rule: String,
    /// The violation outcome this place shows: the one the unit was decided
    /// as, or another of the rule's violations found beside it.
    pub outcome: String,
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
    /// One per rule and unit that could not be decided or explained — a call
    /// failed or its reply did not match its schema — and one per System One
    /// request that failed (its questions went to the model instead). Each
    /// names its rule or file and first line.
    pub failures: Vec<anyhow::Error>,
    /// Tokens used by model calls, by rule id: every call that was answered,
    /// whether or not its answer could be read.
    pub usage: BTreeMap<String, Usage>,
    /// Tokens used by System One, for every rule together.
    pub system_one: Usage,
    /// How each rule decided its units, by outcome name; `unsure` counts the
    /// units still undecided at the explain level.
    pub decisions: BTreeMap<String, BTreeMap<String, u64>>,
    /// Units each rule's question went to the model for, because System One
    /// did not answer it confidently.
    pub escalated: BTreeMap<String, u64>,
    /// What the findings are symptoms of, once [`diagnose`] has been asked;
    /// [`synthesize`] leads with these.
    pub diagnoses: Vec<Diagnosis>,
}

/// What a cluster of findings in one module is a symptom of.
#[derive(Debug, Clone)]
pub struct Diagnosis {
    /// The module the symptoms are in.
    pub component: PathBuf,
    /// What the code is, in its domain's words: "an assembler".
    pub thing: String,
    /// The thing described from first principles, by a call that never saw
    /// the code.
    pub denotation: String,
    /// The description's parts and behaviours mapped onto the code: where
    /// each is — a type, a function, a convention, nowhere — and what the
    /// code has that the description does not.
    pub shape: String,
    /// The model of the thing that the code is missing or has wrong.
    pub diagnosis: String,
    /// The code with the thing's shape: what becomes a type, what is
    /// deleted, what falls out.
    pub falls_out: String,
    /// The findings it explains, as indexes into [`Report::findings`].
    pub explains: Vec<usize>,
}

/// What [`diagnose`] produced.
#[derive(Default)]
pub struct Diagnoses {
    pub diagnoses: Vec<Diagnosis>,
    /// One per module or thing whose call failed or whose reply did not
    /// match its schema; the others still count.
    pub failures: Vec<anyhow::Error>,
    /// Tokens every diagnosis call used.
    pub usage: Usage,
}

/// Every call a review of `files` under `rules` makes, in file order. Rules
/// that apply to a file and see the same unit and context share its calls;
/// a crate-wide rule gets one call per crate, its files under `== path ==`
/// headers. Files in no known language, and rules whose `Files` exclude a
/// file, plan nothing.
///
/// # Errors
///
/// A file or module root cannot be read, or a file cannot be parsed.
pub fn plan(rules: &[Rule], files: &[PathBuf]) -> Result<Vec<Call>> {
    plan::plan(rules, files)
}

/// Makes every call in `plan` and gathers the decisions and findings.
///
/// A rule's model prompts hold its question, its outcomes' meanings and its
/// skills; the user prompt names the file, holds the snippet, then any
/// module root, marked as context only. Every reply is held to a JSON
/// schema by the backend; one that does not match is a failure, as is a call
/// that errs. One failing does not stop the others.
///
/// # Errors
///
/// A call's task panicked.
pub async fn review<A: Ask, D: Decide>(
    reviewers: Arc<Reviewers<A, D>>,
    rules: Arc<Vec<Rule>>,
    plan: Vec<Call>,
) -> Result<Report> {
    run::review(reviewers, rules, plan).await
}

/// What the findings are symptoms of: from symptom to diagnosis.
///
/// A module whose findings converge — at least three, from at least two
/// rules — is diagnosed in three steps:
///
/// 1. One call at [`ModelLevel::Strong`](crate::model::ModelLevel::Strong),
///    shown the symptoms and the module's outline, names the things they
///    are about in their domain's words — "an assembler" — each with the
///    symptoms that concern it.
/// 2. For each thing at least two symptoms concern, one call at
///    [`ModelLevel::Frontier`](crate::model::ModelLevel::Frontier) describes
///    it from first principles: what it is, its parts, what it does, how it
///    behaves. It is shown only the thing's name and context, never the
///    code, so it cannot borrow the code's model of itself.
/// 3. One more frontier call translates that description's shape onto the
///    code's — where each part is, or that it is nowhere — and states the
///    diagnosis: the model the code is missing, the symptoms it explains,
///    and what falls out once the code has the thing's shape.
///
/// Every reply is held to a schema. A failed step is a failure in the
/// result and stops only its own module or thing.
///
/// # Errors
///
/// A module's files cannot be read or parsed.
pub async fn diagnose<A: Ask>(agent: &A, report: &Report) -> Result<Diagnoses> {
    diagnose::diagnose(agent, report).await
}

/// The review a person reads, written by one call at
/// [`ModelLevel::Frontier`](crate::model::ModelLevel::Frontier) from every
/// finding in `report`: grouped by file, duplicates across rules merged,
/// trivial or mistaken findings dropped (and counted), recurring patterns
/// called out. The call is told each rule's question and violation meanings
/// and how many reviews failed. It returns the reply verbatim, with its
/// usage.
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
