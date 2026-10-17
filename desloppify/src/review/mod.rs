//! A review: every rule's decision about every unit it applies to, the
//! violations explained, the things they converge on named, and one
//! frontier call's judgment over all of it.
//!
//! [`plan`] decides the calls without making any. [`review`] makes them in
//! System One's cascade: each call's unit goes to a [`Decide`] with every
//! applicable rule's question at once; an answer at or above
//! `min_confidence` settles that rule, and the rest are asked of a model
//! through [`Ask`] at the rule's decide level, escalating a level on
//! `unsure`. A unit decided as a violation is shown to a model at the rule's
//! explain level, which writes the findings. [`identify`] names what the
//! findings in each converging module are about, and describes each thing
//! without the code. [`synthesize`] has one frontier call read all of it,
//! diagnose, find the roots and write the review a person reads.
//!
//! Small models make many calls, each seeing one rule or one module, so none
//! forgets a rule; the one large call is the only one that sees everything.

mod identify;
mod plan;
mod reply;
mod run;
mod synthesize;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;

use crate::agent::{Ask, Usage};
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
    /// The module's `intent.md`, when the rules ask for it and one is found
    /// between the file's directory and its crate root.
    pub(crate) intent: Option<Source>,
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
#[derive(Debug, serde::Serialize)]
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
    /// Sorted by path, then line, then rule, then outcome, then message.
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
    /// What the findings in each converging module are about, once
    /// [`identify`] has been asked; [`synthesize`] diagnoses these.
    pub things: Vec<Thing>,
}

/// A thing several findings in one module are about, named by its domain
/// and described without the code.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Thing {
    /// The module the symptoms are in.
    pub component: PathBuf,
    /// What the code is, in its domain's words: "an assembler".
    pub thing: String,
    /// What it is part of: "in a JIT compiler".
    pub context: String,
    /// The thing described from first principles, by a call that never saw
    /// the code.
    pub description: String,
    /// The findings that concern it, as indexes into [`Report::findings`].
    pub explains: Vec<usize>,
}

/// What [`identify`] produced.
#[derive(Default)]
pub struct Identified {
    pub things: Vec<Thing>,
    /// One per module or thing whose call failed or whose reply did not
    /// match its schema; the others still count.
    pub failures: Vec<anyhow::Error>,
    /// Tokens every call used.
    pub usage: Usage,
}

/// The model of a thing the code is missing or has wrong.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Diagnosis {
    /// The thing, as an index into [`Report::things`].
    pub thing: usize,
    /// The model, stated as what the thing is: "a label is an operand".
    pub diagnosis: String,
    /// The code with the thing's shape: what becomes a type, what is
    /// deleted, what falls out.
    pub falls_out: String,
    /// The findings it explains, as indexes into [`Report::findings`].
    pub explains: Vec<usize>,
}

/// The model beneath several diagnoses, in one module or across several.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Root {
    /// The deepest model the diagnoses support, stated as what the thing is:
    /// "instruction selection is a phase".
    pub root: String,
    /// With that model in place: what stops being special, what is deleted,
    /// and what follows from that, to closure.
    pub falls_out: String,
    /// The diagnoses it explains, as indexes into [`Synthesis::diagnoses`].
    pub explains: Vec<usize>,
}

/// The frontier call's judgment of a review.
#[derive(Debug, Clone)]
pub struct Synthesis {
    /// The review a person reads, in Markdown.
    pub review: String,
    pub diagnoses: Vec<Diagnosis>,
    pub roots: Vec<Root>,
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

/// What the findings are about: the things each converging module's
/// symptoms concern, each described from first principles.
///
/// A module whose findings converge — at least three, from at least two
/// rules — gets one call at [`ModelLevel::Fast`](crate::model::ModelLevel::Fast),
/// shown the symptoms and the module's outline, that names the things they
/// are about in their domain's words — "an assembler" — each with the
/// symptoms that concern it. Each thing at least two symptoms concern gets
/// one more call at the same level that describes it from first principles,
/// shown only the thing's name and context, never the code, so it cannot
/// borrow the code's model of itself.
///
/// Every module, and every thing in one, is asked at once; the result keeps
/// module order, then the namer's. Every reply is held to a schema. A failed
/// call is a failure in the result and stops only its own module or thing.
///
/// # Errors
///
/// A module's files cannot be read or parsed.
pub async fn identify<A: Ask>(agent: &A, report: &Report) -> Result<Identified> {
    identify::identify(agent, report).await
}

/// The judgment: one call at
/// [`ModelLevel::Frontier`](crate::model::ModelLevel::Frontier) shown every
/// rule that found something, every [`Thing`] with its description, the
/// outlines of their modules (bodies elided, to a budget), and every finding
/// numbered. It diagnoses the things, finds the roots beneath the diagnoses
/// — across modules as well as within one — and writes the review: roots,
/// then diagnoses, then the findings by file, duplicates merged, trivial or
/// mistaken ones dropped and counted, recurring patterns called out. The
/// reply is held to a schema; a number past what it indexes names nothing.
///
/// `Ok(None)`, with no call made, when `report` has no findings.
///
/// # Errors
///
/// The call failed, its reply does not match the schema, a diagnosis names
/// a thing that does not exist, or a module's files cannot be read.
pub async fn synthesize<A: Ask>(
    agent: &A,
    rules: &[Rule],
    report: &Report,
) -> Result<Option<Synthesis>> {
    synthesize::synthesize(agent, rules, report).await
}
