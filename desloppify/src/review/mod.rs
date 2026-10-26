//! A review: every rule's decision about every unit it applies to, the
//! violations explained, the things they converge on named, and one
//! frontier call's judgment over all of it.
//!
//! [`plan`] decides the calls without making any. A [`Session`] is who a
//! review asks and how sure System One must be. [`Session::review`] makes the
//! calls in System One's cascade: each call's unit goes to a [`Decide`] with
//! every applicable rule's question at once; an answer at or above the
//! session's minimum confidence settles that rule, and the rest are asked of
//! a model through [`Ask`] at the rule's decide level, escalating a level on
//! `unsure`. A unit decided as a violation is shown to a model at the rule's
//! explain level, which writes the findings.
//! [`Session::identify`] names what the findings in each converging module
//! are about, and describes each thing without the code.
//! [`Session::synthesize`] has one frontier call read all of it, diagnose,
//! find the roots and write the review a person reads.
//!
//! Small models make many calls, each seeing one rule or one module, so none
//! forgets a rule; the one large call is the only one that sees everything.

mod identify;
mod module;
mod numbering;
mod plan;
mod reply;
mod run;
mod synthesize;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Result, ensure};
use serde::Serialize;

use crate::agent::{Ask, Usage};
use crate::decide::Decide;
use crate::rule::Rule;
use crate::snippet::Snippet;

/// One unit of one file, and the rules that ask about it.
pub struct Call {
    /// The rules that ask about this unit; they see the same unit and
    /// context.
    pub(crate) rules: Vec<Arc<Rule>>,
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

/// A position in [`Report::findings`]: how a [`Thing`] or [`Diagnosis`]
/// names a finding. Only code in `review` can make one, from a number a
/// reply echoed that names an item of the list it counts; it is written
/// to `report.json` as that position, from 0.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(transparent)]
pub struct FindingIx(usize);

/// A position in [`Report::things`]: how a [`Diagnosis`] names its thing.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(transparent)]
pub struct ThingIx(usize);

/// A position in [`Synthesis::diagnoses`]: how a [`Root`] names a diagnosis.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(transparent)]
pub struct DiagnosisIx(usize);

/// Every call a review of `files` under `rules` makes, in file order. Rules
/// that apply to a file and see the same unit and context share its calls;
/// a crate-wide rule gets one call per crate, its files under `== path ==`
/// headers. Files in no known language, and rules whose `Files` exclude a
/// file, plan nothing.
///
/// # Errors
///
/// A file or module root cannot be read, or a file cannot be parsed.
pub fn plan(rules: Vec<Rule>, files: &[PathBuf]) -> Result<Vec<Call>> {
    plan::plan(rules, files)
}

/// How sure System One must be to settle a question, unless a session is
/// told otherwise.
pub const DEFAULT_MIN_CONFIDENCE: f64 = 0.7;

/// Who a review asks, and how sure System One must be to settle a question.
pub struct Session<A, D> {
    pub ask: A,
    pub decide: D,
    /// An answer below this confidence goes to the model.
    min_confidence: f64,
}

impl<A, D> Session<A, D> {
    /// A session asking `ask` and, first, `decide`, settling at
    /// [`DEFAULT_MIN_CONFIDENCE`].
    #[must_use]
    pub fn new(ask: A, decide: D) -> Self {
        Self {
            ask,
            decide,
            min_confidence: DEFAULT_MIN_CONFIDENCE,
        }
    }

    /// The same session, with System One settling a question only at
    /// `min_confidence` or above.
    ///
    /// # Errors
    ///
    /// `min_confidence` is not between 0 and 1.
    pub fn settling_at(self, min_confidence: f64) -> Result<Self> {
        ensure!(
            (0.0..=1.0).contains(&min_confidence),
            "a minimum confidence is between 0 and 1, not {min_confidence}"
        );
        Ok(Self {
            min_confidence,
            ..self
        })
    }
}

impl<A: Ask, D: Decide> Session<A, D> {
    /// Makes every call in `calls` and gathers the decisions and findings.
    ///
    /// A rule's model prompts hold its question, its outcomes' meanings and
    /// its skills; the user prompt names the file, holds the snippet, then
    /// any module root, marked as context only. Every reply is held to a JSON
    /// schema by the backend; one that does not match is a failure, as is a
    /// call that errs. One failing does not stop the others.
    ///
    /// # Errors
    ///
    /// A call's task panicked.
    pub async fn review(self: Arc<Self>, calls: Vec<Call>) -> Result<Report> {
        run::review(self, calls).await
    }
}

impl<A: Ask, D> Session<A, D> {
    /// What the findings are about: the things each converging module's
    /// symptoms concern, each described from first principles, set as
    /// `report`'s things; their failures join `report`'s. Returns the tokens
    /// every call used.
    ///
    /// A module whose findings converge — at least three, from at least two
    /// rules — gets one call at
    /// [`ModelLevel::Fast`](crate::model::ModelLevel::Fast), shown the
    /// symptoms and the module's outline, that names the things they are
    /// about in their domain's words — "an assembler" — each with the
    /// symptoms that concern it. Each thing at least two symptoms concern
    /// gets one more call at the same level that describes it from first
    /// principles, shown only the thing's name and context, never the code,
    /// so it cannot borrow the code's model of itself.
    ///
    /// Every module, and every thing in one, is asked at once; the things
    /// keep module order, then the namer's. Every reply is held to a schema.
    /// A failed call is a failure in `report` and stops only its own module
    /// or thing. So does a symptom number that names nothing in its module's
    /// list: the whole naming fails (its tokens are counted), not a shorter
    /// list of symptoms. Asking again replaces the things and adds the
    /// failures again.
    ///
    /// # Errors
    ///
    /// A module's files cannot be read or parsed.
    pub async fn identify(&self, report: &mut Report) -> Result<Usage> {
        identify::identify(&self.ask, report).await
    }

    /// The judgment: one call at
    /// [`ModelLevel::Frontier`](crate::model::ModelLevel::Frontier) shown
    /// every rule that found something, every [`Thing`] with its
    /// description, the outlines of their modules (bodies elided, to a
    /// budget), and every finding numbered. It diagnoses the things, finds
    /// the roots beneath the diagnoses — across modules as well as within
    /// one — and writes the review: roots, then diagnoses, then the findings
    /// by file, duplicates merged, trivial or mistaken ones dropped and
    /// counted, recurring patterns called out. The reply is held to a
    /// schema, and a number in it that names nothing is an error, not a
    /// shorter list.
    ///
    /// `Ok(None)`, with no call made, when `report` has no findings.
    ///
    /// # Errors
    ///
    /// The call failed, its reply does not match the schema, a number in it
    /// names nothing in the list it counts, or a module's files cannot be
    /// read. The call's tokens are then in no usage the caller can read.
    pub async fn synthesize(&self, report: &Report) -> Result<Option<Synthesis>> {
        synthesize::synthesize(&self.ask, report).await
    }
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
    /// The rules that asked a question, by id.
    rules: BTreeMap<String, Arc<Rule>>,
    /// Sorted by path, then line, then rule, then outcome, then message,
    /// when [`Session::review`] returns; never changed after.
    findings: Vec<Finding>,
    /// What the findings in each converging module are about, once
    /// [`Session::identify`] has been asked; [`Session::synthesize`]
    /// diagnoses these.
    things: Vec<Thing>,
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
}

impl Report {
    /// Every finding, in report order.
    #[must_use]
    pub fn findings(&self) -> &[Finding] {
        &self.findings
    }

    /// The finding at `ix`, which a [`Thing`] or [`Diagnosis`] of this report
    /// holds.
    ///
    /// # Panics
    ///
    /// `ix` came from another report and is past this one's findings.
    #[must_use]
    pub fn finding(&self, ix: FindingIx) -> &Finding {
        &self.findings[ix.0]
    }

    /// What the findings are about, once [`Session::identify`] has been
    /// asked.
    #[must_use]
    pub fn things(&self) -> &[Thing] {
        &self.things
    }

    /// The rules that asked a question, by id: a rule no file applied to made
    /// no decision and is not here.
    pub fn rules(&self) -> impl Iterator<Item = &Rule> {
        self.rules.values().map(AsRef::as_ref)
    }
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
    /// The findings that concern it.
    pub explains: Vec<FindingIx>,
}

/// The model of a thing the code is missing or has wrong.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Diagnosis {
    /// The thing.
    pub thing: ThingIx,
    /// The model, stated as what the thing is: "a label is an operand".
    pub diagnosis: String,
    /// The code with the thing's shape: what becomes a type, what is
    /// deleted, what falls out.
    pub falls_out: String,
    /// The findings it explains.
    pub explains: Vec<FindingIx>,
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
    /// The diagnoses it explains.
    pub explains: Vec<DiagnosisIx>,
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
