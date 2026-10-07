//! Rules: a typed decision asked of each unit of code.
//!
//! A rule is `rules/<id>.json`:
//!
//! ```json
//! {
//!   "unit": "functions",
//!   "levels": { "decide": 1, "explain": 2 },
//!   "question": "Does this function nest its main path inside branches that only bail out?",
//!   "fine": { "flat": "Every case that exits is handled first; the main path is not indented." },
//!   "violations": { "nested": "The main path sits inside a branch whose other arm only exits." },
//!   "guidance": "Name the guard that should come first.",
//!   "skills": ["style-guide"]
//! }
//! ```
//!
//! The shape is Jac's meaning-typed `by llm` function: the question and the
//! outcomes' meanings are the signature, and the closed set of outcomes is
//! the return type. A model at `levels.decide` answers each unit with one
//! outcome. `unsure` is always an outcome too, and escalates the question a
//! level, up to `levels.explain`. A unit decided as one of the `violations`
//! is then shown to a model at `levels.explain`, which writes the findings:
//! where in the code, what, and the fix.
//!
//! - `unit` is what one call sees: `"file"`, `"outline"` (the file with every
//!   function body elided), `"functions"`, `"function_signatures"` or
//!   `"types"`. tree-sitter finds units; it does not judge them.
//! - `group` puts a file's units in one call each (`"each"`, the default),
//!   all of a file's in one call (`"file"`), or all of a crate's in one call
//!   (`"crate"`, each file under a `== path ==` header).
//! - `paths` and `exclude` are globs over paths relative to where the review
//!   runs (`pixelflow-*/**`, `**/tests/**`); no `paths` reads every file.
//! - `"context": "module_root"` also shows each call the outline of the
//!   reviewed file's module root.
//!
//! Everything is checked when the rule loads, not halfway through a review.
//!
//! [`load_dir`] reads a directory of rules into [`Rule`]s; everything else
//! here is the rule's shape, read by the rest of the crate.

mod file;
mod files;

use std::path::Path;

use anyhow::Result;
use globset::GlobSet;
use serde::Deserialize;

use crate::language::Language;
use crate::model::ModelLevel;
use crate::skills::Skills;

/// The capture a part's query names: the code each unit holds.
pub const TARGET_CAPTURE: &str = "target";

/// A capture a part's query may name inside its target: the target's text
/// stops where this one starts. A function's signature is its `@target` with
/// its body as `@cut`.
pub const CUT_CAPTURE: &str = "cut";

/// The outcome every decision may give besides its own: the code shown is
/// not enough to decide.
pub const UNSURE: &str = "unsure";

/// Every `*.json` rule in `dir`, sorted by id.
///
/// # Errors
///
/// The first rule that fails to load, naming its file: unreadable, not
/// valid JSON, an unknown field, a level outside 1-4, an explain level below
/// the decide level, no fine or no violation outcome, an outcome named
/// `unsure` or named twice, an unknown skill, a bad glob, or a `group` on a
/// unit that is already a whole file. No rule is returned if any fails.
pub fn load_dir(dir: &Path, skills: &Skills) -> Result<Vec<Rule>> {
    file::load_dir(dir, skills)
}

/// A loaded rule: everything checked, ready to plan calls from.
pub struct Rule {
    /// The rule file's name without `.json`.
    pub id: String,
    pub files: Files,
    pub unit: Unit,
    pub context: Surroundings,
    pub decision: Decision,
    pub levels: Levels,
    /// The text of the rule's skills, for both of its prompts.
    pub skills: String,
}

/// A closed-set question asked of every unit.
pub struct Decision {
    pub question: String,
    /// The fine outcomes first, then the violations, each in file order.
    pub outcomes: Vec<Outcome>,
    /// What a finding for a violation should say, beyond where and what.
    pub guidance: Option<String>,
}

pub struct Outcome {
    pub name: String,
    /// What choosing this outcome asserts about the code.
    pub meaning: String,
    pub verdict: Verdict,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Fine,
    Violation,
}

/// The model levels a rule's two questions are asked at; `explain` is never
/// below `decide`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Levels {
    pub decide: ModelLevel,
    pub explain: ModelLevel,
}

/// The files a rule reads, by path relative to where the review runs.
pub struct Files {
    /// `None` reads every file.
    pub(super) include: Option<GlobSet>,
    pub(super) exclude: GlobSet,
}

impl Files {
    /// Whether the rule reads `path`: it matches an include glob (or there
    /// are none) and no exclude glob. A leading `./` is ignored.
    #[must_use]
    pub fn contains(&self, path: &Path) -> bool {
        files::contains(self, path)
    }
}

/// What one call reviews.
pub enum Unit {
    /// A whole file.
    File,
    /// A whole file with every function body elided: its items and
    /// signatures, at their line numbers.
    Outline,
    /// The parts of a file, grouped into calls.
    Parts {
        part: Part,
        group: Group,
        /// One query per language that has the part.
        queries: Vec<CompiledQuery>,
    },
}

/// A query compiled for its language, with the indexes of its `@target`
/// and, if it has one, its `@cut`.
pub struct CompiledQuery {
    pub language: Language,
    pub query: tree_sitter::Query,
    pub target: u32,
    pub cut: Option<u32>,
}

/// A part of the code every language may have, found by a per-language query
/// (`Language::query`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Part {
    Functions,
    /// A function's signature — attributes, visibility, name, parameters,
    /// return type — without its body.
    FunctionSignatures,
    Types,
}

/// How a file's parts are grouped into calls.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Group {
    /// One call per part.
    #[default]
    Each,
    /// One call per file, holding all its parts.
    File,
    /// One call per crate, holding every part from every file in it, each
    /// file under a `== path ==` header: for patterns only visible across a
    /// whole crate, such as the shapes of all its function signatures.
    Crate,
}

/// What a call sees besides the code under review.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Surroundings {
    /// Nothing else.
    #[default]
    None,
    /// The outline of the root file of the module the reviewed file belongs
    /// to (its `mod.rs`, `lib.rs` or `main.rs`): the contract an
    /// implementation file is judged against.
    ModuleRoot,
}
