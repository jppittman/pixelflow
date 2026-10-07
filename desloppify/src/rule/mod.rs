//! Rules: what code to look at, how hard to think about it, and what to ask.
//!
//! A rule is `rules/<id>.json`:
//!
//! ```json
//! {
//!   "level": 2,
//!   "scope": "function_names",
//!   "review": "together",
//!   "skills": ["style-guide"],
//!   "prompt": "Flag ..."
//! }
//! ```
//!
//! `scope` is the rule's input: `"file"`, a named part of the code
//! (`"functions"`, `"function_names"`, `"function_bodies"`, `"types"`,
//! `"comments"`), or `{"language": "rust", "query": "... @target"}` for
//! anything else. A named part applies to every language that has it; a query
//! to its own language. `paths` and `exclude` are globs over file paths
//! relative to where the review runs (`pixelflow-*/**`, `**/tests/**`); a
//! rule with no `paths` reads every file. `"context": "module_root"` also
//! shows each call the root file of the reviewed file's module. `review`
//! says whether each captured part is reviewed on its own (`"each"`, the default) or all of a file's are reviewed in one
//! call (`"together"`) — the way to ask about consistency across them — or
//! the whole file is reviewed, pointed at the captured lines, if it has any
//! (`"file"`) — for a match that needs its surroundings to be judged.
//!
//! Everything is checked when the rule loads: a bad query or an unknown skill
//! fails there, not halfway through a review.

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

/// The capture a query must name: the code each review sees.
pub const TARGET_CAPTURE: &str = "target";

/// A capture a query may name inside its target: the target's text stops
/// where this one starts. A function's signature is its `@target` with its
/// body as `@cut`.
pub const CUT_CAPTURE: &str = "cut";

/// Every `*.json` rule in `dir`, sorted by id.
///
/// # Errors
///
/// The first rule that fails to load, naming its file: unreadable, not
/// valid JSON, an unknown field, a level outside 1-4, a query that does not
/// compile or has no `@target`, an unknown skill, a bad glob, or a `review`
/// on a `"file"` scope. No rule is returned if any fails.
pub fn load_dir(dir: &Path, skills: &Skills) -> Result<Vec<Rule>> {
    file::load_dir(dir, skills)
}

/// A loaded rule: everything checked, ready to plan calls from.
pub struct Rule {
    /// The rule file's name without `.json`.
    pub id: String,
    pub level: ModelLevel,
    pub files: Files,
    pub scope: Scope,
    pub context: Surroundings,
    /// The skills' text followed by the rule's prompt.
    pub instructions: String,
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

pub enum Scope {
    WholeFile,
    /// One query per language the rule applies to.
    Captures {
        queries: Vec<CompiledQuery>,
        review: Review,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Part {
    Functions,
    FunctionNames,
    /// A function's signature — attributes, visibility, name, parameters,
    /// return type — without its body.
    FunctionSignatures,
    FunctionBodies,
    Types,
    Comments,
}

/// How a file's captures are grouped into calls.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Review {
    /// One call per capture.
    #[default]
    Each,
    /// One call per file, holding all its captures.
    Together,
    /// One call per function that holds a capture, showing the whole
    /// function and naming the captured lines: the query picks where to
    /// look, the function is the context to judge it in. A capture outside
    /// any function is shown alone.
    Function,
    /// One call per file that has a capture, holding the whole file and
    /// naming the captured lines, for a match that needs more than its
    /// function to be judged.
    File,
    /// One call per crate, holding every capture from every file in it, each
    /// file under a `== path ==` header: for patterns only visible across a
    /// whole crate, such as the shapes of all its function signatures.
    Crate,
}

/// What a call sees besides the code under review.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Surroundings {
    /// Nothing else.
    #[default]
    None,
    /// The root file of the module the reviewed file belongs to (its
    /// `mod.rs`, `lib.rs` or `main.rs`): the contract an implementation file
    /// is judged against.
    ModuleRoot,
}
