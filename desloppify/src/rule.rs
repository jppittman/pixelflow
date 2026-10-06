//! Rules: what code to look at, how hard to think about it, and what to ask.
//!
//! A rule is `rules/<id>.json`:
//!
//! ```json
//! {
//!   "level": 2,
//!   "scope": "function_names",
//!   "review": "together",
//!   "skills": ["rust-idioms"],
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
//! shows each call the root file of the reviewed file's module. `review` says whether each captured part is reviewed
//! on its own (`"each"`, the default) or all of a file's are reviewed in one
//! call (`"together"`) — the way to ask about consistency across them — or
//! the whole file is reviewed, pointed at the captured lines, if it has any
//! (`"file"`) — for a match that needs its surroundings to be judged.
//!
//! Everything is checked when the rule loads: a bad query or an unknown skill
//! fails there, not halfway through a review.

use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail};
use globset::{GlobBuilder, GlobSet, GlobSetBuilder};
use serde::Deserialize;

use crate::language::Language;
use crate::model::ModelLevel;
use crate::skills::Skills;

/// The capture a query must name: the code each review sees.
pub const TARGET_CAPTURE: &str = "target";

/// A part of the code every language may have, found by a per-language query
/// (`Language::query`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Part {
    Functions,
    FunctionNames,
    FunctionBodies,
    Types,
    Comments,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ScopeFile {
    Named(NamedScope),
    Query { language: Language, query: String },
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum NamedScope {
    File,
    #[serde(untagged)]
    Part(Part),
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
    /// One call per file that has a capture, holding the whole file and
    /// naming the captured lines: the query picks where to look, the file is
    /// the context to judge it in.
    File,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RuleFile {
    level: ModelLevel,
    scope: ScopeFile,
    #[serde(default)]
    review: Option<Review>,
    #[serde(default)]
    paths: Vec<String>,
    #[serde(default)]
    exclude: Vec<String>,
    #[serde(default)]
    context: Surroundings,
    #[serde(default)]
    skills: Vec<String>,
    prompt: String,
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

/// The files a rule reads, by path relative to where the review runs.
pub struct Files {
    /// `None` reads every file.
    include: Option<GlobSet>,
    exclude: GlobSet,
}

impl Files {
    fn new(include: &[String], exclude: &[String]) -> Result<Self> {
        let include = match include {
            [] => None,
            globs => Some(glob_set(globs)?),
        };
        Ok(Self {
            include,
            exclude: glob_set(exclude)?,
        })
    }

    #[must_use]
    pub fn contains(&self, path: &Path) -> bool {
        // `./src/a.rs` and `src/a.rs` are one file to a glob's author.
        let path: PathBuf = path
            .components()
            .filter(|c| *c != Component::CurDir)
            .collect();
        let included = self.include.as_ref().is_none_or(|set| set.is_match(&path));
        included && !self.exclude.is_match(&path)
    }
}

fn glob_set(globs: &[String]) -> Result<GlobSet> {
    let mut set = GlobSetBuilder::new();
    for glob in globs {
        set.add(GlobBuilder::new(glob).literal_separator(true).build()?);
    }
    Ok(set.build()?)
}

pub struct CompiledQuery {
    pub language: Language,
    pub query: tree_sitter::Query,
    pub target: u32,
}

impl CompiledQuery {
    fn new(language: Language, source: &str) -> Result<Self> {
        let query = tree_sitter::Query::new(&language.grammar(), source)?;
        let Some(target) = query.capture_index_for_name(TARGET_CAPTURE) else {
            bail!("query has no @{TARGET_CAPTURE} capture");
        };
        Ok(Self {
            language,
            query,
            target,
        })
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

pub struct Rule {
    pub id: String,
    pub level: ModelLevel,
    pub files: Files,
    pub scope: Scope,
    pub context: Surroundings,
    /// The skills' text followed by the rule's prompt.
    pub instructions: String,
}

impl Rule {
    fn load(path: &Path, skills: &Skills) -> Result<Self> {
        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .with_context(|| format!("rule file name {} is not UTF-8", path.display()))?
            .to_owned();
        let text = std::fs::read_to_string(path)?;
        let file: RuleFile = serde_json::from_str(&text)?;

        let queries = match file.scope {
            ScopeFile::Named(NamedScope::File) => {
                if file.review.is_some() {
                    bail!("`review` needs a scope that captures parts; a file is already one");
                }
                None
            }
            ScopeFile::Named(NamedScope::Part(part)) => Some(
                Language::ALL
                    .into_iter()
                    .filter_map(|l| l.query(part).map(|q| CompiledQuery::new(l, q)))
                    .collect::<Result<Vec<_>>>()?,
            ),
            ScopeFile::Query { language, query } => {
                Some(vec![CompiledQuery::new(language, &query)?])
            }
        };
        let scope = match queries {
            None => Scope::WholeFile,
            Some(queries) => Scope::Captures {
                queries,
                review: file.review.unwrap_or_default(),
            },
        };

        let mut instructions = String::new();
        for name in &file.skills {
            let Some(skill) = skills.get(name) else {
                bail!("unknown skill `{name}`");
            };
            instructions.push_str(skill);
            instructions.push_str("\n\n");
        }
        instructions.push_str(&file.prompt);

        Ok(Self {
            id,
            level: file.level,
            files: Files::new(&file.paths, &file.exclude)?,
            scope,
            context: file.context,
            instructions,
        })
    }

    /// Every `*.json` rule in `dir`, sorted by id.
    pub fn load_dir(dir: &Path, skills: &Skills) -> Result<Vec<Self>> {
        let mut rules = Vec::new();
        for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
            let path = entry?.path();
            if path.extension().is_some_and(|e| e == "json") {
                rules.push(
                    Self::load(&path, skills)
                        .with_context(|| format!("loading rule {}", path.display()))?,
                );
            }
        }
        rules.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(rules)
    }
}
