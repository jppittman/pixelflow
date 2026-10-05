//! Rules: what to look at, how hard to think about it, and what to ask.
//!
//! A rule is `rules/<id>.json`:
//!
//! ```json
//! {
//!   "level": 1,
//!   "language": "rust",
//!   "query": "(function_item) @target",
//!   "skills": ["rust-idioms"],
//!   "prompt": "Flag ..."
//! }
//! ```
//!
//! `query` is a tree-sitter query whose `@target` captures are reviewed one at
//! a time; without one, the rule reviews each whole file in its language.
//! Everything is checked when the rule loads — a bad query or an unknown skill
//! fails there, not halfway through a review.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::language::Language;
use crate::model::ModelLevel;
use crate::skills::Skills;

/// The capture a rule's query must name: the code each review sees.
pub const TARGET_CAPTURE: &str = "target";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RuleFile {
    level: ModelLevel,
    language: Language,
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    skills: Vec<String>,
    prompt: String,
}

pub enum Target {
    WholeFile,
    Query {
        query: tree_sitter::Query,
        target: u32,
    },
}

pub struct Rule {
    pub id: String,
    pub level: ModelLevel,
    pub language: Language,
    pub target: Target,
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

        let target = match file.query {
            None => Target::WholeFile,
            Some(source) => {
                let query = tree_sitter::Query::new(&file.language.grammar(), &source)?;
                let Some(target) = query.capture_index_for_name(TARGET_CAPTURE) else {
                    bail!("query has no @{TARGET_CAPTURE} capture");
                };
                Target::Query { query, target }
            }
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
            language: file.language,
            target,
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
