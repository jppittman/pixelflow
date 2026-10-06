//! Reading a rule file into a [`Rule`](super::Rule).

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use super::{CompiledQuery, Part, Review, Rule, Scope, Surroundings, TARGET_CAPTURE, files};
use crate::language::Language;
use crate::model::ModelLevel;
use crate::skills::Skills;

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

fn compile(language: Language, source: &str) -> Result<CompiledQuery> {
    let query = tree_sitter::Query::new(&language.grammar(), source)?;
    let Some(target) = query.capture_index_for_name(TARGET_CAPTURE) else {
        bail!("query has no @{TARGET_CAPTURE} capture");
    };
    Ok(CompiledQuery {
        language,
        query,
        target,
    })
}

fn load(path: &Path, skills: &Skills) -> Result<Rule> {
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
                .filter_map(|l| l.query(part).map(|q| compile(l, q)))
                .collect::<Result<Vec<_>>>()?,
        ),
        ScopeFile::Query { language, query } => Some(vec![compile(language, &query)?]),
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

    Ok(Rule {
        id,
        level: file.level,
        files: files::new(&file.paths, &file.exclude)?,
        scope,
        context: file.context,
        instructions,
    })
}

pub(super) fn load_dir(dir: &Path, skills: &Skills) -> Result<Vec<Rule>> {
    let mut rules = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "json") {
            rules.push(
                load(&path, skills).with_context(|| format!("loading rule {}", path.display()))?,
            );
        }
    }
    rules.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(rules)
}
