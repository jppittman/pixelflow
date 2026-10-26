//! Deciding the calls.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};

use super::Call;
use super::module;
use crate::language::Language;
use crate::rule::{Group, Part, Rule, Surroundings, Unit};
use crate::snippet::{Snippet, snippets};

/// What makes two rules' calls about a file the same calls.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Seen {
    File,
    Outline,
    Parts(Part, Group),
}

fn seen(rule: &Rule) -> Seen {
    match &rule.unit {
        Unit::File => Seen::File,
        Unit::Outline => Seen::Outline,
        Unit::Parts { part, group, .. } => Seen::Parts(*part, *group),
    }
}

pub(super) fn plan(rules: Vec<Rule>, files: &[PathBuf]) -> Result<Vec<Call>> {
    let rules: Vec<Arc<Rule>> = rules.into_iter().map(Arc::new).collect();
    let mut calls = Vec::new();
    // A crate-wide call holds every file's parts, so it can only be built
    // once the last file is read; each rule gets its own call per crate.
    let mut crates: BTreeMap<(usize, PathBuf), Vec<(PathBuf, Snippet)>> = BTreeMap::new();
    for path in files {
        let Some(language) = Language::of(path) else {
            continue;
        };
        let source =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut shared: BTreeMap<(Seen, Surroundings), Vec<usize>> = BTreeMap::new();
        for (index, rule) in rules.iter().enumerate() {
            if rule.files.contains(path) {
                shared
                    .entry((seen(rule), rule.context))
                    .or_default()
                    .push(index);
            }
        }
        for ((seen, context), indexes) in shared {
            let snippets = snippets(&rules[indexes[0]], language, &source)
                .with_context(|| format!("parsing {}", path.display()))?;
            if snippets.is_empty() {
                continue;
            }
            if let Seen::Parts(_, Group::Crate) = seen {
                for &index in &indexes {
                    let gathered = crates.entry((index, module::crate_of(path))).or_default();
                    gathered.extend(snippets.iter().cloned().map(|s| (path.clone(), s)));
                }
                continue;
            }
            let root = match context {
                Surroundings::None => None,
                Surroundings::ModuleRoot | Surroundings::ModuleIntent => {
                    module::root_outline(path)?
                }
            };
            let intent = match context {
                Surroundings::ModuleIntent => module::intent(path)?,
                Surroundings::None | Surroundings::ModuleRoot => None,
            };
            let asking: Vec<_> = indexes.iter().map(|&i| rules[i].clone()).collect();
            calls.extend(snippets.into_iter().map(|snippet| Call {
                rules: asking.clone(),
                path: path.clone(),
                snippet,
                root: root.clone(),
                intent: intent.clone(),
            }));
        }
    }
    calls.extend(crates.into_iter().map(|((rule, krate), files)| Call {
        rules: vec![rules[rule].clone()],
        path: krate,
        snippet: under_headers(files),
        root: None,
        intent: None,
    }));
    Ok(calls)
}

/// Every file's snippet, each under a `== path ==` header.
fn under_headers(files: Vec<(PathBuf, Snippet)>) -> Snippet {
    let numbered = files
        .iter()
        .map(|(path, snippet)| format!("== {} ==\n{}", path.display(), snippet.numbered))
        .collect::<Vec<_>>()
        .join("\n");
    Snippet {
        first_line: 1,
        numbered,
    }
}
