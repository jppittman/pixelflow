//! Deciding the calls.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::{Call, Source};
use crate::language::Language;
use crate::rule::{Rule, Surroundings};
use crate::snippet::snippets;

/// File names that make a file its directory's module root.
const ROOT_FILES: [&str; 3] = ["mod.rs", "lib.rs", "main.rs"];

/// The root file of the module `path` belongs to, unless `path` is one.
fn module_root(path: &Path) -> Result<Option<Source>> {
    if path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| ROOT_FILES.contains(&n))
    {
        return Ok(None);
    }
    let Some(dir) = path.parent() else {
        return Ok(None);
    };
    // `foo/bar.rs` belongs to `foo/mod.rs`, or to `foo.rs` in the 2018 layout.
    let candidates = ROOT_FILES
        .iter()
        .map(|name| dir.join(name))
        .chain(std::iter::once(dir.with_extension("rs")));
    for root in candidates {
        if root.is_file() {
            let text = std::fs::read_to_string(&root)
                .with_context(|| format!("reading {}", root.display()))?;
            return Ok(Some(Source { path: root, text }));
        }
    }
    Ok(None)
}

pub(super) fn plan(rules: &[Rule], files: &[PathBuf]) -> Result<Vec<Call>> {
    let mut calls = Vec::new();
    for path in files {
        let Some(language) = Language::of(path) else {
            continue;
        };
        let source =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        for (index, rule) in rules
            .iter()
            .enumerate()
            .filter(|(_, r)| r.files.contains(path))
        {
            let snippets = snippets(rule, language, &source)
                .with_context(|| format!("parsing {}", path.display()))?;
            if snippets.is_empty() {
                continue;
            }
            let root = match rule.context {
                Surroundings::None => None,
                Surroundings::ModuleRoot => module_root(path)?,
            };
            calls.extend(snippets.into_iter().map(|snippet| Call {
                rule: index,
                path: path.clone(),
                snippet,
                root: root.clone(),
            }));
        }
    }
    Ok(calls)
}
