//! Deciding the calls.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::{Call, Source};
use crate::language::Language;
use crate::rule::{Review, Rule, Scope, Surroundings};
use crate::snippet::{Snippet, snippets};

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
    // Crate-wide rules gather each file's captures here, keyed by rule and
    // crate, and become one call per crate once every file is read.
    let mut crates: BTreeMap<(usize, PathBuf), Vec<(PathBuf, Snippet)>> = BTreeMap::new();
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
            if is_crate_wide(rule) {
                let gathered = crates.entry((index, crate_of(path))).or_default();
                gathered.extend(snippets.into_iter().map(|s| (path.clone(), s)));
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
    calls.extend(crates.into_iter().map(|((rule, krate), files)| Call {
        rule,
        path: krate,
        snippet: under_headers(files),
        root: None,
    }));
    Ok(calls)
}

fn is_crate_wide(rule: &Rule) -> bool {
    matches!(
        rule.scope,
        Scope::Captures {
            review: Review::Crate,
            ..
        }
    )
}

/// The directory of the nearest `Cargo.toml` above `path`; the current
/// directory if there is none.
fn crate_of(path: &Path) -> PathBuf {
    path.ancestors()
        .skip(1)
        .find(|dir| dir.join("Cargo.toml").is_file())
        .map_or_else(PathBuf::new, Path::to_path_buf)
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
