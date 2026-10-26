//! What a module is on disk: a directory with a root file, an `intent.md`
//! somewhere above it, and a crate boundary at the nearest `Cargo.toml`.

use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::Source;
use crate::language::Language;
use crate::snippet::{Snippet, outline};

/// File names that make a file its directory's module root.
const ROOT_FILES: [&str; 3] = ["mod.rs", "lib.rs", "main.rs"];

/// The root files `dir` holds, in `ROOT_FILES` order.
fn root_files(dir: &Path) -> Vec<PathBuf> {
    ROOT_FILES
        .iter()
        .map(|name| dir.join(name))
        .filter(|root| root.is_file())
        .collect()
}

/// `path` as its outline; `None` if it is in no known language.
fn outlined(path: &Path) -> Result<Option<Snippet>> {
    let Some(language) = Language::of(path) else {
        return Ok(None);
    };
    let source =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let outlined =
        outline(language, &source).with_context(|| format!("parsing {}", path.display()))?;
    Ok(Some(outlined))
}

/// The outline of the root file of the module `path` belongs to, unless
/// `path` is one. `foo/bar.rs` belongs to the module whose root is
/// `foo/mod.rs`, or `foo.rs` in the 2018 layout: its own directory, no
/// further up.
pub(super) fn root_outline(path: &Path) -> Result<Option<Source>> {
    let is_root = path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| ROOT_FILES.contains(&n));
    if is_root {
        return Ok(None);
    }
    let Some(dir) = path.parent() else {
        return Ok(None);
    };
    let sibling = Some(dir.with_extension("rs")).filter(|root| root.is_file());
    let Some(root) = root_files(dir).into_iter().next().or(sibling) else {
        return Ok(None);
    };
    Ok(outlined(&root)?.map(|outline| Source {
        path: root,
        text: outline.numbered,
    }))
}

/// The nearest `intent.md` from `path`'s directory up to its crate root (the
/// first directory holding a `Cargo.toml`), numbered as a snippet is. A
/// module without one, or a file outside any crate, gets none.
pub(super) fn intent(path: &Path) -> Result<Option<Source>> {
    let mut dir = path.parent();
    while let Some(d) = dir {
        let intent = d.join("intent.md");
        if intent.is_file() {
            let text = std::fs::read_to_string(&intent)
                .with_context(|| format!("reading {}", intent.display()))?;
            let numbered = text
                .lines()
                .enumerate()
                .map(|(i, line)| format!("{:>5} | {line}\n", i + 1))
                .collect();
            return Ok(Some(Source {
                path: intent,
                text: numbered,
            }));
        }
        if d.join("Cargo.toml").is_file() {
            return Ok(None);
        }
        dir = d.parent();
    }
    Ok(None)
}

/// The directory of the nearest `Cargo.toml` above `path`; the current
/// directory if there is none.
pub(super) fn crate_of(path: &Path) -> PathBuf {
    path.ancestors()
        .skip(1)
        .find(|dir| dir.join("Cargo.toml").is_file())
        .map_or_else(PathBuf::new, Path::to_path_buf)
}

/// The module findings in `path` cluster into: the nearest directory, from
/// its own, that holds a root file, so a nested directory without one
/// belongs to its parent's module; a crate-wide finding's path is its crate
/// already. Not [`root_outline`]'s walk, which looks no further than the
/// file's own directory.
pub(super) fn containing(path: &Path) -> PathBuf {
    if path.is_dir() {
        return path.to_path_buf();
    }
    path.ancestors()
        .skip(1)
        .find(|dir| !root_files(dir).is_empty())
        .or_else(|| path.parent())
        .map_or_else(PathBuf::new, Path::to_path_buf)
}

/// The header of a file or module left out of an outline for want of budget.
pub(super) fn omitted(path: &Path) -> String {
    format!("== {} == (omitted: over budget)\n", path.display())
}

/// The outline of the module at `module`: its root files first, then the
/// files `symptoms` are in, most symptoms first, each under a `== path ==`
/// header, to at most `budget` lines (the first file is shown whole).
/// Returns the lines used.
pub(super) fn shape<'p>(
    module: &Path,
    symptoms: impl Iterator<Item = &'p Path>,
    budget: usize,
) -> Result<(String, usize)> {
    let mut by_file: BTreeMap<&Path, usize> = BTreeMap::new();
    for path in symptoms {
        *by_file.entry(path).or_default() += 1;
    }
    let mut roots = root_files(module);
    // Path order (`lib.rs`, `main.rs`, `mod.rs`), not the order the names are
    // listed in: the sort this replaced was by path, and journals key on it.
    roots.sort();
    let mut ranked: Vec<(&Path, usize)> = by_file
        .into_iter()
        .filter(|(path, _)| path.is_file())
        .collect();
    ranked.sort_by_key(|&(_, n)| Reverse(n));
    let mut files: Vec<PathBuf> = roots
        .into_iter()
        .chain(ranked.into_iter().map(|(path, _)| path.to_path_buf()))
        .collect();
    // Only neighbours: journals key on this exact prompt, so a root with
    // symptoms that is not the most symptomatic file stays outlined twice
    // (intent.md, Behind).
    files.dedup();

    let mut shape = String::new();
    let mut lines = 0;
    for path in files {
        let Some(outlined) = outlined(&path)? else {
            continue;
        };
        let length = outlined.numbered.lines().count();
        if lines + length > budget && lines > 0 {
            shape.push_str(&omitted(&path));
            continue;
        }
        lines += length;
        shape.push_str(&format!(
            "== {} ==\n{}\n",
            path.display(),
            outlined.numbered
        ));
    }
    Ok((shape, lines))
}
