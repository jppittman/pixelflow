//! Path filtering by include and exclude globs.

use std::path::{Component, Path, PathBuf};

use anyhow::Result;
use globset::{GlobBuilder, GlobSet, GlobSetBuilder};

use super::Files;

pub(super) fn new(include: &[String], exclude: &[String]) -> Result<Files> {
    let include = match include {
        [] => None,
        globs => Some(glob_set(globs)?),
    };
    Ok(Files {
        include,
        exclude: glob_set(exclude)?,
    })
}

pub(super) fn contains(files: &Files, path: &Path) -> bool {
    // `./src/a.rs` and `src/a.rs` are one file to a glob's author.
    let path: PathBuf = path
        .components()
        .filter(|c| *c != Component::CurDir)
        .collect();
    let included = files.include.as_ref().is_none_or(|set| set.is_match(&path));
    included && !files.exclude.is_match(&path)
}

fn glob_set(globs: &[String]) -> Result<GlobSet> {
    let mut set = GlobSetBuilder::new();
    for glob in globs {
        set.add(GlobBuilder::new(glob).literal_separator(true).build()?);
    }
    Ok(set.build()?)
}
