//! Reading a skills directory.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};

use super::{SKILL_FILE, Skills};

pub(super) fn load(dir: &Path) -> Result<Skills> {
    let mut skills = BTreeMap::new();
    if !dir.exists() {
        return Ok(Skills(skills));
    }
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let skill_dir = entry?.path();
        let file = skill_dir.join(SKILL_FILE);
        if !file.is_file() {
            continue;
        }
        let name = skill_dir
            .file_name()
            .and_then(|n| n.to_str())
            .with_context(|| format!("skill directory {} is not UTF-8", skill_dir.display()))?
            .to_owned();
        let text = std::fs::read_to_string(&file)
            .with_context(|| format!("reading {}", file.display()))?;
        skills.insert(name, text);
    }
    Ok(Skills(skills))
}
