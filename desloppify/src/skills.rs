//! Skills: shared review knowledge a rule pulls into its prompt.
//!
//! A skill is `skills/<name>/SKILL.md`. Rules name the skills they need, and
//! each named skill's text is placed ahead of the rule's prompt.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};

pub const SKILL_FILE: &str = "SKILL.md";

#[derive(Debug, Default)]
pub struct Skills(BTreeMap<String, String>);

impl Skills {
    /// Every skill under `dir`. A missing directory is no skills, not an error.
    pub fn load(dir: &Path) -> Result<Self> {
        let mut skills = BTreeMap::new();
        if !dir.exists() {
            return Ok(Self(skills));
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
        Ok(Self(skills))
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<&str> {
        self.0.get(name).map(String::as_str)
    }
}
