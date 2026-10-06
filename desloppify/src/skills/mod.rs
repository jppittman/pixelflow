//! Skills: shared review knowledge a rule pulls into its prompt.
//!
//! A skill is `skills/<name>/SKILL.md`. Rules name the skills they need, and
//! each named skill's text is placed ahead of the rule's prompt.

mod load;

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;

/// The file a skill's directory must hold to be a skill.
pub const SKILL_FILE: &str = "SKILL.md";

/// Skills by name: a directory's name, mapped to its `SKILL.md` text.
#[derive(Debug, Default)]
pub struct Skills(pub(super) BTreeMap<String, String>);

impl Skills {
    /// The text of the skill called `name`, verbatim.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&str> {
        self.0.get(name).map(String::as_str)
    }
}

/// Every skill under `dir`: each subdirectory holding a [`SKILL_FILE`].
/// Subdirectories without one are not skills and are skipped; a missing `dir`
/// is no skills, not an error.
///
/// # Errors
///
/// `dir` or a skill file exists but cannot be read, or a skill directory's
/// name is not UTF-8.
pub fn load(dir: &Path) -> Result<Skills> {
    load::load(dir)
}
