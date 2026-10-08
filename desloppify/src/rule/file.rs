//! Reading a rule file into a [`Rule`](super::Rule).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

use super::{
    CUT_CAPTURE, CompiledQuery, Decision, Group, Levels, Outcome, Part, Rule, Surroundings,
    TARGET_CAPTURE, UNSURE, Unit, Verdict, files,
};
use crate::language::Language;
use crate::skills::Skills;

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum UnitFile {
    File,
    Outline,
    #[serde(untagged)]
    Part(Part),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RuleFile {
    unit: UnitFile,
    #[serde(default)]
    group: Option<Group>,
    levels: Levels,
    question: String,
    fine: BTreeMap<String, String>,
    violations: BTreeMap<String, String>,
    #[serde(default)]
    guidance: Option<String>,
    #[serde(default)]
    paths: Vec<String>,
    #[serde(default)]
    exclude: Vec<String>,
    #[serde(default)]
    context: Surroundings,
    #[serde(default)]
    skills: Vec<String>,
}

fn compile(language: Language, source: &str) -> Result<CompiledQuery> {
    let query = tree_sitter::Query::new(&language.grammar(), source)?;
    let Some(target) = query.capture_index_for_name(TARGET_CAPTURE) else {
        bail!("query has no @{TARGET_CAPTURE} capture");
    };
    let cut = query.capture_index_for_name(CUT_CAPTURE);
    Ok(CompiledQuery {
        language,
        query,
        target,
        cut,
    })
}

fn unit(unit: UnitFile, group: Option<Group>) -> Result<Unit> {
    let part = match unit {
        UnitFile::File | UnitFile::Outline if group.is_some() => {
            bail!("`group` needs a unit that is part of a file; a file is already one call")
        }
        UnitFile::File => return Ok(Unit::File),
        UnitFile::Outline => return Ok(Unit::Outline),
        UnitFile::Part(part) => part,
    };
    let queries = Language::ALL
        .into_iter()
        .filter_map(|l| l.query(part).map(|q| compile(l, q)))
        .collect::<Result<Vec<_>>>()?;
    Ok(Unit::Parts {
        part,
        group: group.unwrap_or_default(),
        queries,
    })
}

fn decision(file: &RuleFile) -> Result<Decision> {
    ensure!(
        !file.fine.is_empty(),
        "a decision needs at least one fine outcome"
    );
    ensure!(
        !file.violations.is_empty(),
        "a decision needs at least one violation outcome"
    );
    let mut seen = BTreeSet::new();
    let fine = file.fine.iter().map(|o| (o, Verdict::Fine));
    let violations = file.violations.iter().map(|o| (o, Verdict::Violation));
    let mut outcomes = Vec::new();
    for ((name, meaning), verdict) in fine.chain(violations) {
        ensure!(name != UNSURE, "`{UNSURE}` is every decision's own outcome");
        ensure!(seen.insert(name), "outcome `{name}` is named twice");
        outcomes.push(Outcome {
            name: name.clone(),
            meaning: meaning.clone(),
            verdict,
        });
    }
    Ok(Decision {
        question: file.question.clone(),
        outcomes,
        guidance: file.guidance.clone(),
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
    ensure!(
        file.levels.explain >= file.levels.decide,
        "levels.explain is below levels.decide"
    );

    let mut skill_text = String::new();
    for name in &file.skills {
        let Some(skill) = skills.get(name) else {
            bail!("unknown skill `{name}`");
        };
        skill_text.push_str(skill);
        skill_text.push_str("\n\n");
    }

    Ok(Rule {
        id,
        files: files::new(&file.paths, &file.exclude)?,
        decision: decision(&file)?,
        unit: unit(file.unit, file.group)?,
        context: file.context,
        levels: file.levels,
        skills: skill_text,
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
