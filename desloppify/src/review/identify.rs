//! From symptoms to the things they are about: name what each converging
//! module is, and describe each thing without the code. Many small calls,
//! every module at once; the judgment is the synthesis's.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use futures::future::join_all;
use serde::Deserialize;

use super::plan::ROOT_FILES;
use super::{Finding, Identified, Report, Thing};
use crate::agent::{Ask, Question, Usage};
use crate::language::Language;
use crate::model::ModelLevel;
use crate::snippet::outline;

/// A module's things are named once its findings come from at least this
/// many rules: symptoms that converge from different angles.
const MIN_RULES: usize = 2;
/// ... and number at least this many.
const MIN_SYMPTOMS: usize = 3;
/// A thing is described only if this many symptoms concern it.
const MIN_SYMPTOMS_PER_THING: usize = 2;
/// The most outline lines the namer is shown of a module: its root first,
/// then the files with the most symptoms.
const MAX_SHAPE_LINES: usize = 6000;

const IDENTIFIER: &str = "\
Review findings cluster in one module of a codebase; each is a symptom. Name \
the thing the symptoms are about, in its own domain's words, as a \
practitioner would say it: \"an assembler\", \"a register allocator\", \"a \
rate limiter\", \"a terminal emulator\". A module may be several things: \
name each thing that at least two symptoms concern, list those symptoms' \
numbers, and give its context in one phrase — what it is part of, e.g. \"in \
a JIT compiler that emits SIMD kernels for x86-64 and aarch64\". Name what \
the thing is, not what the code calls it, and do not describe the code.";

const DENOTER: &str = "\
Describe the thing named below from first principles, as a textbook or an \
expert practitioner would — not as any particular codebase does. What is \
it? What are its parts, and the vocabulary its practitioners use for them? \
What does it do, and how does it work? How does it behave: its inputs and \
outputs, what it guarantees, what is true of every correct one? Be concrete \
and complete. You have not seen the code; do not guess at it.";

/// One kind of call naming makes.
struct Step {
    level: ModelLevel,
    system: &'static str,
    schema: fn() -> serde_json::Value,
}

const IDENTIFY: Step = Step {
    level: ModelLevel::Fast,
    system: IDENTIFIER,
    schema: || {
        serde_json::json!({
            "type": "object",
            "properties": { "things": { "type": "array", "items": {
                "type": "object",
                "properties": {
                    "thing": { "type": "string" },
                    "context": { "type": "string" },
                    "symptoms": { "type": "array", "items": { "type": "integer" } }
                },
                "required": ["thing", "context", "symptoms"]
            } } },
            "required": ["things"]
        })
    },
};

const DESCRIBE: Step = Step {
    level: ModelLevel::Fast,
    system: DENOTER,
    schema: || {
        serde_json::json!({
            "type": "object",
            "properties": { "description": { "type": "string" } },
            "required": ["description"]
        })
    },
};

/// One module's symptoms: the findings in it, as indexes into the report.
struct Cluster {
    component: PathBuf,
    symptoms: Vec<usize>,
}

pub(super) async fn identify<A: Ask>(agent: &A, report: &Report) -> Result<Identified> {
    let clusters = clusters(&report.findings);
    let named = join_all(
        clusters
            .iter()
            .map(|cluster| named(agent, cluster, &report.findings)),
    )
    .await;
    let mut identified = Identified::default();
    for named in named {
        let named = named?;
        identified.things.extend(named.things);
        identified.failures.extend(named.failures);
        identified.usage += named.usage;
    }
    Ok(identified)
}

/// The things one cluster's symptoms are about, each described.
async fn named<A: Ask>(agent: &A, cluster: &Cluster, findings: &[Finding]) -> Result<Identified> {
    let symptoms = cluster.symptoms.iter().map(|&i| findings[i].path.as_path());
    let (shape, _) = shape(&cluster.component, symptoms, MAX_SHAPE_LINES)?;
    let mut usage = Usage::default();
    let prompt = format!(
        "Module: {}\n\nSymptoms:\n{}\nThe module's outline:\n\n{shape}",
        cluster.component.display(),
        numbered(&cluster.symptoms, findings)
    );
    let parsed = ask(agent, &IDENTIFY, &prompt, &mut usage)
        .await
        .and_then(|reply| {
            serde_json::from_str::<Things>(&reply)
                .with_context(|| format!("reply is not a list of things: {reply}"))
        });
    let named = match parsed {
        Ok(named) => named,
        Err(error) => {
            let error = error.context(format!("naming {}", cluster.component.display()));
            return Ok(Identified {
                failures: vec![error],
                usage,
                ..Identified::default()
            });
        }
    };
    let kept: Vec<(Named, Vec<usize>)> = named
        .things
        .into_iter()
        .filter_map(|thing| {
            let explains = thing.symptoms_in(cluster)?;
            Some((thing, explains))
        })
        .collect();
    let described = join_all(kept.iter().map(|(thing, _)| describe(agent, thing))).await;

    let mut identified = Identified {
        usage,
        ..Identified::default()
    };
    for ((thing, explains), described) in kept.into_iter().zip(described) {
        identified.usage += described.usage;
        match described.description {
            Ok(description) => identified.things.push(Thing {
                component: cluster.component.clone(),
                thing: thing.thing,
                context: thing.context,
                description,
                explains,
            }),
            Err(error) => identified.failures.push(error.context(format!(
                "describing {} in {}",
                thing.thing,
                cluster.component.display()
            ))),
        }
    }
    Ok(identified)
}

struct Described {
    description: Result<String>,
    usage: Usage,
}

/// `thing` from first principles, by a call shown only its name and
/// context, never the code.
async fn describe<A: Ask>(agent: &A, thing: &Named) -> Described {
    let mut usage = Usage::default();
    let named = format!("{}, {}", thing.thing, thing.context);
    let description = ask(agent, &DESCRIBE, &named, &mut usage)
        .await
        .and_then(|reply| {
            let Description { description } = serde_json::from_str(&reply)
                .with_context(|| format!("reply is not a description: {reply}"))?;
            if description.trim().is_empty() {
                bail!("{named} was described as nothing");
            }
            Ok(description)
        });
    Described { description, usage }
}

async fn ask<A: Ask>(agent: &A, step: &Step, prompt: &str, usage: &mut Usage) -> Result<String> {
    let schema = (step.schema)();
    let question = Question {
        level: step.level,
        system: step.system,
        prompt,
        schema: Some(&schema),
    };
    let answer = agent.ask(&question).await?;
    *usage += answer.usage;
    Ok(answer.text)
}

/// The modules whose findings converge: at least [`MIN_SYMPTOMS`] of them,
/// from at least [`MIN_RULES`] rules.
fn clusters(findings: &[Finding]) -> Vec<Cluster> {
    let mut by_component: BTreeMap<PathBuf, Vec<usize>> = BTreeMap::new();
    for (index, finding) in findings.iter().enumerate() {
        by_component
            .entry(component(&finding.path))
            .or_default()
            .push(index);
    }
    by_component
        .into_iter()
        .filter(|(_, symptoms)| {
            let rules: BTreeSet<&str> = symptoms
                .iter()
                .map(|&i| findings[i].rule.as_str())
                .collect();
            symptoms.len() >= MIN_SYMPTOMS && rules.len() >= MIN_RULES
        })
        .map(|(component, symptoms)| Cluster {
            component,
            symptoms,
        })
        .collect()
}

/// The module `path` belongs to: the nearest directory, from its own, that
/// holds a module root; a crate-wide finding's path is its crate already.
fn component(path: &Path) -> PathBuf {
    if path.is_dir() {
        return path.to_path_buf();
    }
    path.ancestors()
        .skip(1)
        .find(|dir| ROOT_FILES.iter().any(|root| dir.join(root).is_file()))
        .or_else(|| path.parent())
        .map_or_else(PathBuf::new, Path::to_path_buf)
}

/// The outline of the module at `component`: its root first, then the files
/// `symptoms` are in, most symptoms first, each under a `== path ==` header,
/// to at most `budget` lines (the first file is shown whole). Returns the
/// lines used.
pub(super) fn shape<'p>(
    component: &Path,
    symptoms: impl Iterator<Item = &'p Path>,
    budget: usize,
) -> Result<(String, usize)> {
    let mut by_file: BTreeMap<&Path, usize> = BTreeMap::new();
    for path in symptoms {
        *by_file.entry(path).or_default() += 1;
    }
    let mut files: Vec<(PathBuf, usize)> = ROOT_FILES
        .iter()
        .map(|root| component.join(root))
        .filter(|root| root.is_file())
        .map(|root| (root, usize::MAX))
        .collect();
    files.extend(
        by_file
            .into_iter()
            .filter(|(path, _)| path.is_file())
            .map(|(path, n)| (path.to_path_buf(), n)),
    );
    files.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    files.dedup_by(|a, b| a.0 == b.0);

    let mut shape = String::new();
    let mut lines = 0;
    for (path, _) in files {
        let Some(language) = Language::of(&path) else {
            continue;
        };
        let source = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let outlined =
            outline(language, &source).with_context(|| format!("parsing {}", path.display()))?;
        let length = outlined.numbered.lines().count();
        if lines + length > budget && lines > 0 {
            shape.push_str(&format!(
                "== {} == (omitted: over budget)\n",
                path.display()
            ));
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

/// The findings at `indexes`, numbered from `S1`.
fn numbered(indexes: &[usize], findings: &[Finding]) -> String {
    indexes
        .iter()
        .enumerate()
        .map(|(n, &i)| {
            let f = &findings[i];
            format!(
                "S{}. {}:{} [{}/{}] {}\n",
                n + 1,
                f.path.display(),
                f.line,
                f.rule,
                f.outcome,
                f.message
            )
        })
        .collect()
}

#[derive(Deserialize)]
struct Things {
    things: Vec<Named>,
}

/// One thing as the namer replied with it.
#[derive(Deserialize)]
struct Named {
    thing: String,
    context: String,
    /// Symptom numbers, from 1, within the cluster.
    symptoms: Vec<usize>,
}

impl Named {
    /// The report indexes of this thing's symptoms, if enough of them are
    /// real.
    fn symptoms_in(&self, cluster: &Cluster) -> Option<Vec<usize>> {
        let mut picked: Vec<usize> = self
            .symptoms
            .iter()
            .filter_map(|n| cluster.symptoms.get(n.checked_sub(1)?).copied())
            .collect();
        picked.sort_unstable();
        picked.dedup();
        (picked.len() >= MIN_SYMPTOMS_PER_THING).then_some(picked)
    }
}

#[derive(Deserialize)]
struct Description {
    description: String,
}
