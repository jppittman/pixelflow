//! From symptoms to a diagnosis: name the thing, describe it without the
//! code, and translate the description's shape onto the code's.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use super::plan::ROOT_FILES;
use super::{Diagnoses, Diagnosis, Finding, Report, Root};
use crate::agent::{Ask, Question};
use crate::language::Language;
use crate::model::ModelLevel;
use crate::snippet::outline;

/// A module is diagnosed once its findings come from at least this many
/// rules: symptoms that converge from different angles.
const MIN_RULES: usize = 2;
/// ... and number at least this many.
const MIN_SYMPTOMS: usize = 3;
/// A thing is described and diagnosed only if this many symptoms concern it.
const MIN_SYMPTOMS_PER_THING: usize = 2;
/// A module's diagnoses are asked for the root beneath them once there are
/// at least this many.
const MIN_DIAGNOSES_FOR_ROOT: usize = 2;
/// The most outline lines the diagnosis is shown of a module: its root
/// first, then the files with the most symptoms.
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

const DIAGNOSER: &str = "\
You are given a thing described from first principles, the shape of the \
code that implements it — an outline: its items and signatures, function \
bodies elided, at their line numbers — and review findings about that code, \
numbered: its symptoms.

First translate the description's shape onto the code's. For each part and \
behaviour in the description, say where it is in the code: a type, a \
function, a value, a convention kept in a comment, or nowhere. Then say what \
the code has that the description does not.

Then diagnose: the model of the thing that the code is missing or has \
wrong, stated as what the thing is (\"a label is an operand\", \"instruction \
selection is a phase\"). Say which symptoms it explains, by number. One \
diagnosis, the deepest one the evidence supports: if it explains only one \
symptom, look further. If the symptoms are not explained by the shape, say \
so and explain none.

Then follow the consequences to closure — the step most often skipped. If \
the code had the thing's shape, what would stop being special, what would \
become a type, what would be deleted? And then what follows from that? Keep \
asking until nothing more falls out, and report the whole chain. Refuse \
exceptions: a fix that keeps one (\"except this register\", \"except this \
construct\") is a symptom the diagnosis has not explained yet.";

const ROOTER: &str = "\
Each diagnosis below explains a cluster of review findings in one module: \
what one thing in the code is missing or has wrong. Diagnoses are \
themselves symptoms. Find the deepest model beneath several of them: the \
thing whose absence or wrong shape makes those diagnoses true at once — a \
missing phase, a missing type, a wrong boundary — stated as what that thing \
is. Say which diagnoses it explains, by number, and why each follows from \
it. Then follow the consequences to closure: with that model in place, what \
stops being special, what is deleted, and what follows from that, until \
nothing more falls out. Refuse exceptions: a model that keeps one (\"except \
this register\", \"except this construct\") has not reached the root. One \
root, the deepest the evidence supports; if the diagnoses share none, say so \
and explain none.";

/// One kind of call a diagnosis makes.
struct Step {
    level: ModelLevel,
    system: &'static str,
    schema: fn() -> serde_json::Value,
}

const IDENTIFY: Step = Step {
    level: ModelLevel::Strong,
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
    level: ModelLevel::Frontier,
    system: DENOTER,
    schema: || {
        serde_json::json!({
            "type": "object",
            "properties": { "description": { "type": "string" } },
            "required": ["description"]
        })
    },
};

const DIAGNOSE: Step = Step {
    level: ModelLevel::Frontier,
    system: DIAGNOSER,
    schema: || {
        serde_json::json!({
            "type": "object",
            "properties": {
                "shape": { "type": "string" },
                "diagnosis": { "type": "string" },
                "falls_out": { "type": "string" },
                "explains": { "type": "array", "items": { "type": "integer" } }
            },
            "required": ["shape", "diagnosis", "falls_out", "explains"]
        })
    },
};

const ROOT: Step = Step {
    level: ModelLevel::Frontier,
    system: ROOTER,
    schema: || {
        serde_json::json!({
            "type": "object",
            "properties": {
                "root": { "type": "string" },
                "falls_out": { "type": "string" },
                "explains": { "type": "array", "items": { "type": "integer" } }
            },
            "required": ["root", "falls_out", "explains"]
        })
    },
};

/// One module's symptoms: the findings in it, as indexes into the report.
struct Cluster {
    component: PathBuf,
    symptoms: Vec<usize>,
}

pub(super) async fn diagnose<A: Ask>(agent: &A, report: &Report) -> Result<Diagnoses> {
    let mut diagnoses = Diagnoses::default();
    for cluster in clusters(&report.findings) {
        let first = diagnoses.diagnoses.len();
        let shape = shape(&cluster, &report.findings)?;
        let symptoms = numbered(&cluster.symptoms, &report.findings);
        let mut session = Session {
            agent,
            usage: &mut diagnoses.usage,
        };
        let things = match session.identify(&cluster, &shape, &symptoms).await {
            Ok(things) => things,
            Err(error) => {
                let error = error.context(format!("naming {}", cluster.component.display()));
                diagnoses.failures.push(error);
                continue;
            }
        };
        for thing in things {
            let Some(explaining) = thing.symptoms_in(&cluster) else {
                continue;
            };
            let found = session
                .describe_and_diagnose(&thing, &shape, &numbered(&explaining, &report.findings))
                .await;
            match found {
                Ok(found) => diagnoses.diagnoses.push(Diagnosis {
                    component: cluster.component.clone(),
                    thing: thing.thing,
                    denotation: found.denotation,
                    shape: found.reply.shape,
                    diagnosis: found.reply.diagnosis,
                    falls_out: found.reply.falls_out,
                    explains: found
                        .reply
                        .explains
                        .iter()
                        .filter_map(|n| explaining.get(n.checked_sub(1)?).copied())
                        .collect(),
                }),
                Err(error) => {
                    let error = error.context(format!(
                        "diagnosing {} in {}",
                        thing.thing,
                        cluster.component.display()
                    ));
                    diagnoses.failures.push(error);
                }
            }
        }

        let found = &diagnoses.diagnoses[first..];
        if found.len() < MIN_DIAGNOSES_FOR_ROOT {
            continue;
        }
        let rooted = session.root(&cluster, found).await;
        match rooted {
            Ok(reply) => {
                let count = diagnoses.diagnoses.len() - first;
                diagnoses.roots.push(Root {
                    component: cluster.component.clone(),
                    root: reply.root,
                    falls_out: reply.falls_out,
                    explains: reply
                        .explains
                        .iter()
                        .filter_map(|n| n.checked_sub(1).filter(|i| *i < count))
                        .map(|i| first + i)
                        .collect(),
                });
            }
            Err(error) => {
                let error = error.context(format!("rooting {}", cluster.component.display()));
                diagnoses.failures.push(error);
            }
        }
    }
    Ok(diagnoses)
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

/// The module's outline: its root first, then its files with the most
/// symptoms, each under a `== path ==` header, to [`MAX_SHAPE_LINES`].
fn shape(cluster: &Cluster, findings: &[Finding]) -> Result<String> {
    let mut by_file: BTreeMap<&Path, usize> = BTreeMap::new();
    for &i in &cluster.symptoms {
        *by_file.entry(findings[i].path.as_path()).or_default() += 1;
    }
    let mut files: Vec<(PathBuf, usize)> = ROOT_FILES
        .iter()
        .map(|root| cluster.component.join(root))
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
        if lines + length > MAX_SHAPE_LINES && lines > 0 {
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
    Ok(shape)
}

/// The findings at `indexes`, numbered from `S1`.
fn numbered(indexes: &[usize], findings: &[Finding]) -> String {
    indexes
        .iter()
        .enumerate()
        .map(|(n, &i)| symptom(n + 1, &findings[i]))
        .collect()
}

fn symptom(number: usize, f: &Finding) -> String {
    format!(
        "S{number}. {}:{} [{}/{}] {}\n",
        f.path.display(),
        f.line,
        f.rule,
        f.outcome,
        f.message
    )
}

#[derive(Deserialize)]
struct Things {
    things: Vec<Thing>,
}

#[derive(Deserialize)]
struct Thing {
    thing: String,
    context: String,
    /// Symptom numbers, from 1, within the cluster.
    symptoms: Vec<usize>,
}

impl Thing {
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

#[derive(Deserialize)]
struct Reply {
    shape: String,
    diagnosis: String,
    falls_out: String,
    /// Symptom numbers, from 1, within this thing's symptoms.
    explains: Vec<usize>,
}

#[derive(Deserialize)]
struct Rooted {
    root: String,
    falls_out: String,
    /// Diagnosis numbers, from 1, within this module's diagnoses.
    explains: Vec<usize>,
}

struct Found {
    denotation: String,
    reply: Reply,
}

/// The calls of one diagnosis, and the tokens they use.
struct Session<'a, A> {
    agent: &'a A,
    usage: &'a mut crate::agent::Usage,
}

impl<A: Ask> Session<'_, A> {
    async fn identify(
        &mut self,
        cluster: &Cluster,
        shape: &str,
        symptoms: &str,
    ) -> Result<Vec<Thing>> {
        let prompt = format!(
            "Module: {}\n\nSymptoms:\n{symptoms}\nThe module's outline:\n\n{shape}",
            cluster.component.display()
        );
        let reply = self.ask(&IDENTIFY, &prompt).await?;
        let things: Things = serde_json::from_str(&reply)
            .with_context(|| format!("reply is not a list of things: {reply}"))?;
        Ok(things.things)
    }

    /// Describes `thing` without the code, then translates that
    /// description onto the code's `shape`.
    async fn describe_and_diagnose(
        &mut self,
        thing: &Thing,
        shape: &str,
        symptoms: &str,
    ) -> Result<Found> {
        let named = format!("{}, {}", thing.thing, thing.context);
        let reply = self.ask(&DESCRIBE, &named).await?;
        let Description { description } = serde_json::from_str(&reply)
            .with_context(|| format!("reply is not a description: {reply}"))?;
        if description.trim().is_empty() {
            bail!("{named} was described as nothing");
        }

        let prompt = format!(
            "The thing: {named}\n\nIt, from first principles:\n\n{description}\n\nThe code's shape:\n\n{shape}\nSymptoms:\n{symptoms}"
        );
        let reply = self.ask(&DIAGNOSE, &prompt).await?;
        let reply: Reply = serde_json::from_str(&reply)
            .with_context(|| format!("reply is not a diagnosis: {reply}"))?;
        Ok(Found {
            denotation: description,
            reply,
        })
    }

    /// The model beneath several of a module's `diagnoses`.
    async fn root(&mut self, cluster: &Cluster, diagnoses: &[Diagnosis]) -> Result<Rooted> {
        let listed: String = diagnoses
            .iter()
            .enumerate()
            .map(|(n, d)| {
                format!(
                    "D{}. {}\nDiagnosis: {}\nWith the thing's shape: {}\n\n",
                    n + 1,
                    d.thing,
                    d.diagnosis,
                    d.falls_out
                )
            })
            .collect();
        let prompt = format!(
            "Module: {}\n\nDiagnoses:\n\n{listed}",
            cluster.component.display()
        );
        let reply = self.ask(&ROOT, &prompt).await?;
        serde_json::from_str(&reply).with_context(|| format!("reply is not a root: {reply}"))
    }

    async fn ask(&mut self, step: &Step, prompt: &str) -> Result<String> {
        let schema = (step.schema)();
        let question = Question {
            level: step.level,
            system: step.system,
            prompt,
            schema: Some(&schema),
        };
        let answer = self.agent.ask(&question).await?;
        *self.usage += answer.usage;
        Ok(answer.text)
    }
}
