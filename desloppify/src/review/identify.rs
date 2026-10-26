//! From symptoms to the things they are about: name what each converging
//! module is, and describe each thing without the code. Many small calls,
//! every module at once; the judgment is the synthesis's.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use futures::future::join_all;
use serde::Deserialize;

use super::module;
use super::numbering::Numbering;
use super::reply::{Step, ask};
use super::{FindingIx, Report, Thing};
use crate::agent::{Ask, Usage};
use crate::model::ModelLevel;

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

fn identify_schema() -> serde_json::Value {
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
}

fn describe_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": { "description": { "type": "string" } },
        "required": ["description"]
    })
}

/// What naming made of some modules: the things, what failed, the tokens used.
#[derive(Default)]
struct Identified {
    things: Vec<Thing>,
    failures: Vec<anyhow::Error>,
    usage: Usage,
}

/// One module's symptoms: the findings in it.
struct Cluster {
    component: PathBuf,
    /// In report order; the namer numbers them `S1`... by position here.
    symptoms: Vec<FindingIx>,
}

impl Cluster {
    fn numbering(&self) -> Numbering {
        Numbering::symptoms(self.symptoms.len())
    }
}

pub(super) async fn identify<A: Ask>(agent: &A, report: &mut Report) -> Result<Usage> {
    let clusters = clusters(report);
    let named = join_all(clusters.iter().map(|cluster| named(agent, cluster, report))).await;
    let mut identified = Identified::default();
    for named in named {
        let named = named?;
        identified.things.extend(named.things);
        identified.failures.extend(named.failures);
        identified.usage += named.usage;
    }
    report.things = identified.things;
    report.failures.extend(identified.failures);
    Ok(identified.usage)
}

/// The things one cluster's symptoms are about, each described.
async fn named<A: Ask>(agent: &A, cluster: &Cluster, report: &Report) -> Result<Identified> {
    let symptoms = cluster
        .symptoms
        .iter()
        .map(|&i| report.finding(i).path.as_path());
    let (shape, _) = module::shape(&cluster.component, symptoms, MAX_SHAPE_LINES)?;
    let mut usage = Usage::default();
    let prompt = format!(
        "Module: {}\n\nSymptoms:\n{}\nThe module's outline:\n\n{shape}",
        cluster.component.display(),
        symptom_lines(cluster, report)
    );
    let schema = identify_schema();
    let step = Step {
        level: ModelLevel::Fast,
        system: IDENTIFIER,
        schema: &schema,
    };
    let parsed = ask(agent, &step, &prompt).await.and_then(|answer| {
        usage += answer.usage;
        let Things { things } = serde_json::from_str(&answer.text)
            .with_context(|| format!("reply is not a list of things: {}", answer.text))?;
        things
            .into_iter()
            .map(|thing| {
                let explains = thing.explains(cluster)?;
                Ok(explains.map(|explains| (thing, explains)))
            })
            .collect::<Result<Vec<_>>>()
    });
    let kept: Vec<(Named, Vec<FindingIx>)> = match parsed {
        Ok(kept) => kept.into_iter().flatten().collect(),
        Err(error) => {
            let error = error.context(format!("naming {}", cluster.component.display()));
            return Ok(Identified {
                failures: vec![error],
                usage,
                ..Identified::default()
            });
        }
    };
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
    let schema = describe_schema();
    let step = Step {
        level: ModelLevel::Fast,
        system: DENOTER,
        schema: &schema,
    };
    let description = ask(agent, &step, &named).await.and_then(|answer| {
        usage += answer.usage;
        let Description { description } = serde_json::from_str(&answer.text)
            .with_context(|| format!("reply is not a description: {}", answer.text))?;
        if description.trim().is_empty() {
            bail!("{named} was described as nothing");
        }
        Ok(description)
    });
    Described { description, usage }
}

/// The modules whose findings converge: at least [`MIN_SYMPTOMS`] of them,
/// from at least [`MIN_RULES`] rules.
fn clusters(report: &Report) -> Vec<Cluster> {
    let mut by_component: BTreeMap<PathBuf, Vec<FindingIx>> = BTreeMap::new();
    for (ix, finding) in report.findings.iter().enumerate() {
        by_component
            .entry(module::containing(&finding.path))
            .or_default()
            .push(FindingIx(ix));
    }
    by_component
        .into_iter()
        .filter(|(_, symptoms)| {
            let rules: BTreeSet<&str> = symptoms
                .iter()
                .map(|&i| report.finding(i).rule.as_str())
                .collect();
            symptoms.len() >= MIN_SYMPTOMS && rules.len() >= MIN_RULES
        })
        .map(|(component, symptoms)| Cluster {
            component,
            symptoms,
        })
        .collect()
}

/// The cluster's symptoms, one line each, numbered from `S1`.
fn symptom_lines(cluster: &Cluster, report: &Report) -> String {
    let symptoms = cluster.numbering();
    cluster
        .symptoms
        .iter()
        .enumerate()
        .map(|(n, &i)| {
            let f = report.finding(i);
            format!(
                "{}. {}:{} [{}/{}] {}\n",
                symptoms.label(n),
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
    symptoms: Vec<usize>,
}

impl Named {
    /// The findings this thing's symptoms name, in report order, each once;
    /// `None` if too few to be a thing.
    ///
    /// # Errors
    ///
    /// A number names no symptom of the cluster.
    fn explains(&self, cluster: &Cluster) -> Result<Option<Vec<FindingIx>>> {
        let mut picked = cluster
            .numbering()
            .resolve_all(&self.symptoms, |n| cluster.symptoms[n])?;
        picked.sort_unstable_by_key(|finding| finding.0);
        picked.dedup_by_key(|finding| finding.0);
        Ok((picked.len() >= MIN_SYMPTOMS_PER_THING).then_some(picked))
    }
}

#[derive(Deserialize)]
struct Description {
    description: String,
}
