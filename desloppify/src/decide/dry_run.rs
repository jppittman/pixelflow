//! [`Decide`](super::Decide) without a model: prices the request and
//! answers every question with its first label.

use anyhow::Result;

use super::{Choice, Decide, Decided, Decisions};
use crate::agent::Usage;

const CHARS_PER_TOKEN: u64 = 4;

pub(super) struct DryRun;

impl Decide for DryRun {
    async fn decide(&self, state: &str, questions: &[Choice<'_>]) -> Result<Decisions> {
        let asked: usize = questions
            .iter()
            .map(|q| {
                q.instructions.len()
                    + q.labels
                        .iter()
                        .map(|(l, m)| l.len() + m.len())
                        .sum::<usize>()
            })
            .sum();
        let answers = questions
            .iter()
            .filter_map(|q| {
                let (label, _) = q.labels.first()?;
                let decided = Decided {
                    label: (*label).to_owned(),
                    confidence: 1.0,
                };
                Some((q.name.to_owned(), decided))
            })
            .collect();
        Ok(Decisions {
            answers,
            usage: Usage {
                calls: 1,
                input: (state.len() + asked) as u64 / CHARS_PER_TOKEN,
                output: 0,
            },
        })
    }
}
