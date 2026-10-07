//! [`Ask`](super::Ask) without a model: prices the call and finds nothing.

use anyhow::Result;

use super::{Answer, Ask, Question, Usage};

const CHARS_PER_TOKEN: u64 = 4;
/// What a dry run answers: for a schema, the reply that says "nothing found"
/// (an object whose arrays are empty); for free text, nothing.
fn empty_reply(schema: Option<&serde_json::Value>) -> String {
    let Some(properties) = schema
        .and_then(|s| s.get("properties"))
        .and_then(|p| p.as_object())
    else {
        return String::new();
    };
    let empty: serde_json::Map<String, serde_json::Value> = properties
        .keys()
        .map(|key| (key.clone(), serde_json::Value::Array(Vec::new())))
        .collect();
    serde_json::Value::Object(empty).to_string()
}

pub(super) struct DryRun;

impl Ask for DryRun {
    async fn ask(&self, question: &Question<'_>) -> Result<Answer> {
        Ok(Answer {
            text: empty_reply(question.schema),
            usage: Usage {
                calls: 1,
                input: (question.system.len() + question.prompt.len()) as u64 / CHARS_PER_TOKEN,
                output: 0,
            },
        })
    }
}
