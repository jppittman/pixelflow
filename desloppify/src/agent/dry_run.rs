//! [`Ask`](super::Ask) without a model: prices the call and finds nothing.

use anyhow::Result;

use super::{Answer, Ask, Question, Usage};

const CHARS_PER_TOKEN: u64 = 4;
/// What a dry run answers for a schema: an object whose enum properties
/// take their first value — a decision's first outcome, which is a fine one
/// — whose arrays are empty — no findings — and whose strings are empty.
/// For free text, nothing.
fn empty_reply(schema: Option<&serde_json::Value>) -> String {
    let Some(properties) = schema
        .and_then(|s| s.get("properties"))
        .and_then(|p| p.as_object())
    else {
        return String::new();
    };
    let empty: serde_json::Map<String, serde_json::Value> = properties
        .iter()
        .map(|(key, property)| {
            let first = property.get("enum").and_then(|e| e.get(0)).cloned();
            let value = match (first, property.get("type").and_then(|t| t.as_str())) {
                (Some(first), _) => first,
                (None, Some("array")) => serde_json::Value::Array(Vec::new()),
                (None, _) => serde_json::Value::String(String::new()),
            };
            (key.clone(), value)
        })
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
