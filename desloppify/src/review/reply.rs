//! What the reviewer is told, and reading its reply.

use anyhow::{Context, Result};
use std::path::PathBuf;

use serde::Deserialize;

use super::Call;
use crate::rule::Rule;

const REVIEWER: &str = "\
You are a code reviewer applying exactly one rule, given below. Report only \
violations of that rule, in the code you are shown: one finding per \
violation, with the line number printed in the code and what is wrong and \
why. When the code comes from several files, each under a `== path ==` \
header, give each finding that header's path. Report no findings if the code \
does not violate the rule.";

/// The shape every reviewer's reply must have.
pub(super) fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "line": { "type": "integer" },
                        "message": { "type": "string" }
                    },
                    "required": ["line", "message"]
                }
            }
        },
        "required": ["findings"]
    })
}

/// A reply in [`schema`]'s shape.
#[derive(Deserialize)]
struct Replies {
    findings: Vec<Reply>,
}

#[derive(Deserialize)]
pub(super) struct Reply {
    /// Set when the code shown came from several files.
    #[serde(default)]
    pub(super) path: Option<PathBuf>,
    pub(super) line: u64,
    pub(super) message: String,
}

/// The system prompt for a call under `rule`.
pub(super) fn preamble(rule: &Rule) -> String {
    format!("{REVIEWER}\n\n{}", rule.instructions)
}

/// The user prompt for `call`: the file, its snippet, and any context.
pub(super) fn prompt(call: &Call) -> String {
    let mut prompt = format!("File: {}\n\n{}", call.path.display(), call.snippet.numbered);
    if let Some(root) = &call.root {
        prompt.push_str(&format!(
            "\nFor context only, not under review — its module root, {}:\n\n{}",
            root.path.display(),
            root.text
        ));
    }
    prompt
}

/// The findings in a reply, which the backend was asked to make satisfy
/// [`schema`].
pub(super) fn parse(reply: &str) -> Result<Vec<Reply>> {
    let replies: Replies = serde_json::from_str(reply)
        .with_context(|| format!("reply is not a findings object: {reply}"))?;
    Ok(replies.findings)
}
