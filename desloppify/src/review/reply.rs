//! What the reviewer is told, and reading its reply.

use anyhow::{Context, Result};
use std::path::PathBuf;

use serde::Deserialize;

use super::Call;
use crate::rule::Rule;

const REVIEWER: &str = "\
You are a code reviewer applying exactly one rule, given below. Report only \
violations of that rule, in the code you are shown. Reply with only a JSON \
array, no prose: one {\"line\": <line number>, \"message\": \"<what and why>\"} \
per violation, using the line numbers printed in the code. When the code \
comes from several files, each under a `== path ==` header, add \"path\": \
\"<that path>\" to each violation. Reply [] if the code does not violate the \
rule.";

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

/// The findings in a reply, tolerating the code fence models like to add.
pub(super) fn parse(reply: &str) -> Result<Vec<Reply>> {
    let body = reply.trim();
    let body = body
        .strip_prefix("```json")
        .or_else(|| body.strip_prefix("```"))
        .unwrap_or(body);
    let body = body.strip_suffix("```").unwrap_or(body);
    serde_json::from_str(body.trim())
        .with_context(|| format!("reply is not a findings array: {reply}"))
}
