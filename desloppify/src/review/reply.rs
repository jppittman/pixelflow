//! What the reviewer is told, and reading its reply.

use anyhow::{Context, Result};
use serde::Deserialize;

pub(super) const REVIEWER: &str = "\
You are a code reviewer applying exactly one rule, given below. Report only \
violations of that rule, in the code you are shown. Reply with only a JSON \
array, no prose: one {\"line\": <line number>, \"message\": \"<what and why>\"} \
per violation, using the line numbers printed in the code. Reply [] if the \
code does not violate the rule.";

#[derive(Deserialize)]
pub(super) struct Reply {
    pub(super) line: u64,
    pub(super) message: String,
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
