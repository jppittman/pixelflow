//! [`Decide`](super::Decide) through TypeSafe AI's System One API.
//!
//! The wire shapes follow the API's OpenAPI schema, as generated into
//! TypeSafe's Python SDK (`typesafe_sdk/_schemas/models.py`).

use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use super::{Choice, Decide, Decided, Decisions, JevConfig};
use crate::agent::Usage;

const PATH: &str = "/v1/systemone";
const API_KEY_ENV: &str = "TYPESAFE_API_KEY";
const BASE_URL_ENV: &str = "TYPESAFE_BASE_URL";
const MODEL_ENV: &str = "TYPESAFE_DEFAULT_MODEL";
const DEFAULT_BASE_URL: &str = "https://api.typesafe.ai";
const DEFAULT_MODEL: &str = "jev-latest";
const TIMEOUT: Duration = Duration::from_secs(30);
/// Retries after the first attempt, as TypeSafe's SDK does by default.
const RETRIES: u32 = 2;
/// The wait before a retry when the server names none.
const BACKOFF: Duration = Duration::from_millis(500);
const RETRY_AFTER: &str = "retry-after";
const RETRY_AFTER_MS: &str = "retry-after-ms";
const MILLIS_PER_SECOND: f64 = 1000.0;
/// The longest server-requested wait honoured; a longer one, like a
/// malformed one, gets the default backoff, as TypeSafe's SDK does.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);

pub(super) struct Jev {
    config: JevConfig,
    http: reqwest::Client,
    in_flight: Semaphore,
}

#[derive(Serialize)]
struct Request<'a> {
    state: &'a str,
    model: &'a str,
    questions: BTreeMap<&'a str, Question<'a>>,
}

#[derive(Serialize)]
struct Question<'a> {
    r#type: &'static str,
    instructions: &'a str,
    criteria: BTreeMap<&'a str, &'a str>,
}

#[derive(Deserialize)]
struct Response {
    answers: BTreeMap<String, Answer>,
    usage: WireUsage,
}

/// Only choice answers are asked for; another type is skipped, not an error,
/// as the SDK does for answer kinds a later API adds.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Answer {
    Choice {
        choice: String,
        confidence: f64,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct WireUsage {
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
}

pub(super) fn from_env(jobs: std::num::NonZeroUsize) -> Result<JevConfig> {
    let var = |name: &str| {
        std::env::var(name)
            .ok()
            .map(|v| v.trim().to_owned())
            .filter(|v| !v.is_empty())
    };
    let Some(api_key) = var(API_KEY_ENV) else {
        bail!("{API_KEY_ENV} is not set");
    };
    Ok(JevConfig {
        base_url: var(BASE_URL_ENV).unwrap_or_else(|| DEFAULT_BASE_URL.to_owned()),
        api_key,
        model: var(MODEL_ENV).unwrap_or_else(|| DEFAULT_MODEL.to_owned()),
        jobs,
    })
}

impl Jev {
    pub(super) fn new(config: JevConfig) -> Self {
        Self {
            in_flight: Semaphore::new(config.jobs.get()),
            http: reqwest::Client::new(),
            config,
        }
    }
}

impl Decide for Jev {
    async fn decide(&self, state: &str, questions: &[Choice<'_>]) -> Result<Decisions> {
        let _permit = self.in_flight.acquire().await?;
        let request = Request {
            state,
            model: &self.config.model,
            questions: questions
                .iter()
                .map(|q| {
                    let question = Question {
                        r#type: "choice",
                        instructions: q.instructions,
                        criteria: q.labels.iter().copied().collect(),
                    };
                    (q.name, question)
                })
                .collect(),
        };
        let body = serde_json::to_vec(&request)?;
        let url = format!("{}{PATH}", self.config.base_url.trim_end_matches('/'));

        let mut retries = 0;
        let response = loop {
            let response = self
                .http
                .post(&url)
                .bearer_auth(&self.config.api_key)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .timeout(TIMEOUT)
                .body(body.clone())
                .send()
                .await
                .with_context(|| format!("POST {url}"))?;
            let status = response.status();
            if status.is_success() {
                break response;
            }
            let retryable = matches!(status.as_u16(), 408 | 429) || status.is_server_error();
            if !retryable || retries == RETRIES {
                let text = response.text().await.unwrap_or_default();
                bail!("POST {url} → {status}: {text}");
            }
            retries += 1;
            tokio::time::sleep(retry_after(response.headers()).unwrap_or(BACKOFF * retries)).await;
        };

        let text = response.bytes().await?;
        let response: Response = serde_json::from_slice(&text)
            .with_context(|| format!("{url} answered with something other than answers"))?;
        let answers = response
            .answers
            .into_iter()
            .filter_map(|(name, answer)| match answer {
                Answer::Choice { choice, confidence } => Some((
                    name,
                    Decided {
                        label: choice,
                        confidence,
                    },
                )),
                Answer::Other => None,
            })
            .collect();
        Ok(Decisions {
            answers,
            usage: Usage {
                calls: 1,
                input: response.usage.input_tokens,
                output: response.usage.output_tokens,
            },
        })
    }
}

/// The server's requested wait: `retry-after-ms`, or `retry-after` in
/// seconds, if it is a wait at all and no longer than [`MAX_RETRY_AFTER`].
/// The HTTP-date form is not read.
fn retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let read = |name: &str, per_second: f64| {
        let value = headers
            .get(name)?
            .to_str()
            .ok()?
            .trim()
            .parse::<f64>()
            .ok()?;
        Duration::try_from_secs_f64(value / per_second).ok()
    };
    read(RETRY_AFTER_MS, MILLIS_PER_SECOND)
        .or_else(|| read(RETRY_AFTER, 1.0))
        .filter(|wait| !wait.is_zero() && *wait <= MAX_RETRY_AFTER)
}
