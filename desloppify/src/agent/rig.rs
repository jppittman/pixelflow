//! [`Ask`](super::Ask) over rig-core's Anthropic and Gemini clients.

use std::error::Error;
use std::num::NonZeroUsize;
use std::time::Duration;

use anyhow::Result;
use rig_core::ProviderError;
use rig_core::completion::CompletionRequest;
use rig_core::providers::{anthropic::Anthropic, gemini::Gemini};
use tokio::sync::Semaphore;

use super::Ask;
use crate::model::{ModelLevel, Provider};
use crate::rate_limit::{BoxError, RateLimiter, Signal};

/// Enough for a list of findings; the reply is JSON, not prose.
const MAX_REPLY_TOKENS: u64 = 4096;

const TOO_MANY_REQUESTS: u16 = 429;
const RETRY_AFTER: &str = "retry-after";

enum Client {
    Anthropic(Anthropic),
    Gemini(Gemini),
}

pub(super) struct RigAgent {
    provider: Provider,
    client: Client,
    limiter: Box<dyn RateLimiter>,
    /// Bounds the calls waiting on the limiter, so a pace it sets reaches
    /// the next call rather than the end of a queue booked at the old one.
    in_flight: Semaphore,
}

impl RigAgent {
    pub(super) fn from_env(
        provider: Provider,
        limiter: Box<dyn RateLimiter>,
        jobs: NonZeroUsize,
    ) -> Result<Self> {
        let client = match provider {
            Provider::Anthropic => Client::Anthropic(Anthropic::from_env()?),
            Provider::Gemini => Client::Gemini(Gemini::from_env()?),
        };
        Ok(Self {
            provider,
            client,
            limiter,
            in_flight: Semaphore::new(jobs.get()),
        })
    }
}

impl Ask for RigAgent {
    async fn ask(&self, level: ModelLevel, preamble: &str, prompt: &str) -> Result<String> {
        let _permit = self.in_flight.acquire().await?;
        let model = self.provider.model(level);
        let mut last: Option<BoxError> = None;
        loop {
            let wait = self
                .limiter
                .wait(last.take())
                .map_err(|e| anyhow::anyhow!(e))?;
            tokio::time::sleep(wait).await;
            let request = CompletionRequest::new(prompt)
                .preamble(preamble)
                .max_tokens(MAX_REPLY_TOKENS);
            let reply = match &self.client {
                Client::Anthropic(client) => client
                    .completion(model)
                    .call(request)
                    .await
                    .map(|r| r.text()),
                Client::Gemini(client) => client
                    .completion(model)
                    .call(request)
                    .await
                    .map(|r| r.text()),
            };
            let error = match reply {
                Ok(text) => return Ok(text),
                Err(error) => error,
            };
            // A 401 or a malformed request fails the same way every time.
            if !error.is_retryable() {
                return Err(error.into());
            }
            last = Some(Box::new(error));
        }
    }
}

pub(super) fn classify(error: &(dyn Error + 'static)) -> Signal {
    let Some(error) = error.downcast_ref::<ProviderError>() else {
        return Signal::Failed { retry_after: None };
    };
    let retry_after = retry_after(error);
    match error.provider_response_status().map(|s| s.as_u16()) {
        Some(TOO_MANY_REQUESTS) => Signal::Throttled { retry_after },
        _ => Signal::Failed { retry_after },
    }
}

fn retry_after(error: &ProviderError) -> Option<Duration> {
    let seconds = error
        .provider_response_headers()?
        .get(RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()?;
    Some(Duration::from_secs(seconds))
}
