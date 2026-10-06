//! The agent: a provider, paced by a rate limiter.

use std::error::Error;
use std::time::Duration;

use anyhow::Result;
use rig_core::ProviderError;
use rig_core::completion::CompletionRequest;
use rig_core::providers::{anthropic::Anthropic, gemini::Gemini};

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

pub struct Agent {
    provider: Provider,
    client: Client,
    limiter: Box<dyn RateLimiter>,
}

impl Agent {
    /// An agent for `provider`, credentialed from its usual environment
    /// variable, calling and retrying as `limiter` allows.
    pub fn from_env(provider: Provider, limiter: Box<dyn RateLimiter>) -> Result<Self> {
        let client = match provider {
            Provider::Anthropic => Client::Anthropic(Anthropic::from_env()?),
            Provider::Gemini => Client::Gemini(Gemini::from_env()?),
        };
        Ok(Self {
            provider,
            client,
            limiter,
        })
    }

    /// Ask the model for `level` to answer `prompt` under `preamble`.
    pub async fn ask(&self, level: ModelLevel, preamble: &str, prompt: &str) -> Result<String> {
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

/// What a provider's failure says about the call rate: a 429 is a throttle,
/// anything else (an outage, a timeout) is not. Either may carry the
/// seconds form of `Retry-After`; the date form is rare enough from model
/// providers to ignore.
#[must_use]
pub fn classify(error: &(dyn Error + 'static)) -> Signal {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn reply(status: u16, retry_after: Option<&str>) -> BoxError {
        let status = http::StatusCode::from_u16(status).unwrap();
        let headers = retry_after.map(|value| {
            let mut headers = http::HeaderMap::new();
            headers.insert(RETRY_AFTER, value.parse().unwrap());
            headers
        });
        Box::new(ProviderError::from_http_response(status, "").with_response_headers(headers))
    }

    #[test]
    fn a_429_is_a_throttle_with_its_retry_after() {
        assert_eq!(
            classify(&*reply(429, Some("7"))),
            Signal::Throttled {
                retry_after: Some(Duration::from_secs(7))
            }
        );
        assert_eq!(
            classify(&*reply(429, Some("Wed, 21 Oct 2015 07:28:00 GMT"))),
            Signal::Throttled { retry_after: None }
        );
    }

    #[test]
    fn an_outage_is_a_failure() {
        assert_eq!(
            classify(&*reply(503, None)),
            Signal::Failed { retry_after: None }
        );
        let other: BoxError = Box::new(std::io::Error::other("reset"));
        assert_eq!(classify(&*other), Signal::Failed { retry_after: None });
    }
}
