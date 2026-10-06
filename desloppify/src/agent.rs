//! The agent: a provider, paced by a rate limiter.

use anyhow::Result;
use rig_core::completion::CompletionRequest;
use rig_core::providers::{anthropic::Anthropic, gemini::Gemini};

use crate::model::{ModelLevel, Provider};
use crate::rate_limit::{BoxError, RateLimiter};

/// Enough for a list of findings; the reply is JSON, not prose.
const MAX_REPLY_TOKENS: u64 = 4096;

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
