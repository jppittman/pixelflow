//! The agent: a provider, retried under a rate limiter.

use anyhow::Result;
use rig_core::completion::CompletionRequest;
use rig_core::providers::{anthropic::Anthropic, gemini::Gemini};

use crate::model::{ModelLevel, Provider};
use crate::rate_limit::RateLimiter;

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
    /// variable, retrying failed calls as `limiter` allows.
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
        loop {
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
            let Some(wait) = self.limiter.on_error(&error) else {
                return Err(error.into());
            };
            tokio::time::sleep(wait).await;
        }
    }
}
