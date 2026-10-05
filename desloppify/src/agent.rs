//! The agent: a provider behind a rate limiter.

use std::num::NonZeroU32;

use anyhow::Result;
use governor::{DefaultDirectRateLimiter, Quota};
use rig_core::completion::CompletionRequest;
use rig_core::providers::{anthropic::Anthropic, gemini::Gemini};

use crate::model::{ModelLevel, Provider};

/// Enough for a list of findings; the reply is JSON, not prose.
const MAX_REPLY_TOKENS: u64 = 4096;

enum Client {
    Anthropic(Anthropic),
    Gemini(Gemini),
}

pub struct Agent {
    provider: Provider,
    client: Client,
    limiter: DefaultDirectRateLimiter,
}

impl Agent {
    /// An agent for `provider`, credentialed from its usual environment
    /// variable, sending at most `per_minute` requests a minute.
    pub fn from_env(provider: Provider, per_minute: NonZeroU32) -> Result<Self> {
        let client = match provider {
            Provider::Anthropic => Client::Anthropic(Anthropic::from_env()?),
            Provider::Gemini => Client::Gemini(Gemini::from_env()?),
        };
        let limiter = DefaultDirectRateLimiter::direct(Quota::per_minute(per_minute));
        Ok(Self {
            provider,
            client,
            limiter,
        })
    }

    /// Ask the model for `level` to answer `prompt` under `preamble`.
    pub async fn ask(&self, level: ModelLevel, preamble: &str, prompt: &str) -> Result<String> {
        self.limiter.until_ready().await;
        let model = self.provider.model(level);
        let request = CompletionRequest::new(prompt)
            .preamble(preamble)
            .max_tokens(MAX_REPLY_TOKENS);
        let text = match &self.client {
            Client::Anthropic(client) => client.completion(model).call(request).await?.text(),
            Client::Gemini(client) => client.completion(model).call(request).await?.text(),
        };
        Ok(text)
    }
}
