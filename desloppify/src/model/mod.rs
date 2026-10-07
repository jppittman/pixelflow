//! Model levels and the providers that resolve them.

use serde::{Deserialize, Serialize};

/// How much model a rule needs, from cheapest to most capable.
///
/// Provider-neutral on purpose: a rule says how hard its question is, and the
/// provider says which model answers questions that hard.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "u64", into = "u64")]
pub enum ModelLevel {
    Lite = 1,
    Fast = 2,
    Strong = 3,
    Frontier = 4,
}

#[derive(Debug, thiserror::Error)]
#[error("model level must be 1-4, got {0}")]
pub struct LevelOutOfRange(u64);

impl TryFrom<u64> for ModelLevel {
    type Error = LevelOutOfRange;

    fn try_from(level: u64) -> Result<Self, Self::Error> {
        match level {
            1 => Ok(Self::Lite),
            2 => Ok(Self::Fast),
            3 => Ok(Self::Strong),
            4 => Ok(Self::Frontier),
            other => Err(LevelOutOfRange(other)),
        }
    }
}

impl ModelLevel {
    /// The next level up; `None` at the top.
    #[must_use]
    pub fn above(self) -> Option<Self> {
        match self {
            Self::Lite => Some(Self::Fast),
            Self::Fast => Some(Self::Strong),
            Self::Strong => Some(Self::Frontier),
            Self::Frontier => None,
        }
    }
}

impl From<ModelLevel> for u64 {
    fn from(level: ModelLevel) -> Self {
        level as u64
    }
}

/// Who answers the prompts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Provider {
    Anthropic,
    Gemini,
}

impl Provider {
    /// The model this provider uses for `level`.
    #[must_use]
    pub fn model(self, level: ModelLevel) -> &'static str {
        match (self, level) {
            (Self::Anthropic, ModelLevel::Lite) => "claude-haiku-4-5",
            (Self::Anthropic, ModelLevel::Fast) => "claude-sonnet-5-5",
            (Self::Anthropic, ModelLevel::Strong) => "claude-opus-5-5",
            (Self::Anthropic, ModelLevel::Frontier) => "claude-fable-5-1",
            (Self::Gemini, ModelLevel::Lite) => "gemini-3.1-flash-lite-preview",
            (Self::Gemini, ModelLevel::Fast) => "gemini-3-flash-preview",
            (Self::Gemini, ModelLevel::Strong) => "gemini-2.5-pro",
            (Self::Gemini, ModelLevel::Frontier) => "gemini-3-pro-preview",
        }
    }
}
