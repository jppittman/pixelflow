//! desloppify: rule-driven code review.
//!
//! Rules (`rules/*.json`) are typed decisions: a question about a unit of
//! code, a closed set of fine and violation outcomes, and the model levels
//! that decide and explain it. Skills (`skills/<name>/SKILL.md`) are shared
//! knowledge a rule pulls into its prompts. [`review`] plans the calls, asks
//! System One ([`decide`]) and then a model ([`agent`]), and gathers the
//! findings.

pub mod agent;
pub mod decide;
mod language;
pub mod model;
pub mod rate_limit;
pub mod review;
pub mod rule;
pub mod skills;
mod snippet;
