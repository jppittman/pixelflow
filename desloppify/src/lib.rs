//! desloppify: rule-driven code review.
//!
//! Rules (`rules/*.json`) say what code to look at — a tree-sitter query, or
//! whole files — how capable a model the question needs (level 1-4), and what
//! to ask. Skills (`skills/<name>/SKILL.md`) are shared knowledge a rule pulls
//! into its prompt. Each matched snippet is one rate-limited agent call.

pub mod agent;
pub mod decide;
pub mod language;
pub mod model;
pub mod rate_limit;
pub mod review;
pub mod rule;
pub mod skills;
pub mod snippet;
