//! Training infrastructure for the extraction (Judge) head.
//!
//! This is the live training surface:
//! - binary corpus I/O
//! - expression serialization helpers
//! - value-head backward pass (`forward_cached` / `backward_value` /
//!   `backward_through_accumulator`)
//! - budget-bounded episode generation (`episodes`)
//!
//! The RL apparatus this module used to host (self-play mask policy,
//! REINFORCE `backward_policy`, PFTJ/PFAD trajectory export, the ES-guided
//! corpus-growth optimizer) was removed per
//! docs/plans/2026-07-07-guided-saturation-redesign.md — see that doc for
//! the full post-mortem and the supervised, hindsight-labeled replacement
//! architecture.

#[cfg(feature = "training")]
pub mod factored;

#[cfg(feature = "training")]
pub mod corpus;

#[cfg(feature = "training")]
pub mod unified_backward;

#[cfg(feature = "training")]
pub mod episodes;
