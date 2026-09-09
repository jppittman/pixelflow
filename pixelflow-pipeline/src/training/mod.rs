//! Corpus and label plumbing shared by every measurement bin.
//!
//! - binary corpus I/O (`corpus`)
//! - the tiered split manifest (`split`, plan 0.2)
//! - expression serialization helpers (`factored`)
//! - the literal-blind structural key both fences quotient by (`structural`)
//! - the numeric quarantine every label source must pass (`quarantine`,
//!   plan 0.4) — shared so every corpus build applies one predicate, not two
//! - label minting: sentinel drift normalization (`mint`)
//!
//! Two training loops used to live here and are gone. The RL apparatus
//! (self-play mask policy, REINFORCE `backward_policy`, PFTJ/PFAD trajectory
//! export, the ES-guided corpus-growth optimizer) was removed per
//! docs/plans/2026-07-07-guided-saturation-redesign.md. The supervised
//! extraction-head trainer that replaced it (value-head backward pass,
//! budget-bounded episodes, the bootstrap binary) was closed as an honest
//! negative — workshop paper on branch `claude/workshop-writeup`, PR #1072,
//! closed without merging (see docs/plans/2026-09-01-schedule-cost-model-denotation.md)
//! — and deleted on 2026-09-01; its history is in VCS.

#[cfg(feature = "training")]
pub mod factored;

#[cfg(feature = "training")]
pub mod corpus;

#[cfg(feature = "training")]
pub mod split;

#[cfg(feature = "training")]
pub mod mint;

#[cfg(feature = "training")]
pub mod structural;

#[cfg(feature = "training")]
#[cfg(feature = "training")]
pub mod guide_linear;

#[cfg(feature = "training")]
pub mod guide_bilinear;

#[cfg(feature = "training")]
pub mod sh_family;

#[cfg(feature = "training")]
pub mod bezier_family;

#[cfg(feature = "training")]
pub mod r2g;
