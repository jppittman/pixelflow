//! Training infrastructure for the unified optimization model.
//!
//! This is the live training surface:
//! - binary corpus I/O
//! - expression serialization helpers
//! - unified Rust↔Python replay payloads
//! - unified backward pass
//! - self-play trajectory generation
//! - ES-guided corpus growth

#[cfg(feature = "training")]
pub mod factored;

#[cfg(feature = "training")]
pub mod corpus;

#[cfg(feature = "training")]
pub mod unified;

#[cfg(feature = "training")]
pub mod unified_backward;

#[cfg(feature = "training")]
pub mod self_play;

#[cfg(feature = "training")]
pub mod gen_es;
