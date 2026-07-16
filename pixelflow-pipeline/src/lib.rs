//! PixelFlow Pipeline: Training and Data Generation.
//!
//! This crate provides the training pipeline for the PixelFlow compiler's
//! cost model: JIT bench harness, corpus generation, and extraction-head
//! (Judge value head) training. See
//! docs/plans/2026-07-07-guided-saturation-redesign.md for the architecture
//! and the (2026-07) removal of the RL self-play/critic loop this crate used
//! to also host.

pub mod jit_bench;

// Training infrastructure (requires std feature)
#[cfg(feature = "training")]
pub mod training;
