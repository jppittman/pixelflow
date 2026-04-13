//! PixelFlow Pipeline: Training and Data Generation.
//!
//! This crate provides the training pipeline for the PixelFlow compiler's
//! optimization system. It uses a two-headed architecture:
//!
//! 1. **Cost Estimation Head**: Predicts expression execution cost (grounded in SIMD benchmarks)
//! 2. **Search Guidance Head**: Estimates if an e-graph path is worth expanding

pub mod fusion;
pub mod jit_bench;

// Training infrastructure (requires std feature)
#[cfg(feature = "training")]
pub mod training;
