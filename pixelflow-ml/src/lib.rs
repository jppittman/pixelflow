//! # PixelFlow Machine Learning Integration
//!
//! This crate provides machine learning primitives for graphics, specifically
//! harmonic attention and spherical harmonic features.
//!
//! (The compiler optimization logic has moved to `pixelflow-compiler`.)

extern crate alloc;

#[cfg(feature = "graphics")]
pub mod graphics;
