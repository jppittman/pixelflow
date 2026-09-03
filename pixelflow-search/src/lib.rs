#![allow(clippy::all)]
#![allow(warnings)]
#![allow(unused)]
extern crate alloc;

#[cfg(test)]
mod arena_corpus;
#[cfg(test)]
mod cost_table_ab;
pub mod egraph;
#[cfg(test)]
mod extraction_gap;
pub mod math;
pub mod nnue;
pub mod runtime;
#[cfg(feature = "saturation-telemetry")]
pub mod telemetry;
