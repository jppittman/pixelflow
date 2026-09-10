//! PixelFlow Pipeline: Training and Data Generation.
//!
//! The measurement harness behind the compiler's cost model: the JIT bench
//! session (`jit_bench`), corpus generation and its quarantine/split/mint
//! plumbing (`training`), and the Guide-program research bins. See
//! docs/plans/2026-07-07-guided-saturation-redesign.md for the architecture,
//! the (2026-07) removal of the RL self-play/critic loop this crate used to
//! host, and docs/plans/2026-09-01-schedule-cost-model-denotation.md (which
//! cites the extraction-head workshop paper, closed without merging as PR
//! #1072 on branch `claude/workshop-writeup`) for the
//! extraction-head program whose shape was deleted on 2026-09-01 and whose
//! denotation (a schedule-cost residual over the table) is kept.

pub mod alloc_probe;
pub mod collapse_bench;
pub mod jit_bench;
pub mod journal;
pub mod poly;
pub mod schema;
pub mod shader_bench;

// Training infrastructure (requires std feature)
#[cfg(feature = "training")]
pub mod training;
