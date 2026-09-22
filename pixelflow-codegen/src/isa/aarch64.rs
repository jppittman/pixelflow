//! The aarch64 probe. Advanced SIMD (NEON) is part of the base architecture
//! every 64-bit Arm implementation ships, so there is nothing to ask the CPU
//! and one tier to run: `emit/aarch64.rs` uses no extension beyond it.

use super::Isa;

/// Every tier this host can execute: NEON, always.
pub(super) fn runnable() -> Result<&'static [Isa], &'static str> {
    Ok(&[Isa::Neon])
}
