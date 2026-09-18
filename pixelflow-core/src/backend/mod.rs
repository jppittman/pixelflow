//! The SIMD backend: what a `Field` *is*.
//!
//! `Field` is one native SIMD vector at the build's width. This module is
//! nothing more than the four concrete lane types — one per ISA level — and
//! the one constructor `Field` keeps from whichever one the build selects:
//! `splat` (broadcast).
//!
//! There is deliberately no trait here. A trait buys polymorphism for code
//! that is generic over the lane type; nothing is — `pixelflow-core/src/lib.rs`
//! picks exactly one concrete type per build via `#[cfg(target_feature)]`, so
//! a `Backend`/`SimdOps` abstraction over "any of the four" had no caller that
//! needed the "any". It used to: this module carried a full SIMD algebra
//! (arithmetic, comparisons, transcendentals) that backed a per-batch
//! "combinator" evaluation tier. That tier is gone
//! (docs/plans/2026-09-06-kernel-with-a-lattice.md) — the JIT emits its own
//! instructions (`pixelflow-codegen`'s per-ISA emitters) and never calls this
//! module — so the algebra it existed for no longer has a reason to exist
//! either. This module and its types are `pub(crate)`: nothing outside this
//! crate should be able to name a lane or a width.

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86;

#[cfg(target_arch = "aarch64")]
pub(crate) mod arm;

pub mod fastmath;
