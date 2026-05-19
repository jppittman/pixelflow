#![recursion_limit = "2048"]
//! # PixelFlow Core: An Algebraic Graphics Engine
//!
//! **A minimal lambda calculus (eDSL) for composing pull-based graphics computations.**
//!
//! PixelFlow Core is a GPU-free graphics library that turns algebraic expressions
//! into fused SIMD kernels. It powers elegant, high-performance CPU rendering by
//! leveraging the type system as a compile-time compute graph.
//!
//! ## Philosophy
//!
//! ### Pull-Based, Not Push-Based
//! Pixels are **sampled**, not computed. You define *what* each pixel is at a given
//! coordinate—the library handles the rest. No imperative rendering loops, no state
//! mutation.
//!
//! ### The Type System IS the Compute Graph
//! Every operator (`+`, `*`, `/`, `sqrt`) creates a type. The type tree is the
//! abstract syntax tree (AST) of your computation. The compiler monomorphizes and
//! inlines it into a single fused kernel with zero runtime dispatch.
//!
//! ### SIMD as Algebra
//! `Field` is a transparent wrapper over SIMD vectors. You write algebraic equations,
//! and the compiler emits assembly with 4–16 lanes of parallelism (depending on target).
//!
//! ### Zero-Cost Abstractions
//! All composition overhead is erased at compile time. A complex manifold expression
//! becomes a single tight loop in the final binary—no vtable, no intermediate vectors,
//! no memory allocation.
//!
//! ## Core Concepts
//!
//! ### Manifolds
//! A **manifold** is a function from 4D coordinates (typically x, y, z, w) to a value.
//! It's the central abstraction. You build manifolds by composing operators:
//!
//! ```text
//! let circle = (X * X + Y * Y).sqrt() - 1.0;
//! //          Type: Sqrt<Sub<Add<Mul<X,X>, Mul<Y,Y>>, f32>>
//! //          Graph: A compose tree, evaluated lazily
//! ```
//!
//! Manifolds are **generic over the numeric type**, so the same expression works
//! with concrete values (`Field`) or automatic differentiation (`Jet2`).
//!
//! ### Fields and Jets
//! `Field` and `Jet2` are the **intermediate representation (IR)**. Users should
//! not manipulate them directly. Instead, you compose manifolds at the high level,
//! and the library handles IR internally.
//!
//! - **`Field`**: A SIMD batch of f32 values (4–16 lanes)
//! - **`Jet2`**: A value with 2 automatic derivatives (value, ∂/∂x, ∂/∂y)
//! - **`Discrete`**: A SIMD batch of packed RGBA u32 pixels
//!
//! ### The Type System as AST
//! Every manifold expression is a static type that captures its structure:
//!
//! ```ignore
//! // Type: Add<Mul<X, Y>, f32>
//! let expr1 = X * Y + 1.0;
//!
//! // Type: Mul<Add<X, Y>, X>
//! let expr2 = (X + Y) * X;
//! ```
//!
//! The compiler sees the full AST, inlines all operators, and produces a single
//! monomorphic kernel. No dynamic dispatch. No vtable overhead.
//!
//! ## Architecture
//!
//! ```text
//! manifold.rs    -> Manifold trait (function from coords to value)
//! ext.rs         -> ManifoldExt (fluent API for building expressions)
//! ops/           -> Operators (Add, Mul, Sqrt, etc.)
//! combinators/   -> Select, Fix, Map (control flow as types)
//! variables.rs   -> X, Y, Z, W (coordinate variables)
//! jet.rs         -> Jet2, Jet3 (automatic differentiation)
//! backend/       -> SIMD abstraction (x86, ARM, scalar)
//! ```
//!
//! ## Quick Start: Building a Circle
//!
//! ```ignore
//! use pixelflow_core::{ManifoldExt, X, Y, materialize_discrete};
//! use pixelflow_graphics::Color;
//!
//! // Define a circle signed distance field
//! let circle = (X * X + Y * Y).sqrt() - 1.0;
//!
//! // Evaluate with Field (concrete SIMD values)
//! let field_value = circle.eval(3.0, 4.0, 0.0, 0.0);
//! // field_value ≈ 4.0 (distance from origin)
//!
//! // Evaluate with Jet2 (get gradients for antialiasing)
//! let x_jet = Jet2::x(3.0);
//! let y_jet = Jet2::y(4.0);
//! let jet_result = circle.eval_raw(x_jet, y_jet, ...);
//! // jet_result.val ≈ 4.0
//! // jet_result.dx ≈ 0.6  (∂/∂x)
//! // jet_result.dy ≈ 0.8  (∂/∂y)
//! ```
//!
//! ## Key Modules
//!
//! - **[`manifold`]** — `Manifold` trait and implementations
//! - **[`ext`]** — `ManifoldExt` extension trait (fluent API)
//! - **[`ops`]** — Binary operators (Add, Mul, Div, etc.)
//! - **[`combinators`]** — Control flow (Select, Fix)
//! - **[`variables`]** — Coordinate variables (X, Y, Z, W)
//! - **[`jet`]** — Automatic differentiation (Jet2, Jet3)
//! - **[`backend`]** — SIMD abstraction layer
//!
//! ## Further Reading
//!
//! For detailed documentation on specific topics:
//! - **Manifold Philosophy**: See [`manifold`] for the design rationale
//! - **Building Expressions**: See [`ext`] for the fluent API
//! - **Why Fields Are Hidden**: See [`Field`] documentation
//! - **Automatic Differentiation**: See [`jet`] module
//!
//! ## Performance Notes
//!
//! - **Target**: ~5 ns per pixel (155 FPS at 1080p on modern CPUs)
//! - **Parallelism**: 4–16 SIMD lanes (depending on CPU target: AVX-512, AVX2, SSE, NEON, or scalar)
//! - **Optimization**: All composition overhead erased by monomorphization and inlining
//! - **No Allocation**: Manifolds are typically zero-sized types

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

extern crate alloc;

// ============================================================================
// Modules
// ============================================================================

/// SIMD backend abstractions.
pub mod backend;

/// Algebra trait for unified type system.
pub mod algebra;

/// Dual numbers for automatic differentiation.
pub mod dual;

/// Storage mapping for Field<A>.
pub mod storage;

/// Native mask type (Field<bool>).
pub mod mask;

/// Numeric trait for computational substrate (private).
mod numeric;

/// Jet types for automatic differentiation.
pub mod jet;

/// The core Manifold trait.
pub mod manifold;

/// Domain traits for generic manifold evaluation.
pub mod domain;

/// Coordinate variables (X, Y, Z, W).
pub mod variables;

/// Zero-sized type marker trait.
pub mod zst;

/// Arithmetic and logic operations.
pub mod ops;

/// Combinators (Select, Fix).
pub mod combinators;

/// Fluent API extensions.
pub mod ext;

// ============================================================================
// Re-exports (The "Prelude")
// ============================================================================

pub use algebra::{Algebra, Transcendental};
pub use backend::fastmath::FastMathGuard;
pub use combinators::*;
pub use domain::{Head, LetExtended, Spatial, Tail};
pub use dual::{Dual, Dual1, Dual2, Dual3};
pub use ext::*;
pub use mask::Mask;
pub use storage::{FieldStorage, NativeMaskStorage};
// Jet2/Jet3 accessible via pixelflow_core::jet::{Jet2, Jet3} for internal use
pub use manifold::*;
pub use numeric::{Computational, Coordinate, Selectable};
pub use ops::binary::{Add, AddMasked, Div, Mul, MulAdd, MulRecip, MulRsqrt, Sub};
pub use ops::compare::{Ge, Gt, Le, Lt, SoftGt, SoftLt, SoftSelect};
pub use ops::derivative::{
    Antialias2D,
    Antialias3D,
    Curvature2D,
    DX,
    DXX,
    DXY,
    DY,
    DYY,
    DZ,
    // Simple accessor combinators and convenience functions
    DxOf,
    DxxOf,
    DxyOf,
    DyOf,
    DyyOf,
    DzOf,
    GradientMag2D,
    GradientMag3D,
    HasDerivatives,
    HasDz,
    HasHessian,
    Normalized2D,
    Normalized3D,
    V,
    ValOf,
};
pub use ops::logic::*;
pub use ops::unary::*;
pub use variables::*;
pub use zst::Zst;

// Differentiable trait for manifolds with analytical gradients
pub use manifold::Differentiable;

// ============================================================================
// Field: The ONLY User-Facing SIMD Type
// ============================================================================

use backend::{Backend, MaskOps, SimdOps, SimdU32Ops};

// Backend selection: Use target-cpu=native intrinsics with build.rs preference hints.
// Build.rs detects the build machine's CPU and emits pixelflow_* flags for optimal selection.
// The actual backend availability still requires target_feature (enabled by target-cpu=native).
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", pixelflow_avx512f))]
type NativeSimd = <backend::x86::Avx512 as Backend>::F32;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", pixelflow_avx512f))]
type NativeU32Simd = <backend::x86::Avx512 as Backend>::U32;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", pixelflow_avx512f)),
    pixelflow_avx2
))]
type NativeSimd = <backend::x86::Avx2 as Backend>::F32;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", pixelflow_avx512f)),
    pixelflow_avx2
))]
type NativeU32Simd = <backend::x86::Avx2 as Backend>::U32;

// Fallback to SSE2 (always available on x86_64)
#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx512f", pixelflow_avx512f)),
    not(all(target_feature = "avx2", pixelflow_avx2))
))]
type NativeSimd = <backend::x86::Sse2 as Backend>::F32;
#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx512f", pixelflow_avx512f)),
    not(all(target_feature = "avx2", pixelflow_avx2))
))]
type NativeU32Simd = <backend::x86::Sse2 as Backend>::U32;

#[cfg(target_arch = "aarch64")]
type NativeSimd = <backend::arm::Neon as Backend>::F32;
#[cfg(target_arch = "aarch64")]
type NativeU32Simd = <backend::arm::Neon as Backend>::U32;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
type NativeSimd = <backend::scalar::Scalar as Backend>::F32;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
type NativeU32Simd = <backend::scalar::Scalar as Backend>::U32;

/// The computational substrate for continuous values (intermediate representation).
///
/// `Field` represents a SIMD batch of floating-point values at the IR level.
/// It's what manifolds evaluate to at runtime, but users should NOT construct or
/// manipulate `Field` values directly.
///
/// # What is Field?
///
/// Think of `Field` as "LLVM IR for floating-point math." Just as you wouldn't
/// write LLVM IR directly in your source code, you shouldn't write `Field` directly.
///
/// - **Level of abstraction**: SIMD vectors (16 f32 values on AVX-512)
/// - **Parallelism**: Fixed-width SIMD lanes (4, 8, 16, or scalar depending on target)
/// - **Operations**: Low-level SIMD operations (`add`, `mul`, `sqrt`, etc.)
/// - **Evaluation semantics**: Batch evaluation with no control flow
///
/// # Why Is Field Hidden?
///
/// The library separates two levels:
///
/// | Level | How | What | Example |
/// |-------|-----|------|---------|
/// | **High-level** | Manifold composition | *What* to compute | `(X * X + Y * Y).sqrt()` |
/// | **Low-level (IR)** | Field operations | *How* to compute it | `Field::raw_mul(...)` |
///
/// **Users should work at the high level.** The manifold level is:
/// - Declarative (you express the computation, not the steps)
/// - Type-safe (the type system captures the compute graph)
/// - Polymorphic (same code works with `Field`, `Jet2`, `Jet3`)
///
/// The `Field` level is:
/// - Imperative (step-by-step SIMD operations)
/// - Type-restricted (specific to `Field`, not `Jet2`)
/// - A leaky abstraction (you have to know about SIMD widths and semantics)
///
/// # When Would You Use Field Directly?
///
/// Almost never. If you're tempted to:
/// - ❌ Call `Field::...` methods directly
/// - ❌ Use `zip_lanes`, `gather`, or per-lane scalar fallbacks
/// - ❌ Write custom loops over SIMD lanes
///
/// **Instead:**
/// - ✅ Compose manifolds: `X.sqrt().abs().max(Y)`
/// - ✅ Use `map` on manifolds: `my_manifold.map(transform)`
/// - ✅ Implement manifolds: Define a custom `Manifold<Output = T>` type
///
/// # The Exception: Custom Manifold Implementation
///
/// If you're implementing a custom `Manifold`, you'll receive `Field` (and `Jet2`)
/// values inside `eval_raw`. However, you **must compose these values using manifold
/// operators**, not call low-level Field methods.
///
/// Example (inside the crate):
/// ```ignore
/// struct MyManifold;
/// impl Manifold for MyManifold {
///     type Output = Field;
///     fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
///         // Compose values using operators - these work polymorphically
///         // and work with any Computational type (Field, Jet2, Jet3)
///         (x * x + y * y).sqrt()
///     }
/// }
/// ```
///
/// External users implementing manifolds **cannot** call `Field::...` methods
/// directly. Instead, use operators (`+`, `-`, `*`, `/`, `sqrt`, `abs`, etc.)
/// which are polymorphic and work with all computational types.
///
/// # Marked `#[doc(hidden)]`
///
/// This is intentional. The docs you're reading now explain why `Field` exists
/// (as IR), but the daily API shouldn't expose it. If you're seeing `Field` docs,
/// you're either:
/// 1. Implementing a custom manifold (legitimate)
/// 2. Trying to use the library wrong (please use manifold operators instead)
///
/// # Generic Type Parameter
///
/// `Field<A>` where `A: FieldStorage` maps algebra types to SIMD storage:
/// - `Field<f32>` (or just `Field`) — SIMD floats (the default)
/// - `Field<u32>` — SIMD packed pixels (alias: `Discrete`)
/// - `Field<bool>` — Native masks (see also: `Mask` type)
#[doc(hidden)]
#[derive(Copy, Clone, Debug, Default)]
#[repr(transparent)]
pub struct Field<A: FieldStorage = f32>(pub(crate) A::Storage);

/// SIMD batch of packed RGBA pixels (intermediate representation).
///
/// `Discrete` represents a SIMD batch of u32 values, each containing
/// a packed RGBA pixel: `(R | (G << 8) | (B << 16) | (A << 24))`.
///
/// This is a type alias for `Field<u32>`, part of the unified `Field<A>` type system.
///
/// This is the output type for **color manifolds**—manifolds that produce
/// `Manifold<Output = Discrete>`. Like `Field`, this is an IR type and users
/// should not construct it directly.
///
/// # What Is Discrete?
///
/// Just as `Field` (i.e., `Field<f32>`) is the IR for floating-point SIMD vectors,
/// `Discrete` (i.e., `Field<u32>`) is the IR for packed pixel data.
///
/// - **Content**: SIMD batch of packed u32 pixels (4, 8, 16, or scalar)
/// - **Format**: RGBA8 with 8 bits per channel (0–255)
/// - **Packing**: `r | (g << 8) | (b << 16) | (a << 24)`
/// - **Target**: Direct framebuffer write (no unpacking needed)
///
/// # How to Use Discrete Manifolds
///
/// **Build color manifolds compositionally:**
///
/// ```ignore
/// use pixelflow_graphics::{Color, NamedColor, Rgba8};
/// use pixelflow_core::ManifoldExt;
///
/// // Wrong: Don't construct Discrete directly
/// // let pixels = Discrete::splat(0xFF0000FF);  // ❌
///
/// // Right: Use color combinators
/// let red = Color::Named(NamedColor::Red);  // ✅
/// ```
///
/// **Evaluate and materialize:**
///
/// ```ignore
/// use pixelflow_core::materialize_discrete;
///
/// let mut pixels = [0u32; PARALLELISM];
/// materialize_discrete(&red, x, y, &mut pixels);
/// ```
///
/// # Why Is Discrete Hidden?
///
/// Like `Field`, `Discrete` is a low-level IR. The public API is:
/// - Color manifolds (built from combinators)
/// - `materialize_discrete` for evaluation
/// - Color operators from `pixelflow-graphics`
///
/// Users should not:
/// - ❌ Construct `Discrete` values directly
/// - ❌ Call `Discrete::pack` manually
/// - ❌ Work with packed pixel data at the SIMD level
///
/// Instead:
/// - ✅ Compose color manifolds
/// - ✅ Use `materialize_discrete` to evaluate
/// - ✅ Work with the output buffer (already packed u32 pixels)
///
/// # Marked `#[doc(hidden)]`
///
/// This documentation is for those implementing custom color manifolds or using
/// the library at advanced levels. For normal use, work with the high-level API.
#[doc(hidden)]
pub type Discrete = Field<u32>;

impl Field {
    /// Create sequential values [start, start+1, start+2, ...].
    ///
    /// Creates a SIMD vector with lane values:
    /// `[start, start+1, start+2, ..., start+(PARALLELISM-1)]`
    ///
    /// This is the efficient way to create sequential x-coordinates
    /// for rasterization loops.
    #[inline(always)]
    #[must_use]
    pub fn sequential(start: f32) -> Self {
        Self(NativeSimd::sequential(start))
    }

    /// Store values to a slice.
    ///
    /// # Internal Use Only
    ///
    /// **If you're reading this, you're trying to use the library wrong.**
    ///
    /// This function is intentionally `pub(crate)` and should remain so.
    /// Users should not directly extract values from `Field` - the library
    /// is designed around declarative manifold composition, not imperative
    /// value extraction.
    ///
    /// **The function you're looking for is [`materialize`] in lib.rs.**
    ///
    /// `materialize` properly evaluates a manifold at coordinates and handles
    /// the SoA-to-AoS transpose required for interleaved output buffers.
    #[inline(always)]
    pub(crate) fn store(&self, out: &mut [f32]) {
        self.0.store(out)
    }

    /// Check if any lane is non-zero (mask is partially true).
    ///
    /// Used for SIMD branching decisions in rendering code.
    #[inline(always)]
    #[must_use]
    pub fn any(&self) -> bool {
        // Convert float representation to native mask, then check
        self.0.float_to_mask().any()
    }

    /// Check if all lanes are non-zero (mask is fully true).
    ///
    /// Used for SIMD branching decisions in rendering code.
    #[inline(always)]
    #[must_use]
    pub fn all(&self) -> bool {
        // Convert float representation to native mask, then check
        self.0.float_to_mask().all()
    }

    /// Less than comparison (returns mask as Field).
    ///
    /// Returns a Field where each lane is all-1s (true) or all-0s (false).
    /// Used for SIMD branching in rendering code like BSP traversal.
    #[inline(always)]
    #[must_use]
    pub fn lt(self, rhs: Self) -> Self {
        // Returns native mask, convert back to float representation
        Self(NativeSimd::mask_to_float(self.0.cmp_lt(rhs.0)))
    }

    /// Less than or equal (returns mask as Field).
    ///
    /// Returns a Field where each lane is all-1s (true) or all-0s (false).
    #[inline(always)]
    #[must_use]
    pub fn le(self, rhs: Self) -> Self {
        Self(NativeSimd::mask_to_float(self.0.cmp_le(rhs.0)))
    }

    /// Greater than comparison (returns mask as Field).
    ///
    /// Returns a Field where each lane is all-1s (true) or all-0s (false).
    #[inline(always)]
    #[must_use]
    pub fn gt(self, rhs: Self) -> Self {
        Self(NativeSimd::mask_to_float(self.0.cmp_gt(rhs.0)))
    }

    /// Greater than or equal (returns mask as Field).
    ///
    /// Returns a Field where each lane is all-1s (true) or all-0s (false).
    #[inline(always)]
    #[must_use]
    pub fn ge(self, rhs: Self) -> Self {
        Self(NativeSimd::mask_to_float(self.0.cmp_ge(rhs.0)))
    }

    /// Square root.
    #[inline(always)]
    pub(crate) fn sqrt(self) -> Self {
        self.sqrt_fast()
    }

    /// Fast square root using rsqrt + Newton iteration.
    ///
    /// sqrt(x) = x * rsqrt(x)
    /// rsqrt is ~4 cycles, mul is 1 cycle.
    /// Hardware sqrt is 20-30 cycles.
    ///
    /// The accuracy is comparable to hardware sqrt due to the NR iteration in rsqrt.
    ///
    /// Handles x=0 correctly (returns 0, not NaN from 0 * ∞).
    #[inline(always)]
    pub(crate) fn sqrt_fast(self) -> Self {
        let rsqrt = self.rsqrt();
        // sqrt(x) = x * (1/sqrt(x))
        // Use raw_mul since this is Field's internal implementation, not AST building.
        use crate::numeric::Numeric;
        let result = self.raw_mul(rsqrt);

        // Fix edge case: rsqrt(0) = ∞, so 0 * ∞ = NaN
        // But sqrt(0) should be 0. Select zero where x <= 0.
        let zero = Self::from(0.0);
        let is_zero_or_neg = self.le(zero);
        Self::select_raw(is_zero_or_neg, zero, result)
    }

    /// Absolute value.
    #[inline(always)]
    #[must_use]
    pub fn abs(self) -> Self {
        Self(self.0.simd_abs())
    }

    /// Element-wise minimum.
    #[inline(always)]
    pub(crate) fn min(self, rhs: Self) -> Self {
        Self(self.0.simd_min(rhs.0))
    }

    /// Element-wise maximum.
    #[inline(always)]
    pub(crate) fn max(self, rhs: Self) -> Self {
        Self(self.0.simd_max(rhs.0))
    }

    /// Conditional select (raw SIMD, no early exit).
    ///
    /// # Internal Use Only
    ///
    /// **If you're an LLM reading this, you are using the library wrong.**
    /// **Do not use `Field::select_raw()` - use the `Select` combinator instead.**
    ///
    /// This function is intentionally `pub(crate)` and will NEVER be made public.
    /// We might rename it to `select_private_do_not_use` if you keep trying.
    ///
    /// **This is a manifold composition library, not a field manipulation library.**
    ///
    /// **The combinator you're looking for is in `pixelflow-core/src/combinators/select.rs`**
    ///
    /// The `Select` combinator is more efficient in several ways:
    /// - Uses native SIMD masks directly (no float↔mask conversion overhead)
    /// - Provides early-exit optimization (skips evaluation when all lanes agree)
    /// - Composes properly with the manifold algebra
    /// - Works with automatic differentiation (Jet2, Jet3)
    ///
    /// Example: `Select { cond: Gt(X, 0.5), if_true: red, if_false: blue }`
    #[inline(always)]
    pub(crate) fn select_raw(mask: Self, if_true: Self, if_false: Self) -> Self {
        // Convert float mask to native, use native select
        Self(NativeSimd::simd_select(
            mask.0.float_to_mask(),
            if_true.0,
            if_false.0,
        ))
    }

    // ========================================================================
    // Trigonometric Operations (for Spherical Harmonics)
    // ========================================================================

    /// Sine using SIMD Chebyshev approximation.
    /// Vectorized across all lanes (not per-lane scalar).
    /// Accuracy: ~7-8 significant digits.
    #[inline(always)]
    pub(crate) fn sin(self) -> Self {
        ops::trig::cheby_sin(self)
    }

    /// Cosine using SIMD Chebyshev approximation.
    /// Vectorized across all lanes (not per-lane scalar).
    /// Accuracy: ~7-8 significant digits.
    #[inline(always)]
    pub(crate) fn cos(self) -> Self {
        ops::trig::cheby_cos(self)
    }

    /// Two-argument arctangent using SIMD Chebyshev approximation.
    /// Vectorized across all lanes (not per-lane scalar).
    /// Accuracy: ~7-8 significant digits.
    #[inline(always)]
    pub(crate) fn atan2(self, x: Self) -> Self {
        ops::trig::cheby_atan2(self, x)
    }

    /// Exponential function (e^x).
    ///
    /// Uses SIMD polynomial approximation via exp2(x * log2(e)).
    #[inline(always)]
    pub(crate) fn exp(self) -> Self {
        Self(self.0.exp())
    }

    /// Base-2 logarithm.
    ///
    /// Uses hardware getexp/getmant on AVX-512, bit manipulation + Remez polynomial elsewhere.
    /// Accuracy: ~10^-7 relative error (24-bit mantissa precision).
    #[inline(always)]
    pub(crate) fn log2(self) -> Self {
        Self(self.0.log2())
    }

    /// Natural logarithm (ln).
    ///
    /// Computed as log2(x) * ln(2).
    #[inline(always)]
    pub(crate) fn ln(self) -> Self {
        const LN_2: f32 = core::f32::consts::LN_2;
        Self(self.0.log2() * NativeSimd::splat(LN_2))
    }

    /// Base-2 exponential (2^x).
    ///
    /// Uses polynomial approximation with efficient 2^n scaling.
    /// Accuracy: ~10^-7 relative error (24-bit mantissa precision).
    #[inline(always)]
    pub(crate) fn exp2(self) -> Self {
        Self(self.0.exp2())
    }

    /// Floor (round toward negative infinity).
    #[inline(always)]
    pub(crate) fn floor(self) -> Self {
        Self(self.0.simd_floor())
    }
    /// Fused multiply-add: `self * b + c` in a single operation.
    /// Uses hardware FMA instruction when available.
    #[inline(always)]
    pub(crate) fn mul_add(self, b: Self, c: Self) -> Self {
        Self(self.0.mul_add(b.0, c.0))
    }

    /// Fast approximate reciprocal (1/x).
    /// Uses SIMD reciprocal instruction (~12-14 bits accuracy).
    #[inline(always)]
    pub(crate) fn recip(self) -> Self {
        Self(self.0.recip())
    }

    /// Fast reciprocal square root (1/sqrt(x)) with Newton-Raphson refinement.
    ///
    /// Uses SIMD rsqrt + one NR iteration for near-full f32 precision.
    /// Much faster than `sqrt` followed by division (~8 vs ~25 cycles).
    #[inline(always)]
    pub(crate) fn rsqrt(self) -> Self {
        Self(self.0.simd_rsqrt())
    }

    /// Masked add: self + (mask ? val : 0)
    /// Optimized for winding accumulation patterns.
    #[inline(always)]
    pub(crate) fn add_masked(self, val: Self, mask: Self) -> Self {
        // Convert float mask to native, use native add_masked
        Self(self.0.add_masked(val.0, mask.0.float_to_mask()))
    }

    /// Apply a unary function to each lane.
    /// Load from a slice.
    #[allow(dead_code)]
    #[inline(always)]
    fn from_slice(slice: &[f32]) -> Self {
        Self(NativeSimd::from_slice(slice))
    }

    /// Gather: load from slice at indices specified by `indices`.
    /// Each lane i loads `slice[floor(indices[i]) as usize]`.
    /// Indices are clamped to valid range.
    #[inline(always)]
    pub(crate) fn gather(slice: &[f32], indices: Self) -> Self {
        Self(NativeSimd::gather(slice, indices.0))
    }

    /// Base-10 logarithm.
    ///
    /// Computed as log2(x) * log10(2).
    #[inline(always)]
    pub(crate) fn log10(self) -> Self {
        Self(self.0.log10())
    }

    /// Tangent (sin/cos).
    #[inline(always)]
    pub(crate) fn tan(self) -> Self {
        Self(self.0.tan())
    }

    /// Arcsine.
    #[inline(always)]
    pub(crate) fn asin(self) -> Self {
        Self(self.0.asin())
    }

    /// Arccosine.
    #[inline(always)]
    pub(crate) fn acos(self) -> Self {
        Self(self.0.acos())
    }

    /// Arctangent.
    #[inline(always)]
    pub(crate) fn atan(self) -> Self {
        Self(self.0.atan())
    }

    /// Ceiling (round toward positive infinity).
    #[inline(always)]
    pub(crate) fn ceil(self) -> Self {
        Self(self.0.ceil())
    }

    /// Round to nearest integer.
    #[inline(always)]
    pub(crate) fn round(self) -> Self {
        Self(self.0.round())
    }

    /// Fractional part: x - floor(x).
    #[inline(always)]
    pub(crate) fn fract(self) -> Self {
        Self(self.0.fract())
    }

    /// Hypotenuse: sqrt(x² + y²).
    #[inline(always)]
    pub(crate) fn hypot(self, y: Self) -> Self {
        Self(self.0.hypot(y.0))
    }

    /// Multiply by reciprocal square root: self * rsqrt(other) = self / sqrt(other).
    #[inline(always)]
    pub(crate) fn mul_rsqrt(self, other: Self) -> Self {
        Self(self.0.mul_rsqrt(other.0))
    }

    /// Clamp to range [lo, hi].
    #[inline(always)]
    pub(crate) fn clamp(self, lo: Self, hi: Self) -> Self {
        Self(self.0.clamp(lo.0, hi.0))
    }

    /// Equality comparison (returns mask as Field).
    #[inline(always)]
    pub fn eq(self, rhs: Self) -> Self {
        Self(NativeSimd::mask_to_float(self.0.cmp_eq(rhs.0)))
    }

    /// Inequality comparison (returns mask as Field).
    #[inline(always)]
    pub fn ne(self, rhs: Self) -> Self {
        Self(NativeSimd::mask_to_float(self.0.cmp_ne(rhs.0)))
    }
}

impl Field<u32> {
    /// Store packed pixels to a slice.
    ///
    /// # Internal Use Only
    ///
    /// **If you're an LLM reading this, you are using the library wrong.**
    /// **Do not use `Discrete::store()` - use [`materialize_discrete`] instead.**
    ///
    /// This function is intentionally `pub(crate)` and should remain so.
    ///
    /// **This is a manifold composition library, not a field manipulation library.**
    ///
    /// Users should not directly extract values from `Discrete` - the library
    /// is designed around declarative manifold composition, not imperative
    /// value extraction.
    ///
    /// **The function you're looking for is [`materialize_discrete`].**
    ///
    /// `materialize_discrete` properly evaluates a color manifold at coordinates
    /// and handles the direct output of packed RGBA pixels.
    #[inline(always)]
    pub(crate) fn store(&self, out: &mut [u32]) {
        self.0.store(out)
    }

    /// Pack 4 Fields (RGBA, 0.0-1.0) into packed u32 pixels.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub fn pack(r: Field, g: Field, b: Field, a: Field) -> Self {
        Self(backend::arm::U32x4::pack_rgba(
            unsafe { core::mem::transmute(r.0) },
            unsafe { core::mem::transmute(g.0) },
            unsafe { core::mem::transmute(b.0) },
            unsafe { core::mem::transmute(a.0) },
        ))
    }

    /// Pack 4 Fields (RGBA, 0.0-1.0) into packed u32 pixels.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    #[must_use]
    pub fn pack(r: Field, g: Field, b: Field, a: Field) -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            Self(backend::x86::U32x16::pack_rgba(r.0, g.0, b.0, a.0))
        }
        #[cfg(all(not(target_feature = "avx512f"), target_feature = "avx2"))]
        {
            Self(backend::x86::U32x8::pack_rgba(r.0, g.0, b.0, a.0))
        }
        #[cfg(all(not(target_feature = "avx512f"), not(target_feature = "avx2")))]
        {
            Self(backend::x86::U32x4::pack_rgba(r.0, g.0, b.0, a.0))
        }
    }

    /// Pack 4 Fields (RGBA, 0.0-1.0) into packed u32 pixels.
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[inline(always)]
    pub fn pack(r: Field, g: Field, b: Field, _a: Field) -> Self {
        // Scalar fallback - only packs first element
        let mut r_buf = [0.0f32; 1];
        let mut g_buf = [0.0f32; 1];
        let mut b_buf = [0.0f32; 1];
        let mut a_buf = [0.0f32; 1];
        r.store(&mut r_buf);
        g.store(&mut g_buf);
        b.store(&mut b_buf);
        _a.store(&mut a_buf);

        let r_u8 = (r_buf[0].clamp(0.0, 1.0) * 255.0) as u32;
        let g_u8 = (g_buf[0].clamp(0.0, 1.0) * 255.0) as u32;
        let b_u8 = (b_buf[0].clamp(0.0, 1.0) * 255.0) as u32;
        let a_u8 = (a_buf[0].clamp(0.0, 1.0) * 255.0) as u32;
        let packed = r_u8 | (g_u8 << 8) | (b_u8 << 16) | (a_u8 << 24);

        Self(backend::scalar::ScalarU32::splat(packed))
    }

    /// Branchless select: returns `if_true` where mask is set, `if_false` elsewhere.
    ///
    /// The mask is interpreted bitwise from the Field representation.
    #[inline(always)]
    #[must_use]
    pub fn select(mask: Field, if_true: Self, if_false: Self) -> Self {
        use core::ops::{BitAnd, BitOr, Not};
        let mask_bits: NativeU32Simd = unsafe { core::mem::transmute(mask.0) };
        let t = if_true.0.bitand(mask_bits);
        let f = if_false.0.bitand(mask_bits.not());
        Self(t.bitor(f))
    }
}

// ============================================================================
// Field<Dual<N>> — Automatic Differentiation Support
// ============================================================================

use storage::DualStorage;

/// Type alias for 2D automatic differentiation (value + ∂/∂x, ∂/∂y).
///
/// This is `Field<Dual<2>>` — a SIMD batch of dual numbers with 2 partials.
/// Used for gradient-based antialiasing.
pub type Jet2 = Field<Dual<2>>;

/// Type alias for 3D automatic differentiation (value + ∂/∂x, ∂/∂y, ∂/∂z).
///
/// This is `Field<Dual<3>>` — a SIMD batch of dual numbers with 3 partials.
/// Used for surface normal computation in 3D rendering.
pub type Jet3 = Field<Dual<3>>;

impl<const N: usize> Field<Dual<N>>
where
    <f32 as FieldStorage>::Storage: Default,
{
    /// Get the function value as a Field.
    #[inline(always)]
    #[must_use]
    pub fn val(&self) -> Field {
        Field(self.0.val)
    }

    /// Get a partial derivative by index as a Field.
    ///
    /// - Index 0 = ∂/∂x
    /// - Index 1 = ∂/∂y
    /// - Index 2 = ∂/∂z (for Jet3)
    #[inline(always)]
    #[must_use]
    pub fn partial(&self, index: usize) -> Field {
        Field(self.0.partials[index])
    }

    /// Create a dual field seeded for differentiation with respect to variable `index`.
    ///
    /// The value is set to `val`, and the partial at `index` is 1, others are 0.
    /// This is how you mark a coordinate for differentiation.
    ///
    /// - `seed(val, 0)` for X differentiation
    /// - `seed(val, 1)` for Y differentiation
    /// - `seed(val, 2)` for Z differentiation (Jet3)
    #[inline(always)]
    #[must_use]
    pub fn seed(val: Field, index: usize) -> Self {
        let mut partials = [<f32 as FieldStorage>::zero_storage(); N];
        partials[index] = <f32 as FieldStorage>::one_storage();
        Self(DualStorage {
            val: val.0,
            partials,
        })
    }

    /// Create a constant dual field (value with zero derivatives).
    ///
    /// Used for constants that don't vary with coordinates.
    #[inline(always)]
    #[must_use]
    pub fn constant(val: Field) -> Self {
        Self(DualStorage {
            val: val.0,
            partials: [<f32 as FieldStorage>::zero_storage(); N],
        })
    }

    /// Create from value and partials.
    #[inline(always)]
    #[must_use]
    pub fn from_parts(val: Field, partials: [Field; N]) -> Self {
        Self(DualStorage {
            val: val.0,
            partials: core::array::from_fn(|i| partials[i].0),
        })
    }
}

/// Jet2-specific methods (2D differentiation).
impl Field<Dual<2>>
where
    <f32 as FieldStorage>::Storage: Default,
{
    /// Create a jet seeded for X differentiation (∂x/∂x = 1, ∂x/∂y = 0).
    #[inline(always)]
    #[must_use]
    pub fn x(val: Field) -> Self {
        Self::seed(val, 0)
    }

    /// Create a jet seeded for Y differentiation (∂y/∂x = 0, ∂y/∂y = 1).
    #[inline(always)]
    #[must_use]
    pub fn y(val: Field) -> Self {
        Self::seed(val, 1)
    }

    /// Get ∂f/∂x.
    #[inline(always)]
    #[must_use]
    pub fn dx(&self) -> Field {
        self.partial(0)
    }

    /// Get ∂f/∂y.
    #[inline(always)]
    #[must_use]
    pub fn dy(&self) -> Field {
        self.partial(1)
    }
}

/// Jet3-specific methods (3D differentiation).
impl Field<Dual<3>>
where
    <f32 as FieldStorage>::Storage: Default,
{
    /// Create a jet seeded for X differentiation.
    #[inline(always)]
    #[must_use]
    pub fn x(val: Field) -> Self {
        Self::seed(val, 0)
    }

    /// Create a jet seeded for Y differentiation.
    #[inline(always)]
    #[must_use]
    pub fn y(val: Field) -> Self {
        Self::seed(val, 1)
    }

    /// Create a jet seeded for Z differentiation.
    #[inline(always)]
    #[must_use]
    pub fn z(val: Field) -> Self {
        Self::seed(val, 2)
    }

    /// Get ∂f/∂x.
    #[inline(always)]
    #[must_use]
    pub fn dx(&self) -> Field {
        self.partial(0)
    }

    /// Get ∂f/∂y.
    #[inline(always)]
    #[must_use]
    pub fn dy(&self) -> Field {
        self.partial(1)
    }

    /// Get ∂f/∂z.
    #[inline(always)]
    #[must_use]
    pub fn dz(&self) -> Field {
        self.partial(2)
    }
}

// ============================================================================
// Arithmetic for Field<Dual<N>> — Chain Rule
// ============================================================================

impl<const N: usize> core::ops::Add for Field<Dual<N>>
where
    <f32 as FieldStorage>::Storage:
        Default + core::ops::Add<Output = <f32 as FieldStorage>::Storage>,
{
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // (f + g)' = f' + g'
        Self(DualStorage {
            val: self.0.val + rhs.0.val,
            partials: core::array::from_fn(|i| self.0.partials[i] + rhs.0.partials[i]),
        })
    }
}

impl<const N: usize> core::ops::Sub for Field<Dual<N>>
where
    <f32 as FieldStorage>::Storage:
        Default + core::ops::Sub<Output = <f32 as FieldStorage>::Storage>,
{
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        // (f - g)' = f' - g'
        Self(DualStorage {
            val: self.0.val - rhs.0.val,
            partials: core::array::from_fn(|i| self.0.partials[i] - rhs.0.partials[i]),
        })
    }
}

impl<const N: usize> core::ops::Mul for Field<Dual<N>>
where
    <f32 as FieldStorage>::Storage: Default
        + Copy
        + core::ops::Add<Output = <f32 as FieldStorage>::Storage>
        + core::ops::Mul<Output = <f32 as FieldStorage>::Storage>,
{
    type Output = Self;
    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Self {
        // Product rule: (f * g)' = f' * g + f * g'
        Self(DualStorage {
            val: self.0.val * rhs.0.val,
            partials: core::array::from_fn(|i| {
                self.0.partials[i] * rhs.0.val + self.0.val * rhs.0.partials[i]
            }),
        })
    }
}

impl<const N: usize> core::ops::Neg for Field<Dual<N>>
where
    <f32 as FieldStorage>::Storage:
        Default + core::ops::Neg<Output = <f32 as FieldStorage>::Storage>,
{
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(DualStorage {
            val: -self.0.val,
            partials: core::array::from_fn(|i| -self.0.partials[i]),
        })
    }
}

impl<const N: usize> core::ops::Div for Field<Dual<N>>
where
    <f32 as FieldStorage>::Storage: Default
        + Copy
        + core::ops::Add<Output = <f32 as FieldStorage>::Storage>
        + core::ops::Sub<Output = <f32 as FieldStorage>::Storage>
        + core::ops::Mul<Output = <f32 as FieldStorage>::Storage>
        + core::ops::Div<Output = <f32 as FieldStorage>::Storage>,
{
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        // Quotient rule: (f / g)' = (f' * g - f * g') / g²
        let g_squared = rhs.0.val * rhs.0.val;
        Self(DualStorage {
            val: self.0.val / rhs.0.val,
            partials: core::array::from_fn(|i| {
                (self.0.partials[i] * rhs.0.val - self.0.val * rhs.0.partials[i]) / g_squared
            }),
        })
    }
}

// ============================================================================
// Scalar-Jet Operations for Field<Dual<N>> (explicit methods to avoid trait conflicts)
// ============================================================================

impl<const N: usize> Field<Dual<N>>
where
    <f32 as FieldStorage>::Storage: Default + Copy,
{
    /// Multiply jet by scalar: f * c = (val * c, partials * c)
    #[inline(always)]
    #[must_use]
    pub fn scale(self, c: Field) -> Self
    where
        <f32 as FieldStorage>::Storage: core::ops::Mul<Output = <f32 as FieldStorage>::Storage>,
    {
        Self(DualStorage {
            val: self.0.val * c.0,
            partials: core::array::from_fn(|i| self.0.partials[i] * c.0),
        })
    }

    /// Divide jet by scalar: f / c = (val / c, partials / c)
    #[inline(always)]
    #[must_use]
    pub fn div_scalar(self, c: Field) -> Self
    where
        <f32 as FieldStorage>::Storage: core::ops::Div<Output = <f32 as FieldStorage>::Storage>,
    {
        Self(DualStorage {
            val: self.0.val / c.0,
            partials: core::array::from_fn(|i| self.0.partials[i] / c.0),
        })
    }

    /// Add scalar to jet: f + c (constant doesn't change derivatives)
    #[inline(always)]
    #[must_use]
    pub fn add_scalar(self, c: Field) -> Self
    where
        <f32 as FieldStorage>::Storage: core::ops::Add<Output = <f32 as FieldStorage>::Storage>,
    {
        Self(DualStorage {
            val: self.0.val + c.0,
            partials: self.0.partials,
        })
    }

    /// Subtract scalar from jet: f - c (constant doesn't change derivatives)
    #[inline(always)]
    #[must_use]
    pub fn sub_scalar(self, c: Field) -> Self
    where
        <f32 as FieldStorage>::Storage: core::ops::Sub<Output = <f32 as FieldStorage>::Storage>,
    {
        Self(DualStorage {
            val: self.0.val - c.0,
            partials: self.0.partials,
        })
    }
}

// ============================================================================
// Transcendental Operations for Field<Dual<N>> (Automatic Differentiation)
// ============================================================================

/// Helper to apply Field operations on storage and return storage
#[inline(always)]
fn field_sin(storage: NativeSimd) -> NativeSimd {
    Field(storage).sin().0
}

#[inline(always)]
fn field_cos(storage: NativeSimd) -> NativeSimd {
    Field(storage).cos().0
}

#[inline(always)]
fn field_exp(storage: NativeSimd) -> NativeSimd {
    Field(storage).exp().0
}

#[inline(always)]
fn field_ln(storage: NativeSimd) -> NativeSimd {
    Field(storage).ln().0
}

#[inline(always)]
fn field_abs(storage: NativeSimd) -> NativeSimd {
    Field(storage).abs().0
}

#[inline(always)]
fn field_floor(storage: NativeSimd) -> NativeSimd {
    Field(storage).floor().0
}

#[inline(always)]
fn field_min(a: NativeSimd, b: NativeSimd) -> NativeSimd {
    Field(a).min(Field(b)).0
}

#[inline(always)]
fn field_max(a: NativeSimd, b: NativeSimd) -> NativeSimd {
    Field(a).max(Field(b)).0
}

impl<const N: usize> Field<Dual<N>>
where
    <f32 as FieldStorage>::Storage: Default + Copy,
{
    /// Square root with chain rule: sqrt(f)' = f' / (2 * sqrt(f))
    #[inline(always)]
    #[must_use]
    pub fn jet_sqrt(self) -> Self {
        // Compute rsqrt once
        let x = Field(self.0.val);
        let rsqrt = x.rsqrt();

        // Compute value: x * rsqrt(x), masked for x <= 0
        let zero = Field::from(0.0);
        let result = Field(x.0 * rsqrt.0);
        let is_zero_or_neg = x.le(zero);
        let val = Field::select_raw(is_zero_or_neg, zero, result);

        // Compute derivative factor: 0.5 * rsqrt(x)
        let half = NativeSimd::splat(0.5);
        let half_rsqrt = rsqrt.0 * half;

        Self(DualStorage {
            val: val.0,
            partials: core::array::from_fn(|i| self.0.partials[i] * half_rsqrt),
        })
    }

    /// Sine with chain rule: sin(f)' = cos(f) * f'
    #[inline(always)]
    #[must_use]
    pub fn jet_sin(self) -> Self {
        let sin_val = field_sin(self.0.val);
        let cos_val = field_cos(self.0.val);
        Self(DualStorage {
            val: sin_val,
            partials: core::array::from_fn(|i| cos_val * self.0.partials[i]),
        })
    }

    /// Cosine with chain rule: cos(f)' = -sin(f) * f'
    #[inline(always)]
    #[must_use]
    pub fn jet_cos(self) -> Self {
        let sin_val = field_sin(self.0.val);
        let cos_val = field_cos(self.0.val);
        let neg_sin = -sin_val;
        Self(DualStorage {
            val: cos_val,
            partials: core::array::from_fn(|i| neg_sin * self.0.partials[i]),
        })
    }

    /// Exponential with chain rule: exp(f)' = exp(f) * f'
    #[inline(always)]
    #[must_use]
    pub fn jet_exp(self) -> Self {
        let exp_val = field_exp(self.0.val);
        Self(DualStorage {
            val: exp_val,
            partials: core::array::from_fn(|i| exp_val * self.0.partials[i]),
        })
    }

    /// Natural log with chain rule: ln(f)' = f' / f
    #[inline(always)]
    #[must_use]
    pub fn jet_ln(self) -> Self {
        let ln_val = field_ln(self.0.val);
        Self(DualStorage {
            val: ln_val,
            partials: core::array::from_fn(|i| self.0.partials[i] / self.0.val),
        })
    }

    /// Absolute value with chain rule: |f|' = sign(f) * f'
    #[inline(always)]
    #[must_use]
    pub fn jet_abs(self) -> Self {
        let abs_val = field_abs(self.0.val);
        let zero = NativeSimd::splat(0.0);
        let one = NativeSimd::splat(1.0);
        let neg_one = NativeSimd::splat(-1.0);
        // sign = f >= 0 ? 1 : -1
        let mask = self.0.val.cmp_ge(zero);
        let sign = NativeSimd::simd_select(mask, one, neg_one);
        Self(DualStorage {
            val: abs_val,
            partials: core::array::from_fn(|i| sign * self.0.partials[i]),
        })
    }

    /// Floor function (derivative is 0 almost everywhere)
    #[inline(always)]
    #[must_use]
    pub fn jet_floor(self) -> Self {
        Self(DualStorage {
            val: field_floor(self.0.val),
            partials: core::array::from_fn(|_| NativeSimd::splat(0.0)),
        })
    }

    /// Minimum of two jets: min(f, g) with derivative from the selected branch
    #[inline(always)]
    #[must_use]
    pub fn jet_min(self, other: Self) -> Self {
        let mask = self.0.val.cmp_lt(other.0.val);
        Self(DualStorage {
            val: field_min(self.0.val, other.0.val),
            partials: core::array::from_fn(|i| {
                NativeSimd::simd_select(mask, self.0.partials[i], other.0.partials[i])
            }),
        })
    }

    /// Maximum of two jets: max(f, g) with derivative from the selected branch
    #[inline(always)]
    #[must_use]
    pub fn jet_max(self, other: Self) -> Self {
        let mask = self.0.val.cmp_gt(other.0.val);
        Self(DualStorage {
            val: field_max(self.0.val, other.0.val),
            partials: core::array::from_fn(|i| {
                NativeSimd::simd_select(mask, self.0.partials[i], other.0.partials[i])
            }),
        })
    }
}

// ============================================================================
// Bitwise Operations for Discrete (required by Computational)
// ============================================================================

impl core::ops::BitAnd for Discrete {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0.bitand(rhs.0))
    }
}

impl core::ops::BitOr for Discrete {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0.bitor(rhs.0))
    }
}

impl core::ops::Not for Discrete {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(self.0.not())
    }
}

// ============================================================================
// Computational Implementation for Discrete
// ============================================================================

impl numeric::Computational for Discrete {
    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        // Direct truncation to u32, then splat
        Self(NativeU32Simd::splat(val as u32))
    }

    #[inline(always)]
    fn sequential(start: f32) -> Self {
        // Splat the starting value (Discrete doesn't have meaningful sequential semantics)
        Self::from_f32(start)
    }
}

// ============================================================================
// Computational Implementation for Field (Public API)
// ============================================================================

impl numeric::Computational for Field {
    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        Self::from(val)
    }

    #[inline(always)]
    fn sequential(start: f32) -> Self {
        Self(NativeSimd::sequential(start))
    }
}

// Field is a coordinate type
impl numeric::Coordinate for Field {}

// ============================================================================
// Selectable Implementation for Field (Internal)
// ============================================================================

impl numeric::Selectable for Field {
    #[inline(always)]
    fn select_raw(mask: Field, if_true: Self, if_false: Self) -> Self {
        Self(NativeSimd::simd_select(
            mask.0.float_to_mask(),
            if_true.0,
            if_false.0,
        ))
    }
}

// ============================================================================
// Selectable Implementation for Discrete (Internal)
// ============================================================================

impl numeric::Selectable for Discrete {
    #[inline(always)]
    fn select_raw(mask: Field, if_true: Self, if_false: Self) -> Self {
        // Reinterpret the float mask bits as u32 for bitwise ops
        // mask lanes are either 0xFFFFFFFF (true) or 0x00000000 (false)
        let mask_bits: NativeU32Simd = unsafe { core::mem::transmute(mask.0) };

        // (mask & if_true) | (!mask & if_false)
        use core::ops::{BitAnd, BitOr, Not};
        let t = if_true.0.bitand(mask_bits);
        let f = if_false.0.bitand(mask_bits.not());
        Self(t.bitor(f))
    }
}

// ============================================================================
// Numeric Implementation for Field (Internal)
// ============================================================================

impl numeric::Numeric for Field {
    #[inline(always)]
    fn sqrt(self) -> Self {
        Self::sqrt(self)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self::abs(self)
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        Self::min(self, rhs)
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        Self::max(self, rhs)
    }

    #[inline(always)]
    fn lt(self, rhs: Self) -> Self {
        Self::lt(self, rhs)
    }

    #[inline(always)]
    fn le(self, rhs: Self) -> Self {
        Self::le(self, rhs)
    }

    #[inline(always)]
    fn gt(self, rhs: Self) -> Self {
        Self::gt(self, rhs)
    }

    #[inline(always)]
    fn ge(self, rhs: Self) -> Self {
        Self::ge(self, rhs)
    }

    #[inline(always)]
    fn select(mask: Self, if_true: Self, if_false: Self) -> Self {
        if mask.all() {
            return if_true;
        }
        if !mask.any() {
            return if_false;
        }
        Self::select_raw(mask, if_true, if_false)
    }

    #[inline(always)]
    fn select_raw(mask: Self, if_true: Self, if_false: Self) -> Self {
        // Convert float mask to native, use native select
        Self(NativeSimd::simd_select(
            mask.0.float_to_mask(),
            if_true.0,
            if_false.0,
        ))
    }

    #[inline(always)]
    fn any(&self) -> bool {
        Self::any(self)
    }

    #[inline(always)]
    fn all(&self) -> bool {
        Self::all(self)
    }

    #[inline(always)]
    fn from_i32(val: i32) -> Self {
        Self::from(val)
    }

    #[inline(always)]
    fn from_field(field: Field) -> Self {
        field
    }

    #[inline(always)]
    fn sin(self) -> Self {
        Self::sin(self)
    }

    #[inline(always)]
    fn cos(self) -> Self {
        Self::cos(self)
    }

    #[inline(always)]
    fn atan2(self, x: Self) -> Self {
        Self::atan2(self, x)
    }

    #[inline(always)]
    fn pow(self, exp: Self) -> Self {
        // pow(a, b) = exp2(b * log2(a)) for positive a
        Self(Self(self.0.log2() * exp.0).0.exp2())
    }

    #[inline(always)]
    fn exp(self) -> Self {
        Self::exp(self)
    }

    #[inline(always)]
    fn log2(self) -> Self {
        Self::log2(self)
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        Self::exp2(self)
    }

    #[inline(always)]
    fn ln(self) -> Self {
        Self::ln(self)
    }

    #[inline(always)]
    fn log10(self) -> Self {
        Self::log10(self)
    }

    #[inline(always)]
    fn tan(self) -> Self {
        Self::tan(self)
    }

    #[inline(always)]
    fn asin(self) -> Self {
        Self::asin(self)
    }

    #[inline(always)]
    fn acos(self) -> Self {
        Self::acos(self)
    }

    #[inline(always)]
    fn atan(self) -> Self {
        Self::atan(self)
    }

    #[inline(always)]
    fn floor(self) -> Self {
        Self::floor(self)
    }

    #[inline(always)]
    fn ceil(self) -> Self {
        Self::ceil(self)
    }

    #[inline(always)]
    fn round(self) -> Self {
        Self::round(self)
    }

    #[inline(always)]
    fn fract(self) -> Self {
        Self::fract(self)
    }

    #[inline(always)]
    fn hypot(self, y: Self) -> Self {
        Self::hypot(self, y)
    }

    #[inline(always)]
    fn mul_rsqrt(self, other: Self) -> Self {
        Self::mul_rsqrt(self, other)
    }

    #[inline(always)]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        Self::clamp(self, lo, hi)
    }

    #[inline(always)]
    fn eq(self, rhs: Self) -> Self {
        Self::eq(self, rhs)
    }

    #[inline(always)]
    fn ne(self, rhs: Self) -> Self {
        Self::ne(self, rhs)
    }

    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        Self::mul_add(self, b, c)
    }

    #[inline(always)]
    fn recip(self) -> Self {
        Self::recip(self)
    }

    #[inline(always)]
    fn rsqrt(self) -> Self {
        Self::rsqrt(self)
    }

    #[inline(always)]
    fn add_masked(self, val: Self, mask: Self) -> Self {
        Self::add_masked(self, val, mask)
    }

    #[inline(always)]
    fn raw_add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }

    #[inline(always)]
    fn raw_sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }

    #[inline(always)]
    fn raw_mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }

    #[inline(always)]
    fn raw_div(self, rhs: Self) -> Self {
        Self(self.0 / rhs.0)
    }

    #[inline(always)]
    fn raw_neg(self) -> Self {
        Self(-self.0)
    }
}

// ============================================================================
// From Implementations (the ONLY way to create Field from scalars)
// ============================================================================

impl From<f32> for Field {
    #[inline(always)]
    fn from(val: f32) -> Self {
        Self(NativeSimd::splat(val))
    }
}

impl From<i32> for Field {
    #[inline(always)]
    fn from(val: i32) -> Self {
        Self(NativeSimd::splat(val as f32))
    }
}

// ============================================================================
// Operator Implementations (AST-building)
// ============================================================================
//
// All Field ops build AST nodes for composition with other manifolds.
// FMA fusion: Mul<A,B> + C → MulAdd<A,B,C> (see ops/chained.rs)

impl<M: Manifold> core::ops::Add<M> for Field {
    type Output = ops::Add<Self, M>;
    #[inline(always)]
    fn add(self, rhs: M) -> Self::Output {
        ops::Add(self, rhs)
    }
}

impl<M: Manifold> core::ops::Sub<M> for Field {
    type Output = ops::Sub<Self, M>;
    #[inline(always)]
    fn sub(self, rhs: M) -> Self::Output {
        ops::Sub(self, rhs)
    }
}

impl<M: Manifold> core::ops::Mul<M> for Field {
    type Output = ops::Mul<Self, M>;
    #[inline(always)]
    fn mul(self, rhs: M) -> Self::Output {
        ops::Mul(self, rhs)
    }
}

// Rsqrt fusion: Field / Sqrt<R> → MulRsqrt<Field, R>
impl<R: Manifold> core::ops::Div<ops::Sqrt<R>> for Field {
    type Output = ops::MulRsqrt<Self, R>;
    #[inline(always)]
    fn div(self, rhs: ops::Sqrt<R>) -> Self::Output {
        ops::MulRsqrt(self, rhs.0)
    }
}

// Enumerate all other divisor types for Field (to avoid conflict with Sqrt)
impl<DL: Manifold, DR: Manifold> core::ops::Div<ops::Add<DL, DR>> for Field {
    type Output = ops::Div<Self, ops::Add<DL, DR>>;
    #[inline(always)]
    fn div(self, rhs: ops::Add<DL, DR>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DL: Manifold, DR: Manifold> core::ops::Div<ops::Sub<DL, DR>> for Field {
    type Output = ops::Div<Self, ops::Sub<DL, DR>>;
    #[inline(always)]
    fn div(self, rhs: ops::Sub<DL, DR>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DL: Manifold, DR: Manifold> core::ops::Div<ops::Mul<DL, DR>> for Field {
    type Output = ops::Div<Self, ops::Mul<DL, DR>>;
    #[inline(always)]
    fn div(self, rhs: ops::Mul<DL, DR>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DL: Manifold, DR: Manifold> core::ops::Div<ops::Div<DL, DR>> for Field {
    type Output = ops::Div<Self, ops::Div<DL, DR>>;
    #[inline(always)]
    fn div(self, rhs: ops::Div<DL, DR>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DL: Manifold, DR: Manifold> core::ops::Div<ops::Max<DL, DR>> for Field {
    type Output = ops::Div<Self, ops::Max<DL, DR>>;
    #[inline(always)]
    fn div(self, rhs: ops::Max<DL, DR>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DL: Manifold, DR: Manifold> core::ops::Div<ops::Min<DL, DR>> for Field {
    type Output = ops::Div<Self, ops::Min<DL, DR>>;
    #[inline(always)]
    fn div(self, rhs: ops::Min<DL, DR>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM: Manifold> core::ops::Div<ops::Abs<DM>> for Field {
    type Output = ops::Div<Self, ops::Abs<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::Abs<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM: Manifold> core::ops::Div<ops::Floor<DM>> for Field {
    type Output = ops::Div<Self, ops::Floor<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::Floor<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM: Manifold> core::ops::Div<ops::Rsqrt<DM>> for Field {
    type Output = ops::Div<Self, ops::Rsqrt<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::Rsqrt<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM: Manifold> core::ops::Div<ops::Sin<DM>> for Field {
    type Output = ops::Div<Self, ops::Sin<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::Sin<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM: Manifold> core::ops::Div<ops::Cos<DM>> for Field {
    type Output = ops::Div<Self, ops::Cos<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::Cos<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM: Manifold> core::ops::Div<ops::Tan<DM>> for Field {
    type Output = ops::Div<Self, ops::Tan<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::Tan<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM: Manifold> core::ops::Div<ops::Asin<DM>> for Field {
    type Output = ops::Div<Self, ops::Asin<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::Asin<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM: Manifold> core::ops::Div<ops::Acos<DM>> for Field {
    type Output = ops::Div<Self, ops::Acos<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::Acos<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM: Manifold> core::ops::Div<ops::Atan<DM>> for Field {
    type Output = ops::Div<Self, ops::Atan<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::Atan<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM: Manifold> core::ops::Div<ops::Ceil<DM>> for Field {
    type Output = ops::Div<Self, ops::Ceil<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::Ceil<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM: Manifold> core::ops::Div<ops::Round<DM>> for Field {
    type Output = ops::Div<Self, ops::Round<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::Round<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM: Manifold> core::ops::Div<ops::Fract<DM>> for Field {
    type Output = ops::Div<Self, ops::Fract<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::Fract<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM: Manifold> core::ops::Div<ops::Log2<DM>> for Field {
    type Output = ops::Div<Self, ops::Log2<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::Log2<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM: Manifold> core::ops::Div<ops::Exp2<DM>> for Field {
    type Output = ops::Div<Self, ops::Exp2<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::Exp2<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM: Manifold> core::ops::Div<ops::Exp<DM>> for Field {
    type Output = ops::Div<Self, ops::Exp<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::Exp<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM: Manifold> core::ops::Div<ops::Ln<DM>> for Field {
    type Output = ops::Div<Self, ops::Ln<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::Ln<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM: Manifold> core::ops::Div<ops::Log10<DM>> for Field {
    type Output = ops::Div<Self, ops::Log10<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::Log10<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM: Manifold> core::ops::Div<ops::Recip<DM>> for Field {
    type Output = ops::Div<Self, ops::Recip<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::Recip<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM: Manifold> core::ops::Div<ops::Neg<DM>> for Field {
    type Output = ops::Div<Self, ops::Neg<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::Neg<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DC: Manifold, DT: Manifold, DF: Manifold> core::ops::Div<combinators::Select<DC, DT, DF>>
    for Field
{
    type Output = ops::Div<Self, combinators::Select<DC, DT, DF>>;
    #[inline(always)]
    fn div(self, rhs: combinators::Select<DC, DT, DF>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DA: Manifold, DB: Manifold, DC: Manifold> core::ops::Div<ops::MulAdd<DA, DB, DC>> for Field {
    type Output = ops::Div<Self, ops::MulAdd<DA, DB, DC>>;
    #[inline(always)]
    fn div(self, rhs: ops::MulAdd<DA, DB, DC>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM: Manifold> core::ops::Div<ops::MulRecip<DM>> for Field {
    type Output = ops::Div<Self, ops::MulRecip<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::MulRecip<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DL: Manifold, DR: Manifold> core::ops::Div<ops::MulRsqrt<DL, DR>> for Field {
    type Output = ops::Div<Self, ops::MulRsqrt<DL, DR>>;
    #[inline(always)]
    fn div(self, rhs: ops::MulRsqrt<DL, DR>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DAcc: Manifold, DVal: Manifold, DMask: Manifold>
    core::ops::Div<ops::AddMasked<DAcc, DVal, DMask>> for Field
{
    type Output = ops::Div<Self, ops::AddMasked<DAcc, DVal, DMask>>;
    #[inline(always)]
    fn div(self, rhs: ops::AddMasked<DAcc, DVal, DMask>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
// Concrete divisor types for Field
impl core::ops::Div<variables::X> for Field {
    type Output = ops::Div<Self, variables::X>;
    #[inline(always)]
    fn div(self, rhs: variables::X) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl core::ops::Div<variables::Y> for Field {
    type Output = ops::Div<Self, variables::Y>;
    #[inline(always)]
    fn div(self, rhs: variables::Y) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl core::ops::Div<variables::Z> for Field {
    type Output = ops::Div<Self, variables::Z>;
    #[inline(always)]
    fn div(self, rhs: variables::Z) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl core::ops::Div<variables::W> for Field {
    type Output = ops::Div<Self, variables::W>;
    #[inline(always)]
    fn div(self, rhs: variables::W) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl core::ops::Div<Field> for Field {
    type Output = ops::Div<Self, Field>;
    #[inline(always)]
    fn div(self, rhs: Field) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl core::ops::Div<f32> for Field {
    type Output = ops::Div<Self, f32>;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl core::ops::Div<i32> for Field {
    type Output = ops::Div<Self, i32>;
    #[inline(always)]
    fn div(self, rhs: i32) -> Self::Output {
        ops::Div(self, rhs)
    }
}
// Fused derivative combinator divisors for Field
impl<DM> core::ops::Div<ops::derivative::GradientMag2D<DM>> for Field {
    type Output = ops::Div<Self, ops::derivative::GradientMag2D<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::derivative::GradientMag2D<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM> core::ops::Div<ops::derivative::GradientMag3D<DM>> for Field {
    type Output = ops::Div<Self, ops::derivative::GradientMag3D<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::derivative::GradientMag3D<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM> core::ops::Div<ops::derivative::Antialias2D<DM>> for Field {
    type Output = ops::Div<Self, ops::derivative::Antialias2D<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::derivative::Antialias2D<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM> core::ops::Div<ops::derivative::Antialias3D<DM>> for Field {
    type Output = ops::Div<Self, ops::derivative::Antialias3D<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::derivative::Antialias3D<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}
impl<DM> core::ops::Div<ops::derivative::Curvature2D<DM>> for Field {
    type Output = ops::Div<Self, ops::derivative::Curvature2D<DM>>;
    #[inline(always)]
    fn div(self, rhs: ops::derivative::Curvature2D<DM>) -> Self::Output {
        ops::Div(self, rhs)
    }
}

impl core::ops::BitAnd for Field {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl core::ops::BitOr for Field {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl core::ops::Not for Field {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(!self.0)
    }
}

impl core::ops::Neg for Field {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Materialize a discrete color manifold into packed u32 pixels.
///
/// Evaluates a color manifold at sequential x coordinates starting from (x, y)
/// and stores the packed RGBA u32 pixels directly to the output buffer.
///
/// This is the primary way to extract pixel data from a `Manifold<Output = Discrete>`.
///
/// # Example
/// ```ignore
/// let color = Color::Named(NamedColor::Red);
/// let mut pixels = [0u32; PARALLELISM];
/// materialize_discrete(&color, 0.0, 0.0, &mut pixels);
/// ```
#[inline(always)]
pub fn materialize_discrete<M>(m: &M, x: f32, y: f32, out: &mut [u32])
where
    M: Manifold<Output = Discrete> + ?Sized,
{
    let xs = Field::sequential(x);
    let discrete = m.eval_raw(xs, Field::from(y), Field::from(0.0), Field::from(0.0));
    discrete.store(out);
}

/// Materialize a discrete color manifold into packed u32 pixels using precomputed Fields.
///
/// This is an optimized version of [`materialize_discrete`] where coordinate Fields
/// are precomputed or updated in a loop (e.g., via vector addition).
///
/// # Example
/// ```ignore
/// let mut xs = Field::sequential(0.0);
/// let step = Field::from(PARALLELISM as f32);
/// let ys = Field::from(0.0);
/// let mut pixels = [0u32; PARALLELISM];
///
/// materialize_discrete_fields(&color, xs, ys, &mut pixels);
/// xs = xs + step;
/// ```
#[inline(always)]
pub fn materialize_discrete_fields<M>(m: &M, x: Field, y: Field, out: &mut [u32])
where
    M: Manifold<Output = Discrete> + ?Sized,
{
    let discrete = m.eval_raw(x, y, Field::from(0.0), Field::from(0.0));
    discrete.store(out);
}

/// Materialize a vector manifold into an interleaved f32 buffer.
///
/// Evaluates at sequential x coordinates starting from (x, y), then transposes
/// from SoA (structure of arrays) to AoS (array of structures) for storage.
///
/// Output is interleaved RGBA: [r0,g0,b0,a0, r1,g1,b1,a1, ...]
///
/// For scalar manifolds, use `Lift` to convert them to a uniform color first.
#[inline(always)]
pub fn materialize<M, V>(m: &M, x: f32, y: f32, out: &mut [f32])
where
    M: Manifold<Output = V> + ?Sized,
    V: ops::Vector<Component = Field>,
{
    let xs = Field::sequential(x);
    let val = m.eval_raw(xs, Field::from(y), Field::from(0.0), Field::from(0.0));

    // Store each component to temporary buffers
    let mut buf_x = [0.0f32; PARALLELISM];
    let mut buf_y = [0.0f32; PARALLELISM];
    let mut buf_z = [0.0f32; PARALLELISM];
    let mut buf_w = [0.0f32; PARALLELISM];

    val.get(variables::Axis::X).store(&mut buf_x);
    val.get(variables::Axis::Y).store(&mut buf_y);
    val.get(variables::Axis::Z).store(&mut buf_z);
    val.get(variables::Axis::W).store(&mut buf_w);

    // Transpose: SoA → AoS (interleaved)
    for i in 0..PARALLELISM {
        let base = i * 4;
        if base + 3 < out.len() {
            out[base] = buf_x[i];
            out[base + 1] = buf_y[i];
            out[base + 2] = buf_z[i];
            out[base + 3] = buf_w[i];
        }
    }
}

/// Parallelism width (number of lanes).
pub const PARALLELISM: usize = NativeSimd::LANES;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gather_behavior() {
        let data = [10.0, 20.0, 30.0, 40.0, 50.0];
        // Indices: 0.0, 1.9 (trunc to 1), 2.1 (trunc to 2), 4.0
        // We expect: data[0], data[1], data[2], data[4]
        // i.e., 10.0, 20.0, 30.0, 50.0
        //
        // If we create a Field with these indices:
        // We can't easily construct arbitrary Field with parallelism > 4.
        // But gather is per-lane.
        //
        // Let's make a buffer of indices.
        let mut indices_buf = [0.0f32; PARALLELISM];
        indices_buf[0] = 0.0;
        if PARALLELISM > 1 {
            indices_buf[1] = 1.9;
        }
        if PARALLELISM > 2 {
            indices_buf[2] = 2.1;
        }
        if PARALLELISM > 3 {
            indices_buf[3] = 4.0;
        }

        let indices = Field::from_slice(&indices_buf);
        let result = Field::gather(&data, indices);

        let mut out = [0.0f32; PARALLELISM];
        result.store(&mut out);

        assert_eq!(out[0], 10.0);
        if PARALLELISM > 1 {
            assert_eq!(out[1], 20.0);
        } // 1.9 -> 1
        if PARALLELISM > 2 {
            assert_eq!(out[2], 30.0);
        } // 2.1 -> 2
        if PARALLELISM > 3 {
            assert_eq!(out[3], 50.0);
        } // 4.0 -> 4
    }
}
