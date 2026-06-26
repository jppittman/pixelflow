//! Zero-sized type marker trait.
//!
//! This module provides a marker trait [`Zst`] for identifying types that have
//! no runtime representation (zero-sized types). This is useful for:
//!
//! - Static assertion that a type compiles to no data
//! - Enabling optimizations based on ZST knowledge
//! - Documenting which types are purely compile-time constructs
//!
//! # What is a ZST?
//!
//! A zero-sized type (ZST) is a type that occupies zero bytes of memory.
//! In Rust, these types exist only at compile time and have no runtime cost.
//!
//! Examples in PixelFlow:
//! - Coordinate variables (`X`, `Y`, `Z`, `W`) - no data, just identity
//! - Spherical harmonic basis functions - only const generic parameters
//! - Operators where all operands are ZSTs - no data to store
//!
//! # Example
//!
//! ```ignore
//! use pixelflow_core::{Zst, X, Y};
//! use pixelflow_core::ops::Add;
//!
//! fn assert_zst<T: Zst>() {}
//!
//! assert_zst::<X>();           // ✓ X is a ZST
//! assert_zst::<Y>();           // ✓ Y is a ZST
//! assert_zst::<Add<X, Y>>();   // ✓ Add<X, Y> is a ZST (both operands are ZSTs)
//! ```

// ============================================================================
// Sealed Trait Pattern
// ============================================================================

mod sealed {
    /// Private sealed trait to prevent external implementations of Zst.
    /// This allows us to use blanket implementations without orphan rule conflicts.
    pub trait Sealed {}
}

// ============================================================================
// Helper Traits for Operator Classification (Sealed)
// ============================================================================

/// Trait for binary operators that expose their operand types.
/// This trait is sealed to ensure mutual exclusivity with UnaryOp and TernaryOp.
pub trait BinaryOp: sealed::Sealed {
    /// The left operand type.
    type Left;
    /// The right operand type.
    type Right;
}

/// Trait for unary operators that expose their inner type.
/// This trait is sealed to ensure mutual exclusivity with BinaryOp and TernaryOp.
pub trait UnaryOp: sealed::Sealed {
    /// The inner operand type.
    type Inner;
}

/// Trait for ternary operators that expose their three operand types.
/// This trait is sealed to ensure mutual exclusivity with BinaryOp and UnaryOp.
pub trait TernaryOp: sealed::Sealed {
    /// The first operand type.
    type First;
    /// The second operand type.
    type Second;
    /// The third operand type.
    type Third;
}

// ============================================================================
// Marker Trait
// ============================================================================

/// Marker trait for zero-sized types.
///
/// Types implementing this trait have `size_of::<T>() == 0` and are purely
/// compile-time constructs with no runtime representation.
///
/// Note: All ZST types in pixelflow-core also implement `Copy` (via derives),
/// but this trait does not require it as a supertrait. The trait is purely
/// for identifying zero-sized types at compile time.
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait Zst: sealed::Sealed {}

// ============================================================================
// Sealed Trait Implementations (Base ZST Types)
// ============================================================================

// Coordinate variables
impl sealed::Sealed for crate::variables::X {}
impl sealed::Sealed for crate::variables::Y {}
impl sealed::Sealed for crate::variables::Z {}
impl sealed::Sealed for crate::variables::W {}

// Spherical harmonics
impl<const L: usize, const M: i32> sealed::Sealed for crate::combinators::SphericalHarmonic<L, M> {}
impl<const L: usize> sealed::Sealed for crate::combinators::ZonalHarmonic<L> {}
impl<const NUM_COEFFS: usize> sealed::Sealed for crate::combinators::ShProject<NUM_COEFFS> {}

// Backend markers (internal, but technically ZSTs)
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
impl sealed::Sealed for crate::backend::scalar::Scalar {}

#[cfg(target_arch = "x86_64")]
impl sealed::Sealed for crate::backend::x86::Sse2 {}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
impl sealed::Sealed for crate::backend::x86::Avx2 {}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
impl sealed::Sealed for crate::backend::x86::Avx512 {}

#[cfg(target_arch = "aarch64")]
impl sealed::Sealed for crate::backend::arm::Neon {}

// Seal all operators (they get Zst from blanket impls below)
impl<L, R> sealed::Sealed for crate::ops::Add<L, R> {}
impl<L, R> sealed::Sealed for crate::ops::Sub<L, R> {}
impl<L, R> sealed::Sealed for crate::ops::Mul<L, R> {}
impl<L, R> sealed::Sealed for crate::ops::Div<L, R> {}
impl<L, R> sealed::Sealed for crate::ops::Max<L, R> {}
impl<L, R> sealed::Sealed for crate::ops::Min<L, R> {}
impl<L, R> sealed::Sealed for crate::ops::MulRsqrt<L, R> {}
impl<L, R> sealed::Sealed for crate::ops::Lt<L, R> {}
impl<L, R> sealed::Sealed for crate::ops::Gt<L, R> {}
impl<L, R> sealed::Sealed for crate::ops::Le<L, R> {}
impl<L, R> sealed::Sealed for crate::ops::Ge<L, R> {}
impl<L, R> sealed::Sealed for crate::ops::SoftGt<L, R> {}
impl<L, R> sealed::Sealed for crate::ops::SoftLt<L, R> {}
impl<L, R> sealed::Sealed for crate::ops::And<L, R> {}
impl<L, R> sealed::Sealed for crate::ops::Or<L, R> {}
impl<M> sealed::Sealed for crate::ops::Sqrt<M> {}
impl<M> sealed::Sealed for crate::ops::Abs<M> {}
impl<M> sealed::Sealed for crate::ops::Floor<M> {}
impl<M> sealed::Sealed for crate::ops::Rsqrt<M> {}
impl<M> sealed::Sealed for crate::ops::Sin<M> {}
impl<M> sealed::Sealed for crate::ops::Cos<M> {}
impl<M> sealed::Sealed for crate::ops::Log2<M> {}
impl<M> sealed::Sealed for crate::ops::Exp2<M> {}
impl<M> sealed::Sealed for crate::ops::Exp<M> {}
impl<M> sealed::Sealed for crate::ops::Neg<M> {}
impl<M> sealed::Sealed for crate::ops::BNot<M> {}
impl<A, B, C> sealed::Sealed for crate::ops::MulAdd<A, B, C> {}
impl<Acc, Val, Mask> sealed::Sealed for crate::ops::AddMasked<Acc, Val, Mask> {}
impl<Mask, IfTrue, IfFalse> sealed::Sealed for crate::ops::SoftSelect<Mask, IfTrue, IfFalse> {}

// Seal combinators
impl<C, T, F> sealed::Sealed for crate::combinators::Select<C, T, F> {}
impl<M, T> sealed::Sealed for crate::combinators::Map<M, T> {}
impl<M, D> sealed::Sealed for crate::combinators::Project<M, D> {}
impl<Cx, Cy, Cz, Cw, M> sealed::Sealed for crate::combinators::At<Cx, Cy, Cz, Cw, M> {}
impl<Seed, Step, Done> sealed::Sealed for crate::combinators::Fix<Seed, Step, Done> {}
impl<Val, Body> sealed::Sealed for crate::combinators::Let<Val, Body> {}
impl<N> sealed::Sealed for crate::combinators::Var<N> {}

// RecFix and Recurse
impl sealed::Sealed for crate::combinators::Recurse {}
impl<T, P> sealed::Sealed for crate::combinators::RecDomain<T, P> {}
impl<N, S, B> sealed::Sealed for crate::combinators::RecFix<N, S, B> {}

// Seal binary type-level number types
impl sealed::Sealed for crate::combinators::UTerm {}
impl sealed::Sealed for crate::combinators::B0 {}
impl sealed::Sealed for crate::combinators::B1 {}
impl<U, B> sealed::Sealed for crate::combinators::UInt<U, B> {}

// Context array position markers (for kernel! macro)
impl sealed::Sealed for crate::combinators::A0 {}
impl sealed::Sealed for crate::combinators::A1 {}
impl sealed::Sealed for crate::combinators::A2 {}
impl sealed::Sealed for crate::combinators::A3 {}

// Context variables (for kernel! macro)
impl<A, const I: usize> sealed::Sealed for crate::combinators::CtxVar<A, I> {}

// ContextFree wrapper (for kernel! macro)
impl<M> sealed::Sealed for crate::combinators::context::ContextFree<M> {}

// Derivative accessor combinators (ValOf, DxOf, DyOf, DzOf, etc.)
impl<M> sealed::Sealed for crate::ops::derivative::ValOf<M> {}
impl<M> sealed::Sealed for crate::ops::derivative::DxOf<M> {}
impl<M> sealed::Sealed for crate::ops::derivative::DyOf<M> {}
impl<M> sealed::Sealed for crate::ops::derivative::DzOf<M> {}
impl<M> sealed::Sealed for crate::ops::derivative::DxxOf<M> {}
impl<M> sealed::Sealed for crate::ops::derivative::DxyOf<M> {}
impl<M> sealed::Sealed for crate::ops::derivative::DyyOf<M> {}
impl<M> sealed::Sealed for crate::ops::derivative::GradientMag2D<M> {}
impl<M> sealed::Sealed for crate::ops::derivative::GradientMag3D<M> {}
impl<M> sealed::Sealed for crate::ops::derivative::Antialias2D<M> {}
impl<M> sealed::Sealed for crate::ops::derivative::Antialias3D<M> {}
impl<M> sealed::Sealed for crate::ops::derivative::Normalized2D<M> {}
impl<M> sealed::Sealed for crate::ops::derivative::Normalized3D<M> {}
impl<M> sealed::Sealed for crate::ops::derivative::Curvature2D<M> {}

// ============================================================================
// Base ZST Implementations
// ============================================================================

// Coordinate variables
impl crate::Zst for crate::variables::X {}
impl crate::Zst for crate::variables::Y {}
impl crate::Zst for crate::variables::Z {}
impl crate::Zst for crate::variables::W {}

// Spherical harmonics
impl<const L: usize, const M: i32> crate::Zst for crate::combinators::SphericalHarmonic<L, M> {}
impl<const L: usize> crate::Zst for crate::combinators::ZonalHarmonic<L> {}
impl<const NUM_COEFFS: usize> crate::Zst for crate::combinators::ShProject<NUM_COEFFS> {}

// Backend markers (internal, but technically ZSTs)
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
impl crate::Zst for crate::backend::scalar::Scalar {}

#[cfg(target_arch = "x86_64")]
impl crate::Zst for crate::backend::x86::Sse2 {}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
impl crate::Zst for crate::backend::x86::Avx2 {}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
impl crate::Zst for crate::backend::x86::Avx512 {}

#[cfg(target_arch = "aarch64")]
impl crate::Zst for crate::backend::arm::Neon {}

// Binary type-level numbers (for Let/Var bindings)
impl crate::Zst for crate::combinators::UTerm {}
impl crate::Zst for crate::combinators::B0 {}
impl crate::Zst for crate::combinators::B1 {}
impl<U: crate::Zst, B: crate::Zst> crate::Zst for crate::combinators::UInt<U, B> {}
impl<N: crate::Zst> crate::Zst for crate::combinators::Var<N> {}

// RecFix and Recurse
impl crate::Zst for crate::combinators::Recurse {}
// RecDomain is NOT a ZST (contains P)
impl<N: crate::Zst, S: crate::Zst, B: crate::Zst> crate::Zst
    for crate::combinators::RecFix<N, S, B>
{
}

// Context array position markers
impl crate::Zst for crate::combinators::A0 {}
impl crate::Zst for crate::combinators::A1 {}
impl crate::Zst for crate::combinators::A2 {}
impl crate::Zst for crate::combinators::A3 {}

impl<A: crate::Zst, const I: usize> crate::Zst for crate::combinators::CtxVar<A, I> {}

// ContextFree wraps manifold references for context-extended domains.
// While not technically zero-sized (contains a reference), we treat it as Zst
// to enable Copy for expression trees containing manifold params.
// This is safe because ContextFree only wraps references, which are trivially copyable.
impl<M: Copy> crate::Zst for crate::combinators::context::ContextFree<M> {}

// Derivative accessor combinators: ZST when M is ZST
impl<M: crate::Zst> crate::Zst for crate::ops::derivative::ValOf<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::derivative::DxOf<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::derivative::DyOf<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::derivative::DzOf<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::derivative::DxxOf<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::derivative::DxyOf<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::derivative::DyyOf<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::derivative::GradientMag2D<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::derivative::GradientMag3D<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::derivative::Antialias2D<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::derivative::Antialias3D<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::derivative::Normalized2D<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::derivative::Normalized3D<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::derivative::Curvature2D<M> {}

// ============================================================================
// Operator Trait Implementations (Enumerate operators once)
// ============================================================================

// Binary operators
impl<L, R> BinaryOp for crate::ops::Add<L, R> {
    type Left = L;
    type Right = R;
}
impl<L, R> BinaryOp for crate::ops::Sub<L, R> {
    type Left = L;
    type Right = R;
}
impl<L, R> BinaryOp for crate::ops::Mul<L, R> {
    type Left = L;
    type Right = R;
}
impl<L, R> BinaryOp for crate::ops::Div<L, R> {
    type Left = L;
    type Right = R;
}
impl<L, R> BinaryOp for crate::ops::Max<L, R> {
    type Left = L;
    type Right = R;
}
impl<L, R> BinaryOp for crate::ops::Min<L, R> {
    type Left = L;
    type Right = R;
}
impl<L, R> BinaryOp for crate::ops::MulRsqrt<L, R> {
    type Left = L;
    type Right = R;
}
impl<L, R> BinaryOp for crate::ops::Lt<L, R> {
    type Left = L;
    type Right = R;
}
impl<L, R> BinaryOp for crate::ops::Gt<L, R> {
    type Left = L;
    type Right = R;
}
impl<L, R> BinaryOp for crate::ops::Le<L, R> {
    type Left = L;
    type Right = R;
}
impl<L, R> BinaryOp for crate::ops::Ge<L, R> {
    type Left = L;
    type Right = R;
}
impl<L, R> BinaryOp for crate::ops::SoftGt<L, R> {
    type Left = L;
    type Right = R;
}
impl<L, R> BinaryOp for crate::ops::SoftLt<L, R> {
    type Left = L;
    type Right = R;
}
impl<L, R> BinaryOp for crate::ops::And<L, R> {
    type Left = L;
    type Right = R;
}
impl<L, R> BinaryOp for crate::ops::Or<L, R> {
    type Left = L;
    type Right = R;
}

// Unary operators
impl<M> UnaryOp for crate::ops::Sqrt<M> {
    type Inner = M;
}
impl<M> UnaryOp for crate::ops::Abs<M> {
    type Inner = M;
}
impl<M> UnaryOp for crate::ops::Floor<M> {
    type Inner = M;
}
impl<M> UnaryOp for crate::ops::Rsqrt<M> {
    type Inner = M;
}
impl<M> UnaryOp for crate::ops::Sin<M> {
    type Inner = M;
}
impl<M> UnaryOp for crate::ops::Cos<M> {
    type Inner = M;
}
impl<M> UnaryOp for crate::ops::Log2<M> {
    type Inner = M;
}
impl<M> UnaryOp for crate::ops::Exp2<M> {
    type Inner = M;
}
impl<M> UnaryOp for crate::ops::Neg<M> {
    type Inner = M;
}
impl<M> UnaryOp for crate::ops::BNot<M> {
    type Inner = M;
}

// Ternary operators
impl<A, B, C> TernaryOp for crate::ops::MulAdd<A, B, C> {
    type First = A;
    type Second = B;
    type Third = C;
}
impl<Acc, Val, Mask> TernaryOp for crate::ops::AddMasked<Acc, Val, Mask> {
    type First = Acc;
    type Second = Val;
    type Third = Mask;
}
impl<Mask, IfTrue, IfFalse> TernaryOp for crate::ops::SoftSelect<Mask, IfTrue, IfFalse> {
    type First = Mask;
    type Second = IfTrue;
    type Third = IfFalse;
}

// ============================================================================
// Zst Implementations for Operators
// ============================================================================
//
// Note: We must enumerate each operator because Rust's coherence rules can't
// prove that BinaryOp, UnaryOp, and TernaryOp are mutually exclusive, even
// though they're sealed. The compiler sees potential overlap and rejects
// blanket implementations.

// Binary operators: ZST + ZST → ZST
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::Add<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::Sub<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::Mul<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::Div<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::Max<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::Min<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::MulRsqrt<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::Lt<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::Gt<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::Le<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::Ge<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::SoftGt<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::SoftLt<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::And<L, R> {}
impl<L: crate::Zst, R: crate::Zst> crate::Zst for crate::ops::Or<L, R> {}

// Unary operators: ZST → ZST
impl<M: crate::Zst> crate::Zst for crate::ops::Sqrt<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::Abs<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::Floor<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::Rsqrt<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::Sin<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::Cos<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::Log2<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::Exp2<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::Exp<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::Neg<M> {}
impl<M: crate::Zst> crate::Zst for crate::ops::BNot<M> {}

// Ternary operators: ZST + ZST + ZST → ZST
impl<A: crate::Zst, B: crate::Zst, C: crate::Zst> crate::Zst for crate::ops::MulAdd<A, B, C> {}
impl<Acc: crate::Zst, Val: crate::Zst, Mask: crate::Zst> crate::Zst
    for crate::ops::AddMasked<Acc, Val, Mask>
{
}
impl<Mask: crate::Zst, IfTrue: crate::Zst, IfFalse: crate::Zst> crate::Zst
    for crate::ops::SoftSelect<Mask, IfTrue, IfFalse>
{
}

// ============================================================================
// Blanket Implementations for Combinators
// ============================================================================

// Select combinator: ZST + ZST + ZST → ZST
impl<C: crate::Zst, T: crate::Zst, F: crate::Zst> crate::Zst
    for crate::combinators::Select<C, T, F>
{
}

// Map combinator: ZST → ZST (but only if T is also ZST)
impl<M: crate::Zst, T: crate::Zst> crate::Zst for crate::combinators::Map<M, T> {}

// Project combinator: ZST → ZST (D is a dimension marker, always ZST)
impl<M: crate::Zst, D: crate::variables::Dimension + Copy> crate::Zst
    for crate::combinators::Project<M, D>
{
}

// At combinator: ZST + ZST + ZST + ZST + ZST → ZST
impl<Cx: crate::Zst, Cy: crate::Zst, Cz: crate::Zst, Cw: crate::Zst, M: crate::Zst> crate::Zst
    for crate::combinators::At<Cx, Cy, Cz, Cw, M>
{
}

// Fix combinator: ZST + ZST + ZST → ZST
impl<Seed: crate::Zst, Step: crate::Zst, Done: crate::Zst> crate::Zst
    for crate::combinators::Fix<Seed, Step, Done>
{
}

// ============================================================================
// Copy Implementations for ZST Operators (Step 3)
// ============================================================================
//
// This is the key insight: we only want Copy on zero-sized types.
// Large expression trees should not be implicitly copied.
//
// Strategy:
// 1. Remove #[derive(Copy)] from all operators/combinators
// 2. Manually impl Copy only when all type parameters are Zst + Copy
// 3. Result: Add<X, Y> is Copy, but Add<Field, Field> is not
//
// Note: We must enumerate each operator because Copy is a foreign trait.
// Blanket implementations on type parameters violate orphan rules.

// Binary operators: Copy when both operands are ZST + Copy
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::Add<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::Sub<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::Mul<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::Div<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::Max<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::Min<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::MulRsqrt<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::Lt<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::Gt<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::Le<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::Ge<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::SoftGt<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::SoftLt<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::And<L, R> {}
impl<L: crate::Zst + Copy, R: crate::Zst + Copy> Copy for crate::ops::Or<L, R> {}

// Unary operators: Copy when operand is ZST + Copy
impl<M: crate::Zst + Copy> Copy for crate::ops::Sqrt<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::Abs<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::Floor<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::Rsqrt<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::Sin<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::Cos<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::Log2<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::Exp2<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::Exp<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::Neg<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::BNot<M> {}

// Ternary operators: Copy when all operands are ZST + Copy
impl<A: crate::Zst + Copy, B: crate::Zst + Copy, C: crate::Zst + Copy> Copy
    for crate::ops::MulAdd<A, B, C>
{
}
impl<Acc: crate::Zst + Copy, Val: crate::Zst + Copy, Mask: crate::Zst + Copy> Copy
    for crate::ops::AddMasked<Acc, Val, Mask>
{
}
impl<Mask: crate::Zst + Copy, IfTrue: crate::Zst + Copy, IfFalse: crate::Zst + Copy> Copy
    for crate::ops::SoftSelect<Mask, IfTrue, IfFalse>
{
}

// Combinators: Copy when all parameters are ZST
impl<C: crate::Zst + Copy, T: crate::Zst + Copy, F: crate::Zst + Copy> Copy
    for crate::combinators::Select<C, T, F>
{
}
impl<M: crate::Zst + Copy, T: crate::Zst + Copy> Copy for crate::combinators::Map<M, T> {}
impl<M: crate::Zst + Copy, D: crate::variables::Dimension + Copy> Copy
    for crate::combinators::Project<M, D>
{
}
impl<
    Cx: crate::Zst + Copy,
    Cy: crate::Zst + Copy,
    Cz: crate::Zst + Copy,
    Cw: crate::Zst + Copy,
    M: crate::Zst + Copy,
> Copy for crate::combinators::At<Cx, Cy, Cz, Cw, M>
{
}
impl<Seed: crate::Zst + Copy, Step: crate::Zst + Copy, Done: crate::Zst + Copy> Copy
    for crate::combinators::Fix<Seed, Step, Done>
{
}

// Derivative accessor combinators: Copy when M is ZST + Copy
impl<M: crate::Zst + Copy> Copy for crate::ops::derivative::ValOf<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::derivative::DxOf<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::derivative::DyOf<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::derivative::DzOf<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::derivative::DxxOf<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::derivative::DxyOf<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::derivative::DyyOf<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::derivative::GradientMag2D<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::derivative::GradientMag3D<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::derivative::Antialias2D<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::derivative::Antialias3D<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::derivative::Normalized2D<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::derivative::Normalized3D<M> {}
impl<M: crate::Zst + Copy> Copy for crate::ops::derivative::Curvature2D<M> {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    // Helper function to assert a type is a ZST
    fn assert_zst<T: crate::Zst>() {
        assert_eq!(core::mem::size_of::<T>(), 0);
    }

    #[test]
    fn test_coordinate_variables_are_zst() {
        assert_zst::<crate::variables::X>();
        assert_zst::<crate::variables::Y>();
        assert_zst::<crate::variables::Z>();
        assert_zst::<crate::variables::W>();
    }

    #[test]
    fn test_spherical_harmonics_are_zst() {
        assert_zst::<crate::combinators::SphericalHarmonic<0, 0>>();
        assert_zst::<crate::combinators::SphericalHarmonic<1, -1>>();
        assert_zst::<crate::combinators::ZonalHarmonic<0>>();
        assert_zst::<crate::combinators::ZonalHarmonic<2>>();
        assert_zst::<crate::combinators::ShProject<9>>();
    }

    #[test]
    fn test_operators_with_zst_operands_are_zst() {
        use crate::variables::{X, Y};

        // Binary operators
        assert_zst::<crate::ops::Add<X, Y>>();
        assert_zst::<crate::ops::Sub<X, Y>>();
        assert_zst::<crate::ops::Mul<X, Y>>();
        assert_zst::<crate::ops::Div<X, Y>>();

        // Unary operators
        assert_zst::<crate::ops::Sqrt<X>>();
        assert_zst::<crate::ops::Abs<Y>>();
        assert_zst::<crate::ops::Sin<X>>();
        assert_zst::<crate::ops::Cos<Y>>();

        // Comparison operators
        assert_zst::<crate::ops::Lt<X, Y>>();
        assert_zst::<crate::ops::Gt<X, Y>>();
    }

    #[test]
    fn test_complex_expression_is_zst() {
        use crate::variables::{X, Y};

        // Test that (X * X + Y * Y).sqrt() is a ZST
        type XSquared = crate::ops::Mul<X, X>;
        type YSquared = crate::ops::Mul<Y, Y>;
        type Sum = crate::ops::Add<XSquared, YSquared>;
        type Distance = crate::ops::Sqrt<Sum>;

        assert_zst::<Distance>();
    }

    #[test]
    fn test_combinators_with_zst_operands_are_zst() {
        use crate::variables::{X, Y, Z};

        // Select combinator
        type Cond = crate::ops::Gt<X, Y>;
        type Select = crate::combinators::Select<Cond, X, Z>;
        assert_zst::<Select>();

        // At combinator
        type At = crate::combinators::At<X, Y, Z, X, Y>;
        assert_zst::<At>();
    }
}
