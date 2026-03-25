//! # Base Operator Overloads
//!
//! This module provides operator overloads for the base types (X, Y, Z, W, f32, i32),
//! enabling expressions like `X + Y` and `X * 2.0`.
//!
//! ## What It Does
//!
//! When you write `X + Y`, Rust needs `X` to implement `Add<Y>`.
//! This module provides those implementations.
//!
//! ## Generated Code Example
//!
//! For `X`, the macro generates:
//! ```rust,ignore
//! impl<Rhs> core::ops::Add<Rhs> for X {
//!     type Output = Add<X, Rhs>;
//!     fn add(self, rhs: Rhs) -> Self::Output { Add(self, rhs) }
//! }
//! // ... and Sub, Mul, Div similarly
//! ```
//!
//! Note: No `Manifold` bound on Rhs! Operators just construct AST nodes.
//! The Manifold bound is checked at evaluation time, not construction time.
//! This allows expressions like `X - Var::<N0>::new()` where Var<N> only
//! implements Manifold for domains with Head trait.
//!
//! ## Rsqrt Fusion
//!
//! When dividing by `Sqrt<R>`, the result is `MulRsqrt<L, R>` instead of `Div<L, Sqrt<R>>`.
//! This uses fast rsqrt (~3 cycles) instead of sqrt (~12) + div (~12).

use super::{
    Abs, Acos, Add, AddMasked, Asin, Atan, Ceil, Cos, Div, Exp, Exp2, Floor, Fract, Ln, Log10,
    Log2, Max, Min, Mul, MulAdd, MulRecip, MulRsqrt, Neg, Recip, Round, Rsqrt, Sin, Sqrt, Sub, Tan,
};
use crate::Field;
use crate::combinators::Select;
use crate::combinators::binding::Var;
use crate::combinators::context::CtxVar;
use crate::variables::{W, X, Y, Z};

// ============================================================================
// The Macro
// ============================================================================

/// Implements binary operators for a single base manifold type.
/// Includes rsqrt fusion: L / Sqrt<R> → MulRsqrt<L, R>
///
/// Note: No `Manifold` bound on the Rhs type! Operators construct AST nodes
/// without validating that operands are manifolds. Validation happens at
/// evaluation time, allowing expressions with Var<N> (which only implements
/// Manifold for domains with Head trait).
macro_rules! impl_binary_ops_for {
    ($ty:ty) => {
        impl<Rhs> core::ops::Add<Rhs> for $ty {
            type Output = Add<$ty, Rhs>;
            fn add(self, rhs: Rhs) -> Self::Output {
                Add(self, rhs)
            }
        }

        impl<Rhs> core::ops::Sub<Rhs> for $ty {
            type Output = Sub<$ty, Rhs>;
            fn sub(self, rhs: Rhs) -> Self::Output {
                Sub(self, rhs)
            }
        }

        impl<Rhs> core::ops::Mul<Rhs> for $ty {
            type Output = Mul<$ty, Rhs>;
            fn mul(self, rhs: Rhs) -> Self::Output {
                Mul(self, rhs)
            }
        }

        // Rsqrt fusion: L / Sqrt<R> → MulRsqrt<L, R>
        impl<R> core::ops::Div<Sqrt<R>> for $ty {
            type Output = MulRsqrt<$ty, R>;
            #[inline(always)]
            fn div(self, rhs: Sqrt<R>) -> Self::Output {
                MulRsqrt(self, rhs.0)
            }
        }

        // Enumerate all other divisor types to avoid conflict with Sqrt
        // Binary ops
        impl_base_div!($ty, Add<DL, DR>);
        impl_base_div!($ty, Sub<DL, DR>);
        impl_base_div!($ty, Mul<DL, DR>);
        impl_base_div!($ty, Div<DL, DR>);
        impl_base_div!($ty, Max<DL, DR>);
        impl_base_div!($ty, Min<DL, DR>);
        // Unary ops
        impl_base_div!($ty, Abs<DM>);
        impl_base_div!($ty, Floor<DM>);
        impl_base_div!($ty, Ceil<DM>);
        impl_base_div!($ty, Round<DM>);
        impl_base_div!($ty, Fract<DM>);
        impl_base_div!($ty, Rsqrt<DM>);
        impl_base_div!($ty, Sin<DM>);
        impl_base_div!($ty, Cos<DM>);
        impl_base_div!($ty, Tan<DM>);
        impl_base_div!($ty, Asin<DM>);
        impl_base_div!($ty, Acos<DM>);
        impl_base_div!($ty, Atan<DM>);
        impl_base_div!($ty, Neg<DM>);
        impl_base_div!($ty, Log2<DM>);
        impl_base_div!($ty, Exp2<DM>);
        impl_base_div!($ty, Exp<DM>);
        impl_base_div!($ty, Ln<DM>);
        impl_base_div!($ty, Log10<DM>);
        impl_base_div!($ty, Recip<DM>);
        // Ternary/compound ops
        impl_base_div!($ty, Select<DC, DT, DF>);
        impl_base_div!($ty, MulAdd<DA, DB, DC2>);
        impl_base_div!($ty, MulRecip<DM2>);
        impl_base_div!($ty, MulRsqrt<DL2, DR2>);
        impl_base_div!($ty, AddMasked<DAcc, DVal, DMask>);
        impl_base_div!($ty, Var<DN>);

        // CtxVar (kernel! macro constants) - special case due to const generic
        impl<__A, const __I: usize> core::ops::Div<CtxVar<__A, __I>> for $ty {
            type Output = Div<$ty, CtxVar<__A, __I>>;
            #[inline(always)]
            fn div(self, rhs: CtxVar<__A, __I>) -> Self::Output { Div(self, rhs) }
        }

        // Concrete divisor types
        impl_base_div_concrete!($ty, X);
        impl_base_div_concrete!($ty, Y);
        impl_base_div_concrete!($ty, Z);
        impl_base_div_concrete!($ty, W);
        impl_base_div_concrete!($ty, Field);
        impl_base_div_concrete!($ty, f32);
        impl_base_div_concrete!($ty, i32);
    };
}

/// Generate Div impl for base type with generic divisor
macro_rules! impl_base_div {
    ($self_ty:ty, $div_ty:ident <$($dg:ident),*>) => {
        impl<$($dg),*> core::ops::Div<$div_ty<$($dg),*>> for $self_ty {
            type Output = Div<$self_ty, $div_ty<$($dg),*>>;
            #[inline(always)]
            fn div(self, rhs: $div_ty<$($dg),*>) -> Self::Output { Div(self, rhs) }
        }
    };
}

/// Generate Div impl for base type with concrete divisor
macro_rules! impl_base_div_concrete {
    ($self_ty:ty, $div_ty:ty) => {
        impl core::ops::Div<$div_ty> for $self_ty {
            type Output = Div<$self_ty, $div_ty>;
            #[inline(always)]
            fn div(self, rhs: $div_ty) -> Self::Output {
                Div(self, rhs)
            }
        }
    };
}

// ============================================================================
// Apply to Base Types
// ============================================================================

impl_binary_ops_for!(X);
impl_binary_ops_for!(Y);
impl_binary_ops_for!(Z);
impl_binary_ops_for!(W);

// ============================================================================
// Scalar-on-LHS Operators (e.g., 1.0 - X)
// ============================================================================
//
// Orphan rules require specific impls for each AST type, not generic over trait.
// This macro generates `impl Add<AstType> for f32` for each AST type.

macro_rules! impl_scalar_lhs_ops {
    // For CtxVar with const generic (type + const INDEX)
    // Must be first to avoid matching ($ty:ty)
    (CtxVar) => {
        impl<__A, const __I: usize> core::ops::Add<CtxVar<__A, __I>> for f32 {
            type Output = Add<f32, CtxVar<__A, __I>>;
            #[inline(always)]
            fn add(self, rhs: CtxVar<__A, __I>) -> Self::Output { Add(self, rhs) }
        }
        impl<__A, const __I: usize> core::ops::Sub<CtxVar<__A, __I>> for f32 {
            type Output = Sub<f32, CtxVar<__A, __I>>;
            #[inline(always)]
            fn sub(self, rhs: CtxVar<__A, __I>) -> Self::Output { Sub(self, rhs) }
        }
        impl<__A, const __I: usize> core::ops::Mul<CtxVar<__A, __I>> for f32 {
            type Output = Mul<f32, CtxVar<__A, __I>>;
            #[inline(always)]
            fn mul(self, rhs: CtxVar<__A, __I>) -> Self::Output { Mul(self, rhs) }
        }
        impl<__A, const __I: usize> core::ops::Div<CtxVar<__A, __I>> for f32 {
            type Output = Div<f32, CtxVar<__A, __I>>;
            #[inline(always)]
            fn div(self, rhs: CtxVar<__A, __I>) -> Self::Output { Div(self, rhs) }
        }
    };
    ($ty:ident <$($gen:ident),*>) => {
        impl<$($gen),*> core::ops::Add<$ty<$($gen),*>> for f32 {
            type Output = Add<f32, $ty<$($gen),*>>;
            #[inline(always)]
            fn add(self, rhs: $ty<$($gen),*>) -> Self::Output { Add(self, rhs) }
        }
        impl<$($gen),*> core::ops::Sub<$ty<$($gen),*>> for f32 {
            type Output = Sub<f32, $ty<$($gen),*>>;
            #[inline(always)]
            fn sub(self, rhs: $ty<$($gen),*>) -> Self::Output { Sub(self, rhs) }
        }
        impl<$($gen),*> core::ops::Mul<$ty<$($gen),*>> for f32 {
            type Output = Mul<f32, $ty<$($gen),*>>;
            #[inline(always)]
            fn mul(self, rhs: $ty<$($gen),*>) -> Self::Output { Mul(self, rhs) }
        }
        impl<$($gen),*> core::ops::Div<$ty<$($gen),*>> for f32 {
            type Output = Div<f32, $ty<$($gen),*>>;
            #[inline(always)]
            fn div(self, rhs: $ty<$($gen),*>) -> Self::Output { Div(self, rhs) }
        }
    };
    // For non-generic types
    ($ty:ty) => {
        impl core::ops::Add<$ty> for f32 {
            type Output = Add<f32, $ty>;
            #[inline(always)]
            fn add(self, rhs: $ty) -> Self::Output { Add(self, rhs) }
        }
        impl core::ops::Sub<$ty> for f32 {
            type Output = Sub<f32, $ty>;
            #[inline(always)]
            fn sub(self, rhs: $ty) -> Self::Output { Sub(self, rhs) }
        }
        impl core::ops::Mul<$ty> for f32 {
            type Output = Mul<f32, $ty>;
            #[inline(always)]
            fn mul(self, rhs: $ty) -> Self::Output { Mul(self, rhs) }
        }
        impl core::ops::Div<$ty> for f32 {
            type Output = Div<f32, $ty>;
            #[inline(always)]
            fn div(self, rhs: $ty) -> Self::Output { Div(self, rhs) }
        }
    };
}

// Coordinate variables
impl_scalar_lhs_ops!(X);
impl_scalar_lhs_ops!(Y);
impl_scalar_lhs_ops!(Z);
impl_scalar_lhs_ops!(W);

// Binary ops
impl_scalar_lhs_ops!(Add<L, R>);
impl_scalar_lhs_ops!(Sub<L, R>);
impl_scalar_lhs_ops!(Mul<L, R>);
impl_scalar_lhs_ops!(Div<L, R>);
impl_scalar_lhs_ops!(Max<L, R>);
impl_scalar_lhs_ops!(Min<L, R>);

// Unary ops
impl_scalar_lhs_ops!(Sqrt<M>);
impl_scalar_lhs_ops!(Abs<M>);
impl_scalar_lhs_ops!(Floor<M>);
impl_scalar_lhs_ops!(Rsqrt<M>);
impl_scalar_lhs_ops!(Sin<M>);
impl_scalar_lhs_ops!(Cos<M>);

// Compound ops
impl_scalar_lhs_ops!(MulAdd<A, B, C>);
impl_scalar_lhs_ops!(MulRecip<M>);
impl_scalar_lhs_ops!(MulRsqrt<L, R>);

// Select
impl_scalar_lhs_ops!(Select<C, T, F>);

// Binding combinators (for kernel! macro)
impl_scalar_lhs_ops!(Var<N>);
impl_scalar_lhs_ops!(CtxVar);
