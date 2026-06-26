//! # Domain Traits for Generic Manifold Evaluation
//!
//! This module defines traits for accessing values from generic domain types.
//! These traits enable the domain-generic `Manifold<P>` abstraction where `P`
//! can be any domain type (2D, 3D, 4D coordinates, or extended with let bindings).
//!
//! ## Architecture
//!
//! The domain is a nested tuple that represents the evaluation context:
//!
//! - **2D base**: `(I, I)` - spatial coordinates (x, y)
//! - **3D base**: `(I, I, I)` - spatial coordinates (x, y, z)
//! - **4D base**: `(I, I, I, I)` - spatial coordinates (x, y, z, w)
//! - **Let binding**: `LetExtended<V, Rest>` - bound value V prepended to rest of domain
//!
//! ## Example
//!
//! ```ignore
//! // 2D kernel evaluates on (x, y)
//! let circle = (X * X + Y * Y).sqrt() - 1.0;
//! circle.eval((field_x, field_y))
//!
//! // With let binding
//! let optimized = Let(
//!     (X * X + Y * Y).sqrt(),  // val: compute distance once
//!     Var::<N0> - 1.0          // body: use it
//! );
//! // Domain flows: (x, y) → Let → LetExtended(dist, (x, y)) → body
//! ```

use crate::numeric::Computational;

// ============================================================================
// Spatial Trait: Access Spatial Coordinates
// ============================================================================

/// Access spatial coordinates from a domain.
///
/// This trait provides access to the underlying spatial coordinates (x, y, z, w)
/// regardless of how many let bindings have been layered on top.
///
/// The associated type `Coord` represents the coordinate value type (e.g., `Field`, `Jet2`).
///
/// # GLSL Zero-Padding Rule
///
/// Following GLSL/HLSL conventions, missing dimensions return zero:
/// - 2D domain `(I, I)`: `z()` and `w()` return `I::from_f32(0.0)`
/// - 3D domain `(I, I, I)`: `w()` returns `I::from_f32(0.0)`
/// - 4D domain `(I, I, I, I)`: all coordinates available
///
/// This makes kernels portable across dimensions.
pub trait Spatial {
    /// The coordinate value type (e.g., `Field`, `Jet2`).
    type Coord;
    /// The scalar type that coordinates reduce to via V() extraction.
    ///
    /// For `Field` domains, this is `Field`.
    /// For `Jet3` domains, this is also `Field` (the value component).
    /// Used by `.at()` coordinate transformations in kernels.
    type Scalar;
    /// Get the X coordinate.
    fn x(&self) -> Self::Coord;
    /// Get the Y coordinate.
    fn y(&self) -> Self::Coord;
    /// Get the Z coordinate (returns zero for 2D domains).
    fn z(&self) -> Self::Coord;
    /// Get the W coordinate (returns zero for 2D/3D domains).
    fn w(&self) -> Self::Coord;
}

// ============================================================================
// Head Trait: Access First Element of Domain Stack
// ============================================================================

/// Access the head (first element) of a domain stack.
///
/// Used by `Var<N0>` (index 0) to read the most recently bound value.
///
/// # Example
///
/// For domain `(v0, (v1, (x, y)))`:
/// - `Head::head()` returns `v0`
pub trait Head {
    /// The type of the head value.
    type Value;
    /// Get the head value.
    fn head(&self) -> Self::Value;
}

// ============================================================================
// Tail Trait: Access Rest of Domain Stack
// ============================================================================

/// Access the tail (rest) of a domain stack.
///
/// Used by `Var<N>` (where N > 0) to recurse through let bindings.
///
/// # Example
///
/// For domain `(v0, (v1, (x, y)))`:
/// - `Tail::tail()` returns `(v1, (x, y))`
pub trait Tail {
    /// The type of the tail.
    type Rest;
    /// Get the tail.
    fn tail(&self) -> Self::Rest;
}

// ============================================================================
// Spatial Implementations for Base Domains (Tuples)
// ============================================================================

// 2D base domain: (I, I)
// z and w are zero-padded per GLSL conventions
impl<I: Copy + Computational> Spatial for (I, I) {
    type Coord = I;
    type Scalar = crate::Field;
    #[inline(always)]
    fn x(&self) -> I {
        self.0
    }
    #[inline(always)]
    fn y(&self) -> I {
        self.1
    }
    #[inline(always)]
    fn z(&self) -> I {
        I::from_f32(0.0)
    }
    #[inline(always)]
    fn w(&self) -> I {
        I::from_f32(0.0)
    }
}

// 3D base domain: (I, I, I)
// w is zero-padded per GLSL conventions
impl<I: Copy + Computational> Spatial for (I, I, I) {
    type Coord = I;
    type Scalar = crate::Field;
    #[inline(always)]
    fn x(&self) -> I {
        self.0
    }
    #[inline(always)]
    fn y(&self) -> I {
        self.1
    }
    #[inline(always)]
    fn z(&self) -> I {
        self.2
    }
    #[inline(always)]
    fn w(&self) -> I {
        I::from_f32(0.0)
    }
}

// 4D base domain: (I, I, I, I)
// All coordinates available
impl<I: Copy> Spatial for (I, I, I, I) {
    type Coord = I;
    type Scalar = crate::Field;
    #[inline(always)]
    fn x(&self) -> I {
        self.0
    }
    #[inline(always)]
    fn y(&self) -> I {
        self.1
    }
    #[inline(always)]
    fn z(&self) -> I {
        self.2
    }
    #[inline(always)]
    fn w(&self) -> I {
        self.3
    }
}

// ============================================================================
// Head/Tail Implementations for Tuple Stacks
// ============================================================================

// For a 2-tuple (V, Rest), the head is V and tail is Rest
impl<V: Copy, Rest> Head for (V, Rest) {
    type Value = V;
    #[inline(always)]
    fn head(&self) -> V {
        self.0
    }
}

impl<V, Rest: Copy> Tail for (V, Rest) {
    type Rest = Rest;
    #[inline(always)]
    fn tail(&self) -> Rest {
        self.1
    }
}

// ============================================================================
// Spatial for Let-Extended Domains
// ============================================================================

// Recursive case: Let binding layer `(V, Rest)` delegates to Rest
// This allows X, Y, Z, W to "see through" let bindings to the base coordinates
//
// Note: This impl has a coherence issue with the base 2-tuple impl.
// We need to use a marker trait to distinguish base domains from let-extended domains.

/// Marker trait for base spatial domains.
/// These are the "bottom" of the domain stack: (I, I), (I, I, I), (I, I, I, I).
pub trait BaseDomain {}

impl<I> BaseDomain for (I, I) {}
impl<I> BaseDomain for (I, I, I) {}
impl<I> BaseDomain for (I, I, I, I) {}

/// Wrapper to mark a domain as let-extended.
///
/// This allows the type system to distinguish `LetExtended<V, Rest>` as a let-binding
/// from a base 2D domain `(I, I)`.
///
/// The wrapper preserves spatial access by delegating to `Rest`, allowing
/// X, Y, Z, W to "see through" let bindings to the base coordinates.
#[derive(Clone, Copy, Debug)]
pub struct LetExtended<V, Rest>(pub V, pub Rest);

impl<V: Copy, Rest: Copy> Head for LetExtended<V, Rest> {
    type Value = V;
    #[inline(always)]
    fn head(&self) -> V {
        self.0
    }
}

impl<V, Rest: Copy> Tail for LetExtended<V, Rest> {
    type Rest = Rest;
    #[inline(always)]
    fn tail(&self) -> Rest {
        self.1
    }
}

impl<V, Rest> Spatial for LetExtended<V, Rest>
where
    Rest: Spatial,
{
    type Coord = Rest::Coord;
    type Scalar = Rest::Scalar;
    #[inline(always)]
    fn x(&self) -> Self::Coord {
        self.1.x()
    }
    #[inline(always)]
    fn y(&self) -> Self::Coord {
        self.1.y()
    }
    #[inline(always)]
    fn z(&self) -> Self::Coord {
        self.1.z()
    }
    #[inline(always)]
    fn w(&self) -> Self::Coord {
        self.1.w()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Field;

    #[test]
    fn test_2d_spatial() {
        let domain = (Field::from(3.0), Field::from(4.0));
        let mut buf = [0.0f32; crate::PARALLELISM];

        domain.x().store(&mut buf);
        assert_eq!(buf[0], 3.0);

        domain.y().store(&mut buf);
        assert_eq!(buf[0], 4.0);

        // z and w should be zero-padded
        domain.z().store(&mut buf);
        assert_eq!(buf[0], 0.0);

        domain.w().store(&mut buf);
        assert_eq!(buf[0], 0.0);
    }

    #[test]
    fn test_4d_spatial() {
        let domain = (
            Field::from(1.0),
            Field::from(2.0),
            Field::from(3.0),
            Field::from(4.0),
        );
        let mut buf = [0.0f32; crate::PARALLELISM];

        domain.x().store(&mut buf);
        assert_eq!(buf[0], 1.0);

        domain.y().store(&mut buf);
        assert_eq!(buf[0], 2.0);

        domain.z().store(&mut buf);
        assert_eq!(buf[0], 3.0);

        domain.w().store(&mut buf);
        assert_eq!(buf[0], 4.0);
    }

    #[test]
    fn test_let_extended() {
        // Simulate: let v = 10.0 in ... on 2D domain
        let base = (Field::from(3.0), Field::from(4.0));
        let extended = LetExtended(Field::from(10.0), base);

        let mut buf = [0.0f32; crate::PARALLELISM];

        // Head should be the bound value
        extended.head().store(&mut buf);
        assert_eq!(buf[0], 10.0);

        // Spatial should see through to base
        extended.x().store(&mut buf);
        assert_eq!(buf[0], 3.0);

        extended.y().store(&mut buf);
        assert_eq!(buf[0], 4.0);
    }

    #[test]
    fn test_nested_let() {
        // let a = 10.0 in let b = 20.0 in ... on 2D domain
        let base = (Field::from(3.0), Field::from(4.0));
        let inner = LetExtended(Field::from(20.0), base); // b = 20.0
        let outer = LetExtended(Field::from(10.0), inner); // a = 10.0

        let mut buf = [0.0f32; crate::PARALLELISM];

        // Head of outer is 10.0
        outer.head().store(&mut buf);
        assert_eq!(buf[0], 10.0);

        // Head of tail is 20.0
        outer.tail().head().store(&mut buf);
        assert_eq!(buf[0], 20.0);

        // Spatial coordinates still accessible
        outer.x().store(&mut buf);
        assert_eq!(buf[0], 3.0);
    }
}
