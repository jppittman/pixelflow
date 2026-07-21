//! Antialiasing via automatic differentiation.
//!
//! [`Antialiased`] evaluates a domain-generic coverage manifold over `Jet2`
//! coordinates seeded in screen space. Inside the font pipeline each edge
//! crossing computes a gradient-normalized ramp
//! `clamp(d / (‖∇d‖ + ε) + 0.5, 0, 1)`; with the derivatives seeded here, the
//! chain rule propagates ‖∇d‖ through every coordinate transform, so the ramp
//! is ~1 *screen* pixel wide at any glyph scale.
//!
//! The same manifold evaluated directly over `Field` coordinates (i.e. without
//! this wrapper) has zero derivatives everywhere, and the ramp degenerates to
//! the classic hard 0/1 step. `glyph` = hard, `Antialiased::new(glyph)` (or
//! `aa(glyph)`) = antialiased. No smoothstep — the gradient IS the
//! antialiasing.
//!
//! Two consumers:
//! - **Uncached text**: `aa(text(&font, s, size))` evaluates the analytical
//!   pipeline with AA every sample.
//! - **The glyph cache**: `fonts::cache::CachedGlyph::new` bakes through
//!   `Antialiased` once per (glyph, size bucket), collapsing the AA coverage
//!   to an f32 lattice that is then bilinearly indexed at render time.
//!
//! Hand-written combinator (not a `kernel!`): it changes the evaluation domain
//! (`Field` in, `Jet2` inside), which the kernel macro cannot express; this is
//! the same shape as `pixelflow_core::WithGradient`, but collapsing to a
//! `Field` coverage output instead of exposing the jet.

use pixelflow_core::jet::Jet2;
use pixelflow_core::{Field, Manifold};

/// The standard 4D Field domain type.
type Field4 = (Field, Field, Field, Field);

/// The 4D Jet2 domain type (2D autodiff seeded in screen space).
type Jet2x4 = (Jet2, Jet2, Jet2, Jet2);

/// Antialiased coverage: evaluates the inner manifold over `Jet2` coordinates.
///
/// Seeds `∂/∂x` and `∂/∂y` at the screen-space sample position; `z`/`w` are
/// constants. The inner manifold must be generic over the evaluation domain
/// and produce `Field` coverage (the whole glyph pipeline is).
///
/// # Example
///
/// ```ignore
/// let glyph = font.glyph_scaled('g', 16.0).unwrap(); // hard coverage
/// let smooth = Antialiased::new(glyph);              // antialiased coverage
/// ```
#[derive(Clone, Debug)]
pub struct Antialiased<M> {
    /// The coverage manifold to evaluate with autodiff coordinates.
    pub inner: M,
}

impl<M> Antialiased<M> {
    /// Wrap a coverage manifold for antialiased evaluation.
    #[inline(always)]
    pub fn new(inner: M) -> Self {
        Self { inner }
    }
}

/// Wrap a coverage manifold for antialiased evaluation (function form).
#[inline(always)]
pub fn aa<M>(inner: M) -> Antialiased<M>
where
    M: Manifold<Jet2x4, Output = Field>,
{
    Antialiased::new(inner)
}

impl<M> Manifold<Field4> for Antialiased<M>
where
    M: Manifold<Jet2x4, Output = Field>,
{
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Field {
        let (x, y, z, w) = p;
        // Seed screen-space derivatives: ∂x/∂x = 1, ∂y/∂y = 1.
        self.inner
            .eval((Jet2::x(x), Jet2::y(y), Jet2::constant(z), Jet2::constant(w)))
    }
}
