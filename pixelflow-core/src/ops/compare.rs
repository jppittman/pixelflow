//! Comparison operations: element-wise lt, le, gt, ge, and soft comparisons.

use crate::Manifold;
use crate::jet::Jet2;
use crate::numeric::Computational;
use pixelflow_compiler::Element;

// ============================================================================
// Hard Comparisons (generic over Numeric)
// ============================================================================

/// Less than: L < R
#[derive(Clone, Debug, Element)]
pub struct Lt<L, R>(pub L, pub R);

/// Greater than: L > R
#[derive(Clone, Debug, Element)]
pub struct Gt<L, R>(pub L, pub R);

/// Less than or equal: L <= R
#[derive(Clone, Debug, Element)]
pub struct Le<L, R>(pub L, pub R);

/// Greater than or equal: L >= R
#[derive(Clone, Debug, Element)]
pub struct Ge<L, R>(pub L, pub R);

/// Equality comparison: L == R
#[derive(Clone, Debug, Element)]
pub struct Eq<L, R>(pub L, pub R);

/// Inequality comparison: L != R
#[derive(Clone, Debug, Element)]
pub struct Ne<L, R>(pub L, pub R);

// Select is defined in combinators/select.rs with early-exit optimization.
// Use `pixelflow_core::Select` from there.

// ============================================================================
// Domain-Generic Manifold Implementations for Comparisons
// ============================================================================
//
// Comparisons normalize both sides to Field via Into<Field>, enabling:
// - Field < Field (identity conversion)
// - Jet3 < Jet3 (both convert via .val())
// - Field < Jet3 (cross-type, both normalize to Field)
// - Jet3 < Field (cross-type, both normalize to Field)
//
// Output is always Field, which is the sensible type for boolean masks.

impl<P, L, R, OL, OR> Manifold<P> for Lt<L, R>
where
    P: Copy + Send + Sync,
    L: Manifold<P, Output = OL>,
    R: Manifold<P, Output = OR>,
    OL: Into<crate::Field> + Copy,
    OR: Into<crate::Field> + Copy,
{
    type Output = crate::Field;
    #[inline(always)]
    fn eval(&self, p: P) -> crate::Field {
        let l: crate::Field = self.0.eval(p).into();
        let r: crate::Field = self.1.eval(p).into();
        l.lt(r)
    }
}

impl<P, L, R, OL, OR> Manifold<P> for Gt<L, R>
where
    P: Copy + Send + Sync,
    L: Manifold<P, Output = OL>,
    R: Manifold<P, Output = OR>,
    OL: Into<crate::Field> + Copy,
    OR: Into<crate::Field> + Copy,
{
    type Output = crate::Field;
    #[inline(always)]
    fn eval(&self, p: P) -> crate::Field {
        let l: crate::Field = self.0.eval(p).into();
        let r: crate::Field = self.1.eval(p).into();
        l.gt(r)
    }
}

impl<P, L, R, OL, OR> Manifold<P> for Le<L, R>
where
    P: Copy + Send + Sync,
    L: Manifold<P, Output = OL>,
    R: Manifold<P, Output = OR>,
    OL: Into<crate::Field> + Copy,
    OR: Into<crate::Field> + Copy,
{
    type Output = crate::Field;
    #[inline(always)]
    fn eval(&self, p: P) -> crate::Field {
        let l: crate::Field = self.0.eval(p).into();
        let r: crate::Field = self.1.eval(p).into();
        l.le(r)
    }
}

impl<P, L, R, OL, OR> Manifold<P> for Ge<L, R>
where
    P: Copy + Send + Sync,
    L: Manifold<P, Output = OL>,
    R: Manifold<P, Output = OR>,
    OL: Into<crate::Field> + Copy,
    OR: Into<crate::Field> + Copy,
{
    type Output = crate::Field;
    #[inline(always)]
    fn eval(&self, p: P) -> crate::Field {
        let l: crate::Field = self.0.eval(p).into();
        let r: crate::Field = self.1.eval(p).into();
        l.ge(r)
    }
}

impl<P, L, R, OL, OR> Manifold<P> for Eq<L, R>
where
    P: Copy + Send + Sync,
    L: Manifold<P, Output = OL>,
    R: Manifold<P, Output = OR>,
    OL: Into<crate::Field> + Copy,
    OR: Into<crate::Field> + Copy,
{
    type Output = crate::Field;
    #[inline(always)]
    fn eval(&self, p: P) -> crate::Field {
        let l: crate::Field = self.0.eval(p).into();
        let r: crate::Field = self.1.eval(p).into();
        l.eq(r)
    }
}

impl<P, L, R, OL, OR> Manifold<P> for Ne<L, R>
where
    P: Copy + Send + Sync,
    L: Manifold<P, Output = OL>,
    R: Manifold<P, Output = OR>,
    OL: Into<crate::Field> + Copy,
    OR: Into<crate::Field> + Copy,
{
    type Output = crate::Field;
    #[inline(always)]
    fn eval(&self, p: P) -> crate::Field {
        let l: crate::Field = self.0.eval(p).into();
        let r: crate::Field = self.1.eval(p).into();
        l.ne(r)
    }
}

// ============================================================================
// Smooth/Sigmoid Comparisons (Jet2-specific for gradients)
// ============================================================================

/// Smooth greater-than using sigmoid: sigmoid((L - R) / k).
/// Returns ~0 when L << R, ~1 when L >> R, smooth transition in between.
/// Smaller k = sharper transition.
///
/// **Jet2-specific**: Only works with Jet2 to provide smooth derivatives.
/// For Field evaluation, use hard Gt.
#[derive(Clone, Debug, Element)]
pub struct SoftGt<L, R> {
    /// Left operand.
    pub left: L,
    /// Right operand.
    pub right: R,
    /// Transition sharpness (smaller = sharper).
    pub sharpness: f32,
}

/// Smooth less-than: sigmoid((R - L) / k).
/// **Jet2-specific** for smooth derivatives.
#[derive(Clone, Debug, Element)]
pub struct SoftLt<L, R> {
    /// Left operand.
    pub left: L,
    /// Right operand.
    pub right: R,
    /// Transition sharpness (smaller = sharper).
    pub sharpness: f32,
}

/// Smooth select: blend between if_false and if_true based on smooth mask.
/// result = if_false + mask * (if_true - if_false)
///
/// **Always returns Jet2** and **only takes Manifold<Jet2> inputs**.
/// For Field select, use hard Select.
#[derive(Clone, Debug, Element)]
pub struct SoftSelect<Mask, IfTrue, IfFalse> {
    /// The smooth mask (0.0 to 1.0).
    pub mask: Mask,
    /// Value when mask is 1.0.
    pub if_true: IfTrue,
    /// Value when mask is 0.0.
    pub if_false: IfFalse,
}

// Hermite interpolation coefficients for smoothstep: 3t² - 2t³
const HERMITE_CUBIC: f32 = -2.0;
const HERMITE_QUAD: f32 = 3.0;

/// Smooth sigmoid via Hermite polynomial (smoothstep).
/// t = clamp((diff/k + 1)/2, 0, 1)
/// result = 3t² - 2t³
#[inline(always)]
fn smoothstep_sigmoid(diff: Jet2, sharpness: f32) -> Jet2 {
    let k = Jet2::from_f32(sharpness);
    let t = ((diff / k) + Jet2::from_f32(1.0)) / Jet2::from_f32(2.0);
    let t = t.max(Jet2::from_f32(0.0)).min(Jet2::from_f32(1.0));

    let t2 = t * t;
    let t3 = t2 * t;
    t3 * Jet2::from_f32(HERMITE_CUBIC) + t2 * Jet2::from_f32(HERMITE_QUAD)
}

// SoftGt, SoftLt, SoftSelect are Jet2-specific and work on 4D Jet2 domains
impl<L, R> Manifold<(Jet2, Jet2, Jet2, Jet2)> for SoftGt<L, R>
where
    L: Manifold<(Jet2, Jet2, Jet2, Jet2), Output = Jet2>,
    R: Manifold<(Jet2, Jet2, Jet2, Jet2), Output = Jet2>,
{
    type Output = Jet2;
    #[inline(always)]
    fn eval(&self, p: (Jet2, Jet2, Jet2, Jet2)) -> Jet2 {
        let left_val = self.left.eval(p);
        let right_val = self.right.eval(p);
        let diff = left_val - right_val;

        smoothstep_sigmoid(diff, self.sharpness)
    }
}

impl<L, R> Manifold<(Jet2, Jet2, Jet2, Jet2)> for SoftLt<L, R>
where
    L: Manifold<(Jet2, Jet2, Jet2, Jet2), Output = Jet2>,
    R: Manifold<(Jet2, Jet2, Jet2, Jet2), Output = Jet2>,
{
    type Output = Jet2;
    #[inline(always)]
    fn eval(&self, p: (Jet2, Jet2, Jet2, Jet2)) -> Jet2 {
        let left_val = self.left.eval(p);
        let right_val = self.right.eval(p);
        let diff = right_val - left_val; // Reversed for Lt

        smoothstep_sigmoid(diff, self.sharpness)
    }
}

/// SoftSelect always returns Jet2, only takes Manifold inputs on 4D Jet2 domain
impl<Mask, IfTrue, IfFalse> Manifold<(Jet2, Jet2, Jet2, Jet2)> for SoftSelect<Mask, IfTrue, IfFalse>
where
    Mask: Manifold<(Jet2, Jet2, Jet2, Jet2), Output = Jet2>,
    IfTrue: Manifold<(Jet2, Jet2, Jet2, Jet2), Output = Jet2>,
    IfFalse: Manifold<(Jet2, Jet2, Jet2, Jet2), Output = Jet2>,
{
    type Output = Jet2;
    #[inline(always)]
    fn eval(&self, p: (Jet2, Jet2, Jet2, Jet2)) -> Jet2 {
        let mask_val = self.mask.eval(p);
        let true_val = self.if_true.eval(p);
        let false_val = self.if_false.eval(p);

        // Linear blend with smooth mask
        false_val + mask_val * (true_val - false_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Field, Mask, PARALLELISM};

    type Field4 = (Field, Field, Field, Field);
    type Jet2_4 = (Jet2, Jet2, Jet2, Jet2);

    fn first_lane(f: Field) -> f32 {
        let mut buf = [0.0f32; PARALLELISM];
        f.store(&mut buf);
        buf[0]
    }

    /// Comparisons produce a `Field` mask; `Mask::from` is the crate's
    /// public bridge from that mask-as-Field back to boolean truth.
    fn eval_bool<M: Manifold<Field4, Output = Field>>(cond: M) -> bool {
        let result = cond.eval((
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        ));
        Mask::from(result).all()
    }

    // Constructed directly as `Le(l, r)` / `Ge(l, r)` / etc. (this file's
    // own tuple structs, re-exported at the crate root) rather than via
    // `Field`'s inherent `Algebra` comparison methods of the same name,
    // so these tests exercise the `Manifold::eval` bodies defined above.

    #[test]
    fn le_covers_less_equal_and_greater() {
        assert!(eval_bool(Le(Field::from(1.0), Field::from(2.0))));
        assert!(eval_bool(Le(Field::from(2.0), Field::from(2.0))));
        assert!(!eval_bool(Le(Field::from(3.0), Field::from(2.0))));
    }

    #[test]
    fn ge_covers_greater_equal_and_less() {
        assert!(eval_bool(Ge(Field::from(3.0), Field::from(2.0))));
        assert!(eval_bool(Ge(Field::from(2.0), Field::from(2.0))));
        assert!(!eval_bool(Ge(Field::from(1.0), Field::from(2.0))));
    }

    #[test]
    fn eq_covers_equal_and_unequal() {
        assert!(eval_bool(Eq(Field::from(2.0), Field::from(2.0))));
        assert!(!eval_bool(Eq(Field::from(2.0), Field::from(3.0))));
    }

    #[test]
    fn ne_covers_unequal_and_equal() {
        assert!(eval_bool(Ne(Field::from(2.0), Field::from(3.0))));
        assert!(!eval_bool(Ne(Field::from(2.0), Field::from(2.0))));
    }

    fn jet_domain(val: f32) -> Jet2_4 {
        let jet = Jet2::constant(Field::from(val));
        (jet, jet, jet, jet)
    }

    #[test]
    fn soft_gt_saturates_above_and_below_boundary() {
        let far_above = SoftGt {
            left: Jet2::constant(Field::from(10.0)),
            right: Jet2::constant(Field::from(0.0)),
            sharpness: 0.5,
        };
        let far_below = SoftGt {
            left: Jet2::constant(Field::from(-10.0)),
            right: Jet2::constant(Field::from(0.0)),
            sharpness: 0.5,
        };
        assert!((first_lane(far_above.eval(jet_domain(0.0)).val) - 1.0).abs() < 1e-3);
        assert!(first_lane(far_below.eval(jet_domain(0.0)).val).abs() < 1e-3);
    }

    #[test]
    fn soft_gt_at_boundary_is_half() {
        let at_boundary = SoftGt {
            left: Jet2::constant(Field::from(5.0)),
            right: Jet2::constant(Field::from(5.0)),
            sharpness: 0.5,
        };
        assert!((first_lane(at_boundary.eval(jet_domain(0.0)).val) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn soft_lt_saturates_above_and_below_boundary() {
        let far_above = SoftLt {
            left: Jet2::constant(Field::from(10.0)),
            right: Jet2::constant(Field::from(0.0)),
            sharpness: 0.5,
        };
        let far_below = SoftLt {
            left: Jet2::constant(Field::from(-10.0)),
            right: Jet2::constant(Field::from(0.0)),
            sharpness: 0.5,
        };
        assert!(first_lane(far_above.eval(jet_domain(0.0)).val).abs() < 1e-3);
        assert!((first_lane(far_below.eval(jet_domain(0.0)).val) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn soft_select_picks_true_and_false_branches() {
        let pick_true = SoftSelect {
            mask: Jet2::constant(Field::from(1.0)),
            if_true: Jet2::constant(Field::from(7.0)),
            if_false: Jet2::constant(Field::from(3.0)),
        };
        let pick_false = SoftSelect {
            mask: Jet2::constant(Field::from(0.0)),
            if_true: Jet2::constant(Field::from(7.0)),
            if_false: Jet2::constant(Field::from(3.0)),
        };
        assert!((first_lane(pick_true.eval(jet_domain(0.0)).val) - 7.0).abs() < 1e-6);
        assert!((first_lane(pick_false.eval(jet_domain(0.0)).val) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn soft_select_blends_at_half_mask() {
        let blended = SoftSelect {
            mask: Jet2::constant(Field::from(0.5)),
            if_true: Jet2::constant(Field::from(10.0)),
            if_false: Jet2::constant(Field::from(0.0)),
        };
        assert!((first_lane(blended.eval(jet_domain(0.0)).val) - 5.0).abs() < 1e-6);
    }
}
