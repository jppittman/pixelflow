//! A constant is a typed literal.
//!
//! One lane is 32 bits, and what those bits *mean* is the literal's type,
//! not a convention a reader recovers from the pattern. [`Lit::Real`] is a
//! number; [`Lit::Mask`] is a comparison's answer, which every backend
//! executes as an all-ones or all-zero lane and blends bitwise (`If`'s final
//! select, `BitAnd`, `BitOr`). The two never alias: a true mask and the NaN
//! whose bits it shares are different literals, so they are different arena
//! nodes and different e-classes, and nothing downstream has to ask "is this
//! NaN really a mask?".
//!
//! Equality, hashing and order are by *typed value*: a real by its bits (so
//! `-0.0` and `0.0` are two constants and a NaN payload is equal to itself,
//! which interning and the content key both need), a mask by its truth.
//! `f32::eq` is neither reflexive nor what a key wants, so `Lit` implements
//! the traits itself rather than deriving them.
//!
//! An integer literal is not here yet: the integer-domain primitives
//! (`TruncToInt`, `IAdd`, `Shl`, …) still carry their result in a real lane
//! by convention, and typing them is the reinterpretation CL's to pay for
//! (`Bits`/`Float`), after which `Lit::Int` has an op to come from.

use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};

/// A constant: a number, or a mask.
#[derive(Clone, Copy, Debug)]
pub enum Lit {
    /// A number, held as the `f32` it denotes.
    Real(f32),
    /// A comparison's answer: `true` is an all-ones lane, `false` all-zero.
    Mask(bool),
}

impl Lit {
    /// The number, if this literal is one.
    #[inline]
    #[must_use]
    pub const fn real(self) -> Option<f32> {
        match self {
            Self::Real(v) => Some(v),
            Self::Mask(_) => None,
        }
    }

    /// The truth, if this literal is a mask.
    #[inline]
    #[must_use]
    pub const fn mask(self) -> Option<bool> {
        match self {
            Self::Mask(b) => Some(b),
            Self::Real(_) => None,
        }
    }

    /// The 32-bit lane this literal tabulates to: a real's IEEE 754 bits, a
    /// mask's all-ones or all-zero.
    ///
    /// The data plane's view of a constant, for the backend that materializes
    /// one and the formats that fingerprint an arena by its bits. It is a
    /// projection, not an identity: two literals with equal `bits()` need not
    /// be equal, which is the whole point of the type.
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u32 {
        match self {
            Self::Real(v) => v.to_bits(),
            Self::Mask(true) => u32::MAX,
            Self::Mask(false) => 0,
        }
    }

    /// The total order's key: the variant, then the value as the variant
    /// compares it.
    #[inline]
    const fn key(self) -> (u8, u32) {
        match self {
            Self::Real(v) => (0, v.to_bits()),
            Self::Mask(b) => (1, b as u32),
        }
    }
}

impl PartialEq for Lit {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl Eq for Lit {}

impl Hash for Lit {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key().hash(state);
    }
}

impl PartialOrd for Lit {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Lit {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.key().cmp(&other.key())
    }
}

impl fmt::Display for Lit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Real(v) => write!(f, "{v}"),
            Self::Mask(b) => write!(f, "{b}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Lit;

    /// A true mask and the NaN that shares its lane are two literals. This is
    /// the aliasing the type exists to close: keyed by bits they were one
    /// e-class, and a rule that fired on the NaN fired on the mask.
    #[test]
    fn a_true_mask_is_not_the_nan_with_its_bits() {
        let mask = Lit::Mask(true);
        let nan = Lit::Real(f32::from_bits(u32::MAX));
        assert_eq!(mask.bits(), nan.bits(), "they do share a lane");
        assert_ne!(mask, nan);
        assert_ne!(Lit::Mask(false), Lit::Real(0.0));
    }

    #[test]
    fn reals_compare_by_bits_so_signed_zeros_differ_and_nan_is_itself() {
        assert_ne!(Lit::Real(0.0), Lit::Real(-0.0));
        assert_eq!(Lit::Real(f32::NAN), Lit::Real(f32::NAN));
        assert_eq!(Lit::Real(1.5), Lit::Real(1.5));
    }

    #[test]
    fn a_mask_tabulates_to_all_ones_or_all_zero() {
        assert_eq!(Lit::Mask(true).bits(), u32::MAX);
        assert_eq!(Lit::Mask(false).bits(), 0);
        assert_eq!(Lit::Real(1.0).bits(), 0x3f80_0000);
    }

    #[test]
    fn each_variant_answers_only_its_own_projection() {
        assert_eq!(Lit::Real(2.0).real(), Some(2.0));
        assert_eq!(Lit::Real(2.0).mask(), None);
        assert_eq!(Lit::Mask(true).mask(), Some(true));
        assert_eq!(Lit::Mask(true).real(), None);
    }
}
