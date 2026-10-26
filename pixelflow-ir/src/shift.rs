//! How far a shift moves the bits of a word.
//!
//! The one place that says which counts a 32-bit lane can be shifted by. The
//! arena spells a count as a `Const` operand, where it can be any `f32`; this
//! is what is left once that spelling has been checked, so every consumer of a
//! count (the constant folder, the e-graph's shift node, the emitter's
//! immediate) asks the same question of the same type.

/// How far a shift moves the bits of a word: an integer in
/// `0..`[`ShiftCount::WORD_BITS`].
///
/// A count outside that range has three answers on the three targets (see
/// [`OpKind::fold_is_platform_specific`](crate::OpKind::fold_is_platform_specific)),
/// so it is not a count and has no value of this type. The one constructor
/// takes the arena's `f32` literal and refuses everything a word cannot be
/// shifted by.
///
/// The folder and the emitter ask this type rather than each testing the
/// range themselves, and it accepts exactly what each of them accepted before
/// it existed: `(0.0..32.0).contains(&literal) && (literal as u32) as f32 ==
/// literal`.
///
/// The field is a `u8` because the range is the word's width, which the
/// hardware fixes, and no program can grow it: a wider field would let no
/// program express a count more.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ShiftCount(u8);

impl ShiftCount {
    /// The width in bits of the 32-bit word a shift moves bits within, which
    /// bounds its count.
    pub const WORD_BITS: u32 = 32;

    /// The count a literal names, or `None` if it is not an integer in
    /// `0..`[`WORD_BITS`](Self::WORD_BITS).
    ///
    /// A count is an integer, and `-0.0` is the integer `0` (it compares equal
    /// to `0.0`, and every target shifts by zero for it), so it names the
    /// count `0`. It does not read back as `-0.0`: see [`Self::literal`].
    #[must_use]
    pub fn from_literal(literal: f32) -> Option<Self> {
        if !(literal >= 0.0 && literal < Self::WORD_BITS as f32) {
            return None;
        }
        // In range, so the cast is exact and total.
        let count = literal as u8;
        (f32::from(count) == literal).then_some(Self(count))
    }

    /// The count named by an integer literal.
    #[must_use]
    pub fn from_int(literal: i32) -> Option<Self> {
        (literal >= 0 && literal < Self::WORD_BITS as i32).then_some(Self(literal as u8))
    }

    /// The integer literal used by the typed arena spelling.
    #[must_use]
    pub const fn int_literal(self) -> i32 {
        self.0 as i32
    }

    /// The count as the literal the arena spells it with: `+0.0..31.0`. The
    /// count `0` reads back as `+0.0` even when it came from `-0.0`, which
    /// names the same count.
    #[must_use]
    pub fn literal(self) -> f32 {
        f32::from(self.0)
    }

    /// The count in bits.
    #[must_use]
    pub fn bits(self) -> u32 {
        u32::from(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every integer in range is a count and reads back as the literal it
    /// came from; the neighbours on both sides, and every value whose cast
    /// would alias a legal count, are not.
    #[test]
    fn a_count_is_exactly_an_integer_below_the_word_width() {
        for bits in 0..ShiftCount::WORD_BITS {
            let count = ShiftCount::from_literal(bits as f32).expect("in range");
            assert_eq!(count.bits(), bits);
            assert_eq!(count.literal(), bits as f32);
        }
        for refused in [
            -1.0,
            -0.5,
            0.5,
            2.5,
            31.5,
            32.0,
            33.0,
            256.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            assert_eq!(ShiftCount::from_literal(refused), None, "{refused}");
        }
    }

    /// A count is the integer a literal names, so `-0.0` (which compares equal
    /// to `0.0` and which every target shifts by zero for) is the count `0`.
    /// It reads back as the canonical `+0.0`, the same count spelled once.
    #[test]
    fn a_negative_zero_is_the_count_zero_and_reads_back_as_positive_zero() {
        let count = ShiftCount::from_literal(-0.0).expect("-0.0 is the integer 0");
        assert_eq!(Some(count), ShiftCount::from_literal(0.0));
        assert_eq!(count.bits(), 0);
        assert_eq!(count.literal().to_bits(), 0.0f32.to_bits());
    }

    /// Every count reads back as its canonical literal, and that literal is a
    /// count naming the same value: reading back loses nothing.
    #[test]
    fn a_count_reads_back_as_a_literal_naming_it() {
        for bits in 0..ShiftCount::WORD_BITS {
            let count = ShiftCount::from_literal(bits as f32).expect("in range");
            assert_eq!(count.literal().to_bits(), (bits as f32).to_bits());
            assert_eq!(ShiftCount::from_literal(count.literal()), Some(count));
        }
    }

    /// Counts order as the integers they are.
    #[test]
    fn counts_order_by_their_bits() {
        let count = |bits: u32| ShiftCount::from_literal(bits as f32).expect("in range");
        for bits in 1..ShiftCount::WORD_BITS {
            assert!(count(bits - 1) < count(bits), "{bits}");
        }
    }
}
