//! A shift's place in the e-graph: which way it moves the bits, and how the
//! arena's spelling of it is read.
//!
//! A shift is an operator with a parameter, not a binary operation: its count
//! is a [`ShiftCount`] that no rule may rewrite, no class holds, and nothing
//! repairs after extraction. The arena spells the count as a `Const` operand
//! (`Shl(x, Const)`), and that spelling is where the count can be anything an
//! `f32` can be. [`spelled`] checks the spelling, so a node built from it is
//! already a valid shift.

use pixelflow_ir::{Children, Ir, Lit, OpKind, Shape, ShiftCount};

/// Which way a shift moves the bits of a word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ShiftDirection {
    /// Toward the high bits: [`OpKind::Shl`].
    Left,
    /// Toward the low bits: [`OpKind::Shr`].
    Right,
}

impl ShiftDirection {
    /// The direction `op` shifts in, if it is a shift.
    #[must_use]
    pub fn of(op: OpKind) -> Option<Self> {
        match op {
            OpKind::Shl => Some(Self::Left),
            OpKind::Shr => Some(Self::Right),
            _ => None,
        }
    }

    /// The operation this direction names.
    #[must_use]
    pub fn op(self) -> OpKind {
        match self {
            Self::Left => OpKind::Shl,
            Self::Right => OpKind::Shr,
        }
    }
}

/// The operand and count of a shift as the arena spells it, `Shl(x, Const)`:
/// the operand's reference, and the count the second child's literal names.
/// `None` if that child is not a `Const` holding a valid count.
///
/// The one reading of the arena's spelling, for everything that carries a
/// term into a graph: the count is read here and never visited, so no class
/// is made for it.
pub(crate) fn spelled<I: Ir>(
    term: &I,
    children: Children<'_, I::Ref>,
) -> Option<(I::Ref, ShiftCount)> {
    let (operand, count) = (children.get(0)?, children.get(1)?);
    let Shape::Const(Lit::Real(literal)) = term.project(count) else {
        return None;
    };
    Some((operand, ShiftCount::from_literal(literal)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only the two shifts have a direction, and each names its own op.
    #[test]
    fn a_direction_names_its_op_and_only_shifts_have_one() {
        for direction in [ShiftDirection::Left, ShiftDirection::Right] {
            assert_eq!(ShiftDirection::of(direction.op()), Some(direction));
        }
        assert_eq!(ShiftDirection::of(OpKind::Add), None);
        assert_eq!(ShiftDirection::of(OpKind::Const), None);
    }
}
