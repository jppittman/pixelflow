//! The assembler's vocabulary: the names an emitted program's positions go by.
//!
//! Imports nothing from the crate.

/// The name of a position. Minted with the thing at that position — a block,
/// the constant pool — and nowhere else: there is no constructor but
/// [`Labels::mint`], so a label cannot be spelled, only handed on.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Label(u64);

/// A program's label namespace, and the only mint. Whoever is building the
/// program owns it, so minting needs `&mut` to it.
pub(super) struct Labels {
    next: u64,
}

impl Labels {
    pub(super) const fn new() -> Self {
        Self { next: 0 }
    }

    pub(super) fn mint(&mut self) -> Label {
        let label = Label(self.next);
        self.next += 1;
        label
    }
}

/// How a label field is filled once the label's position is known: fill the
/// displacement of the instruction emitted at `at` so that it reaches
/// `target`. Both are offsets from the start of the program. The encoder
/// that wrote the field supplies it — an x86 `rel32` four bytes in, an
/// aarch64 `imm19` five bits up in the word.
pub(super) type Patch = fn(code: &mut [u8], at: usize, target: usize);
