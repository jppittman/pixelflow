//! The assembler: a function from an assembly program to its binary.
//!
//! Imports nothing from the crate.

use alloc::vec::Vec;

/// The name of a position. Minted with the thing at that position — a block,
/// the constant pool — and nowhere else: there is no constructor but
/// [`Labels::mint`], so a label cannot be spelled, only handed on.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Label(u64);

/// A program's label namespace, and the only mint. Whoever is building the
/// program owns it, so minting needs `&mut` to it.
#[derive(Default)]
pub(super) struct Labels {
    next: u64,
}

impl Labels {
    pub(super) fn mint(&mut self) -> Label {
        let label = Label(self.next);
        self.next += 1;
        label
    }
}

/// How a label field is filled once the label's position is known: write the
/// displacement of the field at `at` so that it reaches `target`. Both are
/// offsets from the start of the program. The encoder that wrote the field
/// supplies it — an x86 `rel32`, an aarch64 `imm19` five bits up in the word.
pub(super) type Patch = fn(code: &mut [u8], at: usize, target: usize);

/// One thing in a section of a program.
pub(super) enum Item<I> {
    /// The label names the address of whatever comes next.
    Bind(Label),
    /// An instruction; the assembler's caller says how it encodes.
    Inst(I),
    /// Zero bytes up to the next multiple of this.
    Align(u64),
    /// Bytes as they are.
    Bytes(Vec<u8>),
}

/// One kernel's assembly program: its code, then its data. The program owns
/// the mint of every label in it.
pub(super) struct AsmProgram<I> {
    pub(super) text: Vec<Item<I>>,
    pub(super) data: Vec<Item<I>>,
    pub(super) labels: Labels,
}

/// What an encoder writes into: one instruction's bytes, and its label fields.
pub(super) struct Encoding<'a> {
    code: &'a mut Vec<u8>,
    /// Where this instruction starts in `code`.
    start: usize,
    fields: &'a mut Vec<(usize, Label, Patch)>,
}

impl Encoding<'_> {
    pub(super) fn bytes(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
    }

    /// A field starting `at` bytes into this instruction that names `label`.
    pub(super) fn field(&mut self, at: usize, label: Label, patch: Patch) {
        self.fields.push((self.start + at, label, patch));
    }
}

/// Lay out `text` then `data`, encode each instruction with `encode`, and
/// fill in every label field.
///
/// # Panics
///
/// If a label is bound twice, bound by a program that did not mint it, or
/// named and never bound. Only this crate writes these programs, so each is a
/// bug here rather than a fact about the kernel being compiled.
pub(super) fn assemble<I>(
    program: &AsmProgram<I>,
    encode: impl Fn(&I, &mut Encoding<'_>),
) -> Vec<u8> {
    let mut code = Vec::new();
    let mut fields = Vec::new();
    let mut addresses: Vec<Option<usize>> = alloc::vec![None; program.labels.next as usize];
    for item in program.text.iter().chain(&program.data) {
        match item {
            Item::Bind(label) => {
                let Some(address) = addresses.get_mut(label.0 as usize) else {
                    panic!("{label:?} was not minted by this program")
                };
                assert!(
                    address.replace(code.len()).is_none(),
                    "{label:?} was written twice"
                );
            }
            Item::Inst(inst) => {
                let start = code.len();
                encode(
                    inst,
                    &mut Encoding {
                        code: &mut code,
                        start,
                        fields: &mut fields,
                    },
                );
            }
            Item::Align(to) => code.resize(code.len().next_multiple_of(*to as usize), 0),
            Item::Bytes(bytes) => code.extend_from_slice(bytes),
        }
    }
    for (at, label, patch) in fields {
        let Some(&Some(target)) = addresses.get(label.0 as usize) else {
            panic!("{label:?} is branched to but never written")
        };
        patch(&mut code, at, target);
    }
    code
}
