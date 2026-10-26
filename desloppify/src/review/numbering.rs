//! The numbers a model echoes back, each a position in a list it counts. Three
//! lists are printed to it with a letter (`F1` findings and `T1` things in the
//! synthesis brief, `S1` a cluster's symptoms in the namer's); the fourth, the
//! diagnoses a root explains (`D1`), is its own reply's, counted in the order
//! it wrote them, and never printed. The synthesis prompt names `F` and `T`
//! too, so it and this file must agree; the golden transcript pins both.

use anyhow::{Context, Result};

/// A list as a model counts it: item `i` is `{prefix}{i + 1}`, and a number it
/// echoes resolves against the same list or is an error. The only place a
/// position is shifted by one.
pub(super) struct Numbering {
    prefix: char,
    len: usize,
}

impl Numbering {
    /// The findings as the synthesis numbers them: `F1`...
    pub(super) fn findings(len: usize) -> Self {
        Self { prefix: 'F', len }
    }

    /// The things as the synthesis numbers them: `T1`...
    pub(super) fn things(len: usize) -> Self {
        Self { prefix: 'T', len }
    }

    /// A reply's own diagnoses, as its roots number them: `D1`...
    pub(super) fn diagnoses(len: usize) -> Self {
        Self { prefix: 'D', len }
    }

    /// A cluster's symptoms, as the namer numbers them: `S1`...
    pub(super) fn symptoms(len: usize) -> Self {
        Self { prefix: 'S', len }
    }

    /// What the model reads: `F3` for the third item.
    pub(super) fn label(&self, position: usize) -> String {
        format!("{}{}", self.prefix, position + 1)
    }

    /// The item a number names: its position, as `mint` makes it an item.
    ///
    /// # Errors
    ///
    /// The number is 0 or past the list.
    pub(super) fn resolve<I>(&self, number: usize, mint: impl Fn(usize) -> I) -> Result<I> {
        number
            .checked_sub(1)
            .filter(|position| *position < self.len)
            .map(mint)
            .with_context(|| {
                format!(
                    "{}{number} names nothing: the list has {}",
                    self.prefix, self.len
                )
            })
    }

    /// Every item `numbers` name, in their order.
    ///
    /// # Errors
    ///
    /// Any number names nothing.
    pub(super) fn resolve_all<I>(
        &self,
        numbers: &[usize],
        mint: impl Fn(usize) -> I,
    ) -> Result<Vec<I>> {
        numbers.iter().map(|&n| self.resolve(n, &mint)).collect()
    }
}
