//! Binding bound-memory buffers to their declared slots for execution.
//!
//! An expression [`Environment`] declares buffers by *shape* ([`BufferDecl`]) via a
//! [`BufferId`]. Before a kernel that contains `Gather` nodes can run, each
//! slot must be bound to actual contents. This module provides the binding
//! used by the reference interpreter ([`crate::eval`]); the JIT path will
//! grow its own owned/`Arc` binding in M2 (see `KERNELS_AND_LATTICES.md`).
//!
//! Bindings here *borrow* their contents: a [`BindingTable`] is valid for the
//! duration of one evaluation, not the lifetime of a compiled kernel.

use crate::arena::{BufferId, UniformId, UniformIdentity};
use crate::expr::Environment;
use alloc::vec::Vec;

/// Why binding a buffer table failed. Binding fails loud rather than reading
/// out of bounds — consistent with the workspace's no-silent-failure rule.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BindError {
    /// The number of supplied slices does not match the arena's buffer count.
    Count {
        /// Buffers the arena declares.
        declared: usize,
        /// Slices supplied.
        supplied: usize,
    },
    /// A slice length does not match its declared `width * height`.
    Length {
        /// The offending slot.
        slot: u16,
        /// Length the declaration requires.
        expected: usize,
        /// Length supplied.
        actual: usize,
    },
    /// A uniform value was supplied for an identity the arena does not
    /// declare.
    Uniform(UniformIdentity),
}

impl core::fmt::Display for BindError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BindError::Count { declared, supplied } => write!(
                f,
                "binding count mismatch: graph declares {declared} buffer(s), {supplied} supplied"
            ),
            BindError::Length {
                slot,
                expected,
                actual,
            } => write!(
                f,
                "buffer slot {slot}: declared length {expected}, bound slice has {actual}"
            ),
            BindError::Uniform(id) => write!(f, "{id:?} is not a uniform this arena declares"),
        }
    }
}

/// Borrowed contents for every buffer an [`Environment`] declares, indexed by
/// [`BufferId`]. Row-major, `stride == width`, matching `BufferDecl`.
///
/// Also the values of the arena's uniforms — the oracle's block. They are
/// supplied **by identity** ([`BindingTable::bind_uniforms`]), never as a
/// positional slice: a compiled kernel's block is laid out in the link's
/// order and an arena's table in its own, and the two disagree in general,
/// so a `&[f32]` that means one of them cannot be the type that means the
/// other. A table bound without uniforms evaluates every one at its declared
/// default, which is what a bake without a block does.
#[derive(Clone, Debug)]
pub struct BindingTable<'a> {
    slots: Vec<&'a [f32]>,
    /// One value per uniform slot, in [`UniformId`] order — the arena's
    /// order, resolved from identities here — or empty for "every default".
    uniforms: Vec<f32>,
}

impl<'a> BindingTable<'a> {
    /// Bind a value to each of the named uniforms; every other slot keeps its
    /// declared default. Identities, not positions: the caller cannot get
    /// the order wrong, and a kernel's block converts by naming each entry.
    ///
    /// # Errors
    ///
    /// Returns [`BindError::Uniform`] for an identity the arena does not
    /// declare — a composition mistake, and the pixels would be plausible.
    pub fn bind_uniforms(
        mut self,
        env: &Environment,
        values: &[(UniformIdentity, f32)],
    ) -> Result<Self, BindError> {
        let decls = &env.uniforms;
        if self.uniforms.len() != decls.len() {
            self.uniforms = decls.iter().map(|d| d.default).collect();
        }
        for &(id, v) in values {
            let slot = decls
                .iter()
                .position(|d| d.id == id)
                .ok_or(BindError::Uniform(id))?;
            self.uniforms[slot] = v;
        }
        Ok(self)
    }

    /// The value bound to uniform slot `id`, or `None` to mean its default.
    #[inline]
    #[must_use]
    pub fn uniform(&self, id: UniformId) -> Option<f32> {
        self.uniforms.get(id.0 as usize).copied()
    }

    /// Bind `slices` to the graph's buffer slots, in [`BufferId`] order.
    ///
    /// Validates that the count and every length match the declarations, so a
    /// later `Gather` can index without bounds surprises.
    ///
    /// # Errors
    ///
    /// Returns [`BindError`] if the count or any length disagrees with the
    /// graph's [`BufferDecl`]s.
    pub fn bind(env: &Environment, slices: &[&'a [f32]]) -> Result<Self, BindError> {
        let decls = &env.buffers;
        if decls.len() != slices.len() {
            return Err(BindError::Count {
                declared: decls.len(),
                supplied: slices.len(),
            });
        }
        for (i, (decl, slice)) in decls.iter().zip(slices.iter()).enumerate() {
            let expected = decl.width as usize * decl.height as usize;
            if slice.len() != expected {
                return Err(BindError::Length {
                    slot: i as u16,
                    expected,
                    actual: slice.len(),
                });
            }
        }
        Ok(Self {
            slots: slices.to_vec(),
            uniforms: Vec::new(),
        })
    }

    /// An empty binding table, for arenas that declare no buffers.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            slots: Vec::new(),
            uniforms: Vec::new(),
        }
    }

    /// The contents bound to `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of range for this table.
    #[inline]
    #[must_use]
    pub fn slot(&self, id: BufferId) -> &'a [f32] {
        self.slots[id.0 as usize]
    }
}
