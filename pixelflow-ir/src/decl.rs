//! The declarations an expression graph's leaves index, and the index spaces
//! `Var` is drawn from.
//!
//! None of this is storage: a [`BufferDecl`] says what memory a `Buffer` leaf
//! names and a [`UniformDecl`] what scalar a `Uniform` leaf names, while the
//! graph itself lives in a [`Dag<ExprData>`](crate::dag::Dag). The two are
//! paired by an [`Environment`](crate::expr::Environment).
//!
//! Identity here is *provenance*, not position: a slot index means something
//! only to the one table it indexes, so merging two fragments has to compare
//! something that outlives either table. Minting is that something.

/// Coordinate axes a lattice has, and so the coordinate `Var` indices: `X = 0`,
/// `Y = 1`.
///
/// There were four. Z and W had extent 1 in every production call — an axis
/// that never varies is not an axis — so they left the language and the
/// scalars they carried became [`UniformDecl`]s
/// (docs/plans/2026-09-06-lattice-is-the-index.md).
pub const COORD_AXES: usize = 2;

/// The `Var` indices Z and W had. Reserved, never reissued: a reduction
/// binder taking one of them would make a graph written before the change
/// read back as a different program.
pub const RETIRED_COORD_AXES: [u8; 2] = [2, 3];

/// The first `Var` index a reduction binder takes.
///
/// Binder indices are dense from here, and *here* is past the reserved
/// [`RETIRED_COORD_AXES`] rather than past [`COORD_AXES`] — which is what
/// keeps them where they were when there were four axes. The gap between
/// the two is the reason `Var`'s meanings do not collide.
pub const REDUCE_BINDER_BASE: u8 = COORD_AXES as u8 + RETIRED_COORD_AXES.len() as u8;

/// How many reduction binders the index space holds, starting at
/// [`REDUCE_BINDER_BASE`] — the depth of nested folds a kernel may carry.
pub const REDUCE_BINDERS: u8 = 4;

/// The `Var` indices a reduction binder may take.
#[must_use]
pub fn reduce_binders() -> core::ops::Range<u8> {
    REDUCE_BINDER_BASE..REDUCE_BINDER_BASE + REDUCE_BINDERS
}

// ───────────────────────────────────────── Buffers ────────────────────────────

/// Slot index into an [`Environment`](crate::expr::Environment)'s buffer
/// table. Copy, 2 bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct BufferId(pub u16);

/// Which block of memory a declaration refers to, independent of any table.
///
/// [`BufferId`] is a slot index into *one* environment's table, so it cannot
/// answer "the same buffer?" across a merge — two fragments each call their own
/// buffer slot 0. Extents cannot answer it either: two atlases of equal size
/// are a coincidence, not a fact, and treating them as one would bind a single
/// pointer for both and silently read the wrong pixels.
///
/// So identity is provenance. You get one by minting it, and copy it into
/// every declaration that names that memory; nothing else can collide with it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct BufferIdentity(u32);

impl BufferIdentity {
    /// Mint an identity distinct from every other in this process.
    ///
    /// # Panics
    ///
    /// Panics if the counter is exhausted, rather than wrapping onto a live
    /// identity and aliasing two unrelated buffers.
    #[must_use]
    pub fn mint() -> Self {
        static NEXT: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
        Self(mint_identity(&NEXT, "BufferIdentity"))
    }
}

/// The one counter discipline behind every provenance identity.
///
/// `fetch_add` + assert was wrong: the add WRAPS before the assert fires, so
/// if that panic is ever caught — or merely unwinds a non-fatal worker thread
/// — the counter has already returned to 0 and the next mint hands out an
/// identity that is still live. Two unrelated buffers (or uniforms) would
/// then compare identical and merge into one splice/JIT slot. `fetch_update`
/// declining to store leaves the counter permanently exhausted instead.
fn mint_identity(counter: &core::sync::atomic::AtomicU32, what: &str) -> u32 {
    counter
        .fetch_update(
            core::sync::atomic::Ordering::Relaxed,
            core::sync::atomic::Ordering::Relaxed,
            |n| n.checked_add(1),
        )
        .unwrap_or_else(|_| panic!("{what}: counter exhausted"))
}

/// Declaration of a bound memory buffer: the static shape of a collapsed
/// lattice. The extents are part of the IR (like a `Const`) even though the
/// contents are bound later, at JIT-compile time. Static extents are what
/// allow the emitter to fold address arithmetic, drop provably in-bounds
/// clamps, and unroll reduction loops.
///
/// Layout is row-major with `stride == width`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BufferDecl {
    /// Which memory this names. Two declarations of the same buffer must carry
    /// the same identity — that is what lets a merge collapse them into one
    /// slot instead of binding the pointer twice.
    pub id: BufferIdentity,
    /// X extent (samples per row).
    pub width: u32,
    /// Y extent (number of rows).
    pub height: u32,
}

// ───────────────────────────────────────── Uniforms ───────────────────────────

/// Slot index into an [`Environment`](crate::expr::Environment)'s uniform
/// table. Copy, 2 bytes. Not an identity: two environments each call their own
/// first uniform slot 0.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct UniformId(pub u16);

/// Which uniform a declaration refers to, independent of any table.
///
/// A uniform is a scalar that is invariant across the lattice and supplied
/// at call time — the JIT tier's spelling of a builder's struct field. Its
/// identity is the *instance*: two instances of the same builder are two
/// factors of the kernel's parameter space, and one instance read from
/// twenty places is one factor. Neither a name nor a fragment-local index can
/// say that across a splice, so, exactly as for [`BufferIdentity`], identity
/// is provenance: minted once, copied into every declaration of the instance.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct UniformIdentity(u32);

impl UniformIdentity {
    /// Mint an identity distinct from every other in this process.
    ///
    /// # Panics
    ///
    /// Panics if the counter is exhausted, rather than wrapping onto a live
    /// identity and aliasing two unrelated uniforms.
    #[must_use]
    pub fn mint() -> Self {
        static NEXT: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
        Self(mint_identity(&NEXT, "UniformIdentity"))
    }
}

/// Declaration of a uniform: its identity plus the value the kernel holds
/// for it when nothing has been bound. The default is part of the IR so that
/// a bake without a block, and the scalar oracle, are total.
///
/// Compared and hashed by the default's *bit pattern*, so a declaration is a
/// plain key: `-0.0` and `0.0` are two defaults, and a NaN default is equal
/// to itself.
#[derive(Clone, Copy, Debug)]
pub struct UniformDecl {
    /// Which instance this names.
    pub id: UniformIdentity,
    /// The value bound when no block supplies one.
    pub default: f32,
}

impl PartialEq for UniformDecl {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.default.to_bits() == other.default.to_bits()
    }
}

impl Eq for UniformDecl {}

impl core::hash::Hash for UniformDecl {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.default.to_bits().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minted_identities_are_distinct() {
        assert_ne!(BufferIdentity::mint(), BufferIdentity::mint());
        assert_ne!(UniformIdentity::mint(), UniformIdentity::mint());
    }

    #[test]
    fn a_uniform_declaration_compares_by_bit_pattern() {
        let id = UniformIdentity::mint();
        let pos = UniformDecl {
            id,
            default: 0.0f32,
        };
        let neg = UniformDecl {
            id,
            default: -0.0f32,
        };
        assert_ne!(pos, neg, "-0.0 and 0.0 are two defaults");
        let nan = UniformDecl {
            id,
            default: f32::NAN,
        };
        assert_eq!(nan, nan, "a NaN default is equal to itself");
    }

    #[test]
    fn the_binder_space_sits_past_the_retired_axes() {
        assert_eq!(REDUCE_BINDER_BASE, 4);
        assert!(reduce_binders().all(|b| !RETIRED_COORD_AXES.contains(&b)));
        assert!(reduce_binders().all(|b| (b as usize) >= COORD_AXES));
        assert_eq!(reduce_binders().count(), REDUCE_BINDERS as usize);
    }
}
