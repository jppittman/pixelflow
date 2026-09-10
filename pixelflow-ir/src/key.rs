//! What it means for two kernels to be *the same kernel*.
//!
//! [`canonical`] walks the subgraph reachable from a root in ascending id
//! order, remaps child references densely, and encodes each node — buffer and
//! uniform leaves by dense slot rather than by minted identity, so two
//! compositions of one shape canonicalize alike. Construction garbage (the
//! unreachable nodes an append-only arena accumulates) never enters the walk,
//! so build history does not perturb the result.
//!
//! That walk IS what says two kernels are the same kernel. It lived in
//! `pixelflow-codegen`'s `jit_cache` as the compile cache's key; it depends on
//! nothing but [`ExprArena`], and a kernel's identity is not codegen's private
//! business — a reference names a kernel by exactly this
//! (docs/plans/2026-09-09-composition-is-linking.md §2), so it belongs at the
//! bottom of the dependency graph where every crate can see it.
//!
//! The identity is the whole [`Canonical`], not its `key` field. The bytes
//! answer *what code to emit*, which is deliberately blind to which memory a
//! slot binds; the two tables beside them answer *which memory*, and a
//! reference has to carry both. The compile cache keys on the bytes alone
//! because it wants exactly that blindness — one region, many links — and
//! that difference is the one thing easy to get wrong here.
//!
//! [`KernelKey`] is the fixed-size digest of the whole form, which is what
//! lets an identity be a leaf in an arena. It is **not** a substitute for it:
//! a digest can collide, so anything keyed on one keeps the full form and
//! compares it (see [`KernelStore`](crate::store::KernelStore)).

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::arena::{BufferDecl, UniformDecl};
use crate::dag::Node;
use crate::expr::{Environment, ExprData};

/// The identity of a kernel: a 64-bit digest of its whole [`Canonical`] form
/// — the shape bytes **and** the link.
///
/// Both halves, because the shape bytes alone are deliberately blind to
/// *which* memory a slot binds: they number buffers and uniforms by dense
/// slot rather than by minted identity, which is exactly what lets a thousand
/// circles reading a thousand different atlases share one compiled region.
/// The compile cache can live with that because it hands the link back beside
/// the code; a reference store cannot, because `resolve` must return the very
/// kernel that was interned. Two samplers over two different 4×3 tables have
/// identical shape bytes and are not the same kernel.
///
/// **Why 64 bits and not 128.** An identity has to be nameable *inside* an
/// arena — `ExprNode::Ref(KernelKey)` — and `ExprNode` is held to 16 bytes by
/// a static assertion in [`arena`](crate::arena), because every node of every
/// kernel pays for that width. A 128-bit payload plus a discriminant is 24,
/// which is a 50% memory increase on every arena in the process to buy
/// collision resistance that is not needed: the store keyed on this compares
/// the *full* canonical bytes on every lookup and panics on a mismatch, so a
/// collision is a loud programming-error-class event, never a kernel silently
/// resolving to the wrong body.
///
/// The digest is computed here rather than through `core::hash` so that it is
/// a fixed function of the bytes: `DefaultHasher`'s algorithm is explicitly
/// not stable across Rust releases, and an identity that changes when the
/// toolchain does is not an identity.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct KernelKey(u64);

/// FNV-1a's 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a's 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
/// SplitMix64's finalizer constants: FNV-1a alone avalanches poorly in its
/// low bits, and the store indexes a hash map by this value.
const MIX_ODD_A: u64 = 0xbf58_476d_1ce4_e5b9;
const MIX_ODD_B: u64 = 0x94d0_49bb_1331_11eb;
const MIX_SHIFT_A: u32 = 30;
const MIX_SHIFT_B: u32 = 27;
const MIX_SHIFT_C: u32 = 31;

/// FNV-1a as a [`core::hash::Hasher`], so the link tables can be folded into
/// the digest through their derived [`Hash`](core::hash::Hash) — the
/// identities they carry have no byte accessor, and adding one to widen a
/// private field is a worse trade than using the trait that already exists.
struct Fnv1a(u64);

impl core::hash::Hasher for Fnv1a {
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= u64::from(*b);
            self.0 = self.0.wrapping_mul(FNV_PRIME);
        }
    }

    fn finish(&self) -> u64 {
        // FNV-1a avalanches poorly in its low bits and the store indexes a
        // hash map by this, so finish with SplitMix64's mixer.
        let mut h = self.0;
        h ^= h >> MIX_SHIFT_A;
        h = h.wrapping_mul(MIX_ODD_A);
        h ^= h >> MIX_SHIFT_B;
        h = h.wrapping_mul(MIX_ODD_B);
        h ^= h >> MIX_SHIFT_C;
        h
    }
}

impl KernelKey {
    /// The key of the kernel rooted at `root` in `arena` — its whole
    /// canonical form, shape and link alike.
    #[must_use]
    pub fn of(root: Node<'_, ExprData>, env: &Environment) -> Self {
        Self::of_canonical(&canonical(root, env))
    }

    /// The key of an already-computed canonical form.
    #[must_use]
    pub fn of_canonical(form: &Canonical) -> Self {
        use core::hash::{Hash, Hasher};
        let mut h = Fnv1a(FNV_OFFSET_BASIS);
        h.write(&form.key);
        form.buffers.hash(&mut h);
        form.uniforms.hash(&mut h);
        Self(h.finish())
    }

    /// This key's bits, for a store that must index by them.
    #[must_use]
    pub fn bits(self) -> u64 {
        self.0
    }

    /// A key with these exact bits — the collision test's only way to force
    /// two different canonical forms onto one key.
    #[cfg(test)]
    pub(crate) const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }
}

/// The canonical form of a reachable subgraph: its shape bytes, and the
/// buffer and uniform declarations in the dense order those bytes number them
/// by.
///
/// The two tables come out of the same walk that produces the bytes because
/// they must agree with it: the code is compiled against the dense slots, and
/// the tables say which identity each slot binds. All three together are a
/// kernel's identity — [`KernelKey`] digests all three, and
/// `pixelflow-codegen`'s compile cache keys on `key` alone precisely because
/// it wants the *opposite* of an identity there (one compiled region per
/// shape, many links).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Canonical {
    /// The canonical serialization of the graph's shape.
    pub key: Vec<u8>,
    /// The buffer each dense slot binds, in slot order.
    pub buffers: Vec<BufferDecl>,
    /// The uniform each dense offset holds, in offset order.
    pub uniforms: Vec<UniformDecl>,
}

/// Canonical serialization of the subgraph reachable from `root`: nodes in
/// ascending original id order (the arena is append-only, so children always
/// precede parents), child references remapped to dense indices, and buffer
/// and uniform leaves remapped to dense slots by first occurrence in that
/// same order — never by identity, which is what lets two compositions of
/// one shape share code.
#[must_use]
pub fn canonical(root: Node<'_, ExprData>, env: &Environment) -> Canonical {
    let reachable: BTreeSet<Node<'_, ExprData>> = root.descendants().collect();
    // Canonical identity follows expression first occurrence, not builder
    // insertion order.  Reversing the root-first traversal yields the same
    // children-before-parent order required by the dense child references,
    // while making two equivalent graphs with different declaration setup
    // serialize identically.
    let mut ordered: Vec<Node<'_, ExprData>> = root.descendants().collect();
    ordered.reverse();
    let mut dense: BTreeMap<Node<'_, ExprData>, u32> = BTreeMap::new();
    let mut next = 0u32;
    let mut key: Vec<u8> = Vec::with_capacity(reachable.len() * 8);
    let mut buffers: Vec<BufferDecl> = Vec::new();
    let mut uniforms: Vec<UniformDecl> = Vec::new();

    let push_id = |key: &mut Vec<u8>, dense: &BTreeMap<Node<'_, ExprData>, u32>, id| {
        let d = *dense.get(&id).expect("child densified before parent");
        key.extend_from_slice(&d.to_le_bytes());
    };
    /// The dense slot of `decl` in `table`, appending it on first sight.
    fn dense_slot<T: PartialEq + Copy>(table: &mut Vec<T>, decl: T) -> u16 {
        let slot = table.iter().position(|d| *d == decl).unwrap_or_else(|| {
            table.push(decl);
            table.len() - 1
        });
        u16::try_from(slot).expect("dense slot fits the table index width")
    }

    for node in ordered.into_iter().filter(|node| reachable.contains(node)) {
        match *node {
            ExprData::Var(i) => {
                key.push(0);
                key.push(i);
            }
            ExprData::Const(v) => {
                key.push(1);
                key.extend_from_slice(&v.to_le_bytes());
            }
            ExprData::Param(i) => {
                key.push(2);
                key.push(i);
            }
            ExprData::Op(op) if node.child_count() == 1 => {
                key.push(3);
                key.extend_from_slice(&op.marshal().to_bytes());
                push_id(&mut key, &dense, node.children().next().unwrap());
            }
            ExprData::Op(op) if node.child_count() == 2 => {
                key.push(4);
                key.extend_from_slice(&op.marshal().to_bytes());
                for child in node.children() {
                    push_id(&mut key, &dense, child);
                }
            }
            ExprData::Op(op) if node.child_count() == 3 => {
                key.push(5);
                key.extend_from_slice(&op.marshal().to_bytes());
                for child in node.children() {
                    push_id(&mut key, &dense, child);
                }
            }
            ExprData::Op(op) => {
                key.push(6);
                key.extend_from_slice(&op.marshal().to_bytes());
                key.extend_from_slice(&(node.child_count() as u16).to_le_bytes());
                for child in node.children() {
                    push_id(&mut key, &dense, child);
                }
            }
            // Slot by first occurrence, extents in the key: the code folds
            // its address arithmetic against them.
            ExprData::Buffer(b) => {
                let decl = *env
                    .buffers
                    .get(b.0 as usize)
                    .expect("buffer slot in environment");
                key.push(7);
                key.extend_from_slice(&dense_slot(&mut buffers, decl).to_le_bytes());
                key.extend_from_slice(&decl.width.to_le_bytes());
                key.extend_from_slice(&decl.height.to_le_bytes());
            }
            // Offset by first occurrence; the default is the block's
            // business, not the code's.
            ExprData::Uniform(u) => {
                let decl = *env
                    .uniforms
                    .get(u.0 as usize)
                    .expect("uniform slot in environment");
                key.push(8);
                key.extend_from_slice(&dense_slot(&mut uniforms, decl).to_le_bytes());
            }
            // A leaf with an identity of its own, like `Buffer`: the key it
            // names is exactly the referent's canonical bytes digested, so
            // encoding the key is encoding the referent.
            ExprData::Ref(key_of) => {
                key.push(9);
                key.extend_from_slice(&key_of.bits().to_le_bytes());
            }
            // The fold is metadata, so it is *in the tag bytes* rather than
            // encoded as three child nodes. Two folds over the same body
            // under different algebras, binders or ranges are different
            // kernels, and this is where that is said.
            ExprData::Reduce(fold) => {
                key.push(10);
                key.extend_from_slice(&fold.to_bits().to_le_bytes());
                push_id(
                    &mut key,
                    &dense,
                    node.children().next().expect("reduce body"),
                );
            }
        }
        dense.insert(node, next);
        next += 1;
    }

    Canonical {
        key,
        buffers,
        uniforms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExprBuilder, ExprGraph, OpKind, Uniform};

    fn key(graph: &ExprGraph) -> KernelKey {
        KernelKey::of(graph.root(), graph.environment())
    }

    fn circle(garbage: bool) -> ExprGraph {
        let mut b = ExprBuilder::new();
        if garbage {
            let g = b.constant(123.0);
            let _ = b.unary(OpKind::Sqrt, g);
        }
        let x = b.var(0);
        let y = b.var(1);
        let x2 = b.binary(OpKind::Mul, x, x);
        let y2 = b.binary(OpKind::Mul, y, y);
        let s = b.binary(OpKind::Add, x2, y2);
        let root = b.unary(OpKind::Sqrt, s);
        b.finish_one(root)
    }

    #[test]
    fn same_content_is_the_same_key() {
        let a = circle(false);
        let b = circle(false);
        assert_eq!(key(&a), key(&b));
        assert_eq!(
            canonical(a.root(), a.environment()).key,
            canonical(b.root(), b.environment()).key
        );
    }

    #[test]
    fn different_content_is_a_different_key() {
        let a = circle(false);
        let mut b = ExprBuilder::new();
        let x = b.var(0);
        let y = b.var(1);
        let rb = b.binary(OpKind::Sub, x, y);
        let b = b.finish_one(rb);
        assert_ne!(key(&a), key(&b));
    }

    #[test]
    fn the_key_ignores_construction_garbage() {
        let clean = circle(false);
        let littered = circle(true);
        assert_eq!(key(&clean), key(&littered));
    }

    #[test]
    fn the_key_separates_two_buffers_of_equal_extents() {
        let read = |decl: BufferDecl| {
            let mut b = ExprBuilder::new();
            let buffer = b.buffer(decl);
            let x = b.var(0);
            let y = b.var(1);
            let root = b.ternary(OpKind::Gather, buffer, x, y);
            b.finish_one(root)
        };
        let shape = |id| BufferDecl {
            id,
            width: 4,
            height: 3,
        };
        let a = read(shape(crate::arena::BufferIdentity::mint()));
        let b = read(shape(crate::arena::BufferIdentity::mint()));
        assert_eq!(
            canonical(a.root(), a.environment()).key,
            canonical(b.root(), b.environment()).key
        );
        assert_ne!(key(&a), key(&b));
    }

    #[test]
    fn the_key_separates_two_uniform_instances() {
        let read = |decl| {
            let mut b = ExprBuilder::new();
            let x = b.var(0);
            let uniform = b.uniform(decl);
            let root = b.binary(OpKind::Mul, x, uniform);
            b.finish_one(root)
        };
        let a = read(Uniform::new(0.25).decl());
        let b = read(Uniform::new(0.25).decl());
        assert_eq!(
            canonical(a.root(), a.environment()).key,
            canonical(b.root(), b.environment()).key
        );
        assert_ne!(key(&a), key(&b));
    }

    #[test]
    fn a_ref_leaf_is_keyed_by_what_it_names() {
        let a = circle(false);
        let mut b = ExprBuilder::new();
        let x = b.var(0);
        let rb = b.unary(OpKind::Neg, x);
        let b = b.finish_one(rb);
        let mut host = ExprBuilder::new();
        let ra = host.reference(key(&a));
        let rb = host.reference(key(&b));
        let host = host.finish(&[ra, rb]);
        assert_ne!(
            canonical(host.rooted().entry_at(0), host.environment()).key,
            canonical(host.rooted().entry_at(1), host.environment()).key
        );
    }
}
