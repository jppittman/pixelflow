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

use alloc::vec;
use alloc::vec::Vec;

use crate::arena::{BufferDecl, ExprArena, ExprId, ExprNode, UniformDecl};

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
    pub fn of(arena: &ExprArena, root: ExprId) -> Self {
        Self::of_canonical(&canonical(arena, root))
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
#[derive(Clone, PartialEq, Eq)]
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
pub fn canonical(arena: &ExprArena, root: ExprId) -> Canonical {
    let len = arena.nodes_raw().len();
    let mut reachable = vec![false; len];
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if core::mem::replace(&mut reachable[id.0 as usize], true) {
            continue;
        }
        stack.extend(arena.children(id));
    }

    // Dense remap in ascending id order.
    let mut dense: Vec<u32> = vec![u32::MAX; len];
    let mut next = 0u32;
    let mut key: Vec<u8> = Vec::with_capacity(len * 8);
    let mut buffers: Vec<BufferDecl> = Vec::new();
    let mut uniforms: Vec<UniformDecl> = Vec::new();

    let push_id = |key: &mut Vec<u8>, dense: &[u32], id: ExprId| {
        let d = dense[id.0 as usize];
        debug_assert_ne!(d, u32::MAX, "child densified before parent");
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

    for idx in 0..len {
        if !reachable[idx] {
            continue;
        }
        match arena.node(ExprId(idx as u32)) {
            ExprNode::Var(i) => {
                key.push(0);
                key.push(*i);
            }
            ExprNode::Const(v) => {
                key.push(1);
                key.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            ExprNode::Param(i) => {
                key.push(2);
                key.push(*i);
            }
            ExprNode::Unary(op, a) => {
                key.push(3);
                key.extend_from_slice(&op.marshal().to_bytes());
                push_id(&mut key, &dense, *a);
            }
            ExprNode::Binary(op, a, b) => {
                key.push(4);
                key.extend_from_slice(&op.marshal().to_bytes());
                push_id(&mut key, &dense, *a);
                push_id(&mut key, &dense, *b);
            }
            ExprNode::Ternary(op, a, b, c) => {
                key.push(5);
                key.extend_from_slice(&op.marshal().to_bytes());
                push_id(&mut key, &dense, *a);
                push_id(&mut key, &dense, *b);
                push_id(&mut key, &dense, *c);
            }
            ExprNode::Nary(op, start, n) => {
                key.push(6);
                key.extend_from_slice(&op.marshal().to_bytes());
                key.extend_from_slice(&n.to_le_bytes());
                let (s, l) = (*start as usize, *n as usize);
                for child in &arena.nary_children_raw()[s..s + l] {
                    push_id(&mut key, &dense, *child);
                }
            }
            // Slot by first occurrence, extents in the key: the code folds
            // its address arithmetic against them.
            ExprNode::Buffer(b) => {
                let decl = *arena.buffer_decl(*b);
                key.push(7);
                key.extend_from_slice(&dense_slot(&mut buffers, decl).to_le_bytes());
                key.extend_from_slice(&decl.width.to_le_bytes());
                key.extend_from_slice(&decl.height.to_le_bytes());
            }
            // Offset by first occurrence; the default is the block's
            // business, not the code's.
            ExprNode::Uniform(u) => {
                let decl = *arena.uniform_decl(*u);
                key.push(8);
                key.extend_from_slice(&dense_slot(&mut uniforms, decl).to_le_bytes());
            }
            // A leaf with an identity of its own, like `Buffer`: the key it
            // names is exactly the referent's canonical bytes digested, so
            // encoding the key is encoding the referent.
            ExprNode::Ref(key_of) => {
                key.push(9);
                key.extend_from_slice(&key_of.bits().to_le_bytes());
            }
            // The fold is metadata, so it is *in the tag bytes* rather than
            // encoded as three child nodes. Two folds over the same body
            // under different algebras, binders or ranges are different
            // kernels, and this is where that is said.
            ExprNode::Reduce { fold, body } => {
                key.push(10);
                key.extend_from_slice(&fold.to_bits().to_le_bytes());
                push_id(&mut key, &dense, *body);
            }
        }
        dense[idx] = next;
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
    use crate::kind::OpKind;

    /// `√(x² + y²)`, optionally preceded by unreachable construction garbage.
    fn circle(garbage: bool) -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        if garbage {
            let g = a.push_const(123.0);
            let _ = a.push_unary(OpKind::Sqrt, g);
        }
        let x = a.push_var(0);
        let y = a.push_var(1);
        let x2 = a.push_binary(OpKind::Mul, x, x);
        let y2 = a.push_binary(OpKind::Mul, y, y);
        let s = a.push_binary(OpKind::Add, x2, y2);
        let root = a.push_unary(OpKind::Sqrt, s);
        (a, root)
    }

    #[test]
    fn same_content_is_the_same_key() {
        let (a, ra) = circle(false);
        let (b, rb) = circle(false);
        assert_eq!(KernelKey::of(&a, ra), KernelKey::of(&b, rb));
        assert_eq!(canonical(&a, ra).key, canonical(&b, rb).key);
    }

    #[test]
    fn different_content_is_a_different_key() {
        let (a, ra) = circle(false);
        let mut b = ExprArena::new();
        let x = b.push_var(0);
        let y = b.push_var(1);
        let rb = b.push_binary(OpKind::Sub, x, y);
        assert_ne!(KernelKey::of(&a, ra), KernelKey::of(&b, rb));
    }

    /// Construction garbage and a different id ordering for the *same*
    /// reachable subgraph must not perturb the identity — otherwise two
    /// build histories of one kernel would be two kernels.
    #[test]
    fn the_key_ignores_node_ordering_and_garbage() {
        let (clean, rc) = circle(false);
        let (littered, rl) = circle(true);
        assert_ne!(
            clean.nodes_raw().len(),
            littered.nodes_raw().len(),
            "the littered arena must actually hold more nodes"
        );
        assert_ne!(rc, rl, "and its root must sit at a different id");
        assert_eq!(canonical(&clean, rc).key, canonical(&littered, rl).key);
        assert_eq!(KernelKey::of(&clean, rc), KernelKey::of(&littered, rl));
    }

    /// The shape bytes are *deliberately* blind to which memory a slot binds
    /// — that is what lets one compiled region serve a thousand atlases — so
    /// the key must not be. Two samplers over two different tables of equal
    /// extents are two kernels, and a store that conflated them would resolve
    /// a reference to the wrong memory and render the wrong picture.
    #[test]
    fn the_key_separates_two_buffers_of_equal_extents() {
        let read = |decl: BufferDecl| {
            let mut a = ExprArena::new();
            let slot = a.declare_buffer(decl);
            let x = a.push_var(0);
            let y = a.push_var(1);
            let root = a.push_gather(slot, x, y);
            (a, root)
        };
        let shape = |id| BufferDecl {
            id,
            width: 4,
            height: 3,
        };
        let (a, ra) = read(shape(crate::arena::BufferIdentity::mint()));
        let (b, rb) = read(shape(crate::arena::BufferIdentity::mint()));

        assert_eq!(
            canonical(&a, ra).key,
            canonical(&b, rb).key,
            "one shape, so one compiled region — this is the compile cache's              whole point and must not change"
        );
        assert_ne!(
            KernelKey::of(&a, ra),
            KernelKey::of(&b, rb),
            "but two memories, so two kernels"
        );
    }

    /// The same for uniforms: two instances of one builder are two arguments,
    /// and a reference to either must not resolve to the other.
    #[test]
    fn the_key_separates_two_uniform_instances() {
        let read = |decl: UniformDecl| {
            let mut a = ExprArena::new();
            let slot = a.declare_uniform(decl);
            let u = a.push_uniform(slot);
            let x = a.push_var(0);
            let root = a.push_binary(OpKind::Mul, x, u);
            (a, root)
        };
        let (a, ra) = read(crate::Uniform::new(0.25).decl());
        let (b, rb) = read(crate::Uniform::new(0.25).decl());
        assert_eq!(canonical(&a, ra).key, canonical(&b, rb).key);
        assert_ne!(KernelKey::of(&a, ra), KernelKey::of(&b, rb));
    }

    /// A `Ref` leaf is keyed by the identity it names, and two references to
    /// different kernels are different content.
    #[test]
    fn a_ref_leaf_is_keyed_by_what_it_names() {
        let (a, ra) = circle(false);
        let mut b = ExprArena::new();
        let x = b.push_var(0);
        let rb = b.push_unary(OpKind::Neg, x);

        let mut host = ExprArena::new();
        let ref_a = host.push_ref(KernelKey::of(&a, ra));
        let ref_b = host.push_ref(KernelKey::of(&b, rb));
        assert_ne!(
            canonical(&host, ref_a).key,
            canonical(&host, ref_b).key,
            "two references naming different kernels are different content"
        );

        let mut twin = ExprArena::new();
        let ref_a_again = twin.push_ref(KernelKey::of(&a, ra));
        assert_eq!(
            canonical(&host, ref_a).key,
            canonical(&twin, ref_a_again).key
        );
    }
}
