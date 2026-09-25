//! What it means for two kernels to be *the same kernel*.
//!
//! [`canonical`] walks the subgraph reachable from a root in post-order from
//! that root, hash-consing structurally equal subterms, and encodes each node
//! by its tag, its payload and its children's canonical ids — buffer and
//! uniform leaves by dense slot rather than by minted identity, so two
//! compositions of one shape canonicalize alike. Neither construction garbage
//! (the unreachable nodes an append-only arena accumulates) nor the order a
//! builder pushed the reachable ones in enters the walk, so build history does
//! not perturb the result.
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
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Canonical {
    /// The canonical serialization of the graph's shape.
    pub key: Vec<u8>,
    /// The buffer each dense slot binds, in slot order.
    pub buffers: Vec<BufferDecl>,
    /// The uniform each dense offset holds, in offset order.
    pub uniforms: Vec<UniformDecl>,
}

/// Canonical serialization of the subgraph reachable from `root`: a
/// post-order walk from the root, children first to last, in which
/// structurally equal subterms are hash-consed — a node's canonical id is a
/// function of its tag, its payload bits and its children's canonical ids,
/// and of nothing else. Buffer and uniform leaves are numbered by dense slot
/// by first occurrence in that same walk — never by identity, which is what
/// lets two compositions of one shape share code.
///
/// The arena's own ids do not enter the result. An id is where a builder
/// happened to push a node, and two constructions of one kernel push in two
/// orders: `X + Y` with its leaves pushed either way round, or a
/// [`Kernel::sum`](crate::Kernel::sum) whose head is copied and whose tail is
/// spliced, so that every leaf the pieces share is interned at the head's
/// position and sorts to the front of every later piece's ascending-id walk
/// but not the head's own — which is how N summed pieces keyed as `N−1` of one
/// form and `1` of another. Walking from the root instead makes the bytes a
/// function of the term the root *denotes* — its tree unfolding — and a
/// duplicated subterm and a shared one unfold alike.
///
/// **Why no two programs share a key.** Each emitted node names its children
/// by canonical ids smaller than its own, every tag has a fixed encoding
/// length (an n-ary node carries its count), and the root is the last node
/// emitted: it cannot equal an earlier one, because that one would be its own
/// descendant. So the bytes parse back into exactly one tree over (tag,
/// payload, dense slot), and two programs with different unfoldings have
/// different bytes. What is deliberately *not* in the bytes — which memory a
/// slot binds — is the two tables beside them.
#[must_use]
pub fn canonical(arena: &ExprArena, root: ExprId) -> Canonical {
    let mut walk = Walk::new(arena);
    walk.post_order(root);
    walk.finish()
}

/// The canonical id of an arena node the walk has not emitted yet.
// A canonical id is bounded by the arena's node count, so its width is
// `ExprId`'s; widening `ExprId` widens these and the key's child references.
const UNVISITED: u32 = u32::MAX;

/// What [`canonical`]'s post-order walk carries: the bytes so far, the
/// hash-cons table over them, and the two link tables the bytes number by.
///
/// The table is the classic one: open addressing over canonical ids, probed
/// by the digest of a node's encoding, never more than half full because it
/// is sized to the arena up front. It holds no copy of any encoding — a
/// node's bytes live once, in `key`, and a probe compares against them there
/// — so emitting a node allocates nothing. A general-purpose map keyed on an
/// owned copy of the bytes measured 7× the cost of the whole walk on a glyph
/// kernel, in the copy, the second hash of it, the growth rehashes and the
/// free; keyed on the digest alone with a chain beside it, still 4×.
struct Walk<'a> {
    arena: &'a ExprArena,
    /// Arena id → canonical id, [`UNVISITED`] until the node is emitted.
    canon: Vec<u32>,
    /// The canonical id whose encoding digests to this slot, or
    /// [`UNVISITED`]; a power of two long, at most half full.
    table: Vec<u32>,
    /// Canonical id → where its encoding starts in `key`; it ends where the
    /// next one starts, or where the key does.
    starts: Vec<usize>,
    key: Vec<u8>,
    /// One node's encoding, reused across nodes.
    scratch: Vec<u8>,
    buffers: Vec<BufferDecl>,
    uniforms: Vec<UniformDecl>,
}

/// The table holds twice the arena's nodes, so a probe sequence is short
/// however the digests fall — every canonical id is one arena node, so it can
/// never fill past half.
const TABLE_SLOTS_PER_NODE: usize = 2;

/// One step of the iterative post-order: reach a node, or emit one whose
/// children have all been emitted.
enum Step {
    Descend(ExprId),
    Emit(ExprId),
}

/// What a probe for one encoding finds: the canonical id already emitted
/// under it, or the empty slot it would take.
enum Probe {
    Emitted(u32),
    Empty(usize),
}

impl<'a> Walk<'a> {
    fn new(arena: &'a ExprArena) -> Self {
        let len = arena.len();
        Self {
            arena,
            canon: vec![UNVISITED; len],
            table: vec![UNVISITED; (len * TABLE_SLOTS_PER_NODE).next_power_of_two()],
            starts: Vec::with_capacity(len),
            key: Vec::with_capacity(len * 8),
            scratch: Vec::new(),
            buffers: Vec::new(),
            uniforms: Vec::new(),
        }
    }

    fn post_order(&mut self, root: ExprId) {
        let mut work = vec![Step::Descend(root)];
        while let Some(step) = work.pop() {
            match step {
                Step::Descend(id) => {
                    if self.canon[id.0 as usize] != UNVISITED {
                        continue;
                    }
                    work.push(Step::Emit(id));
                    // Reversed, so the stack pops the first child first.
                    work.extend(self.arena.children(id).rev().map(Step::Descend));
                }
                Step::Emit(id) => self.emit(id),
            }
        }
    }

    /// Emit `id`, whose children are all emitted: encode it, and give it the
    /// canonical id of an earlier node with the same encoding if there is
    /// one, or the next id and a place in the key if there is not.
    fn emit(&mut self, id: ExprId) {
        // The DAG has no cycles, so nothing above an `Emit` on the stack can
        // be a second `Descend` of the same node.
        debug_assert_eq!(
            self.canon[id.0 as usize], UNVISITED,
            "a node is emitted once"
        );
        self.encode(id);
        let canonical_id = match self.probe() {
            Probe::Emitted(seen) => seen,
            Probe::Empty(slot) => self.intern(slot),
        };
        self.canon[id.0 as usize] = canonical_id;
    }

    /// Linear probing from `scratch`'s digest: the canonical id already
    /// emitted with exactly these bytes, or the empty slot the probe stopped
    /// at. It stops, because the table is at most half full.
    fn probe(&self) -> Probe {
        let digest = {
            use core::hash::Hasher;
            let mut h = Fnv1a(FNV_OFFSET_BASIS);
            h.write(&self.scratch);
            h.finish()
        };
        let mask = self.table.len() - 1;
        let mut slot = digest as usize & mask;
        loop {
            match self.table[slot] {
                UNVISITED => return Probe::Empty(slot),
                seen if self.encoding(seen) == self.scratch.as_slice() => {
                    return Probe::Emitted(seen);
                }
                _ => slot = (slot + 1) & mask,
            }
        }
    }

    /// The bytes canonical id `c` was emitted as.
    fn encoding(&self, c: u32) -> &[u8] {
        let c = c as usize;
        let start = self.starts[c];
        let end = self.starts.get(c + 1).copied().unwrap_or(self.key.len());
        &self.key[start..end]
    }

    /// Give `scratch`'s encoding the next canonical id, its place in the
    /// key, and the table `slot` its probe ended at.
    fn intern(&mut self, slot: usize) -> u32 {
        let fresh = u32::try_from(self.starts.len())
            .expect("canonical ids are dense over the arena, which indexes by u32");
        self.starts.push(self.key.len());
        self.key.extend_from_slice(&self.scratch);
        self.table[slot] = fresh;
        fresh
    }

    /// The canonical bytes of the child `id`, which must be emitted already.
    fn push_child(&mut self, id: ExprId) {
        let c = self.canon[id.0 as usize];
        debug_assert_ne!(c, UNVISITED, "child emitted before parent");
        self.scratch.extend_from_slice(&c.to_le_bytes());
    }

    /// `id`'s encoding into `scratch`: its tag, its payload, its children's
    /// canonical ids.
    fn encode(&mut self, id: ExprId) {
        self.scratch.clear();
        match self.arena.node(id) {
            ExprNode::Var(i) => {
                self.scratch.push(0);
                self.scratch.push(i);
            }
            ExprNode::Const(v) => {
                self.scratch.push(1);
                self.scratch.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            ExprNode::Param(i) => {
                self.scratch.push(2);
                self.scratch.push(i);
            }
            ExprNode::Unary(op, a) => {
                self.scratch.push(3);
                self.scratch.extend_from_slice(&op.marshal().to_bytes());
                self.push_child(a);
            }
            ExprNode::Binary(op, a, b) => {
                self.scratch.push(4);
                self.scratch.extend_from_slice(&op.marshal().to_bytes());
                self.push_child(a);
                self.push_child(b);
            }
            ExprNode::Ternary(op, a, b, c) => {
                self.scratch.push(5);
                self.scratch.extend_from_slice(&op.marshal().to_bytes());
                self.push_child(a);
                self.push_child(b);
                self.push_child(c);
            }
            ExprNode::Nary(op, _) => {
                let children = self.arena.children(id);
                let n = u16::try_from(children.len())
                    .expect("push_nary already asserted children.len() <= u16::MAX");
                self.scratch.push(6);
                self.scratch.extend_from_slice(&op.marshal().to_bytes());
                self.scratch.extend_from_slice(&n.to_le_bytes());
                for child in children {
                    self.push_child(child);
                }
            }
            // Slot by first occurrence, extents in the key: the code folds
            // its address arithmetic against them.
            ExprNode::Buffer(b) => {
                let decl = *self.arena.buffer_decl(b);
                self.scratch.push(7);
                let slot = dense_slot(&mut self.buffers, decl);
                self.scratch.extend_from_slice(&slot.to_le_bytes());
                self.scratch.extend_from_slice(&decl.width.to_le_bytes());
                self.scratch.extend_from_slice(&decl.height.to_le_bytes());
            }
            // Offset by first occurrence; the default is the block's
            // business, not the code's.
            ExprNode::Uniform(u) => {
                let decl = *self.arena.uniform_decl(u);
                self.scratch.push(8);
                let slot = dense_slot(&mut self.uniforms, decl);
                self.scratch.extend_from_slice(&slot.to_le_bytes());
            }
            // A leaf with an identity of its own, like `Buffer`: the key it
            // names is exactly the referent's canonical bytes digested, so
            // encoding the key is encoding the referent.
            ExprNode::Ref(key_of) => {
                self.scratch.push(9);
                self.scratch.extend_from_slice(&key_of.bits().to_le_bytes());
            }
            // The fold is metadata, so it is *in the tag bytes* rather than
            // encoded as three child nodes. Two folds over the same body
            // under different algebras, binders or ranges are different
            // kernels, and this is where that is said.
            ExprNode::Reduce { fold, body } => {
                self.scratch.push(10);
                self.scratch
                    .extend_from_slice(&fold.to_bits().to_le_bytes());
                self.push_child(body);
            }
            // The mask is a real child, numbered like any other; `on` and
            // `off` are content-addressed names, keyed the same way `Ref`
            // keys its one — so a `Guard` over the same mask and the same
            // two arms canonicalizes identically, and a different arm on
            // either side is a different key.
            ExprNode::Guard { mask, on, off } => {
                self.scratch.push(11);
                self.push_child(mask);
                self.scratch.extend_from_slice(&on.bits().to_le_bytes());
                self.scratch.extend_from_slice(&off.bits().to_le_bytes());
            }
            // The value is a real child; the three binders are metadata in
            // the tag bytes, as a fold's is: a store of the same value
            // under different binders is a different program.
            ExprNode::Write {
                row,
                col,
                lane,
                value,
            } => {
                self.scratch.push(12);
                self.push_child(value);
                self.scratch
                    .extend_from_slice(&[row.slot(), col.slot(), lane.slot()]);
            }
        }
    }

    fn finish(self) -> Canonical {
        Canonical {
            key: self.key,
            buffers: self.buffers,
            uniforms: self.uniforms,
        }
    }
}

/// The dense slot of `decl` in `table`, appending it on first sight.
fn dense_slot<T: PartialEq + Copy>(table: &mut Vec<T>, decl: T) -> u16 {
    let slot = table.iter().position(|d| *d == decl).unwrap_or_else(|| {
        table.push(decl);
        table.len() - 1
    });
    u16::try_from(slot).expect("dense slot fits the table index width")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kind::OpKind;

    /// Pins `canonical`'s bytes for a root that reaches an `Nary` node — the
    /// one shape whose encoding could drift when its children stop coming
    /// from a raw slab offset (Stage A,
    /// docs/plans/2026-09-09-exprarena-on-dag.md) and its node stops naming
    /// one at all (Stage B). Written out in the walk's own order — post-order
    /// from the root, children first to last — so the arena's push order
    /// (`v0` first) is visibly *not* what the bytes follow: `v1` is the
    /// root's first child, so it is emitted first, and `v0` is reached
    /// through `inner` before the root's own third child names it again.
    #[test]
    fn nary_canonical_bytes_are_pinned() {
        let mut arena = ExprArena::new();
        let v0 = arena.push_var(0);
        let v1 = arena.push_var(1);
        let c = arena.push_const(2.0);
        // A second Nary node makes the first's slab position nonzero, which
        // is what would expose an off-by-one in the child slice.
        let inner = arena.push_nary(OpKind::Tuple, &[v0, c]);
        let root = arena.push_nary(OpKind::Tuple, &[v1, inner, v0]);

        #[rustfmt::skip]
        let expected: &[u8] = &[
            // v1 = Var(1): the root's first child -> canonical 0
            0, 1,
            // v0 = Var(0): inner's first child -> canonical 1
            0, 0,
            // c = Const(2.0) -> canonical 2
            1, 0, 0, 0, 0x40,
            // inner = Nary(Tuple, [v0, c]) -> [1, 2], canonical 3
            6, 37, 2, 0, 1, 0, 0, 0, 2, 0, 0, 0,
            // root = Nary(Tuple, [v1, inner, v0]) -> [0, 3, 1], canonical 4
            6, 37, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0,
        ];
        assert_eq!(canonical(&arena, root).key, expected);
    }

    /// `X + Y` with its two leaves pushed in either order is one program, and
    /// the key says so — the arena's ids are where a builder happened to put
    /// the nodes, not part of what the root denotes.
    #[test]
    fn x_plus_y_pushed_in_either_order_is_one_key() {
        let mut x_first = ExprArena::new();
        let x = x_first.push_var(0);
        let y = x_first.push_var(1);
        let root_x_first = x_first.push_binary(OpKind::Add, x, y);

        let mut y_first = ExprArena::new();
        let y = y_first.push_var(1);
        let x = y_first.push_var(0);
        let root_y_first = y_first.push_binary(OpKind::Add, x, y);

        assert_ne!(
            x_first.node(root_x_first),
            y_first.node(root_y_first),
            "the two roots really name their children by different ids"
        );
        assert_eq!(
            canonical(&x_first, root_x_first),
            canonical(&y_first, root_y_first)
        );
        assert_eq!(
            KernelKey::of(&x_first, root_x_first),
            KernelKey::of(&y_first, root_y_first)
        );
    }

    /// A duplicated subterm and a shared one unfold to the same tree, so they
    /// are one key. Construction interns by slot, so the only duplicate an
    /// arena can carry is a second slot naming the same uniform instance — a
    /// leaf interning cannot see through, and exactly what `relink` folds
    /// back to one slot by identity.
    #[test]
    fn a_duplicated_subterm_and_a_shared_one_are_one_key() {
        let decl = crate::Uniform::new(0.25).decl();

        let mut shared = ExprArena::new();
        let u = shared.declare_uniform(decl);
        let x = shared.push_var(0);
        let leaf = shared.push_uniform(u);
        let xu = shared.push_binary(OpKind::Mul, x, leaf);
        let root_shared = shared.push_binary(OpKind::Add, xu, xu);

        let mut duplicated = ExprArena::new();
        let u1 = duplicated.declare_uniform(decl);
        let u2 = duplicated.declare_uniform(decl);
        let x = duplicated.push_var(0);
        let leaf1 = duplicated.push_uniform(u1);
        let leaf2 = duplicated.push_uniform(u2);
        let xu1 = duplicated.push_binary(OpKind::Mul, x, leaf1);
        let xu2 = duplicated.push_binary(OpKind::Mul, x, leaf2);
        let root_duplicated = duplicated.push_binary(OpKind::Add, xu1, xu2);

        assert_eq!(
            duplicated.len(),
            shared.len() + 2,
            "the duplicated arena really holds a second leaf and a second product"
        );
        assert_eq!(
            canonical(&duplicated, root_duplicated),
            canonical(&shared, root_shared)
        );
        assert_eq!(
            KernelKey::of(&duplicated, root_duplicated),
            KernelKey::of(&shared, root_shared)
        );
    }

    /// One piece of a glyph in its uniform form: `σ·area(x < e)` over its
    /// own two instances.
    fn piece() -> crate::Kernel {
        use crate::{Kernel, Uniform};
        let (zero, one) = (Kernel::constant(0.0), Kernel::constant(1.0));
        let edge = Uniform::new(0.5).kernel();
        let sigma = Uniform::new(1.0).kernel();
        let left = Kernel::x().lt(&edge).select(&one, &zero);
        sigma.mul(&left.area())
    }

    /// Every interval fold reachable in `arena`, by id.
    fn interval_folds(arena: &ExprArena) -> Vec<ExprId> {
        arena
            .nodes()
            .filter_map(|(id, node)| match node {
                ExprNode::Reduce {
                    fold: crate::Fold::Interval(_),
                    ..
                } => Some(id),
                _ => None,
            })
            .collect()
    }

    /// N pieces summed by `Kernel::sum`: every instance's interval folds have
    /// the one canonical form the piece has on its own. This is the measured
    /// failure: `sum` copies its head and splices its tail, so every leaf the
    /// pieces share is interned at the head's position, which sorted it to
    /// the front of each spliced piece's ascending-id walk and not the
    /// head's — N summed pieces keyed as `N−1` of one form and `1` of another.
    #[test]
    fn every_instance_of_a_summed_piece_has_one_fold_key() {
        const PIECES: usize = 3;
        let pieces: Vec<crate::Kernel> = (0..PIECES).map(|_| piece()).collect();
        let (alone, _) = pieces[0].parts();
        let alone_folds: Vec<Vec<u8>> = interval_folds(alone)
            .into_iter()
            .map(|id| canonical(alone, id).key)
            .collect();
        assert_eq!(alone_folds.len(), 2, "`area` is two nested interval folds");
        assert_ne!(alone_folds[0], alone_folds[1], "the inner and the outer");

        let sum = crate::Kernel::sum(&pieces);
        let (arena, _) = sum.parts();
        assert!(
            arena.len() < PIECES * alone.len(),
            "the sum really shares leaves between its pieces"
        );
        let summed_folds: Vec<Vec<u8>> = interval_folds(arena)
            .into_iter()
            .map(|id| canonical(arena, id).key)
            .collect();
        assert_eq!(summed_folds.len(), PIECES * alone_folds.len());
        for fold in &alone_folds {
            assert_eq!(
                summed_folds.iter().filter(|k| *k == fold).count(),
                PIECES,
                "every instance's fold keys as the piece's own does"
            );
        }
    }

    /// Two genuinely different programs never share a key: operand order,
    /// how many arguments are read, a fold's range and a fold's binder each
    /// tell two programs apart on their own.
    #[test]
    fn two_different_programs_are_two_keys() {
        use crate::fold::{Binder, Fold, Monoid};

        let x_minus_y = {
            let mut a = ExprArena::new();
            let (x, y) = (a.push_var(0), a.push_var(1));
            let r = a.push_binary(OpKind::Sub, x, y);
            KernelKey::of(&a, r)
        };
        let y_minus_x = {
            let mut a = ExprArena::new();
            let (x, y) = (a.push_var(0), a.push_var(1));
            let r = a.push_binary(OpKind::Sub, y, x);
            KernelKey::of(&a, r)
        };
        assert_ne!(x_minus_y, y_minus_x, "X − Y is not Y − X");

        let one_argument_read_twice = {
            let mut a = ExprArena::new();
            let u = a.declare_uniform(crate::Uniform::new(0.0).decl());
            let leaf = a.push_uniform(u);
            let r = a.push_binary(OpKind::Add, leaf, leaf);
            canonical(&a, r).key
        };
        let two_arguments = {
            let mut a = ExprArena::new();
            let u = a.declare_uniform(crate::Uniform::new(0.0).decl());
            let v = a.declare_uniform(crate::Uniform::new(0.0).decl());
            let (lu, lv) = (a.push_uniform(u), a.push_uniform(v));
            let r = a.push_binary(OpKind::Add, lu, lv);
            canonical(&a, r).key
        };
        assert_ne!(
            one_argument_read_twice, two_arguments,
            "u + u reads one argument, u + v reads two"
        );

        let slot = |s: u8| Binder::from_slot(s).expect("a binder slot");
        let folded = |fold: Fold| {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let body = a.push_binary(OpKind::Mul, x, x);
            let r = a.push_reduce(fold, body);
            KernelKey::of(&a, r)
        };
        let over_three = folded(Fold::new(Monoid::SUM, slot(0), 0..3));
        let over_four = folded(Fold::new(Monoid::SUM, slot(0), 0..4));
        let under_another_binder = folded(Fold::new(Monoid::SUM, slot(1), 0..3));
        assert_ne!(over_three, over_four, "a fold's range is part of its key");
        assert_ne!(
            over_three, under_another_binder,
            "a fold's binder is part of its key"
        );
    }

    /// A store of one value under different binders is a different program,
    /// and the key says so from the tag bytes alone.
    #[test]
    fn a_write_under_different_binders_is_a_different_key() {
        use crate::fold::Binder;
        let slot = |s: u8| Binder::from_slot(s).expect("a binder slot");
        let (row, col, lane) = (slot(0), slot(1), slot(2));
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let one = a.push_write(row, col, lane, x);
        let other = a.push_write(col, row, lane, x);
        let same = a.push_write(row, col, lane, x);
        assert_ne!(canonical(&a, one).key, canonical(&a, other).key);
        assert_eq!(canonical(&a, one).key, canonical(&a, same).key);
    }

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
            clean.len(),
            littered.len(),
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

    /// A `Guard` over the same mask and the same two arm keys canonicalizes
    /// identically in two different arenas — the property that lets one
    /// `Guard` be recognized as the same kernel as another, exactly as two
    /// `Ref`s to the same key already are.
    #[test]
    fn a_guard_over_the_same_mask_and_arms_is_the_same_key() {
        let on = KernelKey::from_bits(0xAAAA);
        let off = KernelKey::from_bits(0xBBBB);

        let mut a = ExprArena::new();
        let (xa, za) = (a.push_var(0), a.push_const(0.0));
        let mask_a = a.push_binary(OpKind::Lt, xa, za);
        let guard_a = a.push_guard(mask_a, on, off);

        let mut b = ExprArena::new();
        let (xb, zb) = (b.push_var(0), b.push_const(0.0));
        let mask_b = b.push_binary(OpKind::Lt, xb, zb);
        let guard_b = b.push_guard(mask_b, on, off);

        assert_eq!(canonical(&a, guard_a).key, canonical(&b, guard_b).key);
        assert_eq!(KernelKey::of(&a, guard_a), KernelKey::of(&b, guard_b));
    }

    /// A different `on` arm, a different `off` arm, or a different mask each
    /// change the key on their own — none of the three is redundant with the
    /// other two.
    #[test]
    fn a_guard_key_depends_on_the_mask_and_on_both_arms_independently() {
        let mask = |a: &mut ExprArena| a.push_var(0);
        let on = KernelKey::from_bits(1);
        let off = KernelKey::from_bits(2);

        let mut base_arena = ExprArena::new();
        let base_mask = mask(&mut base_arena);
        let base_guard = base_arena.push_guard(base_mask, on, off);
        let base = KernelKey::of(&base_arena, base_guard);

        let mut diff_on_arena = ExprArena::new();
        let diff_on_mask = mask(&mut diff_on_arena);
        let diff_on_guard = diff_on_arena.push_guard(diff_on_mask, KernelKey::from_bits(99), off);
        let diff_on = KernelKey::of(&diff_on_arena, diff_on_guard);
        assert_ne!(base, diff_on, "a different `on` arm must change the key");

        let mut diff_off_arena = ExprArena::new();
        let diff_off_mask = mask(&mut diff_off_arena);
        let diff_off_guard = diff_off_arena.push_guard(diff_off_mask, on, KernelKey::from_bits(99));
        let diff_off = KernelKey::of(&diff_off_arena, diff_off_guard);
        assert_ne!(base, diff_off, "a different `off` arm must change the key");

        let mut diff_mask_arena = ExprArena::new();
        let diff_mask = diff_mask_arena.push_var(1); // Y instead of X
        let diff_mask_guard = diff_mask_arena.push_guard(diff_mask, on, off);
        let diff_mask_key = KernelKey::of(&diff_mask_arena, diff_mask_guard);
        assert_ne!(base, diff_mask_key, "a different mask must change the key");

        // And swapping the two arms is not the same kernel either — `on` and
        // `off` name different branches of the same test.
        let mut swapped_arena = ExprArena::new();
        let swapped_mask = mask(&mut swapped_arena);
        let swapped_guard = swapped_arena.push_guard(swapped_mask, off, on);
        let swapped = KernelKey::of(&swapped_arena, swapped_guard);
        assert_ne!(base, swapped, "on and off are not interchangeable");
    }
}
