#![allow(clippy::should_implement_trait)]
//! An arena-backed DAG whose consumers never see the arena.
//!
//! `no_std`; needs `alloc` for the two backing `Vec`s and the memo.
//!
//! Construction is index-based and private to `Builder`, which hands
//! back an `Id`.
//! Consumption is handle-based: everything downstream takes a
//! `Node<'a, T>`, a `Copy` pair of `(&Dag, u32)`, which exposes
//! only `.get()`, `Deref<Target = T>`, `.children()`, `.descendants()`.
//!
//! Acyclicity is an invariant of the constructor, not a runtime check:
//! a node may only point at nodes that already exist, so every edge
//! goes from a higher index to a lower one. Consequences worth knowing:
//!   - index order is a reverse topological order (children first);
//!   - no cycle detection, no `visited` set needed to terminate;
//!   - a node's edges are contiguous, so children iterate as a slice.
//!
//! The lifecycle is build-once: `Builder` accumulates, `finish()` freezes
//! it into a `Rooted`, and reading happens after. Composition is therefore
//! always "fresh graph, splice, done" — which is what every transform in
//! `passes.rs` was already doing to the append-only `ExprArena` this
//! replaced, one `&mut self` method at a time
//! (`docs/plans/2026-09-09-exprarena-on-dag.md`).

extern crate alloc;
#[cfg(test)]
extern crate std;

use alloc::vec;
use alloc::vec::Vec;

// `BTreeMap` (needs only `Ord`, always available via `alloc`) is the
// unconditional fallback so this crate keeps building under
// `--no-default-features` — no_std, no optional deps. `hash-memo` upgrades
// it to a `hashbrown::HashMap` (needs `Eq + Hash` instead) for callers who
// want the faster lookup and can afford the dependency.
#[cfg(feature = "hash-memo")]
type Memo<K, V> = hashbrown::HashMap<K, V>;
#[cfg(not(feature = "hash-memo"))]
type Memo<K, V> = alloc::collections::BTreeMap<K, V>;

use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::Deref;

/// Build-time handle. Opaque, and never needed by consumers.
///
/// The guarantee is over *construction* and *observation* both: there is no
/// public constructor and no public accessor, so the only way to hold one is
/// to have been given it by `push`/`intern`, the only thing it can be spent
/// on is the `Builder` that issued it, and `Debug` does not render the
/// underlying index — printing an `Id` learns nothing that could be turned
/// back into one or compared against another builder's.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct Id(u32);

impl fmt::Debug for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // No index: printing an `Id` must not hand back the thing the type
        // exists to withhold. `Dag::push`'s panic on a foreign/not-yet-built
        // child names the failure, not the index, for the same reason.
        f.write_str("Id")
    }
}

struct Slot<T> {
    value: T,
    edge_start: u32,
    edge_len: u32,
}

/// Which DAG a [`Scratch`] or [`SideTable`] was built for.
///
/// Minted, not read off the DAG's address. An address answers "where does
/// this live right now", which is not the question: a `Dag` that is moved —
/// boxed, or returned in a tuple beside its own scratch — is the same DAG at
/// a new address, and a fresh `Dag` allocated where a dropped one used to sit
/// is a different DAG at the same one. Both mistakes are silent under a
/// pointer comparison and impossible under a minted one.
///
/// Same discipline, for the same reason, as `BufferIdentity` and
/// `UniformIdentity` in `decl.rs`: identity is provenance.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct DagIdentity(u32);

impl DagIdentity {
    /// Mint an identity distinct from every other in this process.
    ///
    /// # Panics
    ///
    /// Panics if the counter is exhausted, rather than wrapping onto a live
    /// identity and letting two unrelated DAGs share a scratch.
    fn mint() -> Self {
        static NEXT: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
        // `fetch_update`, not `fetch_add` + assert: the add would wrap
        // *before* the assert fires, so a caught panic would leave the
        // counter back on a live identity. Declining to store leaves it
        // permanently exhausted instead. (`decl.rs`'s `mint_identity` has
        // the long version of this note.)
        Self(
            NEXT.fetch_update(
                core::sync::atomic::Ordering::Relaxed,
                core::sync::atomic::Ordering::Relaxed,
                |n| n.checked_add(1),
            )
            .unwrap_or_else(|_| panic!("DagIdentity: counter exhausted")),
        )
    }
}

pub struct Dag<T> {
    identity: DagIdentity,
    nodes: Vec<Slot<T>>,
    edges: Vec<u32>,
}

impl<T: Clone> Clone for Dag<T> {
    fn clone(&self) -> Self {
        Dag {
            // A clone is a different DAG, so it mints rather than copies:
            // the original's `Scratch` must not silently accept it. Today
            // the two agree on length and shape, so accepting would be
            // harmless — which is exactly the kind of "true until someone
            // edits it" reasoning the identity exists to stop relying on.
            identity: DagIdentity::mint(),
            nodes: self
                .nodes
                .iter()
                .map(|s| Slot {
                    value: s.value.clone(),
                    edge_start: s.edge_start,
                    edge_len: s.edge_len,
                })
                .collect(),
            edges: self.edges.clone(),
        }
    }
}

impl<T> Dag<T> {
    fn new() -> Self {
        Dag {
            identity: DagIdentity::mint(),
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Add a node whose children already exist. Panics on a foreign or
    /// not-yet-created child, which is the only way a cycle could appear.
    fn push(&mut self, value: T, children: &[Id]) -> Id {
        let edge_start = self.edges.len() as u32;
        let here = self.nodes.len() as u32;
        for c in children {
            assert!(c.0 < here, "child does not exist yet in this builder");
            self.edges.push(c.0);
        }
        self.nodes.push(Slot {
            value,
            edge_start,
            edge_len: children.len() as u32,
        });
        Id(here)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Every node, children strictly before parents.
    #[must_use]
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = Node<'_, T>> + '_ {
        (0..self.nodes.len() as u32).map(move |ix| Node { dag: self, ix })
    }

    /// Nodes nothing points at. O(V + E).
    pub fn roots(&self) -> impl Iterator<Item = Node<'_, T>> + '_ {
        let mut has_parent = vec![false; self.nodes.len()];
        for &e in &self.edges {
            has_parent[e as usize] = true;
        }
        self.iter().filter(move |n| !has_parent[n.ix as usize])
    }
}

/// A node, borrowed. This is the whole consumption surface.
pub struct Node<'a, T> {
    dag: &'a Dag<T>,
    ix: u32,
}

// Manual: derive would demand `T: Copy`/`T: Clone`.
impl<'a, T> Clone for Node<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, T> Copy for Node<'a, T> {}

impl<'a, T> Node<'a, T> {
    /// Borrow for the arena's lifetime, not the handle's.
    #[must_use]
    pub fn get(self) -> &'a T {
        &self.dag.nodes[self.ix as usize].value
    }

    /// The DAG this node belongs to.
    #[must_use]
    pub fn dag(self) -> &'a Dag<T> {
        self.dag
    }

    fn ix(self) -> u32 {
        self.ix
    }

    #[must_use]
    pub fn children(self) -> Children<'a, T> {
        let slot = &self.dag.nodes[self.ix as usize];
        let start = slot.edge_start as usize;
        let end = start + slot.edge_len as usize;
        Children {
            dag: self.dag,
            edges: self.dag.edges[start..end].iter(),
        }
    }

    #[must_use]
    pub fn child_count(self) -> usize {
        self.dag.nodes[self.ix as usize].edge_len as usize
    }

    #[must_use]
    pub fn is_leaf(self) -> bool {
        self.child_count() == 0
    }

    /// Self and everything reachable, each yielded once, parents before
    /// children. Allocates one bitmap per call.
    #[must_use]
    pub fn descendants(self) -> Descendants<'a, T> {
        Descendants {
            dag: self.dag,
            stack: vec![self.ix],
            seen: vec![false; self.dag.nodes.len()],
        }
    }

    /// Same traversal as `descendants`, borrowing its state instead of
    /// allocating it. The scratch must come from this DAG.
    pub fn descendants_in<'s>(self, scratch: &'s mut Scratch) -> DescendantsIn<'a, 's, T> {
        scratch.begin(self);
        DescendantsIn {
            dag: self.dag,
            scratch,
        }
    }

    /// The payload stored at this node.
    #[must_use]
    pub fn data(&self) -> &T {
        self
    }
}

impl<'a, T> Deref for Node<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.dag.nodes[self.ix as usize].value
    }
}

// Identity is (arena, index) — two handles into different arenas are
// never equal even if their values are.
impl<'a, T> PartialEq for Node<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        self.ix == other.ix && core::ptr::eq(self.dag, other.dag)
    }
}
impl<'a, T> Eq for Node<'a, T> {}
impl<'a, T> Hash for Node<'a, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ix.hash(state);
        (self.dag as *const Dag<T>).hash(state);
    }
}

impl<'a, T> PartialOrd for Node<'a, T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a, T> Ord for Node<'a, T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        (self.dag as *const Dag<T>)
            .cmp(&(other.dag as *const Dag<T>))
            .then_with(|| self.ix.cmp(&other.ix))
    }
}

impl<'a, T: fmt::Debug> fmt::Debug for Node<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("value", self.get())
            .field("children", &self.child_count())
            .finish()
    }
}

pub struct Children<'a, T> {
    dag: &'a Dag<T>,
    edges: core::slice::Iter<'a, u32>,
}

impl<'a, T> Iterator for Children<'a, T> {
    type Item = Node<'a, T>;
    fn next(&mut self) -> Option<Self::Item> {
        let dag = self.dag;
        self.edges.next().map(|&ix| Node { dag, ix })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.edges.size_hint()
    }
}
impl<'a, T> ExactSizeIterator for Children<'a, T> {}
impl<'a, T> DoubleEndedIterator for Children<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let dag = self.dag;
        self.edges.next_back().map(|&ix| Node { dag, ix })
    }
}

pub struct Descendants<'a, T> {
    dag: &'a Dag<T>,
    stack: Vec<u32>,
    seen: Vec<bool>,
}

impl<'a, T> Iterator for Descendants<'a, T> {
    type Item = Node<'a, T>;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(ix) = self.stack.pop() {
            if core::mem::replace(&mut self.seen[ix as usize], true) {
                continue;
            }
            let slot = &self.dag.nodes[ix as usize];
            let start = slot.edge_start as usize;
            let end = start + slot.edge_len as usize;
            self.stack
                .extend(self.dag.edges[start..end].iter().rev().copied());
            return Some(Node { dag: self.dag, ix });
        }
        None
    }
}

/// Hash-consing front end. Structurally identical terms collapse to one
/// node, so `Id` equality *is* structural equality and the DAG shares
/// maximally without the caller tracking what it has already built.
///
/// The bound lives here and dies at `finish()`: `Dag<T>` and `Node<'_, T>`
/// stay unbounded, so consumers inherit nothing from the interning
/// strategy. Which bound depends on which `Memo` backs it — `Ord + Clone`
/// for the always-available `BTreeMap` fallback, `Eq + Hash + Clone` when
/// `hash-memo` swaps in a `HashMap`.
#[cfg(not(feature = "hash-memo"))]
pub(crate) trait Key: Ord + Clone {}
#[cfg(not(feature = "hash-memo"))]
impl<T: Ord + Clone> Key for T {}

#[cfg(feature = "hash-memo")]
pub(crate) trait Key: Eq + Hash + Clone {}
#[cfg(feature = "hash-memo")]
impl<T: Eq + Hash + Clone> Key for T {}

pub(crate) struct Builder<T: Key> {
    dag: Dag<T>,
    // Keyed on raw indices rather than `Id`, so the memo's `Ord`/`Hash`
    // requirement lands on a `u32` instead of forcing those bounds onto the
    // public handle, which needs neither.
    memo: Memo<(T, Vec<u32>), u32>,
}

impl<T: Key> Default for Builder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Key> Builder<T> {
    #[must_use]
    pub(crate) fn new() -> Self {
        Builder {
            dag: Dag::new(),
            memo: Memo::new(),
        }
    }

    #[must_use]
    pub(crate) fn with_capacity(nodes: usize, edges: usize) -> Self {
        Builder {
            dag: Dag {
                identity: DagIdentity::mint(),
                nodes: Vec::with_capacity(nodes),
                edges: Vec::with_capacity(edges),
            },
            memo: Memo::new(),
        }
    }

    pub(crate) fn intern(&mut self, value: T, children: &[Id]) -> Id {
        let key = (value, children.iter().map(|c| c.0).collect::<Vec<_>>());
        if let Some(&ix) = self.memo.get(&key) {
            return Id(ix);
        }
        let id = self.dag.push(key.0.clone(), children);
        self.memo.insert(key, id.0);
        id
    }

    /// Uninterned insert, for nodes that must stay distinct despite
    /// comparing equal (fresh temporaries, debug markers).
    pub(crate) fn push_unique(&mut self, value: T, children: &[Id]) -> Id {
        self.dag.push(value, children)
    }

    /// Read back a node this builder has already created.
    ///
    /// Building is not always write-only: an [`Ir`](crate::term::Ir)
    /// implementation has to `project` what it just embedded, and a transform
    /// that rewrites a term it is midway through rebuilding has to look at it.
    /// The handle is the ordinary consumption handle, borrowed from the
    /// partial DAG — nothing here can observe an index or mutate a node.
    pub(crate) fn get(&self, id: Id) -> Node<'_, T> {
        Node {
            dag: &self.dag,
            ix: id.0,
        }
    }

    /// How many nodes exist so far.
    pub(crate) fn len(&self) -> usize {
        self.dag.len()
    }

    /// Spends the ids. Entry points are recorded as indices, so the
    /// caller's `Id`s die with the builder and consumers start from
    /// `Rooted::entries()` — a handle iterator.
    #[must_use]
    pub(crate) fn finish(self, entries: &[Id]) -> Rooted<T> {
        let n = self.dag.nodes.len() as u32;
        let entries = entries
            .iter()
            .map(|e| {
                assert!(e.0 < n, "entry point from a foreign builder");
                e.0
            })
            .collect();
        Rooted {
            dag: self.dag,
            entries,
        }
    }
}

/// A finished DAG plus its declared entry points. This is what crosses
/// the phase boundary; `Id` does not.
pub struct Rooted<T> {
    dag: Dag<T>,
    entries: Vec<u32>,
}

impl<T: Clone> Clone for Rooted<T> {
    fn clone(&self) -> Self {
        Rooted {
            dag: self.dag.clone(),
            entries: self.entries.clone(),
        }
    }
}

impl<T> Rooted<T> {
    /// The nodes the builder was told to keep.
    #[must_use]
    pub fn entries(&self) -> impl ExactSizeIterator<Item = Node<'_, T>> + '_ {
        self.entries
            .iter()
            .map(move |&ix| Node { dag: &self.dag, ix })
    }

    #[must_use]
    pub fn entry(&self) -> Node<'_, T> {
        assert_eq!(self.entries.len(), 1, "not a single-entry DAG");
        Node {
            dag: &self.dag,
            ix: self.entries[0],
        }
    }

    /// Access the i-th declared entry point.
    #[must_use]
    pub fn entry_at(&self, idx: usize) -> Node<'_, T> {
        Node {
            dag: &self.dag,
            ix: self.entries[idx],
        }
    }
}

impl<T> Deref for Rooted<T> {
    type Target = Dag<T>;
    fn deref(&self) -> &Dag<T> {
        &self.dag
    }
}

/// Reusable walk state: the visited marks plus the DFS stack. Created
/// once, cleared per walk, so a traversal in a loop allocates nothing
/// after the stack reaches its high-water mark.
///
/// The clear is an O(V) `fill`, which beats the O(V) zeroing `vec!` was
/// already doing and drops the allocator call. If you walk tiny
/// subgraphs of a large arena often enough for that memset to show up,
/// swap `seen` for `Vec<u32>` epoch stamps: O(1) reset, 4x the memory.
pub struct Scratch {
    seen: Vec<bool>,
    stack: Vec<u32>,
    owner: DagIdentity,
}

impl Scratch {
    fn begin<T>(&mut self, n: Node<'_, T>) {
        assert_eq!(
            n.dag.identity, self.owner,
            "scratch used with a node from another DAG"
        );
        self.seen.fill(false);
        self.stack.clear();
        self.stack.push(n.ix);
    }

    /// Nodes the scratch can mark. Equals the arena's node count.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.seen.len()
    }
}

pub struct DescendantsIn<'a, 's, T> {
    dag: &'a Dag<T>,
    scratch: &'s mut Scratch,
}

impl<'a, 's, T> Iterator for DescendantsIn<'a, 's, T> {
    type Item = Node<'a, T>;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(ix) = self.scratch.stack.pop() {
            if core::mem::replace(&mut self.scratch.seen[ix as usize], true) {
                continue;
            }
            let s = &self.dag.nodes[ix as usize];
            let (start, len) = (s.edge_start as usize, s.edge_len as usize);
            self.scratch
                .stack
                .extend(self.dag.edges[start..start + len].iter().rev().copied());
            return Some(Node { dag: self.dag, ix });
        }
        None
    }
}

pub struct SideTable<V> {
    vals: Vec<V>,
    owner: DagIdentity,
}

impl<T> Dag<T> {
    /// Allocate reusable traversal state for `descendants_in`.
    ///
    /// Bound to this DAG's identity, not its address, so the DAG may be
    /// moved afterwards — boxed, or returned in a tuple beside the scratch
    /// itself — without invalidating it.
    #[must_use]
    pub fn scratch(&self) -> Scratch {
        Scratch {
            seen: vec![false; self.nodes.len()],
            stack: Vec::new(),
            owner: self.identity,
        }
    }

    /// Dense per-node storage for an analysis. Bound to this DAG's identity
    /// on the same terms as [`Dag::scratch`].
    pub fn side_table<V: Clone>(&self, init: V) -> SideTable<V> {
        SideTable {
            vals: vec![init; self.nodes.len()],
            owner: self.identity,
        }
    }
}

impl<V> SideTable<V> {
    fn slot<T>(&self, n: Node<'_, T>) -> usize {
        assert_eq!(
            n.dag.identity, self.owner,
            "side table used with a node from another DAG"
        );
        n.ix() as usize
    }

    #[must_use]
    pub fn get<T>(&self, n: Node<'_, T>) -> &V {
        &self.vals[self.slot(n)]
    }

    pub fn set<T>(&mut self, n: Node<'_, T>, v: V) {
        let i = self.slot(n);
        self.vals[i] = v;
    }
}

impl<'a, T, V> core::ops::Index<Node<'a, T>> for SideTable<V> {
    type Output = V;
    fn index(&self, n: Node<'a, T>) -> &V {
        &self.vals[self.slot(n)]
    }
}

impl<'a, T, V> core::ops::IndexMut<Node<'a, T>> for SideTable<V> {
    fn index_mut(&mut self, n: Node<'a, T>) -> &mut V {
        let i = self.slot(n);
        &mut self.vals[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec as StdVec;

    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
    enum Op {
        Var(&'static str),
        Add,
        Mul,
    }

    fn depth<T>(n: Node<'_, T>) -> usize {
        1 + n.children().map(depth).max().unwrap_or(0)
    }

    fn build() -> Rooted<Op> {
        let mut b = Builder::new();
        let x = b.intern(Op::Var("x"), &[]);
        let y = b.intern(Op::Var("y"), &[]);
        let add = b.intern(Op::Add, &[x, y]);
        let root = b.intern(Op::Mul, &[add, x]);
        b.finish(&[root])
    }

    #[test]
    fn consumption_never_names_an_id() {
        let g = build();
        let r = g.entry();
        assert_eq!(depth(r), 3);
        assert_eq!(g.len(), 4);
        assert_eq!(r.descendants().count(), 4); // x shared, visited once
    }

    #[test]
    fn interning_is_structural_equality() {
        let mut b = Builder::new();
        let x = b.intern(Op::Var("x"), &[]);
        let y = b.intern(Op::Var("y"), &[]);
        assert_eq!(b.intern(Op::Add, &[x, y]), b.intern(Op::Add, &[x, y]));
        assert_ne!(b.intern(Op::Add, &[x, y]), b.intern(Op::Add, &[y, x]));
        assert_eq!(b.finish(&[]).len(), 4);
    }

    #[test]
    fn side_table_does_the_indexing() {
        let g = build();
        // Bottom-up: iter() is children-before-parents, so one pass suffices.
        let mut size = g.side_table(0usize);
        for n in g.iter() {
            size[n] = 1 + n.children().map(|c| size[c]).sum::<usize>();
        }
        assert_eq!(size[g.entry()], 5); // counts the shared x twice, by path
    }

    #[test]
    fn descendants_in_matches_and_reuses() {
        let g = build();
        let mut sc = g.scratch();
        assert_eq!(sc.capacity(), g.len());

        let owned: StdVec<_> = g.entry().descendants().map(|n| n.get().clone()).collect();
        for _ in 0..3 {
            let borrowed: StdVec<_> = g
                .entry()
                .descendants_in(&mut sc)
                .map(|n| n.get().clone())
                .collect();
            assert_eq!(borrowed, owned);
        }

        // Abandoning a walk part-way must not poison the next one.
        let _ = g.entry().descendants_in(&mut sc).next();
        let after: StdVec<_> = g
            .entry()
            .descendants_in(&mut sc)
            .map(|n| n.get().clone())
            .collect();
        assert_eq!(after, owned);
    }

    // `repeated_walks_do_not_allocate` (proving `descendants_in` amortizes to
    // zero allocations) lives in `tests/dag_scratch_allocation.rs`, not here:
    // it needs a process-wide `#[global_allocator]` to count, and `cargo
    // test` runs this module's tests concurrently with every other test in
    // the same `--lib` binary — unrelated tests allocating on other threads
    // pollute a shared counter. A separate integration-test binary is its
    // own process, so nothing else is allocating into the same count.

    #[test]
    #[should_panic(expected = "another DAG")]
    fn scratch_is_arena_checked() {
        let g = build();
        let h = build();
        let mut sc = g.scratch();
        let _ = h.entry().descendants_in(&mut sc).count();
    }

    #[test]
    #[should_panic(expected = "another DAG")]
    fn side_table_is_arena_checked() {
        let g = build();
        let h = build();
        let t = g.side_table(0usize);
        let _ = t[h.entry()];
    }

    #[test]
    fn moving_the_dag_does_not_invalidate_its_scratch_or_side_table() {
        // The reason both are keyed on a minted identity rather than on
        // `&self`'s address: a DAG can be moved after handing one out, and
        // it is still the same DAG. Under an address comparison this
        // panicked with "from another DAG".
        let g = build();
        let mut sc = g.scratch();
        let mut sizes = g.side_table(0usize);

        let g = Box::new(g); // moves the Dag; its address changes

        assert_eq!(g.entry().descendants_in(&mut sc).count(), 4);
        for n in g.iter() {
            sizes[n] = 1 + n.children().map(|c| sizes[c]).sum::<usize>();
        }
        assert_eq!(sizes[g.entry()], 5);
    }

    #[test]
    fn a_clone_is_a_different_dag() {
        // Structurally identical, so accepting the original's scratch would
        // be harmless *today* — which is why it is refused: that harmlessness
        // is a property of the current code, not of the type.
        let g = build();
        let mut sc = g.scratch();
        let twin = g.clone();
        assert_eq!(g.entry().descendants_in(&mut sc).count(), 4);
        assert_eq!(twin.len(), g.len());

        let mut twin_sc = twin.scratch();
        assert_eq!(twin.entry().descendants_in(&mut twin_sc).count(), 4);
    }

    #[test]
    #[should_panic(expected = "another DAG")]
    fn a_clone_does_not_inherit_the_originals_scratch() {
        let g = build();
        let mut sc = g.scratch();
        let twin = g.clone();
        let _ = twin.entry().descendants_in(&mut sc).count();
    }

    #[test]
    fn topological_by_construction() {
        let g = build();
        let mut done = g.side_table(false);
        for n in g.iter() {
            assert!(n.children().all(|c| done[c]));
            done[n] = true;
        }
    }
}
