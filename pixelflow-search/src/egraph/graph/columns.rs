//! The store's per-id columns: what an id indexes, and who may index it.
//!
//! Every id [`EGraph::add`](super::EGraph::add) mints owns one entry in each
//! column, pushed together by [`Columns::push`] and by nothing else, so the
//! columns are equally long because the type has no other way to grow them.
//! The id is also the node's index in the arena: the node a memo miss adds is
//! pushed at the arena's length, and the id it gets is that index.
//!
//! The columns are two types. [`Columns`] holds what a frozen graph keeps
//! (`super::FrozenGraph`); [`SaturationColumns`] holds what only saturation
//! reads (`last_changed`, the kind index, `queued`). `push` grows both, and is
//! the only thing that grows either.
//!
//! An id is read three ways, and the columns keep the readings apart by the
//! type that indexes them:
//!
//! - **per node**: the id names a node (`nodes`, the arena), the slot it was
//!   minted into (`parent`), the member after it in its class's list (`next`)
//!   and whether it has left its class for good (the retired bit). Indexed by
//!   [`EClassId`], and by [`ENodeId`] for the arena, which is the same
//!   integer.
//! - **per representative**: the id is the answer of [`Columns::root`], the
//!   class whose block of the equivalence the list's `head` and `tail`, the
//!   facts and the kind mask describe. Indexed by [`Root`], which only `root`
//!   and `root_mut` return, so a fact is never read at an id nobody asked
//!   `root` about.
//! - **per worklist slot**: the id is where a [`Slot`] was pushed, which
//!   `queued` is keyed by, whatever that id has since been merged into.
//!   Indexed by [`Slot`].
//!
//! A class's members are a singly linked list through the arena: `head`
//! and `tail` per representative, `next` per node, the end a private
//! sentinel. Walking `next` from `head[c]` yields the members in the graph's
//! order and stops at `tail[c]`; both are the end exactly when `c` has no
//! members, and always for a merged-away class. `add` gives `[t]`, a union
//! splices the child's list after the parent's tail, and a rebuild item
//! detaches its class's list, keeps what survives in a list of its own and
//! splices that after the class it ends up in. The links, `live` and the
//! retired bit are private, and `push`, `absorb`, `detach`, `keep`,
//! `write_back` and `retire` are the only writes to them, so a class is
//! filled or emptied only here and the live set is restated at each of
//! those that can change it.

use alloc::vec::Vec;
use core::marker::PhantomData;
use core::ops::{Index, IndexMut};
use pixelflow_ir::kind::{OpKind, OpSet};
use pixelflow_ir::{Lit, Variance};

use crate::egraph::live::{ClassSet, ClassesByKind};
use crate::egraph::node::{EClassId, ENode};
use crate::egraph::provenance::ENodeId;

/// A class that was canonical when it was asked for: the answer of
/// [`Columns::root`] or [`Columns::root_mut`], and nothing else makes one.
///
/// It is a name for the class at that moment, and what the type enforces is
/// that a per-representative column is not indexed by an id nobody asked
/// `root` about. It does not follow the class through a later union, which
/// leaves a `Root` naming a slot that is no longer a representative. So a
/// `Root` is not held across a union of its class: ask again, unless the
/// holder needs the name the class had and says so (`union` reads the
/// merged-away side after linking it, `rebuild_budgeted` keeps its item's).
/// It indexes the per-representative columns of both [`Columns`] and
/// [`SaturationColumns`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Root(EClassId);

impl Root {
    /// The class's id.
    pub(super) fn id(self) -> EClassId {
        self.0
    }
}

/// A place on the rebuild worklist: the class a union pushed, as it was
/// named then. The class has usually been merged away by the time the slot
/// pops, so [`Columns::root_of`] asks what it is now, and `queued` is cleared
/// by the slot, which is the key a union's guard set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Slot(EClassId);

impl Slot {
    /// The slot a union pushes for `root`.
    pub(super) fn at(root: Root) -> Self {
        Self(root.0)
    }
}

/// One entry per id, indexed by the type `I` that names what the entry
/// describes. Its only growth is [`Columns::push`].
pub(crate) struct Column<I, T> {
    entries: Vec<T>,
    index: PhantomData<fn(I)>,
}

impl<I, T> Column<I, T> {
    fn push(&mut self, entry: T) {
        self.entries.push(entry);
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }
}

impl<I, T> Default for Column<I, T> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            index: PhantomData,
        }
    }
}

impl<I, T: Clone> Clone for Column<I, T> {
    fn clone(&self) -> Self {
        Self {
            entries: self.entries.clone(),
            index: PhantomData,
        }
    }
}

/// `column[key]` for a column indexed by `$column` and a key of type `$key`
/// whose entry is at `$at(key)`.
macro_rules! index_by {
    ($column:ty, $key:ty, $at:expr) => {
        impl<T> Index<$key> for Column<$column, T> {
            type Output = T;

            fn index(&self, key: $key) -> &T {
                &self.entries[$at(key)]
            }
        }
    };
}

/// `column[key] = ..`, which is only offered for a key that names a class
/// that is canonical or a slot, never for a raw id.
macro_rules! index_mut_by {
    ($column:ty, $key:ty, $at:expr) => {
        impl<T> IndexMut<$key> for Column<$column, T> {
            fn index_mut(&mut self, key: $key) -> &mut T {
                &mut self.entries[$at(key)]
            }
        }
    };
}

// A per-node column reads at any id, at a node id (the arena's own index) and,
// being per-node, at a `Root` too.
index_by!(EClassId, EClassId, |id: EClassId| id.index());
index_by!(EClassId, ENodeId, |tag: ENodeId| tag.0 as usize);
index_by!(EClassId, Root, |root: Root| root.0.index());
index_mut_by!(EClassId, Root, |root: Root| root.0.index());
// A per-representative column reads only at a representative.
index_by!(Root, Root, |root: Root| root.0.index());
index_mut_by!(Root, Root, |root: Root| root.0.index());
// A per-slot column reads only at a slot.
index_by!(Slot, Slot, |slot: Slot| slot.0.index());
index_mut_by!(Slot, Slot, |slot: Slot| slot.0.index());

/// What `add` seeds one id's entries with.
pub(super) struct Birth {
    /// The canonical node `add` pushes into the arena.
    pub(super) node: ENode,
    pub(super) const_fact: Option<Lit>,
    pub(super) variance: Variance,
    /// The node's shape, as `ENode::shape` names it: the one kind the
    /// new class starts with.
    pub(super) shape: Option<OpKind>,
    /// The mutation counter's value as of this `add`.
    pub(super) stamp: u64,
}

/// A place in a member list: a node id, or the end of the list. The end is
/// `u64::MAX`, which no node id reaches (ids range over `EClassId`, a `u32`),
/// so a link is 8 bytes where an `Option<ENodeId>` would be 16. Private to this
/// module: outside it a list is a [`Members`] walk and the end is `None`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Link(u64);

impl Link {
    const END: Self = Self(u64::MAX);

    fn to(tag: ENodeId) -> Self {
        debug_assert_ne!(tag.0, u64::MAX, "no node id is the end of a list");
        Self(tag.0)
    }

    fn get(self) -> Option<ENodeId> {
        (self != Self::END).then_some(ENodeId(self.0))
    }
}

/// A class's members in law 8's order: a walk down `next` from the class's
/// head, one dependent load per member.
#[derive(Clone)]
pub(super) struct Members<'a> {
    links: &'a Column<EClassId, Link>,
    at: Link,
}

impl Iterator for Members<'_> {
    type Item = ENodeId;

    fn next(&mut self) -> Option<ENodeId> {
        let tag = self.at.get()?;
        self.at = self.links[tag];
        Some(tag)
    }
}

/// A rebuild item's detached members, not yet walked: the chain `detach` took
/// off the class. No class lists them, so no union can write their links
/// before [`Columns::walk`] reads them.
pub(super) struct Detached(Link);

/// A rebuild item's kept members in walk order: a list through `next`, ended
/// after every [`Columns::keep`], that no class holds until
/// [`Columns::write_back`].
pub(super) struct Kept {
    head: Link,
    tail: Link,
}

impl Kept {
    /// No member kept yet.
    pub(super) fn new() -> Self {
        Self {
            head: Link::END,
            tail: Link::END,
        }
    }
}

/// The per-id columns and the union-find they share. See the module docs.
#[derive(Clone, Default)]
pub(crate) struct Columns {
    /// The node each id names, in id order: the arena. `add` pushes a node
    /// that is not a memo hit at its length, and the id it returns is that
    /// index. A stored operand names the class it named when the node was
    /// last canonicalized (see [`Self::set_node`]).
    nodes: Column<EClassId, ENode>,
    /// `parent[t]` is the next id toward `t`'s representative; a
    /// representative is its own parent.
    parent: Column<EClassId, EClassId>,
    /// Whether the node at each id has left its class for good: set by
    /// [`Self::retire`], never cleared.
    retired: Column<EClassId, bool>,
    /// How many ids are retired.
    retired_count: usize,
    /// Per node: the member after this one in its class's list, or the end.
    /// The end for a class's last member and for a retired node. While a
    /// rebuild item walks its detached members, the chain they still form
    /// (see [`Detached`]).
    next: Column<EClassId, Link>,
    /// Per representative: the first member of the class's list, or the end
    /// exactly when the class has no members (a merged-away class, or one a
    /// rebuild item has detached).
    head: Column<Root, Link>,
    /// Per representative: the last member, so a union's splice is O(1). The
    /// end exactly when `head` is.
    tail: Column<Root, Link>,
    /// The classes with at least one member, as a set: the representatives
    /// whose list is not empty. Restated by `push`, `absorb`, `detach` and
    /// `write_back`, the writes that can change whether a class has members
    /// (`keep` and `retire` cannot: a kept node is in no class until
    /// `write_back`), so it agrees with the lists by construction.
    live: ClassSet,
    /// The constant each class is known to equal, as f32 bits — maintained
    /// independently of the member list on purpose.
    /// `rebuild` detaches a class's members and only THEN performs congruence
    /// unions, so a guard that scanned the member list saw an empty class and
    /// waved contradictory merges through at exactly the moment congruence
    /// closure does its work. The fact must outlive the members.
    pub(super) const_fact: Column<Root, Option<Lit>>,
    /// Which binders each class may depend on — the per-class variance fact
    /// (docs/archive/plans/2026-09-23-an-integral-is-a-fold.md §1, §3), kept beside
    /// `const_fact` and out of the member list for the same reason: a guard
    /// read mid-`rebuild` must not see a detached class.
    ///
    /// **Law:** `var_fact[C] ⊇ var(⟦C⟧)` — the fact over-approximates the
    /// variance of the one function every member of `C` denotes, so a bit
    /// absent from the fact is a binder the class is provably constant along.
    ///
    /// - **Seeded** in `EGraph::add` by `ENode::variance`, the node's
    ///   transfer function, over its children's facts *as they stand then*.
    ///   Each child's fact over-approximates the child, and the transfer is
    ///   monotone, so the seed over-approximates the node.
    /// - **Merged** in `EGraph::union` by *intersection*. Both classes denote
    ///   the same function and both facts over-approximate it, so a bit either
    ///   one lacks is a bit the function lacks. A fact only ever shrinks from
    ///   its seed and never grows up from `∅` — the greatest fixpoint, not
    ///   the least — which is why a cycle cannot certify an invariance: a
    ///   class holding `x` and `x + 0` got each member's fact from facts that
    ///   already existed, never from an assumption about itself.
    /// - **Never repaired upward.** A union that shrinks a child's fact does
    ///   not revisit the parents seeded from the larger one. A stale parent
    ///   fact is still a superset, so a rule reading it may miss an
    ///   opportunity but can never fire wrongly; soundness needs no repair.
    pub(super) var_fact: Column<Root, Variance>,
}

/// The per-id columns only saturation reads: the stamps the scan's dirty check
/// compares, the kind index the scan's class filter reads, and the worklist
/// flag. Grown only by [`Columns::push`], with the columns it holds, so the two
/// types are equally long because neither has another way to grow; dropped
/// with the rest of saturation's state by `EGraph::freeze`.
#[derive(Clone, Default)]
pub(crate) struct SaturationColumns {
    /// The mutation counter's value as of the most recent time canonical
    /// class `c`'s own node list gained content — set at creation in `add`
    /// and bumped in `union` when `c` survives as the merge's parent.
    pub(super) last_changed: Column<Root, u64>,
    /// The kinds this class's members have had. Set in `add`, ORed in `union`
    /// and never cleared, so **`kind_mask[C] ⊇ {shape(n) : n a member of C}`**:
    /// a node that retires leaves its kind behind, and a rule's per-node
    /// `admits` check, which stays, rejects what the stale bit let through.
    /// Checked by `EGraph::kind_index_covers_members` once per full rebuild.
    pub(super) kind_mask: Column<Root, OpSet>,
    /// The transpose of `kind_mask`: per kind, the classes whose mask holds
    /// it. Written wherever the mask is, so the two agree by construction;
    /// this is the one a rule scan reads, to visit only classes that can
    /// root it. Keyed by raw id, not [`Root`]: the set type is shared with
    /// the live set and lives outside this module, and only this module
    /// writes it. A merged-away class's bit is left set and filtered by
    /// `live`, as the mask's stale bits are filtered by the per-node check.
    pub(super) by_kind: ClassesByKind,
    /// Whether the slot has an entry on the worklist awaiting
    /// `rebuild_budgeted`. Guards `union`'s push: several unions landing on
    /// the same parent in one round used to enqueue it once per union, and
    /// `rebuild_budgeted` paid a full canonicalize-and-memo-probe pass over
    /// that class's member list per redundant entry even though only the
    /// state after the *last* union in the round matters.
    pub(super) queued: Column<Slot, bool>,
}

/// The tag of the node at `id`: a node's id and the class it founds are one
/// integer.
pub(super) fn tag_of(id: EClassId) -> ENodeId {
    ENodeId(u64::from(id.0))
}

impl Columns {
    /// The number of ids minted: every column's length.
    pub(super) fn len(&self) -> usize {
        self.parent.len()
    }

    /// Mint the next id, with its entry in every column, `saturation`'s
    /// included. The one place a column of either type grows.
    pub(super) fn push(&mut self, saturation: &mut SaturationColumns, birth: Birth) -> EClassId {
        let id = EClassId::from_index(self.len());
        self.nodes.push(birth.node);
        self.parent.push(id);
        self.retired.push(false);
        self.next.push(Link::END);
        self.head.push(Link::to(tag_of(id)));
        self.tail.push(Link::to(tag_of(id)));
        self.const_fact.push(birth.const_fact);
        self.var_fact.push(birth.variance);
        saturation.last_changed.push(birth.stamp);
        let kinds = match birth.shape {
            Some(kind) => {
                saturation.by_kind.insert(kind, id);
                OpSet::EMPTY.with(kind)
            }
            None => OpSet::EMPTY,
        };
        saturation.kind_mask.push(kinds);
        saturation.queued.push(false);
        self.live.insert(id);
        id
    }

    /// The node `tag` names. Panics past the end: a caller holding a tag it
    /// did not read from a member list asks [`Self::class_of`] first.
    pub(super) fn node(&self, tag: ENodeId) -> &ENode {
        &self.nodes[tag]
    }

    /// Replace node `tag`'s stored form: rebuild's canonicalization, in place.
    /// The arena is append-only in length and not in content, because a stored
    /// operand is state that rules can read.
    pub(super) fn set_node(&mut self, tag: ENodeId, node: ENode) {
        self.nodes.entries[tag.0 as usize] = node;
    }

    /// The class id `tag` was minted as, or `None` for a tag past this
    /// graph's end.
    pub(super) fn class_of(&self, tag: ENodeId) -> Option<EClassId> {
        usize::try_from(tag.0)
            .ok()
            .filter(|&index| index < self.len())
            .map(EClassId::from_index)
    }

    /// The member ids of `class`, in the order the graph gives them.
    pub(super) fn members(&self, class: Root) -> Members<'_> {
        Members {
            links: &self.next,
            at: self.head[class],
        }
    }

    /// Whether the node at `tag` has left its class for good.
    pub(super) fn is_retired(&self, tag: ENodeId) -> bool {
        self.retired[tag]
    }

    /// How many nodes have left their class for good.
    pub(super) fn retired_count(&self) -> usize {
        self.retired_count
    }

    /// Take `tag` out of membership for good. Only a refused union during a
    /// rebuild item calls it.
    pub(super) fn retire(&mut self, tag: ENodeId) {
        let slot = &mut self.retired.entries[tag.0 as usize];
        debug_assert!(!*slot, "a node is retired once");
        *slot = true;
        // Its link was read by `walk` before it was retired; a retired node
        // ends no list and is followed by nothing.
        self.next.entries[tag.0 as usize] = Link::END;
        self.retired_count += 1;
    }

    /// The classes with at least one member, which this module restates at
    /// each write that can change whether a class has members.
    pub(super) fn live(&self) -> &ClassSet {
        &self.live
    }

    /// `union`'s member move: `parent`'s members, then `child`'s, by a splice.
    /// Both are representatives now; the caller links them afterwards.
    pub(super) fn absorb(&mut self, child: Root, parent: Root) {
        let (first, last) = (self.head[child], self.tail[child]);
        self.head[child] = Link::END;
        self.tail[child] = Link::END;
        self.splice(parent, first, last);
        self.live.remove(child.id());
        self.set_live(parent);
    }

    /// A rebuild item's start: take `item`'s members out as a chain, so a union
    /// during the item splices into the class and not into the chain walked.
    pub(super) fn detach(&mut self, item: Root) -> Detached {
        self.live.remove(item.id());
        let first = self.head[item];
        self.head[item] = Link::END;
        self.tail[item] = Link::END;
        Detached(first)
    }

    /// The next detached member, with the walk advanced past it first: its
    /// link is read here, before `keep` or `retire` rewrites it.
    pub(super) fn walk(&self, detached: &mut Detached) -> Option<ENodeId> {
        let tag = detached.0.get()?;
        detached.0 = self.next[tag];
        Some(tag)
    }

    /// Append a walked member to the item's kept list, which stays ended.
    pub(super) fn keep(&mut self, kept: &mut Kept, tag: ENodeId) {
        self.next.entries[tag.0 as usize] = Link::END;
        match kept.tail.get() {
            Some(last) => self.next.entries[last.0 as usize] = Link::to(tag),
            None => kept.head = Link::to(tag),
        }
        kept.tail = Link::to(tag);
    }

    /// A rebuild item's end: the kept list goes after the members of the class
    /// `item` is in now, resolved here so no caller can name a merged-away
    /// class.
    pub(super) fn write_back(&mut self, item: Root, kept: Kept) {
        let dest = self.root(item.id());
        self.splice(dest, kept.head, kept.tail);
        self.set_live(dest);
    }

    /// Put the list `first ..= last` after `class`'s last member, O(1). An
    /// empty list changes nothing; an empty class takes `first` as its head.
    fn splice(&mut self, class: Root, first: Link, last: Link) {
        if first == Link::END {
            return;
        }
        match self.tail[class].get() {
            Some(end) => self.next.entries[end.0 as usize] = first,
            None => self.head[class] = first,
        }
        self.tail[class] = last;
    }

    /// Restate `class`'s liveness from its list.
    fn set_live(&mut self, class: Root) {
        match self.head[class] == Link::END {
            true => self.live.remove(class.id()),
            false => self.live.insert(class.id()),
        }
    }

    /// The representative of `id`'s block.
    pub(super) fn root(&self, id: EClassId) -> Root {
        let mut current = id;
        while self.parent[current] != current {
            current = self.parent[current];
        }
        Root(current)
    }

    /// [`Self::root`], compressing the path it walks.
    pub(super) fn root_mut(&mut self, id: EClassId) -> Root {
        let mut current = id;
        let mut path = Vec::new();
        while self.parent[current] != current {
            path.push(current);
            current = self.parent[current];
        }
        for node in path {
            self.parent.entries[node.index()] = current;
        }
        Root(current)
    }

    /// The representative `slot`'s class has now.
    pub(super) fn root_of(&self, slot: Slot) -> Root {
        self.root(slot.0)
    }

    /// Make `child`'s block part of `parent`'s.
    pub(super) fn link(&mut self, child: Root, parent: Root) {
        self.parent.entries[child.0.index()] = parent.0;
    }
}

impl SaturationColumns {
    /// Fold `child`'s kinds into `parent`'s, for a `child` being merged into
    /// `parent`: its members become `parent`'s.
    pub(super) fn merge_kinds(&mut self, child: Root, parent: Root) {
        let absorbed = self.kind_mask[child];
        self.kind_mask[parent] = self.kind_mask[parent].union(absorbed);
        // Inserting a kind the parent already has changes nothing, and is
        // cheaper than asking: skipping those cost 19 M Ir on the 29 programs.
        for kind in absorbed.iter() {
            self.by_kind.insert(kind, parent.0);
        }
    }
}
