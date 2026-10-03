//! A tree grown a leaf at a time, with ancestor queries in O(log depth).
//!
//! Regions of a scope nest as `If`s do, and nest as deeply as a program
//! writes them: an `else if` ladder a thousand rungs long is a tree a thousand
//! deep. A plain climb to the common ancestor then costs a thousand steps per
//! read, and the scope costs the square of its length. Each node therefore
//! carries one skip pointer, placed by Myers' skew-binary rule so that a
//! climb to any depth takes O(log depth) hops — one word per node, where a
//! table of every power-of-two ancestor would cost a word per level.
//!
//! Nodes are numbered in the order they are added, so a parent's number is
//! always smaller than its children's, and the root is node 0.

use alloc::vec::Vec;
#[cfg(test)]
use core::cell::Cell;

/// A tree of nodes `0..len()`, rooted at 0.
pub(crate) struct Tree {
    parent: Vec<usize>,
    depth: Vec<usize>,
    /// A proper ancestor, chosen so climbs by jumps stay logarithmic.
    jump: Vec<usize>,
    /// Hops taken by ancestor queries: a count and not a clock, so a test
    /// can pin how a pass grows and fail the same way on every host.
    #[cfg(test)]
    hops: Cell<usize>,
}

impl Tree {
    /// A tree of the root alone.
    pub(crate) fn rooted() -> Self {
        Self {
            parent: alloc::vec![0],
            depth: alloc::vec![0],
            jump: alloc::vec![0],
            #[cfg(test)]
            hops: Cell::new(0),
        }
    }

    /// Add a leaf under `parent` and return its number.
    pub(crate) fn grow(&mut self, parent: usize) -> usize {
        let node = self.parent.len();
        let near = self.jump[parent];
        let far = self.jump[near];
        // Skew-binary rule: jump from the parent's jump target's own target
        // when the two hops are the same length, else just to the parent.
        let jump = if self.depth[parent] - self.depth[near] == self.depth[near] - self.depth[far] {
            far
        } else {
            parent
        };
        self.parent.push(parent);
        self.depth.push(self.depth[parent] + 1);
        self.jump.push(jump);
        node
    }

    /// How many nodes there are.
    pub(crate) fn len(&self) -> usize {
        self.parent.len()
    }

    /// `node`'s parent; the root's is itself.
    pub(crate) fn parent(&self, node: usize) -> usize {
        self.parent[node]
    }

    /// `node`'s distance from the root.
    pub(crate) fn depth(&self, node: usize) -> usize {
        self.depth[node]
    }

    /// `node`'s ancestor at `depth`, which must not be deeper than `node`.
    pub(crate) fn ancestor_at(&self, mut node: usize, depth: usize) -> usize {
        while self.depth[node] > depth {
            self.hop();
            node = if self.depth[self.jump[node]] >= depth {
                self.jump[node]
            } else {
                self.parent[node]
            };
        }
        node
    }

    /// The deepest node that is an ancestor of both `a` and `b` (or either).
    pub(crate) fn common_ancestor(&self, a: usize, b: usize) -> usize {
        let (mut a, mut b) = match self.depth[a].cmp(&self.depth[b]) {
            core::cmp::Ordering::Greater => (self.ancestor_at(a, self.depth[b]), b),
            core::cmp::Ordering::Less => (a, self.ancestor_at(b, self.depth[a])),
            core::cmp::Ordering::Equal => (a, b),
        };
        while a != b {
            self.hop();
            (a, b) = if self.jump[a] != self.jump[b] {
                (self.jump[a], self.jump[b])
            } else {
                (self.parent[a], self.parent[b])
            };
        }
        a
    }

    /// Whether `inner` is `outer` or below it.
    #[cfg(test)]
    pub(crate) fn is_within(&self, inner: usize, outer: usize) -> bool {
        self.depth[inner] >= self.depth[outer]
            && self.ancestor_at(inner, self.depth[outer]) == outer
    }

    fn hop(&self) {
        #[cfg(test)]
        self.hops.set(self.hops.get() + 1);
    }

    /// The hops ancestor queries have taken.
    #[cfg(test)]
    pub(crate) fn hops(&self) -> usize {
        self.hops.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A chain `n` deep: node `k` is the child of node `k - 1`.
    fn chain(n: usize) -> Tree {
        let mut tree = Tree::rooted();
        for k in 1..=n {
            tree.grow(k - 1);
        }
        tree
    }

    #[test]
    fn the_common_ancestor_of_two_branches_is_where_they_split() {
        let mut tree = Tree::rooted();
        let left = tree.grow(0);
        let right = tree.grow(0);
        let deep = tree.grow(left);
        let deeper = tree.grow(deep);
        assert_eq!(tree.common_ancestor(deeper, right), 0);
        assert_eq!(tree.common_ancestor(deeper, deep), deep);
        assert_eq!(tree.common_ancestor(deeper, left), left);
        assert!(tree.is_within(deeper, left));
        assert!(!tree.is_within(right, left));
    }

    /// Climbing a chain a thousand deep takes logarithmically many hops, not a
    /// thousand: the whole point of the skip pointers.
    #[test]
    fn climbing_a_deep_chain_is_logarithmic() {
        let tree = chain(1 << 12);
        let deepest = 1 << 12;
        assert_eq!(tree.ancestor_at(deepest, 7), 7);
        assert!(
            tree.hops() <= 4 * 12,
            "{} hops to climb 4096 levels",
            tree.hops()
        );
    }
}
