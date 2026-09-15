//! The content-addressed store a [`Ref`](crate::arena::ExprNode::Ref) names
//! into.
//!
//! Process-global, the way [`BufferIdentity::mint`](crate::arena::BufferIdentity::mint)
//! is process-unique: a [`KernelKey`] means the same kernel everywhere in one
//! program, and nothing outside a program can hold one. Interning is
//! idempotent — the same content interns to the same key and is stored once —
//! so `by_ref` on a kernel a thousand times costs one entry.
//!
//! **A collision never resolves to the wrong kernel.** [`KernelKey`] is a
//! 64-bit digest (see its docs for why it cannot be wider), so the store keeps
//! the full [`Canonical`] form beside each entry — shape bytes and link — and
//! compares it on every intern. Two different kernels landing on one key is a
//! programming-error-class event with no recovery worth writing — the two
//! kernels are both correct and one of them would silently become the other —
//! so it panics rather than returning an error nobody could act on.
//!
//! The store is unbounded and never evicts, for the reason
//! `pixelflow-codegen`'s compile cache is: entries are made at composition
//! time, not per frame, so the population is the program's distinct kernel
//! set.
//!
//! **This module is the `std` feature**, and that is the whole of what the
//! feature buys: a process-global map needs a lock, `core` has none, and a
//! hand-rolled spinlock is a synchronisation primitive the workspace has no
//! other use for. So under `no_std` there is no store — and therefore no
//! [`Kernel::by_ref`](crate::kernel::Kernel::by_ref), and therefore no way to
//! mint a `Ref` at all. The language is the same minus the ability to *name*
//! a kernel: composition still splices, which is what it did everywhere
//! before a name existed.
//!
//! That chain is a type-level statement rather than a comment, which it has
//! to be — an earlier revision of this doc called the `no_std` build "a
//! known, unexercised gap" and joined it, and the job that exercises it went
//! red the first time this module reached CI.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::{Mutex, OnceLock};

use crate::kernel::Kernel;
use crate::key::{Canonical, KernelKey, canonical};

/// One interned kernel: the identity in full, and the kernel it names.
struct Interned {
    /// The canonical form the key digests — shape bytes *and* link. Kept so
    /// a lookup can *verify* rather than trust the digest.
    canonical: Canonical,
    /// The whole value — arena, root, and the tabulations it carries — so a
    /// reference expands to everything the referent was, not to its graph
    /// alone.
    kernel: Kernel,
}

static STORE: OnceLock<Mutex<HashMap<KernelKey, Interned>>> = OnceLock::new();

/// The process-global store of kernels, addressed by content.
///
/// A namespace rather than a value: there is exactly one store, for the same
/// reason there is exactly one `BufferIdentity` counter — an identity that
/// meant different things in two stores would not be an identity.
pub struct KernelStore;

impl KernelStore {
    /// Intern `kernel`, returning the key that names it.
    ///
    /// Idempotent: interning the same content twice yields the same key and
    /// stores one entry.
    ///
    /// # Panics
    ///
    /// Panics if a *different* kernel is already interned under this key —
    /// a digest collision, which would otherwise silently rename one kernel
    /// to the other.
    #[must_use]
    pub fn intern(kernel: &Kernel) -> KernelKey {
        let (arena, root) = kernel.parts();
        let form = canonical(arena, root);
        let key = KernelKey::of_canonical(&form);
        intern_with(key, form, kernel)
    }

    /// The kernel `key` names, or `None` if nothing was interned under it.
    ///
    /// A `Kernel` is `Arc<KernelData>`, so this is a refcount bump handing
    /// back the whole value — arena, root, and carried tabulations — not a
    /// copy and not a fragment of one.
    #[must_use]
    pub fn resolve(key: KernelKey) -> Option<Kernel> {
        let store = STORE.get()?;
        let guard = store.lock().expect("KernelStore: lock poisoned");
        guard.get(&key).map(|held| held.kernel.clone())
    }
}

/// Insert `kernel` under `key`, verifying that `key` does not already name
/// different content.
///
/// Separate from [`KernelStore::intern`] so the collision path can be reached
/// by a test: forcing two canonical byte strings onto one key is not something
/// the digest will do on demand.
fn intern_with(key: KernelKey, form: Canonical, kernel: &Kernel) -> KernelKey {
    let offered = form.key.len();
    let store = STORE.get_or_init(|| Mutex::new(HashMap::new()));
    // The collision is detected under the lock and reported *outside* it:
    // panicking with the guard alive poisons the store for the rest of the
    // process, turning one bad intern into every later `resolve` failing for
    // an unrelated reason.
    let collision = {
        let mut guard = store.lock().expect("KernelStore: lock poisoned");
        match guard.entry(key) {
            Entry::Occupied(held) => {
                let held = held.get();
                (held.canonical != form).then_some(held.canonical.key.len())
            }
            Entry::Vacant(slot) => {
                slot.insert(Interned {
                    canonical: form,
                    kernel: kernel.clone(),
                });
                None
            }
        }
    };
    if let Some(held) = collision {
        panic!(
            "KernelStore: {key:?} already names a different kernel \
             ({held} canonical bytes held, {offered} offered) — a content-hash \
             collision, which cannot be resolved because both kernels are \
             correct and either answer is the other one's bug"
        );
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Kernel;

    #[test]
    fn interning_is_idempotent_and_resolves_to_the_content() {
        let k = Kernel::x().mul(&Kernel::y()).add(&Kernel::constant(7.5));
        let first = KernelStore::intern(&k);
        let again = KernelStore::intern(&k);
        assert_eq!(first, again, "the same content interns to one key");

        // A structurally identical kernel built separately is the same kernel.
        let twin = Kernel::x().mul(&Kernel::y()).add(&Kernel::constant(7.5));
        assert_eq!(KernelStore::intern(&twin), first);

        let got = KernelStore::resolve(first).expect("interned kernels resolve");
        let (want_arena, want_root) = k.parts();
        let (got_arena, got_root) = got.parts();
        assert!(got_arena.subtree_eq(got_root, want_arena, want_root));
    }

    #[test]
    fn different_content_interns_to_different_keys() {
        let a = KernelStore::intern(&Kernel::x().sub(&Kernel::constant(11.25)));
        let b = KernelStore::intern(&Kernel::x().sub(&Kernel::constant(11.5)));
        assert_ne!(a, b);
    }

    #[test]
    fn an_unknown_key_resolves_to_nothing() {
        assert!(KernelStore::resolve(KernelKey::from_bits(0xdead_beef_dead_beef)).is_none());
    }

    /// The whole point of keeping the canonical bytes: a key that already
    /// names different content is refused loudly, so a 64-bit digest can
    /// never quietly rename one kernel to another.
    #[test]
    #[should_panic(expected = "already names a different kernel")]
    fn a_collision_is_refused_rather_than_resolved() {
        const FORCED: KernelKey = KernelKey::from_bits(0x0123_4567_89ab_cdef);
        let first = Kernel::x();
        let second = Kernel::y();
        let (a_arena, a_root) = first.parts();
        let (b_arena, b_root) = second.parts();
        let a_form = canonical(a_arena, a_root);
        let b_form = canonical(b_arena, b_root);
        assert!(a_form != b_form, "the two kernels must differ");

        let _ = intern_with(FORCED, a_form, &first);
        let _refused = intern_with(FORCED, b_form, &second);
    }
}
