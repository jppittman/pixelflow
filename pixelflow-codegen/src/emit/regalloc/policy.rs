//! What the allocator prefers, apart from how it finds the registers.
//!
//! Two policies, both pure: what giving up a register costs
//! ([`EvictionRank`]), and what carrying a root across a loop saves (the
//! `*_saved` prices and [`carried`]). Neither looks at a schedule or a
//! register, so whatever finds the registers asks them the same questions.

use alloc::vec;
use alloc::vec::Vec;

use crate::emit::FileId;

/// What it costs the instruction being placed to lose one of its own reads —
/// the tier that outranks every kind of deferred traffic, and the reason an
/// operand's register is a *priced* choice rather than a forbidden one.
///
/// Ordered cheapest first. The distinction between the two read-here cases is
/// what makes the exhausted pool feasible: when every held register belongs to
/// something this instruction reads, the loser has to be one of them, and only
/// one kind of them costs no further register.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(super) enum ReadHere {
    /// Not read by this instruction or by a guard emitted before it.
    No,
    /// Read here, and it is the operand the encoding consumes *from the
    /// destination*
    /// ([`OperandSource::Destination`](super::OperandSource::Destination)):
    /// losing its register means one reload — into `dst`, which is the
    /// register it is losing — and no other register at all.
    FromDst,
    /// Read here and needs a register of its own to be read from: a reload
    /// register the pool then has to find too, or a guard's mask register for
    /// a branch emitted before the instruction.
    NeedsRegister,
}

/// Whether giving up a value's register costs a store, cheapest first.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(super) enum Store {
    /// The value is defined again where it is read: its definition is
    /// rematerializable, so it has no slot and never will.
    Never,
    /// The slot already holds the value.
    NotNeeded,
    /// The register holds the only copy.
    Needed,
}

/// What giving up a register costs, cheapest first — the order eviction picks
/// its loser in.
///
/// A value defined again where it is read needs no store, nor does one whose
/// slot already holds it; anything else has to be written out. Belady's
/// distance breaks ties *within* a tier and only within one: the traffic an
/// eviction causes outweighs how long it waits to cause it.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct EvictionRank {
    /// Read by the instruction being placed — see [`ReadHere`].
    ///
    /// The fields below price the traffic an eviction *defers*; for a value
    /// read right here there is nothing to defer, so taking its register buys
    /// a reload inside this very instruction. Without this, a value already in
    /// its slot is the standing favourite — and at a read, the standing
    /// favourite is whichever value the instruction is reading.
    ///
    /// Answered from the instruction's own read set, never from the read
    /// cursor: the kept-reload step advances the cursor past the current index
    /// (`next_read(operand, i + 1)`), so by the time the destination is
    /// contested a just-kept operand would read as "not needed now".
    read_here: ReadHere,
    /// Whether losing the register costs a store: not for a value defined
    /// again where it is read, nor once the slot holds it.
    store: Store,
    /// Nearest next read *last*, so the cheapest loser is the one used
    /// farthest out.
    nearest: core::cmp::Reverse<usize>,
}

impl EvictionRank {
    /// The rank of a value read here as `read_here`, whose next read is
    /// `distance` instructions away if it has one.
    pub(super) fn new(read_here: ReadHere, store: Store, distance: Option<usize>) -> Self {
        Self {
            read_here,
            store,
            nearest: core::cmp::Reverse(distance.unwrap_or(usize::MAX)),
        }
    }
}

/// The per-file budgets [`carried`] fills, indexed by `FileId as usize`. A
/// carry holds one register of one file from before a loop to its latch;
/// `Flags` is never carried, and the Opmask budget is zero: a predicate live
/// into a loop head is stored there.
pub(super) type Budget = [usize; 4];

/// The vector registers a loop's body keeps for itself when it carries as many
/// as it can: a file's carry budget is its members minus this.
///
/// Seven is the floor the legacy allocator reserved, `MIN_SCRATCH`
/// (`Scratch::MAX_TEMPS + 3`), and the constant the carry pricing was fitted
/// with (escape-hatches, 2026-09-04 and 05, where three principled
/// replacements lost to it). It is kept so that the switch to selection
/// changes how a carry is represented and not how many there are: 16 vector
/// registers carry 9, as they do today. What the body needs from the seven is
/// the widest instruction's operands (a `MulAdd` reads three and writes one)
/// and the reloads and copies the allocator inserts beside it.
pub(super) const CARRY_RESERVE: usize = 7;

/// The same for the general file. Five leaves x86's nine members four carries:
/// legacy's two pointer carries (its pool was `r9`-`r11`, one of them its
/// floor), plus `out` and `pitch`, which legacy kept outside the pool and
/// selection leases like any other value; the pool base that was a fifth is
/// gone (constants are RIP-relative). aarch64 keeps the same reserve, and the
/// difference from its legacy carries is attributed where that backend switches.
pub(super) const GENERAL_CARRY_RESERVE: usize = 5;

/// What carrying a root saves, per call, when `reads` reads of it sit in a
/// scope that runs `trips` times: one reload per read per run.
pub(super) const fn reads_saved(reads: usize, trips: usize) -> usize {
    reads * trips
}

/// What carrying a loop's own binder, or its accumulator, saves per call: the
/// trip test and the step read the binder once each per trip, and the combine
/// reloads and stores the accumulator once each. (A `SEQ` fold's accumulator
/// is never read or written, so it is never priced.)
pub(super) const fn loop_state_saved(trips: usize) -> usize {
    2 * trips
}

/// A root that could be carried, and what it would cost the scopes it is
/// live across.
pub(super) struct Candidate<R> {
    pub(super) weight: usize,
    pub(super) class: FileId,
    /// The scopes the carry is live across, by index.
    pub(super) live_across: Vec<usize>,
    pub(super) root: R,
}

/// The roots to carry, hottest first: the candidates by weight, then taken
/// greedily while no scope they are live across has more of their class
/// carried than `above_floor` allows.
///
/// `scopes` is how many scopes `live_across` indexes into. The sort is
/// stable, so the order the candidates arrive in breaks ties.
pub(super) fn carried<R>(
    mut candidates: Vec<Candidate<R>>,
    scopes: usize,
    above_floor: Budget,
) -> Vec<R> {
    candidates.sort_by_key(|c| core::cmp::Reverse(c.weight));

    let mut count: Vec<Budget> = vec![[0; 4]; scopes];
    let mut taken = Vec::new();
    for candidate in candidates {
        let class = candidate.class as usize;
        if candidate
            .live_across
            .iter()
            .any(|&s| count[s][class] >= above_floor[class])
        {
            continue;
        }
        for &s in &candidate.live_across {
            count[s][class] += 1;
        }
        taken.push(candidate.root);
    }
    taken
}
