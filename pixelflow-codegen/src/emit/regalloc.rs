//! Register allocation for scheduled DAG expressions.
//!
//! Allocation is one algorithm ([`LinearScan`]) parameterised by one
//! description of the target ([`RegisterFile`]). Everything that differs
//! between x86-64 and aarch64 — which registers hold the coordinate inputs,
//! where the allocatable window starts and how wide it is, which fixed
//! registers spilled operands reload into, how many bytes a spilled vector
//! occupies — is a field of that struct and appears nowhere else. Backends
//! declare one `const` and the allocator is architecture-independent.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use super::guards::{SelectGuard, analyze_select_guards};
use super::{Gpr, KReg, Reg, ScheduledOp, operand_sources, reloads_wanted};

/// A value in the program (SSA-style).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(pub u32);

/// One step of a schedule: a value, and the operation that defines it.
///
/// Every step defines exactly one value — the DAG is in SSA form — so a
/// schedule is a sequence of these, and a value's program point is its index
/// in that sequence.
#[derive(Clone, Debug)]
pub struct Def {
    /// The value this step defines.
    pub value: ValueId,
    /// The operation that computes it.
    pub op: ScheduledOp,
}

/// The complete platform-dependent surface of register allocation.
///
/// Allocation policy is target-independent; only these numbers are not. A
/// backend declares one of these as a `const` next to its encodings, and
/// [`RegisterFile::checked`] turns a layout that contradicts itself — a reload
/// register inside the allocatable window, say — into a build error rather
/// than a miscompile that shows up as wrong pixels.
///
/// The register file is described once and consulted everywhere, and it now
/// says only what a target *is*: which registers carry the inputs, which the
/// allocator may hand out, how many an encoding or a guard destroys, and how
/// wide a spilled register is. Nothing in it is a register held back for a
/// need some other file knows about.
/// A set of registers from one file, as a bitmask over register numbers.
///
/// The allocatable pool used to be a base plus a count — a contiguous run. On
/// every real target the free registers were *not* contiguous: registers held
/// back by hand sat in the middle of the range. A range can only ever name
/// whichever free registers happen to be adjacent, so it silently rounded the
/// pool down to a fraction of the machine.
///
/// A set says the true thing: these registers, whichever they are.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RegSet(u32);

impl RegSet {
    /// The empty set.
    pub const EMPTY: Self = Self(0);

    /// The set containing exactly `regs`.
    #[must_use]
    pub const fn of(regs: &[Reg]) -> Self {
        let mut bits = 0u32;
        let mut i = 0;
        while i < regs.len() {
            let r = regs[i].0;
            assert!(
                r < 32,
                "register number out of range for a 32-register file"
            );
            bits |= 1 << r;
            i += 1;
        }
        Self(bits)
    }

    /// The contiguous run `base .. base + count`.
    #[must_use]
    pub const fn range(base: u8, count: u8) -> Self {
        let mut bits = 0u32;
        let mut i = 0;
        while i < count {
            let r = base + i;
            assert!(
                r < 32,
                "register number out of range for a 32-register file"
            );
            bits |= 1 << r;
            i += 1;
        }
        Self(bits)
    }

    /// This set plus every member of `other`.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// This set minus every member of `other`.
    ///
    /// How a scope inside a loop sees the pool: a register carrying a value
    /// across that loop is not available to anything the loop contains.
    #[must_use]
    pub const fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    #[must_use]
    pub const fn contains(self, r: Reg) -> bool {
        r.0 < 32 && self.0 & (1 << r.0) != 0
    }

    /// How many registers the set holds.
    #[must_use]
    pub const fn len(self) -> u8 {
        self.0.count_ones() as u8
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The lowest `n` members, or all of them if the set is smaller.
    ///
    /// This is how [`EmitCtx::max_regs`](super::EmitCtx) forces spilling for
    /// pressure testing: it only ever shrinks.
    #[must_use]
    pub const fn take(self, n: u8) -> Self {
        let mut kept = 0u32;
        let mut taken = 0u8;
        let mut r = 0u8;
        while r < 32 {
            if taken < n && self.0 & (1 << r) != 0 {
                kept |= 1 << r;
                taken += 1;
            }
            r += 1;
        }
        Self(kept)
    }

    /// Members low to high.
    pub fn iter(self) -> impl Iterator<Item = Reg> + use<> {
        (0u8..32).filter(move |r| self.0 & (1 << r) != 0).map(Reg)
    }
}

/// A set of general-purpose registers, as a bitmask over register numbers.
///
/// The GPR file and the vector file ([`RegSet`]) are different physical
/// register files — `rax` is not `xmm0` — so a GPR pool needs its own type,
/// not a second meaning for `RegSet`. It is not [`RegSet`] made generic over
/// [`Gpr`]: `of`/`contains`/`len` run in `const fn` (a backend's
/// `RegisterFile` is declared as a `const`), and stable Rust cannot dispatch
/// a trait method from a const context — so a register-newtype-generic
/// bitset cannot itself be `const`. Small and duplicated beats generic and
/// non-const.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GprSet(u32);

impl GprSet {
    /// The empty set.
    pub const EMPTY: Self = Self(0);

    /// The set containing exactly `regs`.
    #[must_use]
    pub const fn of(regs: &[Gpr]) -> Self {
        let mut bits = 0u32;
        let mut i = 0;
        while i < regs.len() {
            let r = regs[i].0;
            assert!(r < 32, "GPR number out of range for a 32-register file");
            bits |= 1 << r;
            i += 1;
        }
        Self(bits)
    }

    #[must_use]
    pub const fn contains(self, r: Gpr) -> bool {
        r.0 < 32 && self.0 & (1 << r.0) != 0
    }

    /// How many registers the set holds.
    #[must_use]
    pub const fn len(self) -> u8 {
        self.0.count_ones() as u8
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Members low to high.
    pub fn iter(self) -> impl Iterator<Item = Gpr> + use<> {
        (0u8..32).filter(move |r| self.0 & (1 << r) != 0).map(Gpr)
    }
}

/// A set of AVX-512 mask registers (`k0..k7`), as a bitmask.
///
/// See [`GprSet`] for why this is a third concrete bitset rather than a
/// generic one: the same const-fn constraint applies.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MaskSet(u8);

impl MaskSet {
    /// The empty set.
    pub const EMPTY: Self = Self(0);

    /// The set containing exactly `regs`.
    #[must_use]
    pub const fn of(regs: &[KReg]) -> Self {
        let mut bits = 0u8;
        let mut i = 0;
        while i < regs.len() {
            let r = regs[i].0;
            assert!(r < 8, "AVX-512 has only k0..k7");
            bits |= 1 << r;
            i += 1;
        }
        Self(bits)
    }

    #[must_use]
    pub const fn contains(self, r: KReg) -> bool {
        r.0 < 8 && self.0 & (1 << r.0) != 0
    }

    /// How many registers the set holds.
    #[must_use]
    pub const fn len(self) -> u8 {
        self.0.count_ones() as u8
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Members low to high.
    pub fn iter(self) -> impl Iterator<Item = KReg> + use<> {
        (0u8..8).filter(move |r| self.0 & (1 << r) != 0).map(KReg)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct RegisterFile {
    /// Registers holding the coordinate inputs, in order: X, Y, Z, W.
    ///
    /// A `Var(i)` in the schedule is pre-colored to `inputs[i]`; a `Var` past
    /// the end of this array is a value the target cannot supply.
    pub inputs: [Reg; 4],

    /// Every register the allocator may hand out.
    ///
    /// Everything outside it — inputs, callee-saved registers, the backend's
    /// own `fixed` scratch — is off limits by construction, and
    /// [`RegisterFile::checked`] proves the separation.
    pub scratch: RegSet,

    /// How many registers a `Select` short-circuit guard destroys while
    /// reducing its mask to a branch condition.
    ///
    /// One on aarch64, where `UMAXV`/`UMINV` write a scalar into a vector
    /// register before it can reach a general one; zero on the x86 tiers,
    /// whose guards go through `movmskps`/`kortest` and the flags. It is a
    /// count rather than a flag because it is the same kind of statement as
    /// [`RegisterFile::temps_for`] — how many registers an emission destroys —
    /// and a count is what the allocator reserves against.
    ///
    /// A guard is emitted *between* instructions, at the head of a guarded arm
    /// and at the `Select` that owns it, so this is reserved on those
    /// instructions and nowhere else.
    pub guard_temps: u8,

    /// Registers the backend's own instruction emission clobbers, outside the
    /// allocator's knowledge: an ISA-level temp for a two-operand form, a
    /// gather's index register, and the like.
    ///
    /// The allocator never hands these out; declaring them is what lets
    /// [`RegisterFile::checked`] prove they miss the pool, the inputs and the
    /// reload pair — rather than a comment in an ISA file asserting it. Anything a backend takes for itself belongs here.
    pub fixed: &'static [Reg],

    /// How many registers this backend's encoding of `op` needs beyond the
    /// operands and destination — the instruction temps.
    ///
    /// A two-operand ISA needs one to break a destructive hazard; a sign-flip
    /// needs one to hold the mask. Those used to be `const`s outside the pool,
    /// reserved for the whole kernel because one instruction in it might want
    /// one. Declaring the demand here instead makes the temp an *allocated*
    /// value with a live range of exactly one instruction, so the register is
    /// the allocator's everywhere else.
    ///
    /// It lives on the register file because the file is already "the whole of
    /// what allocation needs to know about the target" — a backend that needs
    /// a temp is stating a fact about its register requirements, which is what
    /// this type is for.
    pub temps_for: fn(&ScheduledOp) -> u8,

    /// Bytes one register occupies when spilled — the backend's vector width.
    ///
    /// 16 for SSE2 and NEON, 32 for AVX2, 64 for AVX-512. This is the stride
    /// [`FrameLayout`](super::FrameLayout) lays spill slots out at, so every
    /// offset the emitter sees is already a real byte displacement. It was
    /// once a universal 16 that each wide backend divided back out at its
    /// every use site; a slot offset that failed to be a multiple of 16 would
    /// then have truncated two live values onto the same stack slot.
    pub vector_bytes: u32,

    /// The GPR holding the JIT ABI's context-pointer argument (the array of
    /// buffer base pointers a `Gather`/`Uniform` load indexes into), if this
    /// target's encodings read one. `None` on a backend with neither op.
    ///
    /// Pinned like a vector `Var` input — fixed by the calling convention,
    /// never itself allocated — and declared here for the same reason
    /// [`RegisterFile::inputs`] is: so [`RegisterFile::checked`] can prove it
    /// misses [`RegisterFile::gpr_scratch`], rather than a comment asserting
    /// the two constants never collide.
    pub gpr_ctx: Option<Gpr>,

    /// GPRs the allocator may hand out as instruction-scoped scratch.
    ///
    /// Unlike [`RegisterFile::scratch`], nothing here ever carries a value
    /// across instructions — a `Gather`/`Uniform`'s address arithmetic is the
    /// only demand this register file was ever chosen by hand to serve, and
    /// it is one instruction's worth of scratch every time. So there is no
    /// GPR-class liveness, no spilling and no eviction: each instruction
    /// simply takes the low members of this set it needs, in order, which
    /// always succeeds because nothing else is ever concurrently live in it.
    pub gpr_scratch: GprSet,

    /// How many GPRs this backend's encoding of `op` needs beyond
    /// [`RegisterFile::gpr_ctx`] — the GPR-class
    /// [`RegisterFile::temps_for`].
    pub gpr_temps_for: fn(&ScheduledOp) -> u8,

    /// AVX-512 mask registers (`k0..k7`) the allocator may hand out as
    /// instruction-scoped scratch: a compare's `vcmpps` destination before it
    /// is widened to a vector mask. Empty on every other backend, which has
    /// no mask-register file at all — masks there are ordinary vectors.
    pub mask_scratch: MaskSet,

    /// The mask-class [`RegisterFile::temps_for`].
    pub mask_temps_for: fn(&ScheduledOp) -> u8,

    /// How many mask registers a `Select` short-circuit guard destroys
    /// reducing its mask to a branch condition — the mask-class
    /// [`RegisterFile::guard_temps`]. AVX-512's guard needs one (`vptestmd`'s
    /// `k`-register destination before `kortestw` reads it into the flags);
    /// every other backend's guard needs none.
    pub mask_guard_temps: u8,
}

impl RegisterFile {
    /// Reject a register file whose regions overlap.
    ///
    /// Call it on every backend's `const` declaration: const evaluation runs
    /// the checks at build time, so an allocatable window that swallows a
    /// reload register cannot reach a running kernel.
    #[must_use]
    pub const fn checked(self) -> Self {
        assert!(
            self.scratch.len() >= Self::MIN_SCRATCH,
            "the allocatable pool is too small for the widest instruction's \
             operands, scratch and destination at once"
        );

        let mut i = 0;
        while i < self.inputs.len() {
            assert!(
                !self.scratch.contains(self.inputs[i]),
                "an input register is inside the allocatable pool: the \
                 allocator would hand a pre-colored register out twice"
            );
            i += 1;
        }

        assert!(
            self.guard_temps as usize <= 1,
            "a backend's Select guard asked for more scratch than `Scratch` \
             reserves for one"
        );

        assert!(
            self.vector_bytes >= 16 && self.vector_bytes.is_power_of_two(),
            "vector_bytes must be a power of two of at least 16"
        );

        // Everything a backend reserves for its own emission must miss every
        // register the allocator reasons about. Without this the disjointness
        // lives only in comments.
        let mut i = 0;
        while i < self.fixed.len() {
            assert!(
                !self.scratch.contains(self.fixed[i]),
                "a fixed backend scratch register is inside the allocatable \
                 pool: emitting an instruction would clobber a live value"
            );
            let mut k = 0;
            while k < self.inputs.len() {
                assert!(
                    self.fixed[i].0 != self.inputs[k].0,
                    "a fixed backend scratch register aliases a coordinate input"
                );
                k += 1;
            }
            i += 1;
        }

        // The GPR-class mirror of the vector checks above: the context
        // pointer is pinned like an `inputs` register, so it must miss the
        // pool the allocator hands out from.
        if let Some(ctx) = self.gpr_ctx {
            assert!(
                !self.gpr_scratch.contains(ctx),
                "the GPR context-pointer input is inside the allocatable GPR pool"
            );
        }

        assert!(
            self.mask_guard_temps as usize <= 1,
            "a backend's Select guard asked for more mask scratch than \
             `Scratch` reserves for one"
        );

        self
    }

    /// The smallest pool any schedule can be allocated against.
    ///
    /// Every *value* survives a small pool by spilling, so the pool has no
    /// lower bound from values alone — that is why one register used to be an
    /// acceptable budget. Instruction scratch has no such escape: it is
    /// registers the encoder destroys mid-instruction, each of which must be a
    /// register, and one that is neither the destination nor any operand.
    ///
    /// **Seven**, and it is a computation rather than a constant. Every
    /// register one instruction needs at once is now the allocator's, so the
    /// floor is the maximum over ops and backends of
    ///
    /// ```text
    ///   temps(op)                    // RegisterFile::temps_for, 0..=MAX_TEMPS
    /// + operands(op)                 // each one either resident or reloaded
    /// + guard                        // a guarded arm's mask, plus guard_temps
    /// + result                       // the scope's tail materialization
    /// + 1                            // the destination
    /// ```
    ///
    /// where an operand costs one register whether it is *resident* (holding a
    /// pool register) or *reloaded* (holding one of this instruction's reload
    /// reservations) — which is why the count is over operands rather than
    /// over spilled operands, and why the dst-as-reload-target rule below
    /// lowers instantaneous pressure without lowering this floor.
    ///
    /// | worst instruction | temps | operands | guard | result | dst | total |
    /// |---|---|---|---|---|---|---|
    /// | AVX2 gather at a guarded arm's head | 4 | 1 | 1 | 0 | 1 | **7** |
    /// | AVX2 gather anywhere else | 4 | 1 | 0 | 0 | 1 | 6 |
    /// | aarch64 `Select` at a guarded arm's head | 0 | 3 | 2 | 0 | 1 | 6 |
    /// | SSE2 `Select` at a guarded arm's head | 1 | 3 | 1 | 0 | 1 | 6 |
    ///
    /// The `result` column is 0 everywhere because it is reserved only for a
    /// body whose root was hoisted out entirely — a placeholder that emits
    /// nothing, has no operands, no temps and no destination.
    ///
    /// Below this, shrinking the pool stops producing more spilling and starts
    /// producing an instruction with nowhere to put its scratch: every *value*
    /// survives a small pool by going to memory, and scratch the encoder
    /// destroys mid-instruction has no such escape.
    pub const MIN_SCRATCH: u8 = Scratch::MAX_TEMPS as u8 + 3;

    /// Cap the scratch pool at a smaller budget, leaving every other region
    /// where it is.
    ///
    /// This is how [`EmitCtx::max_regs`](super::EmitCtx) forces spilling for
    /// pressure testing. It only ever *shrinks* the pool: a budget above the
    /// target's own count would hand the allocator registers this file has
    /// reserved for reloads or builtins. It does not shrink past
    /// [`MIN_SCRATCH`](Self::MIN_SCRATCH), which is not a budget question but
    /// an encoding one.
    #[must_use]
    pub const fn capped(self, max_scratch: Option<u8>) -> Self {
        match max_scratch {
            Some(n) => Self {
                scratch: self.scratch.take(if n < Self::MIN_SCRATCH {
                    Self::MIN_SCRATCH
                } else {
                    n
                }),
                ..self
            },
            None => self,
        }
    }

    /// This file as a scope *inside* a loop sees it: the pool minus every
    /// register carrying a value across that loop.
    ///
    /// Carried registers are not `fixed` — `fixed` is what a backend holds for
    /// its own encodings, and nothing does any more. These are ordinary
    /// allocations of an outer scope whose live range spans the scopes within,
    /// which is exactly what the nest's liveness says and what allocating each
    /// region against the full pool used to ignore.
    #[must_use]
    pub const fn inside(self, carried: RegSet) -> Self {
        Self {
            scratch: self.scratch.without(carried),
            ..self
        }
    }

    /// The allocatable scratch registers, low to high.
    fn scratch(&self) -> impl Iterator<Item = Reg> + use<> {
        self.scratch.iter()
    }
}

/// Which scope of a loop nest: the enclosing regions, outermost first, and
/// then the innermost body.
///
/// A **name**, not a coordinate. It used to be half of one — [`Point`] was
/// `(scope, index)` ordered lexicographically — and that only worked because
/// this nest is a *chain*: scopes totally ordered by nesting, and all of an
/// outer scope's code preceding all of an inner scope's. The second stops
/// being true the moment a scope opens in the *middle* of another, which is
/// what a surviving `Reduce` is: `body`'s own defs sit on both sides of the
/// fold's. So the ordering moved to where it is always meaningful — within
/// one scope — and this is now only the key that says which one.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Scope {
    /// An enclosing region; `Region(0)` is the outermost.
    Region(usize),
    /// The innermost body, inside every back edge.
    Body,
}

/// A program point: a position in one scope's schedule.
///
/// Which scope is not part of it. A [`Placement`] is one scope's answer, and
/// every query against one is asked from inside that scope, so carrying the
/// scope here would be carrying it twice — and a comparison between two
/// scopes' points is exactly the question a loop nest makes meaningless.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Point {
    /// Position in the scope's schedule.
    pub index: usize,
}

impl Point {
    /// The first point of a scope: where an iteration begins, and where a
    /// value an enclosing scope parked is picked up.
    pub const HEAD: Self = Self { index: 0 };

    /// The last point of a scope — after everything it schedules.
    ///
    /// "Where does this value end an iteration", which is the question a back
    /// edge asks. It used to name the end of the whole *nest*, which is the
    /// same point only for the innermost scope; every scope has a back edge of
    /// its own to reconcile.
    pub const TAIL: Self = Self { index: usize::MAX };
}

/// Where the allocator decided a value lives, over one range of its life.
///
/// Deliberately carries no stack address: choosing that a value spills and
/// choosing *where* it spills are different decisions, and the second belongs
/// to [`FrameLayout`](super::FrameLayout), which is what knows about frames.
/// The emitter reads the composition of the two as [`Loc`](super::Loc).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Where {
    /// In this register.
    Reg(Reg),
    /// Evicted to a stack slot.
    Spilled,
    /// Evicted, but it is a constant (these are the `f32` bits): it lives
    /// nowhere and is re-emitted at each use, which beats a store plus a
    /// reload.
    Remat(u32),
}

/// One range of a value's life: from `from` (inclusive) until the next range's
/// `from` (exclusive), the value lives at `at`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Span {
    /// The program point this range starts at.
    pub from: Point,
    /// Where the value lives over it.
    pub at: Where,
}

/// Where a value lives, at every point of **one scope**.
///
/// A **non-empty, strictly increasing sequence** of [`Span`]s. Non-empty by
/// construction — a value lives somewhere from its definition on — which is
/// why the first range is a field rather than the head of a `Vec` something
/// could empty; the rest is usually empty, and an empty `Vec` does not
/// allocate.
///
/// One scope, not the nest, because a sequence is what a *straight line* has.
/// A scope is a loop body and is scanned straight through; the nest is a tree,
/// and a value's life across it is not an interval sequence in any coordinate
/// a tree admits. What crosses a scope boundary is carried by the park —
/// a slot, or a register held across the loops — and the scope inside reads
/// that as its own first span.
///
/// One location for a whole life was the old shape, and it is the shape that
/// makes two things unsayable. A value hot in part of a region and cold in the
/// rest cannot hold a register for the hot part only; and a root computed in
/// an outer region and read inside the loops within cannot be in that region's
/// register *and* in a slot for the loops — which is two locations over one
/// life, and is what the nest-wide map here has to express. That second one is
/// why the `carries` side-channel is gone: it was the half of this answer the
/// old shape could not hold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    first: Span,
    rest: Vec<Span>,
}

impl Placement {
    /// A placement that starts at `first` and never changes.
    #[must_use]
    pub fn new(first: Span) -> Self {
        Self {
            first,
            rest: Vec::new(),
        }
    }

    /// The same placement, changing to `next.at` from `next.from` onward.
    ///
    /// # Panics
    /// In debug builds, if `next` does not start strictly after every range
    /// already here — the sequence is increasing, and an out-of-order range
    /// would make [`Placement::at`] answer with a location the value had
    /// already left.
    #[must_use]
    pub fn then(mut self, next: Span) -> Self {
        debug_assert!(
            next.from > self.rest.last().unwrap_or(&self.first).from,
            "placement ranges must strictly increase"
        );
        self.rest.push(next);
        self
    }

    /// The point this value is defined at — where its first range starts.
    #[must_use]
    pub fn defined_at(&self) -> Point {
        self.first.from
    }

    /// Where the value lives at `point`.
    ///
    /// Total: the last range whose `from` is at or before `point`. A query
    /// before the definition — which no caller can make, since an operand is
    /// read after it is defined — answers with the first range rather than
    /// panicking.
    #[must_use]
    pub fn at(&self, point: Point) -> Where {
        let after = self.rest.partition_point(|s| s.from <= point);
        match after.checked_sub(1) {
            Some(i) => self.rest[i].at,
            None => self.first.at,
        }
    }

    /// Every range of this value's life, in order.
    pub fn spans(&self) -> impl Iterator<Item = Span> + use<'_> {
        core::iter::once(self.first).chain(self.rest.iter().copied())
    }

    /// Every location this value occupies, in order.
    pub fn locations(&self) -> impl Iterator<Item = Where> + use<'_> {
        self.spans().map(|s| s.at)
    }

    /// Whether any range of this value's life is in a stack slot.
    #[must_use]
    pub fn spills(&self) -> bool {
        self.locations().any(|at| at == Where::Spilled)
    }

    /// Every register this value occupies over its life.
    pub fn registers(&self) -> impl Iterator<Item = Reg> + use<'_> {
        self.locations().filter_map(|at| match at {
            Where::Reg(r) => Some(r),
            Where::Spilled | Where::Remat(_) => None,
        })
    }
}

/// One scope's evaluation order and instruction scratch — everything about a
/// scope that is not a placement.
///
/// The schedule is an *output* because choosing it is part of allocating.
/// [`LinearScan`] hands back the order it was given; an allocator whose
/// register assignment falls out of evaluation order — Sethi-Ullman, where the
/// heavier subtree is emitted first and the register is a function of tree
/// position — hands back the order it chose.
#[derive(Debug)]
struct ScopeCode {
    /// Dense by `ValueId.0`: where each value this scope touches lives, at
    /// every point of it. `None` for a value this scope never sees — which is
    /// most of them, in most scopes.
    placements: Vec<Option<Placement>>,
    /// Evaluation order: the schedule the emitter walks.
    schedule: Vec<Def>,
    /// The scratch each position in `schedule` may destroy.
    ///
    /// Indexed by schedule position rather than by value because scratch is not
    /// a value: it holds nothing before the instruction and nothing after, so
    /// it has no `ValueId` and no place in the placements.
    scratch: Vec<Scratch>,
    /// Values this scope computes for the scopes inside it, in slot order.
    /// Empty for the body, which parks nothing.
    roots: Vec<ValueId>,
}

/// The registers one instruction may destroy for its own duration.
///
/// Each field is a role, not a slot — the allocator picks a register for each
/// one it fills, disjoint from the instruction's operands, its destination, and
/// each other. Reading them positionally out of a shared array is exactly the
/// convention this replaced.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Scratch {
    /// The encoding's own scratch: a sign mask, a Newton-Raphson correction,
    /// the halves a gather assembles its result from — as many registers as
    /// [`RegisterFile::temps_for`] asked for, and no more.
    ///
    /// Private, and read through [`Scratch::temp`], because the *order* here is
    /// a contract between one backend's `temps_for` and that same backend's
    /// encoder. That is a local agreement inside one ISA file; letting anything
    /// else index it would make it a convention spanning the codebase, which is
    /// what the roles below exist to avoid.
    temps: [Option<Reg>; Scratch::MAX_TEMPS],

    /// Registers this instruction's operands are reloaded into, for the
    /// operands that are not in one at this point.
    ///
    /// Private, and read through [`Scratch::reload`], because *which* operand
    /// takes which of these is not positional guesswork: it is
    /// [`operand_sources`](super::operand_sources), the one function both the
    /// allocator (counting reservations) and the emitter (naming registers)
    /// call. A second copy of that mapping would be a convention spanning two
    /// files, and it is precisely the convention that has to hold
    /// register-for-register.
    ///
    /// This is where `arm_reload` went. A `Select`'s second spilled arm is not
    /// a role of its own — it is operand 2 with nowhere to be — and the same
    /// was true of `RegisterFile::reload[1]`, which served every other operand
    /// of every other instruction from outside the pool.
    reloads: [Option<Reg>; Scratch::MAX_RELOADS],

    /// A register to reload a short-circuit guard's mask into, when the mask
    /// is not in one where the guard is emitted.
    ///
    /// One suffices however many guards begin here: each resolves its mask and
    /// branches immediately, so the register is dead again before the next
    /// one needs it.
    pub guard_mask: Option<Reg>,

    /// A register the guard's mask reduction destroys — see
    /// [`RegisterFile::guard_temps`]. `None` on the tiers whose guards go
    /// through the flags.
    pub guard_temp: Option<Reg>,

    /// A register to materialize this scope's result into, reserved on the
    /// scope's last instruction.
    ///
    /// The result is usually the last instruction's own destination and this
    /// goes unused. It is not always: a body whose root was hoisted out
    /// entirely reads that root from its park, and the scaffold needs it in a
    /// register to store.
    pub result: Option<Reg>,

    /// GPR-class temps this instruction reserved — see
    /// [`RegisterFile::gpr_temps_for`]. Read through [`Scratch::gpr_temp`],
    /// for the same reason [`Scratch::temp`] is private: the order is a
    /// contract between one backend's `gpr_temps_for` and that backend's own
    /// encoder, not a codebase-wide convention.
    gpr_temps: [Option<Gpr>; Scratch::MAX_GPR_TEMPS],

    /// Mask-class temps this instruction reserved — see
    /// [`RegisterFile::mask_temps_for`]. Read through [`Scratch::mask_temp`].
    mask_temps: [Option<KReg>; Scratch::MAX_MASK_TEMPS],

    /// A mask register the guard's mask reduction destroys — the mask-class
    /// [`Scratch::guard_temp`]. `None` on every backend but AVX-512, whose
    /// guard reduces the mask into a `k`-register (`vptestmd`) before
    /// `kortestw` reads it into the flags.
    pub mask_guard_temp: Option<KReg>,
}

impl Scratch {
    /// The most scratch registers any one encoding asks for.
    ///
    /// Four: AVX2 assembles a 256-bit gather from two 128-bit halves, which
    /// costs the half-sequence's own index and value registers plus one of
    /// each to carry the high half while the low one is built.
    pub const MAX_TEMPS: usize = 4;

    /// The most reload targets any one instruction asks for.
    ///
    /// Two. Three operands is the widest op, and the one that must reach the
    /// destination anyway — a `Select`'s mask, an FMA's addend, a two-operand
    /// binary's left — is reloaded straight into it rather than into a
    /// reservation.
    pub const MAX_RELOADS: usize = 2;

    /// The most GPR-class temps any one encoding asks for.
    ///
    /// Three: aarch64's scalar-load gather needs a base pointer, a per-lane
    /// index and a loaded value, each a GPR.
    pub const MAX_GPR_TEMPS: usize = 3;

    /// The most mask-class temps any one encoding asks for.
    ///
    /// One: AVX-512 is the only backend with a mask-register file at all, and
    /// every one of its uses — a compare's `vcmpps` destination, a guard's
    /// `vptestmd` destination — needs exactly one `k`-register, transiently,
    /// never two at once.
    pub const MAX_MASK_TEMPS: usize = 1;

    /// A `Scratch` with the registers a test wants to hand an encoder.
    ///
    /// The allocator is what fills these in production; a test that exercises
    /// one encoding in isolation has no allocator, so it says outright which
    /// registers the encoder may destroy.
    #[cfg(test)]
    #[must_use]
    pub const fn for_test(
        temps: Option<[Reg; Self::MAX_TEMPS]>,
        reloads: [Option<Reg>; Self::MAX_RELOADS],
    ) -> Self {
        Self::for_test_with_classes(temps, reloads, None, None)
    }

    /// [`Scratch::for_test`], additionally handing an encoder the GPR- and
    /// mask-class scratch it asks for — the class-B tests (a backend's
    /// `Gather`/`Uniform`/compare coverage) need these too.
    #[cfg(test)]
    #[must_use]
    pub const fn for_test_with_classes(
        temps: Option<[Reg; Self::MAX_TEMPS]>,
        reloads: [Option<Reg>; Self::MAX_RELOADS],
        gpr_temps: Option<[Gpr; Self::MAX_GPR_TEMPS]>,
        mask_temp: Option<KReg>,
    ) -> Self {
        let temps = match temps {
            Some([a, b, c, d]) => [Some(a), Some(b), Some(c), Some(d)],
            None => [None; Self::MAX_TEMPS],
        };
        let gpr_temps = match gpr_temps {
            Some([a, b, c]) => [Some(a), Some(b), Some(c)],
            None => [None; Self::MAX_GPR_TEMPS],
        };
        Self {
            temps,
            reloads,
            guard_mask: None,
            guard_temp: None,
            result: None,
            gpr_temps,
            mask_temps: [mask_temp],
            mask_guard_temp: mask_temp,
        }
    }

    /// The `i`'th register this instruction's encoding asked for.
    ///
    /// `i` is the backend's own numbering, matching the count its
    /// [`RegisterFile::temps_for`] returned.
    #[must_use]
    pub fn temp(&self, i: usize) -> Option<Reg> {
        self.temps.get(i).copied().flatten()
    }

    /// The `i`'th reload target this instruction reserved.
    ///
    /// `i` is [`operand_sources`](super::operand_sources)' numbering, which is
    /// operand order over the operands that need one.
    #[must_use]
    pub fn reload(&self, i: usize) -> Option<Reg> {
        self.reloads.get(i).copied().flatten()
    }

    /// The `i`'th GPR this instruction's encoding asked for.
    ///
    /// `i` is the backend's own numbering, matching the count its
    /// [`RegisterFile::gpr_temps_for`] returned.
    #[must_use]
    pub fn gpr_temp(&self, i: usize) -> Option<Gpr> {
        self.gpr_temps.get(i).copied().flatten()
    }

    /// The `i`'th mask register this instruction's encoding asked for.
    ///
    /// `i` is the backend's own numbering, matching the count its
    /// [`RegisterFile::mask_temps_for`] returned.
    #[must_use]
    pub fn mask_temp(&self, i: usize) -> Option<KReg> {
        self.mask_temps.get(i).copied().flatten()
    }
}

/// What an allocator makes of a [`ScopedSchedule`]: **one** answer to one
/// question — where does each value live, at every point in the nest.
///
/// The per-region placement maps and the `carries` side-channel were two
/// encodings of that answer, and one of them existed because the other could
/// not say it: a root computed in a region and read by the loops inside lives
/// in that region's register and then in a slot, which is two locations over
/// one life. A ranged [`Placement`] says that directly, so there is one map
/// here and nothing beside it.
///
/// `ValueId`s are *not* partitioned by the nest — a `Var` or `Const` leaf
/// feeding both an invariant expression and a varying one appears in both
/// scopes' schedules, with an independently chosen location in each. That is
/// why a placement belongs to a [`ScopeCode`] rather than to the nest: the two
/// answers are both true, of different scopes, and a nest-wide map has room
/// for only one of them.
#[derive(Debug)]
pub struct NestAllocation {
    /// One per input region, in the same order — outermost first.
    regions: Vec<ScopeCode>,
    /// The innermost body.
    body: ScopeCode,
}

impl NestAllocation {
    /// How many enclosing regions the nest has.
    #[must_use]
    pub fn regions(&self) -> usize {
        self.regions.len()
    }

    /// The allocation as `scope` reads it.
    ///
    /// # Panics
    /// If `scope` names a region this nest does not have.
    #[must_use]
    pub fn scope(&self, scope: Scope) -> Allocation<'_> {
        assert!(
            self.code(scope).is_some(),
            "{scope:?} is not a scope of this nest"
        );
        Allocation { nest: self, scope }
    }

    /// The innermost body — the whole answer for a loop-free schedule.
    #[must_use]
    pub fn body(&self) -> Allocation<'_> {
        self.scope(Scope::Body)
    }

    fn code(&self, scope: Scope) -> Option<&ScopeCode> {
        match scope {
            Scope::Region(i) => self.regions.get(i),
            Scope::Body => Some(&self.body),
        }
    }

    /// The register a root is **carried** in across the loops inside the
    /// region that computes it, if it is carried at all.
    ///
    /// A root is carried exactly when the innermost scope picks it up in a
    /// register: the body reads it from there on every iteration instead of
    /// reloading it from a slot at every use. There is no separate map saying
    /// so — that map was `carries`, and the park the body starts from already
    /// answers.
    ///
    /// # Panics
    /// If `root` is in no scope of this nest.
    #[must_use]
    pub fn carried(&self, root: ValueId) -> Option<Reg> {
        match self.body().at_head(root) {
            Where::Reg(r) => Some(r),
            Where::Spilled | Where::Remat(_) => None,
        }
    }

    /// Override where a value lives, for the whole of its life in `scope`.
    ///
    /// The emitter's own tests pin a value somewhere the allocator did not
    /// choose. One write, so the placement cannot desync from itself.
    ///
    /// # Panics
    /// If `v` is not in `scope` — a placement has to start somewhere, and only
    /// the allocation knows where `v` is defined.
    pub fn place(&mut self, scope: Scope, v: ValueId, at: Where) {
        let from = self.scope(scope).placement(v).defined_at();
        let code = match scope {
            Scope::Region(i) => &mut self.regions[i],
            Scope::Body => &mut self.body,
        };
        code.placements[v.0 as usize] = Some(Placement::new(Span { from, at }));
    }
}

/// The allocation as one scope reads it: that scope's schedule, scratch and
/// placements.
///
/// The scope is baked in, so callers hand over a *local* schedule index and
/// cannot name a point in some other scope by accident.
#[derive(Copy, Clone, Debug)]
pub struct Allocation<'a> {
    nest: &'a NestAllocation,
    scope: Scope,
}

impl<'a> Allocation<'a> {
    /// Evaluation order: the schedule the emitter walks.
    #[must_use]
    pub fn schedule(&self) -> &'a [Def] {
        &self.code().schedule
    }

    /// The values this scope computes for the scopes inside it, in slot order.
    #[must_use]
    pub fn roots(&self) -> &'a [ValueId] {
        &self.code().roots
    }

    /// The scratch the instruction at schedule position `i` may destroy.
    #[must_use]
    pub fn scratch(&self, i: usize) -> Scratch {
        self.code().scratch.get(i).copied().unwrap_or_default()
    }

    /// Where `v` lives at position `index` of this scope.
    ///
    /// The query carries its own point because a placement is a schedule, not
    /// an annotation: asking where a value is without saying *when* is a
    /// question with no answer once a live range can be split.
    ///
    /// # Panics
    /// If this scope never sees `v`.
    #[must_use]
    pub fn where_at(&self, v: ValueId, index: usize) -> Where {
        self.placement(v).at(Point { index })
    }

    /// Where `v` lives when this scope is entered.
    ///
    /// For a value an enclosing scope computed, this is its park — the one
    /// place it is on both of the head's predecessors, the fall-through and
    /// the back edge — and it is where every iteration expects to find it.
    ///
    /// # Panics
    /// If this scope never sees `v`.
    #[must_use]
    pub fn at_head(&self, v: ValueId) -> Where {
        self.where_at(v, Point::HEAD.index)
    }

    /// The register a root is carried in across the loops inside its region.
    #[must_use]
    pub fn carried(&self, root: ValueId) -> Option<Reg> {
        self.nest.carried(root)
    }

    /// Where `v` lives at every point of this scope.
    ///
    /// # Panics
    /// If this scope never sees `v`.
    #[must_use]
    pub fn placement(&self, v: ValueId) -> &'a Placement {
        self.placement_of(v)
            .unwrap_or_else(|| panic!("{v:?} is not in {:?}", self.scope))
    }

    /// Where `v` lives at every point of this scope, or `None` if it never
    /// reaches here.
    ///
    /// The fallible form, for the questions asked *about* a scope rather than
    /// from inside it — "does the loop within hold this in one register the
    /// whole way", where a value the loop never reads is vacuously fine.
    #[must_use]
    pub fn placement_of(&self, v: ValueId) -> Option<&'a Placement> {
        self.code().placements.get(v.0 as usize)?.as_ref()
    }

    /// The points of this scope at which `v` changes place, in order.
    ///
    /// This is what lets the emitter maintain its location table incrementally
    /// — one pass, O(total spans) — rather than asking where every value is at
    /// every instruction.
    ///
    /// # Panics
    /// If this scope never sees `v`.
    pub fn transitions(self, v: ValueId) -> impl Iterator<Item = (usize, Where)> + use<'a> {
        self.placement(v).spans().map(|s| (s.from.index, s.at))
    }

    /// The scopes inside this one, outermost first.
    ///
    /// A root this scope parks is picked up by each of them, so "where does
    /// the code within keep it" is a question about all of them together.
    pub fn within(self) -> impl Iterator<Item = Allocation<'a>> + use<'a> {
        let nest = self.nest;
        let first = match self.scope {
            Scope::Region(i) => i + 1,
            Scope::Body => usize::MAX,
        };
        (first..nest.regions.len())
            .map(Scope::Region)
            .chain((first != usize::MAX).then_some(Scope::Body))
            .map(move |scope| Allocation { nest, scope })
    }

    /// Whether this scope *reads* `v` from an enclosing region's park rather
    /// than computing it.
    ///
    /// Such a value's entry in this schedule is a placeholder, and its address
    /// is a hoist slot that outlives every region's own frame — so it is not
    /// this frame's to place. Narrower than "defined elsewhere": a `Const`
    /// leaf shared with an enclosing region is genuinely computed here too,
    /// and does need a location of its own.
    #[must_use]
    pub fn parked_by_an_enclosing_scope(&self, v: ValueId) -> bool {
        let outside = match self.scope {
            Scope::Region(i) => i,
            Scope::Body => self.nest.regions.len(),
        };
        self.nest.regions[..outside]
            .iter()
            .any(|r| r.roots.contains(&v))
    }

    fn code(&self) -> &'a ScopeCode {
        self.nest
            .code(self.scope)
            .unwrap_or_else(|| unreachable!("`NestAllocation::scope` checked this"))
    }
}

/// Assign physical registers to an expression DAG.
///
/// A pure function from a program and a register file to a placement for every
/// value in it. Purity is load-bearing, not incidental: the collapse-loop
/// driver runs allocation once to size a stack frame and again to emit into
/// that frame, and a disagreement between the two runs misplaces every spill.
///
/// Allocation is not a local decision. Liveness needs the whole program, and
/// the eviction rule that makes the difference — Belady's, evict whatever is
/// used farthest in the future — is *defined* in terms of the future. So an
/// implementation sees the entire DAG, and owes an answer for every value in
/// the schedule it returns.
///
/// Running out of registers is not a failure; it is a spill. There is no error
/// case: a DAG this cannot allocate is a DAG the pipeline should never have
/// produced, and it panics at the point of failure rather than handing a
/// caller a string it can only propagate.
pub trait RegisterAllocator {
    /// Place every value in a loop nest, and choose the evaluation order.
    ///
    /// This is the whole job, and it is deliberately the *only* required
    /// method. The nest — not a flat schedule — is the honest input, because
    /// where a value is read decides what keeping it in a register is worth:
    /// a read inside a loop costs its reload once per iteration, a read in the
    /// prologue costs it once. An allocator handed a flat `Vec<Def>` cannot
    /// tell those apart, so it cannot price them, and the only policy it can
    /// implement is one that ignores the difference.
    ///
    /// Taking the nest by value because choosing the order is part of the job:
    /// an implementation may permute what it is handed, and returns the order
    /// it settled on.
    fn allocate_nest(&self, nest: ScopedSchedule, file: &RegisterFile) -> NestAllocation;

    /// A loop-free schedule, which is the degenerate nest: one body, no
    /// regions, nothing carried across anything.
    ///
    /// Provided rather than required, and in that direction on purpose. It
    /// used to be the other way round — `allocate` required, `allocate_nest`
    /// defaulted to calling it once per region — which put the loop policy in
    /// this trait's default body, where every implementation inherited it and
    /// none of them owned it. A trait should say what an allocator answers,
    /// not how.
    fn allocate(&self, dag: Vec<Def>, file: &RegisterFile) -> NestAllocation {
        self.allocate_nest(
            ScopedSchedule {
                regions: Vec::new(),
                body: dag,
            },
            file,
        )
    }
}

/// A schedule split by scope: the innermost body, plus one region per binder
/// it can be lifted out of.
///
/// This is the loop nest as data. `regions` runs outermost first, so
/// `regions[0]` is what happens once per call and the last region is what
/// happens once per iteration of the next-to-innermost binder; `body` is what
/// happens at every sample. Each region's `roots` are the values it computes
/// for the regions inside it.
///
/// The body is a field rather than the last element of `regions` because it
/// is genuinely a different thing: it is inside every back edge, it produces
/// the result, and it parks nothing. A list that could hold it would let the
/// two be confused.
pub struct ScopedSchedule {
    /// One per binder, outermost first. A binder nothing can be lifted out of
    /// still gets a region; it is simply empty.
    pub regions: Vec<ScopeRegion>,
    /// The innermost region: evaluated at every sample.
    pub body: Vec<Def>,
}

/// One scope of a [`ScopedSchedule`].
pub struct ScopeRegion {
    /// Values this region computes for the ones inside it.
    pub roots: Vec<ValueId>,
    /// What it computes, in topological order.
    pub schedule: Vec<Def>,
}

/// Linear scan with Belady eviction, live-range splitting and constant
/// rematerialization.
///
/// One forward pass per scope. At each instruction it
/// 1. frees registers whose owner is not read again,
/// 2. reserves the instruction's scratch,
/// 3. brings a spilled operand back into a pool register and *keeps* it there
///    when it is read again soon enough to be worth one,
/// 4. gives the destination a free register, or evicts.
///
/// Eviction **splits**: the loser keeps the register it held up to that point
/// and its life continues in its slot, or — for a constant — nowhere at all,
/// since re-emitting the load beats a store plus a reload. A single location
/// per value made a value's whole life pay for the moment of pressure that
/// evicted it.
///
/// The evaluation order it returns is the one it was given: the arena's
/// append-only structure already guarantees a topological order, so there is
/// nothing to linearize.
///
/// Coordinate inputs are pinned to `file.inputs` and never enter the scratch
/// pool, which [`RegisterFile::checked`] keeps disjoint from them.
///
/// O(n × k) for n values and a k-register pool. For the pools here (6–26
/// registers) that is effectively O(n).
///
/// The DAGs reaching this are in SSA form, which makes their interference
/// graphs chordal — the shape on which greedy coloring is optimal. What
/// coloring does not decide, and this does, is *spill placement*: where a live
/// range is cut, and where the value comes back.
#[derive(Copy, Clone, Debug, Default)]
pub struct LinearScan;

impl RegisterAllocator for LinearScan {
    fn allocate_nest(&self, nest: ScopedSchedule, file: &RegisterFile) -> NestAllocation {
        // Outermost first, because that is the direction liveness flows: a
        // value a region computes for the scopes inside it is live across
        // every iteration of every loop between here and its last use. A
        // register holding it is therefore unavailable to all of them, which
        // is what allocating each region against the full pool used to miss.
        let mut carried = RegSet::EMPTY;
        // Roots of the regions already handled, and where each of them lives
        // for the whole of every scope inside: the register carrying it, or
        // its park slot. A scan inside reads this rather than choosing, which
        // is what lets it tell a resident operand from one it has to reload —
        // the question every reservation now turns on. It is also that scope's
        // whole answer for the value, since nothing inside may move it.
        let mut parked: BTreeMap<ValueId, Where> = BTreeMap::new();
        let mut regions: Vec<ScopeCode> = Vec::with_capacity(nest.regions.len());

        // Uses in the innermost body are what a carry actually saves: one
        // reload per use, per iteration. Counted once, up front.
        let mut body_uses: BTreeMap<ValueId, usize> = BTreeMap::new();
        for def in &nest.body {
            for operand in operands(&def.op) {
                *body_uses.entry(operand).or_insert(0) += 1;
            }
        }

        for (index, region) in nest.regions.into_iter().enumerate() {
            let scope = Scope::Region(index);
            let scoped = file.inside(carried);
            let scan = self.scan(region.schedule, &scoped, &parked);

            // Carry the roots the body reads most, while the body keeps a
            // pool it can still allocate in. Every carry costs one register
            // for the whole loop and saves one reload per use per iteration,
            // so the ordering is by use count and the cap is the floor.
            let budget = file
                .scratch
                .len()
                .saturating_sub(carried.len())
                .saturating_sub(RegisterFile::MIN_SCRATCH);
            let mut ranked: Vec<(usize, ValueId)> = region
                .roots
                .iter()
                .map(|v| (body_uses.get(v).copied().unwrap_or(0), *v))
                .filter(|(uses, _)| *uses > 0)
                .collect();
            // Descending by use count, then by id so the choice is
            // deterministic — two roots read the same number of times must
            // not depend on map iteration order.
            ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.0.cmp(&b.1.0)));

            // Everything the region's own code touches: placed values AND
            // per-instruction scratch. A temp is a pool register that no
            // placement records, so taking the complement of the locations
            // alone would hand out a register the prologue destroys.
            let mut used: Vec<Reg> = scan.registers().collect();
            for scratch in &scan.scratch {
                used.extend(scratch.temps.iter().flatten().copied());
                used.extend(scratch.reloads.iter().flatten().copied());
                used.extend(scratch.guard_mask);
                used.extend(scratch.guard_temp);
                used.extend(scratch.result);
            }
            let free = scoped.scratch.without(RegSet::of(&used));
            let mut available = free.iter();
            let mut carries: BTreeMap<ValueId, Reg> = BTreeMap::new();
            for (_, vid) in ranked.into_iter().take(budget as usize) {
                let Some(reg) = available.next() else { break };
                carried = carried.union(RegSet::of(&[reg]));
                carries.insert(vid, reg);
            }

            let placements = record(&scan, &parked);

            // A root outlives the region computing it, and where the scopes
            // inside find it is *their* answer, not another range on this
            // one's: either the register carrying it across the loops, or the
            // slot it was parked in. `parked` is how they are told, and
            // `record` turns it into their first span.
            for root in &region.roots {
                let at = match carries.get(root) {
                    Some(reg) => Where::Reg(*reg),
                    None => Where::Spilled,
                };
                assert!(
                    placements.get(root.0 as usize).is_some_and(Option::is_some),
                    "a region computes its own roots, but {root:?} is not in {scope:?}"
                );
                parked.insert(*root, at);
            }

            regions.push(ScopeCode {
                placements,
                schedule: scan.schedule,
                scratch: scan.scratch,
                roots: region.roots,
            });
        }

        let body = self.scan(nest.body, &file.inside(carried), &parked);

        NestAllocation {
            regions,
            body: ScopeCode {
                placements: record(&body, &parked),
                schedule: body.schedule,
                scratch: body.scratch,
                roots: Vec::new(),
            },
        }
    }
}

/// One scope's scan as placements.
///
/// A `parked` value's ranges are *replaced* rather than merged: its entry in
/// this schedule is a placeholder the emitter never emits, and the region that
/// computes it already said where this scope finds it — at the head, and for
/// the whole of it, since nothing here may move a value the loops outside are
/// holding. Everything else gets its own ranges, including a `Var`/`Const`
/// leaf an enclosing scope also computes, which is genuinely rebuilt here and
/// genuinely may land somewhere else.
fn record(scan: &Scan, parked: &BTreeMap<ValueId, Where>) -> Vec<Option<Placement>> {
    let mut placements: Vec<Option<Placement>> = alloc::vec![None; scan.ranges.len()];
    for (key, ranges) in scan.ranges.iter().enumerate() {
        if parked.contains_key(&ValueId(key as u32)) {
            continue;
        }
        for &(index, at) in ranges {
            let from = Point { index };
            let slot = &mut placements[key];
            *slot = Some(match slot.take() {
                // Consecutive ranges at the same place are one range: an
                // eviction that put a value back where it already was is not a
                // move, and a repeated span would break the strict increase.
                Some(prior) if prior.at(from) != at => prior.then(Span { from, at }),
                Some(prior) => prior,
                None => Placement::new(Span { from, at }),
            });
        }
    }
    for (&v, &at) in parked {
        if placements.len() <= v.0 as usize {
            placements.resize(v.0 as usize + 1, None);
        }
        placements[v.0 as usize] = Some(Placement::new(Span {
            from: Point::HEAD,
            at,
        }));
    }
    placements
}

/// One scope, scanned straight through: the ranges each value's life is cut
/// into here, and the scratch each instruction may destroy.
///
/// Ranges rather than one location, because eviction **splits**: a value keeps
/// the register it held up to the point it lost, and may come back into one at
/// a later read. [`record`] turns these into this scope's [`Placement`]s,
/// which is nearly a rename — the work it does is folding in what an enclosing
/// scope parked, since that is the one thing a scan of this scope alone cannot
/// know.
struct Scan {
    schedule: Vec<Def>,
    /// Dense by `ValueId.0`: this scope's ranges for that value, in strictly
    /// increasing schedule order. Empty for a value this scope does not place.
    ranges: Vec<Vec<(usize, Where)>>,
    scratch: Vec<Scratch>,
}

impl Scan {
    /// Every register any value occupies at any point of this scope.
    fn registers(&self) -> impl Iterator<Item = Reg> + use<'_> {
        self.ranges.iter().flatten().filter_map(|(_, at)| match at {
            Where::Reg(r) => Some(*r),
            Where::Spilled | Where::Remat(_) => None,
        })
    }
}

/// What giving up a register costs, cheapest first — the order eviction picks
/// its loser in.
///
/// A constant is recomputed and touches no memory at all; a value already in
/// its slot needs no store; anything else has to be written out. Belady's
/// distance breaks ties *within* a tier and only within one: the traffic an
/// eviction causes outweighs how long it waits to cause it.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct EvictionRank {
    /// Read by the instruction being placed — the one thing that outranks
    /// cheap traffic, and the reason an operand's register is a last resort
    /// rather than a forbidden one.
    ///
    /// The tiers below price the traffic an eviction *defers*; for a value read
    /// right here there is nothing to defer, so taking its register buys a
    /// reload inside this very instruction. Without this, a value already in
    /// its slot is the standing favourite — and at a read, the standing
    /// favourite is whichever value the instruction is reading.
    needed_now: bool,
    /// 0 = rematerialized, 1 = slot already valid, 2 = needs a store.
    traffic: u8,
    /// Nearest next read *last*, so the cheapest loser is the one used
    /// farthest out.
    nearest: core::cmp::Reverse<usize>,
}

/// The pool slots one instruction has already claimed, in the order it claimed
/// them.
///
/// Sized by the roles an instruction can fill at once — its encoding's temps,
/// its operand reloads, a guard's mask and scratch, and the scope's result —
/// which is the same list [`RegisterFile::MIN_SCRATCH`] is derived from.
struct Reservations {
    slots: [Option<usize>; Self::ROLES],
    filled: usize,
}

impl Reservations {
    const ROLES: usize = Scratch::MAX_TEMPS + Scratch::MAX_RELOADS + 3;

    fn new() -> Self {
        Self {
            slots: [None; Self::ROLES],
            filled: 0,
        }
    }

    fn push(&mut self, slot: usize) {
        self.slots[self.filled] = Some(slot);
        self.filled += 1;
    }

    fn holds(&self, slot: usize) -> bool {
        self.slots[..self.filled].contains(&Some(slot))
    }
}

/// The forward pass's state: who owns which pool register, where every value
/// is at the point reached, and what each life has been cut into so far.
///
/// One struct rather than nine locals threaded through five helpers — the
/// eviction rule reads four of them at once.
struct Pass {
    /// The pool, low to high. `owner` is indexed the same way.
    pool: Vec<Reg>,
    /// The value currently held in each pool register.
    owner: Vec<Option<ValueId>>,
    /// Where each value is at the point the pass has reached.
    at: Vec<Option<Where>>,
    /// The ranges settled so far, per value, in increasing index order.
    ranges: Vec<Vec<(usize, Where)>>,
    /// The `f32` bits of every value that is a constant — the ones that come
    /// back by being recomputed rather than reloaded.
    const_bits: Vec<Option<u32>>,
    /// Whether the value's slot already holds it, so losing a register again
    /// costs no store. True from the first range that puts it in memory,
    /// because a value in memory anywhere is stored right after its definition.
    in_slot: Vec<bool>,
    /// Where each value is defined, or `usize::MAX` for one this scope reads
    /// without computing.
    defined_at: Vec<usize>,
    /// Dense by `ValueId.0`: a value an enclosing scope computed and left
    /// somewhere fixed for the whole of this one.
    ///
    /// Its entry in this schedule is a placeholder that emits nothing, and its
    /// location is the enclosing scope's answer, not this scan's — so it takes
    /// no register here and its residency is read from the park rather than
    /// from the placeholder.
    live_in: Vec<bool>,
    /// Read positions per value, ascending, with a cursor that only advances —
    /// so the pass costs one step per read rather than a search per eviction.
    reads: Vec<Vec<usize>>,
    cursor: Vec<usize>,
}

impl Pass {
    fn new(
        dag: &[Def],
        file: &RegisterFile,
        vec_len: usize,
        live_in: &BTreeMap<ValueId, Where>,
    ) -> Self {
        let mut reads: Vec<Vec<usize>> = vec![Vec::new(); vec_len];
        let mut const_bits: Vec<Option<u32>> = vec![None; vec_len];
        let mut defined_at: Vec<usize> = vec![usize::MAX; vec_len];
        for (i, def) in dag.iter().enumerate() {
            defined_at[def.value.0 as usize] = i;
            if let ScheduledOp::Const(val) = def.op {
                const_bits[def.value.0 as usize] = Some(val.to_bits());
            }
            for operand in operands(&def.op) {
                let r = &mut reads[operand.0 as usize];
                if r.last() != Some(&i) {
                    r.push(i);
                }
            }
        }
        let mut at: Vec<Option<Where>> = vec![None; vec_len];
        let mut is_live_in = vec![false; vec_len];
        for (v, park) in live_in {
            let k = v.0 as usize;
            if k >= vec_len {
                continue; // Parked by an enclosing scope; not read here.
            }
            // The enclosing scope's answer, from the first point of this one.
            // Its placeholder is neither a definition (it emits nothing) nor a
            // constant (its op says `Const(0.0)`, which is not the value).
            at[k] = Some(*park);
            const_bits[k] = None;
            is_live_in[k] = true;
        }
        Self {
            pool: file.scratch().collect(),
            owner: vec![None; file.scratch.len() as usize],
            at,
            ranges: vec![Vec::new(); vec_len],
            const_bits,
            in_slot: vec![false; vec_len],
            defined_at,
            live_in: is_live_in,
            reads,
            cursor: vec![0; vec_len],
        }
    }

    /// Whether `v` is in a register at the point this pass has reached.
    fn is_resident(&self, v: ValueId) -> bool {
        matches!(self.at[v.0 as usize], Some(Where::Reg(_)))
    }

    /// Claim one more pool register for this instruction's own use.
    ///
    /// Disjoint from every register the instruction reads (`live`, its
    /// operands and its guards' masks), from every role it has already filled,
    /// and — because the destination is claimed last, against the same two
    /// exclusions — from the destination.
    fn reserve(&mut self, index: usize, taken: &mut Reservations, live: &[ValueId]) -> Reg {
        let open = self.without_operands(taken, live);
        let slot = self.claim(index, &open);
        taken.push(slot);
        self.pool[slot]
    }

    /// The next read of `v` at or after `from`.
    fn next_read(&mut self, v: ValueId, from: usize) -> Option<usize> {
        let k = v.0 as usize;
        while self.cursor[k] < self.reads[k].len() && self.reads[k][self.cursor[k]] < from {
            self.cursor[k] += 1;
        }
        self.reads[k].get(self.cursor[k]).copied()
    }

    /// What evicting `v` at `from` would cost. See [`EvictionRank`].
    fn rank(&mut self, v: ValueId, from: usize) -> EvictionRank {
        let k = v.0 as usize;
        let traffic = if self.const_bits[k].is_some() {
            0
        } else if self.in_slot[k] {
            1
        } else {
            2
        };
        let distance = self.next_read(v, from).map_or(usize::MAX, |r| r - from);
        EvictionRank {
            needed_now: distance == 0,
            traffic,
            nearest: core::cmp::Reverse(distance),
        }
    }

    /// Record that `v` lives at `to` from `index` on.
    fn place(&mut self, v: ValueId, index: usize, to: Where) {
        let k = v.0 as usize;
        match self.ranges[k].last_mut() {
            Some(last) if last.0 == index => last.1 = to,
            _ => self.ranges[k].push((index, to)),
        }
        self.at[k] = Some(to);
        if to == Where::Spilled {
            self.in_slot[k] = true;
        }
    }

    /// Where `v` goes when it loses its register: nowhere at all if it is a
    /// constant, and its slot otherwise.
    fn out_of_register(&self, v: ValueId) -> Where {
        match self.const_bits[v.0 as usize] {
            Some(bits) => Where::Remat(bits),
            None => Where::Spilled,
        }
    }

    /// Hand `slot` to something else at `index`, splitting whatever held it:
    /// the earlier range stands, and a new one starts here.
    fn split_out(&mut self, slot: usize, index: usize) {
        if let Some(loser) = self.owner[slot].take() {
            let to = self.out_of_register(loser);
            self.place(loser, index, to);
        }
    }

    /// Put `v` in pool slot `slot` from `index` on.
    fn occupy(&mut self, v: ValueId, slot: usize, index: usize) {
        self.owner[slot] = Some(v);
        self.place(v, index, Where::Reg(self.pool[slot]));
    }

    /// Free every register whose owner is not read again.
    fn expire(&mut self, index: usize) {
        for slot in 0..self.owner.len() {
            if let Some(v) = self.owner[slot]
                && self.next_read(v, index).is_none()
            {
                self.owner[slot] = None;
            }
        }
    }

    /// Pool slots this instruction may still draw on: every one it has not
    /// already claimed for a scratch role.
    ///
    /// An operand's register is in here and does not need excluding. The
    /// encoder reads every source before it writes the destination, so a
    /// destination that takes an operand's register only makes that operand
    /// non-resident *here*, and `resolve_operands` reads it back from the slot
    /// its definition wrote. [`EvictionRank`] is what keeps that a last resort.
    fn open(&self, taken: &Reservations) -> Vec<usize> {
        (0..self.owner.len()).filter(|k| !taken.holds(*k)).collect()
    }

    /// Pool slots an instruction may destroy *before* reading its operands —
    /// which is what scratch is, and what a kept reload must not displace,
    /// both being wanted in a register at this same point. Sharing one with an
    /// operand would feed the instruction its own temp.
    fn without_operands(&self, taken: &Reservations, reads: &[ValueId]) -> Vec<usize> {
        self.open(taken)
            .into_iter()
            .filter(|k| self.owner[*k].is_none_or(|v| !reads.contains(&v)))
            .collect()
    }

    /// The slot to give up at `index`, among `open` ones that hold something.
    fn loser(&mut self, open: &[usize], index: usize) -> Option<usize> {
        let held: Vec<(usize, ValueId)> = open
            .iter()
            .filter_map(|k| self.owner[*k].map(|v| (*k, v)))
            .collect();
        held.into_iter()
            .min_by_key(|(_, v)| self.rank(*v, index))
            .map(|(k, _)| k)
    }

    /// A pool register for something defined or reloaded at `index`: a free one
    /// if there is one, and otherwise the one whose occupant is cheapest to
    /// evict — which splits that occupant's live range here.
    ///
    /// Always answers. The floor [`RegisterFile::MIN_SCRATCH`] is what makes
    /// that true: the widest demand is a gather's one operand plus four temps,
    /// or a ternary's three operands plus a temp and an arm reload — five
    /// exclusions either way, against a pool of at least six.
    fn claim(&mut self, index: usize, open: &[usize]) -> usize {
        if let Some(free) = open.iter().copied().find(|k| self.owner[*k].is_none()) {
            return free;
        }
        let slot = self.loser(open, index).unwrap_or_else(|| {
            unreachable!(
                "every pool register is this instruction's own scratch or one of \
                 its operands, against a floor of {}",
                RegisterFile::MIN_SCRATCH
            )
        });
        self.split_out(slot, index);
        slot
    }
}

/// For each schedule index, the narrowest `Select` arm containing it.
///
/// The narrowest and not the outermost: ending a kept reload at the inner arm's
/// end is safe under the outer one too, since every read between the two ends
/// is inside the outer arm and so is skipped along with the load it would name.
fn guarded_arms(guards: &[SelectGuard], len: usize) -> Vec<Option<(usize, usize)>> {
    let mut arms: Vec<Option<(usize, usize)>> = vec![None; len];
    for guard in guards {
        for (start, end) in guard.ranges.values().copied() {
            if start == end {
                continue;
            }
            for arm in &mut arms[start..end] {
                if arm.is_none_or(|(s, e)| end - start < e - s) {
                    *arm = Some((start, end));
                }
            }
        }
    }
    arms
}

/// For each schedule index, the masks a short-circuit branch reads *there*.
///
/// A guard is emitted before the first instruction of each non-empty arm, and
/// again at the `Select` itself for the uniform-mask wrapper. Those are the
/// only points that need a mask in a register outside an instruction's own
/// operands, and they are the points the allocator reserves
/// [`Scratch::guard_mask`] and [`Scratch::guard_temp`] on.
///
/// Several guards can begin at one index (nested `Select`s); one reservation
/// covers them all, because each resolves its mask and branches before the
/// next one runs.
fn guard_sites(guards: &[SelectGuard], len: usize) -> Vec<Vec<ValueId>> {
    let mut sites: Vec<Vec<ValueId>> = (0..len).map(|_| Vec::new()).collect();
    for guard in guards {
        let mut at = |i: usize| {
            let site = &mut sites[i];
            if !site.contains(&guard.mask_vid) {
                site.push(guard.mask_vid);
            }
        };
        let mut guarded = false;
        for (start, end) in guard.ranges.values().copied() {
            if start == end {
                continue;
            }
            guarded = true;
            at(start);
        }
        if guarded {
            at(guard.select_idx);
        }
    }
    sites
}

impl LinearScan {
    /// One region, scanned straight through.
    ///
    /// `live_in` is where each value an enclosing scope parked lives for the
    /// whole of this one — the answer this scan must read rather than choose,
    /// because that scope already chose it.
    fn scan(&self, dag: Vec<Def>, file: &RegisterFile, live_in: &BTreeMap<ValueId, Where>) -> Scan {
        let vec_len = dag
            .iter()
            .map(|def| def.value.0 as usize + 1)
            .max()
            .unwrap_or(0);
        let mut scratch_for: Vec<Scratch> = vec![Scratch::default(); dag.len()];

        if dag.is_empty() {
            return Scan {
                schedule: dag,
                ranges: Vec::new(),
                scratch: scratch_for,
            };
        }

        let mut pass = Pass::new(&dag, file, vec_len, live_in);

        // The arms a `Select` guard may skip. A register range that begins at a
        // read inside one, for a value defined outside it, must end there too:
        // after the arm a read has to name what it named before, because the
        // skipped path never ran the load. Eviction inside an arm needs no such
        // rule — the value's slot was written at its definition, which every
        // path reaching any of its readers ran.
        let guards = analyze_select_guards(&dag);
        let arms = guarded_arms(&guards, dag.len());
        let sites = guard_sites(&guards, dag.len());
        let mut reverts: Vec<Vec<(ValueId, Where, usize)>> =
            (0..dag.len()).map(|_| Vec::new()).collect();
        // Pool slots a definition held for its own instruction and no longer:
        // see the destination below. Indexed by the point the range ends at.
        let mut demotions: Vec<Vec<(ValueId, usize)>> =
            (0..dag.len()).map(|_| Vec::new()).collect();

        for (i, def) in dag.iter().enumerate() {
            for (v, slot) in core::mem::take(&mut demotions[i]) {
                if pass.owner[slot] == Some(v) {
                    pass.owner[slot] = None;
                    let to = pass.out_of_register(v);
                    pass.place(v, i, to);
                }
            }
            for (v, back, slot) in core::mem::take(&mut reverts[i]) {
                if pass.owner[slot] == Some(v) {
                    pass.owner[slot] = None;
                    pass.place(v, i, back);
                }
            }
            pass.expire(i);

            let mut reads: Vec<ValueId> = Vec::new();
            for operand in operands(&def.op) {
                if !reads.contains(&operand) {
                    reads.push(operand);
                }
            }
            // What this instruction reads, its guards included. A guard runs
            // *before* the instruction and reads a mask that is nobody's
            // operand there, so without this a temp could take the register
            // the branch is about to test.
            let mut live_here = reads.clone();
            for mask in &sites[i] {
                if !live_here.contains(mask) {
                    live_here.push(*mask);
                }
            }

            // Scratch, reserved before anything else this instruction wants:
            // the encoder writes it while every operand is still live and
            // before the destination is written, so it may share a register
            // with neither.
            let mut taken = Reservations::new();

            let wanted = (file.temps_for)(&def.op) as usize;
            assert!(
                wanted <= Scratch::MAX_TEMPS,
                "a backend asked for {wanted} scratch registers for one \
                 instruction; `Scratch::MAX_TEMPS` is {}",
                Scratch::MAX_TEMPS
            );
            for role in 0..wanted {
                scratch_for[i].temps[role] = Some(pass.reserve(i, &mut taken, &live_here));
            }

            // GPR- and mask-class scratch, reserved the same way but against
            // their own pools: nothing else in the schedule ever asks for a
            // GPR or a mask register, so there is no interference to track and
            // no eviction to perform — each instruction simply takes the low
            // members of the class pool it needs.
            let gpr_wanted = (file.gpr_temps_for)(&def.op) as usize;
            assert!(
                gpr_wanted <= Scratch::MAX_GPR_TEMPS,
                "a backend asked for {gpr_wanted} GPR scratch registers for \
                 one instruction; `Scratch::MAX_GPR_TEMPS` is {}",
                Scratch::MAX_GPR_TEMPS
            );
            assert!(
                file.gpr_scratch.len() as usize >= gpr_wanted,
                "{:?} needs {gpr_wanted} GPRs but `RegisterFile::gpr_scratch` \
                 holds only {}",
                def.op,
                file.gpr_scratch.len()
            );
            for (role, reg) in file.gpr_scratch.iter().take(gpr_wanted).enumerate() {
                scratch_for[i].gpr_temps[role] = Some(reg);
            }

            let mask_wanted = (file.mask_temps_for)(&def.op) as usize;
            assert!(
                mask_wanted <= Scratch::MAX_MASK_TEMPS,
                "a backend asked for {mask_wanted} mask scratch registers for \
                 one instruction; `Scratch::MAX_MASK_TEMPS` is {}",
                Scratch::MAX_MASK_TEMPS
            );
            assert!(
                file.mask_scratch.len() as usize >= mask_wanted,
                "{:?} needs {mask_wanted} mask registers but \
                 `RegisterFile::mask_scratch` holds only {}",
                def.op,
                file.mask_scratch.len()
            );
            for (role, reg) in file.mask_scratch.iter().take(mask_wanted).enumerate() {
                scratch_for[i].mask_temps[role] = Some(reg);
            }

            // A read of a value that is not in a register: bring it back into
            // one and *keep* it there, when it is read again before the keeping
            // has to stop. That is what splitting buys — a value spends the
            // pressured stretch in memory and the rest in a register, instead
            // of one or the other for the whole of its life.
            for operand in reads.clone() {
                // Only a value whose return costs memory traffic is worth a
                // register. A constant lives nowhere and is rebuilt in one
                // instruction, which is the same instruction a reload would
                // be — and eviction ranks constants cheapest to give up, so
                // keeping one buys a register the very next definition takes
                // back. The two rules would otherwise fight, and a constant
                // would spend the kernel bouncing in and out of the pool.
                if !matches!(pass.at[operand.0 as usize], Some(Where::Spilled)) {
                    continue;
                }
                // A parked root's location belongs to the scope that computed
                // it; keeping it here would be this scan recording a range for
                // a value it does not place.
                if pass.live_in[operand.0 as usize] {
                    continue;
                }
                let open = pass.without_operands(&taken, &live_here);
                if open.is_empty() {
                    break;
                }
                let stop = arms[i].and_then(|(start, end)| {
                    (pass.defined_at[operand.0 as usize] < start).then_some(end)
                });
                let Some(next) = pass.next_read(operand, i + 1) else {
                    continue; // Read once more and then done: a scratch will do.
                };
                if stop.is_some_and(|end| next >= end) {
                    continue; // The range would end before the read it is for.
                }
                let back = pass.at[operand.0 as usize]
                    .unwrap_or_else(|| unreachable!("resident values were skipped"));
                let slot = pass.claim(i, &open);
                pass.occupy(operand, slot, i);
                if let Some(end) = stop
                    && end < dag.len()
                {
                    reverts[end].push((operand, back, slot));
                }
            }

            // A guard's own two registers, on the instruction it is emitted
            // before. The mask needs one only when it is not in a register
            // here — which the kept reloads above may just have changed.
            if !sites[i].is_empty() {
                if sites[i].iter().any(|m| !pass.is_resident(*m)) {
                    scratch_for[i].guard_mask = Some(pass.reserve(i, &mut taken, &live_here));
                }
                for _ in 0..file.guard_temps {
                    scratch_for[i].guard_temp = Some(pass.reserve(i, &mut taken, &live_here));
                }
                // The mask-class mirror: AVX-512's guard reduces the mask
                // with `vptestmd` into a `k`-register the vector pool cannot
                // see, so this comes from `mask_scratch` rather than `pass`.
                for reg in file
                    .mask_scratch
                    .iter()
                    .take(file.mask_guard_temps as usize)
                {
                    scratch_for[i].mask_guard_temp = Some(reg);
                }
            }

            // One register per operand this instruction has to reload, named
            // by the same function the emitter reads. Residency is final here:
            // eviction splits a range rather than rewriting one, so an operand
            // in a register now is in a register when this instruction is
            // emitted, and the destination below cannot take it back.
            let mut resident = [true; 3];
            for (k, operand) in operands(&def.op).enumerate() {
                resident[k] = pass.is_resident(operand);
            }
            let sources = operand_sources(&def.op, resident);
            for role in 0..reloads_wanted(sources) {
                scratch_for[i].reloads[role] = Some(pass.reserve(i, &mut taken, &live_here));
            }

            // The scope's result is materialized after its last instruction,
            // and it needs a register of its own in exactly one case: the
            // whole body was hoisted out, so its root is read from a park
            // rather than computed. Every other root is the last
            // instruction's own destination, which is a register.
            if i + 1 == dag.len()
                && pass.live_in[def.value.0 as usize]
                && !pass.is_resident(def.value)
            {
                scratch_for[i].result = Some(pass.reserve(i, &mut taken, &live_here));
            }

            // A placeholder for a value an enclosing scope parked: it emits
            // nothing, so it writes no register and gets no range here.
            if pass.live_in[def.value.0 as usize] {
                continue;
            }

            // Coordinate inputs are pinned to the registers the ABI delivers
            // them in. The scratch pool excludes those registers, so a pinned
            // value never competes for one.
            if let ScheduledOp::Var(k) = def.op {
                pass.place(def.value, i, Where::Reg(input_register(file, k)));
                continue;
            }

            // **The destination always gets a register.** A definition is the
            // one write in an instruction, and there is no register outside the
            // pool left to write to — so the loser of the contest below is the
            // *occupant*, and the value being defined loses only the right to
            // *keep* what it was given. That is `Where(v, def(v)) == Reg(_)`,
            // which is what dissolves the fixed destination register.
            //
            // A rematerialized constant is the exception, and it is not one in
            // spirit: its definition emits nothing at all, so it needs nothing
            // to write to.
            let open = pass.without_operands(&taken, &live_here);
            if let Some(free) = open.iter().copied().find(|k| pass.owner[*k].is_none()) {
                pass.occupy(def.value, free, i);
                continue;
            }
            let slot = pass.loser(&open, i).unwrap_or_else(|| {
                unreachable!(
                    "every pool register is this instruction's own scratch or one of \
                     its operands, against a floor of {}",
                    RegisterFile::MIN_SCRATCH
                )
            });
            let occupant = pass.owner[slot].unwrap_or_else(|| unreachable!("a loser holds one"));
            // Whether the new value keeps the register past this instruction,
            // by the rule that chose the slot: its own rank against the
            // occupant's. A definition has written nothing yet, so its slot is
            // never the cheap kind.
            let new_rank = EvictionRank {
                // A definition is a write; nothing reads it here.
                needed_now: false,
                traffic: if pass.const_bits[def.value.0 as usize].is_some() {
                    0
                } else {
                    2
                },
                nearest: core::cmp::Reverse(
                    pass.next_read(def.value, i).map_or(usize::MAX, |r| r - i),
                ),
            };
            let keeps = new_rank > pass.rank(occupant, i);
            if !keeps && pass.const_bits[def.value.0 as usize].is_some() {
                // Nothing to write: the definition of a rematerialized constant
                // emits no instruction, so it takes no register and evicts no
                // one. Reserving one for it would cost a live value its
                // register to hold a value the emitter never computes.
                pass.place(def.value, i, pass.out_of_register(def.value));
                continue;
            }
            pass.split_out(slot, i);
            pass.occupy(def.value, slot, i);
            if !keeps && i + 1 < dag.len() {
                // Given a register to be written into and stored from — the
                // store goes right after the definition, as it does for any
                // value with a slot — and not to keep.
                demotions[i + 1].push((def.value, slot));
            }
        }

        Scan {
            schedule: dag,
            ranges: pass.ranges,
            scratch: scratch_for,
        }
    }
}

/// The register a `Var(i)` is pinned to.
///
/// A `Var` past the coordinate inputs is not a kernel the target cannot run;
/// it is a `Var` that should have been lowered away (a reduce binder, a
/// manifold-param slot) or an axis that does not exist. Both are bugs upstream
/// of here, and neither is something a caller could act on — so it fails at
/// the point of failure, with a stack trace, rather than as a string three
/// frames up.
#[cold]
#[inline(never)]
fn var_out_of_range(i: u8, inputs: usize) -> ! {
    panic!(
        "Var({i}) names coordinate {i} but the target supplies {inputs} \
         (X, Y, Z, W) — `passes::legalize` lowers every other Var, so this is \
         a bypassed pipeline or a missing lowering, not a bad kernel"
    )
}

fn input_register(file: &RegisterFile, i: u8) -> Reg {
    match file.inputs.get(i as usize) {
        Some(&reg) => reg,
        None => var_out_of_range(i, file.inputs.len()),
    }
}

/// A backend whose encodings never need a register beyond their operands.
///
/// The default for [`RegisterFile::temps_for`]; naming it keeps the field
/// total, so a new backend states its answer rather than inheriting one.
#[must_use]
pub fn no_temps(_op: &ScheduledOp) -> u8 {
    0
}

/// The values an operation reads, in operand order.
pub(crate) fn operands(sop: &ScheduledOp) -> impl Iterator<Item = ValueId> + use<'_> {
    let (a, b, c) = match sop {
        ScheduledOp::Var(_) | ScheduledOp::Const(_) | ScheduledOp::Uniform(_) => (None, None, None),
        ScheduledOp::Unary(_, a) | ScheduledOp::ShiftImm(_, a, _) | ScheduledOp::Gather(a, _) => {
            (Some(*a), None, None)
        }
        ScheduledOp::Binary(_, a, b) => (Some(*a), Some(*b), None),
        ScheduledOp::Ternary(_, a, b, c) => (Some(*a), Some(*b), Some(*c)),
    };
    [a, b, c].into_iter().flatten()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::kind::OpKind;

    /// The smallest pool a register file may declare, so pressure tests need
    /// only a handful of values to reach spilling.
    const TEST_FILE: RegisterFile = RegisterFile {
        fixed: &[],
        inputs: [Reg(0), Reg(1), Reg(2), Reg(3)],
        scratch: RegSet::range(4, RegisterFile::MIN_SCRATCH),
        temps_for: no_temps,
        guard_temps: 0,
        vector_bytes: 16,
        gpr_ctx: None,
        gpr_scratch: GprSet::EMPTY,
        gpr_temps_for: no_temps,
        mask_scratch: MaskSet::EMPTY,
        mask_temps_for: no_temps,
        mask_guard_temps: 0,
    }
    .checked();

    /// `TEST_FILE`, but every `Neg` asks for a temp — the AVX2 sign-mask case.
    const TEMP_FILE: RegisterFile = RegisterFile {
        temps_for: neg_wants_a_temp,
        ..TEST_FILE
    }
    .checked();

    /// `TEMP_FILE` with headroom above [`RegisterFile::MIN_SCRATCH`].
    ///
    /// The nest tests need a pool that can spare a register to carry, and the
    /// minimum-sized file by construction cannot: the carry budget is
    /// `pool - MIN_SCRATCH`, which is zero there. Ten registers mirrors the
    /// SSE2 tier, whose budget is four.
    const NEST_FILE: RegisterFile = RegisterFile {
        scratch: RegSet::range(4, 7).union(RegSet::of(&[Reg(13), Reg(14), Reg(15)])),
        ..TEMP_FILE
    }
    .checked();

    fn neg_wants_a_temp(op: &ScheduledOp) -> u8 {
        u8::from(matches!(op, ScheduledOp::Unary(OpKind::Neg, _)))
    }

    /// A point in the innermost body — the only scope a loop-free
    /// allocation has.
    fn body(index: usize) -> Point {
        Point { index }
    }

    fn def(value: u32, op: ScheduledOp) -> Def {
        Def {
            value: ValueId(value),
            op,
        }
    }

    fn alloc(schedule: Vec<Def>) -> NestAllocation {
        LinearScan.allocate(schedule, &TEST_FILE)
    }

    /// How many of a loop-free allocation's values are in a stack slot.
    fn spill_count(a: &NestAllocation) -> usize {
        a.body()
            .schedule()
            .iter()
            .filter(|d| a.body().placement(d.value).spills())
            .count()
    }

    /// Where `v` lives at its own definition, in `scope`.
    ///
    /// A placement is a schedule, so every query needs a point; the point a
    /// test means when it asks "where did the allocator put this" is the
    /// definition, and finding it is the schedule's job rather than each
    /// assertion's.
    fn at_def(a: &Allocation<'_>, v: ValueId) -> Where {
        let i = a
            .schedule()
            .iter()
            .position(|d| d.value == v)
            .unwrap_or_else(|| panic!("{v:?} is not in this schedule"));
        a.where_at(v, i)
    }

    /// `at_def` against a loop-free allocation, whose one scope is the body.
    fn at(a: &NestAllocation, v: ValueId) -> Where {
        at_def(&a.body(), v)
    }

    /// Every place `v` occupies over its whole life.
    ///
    /// The question to ask once eviction splits a range rather than condemning
    /// a life: `at` answers where a value *starts*, which stopped being the
    /// same as where it spends its time.
    fn ever(a: &NestAllocation, v: ValueId) -> Vec<Where> {
        a.body().placement(v).locations().collect()
    }

    /// `v2 = X + Y`.
    fn add_xy() -> Vec<Def> {
        vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Binary(OpKind::Add, ValueId(0), ValueId(1))),
        ]
    }

    #[test]
    fn an_empty_schedule_allocates_nothing() {
        let a = alloc(vec![]);
        assert!(a.body().schedule().is_empty());
        assert_eq!(spill_count(&a), 0);
    }

    /// The schedule is an output, and for linear scan it is the input order:
    /// the arena's append-only structure already guarantees topological order.
    #[test]
    fn linear_scan_returns_the_order_it_was_given() {
        let a = alloc(add_xy());
        let order: Vec<ValueId> = a.body().schedule().iter().map(|d| d.value).collect();
        assert_eq!(order, vec![ValueId(0), ValueId(1), ValueId(2)]);
    }

    #[test]
    fn vars_are_pinned_to_the_input_registers() {
        let a = alloc(add_xy());
        assert_eq!(at(&a, ValueId(0)), Where::Reg(Reg(0)), "X");
        assert_eq!(at(&a, ValueId(1)), Where::Reg(Reg(1)), "Y");
        // The sum takes the pool, never an input register.
        assert_eq!(at(&a, ValueId(2)), Where::Reg(Reg(4)));
        assert_eq!(spill_count(&a), 0);
    }

    /// Every value the emitter walks has a placement — the invariant that used
    /// to be spread across three parallel maps and checked at runtime.
    #[test]
    fn placement_is_total_over_the_schedule() {
        let a = alloc(add_xy());
        let body = a.body();
        let placed: Vec<Where> = body.schedule().iter().map(|d| at(&a, d.value)).collect();
        assert_eq!(placed.len(), body.schedule().len());
    }

    // -------------------------------------------------------------------------
    // The placement sequence itself
    // -------------------------------------------------------------------------

    /// A one-range placement answers the same thing at every point — which is
    /// what makes today's allocator's output a special case of the sequence
    /// rather than a different kind of answer.
    #[test]
    fn a_single_range_answers_everywhere() {
        let p = Placement::new(Span {
            from: body(0),
            at: Where::Reg(Reg(7)),
        });
        for i in [0usize, 1, 99] {
            assert_eq!(p.at(body(i)), Where::Reg(Reg(7)));
        }
        assert!(!p.spills());
        assert_eq!(p.registers().collect::<Vec<_>>(), vec![Reg(7)]);
    }

    /// The case the whole type exists for: a value in a register up to a
    /// point and in a slot after it. Nothing emits this yet — the policy that
    /// will is the work this API unblocks — so it is checked here directly.
    #[test]
    fn a_ranged_placement_answers_per_point() {
        let p = Placement::new(Span {
            from: body(3),
            at: Where::Reg(Reg(5)),
        })
        .then(Span {
            from: body(9),
            at: Where::Spilled,
        });

        assert_eq!(p.at(body(3)), Where::Reg(Reg(5)), "the first range starts");
        assert_eq!(p.at(body(8)), Where::Reg(Reg(5)), "still the first range");
        assert_eq!(p.at(body(9)), Where::Spilled, "`from` is inclusive");
        assert_eq!(p.at(body(400)), Where::Spilled, "the last range runs on");

        // Total below the definition too: an answer, not a panic.
        assert_eq!(p.at(body(0)), Where::Reg(Reg(5)));
        assert_eq!(p.defined_at(), body(3));

        // A value in a slot for *part* of its life still needs a slot, and the
        // register it held earlier is still a register something wrote.
        assert!(p.spills());
        assert_eq!(p.registers().collect::<Vec<_>>(), vec![Reg(5)]);
    }

    #[test]
    #[should_panic(expected = "names coordinate 4")]
    fn a_var_past_the_input_registers_panics() {
        let _ = alloc(vec![def(0, ScheduledOp::Var(4))]);
    }

    /// Two values whose live ranges do not overlap may share one register —
    /// that is the whole point of tracking last use rather than defs.
    #[test]
    fn disjoint_live_ranges_share_a_register() {
        let a = alloc(vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Binary(OpKind::Add, ValueId(0), ValueId(1))),
            def(3, ScheduledOp::Unary(OpKind::Neg, ValueId(2))),
            def(4, ScheduledOp::Unary(OpKind::Abs, ValueId(3))),
        ]);
        assert_eq!(spill_count(&a), 0);
        // v2's last use is v3, so v4 may reuse v2's register.
        assert_eq!(at(&a, ValueId(2)), at(&a, ValueId(4)));
    }

    // -------------------------------------------------------------------------
    // Instruction temps
    // -------------------------------------------------------------------------

    /// The temp is a real register from the pool, and it is nobody's operand
    /// and not the destination — the encoder writes it while the operands are
    /// still live and before it writes `dst`.
    #[test]
    fn a_temp_collides_with_neither_the_operand_nor_the_destination() {
        let schedule = vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Unary(OpKind::Neg, ValueId(0))),
        ];
        let a = LinearScan.allocate(schedule, &TEMP_FILE);

        let temp = a.body().scratch(1).temp(0).expect("`Neg` asked for a temp");
        assert!(
            TEMP_FILE.scratch.contains(temp),
            "{temp:?} is not the pool's"
        );
        assert_ne!(Where::Reg(temp), at(&a, ValueId(0)));
        assert_ne!(Where::Reg(temp), at(&a, ValueId(1)));
    }

    /// A `Select` with spilled arms gets a reload target each, disjoint from
    /// its operands, its destination and its encoding temp — the registers the
    /// instruction is using at once.
    ///
    /// The second of them used to be `select_reload`, then `arm_reload`; it is
    /// operand 2's entry in `Scratch::reload` now, chosen by the same
    /// `operand_sources` the emitter reads.
    #[test]
    fn a_select_reserves_a_target_for_each_arm_it_has_to_reload() {
        // The mask and both arms are computed first and read last, with enough
        // filler between them to push them out of a pool sized at the floor —
        // which is what makes this a test about reload targets rather than
        // about a `Select` whose operands all happen to be resident.
        let width = u32::from(RegisterFile::MIN_SCRATCH) + 1;
        let mut schedule = vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Binary(OpKind::Lt, ValueId(0), ValueId(1))),
            def(3, ScheduledOp::Binary(OpKind::Add, ValueId(0), ValueId(1))),
            def(4, ScheduledOp::Binary(OpKind::Sub, ValueId(0), ValueId(1))),
        ];
        for i in 0..width {
            schedule.push(def(
                10 + i,
                ScheduledOp::Binary(OpKind::Mul, ValueId(0), ValueId(1)),
            ));
        }
        let mut acc = ValueId(10);
        for i in 1..width {
            schedule.push(def(
                100 + i,
                ScheduledOp::Binary(OpKind::Add, acc, ValueId(10 + i)),
            ));
            acc = ValueId(100 + i);
        }
        let select = 200;
        schedule.push(def(
            select,
            ScheduledOp::Ternary(OpKind::Select, ValueId(2), ValueId(3), ValueId(4)),
        ));
        let at_select = schedule.len() - 1;
        let a = LinearScan.allocate(schedule, &TEMP_FILE);
        let s = a.body().scratch(at_select);

        // The reservation answers the question it is for: how many of this
        // instruction's operands are not in a register where it runs.
        let resident = [
            matches!(a.body().where_at(ValueId(2), at_select), Where::Reg(_)),
            matches!(a.body().where_at(ValueId(3), at_select), Where::Reg(_)),
            matches!(a.body().where_at(ValueId(4), at_select), Where::Reg(_)),
        ];
        let op = ScheduledOp::Ternary(OpKind::Select, ValueId(2), ValueId(3), ValueId(4));
        let wanted = reloads_wanted(operand_sources(&op, resident));
        assert!(wanted > 0, "no arm spilled, so nothing here is reserved");
        for role in 0..wanted {
            let arm = s
                .reload(role)
                .unwrap_or_else(|| panic!("reload target {role} was not reserved"));
            assert!(TEMP_FILE.scratch.contains(arm), "{arm:?} is not the pool's");
            for v in [ValueId(2), ValueId(3), ValueId(4), ValueId(select)] {
                assert_ne!(
                    Where::Reg(arm),
                    a.body().where_at(v, at_select),
                    "{arm:?} is still holding {v:?} when the Select reloads into it"
                );
            }
            assert_ne!(Some(arm), s.temp(0), "the two roles must be two registers");
            for other in 0..role {
                assert_ne!(Some(arm), s.reload(other), "two operands, one register");
            }
        }
        assert_eq!(s.reload(wanted), None, "nothing reserved past the demand");
    }

    /// Nothing is reserved for an operand that is already in a register.
    #[test]
    fn a_resident_operand_reserves_no_reload_target() {
        let a = LinearScan.allocate(add_xy(), &TEMP_FILE);
        let s = a.body().scratch(2);
        assert_eq!(s.reload(0), None);
        assert_eq!(s.reload(1), None);
    }

    /// A pool shrunk below what an encoding needs is not a smaller budget, it
    /// is an instruction with nowhere to put its temp — so `capped` holds the
    /// floor rather than letting `max_regs` reach through it.
    #[test]
    fn capping_the_pool_stops_at_the_floor() {
        let tiny = TEMP_FILE.capped(Some(1));
        assert_eq!(tiny.scratch.len(), RegisterFile::MIN_SCRATCH);

        // The shape that used to fall through: a `Neg` whose operand is a
        // computed value, so the operand is pool-resident rather than
        // pre-colored into an input register.
        let a = LinearScan.allocate(
            vec![
                def(0, ScheduledOp::Var(0)),
                def(1, ScheduledOp::Var(1)),
                def(2, ScheduledOp::Binary(OpKind::Add, ValueId(0), ValueId(1))),
                def(3, ScheduledOp::Unary(OpKind::Neg, ValueId(2))),
            ],
            &tiny,
        );
        let temp = a.body().scratch(3).temp(0).expect("`Neg` asked for a temp");
        assert_ne!(Where::Reg(temp), at(&a, ValueId(2)));
    }

    /// Only the ops that asked get one — the register is the allocator's
    /// everywhere else, which is the whole reason for asking per-op.
    #[test]
    fn an_op_that_asks_for_no_temp_gets_none() {
        let a = LinearScan.allocate(add_xy(), &TEMP_FILE);
        assert_eq!(a.body().scratch(2).temp(0), None, "`Add` asked for no temp");
    }

    /// The temp's live range is one instruction: the next one may take the
    /// same register for its result.
    #[test]
    fn a_temp_is_free_again_at_the_next_instruction() {
        let schedule = vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Unary(OpKind::Neg, ValueId(0))),
            def(2, ScheduledOp::Unary(OpKind::Sqrt, ValueId(1))),
            def(3, ScheduledOp::Unary(OpKind::Sqrt, ValueId(2))),
            def(4, ScheduledOp::Unary(OpKind::Sqrt, ValueId(3))),
        ];
        let a = LinearScan.allocate(schedule, &TEMP_FILE);
        assert_eq!(spill_count(&a), 0, "four values fit a four-register pool");

        let temp = a.body().scratch(1).temp(0).expect("`Neg` asked for a temp");
        let reused = a
            .body()
            .schedule()
            .iter()
            .any(|d| at(&a, d.value) == Where::Reg(temp));
        assert!(reused, "{temp:?} went back to the pool after the `Neg`");
    }

    /// More values live at once than the pool holds.
    ///
    /// Sized from `MIN_SCRATCH`, not from a literal: `TEST_FILE`'s pool *is*
    /// the floor, and a width written in as a number stops being pressure the
    /// moment the floor moves.
    #[test]
    fn pressure_beyond_the_pool_spills() {
        let width = u32::from(RegisterFile::MIN_SCRATCH) + 1;
        let mut schedule = vec![def(0, ScheduledOp::Var(0))];
        for i in 1..=width {
            schedule.push(def(i, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        }
        let mut acc = ValueId(1);
        for i in 2..=width {
            schedule.push(def(
                width + i,
                ScheduledOp::Binary(OpKind::Add, acc, ValueId(i)),
            ));
            acc = ValueId(width + i);
        }
        let a = alloc(schedule);
        assert!(spill_count(&a) > 0);
        // And it spills by *splitting*: the loser keeps the register it held
        // up to the point it lost, so a value in memory later is in a register
        // earlier. Condemning a whole life to memory — which is what one
        // location per value could say and this cannot — would leave this
        // empty.
        let split: Vec<ValueId> = a
            .body()
            .schedule()
            .iter()
            .map(|d| d.value)
            .filter(|v| {
                let places = ever(&a, *v);
                places.contains(&Where::Spilled)
                    && places.iter().any(|w| matches!(w, Where::Reg(_)))
            })
            .collect();
        assert!(
            !split.is_empty(),
            "every spilled value went to memory for its whole life; nothing was split"
        );
    }

    /// A constant under pressure is rematerialized, never spilled: re-emitting
    /// the load beats a store plus a reload.
    #[test]
    fn constants_are_rematerialized_rather_than_spilled() {
        let width = u32::from(RegisterFile::MIN_SCRATCH) + 1;
        let mut schedule = vec![def(0, ScheduledOp::Var(0))];
        for i in 1..=width {
            schedule.push(def(i, ScheduledOp::Const(i as f32)));
        }
        let mut acc = ValueId(1);
        for i in 2..=width {
            schedule.push(def(
                width + i,
                ScheduledOp::Binary(OpKind::Add, acc, ValueId(i)),
            ));
            acc = ValueId(width + i);
        }
        let a = alloc(schedule);

        let remat: Vec<(ValueId, u32)> = (1..=width)
            .flat_map(|i| {
                ever(&a, ValueId(i))
                    .into_iter()
                    .filter_map(move |w| match w {
                        Where::Remat(bits) => Some((ValueId(i), bits)),
                        _ => None,
                    })
            })
            .collect();
        assert!(!remat.is_empty(), "constants under pressure should remat");
        assert_eq!(spill_count(&a), 0, "no constant belongs in a spill slot");
        for (vid, bits) in remat {
            assert_eq!(
                bits,
                (vid.0 as f32).to_bits(),
                "{vid:?} rematerializes the wrong constant"
            );
        }
    }

    /// Belady: with no constants in play, the value used farthest in the
    /// future is the one that goes to memory.
    ///
    /// The scenario separates Belady from FIFO and LRU deliberately. Four
    /// values fill the pool in the order v1..v4, then v5 forces an eviction —
    /// but they are *consumed* in that same order, so v1 is simultaneously the
    /// oldest, the least recently used, and the one needed soonest. FIFO and
    /// LRU both evict v1. Only a rule that looks forward evicts v4.
    #[test]
    fn belady_evicts_the_value_used_farthest_out() {
        // One more independent value than the pool holds, so exactly one must
        // go to memory and the test is about *which*.
        let live = u32::from(RegisterFile::MIN_SCRATCH) + 1;
        let mut schedule = vec![def(0, ScheduledOp::Var(0))];
        for i in 1..=live {
            schedule.push(def(i, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        }
        // Consume v1 first, then v2, v3, … and v`live-1` last.
        let mut acc = ValueId(live);
        for i in 1..live {
            schedule.push(def(
                100 + i,
                ScheduledOp::Binary(OpKind::Add, acc, ValueId(i)),
            ));
            acc = ValueId(100 + i);
        }
        let a = alloc(schedule);

        assert!(
            ever(&a, ValueId(live - 1)).contains(&Where::Spilled),
            "v{live_minus_1} is needed last, so it is the one to evict",
            live_minus_1 = live - 1
        );
        assert!(
            !ever(&a, ValueId(1)).contains(&Where::Spilled),
            "v1 is needed next, so it must keep its register for the whole of \
             its life — evicting it is what FIFO and LRU would have done"
        );
    }

    /// Purity is load-bearing: the collapse driver sizes a frame with one run
    /// and emits into it with another, and a disagreement misplaces every slot.
    #[test]
    fn allocation_is_deterministic() {
        let a = alloc(add_xy());
        let b = alloc(add_xy());
        for d in a.body().schedule() {
            assert_eq!(at(&a, d.value), at(&b, d.value));
        }
        assert_eq!(spill_count(&a), spill_count(&b));
    }

    /// A hoisted value is pinned to the slot its prologue parked it in,
    /// overriding whatever the allocator gave the placeholder def.
    #[test]
    fn a_placement_can_be_overridden() {
        let mut a = alloc(add_xy());
        assert!(matches!(at(&a, ValueId(2)), Where::Reg(_)));
        a.place(Scope::Body, ValueId(2), Where::Spilled);
        assert_eq!(at(&a, ValueId(2)), Where::Spilled);
        assert_eq!(spill_count(&a), 1);
    }

    /// All three `Ternary` operands count toward liveness. Missing one frees a
    /// register that is still in use.
    #[test]
    fn every_ternary_operand_extends_liveness() {
        let sel = ScheduledOp::Ternary(OpKind::Select, ValueId(0), ValueId(1), ValueId(2));
        assert_eq!(
            operands(&sel).collect::<Vec<_>>(),
            vec![ValueId(0), ValueId(1), ValueId(2)]
        );
        let a = alloc(vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Var(2)),
            def(3, sel),
        ]);
        assert_eq!(spill_count(&a), 0);
    }

    /// A destination never lands in a register one of its own operands is
    /// still living in — the invariant `resolve_operands` reads back off the
    /// allocation, and the reason SSE2 can write its two-operand form
    /// directly.
    ///
    /// `dst op= right` corrupts `right` when `dst == right` and `dst != left`.
    /// That assignment is unrepresentable here: a destination is drawn from
    /// the slots this instruction has not claimed for a scratch role *and*
    /// that no value it reads is living in, so it can only take a slot whose
    /// occupant is dead or evicted here. The backend needs no stashing temp to
    /// route around a case that cannot arise, which is what `emit_binary_safe`
    /// used to be and what held xmm10 out of every kernel's pool.
    #[test]
    fn a_destination_never_lands_on_a_resident_operand() {
        // Wide enough to evict: `width` values all live at once over a pool of
        // `MIN_SCRATCH`, then folded pairwise so every fold reads two of them.
        let width = u32::from(RegisterFile::MIN_SCRATCH) * 3;
        let mut schedule = vec![def(0, ScheduledOp::Var(0))];
        for i in 1..=width {
            schedule.push(def(i, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        }
        let mut acc = ValueId(1);
        for i in 2..=width {
            schedule.push(def(
                width + i,
                ScheduledOp::Binary(OpKind::Sub, acc, ValueId(i)),
            ));
            acc = ValueId(width + i);
        }

        // Both files: `TEMP_FILE` also reserves a temp before placing the
        // destination, which is the other way a destination could be pushed
        // onto an operand.
        for file in [&TEST_FILE, &TEMP_FILE] {
            let a = LinearScan.allocate(schedule.clone(), file);
            assert!(
                spill_count(&a) > 0,
                "the schedule has to reach eviction for this to test anything"
            );
            // Pairs where both ends are the pool's — the case the two-operand
            // form would corrupt. An input register can never collide with a
            // destination (the pool excludes them), so counting those would
            // let the test pass vacuously.
            let mut contested = 0;
            let body = a.body();
            for (i, d) in body.schedule().iter().enumerate() {
                // At the instruction's own point: that is where a destination
                // and its operands would collide.
                let Where::Reg(dst) = body.where_at(d.value, i) else {
                    continue; // A rematerialized constant: its definition emits nothing.
                };
                for operand in operands(&d.op) {
                    let at = body.where_at(operand, i);
                    assert_ne!(
                        at,
                        Where::Reg(dst),
                        "{:?}: destination {dst:?} is where its operand {operand:?} lives",
                        d.value
                    );
                    if matches!(at, Where::Reg(r) if file.scratch.contains(r)) {
                        contested += 1;
                    }
                }
            }
            assert!(
                contested > 0,
                "no instruction read a pool-resident operand, so nothing above \
                 could have collided"
            );
        }
    }

    /// A carried register is untouched by every scope inside the loop.
    ///
    /// This is the whole safety property of showing the allocator the nest. A
    /// value the outer region leaves in a register is read by the body on
    /// every iteration, so anything the body writes there is a miscompile that
    /// only shows up as wrong pixels. The body's pool excludes carries by
    /// construction (`RegisterFile::inside`) — this is what says so out loud,
    /// and it checks the *temps* too, which are pool registers no `Placement`
    /// records.
    #[test]
    fn a_carried_register_is_untouched_by_everything_inside_the_loop() {
        // An outer region computing invariants, and a body that reads them.
        let mut outer = vec![def(0, ScheduledOp::Var(1))];
        let width = 6u32;
        for i in 1..=width {
            outer.push(def(i, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        }
        let roots: Vec<ValueId> = (1..=width).map(ValueId).collect();

        let mut body = vec![def(100, ScheduledOp::Var(0))];
        let mut acc = ValueId(100);
        for (i, root) in roots.iter().enumerate() {
            body.push(def(
                200 + i as u32,
                ScheduledOp::Binary(OpKind::Add, acc, *root),
            ));
            acc = ValueId(200 + i as u32);
        }

        let nest = ScopedSchedule {
            regions: vec![ScopeRegion {
                roots: roots.clone(),
                schedule: outer,
            }],
            body,
        };
        let alloc = LinearScan.allocate_nest(nest, &NEST_FILE);

        let carries: Vec<(ValueId, Reg)> = roots
            .iter()
            .filter_map(|v| alloc.carried(*v).map(|r| (*v, r)))
            .collect();
        assert!(
            !carries.is_empty(),
            "nothing was carried, so this test asserts nothing about carrying"
        );

        // Every register any scope inside the loop can write. A carried root
        // is *not* one of them: its placement inside the loop is the carry,
        // and the register it held in the producing region belongs to that
        // region — so the roots are excluded rather than counted.
        let body = alloc.body();
        let mut inside: Vec<Reg> = body
            .schedule()
            .iter()
            .filter(|d| !roots.contains(&d.value))
            .flat_map(|d| body.placement(d.value).registers())
            .collect();
        for i in 0..body.schedule().len() {
            let s = body.scratch(i);
            inside.extend((0..Scratch::MAX_TEMPS).filter_map(|k| s.temp(k)));
            inside.extend((0..Scratch::MAX_RELOADS).filter_map(|k| s.reload(k)));
            inside.extend(s.guard_mask);
            inside.extend(s.guard_temp);
            inside.extend(s.result);
        }

        for (vid, carry) in &carries {
            assert!(
                !inside.contains(carry),
                "{vid:?} is carried in {carry:?}, which the body also writes"
            );
        }
    }

    /// A root's placement is the whole of what used to need a `carries` map
    /// beside it: its register inside the region that computes it, and then —
    /// from the first point of the loops within — either the carry or a slot.
    #[test]
    fn a_root_is_placed_twice_and_says_for_itself_whether_it_is_carried() {
        let mut outer = vec![def(0, ScheduledOp::Var(1))];
        let width = 6u32;
        for i in 1..=width {
            outer.push(def(i, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        }
        let roots: Vec<ValueId> = (1..=width).map(ValueId).collect();

        let mut body = vec![def(100, ScheduledOp::Var(0))];
        let mut acc = ValueId(100);
        for (i, root) in roots.iter().enumerate() {
            body.push(def(
                200 + i as u32,
                ScheduledOp::Binary(OpKind::Add, acc, *root),
            ));
            acc = ValueId(200 + i as u32);
        }

        let alloc = LinearScan.allocate_nest(
            ScopedSchedule {
                regions: vec![ScopeRegion {
                    roots: roots.clone(),
                    schedule: outer,
                }],
                body,
            },
            &NEST_FILE,
        );

        let (region, inner) = (alloc.scope(Scope::Region(0)), alloc.body());
        let (mut carried, mut parked) = (0, 0);
        for root in &roots {
            let p = region
                .placement_of(*root)
                .unwrap_or_else(|| panic!("{root:?} is computed by the outer region"));
            // Inside the region: whatever the scan chose, and a pool register
            // there — never the carry, which is picked from what the region
            // leaves free.
            let inside_region = p.at(Point::TAIL);
            let in_the_loop = inner.at_head(*root);
            assert_ne!(
                inside_region, in_the_loop,
                "{root:?} would be in the same place in both scopes, but a root \
                 always changes place at the loop it is read inside"
            );
            match alloc.carried(*root) {
                Some(reg) => {
                    assert_eq!(in_the_loop, Where::Reg(reg));
                    carried += 1;
                }
                None => {
                    assert_eq!(in_the_loop, Where::Spilled, "a parked root is in a slot");
                    parked += 1;
                }
            }
        }
        assert!(carried > 0, "the file has budget, so something is carried");
        assert!(
            parked > 0,
            "the budget is smaller than the root count, so something is parked"
        );
    }

    /// The same teeth for a backend's own scratch: `fixed` is declared so this
    /// is a const-eval failure rather than an argument in a comment.
    #[test]
    #[should_panic(expected = "fixed backend scratch register is inside the allocatable pool")]
    fn a_fixed_register_inside_the_pool_is_refused() {
        let _refused = RegisterFile {
            fixed: &[Reg(5)], // inside TEST_FILE's 4..8 pool
            ..TEST_FILE
        }
        .checked();
    }

    /// A set is not a range: the pool may hold registers on both sides of a
    /// reserved one, which is the whole reason `RegSet` replaced base+count.
    #[test]
    fn the_pool_may_straddle_a_reserved_register() {
        let straddling = RegisterFile {
            scratch: RegSet::of(&[Reg(4), Reg(5), Reg(6), Reg(7), Reg(8), Reg(14), Reg(15)]),
            ..TEST_FILE
        }
        .checked();
        assert_eq!(straddling.scratch.len(), RegisterFile::MIN_SCRATCH);
        let regs: alloc::vec::Vec<Reg> = straddling.scratch.iter().collect();
        assert_eq!(
            regs,
            alloc::vec![Reg(4), Reg(5), Reg(6), Reg(7), Reg(8), Reg(14), Reg(15)]
        );
    }

    /// `capped` shrinks a non-contiguous pool to its lowest members.
    #[test]
    fn capping_takes_the_lowest_members() {
        let set = RegSet::of(&[Reg(4), Reg(9), Reg(14), Reg(15)]);
        let regs: alloc::vec::Vec<Reg> = set.take(2).iter().collect();
        assert_eq!(regs, alloc::vec![Reg(4), Reg(9)]);
        assert_eq!(set.take(99).len(), 4, "capping never grows the pool");
    }

    #[test]
    #[should_panic(expected = "input register is inside the allocatable pool")]
    fn an_input_register_inside_the_pool_is_refused() {
        let _refused = RegisterFile {
            inputs: [Reg(0), Reg(1), Reg(2), Reg(4)],
            ..TEST_FILE
        }
        .checked();
    }

    #[test]
    #[should_panic(expected = "vector_bytes")]
    fn a_non_power_of_two_vector_width_is_refused() {
        let _refused = RegisterFile {
            vector_bytes: 24,
            ..TEST_FILE
        }
        .checked();
    }
}
