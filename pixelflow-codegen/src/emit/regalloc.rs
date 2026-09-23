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

use super::guards::{FoldReads, SelectArm, SelectGuard, analyze_select_guards};
use super::{Gpr, KReg, OperandSource, Reg, ScheduledOp, operand_sources, reloads_wanted};

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
    /// Every register the allocator may hand out.
    ///
    /// Everything outside it — callee-saved registers, the backend's own
    /// `fixed` scratch — is off limits by construction, and
    /// [`RegisterFile::checked`] proves the separation. No register is held
    /// for an input: the collapse ABI passes pointers and a pitch, never a
    /// vector, so every vector register the ABI does not preserve is the
    /// allocator's.
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

    /// The GPR holding the JIT ABI's output-plane argument — where a `Write`
    /// stores — if this target's encodings emit one. Pinned like
    /// [`RegisterFile::gpr_ctx`], for the same reason.
    pub gpr_out: Option<Gpr>,

    /// The GPR holding the JIT ABI's pitch argument — elements between two
    /// rows of the output plane, which a `Write`'s address multiplies its row
    /// by. Pinned like [`RegisterFile::gpr_ctx`].
    pub gpr_pitch: Option<Gpr>,

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
            i += 1;
        }

        // The GPR-class mirror of the vector checks above: the three ABI
        // pointers are pinned, so each must miss the pool the allocator hands
        // out from, and no two may share a register.
        let pinned = [self.gpr_ctx, self.gpr_out, self.gpr_pitch];
        let mut i = 0;
        while i < pinned.len() {
            if let Some(reg) = pinned[i] {
                assert!(
                    !self.gpr_scratch.contains(reg),
                    "an ABI GPR input is inside the allocatable GPR pool"
                );
                let mut k = i + 1;
                while k < pinned.len() {
                    if let Some(other) = pinned[k] {
                        assert!(reg.0 != other.0, "two ABI GPR inputs share a register");
                    }
                    k += 1;
                }
            }
            i += 1;
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

/// Which scope of a loop nest: the body the call runs once, or one of the
/// folds nested in it.
///
/// A **name**, not a coordinate. It used to be half of one — [`Point`] was
/// `(scope, index)` ordered lexicographically — and that only worked while
/// the nest was a *chain*: scopes totally ordered by nesting, and all of an
/// outer scope's code preceding all of an inner scope's. The second stops
/// being true the moment a scope opens in the *middle* of another, which is
/// what a surviving `Reduce` is: the parent's own defs sit on both sides of
/// the fold's. So the ordering moved to where it is always meaningful —
/// within one scope — and this is now only the key that says which one.
///
/// The nest is a **tree**: every fold hangs off whichever scope holds its
/// def. The lattice's own rows and columns are folds too
/// (docs/plans/2026-09-16-collapse-is-a-fold.md), so there is no spine of
/// regions any more — there is the body, run once per call, and folds all
/// the way down. The derived `Ord` is therefore **not** nesting order — it
/// is a total order over names, for deterministic keying.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Scope {
    /// The whole function: what runs once per call, and holds the outermost
    /// fold's def.
    Body,
    /// A surviving `Reduce`'s loop body, indexing
    /// [`NestAllocation::folds`]. It opens in the *middle* of its parent's
    /// schedule, which is what makes the nest a tree.
    Fold(usize),
    /// One arm of a surviving `Guard`, indexing
    /// [`NestAllocation::guard_arms`]. Also opens in the middle of its
    /// parent's schedule — at the `Guard` def, exactly as a fold opens at
    /// its `Reduce` — but it is a branch, not a loop: it runs at most once
    /// per time its parent's def is reached, carries nothing in (a guard
    /// arm's arena is wholly separate from its parent's — a name, not a
    /// closure), and hands back one result the parent stores to a shared
    /// slot (docs/plans/2026-09-12-emit-should-just-emit.md §3). Additive
    /// to [`Scope::Fold`] rather than unified with it: the two are the same
    /// shape to the frame layout (a region opening mid-schedule) but
    /// different in what crosses the boundary, and every existing `Fold`
    /// consumer already assumes a `Reduce` at the opening def — keeping
    /// this a separate variant means those consumers need no change to
    /// keep answering exactly as they did before a `Guard` ever reached
    /// this allocator (the G2 byte-identity gate).
    GuardArm(usize),
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
    roots: Vec<ValueId>,
    /// This scope's `Select` guards, straight from the [`Scan`] that produced
    /// `schedule` — see [`Allocation::select_guards`].
    guards: Vec<SelectGuard>,
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

    /// The temps a surviving `Reduce` def reserves: two, both transient —
    /// the trip test's bound and its mask, reused by the combine's reload
    /// and the step's scratch. Nothing here lives across the loop's body:
    /// the binder and the accumulator are the fold's *roots*, placed by
    /// `allocate_nest` like any other root — carried in a register the
    /// budget allows, or parked in a slot — so the body's pool is the whole
    /// pool minus what is carried, and a reservation across a body is not a
    /// thing this type can express. Every backend's `temps_for` answers this
    /// for a `Reduce`.
    pub const REDUCE_TEMPS: usize = 2;

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
    /// The body: what runs once per call.
    body: ScopeCode,
    /// The surviving folds, indexed by [`Scope::Fold`]. Flat storage; the tree
    /// is each entry's [`FoldScope::parent`].
    folds: Vec<FoldScope>,
    /// The surviving `Guard`s' arms, indexed by [`Scope::GuardArm`]. Paired:
    /// [`RegisterAllocator::allocate_nest`] pushes a `Guard`'s
    /// [`SelectArm::True`] entry immediately before its
    /// [`SelectArm::False`] one, so `guard_arms[2*k]`/`guard_arms[2*k + 1]`
    /// are one `Guard`'s two arms — see [`NestAllocation::guard_count`].
    guard_arms: Vec<GuardArmScope>,
}

/// A surviving `Reduce`'s loop body: a scope, plus where it opens.
///
/// The parent is a pointer rather than recursion, so storage stays flat —
/// which is what the dense placement vectors want — and depth is never a case
/// anything special-cases. A fold inside a fold is simply `parent:
/// Scope::Fold(j)`.
#[derive(Debug)]
struct FoldScope {
    /// The scope whose schedule holds this loop's def.
    parent: Scope,
    /// Which def — the position in `parent`'s schedule of the `Reduce` this is
    /// the body of. A fold opens in the *middle* of its parent, and this is
    /// where.
    at: usize,
    /// The loop's own two values and where it keeps them.
    roots: FoldRoots,
    /// Everything else a scope has.
    ///
    /// A whole [`ScopeCode`], not a bare schedule: the value a fold carries
    /// lives across its back edge, and where a value lives is a *placement*.
    /// A fold scope given only an evaluation order would be the one scope that
    /// exists for a carried value with nowhere to record where that value is.
    code: ScopeCode,
}

/// One arm of a surviving `Guard`: a scope, plus where it opens and which
/// arm it is.
///
/// No `roots` field, unlike [`FoldScope`] — a guard arm's schedule comes
/// from a wholly separate arena (its `KernelKey`'s referent), so it reads
/// nothing from its parent but the branch condition, which the parent
/// resolves *before* branching, not as a value inside this scope. Its own
/// result is not carried anywhere either (see the module's non-goals for
/// this stage): the parent reads it from a slot the driver dedicates to the
/// `Guard`, exactly as it reads a fold's accumulator from one.
#[derive(Debug)]
struct GuardArmScope {
    /// The scope whose schedule holds the `Guard` def this is an arm of.
    parent: Scope,
    /// The `Guard` def's position in `parent`'s schedule.
    at: usize,
    /// Which of the `Guard`'s two arms this is.
    arm: SelectArm,
    /// Everything else a scope has — this arm's own evaluation order,
    /// placements and scratch, exactly as a fold's body has its own.
    code: ScopeCode,
}

/// A surviving fold's own roots — the two values its loop carries across
/// its back edge — and where the allocator put each.
///
/// Placed by [`RegisterAllocator::allocate_nest`] the way a region's roots
/// are: in a register while the budget allows, in a slot otherwise. Nothing
/// is reserved for either by fiat, so a body's pool is the whole pool minus
/// what is carried, at any depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldRoots {
    /// Where the binder lives for the whole of the loop. The loop's trip
    /// test and step read it there, and so does every scope inside that
    /// reads the binder's `Var` — each scope finds that `Var` in its own
    /// schedule by the binder's number, since two sibling folds binding the
    /// same slot share one `Var` node and the answer differs per fold.
    pub binder: Where,
    /// Where the accumulator lives across the loop's iterations. Its result
    /// is in the accumulator's slot either way once the loop exits, which is
    /// where the parent's placement of the `Reduce` def says it is.
    pub accumulator: Where,
}

impl NestAllocation {
    /// How many surviving folds this nest has.
    #[must_use]
    pub fn fold_count(&self) -> usize {
        self.folds.len()
    }

    /// How many surviving `Guard`s this nest has — half of
    /// [`NestAllocation::guard_arms`]'s length, since every guard pushes
    /// its `True` and `False` arm together (see that field's doc).
    #[must_use]
    pub fn guard_count(&self) -> usize {
        self.guard_arms.len() / 2
    }

    /// The `ValueId` guard `k`'s `Guard` def names — the identity the driver
    /// dedicates a result slot to, mirroring [`NestAllocation::fold_reduce_vid`].
    ///
    /// # Panics
    /// If `k` names no guard in this nest, or its parent's schedule does not
    /// reach the position it opens at.
    #[must_use]
    pub fn guard_reduce_vid(&self, k: usize) -> ValueId {
        let arm = &self.guard_arms[2 * k];
        self.code(arm.parent)
            .and_then(|c| c.schedule.get(arm.at))
            .map(|def| def.value)
            .unwrap_or_else(|| {
                panic!(
                    "Guard({k})'s parent {:?} has no def at {}",
                    arm.parent, arm.at
                )
            })
    }

    /// The scope guard `k`'s two arms open inside.
    ///
    /// # Panics
    /// If `k` names no guard in this nest.
    #[must_use]
    pub fn guard_parent(&self, k: usize) -> Scope {
        self.guard_arms
            .get(2 * k)
            .unwrap_or_else(|| panic!("Guard({k}) is not a guard of this nest"))
            .parent
    }

    /// The scope fold `j`'s loop opens inside.
    ///
    /// A fold's loop runs in the middle of its parent's schedule, with the
    /// parent's spilled values live across it, which is why the frame is
    /// laid out as a tree and a fold's base is its parent's top. See
    /// [`StackFrame::with_base`](crate::emit::StackFrame::with_base).
    ///
    /// # Panics
    /// If `j` names no fold in this nest.
    #[must_use]
    pub(crate) fn fold_parent(&self, j: usize) -> Scope {
        self.folds
            .get(j)
            .unwrap_or_else(|| panic!("Fold({j}) is not a fold of this nest"))
            .parent
    }

    /// The `ValueId` fold `j`'s `Reduce` def names — its accumulator's own
    /// identity, for a driver assigning it a slot address before any scope
    /// is emitted.
    ///
    /// # Panics
    /// If `j` names no fold in this nest, or its parent's schedule does not
    /// reach the position it opens at (an inconsistency between this nest's
    /// own folds and the schedule that produced them).
    #[must_use]
    pub fn fold_reduce_vid(&self, j: usize) -> ValueId {
        let fold = &self.folds[j];
        self.code(fold.parent)
            .and_then(|c| c.schedule.get(fold.at))
            .map(|def| def.value)
            .unwrap_or_else(|| {
                panic!(
                    "Fold({j})'s parent {:?} has no def at {}",
                    fold.parent, fold.at
                )
            })
    }

    /// Fold `j`'s own roots and where its loop keeps them.
    ///
    /// # Panics
    /// If `j` names no fold in this nest.
    #[must_use]
    pub fn fold_roots(&self, j: usize) -> FoldRoots {
        self.folds
            .get(j)
            .unwrap_or_else(|| panic!("Fold({j}) is not a fold of this nest"))
            .roots
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

    /// The body — the whole answer for a loop-free schedule.
    #[must_use]
    pub fn body(&self) -> Allocation<'_> {
        self.scope(Scope::Body)
    }

    fn code(&self, scope: Scope) -> Option<&ScopeCode> {
        match scope {
            Scope::Body => Some(&self.body),
            Scope::Fold(i) => self.folds.get(i).map(|f| &f.code),
            Scope::GuardArm(i) => self.guard_arms.get(i).map(|g| &g.code),
        }
    }

    /// The scope `scope` opens inside, or `None` for the body, which opens
    /// inside nothing.
    fn parent_of(&self, scope: Scope) -> Option<Scope> {
        match scope {
            Scope::Body => None,
            Scope::Fold(i) => self.folds.get(i).map(|f| f.parent),
            Scope::GuardArm(i) => self.guard_arms.get(i).map(|g| g.parent),
        }
    }

    /// Every scope of this nest, in no particular order.
    fn scopes(&self) -> impl Iterator<Item = Scope> + use<'_> {
        core::iter::once(Scope::Body)
            .chain((0..self.folds.len()).map(Scope::Fold))
            .chain((0..self.guard_arms.len()).map(Scope::GuardArm))
    }

    /// Whether `outer` contains `inner` — i.e. `outer`'s code runs `inner`.
    ///
    /// Reflexive: a scope encloses itself, so "is this value in scope here"
    /// needs no special case for the scope asking.
    fn encloses(&self, outer: Scope, inner: Scope) -> bool {
        let mut at = Some(inner);
        while let Some(scope) = at {
            if scope == outer {
                return true;
            }
            at = self.parent_of(scope);
        }
        false
    }

    /// The register a root is **carried** in across the loops inside the
    /// scope that computes it, if it is carried at all.
    ///
    /// A root is carried exactly when the scopes inside pick it up in a
    /// register: each reads it from there on every iteration instead of
    /// reloading it from a slot at every use. There is no separate map saying
    /// so — that map was `carries`, and the park a scope inside starts from
    /// already answers, and every scope inside agrees, since none may move
    /// a value the scope outside is holding.
    ///
    /// `None` for a value no scope parks.
    #[must_use]
    pub fn carried(&self, root: ValueId) -> Option<Reg> {
        let parking = self
            .scopes()
            .find(|s| self.code(*s).is_some_and(|c| c.roots.contains(&root)))?;
        self.scope(parking)
            .within()
            .find_map(|inner| inner.placement_of(root).map(|p| p.at(Point::HEAD)))
            .and_then(|at| match at {
                Where::Reg(r) => Some(r),
                Where::Spilled | Where::Remat(_) => None,
            })
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
            Scope::Body => &mut self.body,
            Scope::Fold(i) => &mut self.folds[i].code,
            Scope::GuardArm(i) => &mut self.guard_arms[i].code,
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

    /// This scope's `Select` short-circuit guards, as analyzed once during
    /// allocation.
    ///
    /// The schedule an emitter reads here is the one the allocator scanned —
    /// [`schedule`](Self::schedule) never reorders it — so the guard analysis
    /// is the same question asked and answered twice. This is the answer on
    /// file; nothing downstream needs to ask again.
    #[must_use]
    pub(crate) fn select_guards(&self) -> &'a [SelectGuard] {
        &self.code().guards
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

    /// Where this scope opens in its parent: the parent, and the position in
    /// its schedule. `None` for the body, which is the whole function.
    ///
    /// Only a fold answers: it starts at a def, which is exactly what makes
    /// the nest a tree, so this is the query that distinguishes the two.
    ///
    /// A guard arm also opens at a def — its parent's `Guard` — but answers
    /// `None` here regardless: this query exists for the one walk that reads
    /// it, `emit_scope`'s climb to the binders every enclosing fold's `Write`
    /// needs, and a guard arm's own body can never contain a `Write` (its
    /// arena is wholly separate and never reaches `passes::lattice::collapse`
    /// — see [`Scope::GuardArm`]'s doc), so that walk has nothing to learn by
    /// climbing past one. Its ancestors above the guard are still real folds,
    /// but a *different* [`Allocation`] — the one for the scope the guard
    /// itself sits in — is what would climb into them.
    #[must_use]
    pub fn opens_at(&self) -> Option<(Scope, usize)> {
        match self.scope {
            Scope::Fold(i) => {
                let fold = &self.nest.folds[i];
                Some((fold.parent, fold.at))
            }
            Scope::Body | Scope::GuardArm(_) => None,
        }
    }

    /// Which scope this allocation answers for.
    ///
    /// Not a coordinate — see [`Scope`]'s own doc — but the key an emitter
    /// needs to ask [`Allocation::fold_opening_at`] from the right place.
    #[must_use]
    pub fn scope(&self) -> Scope {
        self.scope
    }

    /// The scope of this same nest that opens *at* `at` in this schedule —
    /// [`Allocation::opens_at`]'s query from the other end, asked by the
    /// emitter walking a schedule position by position rather than by a
    /// fold looking for its own parent.
    ///
    /// A linear scan of the nest's folds: there are a handful per kernel at
    /// most, and this is asked once per schedule position during emission.
    #[must_use]
    pub fn fold_opening_at(&self, at: usize) -> Option<Scope> {
        self.nest
            .folds
            .iter()
            .position(|f| f.parent == self.scope && f.at == at)
            .map(Scope::Fold)
    }

    /// [`Allocation::fold_opening_at`] for a `Guard`'s arm: the scope `arm`
    /// opens at position `at` of this schedule, or `None` when this scope is
    /// not the one that opens it — either an enclosing scope's `Guard`, read
    /// here as a placeholder, or a position with no `Guard` at all.
    #[must_use]
    pub fn guard_opening_at(&self, at: usize, arm: SelectArm) -> Option<Scope> {
        self.nest
            .guard_arms
            .iter()
            .position(|g| g.parent == self.scope && g.at == at && g.arm == arm)
            .map(Scope::GuardArm)
    }

    /// This fold scope's own roots and where its loop keeps them — what the
    /// emitter opening the loop seeds, tests, combines into and steps.
    ///
    /// # Panics
    /// If this scope is not a fold: the body and a guard arm have no binder
    /// or accumulator, so the question has no answer there.
    #[must_use]
    pub fn fold_roots(&self) -> FoldRoots {
        match self.scope {
            Scope::Fold(j) => self.nest.fold_roots(j),
            Scope::Body | Scope::GuardArm(_) => {
                panic!(
                    "{:?} is not a fold, so it has no binder or accumulator",
                    self.scope
                )
            }
        }
    }

    /// This nest's own view of `scope` — a sibling, an ancestor, or a
    /// descendant of the scope this [`Allocation`] answers for.
    ///
    /// # Panics
    /// If `scope` names a scope this nest does not have (see
    /// [`NestAllocation::scope`]).
    #[must_use]
    pub fn sibling(&self, scope: Scope) -> Self {
        self.nest.scope(scope)
    }

    /// The scopes inside this one, in no particular order.
    ///
    /// A root this scope parks is picked up by each of them, so "where does
    /// the code within keep it" is a question about all of them together.
    ///
    /// A **subtree** walk: two folds hanging off one parent are siblings,
    /// each inside the parent and neither inside the other. Answering
    /// "inside" positionally would put a fold's carried value under a
    /// sibling that never runs it.
    pub fn within(self) -> impl Iterator<Item = Allocation<'a>> + use<'a> {
        let nest = self.nest;
        let me = self.scope;
        nest.scopes()
            .filter(move |s| *s != me && nest.encloses(me, *s))
            .map(move |scope| Allocation { nest, scope })
    }

    /// Whether this scope *reads* `v` from an enclosing scope's park rather
    /// than computing it.
    ///
    /// Such a value's entry in this schedule is a placeholder, and its address
    /// is a park slot that outlives every scope's own frame — so it is not
    /// this frame's to place. Narrower than "defined elsewhere": a `Const`
    /// leaf shared with an enclosing scope is genuinely computed here too,
    /// and does need a location of its own.
    ///
    /// Walks up [`NestAllocation::parent_of`]: a scope's ancestors are the
    /// scopes that actually run it.
    #[must_use]
    pub fn parked_by_an_enclosing_scope(&self, v: ValueId) -> bool {
        let mut at = self.nest.parent_of(self.scope);
        while let Some(scope) = at {
            if self.nest.code(scope).is_some_and(|c| c.roots.contains(&v)) {
                return true;
            }
            at = self.nest.parent_of(scope);
        }
        false
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
    /// folds, nothing carried across anything.
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
                body: ScopeRegion {
                    roots: Vec::new(),
                    schedule: dag,
                },
                folds: Vec::new(),
                guard_arms: Vec::new(),
            },
            file,
        )
    }
}

/// A schedule split by scope: the body the call runs once, and the folds
/// nested in it.
///
/// This is the loop nest as data. `body` is what happens once per call — the
/// per-call values, and the outermost fold's def — and each fold's schedule
/// is what happens once per trip of its loop. Each scope's `roots` are the
/// values it computes for the scopes inside it: placed here, by the
/// outermost scope binding every binder the value depends on, so a value
/// depending on nothing is computed once per call and one depending only on
/// the row binder once per row (docs/plans/2026-09-16-collapse-is-a-fold.md
/// §2.2).
///
/// The body is a field rather than `folds[0]` because it is genuinely a
/// different thing: it wraps everything and opens nowhere.
pub struct ScopedSchedule {
    /// What runs once per call.
    pub body: ScopeRegion,
    /// The surviving folds, in [`Scope::Fold`] order — every one of them,
    /// the lattice's own included.
    pub folds: Vec<ScopeFold>,
    /// The surviving `Guard`s' arms, in [`Scope::GuardArm`] order. Built
    /// separately from `folds` — after [`RegisterAllocator::allocate_nest`]'s
    /// caller has already carved the folds out and clustered their arms —
    /// because a guard arm's schedule does not come from carving anything
    /// out of this nest's own; it comes from resolving a wholly separate
    /// `KernelKey` (see [`Scope::GuardArm`]'s doc).
    pub guard_arms: Vec<ScopeGuardArm>,
}

/// One scope of a [`ScopedSchedule`] that opens nowhere: the body.
pub struct ScopeRegion {
    /// Values this scope computes for the ones inside it.
    pub roots: Vec<ValueId>,
    /// What it computes, in topological order.
    pub schedule: Vec<Def>,
}

/// One surviving fold of a [`ScopedSchedule`]: a scope that opens in the
/// middle of another scope.
pub struct ScopeFold {
    /// The scope whose schedule holds this loop's def.
    pub parent: Scope,
    /// Which def of `parent` — the `Reduce` this is the body of.
    pub at: usize,
    /// Values this fold computes for the scopes inside it.
    pub roots: Vec<ValueId>,
    /// The loop body, in topological order.
    pub schedule: Vec<Def>,
}

/// One arm of a surviving `Guard`, as an input to
/// [`RegisterAllocator::allocate_nest`]: a scope that opens in the middle of
/// another scope, exactly like [`ScopeFold`], but with no `roots` — its
/// schedule is wholly self-contained (a separate arena's own, freshly
/// numbered), so it has nothing to read from its parent beyond the branch
/// condition the parent resolves before ever reaching this scope.
pub struct ScopeGuardArm {
    /// The scope whose schedule holds the `Guard` def this is an arm of.
    pub parent: Scope,
    /// The `Guard` def's position in `parent`'s schedule.
    pub at: usize,
    /// Which of the `Guard`'s two arms this is.
    pub arm: SelectArm,
    /// This arm's own evaluation order, in topological order, ending at the
    /// value the parent stores to the `Guard`'s result slot.
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
/// O(n × k) for n values and a k-register pool. For the pools here (6–26
/// registers) that is effectively O(n).
///
/// The DAGs reaching this are in SSA form, which makes their interference
/// graphs chordal — the shape on which greedy coloring is optimal. What
/// coloring does not decide, and this does, is *spill placement*: where a live
/// range is cut, and where the value comes back.
#[derive(Copy, Clone, Debug, Default)]
pub struct LinearScan;

/// Which roots of a nest are carried in registers — the decision every
/// scope's allocation then follows.
///
/// One ranking over every candidate — each scope's roots, and each fold's
/// binder and accumulator — by what a carry saves per call. A scope's root
/// saves one reload per read in every scope inside, each read weighted by
/// how many times per call the reading scope runs. A fold root saves its
/// loop's own traffic — the trip test and the step for a binder, the
/// combine's reload and store for an accumulator — plus one reload per read
/// of the binder inside, all multiplied by how many times per call the loop
/// runs, so a counter stepped thirty times a batch outranks a root read once.
/// Every trip count is static now that the lattice's rows and columns are
/// folds like any other, so there is one scale and no tier.
///
/// Greedy, under one constraint: a carry is live across every scope inside
/// the one that computes it, and no scope may have more registers carried
/// across it than the pool has above the floor. That is what keeps every
/// scope's own allocation possible at any depth, and it is the whole of what
/// used to be a per-scope budget.
struct CarryPlan {
    /// Per scope (the body, then the folds), the roots to carry, hottest
    /// first.
    scope_roots: Vec<Vec<ValueId>>,
    /// Per fold, whether its binder is carried.
    fold_binder: Vec<bool>,
    /// Per fold, whether its accumulator is carried.
    fold_accumulator: Vec<bool>,
}

/// The `Var` `var` is read through in `schedule`, if it is read there.
///
/// By number, not identity: a `Var` node is shared by every fold binding the
/// same slot, so which loop's counter it names is a fact about the schedule
/// reading it.
fn var_in(schedule: &[Def], var: u8) -> Option<ValueId> {
    schedule
        .iter()
        .find(|d| matches!(d.op, ScheduledOp::Var(v) if v == var))
        .map(|d| d.value)
}

/// The scope index the count is kept under: the body first, then the folds.
///
/// A guard arm never reaches this: [`plan_carries`] and
/// [`LinearScan::allocate_nest`]'s `carried_into`/`parked_by` bookkeeping are
/// about what a scope *carries into the scopes inside it*, and a guard arm
/// carries nothing in and opens nothing of its own (see [`Scope::GuardArm`]'s
/// doc) — so nothing ever indexes either by one.
fn scope_ix(scope: Scope) -> usize {
    match scope {
        Scope::Body => 0,
        Scope::Fold(j) => j + 1,
        Scope::GuardArm(_) => unreachable!(
            "scope_ix: a guard arm carries nothing into scopes inside it and \
             opens none of its own, so plan_carries and allocate_nest never index one"
        ),
    }
}

fn plan_carries(nest: &ScopedSchedule, above_floor: usize) -> CarryPlan {
    let folds = nest.folds.len();
    for (index, fold) in nest.folds.iter().enumerate() {
        assert!(
            match fold.parent {
                // A fold nested in a fold needs its parent's answer, so
                // parents come first. Also makes a parent cycle unsayable.
                Scope::Fold(j) => j < index,
                Scope::Body => true,
                Scope::GuardArm(_) => unreachable!("a fold's parent is never a guard arm"),
            },
            "Fold({index})'s parent {:?} is not an earlier scope",
            fold.parent
        );
    }

    let schedule_of = |scope: Scope| -> &[Def] {
        match scope {
            Scope::Body => &nest.body.schedule,
            Scope::Fold(j) => &nest.folds[j].schedule,
            Scope::GuardArm(_) => unreachable!("plan_carries never asks about a guard arm"),
        }
    };
    let roots_of = |scope: Scope| -> &[ValueId] {
        match scope {
            Scope::Body => &nest.body.roots,
            Scope::Fold(j) => &nest.folds[j].roots,
            Scope::GuardArm(_) => unreachable!("plan_carries never asks about a guard arm"),
        }
    };
    let meta_of = |j: usize| -> &pixelflow_ir::fold::RangeFold {
        let fold = &nest.folds[j];
        let def = &schedule_of(fold.parent)[fold.at];
        let ScheduledOp::Reduce(meta, _) = &def.op else {
            panic!(
                "Fold({j})'s parent def at {:?}[{}] is not a Reduce",
                fold.parent, fold.at
            )
        };
        meta
    };
    // How many times per call fold `j`'s body runs: its own trip count times
    // every enclosing fold's. The body runs once.
    let mut trips: Vec<usize> = Vec::with_capacity(folds);
    for j in 0..folds {
        let own = meta_of(j).len() as usize;
        trips.push(match nest.folds[j].parent {
            Scope::Fold(p) => own * trips[p],
            Scope::Body => own,
            Scope::GuardArm(_) => unreachable!("a fold's parent is never a guard arm"),
        });
    }
    let trips_of = |scope: Scope| match scope {
        Scope::Body => 1,
        Scope::Fold(j) => trips[j],
        Scope::GuardArm(_) => unreachable!("plan_carries never asks about a guard arm"),
    };
    // The folds from `k` up to `j`, `k` first, if `j` encloses `k` (or is
    // it); empty otherwise.
    let chain_up_to = |k: usize, j: usize| -> Vec<usize> {
        let mut chain = vec![k];
        let mut at = k;
        while at != j {
            match nest.folds[at].parent {
                Scope::Fold(p) => {
                    chain.push(p);
                    at = p;
                }
                Scope::Body => return Vec::new(),
                Scope::GuardArm(_) => unreachable!("a fold's parent is never a guard arm"),
            }
        }
        chain
    };
    // Whether `outer` runs `inner`: `inner` is `outer`, or nested in it.
    let inside = |outer: Scope, inner: Scope| -> bool {
        match (outer, inner) {
            (Scope::Body, _) => true,
            (Scope::Fold(_), Scope::Body) => false,
            (Scope::Fold(o), Scope::Fold(i)) => !chain_up_to(i, o).is_empty(),
            (Scope::GuardArm(_), _) | (_, Scope::GuardArm(_)) => {
                unreachable!("plan_carries never asks about a guard arm")
            }
        }
    };
    let scopes = || core::iter::once(Scope::Body).chain((0..folds).map(Scope::Fold));
    // Every read of fold `j`'s binder inside it, weighted by how often the
    // reading fold runs — across `j`'s own body and every fold within,
    // except where a fold in between rebinds the same slot.
    let binder_reads = |j: usize| -> usize {
        let var = meta_of(j).binder().var();
        (0..folds)
            .map(|k| {
                let chain = chain_up_to(k, j);
                if chain.is_empty()
                    || chain[..chain.len() - 1]
                        .iter()
                        .any(|f| meta_of(*f).binder().var() == var)
                {
                    return 0;
                }
                let Some(bv) = var_in(&nest.folds[k].schedule, var) else {
                    return 0;
                };
                let reads = nest.folds[k]
                    .schedule
                    .iter()
                    .flat_map(|d| operands(&d.op))
                    .filter(|o| *o == bv)
                    .count();
                reads * trips[k]
            })
            .sum()
    };

    enum Root {
        Scope(Scope, ValueId),
        Binder(usize),
        Accumulator(usize),
    }
    struct Candidate {
        weight: usize,
        live_across: Vec<usize>,
        root: Root,
    }
    let mut candidates: Vec<Candidate> = Vec::new();

    // A scope's root is read by the scopes inside it, each read costing a
    // reload every time that scope runs; the carry is live across all of
    // them.
    for scope in scopes() {
        let within: Vec<Scope> = scopes()
            .filter(|s| *s != scope && inside(scope, *s))
            .collect();
        let live_across: Vec<usize> = within.iter().map(|s| scope_ix(*s)).collect();
        // By id within the scope, so two roots read the same number of
        // times are ordered by something other than map order.
        let mut roots: Vec<ValueId> = roots_of(scope).to_vec();
        roots.sort_by_key(|v| v.0);
        for v in roots {
            let weight: usize = within
                .iter()
                .map(|s| {
                    schedule_of(*s)
                        .iter()
                        .flat_map(|d| operands(&d.op))
                        .filter(|o| *o == v)
                        .count()
                        * trips_of(*s)
                })
                .sum();
            if weight == 0 {
                continue;
            }
            candidates.push(Candidate {
                weight,
                live_across: live_across.clone(),
                root: Root::Scope(scope, v),
            });
        }
    }
    for (j, &trips_j) in trips.iter().enumerate() {
        let live_across: Vec<usize> = (0..folds)
            .filter(|&k| !chain_up_to(k, j).is_empty())
            .map(|k| scope_ix(Scope::Fold(k)))
            .collect();
        // The trip test and the step read the binder once each per trip;
        // the combine reloads and stores the accumulator once each — unless
        // the fold is over the unit monoid, whose accumulator nothing ever
        // reads or writes.
        let accumulates = meta_of(j).monoid() != pixelflow_ir::fold::Monoid::SEQ;
        candidates.push(Candidate {
            weight: 2 * trips_j + binder_reads(j),
            live_across: live_across.clone(),
            root: Root::Binder(j),
        });
        if accumulates {
            candidates.push(Candidate {
                weight: 2 * trips_j,
                live_across,
                root: Root::Accumulator(j),
            });
        }
    }
    // Stable, so the order above breaks ties: outer scopes' roots before
    // inner, a binder before its accumulator, lower ids first.
    candidates.sort_by_key(|c| core::cmp::Reverse(c.weight));

    let mut count = vec![0usize; 1 + folds];
    let mut plan = CarryPlan {
        scope_roots: vec![Vec::new(); 1 + folds],
        fold_binder: vec![false; folds],
        fold_accumulator: vec![false; folds],
    };
    for candidate in candidates {
        if candidate
            .live_across
            .iter()
            .any(|&s| count[s] >= above_floor)
        {
            continue;
        }
        for &s in &candidate.live_across {
            count[s] += 1;
        }
        match candidate.root {
            Root::Scope(scope, v) => plan.scope_roots[scope_ix(scope)].push(v),
            Root::Binder(j) => plan.fold_binder[j] = true,
            Root::Accumulator(j) => plan.fold_accumulator[j] = true,
        }
    }
    plan
}

/// Every register a scan's own code touches: placed values AND
/// per-instruction scratch. A temp is a pool register that no placement
/// records, so taking the complement of the locations alone would hand out a
/// register the scope destroys.
fn registers_used(scan: &Scan) -> RegSet {
    let mut used: Vec<Reg> = scan.registers().collect();
    for scratch in &scan.scratch {
        used.extend(scratch.temps.iter().flatten().copied());
        used.extend(scratch.reloads.iter().flatten().copied());
        used.extend(scratch.guard_mask);
        used.extend(scratch.guard_temp);
        used.extend(scratch.result);
    }
    RegSet::of(&used)
}

impl RegisterAllocator for LinearScan {
    fn allocate_nest(&self, nest: ScopedSchedule, file: &RegisterFile) -> NestAllocation {
        // Which roots are carried, decided once over the whole nest; each
        // scope below picks the registers.
        let above_floor = file.scratch.len().saturating_sub(RegisterFile::MIN_SCRATCH) as usize;
        let plan = plan_carries(&nest, above_floor);

        // Outermost first, because that is the direction liveness flows: a
        // value a scope computes for the scopes inside it is live across
        // every iteration of every loop between here and its last use. A
        // register holding it is therefore unavailable to all of them.
        //
        // Per scope, once scanned: what the scopes inside may not allocate
        // (the ancestors' carries, the scope's own fold roots' carries, and
        // the carries of its own roots), and where each root an ancestor or
        // the scope itself parked lives for the whole of every scope inside —
        // the register carrying it, or its park slot. A scan inside reads
        // this rather than choosing, which is what lets it tell a resident
        // operand from one it has to reload, and it is that scope's whole
        // answer for the value, since nothing inside may move it.
        let mut carried_into: Vec<RegSet> = Vec::with_capacity(1 + nest.folds.len());
        let mut parked_by: Vec<BTreeMap<ValueId, Where>> = Vec::with_capacity(1 + nest.folds.len());

        // Carry the roots the plan chose, hottest first, from the registers
        // the scope's own code leaves free, and record where each root lives
        // for the scopes inside.
        let park_roots = |scan: &Scan,
                          roots: &[ValueId],
                          chosen: &[ValueId],
                          free: RegSet,
                          parked: &mut BTreeMap<ValueId, Where>|
         -> RegSet {
            let mut available = free.iter();
            let mut own = RegSet::EMPTY;
            let mut carries: BTreeMap<ValueId, Reg> = BTreeMap::new();
            for &vid in chosen {
                let Some(reg) = available.next() else { break };
                own = own.union(RegSet::of(&[reg]));
                carries.insert(vid, reg);
            }
            for root in roots {
                let at = match carries.get(root) {
                    Some(reg) => Where::Reg(*reg),
                    None => Where::Spilled,
                };
                assert!(
                    scan.ranges
                        .get(root.0 as usize)
                        .is_some_and(|r| !r.is_empty()),
                    "a scope computes its own roots, but {root:?} is not in it"
                );
                parked.insert(*root, at);
            }
            own
        };

        // Every scope's `Select` guards, before any scan: what an arm may own
        // depends on what the loops the scope opens read from it
        // (`FoldReads`), which takes those loops' schedules — and the loop
        // below reaches a fold only after the scope it opens in.
        let guards_in = |scope: Scope, schedule: &[Def], roots: &[ValueId]| {
            let reads = FoldReads::new(
                schedule,
                nest.folds
                    .iter()
                    .filter(|fold| fold.parent == scope)
                    .map(|fold| (schedule[fold.at].value, fold.schedule.as_slice())),
            );
            analyze_select_guards(schedule, roots, &reads)
        };
        let mut guards: Vec<Vec<SelectGuard>> = core::iter::once(guards_in(
            Scope::Body,
            &nest.body.schedule,
            &nest.body.roots,
        ))
        .chain(
            nest.folds
                .iter()
                .enumerate()
                .map(|(j, fold)| guards_in(Scope::Fold(j), &fold.schedule, &fold.roots)),
        )
        .collect();

        let body_scan = self.scan(
            nest.body.schedule,
            file,
            &BTreeMap::new(),
            core::mem::take(&mut guards[scope_ix(Scope::Body)]),
        );
        let mut body_parked: BTreeMap<ValueId, Where> = BTreeMap::new();
        let body_carries = park_roots(
            &body_scan,
            &nest.body.roots,
            &plan.scope_roots[scope_ix(Scope::Body)],
            file.scratch.without(registers_used(&body_scan)),
            &mut body_parked,
        );
        let body_code = ScopeCode {
            placements: record(&body_scan, &BTreeMap::new()),
            schedule: body_scan.schedule,
            scratch: body_scan.scratch,
            roots: nest.body.roots,
            guards: body_scan.guards,
        };
        carried_into.push(body_carries);
        parked_by.push(body_parked);

        // Each surviving fold, against the pool minus what its *ancestors*
        // carry and the parks they left it. A fold's own roots — its binder
        // and its accumulator — are placed here the way a scope's roots are
        // placed above: carried in a register while the budget allows, parked
        // in a slot otherwise. Nothing is reserved across a body by fiat; the
        // binder used to be, pinned to a temp for the loop's whole life and
        // kept out of every pool inside, which put a depth on how many folds
        // could nest before a pool fell under the floor. The allocator decides
        // now, and at the floor it decides "slot", which always fits.
        let mut folds: Vec<FoldScope> = Vec::with_capacity(nest.folds.len());
        // Per fold, the `Var` its binder is and where it lives, for a fold
        // nested inside it that reads the outer index (a contraction does).
        let mut binders: Vec<(u8, Where)> = Vec::with_capacity(nest.folds.len());
        // Where every fold of the nest opens, so a `Reduce` def in a fold's
        // schedule that opens none of them can be told from one that does:
        // it is a placeholder for an enclosing scope's fold, read from the
        // accumulator slot that loop left its result in.
        let opens: Vec<(Scope, usize)> = nest.folds.iter().map(|f| (f.parent, f.at)).collect();
        for (index, fold) in nest.folds.into_iter().enumerate() {
            // The `Reduce` def this fold is the body of — read back out of
            // whichever scope's schedule holds it, the same schedule
            // `Allocation::opens_at` names, so the fold's own metadata (its
            // monoid, its binder, its range) never needs restating on
            // `ScopeFold` itself.
            let parent_code: &ScopeCode = match fold.parent {
                Scope::Body => &body_code,
                Scope::Fold(j) => &folds[j].code,
                Scope::GuardArm(_) => {
                    unreachable!("a fold's parent is never a guard arm (no fold nests in one)")
                }
            };
            let carried_into_parent = carried_into[scope_ix(fold.parent)];
            let ScheduledOp::Reduce(fold_meta, _) = &parent_code.schedule[fold.at].op else {
                panic!(
                    "Fold({index})'s parent def at {:?}[{}] is not a Reduce",
                    fold.parent, fold.at
                );
            };
            let binder_var = fold_meta.binder().var();
            // The binder's `Var` as this body reads it, if it does (an
            // unusual but valid fold never does).
            let binder_vid = var_in(&fold.schedule, binder_var);

            // The fold's roots, carried as the plan decided, from the
            // registers free at the def — everything the parent holds
            // resident was evicted there (see `scan`), so that is the pool
            // minus the ancestors' carries and the def's own scratch.
            let parent_scratch = &parent_code.scratch[fold.at];
            let mut taken_at_def: Vec<Reg> = Vec::new();
            taken_at_def.extend(parent_scratch.temps.iter().flatten().copied());
            taken_at_def.extend(parent_scratch.reloads.iter().flatten().copied());
            taken_at_def.extend(parent_scratch.guard_mask);
            taken_at_def.extend(parent_scratch.guard_temp);
            taken_at_def.extend(parent_scratch.result);
            let free = file
                .scratch
                .without(carried_into_parent)
                .without(RegSet::of(&taken_at_def));
            let mut available = free.iter();
            let mut own = RegSet::EMPTY;
            let (mut binder_at, mut accumulator_at) = (Where::Spilled, Where::Spilled);
            for (wanted, at) in [
                (plan.fold_binder[index], &mut binder_at),
                (plan.fold_accumulator[index], &mut accumulator_at),
            ] {
                if !wanted {
                    continue;
                }
                let Some(reg) = available.next() else { break };
                own = own.union(RegSet::of(&[reg]));
                *at = Where::Reg(reg);
            }
            let carried_in = carried_into_parent.union(own);

            // What the body finds parked: the ancestors' roots, a `Reduce`
            // it reads but does not open (an enclosing scope's fold, in its
            // slot), every enclosing fold's binder where *that* loop keeps
            // it, and its own binder where this loop keeps it — last, so a
            // binder that shadows an enclosing one is this fold's.
            let mut fold_parked = parked_by[scope_ix(fold.parent)].clone();
            for (at, def) in fold.schedule.iter().enumerate() {
                if matches!(def.op, ScheduledOp::Reduce(..))
                    && !opens.contains(&(Scope::Fold(index), at))
                {
                    fold_parked.insert(def.value, Where::Spilled);
                }
            }
            let mut up = fold.parent;
            while let Scope::Fold(j) = up {
                let (var, at) = binders[j];
                if let Some(bv) = var_in(&fold.schedule, var) {
                    fold_parked.insert(bv, at);
                }
                up = folds[j].parent;
            }
            if let Some(bv) = binder_vid {
                fold_parked.insert(bv, binder_at);
            }
            binders.push((binder_var, binder_at));

            let scan = self.scan(
                fold.schedule,
                &file.inside(carried_in),
                &fold_parked,
                core::mem::take(&mut guards[scope_ix(Scope::Fold(index))]),
            );
            // This fold's own roots, for the folds inside it: carried from
            // what its own code leaves free, or parked.
            let root_carries = park_roots(
                &scan,
                &fold.roots,
                &plan.scope_roots[scope_ix(Scope::Fold(index))],
                file.scratch
                    .without(carried_in)
                    .without(registers_used(&scan)),
                &mut fold_parked,
            );
            carried_into.push(carried_in.union(root_carries));
            parked_by.push(fold_parked.clone());
            // The scan's parks are what came *in*; what it parks for the
            // scopes within is its own answer, recorded above and not a
            // range on this scope's placement of the value.
            let came_in: BTreeMap<ValueId, Where> = fold_parked
                .iter()
                .filter(|(v, _)| !fold.roots.contains(v))
                .map(|(v, at)| (*v, *at))
                .collect();
            folds.push(FoldScope {
                parent: fold.parent,
                at: fold.at,
                roots: FoldRoots {
                    binder: binder_at,
                    accumulator: accumulator_at,
                },
                code: ScopeCode {
                    placements: record(&scan, &came_in),
                    schedule: scan.schedule,
                    scratch: scan.scratch,
                    roots: fold.roots,
                    guards: scan.guards,
                },
            });
        }

        // Every surviving `Guard`'s two arms. Independent of `plan_carries`
        // and the `CarryPlan` above: an arm's schedule is wholly
        // self-contained (see `ScopeGuardArm`'s doc), so it has no roots to
        // rank against anything and nothing of the fold machinery above
        // applies — it needs only the register pool free at the point its
        // `Guard` def sits, exactly the pool a fold body would get if it
        // opened there instead (`Scan::scan`'s pre-emptive eviction at a
        // `Guard` def, mirroring what it already does at a `Reduce`, is what
        // makes "the whole pool minus the ancestors' carries" sound here: no
        // parent value survives in a register across this position for the
        // arm to clobber).
        let mut guard_arms: Vec<GuardArmScope> = Vec::with_capacity(nest.guard_arms.len());
        for arm in nest.guard_arms {
            let parent_code: &ScopeCode = match arm.parent {
                Scope::Body => &body_code,
                Scope::Fold(j) => &folds[j].code,
                Scope::GuardArm(_) => {
                    unreachable!("a guard arm's parent is never a guard arm (none nest in one)")
                }
            };
            let carried_into_parent = carried_into[scope_ix(arm.parent)];
            let parent_scratch = &parent_code.scratch[arm.at];
            let mut taken_at_def: Vec<Reg> = Vec::new();
            taken_at_def.extend(parent_scratch.temps.iter().flatten().copied());
            taken_at_def.extend(parent_scratch.reloads.iter().flatten().copied());
            taken_at_def.extend(parent_scratch.guard_mask);
            taken_at_def.extend(parent_scratch.guard_temp);
            taken_at_def.extend(parent_scratch.result);
            let inside = file.inside(carried_into_parent.union(RegSet::of(&taken_at_def)));
            // An arm parks nothing and opens no fold: its schedule is a
            // separate arena's, which `extract_guards` carves nothing out of.
            let arm_guards = analyze_select_guards(&arm.schedule, &[], &FoldReads::default());
            let scan = self.scan(arm.schedule, &inside, &BTreeMap::new(), arm_guards);
            guard_arms.push(GuardArmScope {
                parent: arm.parent,
                at: arm.at,
                arm: arm.arm,
                code: ScopeCode {
                    placements: record(&scan, &BTreeMap::new()),
                    schedule: scan.schedule,
                    scratch: scan.scratch,
                    roots: Vec::new(),
                    guards: scan.guards,
                },
            });
        }

        NestAllocation {
            body: body_code,
            folds,
            guard_arms,
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
    /// This scope's `Select` guards, analyzed once against `schedule` here and
    /// carried into its [`ScopeCode`] rather than recomputed at emission: the
    /// schedule a scope emits is the one it was scanned with, unchanged, so a
    /// second analysis of it would answer a question already on file.
    guards: Vec<SelectGuard>,
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
/// What it costs the instruction being placed to lose one of its own reads —
/// the tier that outranks every kind of deferred traffic, and the reason an
/// operand's register is a *priced* choice rather than a forbidden one.
///
/// Ordered cheapest first. The distinction between the two read-here cases is
/// what makes the exhausted pool feasible: when every held register belongs to
/// something this instruction reads, the loser has to be one of them, and only
/// one kind of them costs no further register.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum ReadHere {
    /// Not read by this instruction or by a guard emitted before it.
    No,
    /// Read here, and it is the operand the encoding consumes *from the
    /// destination* ([`OperandSource::Destination`]): losing its register
    /// means one reload — into `dst`, which is the register it is losing —
    /// and no other register at all.
    FromDst,
    /// Read here and needs a register of its own to be read from: a reload
    /// register the pool then has to find too, or a guard's mask register for
    /// a branch emitted before the instruction.
    NeedsRegister,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct EvictionRank {
    /// Read by the instruction being placed — see [`ReadHere`].
    ///
    /// The tiers below price the traffic an eviction *defers*; for a value read
    /// right here there is nothing to defer, so taking its register buys a
    /// reload inside this very instruction. Without this, a value already in
    /// its slot is the standing favourite — and at a read, the standing
    /// favourite is whichever value the instruction is reading.
    ///
    /// Answered from the instruction's own read set, never from the read
    /// cursor: the kept-reload step advances the cursor past the current index
    /// (`next_read(operand, i + 1)`), so by the time the destination is
    /// contested a just-kept operand would read as "not needed now".
    read_here: ReadHere,
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
/// its operand reloads, a guard's mask and scratch, the scope's result, and
/// the destination itself — which is the same list [`RegisterFile::MIN_SCRATCH`]
/// is derived from.
struct Reservations {
    slots: [Option<usize>; Self::ROLES],
    filled: usize,
}

impl Reservations {
    /// `+ 4`: guard mask, guard temp, result, and the destination. The
    /// destination is in here because it is claimed *before* the reloads and
    /// the guard's registers now, so those have to see it as taken rather
    /// than rely on being chosen after it.
    const ROLES: usize = Scratch::MAX_TEMPS + Scratch::MAX_RELOADS + 4;

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
    /// `sites[i]` is every mask a guard emitted *before* instruction `i` reads
    /// ([`guard_sites`]). A guard's read is a read: it decides `expire`, the
    /// Belady distance and the read-here tier exactly as an operand's does,
    /// and leaving it out made a mask read only by a branch the preferred
    /// eviction at the very index the branch tests it.
    fn new(
        dag: &[Def],
        file: &RegisterFile,
        vec_len: usize,
        live_in: &BTreeMap<ValueId, Where>,
        sites: &[Vec<ValueId>],
    ) -> Self {
        let mut reads: Vec<Vec<usize>> = vec![Vec::new(); vec_len];
        let mut const_bits: Vec<Option<u32>> = vec![None; vec_len];
        let mut defined_at: Vec<usize> = vec![usize::MAX; vec_len];
        for (i, def) in dag.iter().enumerate() {
            defined_at[def.value.0 as usize] = i;
            if let ScheduledOp::Const(val) = def.op {
                const_bits[def.value.0 as usize] = Some(val.to_bits());
            }
            // Operands and guard masks together, so each value's read list
            // stays ascending with one entry per index.
            for read in operands(&def.op).chain(sites[i].iter().copied()) {
                let r = &mut reads[read.0 as usize];
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
    /// operands and its guards' masks) and from every role it has already
    /// filled — the destination included, once it has been claimed, because
    /// it is pushed into `taken` like any other role.
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
    ///
    /// `read_here` is the instruction's own read set with its tier per value
    /// ([`ReadHere`]); a value not in it is not read here. Empty where the
    /// candidates already exclude everything the instruction reads.
    fn rank(&mut self, v: ValueId, from: usize, read_here: &[(ValueId, ReadHere)]) -> EvictionRank {
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
            read_here: read_here
                .iter()
                .find(|(r, _)| *r == v)
                .map_or(ReadHere::No, |(_, tier)| *tier),
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
    /// already claimed for a role.
    ///
    /// An operand's register is in here and does not need excluding — for the
    /// **destination**, which is the one role claimed from this set. Taking it
    /// makes that operand non-resident at this index, *before* the reload
    /// count is taken, so `resolve_operands` reloads it from the slot its
    /// definition wrote — into `dst` for the operand the encoding consumes
    /// there, into a reserved reload register otherwise. That is the whole
    /// of what the encoders tolerate: no encoder reads every source before
    /// writing `dst` (SSE2's `movaps dst, src1` prelude; `setup_mov` ahead of
    /// a `Select` or FMA on every ISA), so a *resident* operand in `dst`'s
    /// register would be corrupted. A displaced one is not resident, which is
    /// why the split is recorded at this index and not the next.
    /// [`EvictionRank`] prices it so it stays a last resort.
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
    fn loser(
        &mut self,
        open: &[usize],
        index: usize,
        read_here: &[(ValueId, ReadHere)],
    ) -> Option<usize> {
        let held: Vec<(usize, ValueId)> = open
            .iter()
            .filter_map(|k| self.owner[*k].map(|v| (*k, v)))
            .collect();
        held.into_iter()
            .min_by_key(|(_, v)| self.rank(*v, index, read_here))
            .map(|(k, _)| k)
    }

    /// A pool register for a scratch role or a kept reload at `index`: a free
    /// one if there is one, and otherwise the one whose occupant is cheapest
    /// to evict — which splits that occupant's live range here.
    ///
    /// The candidates here already exclude everything the instruction reads
    /// (see [`Self::without_operands`]), so no occupant is read here and the
    /// read-here tier is uniformly [`ReadHere::No`].
    ///
    /// Answers as long as the pool leaves one register past the roles: the
    /// floor [`RegisterFile::MIN_SCRATCH`] is derived from the widest set of
    /// roles one instruction can hold at once. A pool cut below it — a fold
    /// scope's is its parent's minus the carried registers minus the loop's
    /// own three — is a floor bug to fix at the floor, not here.
    fn claim(&mut self, index: usize, open: &[usize]) -> usize {
        if let Some(free) = open.iter().copied().find(|k| self.owner[*k].is_none()) {
            return free;
        }
        let slot = self.loser(open, index, &[]).unwrap_or_else(|| {
            unreachable!(
                "every pool register is already one of this instruction's roles \
                 or a value it reads, against a floor of {}",
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
    ///
    /// `guards` are this schedule's `Select` guards: [`analyze_select_guards`]
    /// over it, told what the scopes inside it read — its roots, which no arm
    /// may own (a skipped arm would leave the park unwritten for a loop that
    /// runs regardless), and what each loop it opens reads, which no arm may
    /// own unless the loop is skipped with it.
    fn scan(
        &self,
        dag: Vec<Def>,
        file: &RegisterFile,
        live_in: &BTreeMap<ValueId, Where>,
        guards: Vec<SelectGuard>,
    ) -> Scan {
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
                guards,
            };
        }

        // The arms a `Select` guard may skip. A register range that begins at a
        // read inside one, for a value defined outside it, must end there too:
        // after the arm a read has to name what it named before, because the
        // skipped path never ran the load. Eviction inside an arm needs no such
        // rule — the value's slot was written at its definition, which every
        // path reaching any of its readers ran.
        let arms = guarded_arms(&guards, dag.len());
        let sites = guard_sites(&guards, dag.len());

        // After `sites`: a guard's read of its mask is a read the pass has to
        // know about from the start (see `Pass::new`).
        let mut pass = Pass::new(&dag, file, vec_len, live_in, &sites);
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

            // A surviving `Reduce`'s own body is emitted through a wholly
            // separate, nested register allocation (`allocate_nest`'s fold
            // scope, recursed into via `Allocation::sibling`) that starts
            // fresh over the pool minus only what is carried into it -- it
            // has no visibility into what *this* scope currently holds
            // resident, and no reason not to reuse any of it. A value this
            // scope still needs after the
            // loop must therefore not be sitting in a register *across*
            // it: evict everything resident into its slot here, exactly as
            // a call to something that clobbers the whole register file
            // would force a caller to. `split_out` is the same eviction
            // every ordinary loser of the destination contest below goes
            // through (constants remat instead of spilling); the only
            // difference is that here it runs for every occupant at once,
            // pre-emptively, rather than one at a time as something else
            // claims the slot. Without this, `extract_folds`'s "a shared
            // invariant leaf stays in both places, recomputed" is only
            // true of the arena -- the register that held the outer
            // copy's result can be clobbered by the fold's own recompute
            // of the identical value, and whichever one the loop's last
            // iteration leaves behind is read back instead of the outer
            // scope's own answer.
            //
            // Not for a live-in `Reduce`: that is a placeholder for a loop an
            // enclosing scope already ran, and nothing runs here.
            //
            // A `Guard` earns the identical treatment for the identical
            // reason: its two arms are each a wholly separate, freshly
            // allocated scope (`LinearScan::allocate_nest`'s guard-arm loop)
            // that starts fresh over the pool minus only what is carried in,
            // with no visibility into what this scope holds resident and no
            // reason not to reuse any of it.
            if matches!(def.op, ScheduledOp::Reduce(..) | ScheduledOp::Guard(..))
                && !pass.live_in[def.value.0 as usize]
            {
                for slot in 0..pass.owner.len() {
                    if pass.owner[slot].is_some() {
                        pass.split_out(slot, i);
                    }
                }
            }

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

            // A live-in def emits nothing, so its encoding wants nothing —
            // a placeholder for an enclosing scope's fold would otherwise
            // reserve a loop's three temps for a loop that does not run here.
            let wanted = if pass.live_in[def.value.0 as usize] {
                0
            } else {
                (file.temps_for)(&def.op) as usize
            };
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
                // A value whose location was already settled *at this index* —
                // by a revert at an arm's end, or a demotion — is served by a
                // scratch reload here, never re-kept. `Pass::place` overwrites
                // a same-index range rather than appending one, so keeping it
                // would turn the `(i, Spilled)` just recorded into `(i, Reg)`:
                // on a guard's skipped path that register was never loaded,
                // and the emitter would copy from it as if it had been. This
                // was live — a value spilled before a guarded arm, kept inside
                // it, reverted at its end and read at the next arm's head
                // came back wrong on the skipped path at the floor.
                if pass.ranges[operand.0 as usize]
                    .last()
                    .is_some_and(|(at, _)| *at == i)
                {
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

            // **The destination, before anything that reads residency.** The
            // guard's mask register and the operand reloads below are counted
            // from which values are in a register *at this index*, and the
            // emitter counts them again from the same table when it resolves
            // the instruction's operands — so the two answers agree only if
            // nothing changes residency in between. The destination can: it
            // may take an operand's register. So it is decided first, and the
            // counts are taken from what it left.
            //
            // A definition is the one write in an instruction, and there is no
            // register outside the pool left to write to — so the loser of the
            // contest is the *occupant*, and the value being defined loses only
            // the right to *keep* what it was given. That is
            // `Where(v, def(v)) == Reg(_)`, which is what dissolves the fixed
            // destination register.
            //
            // Three definitions write nothing here and take no pool register:
            // a placeholder for a value an enclosing scope parked (its location
            // is that scope's answer — a fold's binder `Var` is one, and a
            // `Var` that is not parked names a binder no enclosing fold
            // binds, which no legal schedule has), a surviving `Reduce`'s
            // result (its accumulator's slot, where the loop leaves it
            // whether or not `allocate_nest` carried the accumulator across
            // the iterations — docs/plans/2026-09-10-a-surviving-reduce-is-a-loop.md,
            // "the design decision that makes this tractable"; the driver
            // pins the real address afterward through
            // `FrameLayout::pin_slot`), and the two unit-typed effects, a
            // `Write` and a `Seq`, which define no value at all. A
            // rematerialized constant that loses the contest is the fourth:
            // its definition emits no instruction, so it needs nothing to
            // write to.
            let destination: Option<usize> = if pass.live_in[def.value.0 as usize] {
                None
            } else if let ScheduledOp::Var(k) = def.op {
                panic!(
                    "Var({k}) is read by a scope no enclosing fold binds it in — \
                     a coordinate that survived `passes::lattice::collapse`, or a \
                     binder outside its fold"
                )
            } else if let ScheduledOp::Reduce(..)
            | ScheduledOp::Write { .. }
            | ScheduledOp::Seq(..)
            | ScheduledOp::Guard(..) = def.op
            {
                // A `Guard`'s result, the same way and for the same reason as
                // a `Reduce`'s: two different scopes (its two arms) each
                // store it, so no single one of them owns "the destination
                // register" — the driver dedicates it a slot instead (see
                // `ScheduledOp::Guard`'s doc and the module's non-goals).
                pass.place(def.value, i, Where::Spilled);
                None
            } else {
                // What each value this instruction reads would cost to lose
                // its register *here* — see `ReadHere`. Membership, not the
                // read cursor: the kept reloads above advanced it past `i`.
                // An operand's tier is what `operand_sources` would route it
                // through were it the one non-resident: into `dst` costs no
                // further register; anything else needs a reload register the
                // same pool then has to find. A value read twice by one
                // instruction takes the harsher of its two tiers. A guard's
                // mask read before the instruction needs `guard_mask`, so it
                // is never the cheap kind.
                let ops: Vec<ValueId> = operands(&def.op).collect();
                let mut resident = [true; 3];
                for (k, operand) in ops.iter().enumerate() {
                    resident[k] = pass.is_resident(*operand);
                }
                let mut read_here: Vec<(ValueId, ReadHere)> = Vec::new();
                let mut note = |v: ValueId, tier: ReadHere| match read_here
                    .iter_mut()
                    .find(|(r, _)| *r == v)
                {
                    Some((_, held)) => *held = (*held).max(tier),
                    None => read_here.push((v, tier)),
                };
                for (k, operand) in ops.iter().enumerate() {
                    let mut without = resident;
                    without[k] = false;
                    let tier = match operand_sources(&def.op, without)[k] {
                        OperandSource::Destination => ReadHere::FromDst,
                        OperandSource::Reload(_) => ReadHere::NeedsRegister,
                        OperandSource::Resident => {
                            unreachable!("an operand forced non-resident is not Resident")
                        }
                    };
                    note(*operand, tier);
                }
                for mask in &sites[i] {
                    note(*mask, ReadHere::NeedsRegister);
                }

                let open = pass.open(&taken);
                if let Some(free) = open.iter().copied().find(|k| pass.owner[*k].is_none()) {
                    pass.occupy(def.value, free, i);
                    Some(free)
                } else {
                    let slot = pass.loser(&open, i, &read_here).unwrap_or_else(|| {
                        unreachable!(
                            "the pool is at most this instruction's temps against a \
                             floor of {}, so some register is open and held",
                            RegisterFile::MIN_SCRATCH
                        )
                    });
                    let occupant =
                        pass.owner[slot].unwrap_or_else(|| unreachable!("a loser holds one"));
                    // Whether the new value keeps the register past this
                    // instruction, by the rule that chose the slot: its own
                    // rank against the occupant's. A definition has written
                    // nothing yet, so its slot is never the cheap kind.
                    let new_rank = EvictionRank {
                        // A definition is a write; nothing reads it here.
                        read_here: ReadHere::No,
                        traffic: if pass.const_bits[def.value.0 as usize].is_some() {
                            0
                        } else {
                            2
                        },
                        nearest: core::cmp::Reverse(
                            pass.next_read(def.value, i).map_or(usize::MAX, |r| r - i),
                        ),
                    };
                    let keeps = new_rank > pass.rank(occupant, i, &read_here);
                    if !keeps && pass.const_bits[def.value.0 as usize].is_some() {
                        // Nothing to write: the definition of a rematerialized
                        // constant emits no instruction, so it takes no
                        // register and evicts no one. Reserving one for it
                        // would cost a live value its register to hold a value
                        // the emitter never computes.
                        pass.place(def.value, i, pass.out_of_register(def.value));
                        None
                    } else {
                        // Split at `i`, not `i + 1`: a displaced operand is
                        // reloaded by this instruction from the slot its
                        // definition wrote, and that reload is what the
                        // counts below have to see. (The register still holds
                        // it until the write, but no encoder relies on that —
                        // see `Pass::open`.)
                        pass.split_out(slot, i);
                        pass.occupy(def.value, slot, i);
                        if !keeps && i + 1 < dag.len() {
                            // Given a register to be written into and stored
                            // from — the store goes right after the definition,
                            // as it does for any value with a slot — and not to
                            // keep.
                            demotions[i + 1].push((def.value, slot));
                        }
                        Some(slot)
                    }
                }
            };
            // A role like any other from here on: the reloads and the guard's
            // registers may not land on it.
            if let Some(slot) = destination {
                taken.push(slot);
            }

            // A guard's own two registers, on the instruction it is emitted
            // before. The mask needs one only when it is not in a register
            // here — which the kept reloads, and now the destination, may
            // just have changed.
            //
            // A surviving `Reduce`'s own trip test needs exactly the same
            // thing (a mask reduced to a branch condition) and is emitted
            // the same way, in place of this instruction — so it reserves
            // through the same gate rather than a second one, even though
            // `sites[i]` (built from `Select`s alone) never names it.
            // A `Guard`'s own branch needs exactly the same two registers,
            // for exactly the same reason, at exactly the same point — its
            // mask test and branch are emitted in place of this instruction
            // too.
            let is_reduce_or_guard =
                matches!(def.op, ScheduledOp::Reduce(..) | ScheduledOp::Guard(..));
            if !sites[i].is_empty() || is_reduce_or_guard {
                // A `Reduce`'s own trip test builds its mask fresh into a
                // temp every time (`t0` in `emit_scope`'s loop, never a
                // reload of some value already computed elsewhere), so
                // `sites[i]` — the only other source `guard_mask` answers
                // for — is what decides here for both: empty for a `Reduce`
                // (it never asks), and, for a `Guard`, its own one operand,
                // added because `sites` is built from `Select`s alone and
                // does not already name it.
                let guard_op_mask = match def.op {
                    ScheduledOp::Guard(mask, ..) => Some(mask),
                    _ => None,
                };
                if sites[i].iter().any(|m| !pass.is_resident(*m))
                    || guard_op_mask.is_some_and(|m| !pass.is_resident(m))
                {
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
            // by the same function the emitter reads. Residency is final here
            // because the destination was chosen above: eviction splits a
            // range rather than rewriting one, so an operand in a register now
            // is in a register when this instruction is emitted, and an
            // operand the destination displaced is already out of one.
            let mut resident = [true; 3];
            for (k, operand) in operands(&def.op).enumerate() {
                resident[k] = pass.is_resident(operand);
            }
            let sources = operand_sources(&def.op, resident);
            for role in 0..reloads_wanted(sources) {
                scratch_for[i].reloads[role] = Some(pass.reserve(i, &mut taken, &live_here));
            }

            // The scope's result is materialized after its last instruction,
            // and it needs a register of its own in two cases: the whole
            // body was hoisted out, so its root is read from a park rather
            // than computed, or a surviving fold's own `Reduce` is the
            // schedule's root, whose accumulator is a slot by construction
            // (see the destination above) — both never resident for the same
            // reason, a value with no register to be the "last instruction's
            // own destination" in. Every other root is exactly that
            // destination. A root that is no value at all — a `Write`, a
            // `Seq`, a fold over the unit monoid — has nothing to
            // materialize.
            let unit_root = match &def.op {
                ScheduledOp::Write { .. } | ScheduledOp::Seq(..) => true,
                ScheduledOp::Reduce(fold, _) => fold.monoid() == pixelflow_ir::fold::Monoid::SEQ,
                _ => false,
            };
            if i + 1 == dag.len()
                && !unit_root
                && !pass.is_resident(def.value)
                && (pass.live_in[def.value.0 as usize]
                    || matches!(def.op, ScheduledOp::Reduce(..) | ScheduledOp::Guard(..)))
            {
                scratch_for[i].result = Some(pass.reserve(i, &mut taken, &live_here));
            }
        }

        Scan {
            schedule: dag,
            ranges: pass.ranges,
            scratch: scratch_for,
            guards,
        }
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

/// The values an operation reads *as registers*, in operand order.
///
/// A `Reduce` is a leaf here, the same as `Uniform` — by the time one reaches
/// a schedule this function walks, `extract_folds` has already carved its
/// body out into its own `ScopeFold`; the `ValueId` `ScheduledOp::Reduce`
/// still carries is `schedule_variance`'s and `extract_folds`'s own concern
/// (they run before extraction, and after respectively, over different
/// schedules), never an operand this scope's allocation resolves. What the
/// loop it opens reads from this scope is a dependency all the same, and the
/// guard analysis has it as one: [`FoldReads`]. A `Seq`
/// sequences two effects and reads no register; a `Write` reads the one
/// value it stores — its row and column are binders, found where their
/// folds keep them, not operands.
pub(crate) fn operands(sop: &ScheduledOp) -> impl Iterator<Item = ValueId> + use<'_> {
    let (a, b, c) = match sop {
        ScheduledOp::Var(_)
        | ScheduledOp::Lanes(_)
        | ScheduledOp::Const(_)
        | ScheduledOp::Uniform(_)
        | ScheduledOp::Reduce(..)
        | ScheduledOp::Seq(..) => (None, None, None),
        ScheduledOp::Unary(_, a) | ScheduledOp::ShiftImm(_, a, _) | ScheduledOp::Gather(a, _) => {
            (Some(*a), None, None)
        }
        ScheduledOp::Write { value, .. } => (Some(*value), None, None),
        // A `Guard`'s mask is the one register operand its own def reads —
        // its two arms are names into a wholly separate arena, not values in
        // this schedule, exactly as a `Reduce`'s body is not (see the doc
        // above) but without even that much: an arm's operands are its own
        // scope's concern (`LinearScan::allocate_nest`'s guard-arm loop),
        // never this scope's.
        ScheduledOp::Guard(mask, _, _) => (Some(*mask), None, None),
        ScheduledOp::Binary(_, a, b) => (Some(*a), Some(*b), None),
        ScheduledOp::Ternary(_, a, b, c) => (Some(*a), Some(*b), Some(*c)),
    };
    [a, b, c].into_iter().flatten()
}

/// Every value an operation is *built from*: its register operands, plus the
/// body a `Reduce` folds and the two effects a `Seq` orders — the children a
/// walk of the DAG's structure follows, as opposed to the registers an
/// instruction reads ([`operands`]).
pub(crate) fn structural_children(sop: &ScheduledOp) -> impl Iterator<Item = ValueId> + use<'_> {
    let extra = match sop {
        ScheduledOp::Reduce(_, body) => [Some(*body), None],
        ScheduledOp::Seq(a, b) => [Some(*a), Some(*b)],
        _ => [None, None],
    };
    operands(sop).chain(extra.into_iter().flatten())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::kind::OpKind;

    // --- bit sets ---

    #[test]
    fn reg_set_of_contains_exactly_the_given_registers() {
        let s = RegSet::of(&[Reg(2), Reg(5)]);
        assert!(s.contains(Reg(2)));
        assert!(s.contains(Reg(5)));
        assert!(!s.contains(Reg(3)));
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn reg_set_range_is_the_contiguous_run_it_names() {
        let s = RegSet::range(4, 3);
        assert!(!s.contains(Reg(3)));
        assert!(s.contains(Reg(4)));
        assert!(s.contains(Reg(5)));
        assert!(s.contains(Reg(6)));
        assert!(!s.contains(Reg(7)));
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn reg_set_union_holds_every_member_of_both_sets_and_nothing_else() {
        let u = RegSet::of(&[Reg(1)]).union(RegSet::of(&[Reg(2)]));
        assert!(u.contains(Reg(1)));
        assert!(u.contains(Reg(2)));
        assert!(!u.contains(Reg(3)));
        assert_eq!(u.len(), 2);

        // A member shared by both operands must survive the union: `^`
        // would cancel it out where `|` keeps it.
        let shared = RegSet::of(&[Reg(1), Reg(2)]).union(RegSet::of(&[Reg(2), Reg(3)]));
        assert_eq!(shared, RegSet::of(&[Reg(1), Reg(2), Reg(3)]));
    }

    #[test]
    fn reg_set_without_removes_only_the_named_members() {
        let d = RegSet::of(&[Reg(1), Reg(2), Reg(3)]).without(RegSet::of(&[Reg(2)]));
        assert!(d.contains(Reg(1)));
        assert!(!d.contains(Reg(2)));
        assert!(d.contains(Reg(3)));
    }

    #[test]
    fn reg_set_contains_is_false_one_past_its_highest_member() {
        let s = RegSet::of(&[Reg(31)]);
        assert!(s.contains(Reg(31)));
        assert!(!s.contains(Reg(30)));
        // Reg(31) is the highest bit a 32-register file can hold; a set
        // holding it must not treat Reg(32) as a member too (an `<=` bound
        // check would shift by 32, which is out of range for a `u32`).
        assert!(!s.contains(Reg(32)));
    }

    #[test]
    fn reg_set_is_empty_is_true_only_for_the_empty_set() {
        assert!(RegSet::EMPTY.is_empty());
        assert!(!RegSet::of(&[Reg(0)]).is_empty());
    }

    #[test]
    fn reg_set_take_keeps_only_the_lowest_n_members() {
        let t = RegSet::of(&[Reg(1), Reg(3), Reg(5)]).take(2);
        assert!(t.contains(Reg(1)));
        assert!(t.contains(Reg(3)));
        assert!(!t.contains(Reg(5)));
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn reg_set_take_beyond_its_size_returns_the_whole_set() {
        let s = RegSet::of(&[Reg(1), Reg(3)]);
        assert_eq!(s.take(10), s);
    }

    #[test]
    fn reg_set_iter_yields_every_member_low_to_high() {
        let s = RegSet::of(&[Reg(5), Reg(1), Reg(3)]);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![Reg(1), Reg(3), Reg(5)]);
    }

    #[test]
    fn gpr_set_of_contains_exactly_the_given_registers() {
        let s = GprSet::of(&[Gpr(2), Gpr(5)]);
        assert!(s.contains(Gpr(2)));
        assert!(s.contains(Gpr(5)));
        assert!(!s.contains(Gpr(3)));
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn gpr_set_contains_is_false_one_past_its_highest_member() {
        let s = GprSet::of(&[Gpr(31)]);
        assert!(s.contains(Gpr(31)));
        assert!(!s.contains(Gpr(30)));
        assert!(!s.contains(Gpr(32)));
    }

    #[test]
    fn gpr_set_is_empty_is_true_only_for_the_empty_set() {
        assert!(GprSet::EMPTY.is_empty());
        assert!(!GprSet::of(&[Gpr(0)]).is_empty());
    }

    #[test]
    fn gpr_set_iter_yields_every_member_low_to_high() {
        let s = GprSet::of(&[Gpr(5), Gpr(1)]);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![Gpr(1), Gpr(5)]);
    }

    #[test]
    fn mask_set_of_contains_exactly_the_given_registers() {
        let s = MaskSet::of(&[KReg(1), KReg(4)]);
        assert!(s.contains(KReg(1)));
        assert!(s.contains(KReg(4)));
        assert!(!s.contains(KReg(2)));
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn mask_set_contains_is_false_one_past_its_highest_member() {
        let s = MaskSet::of(&[KReg(7)]);
        assert!(s.contains(KReg(7)));
        assert!(!s.contains(KReg(6)));
        assert!(!s.contains(KReg(8)));
    }

    #[test]
    fn mask_set_is_empty_is_true_only_for_the_empty_set() {
        assert!(MaskSet::EMPTY.is_empty());
        assert!(!MaskSet::of(&[KReg(0)]).is_empty());
    }

    #[test]
    fn mask_set_iter_yields_every_member_low_to_high() {
        let s = MaskSet::of(&[KReg(3), KReg(0)]);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![KReg(0), KReg(3)]);
    }

    /// The smallest pool a register file may declare, so pressure tests need
    /// only a handful of values to reach spilling.
    const TEST_FILE: RegisterFile = RegisterFile {
        fixed: &[],
        scratch: RegSet::range(4, RegisterFile::MIN_SCRATCH),
        temps_for: no_temps,
        guard_temps: 0,
        vector_bytes: 16,
        gpr_ctx: None,
        gpr_out: None,
        gpr_pitch: None,
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

    /// A leaf every schedule here starts from.
    ///
    /// It used to be `Var(0)`, a coordinate pinned to an input register. The
    /// collapse ABI passes no vectors, so a `Var` reaching an allocation names
    /// a fold binder and nothing else; a uniform's broadcast load is what a
    /// loop-free schedule's leaf is.
    fn leaf(value: u32) -> Def {
        def(
            value,
            ScheduledOp::Uniform(crate::emit::UniformLoad {
                ctx_slot: 0,
                offset: value as u16,
            }),
        )
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

    /// `v2 = v0 + v1`, over two leaves.
    fn add_two_leaves() -> Vec<Def> {
        vec![
            leaf(0),
            leaf(1),
            def(2, ScheduledOp::Binary(OpKind::Add, ValueId(0), ValueId(1))),
        ]
    }

    // --- scan internals: record, guarded_arms, Reservations, Pass ---

    /// A second placement recorded at the same schedule index replaces the
    /// first rather than appending a range: an eviction that puts a value
    /// back where it already was is not a move, and a repeated `from` would
    /// break `Placement`'s strictly-increasing invariant.
    #[test]
    fn record_collapses_consecutive_ranges_at_the_same_index_into_one() {
        let scan = Scan {
            schedule: Vec::new(),
            ranges: vec![vec![(0, Where::Reg(Reg(4))), (2, Where::Reg(Reg(4)))]],
            scratch: Vec::new(),
            guards: Vec::new(),
        };
        let placements = record(&scan, &BTreeMap::new());
        let placement = placements[0].as_ref().expect("value 0 was placed");
        assert_eq!(
            placement.spans().collect::<Vec<_>>(),
            vec![Span {
                from: Point { index: 0 },
                at: Where::Reg(Reg(4)),
            }],
            "an eviction that put the value back in the same register it \
             already held is not a move, so the second entry must not add \
             a second span"
        );
    }

    /// The narrowest arm containing an index wins, not the first or the
    /// widest: ending a kept reload at the inner arm's end is safe under an
    /// outer one too, so the narrower answer is always the safe one to keep.
    #[test]
    fn guarded_arms_prefers_the_narrowest_covering_arm() {
        use super::super::guards::ArmPair;
        let guard = |select_idx: usize, mask: u32, true_arm: (usize, usize)| SelectGuard {
            select_idx,
            mask_vid: ValueId(mask),
            ranges: ArmPair::new(true_arm, (0, 0)),
        };
        // A clear case (width 10 vs. width 2): the narrower arm wins.
        let wide = guard(100, 0, (0, 10));
        let narrow = guard(101, 1, (3, 5));
        // A genuine tie (width 2 each, processed first): the earlier one is
        // kept rather than overwritten — distinguishes `<` from `<=`/`==`.
        let tie_a = guard(102, 2, (15, 17));
        let tie_b = guard(103, 3, (16, 18));
        // A clear case the other way (width 3 vs. width 1): distinguishes
        // `<` from `>`, which the tie case above cannot (both agree there).
        let wider = guard(104, 4, (25, 28));
        let narrower = guard(105, 5, (26, 27));
        // The new arm's own width (`end - start`): distinguishes `-` from
        // `+`/`/` on that computation specifically.
        let own_width_prior = guard(106, 6, (35, 37));
        let own_width_narrower = guard(107, 7, (36, 37));
        // The stored arm's width (`e - s`): a tie (width 4 each) that must
        // stay with the first-recorded arm — distinguishes `-` from `+`/`/`
        // on *that* computation, which the tie case above cannot (it never
        // exercises a stored span with a nonzero `s`).
        let stored_width_prior = guard(108, 8, (41, 45));
        let stored_width_current = guard(109, 9, (40, 44));

        let arms = guarded_arms(
            &[
                wide,
                narrow,
                tie_a,
                tie_b,
                wider,
                narrower,
                own_width_prior,
                own_width_narrower,
                stored_width_prior,
                stored_width_current,
            ],
            46,
        );

        for (i, arm) in arms.iter().enumerate().take(10) {
            let expected = if (3..5).contains(&i) {
                Some((3, 5))
            } else {
                Some((0, 10))
            };
            assert_eq!(*arm, expected, "index {i}: narrowest of a clear pair");
        }
        assert_eq!(arms[15], Some((15, 17)), "only tie_a covers index 15");
        assert_eq!(
            arms[16],
            Some((15, 17)),
            "tie_a and tie_b tie in width at index 16; the first recorded must stand"
        );
        assert_eq!(arms[17], Some((16, 18)), "only tie_b covers index 17");
        assert_eq!(
            arms[25],
            Some((25, 28)),
            "only the wider arm covers index 25"
        );
        assert_eq!(
            arms[26],
            Some((26, 27)),
            "the strictly narrower arm must win at index 26"
        );
        assert_eq!(
            arms[27],
            Some((25, 28)),
            "only the wider arm covers index 27"
        );
        assert_eq!(
            arms[35],
            Some((35, 37)),
            "only the first arm covers index 35"
        );
        assert_eq!(
            arms[36],
            Some((36, 37)),
            "the new arm's own width (1) must beat the stored one (2) at index 36"
        );
        for (i, arm) in arms.iter().enumerate().take(44).skip(41) {
            assert_eq!(
                *arm,
                Some((41, 45)),
                "index {i}: tied stored width (4 each) keeps the first-recorded arm"
            );
        }
        assert_eq!(
            arms[40],
            Some((40, 44)),
            "only the second arm covers index 40"
        );
        assert_eq!(
            arms[44],
            Some((41, 45)),
            "only the first arm covers index 44"
        );
    }

    /// `ROLES` is guard mask, guard temp, result and the destination on top
    /// of the widest encoding's temps and reloads — not any other mix of the
    /// same numbers.
    #[test]
    fn reservations_roles_covers_temps_reloads_and_the_four_named_slots() {
        assert_eq!(
            Reservations::ROLES,
            10,
            "Scratch::MAX_TEMPS (4) + Scratch::MAX_RELOADS (2) + 4 named roles"
        );
    }

    /// `rank`'s distance is measured forward from the point asked about, not
    /// from the value's own definition.
    #[test]
    fn rank_measures_distance_from_the_point_asked_about() {
        let dag = vec![
            leaf(0),
            leaf(1),
            leaf(2),
            def(3, ScheduledOp::Unary(OpKind::Neg, ValueId(0))),
        ];
        let sites = vec![Vec::new(); dag.len()];
        let mut pass = Pass::new(&dag, &TEST_FILE, dag.len(), &BTreeMap::new(), &sites);
        let rank = pass.rank(ValueId(0), 1, &[]);
        assert_eq!(
            rank.nearest.0, 2,
            "value 0's only read is at index 3, two steps ahead of index 1"
        );
    }

    /// A second `place` at the same index overwrites the first, matching
    /// `record`'s own rule for the ranges it consumes.
    #[test]
    fn place_overwrites_a_range_recorded_at_the_same_index() {
        let dag = vec![leaf(0)];
        let sites = vec![Vec::new(); 1];
        let mut pass = Pass::new(&dag, &TEST_FILE, 1, &BTreeMap::new(), &sites);
        pass.place(ValueId(0), 3, Where::Reg(Reg(4)));
        pass.place(ValueId(0), 3, Where::Spilled);
        assert_eq!(
            pass.ranges[0],
            vec![(3, Where::Spilled)],
            "the second placement at index 3 must replace the first, not add a range"
        );
    }

    /// `place` marks a value's slot as already valid in memory exactly when
    /// it places it `Spilled` — never for a register or a remat, and it
    /// never un-marks a value memory has already seen once.
    #[test]
    fn place_marks_in_slot_only_when_placing_spilled() {
        let dag = vec![leaf(0), leaf(1)];
        let sites = vec![Vec::new(); 2];
        let mut pass = Pass::new(&dag, &TEST_FILE, 2, &BTreeMap::new(), &sites);
        pass.place(ValueId(0), 0, Where::Reg(Reg(4)));
        assert!(
            !pass.in_slot[0],
            "landing in a register is not a reason to believe memory is valid"
        );
        pass.place(ValueId(1), 0, Where::Spilled);
        assert!(
            pass.in_slot[1],
            "placing a value Spilled is exactly what makes its slot valid"
        );
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
        let a = alloc(add_two_leaves());
        let order: Vec<ValueId> = a.body().schedule().iter().map(|d| d.value).collect();
        assert_eq!(order, vec![ValueId(0), ValueId(1), ValueId(2)]);
    }

    /// Every value takes a pool register.
    ///
    /// There is no pre-colored input any more: the collapse ABI passes
    /// pointers and a pitch, never a vector, so `xmm0-3` / `v0-v3` are the
    /// allocator's like every other register the ABI does not preserve.
    #[test]
    fn every_value_takes_a_pool_register() {
        let a = alloc(add_two_leaves());
        for d in a.body().schedule() {
            let Where::Reg(r) = at(&a, d.value) else {
                panic!("{:?} is not in a register", d.value)
            };
            assert!(TEST_FILE.scratch.contains(r), "{r:?} is not the pool's");
        }
        assert_eq!(spill_count(&a), 0);
    }

    /// Every value the emitter walks has a placement — the invariant that used
    /// to be spread across three parallel maps and checked at runtime.
    #[test]
    fn placement_is_total_over_the_schedule() {
        let a = alloc(add_two_leaves());
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

    /// A `Var` no enclosing fold binds is refused: either a coordinate that
    /// survived the lattice's own folds, or a binder read outside its loop.
    #[test]
    #[should_panic(expected = "a coordinate that survived")]
    fn a_var_no_enclosing_fold_binds_is_refused() {
        let _ = alloc(vec![def(0, ScheduledOp::Var(4))]);
    }

    /// Two values whose live ranges do not overlap may share one register —
    /// that is the whole point of tracking last use rather than defs.
    #[test]
    fn disjoint_live_ranges_share_a_register() {
        let a = alloc(vec![
            leaf(0),
            leaf(1),
            def(2, ScheduledOp::Binary(OpKind::Add, ValueId(0), ValueId(1))),
            def(3, ScheduledOp::Unary(OpKind::Neg, ValueId(2))),
            def(4, ScheduledOp::Unary(OpKind::Abs, ValueId(3))),
        ]);
        assert_eq!(spill_count(&a), 0);
        // Five values, never five of them live at once: the leaves die at
        // the sum, the sum at the negation. The allocator takes the lowest
        // free register, so a range that ended hands its register on — which
        // is what tracking last use rather than defs buys.
        let mut used: Vec<u8> = a
            .body()
            .schedule()
            .iter()
            .map(|d| match at(&a, d.value) {
                Where::Reg(r) => r.0,
                other => panic!("{:?} is at {other:?}, not in a register", d.value),
            })
            .collect();
        let values = used.len();
        used.sort_unstable();
        used.dedup();
        assert!(
            used.len() < values,
            "{values} values took {} distinct registers, so no disjoint pair \
             shared one",
            used.len()
        );
    }

    // -------------------------------------------------------------------------
    // Instruction temps
    // -------------------------------------------------------------------------

    /// The temp is a real register from the pool, and it is nobody's operand
    /// and not the destination — the encoder writes it while the operands are
    /// still live and before it writes `dst`.
    #[test]
    fn a_temp_collides_with_neither_the_operand_nor_the_destination() {
        let schedule = vec![leaf(0), def(1, ScheduledOp::Unary(OpKind::Neg, ValueId(0)))];
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
            leaf(0),
            leaf(1),
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

    /// `select_guards` is the guard analysis on file, not an empty stand-in:
    /// a schedule with a genuine exclusive arm reports it back unchanged.
    #[test]
    fn select_guards_reports_the_arm_the_schedule_actually_has() {
        let schedule = vec![
            leaf(0),
            leaf(1),
            def(2, ScheduledOp::Unary(OpKind::Rsqrt, ValueId(1))),
            def(
                3,
                ScheduledOp::Ternary(OpKind::Select, ValueId(0), ValueId(2), ValueId(0)),
            ),
        ];
        let a = alloc(schedule);
        let guards = a.body().select_guards();
        assert_eq!(guards.len(), 1, "the schedule has exactly one Select");
        assert_eq!(guards[0].select_idx, 3);
        assert_eq!(guards[0].mask_vid, ValueId(0));
        assert_eq!(guards[0].true_range(), (1, 3));
        assert_eq!(guards[0].false_range(), (3, 3));
    }

    /// A spilled operand read inside a guarded arm is *not* worth promoting
    /// back into a register when its very next read is at or after the arm's
    /// end: the promotion would not even reach the read it was for, so an
    /// ordinary reload serves it just as well.
    ///
    /// `X` is forced to spill under pressure (seven fillers exactly fill
    /// `TEST_FILE`'s pool, and `X`'s own next read — deep inside the arm — is
    /// the farthest among the candidates, so `X` is the one evicted). The
    /// arm then reads `X` twice: once inside it, and again as the `Select`'s
    /// own false-arm operand — a read at exactly the arm's end.
    #[test]
    fn a_spilled_operand_read_again_only_at_the_arms_end_is_not_kept() {
        let x = ValueId(1);
        let mut schedule = vec![leaf(0), def(1, ScheduledOp::Unary(OpKind::Neg, ValueId(0)))];
        let fillers: Vec<u32> = (10..16).collect(); // f1..f6.
        for &f in &fillers {
            schedule.push(def(f, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        }
        schedule.push(def(16, ScheduledOp::Unary(OpKind::Neg, ValueId(0)))); // f7: 8th value, forces eviction.
        let eviction_index = schedule.len() - 1; // index 8.
        for (i, &f) in fillers.iter().enumerate() {
            schedule.push(def(
                100 + i as u32,
                ScheduledOp::Unary(OpKind::Neg, ValueId(f)),
            )); // dist 1..6.
        }
        schedule.push(leaf(50)); // mask, index 15.
        let rsqrt_index = schedule.len();
        schedule.push(def(60, ScheduledOp::Unary(OpKind::Rsqrt, x))); // index 16: reads X.
        let select_index = schedule.len();
        schedule.push(def(
            70,
            ScheduledOp::Ternary(OpKind::Select, ValueId(50), ValueId(60), x), // X again, as the false arm.
        ));

        let a = alloc(schedule);
        let guards = a.body().select_guards();
        assert_eq!(
            guards
                .iter()
                .find(|g| g.select_idx == select_index)
                .map(SelectGuard::true_range),
            Some((rsqrt_index, select_index)),
            "fixture assumes the Rsqrt alone forms the true arm's exclusive range"
        );
        assert_eq!(
            a.body().where_at(x, eviction_index),
            Where::Spilled,
            "fixture assumes X, not a filler, is the one evicted"
        );
        assert_eq!(
            a.body().where_at(x, rsqrt_index),
            Where::Spilled,
            "X's next read (the Select's own false arm) lands exactly at the \
             arm's end, so keeping X in a register here would not even reach \
             it — an ordinary reload serves the Rsqrt instead"
        );
    }

    /// A value defined *at* a guarded arm's own first instruction is arm-
    /// internal, not something the arm merely reads from outside — so even
    /// once pressure evicts and then re-promotes it within the same arm, it
    /// gets no revert scheduled at the arm's end: nothing outside the arm
    /// ever reads it, by the same exclusivity that put it in the arm at all.
    ///
    /// The pressure has to come from *inside* the arm: an external filler's
    /// own read is necessarily scheduled after the whole arm (anything it
    /// reads earlier would break the arm's contiguous range), which always
    /// makes it farther out than a within-arm read and so never a genuine
    /// rival for `loser`. Eight independent leaves feeding one reduction give
    /// the arm its own pressure — `Y` is the first leaf (so `defined_at`
    /// equals the arm's `start` exactly) but the last one consumed, so it is
    /// the one `loser` evicts when the eighth leaf needs a register. `Y` is
    /// then read twice more, and directly again right before the `Select` —
    /// that last read is what keeps it resident (protected as an operand)
    /// all the way to the arm's end without any other reason to hold it,
    /// which is what makes a spurious revert there observable.
    #[test]
    fn a_value_defined_at_the_arms_own_start_needs_no_revert_at_its_end() {
        let y = ValueId(1);
        let mut schedule = vec![leaf(0), def(1, ScheduledOp::Unary(OpKind::Neg, ValueId(0)))];
        let leaves: Vec<u32> = (10..17).collect(); // 7 more leaves alongside Y: 8 live at once.
        for &l in &leaves {
            schedule.push(def(l, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        }
        let eviction_index = schedule.len() - 1; // the 8th leaf: pool is full, so this evicts Y.
        // Pair the other seven leaves off soonest-first, so Y — read only by
        // the last pair — is the farthest-out candidate when the 8th leaf
        // needs a register.
        let p1 = 200;
        schedule.push(def(
            p1,
            ScheduledOp::Binary(OpKind::Add, ValueId(leaves[0]), ValueId(leaves[1])),
        ));
        let p2 = 201;
        schedule.push(def(
            p2,
            ScheduledOp::Binary(OpKind::Add, ValueId(leaves[2]), ValueId(leaves[3])),
        ));
        let p3 = 202;
        schedule.push(def(
            p3,
            ScheduledOp::Binary(OpKind::Add, ValueId(leaves[4]), ValueId(leaves[5])),
        ));
        let p4 = 203; // Y's first re-read, and the farthest among the pairs' operands at eviction time.
        schedule.push(def(
            p4,
            ScheduledOp::Binary(OpKind::Add, y, ValueId(leaves[6])),
        ));
        let extra = 204; // Y's second re-read.
        schedule.push(def(extra, ScheduledOp::Binary(OpKind::Add, ValueId(p4), y)));
        let q1 = 210;
        schedule.push(def(
            q1,
            ScheduledOp::Binary(OpKind::Add, ValueId(p1), ValueId(p2)),
        ));
        let q2 = 211;
        schedule.push(def(
            q2,
            ScheduledOp::Binary(OpKind::Add, ValueId(p3), ValueId(extra)),
        ));
        let q3 = 212;
        schedule.push(def(
            q3,
            ScheduledOp::Binary(OpKind::Add, ValueId(q1), ValueId(q2)),
        ));
        let root = 220; // Y's third re-read, right before the Select.
        schedule.push(def(root, ScheduledOp::Binary(OpKind::Add, ValueId(q3), y)));
        let select_index = schedule.len();
        schedule.push(def(
            70,
            ScheduledOp::Ternary(OpKind::Select, ValueId(0), ValueId(root), ValueId(0)),
        ));

        let a = alloc(schedule);
        let guards = a.body().select_guards();
        assert_eq!(
            guards
                .iter()
                .find(|g| g.select_idx == select_index)
                .map(SelectGuard::true_range),
            Some((1, select_index)),
            "fixture assumes the whole reduction, starting at Y's own \
             definition, is the true arm's exclusive range"
        );
        assert_eq!(
            a.body().where_at(y, eviction_index),
            Where::Spilled,
            "fixture assumes Y, not one of the other leaves, is the one evicted"
        );
        assert!(
            matches!(a.body().where_at(y, select_index), Where::Reg(_)),
            "Y is arm-internal from its own definition on, so re-promoting it \
             inside the arm needs no revert at the arm's end — it should \
             still be resident at the Select"
        );
    }

    /// Nothing is reserved for an operand that is already in a register.
    #[test]
    fn a_resident_operand_reserves_no_reload_target() {
        let a = LinearScan.allocate(add_two_leaves(), &TEMP_FILE);
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
        // computed value, so the temp has to miss a pool register that is
        // already holding something.
        let a = LinearScan.allocate(
            vec![
                leaf(0),
                leaf(1),
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
        let a = LinearScan.allocate(add_two_leaves(), &TEMP_FILE);
        assert_eq!(a.body().scratch(2).temp(0), None, "`Add` asked for no temp");
    }

    /// The temp's live range is one instruction: the next one may take the
    /// same register for its result.
    #[test]
    fn a_temp_is_free_again_at_the_next_instruction() {
        let schedule = vec![
            leaf(0),
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
        let mut schedule = vec![leaf(0)];
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
        let mut schedule = vec![leaf(0)];
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

    /// A constant that loses its own keep contest never touches a register at
    /// all: it is rebuilt at each use, so eviction never has to run for it.
    ///
    /// Seven fillers fill `TEST_FILE`'s pool exactly, each with a read late
    /// enough to survive to the constant's own definition (an unread value
    /// would simply expire, never reaching a contest at all); the eighth
    /// definition, a constant, forces an eviction, and its own contest is
    /// decided by `traffic` alone (0 for a constant against 2 for the
    /// filler), regardless of how the reads are staggered.
    #[test]
    fn a_constant_that_loses_its_keep_contest_is_never_given_a_register() {
        // `leaf(0)` occupies a pool register of its own and stays live for
        // every filler's definition, so the set that fills `TEST_FILE`'s pool
        // exactly is the leaf plus `MIN_SCRATCH - 1` fillers. It used to be
        // `MIN_SCRATCH` of them, back when the leaf was a coordinate sitting
        // in an input register outside the pool; the collapse ABI passes no
        // vectors, so there is no such register any more.
        let pool = RegisterFile::MIN_SCRATCH as u32;
        let mut schedule = vec![leaf(0)];
        let fillers: Vec<u32> = (10..10 + pool - 1).collect();
        for &f in &fillers {
            schedule.push(def(f, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        }
        schedule.push(def(90, ScheduledOp::Const(99.0)));
        // The pool is full, so this definition evicts someone.
        let const_def_index = schedule.len() - 1;
        // The leaf is read once more, first and nearest, for two reasons: it
        // is still live at the constant's definition (so the pool really is
        // full there, and a contest really does run), and its next read is
        // the soonest of anyone's (so it is not the occupant the constant
        // evicts — a filler is).
        schedule.push(def(99, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        // Every filler reads once, after the constant's own definition, so
        // none of them expire before the constant's contest runs.
        for (i, &f) in fillers.iter().enumerate() {
            schedule.push(def(
                100 + i as u32,
                ScheduledOp::Unary(OpKind::Neg, ValueId(f)),
            ));
        }
        let a = alloc(schedule);
        assert_eq!(
            a.body().where_at(ValueId(90), const_def_index),
            Where::Remat(99.0f32.to_bits()),
            "the constant should never occupy a register even at its own \
             definition, let alone evict the filler it forced open"
        );
    }

    /// A destination that forces an eviction to be written keeps the register
    /// past its own instruction only when its own next read beats what the
    /// occupant it evicted offered — not merely because it forced room open.
    ///
    /// Seven fillers fill `TEST_FILE`'s pool exactly; an eighth definition
    /// forces an eviction, and the evicted occupant (`f7`, farthest among the
    /// fillers) sets the bar the new definition has to beat. The check runs
    /// at a `Var` spacer right after `new_val`'s own definition — not the
    /// literal next instruction, which would otherwise need a destination of
    /// its own and could evict `new_val` on that unrelated contest, masking
    /// whether *this* one demoted it.
    #[test]
    fn a_destination_that_forces_an_eviction_but_reads_later_than_the_occupant_is_spilled_next() {
        // `leaf(0)` occupies a pool register of its own and stays live for
        // every filler's definition, so the set that fills `TEST_FILE`'s pool
        // exactly is the leaf plus `MIN_SCRATCH - 1` fillers. It used to be
        // `MIN_SCRATCH` of them, back when the leaf was a coordinate sitting
        // in an input register outside the pool; the collapse ABI passes no
        // vectors, so there is no such register any more.
        let pool = RegisterFile::MIN_SCRATCH as u32;
        let mut schedule = vec![leaf(0)];
        let fillers: Vec<u32> = (10..10 + pool - 1).collect();
        for &f in &fillers {
            schedule.push(def(f, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        }
        let new_val = 90;
        schedule.push(def(new_val, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        // The spacer is a `Seq`: it defines no value and so takes no pool
        // register, which is what this check needs and what a `leaf` spacer
        // can no longer be.
        schedule.push(def(91, ScheduledOp::Seq(ValueId(0), ValueId(0))));
        let check_index = schedule.len() - 1;
        // f1..f7 read once each, staggered — f1 soonest, f7 last of the
        // fillers (distance 7 from `new_val`'s own definition) — and
        // `new_val`'s own read even later still (distance 9), so it is used
        // farther out than the occupant (`f7`, distance 8) that `loser`
        // picks to evict for it. The check runs at the spacer right after
        // `new_val`'s own definition, not the literal next instruction,
        // which would otherwise need a destination of its own and could
        // evict `new_val` on that unrelated contest, masking whether *this*
        // one demoted it.
        for (i, &f) in fillers.iter().enumerate() {
            schedule.push(def(
                100 + i as u32,
                ScheduledOp::Unary(OpKind::Neg, ValueId(f)),
            ));
        }
        schedule.push(def(200, ScheduledOp::Unary(OpKind::Neg, ValueId(new_val))));

        let a = alloc(schedule);
        assert_eq!(
            a.body().where_at(ValueId(new_val), check_index),
            Where::Spilled,
            "new_val forced f7's eviction to be written, but its own next \
             read is even farther out than f7's was, so it does not keep \
             the register past its own instruction"
        );
    }

    /// The mirror image of the above: a destination whose own next read beats
    /// the occupant's keeps the register, so it is not queued for a demotion
    /// at all.
    #[test]
    fn a_destination_that_reads_sooner_than_the_occupant_keeps_its_register() {
        // `leaf(0)` occupies a pool register of its own and stays live for
        // every filler's definition, so the set that fills `TEST_FILE`'s pool
        // exactly is the leaf plus `MIN_SCRATCH - 1` fillers. It used to be
        // `MIN_SCRATCH` of them, back when the leaf was a coordinate sitting
        // in an input register outside the pool; the collapse ABI passes no
        // vectors, so there is no such register any more.
        let pool = RegisterFile::MIN_SCRATCH as u32;
        let mut schedule = vec![leaf(0)];
        let fillers: Vec<u32> = (10..10 + pool - 1).collect();
        for &f in &fillers {
            schedule.push(def(f, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        }
        let new_val = 90;
        schedule.push(def(new_val, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        let def_index = schedule.len() - 1;
        // f1..f6 read soon (distance 1..6); f7 read at distance 10, the
        // farthest among the fillers, so `loser` picks it. `new_val` is read
        // at distance 2 — nearer than f7's 10 — so it should win the contest
        // and keep the register `f7` gave up.
        schedule.push(def(
            300,
            ScheduledOp::Unary(OpKind::Neg, ValueId(fillers[0])),
        ));
        schedule.push(def(301, ScheduledOp::Unary(OpKind::Neg, ValueId(new_val))));
        for &f in &fillers[1..fillers.len() - 1] {
            schedule.push(def(300 + f, ScheduledOp::Unary(OpKind::Neg, ValueId(f))));
        }
        schedule.push(def(
            999,
            ScheduledOp::Unary(OpKind::Neg, ValueId(*fillers.last().unwrap())),
        ));

        let a = alloc(schedule);
        assert!(
            matches!(
                a.body().where_at(ValueId(new_val), def_index + 1),
                Where::Reg(_)
            ),
            "new_val reads sooner than the occupant it evicted, so it should \
             still be resident one instruction later"
        );
    }

    /// A tie between the new definition and the occupant it evicted goes to
    /// the occupant: `keeps` is a strict `>`, not `>=`.
    ///
    /// As above, the check runs at a `Var` spacer right after `new_val`'s
    /// own definition, so an unrelated instruction's own destination contest
    /// (which would also be entitled to evict `new_val`, tie or no tie)
    /// cannot stand in for the answer this test is actually asking.
    #[test]
    fn a_tie_with_the_evicted_occupant_does_not_keep_the_new_definition() {
        // `leaf(0)` occupies a pool register of its own and stays live for
        // every filler's definition, so the set that fills `TEST_FILE`'s pool
        // exactly is the leaf plus `MIN_SCRATCH - 1` fillers. It used to be
        // `MIN_SCRATCH` of them, back when the leaf was a coordinate sitting
        // in an input register outside the pool; the collapse ABI passes no
        // vectors, so there is no such register any more.
        let pool = RegisterFile::MIN_SCRATCH as u32;
        let mut schedule = vec![leaf(0)];
        let fillers: Vec<u32> = (10..10 + pool - 1).collect();
        for &f in &fillers {
            schedule.push(def(f, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        }
        let new_val = 90;
        schedule.push(def(new_val, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        // As in the test above, a `Seq` is the spacer that takes no register.
        schedule.push(def(91, ScheduledOp::Seq(ValueId(0), ValueId(0))));
        let check_index = schedule.len() - 1;
        // f1..f6 read soon, so f7 (unread so far) is the farthest among the
        // fillers and is the one `loser` evicts.
        let (early, last) = fillers.split_at(fillers.len() - 1);
        for (i, &f) in early.iter().enumerate() {
            schedule.push(def(
                100 + i as u32,
                ScheduledOp::Unary(OpKind::Neg, ValueId(f)),
            ));
        }
        // One instruction reads both f7 and new_val, so both have the exact
        // same next-read distance from `new_val`'s own definition — a
        // genuine tie.
        schedule.push(def(
            999,
            ScheduledOp::Binary(OpKind::Add, ValueId(last[0]), ValueId(new_val)),
        ));

        let a = alloc(schedule);
        assert_eq!(
            a.body().where_at(ValueId(new_val), check_index),
            Where::Spilled,
            "tied against the occupant it evicted, new_val must not keep the \
             register — `keeps` requires strictly beating it"
        );
    }

    /// A demotion is never queued past the schedule's own end: the loser of
    /// the schedule's very last instruction has no `i + 1` to be reset at,
    /// and `demotions` is sized to `dag.len()`, so queuing one there would be
    /// an out-of-bounds write, not merely a wasted one.
    ///
    /// Eight unread values tied at "never read again": the eighth (also the
    /// schedule's last instruction) forces an eviction among the other
    /// seven, and every candidate — including the new definition itself —
    /// has the identical worst-case rank, so it loses its own contest
    /// exactly as any of the others would have.
    #[test]
    fn a_demoted_last_instruction_queues_no_demotion_past_the_schedule() {
        let mut schedule = vec![leaf(0)];
        for f in 10..17u32 {
            schedule.push(def(f, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        }
        schedule.push(def(20, ScheduledOp::Unary(OpKind::Neg, ValueId(0)))); // 8th value: the last instruction.
        let last = schedule.len() - 1;

        let a = alloc(schedule);
        assert!(
            matches!(a.body().where_at(ValueId(20), last), Where::Reg(_)),
            "nothing later reverses its own destination write; only a queued \
             demotion could, and there is nowhere to queue one to"
        );
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
        let mut schedule = vec![leaf(0)];
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
        let a = alloc(add_two_leaves());
        let b = alloc(add_two_leaves());
        for d in a.body().schedule() {
            assert_eq!(at(&a, d.value), at(&b, d.value));
        }
        assert_eq!(spill_count(&a), spill_count(&b));
    }

    /// A hoisted value is pinned to the slot its prologue parked it in,
    /// overriding whatever the allocator gave the placeholder def.
    #[test]
    fn a_placement_can_be_overridden() {
        let mut a = alloc(add_two_leaves());
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
        let a = alloc(vec![leaf(0), leaf(1), leaf(2), def(3, sel)]);
        assert_eq!(spill_count(&a), 0);
    }

    /// A destination never lands in a register one of its own operands is
    /// still living in — the invariant `resolve_operands` reads back off the
    /// allocation, and the reason SSE2 can write its two-operand form
    /// directly.
    ///
    /// `dst op= right` corrupts `right` when `dst == right` and `dst != left`.
    /// The destination *may* take an operand's register — it is a priced
    /// candidate, not an excluded one — but when it does, the eviction is
    /// recorded at this very index, so that operand is no longer *resident*
    /// here: it is reloaded from its slot, into `dst` if it is the operand
    /// the encoding consumes there and into a reserved register otherwise.
    /// What this asserts is the residency view, which is what the encoders
    /// see: at the instruction's own point, no operand still in a register is
    /// in `dst`'s. The backend needs no stashing temp to route around a case
    /// that cannot arise, which is what `emit_binary_safe` used to be and what
    /// held xmm10 out of every kernel's pool.
    #[test]
    fn a_destination_never_lands_on_a_resident_operand() {
        // Wide enough to evict: `width` values all live at once over a pool of
        // `MIN_SCRATCH`, then folded pairwise so every fold reads two of them.
        let width = u32::from(RegisterFile::MIN_SCRATCH) * 3;
        let mut schedule = vec![leaf(0)];
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
            // form would corrupt.
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

    /// The allocator reserved exactly the registers the emitter will ask for.
    ///
    /// `operand_sources` is one statement read twice — the allocator counts
    /// its `Reload`s to reserve, the emitter names the register each lands in
    /// — and the two agree only if residency is the same at both readings.
    /// The destination contest can change residency (it may evict an operand),
    /// so it runs *before* the counts; this is the check that it does. The
    /// same for a guard's mask: a branch emitted before the instruction needs
    /// `guard_mask` exactly when its mask is not in a register at that index.
    ///
    /// An earlier attempt at letting the destination take an operand's
    /// register left the counts where they were, and the emitter panicked with
    /// "operand k needs reload register n, which the allocator did not
    /// reserve" — the failure this test turns into a named assertion.
    fn assert_reservations_match_residency(a: &Allocation<'_>, file: &RegisterFile) {
        let schedule = a.schedule();
        let sites = guard_sites(a.select_guards(), schedule.len());
        for (i, d) in schedule.iter().enumerate() {
            if matches!(d.op, ScheduledOp::Reduce(..)) {
                continue; // Its own trip test reserves through the guard gate.
            }
            let in_register = |v: ValueId| matches!(a.where_at(v, i), Where::Reg(_));
            let mut resident = [true; 3];
            for (k, operand) in operands(&d.op).enumerate() {
                resident[k] = in_register(operand);
            }
            let want = reloads_wanted(operand_sources(&d.op, resident));
            let scratch = a.scratch(i);
            let have = (0..Scratch::MAX_RELOADS)
                .filter(|k| scratch.reload(*k).is_some())
                .count();
            assert_eq!(
                have, want,
                "{:?} at {i}: the allocator reserved {have} reload registers but \
                 resolve_operands will ask for {want}",
                d.value
            );
            let mask_needs_one = sites[i].iter().any(|m| !in_register(*m));
            assert!(
                !mask_needs_one || scratch.guard_mask.is_some(),
                "{:?} at {i}: a guard's mask is not in a register here and no \
                 guard_mask was reserved",
                d.value
            );
            // Nothing the instruction reads from a register is in `dst`'s —
            // the one alias every encoder assumes never happens.
            if let Where::Reg(dst) = a.where_at(d.value, i) {
                for operand in operands(&d.op) {
                    assert_ne!(
                        a.where_at(operand, i),
                        Where::Reg(dst),
                        "{:?} at {i}: destination {dst:?} holds resident operand {operand:?}",
                        d.value
                    );
                }
                for mask in &sites[i] {
                    if file.scratch.contains(dst) {
                        assert_ne!(
                            a.where_at(*mask, i),
                            Where::Reg(dst),
                            "{:?} at {i}: destination {dst:?} holds a guard's mask {mask:?}",
                            d.value
                        );
                    }
                }
            }
        }
    }

    /// The reservation contract holds under pressure, with and without temps,
    /// over a schedule wide enough to evict at every step.
    #[test]
    fn reservations_match_residency_under_pressure() {
        let width = u32::from(RegisterFile::MIN_SCRATCH) * 3;
        let mut schedule = vec![leaf(0), leaf(1)];
        for i in 2..=width {
            schedule.push(def(i, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        }
        // Folds that read three live values at once, so an instruction can
        // find every open register held by something it reads.
        let mut acc = ValueId(2);
        for i in 3..=width {
            let mask = ValueId(i);
            schedule.push(def(
                width + i,
                ScheduledOp::Ternary(OpKind::Select, mask, acc, ValueId(1)),
            ));
            acc = ValueId(width + i);
        }
        for file in [&TEST_FILE, &TEMP_FILE] {
            let a = LinearScan.allocate(schedule.clone(), file);
            assert!(spill_count(&a) > 0, "the schedule has to reach eviction");
            assert_reservations_match_residency(&a.body(), file);
        }
    }

    /// A four-trip fold, the metadata every fixture below hangs a scope off.
    fn fold_meta() -> pixelflow_ir::fold::RangeFold {
        use pixelflow_ir::fold::{Binder, Monoid, RangeFold};
        RangeFold::new(
            Monoid::SUM,
            Binder::from_slot(0).expect("slot 0 exists"),
            0..4,
        )
    }

    /// A nest of three folds: two siblings off the body, and one nested
    /// inside the first.
    ///
    /// The shape every test below needs, and the smallest one where a chain
    /// and a tree give different answers: the body runs all three, `Fold(0)`
    /// runs `Fold(2)`, and `Fold(1)` — a *later* scope than `Fold(0)` and an
    /// *earlier* one than `Fold(2)` — runs neither and is run by neither.
    fn nest_with_sibling_folds() -> NestAllocation {
        let park = ValueId(5);
        LinearScan.allocate_nest(
            ScopedSchedule {
                body: ScopeRegion {
                    roots: vec![park],
                    schedule: vec![
                        leaf(0),
                        def(5, ScheduledOp::Unary(OpKind::Neg, ValueId(0))),
                        // A fold's parent def is always a `Reduce` — that is
                        // where `allocate_nest` reads the binder and the
                        // fold's own metadata back out, so the fixture has to
                        // be one.
                        def(1, ScheduledOp::Reduce(fold_meta(), park)),
                        def(2, ScheduledOp::Reduce(fold_meta(), park)),
                        def(3, ScheduledOp::Binary(OpKind::Add, ValueId(1), ValueId(2))),
                    ],
                },
                folds: vec![
                    ScopeFold {
                        parent: Scope::Body,
                        at: 2,
                        roots: vec![ValueId(10)],
                        schedule: vec![
                            def(10, ScheduledOp::Unary(OpKind::Neg, park)),
                            def(11, ScheduledOp::Reduce(fold_meta(), ValueId(10))),
                        ],
                    },
                    ScopeFold {
                        parent: Scope::Body,
                        at: 3,
                        roots: vec![ValueId(20)],
                        schedule: vec![def(20, ScheduledOp::Unary(OpKind::Neg, park))],
                    },
                    ScopeFold {
                        parent: Scope::Fold(0),
                        at: 1,
                        roots: Vec::new(),
                        schedule: vec![def(50, ScheduledOp::Unary(OpKind::Neg, ValueId(10)))],
                    },
                ],
                guard_arms: Vec::new(),
            },
            &NEST_FILE,
        )
    }

    /// A fold's roots and the body's are one ranking, by what a carry saves
    /// per call, and nothing is reserved for either.
    ///
    /// One register at a time from the floor up, over a nest with two body
    /// roots the fold reads once each and a fold of four trips whose body
    /// reads its binder every trip. The constraint is what the binder's old
    /// reservation could not promise past a depth: no scope has more carried
    /// across it than the pool has above the floor, so the pool a scope
    /// allocates in never falls under it. The order is the ranking: the
    /// binder is read four times a call to a body root's once, so it goes
    /// first, then its accumulator, then the body's roots.
    #[test]
    fn a_folds_roots_and_the_bodys_are_one_ranking() {
        let floor = RegisterFile::MIN_SCRATCH;
        // (body roots carried, binder carried, accumulator carried), one
        // entry per register above the floor, from zero.
        let ladder = [
            (0usize, false, false),
            (0, true, false),
            (0, true, true),
            (1, true, true),
            (2, true, true),
        ];
        for (above, (roots_carried, binder_carried, acc_carried)) in ladder.into_iter().enumerate()
        {
            let file = RegisterFile {
                scratch: RegSet::range(4, floor + above as u8),
                ..TEMP_FILE
            }
            .checked();
            let (a, b) = (ValueId(2), ValueId(3));
            let alloc = LinearScan.allocate_nest(
                ScopedSchedule {
                    body: ScopeRegion {
                        roots: vec![a, b],
                        schedule: vec![
                            leaf(0),
                            def(2, ScheduledOp::Unary(OpKind::Neg, ValueId(0))),
                            def(3, ScheduledOp::Unary(OpKind::Neg, a)),
                            def(1, ScheduledOp::Reduce(fold_meta(), ValueId(0))),
                            def(100, ScheduledOp::Binary(OpKind::Add, ValueId(1), b)),
                        ],
                    },
                    folds: vec![ScopeFold {
                        parent: Scope::Body,
                        at: 3,
                        roots: Vec::new(),
                        schedule: vec![
                            def(50, ScheduledOp::Var(fold_meta().binder().var())),
                            def(51, ScheduledOp::Binary(OpKind::Add, ValueId(50), a)),
                            def(52, ScheduledOp::Binary(OpKind::Add, ValueId(51), b)),
                        ],
                    }],
                    guard_arms: Vec::new(),
                },
                &file,
            );
            let body = alloc.body();
            let carried = body
                .roots()
                .iter()
                .filter(|r| body.carried(**r).is_some())
                .count();
            assert_eq!(
                carried, roots_carried,
                "{above} above the floor: body roots carried"
            );
            let roots = alloc.fold_roots(0);
            let in_register = |at: Where| matches!(at, Where::Reg(_));
            assert_eq!(
                in_register(roots.binder),
                binder_carried,
                "{above} above the floor: binder at {:?}",
                roots.binder
            );
            assert_eq!(
                in_register(roots.accumulator),
                acc_carried,
                "{above} above the floor: accumulator at {:?}",
                roots.accumulator
            );
            // The fold's body allocates in the pool minus everything carried
            // into it — the body's carries and its own — and that never falls
            // under the floor.
            let carried_into = carried
                + [roots.binder, roots.accumulator]
                    .into_iter()
                    .filter(|at| in_register(*at))
                    .count();
            assert!(
                file.scratch.len() as usize - carried_into >= floor as usize,
                "{above} above the floor: the fold's body was handed {} registers",
                file.scratch.len() as usize - carried_into
            );
        }
    }

    /// A root nothing inside the fold reads is never a carry candidate, no
    /// matter how much budget is free: carrying it would spend a register for
    /// the whole loop to save reloads that do not exist.
    ///
    /// `read` is the control. Both roots are parked by the same scope, in one
    /// allocation, under a budget that demonstrably carries one of them — so
    /// the only thing that can be refusing the other is the zero-use filter.
    #[test]
    fn a_root_the_fold_never_reads_is_never_carried() {
        let read = ValueId(5);
        let unused = ValueId(6);
        let alloc = LinearScan.allocate_nest(
            ScopedSchedule {
                body: ScopeRegion {
                    roots: vec![read, unused],
                    schedule: vec![
                        leaf(0),
                        def(5, ScheduledOp::Unary(OpKind::Neg, ValueId(0))),
                        def(6, ScheduledOp::Unary(OpKind::Neg, ValueId(0))),
                        def(1, ScheduledOp::Reduce(fold_meta(), read)),
                    ],
                },
                folds: vec![ScopeFold {
                    parent: Scope::Body,
                    at: 3,
                    roots: Vec::new(),
                    // The fold names `read` every trip and `unused` never.
                    schedule: vec![def(50, ScheduledOp::Unary(OpKind::Neg, read))],
                }],
                guard_arms: Vec::new(),
            },
            &NEST_FILE,
        );
        assert!(
            alloc.body().carried(read).is_some(),
            "fixture assumes NEST_FILE's budget carries a root the fold does read"
        );
        assert_eq!(
            alloc.body().carried(unused),
            None,
            "budget is available (the control above proves it), so only the \
             zero-use filter can be refusing this"
        );
    }

    /// `within` is a subtree, not a suffix of a chain.
    ///
    /// A fold opens in the *middle* of its parent, so it is a sibling of the
    /// folds hanging off the same scope — the body runs both, and neither
    /// runs the other. Answering positionally (everything after me) would
    /// report `Fold(2)` as inside `Fold(1)`, and the register `Fold(2)`
    /// carries across its back edge would be handed out inside a loop that
    /// never runs it.
    #[test]
    fn a_fold_and_a_later_sibling_are_not_nested() {
        let alloc = nest_with_sibling_folds();
        let within = |s: Scope| {
            let mut v: Vec<Scope> = alloc.scope(s).within().map(|a| a.scope).collect();
            v.sort_unstable();
            v
        };

        let mut all_inside = vec![Scope::Fold(0), Scope::Fold(1), Scope::Fold(2)];
        all_inside.sort_unstable();
        assert_eq!(within(Scope::Body), all_inside, "the body runs everything");
        assert_eq!(
            within(Scope::Fold(0)),
            vec![Scope::Fold(2)],
            "the nested fold is inside the fold whose body holds its def"
        );
        assert_eq!(
            within(Scope::Fold(1)),
            Vec::new(),
            "a later scope is not a nested one"
        );
        assert_eq!(within(Scope::Fold(2)), Vec::new());
    }

    /// A fold opens at a def of its parent; the body does not open anywhere,
    /// because it wraps the whole of what is inside it.
    ///
    /// This is the query the emitter puts the back edge at, and the one
    /// question whose answer differs between a scope that surrounds its
    /// parent's code and one that interrupts it.
    #[test]
    fn only_a_fold_opens_partway_through_its_parent() {
        let alloc = nest_with_sibling_folds();
        assert_eq!(
            alloc.scope(Scope::Fold(0)).opens_at(),
            Some((Scope::Body, 2)),
            "the fold opens at the def it is the body of"
        );
        assert_eq!(
            alloc.scope(Scope::Fold(2)).opens_at(),
            Some((Scope::Fold(0), 1)),
            "and a nested fold opens inside its parent's schedule"
        );
        assert_eq!(alloc.body().opens_at(), None);
    }

    /// `fold_opening_at` asks [`Allocation::opens_at`]'s question from the
    /// other end, and both halves of its match must hold: the right parent
    /// at the wrong position finds nothing, just as the wrong parent would.
    #[test]
    fn fold_opening_at_matches_the_position_as_well_as_the_parent() {
        let alloc = nest_with_sibling_folds();
        let body = alloc.body();
        assert_eq!(
            body.fold_opening_at(2),
            Some(Scope::Fold(0)),
            "the fold does open here"
        );
        assert_eq!(
            body.fold_opening_at(0),
            None,
            "the body is the fold's parent, but the fold opens at 2, not 0"
        );
    }

    /// A scope encloses itself, so "is this in scope here" needs no special
    /// case for the asker — but `within` still excludes it, because the
    /// question there is what the code *inside* does.
    #[test]
    fn enclosing_is_reflexive_and_within_is_not() {
        let alloc = nest_with_sibling_folds();
        for scope in alloc.scopes() {
            assert!(alloc.encloses(scope, scope), "{scope:?} encloses itself");
            assert!(
                !alloc.scope(scope).within().any(|a| a.scope == scope),
                "{scope:?} is not within itself"
            );
        }
    }

    /// A fold reads its *ancestors'* parks, and a value parked by a scope
    /// beside it is not one of them.
    ///
    /// The prefix-of-the-chain form answered this by position, which for
    /// `Fold(2)` would have counted `Fold(1)`'s roots — values computed by a
    /// loop it never enters.
    #[test]
    fn a_fold_is_parked_by_its_ancestors_only() {
        let alloc = nest_with_sibling_folds();
        let nested = alloc.scope(Scope::Fold(2));

        assert!(
            nested.parked_by_an_enclosing_scope(ValueId(10)),
            "Fold(0) is the nested fold's parent, so its root is parked for it"
        );
        assert!(
            !nested.parked_by_an_enclosing_scope(ValueId(20)),
            "Fold(1) is beside the nested fold, so its root is not"
        );
        assert!(
            nested.parked_by_an_enclosing_scope(ValueId(5)),
            "the body encloses everything, so its root is parked for it too"
        );
        assert!(
            !alloc
                .scope(Scope::Fold(1))
                .parked_by_an_enclosing_scope(ValueId(10)),
            "and the sibling reads neither of the other branch's parks"
        );
    }

    /// A fold whose parent is a later fold is refused rather than allocated.
    ///
    /// Parents are allocated first, so a forward reference is both unanswerable
    /// and the only way to write a cycle. Refusing it here makes the cycle
    /// unrepresentable in an allocation rather than an infinite walk in
    /// `encloses`.
    #[test]
    #[should_panic(expected = "is not an earlier scope")]
    fn a_folds_parent_must_already_exist() {
        let _ = LinearScan.allocate_nest(
            ScopedSchedule {
                body: ScopeRegion {
                    roots: Vec::new(),
                    schedule: vec![
                        leaf(0),
                        def(1, ScheduledOp::Reduce(fold_meta(), ValueId(0))),
                    ],
                },
                folds: vec![ScopeFold {
                    parent: Scope::Fold(1),
                    at: 1,
                    roots: Vec::new(),
                    schedule: vec![leaf(50)],
                }],
                guard_arms: Vec::new(),
            },
            &NEST_FILE,
        );
    }

    /// The body computing six roots, and one fold that reads all of them.
    ///
    /// The shape both carry tests below need: more roots than the pool has
    /// above the floor, so some are carried and some are parked.
    fn nest_with_a_read_loop() -> (Vec<ValueId>, NestAllocation) {
        let width = 6u32;
        let mut outer = vec![leaf(0)];
        for i in 1..=width {
            outer.push(def(i, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        }
        let roots: Vec<ValueId> = (1..=width).map(ValueId).collect();

        let mut inner = vec![leaf(100)];
        let mut acc = ValueId(100);
        for (i, root) in roots.iter().enumerate() {
            inner.push(def(
                200 + i as u32,
                ScheduledOp::Binary(OpKind::Add, acc, *root),
            ));
            acc = ValueId(200 + i as u32);
        }

        let at = outer.len();
        outer.push(def(99, ScheduledOp::Reduce(fold_meta(), ValueId(0))));
        let alloc = LinearScan.allocate_nest(
            ScopedSchedule {
                body: ScopeRegion {
                    roots: roots.clone(),
                    schedule: outer,
                },
                folds: vec![ScopeFold {
                    parent: Scope::Body,
                    at,
                    roots: Vec::new(),
                    schedule: inner,
                }],
                guard_arms: Vec::new(),
            },
            &NEST_FILE,
        );
        (roots, alloc)
    }

    /// A carried register is untouched by every scope inside the loop.
    ///
    /// This is the whole safety property of showing the allocator the nest. A
    /// value the enclosing scope leaves in a register is read by the loop on
    /// every iteration, so anything the loop writes there is a miscompile that
    /// only shows up as wrong pixels. The loop's pool excludes carries by
    /// construction (`RegisterFile::inside`) — this is what says so out loud,
    /// and it checks the *temps* too, which are pool registers no `Placement`
    /// records.
    #[test]
    fn a_carried_register_is_untouched_by_everything_inside_the_loop() {
        let (roots, alloc) = nest_with_a_read_loop();

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
        // and the register it held in the scope that computed it belongs to
        // that scope — so the roots are excluded rather than counted.
        let inner = alloc.scope(Scope::Fold(0));
        let mut inside: Vec<Reg> = inner
            .schedule()
            .iter()
            .filter(|d| !roots.contains(&d.value))
            .flat_map(|d| inner.placement(d.value).registers())
            .collect();
        for i in 0..inner.schedule().len() {
            let s = inner.scratch(i);
            inside.extend((0..Scratch::MAX_TEMPS).filter_map(|k| s.temp(k)));
            inside.extend((0..Scratch::MAX_RELOADS).filter_map(|k| s.reload(k)));
            inside.extend(s.guard_mask);
            inside.extend(s.guard_temp);
            inside.extend(s.result);
        }

        for (vid, carry) in &carries {
            assert!(
                !inside.contains(carry),
                "{vid:?} is carried in {carry:?}, which the loop also writes"
            );
        }
    }

    /// The schedule's last instruction needs no separate result register when
    /// its own destination already gave it one: the three-way `&&` in the
    /// reservation's guard all have to hold, and here none of them do.
    #[test]
    fn the_last_instructions_own_destination_needs_no_extra_result_register() {
        let a = alloc(vec![
            leaf(0),
            leaf(1),
            def(2, ScheduledOp::Binary(OpKind::Add, ValueId(0), ValueId(1))),
        ]);
        let last = a.body().schedule().len() - 1;
        assert!(
            matches!(a.body().where_at(ValueId(2), last), Where::Reg(_)),
            "fixture assumes the root already has a register from its own destination"
        );
        assert_eq!(
            a.body().scratch(last).result,
            None,
            "a root that already computed into a register needs no separate result slot"
        );
    }

    /// A root already resident *because it was carried* also needs no extra
    /// result register, even though it is `live_in` — the other half of the
    /// same three-way `&&`: `live_in` alone is not enough, residency is what
    /// decides it.
    #[test]
    fn a_carried_roots_result_register_is_not_reserved_when_it_is_already_resident() {
        let root = ValueId(5);
        let alloc = LinearScan.allocate_nest(
            ScopedSchedule {
                body: ScopeRegion {
                    roots: vec![root],
                    schedule: vec![
                        leaf(0),
                        def(5, ScheduledOp::Unary(OpKind::Neg, ValueId(0))),
                        def(1, ScheduledOp::Reduce(fold_meta(), root)),
                    ],
                },
                folds: vec![ScopeFold {
                    parent: Scope::Body,
                    at: 2,
                    roots: Vec::new(),
                    schedule: vec![
                        // A real use, so `root` is ranked for carrying at all.
                        def(200, ScheduledOp::Unary(OpKind::Neg, root)),
                        // The enclosing park's own placeholder, last in the
                        // schedule — the live_in value this final check
                        // answers for.
                        def(5, ScheduledOp::Const(0.0)),
                    ],
                }],
                guard_arms: Vec::new(),
            },
            &NEST_FILE,
        );
        assert!(
            alloc.body().carried(root).is_some(),
            "fixture assumes NEST_FILE's budget carries the only root"
        );
        let inside = alloc.scope(Scope::Fold(0));
        let last = inside.schedule().len() - 1;
        assert!(
            matches!(inside.where_at(root, last), Where::Reg(_)),
            "a carried root is resident from a register, not a slot"
        );
        assert_eq!(
            inside.scratch(last).result,
            None,
            "already resident through the carry, so no result register is reserved for it"
        );
    }

    /// A root's placement is the whole of what used to need a `carries` map
    /// beside it: its register inside the scope that computes it, and then —
    /// from the first point of the loops within — either the carry or a slot.
    #[test]
    fn a_root_is_placed_twice_and_says_for_itself_whether_it_is_carried() {
        let (roots, alloc) = nest_with_a_read_loop();
        let (outer, inner) = (alloc.body(), alloc.scope(Scope::Fold(0)));
        let (mut carried, mut parked) = (0, 0);
        for root in &roots {
            let p = outer
                .placement_of(*root)
                .unwrap_or_else(|| panic!("{root:?} is computed by the body"));
            // Inside the computing scope: whatever the scan chose, and a pool
            // register there — never the carry, which is picked from what
            // that scope leaves free.
            let where_computed = p.at(Point::TAIL);
            let in_the_loop = inner.at_head(*root);
            assert_ne!(
                where_computed, in_the_loop,
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
    #[should_panic(expected = "vector_bytes")]
    fn a_non_power_of_two_vector_width_is_refused() {
        let _refused = RegisterFile {
            vector_bytes: 24,
            ..TEST_FILE
        }
        .checked();
    }
}
