//! What the emitter actually emitted, counted per scope.
//!
//! Three allocator policies have now been built and rejected on a static
//! quantity — memory operations, code bytes — that turned out not to predict
//! wall clock (`docs/plans/2026-09-01-register-allocation-escape-hatches.md`,
//! the 2026-09-04 blocks). Fitting a cost model to time needs the static
//! features and the time side by side, per kernel, per allocation, per tier,
//! and this is the static half: the counts the driver produces on its way to
//! machine code, attributed to the scope that executes them, so a trip count
//! can weight them afterwards.
//!
//! It is a *count*, never a decision. Nothing in the emitter or the allocator
//! reads a [`ScopeTraffic`]; it rides out on [`CompileResult`](super::CompileResult)
//! for a measurement harness to record. That is deliberate — an allocator that
//! optimized this number is exactly the thing the measurements above refused.
//!
//! The counting is done by [`Counting`], a decorator over the private
//! `IsaBackend` seam rather than a set of increments at the driver's emission
//! sites. A decorator cannot miss a site: every byte the driver emits goes
//! through one of these methods, so a new emission path is counted the day it
//! is written, and a trait method that disappears is a compile error rather
//! than a silently dropped term.

use super::regalloc::{Scope, ValueId};
use super::{Binding, InstructionPlan, IsaBackend, Loc, Reg, Reload, WritePlan};
use crate::error::CompileError;
use alloc::vec::Vec;

/// Emitted traffic within one scope of the nest.
///
/// A scope's counts are *static*: what the scope's code contains, not what a
/// call executes. Multiply by the scope's trip count for the dynamic figure —
/// which is the whole reason the split by scope exists, since an instruction
/// in the column fold runs `rows × batches` times and one in the body once.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScopeTraffic {
    /// Scheduled operations emitted (one per `InstructionPlan`).
    pub instructions: u32,
    /// Stack loads emitted as part of one instruction's operand resolution:
    /// the value is fetched into a register that instruction reserved, used,
    /// and forgotten.
    pub loads_transient: u32,
    /// Stack loads the driver emits *between* instructions — a range the
    /// allocator chose to bring back into a register the value then keeps, a
    /// scope head's reconciliation, a guard's mask, a fold's slot-held root.
    ///
    /// The split is by *which emission path*, not by which register the load
    /// targets. It used to be the latter, and that stopped being derivable
    /// when reload targets became per-instruction reservations drawn from the
    /// pool (#1158): every load now lands in a pool register, so the register
    /// number no longer says what the load bought. The call site does, and it
    /// always did — an `InstructionPlan`'s reloads serve one instruction by
    /// definition.
    pub loads_kept: u32,
    /// Constants re-emitted instead of loaded. Not a memory operation on x86,
    /// where the immediate is inline; on aarch64 it may reach the constant
    /// pool, which is why it is counted apart from both.
    pub remats: u32,
    /// Stack stores emitted: spills, parks, a fold's slot-held roots.
    pub stores: u32,
    /// The lattice's own stores — one per `Write`, whatever its width. Not a
    /// spill: the output plane is the kernel's result, not its scratch.
    pub writes: u32,
    /// Bytes of machine code the scope's own instructions occupy, excluding
    /// the scopes nested inside it, so every byte lands in exactly one scope.
    pub bytes: u32,
}

impl ScopeTraffic {
    /// Loads plus stores — the quantity #1150's table reports, and the one
    /// the 2026-09-04 measurements found does not predict AVX-512 time.
    #[must_use]
    pub const fn memory_ops(&self) -> u32 {
        self.loads_transient + self.loads_kept + self.stores
    }
}

/// The whole nest's traffic, plus the target facts a cost model needs to
/// price it (a 64-byte spill is not a 16-byte one).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EmitTraffic {
    /// One entry per scope, the body first and then the folds in nest order
    /// (`Scope::Fold(j)` at `j + 1`) — the lattice's row and column folds
    /// among them, since they are folds like any other.
    pub scopes: Vec<ScopeTraffic>,
    /// How many times one call runs each scope, indexed like `scopes`: the
    /// body once, a fold its trip count times its parent's.
    pub trips: Vec<u64>,
    /// The function around the nest: its frame, the body's place in it and
    /// the return. The same code under every allocation of a kernel on a
    /// given target, so it cannot explain a difference between two
    /// allocations — recorded separately rather than folded into a scope so
    /// that stays visible.
    pub scaffold: ScopeTraffic,
    /// Bytes after the return: aarch64's constant pool and the padding that
    /// aligns it, nothing on x86. The pool is the kernel's; the padding
    /// follows the code's length, so this is the one count that can differ
    /// between two allocations of a kernel with no instruction differing, by
    /// less than [`CONST_POOL_ALIGN`](super::aarch64::CONST_POOL_ALIGN).
    pub trailing: u32,
    /// Bytes one spilled register occupies: the backend's vector width.
    pub vector_bytes: u32,
    /// Registers the allocator had to hand out.
    pub pool: u8,
    /// Parked roots that hold a register across the scopes inside them
    /// rather than a slot.
    pub carried: u32,
}

impl EmitTraffic {
    /// Order what [`Counting`] recorded per scope, in whatever order the
    /// scopes finished, into `scopes`' index: the body first, then the folds.
    /// `count` is the number of scopes; one nothing was recorded for is empty.
    #[must_use]
    pub fn by_index(recorded: Vec<(Scope, ScopeTraffic)>, count: usize) -> Vec<ScopeTraffic> {
        let mut scopes = alloc::vec![ScopeTraffic::default(); count];
        for (scope, traffic) in recorded {
            scopes[scope_ix(scope)] = traffic;
        }
        scopes
    }

    /// The body's traffic: what runs once per call.
    #[must_use]
    pub fn body(&self) -> ScopeTraffic {
        self.scopes.first().copied().unwrap_or_default()
    }

    /// Every scope's bytes plus the function's own and what trails its return
    /// — the whole of what was emitted, by construction.
    #[must_use]
    pub fn bytes(&self) -> u32 {
        self.scopes.iter().map(|s| s.bytes).sum::<u32>() + self.scaffold.bytes + self.trailing
    }

    /// Memory operations one call executes: each scope's, weighted by how
    /// many times the call runs it.
    ///
    /// The scaffold is excluded: it is the same code under every allocation of
    /// a kernel, so including it only adds a constant to both sides of every
    /// comparison this number exists to make.
    #[must_use]
    pub fn dynamic_memory_ops(&self) -> u64 {
        self.scopes
            .iter()
            .zip(&self.trips)
            .map(|(s, trips)| u64::from(s.memory_ops()) * trips)
            .sum()
    }
}

/// The index a scope's count is kept under: the body first, then the folds.
fn scope_ix(scope: Scope) -> usize {
    match scope {
        Scope::Body => 0,
        Scope::Fold(j) => j + 1,
    }
}

/// One open scope's running count, and the bytes of the scopes that finished
/// inside it, so its own `bytes` can exclude them.
#[derive(Default)]
struct Open {
    traffic: ScopeTraffic,
    nested_bytes: u32,
}

/// An `IsaBackend` that counts what it forwards.
///
/// Scopes nest, so the counts do: `scope_begin` opens a fresh count that
/// everything emitted until the matching `scope_end` lands in, and closing it
/// records the count under the scope's name. What is emitted outside every
/// scope — the function's frame and trailer — accumulates at the base, read
/// off by `take`.
pub(super) struct Counting<'a, B: IsaBackend> {
    inner: &'a mut B,
    base: ScopeTraffic,
    open: Vec<Open>,
    closed: Vec<(Scope, ScopeTraffic)>,
}

impl<'a, B: IsaBackend> Counting<'a, B> {
    pub(super) fn new(inner: &'a mut B) -> Self {
        Self {
            inner,
            base: ScopeTraffic::default(),
            open: Vec::new(),
            closed: Vec::new(),
        }
    }

    /// The count everything emitted right now lands in.
    fn current(&mut self) -> &mut ScopeTraffic {
        match self.open.last_mut() {
            Some(open) => &mut open.traffic,
            None => &mut self.base,
        }
    }

    /// The traffic emitted outside every scope since the last `take`, with
    /// those counters reset; `bytes` is the caller's measure of it.
    pub(super) fn take(&mut self, bytes: u32) -> ScopeTraffic {
        debug_assert!(self.open.is_empty(), "take while a scope is open");
        let mut taken = core::mem::take(&mut self.base);
        taken.bytes = bytes;
        taken
    }

    /// Every scope closed so far, in the order they closed.
    pub(super) fn scopes(&mut self) -> Vec<(Scope, ScopeTraffic)> {
        core::mem::take(&mut self.closed)
    }
}

impl<B: IsaBackend> IsaBackend for Counting<'_, B> {
    fn jump(&mut self, asm: &mut super::Assembly, label: super::Label) {
        self.inner.jump(asm, label);
    }

    fn register_file(&self) -> super::regalloc::RegisterFile {
        self.inner.register_file()
    }

    fn begin(&mut self, schedule: &[super::regalloc::Def]) -> Result<(), CompileError> {
        self.inner.begin(schedule)
    }

    fn frame_ready(&mut self, frame_size: u32) {
        self.inner.frame_ready(frame_size);
    }

    fn emit_plan(
        &mut self,
        code: &mut Vec<u8>,
        plan: &InstructionPlan,
    ) -> Result<(), CompileError> {
        let current = self.current();
        current.instructions += 1;
        for reload in &plan.reloads {
            match reload {
                Reload::FromStack { .. } => current.loads_transient += 1,
                Reload::Const { .. } => current.remats += 1,
            }
        }
        // No store here: a plan's destination is always a register since
        // #1158, and the one place a value reaches its slot is the emit
        // loop's store-after-definition, which arrives through `emit_store`.
        self.inner.emit_plan(code, plan)
    }

    fn emit_mov(&mut self, code: &mut Vec<u8>, dst: Reg, src: Reg) {
        self.inner.emit_mov(code, dst, src);
    }

    fn emit_store(
        &mut self,
        code: &mut Vec<u8>,
        src: Reg,
        offset: u32,
    ) -> Result<(), CompileError> {
        self.current().stores += 1;
        self.inner.emit_store(code, src, offset)
    }

    fn emit_resolve(
        &mut self,
        code: &mut Vec<u8>,
        vid: ValueId,
        target: Reg,
        locs: &[Option<Binding>],
    ) -> Reg {
        match locs.get(vid.0 as usize).copied().flatten() {
            Some(Binding::Loc(Loc::Slot(_))) => self.current().loads_kept += 1,
            Some(Binding::Remat(_)) => self.current().remats += 1,
            // Already in a register, or not placed at all: nothing is emitted.
            Some(Binding::Loc(Loc::Reg(_))) | None => {}
        }
        self.inner.emit_resolve(code, vid, target, locs)
    }

    fn branch_if_arm_is_dead(
        &mut self,
        asm: &mut super::Assembly,
        test: super::MaskTest,
        label: super::Label,
    ) {
        self.inner.branch_if_arm_is_dead(asm, test, label);
    }

    fn frame_alloc(&mut self, code: &mut Vec<u8>, bytes: u32) {
        self.inner.frame_alloc(code, bytes);
    }

    fn frame_free(&mut self, code: &mut Vec<u8>, bytes: u32) {
        self.inner.frame_free(code, bytes);
    }

    fn anchor(&mut self, asm: &mut super::Assembly) {
        self.inner.anchor(asm);
    }

    fn finish(&mut self, asm: &mut super::Assembly) {
        self.inner.finish(asm);
    }

    fn slot_store(&mut self, code: &mut Vec<u8>, src: Reg, offset: u32) {
        self.current().stores += 1;
        self.inner.slot_store(code, src, offset);
    }

    fn slot_load(&mut self, code: &mut Vec<u8>, dst: Reg, offset: u32) {
        // A root a fold reloads from its slot is read for the whole iteration
        // that follows it, not for one instruction.
        self.current().loads_kept += 1;
        self.inner.slot_load(code, dst, offset);
    }

    fn scope_begin(&mut self) {
        self.open.push(Open::default());
        self.inner.scope_begin();
    }

    fn scope_end(&mut self, scope: Scope, bytes: u32) {
        let Open {
            mut traffic,
            nested_bytes,
        } = self.open.pop().expect("scope_end without a scope_begin");
        traffic.bytes = bytes - nested_bytes;
        if let Some(parent) = self.open.last_mut() {
            parent.nested_bytes += bytes;
        }
        self.closed.push((scope, traffic));
        self.inner.scope_end(scope, bytes);
    }

    fn add_scalar(&mut self, code: &mut Vec<u8>, dst: Reg, scratch: Reg, scalar: f32) {
        self.inner.add_scalar(code, dst, scratch, scalar);
    }

    fn load_const(&mut self, code: &mut Vec<u8>, dst: Reg, val: f32) {
        self.inner.load_const(code, dst, val);
    }

    fn alu(
        &mut self,
        code: &mut Vec<u8>,
        op: pixelflow_ir::kind::OpKind,
        dst: Reg,
        srcs: [Reg; 2],
    ) {
        self.inner.alu(code, op, dst, srcs);
    }

    // Explicit rather than inherited: the trait's default for `test_ge`
    // calls `self.alu`, which through this wrapper would call `Counting`'s
    // own `alu` — never reaching a backend's own `test_ge` override (only
    // AVX-512 has one). Forwarding the call itself, not its default body, is
    // what keeps that override reachable through the decorator.
    fn test_ge(
        &mut self,
        code: &mut Vec<u8>,
        dst: Reg,
        srcs: [Reg; 2],
        mask_scratch: Option<super::KReg>,
    ) {
        self.inner.test_ge(code, dst, srcs, mask_scratch);
    }

    fn emit_write(&mut self, code: &mut Vec<u8>, write: &WritePlan) {
        self.current().writes += 1;
        self.inner.emit_write(code, write);
    }

    fn emit_ret(&mut self, code: &mut Vec<u8>) {
        self.inner.emit_ret(code);
    }
}

#[cfg(test)]
mod tests {
    use crate::emit::EmitCtx;
    use pixelflow_ir::OpKind;
    use pixelflow_ir::arena::{ExprArena, ExprId};

    use super::super::regalloc::{self, Scope};
    use super::super::storage::Slot;
    use super::super::{
        Assembly, Binding, InstructionPlan, IsaBackend, Label, Loc, MaskTest, Reg, Reload,
        ResolvedOp, WritePlan,
    };
    use super::{Counting, EmitTraffic, ScopeTraffic};
    use crate::error::CompileError;
    use pixelflow_ir::LatticeShape;

    /// An [`IsaBackend`] that does nothing but hand back what a test told it
    /// to, so [`Counting`]'s own counting and forwarding can be pinned
    /// without a real encoder or a compiled kernel.
    struct RecordingBackend {
        begin_result: Result<(), CompileError>,
        anchor_calls: u32,
        finish_calls: u32,
    }

    impl RecordingBackend {
        fn new() -> Self {
            Self {
                begin_result: Ok(()),
                anchor_calls: 0,
                finish_calls: 0,
            }
        }
    }

    impl IsaBackend for RecordingBackend {
        fn jump(&mut self, _asm: &mut Assembly, _label: Label) {}

        fn register_file(&self) -> regalloc::RegisterFile {
            unimplemented!("not exercised by the traffic-counting tests")
        }

        fn begin(&mut self, _schedule: &[regalloc::Def]) -> Result<(), CompileError> {
            self.begin_result
        }

        fn emit_plan(
            &mut self,
            _code: &mut Vec<u8>,
            _plan: &InstructionPlan,
        ) -> Result<(), CompileError> {
            Ok(())
        }

        fn emit_mov(&mut self, _code: &mut Vec<u8>, _dst: Reg, _src: Reg) {}

        fn emit_store(
            &mut self,
            _code: &mut Vec<u8>,
            _src: Reg,
            _offset: u32,
        ) -> Result<(), CompileError> {
            Ok(())
        }

        fn emit_resolve(
            &mut self,
            _code: &mut Vec<u8>,
            _vid: regalloc::ValueId,
            target: Reg,
            _locs: &[Option<Binding>],
        ) -> Reg {
            target
        }

        fn branch_if_arm_is_dead(&mut self, _asm: &mut Assembly, _test: MaskTest, _label: Label) {}

        fn frame_alloc(&mut self, _code: &mut Vec<u8>, _bytes: u32) {}

        fn frame_free(&mut self, _code: &mut Vec<u8>, _bytes: u32) {}

        fn anchor(&mut self, _asm: &mut Assembly) {
            self.anchor_calls += 1;
        }

        fn finish(&mut self, _asm: &mut Assembly) {
            self.finish_calls += 1;
        }

        fn slot_store(&mut self, _code: &mut Vec<u8>, _src: Reg, _offset: u32) {}

        fn slot_load(&mut self, _code: &mut Vec<u8>, _dst: Reg, _offset: u32) {}

        fn add_scalar(&mut self, _code: &mut Vec<u8>, _dst: Reg, _scratch: Reg, _scalar: f32) {}

        fn load_const(&mut self, _code: &mut Vec<u8>, _dst: Reg, _val: f32) {}

        fn alu(&mut self, _code: &mut Vec<u8>, _op: OpKind, _dst: Reg, _srcs: [Reg; 2]) {}

        fn emit_write(&mut self, _code: &mut Vec<u8>, _write: &WritePlan) {}

        fn emit_ret(&mut self, _code: &mut Vec<u8>) {}
    }

    /// `memory_ops` is read by a measurement harness, not by anything this
    /// crate itself branches on, so a wrong formula would ship silently
    /// unless a test pins the exact arithmetic against values that cannot
    /// agree by coincidence.
    #[test]
    fn memory_ops_sums_transient_loads_kept_loads_and_stores() {
        let traffic = ScopeTraffic {
            loads_transient: 3,
            loads_kept: 5,
            stores: 7,
            ..ScopeTraffic::default()
        };
        assert_eq!(traffic.memory_ops(), 15);
    }

    /// Same reasoning as `memory_ops` above, one level up: each scope is
    /// weighted by its own trip count, so the test's inputs are chosen so
    /// every wrong weighting or wrong operator lands on a different total.
    #[test]
    fn dynamic_memory_ops_weights_each_scope_by_its_trip_count() {
        let body = ScopeTraffic {
            loads_transient: 1,
            ..ScopeTraffic::default()
        };
        let rows = ScopeTraffic {
            loads_transient: 2,
            stores: 1,
            ..ScopeTraffic::default()
        };
        let cols = ScopeTraffic {
            loads_transient: 4,
            stores: 1,
            ..ScopeTraffic::default()
        };
        let recorded = alloc::vec![
            (Scope::Fold(1), cols),
            (Scope::Body, body),
            (Scope::Fold(0), rows)
        ];
        let traffic = EmitTraffic {
            scopes: EmitTraffic::by_index(recorded, 3),
            trips: alloc::vec![1, 6, 42],
            ..EmitTraffic::default()
        };

        assert_eq!(traffic.scopes, alloc::vec![body, rows, cols]);
        assert_eq!(traffic.dynamic_memory_ops(), 1 + 3 * 6 + 5 * 42);
    }

    /// `begin` is the one place a backend can refuse to compile at all
    /// (aarch64's constant pool overflowing its 12-bit `LDR` offset); the
    /// decorator must hand that error back rather than swallowing it.
    #[test]
    fn begin_propagates_the_inner_backends_error_instead_of_swallowing_it() {
        let mut backend = RecordingBackend::new();
        backend.begin_result = Err(CompileError::BudgetExceeded("stub"));
        let mut counting = Counting::new(&mut backend);

        assert_eq!(
            counting.begin(&[]),
            Err(CompileError::BudgetExceeded("stub"))
        );
    }

    /// A schedule can carry both kinds of reload in one instruction; each
    /// must land in its own counter rather than either being folded into, or
    /// masking, the other.
    #[test]
    fn emit_plan_counts_a_stack_reload_and_a_const_reload_separately() {
        let mut backend = RecordingBackend::new();
        let mut counting = Counting::new(&mut backend);
        let mut code = Vec::new();
        let plan = InstructionPlan {
            reloads: vec![
                Reload::FromStack {
                    target: Reg(0),
                    slot: Slot::new(0, 16),
                },
                Reload::Const {
                    target: Reg(1),
                    val_bits: 0x3f80_0000,
                },
            ],
            op: ResolvedOp::Nop,
            setup_mov: None,
            scratch: regalloc::Scratch::for_test(None, [None, None]),
        };

        counting.emit_plan(&mut code, &plan).expect("emit_plan");
        let traffic = counting.take(0);

        assert_eq!(traffic.instructions, 1);
        assert_eq!(
            traffic.loads_transient, 1,
            "a stack reload was not counted: {traffic:?}"
        );
        assert_eq!(
            traffic.remats, 1,
            "a constant reload was not counted: {traffic:?}"
        );
    }

    /// A value resolved from its spilled slot is a kept load, not a remat —
    /// the two counters back different rows of the cost model and must not
    /// bleed into each other.
    #[test]
    fn emit_resolve_counts_a_spilled_value_as_a_kept_load() {
        let mut backend = RecordingBackend::new();
        let mut counting = Counting::new(&mut backend);
        let mut code = Vec::new();
        let locs = [Some(Binding::Loc(Loc::Slot(Slot::new(0, 16))))];

        counting.emit_resolve(&mut code, regalloc::ValueId(0), Reg(0), &locs);
        let traffic = counting.take(0);

        assert_eq!(
            traffic.loads_kept, 1,
            "a value resolved from a stack slot was not counted as a kept load: {traffic:?}"
        );
        assert_eq!(traffic.remats, 0);
    }

    /// The `Remat` mirror of the case above: re-emitting a constant is not a
    /// memory operation and must not be counted as one.
    #[test]
    fn emit_resolve_counts_a_rematerialized_constant_as_a_remat_not_a_load() {
        let mut backend = RecordingBackend::new();
        let mut counting = Counting::new(&mut backend);
        let mut code = Vec::new();
        let locs = [Some(Binding::Remat(0x3f80_0000))];

        counting.emit_resolve(&mut code, regalloc::ValueId(0), Reg(0), &locs);
        let traffic = counting.take(0);

        assert_eq!(
            traffic.remats, 1,
            "a rematerialized constant was not counted: {traffic:?}"
        );
        assert_eq!(traffic.loads_kept, 0);
    }

    /// A value already resident in a register costs nothing to resolve, so
    /// resolving it must not move any counter.
    #[test]
    fn emit_resolve_counts_nothing_for_a_value_already_in_a_register() {
        let mut backend = RecordingBackend::new();
        let mut counting = Counting::new(&mut backend);
        let mut code = Vec::new();
        let locs = [Some(Binding::Loc(Loc::Reg(Reg(3))))];

        counting.emit_resolve(&mut code, regalloc::ValueId(0), Reg(0), &locs);

        assert_eq!(counting.take(0), ScopeTraffic::default());
    }

    /// The function's hooks carry no counter of their own, but aarch64's
    /// backend overrides both to seed and flush its literal pool — if the
    /// decorator ever stopped forwarding them, that pool would silently go
    /// missing on that target.
    #[test]
    fn anchor_and_finish_forward_to_the_inner_backend() {
        let mut backend = RecordingBackend::new();
        let mut asm = Assembly::default();
        {
            let mut counting = Counting::new(&mut backend);
            counting.anchor(&mut asm);
            counting.finish(&mut asm);
        }

        assert_eq!(backend.anchor_calls, 1);
        assert_eq!(backend.finish_calls, 1);
    }

    /// A fold's own store/reload path (`slot_store`/`slot_load`) is separate
    /// from an instruction's operand resolution and must count independently
    /// of it.
    #[test]
    fn slot_store_and_slot_load_each_count_once_per_call() {
        let mut backend = RecordingBackend::new();
        let mut counting = Counting::new(&mut backend);
        let mut code = Vec::new();

        counting.slot_store(&mut code, Reg(0), 0);
        counting.slot_load(&mut code, Reg(0), 0);
        let traffic = counting.take(0);

        assert_eq!(
            traffic.stores, 1,
            "slot_store did not count as a store: {traffic:?}"
        );
        assert_eq!(
            traffic.loads_kept, 1,
            "slot_load did not count as a kept load: {traffic:?}"
        );
    }

    /// Scopes nest, and a nested scope's bytes are its own: a parent's count
    /// is what the parent emitted, not what it contains.
    #[test]
    fn a_nested_scopes_bytes_are_excluded_from_its_parents() {
        let mut backend = RecordingBackend::new();
        let mut counting = Counting::new(&mut backend);
        let mut code = Vec::new();

        counting.scope_begin();
        counting.slot_store(&mut code, Reg(0), 0);
        counting.scope_begin();
        counting.slot_load(&mut code, Reg(0), 0);
        counting.scope_end(Scope::Fold(0), 4);
        counting.scope_end(Scope::Body, 10);

        let scopes = counting.scopes();
        assert_eq!(scopes.len(), 2);
        assert_eq!(
            (scopes[0].0, scopes[0].1.bytes, scopes[0].1.loads_kept),
            (Scope::Fold(0), 4, 1)
        );
        assert_eq!(
            (scopes[1].0, scopes[1].1.bytes, scopes[1].1.stores),
            (Scope::Body, 6, 1)
        );
        assert_eq!(counting.take(0), ScopeTraffic::default());
    }

    /// Registers to allocate in the pressure test: small enough that a
    /// deliberately wide expression cannot fit, on every tier.
    const TIGHT_POOL: u8 = 7;

    /// A wide sum whose terms are all pushed before any is consumed, so more
    /// values are live at once than `TIGHT_POOL` can hold.
    fn wide_live_range_kernel(terms: usize) -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let live: Vec<ExprId> = (0..terms)
            .map(|i| {
                let c = a.push_const(0.25 + i as f32 * 0.125);
                let scaled = a.push_binary(OpKind::Mul, x, c);
                a.push_binary(OpKind::Add, scaled, y)
            })
            .collect();
        let root = live
            .iter()
            .skip(1)
            .fold(live[0], |acc, &t| a.push_binary(OpKind::Add, acc, t));
        (a, root)
    }

    const SHAPE: LatticeShape = LatticeShape::new([16, 2]);

    /// The completeness property the decorator exists to have: every byte the
    /// driver emitted landed in exactly one scope's count.
    ///
    /// This is what a set of increments at the driver's emission sites cannot
    /// promise — one forgotten site there is a silently missing term, and here
    /// it is a failing assertion.
    #[test]
    fn every_emitted_byte_is_attributed_to_exactly_one_scope() {
        for terms in [2usize, 8, 24] {
            let (arena, root) = wide_live_range_kernel(terms);
            let result = EmitCtx::with_max_regs(TIGHT_POOL)
                .compile(&arena, root, SHAPE)
                .expect("compile");
            let t = &result.traffic;
            assert_eq!(
                t.bytes() as usize,
                result.code.len(),
                "{terms} terms: {} bytes attributed, {} emitted",
                t.bytes(),
                result.code.len()
            );
        }
    }

    /// Spilling under a tight pool must show up as traffic. If a future
    /// emission path routes a store or a reload around the decorator, this is
    /// the test that notices — the store-after-definition path in particular,
    /// which moved out of `InstructionPlan` and into `emit_store` when class C
    /// closed.
    #[test]
    fn a_kernel_that_must_spill_reports_stores_and_loads() {
        let (arena, root) = wide_live_range_kernel(24);
        let result = EmitCtx::with_max_regs(TIGHT_POOL)
            .compile(&arena, root, SHAPE)
            .expect("compile");
        assert!(
            result.spill_count > 0,
            "24 values live against a {TIGHT_POOL}-register pool did not spill; \
             the scenario has stopped testing its subject"
        );
        let t = &result.traffic;
        let stores: u32 = t.scopes.iter().map(|s| s.stores).sum();
        let loads: u32 = t
            .scopes
            .iter()
            .map(|s| s.loads_transient + s.loads_kept)
            .sum();
        let instructions: u32 = t.scopes.iter().map(|s| s.instructions).sum();
        assert!(
            stores > 0,
            "values reached a frame slot with no store counted: {t:?}"
        );
        assert!(loads > 0, "values were spilled and never reloaded: {t:?}");
        assert!(
            instructions > 0,
            "a kernel with a body emitted no scheduled operation: {t:?}"
        );
    }

    /// The lattice's stores are counted apart from spills: a kernel that
    /// spills nothing still writes every batch of every row.
    #[test]
    fn every_batch_of_every_row_is_one_write() {
        let (arena, root) = wide_live_range_kernel(2);
        let result = crate::emit::compile(&arena, root, SHAPE).expect("compile");
        let t = &result.traffic;
        let dynamic_writes: u64 = t
            .scopes
            .iter()
            .zip(&t.trips)
            .map(|(s, trips)| u64::from(s.writes) * trips)
            .sum();
        let lanes = u64::from(crate::JIT_VECTOR_BYTES as u32 / 4);
        let [width, rows] = SHAPE.extent().map(u64::from);
        assert_eq!(dynamic_writes, rows * width.div_ceil(lanes));
    }

    /// The scaffold is the same code under every allocation of a kernel, so it
    /// is counted apart from the scopes rather than folded into one — a
    /// difference between two allocations must not be able to hide there.
    ///
    /// Every backend, from this host, since each emits its own frame. What
    /// trails the return is counted apart again: aarch64's constant pool is
    /// the kernel's, but the padding that aligns it follows the code's
    /// length, so that count may move with the budget by less than one
    /// alignment — and it is the only count that may.
    #[test]
    fn the_scaffolds_traffic_does_not_move_with_the_pool() {
        use crate::emit::tests::schedule_for;
        use crate::emit::{
            BYTES_PER_LANE, IsaBackend, aarch64, avx2, avx512, compile_via_backend, x86_64,
        };

        fn traffic<B: IsaBackend>(mut backend: B, arena: &ExprArena, root: ExprId) -> EmitTraffic {
            let lanes = backend.register_file().vector_bytes / BYTES_PER_LANE;
            let schedule = schedule_for(arena, root, SHAPE, lanes);
            compile_via_backend(schedule, &mut backend)
                .expect("compile")
                .traffic
        }
        let (arena, root) = wide_live_range_kernel(24);
        let tight = || EmitCtx::with_max_regs(TIGHT_POOL);
        let loose = EmitCtx::default;
        let budgets = [
            (
                "NEON",
                traffic(aarch64::driver::Aarch64Backend::new(tight()), &arena, root),
                traffic(aarch64::driver::Aarch64Backend::new(loose()), &arena, root),
            ),
            (
                "SSE2",
                traffic(x86_64::driver::X86Backend::new(tight()), &arena, root),
                traffic(x86_64::driver::X86Backend::new(loose()), &arena, root),
            ),
            (
                "AVX2",
                traffic(avx2::driver::Avx2Backend::new(tight()), &arena, root),
                traffic(avx2::driver::Avx2Backend::new(loose()), &arena, root),
            ),
            (
                "AVX-512",
                traffic(avx512::driver::Avx512Backend::new(tight()), &arena, root),
                traffic(avx512::driver::Avx512Backend::new(loose()), &arena, root),
            ),
        ];
        for (name, t, l) in budgets {
            assert_eq!(
                t.scaffold, l.scaffold,
                "{name}: the scaffold changed with the register budget"
            );
            assert!(
                t.trailing.abs_diff(l.trailing) < aarch64::CONST_POOL_ALIGN as u32,
                "{name}: what trails the return changed with the register budget by more \
                 than the pool's alignment: {} vs {} bytes",
                t.trailing,
                l.trailing
            );
        }
    }
}
