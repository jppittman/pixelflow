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

use super::regalloc::ValueId;
use super::{Binding, InstructionPlan, IsaBackend, KReg, Loc, Reg, Reload};
use crate::error::CompileError;

/// Emitted traffic within one scope of the collapse nest.
///
/// A scope's counts are *static*: what the scope's code contains, not what a
/// call executes. Multiply by the scope's trip count for the dynamic figure —
/// which is the whole reason the split by scope exists, since a body
/// instruction runs `rows × groups` times and a prologue instruction once.
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
    /// scope head's reconciliation, a guard's mask, the scaffold's coordinate
    /// reloads.
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
    /// Stack stores emitted, including the scaffold's coordinate saves.
    pub stores: u32,
    /// Bytes of machine code the scope occupies.
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EmitTraffic {
    /// The once-per-call prologue (X- and Y-invariant values).
    pub frame: ScopeTraffic,
    /// The once-per-row prologue (X-invariant values).
    pub row: ScopeTraffic,
    /// The innermost batch body.
    pub body: ScopeTraffic,
    /// The collapse scaffold itself: coordinate slot traffic, induction
    /// stepping, the output store. Constant for a given target, so it cannot
    /// explain a difference between two allocations of one kernel — recorded
    /// separately rather than folded into a scope so that stays visible.
    pub scaffold: ScopeTraffic,
    /// Bytes one spilled register occupies: the backend's vector width.
    pub vector_bytes: u32,
    /// Registers the allocator had to hand out.
    pub pool: u8,
    /// Hoist roots that hold a register across the loops inside them rather
    /// than a slot.
    pub carried: u32,
}

impl EmitTraffic {
    /// Memory operations one call executes, given each scope's trip count.
    ///
    /// The scaffold is excluded: it is the same code under every allocation of
    /// a kernel, so including it only adds a constant to both sides of every
    /// comparison this number exists to make.
    #[must_use]
    pub const fn dynamic_memory_ops(&self, rows: u64, groups: u64) -> u64 {
        self.frame.memory_ops() as u64
            + self.row.memory_ops() as u64 * rows
            + self.body.memory_ops() as u64 * rows * groups
    }
}

/// An `IsaBackend` that counts what it forwards.
///
/// `take` reads and clears, so the driver brackets each scope's emission with
/// one call and the counts partition by construction.
pub(super) struct Counting<'a, B: IsaBackend> {
    inner: &'a mut B,
    traffic: ScopeTraffic,
}

impl<'a, B: IsaBackend> Counting<'a, B> {
    pub(super) fn new(inner: &'a mut B) -> Self {
        Self {
            inner,
            traffic: ScopeTraffic::default(),
        }
    }

    /// The traffic since the last `take`, with the counters reset.
    pub(super) fn take(&mut self, bytes: u32) -> ScopeTraffic {
        let mut taken = core::mem::take(&mut self.traffic);
        taken.bytes = bytes;
        taken
    }
}

impl<B: IsaBackend> IsaBackend for Counting<'_, B> {
    type Branch = B::Branch;

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
        self.traffic.instructions += 1;
        for reload in &plan.reloads {
            match reload {
                Reload::FromStack { .. } => self.traffic.loads_transient += 1,
                Reload::Const { .. } => self.traffic.remats += 1,
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
        self.traffic.stores += 1;
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
            Some(Binding::Loc(Loc::Slot(_))) => self.traffic.loads_kept += 1,
            Some(Binding::Remat(_)) => self.traffic.remats += 1,
            // Already in a register, or not placed at all: nothing is emitted.
            Some(Binding::Loc(Loc::Reg(_))) | None => {}
        }
        self.inner.emit_resolve(code, vid, target, locs)
    }

    fn emit_skip_if_all_false(
        &mut self,
        code: &mut Vec<u8>,
        mask_reg: Reg,
        scratch: Option<Reg>,
        mask_scratch: Option<KReg>,
    ) -> Self::Branch {
        self.inner
            .emit_skip_if_all_false(code, mask_reg, scratch, mask_scratch)
    }

    fn emit_skip_if_all_true(
        &mut self,
        code: &mut Vec<u8>,
        mask_reg: Reg,
        scratch: Option<Reg>,
        mask_scratch: Option<KReg>,
    ) -> Self::Branch {
        self.inner
            .emit_skip_if_all_true(code, mask_reg, scratch, mask_scratch)
    }

    fn emit_jump(&mut self, code: &mut Vec<u8>) -> Self::Branch {
        self.inner.emit_jump(code)
    }

    fn patch_branch(&mut self, code: &mut Vec<u8>, branch: Self::Branch, target: usize) {
        self.inner.patch_branch(code, branch, target);
    }

    fn body_frame_bytes(&self, frame_size: u32) -> u32 {
        self.inner.body_frame_bytes(frame_size)
    }

    fn frame_alloc(&mut self, code: &mut Vec<u8>, bytes: u32) {
        self.inner.frame_alloc(code, bytes);
    }

    fn frame_free(&mut self, code: &mut Vec<u8>, bytes: u32) {
        self.inner.frame_free(code, bytes);
    }

    fn scaffold_anchor(&mut self, code: &mut Vec<u8>) {
        self.inner.scaffold_anchor(code);
    }

    fn scaffold_finish(&mut self, code: &mut Vec<u8>) {
        self.inner.scaffold_finish(code);
    }

    fn slot_store(&mut self, code: &mut Vec<u8>, src: Reg, offset: u32) {
        self.traffic.stores += 1;
        self.inner.slot_store(code, src, offset);
    }

    fn slot_load(&mut self, code: &mut Vec<u8>, dst: Reg, offset: u32) {
        // A coordinate the scaffold reloads is read for the whole iteration
        // that follows it, not for one instruction.
        self.traffic.loads_kept += 1;
        self.inner.slot_load(code, dst, offset);
    }

    fn latch_bounds(&mut self, code: &mut Vec<u8>) {
        self.inner.latch_bounds(code);
    }

    fn counter_clear(&mut self, code: &mut Vec<u8>, counter: super::Counter) {
        self.inner.counter_clear(code, counter);
    }

    fn counter_step(&mut self, code: &mut Vec<u8>, counter: super::Counter) {
        self.inner.counter_step(code, counter);
    }

    fn branch_if_counter_done(
        &mut self,
        code: &mut Vec<u8>,
        counter: super::Counter,
    ) -> Self::Branch {
        self.inner.branch_if_counter_done(code, counter)
    }

    fn store_result(&mut self, code: &mut Vec<u8>, src: Reg) {
        self.inner.store_result(code, src);
    }

    fn advance_out(&mut self, code: &mut Vec<u8>, step: super::OutStep) {
        self.inner.advance_out(code, step);
    }

    fn add_scalar(&mut self, code: &mut Vec<u8>, dst: Reg, scratch: Reg, scalar: f32) {
        self.inner.add_scalar(code, dst, scratch, scalar);
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

    use super::super::regalloc;
    use super::super::storage::Slot;
    use super::super::{
        Binding, Counter, InstructionPlan, IsaBackend, KReg, Loc, OutStep, Reg, Reload, ResolvedOp,
    };
    use super::{Counting, EmitTraffic, ScopeTraffic};
    use crate::error::CompileError;

    /// An [`IsaBackend`] that does nothing but hand back what a test told it
    /// to, so [`Counting`]'s own counting and forwarding can be pinned
    /// without a real encoder or a compiled kernel.
    struct RecordingBackend {
        begin_result: Result<(), CompileError>,
        scaffold_anchor_calls: u32,
        scaffold_finish_calls: u32,
    }

    impl RecordingBackend {
        fn new() -> Self {
            Self {
                begin_result: Ok(()),
                scaffold_anchor_calls: 0,
                scaffold_finish_calls: 0,
            }
        }
    }

    impl IsaBackend for RecordingBackend {
        type Branch = ();

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

        fn emit_skip_if_all_false(
            &mut self,
            _code: &mut Vec<u8>,
            _mask_reg: Reg,
            _scratch: Option<Reg>,
            _mask_scratch: Option<KReg>,
        ) -> Self::Branch {
        }

        fn emit_skip_if_all_true(
            &mut self,
            _code: &mut Vec<u8>,
            _mask_reg: Reg,
            _scratch: Option<Reg>,
            _mask_scratch: Option<KReg>,
        ) -> Self::Branch {
        }

        fn emit_jump(&mut self, _code: &mut Vec<u8>) -> Self::Branch {}

        fn patch_branch(&mut self, _code: &mut Vec<u8>, _branch: Self::Branch, _target: usize) {}

        fn frame_alloc(&mut self, _code: &mut Vec<u8>, _bytes: u32) {}

        fn frame_free(&mut self, _code: &mut Vec<u8>, _bytes: u32) {}

        fn scaffold_anchor(&mut self, _code: &mut Vec<u8>) {
            self.scaffold_anchor_calls += 1;
        }

        fn scaffold_finish(&mut self, _code: &mut Vec<u8>) {
            self.scaffold_finish_calls += 1;
        }

        fn slot_store(&mut self, _code: &mut Vec<u8>, _src: Reg, _offset: u32) {}

        fn slot_load(&mut self, _code: &mut Vec<u8>, _dst: Reg, _offset: u32) {}

        fn counter_clear(&mut self, _code: &mut Vec<u8>, _counter: Counter) {}

        fn counter_step(&mut self, _code: &mut Vec<u8>, _counter: Counter) {}

        fn branch_if_counter_done(
            &mut self,
            _code: &mut Vec<u8>,
            _counter: Counter,
        ) -> Self::Branch {
        }

        fn store_result(&mut self, _code: &mut Vec<u8>, _src: Reg) {}

        fn advance_out(&mut self, _code: &mut Vec<u8>, _step: OutStep) {}

        fn add_scalar(&mut self, _code: &mut Vec<u8>, _dst: Reg, _scratch: Reg, _scalar: f32) {}

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

    /// Same reasoning as `memory_ops` above, one level up: the three scopes'
    /// weights (1, `rows`, `rows * groups`) are exactly what makes this
    /// number differ from a plain sum, so the test's inputs are chosen so
    /// every wrong weighting or wrong operator lands on a different total.
    #[test]
    fn dynamic_memory_ops_weights_row_and_body_scopes_by_their_trip_counts() {
        let mut traffic = EmitTraffic::default();
        traffic.frame.loads_transient = 1;
        traffic.row.loads_transient = 2;
        traffic.row.stores = 1;
        traffic.body.loads_transient = 4;
        traffic.body.stores = 1;

        assert_eq!(traffic.dynamic_memory_ops(6, 7), 229);
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

    /// The scaffold hooks carry no counter of their own, but aarch64's
    /// backend overrides both to seed and flush its literal pool — if the
    /// decorator ever stopped forwarding them, that pool would silently go
    /// missing on that target.
    #[test]
    fn scaffold_anchor_and_finish_forward_to_the_inner_backend() {
        let mut backend = RecordingBackend::new();
        let mut code = Vec::new();
        {
            let mut counting = Counting::new(&mut backend);
            counting.scaffold_anchor(&mut code);
            counting.scaffold_finish(&mut code);
        }

        assert_eq!(backend.scaffold_anchor_calls, 1);
        assert_eq!(backend.scaffold_finish_calls, 1);
    }

    /// The scaffold's own store/reload path (`slot_store`/`slot_load`) is
    /// separate from an instruction's operand resolution and must count
    /// independently of it.
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

    /// Registers to allocate in the pressure test: small enough that a
    /// deliberately wide expression cannot fit, on every tier.
    const TIGHT_POOL: u8 = 4;

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
                .compile(&arena, root)
                .expect("compile");
            let t = &result.traffic;
            let attributed = t.frame.bytes + t.row.bytes + t.body.bytes + t.scaffold.bytes;
            assert_eq!(
                attributed as usize,
                result.code.len(),
                "{terms} terms: {attributed} bytes attributed, {} emitted",
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
            .compile(&arena, root)
            .expect("compile");
        assert!(
            result.spill_count > 0,
            "24 values live against a {TIGHT_POOL}-register pool did not spill; \
             the scenario has stopped testing its subject"
        );
        let body = result.traffic.body;
        assert!(
            body.stores > 0,
            "values reached a frame slot with no store counted: {body:?}"
        );
        assert!(
            body.loads_transient + body.loads_kept > 0,
            "values were spilled and never reloaded: {body:?}"
        );
        assert!(
            body.instructions > 0,
            "a kernel with a body emitted no scheduled operation: {body:?}"
        );
    }

    /// The scaffold is the same code under every allocation of a kernel, so it
    /// is counted apart from the scopes rather than folded into one — a
    /// difference between two allocations must not be able to hide there.
    #[test]
    fn the_scaffolds_traffic_does_not_move_with_the_pool() {
        let (arena, root) = wide_live_range_kernel(24);
        let tight = EmitCtx::with_max_regs(TIGHT_POOL)
            .compile(&arena, root)
            .expect("compile");
        let loose = crate::emit::compile(&arena, root).expect("compile");
        assert_eq!(
            tight.traffic.scaffold, loose.traffic.scaffold,
            "the scaffold changed with the register budget"
        );
    }
}
