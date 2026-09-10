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
use super::{Binding, InstructionPlan, IsaBackend, Loc, Reg, Reload};
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
    type Cond = B::Cond;

    const JUMP: Self::Cond = B::JUMP;

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

    fn compare_mask(
        &mut self,
        code: &mut Vec<u8>,
        mask_reg: Reg,
        scratch: Option<Reg>,
        arm: super::SelectArm,
    ) -> Self::Cond {
        self.inner.compare_mask(code, mask_reg, scratch, arm)
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

    fn compare_counter(&mut self, code: &mut Vec<u8>, counter: super::Counter) -> Self::Cond {
        self.inner.compare_counter(code, counter)
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
