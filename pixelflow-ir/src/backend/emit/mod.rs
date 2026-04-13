//! JIT code emission for expression trees and DAGs.
//!
//! Two register allocation strategies:
//!
//! ## Sethi-Ullman (for trees)
//!
//! Register allocation emerges from tree structure.
//! Sethi-Ullman labeling computes minimum registers needed.
//! Register assignment is a FUNCTION of tree position (depth), not stateful allocation.
//!
//! ```text
//! emit : Expr × Depth → (Code, Reg)
//! ```
//!
//! No explicit alloc/free - the recursion depth IS the register.
//!
//! ## Linear Scan (for DAGs)
//!
//! For expressions with shared subexpressions (from e-graph extraction),
//! linear scan with Belady eviction handles register allocation in O(n×k).
//! With 22 scratch registers and typical live sets of 10-15, this is
//! effectively O(n) and produces near-optimal allocations.
//! See [`regalloc`] module.
//!
//! ## Spilling
//!
//! When `max_regs` is set lower than needed, we spill to stack:
//! - Spilled values stored via STR to [SP, #offset]
//! - Reloaded via LDR to dedicated reload register before use
//! - This lets ML models learn register pressure vs spill tradeoffs

pub mod aarch64;
pub mod executable;
pub mod regalloc;
pub mod x86_64;

use crate::kind::OpKind;

#[cfg(target_arch = "x86_64")]
use crate::expr::Expr;

use alloc::vec::Vec;

/// Constant pool: maps f32 bit patterns to pool indices.
///
/// Non-zero, non-FMOV-encodable constants are stored in a data section after
/// the RET instruction. Each entry is 16 bytes (the f32 splatted 4x to fill
/// a 128-bit NEON register). During code emission, these constants are loaded
/// with a single `LDR Qt, [X17, #offset]` instead of the 3-instruction
/// MOVZ+MOVK+DUP sequence.
pub(crate) struct ConstPool {
    /// Deduplicated entries: f32 bit patterns in pool order.
    entries: Vec<u32>,
    /// Map from f32 bits → pool index.
    index: alloc::collections::BTreeMap<u32, u16>,
}

impl ConstPool {
    /// Create an empty constant pool.
    pub(crate) fn new() -> Self {
        Self {
            entries: Vec::new(),
            index: alloc::collections::BTreeMap::new(),
        }
    }

    /// Build a constant pool from a schedule, pre-seeding constants that need pool entries.
    fn from_schedule(schedule: &[(regalloc::ValueId, ScheduledOp)]) -> Result<Self, &'static str> {
        let mut pool = Self::new();
        for (_, op) in schedule {
            if let ScheduledOp::Const(val) = op {
                if aarch64::needs_const_pool(*val) {
                    pool.push_f32(*val)?;
                }
            }
        }
        Ok(pool)
    }

    /// Insert an f32 into the pool (deduplicating by bit pattern) and return
    /// the byte offset for an `LDR Qt, [X17, #offset]` load.
    ///
    /// Zero and FMOV-encodable constants are NOT filtered here — callers
    /// that want the fast path should check `needs_const_pool` first.
    /// Builtin emitters call this unconditionally because every constant
    /// they use benefits from the pool (they are transcendental coefficients,
    /// never zero or FMOV-encodable).
    pub(crate) fn push_f32(&mut self, val: f32) -> Result<u16, &'static str> {
        let bits = val.to_bits();
        if let Some(&idx) = self.index.get(&bits) {
            return Ok(idx * 16);
        }
        let idx = self.entries.len();
        if idx >= 4096 {
            return Err(
                "constant pool overflow: exceeded 12-bit LDR offset limit (max 4095 entries)",
            );
        }
        self.entries.push(bits);
        self.index.insert(bits, idx as u16);
        Ok((idx * 16) as u16)
    }

    /// Get the byte offset for a constant, or None if it's not in the pool.
    fn offset_for(&self, val_bits: u32) -> Option<u16> {
        self.index.get(&val_bits).map(|&idx| idx * 16)
    }

    /// Returns true if the pool has any entries.
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Emit a constant load, using the constant pool when available.
///
/// Falls back to `emit_fmov_imm` for zero and FMOV-encodable values.
fn emit_const_load(code: &mut Vec<u8>, dst: Reg, val_bits: u32, pool: &ConstPool) {
    if let Some(offset) = pool.offset_for(val_bits) {
        aarch64::emit_ldr_q_x17(code, dst, offset);
    } else {
        let scratch = [Reg(28), Reg(29), Reg(30), Reg(31)];
        aarch64::emit_fmov_imm(code, dst, f32::from_bits(val_bits), scratch);
    }
}

/// Physical register index.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Reg(pub u8);

/// Location of a value: either in a register or spilled to stack.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Loc {
    /// Value is in a register.
    Reg(Reg),
    /// Value is spilled to stack at this offset from SP.
    Spill(u32),
}

impl Loc {
    /// Get the register, panicking if spilled.
    pub fn reg(self) -> Reg {
        match self {
            Loc::Reg(r) => r,
            Loc::Spill(off) => panic!("expected register, got spill slot {}", off),
        }
    }
}

/// Stack frame layout computed from register allocation.
///
/// Pure data: computed from the list of spilled values.
/// Maps each spilled ValueId to its stack offset and computes
/// the total frame size (16-byte aligned for AArch64 SP).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameLayout {
    /// Map from spilled ValueId to stack offset.
    pub spill_slots: alloc::collections::BTreeMap<regalloc::ValueId, u32>,
    /// Total frame size (16-byte aligned).
    pub frame_size: u32,
}

impl FrameLayout {
    /// Compute frame layout from spilled values.
    /// Pure function: spilled list → layout.
    pub fn from_allocation(spilled: &[regalloc::ValueId]) -> Result<Self, &'static str> {
        use alloc::collections::BTreeMap;
        // 2MB max frame — generous but prevents runaway allocations.
        const MAX_FRAME: u32 = 2 * 1024 * 1024;
        let mut spill_slots = BTreeMap::new();
        let mut offset = 0u32;
        for &v in spilled {
            if offset > MAX_FRAME - 16 {
                return Err("spill frame overflow: exceeds 2MB stack limit");
            }
            spill_slots.insert(v, offset);
            offset += 16; // 128-bit vector
        }
        let aligned = (offset + 15) & !15;
        Ok(Self {
            spill_slots,
            frame_size: aligned,
        })
    }
}

/// A concrete instruction to emit, with all registers resolved.
/// Pure data — no side effects, no mutation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedOp {
    /// No-op (variable already in input register).
    Nop,
    /// Load constant into dst.
    LoadConst { dst: Reg, val_bits: u32 },
    /// Unary: dst = op(src).
    Unary { op: OpKind, dst: Reg, src: Reg },
    /// Binary: dst = op(left, right).
    Binary {
        op: OpKind,
        dst: Reg,
        left: Reg,
        right: Reg,
    },
    /// Fused multiply-add via FMLA: dst = c + a*b.
    /// Requires dst to hold c before FMLA.
    FusedMulAdd { dst: Reg, a: Reg, b: Reg },
    /// Decomposed multiply-add: FMUL(dst, a, b) then reload c, then FADD(dst, dst, c).
    /// Used when a and b are both spilled (can't load both + c simultaneously).
    /// `c_deferred`: if Some, c must be reloaded *after* FMUL.
    DecomposedMulAdd {
        dst: Reg,
        a: Reg,
        b: Reg,
        c: Reg,
        c_deferred: Option<DeferredReload>,
    },
    /// BSL select: dst = mask ? if_true : if_false (mask pre-loaded into dst).
    Select {
        dst: Reg,
        if_true: Reg,
        if_false: Reg,
    },
    /// Clamp: FMIN(dst, val, hi), then reload lo, then FMAX(dst, dst, lo).
    /// `lo_deferred`: if Some, lo must be reloaded *after* FMIN.
    Clamp {
        dst: Reg,
        val: Reg,
        lo: Reg,
        hi: Reg,
        lo_deferred: Option<DeferredReload>,
    },
}

/// A deferred reload: value loaded mid-instruction (after a partial computation).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeferredReload {
    /// Load from stack at the given SP offset.
    FromStack(u32),
    /// Rematerialize a constant.
    Const(u32),
}

/// Reload instruction: load a value into a register.
///
/// Either reload from stack (spilled) or rematerialize a constant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reload {
    /// Load from stack at the given SP offset.
    FromStack { target: Reg, offset: u32 },
    /// Rematerialize a constant (emit FMOV immediate).
    Const { target: Reg, val_bits: u32 },
}

/// Store instruction: spill computed value to stack.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Store {
    pub src: Reg,
    pub offset: u32,
}

/// Fully resolved instruction with reloads and optional store.
#[derive(Clone, Debug)]
pub struct InstructionPlan {
    /// Reloads to emit before the main op.
    pub reloads: Vec<Reload>,
    /// The main operation.
    pub op: ResolvedOp,
    /// Optional MOV to set up accumulator/mask before main op.
    pub setup_mov: Option<(Reg, Reg)>,
    /// Store to emit after the main op (if dst is spilled).
    pub store: Option<Store>,
}

/// Emission context with register budget for ML training.
#[derive(Clone, Debug)]
pub struct EmitCtx {
    /// Maximum scratch registers before spilling (ML parameter).
    /// Default: 24 (v4-v27 on ARM64).
    pub max_regs: u8,
    /// Current spill offset from SP.
    pub spill_offset: u32,
    /// Number of spills performed (for cost modeling).
    pub spill_count: u32,
}

impl Default for EmitCtx {
    fn default() -> Self {
        Self {
            max_regs: 22, // v4-v25 (v26-v27 reserved for RELOAD_REGS)
            spill_offset: 0,
            spill_count: 0,
        }
    }
}

impl EmitCtx {
    /// Create context with custom register budget.
    pub fn with_max_regs(max_regs: u8) -> Self {
        Self {
            max_regs,
            ..Default::default()
        }
    }

    /// Allocate a spill slot, returns offset.
    pub fn alloc_spill(&mut self) -> u32 {
        let off = self.spill_offset;
        self.spill_offset += 16; // 128-bit vector
        self.spill_count += 1;
        off
    }
}

/// Input registers: X=v0, Y=v1, Z=v2, W=v3
pub const INPUT_REGS: [Reg; 4] = [Reg(0), Reg(1), Reg(2), Reg(3)];

/// First scratch register (after inputs)
pub const SCRATCH_BASE: u8 = 4;

/// Reload registers for spilled values.
///
/// When an operation's operands are spilled to stack, each is loaded into a
/// reload register. Two registers suffice: the destination register doubles
/// as a temporary for one operand (ARM reads all sources before writing dest).
///
/// Layout (ARM64):
///   v0-v3:   INPUT_REGS (X, Y, Z, W)
///   v4-v25:  allocatable scratch (max_regs=22)
///   v26-v27: RELOAD_REGS (spill reload area)
///   v28-v31: immediate construction scratch (emit_fmov_imm)
#[cfg(target_arch = "aarch64")]
pub const RELOAD_REGS: [Reg; 2] = [Reg(26), Reg(27)];

/// For backward compatibility with the tree-walk emitter.
#[cfg(target_arch = "aarch64")]
pub const RELOAD_REG: Reg = Reg(27);

/// Dedicated register for reloading spilled values (x86-64: xmm12)
#[cfg(target_arch = "x86_64")]
pub const RELOAD_REG: Reg = Reg(12);

/// x86-64 reload registers (placeholder — x86 DAG path not yet implemented).
#[cfg(target_arch = "x86_64")]
pub const RELOAD_REGS: [Reg; 2] = [Reg(11), Reg(12)];

/// Sethi-Ullman label: minimum registers needed to evaluate this subtree.
/// This is a catamorphism (fold) over the expression tree.
#[cfg(target_arch = "x86_64")]
pub fn needs(expr: &Expr) -> usize {
    match expr {
        // Leaves need 1 register to hold their value
        Expr::Var(_) => 1,
        Expr::Const(_) => 1,
        Expr::Param(i) => panic!(
            "Expr::Param({}) reached the JIT emitter — call substitute_params before compile()",
            i
        ),

        // Unary: same as child (result overwrites input)
        Expr::Unary(_, child) => needs(child),

        // Binary: Sethi-Ullman magic
        Expr::Binary(_, left, right) => {
            let l = needs(left);
            let r = needs(right);
            if l == r { l + 1 } else { l.max(r) }
        }

        // Ternary: need to hold all three, then combine
        Expr::Ternary(_, a, b, c) => {
            let na = needs(a);
            let nb = needs(b);
            let nc = needs(c);
            // Conservative: max + ties
            let max = na.max(nb).max(nc);
            if (na == nb) || (nb == nc) || (na == nc) {
                max + 1
            } else {
                max
            }
        }

        Expr::Nary(_, children) => {
            children.iter().map(needs).max().unwrap_or(0) + children.len() - 1
        }
    }
}

// =============================================================================
// Functional Emitter (x86-64)
// =============================================================================

#[cfg(target_arch = "x86_64")]
pub fn emit(expr: &Expr, depth: u8) -> Result<(Vec<u8>, Reg), &'static str> {
    use x86_64::*;

    match expr {
        Expr::Var(i) => {
            if *i as usize >= INPUT_REGS.len() {
                return Err("variable index out of range");
            }
            Ok((vec![], INPUT_REGS[*i as usize]))
        }

        Expr::Const(val) => {
            let dst = Reg(SCRATCH_BASE + depth);
            let mut code = Vec::new();
            let scratch = [Reg(13), Reg(14), Reg(15), Reg(15)];
            emit_const(&mut code, dst, *val, scratch);
            Ok((code, dst))
        }

        Expr::Unary(op, child) => {
            let (mut code, src) = emit(child, depth)?;
            let dst = Reg(SCRATCH_BASE + depth);
            let scratch = [Reg(13), Reg(14), Reg(15), Reg(15)];
            emit_unary(&mut code, *op, dst, src, scratch);
            Ok((code, dst))
        }

        Expr::Binary(op, left, right) => {
            let n_l = needs(left);
            let n_r = needs(right);
            let dst = Reg(SCRATCH_BASE + depth);

            if n_l >= n_r {
                let (mut code, l_reg) = emit(left, depth)?;
                let (r_code, r_reg) = emit(right, depth + 1)?;
                code.extend(r_code);
                match op {
                    OpKind::Atan2 => {
                        let scratch = [Reg(13), Reg(14), Reg(15), Reg(15)];
                        x86_64::emit_binary_transcendental(
                            &mut code, *op, dst, l_reg, r_reg, scratch,
                        );
                    }
                    _ => emit_binary(&mut code, *op, dst, l_reg, r_reg),
                }
                Ok((code, dst))
            } else {
                let (mut code, r_reg) = emit(right, depth)?;
                let (l_code, l_reg) = emit(left, depth + 1)?;
                code.extend(l_code);
                match op {
                    OpKind::Atan2 => {
                        let scratch = [Reg(13), Reg(14), Reg(15), Reg(15)];
                        x86_64::emit_binary_transcendental(
                            &mut code, *op, dst, l_reg, r_reg, scratch,
                        );
                    }
                    _ => emit_binary(&mut code, *op, dst, l_reg, r_reg),
                }
                Ok((code, dst))
            }
        }

        Expr::Ternary(op, a, b, c) => {
            let dst = Reg(SCRATCH_BASE + depth);

            match op {
                OpKind::MulAdd => {
                    // x86 doesn't have FMLA, use FMUL + FADD
                    let (mut code, a_reg) = emit(a, depth)?;
                    let (b_code, b_reg) = emit(b, depth + 1)?;
                    let (c_code, c_reg) = emit(c, depth + 2)?;

                    code.extend(b_code);
                    code.extend(c_code);

                    // dst = a * b
                    emit_binary(&mut code, OpKind::Mul, dst, a_reg, b_reg);
                    // dst = dst + c
                    emit_binary(&mut code, OpKind::Add, dst, dst, c_reg);
                    Ok((code, dst))
                }

                _ => Err("ternary emit not implemented"),
            }
        }

        Expr::Nary(_, _) => Err("Nary not supported in JIT"),
    }
}

// =============================================================================
// High-level API
// =============================================================================

/// Compile result with metadata for ML training.
pub struct CompileResult {
    /// The executable code.
    pub code: executable::ExecutableCode,
    /// Number of spills performed.
    pub spill_count: u32,
    /// Total stack space used for spills (bytes).
    pub spill_bytes: u32,
    /// Register budget that was used.
    pub max_regs: u8,
}

/// Compile an expression to executable code.
#[cfg(target_arch = "x86_64")]
pub fn compile(expr: &Expr) -> Result<executable::ExecutableCode, &'static str> {
    compile_dag(expr).map(|r| r.code)
}

/// Compile a DAG (from e-graph extraction) using graph coloring.
///
/// Unlike `compile`, this handles shared subexpressions properly.
/// Each unique subexpression is evaluated exactly once and its result
/// is kept in a register (or spilled) for all uses.
#[cfg(target_arch = "x86_64")]
pub fn compile_dag(expr: &Expr) -> Result<CompileResult, &'static str> {
    compile_dag_with_ctx(expr, EmitCtx::default())
}

/// Compile DAG with explicit register budget.
///
/// Pipeline as composition of pure morphisms:
/// ```text
/// Expr →[lower]→ Expr(primitive) →[linearize]→ Schedule →[analyze]→ Graph
///   →[color]→ Allocation →[layout]→ FrameLayout →[resolve]→ InstructionPlan
///   →[emit]→ MachineCode
/// ```
#[cfg(target_arch = "x86_64")]
pub fn compile_dag_with_ctx(expr: &Expr, ctx: EmitCtx) -> Result<CompileResult, &'static str> {
    let (schedule, _structural_cache, uses_map) = linearize_dag(expr);
    compile_from_schedule(schedule, uses_map, ctx)
}

/// Compile an [`ExprArena`] DAG to executable code.
///
/// This is the arena counterpart of [`compile`]. The arena's topological node
/// order eliminates the linearization pass entirely.
#[cfg(target_arch = "aarch64")]
pub fn compile_arena(
    arena: &crate::arena::ExprArena,
    root: crate::arena::ExprId,
) -> Result<executable::ExecutableCode, &'static str> {
    compile_arena_dag(arena, root).map(|r| r.code)
}

/// Compile an [`ExprArena`] DAG using graph coloring / linear scan register allocation.
///
/// Arena counterpart of [`compile_dag`]. The arena IS the linearized schedule,
/// so step 1 of the pipeline is free.
#[cfg(target_arch = "aarch64")]
pub fn compile_arena_dag(
    arena: &crate::arena::ExprArena,
    root: crate::arena::ExprId,
) -> Result<CompileResult, &'static str> {
    compile_arena_dag_with_ctx(arena, root, EmitCtx::default())
}

/// Compile an [`ExprArena`] DAG with explicit register budget.
///
/// Arena counterpart of [`compile_dag_with_ctx`]. Skips `linearize_dag`
/// entirely -- the arena's append-only structure guarantees topological order,
/// so `ExprId` maps 1:1 to `ValueId` for reachable nodes.
#[cfg(target_arch = "aarch64")]
pub fn compile_arena_dag_with_ctx(
    arena: &crate::arena::ExprArena,
    root: crate::arena::ExprId,
    ctx: EmitCtx,
) -> Result<CompileResult, &'static str> {
    let schedule = arena_to_schedule(arena, root);
    let uses_map = arena_to_uses(&schedule);
    compile_from_schedule(schedule, uses_map, ctx)
}

/// Compile an [`ExprArena`] DAG into a scanline kernel that processes an entire
/// row of pixels in a single call with no per-pixel Rust-JIT boundary crossing.
///
/// The emitted code contains its own loop: Y/Z/W stay in NEON registers across
/// all iterations (loop-invariant by construction), only X is loaded per pixel
/// from the input array. This eliminates the `extern "C"` function pointer
/// overhead that dominates single-pixel `KernelFn` performance.
///
/// When the expression contains X-invariant subexpressions (depending only on Y,
/// Z, or W), those are automatically hoisted into a setup block before the loop.
/// If there are no hoistable nodes, this degrades to the flat (non-hoisted) path.
///
/// Returns a [`ScanlineCompileResult`] containing the executable code and metadata.
#[cfg(target_arch = "aarch64")]
pub fn compile_arena_dag_scanline(
    arena: &crate::arena::ExprArena,
    root: crate::arena::ExprId,
) -> Result<ScanlineCompileResult, &'static str> {
    compile_arena_dag_scanline_hoisted(arena, root)
}

/// Compile a scanline kernel with explicit register budget.
#[cfg(target_arch = "aarch64")]
pub fn compile_arena_dag_scanline_with_ctx(
    arena: &crate::arena::ExprArena,
    root: crate::arena::ExprId,
    ctx: EmitCtx,
) -> Result<ScanlineCompileResult, &'static str> {
    let schedule = arena_to_schedule(arena, root);
    let uses_map = arena_to_uses(&schedule);
    compile_scanline_from_schedule(schedule, uses_map, ctx)
}

// =============================================================================
// Variance-aware hoisted scanline compilation
// =============================================================================

/// Maximum number of persistent (callee-saved) slots available for hoisted values.
///
/// This is a platform-specific constant: the backend decides how many physical
/// locations (callee-saved registers, dedicated stack slots, etc.) can hold
/// values that persist across the inner loop boundary.
///
/// - **aarch64**: 8 (NEON v8-v15 per AAPCS64)
/// - **x86-64**: TBD (will use callee-saved XMM/YMM or stack)
///
/// The hoisting logic is platform-agnostic; only the count matters. The backend
/// maps slot index 0..N to physical locations.
#[cfg(target_arch = "aarch64")]
pub const MAX_PERSISTENT_SLOTS: usize = 8;

#[cfg(target_arch = "x86_64")]
pub const MAX_PERSISTENT_SLOTS: usize = 0; // Not yet implemented

/// A two-phase schedule: setup (loop-invariant) then loop (per-pixel).
///
/// Produced by [`arena_to_hoisted_schedule`]. The `schedule` Vec contains all
/// nodes in variance-aware topological order: setup nodes at indices `0..split`,
/// loop nodes at indices `split..`. Within each phase, topological order is
/// maintained.
pub struct HoistedSchedule {
    /// Full schedule in variance-aware topological order.
    /// Setup nodes come first (indices `0..split`), loop nodes after (`split..`).
    pub schedule: Vec<(regalloc::ValueId, ScheduledOp)>,
    /// Index where setup ends and loop begins.
    pub split: usize,
    /// Number of setup values that cross the phase boundary (need persistent slots).
    /// Only setup values referenced by at least one loop node count here.
    pub num_hoisted: usize,
}

/// Build a two-phase hoisted schedule from an arena, partitioning nodes by variance.
///
/// The `partition` predicate decides which nodes go in the setup phase vs the loop.
/// It receives the [`Variance`] of each node and returns `true` for setup nodes.
///
/// # Design note: parameterized partition predicate
///
/// The partition predicate is a closure parameter (not hardcoded) so that different
/// lattice shapes can define different scope boundaries in the future. For example:
/// - Default: `|v| v.is_x_invariant() && !v.is_const()` hoists X-invariant non-constants
///   out of the pixel loop.
/// - Future: `|v| v.is_x_invariant() && v.is_y_invariant() && !v.is_const()` could
///   hoist XY-invariant values out of both pixel and scanline loops for tile-level
///   evaluation.
/// - Future: A custom lattice over {pixel, scanline, tile, frame} scopes could use
///   arbitrary predicate logic.
///
/// # Panics
///
/// Panics if a `Param` or `Nary` node is encountered in the arena.
pub fn arena_to_hoisted_schedule<F>(
    arena: &crate::arena::ExprArena,
    root: crate::arena::ExprId,
    partition: F,
) -> HoistedSchedule
where
    F: Fn(crate::variance::Variance) -> bool,
{
    use crate::arena::{ExprId, ExprNode};
    use regalloc::ValueId;

    let variance = crate::variance::compute_arena_variance(arena);
    let len = arena.len();

    // Mark reachable nodes from root.
    let mut reachable = alloc::vec![false; len];
    mark_reachable(arena, root, &mut reachable);

    // Classify each reachable node as setup or loop.
    let mut is_setup = alloc::vec![false; len];
    for idx in 0..len {
        if !reachable[idx] {
            continue;
        }
        let v = variance[idx];
        // Constants and variables are trivial -- the register allocator handles them
        // cheaply (precolored or rematerialized). Only non-trivial computation nodes
        // benefit from hoisting.
        let node = arena.node(ExprId(idx as u32));
        let is_trivial = matches!(
            node,
            ExprNode::Var(_) | ExprNode::Const(_) | ExprNode::Param(_)
        );
        is_setup[idx] = !is_trivial && partition(v);
    }

    // Build schedule in two passes: setup nodes first, then loop nodes.
    // Both phases maintain topological order (arena indices are already topological).
    let mut id_map = alloc::vec![ValueId(u32::MAX); len];
    let mut schedule = Vec::new();
    let mut next_id = 0u32;

    let map_child = |child: &ExprId, id_map: &[ValueId]| -> ValueId {
        let mapped = id_map[child.0 as usize];
        assert!(
            mapped.0 != u32::MAX,
            "arena_to_hoisted_schedule: child ExprId({}) not yet mapped -- \
             arena is not in topological order or child is unreachable",
            child.0
        );
        mapped
    };

    // Phase 1: emit setup nodes (those selected by the partition predicate).
    // We must also emit their transitive dependencies even if those deps are
    // trivial (Var, Const), because the setup nodes reference them.
    //
    // Since the arena is in topological order, we walk forward and emit any
    // reachable node whose transitive closure is entirely within setup OR
    // is a trivial leaf. The simplest correct approach: emit a node in setup
    // if (a) it IS a setup node, or (b) all of its users in the reachable
    // subgraph are setup nodes. But that requires a reverse-dependency analysis
    // which is expensive.
    //
    // Simpler approach: emit ALL reachable nodes in topological order but
    // partitioned. Trivial nodes (Var, Const) go to whichever phase first
    // needs them. Since both phases may reference the same Var, we emit
    // trivial nodes in their original position and let regalloc handle the
    // lifetime.
    //
    // Simplest correct approach: emit setup nodes first (maintaining topo order
    // among themselves), then loop nodes (maintaining topo order among themselves).
    // Trivial nodes go into whichever phase they belong to based on the partition
    // predicate -- but since we excluded trivials from setup, they always go to loop.
    // However, setup nodes may reference Var nodes. Since Var nodes are always at
    // lower indices than their users, and setup nodes are emitted before loop nodes,
    // we need Var/Const nodes referenced by setup nodes to be emitted first.
    //
    // Solution: any reachable node that is a dependency of a setup node AND is not
    // itself a setup node gets promoted to setup (forced into the setup phase).

    // Compute which nodes need to be in setup: setup nodes + their transitive deps.
    let mut in_setup_phase = alloc::vec![false; len];
    // First mark explicit setup nodes.
    for idx in 0..len {
        if reachable[idx] && is_setup[idx] {
            in_setup_phase[idx] = true;
        }
    }
    // Promote transitive dependencies of setup nodes into the setup phase.
    // Walk forward (topological order): if a node is in setup, ensure all its
    // children are too (they come earlier in the arena, so mark them and
    // re-scan). Alternatively, walk backward.
    // Backward pass: for each setup node, mark all children as setup.
    // Since children have lower indices, a single backward pass suffices.
    for idx in (0..len).rev() {
        if !in_setup_phase[idx] {
            continue;
        }
        let expr_id = ExprId(idx as u32);
        for child in arena.children(expr_id) {
            if reachable[child.0 as usize] {
                in_setup_phase[child.0 as usize] = true;
            }
        }
    }

    // Emit setup-phase nodes in topological order.
    for idx in 0..len {
        if !reachable[idx] || !in_setup_phase[idx] {
            continue;
        }
        let expr_id = ExprId(idx as u32);
        let node = arena.node(expr_id);
        let vid = ValueId(next_id);
        next_id += 1;
        id_map[idx] = vid;

        let sched_op = match node {
            ExprNode::Var(i) => ScheduledOp::Var(*i),
            ExprNode::Const(v) => ScheduledOp::Const(*v),
            ExprNode::Param(i) => panic!(
                "ExprNode::Param({i}) reached the JIT emitter -- \
                 call substitute_params before compile_arena()"
            ),
            ExprNode::Unary(op, child) => ScheduledOp::Unary(*op, map_child(child, &id_map)),
            ExprNode::Binary(op, a, b) => {
                ScheduledOp::Binary(*op, map_child(a, &id_map), map_child(b, &id_map))
            }
            ExprNode::Ternary(op, a, b, c) => ScheduledOp::Ternary(
                *op,
                map_child(a, &id_map),
                map_child(b, &id_map),
                map_child(c, &id_map),
            ),
            ExprNode::Nary(_, _, _) => panic!("Nary not supported in JIT arena compilation"),
        };
        schedule.push((vid, sched_op));
    }

    let split = schedule.len();

    // Phase 2: emit loop-phase nodes in topological order.
    for idx in 0..len {
        if !reachable[idx] || in_setup_phase[idx] {
            continue;
        }
        let expr_id = ExprId(idx as u32);
        let node = arena.node(expr_id);
        let vid = ValueId(next_id);
        next_id += 1;
        id_map[idx] = vid;

        let sched_op = match node {
            ExprNode::Var(i) => ScheduledOp::Var(*i),
            ExprNode::Const(v) => ScheduledOp::Const(*v),
            ExprNode::Param(i) => panic!(
                "ExprNode::Param({i}) reached the JIT emitter -- \
                 call substitute_params before compile_arena()"
            ),
            ExprNode::Unary(op, child) => ScheduledOp::Unary(*op, map_child(child, &id_map)),
            ExprNode::Binary(op, a, b) => {
                ScheduledOp::Binary(*op, map_child(a, &id_map), map_child(b, &id_map))
            }
            ExprNode::Ternary(op, a, b, c) => ScheduledOp::Ternary(
                *op,
                map_child(a, &id_map),
                map_child(b, &id_map),
                map_child(c, &id_map),
            ),
            ExprNode::Nary(_, _, _) => panic!("Nary not supported in JIT arena compilation"),
        };
        schedule.push((vid, sched_op));
    }

    // Count hoisted values: setup ValueIds referenced by loop nodes.
    // These are the values that must persist across the phase boundary.
    let setup_vids: alloc::collections::BTreeSet<regalloc::ValueId> =
        schedule[..split].iter().map(|(vid, _)| *vid).collect();

    let mut boundary_crossers: alloc::collections::BTreeSet<regalloc::ValueId> =
        alloc::collections::BTreeSet::new();

    for (_, sched_op) in &schedule[split..] {
        let operands: alloc::vec::Vec<regalloc::ValueId> = match sched_op {
            ScheduledOp::Var(_) | ScheduledOp::Const(_) => alloc::vec![],
            ScheduledOp::Unary(_, a) => alloc::vec![*a],
            ScheduledOp::Binary(_, a, b) => alloc::vec![*a, *b],
            ScheduledOp::Ternary(_, a, b, c) => alloc::vec![*a, *b, *c],
        };
        for op_vid in operands {
            if setup_vids.contains(&op_vid) {
                boundary_crossers.insert(op_vid);
            }
        }
    }

    let num_hoisted = boundary_crossers.len();

    HoistedSchedule {
        schedule,
        split,
        num_hoisted,
    }
}

/// Default partition predicate: X-invariant non-constants go to setup.
///
/// This hoists values that do not depend on X (the per-pixel coordinate) but
/// are not compile-time constants (which are rematerialized cheaply by the
/// register allocator). The result is that expressions depending only on Y, Z,
/// or W are computed once before the pixel loop.
#[inline]
pub fn default_hoist_predicate(v: crate::variance::Variance) -> bool {
    v.is_x_invariant() && !v.is_const()
}

/// Compile an [`ExprArena`] DAG into a scanline kernel with variance-aware hoisting.
///
/// This is the hoisting-enabled variant of [`compile_arena_dag_scanline`].
/// X-invariant non-constant subexpressions are computed once in a setup block
/// before the pixel loop and held in callee-saved NEON registers (v8-v15 on
/// aarch64). If there are no hoistable nodes (`split == 0`), this degrades
/// to the same behavior as the non-hoisted path.
///
/// Returns a [`ScanlineCompileResult`] containing the executable code and metadata.
#[cfg(target_arch = "aarch64")]
pub fn compile_arena_dag_scanline_hoisted(
    arena: &crate::arena::ExprArena,
    root: crate::arena::ExprId,
) -> Result<ScanlineCompileResult, &'static str> {
    let hoisted = arena_to_hoisted_schedule(arena, root, default_hoist_predicate);

    // If nothing to hoist, fall back to the non-hoisted path (no overhead).
    if hoisted.split == 0 || hoisted.num_hoisted == 0 {
        let uses_map = arena_to_uses(&hoisted.schedule);
        return compile_scanline_from_schedule(hoisted.schedule, uses_map, EmitCtx::default());
    }

    // Enforce platform limit on persistent slots.
    if hoisted.num_hoisted > MAX_PERSISTENT_SLOTS {
        // Too many hoisted values to fit in callee-saved registers.
        // Fall back to the non-hoisted path rather than silently miscompiling.
        let flat_schedule = arena_to_schedule(arena, root);
        let uses_map = arena_to_uses(&flat_schedule);
        return compile_scanline_from_schedule(flat_schedule, uses_map, EmitCtx::default());
    }

    compile_scanline_hoisted(hoisted)
}

/// Inner compilation for the hoisted scanline path.
///
/// Emits:
///   1. Extended prologue (save v8-v(7+num_hoisted) + GP callee saves)
///   2. Setup block: emit schedule[0..split] -- results in callee-saved NEON regs
///   3. Early-exit CBZ (count == 0)
///   4. Loop header: reload Y/Z/W, load X[i]
///   5. Loop body: emit schedule[split..]
///   6. Extended epilogue (store result, loop back, restore, RET)
#[cfg(target_arch = "aarch64")]
fn compile_scanline_hoisted(
    hoisted: HoistedSchedule,
) -> Result<ScanlineCompileResult, &'static str> {
    let HoistedSchedule {
        schedule,
        split,
        num_hoisted,
    } = hoisted;
    let num_hoisted_u8 = num_hoisted as u8;

    // --- Identify which setup ValueIds cross the boundary ---
    let setup_vids: alloc::collections::BTreeSet<regalloc::ValueId> =
        schedule[..split].iter().map(|(vid, _)| *vid).collect();

    let mut boundary_crossers_set: alloc::collections::BTreeSet<regalloc::ValueId> =
        alloc::collections::BTreeSet::new();

    for (_, sched_op) in &schedule[split..] {
        let operands: alloc::vec::Vec<regalloc::ValueId> = match sched_op {
            ScheduledOp::Var(_) | ScheduledOp::Const(_) => alloc::vec![],
            ScheduledOp::Unary(_, a) => alloc::vec![*a],
            ScheduledOp::Binary(_, a, b) => alloc::vec![*a, *b],
            ScheduledOp::Ternary(_, a, b, c) => alloc::vec![*a, *b, *c],
        };
        for op_vid in operands {
            if setup_vids.contains(&op_vid) {
                boundary_crossers_set.insert(op_vid);
            }
        }
    }

    // Map boundary-crossing ValueIds to persistent slot indices (0-based).
    // On aarch64, slot i maps to register v(8+i).
    let boundary_crossers: alloc::vec::Vec<regalloc::ValueId> =
        boundary_crossers_set.into_iter().collect();
    assert_eq!(
        boundary_crossers.len(),
        num_hoisted,
        "BUG: boundary crosser count mismatch: {} vs {}",
        boundary_crossers.len(),
        num_hoisted
    );

    // --- Register allocation ---
    // Precolor: input variables → v0-v3, hoisted boundary values → v8-v(7+num_hoisted).
    let mut precolored: alloc::collections::BTreeMap<regalloc::ValueId, Reg> =
        alloc::collections::BTreeMap::new();
    for (vid, op) in &schedule {
        if let ScheduledOp::Var(i) = op {
            if (*i as usize) >= INPUT_REGS.len() {
                return Err("variable index out of range");
            }
            precolored.insert(*vid, INPUT_REGS[*i as usize]);
        }
    }
    for (slot_idx, &vid) in boundary_crossers.iter().enumerate() {
        // Map persistent slot index to physical register: v(8 + slot_idx).
        precolored.insert(vid, Reg(8 + slot_idx as u8));
    }

    // Build uses map for the full schedule.
    let uses_map = arena_to_uses(&schedule);

    // Reduce scratch register budget: v8-v(7+num_hoisted) are reserved for hoisted values.
    // The default allocatable range is v4-v25 (22 regs). We shrink the top end by num_hoisted.
    let ctx = EmitCtx::default();
    let adjusted_max_regs = ctx.max_regs.saturating_sub(num_hoisted_u8);
    assert!(
        adjusted_max_regs >= 4,
        "register budget too small after reserving {num_hoisted} persistent slots: \
         only {adjusted_max_regs} scratch registers remain (need at least 4)"
    );

    let allocation = regalloc::linear_scan(
        &schedule,
        &uses_map,
        &precolored,
        adjusted_max_regs,
        SCRATCH_BASE,
    );

    let layout = FrameLayout::from_allocation(&allocation.spilled)?;
    let select_guards = analyze_select_guards(&schedule);

    // Build branch start/end structures.
    use alloc::collections::BTreeMap;
    let sched_len = schedule.len();

    struct PendingBranch {
        guard_idx: usize,
        arm: u8,
    }

    let mut branch_starts: alloc::vec::Vec<alloc::vec::Vec<PendingBranch>> =
        (0..sched_len).map(|_| alloc::vec::Vec::new()).collect();
    let mut branch_ends: alloc::vec::Vec<alloc::vec::Vec<usize>> =
        (0..sched_len).map(|_| alloc::vec::Vec::new()).collect();

    for (gi, guard) in select_guards.iter().enumerate() {
        if guard.true_range.0 != guard.true_range.1 {
            branch_starts[guard.true_range.0].push(PendingBranch {
                guard_idx: gi,
                arm: 0,
            });
            if guard.true_range.1 < sched_len {
                branch_ends[guard.true_range.1].push(gi);
            }
        }
        if guard.false_range.0 != guard.false_range.1 {
            branch_starts[guard.false_range.0].push(PendingBranch {
                guard_idx: gi,
                arm: 1,
            });
            if guard.false_range.1 < sched_len {
                branch_ends[guard.false_range.1].push(gi);
            }
        }
    }

    // Build dense lookup structures.
    let max_vid = schedule.iter().map(|(v, _)| v.0).max().unwrap_or(0) as usize;
    let mut reg_for: alloc::vec::Vec<Option<Reg>> = alloc::vec![None; max_vid + 1];
    for (&vid, &reg) in &allocation.assignment {
        reg_for[vid.0 as usize] = Some(reg);
    }
    let mut spill_for: alloc::vec::Vec<Option<u32>> = alloc::vec![None; max_vid + 1];
    for (&vid, &offset) in &layout.spill_slots {
        spill_for[vid.0 as usize] = Some(offset);
    }
    let mut remat_for: alloc::vec::Vec<Option<u32>> = alloc::vec![None; max_vid + 1];
    for (&vid, &bits) in &allocation.rematerialize {
        remat_for[vid.0 as usize] = Some(bits);
    }

    let mut pool = ConstPool::from_schedule(&schedule)?;

    const BUILTIN_HEADROOM: usize = 128;
    if pool.entries.len() + BUILTIN_HEADROOM > 4095 {
        return Err("expression too large: constant pool would exceed 12-bit LDR offset limit");
    }

    // === Emit code ===
    let mut code = Vec::new();

    // 1. Extended prologue: save GP callee saves + v8-v(7+num_hoisted).
    //    The setup block is emitted AFTER the prologue but BEFORE the CBZ/loop header.
    //    The prologue already positions the early-exit CBZ after the setup block location.
    let hoisted_prologue = aarch64::emit_scanline_prologue_hoisted(&mut code, num_hoisted_u8);

    // Spill frame for kernel computation (setup + loop both use it).
    if layout.frame_size > 0 {
        aarch64::emit_sub_sp(&mut code, layout.frame_size);
    }

    // ADR X17 placeholder for constant pool.
    let adr_patch_pos = aarch64::emit_adr_x17_placeholder(&mut code);

    // 2. Setup block: emit schedule[0..split].
    //    These nodes compute X-invariant values; results land in precolored
    //    callee-saved NEON registers (v8-v15) via the register allocation.
    let mut pending_patches: BTreeMap<(usize, u8), usize> = BTreeMap::new();

    for (sched_idx, (vid, sched_op)) in schedule[..split].iter().enumerate() {
        // Branch guards (same logic as non-hoisted path).
        if !branch_starts[sched_idx].is_empty() {
            let n_branches = branch_starts[sched_idx].len();
            for bi in 0..n_branches {
                let (guard_idx, arm) = {
                    let pb = &branch_starts[sched_idx][bi];
                    (pb.guard_idx, pb.arm)
                };
                let guard = &select_guards[guard_idx];
                let mask_reg = emit_resolve_dense(
                    &mut code,
                    guard.mask_vid,
                    RELOAD_REG,
                    &reg_for,
                    &spill_for,
                    &remat_for,
                    &pool,
                );
                match arm {
                    0 => {
                        let scratch = Reg(28);
                        aarch64::emit_umaxv(&mut code, scratch, mask_reg);
                        aarch64::emit_fmov_to_gp(&mut code, scratch);
                        let patch = aarch64::emit_cbz_w16(&mut code);
                        pending_patches.insert((guard_idx, 0), patch);
                    }
                    1 => {
                        let scratch = Reg(28);
                        aarch64::emit_uminv(&mut code, scratch, mask_reg);
                        aarch64::emit_fmov_to_gp(&mut code, scratch);
                        aarch64::emit32(&mut code, 0x2A3003F0); // MVN W16, W16
                        let patch = aarch64::emit_cbz_w16(&mut code);
                        pending_patches.insert((guard_idx, 1), patch);
                    }
                    _ => unreachable!(),
                }
            }
        }

        if !branch_ends[sched_idx].is_empty() {
            let n_ends = branch_ends[sched_idx].len();
            for ei in 0..n_ends {
                let gi = branch_ends[sched_idx][ei];
                if let Some(patch_pos) = pending_patches.remove(&(gi, 0)) {
                    let target = code.len();
                    aarch64::patch_cbz_cbnz(&mut code, patch_pos, target);
                }
                if let Some(patch_pos) = pending_patches.remove(&(gi, 1)) {
                    let target = code.len();
                    aarch64::patch_cbz_cbnz(&mut code, patch_pos, target);
                }
            }
        }

        let dst_loc = resolve_dst_loc_dense(*vid, &reg_for, &spill_for, &remat_for);
        let plan = resolve_operands(
            sched_op,
            dst_loc,
            &allocation.assignment,
            &layout.spill_slots,
            &allocation.rematerialize,
        )?;
        emit_instruction_plan(&mut code, &plan, &mut pool)?;
    }

    // --- The prologue's CBZ (early exit) was already emitted. ---
    // --- The prologue's loop header (reload Y/Z/W + LDR X[i]) follows. ---
    // At this point, the hoisted values are in v8..v(7+num_hoisted) and the
    // loop is ready to begin.

    // 3. Loop body: emit schedule[split..].
    for sched_idx in split..sched_len {
        let (vid, sched_op) = &schedule[sched_idx];

        // Branch guards.
        if !branch_starts[sched_idx].is_empty() {
            let n_branches = branch_starts[sched_idx].len();
            for bi in 0..n_branches {
                let (guard_idx, arm) = {
                    let pb = &branch_starts[sched_idx][bi];
                    (pb.guard_idx, pb.arm)
                };
                let guard = &select_guards[guard_idx];
                let mask_reg = emit_resolve_dense(
                    &mut code,
                    guard.mask_vid,
                    RELOAD_REG,
                    &reg_for,
                    &spill_for,
                    &remat_for,
                    &pool,
                );
                match arm {
                    0 => {
                        let scratch = Reg(28);
                        aarch64::emit_umaxv(&mut code, scratch, mask_reg);
                        aarch64::emit_fmov_to_gp(&mut code, scratch);
                        let patch = aarch64::emit_cbz_w16(&mut code);
                        pending_patches.insert((guard_idx, 0), patch);
                    }
                    1 => {
                        let scratch = Reg(28);
                        aarch64::emit_uminv(&mut code, scratch, mask_reg);
                        aarch64::emit_fmov_to_gp(&mut code, scratch);
                        aarch64::emit32(&mut code, 0x2A3003F0);
                        let patch = aarch64::emit_cbz_w16(&mut code);
                        pending_patches.insert((guard_idx, 1), patch);
                    }
                    _ => unreachable!(),
                }
            }
        }

        if !branch_ends[sched_idx].is_empty() {
            let n_ends = branch_ends[sched_idx].len();
            for ei in 0..n_ends {
                let gi = branch_ends[sched_idx][ei];
                if let Some(patch_pos) = pending_patches.remove(&(gi, 0)) {
                    let target = code.len();
                    aarch64::patch_cbz_cbnz(&mut code, patch_pos, target);
                }
                if let Some(patch_pos) = pending_patches.remove(&(gi, 1)) {
                    let target = code.len();
                    aarch64::patch_cbz_cbnz(&mut code, patch_pos, target);
                }
            }
        }

        let dst_loc = resolve_dst_loc_dense(*vid, &reg_for, &spill_for, &remat_for);
        let plan = resolve_operands(
            sched_op,
            dst_loc,
            &allocation.assignment,
            &layout.spill_slots,
            &allocation.rematerialize,
        )?;

        // Select short-circuit (same logic as compile_scanline_from_schedule).
        if let ScheduledOp::Ternary(OpKind::Select, mask_vid, true_vid, false_vid) = sched_op {
            let guard = select_guards.iter().find(|g| g.select_idx == sched_idx);
            if let Some(guard) = guard {
                let has_true_guard = guard.true_range.0 != guard.true_range.1;
                let has_false_guard = guard.false_range.0 != guard.false_range.1;

                if has_true_guard || has_false_guard {
                    let mask_reg = emit_resolve_dense(
                        &mut code, *mask_vid, RELOAD_REG, &reg_for, &spill_for, &remat_for, &pool,
                    );
                    let dst = match dst_loc {
                        Loc::Reg(r) => r,
                        Loc::Spill(_) => RELOAD_REGS[0],
                    };
                    let true_reg = reg_for.get(true_vid.0 as usize).and_then(|r| *r);
                    let false_reg = reg_for.get(false_vid.0 as usize).and_then(|r| *r);

                    let scratch = Reg(28);
                    aarch64::emit_umaxv(&mut code, scratch, mask_reg);
                    aarch64::emit_fmov_to_gp(&mut code, scratch);
                    let all_false_branch = aarch64::emit_cbz_w16(&mut code);

                    aarch64::emit_uminv(&mut code, scratch, mask_reg);
                    aarch64::emit_fmov_to_gp(&mut code, scratch);
                    aarch64::emit32(&mut code, 0x2A3003F0); // MVN W16, W16
                    let all_true_branch = aarch64::emit_cbz_w16(&mut code);

                    emit_instruction_plan(&mut code, &plan, &mut pool)?;
                    let skip_to_end = aarch64::emit_b(&mut code);

                    let all_false_target = code.len();
                    if let Some(freg) = false_reg {
                        emit_mov_reg(&mut code, dst, freg);
                    } else {
                        emit_resolve_dense(
                            &mut code, *false_vid, dst, &reg_for, &spill_for, &remat_for, &pool,
                        );
                    }
                    let skip_to_end2 = aarch64::emit_b(&mut code);

                    let all_true_target = code.len();
                    if let Some(treg) = true_reg {
                        emit_mov_reg(&mut code, dst, treg);
                    } else {
                        emit_resolve_dense(
                            &mut code, *true_vid, dst, &reg_for, &spill_for, &remat_for, &pool,
                        );
                    }

                    let end_target = code.len();
                    aarch64::patch_cbz_cbnz(&mut code, all_false_branch, all_false_target);
                    aarch64::patch_cbz_cbnz(&mut code, all_true_branch, all_true_target);
                    aarch64::patch_b(&mut code, skip_to_end, end_target);
                    aarch64::patch_b(&mut code, skip_to_end2, end_target);

                    if let Loc::Spill(offset) = dst_loc {
                        aarch64::emit_str_sp(&mut code, dst, offset);
                    }
                    continue;
                }
            }
        }

        emit_instruction_plan(&mut code, &plan, &mut pool)?;
    }

    assert!(
        pending_patches.is_empty(),
        "BUG: {} Select short-circuit branches were never patched",
        pending_patches.len()
    );

    // 4. Resolve result register.
    let root_vid = schedule.last().map(|(v, _)| *v).expect("empty schedule");
    let result_reg = emit_resolve_dense(
        &mut code, root_vid, RELOAD_REG, &reg_for, &spill_for, &remat_for, &pool,
    );

    // Restore spill frame before epilogue.
    if layout.frame_size > 0 {
        aarch64::emit_add_sp(&mut code, layout.frame_size);
    }

    // 5. Extended epilogue: store result, loop back, restore callee saves, RET.
    aarch64::emit_scanline_epilogue_hoisted(&mut code, &hoisted_prologue, result_reg);

    // 6. Constant pool (after all code including the RET).
    if !pool.is_empty() {
        let adr_pos = adr_patch_pos;
        let estimated_offset = (code.len() as i64) - (adr_pos as i64);
        let needs_adrp = estimated_offset >= (1 << 20) - 32;

        if needs_adrp {
            code.splice(adr_pos + 4..adr_pos + 4, [0, 0, 0, 0]);
        }

        while code.len() % 16 != 0 {
            code.push(0);
        }
        let pool_start = code.len();
        for &bits in &pool.entries {
            aarch64::emit_pool_entry(&mut code, bits);
        }
        aarch64::patch_adr_or_adrp(&mut code, adr_pos, pool_start, needs_adrp);
    }

    let exec = unsafe { executable::ExecutableCode::from_code(&code)? };

    Ok(ScanlineCompileResult {
        code: exec,
        spill_count: layout.spill_slots.len() as u32,
        spill_bytes: layout.frame_size,
        max_regs: ctx.max_regs,
    })
}

/// Compile result for a scanline kernel.
pub struct ScanlineCompileResult {
    /// The executable scanline kernel.
    pub code: executable::ExecutableCode,
    /// Number of register spills.
    pub spill_count: u32,
    /// Stack bytes used for spills.
    pub spill_bytes: u32,
    /// Register budget used.
    pub max_regs: u8,
}

/// Compile a scanline kernel from a schedule (shared backend for arena and `Expr` paths).
///
/// This wraps the kernel body in a loop:
///   1. Scanline prologue (save regs, shuffle Y/Z/W, set up loop vars)
///   2. Per-iteration: load X[i] from array, execute kernel body, store result
///   3. Increment, compare, branch back
///   4. Epilogue (restore regs, RET)
#[cfg(target_arch = "aarch64")]
fn compile_scanline_from_schedule(
    schedule: Vec<(regalloc::ValueId, ScheduledOp)>,
    uses_map: Vec<Vec<regalloc::ValueId>>,
    ctx: EmitCtx,
) -> Result<ScanlineCompileResult, &'static str> {
    // Register allocation (identical to single-pixel path).
    let mut precolored: alloc::collections::BTreeMap<regalloc::ValueId, Reg> =
        alloc::collections::BTreeMap::new();
    for (vid, op) in &schedule {
        if let ScheduledOp::Var(i) = op {
            if (*i as usize) >= INPUT_REGS.len() {
                return Err("variable index out of range");
            }
            precolored.insert(*vid, INPUT_REGS[*i as usize]);
        }
    }

    let allocation = regalloc::linear_scan(
        &schedule,
        &uses_map,
        &precolored,
        ctx.max_regs,
        SCRATCH_BASE,
    );

    let layout = FrameLayout::from_allocation(&allocation.spilled)?;
    let select_guards = analyze_select_guards(&schedule);

    // Build branch start/end structures (same as compile_from_schedule).
    use alloc::collections::BTreeMap;
    let sched_len = schedule.len();

    struct PendingBranch {
        guard_idx: usize,
        arm: u8,
    }

    let mut branch_starts: alloc::vec::Vec<alloc::vec::Vec<PendingBranch>> =
        (0..sched_len).map(|_| alloc::vec::Vec::new()).collect();
    let mut branch_ends: alloc::vec::Vec<alloc::vec::Vec<usize>> =
        (0..sched_len).map(|_| alloc::vec::Vec::new()).collect();

    for (gi, guard) in select_guards.iter().enumerate() {
        if guard.true_range.0 != guard.true_range.1 {
            branch_starts[guard.true_range.0].push(PendingBranch {
                guard_idx: gi,
                arm: 0,
            });
            if guard.true_range.1 < sched_len {
                branch_ends[guard.true_range.1].push(gi);
            }
        }
        if guard.false_range.0 != guard.false_range.1 {
            branch_starts[guard.false_range.0].push(PendingBranch {
                guard_idx: gi,
                arm: 1,
            });
            if guard.false_range.1 < sched_len {
                branch_ends[guard.false_range.1].push(gi);
            }
        }
    }

    // Build dense lookup structures.
    let max_vid = schedule.iter().map(|(v, _)| v.0).max().unwrap_or(0) as usize;
    let mut reg_for: alloc::vec::Vec<Option<Reg>> = alloc::vec![None; max_vid + 1];
    for (&vid, &reg) in &allocation.assignment {
        reg_for[vid.0 as usize] = Some(reg);
    }
    let mut spill_for: alloc::vec::Vec<Option<u32>> = alloc::vec![None; max_vid + 1];
    for (&vid, &offset) in &layout.spill_slots {
        spill_for[vid.0 as usize] = Some(offset);
    }
    let mut remat_for: alloc::vec::Vec<Option<u32>> = alloc::vec![None; max_vid + 1];
    for (&vid, &bits) in &allocation.rematerialize {
        remat_for[vid.0 as usize] = Some(bits);
    }

    let mut pool = ConstPool::from_schedule(&schedule)?;

    const BUILTIN_HEADROOM: usize = 128;
    if pool.entries.len() + BUILTIN_HEADROOM > 4095 {
        return Err("expression too large: constant pool would exceed 12-bit LDR offset limit");
    }

    // === Emit code ===
    let mut code = Vec::new();

    // 1. Scanline prologue: save regs, shuffle Y/Z/W, set up loop.
    let scanline_prologue = aarch64::emit_scanline_prologue(&mut code);

    // 2. Spill frame (inside the loop — stack allocated once in prologue).
    //    The scanline prologue already adjusted SP by 48 for callee saves.
    //    We need additional stack for kernel spills.
    if layout.frame_size > 0 {
        aarch64::emit_sub_sp(&mut code, layout.frame_size);
    }

    // ADR X17 placeholder for constant pool.
    let adr_patch_pos = aarch64::emit_adr_x17_placeholder(&mut code);

    // 3. Kernel body (identical to single-pixel emit).
    let mut pending_patches: BTreeMap<(usize, u8), usize> = BTreeMap::new();

    for (sched_idx, (vid, sched_op)) in schedule.iter().enumerate() {
        // Guard branch starts
        if !branch_starts[sched_idx].is_empty() {
            let n_branches = branch_starts[sched_idx].len();
            for bi in 0..n_branches {
                let (guard_idx, arm) = {
                    let pb = &branch_starts[sched_idx][bi];
                    (pb.guard_idx, pb.arm)
                };
                let guard = &select_guards[guard_idx];
                let mask_reg = emit_resolve_dense(
                    &mut code,
                    guard.mask_vid,
                    RELOAD_REG,
                    &reg_for,
                    &spill_for,
                    &remat_for,
                    &pool,
                );

                match arm {
                    0 => {
                        let scratch = Reg(28);
                        aarch64::emit_umaxv(&mut code, scratch, mask_reg);
                        aarch64::emit_fmov_to_gp(&mut code, scratch);
                        let patch = aarch64::emit_cbz_w16(&mut code);
                        pending_patches.insert((guard_idx, 0), patch);
                    }
                    1 => {
                        let scratch = Reg(28);
                        aarch64::emit_uminv(&mut code, scratch, mask_reg);
                        aarch64::emit_fmov_to_gp(&mut code, scratch);
                        aarch64::emit32(&mut code, 0x2A3003F0); // MVN W16, W16
                        let patch = aarch64::emit_cbz_w16(&mut code);
                        pending_patches.insert((guard_idx, 1), patch);
                    }
                    _ => unreachable!(),
                }
            }
        }

        // Guard branch ends
        if !branch_ends[sched_idx].is_empty() {
            let n_ends = branch_ends[sched_idx].len();
            for ei in 0..n_ends {
                let gi = branch_ends[sched_idx][ei];
                if let Some(patch_pos) = pending_patches.remove(&(gi, 0)) {
                    let target = code.len();
                    aarch64::patch_cbz_cbnz(&mut code, patch_pos, target);
                }
                if let Some(patch_pos) = pending_patches.remove(&(gi, 1)) {
                    let target = code.len();
                    aarch64::patch_cbz_cbnz(&mut code, patch_pos, target);
                }
            }
        }

        // Emit the instruction
        let dst_loc = resolve_dst_loc_dense(*vid, &reg_for, &spill_for, &remat_for);
        let plan = resolve_operands(
            sched_op,
            dst_loc,
            &allocation.assignment,
            &layout.spill_slots,
            &allocation.rematerialize,
        )?;

        // Select short-circuit (same logic as compile_from_schedule)
        if let ScheduledOp::Ternary(OpKind::Select, mask_vid, true_vid, false_vid) = sched_op {
            let guard = select_guards.iter().find(|g| g.select_idx == sched_idx);
            if let Some(guard) = guard {
                let has_true_guard = guard.true_range.0 != guard.true_range.1;
                let has_false_guard = guard.false_range.0 != guard.false_range.1;

                if has_true_guard || has_false_guard {
                    let mask_reg = emit_resolve_dense(
                        &mut code, *mask_vid, RELOAD_REG, &reg_for, &spill_for, &remat_for, &pool,
                    );
                    let dst = match dst_loc {
                        Loc::Reg(r) => r,
                        Loc::Spill(_) => RELOAD_REGS[0],
                    };
                    let true_reg = reg_for.get(true_vid.0 as usize).and_then(|r| *r);
                    let false_reg = reg_for.get(false_vid.0 as usize).and_then(|r| *r);

                    let scratch = Reg(28);
                    aarch64::emit_umaxv(&mut code, scratch, mask_reg);
                    aarch64::emit_fmov_to_gp(&mut code, scratch);
                    let all_false_branch = aarch64::emit_cbz_w16(&mut code);

                    aarch64::emit_uminv(&mut code, scratch, mask_reg);
                    aarch64::emit_fmov_to_gp(&mut code, scratch);
                    aarch64::emit32(&mut code, 0x2A3003F0); // MVN W16, W16
                    let all_true_branch = aarch64::emit_cbz_w16(&mut code);

                    emit_instruction_plan(&mut code, &plan, &mut pool)?;
                    let skip_to_end = aarch64::emit_b(&mut code);

                    let all_false_target = code.len();
                    if let Some(freg) = false_reg {
                        emit_mov_reg(&mut code, dst, freg);
                    } else {
                        emit_resolve_dense(
                            &mut code, *false_vid, dst, &reg_for, &spill_for, &remat_for, &pool,
                        );
                    }
                    let skip_to_end2 = aarch64::emit_b(&mut code);

                    let all_true_target = code.len();
                    if let Some(treg) = true_reg {
                        emit_mov_reg(&mut code, dst, treg);
                    } else {
                        emit_resolve_dense(
                            &mut code, *true_vid, dst, &reg_for, &spill_for, &remat_for, &pool,
                        );
                    }

                    let end_target = code.len();
                    aarch64::patch_cbz_cbnz(&mut code, all_false_branch, all_false_target);
                    aarch64::patch_cbz_cbnz(&mut code, all_true_branch, all_true_target);
                    aarch64::patch_b(&mut code, skip_to_end, end_target);
                    aarch64::patch_b(&mut code, skip_to_end2, end_target);

                    if let Loc::Spill(offset) = dst_loc {
                        aarch64::emit_str_sp(&mut code, dst, offset);
                    }
                    continue;
                }
            }
        }

        emit_instruction_plan(&mut code, &plan, &mut pool)?;
    }

    assert!(
        pending_patches.is_empty(),
        "BUG: {} Select short-circuit branches were never patched",
        pending_patches.len()
    );

    // 4. Move result to v0 (same as single-pixel epilogue).
    let root_vid = schedule.last().map(|(v, _)| *v).expect("empty schedule");
    let result_reg = emit_resolve_dense(
        &mut code, root_vid, RELOAD_REG, &reg_for, &spill_for, &remat_for, &pool,
    );

    // Restore spill frame before the scanline epilogue touches SP.
    if layout.frame_size > 0 {
        aarch64::emit_add_sp(&mut code, layout.frame_size);
    }

    // 5. Scanline epilogue: store result, loop back, restore, RET.
    aarch64::emit_scanline_epilogue(&mut code, &scanline_prologue, result_reg);

    // 6. Constant pool (after all code including the RET).
    if !pool.is_empty() {
        let adr_pos = adr_patch_pos;
        let estimated_offset = (code.len() as i64) - (adr_pos as i64);
        let needs_adrp = estimated_offset >= (1 << 20) - 32;

        if needs_adrp {
            code.splice(adr_pos + 4..adr_pos + 4, [0, 0, 0, 0]);
        }

        while code.len() % 16 != 0 {
            code.push(0);
        }
        let pool_start = code.len();
        for &bits in &pool.entries {
            aarch64::emit_pool_entry(&mut code, bits);
        }
        aarch64::patch_adr_or_adrp(&mut code, adr_pos, pool_start, needs_adrp);
    }

    let exec = unsafe { executable::ExecutableCode::from_code(&code)? };

    Ok(ScanlineCompileResult {
        code: exec,
        spill_count: layout.spill_slots.len() as u32,
        spill_bytes: layout.frame_size,
        max_regs: ctx.max_regs,
    })
}

// =============================================================================
// Amortized compilation workspace
// =============================================================================

/// Pre-allocated workspace for amortized JIT compilation.
///
/// Owns a reusable [`executable::CodeBuffer`] and pre-sized scratch vectors,
/// eliminating mmap/munmap syscalls and Vec allocations on every compile.
///
/// # Usage
///
/// ```ignore
/// let mut ws = CompileWorkspace::new(65536)?;
/// loop {
///     let func = ws.compile_arena(&arena, root)?;
///     // Use func... (valid until next compile_arena call)
/// }
/// ```
///
/// # Safety
///
/// The returned `KernelFn` is invalidated by the next call to `compile_arena`.
/// The caller must not hold references across compiles.
#[cfg(target_arch = "aarch64")]
pub struct CompileWorkspace {
    /// Reusable executable memory region.
    code_buf: executable::CodeBuffer,
    /// Scratch buffer for schedule entries.
    schedule: Vec<(regalloc::ValueId, ScheduledOp)>,
    /// Scratch buffer for uses map.
    uses_map: Vec<Vec<regalloc::ValueId>>,
    /// Scratch buffer for reachability marking.
    reachable: Vec<bool>,
    /// Scratch buffer for ExprId -> ValueId mapping.
    id_map: Vec<regalloc::ValueId>,
    /// Register budget.
    ctx: EmitCtx,
}

#[cfg(target_arch = "aarch64")]
impl CompileWorkspace {
    /// Create a new workspace with the given code buffer capacity (bytes).
    ///
    /// `code_capacity` is rounded up to the system page size. 64KB is
    /// generous for expressions up to ~500 nodes.
    pub fn new(code_capacity: usize) -> Result<Self, &'static str> {
        Ok(Self {
            code_buf: executable::CodeBuffer::new(code_capacity)?,
            schedule: Vec::with_capacity(256),
            uses_map: Vec::with_capacity(256),
            reachable: Vec::with_capacity(256),
            id_map: Vec::with_capacity(256),
            ctx: EmitCtx::default(),
        })
    }

    /// Set the register budget for subsequent compiles.
    pub fn set_max_regs(&mut self, max_regs: u8) {
        self.ctx.max_regs = max_regs;
    }

    /// Compile an arena DAG, returning a function pointer into reusable memory.
    ///
    /// # Safety
    ///
    /// The returned function pointer is valid only until the next call to
    /// `compile_arena`. The caller must ensure no concurrent use.
    pub unsafe fn compile_arena(
        &mut self,
        arena: &crate::arena::ExprArena,
        root: crate::arena::ExprId,
    ) -> Result<executable::KernelFn, &'static str> {
        // 1. Build schedule into pre-allocated buffers.
        self.schedule.clear();
        let len = arena.len();

        // Grow scratch buffers if needed (retains capacity across compiles).
        self.reachable.clear();
        self.reachable.resize(len, false);
        self.id_map.clear();
        self.id_map.resize(len, regalloc::ValueId(u32::MAX));

        mark_reachable(arena, root, &mut self.reachable);

        let id_map = &mut self.id_map;
        let schedule = &mut self.schedule;
        let reachable = &self.reachable;

        let mut next_id = 0u32;
        for idx in 0..len {
            if !reachable[idx] {
                continue;
            }
            let expr_id = crate::arena::ExprId(idx as u32);
            let node = arena.node(expr_id);
            let vid = regalloc::ValueId(next_id);
            next_id += 1;
            id_map[idx] = vid;

            let map_child = |child: &crate::arena::ExprId| -> regalloc::ValueId {
                let mapped = id_map[child.0 as usize];
                assert!(
                    mapped.0 != u32::MAX,
                    "CompileWorkspace: child ExprId({}) not mapped",
                    child.0
                );
                mapped
            };

            let sched_op = match node {
                crate::arena::ExprNode::Var(i) => ScheduledOp::Var(*i),
                crate::arena::ExprNode::Const(v) => ScheduledOp::Const(*v),
                crate::arena::ExprNode::Param(i) => panic!(
                    "ExprNode::Param({}) reached CompileWorkspace — \
                     call substitute_params before compile",
                    i
                ),
                crate::arena::ExprNode::Unary(op, child) => {
                    ScheduledOp::Unary(*op, map_child(child))
                }
                crate::arena::ExprNode::Binary(op, a, b) => {
                    ScheduledOp::Binary(*op, map_child(a), map_child(b))
                }
                crate::arena::ExprNode::Ternary(op, a, b, c) => {
                    ScheduledOp::Ternary(*op, map_child(a), map_child(b), map_child(c))
                }
                crate::arena::ExprNode::Nary(_, _, _) => {
                    panic!("Nary not supported in JIT arena compilation")
                }
            };
            schedule.push((vid, sched_op));
        }

        // 2. Build uses_map.
        self.uses_map.clear();
        for (_, op) in &self.schedule {
            let uses = match op {
                ScheduledOp::Var(_) | ScheduledOp::Const(_) => Vec::new(),
                ScheduledOp::Unary(_, a) => alloc::vec![*a],
                ScheduledOp::Binary(_, a, b) => alloc::vec![*a, *b],
                ScheduledOp::Ternary(_, a, b, c) => alloc::vec![*a, *b, *c],
            };
            self.uses_map.push(uses);
        }

        // 3. Compile via the shared backend (schedule + uses_map -> executable).
        //    compile_from_schedule owns the Vecs (moved), so swap them out
        //    and restore empty Vecs with retained capacity afterward.
        let schedule_owned = core::mem::take(&mut self.schedule);
        let uses_owned = core::mem::take(&mut self.uses_map);
        let result = compile_from_schedule(
            schedule_owned,
            uses_owned,
            EmitCtx {
                max_regs: self.ctx.max_regs,
                spill_offset: 0,
                spill_count: 0,
            },
        )?;

        // Copy the compiled bytes into our reusable code buffer.
        let bytes = result.code.as_bytes();
        // SAFETY: The caller guarantees no concurrent use and that previous
        // function pointers are not held across this call.
        let func: executable::KernelFn = unsafe { self.code_buf.write_code(bytes)? };
        Ok(func)
    }
}

/// Shared compilation backend: schedule + uses_map -> CompileResult.
///
/// Both `compile_dag_with_ctx` (Arc<Expr>) and `compile_arena_dag_with_ctx`
/// (arena) produce the same `(schedule, uses_map)` format and then converge here.
#[cfg(target_arch = "aarch64")]
fn compile_from_schedule(
    schedule: Vec<(regalloc::ValueId, ScheduledOp)>,
    uses_map: Vec<Vec<regalloc::ValueId>>,
    ctx: EmitCtx,
) -> Result<CompileResult, &'static str> {
    // 2. Collect pre-colored values (variables -> input registers).
    let mut precolored: alloc::collections::BTreeMap<regalloc::ValueId, Reg> =
        alloc::collections::BTreeMap::new();
    for (vid, op) in &schedule {
        if let ScheduledOp::Var(i) = op {
            if (*i as usize) >= INPUT_REGS.len() {
                return Err("variable index out of range");
            }
            precolored.insert(*vid, INPUT_REGS[*i as usize]);
        }
    }

    // 3. Register allocation: linear scan with Belady eviction.
    //
    // Our expressions are pure arithmetic DAGs in SSA form with ~22 scratch
    // registers and typical live sets of 10-15. Graph coloring gives optimal
    // register assignment (minimum registers) but builds an expensive O(n²)
    // interference graph. With abundant registers relative to live-set width,
    // linear scan produces identical or near-identical allocations at O(n×k)
    // cost — effectively O(n) for k=22.
    let allocation = regalloc::linear_scan(
        &schedule,
        &uses_map,
        &precolored,
        ctx.max_regs,
        SCRATCH_BASE,
    );

    // 5. Layout: Allocation → FrameLayout
    let layout = FrameLayout::from_allocation(&allocation.spilled)?;

    // 6. Analyze Select nodes for short-circuit opportunities
    let select_guards = analyze_select_guards(&schedule);

    // Build lookup structures for guard boundaries.
    // For each schedule index, track if it's the start/end of a guarded arm region.
    //
    // At true_range.start:  emit "if mask all-false, skip to true_range.end"
    // At false_range.start: emit "if mask all-true, skip to false_range.end"
    // At select_idx: the BSL itself now only runs when lanes diverge (mixed mask)
    use alloc::collections::BTreeMap;

    // Map: schedule_idx → (guard_ref_idx, arm_kind)
    // arm_kind: 0 = true-arm start, 1 = false-arm start
    // At Select node: emit short-circuit MOVs + branch around BSL
    struct PendingBranch {
        /// Index into select_guards
        guard_idx: usize,
        /// 0 = true arm, 1 = false arm
        arm: u8,
    }

    // Pre-compute: for each schedule index, what branches start/end here.
    // Dense Vecs indexed by schedule index replace BTreeMaps for O(1) access.
    let sched_len = schedule.len();
    let mut branch_starts: alloc::vec::Vec<alloc::vec::Vec<PendingBranch>> =
        (0..sched_len).map(|_| alloc::vec::Vec::new()).collect();
    let mut branch_ends: alloc::vec::Vec<alloc::vec::Vec<usize>> =
        (0..sched_len).map(|_| alloc::vec::Vec::new()).collect();

    for (gi, guard) in select_guards.iter().enumerate() {
        if guard.true_range.0 != guard.true_range.1 {
            branch_starts[guard.true_range.0].push(PendingBranch {
                guard_idx: gi,
                arm: 0,
            });
            if guard.true_range.1 < sched_len {
                branch_ends[guard.true_range.1].push(gi);
            }
        }
        if guard.false_range.0 != guard.false_range.1 {
            branch_starts[guard.false_range.0].push(PendingBranch {
                guard_idx: gi,
                arm: 1,
            });
            if guard.false_range.1 < sched_len {
                branch_ends[guard.false_range.1].push(gi);
            }
        }
    }

    // Build dense register/spill/remat lookup Vecs for O(1) access during emission.
    // These replace repeated BTreeMap lookups that occur 3-6 times per schedule node.
    let max_vid = schedule.iter().map(|(v, _)| v.0).max().unwrap_or(0) as usize;
    let mut reg_for: alloc::vec::Vec<Option<Reg>> = alloc::vec![None; max_vid + 1];
    for (&vid, &reg) in &allocation.assignment {
        reg_for[vid.0 as usize] = Some(reg);
    }
    let mut spill_for: alloc::vec::Vec<Option<u32>> = alloc::vec![None; max_vid + 1];
    for (&vid, &offset) in &layout.spill_slots {
        spill_for[vid.0 as usize] = Some(offset);
    }
    let mut remat_for: alloc::vec::Vec<Option<u32>> = alloc::vec![None; max_vid + 1];
    for (&vid, &bits) in &allocation.rematerialize {
        remat_for[vid.0 as usize] = Some(bits);
    }

    // 7. Build constant pool from schedule (pre-seed with ScheduledOp::Const values;
    //    builtins will add their polynomial coefficients incrementally during emission).
    let mut pool = ConstPool::from_schedule(&schedule)?;

    // Guard: bail early if the expression alone nearly fills the pool.
    // Builtins (sin, cos, atan2, etc.) add up to ~60 polynomial coefficients
    // during emission. If the expression constants + builtin headroom exceed
    // the aarch64 12-bit LDR offset limit, the expression is too large to compile.
    const BUILTIN_HEADROOM: usize = 128;
    if pool.entries.len() + BUILTIN_HEADROOM > 4095 {
        return Err("expression too large: constant pool would exceed 12-bit LDR offset limit");
    }

    // 8. Resolve + Emit: (Schedule, Allocation, Layout) → MachineCode
    let mut code = Vec::new();

    // Prologue: allocate stack frame if we spilled
    if layout.frame_size > 0 {
        aarch64::emit_sub_sp(&mut code, layout.frame_size);
    }

    // Always emit ADR X17 placeholder — builtins (sin, cos, atan2, etc.) add
    // their polynomial coefficients to the pool incrementally during emission,
    // so we can't know at this point whether the pool will be empty.
    let adr_patch_pos = aarch64::emit_adr_x17_placeholder(&mut code);

    // Track pending branch patches: (guard_idx, arm) → code position to patch
    let mut pending_patches: BTreeMap<(usize, u8), usize> = BTreeMap::new();

    // Emit each scheduled operation via resolve → plan → emit,
    // with Select short-circuit branches inserted at guard boundaries.
    for (sched_idx, (vid, sched_op)) in schedule.iter().enumerate() {
        // Check if any guard branches need to be emitted before this instruction
        if !branch_starts[sched_idx].is_empty() {
            // Drain indices to avoid borrow issues with `code`
            let n_branches = branch_starts[sched_idx].len();
            for bi in 0..n_branches {
                let (guard_idx, arm) = {
                    let pb = &branch_starts[sched_idx][bi];
                    (pb.guard_idx, pb.arm)
                };
                let guard = &select_guards[guard_idx];
                // Resolve mask register
                let mask_reg = emit_resolve_dense(
                    &mut code,
                    guard.mask_vid,
                    RELOAD_REG,
                    &reg_for,
                    &spill_for,
                    &remat_for,
                    &pool,
                );

                match arm {
                    0 => {
                        // True arm start: skip if mask is all-false (no true lanes).
                        // UMAXV S_scratch, V_mask.4S → if max==0, all lanes are false
                        // We use v28 as scratch for the horizontal reduction
                        let scratch = Reg(28);
                        aarch64::emit_umaxv(&mut code, scratch, mask_reg);
                        aarch64::emit_fmov_to_gp(&mut code, scratch);
                        let patch = aarch64::emit_cbz_w16(&mut code);
                        pending_patches.insert((guard_idx, 0), patch);
                    }
                    1 => {
                        // False arm start: skip if mask is all-true (no false lanes).
                        // UMINV S_scratch, V_mask.4S → if min==0xFFFFFFFF, all lanes are true
                        let scratch = Reg(28);
                        aarch64::emit_uminv(&mut code, scratch, mask_reg);
                        aarch64::emit_fmov_to_gp(&mut code, scratch);
                        // If UMINV result != 0, the minimum lane is nonzero, so check if it's all-ones.
                        // Actually: if all lanes are 0xFFFFFFFF, UMINV = 0xFFFFFFFF.
                        // We want to skip if ALL true, i.e., UMINV == 0xFFFFFFFF.
                        // 0xFFFFFFFF as u32 is -1. CBZ won't fire. CBNZ will fire (skip).
                        // But we need "skip if all-true" = "skip if UMINV == 0xFFFFFFFF"
                        // = "skip if W16 == 0xFFFFFFFF"
                        // We can use: CMP W16, #0; CSINV W16, WZR, WZR, NE; CBZ W16, skip
                        // Or simpler: MVN W16, W16; CBZ W16, skip (if ~0xFFFFFFFF == 0)
                        // MVN W16, W16 — bitwise NOT
                        aarch64::emit32(&mut code, 0x2A3003F0); // ORN W16, WZR, W16 = MVN W16, W16
                        let patch = aarch64::emit_cbz_w16(&mut code);
                        pending_patches.insert((guard_idx, 1), patch);
                    }
                    _ => unreachable!(),
                }
            }
        }

        // Check if any guard branches end at this instruction (patch targets)
        if !branch_ends[sched_idx].is_empty() {
            let n_ends = branch_ends[sched_idx].len();
            for ei in 0..n_ends {
                let gi = branch_ends[sched_idx][ei];
                // Patch the true-arm branch (arm=0) if it exists
                if let Some(patch_pos) = pending_patches.remove(&(gi, 0)) {
                    let target = code.len();
                    aarch64::patch_cbz_cbnz(&mut code, patch_pos, target);
                }
                // Patch the false-arm branch (arm=1) if it exists
                if let Some(patch_pos) = pending_patches.remove(&(gi, 1)) {
                    let target = code.len();
                    aarch64::patch_cbz_cbnz(&mut code, patch_pos, target);
                }
            }
        }

        // Emit the instruction itself
        let dst_loc = resolve_dst_loc_dense(*vid, &reg_for, &spill_for, &remat_for);
        let plan = resolve_operands(
            sched_op,
            dst_loc,
            &allocation.assignment,
            &layout.spill_slots,
            &allocation.rematerialize,
        )?;

        // For Select nodes that have guards, emit a short-circuit wrapper:
        // If mask is uniform, just MOV the correct arm to dst (BSL not needed).
        if let ScheduledOp::Ternary(OpKind::Select, mask_vid, true_vid, false_vid) = sched_op {
            let guard = select_guards.iter().find(|g| g.select_idx == sched_idx);
            if let Some(guard) = guard {
                let has_true_guard = guard.true_range.0 != guard.true_range.1;
                let has_false_guard = guard.false_range.0 != guard.false_range.1;

                if has_true_guard || has_false_guard {
                    // Resolve mask register for the all-lanes check
                    let mask_reg = emit_resolve_dense(
                        &mut code, *mask_vid, RELOAD_REG, &reg_for, &spill_for, &remat_for, &pool,
                    );

                    let dst = match dst_loc {
                        Loc::Reg(r) => r,
                        Loc::Spill(_) => RELOAD_REGS[0],
                    };

                    // Resolve true and false arm registers via dense O(1) lookup
                    let true_reg = reg_for.get(true_vid.0 as usize).and_then(|r| *r);
                    let false_reg = reg_for.get(false_vid.0 as usize).and_then(|r| *r);

                    // Check all-false: UMAXV → FMOV → CBZ (if max==0, mask is all-false → use false arm)
                    let scratch = Reg(28);
                    aarch64::emit_umaxv(&mut code, scratch, mask_reg);
                    aarch64::emit_fmov_to_gp(&mut code, scratch);
                    let all_false_branch = aarch64::emit_cbz_w16(&mut code);

                    // Check all-true: UMINV → FMOV → MVN → CBZ (if ~min==0, mask is all-true → use true arm)
                    aarch64::emit_uminv(&mut code, scratch, mask_reg);
                    aarch64::emit_fmov_to_gp(&mut code, scratch);
                    aarch64::emit32(&mut code, 0x2A3003F0); // MVN W16, W16
                    let all_true_branch = aarch64::emit_cbz_w16(&mut code);

                    // Mixed path: emit BSL (both arms already computed)
                    emit_instruction_plan(&mut code, &plan, &mut pool)?;
                    let skip_to_end = aarch64::emit_b(&mut code);

                    // All-false target: MOV dst ← false_arm
                    let all_false_target = code.len();
                    if let Some(freg) = false_reg {
                        emit_mov_reg(&mut code, dst, freg);
                    } else {
                        emit_resolve_dense(
                            &mut code, *false_vid, dst, &reg_for, &spill_for, &remat_for, &pool,
                        );
                    }
                    let skip_to_end2 = aarch64::emit_b(&mut code);

                    // All-true target: MOV dst ← true_arm
                    let all_true_target = code.len();
                    if let Some(treg) = true_reg {
                        emit_mov_reg(&mut code, dst, treg);
                    } else {
                        emit_resolve_dense(
                            &mut code, *true_vid, dst, &reg_for, &spill_for, &remat_for, &pool,
                        );
                    }

                    // End target (after all paths)
                    let end_target = code.len();

                    // Patch branches
                    aarch64::patch_cbz_cbnz(&mut code, all_false_branch, all_false_target);
                    aarch64::patch_cbz_cbnz(&mut code, all_true_branch, all_true_target);
                    aarch64::patch_b(&mut code, skip_to_end, end_target);
                    aarch64::patch_b(&mut code, skip_to_end2, end_target);

                    // Store if spilled
                    if let Loc::Spill(offset) = dst_loc {
                        aarch64::emit_str_sp(&mut code, dst, offset);
                    }

                    continue; // Skip the normal emit_instruction_plan below
                }
            }
        }

        emit_instruction_plan(&mut code, &plan, &mut pool)?;
    }

    // Verify no pending patches remain unresolved
    assert!(
        pending_patches.is_empty(),
        "BUG: {} Select short-circuit branches were never patched — \
         arm regions don't end before the schedule does",
        pending_patches.len()
    );

    // Epilogue: move result to v0, restore SP, RET
    let root = schedule.last().map(|(v, _)| *v).expect("empty schedule");
    let result_reg = emit_resolve_dense(
        &mut code, root, RELOAD_REG, &reg_for, &spill_for, &remat_for, &pool,
    );

    if result_reg.0 != 0 {
        emit_mov_reg(&mut code, Reg(0), result_reg);
    }

    if layout.frame_size > 0 {
        aarch64::emit_add_sp(&mut code, layout.frame_size);
    }

    // RET
    code.extend_from_slice(&0xD65F03C0u32.to_le_bytes());

    // Emit constant pool after RET and patch ADR X17.
    // If the pool ended up empty (no constants needed), the ADR is harmless —
    // it just sets X17 to an unused address. The 4-byte placeholder cost is
    // negligible compared to the code savings from pool loads.
    if !pool.is_empty() {
        let adr_pos = adr_patch_pos;
        // If the pool is going to be far away, upgrade ADR to ADRP + ADD.
        // We check against 1MB (1 << 20) minus a small margin for alignment padding.
        let estimated_offset = (code.len() as i64) - (adr_pos as i64);
        let needs_adrp = estimated_offset >= (1 << 20) - 32;

        if needs_adrp {
            // We need 8 bytes instead of 4. Insert 4 dummy bytes right after the ADR placeholder.
            // Since this is in the prologue, no PC-relative branches cross this insertion point,
            // meaning all previously emitted branches remain perfectly valid.
            code.splice(adr_pos + 4..adr_pos + 4, [0, 0, 0, 0]);
        }

        // Align pool start to 16 bytes (LDR Q requires 16-byte aligned data)
        while code.len() % 16 != 0 {
            code.push(0);
        }
        let pool_start = code.len();
        for &bits in &pool.entries {
            aarch64::emit_pool_entry(&mut code, bits);
        }
        aarch64::patch_adr_or_adrp(&mut code, adr_pos, pool_start, needs_adrp);
    }

    let exec = unsafe { executable::ExecutableCode::from_code(&code)? };

    Ok(CompileResult {
        code: exec,
        spill_count: layout.spill_slots.len() as u32,
        spill_bytes: layout.frame_size,
        max_regs: ctx.max_regs,
    })
}

/// Info about an operation in the schedule.
#[derive(Debug, Clone)]
pub enum ScheduledOp {
    /// Variable reference (input register)
    Var(u8),
    /// Constant value
    Const(f32),
    /// Unary op with input value
    Unary(OpKind, regalloc::ValueId),
    /// Binary op with input values
    Binary(OpKind, regalloc::ValueId, regalloc::ValueId),
    /// Ternary op with input values
    Ternary(
        OpKind,
        regalloc::ValueId,
        regalloc::ValueId,
        regalloc::ValueId,
    ),
}

/// Structural key for hash-consing during linearization.
///
/// After `lower::lower()` expands compound ops, cloned subtrees are
/// pointer-distinct but structurally identical. Keying by (op, child_ids)
/// collapses them back into shared schedule entries.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum StructuralKey {
    Var(u8),
    Const(u32), // f32::to_bits()
    Unary(OpKind, regalloc::ValueId),
    Binary(OpKind, regalloc::ValueId, regalloc::ValueId),
    Ternary(
        OpKind,
        regalloc::ValueId,
        regalloc::ValueId,
        regalloc::ValueId,
    ),
}

/// Linearize a DAG into a schedule with value IDs (iterative post-order).
///
/// Uses structural hash-consing: two expression nodes with the same op and
/// the same child ValueIds produce the same schedule entry. This collapses
/// the exponential blowup from `lower::lower()` cloning subtrees.
///
/// Returns (schedule, structural_key_to_value_id, uses_map)
///
/// `uses_map` is a dense `Vec` indexed by `ValueId.0`: for each value, the list
/// of child ValueIds it consumes. This replaces `BTreeMap<ValueId, Vec<ValueId>>`
/// for O(1) lookups on the hot path.
#[cfg(target_arch = "x86_64")]
fn linearize_dag(
    expr: &Expr,
) -> (
    Vec<(regalloc::ValueId, ScheduledOp)>,
    alloc::collections::BTreeMap<StructuralKey, regalloc::ValueId>,
    Vec<Vec<regalloc::ValueId>>,
) {
    use alloc::collections::BTreeMap;
    use regalloc::ValueId;

    let mut schedule = Vec::new();
    let mut ptr_cache: BTreeMap<*const Expr, ValueId> = BTreeMap::new();
    let mut structural_cache: BTreeMap<StructuralKey, ValueId> = BTreeMap::new();
    let mut uses_map: Vec<Vec<ValueId>> = Vec::new();
    let mut next_id = 0u32;

    // Work stack item: a node to process. On first visit we push children,
    // then re-push self. On second visit (all children done) we emit.
    // We track "expanded" to know if children were already pushed.
    enum WorkItem<'a> {
        /// First visit: need to check cache or push children + re-push as Emit.
        Enter(&'a Expr),
        /// Children are done — build structural key and emit.
        Emit(&'a Expr),
    }

    let mut work_stack: Vec<WorkItem<'_>> = Vec::new();
    work_stack.push(WorkItem::Enter(expr));

    while let Some(item) = work_stack.pop() {
        match item {
            WorkItem::Enter(node) => {
                let ptr = node as *const Expr;
                if ptr_cache.contains_key(&ptr) {
                    continue; // already fully processed
                }

                match node {
                    // Leaves: emit immediately, no children to process.
                    Expr::Var(_) | Expr::Const(_) => {
                        work_stack.push(WorkItem::Emit(node));
                    }
                    Expr::Param(i) => panic!(
                        "Expr::Param({}) reached the JIT emitter — call substitute_params before compile()",
                        i
                    ),
                    Expr::Unary(_, child) => {
                        work_stack.push(WorkItem::Emit(node));
                        work_stack.push(WorkItem::Enter(child));
                    }
                    Expr::Binary(_, left, right) => {
                        work_stack.push(WorkItem::Emit(node));
                        work_stack.push(WorkItem::Enter(right));
                        work_stack.push(WorkItem::Enter(left));
                    }
                    Expr::Ternary(_, a, b, c) => {
                        work_stack.push(WorkItem::Emit(node));
                        work_stack.push(WorkItem::Enter(c));
                        work_stack.push(WorkItem::Enter(b));
                        work_stack.push(WorkItem::Enter(a));
                    }
                    Expr::Nary(_, _) => panic!("Nary not supported in DAG compilation"),
                }
            }
            WorkItem::Emit(node) => {
                let ptr = node as *const Expr;
                // Could have been processed by a duplicate Enter that resolved earlier.
                if let Some(&id) = ptr_cache.get(&ptr) {
                    let _ = id; // already cached
                    continue;
                }

                let (key, sched_op, child_ids) = match node {
                    Expr::Var(i) => {
                        let key = StructuralKey::Var(*i);
                        (key, ScheduledOp::Var(*i as u8), Vec::new())
                    }
                    Expr::Const(v) => {
                        let key = StructuralKey::Const(v.to_bits());
                        (key, ScheduledOp::Const(*v), Vec::new())
                    }
                    Expr::Param(i) => panic!(
                        "Expr::Param({}) reached the JIT emitter — call substitute_params before compile()",
                        i
                    ),
                    Expr::Unary(op, child) => {
                        let child_id = *ptr_cache.get(&(child.as_ref() as *const Expr))
                            .expect("linearize_dag: child not yet processed for Unary — broken traversal order");
                        let key = StructuralKey::Unary(*op, child_id);
                        (
                            key,
                            ScheduledOp::Unary(*op, child_id),
                            alloc::vec![child_id],
                        )
                    }
                    Expr::Binary(op, left, right) => {
                        let l_id = *ptr_cache.get(&(left.as_ref() as *const Expr))
                            .expect("linearize_dag: left child not yet processed for Binary — broken traversal order");
                        let r_id = *ptr_cache.get(&(right.as_ref() as *const Expr))
                            .expect("linearize_dag: right child not yet processed for Binary — broken traversal order");
                        let key = StructuralKey::Binary(*op, l_id, r_id);
                        (
                            key,
                            ScheduledOp::Binary(*op, l_id, r_id),
                            alloc::vec![l_id, r_id],
                        )
                    }
                    Expr::Ternary(op, a, b, c) => {
                        let a_id = *ptr_cache.get(&(a.as_ref() as *const Expr))
                            .expect("linearize_dag: child a not yet processed for Ternary — broken traversal order");
                        let b_id = *ptr_cache.get(&(b.as_ref() as *const Expr))
                            .expect("linearize_dag: child b not yet processed for Ternary — broken traversal order");
                        let c_id = *ptr_cache.get(&(c.as_ref() as *const Expr))
                            .expect("linearize_dag: child c not yet processed for Ternary — broken traversal order");
                        let key = StructuralKey::Ternary(*op, a_id, b_id, c_id);
                        (
                            key,
                            ScheduledOp::Ternary(*op, a_id, b_id, c_id),
                            alloc::vec![a_id, b_id, c_id],
                        )
                    }
                    Expr::Nary(_, _) => panic!("Nary not supported in DAG compilation"),
                };

                // Structural dedup.
                if let Some(&existing_id) = structural_cache.get(&key) {
                    ptr_cache.insert(ptr, existing_id);
                    continue;
                }

                let my_id = ValueId(next_id);
                next_id += 1;

                // Grow the dense uses_map to cover my_id.
                assert_eq!(
                    my_id.0 as usize,
                    uses_map.len(),
                    "ValueId should be sequential"
                );
                uses_map.push(child_ids);

                schedule.push((my_id, sched_op));
                structural_cache.insert(key, my_id);
                ptr_cache.insert(ptr, my_id);
            }
        }
    }

    (schedule, structural_cache, uses_map)
}

// =============================================================================
// Arena to Schedule (zero-cost linearization)
// =============================================================================

/// Mark nodes reachable from `root` via DFS.
///
/// The arena may contain garbage nodes from junkify passes; only nodes
/// transitively referenced by `root` should appear in the schedule.
fn mark_reachable(
    arena: &crate::arena::ExprArena,
    root: crate::arena::ExprId,
    reachable: &mut [bool],
) {
    let mut stack = alloc::vec![root];
    while let Some(id) = stack.pop() {
        let idx = id.0 as usize;
        if reachable[idx] {
            continue;
        }
        reachable[idx] = true;
        for child in arena.children(id) {
            if !reachable[child.0 as usize] {
                stack.push(child);
            }
        }
    }
}

/// Build a schedule directly from an [`ExprArena`], skipping `linearize_dag`.
///
/// The arena stores nodes in topological order (children before parents by
/// construction). We filter to reachable nodes, remap `ExprId` to `ValueId`,
/// and translate `ExprNode` to `ScheduledOp`.
///
/// # Panics
///
/// Panics if a `Param` or `Nary` node is encountered (these are not expected
/// in JIT compilation).
fn arena_to_schedule(
    arena: &crate::arena::ExprArena,
    root: crate::arena::ExprId,
) -> Vec<(regalloc::ValueId, ScheduledOp)> {
    use crate::arena::{ExprId, ExprNode};
    use regalloc::ValueId;

    let len = arena.len();
    let mut reachable = alloc::vec![false; len];
    mark_reachable(arena, root, &mut reachable);

    // ExprId to ValueId mapping. u32::MAX = unmapped (unreachable).
    let mut id_map = alloc::vec![ValueId(u32::MAX); len];
    let mut schedule = Vec::new();
    let mut next_id = 0u32;

    for idx in 0..len {
        if !reachable[idx] {
            continue;
        }
        let expr_id = ExprId(idx as u32);
        let node = arena.node(expr_id);
        let vid = ValueId(next_id);
        next_id += 1;
        id_map[idx] = vid;

        let map_child = |child: &ExprId| -> ValueId {
            let mapped = id_map[child.0 as usize];
            assert!(
                mapped.0 != u32::MAX,
                "arena_to_schedule: child ExprId({}) not yet mapped -- \
                 arena is not in topological order or child is unreachable",
                child.0
            );
            mapped
        };

        let sched_op = match node {
            ExprNode::Var(i) => ScheduledOp::Var(*i),
            ExprNode::Const(v) => ScheduledOp::Const(*v),
            ExprNode::Param(i) => panic!(
                "ExprNode::Param({}) reached the JIT emitter -- \
                 call substitute_params before compile_arena()",
                i
            ),
            ExprNode::Unary(op, child) => ScheduledOp::Unary(*op, map_child(child)),
            ExprNode::Binary(op, a, b) => ScheduledOp::Binary(*op, map_child(a), map_child(b)),
            ExprNode::Ternary(op, a, b, c) => {
                ScheduledOp::Ternary(*op, map_child(a), map_child(b), map_child(c))
            }
            ExprNode::Nary(_, _, _) => panic!("Nary not supported in JIT arena compilation"),
        };
        schedule.push((vid, sched_op));
    }
    schedule
}

/// Build uses_map from a schedule (children of each operation).
///
/// Returns a dense `Vec` indexed by `ValueId.0`: for each value, the list
/// of child `ValueId`s it consumes.
fn arena_to_uses(schedule: &[(regalloc::ValueId, ScheduledOp)]) -> Vec<Vec<regalloc::ValueId>> {
    schedule
        .iter()
        .map(|(_, op)| match op {
            ScheduledOp::Var(_) | ScheduledOp::Const(_) => Vec::new(),
            ScheduledOp::Unary(_, a) => alloc::vec![*a],
            ScheduledOp::Binary(_, a, b) => alloc::vec![*a, *b],
            ScheduledOp::Ternary(_, a, b, c) => alloc::vec![*a, *b, *c],
        })
        .collect()
}

// =============================================================================
// Select Short-Circuit Analysis
// =============================================================================

/// Describes a Select node's short-circuit structure in the schedule.
///
/// For `Select(mask, if_true, if_false)`, identifies contiguous ranges of
/// schedule entries that are exclusive to each arm (not shared with mask
/// or the other arm). These ranges can be guarded by conditional branches.
#[derive(Debug, Clone)]
struct SelectGuard {
    /// Schedule index of the Select node itself.
    select_idx: usize,
    /// ValueId of the mask operand (already computed before arms).
    mask_vid: regalloc::ValueId,
    /// Range of schedule indices exclusive to the true arm: [true_start, true_end).
    /// Empty if true_start == true_end.
    true_range: (usize, usize),
    /// Range of schedule indices exclusive to the false arm: [false_start, false_end).
    false_range: (usize, usize),
}

/// Compute the transitive dependencies of a ValueId in the schedule.
///
/// `schedule_ops` is a dense Vec indexed by `ValueId.0`, pre-built by the
/// caller so each lookup is O(1) instead of O(n).
fn transitive_deps(
    vid: regalloc::ValueId,
    schedule_ops: &[Option<ScheduledOp>],
) -> alloc::collections::BTreeSet<regalloc::ValueId> {
    use alloc::collections::BTreeSet;

    let mut deps = BTreeSet::new();
    let mut worklist = alloc::vec![vid];
    while let Some(v) = worklist.pop() {
        if !deps.insert(v) {
            continue;
        }
        // O(1) lookup via dense Vec indexed by ValueId.0
        if let Some(Some(sop)) = schedule_ops.get(v.0 as usize) {
            match sop {
                ScheduledOp::Var(_) | ScheduledOp::Const(_) => {}
                ScheduledOp::Unary(_, c) => {
                    worklist.push(*c);
                }
                ScheduledOp::Binary(_, l, r) => {
                    worklist.push(*l);
                    worklist.push(*r);
                }
                ScheduledOp::Ternary(_, a, b, c) => {
                    worklist.push(*a);
                    worklist.push(*b);
                    worklist.push(*c);
                }
            }
        }
    }
    deps
}

/// Analyze the schedule for Select nodes and compute short-circuit guard ranges.
///
/// For each Select, partitions schedule entries into:
/// - Shared: needed by mask, or by both arms (must always execute)
/// - True-exclusive: only needed by the true arm (skip if mask all-false)
/// - False-exclusive: only needed by the false arm (skip if mask all-true)
///
/// Returns guards sorted by select_idx (ascending).
fn analyze_select_guards(schedule: &[(regalloc::ValueId, ScheduledOp)]) -> Vec<SelectGuard> {
    use alloc::collections::BTreeSet;

    let mut guards = Vec::new();

    if schedule.is_empty() {
        return guards;
    }

    // Build dense lookup: schedule_ops[vid.0] = Some(&ScheduledOp) for O(1) child traversal.
    // ValueIds are sequential starting from 0 (guaranteed by linearize_dag / arena_to_schedule).
    let max_vid = schedule.iter().map(|(v, _)| v.0).max().unwrap_or(0) as usize;
    let mut schedule_ops: alloc::vec::Vec<Option<ScheduledOp>> = alloc::vec![None; max_vid + 1];
    for (vid, sop) in schedule {
        schedule_ops[vid.0 as usize] = Some(sop.clone());
    }

    // Build dense lookup: vid_to_sched_idx[vid.0] = schedule position (u32::MAX = absent).
    let mut vid_to_sched_idx: alloc::vec::Vec<usize> = alloc::vec![usize::MAX; max_vid + 1];
    for (i, (vid, _)) in schedule.iter().enumerate() {
        vid_to_sched_idx[vid.0 as usize] = i;
    }

    for (i, (_vid, sop)) in schedule.iter().enumerate() {
        if let ScheduledOp::Ternary(OpKind::Select, mask_vid, true_vid, false_vid) = sop {
            // Compute transitive deps for each subtree using the dense O(1) lookup
            let mask_deps = transitive_deps(*mask_vid, &schedule_ops);
            let true_deps = transitive_deps(*true_vid, &schedule_ops);
            let false_deps = transitive_deps(*false_vid, &schedule_ops);

            // True-exclusive: in true_deps but NOT in mask_deps and NOT in false_deps
            let true_exclusive: BTreeSet<regalloc::ValueId> = true_deps
                .difference(&mask_deps)
                .copied()
                .collect::<BTreeSet<_>>()
                .difference(&false_deps)
                .copied()
                .collect();

            // False-exclusive: in false_deps but NOT in mask_deps and NOT in true_deps
            let false_exclusive: BTreeSet<regalloc::ValueId> = false_deps
                .difference(&mask_deps)
                .copied()
                .collect::<BTreeSet<_>>()
                .difference(&true_deps)
                .copied()
                .collect();

            // Map to schedule indices using dense O(1) lookup
            let true_indices: BTreeSet<usize> = true_exclusive
                .iter()
                .filter_map(|v| {
                    let idx = *vid_to_sched_idx.get(v.0 as usize)?;
                    if idx == usize::MAX { None } else { Some(idx) }
                })
                .collect();
            let false_indices: BTreeSet<usize> = false_exclusive
                .iter()
                .filter_map(|v| {
                    let idx = *vid_to_sched_idx.get(v.0 as usize)?;
                    if idx == usize::MAX { None } else { Some(idx) }
                })
                .collect();

            // Get contiguous ranges (min..max+1)
            let true_range = if true_indices.is_empty() {
                (i, i) // empty range
            } else {
                let start = *true_indices
                    .iter()
                    .next()
                    .expect("non-empty set has first element");
                let end = *true_indices
                    .iter()
                    .next_back()
                    .expect("non-empty set has last element")
                    + 1;
                // Verify contiguity: all indices in [start, end) should be either
                // true_exclusive or shared. If there are false_exclusive nodes
                // interleaved, we can't use a simple branch guard.
                let has_false_in_range = (start..end).any(|idx| false_indices.contains(&idx));
                if has_false_in_range {
                    (i, i) // can't guard — fall back to BSL
                } else {
                    (start, end)
                }
            };

            let false_range = if false_indices.is_empty() {
                (i, i)
            } else {
                let start = *false_indices
                    .iter()
                    .next()
                    .expect("non-empty set has first element");
                let end = *false_indices
                    .iter()
                    .next_back()
                    .expect("non-empty set has last element")
                    + 1;
                let has_true_in_range = (start..end).any(|idx| true_indices.contains(&idx));
                if has_true_in_range {
                    (i, i)
                } else {
                    (start, end)
                }
            };

            // Only create a guard if at least one arm has exclusive nodes
            if true_range.0 != true_range.1 || false_range.0 != false_range.1 {
                guards.push(SelectGuard {
                    select_idx: i,
                    mask_vid: *mask_vid,
                    true_range,
                    false_range,
                });
            }
        }
    }

    guards
}

/// Emit MOV (vector register copy) — used by emit_instruction_plan.
#[cfg(target_arch = "aarch64")]
fn emit_mov_reg(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    if dst.0 != src.0 {
        // ORR Vd.16B, Vn.16B, Vn.16B
        let inst = 0x4EA01C00u32 | (dst.0 as u32) | ((src.0 as u32) << 5) | ((src.0 as u32) << 16);
        code.extend_from_slice(&inst.to_le_bytes());
    }
}

/// Resolve the destination location for a value in the DAG.
fn resolve_dst_loc(
    vid: regalloc::ValueId,
    assignment: &alloc::collections::BTreeMap<regalloc::ValueId, Reg>,
    spill_slots: &alloc::collections::BTreeMap<regalloc::ValueId, u32>,
    rematerialize: &alloc::collections::BTreeMap<regalloc::ValueId, u32>,
) -> Loc {
    if let Some(&reg) = assignment.get(&vid) {
        Loc::Reg(reg)
    } else if let Some(&offset) = spill_slots.get(&vid) {
        Loc::Spill(offset)
    } else if rematerialize.contains_key(&vid) {
        // Rematerialized values don't need a destination — they're constants
        // that will be re-emitted on each use. The "definition" is a no-op
        // (the ScheduledOp::Const will be skipped by resolve_operands).
        // Use Reg(0) as a dummy — won't be written.
        Loc::Reg(RELOAD_REGS[0])
    } else {
        panic!(
            "value {:?} has no assignment, spill slot, or rematerialize entry",
            vid
        );
    }
}

/// Resolve a scheduled operation into a concrete instruction plan.
///
/// This is a PURE FUNCTION: no mutation, no side effects, no code emission.
/// Given the scheduled op, destination location, register assignments, and
/// spill slots, it computes exactly which registers to use and what
/// reload/store instructions are needed.
///
/// The two reload registers (RELOAD_REGS[0], RELOAD_REGS[1]) are used as:
///   - RELOAD_REGS[0] (v26): destination when dst is spilled, also temporary
///   - RELOAD_REGS[1] (v27): operand reload, never aliases dst
///
/// The "dst-as-temporary" trick: ARM NEON reads all sources before writing
/// the destination, so loading a spilled operand into dst is safe for binary
/// ops (the instruction reads before it writes).
pub fn resolve_operands(
    op: &ScheduledOp,
    dst_loc: Loc,
    assignment: &alloc::collections::BTreeMap<regalloc::ValueId, Reg>,
    spill_slots: &alloc::collections::BTreeMap<regalloc::ValueId, u32>,
    rematerialize: &alloc::collections::BTreeMap<regalloc::ValueId, u32>,
) -> Result<InstructionPlan, &'static str> {
    let tmp_op = RELOAD_REGS[1]; // v27 — always safe for operand reload

    // Compute destination: a real register, or RELOAD_REGS[0] (v26) if spilled.
    let dst = match dst_loc {
        Loc::Reg(r) => r,
        Loc::Spill(_) => RELOAD_REGS[0],
    };

    let mut reloads = Vec::new();
    let mut setup_mov = None;

    // Resolve a value to its register, or plan a reload from stack/constant into `target`.
    let resolve = |v: regalloc::ValueId, target: Reg, reloads: &mut Vec<Reload>| -> Reg {
        if let Some(&reg) = assignment.get(&v) {
            reg
        } else if let Some(&bits) = rematerialize.get(&v) {
            reloads.push(Reload::Const {
                target,
                val_bits: bits,
            });
            target
        } else if let Some(&offset) = spill_slots.get(&v) {
            reloads.push(Reload::FromStack { target, offset });
            target
        } else {
            panic!(
                "value {:?} not found in assignment, spill slots, or rematerialize",
                v
            );
        }
    };

    let resolved_op = match op {
        ScheduledOp::Var(_) => {
            // Precolored to input register — no code needed.
            ResolvedOp::Nop
        }
        ScheduledOp::Const(val) => ResolvedOp::LoadConst {
            dst,
            val_bits: val.to_bits(),
        },
        ScheduledOp::Unary(op_kind, child) => {
            let src = resolve(*child, tmp_op, &mut reloads);
            ResolvedOp::Unary {
                op: *op_kind,
                dst,
                src,
            }
        }
        ScheduledOp::Binary(op_kind, left, right) => {
            let l_spilled = !assignment.contains_key(left);
            let r_spilled = !assignment.contains_key(right);

            let (l_reg, r_reg) = match (l_spilled, r_spilled) {
                (false, false) => (assignment[left], assignment[right]),
                (true, false) => {
                    let l = resolve(*left, tmp_op, &mut reloads);
                    (l, assignment[right])
                }
                (false, true) => {
                    let r = resolve(*right, tmp_op, &mut reloads);
                    (assignment[left], r)
                }
                (true, true) => {
                    // Both spilled. Left → dst (temporary), right → tmp_op.
                    let l = resolve(*left, dst, &mut reloads);
                    let r = resolve(*right, tmp_op, &mut reloads);
                    (l, r)
                }
            };
            ResolvedOp::Binary {
                op: *op_kind,
                dst,
                left: l_reg,
                right: r_reg,
            }
        }
        ScheduledOp::Ternary(op_kind, a, b, c) => {
            let a_spilled = !assignment.contains_key(a);
            let b_spilled = !assignment.contains_key(b);

            match op_kind {
                OpKind::MulAdd => {
                    // MulAdd(a, b, c) = a*b + c.
                    if a_spilled && b_spilled {
                        // Decompose: FMUL(dst, a, b) then FADD(dst, dst, c).
                        // a → dst (temp), b → tmp_op loaded upfront.
                        // c is loaded AFTER FMUL (may reuse tmp_op).
                        let a_reg = resolve(*a, dst, &mut reloads);
                        let b_reg = resolve(*b, tmp_op, &mut reloads);
                        // c is deferred — don't add to upfront reloads.
                        let (c_reg, c_deferred) = if let Some(&reg) = assignment.get(c) {
                            (reg, None)
                        } else if let Some(&bits) = rematerialize.get(c) {
                            (tmp_op, Some(DeferredReload::Const(bits)))
                        } else if let Some(&offset) = spill_slots.get(c) {
                            (tmp_op, Some(DeferredReload::FromStack(offset)))
                        } else {
                            panic!(
                                "value {:?} not found in assignment, spill slots, or rematerialize",
                                c
                            );
                        };
                        ResolvedOp::DecomposedMulAdd {
                            dst,
                            a: a_reg,
                            b: b_reg,
                            c: c_reg,
                            c_deferred,
                        }
                    } else {
                        // FMLA path: dst += a * b, so dst must hold c first.
                        // At most 1 of {a, b} is spilled → tmp_op handles it.
                        let c_reg = resolve(*c, dst, &mut reloads);
                        if dst.0 != c_reg.0 {
                            setup_mov = Some((dst, c_reg));
                        }
                        let a_reg = resolve(*a, tmp_op, &mut reloads);
                        let b_reg = resolve(*b, tmp_op, &mut reloads);
                        ResolvedOp::FusedMulAdd {
                            dst,
                            a: a_reg,
                            b: b_reg,
                        }
                    }
                }
                OpKind::Select => {
                    // BSL: dst must hold mask (a).
                    // BSL is 3-input RMW: b and c must not alias each other.
                    if b_spilled && !assignment.contains_key(c) {
                        return Err("Select with both if_true and if_false spilled not supported");
                    }
                    let a_reg = resolve(*a, tmp_op, &mut reloads);
                    if dst.0 != a_reg.0 {
                        setup_mov = Some((dst, a_reg));
                    }
                    let b_reg = resolve(*b, tmp_op, &mut reloads);
                    let c_reg = resolve(*c, tmp_op, &mut reloads);
                    ResolvedOp::Select {
                        dst,
                        if_true: b_reg,
                        if_false: c_reg,
                    }
                }
                OpKind::Clamp => {
                    // clamp(a, lo=b, hi=c) = max(lo, min(a, c))
                    // Decomposed: FMIN(dst, val, hi) then FMAX(dst, dst, lo).
                    // val and hi loaded upfront; lo is deferred (loaded after FMIN).
                    let c_spilled = !assignment.contains_key(c);
                    let (val_reg, hi_reg) = if a_spilled && c_spilled {
                        // Both need reload — use dst-as-temp for val, tmp_op for hi.
                        let a_reg = resolve(*a, dst, &mut reloads);
                        let c_reg = resolve(*c, tmp_op, &mut reloads);
                        (a_reg, c_reg)
                    } else {
                        // At most one spilled — tmp_op suffices.
                        let a_reg = resolve(*a, tmp_op, &mut reloads);
                        let c_reg = resolve(*c, tmp_op, &mut reloads);
                        (a_reg, c_reg)
                    };
                    // lo is deferred — loaded after FMIN.
                    let (lo_reg, lo_deferred) = if let Some(&reg) = assignment.get(b) {
                        (reg, None)
                    } else if let Some(&bits) = rematerialize.get(b) {
                        (tmp_op, Some(DeferredReload::Const(bits)))
                    } else if let Some(&offset) = spill_slots.get(b) {
                        (tmp_op, Some(DeferredReload::FromStack(offset)))
                    } else {
                        panic!(
                            "value {:?} not found in assignment, spill slots, or rematerialize",
                            b
                        );
                    };
                    ResolvedOp::Clamp {
                        dst,
                        val: val_reg,
                        lo: lo_reg,
                        hi: hi_reg,
                        lo_deferred,
                    }
                }
                _ => return Err("unsupported ternary op in DAG compilation"),
            }
        }
    };

    // Store result if destination is spilled.
    let store = if let Loc::Spill(offset) = dst_loc {
        Some(Store { src: dst, offset })
    } else {
        None
    };

    Ok(InstructionPlan {
        reloads,
        op: resolved_op,
        setup_mov,
        store,
    })
}

/// Emit machine code for a resolved instruction plan.
///
/// This is a DETERMINISTIC DISPATCH: given a plan, emit the exact
/// instructions. No decisions are made here — all decisions were
/// made by resolve_operands.
#[cfg(target_arch = "aarch64")]
fn emit_instruction_plan(
    code: &mut Vec<u8>,
    plan: &InstructionPlan,
    pool: &mut ConstPool,
) -> Result<(), &'static str> {
    use aarch64::*;

    // 1. Emit reloads (from stack or rematerialized constants)
    for reload in &plan.reloads {
        match reload {
            Reload::FromStack { target, offset } => {
                emit_ldr_sp(code, *target, *offset);
            }
            Reload::Const { target, val_bits } => {
                emit_const_load(code, *target, *val_bits, pool);
            }
        }
    }

    // 2. Emit setup MOV (for FMLA accumulator or BSL mask)
    if let Some((dst, src)) = plan.setup_mov {
        emit_mov_reg(code, dst, src);
    }

    // 3. Emit main op
    match &plan.op {
        ResolvedOp::Nop => {}
        ResolvedOp::LoadConst { dst, val_bits } => {
            emit_const_load(code, *dst, *val_bits, pool);
        }
        ResolvedOp::Unary { op, dst, src } => {
            let scratch = [Reg(28), Reg(29), Reg(30), Reg(31)];
            emit_unary(code, pool, *op, *dst, *src, scratch)?;
        }
        ResolvedOp::Binary {
            op,
            dst,
            left,
            right,
        } => match op {
            OpKind::Pow | OpKind::Hypot | OpKind::Atan2 => {
                let scratch = [Reg(28), Reg(29), Reg(30), Reg(31)];
                aarch64::emit_binary_transcendental(code, pool, *op, *dst, *left, *right, scratch)?;
            }
            _ => emit_binary(code, *op, *dst, *left, *right),
        },
        ResolvedOp::FusedMulAdd { dst, a, b } => {
            // setup_mov already placed c into dst
            emit_fmla(code, *dst, *a, *b);
        }
        ResolvedOp::DecomposedMulAdd {
            dst,
            a,
            b,
            c,
            c_deferred,
        } => {
            // FMUL(dst, a, b) — consumes a and b (loaded upfront).
            emit_fmul(code, *dst, *a, *b);
            // Reload c after FMUL (c may reuse tmp_op which held b).
            emit_deferred(code, *c, c_deferred.as_ref(), pool);
            // FADD(dst, dst, c)
            emit_fadd(code, *dst, *dst, *c);
        }
        ResolvedOp::Select {
            dst,
            if_true,
            if_false,
        } => {
            // setup_mov already placed mask into dst
            emit_bsl(code, *dst, *if_true, *if_false);
        }
        ResolvedOp::Clamp {
            dst,
            val,
            lo,
            hi,
            lo_deferred,
        } => {
            // FMIN(dst, val, hi) — consumes val and hi (loaded upfront).
            emit_fmin(code, *dst, *val, *hi);
            // Reload lo after FMIN (lo may reuse tmp_op which held val or hi).
            emit_deferred(code, *lo, lo_deferred.as_ref(), pool);
            // FMAX(dst, dst, lo)
            emit_fmax(code, *dst, *dst, *lo);
        }
    }

    // 4. Emit store
    if let Some(store) = &plan.store {
        emit_str_sp(code, store.src, store.offset);
    }

    Ok(())
}

/// Resolve a value to a register at emit time, emitting a reload if necessary.
///
/// Checks in order: register assignment, rematerialize (constant load),
/// spill slot (stack load). Panics if value is not found anywhere.
fn emit_resolve(
    code: &mut Vec<u8>,
    vid: regalloc::ValueId,
    target: Reg,
    assignment: &alloc::collections::BTreeMap<regalloc::ValueId, Reg>,
    spill_slots: &alloc::collections::BTreeMap<regalloc::ValueId, u32>,
    rematerialize: &alloc::collections::BTreeMap<regalloc::ValueId, u32>,
    pool: &ConstPool,
) -> Reg {
    if let Some(&reg) = assignment.get(&vid) {
        reg
    } else if let Some(&bits) = rematerialize.get(&vid) {
        emit_const_load(code, target, bits, pool);
        target
    } else if let Some(&offset) = spill_slots.get(&vid) {
        aarch64::emit_ldr_sp(code, target, offset);
        target
    } else {
        panic!(
            "value {:?} has no register, spill slot, or rematerialize entry",
            vid
        );
    }
}

/// Emit a deferred reload: either from stack or rematerialized constant.
fn emit_deferred(
    code: &mut Vec<u8>,
    target: Reg,
    deferred: Option<&DeferredReload>,
    pool: &ConstPool,
) {
    match deferred {
        Some(DeferredReload::FromStack(offset)) => {
            aarch64::emit_ldr_sp(code, target, *offset);
        }
        Some(DeferredReload::Const(val_bits)) => {
            emit_const_load(code, target, *val_bits, pool);
        }
        None => {}
    }
}

/// Dense-Vec variant of `resolve_dst_loc` for use in the hot emit loop.
///
/// Accepts pre-built `Vec<Option<T>>` slices indexed by `ValueId.0` for O(1)
/// lookups instead of O(log n) BTreeMap lookups.
fn resolve_dst_loc_dense(
    vid: regalloc::ValueId,
    reg_for: &[Option<Reg>],
    spill_for: &[Option<u32>],
    remat_for: &[Option<u32>],
) -> Loc {
    let idx = vid.0 as usize;
    if let Some(Some(reg)) = reg_for.get(idx) {
        Loc::Reg(*reg)
    } else if let Some(Some(offset)) = spill_for.get(idx) {
        Loc::Spill(*offset)
    } else if matches!(remat_for.get(idx), Some(Some(_))) {
        // Rematerialized values don't need a destination — they're constants
        // that will be re-emitted on each use. The "definition" is a no-op.
        // Use RELOAD_REGS[0] as a dummy — won't be written.
        Loc::Reg(RELOAD_REGS[0])
    } else {
        panic!(
            "value {:?} has no assignment, spill slot, or rematerialize entry",
            vid
        );
    }
}

/// Dense-Vec variant of `emit_resolve` for use in the hot emit loop.
///
/// Accepts pre-built `Vec<Option<T>>` slices indexed by `ValueId.0` for O(1)
/// lookups instead of O(log n) BTreeMap lookups.
#[cfg(target_arch = "aarch64")]
fn emit_resolve_dense(
    code: &mut Vec<u8>,
    vid: regalloc::ValueId,
    target: Reg,
    reg_for: &[Option<Reg>],
    spill_for: &[Option<u32>],
    remat_for: &[Option<u32>],
    pool: &ConstPool,
) -> Reg {
    let idx = vid.0 as usize;
    if let Some(Some(reg)) = reg_for.get(idx) {
        *reg
    } else if let Some(Some(bits)) = remat_for.get(idx) {
        emit_const_load(code, target, *bits, pool);
        target
    } else if let Some(Some(offset)) = spill_for.get(idx) {
        aarch64::emit_ldr_sp(code, target, *offset);
        target
    } else {
        panic!(
            "value {:?} has no register, spill slot, or rematerialize entry",
            vid
        );
    }
}

/// Compile a DAG to executable code (x86-64).
///
/// Alias for `compile` on x86-64; graph coloring is not yet implemented for
/// this target. Expressions generated by ExprGenerator are trees (no sharing),
/// so the Sethi-Ullman emitter is correct and sufficient.
#[cfg(target_arch = "x86_64")]
pub fn compile_dag(expr: &Expr) -> Result<CompileResult, &'static str> {
    const MAX_DEPTH: usize = 64;
    let depth = expr.depth();
    if depth > MAX_DEPTH {
        return Err("expression too deep");
    }

    let code = compile(expr)?;
    Ok(CompileResult {
        code,
        spill_count: 0,
        spill_bytes: 0,
        max_regs: EmitCtx::default().max_regs,
    })
}

/// Compile an expression to executable code (x86-64).
#[cfg(target_arch = "x86_64")]
pub fn compile(expr: &Expr) -> Result<executable::ExecutableCode, &'static str> {
    // Lower compound ops to primitives
    let lowered = lower::lower(expr);

    let (mut code, result_reg) = emit(&lowered, 0)?;

    // Move result to xmm0 if not already there
    if result_reg.0 != 0 {
        x86_64::emit_movaps(&mut code, Reg(0), result_reg);
    }

    // RET
    code.push(0xC3);

    unsafe { executable::ExecutableCode::from_code(&code) }
}

/// Compile an [`ExprArena`] DAG to executable code (x86-64).
///
/// x86-64 DAG compilation is not yet implemented. This converts the arena
/// to an [`Expr`] tree and delegates to the Sethi-Ullman emitter.
#[cfg(target_arch = "x86_64")]
pub fn compile_arena(
    arena: &crate::arena::ExprArena,
    root: crate::arena::ExprId,
) -> Result<executable::ExecutableCode, &'static str> {
    compile_arena_dag(arena, root).map(|r| r.code)
}

/// Compile an [`ExprArena`] DAG using graph coloring (x86-64 stub).
///
/// Converts the arena to an [`Expr`] tree and delegates to [`compile_dag`].
#[cfg(target_arch = "x86_64")]
pub fn compile_arena_dag(
    arena: &crate::arena::ExprArena,
    root: crate::arena::ExprId,
) -> Result<CompileResult, &'static str> {
    let expr = arena.to_expr(root);
    compile_dag(&expr)
}

/// Compile an [`ExprArena`] DAG with explicit register budget (x86-64 stub).
///
/// Converts the arena to an [`Expr`] tree and delegates to [`compile_dag`].
/// The `ctx` parameter is currently ignored on x86-64 (register budget not
/// yet supported).
#[cfg(target_arch = "x86_64")]
pub fn compile_arena_dag_with_ctx(
    arena: &crate::arena::ExprArena,
    root: crate::arena::ExprId,
    _ctx: EmitCtx,
) -> Result<CompileResult, &'static str> {
    let expr = arena.to_expr(root);
    compile_dag(&expr)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;

    #[test]
    fn test_needs_simple() {
        // X + Y: both leaves need 1, binary needs max(1,1)+1 = 2
        let expr = Expr::Binary(OpKind::Add, Arc::new(Expr::Var(0)), Arc::new(Expr::Var(1)));
        assert_eq!(needs(&expr), 2);
    }

    #[test]
    fn test_needs_unbalanced() {
        // (X + Y) + Z: left needs 2, right needs 1, total = max(2,1) = 2
        let left = Expr::Binary(OpKind::Add, Arc::new(Expr::Var(0)), Arc::new(Expr::Var(1)));
        let expr = Expr::Binary(OpKind::Add, Arc::new(left), Arc::new(Expr::Var(2)));
        assert_eq!(needs(&expr), 2);
    }

    #[test]
    fn test_needs_balanced_deep() {
        // (X + Y) + (Z + W): both sides need 2, total = 2+1 = 3
        let left = Expr::Binary(OpKind::Add, Arc::new(Expr::Var(0)), Arc::new(Expr::Var(1)));
        let right = Expr::Binary(OpKind::Add, Arc::new(Expr::Var(2)), Arc::new(Expr::Var(3)));
        let expr = Expr::Binary(OpKind::Add, Arc::new(left), Arc::new(right));
        assert_eq!(needs(&expr), 3);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_spill_forced() {
        // (X + Y) + (Z + W) with max_regs=2 forces spilling via DAG
        let left = Expr::Binary(OpKind::Add, Arc::new(Expr::Var(0)), Arc::new(Expr::Var(1)));
        let right = Expr::Binary(OpKind::Add, Arc::new(Expr::Var(2)), Arc::new(Expr::Var(3)));
        let expr = Expr::Binary(OpKind::Add, Arc::new(left), Arc::new(right));

        let ctx = EmitCtx::with_max_regs(2);
        let result = compile_dag_with_ctx(&expr, ctx).expect("compile failed");

        assert!(result.spill_count > 0, "expected spills with max_regs=2");

        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(1.0);
            let y = vdupq_n_f32(2.0);
            let z = vdupq_n_f32(3.0);
            let w = vdupq_n_f32(4.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            // (1+2) + (3+4) = 10
            assert_eq!(vgetq_lane_f32(out, 0), 10.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_no_spill_with_enough_regs() {
        let left = Expr::Binary(OpKind::Add, Arc::new(Expr::Var(0)), Arc::new(Expr::Var(1)));
        let right = Expr::Binary(OpKind::Add, Arc::new(Expr::Var(2)), Arc::new(Expr::Var(3)));
        let expr = Expr::Binary(OpKind::Add, Arc::new(left), Arc::new(right));

        let ctx = EmitCtx::default();
        let result = compile_dag_with_ctx(&expr, ctx).expect("compile failed");

        assert_eq!(result.spill_count, 0, "should not spill with 22 registers");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_spill_deeply_nested() {
        // Chain: ((((X + Y) + Z) + W) + X) with max_regs=1
        let e1 = Expr::Binary(OpKind::Add, Arc::new(Expr::Var(0)), Arc::new(Expr::Var(1)));
        let e2 = Expr::Binary(OpKind::Add, Arc::new(e1), Arc::new(Expr::Var(2)));
        let e3 = Expr::Binary(OpKind::Add, Arc::new(e2), Arc::new(Expr::Var(3)));
        let expr = Expr::Binary(OpKind::Add, Arc::new(e3), Arc::new(Expr::Var(0)));

        let ctx = EmitCtx::with_max_regs(1);
        let result = compile_dag_with_ctx(&expr, ctx).expect("compile failed");

        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(1.0);
            let y = vdupq_n_f32(2.0);
            let z = vdupq_n_f32(3.0);
            let w = vdupq_n_f32(4.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            // ((((1+2)+3)+4)+1) = 11
            assert_eq!(vgetq_lane_f32(out, 0), 11.0);
        }
    }

    // =========================================================================
    // Graph Coloring (DAG) Tests
    // =========================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dag_simple() {
        // Simple expression: X + Y (no sharing)
        let expr = Expr::Binary(OpKind::Add, Arc::new(Expr::Var(0)), Arc::new(Expr::Var(1)));

        let result = compile_dag(&expr).expect("DAG compile failed");
        assert_eq!(result.spill_count, 0);

        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(3.0);
            let y = vdupq_n_f32(4.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            assert_eq!(vgetq_lane_f32(out, 0), 7.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dag_with_constant() {
        // X * 2.0 + Y
        let expr = Expr::Binary(
            OpKind::Add,
            Arc::new(Expr::Binary(
                OpKind::Mul,
                Arc::new(Expr::Var(0)),
                Arc::new(Expr::Const(2.0)),
            )),
            Arc::new(Expr::Var(1)),
        );

        let result = compile_dag(&expr).expect("DAG compile failed");

        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(3.0);
            let y = vdupq_n_f32(4.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            // 3*2 + 4 = 10
            assert_eq!(vgetq_lane_f32(out, 0), 10.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dag_with_spill() {
        // Complex expression with limited registers
        let left = Expr::Binary(OpKind::Add, Arc::new(Expr::Var(0)), Arc::new(Expr::Var(1)));
        let right = Expr::Binary(OpKind::Add, Arc::new(Expr::Var(2)), Arc::new(Expr::Var(3)));
        let expr = Expr::Binary(OpKind::Add, Arc::new(left), Arc::new(right));

        // Compile with only 2 registers - should require spilling
        let ctx = EmitCtx::with_max_regs(2);
        let result = compile_dag_with_ctx(&expr, ctx).expect("DAG compile failed");

        // Graph coloring may or may not spill depending on the graph structure
        // The important thing is correctness
        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(1.0);
            let y = vdupq_n_f32(2.0);
            let z = vdupq_n_f32(3.0);
            let w = vdupq_n_f32(4.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            // (1+2) + (3+4) = 10
            assert_eq!(vgetq_lane_f32(out, 0), 10.0);
        }
    }

    #[test]
    fn test_linearize_dag() {
        // Test the linearization function
        let expr = Expr::Binary(OpKind::Add, Arc::new(Expr::Var(0)), Arc::new(Expr::Var(1)));

        let (schedule, _structural_cache, uses_map) = linearize_dag(&expr);

        // Should have 3 values: X, Y, X+Y
        assert_eq!(schedule.len(), 3);

        // Root (X+Y) should use both X and Y
        let root_id = schedule.last().unwrap().0;
        let root_uses = uses_map
            .get(root_id.0 as usize)
            .expect("root should have uses");
        assert_eq!(root_uses.len(), 2);
    }

    #[test]
    fn test_linearize_dag_structural_dedup() {
        // Verify that cloned subtrees are collapsed by structural hash-consing.
        // Build: (X + Y) + (X + Y) where the two (X+Y) are distinct allocations.
        let make_sum = || Expr::Binary(OpKind::Add, Arc::new(Expr::Var(0)), Arc::new(Expr::Var(1)));
        let expr = Expr::Binary(OpKind::Add, Arc::new(make_sum()), Arc::new(make_sum()));

        let (schedule, _structural_cache, _uses_map) = linearize_dag(&expr);

        // Without dedup: X, Y, X+Y, X', Y', X'+Y', (X+Y)+(X'+Y') = 7 nodes
        // With dedup: X, Y, X+Y, (X+Y)+(X+Y) = 4 nodes
        assert_eq!(
            schedule.len(),
            4,
            "structural dedup should collapse cloned (X+Y)"
        );
    }

    // =========================================================================
    // FrameLayout unit tests
    // =========================================================================

    #[test]
    fn test_frame_layout_empty() {
        let layout = FrameLayout::from_allocation(&[]).unwrap();
        assert_eq!(layout.frame_size, 0);
        assert!(layout.spill_slots.is_empty());
    }

    #[test]
    fn test_frame_layout_one_spill() {
        let layout = FrameLayout::from_allocation(&[regalloc::ValueId(5)]).unwrap();
        assert_eq!(layout.frame_size, 16);
        assert_eq!(layout.spill_slots[&regalloc::ValueId(5)], 0);
    }

    #[test]
    fn test_frame_layout_alignment() {
        let spilled = [
            regalloc::ValueId(1),
            regalloc::ValueId(2),
            regalloc::ValueId(3),
        ];
        let layout = FrameLayout::from_allocation(&spilled).unwrap();
        // 3 * 16 = 48, already aligned
        assert_eq!(layout.frame_size, 48);
        assert_eq!(layout.spill_slots[&regalloc::ValueId(1)], 0);
        assert_eq!(layout.spill_slots[&regalloc::ValueId(2)], 16);
        assert_eq!(layout.spill_slots[&regalloc::ValueId(3)], 32);
    }

    // =========================================================================
    // resolve_operands unit tests — the spill logic that was buggy
    // =========================================================================

    /// Helper: build minimal assignment + spill maps for resolve_operands tests.
    fn make_maps(
        assigned: &[(u32, u8)],
        spilled: &[(u32, u32)],
    ) -> (
        alloc::collections::BTreeMap<regalloc::ValueId, Reg>,
        alloc::collections::BTreeMap<regalloc::ValueId, u32>,
        alloc::collections::BTreeMap<regalloc::ValueId, u32>,
    ) {
        use alloc::collections::BTreeMap;
        let mut assignment = BTreeMap::new();
        for &(v, r) in assigned {
            assignment.insert(regalloc::ValueId(v), Reg(r));
        }
        let mut slots = BTreeMap::new();
        for &(v, off) in spilled {
            slots.insert(regalloc::ValueId(v), off);
        }
        (assignment, slots, BTreeMap::new())
    }

    #[test]
    fn test_resolve_binary_no_spills() {
        // left=v4, right=v5, dst=v6 — all in registers
        let (assign, spills, remat) = make_maps(&[(0, 4), (1, 5), (2, 6)], &[]);
        let op = ScheduledOp::Binary(OpKind::Add, regalloc::ValueId(0), regalloc::ValueId(1));
        let plan = resolve_operands(&op, Loc::Reg(Reg(6)), &assign, &spills, &remat).unwrap();

        assert!(plan.reloads.is_empty());
        assert!(plan.store.is_none());
        assert_eq!(
            plan.op,
            ResolvedOp::Binary {
                op: OpKind::Add,
                dst: Reg(6),
                left: Reg(4),
                right: Reg(5)
            }
        );
    }

    #[test]
    fn test_resolve_binary_left_spilled() {
        // left spilled at offset 0, right in v5
        let (assign, spills, remat) = make_maps(&[(1, 5), (2, 6)], &[(0, 0)]);
        let op = ScheduledOp::Binary(OpKind::Add, regalloc::ValueId(0), regalloc::ValueId(1));
        let plan = resolve_operands(&op, Loc::Reg(Reg(6)), &assign, &spills, &remat).unwrap();

        assert_eq!(plan.reloads.len(), 1);
        assert_eq!(
            plan.reloads[0],
            Reload::FromStack {
                target: RELOAD_REGS[1],
                offset: 0
            }
        );
        assert_eq!(
            plan.op,
            ResolvedOp::Binary {
                op: OpKind::Add,
                dst: Reg(6),
                left: RELOAD_REGS[1],
                right: Reg(5)
            }
        );
    }

    #[test]
    fn test_resolve_binary_both_spilled() {
        // Both spilled: left → dst (temp trick), right → tmp_op
        let (assign, spills, remat) = make_maps(&[(2, 6)], &[(0, 0), (1, 16)]);
        let op = ScheduledOp::Binary(OpKind::Mul, regalloc::ValueId(0), regalloc::ValueId(1));
        let plan = resolve_operands(&op, Loc::Reg(Reg(6)), &assign, &spills, &remat).unwrap();

        assert_eq!(plan.reloads.len(), 2);
        // left → dst (v6), right → tmp_op (v27)
        assert_eq!(
            plan.reloads[0],
            Reload::FromStack {
                target: Reg(6),
                offset: 0
            }
        );
        assert_eq!(
            plan.reloads[1],
            Reload::FromStack {
                target: RELOAD_REGS[1],
                offset: 16
            }
        );
        assert_eq!(
            plan.op,
            ResolvedOp::Binary {
                op: OpKind::Mul,
                dst: Reg(6),
                left: Reg(6),
                right: RELOAD_REGS[1]
            }
        );
    }

    #[test]
    fn test_resolve_dst_spilled_generates_store() {
        // dst is spilled → compute into RELOAD_REGS[0], then store
        let (assign, spills, remat) = make_maps(&[(0, 4), (1, 5)], &[(2, 32)]);
        let op = ScheduledOp::Binary(OpKind::Add, regalloc::ValueId(0), regalloc::ValueId(1));
        let plan = resolve_operands(&op, Loc::Spill(32), &assign, &spills, &remat).unwrap();

        // dst should be RELOAD_REGS[0] since result is spilled
        assert_eq!(
            plan.op,
            ResolvedOp::Binary {
                op: OpKind::Add,
                dst: RELOAD_REGS[0],
                left: Reg(4),
                right: Reg(5)
            }
        );
        assert_eq!(
            plan.store,
            Some(Store {
                src: RELOAD_REGS[0],
                offset: 32
            })
        );
    }

    #[test]
    fn test_resolve_muladd_fmla_path() {
        // a in reg, b in reg, c in reg → FMLA with setup_mov for c→dst
        let (assign, spills, remat) = make_maps(&[(0, 4), (1, 5), (2, 7), (3, 8)], &[]);
        let op = ScheduledOp::Ternary(
            OpKind::MulAdd,
            regalloc::ValueId(0),
            regalloc::ValueId(1),
            regalloc::ValueId(2),
        );
        let plan = resolve_operands(&op, Loc::Reg(Reg(8)), &assign, &spills, &remat).unwrap();

        assert!(plan.reloads.is_empty());
        // c=v7 ≠ dst=v8, so setup_mov should copy c → dst
        assert_eq!(plan.setup_mov, Some((Reg(8), Reg(7))));
        assert_eq!(
            plan.op,
            ResolvedOp::FusedMulAdd {
                dst: Reg(8),
                a: Reg(4),
                b: Reg(5)
            }
        );
    }

    #[test]
    fn test_resolve_muladd_decomposed_both_ab_spilled() {
        // a and b both spilled → decomposed FMUL+FADD path
        // c in register
        let (assign, spills, remat) = make_maps(&[(2, 7), (3, 8)], &[(0, 0), (1, 16)]);
        let op = ScheduledOp::Ternary(
            OpKind::MulAdd,
            regalloc::ValueId(0),
            regalloc::ValueId(1),
            regalloc::ValueId(2),
        );
        let plan = resolve_operands(&op, Loc::Reg(Reg(8)), &assign, &spills, &remat).unwrap();

        // a → dst, b → tmp_op loaded upfront
        assert_eq!(plan.reloads.len(), 2);
        assert_eq!(
            plan.reloads[0],
            Reload::FromStack {
                target: Reg(8),
                offset: 0
            }
        );
        assert_eq!(
            plan.reloads[1],
            Reload::FromStack {
                target: RELOAD_REGS[1],
                offset: 16
            }
        );
        // c is in a register, no deferred reload needed
        match &plan.op {
            ResolvedOp::DecomposedMulAdd {
                dst,
                a,
                b,
                c,
                c_deferred,
            } => {
                assert_eq!(*dst, Reg(8));
                assert_eq!(*a, Reg(8));
                assert_eq!(*b, RELOAD_REGS[1]);
                assert_eq!(*c, Reg(7));
                assert_eq!(*c_deferred, None);
            }
            other => panic!("expected DecomposedMulAdd, got {:?}", other),
        }
    }

    #[test]
    fn test_resolve_muladd_decomposed_all_three_spilled() {
        // a, b, c all spilled → decomposed with deferred c reload
        let (assign, spills, remat) = make_maps(&[(3, 8)], &[(0, 0), (1, 16), (2, 32)]);
        let op = ScheduledOp::Ternary(
            OpKind::MulAdd,
            regalloc::ValueId(0),
            regalloc::ValueId(1),
            regalloc::ValueId(2),
        );
        let plan = resolve_operands(&op, Loc::Reg(Reg(8)), &assign, &spills, &remat).unwrap();

        // Only a and b reloads upfront — c is deferred
        assert_eq!(plan.reloads.len(), 2);
        match &plan.op {
            ResolvedOp::DecomposedMulAdd { c, c_deferred, .. } => {
                assert_eq!(*c, RELOAD_REGS[1]); // will be reloaded into tmp_op
                assert_eq!(*c_deferred, Some(DeferredReload::FromStack(32)));
            }
            other => panic!("expected DecomposedMulAdd, got {:?}", other),
        }
    }

    #[test]
    fn test_resolve_var_is_nop() {
        let (assign, spills, remat) = make_maps(&[(0, 0)], &[]);
        let op = ScheduledOp::Var(0);
        let plan = resolve_operands(&op, Loc::Reg(Reg(0)), &assign, &spills, &remat).unwrap();
        assert_eq!(plan.op, ResolvedOp::Nop);
        assert!(plan.reloads.is_empty());
        assert!(plan.store.is_none());
    }

    #[test]
    fn test_resolve_const() {
        let (assign, spills, remat) = make_maps(&[(0, 6)], &[]);
        let op = ScheduledOp::Const(3.14);
        let plan = resolve_operands(&op, Loc::Reg(Reg(6)), &assign, &spills, &remat).unwrap();
        assert_eq!(
            plan.op,
            ResolvedOp::LoadConst {
                dst: Reg(6),
                val_bits: 3.14f32.to_bits()
            }
        );
    }

    // =========================================================================
    // DAG integration tests — expressions that previously crashed (SIGSEGV)
    // =========================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dag_lowered_sin() {
        // sin(X) through DAG path — triggers spilling from lowered expansion
        let expr = Expr::Unary(OpKind::Sin, Arc::new(Expr::Var(0)));
        let result = compile_dag(&expr).expect("DAG compile of sin(X) failed");

        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(0.0);
            let y = vdupq_n_f32(0.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            let val = vgetq_lane_f32(out, 0);
            assert!(val.abs() < 0.01, "sin(0) = {}, expected ~0", val);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dag_deep_chain_with_spills() {
        // Deep chain of ops with limited registers — stresses the spill/reload
        // logic that previously caused SIGSEGV. Uses only primitive ops (no pow)
        // to avoid exponential blowup from lowering.
        //
        // expr = sin(X) + cos(Y) * floor(Z + W)
        // After lowering, sin and cos expand to Horner polynomials with ~20 nodes each.
        // With max_regs=4, nearly everything must spill.
        let sin_x = Expr::Unary(OpKind::Sin, Arc::new(Expr::Var(0)));
        let cos_y = Expr::Unary(OpKind::Cos, Arc::new(Expr::Var(1)));
        let floor_zw = Expr::Unary(
            OpKind::Floor,
            Arc::new(Expr::Binary(
                OpKind::Add,
                Arc::new(Expr::Var(2)),
                Arc::new(Expr::Var(3)),
            )),
        );
        let cos_times_floor = Expr::Binary(OpKind::Mul, Arc::new(cos_y), Arc::new(floor_zw));
        let expr = Expr::Binary(OpKind::Add, Arc::new(sin_x), Arc::new(cos_times_floor));

        let ctx = EmitCtx::with_max_regs(4);
        let result =
            compile_dag_with_ctx(&expr, ctx).expect("DAG compile with heavy spills failed");
        // Linear scan may or may not spill — correctness is what matters.

        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(0.0); // sin(0) = 0
            let y = vdupq_n_f32(0.0); // cos(0) = 1
            let z = vdupq_n_f32(2.3);
            let w = vdupq_n_f32(0.5); // floor(2.3+0.5) = floor(2.8) = 2

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            let val = vgetq_lane_f32(out, 0);
            // sin(0) + cos(0) * floor(2.8) = 0 + 1*2 = 2.0
            // (polynomial approximations may be slightly off)
            assert!((val - 2.0).abs() < 0.05, "expected ~2.0, got {}", val);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dag_muladd_spilled() {
        // MulAdd with max_regs=3 — forces spilling of operands
        let expr = Expr::Ternary(
            OpKind::MulAdd,
            Arc::new(Expr::Binary(
                OpKind::Add,
                Arc::new(Expr::Var(0)),
                Arc::new(Expr::Var(1)),
            )),
            Arc::new(Expr::Binary(
                OpKind::Add,
                Arc::new(Expr::Var(2)),
                Arc::new(Expr::Var(3)),
            )),
            Arc::new(Expr::Binary(
                OpKind::Mul,
                Arc::new(Expr::Var(0)),
                Arc::new(Expr::Const(2.0)),
            )),
        );

        let ctx = EmitCtx::with_max_regs(3);
        let result = compile_dag_with_ctx(&expr, ctx).expect("DAG muladd with spills failed");

        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(1.0);
            let y = vdupq_n_f32(2.0);
            let z = vdupq_n_f32(3.0);
            let w = vdupq_n_f32(4.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            // (1+2)*(3+4) + (1*2) = 3*7 + 2 = 23
            let val = vgetq_lane_f32(out, 0);
            assert_eq!(val, 23.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dag_clamp_basic() {
        // clamp(X, 0.0, 1.0) through DAG
        let expr = Expr::Ternary(
            OpKind::Clamp,
            Arc::new(Expr::Var(0)),
            Arc::new(Expr::Const(0.0)),
            Arc::new(Expr::Const(1.0)),
        );

        let result = compile_dag(&expr).expect("DAG clamp failed");

        unsafe {
            use core::arch::aarch64::*;

            let func: executable::KernelFn = result.code.as_fn();
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            // Below range
            let out = func(vdupq_n_f32(-5.0), z, z, w);
            assert_eq!(vgetq_lane_f32(out, 0), 0.0);

            // In range
            let out = func(vdupq_n_f32(0.5), z, z, w);
            assert_eq!(vgetq_lane_f32(out, 0), 0.5);

            // Above range
            let out = func(vdupq_n_f32(5.0), z, z, w);
            assert_eq!(vgetq_lane_f32(out, 0), 1.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dag_compile_via_compile() {
        // Verify compile() delegates to compile_dag() and produces correct results.
        let expr = Expr::Binary(
            OpKind::Mul,
            Arc::new(Expr::Binary(
                OpKind::Add,
                Arc::new(Expr::Var(0)),
                Arc::new(Expr::Const(1.5)),
            )),
            Arc::new(Expr::Binary(
                OpKind::Sub,
                Arc::new(Expr::Var(1)),
                Arc::new(Expr::Var(2)),
            )),
        );

        let code = compile(&expr).expect("compile failed");

        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(2.5);
            let y = vdupq_n_f32(7.0);
            let z = vdupq_n_f32(3.0);
            let w = vdupq_n_f32(0.0);

            let func: executable::KernelFn = code.as_fn();
            let out = vgetq_lane_f32(func(x, y, z, w), 0);
            // (2.5 + 1.5) * (7.0 - 3.0) = 4.0 * 4.0 = 16.0
            assert_eq!(out, 16.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dag_pow_compiles_without_overflow() {
        // pow(X, Y) = exp2(Y * log2(X)) — previously caused spill overflow
        // due to lowering blowup. With structural hash-consing, internal clones
        // collapse and the schedule stays manageable.
        //
        // Note: we don't test numerical accuracy here — the polynomial
        // approximations for exp2/log2 have limited range. The point is that
        // compile_dag doesn't panic or overflow.
        let expr = Expr::Binary(OpKind::Pow, Arc::new(Expr::Var(0)), Arc::new(Expr::Var(1)));

        let result = compile_dag(&expr).expect("DAG compile of pow(X, Y) failed");

        // Verify it compiled and produced executable code — no spill overflow.
        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(1.5);
            let y = vdupq_n_f32(1.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            let val = vgetq_lane_f32(out, 0);
            // pow(1.5, 1.0) should be ~1.5 — identity exponent
            assert!(
                val.is_finite(),
                "pow(1.5, 1.0) produced non-finite: {}",
                val
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dag_double_exp() {
        // exp(exp(W)) + Z — expression #7117 that caused bench_jit_corpus to hang.
        //
        // Double exponential: lower_exp chains two exp2 polynomial expansions.
        // Each exp2 clones its input ~5 times for the Horner chain,
        // so the lowered tree is ~1300 nodes before structural dedup.
        //
        // This test verifies:
        // 1. compile_dag doesn't hang or overflow
        // 2. The emitted code runs without crashing
        // 3. The result is finite
        let expr = Expr::Binary(
            OpKind::Add,
            Arc::new(Expr::Unary(
                OpKind::Exp,
                Arc::new(Expr::Unary(OpKind::Exp, Arc::new(Expr::Var(3)))),
            )),
            Arc::new(Expr::Var(2)),
        );

        let result = compile_dag(&expr).expect("DAG compile of exp(exp(W))+Z failed");

        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(0.5);
            let y = vdupq_n_f32(0.7);
            let z = vdupq_n_f32(1.0);
            let w = vdupq_n_f32(0.0); // exp(exp(0)) = exp(1) ≈ 2.718

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            let val = vgetq_lane_f32(out, 0);
            // exp(exp(0)) + 1.0 ≈ 2.718 + 1.0 = 3.718
            // Polynomial approximation may be off, but result must be finite.
            assert!(
                val.is_finite(),
                "exp(exp(0))+1 produced non-finite: {}",
                val
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dag_tan_specific_repro() {
        // This value came directly from a rewrite-bug localization pass.
        // The tolerance here is intentionally tight enough to catch the
        // current large mismatch, not just generic approximation noise.
        let expr = Expr::Unary(OpKind::Tan, Arc::new(Expr::Var(0)));

        let result = compile_dag(&expr).expect("DAG compile of tan(X) failed");

        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(-1.6860065);
            let y = vdupq_n_f32(0.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            let val = vgetq_lane_f32(out, 0);
            let expected = (-1.6860065_f32).tan();
            assert!(val.is_finite(), "tan(-1.6860065) produced non-finite: {}", val);
            assert!(
                (val - expected).abs() < 0.25,
                "tan(-1.6860065) = {}, expected ~{}",
                val,
                expected
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dag_sin_cos_specific_tan_repro_input() {
        // Same input as the tan repro, but testing sin/cos individually tells
        // us whether tan is broken directly or just amplifying smaller errors.
        let sin_expr = Expr::Unary(OpKind::Sin, Arc::new(Expr::Var(0)));
        let cos_expr = Expr::Unary(OpKind::Cos, Arc::new(Expr::Var(0)));

        let sin_result = compile_dag(&sin_expr).expect("DAG compile of sin(X) failed");
        let cos_result = compile_dag(&cos_expr).expect("DAG compile of cos(X) failed");

        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(-1.6860065);
            let y = vdupq_n_f32(0.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let sin_func: executable::KernelFn = sin_result.code.as_fn();
            let cos_func: executable::KernelFn = cos_result.code.as_fn();

            let sin_val = vgetq_lane_f32(sin_func(x, y, z, w), 0);
            let cos_val = vgetq_lane_f32(cos_func(x, y, z, w), 0);
            let sin_expected = (-1.6860065_f32).sin();
            let cos_expected = (-1.6860065_f32).cos();

            assert!(
                (sin_val - sin_expected).abs() < 0.02,
                "sin(-1.6860065) = {}, expected ~{}",
                sin_val,
                sin_expected
            );
            assert!(
                (cos_val - cos_expected).abs() < 0.02,
                "cos(-1.6860065) = {}, expected ~{}",
                cos_val,
                cos_expected
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dag_atan2_compiles() {
        // atan2(X, Y) should compile and produce finite result.
        // Verifies the new atan2 JIT builtin works through the DAG pipeline.
        let expr = Expr::Binary(
            OpKind::Atan2,
            Arc::new(Expr::Var(0)),
            Arc::new(Expr::Var(1)),
        );

        let result = compile_dag(&expr).expect("DAG compile of atan2(X, Y) failed");

        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(1.0);
            let y = vdupq_n_f32(1.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            let val = vgetq_lane_f32(out, 0);
            // atan2(1.0, 1.0) = π/4 ≈ 0.785
            // The 4-coefficient polynomial has ~0.06 error at the t=1 boundary.
            assert!(val.is_finite(), "atan2(1, 1) produced non-finite: {}", val);
            assert!(
                (val - core::f32::consts::FRAC_PI_4).abs() < 0.07,
                "atan2(1, 1) = {}, expected ~{}",
                val,
                core::f32::consts::FRAC_PI_4
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dag_asin_compiles() {
        // asin(X) should compile through the unary transcendental path.
        let expr = Expr::Unary(OpKind::Asin, Arc::new(Expr::Var(0)));

        let result = compile_dag(&expr).expect("DAG compile of asin(X) failed");

        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(0.5);
            let y = vdupq_n_f32(0.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            let val = vgetq_lane_f32(out, 0);
            // asin(0.5) = π/6 ≈ 0.5236
            let expected = 0.5_f32.asin();
            assert!(val.is_finite(), "asin(0.5) produced non-finite: {}", val);
            assert!(
                (val - expected).abs() < 0.02,
                "asin(0.5) = {}, expected ~{}",
                val,
                expected
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dag_acos_compiles() {
        // acos(X) should compile through the unary transcendental path.
        let expr = Expr::Unary(OpKind::Acos, Arc::new(Expr::Var(0)));

        let result = compile_dag(&expr).expect("DAG compile of acos(X) failed");

        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(0.5);
            let y = vdupq_n_f32(0.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            let val = vgetq_lane_f32(out, 0);
            // acos(0.5) = π/3 ≈ 1.0472
            let expected = 0.5_f32.acos();
            assert!(val.is_finite(), "acos(0.5) produced non-finite: {}", val);
            assert!(
                (val - expected).abs() < 0.02,
                "acos(0.5) = {}, expected ~{}",
                val,
                expected
            );
        }
    }

    /// Test that Select short-circuits: when mask is all-true, the false arm
    /// (which contains a division by zero) must NOT produce NaN in the output.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_select_short_circuit_avoids_div_by_zero() {
        // Build: Select(X > 0, Y, Z / 0.0)
        // When X is all-positive (mask all-true), the false arm Z/0.0 should
        // be skipped and the result should be Y, not NaN.
        let mask = Expr::Binary(
            OpKind::Gt,
            Arc::new(Expr::Var(0)), // X
            Arc::new(Expr::Const(0.0)),
        );
        let true_arm = Expr::Var(1); // Y
        let false_arm = Expr::Binary(
            OpKind::Div,
            Arc::new(Expr::Var(2)),     // Z
            Arc::new(Expr::Const(0.0)), // divide by zero!
        );
        let expr = Expr::Ternary(
            OpKind::Select,
            Arc::new(mask),
            Arc::new(true_arm),
            Arc::new(false_arm),
        );

        let result = compile_dag(&expr).expect("DAG compile of Select failed");

        unsafe {
            use core::arch::aarch64::*;
            // X = all 1.0 (positive → mask all-true)
            let x = vdupq_n_f32(1.0);
            let y = vdupq_n_f32(42.0);
            let z = vdupq_n_f32(7.0);
            let w = vdupq_n_f32(0.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            let val = vgetq_lane_f32(out, 0);
            assert!(
                val.is_finite(),
                "Select short-circuit failed: got {} (expected 42.0, false arm div-by-zero leaked through)",
                val
            );
            assert!(
                (val - 42.0).abs() < 1e-6,
                "Select short-circuit: got {}, expected 42.0",
                val
            );
        }
    }

    /// Test Select with all-false mask: should return false arm.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_select_short_circuit_all_false() {
        // Select(X > 0, Y / 0.0, Z) with X all-negative
        let mask = Expr::Binary(
            OpKind::Gt,
            Arc::new(Expr::Var(0)),
            Arc::new(Expr::Const(0.0)),
        );
        let true_arm = Expr::Binary(
            OpKind::Div,
            Arc::new(Expr::Var(1)),
            Arc::new(Expr::Const(0.0)), // div by zero in true arm
        );
        let false_arm = Expr::Var(2); // Z

        let expr = Expr::Ternary(
            OpKind::Select,
            Arc::new(mask),
            Arc::new(true_arm),
            Arc::new(false_arm),
        );

        let result = compile_dag(&expr).expect("DAG compile of Select failed");

        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(-1.0); // negative → mask all-false
            let y = vdupq_n_f32(7.0);
            let z = vdupq_n_f32(99.0);
            let w = vdupq_n_f32(0.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            let val = vgetq_lane_f32(out, 0);
            assert!(
                val.is_finite(),
                "Select short-circuit (all-false) failed: got {} (expected 99.0)",
                val
            );
            assert!(
                (val - 99.0).abs() < 1e-6,
                "Select short-circuit (all-false): got {}, expected 99.0",
                val
            );
        }
    }

    /// Test Select with mixed mask: BSL path, both arms evaluated.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_select_mixed_mask_uses_bsl() {
        // Select(X > 0, Y, Z) with X = [1, -1, 1, -1]
        let mask = Expr::Binary(
            OpKind::Gt,
            Arc::new(Expr::Var(0)),
            Arc::new(Expr::Const(0.0)),
        );
        let true_arm = Expr::Var(1);
        let false_arm = Expr::Var(2);

        let expr = Expr::Ternary(
            OpKind::Select,
            Arc::new(mask),
            Arc::new(true_arm),
            Arc::new(false_arm),
        );

        let result = compile_dag(&expr).expect("DAG compile of Select failed");

        unsafe {
            use core::arch::aarch64::*;
            // Mixed mask: lanes 0,2 true, lanes 1,3 false
            let x_vals: [f32; 4] = [1.0, -1.0, 1.0, -1.0];
            let x = vld1q_f32(x_vals.as_ptr());
            let y = vdupq_n_f32(10.0);
            let z = vdupq_n_f32(20.0);
            let w = vdupq_n_f32(0.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);

            // Lane 0: mask true → Y = 10
            assert!(
                (vgetq_lane_f32(out, 0) - 10.0).abs() < 1e-6,
                "lane 0: expected 10.0, got {}",
                vgetq_lane_f32(out, 0)
            );
            // Lane 1: mask false → Z = 20
            assert!(
                (vgetq_lane_f32(out, 1) - 20.0).abs() < 1e-6,
                "lane 1: expected 20.0, got {}",
                vgetq_lane_f32(out, 1)
            );
            // Lane 2: mask true → Y = 10
            assert!(
                (vgetq_lane_f32(out, 2) - 10.0).abs() < 1e-6,
                "lane 2: expected 10.0, got {}",
                vgetq_lane_f32(out, 2)
            );
            // Lane 3: mask false → Z = 20
            assert!(
                (vgetq_lane_f32(out, 3) - 20.0).abs() < 1e-6,
                "lane 3: expected 20.0, got {}",
                vgetq_lane_f32(out, 3)
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dag_ne_correctness() {
        // Ne(X, Y) should produce all-ones (as float: NaN / -NaN) when X != Y,
        // and all-zeros (0.0) when X == Y.
        let expr = Expr::Binary(OpKind::Ne, Arc::new(Expr::Var(0)), Arc::new(Expr::Var(1)));

        let result = compile_dag(&expr).expect("DAG compile of Ne failed");

        unsafe {
            use core::arch::aarch64::*;

            // Test 1: X != Y → should produce non-zero (all-ones mask)
            let x = vdupq_n_f32(3.0);
            let y = vdupq_n_f32(4.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            // All-ones mask reinterpreted as f32 is NaN; check bits are 0xFFFFFFFF
            let bits: u32 = vgetq_lane_f32(out, 0).to_bits();
            assert_eq!(
                bits, 0xFFFF_FFFF,
                "Ne(3.0, 4.0) should be all-ones mask, got 0x{:08X}",
                bits
            );

            // Test 2: X == Y → should produce all-zeros (0.0)
            let x_eq = vdupq_n_f32(5.0);
            let y_eq = vdupq_n_f32(5.0);
            let out_eq = func(x_eq, y_eq, z, w);
            let bits_eq: u32 = vgetq_lane_f32(out_eq, 0).to_bits();
            assert_eq!(
                bits_eq, 0x0000_0000,
                "Ne(5.0, 5.0) should be all-zeros, got 0x{:08X}",
                bits_eq
            );
        }
    }

    // =========================================================================
    // Arena compilation tests
    // =========================================================================

    #[test]
    fn test_arena_to_schedule_simple() {
        use crate::arena::ExprArena;

        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let sum = arena.push_binary(OpKind::Add, x, y);

        let schedule = arena_to_schedule(&arena, sum);

        // Should have 3 values: X, Y, X+Y
        assert_eq!(
            schedule.len(),
            3,
            "expected 3 schedule entries, got {}",
            schedule.len()
        );

        // Verify the operations
        assert!(matches!(schedule[0].1, ScheduledOp::Var(0)));
        assert!(matches!(schedule[1].1, ScheduledOp::Var(1)));
        assert!(matches!(
            schedule[2].1,
            ScheduledOp::Binary(OpKind::Add, _, _)
        ));
    }

    #[test]
    fn test_arena_to_schedule_filters_unreachable() {
        use crate::arena::ExprArena;

        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let _garbage = arena.push_const(999.0); // unreachable
        let y = arena.push_var(1);
        let sum = arena.push_binary(OpKind::Add, x, y);

        let schedule = arena_to_schedule(&arena, sum);

        // Should have 3 values (garbage node filtered out)
        assert_eq!(
            schedule.len(),
            3,
            "unreachable garbage node should be filtered"
        );
    }

    #[test]
    fn test_arena_to_uses() {
        use crate::arena::ExprArena;

        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let sum = arena.push_binary(OpKind::Add, x, y);

        let schedule = arena_to_schedule(&arena, sum);
        let uses = arena_to_uses(&schedule);

        assert_eq!(uses.len(), 3);
        assert!(uses[0].is_empty(), "Var should have no uses");
        assert!(uses[1].is_empty(), "Var should have no uses");
        assert_eq!(uses[2].len(), 2, "Binary should use 2 children");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_arena_compile_simple() {
        use crate::arena::ExprArena;

        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let sum = arena.push_binary(OpKind::Add, x, y);

        let result = compile_arena_dag(&arena, sum).expect("arena DAG compile failed");
        assert_eq!(result.spill_count, 0);

        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(3.0);
            let y = vdupq_n_f32(4.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            assert_eq!(vgetq_lane_f32(out, 0), 7.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_arena_compile_with_constant() {
        use crate::arena::ExprArena;

        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let two = arena.push_const(2.0);
        let y = arena.push_var(1);
        let prod = arena.push_binary(OpKind::Mul, x, two);
        let sum = arena.push_binary(OpKind::Add, prod, y);

        let result = compile_arena_dag(&arena, sum).expect("arena DAG compile failed");

        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(3.0);
            let y = vdupq_n_f32(4.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            // 3*2 + 4 = 10
            assert_eq!(vgetq_lane_f32(out, 0), 10.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_arena_compile_with_spills() {
        use crate::arena::ExprArena;

        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let z = arena.push_var(2);
        let w = arena.push_var(3);
        let left = arena.push_binary(OpKind::Add, x, y);
        let right = arena.push_binary(OpKind::Add, z, w);
        let root = arena.push_binary(OpKind::Add, left, right);

        let ctx = EmitCtx::with_max_regs(2);
        let result = compile_arena_dag_with_ctx(&arena, root, ctx)
            .expect("arena DAG compile with spills failed");

        assert!(result.spill_count > 0, "expected spills with max_regs=2");

        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(1.0);
            let y = vdupq_n_f32(2.0);
            let z = vdupq_n_f32(3.0);
            let w = vdupq_n_f32(4.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            // (1+2) + (3+4) = 10
            assert_eq!(vgetq_lane_f32(out, 0), 10.0);
        }
    }

    /// Verify arena compilation matches direct `Expr` compilation for the same expression.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_arena_matches_expr_compile() {
        use crate::arena::ExprArena;

        // Build expression both ways: (X * 2.0 + Y) * (Z - W)
        let expr = Expr::Binary(
            OpKind::Mul,
            Arc::new(Expr::Binary(
                OpKind::Add,
                Arc::new(Expr::Binary(
                    OpKind::Mul,
                    Arc::new(Expr::Var(0)),
                    Arc::new(Expr::Const(2.0)),
                )),
                Arc::new(Expr::Var(1)),
            )),
            Arc::new(Expr::Binary(
                OpKind::Sub,
                Arc::new(Expr::Var(2)),
                Arc::new(Expr::Var(3)),
            )),
        );

        let (arena, root) = ExprArena::from_expr(&expr);

        let expr_result = compile_dag(&expr).expect("Expr compile failed");
        let arena_result = compile_arena_dag(&arena, root).expect("arena compile failed");

        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(2.5);
            let y = vdupq_n_f32(1.0);
            let z = vdupq_n_f32(7.0);
            let w = vdupq_n_f32(3.0);

            let expr_fn: executable::KernelFn = expr_result.code.as_fn();
            let arena_fn: executable::KernelFn = arena_result.code.as_fn();

            let expr_out = vgetq_lane_f32(expr_fn(x, y, z, w), 0);
            let arena_out = vgetq_lane_f32(arena_fn(x, y, z, w), 0);

            // (2.5*2.0 + 1.0) * (7.0 - 3.0) = 6.0 * 4.0 = 24.0
            assert_eq!(expr_out, 24.0, "Expr result wrong");
            assert_eq!(arena_out, 24.0, "arena result wrong");
        }
    }

    // =====================================================================
    // Scanline kernel tests
    // =====================================================================

    /// Verify the register-offset LDR/STR encodings work at all.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_scanline_handcoded_add() {
        use core::arch::aarch64::*;

        // Hand-coded scanline kernel: output[i] = X[i] + Y
        // ABI: x0=input_ptr, v0=Y, x1=output_ptr, x2=count
        //
        // Approach: dead-simple loop using x0/x1/x2 directly, no callee-save needed.
        //   CBZ x2, done
        // loop:
        //   LDR Q1, [x0], #16        // load X[i], post-increment
        //   FADD V1.4S, V1.4S, V0.4S // X[i] + Y
        //   STR Q1, [x1], #16        // store result, post-increment
        //   SUBS x2, x2, #1
        //   B.NE loop
        // done:
        //   RET
        let mut code = Vec::new();

        // CBZ x2, done (patch later)
        let cbz_pos = code.len();
        aarch64::emit32(&mut code, 0xB4000002); // CBZ X2, #0 (placeholder)

        // loop:
        let loop_pos = code.len();
        // LDR Q1, [X0], #16 (post-index immediate)
        // LDRQ post-index: 00_111_100_01_0_000010000_01_Rn_Rt
        // imm9 = 16 = 0x010
        // 00_111_1_00_01_0_000010000_01_00000_00001
        aarch64::emit32(&mut code, 0x3CC10401); // LDR Q1, [X0], #16

        // FADD V1.4S, V1.4S, V0.4S
        aarch64::emit32(&mut code, 0x4E20D421);

        // STR Q1, [X1], #16 (post-index immediate)
        // 00_111_1_00_00_0_000010000_01_Rn_Rt
        aarch64::emit32(&mut code, 0x3C810421); // STR Q1, [X1], #16

        // SUBS X2, X2, #1
        aarch64::emit32(&mut code, 0xF1000442); // SUBS X2, X2, #1

        // B.NE loop
        let bne_pos = code.len();
        let offset = (loop_pos as i64 - bne_pos as i64) / 4;
        let imm19 = (offset as u32) & 0x7FFFF;
        aarch64::emit32(&mut code, 0x54000000 | (imm19 << 5) | 0x01); // B.NE

        // done: patch CBZ
        let done_pos = code.len();
        {
            let patch_off = ((done_pos as i64 - cbz_pos as i64) / 4) as u32 & 0x7FFFF;
            let existing = u32::from_le_bytes([
                code[cbz_pos],
                code[cbz_pos + 1],
                code[cbz_pos + 2],
                code[cbz_pos + 3],
            ]);
            let patched = (existing & 0xFF00001F) | (patch_off << 5);
            code[cbz_pos..cbz_pos + 4].copy_from_slice(&patched.to_le_bytes());
        }

        // RET
        aarch64::emit32(&mut code, 0xD65F03C0);

        unsafe {
            let exec = executable::ExecutableCode::from_code(&code).expect("mmap failed");
            let func: executable::ScanlineKernelFn = exec.as_fn();

            let xs = [vdupq_n_f32(1.0), vdupq_n_f32(2.0)];
            let y = vdupq_n_f32(100.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);
            let mut out = [vdupq_n_f32(-999.0); 2];

            func(xs.as_ptr(), y, z, w, out.as_mut_ptr(), 2);

            let v0 = vgetq_lane_f32(out[0], 0);
            let v1 = vgetq_lane_f32(out[1], 0);
            assert_eq!(v0, 101.0, "handcoded scanline out[0]");
            assert_eq!(v1, 102.0, "handcoded scanline out[1]");
        }
    }

    /// Test the register-offset LDR/STR encoding with a manual loop.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_scanline_reg_offset_encoding() {
        use core::arch::aarch64::*;

        // Test: load+store one Q value using post-index addressing.
        // ABI: x0=input_ptr, v0=Y(100.0), x1=output_ptr, x2=count(1)
        let mut code = Vec::new();

        // LDR Q1, [X0], #16  (load and advance)
        aarch64::emit32(&mut code, 0x3CC10401u32); // known-good encoding from handcoded test

        // FADD V1.4S, V1.4S, V0.4S  (X[0] + Y)
        aarch64::emit32(&mut code, 0x4E20D421);

        // STR Q1, [X1], #16  (store and advance)
        aarch64::emit32(&mut code, 0x3C810421u32); // known-good encoding

        // RET
        aarch64::emit32(&mut code, 0xD65F03C0);

        unsafe {
            let exec = executable::ExecutableCode::from_code(&code).expect("mmap failed");
            let func: executable::ScanlineKernelFn = exec.as_fn();

            let xs = [vdupq_n_f32(1.0)];
            let y = vdupq_n_f32(100.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);
            let mut out = [vdupq_n_f32(-999.0)];

            func(xs.as_ptr(), y, z, w, out.as_mut_ptr(), 1);

            let v = vgetq_lane_f32(out[0], 0);
            assert_eq!(v, 101.0, "post-index single out[0]");
        }
    }

    /// Test scanline kernel for X + Y (simplest possible expression).
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_scanline_add_xy() {
        use crate::arena::ExprArena;
        use core::arch::aarch64::*;

        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let root = arena.push_binary(OpKind::Add, x, y);

        let result = compile_arena_dag_scanline(&arena, root).expect("scanline compilation failed");

        let scanline = crate::jit_manifold::ScanlineJitManifold::new(result.code);

        unsafe {
            let xs = [
                vdupq_n_f32(1.0),
                vdupq_n_f32(2.0),
                vdupq_n_f32(3.0),
                vdupq_n_f32(10.0),
            ];
            let y = vdupq_n_f32(100.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);
            let mut output = [vdupq_n_f32(0.0); 4];

            scanline.eval_scanline(&xs, y, z, w, &mut output);

            assert_eq!(vgetq_lane_f32(output[0], 0), 101.0, "1 + 100");
            assert_eq!(vgetq_lane_f32(output[1], 0), 102.0, "2 + 100");
            assert_eq!(vgetq_lane_f32(output[2], 0), 103.0, "3 + 100");
            assert_eq!(vgetq_lane_f32(output[3], 0), 110.0, "10 + 100");
        }
    }

    /// Test scanline kernel for return X (identity).
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_scanline_return_x() {
        use crate::arena::ExprArena;
        use core::arch::aarch64::*;

        let mut arena = ExprArena::new();
        let root = arena.push_var(0); // Just return X

        let result = compile_arena_dag_scanline(&arena, root).expect("scanline compilation failed");

        let scanline = crate::jit_manifold::ScanlineJitManifold::new(result.code);

        unsafe {
            let xs = [vdupq_n_f32(42.0), vdupq_n_f32(7.0)];
            let y = vdupq_n_f32(0.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);
            let mut output = [vdupq_n_f32(0.0); 2];

            scanline.eval_scanline(&xs, y, z, w, &mut output);

            assert_eq!(vgetq_lane_f32(output[0], 0), 42.0);
            assert_eq!(vgetq_lane_f32(output[1], 0), 7.0);
        }
    }

    /// Test scanline kernel with constant: (X * 2.0) + 3.0
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_scanline_with_constants() {
        use crate::arena::ExprArena;
        use core::arch::aarch64::*;

        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let two = arena.push_const(2.0);
        let three = arena.push_const(3.0);
        let x_times_2 = arena.push_binary(OpKind::Mul, x, two);
        let root = arena.push_binary(OpKind::Add, x_times_2, three);

        let result = compile_arena_dag_scanline(&arena, root).expect("scanline compilation failed");

        let scanline = crate::jit_manifold::ScanlineJitManifold::new(result.code);

        unsafe {
            let xs = [vdupq_n_f32(0.0), vdupq_n_f32(1.0), vdupq_n_f32(5.0)];
            let y = vdupq_n_f32(0.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);
            let mut output = [vdupq_n_f32(0.0); 3];

            scanline.eval_scanline(&xs, y, z, w, &mut output);

            assert_eq!(vgetq_lane_f32(output[0], 0), 3.0, "0*2+3");
            assert_eq!(vgetq_lane_f32(output[1], 0), 5.0, "1*2+3");
            assert_eq!(vgetq_lane_f32(output[2], 0), 13.0, "5*2+3");
        }
    }

    /// Test scanline matches single-pixel results for a complex expression.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_scanline_matches_single_pixel() {
        use crate::arena::ExprArena;
        use core::arch::aarch64::*;

        // (X + Y) * (Z - W)
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let z = arena.push_var(2);
        let w = arena.push_var(3);
        let sum = arena.push_binary(OpKind::Add, x, y);
        let diff = arena.push_binary(OpKind::Sub, z, w);
        let root = arena.push_binary(OpKind::Mul, sum, diff);

        // Compile both variants from the same arena.
        let single = compile_arena_dag(&arena, root).expect("single-pixel compile failed");
        let scanline_result =
            compile_arena_dag_scanline(&arena, root).expect("scanline compile failed");

        let single_jit = crate::jit_manifold::JitManifold::new(single.code);
        let scanline_jit = crate::jit_manifold::ScanlineJitManifold::new(scanline_result.code);

        unsafe {
            let y_val = vdupq_n_f32(2.0);
            let z_val = vdupq_n_f32(7.0);
            let w_val = vdupq_n_f32(3.0);

            let xs = [vdupq_n_f32(1.0), vdupq_n_f32(5.0), vdupq_n_f32(-3.0)];
            let mut scanline_out = [vdupq_n_f32(0.0); 3];
            scanline_jit.eval_scanline(&xs, y_val, z_val, w_val, &mut scanline_out);

            for (i, &x_val) in xs.iter().enumerate() {
                let single_result = single_jit.call(x_val, y_val, z_val, w_val);
                let s = vgetq_lane_f32(single_result, 0);
                let sl = vgetq_lane_f32(scanline_out[i], 0);
                assert_eq!(s, sl, "scanline[{i}] mismatch: single={s}, scanline={sl}");
            }
        }
    }

    /// Test scanline with empty input (should be a no-op).
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_scanline_empty() {
        use crate::arena::ExprArena;
        use core::arch::aarch64::*;

        let mut arena = ExprArena::new();
        let root = arena.push_var(0);

        let result = compile_arena_dag_scanline(&arena, root).expect("scanline compilation failed");

        let scanline = crate::jit_manifold::ScanlineJitManifold::new(result.code);

        unsafe {
            let xs: &[float32x4_t] = &[];
            let y = vdupq_n_f32(0.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);
            let output: &mut [float32x4_t] = &mut [];

            // Should not crash or touch any memory.
            scanline.eval_scanline(xs, y, z, w, output);
        }
    }
}
