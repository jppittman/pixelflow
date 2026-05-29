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
use crate::arena::{ExprArena, ExprId, ExprNode};

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
///
/// This is a catamorphism (fold) over the [`ExprArena`] DAG, walked directly by
/// `ExprId` — no intermediate `Expr` tree is materialized. Shared subexpressions
/// are labeled as trees (the x86-64 emitter re-emits them), which matches the
/// behavior of the previous `to_expr` bridge minus the heap allocation.
#[cfg(target_arch = "x86_64")]
fn needs_arena(arena: &ExprArena, id: ExprId) -> usize {
    match arena.node(id) {
        // Leaves need 1 register to hold their value
        ExprNode::Var(_) | ExprNode::Const(_) => 1,
        ExprNode::Param(i) => panic!(
            "ExprNode::Param({}) reached the JIT emitter — call substitute_params before compile",
            i
        ),

        // Unary: same as child (result overwrites input)
        ExprNode::Unary(_, child) => needs_arena(arena, *child),

        // Binary: Sethi-Ullman magic
        ExprNode::Binary(_, left, right) => {
            let l = needs_arena(arena, *left);
            let r = needs_arena(arena, *right);
            if l == r { l + 1 } else { l.max(r) }
        }

        // Ternary: need to hold all three, then combine
        ExprNode::Ternary(_, a, b, c) => {
            let na = needs_arena(arena, *a);
            let nb = needs_arena(arena, *b);
            let nc = needs_arena(arena, *c);
            // Conservative: max + ties
            let max = na.max(nb).max(nc);
            if (na == nb) || (nb == nc) || (na == nc) {
                max + 1
            } else {
                max
            }
        }

        ExprNode::Nary(_, start, len) => {
            let children = arena.nary_children_slice(*start, *len);
            children
                .iter()
                .map(|c| needs_arena(arena, *c))
                .max()
                .unwrap_or(0)
                + children.len().saturating_sub(1)
        }
    }
}

// =============================================================================
// Functional Emitter (x86-64)
// =============================================================================

/// Emit code for the subtree rooted at `id`, walking the [`ExprArena`] directly.
///
/// Returns the generated bytes and the register holding the result. No `Expr`
/// tree is allocated; the arena is the IR end to end.
#[cfg(target_arch = "x86_64")]
fn emit_arena(arena: &ExprArena, id: ExprId, depth: u8) -> Result<(Vec<u8>, Reg), &'static str> {
    use x86_64::*;

    match arena.node(id) {
        ExprNode::Var(i) => {
            if *i as usize >= INPUT_REGS.len() {
                return Err("variable index out of range");
            }
            Ok((Vec::new(), INPUT_REGS[*i as usize]))
        }

        ExprNode::Const(val) => {
            let dst = Reg(SCRATCH_BASE + depth);
            let mut code = Vec::new();
            let scratch = [Reg(13), Reg(14), Reg(15), Reg(15)];
            emit_const(&mut code, dst, *val, scratch);
            Ok((code, dst))
        }

        ExprNode::Param(_) => Err("Param not supported directly here"),

        ExprNode::Unary(op, child) => {
            let (mut code, src) = emit_arena(arena, *child, depth)?;
            let dst = Reg(SCRATCH_BASE + depth);
            let scratch = [Reg(13), Reg(14), Reg(15), Reg(15)];
            emit_unary(&mut code, *op, dst, src, scratch);
            Ok((code, dst))
        }

        ExprNode::Binary(op, left, right) => {
            let n_l = needs_arena(arena, *left);
            let n_r = needs_arena(arena, *right);
            let dst = Reg(SCRATCH_BASE + depth);

            // Sethi-Ullman: evaluate the heavier child first so it can use the
            // lower (already-free) register pressure window.
            let (mut code, l_reg, r_reg) = if n_l >= n_r {
                let (mut code, l_reg) = emit_arena(arena, *left, depth)?;
                let (r_code, r_reg) = emit_arena(arena, *right, depth + 1)?;
                code.extend(r_code);
                (code, l_reg, r_reg)
            } else {
                let (mut code, r_reg) = emit_arena(arena, *right, depth)?;
                let (l_code, l_reg) = emit_arena(arena, *left, depth + 1)?;
                code.extend(l_code);
                (code, l_reg, r_reg)
            };

            match op {
                OpKind::Atan2 => {
                    let scratch = [Reg(13), Reg(14), Reg(15), Reg(15)];
                    x86_64::emit_binary_transcendental(&mut code, *op, dst, l_reg, r_reg, scratch);
                }
                _ => emit_binary(&mut code, *op, dst, l_reg, r_reg),
            }
            Ok((code, dst))
        }

        ExprNode::Ternary(op, a, b, c) => {
            let dst = Reg(SCRATCH_BASE + depth);

            match op {
                OpKind::MulAdd => {
                    // x86 doesn't have FMLA, use FMUL + FADD
                    let (mut code, a_reg) = emit_arena(arena, *a, depth)?;
                    let (b_code, b_reg) = emit_arena(arena, *b, depth + 1)?;
                    let (c_code, c_reg) = emit_arena(arena, *c, depth + 2)?;

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

        ExprNode::Nary(_, _, _) => Err("Nary not supported in JIT"),
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
/// The arena IS the linearized schedule, so the linearization pass is free:
/// `ExprId` maps 1:1 to `ValueId` for reachable nodes.
#[cfg(target_arch = "aarch64")]
pub fn compile_arena_dag(
    arena: &crate::arena::ExprArena,
    root: crate::arena::ExprId,
) -> Result<CompileResult, &'static str> {
    compile_arena_dag_with_ctx(arena, root, EmitCtx::default())
}

/// Compile an [`ExprArena`] DAG with explicit register budget.
///
/// The arena's append-only structure guarantees topological order, so the
/// schedule comes for free -- `ExprId` maps 1:1 to `ValueId` for reachable nodes.
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
/// `compile_arena_dag_with_ctx` and the scanline compilers all produce the same
/// `(schedule, uses_map)` format and then converge here.
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

/// Build a schedule directly from an [`ExprArena`].
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
    // ValueIds are sequential starting from 0 (guaranteed by arena_to_schedule).
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

/// Compile an [`ExprArena`] DAG to executable code (x86-64).
#[cfg(target_arch = "x86_64")]
pub fn compile_arena(
    arena: &ExprArena,
    root: ExprId,
) -> Result<executable::ExecutableCode, &'static str> {
    compile_arena_dag(arena, root).map(|r| r.code)
}

/// Compile an [`ExprArena`] DAG to executable code (x86-64).
///
/// Walks the arena directly with a Sethi-Ullman tree emitter — no intermediate
/// `Expr` tree is materialized. Expressions produced by the extraction pipeline
/// are trees (no DAG sharing), so tree-walk emission is exact; shared nodes are
/// simply re-emitted.
#[cfg(target_arch = "x86_64")]
pub fn compile_arena_dag(arena: &ExprArena, root: ExprId) -> Result<CompileResult, &'static str> {
    compile_arena_dag_with_ctx(arena, root, EmitCtx::default())
}

/// Compile an [`ExprArena`] DAG with an explicit register budget (x86-64).
///
/// The `ctx` budget is advisory on x86-64: the Sethi-Ullman emitter assigns
/// scratch registers by tree position rather than performing graph-coloring
/// allocation, so it never spills within the supported depth bound.
#[cfg(target_arch = "x86_64")]
pub fn compile_arena_dag_with_ctx(
    arena: &ExprArena,
    root: ExprId,
    _ctx: EmitCtx,
) -> Result<CompileResult, &'static str> {
    const MAX_DEPTH: usize = 64;
    if arena.depth(root) > MAX_DEPTH {
        return Err("expression too deep");
    }

    let (mut code, result_reg) = emit_arena(arena, root, 0)?;

    // Move result to xmm0 if not already there.
    if result_reg.0 != 0 {
        x86_64::emit_movaps(&mut code, Reg(0), result_reg);
    }

    // RET
    code.push(0xC3);

    let code = unsafe { executable::ExecutableCode::from_code(&code)? };
    Ok(CompileResult {
        code,
        spill_count: 0,
        spill_bytes: 0,
        max_regs: EmitCtx::default().max_regs,
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_arch = "aarch64")]
    use alloc::boxed::Box;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_needs_simple() {
        // X + Y: both leaves need 1, binary needs max(1,1)+1 = 2
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let root = arena.push_binary(OpKind::Add, x, y);
        assert_eq!(needs_arena(&arena, root), 2);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_needs_unbalanced() {
        // (X + Y) + Z: left needs 2, right needs 1, total = max(2,1) = 2
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let left = arena.push_binary(OpKind::Add, x, y);
        let z = arena.push_var(2);
        let root = arena.push_binary(OpKind::Add, left, z);
        assert_eq!(needs_arena(&arena, root), 2);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_needs_balanced_deep() {
        // (X + Y) + (Z + W): both sides need 2, total = 2+1 = 3
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let left = arena.push_binary(OpKind::Add, x, y);
        let z = arena.push_var(2);
        let w = arena.push_var(3);
        let right = arena.push_binary(OpKind::Add, z, w);
        let root = arena.push_binary(OpKind::Add, left, right);
        assert_eq!(needs_arena(&arena, root), 3);
    }

    // =========================================================================
    // Graph Coloring (DAG) Tests
    // =========================================================================

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

    /// Test that Select short-circuits: when mask is all-true, the false arm
    /// (which contains a division by zero) must NOT produce NaN in the output.
    /// Test Select with all-false mask: should return false arm.
    /// Test Select with mixed mask: BSL path, both arms evaluated.
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
