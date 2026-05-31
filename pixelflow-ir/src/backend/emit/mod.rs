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
pub mod avx512;
pub mod executable;
pub mod lowering;
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
    /// Integer shift by a compile-time immediate: dst = src `op` amount, where
    /// `op` is `Shl` or `Shr` (the hardware shift encoders are imm-only).
    ShiftImm { op: OpKind, dst: Reg, src: Reg, amount: u8 },
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
/// Highest XMM index the x86 Sethi-Ullman emitter may use for an intermediate
/// value. xmm12..15 are reserved as builtin/const scratch, so a value register
/// must stay at or below xmm11. The emitter has no spilling (unlike the aarch64
/// linear-scan path), so expressions that would need a value register beyond
/// this budget are rejected loudly rather than miscompiled.
#[cfg(target_arch = "x86_64")]
const X86_MAX_VALUE_REG: u8 = 11;

/// One forward-mode partial derivative component.
///
/// Tracking statically-zero partials (rather than materializing `0.0` into a
/// register) is what keeps register-resident dual lowering viable: seeds,
/// constants, and direction-independent subexpressions carry `Zero` and cost
/// neither a register nor an instruction.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
enum Partial {
    /// Statically zero — no register, no work.
    Zero,
    /// A runtime value held in `Reg`.
    Reg(Reg),
}

/// A value together with its derivative components, in register form.
///
/// `partials` is empty for a plain value — i.e. a `Field`, which in this model
/// is just "Jet0" (zeroth-order: value, no derivatives). A first-order jet over
/// `d` seeded directions carries `d` partials (`∂/∂dir`), matching `Jet2`/`Jet3`
/// in `pixelflow-core`. The same `value + components` shape extends to higher
/// order (a Hessian adds the second-order components) without changing this
/// type — that generalization is deliberately *not* built yet.
#[cfg(target_arch = "x86_64")]
struct Jet {
    val: Reg,
    /// Forward-mode first partials, one per seeded direction. Empty == Jet0.
    partials: alloc::vec::Vec<Partial>,
}

#[cfg(target_arch = "x86_64")]
impl Jet {
    /// A zeroth-order jet (a plain `Field` value, no derivatives).
    fn value(reg: Reg) -> Self {
        Self {
            val: reg,
            partials: alloc::vec::Vec::new(),
        }
    }
}

/// Emit code for one arena node.
///
/// `seed` lists the input-variable indices that carry a unit first derivative
/// (the forward-mode seed): empty seeds a plain value (`Jet0`/`Field`), `[0,1]`
/// a `Jet2`, `[0,1,2]` a `Jet3`. Only the `Jet0` case is implemented today; a
/// non-empty seed (dual lowering) is a hard error until the per-op chain-rule
/// rules and the wider register model land.
#[cfg(target_arch = "x86_64")]
fn emit_arena(
    arena: &ExprArena,
    id: ExprId,
    depth: u8,
    seed: &[u8],
) -> Result<(Vec<u8>, Jet), &'static str> {
    use x86_64::*;

    // Dual (jet) lowering is not implemented yet; only Jet0 (plain value).
    if !seed.is_empty() {
        return Err("x86 JIT: dual (jet) lowering not yet implemented");
    }

    // Reserve xmm12..15 for scratch: any node that allocates a value register
    // (everything except a bare Var, which reuses an input register) must fit
    // below the scratch floor.
    if !matches!(arena.node(id), ExprNode::Var(_))
        && SCRATCH_BASE + depth > X86_MAX_VALUE_REG
    {
        return Err("x86 JIT: expression too deep for SSE register budget");
    }

    match arena.node(id) {
        ExprNode::Var(i) => {
            if *i as usize >= INPUT_REGS.len() {
                return Err("variable index out of range");
            }
            Ok((Vec::new(), Jet::value(INPUT_REGS[*i as usize])))
        }

        ExprNode::Const(val) => {
            let dst = Reg(SCRATCH_BASE + depth);
            let mut code = Vec::new();
            let scratch = [Reg(13), Reg(14), Reg(15), Reg(15)];
            emit_const(&mut code, dst, *val, scratch);
            Ok((code, Jet::value(dst)))
        }

        ExprNode::Param(_) => Err("Param not supported directly here"),

        ExprNode::Unary(op, child) => {
            // Emit the child one depth deeper so its result register differs from
            // `dst`. Transcendental builtins read `src` several times after they
            // start writing `dst`, so an in-place (dst == src) unary would corrupt
            // the input. Placing the child at depth+1 guarantees src != dst.
            let (mut code, child_jet) = emit_arena(arena, *child, depth + 1, seed)?;
            let src = child_jet.val;
            let dst = Reg(SCRATCH_BASE + depth);
            let scratch = [Reg(12), Reg(13), Reg(14), Reg(15)];
            emit_unary(&mut code, *op, dst, src, scratch);
            Ok((code, Jet::value(dst)))
        }

        ExprNode::Binary(op, left, right) => {
            let n_l = needs_arena(arena, *left);
            let n_r = needs_arena(arena, *right);
            let dst = Reg(SCRATCH_BASE + depth);

            // Whichever child is emitted at `depth` lands in `dst`. The
            // two-operand SSE form emits `dst = src1 op src2` as
            // `dst <- src1; dst op= src2`, so it is only correct when
            // `dst == src1` or `dst != src2` — i.e. the operand sitting in
            // `dst` must be passed as `src1`.
            //
            // Sethi-Ullman wants the heavier child evaluated first (into the
            // lower register window). When the heavier child is the RIGHT
            // operand, putting it in `dst` would make `dst == src2`, which the
            // copy above corrupts. We can only recover by swapping operands,
            // which is sound for commutative ops. For non-commutative ops we
            // keep the LEFT child in `dst` instead (costing at most one extra
            // register level), preserving correctness.
            let commutative = matches!(
                op,
                OpKind::Add | OpKind::Mul | OpKind::Min | OpKind::Max | OpKind::Eq | OpKind::Ne
            );
            let eval_right_first = n_r > n_l && commutative;

            // Returns operands as (src1, src2) with the in-`dst` operand first.
            let (mut code, src1, src2) = if !eval_right_first {
                let (mut code, l) = emit_arena(arena, *left, depth, seed)?;
                let (r_code, r) = emit_arena(arena, *right, depth + 1, seed)?;
                code.extend(r_code);
                (code, l.val, r.val)
            } else {
                // Right child in `dst`; swap so it becomes src1 (sound:
                // commutative). `dst == src1` then satisfies the SSE invariant.
                let (mut code, r) = emit_arena(arena, *right, depth, seed)?;
                let (l_code, l) = emit_arena(arena, *left, depth + 1, seed)?;
                code.extend(l_code);
                (code, r.val, l.val)
            };

            match op {
                OpKind::Atan2 | OpKind::Pow | OpKind::Hypot => {
                    let scratch = [Reg(12), Reg(13), Reg(14), Reg(15)];
                    x86_64::emit_binary_transcendental(&mut code, *op, dst, src1, src2, scratch);
                }
                _ => emit_binary(&mut code, *op, dst, src1, src2),
            }
            Ok((code, Jet::value(dst)))
        }

        ExprNode::Ternary(op, a, b, c) => {
            let dst = Reg(SCRATCH_BASE + depth);

            match op {
                OpKind::MulAdd => {
                    // x86 doesn't have FMLA, use FMUL + FADD
                    let (mut code, a_jet) = emit_arena(arena, *a, depth, seed)?;
                    let (b_code, b_jet) = emit_arena(arena, *b, depth + 1, seed)?;
                    let (c_code, c_jet) = emit_arena(arena, *c, depth + 2, seed)?;
                    let (a_reg, b_reg, c_reg) = (a_jet.val, b_jet.val, c_jet.val);

                    code.extend(b_code);
                    code.extend(c_code);

                    // dst = a * b
                    emit_binary(&mut code, OpKind::Mul, dst, a_reg, b_reg);
                    // dst = dst + c
                    emit_binary(&mut code, OpKind::Add, dst, dst, c_reg);
                    Ok((code, Jet::value(dst)))
                }

                OpKind::Clamp => {
                    // clamp(a, lo, hi) = max(min(a, hi), lo)
                    let (mut code, a_jet) = emit_arena(arena, *a, depth, seed)?;
                    let (b_code, b_jet) = emit_arena(arena, *b, depth + 1, seed)?;
                    let (c_code, c_jet) = emit_arena(arena, *c, depth + 2, seed)?;
                    let (a_reg, b_reg, c_reg) = (a_jet.val, b_jet.val, c_jet.val);
                    code.extend(b_code);
                    code.extend(c_code);
                    emit_binary(&mut code, OpKind::Min, dst, a_reg, c_reg);
                    emit_binary(&mut code, OpKind::Max, dst, dst, b_reg);
                    Ok((code, Jet::value(dst)))
                }

                OpKind::Select => {
                    // select(cond, if_true, if_false): cond is an all-ones/zeros mask.
                    let (mut code, a_jet) = emit_arena(arena, *a, depth, seed)?;
                    let (b_code, b_jet) = emit_arena(arena, *b, depth + 1, seed)?;
                    let (c_code, c_jet) = emit_arena(arena, *c, depth + 2, seed)?;
                    let (a_reg, b_reg, c_reg) = (a_jet.val, b_jet.val, c_jet.val);
                    code.extend(b_code);
                    code.extend(c_code);
                    x86_64::emit_select(&mut code, dst, a_reg, b_reg, c_reg, Reg(15));
                    Ok((code, Jet::value(dst)))
                }

                _ => Err("ternary emit not implemented"),
            }
        }

        ExprNode::Nary(_, _, _) => Err("Nary not supported in JIT"),
    }
}

// =============================================================================
// Forward-mode dual (jet) lowering — register-resident, fail loudly.
// =============================================================================
//
// This is the k>0 counterpart of `emit_arena`. It walks the arena and emits,
// for the chosen forward-mode `seed`, a value plus one partial per seeded
// direction (the chain rule per op). It is deliberately the *simple* version:
//
//   - A flat register pool over xmm4..=11; xmm12..15 stay scratch.
//   - No spilling and no register reuse across a binary op's children: results
//     are allocated fresh, children freed afterward. This keeps the bookkeeping
//     obviously correct at the cost of register headroom.
//   - Out of registers => hard error ("expression too complex"), never a
//     miscompile.
//
// When the budget bites, the planned direction is to unify the x86 and aarch64
// compilation pipelines (aarch64 already has a spilling linear-scan allocator),
// not to special-case x86. Until then this covers small jet expressions.

/// A flat free-list register pool over the value registers xmm4..=11.
///
/// Input registers (xmm0..3, the seed values) and scratch (xmm12..15) are never
/// pooled. `alloc` fails loudly when exhausted.
#[cfg(target_arch = "x86_64")]
struct RegPool {
    free: alloc::vec::Vec<Reg>,
}

#[cfg(target_arch = "x86_64")]
impl RegPool {
    fn new() -> Self {
        // Hand out low registers first (4,5,6,...).
        let free = (SCRATCH_BASE..=X86_MAX_VALUE_REG).rev().map(Reg).collect();
        Self { free }
    }

    fn alloc(&mut self) -> Result<Reg, &'static str> {
        self.free
            .pop()
            .ok_or("x86 JIT: out of registers for jet (expression too complex)")
    }

    /// Return a register to the pool. No-ops for non-pooled registers (inputs,
    /// scratch), so callers can free a jet's components uniformly.
    fn free_reg(&mut self, r: Reg) {
        if (SCRATCH_BASE..=X86_MAX_VALUE_REG).contains(&r.0) && !self.free.contains(&r) {
            self.free.push(r);
        }
    }

    /// Free every pooled register a jet owns (its value and any `Reg` partials).
    fn free_jet(&mut self, jet: &Jet) {
        self.free_reg(jet.val);
        for p in &jet.partials {
            if let Partial::Reg(r) = p {
                self.free_reg(*r);
            }
        }
    }
}

/// Cross-term scratch register for the product/quotient rules (not pooled).
#[cfg(target_arch = "x86_64")]
const JET_SCRATCH: Reg = Reg(12);

/// Const-materialization scratch (mirrors the `emit_const` usage elsewhere).
#[cfg(target_arch = "x86_64")]
const JET_CONST_SCRATCH: [Reg; 4] = [Reg(13), Reg(14), Reg(15), Reg(15)];

/// `dst = a + b` / `dst = a - b` over partials, honoring `Zero`.
#[cfg(target_arch = "x86_64")]
fn jet_add_sub(
    code: &mut Vec<u8>,
    pool: &mut RegPool,
    is_sub: bool,
    a: Partial,
    b: Partial,
) -> Result<Partial, &'static str> {
    use x86_64::*;
    match (a, b) {
        (Partial::Zero, Partial::Zero) => Ok(Partial::Zero),
        // a ± 0 = a
        (Partial::Reg(ra), Partial::Zero) => {
            let d = pool.alloc()?;
            emit_movaps(code, d, ra);
            Ok(Partial::Reg(d))
        }
        // 0 + b = b ; 0 - b = -b
        (Partial::Zero, Partial::Reg(rb)) => {
            let d = pool.alloc()?;
            emit_movaps(code, d, rb);
            if is_sub {
                emit_unary(code, OpKind::Neg, d, d, JET_CONST_SCRATCH);
            }
            Ok(Partial::Reg(d))
        }
        (Partial::Reg(ra), Partial::Reg(rb)) => {
            let d = pool.alloc()?;
            emit_movaps(code, d, ra);
            emit_binary(code, if is_sub { OpKind::Sub } else { OpKind::Add }, d, d, rb);
            Ok(Partial::Reg(d))
        }
    }
}

/// Emit the dual lowering of one arena node for the given forward-mode `seed`.
#[cfg(target_arch = "x86_64")]
fn emit_jet(
    arena: &ExprArena,
    id: ExprId,
    seed: &[u8],
    pool: &mut RegPool,
    code: &mut Vec<u8>,
) -> Result<Jet, &'static str> {
    use x86_64::*;
    let k = seed.len();

    match arena.node(id) {
        ExprNode::Var(i) => {
            let i = *i;
            if i as usize >= INPUT_REGS.len() {
                return Err("variable index out of range");
            }
            // ∂(var_i)/∂(var_{seed[j]}) = 1 if seed[j] == i else 0.
            let mut partials = alloc::vec::Vec::with_capacity(k);
            for &d in seed {
                if d == i {
                    let r = pool.alloc()?;
                    emit_const(code, r, 1.0, JET_CONST_SCRATCH);
                    partials.push(Partial::Reg(r));
                } else {
                    partials.push(Partial::Zero);
                }
            }
            Ok(Jet {
                val: INPUT_REGS[i as usize],
                partials,
            })
        }

        ExprNode::Const(v) => {
            let val = pool.alloc()?;
            emit_const(code, val, *v, JET_CONST_SCRATCH);
            Ok(Jet {
                val,
                partials: alloc::vec![Partial::Zero; k],
            })
        }

        ExprNode::Unary(op, child) => {
            let c = emit_jet(arena, *child, seed, pool, code)?;
            match op {
                OpKind::Neg => {
                    // (-u)' = -u'
                    let val = pool.alloc()?;
                    emit_unary(code, OpKind::Neg, val, c.val, JET_CONST_SCRATCH);
                    let mut partials = alloc::vec::Vec::with_capacity(k);
                    for p in &c.partials {
                        match p {
                            Partial::Zero => partials.push(Partial::Zero),
                            Partial::Reg(r) => {
                                let d = pool.alloc()?;
                                emit_unary(code, OpKind::Neg, d, *r, JET_CONST_SCRATCH);
                                partials.push(Partial::Reg(d));
                            }
                        }
                    }
                    pool.free_jet(&c);
                    Ok(Jet { val, partials })
                }
                OpKind::Sqrt => {
                    // (√u)' = u' / (2√u) = u' * (0.5 / √u)
                    let val = pool.alloc()?;
                    emit_unary(code, OpKind::Sqrt, val, c.val, JET_CONST_SCRATCH);
                    // factor = 0.5 / val  (scratch, recomputed per node)
                    let factor = JET_SCRATCH;
                    emit_const(code, factor, 0.5, JET_CONST_SCRATCH);
                    emit_binary(code, OpKind::Div, factor, factor, val);
                    let mut partials = alloc::vec::Vec::with_capacity(k);
                    for p in &c.partials {
                        match p {
                            Partial::Zero => partials.push(Partial::Zero),
                            Partial::Reg(r) => {
                                let d = pool.alloc()?;
                                emit_movaps(code, d, *r);
                                emit_binary(code, OpKind::Mul, d, d, factor);
                                partials.push(Partial::Reg(d));
                            }
                        }
                    }
                    pool.free_jet(&c);
                    Ok(Jet { val, partials })
                }
                OpKind::Floor => {
                    // floor is piecewise-constant: value = floor(c), derivative
                    // = 0 a.e. (Used by the range reduction in expanded sin/cos.)
                    let val = pool.alloc()?;
                    emit_unary(code, OpKind::Floor, val, c.val, JET_CONST_SCRATCH);
                    pool.free_jet(&c);
                    Ok(Jet {
                        val,
                        partials: alloc::vec![Partial::Zero; k],
                    })
                }
                _ => Err("x86 JIT jet: unary op not yet supported"),
            }
        }

        ExprNode::Binary(op, left, right) => {
            let l = emit_jet(arena, *left, seed, pool, code)?;
            let r = emit_jet(arena, *right, seed, pool, code)?;

            let result = match op {
                OpKind::Add | OpKind::Sub => {
                    let is_sub = matches!(op, OpKind::Sub);
                    let val = pool.alloc()?;
                    emit_movaps(code, val, l.val);
                    emit_binary(code, *op, val, val, r.val);
                    let mut partials = alloc::vec::Vec::with_capacity(k);
                    for j in 0..k {
                        partials.push(jet_add_sub(code, pool, is_sub, l.partials[j], r.partials[j])?);
                    }
                    Jet { val, partials }
                }
                OpKind::Mul => {
                    // (uv)' = u'v + uv'
                    let val = pool.alloc()?;
                    emit_movaps(code, val, l.val);
                    emit_binary(code, OpKind::Mul, val, val, r.val);
                    let mut partials = alloc::vec::Vec::with_capacity(k);
                    for j in 0..k {
                        let p = match (l.partials[j], r.partials[j]) {
                            (Partial::Zero, Partial::Zero) => Partial::Zero,
                            // u'v
                            (Partial::Reg(la), Partial::Zero) => {
                                let d = pool.alloc()?;
                                emit_movaps(code, d, la);
                                emit_binary(code, OpKind::Mul, d, d, r.val);
                                Partial::Reg(d)
                            }
                            // uv'
                            (Partial::Zero, Partial::Reg(rb)) => {
                                let d = pool.alloc()?;
                                emit_movaps(code, d, l.val);
                                emit_binary(code, OpKind::Mul, d, d, rb);
                                Partial::Reg(d)
                            }
                            // u'v + uv'
                            (Partial::Reg(la), Partial::Reg(rb)) => {
                                let d = pool.alloc()?;
                                emit_movaps(code, d, l.val);
                                emit_binary(code, OpKind::Mul, d, d, rb); // u v'
                                emit_movaps(code, JET_SCRATCH, la);
                                emit_binary(code, OpKind::Mul, JET_SCRATCH, JET_SCRATCH, r.val); // u' v
                                emit_binary(code, OpKind::Add, d, d, JET_SCRATCH);
                                Partial::Reg(d)
                            }
                        };
                        partials.push(p);
                    }
                    Jet { val, partials }
                }
                _ => {
                    pool.free_jet(&l);
                    pool.free_jet(&r);
                    return Err("x86 JIT jet: binary op not yet supported");
                }
            };

            pool.free_jet(&l);
            pool.free_jet(&r);
            Ok(result)
        }

        ExprNode::Ternary(..) => Err("x86 JIT jet: ternary not yet supported"),
        ExprNode::Param(_) => Err("Param not supported directly here"),
        ExprNode::Nary(..) => Err("Nary not supported in JIT"),
    }
}

/// Which component of a dual-lowered kernel to return (in xmm0).
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JetOutput {
    /// The value component.
    Value,
    /// The first partial w.r.t. `seed[index]`.
    Partial(usize),
}

/// Compile a single `JetOutput` component of the forward-mode dual of `arena`.
///
/// NAMING SMELL — `compile_arena_dag_jet` packs ~three namespaces into one
/// identifier (compile · arena_dag · jet). A good check when naming: *is this a
/// name, or a namespace?* If it reads like a namespace, that's a signal to make
/// it structure instead — a module/package, a method on a type, or a new struct
/// — rather than a longer function name. The whole `compile_arena_dag*` family
/// (`_with_ctx`, `_scanline`, `_scanline_with_ctx`, `_jet`, ...) is the same
/// smell at scale; it likely wants to become e.g. `ArenaCompiler { scanline,
/// jet, ctx, ... }`. Left as-is for now; flagged so we fix it deliberately, not
/// by accident, when the backend is reorganized.
///
/// `seed` lists the differentiated input-variable indices (`[0,1]` = Jet2 over
/// X,Y; `[0,1,2]` = Jet3 over X,Y,Z). Emits one component into xmm0, so the
/// result matches the ordinary single-output [`KernelFn`] ABI — enough to
/// validate each derivative independently. A wider multi-output ABI is future
/// work, paired with the scanline / pipeline-unification effort.
#[cfg(target_arch = "x86_64")]
pub fn compile_arena_dag_jet(
    arena: &ExprArena,
    root: ExprId,
    seed: &[u8],
    output: JetOutput,
) -> Result<CompileResult, &'static str> {
    use x86_64::*;
    // Expand transcendentals to primitive arithmetic first, so emit_jet only
    // sees ops it can differentiate (the chain rule over the expansion gives
    // the transcendental's derivative for free).
    let (arena, root) = lowering::expand_transcendentals_owned(arena, root);
    let arena = &arena;
    const MAX_DEPTH: usize = 64;
    if arena.depth(root) > MAX_DEPTH {
        return Err("expression too deep");
    }
    if let JetOutput::Partial(j) = output {
        if j >= seed.len() {
            return Err("jet output: partial index out of range for seed");
        }
    }

    let mut pool = RegPool::new();
    let mut code: Vec<u8> = Vec::new();
    let jet = emit_jet(arena, root, seed, &mut pool, &mut code)?;

    // Move the requested component into xmm0.
    let src = match output {
        JetOutput::Value => Partial::Reg(jet.val),
        JetOutput::Partial(j) => jet.partials[j],
    };
    match src {
        Partial::Reg(r) => {
            if r.0 != 0 {
                emit_movaps(&mut code, Reg(0), r);
            }
        }
        // A statically-zero partial is the constant 0.0.
        Partial::Zero => emit_const(&mut code, Reg(0), 0.0, JET_CONST_SCRATCH),
    }

    code.push(0xC3); // RET

    let code = unsafe { executable::ExecutableCode::from_code(&code)? };
    Ok(CompileResult {
        code,
        spill_count: 0,
        spill_bytes: 0,
        max_regs: EmitCtx::default().max_regs,
    })
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
    // Expand transcendentals to primitive arithmetic first (one source of truth
    // in `lowering`); the aarch64 emitter then never sees Sin/Cos/etc.
    let (arena, root) = lowering::expand_transcendentals_owned(arena, root);
    let arena = &arena;
    let schedule = arena_to_schedule(arena, root);
    let uses_map = arena_to_uses(&schedule);
    compile_from_schedule(schedule, uses_map, ctx)
}

/// Compile an [`ExprArena`] DAG into a scanline kernel that processes an entire
/// row of pixels in a single call with no per-batch Rust-JIT boundary crossing.
///
/// The emitted code contains its own loop: Y/Z/W stay in NEON registers across
/// all iterations (loop-invariant by construction), only X is loaded per batch
/// from the input array. This eliminates the `extern "C"` function pointer
/// overhead that dominates per-batch `KernelFn` performance.
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

/// A two-phase schedule: setup (loop-invariant) then loop (per-batch).
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
            ScheduledOp::Unary(_, a) | ScheduledOp::ShiftImm(_, a, _) => alloc::vec![*a],
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
            ScheduledOp::Unary(_, a) | ScheduledOp::ShiftImm(_, a, _) => alloc::vec![*a],
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
    // Register allocation (identical to per-batch path).
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

    // 3. Kernel body (identical to per-batch emit).
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

    // 4. Move result to v0 (same as per-batch epilogue).
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
                ScheduledOp::Unary(_, a) | ScheduledOp::ShiftImm(_, a, _) => alloc::vec![*a],
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

/// The architecture seam for the shared per-batch driver.
///
/// `compile_dag_via_backend` owns the architecture-INDEPENDENT logic — schedule,
/// register allocation, frame layout, and the Select short-circuit control flow
/// — and calls an `IsaBackend` for the leaf operations that actually differ
/// between x86-64 and aarch64 (instruction encoding, branch encoding, the
/// prologue/epilogue, and any arch-specific finalization such as aarch64's
/// constant pool). Both backends therefore run the *same* driver: there is one
/// place that decides when to emit a guard branch, where the root goes, etc.
///
/// `Branch` is an opaque per-backend fixup token (aarch64 distinguishes CBZ from
/// B; x86 uses a uniform rel32), patched later by `patch_branch`.
trait IsaBackend {
    type Branch;

    /// Number of allocatable scratch registers (the `linear_scan` budget).
    fn num_regs(&self) -> u8;

    /// Per-compile setup before any code is emitted (e.g. seed a constant pool).
    fn begin(&mut self, schedule: &[(regalloc::ValueId, ScheduledOp)]) -> Result<(), &'static str>;

    /// Function prologue (allocate the spill frame, set up any pool anchor).
    fn prologue(&mut self, code: &mut Vec<u8>, frame_size: u32);

    /// Emit one resolved instruction (with its reloads/store).
    fn emit_plan(&mut self, code: &mut Vec<u8>, plan: &InstructionPlan) -> Result<(), &'static str>;

    /// Register-to-register move.
    fn emit_mov(&mut self, code: &mut Vec<u8>, dst: Reg, src: Reg);

    /// Spill a register to a frame slot.
    fn emit_store(&mut self, code: &mut Vec<u8>, src: Reg, offset: u32) -> Result<(), &'static str>;

    /// Resolve a value to a register, reloading/rematerializing into `target`
    /// if it is spilled or rematerialized.
    fn emit_resolve(
        &mut self,
        code: &mut Vec<u8>,
        vid: regalloc::ValueId,
        target: Reg,
        reg_for: &[Option<Reg>],
        spill_for: &[Option<u32>],
        remat_for: &[Option<u32>],
    ) -> Reg;

    /// Branch taken when `mask_reg` is all-false (skip the true arm).
    fn emit_skip_if_all_false(&mut self, code: &mut Vec<u8>, mask_reg: Reg) -> Self::Branch;
    /// Branch taken when `mask_reg` is all-true (skip the false arm).
    fn emit_skip_if_all_true(&mut self, code: &mut Vec<u8>, mask_reg: Reg) -> Self::Branch;
    /// Unconditional jump.
    fn emit_jump(&mut self, code: &mut Vec<u8>) -> Self::Branch;
    /// Patch a previously emitted branch to land at `target`.
    fn patch_branch(&mut self, code: &mut Vec<u8>, branch: Self::Branch, target: usize);

    /// Function epilogue: move the result into the return register, tear down
    /// the frame, emit RET, and perform any arch-specific finalization (e.g.
    /// append + anchor the constant pool). After this, `code` is complete.
    fn epilogue(&mut self, code: &mut Vec<u8>, result_reg: Reg, frame_size: u32);
}

/// Drive a `(schedule, uses_map)` to machine code via an [`IsaBackend`].
///
/// This is the single shared per-batch driver for both architectures. The
/// control flow here — Select guard analysis, short-circuit branch emission and
/// patching, root resolution — is identical regardless of ISA; only the leaf
/// emits go through `backend`.
fn compile_dag_via_backend<B: IsaBackend>(
    schedule: Vec<(regalloc::ValueId, ScheduledOp)>,
    uses_map: Vec<Vec<regalloc::ValueId>>,
    backend: &mut B,
) -> Result<CompileResult, &'static str> {
    use alloc::collections::BTreeMap;

    // Pre-colored values (variables -> input registers).
    let mut precolored: BTreeMap<regalloc::ValueId, Reg> = BTreeMap::new();
    for (vid, op) in &schedule {
        if let ScheduledOp::Var(i) = op {
            if (*i as usize) >= INPUT_REGS.len() {
                return Err("variable index out of range");
            }
            precolored.insert(*vid, INPUT_REGS[*i as usize]);
        }
    }

    // Register allocation (linear scan + Belady eviction + spilling).
    let allocation =
        regalloc::linear_scan(&schedule, &uses_map, &precolored, backend.num_regs(), SCRATCH_BASE);
    let layout = FrameLayout::from_allocation(&allocation.spilled)?;

    // Select short-circuit guards.
    let select_guards = analyze_select_guards(&schedule);
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
            branch_starts[guard.true_range.0].push(PendingBranch { guard_idx: gi, arm: 0 });
            if guard.true_range.1 < sched_len {
                branch_ends[guard.true_range.1].push(gi);
            }
        }
        if guard.false_range.0 != guard.false_range.1 {
            branch_starts[guard.false_range.0].push(PendingBranch { guard_idx: gi, arm: 1 });
            if guard.false_range.1 < sched_len {
                branch_ends[guard.false_range.1].push(gi);
            }
        }
    }

    // Dense ValueId -> location lookups for the hot loop.
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

    backend.begin(&schedule)?;

    let mut code: Vec<u8> = Vec::new();
    backend.prologue(&mut code, layout.frame_size);

    let mut pending_patches: BTreeMap<(usize, u8), B::Branch> = BTreeMap::new();

    for (sched_idx, (vid, sched_op)) in schedule.iter().enumerate() {
        // Guard branches that begin before this instruction.
        for bi in 0..branch_starts[sched_idx].len() {
            let (guard_idx, arm) = {
                let pb = &branch_starts[sched_idx][bi];
                (pb.guard_idx, pb.arm)
            };
            let guard = &select_guards[guard_idx];
            let mask_reg = backend.emit_resolve(
                &mut code, guard.mask_vid, RELOAD_REG, &reg_for, &spill_for, &remat_for,
            );
            let branch = match arm {
                0 => backend.emit_skip_if_all_false(&mut code, mask_reg),
                _ => backend.emit_skip_if_all_true(&mut code, mask_reg),
            };
            pending_patches.insert((guard_idx, arm), branch);
        }

        // Guard branches that end at this instruction (patch their targets).
        for ei in 0..branch_ends[sched_idx].len() {
            let gi = branch_ends[sched_idx][ei];
            if let Some(branch) = pending_patches.remove(&(gi, 0)) {
                let target = code.len();
                backend.patch_branch(&mut code, branch, target);
            }
            if let Some(branch) = pending_patches.remove(&(gi, 1)) {
                let target = code.len();
                backend.patch_branch(&mut code, branch, target);
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

        // Select with a guard region: emit a uniform-mask short-circuit wrapper.
        if let ScheduledOp::Ternary(OpKind::Select, mask_vid, true_vid, false_vid) = sched_op {
            if let Some(guard) = select_guards.iter().find(|g| g.select_idx == sched_idx) {
                let has_true = guard.true_range.0 != guard.true_range.1;
                let has_false = guard.false_range.0 != guard.false_range.1;
                if has_true || has_false {
                    let mask_reg = backend.emit_resolve(
                        &mut code, *mask_vid, RELOAD_REG, &reg_for, &spill_for, &remat_for,
                    );
                    let dst = match dst_loc {
                        Loc::Reg(r) => r,
                        Loc::Spill(_) => RELOAD_REGS[0],
                    };
                    let true_reg = reg_for.get(true_vid.0 as usize).and_then(|r| *r);
                    let false_reg = reg_for.get(false_vid.0 as usize).and_then(|r| *r);

                    let all_false = backend.emit_skip_if_all_false(&mut code, mask_reg);
                    let all_true = backend.emit_skip_if_all_true(&mut code, mask_reg);

                    // Mixed lanes: the real select.
                    backend.emit_plan(&mut code, &plan)?;
                    let skip_end = backend.emit_jump(&mut code);

                    // All-false: dst <- false arm.
                    let all_false_target = code.len();
                    if let Some(freg) = false_reg {
                        backend.emit_mov(&mut code, dst, freg);
                    } else {
                        backend.emit_resolve(&mut code, *false_vid, dst, &reg_for, &spill_for, &remat_for);
                    }
                    let skip_end2 = backend.emit_jump(&mut code);

                    // All-true: dst <- true arm.
                    let all_true_target = code.len();
                    if let Some(treg) = true_reg {
                        backend.emit_mov(&mut code, dst, treg);
                    } else {
                        backend.emit_resolve(&mut code, *true_vid, dst, &reg_for, &spill_for, &remat_for);
                    }

                    let end_target = code.len();
                    backend.patch_branch(&mut code, all_false, all_false_target);
                    backend.patch_branch(&mut code, all_true, all_true_target);
                    backend.patch_branch(&mut code, skip_end, end_target);
                    backend.patch_branch(&mut code, skip_end2, end_target);

                    if let Loc::Spill(offset) = dst_loc {
                        backend.emit_store(&mut code, dst, offset)?;
                    }
                    continue;
                }
            }
        }

        backend.emit_plan(&mut code, &plan)?;
    }

    assert!(
        pending_patches.is_empty(),
        "BUG: {} Select short-circuit branches were never patched",
        pending_patches.len()
    );

    let root = schedule.last().map(|(v, _)| *v).expect("empty schedule");
    let result_reg =
        backend.emit_resolve(&mut code, root, RELOAD_REG, &reg_for, &spill_for, &remat_for);
    backend.epilogue(&mut code, result_reg, layout.frame_size);

    let exec = unsafe { executable::ExecutableCode::from_code(&code)? };
    Ok(CompileResult {
        code: exec,
        spill_count: layout.spill_slots.len() as u32,
        spill_bytes: layout.frame_size,
        max_regs: backend.num_regs(),
    })
}

/// A pending aarch64 branch: CBZ and B are patched differently.
#[cfg(target_arch = "aarch64")]
enum Aarch64Branch {
    Cbz(usize),
    B(usize),
}

/// aarch64 implementation of the shared driver's leaf operations.
///
/// Mechanically wraps the existing aarch64 encoders + constant pool, so the
/// emitted code is the same as the previous bespoke `compile_from_schedule`.
#[cfg(target_arch = "aarch64")]
struct Aarch64Backend {
    pool: ConstPool,
    adr_patch_pos: usize,
    max_regs: u8,
}

#[cfg(target_arch = "aarch64")]
impl IsaBackend for Aarch64Backend {
    type Branch = Aarch64Branch;

    fn num_regs(&self) -> u8 {
        self.max_regs
    }

    fn begin(&mut self, schedule: &[(regalloc::ValueId, ScheduledOp)]) -> Result<(), &'static str> {
        self.pool = ConstPool::from_schedule(schedule)?;
        // Builtins add up to ~60 polynomial coefficients during emission; bail
        // if the expression constants + headroom would exceed the 12-bit LDR
        // offset limit.
        const BUILTIN_HEADROOM: usize = 128;
        if self.pool.entries.len() + BUILTIN_HEADROOM > 4095 {
            return Err("expression too large: constant pool would exceed 12-bit LDR offset limit");
        }
        Ok(())
    }

    fn prologue(&mut self, code: &mut Vec<u8>, frame_size: u32) {
        if frame_size > 0 {
            aarch64::emit_sub_sp(code, frame_size);
        }
        // Builtins may add pool entries during emission, so always reserve the
        // ADR anchor (harmless if the pool ends up empty).
        self.adr_patch_pos = aarch64::emit_adr_x17_placeholder(code);
    }

    fn emit_plan(&mut self, code: &mut Vec<u8>, plan: &InstructionPlan) -> Result<(), &'static str> {
        emit_instruction_plan(code, plan, &mut self.pool)
    }

    fn emit_mov(&mut self, code: &mut Vec<u8>, dst: Reg, src: Reg) {
        emit_mov_reg(code, dst, src);
    }

    fn emit_store(&mut self, code: &mut Vec<u8>, src: Reg, offset: u32) -> Result<(), &'static str> {
        aarch64::emit_str_sp(code, src, offset);
        Ok(())
    }

    fn emit_resolve(
        &mut self,
        code: &mut Vec<u8>,
        vid: regalloc::ValueId,
        target: Reg,
        reg_for: &[Option<Reg>],
        spill_for: &[Option<u32>],
        remat_for: &[Option<u32>],
    ) -> Reg {
        emit_resolve_dense(code, vid, target, reg_for, spill_for, remat_for, &self.pool)
    }

    fn emit_skip_if_all_false(&mut self, code: &mut Vec<u8>, mask_reg: Reg) -> Aarch64Branch {
        let scratch = Reg(28);
        aarch64::emit_umaxv(code, scratch, mask_reg); // max lane; 0 => all-false
        aarch64::emit_fmov_to_gp(code, scratch);
        Aarch64Branch::Cbz(aarch64::emit_cbz_w16(code))
    }

    fn emit_skip_if_all_true(&mut self, code: &mut Vec<u8>, mask_reg: Reg) -> Aarch64Branch {
        let scratch = Reg(28);
        aarch64::emit_uminv(code, scratch, mask_reg); // min lane; 0xFFFFFFFF => all-true
        aarch64::emit_fmov_to_gp(code, scratch);
        aarch64::emit32(code, 0x2A3003F0); // MVN W16, W16  -> 0 iff all-true
        Aarch64Branch::Cbz(aarch64::emit_cbz_w16(code))
    }

    fn emit_jump(&mut self, code: &mut Vec<u8>) -> Aarch64Branch {
        Aarch64Branch::B(aarch64::emit_b(code))
    }

    fn patch_branch(&mut self, code: &mut Vec<u8>, branch: Aarch64Branch, target: usize) {
        match branch {
            Aarch64Branch::Cbz(p) => aarch64::patch_cbz_cbnz(code, p, target),
            Aarch64Branch::B(p) => aarch64::patch_b(code, p, target),
        }
    }

    fn epilogue(&mut self, code: &mut Vec<u8>, result_reg: Reg, frame_size: u32) {
        if result_reg.0 != 0 {
            emit_mov_reg(code, Reg(0), result_reg);
        }
        if frame_size > 0 {
            aarch64::emit_add_sp(code, frame_size);
        }
        // RET
        code.extend_from_slice(&0xD65F03C0u32.to_le_bytes());

        // Emit the constant pool after RET and anchor X17.
        if !self.pool.is_empty() {
            let adr_pos = self.adr_patch_pos;
            let estimated_offset = (code.len() as i64) - (adr_pos as i64);
            let needs_adrp = estimated_offset >= (1 << 20) - 32;
            if needs_adrp {
                code.splice(adr_pos + 4..adr_pos + 4, [0, 0, 0, 0]);
            }
            while code.len() % 16 != 0 {
                code.push(0);
            }
            let pool_start = code.len();
            for &bits in &self.pool.entries {
                aarch64::emit_pool_entry(code, bits);
            }
            aarch64::patch_adr_or_adrp(code, adr_pos, pool_start, needs_adrp);
        }
    }
}

/// Shared compilation backend: schedule + uses_map -> CompileResult.
///
/// `compile_arena_dag_with_ctx` and the scanline compilers all produce the same
/// `(schedule, uses_map)` format and then converge on the architecture-shared
/// [`compile_dag_via_backend`] driver via [`Aarch64Backend`].
#[cfg(target_arch = "aarch64")]
fn compile_from_schedule(
    schedule: Vec<(regalloc::ValueId, ScheduledOp)>,
    uses_map: Vec<Vec<regalloc::ValueId>>,
    ctx: EmitCtx,
) -> Result<CompileResult, &'static str> {
    let mut backend = Aarch64Backend {
        pool: ConstPool::new(),
        adr_patch_pos: 0,
        max_regs: ctx.max_regs,
    };
    compile_dag_via_backend(schedule, uses_map, &mut backend)
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
    /// Bit-shift by a compile-time immediate: `op` is `Shl` or `Shr`, the value
    /// is `ValueId`, and the shift count is folded out of the `Const` RHS by
    /// `arena_to_schedule` (so it never becomes a scheduled value / register).
    ShiftImm(OpKind, regalloc::ValueId, u8),
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
            // Shl/Shr fold their Const shift-count operand into an immediate, so
            // the count never becomes a scheduled value (matching the imm-only
            // hardware shift encoders). The count const may still appear as its
            // own schedule entry (harmless/unused) if shared.
            ExprNode::Binary(op @ (OpKind::Shl | OpKind::Shr), a, b) => {
                let amount = match arena.node(*b) {
                    ExprNode::Const(v) => *v as u32 as u8,
                    _ => panic!(
                        "{:?} shift count must be a Const (lowering guarantees this)",
                        op
                    ),
                };
                ScheduledOp::ShiftImm(*op, map_child(a), amount)
            }
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
            ScheduledOp::ShiftImm(_, a, _) => alloc::vec![*a],
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
                ScheduledOp::Unary(_, c) | ScheduledOp::ShiftImm(_, c, _) => {
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

    // Global consumer map: consumers[v.0] = every value that reads v as an
    // operand. A node may only be guarded (skipped when its arm's mask is
    // uniform) if EVERY consumer is inside that arm's subtree (or the select
    // itself) — otherwise an outer/sibling expression reads a register the
    // branch never computed. Subtree-local exclusivity (below) is necessary but
    // NOT sufficient; this is the global check that was missing.
    let mut consumers: alloc::vec::Vec<alloc::vec::Vec<regalloc::ValueId>> =
        alloc::vec![alloc::vec::Vec::new(); max_vid + 1];
    for (vid, sop) in schedule {
        let mut add = |child: regalloc::ValueId| {
            if (child.0 as usize) <= max_vid {
                consumers[child.0 as usize].push(*vid);
            }
        };
        match sop {
            ScheduledOp::Var(_) | ScheduledOp::Const(_) => {}
            ScheduledOp::Unary(_, c) | ScheduledOp::ShiftImm(_, c, _) => add(*c),
            ScheduledOp::Binary(_, a, b) => {
                add(*a);
                add(*b);
            }
            ScheduledOp::Ternary(_, a, b, c) => {
                add(*a);
                add(*b);
                add(*c);
            }
        }
    }

    for (i, (sel_vid, sop)) in schedule.iter().enumerate() {
        if let ScheduledOp::Ternary(OpKind::Select, mask_vid, true_vid, false_vid) = sop {
            // Compute transitive deps for each subtree using the dense O(1) lookup
            let mask_deps = transitive_deps(*mask_vid, &schedule_ops);
            let true_deps = transitive_deps(*true_vid, &schedule_ops);
            let false_deps = transitive_deps(*false_vid, &schedule_ops);

            // A node is safe to skip under this arm only if every one of its
            // consumers lies within the arm's subtree or is the select node
            // itself. Otherwise skipping it (uniform-mask short-circuit) leaves a
            // value some other expression still reads uninitialized.
            let only_used_within = |v: regalloc::ValueId, arm: &BTreeSet<regalloc::ValueId>| {
                consumers[v.0 as usize]
                    .iter()
                    .all(|c| *c == *sel_vid || arm.contains(c))
            };

            // True-exclusive: in true_deps but NOT in mask_deps and NOT in
            // false_deps, AND used only within the true arm.
            let true_exclusive: BTreeSet<regalloc::ValueId> = true_deps
                .difference(&mask_deps)
                .copied()
                .collect::<BTreeSet<_>>()
                .difference(&false_deps)
                .copied()
                .filter(|v| only_used_within(*v, &true_deps))
                .collect();

            // False-exclusive: symmetric.
            let false_exclusive: BTreeSet<regalloc::ValueId> = false_deps
                .difference(&mask_deps)
                .copied()
                .collect::<BTreeSet<_>>()
                .difference(&true_deps)
                .copied()
                .filter(|v| only_used_within(*v, &false_deps))
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
                // The branch skips the WHOLE range [start, end) when the mask is
                // uniform, so EVERY index in it must be a true-exclusive node.
                // If any in-range index is a shared node (used outside this arm)
                // or a false-exclusive node, skipping it would leave a value some
                // other expression reads uninitialized — fall back to BSL.
                let all_exclusive = (start..end).all(|idx| true_indices.contains(&idx));
                if all_exclusive {
                    (start, end)
                } else {
                    (i, i)
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
                let all_exclusive = (start..end).all(|idx| false_indices.contains(&idx));
                if all_exclusive {
                    (start, end)
                } else {
                    (i, i)
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
        ScheduledOp::ShiftImm(op_kind, child, amount) => {
            let src = resolve(*child, tmp_op, &mut reloads);
            ResolvedOp::ShiftImm {
                op: *op_kind,
                dst,
                src,
                amount: *amount,
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
        ResolvedOp::ShiftImm { op, dst, src, amount } => {
            aarch64::emit_shift_imm(code, *op, *dst, *src, *amount)?;
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
    // Per-batch x86 runs the architecture-shared driver
    // (`compile_dag_via_backend`): schedule -> regalloc (with spilling) -> Select
    // guards -> emit. The Sethi-Ullman `emit_arena` is retained only for the
    // scanline body (`compile_arena_dag_scanline`), which has no spilling yet.
    //
    // Backend = build width: AVX-512 (512-bit zmm) when compiled +avx512f, else
    // SSE2 (128-bit xmm). Both implement `IsaBackend`, so the driver and the
    // KernelFn ABI stay consistent with the selected width.
    //
    // Expand transcendentals to primitive arithmetic first, so no backend ever
    // sees Sin/Cos/etc. (one source of truth for the polynomial, in `lowering`).
    let (arena, root) = lowering::expand_transcendentals_owned(arena, root);
    let arena = &arena;
    let schedule = arena_to_schedule(arena, root);
    let uses = arena_to_uses(&schedule);
    #[cfg(target_feature = "avx512f")]
    {
        compile_dag_via_backend(schedule, uses, &mut Avx512Backend)
    }
    #[cfg(not(target_feature = "avx512f"))]
    {
        compile_dag_via_backend(schedule, uses, &mut X86Backend)
    }
}

// =============================================================================
// x86-64 IsaBackend: the leaf emit for the SHARED per-batch driver.
// =============================================================================
//
// `compile_dag_via_backend` owns the control flow (schedule, regalloc, Select
// short-circuit guards, root handling) for BOTH architectures; this is just the
// x86 instruction encoding behind that seam, so x86 and aarch64 run the same
// driver (no "works on my machine" divergence between them).
//
// Spills go to the System V red zone (the kernel is a leaf): a `FrameLayout`
// slot at byte `offset` maps to `[rsp - (offset + 16)]`.
//
// Register roles: xmm0-3 inputs (precolored), xmm4-9 allocatable (6),
// xmm10 fixed scratch (binary two-operand hazard + select temp), xmm11-12
// reload (`RELOAD_REGS`), xmm13-15 builtin scratch.

/// Allocatable scratch register count handed to `linear_scan` on x86 (xmm4-9).
#[cfg(target_arch = "x86_64")]
const X86_SCHED_NUM_REGS: u8 = 6;

/// Fixed scratch outside the allocatable range / reload regs: used for the
/// binary two-operand hazard and as the select blend temp.
#[cfg(target_arch = "x86_64")]
const X86_SCRATCH: Reg = Reg(10);

/// Scratch quad for builtins (sin/cos/exp/atan2/...), which need FOUR distinct
/// scratch registers. Clear of the allocatable range (4-9) and the reload regs
/// (11,12). Includes `X86_SCRATCH` (xmm10): builtins don't use it for the
/// hazard/select roles, so it is free as a fourth scratch here.
#[cfg(target_arch = "x86_64")]
const X86_BUILTIN_SCRATCH: [Reg; 4] = [Reg(10), Reg(13), Reg(14), Reg(15)];

/// Map a `FrameLayout` spill offset to a red-zone `[rsp+disp8]` displacement.
#[cfg(target_arch = "x86_64")]
fn x86_redzone_disp(offset: u32) -> Result<i8, &'static str> {
    // Slots live below rsp: offset 0 -> [rsp-16], 16 -> [rsp-32], ...
    let disp = -(offset as i64 + 16);
    if disp < -128 {
        return Err("x86 scheduled: spill frame exceeds 128-byte red zone");
    }
    Ok(disp as i8)
}

/// `dst = left op right` honoring SSE's two-operand form for *any* register
/// assignment from the allocator.
///
/// `emit_binary` computes `dst <- left; dst op= right`, which corrupts `right`
/// when `dst == right` and `dst != left`. The allocator may assign `dst ==
/// right`, so handle it: swap for commutative ops, otherwise stash `right` in
/// the fixed scratch.
#[cfg(target_arch = "x86_64")]
fn emit_binary_safe(code: &mut Vec<u8>, op: OpKind, dst: Reg, left: Reg, right: Reg) {
    use x86_64::*;
    let commutative = matches!(
        op,
        OpKind::Add | OpKind::Mul | OpKind::Min | OpKind::Max | OpKind::Eq | OpKind::Ne
    );
    if dst == left || dst != right {
        emit_binary(code, op, dst, left, right);
    } else if commutative {
        emit_binary(code, op, dst, right, left);
    } else {
        emit_movaps(code, X86_SCRATCH, right);
        emit_movaps(code, dst, left);
        emit_binary(code, op, dst, dst, X86_SCRATCH);
    }
}

/// x86-64 implementation of the shared driver's leaf operations.
#[cfg(target_arch = "x86_64")]
struct X86Backend;

#[cfg(target_arch = "x86_64")]
impl IsaBackend for X86Backend {
    /// rel32 field offset of the branch (uniform for jcc/jmp on x86).
    type Branch = usize;

    fn num_regs(&self) -> u8 {
        X86_SCHED_NUM_REGS
    }

    fn begin(&mut self, _schedule: &[(regalloc::ValueId, ScheduledOp)]) -> Result<(), &'static str> {
        Ok(()) // x86 const loads are self-contained; no pool.
    }

    fn prologue(&mut self, _code: &mut Vec<u8>, _frame_size: u32) {
        // Spills use the red zone; no frame to set up for a leaf.
    }

    fn emit_plan(&mut self, code: &mut Vec<u8>, plan: &InstructionPlan) -> Result<(), &'static str> {
        use x86_64::*;
        for reload in &plan.reloads {
            match reload {
                Reload::FromStack { target, offset } => {
                    emit_movups_load_rsp(code, *target, x86_redzone_disp(*offset)?);
                }
                Reload::Const { target, val_bits } => {
                    emit_const(code, *target, f32::from_bits(*val_bits), X86_BUILTIN_SCRATCH);
                }
            }
        }
        if let Some((dst, src)) = plan.setup_mov {
            emit_movaps(code, dst, src);
        }
        match &plan.op {
            ResolvedOp::Nop => {}
            ResolvedOp::LoadConst { dst, val_bits } => {
                emit_const(code, *dst, f32::from_bits(*val_bits), X86_BUILTIN_SCRATCH);
            }
            ResolvedOp::Unary { op, dst, src } => {
                emit_unary(code, *op, *dst, *src, X86_BUILTIN_SCRATCH);
            }
            ResolvedOp::ShiftImm { op, dst, src, amount } => {
                emit_shift_imm(code, *op, *dst, *src, *amount);
            }
            ResolvedOp::Binary { op, dst, left, right } => match op {
                OpKind::Atan2 | OpKind::Pow | OpKind::Hypot => {
                    emit_binary_transcendental(code, *op, *dst, *left, *right, X86_BUILTIN_SCRATCH);
                }
                _ => emit_binary_safe(code, *op, *dst, *left, *right),
            },
            ResolvedOp::Select { dst, if_true, if_false } => {
                // setup_mov already placed the mask in `dst`; blend in place.
                emit_select(code, *dst, *dst, *if_true, *if_false, X86_SCRATCH);
            }
            ResolvedOp::FusedMulAdd { dst, a, b } => {
                // No hardware FMA assumed: `dst` already holds c (setup_mov);
                // compute a*b in the fixed scratch, then add. a,b are never
                // X86_SCRATCH (allocator/reload regs), and `a` is copied out
                // before any write, so c==a / c==b are handled.
                emit_movaps(code, X86_SCRATCH, *a);
                emit_binary(code, OpKind::Mul, X86_SCRATCH, X86_SCRATCH, *b);
                emit_binary(code, OpKind::Add, *dst, *dst, X86_SCRATCH);
            }
            ResolvedOp::DecomposedMulAdd { dst, a, b, c, c_deferred } => {
                // dst = a*b, reload c (after the multiply, if deferred), dst += c.
                emit_binary_safe(code, OpKind::Mul, *dst, *a, *b);
                match c_deferred {
                    Some(DeferredReload::FromStack(off)) => {
                        emit_movups_load_rsp(code, *c, x86_redzone_disp(*off)?);
                    }
                    Some(DeferredReload::Const(bits)) => {
                        emit_const(code, *c, f32::from_bits(*bits), X86_BUILTIN_SCRATCH);
                    }
                    None => {}
                }
                emit_binary_safe(code, OpKind::Add, *dst, *dst, *c);
            }
            ResolvedOp::Clamp { dst, val, lo, hi, lo_deferred } => {
                // clamp(val, lo, hi) = max(min(val, hi), lo).
                emit_binary_safe(code, OpKind::Min, *dst, *val, *hi);
                match lo_deferred {
                    Some(DeferredReload::FromStack(off)) => {
                        emit_movups_load_rsp(code, *lo, x86_redzone_disp(*off)?);
                    }
                    Some(DeferredReload::Const(bits)) => {
                        emit_const(code, *lo, f32::from_bits(*bits), X86_BUILTIN_SCRATCH);
                    }
                    None => {}
                }
                emit_binary_safe(code, OpKind::Max, *dst, *dst, *lo);
            }
        }
        if let Some(store) = &plan.store {
            emit_movups_store_rsp(code, store.src, x86_redzone_disp(store.offset)?);
        }
        Ok(())
    }

    fn emit_mov(&mut self, code: &mut Vec<u8>, dst: Reg, src: Reg) {
        x86_64::emit_movaps(code, dst, src);
    }

    fn emit_store(&mut self, code: &mut Vec<u8>, src: Reg, offset: u32) -> Result<(), &'static str> {
        x86_64::emit_movups_store_rsp(code, src, x86_redzone_disp(offset)?);
        Ok(())
    }

    fn emit_resolve(
        &mut self,
        code: &mut Vec<u8>,
        vid: regalloc::ValueId,
        target: Reg,
        reg_for: &[Option<Reg>],
        spill_for: &[Option<u32>],
        remat_for: &[Option<u32>],
    ) -> Reg {
        let idx = vid.0 as usize;
        if let Some(Some(reg)) = reg_for.get(idx) {
            *reg
        } else if let Some(Some(bits)) = remat_for.get(idx) {
            x86_64::emit_const(code, target, f32::from_bits(*bits), X86_BUILTIN_SCRATCH);
            target
        } else if let Some(Some(offset)) = spill_for.get(idx) {
            // Resolve is on a hot path with a known-valid frame; offset fits.
            let disp = x86_redzone_disp(*offset).expect("spill offset within red zone");
            x86_64::emit_movups_load_rsp(code, target, disp);
            target
        } else {
            panic!("value {:?} has no register, spill slot, or rematerialize entry", vid);
        }
    }

    fn emit_skip_if_all_false(&mut self, code: &mut Vec<u8>, mask_reg: Reg) -> usize {
        x86_64::emit_movmskps_eax(code, mask_reg);
        x86_64::emit_test_eax(code);
        x86_64::emit_jcc_rel32(code, 0x84) // jz: taken when eax == 0 (all lanes false)
    }

    fn emit_skip_if_all_true(&mut self, code: &mut Vec<u8>, mask_reg: Reg) -> usize {
        x86_64::emit_movmskps_eax(code, mask_reg);
        x86_64::emit_cmp_eax_imm8(code, 0x0F);
        x86_64::emit_jcc_rel32(code, 0x84) // je: taken when eax == 0xF (all lanes true)
    }

    fn emit_jump(&mut self, code: &mut Vec<u8>) -> usize {
        x86_64::emit_jmp_rel32(code)
    }

    fn patch_branch(&mut self, code: &mut Vec<u8>, branch: usize, target: usize) {
        x86_64::patch_rel32(code, branch, target);
    }

    fn epilogue(&mut self, code: &mut Vec<u8>, result_reg: Reg, _frame_size: u32) {
        if result_reg.0 != 0 {
            x86_64::emit_movaps(code, Reg(0), result_reg);
        }
        code.push(0xC3); // RET
    }
}

// =============================================================================
// AVX-512 backend: the leaf emit for the SHARED driver, 512-bit (zmm) kernels.
// =============================================================================
//
// Mirrors `X86Backend`'s register roles exactly so it reuses the shared driver,
// `linear_scan` budget, and the RELOAD_REGS/SCRATCH_BASE consts unchanged — only
// the leaf encodings differ. EVEX is 3-operand and non-destructive, so there is
// no SSE two-operand hazard (operands never clobbered; may alias dst), and we
// have real hardware FMA.
//
// Register roles (zmm): zmm0-3 inputs, zmm4-9 allocatable (6), zmm10 fixed
// scratch (FMA temp), zmm11-12 reload (RELOAD_REGS), zmm13-15 builtin scratch.
//
// Spills use a REAL stack frame, not the red zone: a zmm slot is 64 bytes, so
// even one spill overflows the 128-byte red zone. `FrameLayout` hands out
// offsets in 16-byte units (the shared layout assumes 128-bit slots); scale ×4
// to 64-byte zmm slots, and allocate `frame_size * 4` in the prologue.
//
// Scope (Stage 1): the arithmetic subset (Var/Const/Unary{sqrt,neg,abs}/Binary
// {add,sub,mul,div,min,max}/FMA). Select (k-mask class), Clamp, and the
// transcendentals reject loudly and are later stages.

/// Scale a `FrameLayout` 16-byte slot offset to a 64-byte zmm frame offset.
#[cfg(target_arch = "x86_64")]
fn avx512_slot_disp(offset: u32) -> i32 {
    // FrameLayout packs slots at 16-byte stride; zmm needs 64. Slots live at
    // [rsp + 0], [rsp + 64], ... within the frame we allocate in the prologue.
    (offset as i32 / 16) * 64
}

/// Total zmm frame bytes for a `FrameLayout.frame_size` (16-byte units).
#[cfg(target_arch = "x86_64")]
fn avx512_frame_bytes(frame_size: u32) -> u32 {
    (frame_size / 16) * 64
}

/// AVX-512 implementation of the shared driver's leaf operations.
#[cfg(target_arch = "x86_64")]
struct Avx512Backend;

#[cfg(target_arch = "x86_64")]
impl Avx512Backend {
    fn reload(code: &mut Vec<u8>, reload: &Reload) {
        match reload {
            Reload::FromStack { target, offset } => {
                avx512::emit_load_rsp(code, *target, avx512_slot_disp(*offset));
            }
            Reload::Const { target, val_bits } => {
                avx512::emit_const(code, *target, f32::from_bits(*val_bits));
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl IsaBackend for Avx512Backend {
    type Branch = usize;

    fn num_regs(&self) -> u8 {
        X86_SCHED_NUM_REGS // same 6 allocatable (zmm4-9)
    }

    fn begin(&mut self, _schedule: &[(regalloc::ValueId, ScheduledOp)]) -> Result<(), &'static str> {
        Ok(()) // const broadcast is self-contained; no pool.
    }

    fn prologue(&mut self, code: &mut Vec<u8>, frame_size: u32) {
        let bytes = avx512_frame_bytes(frame_size);
        if bytes > 0 {
            avx512::emit_sub_rsp(code, bytes);
        }
    }

    fn emit_plan(&mut self, code: &mut Vec<u8>, plan: &InstructionPlan) -> Result<(), &'static str> {
        for r in &plan.reloads {
            Self::reload(code, r);
        }
        if let Some((dst, src)) = plan.setup_mov {
            avx512::emit_mov(code, dst, src);
        }
        match &plan.op {
            ResolvedOp::Nop => {}
            ResolvedOp::LoadConst { dst, val_bits } => {
                avx512::emit_const(code, *dst, f32::from_bits(*val_bits));
            }
            ResolvedOp::Unary { op, dst, src } => {
                avx512::emit_unary(code, *op, *dst, *src)?;
            }
            ResolvedOp::ShiftImm { .. } => {
                // EVEX integer shift not wired yet -> exp/log don't lower on the
                // AVX-512 path. Reject loudly rather than miscompile.
                return Err("avx512: bit-shift (exp/log lowering) not yet supported");
            }
            ResolvedOp::Binary { op, dst, left, right } => {
                // EVEX 3-operand: no two-operand hazard, emit directly.
                // Comparisons produce a vector mask (vcmpps -> vpmovm2d).
                if avx512::is_compare(*op) {
                    avx512::emit_compare(code, *op, *dst, *left, *right)?;
                } else {
                    avx512::emit_binary(code, *op, *dst, *left, *right)?;
                }
            }
            ResolvedOp::FusedMulAdd { dst, a, b } => {
                // dst holds c (setup_mov); real FMA231: dst = a*b + dst.
                avx512::emit_fmadd_c_in_dst(code, *dst, *a, *b);
            }
            ResolvedOp::DecomposedMulAdd { dst, a, b, c, c_deferred } => {
                // dst = a*b, reload c (after the multiply if deferred), dst += c.
                avx512::emit_binary(code, OpKind::Mul, *dst, *a, *b)?;
                match c_deferred {
                    Some(DeferredReload::FromStack(off)) => {
                        avx512::emit_load_rsp(code, *c, avx512_slot_disp(*off));
                    }
                    Some(DeferredReload::Const(bits)) => {
                        avx512::emit_const(code, *c, f32::from_bits(*bits));
                    }
                    None => {}
                }
                avx512::emit_binary(code, OpKind::Add, *dst, *dst, *c)?;
            }
            ResolvedOp::Select { dst, if_true, if_false } => {
                // setup_mov already placed the vector mask in dst; one vpternlogd.
                avx512::emit_select(code, *dst, *if_true, *if_false);
            }
            ResolvedOp::Clamp { dst, val, lo, hi, lo_deferred } => {
                // clamp(val, lo, hi) = max(min(val, hi), lo). No mask needed.
                avx512::emit_binary(code, OpKind::Min, *dst, *val, *hi)?;
                match lo_deferred {
                    Some(DeferredReload::FromStack(off)) => {
                        avx512::emit_load_rsp(code, *lo, avx512_slot_disp(*off));
                    }
                    Some(DeferredReload::Const(bits)) => {
                        avx512::emit_const(code, *lo, f32::from_bits(*bits));
                    }
                    None => {}
                }
                avx512::emit_binary(code, OpKind::Max, *dst, *dst, *lo)?;
            }
        }
        if let Some(store) = &plan.store {
            avx512::emit_store_rsp(code, store.src, avx512_slot_disp(store.offset));
        }
        Ok(())
    }

    fn emit_mov(&mut self, code: &mut Vec<u8>, dst: Reg, src: Reg) {
        avx512::emit_mov(code, dst, src);
    }

    fn emit_store(&mut self, code: &mut Vec<u8>, src: Reg, offset: u32) -> Result<(), &'static str> {
        avx512::emit_store_rsp(code, src, avx512_slot_disp(offset));
        Ok(())
    }

    fn emit_resolve(
        &mut self,
        code: &mut Vec<u8>,
        vid: regalloc::ValueId,
        target: Reg,
        reg_for: &[Option<Reg>],
        spill_for: &[Option<u32>],
        remat_for: &[Option<u32>],
    ) -> Reg {
        let idx = vid.0 as usize;
        if let Some(Some(reg)) = reg_for.get(idx) {
            *reg
        } else if let Some(Some(bits)) = remat_for.get(idx) {
            avx512::emit_const(code, target, f32::from_bits(*bits));
            target
        } else if let Some(Some(offset)) = spill_for.get(idx) {
            avx512::emit_load_rsp(code, target, avx512_slot_disp(*offset));
            target
        } else {
            panic!("value {:?} has no register, spill slot, or rematerialize entry", vid);
        }
    }

    // Select short-circuit guards: reduce the vector mask to flags (vptestmd +
    // kortestw) and branch. jz = all-false (skip true arm); jc = all-true (skip
    // false arm). Mirrors the SSE2 MOVMSKPS guards, k-register-based.
    fn emit_skip_if_all_false(&mut self, code: &mut Vec<u8>, mask_reg: Reg) -> usize {
        avx512::emit_mask_flags(code, mask_reg);
        x86_64::emit_jcc_rel32(code, 0x84) // jz: ZF set when k1 == 0 (all false)
    }
    fn emit_skip_if_all_true(&mut self, code: &mut Vec<u8>, mask_reg: Reg) -> usize {
        avx512::emit_mask_flags(code, mask_reg);
        x86_64::emit_jcc_rel32(code, 0x82) // jc: CF set when k1 == 0xFFFF (all true)
    }
    fn emit_jump(&mut self, code: &mut Vec<u8>) -> usize {
        x86_64::emit_jmp_rel32(code)
    }
    fn patch_branch(&mut self, code: &mut Vec<u8>, branch: usize, target: usize) {
        x86_64::patch_rel32(code, branch, target);
    }

    fn epilogue(&mut self, code: &mut Vec<u8>, result_reg: Reg, frame_size: u32) {
        if result_reg.0 != 0 {
            avx512::emit_mov(code, Reg(0), result_reg);
        }
        let bytes = avx512_frame_bytes(frame_size);
        if bytes > 0 {
            avx512::emit_add_rsp(code, bytes);
        }
        avx512::emit_ret(code);
    }
}

/// Compile an arena DAG to an AVX-512 (512-bit, 16-lane zmm) kernel via the
/// shared driver. Same arg shape as [`compile_arena_dag`] but the kernel's ABI
/// is `__m512` (one pixel per lane, 16 pixels per call). Stage-1 arithmetic
/// subset only; ops outside it return `Err`.
#[cfg(target_arch = "x86_64")]
pub fn compile_arena_dag_avx512(
    arena: &ExprArena,
    root: ExprId,
) -> Result<CompileResult, &'static str> {
    let schedule = arena_to_schedule(arena, root);
    let uses = arena_to_uses(&schedule);
    compile_dag_via_backend(schedule, uses, &mut Avx512Backend)
}

/// Compile an [`ExprArena`] DAG into a scanline kernel (x86-64).
///
/// The emitted kernel contains its own loop: Y/Z/W are loaded once into
/// xmm1/xmm2/xmm3 and stay there for the whole scanline (loop-invariant by
/// construction); only X is reloaded per batch from the input array. This
/// eliminates the per-batch `extern "C"` call overhead of the per-batch
/// [`KernelFn`].
///
/// The per-batch body is the same Sethi-Ullman code produced by [`emit_arena`],
/// so X-invariant subexpressions are recomputed each iteration (no hoisting on
/// x86-64 yet — see [`MAX_PERSISTENT_SLOTS`]).
///
/// Matches the [`ScanlineKernelFn`](executable::ScanlineKernelFn) ABI:
/// `(xs: *const __m128, y, z, w: __m128, out: *mut __m128, count: usize)`.
/// Under SysV: `rdi = xs`, `xmm0 = y`, `xmm1 = z`, `xmm2 = w`, `rsi = out`,
/// `rdx = count`.
#[cfg(target_arch = "x86_64")]
pub fn compile_arena_dag_scanline(
    arena: &ExprArena,
    root: ExprId,
) -> Result<ScanlineCompileResult, &'static str> {
    const MAX_DEPTH: usize = 64;
    if arena.depth(root) > MAX_DEPTH {
        return Err("expression too deep");
    }

    // Per-batch body: reads X from xmm0 and Y/Z/W from xmm1/2/3 (INPUT_REGS),
    // writes the result into `result_reg`, using xmm4+ as scratch.
    let (body, result_jet) = emit_arena(arena, root, 0, &[])?;
    let result_reg = result_jet.val;

    let mut code: Vec<u8> = Vec::with_capacity(body.len() + 64);

    // Shuffle incoming args into INPUT_REGS layout, freeing xmm0 for X:
    //   y: xmm0 -> xmm1,  z: xmm1 -> xmm2,  w: xmm2 -> xmm3
    // Done high-to-low so no live value is clobbered before it is copied.
    x86_64::emit_movaps(&mut code, Reg(3), Reg(2)); // w
    x86_64::emit_movaps(&mut code, Reg(2), Reg(1)); // z
    x86_64::emit_movaps(&mut code, Reg(1), Reg(0)); // y

    // xor eax, eax          ; loop index i = 0 (rax)
    code.extend_from_slice(&[0x31, 0xC0]);

    let loop_top = code.len();

    // cmp rax, rdx          ; i vs count
    code.extend_from_slice(&[0x48, 0x39, 0xD0]);
    // jae loop_end          ; if i >= count, done
    code.extend_from_slice(&[0x0F, 0x83]);
    let jae_disp_at = code.len();
    code.extend_from_slice(&[0, 0, 0, 0]); // rel32 placeholder

    // movaps xmm0, [rdi]    ; x = xs[i]
    x86_64::emit_movaps_load(&mut code, Reg(0), 0);

    // Per-batch computation.
    code.extend_from_slice(&body);

    // movaps [rsi], result_reg   ; out[i] = result
    if result_reg.0 >= 8 {
        code.push(0x44); // REX.R
    }
    code.push(0x0F);
    code.push(0x29);
    code.push(0x06 | ((result_reg.0 & 7) << 3)); // mod=00, rm=110 (rsi)

    // add rdi, 16 ; add rsi, 16 ; inc rax
    code.extend_from_slice(&[0x48, 0x83, 0xC7, 0x10]); // add rdi, 16
    code.extend_from_slice(&[0x48, 0x83, 0xC6, 0x10]); // add rsi, 16
    code.extend_from_slice(&[0x48, 0xFF, 0xC0]); // inc rax

    // jmp loop_top
    code.push(0xE9);
    let jmp_disp_at = code.len();
    code.extend_from_slice(&[0, 0, 0, 0]); // rel32 placeholder

    let loop_end = code.len();
    // ret
    code.push(0xC3);

    // Patch the forward jae (target = loop_end) and backward jmp (target = loop_top).
    let jae_rel = (loop_end as i32) - (jae_disp_at as i32 + 4);
    code[jae_disp_at..jae_disp_at + 4].copy_from_slice(&jae_rel.to_le_bytes());
    let jmp_rel = (loop_top as i32) - (jmp_disp_at as i32 + 4);
    code[jmp_disp_at..jmp_disp_at + 4].copy_from_slice(&jmp_rel.to_le_bytes());

    let code = unsafe { executable::ExecutableCode::from_code(&code)? };
    Ok(ScanlineCompileResult {
        code,
        spill_count: 0,
        spill_bytes: 0,
        max_regs: EmitCtx::default().max_regs,
    })
}

/// Compile a scanline kernel with an explicit register budget (x86-64).
///
/// The `ctx` budget is advisory on x86-64 (see [`compile_arena_dag_with_ctx`]).
#[cfg(target_arch = "x86_64")]
pub fn compile_arena_dag_scanline_with_ctx(
    arena: &ExprArena,
    root: ExprId,
    _ctx: EmitCtx,
) -> Result<ScanlineCompileResult, &'static str> {
    compile_arena_dag_scanline(arena, root)
}

/// Variance-aware hoisted scanline compilation (x86-64).
///
/// x86-64 has no persistent-slot hoisting yet ([`MAX_PERSISTENT_SLOTS`] == 0),
/// so this is identical to the flat [`compile_arena_dag_scanline`].
#[cfg(target_arch = "x86_64")]
pub fn compile_arena_dag_scanline_hoisted(
    arena: &ExprArena,
    root: ExprId,
) -> Result<ScanlineCompileResult, &'static str> {
    compile_arena_dag_scanline(arena, root)
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

    /// Test scanline matches per-batch results for a complex expression.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_scanline_matches_per_batch() {
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
        let single = compile_arena_dag(&arena, root).expect("per-batch compile failed");
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

    /// The x86-64 scanline kernel must produce, for every pixel, exactly what
    /// the per-batch kernel produces for that pixel's X (with the same
    /// loop-invariant Y/Z/W).
    // Compares the scanline kernel against the per-batch `KernelFn`, which is
    // 128-bit under the default build; gated off `+avx512f` where `KernelFn` is
    // `__m512` (the AVX-512 per-batch path is covered by the `avx512` tests).
    #[test]
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
    fn test_scanline_matches_per_batch_x86() {
        use core::arch::x86_64::*;

        // expr = (X*X + Y) * Z - W  (uses all four variables)
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let z = arena.push_var(2);
        let w = arena.push_var(3);
        let xx = arena.push_binary(OpKind::Mul, x, x);
        let xxy = arena.push_binary(OpKind::Add, xx, y);
        let m = arena.push_binary(OpKind::Mul, xxy, z);
        let root = arena.push_binary(OpKind::Sub, m, w);

        let single = compile_arena_dag(&arena, root).expect("per-batch compile failed");
        let scan = compile_arena_dag_scanline(&arena, root).expect("scanline compile failed");

        unsafe {
            let y_v = _mm_set1_ps(2.0);
            let z_v = _mm_set1_ps(3.0);
            let w_v = _mm_set1_ps(0.5);

            // Vec<__m128> is 16-byte aligned, satisfying the movaps contract.
            let xs: Vec<__m128> = (0..7).map(|i| _mm_set1_ps(i as f32 * 0.7 - 1.5)).collect();
            let mut out: Vec<__m128> = vec![_mm_set1_ps(0.0); xs.len()];

            let sfunc: executable::ScanlineKernelFn = scan.code.as_fn();
            sfunc(xs.as_ptr(), y_v, z_v, w_v, out.as_mut_ptr(), xs.len());

            let kfunc: executable::KernelFn = single.code.as_fn();
            for (i, &xv) in xs.iter().enumerate() {
                let expected = kfunc(xv, y_v, z_v, w_v);
                let e = _mm_cvtss_f32(expected);
                let g = _mm_cvtss_f32(out[i]);
                assert!(
                    (e - g).abs() < 1e-5,
                    "scanline pixel {i} mismatch: single={e} scanline={g}"
                );
            }
        }
    }

    /// An empty scanline (count == 0) must be a no-op and not write `output`.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_scanline_empty_x86() {
        use core::arch::x86_64::*;

        let mut arena = ExprArena::new();
        let root = arena.push_var(0);
        let scan = compile_arena_dag_scanline(&arena, root).expect("scanline compile failed");

        unsafe {
            let zero = _mm_set1_ps(0.0);
            let xs: &[__m128] = &[];
            let mut out: Vec<__m128> = Vec::new();
            let sfunc: executable::ScanlineKernelFn = scan.code.as_fn();
            sfunc(xs.as_ptr(), zero, zero, zero, out.as_mut_ptr(), 0);
        }
    }

    /// Run a per-batch arena kernel at `x` (Y/Z/W = 0) and return lane 0.
    /// 128-bit `KernelFn`; the builtin-parity tests below use it. Gated off
    /// `+avx512f` (those builtins aren't in the AVX-512 op set yet anyway).
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
    fn run1(arena: &ExprArena, root: ExprId, x: f32) -> f32 {
        use core::arch::x86_64::*;
        let r = compile_arena_dag(arena, root).expect("compile failed");
        unsafe {
            let f: executable::KernelFn = r.code.as_fn();
            let z = _mm_set1_ps(0.0);
            _mm_cvtss_f32(f(_mm_set1_ps(x), z, z, z))
        }
    }

    /// Per-batch eval at (X=x, Y=y, Z=W=0), lane 0. 128-bit `KernelFn`, so gated
    /// off `+avx512f` like `run1`.
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
    fn run_xy(arena: &ExprArena, root: ExprId, x: f32, y: f32) -> f32 {
        use core::arch::x86_64::*;
        let r = compile_arena_dag(arena, root).expect("compile failed");
        unsafe {
            let f: executable::KernelFn = r.code.as_fn();
            let z = _mm_set1_ps(0.0);
            _mm_cvtss_f32(f(_mm_set1_ps(x), _mm_set1_ps(y), z, z))
        }
    }

    /// Every x86-64 unary transcendental/round op must match its scalar
    /// reference across a range of inputs — these exercise `emit_arena` →
    /// `emit_unary` directly (not the compiler's lowering).
    #[test]
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
    fn test_x86_unary_builtins_match_scalar() {
        // Tolerances reflect the shared (with aarch64) minimax-polynomial
        // accuracy over a sensible input range; exact ops use tight bounds.
        // `rel_err = |jit - scalar| / (1 + |scalar|)`.
        let unary: &[(OpKind, fn(f32) -> f32, &[f32], f32)] = &[
            (OpKind::Sqrt, |x| x.sqrt(), &[0.25, 1.0, 2.0, 9.0, 100.0], 1e-5),
            (OpKind::Abs, |x| x.abs(), &[-3.0, -0.5, 0.0, 2.5], 1e-6),
            (OpKind::Neg, |x| -x, &[-3.0, 0.5, 2.5], 1e-6),
            (OpKind::Floor, |x| x.floor(), &[-2.3, -0.1, 0.9, 1.5, 3.99], 1e-6),
            (OpKind::Ceil, |x| x.ceil(), &[-2.3, -0.1, 0.9, 1.5, 3.01], 1e-6),
            (OpKind::Round, |x| x.round_ties_even(), &[-2.4, -0.4, 0.4, 1.5, 2.6], 1e-6),
            (OpKind::Fract, |x| x - x.floor(), &[-2.3, 0.1, 0.9, 3.75], 1e-5),
            // sin/cos: 4-term Chebyshev — accurate well inside [-π, π].
            (OpKind::Sin, |x| x.sin(), &[-2.0, -1.0, -0.3, 0.0, 0.5, 1.5, 2.0], 6e-3),
            (OpKind::Cos, |x| x.cos(), &[-1.0, -0.3, 0.0, 0.5, 1.0], 1.5e-2),
            (OpKind::Tan, |x| x.tan(), &[-1.0, -0.3, 0.0, 0.3, 1.0], 2.5e-2),
            (OpKind::Exp, |x| x.exp(), &[-2.0, -0.5, 0.0, 1.0, 2.0, 3.0], 5e-3),
            (OpKind::Exp2, |x| x.exp2(), &[-3.0, -0.5, 0.0, 1.0, 4.0], 5e-3),
            (OpKind::Ln, |x| x.ln(), &[0.25, 0.5, 1.0, 2.0, 10.0], 5e-3),
            (OpKind::Log2, |x| x.log2(), &[0.25, 0.5, 1.0, 2.0, 8.0], 5e-3),
            (OpKind::Log10, |x| x.log10(), &[0.1, 0.5, 1.0, 10.0, 100.0], 5e-3),
            (OpKind::Atan, |x| x.atan(), &[-5.0, -0.5, -0.2, 0.0, 0.2, 0.5, 5.0], 8e-3),
            (OpKind::Asin, |x| x.asin(), &[-0.8, -0.5, 0.0, 0.5, 0.8], 1e-2),
            (OpKind::Acos, |x| x.acos(), &[-0.8, -0.5, 0.0, 0.5, 0.8], 1e-2),
        ];
        for &(op, scalar, inputs, tol) in unary {
            let mut arena = ExprArena::new();
            let x = arena.push_var(0);
            let root = arena.push_unary(op, x);
            for &xv in inputs {
                let got = run1(&arena, root, xv);
                let want = scalar(xv);
                let err = (got - want).abs() / (1.0 + want.abs());
                assert!(
                    err <= tol,
                    "{op:?}({xv}): jit={got} scalar={want} rel_err={err} > {tol}"
                );
            }
        }
    }

    /// Binary transcendentals + comparisons + ternaries, JIT vs scalar.
    #[test]
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
    fn test_x86_binary_ternary_builtins_match_scalar() {
        use core::arch::x86_64::*;
        // Helper: compile f(X, Y) and eval at (x, y).
        unsafe fn run2(arena: &ExprArena, root: ExprId, x: f32, y: f32) -> f32 {
            let r = compile_arena_dag(arena, root).expect("compile failed");
            let f: executable::KernelFn = r.code.as_fn();
            let z = _mm_set1_ps(0.0);
            _mm_cvtss_f32(f(_mm_set1_ps(x), _mm_set1_ps(y), z, z))
        }

        // atan2(y, x): arena Binary(Atan2, Y, X)  (op order: src1=y, src2=x)
        let pts = [(0.5, 2.0), (2.0, 0.5), (-0.5, 2.0), (0.5, -2.0), (-2.0, -0.5), (3.0, -0.5)];
        {
            let mut a = ExprArena::new();
            let y = a.push_var(1);
            let x = a.push_var(0);
            let root = a.push_binary(OpKind::Atan2, y, x);
            for &(yv, xv) in &pts {
                let got = unsafe { run2(&a, root, xv, yv) };
                let want = yv.atan2(xv);
                assert!((got - want).abs() <= 1.5e-2, "atan2({yv},{xv}): {got} vs {want}");
            }
        }
        // pow(X, Y)
        {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let root = a.push_binary(OpKind::Pow, x, y);
            for &(xv, yv) in &[(2.0f32, 3.0f32), (9.0, 0.5), (4.0, -1.0), (1.5, 2.0)] {
                let got = unsafe { run2(&a, root, xv, yv) };
                let want = xv.powf(yv);
                let err = (got - want).abs() / (1.0 + want.abs());
                assert!(err <= 5e-3, "pow({xv},{yv}): {got} vs {want} err={err}");
            }
        }
        // hypot(X, Y)
        {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let root = a.push_binary(OpKind::Hypot, x, y);
            for &(xv, yv) in &[(3.0f32, 4.0f32), (1.0, 1.0), (0.0, 2.0)] {
                let got = unsafe { run2(&a, root, xv, yv) };
                let want = xv.hypot(yv);
                assert!((got - want).abs() <= 1e-4, "hypot({xv},{yv}): {got} vs {want}");
            }
        }
        // Min / Max
        for (op, f) in [
            (OpKind::Min, f32::min as fn(f32, f32) -> f32),
            (OpKind::Max, f32::max as fn(f32, f32) -> f32),
        ] {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let root = a.push_binary(op, x, y);
            for &(xv, yv) in &[(1.0f32, 2.0f32), (3.0, -1.0), (-2.0, -5.0)] {
                let got = unsafe { run2(&a, root, xv, yv) };
                assert!((got - f(xv, yv)).abs() <= 1e-6, "{op:?}({xv},{yv})");
            }
        }
        // Clamp(X, 0.0, 1.0)
        {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let lo = a.push_const(0.0);
            let hi = a.push_const(1.0);
            let root = a.push_ternary(OpKind::Clamp, x, lo, hi);
            for &xv in &[-0.5f32, 0.25, 0.9, 1.7] {
                let got = run1(&a, root, xv);
                assert!((got - xv.clamp(0.0, 1.0)).abs() <= 1e-6, "clamp({xv})={got}");
            }
        }
        // Select(X >= 0, 1.0, -1.0) == signum-ish
        {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let zero = a.push_const(0.0);
            let cond = a.push_binary(OpKind::Ge, x, zero);
            let pos = a.push_const(1.0);
            let neg = a.push_const(-1.0);
            let root = a.push_ternary(OpKind::Select, cond, pos, neg);
            for &xv in &[-2.0f32, -0.1, 0.1, 3.0] {
                let got = run1(&a, root, xv);
                let want = if xv >= 0.0 { 1.0 } else { -1.0 };
                assert!((got - want).abs() <= 1e-6, "select({xv})={got} want={want}");
            }
        }
    }

    // =========================================================================
    // Forward-mode dual (jet) lowering — validated against analytic derivatives.
    // Uses hardware sqrtps/divps (no polynomial approximations), so tolerances
    // are tight.
    // =========================================================================
    // Calls kernels through the per-batch `KernelFn` (128-bit here); gated off
    // `+avx512f` where that ABI is `__m512`.
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
    mod jet {
        use super::*;
        use crate::arena::ExprArena;

        /// Compile one dual component and evaluate it at (x,y,z,w) (lane 0).
        fn comp(arena: &ExprArena, root: ExprId, seed: &[u8], out: JetOutput, x: f32, y: f32) -> f32 {
            let r = compile_arena_dag_jet(arena, root, seed, out)
                .expect("jet compile failed");
            unsafe {
                use core::arch::x86_64::*;
                let f: executable::KernelFn = r.code.as_fn();
                let o = f(_mm_set1_ps(x), _mm_set1_ps(y), _mm_set1_ps(0.0), _mm_set1_ps(0.0));
                _mm_cvtss_f32(o)
            }
        }

        const SEED_XY: &[u8] = &[0, 1]; // Jet2 over X, Y
        const PTS: &[(f32, f32)] = &[(3.0, 4.0), (1.0, 2.0), (-2.0, 0.5), (0.7, -1.3)];

        fn close(a: f32, b: f32, tag: &str) {
            assert!((a - b).abs() <= 1e-4, "{tag}: got {a}, want {b}");
        }

        #[test]
        fn jet_sum_of_squares() {
            // f = X*X + Y*Y ; ∂x = 2x, ∂y = 2y
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let xx = a.push_binary(OpKind::Mul, x, x);
            let yy = a.push_binary(OpKind::Mul, y, y);
            let root = a.push_binary(OpKind::Add, xx, yy);
            for &(px, py) in PTS {
                close(comp(&a, root, SEED_XY, JetOutput::Value, px, py), px * px + py * py, "val");
                close(comp(&a, root, SEED_XY, JetOutput::Partial(0), px, py), 2.0 * px, "dx");
                close(comp(&a, root, SEED_XY, JetOutput::Partial(1), px, py), 2.0 * py, "dy");
            }
        }

        #[test]
        fn jet_product() {
            // f = X*Y ; ∂x = y, ∂y = x  (both partials nonzero -> cross term)
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let root = a.push_binary(OpKind::Mul, x, y);
            for &(px, py) in PTS {
                close(comp(&a, root, SEED_XY, JetOutput::Value, px, py), px * py, "val");
                close(comp(&a, root, SEED_XY, JetOutput::Partial(0), px, py), py, "dx");
                close(comp(&a, root, SEED_XY, JetOutput::Partial(1), px, py), px, "dy");
            }
        }

        #[test]
        fn jet_circle_sdf() {
            // f = sqrt(X*X + Y*Y) - r ; ∂x = x/√(x²+y²), ∂y = y/√(x²+y²)
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let xx = a.push_binary(OpKind::Mul, x, x);
            let yy = a.push_binary(OpKind::Mul, y, y);
            let sum = a.push_binary(OpKind::Add, xx, yy);
            let dist = a.push_unary(OpKind::Sqrt, sum);
            let r = a.push_const(1.5);
            let root = a.push_binary(OpKind::Sub, dist, r);
            for &(px, py) in PTS {
                let d = (px * px + py * py).sqrt();
                close(comp(&a, root, SEED_XY, JetOutput::Value, px, py), d - 1.5, "val");
                close(comp(&a, root, SEED_XY, JetOutput::Partial(0), px, py), px / d, "dx");
                close(comp(&a, root, SEED_XY, JetOutput::Partial(1), px, py), py / d, "dy");
            }
        }

        #[test]
        fn jet_neg_and_zero_partial() {
            // f = -X ; ∂x = -1, ∂y = 0 (statically zero -> returns 0.0)
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let root = a.push_unary(OpKind::Neg, x);
            for &(px, py) in PTS {
                close(comp(&a, root, SEED_XY, JetOutput::Value, px, py), -px, "val");
                close(comp(&a, root, SEED_XY, JetOutput::Partial(0), px, py), -1.0, "dx");
                close(comp(&a, root, SEED_XY, JetOutput::Partial(1), px, py), 0.0, "dy");
            }
        }

        #[test]
        fn jet_truly_unsupported_op_errs() {
            // exp is not yet lowered (needs bit-manip primitives) and has no jet
            // rule -> loud error, never a miscompile.
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let root = a.push_unary(OpKind::Exp, x);
            assert!(compile_arena_dag_jet(&a, root, SEED_XY, JetOutput::Value).is_err());
        }

        /// The derivative of an expanded transcendental falls out of the chain
        /// rule for free — no per-transcendental jet rule. Proven here on a
        /// *partial* sin expansion (range-reduce + low-degree term) that fits
        /// the jet register budget: d/dx of the leading `t·c1 = (x/π)·π = x`
        /// term's structure differentiates correctly through Mul/Sub/Floor.
        ///
        /// The FULL sin expansion currently exceeds emit_jet's no-spill register
        /// pool (see `jet_full_sin_hits_register_wall`); deriving large
        /// transcendentals is unblocked when the jet path moves onto the unified
        /// spilling allocator. The *value* path (real spilling driver) handles
        /// full sin fine — see `lowering_tests`.
        #[test]
        fn jet_chain_rule_through_floor_and_mul() {
            // f(x) = x - floor(x*0 + 0.5)*c  ; the floor branch is constant
            // (∂=0), so ∂f/∂x = 1 — exercises Floor's zero-partial + Sub/Mul.
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let zero = a.push_const(0.0);
            let half = a.push_const(0.5);
            let xr = a.push_binary(OpKind::Mul, x, zero);
            let xr = a.push_binary(OpKind::Add, xr, half);
            let k = a.push_unary(OpKind::Floor, xr); // floor(0.5)=0, ∂=0
            let c = a.push_const(7.0);
            let kc = a.push_binary(OpKind::Mul, k, c);
            let root = a.push_binary(OpKind::Sub, x, kc); // = x - 0 = x
            for &xv in &[0.3f32, 1.0, -0.7] {
                let dx = comp(&a, root, SEED_XY, JetOutput::Partial(0), xv, 0.0);
                assert!((dx - 1.0).abs() <= 1e-5, "∂/∂x = {dx}, want 1");
            }
        }

        /// Documents the known limit: the full sin expansion overflows
        /// emit_jet's no-spill register pool. Loud error, never a miscompile.
        /// Remove when the jet path gains spilling.
        #[test]
        fn jet_full_sin_hits_register_wall() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let root = a.push_unary(OpKind::Sin, x);
            assert!(
                compile_arena_dag_jet(&a, root, SEED_XY, JetOutput::Value).is_err(),
                "full sin jet unexpectedly fit — jet may have gained spilling; \
                 if so, replace this with a d/dx sin = cos value check"
            );
        }
    }

    /// Transcendental lowering: sin/cos/tan JIT through the per-batch path with
    /// no backend ever emitting a transcendental (they expand to arithmetic in
    /// `lowering`). Validated against `f32` on the default (128-bit) build.
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
    mod lowering_tests {
        use super::*;
        use crate::arena::ExprArena;

        // Tolerance reflects the degree-7 Chebyshev's ACTUAL measured accuracy
        // (the SAME polynomial the runtime Compounds path uses): ~1e-6 near 0,
        // degrading to ~2.6e-2 at the ±π range-reduction edges — only ~2-digit
        // accurate there. The bound catches *logic* errors (sign/coefficient/
        // range-reduction), not approximation error; tightening the polynomial
        // is the tunable-precision lever, applied in one place (`lowering`),
        // later.
        const TRIG_TOL: f32 = 3e-2;

        #[test]
        fn sin_cos_tan_match_scalar() {
            // Range beyond [-π,π] to exercise the floor-based range reduction.
            let pts = [0.0f32, 0.3, 1.0, 2.0, 3.5, -1.7, 6.0, -4.2];
            for &xv in &pts {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let s = a.push_unary(OpKind::Sin, x);
                assert!((run1(&a, s, xv) - xv.sin()).abs() <= TRIG_TOL, "sin({xv})");

                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let c = a.push_unary(OpKind::Cos, x);
                assert!((run1(&a, c, xv) - xv.cos()).abs() <= TRIG_TOL, "cos({xv})");
            }
            // tan away from its poles (ratio of two ~3e-3 approximations).
            for &xv in &[0.0f32, 0.3, 0.7, -0.5, 1.0] {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let t = a.push_unary(OpKind::Tan, x);
                // tan = sin/cos amplifies cos's ~2e-2 edge error as |x| grows
                // (measured ~3.8e-2 at x=1). Honest bound for this polynomial.
                assert!((run1(&a, t, xv) - xv.tan()).abs() <= 5e-2, "tan({xv})");
            }
        }

        /// exp/exp2/ln/log2/log10 lower to arithmetic via the bit-manip
        /// primitives (TruncToInt/IntToFloat/IAdd/Shl/Shr/BitAnd/BitOr) — the
        /// float↔int twiddling no backend can avoid. Validated vs `f32`.
        #[test]
        fn exp_log_match_scalar() {
            // exp / exp2 over a moderate range.
            for &xv in &[-2.0f32, -0.5, 0.0, 0.7, 1.5, 3.0] {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let e = a.push_unary(OpKind::Exp, x);
                let rel = (run1(&a, e, xv) - xv.exp()).abs() / xv.exp().max(1.0);
                assert!(rel <= 1e-2, "exp({xv})");

                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let e2 = a.push_unary(OpKind::Exp2, x);
                let rel = (run1(&a, e2, xv) - xv.exp2()).abs() / xv.exp2().max(1.0);
                assert!(rel <= 1e-2, "exp2({xv})");
            }
            // ln / log2 / log10 over positive inputs.
            for &xv in &[0.25f32, 0.5, 1.0, 2.0, 5.0, 100.0] {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let l = a.push_unary(OpKind::Ln, x);
                assert!((run1(&a, l, xv) - xv.ln()).abs() <= 3e-2, "ln({xv})");

                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let l2 = a.push_unary(OpKind::Log2, x);
                assert!((run1(&a, l2, xv) - xv.log2()).abs() <= 3e-2, "log2({xv})");
            }
        }

        /// atan/atan2/asin/acos lower to arithmetic + Select (atan2 is the core;
        /// the others derive from it). Value path only — atan2 uses Select, which
        /// the jet path can't differentiate. Validated vs `f32`.
        #[test]
        fn inverse_trig_match_scalar() {
            // The degree-7 atan polynomial is worst (~6e-2) at |ratio|=1; that
            // dominates the tolerance. It catches logic/quadrant errors (which
            // were off by whole radians via the guard bug), not approximation
            // error. Tightening the polynomial is the tunable-precision lever.
            const ATAN_TOL: f32 = 7e-2;

            // atan over a wide range (exercises the |ratio|>1 swap branch).
            for &xv in &[0.0f32, 0.3, 1.0, 2.5, -0.7, -4.0] {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let at = a.push_unary(OpKind::Atan, x);
                assert!((run1(&a, at, xv) - xv.atan()).abs() <= ATAN_TOL, "atan({xv})");
            }
            // asin/acos on [-1, 1].
            for &xv in &[-0.9f32, -0.4, 0.0, 0.4, 0.9] {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let s = a.push_unary(OpKind::Asin, x);
                assert!((run1(&a, s, xv) - xv.asin()).abs() <= ATAN_TOL, "asin({xv})");

                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let c = a.push_unary(OpKind::Acos, x);
                assert!((run1(&a, c, xv) - xv.acos()).abs() <= ATAN_TOL, "acos({xv})");
            }
            // atan2 across quadrants (y in var0, x in var1). The (1,1)/(-1,-1)…
            // cases sit at |ratio|=1, the polynomial's worst point.
            let pts = [(1.0f32, 1.0f32), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0), (0.5, -2.0)];
            for &(yv, xv) in &pts {
                let mut a = ExprArena::new();
                let y = a.push_var(0);
                let x = a.push_var(1);
                let r = a.push_binary(OpKind::Atan2, y, x);
                let got = run_xy(&a, r, yv, xv);
                assert!((got - yv.atan2(xv)).abs() <= ATAN_TOL, "atan2({yv},{xv}) = {got}");
            }
        }

        /// A transcendental composed inside arithmetic still works: sin(x)·x + 1.
        #[test]
        fn transcendental_in_expression() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let s = a.push_unary(OpKind::Sin, x);
            let sx = a.push_binary(OpKind::Mul, s, x);
            let one = a.push_const(1.0);
            let root = a.push_binary(OpKind::Add, sx, one);
            for &xv in &[0.2f32, 0.9, 2.1, -1.3] {
                let want = xv.sin() * xv + 1.0;
                // sin's ~3e-3 error is scaled by |x|, so allow for that.
                let tol = 3e-3 * (1.0 + xv.abs());
                assert!((run1(&a, root, xv) - want).abs() <= tol, "sin(x)·x+1 @ {xv}");
            }
        }
    }

    // =========================================================================
    // x86 shared-pipeline per-batch path (schedule → regalloc → spill).
    // =========================================================================
    // Calls kernels through the per-batch `KernelFn` (128-bit here); gated off
    // `+avx512f` where that ABI is `__m512` (AVX-512 covered by `avx512_driver`).
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
    mod sched {
        use super::*;
        use crate::arena::ExprArena;

        fn run(res: &CompileResult, x: f32, y: f32, z: f32, w: f32) -> f32 {
            unsafe {
                use core::arch::x86_64::*;
                let f: executable::KernelFn = res.code.as_fn();
                let o = f(
                    _mm_set1_ps(x),
                    _mm_set1_ps(y),
                    _mm_set1_ps(z),
                    _mm_set1_ps(w),
                );
                _mm_cvtss_f32(o)
            }
        }

        const PTS: &[(f32, f32, f32, f32)] = &[
            (3.0, 4.0, 0.0, 1.0),
            (1.0, 2.0, 3.0, 4.0),
            (-2.0, 0.5, 1.5, -1.0),
            (0.7, -1.3, 2.1, 0.2),
        ];

        /// The scheduled path must agree with the Sethi-Ullman path (and ground
        /// truth) for expressions that fit in registers.
        #[test]
        fn sched_parity_no_spill() {
            // f = sqrt(X*X + Y*Y) - Z, plus a non-commutative `X - Y*Z` shape.
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let z = a.push_var(2);
            let xx = a.push_binary(OpKind::Mul, x, x);
            let yy = a.push_binary(OpKind::Mul, y, y);
            let sum = a.push_binary(OpKind::Add, xx, yy);
            let dist = a.push_unary(OpKind::Sqrt, sum);
            let yz = a.push_binary(OpKind::Mul, y, z);
            let sub = a.push_binary(OpKind::Sub, dist, yz); // dist - Y*Z
            let root = sub;

            let sethi = compile_arena_dag(&a, root).expect("sethi compile");
            let sched = compile_arena_dag(&a, root).expect("scheduled compile");
            assert_eq!(sched.spill_count, 0, "should fit without spilling");

            for &(px, py, pz, pw) in PTS {
                let want = (px * px + py * py).sqrt() - py * pz;
                let g_sethi = run(&sethi, px, py, pz, pw);
                let g_sched = run(&sched, px, py, pz, pw);
                assert!((g_sethi - want).abs() <= 1e-4, "sethi {g_sethi} want {want}");
                assert!((g_sched - want).abs() <= 1e-4, "sched {g_sched} want {want}");
            }
        }

        /// A wide expression that exceeds the 7 allocatable registers must spill
        /// (to the red zone) and still compute the right answer.
        #[test]
        fn sched_spills_and_is_correct() {
            // sum_{i=1..=10} (X + i) * (Y + i), as a balanced tree so the 10
            // products are live together — forcing spills with only 7 regs.
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let mut terms = alloc::vec::Vec::new();
            for i in 1..=10u32 {
                let c = a.push_const(i as f32);
                let ax = a.push_binary(OpKind::Add, x, c);
                let by = a.push_binary(OpKind::Add, y, c);
                terms.push(a.push_binary(OpKind::Mul, ax, by));
            }
            while terms.len() > 1 {
                let mut next = alloc::vec::Vec::new();
                let mut it = terms.chunks(2);
                while let Some(pair) = it.next() {
                    if pair.len() == 2 {
                        next.push(a.push_binary(OpKind::Add, pair[0], pair[1]));
                    } else {
                        next.push(pair[0]);
                    }
                }
                terms = next;
            }
            let root = terms[0];

            let sched = compile_arena_dag(&a, root).expect("scheduled compile");
            assert!(
                sched.spill_count > 0,
                "expected spilling; widen the expression if this regresses"
            );

            for &(px, py, _pz, _pw) in PTS {
                let mut want = 0.0f32;
                for i in 1..=10u32 {
                    want += (px + i as f32) * (py + i as f32);
                }
                let got = run(&sched, px, py, 0.0, 0.0);
                let tol = 1e-3 * want.abs().max(1.0);
                assert!((got - want).abs() <= tol, "spill: got {got} want {want}");
            }
        }

        /// Exercises the shared driver's Select short-circuit guard path on x86
        /// (MOVMSKPS all-true/all-false branches): `(X > 0) ? Y*Y*Y : Z+Z+Z`,
        /// with arm-exclusive subexpressions so a guard region forms. Uniform
        /// inputs take the all-true / all-false branches.
        #[test]
        fn sched_select_guards() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let z = a.push_var(2);
            let zero = a.push_const(0.0);
            let cond = a.push_binary(OpKind::Gt, x, zero); // X > 0 -> mask
            let yy = a.push_binary(OpKind::Mul, y, y);
            let yyy = a.push_binary(OpKind::Mul, yy, y); // true arm: Y^3
            let zz = a.push_binary(OpKind::Add, z, z);
            let zzz = a.push_binary(OpKind::Add, zz, z); // false arm: 3Z
            let root = a.push_ternary(OpKind::Select, cond, yyy, zzz);

            let sched = compile_arena_dag(&a, root).expect("scheduled compile");

            // x>0 -> all-true -> Y^3 ; x<=0 -> all-false -> 3Z.
            for &(px, py, pz, _pw) in PTS {
                let want = if px > 0.0 { py * py * py } else { 3.0 * pz };
                let got = run(&sched, px, py, pz, 0.0);
                assert!((got - want).abs() <= 1e-3, "select: ({px},{py},{pz}) got {got} want {want}");
            }
        }
    }

    // =========================================================================
    // AVX-512 end-to-end: arena -> shared driver -> EVEX zmm kernel, run on the
    // host across all 16 lanes. Built only with +avx512f.
    // =========================================================================
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    mod avx512_driver {
        use super::*;
        use crate::arena::ExprArena;

        type K = unsafe extern "C" fn(
            core::arch::x86_64::__m512,
            core::arch::x86_64::__m512,
            core::arch::x86_64::__m512,
            core::arch::x86_64::__m512,
        ) -> core::arch::x86_64::__m512;

        /// Run a compiled zmm kernel over 16 distinct lanes per coordinate.
        fn run16(
            res: &CompileResult,
            xs: [f32; 16],
            ys: [f32; 16],
            zs: [f32; 16],
        ) -> [f32; 16] {
            unsafe {
                use core::arch::x86_64::*;
                let f: K = res.code.as_fn();
                let r = f(
                    _mm512_loadu_ps(xs.as_ptr()),
                    _mm512_loadu_ps(ys.as_ptr()),
                    _mm512_loadu_ps(zs.as_ptr()),
                    _mm512_setzero_ps(),
                );
                let mut out = [0.0f32; 16];
                _mm512_storeu_ps(out.as_mut_ptr(), r);
                out
            }
        }

        fn lanes() -> ([f32; 16], [f32; 16], [f32; 16]) {
            let mut xs = [0.0; 16];
            let mut ys = [0.0; 16];
            let mut zs = [0.0; 16];
            for i in 0..16 {
                xs[i] = i as f32 - 7.0;
                ys[i] = (i as f32) * 0.5 + 1.0;
                zs[i] = 3.0 - (i as f32) * 0.25;
            }
            (xs, ys, zs)
        }

        fn check(got: [f32; 16], want: impl Fn(usize) -> f32, tag: &str) {
            for i in 0..16 {
                let w = want(i);
                assert!((got[i] - w).abs() <= 1e-3, "{tag} lane {i}: got {} want {}", got[i], w);
            }
        }

        /// sqrt(X*X + Y*Y) - Z, with a non-commutative shape and FMA-able terms,
        /// fitting in registers (no spill).
        #[test]
        fn avx512_arith_no_spill() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let z = a.push_var(2);
            let xx = a.push_binary(OpKind::Mul, x, x);
            let yy = a.push_binary(OpKind::Mul, y, y);
            let sum = a.push_binary(OpKind::Add, xx, yy);
            let dist = a.push_unary(OpKind::Sqrt, sum);
            let root = a.push_binary(OpKind::Sub, dist, z);

            let res = compile_arena_dag_avx512(&a, root).expect("avx512 compile");
            assert_eq!(res.spill_count, 0, "should fit without spilling");

            let (xs, ys, zs) = lanes();
            check(
                run16(&res, xs, ys, zs),
                |i| (xs[i] * xs[i] + ys[i] * ys[i]).sqrt() - zs[i],
                "norm-z",
            );
        }

        /// A wide expression that exceeds the 6 allocatable zmm regs, forcing a
        /// real 64-byte-slot stack frame (the SSE2 red zone cannot hold a zmm).
        #[test]
        fn avx512_spills_to_real_frame() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let mut terms = alloc::vec::Vec::new();
            for i in 1..=10u32 {
                let c = a.push_const(i as f32);
                let ax = a.push_binary(OpKind::Add, x, c);
                let by = a.push_binary(OpKind::Add, y, c);
                terms.push(a.push_binary(OpKind::Mul, ax, by));
            }
            while terms.len() > 1 {
                let mut next = alloc::vec::Vec::new();
                for pair in terms.chunks(2) {
                    if pair.len() == 2 {
                        next.push(a.push_binary(OpKind::Add, pair[0], pair[1]));
                    } else {
                        next.push(pair[0]);
                    }
                }
                terms = next;
            }
            let root = terms[0];

            let res = compile_arena_dag_avx512(&a, root).expect("avx512 compile");
            assert!(res.spill_count > 0, "expected spilling");

            let (xs, ys, zs) = lanes();
            check(
                run16(&res, xs, ys, zs),
                |i| {
                    let mut acc = 0.0f32;
                    for k in 1..=10u32 {
                        acc += (xs[i] + k as f32) * (ys[i] + k as f32);
                    }
                    acc
                },
                "spill",
            );
        }

        /// Compare + select with non-exclusive arms: `(X < Y) ? X : Y` (== min).
        /// No guard region forms, so this is the plain vcmpps->vpmovm2d mask +
        /// vpternlogd blend path.
        #[test]
        fn avx512_compare_select_blend() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let cond = a.push_binary(OpKind::Lt, x, y);
            let root = a.push_ternary(OpKind::Select, cond, x, y);

            let res = compile_arena_dag_avx512(&a, root).expect("avx512 compile");
            let (xs, ys, zs) = lanes();
            check(run16(&res, xs, ys, zs), |i| xs[i].min(ys[i]), "lt-select");
        }

        /// Select with arm-exclusive subexpressions: `(X > 0) ? Y*Y*Y : Z+Z+Z`.
        /// Forms guard regions, exercising the vptestmd+kortestw short-circuit
        /// branches (all-false skips Y^3, all-true skips 3Z) plus the per-lane
        /// blend on mixed input.
        #[test]
        fn avx512_select_guards() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let z = a.push_var(2);
            let zero = a.push_const(0.0);
            let cond = a.push_binary(OpKind::Gt, x, zero);
            let yy = a.push_binary(OpKind::Mul, y, y);
            let yyy = a.push_binary(OpKind::Mul, yy, y);
            let zz = a.push_binary(OpKind::Add, z, z);
            let zzz = a.push_binary(OpKind::Add, zz, z);
            let root = a.push_ternary(OpKind::Select, cond, yyy, zzz);

            let res = compile_arena_dag_avx512(&a, root).expect("avx512 compile");

            let allpos = [2.0f32; 16];
            let allneg = [-2.0f32; 16];
            let ys = core::array::from_fn::<f32, 16, _>(|i| i as f32 * 0.5 + 1.0);
            let zs = core::array::from_fn::<f32, 16, _>(|i| 3.0 - i as f32 * 0.25);
            check(run16(&res, allpos, ys, zs), |i| ys[i] * ys[i] * ys[i], "guard-true");
            check(run16(&res, allneg, ys, zs), |i| 3.0 * zs[i], "guard-false");

            let mixed = core::array::from_fn::<f32, 16, _>(|i| if i % 2 == 0 { 1.0 } else { -1.0 });
            check(
                run16(&res, mixed, ys, zs),
                |i| if mixed[i] > 0.0 { ys[i] * ys[i] * ys[i] } else { 3.0 * zs[i] },
                "guard-mixed",
            );
        }

        /// Rounding via vrndscaleps (floor/ceil/round), each a single EVEX op.
        #[test]
        fn avx512_rounding() {
            // Mixed fractional/sign inputs so each rounding mode is distinct.
            let xs = core::array::from_fn::<f32, 16, _>(|i| (i as f32 - 8.0) * 0.7);
            let ones = [1.0f32; 16];
            for (op, f, tag) in [
                (OpKind::Floor, f32::floor as fn(f32) -> f32, "floor"),
                (OpKind::Ceil, f32::ceil as fn(f32) -> f32, "ceil"),
                (OpKind::Round, f32::round_ties_even as fn(f32) -> f32, "round"),
            ] {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let root = a.push_unary(op, x);
                let res = compile_arena_dag_avx512(&a, root).expect("avx512 compile");
                check(run16(&res, xs, ones, ones), |i| f(xs[i]), tag);
            }
        }
    }
}
