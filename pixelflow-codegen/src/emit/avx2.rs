//! x86-64 AVX2 (VEX.256) JIT encoder — 256-bit, 8-lane `ymm` kernels.
//!
//! The middle width between the SSE2 leaf encoders (`x86_64.rs`, 128-bit) and
//! the AVX-512 EVEX encoders (`avx512.rs`, 512-bit). Register numbering is
//! identical to SSE2 (ymm0-15, no extended file — AVX2 has no REX2/EVEX), so
//! this backend reuses the exact register-role layout `X86Backend` already
//! established (inputs 0-3, allocatable 4-9, fixed scratch 10, reload 11-12,
//! builtin scratch 10/13-15): only the instruction *encoding* changes.
//!
//! Unlike legacy SSE2, VEX is 3-operand and non-destructive — same property
//! AVX-512's EVEX has — so there is no two-operand hazard to route around
//! (contrast `emit_binary_safe` in `mod.rs`, needed only by the SSE2 path).
//! Comparisons are simpler here than on AVX-512: `vcmpps` writes an ordinary
//! all-ones/all-zeros `ymm` directly (no k-register, no mask-to-vector
//! conversion) — the same representation NEON and SSE2 already use.
//!
//! Spills use a real stack frame, not the red zone: mirrors `avx512.rs`'s
//! reasoning (a `ymm` slot is 32 bytes; keeping the red-zone arithmetic exact
//! for two different slot widths is not worth it for a bit of frame reuse on
//! tiny kernels).
//!
//! Gather has no direct AVX2 hardware analogue reused here: `vgatherdps`'s
//! VSIB + vector-mask-with-clearing semantics are a bigger lift than this
//! backend's scope warrants, so — like `X86Backend` — a gather is assembled
//! from scalar loads via the existing 128-bit lane-insert sequence
//! (`x86_64::emit_gather_scalar`), run once per 128-bit half and combined
//! with `vinsertf128`.

use super::x86_64;
use super::x86_64::{Disp, Imm8, Imm32, Mem, NoDisp, gpr};
use super::{Reg, unimplemented_op};
use alloc::vec::Vec;
use pixelflow_ir::OpKind;

// The AVX2 tier requires FMA3. No shipping x86-64 CPU has ever offered AVX2
// without it (Intel: both since Haswell; AMD: FMA3 predates AVX2 by a
// generation) — the industry itself codifies the pairing as x86-64-v3. A
// hypothetical AVX2-without-FMA build is not a smaller tier, it is a paper
// configuration: it forked `emit_fmadd_c_in_dst` into a value-semantics
// variant (one rounding vs. two, see CLAUDE.md's `MulAdd` platform-divergence
// table) that no real machine ever exercised, and that fork was directly
// responsible for a P2 (two materially different kernels sharing one
// environment fingerprint — see `pixelflow-pipeline/src/journal.rs`). Fail
// loudly at compile time rather than silently degrading precision.
#[cfg(all(target_feature = "avx2", not(target_feature = "fma")))]
compile_error!(
    "the AVX2 backend requires FMA3 (no shipping CPU has AVX2 without it); \
     build with `-C target-feature=+avx2,+fma` or `-C target-cpu=x86-64-v3`"
);

// =============================================================================
// VEX.256 encoder
// =============================================================================

/// Which legacy-prefix byte the VEX prefix implies (the `pp` field).
#[derive(Clone, Copy)]
enum Pp {
    /// No implied prefix.
    None = 0,
    /// `66`
    P66 = 1,
    /// `F3`
    F3 = 2,
}

/// Which opcode map the instruction lives in (the field Intel calls
/// `mmmmm` — a map *selector*, nothing more).
#[derive(Clone, Copy)]
enum Map {
    /// `0F`
    M0F = 1,
    /// `0F38`
    M0F38 = 2,
    /// `0F3A`
    M0F3A = 3,
}

/// The identity of one VEX-256 instruction: opcode map, implied legacy
/// prefix, W bit, opcode byte. This quadruple is *which instruction* — it is
/// constant per mnemonic, so each mnemonic below states it exactly once and
/// the operand form (`rrr`/`imm`/`rm`) supplies the per-call parts.
#[derive(Clone, Copy)]
struct Vex {
    map: Map,
    pp: Pp,
    w: bool,
    opcode: u8,
}

/// Sentinel for an unused VEX.vvvv source (2-operand forms): index 0 inverts
/// to `1111`, the required "unused" encoding.
const UNUSED_VVVV: u8 = 0;

impl Vex {
    const fn new(map: Map, pp: Pp, opcode: u8) -> Self {
        Self {
            map,
            pp,
            w: false,
            opcode,
        }
    }
    /// Map `0F`, no prefix — the packed-single arithmetic family.
    const fn m0f(opcode: u8) -> Self {
        Self::new(Map::M0F, Pp::None, opcode)
    }
    /// Map `0F`, `66` — the integer-domain family.
    const fn m0f_66(opcode: u8) -> Self {
        Self::new(Map::M0F, Pp::P66, opcode)
    }
    /// Map `0F`, `F3`.
    const fn m0f_f3(opcode: u8) -> Self {
        Self::new(Map::M0F, Pp::F3, opcode)
    }
    /// Map `0F38`, `66`.
    const fn m0f38_66(opcode: u8) -> Self {
        Self::new(Map::M0F38, Pp::P66, opcode)
    }
    /// Map `0F3A`, `66` — the imm8 family (round, insert/extract).
    const fn m0f3a_66(opcode: u8) -> Self {
        Self::new(Map::M0F3A, Pp::P66, opcode)
    }

    /// Attach an imm8 (`vcmpps` predicate, rounding mode, shift count, lane
    /// index); the returned value emits it after the instruction.
    const fn imm(self, imm: u8) -> VexImm {
        VexImm { vex: self, imm }
    }

    /// Register-register-register form: `op dst, vvvv, rm`.
    fn rrr(self, code: &mut Vec<u8>, dst: u8, vvvv: u8, rm: u8) {
        let rbit = if dst >= 8 { 0x00 } else { 0x80 };
        let xbit = 0x40;
        let bbit = if rm >= 8 { 0x00 } else { 0x20 };
        code.push(0xC4);
        code.push(rbit | xbit | bbit | self.map as u8);
        code.push(((self.w as u8) << 7) | ((!vvvv & 0xF) << 3) | (1 << 2) | self.pp as u8); // L=1
        code.push(self.opcode);
        code.push(0xC0 | ((dst & 7) << 3) | (rm & 7));
    }

    /// `op reg, [addr]` — the memory-operand form, for any base and any
    /// displacement mode. The prefix is VEX's; the ModRM/SIB/displacement tail
    /// is the architecture's, so it comes from `x86_64::mem_operand`.
    fn rm<D: Disp>(self, code: &mut Vec<u8>, reg: u8, addr: Mem<D>) {
        // R and B are stored inverted; X is unused (no index register).
        let rbit = if reg >= 8 { 0x00 } else { 0x80 };
        let bbit = if addr.base.0 >= 8 { 0x00 } else { 0x20 };
        code.push(0xC4);
        code.push(rbit | 0x40 | bbit | self.map as u8);
        code.push(((self.w as u8) << 7) | (0xF << 3) | (1 << 2) | self.pp as u8); // vvvv unused, L=1
        code.push(self.opcode);
        x86_64::mem_operand(code, reg, addr);
    }
}

/// A [`Vex`] instruction carrying its imm8.
#[derive(Clone, Copy)]
struct VexImm {
    vex: Vex,
    imm: u8,
}

impl VexImm {
    /// Register form with the imm8 appended.
    fn rrr(self, code: &mut Vec<u8>, dst: u8, vvvv: u8, rm: u8) {
        self.vex.rrr(code, dst, vvvv, rm);
        code.push(self.imm);
    }
}

// --- packed-single arithmetic (0F, no prefix, W0) ---
fn vaddps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f(0x58).rrr(c, d, s1, s2);
}
fn vsubps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f(0x5C).rrr(c, d, s1, s2);
}
fn vmulps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f(0x59).rrr(c, d, s1, s2);
}
fn vdivps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f(0x5E).rrr(c, d, s1, s2);
}
fn vminps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f(0x5D).rrr(c, d, s1, s2);
}
fn vmaxps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f(0x5F).rrr(c, d, s1, s2);
}
fn vsqrtps(c: &mut Vec<u8>, d: u8, s: u8) {
    Vex::m0f(0x51).rrr(c, d, UNUSED_VVVV, s);
}
fn vrsqrtps(c: &mut Vec<u8>, d: u8, s: u8) {
    Vex::m0f(0x52).rrr(c, d, UNUSED_VVVV, s);
}
fn vrcpps(c: &mut Vec<u8>, d: u8, s: u8) {
    Vex::m0f(0x53).rrr(c, d, UNUSED_VVVV, s);
}

// --- bitwise (0F, no prefix, W0) ---
fn vandps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f(0x54).rrr(c, d, s1, s2);
}
fn vandnps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f(0x55).rrr(c, d, s1, s2);
}
fn vorps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f(0x56).rrr(c, d, s1, s2);
}
fn vxorps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f(0x57).rrr(c, d, s1, s2);
}

// --- comparisons (0F, no prefix, W0; imm8 predicate) ---
const CMP_EQ: u8 = 0;
const CMP_LT: u8 = 1;
const CMP_LE: u8 = 2;
const CMP_NEQ: u8 = 4;
const CMP_GE: u8 = 5;
const CMP_NLE: u8 = 6; // > (unordered-safe "not less-or-equal")

fn vcmpps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8, pred: u8) {
    Vex::m0f(0xC2).imm(pred).rrr(c, d, s1, s2);
}

fn cmp_pred(op: OpKind) -> Option<u8> {
    Some(match op {
        OpKind::Eq => CMP_EQ,
        OpKind::Ne => CMP_NEQ,
        OpKind::Lt => CMP_LT,
        OpKind::Le => CMP_LE,
        OpKind::Gt => CMP_NLE,
        OpKind::Ge => CMP_GE,
        _ => return None,
    })
}

/// Whether `op` is a comparison handled by [`emit_binary`].
#[must_use]
pub fn is_compare(op: OpKind) -> bool {
    cmp_pred(op).is_some()
}

// --- rounding (0F3A, 66 prefix, W0; imm8) ---
fn vroundps(c: &mut Vec<u8>, d: u8, s: u8, imm: u8) {
    Vex::m0f3a_66(0x08).imm(imm).rrr(c, d, UNUSED_VVVV, s);
}

// --- int/float convert (0F, W0) ---
fn vcvttps2dq(c: &mut Vec<u8>, d: u8, s: u8) {
    Vex::m0f_f3(0x5B).rrr(c, d, UNUSED_VVVV, s); // F3 prefix
}
fn vcvtdq2ps(c: &mut Vec<u8>, d: u8, s: u8) {
    Vex::m0f(0x5B).rrr(c, d, UNUSED_VVVV, s); // no prefix
}

// --- integer-domain (66 prefix, 0F, W0) ---
fn vpaddd(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f_66(0xFE).rrr(c, d, s1, s2);
}
fn vpslld_imm(c: &mut Vec<u8>, d: u8, s: u8, imm: u8) {
    Vex::m0f_66(0x72).imm(imm).rrr(c, 6, d, s); // /6, dst=vvvv, src=rm
}
fn vpsrld_imm(c: &mut Vec<u8>, d: u8, s: u8, imm: u8) {
    Vex::m0f_66(0x72).imm(imm).rrr(c, 2, d, s); // /2
}

// --- lane insert/extract between 256-bit and 128-bit (0F3A, 66 prefix, W0) ---
/// `vinsertf128 ymmDST, ymmSRC1, xmmSRC2, imm8[0]` — copy `src1`, then place
/// `src2` into the low (`imm=0`) or high (`imm=1`) 128 bits.
fn vinsertf128(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8, imm: u8) {
    Vex::m0f3a_66(0x18).imm(imm).rrr(c, d, s1, s2);
}
/// `vextractf128 xmmDST, ymmSRC, imm8[0]` — extract the low (`imm=0`) or high
/// (`imm=1`) 128 bits of `src` into `dst`.
fn vextractf128(c: &mut Vec<u8>, d: u8, s: u8, imm: u8) {
    // VEX.256.66.0F3A.W0 19 /r ib — note dst is the ModRM.rm operand here
    // (the reverse of the usual direction: register source, register/mem dest).
    Vex::m0f3a_66(0x19).imm(imm).rrr(c, s, UNUSED_VVVV, d);
}

/// `vmovaps ymmDST, ymmSRC` — register copy.
pub fn emit_mov(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    if dst.0 == src.0 {
        return;
    }
    Vex::m0f(0x28).rrr(code, dst.0, UNUSED_VVVV, src.0);
}

/// A slot in the allocated spill frame. AVX2 kernels are leaves with no base
/// pointer, so a slot *is* `rsp + offset`.
const fn frame_slot(offset: u32) -> Mem<Imm32> {
    Mem {
        base: gpr::RSP,
        disp: Imm32(offset as i32),
    }
}

/// `vmovups ymmDST, [addr]` — 256-bit load.
pub fn emit_load<D: Disp>(code: &mut Vec<u8>, dst: Reg, addr: Mem<D>) {
    Vex::m0f(0x10).rm(code, dst.0, addr);
}

/// `vmovups [addr], ymmSRC` — 256-bit store.
pub fn emit_store<D: Disp>(code: &mut Vec<u8>, addr: Mem<D>, src: Reg) {
    Vex::m0f(0x11).rm(code, src.0, addr);
}

/// Where [`emit_const`] stages an f32 before broadcasting it: four bytes of
/// red zone below `rsp`, never touched by a spill frame (which lives at
/// `[rsp .. rsp+N)`).
const RED_ZONE_CONST: Mem<Imm8> = Mem {
    base: gpr::RSP,
    disp: Imm8(-4),
};

/// Broadcast an f32 constant to all 8 lanes of `dst` via the stack (red zone,
/// `[rsp-4]`; never affected by a spill frame, which lives at `[rsp..rsp+N)`).
pub fn emit_const(code: &mut Vec<u8>, dst: Reg, val: f32) {
    let bits = val.to_bits();
    if bits == 0 {
        vxorps(code, dst.0, dst.0, dst.0);
        return;
    }
    // mov dword [rsp-4], imm32
    code.extend_from_slice(&[0xC7, 0x44, 0x24, 0xFC]);
    code.extend_from_slice(&bits.to_le_bytes());
    // vbroadcastss ymm, [rsp-4]  (VEX.256.66.0F38.W0 18 /r)
    Vex::m0f38_66(0x18).rm(code, dst.0, RED_ZONE_CONST);
}

// =============================================================================
// Stack frame (real frame; a ymm spill is 32 bytes)
// =============================================================================

// =============================================================================
// Op dispatch
// =============================================================================

/// `dst = op(src1, src2)`. VEX is 3-operand/non-destructive: operands are
/// never clobbered and may alias `dst`. Comparisons produce an ordinary
/// all-ones/all-zeros vector directly (no k-register step, unlike AVX-512).
pub fn emit_binary(code: &mut Vec<u8>, op: OpKind, dst: Reg, src1: Reg, src2: Reg) {
    let (d, s1, s2) = (dst.0, src1.0, src2.0);
    if let Some(pred) = cmp_pred(op) {
        vcmpps(code, d, s1, s2, pred);
        return;
    }
    match op {
        OpKind::Add => vaddps(code, d, s1, s2),
        OpKind::Sub => vsubps(code, d, s1, s2),
        OpKind::Mul => vmulps(code, d, s1, s2),
        OpKind::Div => vdivps(code, d, s1, s2),
        OpKind::Min => vminps(code, d, s1, s2),
        OpKind::Max => vmaxps(code, d, s1, s2),
        OpKind::BitAnd => vandps(code, d, s1, s2),
        OpKind::BitOr => vorps(code, d, s1, s2),
        OpKind::IAdd => vpaddd(code, d, s1, s2),
        _ => unimplemented_op("avx2", op),
    }
}

/// `dst = op(src)`.
pub fn emit_unary(code: &mut Vec<u8>, op: OpKind, dst: Reg, src: Reg) {
    match op {
        OpKind::Sqrt => vsqrtps(code, dst.0, src.0),
        OpKind::Rsqrt => vrsqrtps(code, dst.0, src.0),
        OpKind::Recip => vrcpps(code, dst.0, src.0),
        OpKind::Neg => {
            emit_const(code, UNARY_SCRATCH, f32::from_bits(0x8000_0000));
            vxorps(code, dst.0, src.0, UNARY_SCRATCH.0);
        }
        OpKind::Abs => {
            emit_const(code, UNARY_SCRATCH, f32::from_bits(0x7FFF_FFFF));
            vandps(code, dst.0, src.0, UNARY_SCRATCH.0);
        }
        // imm8: bits[3:0] = rounding mode (0=nearest, 1=floor, 2=ceil).
        OpKind::Floor => vroundps(code, dst.0, src.0, 0x01),
        OpKind::Ceil => vroundps(code, dst.0, src.0, 0x02),
        OpKind::Round => vroundps(code, dst.0, src.0, 0x00),
        OpKind::TruncToInt => vcvttps2dq(code, dst.0, src.0),
        OpKind::IntToFloat => vcvtdq2ps(code, dst.0, src.0),
        _ => unimplemented_op("avx2", op),
    }
}

/// Scratch register for unary mask materialization (neg/abs) and the select
/// blend temp. ymm15 is outside the allocatable range (4-9), reload regs
/// (11-12), and inputs (0-3).
const UNARY_SCRATCH: Reg = Reg(15);

/// Emit a shift of i32 lanes by a compile-time immediate.
pub fn emit_shift_imm(code: &mut Vec<u8>, op: OpKind, dst: Reg, src: Reg, amount: u8) {
    match op {
        OpKind::Shl => vpslld_imm(code, dst.0, src.0, amount),
        OpKind::Shr => vpsrld_imm(code, dst.0, src.0, amount),
        _ => unimplemented_op("avx2", op),
    }
}

/// `dst = mask ? if_true : if_false` (bit-select; mask already in `dst`, same
/// convention as SSE2/AVX-512). `tmp` differs from all of `dst`/`if_true`/`if_false`.
pub fn emit_select(code: &mut Vec<u8>, dst: Reg, if_true: Reg, if_false: Reg) {
    let tmp = UNARY_SCRATCH;
    debug_assert!(tmp.0 != dst.0 && tmp.0 != if_true.0 && tmp.0 != if_false.0);
    vandps(code, tmp.0, dst.0, if_true.0); // tmp = mask & if_true
    vandnps(code, dst.0, dst.0, if_false.0); // dst = ~mask & if_false
    vorps(code, dst.0, tmp.0, dst.0); // dst = blended
}

/// `vmovmskps eax, ymmSRC` — gather the 8 lane sign bits into eax[7:0].
pub fn emit_movmskps_eax(code: &mut Vec<u8>, src: Reg) {
    Vex::m0f(0x50).rrr(code, 0, UNUSED_VVVV, src.0);
}

/// `cmp al, imm8` — unlike `cmp eax, imm8` (sign-extending `0x83`), this
/// compares the raw byte pattern, which is what an 8-lane all-true check
/// (`eax == 0xFF`) needs (`0x83`'s sign-extension would compare against
/// `0xFFFFFFFF`, which `vmovmskps`'s zero-extended result can never equal).
pub fn emit_cmp_al_imm8(code: &mut Vec<u8>, imm: u8) {
    code.push(0x3C);
    code.push(imm);
}

/// `vfmadd231ps ymmD, ymmA, ymmB` — `dst = a*b + dst` (231 form: dst is the
/// addend going in, `a`/`b` the product). VEX.256.66.0F38.W0 B8 /r, same
/// opcode as `avx512.rs`'s EVEX form, just VEX-encoded at 256 bits.
/// `target_feature = "fma"` (FMA3) is not implied by `avx2` alone in rustc's
/// feature model, which is why this file's own `compile_error!` pins the two
/// together for this backend — see the module-top comment.
fn vfmadd231ps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    Vex::m0f38_66(0xB8).rrr(c, d, s1, s2);
}

/// Fused multiply-add: `dst` already holds `c`; computes `dst = a*b + dst`.
///
/// Always real hardware FMA: the AVX2 tier requires `+fma` (see the
/// module-top `compile_error!`), so there is no software mul+add fallback to
/// choose between here. This rounds once, exactly matching the reference
/// interpreter under `fp-contract=fast` — `eval_scalar`'s scalar `a*b+c` gets
/// contracted to an `fma` instruction by LLVM under `+fma` too, so a
/// software two-step mul-then-add would round twice and disagree in the last
/// bit.
///
/// The two-roundings case still exists, just not in *this* function. It is
/// the SSE2 baseline's only option — `x86_64.rs`'s own `FusedMulAdd` arm is a
/// `movaps`/`mulps`/`addps` stand-in, as is `pixelflow-core`'s x86 backend —
/// and it is also what `DecomposedMulAdd` does on every tier, this one
/// included, whenever register pressure pulls `a` and `b` apart from `c`.
/// Both are pinned as bytes by `emit::tests::muladd_encoding` and as values
/// by `tests/muladd_rounding.rs`.
pub fn emit_fmadd_c_in_dst(code: &mut Vec<u8>, dst: Reg, a: Reg, b: Reg) {
    vfmadd231ps(code, dst.0, a.0, b.0);
}

// =============================================================================
// Bound-memory gather (RawGather lowering target)
//
// No native vgatherdps here (see the module doc): truncate all 8 lanes at
// once, split into two 128-bit halves, run the existing SSE2/AVX scalar-load
// sequence (`x86_64::emit_gather_scalar`) on each half (it only ever touches
// the low 128 bits of whatever register it's given — ymm0's low 128 IS
// xmm0), then recombine with vinsertf128.
// =============================================================================

/// Scratch the 256-bit gather clobbers: the 128-bit sequence's own scratch,
/// which both halves reuse, plus the two vector registers that carry the high
/// half while the low half is being assembled. All of it must be distinct from
/// the gather's `dst` and `idx`.
#[derive(Clone, Copy)]
pub struct GatherScratch {
    /// Scratch for one 128-bit half — see [`x86_64::GatherScratch`].
    pub half: x86_64::GatherScratch,
    /// Vector register receiving lanes 4..8 of the float indices.
    pub idx_hi: Reg,
    /// Vector register receiving the high half's gathered values.
    pub res_hi: Reg,
}

/// `dst = buffer[slot][idx_lane]` for 8 lanes. `idx` holds FLOAT indices
/// (already clamped in range by the `Gather` lowering — `x86_64::emit_gather_scalar`
/// does its own float->int truncation per half, so `idx` must not be
/// pre-truncated here). Clobbers everything in `s`.
pub fn emit_gather_scalar(code: &mut Vec<u8>, dst: Reg, idx: Reg, slot: u16, s: GatherScratch) {
    // idx's low 128 already holds lanes 0..4 (float); split off lanes 4..8
    // into idx_hi before either gather call touches idx/dst (which may alias).
    vextractf128(code, s.idx_hi.0, idx.0, 1);

    // Low half: lanes 0..4. May write dst == idx (the callee handles that:
    // it converts idx to int in scratch before ever writing dst).
    x86_64::emit_gather_scalar(code, dst, idx, slot, s.half);
    // High half: lanes 4..8, into res_hi (a 128-bit scratch distinct from dst).
    x86_64::emit_gather_scalar(code, s.res_hi, s.idx_hi, slot, s.half);

    // Recombine: dst[0..4] already holds the low half; splice in the high.
    vinsertf128(code, dst.0, dst.0, s.res_hi.0, 1);
}

#[cfg(test)]
mod tests {
    //! Hardware validation, mirroring `avx512.rs`'s runtime test tier: JIT real
    //! `ymm` kernels and execute them on the host.
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    mod runtime {
        use super::super::*;
        use crate::emit::executable::ExecutableCode;
        use core::arch::x86_64::*;

        #[allow(improper_ctypes_definitions)]
        type K = unsafe extern "C" fn(__m256, __m256, __m256, __m256) -> __m256;

        fn run(body: &[u8], xs: [f32; 8], ys: [f32; 8], zs: [f32; 8]) -> [f32; 8] {
            let mut code = body.to_vec();
            crate::emit::x86_64::ret(&mut code);
            let exec = unsafe { ExecutableCode::from_code(&code).expect("mmap") };
            unsafe {
                let f: K = exec.as_fn();
                let r = f(
                    _mm256_loadu_ps(xs.as_ptr()),
                    _mm256_loadu_ps(ys.as_ptr()),
                    _mm256_loadu_ps(zs.as_ptr()),
                    _mm256_setzero_ps(),
                );
                let mut out = [0.0f32; 8];
                _mm256_storeu_ps(out.as_mut_ptr(), r);
                out
            }
        }

        fn lanes() -> ([f32; 8], [f32; 8], [f32; 8]) {
            let mut xs = [0.0; 8];
            let mut ys = [0.0; 8];
            let mut zs = [0.0; 8];
            for i in 0..8 {
                xs[i] = i as f32 - 3.5;
                ys[i] = (i as f32) * 0.5 + 1.0;
                zs[i] = 3.0 - (i as f32) * 0.25;
            }
            (xs, ys, zs)
        }

        /// One row of the binary-op table: the op and its scalar reference.
        type BinaryCase = (OpKind, fn(f32, f32) -> f32);

        fn check(got: [f32; 8], want: impl Fn(usize) -> f32, tag: &str) {
            for (i, &g) in got.iter().enumerate() {
                let w = want(i);
                assert!((g - w).abs() <= 1e-3, "{tag} lane {i}: got {g} want {w}");
            }
        }

        /// Bit-exact check for mask results: an all-ones/all-zeros lane is
        /// NaN under float subtraction, so `check`'s epsilon comparison can't
        /// be used for compares/selects' underlying mask bit pattern.
        fn check_bits(got: [f32; 8], want: impl Fn(usize) -> u32, tag: &str) {
            for (i, &g) in got.iter().enumerate() {
                let w = want(i);
                assert_eq!(
                    g.to_bits(),
                    w,
                    "{tag} lane {i}: got {:#x} want {:#x}",
                    g.to_bits(),
                    w
                );
            }
        }

        const X: Reg = Reg(0);
        const Y: Reg = Reg(1);
        const Z: Reg = Reg(2);

        #[test]
        fn binary_ops() {
            let (xs, ys, zs) = lanes();
            let cases: &[BinaryCase] = &[
                (OpKind::Add, |a, b| a + b),
                (OpKind::Sub, |a, b| a - b),
                (OpKind::Mul, |a, b| a * b),
                (OpKind::Div, |a, b| a / b),
                (OpKind::Min, |a, b| a.min(b)),
                (OpKind::Max, |a, b| a.max(b)),
            ];
            for &(op, f) in cases {
                let mut c = Vec::new();
                emit_binary(&mut c, op, X, X, Y);
                check(run(&c, xs, ys, zs), |i| f(xs[i], ys[i]), "binary");
            }
        }

        #[test]
        fn compare_lt() {
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            emit_binary(&mut c, OpKind::Lt, X, X, Y);
            check_bits(
                run(&c, xs, ys, zs),
                |i| if xs[i] < ys[i] { 0xFFFF_FFFF } else { 0 },
                "lt mask",
            );
        }

        #[test]
        fn sqrt_and_neg_abs() {
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            emit_unary(&mut c, OpKind::Sqrt, X, Y);
            check(run(&c, xs, ys, zs), |i| ys[i].sqrt(), "sqrt");

            let mut c = Vec::new();
            emit_unary(&mut c, OpKind::Neg, X, X);
            check(run(&c, xs, ys, zs), |i| -xs[i], "neg");

            let mut c = Vec::new();
            emit_unary(&mut c, OpKind::Abs, X, X);
            check(run(&c, xs, ys, zs), |i| xs[i].abs(), "abs");
        }

        #[test]
        fn select_blend() {
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            emit_binary(&mut c, OpKind::Lt, Reg(5), X, Y); // mask
            emit_mov(&mut c, Reg(6), Reg(5));
            emit_select(&mut c, Reg(6), X, Y); // dst = mask ? x : y
            emit_mov(&mut c, X, Reg(6));
            check(
                run(&c, xs, ys, zs),
                |i| if xs[i] < ys[i] { xs[i] } else { ys[i] },
                "select",
            );
        }

        #[test]
        fn const_broadcast_and_fma() {
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            emit_const(&mut c, Reg(5), 2.5);
            emit_binary(&mut c, OpKind::Add, X, X, Reg(5));
            check(run(&c, xs, ys, zs), |i| xs[i] + 2.5, "const+add");

            let mut c = Vec::new();
            emit_mov(&mut c, Reg(5), Z);
            emit_fmadd_c_in_dst(&mut c, Reg(5), X, Y);
            emit_mov(&mut c, X, Reg(5));
            check(run(&c, xs, ys, zs), |i| xs[i] * ys[i] + zs[i], "fma sw");
        }

        /// The FMA bytes really are an FMA: **one** rounding, not a multiply
        /// followed by an add.
        ///
        /// `const_broadcast_and_fma`'s 1e-3 tolerance cannot tell those apart — the whole
        /// difference is the last mantissa bit — so a stand-in built out of a
        /// multiply and an add would pass it. `1.0000001 * 4097 + 4097` is one
        /// of the inputs CLAUDE.md's `MulAdd` row is about, where the two
        /// forms genuinely disagree, and this asserts the bits.
        #[test]
        fn fma_rounds_once() {
            let xs = [1.000_000_1f32; 8];
            let ys = [4097.0f32; 8];
            let zs = [4097.0f32; 8];
            let one = xs[0].mul_add(ys[0], zs[0]);
            // `black_box` stops LLVM contracting the reference into the very
            // instruction it exists to be different from.
            let two = core::hint::black_box(xs[0] * ys[0]) + zs[0];
            assert_ne!(
                one.to_bits(),
                two.to_bits(),
                "this input no longer separates one rounding from two"
            );

            let mut c = Vec::new();
            emit_mov(&mut c, Reg(5), Z);
            emit_fmadd_c_in_dst(&mut c, Reg(5), X, Y);
            emit_mov(&mut c, X, Reg(5));
            for (i, &g) in run(&c, xs, ys, zs).iter().enumerate() {
                assert_eq!(
                    g.to_bits(),
                    one.to_bits(),
                    "lane {i}: {g} rounded twice; the fused answer is {one}"
                );
            }
        }

        #[test]
        fn spill_frame_roundtrip() {
            let (xs, ys, zs) = lanes();
            let mut c = Vec::new();
            crate::emit::x86_64::emit_sub_rsp(&mut c, 32);
            emit_binary(&mut c, OpKind::Mul, Reg(6), X, Y);
            emit_store(&mut c, frame_slot(0), Reg(6));
            emit_binary(&mut c, OpKind::Add, Reg(6), X, X); // clobber
            emit_load(&mut c, X, frame_slot(0));
            crate::emit::x86_64::emit_add_rsp(&mut c, 32);
            check(run(&c, xs, ys, zs), |i| xs[i] * ys[i], "spill roundtrip");
        }

        #[test]
        fn gather_from_buffer() {
            // Matches the production ABI (mod.rs's ResolvedOp::Gather): the
            // first arg is a context pointer to an ARRAY of buffer base
            // pointers (one per slot), not a buffer pointer directly —
            // `x86_64::emit_gather_scalar` loads `[ctx_gpr + slot*8]` to get
            // the real base. `emit_load_ptr_from_ctx`'s doc calls this out.
            #[allow(improper_ctypes_definitions)]
            type G = unsafe extern "C" fn(*const *const f32, __m256) -> __m256;

            let mut c = Vec::new();
            // idx (zmm/ymm0) -> int truncate happens inside emit_gather_scalar.
            let s = x86_64::GatherScratch {
                base_gpr: 0,  // rax
                index_gpr: 1, // rcx
                ctx_gpr: 7,   // rdi
                idx_lanes: Reg(13),
                value: Reg(14),
            };
            emit_gather_scalar(
                &mut c,
                Reg(0),
                Reg(0),
                0,
                GatherScratch {
                    half: s,
                    idx_hi: Reg(9),
                    res_hi: Reg(8),
                },
            );
            crate::emit::x86_64::ret(&mut c);

            let buf: Vec<f32> = (0..64).map(|i| (i as f32) * 1.5 + 0.25).collect();
            let idx: [f32; 8] = [0.0, 63.0, 1.0, 2.0, 10.0, 5.0, 32.0, 7.0];
            let ctx: [*const f32; 1] = [buf.as_ptr()];

            let exec = unsafe { ExecutableCode::from_code(&c).expect("mmap") };
            let out = unsafe {
                let f: G = exec.as_fn();
                let r = f(ctx.as_ptr(), _mm256_loadu_ps(idx.as_ptr()));
                let mut out = [0.0f32; 8];
                _mm256_storeu_ps(out.as_mut_ptr(), r);
                out
            };

            for i in 0..8 {
                let want = buf[idx[i] as usize];
                assert_eq!(out[i], want, "gather lane {i}: idx {}", idx[i]);
            }
        }
    }
}

// =============================================================================
// The AVX2 `IsaBackend` driver
// =============================================================================

/// The AVX2 half of code generation.
///
/// **This file is where AVX2-specific bugs live, and the only place they
/// can.** Emission is a pure function into `Vec<u8>`, so everything here
/// compiles, typechecks and is swept for op coverage on every host, whatever
/// CPU it has. Only [`Native`](super::super::Native) decides which backend a
/// build instantiates, and only [`executable`](super::super::executable) needs
/// the matching hardware.
///
/// The consequence worth stating: a change that does not touch an ISA file
/// cannot introduce a platform-specific bug. That is the bargain `unsafe`
/// makes — confine what cannot be checked, so the rest is checked by
/// construction.
///
/// Dead only in a build that selected a *different* `Native`. The condition
/// mirrors this backend's `Native` alias, so a genuinely unused item in the
/// backend this build actually compiles still trips `dead_code`; an
/// unconditional allow here would hide it from CI's `clippy -D warnings`.
#[cfg_attr(
    not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512f")
    )),
    allow(dead_code)
)]
pub(crate) mod driver {
    use super::super::*;
    use super::{Mem, NoDisp, frame_slot};
    use crate::emit::x86_64 as x86;
    use crate::emit::x86_64::driver::SSE2_FILE;
    use crate::error::CompileError;
    use alloc::vec::Vec;
    use pixelflow_ir::kind::OpKind;

    /// The AVX2 register file (ymm, 256-bit).
    ///
    /// Same register roles as SSE2 (ymm0-15 is the same physical file as xmm0-15)
    /// at twice the width, but with a pool of **four**, two fewer than SSE2's six.
    /// AVX2's gather splits into 128-bit halves and so needs two scratch registers
    /// beyond the pair SSE2 uses (ymm13/14) to hold the high-half indices and the
    /// high-half result across the recombine. Those live in ymm8/ymm9, which must
    /// therefore sit OUTSIDE the allocator's range: with six allocatable (ymm4-9)
    /// the allocator could hand `dst` or `idx` an ymm8/9 that the gather then
    /// overwrites mid-sequence, silently returning wrong lanes — reachable
    /// whenever five values stay live across a gather.
    ///
    /// The cost is more spilling in AVX2 kernels generally, to fix a bug on the
    /// gather path specifically. Spilling the two half-temporaries to the red zone
    /// instead would restore the sixth register; that is a contained change to
    /// `super::emit_gather_scalar` and is the better long-term fix.
    const AVX2_FILE: regalloc::RegisterFile = regalloc::RegisterFile {
        // ymm4-7. ymm8/ymm9 carry the gather's high half and ymm14/ymm15 its
        // low half and the unary temp, so a sixteen-register file leaves four.
        scratch: regalloc::RegSet::range(4, 4),
        // ymm13: outside the allocatable range and the reload pair; the AVX2
        // select is a VEX blend with no internal temp.
        select_reload: Reg(13),
        // ymm15: `emit_unary`'s sign-mask temp.
        fixed: &[
            super::UNARY_SCRATCH,
            x86_64::GATHER_VALUE,
            x86_64::GATHER_IDX,
            Reg(8),
            Reg(9),
        ],
        vector_bytes: 32,
        ..SSE2_FILE
    }
    .checked();
    /// AVX2 implementation of the shared driver's leaf operations.
    pub(crate) struct Avx2Backend {
        file: regalloc::RegisterFile,
    }

    impl Avx2Backend {
        pub(crate) fn new(ctx: EmitCtx) -> Self {
            Self {
                file: AVX2_FILE.capped(ctx.max_regs),
            }
        }

        fn reload(code: &mut Vec<u8>, reload: &Reload) {
            match reload {
                Reload::FromStack { target, offset } => {
                    super::emit_load(code, *target, frame_slot(*offset));
                }
                Reload::Const { target, val_bits } => {
                    super::emit_const(code, *target, f32::from_bits(*val_bits));
                }
            }
        }
    }

    impl IsaBackend for Avx2Backend {
        type Branch = usize;

        fn register_file(&self) -> regalloc::RegisterFile {
            self.file
        }

        fn begin(&mut self, _schedule: &[regalloc::Def]) -> Result<(), CompileError> {
            Ok(()) // const broadcast is self-contained; no pool.
        }

        fn emit_plan(
            &mut self,
            code: &mut Vec<u8>,
            plan: &InstructionPlan,
        ) -> Result<(), CompileError> {
            for r in &plan.reloads {
                Self::reload(code, r);
            }
            if let Some((dst, src)) = plan.setup_mov {
                super::emit_mov(code, dst, src);
            }
            match &plan.op {
                ResolvedOp::Nop => {}
                ResolvedOp::LoadConst { dst, val_bits } => {
                    super::emit_const(code, *dst, f32::from_bits(*val_bits));
                }
                ResolvedOp::Unary { op, dst, src } => {
                    super::emit_unary(code, *op, *dst, *src);
                }
                ResolvedOp::ShiftImm {
                    op,
                    dst,
                    src,
                    amount,
                } => {
                    super::emit_shift_imm(code, *op, *dst, *src, *amount);
                }
                ResolvedOp::Gather { dst, idx, slot } => {
                    // Context pointer (array of buffer base pointers) arrives in
                    // rdi; arithmetic/const emit never touches rdi, so it
                    // survives to here. ymm13/14 mirror X86Backend's gather
                    // scratch; ymm8/9 are the AVX2-only high-half scratch this
                    // two-half gather needs (see `super::emit_gather_scalar`).
                    // ymm8/9 are non-allocatable by construction — see
                    // `AVX2_SCHED_NUM_REGS`, which caps the pool at ymm4-7 so the
                    // allocator can never place `dst`/`idx` where this clobbers.
                    super::emit_gather_scalar(
                        code,
                        *dst,
                        *idx,
                        *slot,
                        super::GatherScratch {
                            half: x86_64::GatherScratch {
                                base_gpr: 0,  // rax
                                index_gpr: 1, // rcx
                                ctx_gpr: 7,   // rdi
                                idx_lanes: x86_64::GATHER_IDX,
                                value: x86_64::GATHER_VALUE,
                            },
                            idx_hi: Reg(9),
                            res_hi: Reg(8),
                        },
                    );
                }
                ResolvedOp::Binary {
                    op,
                    dst,
                    left,
                    right,
                } => {
                    // VEX 3-operand: no two-operand hazard, emit directly.
                    super::emit_binary(code, *op, *dst, *left, *right);
                }
                ResolvedOp::FusedMulAdd { dst, a, b } => {
                    super::emit_fmadd_c_in_dst(code, *dst, *a, *b);
                }
                ResolvedOp::DecomposedMulAdd {
                    dst,
                    a,
                    b,
                    c,
                    c_deferred,
                } => {
                    super::emit_binary(code, OpKind::Mul, *dst, *a, *b);
                    match c_deferred {
                        Some(DeferredReload::FromStack(off)) => {
                            super::emit_load(code, *c, frame_slot(*off));
                        }
                        Some(DeferredReload::Const(bits)) => {
                            super::emit_const(code, *c, f32::from_bits(*bits));
                        }
                        None => {}
                    }
                    super::emit_binary(code, OpKind::Add, *dst, *dst, *c);
                }
                ResolvedOp::Select {
                    dst,
                    if_true,
                    if_false,
                } => {
                    // setup_mov already placed the vector mask in dst.
                    super::emit_select(code, *dst, *if_true, *if_false);
                }
            }
            if let Some(store) = &plan.store {
                super::emit_store(code, frame_slot(store.offset), store.src);
            }
            Ok(())
        }

        fn emit_mov(&mut self, code: &mut Vec<u8>, dst: Reg, src: Reg) {
            super::emit_mov(code, dst, src);
        }

        fn emit_store(
            &mut self,
            code: &mut Vec<u8>,
            src: Reg,
            offset: u32,
        ) -> Result<(), CompileError> {
            super::emit_store(code, frame_slot(offset), src);
            Ok(())
        }

        fn emit_resolve(
            &mut self,
            code: &mut Vec<u8>,
            vid: regalloc::ValueId,
            target: Reg,
            locs: &[Option<Loc>],
        ) -> Reg {
            match location_of(locs, vid) {
                Loc::Reg(reg) => reg,
                Loc::Remat(bits) => {
                    super::emit_const(code, target, f32::from_bits(bits));
                    target
                }
                Loc::Spill(offset) => {
                    super::emit_load(code, target, frame_slot(offset));
                    target
                }
            }
        }

        // Select short-circuit guards: vmovmskps -> eax[7:0], same shape as
        // X86Backend's MOVMSKPS guards but 8 lanes wide (al == 0xFF for
        // all-true, not 0x0F — see `super::emit_cmp_al_imm8`'s doc for why the
        // sign-extending `cmp eax, imm8` X86Backend uses doesn't work here).
        fn emit_skip_if_all_false(&mut self, code: &mut Vec<u8>, mask_reg: Reg) -> usize {
            super::emit_movmskps_eax(code, mask_reg);
            x86_64::emit_test_eax(code);
            x86_64::je(code).field() // ZF set when eax == 0 (all lanes false)
        }

        fn emit_skip_if_all_true(&mut self, code: &mut Vec<u8>, mask_reg: Reg) -> usize {
            super::emit_movmskps_eax(code, mask_reg);
            super::emit_cmp_al_imm8(code, 0xFF);
            x86_64::je(code).field() // ZF set when al == 0xFF (all lanes true)
        }

        fn emit_jump(&mut self, code: &mut Vec<u8>) -> usize {
            x86_64::emit_jmp_rel32(code)
        }
        fn patch_branch(&mut self, code: &mut Vec<u8>, branch: usize, target: usize) {
            x86_64::patch_rel32(code, branch, target);
        }

        // Same scaffold register roles as SSE2 — see `x86_64::scaffold` — at
        // this vector width. Unlike SSE2 there is no red-zone mode: the body
        // always spills into an allocated frame, and the scaffold's coordinate
        // slots sit above it.

        fn frame_alloc(&mut self, code: &mut Vec<u8>, bytes: u32) {
            x86::emit_sub_rsp(code, bytes);
        }

        fn frame_free(&mut self, code: &mut Vec<u8>, bytes: u32) {
            x86::emit_add_rsp(code, bytes);
        }

        fn slot_store(&mut self, code: &mut Vec<u8>, src: Reg, offset: u32) {
            super::emit_store(code, frame_slot(offset), src);
        }

        fn slot_load(&mut self, code: &mut Vec<u8>, dst: Reg, offset: u32) {
            super::emit_load(code, dst, frame_slot(offset));
        }

        fn latch_bounds(&mut self, code: &mut Vec<u8>) {
            x86::scaffold::latch_bounds(code);
        }

        fn counter_clear(&mut self, code: &mut Vec<u8>, counter: Counter) {
            x86::scaffold::counter_clear(code, counter);
        }

        fn counter_step(&mut self, code: &mut Vec<u8>, counter: Counter) {
            x86::scaffold::counter_step(code, counter);
        }

        fn branch_if_counter_done(&mut self, code: &mut Vec<u8>, counter: Counter) -> usize {
            x86::scaffold::branch_if_counter_done(code, counter)
        }

        fn store_result(&mut self, code: &mut Vec<u8>, src: Reg) {
            super::emit_store(
                code,
                Mem {
                    base: x86::scaffold::OUT_PTR,
                    disp: NoDisp,
                },
                src,
            );
        }

        fn advance_out(&mut self, code: &mut Vec<u8>, step: OutStep) {
            x86::scaffold::advance_out(code, step, self.file.vector_bytes);
        }

        fn add_scalar(&mut self, code: &mut Vec<u8>, dst: Reg, scratch: Reg, scalar: f32) {
            super::emit_const(code, scratch, scalar);
            super::emit_binary(code, OpKind::Add, dst, dst, scratch);
        }

        fn emit_ret(&mut self, code: &mut Vec<u8>) {
            x86::ret(code);
        }
    }
}
