//! EVEX-encoding proof for the forthcoming AVX-512 JIT backend.
//!
//! Before building the full 512-bit backend (zmm register file, k-mask register
//! class, `__m512` ABI), this validates the riskiest foundation in isolation:
//! that our hand-rolled EVEX encoder produces byte-correct AVX-512 instructions.
//!
//! Two layers of validation:
//!   1. Byte tests (always run): emitted bytes match reference encodings.
//!   2. Runtime test (only when built `+avx512f`): JIT a tiny zmm kernel and
//!      execute it on the host, checking all 16 lanes — the ground truth.
//!
//! Run the runtime test with:
//!   RUSTFLAGS="-C target-feature=+avx512f" cargo test -p pixelflow-ir \
//!     --test avx512_evex_proof

#[cfg(target_feature = "avx512f")]
use pixelflow_ir::backend::emit::executable::ExecutableCode;

// ============================================================================
// Minimal EVEX encoder (512-bit). Promoted to a real module once proven.
// ============================================================================

/// Opcode map: which escape the opcode lives in.
#[derive(Clone, Copy)]
enum Map {
    /// 0F
    M0F = 1,
    /// 0F38
    M0F38 = 2,
}

/// Mandatory prefix encoded in EVEX.pp.
#[derive(Clone, Copy)]
enum Pp {
    /// none (packed single)
    None = 0,
    /// 66
    P66 = 1,
}

/// Emit a 512-bit EVEX 3-operand reg/reg/reg instruction:
/// `op zmmDST, zmmSRC1, zmmSRC2` where SRC1 is the EVEX.vvvv non-destructive
/// source and SRC2 is the ModRM r/m. Registers may be zmm0..zmm31.
fn evex_rrr(code: &mut Vec<u8>, map: Map, pp: Pp, opcode: u8, dst: u8, src1: u8, src2: u8) {
    // EVEX stores the high register bits INVERTED.
    let r = ((dst >> 3) & 1) ^ 1; // ModRM.reg bit3
    let rp = ((dst >> 4) & 1) ^ 1; // ModRM.reg bit4 (R')
    let b = ((src2 >> 3) & 1) ^ 1; // ModRM.r/m bit3
    let x = ((src2 >> 4) & 1) ^ 1; // ModRM.r/m bit4 (EVEX.X extends r/m for reg form)
    let vvvv = (!src1) & 0x0F; // low 4 bits of vvvv, inverted
    let vp = ((src1 >> 4) & 1) ^ 1; // vvvv bit4 (V'), inverted

    let p0 = (r << 7) | (x << 6) | (b << 5) | (rp << 4) | (map as u8); // R X B R' 00 mm
    let p1 = (0 << 7) | (vvvv << 3) | (1 << 2) | (pp as u8); // W=0, vvvv, 1, pp
    let p2 = (0 << 7) | (0b10 << 5) | (0 << 4) | (vp << 3) | 0; // z=0, L'L=10 (512), b=0, V', aaa=0

    code.push(0x62);
    code.push(p0);
    code.push(p1);
    code.push(p2);
    code.push(opcode);
    code.push(0xC0 | ((dst & 7) << 3) | (src2 & 7)); // mod=11
}

fn vaddps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    evex_rrr(c, Map::M0F, Pp::None, 0x58, d, s1, s2);
}
fn vmulps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    evex_rrr(c, Map::M0F, Pp::None, 0x59, d, s1, s2);
}
/// vfmadd213ps zmmD, zmmS1, zmmS2 :  D = S1*D + S2  (213 ordering).
fn vfmadd213ps(c: &mut Vec<u8>, d: u8, s1: u8, s2: u8) {
    evex_rrr(c, Map::M0F38, Pp::P66, 0xA8, d, s1, s2);
}

// ============================================================================
// Byte-correctness tests (always run; reference encodings hand-derived).
// ============================================================================

#[test]
fn evex_addps_bytes() {
    // vaddps zmm0, zmm0, zmm1  ->  62 F1 7C 48 58 C1
    let mut c = Vec::new();
    vaddps(&mut c, 0, 0, 1);
    assert_eq!(c, [0x62, 0xF1, 0x7C, 0x48, 0x58, 0xC1], "{c:02X?}");
}

#[test]
fn evex_mulps_bytes() {
    // vmulps zmm0, zmm0, zmm2  ->  62 F1 7C 48 59 C2
    let mut c = Vec::new();
    vmulps(&mut c, 0, 0, 2);
    assert_eq!(c, [0x62, 0xF1, 0x7C, 0x48, 0x59, 0xC2], "{c:02X?}");
}

#[test]
fn evex_mulps_high_regs_bytes() {
    // vmulps zmm9, zmm20, zmm30 exercises every high-bit inversion path.
    //   reg=9  (bit3=1 -> R=0),   r/m=30 (bit3=1->B=0, bit4=1->X=0),
    //   vvvv=20 (low=4 -> ~=B; bit4=1 -> V'=0)
    // Expected: 62 [P0] [P1] [P2] 59 [modrm]
    //   P0 = R(0) X(0) B(0) R'(1) 00 mm(01) = 0001_0001 = 0x11
    //   vvvv low4 of 20 = 4 (0100) -> inverted 1011; P1 = 0 1011 1 00 = 0x5C
    //   V' = ~(20>>4 &1)=~1=0;  P2 = 0 10 0 0 000 = 0x40
    //   modrm = C0 | (9&7)<<3 | (30&7) = C0 | (1<<3) | 6 = 0xCE
    let mut c = Vec::new();
    vmulps(&mut c, 9, 20, 30);
    assert_eq!(c, [0x62, 0x11, 0x5C, 0x40, 0x59, 0xCE], "{c:02X?}");
}

#[test]
fn evex_fmadd_bytes() {
    // vfmadd213ps zmm0, zmm1, zmm2  ->  62 F2 75 48 A8 C2
    //   map=0F38 (mm=10) -> P0 = 1111_0010 = 0xF2
    //   vvvv=~1=1110; pp=66(01); P1 = 0 1110 1 01 = 0x75
    //   P2 = 0 10 0 1 000 = 0x48
    //   modrm = C0 | 0 | 2 = 0xC2
    let mut c = Vec::new();
    vfmadd213ps(&mut c, 0, 1, 2);
    assert_eq!(c, [0x62, 0xF2, 0x75, 0x48, 0xA8, 0xC2], "{c:02X?}");
}

// ============================================================================
// Runtime proof — executes a JIT'd 512-bit kernel on the host.
// Only built when the crate is compiled with AVX-512 enabled.
// ============================================================================

#[cfg(target_feature = "avx512f")]
#[test]
fn evex_runtime_16_lanes() {
    use core::arch::x86_64::*;

    // System V passes/returns __m512 in zmm0-7: X=zmm0, Y=zmm1, Z=zmm2.
    type Kernel = unsafe extern "C" fn(__m512, __m512, __m512, __m512) -> __m512;

    // Kernel A: (X + Y) * Z
    //   vaddps zmm0, zmm0, zmm1 ; vmulps zmm0, zmm0, zmm2 ; ret
    let mut a = Vec::new();
    vaddps(&mut a, 0, 0, 1);
    vmulps(&mut a, 0, 0, 2);
    a.push(0xC3);
    let code_a = unsafe { ExecutableCode::from_code(&a).expect("mmap A") };

    // Kernel B: X*Y + Z via real FMA
    //   vfmadd213ps zmm0, zmm1, zmm2  (zmm0 = zmm1*zmm0 + zmm2) ; ret
    let mut b = Vec::new();
    vfmadd213ps(&mut b, 0, 1, 2);
    b.push(0xC3);
    let code_b = unsafe { ExecutableCode::from_code(&b).expect("mmap B") };

    unsafe {
        let fa: Kernel = code_a.as_fn();
        let fb: Kernel = code_b.as_fn();

        // 16 distinct lanes so we'd catch a width/lane bug.
        let mut xs = [0.0f32; 16];
        let mut ys = [0.0f32; 16];
        let mut zs = [0.0f32; 16];
        for i in 0..16 {
            xs[i] = i as f32;
            ys[i] = (2 * i) as f32 + 1.0;
            zs[i] = (i as f32) * 0.5 - 3.0;
        }
        let x = _mm512_loadu_ps(xs.as_ptr());
        let y = _mm512_loadu_ps(ys.as_ptr());
        let z = _mm512_loadu_ps(zs.as_ptr());
        let zero = _mm512_setzero_ps();

        let ra = fa(x, y, z, zero);
        let rb = fb(x, y, z, zero);
        let mut oa = [0.0f32; 16];
        let mut ob = [0.0f32; 16];
        _mm512_storeu_ps(oa.as_mut_ptr(), ra);
        _mm512_storeu_ps(ob.as_mut_ptr(), rb);

        for i in 0..16 {
            let want_a = (xs[i] + ys[i]) * zs[i];
            let want_b = xs[i] * ys[i] + zs[i];
            assert!((oa[i] - want_a).abs() <= 1e-4, "lane {i}: (X+Y)*Z = {} want {}", oa[i], want_a);
            assert!((ob[i] - want_b).abs() <= 1e-4, "lane {i}: X*Y+Z = {} want {}", ob[i], want_b);
        }
    }
}
