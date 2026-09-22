//! The x86-64 probe: what each tier's encoder needs, asked of CPUID.
//!
//! Every requirement names the instructions in the emitter that need it, so
//! a new encoding in `emit/avx2.rs` or `emit/avx512.rs` reaching for another
//! extension has one place to declare it — and `xtask isa-matrix` keeps its
//! own copy of these names honest by only running a tier this probe would
//! select.

use super::Isa;

/// One CPUID feature an encoder's instructions need: the name CPUID (and
/// this crate's refusals) spell it by, and the probe for it.
/// `is_x86_feature_detected!` takes only a literal, so each feature carries
/// its own.
struct Feature {
    name: &'static str,
    present: fn() -> bool,
}

/// The floor: what the AVX2 tier (`emit/avx2.rs`) executes, every feature
/// required.
///
/// - `avx2` — the 256-bit integer forms: `vpaddd`/`vpslld`/`vpsrld ymm`
///   (`IAdd`, `ShiftImm`), `vpmovzxbd ymm, xmm` (the lane iota),
///   `vgatherdps ymm` (`Gather`), and `vbroadcastss ymm, xmm` from a
///   register (`Const`, `Uniform`), which is AVX2 where only the
///   memory-source form is AVX. AVX2 implies AVX, which is the rest of the
///   encoder: the `ymm` float arithmetic and `vfmadd`'s operands, `vcmpps`,
///   `vroundps`, `vinsertf128`/`vextractf128`, `vmovmskps`,
///   `vcvttps2dq`/`vcvtdq2ps`, and the OS having enabled `ymm` state, which
///   `is_x86_feature_detected!` checks with the bit.
/// - `fma` — `vfmadd231ps` (`emit_fmadd_c_in_dst`), the one-rounding
///   `MulAdd`. Not implied by `avx2` in CPUID any more than in rustc's feature
///   model, and no shipping CPU has ever offered one without the other
///   (Intel: both since Haswell; AMD: FMA3 predates AVX2 by a generation —
///   x86-64-v3 codifies the pairing). So AVX2-without-FMA is not a narrower
///   tier but a paper configuration; it once forked `emit_fmadd_c_in_dst`
///   into a two-rounding variant no real machine exercised, and that fork
///   put two materially different kernels under one environment fingerprint
///   (`pixelflow-pipeline/src/journal.rs`). A host lacking `fma` is refused.
const AVX2_FLOOR: &[Feature] = &[
    Feature {
        name: "avx2",
        present: || std::is_x86_feature_detected!("avx2"),
    },
    Feature {
        name: "fma",
        present: || std::is_x86_feature_detected!("fma"),
    },
];

/// What the AVX-512 tier (`emit/avx512.rs`) needs beyond the floor.
///
/// - `avx512f` — the `zmm` register file (`zmm16..31` included) and the EVEX
///   encoding of everything arithmetic; `vcmpps` into a `k` register and the
///   `kmovw`/`kortestw` that read it (`emit_compare`, a guard's all-lanes
///   test); `vptestmd`; `vpternlogd` (`Select`'s blend); `vrndscaleps`
///   (`Floor`/`Round`); `vrcp14ps`/`vrsqrt14ps`; the writemasked `vmovups`
///   store of a remainder (`emit_write`); `vgatherdps zmm`; `vpmovzxbd zmm`;
///   and `vcvttss2si`/`vmovq` in their EVEX forms.
/// - `avx512dq` — `vpmovm2d`, which widens every comparison's `k` mask back
///   into a per-lane vector (the file's header says why F alone would fault
///   on any kernel with a comparison); the EVEX forms of the float logicals
///   `vandps`/`vorps`/`vxorps` (`BitAnd`/`BitOr`, `Neg`/`Abs`, the mask
///   blend); and EVEX `vpinsrq`. All DQ, not F.
const AVX512_TIER: &[Feature] = &[
    Feature {
        name: "avx512f",
        present: || std::is_x86_feature_detected!("avx512f"),
    },
    Feature {
        name: "avx512dq",
        present: || std::is_x86_feature_detected!("avx512dq"),
    },
];

fn missing(features: &'static [Feature]) -> Option<&'static str> {
    features.iter().find(|f| !(f.present)()).map(|f| f.name)
}

/// Every tier this host can execute, widest first — or the floor feature it
/// lacks.
pub(super) fn runnable() -> Result<&'static [Isa], &'static str> {
    if let Some(feature) = missing(AVX2_FLOOR) {
        return Err(feature);
    }
    if missing(AVX512_TIER).is_none() {
        return Ok(&[Isa::Avx512, Isa::Avx2]);
    }
    Ok(&[Isa::Avx2])
}
