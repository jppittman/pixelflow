//! Which instruction set the JIT emits for: decided once per process, by the
//! CPU, at the first call to [`detect`].
//!
//! Every backend in [`crate::emit`] compiles on every host — emission is a
//! pure function into bytes — so the target never decided which backends
//! *exist*, only which one is *instantiated*. Until this module that choice
//! was `cfg(target_feature)`: the build's flags, which a plain `cargo build`
//! never sets, so every x86-64 host ran 128-bit SSE2 kernels however wide the
//! machine was, and each wider tier was a separate build of the whole
//! workspace that only `xtask isa-matrix` ever made. Now the choice is
//! CPUID's: the widest tier this host can execute, with a floor of AVX2+FMA on
//! x86-64 (there is no SSE2 tier any more) and NEON on aarch64. A host below
//! the floor is refused, loudly, with the feature it lacks named.
//!
//! The width follows the tier ([`jit_vector_bytes`]), and nothing else in the
//! workspace holds one: `pixelflow-core` asks this module rather than
//! carrying a vector type of its own to assert against.
//!
//! `PIXELFLOW_ISA` overrides the choice *downward* — `avx2` on an AVX-512
//! host runs the 256-bit backend, which is how one machine tests both x86
//! backends — and is refused, never silently downgraded, when it names a tier
//! the host cannot run. Read once, with the detection: the tier is a fact
//! about the process, and a compile cache keyed on shape assumes it does not
//! change under it.
//!
//! See docs/plans/2026-09-22-the-isa-is-decided-at-startup.md.

use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
mod x86_64;
#[cfg(target_arch = "x86_64")]
use x86_64::runnable;
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) use x86_64::skip_unless_host_runs;

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "aarch64")]
use aarch64::runnable;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!("pixelflow-codegen emits x86-64 (AVX2, AVX-512) and aarch64 (NEON) code only");

/// The instruction-set tier the JIT emits for. One per process.
///
/// Each names one backend in [`crate::emit`] and one vector width; the two
/// x86-64 tiers are ordered, and a host that can run `Avx512` can run `Avx2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    /// x86-64, 256-bit `ymm` kernels: AVX2 with FMA3, the floor.
    Avx2,
    /// x86-64, 512-bit `zmm` kernels: AVX-512F with DQ.
    Avx512,
    /// aarch64, 128-bit NEON: the architecture's baseline, so its only tier.
    Neon,
}

impl Isa {
    /// Every tier, in the order `PIXELFLOW_ISA`'s error message lists them.
    const ALL: [Self; 3] = [Self::Avx2, Self::Avx512, Self::Neon];

    /// Bytes in one vector register of this tier: the batch the lattice's
    /// lane fold is executed by. An ISA-defined width, which is why it is a
    /// table here rather than something read off a backend — a `zmm` is 64
    /// bytes by definition, and `emit`'s tests pin each backend's register
    /// file to this.
    #[must_use]
    pub const fn vector_bytes(self) -> usize {
        match self {
            Self::Avx2 => 32,
            Self::Avx512 => 64,
            Self::Neon => 16,
        }
    }

    /// The name `PIXELFLOW_ISA` spells this tier as, and a summary prints.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
            Self::Neon => "neon",
        }
    }

    fn parse(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|isa| name.trim().eq_ignore_ascii_case(isa.name()))
    }
}

/// The environment override, for diagnosis and for CI. See [`requested`].
const OVERRIDE_VAR: &str = "PIXELFLOW_ISA";

/// The tier this process emits for.
///
/// Decided on the first call and fixed for the life of the process: the
/// widest tier the host can execute, unless `PIXELFLOW_ISA` asks for a
/// narrower one it can also execute.
///
/// # Panics
///
/// On a host below the floor (x86-64 without AVX2 or without FMA), naming the
/// missing feature; on a `PIXELFLOW_ISA` that is not a tier's name; and on one
/// that names a tier this host cannot execute. Each is a refusal rather than
/// a fallback, so a run always emits what it says it emits.
#[must_use]
pub fn detect() -> Isa {
    static ISA: OnceLock<Isa> = OnceLock::new();
    *ISA.get_or_init(|| {
        let runnable = runnable().unwrap_or_else(|missing| {
            panic!(
                "this CPU lacks `{missing}`, and the JIT has no tier below AVX2+FMA on x86-64: \
                 pixelflow emits AVX2 or AVX-512 code on x86-64 and NEON on aarch64, and \
                 nothing narrower (docs/plans/2026-09-22-the-isa-is-decided-at-startup.md)"
            )
        });
        match requested() {
            None => runnable[0],
            Some(isa) if runnable.contains(&isa) => isa,
            Some(isa) => panic!(
                "{OVERRIDE_VAR}={} names a tier this host cannot execute; it can run {}. The \
                 override is refused rather than downgraded so that a run emits exactly the \
                 tier it was asked for, or nothing.",
                isa.name(),
                names(runnable)
            ),
        }
    })
}

/// Byte width of the vector the JIT emits for on this host: 64 (AVX-512), 32
/// (AVX2) or 16 (NEON). [`detect`]'s width; the one width in the workspace.
#[must_use]
pub fn jit_vector_bytes() -> usize {
    detect().vector_bytes()
}

/// Whether this host can execute `isa`'s kernels — regardless of which tier
/// [`detect`] chose, and without the below-floor panic, since a test asking
/// whether it can run is not a compile asking for a backend.
#[must_use]
pub fn host_runs(isa: Isa) -> bool {
    runnable().is_ok_and(|tiers| tiers.contains(&isa))
}

/// `PIXELFLOW_ISA`, if set.
///
/// | value | effect |
/// |---|---|
/// | unset | the widest tier the host can execute |
/// | `avx2`, `avx512`, `neon` (case-insensitive) | that tier, if the host can execute it; refused otherwise |
/// | anything else | panic, quoting the offending value — no silent failures |
fn requested() -> Option<Isa> {
    let raw = std::env::var_os(OVERRIDE_VAR)?;
    let parsed = raw.to_str().and_then(Isa::parse);
    Some(parsed.unwrap_or_else(|| {
        panic!(
            "{OVERRIDE_VAR}={raw:?} is not one of {}. This variable can only pick a tier the \
             host can execute, never invent one — so an unrecognized value fails loudly rather \
             than picking a default.",
            names(&Isa::ALL)
        )
    }))
}

fn names(tiers: &[Isa]) -> String {
    tiers
        .iter()
        .map(|isa| format!("`{}`", isa.name()))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_host_can_execute_the_tier_it_detected() {
        assert!(host_runs(detect()));
    }

    #[test]
    fn the_width_is_the_tiers() {
        assert_eq!(jit_vector_bytes(), detect().vector_bytes());
    }

    #[test]
    fn a_name_round_trips_and_is_case_insensitive() {
        for isa in Isa::ALL {
            assert_eq!(Isa::parse(isa.name()), Some(isa));
            assert_eq!(Isa::parse(&isa.name().to_ascii_uppercase()), Some(isa));
            assert_eq!(Isa::parse(&format!(" {} ", isa.name())), Some(isa));
        }
        assert_eq!(Isa::parse("sse2"), None, "there is no SSE2 tier");
        assert_eq!(Isa::parse(""), None);
    }

    #[test]
    fn the_widest_runnable_tier_is_first() {
        let tiers = runnable().expect("this host runs the tests, so it is above the floor");
        let widest = tiers.iter().map(|isa| isa.vector_bytes()).max();
        assert_eq!(Some(tiers[0].vector_bytes()), widest);
    }
}
