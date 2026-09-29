//! Content-hash identity for persisted formats — the single mechanism that
//! replaces hand-bumped magic/version integers as the thing that gates
//! loading (docs/plans/2026-08-17-cost-model-domain.md, J9 — kills P1(b)).
//!
//! A hand-bumped integer is a promise a human has to remember to keep every
//! time a serialized field's MEANING changes, not just whenever the byte
//! layout does. The corpus format's `VERSION` comment documents exactly this
//! failure mode by naming it three times over (op renumbering, subtree
//! compaction, dedup-key change) — three bumps a human had to remember to
//! make, on top of getting the diagnosis right. `SchemaIdentity` removes the
//! separate step: the identity IS the content hash of a written description
//! of the format (`SCHEMA`), so there is no version integer to forget to
//! move. Changing what a field means without updating `SCHEMA` leaves the
//! description wrong — a review-time smell — but the identity can never
//! silently drift from the text that produced it, and a stale file can never
//! load unnoticed: the whole point is to turn "forgot to bump" into
//! "impossible to forget", per this repo's NO-SILENT-FAILURES rule.

use std::io;

/// FNV-1a 64, as a `const fn` so [`SchemaIdentity::SCHEMA_IDENTITY`] can be
/// derived at compile time from `SCHEMA`'s bytes.
///
/// The one hash every content identity in `pixelflow-pipeline` is built
/// from — schema identities here, the mint sidecar's `weights_identity`, and
/// the journal's corpus/diff/config identities — and the same function the
/// byte-identity probes digest emitted code with: it is
/// `pixelflow_codegen::fnv1a64`, imported, not restated.
pub use pixelflow_codegen::fnv1a64 as fnv1a64_const;

/// [`fnv1a64_const`], hex-formatted, for runtime content hashes (weight
/// bytes, corpus bytes, diff text) that are not compile-time constants.
#[must_use]
pub fn fnv1a64_hex(bytes: &[u8]) -> String {
    format!("{:016x}", fnv1a64_const(bytes))
}

/// A persisted artifact whose on-disk identity is *derived* from a written
/// description of its layout and field semantics, not hand-maintained beside
/// it.
///
/// # Contract
/// - `SCHEMA` enumerates this format's fields, dimensions, and what each one
///   MEANS — the description that would otherwise live only in a doc comment
///   a version bump could silently outrun.
/// - `SCHEMA_IDENTITY` is `SCHEMA`'s content hash (the default is correct for
///   every implementer; it exists as an associated const only so it can be
///   used in a `const` context such as a fixed-size header field).
/// - `MAGIC` is a short human-readable prefix, kept for `file`(1)-style
///   inspection. It is a courtesy — `SCHEMA_IDENTITY` is the gate.
///
/// Every loader must assert `stored == Self::SCHEMA_IDENTITY` and refuse
/// loudly on mismatch, via [`identity_mismatch`] or an equivalent panic that
/// names the regeneration command.
///
/// Deliberately no format-evolution ceremony: an artifact that mismatches is
/// regenerated, not migrated — these are scratch/derived files, cheaper to
/// remint than to version (docs/plans/2026-08-17-cost-model-domain.md §2).
pub trait SchemaIdentity {
    /// Short human-readable prefix. Courtesy only; never the gate.
    const MAGIC: &'static str;
    /// The written description of this format's fields and their meaning.
    /// The sole input to `SCHEMA_IDENTITY` — edit this when a field's
    /// meaning changes, and the identity changes with it, by construction.
    const SCHEMA: &'static str;
    /// Content hash of `SCHEMA`. Do not override.
    const SCHEMA_IDENTITY: u64 = fnv1a64_const(Self::SCHEMA.as_bytes());
}

/// Unix seconds now — the one clock every persisted-artifact timestamp
/// ([`crate::journal::JournalEntry::ts_unix`]) is stamped from, so two
/// artifacts cannot drift into different clocks. Lives here (unconditionally
/// compiled) so `journal` — used regardless of the `training` feature — can
/// share it.
///
/// # Panics
///
/// Panics if the system clock is before the Unix epoch.
#[must_use]
pub fn unix_now_s() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is before the unix epoch")
        .as_secs()
}

/// The standard loud-refusal error for a [`SchemaIdentity`] mismatch: names
/// what was stored, what was expected, and the command that regenerates the
/// artifact. Every `SchemaIdentity` loader should return this (or panic with
/// an equivalent message) rather than composing its own wording.
#[must_use]
pub fn identity_mismatch(kind: &str, stored: u64, expected: u64, regen_cmd: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
            "{kind} schema identity mismatch: stored {stored:016x}, expected {expected:016x}. \
             The format's meaning changed since this file was written, so reading it further \
             would silently misinterpret its bytes. Regenerate it with:\n  {regen_cmd}"
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture;
    impl SchemaIdentity for Fixture {
        const MAGIC: &'static str = "FIXT";
        const SCHEMA: &'static str = "field a: f32 kernel cost in ns; field b: u32 sample count";
    }

    #[test]
    fn identity_is_deterministic_and_nonzero() {
        assert_eq!(Fixture::SCHEMA_IDENTITY, Fixture::SCHEMA_IDENTITY);
        assert_ne!(Fixture::SCHEMA_IDENTITY, 0);
    }

    #[test]
    fn different_schema_text_yields_different_identity() {
        // Same field, different meaning: exactly the case a hand-bumped
        // integer can silently fail to reflect (P1(b)).
        const A: &str = "field a means kernel cost";
        const B: &str = "field a means kernel latency";
        assert_ne!(fnv1a64_const(A.as_bytes()), fnv1a64_const(B.as_bytes()));
    }

    #[test]
    fn fnv1a64_hex_and_fnv1a64_const_should_agree_on_the_same_bytes() {
        let bytes = b"some artifact bytes";
        assert_eq!(fnv1a64_hex(bytes), format!("{:016x}", fnv1a64_const(bytes)));
    }

    #[test]
    fn mismatch_error_names_the_regen_command_and_both_identities() {
        let e = identity_mismatch("corpus", 1, 2, "cargo run --bin gen_bench_corpus");
        let msg = e.to_string();
        assert!(msg.contains("gen_bench_corpus"), "{msg}");
        assert!(msg.contains("0000000000000001"), "{msg}");
        assert!(msg.contains("0000000000000002"), "{msg}");
    }
}
