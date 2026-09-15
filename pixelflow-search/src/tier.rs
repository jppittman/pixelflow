//! Which compilation tier a saturation belongs to.
//!
//! Telemetry is this type's only consumer today, which made it look like a
//! telemetry detail — it lived inside the feature-gated `telemetry` module,
//! and only that module's code could name it. It is not a detail. The tier
//! decides *behavior*: a macro-tier saturation runs inside rustc's own
//! process, so a line it writes to stderr is read by cargo's
//! `--message-format=json` parser and forwarded as a compiler message; a
//! runtime-tier one writes to an ordinary process's own stream, where no such
//! collision exists. `telemetry` has to know which, and the only thing that
//! can know is the saturation itself.
//!
//! So it is compiled unconditionally and `Saturate` carries one. That is what
//! makes the tier a property of the pass rather than an argument its one
//! call site remembered to pass — the arrangement under which the macro tier,
//! on adopting the shared `Saturate`, silently inherited `Tier::Runtime` and
//! would have corrupted the JSON message stream again.

/// The compilation tier a saturation runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// Bake time: [`crate::runtime::optimize_runtime_arena`], in an ordinary
    /// process.
    Runtime,
    /// Macro-expansion time, inside rustc — `pixelflow_compiler::kernel`.
    Macro,
}

impl Tier {
    /// What must precede this tier's telemetry record on stderr.
    ///
    /// Empty for [`Self::Runtime`], whose stderr is an ordinary process's own
    /// stream: the record goes out bare and is directly JSONL-parseable.
    ///
    /// Non-empty, and load-bearing, for [`Self::Macro`]. That stderr *is*
    /// rustc's, and under `cargo --message-format=json` cargo parses each of
    /// its lines and forwards whatever parses as JSON as a
    /// `"reason":"compiler-message"` event. The record is valid JSON but not
    /// diagnostic-shaped, so a bare line gets relayed downstream as a bogus
    /// compiler message and corrupts the stream for every consumer —
    /// confirmed empirically, which is why this prefix exists. A leading
    /// non-`{` character is enough to make the line fail JSON parsing, so
    /// cargo relays it as ordinary text: still readable by a human, no longer
    /// mistakable for a diagnostic.
    #[must_use]
    pub const fn stderr_prefix(self) -> &'static str {
        match self {
            Self::Runtime => "",
            Self::Macro => "saturation-telemetry(macro): ",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Tier;

    /// The invariant [`Tier::stderr_prefix`] exists for, asserted rather than
    /// described.
    ///
    /// This is the check that was missing. Nothing in CI exercises the
    /// telemetry feature beyond compiling it, so when the macro tier adopted
    /// the shared `Saturate` pass it silently inherited `Tier::Runtime` and
    /// would have gone back to emitting a bare JSON line into rustc's stderr.
    /// A human caught it by reading. This is what catches it next time.
    #[test]
    fn only_a_runtime_record_may_lead_with_a_brace() {
        // Stands in for a JSON parser: cargo's gives up on a line whose first
        // character cannot begin a JSON object, which is the whole mechanism.
        let line = |tier: Tier| format!("{}{}", tier.stderr_prefix(), r#"{"tier":"x"}"#);

        assert!(
            !line(Tier::Macro).starts_with('{'),
            "a macro-tier record must not parse as JSON — it lands in rustc's \
             stderr, where cargo would forward it as a compiler message"
        );
        assert!(
            line(Tier::Runtime).starts_with('{'),
            "a runtime-tier record must stay directly JSONL-parseable"
        );
    }
}
