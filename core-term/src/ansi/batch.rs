// src/ansi/batch.rs

//! The parser's output: one PTY read as text runs interleaved with commands.

use super::commands::AnsiCommand;

/// One parsed chunk of PTY output.
///
/// Denotes the sequence `text₀ cmd₀ text₁ cmd₁ … textₙ`: printable text,
/// interrupted by the commands that were found between its runs. It is stored
/// as two arrays rather than one entry per character — every printable
/// character lands in a single UTF-8 buffer, and each command records the
/// offset into that buffer at which it occurred. A run of text therefore
/// costs its bytes, not a 32-byte command per character.
///
/// The fields are private so the ordering invariant (offsets non-decreasing,
/// within `text`, on character boundaries) holds by construction: the only
/// writers are `push_char`, `push_str` and `push_command`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct AnsiBatch {
    text: String,
    commands: Vec<(usize, AnsiCommand)>,
}

/// Receives a batch's contents, in order.
pub trait AnsiSink {
    /// A run of printable text. Never empty.
    fn text(&mut self, run: &str);
    /// A command that occurred after all text delivered so far.
    fn command(&mut self, command: AnsiCommand);
}

impl AnsiBatch {
    /// Whether the batch holds neither text nor commands.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.is_empty() && self.commands.is_empty()
    }

    /// Delivers the batch to `sink` in order and leaves it empty, keeping its
    /// allocations for reuse.
    pub fn drain_into(&mut self, sink: &mut impl AnsiSink) {
        let mut delivered = 0;
        for (offset, command) in self.commands.drain(..) {
            if offset > delivered {
                sink.text(&self.text[delivered..offset]);
                delivered = offset;
            }
            sink.command(command);
        }
        if self.text.len() > delivered {
            sink.text(&self.text[delivered..]);
        }
        self.text.clear();
    }

    pub(super) fn reserve_text(&mut self, additional: usize) {
        self.text.reserve(additional);
    }

    pub(super) fn push_char(&mut self, c: char) {
        self.text.push(c);
    }

    pub(super) fn push_str(&mut self, run: &str) {
        self.text.push_str(run);
    }

    pub(super) fn push_command(&mut self, command: AnsiCommand) {
        self.commands.push((self.text.len(), command));
    }
}
