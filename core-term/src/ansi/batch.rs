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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ansi::commands::C0Control;

    /// A sink that records what it was handed and enforces `AnsiSink::text`'s
    /// own documented contract: a delivered run is never empty.
    #[derive(Default)]
    struct Recording {
        texts: Vec<String>,
        commands: Vec<AnsiCommand>,
    }

    impl AnsiSink for Recording {
        fn text(&mut self, run: &str) {
            assert!(
                !run.is_empty(),
                "AnsiSink::text must never be called with an empty run"
            );
            self.texts.push(run.to_string());
        }

        fn command(&mut self, command: AnsiCommand) {
            self.commands.push(command);
        }
    }

    fn bell() -> AnsiCommand {
        AnsiCommand::C0Control(C0Control::BEL)
    }

    #[test]
    fn it_should_report_empty_for_a_default_batch() {
        assert!(AnsiBatch::default().is_empty());
    }

    #[test]
    fn it_should_report_not_empty_once_text_is_pushed() {
        let mut batch = AnsiBatch::default();
        batch.push_str("x");
        assert!(!batch.is_empty());
    }

    #[test]
    fn it_should_report_not_empty_once_a_command_is_pushed() {
        let mut batch = AnsiBatch::default();
        batch.push_command(bell());
        assert!(!batch.is_empty());
    }

    #[test]
    fn it_should_not_deliver_an_empty_text_run_between_two_adjacent_commands() {
        let mut batch = AnsiBatch::default();
        batch.push_command(bell());
        batch.push_command(bell());

        let mut sink = Recording::default();
        batch.drain_into(&mut sink);

        assert_eq!(sink.texts, Vec::<String>::new());
        assert_eq!(sink.commands, vec![bell(), bell()]);
    }

    #[test]
    fn it_should_not_deliver_an_empty_trailing_text_run_after_the_last_command() {
        let mut batch = AnsiBatch::default();
        batch.push_str("hi");
        batch.push_command(bell());

        let mut sink = Recording::default();
        batch.drain_into(&mut sink);

        assert_eq!(sink.texts, vec!["hi".to_string()]);
        assert_eq!(sink.commands, vec![bell()]);
    }

    #[test]
    fn it_should_deliver_interleaved_text_and_commands_in_order() {
        let mut batch = AnsiBatch::default();
        batch.push_str("ab");
        batch.push_command(bell());
        batch.push_str("cd");

        let mut sink = Recording::default();
        batch.drain_into(&mut sink);

        assert_eq!(sink.texts, vec!["ab".to_string(), "cd".to_string()]);
        assert_eq!(sink.commands, vec![bell()]);
    }

    #[test]
    fn it_should_leave_the_batch_empty_after_drain_into() {
        let mut batch = AnsiBatch::default();
        batch.push_str("x");
        batch.push_command(bell());

        let mut sink = Recording::default();
        batch.drain_into(&mut sink);

        assert!(batch.is_empty());
    }
}
