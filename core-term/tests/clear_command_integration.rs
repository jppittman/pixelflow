use core_term::ansi::{AnsiCommand, AnsiParser, AnsiProcessor, AnsiSink};
use core_term::io::pty::{NixPty, PtyConfig};
use core_term::term::{EmulatorInput, TerminalEmulator};
use std::io::{ErrorKind, Read};
use std::thread;
use std::time::{Duration, Instant};

/// Feeds a batch to the emulator the way the app does.
struct Emulator<'a>(&'a mut TerminalEmulator);

impl AnsiSink for Emulator<'_> {
    fn text(&mut self, run: &str) {
        self.0.print_text(run);
    }

    fn command(&mut self, command: AnsiCommand) {
        drop(self.0.interpret_input(EmulatorInput::Ansi(command)));
    }
}

#[test]
fn clear_command_emits_an_erase_sequence_when_parent_term_is_dumb() {
    // App launchers can supply TERM=dumb. CoreTerm must advertise its own
    // capabilities to the child instead of inheriting the launcher's terminal.
    std::env::set_var("TERM", "dumb");

    let config = PtyConfig {
        command_executable: "clear",
        args: &[],
        initial_cols: 80,
        initial_rows: 24,
        working_directory: None,
    };
    let mut pty =
        NixPty::spawn_with_config(&config).expect("failed to spawn clear in CoreTerm PTY");
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut output = Vec::new();
    let mut buffer = [0_u8; 256];

    loop {
        match pty.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => output.extend_from_slice(&buffer[..read]),
            Err(error) if error.kind() == ErrorKind::WouldBlock && Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(10));
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                panic!("clear did not exit within 5 seconds; output: {output:?}");
            }
            Err(error) => panic!("failed reading clear output: {error}; output: {output:?}"),
        }
    }

    assert!(
        output.starts_with(b"\x1b["),
        "clear must emit a terminal control sequence, got: {}",
        String::from_utf8_lossy(&output)
    );

    let mut parser = AnsiProcessor::new();
    let mut emulator = TerminalEmulator::new(80, 24);
    parser
        .process_bytes(b"visible text")
        .drain_into(&mut Emulator(&mut emulator));
    assert_eq!(
        emulator
            .get_render_snapshot()
            .expect("missing seeded terminal snapshot")
            .lines[0]
            .cells[0]
            .display_char(),
        'v'
    );

    parser
        .process_bytes(&output)
        .drain_into(&mut Emulator(&mut emulator));
    let snapshot = emulator
        .get_render_snapshot()
        .expect("clear did not produce a terminal snapshot");
    assert!(
        snapshot
            .lines
            .iter()
            .flat_map(|line| line.cells.iter())
            .all(|cell| cell.display_char() == ' '),
        "clear output reached the emulator but left visible cells behind"
    );
}
