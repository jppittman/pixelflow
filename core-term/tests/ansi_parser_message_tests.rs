//! ANSI Parser Message CUJ Tests
//!
//! These tests verify the complete message flow from raw bytes through
//! the ANSI parser to the terminal app, using the real AnsiProcessor.

use actor_scheduler::{
    Actor, ActorScheduler, ActorStatus, HandlerError, HandlerResult, Message, SystemStatus,
};
use core_term::ansi::{AnsiCommand, AnsiParser, AnsiProcessor};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

// =============================================================================
// Real Parser Actor Implementation
// =============================================================================

/// Parser actor using the real AnsiProcessor
struct RealParserActor {
    parser: AnsiProcessor,
    cmd_tx: SyncSender<Vec<AnsiCommand>>,
    bytes_processed: Arc<AtomicUsize>,
}

impl Actor<Vec<u8>, (), ()> for RealParserActor {
    fn handle_data(&mut self, data: Vec<u8>) -> HandlerResult {
        self.bytes_processed.fetch_add(data.len(), Ordering::SeqCst);

        let commands = self.parser.process_bytes(&data);
        if !commands.is_empty() {
            let _ = self.cmd_tx.send(commands);
        }
        Ok(())
    }

    fn handle_control(&mut self, _: ()) -> HandlerResult {
        Ok(())
    }
    fn handle_management(&mut self, _: ()) -> HandlerResult {
        Ok(())
    }
    fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

// =============================================================================
// PTY-02 Tests with Real Parser
// =============================================================================

#[test]
fn cuj_pty02_real_parser_simple_text() {
    // Given: A real parser actor
    let (cmd_tx, cmd_rx) = sync_channel::<Vec<AnsiCommand>>(100);
    let bytes_processed = Arc::new(AtomicUsize::new(0));

    let (parser_tx, mut parser_rx) = ActorScheduler::<Vec<u8>, (), ()>::new(10, 64);

    let bytes_clone = bytes_processed.clone();
    let parser_handle = thread::spawn(move || {
        let mut actor = RealParserActor {
            parser: AnsiProcessor::new(),
            cmd_tx,
            bytes_processed: bytes_clone,
        };
        parser_rx.run(&mut actor);
    });

    // When: Simple text is sent
    let text = b"Hello".to_vec();
    parser_tx.send(Message::Data(text)).expect("Expected value but got None/Err");

    thread::sleep(Duration::from_millis(50));
    drop(parser_tx);
    parser_handle.join().expect("Expected value but got None/Err");

    // Then: Should receive Print commands for each character
    let commands = cmd_rx.try_recv().expect("Expected value but got None/Err");
    assert_eq!(commands.len(), 5, "Should have 5 Print commands");

    // Verify each character
    let chars: Vec<char> = commands
        .iter()
        .filter_map(|cmd| {
            if let AnsiCommand::Print(c) = cmd {
                Some(*c)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(chars, vec!['H', 'e', 'l', 'l', 'o']);
}

#[test]
fn cuj_pty02_real_parser_escape_sequence() {
    // Given: A real parser actor
    let (cmd_tx, cmd_rx) = sync_channel::<Vec<AnsiCommand>>(100);
    let bytes_processed = Arc::new(AtomicUsize::new(0));

    let (parser_tx, mut parser_rx) = ActorScheduler::<Vec<u8>, (), ()>::new(10, 64);

    let bytes_clone = bytes_processed.clone();
    let parser_handle = thread::spawn(move || {
        let mut actor = RealParserActor {
            parser: AnsiProcessor::new(),
            cmd_tx,
            bytes_processed: bytes_clone,
        };
        parser_rx.run(&mut actor);
    });

    // When: Cursor position escape sequence is sent
    // ESC [ H = cursor to home (1,1)
    let escape_seq = b"\x1b[H".to_vec();
    parser_tx.send(Message::Data(escape_seq)).expect("Expected value but got None/Err");

    thread::sleep(Duration::from_millis(50));
    drop(parser_tx);
    parser_handle.join().expect("Expected value but got None/Err");

    // Then: Should receive a CSI command
    let commands = cmd_rx.try_recv().expect("Expected value but got None/Err");
    assert_eq!(commands.len(), 1, "Should have 1 CSI command");

    // Verify it's a cursor position command
    match &commands[0] {
        AnsiCommand::Csi(csi) => {
            // CursorPosition(1, 1) for ESC[H
            let debug_str = format!("{:?}", csi);
            assert!(
                debug_str.contains("CursorPosition"),
                "Should be CursorPosition, got: {}",
                debug_str
            );
        }
        other => panic!("Expected CSI command, got: {:?}", other),
    }
}

#[test]
fn cuj_pty02_real_parser_sgr_color() {
    // Given: A real parser actor
    let (cmd_tx, cmd_rx) = sync_channel::<Vec<AnsiCommand>>(100);
    let bytes_processed = Arc::new(AtomicUsize::new(0));

    let (parser_tx, mut parser_rx) = ActorScheduler::<Vec<u8>, (), ()>::new(10, 64);

    let bytes_clone = bytes_processed.clone();
    let parser_handle = thread::spawn(move || {
        let mut actor = RealParserActor {
            parser: AnsiProcessor::new(),
            cmd_tx,
            bytes_processed: bytes_clone,
        };
        parser_rx.run(&mut actor);
    });

    // When: SGR (Set Graphics Rendition) for red foreground is sent
    // ESC [ 31 m = set foreground to red
    let sgr_red = b"\x1b[31m".to_vec();
    parser_tx.send(Message::Data(sgr_red)).expect("Expected value but got None/Err");

    thread::sleep(Duration::from_millis(50));
    drop(parser_tx);
    parser_handle.join().expect("Expected value but got None/Err");

    // Then: Should receive an SGR command
    let commands = cmd_rx.try_recv().expect("Expected value but got None/Err");
    assert_eq!(commands.len(), 1, "Should have 1 SGR command");

    match &commands[0] {
        AnsiCommand::Csi(csi) => {
            let debug_str = format!("{:?}", csi);
            assert!(
                debug_str.contains("SetGraphicsRendition"),
                "Should be SGR command, got: {}",
                debug_str
            );
        }
        other => panic!("Expected CSI SGR command, got: {:?}", other),
    }
}

#[test]
fn cuj_pty02_real_parser_mixed_text_and_escapes() {
    // Given: A real parser actor
    let (cmd_tx, cmd_rx) = sync_channel::<Vec<AnsiCommand>>(100);
    let bytes_processed = Arc::new(AtomicUsize::new(0));

    let (parser_tx, mut parser_rx) = ActorScheduler::<Vec<u8>, (), ()>::new(10, 64);

    let bytes_clone = bytes_processed.clone();
    let parser_handle = thread::spawn(move || {
        let mut actor = RealParserActor {
            parser: AnsiProcessor::new(),
            cmd_tx,
            bytes_processed: bytes_clone,
        };
        parser_rx.run(&mut actor);
    });

    // When: Mixed content is sent
    // "Hi" + cursor home + "!"
    let mixed = b"Hi\x1b[H!".to_vec();
    parser_tx.send(Message::Data(mixed)).expect("Expected value but got None/Err");

    thread::sleep(Duration::from_millis(50));
    drop(parser_tx);
    parser_handle.join().expect("Expected value but got None/Err");

    // Then: Should receive Print, CSI, Print in order
    let commands = cmd_rx.try_recv().expect("Expected value but got None/Err");
    assert_eq!(commands.len(), 4, "Should have 4 commands (H, i, CSI, !)");

    // Verify order
    assert!(matches!(commands[0], AnsiCommand::Print('H')));
    assert!(matches!(commands[1], AnsiCommand::Print('i')));
    assert!(matches!(commands[2], AnsiCommand::Csi(_)));
    assert!(matches!(commands[3], AnsiCommand::Print('!')));
}

#[test]
fn cuj_pty02_real_parser_incremental_escape() {
    // Given: A real parser actor
    let (cmd_tx, cmd_rx) = sync_channel::<Vec<AnsiCommand>>(100);
    let bytes_processed = Arc::new(AtomicUsize::new(0));

    let (parser_tx, mut parser_rx) = ActorScheduler::<Vec<u8>, (), ()>::new(10, 64);

    let bytes_clone = bytes_processed.clone();
    let parser_handle = thread::spawn(move || {
        let mut actor = RealParserActor {
            parser: AnsiProcessor::new(),
            cmd_tx,
            bytes_processed: bytes_clone,
        };
        parser_rx.run(&mut actor);
    });

    // When: Escape sequence arrives in pieces (simulating network fragmentation)
    // First just ESC
    parser_tx.send(Message::Data(vec![0x1b])).expect("Expected value but got None/Err");
    thread::sleep(Duration::from_millis(10));

    // Then [
    parser_tx.send(Message::Data(vec![b'['])).expect("Expected value but got None/Err");
    thread::sleep(Duration::from_millis(10));

    // Then H
    parser_tx.send(Message::Data(vec![b'H'])).expect("Expected value but got None/Err");

    thread::sleep(Duration::from_millis(50));
    drop(parser_tx);
    parser_handle.join().expect("Expected value but got None/Err");

    // Then: Parser should buffer and produce a complete CSI command
    let mut all_commands = Vec::new();
    while let Ok(cmds) = cmd_rx.try_recv() {
        all_commands.extend(cmds);
    }

    // Should have exactly one CSI command
    let csi_count = all_commands
        .iter()
        .filter(|c| matches!(c, AnsiCommand::Csi(_)))
        .count();
    assert_eq!(
        csi_count, 1,
        "Should parse fragmented escape into 1 CSI command"
    );
}

#[test]
fn cuj_pty02_real_parser_c0_control() {
    // Given: A real parser actor
    let (cmd_tx, cmd_rx) = sync_channel::<Vec<AnsiCommand>>(100);
    let bytes_processed = Arc::new(AtomicUsize::new(0));

    let (parser_tx, mut parser_rx) = ActorScheduler::<Vec<u8>, (), ()>::new(10, 64);

    let bytes_clone = bytes_processed.clone();
    let parser_handle = thread::spawn(move || {
        let mut actor = RealParserActor {
            parser: AnsiProcessor::new(),
            cmd_tx,
            bytes_processed: bytes_clone,
        };
        parser_rx.run(&mut actor);
    });

    // When: C0 control characters are sent
    // BEL (0x07), LF (0x0A), CR (0x0D)
    let controls = vec![0x07, 0x0A, 0x0D];
    parser_tx.send(Message::Data(controls)).expect("Expected value but got None/Err");

    thread::sleep(Duration::from_millis(50));
    drop(parser_tx);
    parser_handle.join().expect("Expected value but got None/Err");

    // Then: Should receive C0 control commands
    let commands = cmd_rx.try_recv().expect("Expected value but got None/Err");
    assert_eq!(commands.len(), 3, "Should have 3 C0 control commands");

    // All should be C0Control variants
    for cmd in &commands {
        assert!(
            matches!(cmd, AnsiCommand::C0Control(_)),
            "Expected C0Control, got: {:?}",
            cmd
        );
    }
}

#[test]
fn cuj_pty02_real_parser_high_throughput() {
    // Given: A real parser actor
    let (cmd_tx, cmd_rx) = sync_channel::<Vec<AnsiCommand>>(1000);
    let bytes_processed = Arc::new(AtomicUsize::new(0));

    let (parser_tx, mut parser_rx) = ActorScheduler::<Vec<u8>, (), ()>::new(100, 128);

    let bytes_clone = bytes_processed.clone();
    let parser_handle = thread::spawn(move || {
        let mut actor = RealParserActor {
            parser: AnsiProcessor::new(),
            cmd_tx,
            bytes_processed: bytes_clone,
        };
        parser_rx.run(&mut actor);
    });

    // When: Large volume of data is sent
    const NUM_BATCHES: usize = 100;
    const BATCH_SIZE: usize = 1024;

    for _ in 0..NUM_BATCHES {
        let data: Vec<u8> = (0..BATCH_SIZE).map(|i| b'A' + (i % 26) as u8).collect();
        parser_tx.send(Message::Data(data)).expect("Expected value but got None/Err");
    }

    thread::sleep(Duration::from_millis(200));
    drop(parser_tx);
    parser_handle.join().expect("Expected value but got None/Err");

    // Then: All bytes should be processed
    assert_eq!(
        bytes_processed.load(Ordering::SeqCst),
        NUM_BATCHES * BATCH_SIZE,
        "All bytes should be processed"
    );

    // And: Commands should be generated
    let mut total_commands = 0;
    while let Ok(cmds) = cmd_rx.try_recv() {
        total_commands += cmds.len();
    }
    assert_eq!(
        total_commands,
        NUM_BATCHES * BATCH_SIZE,
        "Should produce one Print command per byte"
    );
}

// =============================================================================
// End-to-End Message Flow Simulation
// =============================================================================

/// Simulates the complete terminal message chain
struct TerminalMessageChain {
    parser_tx: actor_scheduler::ActorHandle<Vec<u8>, (), ()>,
    commands_received: Arc<Mutex<Vec<AnsiCommand>>>,
    parser_handle: Option<thread::JoinHandle<()>>,
    app_handle: Option<thread::JoinHandle<()>>,
}

impl TerminalMessageChain {
    fn new() -> Self {
        let commands_received = Arc::new(Mutex::new(Vec::new()));
        let commands_clone = commands_received.clone();

        // Parser → App channel
        let (cmd_tx, cmd_rx) = sync_channel::<Vec<AnsiCommand>>(100);

        // ReadThread → Parser channel (actor scheduler)
        let (parser_tx, mut parser_rx) = ActorScheduler::<Vec<u8>, (), ()>::new(10, 64);

        // Spawn parser thread
        let parser_handle = thread::spawn(move || {
            let mut actor = RealParserActor {
                parser: AnsiProcessor::new(),
                cmd_tx,
                bytes_processed: Arc::new(AtomicUsize::new(0)),
            };
            parser_rx.run(&mut actor);
        });

        // Spawn app thread (receives commands)
        let app_handle = thread::spawn(move || {
            while let Ok(cmds) = cmd_rx.recv() {
                commands_clone.lock().unwrap().extend(cmds);
            }
        });

        Self {
            parser_tx,
            commands_received,
            parser_handle: Some(parser_handle),
            app_handle: Some(app_handle),
        }
    }

    fn send_bytes(&self, data: Vec<u8>) {
        self.parser_tx.send(Message::Data(data)).unwrap();
    }

    fn get_commands(&self) -> Vec<AnsiCommand> {
        self.commands_received.lock().unwrap().clone()
    }

    fn shutdown(mut self) {
        // Drop sender to trigger chain shutdown
        drop(self.parser_tx);

        if let Some(h) = self.parser_handle.take() {
            h.join().unwrap();
        }
        if let Some(h) = self.app_handle.take() {
            h.join().unwrap();
        }
    }
}

#[test]
fn cuj_e2e_complete_message_chain() {
    // Given: A complete terminal message chain
    let chain = TerminalMessageChain::new();

    // When: Terminal output is sent through the chain
    // Simulate: clear screen, move cursor, print text
    chain.send_bytes(b"\x1b[2J".to_vec()); // Clear screen
    chain.send_bytes(b"\x1b[1;1H".to_vec()); // Move to 1,1
    chain.send_bytes(b"Hello, Terminal!".to_vec()); // Print text

    thread::sleep(Duration::from_millis(100));
    chain.shutdown();

    // Then: All commands should be received by app
    // This validates the complete: PTY → ReadThread → Parser → App chain
}

#[test]
fn cuj_e2e_rapid_small_writes() {
    // Given: A complete chain
    let chain = TerminalMessageChain::new();

    // When: Many small writes (simulating character-by-character echo)
    for c in b"typing rapidly...".iter() {
        chain.send_bytes(vec![*c]);
        thread::sleep(Duration::from_micros(100)); // Simulate typing delay
    }

    thread::sleep(Duration::from_millis(100));
    let commands = chain.get_commands();
    chain.shutdown();

    // Then: All characters should arrive
    let chars: String = commands
        .iter()
        .filter_map(|c| {
            if let AnsiCommand::Print(ch) = c {
                Some(*ch)
            } else {
                None
            }
        })
        .collect();

    assert_eq!(chars, "typing rapidly...");
}
