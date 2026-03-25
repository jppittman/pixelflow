//! Actor roundtrip tests ensuring message delivery and processing correctness.
//!
//! These tests verify complete message roundtrips through actors:
//! - Send message → Actor processes → Verify output
//!
//! Following TDD principles: write tests first, uncover bugs, fix them.

use actor_scheduler::{
    Actor, ActorScheduler, ActorStatus, HandlerError, HandlerResult, Message, SystemStatus,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

// =============================================================================
// ParserActor Roundtrip Tests
// =============================================================================

/// Test fixture simulating the ParserActor behavior
struct TestParserActor {
    output_tx: std::sync::mpsc::SyncSender<Vec<TestAnsiCommand>>,
    bytes_processed: Arc<AtomicUsize>,
}

#[derive(Debug, Clone, PartialEq)]
enum TestAnsiCommand {
    Print(char),
    C0Newline,
    C0CarriageReturn,
    CsiCursorUp(u32),
}

impl Actor<Vec<u8>, (), ()> for TestParserActor {
    fn handle_data(&mut self, bytes: Vec<u8>) -> HandlerResult {
        self.bytes_processed
            .fetch_add(bytes.len(), Ordering::SeqCst);

        let mut commands = Vec::new();
        let mut i = 0;

        while i < bytes.len() {
            let b = bytes[i];
            match b {
                b'\n' => commands.push(TestAnsiCommand::C0Newline),
                b'\r' => commands.push(TestAnsiCommand::C0CarriageReturn),
                b'\x1b' if i + 2 < bytes.len() && bytes[i + 1] == b'[' && bytes[i + 2] == b'A' => {
                    commands.push(TestAnsiCommand::CsiCursorUp(1));
                    i += 2;
                }
                b if b.is_ascii_graphic() || b == b' ' => {
                    commands.push(TestAnsiCommand::Print(b as char));
                }
                _ => {}
            }
            i += 1;
        }

        if !commands.is_empty() {
            let _ = self.output_tx.send(commands);
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

#[test]
fn parser_roundtrip_simple_text() {
    let (output_tx, output_rx) = std::sync::mpsc::sync_channel(10);
    let bytes_processed = Arc::new(AtomicUsize::new(0));

    let (tx, mut rx) = ActorScheduler::<Vec<u8>, (), ()>::new(10, 64);

    let bytes_clone = bytes_processed.clone();
    let handle = thread::spawn(move || {
        let mut actor = TestParserActor {
            output_tx,
            bytes_processed: bytes_clone,
        };
        rx.run(&mut actor);
    });

    // Roundtrip: send bytes → expect parsed commands
    let input = b"Hello".to_vec();
    tx.send(Message::Data(input)).expect("test failure");

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().expect("test failure");

    // Verify roundtrip
    assert_eq!(bytes_processed.load(Ordering::SeqCst), 5);

    let commands = output_rx.recv_timeout(Duration::from_millis(100)).expect("test failure");
    assert_eq!(commands.len(), 5);
    assert_eq!(commands[0], TestAnsiCommand::Print('H'));
    assert_eq!(commands[1], TestAnsiCommand::Print('e'));
    assert_eq!(commands[2], TestAnsiCommand::Print('l'));
    assert_eq!(commands[3], TestAnsiCommand::Print('l'));
    assert_eq!(commands[4], TestAnsiCommand::Print('o'));
}

#[test]
fn parser_roundtrip_ansi_escape_sequence() {
    let (output_tx, output_rx) = std::sync::mpsc::sync_channel(10);
    let bytes_processed = Arc::new(AtomicUsize::new(0));

    let (tx, mut rx) = ActorScheduler::<Vec<u8>, (), ()>::new(10, 64);

    let bytes_clone = bytes_processed.clone();
    let handle = thread::spawn(move || {
        let mut actor = TestParserActor {
            output_tx,
            bytes_processed: bytes_clone,
        };
        rx.run(&mut actor);
    });

    // Roundtrip: send ANSI escape sequence → expect parsed CSI command
    let input = b"\x1b[A".to_vec(); // Cursor Up
    tx.send(Message::Data(input)).expect("test failure");

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().expect("test failure");

    let commands = output_rx.recv_timeout(Duration::from_millis(100)).expect("test failure");
    assert_eq!(commands.len(), 1);
    assert_eq!(commands[0], TestAnsiCommand::CsiCursorUp(1));
}

#[test]
fn parser_roundtrip_mixed_content() {
    let (output_tx, output_rx) = std::sync::mpsc::sync_channel(10);
    let bytes_processed = Arc::new(AtomicUsize::new(0));

    let (tx, mut rx) = ActorScheduler::<Vec<u8>, (), ()>::new(10, 64);

    let bytes_clone = bytes_processed.clone();
    let handle = thread::spawn(move || {
        let mut actor = TestParserActor {
            output_tx,
            bytes_processed: bytes_clone,
        };
        rx.run(&mut actor);
    });

    // Roundtrip: mixed text and control characters
    let input = b"Hi\nWorld\r".to_vec();
    tx.send(Message::Data(input)).expect("test failure");

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().expect("test failure");

    let commands = output_rx.recv_timeout(Duration::from_millis(100)).expect("test failure");
    assert_eq!(commands.len(), 9);
    assert_eq!(commands[0], TestAnsiCommand::Print('H'));
    assert_eq!(commands[1], TestAnsiCommand::Print('i'));
    assert_eq!(commands[2], TestAnsiCommand::C0Newline);
    assert_eq!(commands[3], TestAnsiCommand::Print('W'));
    assert_eq!(commands[4], TestAnsiCommand::Print('o'));
    assert_eq!(commands[5], TestAnsiCommand::Print('r'));
    assert_eq!(commands[6], TestAnsiCommand::Print('l'));
    assert_eq!(commands[7], TestAnsiCommand::Print('d'));
    assert_eq!(commands[8], TestAnsiCommand::C0CarriageReturn);
}

#[test]
fn parser_roundtrip_multiple_batches_preserve_order() {
    let (output_tx, output_rx) = std::sync::mpsc::sync_channel(100);
    let bytes_processed = Arc::new(AtomicUsize::new(0));

    let (tx, mut rx) = ActorScheduler::<Vec<u8>, (), ()>::new(10, 64);

    let bytes_clone = bytes_processed.clone();
    let handle = thread::spawn(move || {
        let mut actor = TestParserActor {
            output_tx,
            bytes_processed: bytes_clone,
        };
        rx.run(&mut actor);
    });

    // Send multiple batches
    tx.send(Message::Data(b"First".to_vec())).expect("test failure");
    tx.send(Message::Data(b"Second".to_vec())).expect("test failure");
    tx.send(Message::Data(b"Third".to_vec())).expect("test failure");

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    handle.join().expect("test failure");

    // Collect all commands in order
    let mut all_commands = Vec::new();
    while let Ok(batch) = output_rx.try_recv() {
        all_commands.extend(batch);
    }

    // Verify order preserved
    let text: String = all_commands
        .iter()
        .filter_map(|cmd| match cmd {
            TestAnsiCommand::Print(c) => Some(*c),
            _ => None,
        })
        .collect();

    assert_eq!(text, "FirstSecondThird");
}

#[test]
fn parser_roundtrip_empty_input_no_output() {
    let (output_tx, output_rx) = std::sync::mpsc::sync_channel(10);
    let bytes_processed = Arc::new(AtomicUsize::new(0));

    let (tx, mut rx) = ActorScheduler::<Vec<u8>, (), ()>::new(10, 64);

    let bytes_clone = bytes_processed.clone();
    let handle = thread::spawn(move || {
        let mut actor = TestParserActor {
            output_tx,
            bytes_processed: bytes_clone,
        };
        rx.run(&mut actor);
    });

    // Roundtrip: empty input → no output
    tx.send(Message::Data(vec![])).expect("test failure");

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().expect("test failure");

    // Verify no commands produced
    assert!(output_rx.try_recv().is_err());
}

// =============================================================================
// TerminalApp-like Actor Roundtrip Tests
// =============================================================================

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
enum TestEngineControl {
    Resize(u32, u32),
    Close,
}

#[derive(Debug, Clone, PartialEq)]
enum TestEngineManagement {
    KeyPress(char),
    SpecialKey(String),
}

#[derive(Debug, Clone, PartialEq)]
enum TestEngineData {
    FrameRequest,
}

struct TestTerminalAppActor {
    pty_output_tx: std::sync::mpsc::SyncSender<Vec<u8>>,
    resize_count: Arc<AtomicUsize>,
    keypress_count: Arc<AtomicUsize>,
    frame_count: Arc<AtomicUsize>,
}

impl Actor<TestEngineData, TestEngineControl, TestEngineManagement> for TestTerminalAppActor {
    fn handle_data(&mut self, data: TestEngineData) -> HandlerResult {
        match data {
            TestEngineData::FrameRequest => {
                self.frame_count.fetch_add(1, Ordering::SeqCst);
            }
        }
        Ok(())
    }

    fn handle_control(&mut self, ctrl: TestEngineControl) -> HandlerResult {
        match ctrl {
            TestEngineControl::Resize(width, height) => {
                self.resize_count.fetch_add(1, Ordering::SeqCst);
                // In real app, this would update terminal dimensions
                let _ = self
                    .pty_output_tx
                    .send(format!("RESIZE:{}x{}", width, height).into_bytes());
            }
            TestEngineControl::Close => {
                // Handle close
            }
        }
        Ok(())
    }

    fn handle_management(&mut self, mgmt: TestEngineManagement) -> HandlerResult {
        match mgmt {
            TestEngineManagement::KeyPress(c) => {
                self.keypress_count.fetch_add(1, Ordering::SeqCst);
                let _ = self.pty_output_tx.send(vec![c as u8]);
            }
            TestEngineManagement::SpecialKey(key) => {
                self.keypress_count.fetch_add(1, Ordering::SeqCst);
                // Simulate ANSI escape sequence for special keys
                let bytes = match key.as_str() {
                    "Up" => b"\x1b[A".to_vec(),
                    "Down" => b"\x1b[B".to_vec(),
                    "Right" => b"\x1b[C".to_vec(),
                    "Left" => b"\x1b[D".to_vec(),
                    _ => vec![],
                };
                if !bytes.is_empty() {
                    let _ = self.pty_output_tx.send(bytes);
                }
            }
        }
        Ok(())
    }

    fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

#[test]
fn terminal_app_roundtrip_key_input() {
    let (pty_tx, pty_rx) = std::sync::mpsc::sync_channel(10);
    let resize_count = Arc::new(AtomicUsize::new(0));
    let keypress_count = Arc::new(AtomicUsize::new(0));
    let frame_count = Arc::new(AtomicUsize::new(0));

    let (tx, mut rx) =
        ActorScheduler::<TestEngineData, TestEngineControl, TestEngineManagement>::new(10, 64);

    let resize_clone = resize_count.clone();
    let keypress_clone = keypress_count.clone();
    let frame_clone = frame_count.clone();

    let handle = thread::spawn(move || {
        let mut actor = TestTerminalAppActor {
            pty_output_tx: pty_tx,
            resize_count: resize_clone,
            keypress_count: keypress_clone,
            frame_count: frame_clone,
        };
        rx.run(&mut actor);
    });

    // Roundtrip: send key press → expect PTY output
    tx.send(Message::Management(TestEngineManagement::KeyPress('a')))
        .expect("test failure");
    tx.send(Message::Management(TestEngineManagement::KeyPress('b')))
        .expect("test failure");

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().expect("test failure");

    // Verify roundtrip
    assert_eq!(keypress_count.load(Ordering::SeqCst), 2);

    let output1 = pty_rx.recv_timeout(Duration::from_millis(100)).expect("test failure");
    let output2 = pty_rx.recv_timeout(Duration::from_millis(100)).expect("test failure");

    assert_eq!(output1, vec![b'a']);
    assert_eq!(output2, vec![b'b']);
}

#[test]
fn terminal_app_roundtrip_special_key() {
    let (pty_tx, pty_rx) = std::sync::mpsc::sync_channel(10);
    let resize_count = Arc::new(AtomicUsize::new(0));
    let keypress_count = Arc::new(AtomicUsize::new(0));
    let frame_count = Arc::new(AtomicUsize::new(0));

    let (tx, mut rx) =
        ActorScheduler::<TestEngineData, TestEngineControl, TestEngineManagement>::new(10, 64);

    let resize_clone = resize_count.clone();
    let keypress_clone = keypress_count.clone();
    let frame_clone = frame_count.clone();

    let handle = thread::spawn(move || {
        let mut actor = TestTerminalAppActor {
            pty_output_tx: pty_tx,
            resize_count: resize_clone,
            keypress_count: keypress_clone,
            frame_count: frame_clone,
        };
        rx.run(&mut actor);
    });

    // Roundtrip: send arrow key → expect ANSI escape sequence
    tx.send(Message::Management(TestEngineManagement::SpecialKey(
        "Up".to_string(),
    )))
    .expect("test failure");

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().expect("test failure");

    assert_eq!(keypress_count.load(Ordering::SeqCst), 1);

    let output = pty_rx.recv_timeout(Duration::from_millis(100)).expect("test failure");
    assert_eq!(output, b"\x1b[A");
}

#[test]
fn terminal_app_roundtrip_resize_control() {
    let (pty_tx, pty_rx) = std::sync::mpsc::sync_channel(10);
    let resize_count = Arc::new(AtomicUsize::new(0));
    let keypress_count = Arc::new(AtomicUsize::new(0));
    let frame_count = Arc::new(AtomicUsize::new(0));

    let (tx, mut rx) =
        ActorScheduler::<TestEngineData, TestEngineControl, TestEngineManagement>::new(10, 64);

    let resize_clone = resize_count.clone();
    let keypress_clone = keypress_count.clone();
    let frame_clone = frame_count.clone();

    let handle = thread::spawn(move || {
        let mut actor = TestTerminalAppActor {
            pty_output_tx: pty_tx,
            resize_count: resize_clone,
            keypress_count: keypress_clone,
            frame_count: frame_clone,
        };
        rx.run(&mut actor);
    });

    // Roundtrip: send resize control → expect notification
    tx.send(Message::Control(TestEngineControl::Resize(1920, 1080)))
        .expect("test failure");

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().expect("test failure");

    assert_eq!(resize_count.load(Ordering::SeqCst), 1);

    let output = pty_rx.recv_timeout(Duration::from_millis(100)).expect("test failure");
    assert_eq!(output, b"RESIZE:1920x1080");
}

#[test]
fn terminal_app_roundtrip_frame_request() {
    let (pty_tx, _pty_rx) = std::sync::mpsc::sync_channel(10);
    let resize_count = Arc::new(AtomicUsize::new(0));
    let keypress_count = Arc::new(AtomicUsize::new(0));
    let frame_count = Arc::new(AtomicUsize::new(0));

    let (tx, mut rx) =
        ActorScheduler::<TestEngineData, TestEngineControl, TestEngineManagement>::new(10, 64);

    let resize_clone = resize_count.clone();
    let keypress_clone = keypress_count.clone();
    let frame_clone = frame_count.clone();

    let handle = thread::spawn(move || {
        let mut actor = TestTerminalAppActor {
            pty_output_tx: pty_tx,
            resize_count: resize_clone,
            keypress_count: keypress_clone,
            frame_count: frame_clone,
        };
        rx.run(&mut actor);
    });

    // Roundtrip: send frame requests
    tx.send(Message::Data(TestEngineData::FrameRequest))
        .expect("test failure");
    tx.send(Message::Data(TestEngineData::FrameRequest))
        .expect("test failure");
    tx.send(Message::Data(TestEngineData::FrameRequest))
        .expect("test failure");

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().expect("test failure");

    assert_eq!(frame_count.load(Ordering::SeqCst), 3);
}

#[test]
fn terminal_app_roundtrip_priority_ordering() {
    let (pty_tx, pty_rx) = std::sync::mpsc::sync_channel(100);
    let resize_count = Arc::new(AtomicUsize::new(0));
    let keypress_count = Arc::new(AtomicUsize::new(0));
    let frame_count = Arc::new(AtomicUsize::new(0));

    let (tx, mut rx) =
        ActorScheduler::<TestEngineData, TestEngineControl, TestEngineManagement>::new(10, 64);

    let resize_clone = resize_count.clone();
    let keypress_clone = keypress_count.clone();
    let frame_clone = frame_count.clone();

    let handle = thread::spawn(move || {
        let mut actor = TestTerminalAppActor {
            pty_output_tx: pty_tx,
            resize_count: resize_clone,
            keypress_count: keypress_clone,
            frame_count: frame_clone,
        };
        rx.run(&mut actor);
    });

    // Send in reverse priority order
    tx.send(Message::Data(TestEngineData::FrameRequest))
        .expect("test failure");
    tx.send(Message::Management(TestEngineManagement::KeyPress('x')))
        .expect("test failure");
    tx.send(Message::Control(TestEngineControl::Resize(800, 600)))
        .expect("test failure");

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    handle.join().expect("test failure");

    // Verify all processed
    assert_eq!(resize_count.load(Ordering::SeqCst), 1);
    assert_eq!(keypress_count.load(Ordering::SeqCst), 1);
    assert_eq!(frame_count.load(Ordering::SeqCst), 1);

    // Control should be processed first (resize)
    let first = pty_rx.recv_timeout(Duration::from_millis(100)).expect("test failure");
    assert_eq!(first, b"RESIZE:800x600");

    // Then management (keypress)
    let second = pty_rx.recv_timeout(Duration::from_millis(100)).expect("test failure");
    assert_eq!(second, vec![b'x']);
}

// =============================================================================
// Multi-Actor Chain Roundtrip Tests
// =============================================================================

#[test]
fn multi_actor_chain_roundtrip() {
    // Setup: Parser → App chain
    let (app_input_tx, app_input_rx) = std::sync::mpsc::sync_channel(10);
    let (app_output_tx, app_output_rx) = std::sync::mpsc::sync_channel(10);

    // Parser actor
    let (parser_tx, mut parser_rx) = ActorScheduler::<Vec<u8>, (), ()>::new(10, 64);
    let bytes_processed = Arc::new(AtomicUsize::new(0));
    let bytes_clone = bytes_processed.clone();

    let parser_handle = thread::spawn(move || {
        let mut parser = TestParserActor {
            output_tx: app_input_tx,
            bytes_processed: bytes_clone,
        };
        parser_rx.run(&mut parser);
    });

    // Simulated App actor (just forwards parsed commands as text)
    struct ForwardingActor {
        output_tx: std::sync::mpsc::SyncSender<String>,
    }

    impl Actor<Vec<TestAnsiCommand>, (), ()> for ForwardingActor {
        fn handle_data(&mut self, cmds: Vec<TestAnsiCommand>) -> HandlerResult {
            let text: String = cmds
                .iter()
                .filter_map(|cmd| match cmd {
                    TestAnsiCommand::Print(c) => Some(*c),
                    _ => None,
                })
                .collect();
            if !text.is_empty() {
                let _ = self.output_tx.send(text);
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

    let (app_tx, mut app_rx) = ActorScheduler::<Vec<TestAnsiCommand>, (), ()>::new(10, 64);

    let _app_handle = thread::spawn(move || {
        let mut app = ForwardingActor {
            output_tx: app_output_tx,
        };
        app_rx.run(&mut app);
    });

    // Bridge: forward parser output to app input
    let bridge_handle = thread::spawn(move || {
        while let Ok(commands) = app_input_rx.recv() {
            if app_tx.send(Message::Data(commands)).is_err() {
                break;
            }
        }
    });

    // Roundtrip: raw bytes → parser → app → final output
    parser_tx.send(Message::Data(b"Hello".to_vec())).expect("test failure");
    parser_tx.send(Message::Data(b" World".to_vec())).expect("test failure");

    thread::sleep(Duration::from_millis(100));
    drop(parser_tx);

    parser_handle.join().expect("test failure");
    bridge_handle.join().expect("test failure");

    // Verify complete chain
    assert_eq!(bytes_processed.load(Ordering::SeqCst), 11); // "Hello World"

    let output1 = app_output_rx
        .recv_timeout(Duration::from_millis(100))
        .expect("test failure");
    let output2 = app_output_rx
        .recv_timeout(Duration::from_millis(100))
        .expect("test failure");

    assert_eq!(output1, "Hello");
    assert_eq!(output2, " World");
}

// =============================================================================
// Error Handling and Edge Cases
// =============================================================================

#[test]
fn roundtrip_handles_actor_panic_gracefully() {
    struct PanickyActor {
        panic_on_message: usize,
        message_count: Arc<AtomicUsize>,
    }

    impl Actor<String, (), ()> for PanickyActor {
        fn handle_data(&mut self, _msg: String) -> HandlerResult {
            let count = self.message_count.fetch_add(1, Ordering::SeqCst) + 1;
            if count == self.panic_on_message {
                panic!("Intentional panic for testing");
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

    let message_count = Arc::new(AtomicUsize::new(0));
    let (tx, mut rx) = ActorScheduler::<String, (), ()>::new(10, 64);

    let count_clone = message_count.clone();
    let handle = thread::spawn(move || {
        let mut actor = PanickyActor {
            panic_on_message: 3,
            message_count: count_clone,
        };
        rx.run(&mut actor);
    });

    // Send messages
    for i in 1..=5 {
        let _ = tx.send(Message::Data(format!("msg{}", i)));
    }

    thread::sleep(Duration::from_millis(100));
    drop(tx);

    // Thread should panic on message 3
    let result = handle.join();
    assert!(result.is_err(), "Actor should panic");

    // Only first 2 messages processed before panic
    assert_eq!(message_count.load(Ordering::SeqCst), 3);
}

#[test]
fn roundtrip_sender_dropped_during_processing() {
    let processed = Arc::new(AtomicUsize::new(0));
    let (tx, mut rx) = ActorScheduler::<usize, (), ()>::new(10, 64);

    let processed_clone = processed.clone();
    let started = Arc::new(AtomicBool::new(false));
    let started_clone = started.clone();

    let handle = thread::spawn(move || {
        struct SlowActor {
            processed: Arc<AtomicUsize>,
            started: Arc<AtomicBool>,
        }
        impl Actor<usize, (), ()> for SlowActor {
            fn handle_data(&mut self, _: usize) -> HandlerResult {
                self.started.store(true, Ordering::SeqCst);
                thread::sleep(Duration::from_millis(50));
                self.processed.fetch_add(1, Ordering::SeqCst);
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
        rx.run(&mut SlowActor {
            processed: processed_clone,
            started: started_clone,
        });
    });

    // Send message
    tx.send(Message::Data(1)).expect("test failure");

    // Wait for processing to start
    while !started.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(1));
    }

    // Drop sender while actor is processing
    drop(tx);

    handle.join().expect("test failure");

    // Actor should finish processing current message
    assert_eq!(processed.load(Ordering::SeqCst), 1);
}

// =============================================================================
// PTY Resize Actor Boundary Tests
// =============================================================================

use core_term::io::PtyCommand;

/// Test that PtyCommand::Resize is properly sent and received at the actor boundary.
/// This tests the channel contract between TerminalApp and WriteThread.
#[test]
fn pty_command_resize_delivery_at_actor_boundary() {
    let (tx, rx) = std::sync::mpsc::sync_channel::<PtyCommand>(16);

    // Simulate sending resize command from TerminalApp
    tx.send(PtyCommand::Resize(core_term::io::Resize {
        cols: 120,
        rows: 40,
    }))
    .expect("Should send resize command");

    // Simulate receiving in WriteThread
    let cmd = rx.recv_timeout(Duration::from_millis(100)).expect("test failure");

    assert_eq!(
        cmd,
        PtyCommand::Resize(core_term::io::Resize {
            cols: 120,
            rows: 40
        })
    );
}

/// Test that multiple resize commands maintain ordering (FIFO).
#[test]
fn pty_command_resize_ordering_preserved() {
    let (tx, rx) = std::sync::mpsc::sync_channel::<PtyCommand>(16);

    // Send multiple resize commands
    tx.send(PtyCommand::Resize(core_term::io::Resize {
        cols: 80,
        rows: 24,
    }))
    .expect("test failure");
    tx.send(PtyCommand::Resize(core_term::io::Resize {
        cols: 120,
        rows: 40,
    }))
    .expect("test failure");
    tx.send(PtyCommand::Resize(core_term::io::Resize {
        cols: 200,
        rows: 60,
    }))
    .expect("test failure");

    // Verify FIFO order
    assert_eq!(
        rx.recv_timeout(Duration::from_millis(100)).expect("test failure"),
        PtyCommand::Resize(core_term::io::Resize { cols: 80, rows: 24 })
    );
    assert_eq!(
        rx.recv_timeout(Duration::from_millis(100)).expect("test failure"),
        PtyCommand::Resize(core_term::io::Resize {
            cols: 120,
            rows: 40
        })
    );
    assert_eq!(
        rx.recv_timeout(Duration::from_millis(100)).expect("test failure"),
        PtyCommand::Resize(core_term::io::Resize {
            cols: 200,
            rows: 60
        })
    );
}

/// Test that Write and Resize commands are correctly interleaved.
#[test]
fn pty_command_write_and_resize_interleaved() {
    let (tx, rx) = std::sync::mpsc::sync_channel::<PtyCommand>(16);

    // Interleave write and resize commands
    tx.send(PtyCommand::Write(b"hello".to_vec())).expect("test failure");
    tx.send(PtyCommand::Resize(core_term::io::Resize {
        cols: 100,
        rows: 50,
    }))
    .expect("test failure");
    tx.send(PtyCommand::Write(b"world".to_vec())).expect("test failure");

    // Verify interleaved order
    assert_eq!(
        rx.recv_timeout(Duration::from_millis(100)).expect("test failure"),
        PtyCommand::Write(b"hello".to_vec())
    );
    assert_eq!(
        rx.recv_timeout(Duration::from_millis(100)).expect("test failure"),
        PtyCommand::Resize(core_term::io::Resize {
            cols: 100,
            rows: 50
        })
    );
    assert_eq!(
        rx.recv_timeout(Duration::from_millis(100)).expect("test failure"),
        PtyCommand::Write(b"world".to_vec())
    );
}

/// Test that sender drop properly signals channel closure.
#[test]
fn pty_command_channel_closure_on_sender_drop() {
    let (tx, rx) = std::sync::mpsc::sync_channel::<PtyCommand>(16);

    tx.send(PtyCommand::Resize(core_term::io::Resize {
        cols: 80,
        rows: 24,
    }))
    .expect("test failure");

    drop(tx);

    // First recv should succeed
    assert!(rx.recv().is_ok());

    // Second recv should fail (channel closed)
    assert!(rx.recv().is_err());
}

/// Test resize command with extreme values (boundary conditions).
#[test]
fn pty_command_resize_boundary_values() {
    let (tx, rx) = std::sync::mpsc::sync_channel::<PtyCommand>(16);

    // Minimum valid size
    tx.send(PtyCommand::Resize(core_term::io::Resize {
        cols: 1,
        rows: 1,
    }))
    .expect("test failure");

    // Maximum u16 values
    tx.send(PtyCommand::Resize(core_term::io::Resize {
        cols: u16::MAX,
        rows: u16::MAX,
    }))
    .expect("test failure");

    assert_eq!(
        rx.recv_timeout(Duration::from_millis(100)).expect("test failure"),
        PtyCommand::Resize(core_term::io::Resize { cols: 1, rows: 1 })
    );
    assert_eq!(
        rx.recv_timeout(Duration::from_millis(100)).expect("test failure"),
        PtyCommand::Resize(core_term::io::Resize {
            cols: u16::MAX,
            rows: u16::MAX
        })
    );
}

/// Test that resize commands from multiple threads are all delivered.
#[test]
fn pty_command_resize_from_multiple_senders() {
    let (tx, rx) = std::sync::mpsc::sync_channel::<PtyCommand>(32);

    let tx1 = tx.clone();
    let tx2 = tx.clone();

    let h1 = thread::spawn(move || {
        for i in 0..5 {
            tx1.send(PtyCommand::Resize(core_term::io::Resize {
                cols: 100 + i,
                rows: 50,
            }))
            .expect("test failure");
        }
    });

    let h2 = thread::spawn(move || {
        for i in 0..5 {
            tx2.send(PtyCommand::Write(format!("msg{}", i).into_bytes()))
                .expect("test failure");
        }
    });

    h1.join().expect("test failure");
    h2.join().expect("test failure");
    drop(tx); // Drop original sender

    // Count received commands
    let mut resize_count = 0;
    let mut write_count = 0;

    while let Ok(cmd) = rx.try_recv() {
        match cmd {
            PtyCommand::Resize(_) => resize_count += 1,
            PtyCommand::Write(_) => write_count += 1,
        }
    }

    assert_eq!(resize_count, 5, "Should receive 5 resize commands");
    assert_eq!(write_count, 5, "Should receive 5 write commands");
}
