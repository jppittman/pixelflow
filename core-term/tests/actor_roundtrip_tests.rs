//! Actor roundtrip tests ensuring message delivery and processing correctness.
//!
//! These tests verify complete message roundtrips through actors:
//! - Send message → Actor processes → Verify output
//!
//! Following TDD principles: write tests first, uncover bugs, fix them.

use actor_scheduler::{
    Actor, ActorBuilder, ActorScheduler, ActorStatus, HandlerError, HandlerResult, Message,
    SystemStatus,
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
            self.output_tx.send(commands).ok();
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
    tx.send(Message::Data(input)).unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // Verify roundtrip
    assert_eq!(bytes_processed.load(Ordering::SeqCst), 5);

    let commands = output_rx.recv_timeout(Duration::from_millis(100)).unwrap();
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
    tx.send(Message::Data(input)).unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    let commands = output_rx.recv_timeout(Duration::from_millis(100)).unwrap();
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
    tx.send(Message::Data(input)).unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    let commands = output_rx.recv_timeout(Duration::from_millis(100)).unwrap();
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
    tx.send(Message::Data(b"First".to_vec())).unwrap();
    tx.send(Message::Data(b"Second".to_vec())).unwrap();
    tx.send(Message::Data(b"Third".to_vec())).unwrap();

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    handle.join().unwrap();

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
    tx.send(Message::Data(vec![])).unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

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
                self.pty_output_tx
                    .send(format!("RESIZE:{}x{}", width, height).into_bytes())
                    .ok();
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
                self.pty_output_tx.send(vec![c as u8]).ok();
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
                    self.pty_output_tx.send(bytes).ok();
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
        .unwrap();
    tx.send(Message::Management(TestEngineManagement::KeyPress('b')))
        .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // Verify roundtrip
    assert_eq!(keypress_count.load(Ordering::SeqCst), 2);

    let output1 = pty_rx.recv_timeout(Duration::from_millis(100)).unwrap();
    let output2 = pty_rx.recv_timeout(Duration::from_millis(100)).unwrap();

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
    .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    assert_eq!(keypress_count.load(Ordering::SeqCst), 1);

    let output = pty_rx.recv_timeout(Duration::from_millis(100)).unwrap();
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
        .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    assert_eq!(resize_count.load(Ordering::SeqCst), 1);

    let output = pty_rx.recv_timeout(Duration::from_millis(100)).unwrap();
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
        .unwrap();
    tx.send(Message::Data(TestEngineData::FrameRequest))
        .unwrap();
    tx.send(Message::Data(TestEngineData::FrameRequest))
        .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

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
        .unwrap();
    tx.send(Message::Management(TestEngineManagement::KeyPress('x')))
        .unwrap();
    tx.send(Message::Control(TestEngineControl::Resize(800, 600)))
        .unwrap();

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    handle.join().unwrap();

    // Verify all processed
    assert_eq!(resize_count.load(Ordering::SeqCst), 1);
    assert_eq!(keypress_count.load(Ordering::SeqCst), 1);
    assert_eq!(frame_count.load(Ordering::SeqCst), 1);

    // Control should be processed first (resize)
    let first = pty_rx.recv_timeout(Duration::from_millis(100)).unwrap();
    assert_eq!(first, b"RESIZE:800x600");

    // Then management (keypress)
    let second = pty_rx.recv_timeout(Duration::from_millis(100)).unwrap();
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
                self.output_tx.send(text).ok();
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
    parser_tx.send(Message::Data(b"Hello".to_vec())).unwrap();
    parser_tx.send(Message::Data(b" World".to_vec())).unwrap();

    thread::sleep(Duration::from_millis(100));
    drop(parser_tx);

    parser_handle.join().unwrap();
    bridge_handle.join().unwrap();

    // Verify complete chain
    assert_eq!(bytes_processed.load(Ordering::SeqCst), 11); // "Hello World"

    let output1 = app_output_rx
        .recv_timeout(Duration::from_millis(100))
        .unwrap();
    let output2 = app_output_rx
        .recv_timeout(Duration::from_millis(100))
        .unwrap();

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
        tx.send(Message::Data(format!("msg{}", i))).ok();
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
    tx.send(Message::Data(1)).unwrap();

    // Wait for processing to start
    while !started.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(1));
    }

    // Drop sender while actor is processing
    drop(tx);

    handle.join().unwrap();

    // Actor should finish processing current message
    assert_eq!(processed.load(Ordering::SeqCst), 1);
}

// =============================================================================
// PTY Writer Actor Boundary Tests
// =============================================================================
//
// The app talks to the PTY writer actor over two lanes: bytes for the shell
// on Data, `WriterControl::Resize` on Control. These tests pin the contract
// at that boundary using a probe actor in place of the real PtyWriter.

use core_term::io::event_monitor_actor::{NoManagement, WriterControl};
use core_term::io::Resize;

/// Records everything the writer actor would have received, in drain order.
#[derive(Default)]
struct WriterProbe {
    received: Vec<WriterEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum WriterEvent {
    Write(Vec<u8>),
    Resize(Resize),
}

impl Actor<Vec<u8>, WriterControl, NoManagement> for WriterProbe {
    fn handle_data(&mut self, bytes: Vec<u8>) -> HandlerResult {
        self.received.push(WriterEvent::Write(bytes));
        Ok(())
    }
    fn handle_control(&mut self, msg: WriterControl) -> HandlerResult {
        let WriterControl::Resize(resize) = msg;
        self.received.push(WriterEvent::Resize(resize));
        Ok(())
    }
    fn handle_management(&mut self, _msg: NoManagement) -> HandlerResult {
        Ok(())
    }
    fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

fn drain_probe(
    rx: &mut ActorScheduler<Vec<u8>, WriterControl, NoManagement>,
    probe: &mut WriterProbe,
) {
    for _ in 0..8 {
        if rx.poll_once(probe).is_some() {
            break;
        }
    }
}

/// A resize sent on the control lane reaches the writer actor.
#[test]
fn pty_writer_resize_delivery_at_actor_boundary() {
    let (tx, mut rx) = ActorScheduler::<Vec<u8>, WriterControl, NoManagement>::new(16, 16);

    tx.send(Message::Control(WriterControl::Resize(Resize {
        cols: 120,
        rows: 40,
    })))
    .expect("Should send resize command");

    let mut probe = WriterProbe::default();
    drain_probe(&mut rx, &mut probe);

    assert_eq!(
        probe.received,
        vec![WriterEvent::Resize(Resize {
            cols: 120,
            rows: 40
        })]
    );
}

/// Resizes stay FIFO within the control lane.
#[test]
fn pty_writer_resize_ordering_preserved() {
    let (tx, mut rx) = ActorScheduler::<Vec<u8>, WriterControl, NoManagement>::new(16, 16);

    for (cols, rows) in [(80, 24), (120, 40), (200, 60)] {
        tx.send(Message::Control(WriterControl::Resize(Resize {
            cols,
            rows,
        })))
        .unwrap();
    }

    let mut probe = WriterProbe::default();
    drain_probe(&mut rx, &mut probe);

    assert_eq!(
        probe.received,
        vec![
            WriterEvent::Resize(Resize { cols: 80, rows: 24 }),
            WriterEvent::Resize(Resize {
                cols: 120,
                rows: 40
            }),
            WriterEvent::Resize(Resize {
                cols: 200,
                rows: 60
            }),
        ]
    );
}

/// The point of the lane split: a resize queued *after* bulk writes is
/// drained *before* them. Control preempts Data.
#[test]
fn pty_writer_resize_preempts_queued_writes() {
    let (tx, mut rx) = ActorScheduler::<Vec<u8>, WriterControl, NoManagement>::new(16, 16);

    tx.send(Message::Data(b"hello".to_vec())).unwrap();
    tx.send(Message::Data(b"world".to_vec())).unwrap();
    tx.send(Message::Control(WriterControl::Resize(Resize {
        cols: 100,
        rows: 50,
    })))
    .unwrap();

    let mut probe = WriterProbe::default();
    drain_probe(&mut rx, &mut probe);

    assert_eq!(
        probe.received,
        vec![
            WriterEvent::Resize(Resize {
                cols: 100,
                rows: 50
            }),
            WriterEvent::Write(b"hello".to_vec()),
            WriterEvent::Write(b"world".to_vec()),
        ],
        "resize should jump ahead of queued writes"
    );
}

/// Dropping every producer completes the writer's scheduler after the
/// buffered messages drain.
#[test]
fn pty_writer_completes_on_handle_drop() {
    let (tx, mut rx) = ActorScheduler::<Vec<u8>, WriterControl, NoManagement>::new(16, 16);

    tx.send(Message::Data(b"last words".to_vec())).unwrap();
    drop(tx);

    let mut probe = WriterProbe::default();
    let phase = loop {
        if let Some(phase) = rx.poll_once(&mut probe) {
            break phase;
        }
    };

    assert_eq!(phase, actor_scheduler::PodPhase::Completed);
    assert_eq!(probe.received, vec![WriterEvent::Write(b"last words".to_vec())]);
}

/// Resize survives boundary values.
#[test]
fn pty_writer_resize_boundary_values() {
    let (tx, mut rx) = ActorScheduler::<Vec<u8>, WriterControl, NoManagement>::new(16, 16);

    for (cols, rows) in [(1, 1), (u16::MAX, u16::MAX)] {
        tx.send(Message::Control(WriterControl::Resize(Resize {
            cols,
            rows,
        })))
        .unwrap();
    }

    let mut probe = WriterProbe::default();
    drain_probe(&mut rx, &mut probe);

    assert_eq!(
        probe.received,
        vec![
            WriterEvent::Resize(Resize { cols: 1, rows: 1 }),
            WriterEvent::Resize(Resize {
                cols: u16::MAX,
                rows: u16::MAX
            }),
        ]
    );
}

/// Multiple producers (dedicated SPSC handles) all deliver.
#[test]
fn pty_writer_receives_from_multiple_producers() {
    let mut builder = ActorBuilder::<Vec<u8>, WriterControl, NoManagement>::new(32, None);
    let tx1 = builder.add_producer();
    let tx2 = builder.add_producer();
    let mut rx = builder.build();

    let h1 = thread::spawn(move || {
        for i in 0..5u16 {
            tx1.send(Message::Control(WriterControl::Resize(Resize {
                cols: 100 + i,
                rows: 50,
            })))
            .unwrap();
        }
    });

    let h2 = thread::spawn(move || {
        for i in 0..5 {
            tx2.send(Message::Data(format!("msg{}", i).into_bytes()))
                .unwrap();
        }
    });

    h1.join().unwrap();
    h2.join().unwrap();

    let mut probe = WriterProbe::default();
    loop {
        if rx.poll_once(&mut probe).is_some() {
            break;
        }
    }

    let resize_count = probe
        .received
        .iter()
        .filter(|e| matches!(e, WriterEvent::Resize(_)))
        .count();
    let write_count = probe
        .received
        .iter()
        .filter(|e| matches!(e, WriterEvent::Write(_)))
        .count();

    assert_eq!(resize_count, 5, "Should receive 5 resize commands");
    assert_eq!(write_count, 5, "Should receive 5 write commands");
}
