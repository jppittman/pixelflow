//! Actor roundtrip tests ensuring message delivery and processing correctness
//! for the real `PtyWriter` actor boundary.
//!
//! This file used to also carry a `ParserActor`/`TerminalApp` section built
//! entirely on hand-rolled test doubles (`TestParserActor`, `TestAnsiCommand`,
//! `TestEngineManagement`) that reimplemented ANSI parsing and key-to-escape
//! translation to exercise `actor_scheduler`'s generic delivery mechanics
//! under a `core-term`-flavored costume — never `core_term::ansi::AnsiProcessor`
//! or `core_term::term::emulator::key_translator`. It was removed as part of
//! the 2026-09-16 test-quality-audit follow-up's deferred item: every
//! guarantee it claimed to cover already exists against real types elsewhere
//! (see `ansi_parser_message_tests.rs` for the real `AnsiProcessor` pipeline,
//! `core-term/src/term/tests.rs` and
//! `core-term/src/term/emulator/key_translator.rs`'s own unit tests for real
//! key translation, and `actor-scheduler/tests/priority_and_order.rs` for
//! `ActorScheduler`'s own priority/FIFO ordering contract), so nothing here
//! replaces it.
//!
//! The `pty_writer_*` section below remains the pattern to follow for a new
//! test in this file: it drives the real
//! `core_term::io::event_monitor_actor::WriterControl`/`core_term::io::Resize`
//! types through a probe actor standing in for the real `PtyWriter`.

use actor_scheduler::{
    Actor, ActorBuilder, ActorScheduler, ActorStatus, HandlerError, HandlerResult, Message,
    SystemStatus,
};
use std::thread;

// =============================================================================
// PTY Writer Actor Boundary Tests
// =============================================================================
//
// The app talks to the PTY writer actor over two lanes: bytes for the shell
// on Data, `WriterControl::Resize` on Control. These tests pin the contract
// at that boundary using a probe actor in place of the real PtyWriter.

use core_term::io::event_monitor_actor::WriterControl;
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

impl Actor<Vec<u8>, WriterControl, ()> for WriterProbe {
    fn handle_data(&mut self, bytes: Vec<u8>) -> HandlerResult {
        self.received.push(WriterEvent::Write(bytes));
        Ok(())
    }
    fn handle_control(&mut self, msg: WriterControl) -> HandlerResult {
        let WriterControl::Resize(resize) = msg;
        self.received.push(WriterEvent::Resize(resize));
        Ok(())
    }
    fn handle_management(&mut self, _msg: ()) -> HandlerResult {
        Ok(())
    }
    fn handle_os(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

fn drain_probe(rx: &mut ActorScheduler<Vec<u8>, WriterControl, ()>, probe: &mut WriterProbe) {
    for _ in 0..8 {
        if rx.poll_once(probe) {
            break;
        }
    }
}

/// A resize sent on the control lane reaches the writer actor.
#[test]
fn pty_writer_resize_delivery_at_actor_boundary() {
    let (tx, mut rx) = ActorScheduler::<Vec<u8>, WriterControl, ()>::new(16, 16);

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
    let (tx, mut rx) = ActorScheduler::<Vec<u8>, WriterControl, ()>::new(16, 16);

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
    let (tx, mut rx) = ActorScheduler::<Vec<u8>, WriterControl, ()>::new(16, 16);

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
    let (tx, mut rx) = ActorScheduler::<Vec<u8>, WriterControl, ()>::new(16, 16);

    tx.send(Message::Data(b"last words".to_vec())).unwrap();
    drop(tx);

    let mut probe = WriterProbe::default();
    while !rx.poll_once(&mut probe) {}

    assert_eq!(
        probe.received,
        vec![WriterEvent::Write(b"last words".to_vec())]
    );
}

/// Resize survives boundary values.
#[test]
fn pty_writer_resize_boundary_values() {
    let (tx, mut rx) = ActorScheduler::<Vec<u8>, WriterControl, ()>::new(16, 16);

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
    let mut builder = ActorBuilder::<Vec<u8>, WriterControl, ()>::new(32, None);
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
    while !rx.poll_once(&mut probe) {}

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
