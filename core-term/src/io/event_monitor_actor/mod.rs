// src/io/event_monitor_actor/mod.rs

//! # PTY Actor Pipeline
//!
//! Three actors on the actor-scheduler, one per PTY concern:
//!
//! ```text
//!                    recycled buffers (Data lane)
//!            ┌──────────────────────────────────────┐
//!            ▼                                      │
//!      ┌───────────┐   raw bytes (Data)    ┌────────┴───┐   AnsiCommands
//!      │ PtyReader │──────────────────────▶│ PtyParser  │───────────────▶ app
//!      │ park:     │                       │ (message-  │
//!      │ epoll/    │── ChildExited ──────▶ │  driven)   │
//!      │ kqueue    │              (to app) └────────────┘
//!      └───────────┘
//!
//!      ┌───────────┐   Data: Write(bytes)
//!      │ PtyWriter │◀─────────────────────  app
//!      │ owns PTY  │   Control: Resize      (resize preempts queued writes)
//!      └───────────┘
//! ```
//!
//! **Blocking model.** The reader and writer bridge to the OS inside `park()`:
//! each blocks in `epoll_wait`/`kevent` on `{pty, waker}`, where the waker is
//! an [`FdWaker`](crate::io::waker::FdWaker) wired into the actor's
//! `ActorBuilder` — any message send interrupts the poll. The parser is purely
//! message-driven and blocks on the scheduler doorbell like any other actor.
//!
//! **Backpressure.** A fixed population of read buffers circulates
//! reader → parser → reader on data lanes. When all buffers are in flight the
//! reader stops reading; output accumulates in the kernel PTY buffer, which
//! blocks the shell. Nothing allocates after spawn and nothing is dropped.
//!
//! **Lifecycle.** [`EventMonitorBuilder::new`] wires channels (no threads),
//! [`EventMonitorBuilder::writer_handle`] mints the app's handle to the
//! writer, and [`EventMonitorBuilder::spawn`] starts the three threads.
//! Dropping [`EventMonitorActor`] sends `Message::Shutdown` to each actor
//! (the waker interrupts any in-flight poll) and joins the threads. On PTY
//! EOF the reader notifies the app via `PtySender::send_child_exited`.

mod parser;
mod reader;
mod writer;

use crate::io::pty::NixPty;
use crate::io::traits::PtySender;
use crate::io::waker::FdWaker;
use crate::io::Resize;
use anyhow::{Context, Result};
use actor_scheduler::{
    ActorBuilder, ActorHandle, Message, ShutdownMode, WakeHandler,
};
use log::*;
use parser::PtyParser;
use reader::{PtyReader, POOL_SIZE, READ_BUFFER_SIZE};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;
use writer::PtyWriter;

/// Control messages for the PTY reader (none yet; `Pause`/`Resume` flow
/// control is a planned extension).
#[derive(Debug, Clone)]
pub struct NoControl;

/// Placeholder for actors with no management lane.
#[derive(Debug, Clone)]
pub struct NoManagement;

/// Control messages for the PTY parser.
#[derive(Debug, Clone)]
pub enum ParserControl {
    /// Discard any half-parsed escape-sequence state.
    Reset,
}

/// A read result flowing reader → parser: `data[..len]` holds PTY output.
///
/// The buffer itself always keeps `len() == READ_BUFFER_SIZE`; carrying the
/// valid length out-of-band means buffers are never resized as they circulate,
/// so a read never pays a re-zeroing memset and a recycle is a pure move.
#[derive(Debug)]
pub(crate) struct FilledBuf {
    pub(crate) data: Vec<u8>,
    pub(crate) len: usize,
}

impl FilledBuf {
    /// The valid portion of the buffer.
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.data[..self.len]
    }
}

/// Control messages for the PTY writer. Drained before queued Data (writes),
/// which is exactly the priority a resize wants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriterControl {
    /// Set the PTY window size (`TIOCSWINSZ` + `SIGWINCH`).
    Resize(Resize),
}

/// The app's handle to the PTY writer: `Data` = bytes for the shell,
/// `Control` = resize.
pub type PtyWriterHandle = ActorHandle<Vec<u8>, WriterControl, NoManagement>;

/// Data-lane buffer size per producer. Must exceed [`POOL_SIZE`] so the
/// bootstrap seeding and a full pool recycle never block.
const READER_LANE_SIZE: usize = 16;
/// Parser inbox size for raw byte batches (matches the previous design).
const PARSER_LANE_SIZE: usize = 64;
/// Parser batches drained per wake (matches the previous burst limit).
const PARSER_BURST: usize = 10;
/// Writer inbox size for pending write requests from the app.
const WRITER_LANE_SIZE: usize = 128;
/// Writer messages drained per wake.
const WRITER_BURST: usize = 64;
/// On shutdown, how long the parser/writer drain queued work before exiting.
const DRAIN_TIMEOUT: Duration = Duration::from_millis(250);

/// Phase 1 of PTY pipeline construction: channels exist, no threads yet.
///
/// Exists so the app can be handed its writer handle *before* the pipeline
/// threads start (the pipeline in turn needs `PtySender`s minted by the app's
/// builder — a cycle that two-phase construction breaks).
pub struct EventMonitorBuilder {
    pty: NixPty,
    reader_builder: ActorBuilder<Vec<u8>, NoControl, NoManagement>,
    parser_builder: ActorBuilder<FilledBuf, ParserControl, NoManagement>,
    writer_builder: ActorBuilder<Vec<u8>, WriterControl, NoManagement>,
    reader_waker: Arc<FdWaker>,
    writer_waker: Arc<FdWaker>,
}

impl EventMonitorBuilder {
    /// Creates the channel plumbing for the PTY pipeline.
    ///
    /// `pty` is the primary master handle; the writer thread takes ownership
    /// of it (RAII: closing it hangs up the child).
    pub fn new(pty: NixPty) -> Result<Self> {
        let reader_waker = Arc::new(FdWaker::new().context("Failed to create reader waker")?);
        let writer_waker = Arc::new(FdWaker::new().context("Failed to create writer waker")?);

        Ok(Self {
            pty,
            reader_builder: ActorBuilder::new(
                READER_LANE_SIZE,
                Some(reader_waker.clone() as Arc<dyn WakeHandler>),
            ),
            parser_builder: ActorBuilder::new(PARSER_LANE_SIZE, None),
            writer_builder: ActorBuilder::new(
                WRITER_LANE_SIZE,
                Some(writer_waker.clone() as Arc<dyn WakeHandler>),
            ),
            reader_waker,
            writer_waker,
        })
    }

    /// Mints a handle to the PTY writer for the app.
    pub fn writer_handle(&mut self) -> PtyWriterHandle {
        self.writer_builder.add_producer()
    }

    /// Spawns the reader, parser, and writer threads.
    ///
    /// # Arguments
    ///
    /// * `parser_sink` - The parser's route to the app (parsed `AnsiCommand`s)
    /// * `reader_sink` - The reader's route to the app (`ChildExited`)
    pub fn spawn(
        mut self,
        parser_sink: Box<dyn PtySender>,
        reader_sink: Box<dyn PtySender>,
    ) -> Result<EventMonitorActor> {
        // Cross-actor handles.
        let parser_tx = self.parser_builder.add_producer(); // reader → parser
        let recycler_tx = self.reader_builder.add_producer(); // parser → reader
        // Shutdown handles; the reader's doubles as the pool-seeding producer.
        let reader_ctl = self.reader_builder.add_producer();
        let parser_ctl = self.parser_builder.add_producer();
        let writer_ctl = self.writer_builder.add_producer();

        let mut reader_rx = self
            .reader_builder
            .build_with_burst(POOL_SIZE, ShutdownMode::Immediate);
        let mut parser_rx = self.parser_builder.build_with_burst(
            PARSER_BURST,
            ShutdownMode::DrainAll {
                timeout: DRAIN_TIMEOUT,
            },
        );
        let mut writer_rx = self.writer_builder.build_with_burst(
            WRITER_BURST,
            ShutdownMode::DrainAll {
                timeout: DRAIN_TIMEOUT,
            },
        );

        let pty_read = self
            .pty
            .try_clone()
            .context("Failed to clone PTY for read thread")?;

        let mut actor = EventMonitorActor {
            reader_ctl,
            parser_ctl,
            writer_ctl,
            reader_join: None,
            parser_join: None,
            writer_join: None,
        };

        // Actors are constructed on their own threads: the event monitors
        // (epoll/kqueue handles) never cross a thread boundary.
        let reader_waker = self.reader_waker;
        actor.reader_join = Some(
            std::thread::Builder::new()
                .name("pty-reader".to_string())
                .spawn(move || {
                    match PtyReader::new(pty_read, reader_waker, parser_tx, reader_sink) {
                        Ok(mut reader) => {
                            debug!("PTY reader started");
                            reader_rx.run(&mut reader);
                            debug!("PTY reader exited");
                        }
                        Err(e) => error!("PTY reader failed to initialize: {}", e),
                    }
                })
                .context("Failed to spawn PTY reader thread")?,
        );

        actor.parser_join = Some(
            std::thread::Builder::new()
                .name("pty-parser".to_string())
                .spawn(move || {
                    debug!("PTY parser started");
                    let mut parser = PtyParser::new(parser_sink, recycler_tx);
                    parser_rx.run(&mut parser);
                    debug!("PTY parser exited");
                })
                .context("Failed to spawn PTY parser thread")?,
        );

        let pty = self.pty;
        let writer_waker = self.writer_waker;
        actor.writer_join = Some(
            std::thread::Builder::new()
                .name("pty-writer".to_string())
                .spawn(move || match PtyWriter::new(pty, writer_waker) {
                    Ok(mut writer) => {
                        debug!("PTY writer started");
                        writer_rx.run(&mut writer);
                        debug!("PTY writer exited - closing PTY");
                    }
                    Err(e) => error!("PTY writer failed to initialize: {}", e),
                })
                .context("Failed to spawn PTY writer thread")?,
        );

        // Seed the reader's buffer pool through its own data lane. These are
        // the only buffer allocations the pipeline ever makes, and the first
        // doorbell ring that moves the reader from its initial recv() into
        // park()'s poll loop. Buffers ship at full length (see the pool
        // invariant on PtyReader) so reads never resize them.
        for _ in 0..POOL_SIZE {
            actor
                .reader_ctl
                .send(Message::Data(vec![0u8; READ_BUFFER_SIZE]))
                .context("Failed to seed PTY reader buffer pool")?;
        }

        info!("PTY pipeline spawned: reader, parser, and writer actors");
        Ok(actor)
    }
}

/// Running PTY pipeline. Dropping it shuts the three actors down and joins
/// their threads.
pub struct EventMonitorActor {
    reader_ctl: ActorHandle<Vec<u8>, NoControl, NoManagement>,
    parser_ctl: ActorHandle<FilledBuf, ParserControl, NoManagement>,
    writer_ctl: PtyWriterHandle,
    reader_join: Option<JoinHandle<()>>,
    parser_join: Option<JoinHandle<()>>,
    writer_join: Option<JoinHandle<()>>,
}

fn join_pty_thread(handle: Option<JoinHandle<()>>, name: &str) {
    let Some(handle) = handle else { return };
    if let Err(panic_payload) = handle.join() {
        if std::thread::panicking() {
            // Already unwinding — can't double-panic, just log.
            eprintln!("{} thread panicked (during unwind): {:?}", name, panic_payload);
        } else {
            std::panic::resume_unwind(panic_payload);
        }
    }
}

impl Drop for EventMonitorActor {
    fn drop(&mut self) {
        debug!("EventMonitorActor dropped, shutting down PTY actors");

        // Explicit shutdown signals; each actor's waker interrupts any poll
        // it is blocked in. Errors mean the actor already exited.
        if let Err(e) = self.writer_ctl.send(Message::Shutdown) {
            debug!("PTY writer already gone at shutdown: {}", e);
        }
        if let Err(e) = self.reader_ctl.send(Message::Shutdown) {
            debug!("PTY reader already gone at shutdown: {}", e);
        }
        if let Err(e) = self.parser_ctl.send(Message::Shutdown) {
            debug!("PTY parser already gone at shutdown: {}", e);
        }

        join_pty_thread(self.writer_join.take(), "pty-writer");
        join_pty_thread(self.reader_join.take(), "pty-reader");
        join_pty_thread(self.parser_join.take(), "pty-parser");

        debug!("EventMonitorActor cleanup complete");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ansi::commands::AnsiCommand;
    use crate::io::pty::PtyConfig;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    /// PtySender double that records parsed output and the child-exit signal.
    #[derive(Clone, Default)]
    struct CaptureSink {
        commands: Arc<Mutex<Vec<AnsiCommand>>>,
        child_exited: Arc<AtomicBool>,
    }

    impl PtySender for CaptureSink {
        fn send(&self, cmds: Vec<AnsiCommand>) -> Result<(), anyhow::Error> {
            self.commands.lock().unwrap().extend(cmds);
            Ok(())
        }
        fn send_child_exited(&self) -> Result<(), anyhow::Error> {
            self.child_exited.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    impl CaptureSink {
        fn printed_text(&self) -> String {
            self.commands
                .lock()
                .unwrap()
                .iter()
                .filter_map(|cmd| match cmd {
                    AnsiCommand::Print(c) => Some(*c),
                    _ => None,
                })
                .collect()
        }

        fn wait_for(&self, timeout: Duration, pred: impl Fn(&Self) -> bool) -> bool {
            let deadline = Instant::now() + timeout;
            while Instant::now() < deadline {
                if pred(self) {
                    return true;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            false
        }
    }

    fn spawn_pipeline(
        command: &str,
        args: &[&str],
    ) -> (CaptureSink, PtyWriterHandle, EventMonitorActor) {
        let pty = NixPty::spawn_with_config(&PtyConfig {
            command_executable: command,
            args,
            initial_cols: 80,
            initial_rows: 24,
        })
        .expect("Failed to spawn PTY");

        let sink = CaptureSink::default();
        let mut builder = EventMonitorBuilder::new(pty).expect("builder");
        let writer = builder.writer_handle();
        let actor = builder
            .spawn(Box::new(sink.clone()), Box::new(sink.clone()))
            .expect("spawn pipeline");
        (sink, writer, actor)
    }

    /// Shell output flows PTY → reader → parser → sink, and EOF is reported.
    #[test]
    fn pipeline_delivers_output_and_child_exit() {
        let (sink, writer, actor) =
            spawn_pipeline("/bin/sh", &["-c", "printf 'hello-pipeline'"]);

        assert!(
            sink.wait_for(Duration::from_secs(5), |s| s
                .printed_text()
                .contains("hello-pipeline")),
            "expected shell output, got: {:?}",
            sink.printed_text()
        );
        assert!(
            sink.wait_for(Duration::from_secs(5), |s| s
                .child_exited
                .load(Ordering::SeqCst)),
            "reader should report child exit after EOF"
        );

        drop(writer);
        drop(actor); // sends Shutdown x3 and joins all threads
    }

    /// Full loop: app → writer → PTY → child (cat) → reader → parser → sink.
    #[test]
    fn pipeline_round_trips_writes_through_child() {
        let (sink, writer, actor) = spawn_pipeline("/bin/cat", &[]);

        writer
            .send(Message::Data(b"echo-me".to_vec()))
            .expect("write to shell");

        assert!(
            sink.wait_for(Duration::from_secs(5), |s| s
                .printed_text()
                .contains("echo-me")),
            "expected cat to echo input back, got: {:?}",
            sink.printed_text()
        );

        drop(writer);
        drop(actor);
    }

    /// Shutdown works even when the child is still alive and the reader is
    /// blocked in its poll — the waker must interrupt it.
    #[test]
    fn shutdown_interrupts_blocked_reader() {
        let (_sink, writer, actor) = spawn_pipeline("/bin/cat", &[]);

        // Give the reader time to enter its epoll/kqueue wait.
        std::thread::sleep(Duration::from_millis(100));

        let start = Instant::now();
        drop(writer);
        drop(actor); // must not hang
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "shutdown should not block on a live child"
        );
    }
}
