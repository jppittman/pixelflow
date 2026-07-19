// src/io/event_monitor_actor/mod.rs

//! # PTY Actor Troupe
//!
//! The PTY pipeline is an [`actor_scheduler`] troupe of three actors, wired by
//! the [`troupe!`](actor_scheduler::troupe) macro:
//!
//! ```text
//!                    recycled buffers (Data lane)
//!            ┌──────────────────────────────────────┐
//!            ▼                                      │
//!      ┌───────────┐   FilledBuf (Data)    ┌────────┴───┐   AnsiCommands
//!      │ PtyReader │──────────────────────▶│ PtyParser  │───────────────▶ app
//!      │ [main]    │                       │ (message-  │
//!      │ epoll/kq  │── ChildExited ──────▶ │  driven)   │
//!      └───────────┘              (to app) └────────────┘
//!
//!      ┌───────────┐   Data: Write(bytes)
//!      │ PtyWriter │◀─────────────────────  app
//!      │ [expose]  │   Control: Resize      (resize preempts queued writes)
//!      └───────────┘
//! ```
//!
//! **Cross-actor wiring** (reader→parser bytes, parser→reader recycle) comes
//! from the generated `Directory`. **External resources** — the `NixPty`, the
//! per-actor [`FdWaker`], and the app-facing [`PtySender`]s — can't ride the
//! `Directory` (which only knows troupe-internal handles), so they arrive via
//! a `Bind` management message once the troupe is running. This mirrors how
//! `VsyncActor` is configured post-construction. Management drains before Data
//! in every scheduler wake, so an actor is always bound before it handles its
//! first byte.
//!
//! **Blocking model.** The reader and writer bridge to the OS inside `park()`,
//! blocking in `epoll_wait`/`kevent` on `{pty, waker}`; each is a `[waker]`
//! slot so a send interrupts the poll. The parser is message-driven.
//!
//! **Lifecycle.** [`PtyTroupe::new`] wires channels and wakers (no threads),
//! [`PtyTroupe::writer_handle`] mints the app's handle to the writer, and
//! [`PtyTroupe::spawn`] sends the `Bind`s and runs `play()` on a dedicated
//! thread. Dropping the returned [`PtyTroupeHandle`] sends `Shutdown` to each
//! actor (the waker interrupts any in-flight poll) and joins the thread.

mod parser;
mod reader;
mod writer;

use crate::io::pty::NixPty;
use crate::io::traits::PtySender;
use crate::io::waker::FdWaker;
use crate::io::Resize;
use actor_scheduler::{ActorHandle, Message, WakeHandler};
use anyhow::{Context, Result};
use log::*;
use parser::PtyParser;
use reader::PtyReader;
use std::sync::Arc;
use std::thread::JoinHandle;
use writer::PtyWriter;

// ── Lane message types ──────────────────────────────────────────────────────

/// Placeholder for the reader's unused control lane.
#[derive(Debug, Clone)]
pub struct NoControl;

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

/// Control messages for the PTY parser.
#[derive(Debug, Clone)]
pub enum ParserControl {
    /// Discard any half-parsed escape-sequence state.
    Reset,
}

/// Control messages for the PTY writer. Drained before queued Data (writes),
/// which is exactly the priority a resize wants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriterControl {
    /// Set the PTY window size (`TIOCSWINSZ` + `SIGWINCH`).
    Resize(Resize),
}

// ── Bind (management) messages: deliver post-construction resources ──────────

/// Reader management lane: resource delivery.
pub enum ReaderManagement {
    /// Hand the reader its PTY (a read clone), waker, and app sink. Builds the
    /// event monitor and seeds the buffer pool.
    Bind {
        pty: NixPty,
        waker: Arc<FdWaker>,
        app_tx: Box<dyn PtySender>,
    },
}

/// Parser management lane: resource delivery.
pub enum ParserManagement {
    /// Hand the parser its app sink for parsed commands.
    Bind { app_tx: Box<dyn PtySender> },
}

/// Writer management lane: resource delivery.
pub enum WriterManagement {
    /// Hand the writer its primary PTY and waker. Builds the event monitor and
    /// flushes anything the app queued before the bind landed.
    Bind { pty: NixPty, waker: Arc<FdWaker> },
}

// Manual Debug: the payloads (NixPty, FdWaker, Box<dyn PtySender>) aren't all
// Debug, and the scheduler never formats these anyway.
impl std::fmt::Debug for ReaderManagement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ReaderManagement::Bind")
    }
}
impl std::fmt::Debug for ParserManagement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ParserManagement::Bind")
    }
}
impl std::fmt::Debug for WriterManagement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("WriterManagement::Bind")
    }
}

/// The app's handle to the PTY writer: `Data` = bytes for the shell,
/// `Control` = resize.
pub type PtyWriterHandle = ActorHandle<Vec<u8>, WriterControl, WriterManagement>;

// ── Troupe declaration ──────────────────────────────────────────────────────

// All three are [expose] so `PtyTroupe` can mint handles to send Bind and
// Shutdown. Reader and writer are [waker] because they block in park().
actor_scheduler::troupe! {
    reader: PtyReader [main, expose, waker],
    parser: PtyParser [expose],
    writer: PtyWriter [expose, waker],
}

// ── Public wrapper ──────────────────────────────────────────────────────────

/// Phase 1 of PTY pipeline construction: channels and wakers exist, no threads.
///
/// Two-phase so the app can be handed its writer handle before the pipeline
/// starts (the pipeline in turn needs `PtySender`s minted by the app's builder
/// — a cycle two-phase construction breaks).
pub struct PtyTroupe {
    troupe: Option<Troupe>,
    pty: Option<NixPty>,
    reader_waker: Arc<FdWaker>,
    writer_waker: Arc<FdWaker>,
}

impl PtyTroupe {
    /// Create the troupe scaffolding around a freshly spawned PTY.
    ///
    /// `pty` is the primary master handle; the writer actor takes ownership of
    /// it (RAII: closing it hangs up the child).
    pub fn new(pty: NixPty) -> Result<Self> {
        let reader_waker = Arc::new(FdWaker::new().context("Failed to create reader waker")?);
        let writer_waker = Arc::new(FdWaker::new().context("Failed to create writer waker")?);

        let troupe = Troupe::new_with_wakers(Wakers {
            reader: Some(reader_waker.clone() as Arc<dyn WakeHandler>),
            writer: Some(writer_waker.clone() as Arc<dyn WakeHandler>),
        });

        Ok(Self {
            troupe: Some(troupe),
            pty: Some(pty),
            reader_waker,
            writer_waker,
        })
    }

    /// Mint a handle to the PTY writer for the app.
    pub fn writer_handle(&mut self) -> PtyWriterHandle {
        self.troupe
            .as_mut()
            .expect("PtyTroupe already spawned")
            .exposed()
            .writer
    }

    /// Bind the actors to their resources and run the troupe on a dedicated
    /// thread.
    ///
    /// * `parser_sink` - the parser's route to the app (parsed `AnsiCommand`s)
    /// * `reader_sink` - the reader's route to the app (`ChildExited`)
    pub fn spawn(
        mut self,
        parser_sink: Box<dyn PtySender>,
        reader_sink: Box<dyn PtySender>,
    ) -> Result<PtyTroupeHandle> {
        let mut troupe = self.troupe.take().expect("PtyTroupe already spawned");
        let pty = self.pty.take().expect("PtyTroupe already spawned");
        let pty_read = pty.try_clone().context("Failed to clone PTY for reader")?;

        // One handle set for both Bind delivery and later Shutdown.
        let ctl = troupe.exposed();

        // Bind lands on the Management lane, drained before any Data, so each
        // actor is fully configured before it processes its first byte.
        ctl.reader
            .send(Message::Management(ReaderManagement::Bind {
                pty: pty_read,
                waker: self.reader_waker.clone(),
                app_tx: reader_sink,
            }))
            .context("Failed to bind PTY reader")?;
        ctl.parser
            .send(Message::Management(ParserManagement::Bind {
                app_tx: parser_sink,
            }))
            .context("Failed to bind PTY parser")?;
        ctl.writer
            .send(Message::Management(WriterManagement::Bind {
                pty,
                waker: self.writer_waker.clone(),
            }))
            .context("Failed to bind PTY writer")?;

        let join = std::thread::Builder::new()
            .name("pty-troupe".to_string())
            .spawn(move || {
                if let Err(e) = troupe.play() {
                    error!("PTY troupe exited with error: {}", e);
                }
            })
            .context("Failed to spawn PTY troupe thread")?;

        info!("PTY troupe spawned: reader, parser, writer");
        Ok(PtyTroupeHandle {
            ctl,
            join: Some(join),
        })
    }
}

/// Running PTY troupe. Dropping it shuts the actors down and joins the thread.
pub struct PtyTroupeHandle {
    ctl: ExposedHandles,
    join: Option<JoinHandle<()>>,
}

impl Drop for PtyTroupeHandle {
    fn drop(&mut self) {
        debug!("PtyTroupeHandle dropped, shutting down PTY actors");

        // Each Shutdown also rings that actor's waker, interrupting any poll.
        // Errors mean the actor already exited.
        for (name, result) in [
            ("writer", self.ctl.writer.send(Message::Shutdown)),
            ("reader", self.ctl.reader.send(Message::Shutdown)),
            ("parser", self.ctl.parser.send(Message::Shutdown)),
        ] {
            if let Err(e) = result {
                debug!("PTY {} already gone at shutdown: {}", name, e);
            }
        }

        if let Some(join) = self.join.take() {
            if let Err(panic_payload) = join.join() {
                if std::thread::panicking() {
                    eprintln!(
                        "PTY troupe thread panicked (during unwind): {:?}",
                        panic_payload
                    );
                } else {
                    std::panic::resume_unwind(panic_payload);
                }
            }
        }

        debug!("PTY troupe cleanup complete");
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

    fn spawn_troupe(
        command: &str,
        args: &[&str],
    ) -> (CaptureSink, PtyWriterHandle, PtyTroupeHandle) {
        let pty = NixPty::spawn_with_config(&PtyConfig {
            command_executable: command,
            args,
            initial_cols: 80,
            initial_rows: 24,
        })
        .expect("Failed to spawn PTY");

        let sink = CaptureSink::default();
        let mut troupe = PtyTroupe::new(pty).expect("troupe");
        let writer = troupe.writer_handle();
        let handle = troupe
            .spawn(Box::new(sink.clone()), Box::new(sink.clone()))
            .expect("spawn troupe");
        (sink, writer, handle)
    }

    #[test]
    fn troupe_delivers_output_and_child_exit() {
        let (sink, writer, handle) = spawn_troupe("/bin/sh", &["-c", "printf 'hello-troupe'"]);

        assert!(
            sink.wait_for(Duration::from_secs(5), |s| s
                .printed_text()
                .contains("hello-troupe")),
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
        drop(handle);
    }

    #[test]
    fn troupe_round_trips_writes_through_child() {
        let (sink, writer, handle) = spawn_troupe("/bin/cat", &[]);

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
        drop(handle);
    }

    #[test]
    fn shutdown_interrupts_blocked_reader() {
        let (_sink, writer, handle) = spawn_troupe("/bin/cat", &[]);

        // Give the reader time to enter its epoll/kqueue wait.
        std::thread::sleep(Duration::from_millis(100));

        let start = Instant::now();
        drop(writer);
        drop(handle); // must not hang
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "shutdown should not block on a live child"
        );
    }

    #[test]
    fn resize_before_bind_is_applied() {
        // A resize sent on the writer handle immediately (likely before the
        // Bind lands, since Control outranks Management) must not be lost or
        // panic — the writer coalesces it and applies it at Bind.
        let pty = NixPty::spawn_with_config(&PtyConfig {
            command_executable: "/bin/cat",
            args: &[],
            initial_cols: 80,
            initial_rows: 24,
        })
        .expect("pty");
        let sink = CaptureSink::default();
        let mut troupe = PtyTroupe::new(pty).expect("troupe");
        let writer = troupe.writer_handle();

        // Fire a resize before spawn() even sends the Bind.
        writer
            .send(Message::Control(WriterControl::Resize(Resize {
                cols: 100,
                rows: 40,
            })))
            .expect("early resize");

        let handle = troupe
            .spawn(Box::new(sink.clone()), Box::new(sink.clone()))
            .expect("spawn");

        // If the early resize wedged the writer, a subsequent echo wouldn't
        // round-trip. It does, so the writer survived the pre-bind resize.
        writer
            .send(Message::Data(b"after-resize".to_vec()))
            .expect("write");
        assert!(
            sink.wait_for(Duration::from_secs(5), |s| s
                .printed_text()
                .contains("after-resize")),
            "writer should work after a pre-bind resize"
        );

        drop(writer);
        drop(handle);
    }
}
