// src/io/event_monitor_actor/writer.rs

//! PTY writer actor.
//!
//! Owns the primary PTY FD (RAII: dropping it hangs up the child). Bytes for
//! the shell arrive on the data lane; `Resize` arrives on the control lane and
//! therefore preempts any queued bulk writes — the scheduler drains Control
//! before Data by construction.
//!
//! Writes are nonblocking. If the kernel PTY buffer fills (a stopped shell mid
//! large paste), the unwritten tail is queued in `pending` and `park()` waits
//! for `EPOLLOUT` — with the waker registered alongside, so Resize and
//! Shutdown still get through while the queue drains. The old thread's
//! blocking `write_all` wedged everything behind a full buffer.

use super::{NoManagement, WriterControl};
use crate::io::event::{EventFlags, EventMonitor};
use crate::io::pty::{NixPty, PtyChannel};
use crate::io::waker::FdWaker;
use actor_scheduler::{Actor, ActorStatus, HandlerError, HandlerResult, SystemStatus};
use log::*;
use std::collections::VecDeque;
use std::io::Write;
use std::sync::Arc;

const TOKEN_PTY: u64 = 0;
const TOKEN_WAKER: u64 = 1;

pub(super) struct PtyWriter {
    pty: NixPty,
    monitor: EventMonitor,
    waker: Arc<FdWaker>,
    /// Bytes accepted but not yet written to the PTY.
    pending: VecDeque<Vec<u8>>,
    /// Offset of the first unwritten byte in `pending.front()`.
    cursor: usize,
    /// Set on unrecoverable write errors (child gone); further writes drop.
    broken: bool,
    events: Vec<crate::io::event::Event>,
}

impl PtyWriter {
    /// Builds the writer and registers the PTY (write interest) and waker.
    /// Called on the writer thread (the monitor stays thread-local).
    pub(super) fn new(pty: NixPty, waker: Arc<FdWaker>) -> anyhow::Result<Self> {
        let monitor = EventMonitor::new()?;
        // Registered permanently; park() only polls while `pending` is
        // non-empty, so level-triggered "always writable" costs nothing.
        monitor.add(&pty, TOKEN_PTY, EventFlags::EPOLLOUT)?;
        monitor.add(&*waker, TOKEN_WAKER, EventFlags::EPOLLIN)?;
        Ok(Self {
            pty,
            monitor,
            waker,
            pending: VecDeque::new(),
            cursor: 0,
            broken: false,
            events: Vec::with_capacity(16),
        })
    }

    /// Write queued bytes until done or the kernel buffer is full.
    fn flush_pending(&mut self) {
        while let Some(front) = self.pending.front() {
            match self.pty.write(&front[self.cursor..]) {
                Ok(n) => {
                    self.cursor += n;
                    if self.cursor == front.len() {
                        self.pending.pop_front();
                        self.cursor = 0;
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return,
                Err(e) => {
                    error!("PTY write error, dropping {} queued buffers: {}", self.pending.len(), e);
                    self.pending.clear();
                    self.cursor = 0;
                    self.broken = true;
                    return;
                }
            }
        }
    }
}

impl Actor<Vec<u8>, WriterControl, NoManagement> for PtyWriter {
    fn handle_data(&mut self, bytes: Vec<u8>) -> HandlerResult {
        if self.broken || bytes.is_empty() {
            return Ok(());
        }
        self.pending.push_back(bytes);
        // Common case: queue was empty and the PTY is writable, so this
        // completes immediately and park() never needs to poll.
        self.flush_pending();
        Ok(())
    }

    fn handle_control(&mut self, msg: WriterControl) -> HandlerResult {
        match msg {
            WriterControl::Resize(resize) => {
                // Resize failures are non-fatal (child may have exited).
                if let Err(e) = self.pty.resize(resize.cols, resize.rows) {
                    warn!(
                        "Failed to resize PTY to {}x{}: {}",
                        resize.cols, resize.rows, e
                    );
                } else {
                    debug!("PTY resized to {}x{}", resize.cols, resize.rows);
                }
            }
        }
        Ok(())
    }

    fn handle_management(&mut self, _msg: NoManagement) -> HandlerResult {
        Ok(())
    }

    fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        if self.broken || self.pending.is_empty() {
            // Message-driven: let the scheduler block on the doorbell.
            return Ok(ActorStatus::Idle);
        }

        // Kernel buffer was full; wait until it drains or a message arrives.
        self.monitor
            .events(&mut self.events, -1)
            .map_err(|e| HandlerError::recoverable(format!("PTY writer poll failed: {e}")))?;

        let mut pty_writable = false;
        for event in &self.events {
            match event.token {
                TOKEN_WAKER => self.waker.drain(),
                TOKEN_PTY => pty_writable = true,
                other => debug!("PTY writer: unexpected poll token {}", other),
            }
        }

        if pty_writable {
            self.flush_pending();
        }

        Ok(ActorStatus::Busy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::pty::PtyConfig;
    use crate::io::Resize;
    use actor_scheduler::{ActorScheduler, Message};

    fn spawn_cat_pty() -> NixPty {
        let config = PtyConfig {
            command_executable: "/bin/cat",
            args: &[],
            initial_cols: 80,
            initial_rows: 24,
        };
        NixPty::spawn_with_config(&config).expect("Failed to spawn PTY")
    }

    fn run_writer(
        pty: NixPty,
        waker: Arc<FdWaker>,
        mut rx: ActorScheduler<Vec<u8>, WriterControl, NoManagement>,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let mut writer = PtyWriter::new(pty, waker).expect("writer construction");
            rx.run(&mut writer);
        })
    }

    #[test]
    fn writer_handles_resize_and_write() {
        let waker = Arc::new(FdWaker::new().expect("waker"));
        let (tx, rx) = ActorScheduler::new_with_wake_handler(10, 16, Some(waker.clone()));

        let handle = run_writer(spawn_cat_pty(), waker, rx);

        tx.send(Message::Control(WriterControl::Resize(Resize {
            cols: 120,
            rows: 40,
        })))
        .expect("send resize");
        tx.send(Message::Data(b"hello".to_vec())).expect("send write");

        tx.send(Message::Shutdown).expect("send shutdown");
        handle.join().expect("writer thread");
    }

    #[test]
    fn writer_exits_when_handles_drop() {
        let waker = Arc::new(FdWaker::new().expect("waker"));
        let (tx, rx) = ActorScheduler::new_with_wake_handler(10, 16, Some(waker.clone()));

        let handle = run_writer(spawn_cat_pty(), waker, rx);

        tx.send(Message::Data(b"line1\n".to_vec())).expect("send");
        drop(tx);

        handle.join().expect("writer thread");
    }
}
