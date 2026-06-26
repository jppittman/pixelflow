// src/io/event_monitor_actor/write_thread/mod.rs

use crate::io::pty::{NixPty, PtyChannel};
use crate::io::PtyCommand;
use log::*;
use std::io::Write;
use std::sync::mpsc::Receiver;
use std::thread::JoinHandle;

pub struct WriteThread {
    join_handle: Option<JoinHandle<()>>,
}

impl WriteThread {
    /// Spawns the write thread.
    ///
    /// # Arguments
    ///
    /// * `mut pty` - The primary PTY handle (will be closed when thread exits)
    /// * `rx` - Channel to receive commands (writes and resizes)
    pub fn spawn(mut pty: NixPty, rx: Receiver<PtyCommand>) -> anyhow::Result<Self> {
        let handle = std::thread::Builder::new()
            .name("pty-writer".to_string())
            .spawn(move || {
                debug!("Write thread started");
                while let Ok(cmd) = rx.recv() {
                    match cmd {
                        PtyCommand::Write(data) => {
                            if let Err(e) = pty.write_all(&data) {
                                error!("Failed to write to PTY: {}", e);
                                break;
                            }
                            if let Err(e) = pty.flush() {
                                error!("Failed to flush PTY: {}", e);
                                break;
                            }
                        }
                        PtyCommand::Resize(resize) => {
                            if let Err(e) = pty.resize(resize.cols, resize.rows) {
                                // Log but don't break - resize failures are non-fatal
                                // (e.g., child process may have exited)
                                warn!(
                                    "Failed to resize PTY to {}x{}: {}",
                                    resize.cols, resize.rows, e
                                );
                            } else {
                                debug!("PTY resized to {}x{}", resize.cols, resize.rows);
                            }
                        }
                    }
                }
                debug!("Write thread exited - closing PTY");
                // PTY is dropped here, closing the FD
            })?;

        Ok(Self {
            join_handle: Some(handle),
        })
    }
}

impl Drop for WriteThread {
    fn drop(&mut self) {
        // Closing the channel (which happens when SyncSender is dropped by owner)
        // will cause the loop to exit.
        if let Some(handle) = self.join_handle.take() {
            if let Err(panic_payload) = handle.join() {
                if std::thread::panicking() {
                    eprintln!("Write thread panicked (during unwind): {:?}", panic_payload);
                } else {
                    std::panic::resume_unwind(panic_payload);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::pty::PtyConfig;
    use std::sync::mpsc::sync_channel;

    #[test]
    fn test_write_thread_handles_resize_command() {
        // Create a real PTY for testing
        let config = PtyConfig {
            command_executable: "/bin/cat",
            args: &[],
            initial_cols: 80,
            initial_rows: 24,
        };

        let pty = NixPty::spawn_with_config(&config).expect("Failed to spawn PTY");
        let (tx, rx) = sync_channel::<PtyCommand>(16);

        let write_thread = WriteThread::spawn(pty, rx).expect("Failed to spawn write thread");

        // Send resize command
        tx.send(PtyCommand::Resize(crate::io::Resize {
            cols: 120,
            rows: 40,
        }))
        .expect("Failed to send resize");

        // Send some data
        tx.send(PtyCommand::Write(b"hello".to_vec()))
            .expect("Failed to send write");

        // Drop sender to close the channel and terminate the thread
        drop(tx);

        // Thread should exit cleanly
        drop(write_thread);
    }

    #[test]
    fn test_write_thread_handles_write_command() {
        let config = PtyConfig {
            command_executable: "/bin/cat",
            args: &[],
            initial_cols: 80,
            initial_rows: 24,
        };

        let pty = NixPty::spawn_with_config(&config).expect("Failed to spawn PTY");
        let (tx, rx) = sync_channel::<PtyCommand>(16);

        let write_thread = WriteThread::spawn(pty, rx).expect("Failed to spawn write thread");

        // Send multiple write commands
        tx.send(PtyCommand::Write(b"line1\n".to_vec()))
            .expect("Failed to send");
        tx.send(PtyCommand::Write(b"line2\n".to_vec()))
            .expect("Failed to send");

        drop(tx);
        drop(write_thread);
    }
}
