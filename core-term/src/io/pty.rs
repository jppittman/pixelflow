// src/os/pty.rs

use anyhow::{Context, Result};
use std::ffi::CString;
use std::io::{Read, Result as IoResult, Write};
use std::os::unix::io::{AsFd, AsRawFd, OwnedFd, RawFd};
use std::sync::Arc;

use nix::fcntl::{fcntl, FcntlArg, FdFlag, OFlag};
use nix::pty::openpty;
use nix::sys::signal::{kill, Signal};
use nix::sys::termios;
use nix::sys::wait::{waitpid, WaitPidFlag, WaitStatus};
use nix::unistd::Pid;
use std::io::{Error as IoError, ErrorKind as IoErrorKind};

/// `POSIX_SPAWN_SETSID` — the libc crate doesn't expose it for Apple targets.
/// Value from macOS `<spawn.h>` (available since 10.13).
#[cfg(target_os = "macos")]
const POSIX_SPAWN_SETSID: libc::c_int = 0x0400;
#[cfg(not(target_os = "macos"))]
use libc::POSIX_SPAWN_SETSID;

/// Upholds [`NixPty::spawn_with_config`]'s single-threaded contract inside
/// the test binary, where libtest runs the crate's PTY-spawning tests as
/// parallel threads.
///
/// Test-only on purpose. Production honours the contract structurally —
/// one PTY, spawned on the main thread — so shipping a lock for it would
/// be machinery for a race that cannot occur. Every in-crate spawn site
/// (`io::pty_tests`, `io::event_monitor_actor`, `terminal_app`) funnels
/// through `spawn_with_config`, so gating it here covers them all; no
/// integration test spawns a PTY, which is what makes `cfg(test)` a
/// sufficient boundary.
#[cfg(test)]
static OPENPTY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Configuration for spawning a PTY.
#[derive(Debug, Clone)]
pub struct PtyConfig<'a> {
    /// The executable to run (e.g., "/bin/bash").
    pub command_executable: &'a str,
    /// Arguments to the executable.
    pub args: &'a [&'a str],
    /// Initial columns.
    pub initial_cols: u16,
    /// Initial rows.
    pub initial_rows: u16,
}

/// Trait abstracting a PTY channel.
///
/// Allows reading/writing data and managing the PTY session (resizing, PID access).
pub trait PtyChannel: Read + Write + AsRawFd + Send + Sync {
    /// Resizes the PTY window.
    ///
    /// # Parameters
    /// * `cols` - New width in columns.
    /// * `rows` - New height in rows.
    fn resize(&self, cols: u16, rows: u16) -> Result<()>;

    /// Returns the process ID of the child process attached to the PTY.
    fn child_pid(&self) -> Pid;
}

/// Implementation of `PtyChannel` using `nix` for POSIX systems.
#[derive(Debug)]
pub struct NixPty {
    master_fd: Arc<OwnedFd>,
    child_pid: Option<Pid>,
}

impl NixPty {
    fn set_pty_size_internal<Fd: AsFd>(fd: Fd, cols: u16, rows: u16) -> anyhow::Result<()> {
        use nix::pty::Winsize;
        let raw_fd = fd.as_fd().as_raw_fd();
        nix::ioctl_write_ptr_bad!(tcsetwinsize, nix::libc::TIOCSWINSZ, Winsize);
        let winsize = Winsize {
            ws_row: rows,
            ws_col: cols,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        unsafe { tcsetwinsize(raw_fd, &winsize) }
            .map_err(|e| anyhow::anyhow!("ioctl TIOCSWINSZ failed for fd {}: {}", raw_fd, e))?;
        log::trace!("NixPty: Set PTY fd {} size to {}x{}", raw_fd, cols, rows);
        Ok(())
    }

    /// Spawns a new process connected to a PTY using the given configuration.
    ///
    /// The child is created with `posix_spawnp`, **never `fork`**. This
    /// process is heavily threaded (actor system; libtest runs tests as
    /// threads), and on macOS `fork` runs libmalloc's atfork handlers, which
    /// take every malloc zone lock. A concurrent fork + allocation reliably
    /// wedges the whole process at 0% CPU (`_xzm_fork_lock_wait` in the
    /// forking thread, `_os_unfair_lock_lock_slow` under `malloc` everywhere
    /// else). `posix_spawn` runs no user-space code in the child: on macOS it
    /// is a kernel-side spawn, on Linux glibc a vfork-style `clone` executing
    /// only direct syscalls — no atfork handlers ever fire.
    ///
    /// The old `pre_exec` work maps onto spawn-native mechanisms:
    /// - `setsid()`   → `POSIX_SPAWN_SETSID` attribute flag
    /// - `TIOCSCTTY`  → the child *opens the slave path* as fd 0 after
    ///   `setsid`: a session leader's first tty open acquires it as the
    ///   controlling terminal on both Linux and the BSDs
    /// - `dup2(0/1/2)` → file actions (`addopen` + `adddup2`)
    ///
    /// # Threading
    ///
    /// **Call this from one thread.** It is not safe to call concurrently:
    /// `openpty(3)` is not thread-safe on macOS, and racing calls fail
    /// intermittently with a corrupted (negative) errno that surfaces as
    /// `Errno::UnknownErrno` — 107 failures in 57,600 calls at 48 threads
    /// on 12 cores, versus 0 when serialized. See
    /// `docs/bugs/2026-07-21-openpty-not-thread-safe.md`.
    ///
    /// The terminal satisfies this structurally rather than by locking:
    /// `main` spawns the single PTY before the troupe's threads exist and
    /// hands it to `PtyTroupe::new`, which delivers it to the actors as a
    /// `Bind` message. If a second PTY is ever needed (tabs, splits), give
    /// spawning a single owner rather than making this function reentrant.
    ///
    /// # Parameters
    /// * `config` - Configuration for the PTY and command.
    ///
    /// # Returns
    /// * A new `NixPty` instance in the parent process.
    pub fn spawn_with_config(config: &PtyConfig) -> Result<Self> {
        let pty_results = {
            #[cfg(test)]
            let _guard = OPENPTY_TEST_LOCK.lock().expect(
                "OPENPTY_TEST_LOCK poisoned: a prior openpty call panicked while holding it",
            );
            openpty(None, None).with_context(|| "Failed to open PTY (nix::pty::openpty call)")?
        };
        let master_fd = pty_results.master;
        let slave_fd = pty_results.slave;

        // Neither parent fd may leak into the child: it re-opens the slave by
        // path, and an inherited master would keep the PTY alive after we
        // close ours.
        Self::set_cloexec(&master_fd)?;
        Self::set_cloexec(&slave_fd)?;

        // Configure slave PTY attributes (in the parent, operating on the slave FD).
        // This is safe because slave_fd refers to the same underlying PTY.
        let mut termios_attrs =
            termios::tcgetattr(&slave_fd).with_context(|| "Failed to get terminal attributes")?;
        termios::cfmakeraw(&mut termios_attrs);
        termios_attrs.local_flags |= termios::LocalFlags::ISIG;
        termios_attrs.input_flags |= termios::InputFlags::ICRNL;
        termios::tcsetattr(&slave_fd, termios::SetArg::TCSANOW, &termios_attrs)
            .with_context(|| "Failed to set terminal attributes to raw mode")?;

        // Size the PTY before the child starts so the shell's first size
        // query already sees the real dimensions.
        Self::set_pty_size_internal(&slave_fd, config.initial_cols, config.initial_rows)
            .with_context(|| "Failed to set initial PTY size")?;

        log::debug!(
            "Spawning command: {} with args {:?}",
            config.command_executable,
            config.args
        );

        let child_pid = Self::posix_spawn_child(config, &slave_fd)?;
        log::debug!(
            "Parent: Spawned child with PID {}, PTY master FD {}",
            child_pid,
            master_fd.as_raw_fd()
        );

        Self::set_fd_nonblocking(&master_fd)
            .with_context(|| "Parent: Failed to set master PTY to non-blocking")?;

        // slave_fd is dropped here, closing the parent's handle to the slave PTY.
        // The child has its own copies (0, 1, 2).

        Ok(NixPty {
            master_fd: Arc::new(master_fd),
            child_pid: Some(child_pid),
        })
    }

    /// Spawn the PTY child via `posix_spawnp`. See [`spawn_with_config`]
    /// for why this must not fork.
    ///
    /// [`spawn_with_config`]: Self::spawn_with_config
    fn posix_spawn_child(config: &PtyConfig, slave_fd: &OwnedFd) -> Result<Pid> {
        use std::os::unix::ffi::OsStringExt;

        // All heap work happens here in the parent, before the spawn call.
        let slave_path =
            nix::unistd::ttyname(slave_fd.as_fd()).context("Failed to resolve PTY slave path")?;
        let slave_path = CString::new(slave_path.into_os_string().into_vec())
            .context("PTY slave path contains NUL")?;
        let exe =
            CString::new(config.command_executable).context("Command executable contains NUL")?;
        let mut argv: Vec<CString> = Vec::with_capacity(config.args.len() + 1);
        argv.push(exe.clone());
        for arg in config.args {
            argv.push(CString::new(*arg).context("Command argument contains NUL")?);
        }
        // Inherit our environment (a NUL inside a var can't cross exec; skip it).
        let env: Vec<CString> = std::env::vars_os()
            .filter_map(|(key, value)| {
                let mut kv = key.into_vec();
                kv.push(b'=');
                kv.extend(value.into_vec());
                CString::new(kv).ok()
            })
            .collect();

        let mut argv_ptrs: Vec<*mut libc::c_char> = argv
            .iter()
            .map(|s| s.as_ptr() as *mut libc::c_char)
            .collect();
        argv_ptrs.push(std::ptr::null_mut());
        let mut env_ptrs: Vec<*mut libc::c_char> = env
            .iter()
            .map(|s| s.as_ptr() as *mut libc::c_char)
            .collect();
        env_ptrs.push(std::ptr::null_mut());

        let mut pid: libc::pid_t = 0;
        let rc = unsafe {
            let mut actions: libc::posix_spawn_file_actions_t = std::mem::zeroed();
            let mut attr: libc::posix_spawnattr_t = std::mem::zeroed();
            libc::posix_spawn_file_actions_init(&mut actions);
            libc::posix_spawnattr_init(&mut attr);

            // Child (in order): setsid via attr flag, then open the slave as
            // stdin — the open is what acquires the controlling terminal,
            // since by then the child is a session leader — then clone it to
            // stdout/stderr.
            libc::posix_spawn_file_actions_addopen(
                &mut actions,
                libc::STDIN_FILENO,
                slave_path.as_ptr(),
                libc::O_RDWR,
                0,
            );
            libc::posix_spawn_file_actions_adddup2(
                &mut actions,
                libc::STDIN_FILENO,
                libc::STDOUT_FILENO,
            );
            libc::posix_spawn_file_actions_adddup2(
                &mut actions,
                libc::STDIN_FILENO,
                libc::STDERR_FILENO,
            );

            // Reset every signal disposition and unblock everything. Rust
            // sets SIGPIPE to SIG_IGN, and ignored dispositions survive
            // exec — without this, `foo | head` pipelines in the shell
            // would never terminate on a closed pipe.
            let mut default_signals: libc::sigset_t = std::mem::zeroed();
            libc::sigfillset(&mut default_signals);
            libc::posix_spawnattr_setsigdefault(&mut attr, &default_signals);
            let mut empty_mask: libc::sigset_t = std::mem::zeroed();
            libc::sigemptyset(&mut empty_mask);
            libc::posix_spawnattr_setsigmask(&mut attr, &empty_mask);
            libc::posix_spawnattr_setflags(
                &mut attr,
                (POSIX_SPAWN_SETSID | libc::POSIX_SPAWN_SETSIGDEF | libc::POSIX_SPAWN_SETSIGMASK)
                    as libc::c_short,
            );

            let rc = libc::posix_spawnp(
                &mut pid,
                exe.as_ptr(),
                &actions,
                &attr,
                argv_ptrs.as_ptr(),
                env_ptrs.as_ptr(),
            );

            libc::posix_spawn_file_actions_destroy(&mut actions);
            libc::posix_spawnattr_destroy(&mut attr);
            rc
        };

        if rc != 0 {
            // posix_spawn reports errors as a return value, not via errno.
            return Err(IoError::from_raw_os_error(rc)).with_context(|| {
                format!("Failed to spawn command '{}'", config.command_executable)
            });
        }
        Ok(Pid::from_raw(pid))
    }

    /// Creates a clone of the PTY handle for reading.
    /// The clone shares the file descriptor but does not own the child process.
    pub fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            master_fd: self.master_fd.clone(),
            child_pid: None,
        })
    }

    /// Helper to spawn a shell command (deprecated/unimplemented convenience method).
    pub fn spawn_shell_command(
        _shell_command_str: &str,
        _initial_cols: u16,
        _initial_rows: u16,
    ) -> Result<Self> {
        unimplemented!("spawn_shell_command is not fully implemented with OwnedFd yet.");
    }

    fn set_cloexec<Fd: AsFd>(fd: Fd) -> Result<()> {
        let raw_fd = fd.as_fd().as_raw_fd();
        fcntl(fd.as_fd(), FcntlArg::F_SETFD(FdFlag::FD_CLOEXEC))
            .with_context(|| format!("Failed to set FD_CLOEXEC for fd {}", raw_fd))?;
        log::trace!("NixPty: Set FD_CLOEXEC on fd {}", raw_fd);
        Ok(())
    }

    fn set_fd_nonblocking<Fd: AsFd>(fd: Fd) -> Result<()> {
        let raw_fd = fd.as_fd().as_raw_fd();
        let flags = fcntl(fd.as_fd(), FcntlArg::F_GETFL)
            .with_context(|| format!("Failed to get FD flags for fd {}", raw_fd))?;
        let mut non_blocking_flags = OFlag::from_bits_truncate(flags);
        non_blocking_flags.insert(OFlag::O_NONBLOCK);
        fcntl(fd.as_fd(), FcntlArg::F_SETFL(non_blocking_flags))
            .with_context(|| format!("Failed to set FD {} to non-blocking", raw_fd))?;
        log::trace!("NixPty: Set FD {} to non-blocking", raw_fd);
        Ok(())
    }

    /// Terminates the child process.
    #[allow(dead_code)]
    pub fn terminate_child_process(&mut self) -> Result<()> {
        if let Some(pid) = self.child_pid {
            log::info!("Terminating child process {}", pid);
            kill(pid, Some(Signal::SIGKILL))
                .with_context(|| format!("Failed to send SIGKILL to child process {}", pid))
        } else {
            Ok(())
        }
    }
}

impl Drop for NixPty {
    fn drop(&mut self) {
        let master_raw_fd = self.master_fd.as_raw_fd();
        log::debug!(
            "NixPty drop: Cleaning up PTY master_fd: {} (child_pid: {:?})",
            master_raw_fd,
            self.child_pid
        );
        // self.master_fd (Arc<OwnedFd>) is dropped automatically.
        // The underlying FD closes only when strong_count hits 0.

        let pid = match self.child_pid {
            Some(p) => p,
            None => {
                // This is a clone (Read Thread), so we don't manage the child process.
                return;
            }
        };

        if pid.as_raw() <= 0 {
            log::debug!(
                "NixPty drop: Invalid child PID ({}), skipping child process handling.",
                pid
            );
            return;
        }

        match waitpid(pid, Some(WaitPidFlag::WNOHANG)) {
            Ok(WaitStatus::StillAlive) => {
                log::debug!(
                    "NixPty drop: Child process {} is still alive. Sending SIGHUP.",
                    pid
                );
                if let Err(e) = kill(pid, Some(Signal::SIGHUP)) {
                    log::warn!(
                        "NixPty drop: Failed to send SIGHUP to child process {}: {}",
                        pid,
                        e
                    );
                } else {
                    log::debug!(
                        "NixPty drop: Successfully sent SIGHUP to child process {}.",
                        pid
                    );
                }
            }
            Ok(status) => {
                log::debug!(
                    "NixPty drop: Child process {} already exited or changed state: {:?}",
                    pid,
                    status
                );
            }
            Err(e) => {
                // nix::Error
                if matches!(e, nix::Error::ECHILD) || matches!(e, nix::Error::ESRCH) {
                    log::debug!(
                        "NixPty drop: Child process {} does not exist or is not a child (waitpid error: {}). Already reaped?",
                        pid,
                        e
                    );
                } else {
                    log::warn!(
                        "NixPty drop: Error checking child process {} status with waitpid: {}",
                        pid,
                        e
                    );
                }
            }
        }
    }
}

impl Read for NixPty {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
        let master_raw_fd = self.master_fd.as_raw_fd();
        log::trace!("NixPty::read attempting to read from fd {}", master_raw_fd);
        match nix::unistd::read(&self.master_fd, buf) {
            // Pass &OwnedFd
            Ok(bytes_read) => {
                log::trace!(
                    "NixPty::read successfully read {} bytes from fd {}",
                    bytes_read,
                    master_raw_fd
                );
                Ok(bytes_read)
            }
            Err(nix::Error::EIO) => Ok(0),
            Err(nix_err) => {
                if matches!(nix_err, nix::Error::EAGAIN)
                    || matches!(nix_err, nix::Error::EWOULDBLOCK)
                {
                    log::debug!(
                        "NixPty::read on fd {}: Got {}, mapping to WouldBlock",
                        master_raw_fd,
                        nix_err
                    );
                    Err(IoError::new(IoErrorKind::WouldBlock, nix_err))
                } else {
                    log::warn!(
                        "NixPty::read on fd {}: Got unhandled nix::Error {}, mapping to Other",
                        master_raw_fd,
                        nix_err
                    );
                    Err(IoError::other(nix_err))
                }
            }
        }
    }
}

impl Write for NixPty {
    fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        let master_raw_fd = self.master_fd.as_raw_fd();
        log::trace!(
            "NixPty::write attempting to write {} bytes to fd {}",
            buf.len(),
            master_raw_fd
        );
        match nix::unistd::write(&self.master_fd, buf) {
            // Pass &OwnedFd
            Ok(bytes_written) => {
                log::trace!(
                    "NixPty::write successfully wrote {} bytes to fd {}",
                    bytes_written,
                    master_raw_fd
                );
                Ok(bytes_written)
            }
            Err(nix_err) => {
                if matches!(nix_err, nix::Error::EAGAIN)
                    || matches!(nix_err, nix::Error::EWOULDBLOCK)
                {
                    log::debug!(
                        "NixPty::write on fd {}: Got {}, mapping to WouldBlock",
                        master_raw_fd,
                        nix_err
                    );
                    Err(IoError::new(IoErrorKind::WouldBlock, nix_err))
                } else {
                    log::warn!(
                        "NixPty::write on fd {}: Got unhandled nix::Error {}, mapping to Other",
                        master_raw_fd,
                        nix_err
                    );
                    Err(IoError::other(nix_err))
                }
            }
        }
    }

    fn flush(&mut self) -> IoResult<()> {
        log::trace!("NixPty::flush called for fd {}", self.master_fd.as_raw_fd());
        Ok(())
    }
}

impl AsRawFd for NixPty {
    fn as_raw_fd(&self) -> RawFd {
        self.master_fd.as_raw_fd()
    }
}

impl PtyChannel for NixPty {
    fn resize(&self, cols: u16, rows: u16) -> anyhow::Result<()> {
        Self::set_pty_size_internal(&*self.master_fd, cols, rows).with_context(|| {
            format!(
                "NixPty: Failed to set PTY size to {}x{} for fd {}",
                cols,
                rows,
                self.master_fd.as_raw_fd()
            )
        })?;

        if let Some(pid) = self.child_pid {
            kill(pid, Some(Signal::SIGWINCH)).with_context(|| {
                format!("NixPty: Failed to send SIGWINCH to child process {}", pid)
            })?;

            log::debug!(
                "NixPty: Resized PTY to {}x{} and sent SIGWINCH to PID {}",
                cols,
                rows,
                pid
            );
        } else {
            log::warn!(
                "NixPty: Resize called on a clone (no PID). PTY size set, but SIGWINCH not sent."
            );
        }

        Ok(())
    }

    fn child_pid(&self) -> Pid {
        self.child_pid.unwrap_or_else(|| Pid::from_raw(0))
    }
}
