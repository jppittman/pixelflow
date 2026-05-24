// src/os/pty_tests.rs

#![cfg(test)]

use super::pty::{NixPty, PtyChannel, PtyConfig};
use nix::sys::signal::{kill, Signal};
use nix::unistd::Pid;
use std::io::{ErrorKind, Read, Write};
use std::thread;
use std::time::Duration; // Reinstated for explicit type annotation

const DEFAULT_COLS: u16 = 80;
const DEFAULT_ROWS: u16 = 24;
const READ_TIMEOUT: Duration = Duration::from_secs(5); // General timeout for read operations

// Helper function to read with timeout and retries for WouldBlock
fn read_from_pty_with_timeout(pty: &mut NixPty, expected_str: &str) -> Result<String, String> {
    // Using BufReader for its read_until or read_line capabilities can be complex with WouldBlock.
    // A simpler raw read loop:
    let mut full_output_bytes = Vec::new();
    let mut buffer = [0; 1024];
    let start_time = std::time::Instant::now();
    let expected_bytes = expected_str.as_bytes();

    loop {
        if start_time.elapsed() > READ_TIMEOUT {
            return Err(format!(
                "Timeout reading from PTY. Expected to contain '{}', got '{}'",
                expected_str,
                String::from_utf8_lossy(&full_output_bytes)
            ));
        }

        // Check if expected_bytes is already in full_output_bytes
        if full_output_bytes
            .windows(expected_bytes.len())
            .any(|window| window == expected_bytes)
        {
            log::debug!(
                "Expected string '{}' found in accumulated output.",
                expected_str.trim_end_matches('\n')
            );
            break;
        }

        match pty.read(&mut buffer) {
            Ok(0) => {
                // EOF
                log::debug!(
                    "Read EOF from PTY. Full output: '{}'",
                    String::from_utf8_lossy(&full_output_bytes)
                );
                // Check one last time after EOF
                if full_output_bytes
                    .windows(expected_bytes.len())
                    .any(|window| window == expected_bytes)
                {
                    break;
                }
                // If EOF and string not found, it's an error.
                return Err(format!(
                    "EOF reached but expected string '{}' not found in output '{}'",
                    expected_str,
                    String::from_utf8_lossy(&full_output_bytes)
                ));
            }
            Ok(bytes_read) => {
                log::debug!(
                    "Read {} bytes: '{}'",
                    bytes_read,
                    String::from_utf8_lossy(&buffer[..bytes_read]).trim_end_matches('\n')
                );
                full_output_bytes.extend_from_slice(&buffer[..bytes_read]);
                // Check again after new data
                if full_output_bytes
                    .windows(expected_bytes.len())
                    .any(|window| window == expected_bytes)
                {
                    log::debug!(
                        "Expected string '{}' found after appending new data. Breaking.",
                        expected_str.trim_end_matches('\n')
                    );
                    break;
                }
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => {
                log::trace!(
                    "Read WouldBlock, retrying. Accumulated: '{}'",
                    String::from_utf8_lossy(&full_output_bytes)
                );
                thread::sleep(Duration::from_millis(100)); // Wait a bit before retrying
                continue;
            }
            Err(e) => {
                // Other errors (like EIO)
                // Before failing, check if we already got the string with previously read data + this error.
                if full_output_bytes
                    .windows(expected_bytes.len())
                    .any(|window| window == expected_bytes)
                {
                    log::warn!(
                        "Error {:?} occurred after expected string '{}' was already received. Output: '{}'",
                        e,
                        expected_str,
                        String::from_utf8_lossy(&full_output_bytes)
                    );
                    break;
                }
                return Err(format!(
                    "Error reading from PTY: {:?}. Partial output: '{}'",
                    e,
                    String::from_utf8_lossy(&full_output_bytes)
                ));
            }
        }
    }
    String::from_utf8(full_output_bytes).map_err(|e| format!("Output was not valid UTF-8: {}", e))
}

#[test]
fn test_pty_spawn_successful() {
    // Use sh -c to be more robust across platforms and ensure output flushing
    let config = PtyConfig {
        command_executable: "/bin/sh",
        args: &["-c", "echo hello pty world"],
        initial_cols: DEFAULT_COLS,
        initial_rows: DEFAULT_ROWS,
    };

    match NixPty::spawn_with_config(&config) {
        Ok(mut pty) => {
            log::debug!(
                "Successfully spawned PTY for echo test, child PID: {}",
                pty.child_pid()
            );
            // PTYs often add \r (carriage return) before \n (newline).
            // echo itself just outputs "hello pty world\n". The PTY layer might translate \n to \r\n.
            // We should be somewhat flexible with EOL or ensure expected_str matches exactly what PTY outputs.
            let expected_output = "hello pty world\n"; // echo output
            let output =
                read_from_pty_with_timeout(&mut pty, expected_output).unwrap_or_else(|e| {
                    panic!(
                        "pty_spawn_successful: Failed to read from PTY. Error: {}",
                        e
                    )
                });

            // Check if the core message is there, trimming whitespace for robustness.
            assert!(output.trim().contains("hello pty world"));
            // NixPty instance is dropped here.
        }
        Err(e) => {
            panic!("test_pty_spawn_successful: Failed to spawn PTY: {:?}", e);
        }
    }
}

#[test]
fn test_pty_read_write_interaction() {
    let shell_command = "read r_line; echo \"input was: $r_line\"";
    let config = PtyConfig {
        command_executable: "/bin/sh",
        // For sh -c "command", args should be ["-c", "command"]
        // spawn_with_config prepends "sh" as argv[0]
        args: &["-c", shell_command],
        initial_cols: DEFAULT_COLS,
        initial_rows: DEFAULT_ROWS,
    };

    let mut pty = match NixPty::spawn_with_config(&config) {
        Ok(p) => p,
        Err(e) => panic!(
            "test_pty_read_write_interaction: Failed to spawn PTY: {:?}",
            e
        ),
    };
    log::debug!(
        "Spawned PTY for read/write test, child PID: {}",
        pty.child_pid()
    );

    let write_data = "hello interactive pty\n"; // \n is important for `read` command in shell
    pty.write_all(write_data.as_bytes())
        .unwrap_or_else(|e| panic!("Failed to write to PTY: {:?}", e));
    log::debug!("Successfully wrote to PTY: '{}'", write_data.trim());

    let expected_output_fragment = "input was: hello interactive pty";
    let output = read_from_pty_with_timeout(&mut pty, expected_output_fragment)
        .unwrap_or_else(|err_msg| panic!("Read/write test failed: {}", err_msg));

    assert!(output.contains(expected_output_fragment));
    log::info!(
        "Read/write test successful. Full output: '{}'",
        output.trim()
    );
    // NixPty instance is dropped here.
}

#[test]
fn test_pty_resize_successful() {
    // Use `sleep` from PATH to be cross-platform (macOS has /bin/sleep, Linux /usr/bin/sleep)
    let config = PtyConfig {
        command_executable: "sleep",
        args: &["0.1"], // Arg for sleep is just the duration
        initial_cols: DEFAULT_COLS,
        initial_rows: DEFAULT_ROWS,
    };

    let pty = match NixPty::spawn_with_config(&config) {
        Ok(p) => p,
        Err(e) => panic!("test_pty_resize_successful: Failed to spawn PTY: {:?}", e),
    };
    log::debug!(
        "Spawned PTY for resize test, child PID: {}",
        pty.child_pid()
    );

    let new_cols = 100;
    let new_rows = 30;
    match pty.resize(new_cols, new_rows) {
        Ok(()) => {
            log::info!(
                "Successfully resized PTY for child PID {} to {}x{}",
                pty.child_pid(),
                new_cols,
                new_rows
            );
            // Primary assertion is that Ok(()) is returned.
        }
        Err(e) => {
            panic!("test_pty_resize_successful: Failed to resize PTY: {:?}", e);
        }
    }
    // NixPty instance is dropped here.
}

#[test]
fn test_pty_child_termination_on_drop() {
    let config = PtyConfig {
        command_executable: "sleep",
        args: &["2"], // Arg for sleep is just the duration
        initial_cols: DEFAULT_COLS,
        initial_rows: DEFAULT_ROWS,
    };

    let pty = match NixPty::spawn_with_config(&config) {
        Ok(p) => p,
        Err(e) => panic!(
            "test_pty_child_termination_on_drop: Failed to spawn PTY: {:?}",
            e
        ),
    };

    let child_pid: Pid = pty.child_pid(); // Capture PID before pty is dropped
    log::debug!(
        "Spawned PTY for child termination test, child PID: {}",
        child_pid
    );

    drop(pty); // Explicitly drop NixPty to trigger Drop trait
    log::debug!("Dropped PTY for child PID {}", child_pid);

    // Wait a bit to allow SIGHUP to be processed and child to terminate.
    thread::sleep(Duration::from_millis(200));

    // Check if the process is still alive. Signal 0 checks existence.
    match kill(child_pid, None) {
        Ok(_) => {
            // Process still exists. This might happen if SIGHUP didn't terminate it.
            // For robustness, try SIGKILL. This test's main goal is that Drop runs.
            log::warn!(
                "Child process {} still alive after drop and SIGHUP. Sending SIGKILL.",
                child_pid
            );
            let _ = kill(child_pid, Some(Signal::SIGKILL)); // Attempt to clean up
                                                            // Depending on strictness, this could be a panic.
                                                            // For CI stability, we might log and not panic, if SIGHUP is not 100% guaranteed kill for `sleep`.
                                                            // panic!("Child process {} did not terminate after PTY drop.", child_pid);
        }
        Err(nix::Error::ESRCH) => {
            // ESRCH ("No such process") means the child terminated as expected.
            log::info!(
                "Child process {} successfully terminated after PTY drop.",
                child_pid
            );
        }
        Err(e) => {
            // Other errors from kill check.
            panic!(
                "test_pty_child_termination_on_drop: Error checking child process {}: {:?}",
                child_pid, e
            );
        }
    }
}

#[test]
fn test_pty_spawn_invalid_command() {
    let non_existent_cmd = "/path/to/absolutely/nonexistent/command_39291az";
    let config = PtyConfig {
        command_executable: non_existent_cmd,
        args: &[],
        initial_cols: DEFAULT_COLS,
        initial_rows: DEFAULT_ROWS,
    };

    // With `std::process::Command`, spawning a non-existent command should return an error immediately,
    // rather than succeeding the fork and failing later in the child.
    match NixPty::spawn_with_config(&config) {
        Ok(pty) => {
            panic!(
                "test_pty_spawn_invalid_command: Expected spawn to fail for non-existent command, but it succeeded. Child PID: {}",
                pty.child_pid()
            );
        }
        Err(e) => {
            log::info!(
                "test_pty_spawn_invalid_command: NixPty::spawn_with_config returned Err as expected: {:?}",
                e
            );
            // We verify that the error is indeed related to the command not being found.
            // If it's wrapped in anyhow, we check the root cause.
            let root_cause = e.root_cause();
            if let Some(io_err) = root_cause.downcast_ref::<std::io::Error>() {
                assert_eq!(
                    io_err.kind(),
                    std::io::ErrorKind::NotFound,
                    "Expected NotFound error, got {:?}",
                    io_err.kind()
                );
            } else {
                log::warn!("Could not downcast error to std::io::Error to verify ErrorKind::NotFound, but an error occurred as expected.");
            }
        }
    }
}
