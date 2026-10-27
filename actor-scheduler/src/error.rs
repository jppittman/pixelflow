//! Error types for the actor scheduler library.

/// Error returned when sending to an actor fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendError {
    /// Timed out waiting for channel capacity
    Timeout,
    /// Channel disconnected - receiver dropped
    Disconnected,
}

impl std::fmt::Display for SendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SendError::Timeout => write!(f, "timeout sending message to actor"),
            SendError::Disconnected => write!(f, "actor channel disconnected"),
        }
    }
}

impl std::error::Error for SendError {}

impl<T> From<crate::spsc::TrySendError<T>> for SendError {
    fn from(err: crate::spsc::TrySendError<T>) -> Self {
        match err {
            crate::spsc::TrySendError::Full(_) => SendError::Timeout,
            crate::spsc::TrySendError::Disconnected(_) => SendError::Disconnected,
        }
    }
}

/// Error from an actor handler. Every handler failure panics — there is no recoverable
/// severity to select, because every build profile sets `panic = "abort"`, so a "recoverable"
/// error was never actually recoverable: nothing could unwind to a supervisor that might
/// restart it. This type still exists, as a plain wrapper, because `?` inside a handler is
/// worth keeping — the panic happens once, formatted, at the scheduler boundary, rather than
/// scattering `panic!` calls through every handler body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandlerError(String);

impl HandlerError {
    /// Wrap a failure message.
    pub fn new(msg: impl Into<String>) -> Self {
        HandlerError(msg.into())
    }

    /// Panic with this error's message. The one formatting site every `Err(e)` at a scheduler
    /// boundary funnels through.
    pub(crate) fn panic(self) -> ! {
        panic!("actor handler failed: {self}")
    }
}

impl std::fmt::Display for HandlerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for HandlerError {}

/// Result type for actor handlers.
pub type HandlerResult = Result<(), HandlerError>;

/// Result of draining messages from a channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainStatus {
    /// Channel empty, all messages processed
    Empty,
    /// Hit burst limit, more messages may be available
    More,
    /// Channel disconnected (senders dropped)
    Disconnected,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_says_the_send_timed_out_when_the_lane_stayed_full() {
        assert_eq!(
            SendError::Timeout.to_string(),
            "timeout sending message to actor"
        );
    }

    #[test]
    fn display_says_the_channel_disconnected_when_the_receiver_is_gone() {
        assert_eq!(
            SendError::Disconnected.to_string(),
            "actor channel disconnected"
        );
    }
}
