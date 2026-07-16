//! ServiceHandle: stable actor address that survives pod restarts.
//!
//! The analogy is a Kubernetes `Service` / ClusterIP: the address is stable, but
//! the backing pod (SPSC channel) can be torn down and replaced. Callers always
//! call the same `send()` method; the reconnect path is transparent.
//!
//! # Hot path (pod running)
//!
//! ```text
//! handle.send(msg)  →  ActorHandle::send()  →  SPSC ring buffer
//! ```
//!
//! Zero overhead: one `ActorHandle::send()` call, no atomic loads beyond the
//! SPSC itself. `ServiceHandle` is `!Clone` for the same reason `ActorHandle`
//! is — it owns a dedicated SPSC endpoint.
//!
//! # Cold path (pod restarting)
//!
//! ```text
//! handle.send(msg)
//!   → ActorHandle::send() → SendError::Disconnected   (msg consumed/dropped)
//!   → slot.reconnect(timeout)                          (blocks until pod Running)
//!   → self.connection = fresh ActorHandle              (update live endpoint)
//!   → Err(ServiceError::Reconnected)                   (caller retries if needed)
//! ```
//!
//! The message sent during the disconnect window is lost — same as a TCP segment
//! lost during a server restart. The caller decides whether to retry.

use std::sync::Arc;
use std::time::Duration;

use crate::registry::{PodGone, PodSlot};
use crate::{ActorHandle, Message, SendError};

/// Default timeout when waiting for a restarting pod to come back up.
///
/// Long enough to survive a slow restart but short enough to surface
/// runaway restart loops as errors rather than infinite hangs.
const DEFAULT_RECONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Why a `ServiceHandle::send()` call failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceError {
    /// The pod was restarting; the endpoint has been refreshed.
    ///
    /// The message sent during the restart window was lost. Retry the send
    /// if the message is critical; otherwise the next send will succeed.
    Reconnected,

    /// The pod is permanently stopped and cannot be restarted.
    PodGone,

    /// Send timed out waiting for the pod to restart, or backpressure timeout
    /// on the data lane / control+management backoff.
    Timeout,
}

impl std::fmt::Display for ServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServiceError::Reconnected => {
                write!(f, "pod restarted; message lost, endpoint refreshed")
            }
            ServiceError::PodGone => write!(f, "pod permanently stopped"),
            ServiceError::Timeout => write!(f, "send timed out"),
        }
    }
}

impl std::error::Error for ServiceError {}

impl From<PodGone> for ServiceError {
    fn from(e: PodGone) -> Self {
        match e {
            PodGone::Stopped => ServiceError::PodGone,
            PodGone::Timeout => ServiceError::Timeout,
        }
    }
}

/// Stable actor address that survives pod restarts.
///
/// `ServiceHandle` wraps an `ActorHandle` with a reconnect path. On hot sends the
/// cost is identical to a raw `ActorHandle::send()`. On disconnect it blocks in
/// `PodSlot::reconnect()` until the Kubelet publishes a fresh endpoint.
///
/// Not `Clone` — each `ServiceHandle` owns a dedicated SPSC endpoint (one slot in
/// the `ShardedInbox`). Use [`ActorBuilder::add_producer`] / the `troupe!` macro to
/// create independent handles for independent senders.
pub struct ServiceHandle<D, C, M> {
    /// Live SPSC endpoint. Replaced on reconnect.
    connection: ActorHandle<D, C, M>,
    /// Reconnect synchronisation point. Shared with the Kubelet.
    slot: Arc<PodSlot<D, C, M>>,
    /// How long to wait for the pod to come back before giving up.
    reconnect_timeout: Duration,
}

impl<D, C, M> ServiceHandle<D, C, M> {
    /// Wrap an existing `ActorHandle` in a `ServiceHandle`.
    ///
    /// `slot` must be the same slot registered with the Kubelet for this actor's
    /// pod. Typically created by the `troupe!` macro during bootstrap.
    #[must_use]
    pub fn new(connection: ActorHandle<D, C, M>, slot: Arc<PodSlot<D, C, M>>) -> Self {
        Self {
            connection,
            slot,
            reconnect_timeout: DEFAULT_RECONNECT_TIMEOUT,
        }
    }

    /// Override the default reconnect timeout.
    #[must_use]
    pub fn with_reconnect_timeout(mut self, timeout: Duration) -> Self {
        self.reconnect_timeout = timeout;
        self
    }

    /// Send a message to the actor.
    ///
    /// # Hot path (pod running)
    ///
    /// Delegates directly to `ActorHandle::send()`. No overhead beyond the SPSC send.
    ///
    /// # Cold path (pod restarting)
    ///
    /// On `SendError::Disconnected`, blocks until the pod restarts (or timeout),
    /// replaces the inner `ActorHandle`, and returns `Err(ServiceError::Reconnected)`.
    /// The original message was consumed by the failed send and cannot be recovered.
    /// The caller should retry the send if the message was critical.
    ///
    /// # Errors
    ///
    /// - `ServiceError::Reconnected` — pod was down, now back up; retry the send
    /// - `ServiceError::PodGone` — pod permanently stopped
    /// - `ServiceError::Timeout` — send timeout (backpressure or reconnect timeout)
    pub fn send<T: Into<Message<D, C, M>>>(&mut self, msg: T) -> Result<(), ServiceError> {
        match self.connection.send(msg) {
            Ok(()) => Ok(()),
            Err(SendError::Disconnected) => {
                // Block until the Kubelet publishes a fresh handle
                let fresh = self.slot.reconnect(self.reconnect_timeout)?;
                self.connection = fresh;
                // Original message lost during the restart window (same as TCP)
                Err(ServiceError::Reconnected)
            }
            Err(SendError::Timeout) => Err(ServiceError::Timeout),
        }
    }
}

impl<D, C, M> std::fmt::Debug for ServiceHandle<D, C, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServiceHandle")
            .field("reconnect_timeout", &self.reconnect_timeout)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Actor, ActorBuilder, ActorHandle, ActorScheduler, ActorStatus, HandlerError, HandlerResult,
        Message, SystemStatus,
    };
    use std::thread;
    use std::time::Duration;

    struct Noop;
    impl Actor<i32, i32, i32> for Noop {
        fn handle_data(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn handle_control(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _: i32) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    /// Spawn a pod with two handles: one for the ServiceHandle, one to send Shutdown.
    ///
    /// Dropping the JoinHandle does NOT stop the thread; you must call
    /// `kill.send(Message::Shutdown)` and then `join.join()` to cleanly
    /// terminate the pod and disconnect the SPSC channels.
    #[allow(clippy::type_complexity)]
    fn spawn_pod() -> (
        ActorHandle<i32, i32, i32>, // svc: for ServiceHandle
        ActorHandle<i32, i32, i32>, // kill: send Shutdown to stop pod
        thread::JoinHandle<()>,
    ) {
        let mut builder = ActorBuilder::<i32, i32, i32>::new(100, None);
        let svc_handle = builder.add_producer();
        let kill_handle = builder.add_producer();
        let mut scheduler = builder.build();
        let join = thread::spawn(move || {
            scheduler.run(&mut Noop);
        });
        (svc_handle, kill_handle, join)
    }

    fn kill_and_join(kill: ActorHandle<i32, i32, i32>, join: thread::JoinHandle<()>) {
        kill.send(Message::Shutdown).unwrap();
        join.join().unwrap();
    }

    #[test]
    fn hot_path_send_succeeds() {
        let (svc_handle, kill, pod) = spawn_pod();
        let slot = PodSlot::connected();
        let mut svc = ServiceHandle::new(svc_handle, slot);

        svc.send(Message::Data(42)).unwrap();
        svc.send(Message::Control(1)).unwrap();

        drop(svc);
        kill_and_join(kill, pod);
    }

    #[test]
    fn reconnects_after_pod_restart_and_returns_reconnected() {
        let (svc_handle, kill, pod) = spawn_pod();
        let slot = PodSlot::connected();
        let mut svc = ServiceHandle::new(svc_handle, slot.clone())
            .with_reconnect_timeout(Duration::from_secs(5));

        // Kill the pod cleanly and wait for it to fully exit
        kill_and_join(kill, pod);

        // Mark slot restarting — what the Kubelet does on pod exit
        slot.mark_restarting();

        // Spawn a fresh pod in the background; publish its handle after a short delay
        let slot_clone = slot.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(30));
            let (new_handle, mut new_scheduler) = ActorScheduler::<i32, i32, i32>::new(10, 100);
            slot_clone.publish(new_handle);
            new_scheduler.run(&mut Noop);
        });

        // send() hits Disconnected → blocks → reconnects → returns Reconnected
        let result = svc.send(Message::Data(99));
        assert_eq!(result, Err(ServiceError::Reconnected));

        // Endpoint is refreshed; next send succeeds
        svc.send(Message::Data(100)).unwrap();
    }

    #[test]
    fn returns_pod_gone_when_slot_stopped() {
        let (svc_handle, kill, pod) = spawn_pod();
        let slot = PodSlot::connected();
        let mut svc = ServiceHandle::new(svc_handle, slot.clone())
            .with_reconnect_timeout(Duration::from_secs(5));

        // Kill pod, then mark permanently stopped
        kill_and_join(kill, pod);
        slot.stop();

        let result = svc.send(Message::Data(1));
        assert_eq!(result, Err(ServiceError::PodGone));
    }

    #[test]
    fn returns_timeout_when_pod_does_not_restart_in_time() {
        let (svc_handle, kill, pod) = spawn_pod();
        let slot = PodSlot::connected();
        let mut svc = ServiceHandle::new(svc_handle, slot.clone())
            .with_reconnect_timeout(Duration::from_millis(50));

        kill_and_join(kill, pod);
        slot.mark_restarting(); // pod down, Kubelet never publishes

        let result = svc.send(Message::Data(1));
        assert_eq!(result, Err(ServiceError::Timeout));
    }
}
