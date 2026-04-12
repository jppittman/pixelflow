//! Pod slot: the reconnect primitive shared between a `ServiceHandle` and the Kubelet.
//!
//! Each `ServiceHandle<D,C,M>` owns an `Arc<PodSlot<D,C,M>>`. The Kubelet holds the
//! other `Arc` for every slot it manages. When a pod restarts, the Kubelet:
//!
//! 1. Calls `slot.publish(new_handle)` for each registered slot.
//! 2. Any `ServiceHandle` blocked in `reconnect()` wakes and takes its fresh handle.
//!
//! # One slot per ServiceHandle
//!
//! `ActorHandle` is intentionally non-`Clone` — it owns a dedicated SPSC endpoint.
//! Each `ServiceHandle` therefore has its own slot, exactly as each TCP client has
//! its own socket to the server.

use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::ActorHandle;

/// Why `reconnect()` failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PodGone {
    /// The pod is permanently stopped (`RestartPolicy::Never` and the pod exited,
    /// or the Kubelet called `stop()`).
    Stopped,
    /// Reconnect wait exceeded the caller-supplied timeout.
    Timeout,
}

impl std::fmt::Display for PodGone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PodGone::Stopped => write!(f, "pod permanently stopped"),
            PodGone::Timeout => write!(f, "reconnect timed out waiting for pod restart"),
        }
    }
}

impl std::error::Error for PodGone {}

/// Internal state of a pod slot.
enum SlotState<D, C, M> {
    /// `ServiceHandle` holds the live handle; no reconnect in progress.
    Connected,
    /// Pod exited. Kubelet has not yet published a replacement handle.
    Restarting,
    /// Kubelet published a fresh handle. Awaiting pickup by `ServiceHandle`.
    Ready(ActorHandle<D, C, M>),
    /// Pod is permanently stopped. All `reconnect()` calls return `PodGone::Stopped`.
    Stopped,
}

/// Reconnect synchronisation point shared between a `ServiceHandle` and the Kubelet.
///
/// The Kubelet populates the slot via [`publish`](PodSlot::publish) after restarting
/// the pod and creating a fresh `ActorHandle` for this slot. The `ServiceHandle`
/// calls [`reconnect`](PodSlot::reconnect) when it hits `SendError::Disconnected`.
pub struct PodSlot<D, C, M> {
    state: Mutex<SlotState<D, C, M>>,
    ready: Condvar,
}

impl<D, C, M> PodSlot<D, C, M> {
    /// Create a slot in the `Connected` state.
    ///
    /// Use `Connected` at bootstrap — the `ServiceHandle` already has a live handle
    /// from the initial `ActorBuilder::add_producer()` call.
    #[must_use]
    pub fn connected() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(SlotState::Connected),
            ready: Condvar::new(),
        })
    }

    /// Called by the Kubelet when the pod has died.
    ///
    /// Transitions `Connected → Restarting`. Subsequent `reconnect()` calls will
    /// block until `publish()` or `stop()` is called.
    pub fn mark_restarting(&self) {
        *self.state.lock().unwrap() = SlotState::Restarting;
    }

    /// Called by the Kubelet after creating a fresh `ActorHandle` for this slot.
    ///
    /// Transitions `Restarting → Ready(handle)` and wakes any blocked `reconnect()`.
    pub fn publish(&self, handle: ActorHandle<D, C, M>) {
        *self.state.lock().unwrap() = SlotState::Ready(handle);
        self.ready.notify_one();
    }

    /// Called by the Kubelet when the pod will never restart.
    ///
    /// Transitions to `Stopped` and unblocks all `reconnect()` waiters with
    /// `Err(PodGone::Stopped)`.
    pub fn stop(&self) {
        *self.state.lock().unwrap() = SlotState::Stopped;
        self.ready.notify_all();
    }

    /// Called by `ServiceHandle` on `SendError::Disconnected`.
    ///
    /// Blocks until the Kubelet calls `publish()` (or `stop()`), then returns the
    /// fresh `ActorHandle`. The slot transitions back to `Connected` on success.
    ///
    /// # Errors
    ///
    /// - `PodGone::Stopped` — pod is permanently stopped
    /// - `PodGone::Timeout` — `timeout` elapsed before the pod restarted
    pub fn reconnect(&self, timeout: Duration) -> Result<ActorHandle<D, C, M>, PodGone> {
        let mut state = self.state.lock().unwrap();
        let deadline = std::time::Instant::now() + timeout;

        loop {
            match &*state {
                SlotState::Ready(_) => {
                    // Take the handle, transition back to Connected
                    let SlotState::Ready(handle) =
                        std::mem::replace(&mut *state, SlotState::Connected)
                    else {
                        unreachable!()
                    };
                    return Ok(handle);
                }
                SlotState::Stopped => return Err(PodGone::Stopped),
                SlotState::Connected | SlotState::Restarting => {
                    let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                    if remaining.is_zero() {
                        return Err(PodGone::Timeout);
                    }
                    let (guard, timed_out) = self.ready.wait_timeout(state, remaining).unwrap();
                    state = guard;
                    if timed_out.timed_out() {
                        return Err(PodGone::Timeout);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn make_handle_and_scheduler() -> (
        ActorHandle<i32, i32, i32>,
        crate::ActorScheduler<i32, i32, i32>,
    ) {
        crate::ActorScheduler::new(10, 100)
    }

    #[test]
    fn publish_then_reconnect_succeeds() {
        let (handle, _scheduler) = make_handle_and_scheduler();
        let slot = PodSlot::<i32, i32, i32>::connected();

        slot.mark_restarting();

        // Publish a fresh handle (in a real system the Kubelet does this)
        let (new_handle, _new_scheduler) = make_handle_and_scheduler();
        slot.publish(new_handle);

        // reconnect() should return immediately
        let result = slot.reconnect(Duration::from_secs(1));
        assert!(result.is_ok(), "should have received the published handle");

        drop(handle);
    }

    #[test]
    fn reconnect_blocks_until_publish() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let slot = Arc::new(PodSlot::<i32, i32, i32>::connected());
        slot.mark_restarting();

        let received = Arc::new(AtomicBool::new(false));
        let slot_clone = slot.clone();
        let received_clone = received.clone();

        let waiter = thread::spawn(move || {
            let result = slot_clone.reconnect(Duration::from_secs(5));
            received_clone.store(true, Ordering::SeqCst);
            result.is_ok()
        });

        // Give waiter time to block
        thread::sleep(Duration::from_millis(20));
        assert!(!received.load(Ordering::SeqCst), "should still be waiting");

        let (new_handle, _) = make_handle_and_scheduler();
        slot.publish(new_handle);

        assert!(
            waiter.join().expect("Expected value but got None/Err"),
            "reconnect should succeed after publish"
        );
        assert!(received.load(Ordering::SeqCst));
    }

    #[test]
    fn reconnect_times_out_when_no_publish() {
        let slot = PodSlot::<i32, i32, i32>::connected();
        slot.mark_restarting();

        let result = slot.reconnect(Duration::from_millis(30));
        assert_eq!(result.unwrap_err(), PodGone::Timeout);
    }

    #[test]
    fn stop_unblocks_reconnect_with_stopped_error() {
        let slot = Arc::new(PodSlot::<i32, i32, i32>::connected());
        slot.mark_restarting();

        let slot_clone = slot.clone();
        let waiter = thread::spawn(move || slot_clone.reconnect(Duration::from_secs(5)));

        thread::sleep(Duration::from_millis(20));
        slot.stop();

        assert_eq!(waiter.join().expect("Expected value but got None/Err").unwrap_err(), PodGone::Stopped);
    }

    #[test]
    fn reconnect_on_stopped_slot_returns_immediately() {
        let slot = PodSlot::<i32, i32, i32>::connected();
        slot.stop();

        let start = std::time::Instant::now();
        let result = slot.reconnect(Duration::from_secs(5));
        assert!(
            start.elapsed() < Duration::from_millis(10),
            "should be immediate"
        );
        assert_eq!(result.unwrap_err(), PodGone::Stopped);
    }
}
