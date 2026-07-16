//! Zero-copy frame packet infrastructure for v11.0 architecture.
//!
//! This module provides the `FramePacket<T>` type for transferring surfaces
//! between the logic thread and render thread without copying pixel data.

use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;

/// A frame packet containing a composed surface and recycle channel.
///
/// The packet transfers ownership of the surface to the engine for rendering,
/// then returns via the recycle channel for reuse (zero-copy ping-pong).
///
/// The `recycle_tx` is wrapped in Arc - this is the "ghetto borrow" pattern
/// that works across thread boundaries. The Arc ensures we're explicit about
/// shared ownership of the return channel.
///
/// # Type Parameters
/// * `T` - The surface type (e.g., `TerminalSurface`)
pub struct FramePacket<T: Send> {
    /// The composed surface to render.
    pub surface: T,

    /// Channel for returning the packet after rendering (Arc-wrapped).
    pub recycle_tx: Arc<SyncSender<FramePacket<T>>>,
}

impl<T: Send> std::fmt::Debug for FramePacket<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FramePacket")
            .field("surface", &"<surface>")
            .field("recycle_tx", &"<channel>")
            .finish()
    }
}

impl<T: Send> FramePacket<T> {
    /// Creates a new frame packet with the given surface and recycle channel.
    pub fn new(surface: T, recycle_tx: Arc<SyncSender<FramePacket<T>>>) -> Self {
        Self {
            surface,
            recycle_tx,
        }
    }

    /// Recycles this packet back to the logic thread.
    ///
    /// Consumes self and sends it through the recycle channel.
    /// The Arc clone is just a refcount bump - no data copied.
    /// If the channel is closed (shutdown), the packet is simply dropped.
    pub fn recycle(self) {
        let tx = Arc::clone(&self.recycle_tx);
        tx.send(self).ok();
    }
}

/// Handle for submitting frames to the engine from the logic thread.
///
/// This is the "producer" side of the channel.
pub struct EngineHandle<T: Send> {
    submit_tx: SyncSender<FramePacket<T>>,
}

impl<T: Send> EngineHandle<T> {
    /// Creates a new engine handle with the given submit channel.
    #[must_use]
    pub fn new(submit_tx: SyncSender<FramePacket<T>>) -> Self {
        Self { submit_tx }
    }

    /// Submits a frame packet for rendering.
    ///
    /// This call may block if the channel buffer is full (back-pressure).
    pub fn submit_frame(&self, packet: FramePacket<T>) -> Result<(), FramePacket<T>> {
        self.submit_tx.send(packet).map_err(|e| e.0)
    }

    /// Tries to submit a frame packet without blocking.
    ///
    /// Returns the packet back if the channel is full.
    pub fn try_submit_frame(&self, packet: FramePacket<T>) -> Result<(), FramePacket<T>> {
        self.submit_tx.try_send(packet).map_err(|e| match e {
            std::sync::mpsc::TrySendError::Full(p) => p,
            std::sync::mpsc::TrySendError::Disconnected(p) => p,
        })
    }
}

impl<T: Send> Clone for EngineHandle<T> {
    fn clone(&self) -> Self {
        Self {
            submit_tx: self.submit_tx.clone(),
        }
    }
}

/// Creates a channel pair for frame submission.
///
/// Returns (handle, receiver) where:
/// - `handle` is used by the logic thread to submit frames
/// - `receiver` is used by the engine to receive frames
///
/// The channel has a buffer of 1 slot for ping-pong operation.
#[must_use]
pub fn create_frame_channel<T: Send>() -> (EngineHandle<T>, Receiver<FramePacket<T>>) {
    let (tx, rx) = sync_channel(1);
    (EngineHandle::new(tx), rx)
}

/// Creates a recycle channel for returning packets to the logic thread.
///
/// Returns (sender, receiver) where:
/// - `sender` is Arc-wrapped and cloned into each FramePacket
/// - `receiver` is held by the logic thread to get packets back
#[must_use]
pub fn create_recycle_channel<T: Send>(
) -> (Arc<SyncSender<FramePacket<T>>>, Receiver<FramePacket<T>>) {
    let (tx, rx) = sync_channel(2); // 2 slots for double-buffering
    (Arc::new(tx), rx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_core::Discrete;
    use pixelflow_core::Field;
    use pixelflow_core::Manifold;

    // A minimal test surface
    #[derive(Clone, Copy)]
    struct TestSurface {
        color: f32,
    }

    type Field4 = (Field, Field, Field, Field);

    impl Manifold<Field4> for TestSurface {
        type Output = Discrete;
        fn eval(&self, _p: Field4) -> Discrete {
            Discrete::pack(
                Field::from(self.color),
                Field::from(self.color),
                Field::from(self.color),
                Field::from(1.0),
            )
        }
    }

    #[test]
    fn test_create_channels() {
        let (_handle, _rx) = create_frame_channel::<TestSurface>();
        let (_recycle_tx, _recycle_rx) = create_recycle_channel::<TestSurface>();
    }

    #[test]
    fn test_submit_and_receive() {
        let (handle, rx) = create_frame_channel::<TestSurface>();
        let (recycle_tx, _recycle_rx) = create_recycle_channel::<TestSurface>();

        let surface = TestSurface { color: 1.0 };
        let packet = FramePacket::new(surface, recycle_tx);

        handle.submit_frame(packet).unwrap();

        let received = rx.recv().unwrap();
        assert_eq!(received.surface.color, 1.0);
    }

    #[test]
    fn test_recycle_loop() {
        let (handle, rx) = create_frame_channel::<TestSurface>();
        let (recycle_tx, recycle_rx) = create_recycle_channel::<TestSurface>();

        // Create and submit a packet
        let surface = TestSurface { color: 1.0 };
        let packet = FramePacket::new(surface, Arc::clone(&recycle_tx));
        handle.submit_frame(packet).unwrap();

        // Receive and "render" it
        let received = rx.recv().unwrap();
        assert_eq!(received.surface.color, 1.0);

        // Recycle using the method (clean API)
        received.recycle();

        // Logic thread receives the recycled packet
        let recycled = recycle_rx.recv().unwrap();
        assert_eq!(recycled.surface.color, 1.0);
    }
}
