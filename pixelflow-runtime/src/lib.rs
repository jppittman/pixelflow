#![allow(clippy::large_enum_variant)]
#![allow(clippy::new_without_default)]
// Public API - organized under api module
pub mod api;

// Internal modules
pub mod config;
pub mod display;
pub mod engine_troupe;
pub mod error;
pub mod frame;
pub mod input;
pub mod pixel;
pub mod platform;
pub mod render_pool;
pub mod testing;
pub mod traits;
pub mod vsync_actor;

// Re-export public API types at crate root (new, preferred)
pub use api::public::*;
pub use error::RuntimeError;

// Re-export priority-channel as actor module
/// Actor model primitives (message passing, scheduling, priority lanes)
pub use actor_scheduler as actor;

// Convenience re-exports at crate root (for backward compatibility)
pub use actor_scheduler::{Actor, ActorHandle, ActorScheduler, Message, SendError, WakeHandler};

// Make private API available throughout crate (not exported)
#[allow(unused_imports)]
use api::private::*;

// Legacy re-exports (these should be moved to api::public or removed)
// TODO: Clean up - EngineActorHandle/Scheduler are used by legacy platform code
pub use config::{EngineConfig, PerformanceConfig, WindowConfig};
pub use display::messages::{DisplayControl, DisplayMgmt};

// DEPRECATED: // Export Troupe as EngineTroupe for backward compat
pub use engine_troupe::Troupe as EngineTroupe;
pub use frame::{create_frame_channel, create_recycle_channel, EngineHandle, FramePacket};

#[cfg(all(use_web_display, target_arch = "wasm32"))]
use wasm_bindgen::prelude::*;

// This code is dogshit and should be in the platform itself....
#[cfg(all(use_web_display, target_arch = "wasm32"))]
#[wasm_bindgen]
pub fn pixelflow_init_worker(
    canvas: web_sys::OffscreenCanvas,
    sab: js_sys::SharedArrayBuffer,
    scale_factor: f64,
) {
    crate::display::drivers::web::init_resources(canvas, sab, scale_factor);
}

// This code is dogshit and should be in the platform itself....
#[cfg(all(use_web_display, target_arch = "wasm32"))]
#[wasm_bindgen]
pub fn pixelflow_dispatch_event(
    sab: js_sys::SharedArrayBuffer,
    event_val: wasm_bindgen::JsValue,
) -> Result<(), wasm_bindgen::JsValue> {
    use crate::display::drivers::web::ipc::SharedRingBuffer;
    use crate::display::DisplayEvent;

    let event: DisplayEvent = serde_wasm_bindgen::from_value(event_val).map_err(|e| {
        wasm_bindgen::JsValue::from_str(&format!("Failed to deserialize event: {}", e))
    })?;

    let ipc = SharedRingBuffer::new(&sab);
    ipc.write(&event)
        .map_err(|e| wasm_bindgen::JsValue::from_str(&format!("Failed to write event: {}", e)))?;
    Ok(())
}
// /// * `app` - The application logic implementing `Application`.
// /// * `engine_handle` - Handle for app to send responses back to engine (renders, actions).
// /// * `config` - Engine configuration.
// pub fn run(
//     app: impl Application + Send + 'static,
//     engine_handle: api::private::EngineActorHandle,
//     scheduler: api::private::EngineActorScheduler,
//     config: EngineConfig,
// ) -> anyhow::Result<()> {
//     let platform = EnginePlatform::new(app, engine_handle, scheduler, config)?;
//     platform.run()
// }
