pub mod waker;

#[cfg(target_os = "macos")]
pub mod macos;

#[cfg(target_os = "linux")]
pub mod linux;

#[cfg(target_os = "macos")]
pub use macos::*;

#[cfg(target_os = "linux")]
pub use linux::*;

pub use pixelflow_graphics::render::color::PlatformPixel;

pub type PlatformDriver = crate::display::driver::DriverActor<ActivePlatform>;

// Re-export platform-specific types
#[cfg(target_os = "macos")]
pub type ActivePlatform = crate::display::platform::PlatformActor<macos::MetalOps>;

#[cfg(target_os = "linux")]
pub type ActivePlatform = crate::display::platform::PlatformActor<linux::LinuxOps>;
