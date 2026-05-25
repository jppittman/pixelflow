use crate::api::public::WindowDescriptor;
use crate::error::RuntimeError;
use crate::platform::macos::cocoa::{NSPoint, NSRect, NSSize, NSView, NSWindow};
use crate::platform::macos::sys::{self, Id, BOOL, YES};
use std::ffi::c_void;

pub struct MacWindow {
    pub(crate) window: NSWindow,
    pub(crate) view: NSView,
    pub(crate) layer: Id, // CAMetalLayer
    pub(crate) current_width: u32,
    pub(crate) current_height: u32,
}

impl MacWindow {
    pub fn new(desc: WindowDescriptor) -> Result<Self, RuntimeError> {
        let rect = NSRect::new(
            NSPoint::new(0.0, 0.0),
            NSSize::new(desc.width as f64, desc.height as f64),
        );

        // NSWindowStyleMask: Titled | Closable | Miniaturizable | Resizable
        let style_mask = 1 | 2 | 4 | 8;
        // NSBackingStoreBuffered = 2
        let backing = 2;

        let window = NSWindow::alloc().init_with_content_rect(rect, style_mask, backing, false);
        window.set_title(&desc.title);

        let view = NSView::alloc().init_with_frame(rect);

        // Create Metal Layer
        let layer = unsafe {
            let cls = sys::class(b"CAMetalLayer\0");
            sys::send(cls, sys::sel(b"layer\0"))
        };

        unsafe {
            // [layer setDevice: MTLCreateSystemDefaultDevice()]
            #[link(name = "Metal", kind = "framework")]
            extern "C" {
                fn MTLCreateSystemDefaultDevice() -> Id;
            }
            let device = MTLCreateSystemDefaultDevice();
            if device.is_null() {
                return Err(RuntimeError::MetalDeviceError);
            }
            sys::send_1::<(), Id>(layer, sys::sel(b"setDevice:\0"), device);

            // [layer setPixelFormat: 70 (RGBA8Unorm)]
            sys::send_1::<(), u64>(layer, sys::sel(b"setPixelFormat:\0"), 70);

            // [layer setFramebufferOnly: YES] - optimization
            sys::send_1::<(), BOOL>(layer, sys::sel(b"setFramebufferOnly:\0"), YES);

            // Enable display sync - hardware VSync handles frame pacing
            sys::send_1::<(), BOOL>(layer, sys::sel(b"setDisplaySyncEnabled:\0"), YES);

            // [view setLayer: layer]
            sys::send_1::<(), Id>(view.0, sys::sel(b"setLayer:\0"), layer);

            // [view setWantsLayer: YES]
            view.set_wants_layer(true);
        }

        window.set_content_view(view);
        window.make_key_and_order_front();

        // Center window?
        unsafe {
            sys::send::<()>(window.0, sys::sel(b"center\0"));
        }

        Ok(Self {
            window,
            view,
            layer,
            current_width: desc.width,
            current_height: desc.height,
        })
    }
}

impl MacWindow {
    pub fn set_title(&mut self, title: &str) {
        self.window.set_title(title);
    }

    pub fn set_size(&mut self, width: u32, height: u32) {
        let size = sys::CGSize {
            width: width as f64,
            height: height as f64,
        };
        unsafe {
            // Need to get current origin to keep it in place, or just set size?
            // "setContentSize:" is easier for content variance.
            sys::send_1::<(), sys::CGSize>(self.window.0, sys::sel(b"setContentSize:\0"), size);
        }
    }

    pub fn size(&self) -> (u32, u32) {
        (self.current_width, self.current_height)
    }

    pub fn scale_factor(&self) -> f64 {
        unsafe {
            let scale: f64 = sys::send(self.window.0, sys::sel(b"backingScaleFactor\0"));
            scale
        }
    }

    pub fn set_cursor(&mut self, icon: crate::api::public::CursorIcon) {
        use crate::api::public::CursorIcon;
        unsafe {
            let cursor_class = sys::class(b"NSCursor\0");
            let cursor: Id = match icon {
                CursorIcon::Default => sys::send(cursor_class, sys::sel(b"arrowCursor\0")),
                CursorIcon::Pointer => sys::send(cursor_class, sys::sel(b"pointingHandCursor\0")),
                CursorIcon::Text => sys::send(cursor_class, sys::sel(b"IBeamCursor\0")),
            };
            sys::send::<()>(cursor, sys::sel(b"set\0"));
        }
    }

    pub fn set_visible(&mut self, visible: bool) {
        if visible {
            self.window.make_key_and_order_front();
        } else {
            unsafe {
                sys::send::<()>(self.window.0, sys::sel(b"orderOut:\0"));
            }
        }
    }

    pub fn request_redraw(&mut self) {
        unsafe {
            sys::send_1::<(), BOOL>(self.view.0, sys::sel(b"setNeedsDisplay:\0"), YES);
        }
    }

    /// Present a frame to the window and return it for reuse.
    pub fn present(
        &mut self,
        frame: pixelflow_graphics::render::Frame<pixelflow_graphics::render::color::Rgba8>,
    ) -> pixelflow_graphics::render::Frame<pixelflow_graphics::render::color::Rgba8> {
        // Metal presentation logic.
        unsafe {
            // Ensure drawable size matches frame
            // setDrawableSize: takes CGSize (2x f64 = 16 bytes), not MTLSize (3x usize = 24 bytes)
            let size = sys::CGSize {
                width: frame.width as f64,
                height: frame.height as f64,
            };
            sys::send_1::<(), sys::CGSize>(self.layer, sys::sel(b"setDrawableSize:\0"), size);

            let drawable: Id = sys::send(self.layer, sys::sel(b"nextDrawable\0"));
            if !drawable.is_null() {
                let texture: Id = sys::send(drawable, sys::sel(b"texture\0"));

                let region =
                    sys::MTLRegion::new_2d(0, 0, frame.width as usize, frame.height as usize);
                let bytes = frame.as_bytes().as_ptr() as *const c_void;
                let bytes_per_row = (frame.width as usize) * 4;

                sys::send_4::<(), sys::MTLRegion, usize, *const c_void, usize>(
                    texture,
                    sys::sel(b"replaceRegion:mipmapLevel:withBytes:bytesPerRow:\0"),
                    region,
                    0,
                    bytes,
                    bytes_per_row,
                );

                sys::send::<()>(drawable, sys::sel(b"present\0"));
            }
        }
        // Return the frame for reuse
        frame
    }

    pub fn poll_resize(&mut self) -> Option<(u32, u32)> {
        unsafe {
            // View frame relative to window content rect
            // frame includes title bar, so we use contentView bounds for accuracy.
            // Check content view frame? frame includes title bar.
            // We want content size.
            let view: sys::Id = sys::send(self.window.0, sys::sel(b"contentView\0"));
            let bounds: sys::CGRect = sys::send(view, sys::sel(b"bounds\0"));

            let width = bounds.size.width as u32;
            let height = bounds.size.height as u32;

            if width != self.current_width || height != self.current_height {
                self.current_width = width;
                self.current_height = height;
                // Update drawable size immediately to avoid flickering
                self.set_size(width, height);
                return Some((width, height));
            }
        }
        None
    }
}

#[cfg(test)]
#[cfg(target_os = "macos")]
mod tests {
    use super::*;
    use crate::platform::macos::sys::{self, Id};

    #[test]
    #[ignore = "Requires macOS window server and Metal device"]
    fn test_window_creation() {
        let desc = WindowDescriptor {
            width: 800,
            height: 600,
            title: "Test Window".to_string(),
            ..Default::default()
        };

        let window = MacWindow::new(desc).expect("Failed to create window");

        // Verify window is not null
        assert!(!window.window.0.is_null());

        // Verify view is not null
        assert!(!window.view.0.is_null());

        // Verify initial size
        assert_eq!(window.current_width, 800);
        assert_eq!(window.current_height, 600);
    }

    #[test]
    #[ignore = "Requires macOS window server and Metal device"]
    fn test_metal_layer_config() {
        let desc = WindowDescriptor {
            width: 100,
            height: 100,
            title: "Layer Test".to_string(),
            ..Default::default()
        };
        let window = MacWindow::new(desc).expect("Failed to create window");

        unsafe {
            let layer = window.layer;
            assert!(!layer.is_null());

            // Check pixel format (70 = BGRA8Unorm or 80 = RGBA8Unorm?)
            // The code sets 70.
            let format: u64 = sys::send(layer, sys::sel(b"pixelFormat\0"));
            assert_eq!(
                format, 70,
                "Pixel format should be 70 (BGRA8Unorm_sRGB or similar)"
            );

            // Check device is attached
            let device: Id = sys::send(layer, sys::sel(b"device\0"));
            assert!(!device.is_null(), "Metal layer must have a device attached");

            // Check framebufferOnly is YES
            let fb_only: sys::BOOL = sys::send(layer, sys::sel(b"framebufferOnly\0"));
            assert_eq!(
                fb_only,
                sys::YES,
                "framebufferOnly should be YES for performance"
            );
        }
    }

    #[test]
    #[ignore = "Requires macOS window server and Metal device"]
    fn test_resize_state() {
        let desc = WindowDescriptor {
            width: 200,
            height: 200,
            title: "Resize Test".to_string(),
            ..Default::default()
        };
        let mut window = MacWindow::new(desc).expect("Failed to create window");

        // Simulate resize by calling set_size (which calls setContentSize:)
        window.set_size(400, 300);

        // Check internal state
        // Note: poll_resize relies on the actual window reporting a new size.
        // In a headless CI environment without a real window server, this might not reflect immediately
        // or might need a runloop spin.
        // However, we can check if our wrapper updated its tracker if we trust set_size to do so,
        // OR we trust poll_resize to return Some if the OS updated it.
        // Let's rely on poll_resize returning the new size if the OS processed the setContentSize.

        // spin runloop briefly? (Not easily possible without NSRunLoop access)
        // Instead, valid that we can set it.

        // We'll trust checking the window's content view frame directly via sys
        unsafe {
            let view: Id = sys::send(window.window.0, sys::sel(b"contentView\0"));
            let bounds: sys::CGRect = sys::send(view, sys::sel(b"bounds\0"));
            assert_eq!(bounds.size.width as u32, 400);
            assert_eq!(bounds.size.height as u32, 300);
        }
    }

    #[test]
    fn test_abi_alignment() {
        // Verify our assumption about ABI alignment for MTLSize vs CGSize
        // This is a static check of our struct definitions vs likely platform values
        use std::mem;

        // These sizes must match what the platform expects.
        // On 64-bit macOS:
        // CGFloat is double (8 bytes)
        assert_eq!(mem::size_of::<sys::CGSize>(), 16);
        assert_eq!(mem::align_of::<sys::CGSize>(), 8);

        // NSUInteger is u64 (8 bytes)
        // MTLSize is 3x NSUInteger
        assert_eq!(mem::size_of::<sys::MTLSize>(), 24);
        assert_eq!(mem::align_of::<sys::MTLSize>(), 8);

        // Ensure we are using the right types in window.rs
    }
}
