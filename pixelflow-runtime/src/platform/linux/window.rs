//! X11 Window wrapper.
//!
//! Encapsulates X11 window creation, drawing, and state management.

use crate::api::public::{CursorIcon, WindowDescriptor};
use crate::error::RuntimeError;
use crate::platform::waker::X11Waker;
use log::info;
use pixelflow_graphics::render::color::Bgra8;
use pixelflow_graphics::render::Frame;
use std::ffi::{CStr, CString};
use std::mem;
use std::os::raw::{c_char, c_int};
use std::ptr;
use x11::xlib;

// Standard X11 cursor font constants (from X11/cursorfont.h)
const XC_ARROW: u32 = 2;
const XC_HAND2: u32 = 60;
const XC_XTERM: u32 = 152;

#[derive(Debug, Clone, Copy)]
pub struct SelectionAtoms {
    pub clipboard: xlib::Atom,
    pub targets: xlib::Atom,
    pub utf8_string: xlib::Atom,
    pub text: xlib::Atom,
    pub xa_string: xlib::Atom,
}

impl SelectionAtoms {
    fn new(display: *mut xlib::Display) -> Self {
        unsafe {
            let intern =
                |name: &[u8]| xlib::XInternAtom(display, name.as_ptr() as *const i8, xlib::False);
            Self {
                clipboard: intern(b"CLIPBOARD\0"),
                targets: intern(b"TARGETS\0"),
                utf8_string: intern(b"UTF8_STRING\0"),
                text: intern(b"TEXT\0"),
                xa_string: xlib::XA_STRING,
            }
        }
    }
}

pub struct X11Window {
    pub display: *mut xlib::Display,
    pub screen: c_int,
    pub window: xlib::Window,
    pub gc: xlib::GC,
    pub wm_delete_window: xlib::Atom,
    pub atoms: SelectionAtoms,
    pub wake_atom: xlib::Atom,
    pub xrm_db: Option<xlib::XrmDatabase>,

    // Window state
    pub width: u32,
    pub height: u32,
    pub scale_factor: f64,
    pub clipboard_data: String,
}

// SAFETY: X11 pointers are safe to share across threads if XInitThreads() is called.
// We call XInitThreads() in X11Window::new().
unsafe impl Send for X11Window {}

impl X11Window {
    pub fn new(settings: &WindowDescriptor, waker: &X11Waker) -> Result<Self, RuntimeError> {
        unsafe {
            if xlib::XInitThreads() == 0 {
                return Err(RuntimeError::XInitThreadsFailed);
            }

            let display = xlib::XOpenDisplay(ptr::null());
            if display.is_null() {
                return Err(RuntimeError::XOpenDisplayFailed);
            }

            let screen = xlib::XDefaultScreen(display);
            let root = xlib::XRootWindow(display, screen);

            // Intern Atoms
            let wm_delete_name = CString::new("WM_DELETE_WINDOW").unwrap();
            let wm_delete_window = xlib::XInternAtom(display, wm_delete_name.as_ptr(), xlib::False);
            let atoms = SelectionAtoms::new(display);

            let black = xlib::XBlackPixel(display, screen);
            let white = xlib::XWhitePixel(display, screen);

            let width = settings.width;
            let height = settings.height;

            let window =
                xlib::XCreateSimpleWindow(display, root, 0, 0, width, height, 0, white, black);

            if let Ok(c_title) = CString::new(settings.title.as_str()) {
                xlib::XStoreName(display, window, c_title.as_ptr());
            }

            // Initialize waker target
            waker.set_target(display, window);
            let wake_atom = waker.wake_atom().unwrap();

            // Select Input Events
            xlib::XSelectInput(
                display,
                window,
                xlib::KeyPressMask
                    | xlib::KeyReleaseMask
                    | xlib::ButtonPressMask
                    | xlib::ButtonReleaseMask
                    | xlib::PointerMotionMask
                    | xlib::StructureNotifyMask
                    | xlib::FocusChangeMask
                    | xlib::ExposureMask
                    | xlib::PropertyChangeMask,
            );

            // Set WM Protocols (for close button)
            xlib::XSetWMProtocols(display, window, &wm_delete_window as *const _ as *mut _, 1);

            // Create Graphics Context
            let gc = xlib::XCreateGC(display, window, 0, ptr::null_mut());
            xlib::XSetForeground(display, gc, white);
            xlib::XSetBackground(display, gc, black);

            // Map and Flush
            xlib::XMapWindow(display, window);
            xlib::XFlush(display);

            // XRM Database for DPI
            xlib::XrmInitialize();
            let resource_string = xlib::XResourceManagerString(display);
            let xrm_db = if resource_string.is_null() {
                None
            } else {
                Some(xlib::XrmGetStringDatabase(resource_string))
            };

            let mut win = Self {
                display,
                screen,
                window,
                gc,
                wm_delete_window,
                atoms,
                wake_atom,
                xrm_db,
                width,
                height,
                scale_factor: 1.0,
                clipboard_data: String::new(),
            };

            win.scale_factor = win.query_scale_factor();
            Ok(win)
        }
    }

    /// Query Xft.dpi from X resources.
    fn query_scale_factor(&self) -> f64 {
        let Some(xrm_db) = self.xrm_db else {
            return 1.0;
        };
        unsafe {
            let name = CString::new("Xft.dpi").unwrap();
            let class = CString::new("Xft.Dpi").unwrap();
            let mut type_return: *mut c_char = ptr::null_mut();
            let mut value_return: xlib::XrmValue = mem::zeroed();

            if xlib::XrmGetResource(
                xrm_db,
                name.as_ptr(),
                class.as_ptr(),
                &mut type_return,
                &mut value_return,
            ) == xlib::True
                && !value_return.addr.is_null()
            {
                if let Ok(dpi_str) = CStr::from_ptr(value_return.addr as *const c_char).to_str() {
                    if let Ok(dpi) = dpi_str.parse::<f64>() {
                        info!("X11: Xft.dpi = {}, scale = {:.2}", dpi, dpi / 96.0);
                        return dpi / 96.0;
                    }
                }
            }
            1.0
        }
    }

    pub fn present(&mut self, frame: Frame<Bgra8>) -> (Frame<Bgra8>, Result<(), RuntimeError>) {
        unsafe {
            let depth = xlib::XDefaultDepth(self.display, self.screen);
            let visual = xlib::XDefaultVisual(self.display, self.screen);
            let data_ptr = frame.data.as_ptr() as *mut i8;

            let image = xlib::XCreateImage(
                self.display,
                visual,
                depth as u32,
                xlib::ZPixmap,
                0,
                data_ptr,
                frame.width as u32,
                frame.height as u32,
                32,
                0,
            );

            if image.is_null() {
                return (frame, Err(RuntimeError::XCreateImageFailed));
            }

            xlib::XPutImage(
                self.display,
                self.window,
                self.gc,
                image,
                0,
                0,
                0,
                0,
                frame.width as u32,
                frame.height as u32,
            );

            // Detach data to prevent XDestroyImage from freeing Rust memory
            (*image).data = ptr::null_mut();
            xlib::XDestroyImage(image);
            xlib::XFlush(self.display);
        }
        (frame, Ok(()))
    }

    pub fn set_title(&self, title: &str) {
        unsafe {
            if let Ok(c_title) = CString::new(title) {
                xlib::XStoreName(self.display, self.window, c_title.as_ptr());
                xlib::XFlush(self.display);
            }
        }
    }

    pub fn set_size(&self, width: u32, height: u32) {
        unsafe {
            xlib::XResizeWindow(self.display, self.window, width, height);
            xlib::XFlush(self.display);
        }
    }

    pub fn set_cursor(&self, icon: CursorIcon) {
        unsafe {
            let shape = match icon {
                CursorIcon::Default => XC_ARROW,
                CursorIcon::Pointer => XC_HAND2,
                CursorIcon::Text => XC_XTERM,
            };
            let cursor = xlib::XCreateFontCursor(self.display, shape);
            xlib::XDefineCursor(self.display, self.window, cursor);
            xlib::XFreeCursor(self.display, cursor);
            xlib::XFlush(self.display);
        }
    }

    pub fn copy_to_clipboard(&mut self, text: &str) {
        self.clipboard_data = text.to_string();
        unsafe {
            xlib::XSetSelectionOwner(
                self.display,
                self.atoms.clipboard,
                self.window,
                xlib::CurrentTime,
            );
            xlib::XFlush(self.display);
        }
    }

    pub fn request_paste(&self) {
        unsafe {
            xlib::XConvertSelection(
                self.display,
                self.atoms.clipboard,
                self.atoms.utf8_string,
                self.atoms.clipboard,
                self.window,
                xlib::CurrentTime,
            );
            xlib::XFlush(self.display);
        }
    }

    pub fn show(&self) {
        unsafe {
            xlib::XMapRaised(self.display, self.window);
            xlib::XFlush(self.display);
        }
    }

    pub fn hide(&self) {
        unsafe {
            xlib::XUnmapWindow(self.display, self.window);
            xlib::XFlush(self.display);
        }
    }

    pub fn bell(&self) {
        unsafe {
            xlib::XBell(self.display, 0);
            xlib::XFlush(self.display);
        }
    }
}

impl Drop for X11Window {
    fn drop(&mut self) {
        // Clear waker first - prevents BadWindow errors from XSendEvent
        // after we destroy the window below.
        if let Some(w) = super::platform::SHARED_WAKER.get() {
            w.clear_target()
        }

        unsafe {
            if let Some(xrm_db) = self.xrm_db {
                xlib::XrmDestroyDatabase(xrm_db);
            }
            xlib::XFreeGC(self.display, self.gc);
            xlib::XDestroyWindow(self.display, self.window);
            xlib::XCloseDisplay(self.display);
        }
        info!("X11Window dropped - resources cleaned up");
    }
}
