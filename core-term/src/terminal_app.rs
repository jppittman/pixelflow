use crate::ansi::commands::AnsiCommand;
use crate::color::Color;
use crate::config::Config;
use crate::glyph::Glyph;
use crate::io::traits::PtySender;
use crate::io::PtyCommand;
use crate::messages::TerminalData;
use crate::term::TerminalEmulator;
use actor_scheduler::{
    Actor, ActorBuilder, ActorHandle, ActorStatus, HandlerError, HandlerResult, Message,
    SystemStatus,
};
use pixelflow_core::{And, At, Discrete, Ge, Le, Manifold, ManifoldExt, Select, Sub, W, X, Y, Z};

/// Adapter to send PTY commands to TerminalApp actor.
pub struct TerminalAppSender {
    handle: ActorHandle<TerminalData, EngineEventControl, EngineEventManagement>,
}

impl TerminalAppSender {
    pub fn new(
        handle: ActorHandle<TerminalData, EngineEventControl, EngineEventManagement>,
    ) -> Self {
        Self { handle }
    }
}

impl PtySender for TerminalAppSender {
    fn send(&self, cmds: Vec<AnsiCommand>) -> Result<(), anyhow::Error> {
        self.handle
            .send(Message::Data(TerminalData::Pty(cmds)))
            .map_err(|e| anyhow::anyhow!("Failed to send PTY data to app: {}", e))
    }
}

/// Helper to create a positioned terminal cell with background blending.
use pixelflow_graphics::fonts::loader::{LoadedFont, MmapSource};
use pixelflow_graphics::{CachedGlyph, GlyphCache, Positioned, SpatialBSP};
use pixelflow_runtime::api::private::EngineData;
use pixelflow_runtime::api::public::AppData;
use pixelflow_runtime::api::public::EngineHandle;
use pixelflow_runtime::platform::ColorCube;
use pixelflow_runtime::{EngineEventControl, EngineEventData, EngineEventManagement};
use std::sync::mpsc::SyncSender;
use std::sync::Arc;

/// Font filename (looked up in multiple locations)
const FONT_FILENAME: &str = "NotoSansMono-Regular.ttf";

/// Find the font file, trying multiple locations:
/// 1. macOS app bundle Resources directory (for bundled app)
/// 2. Workspace-relative path (for cargo run from workspace root)
/// 3. Crate-relative path (for tests)
fn find_font_path() -> std::path::PathBuf {
    // Try bundle Resources directory first (macOS app bundle)
    if let Ok(exe_path) = std::env::current_exe() {
        // exe is at CoreTerm.app/Contents/MacOS/CoreTerm
        // Resources is at CoreTerm.app/Contents/Resources/
        let bundle_font = exe_path
            .parent()
            .and_then(|macos_dir| macos_dir.parent())
            .map(|contents_dir| contents_dir.join("Resources").join(FONT_FILENAME));

        if let Some(path) = bundle_font {
            if path.exists() {
                log::info!("Using bundled font: {}", path.display());
                return path;
            }
        }
    }

    let workspace_path =
        std::path::PathBuf::from(format!("pixelflow-graphics/assets/{}", FONT_FILENAME));
    if workspace_path.exists() {
        log::info!("Using workspace font: {}", workspace_path.display());
        return workspace_path;
    }

    let crate_path =
        std::path::PathBuf::from(format!("../pixelflow-graphics/assets/{}", FONT_FILENAME));
    if crate_path.exists() {
        log::info!("Using crate-relative font: {}", crate_path.display());
        return crate_path;
    }

    // Return workspace path and let MmapSource::open fail with a good error
    workspace_path
}

/// Bounded glyph manifold (returns coverage in [0,1], 0 if out of bounds).
/// Select<Cond, CachedGlyph, f32>
type BoundedGlyph =
    Select<And<And<And<Ge<X, f32>, Le<X, f32>>, Ge<Y, f32>>, Le<Y, f32>>, CachedGlyph, f32>;

/// Positioned glyph manifold
type PositionedGlyph = At<Sub<X, f32>, Sub<Y, f32>, Z, W, BoundedGlyph>;

/// Layout parameters for a terminal cell.
#[derive(Clone, Copy)]
struct CellLayout {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

/// Color parameters for a terminal cell.
#[derive(Clone, Copy)]
struct CellColors {
    fg: [f32; 4],
    bg: [f32; 4],
}

/// Terminal application implementing Actor trait.
///
/// Receives engine events (frame requests, input) and responds with rendered
/// terminal content via the engine handle.
pub struct TerminalApp {
    pub emulator: TerminalEmulator,
    pty_tx: SyncSender<PtyCommand>,
    config: Config,
    engine_tx: EngineHandle,
    /// Memory-mapped font file.
    loaded_font: Arc<LoadedFont<MmapSource>>,
    /// Cached rasterized glyphs.
    glyph_cache: GlyphCache,
    /// Currently pressed mouse button, tracked for motion reporting.
    /// Set on MouseClick, cleared on MouseRelease.
    pressed_mouse_button: Option<pixelflow_runtime::input::MouseButton>,
}

/// Parameters for constructing a TerminalApp.
pub struct TerminalAppParams {
    /// Terminal emulator instance.
    pub emulator: TerminalEmulator,
    /// Channel to send commands to PTY (writes and resizes).
    pub pty_tx: SyncSender<PtyCommand>,
    /// Application configuration.
    pub config: Config,
    /// Unregistered engine handle (app will call register()).
    pub unregistered_engine: pixelflow_runtime::UnregisteredEngineHandle,
    /// Window configuration for registration.
    pub window_config: pixelflow_runtime::WindowConfig,
}

impl TerminalApp {
    /// Helper to create a positioned terminal cell with background blending.
    ///
    /// Composition: bg + cov * (fg - bg)
    #[inline(always)]
    fn make_terminal_cell(
        glyph: CachedGlyph,
        layout: CellLayout,
        colors: CellColors,
    ) -> impl Manifold<Output = Discrete> + Clone {
        // IMPORTANT: Bound BEFORE translating to avoid evaluating every glyph for every pixel
        let cond = X.ge(0.0) & X.le(layout.width) & Y.ge(0.0) & Y.le(layout.height);
        let bounded = Select {
            cond,
            if_true: glyph,
            if_false: 0.0f32,
        };

        let positioned = At {
            inner: bounded,
            x: X - layout.x,
            y: Y - layout.y,
            z: Z,
            w: W,
        };

        let lerp = X + Z * (Y - X);

        // Helper to blend a single channel
        let blend_channel = |bg: f32, fg: f32, coverage: &PositionedGlyph| At {
            inner: lerp,
            x: bg,
            y: fg,
            z: coverage.clone(),
            w: 0.0,
        };

        let r = blend_channel(colors.bg[0], colors.fg[0], &positioned);
        let g = blend_channel(colors.bg[1], colors.fg[1], &positioned);
        let b = blend_channel(colors.bg[2], colors.fg[2], &positioned);

        let blended = At {
            inner: ColorCube::default(),
            x: r,
            y: g,
            z: b,
            w: 1.0,
        };

        let in_bounds = X.ge(layout.x)
            & X.le(layout.x + layout.width)
            & Y.ge(layout.y)
            & Y.le(layout.y + layout.height);

        let transparent = At {
            inner: ColorCube::default(),
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        };

        Select {
            cond: in_bounds,
            if_true: blended,
            if_false: transparent,
        }
    }

    /// Creates a new terminal app (internal - use spawn_terminal_app instead).
    fn new_registered(params: TerminalAppParamsRegistered) -> Self {
        // Memory-map the font file from the appropriate location
        let font_path = find_font_path();
        let source = MmapSource::open(&font_path).unwrap_or_else(|e| {
            panic!("Failed to open font file at {}: {}", font_path.display(), e)
        });

        let loaded_font = Arc::new(LoadedFont::new(source).expect("Failed to parse font"));

        // Create glyph cache and pre-warm with ASCII
        let cell_height = params.config.appearance.cell_height_px as f32;
        let mut glyph_cache = GlyphCache::with_capacity(128);
        glyph_cache.warm_ascii(&loaded_font.font(), cell_height);

        Self {
            emulator: params.emulator,
            pty_tx: params.pty_tx,
            config: params.config,
            engine_tx: params.engine_tx,
            loaded_font,
            glyph_cache,
            pressed_mouse_button: None,
        }
    }

    /// Build a render manifold from the current terminal state.
    fn build_manifold(
        &mut self,
    ) -> (
        Arc<dyn Manifold<Output = Discrete> + Send + Sync>,
        (f32, f32),
    ) {
        // Get terminal snapshot
        let snapshot = match self.emulator.get_render_snapshot() {
            Some(s) => s,
            None => {
                let (r, g, b, a) = self.config.colors.background.to_f32_rgba();
                return (
                    Arc::new(At {
                        inner: ColorCube::default(),
                        x: r,
                        y: g,
                        z: b,
                        w: a,
                    }),
                    (0.0, 0.0),
                );
            }
        };

        let (cols, rows) = snapshot.dimensions;
        let cell_width = snapshot.cell_width_px as f32;
        let cell_height = snapshot.cell_height_px as f32;
        let grid_width = cols as f32 * cell_width;
        let grid_height = rows as f32 * cell_height;

        // Debug: Log dimensions once per build
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            log::info!(
                "Terminal snapshot: {}x{} cells, cell size {}x{} px, grid {}x{} px",
                cols,
                rows,
                cell_width,
                cell_height,
                grid_width,
                grid_height
            );
        }

        // Default colors
        let default_fg = self.config.colors.foreground;
        let default_bg = self.config.colors.background;

        // Build 2-level BSP: Vertical (Rows) -> Horizontal (Cells)
        let mut row_items = Vec::new();

        for row in 0..rows {
            let line = &snapshot.lines[row];
            let mut cell_items = Vec::new();

            for col in 0..cols {
                let glyph = &line.cells[col];

                let (ch, fg_color, cell_bg) = match glyph {
                    Glyph::Single(cc) | Glyph::WidePrimary(cc) => {
                        let fg = if cc.attr.fg == Color::Default {
                            default_fg
                        } else {
                            cc.attr.fg
                        };
                        let bg = if cc.attr.bg == Color::Default {
                            default_bg
                        } else {
                            cc.attr.bg
                        };
                        (cc.c, fg, bg)
                    }
                    Glyph::WideSpacer => continue, // Skip spacers
                };

                // Get cached glyph - glyph_scaled now accounts for descenders
                if let Some(cached) =
                    self.glyph_cache
                        .get(&self.loaded_font.font(), ch, cell_height)
                {
                    let (fg_r, fg_g, fg_b, fg_a) = fg_color.to_f32_rgba();
                    let (bg_r, bg_g, bg_b, bg_a) = cell_bg.to_f32_rgba();

                    let x = col as f32 * cell_width;
                    let y = row as f32 * cell_height;

                    cell_items.push(Positioned {
                        bounds: (x, y, x + cell_width, y + cell_height),
                        leaf: Self::make_terminal_cell(
                            cached,
                            CellLayout {
                                x,
                                y,
                                width: cell_width,
                                height: cell_height,
                            },
                            CellColors {
                                fg: [fg_r, fg_g, fg_b, fg_a],
                                bg: [bg_r, bg_g, bg_b, bg_a],
                            },
                        ),
                    });
                }
            }

            // If row has cells, wrap them in a horizontal BSP and add to row list
            if !cell_items.is_empty() {
                let y_min = row as f32 * cell_height;
                let y_max = y_min + cell_height;

                row_items.push(Positioned {
                    bounds: (0.0, y_min, grid_width, y_max),
                    leaf: SpatialBSP::from_positioned(cell_items),
                });
            }
        }

        // If no rows have content, just return background
        if row_items.is_empty() {
            let (r, g, b, a) = default_bg.to_f32_rgba();
            return (
                Arc::new(At {
                    inner: ColorCube::default(),
                    x: r,
                    y: g,
                    z: b,
                    w: a,
                }),
                (grid_width, grid_height),
            );
        }

        // Build top-level vertical BSP from row items
        log::debug!(
            "Building BSP with {} row_items (from {} rows), grid {}x{}",
            row_items.len(),
            rows,
            grid_width,
            grid_height
        );
        let top_bsp = SpatialBSP::from_positioned(row_items);
        (Arc::new(top_bsp), (grid_width, grid_height))
    }

    /// Send a rendered frame to the engine.
    fn send_frame(&mut self) {
        let (manifold, grid_bounds) = self.build_manifold();

        // 1. Create default background manifold
        let default_bg = self.config.colors.background;
        let (r, g, b, a) = default_bg.to_f32_rgba();
        let background = At {
            inner: ColorCube::default(),
            x: r,
            y: g,
            z: b,
            w: a,
        };

        // 2. Wrap SpatialBSP in a Select that clips to grid bounds
        // cond = (x >= 0) & (x < grid_width) & (y >= 0) & (y < grid_height)
        let (gw, gh) = grid_bounds;
        let cond = X.ge(0.0) & X.lt(gw) & Y.ge(0.0) & Y.lt(gh);

        let scene = Select {
            cond,
            if_true: manifold,
            if_false: background,
        };

        let data = AppData::RenderSurface(Arc::new(scene));
        if let Err(e) = self
            .engine_tx
            .send(Message::Data(EngineData::FromApp(data)))
        {
            log::warn!("Failed to send frame to engine: {}", e);
        }
    }
}

impl Actor<TerminalData, EngineEventControl, EngineEventManagement> for TerminalApp {
    fn handle_data(&mut self, data: TerminalData) -> HandlerResult {
        match data {
            TerminalData::Engine(EngineEventData::RequestFrame { .. }) => {
                // Engine is requesting a frame - build and send it
                self.send_frame();
            }
            TerminalData::Pty(commands) => {
                use crate::term::EmulatorInput;
                // Process incoming ANSI commands
                for cmd in commands {
                    self.emulator.interpret_input(EmulatorInput::Ansi(cmd));
                }
                // We don't necessarily send a frame here anymore, relying on VSync (RequestFrame)
                // or we could trigger a redraw if we want immediate feedback (but risk flooding)
                // For now, let's just update state. The next RequestFrame will pick it up.
            }
        }
        Ok(())
    }

    fn handle_control(&mut self, ctrl: EngineEventControl) -> HandlerResult {
        match ctrl {
            EngineEventControl::WindowCreated {
                id,
                width_px,
                height_px,
                scale,
            } => {
                log::info!(
                    "[TERM] Window created: id={}, {}x{} pixels, scale={}",
                    id.0,
                    width_px,
                    height_px,
                    scale
                );

                // Window is now ready - send initial frame to start VSync loop
                self.send_frame();
            }
            EngineEventControl::Resized {
                id: _,
                width_px,
                height_px,
            } => {
                use crate::term::{ControlEvent, EmulatorAction, EmulatorInput};
                // Convert u32 pixels to u16 for ControlEvent
                // Saturate at u16::MAX to prevent overflow panics
                let width_u16 = width_px.min(u16::MAX as u32) as u16;
                let height_u16 = height_px.min(u16::MAX as u32) as u16;

                let input = EmulatorInput::Control(ControlEvent::Resize {
                    width_px: width_u16,
                    height_px: height_u16,
                });

                // Process the resize and handle the resulting action
                if let Some(action) = self.emulator.interpret_input(input) {
                    if let EmulatorAction::ResizePty { cols, rows } = action {
                        // Send resize command to PTY write thread
                        if let Err(e) = self
                            .pty_tx
                            .send(PtyCommand::Resize(crate::io::Resize { cols, rows }))
                        {
                            log::warn!("Failed to send PTY resize command: {}", e);
                        }
                    }
                }

                // Request a redraw after resize
                self.send_frame();
            }
            EngineEventControl::CloseRequested => {
                unimplemented!("CloseRequested handler - need to cleanup and shutdown");
            }
            EngineEventControl::ScaleChanged { id, scale } => {
                unimplemented!(
                    "ScaleChanged: id={}, scale={} - need to adjust font sizes and redraw",
                    id.0,
                    scale
                );
            }
        }
        Ok(())
    }

    fn handle_management(&mut self, mgmt: EngineEventManagement) -> HandlerResult {
        match mgmt {
            EngineEventManagement::KeyDown { key, mods, text } => {
                use crate::term::{EmulatorAction, EmulatorInput, UserInputAction};

                let input = EmulatorInput::User(UserInputAction::KeyInput {
                    symbol: key,
                    modifiers: mods,
                    text: text.map(|s| std::borrow::Cow::Owned(s)),
                });

                if let Some(action) = self.emulator.interpret_input(input) {
                    match action {
                        EmulatorAction::WritePty(bytes) => {
                            if let Err(e) = self.pty_tx.send(PtyCommand::Write(bytes)) {
                                log::warn!("Failed to send input to PTY: {}", e);
                            }
                        }
                        EmulatorAction::Quit => {
                            // Handle quit - send quit to engine
                            use pixelflow_runtime::api::public::AppManagement;
                            self.engine_tx
                                .send(Message::Management(AppManagement::Quit))
                                .expect("Failed to send Quit to engine");
                        }
                        EmulatorAction::SetTitle(_title) => {
                            unimplemented!("EmulatorAction::SetTitle");
                        }
                        EmulatorAction::RingBell => {
                            unimplemented!("EmulatorAction::RingBell");
                        }
                        EmulatorAction::RequestRedraw => {
                            self.send_frame();
                        }
                        EmulatorAction::SetCursorVisibility(_visible) => {
                            unimplemented!("EmulatorAction::SetCursorVisibility");
                        }
                        EmulatorAction::CopyToClipboard(_text) => {
                            unimplemented!("EmulatorAction::CopyToClipboard");
                        }
                        EmulatorAction::RequestClipboardContent => {
                            unimplemented!("EmulatorAction::RequestClipboardContent");
                        }
                        EmulatorAction::ResizePty { cols, rows } => {
                            // Send resize command to PTY write thread
                            if let Err(e) = self
                                .pty_tx
                                .send(PtyCommand::Resize(crate::io::Resize { cols, rows }))
                            {
                                log::warn!("Failed to send PTY resize command: {}", e);
                            }
                        }
                    }
                }
            }
            EngineEventManagement::MouseClick { button, x, y } => {

                let col = (x / self.config.appearance.cell_width_px as u32) as usize;
                let row = (y / self.config.appearance.cell_height_px as u32) as usize;
                log::trace!(
                    "Mouse click: button={:?} at cell ({}, {})",
                    button,
                    col,
                    row
                );
                self.pressed_mouse_button = Some(button);
                if let Some(bytes) =
                    self.emulator
                        .encode_mouse_event(crate::term::MouseEncodingParams { button, col, row, kind: crate::term::MouseEventKind::Press })
                {
                    if let Err(e) = self.pty_tx.send(PtyCommand::Write(bytes)) {
                        log::warn!("Failed to send mouse press to PTY: {}", e);
                    }
                }
            }
            EngineEventManagement::MouseRelease { button, x, y } => {

                let col = (x / self.config.appearance.cell_width_px as u32) as usize;
                let row = (y / self.config.appearance.cell_height_px as u32) as usize;
                log::trace!(
                    "Mouse release: button={:?} at cell ({}, {})",
                    button,
                    col,
                    row
                );
                self.pressed_mouse_button = None;
                if let Some(bytes) =
                    self.emulator
                        .encode_mouse_event(crate::term::MouseEncodingParams { button, col, row, kind: crate::term::MouseEventKind::Release })
                {
                    if let Err(e) = self.pty_tx.send(PtyCommand::Write(bytes)) {
                        log::warn!("Failed to send mouse release to PTY: {}", e);
                    }
                }
            }
            EngineEventManagement::MouseMove { x, y, mods: _ } => {

                let col = (x / self.config.appearance.cell_width_px as u32) as usize;
                let row = (y / self.config.appearance.cell_height_px as u32) as usize;
                log::trace!("Mouse move: cell ({}, {})", col, row);
                // any-event mode (1003) reports all motion;
                // button-event mode (1002) only reports motion while a button is held
                if self.emulator.reports_all_motion() {
                    let button = self
                        .pressed_mouse_button
                        .unwrap_or(pixelflow_runtime::input::MouseButton::Left);
                    if let Some(bytes) =
                        self.emulator
                            .encode_mouse_event(crate::term::MouseEncodingParams { button, col, row, kind: crate::term::MouseEventKind::Motion })
                    {
                        if let Err(e) = self.pty_tx.send(PtyCommand::Write(bytes)) {
                            log::warn!("Failed to send mouse motion to PTY: {}", e);
                        }
                    }
                } else if self.emulator.reports_button_motion() {
                    // button-event mode: only report when a button is held
                    if let Some(button) = self.pressed_mouse_button {
                        if let Some(bytes) = self.emulator.encode_mouse_event(
                            crate::term::MouseEncodingParams { button, col, row, kind: crate::term::MouseEventKind::Motion }
                        ) {
                            if let Err(e) = self.pty_tx.send(PtyCommand::Write(bytes)) {
                                log::warn!("Failed to send mouse motion to PTY: {}", e);
                            }
                        }
                    }
                }
            }
            EngineEventManagement::MouseScroll {
                x,
                y,
                dx: _,
                dy,
                mods: _,
            } => {
                log::trace!("Mouse scroll: delta dy={}", dy);
                // When mouse tracking is active, report scroll as button press events
                if self.emulator.is_mouse_tracking_active() && dy != 0.0 {

                    use pixelflow_runtime::input::MouseButton;
                    let col = (x / self.config.appearance.cell_width_px as u32) as usize;
                    let row = (y / self.config.appearance.cell_height_px as u32) as usize;
                    let button = if dy < 0.0 {
                        MouseButton::ScrollUp
                    } else {
                        MouseButton::ScrollDown
                    };
                    if let Some(bytes) =
                        self.emulator
                            .encode_mouse_event(crate::term::MouseEncodingParams { button, col, row, kind: crate::term::MouseEventKind::Press })
                    {
                        if let Err(e) = self.pty_tx.send(PtyCommand::Write(bytes)) {
                            log::warn!("Failed to send mouse scroll to PTY: {}", e);
                        }
                    }
                } else {
                    // Scrollback navigation: negative dy scrolls up (into history),
                    // positive dy scrolls down (toward live screen)
                    // Scale by 3 lines per scroll unit for better UX
                    let scroll_lines = -(dy as i32) * 3;
                    if self.emulator.scroll_viewport(scroll_lines) {
                        // Viewport changed, send frame immediately for responsive scrolling
                        self.send_frame();
                    }
                }
            }
            EngineEventManagement::FocusGained => {
                log::trace!("Focus gained");
                // Some applications care about focus for bracketed paste mode
                // Could send \x1b[I if bracketed paste is enabled
            }
            EngineEventManagement::FocusLost => {
                log::trace!("Focus lost");
                // Some applications care about focus for bracketed paste mode
                // Could send \x1b[O if bracketed paste is enabled
            }
            EngineEventManagement::Paste(text) => {
                log::trace!("Paste: {} bytes", text.len());
                // Send pasted text to PTY
                self.pty_tx
                    .send(PtyCommand::Write(text.into_bytes()))
                    .expect("Failed to send paste to PTY");
            }
        }
        Ok(())
    }

    fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        // No polling needed - PTY data comes in via handle_data
        Ok(ActorStatus::Idle)
    }
}

/// Creates terminal app and spawns it in a thread.
///
/// This function handles registration atomically:
/// 1. Creates the app actor's channel
/// 2. Registers the app with the engine (sends RegisterApp + CreateWindow)
/// 3. Spawns the app thread with the registered engine handle
pub fn spawn_terminal_app(
    params: TerminalAppParams,
) -> std::io::Result<(
    actor_scheduler::ActorHandle<TerminalData, EngineEventControl, EngineEventManagement>,
    actor_scheduler::ActorHandle<TerminalData, EngineEventControl, EngineEventManagement>,
    std::thread::JoinHandle<()>,
)> {
    // Create app actor's channels using ActorBuilder (SPSC - each producer is unique)
    // ActorHandle is not Clone; each consumer needs its own dedicated handle.
    let mut builder =
        ActorBuilder::<TerminalData, EngineEventControl, EngineEventManagement>::new(128, None);
    let app_handle = builder.add_producer(); // For the caller (returns to main, keep-alive)
    let pty_handle = builder.add_producer(); // For the PTY sender (TerminalAppSender)
    let adapter_handle = builder.add_producer(); // For TerminalAppAdapter (engine→app)
    let mut app_rx = builder.build_with_burst(10, actor_scheduler::ShutdownMode::default());

    // Register with engine (sends RegisterApp + CreateWindow atomically)
    use pixelflow_runtime::api::public::{Application, EngineEvent};
    use pixelflow_runtime::WindowDescriptor;

    struct TerminalAppAdapter {
        // Mutex satisfies Sync for Arc<dyn Application + Send + Sync>.
        // No contention — only the engine actor thread calls send().
        handle: std::sync::Mutex<
            actor_scheduler::ActorHandle<TerminalData, EngineEventControl, EngineEventManagement>,
        >,
    }

    impl Application for TerminalAppAdapter {
        fn send(&self, event: EngineEvent) -> Result<(), pixelflow_runtime::error::RuntimeError> {
            let msg = match event {
                EngineEvent::Data(d) => Message::Data(TerminalData::Engine(d)),
                EngineEvent::Control(c) => Message::Control(c),
                EngineEvent::Management(m) => Message::Management(m),
            };
            self.handle
                .lock()
                .unwrap()
                .send(msg)
                .map_err(|e| pixelflow_runtime::error::RuntimeError::EventSendError(e.to_string()))
        }
    }

    let window_descriptor = WindowDescriptor {
        width: params.window_config.width,
        height: params.window_config.height,
        title: params.window_config.title.clone(),
        resizable: true,
    };

    let app_arc = std::sync::Arc::new(TerminalAppAdapter {
        handle: std::sync::Mutex::new(adapter_handle),
    });
    let engine_tx = params
        .unregistered_engine
        .register(app_arc, window_descriptor)
        .expect("Failed to register app with engine");

    log::info!("[TERM] App registered with engine, window creation requested");

    // Create app with registered engine handle
    let app_params_registered = TerminalAppParamsRegistered {
        emulator: params.emulator,
        pty_tx: params.pty_tx,
        config: params.config,
        engine_tx,
    };

    let mut app = TerminalApp::new_registered(app_params_registered);

    // Spawn app thread
    let handle = std::thread::Builder::new()
        .name("terminal-app".to_string())
        .spawn(move || {
            app_rx.run(&mut app);
        })?;

    Ok((app_handle, pty_handle, handle))
}

/// Parameters after registration (internal use).
struct TerminalAppParamsRegistered {
    emulator: TerminalEmulator,
    pty_tx: SyncSender<PtyCommand>,
    config: Config,
    engine_tx: EngineHandle,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::PtyCommand;
    use crate::term::TerminalEmulator;
    use actor_scheduler::Actor;
    use pixelflow_runtime::input::{KeySymbol, Modifiers};
    use pixelflow_runtime::{EngineEventControl, EngineEventManagement, WindowId};
    use std::sync::mpsc::Receiver;

    // Define a DummyPixel struct for testing
    #[allow(dead_code)]
    #[derive(Debug, Clone, Copy, Default, PartialEq)]
    struct DummyPixel;
    impl pixelflow_graphics::render::Pixel for DummyPixel {
        fn from_u32(_: u32) -> Self {
            Self
        }
        fn to_u32(self) -> u32 {
            0
        }
        fn from_rgba(_r: f32, _g: f32, _b: f32, _a: f32) -> Self {
            Self
        }
    }

    // Helper to create a test instance
    // Returns scheduler to keep doorbell channel alive during test
    // Returns None if font is missing/invalid (e.g. LFS pointer), skipping the test.
    fn create_test_app() -> Option<(
        TerminalApp,
        Receiver<PtyCommand>,
        pixelflow_runtime::api::private::EngineActorHandle,
        pixelflow_runtime::api::private::EngineActorScheduler,
    )> {
        // Check font availability to avoid panic if LFS not present
        let font_path = find_font_path();
        if !font_path.exists() {
            eprintln!(
                "Test skipped: Font file not found at {}",
                font_path.display()
            );
            return None;
        }
        if let Ok(metadata) = std::fs::metadata(&font_path) {
            if metadata.len() < 1000 {
                eprintln!(
                    "Test skipped: Font file at {} appears to be an LFS pointer (size < 1000 bytes)",
                    font_path.display()
                );
                return None;
            }
        }

        let emulator = TerminalEmulator::new(80, 24);
        let (pty_tx, pty_rx) = std::sync::mpsc::sync_channel(128);

        // Create engine handles with ActorBuilder (SPSC - each producer is unique)
        let mut engine_builder = actor_scheduler::ActorBuilder::<
            pixelflow_runtime::api::private::EngineData,
            pixelflow_runtime::api::private::EngineControl,
            pixelflow_runtime::api::public::AppManagement,
        >::new(10, None);
        let engine_tx = engine_builder.add_producer(); // For test inspection
        let engine_tx_for_test = engine_builder.add_producer(); // For EngineHandle
        let engine_scheduler =
            engine_builder.build_with_burst(10, actor_scheduler::ShutdownMode::default());

        let config = Config::default();
        let params = TerminalAppParamsRegistered {
            emulator,
            pty_tx,
            config,
            engine_tx: EngineHandle::new_for_test(engine_tx_for_test),
        };
        let app = TerminalApp::new_registered(params);

        Some((app, pty_rx, engine_tx, engine_scheduler))
    }

    #[test]
    fn test_handle_control_resize() {
        let (mut app, pty_rx, _, _scheduler) = match create_test_app() {
            Some(v) => v,
            None => return,
        };

        // Initial size is 80x24
        let snapshot_initial = app.emulator.get_render_snapshot().expect("Snapshot");
        assert_eq!(snapshot_initial.dimensions, (80, 24));

        // Send resize event
        // Default config: cell width 10, height 16.
        // Resize to 1000x800 -> 100x50 cells.
        let resize_event = EngineEventControl::Resized {
            id: WindowId(0),
            width_px: 1000,
            height_px: 800,
        };
        let _ = app.handle_control(resize_event);

        // Verify resize via snapshot
        let snapshot_new = app.emulator.get_render_snapshot().expect("Snapshot");
        assert_eq!(
            snapshot_new.dimensions,
            (100, 50),
            "Emulator should have resized to 100x50"
        );

        // Verify PTY resize command was sent
        let cmd = pty_rx.try_recv().expect("Should receive resize command");
        assert_eq!(
            cmd,
            PtyCommand::Resize(crate::io::Resize {
                cols: 100,
                rows: 50
            }),
            "PTY resize command should match new dimensions"
        );
    }

    #[test]
    fn test_handle_management_keydown() {
        let (mut app, pty_rx, _, _scheduler) = match create_test_app() {
            Some(v) => v,
            None => return,
        };

        // Simulate KeyDown
        let key_event = EngineEventManagement::KeyDown {
            key: KeySymbol::Char('a'),
            mods: Modifiers::empty(),
            text: Some("a".to_string()),
        };

        let _ = app.handle_management(key_event);

        // We expect 'a' to be sent to PTY wrapped in PtyCommand::Write
        let received = pty_rx.try_recv();
        assert!(received.is_ok(), "Should receive data on PTY channel");
        let cmd = received.unwrap();
        assert_eq!(cmd, PtyCommand::Write(vec![b'a']));
    }
}
