use crate::ansi::{AnsiBatch, AnsiCommand, AnsiSink};
use crate::color::Color;
use crate::config::Config;
use crate::glyph::Glyph;
use crate::io::event_monitor_actor::{PtyWriterHandle, WriterControl};
use crate::io::traits::PtySender;
use crate::io::Resize;
use crate::messages::TerminalData;
use crate::term::action::{Selection, SelectionReport};
use crate::term::{EmulatorAction, EmulatorInput, TerminalEmulator, UserInputAction};
use actor_scheduler::{
    Actor, ActorBuilder, ActorHandle, ActorStatus, HandlerError, HandlerResult, Message,
    SystemStatus,
};
use pixelflow_core::{CellGridMetrics, CellGridShape};
use pixelflow_graphics::render::cell_grid::{CellGridPackedManifold, CellGridPackedParams};
use pixelflow_graphics::render::scene::{constant_platform_scene, Scene};

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

/// Feeds a PTY batch to the emulator: text as runs, commands one at a time,
/// keeping every action the commands ask for so none is lost.
struct EmulatorSink<'a> {
    emulator: &'a mut TerminalEmulator,
    actions: &'a mut Vec<EmulatorAction>,
}

impl AnsiSink for EmulatorSink<'_> {
    fn text(&mut self, run: &str) {
        self.emulator.print_text(run);
    }

    fn command(&mut self, command: AnsiCommand) {
        if let Some(action) = self.emulator.interpret_input(EmulatorInput::Ansi(command)) {
            self.actions.push(action);
        }
    }
}

impl PtySender for TerminalAppSender {
    fn send(&self, batch: AnsiBatch) -> Result<(), anyhow::Error> {
        self.handle
            .send(Message::Data(TerminalData::Pty(batch)))
            .map_err(|e| anyhow::anyhow!("Failed to send PTY data to app: {}", e))
    }

    fn send_child_exited(&self) -> Result<(), anyhow::Error> {
        self.handle
            .send(Message::Data(TerminalData::ChildExited))
            .map_err(|e| anyhow::anyhow!("Failed to send child exit to app: {}", e))
    }
}

use pixelflow_graphics::fonts::loader::{LoadedFont, MmapSource};
use pixelflow_graphics::fonts::GlyphAtlas;
use pixelflow_runtime::api::private::EngineData;
use pixelflow_runtime::api::public::EngineHandle;
use pixelflow_runtime::api::public::{AppData, AppManagement};
use pixelflow_runtime::input::MouseButton;
use pixelflow_runtime::{EngineEventControl, EngineEventData, EngineEventManagement};
use std::collections::VecDeque;
use std::sync::Arc;

/// Font filename (looked up in multiple locations)
const FONT_FILENAME: &str = "NotoSansMono-Regular.ttf";

/// Atlas slots before the first growth (ASCII plus headroom).
const ATLAS_CAPACITY: usize = 128;

/// Scrollback navigation scales wheel delta by this many lines per unit,
/// since a raw 1:1 mapping feels sluggish for keyboard-free scrolling.
const SCROLL_LINES_PER_UNIT: i32 = 3;

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

/// Terminal application implementing Actor trait.
///
/// Receives engine events (frame requests, input) and responds with rendered
/// terminal content via the engine handle.
pub struct TerminalApp {
    pub emulator: TerminalEmulator,
    pty_writer: PtyWriterHandle,
    config: Config,
    engine_tx: EngineHandle,
    /// Memory-mapped font file.
    loaded_font: Arc<LoadedFont<MmapSource>>,
    /// Baked glyph coverage tiles, gathered by the scene kernel.
    atlas: GlyphAtlas,
    /// The compiled cell-grid scene and the metric it is currently drawn at;
    /// `None` until the first frame. See [`CellGridScene`].
    scene: Option<CellGridScene>,
    /// The solid-background scene and the device-pixel frame it was compiled
    /// for. Only ever drawn before anything has been presented (see
    /// [`TerminalApp::build_scene`]); cached because a `Scene` is a compiled
    /// kernel and compiling one per frame request would be absurd for four
    /// constants.
    background: Option<([u32; 2], Scene)>,
    /// Whether any scene has been submitted yet. Synchronized output holds
    /// the last frame, which only exists once this is true; before that, the
    /// solid background is the only honest thing to show.
    has_presented: bool,
    /// Currently pressed mouse button, tracked for motion reporting.
    /// Set on MouseClick, cleared on MouseRelease.
    pressed_mouse_button: Option<pixelflow_runtime::input::MouseButton>,
    /// Actions a PTY batch asked for, performed once the batch is applied.
    /// Kept between batches so a steady stream of replies allocates nothing.
    pty_actions: Vec<EmulatorAction>,
    /// Selection reads asked of the engine and not yet answered, oldest
    /// first, with what each answer is for. The engine answers every read
    /// once, in order per selection.
    clipboard_reads: VecDeque<(Selection, ClipboardRead)>,
    /// Device pixels per point of the current display (backing scale).
    /// The scene stays in point space; this is only a density hint for the
    /// glyph cache so bakes match the platform's sample lattice.
    density: f32,
    /// The window's size in LOGICAL POINTS, from the last `WindowCreated` /
    /// `Resized` (`Surface::width_px`, which is points despite the name).
    /// Scaled by `density` it is the device-pixel lattice the cell-grid
    /// kernels are compiled for, so it is part of the [`CellGridShape`] the
    /// recompile check compares.
    frame_px: [u32; 2],
}

/// What a selection read is for.
enum ClipboardRead {
    /// Paste the content into the program.
    Paste,
    /// Answer the program's OSC 52 query with it.
    Report(SelectionReport),
}

/// The compiled cell-grid scene, the metric it is drawn at, and that
/// metric's block: ONE packed kernel producing finished `u32` pixels, byte
/// order bound by the platform's ColorCube inside pixelflow-graphics.
///
/// **The two halves change on different schedules, which is why they are
/// different fields.** The program is compiled against *extents* — grid
/// dimensions, atlas extents, the frame's pixel size — and only those
/// recompile it. The metric (cell size, sample density, tile size, display
/// scale) is a block of uniforms, so a font-size or DPI change rewrites eight
/// floats and reuses the kernel. Neither happens per frame: `frame` is
/// refcounts, which is what keeps the render path free of per-frame
/// allocation.
struct CellGridScene {
    program: CellGridPackedManifold,
    metrics: CellGridMetrics,
    params: CellGridPackedParams,
}

impl CellGridScene {
    fn new(program: CellGridPackedManifold, metrics: CellGridMetrics) -> Self {
        let params = program.params(&metrics);
        Self {
            program,
            metrics,
            params,
        }
    }

    /// Adopt `metrics`, laying out a new block only when it actually moved.
    /// Laying one out allocates (the values are shared with the frame still
    /// in flight), so doing it per frame would put a heap allocation on the
    /// render path for no change at all.
    fn set_metrics(&mut self, metrics: CellGridMetrics) {
        if self.metrics != metrics {
            self.params = self.program.params(&metrics);
            self.metrics = metrics;
        }
    }
}

/// Parameters for constructing a TerminalApp.
pub struct TerminalAppParams {
    /// Terminal emulator instance.
    pub emulator: TerminalEmulator,
    /// Handle to the PTY writer actor (Data = bytes, Control = resize).
    pub pty_writer: PtyWriterHandle,
    /// Application configuration.
    pub config: Config,
    /// Unregistered engine handle (app will call register()).
    pub unregistered_engine: pixelflow_runtime::UnregisteredEngineHandle,
    /// Window configuration for registration.
    pub window_config: pixelflow_runtime::WindowConfig,
}

impl TerminalApp {
    /// Send bytes to the shell via the PTY writer's data lane.
    fn write_pty(&self, bytes: Vec<u8>) {
        if let Err(e) = self.pty_writer.send(Message::Data(bytes)) {
            log::warn!("Failed to send input to PTY writer: {}", e);
        }
    }

    /// Reports a mouse event to the program, in whatever encoding it asked for.
    fn report_mouse(
        &self,
        button: MouseButton,
        (x, y): (u32, u32),
        kind: crate::term::MouseEventKind,
    ) {
        let (col, row) = self.emulator.cell_at(x, y);
        let params = crate::term::MouseEncodingParams {
            button,
            col,
            row,
            kind,
        };
        if let Some(bytes) = self.emulator.encode_mouse_event(params) {
            self.write_pty(bytes);
        }
    }

    /// Hands a user action to the emulator and carries out what it asks for.
    fn interpret_user_input(&mut self, input: UserInputAction) {
        if let Some(action) = self.emulator.interpret_input(EmulatorInput::User(input)) {
            self.perform(action);
        }
    }

    /// Carries out what the emulator asked of the world outside it.
    fn perform(&mut self, action: EmulatorAction) {
        match action {
            EmulatorAction::WritePty(bytes) => self.write_pty(bytes),
            EmulatorAction::ResizePty { cols, rows } => self.resize_pty(cols, rows),
            EmulatorAction::RequestRedraw => self.send_frame(),
            EmulatorAction::Quit => self.request_quit(),
            EmulatorAction::SetTitle(title) => self.request_engine(AppManagement::SetTitle(title)),
            EmulatorAction::RingBell => self.request_engine(AppManagement::Bell),
            EmulatorAction::Copy { selection, text } => {
                self.request_engine(AppManagement::Copy { selection, text })
            }
            // The text comes back as `EngineEventManagement::Paste`.
            EmulatorAction::RequestClipboardContent(selection) => {
                self.read_selection(selection, ClipboardRead::Paste)
            }
            EmulatorAction::ReportSelection { selection, report } => {
                if !self.config.behavior.allow_clipboard_read {
                    log::debug!("OSC 52: clipboard reads are disabled; not answering");
                    return;
                }
                self.read_selection(selection, ClipboardRead::Report(report))
            }
            EmulatorAction::ToggleFullscreen => {
                self.request_engine(AppManagement::ToggleFullscreen)
            }
        }
    }

    /// Asks the engine for a selection's content, remembering what it is for.
    fn read_selection(&mut self, selection: Selection, purpose: ClipboardRead) {
        self.clipboard_reads.push_back((selection, purpose));
        self.request_engine(AppManagement::RequestPaste(selection));
    }

    /// Delivers a selection's content to the read that asked for it. Content
    /// nothing asked for is a paste.
    fn selection_read(&mut self, selection: Selection, text: String) {
        let purpose = self
            .clipboard_reads
            .iter()
            .position(|(asked, _)| *asked == selection)
            .and_then(|index| self.clipboard_reads.remove(index))
            .map(|(_, purpose)| purpose);
        match purpose {
            Some(ClipboardRead::Report(report)) => self.write_pty(report.reply(&text)),
            Some(ClipboardRead::Paste) | None if text.is_empty() => {}
            Some(ClipboardRead::Paste) | None => {
                self.interpret_user_input(UserInputAction::PasteText(text))
            }
        }
    }

    /// Asks the engine for something the terminal cannot do itself.
    fn request_engine(&self, request: AppManagement) {
        if let Err(e) = self.engine_tx.send(Message::Management(request)) {
            log::warn!("Failed to send request to engine: {}", e);
        }
    }

    /// Shuts the application down. Without an engine there is no window to
    /// keep alive, so a failure to ask is fatal.
    fn request_quit(&self) {
        self.engine_tx
            .send(Message::Management(AppManagement::Quit))
            .expect("Failed to send Quit to engine");
    }

    /// Resize the PTY via the writer's control lane (preempts queued writes).
    fn resize_pty(&self, cols: u16, rows: u16) {
        if let Err(e) = self
            .pty_writer
            .send(Message::Control(WriterControl::Resize(Resize {
                cols,
                rows,
            })))
        {
            log::warn!("Failed to send PTY resize command: {}", e);
        }
    }

    /// Creates a new terminal app (internal - use spawn_terminal_app instead).
    fn new_registered(params: TerminalAppParamsRegistered) -> Self {
        let font_path = params.font_path;
        let source = MmapSource::open(&font_path).unwrap_or_else(|e| {
            panic!("Failed to open font file at {}: {}", font_path.display(), e)
        });

        let loaded_font = Arc::new(LoadedFont::new(source).unwrap_or_else(|| {
            // `expect("Failed to parse font")` discarded the path and the
            // size — and cost a CI round-trip with a bespoke diagnostic
            // harness to learn that the "font" was a 131-byte Git LFS
            // pointer. The parse returns no error of its own, so what the
            // file actually IS is the only signal available; report it.
            let size = std::fs::metadata(&font_path).map(|m| m.len());
            let lfs_pointer = std::fs::read(&font_path)
                .is_ok_and(|b| b.starts_with(b"version https://git-lfs.github.com/spec/v1"));
            panic!(
                "Failed to parse font at {} ({}){}",
                font_path.display(),
                match size {
                    Ok(n) => format!("{n} bytes"),
                    Err(e) => format!("size unknown: {e}"),
                },
                if lfs_pointer {
                    " — this file is a Git LFS pointer, not a font. Run \
                     `git lfs pull` (or check out with LFS enabled)."
                } else {
                    ""
                }
            )
        }));

        // Bake the ASCII set into the atlas. Density 1.0 until the platform
        // reports the real backing scale via WindowCreated.
        let cell_height = params.config.appearance.cell_height_px as f32;
        let mut atlas = GlyphAtlas::new(cell_height, 1.0, ATLAS_CAPACITY);
        atlas.warm(&loaded_font.font(), ' '..='~');

        Self {
            emulator: params.emulator,
            pty_writer: params.pty_writer,
            config: params.config,
            engine_tx: params.engine_tx,
            loaded_font,
            atlas,
            scene: None,
            background: None,
            has_presented: false,
            frame_px: [0, 0],
            pressed_mouse_button: None,
            pty_actions: Vec::new(),
            clipboard_reads: VecDeque::new(),
            density: 1.0,
        }
    }

    /// Adopt a new display density (device pixels per point). The atlas
    /// rebuild and the scene-program recompile both happen lazily in
    /// `build_scene`, which sizes them from the snapshot's actual cell
    /// geometry.
    fn set_density(&mut self, scale: f64) {
        assert!(
            scale.is_finite() && scale > 0.0,
            "invalid display scale: {scale}"
        );
        self.density = scale as f32;
    }

    /// Rebuild the atlas when its bake parameters no longer match the cell
    /// geometry the emulator is actually using — first frame after a density
    /// change, or a snapshot cell height that differs from the config the
    /// startup atlas was sized from. The atlas is bound to one (font, size,
    /// density); following the snapshot here is what keeps that binding
    /// honest.
    fn ensure_atlas(&mut self, cell_height: f32) {
        if self.atlas.size_pt() == cell_height && self.atlas.density() == self.density {
            return;
        }
        self.atlas = GlyphAtlas::new(cell_height, self.density, ATLAS_CAPACITY);
        self.atlas.warm(&self.loaded_font.font(), ' '..='~');
    }

    /// The device-pixel lattice the scene kernels bake over: the window's
    /// logical points scaled by the display density.
    ///
    /// Window events carry LOGICAL POINTS (`Surface::width_px`) while the
    /// frame buffer the renderer hands us is device pixels, so the conversion
    /// is not optional. `None` before the first window event, when there is
    /// no window to measure and nothing to present into.
    fn device_frame(&self) -> Option<[u32; 2]> {
        let scaled = [
            (self.frame_px[0] as f32 * self.density).round() as u32,
            (self.frame_px[1] as f32 * self.density).round() as u32,
        ];
        (scaled[0] != 0 && scaled[1] != 0).then_some(scaled)
    }

    /// A solid `rgba` over the whole frame, compiled once per frame size.
    ///
    /// `None` when no window has been sized yet: there is nothing to present
    /// into, so the engine is answered with [`AppData::Skipped`].
    fn background_scene(&mut self, rgba: [f32; 4]) -> Option<Scene> {
        let frame = self.device_frame()?;
        let hit = self.background.as_ref().filter(|(at, _)| *at == frame);
        if hit.is_none() {
            self.background = Some((frame, constant_platform_scene(rgba, frame)));
        }
        self.background.as_ref().map(|(_, scene)| scene.clone())
    }

    /// Build the frame scene: the JIT cell-grid program over the glyph
    /// atlas and this snapshot's per-cell data.
    ///
    /// The per-frame work is filling one flat `f32` buffer (10 floats per
    /// cell); the compiled program is reused until an EXTENT changes — the
    /// grid's dimensions, the atlas's, or the window's. A change of *metric*
    /// (font size, display scale, tile size) rewrites the uniform block and
    /// keeps the program, so only a window resize or an atlas growth is a
    /// recompile — four channel kernels, sized independently of the grid.
    /// `None` means nothing has changed since the last frame and the engine
    /// should be answered with [`AppData::Skipped`] rather than a scene.
    fn build_scene(&mut self) -> Option<Scene> {
        let (dbg_r, dbg_g, dbg_b, dbg_a) = self.config.colors.background.to_f32_rgba();

        // Get terminal snapshot
        let snapshot = match self.emulator.get_render_snapshot() {
            Some(s) => s,
            None => {
                // Synchronized output (DECSET 2026): the application asked us
                // to hold the last frame until its batch ends, so painting
                // anything here — the background included — is exactly what
                // the mode forbids. Skip, and the engine keeps what is shown.
                // Any mutations made during the batch set per-line dirt that
                // survives (only a delivered snapshot clears it), so the
                // batch-end frame renders; a batch with no mutations leaves
                // the held frame, which is correct, not stale.
                if self.has_presented {
                    return None;
                }
                // Nothing was ever presented: there is no frame to hold, so
                // paint the default background over the whole frame. Four
                // constant channel kernels compiled at the frame's own shape
                // — a packed scene like every other, cached because its
                // extents are what it was compiled for.
                return self.background_scene([dbg_r, dbg_g, dbg_b, dbg_a]);
            }
        };

        let (cols, rows) = snapshot.dimensions;
        let cell_width = snapshot.cell_width_px as f32;
        let cell_height = snapshot.cell_height_px as f32;

        // This is the only place the answer is available: `get_render_snapshot`
        // stamps each line's dirt and then calls `mark_all_clean`, so the flags
        // live in the snapshot and nowhere else once it returns.
        //
        // Per-line dirt is sufficient for the *content* half of the question —
        // the snapshot dirties the rows a moved/restyled cursor left and
        // entered (`last_cursor_mark` in the emulator) and selection changes
        // mark their lines (`screen.rs`), so those are not separate cases. It
        // says nothing about *geometry*, which is why the grid shape is checked
        // separately below rather than trusted to come with a dirty line.
        let nothing_drawn_changed = !snapshot.lines.iter().any(|line| line.is_dirty);
        let geometry_matches = self.scene.as_ref().is_some_and(|scene| {
            let shape = scene.program.shape();
            shape.cols == cols as u32
                && shape.rows == rows as u32
                && scene.metrics.cell_w == cell_width
                && scene.metrics.cell_h == cell_height
                && scene.metrics.scale == self.density
        });
        // `geometry_matches` is false while `scene` is `None`, so the first
        // frame after startup always draws.
        if nothing_drawn_changed && geometry_matches {
            return None;
        }

        // Default colors
        let default_fg = self.config.colors.foreground;
        let default_bg = self.config.colors.background;

        // The snapshot's cell geometry is the source of truth for the
        // atlas's bake size, not the startup config.
        self.ensure_atlas(cell_height);

        // Fill the cell buffer FIRST: baking a previously unseen glyph may
        // grow the atlas, and the program must be compiled against the
        // atlas extents the frame actually binds.
        let font = self.loaded_font.font();
        let blank = self.atlas.blank_uv();
        let mut cells = Vec::with_capacity(cols * rows * pixelflow_core::CELL_STRIDE);
        for row in 0..rows {
            let line = &snapshot.lines[row];
            for col in 0..cols {
                let (uv, fg, bg): ((f32, f32), Color, Color) = match &line.cells[col] {
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
                        (self.atlas.uv(&font, cc.c), fg, bg)
                    }
                    // A wide glyph's spacer cell shows its background.
                    Glyph::WideSpacer => (blank, default_fg, default_bg),
                };
                let (fg_r, fg_g, fg_b, _) = fg.to_f32_rgba();
                let (bg_r, bg_g, bg_b, _) = bg.to_f32_rgba();
                cells
                    .extend_from_slice(&[uv.0, uv.1, fg_r, fg_g, fg_b, 1.0, bg_r, bg_g, bg_b, 1.0]);
            }
        }

        // The metric is in POINT space; the display scale is the contramap
        // onto the frame's device-pixel lattice, not a factor folded into
        // every extent. The atlas is baked at `density` texels per point, so
        // that is exactly what `density` means here.
        let metrics = CellGridMetrics {
            cell_w: cell_width,
            cell_h: cell_height,
            density: self.density,
            tile_w: self.atlas.tile_px() as u32,
            tile_h: self.atlas.tile_px() as u32,
            scale: self.density,
        };
        // The lattice the compiled kernels bake over. Before the first window
        // event there is no window to measure: fall back to the grid's own
        // extent, which is what the kernels would bake if the surface were
        // tight to the grid.
        let frame_px = self.device_frame().unwrap_or([
            (cols as f32 * cell_width * self.density).round() as u32,
            (rows as f32 * cell_height * self.density).round() as u32,
        ]);
        let shape = CellGridShape {
            cols: cols as u32,
            rows: rows as u32,
            atlas_width: self.atlas.width() as u32,
            atlas_height: self.atlas.height() as u32,
            frame_w: frame_px[0],
            frame_h: frame_px[1],
        };
        // (Re)compile only when an EXTENT moved — the grid gaining a column,
        // the atlas growing, the window changing size. A change of metric
        // (font size, display scale, tile size) is a parameter write into the
        // block, which is what `set_metrics` does below.
        let scene = match self.scene.take() {
            Some(scene) if *scene.program.shape() == shape => scene,
            _ => {
                log::info!(
                    "Compiling cell-grid scene: {}x{} cells, frame {}x{} px, atlas {}x{} texels",
                    cols,
                    rows,
                    shape.frame_w,
                    shape.frame_h,
                    shape.atlas_width,
                    shape.atlas_height
                );
                let program = pixelflow_graphics::render::scene::compile_platform_cell_grid(
                    shape,
                    [dbg_r, dbg_g, dbg_b, dbg_a],
                );
                CellGridScene::new(program, metrics)
            }
        };
        let scene = self.scene.insert(scene);
        scene.set_metrics(metrics);
        Some(Scene::CellGrid(scene.program.frame(
            &scene.params,
            Arc::new(cells),
            self.atlas.buffer(),
        )))
    }

    /// Answer the engine's frame request.
    ///
    /// Either with a scene, or — when nothing has changed — with
    /// [`AppData::Skipped`], which returns the vsync token without waking the
    /// coordinator. An idle terminal must not rasterize: the request arrives on
    /// a fixed timer (`target_fps`, not a display link), so answering every one
    /// with pixels means re-rendering identical output for as long as the
    /// window is open.
    fn send_frame(&mut self) {
        // The scene kernel paints the default background outside the grid
        // itself, so the frame is the scene — no outer clip/backdrop layer.
        let data = match self.build_scene() {
            Some(scene) => AppData::RenderSurface(scene),
            None => AppData::Skipped,
        };
        let presented = matches!(data, AppData::RenderSurface(_));
        match self
            .engine_tx
            .send(Message::Data(EngineData::FromApp(data)))
        {
            // Only a delivered scene counts: synchronized output holds "the
            // last frame", which must mean one that actually reached the
            // engine.
            Ok(()) => self.has_presented |= presented,
            Err(e) => log::warn!("Failed to send frame to engine: {}", e),
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
            TerminalData::Pty(mut batch) => {
                // Drawing is left to the next vsync frame request; only the
                // actions (replies to the shell, title, bell, ...) run now.
                let mut actions = std::mem::take(&mut self.pty_actions);
                batch.drain_into(&mut EmulatorSink {
                    emulator: &mut self.emulator,
                    actions: &mut actions,
                });
                for action in actions.drain(..) {
                    self.perform(action);
                }
                self.pty_actions = actions;
            }
            TerminalData::ChildExited => {
                log::info!("PTY child exited, shutting down");
                self.request_quit();
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
                    "[TERM] Window created: id={}, {}x{} points, scale={}",
                    id.0,
                    width_px,
                    height_px,
                    scale
                );
                self.set_density(scale);
                self.frame_px = [width_px, height_px];

                // Window is now ready - send initial frame to start VSync loop
                self.send_frame();
            }
            EngineEventControl::Resized {
                id: _,
                width_px,
                height_px,
            } => {
                self.frame_px = [width_px, height_px];
                use crate::term::{ControlEvent, EmulatorInput};
                // Convert u32 pixels to u16 for ControlEvent
                // Saturate at u16::MAX to prevent overflow panics
                let width_u16 = width_px.min(u16::MAX as u32) as u16;
                let height_u16 = height_px.min(u16::MAX as u32) as u16;

                let input = EmulatorInput::Control(ControlEvent::Resize {
                    width_px: width_u16,
                    height_px: height_u16,
                });

                if let Some(action) = self.emulator.interpret_input(input) {
                    self.perform(action);
                }

                // Request a redraw after resize
                self.send_frame();
            }
            EngineEventControl::CloseRequested => {
                // The engine is already running its shutdown cascade (vsync,
                // rasterizer, driver, itself). Our job is local cleanup: stop
                // the PTY writer so no further writes race the teardown. The
                // PTY master closes when the process exits, which delivers
                // SIGHUP to the child shell.
                log::info!("[TERM] Close requested; shutting down PTY writer");
                self.pty_writer
                    .send(Message::Shutdown)
                    .expect("Failed to shutdown PTY writer on CloseRequested");
            }
            EngineEventControl::ScaleChanged { id, scale } => {
                // The scene stays in point space (grid, cell metrics, mouse
                // math are all unchanged); only the glyph cache cares, since
                // its baked lattices must match the new sample density.
                log::info!("[TERM] Scale changed: id={}, scale={}", id.0, scale);
                self.set_density(scale);
                self.send_frame();
            }
        }
        Ok(())
    }

    fn handle_management(&mut self, mgmt: EngineEventManagement) -> HandlerResult {
        match mgmt {
            EngineEventManagement::KeyDown { key, mods, text } => {
                // A bound chord is the terminal's own command; anything else
                // is typing for the program.
                let input = crate::keys::map_key_event_to_action(key, mods, &self.config)
                    .unwrap_or(UserInputAction::KeyInput {
                        symbol: key,
                        modifiers: mods,
                        text: text.map(std::borrow::Cow::Owned),
                    });
                self.interpret_user_input(input);
            }
            EngineEventManagement::MouseClick { button, x, y } => {
                self.pressed_mouse_button = Some(button);
                if self.emulator.is_mouse_tracking_active() {
                    self.report_mouse(button, (x, y), crate::term::MouseEventKind::Press);
                    return Ok(());
                }
                match button {
                    MouseButton::Left => {
                        self.interpret_user_input(UserInputAction::StartSelection {
                            x_px: saturate_u16(x),
                            y_px: saturate_u16(y),
                        })
                    }
                    MouseButton::Middle => {
                        self.interpret_user_input(UserInputAction::RequestPrimaryPaste)
                    }
                    _ => {}
                }
            }
            EngineEventManagement::MouseRelease { button, x, y } => {
                self.pressed_mouse_button = None;
                if self.emulator.is_mouse_tracking_active() {
                    self.report_mouse(button, (x, y), crate::term::MouseEventKind::Release);
                    return Ok(());
                }
                if button == MouseButton::Left {
                    self.interpret_user_input(UserInputAction::ApplySelectionClear);
                }
            }
            EngineEventManagement::MouseMove { x, y, mods: _ } => {
                // any-event mode (1003) reports all motion; button-event mode
                // (1002) only motion while a button is held.
                if self.emulator.reports_all_motion() {
                    let button = self.pressed_mouse_button.unwrap_or(MouseButton::Left);
                    self.report_mouse(button, (x, y), crate::term::MouseEventKind::Motion);
                    return Ok(());
                }
                if self.emulator.reports_button_motion() {
                    if let Some(button) = self.pressed_mouse_button {
                        self.report_mouse(button, (x, y), crate::term::MouseEventKind::Motion);
                    }
                    return Ok(());
                }
                if self.emulator.is_mouse_tracking_active() {
                    return Ok(());
                }
                if self.pressed_mouse_button == Some(MouseButton::Left) {
                    self.interpret_user_input(UserInputAction::ExtendSelection {
                        x_px: saturate_u16(x),
                        y_px: saturate_u16(y),
                    });
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
                    let (col, row) = self.emulator.cell_at(x, y);
                    let button = if dy < 0.0 {
                        MouseButton::ScrollUp
                    } else {
                        MouseButton::ScrollDown
                    };
                    if let Some(bytes) =
                        self.emulator
                            .encode_mouse_event(crate::term::MouseEncodingParams {
                                button,
                                col,
                                row,
                                kind: crate::term::MouseEventKind::Press,
                            })
                    {
                        self.write_pty(bytes);
                    }
                } else {
                    // Scrollback navigation: negative dy scrolls up (into history),
                    // positive dy scrolls down (toward live screen)
                    let scroll_lines = -(dy as i32) * SCROLL_LINES_PER_UNIT;
                    if self.emulator.scroll_viewport(scroll_lines) {
                        // Viewport changed, send frame immediately for responsive scrolling
                        self.send_frame();
                    }
                }
            }
            EngineEventManagement::FocusGained => {
                self.interpret_user_input(UserInputAction::FocusGained);
            }
            EngineEventManagement::FocusLost => {
                self.interpret_user_input(UserInputAction::FocusLost);
            }
            EngineEventManagement::Paste { selection, text } => {
                self.selection_read(selection, text);
            }
        }
        Ok(())
    }

    fn handle_os(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        // No polling needed - PTY data comes in via handle_data
        Ok(ActorStatus::Idle)
    }
}

/// A window coordinate as the emulator's selection actions take it.
fn saturate_u16(px: u32) -> u16 {
    u16::try_from(px).unwrap_or(u16::MAX)
}

/// Handles returned by [`spawn_terminal_app`]: a keep-alive handle for the
/// caller, handles for the PTY parser and reader sinks, and the app thread's
/// join handle.
pub type TerminalAppHandles = (
    actor_scheduler::ActorHandle<TerminalData, EngineEventControl, EngineEventManagement>,
    actor_scheduler::ActorHandle<TerminalData, EngineEventControl, EngineEventManagement>,
    actor_scheduler::ActorHandle<TerminalData, EngineEventControl, EngineEventManagement>,
    std::thread::JoinHandle<()>,
);

/// Creates terminal app and spawns it in a thread.
///
/// This function handles registration atomically:
/// 1. Creates the app actor's channel
/// 2. Registers the app with the engine (sends RegisterApp + CreateWindow)
/// 3. Spawns the app thread with the registered engine handle
pub fn spawn_terminal_app(params: TerminalAppParams) -> std::io::Result<TerminalAppHandles> {
    // Create app actor's channels using ActorBuilder (SPSC - each producer is unique)
    // ActorHandle is not Clone; each consumer needs its own dedicated handle.
    let mut builder =
        ActorBuilder::<TerminalData, EngineEventControl, EngineEventManagement>::new(128, None);
    let app_handle = builder.add_producer(); // For the caller (returns to main, keep-alive)
    let parser_handle = builder.add_producer(); // For the PTY parser sink (AnsiCommands)
    let reader_handle = builder.add_producer(); // For the PTY reader sink (ChildExited)
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
        pty_writer: params.pty_writer,
        config: params.config,
        engine_tx,
        font_path: find_font_path(),
    };

    let mut app = TerminalApp::new_registered(app_params_registered);

    // Spawn app thread
    let handle = std::thread::Builder::new()
        .name("terminal-app".to_string())
        .spawn(move || {
            app_rx.run(&mut app);
        })?;

    Ok((app_handle, parser_handle, reader_handle, handle))
}

/// Parameters after registration (internal use).
struct TerminalAppParamsRegistered {
    emulator: TerminalEmulator,
    pty_writer: PtyWriterHandle,
    config: Config,
    engine_tx: EngineHandle,
    /// The font file to memory-map.
    font_path: std::path::PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::event_monitor_actor::WriterManagement;
    use crate::term::TerminalEmulator;
    use actor_scheduler::{
        Actor, ActorScheduler, ActorStatus, HandlerError, HandlerResult, SystemStatus,
    };
    use pixelflow_runtime::input::{KeySymbol, Modifiers};
    use pixelflow_runtime::{EngineEventControl, EngineEventManagement, WindowId};

    /// Test double for the PTY writer actor: records what the app sends.
    #[derive(Default)]
    struct WriterProbe {
        data: Vec<Vec<u8>>,
        resizes: Vec<Resize>,
    }

    impl Actor<Vec<u8>, WriterControl, WriterManagement> for WriterProbe {
        fn handle_data(&mut self, bytes: Vec<u8>) -> HandlerResult {
            self.data.push(bytes);
            Ok(())
        }
        fn handle_control(&mut self, msg: WriterControl) -> HandlerResult {
            let WriterControl::Resize(resize) = msg;
            self.resizes.push(resize);
            Ok(())
        }
        fn handle_management(&mut self, _msg: WriterManagement) -> HandlerResult {
            Ok(())
        }
        fn handle_os(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    type WriterScheduler = ActorScheduler<Vec<u8>, WriterControl, WriterManagement>;

    /// Drain everything the app has sent to the writer so far.
    fn drain_writer(rx: &mut WriterScheduler, probe: &mut WriterProbe) {
        for _ in 0..4 {
            if rx.poll_once(probe) {
                break;
            }
        }
    }

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

    /// The committed fallback font. The app's own font is LFS-tracked, and a
    /// checkout without git-lfs holds a pointer there, not a font; the
    /// fallback is exempt from LFS so these tests always run.
    fn test_font_path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../pixelflow-graphics/assets/DejaVuSansMono-Fallback.ttf")
    }

    // Helper to create a test instance
    // Returns scheduler to keep doorbell channel alive during test
    fn create_test_app() -> (
        TerminalApp,
        WriterScheduler,
        pixelflow_runtime::api::private::EngineActorHandle,
        pixelflow_runtime::api::private::EngineActorScheduler,
    ) {
        create_test_app_with(Config::default())
    }

    fn create_test_app_with(
        config: Config,
    ) -> (
        TerminalApp,
        WriterScheduler,
        pixelflow_runtime::api::private::EngineActorHandle,
        pixelflow_runtime::api::private::EngineActorScheduler,
    ) {
        let emulator = TerminalEmulator::new(80, 24);
        let (pty_writer, writer_rx) =
            ActorScheduler::<Vec<u8>, WriterControl, WriterManagement>::new(64, 128);

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

        let params = TerminalAppParamsRegistered {
            emulator,
            pty_writer,
            config,
            engine_tx: EngineHandle::new_for_test(engine_tx_for_test),
            font_path: test_font_path(),
        };
        let app = TerminalApp::new_registered(params);

        (app, writer_rx, engine_tx, engine_scheduler)
    }

    #[test]
    fn it_should_resize_the_emulator_and_forward_a_pty_resize_on_control_resize() {
        let (mut app, mut writer_rx, _, _scheduler) = create_test_app();

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
        app.handle_control(resize_event)
            .expect("handle_control should succeed");

        // Verify resize via snapshot
        let snapshot_new = app.emulator.get_render_snapshot().expect("Snapshot");
        assert_eq!(
            snapshot_new.dimensions,
            (100, 50),
            "Emulator should have resized to 100x50"
        );

        // Verify the resize went out on the writer's control lane
        let mut probe = WriterProbe::default();
        drain_writer(&mut writer_rx, &mut probe);
        assert_eq!(
            probe.resizes,
            vec![Resize {
                cols: 100,
                rows: 50
            }],
            "PTY resize command should match new dimensions"
        );
    }

    /// Engine double that discards every message, so the app's frame sends
    /// never block. Runs on its own thread like the real engine actor.
    struct EngineDiscard;

    impl
        Actor<
            pixelflow_runtime::api::private::EngineData,
            pixelflow_runtime::api::private::EngineControl,
            pixelflow_runtime::api::public::AppManagement,
        > for EngineDiscard
    {
        fn handle_data(
            &mut self,
            _msg: pixelflow_runtime::api::private::EngineData,
        ) -> HandlerResult {
            Ok(())
        }
        fn handle_control(
            &mut self,
            _msg: pixelflow_runtime::api::private::EngineControl,
        ) -> HandlerResult {
            Ok(())
        }
        fn handle_management(
            &mut self,
            _msg: pixelflow_runtime::api::public::AppManagement,
        ) -> HandlerResult {
            Ok(())
        }
        fn handle_os(&mut self, _hint: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    /// End-to-end regression for "can't Ctrl-C out of `yes`": real PTY troupe,
    /// real TerminalApp actor on a real scheduler, production-shaped producer
    /// set and burst limits. Floods the pipeline with `yes` output, then
    /// injects the exact KeyDown the X11 mapper produces for Ctrl+C and
    /// asserts the child dies (i.e. 0x03 reached the PTY line discipline).
    #[test]
    fn ctrl_c_interrupts_yes_flood() {
        use crate::io::event_monitor_actor::PtyTroupe;
        use crate::io::pty::{NixPty, PtyChannel, PtyConfig};
        use std::time::{Duration, Instant};

        let pty = NixPty::spawn_with_config(&PtyConfig {
            command_executable: "/bin/sh",
            args: &["-c", "yes"],
            initial_cols: 80,
            initial_rows: 24,
            working_directory: None,
        })
        .expect("spawn pty running yes");
        let child = pty.child_pid();

        let mut troupe = PtyTroupe::new(pty).expect("pty troupe");
        let pty_writer = troupe.writer_handle();

        // Engine double on its own thread (real engine drains fast; so does this).
        let mut engine_builder = actor_scheduler::ActorBuilder::<
            pixelflow_runtime::api::private::EngineData,
            pixelflow_runtime::api::private::EngineControl,
            pixelflow_runtime::api::public::AppManagement,
        >::new(1024, None);
        let engine_tx = engine_builder.add_producer();
        let mut engine_rx = engine_builder.build();
        let engine_thread = std::thread::spawn(move || {
            engine_rx.run(&mut EngineDiscard);
        });

        // App channels mirror spawn_terminal_app: 4 producers, data burst 10.
        let mut builder =
            ActorBuilder::<TerminalData, EngineEventControl, EngineEventManagement>::new(128, None);
        let key_tx = builder.add_producer(); // stands in for the engine adapter
        let parser_handle = builder.add_producer();
        let reader_handle = builder.add_producer();
        let keepalive = builder.add_producer();
        let mut app_rx = builder.build_with_burst(10, actor_scheduler::ShutdownMode::default());

        let mut app = TerminalApp::new_registered(TerminalAppParamsRegistered {
            emulator: TerminalEmulator::new(80, 24),
            pty_writer,
            config: Config::default(),
            engine_tx: EngineHandle::new_for_test(engine_tx),
            font_path: test_font_path(),
        });
        let app_thread = std::thread::spawn(move || {
            app_rx.run(&mut app);
        });

        let troupe_handle = troupe
            .spawn(
                Box::new(TerminalAppSender::new(parser_handle)),
                Box::new(TerminalAppSender::new(reader_handle)),
            )
            .expect("spawn pty troupe");

        // Engine-adapter stand-in on its own thread (the handle is single-owner):
        // hammers the adapter shard with RequestFrame at ~155Hz so the app is
        // doing full-grid send_frame work, then injects Ctrl+C after 2s of
        // flood WHILE the frame pressure keeps running — just like production,
        // where vsync doesn't pause because a key was pressed.
        //
        // The KeyDown is exactly what platform/linux/events.rs reports for
        // Ctrl+C: XLookupString applies the control translation, so text and
        // symbol are both ETX.
        let stop_vsync = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let vsync_stop = stop_vsync.clone();
        let adapter_thread = std::thread::spawn(move || {
            use pixelflow_runtime::api::public::EngineEventData;
            let interval = Duration::from_micros(6450);
            let key_at = Instant::now() + Duration::from_secs(2);
            let mut key_sent = false;
            while !vsync_stop.load(std::sync::atomic::Ordering::Relaxed) {
                let now = Instant::now();
                key_tx
                    .send(Message::Data(TerminalData::Engine(
                        EngineEventData::RequestFrame {
                            timestamp: now,
                            target_timestamp: now + interval,
                            refresh_interval: interval,
                        },
                    )))
                    .expect("send RequestFrame");
                if !key_sent && Instant::now() >= key_at {
                    key_tx
                        .send(Message::Management(EngineEventManagement::KeyDown {
                            key: pixelflow_runtime::input::KeySymbol::Char('c'),
                            mods: pixelflow_runtime::input::Modifiers::CONTROL,
                            text: Some("\u{3}".to_string()),
                        }))
                        .expect("send ctrl-c keydown");
                    key_sent = true;
                }
                std::thread::sleep(interval);
            }
        });

        // The line discipline should SIGINT the foreground job promptly.
        // (Key is injected ~2s in; allow generous slack on loaded CI.)
        let deadline = Instant::now() + Duration::from_secs(15);
        let mut dead = false;
        while Instant::now() < deadline {
            use nix::sys::wait::{waitpid, WaitPidFlag, WaitStatus};
            match waitpid(child, Some(WaitPidFlag::WNOHANG)) {
                Ok(WaitStatus::StillAlive) => std::thread::sleep(Duration::from_millis(50)),
                _ => {
                    dead = true;
                    break;
                }
            }
        }

        // Cleanup regardless of outcome so a failure doesn't leak `yes`.
        if !dead {
            let _killed = nix::sys::signal::kill(child, nix::sys::signal::Signal::SIGKILL);
        }
        stop_vsync.store(true, std::sync::atomic::Ordering::Relaxed);
        adapter_thread.join().expect("adapter thread");
        drop(troupe_handle); // shuts down reader/parser/writer, joins troupe thread
        drop(keepalive);
        app_thread.join().expect("app thread");
        engine_thread.join().expect("engine thread");

        assert!(
            dead,
            "Ctrl+C did not interrupt `yes` within 10s — input path is wedged"
        );
    }

    #[test]
    fn it_should_write_the_typed_character_to_the_pty_on_keydown() {
        let (mut app, mut writer_rx, _, _scheduler) = create_test_app();

        // Simulate KeyDown
        let key_event = EngineEventManagement::KeyDown {
            key: KeySymbol::Char('a'),
            mods: Modifiers::empty(),
            text: Some("a".to_string()),
        };

        app.handle_management(key_event)
            .expect("handle_management should succeed");

        // We expect 'a' on the writer's data lane
        let mut probe = WriterProbe::default();
        drain_writer(&mut writer_rx, &mut probe);
        assert_eq!(probe.data, vec![vec![b'a']]);
    }

    /// Test double for the engine actor: records the scenes the app renders
    /// and the requests it makes.
    #[derive(Default)]
    struct EngineProbe {
        scenes: Vec<Scene>,
        requests: Vec<pixelflow_runtime::api::public::AppManagement>,
    }

    impl
        Actor<
            EngineData,
            pixelflow_runtime::api::private::EngineControl,
            pixelflow_runtime::api::public::AppManagement,
        > for EngineProbe
    {
        fn handle_data(&mut self, msg: EngineData) -> HandlerResult {
            if let EngineData::FromApp(AppData::RenderSurface(scene)) = msg {
                self.scenes.push(scene);
            }
            Ok(())
        }
        fn handle_control(
            &mut self,
            _msg: pixelflow_runtime::api::private::EngineControl,
        ) -> HandlerResult {
            Ok(())
        }
        fn handle_management(
            &mut self,
            msg: pixelflow_runtime::api::public::AppManagement,
        ) -> HandlerResult {
            self.requests.push(msg);
            Ok(())
        }
        fn handle_os(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    /// Drain the frame(s) the app has sent to the engine so far.
    fn drain_engine(
        rx: &mut pixelflow_runtime::api::private::EngineActorScheduler,
        probe: &mut EngineProbe,
    ) {
        for _ in 0..4 {
            if rx.poll_once(probe) {
                break;
            }
        }
    }

    fn request_frame() -> TerminalData {
        let now = std::time::Instant::now();
        TerminalData::Engine(EngineEventData::RequestFrame {
            timestamp: now,
            target_timestamp: now,
            refresh_interval: std::time::Duration::from_millis(16),
        })
    }

    #[test]
    fn scene_paints_default_background_and_recompiles_on_resize() {
        // The app compiles its kernel for the PLATFORM's pixel format, so
        // the frame must be that format too — a hardcoded one was correct
        // only while rendering converted per pixel.
        use pixelflow_graphics::render::color::PlatformPixel;
        use pixelflow_graphics::render::frame::Frame;

        let (mut app, _writer_rx, _tx, mut engine_scheduler) = create_test_app();
        let (r, g, b, _) = app.config.colors.background.to_f32_rgba();
        let close = |got: u8, want: f32| (got as f32 - want * 255.0).abs() <= 2.0;

        // A blank 80x24 screen is spaces on the default background: the JIT
        // cell-grid scene the app sends the engine must rasterize to that
        // color. Drive it through the actor's real entry point — a
        // RequestFrame data message — rather than the private scene-builder.
        app.handle_data(request_frame()).expect("request frame");
        let mut probe = EngineProbe::default();
        drain_engine(&mut engine_scheduler, &mut probe);
        let scene = probe.scenes.pop().expect("app sent a frame");
        let mut frame = Frame::<PlatformPixel>::new(16, 16);
        scene.render(&mut frame, 1);
        let px = frame.data[8 * 16 + 8];
        assert!(
            close(px.r(), r) && close(px.g(), g) && close(px.b(), b),
            "blank scene pixel {:?} != default background ({r}, {g}, {b})",
            (px.r(), px.g(), px.b()),
        );

        // A resize is a recompile: the cell buffer the next frame builds is
        // always sized to the CURRENT terminal snapshot, so a program whose
        // geometry failed to move with it would panic on the length
        // mismatch the moment the next frame binds it. Rendering
        // successfully after the resize is itself the proof.
        let resize = EngineEventControl::Resized {
            id: WindowId(0),
            width_px: 500,
            height_px: 320,
        };
        app.handle_control(resize).expect("resize");
        app.handle_data(request_frame())
            .expect("request frame after resize");
        let mut probe = EngineProbe::default();
        drain_engine(&mut engine_scheduler, &mut probe);
        let scene = probe.scenes.pop().expect("app sent a post-resize frame");
        let mut frame = Frame::<PlatformPixel>::new(16, 16);
        scene.render(&mut frame, 1);
        let px = frame.data[8 * 16 + 8];
        assert!(
            close(px.r(), r) && close(px.g(), g) && close(px.b(), b),
            "post-resize scene pixel {:?} != default background ({r}, {g}, {b})",
            (px.r(), px.g(), px.b()),
        );
    }

    /// PTY output, as the parser actor delivers it.
    fn pty(bytes: &[u8]) -> TerminalData {
        use crate::ansi::{AnsiParser, AnsiProcessor};
        TerminalData::Pty(AnsiProcessor::new().process_bytes(bytes))
    }

    #[test]
    fn a_device_status_request_from_the_shell_is_answered_on_the_pty() {
        let (mut app, mut writer_rx, _tx, _engine) = create_test_app();

        app.handle_data(pty(b"\x1b[6n")).expect("pty data");

        let mut probe = WriterProbe::default();
        drain_writer(&mut writer_rx, &mut probe);
        assert_eq!(probe.data, vec![b"\x1b[1;1R".to_vec()]);
    }

    #[test]
    fn a_bell_from_the_shell_rings_the_engine_bell() {
        let (mut app, _writer_rx, _tx, mut engine) = create_test_app();

        app.handle_data(pty(b"\x07")).expect("pty data");

        let mut probe = EngineProbe::default();
        drain_engine(&mut engine, &mut probe);
        assert!(matches!(
            probe.requests.as_slice(),
            [pixelflow_runtime::api::public::AppManagement::Bell]
        ));
    }

    #[test]
    fn a_title_from_the_shell_sets_the_window_title() {
        let (mut app, _writer_rx, _tx, mut engine) = create_test_app();

        app.handle_data(pty(b"\x1b]2;build: ok\x07"))
            .expect("pty data");

        let mut probe = EngineProbe::default();
        drain_engine(&mut engine, &mut probe);
        assert!(matches!(
            probe.requests.as_slice(),
            [pixelflow_runtime::api::public::AppManagement::SetTitle(title)] if title == "build: ok"
        ));
    }

    #[test]
    fn the_paste_binding_asks_the_engine_for_the_clipboard() {
        use pixelflow_runtime::input::{KeySymbol, Modifiers};
        let (mut app, _writer_rx, _tx, mut engine) = create_test_app();

        // As X11 reports Ctrl+Shift+V: the shifted keysym, and what it typed.
        app.handle_management(EngineEventManagement::KeyDown {
            key: KeySymbol::Char('V'),
            mods: Modifiers::CONTROL | Modifiers::SHIFT,
            text: Some("\u{16}".to_string()),
        })
        .expect("key down");

        let mut probe = EngineProbe::default();
        drain_engine(&mut engine, &mut probe);
        assert!(matches!(
            probe.requests.as_slice(),
            [pixelflow_runtime::api::public::AppManagement::RequestPaste(
                pixelflow_runtime::input::Selection::Clipboard
            )]
        ));
    }

    #[test]
    fn pasted_text_reaches_the_shell_bracketed_when_it_asked() {
        let (mut app, mut writer_rx, _tx, _engine) = create_test_app();

        app.handle_data(pty(b"\x1b[?2004h")).expect("pty data");
        app.handle_management(clipboard_answer(Selection::Clipboard, "ls\n"))
            .expect("paste");

        let mut probe = WriterProbe::default();
        drain_writer(&mut writer_rx, &mut probe);
        assert_eq!(probe.data, vec![b"\x1b[200~ls\n\x1b[201~".to_vec()]);
    }

    fn clipboard_answer(selection: Selection, text: &str) -> EngineEventManagement {
        EngineEventManagement::Paste {
            selection,
            text: text.to_string(),
        }
    }

    #[test]
    fn an_osc_52_query_is_answered_with_the_clipboard() {
        let (mut app, mut writer_rx, _tx, mut engine) = create_test_app();

        app.handle_data(pty(b"\x1b]52;c;?\x07")).expect("pty data");
        let mut requests = EngineProbe::default();
        drain_engine(&mut engine, &mut requests);
        assert!(matches!(
            requests.requests.as_slice(),
            [AppManagement::RequestPaste(Selection::Clipboard)]
        ));

        app.handle_management(clipboard_answer(Selection::Clipboard, "hi"))
            .expect("clipboard answer");
        let mut probe = WriterProbe::default();
        drain_writer(&mut writer_rx, &mut probe);
        assert_eq!(probe.data, vec![b"\x1b]52;c;aGk=\x1b\\".to_vec()]);
    }

    #[test]
    fn an_osc_52_query_goes_unanswered_when_clipboard_reads_are_off() {
        let mut config = Config::default();
        config.behavior.allow_clipboard_read = false;
        let (mut app, mut writer_rx, _tx, mut engine) = create_test_app_with(config);

        app.handle_data(pty(b"\x1b]52;c;?\x07")).expect("pty data");

        let mut requests = EngineProbe::default();
        drain_engine(&mut engine, &mut requests);
        assert!(
            requests.requests.is_empty(),
            "the clipboard must not be read"
        );
        let mut probe = WriterProbe::default();
        drain_writer(&mut writer_rx, &mut probe);
        assert!(probe.data.is_empty());
    }

    #[test]
    fn each_clipboard_answer_goes_to_the_read_that_asked_for_it() {
        let (mut app, mut writer_rx, _tx, _engine) = create_test_app();

        // A query of the primary selection, then a clipboard query and a
        // clipboard paste. The primary owner answers last.
        app.handle_data(pty(b"\x1b]52;p;?\x1b\\\x1b]52;c;?\x07"))
            .expect("pty data");
        app.handle_management(EngineEventManagement::KeyDown {
            key: pixelflow_runtime::input::KeySymbol::Char('V'),
            mods: pixelflow_runtime::input::Modifiers::CONTROL
                | pixelflow_runtime::input::Modifiers::SHIFT,
            text: None,
        })
        .expect("paste binding");
        for (selection, text) in [
            (Selection::Clipboard, "first"),
            (Selection::Clipboard, "second"),
            (Selection::Primary, "third"),
        ] {
            app.handle_management(clipboard_answer(selection, text))
                .expect("clipboard answer");
        }

        let mut probe = WriterProbe::default();
        drain_writer(&mut writer_rx, &mut probe);
        assert_eq!(
            probe.data,
            vec![
                b"\x1b]52;c;Zmlyc3Q=\x1b\\".to_vec(),
                b"second".to_vec(),
                b"\x1b]52;p;dGhpcmQ=\x1b\\".to_vec(),
            ]
        );
    }

    #[test]
    fn an_empty_clipboard_pastes_nothing() {
        let (mut app, mut writer_rx, _tx, _engine) = create_test_app();

        app.handle_data(pty(b"\x1b[?2004h")).expect("pty data");
        app.handle_management(clipboard_answer(Selection::Clipboard, ""))
            .expect("paste");

        let mut probe = WriterProbe::default();
        drain_writer(&mut writer_rx, &mut probe);
        assert!(probe.data.is_empty(), "no bracketed empty paste");
    }

    #[test]
    fn focus_changes_reach_the_shell_once_it_asks_for_them() {
        let (mut app, mut writer_rx, _tx, _engine) = create_test_app();

        app.handle_management(EngineEventManagement::FocusLost)
            .expect("focus lost");
        app.handle_data(pty(b"\x1b[?1004h")).expect("pty data");
        app.handle_management(EngineEventManagement::FocusGained)
            .expect("focus gained");

        let mut probe = WriterProbe::default();
        drain_writer(&mut writer_rx, &mut probe);
        assert_eq!(probe.data, vec![b"\x1b[I".to_vec()]);
    }

    #[test]
    fn the_zoom_binding_resizes_the_pty_and_the_next_frame_draws_at_the_new_size() {
        use pixelflow_runtime::input::{KeySymbol, Modifiers};
        let (mut app, mut writer_rx, _tx, mut engine) = create_test_app();
        app.handle_control(EngineEventControl::Resized {
            id: WindowId(0),
            width_px: 800,
            height_px: 480,
        })
        .expect("resize");

        // As X11 reports Ctrl+Shift+=: the shifted keysym.
        app.handle_management(EngineEventManagement::KeyDown {
            key: KeySymbol::Char('+'),
            mods: Modifiers::CONTROL | Modifiers::SHIFT,
            text: None,
        })
        .expect("key down");

        let mut probe = WriterProbe::default();
        drain_writer(&mut writer_rx, &mut probe);
        let [initial, zoomed] = probe.resizes.as_slice() else {
            panic!(
                "expected the window resize and the zoom, got {:?}",
                probe.resizes
            );
        };
        assert!(zoomed.cols < initial.cols && zoomed.rows < initial.rows);

        // The cell buffer is sized to the zoomed grid; a program still
        // compiled for the old one would panic binding it.
        app.handle_data(request_frame()).expect("frame after zoom");
        let mut engine_probe = EngineProbe::default();
        drain_engine(&mut engine, &mut engine_probe);
        assert!(!engine_probe.scenes.is_empty(), "a zoomed frame was drawn");
    }

    #[test]
    fn a_mouse_drag_selects_and_becomes_the_primary_selection() {
        let (mut app, _writer_rx, _tx, mut engine) = create_test_app();
        app.handle_data(pty(b"hello world")).expect("pty data");

        // Default cells are 10x16 points: drag across "hello" on row 0.
        for event in [
            EngineEventManagement::MouseClick {
                button: MouseButton::Left,
                x: 1,
                y: 1,
            },
            EngineEventManagement::MouseMove {
                x: 41,
                y: 1,
                mods: Default::default(),
            },
            EngineEventManagement::MouseRelease {
                button: MouseButton::Left,
                x: 41,
                y: 1,
            },
        ] {
            app.handle_management(event).expect("mouse");
        }

        let mut probe = EngineProbe::default();
        drain_engine(&mut engine, &mut probe);
        assert!(
            probe.requests.iter().any(|request| matches!(
                request,
                pixelflow_runtime::api::public::AppManagement::Copy {
                    selection: pixelflow_runtime::input::Selection::Primary,
                    text,
                } if text == "hello"
            )),
            "requests: {:?}",
            probe.requests
        );
    }

    #[test]
    fn a_middle_click_pastes_the_primary_selection_and_f11_toggles_fullscreen() {
        use pixelflow_runtime::input::{KeySymbol, Modifiers};
        let (mut app, _writer_rx, _tx, mut engine) = create_test_app();

        app.handle_management(EngineEventManagement::MouseClick {
            button: MouseButton::Middle,
            x: 5,
            y: 5,
        })
        .expect("middle click");
        app.handle_management(EngineEventManagement::KeyDown {
            key: KeySymbol::F11,
            mods: Modifiers::empty(),
            text: None,
        })
        .expect("f11");

        let mut probe = EngineProbe::default();
        drain_engine(&mut engine, &mut probe);
        assert!(
            matches!(
                probe.requests.as_slice(),
                [
                    pixelflow_runtime::api::public::AppManagement::RequestPaste(
                        pixelflow_runtime::input::Selection::Primary
                    ),
                    pixelflow_runtime::api::public::AppManagement::ToggleFullscreen,
                ]
            ),
            "requests: {:?}",
            probe.requests
        );
    }
}
