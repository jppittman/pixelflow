//! Engine Troupe - Render pipeline actor coordination using troupe! macro.
//!
//! `EngineHandler` is the thin adapter around [`EngineCore`] (`engine_core.rs`): the core
//! decides, in the mealy-transducer style, what should happen and returns it as an
//! [`EngineOut`] word; `flush` is this file's hand-written `Wiring` — the one place that word
//! meets real channels.

use crate::api::private::{EngineControl, EngineData, GreenReadyBundle};
use crate::api::public::{AppManagement, Application};
use crate::config::EngineConfig;
use crate::coordinator_node::{CoordinatorCore, CoordinatorData, CoordinatorWiring};
use crate::display::driver::DriverActor;
use crate::display::messages::{DisplayControl, DisplayData, DisplayMgmt, WindowMeta};
use crate::display::platform::PlatformActor;
use crate::engine_core::{EngineCore, EngineOut};
use crate::error::RuntimeError;
use crate::platform::{ActivePlatform, PlatformPixel};
use crate::vsync_actor::{RenderedResponse, VsyncCommand, VsyncCore, VsyncManagement, VsyncWiring};
use actor_scheduler::actors::{Schedule, Timer};
use actor_scheduler::host::{GreenThread, Host};
use actor_scheduler::mealy::{send_port, Flush, Lanes, NoLane, Node, Transducer};
use actor_scheduler::{
    green_channel, Actor, ActorBuilder, ActorHandle, ActorScheduler, ActorStatus, ActorTypes,
    GreenSender, HandlerError, HandlerResult, Message, SchedulerParams, SendError, SystemStatus,
    TroupeActor,
};
use pixelflow_graphics::render::renderer::{RenderResponse, RendererActor, RendererHandle};
use std::convert::Infallible;
use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::Duration;

/// Engine handler - coordinates app, rendering, display.
pub struct EngineHandler {
    /// Handle to the display driver actor.
    driver: ActorHandle<DisplayData, DisplayControl, DisplayMgmt>,
    /// The vsync green node's control edge (`UpdateRefreshRate`, `ReturnToken`, ...).
    vsync_control: Option<GreenSender<VsyncCommand>>,
    /// A handle to the green host itself, for the shutdown cascade — sending it
    /// `Message::Shutdown` stops both green nodes it hosts (vsync and the render coordinator)
    /// along with everything else the host owns.
    vsync_host: Option<ActorHandle<Infallible, Infallible, Infallible>>,
    /// The render coordinator's data lane (`coordinator_node.rs`): window grants, scene
    /// submissions, and advance nudges, relayed from the driver/app/vsync edges below. `None`
    /// until `EngineControl::GreenReady` arrives.
    coordinator: Option<GreenSender<CoordinatorData>>,
    /// Handle to self (for shutdown).
    self_handle: Option<ActorHandle<EngineData, EngineControl, AppManagement>>,
    /// Handle to the renderer response-forwarding actor (spawned in `with_config`, before the
    /// green host, via the free `spawn_renderer` function; kept only for the shutdown
    /// cascade — the renderer's own live handle belongs to the coordinator's wiring now, see
    /// `GreenReadyBundle`'s doc).
    renderer_forwarder: Option<ActorHandle<Infallible, Infallible, Infallible>>,
    /// Handle to the application (for event forwarding).
    app_handle: Option<Arc<dyn Application + Send + Sync>>,
    /// The pure mediator: decides what to do with each message and returns it as an
    /// [`EngineOut`] word for `flush` to deliver. Owns none of the handles above.
    core: EngineCore,
}

// ActorTypes impls - required for troupe! macro
impl ActorTypes for EngineHandler {
    type Data = EngineData;
    type Control = EngineControl;
    type Management = AppManagement;
}

impl ActorTypes for DriverActor<ActivePlatform> {
    type Data = DisplayData;
    type Control = DisplayControl;
    type Management = DisplayMgmt;
}

// Generate troupe structures using macro
// Note: Renderer is NOT in the troupe - it uses a bootstrap handshake pattern
// that enforces type-level guarantees about initialization order.
//
// `driver` is `[expose]` as well as `[main]` now: the render coordinator's own green node
// (`coordinator_node.rs`, step 5c) needs a driver handle of its own, for `Present` and
// `RequestWindow` — sends the engine no longer makes on its behalf.
actor_scheduler::troupe! {
    driver: DriverActor<ActivePlatform> [main, expose],
    engine: EngineHandler [expose],
}

// Implement Actor for EngineHandler
impl Actor<EngineData, EngineControl, AppManagement> for EngineHandler {
    fn handle_data(&mut self, data: EngineData) -> HandlerResult {
        let out = self.core.step_data(data)?;
        self.flush(out);
        Ok(())
    }

    fn handle_control(&mut self, ctrl: EngineControl) -> HandlerResult {
        // Handle-carrying: intercepted before the core ever sees it, since a pure core has no
        // field to hold an `ActorHandle` in.
        let ctrl = match ctrl {
            EngineControl::GreenReady(bundle) => {
                let GreenReadyBundle {
                    vsync_control,
                    coordinator,
                    host,
                    forwarder,
                } = *bundle;
                self.vsync_control = Some(vsync_control);
                self.coordinator = Some(coordinator);
                self.vsync_host = Some(host);
                self.renderer_forwarder = Some(forwarder);
                return Ok(());
            }
            other => other,
        };
        let out = self.core.step_control(ctrl)?;
        self.flush(out);
        Ok(())
    }

    fn handle_management(&mut self, mgmt: AppManagement) -> HandlerResult {
        // Handle-carrying / bootstrap messages: intercepted before the core ever sees them, for
        // the same reason as `EngineControl::GreenReady` above.
        let mgmt = match mgmt {
            AppManagement::Configure(config) => {
                // The renderer is spawned by `with_config` itself now, before the green host —
                // it needs the coordinator's management-lane sender as its forwarder's target,
                // which does not exist until bootstrap builds it (`spawn_renderer`). Nothing
                // engine-side to do with this any more but log; kept as a message rather than
                // deleted outright in case a future engine-side setting wants to ride it.
                log::info!(
                    "Engine configured: {} render threads (renderer already spawned at bootstrap)",
                    config.performance.render_threads
                );
                return Ok(());
            }
            AppManagement::RegisterApp(app) => {
                log::info!("Application handle registered");
                self.app_handle = Some(app);
                return Ok(());
            }
            other => other,
        };
        let out = self.core.step_management(mgmt)?;
        self.flush(out);
        Ok(())
    }

    fn handle_os(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        // engine has no external channels which might be busy
        Ok(ActorStatus::Idle)
    }
}

/// Bridges the renderer's response channel straight into the render coordinator's own green
/// node — no engine round-trip (step 5c of the mesh migration doc).
///
/// [`RendererActor`] hands completed frames back over a bare `mpsc::Sender` so
/// pixelflow-graphics stays decoupled from any particular consumer's message enum. This actor
/// is the one place that channel meets the coordinator's management lane — `handle_os` blocking
/// on `response_rx.recv()` *is* the actor, the same way `PtyReader::handle_os` blocking on
/// `epoll_wait` is that actor's entire job.
///
/// The forward to `coordinator` uses [`GreenSender::send`], not the scheduler's port sender:
/// this is a handler pushing into another actor's inbox, not a `Wiring::flush`, so the
/// bounded-backoff-then-loud-timeout posture is the right one — `mealy::send_port` is reserved
/// for flush (see its doc).
///
/// The scheduler only calls `handle_os` after the doorbell has woken at least once, so — same
/// as `PtyReader` waiting for its first `Bind` — this actor needs one doorbell ring to start its
/// loop; `spawn_renderer` rings it right after spawning the thread. Once `handle_os` returns
/// `Busy` the scheduler keeps re-entering it without waiting on the doorbell again, so from then
/// on the loop is self-sustaining.
struct RasterizerForwarder {
    response_rx: std::sync::mpsc::Receiver<RenderResponse<PlatformPixel, WindowMeta>>,
    coordinator: GreenSender<RenderResponse<PlatformPixel, WindowMeta>>,
    /// Sends itself `Shutdown` once the renderer drops its sender, so the scheduler loop
    /// (and the OS thread underneath it) actually exits instead of parking on an empty
    /// doorbell forever. `Quit`/`AppManagement::Quit`/`CloseRequested` also send `Shutdown`
    /// here directly; this is the fallback for whichever signal arrives second.
    self_handle: Option<ActorHandle<Infallible, Infallible, Infallible>>,
}

impl ActorTypes for RasterizerForwarder {
    type Data = Infallible;
    type Control = Infallible;
    type Management = Infallible;
}

impl Actor<Infallible, Infallible, Infallible> for RasterizerForwarder {
    fn handle_data(&mut self, msg: Infallible) -> HandlerResult {
        match msg {}
    }

    fn handle_control(&mut self, msg: Infallible) -> HandlerResult {
        match msg {}
    }

    fn handle_management(&mut self, msg: Infallible) -> HandlerResult {
        match msg {}
    }

    fn handle_os(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        match self.response_rx.recv() {
            Ok(response) => match self.coordinator.send(response) {
                Ok(()) => Ok(ActorStatus::Busy),
                Err(SendError::Disconnected) => {
                    log::warn!("Coordinator gone; renderer forwarder shutting down");
                    self.shut_down();
                    Ok(ActorStatus::Idle)
                }
                // The render credit bounds this edge to one in flight, so the ring should never
                // stay full past the backoff window — a timeout means the host is wedged, and
                // the answer is a loud crash, not quiet retrying.
                Err(SendError::Timeout) => {
                    panic!("green host unresponsive: coordinator forward relay timed out")
                }
            },
            Err(_) => {
                // The renderer shut down and dropped its sender; there is nothing left to
                // forward, so this actor's work is done too.
                self.shut_down();
                Ok(ActorStatus::Idle)
            }
        }
    }
}

impl RasterizerForwarder {
    fn shut_down(&mut self) {
        if let Some(handle) = self.self_handle.take() {
            if let Err(e) = handle.send(Message::Shutdown) {
                log::debug!("Renderer forwarder self-shutdown send failed: {}", e);
            }
        }
    }
}

impl EngineHandler {
    /// Deliver an [`EngineOut`] — the hand-written `Wiring` for [`EngineCore`]'s output word,
    /// until the engine runs as a green node behind the generated port/wiring machinery.
    ///
    /// Order matters here in one place: `app` is flushed first and `quit` last, so a
    /// `CloseRequested` (which sets both `app` and `quit` in the same word) reaches the app
    /// before the shutdown cascade drops its handle — matching what all three quit paths did
    /// before this split.
    fn flush(&mut self, out: EngineOut) {
        if let Some(event) = out.app {
            // Silently dropped if no app is registered yet, matching every `if let Some(app) =
            // &self.app_handle` guard this replaces.
            if let Some(app) = &self.app_handle {
                app.send(event)
                    .expect("failed to send to app. it probably crashed");
            }
        }

        if let Some(ctrl) = out.driver_control {
            self.driver
                .send(Message::Control(ctrl))
                .expect("Failed to relay control to driver");
        }

        if let Some(mgmt) = out.driver_mgmt {
            self.driver
                .send(Message::Management(mgmt))
                .expect("Failed to relay management to driver");
        }

        if let Some(data) = out.coordinator {
            self.send_coordinator(data);
        }

        if let Some(cmd) = out.vsync_control {
            self.send_vsync_control(cmd);
        }

        if out.quit {
            self.shut_down();
        }
    }

    /// The `coordinator` port: every window grant, scene submission, and advance nudge the
    /// engine relays to the render coordinator's own green node.
    ///
    /// This edge is not credit-bounded — `Submit`s are unsolicited (the app pushes frames on
    /// resize and interactive redraws, outside the vsync token protocol) and `Advance`s follow
    /// window events, which a resize drag emits in floods — so a full ring is ordinary
    /// backpressure, and the policy for that belongs to the scheduler, not here:
    /// [`GreenSender::send`] blocks with the same bounded backoff as `ActorHandle::send`, then
    /// fails with `Timeout` if the host stays unresponsive past the window. The wait cannot
    /// deadlock — the green side never *blocks* toward the engine (its wirings `send_port` and
    /// park), so the host is always live and draining while this thread waits. And the timeout
    /// is not tolerated: a wedged green host is a broken runtime, and per the fail-fast mantra
    /// the answer is a crash that says so, not quiet degradation.
    fn send_coordinator(&mut self, data: CoordinatorData) {
        let Some(tx) = &self.coordinator else {
            return; // Not configured yet, or the green host has already shut down.
        };
        match tx.send(data) {
            // Disconnected is the ordinary shutdown race: the host is gone and everything it
            // hosted dies with the process.
            Ok(()) | Err(SendError::Disconnected) => {}
            Err(SendError::Timeout) => {
                panic!("green host unresponsive: coordinator relay timed out")
            }
        }
    }

    /// The `vsync_control` port — `ReturnToken` and `UpdateRefreshRate`. Same discipline as
    /// [`Self::send_coordinator`]: the scheduler's bounded blocking is the backpressure, a
    /// timeout is a wedged host and fails loudly, and a disconnect is an ordinary shutdown
    /// race. (In practice this edge never even blocks: `ReturnToken`s outstanding are bounded
    /// by `MAX_TOKENS`, below the ring's capacity.)
    fn send_vsync_control(&mut self, cmd: VsyncCommand) {
        let Some(tx) = &self.vsync_control else {
            return; // Not configured yet, or the green host has already shut down.
        };
        match tx.send(cmd) {
            Ok(()) | Err(SendError::Disconnected) => {}
            Err(SendError::Timeout) => {
                panic!("green host unresponsive: vsync control relay timed out")
            }
        }
    }

    /// The shutdown cascade all three quit paths (`EngineControl::Quit`,
    /// `AppManagement::Quit`, `DisplayEvent::CloseRequested`) share: the green host, the
    /// renderer forwarder, drop the app handle, driver, then self. Any app notification for a
    /// `CloseRequested` has already gone out via the `app` port earlier in [`Self::flush`].
    ///
    /// The renderer itself is *not* sent an explicit `Shutdown` here — it cannot be: its one
    /// live handle belongs to the coordinator's wiring (see `GreenReadyBundle`'s doc), which
    /// this cascade already reaches indirectly. Shutting down the green host drops `Host`, which
    /// drops the coordinator `Node` and its `CoordinatorWiring` along with it — including that
    /// sole renderer handle. The renderer's own scheduler sees every one of its lanes
    /// disconnect and halts gracefully (the same "all disconnected ⇒ normal shutdown" path the
    /// forwarder below already relies on one hop downstream), so the cascade still reaches it,
    /// just by the handle dropping rather than a message arriving.
    fn shut_down(&mut self) {
        if let Some(host) = &self.vsync_host {
            host.send(Message::Shutdown)
                .expect("Failed to shutdown green host on Quit");
        }
        // The host shutting down drops its lanes' receivers; drop our ends too so nothing
        // downstream mistakes a dead node for one still configured.
        self.vsync_control = None;
        self.coordinator = None;
        if let Some(forwarder) = &self.renderer_forwarder {
            forwarder
                .send(Message::Shutdown)
                .expect("Failed to shutdown renderer forwarder on Quit");
        }
        self.app_handle = None;
        self.driver
            .send(Message::Shutdown)
            .expect("Failed to shutdown driver on Quit");
        if let Some(self_handle) = &self.self_handle {
            self_handle
                .send(Message::Shutdown)
                .expect("Failed to shutdown engine on Quit");
        }
    }
}

/// Spawn the renderer actor with its bootstrap handshake, and the forwarder that turns its
/// bare `mpsc` responses into direct sends on the coordinator's management lane.
///
/// A free function, not a method on `EngineHandler`: nothing here needs the engine any more once
/// the forwarder's target is the coordinator instead of it (step 5c of the mesh migration doc).
/// Called once from `Troupe::with_config`, before the green-host thread spawns — it needs only
/// `render_threads` from config and the coordinator's already-built management-lane sender.
///
/// Returns the renderer's own handle (which goes straight into `CoordinatorWiring`, never to
/// the engine — see `GreenReadyBundle`'s doc for why it can't be shared) and the forwarder's
/// handle (which the engine keeps for its shutdown cascade).
fn spawn_renderer(
    render_threads: usize,
    coordinator: GreenSender<RenderResponse<PlatformPixel, WindowMeta>>,
) -> (
    RendererHandle<PlatformPixel, WindowMeta>,
    ActorHandle<Infallible, Infallible, Infallible>,
) {
    // Step 1: Spawn renderer with setup handle
    let (setup_handle, _rasterizer_thread) =
        RendererActor::<PlatformPixel, WindowMeta>::spawn_with_setup(render_threads);

    // Step 2: Create response channel (the forwarder receives render results here)
    let (response_tx, response_rx) =
        std::sync::mpsc::channel::<RenderResponse<PlatformPixel, WindowMeta>>();

    // Step 3: Run the forwarder as a real actor rather than a bare thread — it is addressable (a
    // real Shutdown on the Quit paths, not an implicit exit whenever the renderer happens to
    // drop its sender), managed the same way the rest of the troupe is. Its lanes are
    // `Infallible`: nothing but `Shutdown` and the doorbell ever reaches it, so
    // `data_buffer_size` of 1 is a formality.
    let mut builder = ActorBuilder::new(1, None);
    let self_handle = builder.add_producer();
    let forwarder_handle = builder.add_producer();
    let mut forwarder_scheduler: ActorScheduler<Infallible, Infallible, Infallible> =
        builder.build();
    let mut forwarder = RasterizerForwarder {
        response_rx,
        coordinator,
        self_handle: Some(self_handle),
    };
    std::thread::Builder::new()
        .name("renderer-forwarder".into())
        .spawn(move || {
            forwarder_scheduler.run(&mut forwarder);
        })
        .expect("failed to spawn renderer forwarder thread");
    // The scheduler blocks on its doorbell until woken; with `Infallible` lanes there is no
    // message to send, so ring the doorbell directly to start the loop.
    forwarder_handle.waker().wake();

    // Step 4: Complete bootstrap - register response channel and get full handle
    let rasterizer_handle = setup_handle.register(response_tx);

    log::info!("Renderer actor initialized via bootstrap");
    (rasterizer_handle, forwarder_handle)
}

// Implement TroupeActor for EngineHandler — takes ownership of per-actor Directory
impl TroupeActor<Directory> for EngineHandler {
    fn new(dir: Directory) -> Self {
        Self {
            driver: dir.driver,
            vsync_control: None, // Set via EngineControl::GreenReady once the green host is up
            vsync_host: None,
            coordinator: None, // Set via EngineControl::GreenReady once the green host is up
            self_handle: Some(dir.engine),
            renderer_forwarder: None, // Set via EngineControl::GreenReady
            app_handle: None,
            core: EngineCore::new(),
        }
    }
}

// Implement TroupeActor for DriverActor — takes ownership of per-actor Directory
impl TroupeActor<Directory> for DriverActor<ActivePlatform> {
    fn new(dir: Directory) -> Self {
        #[cfg(target_os = "macos")]
        {
            use crate::platform::MetalOps;
            let ops = MetalOps::new().expect("Failed to create Metal ops");
            let platform = PlatformActor::new(ops, dir.engine);
            DriverActor::new(platform)
        }
        #[cfg(target_os = "linux")]
        {
            use crate::platform::linux::LinuxOps;
            let ops = LinuxOps::new().expect("Failed to create Linux ops");
            let platform = PlatformActor::new(ops, dir.engine);
            DriverActor::new(platform)
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            let _ = dir;
            panic!("Unsupported platform");
        }
    }
}

impl Troupe {
    /// Create troupe, configure the engine, and bring vsync *and the render coordinator* up as
    /// green nodes (docs/designs/actor-scheduler-mealy-transducer.md §5, and step 5c of
    /// `pixelflow-runtime-engine-mesh-migration.md` §8): one "green-host" thread runs a [`Host`]
    /// hosting both [`Node`]s, stepped by its own [`GreenThread`] rather than either living on a
    /// dedicated OS thread of its own. The one thread this bootstrap still spawns for vsync is
    /// the [`Timer`] clock — a green actor may not block, and a clock has to wait.
    pub fn with_config(config: EngineConfig) -> Result<Self, RuntimeError> {
        // The JIT's ISA tier is decided here, once, before any window opens: a
        // CPU below the floor is refused now, naming the feature it lacks,
        // rather than by the first frame's kernel faulting on an instruction
        // it cannot execute.
        log::info!("JIT ISA tier: {}", pixelflow_codegen::isa::detect().name());

        // Create troupe with platform-specific waker for the main (driver) actor
        #[cfg(target_os = "macos")]
        let mut troupe = {
            use crate::platform::waker::CocoaWaker;
            Self::new_with_waker(Some(std::sync::Arc::new(CocoaWaker::new())))
        };
        #[cfg(target_os = "linux")]
        let mut troupe = {
            use crate::platform::linux::set_shared_waker;
            use crate::platform::waker::X11Waker;
            let waker = X11Waker::new();
            set_shared_waker(waker.clone());
            Self::new_with_waker(Some(std::sync::Arc::new(waker)))
        };
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        let mut troupe = Self::new();

        // Create SPSC handles for initialization (each exposed() creates unique channels).
        // `init.driver` is the coordinator's own driver handle — grabbed here rather than with a
        // separate `exposed()` call, since one call already mints every exposed actor's handle.
        let init = troupe.exposed(); // engine: config messages; driver: the coordinator's handle
        let vsync_engine = troupe.exposed(); // dedicated engine producer for vsync ticks

        // Configure the engine with window settings
        init.engine
            .send(Message::Management(AppManagement::Configure(
                config.clone(),
            )))
            .map_err(|e| RuntimeError::InitError(format!("Failed to configure engine: {}", e)))?;

        // ── The green host: vsync AND the render coordinator ────────────────────────────
        //
        // A host has no lanes of its own (all three types are `Infallible`); work reaches it by
        // pushing straight into an adopted node's inbox with a `GreenSender` and ringing the
        // host's doorbell. `host_shutdown` is kept (handed to the engine below, for the shutdown
        // cascade); `host_kick` only lends its `Waker` to the green channels and gives the
        // scheduler its initial kick once the thread is running.
        let mut host_builder = ActorBuilder::<Infallible, Infallible, Infallible>::new(1, None);
        let host_shutdown = host_builder.add_producer();
        let host_kick = host_builder.add_producer();
        let mut host_sched = host_builder.build();

        let waker = host_kick.waker();
        // Capacities >= MAX_TOKENS (100, `vsync_actor.rs`): the credit argument (design doc
        // §3.2) needs the ring to hold every possible outstanding tick/`ReturnToken`, so `Full`
        // on either is provably a bug rather than backpressure to tolerate.
        let (vsync_data_tx, vsync_data_rx) = green_channel::<RenderedResponse>(128, waker.clone());
        let (vsync_control_tx, vsync_control_rx) =
            green_channel::<VsyncCommand>(128, waker.clone());
        // Capacity 1 is enough: a queued tick is as good as a fresh one, and the clock is the
        // only producer.
        let (tick_tx, tick_rx) = green_channel::<VsyncManagement>(2, waker.clone());

        // The coordinator's two lanes (§5c): data, fed by the engine shell
        // (Submit/Granted/Advance) — 256 covers every vsync-token-bounded frame request plus
        // every buffer grant that could be outstanding; management, fed directly by the
        // renderer forwarder (`RenderResponse`) — bounded by the render credit (<=1 in
        // flight), so its capacity is a formality, same as the forwarder's own ring.
        let (coordinator_data_tx, coordinator_data_rx) =
            green_channel::<CoordinatorData>(256, waker.clone());
        let (coordinator_mgmt_tx, coordinator_mgmt_rx) =
            green_channel::<RenderResponse<PlatformPixel, WindowMeta>>(8, waker);

        // The renderer is spawned here now, not by the engine (§8 of the mesh migration doc):
        // it needs only `render_threads` from config and the coordinator's management-lane
        // sender as the forwarder's target, so the `SetRasterizerForwardHandle` round trip that
        // used to gate `Configure` is gone entirely.
        let (renderer, forwarder) =
            spawn_renderer(config.performance.render_threads, coordinator_mgmt_tx);

        init.engine
            .send(Message::Control(EngineControl::GreenReady(Box::new(
                GreenReadyBundle {
                    vsync_control: vsync_control_tx,
                    coordinator: coordinator_data_tx,
                    host: host_shutdown,
                    forwarder,
                },
            ))))
            .map_err(|e| {
                RuntimeError::InitError(format!("Failed to hand green handles to engine: {}", e))
            })?;

        let refresh_rate = config.performance.target_fps as f64;
        let tick_interval = Duration::from_secs_f64(1.0 / refresh_rate);

        // Built here, on the calling thread, then moved into the green-host thread below —
        // `Timer` is `Send`, so this doesn't need to happen on the thread that owns it.
        let clock = Timer::spawn("vsync-clock", Schedule::Every(tick_interval), move || {
            let mut port = Some(VsyncManagement::Tick);
            match send_port(&mut port, &tick_tx) {
                Flush::Done => ControlFlow::Continue(()),
                // A queued tick is as good as this one.
                Flush::Blocked => ControlFlow::Continue(()),
                Flush::Disconnected => ControlFlow::Break(()),
            }
        });

        let coordinator_driver = init.driver;

        std::thread::Builder::new()
            .name("green-host".to_string())
            .spawn(move || {
                let mut host = Host::new();

                let vsync_core = VsyncCore::started(refresh_rate);
                let vsync_wiring = VsyncWiring {
                    engine: vsync_engine.engine,
                    clock,
                };
                let vsync_node = Node::new_with_lanes(
                    vsync_core,
                    Lanes {
                        control: vsync_control_rx,
                        management: tick_rx,
                        data: vsync_data_rx,
                    },
                    vsync_wiring,
                    SchedulerParams::DEFAULT,
                );
                host.adopt(vsync_node);

                // Adopted after vsync: sweep order between the two is not load-bearing here —
                // vsync's tick still relays through the engine rather than the coordinator, so
                // neither node's completion in one sweep depends on the other having already
                // run in that same sweep.
                let coordinator_core = CoordinatorCore::new();
                let coordinator_wiring = CoordinatorWiring {
                    renderer,
                    driver: coordinator_driver,
                    vsync: vsync_data_tx,
                };
                let coordinator_node = Node::new_with_lanes(
                    coordinator_core,
                    Lanes {
                        control: NoLane::new(),
                        management: coordinator_mgmt_rx,
                        data: coordinator_data_rx,
                    },
                    coordinator_wiring,
                    SchedulerParams::DEFAULT,
                );
                host.adopt(coordinator_node);

                let mut green_thread = GreenThread::new(host);
                host_sched.run(&mut green_thread);
            })
            .expect("failed to spawn green host thread");
        // The scheduler blocks on its doorbell until woken; ring it once to start the sweep
        // loop, the same kick `spawn_renderer`'s `RasterizerForwarder` gives itself.
        host_kick.waker().wake();

        Ok(troupe)
    }

    /// Get an unregistered engine handle.
    ///
    /// Creates a new SPSC producer for the engine actor.
    /// Must be called before `play()` (which consumes the builders).
    pub fn engine_handle(&mut self) -> crate::api::public::UnregisteredEngineHandle {
        let handles = self.exposed();
        crate::api::public::UnregisteredEngineHandle::new(handles.engine)
    }

    /// Get the raw engine actor handle for advanced use cases.
    ///
    /// Creates a new SPSC producer for the engine actor.
    /// Must be called before `play()` (which consumes the builders).
    pub fn raw_engine_handle(&mut self) -> crate::api::private::EngineActorHandle {
        let handles = self.exposed();
        handles.engine
    }
}

#[cfg(test)]
mod tests {
    //! `EngineHandler`, driven directly.
    //!
    //! The render protocol itself — request/render/present, resize races, `holds_buffer` — no
    //! longer runs through the engine at all, so those interaction tests moved to
    //! `coordinator_node.rs`'s `node_tests` (see that module's doc for the full list and why
    //! each one still exists). What is left here is what the engine still actually does: relay
    //! driver/app/vsync events to the render coordinator's `CoordinatorData` port, plus its own
    //! shutdown cascade.

    use super::*;
    use crate::api::public::{AppData, WindowId};
    use crate::coordinator_node::CoordinatorData;
    use crate::display::messages::{DisplayEvent, Surface};
    use actor_scheduler::spsc::{spsc_channel, SpscReceiver};
    use actor_scheduler::ActorScheduler;
    use std::time::Instant;

    /// Scheduler tuning for the fixture: small, since nothing here approaches a real ring.
    const LANE_BURST: usize = 16;
    const LANE_BUFFER: usize = 16;

    /// The lattice the fixture scene is compiled for. These tests never bake
    /// it — they assert which port fired, not what reached a buffer — so it
    /// only has to be a legal frame; the windows they grant are this size.
    const FIXTURE_FRAME: [u32; 2] = [100, 100];

    /// A constant black scene — these tests care about which port fires, not what is drawn.
    fn manifold() -> pixelflow_graphics::render::scene::Scene {
        crate::testing::black_scene(FIXTURE_FRAME)
    }

    /// `EngineHandler` wired to a real driver scheduler (so `driver_control`/`driver_mgmt`
    /// relays are observable) and a held `coordinator` receiver (so `CoordinatorData` arrivals
    /// are observable), with no green host, thread, or renderer in the loop.
    struct Rig {
        engine: EngineHandler,
        coordinator_rx: SpscReceiver<CoordinatorData>,
    }

    impl Rig {
        fn new() -> Self {
            Self::with_ring(64)
        }

        /// A rig with a chosen coordinator-ring capacity, for the overload-policy tests below —
        /// filling a 64-slot ring by hand would just bury the assertion in setup.
        fn with_ring(capacity: usize) -> Self {
            Self::with_ring_and_params(capacity, SchedulerParams::DEFAULT)
        }

        /// [`with_ring`](Self::with_ring) with explicit backoff tuning, so the timeout test can
        /// wedge in milliseconds instead of the production backoff window.
        fn with_ring_and_params(capacity: usize, params: SchedulerParams) -> Self {
            let (driver, _driver_sched) = ActorScheduler::new(LANE_BURST, LANE_BUFFER);

            let waker = {
                let (handle, _sched) =
                    ActorScheduler::<Infallible, Infallible, Infallible>::new(1, 1);
                handle.waker()
            };
            let (coordinator_tx, coordinator_rx) = spsc_channel::<CoordinatorData>(capacity);
            let coordinator = GreenSender::new_with_params(coordinator_tx, waker, params);

            let engine = EngineHandler {
                driver,
                vsync_control: None,
                vsync_host: None,
                coordinator: Some(coordinator),
                self_handle: None,
                renderer_forwarder: None,
                app_handle: None,
                core: EngineCore::new(),
            };

            Self {
                engine,
                coordinator_rx,
            }
        }

        fn feed(&mut self, data: EngineData) {
            self.engine
                .handle_data(data)
                .expect("engine handled the message");
        }
    }

    #[test]
    fn render_surface_relays_a_submit_to_the_coordinator() {
        let mut rig = Rig::new();
        rig.feed(EngineData::FromApp(AppData::RenderSurface(manifold())));
        assert!(matches!(
            rig.coordinator_rx.try_recv(),
            Ok(CoordinatorData::Submit(_))
        ));
    }

    #[test]
    fn skipped_frame_does_not_touch_the_coordinator() {
        let mut rig = Rig::new();
        rig.feed(EngineData::FromApp(AppData::Skipped));
        assert!(rig.coordinator_rx.try_recv().is_err());
    }

    #[test]
    fn window_granted_relays_to_the_coordinator() {
        use crate::display::messages::{Generation, Window, WindowMeta};

        let mut rig = Rig::new();
        let window = Window::rejoin(
            pixelflow_graphics::render::Frame::new(100, 100),
            WindowMeta {
                id: WindowId(1),
                width_px: 100,
                height_px: 100,
                scale: 1.0,
                generation: Generation::NONE,
            },
        );
        rig.feed(EngineData::WindowGranted(window));
        assert!(matches!(
            rig.coordinator_rx.try_recv(),
            Ok(CoordinatorData::Granted(_))
        ));
    }

    /// Fill a rig's coordinator ring to the brim with advance nudges, via `send_port` (the
    /// scheduler's own non-blocking port sender) — the engine relay itself blocks on Full, so
    /// it can never be used to *reach* Full.
    fn fill_coordinator_ring(rig: &Rig) {
        let tx = rig
            .engine
            .coordinator
            .as_ref()
            .expect("rig always wires a coordinator");
        loop {
            let mut port = Some(CoordinatorData::Advance);
            if send_port(&mut port, tx) != Flush::Done {
                break;
            }
        }
    }

    /// The overload case review flagged on #944: unsolicited app submissions are not bounded by
    /// the vsync credit, so a full coordinator ring is ordinary backpressure. The policy is the
    /// scheduler's (`GreenSender::send`): the relay waits for the ring to drain and delivers —
    /// nothing is shed, so the newest scene can never be silently lost, and nothing panics
    /// while the host is live.
    #[test]
    fn relays_wait_out_a_full_coordinator_ring_and_lose_nothing() {
        use crate::display::messages::{Generation, Window, WindowMeta};

        let mut rig = Rig::with_ring(2);
        fill_coordinator_ring(&rig);

        // Drain the ring from another thread after a delay, standing in for the green host: the
        // engine's relays must block until space appears, then deliver.
        let (done_tx, done_rx) = std::sync::mpsc::channel::<CoordinatorData>();
        let mut coordinator_rx = std::mem::replace(
            &mut rig.coordinator_rx,
            spsc_channel::<CoordinatorData>(1).1,
        );
        let drainer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            loop {
                match coordinator_rx.try_recv() {
                    Ok(data) => done_tx.send(data).expect("collector alive"),
                    Err(actor_scheduler::spsc::TryRecvError::Empty) => {
                        std::thread::sleep(Duration::from_millis(1))
                    }
                    Err(actor_scheduler::spsc::TryRecvError::Disconnected) => return,
                }
            }
        });

        // One of each relay against the full ring: all must block briefly and deliver.
        rig.feed(EngineData::FromApp(AppData::RenderSurface(manifold())));
        let window = Window::rejoin(
            pixelflow_graphics::render::Frame::new(100, 100),
            WindowMeta {
                id: WindowId(1),
                width_px: 100,
                height_px: 100,
                scale: 1.0,
                generation: Generation::NONE,
            },
        );
        rig.feed(EngineData::WindowGranted(window));

        drop(rig); // Disconnect the sender so the drainer exits.
        drainer.join().expect("drainer exits cleanly");
        let (mut submits, mut grants) = (0, 0);
        for data in done_rx.iter() {
            match data {
                CoordinatorData::Submit(_) => submits += 1,
                CoordinatorData::Granted(_) => grants += 1,
                CoordinatorData::Advance => {}
            }
        }
        assert_eq!(
            (submits, grants),
            (1, 1),
            "backpressure must deliver, not shed: the newest scene and the buffer both arrive"
        );
    }

    /// The other half of bounded patience: a host that stays unresponsive past the backoff
    /// window is wedged, and the relay fails loudly instead of waiting forever or dropping
    /// quietly.
    #[test]
    #[should_panic(expected = "green host unresponsive")]
    fn a_wedged_green_host_times_out_loudly() {
        // Short backoff so the test's wedged-host wait is milliseconds, not the production
        // window.
        let params = SchedulerParams {
            spin_attempts: 1,
            yield_attempts: 1,
            min_backoff: Duration::from_micros(10),
            max_backoff: Duration::from_micros(200),
            ..SchedulerParams::DEFAULT
        };
        let mut rig = Rig::with_ring_and_params(2, params);
        fill_coordinator_ring(&rig);

        // Nobody drains: the bounded backoff exhausts and the relay must panic.
        rig.feed(EngineData::FromApp(AppData::RenderSurface(manifold())));
    }

    #[test]
    fn vsync_tick_relays_advance_to_the_coordinator() {
        let mut rig = Rig::new();
        let now = Instant::now();
        rig.feed(EngineData::VSync {
            timestamp: now,
            target_timestamp: now,
            refresh_interval: Duration::from_millis(16),
        });
        assert!(matches!(
            rig.coordinator_rx.try_recv(),
            Ok(CoordinatorData::Advance)
        ));
    }

    #[test]
    fn window_created_and_resized_relay_advance_to_the_coordinator() {
        let mut rig = Rig::new();
        let surface = Surface {
            id: WindowId(1),
            width_px: 100,
            height_px: 100,
            frame_width: 100,
            frame_height: 100,
            scale: 1.0,
        };

        rig.feed(EngineData::FromDriver(DisplayEvent::WindowCreated {
            surface,
        }));
        assert!(matches!(
            rig.coordinator_rx.try_recv(),
            Ok(CoordinatorData::Advance)
        ));

        rig.feed(EngineData::FromDriver(DisplayEvent::Resized { surface }));
        assert!(matches!(
            rig.coordinator_rx.try_recv(),
            Ok(CoordinatorData::Advance)
        ));
    }

    /// The forwarder's whole reason to exist: turn the renderer's bare `mpsc` responses into
    /// direct sends on the coordinator's management lane, rather than a bare thread nobody could
    /// signal. Exercises the actual `RasterizerForwarder`, not a stand-in — simpler now than the
    /// engine-routed version, since a `GreenSender`'s receiving end is a plain `SpscReceiver`
    /// and no engine scheduler is needed to observe arrival.
    #[test]
    fn forwarder_relays_responses_and_exits_when_the_rasterizer_disconnects() {
        use crate::display::messages::Generation;

        let waker = {
            let (handle, _sched) = ActorScheduler::<Infallible, Infallible, Infallible>::new(1, 1);
            handle.waker()
        };
        let (coordinator_tx, mut coordinator_rx) =
            spsc_channel::<RenderResponse<PlatformPixel, WindowMeta>>(8);
        let coordinator = GreenSender::new(coordinator_tx, waker);

        let (response_tx, response_rx) = std::sync::mpsc::channel();

        let mut builder = ActorBuilder::new(1, None);
        let self_handle = builder.add_producer();
        let starter_handle = builder.add_producer();
        let mut forwarder_sched: ActorScheduler<Infallible, Infallible, Infallible> =
            builder.build();
        let mut forwarder = RasterizerForwarder {
            response_rx,
            coordinator,
            self_handle: Some(self_handle),
        };
        let thread = std::thread::spawn(move || forwarder_sched.run(&mut forwarder));
        starter_handle.waker().wake();

        let meta = WindowMeta {
            id: WindowId(1),
            width_px: 4,
            height_px: 4,
            scale: 1.0,
            generation: Generation::NONE,
        };
        response_tx
            .send(RenderResponse {
                frame: pixelflow_graphics::render::Frame::new(4, 4),
                render_time: Some(Duration::from_millis(1)),
                meta,
            })
            .expect("forwarder thread should still be listening");

        // Dropping the sender is exactly what the real bootstrap path does when the renderer
        // shuts down; the forwarder must notice and exit rather than parking on an empty
        // doorbell forever. Joining also proves the queued response was relayed before we read
        // the coordinator channel: `mpsc` reports disconnection only after queued messages are
        // drained.
        drop(response_tx);
        thread
            .join()
            .expect("forwarder must exit once the renderer disconnects");

        let response = coordinator_rx
            .try_recv()
            .expect("the forwarder must relay the renderer response");
        assert_eq!(
            (response.meta.width_px, response.meta.height_px),
            (4, 4),
            "the forwarder must translate the renderer's response into a direct coordinator send"
        );
    }

    #[test]
    fn skipped_frame_returns_the_vsync_token() {
        let (driver, _driver_sched) = ActorScheduler::new(LANE_BURST, LANE_BUFFER);
        let waker = {
            let (handle, _sched) = ActorScheduler::<Infallible, Infallible, Infallible>::new(1, 1);
            handle.waker()
        };
        let (vsync_tx, mut vsync_rx) = spsc_channel::<VsyncCommand>(8);
        let vsync_control = GreenSender::new(vsync_tx, waker);

        let mut engine = EngineHandler {
            driver,
            vsync_control: Some(vsync_control),
            vsync_host: None,
            coordinator: None,
            self_handle: None,
            renderer_forwarder: None,
            app_handle: None,
            core: EngineCore::new(),
        };

        engine
            .handle_data(EngineData::FromApp(AppData::Skipped))
            .expect("engine handled the message");

        assert!(
            matches!(vsync_rx.try_recv(), Ok(VsyncCommand::ReturnToken)),
            "a skipped frame must still return the vsync token, or the next frame starves"
        );
    }

    /// No-op stand-ins for `poll_once`-based observation below: these actors are never meant to
    /// do real work, only to let a scheduler report whether a `Shutdown` landed.
    struct NoopDriver;
    impl Actor<DisplayData, DisplayControl, DisplayMgmt> for NoopDriver {
        fn handle_data(&mut self, _msg: DisplayData) -> HandlerResult {
            Ok(())
        }
        fn handle_control(&mut self, _msg: DisplayControl) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _msg: DisplayMgmt) -> HandlerResult {
            Ok(())
        }
        fn handle_os(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    struct NoopEngine;
    impl Actor<EngineData, EngineControl, AppManagement> for NoopEngine {
        fn handle_data(&mut self, _msg: EngineData) -> HandlerResult {
            Ok(())
        }
        fn handle_control(&mut self, _msg: EngineControl) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _msg: AppManagement) -> HandlerResult {
            Ok(())
        }
        fn handle_os(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    struct Never;
    impl Actor<Infallible, Infallible, Infallible> for Never {
        fn handle_data(&mut self, msg: Infallible) -> HandlerResult {
            match msg {}
        }
        fn handle_control(&mut self, msg: Infallible) -> HandlerResult {
            match msg {}
        }
        fn handle_management(&mut self, msg: Infallible) -> HandlerResult {
            match msg {}
        }
        fn handle_os(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    /// Polls `sched` until it reports a `Shutdown` (`poll_once` returning `true`) or gives up —
    /// bounded so a regression fails fast with a clear message instead of hanging the suite.
    fn eventually_shuts_down<D, C, M, A: Actor<D, C, M>>(
        sched: &mut ActorScheduler<D, C, M>,
        actor: &mut A,
    ) -> bool {
        for _ in 0..10_000 {
            if sched.poll_once(actor) {
                return true;
            }
            std::thread::yield_now();
        }
        false
    }

    #[test]
    fn quit_shuts_down_the_driver_and_every_configured_handle() {
        let (driver, mut driver_sched) = ActorScheduler::new(LANE_BURST, LANE_BUFFER);
        let (self_handle, mut self_sched) = ActorScheduler::new(LANE_BURST, LANE_BUFFER);
        let (vsync_host, mut vsync_sched) =
            ActorScheduler::<Infallible, Infallible, Infallible>::new(1, 1);
        let (forwarder, mut forwarder_sched) =
            ActorScheduler::<Infallible, Infallible, Infallible>::new(1, 1);

        let mut engine = EngineHandler {
            driver,
            vsync_control: None,
            vsync_host: Some(vsync_host),
            coordinator: None,
            self_handle: Some(self_handle),
            renderer_forwarder: Some(forwarder),
            app_handle: None,
            core: EngineCore::new(),
        };

        engine
            .handle_control(EngineControl::Quit)
            .expect("quit handled");

        assert!(
            eventually_shuts_down(&mut driver_sched, &mut NoopDriver),
            "the driver must always receive Shutdown on Quit"
        );
        assert!(
            eventually_shuts_down(&mut vsync_sched, &mut Never),
            "the green host must receive Shutdown on Quit when configured"
        );
        assert!(
            eventually_shuts_down(&mut forwarder_sched, &mut Never),
            "the renderer forwarder must receive Shutdown on Quit when configured"
        );
        assert!(
            eventually_shuts_down(&mut self_sched, &mut NoopEngine),
            "the engine must send itself Shutdown on Quit when self_handle is configured"
        );
    }
}
