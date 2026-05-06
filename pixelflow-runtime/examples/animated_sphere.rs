//! Animated Chrome Sphere - windowed animation using the runtime.
//!
//! Demonstrates the **pull-based rendering model**:
//! - Engine sends `RequestFrame` events when ready for a new frame
//! - App responds with a manifold computed at the requested timestamp
//! - No busy loops, no sleeps - vsync drives the cadence
//!
//! Animation approach:
//! - Each frame, a new scene is built with the sphere at the animated position
//! - The position is computed using sin(t * freq) * amplitude at the app level
//!
//! Resize handling:
//! - App receives Resized events and updates stored dimensions
//! - Scene is rebuilt with new dimensions on next frame

use actor_scheduler::Message;
use pixelflow_core::combinators::At;
use pixelflow_core::jet::Jet3;
use pixelflow_core::{Discrete, Field, Manifold, ManifoldCompat};
use pixelflow_graphics::scene3d::{
    plane, ColorChecker, ColorReflect, ColorScreenToDir, ColorSky, ColorSurface,
};
use pixelflow_compiler::ManifoldExpr;
use pixelflow_runtime::api::private::EngineData;
use pixelflow_runtime::api::public::{AppData, EngineEvent, EngineEventControl, EngineEventData};
use pixelflow_runtime::platform::ColorCube;
use pixelflow_runtime::{Application, EngineConfig, EngineTroupe, RuntimeError, WindowConfig};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

type Field4 = (Field, Field, Field, Field);
type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);

// ============================================================================
// GEOMETRY PRIMITIVES (manual struct like chrome_sphere)
// ============================================================================

/// Sphere at given center with radius.
#[derive(Clone, Copy, ManifoldExpr)]
struct SphereAt {
    center: (f32, f32, f32),
    radius: f32,
}

impl Manifold<Jet3_4> for SphereAt {
    type Output = Jet3;

    #[inline]
    fn eval(&self, p: Jet3_4) -> Jet3 {
        let (rx, ry, rz, _w) = p;
        let cx = Jet3::constant(Field::from(self.center.0));
        let cy = Jet3::constant(Field::from(self.center.1));
        let cz = Jet3::constant(Field::from(self.center.2));

        let d_dot_c = rx * cx + ry * cy + rz * cz;
        let c_sq = cx * cx + cy * cy + cz * cz;
        let r_sq = Jet3::constant(Field::from(self.radius * self.radius));
        let discriminant = d_dot_c * d_dot_c - (c_sq - r_sq);

        // Smooth fix for grazing angles
        let epsilon_sq = Jet3::constant(Field::from(0.0001));
        d_dot_c - (discriminant + epsilon_sq).sqrt()
    }
}

// ============================================================================
// SCREEN COORDINATE REMAPPING
// ============================================================================

/// Remap pixel coordinates to normalized screen coordinates for ~60° FOV.
#[derive(Clone, ManifoldExpr)]
struct ColorScreenRemap<M> {
    inner: M,
    width: f32,
    height: f32,
}

impl<M: ManifoldCompat<Field, Output = Discrete>> Manifold<Field4> for ColorScreenRemap<M> {
    type Output = Discrete;

    fn eval(&self, p: Field4) -> Discrete {
        let (px, py, z, w) = p;
        let width = Field::from(self.width);
        let height = Field::from(self.height);

        let scale = Field::from(2.0) / height;
        let x = (px - width * Field::from(0.5)) * scale.clone();
        let y = (height * Field::from(0.5) - py) * scale;

        At {
            inner: &self.inner,
            x,
            y,
            z,
            w,
        }
        .collapse()
    }
}

// ============================================================================
// SCENE CONSTRUCTION - Compositional Animation
// ============================================================================

/// Animation parameters for the oscillating sphere.
const BASE_CENTER: (f32, f32, f32) = (0.0, 0.0, 4.0);
const AMPLITUDE: f32 = 2.0; // Oscillate ±2 units in X
const FREQUENCY: f32 = 1.0; // 1 rad/s
const RADIUS: f32 = 1.0;

/// Build scene with sphere at the given animated position.
///
/// The animation offset is precomputed at the application level using
/// sin(t * frequency) * amplitude, then baked into the sphere's center.
fn build_scene_at_time(
    t: f32,
    width: u32,
    height: u32,
) -> impl Manifold<Output = Discrete> + Clone + Sync + Send {
    // Compute the animated X offset
    let x_offset = (t * FREQUENCY).sin() * AMPLITUDE;
    let cx = BASE_CENTER.0 + x_offset;
    let cy = BASE_CENTER.1;
    let cz = BASE_CENTER.2;

    // Background: floor with checkerboard
    let color_cube = ColorCube::default();
    let world = ColorSurface {
        geometry: plane(-1.0),
        material: ColorChecker::new(color_cube.clone()),
        background: ColorSky::new(color_cube),
    };

    // Sphere at the computed animated position
    let sphere = SphereAt {
        center: (cx, cy, cz),
        radius: RADIUS,
    };

    let scene = ColorSurface {
        geometry: sphere,
        material: ColorReflect {
            inner: world.clone(),
        },
        background: world,
    };

    ColorScreenRemap {
        inner: ColorScreenToDir { inner: scene },
        width: width as f32,
        height: height as f32,
    }
}

// ============================================================================
// APPLICATION - Pull-based rendering
// ============================================================================

/// The animated sphere application.
///
/// Implements the pull-based rendering model:
/// - Receives `RequestFrame` events from the engine
/// - Responds with a manifold at the requested timestamp
/// - Handles resize events to update dimensions
struct AnimatedSphereApp {
    /// Animation start time
    start: Instant,
    /// Handle to send frames back to the engine.
    /// Mutex satisfies Sync for Arc<dyn Application + Send + Sync>.
    /// No contention — only the engine actor thread calls send().
    engine_handle: std::sync::Mutex<pixelflow_runtime::api::private::EngineActorHandle>,
    /// Current width (atomic for interior mutability)
    width: AtomicU32,
    /// Current height (atomic for interior mutability)
    height: AtomicU32,
}

impl Application for AnimatedSphereApp {
    fn send(&self, event: EngineEvent) -> Result<(), RuntimeError> {
        match event {
            // Engine is ready for a frame - this is the pull!
            EngineEvent::Data(EngineEventData::RequestFrame { timestamp, .. }) => {
                log::debug!("App received RequestFrame");

                // Compute elapsed time from animation start
                let t = timestamp.duration_since(self.start).as_secs_f32();

                // Get current dimensions
                let width = self.width.load(Ordering::Relaxed);
                let height = self.height.load(Ordering::Relaxed);

                // Build the scene at this moment in time with current dimensions
                let scene = build_scene_at_time(t, width, height);

                let arc: Arc<dyn Manifold<Output = Discrete> + Send + Sync> = Arc::new(scene);

                // Send the frame back to the engine
                log::debug!("App sending RenderSurface");
                self.engine_handle
                    .lock()
                    .unwrap()
                    .send(Message::Data(EngineData::FromApp(AppData::RenderSurface(
                        arc,
                    ))))
                    .map_err(|e| RuntimeError::EventSendError(e.to_string()))?;
            }
            // Handle resize events
            EngineEvent::Control(EngineEventControl::Resized {
                width_px,
                height_px,
                ..
            }) => {
                log::info!("App: Window resized to {}x{}", width_px, height_px);
                self.width.store(width_px, Ordering::Relaxed);
                self.height.store(height_px, Ordering::Relaxed);
            }
            EngineEvent::Control(EngineEventControl::WindowCreated {
                width_px,
                height_px,
                ..
            }) => {
                log::info!("App: Window created {}x{}", width_px, height_px);
                self.width.store(width_px, Ordering::Relaxed);
                self.height.store(height_px, Ordering::Relaxed);
            }
            EngineEvent::Control(ctrl) => {
                log::debug!("App received Control event: {:?}", ctrl);
            }
            _ => {
                log::debug!("App received other event");
            }
        }
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("Animated Chrome Sphere (Pull-based)");
    println!("====================================");
    println!("Resolution: {}x{}", WIDTH, HEIGHT);
    println!();

    let config = EngineConfig {
        window: WindowConfig {
            title: "Animated Sphere".to_string(),
            width: WIDTH,
            height: HEIGHT,
        },
        ..Default::default()
    };

    let mut troupe = EngineTroupe::with_config(config)?;
    let unregistered_handle = troupe.engine_handle();
    let start = Instant::now();

    // Get the raw engine handle for sending frames back
    // This must be obtained before registration (app needs it to respond to RequestFrame)
    let engine_handle_for_app = troupe.raw_engine_handle();

    // Create the pull-based app (scene is built per-frame with animation)
    let app = AnimatedSphereApp {
        start,
        engine_handle: std::sync::Mutex::new(engine_handle_for_app),
        width: AtomicU32::new(WIDTH),
        height: AtomicU32::new(HEIGHT),
    };

    // Register app and create window
    use pixelflow_runtime::WindowDescriptor;
    let window = WindowDescriptor {
        width: WIDTH,
        height: HEIGHT,
        title: "Animated Sphere".into(),
        resizable: true,
    };
    let _engine_handle = unregistered_handle.register(Arc::new(app), window)?;

    println!("Running... (close window to exit)");
    troupe.play().map_err(|e| anyhow::anyhow!("{}", e))?;

    println!("Done!");
    Ok(())
}
