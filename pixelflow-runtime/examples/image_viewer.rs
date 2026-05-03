//! Simple Image Viewer - displays the chrome sphere in a window.
//!
//! This demonstrates the minimal pixelflow-runtime usage:
//! 1. Create EngineTroupe with config
//! 2. Get engine handle
//! 3. Send a manifold to render
//! 4. Run the event loop

use pixelflow_core::combinators::At;
use pixelflow_core::jet::Jet3;
use pixelflow_core::{Discrete, Field, Manifold, ManifoldCompat};
use pixelflow_compiler::ManifoldExpr;
use pixelflow_runtime::{api::public::AppData, EngineConfig, EngineTroupe, WindowConfig};
use std::sync::Arc;

type Field4 = (Field, Field, Field, Field);
type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);

const W: u32 = 1920;
const H: u32 = 1080;

// Import scene3d types
use pixelflow_graphics::scene3d::{
    plane, ColorChecker, ColorReflect, ColorScreenToDir, ColorSky, ColorSurface,
};
use pixelflow_runtime::platform::ColorCube;

/// Sphere at given center with radius (local to this example).
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

        let epsilon_sq = Jet3::constant(Field::from(0.0001));
        d_dot_c - (discriminant + epsilon_sq).sqrt()
    }
}

/// Screen coordinate remapper for Discrete output.
#[derive(Clone, Copy, ManifoldExpr)]
struct ScreenRemap<M> {
    inner: M,
    width: f32,
    height: f32,
}

impl<M: ManifoldCompat<Field, Output = Discrete>> Manifold<Field4> for ScreenRemap<M> {
    type Output = Discrete;

    fn eval(&self, p: Field4) -> Discrete {
        let (x, y, z, w) = p;
        let scale = 2.0 / self.height;
        let sx = (x - Field::from(self.width * 0.5)) * Field::from(scale);
        let sy = (Field::from(self.height * 0.5) - y) * Field::from(scale);
        At {
            inner: &self.inner,
            x: sx,
            y: sy,
            z,
            w,
        }
        .collapse()
    }
}

/// Build the chrome sphere scene.
fn build_scene() -> impl Manifold<Output = Discrete> + Send + Sync + Clone {
    let color_cube = ColorCube::default();
    let world = ColorSurface {
        geometry: plane(-1.0),
        material: ColorChecker::new(color_cube.clone()),
        background: ColorSky::new(color_cube),
    };

    let scene = ColorSurface {
        geometry: SphereAt {
            center: (0.0, 0.0, 4.0),
            radius: 1.0,
        },
        material: ColorReflect {
            inner: world.clone(),
        },
        background: world,
    };

    ScreenRemap {
        inner: ColorScreenToDir { inner: scene },
        width: W as f32,
        height: H as f32,
    }
}

fn main() -> anyhow::Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("Chrome Sphere + Bezier Patch");
    println!("=============================");
    println!("Resolution: {}x{}", W, H);
    println!();

    // Configure the engine
    let config = EngineConfig {
        window: WindowConfig {
            title: "Chrome Sphere + Bezier Patch".to_string(),
            width: W,
            height: H,
        },
        ..Default::default()
    };

    // Phase 1: Create the troupe
    let mut troupe = EngineTroupe::with_config(config)?;

    // Phase 2: Get unregistered engine handle
    let unregistered_handle = troupe.engine_handle();

    // Create a dummy app that ignores events
    struct DummyApp;
    impl pixelflow_runtime::Application for DummyApp {
        fn send(
            &self,
            _event: pixelflow_runtime::EngineEvent,
        ) -> Result<(), pixelflow_runtime::RuntimeError> {
            Ok(())
        }
    }

    // Register app and create window
    use pixelflow_runtime::WindowDescriptor;
    let window = WindowDescriptor {
        width: W,
        height: H,
        title: "Image Viewer".into(),
        resizable: false,
    };
    let engine_handle = unregistered_handle.register(Arc::new(DummyApp), window)?;

    // Build our scene manifold
    let scene = build_scene();
    let scene_arc: Arc<dyn Manifold<Output = Discrete> + Send + Sync> = Arc::new(scene);

    // Send initial frame
    use actor_scheduler::Message;
    use pixelflow_runtime::api::private::EngineData;

    engine_handle
        .send(Message::Data(EngineData::FromApp(AppData::RenderSurface(
            scene_arc.clone(),
        ))))
        .map_err(|e| anyhow::anyhow!("Failed to send initial frame: {}", e))?;

    println!("Sent initial frame to engine");
    println!("Running event loop... (close window to exit)");

    // Phase 3: Run the event loop (blocks)
    troupe.play().map_err(|e| anyhow::anyhow!("{}", e))?;

    println!("Exited cleanly.");
    Ok(())
}
