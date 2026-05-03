//! Psychedelic Shader - The PixelFlow Way
//!
//! Original GLSL (shadertoy style):
//! ```glsl
//! vec2 p=(FC.xy*2.-r)/r.y,l,v=p*(1.-(l+=abs(.7-dot(p,p))))/.2;
//! for(float i;i++<8.;o+=(sin(v.xyyx)+1.)*abs(v.x-v.y)*.2)
//!   v+=cos(v.yx*i+vec2(0,i)+t)/i+.7;
//! o=tanh(exp(p.y*vec4(1,-1,-2,0))*exp(-4.*l.x)/o);
//! ```
//!
//! The PixelFlow approach: DON'T translate the loop literally.
//! The GLSL loop is just summing interference at different frequencies.
//! That's algebra, not iteration. Express it as manifold composition.
//!
//! This version uses a single unified kernel for all three color channels,
//! enabling CSE (common subexpression elimination) across the entire expression.

use actor_scheduler::Message;
use pixelflow_core::{Discrete, Field, Manifold};
use pixelflow_compiler::kernel;
use pixelflow_runtime::api::private::EngineData;
use pixelflow_runtime::api::public::{AppData, EngineEvent, EngineEventControl, EngineEventData};
use pixelflow_runtime::platform::ColorCube;
use pixelflow_runtime::{Application, EngineConfig, EngineTroupe, RuntimeError, WindowConfig};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

// ============================================================================
// THE SHADER - Morphism from screen coords to color space
// ============================================================================

// The psychedelic scene - contramap screen coords through color cube.
//
// Computes (red, green, blue) ∈ [0,1]³ from screen position and time,
// then contramaps through ColorCube to get Discrete.
kernel!(pub struct PsychedelicScene = |t: f32, width: f32, height: f32| Field -> Discrete {
    // Screen coordinate remapping
    let scale = 2.0 / height;
    let half_width = width * 0.5;
    let half_height = height * 0.5;
    let x = (X - half_width) * scale;
    let y = (half_height - Y) * scale;

    // Time via W coordinate
    let time = W + t;

    // Radial field
    let r_sq = x * x + y * y;
    let radial = (r_sq - 0.7).abs();

    // Swirl
    let swirl_scale = (1.0 - radial) * 5.0;
    let vx = x * swirl_scale;
    let vy = y * swirl_scale;

    // Time-based values
    let phase = time * 0.5;
    let sin_w03 = (time * 0.3).sin();
    let sin_w20 = (time * 2.0).sin();

    // Swirl computation
    let swirl = ((vx + phase).sin() + 1.0) * ((vx + phase) - (vy + phase * 0.7)).abs() * 0.2 + 0.001;

    // Radial falloff with pulsing
    let pulse = 1.0 + sin_w20 * 0.1;
    let radial_factor = (radial * -4.0 * pulse).exp();

    // Red channel
    let y_factor_r = (y * 1.0 + sin_w03 * 0.2).exp();
    let raw_r = y_factor_r * radial_factor / swirl;
    let soft_r = raw_r / (raw_r.abs() + 1.0);
    let red = (soft_r + 1.0) * 0.5;

    // Green channel
    let y_factor_g = (y * -1.0 + sin_w03 * 0.2).exp();
    let raw_g = y_factor_g * radial_factor / swirl;
    let soft_g = raw_g / (raw_g.abs() + 1.0);
    let green = (soft_g + 1.0) * 0.5;

    // Blue channel
    let y_factor_b = (y * -2.0 + sin_w03 * 0.2).exp();
    let raw_b = y_factor_b * radial_factor / swirl;
    let soft_b = raw_b / (raw_b.abs() + 1.0);
    let blue = (soft_b + 1.0) * 0.5;

    // Contramap through color cube
    ColorCube::default().at(red, green, blue, 1.0)
});

// ============================================================================
// APPLICATION
// ============================================================================

struct PsychedelicApp {
    start: Instant,
    // Mutex satisfies Sync for Arc<dyn Application + Send + Sync>.
    // No contention — only the engine actor thread calls send().
    engine_handle: std::sync::Mutex<pixelflow_runtime::api::private::EngineActorHandle>,
    width: AtomicU32,
    height: AtomicU32,
}

impl Application for PsychedelicApp {
    fn send(&self, event: EngineEvent) -> Result<(), RuntimeError> {
        match event {
            EngineEvent::Data(EngineEventData::RequestFrame { timestamp, .. }) => {
                let t = timestamp.duration_since(self.start).as_secs_f32();
                let width = self.width.load(Ordering::Relaxed);
                let height = self.height.load(Ordering::Relaxed);

                // Build scene - contramap through color cube
                let scene = PsychedelicScene::new(t, width as f32, height as f32);

                let arc: Arc<dyn Manifold<Output = Discrete> + Send + Sync> = Arc::new(scene);

                self.engine_handle
                    .lock()
                    .unwrap()
                    .send(Message::Data(EngineData::FromApp(AppData::RenderSurface(
                        arc,
                    ))))
                    .map_err(|e| RuntimeError::EventSendError(e.to_string()))?;
            }
            EngineEvent::Control(EngineEventControl::Resized {
                width_px,
                height_px,
                ..
            }) => {
                self.width.store(width_px, Ordering::Relaxed);
                self.height.store(height_px, Ordering::Relaxed);
            }
            EngineEvent::Control(EngineEventControl::WindowCreated {
                width_px,
                height_px,
                ..
            }) => {
                self.width.store(width_px, Ordering::Relaxed);
                self.height.store(height_px, Ordering::Relaxed);
            }
            _ => {}
        }
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("Psychedelic Shader (PixelFlow Native)");
    println!("=====================================");
    println!("Resolution: {}x{}", WIDTH, HEIGHT);
    println!();

    let config = EngineConfig {
        window: WindowConfig {
            title: "Psychedelic Shader".to_string(),
            width: WIDTH,
            height: HEIGHT,
        },
        ..Default::default()
    };

    let mut troupe = EngineTroupe::with_config(config)?;
    let unregistered_handle = troupe.engine_handle();
    let start = Instant::now();
    let engine_handle_for_app = troupe.raw_engine_handle();

    let app = PsychedelicApp {
        start,
        engine_handle: std::sync::Mutex::new(engine_handle_for_app),
        width: AtomicU32::new(WIDTH),
        height: AtomicU32::new(HEIGHT),
    };

    use pixelflow_runtime::WindowDescriptor;
    let window = WindowDescriptor {
        width: WIDTH,
        height: HEIGHT,
        title: "Psychedelic Shader".into(),
        resizable: true,
    };
    let _engine_handle = unregistered_handle.register(Arc::new(app), window)?;

    println!("Running... (close window to exit)");
    troupe.play().map_err(|e| anyhow::anyhow!("{}", e))?;

    println!("Done!");
    Ok(())
}
