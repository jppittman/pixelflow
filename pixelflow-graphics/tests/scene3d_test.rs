//! The scene contracts: colour, reflection and antialiasing, through the
//! packed lane — four channel kernels compiled at the frame's shape.
//!
//! Named in `.github/workflows/rust.yaml` ("Check scene color, reflection,
//! and antialiasing contracts"), so a rename here is a red check there.

mod common;

use pixelflow_core::Kernel;
use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::scene::{compile_packed_for, Scene};
use pixelflow_graphics::scene3d::{checker, sky, Hit, Plane, Ray, Rgba, Sphere};

fn k(v: f32) -> Kernel {
    Kernel::constant(v)
}

/// The checker floor under the sky, seen along `ray`, filtered over one pixel.
fn world(ray: &Ray, floor_height: f32) -> Rgba {
    let floor = Plane::at_height(k(floor_height)).hit(ray);
    floor.select(
        &checker(&floor.point()[0], &floor.point()[2], &floor.footprint()),
        &sky(ray),
    )
}

fn unit_sphere(ray: &Ray) -> Hit {
    Sphere::new([k(0.0), k(0.0), k(4.0)], k(1.0)).hit(ray)
}

/// A chrome sphere at (0, 0, 4) over a checker floor at `y = -1`, reflecting
/// the floor and the sky.
fn chrome_color(width: usize, height: usize) -> Rgba {
    let ray = Ray::through_screen(width as f32, height as f32);
    let sphere = unit_sphere(&ray);
    let mirrored = ray.reflected(sphere.normal());
    sphere.select(&world(&mirrored, -1.0), &world(&ray, -1.0))
}

fn render(color: &Rgba, width: usize, height: usize) -> Frame<Rgba8> {
    let program = compile_packed_for::<Rgba8>(color, [width as u32, height as u32]);
    let mut frame = Frame::<Rgba8>::new(width as u32, height as u32);
    Scene::Packed(program.bind(&[])).render(&mut frame, 1);
    frame
}

/// A chrome sphere reflecting a checker floor and a blue sky, at the frame
/// size the throughput number is quoted for. No golden: 1920x1080 would be a
/// 6 MB binary for a test whose point is the rate, and `chrome_unit_sphere`
/// pins the same code path against a golden at 400x300.
#[test]
fn color_chrome_sphere() {
    const W: usize = 1920;
    const H: usize = 1080;

    let color = chrome_color(W, H);
    let program = compile_packed_for::<Rgba8>(&color, [W as u32, H as u32]);
    let scene = Scene::Packed(program.bind(&[]));
    let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
    let start = std::time::Instant::now();
    scene.render(&mut frame, 1);
    let elapsed = start.elapsed();
    println!(
        "packed chrome sphere: {elapsed:?} ({:.2} Mpix/s)",
        (W * H) as f64 / elapsed.as_secs_f64() / 1_000_000.0
    );

    let path = std::env::temp_dir().join("pixelflow_color_chrome.ppm");
    common::write_ppm(&path, &frame).unwrap();
    println!("Saved: {}", path.display());

    let center = &frame.data[(H / 2) * W + (W / 2)];
    let sky = &frame.data[0];
    assert!(
        center.r() > 10 || center.g() > 10 || center.b() > 10,
        "the sphere reflects something: r={} g={} b={}",
        center.r(),
        center.g(),
        center.b()
    );
    assert!(
        sky.b() > sky.r(),
        "sky should be blue: r={} b={}",
        sky.r(),
        sky.b()
    );
}

/// The floor and the sky, with no sphere: the checker's antialiasing on its
/// own, over a golden.
#[test]
fn floor_only() {
    const W: usize = 400;
    const H: usize = 300;

    let ray = Ray::through_screen(W as f32, H as f32);
    let frame = render(&world(&ray, -1.0), W, H);

    let path = std::env::temp_dir().join("pixelflow_floor_only.ppm");
    common::write_ppm(&path, &frame).unwrap();
    common::assert_golden("floor_only", &frame, 2, 0.01);

    // Below the horizon is a checkerboard and above it is a smooth sky, which
    // a single pixel cannot tell you — a floor pixel may legitimately be any
    // blend of two cells. A row can: the floor's swings from cell to cell, the
    // sky's barely moves.
    let spread = |row: usize| {
        let r = |x: usize| frame.data[row * W + x].r();
        let (lo, hi) = (0..W)
            .map(r)
            .fold((255u8, 0u8), |(lo, hi), v| (lo.min(v), hi.max(v)));
        hi - lo
    };
    assert!(
        spread(3 * H / 4) > 100,
        "the floor is a checkerboard: its row spans {} of 255",
        spread(3 * H / 4)
    );
    // The sky is not flat across a row either — normalizing the direction
    // tilts a row's rays down towards its ends — but it moves by a tenth of
    // what a cell boundary does.
    assert!(
        spread(H / 4) < 20,
        "the sky is smooth across the frame: its row spans {} of 255",
        spread(H / 4)
    );
    let sky = &frame.data[(H / 4) * W + (W / 2)];
    assert!(
        sky.b() > sky.r(),
        "above the horizon is sky: r={} b={}",
        sky.r(),
        sky.b()
    );
}

/// **The mullet contract, restated for the compiler.**
///
/// The chrome sphere against a golden, small enough to keep in the tree.
#[test]
fn chrome_unit_sphere() {
    const W: usize = 400;
    const H: usize = 300;

    let frame = render(&chrome_color(W, H), W, H);
    let path = std::env::temp_dir().join("pixelflow_chrome_unit_sphere.ppm");
    common::write_ppm(&path, &frame).unwrap();
    common::assert_golden("chrome_unit_sphere", &frame, 2, 0.01);

    let center = &frame.data[(H / 2) * W + (W / 2)];
    let corner = &frame.data[0];
    assert!(center.r() > 10, "the sphere is not black: r={}", center.r());
    assert!(corner.r() > 20, "the corner is sky: r={}", corner.r());
}

/// The sky alone: a gradient in the ray's elevation, no geometry.
#[test]
fn sky_only() {
    const W: usize = 200;
    const H: usize = 150;

    let ray = Ray::through_screen(W as f32, H as f32);
    let frame = render(&sky(&ray), W, H);
    let path = std::env::temp_dir().join("pixelflow_sky_only.ppm");
    common::write_ppm(&path, &frame).unwrap();
    common::assert_golden("sky_only", &frame, 2, 0.0);

    // Looking up is darker blue, looking down is paler: the gradient runs the
    // way the sky's `t` says it does.
    let top = &frame.data[(H / 4) * W + (W / 2)];
    let bottom = &frame.data[(3 * H / 4) * W + (W / 2)];
    assert!(
        top.r() < bottom.r(),
        "sky gradient runs down the frame: top r={} bottom r={}",
        top.r(),
        bottom.r()
    );
}

/// Which worker draws which stripe is a scheduling decision, and no
/// scheduling decision may be visible in the pixels.
#[test]
fn work_stealing_matches_single_threaded() {
    const W: usize = 64;
    const H: usize = 64;

    let program = compile_packed_for::<Rgba8>(&chrome_color(W, H), [W as u32, H as u32]);
    let scene = Scene::Packed(program.bind(&[]));
    let mut one = Frame::<Rgba8>::new(W as u32, H as u32);
    let mut many = Frame::<Rgba8>::new(W as u32, H as u32);
    scene.render(&mut one, 1);
    scene.render(&mut many, 12);
    assert_eq!(one.data, many.data, "thread count must not change pixels");
}
