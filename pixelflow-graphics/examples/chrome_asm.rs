//! What the chrome sphere actually compiles to.
//!
//! `cargo run --release -p pixelflow-graphics --example chrome_asm`
//!
//! The scene is four channel kernels compiled by the JIT, so there is no
//! Rust symbol for `cargo-asm` to look at any more: the code is emitted at
//! run time. This dumps it — size, and the raw bytes to a file — so it can be
//! disassembled directly:
//!
//! ```text
//! objdump -D -b binary -m i386:x86-64 -M intel /tmp/chrome_sphere.bin
//! ```
//!
//! The sizes it prints are also the cheapest look at what the compiler shares:
//! four channels over one geometry emit barely more code than one channel
//! does (the contract `scene3d_test::four_channels_share_one_geometry` pins).

use std::io::Write;

use pixelflow_core::Kernel;
use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::scene::compile_packed_for;
use pixelflow_graphics::scene3d::{checker, sky, Plane, Ray, Rgba, Sphere};

const W: u32 = 1920;
const H: u32 = 1080;

fn k(v: f32) -> Kernel {
    Kernel::constant(v)
}

fn world(ray: &Ray) -> Rgba {
    let floor = Plane::at_height(k(-1.0)).hit(ray);
    floor.select(
        &checker(&floor.point()[0], &floor.point()[2], &floor.footprint()),
        &sky(ray),
    )
}

fn chrome() -> Rgba {
    let ray = Ray::through_screen(W as f32, H as f32);
    let sphere = Sphere::new([k(0.0), k(0.0), k(4.0)], k(1.0)).hit(&ray);
    let mirrored = ray.reflected(sphere.normal());
    sphere.select(&world(&mirrored), &world(&ray))
}

fn main() {
    let color = chrome();
    let nodes: usize = color.fold(
        &|channels: &[Kernel; 4]| {
            channels
                .iter()
                .map(|c| pixelflow_ir::node_count_subtree(c.term().root()))
                .sum()
        },
        &|mask: &Kernel, if_true: usize, if_false: usize| {
            pixelflow_ir::node_count_subtree(mask.term().root()) + if_true + if_false
        },
    );

    let start = std::time::Instant::now();
    let program = compile_packed_for::<Rgba8>(&color, [W, H]);
    let compile = start.elapsed();
    let code = program.code_bytes();

    let one = compile_packed_for::<Rgba8>(
        &color.map_channels(&|ch| [ch[0].clone(), k(0.0), k(0.0), k(1.0)]),
        [W, H],
    );

    println!("chrome sphere at {W}x{H}");
    println!("  {nodes} arena nodes over four channels, before optimization");
    println!("  compiled in {compile:?}");
    println!("  {} bytes of code for four channels", code.len());
    println!(
        "  {} bytes for one channel ({:.2}x)",
        one.code_bytes().len(),
        code.len() as f64 / one.code_bytes().len() as f64
    );

    let path = std::env::temp_dir().join("chrome_sphere.bin");
    let mut file = std::fs::File::create(&path).expect("cannot write the code dump");
    file.write_all(code).expect("cannot write the code dump");
    println!("  wrote {}", path.display());
    println!(
        "  objdump -D -b binary -m i386:x86-64 -M intel {}",
        path.display()
    );
}
