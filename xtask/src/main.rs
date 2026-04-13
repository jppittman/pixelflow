//! Automation tasks for the project.
//!
//! Commands:
//! - `bundle-run`: Build and run the bundled macOS app
//! - `bake-eigen`: Parse Stam's eigenstructure binary and generate Rust consts

use std::env;
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::Command;

/// Entry point for xtask.
fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: cargo xtask <command>");
        eprintln!("Commands:");
        eprintln!("  bundle-run    Build and run the bundled macOS app");
        eprintln!("  bake-eigen    Parse Stam's eigenstructure binary → Rust consts");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "bundle-run" => {
            // Pass through any additional arguments after "bundle-run"
            let extra_args = if args.len() > 2 { &args[2..] } else { &[] };
            bundle_run(extra_args);
        }
        "bake-eigen" => {
            bake_eigen();
        }
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            std::process::exit(1);
        }
    }
}

/// Find the workspace root by looking for Cargo.toml with [workspace]
fn find_workspace_root() -> PathBuf {
    let mut current = env::current_dir().expect("Failed to get current directory");

    loop {
        let cargo_toml = current.join("Cargo.toml");

        if cargo_toml.exists() {
            // Check if this is the workspace root by reading Cargo.toml
            if let Ok(contents) = fs::read_to_string(&cargo_toml) {
                if contents.contains("[workspace]") {
                    return current;
                }
            }
        }

        // Move up to parent directory
        if !current.pop() {
            eprintln!("Could not find workspace root (no Cargo.toml with [workspace] found)");
            std::process::exit(1);
        }
    }
}

/// Builds the project in release mode and bundles it into a macOS .app structure.
/// Then launches the application.
///
/// # Parameters
/// * `extra_args` - Additional arguments to pass to `cargo build`.
fn bundle_run(extra_args: &[String]) {
    // Find workspace root so this works from any subdirectory
    let workspace_root = find_workspace_root();
    println!("Workspace root: {}", workspace_root.display());

    println!("Building core-term in release mode (opt-level=3, LTO)...");

    // Build the project with extra args (e.g., --features profiling)
    let mut cmd = Command::new("cargo");
    cmd.current_dir(&workspace_root); // Run from workspace root
    cmd.args(["build", "--release", "-p", "core-term"]);

    // Filter out --release since we already added it
    let filtered_args: Vec<&String> = extra_args
        .iter()
        .filter(|arg| arg.as_str() != "--release")
        .collect();

    if !filtered_args.is_empty() {
        println!("Additional build args: {:?}", filtered_args);
        cmd.args(&filtered_args);
    }

    let status = cmd.status().expect("Failed to run cargo build");

    if !status.success() {
        eprintln!("Build failed");
        std::process::exit(1);
    }

    // Copy binary to bundle (build.rs creates the bundle structure)
    let binary_src = workspace_root.join("target/release/core-term");
    let binary_dest = workspace_root.join("CoreTerm.app/Contents/MacOS/CoreTerm");

    if !binary_src.exists() {
        eprintln!("Binary not found at {}", binary_src.display());
        std::process::exit(1);
    }

    // Verify binary size - release with LTO should be reasonably sized
    let binary_size = fs::metadata(&binary_src)
        .expect("Failed to get binary metadata")
        .len();
    println!(
        "Binary size: {:.2} MB (release with LTO)",
        binary_size as f64 / (1024.0 * 1024.0)
    );

    println!("Copying binary to bundle...");
    fs::copy(&binary_src, &binary_dest).expect("Failed to copy binary to bundle");

    // Make it executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&binary_dest)
            .expect("Failed to get binary metadata")
            .permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&binary_dest, perms).expect("Failed to set executable permission");
    }

    // Copy icon file to bundle Resources
    let icon_src = workspace_root.join("assets/icons/icon.icns");
    let resources_dir = workspace_root.join("CoreTerm.app/Contents/Resources");
    let icon_dest = resources_dir.join("icon.icns");

    fs::create_dir_all(&resources_dir).expect("Failed to create Resources directory");

    if icon_src.exists() {
        println!("Copying icon to bundle...");
        fs::copy(&icon_src, &icon_dest).expect("Failed to copy icon to bundle");
    } else {
        println!("Warning: Icon not found at {}", icon_src.display());
    }

    // Copy font file to bundle Resources
    let font_src = workspace_root.join("pixelflow-graphics/assets/NotoSansMono-Regular.ttf");
    let font_dest = resources_dir.join("NotoSansMono-Regular.ttf");

    if font_src.exists() {
        println!("Copying font to bundle...");
        fs::copy(&font_src, &font_dest).expect("Failed to copy font to bundle");
    } else {
        eprintln!("ERROR: Font not found at {}", font_src.display());
        std::process::exit(1);
    }

    // Touch the app bundle to invalidate macOS icon cache
    let app_bundle = workspace_root.join("CoreTerm.app");
    println!("Refreshing app bundle metadata...");
    Command::new("touch")
        .arg(&app_bundle)
        .status()
        .expect("Failed to touch app bundle");

    println!("Launching CoreTerm.app...");
    println!("Logs will be written to /tmp/core-term.log");

    // Launch the bundled app using 'open'
    // Logs are written to /tmp/core-term.log (configured in main.rs)
    let status = Command::new("open")
        .arg(&app_bundle)
        .status()
        .expect("Failed to launch app");

    if !status.success() {
        eprintln!("Failed to launch CoreTerm.app");
        std::process::exit(1);
    }

    println!("CoreTerm.app launched successfully!");
    println!("Monitor logs with: tail -f /tmp/core-term.log");
}

// ============================================================================
// Eigenstructure Baking
// ============================================================================

/// Parse Stam's ccdata50NT.dat and generate Rust const arrays.
///
/// Binary format (little-endian):
/// - Header: i32 Nmax (maximum valence, typically 50)
/// - Per valence N (3..=Nmax):
///   - K = 2N + 8 eigenvalues (f64)
///   - K×K inverse eigenvector matrix (f64, row-major)
///   - 3 sets of K×16 spline coefficients (f64)
fn bake_eigen() {
    let workspace_root = find_workspace_root();
    let input_path = workspace_root.join("pixelflow-graphics/assets/ccdata50NT.dat");
    let output_path = workspace_root.join("pixelflow-graphics/src/subdiv/coeffs.rs");

    println!("Reading eigenstructure from: {}", input_path.display());

    let mut file = fs::File::open(&input_path).expect("Failed to open ccdata50NT.dat");
    let mut data = Vec::new();
    file.read_to_end(&mut data).expect("Failed to read file");

    // Parse header: Nmax as i32 (little-endian)
    let nmax = i32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    println!("Maximum valence: {}", nmax);

    let mut offset = 4; // Skip header

    // Collect all eigenstructures
    let mut structures = Vec::new();

    for valence in 3..=nmax {
        let k = 2 * valence + 8; // Number of eigenvalues/bases

        // Read eigenvalues: K f64s
        let mut eigenvalues = Vec::with_capacity(k);
        for _ in 0..k {
            let val = read_f64_le(&data, offset);
            eigenvalues.push(val as f32);
            offset += 8;
        }

        // Read inverse eigenvector matrix: K×K f64s (row-major)
        let mut inv_eigenvectors = Vec::with_capacity(k * k);
        for _ in 0..(k * k) {
            let val = read_f64_le(&data, offset);
            inv_eigenvectors.push(val as f32);
            offset += 8;
        }

        // Read spline coefficients: 3 subpatches × K bases × 16 coeffs
        let mut spline_coeffs = vec![vec![vec![0.0f32; 16]; k]; 3];
        for subpatch_mut in spline_coeffs.iter_mut() {
            for basis_mut in subpatch_mut.iter_mut() {
                for coeff_ref in basis_mut.iter_mut() {
                    let val = read_f64_le(&data, offset);
                    *coeff_ref = val as f32;
                    offset += 8;
                }
            }
        }

        structures.push(EigenData {
            valence,
            k,
            eigenvalues,
            inv_eigenvectors,
            spline_coeffs,
        });
    }

    println!("Parsed {} valences", structures.len());
    println!("Generating Rust source: {}", output_path.display());

    // Generate Rust source
    let mut out = String::new();
    out.push_str("//! Baked Catmull-Clark eigenstructure coefficients.\n");
    out.push_str("//!\n");
    out.push_str("//! Auto-generated by `cargo xtask bake-eigen` from ccdata50NT.dat.\n");
    out.push_str("//! Do not edit manually.\n");
    out.push_str("//!\n");
    out.push_str("//! Source: Stam, \"Exact Evaluation of Catmull-Clark Subdivision Surfaces\"\n");
    out.push_str(
        "//! Data from: https://www.dgp.toronto.edu/~stam/reality/Research/SubdivEval/\n\n",
    );

    out.push_str("/// Maximum supported valence.\n");
    out.push_str(&format!("pub const MAX_VALENCE: usize = {};\n\n", nmax));

    out.push_str("/// Eigenstructure data for a specific valence.\n");
    out.push_str("#[derive(Clone, Debug)]\n");
    out.push_str("pub struct EigenCoeffs {\n");
    out.push_str("    /// Valence (number of edges at extraordinary vertex)\n");
    out.push_str("    pub valence: usize,\n");
    out.push_str("    /// K = 2N + 8 (number of eigenvalues/bases)\n");
    out.push_str("    pub k: usize,\n");
    out.push_str("    /// Eigenvalues (K values)\n");
    out.push_str("    pub eigenvalues: &'static [f32],\n");
    out.push_str("    /// Inverse eigenvector matrix (K×K, row-major)\n");
    out.push_str("    pub inv_eigenvectors: &'static [f32],\n");
    out.push_str("    /// Spline coefficients [subpatch][basis][coeff] flattened\n");
    out.push_str("    /// Layout: 3 subpatches × K bases × 16 bicubic coeffs\n");
    out.push_str("    pub spline_coeffs: &'static [f32],\n");
    out.push_str("}\n\n");

    out.push_str("impl EigenCoeffs {\n");
    out.push_str("    /// Get spline coefficient for subpatch, basis, and coefficient index.\n");
    out.push_str("    #[inline]\n");
    out.push_str(
        "    pub fn spline(&self, subpatch: usize, basis: usize, coeff: usize) -> f32 {\n",
    );
    out.push_str("        self.spline_coeffs[subpatch * self.k * 16 + basis * 16 + coeff]\n");
    out.push_str("    }\n\n");
    out.push_str("    /// Get inverse eigenvector matrix element.\n");
    out.push_str("    #[inline]\n");
    out.push_str("    pub fn inv_eigen(&self, row: usize, col: usize) -> f32 {\n");
    out.push_str("        self.inv_eigenvectors[row * self.k + col]\n");
    out.push_str("    }\n");
    out.push_str("}\n\n");

    // Generate const arrays for each valence
    for s in &structures {
        let prefix = format!("V{}", s.valence);

        // Eigenvalues
        out.push_str(&format!(
            "const {}_EIGENVALUES: [f32; {}] = {};\n",
            prefix,
            s.k,
            format_f32_array(&s.eigenvalues)
        ));

        // Inverse eigenvectors
        out.push_str(&format!(
            "const {}_INV_EIGEN: [f32; {}] = {};\n",
            prefix,
            s.k * s.k,
            format_f32_array(&s.inv_eigenvectors)
        ));

        // Spline coeffs (flattened: 3 × K × 16)
        let mut flat_splines = Vec::with_capacity(3 * s.k * 16);
        for subpatch in 0..3 {
            for basis in 0..s.k {
                flat_splines.extend_from_slice(&s.spline_coeffs[subpatch][basis]);
            }
        }
        out.push_str(&format!(
            "const {}_SPLINES: [f32; {}] = {};\n\n",
            prefix,
            3 * s.k * 16,
            format_f32_array(&flat_splines)
        ));
    }

    // Generate lookup table
    out.push_str("/// Get eigenstructure for a given valence (3..=50).\n");
    out.push_str("pub fn get_eigen(valence: usize) -> Option<EigenCoeffs> {\n");
    out.push_str("    match valence {\n");
    for s in &structures {
        out.push_str(&format!(
            "        {} => Some(EigenCoeffs {{\n            valence: {},\n            k: {},\n            eigenvalues: &V{}_EIGENVALUES,\n            inv_eigenvectors: &V{}_INV_EIGEN,\n            spline_coeffs: &V{}_SPLINES,\n        }}),\n",
            s.valence, s.valence, s.k, s.valence, s.valence, s.valence
        ));
    }
    out.push_str("        _ => None,\n");
    out.push_str("    }\n");
    out.push_str("}\n");

    // Ensure output directory exists
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).expect("Failed to create subdiv directory");
    }

    let mut out_file = fs::File::create(&output_path).expect("Failed to create output file");
    out_file
        .write_all(out.as_bytes())
        .expect("Failed to write output");

    println!("Generated {} bytes of Rust source", out.len());
    println!("Done! Run `cargo fmt -p pixelflow-graphics` to format.");
}

/// Read f64 little-endian from byte slice.
fn read_f64_le(data: &[u8], offset: usize) -> f64 {
    let bytes: [u8; 8] = data[offset..offset + 8].try_into().unwrap();
    f64::from_le_bytes(bytes)
}

/// Format f32 array as Rust literal.
fn format_f32_array(values: &[f32]) -> String {
    let mut s = String::from("[\n    ");
    for (i, v) in values.iter().enumerate() {
        if i > 0 && i % 8 == 0 {
            s.push_str("\n    ");
        }
        // Use enough precision to round-trip
        s.push_str(&format!("{:e}, ", v));
    }
    s.push_str("\n]");
    s
}

/// Temporary struct for collecting parsed data.
struct EigenData {
    valence: usize,
    k: usize,
    eigenvalues: Vec<f32>,
    inv_eigenvectors: Vec<f32>,
    spline_coeffs: Vec<Vec<Vec<f32>>>,
}
