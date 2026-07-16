// build.rs

/// This build script determines which display driver to use based on:
/// 1. DISPLAY_DRIVER environment variable (highest priority)
/// 2. Enabled Cargo features (display_cocoa, display_x11, display_headless)
/// 3. Target OS defaults (fallback)
///
/// It emits cfg flags for conditional compilation and handles platform-specific setup.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/icons/icon.icns");
    println!("cargo:rerun-if-env-changed=DISPLAY_DRIVER");

    // Declare custom cfg names to avoid warnings
    println!("cargo::rustc-check-cfg=cfg(use_cocoa_display)");
    println!("cargo::rustc-check-cfg=cfg(use_x11_display)");
    println!("cargo::rustc-check-cfg=cfg(use_headless_display)");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS")
        .expect("CARGO_CFG_TARGET_OS is not set, cannot determine target platform.");

    // Determine which display driver to use
    let display_driver = determine_display_driver(&target_os);

    // Emit appropriate cfg flag for the selected display driver
    match display_driver.as_str() {
        "cocoa" => {
            println!("cargo:rustc-cfg=use_cocoa_display");
            println!("cargo:warning=Building with Cocoa display driver");
            // Create macOS app bundle if on macOS
            if target_os == "macos" {
                create_macos_app_bundle();
            }
        }
        "x11" => {
            println!("cargo:rustc-cfg=use_x11_display");
            println!("cargo:warning=Building with X11 display driver");
            // Probe for X11 libraries using pkg-config
            let required_libs = ["x11"];
            for lib in required_libs {
                if let Err(e) = pkg_config::probe_library(lib) {
                    eprintln!("Warning: Failed to find library `{}`: {}", lib, e);
                    eprintln!("X11 display driver may not work correctly.");
                }
            }
        }
        "headless" => {
            println!("cargo:rustc-cfg=use_headless_display");
            println!("cargo:warning=Building with Headless display driver (no GUI)");
        }
        _ => {
            panic!("Unknown display driver: {}", display_driver);
        }
    }
}

/// Determines which display driver to use based on environment and features
fn determine_display_driver(target_os: &str) -> String {
    // 1. Check DISPLAY_DRIVER environment variable (highest priority)
    if let Ok(driver) = std::env::var("DISPLAY_DRIVER") {
        let driver_lower = driver.to_lowercase();
        match driver_lower.as_str() {
            "cocoa" | "x11" | "headless" => {
                println!(
                    "cargo:warning=Using display driver from DISPLAY_DRIVER env: {}",
                    driver_lower
                );
                return driver_lower;
            }
            _ => {
                panic!(
                    "Invalid DISPLAY_DRIVER value: '{}'. Must be one of: cocoa, x11, headless",
                    driver
                );
            }
        }
    }

    // 2. Check which display features are enabled
    let has_cocoa =
        cfg!(feature = "display_cocoa") || std::env::var("CARGO_FEATURE_DISPLAY_COCOA").is_ok();
    let has_x11 =
        cfg!(feature = "display_x11") || std::env::var("CARGO_FEATURE_DISPLAY_X11").is_ok();
    let has_headless = cfg!(feature = "display_headless")
        || std::env::var("CARGO_FEATURE_DISPLAY_HEADLESS").is_ok();

    // If a specific feature is enabled, use it (last one wins if multiple)
    if has_headless && !has_x11 && !has_cocoa {
        return "headless".to_string();
    }
    if has_x11 && !has_headless {
        return "x11".to_string();
    }
    if has_cocoa {
        return "cocoa".to_string();
    }

    // 3. Fall back to sensible platform defaults
    match target_os {
        "macos" => "cocoa".to_string(),
        "linux" => "x11".to_string(),
        _ => "headless".to_string(),
    }
}

/// Walk up from the crate manifest directory to find the workspace root
/// (the ancestor directory whose Cargo.toml contains a `[workspace]` table).
/// Mirrors `xtask`'s `find_workspace_root`, since the bundle must land in
/// the same place `xtask bundle-run` expects it.
#[cfg(target_os = "macos")]
fn find_workspace_root(manifest_dir: &std::path::Path) -> std::path::PathBuf {
    let mut current = manifest_dir.to_path_buf();
    loop {
        let cargo_toml = current.join("Cargo.toml");
        if let Ok(contents) = std::fs::read_to_string(&cargo_toml) {
            if contents.contains("[workspace]") {
                return current;
            }
        }
        if !current.pop() {
            panic!(
                "Could not find workspace root above {}",
                manifest_dir.display()
            );
        }
    }
}

#[cfg(target_os = "macos")]
fn create_macos_app_bundle() {
    use std::fs;
    use std::io::Write;
    use std::path::Path;

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let workspace_root = find_workspace_root(Path::new(&manifest_dir));
    let bundle_dir = workspace_root.join("CoreTerm.app/Contents");

    // Create bundle structure (cleanup old bundle if it exists - failure is OK)
    fs::remove_dir_all(workspace_root.join("CoreTerm.app")).ok();
    fs::create_dir_all(bundle_dir.join("MacOS")).expect("Failed to create MacOS directory");
    fs::create_dir_all(bundle_dir.join("Resources")).expect("Failed to create Resources directory");

    // Create Info.plist (required for keyboard input to work on macOS)
    let plist_content = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>CoreTerm</string>
    <key>CFBundleIdentifier</key>
    <string>com.core-term.terminal</string>
    <key>CFBundleName</key>
    <string>CoreTerm</string>
    <key>CFBundleDisplayName</key>
    <string>CoreTerm</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>0.1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSUIElement</key>
    <false/>
</dict>
</plist>
"#;

    let plist_path = bundle_dir.join("Info.plist");
    let mut plist_file = fs::File::create(&plist_path).expect("Failed to create Info.plist");
    plist_file
        .write_all(plist_content.as_bytes())
        .expect("Failed to write Info.plist");
}

#[cfg(not(target_os = "macos"))]
fn create_macos_app_bundle() {
    // No-op on non-macOS platforms
}
