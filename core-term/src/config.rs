// src/config.rs

//! Defines configuration structures and provides global access to the loaded configuration.
//!
//! The application's configuration is loaded once and made available globally
//! via a lazily initialized static variable `CONFIG`.

// --- Crates and Modules ---
use crate::{
    color::{Color, NamedColor},
    keys::{KeySymbol, Modifiers},
    term::action::UserInputAction,
    term::cursor::CursorShape, // Assumes CursorShape is in `crate::term::modes`
};
use log::{error, info};
use pixelflow_runtime::config::PerformanceConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::LazyLock;

// --- Global Configuration Access ---

/// Lazily initialized global static storage for the application's configuration.
pub static CONFIG: LazyLock<Config> = LazyLock::new(|| {
    // This closure is executed once.
    // It attempts to load the configuration and falls back to defaults if necessary.
    match load_config_from_file_or_defaults() {
        Ok(cfg) => {
            info!("Configuration loaded successfully (or defaults used).");
            cfg
        }
        Err(e) => {
            // The terminal still starts, on defaults, but says loudly why
            // the user's settings are not in effect.
            error!(
                "Critical error during configuration loading: {:?}. Using emergency default configuration.",
                e
            );
            Config::default()
        }
    }
});

/// The configuration file's name inside the configuration directory.
const CONFIG_FILE: &str = "core-term/config.json";

/// Where the configuration file lives: `$XDG_CONFIG_HOME/core-term/config.json`,
/// else `~/.config/core-term/config.json`. `None` when neither variable is set.
fn config_path() -> Option<PathBuf> {
    let dir = std::env::var_os("XDG_CONFIG_HOME")
        .filter(|dir| !dir.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    Some(dir.join(CONFIG_FILE))
}

/// Loads the configuration file, or the defaults when there is none.
///
/// Every setting is optional: a file names only what it changes. A file that
/// exists but cannot be read or parsed is an error rather than a silent
/// fall-back, so a typo is reported instead of quietly ignored.
fn load_config_from_file_or_defaults() -> anyhow::Result<Config> {
    let Some(path) = config_path() else {
        info!("No home directory to look for a configuration file in; using defaults.");
        return Ok(Config::default());
    };
    load_config_from(&path)
}

fn load_config_from(path: &std::path::Path) -> anyhow::Result<Config> {
    use anyhow::Context;
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            info!(
                "No configuration file at {}; using defaults.",
                path.display()
            );
            return Ok(Config::default());
        }
        Err(e) => {
            return Err(e).with_context(|| format!("Failed to read {}", path.display()));
        }
    };
    let config = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse {}", path.display()))?;
    info!("Configuration loaded from {}.", path.display());
    Ok(config)
}

// --- Configuration Structures ---

/// Defines a single keybinding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Keybinding {
    pub key: KeySymbol,
    pub mods: Modifiers,
    pub action: UserInputAction,
}

/// Raw configuration structure for deserialization/serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawKeybindingsConfig {
    pub bindings: Vec<Keybinding>,
}

/// Defines the configuration for all keybindings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "RawKeybindingsConfig", into = "RawKeybindingsConfig")]
pub struct KeybindingsConfig {
    pub bindings: Vec<Keybinding>,
    #[serde(skip)]
    pub lookup: HashMap<(KeySymbol, Modifiers), UserInputAction>,
}

impl From<RawKeybindingsConfig> for KeybindingsConfig {
    fn from(raw: RawKeybindingsConfig) -> Self {
        let mut lookup = HashMap::new();
        for binding in &raw.bindings {
            // First match wins, so we use entry(...).or_insert(...) to only insert if not present
            lookup
                .entry(crate::keys::chord(binding.key, binding.mods))
                .or_insert_with(|| binding.action.clone());
        }
        KeybindingsConfig {
            bindings: raw.bindings,
            lookup,
        }
    }
}

impl From<KeybindingsConfig> for RawKeybindingsConfig {
    fn from(config: KeybindingsConfig) -> Self {
        RawKeybindingsConfig {
            bindings: config.bindings,
        }
    }
}

impl Default for KeybindingsConfig {
    fn default() -> Self {
        let raw = RawKeybindingsConfig {
            bindings: vec![
                Keybinding {
                    key: KeySymbol::Char('c'),
                    mods: Modifiers::CONTROL | Modifiers::SHIFT,
                    action: UserInputAction::InitiateCopy,
                },
                Keybinding {
                    key: KeySymbol::Char('v'),
                    mods: Modifiers::CONTROL | Modifiers::SHIFT,
                    action: UserInputAction::RequestClipboardPaste,
                },
                // Paste the primary selection (xterm).
                Keybinding {
                    key: KeySymbol::Insert,
                    mods: Modifiers::SHIFT,
                    action: UserInputAction::RequestPrimaryPaste,
                },
                Keybinding {
                    key: KeySymbol::F11,
                    mods: Modifiers::empty(),
                    action: UserInputAction::RequestToggleFullscreen,
                },
                // Zoom. X11 reports the shifted keysym and macOS the key, so
                // each chord is listed in both forms (US layout).
                Keybinding {
                    key: KeySymbol::Char('+'),
                    mods: Modifiers::CONTROL | Modifiers::SHIFT,
                    action: UserInputAction::RequestZoomIn,
                },
                Keybinding {
                    key: KeySymbol::Char('='),
                    mods: Modifiers::CONTROL | Modifiers::SHIFT,
                    action: UserInputAction::RequestZoomIn,
                },
                Keybinding {
                    key: KeySymbol::Char('_'),
                    mods: Modifiers::CONTROL | Modifiers::SHIFT,
                    action: UserInputAction::RequestZoomOut,
                },
                Keybinding {
                    key: KeySymbol::Char('-'),
                    mods: Modifiers::CONTROL | Modifiers::SHIFT,
                    action: UserInputAction::RequestZoomOut,
                },
                Keybinding {
                    key: KeySymbol::Char(')'),
                    mods: Modifiers::CONTROL | Modifiers::SHIFT,
                    action: UserInputAction::RequestZoomReset,
                },
                Keybinding {
                    key: KeySymbol::Char('0'),
                    mods: Modifiers::CONTROL | Modifiers::SHIFT,
                    action: UserInputAction::RequestZoomReset,
                },
            ],
        };
        KeybindingsConfig::from(raw)
    }
}

/// Represents the complete configuration for the terminal emulator.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct Config {
    pub appearance: AppearanceConfig,
    pub behavior: BehaviorConfig,
    pub performance: PerformanceConfig,
    pub colors: ColorScheme,
    pub shell: ShellConfig,
    pub mouse: MouseConfig,
    pub keybindings: KeybindingsConfig,
}

/// Defines settings related to the visual appearance of the terminal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppearanceConfig {
    pub font: FontConfig,
    pub columns: u16,
    pub rows: u16,
    pub border_pixels: u16,
    pub cursor: CursorConfig,
    pub unfocused_cursor: CursorConfig,
    pub default_title: String,
    pub cell_width_px: usize,
    pub cell_height_px: usize,
}

impl Default for AppearanceConfig {
    fn default() -> Self {
        AppearanceConfig {
            font: FontConfig::default(),
            columns: 80,
            rows: 24,
            border_pixels: 2,
            cursor: CursorConfig::default(),
            unfocused_cursor: CursorConfig {
                shape: CursorShape::SteadyBar,
                blink_timeout_ms: 0,
                thickness: 2,
            },
            default_title: "core-term".to_string(),
            cell_width_px: 10,
            cell_height_px: 16,
        }
    }
}

/// Font configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FontConfig {
    pub normal: String,
    pub bold: String,
    pub italic: String,
    pub bold_italic: String,
    pub size_pt: f64,
    pub cw_scale: f32,
    pub ch_scale: f32,
}

impl Default for FontConfig {
    fn default() -> Self {
        let normal = "Noto Sans Mono:pixelsize=12:antialias=true:autohint=true".to_string();

        FontConfig {
            normal: normal.clone(),
            bold: format!("{}:style=Bold", normal),
            italic: format!("{}:style=Italic", normal),
            bold_italic: format!("{}:style=Bold Italic", normal),
            size_pt: 16.0, // Match cell height for proper scaling
            cw_scale: 1.0,
            ch_scale: 1.0,
        }
    }
}

/// Cursor appearance settings.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct CursorConfig {
    pub shape: CursorShape,
    pub thickness: u16,
    pub blink_timeout_ms: u32,
}

impl Default for CursorConfig {
    fn default() -> Self {
        CursorConfig {
            shape: CursorShape::SteadyBlock,
            thickness: 2,
            blink_timeout_ms: 800,
        }
    }
}

/// Defines settings related to the operational behavior of the terminal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BehaviorConfig {
    pub scrollback_lines: usize,
    pub tabspaces: u8,
    pub word_delimiters: String,
    pub double_click_timeout_ms: u32,
    pub triple_click_timeout_ms: u32,
    pub bell_volume: i8,
    pub term_env_var: String,
    pub allow_alt_screen: bool,
    pub allow_window_ops: bool,
    pub default_origin_mode: bool,
}

impl Default for BehaviorConfig {
    fn default() -> Self {
        BehaviorConfig {
            scrollback_lines: 1000,
            tabspaces: 8,
            word_delimiters: " `\"'()[]{}<>".to_string(),
            double_click_timeout_ms: 300,
            triple_click_timeout_ms: 600,
            bell_volume: 0,
            term_env_var: "core-256color".to_string(),
            allow_alt_screen: true,
            allow_window_ops: false,
            default_origin_mode: false,
        }
    }
}

/// Defines the color scheme for the terminal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ColorScheme {
    pub foreground: Color,
    pub background: Color,
    pub cursor: Color,
    pub reverse_cursor: Color,
    pub ansi: [Color; 16],
}

impl Default for ColorScheme {
    fn default() -> Self {
        ColorScheme {
            foreground: Color::Named(NamedColor::White),
            background: Color::Named(NamedColor::Black),
            cursor: Color::Named(NamedColor::White),
            reverse_cursor: Color::Named(NamedColor::Black),
            ansi: [
                Color::Named(NamedColor::Black),
                Color::Named(NamedColor::Red),
                Color::Named(NamedColor::Green),
                Color::Named(NamedColor::Yellow),
                Color::Named(NamedColor::Blue),
                Color::Named(NamedColor::Magenta),
                Color::Named(NamedColor::Cyan),
                Color::Named(NamedColor::White),
                Color::Named(NamedColor::BrightBlack),
                Color::Named(NamedColor::BrightRed),
                Color::Named(NamedColor::BrightGreen),
                Color::Named(NamedColor::BrightYellow),
                Color::Named(NamedColor::BrightBlue),
                Color::Named(NamedColor::BrightMagenta),
                Color::Named(NamedColor::BrightCyan),
                Color::Named(NamedColor::BrightWhite),
            ],
        }
    }
}

/// Defines settings related to the shell and its execution environment.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ShellConfig {
    pub program: Option<PathBuf>,
    pub args: Vec<String>,
    pub working_directory: Option<PathBuf>,
}

/// Defines settings related to mouse behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MouseConfig {
    pub cursor_shape: String,
    pub force_modifier: String,
}

impl Default for MouseConfig {
    fn default() -> Self {
        MouseConfig {
            cursor_shape: "xterm".to_string(),
            force_modifier: "ShiftMask".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch directory unique to one test.
    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("core-term-config-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    #[test]
    fn a_missing_file_is_the_defaults() {
        let path = scratch("missing").join("config.json");
        let config = load_config_from(&path).expect("defaults");
        assert_eq!(
            config.appearance.cell_height_px,
            Config::default().appearance.cell_height_px
        );
    }

    #[test]
    fn a_file_names_only_what_it_changes() {
        let path = scratch("partial").join("config.json");
        std::fs::write(&path, r#"{ "appearance": { "cell_height_px": 22 } }"#).expect("write");

        let config = load_config_from(&path).expect("parses");
        assert_eq!(config.appearance.cell_height_px, 22);
        assert_eq!(
            config.appearance.cell_width_px,
            Config::default().appearance.cell_width_px,
            "unnamed settings keep their defaults"
        );
    }

    #[test]
    fn every_default_setting_round_trips_through_the_file_format() {
        let path = scratch("round-trip").join("config.json");
        let written = serde_json::to_string_pretty(&Config::default()).expect("serialize");
        std::fs::write(&path, written).expect("write");

        let config = load_config_from(&path).expect("parses what it wrote");
        let copy = Some(UserInputAction::InitiateCopy);
        assert_eq!(
            config.keybindings.lookup.get(&crate::keys::chord(
                KeySymbol::Char('c'),
                Modifiers::CONTROL | Modifiers::SHIFT
            )),
            copy.as_ref(),
            "bindings are rebuilt from the file"
        );
    }

    #[test]
    fn a_file_that_does_not_parse_is_an_error_not_the_defaults() {
        let path = scratch("invalid").join("config.json");
        std::fs::write(&path, r#"{ "appearance": { "cell_height_px": "tall" } }"#).expect("write");

        assert!(load_config_from(&path).is_err());
    }
}
