// src/keys.rs

use crate::config::Config;
use crate::term::action::UserInputAction;
use log::debug;
pub use pixelflow_runtime::input::{KeySymbol, Modifiers};

/// The key a chord names, independent of how a platform reports it.
///
/// Platforms disagree about what a chord *is*: X11 reports Ctrl+Shift+C as
/// the control character it types (U+0003), macOS as the key `c`, and either
/// may add Caps Lock or Num Lock to the modifiers. A binding means the key and
/// the modifiers the user held, so both sides of a lookup are folded to that:
/// lock states dropped, a control character under Ctrl named by its letter,
/// and letters lowercase.
#[must_use]
pub fn chord(key_symbol: KeySymbol, modifiers: Modifiers) -> (KeySymbol, Modifiers) {
    let modifiers = modifiers - (Modifiers::CAPS_LOCK | Modifiers::NUM_LOCK);
    let KeySymbol::Char(c) = key_symbol else {
        return (key_symbol, modifiers);
    };
    let c = match (modifiers.contains(Modifiers::CONTROL), c) {
        (true, '\u{1}'..='\u{1a}') => char::from(b'a' + (c as u8 - 1)),
        _ => c.to_ascii_lowercase(),
    };
    (KeySymbol::Char(c), modifiers)
}

/// Maps a given key symbol and modifiers to a `UserInputAction` based on the provided configuration.
///
/// It performs an O(1) lookup in `config.keybindings.lookup`.
/// If a match is found, it returns a clone of the corresponding `UserInputAction`.
/// Otherwise, it returns `None`.
#[must_use]
pub fn map_key_event_to_action(
    key_symbol: KeySymbol,
    modifiers: Modifiers,
    config: &Config,
) -> Option<UserInputAction> {
    if let Some(action) = config.keybindings.lookup.get(&chord(key_symbol, modifiers)) {
        debug!(
            "Keybinding: {:?} + {:?} => {:?}",
            modifiers, key_symbol, action
        );
        Some(action.clone())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, Keybinding, RawKeybindingsConfig};
    use crate::term::action::UserInputAction;

    fn config_with_bindings(bindings: Vec<Keybinding>) -> Config {
        // Use RawKeybindingsConfig::into() to populate the lookup map
        Config {
            keybindings: RawKeybindingsConfig { bindings }.into(),
            ..Default::default()
        }
    }

    #[test]
    fn map_key_event_to_action_returns_the_bound_action_when_symbol_and_modifiers_match() {
        let bindings = vec![
            Keybinding {
                key: KeySymbol::Char('C'),
                mods: Modifiers::CONTROL | Modifiers::SHIFT,
                action: UserInputAction::InitiateCopy,
            },
            Keybinding {
                key: KeySymbol::Char('Q'),
                mods: Modifiers::CONTROL,
                action: UserInputAction::RequestQuit,
            },
        ];
        let config = config_with_bindings(bindings);

        let result = map_key_event_to_action(
            KeySymbol::Char('C'),
            Modifiers::CONTROL | Modifiers::SHIFT,
            &config,
        );
        assert_eq!(result, Some(UserInputAction::InitiateCopy));

        let result_quit =
            map_key_event_to_action(KeySymbol::Char('Q'), Modifiers::CONTROL, &config);
        assert_eq!(result_quit, Some(UserInputAction::RequestQuit));
    }

    #[test]
    fn map_key_event_to_action_returns_none_when_the_symbol_does_not_match() {
        let bindings = vec![Keybinding {
            key: KeySymbol::Char('C'),
            mods: Modifiers::CONTROL | Modifiers::SHIFT,
            action: UserInputAction::InitiateCopy,
        }];
        let config = config_with_bindings(bindings);

        let result = map_key_event_to_action(
            KeySymbol::Char('X'),
            Modifiers::CONTROL | Modifiers::SHIFT,
            &config,
        );
        assert_eq!(result, None);
    }

    #[test]
    fn map_key_event_to_action_returns_none_when_the_modifiers_do_not_match() {
        let bindings = vec![Keybinding {
            key: KeySymbol::Char('C'),
            mods: Modifiers::CONTROL | Modifiers::SHIFT,
            action: UserInputAction::InitiateCopy,
        }];
        let config = config_with_bindings(bindings);

        let result = map_key_event_to_action(KeySymbol::Char('C'), Modifiers::CONTROL, &config);
        assert_eq!(result, None);
    }

    #[test]
    fn map_key_event_to_action_returns_none_when_no_bindings_are_configured() {
        let config = config_with_bindings(vec![]);
        let result = map_key_event_to_action(
            KeySymbol::Char('C'),
            Modifiers::CONTROL | Modifiers::SHIFT,
            &config,
        );
        assert_eq!(result, None);
    }

    #[test]
    fn default_copy_binding_matches_how_every_platform_reports_ctrl_shift_c() {
        let config = Config::default();
        let ctrl_shift = Modifiers::CONTROL | Modifiers::SHIFT;
        let copy = Some(UserInputAction::InitiateCopy);

        // X11 reports the control character the chord types.
        assert_eq!(
            map_key_event_to_action(KeySymbol::Char('\u{3}'), ctrl_shift, &config),
            copy
        );
        // macOS reports the key.
        assert_eq!(
            map_key_event_to_action(KeySymbol::Char('c'), ctrl_shift, &config),
            copy
        );
        // Either may add a lock state, or report the shifted letter.
        assert_eq!(
            map_key_event_to_action(
                KeySymbol::Char('C'),
                ctrl_shift | Modifiers::CAPS_LOCK | Modifiers::NUM_LOCK,
                &config
            ),
            copy
        );
        // Plain Ctrl+C is the shell's interrupt, not a binding.
        assert_eq!(
            map_key_event_to_action(KeySymbol::Char('\u{3}'), Modifiers::CONTROL, &config),
            None
        );
    }

    #[test]
    fn map_key_event_to_action_returns_the_first_matching_binding_when_several_match() {
        let bindings = vec![
            Keybinding {
                key: KeySymbol::Char('A'),
                mods: Modifiers::ALT,
                action: UserInputAction::RequestZoomIn,
            },
            Keybinding {
                key: KeySymbol::Char('A'),
                mods: Modifiers::ALT,
                action: UserInputAction::RequestZoomOut,
            },
        ];
        let config = config_with_bindings(bindings);
        let result = map_key_event_to_action(KeySymbol::Char('A'), Modifiers::ALT, &config);
        assert_eq!(result, Some(UserInputAction::RequestZoomIn));
    }
}
