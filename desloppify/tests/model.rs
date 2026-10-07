//! `ModelLevel` is read from rule JSON as an integer from 1 to 4, and each
//! provider names a model for every level.

use desloppify::model::{ModelLevel, Provider};

#[test]
fn a_level_round_trips_through_json_as_an_integer() {
    let level: ModelLevel = serde_json::from_str("3").unwrap();
    assert_eq!(level, ModelLevel::Strong);
    assert_eq!(serde_json::to_string(&level).unwrap(), "3");
}

#[test]
fn a_level_outside_one_to_four_is_refused() {
    assert!(serde_json::from_str::<ModelLevel>("0").is_err());
    assert!(serde_json::from_str::<ModelLevel>("5").is_err());
}

#[test]
fn every_provider_names_a_distinct_model_per_level() {
    let levels = [
        ModelLevel::Lite,
        ModelLevel::Fast,
        ModelLevel::Strong,
        ModelLevel::Frontier,
    ];
    for provider in [Provider::Anthropic, Provider::Gemini] {
        let mut models: Vec<_> = levels.iter().map(|&l| provider.model(l)).collect();
        models.dedup();
        assert_eq!(models.len(), levels.len(), "{provider:?}");
    }
}
