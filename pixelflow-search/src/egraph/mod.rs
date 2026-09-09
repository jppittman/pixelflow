//! E-graph based expression optimizer.
//!
//! An e-graph (equality graph) compactly represents many equivalent expressions.
//! We use equality saturation to find the cheapest form of mathematical expressions.
//!
//! # Module Structure
//!
//! - [`node`]: Core data structures (EClassId, Op, ENode)
//! - [`candidate`]: Candidate-local features and the dedup key a guided
//!   saturation loop orders by
//! - [`cost`]: Cost model for extraction
//! - [`rewrite`]: Rewrite rule infrastructure
//! - [`extract`]: Expression tree extraction, including DAG-aware extraction
//! - [`extraction`]: Cost-model policy selection (the static latency prior)
//!   shared by the AOT macro tier and [`crate::runtime`]
//! - [`saturate`]: Budget-limited saturation, plus the size-based
//!   [`saturate::SaturationConfig`] presets both tiers drive it with
//! - [`graph`]: The EGraph itself
//! - `growth` (feature `saturation-telemetry`): per-application growth
//!   telemetry — how many e-nodes/e-classes a rewrite actually added
//! - [`deps`]: Dependency analysis for uniform hoisting
//!
//! Mathematical rewrite rules are now in the [`crate::math`] module.
//!
//! This module is the compile-time-agnostic core: it optimizes an e-graph
//! regardless of when it was built. [`crate::runtime`] is the runtime-facing
//! front door — insert an [`pixelflow_ir::arena::ExprArena`] directly, no AST
//! involved.

pub mod anytime;
pub mod candidate;
pub(crate) mod cost;
pub mod deps;
pub mod derivative;
pub(crate) mod extract;
pub mod fold_rules;
mod graph;
// Per-application growth telemetry (docs/plans/2026-08-31-guide-design-revision.md
// §4.1) has no production consumer yet, exactly like `crate::telemetry` —
// gated behind the same `saturation-telemetry` feature so a build that
// doesn't ask for it doesn't carry the module at all.
#[cfg(feature = "saturation-telemetry")]
mod growth;
mod guided;
pub mod insert;
// The hindsight labeler reads the provenance journal directly
// (`derivation_ancestors`, `Origin`, `Provenance::recorded_count`) — it has
// nothing to compute without it.
#[cfg(feature = "provenance-journal")]
mod labeler;
mod node;
pub mod ops;
pub mod optimizer;
pub mod provenance;
pub mod rewrite;
pub mod rule_order;
pub mod rules;
pub mod saturate;
pub mod template;
// Research only: the read-only half of the extraction-witness instrument
// (docs/plans/2026-09-08-extraction-witnesses.md). Gated behind the feature
// every `pixelflow-pipeline` harness already builds with, so nothing here
// reaches a downstream build.
#[cfg(feature = "provenance-journal")]
pub mod witness;

// Re-export public API
pub use anytime::{
    APP_CHECKPOINT_GRID, AnytimeCheckpoint, AnytimeCurve, AnytimeCurveOutput, run_anytime_curve,
};
pub use candidate::{
    CandidateFeatures, CandidateKey, ClassContentKey, Firing,
    REGISTERED_PRIMARY_BUDGET_APPLICATIONS,
};
pub use cost::{CostFunction, CostModel};
pub use deps::{Deps, DepsAnalysis};
pub use derivative::{ChainRule, derivative_rules};
pub use extract::{
    ChoiceCost, ClaimAudit, CostScale, ExtractedDAG, Extraction, ExtractionObjective,
    ExtractionReport, SHARED_DAG_PASS_BYTE_BUDGET, SharedPassStats,
    build_extracted_dag_from_choices, choices_to_arena, compute_ref_counts, cost_of_choices,
    extract, extract_dag,
};
pub use fold_rules::{EmptyFold, PeelFold, fold_rules};
pub use graph::{
    ApplicationMask, ApplyResult, EGraph, EGraphBatch, HARD_CLASS_LIMIT, MaskScope, RewriteTarget,
    SaturationStats, SaturationStop, ScanStop,
};
#[cfg(feature = "saturation-telemetry")]
pub use growth::{GrowthTelemetry, RuleGrowth};
pub use insert::{Declined, insert, reachable_count};
#[cfg(feature = "provenance-journal")]
pub use labeler::{EpisodeLabels, EpisodeResult, Label, RuleStats, run_episode};
pub use node::{EClassId, ENode};
pub use ops::{Op, Vocabulary};
pub use optimizer::{Budget, Limits, Optimized, Optimizer, OptimizerStats};
#[cfg(feature = "provenance-journal")]
pub use optimizer::{KeepJournal, Observer};
pub use provenance::{ApplicationId, ENodeId, Provenance};
#[cfg(feature = "provenance-journal")]
pub use provenance::{
    ApplicationRecord, Origin, UnionEvent, derivation_ancestors, format_derivation_trace,
};
pub use rewrite::{Rewrite, RewriteAction, TemplateArena};
pub use rules::{Fingerprint, RuleId, RuleSet, rule_label};
pub use saturate::{
    APPLICATIONS_PER_CLASS, CLASSICAL_CLASS_CEILING, CLASSICAL_CLASS_CEILING_CALIBRATED,
    CLASSICAL_CLASS_FLOOR, CLASSICAL_CLASSES_PER_INSERTED_CLASS, InputSize, SaturationConfig,
    SaturationResult, achievable_cost_within_budget, config_for_input, config_for_node_count,
    saturate_with_budget, saturate_with_full_budget,
};
pub use template::TemplateRewrite;

// Re-export rule types from math module for backward compatibility
pub use crate::math::{
    AddNeg,
    // Trig
    AngleAddition,
    Annihilator,
    Associative,
    Commutative,
    Exp2Log2,
    ExpLn,
    // Fusion
    FmaFusion,
    // Exp
    FunctionInverse,
    Homomorphism,
    Identity,
    // Algebra
    InversePair,
    MulRecip,
    // Parity
    Parity,
    ParityKind,
    RecipSqrt,
    Sign,
    algebra_rules,
    // Rule collections
    all_math_rules,
    basic_algebra_rules,
    core_rules,
    exp_rules,
    fusion_rules,
    inverse_pair_rules,
    parity_rules,
    transcendental_rules,
    trig_rules,
};

/// All rewrite rules: 40 math + 2 fusion = 42 total.
///
/// This is the complete rule set for optimization, training, and production.
pub fn all_rules() -> Vec<Box<dyn Rewrite>> {
    crate::math::all_rules()
}

/// Build [`RuleTemplates`] from all registered rules.
///
/// Collects LHS/RHS expression templates from every rule that provides them.
/// Rules without templates (returning `None`) get empty slots.
///
/// # Panics
///
/// Panics if `all_rules()` returns an empty list (should never happen).
#[must_use]
pub fn collect_rule_templates() -> crate::nnue::RuleTemplates {
    let rules = all_rules();
    assert!(
        !rules.is_empty(),
        "collect_rule_templates: all_rules() returned 0 rules"
    );

    let mut templates = crate::nnue::RuleTemplates::with_capacity(rules.len());

    for (idx, rule) in rules.iter().enumerate() {
        templates.build(idx, rule.as_ref());
    }

    templates
}
