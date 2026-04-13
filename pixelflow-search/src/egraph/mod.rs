//! E-graph based expression optimizer.
//!
//! An e-graph (equality graph) compactly represents many equivalent expressions.
//! We use equality saturation to find the cheapest form of mathematical expressions.
//!
//! # Module Structure
//!
//! - [`node`]: Core data structures (EClassId, Op, ENode)
//! - [`cost`]: Cost model for extraction
//! - [`rewrite`]: Rewrite rule infrastructure
//! - [`extract`]: Expression tree extraction, including DAG-aware extraction
//! - [`graph`]: The EGraph itself
//! - [`deps`]: Dependency analysis for uniform hoisting
//! - [`codegen`]: Code generation from extracted expressions (tree & DAG)
//!
//! Mathematical rewrite rules are now in the [`crate::math`] module.

pub mod codegen;
mod cost;
pub mod deps;
pub(crate) mod extract;
mod graph;
mod node;
pub mod ops;
pub mod rewrite;
pub mod saturate;

// Re-export public API
pub use cost::{CostFunction, CostModel};
pub use deps::{Deps, DepsAnalysis};
pub use extract::{
    ExtractedDAG, IncrementalExtractor, build_extracted_dag_from_choices, choices_to_arena,
    compute_ref_counts, extract_dag, extract_neural_to_arena,
};
pub use graph::{ApplyResult, EGraph, EGraphBatch, RewriteTarget};
pub use node::{EClassId, ENode};
pub use ops::Op;
pub use rewrite::{Pattern, Rewrite, RewriteAction};
pub use saturate::{SaturationResult, achievable_cost_within_budget, saturate_with_budget};

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
        if let (Some(lhs), Some(rhs)) = (rule.lhs_template(), rule.rhs_template()) {
            templates.set(idx, lhs, rhs);
        }
        // Rules without templates get None slots (handled by RuleTemplates)
    }

    templates
}
