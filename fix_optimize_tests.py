import re

file = 'pixelflow-compiler/src/optimize.rs'
with open(file, 'r') as f: content = f.read()

# I will refactor ALL tests in `pixelflow-compiler/src/optimize.rs` to follow the Mutant Hunter style guide.
# 1. Rename `test_...` to `[method]_should_[outcome]_when_[condition]` or `[unit]_[state]_[expected_outcome]`
# 2. Replace `.unwrap()` with `.expect("Reason")`
# 3. Remove tests with no assertions (if any)
# 4. Remove `assert!(..., true)` if any

# Step 2: Replace unwrap
content = content.replace('parse(input).unwrap()', 'parse(input).expect("Failed to parse input tokens into KernelDef")')
content = content.replace('analyze(kernel).unwrap()', 'analyze(kernel).expect("Failed to semantically analyze KernelDef")')

# Step 1: Rename tests
replacements = {
    'fn test_constant_folding()': 'fn optimizer_should_fold_constants_when_literals_exist()',
    'fn test_identity_add()': 'fn optimizer_should_remove_addition_when_adding_zero()',
    'fn test_zero_mul()': 'fn optimizer_should_evaluate_to_zero_when_multiplying_by_zero()',
    'fn test_complex_folding()': 'fn optimizer_should_fold_complex_nested_literals_when_present()',
    'fn test_egraph_identity_add()': 'fn egraph_should_remove_addition_when_adding_zero()',
    'fn test_egraph_zero_mul()': 'fn egraph_should_evaluate_to_zero_when_multiplying_by_zero()',
    'fn test_egraph_identity_mul()': 'fn egraph_should_remove_multiplication_when_multiplying_by_one()',
    'fn test_egraph_complex_expression()': 'fn egraph_should_evaluate_complex_expression_to_zero_when_algebra_simplifies_it()',
    'fn test_egraph_sub_self()': 'fn egraph_should_evaluate_to_zero_when_subtracting_self()',
    'fn test_egraph_fma_fusion_with_fma_costs()': 'fn egraph_should_fuse_fma_when_fma_cost_is_cheap()',
    'fn test_egraph_fma_unfused_with_expensive_fma()': 'fn egraph_should_unfuse_fma_when_fma_cost_is_expensive()',
    'fn test_egraph_fma_fused_with_default_costs()': 'fn egraph_should_fuse_fma_when_default_costs_applied()',
    'fn test_egraph_preserves_variables()': 'fn egraph_should_preserve_variables_when_no_optimization_applies()',
    'fn test_egraph_handles_sqrt()': 'fn egraph_should_preserve_sqrt_when_no_optimization_applies()',
    'fn test_egraph_div_sqrt_to_rsqrt()': 'fn egraph_should_convert_div_sqrt_to_rsqrt_when_optimizing()',
    'fn test_egraph_double_negation()': 'fn egraph_should_remove_negation_when_double_negation_applied()',
    'fn test_global_optimization_across_let_bindings()': 'fn global_optimizer_should_propagate_values_across_let_bindings()',
    'fn test_global_optimization_zero_multiplication()': 'fn global_optimizer_should_evaluate_to_zero_across_let_bindings_when_multiplied()',
    'fn test_global_optimization_self_subtraction()': 'fn global_optimizer_should_evaluate_to_zero_across_let_bindings_when_subtracted()',
    'fn test_global_fma_across_bindings()': 'fn global_optimizer_should_fuse_fma_across_let_bindings()',
    'fn test_discriminant_pattern()': 'fn global_optimizer_should_fuse_discriminant_pattern_when_fma_cost_is_cheap()',
    'fn test_discriminant_with_intrinsics()': 'fn global_optimizer_should_fuse_discriminant_intrinsics_when_fma_cost_is_cheap()',
    'fn test_dag_optimization_shared_subexpr()': 'fn dag_optimizer_should_extract_shared_subexpr_when_present()',
    'fn test_dag_optimization_triple_shared()': 'fn dag_optimizer_should_extract_triple_shared_subexpr_when_present()',
    'fn test_dag_optimization_no_sharing()': 'fn dag_optimizer_should_not_extract_subexpr_when_no_sharing_present()',
    'fn test_full_pipeline_discriminant()': 'fn pipeline_should_generate_correct_negation_wrapper_when_optimizing_discriminant()',
}

for k, v in replacements.items():
    content = content.replace(k, v)

with open(file, 'w') as f: f.write(content)
print("done refactoring optimize.rs tests")
