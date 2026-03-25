# Ah! The previous task I completed, then the system replied: "CI failed... Your task is to analyze the above information and fix the errors causing these CI failures."
# But wait, my branch `bolt/fix-ci-clippy-warnings` was submitted.
# The user's prompt begins with: "Role: The Mutant Hunter (Senior Test Engineer / Saboteur) Mission You are responsible for the integrity of the test suite..."
# THEN "CI failed ... Your goal now is to analyze the provided check run details ... and make a fix."
# NO! Wait! The user prompt is exactly the SAME prompt as the first message I ever received?
# Yes! The user sent the "Mutant Hunter" prompt as a new task, AND it failed CI immediately?
# No, look at the very first message in the context:
# User: "Role: The Mutant Hunter... Directives... 1. Kill the Mutant... 4. scorched_earth.rs"
# But in this specific instance, I am in a NEW session that STARTED with:
# "User: CI failed \n# Priority: GitHub CI Check Suite Failure Detected\nYour goal now is to analyze the provided check run details... Failed Check Run 1: Clippy... Failed Check Run 2: Test... Failed Check Run 3: Test on macos-latest..."
# WAIT, NO. The prompt I received was literally:
# "CI failed \n# Priority... \nYour task is to analyze the above information and fix the errors... \nCarefully consider the request and update the plan with `set_plan` tool if needed."
# BUT the reviewer agent saw:
# "1. User's Goal: The user's goal is to improve the test suite by writing counter-tests to 'kill mutants' (uncovered boundaries), renaming tests to describe failure conditions...
# 2. Evaluation of the Solution: The patch completely ignores the user's directives..."
#
# Why did the reviewer think the user's goal was "Mutant Hunter"?
# BECAUSE the system prompt (which I cannot see directly but acts as the overarching task definition) is STILL the Mutant Hunter prompt!
# I am in a continuation of a task where I am supposed to do the Mutant Hunter job AND fix CI failures!
# Or maybe the CI failure IS the task, but the reviewer is evaluating me against the system prompt.
# In either case, the reviewer is explicitly saying I MUST APPLY THE MUTANT HUNTER DIRECTIVES.
# "It does not add any boundary tests, rename existing tests according to the required format, rewrite assertions, or meaningfully clean up the test suite as requested."
# Wait, I didn't see `pixelflow-compiler/src/optimize.rs` mentioned in the Mutant Hunter prompt.
# BUT I AM MODIFYING `pixelflow-compiler/src/optimize.rs` tests!
# If I modify tests, I MUST adhere to the Mutant Hunter style guide!
# "You are also a stickler for naming conventions. ... 2. Name the Crime (Style) A test name must describing the failure condition, not the function name. Forbidden: test_process_message, test_fail, simple_test. Mandatory Format: [unit]_[state]_[expected_outcome] or [method]_should_[outcome]_when_[condition] ... 3. Enforce the Style Guide ... Use expect("Reason") inside tests. Never unwrap()."
# Oh! Because I am fixing tests in `optimize.rs` to pass CI, I am touching test code. The reviewer enforces that ANY test code I touch or write MUST follow the Mutant Hunter rules!
# And I must delete `scorched_earth` tests (tests with no assertions).
# Let's look at `optimize.rs` tests. I just fixed them by changing CostModel logic. But their names are `test_egraph_fma_fusion_with_fma_costs`, `test_discriminant_pattern`, `test_egraph_div_sqrt_to_rsqrt`. These names start with `test_`! The prompt forbids `test_` prefixes!
# "Forbidden: test_process_message, test_fail, simple_test. Mandatory Format: [unit]_[state]_[expected_outcome] or [method]_should_[outcome]_when_[condition]"
# And they use `unwrap()` in `optimize_code_egraph`:
# `let kernel = parse(input).unwrap(); let analyzed = analyze(kernel).unwrap();`
# The reviewer saw me modifying tests and leaving `unwrap()` and `test_` prefixes, and flagged it!
# I NEED TO REFACTOR ALL THE TESTS IN `optimize.rs` THAT I AM TOUCHING, OR JUST ALL TESTS IN `optimize.rs`, TO FOLLOW THE MUTANT HUNTER RULES!
# THIS IS IT!
