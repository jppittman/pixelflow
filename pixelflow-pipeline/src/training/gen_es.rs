//! Evolutionary Strategy (ES) optimizer over `BwdGenConfig` parameters.
//!
//! Adapts expression generation to target the judge's blind spots (highest
//! prediction error) by optimizing the 6 `BwdGenConfig` parameters using
//! natural evolution strategies with rank-normalized fitness.
//!
//! # Algorithm
//!
//! Each `step()`:
//! 1. Sample `population` perturbation vectors from N(0, sigma)
//! 2. Evaluate each candidate config by generating expressions and measuring
//!    |predict_log_cost(expr) - log(benchmark_jit(expr))|
//! 3. Rank-normalize fitnesses to [-0.5, 0.5]
//! 4. Compute gradient and update mean parameters
//!
//! The search space is normalized to [0,1]^6 for uniform step sizes.

use pixelflow_search::nnue::{BwdGenConfig, BwdGenerator, ExprNnue, RuleTemplates};

use crate::jit_bench::benchmark_jit_arena;

/// Number of parameters in the ES search space.
const ES_DIM: usize = 6;

/// Configuration for the ES optimizer.
#[derive(Clone, Debug)]
pub struct GenEsConfig {
    /// Noise standard deviation for perturbations.
    pub sigma: f32,
    /// Learning rate for parameter updates.
    pub alpha: f32,
    /// Number of perturbation candidates per round.
    pub population: usize,
    /// Number of expressions generated per candidate evaluation.
    pub samples_per_candidate: usize,
    /// RNG seed.
    pub seed: u64,
}

impl Default for GenEsConfig {
    fn default() -> Self {
        Self {
            sigma: 0.1,
            alpha: 0.05,
            population: 10,
            samples_per_candidate: 8,
            seed: 42,
        }
    }
}

/// Evolutionary Strategy optimizer for `BwdGenConfig` parameters.
///
/// Maintains a mean vector in [0,1]^6 and perturbs it each step,
/// selecting directions that maximize the judge's prediction error.
pub struct GenEs {
    /// Current mean in [0,1]^6.
    mu: [f32; ES_DIM],
    /// Noise standard deviation.
    sigma: f32,
    /// Learning rate.
    alpha: f32,
    /// Number of perturbation candidates per round.
    population: usize,
    /// Expressions generated per candidate evaluation.
    samples_per_candidate: usize,
    /// LCG PRNG state.
    rng_state: u64,
    /// Mean fitness from the last step (mean absolute prediction error).
    last_mean_fitness: f32,
    /// Rule templates for BwdGenerator junkification.
    templates: RuleTemplates,
}

impl GenEs {
    /// Create a new ES optimizer from configuration.
    #[must_use]
    pub fn new(config: GenEsConfig, templates: RuleTemplates) -> Self {
        let mu = normalize(&BwdGenConfig::default());
        Self {
            mu,
            sigma: config.sigma,
            alpha: config.alpha,
            population: config.population,
            samples_per_candidate: config.samples_per_candidate,
            rng_state: config.seed,
            last_mean_fitness: 0.0,
            templates,
        }
    }

    /// Run one ES step: perturb, evaluate, update.
    ///
    /// Returns the denormalized `BwdGenConfig` corresponding to the updated mean.
    /// If ALL candidates fail JIT benchmarking, returns the current mean unchanged.
    pub fn step(&mut self, judge: &ExprNnue) -> BwdGenConfig {
        // Sample ALL epsilon vectors first, before any evaluation.
        // This keeps epsilon sampling deterministic regardless of JIT failures.
        let mut epsilons = Vec::with_capacity(self.population);
        let mut seeds = Vec::with_capacity(self.population);

        for pop_idx in 0..self.population {
            let mut eps = [0.0f32; ES_DIM];
            for (d, e) in eps.iter_mut().enumerate() {
                let n = self.rand_normal();
                assert!(
                    n.is_finite(),
                    "rand_normal() returned {n} at population={pop_idx}, dim={d}, \
                     rng_state={}",
                    self.rng_state,
                );
                *e = n * self.sigma;
            }
            epsilons.push(eps);
            seeds.push(self.rand_u64());
        }

        // Now evaluate each candidate (this may involve JIT which can fail).
        let mut fitnesses: Vec<Option<f32>> = Vec::with_capacity(self.population);
        for (pop_idx, eps) in epsilons.iter().enumerate() {
            let mut candidate = [0.0f32; ES_DIM];
            for i in 0..ES_DIM {
                candidate[i] = clamp01(self.mu[i] + eps[i]);
            }

            let config = denormalize(&candidate);
            let fitness = self.evaluate_candidate_with_seed(&config, judge, seeds[pop_idx]);
            fitnesses.push(fitness);
        }

        // Collect indices of candidates that succeeded.
        let valid: Vec<usize> = fitnesses
            .iter()
            .enumerate()
            .filter_map(|(i, f)| f.map(|_| i))
            .collect();

        if valid.is_empty() {
            eprintln!(
                "[gen_es] ALL {} candidates failed JIT; skipping update",
                self.population
            );
            return denormalize(&self.mu);
        }

        // Rank-normalize the valid fitnesses to [-0.5, 0.5].
        let mut ranked: Vec<(usize, f32)> = valid
            .iter()
            .map(|&i| (i, fitnesses[i].expect("filtered to Some above")))
            .collect();
        ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let n = ranked.len();
        // Map: rank 0 → -0.5, rank n-1 → +0.5
        let mut normalized = vec![0.0f32; self.population];
        for (rank, &(idx, _)) in ranked.iter().enumerate() {
            normalized[idx] = if n == 1 {
                0.0 // Single candidate: no gradient signal.
            } else {
                (rank as f32) / ((n - 1) as f32) - 0.5
            };
        }

        // Compute mean fitness for diagnostics.
        let fitness_sum: f32 = ranked.iter().map(|&(_, f)| f).sum();
        self.last_mean_fitness = fitness_sum / n as f32;

        // Gradient: (1 / (N * sigma)) * sum(normalized_fitness_i * epsilon_i)
        let scale = 1.0 / (n as f32 * self.sigma);
        let mut grad = [0.0f32; ES_DIM];
        for &idx in &valid {
            let nf = normalized[idx];
            for d in 0..ES_DIM {
                grad[d] += nf * epsilons[idx][d];
            }
        }
        for g in &mut grad {
            *g *= scale;
        }

        // Fail fast on NaN gradient — something upstream is broken.
        for d in 0..ES_DIM {
            assert!(
                grad[d].is_finite(),
                "NaN/Inf in grad[{d}]={}, scale={scale}, n={n}, sigma={}, mu={:?}, \
                 epsilons={:?}, normalized={:?}, fitnesses={:?}",
                grad[d],
                self.sigma,
                self.mu,
                epsilons,
                normalized,
                fitnesses
            );
        }

        // Update mu and clamp to [0, 1].
        for d in 0..ES_DIM {
            self.mu[d] = clamp01(self.mu[d] + self.alpha * grad[d]);
        }

        eprintln!(
            "[gen_es] step: {}/{} candidates valid, mean_fitness={:.4}, mu={:?}",
            n, self.population, self.last_mean_fitness, self.mu,
        );

        denormalize(&self.mu)
    }

    /// Mean absolute prediction error from the last step.
    #[must_use]
    pub fn last_fitness(&self) -> f32 {
        self.last_mean_fitness
    }

    /// Evaluate a single candidate config with a pre-determined seed.
    ///
    /// Returns a composite fitness score combining:
    /// - Judge prediction error: `|predict_log_cost(expr) - log_ns(jit)|`
    ///   (higher = judge blind spot = more valuable training signal)
    /// - Rewrite richness penalty: configs that produce `rewrites_applied == 0`
    ///   generate trajectories where `unoptimized == optimized`, giving the
    ///   e-graph nothing to simplify and producing empty trajectories.
    ///   Such configs are penalized by subtracting from their score.
    ///
    /// Returns None if ALL JIT evaluations fail.
    fn evaluate_candidate_with_seed(
        &self,
        config: &BwdGenConfig,
        judge: &ExprNnue,
        gen_seed: u64,
    ) -> Option<f32> {
        let mut bwd_gen = BwdGenerator::new(gen_seed, config.clone(), self.templates.clone());

        let mut error_sum = 0.0f32;
        let mut count = 0u32;
        let mut zero_rewrite_count = 0u32;

        for _ in 0..self.samples_per_candidate {
            let pair = bwd_gen.generate_arena();

            // Track how often junkification fails to produce any rewrites.
            // Configs that regularly produce rewrites_applied==0 generate
            // unoptimized==optimized seeds which produce empty trajectories.
            if pair.rewrites_applied == 0 {
                zero_rewrite_count += 1;
            }

            // Skip oversized expressions: the AArch64 JIT uses 12-bit LDR offsets
            // (max 4095 entries), so expressions with too many nodes overflow the
            // constant pool. 150 nodes is a safe ceiling that matches self_play.rs.
            //
            // Use arena.node_count_subtree for the optimized subtree node count
            // (O(N) traversal, but avoids materializing the full Expr just
            // to count nodes before the size check).
            if pair.arena.node_count_subtree(pair.optimized) > 150 {
                continue;
            }

            // Benchmark and evaluate directly from the arena.
            match benchmark_jit_arena(&pair.arena, pair.optimized) {
                Ok(b) => {
                    let ns = b.ns;
                    let actual = log_ns(ns);
                    let acc = pixelflow_search::nnue::EdgeAccumulator::from_arena_dedup(
                        &pair.arena,
                        pair.optimized,
                        &judge.embeddings,
                    );
                    let predicted = judge.predict_log_cost_with_features(&acc);
                    let err = libm::fabsf(predicted - actual);
                    if !err.is_finite() {
                        eprintln!(
                            "[ES] non-finite judge error: predicted={predicted} actual={actual} ns={ns}"
                        );
                    } else {
                        error_sum += err;
                        count += 1;
                    }
                }
                Err(e) => {
                    eprintln!("[ES] JIT compile failed: {e}");
                }
            }
        }

        if count == 0 {
            return None;
        }

        let mean_error = error_sum / count as f32;

        // Penalize configs that frequently produce zero-rewrite pairs.
        // The penalty scales linearly with the fraction of zero-rewrite samples:
        //   penalty = mean_error * zero_rewrite_fraction
        // This subtracts from the fitness, steering ES away from configs
        // that can't junkify and would produce empty trajectories.
        let zero_rewrite_fraction = zero_rewrite_count as f32 / self.samples_per_candidate as f32;
        let richness_penalty = mean_error * zero_rewrite_fraction;

        Some(mean_error - richness_penalty)
    }

    // ---- PRNG (LCG, same pattern as rest of codebase) ----

    fn rand_u64(&mut self) -> u64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.rng_state
    }

    fn rand_f32(&mut self) -> f32 {
        (self.rand_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Box-Muller transform: two uniform samples → one standard normal sample.
    fn rand_normal(&mut self) -> f32 {
        let u1 = self.rand_f32().max(1e-10);
        let u2 = self.rand_f32();
        libm::sqrtf(-2.0 * libm::logf(u1)) * libm::cosf(2.0 * core::f32::consts::PI * u2)
    }
}

// ============================================================================
// Normalization: BwdGenConfig ↔ [0,1]^6
// ============================================================================

/// Normalize a `BwdGenConfig` into [0,1]^6.
///
/// | Param              | Range      | Formula                  |
/// |--------------------|------------|--------------------------|
/// | max_depth          | [5, 20]    | (v - 5) / 15             |
/// | leaf_prob          | [0.05, 0.8]| (v - 0.05) / 0.75        |
/// | num_vars           | [1, 4]     | (v - 1) / 3              |
/// | fused_op_prob      | [0.0, 0.8] | v / 0.8                  |
/// | max_junkify_passes | [1, 10]    | (v - 1) / 9              |
/// | junkify_prob       | [0.1, 1.0] | (v - 0.1) / 0.9          |
///
/// `max_junkify_passes` minimum is 1: configs with 0 passes can never apply
/// any junkify rewrites, producing `unoptimized == optimized` and empty
/// trajectories. Similarly `junkify_prob` minimum is 0.1 to ensure nodes
/// are visited with non-negligible probability.
#[must_use]
pub fn normalize(config: &BwdGenConfig) -> [f32; ES_DIM] {
    [
        (config.max_depth as f32 - 5.0) / 15.0,
        (config.leaf_prob - 0.05) / 0.75,
        (config.num_vars as f32 - 1.0) / 3.0,
        config.fused_op_prob / 0.8,
        (config.max_junkify_passes as f32 - 1.0) / 9.0,
        (config.junkify_prob - 0.1) / 0.9,
    ]
}

/// Denormalize [0,1]^6 back into a `BwdGenConfig`.
///
/// Clamps inputs to [0,1] before denormalizing, then rounds usize fields.
#[must_use]
pub fn denormalize(params: &[f32; ES_DIM]) -> BwdGenConfig {
    let p = [
        clamp01(params[0]),
        clamp01(params[1]),
        clamp01(params[2]),
        clamp01(params[3]),
        clamp01(params[4]),
        clamp01(params[5]),
    ];

    let max_depth_f = p[0] * 15.0 + 5.0;
    let leaf_prob = p[1] * 0.75 + 0.05;
    let num_vars_f = p[2] * 3.0 + 1.0;
    let fused_op_prob = p[3] * 0.8;
    // ES range for max_junkify_passes is [1, 10]: lower bound is 1, not 0.
    // A config with max_junkify_passes=0 can never apply any junkify rewrites,
    // so unoptimized == optimized and the e-graph produces empty trajectories.
    let max_junkify_f = p[4] * 9.0 + 1.0;
    // junkify_prob must stay above a floor so at least some nodes are visited.
    // ES range [0.1, 1.0] prevents the degenerate case of prob≈0 where
    // every node is skipped and rewrites_applied stays at 0.
    let junkify_prob = p[5] * 0.9 + 0.1;

    BwdGenConfig {
        max_depth: libm::roundf(max_depth_f) as usize,
        leaf_prob,
        num_vars: (libm::roundf(num_vars_f) as usize).min(4),
        fused_op_prob,
        max_junkify_passes: (libm::roundf(max_junkify_f) as usize).max(1),
        junkify_prob,
        max_junkified_nodes: 80, // Max *additional* nodes from junkification (relative growth cap)
    }
}

/// Convert nanoseconds to log-nanoseconds (floored at 1e-3ns, capped at 1s).
///
/// Panics if `ns` is NaN.
#[must_use]
pub fn log_ns(ns: f64) -> f32 {
    assert!(!ns.is_nan(), "log_ns called with NaN");
    let clamped = if ns < 1e-3 {
        1e-3
    } else if ns > 1e9 {
        1e9
    } else {
        ns
    };
    libm::logf(clamped as f32)
}

/// Clamp to [0, 1].
fn clamp01(v: f32) -> f32 {
    if v < 0.0 {
        0.0
    } else if v > 1.0 {
        1.0
    } else {
        v
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_denormalize_roundtrip_works() {
        let original = BwdGenConfig {
            max_depth: 10, // Within ES range [5, 20]
            ..BwdGenConfig::default()
        };
        let normed = normalize(&original);

        // All normalized values should be in [0, 1].
        for (i, &v) in normed.iter().enumerate() {
            assert!(
                v >= 0.0 && v <= 1.0,
                "normalized[{}] = {} out of [0,1]",
                i,
                v
            );
        }

        let recovered = denormalize(&normed);
        assert_eq!(
            recovered.max_depth, original.max_depth,
            "max_depth mismatch: got {}, expected {}",
            recovered.max_depth, original.max_depth
        );
        assert!(
            libm::fabsf(recovered.leaf_prob - original.leaf_prob) < 0.01,
            "leaf_prob mismatch: got {}, expected {}",
            recovered.leaf_prob,
            original.leaf_prob
        );
        assert_eq!(
            recovered.num_vars, original.num_vars,
            "num_vars mismatch: got {}, expected {}",
            recovered.num_vars, original.num_vars
        );
        assert!(
            libm::fabsf(recovered.fused_op_prob - original.fused_op_prob) < 0.01,
            "fused_op_prob mismatch: got {}, expected {}",
            recovered.fused_op_prob,
            original.fused_op_prob
        );
        assert_eq!(
            recovered.max_junkify_passes, original.max_junkify_passes,
            "max_junkify_passes mismatch: got {}, expected {}",
            recovered.max_junkify_passes, original.max_junkify_passes
        );
        assert!(
            libm::fabsf(recovered.junkify_prob - original.junkify_prob) < 0.01,
            "junkify_prob mismatch: got {}, expected {}",
            recovered.junkify_prob,
            original.junkify_prob
        );
    }

    #[test]
    fn denormalize_clamps_works() {
        // All below minimum → should clamp to range minimums.
        let below = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        let config = denormalize(&below);
        assert_eq!(
            config.max_depth, 5,
            "max_depth should clamp to 5, got {}",
            config.max_depth
        );
        assert!(
            libm::fabsf(config.leaf_prob - 0.05) < 0.01,
            "leaf_prob should clamp to 0.05, got {}",
            config.leaf_prob
        );
        assert_eq!(
            config.num_vars, 1,
            "num_vars should clamp to 1, got {}",
            config.num_vars
        );
        assert!(
            libm::fabsf(config.fused_op_prob) < 0.01,
            "fused_op_prob should clamp to 0.0, got {}",
            config.fused_op_prob
        );
        // min is 1 (not 0) — configs with 0 passes can never junkify
        assert_eq!(
            config.max_junkify_passes, 1,
            "max_junkify_passes should clamp to 1, got {}",
            config.max_junkify_passes
        );
        // min is 0.1 (not 0.0) — prob=0 means no nodes are ever visited
        assert!(
            libm::fabsf(config.junkify_prob - 0.1) < 0.01,
            "junkify_prob should clamp to 0.1, got {}",
            config.junkify_prob
        );

        // All above maximum → should clamp to range maximums.
        let above = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let config = denormalize(&above);
        assert_eq!(
            config.max_depth, 20,
            "max_depth should clamp to 20, got {}",
            config.max_depth
        );
        assert!(
            libm::fabsf(config.leaf_prob - 0.80) < 0.01,
            "leaf_prob should clamp to 0.80, got {}",
            config.leaf_prob
        );
        assert_eq!(
            config.num_vars, 4,
            "num_vars should clamp to 4, got {}",
            config.num_vars
        );
        assert!(
            libm::fabsf(config.fused_op_prob - 0.80) < 0.01,
            "fused_op_prob should clamp to 0.80, got {}",
            config.fused_op_prob
        );
        assert_eq!(
            config.max_junkify_passes, 10,
            "max_junkify_passes should clamp to 10, got {}",
            config.max_junkify_passes
        );
        assert!(
            libm::fabsf(config.junkify_prob - 1.0) < 0.01,
            "junkify_prob should clamp to 1.0, got {}",
            config.junkify_prob
        );
    }

    #[test]
    fn box_muller_distribution_works() {
        let mut es = GenEs {
            mu: [0.5; ES_DIM],
            sigma: 0.1,
            alpha: 0.05,
            population: 10,
            samples_per_candidate: 8,
            rng_state: 12345,
            last_mean_fitness: 0.0,
            templates: RuleTemplates::new(),
        };

        let n = 10_000;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;

        for _ in 0..n {
            let v = es.rand_normal() as f64;
            sum += v;
            sum_sq += v * v;
        }

        let mean = sum / n as f64;
        let variance = sum_sq / n as f64 - mean * mean;
        let std_dev = libm::sqrt(variance);

        assert!(
            libm::fabs(mean) < 0.1,
            "Box-Muller mean should be ~0, got {}",
            mean
        );
        assert!(
            libm::fabs(std_dev - 1.0) < 0.15,
            "Box-Muller std should be ~1.0, got {}",
            std_dev
        );
    }

    #[test]
    fn log_ns_works() {
        // log(1.0) = 0.0
        let v = log_ns(1.0);
        assert!(
            libm::fabsf(v) < 0.001,
            "log_ns(1.0) should be ~0.0, got {}",
            v
        );

        // Values below 1e-3 should be floored to 1e-3.
        let v_low = log_ns(0.0001);
        let v_floor = log_ns(1e-3);
        assert!(
            libm::fabsf(v_low - v_floor) < 0.001,
            "log_ns(0.0001) should equal log_ns(1e-3), got {} vs {}",
            v_low,
            v_floor
        );

        // log(e) ≈ 1.0
        let v_e = log_ns(core::f64::consts::E);
        assert!(
            libm::fabsf(v_e - 1.0) < 0.01,
            "log_ns(e) should be ~1.0, got {}",
            v_e
        );
    }

    #[test]
    fn clamp01_works() {
        assert_eq!(clamp01(-0.5), 0.0);
        assert_eq!(clamp01(0.0), 0.0);
        assert_eq!(clamp01(0.5), 0.5);
        assert_eq!(clamp01(1.0), 1.0);
        assert_eq!(clamp01(1.5), 1.0);
    }

    #[test]
    fn gen_es_new_initializes_from_defaults_works() {
        let es = GenEs::new(GenEsConfig::default(), RuleTemplates::new());

        // mu should be the normalized defaults.
        let expected = normalize(&BwdGenConfig::default());
        for (i, (&got, &exp)) in es.mu.iter().zip(expected.iter()).enumerate() {
            assert!(
                libm::fabsf(got - exp) < 1e-6,
                "mu[{}] mismatch: got {}, expected {}",
                i,
                got,
                exp
            );
        }

        assert_eq!(es.last_fitness(), 0.0);
    }

    #[test]
    fn rand_normal_no_nan_with_interleaved_rng_works() {
        // Reproduce the actual training seed and interleave rand_u64 calls
        // between rand_normal batches (simulating evaluate_candidate flow).
        let mut es = GenEs {
            mu: [0.5; ES_DIM],
            sigma: 0.1,
            alpha: 0.05,
            population: 10,
            samples_per_candidate: 8,
            rng_state: 42u64.wrapping_add(0xE5), // actual training seed
            last_mean_fitness: 0.0,
            templates: RuleTemplates::new(),
        };

        for round in 0..10 {
            // Sample 6 normals (one epsilon vector)
            for d in 0..ES_DIM {
                let v = es.rand_normal() * es.sigma;
                assert!(
                    v.is_finite(),
                    "NaN/Inf from rand_normal at round={round}, dim={d}, rng_state={}",
                    es.rng_state
                );
            }
            // Simulate evaluate_candidate: consume some RNG states
            for _ in 0..20 {
                let _ = es.rand_u64();
            }
        }
    }
}
