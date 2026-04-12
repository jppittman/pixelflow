//! # Training Data Generation for NNUE Instruction Selection
//!
//! This module provides tools for generating training data, inspired by
//! Stockfish's NNUE training pipeline:
//!
//! 1. **Expression Generation**: Generate random expressions (like random positions)
//! 2. **Rewrite Enumeration**: Find all valid rewrites (like legal moves)
//! 3. **Benchmarking**: Measure actual execution cost (like deep search evaluation)
//! 4. **Data Recording**: Store (features, cost) pairs for training
//!
//! ## Usage
//!
//! ```ignore
//! use pixelflow_ml::training::{DataGenerator, DataGenConfig};
//!
//! let mut generator = DataGenerator::new(DataGenConfig::default());
//! let samples = generator.generate_batch(1000);
//!
//! // Write to binpack format (like Stockfish's training data)
//! generator.write_binpack("training_data.bin", &samples)?;
//! ```
//!
//! ## Binpack Format
//!
//! We use a simple binary format inspired by Stockfish's binpack:
//!
//! ```text
//! Header (16 bytes):
//!   - Magic: 0x4E4E5545 ("NNUE")
//!   - Version: u32
//!   - Sample count: u64
//!
//! Per sample:
//!   - Feature count: u16
//!   - Features: [u32; feature_count] (packed feature indices)
//!   - Cost (nanoseconds): u64
//!   - Best rewrite index: u16 (which rewrite rule was best)
//! ```

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use crate::nnue::{
    Expr, ExprGenConfig, ExprGenerator, HalfEPFeature, OpType, RewriteRule, extract_features,
    find_all_rewrites,
};

// ============================================================================
// Benchmarking (requires std)
// ============================================================================

/// Configuration for benchmarking expressions.
#[derive(Clone, Debug)]
pub struct BenchConfig {
    /// Number of iterations for warm-up.
    pub warmup_iters: usize,
    /// Number of iterations for measurement.
    pub measure_iters: usize,
    /// Number of SIMD lanes to simulate (for cost estimation).
    pub simd_width: usize,
    /// Random test points for evaluation.
    pub num_test_points: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            warmup_iters: 100,
            measure_iters: 1000,
            simd_width: 8, // AVX2
            num_test_points: 64,
        }
    }
}

/// Result of benchmarking an expression.
#[derive(Clone, Debug)]
pub struct BenchResult {
    /// Median execution time in nanoseconds.
    pub median_ns: u64,
    /// Mean execution time in nanoseconds.
    pub mean_ns: u64,
    /// Minimum execution time in nanoseconds.
    pub min_ns: u64,
    /// Maximum execution time in nanoseconds.
    pub max_ns: u64,
    /// Standard deviation in nanoseconds.
    pub std_ns: u64,
}

/// Benchmark an expression's evaluation cost.
///
/// This simulates what the expression would cost when compiled to SIMD code
/// by running many evaluations and measuring time.
#[cfg(feature = "std")]
pub fn benchmark_expr(expr: &Expr, config: &BenchConfig) -> BenchResult {
    use std::time::Instant;

    // Generate test points
    let mut rng_state = 12345u64;
    let mut test_points: Vec<[f32; 4]> = Vec::with_capacity(config.num_test_points);
    for _ in 0..config.num_test_points {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = (rng_state >> 40) as f32 / 256.0 - 0.5;
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let y = (rng_state >> 40) as f32 / 256.0 - 0.5;
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let z = (rng_state >> 40) as f32 / 256.0 - 0.5;
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let w = (rng_state >> 40) as f32 / 256.0 - 0.5;
        test_points.push([x, y, z, w]);
    }

    // Warm-up
    let mut sink = 0.0f32;
    for _ in 0..config.warmup_iters {
        for point in &test_points {
            sink += expr.eval(point);
        }
    }

    // Measurement
    let mut times: Vec<u64> = Vec::with_capacity(config.measure_iters);
    for _ in 0..config.measure_iters {
        let start = Instant::now();
        for point in &test_points {
            sink += expr.eval(point);
        }
        let elapsed = start.elapsed().as_nanos() as u64;
        times.push(elapsed);
    }

    // Prevent dead code elimination
    std::hint::black_box(sink);

    // Compute statistics
    times.sort_unstable();
    let median_ns = times[times.len() / 2];
    let mean_ns = times.iter().sum::<u64>() / times.len() as u64;
    let min_ns = times[0];
    let max_ns = times[times.len() - 1];

    let variance = times
        .iter()
        .map(|&t| {
            let diff = t as i64 - mean_ns as i64;
            (diff * diff) as u64
        })
        .sum::<u64>()
        / times.len() as u64;
    let std_ns = (variance as f64).sqrt() as u64;

    BenchResult {
        median_ns,
        mean_ns,
        min_ns,
        max_ns,
        std_ns,
    }
}

/// Estimate cost without benchmarking (fast, approximate).
///
/// Uses the static cost model to estimate expression cost.
pub fn estimate_cost(expr: &Expr) -> usize {
    match expr {
        Expr::Var(_) | Expr::Const(_) => 0,
        Expr::Unary(op, a) => {
            let op_cost = match op {
                OpType::Neg => 1,
                OpType::Abs => 1,
                OpType::Sqrt => 15,
                OpType::Rsqrt => 5,
                _ => 5,
            };
            op_cost + estimate_cost(a)
        }
        Expr::Binary(op, a, b) => {
            let op_cost = match op {
                OpType::Add | OpType::Sub => 4,
                OpType::Mul => 5,
                OpType::Div => 15,
                OpType::Min | OpType::Max => 4,
                OpType::MulRsqrt => 6,
                _ => 5,
            };
            op_cost + estimate_cost(a) + estimate_cost(b)
        }
        Expr::Ternary(op, a, b, c) => {
            let op_cost = match op {
                OpType::MulAdd => 5,
                _ => 10,
            };
            op_cost + estimate_cost(a) + estimate_cost(b) + estimate_cost(c)
        }
    }
}

// ============================================================================
// Training Data Structures
// ============================================================================

/// A training sample for NNUE.
#[derive(Clone, Debug)]
pub struct TrainingSample {
    /// Packed feature indices (sorted for consistency).
    pub features: Vec<u32>,
    /// Execution cost in estimated cycles.
    pub cost: u64,
    /// Index of the best rewrite found (or u16::MAX if none better).
    pub best_rewrite: u16,
    /// Cost improvement from best rewrite (negative = worse).
    pub cost_delta: i64,
}

impl TrainingSample {
    /// Create a new training sample.
    pub fn new(features: Vec<HalfEPFeature>, cost: u64) -> Self {
        let mut packed: Vec<u32> = features.iter().map(|f| f.to_index() as u32).collect();
        packed.sort_unstable();
        packed.dedup();

        Self {
            features: packed,
            cost,
            best_rewrite: u16::MAX,
            cost_delta: 0,
        }
    }
}

// ============================================================================
// Data Generation
// ============================================================================

/// Configuration for data generation.
#[derive(Clone, Debug)]
pub struct DataGenConfig {
    /// Expression generation config.
    pub expr_config: ExprGenConfig,
    /// Benchmarking config.
    pub bench_config: BenchConfig,
    /// Whether to use actual benchmarking (slow) or estimation (fast).
    pub use_benchmarking: bool,
    /// Minimum expression depth to include.
    pub min_depth: usize,
    /// Maximum rewrites to evaluate per expression.
    pub max_rewrites_per_expr: usize,
}

impl Default for DataGenConfig {
    fn default() -> Self {
        Self {
            expr_config: ExprGenConfig::default(),
            bench_config: BenchConfig::default(),
            use_benchmarking: false, // Use estimation by default
            min_depth: 2,
            max_rewrites_per_expr: 100,
        }
    }
}

/// Training data generator.
///
/// This is analogous to Stockfish's self-play data generator.
pub struct DataGenerator {
    /// Configuration.
    pub config: DataGenConfig,
    /// Expression generator.
    expr_gen: ExprGenerator,
    /// Statistics.
    pub stats: DataGenStats,
}

/// Statistics about data generation.
#[derive(Clone, Debug, Default)]
pub struct DataGenStats {
    /// Total expressions generated.
    pub total_exprs: usize,
    /// Expressions with valid rewrites.
    pub exprs_with_rewrites: usize,
    /// Total rewrites found.
    pub total_rewrites: usize,
    /// Rewrites that improved cost.
    pub improving_rewrites: usize,
}

impl DataGenerator {
    /// Create a new data generator.
    pub fn new(seed: u64, config: DataGenConfig) -> Self {
        Self {
            expr_gen: ExprGenerator::new(seed, config.expr_config.clone()),
            config,
            stats: DataGenStats::default(),
        }
    }

    /// Generate a single training sample.
    pub fn generate_sample(&mut self) -> Option<TrainingSample> {
        // Generate expression
        let expr = self.expr_gen.generate();
        self.stats.total_exprs += 1;

        // Skip if too shallow
        if expr.depth() < self.config.min_depth {
            return None;
        }

        // Extract features
        let features = extract_features(&expr);

        // Estimate or benchmark cost
        let base_cost = estimate_cost(&expr) as u64;

        // Find rewrites
        let rewrites = find_all_rewrites(&expr);
        self.stats.total_rewrites += rewrites.len();

        if !rewrites.is_empty() {
            self.stats.exprs_with_rewrites += 1;
        }

        // Create sample
        let mut sample = TrainingSample::new(features, base_cost);

        // Evaluate rewrites to find best
        let mut best_cost = base_cost;
        let mut best_idx = u16::MAX;

        for (i, (_path, _rule, rewritten)) in rewrites.iter().enumerate() {
            if i >= self.config.max_rewrites_per_expr {
                break;
            }

            let rewrite_cost = estimate_cost(rewritten) as u64;

            if rewrite_cost < best_cost {
                best_cost = rewrite_cost;
                best_idx = i as u16;
                self.stats.improving_rewrites += 1;
            }
        }

        sample.best_rewrite = best_idx;
        sample.cost_delta = best_cost as i64 - base_cost as i64;

        Some(sample)
    }

    /// Generate a batch of training samples.
    pub fn generate_batch(&mut self, count: usize) -> Vec<TrainingSample> {
        let mut samples = Vec::with_capacity(count);
        let mut attempts = 0;
        let max_attempts = count * 10; // Avoid infinite loop

        while samples.len() < count && attempts < max_attempts {
            attempts += 1;
            if let Some(sample) = self.generate_sample() {
                samples.push(sample);
            }
        }

        samples
    }
}

// ============================================================================
// Binary Format I/O
// ============================================================================

/// Magic number for binpack files.
pub const BINPACK_MAGIC: u32 = 0x4E4E5545; // "NNUE"

/// Version of the binpack format.
pub const BINPACK_VERSION: u32 = 1;

/// Write samples to binpack format.
#[cfg(feature = "std")]
pub fn write_binpack(path: &str, samples: &[TrainingSample]) -> std::io::Result<()> {
    use std::io::Write;

    let mut file = std::fs::File::create(path)?;

    // Write header
    file.write_all(&BINPACK_MAGIC.to_le_bytes())?;
    file.write_all(&BINPACK_VERSION.to_le_bytes())?;
    file.write_all(&(samples.len() as u64).to_le_bytes())?;

    // Write samples
    for sample in samples {
        // Feature count
        file.write_all(&(sample.features.len() as u16).to_le_bytes())?;

        // Features
        for &f in &sample.features {
            file.write_all(&f.to_le_bytes())?;
        }

        // Cost
        file.write_all(&sample.cost.to_le_bytes())?;

        // Best rewrite
        file.write_all(&sample.best_rewrite.to_le_bytes())?;

        // Cost delta
        file.write_all(&sample.cost_delta.to_le_bytes())?;
    }

    Ok(())
}

/// Read samples from binpack format.
#[cfg(feature = "std")]
pub fn read_binpack(path: &str) -> std::io::Result<Vec<TrainingSample>> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;

    // Read header
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if u32::from_le_bytes(magic) != BINPACK_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid magic number",
        ));
    }

    let mut version = [0u8; 4];
    file.read_exact(&mut version)?;
    if u32::from_le_bytes(version) != BINPACK_VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Unsupported version",
        ));
    }

    let mut count = [0u8; 8];
    file.read_exact(&mut count)?;
    let sample_count = u64::from_le_bytes(count) as usize;

    // Read samples
    let mut samples = Vec::with_capacity(sample_count);

    for _ in 0..sample_count {
        // Feature count
        let mut fc = [0u8; 2];
        file.read_exact(&mut fc)?;
        let feature_count = u16::from_le_bytes(fc) as usize;

        // Features
        let mut features = Vec::with_capacity(feature_count);
        for _ in 0..feature_count {
            let mut f = [0u8; 4];
            file.read_exact(&mut f)?;
            features.push(u32::from_le_bytes(f));
        }

        // Cost
        let mut c = [0u8; 8];
        file.read_exact(&mut c)?;
        let cost = u64::from_le_bytes(c);

        // Best rewrite
        let mut br = [0u8; 2];
        file.read_exact(&mut br)?;
        let best_rewrite = u16::from_le_bytes(br);

        // Cost delta
        let mut cd = [0u8; 8];
        file.read_exact(&mut cd)?;
        let cost_delta = i64::from_le_bytes(cd);

        samples.push(TrainingSample {
            features,
            cost,
            best_rewrite,
            cost_delta,
        });
    }

    Ok(samples)
}

// ============================================================================
// Training Data Statistics
// ============================================================================

/// Compute statistics about a training dataset.
#[derive(Clone, Debug)]
pub struct DatasetStats {
    /// Number of samples.
    pub sample_count: usize,
    /// Average feature count per sample.
    pub avg_features: f64,
    /// Average cost.
    pub avg_cost: f64,
    /// Samples with improving rewrites.
    pub samples_with_improvement: usize,
    /// Average cost improvement when there is one.
    pub avg_improvement: f64,
}

impl DatasetStats {
    /// Compute statistics from a dataset.
    pub fn from_samples(samples: &[TrainingSample]) -> Self {
        if samples.is_empty() {
            return Self {
                sample_count: 0,
                avg_features: 0.0,
                avg_cost: 0.0,
                samples_with_improvement: 0,
                avg_improvement: 0.0,
            };
        }

        let total_features: usize = samples.iter().map(|s| s.features.len()).sum();
        let total_cost: u64 = samples.iter().map(|s| s.cost).sum();

        let improving: Vec<_> = samples.iter().filter(|s| s.cost_delta < 0).collect();

        let total_improvement: i64 = improving.iter().map(|s| -s.cost_delta).sum();

        Self {
            sample_count: samples.len(),
            avg_features: total_features as f64 / samples.len() as f64,
            avg_cost: total_cost as f64 / samples.len() as f64,
            samples_with_improvement: improving.len(),
            avg_improvement: if improving.is_empty() {
                0.0
            } else {
                total_improvement as f64 / improving.len() as f64
            },
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_cost_should_succeed_when_called() {
        // Simple add: x + y
        let expr = Expr::Binary(
            OpType::Add,
            alloc::boxed::Box::new(Expr::Var(0)),
            alloc::boxed::Box::new(Expr::Var(1)),
        );
        let cost = estimate_cost(&expr);
        assert_eq!(cost, 4); // Add cost

        // MulAdd: x * y + z
        let fma = Expr::Ternary(
            OpType::MulAdd,
            alloc::boxed::Box::new(Expr::Var(0)),
            alloc::boxed::Box::new(Expr::Var(1)),
            alloc::boxed::Box::new(Expr::Var(2)),
        );
        let fma_cost = estimate_cost(&fma);
        assert_eq!(fma_cost, 5); // MulAdd cost
    }

    #[test]
    fn training_sample_dedup_should_succeed_when_called() {
        let features = vec![
            HalfEPFeature {
                perspective_op: 0,
                descendant_op: 1,
                depth: 0,
                path: 0,
            },
            HalfEPFeature {
                perspective_op: 0,
                descendant_op: 1,
                depth: 0,
                path: 0,
            }, // duplicate
            HalfEPFeature {
                perspective_op: 0,
                descendant_op: 2,
                depth: 0,
                path: 0,
            },
        ];
        let sample = TrainingSample::new(features, 100);
        assert_eq!(sample.features.len(), 2); // Duplicates removed
    }

    #[test]
    fn data_generator_should_succeed_when_called() {
        let mut generator = DataGenerator::new(42, DataGenConfig::default());
        let samples = generator.generate_batch(10);
        assert!(!samples.is_empty());
        assert!(generator.stats.total_exprs > 0);
    }

    #[test]
    fn dataset_stats_should_succeed_when_called() {
        let samples = vec![
            TrainingSample {
                features: vec![1, 2, 3],
                cost: 100,
                best_rewrite: 0,
                cost_delta: -10,
            },
            TrainingSample {
                features: vec![4, 5],
                cost: 200,
                best_rewrite: u16::MAX,
                cost_delta: 0,
            },
        ];

        let stats = DatasetStats::from_samples(&samples);
        assert_eq!(stats.sample_count, 2);
        assert!((stats.avg_features - 2.5).abs() < 0.01);
        assert!((stats.avg_cost - 150.0).abs() < 0.01);
        assert_eq!(stats.samples_with_improvement, 1);
    }
}
