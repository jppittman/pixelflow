//! Cost model for e-graph extraction.
//!
//! The cost model controls which equivalent expression the e-graph extracts.
//! It uses `OpKind` from `pixelflow-ir` as the canonical operation enumeration.
//!
//! # Architecture
//!
//! The module provides two levels of abstraction:
//!
//! 1. **`CostFunction` trait**: Pluggable interface for any cost estimator
//! 2. **`CostModel` struct**: Hardcoded O(1) lookup table based on OpKind
//!
//! This allows the e-graph extraction to use either:
//! - Fast hardcoded costs (`CostModel`)
//! - Learned neural costs (`ExprNnue` from pixelflow-nnue)
//! - Custom domain-specific cost models

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use super::node::ENode;
use pixelflow_ir::OpKind;

// ============================================================================
// Latency Prior — single source of truth
// ============================================================================

/// Handcrafted per-op cycle-latency estimates, indexed by `OpKind::index()`.
///
/// This is the ONE place these numbers are allowed to live. Both the static
/// [`CostModel`] (via [`CostModel::latency_prior`]) and the NNUE's embedding
/// initialization (`nnue::factored::OpEmbeddings::init_with_latency_prior`)
/// derive their costs from this table so the two representations cannot
/// drift apart. If you're tempted to hand-tune a number in one place,
/// change it here instead.
///
/// Values are approximate cycle counts on typical x86_64/AArch64 SIMD
/// hardware; refine as real measurements come in (see `calibrate_costs`).
pub const LATENCY_PRIOR_CYCLES: [usize; OpKind::COUNT] = [
    0,  // Var - free
    0,  // Const - free
    4,  // Add
    4,  // Sub
    5,  // Mul
    15, // Div
    1,  // Neg
    15, // Sqrt
    5,  // Rsqrt - fast approximation
    1,  // Abs
    4,  // Min
    4,  // Max
    5,  // MulAdd - fused
    10, // Recip
    4,  // Floor
    4,  // Ceil
    4,  // Round
    4,  // Fract
    10, // Sin
    10, // Cos
    10, // Tan
    10, // Asin
    10, // Acos
    10, // Atan
    10, // Exp
    10, // Exp2
    10, // Ln
    10, // Log2
    10, // Log10
    10, // Atan2
    12, // Pow
    8,  // Hypot
    3,  // Lt
    3,  // Le
    3,  // Gt
    3,  // Ge
    3,  // Eq
    3,  // Ne
    4,  // Select
    6,  // Clamp - 2x compare + select
    0,  // Tuple - free (structural)
    // Bit-manip primitives: single cheap integer/convert instructions.
    1, // TruncToInt - cvttps2dq
    1, // IntToFloat - cvtdq2ps
    1, // IAdd - paddd
    1, // Shl
    1, // Shr
    1, // BitAnd
    1, // BitOr
    // Dwrt - rewritten away by the e-graph (chain rule); never emitted.
    // Extraction must never choose a surviving one, so it gets a
    // prohibitive (but finite, non-saturating) cost here. `node_op_cost`
    // additionally hard-codes usize::MAX/4 as a belt-and-suspenders guard.
    1000, // Dwrt
    0,    // Buffer - leaf, free
    10,   // Gather - memory read
    10,   // RawGather - primitive memory read
    0,    // Reduce - lowered (unrolled) before costing
];

// ============================================================================
// Cost Function Trait
// ============================================================================

/// Trait for pluggable cost functions in e-graph extraction.
///
/// Implementors provide a cost estimate for ENodes, enabling different
/// cost models (hardcoded, learned neural, domain-specific) to be used
/// interchangeably during extraction.
///
/// # Contract
///
/// - `node_cost` returns a cost in arbitrary units (lower is better)
/// - Leaves (Var, Const) should typically return 0
/// - Costs should be consistent: same input → same output
///
/// # Example
///
/// ```ignore
/// // Using the hardcoded cost model
/// let costs = CostModel::default();
/// let (tree, cost) = extract(&egraph, root, &costs);
///
/// // Using a learned neural cost model
/// let nnue = ExprNnue::load("model.bin")?;
/// let (tree, cost) = extract(&egraph, root, &nnue);
/// ```
pub trait CostFunction {
    /// Estimate the cost of a single ENode given its parent context.
    ///
    /// This is the atomic operation cost, NOT including children.
    /// The extraction algorithm sums child costs separately.
    ///
    /// `parent` is the OpKind of the operation using this result.
    /// This allows for 'sliding window' optimizations (e.g. FMA detection).
    fn node_cost(&self, node: &ENode, parent: Option<OpKind>) -> usize;

    /// Get the cost of an operation by OpKind (optional, for interop).
    fn cost_by_kind(&self, op: OpKind, parent: Option<OpKind>) -> usize {
        panic!("CostFunction::cost_by_kind not implemented");
    }
}

/// Cost model indexed by OpKind.
///
/// Uses `[usize; OpKind::COUNT]` internally for O(1) lookup.
/// Includes depth penalty for compile-time optimization.
#[derive(Clone, Debug)]
pub struct CostModel {
    /// Cost per operation, indexed by `OpKind::index()`.
    costs: [usize; OpKind::COUNT],
    /// Depth at which to start applying penalties.
    pub depth_threshold: usize,
    /// Penalty per depth level beyond threshold.
    pub depth_penalty: usize,
}

impl Default for CostModel {
    fn default() -> Self {
        Self::new()
    }
}

impl CostModel {
    /// Create a cost model seeded with the handcrafted latency-prior cycle
    /// table ([`LATENCY_PRIOR_CYCLES`]).
    ///
    /// This is the default: an all-zero cost model makes every expression
    /// "free" and extraction degenerates to an arbitrary tie-break, so
    /// zero-cost is never a useful baseline for real extraction. Use
    /// [`CostModel::zero`] explicitly if you actually want all-zero costs
    /// (e.g. to test structural properties independent of cost).
    pub fn new() -> Self {
        Self::latency_prior()
    }

    /// Create a cost model from the handcrafted latency-prior cycle table.
    ///
    /// Source of truth: [`LATENCY_PRIOR_CYCLES`], shared with
    /// `nnue::factored::OpEmbeddings::init_with_latency_prior` so the static
    /// and learned cost models cannot drift apart.
    pub fn latency_prior() -> Self {
        Self {
            costs: LATENCY_PRIOR_CYCLES,
            depth_threshold: 1024, // Effectively disabled
            depth_penalty: 0,
        }
    }

    /// Create an all-zero cost model.
    ///
    /// Every expression costs nothing, so extraction can't distinguish
    /// equivalent forms on cost alone. Only useful for tests that check
    /// structural extraction behavior (DAG sharing, cycle handling, etc.)
    /// independent of any particular cost table.
    pub fn zero() -> Self {
        Self {
            costs: [0usize; OpKind::COUNT],
            depth_threshold: 1024, // Effectively disabled
            depth_penalty: 0,
        }
    }

    /// Create with aggressive depth penalty for complex kernels.
    pub fn shallow() -> Self {
        Self {
            depth_threshold: 16,
            depth_penalty: 500,
            ..Self::new()
        }
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get cost for an OpKind.
    #[inline]
    pub fn cost(&self, op: OpKind) -> usize {
        self.costs[op.index()]
    }

    /// Set cost for an OpKind.
    #[inline]
    pub fn set_cost(&mut self, op: OpKind, cost: usize) {
        self.costs[op.index()] = cost;
    }

    /// Get the raw costs array.
    pub fn costs(&self) -> &[usize; OpKind::COUNT] {
        &self.costs
    }

    /// Get mutable reference to costs array.
    pub fn costs_mut(&mut self) -> &mut [usize; OpKind::COUNT] {
        &mut self.costs
    }

    /// Calculate the hinge penalty for a given depth.
    #[inline]
    pub fn depth_cost(&self, depth: usize) -> usize {
        if depth > self.depth_threshold {
            (depth - self.depth_threshold) * self.depth_penalty
        } else {
            0
        }
    }

    /// Get cost for an ENode.
    ///
    /// Uses `op.kind()` to convert at the boundary from `&dyn Op` to `OpKind`.
    pub fn node_op_cost(&self, node: &ENode) -> usize {
        match node {
            ENode::Var(_) | ENode::Const(_) => 0,
            // `Dwrt` is the internal autodiff marker. It is rewritten away by
            // the chain rule; a surviving one is the (not-yet-wired) jet
            // fallback. Either way extraction must never choose it, so it is
            // prohibitively expensive regardless of the learned weight table.
            ENode::Op { op, .. } if op.kind() == OpKind::Dwrt => usize::MAX / 4,
            ENode::Op { op, .. } => self.cost(op.kind()),
        }
    }

    /// Get cost by operation name (for backward compatibility).
    ///
    /// # Panics
    ///
    /// Panics if `name` does not match a known `OpKind`. Silently mapping
    /// an unrecognized name to `Add`'s cost would let typos and stale
    /// callers pass through with a wrong-but-plausible number — fail loud
    /// instead.
    pub fn cost_by_name(&self, name: &str) -> usize {
        // Map name to OpKind
        let op = match name {
            "var" => OpKind::Var,
            "const" => OpKind::Const,
            "add" => OpKind::Add,
            "sub" => OpKind::Sub,
            "mul" => OpKind::Mul,
            "div" => OpKind::Div,
            "neg" => OpKind::Neg,
            "sqrt" => OpKind::Sqrt,
            "rsqrt" => OpKind::Rsqrt,
            "abs" => OpKind::Abs,
            "min" => OpKind::Min,
            "max" => OpKind::Max,
            "mul_add" => OpKind::MulAdd,
            "recip" => OpKind::Recip,
            "floor" => OpKind::Floor,
            "ceil" => OpKind::Ceil,
            "round" => OpKind::Round,
            "fract" => OpKind::Fract,
            "sin" => OpKind::Sin,
            "cos" => OpKind::Cos,
            "tan" => OpKind::Tan,
            "asin" => OpKind::Asin,
            "acos" => OpKind::Acos,
            "atan" => OpKind::Atan,
            "atan2" => OpKind::Atan2,
            "exp" => OpKind::Exp,
            "exp2" => OpKind::Exp2,
            "ln" => OpKind::Ln,
            "log2" => OpKind::Log2,
            "log10" => OpKind::Log10,
            "pow" => OpKind::Pow,
            "hypot" => OpKind::Hypot,
            "lt" => OpKind::Lt,
            "le" => OpKind::Le,
            "gt" => OpKind::Gt,
            "ge" => OpKind::Ge,
            "eq" => OpKind::Eq,
            "ne" => OpKind::Ne,
            "select" => OpKind::Select,
            "clamp" => OpKind::Clamp,
            "tuple" => OpKind::Tuple,
            _ => panic!("CostModel::cost_by_name: unknown op name {name:?}"),
        };
        self.costs[op.index()]
    }

    // =========================================================================
    // Persistence
    // =========================================================================

    /// Save cost model to a TOML file.
    pub fn save_toml<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut contents = String::from("# Learned cost model weights\n");
        contents.push_str("# Generated from SIMD benchmark measurements\n\n");

        for i in 0..OpKind::COUNT {
            if let Some(op) = OpKind::from_index(i) {
                contents.push_str(&format!("{} = {}\n", op.name(), self.costs[i]));
            }
        }

        contents.push_str(&format!("\ndepth_threshold = {}\n", self.depth_threshold));
        contents.push_str(&format!("depth_penalty = {}\n", self.depth_penalty));

        fs::write(path, contents)
    }

    /// Load cost model from a TOML file.
    ///
    /// Starts from [`CostModel::zero`] (not [`CostModel::latency_prior`]) so
    /// the returned model reflects exactly what's in the file — mixing in
    /// the latency prior for keys the file doesn't mention would silently
    /// blend two cost sources together.
    ///
    /// # Errors
    ///
    /// Returns an error if the file can't be read, or if a `key = value`
    /// line has a value that fails to parse as `usize`. A malformed line is
    /// a real misconfiguration; silently skipping it would let a typo'd
    /// weight file produce a model that looks valid but isn't.
    pub fn load_toml<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let contents = fs::read_to_string(path)?;
        let mut model = Self::zero();

        for (lineno, line) in contents.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let Some((key, value)) = line.split_once('=') else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("cost model TOML line {}: missing '=': {line:?}", lineno + 1),
                ));
            };
            let key = key.trim();
            let value = value.trim();
            let v = value.parse::<usize>().map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "cost model TOML line {}: value {value:?} for key {key:?} is not a usize: {e}",
                        lineno + 1
                    ),
                )
            })?;

            if key == "depth_threshold" {
                model.depth_threshold = v;
            } else if key == "depth_penalty" {
                model.depth_penalty = v;
            } else if let Some(op) = OpKind::from_name(key) {
                // O(1) lookup via OpKind::from_name
                model.costs[op.index()] = v;
            } else {
                // Unrecognized op name: this is an external, evolving file
                // format (older/newer OpKind sets), so we don't hard-fail —
                // but we don't stay silent either.
                eprintln!(
                    "warning: cost model TOML line {}: unknown key {key:?}, ignoring",
                    lineno + 1
                );
            }
        }

        Ok(model)
    }

    /// Try to load from a standard location, falling back to the latency
    /// prior if none is found.
    ///
    /// Each candidate location is tried in order. A location that simply
    /// doesn't exist is expected (most of these are optional overrides) and
    /// is skipped quietly; a location that exists but fails to *parse* is a
    /// real misconfiguration and is reported loudly (`eprintln!`) before
    /// moving on, so a typo'd cost file never fails silently into a
    /// different cost model without a trace.
    pub fn load_or_default() -> Self {
        // Check environment variable first. If the user explicitly set
        // PIXELFLOW_COST_MODEL, a missing/unparsable file is always loud —
        // they asked for this specific file.
        if let Ok(path) = std::env::var("PIXELFLOW_COST_MODEL") {
            match Self::load_toml(&path) {
                Ok(model) => return model,
                Err(e) => eprintln!(
                    "warning: PIXELFLOW_COST_MODEL={path:?} failed to load ({e}); falling back"
                ),
            }
        }

        // Try user config directory.
        if let Some(home) = std::env::var_os("HOME") {
            let config_path = Path::new(&home).join(".config/pixelflow/cost_model.toml");
            match Self::load_toml(&config_path) {
                Ok(model) => return model,
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                    eprintln!(
                        "warning: cost model {config_path:?} exists but failed to load ({e}); falling back"
                    );
                }
                Err(_) => {} // not found: expected, this is an optional override
            }
        }

        // Try workspace data directory (for development).
        let workspace_paths = [
            "pixelflow-ml/data/learned_cost_model.toml",
            "../pixelflow-ml/data/learned_cost_model.toml",
        ];
        for path in workspace_paths {
            match Self::load_toml(path) {
                Ok(model) => return model,
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                    eprintln!(
                        "warning: cost model {path:?} exists but failed to load ({e}); falling back"
                    );
                }
                Err(_) => {} // not found: expected, this is an optional override
            }
        }

        // No override found anywhere: use the handcrafted latency prior.
        // This is a loud, intentional default — NOT the old all-zero
        // fallback, which made every op "free" and was useless for
        // extraction. If you need a genuinely all-zero model, use
        // `CostModel::zero()` explicitly.
        Self::latency_prior()
    }

    // =========================================================================
    // Interop
    // =========================================================================

    /// Create from HashMap (for backward compatibility).
    ///
    /// Starts from [`CostModel::zero`], not [`CostModel::latency_prior`] —
    /// same reasoning as [`CostModel::load_toml`]: the caller handed us an
    /// explicit map, so the result should reflect exactly that map, not a
    /// blend with the handcrafted prior.
    pub fn from_map(costs: &HashMap<String, usize>) -> Self {
        let mut model = Self::zero();
        for (key, &value) in costs {
            if key == "depth_threshold" {
                model.depth_threshold = value;
            } else if key == "depth_penalty" {
                model.depth_penalty = value;
            } else if let Some(op) = OpKind::from_name(key) {
                // O(1) lookup via OpKind::from_name
                model.costs[op.index()] = value;
            }
            // Unknown keys are silently ignored (external data format)
        }
        model
    }

    /// Convert to HashMap for interop.
    pub fn to_map(&self) -> HashMap<String, usize> {
        let mut map = HashMap::new();
        for i in 0..OpKind::COUNT {
            if let Some(op) = OpKind::from_index(i) {
                map.insert(op.name().to_string(), self.costs[i]);
            }
        }
        map.insert("depth_threshold".to_string(), self.depth_threshold);
        map.insert("depth_penalty".to_string(), self.depth_penalty);
        map
    }
}

// ============================================================================
// CostFunction Implementation for CostModel
// ============================================================================

impl CostFunction for CostModel {
    fn node_cost(&self, node: &ENode, _parent: Option<OpKind>) -> usize {
        self.node_op_cost(node)
    }

    fn cost_by_kind(&self, op: OpKind, _parent: Option<OpKind>) -> usize {
        self.cost(op)
    }
}
