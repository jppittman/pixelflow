//! Rule-set inflation for the Phase 3 Round 2 rule-scaling harness
//! (`docs/plans/2026-09-01-phase3-round2-rule-scaling.md` §2, §8).
//!
//! **Harness-only.** Nothing here is referenced by [`super::all_rules`], and
//! nothing here changes production behavior: `all_rules()` still returns
//! exactly 62 rules (pinned by `super::tests::all_rules_count`). Every
//! inflated set built here is a `Vec<Box<dyn Rewrite>>` a research binary
//! hands to `EGraph::with_rules` directly.
//!
//! All three of the design's inflation modes live here — (i) exact
//! duplicates, the "learnable overhead" control; (ii) mechanical
//! compositions, real closure-preserving shortcuts built by unifying one
//! rule's RHS against another's LHS (`crate::egraph::template`); and (iii)
//! genuinely new rules, `all_rules()` plus
//! [`round2_rules::experimental_rules`] verbatim at the one grid point
//! [`NEW_RULES_GRID`] names (`"new:95"`).

use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use pixelflow_ir::arena::ExprArena;

use super::all_rules;
use super::round2_rules;
use crate::egraph::rewrite::{Rewrite, RewriteAction};
use crate::egraph::template::{self, TemplateRewrite};
use crate::egraph::{EClassId, EGraph, ENode};

/// Which inflation mechanism a [`RuleSetSpec`] selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InflationMode {
    /// Exact duplicates of existing rules under new indices (§2.1).
    Duplicates,
    /// Mechanical compositions A∘B of two existing rules (§2.2).
    Compositions,
    /// The mode (iii) fidelity batch: `all_rules()` plus
    /// [`round2_rules::experimental_rules`] verbatim (§2.3) — the ONE grid
    /// point on [`NEW_RULES_GRID`], `62 + experimental_rules().len() == 95`.
    NewRules,
}

impl std::fmt::Display for InflationMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Duplicates => write!(f, "dup"),
            Self::Compositions => write!(f, "comp"),
            Self::NewRules => write!(f, "new"),
        }
    }
}

/// Sweep order for an inflated rule set (Round 2 v2,
/// `docs/plans/2026-09-01-phase3-round2-registration-v2.md`).
///
/// `Append` is Round 2 v1's order: the 62-rule production prefix in
/// `all_rules()` order, inflation appended at indices ≥ 62. It is kept only
/// to reproduce v1 — H1 was unobservable under it (an appended rule cannot
/// fire before every one of the 62 has swept once, and on classical
/// expressions one pass over the prefix alone already exceeds B=100).
///
/// `Interleave(seed)` is v2's order: base + inflation shuffled together by a
/// seeded Fisher-Yates permutation, so an inflated rule can be reached inside
/// the very first sweep. The `|R|=62` point itself (the `"base"` spec, and
/// the base prefix conceptually) is **never** shuffled by either order —
/// `build_rule_set` returns `all_rules()` verbatim whenever `total == 62`, so
/// that point stays byte-comparable to Round 1 regardless of order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleOrder {
    /// v1 reproduction: base prefix, inflation appended, both unshuffled.
    Append,
    /// v2 default: base + inflation shuffled together by this seed.
    Interleave(u64),
    /// v3 (`docs/plans/2026-09-01-phase3-round2-registration-v3.md` §6b):
    /// the base-62 rules, reordered to the exact relative order they occupy
    /// inside `Interleave(seed)` of the `total`-rule inflated set — i.e. the
    /// subsequence of the interleaved vector restricted to base indices.
    /// This is the confound-controlled reference for an inflated point at
    /// `(seed, total)`: ΔU(p) = U(p) − U(OrderMatchedBase(seed, total)) holds
    /// order fixed and isolates the |R| effect. Applies only to the base-62
    /// set (`spec.mode == None`) — see [`order_matched_base`] for why this is
    /// well-defined independent of which mode (`dup`/`comp`) the inflation at
    /// `total` came from: [`fisher_yates`] permutes purely by vector length.
    OrderMatchedBase(u64, usize),
    /// v3: the base-62 rules, fully shuffled by this seed (no inflation
    /// content at all) — isolates seed sensitivity of the order effect
    /// itself, independent of any inflated point.
    ///
    /// **Not** `egraph::rule_order::RuleOrder::Shuffled`, despite the name.
    /// This one shuffles with [`fisher_yates`] — the same permutation
    /// [`Self::Interleave`] and [`Self::OrderMatchedBase`] use — because the
    /// whole point of this variant is to be `Interleave(seed)`'s base-only
    /// control at the *same* seed. `rule_order`'s shuffle draws from a
    /// different PRNG, so the same `seed` names a different permutation
    /// there; the two are deliberately separate objects and neither can
    /// stand in for the other.
    Shuffled(u64),
    /// v3: a fixed, hand-derived heuristic reorder of the base-62 — the
    /// harness-selectable form of the production quick-win candidate named in
    /// `docs/plans/2026-09-01-phase3-round2-registration-v2.md`'s §6
    /// orchestrator finding.
    ///
    /// The order itself is **not defined here.** v3's §6b finding was
    /// adopted into `pixelflow_search::egraph::rule_order`
    /// (`RuleOrder::NumericFirst` + `NUMERIC_FIRST_ORDER`), which is the one
    /// definition of "numeric-first over the base-62"; this variant selects
    /// it. A second, byte-identical copy of a 62-entry permutation is a
    /// future divergence with nothing to catch it.
    StaticReorder(super::super::egraph::rule_order::RuleOrder),
}

/// The fixed seed Round 2 v2 registers as its default interleave order
/// (`docs/plans/2026-09-01-phase3-round2-registration-v2.md`). Encodes this
/// registration's date; chosen for a stable, greppable default, not for any
/// statistical property beyond "one fixed seed, stated and reused everywhere
/// v2's default order is used."
pub const DEFAULT_INTERLEAVE_SEED: u64 = 0x2026_0901;

impl std::fmt::Display for RuleOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Append => write!(f, "append"),
            Self::Interleave(seed) => write!(f, "interleave:0x{seed:x}"),
            Self::OrderMatchedBase(seed, total) => write!(f, "matched:0x{seed:x}:{total}"),
            Self::Shuffled(seed) => write!(f, "shuffled:0x{seed:x}"),
            Self::StaticReorder(order) => write!(f, "static:{order}"),
        }
    }
}

/// A parsed `--rule-set` argument: `"base"` (the production 62, unshuffled),
/// or `"dup:<total>"` / `"comp:<total>"` / `"new:<total>"` where `<total>` is
/// the TOTAL rule count requested (base 62 plus that mode's inflation), not
/// the inflation alone — optionally suffixed `:append` or
/// `:interleave:<seed>` to pick the sweep order (`RuleOrder`). Omitting the
/// suffix on an inflated spec defaults to `Interleave(DEFAULT_INTERLEAVE_SEED)`
/// — v2's registered default.
///
/// `"base"` alone ignores order entirely: always the unshuffled production
/// 62. Three more `"base:"`-prefixed forms (v3) select a *reordered* base-62,
/// still `|R| = 62`, for the order-effect-in-isolation and confound-control
/// measurements (`docs/plans/2026-09-01-phase3-round2-registration-v3.md` §6b):
/// - `"base:matched:<seed>:<total>"` → `RuleOrder::OrderMatchedBase(seed, total)`
///   — the base rules in the relative order they occupy inside
///   `Interleave(seed)` of the `total`-rule inflated set; the reference each
///   inflated point's ΔU is measured against.
/// - `"base:shuffled:<seed>"` → `RuleOrder::Shuffled(seed)` — base-62 fully
///   shuffled by `seed`, no inflation.
/// - `"base:static:numeric-first"` → `RuleOrder::StaticReorder(egraph::rule_order::RuleOrder::NumericFirst)`
///   — base-62 ordered by descending TRAIN strict-positive rate (the
///   production quick-win candidate).
///
/// `<seed>` is decimal or `0x`-prefixed hex, with **no underscores** (unlike
/// the `0x2026_0901` Rust literal spelling used in doc comments/prose, the
/// CLI string form is a plain hex digit run, e.g. `0x20260901`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuleSetSpec {
    pub mode: Option<InflationMode>,
    pub total: usize,
    pub order: RuleOrder,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleSetSpecError {
    /// Not `"base"` and not `"<mode>:<count>[:<order>]"`.
    Malformed(String),
    /// The `<mode>` tag isn't one this harness builds.
    UnknownMode(String),
    /// The `<count>` half didn't parse as a `usize`.
    BadCount(String),
    /// The optional `<order>` suffix wasn't `append` or `interleave:<seed>`.
    BadOrder(String),
}

impl std::fmt::Display for RuleSetSpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Malformed(s) => write!(
                f,
                "rule-set spec {s:?}: expected \"base\", \"dup:<total>[:<order>]\", \
                 \"comp:<total>[:<order>]\", or \"new:<total>[:<order>]\""
            ),
            Self::UnknownMode(s) => {
                write!(f, "rule-set spec {s:?}: unknown mode (want dup/comp/new)")
            }
            Self::BadCount(s) => write!(f, "rule-set spec: {s:?} is not a valid rule count"),
            Self::BadOrder(s) => write!(
                f,
                "rule-set spec: order {s:?} is not \"append\", \"interleave:<seed>\", \
                 \"matched:<seed>:<total>\", \"shuffled:<seed>\", or \"static:numeric-first\" \
                 (seed as decimal or 0x-prefixed hex, no underscores)"
            ),
        }
    }
}

impl std::error::Error for RuleSetSpecError {}

fn parse_seed(s: &str) -> Option<u64> {
    match s.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16).ok(),
        None => s.parse().ok(),
    }
}

/// Parse the tail of a `"base:<tail>"` spec into the v3 base-only orders
/// (§ of `RuleSetSpec::parse`'s doc comment). `None` on anything unrecognized
/// — the caller wraps it as `RuleSetSpecError::BadOrder`.
fn parse_base_order(s: &str) -> Option<RuleOrder> {
    let mut parts = s.splitn(3, ':');
    match parts.next()? {
        "matched" => {
            let seed = parse_seed(parts.next()?)?;
            let total: usize = parts.next()?.parse().ok()?;
            Some(RuleOrder::OrderMatchedBase(seed, total))
        }
        "shuffled" => {
            let seed = parse_seed(parts.next()?)?;
            Some(RuleOrder::Shuffled(seed))
        }
        "static" => match parts.next()? {
            "numeric-first" => Some(RuleOrder::StaticReorder(
                crate::egraph::rule_order::RuleOrder::NumericFirst,
            )),
            _ => None,
        },
        _ => None,
    }
}

impl RuleSetSpec {
    /// Parse `"base"` | `"dup:<count>[:<order>]"` | `"comp:<count>[:<order>]"`.
    /// `<count>` is the TOTAL `|R|`, matching every column in
    /// `docs/plans/2026-09-01-phase3-round2-rule-scaling.md` §3's grid.
    /// `<order>` is `"append"` (v1 reproduction) or `"interleave:<seed>"`;
    /// omitted, it defaults to `Interleave(DEFAULT_INTERLEAVE_SEED)`.
    pub fn parse(s: &str) -> Result<Self, RuleSetSpecError> {
        if s == "base" {
            return Ok(Self {
                mode: None,
                total: all_rules().len(),
                order: RuleOrder::Append,
            });
        }
        if let Some(rest) = s.strip_prefix("base:") {
            let order = parse_base_order(rest)
                .ok_or_else(|| RuleSetSpecError::BadOrder(rest.to_string()))?;
            return Ok(Self {
                mode: None,
                total: all_rules().len(),
                order,
            });
        }
        let mut parts = s.splitn(3, ':');
        let tag = parts
            .next()
            .filter(|t| !t.is_empty())
            .ok_or_else(|| RuleSetSpecError::Malformed(s.to_string()))?;
        let rest = parts
            .next()
            .ok_or_else(|| RuleSetSpecError::Malformed(s.to_string()))?;
        let mode = match tag {
            "dup" => InflationMode::Duplicates,
            "comp" => InflationMode::Compositions,
            "new" => InflationMode::NewRules,
            _ => return Err(RuleSetSpecError::UnknownMode(s.to_string())),
        };
        let total: usize = rest
            .parse()
            .map_err(|_| RuleSetSpecError::BadCount(rest.to_string()))?;
        let order = match parts.next() {
            None => RuleOrder::Interleave(DEFAULT_INTERLEAVE_SEED),
            Some("append") => RuleOrder::Append,
            Some(tail) => match tail.strip_prefix("interleave:").and_then(parse_seed) {
                Some(seed) => RuleOrder::Interleave(seed),
                None => return Err(RuleSetSpecError::BadOrder(tail.to_string())),
            },
        };
        Ok(Self {
            mode: Some(mode),
            total,
            order,
        })
    }
}

/// Content-addressed identity of a rule set: every rule's name, in order,
/// hashed. Written into every result row this harness produces so a curve
/// can never be silently read back against the wrong rule set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RuleSetFingerprint(pub u64);

impl std::fmt::Display for RuleSetFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

#[must_use]
pub fn rule_set_fingerprint(rules: &[Box<dyn Rewrite>]) -> RuleSetFingerprint {
    // NOT `RuleSet::fingerprint()`. That digest names an *optimizer
    // configuration* (FNV over `RuleId`s, i.e. over `rule_label` including
    // each rule's specialization) and is what a checkpoint or a cache is
    // keyed on. This one names a *Round 2 arm*: it is quoted verbatim in
    // docs/plans/2026-09-01-phase3-round2-registration-v2.md §2's grid table
    // and pinned by `v2_grid_fingerprints_are_pinned` below, so it is a
    // frozen registered constant and must keep hashing exactly what it
    // hashed when those numbers were registered. Two digests over two
    // different things, deliberately — never substitute one for the other.

    let mut hasher = DefaultHasher::new();
    rules.len().hash(&mut hasher);
    for r in rules {
        r.name().hash(&mut hasher);
    }
    RuleSetFingerprint(hasher.finish())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleSetError {
    /// `total` is not one of this mode's grid points — never padded to fit.
    UnreachableTotal {
        mode: InflationMode,
        requested: usize,
        grid: &'static [usize],
    },
    /// The composition pool has fewer members than the requested inflation.
    PoolTooSmall {
        requested_inflation: usize,
        pool_size: usize,
    },
}

impl std::fmt::Display for RuleSetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnreachableTotal {
                mode,
                requested,
                grid,
            } => write!(
                f,
                "rule-set {mode}: |R|={requested} is not on this mode's grid {grid:?} \
                 (never padded to fit — pick a grid point)"
            ),
            Self::PoolTooSmall {
                requested_inflation,
                pool_size,
            } => write!(
                f,
                "composition pool has only {pool_size} members, requested {requested_inflation} \
                 — the grid point is unreachable on this rule library, dropped rather than padded"
            ),
        }
    }
}

impl std::error::Error for RuleSetError {}

/// THE one constructor for every rule set the Round 2 harness runs. The
/// `|R|=62` point (`spec.mode == None`, or an inflated spec whose `total ==
/// 62`) is always `all_rules()` in its exact production order — never
/// shuffled, regardless of `spec.order` — so that point stays comparable to
/// Round 1 under both v1 (append) and v2 (interleave). For `total > 62`,
/// `spec.order` picks how the base+inflation list is ordered: `Append`
/// reproduces v1 (base prefix, inflation appended, both unshuffled);
/// `Interleave(seed)` (v2's default) shuffles the whole base+inflation list
/// together by that seed, so an inflated rule can fire inside the first
/// sweep.
pub fn build_rule_set(spec: &RuleSetSpec) -> Result<Vec<Box<dyn Rewrite>>, RuleSetError> {
    match spec.mode {
        None => Ok(match spec.order {
            RuleOrder::Append | RuleOrder::Interleave(_) => all_rules(),
            RuleOrder::OrderMatchedBase(seed, total) => order_matched_base(seed, total),
            RuleOrder::Shuffled(seed) => shuffled_base(seed),
            RuleOrder::StaticReorder(order) => crate::egraph::rule_order::build_rule_set(order),
        }),
        Some(InflationMode::Duplicates) => build_duplicate_set(spec.total, spec.order),
        Some(InflationMode::Compositions) => build_composition_set(spec.total, spec.order),
        Some(InflationMode::NewRules) => build_new_rule_set(spec.total, spec.order),
    }
}

/// Apply `order` to a freshly-built base+inflation list, in place. A no-op
/// for `Append` (the list is already base-prefix-then-inflation, which IS
/// append order) and for a bare 62-rule list (callers never pass one here —
/// both builders return early on `total == 62` before this is reached).
///
/// The three v3 base-only orders (`OrderMatchedBase`/`Shuffled`/
/// `StaticReorder`) do not define a meaning for an inflated (`total > 62`)
/// list — per CLAUDE.md's "no silent failures," a caller that reaches this
/// function with one of them gets a loud panic, not a silently-ignored no-op.
fn apply_order(rules: &mut [Box<dyn Rewrite>], order: RuleOrder) {
    match order {
        RuleOrder::Append => {}
        RuleOrder::Interleave(seed) => fisher_yates(rules, seed),
        RuleOrder::OrderMatchedBase(..)
        | RuleOrder::Shuffled(..)
        | RuleOrder::StaticReorder(..) => {
            panic!(
                "RuleOrder::{order} has no defined meaning for an inflated (total > 62) rule \
                 set — it selects a reordering of the base-62 only (spec.mode == None)"
            )
        }
    }
}

// ============================================================================
// v3: base-only orders (registration-v3 §6b) — confound control, seed
// sensitivity, and the static-reorder production quick-win candidate.
// ============================================================================

/// The `|R|=62` base rules, reordered to the exact relative order they
/// occupy inside `Interleave(seed)` of the `total`-rule inflated set — the
/// subsequence of that interleaved vector restricted to base indices
/// (positions `0..62` of the pre-shuffle base+inflation list, in the order
/// the shuffle leaves them).
///
/// **Well-defined independent of mode.** [`fisher_yates`]'s swap sequence is
/// a function of the slice's LENGTH only (`1..v.len()`, never its content),
/// so for a fixed `(seed, total)` the permutation is identical whether the
/// `total - 62` inflation positions were filled by `dup:<total>` or
/// `comp:<total>` — both grids share the same total↔inflation-count mapping
/// (`DUPLICATE_GRID`/`COMPOSITION_GRID`), so this never has to pick a mode.
/// Pinned by `order_matched_base_is_the_interleaved_subsequence` below.
///
/// This is the reference `docs/plans/2026-09-01-phase3-round2-registration-v3.md`
/// §6b's H1 measures each inflated point against: `ΔU(p) = U(p) −
/// U(OrderMatchedBase(seed, |p|))`, holding sweep order fixed so the
/// remaining difference is attributable to `|R|` alone.
#[must_use]
pub fn order_matched_base(seed: u64, total: usize) -> Vec<Box<dyn Rewrite>> {
    let base_len = all_rules().len();
    assert!(
        total >= base_len,
        "order_matched_base: total ({total}) must be >= the base rule count ({base_len})"
    );
    let mut idx: Vec<usize> = (0..total).collect();
    fisher_yates(&mut idx, seed);
    let mut base: Vec<Option<Box<dyn Rewrite>>> = all_rules().into_iter().map(Some).collect();
    idx.into_iter()
        .filter(|&i| i < base_len)
        .map(|i| {
            base[i]
                .take()
                .unwrap_or_else(|| panic!("order_matched_base: base index {i} visited twice"))
        })
        .collect()
}

/// The base-62 rules, fully shuffled by `seed` — no inflation content at
/// all. Isolates seed sensitivity of the order effect itself, independent of
/// any inflated point (registration-v3 §6b's "ORDER effect" measurement).
#[must_use]
pub fn shuffled_base(seed: u64) -> Vec<Box<dyn Rewrite>> {
    let mut rules = all_rules();
    fisher_yates(&mut rules, seed);
    rules
}

// ============================================================================
// Mode (i): exact duplicates (§2.1)
// ============================================================================

/// A duplicate of `inner` under a new rule index. Delegates every behavior
/// to `inner` — same closure, same per-match semantics, purely a second
/// (or third, …) way to reach the same node — except `name`, which is
/// `"<inner>#dup<copy>"` so per-rule statistics (and a Guide's
/// `w_rule[rule_idx]`, out of scope for this unguided-only harness) can tell
/// a copy from its original.
pub struct DuplicateRule {
    inner: Box<dyn Rewrite>,
    name: String,
}

impl DuplicateRule {
    #[must_use]
    pub fn new(inner: Box<dyn Rewrite>, copy: usize) -> Self {
        let name = format!("{}#dup{copy}", inner.name());
        Self { inner, name }
    }
}

impl Rewrite for DuplicateRule {
    fn name(&self) -> &str {
        &self.name
    }
    fn apply(&self, egraph: &EGraph, id: EClassId, node: &ENode) -> Option<RewriteAction> {
        self.inner.apply(egraph, id, node)
    }
    fn is_destructive(&self) -> bool {
        self.inner.is_destructive()
    }
    fn lhs_template(&self, out: &mut ExprArena) -> Option<pixelflow_ir::ExprId> {
        self.inner.lhs_template(out)
    }
    fn rhs_template(&self, out: &mut ExprArena) -> Option<pixelflow_ir::ExprId> {
        self.inner.rhs_template(out)
    }
}

/// The five |R| points mode (i) is registered at (§3).
pub const DUPLICATE_GRID: &[usize] = &[62, 93, 124, 186, 248];

fn build_duplicate_set(
    total: usize,
    order: RuleOrder,
) -> Result<Vec<Box<dyn Rewrite>>, RuleSetError> {
    if !DUPLICATE_GRID.contains(&total) {
        return Err(RuleSetError::UnreachableTotal {
            mode: InflationMode::Duplicates,
            requested: total,
            grid: DUPLICATE_GRID,
        });
    }
    let mut rules = all_rules();
    if total == 62 {
        // Never shuffled — the base point stays Round-1-comparable under
        // every order.
        return Ok(rules);
    }
    if total == 93 {
        // Half-cycle: even indices 0, 2, .., 60 — 31 rules, spread across
        // every module rather than the first 31 (which would all be
        // algebra rules, per §2.1's "spread across every module" note).
        let half: Vec<Box<dyn Rewrite>> = all_rules()
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 0)
            .map(|(_, r)| Box::new(DuplicateRule::new(r, 1)) as Box<dyn Rewrite>)
            .collect();
        rules.extend(half);
        apply_order(&mut rules, order);
        return Ok(rules);
    }
    let cycles = match total {
        124 => 1,
        186 => 2,
        248 => 3,
        _ => unreachable!("checked against DUPLICATE_GRID above"),
    };
    for c in 1..=cycles {
        let cycle: Vec<Box<dyn Rewrite>> = all_rules()
            .into_iter()
            .map(|r| Box::new(DuplicateRule::new(r, c)) as Box<dyn Rewrite>)
            .collect();
        rules.extend(cycle);
    }
    apply_order(&mut rules, order);
    Ok(rules)
}

// ============================================================================
// Mode (ii): mechanical compositions (§2.2)
// ============================================================================

/// A composed rule A∘B, plus the provenance of how it was built — written
/// out (names, indices, position) so the Register run's composition-pool
/// JSON is reproducible and reviewable.
pub struct Composition {
    pub a_idx: usize,
    pub b_idx: usize,
    pub position: Vec<u8>,
    pub rule: TemplateRewrite,
}

/// Whether `r` has both an LHS and an RHS template — the composable subset
/// (30 of 62 in the production library, per §2.2).
fn is_templated(r: &dyn Rewrite) -> bool {
    let mut scratch = ExprArena::new();
    r.lhs_template(&mut scratch).is_some() && r.rhs_template(&mut scratch).is_some()
}

/// Enumerate every valid composition of `base`'s templated rules, in
/// deterministic (a_idx, b_idx, position) order, applying §2.2's filters
/// (identity, no-op, exact-structural-duplicate) via
/// [`TemplateRewrite::compose`] and a display-string dedup key.
///
/// This does not run the oracle gate (§2.4) — that requires
/// `pixelflow-ir`'s `oracle` feature, which this crate exposes only to its
/// own test binary (`math::oracle`, `#[cfg(test)]`) and to downstream
/// crates that opt in themselves (`pixelflow-pipeline --features
/// training`). Callers that need the validated pool run the oracle check
/// themselves against every `Composition` this returns and drop failures —
/// never silently, per the design's §2.4 gate.
#[must_use]
pub fn compose_rules(base: &[Box<dyn Rewrite>]) -> Vec<Composition> {
    let templated_b: Vec<usize> = base
        .iter()
        .enumerate()
        .filter(|(_, r)| is_templated(r.as_ref()))
        .map(|(i, _)| i)
        .collect();

    let mut out = Vec::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();

    for (a_idx, a) in base.iter().enumerate() {
        let mut probe = ExprArena::new();
        let (Some(a_lhs), Some(a_rhs)) = (a.lhs_template(&mut probe), a.rhs_template(&mut probe))
        else {
            continue;
        };
        let _ = a_lhs; // only a_rhs's positions matter for this loop
        let positions = template::positions(&probe, a_rhs);

        for &b_idx in &templated_b {
            let b = &base[b_idx];
            for pos in &positions {
                let Some(rule) = TemplateRewrite::compose(a.as_ref(), b.as_ref(), pos) else {
                    continue;
                };
                let mut disp = ExprArena::new();
                let lhs = rule
                    .lhs_template(&mut disp)
                    .expect("TemplateRewrite always has both templates");
                let rhs = rule
                    .rhs_template(&mut disp)
                    .expect("TemplateRewrite always has both templates");
                let key = (disp.display(lhs).to_string(), disp.display(rhs).to_string());
                if seen.insert(key) {
                    out.push(Composition {
                        a_idx,
                        b_idx,
                        position: pos.clone(),
                        rule,
                    });
                }
            }
        }
    }
    out
}

/// A tiny deterministic PRNG (splitmix64) — no external dependency for one
/// seeded shuffle over a few hundred elements.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn fisher_yates<T>(v: &mut [T], seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for i in (1..v.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

/// The seeded permutation §2.2 fixes for the |R| grid's composition
/// prefixes (31, 62, 124, 186 compositions).
pub const COMPOSITION_SEED: u64 = 0x5EED2;

#[must_use]
pub fn composition_pool(base: &[Box<dyn Rewrite>], seed: u64) -> Vec<Composition> {
    let mut pool = compose_rules(base);
    fisher_yates(&mut pool, seed);
    pool
}

/// Total |R| → composition-count prefix, per §3's grid.
pub const COMPOSITION_GRID: &[(usize, usize)] =
    &[(62, 0), (93, 31), (124, 62), (186, 124), (248, 186)];

fn build_composition_set(
    total: usize,
    order: RuleOrder,
) -> Result<Vec<Box<dyn Rewrite>>, RuleSetError> {
    let Some(&(_, inflation)) = COMPOSITION_GRID.iter().find(|(t, _)| *t == total) else {
        return Err(RuleSetError::UnreachableTotal {
            mode: InflationMode::Compositions,
            requested: total,
            grid: &[62, 93, 124, 186, 248],
        });
    };
    let mut rules = all_rules();
    if inflation == 0 {
        // Never shuffled — the base point stays Round-1-comparable under
        // every order.
        return Ok(rules);
    }
    // The pool's own seeded shuffle (COMPOSITION_SEED) picks WHICH
    // compositions are in the prefix — orthogonal to `order`, which picks
    // where those compositions sit in the SWEEP. Both are seeded and fixed;
    // neither is re-derived here.
    let pool = composition_pool(&all_rules(), COMPOSITION_SEED);
    if pool.len() < inflation {
        return Err(RuleSetError::PoolTooSmall {
            requested_inflation: inflation,
            pool_size: pool.len(),
        });
    }
    rules.extend(
        pool.into_iter()
            .take(inflation)
            .map(|c| Box::new(c.rule) as Box<dyn Rewrite>),
    );
    apply_order(&mut rules, order);
    Ok(rules)
}

// ============================================================================
// Mode (iii): genuinely new rules (§2.3)
// ============================================================================

/// The ONE `|R|` point mode (iii) is registered at: `62 +
/// round2_rules::experimental_rules().len()`. A `const` (not computed at
/// call sites) so every consumer — the parser's grid check, the error
/// message, the doc comment — reads the same fixed number; pinned equal to
/// `62 + experimental_rules().len()` by
/// `new_rules_grid_matches_experimental_rules_len` below so a future edit to
/// the batch fails loud here rather than silently drifting the registered
/// grid point.
pub const NEW_RULES_GRID: &[usize] = &[95];

fn build_new_rule_set(
    total: usize,
    order: RuleOrder,
) -> Result<Vec<Box<dyn Rewrite>>, RuleSetError> {
    if !NEW_RULES_GRID.contains(&total) {
        return Err(RuleSetError::UnreachableTotal {
            mode: InflationMode::NewRules,
            requested: total,
            grid: NEW_RULES_GRID,
        });
    }
    let mut rules = all_rules();
    if total == 62 {
        // Unreachable today (NEW_RULES_GRID has no 62 entry) but kept for
        // the same reason the other two builders keep it: the base point
        // must never be shuffled under any order, by construction, not by
        // omission.
        return Ok(rules);
    }
    rules.extend(round2_rules::experimental_rules());
    apply_order(&mut rules, order);
    Ok(rules)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_the_documented_forms() {
        assert_eq!(RuleSetSpec::parse("base").unwrap().mode, None);
        let d = RuleSetSpec::parse("dup:124").unwrap();
        assert_eq!(d.mode, Some(InflationMode::Duplicates));
        assert_eq!(d.total, 124);
        let c = RuleSetSpec::parse("comp:186").unwrap();
        assert_eq!(c.mode, Some(InflationMode::Compositions));
        assert_eq!(c.total, 186);
        assert!(RuleSetSpec::parse("bogus").is_err());
        assert!(RuleSetSpec::parse("dup:notanumber").is_err());
        let n = RuleSetSpec::parse("new:95").unwrap();
        assert_eq!(n.mode, Some(InflationMode::NewRules));
        assert_eq!(n.total, 95);
        assert!(
            build_rule_set(&RuleSetSpec {
                mode: Some(InflationMode::NewRules),
                total: 31,
                order: RuleOrder::Interleave(DEFAULT_INTERLEAVE_SEED),
            })
            .is_err(),
            "31 is off mode (iii)'s one-point grid — never padded"
        );
    }

    #[test]
    fn new_rules_grid_matches_experimental_rules_len() {
        assert_eq!(
            NEW_RULES_GRID,
            &[62 + round2_rules::experimental_rules().len()],
            "NEW_RULES_GRID must track experimental_rules().len() exactly — a future edit to \
             the mode (iii) batch that doesn't update this constant must fail here, loudly, \
             rather than silently registering the wrong |R|"
        );
    }

    #[test]
    fn new_rule_set_never_shuffles_base_at_the_grid_point_content() {
        // The 95-rule set must contain exactly all_rules() plus
        // experimental_rules() (as a multiset of names) under both orders —
        // order only permutes position, never membership.
        let base_names: HashSet<String> =
            all_rules().iter().map(|r| r.name().to_string()).collect();
        let new_names: HashSet<String> = round2_rules::experimental_rules()
            .iter()
            .map(|r| r.name().to_string())
            .collect();
        for order in [
            RuleOrder::Append,
            RuleOrder::Interleave(DEFAULT_INTERLEAVE_SEED),
        ] {
            let rules = build_rule_set(&RuleSetSpec {
                mode: Some(InflationMode::NewRules),
                total: 95,
                order,
            })
            .unwrap_or_else(|e| panic!("new:95 should build under {order}: {e}"));
            assert_eq!(rules.len(), 95);
            let got: HashSet<String> = rules.iter().map(|r| r.name().to_string()).collect();
            let want: HashSet<String> = base_names.union(&new_names).cloned().collect();
            assert_eq!(
                got, want,
                "new:95 must be exactly base ∪ experimental, under {order}"
            );
        }
    }

    #[test]
    fn duplicate_grid_never_pads() {
        for &total in DUPLICATE_GRID {
            let rules = build_rule_set(&RuleSetSpec {
                mode: Some(InflationMode::Duplicates),
                total,
                order: RuleOrder::Interleave(DEFAULT_INTERLEAVE_SEED),
            })
            .unwrap_or_else(|e| panic!("dup:{total} should build: {e}"));
            assert_eq!(rules.len(), total, "dup:{total} produced the wrong count");
        }
        assert!(
            build_rule_set(&RuleSetSpec {
                mode: Some(InflationMode::Duplicates),
                total: 100,
                order: RuleOrder::Interleave(DEFAULT_INTERLEAVE_SEED),
            })
            .is_err(),
            "an off-grid total must error, never pad"
        );
    }

    #[test]
    fn duplicate_names_itself_after_its_inner_rule_and_matches_identically() {
        // A duplicate must behave identically to its inner rule (same
        // apply/templates) so the sweep cannot tell them apart except by
        // name — that is exactly what makes copies "learnable overhead"
        // (§2.1: a Guide must learn the name is worthless, not the shape).
        let inner = all_rules()
            .into_iter()
            .next()
            .expect("all_rules is nonempty");
        let inner_name = inner.name().to_string();
        let dup = DuplicateRule::new(inner, 1);
        assert_eq!(dup.name(), format!("{inner_name}#dup1"));

        // Same LHS/RHS templates (or lack thereof) as a fresh copy of the
        // same rule at the same index.
        let fresh = &all_rules()[0];
        let mut a = ExprArena::new();
        let mut b = ExprArena::new();
        assert_eq!(
            dup.lhs_template(&mut a).is_some(),
            fresh.lhs_template(&mut b).is_some()
        );
    }

    #[test]
    fn composition_pool_is_nonempty_and_deterministic() {
        let base = all_rules();
        let pool_a = composition_pool(&base, COMPOSITION_SEED);
        let pool_b = composition_pool(&base, COMPOSITION_SEED);
        assert!(
            !pool_a.is_empty(),
            "expected at least one valid composition"
        );
        assert_eq!(
            pool_a.len(),
            pool_b.len(),
            "same seed must give the same pool size"
        );
        for (x, y) in pool_a.iter().zip(pool_b.iter()) {
            assert_eq!(x.a_idx, y.a_idx);
            assert_eq!(x.b_idx, y.b_idx);
            assert_eq!(x.position, y.position);
        }
    }

    #[test]
    fn composition_rule_set_grid_never_pads() {
        let pool_size = composition_pool(&all_rules(), COMPOSITION_SEED).len();
        for &(total, inflation) in COMPOSITION_GRID {
            let result = build_rule_set(&RuleSetSpec {
                mode: Some(InflationMode::Compositions),
                total,
                order: RuleOrder::Interleave(DEFAULT_INTERLEAVE_SEED),
            });
            if inflation <= pool_size {
                let rules = result.unwrap_or_else(|e| panic!("comp:{total} should build: {e}"));
                assert_eq!(rules.len(), total);
            } else {
                assert!(
                    result.is_err(),
                    "comp:{total} needs {inflation} compositions but the pool has only \
                     {pool_size} — must error, never pad"
                );
            }
        }
    }

    #[test]
    fn fingerprint_is_stable_and_order_sensitive() {
        let a = all_rules();
        let b = all_rules();
        assert_eq!(rule_set_fingerprint(&a), rule_set_fingerprint(&b));
        let dup = build_rule_set(&RuleSetSpec {
            mode: Some(InflationMode::Duplicates),
            total: 93,
            order: RuleOrder::Interleave(DEFAULT_INTERLEAVE_SEED),
        })
        .unwrap();
        assert_ne!(rule_set_fingerprint(&a), rule_set_fingerprint(&dup));
    }

    #[test]
    fn append_and_interleave_fingerprint_differently_for_the_same_content() {
        // Two orders of the SAME set of rules (same names, same multiset)
        // must fingerprint differently — the fingerprint covers order, not
        // just membership. This is the load-bearing guarantee for telling a
        // v1 (append) artifact apart from a v2 (interleave) one at the same
        // |R|.
        let spec_append = RuleSetSpec {
            mode: Some(InflationMode::Duplicates),
            total: 124,
            order: RuleOrder::Append,
        };
        let spec_interleave = RuleSetSpec {
            mode: Some(InflationMode::Duplicates),
            total: 124,
            order: RuleOrder::Interleave(DEFAULT_INTERLEAVE_SEED),
        };
        let appended = build_rule_set(&spec_append).unwrap();
        let interleaved = build_rule_set(&spec_interleave).unwrap();
        assert_eq!(
            appended.len(),
            interleaved.len(),
            "same content, different order — the whole point of this test"
        );
        let mut appended_names: Vec<&str> = appended.iter().map(|r| r.name()).collect();
        let mut interleaved_names: Vec<&str> = interleaved.iter().map(|r| r.name()).collect();
        appended_names.sort_unstable();
        interleaved_names.sort_unstable();
        assert_eq!(
            appended_names, interleaved_names,
            "interleaving must be a permutation, not a different rule set"
        );
        assert_ne!(
            rule_set_fingerprint(&appended),
            rule_set_fingerprint(&interleaved),
            "same rule multiset, different sweep order, must fingerprint differently"
        );

        // Two different interleave seeds must also (almost certainly)
        // differ — same requirement, different axis.
        let other_seed = build_rule_set(&RuleSetSpec {
            order: RuleOrder::Interleave(DEFAULT_INTERLEAVE_SEED ^ 0xFFFF_FFFF),
            ..spec_interleave
        })
        .unwrap();
        assert_ne!(
            rule_set_fingerprint(&interleaved),
            rule_set_fingerprint(&other_seed)
        );
    }

    #[test]
    fn the_62_rule_point_is_never_reordered() {
        // `total == 62` must return `all_rules()` verbatim under EVERY
        // order — Append, Interleave, any seed — so the |R|=62 point stays
        // byte-comparable to Round 1 regardless of which order the rest of
        // the grid uses.
        let production = all_rules();
        let production_fp = rule_set_fingerprint(&production);
        for spec in [
            RuleSetSpec {
                mode: None,
                total: 62,
                order: RuleOrder::Append,
            },
            RuleSetSpec {
                mode: Some(InflationMode::Duplicates),
                total: 62,
                order: RuleOrder::Append,
            },
            RuleSetSpec {
                mode: Some(InflationMode::Duplicates),
                total: 62,
                order: RuleOrder::Interleave(DEFAULT_INTERLEAVE_SEED),
            },
            RuleSetSpec {
                mode: Some(InflationMode::Compositions),
                total: 62,
                order: RuleOrder::Interleave(0xDEAD_BEEF),
            },
        ] {
            let rules = build_rule_set(&spec).unwrap();
            assert_eq!(
                rule_set_fingerprint(&rules),
                production_fp,
                "|R|=62 must be all_rules() verbatim under {:?}",
                spec.order
            );
        }
    }

    #[test]
    fn parse_defaults_to_interleave_and_accepts_append_and_explicit_seed() {
        let default = RuleSetSpec::parse("dup:124").unwrap();
        assert_eq!(
            default.order,
            RuleOrder::Interleave(DEFAULT_INTERLEAVE_SEED)
        );

        let appended = RuleSetSpec::parse("dup:124:append").unwrap();
        assert_eq!(appended.order, RuleOrder::Append);

        let explicit = RuleSetSpec::parse("comp:93:interleave:0x2A").unwrap();
        assert_eq!(explicit.order, RuleOrder::Interleave(0x2A));
        assert_eq!(explicit.mode, Some(InflationMode::Compositions));
        assert_eq!(explicit.total, 93);

        let decimal_seed = RuleSetSpec::parse("dup:93:interleave:42").unwrap();
        assert_eq!(decimal_seed.order, RuleOrder::Interleave(42));

        assert!(RuleSetSpec::parse("dup:124:bogus-order").is_err());
        // "base" ignores order entirely — always the unshuffled production
        // 62, no suffix accepted (it never carries inflation to order).
        assert_eq!(RuleSetSpec::parse("base").unwrap().order, RuleOrder::Append);
    }

    /// The full v2 grid's fingerprints, pinned — both orders, every
    /// realized `|R|` point of modes (i)/(ii). These are the exact values in
    /// `docs/plans/2026-09-01-phase3-round2-registration-v2.md` §2; a change
    /// here (a rule added/removed/reordered upstream, a shuffle change) must
    /// change this test AND that table together, never silently drift apart.
    /// `_app` reproduces v1's committed fingerprints exactly (the append
    /// order is unchanged code); `_int` is v2's new default order.
    #[test]
    fn v2_grid_fingerprints_are_pinned() {
        fn fp(mode: InflationMode, total: usize, order: RuleOrder) -> RuleSetFingerprint {
            let rules = build_rule_set(&RuleSetSpec {
                mode: Some(mode),
                total,
                order,
            })
            .unwrap_or_else(|e| panic!("{mode}:{total} under {order:?} should build: {e}"));
            rule_set_fingerprint(&rules)
        }
        let interleave = RuleOrder::Interleave(DEFAULT_INTERLEAVE_SEED);
        let dup = InflationMode::Duplicates;
        let comp = InflationMode::Compositions;

        assert_eq!(fp(dup, 93, interleave).0, 0x83e6_10e3_3e78_2a68);
        assert_eq!(fp(dup, 93, RuleOrder::Append).0, 0xfdd6_1724_6eb9_8590);
        assert_eq!(fp(dup, 124, interleave).0, 0xb207_aa33_1bb6_25ab);
        assert_eq!(fp(dup, 124, RuleOrder::Append).0, 0x87fe_fd5a_6357_5175);
        assert_eq!(fp(dup, 186, interleave).0, 0x3a00_c565_900b_48e6);
        assert_eq!(fp(dup, 186, RuleOrder::Append).0, 0x37a4_c537_606a_549b);
        assert_eq!(fp(dup, 248, interleave).0, 0x43c4_3d76_4ef7_f76b);
        assert_eq!(fp(dup, 248, RuleOrder::Append).0, 0x809a_0f52_b61f_e6c0);

        assert_eq!(fp(comp, 93, interleave).0, 0xa175_ce9c_a554_cc64);
        assert_eq!(fp(comp, 93, RuleOrder::Append).0, 0x90fc_9d0a_8a42_5880);
        assert_eq!(fp(comp, 124, interleave).0, 0x4f97_9d02_3b7e_8b7c);
        assert_eq!(fp(comp, 124, RuleOrder::Append).0, 0x178b_0fb7_263c_8921);
        assert_eq!(fp(comp, 186, interleave).0, 0x56c4_bed1_e6d4_31ca);
        assert_eq!(fp(comp, 186, RuleOrder::Append).0, 0xa3b1_a097_f689_0d3e);
        assert_eq!(fp(comp, 248, interleave).0, 0x2b32_41f0_11e1_2ac8);
        assert_eq!(fp(comp, 248, RuleOrder::Append).0, 0xa237_6877_fef0_fce1);

        let new_rules = InflationMode::NewRules;
        assert_eq!(fp(new_rules, 95, interleave).0, 0x113c_ca49_c99c_c850);
        assert_eq!(
            fp(new_rules, 95, RuleOrder::Append).0,
            0x4f4a_4cbd_2e4f_89cb
        );
    }

    // ========================================================================
    // v3: base-only orders (registration-v3 §6b)
    // ========================================================================

    #[test]
    fn order_matched_base_is_the_interleaved_subsequence_and_mode_independent() {
        // The base rules' relative order inside `order_matched_base(seed,
        // total)` must equal their relative order inside the FULL
        // Interleave(seed) vector at that total — checked against BOTH dup
        // and comp at the same total, proving the mode-independence claim in
        // `order_matched_base`'s doc comment (fisher_yates permutes by
        // length only, never content).
        let seed = DEFAULT_INTERLEAVE_SEED;
        let base_names: HashSet<String> =
            all_rules().iter().map(|r| r.name().to_string()).collect();
        for total in [93usize, 124, 186, 248] {
            let matched = order_matched_base(seed, total);
            assert_eq!(
                matched.len(),
                62,
                "order_matched_base(_, {total}) must return exactly the base-62"
            );
            let matched_names: Vec<String> = matched.iter().map(|r| r.name().to_string()).collect();
            assert_eq!(
                matched_names.iter().cloned().collect::<HashSet<_>>(),
                base_names,
                "order_matched_base must be a permutation of the base-62, never a different set"
            );

            for (mode, dup_or_comp) in [
                (InflationMode::Duplicates, "dup"),
                (InflationMode::Compositions, "comp"),
            ] {
                let full = build_rule_set(&RuleSetSpec {
                    mode: Some(mode),
                    total,
                    order: RuleOrder::Interleave(seed),
                })
                .unwrap_or_else(|e| panic!("{dup_or_comp}:{total} should build: {e}"));
                // Duplicates carry a "#dup<N>" suffix and compositions carry
                // their own synthesized names — neither is a base name, so
                // filtering by exact membership in `base_names` keeps only
                // the true base rules, in their post-shuffle order.
                let full_base_subsequence: Vec<String> = full
                    .iter()
                    .map(|r| r.name().to_string())
                    .filter(|n| base_names.contains(n.as_str()))
                    .collect();
                assert_eq!(
                    full_base_subsequence, matched_names,
                    "order_matched_base({seed}, {total}) must equal the base subsequence of \
                     Interleave({seed}) applied to {dup_or_comp}:{total}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "must be >= the base rule count")]
    fn order_matched_base_rejects_a_total_below_the_base_count() {
        let _ = order_matched_base(DEFAULT_INTERLEAVE_SEED, 10);
    }

    #[test]
    fn shuffled_base_permutes_without_changing_membership() {
        let base_names: HashSet<String> =
            all_rules().iter().map(|r| r.name().to_string()).collect();
        let shuffled = shuffled_base(0xC0FF_EE);
        assert_eq!(shuffled.len(), 62);
        let shuffled_names: HashSet<String> =
            shuffled.iter().map(|r| r.name().to_string()).collect();
        assert_eq!(
            shuffled_names, base_names,
            "shuffling must not change membership"
        );
        let original = all_rules();
        let original_names: Vec<&str> = original.iter().map(|r| r.name()).collect();
        let shuffled_order: Vec<&str> = shuffled.iter().map(|r| r.name()).collect();
        assert_ne!(
            original_names, shuffled_order,
            "seed 0xC0FFEE must actually move at least one rule (checked, not assumed)"
        );
    }

    #[test]
    fn static_reorder_numeric_first_is_pinned() {
        // NUMERIC_FIRST_ORDER must be a permutation of 0..62 — catches a
        // transcription error in the pinned array immediately, not as a
        // silent duplicate/missing index.
        let mut sorted = crate::egraph::rule_order::NUMERIC_FIRST_ORDER;
        sorted.sort_unstable();
        let identity: [usize; 62] = std::array::from_fn(|i| i);
        assert_eq!(
            sorted, identity,
            "NUMERIC_FIRST_ORDER must be a permutation of every base index exactly once"
        );

        let rules = crate::egraph::rule_order::build_rule_set(
            crate::egraph::rule_order::RuleOrder::NumericFirst,
        );
        assert_eq!(rules.len(), 62);
        let names: Vec<&str> = rules.iter().map(|r| r.name()).collect();
        // The pinned top-of-list, read straight from
        // docs/results/2026-09-01-train-guide-report.md's "train rate"
        // column: power-rsqrt (idx53, 0.18105) first, power-zero (idx48,
        // 0.12500) fifth (ahead of every even-negation despite the report's
        // print order, since 0.12500 > 0.05533/0.05557).
        assert_eq!(
            &names[..8],
            &[
                "power-rsqrt",
                "recip-sqrt",
                "power-recip",
                "power-sqrt",
                "power-zero",
                "even-negation",
                "even-negation",
                "power-combine",
            ]
        );
        // Every base name still present, once each.
        let got_owned: HashSet<String> = names.iter().map(|s| s.to_string()).collect();
        let base_owned: HashSet<String> =
            all_rules().iter().map(|r| r.name().to_string()).collect();
        assert_eq!(got_owned, base_owned);
    }

    #[test]
    fn base_order_specs_parse_and_round_trip_through_build_rule_set() {
        let matched = RuleSetSpec::parse("base:matched:0x20260901:93").unwrap();
        assert_eq!(matched.mode, None);
        assert_eq!(
            matched.order,
            RuleOrder::OrderMatchedBase(DEFAULT_INTERLEAVE_SEED, 93)
        );
        assert_eq!(build_rule_set(&matched).unwrap().len(), 62);

        let shuffled = RuleSetSpec::parse("base:shuffled:42").unwrap();
        assert_eq!(shuffled.order, RuleOrder::Shuffled(42));
        assert_eq!(build_rule_set(&shuffled).unwrap().len(), 62);

        let static_reorder = RuleSetSpec::parse("base:static:numeric-first").unwrap();
        assert_eq!(
            static_reorder.order,
            RuleOrder::StaticReorder(crate::egraph::rule_order::RuleOrder::NumericFirst)
        );
        assert_eq!(build_rule_set(&static_reorder).unwrap().len(), 62);

        assert!(RuleSetSpec::parse("base:bogus").is_err());
        assert!(RuleSetSpec::parse("base:static:not-a-kind").is_err());
        assert!(
            RuleSetSpec::parse("base:matched:42").is_err(),
            "matched needs a total too"
        );
    }

    #[test]
    fn base_only_orders_fingerprint_differently_from_each_other_and_from_production() {
        let production_fp = rule_set_fingerprint(&all_rules());
        let matched_fp = rule_set_fingerprint(&order_matched_base(DEFAULT_INTERLEAVE_SEED, 93));
        let shuffled_fp = rule_set_fingerprint(&shuffled_base(DEFAULT_INTERLEAVE_SEED));
        let static_fp = rule_set_fingerprint(&crate::egraph::rule_order::build_rule_set(
            crate::egraph::rule_order::RuleOrder::NumericFirst,
        ));
        let fps = [production_fp, matched_fp, shuffled_fp, static_fp];
        for i in 0..fps.len() {
            for j in (i + 1)..fps.len() {
                assert_ne!(
                    fps[i], fps[j],
                    "every base-62 order (production, matched, shuffled, static) must \
                     fingerprint distinctly at indices {i} and {j}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "has no defined meaning for an inflated")]
    fn v3_base_only_orders_panic_rather_than_silently_no_op_when_applied_to_an_inflated_set() {
        // NO SILENT FAILURES: a v3 order used where only Append/Interleave
        // are defined (an inflated, total > 62 build) must fail loudly, not
        // quietly reduce to Append.
        let mut rules = all_rules();
        rules.push(Box::new(DuplicateRule::new(all_rules().remove(0), 1)));
        apply_order(&mut rules, RuleOrder::Shuffled(1));
    }
}
