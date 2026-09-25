//! Coverage table for the saturation Guide's candidate-local context
//! (`docs/plans/2026-09-01-guide-candidate-context.md` §2–§3).
//!
//! # Scope: a proxy cell over the CURRENT record schema
//!
//! The design doc's [`CandidateCell`] (§2.3) buckets on `down: DownSig`
//! (per-slot dominant op-group over up to `K_DOWN = 3` bound classes),
//! `up1: Option<Group>` (one-hop parent group), `dcost: DcostBucket`, and
//! `on_path: bool` — all read from [`CandidateContext`], which does not
//! exist yet (`pixelflow-search/src/egraph/candidate.rs` is still Round 1's
//! `CandidateFeatures { key, neighborhood_ops, budget_fraction }`; the
//! `CandidateContext`/`RoundSnapshot` machinery is a sibling agent's work,
//! landing separately). `gen_strict_labels.rs`'s JSONL record today carries
//! exactly `rule_idx`, `rule_name`, `budget_fraction`, and
//! `neighborhood_op_hist` (a per-op histogram of the ONE-HOP CHILD ops of
//! every node in the matched class `c` itself — not a per-operand-slot
//! summary; see `candidate.rs`'s `neighborhood_ops` doc comment).
//!
//! This tool builds the coverage table (§3.2/§3.3's counting rules, applied
//! verbatim) over a [`Cell`] that is a deliberate, documented PROXY for the
//! design's `CandidateCell`:
//!
//! ```text
//! Cell { rule_idx, down_group: Option<Group>, budget: BudgetBucket }
//! ```
//!
//! `down_group` is `dominant(neighborhood_op_hist)` under the same `Group`
//! alphabet and the same argmax/tie-break rule the design doc specifies for
//! `dominant(OpHistogram)` (§2.2) — the single best available stand-in for
//! `DownSig` today, since there is no per-slot data to bucket on. `up1`,
//! `dcost`, and `on_path` are simply absent: there is no source field for
//! them in the current record.
//!
//! # Where the new fields plug in (read this before extending)
//!
//! When `CandidateContext` lands and `gen_strict_labels.rs`'s record grows
//! `down: [ClassSummary; K_DOWN]`, `up: ParentHistogram`, `dcost: i32`,
//! `on_best_path: bool` (design doc §1.4/§8), exactly one function in this
//! file needs to change: [`Cell::of`]. Replace its `down_group` computation
//! with the true per-slot `DownSig` (dominant group of each present
//! `ClassSummary`, at most the two priciest slots kept per §2.3), add
//! `up1: Option<Group>` from `up.hop1`, add `dcost: DcostBucket` from the
//! new `dcost` field via [`DcostBucket::of`] (already written to spec, §2.4,
//! ready to receive a real value), and add `on_path: bool` straight from
//! `on_best_path`. Every other function in this file — the coverage
//! counting (§3.2), the per-rule table, the trig-context comparison — is
//! written against `Cell` and does not know how it was bucketed, so none of
//! it needs to change.
//!
//! `DcostBucket` is included below now, unused by `Cell::of`, precisely so
//! that migration is a one-line change in `Cell` rather than a new type to
//! design under time pressure later.
//!
//! # What this tool does NOT do
//!
//! No cell-oracle AUC/PR-AUC (§3.4) — that statistic needs `dcost` and
//! `on_best_path` to mean anything (a cell without STATE can't separate
//! "pays now" from "pays later"), so it belongs to the full-context
//! `guide_coverage` tool §8 names once `CandidateContext` lands, not to this
//! proxy. This tool answers the §3.2/§3.3 counting questions only: how much
//! of the reachable proxy-cell space did Round 1's mint actually see.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training \
//!     --bin guide_coverage_table -- \
//!     --train pixelflow-pipeline/data/strict_labels_train.jsonl \
//!     --dev pixelflow-pipeline/data/strict_labels_dev.jsonl \
//!     --out-md docs/results/2026-09-01-guide-coverage-round1.md \
//!     --out-json docs/results/2026-09-01-guide-coverage-round1.json
//! ```

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{BufRead, Write as _};

use clap::Parser;
use pixelflow_search::egraph::RuleId;
use serde_json::Value;

#[derive(Parser)]
#[command(name = "guide_coverage_table")]
#[command(about = "Coverage table for the saturation Guide's candidate-local proxy cell")]
struct Args {
    /// TRAIN strict-label JSONL (`gen_strict_labels --out-train`).
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/strict_labels_train.jsonl"
    )]
    train: String,

    /// DEV strict-label JSONL (`gen_strict_labels --out-dev`).
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/strict_labels_dev.jsonl"
    )]
    dev: String,

    /// Markdown report output path.
    #[arg(long)]
    out_md: Option<String>,

    /// JSON report output path.
    #[arg(long)]
    out_json: Option<String>,

    /// n_thin (design doc §3.2): a cell needs at least this many
    /// observations to leave "thin" status.
    #[arg(long, default_value_t = 100)]
    n_thin: u64,

    /// pos_thin (design doc §3.2): a cell needs at least this many
    /// positives to leave "thin" status, but only for rules that have any
    /// positives at all somewhere.
    #[arg(long, default_value_t = 5)]
    pos_thin: u64,
}

// ============================================================================
// The op-group alphabet G (design doc §2.1) — total over every `OpKind`
// variant name string that can appear in `neighborhood_op_hist`'s keys.
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Group {
    Leaf,
    Lin,
    Mul,
    Div,
    Root,
    Trig,
    ExpLog,
    Other,
}

impl Group {
    const ALL: [Group; 8] = [
        Group::Leaf,
        Group::Lin,
        Group::Mul,
        Group::Div,
        Group::Root,
        Group::Trig,
        Group::ExpLog,
        Group::Other,
    ];

    fn label(self) -> &'static str {
        match self {
            Group::Leaf => "LEAF",
            Group::Lin => "LIN",
            Group::Mul => "MUL",
            Group::Div => "DIV",
            Group::Root => "ROOT",
            Group::Trig => "TRIG",
            Group::ExpLog => "EXPLOG",
            Group::Other => "OTHER",
        }
    }

    /// Total over every `OpKind` Debug-name string (`kind.rs`'s `op_table!`
    /// generates `{:?}` == the variant name). Fails loud on an unrecognized
    /// name — a name this function doesn't know means `OpKind` grew a
    /// variant this table wasn't updated for, which is exactly the "a
    /// convention written in a comment is an invariant something else will
    /// eventually break" failure mode CLAUDE.md names; better to panic here
    /// than silently drop the op into a wrong bucket.
    fn of_op_name(name: &str) -> Group {
        match name {
            "Var" | "Const" | "Buffer" | "Tuple" => Group::Leaf,
            "Add" | "Sub" | "Neg" | "Abs" | "Min" | "Max" | "If" | "Lt" | "Le" | "Gt" | "Ge"
            | "Eq" | "Ne" | "Floor" | "Ceil" | "Round" | "TruncToInt" | "IntToFloat" | "IAdd"
            | "Shl" | "Shr" | "BitAnd" | "BitOr" => Group::Lin,
            "Mul" | "MulAdd" => Group::Mul,
            "Div" | "Recip" => Group::Div,
            "Sqrt" | "Rsqrt" => Group::Root,
            "Sin" | "Cos" | "Tan" | "Asin" | "Acos" | "Atan" | "Atan2" => Group::Trig,
            "Exp" | "Exp2" | "Ln" | "Log2" | "Log10" | "Pow" => Group::ExpLog,
            "Dwrt" | "Gather" | "RawGather" | "Reduce" => Group::Other,
            other => panic!(
                "guide_coverage_table: neighborhood_op_hist key {other:?} is not a recognized \
                 OpKind name — Group::of_op_name (design doc §2.1's table) needs a new arm, \
                 not a silent default bucket"
            ),
        }
    }

    /// `dominant(OpHistogram)` (design doc §2.2): argmax over group sums,
    /// ties broken by `Group::ALL`'s table order. `None` for an empty
    /// histogram (a candidate whose matched class's nodes have no children
    /// at all — e.g. a leaf-only match).
    fn dominant(hist: &BTreeMap<String, u64>) -> Option<Group> {
        if hist.is_empty() {
            return None;
        }
        let mut sums: HashMap<Group, u64> = HashMap::new();
        for (op_name, count) in hist {
            *sums.entry(Group::of_op_name(op_name)).or_insert(0) += count;
        }
        Group::ALL
            .into_iter()
            .filter_map(|g| sums.get(&g).map(|&n| (g, n)))
            .max_by_key(|&(_, n)| n)
            .map(|(g, _)| g)
    }
}

// ============================================================================
// `budget_fraction` buckets (design doc §2.5) — reads verbatim from the
// current record's `budget_fraction` field, no proxy needed.
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum BudgetBucket {
    B0_25,
    B25_50,
    B50_100,
    B100_200,
    B200Plus,
}

impl BudgetBucket {
    const ALL: [BudgetBucket; 5] = [
        BudgetBucket::B0_25,
        BudgetBucket::B25_50,
        BudgetBucket::B50_100,
        BudgetBucket::B100_200,
        BudgetBucket::B200Plus,
    ];

    fn label(self) -> &'static str {
        match self {
            BudgetBucket::B0_25 => "[0, 0.25)",
            BudgetBucket::B25_50 => "[0.25, 0.5)",
            BudgetBucket::B50_100 => "[0.5, 1.0)",
            BudgetBucket::B100_200 => "[1.0, 2.0)",
            BudgetBucket::B200Plus => "[2.0, inf)",
        }
    }

    fn of(fraction: f64) -> BudgetBucket {
        if fraction < 0.25 {
            BudgetBucket::B0_25
        } else if fraction < 0.5 {
            BudgetBucket::B25_50
        } else if fraction < 1.0 {
            BudgetBucket::B50_100
        } else if fraction < 2.0 {
            BudgetBucket::B100_200
        } else {
            BudgetBucket::B200Plus
        }
    }
}

// ============================================================================
// `dcost` buckets (design doc §2.4) — written to spec now, NOT wired into
// `Cell::of` yet (no `dcost` field exists in the current record). See the
// module doc's "Where the new fields plug in" for the one-line change that
// activates this when `CandidateContext` lands.
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(dead_code)] // wired in once `dcost` exists on the record — see module doc.
enum DcostBucket {
    LeMinus64,
    Minus63ToMinus9,
    Minus8ToMinus1,
    Zero,
    Plus1ToPlus8,
    Plus9ToPlus63,
    GePlus64,
}

impl DcostBucket {
    #[allow(dead_code)]
    fn of(dcost: i32) -> DcostBucket {
        match dcost {
            i32::MIN..=-64 => DcostBucket::LeMinus64,
            -63..=-9 => DcostBucket::Minus63ToMinus9,
            -8..=-1 => DcostBucket::Minus8ToMinus1,
            0 => DcostBucket::Zero,
            1..=8 => DcostBucket::Plus1ToPlus8,
            9..=63 => DcostBucket::Plus9ToPlus63,
            _ => DcostBucket::GePlus64,
        }
    }
}

// ============================================================================
// The proxy cell. See module doc for exactly what this stands in for and
// exactly what changes when the real fields land.
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Cell {
    rule: RuleId,
    down_group: Option<Group>,
    budget: BudgetBucket,
}

/// |G ∪ {None}| × |BudgetBucket| — the reachable proxy-cell count for ONE
/// rule, independent of the rule's arity (unlike the design's true `DownSig`
/// grid, §3.1, which is 8/36/1 depending on arity: the proxy has only one
/// dominant-group scalar, not per-slot data, so arity cannot change its
/// reachable count). This is smaller than the design's true L1/L2 reachable
/// counts and is reported as such, never conflated with them.
const REACHABLE_PER_RULE: u64 = (Group::ALL.len() as u64 + 1) * BudgetBucket::ALL.len() as u64;

/// `Option<Group>` ranges over `G ∪ {None}` (9 values) — the one place this
/// 9-element grid is spelled out; every reachable-cell walk in this file
/// iterates it rather than re-listing the variants.
const DOWN_GROUP_OPTIONS: [Option<Group>; 9] = [
    None,
    Some(Group::Leaf),
    Some(Group::Lin),
    Some(Group::Mul),
    Some(Group::Div),
    Some(Group::Root),
    Some(Group::Trig),
    Some(Group::ExpLog),
    Some(Group::Other),
];

impl Cell {
    fn of(record: &Record) -> Cell {
        Cell {
            rule: record.rule,
            down_group: Group::dominant(&record.neighborhood_op_hist),
            budget: BudgetBucket::of(record.budget_fraction),
        }
    }
}

// ============================================================================
// Record parsing. The JSONL line IS valid JSON (`neighborhood_op_hist`'s
// keys are written through `{op:?}` on a `String`, which Debug-quotes them)
// — no hand-rolled scanning needed, `serde_json` reads the whole line.
// ============================================================================

struct Record {
    /// The firing rule's stable identity, as `gen_strict_labels` writes it.
    rule: RuleId,
    rule_name: String,
    budget_fraction: f64,
    neighborhood_op_hist: BTreeMap<String, u64>,
    label_positive: bool,
}

fn parse_record(line: &str, path: &str, line_no: usize) -> Record {
    let v: Value = serde_json::from_str(line)
        .unwrap_or_else(|e| panic!("guide_coverage_table: {path}:{line_no}: not valid JSON: {e}"));
    let field = |name: &str| -> &Value {
        v.get(name).unwrap_or_else(|| {
            panic!("guide_coverage_table: {path}:{line_no}: missing field {name:?}")
        })
    };
    let rule_id = field("rule_id")
        .as_u64()
        .unwrap_or_else(|| panic!("guide_coverage_table: {path}:{line_no}: rule_id not a u64"));
    let rule_name = field("rule_name")
        .as_str()
        .unwrap_or_else(|| panic!("guide_coverage_table: {path}:{line_no}: rule_name not a string"))
        .to_string();
    // The label is the key; the id in the row is checked against it rather
    // than trusted, so a dataset whose two halves of the identity disagree
    // stops here (docs/plans/2026-09-02-phase3-forward-port.md §2.2).
    let rule = RuleId::from_label(&rule_name);
    assert_eq!(
        rule.get(),
        rule_id,
        "guide_coverage_table: {path}:{line_no}: row names rule {rule_name:?} but carries \
         rule_id {rule_id}"
    );
    let budget_fraction = field("budget_fraction").as_f64().unwrap_or_else(|| {
        panic!("guide_coverage_table: {path}:{line_no}: budget_fraction not a number")
    });
    let hist_val = field("neighborhood_op_hist")
        .as_object()
        .unwrap_or_else(|| {
            panic!("guide_coverage_table: {path}:{line_no}: neighborhood_op_hist not an object")
        });
    let mut neighborhood_op_hist = BTreeMap::new();
    for (k, val) in hist_val {
        let n = val.as_u64().unwrap_or_else(|| {
            panic!("guide_coverage_table: {path}:{line_no}: neighborhood_op_hist[{k:?}] not a u64")
        });
        neighborhood_op_hist.insert(k.clone(), n);
    }
    let label_positive = field("label_positive").as_bool().unwrap_or_else(|| {
        panic!("guide_coverage_table: {path}:{line_no}: label_positive not a bool")
    });
    Record {
        rule,
        rule_name,
        budget_fraction,
        neighborhood_op_hist,
        label_positive,
    }
}

fn read_records(path: &str) -> Vec<Record> {
    let f = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("guide_coverage_table: cannot open {path}: {e}"));
    let reader = std::io::BufReader::new(f);
    let mut out = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.unwrap_or_else(|e| {
            panic!(
                "guide_coverage_table: {path}: line {} read error: {e}",
                i + 1
            )
        });
        if line.trim().is_empty() {
            continue;
        }
        out.push(parse_record(&line, path, i + 1));
    }
    out
}

// ============================================================================
// Aggregation: §3.2's counting rule, applied to `Cell` at whatever
// granularity `Cell::of` currently bucket at (today: the proxy; unchanged
// once the real fields land).
// ============================================================================

#[derive(Default, Clone, Copy)]
struct CellCount {
    n: u64,
    pos: u64,
}

#[derive(Default, Clone, Copy)]
struct RuleAgg {
    n: u64,
    pos: u64,
    reachable: u64,
    seen: u64,
    covered: u64,
    thin: u64,
    empty: u64,
}

struct Mint {
    /// Every rule identity -> canonical label seen (a rule absent from this
    /// mint's data still gets a row: `all_rules().len()` production rules
    /// exist, see `main`, but a JSONL-only tool has no direct handle on that
    /// count without re-running the e-graph — so the rule universe here is
    /// "every rule this mint's OWN records mention," which under-counts
    /// rules with zero applications logged. That under-count is itself
    /// reported, not hidden: see `main`'s cross-check against
    /// `pixelflow_search::math::all_rules().len()`.
    rule_names: BTreeMap<RuleId, String>,
    cells: HashMap<Cell, CellCount>,
    rule_totals: HashMap<RuleId, (u64, u64)>,
}

fn mint_records(records: &[Record]) -> Mint {
    let mut rule_names = BTreeMap::new();
    let mut cells: HashMap<Cell, CellCount> = HashMap::new();
    let mut rule_totals: HashMap<RuleId, (u64, u64)> = HashMap::new();
    for r in records {
        rule_names
            .entry(r.rule)
            .or_insert_with(|| r.rule_name.clone());
        let cell = Cell::of(r);
        let entry = cells.entry(cell).or_default();
        entry.n += 1;
        if r.label_positive {
            entry.pos += 1;
        }
        let totals = rule_totals.entry(r.rule).or_insert((0, 0));
        totals.0 += 1;
        if r.label_positive {
            totals.1 += 1;
        }
    }
    Mint {
        rule_names,
        cells,
        rule_totals,
    }
}

/// §3.2's status for one cell, given whether ITS rule has any positives at
/// all anywhere in the mint.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CellStatus {
    Empty,
    Thin,
    Covered,
}

fn cell_status(
    count: CellCount,
    rule_has_positives: bool,
    n_thin: u64,
    pos_thin: u64,
) -> CellStatus {
    if count.n == 0 {
        CellStatus::Empty
    } else if count.n < n_thin || (rule_has_positives && count.pos < pos_thin) {
        CellStatus::Thin
    } else {
        CellStatus::Covered
    }
}

fn rule_agg(mint: &Mint, rule: RuleId, n_thin: u64, pos_thin: u64) -> RuleAgg {
    let (n, pos) = mint.rule_totals.get(&rule).copied().unwrap_or((0, 0));
    let rule_has_positives = pos > 0;
    let mut agg = RuleAgg {
        n,
        pos,
        reachable: REACHABLE_PER_RULE,
        ..Default::default()
    };
    // Walk the full reachable grid for this rule (not just cells with data)
    // so empty cells are counted, not skipped.
    for down_group in DOWN_GROUP_OPTIONS {
        for budget in BudgetBucket::ALL {
            let cell = Cell {
                rule,
                down_group,
                budget,
            };
            let count = mint.cells.get(&cell).copied().unwrap_or_default();
            match cell_status(count, rule_has_positives, n_thin, pos_thin) {
                CellStatus::Empty => agg.empty += 1,
                CellStatus::Thin => {
                    agg.thin += 1;
                    agg.seen += 1;
                }
                CellStatus::Covered => {
                    agg.covered += 1;
                    agg.seen += 1;
                }
            }
        }
    }
    agg
}

/// The task's headline number, computed directly (not via `cell_status`,
/// which also folds in `pos_thin`): share of reachable proxy cells, over
/// every rule in `rows`, whose observation count meets `n_thin` on its own.
fn share_reachable_cells_at_least_n(
    mint: &Mint,
    rows: &[(RuleId, String, RuleAgg)],
    n_thin: u64,
) -> f64 {
    let mut at_least_n = 0u64;
    let mut total = 0u64;
    for (id, _, _) in rows {
        for down_group in DOWN_GROUP_OPTIONS {
            for budget in BudgetBucket::ALL {
                let cell = Cell {
                    rule: *id,
                    down_group,
                    budget,
                };
                total += 1;
                if mint.cells.get(&cell).map(|c| c.n).unwrap_or(0) >= n_thin {
                    at_least_n += 1;
                }
            }
        }
    }
    if total > 0 {
        at_least_n as f64 / total as f64
    } else {
        0.0
    }
}

// ============================================================================
// Trig-context comparison (the task's headline): the rules JP's framing
// names, and where their proxy cells actually land.
// ============================================================================

/// (rule_idx, op, parity) for the trig-relevant parity instantiations —
/// `parity_rules()` (`pixelflow-search/src/math/parity.rs`) is
/// `[Sin(odd,30), Tan(odd,31), Asin(odd,32), Atan(odd,33), Cos(even,34),
/// Abs(even,35)]`, immediately after `algebra_rules()`'s 30 rules (0..29)
/// — confirmed against this mint's own `rule_name` field (both
/// "even-negation" and "odd-negation" appear at exactly these indices in
/// the Round 1 dataset). `Abs`(35) is EXCLUDED: it is even-negation, but
/// not a trig function, so it is not part of "even/odd-negation on trig."
/// `trig_rules()` immediately follows at 36..40:
/// `[sin-angle-addition(36), cos-angle-addition(37),
/// reverse-angle-addition(38), half-angle-product(39), pythagorean(40)]`.
/// `pythagorean` is not in the task's named list but IS JP's own leading
/// example ("sin²+cos²→1 pays when it feeds a sqrt or divide" — design doc
/// §1.2 UP) — included as a labeled bonus row, not silently substituted for
/// a named rule.
/// `(canonical rule label, this table's display label)`.
///
/// Keyed by identity, not by the `all_rules()` index the doc comment above
/// quotes: the indices are kept in the prose because that is where the
/// selection came from, but a same-length reorder of `all_rules()` would
/// repoint every one of them silently
/// (docs/plans/2026-09-02-phase3-forward-port.md §2.2), and this table's
/// whole purpose is to name specific rewrites.
const TRIG_RULES: &[(&str, &str)] = &[
    ("odd-negation(Sin)", "odd-negation (Sin)"),
    ("odd-negation(Tan)", "odd-negation (Tan)"),
    ("odd-negation(Asin)", "odd-negation (Asin)"),
    ("odd-negation(Atan)", "odd-negation (Atan)"),
    ("even-negation(Cos)", "even-negation (Cos)"),
    ("sin-angle-addition", "sin-angle-addition"),
    ("cos-angle-addition", "cos-angle-addition"),
    ("reverse-angle-addition", "reverse-angle-addition"),
    ("half-angle-product", "half-angle-product"),
    ("pythagorean", "pythagorean (bonus — JP's example)"),
];

struct TrigRow {
    rule: RuleId,
    label: &'static str,
    n: u64,
    pos: u64,
    seen_cells: u64,
    seen_trig_down_cells: u64,
    n_trig_down: u64,
    down_group_counts: BTreeMap<&'static str, u64>,
    budget_counts: BTreeMap<&'static str, u64>,
}

fn trig_report(mint: &Mint) -> Vec<TrigRow> {
    TRIG_RULES
        .iter()
        .map(|&(rule_label, label)| {
            let rule = RuleId::from_label(rule_label);
            let mut seen_cells = 0u64;
            let mut seen_trig_down_cells = 0u64;
            let mut n = 0u64;
            let mut pos = 0u64;
            let mut n_trig_down = 0u64;
            let mut down_group_counts: BTreeMap<&'static str, u64> = BTreeMap::new();
            let mut budget_counts: BTreeMap<&'static str, u64> = BTreeMap::new();
            for down_group in DOWN_GROUP_OPTIONS {
                let group_label = down_group.map(Group::label).unwrap_or("NONE");
                for budget in BudgetBucket::ALL {
                    let cell = Cell {
                        rule,
                        down_group,
                        budget,
                    };
                    if let Some(count) = mint.cells.get(&cell)
                        && count.n > 0
                    {
                        seen_cells += 1;
                        n += count.n;
                        pos += count.pos;
                        *down_group_counts.entry(group_label).or_insert(0) += count.n;
                        *budget_counts.entry(budget.label()).or_insert(0) += count.n;
                        if down_group == Some(Group::Trig) {
                            seen_trig_down_cells += 1;
                            n_trig_down += count.n;
                        }
                    }
                }
            }
            TrigRow {
                rule,
                label,
                n,
                pos,
                seen_cells,
                seen_trig_down_cells,
                n_trig_down,
                down_group_counts,
                budget_counts,
            }
        })
        .collect()
}

// ============================================================================
// Report rendering.
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn write_markdown(
    path: &str,
    args: &Args,
    train: &Mint,
    dev: &Mint,
    train_n_train: usize,
    dev_n_dev: usize,
) {
    let mut md = String::new();
    md.push_str("# Guide candidate-context proxy coverage — Round 1\n\n");
    md.push_str(&format!(
        "Generated by `guide_coverage_table` over `{}` ({} records) and `{}` ({} records). \
         `n_thin = {}`, `pos_thin = {}` (design doc §3.2, applied verbatim to the proxy cell \
         defined below).\n\n",
        args.train, train_n_train, args.dev, dev_n_dev, args.n_thin, args.pos_thin
    ));
    md.push_str(
        "**Scope note:** this is the PROXY cell — `(rule_idx, dominant(neighborhood_op_hist), \
         budget_bucket)` — not the design's full `CandidateCell` (§2.3), because \
         `CandidateContext` (DOWN/UP/dcost/on_best_path) has not landed in this record schema \
         yet. See the tool's module doc for exactly what changes when it does. Reachable count \
         per rule here is a fixed 45 (`9 down-group options × 5 budget buckets`), smaller and \
         differently structured than the design's true L1/L2 counts (§3.1) — the two are never \
         conflated below.\n\n",
    );

    for (mint, label) in [(train, "TRAIN"), (dev, "DEV")] {
        md.push_str(&format!(
            "## {label}: `{}`\n\n",
            if label == "TRAIN" {
                &args.train
            } else {
                &args.dev
            }
        ));
        let mut total_reachable = 0u64;
        let mut total_seen = 0u64;
        let mut total_covered = 0u64;
        let mut total_thin = 0u64;
        let mut total_empty = 0u64;
        let mut rows: Vec<(RuleId, String, RuleAgg)> = mint
            .rule_names
            .iter()
            .map(|(&id, name)| {
                let agg = rule_agg(mint, id, args.n_thin, args.pos_thin);
                (id, name.clone(), agg)
            })
            .collect();
        // By canonical label: a stable, human-meaningful order that does not
        // depend on any rule vector's positions.
        rows.sort_by(|a, b| a.1.cmp(&b.1));
        for (_, _, agg) in &rows {
            total_reachable += agg.reachable;
            total_seen += agg.seen;
            total_covered += agg.covered;
            total_thin += agg.thin;
            total_empty += agg.empty;
        }
        let share_covered_or_thin = if total_reachable > 0 {
            (total_seen as f64) / (total_reachable as f64)
        } else {
            0.0
        };
        let share_covered = if total_reachable > 0 {
            (total_covered as f64) / (total_reachable as f64)
        } else {
            0.0
        };
        md.push_str(&format!(
            "**Global (over {} rules that fired at least once):** reachable {total_reachable}, \
             seen (n>0) {total_seen} ({:.2}%), covered {total_covered} ({:.2}%), thin \
             {total_thin}, empty {total_empty}.\n\n",
            rows.len(),
            share_covered_or_thin * 100.0,
            share_covered * 100.0,
        ));
        let headline_pct = share_reachable_cells_at_least_n(mint, &rows, args.n_thin);
        md.push_str(&format!(
            "**Headline (design §3.2's threshold, `n >= {}`):** {:.2}% of reachable proxy cells \
             have at least {} observations.\n\n",
            args.n_thin,
            headline_pct * 100.0,
            args.n_thin,
        ));

        md.push_str(
            "| idx | rule | reachable | seen | covered | thin | empty | n | pos | rate |\n",
        );
        md.push_str("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
        for (idx, name, agg) in &rows {
            let rate = if agg.n > 0 {
                agg.pos as f64 / agg.n as f64 * 100.0
            } else {
                0.0
            };
            md.push_str(&format!(
                "| {idx} | {name} | {} | {} | {} | {} | {} | {} | {} | {:.3}% |\n",
                agg.reachable, agg.seen, agg.covered, agg.thin, agg.empty, agg.n, agg.pos, rate
            ));
        }
        md.push('\n');
    }

    md.push_str(
        "## Set difference: cells DEV touches that TRAIN never saw (design §3.3 item 4)\n\n",
    );
    md.push_str(
        "Per rule present in DEV: proxy cells with `n > 0` in DEV whose `n == 0` in TRAIN — \
         the out-of-support trigger rate a live deployment would hit before any Guide model \
         runs.\n\n",
    );
    md.push_str("| idx | rule | dev cells seen | dev cells absent from train | trigger rate |\n");
    md.push_str("|---:|---|---:|---:|---:|\n");
    {
        let mut dev_rules: Vec<(String, RuleId)> = dev
            .rule_names
            .iter()
            .map(|(&id, name)| (name.clone(), id))
            .collect();
        dev_rules.sort();
        for (name, id) in dev_rules {
            let mut dev_seen = 0u64;
            let mut absent_from_train = 0u64;
            for down_group in DOWN_GROUP_OPTIONS {
                for budget in BudgetBucket::ALL {
                    let cell = Cell {
                        rule: id,
                        down_group,
                        budget,
                    };
                    let dev_n = dev.cells.get(&cell).map(|c| c.n).unwrap_or(0);
                    if dev_n > 0 {
                        dev_seen += 1;
                        let train_n = train.cells.get(&cell).map(|c| c.n).unwrap_or(0);
                        if train_n == 0 {
                            absent_from_train += 1;
                        }
                    }
                }
            }
            let trigger_rate = if dev_seen > 0 {
                absent_from_train as f64 / dev_seen as f64 * 100.0
            } else {
                0.0
            };
            md.push_str(&format!(
                "| {} | {name} | {dev_seen} | {absent_from_train} | {trigger_rate:.1}% |\n",
                id.get()
            ));
        }
    }
    md.push('\n');

    md.push_str(
        "## The trig rules: what a trig-dominant kernel would need vs what Round 1 saw\n\n",
    );
    md.push_str(
        "JP's objection in one number: for each of these rules, `seen (down=TRIG)` counts \
         proxy cells whose one-hop child-op histogram is TRIG-dominant — the context a \
         spherical-harmonics/lighting kernel would overwhelmingly present these rules with. \
         `seen (any)` is every proxy cell this rule touched, of 45 reachable. The `down-group \
         breakdown` column names which contexts this rule's applications actually landed in — \
         when TRIG is absent or a minority, this rule's training signal in Round 1's classical \
         corpus comes almost entirely from non-trig neighborhoods.\n\n",
    );
    for (mint, label) in [(train, "TRAIN"), (dev, "DEV")] {
        md.push_str(&format!("### {label}\n\n"));
        md.push_str(
            "| idx | rule | n | pos | seen (any, /45) | seen (down=TRIG, /5) | n in down=TRIG | down-group breakdown (n) | budget breakdown (n) |\n",
        );
        md.push_str("|---:|---|---:|---:|---:|---:|---:|---|---|\n");
        for row in trig_report(mint) {
            let breakdown: String = row
                .down_group_counts
                .iter()
                .map(|(g, n)| format!("{g}={n}"))
                .collect::<Vec<_>>()
                .join(", ");
            let budget_breakdown: String = row
                .budget_counts
                .iter()
                .map(|(b, n)| format!("{b}={n}"))
                .collect::<Vec<_>>()
                .join(", ");
            md.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                row.rule,
                row.label,
                row.n,
                row.pos,
                row.seen_cells,
                row.seen_trig_down_cells,
                row.n_trig_down,
                if breakdown.is_empty() {
                    "(never fired)".to_string()
                } else {
                    breakdown
                },
                if budget_breakdown.is_empty() {
                    "—".to_string()
                } else {
                    budget_breakdown
                },
            ));
        }
        md.push('\n');
    }

    std::fs::write(path, md)
        .unwrap_or_else(|e| panic!("guide_coverage_table: cannot write {path}: {e}"));
    eprintln!("guide_coverage_table: wrote {path}");
}

fn write_json(path: &str, args: &Args, train: &Mint, dev: &Mint) {
    let mut json = String::from("{\n");
    json.push_str(&format!("  \"n_thin\": {},\n", args.n_thin));
    json.push_str(&format!("  \"pos_thin\": {},\n", args.pos_thin));
    json.push_str(&format!(
        "  \"reachable_per_rule\": {REACHABLE_PER_RULE},\n"
    ));
    let write_mint = |json: &mut String, mint: &Mint| {
        json.push_str("    \"per_rule\": [\n");
        let mut rows: Vec<(String, RuleId)> = mint
            .rule_names
            .iter()
            .map(|(&id, name)| (name.clone(), id))
            .collect();
        rows.sort();
        for (i, (name, id)) in rows.iter().enumerate() {
            let agg = rule_agg(mint, *id, args.n_thin, args.pos_thin);
            json.push_str(&format!(
                "      {{\"rule_id\": {}, \"rule\": {name:?}, \"reachable\": {}, \
                 \"seen\": {}, \"covered\": {}, \"thin\": {}, \"empty\": {}, \"n\": {}, \
                 \"pos\": {}}}{}\n",
                id.get(),
                agg.reachable,
                agg.seen,
                agg.covered,
                agg.thin,
                agg.empty,
                agg.n,
                agg.pos,
                if i + 1 < rows.len() { "," } else { "" }
            ));
        }
        json.push_str("    ]\n");
    };
    json.push_str("  \"train\": {\n");
    write_mint(&mut json, train);
    json.push_str("  },\n");
    json.push_str("  \"dev\": {\n");
    write_mint(&mut json, dev);
    json.push_str("  }\n");
    json.push_str("}\n");
    let mut f = std::fs::File::create(path)
        .unwrap_or_else(|e| panic!("guide_coverage_table: cannot create {path}: {e}"));
    f.write_all(json.as_bytes())
        .unwrap_or_else(|e| panic!("guide_coverage_table: cannot write {path}: {e}"));
    eprintln!("guide_coverage_table: wrote {path}");
}

fn main() {
    let args = Args::parse();

    let train_records = read_records(&args.train);
    let dev_records = read_records(&args.dev);
    eprintln!(
        "guide_coverage_table: {} TRAIN records, {} DEV records",
        train_records.len(),
        dev_records.len()
    );

    // Cross-check: how many production rules exist vs how many this mint's
    // own data ever mentions — a rule with zero firings in Round 1's mint
    // is itself a coverage finding (not silently absent from the report).
    let production_rule_count = pixelflow_search::math::all_rules().len();

    let train_mint = mint_records(&train_records);
    let dev_mint = mint_records(&dev_records);

    let train_seen_rules: HashSet<RuleId> = train_mint.rule_names.keys().copied().collect();
    let dev_seen_rules: HashSet<RuleId> = dev_mint.rule_names.keys().copied().collect();
    eprintln!(
        "guide_coverage_table: {production_rule_count} production rules exist; TRAIN mint fired \
         {} of them, DEV mint fired {} of them",
        train_seen_rules.len(),
        dev_seen_rules.len()
    );

    if let Some(out_md) = &args.out_md {
        write_markdown(
            out_md,
            &args,
            &train_mint,
            &dev_mint,
            train_records.len(),
            dev_records.len(),
        );
    }
    if let Some(out_json) = &args.out_json {
        write_json(out_json, &args, &train_mint, &dev_mint);
    }
    if args.out_md.is_none() && args.out_json.is_none() {
        eprintln!(
            "guide_coverage_table: neither --out-md nor --out-json given — nothing written, \
             this run only validated the input files parse and reported the rule-count \
             cross-check above"
        );
    }
}
