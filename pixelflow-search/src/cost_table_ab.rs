//! A/B the extraction output of two latency-prior cost tables on real kernels.
//!
//! A cost-table refresh is a production behavior change: extraction is argmin
//! over the table, so new coefficients move which term comes out. This probe
//! quantifies that move on the 206 real-kernel `.arena` dumps.
//!
//! The one methodological point that makes the comparison honest: **saturation
//! is run once per kernel and both tables extract from the same e-graph.**
//! `Optimizer::run` takes no cost model (the default, unguided path), so the
//! saturated e-graph is a function of the arena and the budget alone. Extracting
//! twice from one e-graph removes saturation nondeterminism from the diff, so
//! every difference reported here is attributable to the table and nothing else.
//!
//! Scoring: each extracted term is materialized (`choices_to_arena`) and priced
//! under BOTH tables. The regression test that matters is
//! `cost_new(term_new) > cost_new(term_old)` — the new table is the better
//! description of the machine, so it is the referee for both terms. A kernel
//! where the new table's own extraction is worse *under the new table* means
//! the greedy chooser lost ground, and that is a hold-the-PR signal.
//!
//! Read-only: nothing here changes production behavior.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use pixelflow_ir::arena::ExprNode;
use pixelflow_ir::kind::OpMap;
use pixelflow_ir::{ExprArena, ExprId, LatticeShape, OpKind};

use crate::arena_corpus::{category_of, load_arena_dump, median, percentile};
use crate::egraph::cost::latency_prior_cycles;
use crate::egraph::extract::extract_dag_scoped;
use crate::egraph::{Budget, CostModel, EClassId, Extraction, Optimizer, choices_to_arena};
use crate::runtime::{arena_to_egraph, reachable_count};

/// The Round 1 table (e4388c54, ~2026-08-14) — the coefficients this refresh
/// replaces. Frozen here as a literal so the A/B has a fixed "before" even
/// after `latency_prior_cycles` is edited.
///
/// Kept as an exhaustive `match` for the same reason the live table is: a new
/// op is a compile error until it is priced, so this snapshot cannot silently
/// read a neighbour's number.
fn round1_cycles() -> OpMap<usize> {
    OpMap::from_fn(|op| match op {
        OpKind::Var => 0,
        OpKind::Const => 0,
        OpKind::Add => 4,
        OpKind::Sub => 4,
        OpKind::Mul => 5,
        OpKind::Div => 11,
        OpKind::Neg => 3,
        OpKind::Sqrt => 15,
        OpKind::Rsqrt => 21,
        OpKind::Abs => 3,
        OpKind::Min => 3,
        OpKind::Max => 3,
        OpKind::MulAdd => 5,
        OpKind::Recip => 16,
        OpKind::Floor => 4,
        OpKind::Ceil => 4,
        OpKind::Round => 4,
        OpKind::Sin => 70,
        OpKind::Cos => 75,
        OpKind::Tan => 87,
        OpKind::Asin => 103,
        OpKind::Acos => 103,
        OpKind::Atan => 79,
        OpKind::Exp => 75,
        OpKind::Exp2 => 69,
        OpKind::Ln => 128,
        OpKind::Log2 => 122,
        OpKind::Log10 => 134,
        OpKind::Atan2 => 79,
        OpKind::Pow => 196,
        OpKind::Lt => 3,
        OpKind::Le => 3,
        OpKind::Gt => 3,
        OpKind::Ge => 3,
        OpKind::Eq => 3,
        OpKind::Ne => 3,
        OpKind::Select => 4,
        OpKind::Tuple => 0,
        OpKind::TruncToInt => 1,
        OpKind::IntToFloat => 1,
        OpKind::IAdd => 1,
        OpKind::Shl => 1,
        OpKind::Shr => 1,
        OpKind::BitAnd => 1,
        OpKind::BitOr => 1,
        OpKind::Dwrt => 1000,
        OpKind::Buffer => 0,
        OpKind::Gather => 10,
        OpKind::RawGather => 10,
        OpKind::Reduce => 0,
    })
}

fn model_from(costs: OpMap<usize>) -> CostModel {
    let mut m = CostModel::zero();
    for op in OpKind::all() {
        m.set_cost(op, costs[op]);
    }
    m
}

/// DAG cost of a materialized arena under `costs`: each reachable node priced
/// once. This is the same walk `ExtractedDAG::dag_cost` is pinned against
/// (`dag_cost_equals_the_materialized_arenas_cost`), applied to a term that was
/// chosen under a *different* table than the one pricing it — which is exactly
/// what cross-scoring needs and what the struct's own field cannot give.
fn arena_dag_cost(arena: &ExprArena, root: ExprId, costs: &CostModel) -> usize {
    let mut seen = vec![false; arena.nodes_raw().len()];
    let mut stack = vec![root];
    let mut total = 0usize;
    while let Some(id) = stack.pop() {
        if core::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        let kind = match arena.node(id) {
            ExprNode::Var(_) | ExprNode::Const(_) | ExprNode::Buffer(_) => None,
            ExprNode::Unary(k, _) | ExprNode::Binary(k, _, _) | ExprNode::Ternary(k, _, _, _) => {
                Some(*k)
            }
            other => panic!("unexpected extracted node {other:?}"),
        };
        if let Some(k) = kind {
            total = total.saturating_add(costs.cost(k));
        }
        stack.extend(arena.children(id));
    }
    total
}

/// Canonical structural rendering of an extracted term, for "did the extracted
/// term change at all?". Two terms are the same iff this string is the same.
fn term_signature(arena: &ExprArena, root: ExprId) -> String {
    let mut out = String::new();
    let mut memo: HashMap<u32, u32> = HashMap::new();
    let mut order: Vec<ExprId> = Vec::new();
    // Post-order over the reachable DAG, each node emitted once.
    let mut stack = vec![(root, false)];
    let mut seen = vec![false; arena.nodes_raw().len()];
    while let Some((id, done)) = stack.pop() {
        if done {
            let n = memo.len() as u32;
            memo.insert(id.0, n);
            order.push(id);
            continue;
        }
        if core::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        stack.push((id, true));
        for c in arena.children(id) {
            stack.push((c, false));
        }
    }
    for id in order {
        let r = |c: ExprId| memo[&c.0];
        match arena.node(id) {
            ExprNode::Var(v) => write!(out, "V{v};").expect("string write"),
            ExprNode::Const(c) => write!(out, "C{:08x};", c.to_bits()).expect("string write"),
            ExprNode::Buffer(b) => write!(out, "B{};", b.0).expect("string write"),
            ExprNode::Unary(k, a) => write!(out, "{k:?}({});", r(*a)).expect("string write"),
            ExprNode::Binary(k, a, b) => {
                write!(out, "{k:?}({},{});", r(*a), r(*b)).expect("string write")
            }
            ExprNode::Ternary(k, a, b, c) => {
                write!(out, "{k:?}({},{},{});", r(*a), r(*b), r(*c)).expect("string write")
            }
            other => panic!("unexpected extracted node {other:?}"),
        }
    }
    out
}

struct Row {
    name: String,
    category: &'static str,
    changed: bool,
    old_under_old: usize,
    new_under_new: usize,
    old_under_new: usize,
    new_under_old: usize,
}

impl Row {
    /// Cost delta under the NEW table (the referee): negative is an
    /// improvement, positive is a regression the refresh caused.
    fn delta_under_new(&self) -> i64 {
        self.new_under_new as i64 - self.old_under_new as i64
    }

    fn pct_under_new(&self) -> f64 {
        if self.old_under_new == 0 {
            return 0.0;
        }
        100.0 * self.delta_under_new() as f64 / self.old_under_new as f64
    }
}

fn measure_kernel(name: &str, category: &'static str, arena: &ExprArena, root: ExprId) -> Row {
    // The same two lowering passes `optimize_runtime_arena_uncached` runs
    // before the e-graph sees the arena.
    let (arena, root) = pixelflow_ir::passes::lower_dwrt_owned(arena, root)
        .unwrap_or_else(|e| panic!("{name}: lower_dwrt failed: {e:?}"));
    let (arena, root) = pixelflow_ir::passes::expand_reduce_owned(&arena, root);
    let node_count = reachable_count(&arena, root);

    // ONE saturation, shared by both tables. `Optimizer::run` takes no cost
    // model, so this e-graph is a function of (arena, budget) alone.
    let mut optimizer = Optimizer::production().budget(Budget::Production);
    let mut egraph = optimizer.egraph();
    let mut memo: HashMap<ExprId, EClassId> = HashMap::new();
    let root_class = arena_to_egraph(&arena, root, &mut egraph, &mut memo)
        .unwrap_or_else(|| panic!("{name}: arena_to_egraph returned None (unsupported node)"));
    let _ = optimizer.run(&mut egraph, root_class, node_count);

    let old = model_from(round1_cycles());
    let new = model_from(latency_prior_cycles());

    let dag_old = extract_dag_scoped(&egraph, root_class, &old, LatticeShape::POINT);
    let dag_new = extract_dag_scoped(&egraph, root_class, &new, LatticeShape::POINT);

    let (a_old, r_old) = choices_to_arena(&Extraction::from_dp(
        &egraph,
        root_class,
        dag_old.choices.clone(),
    ));
    let (a_new, r_new) = choices_to_arena(&Extraction::from_dp(
        &egraph,
        root_class,
        dag_new.choices.clone(),
    ));

    Row {
        name: name.to_string(),
        category,
        changed: term_signature(&a_old, r_old) != term_signature(&a_new, r_new),
        old_under_old: arena_dag_cost(&a_old, r_old, &old),
        new_under_new: arena_dag_cost(&a_new, r_new, &new),
        old_under_new: arena_dag_cost(&a_old, r_old, &new),
        new_under_old: arena_dag_cost(&a_new, r_new, &old),
    }
}

/// THE A/B: how does refreshing the latency-prior table move extraction on
/// real kernels?
///
/// Writes `docs/results/2026-09-02-latency-prior-remeasure.{csv,json}` rows.
#[test]
#[ignore = "offline measurement: PIXELFLOW_COST_AB_ARENA_DIR=<dir of .arena dumps> PIXELFLOW_COST_AB_OUT=<prefix> cargo test -p pixelflow-search --release --lib -- --ignored cost_table_ab_measurement --nocapture"]
fn cost_table_ab_measurement() {
    let dir = PathBuf::from(
        std::env::var("PIXELFLOW_COST_AB_ARENA_DIR")
            .expect("PIXELFLOW_COST_AB_ARENA_DIR must be set"),
    );
    let out_prefix =
        std::env::var("PIXELFLOW_COST_AB_OUT").expect("PIXELFLOW_COST_AB_OUT must be set");

    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().map(|e| e == "arena").unwrap_or(false))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no .arena files in {}", dir.display());

    let mut rows: Vec<Row> = Vec::new();
    for path in &paths {
        let (name, arena, root) = load_arena_dump(path);
        let category = category_of(&path.file_name().expect("file name").to_string_lossy());
        let row = measure_kernel(&name, category, &arena, root);
        eprintln!(
            "{:<32} {:<11} changed={:<5} old/new {:>8} -> {:>8}  ({:+.2}%)",
            row.name,
            row.category,
            row.changed,
            row.old_under_new,
            row.new_under_new,
            row.pct_under_new(),
        );
        rows.push(row);
    }

    let changed = rows.iter().filter(|r| r.changed).count();
    let better = rows.iter().filter(|r| r.delta_under_new() < 0).count();
    let worse = rows.iter().filter(|r| r.delta_under_new() > 0).count();
    let mut pcts: Vec<f64> = rows.iter().map(Row::pct_under_new).collect();

    let mut csv = String::from(
        "name,category,changed,old_under_old,new_under_old,old_under_new,new_under_new,delta_under_new,pct_under_new\n",
    );
    for r in &rows {
        writeln!(
            csv,
            "{},{},{},{},{},{},{},{},{:.4}",
            r.name,
            r.category,
            r.changed,
            r.old_under_old,
            r.new_under_old,
            r.old_under_new,
            r.new_under_new,
            r.delta_under_new(),
            r.pct_under_new(),
        )
        .expect("string write");
    }
    std::fs::write(format!("{out_prefix}.csv"), csv).expect("write csv");

    let json = format!(
        "{{\"kernels\":{},\"changed_term\":{},\"better_under_new\":{},\"worse_under_new\":{},\
         \"median_pct_under_new\":{:.4},\"p95_pct_under_new\":{:.4},\"max_pct_under_new\":{:.4}}}\n",
        rows.len(),
        changed,
        better,
        worse,
        median(&mut pcts.clone()),
        percentile(&mut pcts.clone(), 95.0),
        pcts.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    std::fs::write(format!("{out_prefix}.json"), json).expect("write json");

    eprintln!(
        "\n=== {} kernels: {changed} changed term, {better} better, {worse} worse (referee = NEW table) ===",
        rows.len()
    );
    eprintln!(
        "median {:+.3}%  p95 {:+.3}%  max {:+.3}%",
        median(&mut pcts.clone()),
        percentile(&mut pcts.clone(), 95.0),
        pcts.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::{EGraph, ENode};

    /// The whole A/B rests on `term_signature` being able to tell two terms
    /// apart. A signature function that returned a constant would report "0
    /// kernels changed" no matter what the tables said — the exact shape of
    /// false negative this measurement is most exposed to — so pin that it
    /// actually discriminates, and that it is blind to node *numbering*
    /// (the same term built in a different push order is the same term).
    #[test]
    fn term_signature_distinguishes_terms_and_ignores_node_order() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let two = a.push_const(2.0);
        let mul = a.push_binary(OpKind::Mul, x, two);

        // Same term, built with the const pushed first.
        let mut b = ExprArena::new();
        let two_b = b.push_const(2.0);
        let x_b = b.push_var(0);
        let mul_b = b.push_binary(OpKind::Mul, x_b, two_b);

        // A genuinely different term: Add where the first two used Mul.
        let mut c = ExprArena::new();
        let x_c = c.push_var(0);
        let two_c = c.push_const(2.0);
        let add_c = c.push_binary(OpKind::Add, x_c, two_c);

        assert_eq!(
            term_signature(&a, mul),
            term_signature(&b, mul_b),
            "the same term built in a different push order must sign the same"
        );
        assert_ne!(
            term_signature(&a, mul),
            term_signature(&c, add_c),
            "Mul and Add are different terms and must sign differently"
        );
    }

    /// `arena_dag_cost` must read the table it is handed, not a table baked in
    /// at extraction time — cross-scoring one table's term under the other is
    /// the only thing that can catch a refresh making a kernel worse.
    #[test]
    fn arena_dag_cost_follows_the_table_it_is_given() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let s = a.push_unary(OpKind::Sqrt, x);

        let cheap = model_from(latency_prior_cycles());
        let mut dear = model_from(latency_prior_cycles());
        dear.set_cost(OpKind::Sqrt, 4000);

        assert_eq!(arena_dag_cost(&a, s, &dear), 4000);
        assert_ne!(
            arena_dag_cost(&a, s, &cheap),
            arena_dag_cost(&a, s, &dear),
            "one term priced under two tables must give two prices"
        );
    }

    /// A DAG prices a shared subterm once. If this walked the tree instead,
    /// every cost in the A/B would be inflated on exactly the kernels where
    /// sharing matters most.
    #[test]
    fn arena_dag_cost_prices_a_shared_subterm_once() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let s = a.push_unary(OpKind::Sqrt, x);
        let sq = a.push_binary(OpKind::Mul, s, s);

        let m = model_from(latency_prior_cycles());
        let expect = m.cost(OpKind::Sqrt) + m.cost(OpKind::Mul);
        assert_eq!(arena_dag_cost(&a, sq, &m), expect);
    }
}
