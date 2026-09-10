//! Saturation delta economics: does the Stockfish incrementality argument
//! hold for the Guide's candidate space?
//!
//! # Background
//!
//! The extraction-head program (Phase 2) refuted NNUE incrementality for
//! *extraction* candidates: sibling candidates at one e-class differ by a
//! median 44.9% of their edge multiset
//! (`docs/plans/2026-08-17-egraph-vsa-nnue-research-notes.md`), so an
//! incremental accumulator over extraction candidates buys ~2x, not the
//! chess-engine ~98% (`docs/plans/2026-08-17-cost-model-domain.md`, A6).
//!
//! The Guide (Phase 3) evaluates a *different* object: not sibling
//! extraction candidates, but the e-graph state as rewrite rules fire during
//! saturation — which is monotone and append-only (nodes are created once,
//! via `EGraph::add`'s memo-miss path, and never removed;
//! `pixelflow-search/src/egraph/provenance.rs` module doc). This binary
//! measures whether that structural difference actually buys the
//! incrementality the extraction study refuted, using the same rule set,
//! the same production e-graph, and a real (not synthetic) corpus.
//!
//! # Method
//!
//! **Iteration-level driving uses the real production algorithm, not a
//! reimplementation.** Each round calls `EGraph::saturate_with_limits(1,
//! max_classes, big_timeout)` — literally one outer round of the exact
//! batched algorithm `EGraph::saturate()`/`saturate_with_budget` drive in
//! production, just interrupted after every round instead of running all of
//! them in one call. `match_counts` (the "journal machinery" the task names)
//! is cleared and re-read every round for the exact per-iteration candidate
//! count. `big_timeout` is a generous, never-hit backstop (this binary times
//! nothing and reports nothing time-based); the real stopping conditions are
//! `stats.total_unions == 0` (quiescence — see the phrasing note below),
//! `--max-iterations`, and `--max-classes`.
//!
//! An earlier version of this harness drove single applications directly via
//! `EGraph::apply_single_rule`, scanning matches once per round and firing
//! the whole stale list serially. That measured a real phenomenon — see
//! "A rejected design" below — but it is not what this binary reports, and
//! it is not "standard saturation": the batched algorithm above is.
//!
//! **Per-application attribution is reconstructed post-hoc, exactly, with no
//! extra e-graph walks during the run.** `Provenance` is an append-only log:
//! `origins()` (added for this measurement — see
//! `pixelflow-search/src/egraph/provenance.rs`, purely additive, no existing
//! accessor's behavior touched) maps every `ENodeId` ever minted to the
//! `ApplicationId` that created it (or `Seed`), and nodes are never deleted
//! or mutated once created — only ever relocated between class vectors by
//! `union`'s `extend`. So, after the whole run:
//!
//! 1. One O(final graph size) walk builds `ENodeId -> arity` (arity =
//!    `ENode::children().len()`, i.e. this node's own edge count in a
//!    `GraphAccumulator`-style multiset).
//! 2. One O(final origin count) pass over `origins()` groups `ENodeId`s by
//!    the `ApplicationId` that created them.
//! 3. `applications()` is already in firing order (`ApplicationId` is
//!    assigned sequentially at `record_application` time, and node creation
//!    for application N completes — synchronously, within
//!    `apply_action_from_rule` — before application N+1 is ever recorded).
//!    Replaying that order while accumulating (1)+(2)'s per-application
//!    node/edge counts reconstructs the exact node/edge total at every point
//!    in the run — no repeated graph walks needed, because nothing before
//!    the current point in the replay can ever change (append-only).
//!
//! This is exact for `nodes_added` and `edges_added`. It is **not** exact
//! for unions: `UnionEvent` records a `step` and an optional `rule_idx` but
//! never an `ApplicationId` (`EGraph::union`'s call to `record_union`), so
//! which of several same-step applications a given union belongs to is not
//! recoverable — a real limitation of the batched algorithm's provenance,
//! not of this harness. Unions are instead reported per **iteration**
//! (`Provenance::union_count()` before/after each round — exact at that
//! granularity) alongside `stats.total_unions` (the narrower "distinct
//! committed rewrite actions" count `SaturationResult` already uses
//! elsewhere, which excludes congruence-only merges discovered by the
//! round's closing `rebuild()`).
//!
//! All deltas are reported **as a fraction of the graph size immediately
//! before** (edges as the primary denominator, since `GraphAccumulator`'s
//! own cost scales with edges) — the direct analogue of the extraction
//! study's "symmetric difference / graph size" fraction, computed here over
//! sequential monotone growth instead of sibling-candidate comparison.
//!
//! # A rejected design: single-stepping goes stale fast
//!
//! Scanning `find_rewrite_matches()` once and then firing every match in
//! that list serially via `apply_single_rule` measured a **98.6% stale
//! rate** in a 20-expression pilot: after the first application in a round,
//! nearly every other match in that same stale snapshot no longer applies
//! (`apply_single_rule` re-derives the action from the *current* node at
//! the *current* index and returns `false` cleanly when it no longer
//! matches — no panic, no silent corruption, just a wasted eval). That is a
//! real, useful negative result about naive one-at-a-time firing without a
//! rescan between decisions — see the design doc's implications — but it is
//! a property of that firing strategy, not of "standard saturation", so it
//! is not what this binary's headline numbers measure.
//!
//! # A phrasing note
//!
//! This optimizer is budget-only by design; nothing here claims a certified
//! fixpoint. "Quiescence" below means only the diagnostic condition
//! `stats.total_unions == 0` for one round — no candidate rule matched
//! anything — not a proof the e-graph is closed under the rule set.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training \
//!     --bin guide_scope_saturation_delta -- \
//!     --corpus-dir pixelflow-pipeline/data --limit 0 \
//!     --out docs/results/2026-08-30-guide-scope-saturation-delta.json
//! ```
//! (`--limit 0` = no cap, all of `corpus_dev.bin`; the `--limit`/`--min-expressions`
//! default of 200 is a fast smoke-test size, not the reported measurement.)

use std::collections::{BTreeMap, HashMap};
use std::io::Write as _;
use std::path::PathBuf;

use clap::Parser;

use pixelflow_ir::{Environment, ExprData, Rooted, Term};
use pixelflow_pipeline::training::corpus::read_corpus;
use pixelflow_search::egraph::{EGraph, ENodeId, Origin};
use pixelflow_search::math::all_rules;

#[derive(Parser)]
#[command(name = "guide_scope_saturation_delta")]
#[command(
    about = "Per-application and per-iteration delta economics of standard saturation, for the Guide's incremental-accumulator question"
)]
struct Args {
    /// Directory holding `corpus_dev.bin`.
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    corpus_dir: String,

    /// Minimum number of expressions to measure (task floor: >= 200). Fails
    /// loud rather than silently under-measuring.
    #[arg(long, default_value_t = 200)]
    min_expressions: usize,

    /// Cap on the number of expressions measured (0 = no cap), first N in
    /// corpus order. Bounds the harness's own wall-clock (not part of the
    /// measurement).
    #[arg(long, default_value_t = 200)]
    limit: usize,

    /// Per-expression saturation budget: max batched rounds (each round is
    /// one `EGraph::saturate_with_limits(1, ..)` call, matching production
    /// semantics exactly).
    #[arg(long, default_value_t = 100)]
    max_iterations: usize,

    /// Per-expression e-graph size budget (canonical class count) — the same
    /// role `SaturationConfig::max_classes` plays in production, just a
    /// harness-chosen value bounding the O(graph size) snapshot walk cost.
    #[arg(long, default_value_t = 3000)]
    max_classes: usize,

    /// Write the full structured result as JSON to this path.
    #[arg(long)]
    out: Option<String>,
}

/// One rewrite application, reconstructed post-hoc from `Provenance`.
#[derive(Clone, Copy)]
struct ApplicationSample {
    rule_idx: usize,
    /// This application's 0-based position within its expression's whole
    /// run — lets us bucket "early/mid/late in saturation" without needing
    /// wall-clock or absolute node-count binning.
    order_in_run: usize,
    total_in_run: usize,
    edges_before: usize,
    nodes_before: usize,
    nodes_added: usize,
    edges_added: usize,
}

impl ApplicationSample {
    fn edge_fraction(&self) -> f64 {
        frac(self.edges_added, self.edges_before)
    }
    fn node_fraction(&self) -> f64 {
        frac(self.nodes_added, self.nodes_before)
    }
}

/// One `saturate_with_limits(1, ..)` round: how many candidates a Guide
/// scoring this round's matches would have had to evaluate, and how the
/// graph grew as a result.
#[derive(Clone, Copy)]
struct IterationSample {
    /// Sum of `match_counts` this round — every rule match found, whether or
    /// not it ended up a net change. This is "the journal machinery" the
    /// task names.
    eval_count: usize,
    /// `SaturationStats::total_unions`: distinct committed rewrite actions
    /// this round (excludes congruence-only merges discovered by the
    /// round's closing rebuild — see module doc).
    applied_actions: usize,
    /// `Provenance::union_count()` delta this round: every class merge,
    /// rule-driven and congruence, exact at this granularity.
    unions_added: usize,
    edges_before: usize,
    edges_after: usize,
    canon_before: usize,
    canon_after: usize,
}

impl IterationSample {
    fn edge_fraction(&self) -> f64 {
        frac(
            self.edges_after.saturating_sub(self.edges_before),
            self.edges_before,
        )
    }
}

/// `0.0` for a zero denominator (no evidence, not `NaN`) — same convention
/// as `guide_headroom`'s `ratio()`, for the same reason (an empty seed graph
/// has no "current size" to divide by).
fn frac(delta: usize, base: usize) -> f64 {
    if base == 0 {
        0.0
    } else {
        delta as f64 / base as f64
    }
}

/// Combined (canonical class count, total edge count) in one pass over
/// `class_ids()` — halves the walk cost vs. calling the two independently,
/// since both are O(current graph size) and share the same iteration.
fn snapshot_graph(egraph: &EGraph) -> (usize, usize) {
    let mut classes = 0usize;
    let mut edges = 0usize;
    for id in egraph.class_ids() {
        classes += 1;
        for node in egraph.nodes(id) {
            edges += node.children().len();
        }
    }
    (classes, edges)
}

fn quantile(mut xs: Vec<f64>, p: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("guide_scope_saturation_delta: NaN in sample")
    });
    let n = xs.len();
    if n == 1 {
        return xs[0];
    }
    let pos = p * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        xs[lo]
    } else {
        let frac = pos - lo as f64;
        xs[lo] * (1.0 - frac) + xs[hi] * frac
    }
}

/// Drive one expression's e-graph through standard (batched, production)
/// saturation one round at a time, recording per-iteration samples. Stops
/// at the first of: quiescence (`stats.total_unions == 0` for a round that
/// actually ran — a diagnostic condition, not a certified fixpoint; this
/// optimizer is budget-only by design), the `max_iterations` round cap, or
/// the `max_classes` class cap.
///
/// The class cap needs checking twice, at two different granularities, and
/// conflating them mislabels a truncated run as converged: this function's
/// own pre-round guard (`canon > max_classes`, `canon` from `snapshot_graph`)
/// counts *canonical* (live-root) classes, while `saturate_with_limits`'s
/// own internal check (`self.classes.len() > max_classes`) counts every
/// *allocated* class slot, including ones already unioned away and excluded
/// from canonical iteration. Allocated count is always >= canonical count,
/// so `saturate_with_limits` can hit its cap and return `stats.iterations ==
/// 0, total_unions == 0` (did zero work this round) even when this
/// function's own pre-check just passed. `stats.total_unions == 0` alone
/// cannot tell that apart from "the round ran and genuinely found nothing to
/// union" — only `stats.iterations == 0` can, since `iterations` increments
/// only after both of `saturate_with_limits`'s own cap checks pass.
fn run_saturation_instrumented(
    egraph: &mut EGraph,
    max_iterations: usize,
    max_classes: usize,
) -> (Vec<IterationSample>, bool) {
    let mut iters: Vec<IterationSample> = Vec::new();
    let mut reached_quiescence = false;
    // Generous backstop, never the intended stopping condition — this
    // harness times nothing and reports nothing time-based. Matches the
    // same convention `guide_headroom` documents for `EGraph::saturate()`'s
    // internal deadline.
    let big_timeout = std::time::Duration::from_secs(3600);

    let (mut canon, mut edges) = snapshot_graph(egraph);

    for _round in 0..max_iterations {
        if canon > max_classes {
            break;
        }
        egraph.match_counts.clear();
        let unions_before_round = egraph.provenance().union_count();

        let stats = egraph.saturate_with_limits(1, max_classes, big_timeout);

        // `stats.iterations == 0` means `saturate_with_limits`'s own
        // allocated-class-count cap tripped before this round could run at
        // all (see this function's doc comment) -- budget-exhausted, same
        // disposition as the canonical-count pre-check above, NOT
        // quiescence. Nothing happened this round, so no `IterationSample`
        // is recorded for it either (a zero-change entry would only dilute
        // the per-iteration medians with a round that never actually ran).
        if stats.iterations == 0 {
            break;
        }

        let eval_count: usize = egraph.match_counts.values().sum();

        let (new_canon, new_edges) = snapshot_graph(egraph);
        let unions_after_round = egraph.provenance().union_count();

        iters.push(IterationSample {
            eval_count,
            applied_actions: stats.total_unions,
            unions_added: unions_after_round - unions_before_round,
            edges_before: edges,
            edges_after: new_edges,
            canon_before: canon,
            canon_after: new_canon,
        });

        canon = new_canon;
        edges = new_edges;

        if stats.total_unions == 0 {
            reached_quiescence = true;
            break;
        }
    }

    (iters, reached_quiescence)
}

/// Reconstruct per-application node/edge deltas from `Provenance`, post-hoc,
/// per the module doc's three-step method. `seed_nodes`/`seed_edges` are the
/// graph size right after `add_arena`, before any rewriting — the replay's
/// starting point. Returns the per-application samples plus the count of
/// origin-recorded nodes that were later pruned from the final canonical
/// structure and so excluded from every delta (see step 2's comment).
fn reconstruct_applications(
    egraph: &EGraph,
    seed_nodes: usize,
    seed_edges: usize,
) -> (Vec<ApplicationSample>, usize) {
    // Step 1: ENodeId -> arity, one O(final graph size) walk.
    let mut arity_of: HashMap<ENodeId, usize> = HashMap::new();
    for id in egraph.class_ids() {
        let tags = egraph.tags(id);
        let nodes = egraph.nodes(id);
        debug_assert_eq!(
            tags.len(),
            nodes.len(),
            "guide_scope_saturation_delta: tags/nodes length mismatch — provenance invariant violated"
        );
        for (tag, node) in tags.iter().zip(nodes.iter()) {
            arity_of.insert(*tag, node.children().len());
        }
    }

    // Step 2: ApplicationId -> the ENodeIds it created, one pass over
    // `origins()` -- filtered to nodes that actually survive into the final
    // canonical structure (`arity_of`, step 1). `rebuild_budgeted` can
    // permanently drop a node from its class's node/tag vectors: when
    // canonicalizing finds a memo collision with an existing, *different*
    // class whose constant fact conflicts (`graph.rs`'s `union` refusal for
    // provably-unequal constants -- an ill-conditioned-kernel detector, not
    // routine congruence closure), the colliding node is discarded rather
    // than merged (`if self.find(id) != self.find(existing) { continue; }`).
    // `Provenance::origins()` is an append-only log and still names that
    // node's creating application forever, but the node is gone from every
    // canonical class by the time this function's `arity_of` walk runs. A
    // origin whose tag has no `arity_of` entry is exactly this case: it was
    // minted, then pruned before this post-hoc reconstruction ran. Excluding
    // it from BOTH `nodes_added` and `edges_added` (rather than counting it
    // toward nodes_added while defaulting its arity to 0, which would make
    // the two disagree about the same application) keeps this reconstruction
    // consistent with "graph size in the final, live structure" -- the
    // object an incremental accumulator actually maintains. `dropped_origins`
    // reports how often this fires so the harness's own output states the
    // approximation's scope instead of leaving it silent.
    let mut nodes_by_app: HashMap<u64, Vec<ENodeId>> = HashMap::new();
    let mut dropped_origins = 0usize;
    for (enode_id, origin) in egraph.provenance().origins() {
        if let Origin::Rule(app_id) = origin {
            if arity_of.contains_key(&enode_id) {
                nodes_by_app
                    .entry(app_id.as_u64())
                    .or_default()
                    .push(enode_id);
            } else {
                dropped_origins += 1;
            }
        }
    }

    // Step 3: replay `applications()` in firing order, accumulating.
    let mut running_nodes = seed_nodes;
    let mut running_edges = seed_edges;
    let mut apps: Vec<ApplicationSample> = Vec::new();
    let total_applications = egraph.provenance().recorded_count();

    for (app_id, record) in egraph.provenance().applications() {
        let created = nodes_by_app.get(&app_id.as_u64());
        let nodes_added = created.map(Vec::len).unwrap_or(0);
        let edges_added: usize = created
            .map(|ids| {
                ids.iter()
                    .map(|t| {
                        arity_of.get(t).copied().unwrap_or_else(|| {
                            unreachable!(
                                "guide_scope_saturation_delta: tag survived the \
                                 arity_of filter above but is missing from arity_of \
                                 -- nodes_by_app and arity_of disagree"
                            )
                        })
                    })
                    .sum()
            })
            .unwrap_or(0);

        apps.push(ApplicationSample {
            rule_idx: record.rule_idx,
            order_in_run: apps.len(),
            total_in_run: total_applications,
            edges_before: running_edges,
            nodes_before: running_nodes,
            nodes_added,
            edges_added,
        });

        running_nodes += nodes_added;
        running_edges += edges_added;
    }

    (apps, dropped_origins)
}

/// Bucket an application into thirds of its own expression's run, by firing
/// order — the "does the delta fraction shrink as the graph grows" question,
/// answered without needing absolute size binning (which would conflate
/// expression size with saturation progress).
fn run_third(order_in_run: usize, total_in_run: usize) -> usize {
    if total_in_run <= 1 {
        return 0;
    }
    let pos = order_in_run as f64 / (total_in_run - 1).max(1) as f64;
    if pos < 1.0 / 3.0 {
        0
    } else if pos < 2.0 / 3.0 {
        1
    } else {
        2
    }
}

fn main() {
    let args = Args::parse();

    let corpus_dir = PathBuf::from(&args.corpus_dir);
    let dev_path = corpus_dir.join("corpus_dev.bin");
    let mut entries: Vec<(String, Rooted<ExprData>)> = read_corpus(&dev_path).unwrap_or_else(|e| {
        panic!(
            "guide_scope_saturation_delta: failed to read {}: {e}",
            dev_path.display()
        )
    });
    let total_available = entries.len();

    assert!(
        total_available >= args.min_expressions,
        "guide_scope_saturation_delta: corpus_dev.bin has only {total_available} \
         expressions, need >= {} — regenerate a larger corpus, do not silently \
         measure on too few expressions",
        args.min_expressions
    );

    if args.limit > 0 {
        entries.truncate(args.limit);
    }
    let n = entries.len();
    eprintln!(
        "guide_scope_saturation_delta: measuring {n} expressions (of {total_available} \
         available in corpus_dev.bin), max_iterations={}, max_classes={}",
        args.max_iterations, args.max_classes
    );

    let rule_names: Vec<String> = {
        let probe = EGraph::with_rules(all_rules());
        (0..all_rules().len())
            .map(|i| {
                probe
                    .rule(i)
                    .map(|r| r.name().to_string())
                    .unwrap_or_else(|| format!("<rule {i}>"))
            })
            .collect()
    };

    let mut all_apps: Vec<ApplicationSample> = Vec::new();
    let mut all_iters: Vec<IterationSample> = Vec::new();
    let mut per_rule_fired: BTreeMap<usize, usize> = BTreeMap::new();
    let mut truncated_by_budget = 0usize;
    // Diagnostics for `reconstruct_applications`'s dropped-node filter (see
    // its doc comment): `total_dropped_origins` counts nodes excluded from
    // every delta because a later rebuild pruned them from the final
    // canonical structure; `total_refused_const_unions` is the underlying
    // cause (`EGraph::union`'s conflicting-constant refusal) reported
    // per-expression via the graph's own durable record, to distinguish
    // "the filter fired because of genuine ill-conditioned collisions" from
    // "the filter fired for some other reason" -- if these ever diverge,
    // that is itself a finding.
    let mut total_dropped_origins = 0usize;
    let mut total_refused_const_unions = 0usize;

    // A corpus entry declares no buffers or uniforms — the format refuses to
    // write one down — so one empty environment serves every term.
    let env = Environment::new();
    for (i, (_name, expr)) in entries.iter().enumerate() {
        let term = Term::new(expr.entry(), &env);
        let mut egraph = EGraph::with_rules(all_rules());
        let root_class = pixelflow_search::egraph::insert_term(
            term,
            &mut egraph,
            pixelflow_search::egraph::Vocabulary::Templates,
        )
        .expect("insert into e-graph");
        let _ = root_class; // root only needed to seed the graph; saturation is whole-graph.

        let (seed_canon, seed_edges) = snapshot_graph(&egraph);
        let seed_nodes = egraph.provenance().origin_count();
        let _ = seed_canon;

        let (iters, reached_quiescence) =
            run_saturation_instrumented(&mut egraph, args.max_iterations, args.max_classes);
        if !reached_quiescence {
            truncated_by_budget += 1;
        }

        total_refused_const_unions += egraph.refused_const_unions().len();
        let (apps, dropped_origins) = reconstruct_applications(&egraph, seed_nodes, seed_edges);
        total_dropped_origins += dropped_origins;
        for a in &apps {
            *per_rule_fired.entry(a.rule_idx).or_insert(0) += 1;
        }

        all_apps.extend(apps);
        all_iters.extend(iters);

        if (i + 1) % 50 == 0 || i + 1 == n {
            eprintln!(
                "guide_scope_saturation_delta: {}/{n} expressions measured ({} applications, {} iterations so far)",
                i + 1,
                all_apps.len(),
                all_iters.len()
            );
        }
    }

    // ---- (1) Per-application delta fraction: median/p90, and the
    // early/mid/late-in-run trend. ----
    let edge_fracs: Vec<f64> = all_apps.iter().map(|a| a.edge_fraction()).collect();
    let node_fracs: Vec<f64> = all_apps.iter().map(|a| a.node_fraction()).collect();
    let edge_frac_median = quantile(edge_fracs.clone(), 0.5);
    let edge_frac_p90 = quantile(edge_fracs.clone(), 0.9);
    let node_frac_median = quantile(node_fracs.clone(), 0.5);
    let node_frac_p90 = quantile(node_fracs.clone(), 0.9);

    let mut third_edge_fracs: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut third_nonzero_edge_fracs: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for a in &all_apps {
        let t = run_third(a.order_in_run, a.total_in_run);
        third_edge_fracs[t].push(a.edge_fraction());
        if a.nodes_added > 0 {
            third_nonzero_edge_fracs[t].push(a.edge_fraction());
        }
    }
    let third_medians: [f64; 3] =
        std::array::from_fn(|i| quantile(third_edge_fracs[i].clone(), 0.5));
    // Same early/mid/late split, but conditioned on state-changing
    // applications only — the pooled `third_medians` above are all 0.0
    // because the 91% zero-delta share (see below) floors every third at
    // once, which would otherwise make the "shrinks as the graph grows"
    // trend unobservable.
    let third_nonzero_medians: [f64; 3] =
        std::array::from_fn(|i| quantile(third_nonzero_edge_fracs[i].clone(), 0.5));

    let mut nodes_added_hist: BTreeMap<usize, usize> = BTreeMap::new();
    for a in &all_apps {
        *nodes_added_hist.entry(a.nodes_added).or_insert(0) += 1;
    }

    // Under batched semantics `all_apps` includes every recorded
    // `Provenance` application, not just the ones that created new state —
    // an idempotent re-fire (e.g. `commutative` re-matching a form it
    // already installed, see the sibling report's finding) is still a
    // recorded application with `nodes_added == 0`, and it dominates the
    // pooled median (see module doc / results write-up). Reporting the
    // zero-delta share explicitly, and the edge-fraction distribution
    // conditioned on `nodes_added >= 1`, keeps the headline "median is
    // literally 0" honest instead of silently hiding what it means, and
    // gives a number directly comparable to the superseded single-step
    // report's 0.31%-median headline (computed only over state-changing
    // applications).
    let zero_delta_count = all_apps.iter().filter(|a| a.nodes_added == 0).count();
    let zero_delta_fraction = frac(zero_delta_count, all_apps.len());
    let nonzero_edge_fracs: Vec<f64> = all_apps
        .iter()
        .filter(|a| a.nodes_added > 0)
        .map(|a| a.edge_fraction())
        .collect();
    let nonzero_edge_frac_median = quantile(nonzero_edge_fracs.clone(), 0.5);
    let nonzero_edge_frac_p90 = quantile(nonzero_edge_fracs, 0.9);

    // Per-ITERATION (batch of applications) edge growth fraction — the
    // task's explicit "also per iteration" ask.
    let iter_edge_fracs: Vec<f64> = all_iters.iter().map(|it| it.edge_fraction()).collect();
    let iter_edge_frac_median = quantile(iter_edge_fracs.clone(), 0.5);
    let iter_edge_frac_p90 = quantile(iter_edge_fracs.clone(), 0.9);
    let iter_canon_fracs: Vec<f64> = all_iters
        .iter()
        .map(|it| {
            let delta = it.canon_after as i64 - it.canon_before as i64;
            if it.canon_before == 0 {
                0.0
            } else {
                delta as f64 / it.canon_before as f64
            }
        })
        .collect();
    let iter_canon_frac_median = quantile(iter_canon_fracs, 0.5);

    // Unions: per-iteration exact (both "committed actions" and "all
    // merges including congruence"), since per-application isn't
    // attributable under batched semantics (see module doc).
    let applied_actions: Vec<f64> = all_iters
        .iter()
        .map(|it| it.applied_actions as f64)
        .collect();
    let unions_added: Vec<f64> = all_iters.iter().map(|it| it.unions_added as f64).collect();
    let applied_actions_median = quantile(applied_actions, 0.5);
    let unions_added_median = quantile(unions_added.clone(), 0.5);
    let unions_added_p90 = quantile(unions_added, 0.9);
    let total_congruence_only: u64 = all_iters
        .iter()
        .map(|it| (it.unions_added.saturating_sub(it.applied_actions)) as u64)
        .sum();
    let total_unions_added: u64 = all_iters.iter().map(|it| it.unions_added as u64).sum();

    // ---- (2) Cumulative-vs-incremental work ratio. ----
    let sum_edges_before: u64 = all_apps.iter().map(|a| a.edges_before as u64).sum();
    let sum_edges_added: u64 = all_apps.iter().map(|a| a.edges_added as u64).sum();
    let cumulative_vs_incremental_ratio = if sum_edges_added > 0 {
        sum_edges_before as f64 / sum_edges_added as f64
    } else {
        0.0
    };
    // Derived from `nonzero_edge_frac_median` (state-changing applications
    // only), not the unconditioned `edge_frac_median` -- the latter is
    // exactly 0.0 (91.1% of applications are zero-delta idempotent re-fires,
    // see `zero_delta_fraction` above), so `1.0 / edge_frac_median` would be
    // either a nonsensical 0.0x (guarded away by the `else` branch below) or,
    // read naively, "no speedup" for a population where the real finding is
    // the opposite: most applications need no incremental work at all, and
    // the ones that do are ~731x cheaper than a rebuild. Reported inconsistently
    // with the design doc's headline number before this fix.
    let median_implied_speedup = if nonzero_edge_frac_median > 0.0 {
        1.0 / nonzero_edge_frac_median
    } else {
        0.0
    };

    // ---- (3) Eval-count economics. ----
    let eval_counts: Vec<f64> = all_iters.iter().map(|it| it.eval_count as f64).collect();
    let applied_counts: Vec<f64> = all_iters
        .iter()
        .map(|it| it.applied_actions as f64)
        .collect();
    let eval_median = quantile(eval_counts.clone(), 0.5);
    let eval_p90 = quantile(eval_counts.clone(), 0.9);
    let applied_median = quantile(applied_counts.clone(), 0.5);
    let total_evals: u64 = all_iters.iter().map(|it| it.eval_count as u64).sum();
    let total_applied: u64 = all_iters.iter().map(|it| it.applied_actions as u64).sum();
    let evals_per_applied = if total_applied > 0 {
        total_evals as f64 / total_applied as f64
    } else {
        0.0
    };

    println!("=== Guide scope: saturation delta economics ({n} expressions) ===");
    println!(
        "total applications: {}  total iterations: {}  (expressions that hit a budget cap before reaching quiescence: {}/{n})",
        all_apps.len(),
        all_iters.len(),
        truncated_by_budget
    );
    println!(
        "refused-constant-union events (EGraph::union's ill-conditioned-kernel detector): {total_refused_const_unions}  \
         -> origins excluded from every delta because their node was later pruned from the final canonical structure: \
         {total_dropped_origins}"
    );
    println!();
    println!("--- (1) Per-application delta fraction (vs. graph size immediately before) ---");
    println!("  edges:  median {edge_frac_median:.4}  p90 {edge_frac_p90:.4}");
    println!("  nodes:  median {node_frac_median:.4}  p90 {node_frac_p90:.4}");
    println!(
        "  zero-delta applications (idempotent re-fires, nodes_added == 0): {zero_delta_count}/{} ({:.1}%)",
        all_apps.len(),
        zero_delta_fraction * 100.0
    );
    println!(
        "  edges, conditioned on nodes_added >= 1 (state-changing applications only): median {nonzero_edge_frac_median:.4}  p90 {nonzero_edge_frac_p90:.4}"
    );
    println!(
        "  by position in run (edge-fraction median): early-third {:.4}  mid-third {:.4}  late-third {:.4}  [{}]",
        third_medians[0],
        third_medians[1],
        third_medians[2],
        if third_medians[2] <= third_medians[0] {
            "CONFIRMS shrinking-as-graph-grows"
        } else {
            "REFUTES shrinking-as-graph-grows"
        }
    );
    println!(
        "  same trend, conditioned on nodes_added >= 1: early-third {:.4}  mid-third {:.4}  late-third {:.4}  [{}]",
        third_nonzero_medians[0],
        third_nonzero_medians[1],
        third_nonzero_medians[2],
        if third_nonzero_medians[2] <= third_nonzero_medians[0] {
            "CONFIRMS shrinking-as-graph-grows"
        } else {
            "REFUTES shrinking-as-graph-grows"
        }
    );
    println!(
        "  nodes_added-per-application distribution: {:?}",
        nodes_added_hist
            .iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect::<Vec<_>>()
    );
    println!();
    println!("--- (1b) Per-ITERATION (batch of applications) growth fraction ---");
    println!(
        "  edges: median {iter_edge_frac_median:.4}  p90 {iter_edge_frac_p90:.4}  (n={} iterations)",
        all_iters.len()
    );
    println!("  canonical classes: median {iter_canon_frac_median:.4}");
    println!(
        "  applied actions (SaturationStats::total_unions) per iteration: median {applied_actions_median:.1}"
    );
    println!(
        "  unions_added per iteration (all merges incl. congruence, Provenance::union_count() delta): median {unions_added_median:.1}  p90 {unions_added_p90:.1}"
    );
    println!(
        "  congruence-only merges (unions_added - applied_actions), summed: {total_congruence_only}/{total_unions_added} ({:.1}% of all merges)",
        frac(total_congruence_only as usize, total_unions_added as usize) * 100.0
    );
    println!();
    println!("--- (2) Cumulative-vs-incremental work ratio ---");
    println!("  sum(edges_before) [cumulative, rebuild-from-scratch cost] = {sum_edges_before}");
    println!("  sum(edges_added)  [incremental, O(delta) cost]        = {sum_edges_added}");
    println!(
        "  cumulative/incremental ratio (pooled, work-weighted): {cumulative_vs_incremental_ratio:.2}x"
    );
    println!(
        "  median-edge-fraction-implied speedup (extraction-study-style, 1/nonzero-median): {median_implied_speedup:.2}x"
    );
    println!();
    println!(
        "--- (3) Eval-count economics (candidates scored per iteration, from match_counts) ---"
    );
    println!("  eval_count (matches found per round): median {eval_median:.1}  p90 {eval_p90:.1}");
    println!("  applied_actions per round: median {applied_median:.1}");
    println!("  totals: {total_evals} evals -> {total_applied} committed actions");
    println!("  evals scored per action actually committed: {evals_per_applied:.2}x");
    println!();
    println!("{:<28} {:>10}", "rule", "applications");
    let mut rows: Vec<(usize, usize)> = per_rule_fired.iter().map(|(&k, &v)| (k, v)).collect();
    rows.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
    for (idx, count) in rows.iter().take(20) {
        let name = rule_names
            .get(*idx)
            .cloned()
            .unwrap_or_else(|| format!("<rule {idx}>"));
        println!("{name:<28} {count:>10}");
    }

    if let Some(out_path) = &args.out {
        let mut json = String::new();
        json.push_str("{\n");
        json.push_str(&format!("  \"num_expressions\": {n},\n"));
        json.push_str(&format!("  \"total_applications\": {},\n", all_apps.len()));
        json.push_str(&format!("  \"total_iterations\": {},\n", all_iters.len()));
        json.push_str(&format!(
            "  \"expressions_truncated_by_budget\": {truncated_by_budget},\n"
        ));
        json.push_str(&format!(
            "  \"refused_const_union_events\": {total_refused_const_unions},\n"
        ));
        json.push_str(&format!(
            "  \"dropped_origins_excluded_from_deltas\": {total_dropped_origins},\n"
        ));
        json.push_str("  \"per_application_delta_fraction\": {\n");
        json.push_str(&format!("    \"edge_median\": {edge_frac_median:.6},\n"));
        json.push_str(&format!("    \"edge_p90\": {edge_frac_p90:.6},\n"));
        json.push_str(&format!("    \"node_median\": {node_frac_median:.6},\n"));
        json.push_str(&format!("    \"node_p90\": {node_frac_p90:.6},\n"));
        json.push_str(&format!(
            "    \"edge_median_early_third\": {:.6},\n",
            third_medians[0]
        ));
        json.push_str(&format!(
            "    \"edge_median_mid_third\": {:.6},\n",
            third_medians[1]
        ));
        json.push_str(&format!(
            "    \"edge_median_late_third\": {:.6},\n",
            third_medians[2]
        ));
        json.push_str(&format!("    \"zero_delta_count\": {zero_delta_count},\n"));
        json.push_str(&format!(
            "    \"zero_delta_fraction\": {zero_delta_fraction:.6},\n"
        ));
        json.push_str(&format!(
            "    \"nonzero_edge_median\": {nonzero_edge_frac_median:.6},\n"
        ));
        json.push_str(&format!(
            "    \"nonzero_edge_p90\": {nonzero_edge_frac_p90:.6},\n"
        ));
        json.push_str(&format!(
            "    \"nonzero_edge_median_early_third\": {:.6},\n",
            third_nonzero_medians[0]
        ));
        json.push_str(&format!(
            "    \"nonzero_edge_median_mid_third\": {:.6},\n",
            third_nonzero_medians[1]
        ));
        json.push_str(&format!(
            "    \"nonzero_edge_median_late_third\": {:.6}\n",
            third_nonzero_medians[2]
        ));
        json.push_str("  },\n");
        json.push_str("  \"per_iteration\": {\n");
        json.push_str(&format!(
            "    \"edge_growth_fraction_median\": {iter_edge_frac_median:.6},\n"
        ));
        json.push_str(&format!(
            "    \"edge_growth_fraction_p90\": {iter_edge_frac_p90:.6},\n"
        ));
        json.push_str(&format!(
            "    \"canon_class_growth_fraction_median\": {iter_canon_frac_median:.6},\n"
        ));
        json.push_str(&format!(
            "    \"applied_actions_median\": {applied_actions_median:.6},\n"
        ));
        json.push_str(&format!(
            "    \"unions_added_median\": {unions_added_median:.6},\n"
        ));
        json.push_str(&format!(
            "    \"unions_added_p90\": {unions_added_p90:.6},\n"
        ));
        json.push_str(&format!(
            "    \"congruence_only_merges_total\": {total_congruence_only},\n"
        ));
        json.push_str(&format!("    \"all_merges_total\": {total_unions_added}\n"));
        json.push_str("  },\n");
        json.push_str("  \"cumulative_vs_incremental\": {\n");
        json.push_str(&format!("    \"sum_edges_before\": {sum_edges_before},\n"));
        json.push_str(&format!("    \"sum_edges_added\": {sum_edges_added},\n"));
        json.push_str(&format!(
            "    \"pooled_ratio\": {cumulative_vs_incremental_ratio:.6},\n"
        ));
        json.push_str(&format!(
            "    \"nonzero_median_fraction_implied_speedup\": {median_implied_speedup:.6}\n"
        ));
        json.push_str("  },\n");
        json.push_str("  \"eval_count_economics\": {\n");
        json.push_str(&format!("    \"eval_count_median\": {eval_median:.6},\n"));
        json.push_str(&format!("    \"eval_count_p90\": {eval_p90:.6},\n"));
        json.push_str(&format!(
            "    \"applied_actions_median\": {applied_median:.6},\n"
        ));
        json.push_str(&format!("    \"total_evals\": {total_evals},\n"));
        json.push_str(&format!("    \"total_applied\": {total_applied},\n"));
        json.push_str(&format!(
            "    \"evals_per_applied\": {evals_per_applied:.6}\n"
        ));
        json.push_str("  },\n");
        json.push_str("  \"per_rule_applications\": [\n");
        for (i, (idx, count)) in rows.iter().enumerate() {
            let name = rule_names
                .get(*idx)
                .cloned()
                .unwrap_or_else(|| format!("<rule {idx}>"));
            json.push_str(&format!(
                "    {{\"rule\": \"{name}\", \"rule_idx\": {idx}, \"applications\": {count}}}{}\n",
                if i + 1 < rows.len() { "," } else { "" }
            ));
        }
        json.push_str("  ]\n");
        json.push_str("}\n");

        let mut f = std::fs::File::create(out_path).unwrap_or_else(|e| {
            panic!("guide_scope_saturation_delta: cannot create {out_path}: {e}")
        });
        f.write_all(json.as_bytes()).unwrap_or_else(|e| {
            panic!("guide_scope_saturation_delta: cannot write {out_path}: {e}")
        });
        eprintln!("guide_scope_saturation_delta: wrote {out_path}");
    }
}
