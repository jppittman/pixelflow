# The e-graph engine, profiled: where the cycles and bytes go

**Audience:** whoever picks up e-graph engine work next. **Status:** measured facts and
ranked, cited recommendations, not a plan — none of these changes are made here.

## 0. The ask in one paragraph

The optimizer is one e-graph (`pixelflow-search/src/egraph/`), and every kernel this
compiler ships goes through it, so its own CPU and memory cost is a tax paid on every
compile. This document is a profile of the *engine* — insertion, saturation, extraction,
the data structures underneath — not of what it extracts. That is a different, already-
documented question: `docs/plans/2026-09-06-egraph-at-production-scale.md` is about
*extraction quality* (does the chosen DAG run fast) and explicitly separates "engineering"
from "research" in its §4; this document is the engineering half's sequel, for the cost of
*computing* the extraction and the saturation that precedes it, independent of what they
choose. Headline: **roughly a third of all CPU time in a representative saturate+extract
call is spent in the allocator**, and a second architecturally-quadratic cost
(`shared_dag_dp_pass`) already dominates wall clock once the e-graph is near its class
cap — which is exactly the regime production's largest kernel (the chrome scene, S3b)
runs in today. Both are addressable without touching extraction policy or the cost model.

## 1. Methodology

Two measurement tools, both reproducible from `main`:

- **`pixelflow-pipeline/src/bin/egraph_profile.rs`** (new, this change): builds synthetic
  arenas at six sizes (16 to 16,384 nodes, same SDF-shaped op mix as
  `bench_jit_compile_cost::build_kernel_arena` — Sub/Mul/MulAdd/Select/Sqrt/Max, so numbers
  are comparable to the existing G0 JIT-cost gate) and calls
  `pixelflow_search::runtime::optimize_runtime_arena` — the same saturate-then-extract
  entry point `Lattice::bake` uses — under a counting global allocator (bytes
  allocated/deallocated/peak) and, with `--features profiling`, a `pprof` CPU sampler.
  Every call uses a fresh salt so `optimize_runtime_arena`'s structural cache never hits.
  Run: `cargo run --release -p pixelflow-pipeline --features "training profiling" --bin egraph_profile`.
- **`--features pixelflow-search/saturation-telemetry`**: one JSONL record per saturation
  call (tier, stop reason, classes/applications/iterations at stop, wall clock) —
  corroborates the wall-clock numbers below independently of the harness's own `Instant`
  calls, and explains *why* each size behaves as it does.

Both were run together (`cargo build --release -p pixelflow-pipeline --features "training
profiling" --features pixelflow-search/saturation-telemetry --bin egraph_profile`) so the
scaling table and the stop-reason attribution are the same runs.

## 2. Headline numbers

| nodes | median wall | bytes/call | peak bytes | stop reason | iterations | classes at stop | applications |
|---:|---:|---:|---:|---|---:|---:|---:|
| 16 | 4.4 ms | 4.1 MB | 143 KB | quiesced (fixed point) | 5 | 297 | 1,528 |
| 64 | 16.2 ms | 17.0 MB | 562 KB | quiesced | 5 | 1,185 | 6,433 |
| 256 | 33.9 ms | 33.3 MB | 2.17 MB | class_cap | 3 | 4,212 | 8,537 |
| 1,024 | 22.8 ms | 20.6 MB | 2.59 MB | class_cap | 1 | 4,887 | 3,296 |
| 4,096 | 39.1 ms | 28.8 MB | 4.05 MB | class_cap | 1 | 5,000 | 903 |
| 16,384 | 206 ms | 62.8 MB | 39.4 MB | class_cap | **0** | 16,384 | **0** |

Reading it:

- **Every one of these calls allocates 100–1,000× the bytes it ends up holding live.**
  A 16-node input that quiesces to 297 classes allocates 4.1 MB to get there — ~14 KB per
  live class. That ratio is the transient-allocation story in §4.1–4.2, not extraction
  policy: the graph is small and correct, it just paid for a great deal of scratch work
  to saturate.
- **The two rows that matter for production are opposite ends of this table.** The 193
  small glyph kernels (shape B, `docs/plans/2026-09-06-...md` §1) live in the "quiesced,
  few iterations, small class count" regime (rows 1–2); the chrome scene and other large
  scene kernels (shape A) live in the "class_cap, 1 or 0 iterations, thousands of classes"
  regime (rows 3–6) — the production doc's own number for the chrome scene, "4,913/5,000
  classes, stop after 2 iterations," sits right on this table between rows 3 and 4.
- **Row 6 is a distinct case worth naming.** At 16,384 raw (mostly non-duplicate) nodes,
  hash-consing alone exceeds the 5,000-class cap before a single rewrite iteration runs
  (`iterations: 0, applications: 0`) — the entire 206 ms and 39 MB peak is insertion plus
  one extraction over a maximal e-graph, with *zero* saturation. This isolates extraction's
  own cost from saturation's, which is what §4.1 uses it for.

## 3. CPU flamegraph: where the cycles go

`egraph_profile --features profiling` ran 2,000 distinct-salt 4,096-node calls under
`pprof` (997 Hz), each hitting the 5,000-class cap after one iteration (matching row 5
above — same regime as the chrome scene). Self-time by function, 16,802 samples:

| self-time share | function |
|---:|---|
| 32.8% | `egraph::extract::extract_dag_scoped` (its own DP loops) |
| 23.5% | `__rust_alloc` |
| 9.8% | `__rust_dealloc` |
| 5.4% | `Vec<T>::clone` |
| 4.0% | `egraph::graph::EGraph::apply_rule_at_index_timed` |
| 2.8% | `BTreeMap::insert` |
| 2.6% | `egraph::extract::repair_choices_well_founded` |
| 2.5% | `egraph::extract::post_order` |
| 1.2% | `DefaultHasher::write` |
| 0.8% | `ENode::hash` |

**Allocator functions alone are 33.3% of all sampled CPU time**, and `Vec::clone` is
another 5.4% — over a third of every saturate+extract call's cycles are spent moving
bytes into and out of the heap rather than doing arithmetic on the graph. `post_order`
and the `insert`/`assemble` call sites are where the `BTreeMap`/`BTreeSet` cost
concentrates (§4.5).

## 4. Findings, ranked by measured impact

### 4.1 HIGH — `shared_dag_dp_pass` is O(live_classes²) in both time and memory, run unconditionally on every extraction

`extract_dag_scoped` (`extract.rs:1734–1765`) always runs **two** full DP passes over the
live class set and keeps the cheaper by true `dag_cost`: `tree_dp_pass` (linear) and
`shared_dag_dp_pass` (`extract.rs:1970–2097`), whose own doc comment (`extract.rs:1966–
1969`) states the cost plainly: "space is one bit per REACHABLE class per reachable
class." No threshold guards it — both passes run at every extraction regardless of live
class count (confirmed by grep: `shared_dag_dp_pass` has exactly one call site per branch
in `extract_dag_scoped`/`extract_dag`, no size check before it).

This table's peak-byte column *is* that quadratic term, measured three independent ways:

| live classes (L) | predicted L²/8 bytes | measured peak bytes |
|---:|---:|---:|
| 4,212 | 2.22 MB | 2.17 MB |
| 5,000 | 3.13 MB | 4.05 MB |
| 16,384 | 33.55 MB | 39.36 MB |

The match (within the noise of the DP's other, linear-sized vectors — `best_cost`,
`best_node`, `best_var`, all `O(num_classes)`) is close enough to confirm the bitset
dominates peak memory once L exceeds a few thousand, and the CPU flamegraph agrees:
`extract_dag_scoped`'s own frames are 32.8% of all sampled cycles at L=5,000 — the single
largest function in the whole profile, ahead of saturation itself.

This also gives a mechanistic explanation for a previously-unexplained number in
`docs/plans/2026-09-06-egraph-at-production-scale.md` §3: raising the production class cap
5,000 → 60,000 made the chrome scene's compile **30× slower**. `(60,000/5,000)² = 144×` —
not an exact match (fewer of those 60,000 classes are *reachable* from the root than the
cap itself, and other costs are near-linear), but the same order of magnitude, and it
identifies *which* pass absorbs a raised cap. Anyone revisiting that experiment — the
Guide programme (§5.2 there) explicitly wants to — is reopening this cost, not a fresh
unknown.

**Fix directions**, in order of how much they preserve the pass's no-regression guarantee
(`extract.rs:1723–1725`: "the returned DAG cost can only be lower... never higher"):
- Gate `shared_dag_dp_pass` behind a live-class threshold (e.g. skip it above some L,
  falling back to the tree pass alone) — cheapest to write, weakens the guarantee only
  above the threshold, and needs a number chosen from real corpora (glyph medians run
  ~1,755 live classes per the doc comment; that's clearly worth the second pass, 5,000+
  much less clearly).
- Size the bitset by the same `words = live.div_ceil(BITS)` it already computes, but
  bound *reach-set growth itself* — e.g. abandon sharing-tracking for a class once its
  reach set exceeds a cap, falling back to that class's tree cost — rather than an
  all-or-nothing gate on the whole pass.
- At minimum, whoever next raises `max_classes` for research purposes should re-run this
  document's §2 table at the new cap first, so the CPU/memory cost is a known number
  before the experiment, not a surprise after.

### 4.2 HIGH — saturation rescans every class with every rule every iteration; no dirty tracking

`saturate_bounded` (`graph.rs:1182–1318`) loops `for rule_idx in 0..n_rules` (62 rules,
`graph.rs:1243`) every iteration. Each call goes to `apply_rule_at_index_timed`
(`graph.rs:1349`), which calls `self.canonical_class_ids()` — a fresh `O(classes)` `Vec`
allocation (`graph.rs:1404`) — and then, for **every** class in that list, clones the
class's entire node vector (`let nodes: Vec<ENode> = self.classes[canonical.index()]
.nodes.clone();`, `graph.rs:1420`) before matching. There is no tracking of which classes
changed since a given rule last scanned them — `worklist` (`graph.rs:676`) drives only
congruence-closure rebuild, not match invalidation — so a class untouched since the last
round is still fully rescanned, cloned, and matched against all 62 rules next round. This
is the classic egg-style "match-all, apply-all" loop, missing the semi-naive/incremental
matching refinement (a per-rule "last-scanned generation" compared against a per-class
"last-changed generation," so a rule only rescans classes dirtied since its own last pass)
that discrimination-tree e-graph implementations use to avoid exactly this.

This is directly visible in the numbers: the 16-node case quiesces at 297 classes in 5
iterations (§2) — up to `5 × 62 × 297 ≈ 92,000` class×rule scans, most of them empty and
none of them memoized between rounds — and spends 4.4 ms and 4.1 MB doing it, a >1,000×
allocation-to-live-graph ratio (§2). `apply_rule_at_index_timed` is 4.0% of self-time in
§3's flamegraph, but its `nodes.clone()` call is a direct contributor to the 5.4%
`Vec::clone` and part of the 33.3% allocator line — this function is the dominant source
of both.

**Fix direction:** give each `EClass` a "last-touched" generation counter (bumped whenever
a node is added to it via `add`/`union`), and each rule a "last-scanned-up-to" generation.
A round only re-clones and re-matches classes whose generation exceeds the rule's own —
classes untouched since a rule's last pass are skipped outright. This is a real design
change to the matching loop (it changes what "one iteration" means, and the budget/stop-
reason accounting in `saturate_bounded` would need to keep meaning the same thing to
`SaturationStop`/telemetry consumers), not a local patch — it belongs to whoever next
works on saturation itself, sized accordingly.

### 4.3 MEDIUM — per-node child storage and read pattern: heap-allocated `Vec`, cloned on read

`ENode::Op { op, children: Vec<EClassId> }` (`node.rs:29–47`) heap-allocates one `Vec` per
op-node; op arity is 1–3 for the overwhelming majority of ops (the ternary cases —
`Select`, `MulAdd` — are the widest). `ENode::children()` (`node.rs:79–84`) returns an
owned clone of that `Vec` rather than a slice, and is called from inside per-rule `apply`
implementations that run once per (rule × node) pair every round — `algebra.rs:130,136`,
`candidate.rs:290`, `deps.rs:261`, `labeler.rs:346`, `template.rs:116`. `binary_operands()`
(`node.rs:87–92`) already does this correctly (borrows, no allocation) but isn't used at
every call site that could use it. Given §4.2's rescan volume, this clone runs at the same
frequency as the class-vector clone it sits inside.

**Fix direction:** add a `children_slice(&self) -> &[EClassId]` and switch call sites to
it (removes the clone without touching arity); separately, `Vec<EClassId>` →
`SmallVec<[EClassId; 3]>` (or `arrayvec`) for `ENode::Op.children` itself would remove the
per-node heap allocation for the common arities, at the cost of a new dependency —
`smallvec`/`arrayvec` support `alloc`-only (no separate `std` requirement), consistent
with this module's `extern crate alloc` usage (`lib.rs:4`, `node.rs`'s `use alloc::vec::
Vec`).

### 4.4 MEDIUM — hash-cons memo table (and every other `HashMap` in the module) uses SipHash; no fast hasher in the workspace

`EGraph::memo: HashMap<ENode, EClassId>` (`graph.rs:59`) is std's default `HashMap`
(SipHash), looked up on every `add()` — i.e., every node creation, including every node a
rewrite produces. `match_counts: HashMap<RuleId, usize>` (`graph.rs:69`) is updated on
every match. A workspace-wide grep for `fxhash`/`ahash`/`rustc-hash` returns nothing — no
crate in the tree uses a faster hasher anywhere. `ENode::hash`/`DefaultHasher::write`
combined are ~2% of the §3 flamegraph — real, but smaller than §4.1–4.3; not adversarial
input, so SipHash's DoS resistance buys nothing here.

**Fix direction:** swap `memo` (and, for consistency, `match_counts`) to a fast-hash
`HashMap` (e.g. `rustc_hash::FxHashMap`, already a transitive dependency of other tools in
this workspace via `hashbrown`/`indexmap`'s ecosystem, so the marginal dependency cost is
small). Lowest-risk item in this document — no correctness surface, purely a hasher swap.

### 4.5 MEDIUM — `BTreeMap`/`BTreeSet` used for dense small-integer keys in several hot paths

Two categories, both keying on IDs that are, for the dominant `Ir` impl (`ExprArena`),
plain `u32` (`ExprId(pub u32)`, `pixelflow-ir/src/arena.rs:47`):

- **Insertion**: `insert::insert` (`insert.rs:65`) uses `BTreeMap<I::Ref, EClassId>` as
  its own node-level memo (on top of, and separate from, `EGraph`'s hash-cons `memo`), and
  `insert::reachable_count` (`insert.rs:130–149`) uses `BTreeSet<I::Ref>` for the same
  reachability walk. Both run once per node inserted, on every kernel built.
- **Extraction**: `get_active_classes` (`extract.rs:401–404`), `choices_have_cycle_through`
  (`extract.rs:632`), `backfill_well_founded`'s scan (`extract.rs:859–871`), `post_order`
  (`extract.rs:2107–2111`), `toposort_dag` (`extract.rs:2189–2192`) all use `BTreeSet<u32>`
  for visited-tracking over the same dense, bounded `0..num_classes` space that
  `shared_dag_dp_pass` (§4.1) and `cost_of_choices` (`extract.rs:1624`, `color: Vec<u8>`)
  already handle with a plain `Vec`/bitset elsewhere in the *same file* — the fast pattern
  exists next to the slow one.

§3's flamegraph shows `BTreeMap::insert` alone at 2.8% self-time, with `post_order`,
`assemble`, `insert::insert`, and `insert::reachable_count` as the nearest
`pixelflow_search` frames above the btree calls — a combined ~5% of all cycles in a
pattern that has a strictly cheaper existing alternative in this codebase.

**Fix direction:** where the key type is (or can be shown to be) a dense small integer —
true for `ExprId` today — replace `BTreeMap<K, V>`/`BTreeSet<K>` with `Vec<Option<V>>`/a
bitset indexed by `key as usize`. `insert()` and `reachable_count()` are generic over `I:
Ir`, so this needs either a specialized fast path for index-shaped `I::Ref` or an
associated "dense index" bound on the trait; the five `extract.rs` sites are already
concretely keyed on `u32`/`EClassId` and can switch directly, matching `cost_of_choices`'s
existing `Vec<u8>` pattern.

### 4.6 LOW — saturation's worklist is not deduplicated

`union()` (`graph.rs:676`) unconditionally `self.worklist.push(parent)` with no
"already queued" check. A class receiving several unions in one round is queued once per
union, so `rebuild_budgeted` reprocesses that class's full node vector (canonicalize +
memo probe per node) once per redundant entry, even though only the result after the
*last* union in the round matters. Cheap fix (a "queued" bitset guarding the push), and
the batching design around it (rebuild runs once per `EGraphBatch::drop`, not per union —
see §5) is otherwise sound, so this is strictly a redundant-entry problem, not a
redundant-rebuild-pass one.

### 4.7 LOW — no capacity reservation on graph-growth vectors

No `with_capacity`/`reserve` calls appear in `graph.rs`/`insert.rs` for `classes`,
`parent`, `const_fact`, or `memo`, even though `insert()`'s caller can know the term's
approximate size up front via `reachable_count()` (`insert.rs:130–149`) before ever
calling `insert()`. Every one of these grows by std's doubling strategy — amortized O(1),
but every growth of `memo` rehashes an `ENode`-keyed table (walking each entry's
`children`), and every growth of `classes`/`parent` moves already-allocated `EClass`/
`Vec<EClassId>` data. Pre-sizing from `reachable_count()` (already computed, per §4.5, at
non-trivial cost of its own) is a small, low-risk change once that call is cheap.

## 5. What's fine — don't spend effort here

- **Union-find**: `find_mut` (`graph.rs:531–542`) does real path compression; parent
  selection is deterministic id-order (`graph.rs:665`) rather than union-by-rank, but
  combined with path compression this is not a measured hot spot in §3 and isn't worth
  reprioritizing without evidence it is one.
- **Rebuild batching**: `union()` only queues (`graph.rs:676`); the actual O(class)
  congruence-closure work happens in `rebuild_budgeted`, invoked once per batch/explicit
  rebuild call, not per union. This is the correct design (§4.6's worklist-dedup gap is a
  redundant-*entry* problem inside an otherwise-correctly-batched scheme, not a batching
  failure).
- **`cost.rs`**: `CostModel::cost`/`node_op_cost` are O(1) array lookups into an
  `OpMap<usize>` indexed by `OpKind::index()` — no allocation, nothing to memoize.
- **`rule_order.rs`**: rules are matched by straight sequential `Vec` order, confirming
  there's no missing discrimination-tree index to add on top of §4.2's fix — the dirty-
  tracking scheme there is the right lever, not an indexing structure.
- **`Reranker`/swap-refinement search** (`extract.rs:240–435`, the `IncrementalExtractor`):
  each candidate swap re-materializes the whole arena inside `score()`
  (`extract.rs:392–395`) and would be genuinely expensive if exercised — but per
  `optimizer.rs:656–666` and this codebase's own note that no `Reranker` implementation
  ships, this path is dormant in production today. Worth remembering before it's wired up
  (the schedule-cost work in the 2026-09-06 doc's §5.1 is exactly a candidate to wire it
  in), not worth optimizing now.

## 6. Suggested order of attack

By measured share of the §3 flamegraph, cross-checked against §2's scaling table:

1. **§4.1 (quadratic extraction)** and **§4.2 (rescan-without-dirty-tracking)** together
   account for the majority of both the 33% allocator time and the 32.8%
   `extract_dag_scoped` self-time. They are independent of each other and can be worked in
   either order or in parallel; §4.1 is the more contained change (one function, one
   threshold decision), §4.2 is the larger one (changes what "one saturation round" does).
2. **§4.3–4.5** (small-vec children, fast hasher, `BTreeMap`→`Vec`/bitset for dense keys)
   are each low-risk, independently shippable, and collectively plausible for another
   several points of the allocator/clone/btree lines in §3 — good first PRs for anyone
   picking this up, and safe to land ahead of §4.1/4.2 since they don't touch saturation
   or extraction semantics.
3. **§4.6–4.7** are small enough to fold into whichever of the above touches the same file.

Whatever ships: re-run `egraph_profile` (§1) before and after, on the same node-count
sweep, and diff §2's table — a change here that doesn't move `bytes/call`, `peak bytes`,
or the flamegraph's allocator share hasn't paid for itself. `--features
pixelflow-search/saturation-telemetry` on `bench_scene_chrome`
(`docs/plans/2026-09-06-egraph-at-production-scale.md` §6) is the production-scale check:
compile time and `stop_reason` there must not regress, and the extracted code must stay
byte-identical unless the change is meant to alter extraction (§4.1's threshold choice is
the one item here that can — see its no-regression note).

## 7. Pointers

`pixelflow-search/src/egraph/{graph,extract,node,insert,cost,rewrite,rule_order,
saturate}.rs`; `pixelflow-search/src/runtime.rs` (`optimize_runtime_arena`); the new
`pixelflow-pipeline/src/bin/egraph_profile.rs` and its `--features profiling` pprof path
(same pattern as `bin/bench_jit_profile.rs`); `pixelflow-search/src/telemetry.rs`;
`docs/plans/2026-09-06-egraph-at-production-scale.md` (the extraction-quality half of this
question); `docs/plans/2026-09-01-production-budget-determinism.md` (why saturation
budgets are applications/classes/iterations, never wall clock — unaffected by anything
here, since every fix above is a constant-factor or complexity change to work already
counted by those budgets, not a new budget dimension).
