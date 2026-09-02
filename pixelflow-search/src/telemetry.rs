//! Feature-flagged saturation telemetry: one JSONL record per production
//! e-graph optimizer invocation, gated entirely behind the
//! `saturation-telemetry` cargo feature (std-only, default OFF).
//!
//! # Why a side channel, not a return value
//!
//! `EGraph::optimize_runtime_arena` (this crate's [`crate::runtime`]) and
//! `pixelflow_compiler::optimize::optimize` keep their existing signatures
//! and return types unchanged — telemetry is never threaded through what
//! either function hands back. Each production call site calls
//! [`record`] itself, immediately after its own
//! [`crate::egraph::saturate_with_full_budget`] + extract, only when this
//! feature is compiled in. With the feature off,
//! this module does not exist (see the `#[cfg]` on its declaration in
//! `lib.rs`) and there is nothing left in the binary to call.
//!
//! # Sink
//!
//! One JSON object per line, appended to the path named by the
//! `PIXELFLOW_SATURATION_TELEMETRY` environment variable if set, otherwise
//! written to stderr. Opening or writing the sink is a hard failure
//! (`panic!`) — a telemetry record that silently fails to land is
//! indistinguishable from "nothing happened", which is exactly the failure
//! mode this instrument exists to catch (see this workspace's no-silent-
//! failures rule).
//!
//! The macro tier's stderr fallback is a special case: it runs inside
//! rustc's own process at macro-expansion time, so that stderr IS rustc's
//! diagnostic stream, and a bare JSON object dropped onto it is valid
//! enough to be mistaken for a real compiler message under `cargo ...
//! --message-format=json` (see `write_line`'s doc comment). It is prefixed
//! with plain text there — never emitted bare — so it fails JSON parsing on
//! that path and cargo passes it through as an ordinary line instead of a
//! forwarded diagnostic. The file sink is unaffected either way: a file a
//! caller opted into via the env var is never read by cargo's own
//! diagnostic parser, so it stays pure JSONL for both tiers.
//!
//! # Usage
//!
//! ```text
//! cargo run -p core-term --features saturation-telemetry
//! PIXELFLOW_SATURATION_TELEMETRY=/tmp/sat.jsonl cargo run -p core-term --features saturation-telemetry
//! ```

use std::io::Write as _;
use std::time::Duration;

use crate::egraph::{ClassCeiling, CostModel, OptimizerStats, SaturationStop};
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};

/// Which tier invoked saturation.
///
/// A closed set of two, so [`record`] can serialize it as a JSON string
/// literal directly — unlike `kernel_label`, there is no free-text value
/// here for `escape_json` to have to defend against. Extending "which tier"
/// to a type (rather than leaving it as a caller-supplied `&'static str`,
/// which a future call site could pass anything through) is what makes a
/// third, malformed tier unrepresentable instead of merely undocumented.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// [`crate::runtime::optimize_runtime_arena`].
    Runtime,
    /// `pixelflow_compiler::optimize`, running inside rustc at macro
    /// expansion time.
    Macro,
}

impl Tier {
    fn as_json_str(self) -> &'static str {
        match self {
            Tier::Runtime => "runtime",
            Tier::Macro => "macro",
        }
    }
}

/// Everything one production optimizer invocation — one
/// [`crate::egraph::saturate_with_full_budget`] call plus the extraction that
/// followed it — knows about itself, for [`record`] to serialize.
pub struct SaturationInvocation<'a> {
    /// Which tier invoked saturation.
    pub tier: Tier,
    /// Size of the input, as passed to `config_for_node_count` to select the
    /// budget triple.
    pub node_count: usize,
    /// What [`crate::egraph::Optimizer::run`] reported: the deterministic
    /// limits the run was held to, the rounds and applications it used, the
    /// e-class count it stopped at, and — the field this feature exists to
    /// surface — why it stopped.
    ///
    /// There is no `hard_timeout` here any more. The budget carries no
    /// clock, which is what makes a record comparable to one taken on
    /// another machine; a wall-clock ceiling exists on the optimizer but
    /// panics rather than truncating, so it can never be a stop reason.
    pub stats: &'a OptimizerStats,
    /// Class merges journalled during the run
    /// (`Provenance::union_count`) — never inferred from the stats above.
    pub union_count: usize,
    /// The arena and root this invocation extracted, so [`record`] can cost
    /// it under the static latency-prior model independently of whatever
    /// extraction policy actually chose it.
    pub extracted_arena: &'a ExprArena,
    pub extracted_root: ExprId,
    /// Wall-clock of saturate+extract together. Indicative only — see
    /// `CLAUDE.md`'s floating-point-at-the-edges notes on why timing is a
    /// measurement, not a promised bound.
    pub wall_clock: Duration,
    /// A label for the kernel being optimized, when the call site has one
    /// (e.g. a named `kernel!`'s struct name, or a source span). Never
    /// invented: `None` when the call site genuinely has nothing to name
    /// (an anonymous kernel, or a runtime-composed `Kernel` with no source
    /// identity).
    pub kernel_label: Option<&'a str>,
}

/// Emit one JSONL telemetry record for `inv`. See the module docs for the
/// sink and its failure behavior.
pub fn record(inv: SaturationInvocation<'_>) {
    let cost = latency_prior_cost(inv.extracted_arena, inv.extracted_root);
    let line = format!(
        "{{\"tier\":\"{tier}\",\"node_count\":{node_count},\"max_iterations\":{max_iterations},\
         \"max_classes\":{max_classes},\"max_applications\":{max_applications},\
         \"stop_reason\":\"{stop_reason}\",\"iterations\":{iterations},\
         \"classes_at_stop\":{classes_at_stop},\"application_count\":{application_count},\
         \"union_count\":{union_count},\"extracted_latency_prior_cost\":{cost},\
         \"wall_clock_us\":{wall_clock_us},\"kernel_label\":{kernel_label}}}",
        tier = inv.tier.as_json_str(),
        node_count = inv.node_count,
        max_iterations = inv.stats.limits.iterations,
        max_classes = inv.stats.limits.classes,
        max_applications = json_opt_u64(inv.stats.limits.applications),
        stop_reason = stop_str(inv.stats.stop),
        iterations = inv.stats.iterations,
        classes_at_stop = inv.stats.classes,
        application_count = inv.stats.applications,
        union_count = inv.union_count,
        cost = cost,
        wall_clock_us = inv.wall_clock.as_micros(),
        kernel_label = json_opt_str(inv.kernel_label),
    );
    write_line(inv.tier, &line);
}

fn stop_str(stop: SaturationStop) -> &'static str {
    match stop {
        SaturationStop::Quiesced => "quiesced",
        // Two ceilings, two strings: a reader must be able to tell "the
        // search budget was spent" from "the memory guard fired".
        SaturationStop::ClassCap(ClassCeiling::Live) => "class_cap_live",
        SaturationStop::ClassCap(ClassCeiling::Allocated) => "class_cap_allocated",
        SaturationStop::IterationCeiling => "iteration_ceiling",
        SaturationStop::Timeout => "timeout",
        SaturationStop::ApplicationBudget => "application_budget",
    }
}

/// `null` for an uncapped dimension, rather than a sentinel number a reader
/// would have to know to interpret.
fn json_opt_u64(v: Option<u64>) -> String {
    match v {
        Some(n) => format!("{n}"),
        None => String::from("null"),
    }
}

fn json_opt_str(s: Option<&str>) -> String {
    match s {
        Some(s) => format!("\"{}\"", escape_json(s)),
        None => "null".to_string(),
    }
}

/// Full JSON string escaping (RFC 8259 §7): every control character
/// (U+0000-U+001F) plus `"` and `\`. `kernel_label` is `pub`, so a caller
/// can hand `record` a struct name or span text containing a tab, carriage
/// return, or other control byte — those are JSON-forbidden unescaped in a
/// string literal and would silently corrupt the JSONL line (not just
/// visually mangle it) if left through, the same way an all-ones mask read
/// as a float silently corrupts a value elsewhere in this codebase. This is
/// not "telemetry labels happen to be identifiers today" defended
/// defensively; it is the complete contract for what `pub kernel_label:
/// Option<&str>` promises to accept.
fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            _ => out.push(c),
        }
    }
    out
}

/// Sum of `CostModel::latency_prior()` over every node reachable from
/// `root`. Computed independently here rather than threaded out of
/// extraction because neither `Extraction` nor `choices_to_arena`'s output
/// carries a total cost of its own (extraction tracks per-e-class best cost
/// internally, not on the materialized arena) — this mirrors
/// `crate::runtime`'s own `reachable_count` traversal shape.
fn latency_prior_cost(arena: &ExprArena, root: ExprId) -> usize {
    let costs = CostModel::latency_prior();
    let len = arena.nodes_raw().len();
    let mut seen = vec![false; len];
    let mut stack = vec![root];
    let mut total = 0usize;
    while let Some(id) = stack.pop() {
        if core::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        let op = match arena.node(id) {
            ExprNode::Unary(op, _)
            | ExprNode::Binary(op, _, _)
            | ExprNode::Ternary(op, _, _, _)
            | ExprNode::Nary(op, _, _) => Some(*op),
            ExprNode::Var(_) | ExprNode::Const(_) | ExprNode::Param(_) | ExprNode::Buffer(_) => {
                None
            }
        };
        if let Some(op) = op {
            total += costs.cost(op);
        }
        stack.extend(arena.children(id));
    }
    total
}

/// Append one line to the sink named by `PIXELFLOW_SATURATION_TELEMETRY`, or
/// stderr when unset — plain, for the runtime tier; prefixed with plain
/// text for the macro tier so cargo can't mistake it for a diagnostic (see
/// the `match tier` below). Open/write failures panic.
///
/// Emits the newline-terminated record with a single `write_all` rather than
/// `writeln!`, which is free to issue the line and the trailing newline as
/// two separate `write(2)` calls. Append mode (`O_APPEND`) only makes each
/// individual write atomic, not the pair — two parallel rustc processes (or
/// runtime threads) appending to the same path could otherwise interleave
/// their line and newline writes, corrupting the JSONL corpus with
/// concatenated objects or stray blank lines even though every process
/// respected append mode.
fn write_line(tier: Tier, line: &str) {
    let mut record = String::with_capacity(line.len() + 1);
    record.push_str(line);
    record.push('\n');

    match std::env::var_os("PIXELFLOW_SATURATION_TELEMETRY") {
        Some(path) => {
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .unwrap_or_else(|e| {
                    panic!("saturation-telemetry: failed to open {path:?} for append: {e}")
                });
            file.write_all(record.as_bytes()).unwrap_or_else(|e| {
                panic!("saturation-telemetry: failed to write to {path:?}: {e}")
            });
        }
        None => {
            let stderr = std::io::stderr();
            let mut lock = stderr.lock();
            // `Tier::Macro` runs inside rustc's own process at
            // macro-expansion time: this stderr IS rustc's stderr, not some
            // ordinary binary's. Under `cargo ... --message-format=json`,
            // cargo parses each line of rustc's stderr and forwards
            // whatever parses as JSON as a `"reason":"compiler-message"`
            // event — our record is itself valid JSON (just not
            // diagnostic-shaped), so a bare line here gets misread as a
            // genuine compiler message and forwarded downstream, corrupting
            // the stream for any JSON consumer. Confirmed empirically:
            // `cargo check --features saturation-telemetry
            // --message-format=json` produced exactly these bogus events.
            // A short plain-text prefix guarantees the line fails JSON
            // parsing, so cargo relays it as an ordinary (non-diagnostic)
            // line instead — still visible on stderr for a human, just not
            // parseable as one more compiler message. The runtime tier has
            // no such collision (its stderr is an ordinary process's own
            // stream, never read by cargo's diagnostic parser), so it keeps
            // emitting a bare, directly-JSONL-parseable line.
            let write_result = match tier {
                Tier::Macro => write!(lock, "saturation-telemetry(macro): {record}"),
                Tier::Runtime => lock.write_all(record.as_bytes()),
            };
            write_result
                .unwrap_or_else(|e| panic!("saturation-telemetry: failed to write stderr: {e}"));
        }
    }
}
