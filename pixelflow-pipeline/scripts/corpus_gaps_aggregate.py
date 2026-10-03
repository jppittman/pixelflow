#!/usr/bin/env python3
"""Aggregate corpus_gaps rows into side-by-side tables.

usage: aggregate.py rows.csv out_prefix
writes out_prefix.csv (rows + derived columns), out_prefix.json, out_prefix.tables.md
"""
import csv, json, sys, statistics as st
from collections import defaultdict, Counter

rows_path, out = sys.argv[1:3]

rows = list(csv.DictReader(open(rows_path)))
NUM = [k for k in rows[0].keys() if k not in ("name", "group", "population", "stop", "emit", "ops", "rule_hist")]
for r in rows:
    for k in NUM:
        try:
            r[k] = float(r[k])
        except ValueError:
            r[k] = float("nan")
    r["classes_at_cap"] = 1.0 if r["stop"] == "ClassCap" else 0.0
    r["quiesced"] = 1.0 if r["stop"] == "Quiesced" else 0.0
    r["has_select"] = 1.0 if r["selects"] > 0 else 0.0
    r["has_gather"] = 1.0 if r["gathers"] > 0 else 0.0
    r["has_transc"] = 1.0 if r["transcendentals"] > 0 else 0.0
    r["dup_gt_1"] = 1.0 if r["splice_factor"] > 1.001 else 0.0
    r["ext_shares"] = 1.0 if r["ext_sharing"] > 1.001 else 0.0
    r["hoist_frac"] = (r["frame_instr"] + r["row_instr"]) / max(1.0, r["frame_instr"] + r["row_instr"] + r["body_instr"])
    r["class_per_hc_node"] = r["classes"] / max(1.0, r["nodes_lowered_hc"])
    r["apps_per_hc_node"] = r["applications"] / max(1.0, r["nodes_lowered_hc"])
    r["ext_shrink"] = r["ext_nodes"] / max(1.0, r["nodes_lowered_hc"])
    r["lower_growth"] = r["nodes_lowered_hc"] / max(1.0, r["nodes_hashcons"])
    r["sat_noop"] = 1.0 if r["iterations"] == 0 else 0.0
    r["sel_per_100"] = 100.0 * r["selects"] / max(1.0, r["nodes_hashcons"])
    r["cmp_per_100"] = 100.0 * r["compares"] / max(1.0, r["nodes_hashcons"])
    r["bytes_per_hc_node"] = r["bytes"] / max(1.0, r["nodes_hashcons"])

EXTRA = ["classes_at_cap", "quiesced", "has_select", "has_gather", "has_transc",
         "dup_gt_1", "ext_shares", "hoist_frac", "class_per_hc_node", "apps_per_hc_node", "ext_shrink", "sel_per_100",
         "cmp_per_100", "bytes_per_hc_node"]

def q(vals, p):
    vals = sorted(v for v in vals if v == v)
    if not vals:
        return float("nan")
    i = round((len(vals) - 1) * p)
    return vals[i]

def summarize(sub, col):
    v = [r[col] for r in sub if r[col] == r[col]]
    if not v:
        return None
    return dict(n=len(v), p10=q(v, .1), med=q(v, .5), p90=q(v, .9), mean=sum(v) / len(v), max=max(v), min=min(v))

def fmt(x):
    if x is None or x != x:
        return "—"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    return f"{x:.3g}"

pops = {"real": [r for r in rows if r["population"] == "real"],
        "synthetic": [r for r in rows if r["population"] == "synthetic"]}
groups = defaultdict(list)
for r in rows:
    groups[r["group"]].append(r)

COLS = [
    ("nodes_reachable", "arena nodes at construction (reachable)"),
    ("nodes_hashcons", "nodes after hash-cons"),
    ("splice_factor", "splice-duplication factor (reachable / hash-cons)"),
    ("dup_gt_1", "fraction of kernels with any duplication"),
    ("nodes_lowered_hc", "e-graph input nodes (lowered, hash-consed)"),
    ("lower_growth", "lowering growth (Dwrt/reduce expansion)"),
    ("input_sharing", "input tree cost / dag cost (lowered)"),
    ("ext_sharing", "extracted tree cost / dag cost"),
    ("ext_shares", "fraction with extracted sharing > 1"),
    ("ext_nodes", "extracted DAG nodes"),
    ("ext_shrink", "extracted nodes / e-graph input nodes"),
    ("dag_cost_delta_pct", "extracted dag_cost vs input dag_cost (%)"),
    ("selects", "selects (hash-consed)"),
    ("sel_per_100", "selects per 100 nodes"),
    ("has_select", "fraction with any select"),
    ("compares", "compares"),
    ("compares_feeding_selects", "compares that are select masks"),
    ("select_masks_shared", "masks used by >1 select"),
    ("arm_true_med", "median true-arm reach (nodes)"),
    ("arm_false_med", "median false-arm reach (nodes)"),
    ("arm_excl_frac", "arm-exclusive nodes / arm reach"),
    ("ext_selects", "selects in extracted term"),
    ("ext_trip_sharing", "DP objective (trip-weighted) tree/dag"),
    ("gathers", "gathers"),
    ("has_gather", "fraction with any gather"),
    ("buffers", "buffers"),
    ("uniforms", "uniforms"),
    ("transcendentals", "transcendental nodes"),
    ("has_transc", "fraction with any transcendental"),
    ("depth", "depth"),
    ("hoist_frac", "schedule entries hoisted (frame+row) / all"),
    ("frame_frac", "frame-scope instructions / all"),
    ("row_frac", "row-scope instructions / all"),
    ("body_frac", "pixel-scope instructions / all"),
    ("hoisted", "values LICM hoisted"),
    ("spill_slots", "spill slots"),
    ("bytes", "emitted bytes"),
    ("dyn_memory_ops", "trip-weighted memory ops"),
    ("classes", "e-classes at stop"),
    ("class_per_hc_node", "classes per e-graph input node"),
    ("sat_noop", "fraction where saturation ran 0 iterations (input alone over the cap)"),
    ("classes_at_cap", "fraction stopping on ClassCap"),
    ("quiesced", "fraction quiescing"),
    ("applications", "rule applications"),
    ("apps_per_hc_node", "applications per e-graph input node"),
    ("iterations", "iterations"),
    ("opt_ms", "saturate+extract ms (host loaded; sign only)"),
]

md = []
md.append("### Side by side: REAL vs SYNTHETIC (median [p10, p90]; fractions are means)\n")
md.append("| property | real (n=%d) | synthetic (n=%d) |" % (len(pops["real"]), len(pops["synthetic"])))
md.append("|---|---:|---:|")
summary = {}
for col, label in COLS:
    cells = []
    for p in ("real", "synthetic"):
        s = summarize(pops[p], col)
        summary.setdefault(col, {})[p] = s
        if s is None:
            cells.append("—")
        elif col.startswith(("has_", "dup_", "ext_shares", "classes_at", "quiesced")):
            cells.append(f"{s['mean']:.0%}")
        else:
            cells.append(f"{fmt(s['med'])} [{fmt(s['p10'])}, {fmt(s['p90'])}]")
    md.append(f"| {label} | {cells[0]} | {cells[1]} |")

md.append("\n### By group (median)\n")
gcols = ["nodes_reachable", "nodes_hashcons", "splice_factor", "input_sharing", "ext_sharing", "selects", "compares", "gathers",
         "transcendentals", "hoist_frac", "body_frac", "classes", "applications",
         "classes_at_cap", "quiesced", "bytes", "ext_nodes"]
md.append("| group | n | " + " | ".join(gcols) + " |")
md.append("|---|---:|" + "---:|" * len(gcols))
bygroup = {}
for g in sorted(groups, key=lambda g: (groups[g][0]["population"], g)):
    sub = groups[g]
    cells = []
    bygroup[g] = {}
    for c in gcols:
        s = summarize(sub, c)
        bygroup[g][c] = s
        if s is None:
            cells.append("—")
        elif c in ("classes_at_cap", "quiesced"):
            cells.append(f"{s['mean']:.0%}")
        else:
            cells.append(fmt(s["med"]))
    md.append(f"| {g} ({sub[0]['population']}) | {len(sub)} | " + " | ".join(cells) + " |")

# stop reasons
md.append("\n### Stop reason under production budget\n")
md.append("| population | " + " | ".join(sorted({r['stop'] for r in rows})) + " |")
md.append("|---|" + "---:|" * len({r['stop'] for r in rows}))
stops = {}
for p, sub in pops.items():
    c = Counter(r["stop"] for r in sub)
    stops[p] = dict(c)
    md.append(f"| {p} | " + " | ".join(str(c.get(s, 0)) for s in sorted({r['stop'] for r in rows})) + " |")

# op histogram share
def op_share(sub):
    tot = Counter()
    for r in sub:
        for kv in r["ops"].split(";"):
            if not kv:
                continue
            k, v = kv.split(":")
            tot[k] += int(v)
    n = sum(tot.values()) or 1
    return {k: v / n for k, v in tot.items()}, n

ops = {p: op_share(sub) for p, sub in pops.items()}
allops = sorted(set(ops["real"][0]) | set(ops["synthetic"][0]), key=lambda k: -(ops["real"][0].get(k, 0) + ops["synthetic"][0].get(k, 0)))
md.append("\n### Op-kind share of hash-consed nodes (pooled)\n")
md.append("| op | real | synthetic | ratio real/synth |")
md.append("|---|---:|---:|---:|")
op_table = {}
for k in allops:
    a, b = ops["real"][0].get(k, 0), ops["synthetic"][0].get(k, 0)
    op_table[k] = dict(real=a, synthetic=b)
    ratio = "∞" if b == 0 and a > 0 else ("0" if a == 0 else f"{a / b:.2f}")
    md.append(f"| {k} | {a:.2%} | {b:.2%} | {ratio} |")

# rule fires
def rule_share(sub):
    tot = Counter()
    kernels_with = Counter()
    for r in sub:
        seen = set()
        for kv in r["rule_hist"].split(";"):
            if not kv:
                continue
            k, v = kv.rsplit(":", 1)
            tot[k] += int(v)
            seen.add(k)
        for k in seen:
            kernels_with[k] += 1
    n = sum(tot.values()) or 1
    return {k: v / n for k, v in tot.items()}, kernels_with, n

rules = {p: rule_share(sub) for p, sub in pops.items()}
allrules = sorted(set(rules["real"][0]) | set(rules["synthetic"][0]),
                  key=lambda k: -(rules["real"][0].get(k, 0) + rules["synthetic"][0].get(k, 0)))
md.append("\n### Rule-fire share under production saturation (pooled applications; kernels-with = fraction of kernels the rule fired on)\n")
md.append("| rule | real share | real kernels-with | synthetic share | synthetic kernels-with |")
md.append("|---|---:|---:|---:|---:|")
rule_table = {}
for k in allrules[:40]:
    a, b = rules["real"][0].get(k, 0), rules["synthetic"][0].get(k, 0)
    ka = rules["real"][1].get(k, 0) / max(1, len(pops["real"]))
    kb = rules["synthetic"][1].get(k, 0) / max(1, len(pops["synthetic"]))
    rule_table[k] = dict(real=a, synthetic=b, real_kernels=ka, synthetic_kernels=kb)
    md.append(f"| {k} | {a:.2%} | {ka:.0%} | {b:.2%} | {kb:.0%} |")
only_real = [k for k in allrules if rules["synthetic"][0].get(k, 0) == 0 and rules["real"][0].get(k, 0) > 0]
only_synth = [k for k in allrules if rules["real"][0].get(k, 0) == 0 and rules["synthetic"][0].get(k, 0) > 0]
md.append(f"\nRules firing only on real: {', '.join(only_real) or 'none'}.\n\nRules firing only on synthetic: {', '.join(only_synth) or 'none'}.\n")
md.append(f"\nTotal applications: real {rules['real'][2]:,}, synthetic {rules['synthetic'][2]:,}.\n")

# headline kernels
md.append("\n### The production scenes and grids, individually\n")
hcols = ["nodes_reachable", "nodes_hashcons", "nodes_lowered_hc", "splice_factor", "input_sharing", "ext_sharing", "ext_nodes", "dag_cost_delta_pct", "selects", "compares", "gathers",
         "hoist_frac", "classes", "class_cap", "stop", "applications", "iterations", "bytes", "spill_slots", "opt_ms"]
md.append("| kernel | " + " | ".join(hcols) + " |")
md.append("|---|" + "---:|" * len(hcols))
for r in rows:
    if r["group"] in ("scene", "cellgrid", "psychedelic") or r["name"] in ("shader:mandelbrot_distance", "shader:smooth_min_scene", "glyph32:U+0040", "glyph16:U+004A"):
        md.append(f"| {r['name']} | " + " | ".join(fmt(r[c]) if isinstance(r[c], float) else str(r[c]) for c in hcols) + " |")

open(out + ".tables.md", "w").write("\n".join(md) + "\n")
json.dump(dict(summary=summary, by_group=bygroup, stops=stops, op_share=op_table, rule_share=rule_table,
               only_real_rules=only_real, only_synthetic_rules=only_synth,
               n=dict(real=len(pops["real"]), synthetic=len(pops["synthetic"]))),
          open(out + ".json", "w"), indent=1, default=lambda x: None if x != x else x)
with open(out + ".csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
print("\n".join(md[:60]))
