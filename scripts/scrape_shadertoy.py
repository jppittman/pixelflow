#!/usr/bin/env python3
"""Scrape ShaderToy shaders and extract math expressions via Claude Sonnet.

Usage:
    SHADERTOY_KEY=xxx uv run scripts/scrape_shadertoy.py --count 500

Writes:
    pixelflow-pipeline/data/shadertoy_raw.jsonl   — raw GLSL + metadata
    pixelflow-pipeline/data/raw_shadertoy.jsonl   — extracted kernel code expressions
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic
import requests

SHADERTOY_API = "https://www.shadertoy.com/api/v1"

EXTRACTION_PROMPT = """\
Extract all scalar math subexpressions from this GLSL shader as PixelFlow kernel code.

Rules:
- Variables: X (fragCoord.x normalized 0-1), Y (fragCoord.y normalized 0-1), Z (iTime), W (iMouse.x normalized)
- Infix ops: +, -, *, /
- Unary prefix: - (negation)
- Unary methods: .sin(), .cos(), .tan(), .asin(), .acos(), .atan(), .exp(), .exp2(), .ln(), .log2(), .log10(), .sqrt(), .rsqrt(), .abs(), .recip(), .floor(), .ceil(), .round(), .fract()
- Binary methods: .pow(y), .min(y), .max(y), .atan2(y), .hypot(y)
- Ternary methods: .clamp(lo, hi), .mul_add(b, c)
- Constants: bare floats like 3.14159 or 0.5
- Wrap negative constants in parens: (-(1.5))
- Wrap all binary infix in parens: (X + Y), (X * 0.5)
- Decompose GLSL builtins:
  - mix(a,b,t) = (a) + ((b) - (a)) * (t)
  - mod(x,y) = (x) - (y) * ((x) / (y)).floor()
  - smoothstep(e0,e1,x) = let t = ((x) - (e0)) / ((e1) - (e0)); (t).clamp(0.0, 1.0) then t*t*(3.0 - 2.0*t)
  - length(v) for vec2 = ((vx * vx) + (vy * vy)).sqrt()
  - dot(a,b) for vec2 = (ax * bx) + (ay * by)
  - normalize(v).x for vec2 = vx * ((vx * vx) + (vy * vy)).rsqrt()
- Each expression must be SELF-CONTAINED: only X, Y, Z, W and float constants. No undefined variables.
- Extract interesting subexpressions: color channel computations, distance fields, noise functions, coordinate warps.
- Also extract the FULL expression for each color channel output (R, G, B) if possible.
- Skip anything with texture lookups, loops, or things you can't fully resolve.
- Minimum complexity: at least 3 operations per expression.

Output ONLY the expressions, one per line, nothing else. No comments, no labels, no blank lines.

GLSL shader:
```glsl
{glsl_source}
```"""


def fetch_shader_ids(api_key: str, count: int, sort: str = "popular") -> list[str]:
    """Fetch shader IDs from ShaderToy API."""
    ids = []
    batch_size = 25  # API max per request
    for offset in range(0, count, batch_size):
        n = min(batch_size, count - offset)
        url = f"{SHADERTOY_API}/shaders?key={api_key}&sort={sort}&num={n}&from={offset}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "Results" in data:
            ids.extend(data["Results"])
        else:
            print(f"[WARN] No Results at offset {offset}: {list(data.keys())}", file=sys.stderr)
            break
        time.sleep(0.5)  # Be polite
        print(f"[FETCH] Got {len(ids)}/{count} shader IDs", file=sys.stderr)
    return ids[:count]


def fetch_shader(api_key: str, shader_id: str) -> dict | None:
    """Fetch a single shader's metadata and source."""
    url = f"{SHADERTOY_API}/shaders/{shader_id}?key={api_key}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        shader = data.get("Shader", {})
        info = shader.get("info", {})
        renderpasses = shader.get("renderpass", [])

        # Get the main Image pass
        for rp in renderpasses:
            if rp.get("type") == "image":
                return {
                    "id": shader_id,
                    "name": info.get("name", shader_id),
                    "glsl": rp.get("code", ""),
                }

        # Fallback: first renderpass
        if renderpasses:
            return {
                "id": shader_id,
                "name": info.get("name", shader_id),
                "glsl": renderpasses[0].get("code", ""),
            }
    except Exception as e:
        print(f"[WARN] Failed to fetch {shader_id}: {e}", file=sys.stderr)
    return None


def extract_expressions(client: anthropic.Anthropic, glsl_source: str) -> list[str]:
    """Send GLSL to Sonnet, get back kernel code expressions."""
    prompt = EXTRACTION_PROMPT.replace("{glsl_source}", glsl_source)

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        # Split into lines, filter empty
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        # Basic sanity: must contain at least one of X/Y/Z/W or a float
        valid = []
        for line in lines:
            # Skip lines that look like commentary
            if line.startswith("//") or line.startswith("#") or line.startswith("*"):
                continue
            # Must have at least one variable or be a pure constant expression
            if any(v in line for v in ("X", "Y", "Z", "W")) or any(c.isdigit() for c in line):
                valid.append(line)
        return valid
    except Exception as e:
        print(f"[WARN] Sonnet extraction failed: {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(description="Scrape ShaderToy → kernel code expressions")
    parser.add_argument("--count", type=int, default=100, help="Number of shaders to fetch")
    parser.add_argument("--sort", default="popular", choices=["popular", "newest", "love", "hot"],
                        help="Sort order for shader listing")
    parser.add_argument("--output-dir", default="pixelflow-pipeline/data", help="Output directory")
    parser.add_argument("--skip-existing", action="store_true", help="Skip shaders already in raw file")
    args = parser.parse_args()

    shadertoy_key = os.environ.get("SHADERTOY_KEY")
    if not shadertoy_key:
        print("ERROR: Set SHADERTOY_KEY environment variable", file=sys.stderr)
        print("  Get a free key from https://www.shadertoy.com/myapps", file=sys.stderr)
        sys.exit(1)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_glsl_path = output_dir / "shadertoy_raw.jsonl"
    raw_exprs_path = output_dir / "raw_shadertoy.jsonl"

    # Load existing shader IDs to skip
    existing_ids = set()
    if args.skip_existing and raw_glsl_path.exists():
        with open(raw_glsl_path) as f:
            for line in f:
                d = json.loads(line)
                existing_ids.add(d["id"])
        print(f"[SKIP] {len(existing_ids)} shaders already scraped", file=sys.stderr)

    # Step 1: Fetch shader IDs
    print(f"[FETCH] Getting {args.count} shader IDs (sort={args.sort})...", file=sys.stderr)
    shader_ids = fetch_shader_ids(shadertoy_key, args.count, args.sort)
    shader_ids = [sid for sid in shader_ids if sid not in existing_ids]
    print(f"[FETCH] {len(shader_ids)} new shaders to process", file=sys.stderr)

    # Step 2: Fetch + extract
    client = anthropic.Anthropic(api_key=anthropic_key)
    total_exprs = 0

    with open(raw_glsl_path, "a") as glsl_f, open(raw_exprs_path, "a") as expr_f:
        for i, shader_id in enumerate(shader_ids):
            # Fetch GLSL
            shader = fetch_shader(shadertoy_key, shader_id)
            if not shader or not shader["glsl"].strip():
                print(f"[{i+1}/{len(shader_ids)}] {shader_id}: no source, skipping", file=sys.stderr)
                continue

            # Save raw GLSL
            glsl_f.write(json.dumps(shader) + "\n")
            glsl_f.flush()

            # Skip very short shaders (probably not interesting)
            if len(shader["glsl"]) < 100:
                print(f"[{i+1}/{len(shader_ids)}] {shader_id}: too short ({len(shader['glsl'])} chars), skipping", file=sys.stderr)
                continue

            # Extract expressions via Sonnet
            expressions = extract_expressions(client, shader["glsl"])

            for expr in expressions:
                record = {
                    "name": f"st_{shader_id}_{total_exprs}",
                    "expression": expr,
                    "source": shader_id,
                }
                expr_f.write(json.dumps(record) + "\n")
                total_exprs += 1

            expr_f.flush()

            print(
                f"[{i+1}/{len(shader_ids)}] {shader_id} ({shader['name'][:30]}): "
                f"{len(expressions)} expressions extracted (total: {total_exprs})",
                file=sys.stderr,
            )

            # Rate limit for ShaderToy API (Anthropic handles its own)
            time.sleep(0.5)

    print(f"\n[DONE] {total_exprs} expressions from {len(shader_ids)} shaders", file=sys.stderr)
    print(f"  Raw GLSL:       {raw_glsl_path}", file=sys.stderr)
    print(f"  Raw expressions: {raw_exprs_path}", file=sys.stderr)
    print(f"\n  Next: cargo run --release -p pixelflow-pipeline --features training --bin validate_corpus", file=sys.stderr)


if __name__ == "__main__":
    main()
