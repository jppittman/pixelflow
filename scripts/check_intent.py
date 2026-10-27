#!/usr/bin/env python3
"""Every module directory carries an intent.md, and every intent.md is well formed.

docs/process/2026-10-15-intent-lives-with-the-module.md §4 asked for this check:
a module's intent lives beside its `mod.rs` (or beside `Cargo.toml` for a
crate), under six fixed headings, and names only files the tree has. Presence
and shape are checked here; what the lines *claim* is review's business
(desloppify's module-shape rules read the file as context).

Scope: every workspace member in the root Cargo.toml, and every directory
under a member's `src/` that holds a `mod.rs`. `tests/` directories are not
modules with a contract and are not asked for one.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
HEADINGS = ["## Owns", "## Never", "## Contract", "## Enforced by", "## Behind", "## Decided in"]
PATH_TOKEN = re.compile(r"`([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:rs|sh|py|json|md|ya?ml|toml|txt))`")
MAX_LINES = 80


def members() -> list[Path]:
    text = (ROOT / "Cargo.toml").read_text()
    m = re.search(r"^members\s*=\s*\[(.*?)\]", text, re.S | re.M)
    if not m:
        sys.exit("FAIL: no [workspace] members in Cargo.toml")
    return [ROOT / name.strip().strip('"') for name in m.group(1).split(",") if name.strip()]


def module_dirs(crate: Path) -> list[Path]:
    src = crate / "src"
    if not src.is_dir():
        return []
    return sorted(p.parent for p in src.rglob("mod.rs") if "tests" not in p.parent.parts)


DEATH = re.compile(
    r"delet|remov|retir|\bgone\b|no longer|never exist|used to |replaced by|absorbed|renamed|"
    r"supersed|dropped|split into|moved to|collapsed into|folded into|is not in the tree|"
    r"does not exist|nothing declares|nonexistent|missing",
    re.I,
)
_NAMES: set[str] | None = None


def tracked_basenames() -> set[str]:
    """Every tracked file's basename. A doc cites `lower.rs` or
    `postsubmit-flake-detection.yaml` by name as often as by path, and a
    basename that exists somewhere in the tree is enough to show the file does."""
    global _NAMES
    if _NAMES is None:
        out = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
        _NAMES = {Path(f).name for f in out if f}
    return _NAMES


def resolves(token: str, intent_dir: Path) -> bool:
    if (ROOT / token).exists() or (intent_dir / token).exists():
        return True
    return Path(token).name in tracked_basenames()


def check(intent: Path, failures: list[str]) -> None:
    rel = intent.relative_to(ROOT)
    text = intent.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) > MAX_LINES:
        failures.append(f"{rel}: {len(lines)} lines; the template asks for a page (≤ {MAX_LINES})")
    if "**Is.**" not in text:
        failures.append(f"{rel}: no `**Is.**` sentence")
    positions = []
    for h in HEADINGS:
        found = [i for i, l in enumerate(lines) if l.strip() == h]
        if len(found) != 1:
            failures.append(f"{rel}: heading `{h}` appears {len(found)} times (want exactly 1)")
            continue
        positions.append(found[0])
    if positions != sorted(positions):
        failures.append(f"{rel}: headings out of template order")
    for n, line in enumerate(lines, 1):
        for token in PATH_TOKEN.findall(line):
            if resolves(token, intent.parent):
                continue
            # A Behind line may name a file that is gone or never written; the
            # sentence around it says so, as check-doc-paths.sh also accepts.
            window = " ".join(lines[max(0, n - 3) : n + 2])
            if DEATH.search(window):
                continue
            failures.append(f"{rel}:{n}: names `{token}`, which is not in the tree")


def main() -> int:
    failures: list[str] = []
    checked = 0
    for crate in members():
        for d in [crate, *module_dirs(crate)]:
            intent = d / "intent.md"
            if not intent.is_file():
                failures.append(f"{d.relative_to(ROOT)}: no intent.md (a module directory carries one)")
                continue
            checked += 1
            check(intent, failures)
    for f in failures:
        print(f"FAIL: {f}", file=sys.stderr)
    if failures:
        print(f"\n{len(failures)} problem(s) across {checked} intent files", file=sys.stderr)
        return 1
    print(f"OK: {checked} intent files present, shaped to the template, and every path they name resolves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
